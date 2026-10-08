from __future__ import annotations

import json
import math
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers import AutoModel

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def amp_dtype(name: str):
    name = name.lower()
    if name in {'bf16', 'bfloat16'}:
        return torch.bfloat16
    if name in {'fp16', 'float16', 'half'}:
        return torch.float16
    if name in {'fp32', 'float32'}:
        return torch.float32
    raise ValueError(f'Unsupported dtype: {name}')

def norm_name(x: str) -> str:
    return re.sub('[^a-z0-9]+', '', str(x).lower())

def resolve_col(df: pd.DataFrame, candidates: Sequence[str], what: str) -> str:
    lookup = {norm_name(c): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        nc = norm_name(c)
        if nc in lookup:
            return lookup[nc]
    raise KeyError(f'Cannot find {what}; tried {list(candidates)}')

class LoRALinear(nn.Module):
    """
    Frozen base Linear + trainable low-rank residual.

    y = Wx + alpha/r * B(A(x))

    B is initialized to zero -> exact pretrained behavior at initialization.
    """

    def __init__(self, base: nn.Linear, r=8, alpha=16.0, dropout=0.05):
        super().__init__()
        if r <= 0:
            raise ValueError('LoRA rank must be > 0')
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Linear(base.in_features, self.r, bias=False).to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B = nn.Linear(self.r, base.out_features, bias=False).to(device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))

@dataclass
class LoRAAudit:
    stage_index: int
    stage_depth: int
    selected_block_indices: List[int]
    scale_index: int
    targets: List[str]
    injected_modules: List[str]
    trainable_lora_params: int
    frozen_backbone_params: int

def get_visual(pillar):
    if not hasattr(pillar, 'model') or not hasattr(pillar.model, 'visual'):
        raise RuntimeError('Unexpected Pillar-0 hierarchy: expected pillar.model.visual')
    visual = pillar.model.visual
    if not hasattr(visual, 'atlas_models'):
        raise RuntimeError('Unexpected Pillar-0 hierarchy: visual.atlas_models missing')
    return visual

def inject_lora_last_stage(pillar, r=8, alpha=16.0, dropout=0.05, last_n_blocks=2, targets=('q', 'kv')):
    """
    Conservative adaptation:
    - only last Atlas stage
    - only last N MultiScaleAttentionBlocks
    - only active scale-specific block[0] in the last stage
    - q/kv only by default
    """
    for p in pillar.parameters():
        p.requires_grad_(False)
    visual = get_visual(pillar)
    stage_idx = len(visual.atlas_models) - 1
    stage = visual.atlas_models[stage_idx]
    depth = len(stage.blocks)
    if not 1 <= last_n_blocks <= depth:
        raise ValueError(f'last_n_blocks must be 1..{depth}')
    selected = list(range(depth - last_n_blocks, depth))
    injected = []
    scale_idx = 0
    for bi in selected:
        msa = stage.blocks[bi]
        cross = msa.blocks[scale_idx]
        attn = cross.attn
        for target in targets:
            base = getattr(attn, target, None)
            if not isinstance(base, nn.Linear):
                raise TypeError(f'Expected Linear at block={bi}, scale={scale_idx}, target={target}; got {type(base)}')
            setattr(attn, target, LoRALinear(base, r, alpha, dropout))
            injected.append(f'model.visual.atlas_models.{stage_idx}.blocks.{bi}.blocks.{scale_idx}.attn.{target}')
    trainable = sum((p.numel() for p in pillar.parameters() if p.requires_grad))
    frozen = sum((p.numel() for p in pillar.parameters() if not p.requires_grad))
    return LoRAAudit(stage_index=stage_idx, stage_depth=depth, selected_block_indices=selected, scale_index=scale_idx, targets=list(targets), injected_modules=injected, trainable_lora_params=int(trainable), frozen_backbone_params=int(frozen))

def set_lora_train(module, enabled=True):
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.train(enabled)

def set_lora_requires_grad(module, enabled=True):
    for m in module.modules():
        if isinstance(m, LoRALinear):
            for p in m.lora_A.parameters():
                p.requires_grad_(enabled)
            for p in m.lora_B.parameters():
                p.requires_grad_(enabled)

def lora_state(module):
    return {n: v.detach().cpu() for n, v in module.state_dict().items() if '.lora_A.' in n or '.lora_B.' in n}

@dataclass
class Cols:
    case_id: str
    ct: str
    l3: str
    tumor: str
    recurrence: str
    complication: str

def detect_cols(df):
    return Cols(resolve_col(df, ['case_id', 'patient_id', 'id'], 'case id'), resolve_col(df, ['ct_path', 'ct', 'image_path'], 'CT path'), resolve_col(df, ['l3_path', 'l3_mask', 'l3_mask_path'], 'L3 path'), resolve_col(df, ['tumor_path', 'tumor_mask', 'tumor_mask_path'], 'tumor path'), resolve_col(df, ['recurrence'], 'recurrence label'), resolve_col(df, ['complication'], 'complication label'))

def validate_df(df, c):
    df = df.copy()
    df[c.case_id] = df[c.case_id].astype(str)
    for col in [c.recurrence, c.complication]:
        df[col] = pd.to_numeric(df[col], errors='raise').astype(int)
        if not set(df[col].unique()).issubset({0, 1}):
            raise ValueError(f'{col} must be binary 0/1')
    if df[c.case_id].duplicated().any():
        raise ValueError('Duplicate case_id found')
    missing = []
    for _, r in df.iterrows():
        for col in [c.ct, c.l3, c.tumor]:
            p = Path(str(r[col])).expanduser()
            if not p.exists():
                missing.append((r[c.case_id], col, str(p)))
                if len(missing) >= 20:
                    break
        if len(missing) >= 20:
            break
    if missing:
        raise FileNotFoundError('Missing files (first up to 20):\n' + '\n'.join(map(str, missing)))
    return df.reset_index(drop=True)

CLINICAL_ALIASES = {'Age': ['Age', 'Age (year)', 'Age (years)', 'Age(year)', 'age_year', '年龄'], 'Sex': ['Sex', 'Gender', '性别'], 'ECOG': ['ECOG', 'ECOG PS', 'ECOG-PS'], 'Charlson': ['Charlson', 'Charlson score', 'Charlson index', 'CCI'], 'BMI': ['BMI'], 'PNI': ['PNI', 'PNI (cont)', 'PNI(cont)'], 'NLR': ['NLR', 'NLR (cont)', 'NLR(cont)'], 'PLR': ['PLR', 'PLR (cont)', 'PLR(cont)'], 'CONUT': ['CONUT', 'CONUT score', 'CONUT_score'], 'SII': ['SII', 'SII score', 'SII_score'], 'cT': ['cT', 'clinical T', 'clinical_T'], 'cN': ['cN', 'clinical N', 'clinical_N'], 'cTNM': ['cTNM', 'cTNM stage', 'clinical TNM', 'clinical_TNM'], 'Borrmann': ['Borrmann', 'Borrmann type'], 'Tumor_location': ['Tumor location', 'Tumor loc', 'Tumor_location', 'location'], 'CEA': ['CEA'], 'CA19_9': ['CA19-9', 'CA19_9', 'CA199'], 'CA72_4': ['CA72-4', 'CA72_4', 'CA724']}

def select_clinical_columns(df: pd.DataFrame, explicit: Optional[str]) -> List[str]:
    lookup = {norm_name(c): c for c in df.columns}
    if explicit:
        selected = []
        for raw in [x.strip() for x in explicit.split(',') if x.strip()]:
            if raw in df.columns:
                selected.append(raw)
            elif norm_name(raw) in lookup:
                selected.append(lookup[norm_name(raw)])
            else:
                raise KeyError(f'Clinical column not found: {raw}')
        return selected
    selected = []
    for canonical, aliases in CLINICAL_ALIASES.items():
        found = None
        for a in aliases:
            na = norm_name(a)
            if na in lookup:
                found = lookup[na]
                break
        if found is not None and found not in selected:
            selected.append(found)
    return selected

def infer_clinical_types(train_df: pd.DataFrame, clinical_cols: Sequence[str]) -> Tuple[List[str], List[str], Dict]:
    """
    Conservative fold-specific rule:
    - numeric if >=90% of non-missing entries can be parsed AND >8 unique values;
    - otherwise categorical.

    This makes binary/ordinal-coded Sex/cT/cN/Borrmann/tumor markers categorical
    when they have only a few levels, while continuous age/BMI/PNI/NLR/... remain numeric.
    """
    numeric, categorical = ([], [])
    audit = {}
    for col in clinical_cols:
        s = train_df[col]
        nonmissing = s.dropna()
        if len(nonmissing) == 0:
            categorical.append(col)
            audit[col] = {'type': 'categorical', 'reason': 'all_missing_train'}
            continue
        parsed = pd.to_numeric(nonmissing, errors='coerce')
        parse_rate = float(parsed.notna().mean())
        nunique = int(nonmissing.nunique(dropna=True))
        if parse_rate >= 0.9 and nunique > 8:
            numeric.append(col)
            kind = 'numeric'
        else:
            categorical.append(col)
            kind = 'categorical'
        audit[col] = {'type': kind, 'parse_rate': parse_rate, 'n_unique_train': nunique}
    return (numeric, categorical, audit)

def fit_clinical_preprocessor(train_df: pd.DataFrame, clinical_cols: Sequence[str]):
    num_cols, cat_cols, audit = infer_clinical_types(train_df, clinical_cols)
    transformers = []
    if num_cols:
        num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median', add_indicator=True)), ('scaler', StandardScaler())])
        transformers.append(('num', num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32))])
        transformers.append(('cat', cat_pipe, cat_cols))
    if not transformers:
        raise ValueError('No clinical variables available after selection.')
    prep = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0.0, verbose_feature_names_out=False)
    X = prep.fit_transform(train_df[list(clinical_cols)])
    X = np.asarray(X, dtype=np.float32)
    schema = {'selected_columns': list(clinical_cols), 'numeric_columns': num_cols, 'categorical_columns': cat_cols, 'type_audit': audit, 'output_dim': int(X.shape[1])}
    return (prep, schema)

def transform_clinical(prep, df, clinical_cols):
    X = prep.transform(df[list(clinical_cols)])
    return np.asarray(X, dtype=np.float32)

def make_splits(df, c, path, n_splits, seed):
    if path.exists():
        s = pd.read_csv(path, dtype={'case_id': str})
        if s['case_id'].tolist() == df[c.case_id].tolist() and s['fold'].nunique() == n_splits:
            print(f'[INFO] reuse fixed CV split: {path}')
            return s['fold'].to_numpy(int)
        raise RuntimeError(f'Existing {path} does not match current data/config. Delete it intentionally only if a new split is required.')
    y = df[[c.recurrence, c.complication]].to_numpy(int)
    strata = 2 * y[:, 0] + y[:, 1]
    counts = pd.Series(strata).value_counts()
    if counts.min() < n_splits:
        raise ValueError(f'Joint-label strata counts {counts.to_dict()} cannot support {n_splits} folds')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(df), -1, int)
    for f, (_, va) in enumerate(skf.split(np.zeros(len(df)), strata)):
        folds[va] = f
    pd.DataFrame({'case_id': df[c.case_id], 'recurrence': y[:, 0], 'complication': y[:, 1], 'joint_stratum': strata, 'fold': folds}).to_csv(path, index=False)
    return folds

def load_pillar(model_dir, device):
    m = AutoModel.from_pretrained(str(Path(model_dir).expanduser().resolve()), trust_remote_code=True, local_files_only=True)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval().to(device)

WINDOWS = ['lung', 'mediastinum', 'abdomen', 'liver', 'bone', 'brain', 'subdural', 'stroke', 'temporal_bone', 'soft_tissue', 'minmax']

def ct_tensor(row, c, prepare_hu_volume, make_windowed_tensor, device, dtype):
    hu = prepare_hu_volume(str(row[c.ct]), [1.25, 1.25, 1.25], [384, 384, 384], -1024)
    t = make_windowed_tensor(hu, WINDOWS, device=device, dtype=dtype)
    del hu
    return t

def pos_weights(y, device):
    out = []
    for j in range(2):
        pos = int(y[:, j].sum())
        neg = len(y) - pos
        if pos == 0:
            raise ValueError(f'No positive samples for task {j}')
        out.append(torch.tensor([neg / pos], device=device, dtype=torch.float32))
    return out

def loss_fn(logits, y, w):
    lr = F.binary_cross_entropy_with_logits(logits[:, 0], y[:, 0], pos_weight=w[0])
    lc = F.binary_cross_entropy_with_logits(logits[:, 1], y[:, 1], pos_weight=w[1])
    return 0.5 * (lr + lc)

def metrics(y, p):
    d = {}
    for j, name in enumerate(['recurrence', 'complication']):
        auc = float(roc_auc_score(y[:, j], p[:, j])) if len(np.unique(y[:, j])) > 1 else float('nan')
        ap = float(average_precision_score(y[:, j], p[:, j])) if len(np.unique(y[:, j])) > 1 else float('nan')
        br = float(brier_score_loss(y[:, j], p[:, j]))
        d[name] = {'auc': auc, 'pr_auc': ap, 'brier': br}
    d['mean_auc'] = float(np.nanmean([d['recurrence']['auc'], d['complication']['auc']]))
    return d


def import_project_ops(project_src=None):
    here = Path(__file__).resolve().parent
    candidates = []
    if project_src:
        candidates.append(Path(project_src).expanduser().resolve())
    candidates.extend([here / "src", here.parent / "src"])
    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        from graft_pillar.preprocess import prepare_hu_volume, make_windowed_tensor
    except Exception as exc:
        raise RuntimeError(
            "Cannot import graft_pillar preprocessing. Pass --project-src /path/to/src. "
            f"Original error: {exc}"
        ) from exc
    return prepare_hu_volume, make_windowed_tensor
