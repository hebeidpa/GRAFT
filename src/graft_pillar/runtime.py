"""Inference-only runtime utilities for the released GRAFT model."""
from __future__ import annotations

import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel


WINDOWS = [
    "lung", "mediastinum", "abdomen", "liver", "bone", "brain",
    "subdural", "stroke", "temporal_bone", "soft_tissue", "minmax",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def amp_dtype(name: str) -> torch.dtype:
    value = name.lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _norm_name(value: str) -> str:
    return re.sub("[^a-z0-9]+", "", str(value).lower())


def _resolve_column(
    frame: pd.DataFrame, candidates: Sequence[str], description: str, required: bool = True
) -> str | None:
    lookup = {_norm_name(column): column for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if _norm_name(candidate) in lookup:
            return lookup[_norm_name(candidate)]
    if required:
        raise KeyError(f"Cannot find {description}; expected one of {list(candidates)}")
    return None


@dataclass(frozen=True)
class Columns:
    case_id: str
    ct: str
    l3: str
    tumor: str
    recurrence: str | None
    complication: str | None

    @property
    def has_labels(self) -> bool:
        return self.recurrence is not None and self.complication is not None


def detect_columns(frame: pd.DataFrame) -> Columns:
    return Columns(
        case_id=_resolve_column(frame, ["case_id"], "case ID"),
        ct=_resolve_column(frame, ["ct_path"], "CT path"),
        l3=_resolve_column(frame, ["l3_mask_path", "l3_path"], "L3-mask path"),
        tumor=_resolve_column(
            frame, ["tumor_mask_path", "tumor_path"], "tumour-mask path"
        ),
        recurrence=_resolve_column(
            frame, ["recurrence"], "recurrence label", required=False
        ),
        complication=_resolve_column(
            frame, ["complication"], "complication label", required=False
        ),
    )


def prepare_cohort(frame: pd.DataFrame, columns: Columns, csv_path: Path) -> pd.DataFrame:
    frame = frame.copy()
    if frame[columns.case_id].isna().any():
        raise ValueError("case_id contains missing values")
    frame[columns.case_id] = frame[columns.case_id].astype(str).str.strip()
    if (frame[columns.case_id] == "").any():
        raise ValueError("case_id contains blank values")
    if frame[columns.case_id].duplicated().any():
        raise ValueError("case_id values must be unique")

    for column in (columns.ct, columns.l3, columns.tumor):
        resolved = []
        for raw in frame[column]:
            path = Path(str(raw)).expanduser()
            if not path.is_absolute():
                path = csv_path.parent / path
            resolved.append(str(path.resolve()))
        frame[column] = resolved

    missing = []
    for _, row in frame.iterrows():
        for column in (columns.ct, columns.l3, columns.tumor):
            if not Path(row[column]).is_file():
                missing.append((row[columns.case_id], column, row[column]))
                if len(missing) >= 20:
                    break
        if len(missing) >= 20:
            break
    if missing:
        details = "\n".join(map(str, missing))
        raise FileNotFoundError(f"Missing input files (first 20):\n{details}")

    if (columns.recurrence is None) != (columns.complication is None):
        raise ValueError(
            "Provide both recurrence and complication labels for evaluation, or omit both for inference"
        )
    if columns.has_labels:
        for column in (columns.recurrence, columns.complication):
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
            if not set(frame[column].unique()).issubset({0, 1}):
                raise ValueError(f"{column} must contain only 0/1")
    return frame.reset_index(drop=True)


class LoRALinear(nn.Module):
    """Frozen linear layer plus the low-rank residual stored in the release."""

    def __init__(self, base: nn.Linear, rank=8, alpha=16.0, dropout=0.05):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Linear(base.in_features, rank, bias=False).to(
            device=base.weight.device, dtype=base.weight.dtype
        )
        self.lora_B = nn.Linear(rank, base.out_features, bias=False).to(
            device=base.weight.device, dtype=base.weight.dtype
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, inputs):
        residual = self.lora_B(self.lora_A(self.dropout(inputs)))
        return self.base(inputs) + self.scaling * residual


def _visual_backbone(pillar):
    if not hasattr(pillar, "model") or not hasattr(pillar.model, "visual"):
        raise RuntimeError("Unexpected Pillar-0 model hierarchy")
    visual = pillar.model.visual
    if not hasattr(visual, "atlas_models"):
        raise RuntimeError("Pillar-0 visual.atlas_models is missing")
    return visual


def install_lora_modules(
    pillar, rank=8, alpha=16.0, dropout=0.05, last_n_blocks=2, targets=("q", "kv")
) -> None:
    """Create the LoRA module structure required by the released state dict."""
    for parameter in pillar.parameters():
        parameter.requires_grad_(False)
    stage = _visual_backbone(pillar).atlas_models[-1]
    if not 1 <= last_n_blocks <= len(stage.blocks):
        raise ValueError("Invalid LoRA block count in released checkpoint")
    for block_index in range(len(stage.blocks) - last_n_blocks, len(stage.blocks)):
        attention = stage.blocks[block_index].blocks[0].attn
        for target in targets:
            layer = getattr(attention, target, None)
            if not isinstance(layer, nn.Linear):
                raise TypeError(f"Expected linear attention layer: block={block_index}, target={target}")
            setattr(attention, target, LoRALinear(layer, rank, alpha, dropout))


def load_pillar(model_dir: str | Path, device: torch.device):
    model_path = Path(model_dir).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Pillar0-AbdomenCT directory not found: {model_path}")
    model = AutoModel.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval().to(device)


def transform_clinical(preprocessor, frame: pd.DataFrame, columns: Sequence[str]):
    return np.asarray(preprocessor.transform(frame[list(columns)]), dtype=np.float32)


def ct_tensor(row, columns, prepare_hu_volume, make_windowed_tensor, device, dtype):
    hu = prepare_hu_volume(
        str(row[columns.ct]), [1.25, 1.25, 1.25], [384, 384, 384], -1024
    )
    tensor = make_windowed_tensor(hu, WINDOWS, device=device, dtype=dtype)
    del hu
    return tensor


def preprocessing_functions():
    from .preprocess import make_windowed_tensor, prepare_hu_volume

    return prepare_hu_volume, make_windowed_tensor
