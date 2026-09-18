from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .common import project_paths


def case_id_from_name(value: object) -> str:
    """Normalize a public cohort identifier without imposing a filename pattern."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _resolve_input_path(value: object, data_root: Path) -> str:
    if pd.isna(value) or not str(value).strip():
        return ""
    path = Path(str(value).strip()).expanduser()
    if not path.is_absolute():
        path = data_root / path
    return str(path.resolve())


def build_manifest(cfg: dict) -> Tuple[pd.DataFrame, list[str]]:
    """Build a manifest from one fixed-schema cohort CSV.

    File names and patient counts are unrestricted. Each row explicitly maps a
    case ID to its CT, L3 mask, tumour mask, labels, and clinical variables.
    """
    paths = project_paths(cfg)
    cohort_file = paths["cohort_file"]
    if not cohort_file.exists():
        raise FileNotFoundError(f"cohort CSV 不存在: {cohort_file}")
    cohort = pd.read_csv(cohort_file, encoding="utf-8-sig")
    data_cfg = cfg["data"]
    id_column = data_cfg.get("id_column", "case_id")
    ct_column = data_cfg.get("ct_path_column", "ct_path")
    l3_column = data_cfg.get("l3_path_column", "l3_mask_path")
    tumor_column = data_cfg.get("tumor_path_column", "tumor_mask_path")
    label_columns = list(data_cfg.get("label_columns", ["recurrence", "complication"]))
    required = [id_column, ct_column, l3_column, tumor_column, *label_columns]
    missing = [name for name in required if name not in cohort.columns]
    if missing:
        raise ValueError(f"cohort CSV 缺少基础列: {missing}; 当前列为 {list(cohort.columns)}")

    frame = cohort[required].copy()
    frame["case_id"] = frame[id_column].map(case_id_from_name)
    if (frame["case_id"] == "").any():
        raise ValueError(f"病例标识列 {id_column} 存在空值")
    if frame["case_id"].duplicated().any():
        duplicated = frame.loc[
            frame["case_id"].duplicated(keep=False), "case_id"
        ].tolist()
        raise ValueError(f"cohort CSV 病例 ID 重复: {duplicated[:20]}")

    for column in label_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"标签列 {column} 存在缺失或非数值内容")
        unique = set(values.astype(int).unique().tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(f"标签列 {column} 必须是 0/1，发现 {sorted(unique)}")
        frame[column] = values.astype(int)

    manifest = pd.DataFrame(
        {
            "case_id": frame["case_id"],
            "source_filename": frame[ct_column].astype(str),
            "ct_path": frame[ct_column].map(
                lambda value: _resolve_input_path(value, paths["data_root"])
            ),
            "l3_path": frame[l3_column].map(
                lambda value: _resolve_input_path(value, paths["data_root"])
            ),
            "tumor_path": frame[tumor_column].map(
                lambda value: _resolve_input_path(value, paths["data_root"])
            ),
            **{column: frame[column].astype(int) for column in label_columns},
        }
    )

    warnings: list[str] = []
    if len(label_columns) == 2:
        first, second = label_columns
        joint = manifest.groupby([first, second]).size().to_dict()
        warnings.append(f"联合标签分布 {first}/{second}: {joint}")
    return manifest, warnings
