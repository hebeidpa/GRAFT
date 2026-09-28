from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .manifest import case_id_from_name


# Public input schema. These names are intentionally fixed so that training,
# inference, and external validation use the same clinical definitions.
CLINICAL_NUMERICAL_COLUMNS = (
    "Age (years)",
    "Charlson index",
    "BMI",
    "PNI (cont.)",
    "NLR (cont.)",
    "PLR (cont.)",
    "CONUT score",
    "SII",
    "CEA",
    "CA19-9",
    "CA72-4",
)
CLINICAL_CATEGORICAL_COLUMNS = (
    "Sex",
    "ECOG",
    "cT",
    "cN",
    "cTNM stage",
    "Borrmann type",
    "Tumor location",
)


@dataclass(frozen=True)
class ClinicalSchema:
    file: Path
    id_column: str
    numerical_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]

    @property
    def columns(self) -> list[str]:
        return [*self.numerical_columns, *self.categorical_columns]


def schema_from_config(cfg: dict, file_override: str | Path | None = None) -> ClinicalSchema:
    clinical = cfg.get("clinical", {})
    configured_file = file_override or cfg.get("paths", {}).get("cohort_file")
    if not configured_file:
        raise ValueError("paths.cohort_file 未配置")
    return ClinicalSchema(
        file=Path(configured_file).expanduser(),
        id_column=str(cfg.get("data", {}).get("id_column", "case_id")),
        numerical_columns=CLINICAL_NUMERICAL_COLUMNS,
        categorical_columns=CLINICAL_CATEGORICAL_COLUMNS,
    )


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"临床信息文件不存在: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"临床信息文件只支持 CSV/XLSX/XLS: {path}")


def load_clinical_frame(
    schema: ClinicalSchema,
    case_ids: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    table = _read_table(schema.file)
    required = [schema.id_column, *schema.columns]
    missing_columns = [name for name in required if name not in table.columns]
    if missing_columns:
        raise ValueError(
            f"临床信息文件缺少列: {missing_columns}; 当前列为 {list(table.columns)}"
        )

    table = table[required].copy()
    if table[schema.id_column].isna().any():
        raise ValueError(f"临床病例标识列 {schema.id_column} 存在缺失值")
    table["case_id"] = table[schema.id_column].map(case_id_from_name)
    if (table["case_id"] == "").any():
        raise ValueError(f"临床病例标识列 {schema.id_column} 存在空 ID")
    if table["case_id"].duplicated().any():
        duplicated = table.loc[
            table["case_id"].duplicated(keep=False), "case_id"
        ].astype(str).tolist()
        raise ValueError(f"临床信息文件病例 ID 重复: {duplicated[:20]}")

    for name in schema.numerical_columns:
        original = table[name]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            examples = original.loc[invalid].astype(str).unique().tolist()[:10]
            raise ValueError(f"数值临床变量 {name} 含非数值内容: {examples}")
        table[name] = converted.astype(float)

    for name in schema.categorical_columns:
        table[name] = table[name].map(
            lambda value: np.nan
            if pd.isna(value) or not str(value).strip()
            else str(value).strip()
        )

    requested = [str(case_id) for case_id in case_ids]
    available = set(table["case_id"])
    absent = [case_id for case_id in requested if case_id not in available]
    if absent:
        raise ValueError(f"临床信息缺少 {len(absent)} 个影像病例: {absent[:20]}")

    extras = sorted(available - set(requested))
    warnings: list[str] = []
    if extras:
        warnings.append(f"临床信息有 {len(extras)} 例未用于本次建模: {extras[:20]}")

    aligned = table.set_index("case_id").loc[requested, schema.columns].copy()
    empty_columns = [name for name in schema.columns if aligned[name].isna().all()]
    if empty_columns:
        raise ValueError(f"以下临床变量整列为空，无法填补: {empty_columns}")
    return aligned.reset_index(drop=True), warnings


def make_clinical_preprocessor(schema: ClinicalSchema) -> ColumnTransformer:
    transformers = []
    if schema.numerical_columns:
        numerical = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", keep_empty_features=True),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numerical, list(schema.numerical_columns)))
    if schema.categorical_columns:
        categorical = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent", keep_empty_features=True
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False, dtype=np.float32
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical, list(schema.categorical_columns)))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def clinical_summary(frame: pd.DataFrame, schema: ClinicalSchema) -> dict:
    return {
        "file": str(schema.file.resolve()),
        "id_column": schema.id_column,
        "numerical_columns": list(schema.numerical_columns),
        "categorical_columns": list(schema.categorical_columns),
        "n_cases": int(len(frame)),
        "missing_by_column": {
            name: int(frame[name].isna().sum()) for name in schema.columns
        },
        "unique_by_categorical_column": {
            name: sorted(frame[name].dropna().astype(str).unique().tolist())
            for name in schema.categorical_columns
        },
    }
