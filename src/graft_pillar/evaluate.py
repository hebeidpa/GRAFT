from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

from .common import write_json


def task_metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(
            log_loss(y, np.column_stack([1.0 - probability, probability]), labels=[0, 1])
        ),
        "f1_at_0.5": float(f1_score(y, prediction, zero_division=0)),
        "sensitivity_at_0.5": float(tp / (tp + fn)) if tp + fn else None,
        "specificity_at_0.5": float(tn / (tn + fp)) if tn + fp else None,
        "confusion_at_0.5": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions on a labelled cohort")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="external_validation_metrics.json")
    args = parser.parse_args()
    cohort = pd.read_csv(args.cohort, encoding="utf-8-sig")
    predictions = pd.read_csv(args.predictions, encoding="utf-8-sig")
    required_cohort = ["case_id", "recurrence", "complication"]
    required_predictions = ["case_id", "prob_recurrence", "prob_complication"]
    for name, table, required in [
        ("cohort", cohort, required_cohort),
        ("predictions", predictions, required_predictions),
    ]:
        missing = [column for column in required if column not in table.columns]
        if missing:
            raise ValueError(f"{name} 缺少列: {missing}")
        if table["case_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} 中 case_id 重复")
    merged = cohort[required_cohort].merge(
        predictions[required_predictions], on="case_id", how="inner", validate="one_to_one"
    )
    if len(merged) != len(cohort) or len(merged) != len(predictions):
        raise ValueError(
            f"病例未完全匹配: cohort={len(cohort)}, predictions={len(predictions)}, matched={len(merged)}"
        )
    report = {"n_cases": int(len(merged)), "tasks": {}}
    for task in ["recurrence", "complication"]:
        y = pd.to_numeric(merged[task], errors="raise").to_numpy(dtype=int)
        probability = pd.to_numeric(
            merged[f"prob_{task}"], errors="raise"
        ).to_numpy(dtype=float)
        if set(np.unique(y)).difference({0, 1}):
            raise ValueError(f"{task} 标签必须是 0/1")
        if np.unique(y).size != 2:
            raise ValueError(f"{task} 必须同时包含阴性和阳性病例才能计算 ROC-AUC")
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError(f"prob_{task} 必须是 0 到 1 的有限数值")
        report["tasks"][task] = task_metrics(y, probability)
    output = Path(args.output)
    write_json(report, output)
    print(f"已保存: {output.resolve()}")


if __name__ == "__main__":
    main()
