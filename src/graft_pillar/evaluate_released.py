"""Evaluate a released GRAFT checkpoint on one labelled cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch

from . import training_support as base
from .train_mask_guided import (
    evaluate_internal_validation,
    make_model,
    materialize_and_check,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-csv", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--project-src", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp-dtype", default="bfloat16")
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main():
    cli = parse_args()
    base.set_seed(cli.seed)
    checkpoint_path = Path(cli.checkpoint).expanduser().resolve()
    prep_path = checkpoint_path.with_suffix(".clinical_preprocessor.joblib")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not prep_path.is_file():
        raise FileNotFoundError(prep_path)

    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    runtime_values = dict(checkpoint["args"])
    runtime_values.update({
        "model_dir": cli.model_dir,
        "project_src": cli.project_src,
        "output_dir": cli.output_dir,
        "device": cli.device,
        "amp_dtype": cli.amp_dtype,
        "bootstrap_samples": cli.bootstrap_samples,
        "seed": cli.seed,
    })
    args = SimpleNamespace(**runtime_values)
    dtype = base.amp_dtype(cli.amp_dtype)
    ops = base.import_project_ops(cli.project_src)

    cohort = pd.read_csv(cli.cohort_csv, encoding="utf-8-sig")
    cols = base.detect_cols(cohort)
    cohort = base.validate_df(cohort, cols)
    clinical_cols = list(checkpoint["clinical_schema"]["selected_columns"])
    missing = [name for name in clinical_cols if name not in cohort.columns]
    if missing:
        raise KeyError("Cohort is missing clinical columns: " + ", ".join(missing))

    prep = joblib.load(prep_path)
    first_clinical = base.transform_clinical(prep, cohort.iloc[:1], clinical_cols)
    model, _ = make_model(
        args, int(checkpoint["clinical_schema"]["output_dim"]), device
    )
    materialize_and_check(
        model, cohort.iloc[0], cols, first_clinical, ops, device, dtype
    )
    model.load_state_dict(checkpoint["model_nonpillar_state"], strict=False)
    model.pillar.load_state_dict(checkpoint["lora_state"], strict=False)
    model.eval()

    labels = cohort[[cols.recurrence, cols.complication]].to_numpy(np.float32)
    predictions, metrics = evaluate_internal_validation(
        model, prep, cohort, cols, clinical_cols, labels,
        args, ops, device, dtype,
    )
    out = Path(cli.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "predictions.csv", index=False)
    metrics.update({
        "cases": int(len(cohort)),
        "checkpoint": checkpoint_path.name,
        "evaluation_role": "user-supplied labelled cohort",
    })
    base.save_json(metrics, out / "metrics.json")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("[RESULT]", out / "predictions.csv")


if __name__ == "__main__":
    main()
