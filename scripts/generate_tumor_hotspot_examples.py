#!/usr/bin/env python
"""Select de-identified cases with visible tumour-ROI Grad-CAM signal."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graft_pillar import training_support as base
from generate_gradcam_examples import (
    cam_plane,
    central_slice,
    draw_publication_individual,
    load_fold_model,
    render_case,
    save_single_roi_panels,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--splits-csv", required=True)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--project-src", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--exclude-cases-csv")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2029)
    p.add_argument("--n-examples", type=int, default=5)
    p.add_argument("--candidate-pool", type=int, default=50)
    p.add_argument("--feature-scale", type=int, default=1)
    p.add_argument("--min-score", type=float, default=0.05)
    return p.parse_args()


def tumour_hotspot_score(item):
    tumour = item["tumor"]
    ct = item["ct"]
    z = central_slice(tumour)
    roi = tumour[z] > 0
    if not roi.any():
        return 0.0
    plane = cam_plane(item["recurrence_cam"], z, ct[z].shape, ct.shape[0])
    values = plane[roi]
    q10, q50, q90, q98 = np.percentile(values, [10, 50, 90, 98])
    # Rewards both a strong peak and visible within-ROI heterogeneity.
    return float(q98 + 0.5 * (q90 - q10) + 0.25 * (q98 - q50))


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = base.amp_dtype("bfloat16")
    ops = base.import_project_ops(args.project_src)

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    cols = base.detect_cols(df)
    df = base.validate_df(df, cols)
    clinical_cols = base.select_clinical_columns(df, None)
    split = pd.read_csv(args.splits_csv, dtype={"case_id": str})
    split["case_id"] = split["case_id"].astype(str)
    df[cols.case_id] = df[cols.case_id].astype(str)
    merged = df.merge(split[["case_id", "fold"]], left_on=cols.case_id, right_on="case_id")

    if args.exclude_cases_csv:
        excluded = pd.read_csv(args.exclude_cases_csv, dtype={"case_id": str})
        merged = merged.loc[
            ~merged[cols.case_id].isin(excluded["case_id"].astype(str))
        ].reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    pool_n = min(int(args.candidate_pool), len(merged))
    candidates = merged.iloc[rng.choice(len(merged), size=pool_n, replace=False)].copy()
    candidates["sample_order"] = np.arange(pool_n)

    # Retain only the two strongest cases per fold to control host memory.
    retained = []
    for fold in sorted(candidates["fold"].unique()):
        fold_rows = candidates.loc[candidates["fold"] == fold].sort_values("sample_order")
        first = fold_rows.iloc[[0]]
        model, prep, model_args = load_fold_model(
            int(fold), first, cols, clinical_cols, args, ops, device, dtype
        )
        model_args.visual_feature_scale = int(args.feature_scale)
        fold_best = []
        for _, record in fold_rows.iterrows():
            row = record.to_frame().T
            item = render_case(
                model, prep, model_args, row, cols, clinical_cols, ops, device, dtype
            )
            score = tumour_hotspot_score(item)
            entry = (score, int(record["sample_order"]), str(record[cols.case_id]), int(fold), item)
            fold_best.append(entry)
            fold_best.sort(key=lambda x: x[0], reverse=True)
            while len(fold_best) > 2:
                dropped = fold_best.pop()
                del dropped
            gc.collect()
        retained.extend(fold_best)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    retained.sort(key=lambda x: (-x[0], x[1]))
    selected = [x for x in retained if x[0] >= args.min_score][: args.n_examples]
    if len(selected) < args.n_examples:
        selected = retained[: args.n_examples]
    if len(selected) < args.n_examples:
        raise RuntimeError(f"Only {len(selected)} eligible hotspot cases were found")

    for number, (_, _, _, _, item) in enumerate(selected, start=1):
        draw_publication_individual(item, out, number)
        save_single_roi_panels(item, out, number)

    pd.DataFrame(
        {
            "case_id": [x[2] for x in selected],
            "fold": [x[3] for x in selected],
            "tumour_hotspot_score": [x[0] for x in selected],
        }
    ).to_csv(out / "selected_cases.csv", index=False)
    summary = {
        "seed": args.seed,
        "candidate_pool": pool_n,
        "n_examples": len(selected),
        "selection": "Top tumour-ROI Grad-CAM scores from a random candidate pool",
        "representative_sample": False,
        "minimum_selected_score": min(x[0] for x in selected),
    }
    (out / "README.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out)


if __name__ == "__main__":
    main()
