#!/usr/bin/env python
"""Generate de-identified Grad-CAM examples from completed fold checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from graft_pillar import training_support as base
from graft_pillar.train_mask_guided import (
    case_tensors,
    make_model,
    materialize_and_check,
)


MODALITIES = ["CT", "Tumor", "L3", "Clinical"]
L3_REGIONS = [
    (3, "Skeletal muscle", "#4f9b3f"),
    (2, "Subcutaneous fat", "#e39a24"),
    (1, "Visceral fat", "#3f8eaa"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--splits-csv", required=True)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--project-src", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-examples", type=int, default=6)
    p.add_argument("--feature-scale", type=int, default=1)
    p.add_argument(
        "--exclude-cases-csv",
        help="Optional CSV whose case_id values are excluded before random sampling.",
    )
    p.add_argument(
        "--individual-only", action="store_true",
        help="Save one reference-style ROI figure per case and skip montage files.",
    )
    return p.parse_args()


def normalize_map(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)


def roi_restricted_cam(cam, roi):
    """Normalize CAM within an ROI and make all non-ROI pixels transparent."""
    cam = np.asarray(cam, dtype=np.float32)
    roi = np.asarray(roi, dtype=bool)
    scaled = np.zeros_like(cam, dtype=np.float32)
    values = cam[roi]
    if values.size:
        lo, hi = np.percentile(values, [2.0, 98.0])
        if hi > lo:
            scaled[roi] = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return np.ma.masked_where(~roi, scaled)


def central_slice(mask):
    points = np.argwhere(mask > 0)
    if len(points) == 0:
        return int(mask.shape[0] // 2)
    return int(np.median(points[:, 0]))


def gradcam_3d(logit, feature_map, retain_graph):
    grad = torch.autograd.grad(
        logit, feature_map, retain_graph=retain_graph, create_graph=False
    )[0]
    weights = grad.float().mean(dim=(2, 3, 4), keepdim=True)
    cam = torch.relu((weights * feature_map.float()).sum(dim=1, keepdim=True))
    cam = cam[0, 0]
    lo, hi = cam.amin(), cam.amax()
    cam = (cam - lo) / (hi - lo).clamp_min(1e-8)
    return cam.detach().cpu()


def cam_plane(cam, z, target_hw, target_depth):
    if target_depth <= 1:
        feature_z = 0
    else:
        feature_z = int(round(z * (cam.shape[0] - 1) / (target_depth - 1)))
    plane = cam[feature_z].unsqueeze(0).unsqueeze(0)
    plane = F.interpolate(
        plane, size=target_hw, mode="bilinear", align_corners=False
    )[0, 0]
    return normalize_map(plane.numpy())


def load_fold_model(fold, row, cols, clinical_cols, args, ops, device, dtype):
    fold_dir = Path(args.checkpoint_root) / f"fold_{fold}"
    checkpoint = torch.load(
        fold_dir / "best.pt", map_location=device, weights_only=False
    )
    saved_args = dict(checkpoint["args"])
    saved_args["model_dir"] = args.model_dir
    saved_args["project_src"] = args.project_src
    model_args = SimpleNamespace(**saved_args)
    prep = joblib.load(fold_dir / "best.clinical_preprocessor.joblib")
    clinical = base.transform_clinical(prep, row, clinical_cols)
    model, _ = make_model(model_args, clinical.shape[1], device)
    materialize_and_check(
        model, row.iloc[0], cols, clinical, ops, device, dtype
    )
    model.load_state_dict(checkpoint["model_nonpillar_state"], strict=False)
    model.pillar.load_state_dict(checkpoint["lora_state"], strict=False)
    model.eval()
    return model, prep, model_args


def render_case(model, prep, model_args, row, cols, clinical_cols, ops, device, dtype):
    clinical_np = base.transform_clinical(prep, row, clinical_cols)
    ct, tumor, l3 = case_tensors(
        row.iloc[0], cols, *ops, device, dtype
    )
    ct.requires_grad_(True)
    clinical = torch.from_numpy(clinical_np).to(
        device=device, dtype=torch.float32
    )
    enabled = device.type == "cuda" and dtype != torch.float32
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
        logits, gate_r, gate_c, image = model(
            ct, tumor, l3, clinical, return_gates=True
        )
    feature = image["feature_maps"][model_args.visual_feature_scale]
    recurrence_cam = gradcam_3d(logits[0, 0], feature, retain_graph=True)
    complication_cam = gradcam_3d(logits[0, 1], feature, retain_graph=False)
    probability = torch.sigmoid(logits.float()).detach().cpu().numpy()[0]
    result = {
        "ct": ct.detach().float().cpu().numpy()[0, 2],
        "tumor": tumor.detach().cpu().numpy()[0, 0],
        "l3": l3.detach().cpu().numpy()[0, 0],
        "recurrence_cam": recurrence_cam,
        "complication_cam": complication_cam,
        "probability": probability,
        "gate_r": gate_r.detach().float().cpu().numpy()[0],
        "gate_c": gate_c.detach().float().cpu().numpy()[0],
    }
    del ct, tumor, l3, clinical, logits, image, feature
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def draw_row(axes, item, example_number):
    ct, tumor, l3 = item["ct"], item["tumor"], item["l3"]
    tumor_z, l3_z = central_slice(tumor), central_slice(l3)
    tumor_ct, l3_ct = ct[tumor_z], ct[l3_z]
    recurrence_plane = cam_plane(
        item["recurrence_cam"], tumor_z, tumor_ct.shape, ct.shape[0]
    )
    complication_plane = cam_plane(
        item["complication_cam"], l3_z, l3_ct.shape, ct.shape[0]
    )

    axes[0].imshow(tumor_ct, cmap="gray", vmin=0, vmax=1)
    axes[0].contour(tumor[tumor_z] > 0, levels=[0.5], colors="#ff2d2d", linewidths=1.4)
    axes[0].set_title(f"Example {example_number}\nTumor level", fontsize=9, weight="bold")

    tumor_roi = tumor[tumor_z] > 0
    axes[1].imshow(tumor_ct, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(
        roi_restricted_cam(recurrence_plane, tumor_roi),
        cmap="turbo", alpha=0.72, vmin=0, vmax=1,
    )
    axes[1].contour(tumor_roi, levels=[0.5], colors="white", linewidths=1.0)
    axes[1].set_title(
        f"Recurrence Grad-CAM\nP={item['probability'][0]:.3f}",
        fontsize=9,
        weight="bold",
    )

    axes[2].imshow(l3_ct, cmap="gray", vmin=0, vmax=1)
    masked = np.ma.masked_where(l3[l3_z] == 0, l3[l3_z])
    axes[2].imshow(
        masked,
        cmap=ListedColormap(["#d95f5f", "#63a35c", "#4d78b5"]),
        alpha=0.48,
        vmin=1,
        vmax=3,
    )
    axes[2].set_title("L3 regions", fontsize=9, weight="bold")

    l3_roi = l3[l3_z] > 0
    axes[3].imshow(l3_ct, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(
        roi_restricted_cam(complication_plane, l3_roi),
        cmap="turbo", alpha=0.72, vmin=0, vmax=1,
    )
    axes[3].contour(l3_roi, levels=[0.5], colors="white", linewidths=0.8)
    axes[3].set_title(
        f"Complication Grad-CAM\nP={item['probability'][1]:.3f}",
        fontsize=9,
        weight="bold",
    )

    x = np.arange(4)
    width = 0.36
    axes[4].bar(x - width / 2, item["gate_r"], width, label="Recurrence", color="#b22234")
    axes[4].bar(x + width / 2, item["gate_c"], width, label="Complication", color="#3b7f69")
    axes[4].set_xticks(x, MODALITIES, rotation=35, ha="right", fontsize=7)
    axes[4].set_ylim(0, 1)
    axes[4].set_ylabel("Gate weight", fontsize=7)
    axes[4].set_title("Task-aware modality gates", fontsize=9, weight="bold")
    axes[4].legend(fontsize=6, frameon=False)

    for ax in axes[:4]:
        ax.axis("off")


def crop_bounds(mask, margin=12):
    points = np.argwhere(mask > 0)
    if len(points) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    y0, x0 = points.min(axis=0)
    y1, x1 = points.max(axis=0) + 1
    return (
        max(0, int(y0) - margin), min(mask.shape[0], int(y1) + margin),
        max(0, int(x0) - margin), min(mask.shape[1], int(x1) + margin),
    )


def draw_publication_card(subspec, item, example_number):
    """Draw a compact case card patterned after the manuscript reference figure."""
    grid = subspec.subgridspec(2, 2, wspace=0.04, hspace=0.18)
    axes = [plt.subplot(grid[i, j]) for i in range(2) for j in range(2)]
    ct, tumor, l3 = item["ct"], item["tumor"], item["l3"]
    tumor_z, l3_z = central_slice(tumor), central_slice(l3)
    tumor_ct, l3_ct = ct[tumor_z], ct[l3_z]
    recurrence_plane = cam_plane(
        item["recurrence_cam"], tumor_z, tumor_ct.shape, ct.shape[0]
    )
    complication_plane = cam_plane(
        item["complication_cam"], l3_z, l3_ct.shape, ct.shape[0]
    )

    tumor_mask = tumor[tumor_z] > 0
    y0, y1, x0, x1 = crop_bounds(tumor_mask, margin=18)
    axes[0].imshow(tumor_ct[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1)
    tumor_cam = roi_restricted_cam(recurrence_plane, tumor_mask)
    axes[0].imshow(
        tumor_cam[y0:y1, x0:x1], cmap="turbo", alpha=0.76,
        vmin=0, vmax=1,
    )
    axes[0].contour(
        tumor_mask[y0:y1, x0:x1], levels=[0.5], colors="#ff3030", linewidths=1.25
    )
    axes[0].set_title("Tumour ROI + Grad-CAM", fontsize=8.2, weight="bold")

    for ax, (label, name, color) in zip(axes[1:], L3_REGIONS):
        region = l3[l3_z] == label
        ax.imshow(l3_ct, cmap="gray", vmin=0, vmax=1)
        ax.imshow(
            roi_restricted_cam(complication_plane, region),
            cmap="turbo", alpha=0.76, vmin=0, vmax=1,
        )
        if region.any():
            ax.contour(region, levels=[0.5], colors=color, linewidths=0.8)
        ax.set_title(name, fontsize=8.2, weight="bold", color=color)

    recurrence = float(item["probability"][0])
    complication = float(item["probability"][1])
    if recurrence >= 2 / 3:
        risk, band = "High recurrence risk", "#efb6cf"
    elif recurrence >= 1 / 3:
        risk, band = "Intermediate recurrence risk", "#f3d58a"
    else:
        risk, band = "Low recurrence risk", "#c9e58a"
    axes[0].text(
        0.0, 1.23,
        f"Example {example_number} | {risk}\nP(R)={recurrence:.3f}   P(C)={complication:.3f}",
        transform=axes[0].transAxes, fontsize=8.0, weight="bold",
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=band, edgecolor="none"),
    )
    for ax in axes:
        ax.axis("off")


def draw_publication_montage(items, out):
    order = np.argsort([-float(x["probability"][0]) for x in items])
    ordered = [items[int(i)] for i in order]
    fig = plt.figure(figsize=(15, 10.2), facecolor="white")
    outer = fig.add_gridspec(2, 3, wspace=0.12, hspace=0.20)
    for index, item in enumerate(ordered):
        draw_publication_card(outer[index // 3, index % 3], item, index + 1)
    fig.suptitle(
        "GRAFT mask-guided multimodal explanations",
        fontsize=17, weight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.008,
        "ROI-restricted Grad-CAM: recurrence within tumour; complication within each L3 compartment. "
        "Examples are de-identified and ordered by predicted recurrence risk.",
        ha="center", fontsize=9,
    )
    fig.savefig(
        out / "random6_reference_style_montage.png", dpi=240,
        bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)


def draw_publication_individual(item, out, example_number):
    fig = plt.figure(figsize=(7.0, 6.7), facecolor="white")
    outer = fig.add_gridspec(1, 1)
    draw_publication_card(outer[0, 0], item, example_number)
    fig.text(
        0.5, 0.018,
        "ROI-restricted Grad-CAM: recurrence within tumour; complication within each L3 compartment.",
        ha="center", fontsize=8.5,
    )
    fig.savefig(
        out / f"roi_case_{example_number:02d}.png", dpi=240,
        bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)


def save_single_roi_panels(item, out, example_number):
    """Save tumour, muscle, subcutaneous-fat, and visceral-fat panels separately."""
    ct, tumor, l3 = item["ct"], item["tumor"], item["l3"]
    tumor_z, l3_z = central_slice(tumor), central_slice(l3)
    tumor_ct, l3_ct = ct[tumor_z], ct[l3_z]
    recurrence_plane = cam_plane(
        item["recurrence_cam"], tumor_z, tumor_ct.shape, ct.shape[0]
    )
    complication_plane = cam_plane(
        item["complication_cam"], l3_z, l3_ct.shape, ct.shape[0]
    )

    tumor_mask = tumor[tumor_z] > 0
    y0, y1, x0, x1 = crop_bounds(tumor_mask, margin=18)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), facecolor="white")
    ax.imshow(tumor_ct[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1)
    ax.imshow(
        roi_restricted_cam(recurrence_plane, tumor_mask)[y0:y1, x0:x1],
        cmap="turbo", alpha=0.76, vmin=0, vmax=1,
    )
    ax.contour(
        tumor_mask[y0:y1, x0:x1], levels=[0.5], colors="#ff3030", linewidths=1.4
    )
    ax.set_title(
        f"Example {example_number} | Tumour ROI\nRecurrence Grad-CAM, P={item['probability'][0]:.3f}",
        fontsize=11, weight="bold",
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(
        out / f"case_{example_number:02d}_tumour.png", dpi=260,
        bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)

    slug_by_label = {3: "skeletal_muscle", 2: "subcutaneous_fat", 1: "visceral_fat"}
    for label, name, color in L3_REGIONS:
        region = l3[l3_z] == label
        fig, ax = plt.subplots(figsize=(5.2, 5.2), facecolor="white")
        ax.imshow(l3_ct, cmap="gray", vmin=0, vmax=1)
        ax.imshow(
            roi_restricted_cam(complication_plane, region),
            cmap="turbo", alpha=0.76, vmin=0, vmax=1,
        )
        if region.any():
            ax.contour(region, levels=[0.5], colors=color, linewidths=1.0)
        ax.set_title(
            f"Example {example_number} | {name}\nComplication Grad-CAM, P={item['probability'][1]:.3f}",
            fontsize=11, weight="bold", color=color,
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(
            out / f"case_{example_number:02d}_{slug_by_label[label]}.png",
            dpi=260, bbox_inches="tight", facecolor="white",
        )
        plt.close(fig)


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
        if "case_id" not in excluded.columns:
            raise ValueError("--exclude-cases-csv must contain a case_id column")
        merged = merged.loc[
            ~merged[cols.case_id].isin(excluded["case_id"].astype(str))
        ].reset_index(drop=True)
    if args.n_examples > len(merged):
        raise ValueError(
            f"Requested {args.n_examples} examples but only {len(merged)} are available"
        )
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(merged), size=args.n_examples, replace=False)
    selected = merged.iloc[chosen].copy().reset_index(drop=True)
    selected[[cols.case_id, "fold"]].to_csv(out / "selected_cases.csv", index=False)

    items = [None] * len(selected)
    for fold in sorted(selected["fold"].unique()):
        local_indices = np.where(selected["fold"].to_numpy() == fold)[0]
        first = selected.iloc[[int(local_indices[0])]]
        model, prep, model_args = load_fold_model(
            int(fold), first, cols, clinical_cols, args, ops, device, dtype
        )
        model_args.visual_feature_scale = int(args.feature_scale)
        for index in local_indices:
            row = selected.iloc[[int(index)]]
            items[int(index)] = render_case(
                model, prep, model_args, row, cols, clinical_cols,
                ops, device, dtype,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    panel_out = out / "single_panels"
    if args.individual_only:
        panel_out.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(items):
        draw_publication_individual(item, out, index + 1)
        if args.individual_only:
            save_single_roi_panels(item, panel_out, index + 1)
    if not args.individual_only:
        fig, axes = plt.subplots(len(items), 5, figsize=(15, 3.15 * len(items)))
        if len(items) == 1:
            axes = axes[None, :]
        for index, item in enumerate(items):
            draw_row(axes[index], item, index + 1)
            one, one_axes = plt.subplots(1, 5, figsize=(15, 3.2))
            draw_row(one_axes, item, index + 1)
            one.tight_layout()
            one.savefig(out / f"example_{index+1:02d}.png", dpi=220, bbox_inches="tight")
            plt.close(one)
        fig.suptitle(
            "GRAFT fold-specific Grad-CAM examples",
            fontsize=15,
            weight="bold",
            y=0.998,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.992])
        fig.savefig(out / "random6_gradcam_montage.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        draw_publication_montage(items, out)
    summary = {
        "seed": args.seed,
        "n_examples": len(items),
        "feature_scale": args.feature_scale,
        "note": "Gate weights are attention weights, not SHAP values.",
    }
    (out / "README.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out)


if __name__ == "__main__":
    main()
