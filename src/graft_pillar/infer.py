"""Run frozen GRAFT inference, with optional evaluation when labels are present."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)

from . import runtime
from .mask_guided import FourModalTaskAwareFusion, PillarMaskGuidedEncoder


def aligned_mask_array(mask_path, ct_path, target_spacing, target_shape, label_values=None):
    """Map a sparse mask onto the exact grid used for the Pillar CT tensor."""
    ct = sitk.ReadImage(str(ct_path))
    mask = sitk.ReadImage(str(mask_path))
    old_spacing = ct.GetSpacing()
    old_size = ct.GetSize()
    new_size = [
        int(round(old_size[i] * old_spacing[i] / float(target_spacing[i])))
        for i in range(3)
    ]

    source = sitk.GetArrayFromImage(mask)
    keep = source != 0 if label_values is None else np.isin(source, np.asarray(label_values))
    source_zyx = np.argwhere(keep)
    if len(source_zyx) == 0:
        array = np.zeros(tuple(reversed(new_size)), dtype=np.int16)
    else:
        values = (
            np.ones(len(source_zyx), dtype=np.int16)
            if label_values is None
            else source[tuple(source_zyx.T)].astype(np.int16, copy=False)
        )
        source_xyz = source_zyx[:, ::-1].astype(np.float64)
        scaled = source_xyz * np.asarray(mask.GetSpacing(), dtype=np.float64)
        mask_direction = np.asarray(mask.GetDirection(), dtype=np.float64).reshape(3, 3)
        physical = scaled @ mask_direction.T + np.asarray(mask.GetOrigin(), dtype=np.float64)
        ct_direction = np.asarray(ct.GetDirection(), dtype=np.float64).reshape(3, 3)
        ct_scaled = (
            physical - np.asarray(ct.GetOrigin(), dtype=np.float64)
        ) @ np.linalg.inv(ct_direction).T
        output_xyz = np.floor(
            ct_scaled / np.asarray(target_spacing, dtype=np.float64) + 0.5
        ).astype(np.int64)
        valid = np.ones(len(output_xyz), dtype=bool)
        for axis in range(3):
            valid &= (output_xyz[:, axis] >= 0) & (output_xyz[:, axis] < new_size[axis])
        output_xyz, values = output_xyz[valid], values[valid]
        array = np.zeros(tuple(reversed(new_size)), dtype=np.int16)
        np.maximum.at(
            array, (output_xyz[:, 2], output_xyz[:, 1], output_xyz[:, 0]), values
        )

    target = tuple(int(value) for value in target_shape)
    padded = np.zeros(target, dtype=np.int16)
    source_slices, destination_slices = [], []
    for source_length, destination_length in zip(array.shape, target):
        copy_length = min(source_length, destination_length)
        source_start = max((source_length - copy_length) // 2, 0)
        destination_start = max((destination_length - copy_length) // 2, 0)
        source_slices.append(slice(source_start, source_start + copy_length))
        destination_slices.append(
            slice(destination_start, destination_start + copy_length)
        )
    padded[tuple(destination_slices)] = array[tuple(source_slices)].astype(np.int16)
    return padded


def case_tensors(row, columns, preprocessing, device, dtype):
    prepare_hu_volume, make_windowed_tensor = preprocessing
    ct = runtime.ct_tensor(
        row, columns, prepare_hu_volume, make_windowed_tensor, device, dtype
    )
    spacing, shape = (1.25, 1.25, 1.25), (384, 384, 384)
    tumor_array = aligned_mask_array(
        row[columns.tumor], row[columns.ct], spacing, shape, label_values=None
    )
    l3_array = aligned_mask_array(
        row[columns.l3], row[columns.ct], spacing, shape, label_values=(1, 2, 3)
    )
    tumor_array = (tumor_array > 0).astype(np.uint8)
    if int(tumor_array.sum()) == 0:
        raise ValueError(f"Empty tumour mask after preprocessing: {row[columns.case_id]}")
    missing_l3 = [label for label in (1, 2, 3) if not np.any(l3_array == label)]
    if missing_l3:
        raise ValueError(
            f"L3 labels {missing_l3} absent after preprocessing: {row[columns.case_id]}"
        )
    tumor = torch.from_numpy(tumor_array).unsqueeze(0).unsqueeze(0).to(device)
    l3 = torch.from_numpy(l3_array.astype(np.uint8)).unsqueeze(0).unsqueeze(0).to(device)
    return ct, tumor, l3


class GRAFTModel(nn.Module):
    def __init__(self, pillar, clinical_dim: int, args):
        super().__init__()
        self.pillar = pillar
        self.image_encoder = PillarMaskGuidedEncoder(
            pillar,
            emb_dim=args.emb_dim,
            stage_indices=tuple(args.stage_indices),
            dropout=args.encoder_dropout,
            l3_labels=(1, 2, 3),
        )
        self.fusion = FourModalTaskAwareFusion(
            emb_dim=args.emb_dim,
            clinical_dim=clinical_dim,
            hidden=args.head_hidden,
            dropout=args.head_dropout,
        )
        self.modalities = ("ct", "tumor", "l3", "clinical")

    def forward(self, ct, tumor, l3, clinical):
        image = self.image_encoder(ct, tumor, l3)
        result = self.fusion(
            image["ct_embedding"],
            image["tumor_embedding"],
            image["l3_embedding"],
            clinical,
        )
        logits = torch.stack(
            [result["recurrence_logit"], result["complication_logit"]], dim=1
        )
        return (
            logits,
            result["recurrence_modality_weights"],
            result["complication_modality_weights"],
        )


def build_model(args, clinical_dim: int, device: torch.device):
    pillar = runtime.load_pillar(args.model_dir, device)
    if args.lora_r > 0:
        targets = tuple(value.strip() for value in args.lora_targets.split(",") if value.strip())
        runtime.install_lora_modules(
            pillar,
            rank=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            last_n_blocks=args.lora_blocks,
            targets=targets,
        )
    return GRAFTModel(pillar, clinical_dim, args).to(device)


def materialize(model, row, columns, clinical_row, preprocessing, device, dtype):
    ct, tumor, l3 = case_tensors(row, columns, preprocessing, device, dtype)
    clinical = torch.from_numpy(clinical_row).to(device=device, dtype=torch.float32)
    autocast = device.type == "cuda" and dtype != torch.float32
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=dtype, enabled=autocast
    ):
        model(ct, tumor, l3, clinical)
    del ct, tumor, l3, clinical


def _task_metrics(y_true, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    result = {
        "n": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "sensitivity": float(recall_score(y_true, prediction, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, prediction, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probability))
        result["pr_auc"] = float(average_precision_score(y_true, probability))
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
    return result


def _bootstrap_auc(y_true, probability, samples, seed):
    if samples <= 0 or len(np.unique(y_true)) < 2:
        return None
    rng, values = np.random.default_rng(seed), []
    for _ in range(samples):
        indices = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) == 2:
            values.append(roc_auc_score(y_true[indices], probability[indices]))
    if not values:
        return None
    lower, upper = np.percentile(values, [2.5, 97.5])
    return [float(lower), float(upper)]


def run(args):
    runtime.set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = runtime.amp_dtype(args.amp_dtype)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    preprocessor_path = checkpoint_path.with_suffix(".clinical_preprocessor.joblib")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not preprocessor_path.is_file():
        raise FileNotFoundError(preprocessor_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    values = dict(checkpoint["args"])
    values.update({"model_dir": args.model_dir})
    model_args = SimpleNamespace(**values)
    clinical_columns = list(checkpoint["clinical_schema"]["selected_columns"])

    csv_path = Path(args.cohort_csv).expanduser().resolve()
    cohort = pd.read_csv(csv_path, encoding="utf-8-sig")
    columns = runtime.detect_columns(cohort)
    cohort = runtime.prepare_cohort(cohort, columns, csv_path)
    missing_clinical = [column for column in clinical_columns if column not in cohort]
    if missing_clinical:
        raise KeyError("Missing clinical columns: " + ", ".join(missing_clinical))

    preprocessor = joblib.load(preprocessor_path)
    clinical = runtime.transform_clinical(preprocessor, cohort, clinical_columns)
    expected_dimension = int(checkpoint["clinical_schema"]["output_dim"])
    if clinical.shape[1] != expected_dimension:
        raise RuntimeError(
            f"Clinical feature dimension mismatch: expected {expected_dimension}, got {clinical.shape[1]}"
        )
    preprocessing = runtime.preprocessing_functions()
    model = build_model(model_args, expected_dimension, device)
    materialize(
        model, cohort.iloc[0], columns, clinical[0:1], preprocessing, device, dtype
    )
    model.load_state_dict(checkpoint["model_nonpillar_state"], strict=False)
    model.pillar.load_state_dict(checkpoint["lora_state"], strict=False)
    model.eval()

    probabilities, recurrence_gates, complication_gates = [], [], []
    for index, (_, row) in enumerate(cohort.iterrows(), start=1):
        ct, tumor, l3 = case_tensors(row, columns, preprocessing, device, dtype)
        clinical_row = torch.from_numpy(clinical[index - 1:index]).to(
            device=device, dtype=torch.float32
        )
        autocast = device.type == "cuda" and dtype != torch.float32
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=dtype, enabled=autocast
        ):
            logits, gate_r, gate_c = model(ct, tumor, l3, clinical_row)
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy()[0])
        recurrence_gates.append(gate_r.float().cpu().numpy()[0])
        complication_gates.append(gate_c.float().cpu().numpy()[0])
        del ct, tumor, l3, clinical_row, logits, gate_r, gate_c
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[infer] {index}/{len(cohort)} {row[columns.case_id]}", flush=True)

    probabilities = np.asarray(probabilities, dtype=np.float32)
    recurrence_gates = np.asarray(recurrence_gates, dtype=np.float32)
    complication_gates = np.asarray(complication_gates, dtype=np.float32)
    output = pd.DataFrame({
        "case_id": cohort[columns.case_id].astype(str),
        "recurrence_probability": probabilities[:, 0],
        "complication_probability": probabilities[:, 1],
        "recurrence_prediction": (probabilities[:, 0] >= args.threshold).astype(int),
        "complication_prediction": (probabilities[:, 1] >= args.threshold).astype(int),
        "threshold": float(args.threshold),
    })
    for modality_index, modality in enumerate(model.modalities):
        output[f"recurrence_gate_{modality}"] = recurrence_gates[:, modality_index]
    for modality_index, modality in enumerate(model.modalities):
        output[f"complication_gate_{modality}"] = complication_gates[:, modality_index]

    summary = {
        "cases": int(len(cohort)),
        "checkpoint": checkpoint_path.name,
        "threshold": float(args.threshold),
        "labels_present": bool(columns.has_labels),
    }
    if columns.has_labels:
        labels = cohort[[columns.recurrence, columns.complication]].to_numpy(np.int64)
        output.insert(1, "recurrence_true", labels[:, 0])
        output.insert(2, "complication_true", labels[:, 1])
        summary["recurrence"] = _task_metrics(labels[:, 0], probabilities[:, 0], args.threshold)
        summary["complication"] = _task_metrics(labels[:, 1], probabilities[:, 1], args.threshold)
        summary["recurrence"]["roc_auc_95_ci"] = _bootstrap_auc(
            labels[:, 0], probabilities[:, 0], args.bootstrap_samples, args.seed
        )
        summary["complication"]["roc_auc_95_ci"] = _bootstrap_auc(
            labels[:, 1], probabilities[:, 1], args.bootstrap_samples, args.seed + 1
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "predictions.csv", index=False)
    runtime.save_json(summary, output_dir / "metrics.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[result] {output_dir / 'predictions.csv'}")
    return output, summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", default="bfloat16")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
