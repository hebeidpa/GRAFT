#!/usr/bin/env python
"""Train GRAFT on one cohort and evaluate a separate internal validation cohort.

This entry point replaces the legacy 67-D handcrafted phenotype branch with
deep tumor/L3 ROI embeddings pooled from Pillar-0 intermediate feature maps.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
from sklearn.model_selection import StratifiedShuffleSplit

from . import training_support as base
from .mask_guided import (
    FourModalTaskAwareFusionExample,
    PillarMaskGuidedEncoder,
)


def aligned_mask_array(
    mask_path,
    ct_path,
    target_spacing,
    target_shape,
    label_values=None,
):
    """Resample a label mask onto the exact grid used for Pillar CT input."""
    ct = sitk.ReadImage(str(ct_path))
    mask = sitk.ReadImage(str(mask_path))
    old_spacing = ct.GetSpacing()
    old_size = ct.GetSize()
    new_size = [
        int(round(old_size[i] * old_spacing[i] / float(target_spacing[i])))
        for i in range(3)
    ]

    # Sparse forward mapping preserves single-slice ROIs that can disappear
    # under ordinary nearest-neighbour downsampling. Only foreground voxels
    # are transformed, which is much faster than dilating full CT-sized masks.
    source = sitk.GetArrayFromImage(mask)
    if label_values is None:
        keep = source != 0
    else:
        keep = np.isin(source, np.asarray(label_values))
    source_zyx = np.argwhere(keep)
    if len(source_zyx) == 0:
        arr = np.zeros(tuple(reversed(new_size)), dtype=np.int16)
    else:
        values = (
            np.ones(len(source_zyx), dtype=np.int16)
            if label_values is None
            else source[tuple(source_zyx.T)].astype(np.int16, copy=False)
        )
        source_xyz = source_zyx[:, ::-1].astype(np.float64)
        scaled = source_xyz * np.asarray(mask.GetSpacing(), dtype=np.float64)
        mask_direction = np.asarray(mask.GetDirection(), dtype=np.float64).reshape(3, 3)
        physical = (
            scaled @ mask_direction.T
            + np.asarray(mask.GetOrigin(), dtype=np.float64)
        )
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
        output_xyz = output_xyz[valid]
        values = values[valid]
        arr = np.zeros(tuple(reversed(new_size)), dtype=np.int16)
        np.maximum.at(
            arr,
            (output_xyz[:, 2], output_xyz[:, 1], output_xyz[:, 0]),
            values,
        )

    target = tuple(int(v) for v in target_shape)
    padded = np.zeros(target, dtype=np.int16)
    src_slices, dst_slices = [], []
    for src_len, dst_len in zip(arr.shape, target):
        copy_len = min(src_len, dst_len)
        src_start = max((src_len - copy_len) // 2, 0)
        dst_start = max((dst_len - copy_len) // 2, 0)
        src_slices.append(slice(src_start, src_start + copy_len))
        dst_slices.append(slice(dst_start, dst_start + copy_len))
    padded[tuple(dst_slices)] = arr[tuple(src_slices)].astype(np.int16)
    return padded


def case_tensors(row, cols, prepare_hu_volume, make_windowed_tensor, device, dtype):
    ct = base.ct_tensor(
        row, cols, prepare_hu_volume, make_windowed_tensor, device, dtype
    )
    spacing = (1.25, 1.25, 1.25)
    shape = (384, 384, 384)
    tumor_np = aligned_mask_array(
        row[cols.tumor], row[cols.ct], spacing, shape, label_values=None
    )
    l3_np = aligned_mask_array(
        row[cols.l3], row[cols.ct], spacing, shape, label_values=(1, 2, 3)
    )
    tumor_np = (tumor_np > 0).astype(np.uint8)
    if int(tumor_np.sum()) == 0:
        raise ValueError(f"Empty tumor mask after preprocessing: {row[cols.case_id]}")
    missing_l3 = [v for v in (1, 2, 3) if not np.any(l3_np == v)]
    if missing_l3:
        raise ValueError(
            f"L3 labels {missing_l3} absent after preprocessing: {row[cols.case_id]}"
        )
    tumor = torch.from_numpy(tumor_np).unsqueeze(0).unsqueeze(0).to(device)
    l3 = torch.from_numpy(l3_np.astype(np.uint8)).unsqueeze(0).unsqueeze(0).to(device)
    return ct, tumor, l3


class MaskGuidedPillarNet(nn.Module):
    def __init__(self, pillar, clinical_dim, args):
        super().__init__()
        self.pillar = pillar
        self.image_encoder = PillarMaskGuidedEncoder(
            pillar,
            emb_dim=args.emb_dim,
            stage_indices=tuple(args.stage_indices),
            dropout=args.encoder_dropout,
            l3_labels=(1, 2, 3),
        )
        self.fusion = FourModalTaskAwareFusionExample(
            emb_dim=args.emb_dim,
            clinical_dim=clinical_dim,
            hidden=args.head_hidden,
            dropout=args.head_dropout,
        )
        self.modalities = ["ct", "tumor", "l3", "clinical"]

    def training_mode(self):
        self.train()
        self.pillar.eval()
        base.set_lora_train(self.pillar, True)

    def forward(self, ct, tumor, l3, clinical, return_gates=False):
        image = self.image_encoder(ct, tumor, l3)
        out = self.fusion(
            image["ct_embedding"],
            image["tumor_embedding"],
            image["l3_embedding"],
            clinical,
        )
        logits = torch.stack(
            [out["recurrence_logit"], out["complication_logit"]], dim=1
        )
        if return_gates:
            return (
                logits,
                out["recurrence_modality_weights"],
                out["complication_modality_weights"],
                image,
            )
        return logits


def make_model(args, clinical_dim, device):
    pillar = base.load_pillar(args.model_dir, device)
    audit = None
    if args.lora_r > 0:
        audit = base.inject_lora_last_stage(
            pillar,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            last_n_blocks=args.lora_blocks,
            targets=tuple(x.strip() for x in args.lora_targets.split(",") if x.strip()),
        )
    return MaskGuidedPillarNet(pillar, clinical_dim, args).to(device), audit


def materialize_and_check(model, row, cols, clinical_row, ops, device, dtype):
    prepare_hu_volume, make_windowed_tensor = ops
    ct, tumor, l3 = case_tensors(
        row, cols, prepare_hu_volume, make_windowed_tensor, device, dtype
    )
    clinical = torch.from_numpy(clinical_row).to(device=device, dtype=torch.float32)
    enabled = device.type == "cuda" and dtype != torch.float32
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=dtype, enabled=enabled
    ):
        logits, _, _, image = model(ct, tumor, l3, clinical, return_gates=True)
    shapes = image["feature_shapes"]
    del ct, tumor, l3, clinical, logits, image
    return shapes


def optimizer_for(model, args):
    main = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("pillar.")
    ]
    lora = [p for p in model.pillar.parameters() if p.requires_grad]
    groups = [{"params": main, "lr": args.lr_head, "weight_decay": args.weight_decay}]
    if lora:
        groups.append({"params": lora, "lr": args.lr_lora, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


def binary_focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    """Standard alpha-balanced binary focal loss."""
    targets = targets.float()
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, prob, 1.0 - prob)
    alpha_t = torch.where(
        targets > 0.5,
        torch.as_tensor(alpha, device=logits.device, dtype=logits.dtype),
        torch.as_tensor(1.0 - alpha, device=logits.device, dtype=logits.dtype),
    )
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()


def objective_loss(logits, targets, pos_weights, args):
    """Weighted BCE for recurrence plus focal loss for complications."""
    recurrence = F.binary_cross_entropy_with_logits(
        logits[:, 0].float(),
        targets[:, 0].float(),
        pos_weight=pos_weights[0],
    )
    complication = binary_focal_loss_with_logits(
        logits[:, 1],
        targets[:, 1],
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
    )
    return (
        args.lambda_recurrence * recurrence
        + args.lambda_complication * complication
    )


def bootstrap_metric_intervals(y_true, y_prob, samples, seed):
    """Patient-level percentile bootstrap confidence intervals."""
    rng = np.random.default_rng(seed)
    collected = {
        endpoint: {metric: [] for metric in ("auc", "pr_auc", "brier")}
        for endpoint in ("recurrence", "complication")
    }
    n = len(y_true)
    for _ in range(samples):
        index = rng.integers(0, n, size=n)
        estimate = base.metrics(y_true[index], y_prob[index])
        for endpoint in collected:
            for metric in collected[endpoint]:
                value = estimate[endpoint][metric]
                if np.isfinite(value):
                    collected[endpoint][metric].append(float(value))
    intervals = {}
    for endpoint, metrics in collected.items():
        intervals[endpoint] = {}
        for metric, values in metrics.items():
            intervals[endpoint][metric] = (
                [float(x) for x in np.percentile(values, [2.5, 97.5])]
                if values else [float("nan"), float("nan")]
            )
    return intervals


def save_checkpoint(path, model, clinical_prep, clinical_schema, epoch, metrics, args, audit):
    state = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if not k.startswith("pillar.") and not k.startswith("image_encoder.pillar.")
    }
    torch.save(
        {
            "epoch": int(epoch),
            "model_nonpillar_state": state,
            "lora_state": base.lora_state(model.pillar),
            "clinical_schema": clinical_schema,
            "val_metrics": metrics,
            "modalities": model.modalities,
            "args": {k: v for k, v in vars(args).items() if k not in {"train_csv", "validation_csv", "csv", "model_dir", "project_src", "output_dir"}},
            "lora_audit": None if audit is None else asdict(audit),
        },
        path,
    )
    joblib.dump(clinical_prep, path.with_suffix(".clinical_preprocessor.joblib"))


def dry_run(df, cols, clinical_cols, args, ops, device, dtype):
    prep, schema = base.fit_clinical_preprocessor(df, clinical_cols)
    clinical = base.transform_clinical(prep, df, clinical_cols)
    model, audit = make_model(args, clinical.shape[1], device)
    shapes = materialize_and_check(
        model, df.iloc[0], cols, clinical[0:1], ops, device, dtype
    )
    opt = optimizer_for(model, args)
    ct, tumor, l3 = case_tensors(df.iloc[0], cols, *ops, device, dtype)
    cl = torch.from_numpy(clinical[0:1]).to(device=device, dtype=torch.float32)
    y = torch.tensor(
        [[float(df.iloc[0][cols.recurrence]), float(df.iloc[0][cols.complication])]],
        device=device,
    )
    model.training_mode()
    opt.zero_grad(set_to_none=True)
    enabled = device.type == "cuda" and dtype != torch.float32
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
        logits, gate_r, gate_c, _ = model(ct, tumor, l3, cl, return_gates=True)
        weights = base.pos_weights(
            df[[cols.recurrence, cols.complication]].to_numpy(np.float32), device
        )
        loss = objective_loss(logits.float(), y, weights, args)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], args.grad_clip
    )
    opt.step()
    print("[DRY-RUN] clinical schema:", json.dumps(schema, ensure_ascii=False))
    print("[DRY-RUN] feature shapes:", shapes)
    print("[DRY-RUN] logits:", logits.detach().float().cpu().tolist())
    print("[DRY-RUN] gate recurrence:", dict(zip(model.modalities, gate_r.detach().float().cpu()[0].tolist())))
    print("[DRY-RUN] gate complication:", dict(zip(model.modalities, gate_c.detach().float().cpu()[0].tolist())))
    print("[DRY-RUN] loss:", float(loss.detach().cpu()))
    if device.type == "cuda":
        print(f"[DRY-RUN] peak CUDA={torch.cuda.max_memory_allocated(device)/(1024**3):.2f} GB")
    print("[DRY-RUN] MASK_GUIDED_SUCCESS")


def evaluate_cases(
    model, indices, clinical, labels, df, cols, weights, args, ops, device, dtype,
    fold, epoch, phase,
):
    model.eval()
    probs, losses, gates_r, gates_c = [], [], [], []
    print(
        f"[holdout] epoch={epoch:02d} {phase} start n={len(indices)}",
        flush=True,
    )
    with torch.no_grad():
        for local_i, global_i in enumerate(indices):
            row = df.iloc[global_i]
            ct, tumor, l3 = case_tensors(row, cols, *ops, device, dtype)
            cl = torch.from_numpy(clinical[local_i:local_i+1]).to(
                device=device, dtype=torch.float32
            )
            y = torch.from_numpy(labels[local_i:local_i+1]).to(device=device)
            enabled = device.type == "cuda" and dtype != torch.float32
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                logits, gr, gc, _ = model(ct, tumor, l3, cl, return_gates=True)
                loss = objective_loss(logits.float(), y.float(), weights, args)
            probs.append(torch.sigmoid(logits.float()).cpu().numpy()[0])
            losses.append(float(loss.cpu()))
            gates_r.append(gr.float().cpu().numpy()[0])
            gates_c.append(gc.float().cpu().numpy()[0])
            del ct, tumor, l3, cl, y, logits
            if (local_i + 1) % 5 == 0 or local_i + 1 == len(indices):
                print(
                    f"[holdout] epoch={epoch:02d} {phase} "
                    f"{local_i+1}/{len(indices)}",
                    flush=True,
                )
    return (
        np.asarray(probs, np.float32),
        np.asarray(losses, np.float32),
        np.asarray(gates_r, np.float32),
        np.asarray(gates_c, np.float32),
    )



def load_selected_epoch(selection_dir):
    checkpoint = torch.load(
        selection_dir / "best.pt", map_location="cpu", weights_only=False
    )
    return int(checkpoint["epoch"])


def select_epoch(train_df, cols, clinical_cols, args, ops, device, dtype):
    """Select the epoch using only the 941-case training cohort."""
    selection_dir = Path(args.output_dir) / "epoch_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    if (
        (selection_dir / "best.pt").exists()
        and (selection_dir / "history.csv").exists()
    ):
        epoch = load_selected_epoch(selection_dir)
        print(f"[RESUME] selected epoch={epoch}", flush=True)
        return epoch

    labels = train_df[[cols.recurrence, cols.complication]].to_numpy(np.float32)
    strata = 2 * labels[:, 0].astype(int) + labels[:, 1].astype(int)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=args.early_stopping_fraction,
        random_state=args.seed,
    )
    parameter_indices, stopping_indices = next(
        splitter.split(np.zeros(len(train_df)), strata)
    )
    roles = np.full(len(train_df), "parameter_estimation", dtype=object)
    roles[stopping_indices] = "early_stopping"
    pd.DataFrame({
        "case_id": train_df[cols.case_id].astype(str),
        "role": roles,
    }).to_csv(selection_dir / "split_roles.csv", index=False)

    y_train = labels[parameter_indices]
    y_stop = labels[stopping_indices]
    prep, schema = base.fit_clinical_preprocessor(
        train_df.iloc[parameter_indices], clinical_cols
    )
    clinical_train = base.transform_clinical(
        prep, train_df.iloc[parameter_indices], clinical_cols
    )
    clinical_stop = base.transform_clinical(
        prep, train_df.iloc[stopping_indices], clinical_cols
    )
    base.save_json(schema, selection_dir / "clinical_schema.json")
    model, audit = make_model(args, clinical_train.shape[1], device)
    shapes = materialize_and_check(
        model,
        train_df.iloc[parameter_indices[0]],
        cols,
        clinical_train[0:1],
        ops,
        device,
        dtype,
    )
    base.save_json(
        {"feature_shapes": shapes},
        selection_dir / "mask_guided_feature_shapes.json",
    )
    optimizer = optimizer_for(model, args)
    weights = base.pos_weights(y_train, device)
    rng = np.random.default_rng(args.seed)
    best_loss, best_epoch, bad_epochs, history = np.inf, -1, 0, []

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        lora_on = args.lora_r > 0 and epoch > args.lora_warmup_epochs
        base.set_lora_requires_grad(model.pillar, lora_on)
        model.training_mode()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        order = rng.permutation(len(parameter_indices))
        print(
            f"[SELECTION] epoch={epoch:02d} train start n={len(order)}",
            flush=True,
        )
        for step, local_index in enumerate(order):
            global_index = parameter_indices[local_index]
            row = train_df.iloc[global_index]
            ct, tumor, l3 = case_tensors(row, cols, *ops, device, dtype)
            clinical = torch.from_numpy(
                clinical_train[local_index:local_index + 1]
            ).to(device=device, dtype=torch.float32)
            target = torch.from_numpy(
                y_train[local_index:local_index + 1]
            ).to(device=device)
            enabled = device.type == "cuda" and dtype != torch.float32
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=enabled
            ):
                logits = model(ct, tumor, l3, clinical)
                loss = objective_loss(logits.float(), target.float(), weights, args)
                (loss / args.grad_accum).backward()
            running += float(loss.detach().cpu())
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            del ct, tumor, l3, clinical, target, logits, loss
            if args.empty_cache_each_case:
                torch.cuda.empty_cache()
            if (step + 1) % 5 == 0 or step + 1 == len(order):
                print(
                    f"[SELECTION] epoch={epoch:02d} train "
                    f"{step+1}/{len(order)} loss={running/(step+1):.4f}",
                    flush=True,
                )

        probabilities, losses, gate_r, gate_c = evaluate_cases(
            model,
            stopping_indices,
            clinical_stop,
            y_stop,
            train_df,
            cols,
            weights,
            args,
            ops,
            device,
            dtype,
            0,
            epoch,
            "early-stopping",
        )
        metrics = base.metrics(y_stop, probabilities)
        record = {
            "epoch": epoch,
            "lora_enabled": int(lora_on),
            "train_loss": running / len(parameter_indices),
            "early_stopping_loss": float(np.mean(losses)),
            "recurrence_auc": metrics["recurrence"]["auc"],
            "complication_auc": metrics["complication"]["auc"],
            "mean_auc": metrics["mean_auc"],
            "seconds": time.time() - started,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(selection_dir / "history.csv", index=False)
        print(
            f"[SELECTION] ep={epoch:02d} "
            f"loss={record['train_loss']:.4f}/{record['early_stopping_loss']:.4f} "
            f"AUC_R={record['recurrence_auc']:.4f} "
            f"AUC_C={record['complication_auc']:.4f}",
            flush=True,
        )
        current = record["early_stopping_loss"]
        if np.isfinite(current) and current < best_loss - args.min_delta:
            best_loss, best_epoch, bad_epochs = current, epoch, 0
            checkpoint_metrics = dict(metrics)
            checkpoint_metrics["early_stopping_loss"] = current
            save_checkpoint(
                selection_dir / "best.pt",
                model,
                prep,
                schema,
                epoch,
                checkpoint_metrics,
                args,
                audit,
            )
            np.savez_compressed(
                selection_dir / "best_early_stopping_predictions.npz",
                case_ids=np.asarray(
                    train_df.iloc[stopping_indices][cols.case_id]
                    .astype(str).tolist(),
                    dtype=str,
                ),
                y_true=y_stop,
                y_prob=probabilities,
                gate_recurrence=gate_r,
                gate_complication=gate_c,
                modality_names=np.asarray(model.modalities, dtype=str),
            )
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    if best_epoch < 0:
        raise RuntimeError("No valid epoch was selected")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch


def load_final_model(
    train_df, cols, clinical_cols, args, ops, device, dtype, final_path
):
    checkpoint = torch.load(final_path, map_location=device, weights_only=False)
    prep = joblib.load(final_path.with_suffix(".clinical_preprocessor.joblib"))
    schema = checkpoint["clinical_schema"]
    clinical = base.transform_clinical(prep, train_df.iloc[:1], clinical_cols)
    model, _ = make_model(args, int(schema["output_dim"]), device)
    materialize_and_check(
        model, train_df.iloc[0], cols, clinical, ops, device, dtype
    )
    model.load_state_dict(checkpoint["model_nonpillar_state"], strict=False)
    model.pillar.load_state_dict(checkpoint["lora_state"], strict=False)
    return model, prep, schema


def train_final_model(
    train_df, cols, clinical_cols, args, ops, device, dtype, final_epochs
):
    """Retrain the deployable model on all 941 training cases."""
    out = Path(args.output_dir)
    final_path = out / "final_model.pt"
    if final_path.exists():
        print(f"[FINAL] reuse existing model: {final_path}", flush=True)
        return load_final_model(
            train_df, cols, clinical_cols, args, ops, device, dtype, final_path
        )

    labels = train_df[[cols.recurrence, cols.complication]].to_numpy(np.float32)
    prep, schema = base.fit_clinical_preprocessor(train_df, clinical_cols)
    clinical = base.transform_clinical(prep, train_df, clinical_cols)
    model, audit = make_model(args, clinical.shape[1], device)
    shapes = materialize_and_check(
        model, train_df.iloc[0], cols, clinical[0:1], ops, device, dtype
    )
    print(
        f"[FINAL] train on all n={len(train_df)} for {final_epochs} epochs; "
        f"feature shapes={shapes}",
        flush=True,
    )
    optimizer = optimizer_for(model, args)
    weights = base.pos_weights(labels, device)
    rng = np.random.default_rng(args.seed + 10000)
    history = []

    for epoch in range(1, final_epochs + 1):
        started = time.time()
        lora_on = args.lora_r > 0 and epoch > args.lora_warmup_epochs
        base.set_lora_requires_grad(model.pillar, lora_on)
        model.training_mode()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        order = rng.permutation(len(train_df))
        for step, index in enumerate(order):
            row = train_df.iloc[index]
            ct, tumor, l3 = case_tensors(row, cols, *ops, device, dtype)
            clinical_row = torch.from_numpy(clinical[index:index + 1]).to(
                device=device, dtype=torch.float32
            )
            target = torch.from_numpy(labels[index:index + 1]).to(device=device)
            enabled = device.type == "cuda" and dtype != torch.float32
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=enabled
            ):
                logits = model(ct, tumor, l3, clinical_row)
                loss = objective_loss(logits.float(), target.float(), weights, args)
                (loss / args.grad_accum).backward()
            running += float(loss.detach().cpu())
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            del ct, tumor, l3, clinical_row, target, logits, loss
            if args.empty_cache_each_case:
                torch.cuda.empty_cache()
            if (step + 1) % 20 == 0 or step + 1 == len(order):
                print(
                    f"[FINAL] epoch={epoch:02d} {step+1}/{len(order)} "
                    f"loss={running/(step+1):.4f}",
                    flush=True,
                )
        history.append({
            "epoch": epoch,
            "lora_enabled": int(lora_on),
            "train_loss": running / len(train_df),
            "seconds": time.time() - started,
        })
        pd.DataFrame(history).to_csv(
            out / "final_model_history.csv", index=False
        )

    save_checkpoint(
        final_path,
        model,
        prep,
        schema,
        final_epochs,
        {
            "training_loss": history[-1]["train_loss"],
            "selection_rule": "training-cohort early-stopping loss",
        },
        args,
        audit,
    )
    print(f"[FINAL] saved: {final_path}", flush=True)
    return model, prep, schema


def evaluate_internal_validation(
    model,
    prep,
    validation_df,
    validation_cols,
    clinical_cols,
    training_labels,
    args,
    ops,
    device,
    dtype,
):
    clinical = base.transform_clinical(prep, validation_df, clinical_cols)
    labels = validation_df[
        [validation_cols.recurrence, validation_cols.complication]
    ].to_numpy(np.float32)
    weights = base.pos_weights(training_labels, device)
    indices = np.arange(len(validation_df))
    probabilities, _, gate_r, gate_c = evaluate_cases(
        model,
        indices,
        clinical,
        labels,
        validation_df,
        validation_cols,
        weights,
        args,
        ops,
        device,
        dtype,
        0,
        0,
        "separate-internal-validation",
    )
    predictions = pd.DataFrame({
        "case_id": validation_df[validation_cols.case_id].astype(str),
        "recurrence_true": labels[:, 0].astype(int),
        "complication_true": labels[:, 1].astype(int),
        "recurrence_prob": probabilities[:, 0],
        "complication_prob": probabilities[:, 1],
    })
    for index, name in enumerate(model.modalities):
        predictions[f"gate_R_{name}"] = gate_r[:, index]
        predictions[f"gate_C_{name}"] = gate_c[:, index]
    result = base.metrics(labels, probabilities)
    result["bootstrap_95_ci"] = bootstrap_metric_intervals(
        labels,
        probabilities,
        args.bootstrap_samples,
        args.seed + 20000,
    )
    return predictions, result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", required=True)
    p.add_argument(
        "--validation-csv",
        default=None,
        help="Separate internal-validation CSV; omit for training-only mode",
    )
    p.add_argument("--model-dir", required=True)
    p.add_argument("--project-src", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--early-stopping-fraction", type=float, default=0.20)
    p.add_argument("--clinical-cols", default=None)
    p.add_argument("--stage-indices", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-blocks", type=int, default=2)
    p.add_argument("--lora-targets", default="q,kv")
    p.add_argument("--lora-warmup-epochs", type=int, default=2)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--encoder-dropout", type=float, default=0.10)
    p.add_argument("--head-hidden", type=int, default=64)
    p.add_argument("--head-dropout", type=float, default=0.20)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--focal-alpha", type=float, default=0.25)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--lambda-recurrence", type=float, default=1.0)
    p.add_argument("--lambda-complication", type=float, default=0.75)
    p.add_argument("--lr-lora", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp-dtype", default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--empty-cache-each-case", action="store_true")
    p.add_argument(
        "--final-epochs",
        type=int,
        default=0,
        help="Full-training-cohort epochs; 0 uses the selected epoch",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.early_stopping_fraction < 1.0:
        raise ValueError("early-stopping-fraction must be between 0 and 1")
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")
    base.set_seed(args.seed)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = base.amp_dtype(args.amp_dtype)
    ops = base.import_project_ops(args.project_src)

    train_df = pd.read_csv(args.train_csv, encoding="utf-8-sig")
    train_cols = base.detect_cols(train_df)
    train_df = base.validate_df(train_df, train_cols)
    clinical_cols = base.select_clinical_columns(train_df, args.clinical_cols)
    if not clinical_cols:
        raise RuntimeError("No approved clinical variables detected")
    validation_df = None
    validation_cols = None
    if args.validation_csv:
        validation_df = pd.read_csv(args.validation_csv, encoding="utf-8-sig")
        validation_cols = base.detect_cols(validation_df)
        validation_df = base.validate_df(validation_df, validation_cols)
        missing = [c for c in clinical_cols if c not in validation_df.columns]
        if missing:
            raise KeyError(
                "Validation cohort is missing clinical columns: "
                + ", ".join(missing)
            )
        overlap = set(train_df[train_cols.case_id].astype(str)) & set(
            validation_df[validation_cols.case_id].astype(str)
        )
        if overlap:
            raise ValueError(
                f"Training and validation cohorts share {len(overlap)} case IDs"
            )
    base.save_json(
        {"selected_columns": clinical_cols},
        out / "clinical_columns_selected.json",
    )
    print(
        f"[DATA] training={len(train_df)} "
        f"internal_validation={0 if validation_df is None else len(validation_df)} "
        f"clinical={len(clinical_cols)}"
    )
    if args.dry_run:
        dry_run(
            train_df,
            train_cols,
            clinical_cols,
            args,
            ops,
            device,
            dtype,
        )
        return

    selected_epoch = select_epoch(
        train_df,
        train_cols,
        clinical_cols,
        args,
        ops,
        device,
        dtype,
    )
    final_epochs = max(1, args.final_epochs or selected_epoch)
    model, prep, schema = train_final_model(
        train_df,
        train_cols,
        clinical_cols,
        args,
        ops,
        device,
        dtype,
        final_epochs,
    )
    training_summary = {
        "training_cases": int(len(train_df)),
        "selected_epoch": int(selected_epoch),
        "final_model_epochs": int(final_epochs),
        "validation_status": (
            "pending_separate_cohort"
            if validation_df is None
            else "completed"
        ),
    }
    base.save_json(training_summary, out / "training_summary.json")
    if validation_df is None:
        print(
            "[RESULT] final_model.pt saved; separate internal validation pending",
            flush=True,
        )
        return
    training_labels = train_df[
        [train_cols.recurrence, train_cols.complication]
    ].to_numpy(np.float32)
    predictions, result = evaluate_internal_validation(
        model,
        prep,
        validation_df,
        validation_cols,
        clinical_cols,
        training_labels,
        args,
        ops,
        device,
        dtype,
    )
    predictions.to_csv(
        out / "internal_validation_predictions.csv", index=False
    )
    result["training_cases"] = int(len(train_df))
    result["internal_validation_cases"] = int(len(validation_df))
    result["selected_epoch"] = int(selected_epoch)
    result["final_model_epochs"] = int(final_epochs)
    result["selection_metric"] = "training_cohort_early_stopping_loss"
    result["objective"] = {
        "recurrence": "positive-class-weighted BCE",
        "complication": "alpha-balanced binary focal loss",
        "focal_alpha": args.focal_alpha,
        "focal_gamma": args.focal_gamma,
        "lambda_recurrence": args.lambda_recurrence,
        "lambda_complication": args.lambda_complication,
    }
    base.save_json(result, out / "internal_validation_metrics.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("[RESULT]", out / "internal_validation_predictions.csv")


if __name__ == "__main__":
    main()
