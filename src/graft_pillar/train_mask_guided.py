#!/usr/bin/env python
"""Train mask-guided Pillar-0 + clinical dual-task model.

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
            "args": vars(args),
            "lora_audit": None if audit is None else asdict(audit),
        },
        path,
    )
    joblib.dump(clinical_prep, path.with_suffix(".clinical_preprocessor.joblib"))


def load_saved_fold(fold_dir, fold):
    """Build OOF rows from a completed fold without retraining it."""
    with np.load(fold_dir / "best_val.npz", allow_pickle=True) as z:
        ids = z["case_ids"].astype(str)
        yt, pp = z["y_true"], z["y_prob"]
        gr, gc = z["gate_recurrence"], z["gate_complication"]
        modalities = z["modality_names"].astype(str).tolist()
    history_path = fold_dir / "history.csv"
    history = pd.read_csv(history_path)
    best_row = history.loc[history["mean_auc"].astype(float).idxmax()]
    best_epoch = int(best_row["epoch"])
    out = pd.DataFrame({
        "case_id": ids, "fold": fold,
        "recurrence_true": yt[:, 0].astype(int),
        "complication_true": yt[:, 1].astype(int),
        "recurrence_prob": pp[:, 0], "complication_prob": pp[:, 1],
    })
    for j, name in enumerate(modalities):
        out[f"gate_R_{name}"] = gr[:, j]
        out[f"gate_C_{name}"] = gc[:, j]
    met = base.metrics(yt, pp)
    met.update({"fold": fold, "best_epoch": best_epoch})
    base.save_json(met, fold_dir / "metrics.json")
    return out, met


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
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y)
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
        f"[fold {fold}] epoch={epoch:02d} {phase} start n={len(indices)}",
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
                loss = base.loss_fn(logits.float(), y.float(), weights)
            probs.append(torch.sigmoid(logits.float()).cpu().numpy()[0])
            losses.append(float(loss.cpu()))
            gates_r.append(gr.float().cpu().numpy()[0])
            gates_c.append(gc.float().cpu().numpy()[0])
            del ct, tumor, l3, cl, y, logits
            if (local_i + 1) % 5 == 0 or local_i + 1 == len(indices):
                print(
                    f"[fold {fold}] epoch={epoch:02d} {phase} "
                    f"{local_i+1}/{len(indices)}",
                    flush=True,
                )
    return (
        np.asarray(probs, np.float32),
        np.asarray(losses, np.float32),
        np.asarray(gates_r, np.float32),
        np.asarray(gates_c, np.float32),
    )


def train_fold(fold, df, cols, folds, clinical_cols, args, ops, device, dtype):
    fold_dir = Path(args.output_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    yall = df[[cols.recurrence, cols.complication]].to_numpy(np.float32)

    # The outer test fold is locked until model/epoch selection is complete.
    outer_train = np.where(folds != fold)[0]
    outer_test = np.where(folds == fold)[0]
    outer_strata = (
        2 * yall[outer_train, 0].astype(int) + yall[outer_train, 1].astype(int)
    )
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=args.inner_validation_fraction,
        random_state=args.seed + fold,
    )
    inner_train_rel, inner_val_rel = next(
        splitter.split(np.zeros(len(outer_train)), outer_strata)
    )
    tr = outer_train[inner_train_rel]
    va = outer_train[inner_val_rel]
    te = outer_test
    ytr, yva, yte = yall[tr], yall[va], yall[te]

    split_role = np.full(len(df), "unused", dtype=object)
    split_role[tr] = "inner_train"
    split_role[va] = "inner_validation"
    split_role[te] = "outer_test"
    pd.DataFrame({
        "case_id": df[cols.case_id].astype(str),
        "outer_fold": fold,
        "role": split_role,
    }).to_csv(fold_dir / "split_roles.csv", index=False)

    prep, schema = base.fit_clinical_preprocessor(df.iloc[tr], clinical_cols)
    cltr = base.transform_clinical(prep, df.iloc[tr], clinical_cols)
    clva = base.transform_clinical(prep, df.iloc[va], clinical_cols)
    clte = base.transform_clinical(prep, df.iloc[te], clinical_cols)
    base.save_json(schema, fold_dir / "clinical_schema.json")
    model, audit = make_model(args, cltr.shape[1], device)
    print(f"[fold {fold}] initializing mask-guided feature layers", flush=True)
    shapes = materialize_and_check(
        model, df.iloc[tr[0]], cols, cltr[0:1], ops, device, dtype
    )
    print(f"[fold {fold}] feature shapes={shapes}", flush=True)
    base.save_json({"feature_shapes": shapes}, fold_dir / "mask_guided_feature_shapes.json")
    opt = optimizer_for(model, args)
    weights = base.pos_weights(ytr, device)
    rng = np.random.default_rng(args.seed + fold)
    best, best_epoch, bad, history = -np.inf, -1, 0, []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        lora_on = args.lora_r > 0 and epoch > args.lora_warmup_epochs
        base.set_lora_requires_grad(model.pillar, lora_on)
        model.training_mode()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        order = rng.permutation(len(tr))
        print(
            f"[fold {fold}] epoch={epoch:02d} train start n={len(order)}",
            flush=True,
        )
        for k, li in enumerate(order):
            row = df.iloc[tr[li]]
            ct, tumor, l3 = case_tensors(row, cols, *ops, device, dtype)
            cl = torch.from_numpy(cltr[li:li+1]).to(device=device, dtype=torch.float32)
            y = torch.from_numpy(ytr[li:li+1]).to(device=device)
            enabled = device.type == "cuda" and dtype != torch.float32
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                logits = model(ct, tumor, l3, cl)
                loss = base.loss_fn(logits.float(), y.float(), weights)
                (loss / args.grad_accum).backward()
            running += float(loss.detach().cpu())
            if (k + 1) % args.grad_accum == 0 or k + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip
                )
                opt.step(); opt.zero_grad(set_to_none=True)
            del ct, tumor, l3, cl, y, logits, loss
            if args.empty_cache_each_case:
                torch.cuda.empty_cache()
            if (k + 1) % 5 == 0 or k + 1 == len(order):
                print(
                    f"[fold {fold}] epoch={epoch:02d} train "
                    f"{k+1}/{len(order)} loss={running/(k+1):.4f}",
                    flush=True,
                )

        probs, losses, gates_r, gates_c = evaluate_cases(
            model, va, clva, yva, df, cols, weights, args, ops, device, dtype,
            fold, epoch, "inner-validation",
        )
        met = base.metrics(yva, probs)
        record = {
            "epoch": epoch,
            "lora_enabled": int(lora_on),
            "train_loss": running / len(tr),
            "val_loss": float(np.mean(losses)),
            "recurrence_auc": met["recurrence"]["auc"],
            "complication_auc": met["complication"]["auc"],
            "mean_auc": met["mean_auc"],
            "seconds": time.time() - start,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
        print(
            f"[fold {fold}] ep={epoch:02d} LoRA={'ON' if lora_on else 'OFF'} "
            f"loss={record['train_loss']:.4f}/{record['val_loss']:.4f} "
            f"AUC_R={record['recurrence_auc']:.4f} AUC_C={record['complication_auc']:.4f} "
            f"mean={record['mean_auc']:.4f} time={record['seconds']/60:.1f}m"
        )
        if np.isfinite(met["mean_auc"]) and met["mean_auc"] > best + args.min_delta:
            best, best_epoch, bad = met["mean_auc"], epoch, 0
            save_checkpoint(fold_dir / "best.pt", model, prep, schema, epoch, met, args, audit)
            np.savez_compressed(
                fold_dir / "best_inner_val.npz",
                case_ids=np.asarray(
                    df.iloc[va][cols.case_id].astype(str).tolist(), dtype=str
                ),
                y_true=yva, y_prob=probs,
                gate_recurrence=gates_r, gate_complication=gates_c,
                modality_names=np.asarray(model.modalities),
            )
        else:
            bad += 1
        if bad >= args.patience:
            break

    if best_epoch < 0:
        raise RuntimeError(f"Fold {fold}: no valid checkpoint selected")

    # Load the inner-selected checkpoint, then evaluate the untouched outer
    # test fold exactly once.
    checkpoint = torch.load(
        fold_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_nonpillar_state"], strict=False)
    model.pillar.load_state_dict(checkpoint["lora_state"], strict=False)
    outer_probs, _, outer_gr, outer_gc = evaluate_cases(
        model, te, clte, yte, df, cols, weights, args, ops, device, dtype,
        fold, best_epoch, "outer-test",
    )
    np.savez_compressed(
        fold_dir / "best_val.npz",
        case_ids=np.asarray(df.iloc[te][cols.case_id].astype(str).tolist(), dtype=str),
        y_true=yte,
        y_prob=outer_probs,
        gate_recurrence=outer_gr,
        gate_complication=outer_gc,
        modality_names=np.asarray(model.modalities, dtype=str),
    )
    del model
    torch.cuda.empty_cache()
    return load_saved_fold(fold_dir, fold)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--project-src", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fold", default="0", help="0..n_splits-1 or all")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--inner-validation-fraction", type=float, default=0.20)
    p.add_argument("--clinical-cols", default=None)
    p.add_argument("--stage-indices", type=int, nargs="+", default=[0,1,2])
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
    p.add_argument("--lr-lora", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp-dtype", default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--empty-cache-each-case", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    base.set_seed(args.seed)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = base.amp_dtype(args.amp_dtype)
    prepare_hu_volume, make_windowed_tensor = base.import_project_ops(args.project_src)
    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    cols = base.detect_cols(df)
    df = base.validate_df(df, cols)
    clinical_cols = base.select_clinical_columns(df, args.clinical_cols)
    if not clinical_cols:
        raise RuntimeError("No approved clinical variables detected")
    base.save_json({"selected_columns": clinical_cols}, out / "clinical_columns_selected.json")
    print(f"[DATA] n={len(df)} clinical={len(clinical_cols)}")
    ops = (prepare_hu_volume, make_windowed_tensor)
    if args.dry_run:
        dry_run(df, cols, clinical_cols, args, ops, device, dtype)
        return
    folds = base.make_splits(df, cols, out / "cv_splits.csv", args.n_splits, args.seed)
    selected = list(range(args.n_splits)) if args.fold.lower() == "all" else [int(args.fold)]
    all_oof, fold_metrics = [], []
    for fold in selected:
        fold_dir = out / f"fold_{fold}"
        if (
            (fold_dir / "best.pt").exists()
            and (fold_dir / "best_val.npz").exists()
            and (fold_dir / "history.csv").exists()
        ):
            print(f"[RESUME] reuse completed fold {fold}: {fold_dir}", flush=True)
            oof, met = load_saved_fold(fold_dir, fold)
        else:
            oof, met = train_fold(
                fold, df, cols, folds, clinical_cols, args, ops, device, dtype
            )
        all_oof.append(oof); fold_metrics.append(met)
    oof = pd.concat(all_oof, ignore_index=True)
    name = "oof_predictions.csv" if args.fold.lower() == "all" else f"oof_predictions_fold_{selected[0]}.csv"
    oof.to_csv(out / name, index=False)
    yt = oof[["recurrence_true","complication_true"]].to_numpy(int)
    pp = oof[["recurrence_prob","complication_prob"]].to_numpy(float)
    result = base.metrics(yt, pp); result["fold_metrics"] = fold_metrics
    base.save_json(result, out / "cv_metrics.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("[RESULT]", out / name)


if __name__ == "__main__":
    main()\n