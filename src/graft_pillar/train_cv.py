from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .clinical import (
    clinical_summary,
    load_clinical_frame,
    make_clinical_preprocessor,
    schema_from_config,
)
from .common import configure_logging, load_config, project_paths, set_seed, write_json
from .model import FusionMultiTaskHead


LOG = logging.getLogger(__name__)


def _make_head(input_dim: int, cfg: dict) -> FusionMultiTaskHead:
    return FusionMultiTaskHead(
        input_dim=input_dim,
        head_type=cfg.get("head_type", "linear"),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        dropout=float(cfg.get("dropout", 0.3)),
    )


def _pos_weight(y: np.ndarray) -> torch.Tensor:
    positives = y.sum(axis=0)
    negatives = y.shape[0] - positives
    weights = negatives / np.maximum(positives, 1.0)
    return torch.as_tensor(weights, dtype=torch.float32)


def _fit_with_early_stopping(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    seed: int,
) -> tuple[dict[str, Any], int]:
    set_seed(seed)
    model = _make_head(x_train.shape[1], cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
    )
    train_x = torch.from_numpy(x_train).float()
    train_y = torch.from_numpy(y_train).float()
    val_x = torch.from_numpy(x_val).float()
    val_y = torch.from_numpy(y_val).float()
    pos_weight = _pos_weight(y_train)
    patience = int(cfg.get("patience", 50))
    max_epochs = int(cfg.get("max_epochs", 500))
    best_loss = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = F.binary_cross_entropy_with_logits(
                val_logits, val_y, pos_weight=pos_weight
            ).item()
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    return best_state, best_epoch


def _fit_fixed_epochs(
    x: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    epochs: int,
    seed: int,
) -> FusionMultiTaskHead:
    set_seed(seed)
    model = _make_head(x.shape[1], cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
    )
    tx = torch.from_numpy(x).float()
    ty = torch.from_numpy(y).float()
    pos_weight = _pos_weight(y)
    model.train()
    for _ in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(tx), ty, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _predict(model: FusionMultiTaskHead, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x).float())).numpy()


def _fit_transform_inputs(
    image: np.ndarray,
    clinical: pd.DataFrame,
    fit_indices: np.ndarray,
    transform_indices: np.ndarray,
    schema,
):
    image_scaler = StandardScaler().fit(image[fit_indices])
    clinical_preprocessor = make_clinical_preprocessor(schema)
    clinical_preprocessor.fit(clinical.iloc[fit_indices])
    image_part = image_scaler.transform(image[transform_indices]).astype(np.float32)
    clinical_part = np.asarray(
        clinical_preprocessor.transform(clinical.iloc[transform_indices]),
        dtype=np.float32,
    )
    return (
        np.concatenate([image_part, clinical_part], axis=1),
        image_scaler,
        clinical_preprocessor,
    )


def _metric_values(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])),
        "f1_at_0.5": float(f1_score(y, pred, zero_division=0)),
        "sensitivity_at_0.5": float(sensitivity),
        "specificity_at_0.5": float(specificity),
        "confusion_at_0.5": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _bootstrap_intervals(
    y: np.ndarray,
    p: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    values = {"roc_auc": [], "pr_auc": [], "brier": []}
    n = len(y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yy, pp = y[idx], p[idx]
        values["brier"].append(brier_score_loss(yy, pp))
        if np.unique(yy).size == 2:
            values["roc_auc"].append(roc_auc_score(yy, pp))
            values["pr_auc"].append(average_precision_score(yy, pp))
    intervals = {}
    for name, samples in values.items():
        if samples:
            lo, hi = np.percentile(samples, [2.5, 97.5])
            intervals[name] = [float(lo), float(hi)]
    return intervals


def _plot_curves(labels: np.ndarray, probabilities: np.ndarray, names: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for index, name in enumerate(names):
        fpr, tpr, _ = roc_curve(labels[:, index], probabilities[:, index])
        precision, recall, _ = precision_recall_curve(labels[:, index], probabilities[:, index])
        axes[0].plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(labels[:, index], probabilities[:, index]):.3f}")
        axes[1].plot(recall, precision, label=f"{name} AP={average_precision_score(labels[:, index], probabilities[:, index]):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Repeated-CV ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Repeated-CV precision-recall")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = project_paths(cfg)
    training_cfg = cfg["training"]
    feature_path = paths["output_dir"] / "features.npz"
    if not feature_path.exists():
        raise FileNotFoundError(f"未找到 {feature_path}; 请先运行 graft-extract")

    with np.load(feature_path, allow_pickle=False) as data:
        case_ids = data["case_ids"].astype(str)
        pillar = data["pillar_features"].astype(np.float32)
        mask = data["mask_features"].astype(np.float32)
        labels = data["labels"].astype(np.float32)
        label_names = data["label_names"].astype(str).tolist()
        mask_feature_names = data["mask_feature_names"].astype(str).tolist()
    if labels.shape[1] != 2:
        raise ValueError(f"Expected two labels, got {labels.shape}")
    if not np.isfinite(pillar).all() or not np.isfinite(mask).all():
        raise ValueError("特征中存在 NaN/Inf")

    image = np.concatenate([pillar, mask], axis=1)
    clinical_schema = schema_from_config(cfg)
    clinical, clinical_warnings = load_clinical_frame(clinical_schema, case_ids)
    for warning in clinical_warnings:
        LOG.warning(warning)
    strata = np.asarray([f"{int(a)}{int(b)}" for a, b in labels])
    counts = pd.Series(strata).value_counts()
    n_splits = int(training_cfg.get("n_splits", 5))
    if counts.min() < n_splits:
        raise ValueError(f"联合标签最小类别仅 {counts.min()} 例，不能做 {n_splits} 折分层")
    n_repeats = int(training_cfg.get("n_repeats", 10))
    seed = int(training_cfg.get("seed", 2026))
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    prediction_sum = np.zeros_like(labels, dtype=np.float64)
    prediction_count = np.zeros(len(labels), dtype=np.int32)
    prediction_rows: list[dict[str, Any]] = []
    selected_epochs: list[int] = []

    for split_index, (outer_train, outer_test) in enumerate(splitter.split(image, strata)):
        repeat = split_index // n_splits
        fold = split_index % n_splits
        split_seed = seed + split_index
        inner_train, inner_val = train_test_split(
            outer_train,
            test_size=float(training_cfg.get("validation_fraction", 0.2)),
            random_state=split_seed,
            stratify=strata[outer_train],
        )
        inner_train_x, inner_image_scaler, inner_clinical_preprocessor = _fit_transform_inputs(
            image, clinical, inner_train, inner_train, clinical_schema
        )
        inner_val_x = np.concatenate(
            [
                inner_image_scaler.transform(image[inner_val]).astype(np.float32),
                np.asarray(
                    inner_clinical_preprocessor.transform(clinical.iloc[inner_val]),
                    dtype=np.float32,
                ),
            ],
            axis=1,
        )
        _, best_epoch = _fit_with_early_stopping(
            inner_train_x,
            labels[inner_train],
            inner_val_x,
            labels[inner_val],
            training_cfg,
            split_seed,
        )
        selected_epochs.append(best_epoch)

        outer_train_x, outer_image_scaler, outer_clinical_preprocessor = _fit_transform_inputs(
            image, clinical, outer_train, outer_train, clinical_schema
        )
        outer_model = _fit_fixed_epochs(
            outer_train_x,
            labels[outer_train],
            training_cfg,
            best_epoch,
            split_seed,
        )
        outer_test_image = outer_image_scaler.transform(image[outer_test]).astype(np.float32)
        outer_test_clinical = np.asarray(
            outer_clinical_preprocessor.transform(clinical.iloc[outer_test]),
            dtype=np.float32,
        )
        probabilities = _predict(
            outer_model,
            np.concatenate([outer_test_image, outer_test_clinical], axis=1),
        )
        prediction_sum[outer_test] += probabilities
        prediction_count[outer_test] += 1
        for local_index, patient_index in enumerate(outer_test):
            prediction_rows.append(
                {
                    "case_id": case_ids[patient_index],
                    "repeat": repeat,
                    "fold": fold,
                    **{name: int(labels[patient_index, j]) for j, name in enumerate(label_names)},
                    **{f"prob_{name}": float(probabilities[local_index, j]) for j, name in enumerate(label_names)},
                }
            )
        LOG.info(
            "repeat %s/%s fold %s/%s, epoch=%s",
            repeat + 1,
            n_repeats,
            fold + 1,
            n_splits,
            best_epoch,
        )

    if not np.all(prediction_count == n_repeats):
        raise RuntimeError(f"OOF prediction counts are wrong: {np.unique(prediction_count)}")
    oof = (prediction_sum / prediction_count[:, None]).astype(np.float32)
    output_dir = paths["output_dir"] / training_cfg.get("output_subdir", "training")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "oof_predictions_all.csv", index=False, encoding="utf-8-sig")
    aggregate = pd.DataFrame({"case_id": case_ids})
    for j, name in enumerate(label_names):
        aggregate[name] = labels[:, j].astype(int)
        aggregate[f"prob_{name}"] = oof[:, j]
    aggregate.to_csv(output_dir / "oof_predictions_mean.csv", index=False, encoding="utf-8-sig")

    report: dict[str, Any] = {
        "n_cases": int(len(labels)),
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "joint_label_counts": counts.to_dict(),
        "model_inputs": {
            "pillar_feature_dim": int(pillar.shape[1]),
            "mask_feature_dim": int(mask.shape[1]),
            "clinical": clinical_summary(clinical, clinical_schema),
        },
        "selected_epoch_median": int(round(float(np.median(selected_epochs)))),
        "tasks": {},
    }
    for index, name in enumerate(label_names):
        task = _metric_values(labels[:, index], oof[:, index])
        task["bootstrap_95ci"] = _bootstrap_intervals(
            labels[:, index].astype(int),
            oof[:, index],
            int(training_cfg.get("bootstrap_samples", 2000)),
            seed + index,
        )
        report["tasks"][name] = task
    write_json(report, output_dir / "metrics.json")
    _plot_curves(labels, oof, label_names, output_dir / "evaluation_curves.png")

    # Fit a deployable head on all current cases using the median selected epoch.
    final_epochs = report["selected_epoch_median"]
    all_indices = np.arange(len(image))
    final_x, final_image_scaler, final_clinical_preprocessor = _fit_transform_inputs(
        image, clinical, all_indices, all_indices, clinical_schema
    )
    final_model = _fit_fixed_epochs(
        final_x, labels, training_cfg, final_epochs, seed
    )
    clinical_feature_names = final_clinical_preprocessor.get_feature_names_out().tolist()
    checkpoint = {
        "state_dict": final_model.state_dict(),
        "input_dim": int(final_x.shape[1]),
        "image_feature_dim": int(image.shape[1]),
        "pillar_feature_dim": int(pillar.shape[1]),
        "mask_feature_names": mask_feature_names,
        "clinical_id_column": clinical_schema.id_column,
        "clinical_numerical_columns": list(clinical_schema.numerical_columns),
        "clinical_categorical_columns": list(clinical_schema.categorical_columns),
        "clinical_feature_names": clinical_feature_names,
        "clinical_preprocessor": final_clinical_preprocessor,
        "label_names": label_names,
        "image_scaler_mean": final_image_scaler.mean_.astype(np.float32),
        "image_scaler_scale": final_image_scaler.scale_.astype(np.float32),
        "training_config": training_cfg,
        "epochs": final_epochs,
    }
    torch.save(checkpoint, output_dir / "final_fusion_head.pt")
    pd.DataFrame({"clinical_model_feature": clinical_feature_names}).to_csv(
        output_dir / "clinical_model_features.csv", index=False, encoding="utf-8-sig"
    )
    if training_cfg.get("head_type", "linear") == "linear":
        feature_names = [
            *[f"pillar_{index:04d}" for index in range(pillar.shape[1])],
            *mask_feature_names,
            *clinical_feature_names,
        ]
        feature_groups = [
            *(["pillar"] * pillar.shape[1]),
            *(["mask"] * len(mask_feature_names)),
            *(["clinical"] * len(clinical_feature_names)),
        ]
        weights = final_model.net.weight.detach().cpu().numpy()
        weight_table = pd.DataFrame(
            {"feature": feature_names, "feature_group": feature_groups}
        )
        for index, name in enumerate(label_names):
            weight_table[f"weight_{name}"] = weights[index]
        weight_table.to_csv(
            output_dir / "linear_feature_weights.csv",
            index=False,
            encoding="utf-8-sig",
        )
    LOG.info("交叉验证结果: %s", output_dir / "metrics.json")
    LOG.info("最终双任务头: %s", output_dir / "final_fusion_head.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated stratified CV for the two-label fusion head")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    run(args.config)


if __name__ == "__main__":
    main()
