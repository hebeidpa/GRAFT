from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .clinical import ClinicalSchema, load_clinical_frame
from .model import FusionMultiTaskHead


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict from an extracted features.npz file")
    parser.add_argument("--features", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--clinical",
        default=None,
        help="Clinical CSV/XLSX. Required for checkpoints trained with clinical variables.",
    )
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    with np.load(args.features, allow_pickle=False) as data:
        case_ids = data["case_ids"].astype(str)
        pillar = data["pillar_features"].astype(np.float32)
        mask = data["mask_features"].astype(np.float32)
        names = data["mask_feature_names"].astype(str).tolist()
    if names != checkpoint["mask_feature_names"]:
        raise ValueError("mask feature definitions do not match the checkpoint")
    image = np.concatenate([pillar, mask], axis=1)
    if "clinical_preprocessor" in checkpoint:
        if not args.clinical:
            raise ValueError("该模型包含临床变量，预测时必须提供 --clinical CSV/XLSX")
        schema = ClinicalSchema(
            file=Path(args.clinical).expanduser(),
            id_column=checkpoint["clinical_id_column"],
            numerical_columns=tuple(checkpoint["clinical_numerical_columns"]),
            categorical_columns=tuple(checkpoint["clinical_categorical_columns"]),
        )
        clinical, warnings = load_clinical_frame(schema, case_ids)
        for warning in warnings:
            print(f"警告: {warning}")
        mean = np.asarray(checkpoint["image_scaler_mean"], dtype=np.float32)
        scale = np.asarray(checkpoint["image_scaler_scale"], dtype=np.float32)
        image = ((image - mean) / scale).astype(np.float32)
        clinical_x = np.asarray(
            checkpoint["clinical_preprocessor"].transform(clinical), dtype=np.float32
        )
        x = np.concatenate([image, clinical_x], axis=1)
    else:
        # Backward compatibility with imaging-only v0.3 checkpoints.
        mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
        scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
        x = ((image - mean) / scale).astype(np.float32)
    if x.shape[1] != checkpoint["input_dim"]:
        raise ValueError(f"Input dim {x.shape[1]} != checkpoint dim {checkpoint['input_dim']}")

    cfg = checkpoint["training_config"]
    model = FusionMultiTaskHead(
        input_dim=checkpoint["input_dim"],
        head_type=cfg.get("head_type", "linear"),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        dropout=float(cfg.get("dropout", 0.3)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.from_numpy(x))).numpy()
    result = pd.DataFrame({"case_id": case_ids})
    for index, name in enumerate(checkpoint["label_names"]):
        result[f"prob_{name}"] = probabilities[:, index]
        result[f"pred_{name}_at_0.5"] = (probabilities[:, index] >= 0.5).astype(int)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"已保存: {output.resolve()}")


if __name__ == "__main__":
    main()
