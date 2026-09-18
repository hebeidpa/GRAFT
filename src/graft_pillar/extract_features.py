from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel

from .common import configure_logging, load_config, project_paths, write_json
from .mask_features import extract_mask_features
from .preprocess import make_windowed_tensor, prepare_hu_volume


LOG = logging.getLogger(__name__)


def _autocast_dtype(name: str):
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported autocast dtype: {name}")


def _load_model(model_dir: Path, device: torch.device):
    LOG.info("加载 Pillar-0: %s", model_dir)
    try:
        model = AutoModel.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Transformers 自定义代码缓存不完整。请先运行: "
            f"graft-fix-hf-cache --model-dir \"{model_dir}\""
        ) from exc
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _extract_pillar(model, tensor: torch.Tensor, dtype) -> np.ndarray:
    device_type = tensor.device.type
    amp_enabled = device_type == "cuda" and dtype != torch.float32
    with torch.inference_mode(), torch.autocast(
        device_type=device_type, dtype=dtype, enabled=amp_enabled
    ):
        feature = model.extract_vision_feats(image={"abdomen_ct": tensor})
    if isinstance(feature, (tuple, list)):
        feature = feature[0]
    feature = feature.detach().float().reshape(feature.shape[0], -1).cpu().numpy()
    if feature.shape[1] != 1152:
        raise ValueError(f"Expected 1152 Pillar features, got {feature.shape}")
    return feature[0]


def run(config_path: str, max_cases: int | None = None, overwrite: bool | None = None) -> None:
    cfg = load_config(config_path)
    paths = project_paths(cfg)
    manifest_path = paths["output_dir"] / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"未找到 {manifest_path}; 请先运行 graft-validate")
    manifest = pd.read_csv(manifest_path, dtype={"case_id": str})
    required_paths = ["ct_path", "l3_path", "tumor_path"]
    if manifest[required_paths].isna().any().any():
        raise ValueError("manifest 中存在缺失文件，请先修复数据质检问题")
    if max_cases is not None:
        manifest = manifest.iloc[:max_cases].copy()

    device = torch.device(cfg["extraction"].get("device", "cuda:0"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但 torch.cuda.is_available() 为 False")
    dtype = _autocast_dtype(cfg["extraction"].get("autocast_dtype", "bfloat16"))
    overwrite = cfg["extraction"].get("overwrite", False) if overwrite is None else overwrite
    feature_dir = paths["output_dir"] / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(paths["model_dir"], device)
    pp = cfg["preprocess"]
    l3_labels = cfg["data"].get("l3_labels", [1, 2, 3])
    mask_feature_names: list[str] | None = None

    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        output_path = feature_dir / f"{row.case_id}.npz"
        if output_path.exists() and not overwrite:
            LOG.info("跳过已存在 %s/%s: %s", index, len(manifest), row.case_id)
            continue
        LOG.info("提取 %s/%s: %s", index, len(manifest), row.case_id)
        mask_features, current_names = extract_mask_features(
            row.ct_path, row.l3_path, row.tumor_path, l3_labels
        )
        if mask_feature_names is None:
            mask_feature_names = current_names
        elif mask_feature_names != current_names:
            raise RuntimeError("病例间 mask 特征定义不一致")

        hu = prepare_hu_volume(
            row.ct_path,
            pp["target_spacing_xyz"],
            pp["target_shape_dhw"],
            pp.get("pad_hu", -1024),
        )
        tensor = make_windowed_tensor(hu, pp["windows"], device=device, dtype=dtype)
        pillar_feature = _extract_pillar(model, tensor, dtype)
        np.savez_compressed(
            output_path,
            case_id=np.asarray(row.case_id),
            pillar_feature=pillar_feature.astype(np.float32),
            mask_features=mask_features,
            mask_feature_names=np.asarray(current_names),
            labels=np.asarray(
                [getattr(row, name) for name in cfg["data"]["label_columns"]],
                dtype=np.float32,
            ),
        )
        del tensor, hu
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate all cases in the selected manifest, including cache hits.
    case_ids, pillar_rows, mask_rows, label_rows = [], [], [], []
    for row in manifest.itertuples(index=False):
        path = feature_dir / f"{row.case_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"特征缺失: {path}")
        with np.load(path, allow_pickle=False) as item:
            case_ids.append(str(item["case_id"].item()))
            pillar_rows.append(item["pillar_feature"])
            mask_rows.append(item["mask_features"])
            label_rows.append(item["labels"])
            current_names = item["mask_feature_names"].astype(str).tolist()
            if mask_feature_names is None:
                mask_feature_names = current_names

    aggregate_path = paths["output_dir"] / "features.npz"
    np.savez_compressed(
        aggregate_path,
        case_ids=np.asarray(case_ids),
        pillar_features=np.stack(pillar_rows).astype(np.float32),
        mask_features=np.stack(mask_rows).astype(np.float32),
        mask_feature_names=np.asarray(mask_feature_names),
        labels=np.stack(label_rows).astype(np.float32),
        label_names=np.asarray(cfg["data"]["label_columns"]),
    )
    write_json(
        {
            "n_cases": len(case_ids),
            "pillar_feature_dim": int(np.stack(pillar_rows).shape[1]),
            "mask_feature_dim": int(np.stack(mask_rows).shape[1]),
            "windows": list(pp["windows"]),
            "target_spacing_xyz": list(pp["target_spacing_xyz"]),
            "target_shape_dhw": list(pp["target_shape_dhw"]),
        },
        paths["output_dir"] / "feature_metadata.json",
    )
    LOG.info("聚合特征已保存: %s", aggregate_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen Pillar-0 and mask features")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    run(args.config, args.max_cases, True if args.overwrite else None)


if __name__ == "__main__":
    main()
