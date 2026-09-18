from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import SimpleITK as sitk


def _distribution(prefix: str, values: np.ndarray) -> tuple[list[str], list[float]]:
    names = [
        f"{prefix}_hu_mean",
        f"{prefix}_hu_std",
        f"{prefix}_hu_min",
        f"{prefix}_hu_p10",
        f"{prefix}_hu_p25",
        f"{prefix}_hu_median",
        f"{prefix}_hu_p75",
        f"{prefix}_hu_p90",
        f"{prefix}_hu_max",
    ]
    if values.size == 0:
        return names, [0.0] * len(names)
    q = np.percentile(values.astype(np.float32, copy=False), [10, 25, 50, 75, 90])
    vals = [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(q[0]),
        float(q[1]),
        float(q[2]),
        float(q[3]),
        float(q[4]),
        float(np.max(values)),
    ]
    return names, vals


def _mask_geometry(prefix: str, mask: np.ndarray, spacing_xyz: Sequence[float]) -> tuple[list[str], list[float]]:
    sx, sy, sz = (float(v) for v in spacing_xyz)
    voxel_count = int(np.count_nonzero(mask))
    names = [
        f"{prefix}_voxel_count",
        f"{prefix}_volume_cm3",
        f"{prefix}_extent_x_mm",
        f"{prefix}_extent_y_mm",
        f"{prefix}_extent_z_mm",
        f"{prefix}_active_slices",
        f"{prefix}_max_axial_area_cm2",
    ]
    if voxel_count == 0:
        return names, [0.0] * len(names)
    indices = np.argwhere(mask)  # z,y,x
    spans = indices.max(axis=0) - indices.min(axis=0) + 1
    per_slice = np.count_nonzero(mask, axis=(1, 2))
    values = [
        float(voxel_count),
        float(voxel_count * sx * sy * sz / 1000.0),
        float(spans[2] * sx),
        float(spans[1] * sy),
        float(spans[0] * sz),
        float(np.count_nonzero(per_slice)),
        float(per_slice.max() * sx * sy / 100.0),
    ]
    return names, values


def extract_mask_features(
    ct_path: str,
    l3_path: str,
    tumor_path: str,
    l3_labels: Iterable[int] = (1, 2, 3),
) -> tuple[np.ndarray, list[str]]:
    ct_img = sitk.ReadImage(str(ct_path))
    l3_img = sitk.ReadImage(str(l3_path))
    tumor_img = sitk.ReadImage(str(tumor_path))
    if ct_img.GetSize() != l3_img.GetSize() or ct_img.GetSize() != tumor_img.GetSize():
        raise ValueError("CT/L3/tumor shape mismatch; run graft-validate first")

    ct = sitk.GetArrayFromImage(ct_img).astype(np.float32, copy=False)
    l3 = sitk.GetArrayFromImage(l3_img)
    tumor = sitk.GetArrayFromImage(tumor_img) > 0
    spacing = ct_img.GetSpacing()

    names: list[str] = []
    values: list[float] = []
    n, v = _mask_geometry("tumor", tumor, spacing)
    names.extend(n)
    values.extend(v)
    n, v = _distribution("tumor", ct[tumor])
    names.extend(n)
    values.extend(v)

    total_l3 = np.count_nonzero(l3)
    for label in l3_labels:
        mask = l3 == int(label)
        prefix = f"l3_label{int(label)}"
        n, v = _mask_geometry(prefix, mask, spacing)
        names.extend(n)
        values.extend(v)
        n, v = _distribution(prefix, ct[mask])
        names.extend(n)
        values.extend(v)
        names.append(f"{prefix}_fraction_of_l3")
        values.append(float(np.count_nonzero(mask) / total_l3) if total_l3 else 0.0)

    return np.asarray(values, dtype=np.float32), names
