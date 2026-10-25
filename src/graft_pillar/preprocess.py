from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Sequence

import numpy as np


# Exact CT anatomical windows exposed by the official YalaLab RAVE implementation.
CT_WINDOWS = OrderedDict(
    [
        ("lung", (-600.0, 1500.0)),
        ("mediastinum", (50.0, 400.0)),
        ("abdomen", (40.0, 400.0)),
        ("liver", (80.0, 150.0)),
        ("bone", (400.0, 1800.0)),
        ("brain", (40.0, 80.0)),
        ("subdural", (75.0, 215.0)),
        ("stroke", (40.0, 40.0)),
        ("temporal_bone", (600.0, 2800.0)),
        ("soft_tissue", (50.0, 350.0)),
    ]
)


def resample_ct(
    image,
    target_spacing_xyz: Sequence[float],
    default_hu: float = -1024.0,
):
    """Match RAVE's spacing rule while preserving origin and direction."""
    import SimpleITK as sitk

    old_spacing = image.GetSpacing()
    old_size = image.GetSize()
    new_size = [
        int(round(old_size[i] * old_spacing[i] / float(target_spacing_xyz[i])))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(v) for v in target_spacing_xyz))
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(default_hu))
    return resampler.Execute(sitk.Cast(image, sitk.sitkFloat32))


def center_crop_or_pad_dhw(
    volume: np.ndarray,
    target_shape_dhw: Sequence[int],
    pad_value: float,
) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected (D,H,W), got {volume.shape}")
    target = tuple(int(v) for v in target_shape_dhw)
    out = np.full(target, pad_value, dtype=volume.dtype)

    src_slices = []
    dst_slices = []
    for src_len, dst_len in zip(volume.shape, target):
        copy_len = min(src_len, dst_len)
        src_start = max((src_len - copy_len) // 2, 0)
        dst_start = max((dst_len - copy_len) // 2, 0)
        src_slices.append(slice(src_start, src_start + copy_len))
        dst_slices.append(slice(dst_start, dst_start + copy_len))
    out[tuple(dst_slices)] = volume[tuple(src_slices)]
    return out


def prepare_hu_volume(
    ct_path: str,
    target_spacing_xyz: Sequence[float],
    target_shape_dhw: Sequence[int],
    pad_hu: float = -1024.0,
) -> np.ndarray:
    import SimpleITK as sitk

    image = sitk.ReadImage(str(ct_path))
    image = resample_ct(image, target_spacing_xyz, default_hu=pad_hu)
    volume = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)  # D,H,W
    return center_crop_or_pad_dhw(volume, target_shape_dhw, pad_hu)


def make_windowed_tensor(
    hu_volume: np.ndarray,
    windows: Iterable[str],
    device,
    dtype,
):
    """Create (1,C,D,H,W) directly on the target device to control host RAM."""
    import torch

    names = list(windows)
    unknown = [name for name in names if name != "minmax" and name not in CT_WINDOWS]
    if unknown:
        raise ValueError(f"Unknown CT windows: {unknown}")
    raw = torch.from_numpy(np.ascontiguousarray(hu_volume)).to(device=device, dtype=torch.float32)
    result = torch.empty((1, len(names), *raw.shape), device=device, dtype=dtype)
    raw_min = raw.min()
    raw_max = raw.max()
    for index, name in enumerate(names):
        if name == "minmax":
            channel = (raw - raw_min) / (raw_max - raw_min + 1e-8)
        else:
            center, width = CT_WINDOWS[name]
            lower = center - width / 2.0
            channel = (raw - lower) / width
        result[0, index].copy_(channel.clamp_(0.0, 1.0).to(dtype=dtype))
    del raw
    return result
