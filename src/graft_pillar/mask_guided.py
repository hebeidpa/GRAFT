#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mask_guided_pillar_roi_v1.py

Purpose
-------
Replace the old handcrafted 67-D tumor/L3 phenotype table with
mask-guided deep image representations from Pillar-0 intermediate features.

Data flow
---------
CT -> LoRA-adapted Pillar-0 -> global CT embedding + intermediate 3D feature maps
Tumor mask -> multi-scale alignment -> ROI pooling on Pillar feature maps -> tumor embedding
L3 mask (labels 1/2/3) -> multi-scale alignment -> region-aware ROI pooling -> L3 embedding

Important
---------
1) ONLY CT is fed into Pillar-0.
2) Tumor/L3 masks are NOT fed into Pillar-0.
3) Tumor/L3 masks must already be spatially aligned with the preprocessed CT volume.
4) No radiomics/handcrafted table is produced.
5) The module keeps the computation graph intact, so LoRA can still be trained.

This module is designed to wrap the Pillar-0 object used in the existing
train_pillar0_lora_trimodal_v2.py script.

Expected Pillar hierarchy:
    pillar.model.visual.atlas_models

The wrapper uses forward hooks to collect intermediate outputs from selected
Atlas stages. It supports common 5-D feature maps and [B,N,C] token outputs
when N (or N-1) is a perfect cube.

Recommended first use:
    wrapper = PillarMaskGuidedEncoder(pillar, emb_dim=128)
    wrapper.audit_structure()

Then run ONE sample forward. The returned dict includes feature_shapes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _iter_tensors(x: Any) -> Iterable[torch.Tensor]:
    """Recursively yield tensors from nested outputs."""
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _iter_tensors(v)
    elif isinstance(x, dict):
        for v in x.values():
            yield from _iter_tensors(v)


def _cube_side(n: int) -> int | None:
    if n <= 0:
        return None
    s = int(round(n ** (1.0 / 3.0)))
    for k in range(max(1, s - 2), s + 3):
        if k ** 3 == n:
            return k
    return None


def _to_5d_feature(
    x: torch.Tensor,
    batch_size: int | None = None,
) -> torch.Tensor | None:
    """
    Convert common Atlas-like outputs to [B,C,D,H,W].

    Supported:
      [B,C,D,H,W]
      [B,D,H,W,C]
      [B,N,C] when N or N-1 is a perfect cube
      [B,C,N] when N or N-1 is a perfect cube
    """
    if x.ndim == 5:
        # Heuristic for channel-first vs channel-last.
        common_c = {32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1152, 1536}
        if int(x.shape[1]) in common_c:
            return x
        if int(x.shape[-1]) in common_c:
            return x.permute(0, 4, 1, 2, 3).contiguous()

        # Fallback: channel dimension is usually larger than one spatial side
        # in deep stages.
        if x.shape[-1] > x.shape[1] and x.shape[-1] >= 32:
            return x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    if x.ndim == 3:
        b, a, c = x.shape

        # Pillar Atlas stages expose windowed tokens as
        # [B * num_windows, tokens_per_window, C]. Restore those windows
        # to one continuous 3-D feature map before mask-guided ROI pooling.
        if batch_size is not None and batch_size > 0 and b % batch_size == 0:
            windows_per_case = b // batch_size
            window_side = _cube_side(int(windows_per_case))
            token_side = _cube_side(int(a))
            if window_side is not None and token_side is not None and c >= 8:
                y = x.reshape(
                    batch_size,
                    window_side,
                    window_side,
                    window_side,
                    token_side,
                    token_side,
                    token_side,
                    c,
                )
                y = y.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
                side = window_side * token_side
                return y.reshape(batch_size, c, side, side, side)

        # Assume [B,N,C]
        for n, has_cls in ((a, False), (a - 1, True)):
            s = _cube_side(int(n))
            if s is not None and c >= 8:
                y = x[:, 1:, :] if has_cls else x
                return y.transpose(1, 2).reshape(b, c, s, s, s).contiguous()

        # Try [B,C,N]
        for n, has_cls in ((c, False), (c - 1, True)):
            s = _cube_side(int(n))
            if s is not None and a >= 8:
                y = x[:, :, 1:] if has_cls else x
                return y.reshape(b, a, s, s, s).contiguous()

    return None


def _choose_stage_map(
    stage_output: Any,
    batch_size: int | None = None,
) -> torch.Tensor:
    """
    From a nested Atlas stage output, select one usable 3D spatial map.
    Preference: the candidate with the largest spatial voxel count.
    """
    candidates: List[torch.Tensor] = []
    for t in _iter_tensors(stage_output):
        y = _to_5d_feature(t, batch_size=batch_size)
        if y is not None and y.ndim == 5 and min(y.shape[-3:]) >= 2:
            candidates.append(y)

    if not candidates:
        raw_shapes = [tuple(t.shape) for t in _iter_tensors(stage_output)]
        raise RuntimeError(
            "Could not convert this Atlas-stage output into a 3D feature map. "
            f"Raw tensor shapes: {raw_shapes}. "
            "Run audit_structure()/one-sample forward and inspect the stage outputs."
        )

    # Highest spatial resolution from this stage.
    candidates.sort(
        key=lambda z: int(z.shape[-3] * z.shape[-2] * z.shape[-1]),
        reverse=True,
    )
    return candidates[0]


def _ensure_mask_5d(mask: torch.Tensor) -> torch.Tensor:
    """Return mask as [B,1,D,H,W]."""
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError(
            f"Expected mask [B,1,D,H,W] or [B,D,H,W], got {tuple(mask.shape)}"
        )
    return mask


def align_binary_mask(
    mask: torch.Tensor,
    target_size: Sequence[int],
    preserve_small_roi: bool = False,
) -> torch.Tensor:
    """
    Align a binary mask to a feature-map resolution.

    Tumor:
        preserve_small_roi=True -> adaptive max pooling for downsampling,
        which reduces the chance that a tiny lesion disappears.

    L3 regions:
        preserve_small_roi=False -> nearest-neighbor resize is sufficient
        because the regions are usually much larger.
    """
    mask = _ensure_mask_5d(mask).float()
    target_size = tuple(int(v) for v in target_size)
    current = tuple(int(v) for v in mask.shape[-3:])

    if current == target_size:
        return (mask > 0.5).to(mask.dtype)

    is_downsample = all(t <= s for t, s in zip(target_size, current))

    if preserve_small_roi and is_downsample:
        aligned = F.adaptive_max_pool3d(mask, output_size=target_size)
    else:
        aligned = F.interpolate(mask, size=target_size, mode="nearest")

    return (aligned > 0.5).to(mask.dtype)


# ---------------------------------------------------------------------
# ROI pooling
# ---------------------------------------------------------------------

def masked_avg_max_pool(
    feat: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    feat: [B,C,D,H,W]
    mask: [B,1,D,H,W]

    Returns:
        [B,2C] = masked average + masked maximum

    Empty ROI fallback:
        average = 0, max = 0
    """
    if feat.ndim != 5:
        raise ValueError(f"feat must be 5D, got {tuple(feat.shape)}")

    mask = _ensure_mask_5d(mask).to(device=feat.device, dtype=feat.dtype)
    if tuple(mask.shape[-3:]) != tuple(feat.shape[-3:]):
        raise ValueError("Mask and feature spatial sizes must match before pooling.")

    denom = mask.sum(dim=(2, 3, 4)).clamp_min(1.0)  # [B,1]
    avg = (feat * mask).sum(dim=(2, 3, 4)) / denom   # [B,C]

    valid = mask.sum(dim=(2, 3, 4)) > 0              # [B,1]
    neg = torch.finfo(feat.dtype).min
    masked_feat = feat.masked_fill(mask <= 0, neg)
    mx = masked_feat.amax(dim=(2, 3, 4))             # [B,C]
    mx = torch.where(valid.expand_as(mx), mx, torch.zeros_like(mx))

    return torch.cat([avg, mx], dim=1)


class ROIScaleEncoder(nn.Module):
    """Pool one feature scale and project it to a fixed embedding dimension."""
    def __init__(self, out_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LazyLinear(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(self, feat: torch.Tensor, aligned_mask: torch.Tensor) -> torch.Tensor:
        pooled = masked_avg_max_pool(feat, aligned_mask)
        return self.proj(pooled)


class MultiScaleROIPooler(nn.Module):
    """
    ROI pooling at multiple Pillar stages + learned scale fusion.

    Each scale:
        aligned mask -> masked average/max pooling -> projection to out_dim

    Fusion:
        learned attention across scales -> one ROI embedding
    """
    def __init__(
        self,
        n_scales: int,
        out_dim: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.n_scales = int(n_scales)
        self.scale_encoders = nn.ModuleList(
            [ROIScaleEncoder(out_dim, dropout) for _ in range(self.n_scales)]
        )
        self.scale_score = nn.Linear(out_dim, 1)

    def forward(
        self,
        feature_maps: Sequence[torch.Tensor],
        fullres_mask: torch.Tensor,
        preserve_small_roi: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(feature_maps) != self.n_scales:
            raise ValueError(
                f"Expected {self.n_scales} scales, got {len(feature_maps)}"
            )

        zs = []
        for i, feat in enumerate(feature_maps):
            aligned = align_binary_mask(
                fullres_mask,
                feat.shape[-3:],
                preserve_small_roi=preserve_small_roi,
            )
            z = self.scale_encoders[i](feat, aligned)
            zs.append(z)

        z_stack = torch.stack(zs, dim=1)              # [B,S,E]
        score = self.scale_score(z_stack).squeeze(-1) # [B,S]
        alpha = torch.softmax(score, dim=1)
        fused = (z_stack * alpha.unsqueeze(-1)).sum(dim=1)
        return fused, alpha


class L3RegionFusion(nn.Module):
    """
    Fuse three L3 region embeddings.

    Trainable region embeddings retain label identity, so the fusion is not
    accidentally permutation-invariant across L3 labels 1/2/3.
    """
    def __init__(self, emb_dim: int = 128, n_regions: int = 3):
        super().__init__()
        self.n_regions = int(n_regions)
        self.region_id = nn.Parameter(torch.zeros(1, self.n_regions, emb_dim))
        nn.init.normal_(self.region_id, mean=0.0, std=0.02)

        self.score = nn.Sequential(
            nn.Linear(emb_dim, max(32, emb_dim // 2)),
            nn.GELU(),
            nn.Linear(max(32, emb_dim // 2), 1),
        )
        # Start from near-uniform region weighting.
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, region_embeddings: torch.Tensor):
        # [B,3,E]
        x = region_embeddings + self.region_id
        logits = self.score(x).squeeze(-1)
        alpha = torch.softmax(logits, dim=1)
        fused = (x * alpha.unsqueeze(-1)).sum(dim=1)
        return fused, alpha


# ---------------------------------------------------------------------
# Pillar hooks
# ---------------------------------------------------------------------

def _get_visual(pillar):
    if not hasattr(pillar, "model") or not hasattr(pillar.model, "visual"):
        raise RuntimeError("Expected pillar.model.visual")
    visual = pillar.model.visual
    if not hasattr(visual, "atlas_models"):
        raise RuntimeError("Expected pillar.model.visual.atlas_models")
    return visual


class AtlasStageCollector:
    """
    Capture intermediate Atlas-stage outputs without modifying Pillar source code.
    """
    def __init__(self, pillar, stage_indices: Sequence[int] = (0, 1, 2)):
        self.pillar = pillar
        self.visual = _get_visual(pillar)
        self.stage_indices = tuple(int(i) for i in stage_indices)
        self.outputs: Dict[int, Any] = {}
        self.handles = []

        n = len(self.visual.atlas_models)
        for i in self.stage_indices:
            if not (0 <= i < n):
                raise ValueError(f"stage index {i} out of range 0..{n-1}")
            h = self.visual.atlas_models[i].register_forward_hook(self._make_hook(i))
            self.handles.append(h)

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            self.outputs[idx] = output
        return hook

    def clear(self):
        self.outputs.clear()

    def feature_maps(self, batch_size: int | None = None) -> List[torch.Tensor]:
        missing = [i for i in self.stage_indices if i not in self.outputs]
        if missing:
            raise RuntimeError(
                f"Hooks did not receive outputs from Atlas stages {missing}. "
                "The local Pillar implementation may use a different path."
            )
        return [
            _choose_stage_map(self.outputs[i], batch_size=batch_size)
            for i in self.stage_indices
        ]

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------------
# Final mask-guided Pillar image encoder
# ---------------------------------------------------------------------

class PillarMaskGuidedEncoder(nn.Module):
    """
    Outputs:
        ct_embedding     [B,E]
        tumor_embedding  [B,E]
        l3_embedding     [B,E]

    No handcrafted image table is generated.
    Clinical variables should remain in the separate clinical branch.
    """
    def __init__(
        self,
        pillar,
        emb_dim: int = 128,
        stage_indices: Sequence[int] = (0, 1, 2),
        dropout: float = 0.10,
        l3_labels: Sequence[int] = (1, 2, 3),
    ):
        super().__init__()
        self.pillar = pillar
        self.emb_dim = int(emb_dim)
        self.stage_indices = tuple(stage_indices)
        self.l3_labels = tuple(int(v) for v in l3_labels)

        self.collector = AtlasStageCollector(
            pillar,
            stage_indices=self.stage_indices,
        )

        # Existing Pillar global embedding is 1152-D in the current project.
        self.ct_encoder = nn.Sequential(
            nn.Linear(1152, self.emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.emb_dim),
        )

        n_scales = len(self.stage_indices)

        # Tumor uses preserve-small-ROI downsampling.
        self.tumor_pooler = MultiScaleROIPooler(
            n_scales=n_scales,
            out_dim=self.emb_dim,
            dropout=dropout,
        )

        # The same L3 image encoder is shared across label 1/2/3.
        self.l3_pooler = MultiScaleROIPooler(
            n_scales=n_scales,
            out_dim=self.emb_dim,
            dropout=dropout,
        )
        self.l3_region_fusion = L3RegionFusion(
            emb_dim=self.emb_dim,
            n_regions=len(self.l3_labels),
        )

    def audit_structure(self):
        visual = _get_visual(self.pillar)
        print("Pillar visual:", type(visual).__name__)
        print("Number of Atlas stages:", len(visual.atlas_models))
        for i, stage in enumerate(visual.atlas_models):
            depth = len(stage.blocks) if hasattr(stage, "blocks") else "?"
            print(f"  stage {i}: {type(stage).__name__}, depth={depth}")
        print("Hooked stages:", self.stage_indices)

    def forward(
        self,
        ct: torch.Tensor,
        tumor_mask: torch.Tensor,
        l3_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor | List[Tuple[int, ...]]]:
        """
        ct:
            Pillar-ready CT tensor, e.g. [B,11,384,384,384]

        tumor_mask:
            spatially aligned tumor mask [B,1,384,384,384] or [B,384,384,384]

        l3_mask:
            spatially aligned integer label map with labels 1/2/3,
            [B,1,384,384,384] or [B,384,384,384]
        """
        tumor_mask = _ensure_mask_5d(tumor_mask)
        l3_mask = _ensure_mask_5d(l3_mask)

        if tuple(tumor_mask.shape[-3:]) != tuple(ct.shape[-3:]):
            raise ValueError(
                "Tumor mask is not spatially aligned with Pillar CT input: "
                f"CT={tuple(ct.shape[-3:])}, tumor={tuple(tumor_mask.shape[-3:])}"
            )
        if tuple(l3_mask.shape[-3:]) != tuple(ct.shape[-3:]):
            raise ValueError(
                "L3 mask is not spatially aligned with Pillar CT input: "
                f"CT={tuple(ct.shape[-3:])}, L3={tuple(l3_mask.shape[-3:])}"
            )

        self.collector.clear()

        # IMPORTANT: only CT enters Pillar-0.
        global_feat = self.pillar.extract_vision_feats(
            image={"abdomen_ct": ct}
        )
        if isinstance(global_feat, (tuple, list)):
            global_feat = global_feat[0]
        global_feat = global_feat.reshape(global_feat.shape[0], -1)

        if global_feat.shape[1] != 1152:
            raise RuntimeError(
                f"Expected 1152-D Pillar global feature, got {tuple(global_feat.shape)}"
            )

        # Intermediate Pillar feature maps captured by hooks.
        feature_maps = self.collector.feature_maps(batch_size=int(ct.shape[0]))

        # Global CT representation.
        ct_emb = self.ct_encoder(global_feat)

        # Tumor representation.
        tumor_binary = (tumor_mask > 0).to(ct.dtype)
        tumor_emb, tumor_scale_alpha = self.tumor_pooler(
            feature_maps,
            tumor_binary,
            preserve_small_roi=True,
        )

        # L3 region-specific representations.
        region_embs = []
        region_scale_alphas = []
        for label in self.l3_labels:
            region_mask = (l3_mask == label).to(ct.dtype)
            z, a = self.l3_pooler(
                feature_maps,
                region_mask,
                preserve_small_roi=False,
            )
            region_embs.append(z)
            region_scale_alphas.append(a)

        region_stack = torch.stack(region_embs, dim=1)  # [B,3,E]
        l3_emb, l3_region_alpha = self.l3_region_fusion(region_stack)

        return {
            "ct_embedding": ct_emb,
            "tumor_embedding": tumor_emb,
            "l3_embedding": l3_emb,
            "tumor_scale_weights": tumor_scale_alpha,
            "l3_region_weights": l3_region_alpha,
            "l3_region_scale_weights": torch.stack(region_scale_alphas, dim=1),
            "feature_shapes": [tuple(f.shape) for f in feature_maps],
        }


# ---------------------------------------------------------------------
# Integration example for the current TAGMF model
# ---------------------------------------------------------------------

class FourModalTaskAwareFusionExample(nn.Module):
    """
    Minimal example only.

    Modalities:
        1) global CT embedding
        2) tumor embedding
        3) L3 embedding
        4) clinical embedding

    Replace this with the existing TaskAwareGate implementation if desired.
    """
    def __init__(
        self,
        emb_dim: int = 128,
        clinical_dim: int = 0,
        hidden: int = 64,
        dropout: float = 0.20,
    ):
        super().__init__()
        if clinical_dim <= 0:
            raise ValueError("clinical_dim must be > 0")

        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.LayerNorm(emb_dim),
        )

        n_modalities = 4
        fusion_dim = n_modalities * emb_dim

        self.gate_r = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_modalities),
        )
        self.gate_c = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_modalities),
        )
        nn.init.zeros_(self.gate_r[-1].weight)
        nn.init.zeros_(self.gate_r[-1].bias)
        nn.init.zeros_(self.gate_c[-1].weight)
        nn.init.zeros_(self.gate_c[-1].bias)

        self.head_r = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.head_c = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _task_gate(hs, gate):
        concat = torch.cat(hs, dim=1)
        alpha = torch.softmax(gate(concat), dim=1)
        scale = alpha * len(hs)  # initial scale = 1
        gated = torch.cat(
            [h * scale[:, i:i+1] for i, h in enumerate(hs)],
            dim=1,
        )
        return gated, alpha

    def forward(
        self,
        ct_emb,
        tumor_emb,
        l3_emb,
        clinical,
    ):
        clinical_emb = self.clinical_encoder(clinical)
        hs = [ct_emb, tumor_emb, l3_emb, clinical_emb]

        r, alpha_r = self._task_gate(hs, self.gate_r)
        c, alpha_c = self._task_gate(hs, self.gate_c)

        return {
            "recurrence_logit": self.head_r(r).squeeze(1),
            "complication_logit": self.head_c(c).squeeze(1),
            "recurrence_modality_weights": alpha_r,
            "complication_modality_weights": alpha_c,
        }


# ---------------------------------------------------------------------
# Notes for Dataset integration
# ---------------------------------------------------------------------
#
# Your Dataset/DataLoader should now return:
#
#   ct_tensor      -> [11,384,384,384]  (existing Pillar preprocessing)
#   tumor_mask     -> [1,384,384,384]   (same spatial transform as CT)
#   l3_mask        -> [1,384,384,384]   (same spatial transform as CT)
#   clinical       -> structured clinical vector (the only tabular branch)
#
# DO NOT call extract_mask_features().
# DO NOT create the old 67-D phenotype vector.
# DO NOT fit StandardScaler for image-derived phenotype features.
#
# Training forward:
#
#   img = image_encoder(ct, tumor_mask, l3_mask)
#   out = fusion(
#       img["ct_embedding"],
#       img["tumor_embedding"],
#       img["l3_embedding"],
#       clinical,
#   )
#
