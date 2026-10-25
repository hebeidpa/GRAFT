import unittest

import torch

from graft_pillar.mask_guided import (
    _to_5d_feature,
    align_binary_mask,
    masked_avg_max_pool,
)


class MaskGuidedTests(unittest.TestCase):
    def test_atlas_window_tokens_restore_spatial_map(self):
        cases = [
            ((4096, 64, 384), (1, 384, 64, 64, 64)),
            ((64, 64, 384), (1, 384, 16, 16, 16)),
            ((1, 64, 384), (1, 384, 4, 4, 4)),
        ]
        for source_shape, expected_shape in cases:
            source = torch.empty(source_shape, device="meta")
            result = _to_5d_feature(source, batch_size=1)
            self.assertEqual(tuple(result.shape), expected_shape)

    def test_small_roi_survives_downsampling(self):
        mask = torch.zeros((1, 1, 8, 8, 8))
        mask[0, 0, 3, 3, 3] = 1
        result = align_binary_mask(mask, (2, 2, 2), preserve_small_roi=True)
        self.assertGreater(float(result.sum()), 0.0)

    def test_empty_roi_returns_finite_zero_pool(self):
        feature = torch.randn((1, 4, 3, 3, 3))
        mask = torch.zeros((1, 1, 3, 3, 3))
        pooled = masked_avg_max_pool(feature, mask)
        self.assertEqual(tuple(pooled.shape), (1, 8))
        self.assertTrue(torch.isfinite(pooled).all())
        self.assertTrue(torch.equal(pooled, torch.zeros_like(pooled)))


if __name__ == "__main__":
    unittest.main()
