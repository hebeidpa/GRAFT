from __future__ import annotations

import hashlib
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.exceptions import InconsistentVersionWarning

from graft_pillar.model import FusionMultiTaskHead


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "weights" / "final_fusion_head.pt"
EXPECTED_SHA256 = "06d1bd5c388a5aac7b9e28eee9a3e3e88d0620490fd119ad10f122716a44aa28"


class ReleasedCheckpointTests(unittest.TestCase):
    @staticmethod
    def _load_checkpoint():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            return torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    def test_checksum_schema_and_forward_pass(self):
        digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_SHA256)

        checkpoint = self._load_checkpoint()
        self.assertEqual(checkpoint["label_names"], ["recurrence", "complication"])
        self.assertEqual(checkpoint["image_feature_dim"], 1219)
        self.assertEqual(checkpoint["input_dim"], 1247)
        self.assertEqual(len(checkpoint["mask_feature_names"]), 67)
        self.assertEqual(len(checkpoint["clinical_feature_names"]), 28)

        model = FusionMultiTaskHead(
            input_dim=checkpoint["input_dim"],
            head_type=checkpoint["training_config"]["head_type"],
            hidden_dim=checkpoint["training_config"]["hidden_dim"],
            dropout=checkpoint["training_config"]["dropout"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.inference_mode():
            probability = torch.sigmoid(model(torch.zeros((2, 1247)))).numpy()
        self.assertEqual(probability.shape, (2, 2))
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(((probability >= 0.0) & (probability <= 1.0)).all())

    @unittest.skipUnless(
        sklearn.__version__ == "1.3.2",
        "released sklearn preprocessor requires the pinned scikit-learn 1.3.2",
    )
    def test_pinned_preprocessor_accepts_public_example(self):
        checkpoint = self._load_checkpoint()
        cohort = pd.read_csv(ROOT / "examples" / "synthetic" / "cohort.csv")
        clinical_columns = [
            *checkpoint["clinical_numerical_columns"],
            *checkpoint["clinical_categorical_columns"],
        ]
        clinical = cohort[clinical_columns].copy()
        for name in checkpoint["clinical_categorical_columns"]:
            clinical[name] = clinical[name].map(str)
        encoded = np.asarray(
            checkpoint["clinical_preprocessor"].transform(clinical), dtype=np.float32
        )
        self.assertEqual(encoded.shape, (2, 28))

        categorical = checkpoint["clinical_preprocessor"].named_transformers_[
            "categorical"
        ].named_steps["onehot"]
        for name, categories in zip(
            checkpoint["clinical_categorical_columns"], categorical.categories_
        ):
            self.assertTrue(set(clinical[name]).issubset(set(map(str, categories))))


if __name__ == "__main__":
    unittest.main()
