import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

try:
    import torch

    from graft_pillar.train_cv import run
    from graft_pillar.predict import main as predict_main

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in this test environment")
class ClinicalTrainingIntegrationTests(unittest.TestCase):
    def test_small_fusion_training_writes_deployable_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "outputs"
            output.mkdir()
            n = 16
            case_ids = np.asarray([f"P{i:02d}" for i in range(n)])
            labels = np.asarray([[0, 0], [1, 1]] * (n // 2), dtype=np.float32)
            rng = np.random.default_rng(7)
            np.savez_compressed(
                output / "features.npz",
                case_ids=case_ids,
                pillar_features=rng.normal(size=(n, 4)).astype(np.float32),
                mask_features=rng.normal(size=(n, 2)).astype(np.float32),
                mask_feature_names=np.asarray(["m1", "m2"]),
                labels=labels,
                label_names=np.asarray(["recurrence", "complication"]),
            )
            clinical_path = root / "clinical.csv"
            pd.DataFrame(
                {
                    "Patient_UID": case_ids,
                    "Age (years)": np.linspace(40, 75, n),
                    "Charlson index": [0, 1] * (n // 2),
                    "BMI": [22.0, np.nan] * (n // 2),
                    "PNI (cont.)": np.linspace(40, 55, n),
                    "NLR (cont.)": np.linspace(1.0, 4.0, n),
                    "PLR (cont.)": np.linspace(80, 180, n),
                    "CONUT score": [0, 2] * (n // 2),
                    "SII": np.linspace(300, 900, n),
                    "CEA": np.linspace(1, 8, n),
                    "CA19-9": np.linspace(5, 60, n),
                    "CA72-4": np.linspace(1, 12, n),
                    "Sex": ["F", "M"] * (n // 2),
                    "ECOG": ["0", "1"] * (n // 2),
                    "cT": ["T2", "T3", "T4", "T2"] * (n // 4),
                    "cN": ["N0", "N1"] * (n // 2),
                    "cTNM stage": ["II", "III"] * (n // 2),
                    "Borrmann type": ["I", "III"] * (n // 2),
                    "Tumor location": ["body", "antrum"] * (n // 2),
                }
            ).to_csv(clinical_path, index=False)
            config = {
                "paths": {
                    "data_root": str(root),
                    "cohort_file": str(clinical_path),
                    "model_dir": str(root / "model"),
                    "output_dir": str(output),
                },
                "data": {
                    "id_column": "Patient_UID",
                    "label_columns": ["recurrence", "complication"],
                },
                "clinical": {
                    "enabled": True,
                },
                "training": {
                    "seed": 7,
                    "n_splits": 2,
                    "n_repeats": 1,
                    "validation_fraction": 0.5,
                    "head_type": "linear",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "max_epochs": 1,
                    "patience": 1,
                    "bootstrap_samples": 10,
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
            )

            run(str(config_path))

            training = output / "training"
            checkpoint = torch.load(
                training / "final_fusion_head.pt", map_location="cpu", weights_only=False
            )
            self.assertIn("clinical_preprocessor", checkpoint)
            self.assertEqual(checkpoint["image_feature_dim"], 6)
            self.assertGreater(checkpoint["input_dim"], 6)
            self.assertTrue((training / "linear_feature_weights.csv").exists())
            report = json.loads((training / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["n_cases"], n)
            self.assertEqual(report["model_inputs"]["clinical"]["n_cases"], n)
            prediction_path = root / "predictions.csv"
            with patch(
                "sys.argv",
                [
                    "graft-predict",
                    "--features",
                    str(output / "features.npz"),
                    "--clinical",
                    str(clinical_path),
                    "--checkpoint",
                    str(training / "final_fusion_head.pt"),
                    "--output",
                    str(prediction_path),
                ],
            ):
                predict_main()
            predictions = pd.read_csv(prediction_path)
            self.assertEqual(len(predictions), n)
            self.assertIn("prob_recurrence", predictions.columns)


if __name__ == "__main__":
    unittest.main()
