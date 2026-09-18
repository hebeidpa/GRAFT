import tempfile
import unittest
from pathlib import Path

import pandas as pd

from graft_pillar.manifest import build_manifest


class ManifestTests(unittest.TestCase):
    def test_explicit_paths_allow_arbitrary_filenames_and_case_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cohort_path = root / "cohort.csv"
            pd.DataFrame(
                {
                    "case_id": ["case A", "unrestricted.case-02"],
                    "ct_path": ["images/anything.nii.gz", "other/raw.nii.gz"],
                    "l3_mask_path": ["masks/a.nii.gz", "other/l3.nii.gz"],
                    "tumor_mask_path": ["masks/b.nii.gz", "other/t.nii.gz"],
                    "recurrence": [0, 1],
                    "complication": [1, 0],
                }
            ).to_csv(cohort_path, index=False)
            cfg = {
                "paths": {
                    "data_root": str(root),
                    "cohort_file": str(cohort_path),
                    "model_dir": str(root / "model"),
                    "output_dir": str(root / "outputs"),
                },
                "data": {
                    "id_column": "case_id",
                    "ct_path_column": "ct_path",
                    "l3_path_column": "l3_mask_path",
                    "tumor_path_column": "tumor_mask_path",
                    "label_columns": ["recurrence", "complication"],
                },
            }
            manifest, _ = build_manifest(cfg)
            self.assertEqual(manifest["case_id"].tolist(), ["case A", "unrestricted.case-02"])
            self.assertEqual(
                Path(manifest.loc[0, "ct_path"]),
                (root / "images/anything.nii.gz").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
