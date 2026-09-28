import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from graft_pillar.clinical import (
    ClinicalSchema,
    load_clinical_frame,
    make_clinical_preprocessor,
)


class ClinicalTests(unittest.TestCase):
    def _schema(self, path: Path) -> ClinicalSchema:
        return ClinicalSchema(
            file=path,
            id_column="Patient_UID",
            numerical_columns=("Age", "BMI"),
            categorical_columns=("Sex", "cT"),
        )

    def test_alignment_imputation_and_encoding(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clinical.csv"
            pd.DataFrame(
                {
                    "Patient_UID": ["A", "B", "C"],
                    "Age": [60, 70, 65],
                    "BMI": [22.0, np.nan, 24.0],
                    "Sex": ["M", "F", "M"],
                    "cT": ["T2", "T3", "T4"],
                }
            ).to_csv(path, index=False)
            schema = self._schema(path)
            frame, warnings = load_clinical_frame(schema, ["B", "A"])
            self.assertEqual(frame["Age"].tolist(), [70.0, 60.0])
            self.assertTrue(warnings)
            preprocessor = make_clinical_preprocessor(schema)
            values = np.asarray(preprocessor.fit_transform(frame), dtype=np.float32)
            self.assertEqual(values.shape[0], 2)
            self.assertTrue(np.isfinite(values).all())

    def test_missing_required_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clinical.csv"
            pd.DataFrame(
                {
                    "Patient_UID": ["A"],
                    "Age": [60],
                    "Sex": ["M"],
                    "cT": ["T2"],
                }
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "BMI"):
                load_clinical_frame(self._schema(path), ["A"])

    def test_non_numeric_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clinical.csv"
            pd.DataFrame(
                {
                    "Patient_UID": ["A"],
                    "Age": ["unknown"],
                    "BMI": [22.0],
                    "Sex": ["M"],
                    "cT": ["T2"],
                }
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Age"):
                load_clinical_frame(self._schema(path), ["A"])


if __name__ == "__main__":
    unittest.main()
