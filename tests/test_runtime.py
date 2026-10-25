import tempfile
import unittest
from pathlib import Path

import pandas as pd

from graft_pillar.runtime import detect_columns, prepare_cohort


class RuntimeInputTests(unittest.TestCase):
    def _frame(self, root: Path, labelled: bool):
        for name in ("ct.nii.gz", "l3.nii.gz", "tumor.nii.gz"):
            (root / name).write_bytes(b"test")
        values = {
            "case_id": ["CASE_001"],
            "ct_path": ["ct.nii.gz"],
            "l3_mask_path": ["l3.nii.gz"],
            "tumor_mask_path": ["tumor.nii.gz"],
        }
        if labelled:
            values.update({"recurrence": [0], "complication": [1]})
        return pd.DataFrame(values)

    def test_unlabelled_inference_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frame = self._frame(root, labelled=False)
            columns = detect_columns(frame)
            prepared = prepare_cohort(frame, columns, root / "cohort.csv")
            self.assertFalse(columns.has_labels)
            self.assertTrue(Path(prepared.loc[0, columns.ct]).is_absolute())

    def test_labelled_evaluation_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frame = self._frame(root, labelled=True)
            columns = detect_columns(frame)
            prepared = prepare_cohort(frame, columns, root / "cohort.csv")
            self.assertTrue(columns.has_labels)
            self.assertEqual(prepared[columns.complication].tolist(), [1])

    def test_only_one_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frame = self._frame(root, labelled=False)
            frame["recurrence"] = 0
            columns = detect_columns(frame)
            with self.assertRaisesRegex(ValueError, "both recurrence and complication"):
                prepare_cohort(frame, columns, root / "cohort.csv")


if __name__ == "__main__":
    unittest.main()
