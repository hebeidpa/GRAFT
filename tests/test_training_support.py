import unittest

import pandas as pd

from graft_pillar import training_support as support


class TrainingSupportTests(unittest.TestCase):
    def test_fixed_public_clinical_names_are_detected(self):
        columns = [
            "Age (years)", "Sex", "ECOG", "Charlson index", "BMI",
            "PNI (cont.)", "NLR (cont.)", "PLR (cont.)", "CONUT score",
            "SII", "cT", "cN", "cTNM stage", "Borrmann type",
            "Tumor location", "CEA", "CA19-9", "CA72-4",
        ]
        frame = pd.DataFrame([{name: 1 for name in columns}])
        self.assertEqual(support.select_clinical_columns(frame, None), columns)


if __name__ == "__main__":
    unittest.main()
