import unittest

import numpy as np

from graft_pillar.manifest import case_id_from_name
from graft_pillar.preprocess import CT_WINDOWS, center_crop_or_pad_dhw


class CoreTests(unittest.TestCase):
    def test_case_ids(self):
        self.assertEqual(case_id_from_name("Patient 001"), "Patient 001")
        self.assertEqual(case_id_from_name(" arbitrary.case-id "), "arbitrary.case-id")

    def test_crop_and_pad(self):
        source = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)
        result = center_crop_or_pad_dhw(source, (5, 3, 5), -1)
        self.assertEqual(result.shape, (5, 3, 5))
        self.assertTrue(np.all(result[0] == -1))
        self.assertTrue(np.all(result[-1] == -1))
        np.testing.assert_array_equal(result[1:4], source[:, 1:4, 1:6])

    def test_official_window_count(self):
        self.assertEqual(len(CT_WINDOWS), 10)
        self.assertEqual(CT_WINDOWS["abdomen"], (40.0, 400.0))


if __name__ == "__main__":
    unittest.main()
