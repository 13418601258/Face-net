"""Unit tests for S1 surface Jacobian and probe semantics."""

import sys
import unittest

import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from s1_probes import local_jacobian, probe  # noqa: E402


class S1Tests(unittest.TestCase):
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    reference = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])

    def test_rigid_translation_has_unit_jacobian_and_zero_normal_change(self):
        target = self.reference + [3., -2., 8.]
        metrics = local_jacobian(self.reference, target, self.faces)
        np.testing.assert_allclose(metrics["sigma_min"], [1.], atol=1e-12)
        np.testing.assert_allclose(metrics["sigma_max"], [1.], atol=1e-12)
        np.testing.assert_allclose(metrics["area_ratio"], [1.], atol=1e-12)
        np.testing.assert_allclose(metrics["normal_dot"], [1.], atol=1e-12)

    def test_in_plane_scale_is_detected(self):
        target = self.reference.copy()
        target[1] *= 2
        target[2] *= 3
        metrics = local_jacobian(self.reference, target, self.faces)
        np.testing.assert_allclose(np.sort(metrics["sigma_min"]), [2.], atol=1e-12)
        np.testing.assert_allclose(np.sort(metrics["sigma_max"]), [3.], atol=1e-12)
        np.testing.assert_allclose(metrics["area_ratio"], [6.], atol=1e-12)

    def test_bending_is_normal_change_not_fold_label(self):
        target = self.reference.copy()
        target[2] = [0., 0., 1.]
        metrics = local_jacobian(self.reference, target, self.faces)
        self.assertGreater(metrics["normal_angle_rad"][0], 0)
        self.assertEqual(int(metrics["normal_reversal_faces"] if "normal_reversal_faces" in metrics else 0), 0)

    def test_degenerate_triangle_is_invalid(self):
        target = self.reference.copy()
        target[2] = target[1]
        metrics = local_jacobian(self.reference, target, self.faces)
        self.assertFalse(bool(metrics["valid"][0]))


if __name__ == "__main__":
    unittest.main()
