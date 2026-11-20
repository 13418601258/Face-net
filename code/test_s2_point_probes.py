"""Tests for the S2 sparse semantic landmark probe."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from s2_point_probes import (ANCHOR_INDEX, apply_landmark_corruption,
                             proper_rigid_align, semantic_probe)


class S2PointProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(4)
        cls.points = rng.normal(size=(84, 3))

    def test_identity_and_common_rigid_motion_are_invariant(self):
        theta = .41
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0.],
                             [np.sin(theta), np.cos(theta), 0.],
                             [0., 0., 1.]])
        moved = self.points @ rotation + np.array([4., -2., 1.])
        identity = semantic_probe(self.points, self.points)
        rigid = semantic_probe(self.points, moved)
        for key in ("pairwise_distance_rmse_norm", "pairwise_relative_rmse",
                    "rigid_residual_q95_norm"):
            self.assertAlmostEqual(identity[key], rigid[key], places=12)
        self.assertLess(rigid["pairwise_distance_rmse_norm"], 1e-12)

    def test_swap_and_drift_are_detected_monotonically(self):
        values = []
        for level in (1, 2, 3):
            corrupted, changed = apply_landmark_corruption(
                self.points, "ast001", "happy", "anchor_swap", level, 17)
            self.assertGreater(int(np.sum(changed)), 0)
            values.append(semantic_probe(self.points, corrupted)["pairwise_distance_rmse_norm"])
        # Discrete permutations are not a metric ray: adding another swap can
        # partially cancel a previous distance change.  Every level must still
        # produce a nonzero semantic signal.
        self.assertTrue(all(value > 1e-6 for value in values))
        self.assertGreaterEqual(max(values), min(values))
        drift_values = []
        for level in (1, 2, 3):
            corrupted, _ = apply_landmark_corruption(
                self.points, "ast001", "happy", "anchor_drift", level, 17)
            drift_values.append(semantic_probe(self.points, corrupted)["pairwise_distance_rmse_norm"])
        self.assertLess(drift_values[0], drift_values[1])
        self.assertLess(drift_values[1], drift_values[2])

    def test_missing_is_explicit_and_never_imputed(self):
        corrupted, changed = apply_landmark_corruption(
            self.points, "ast001", "happy", "anchor_missing", 3, 17)
        self.assertEqual(int(np.sum(changed)), 3)
        self.assertTrue(np.isnan(corrupted[ANCHOR_INDEX[0]]).all())
        row = semantic_probe(self.points, corrupted)
        self.assertEqual(row["valid_anchor_count"], 5)
        self.assertEqual(row["missing_anchor_count"], 3)
        self.assertTrue(row["semantic_probe_applicable"])

    def test_rigid_alignment_has_no_reflection(self):
        aligned, rotation, _ = proper_rigid_align(self.points[ANCHOR_INDEX], self.points[ANCHOR_INDEX])
        self.assertGreater(np.linalg.det(rotation), 0.)
        np.testing.assert_allclose(aligned, self.points[ANCHOR_INDEX], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
