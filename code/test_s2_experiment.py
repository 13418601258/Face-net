"""Unit tests for controlled S2 corruption definitions and label masks."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from s1_probes import local_jacobian  # noqa: E402
from s2_experiment import (apply_corruption, evaluate_mapping, face_alerts,
                          fast_local_jacobian, make_neighbors)  # noqa: E402


class S2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A small triangulated plane gives deterministic graph neighborhoods.
        cls.vertices = np.array([[x, y, 0.] for y in range(7) for x in range(7)], dtype=np.float64)
        faces = []
        for y in range(6):
            for x in range(6):
                a = y * 7 + x
                b, c, d = a + 1, a + 7, a + 8
                faces.extend(([a, b, d], [a, d, c]))
        cls.faces = np.asarray(faces, dtype=np.int64)
        cls.neighbors = make_neighbors(len(cls.vertices), cls.faces)

    def test_all_corruptions_are_seed_reproducible_and_masks_match_changes(self):
        for family in ("local_collapse", "local_fold", "local_permutation", "smooth_drift",
                       "duplicate_mapping", "coverage_missing"):
            first = apply_corruption(self.vertices, self.faces, self.neighbors, "ast001", family, 2, 42)
            second = apply_corruption(self.vertices, self.faces, self.neighbors, "ast001", family, 2, 42)
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            np.testing.assert_array_equal(first[2], np.any(first[1][self.faces], axis=1))
            self.assertGreater(np.sum(first[1]), 0)
            self.assertEqual(len(first), 6)

    def test_higher_patch_level_changes_more_vertices_for_patch_errors(self):
        for family in ("local_collapse", "local_fold", "local_permutation"):
            counts = [np.sum(apply_corruption(self.vertices, self.faces, self.neighbors, "ast001", family, level, 7)[1]) for level in (1, 2, 3)]
            self.assertLess(counts[0], counts[1])
            self.assertLess(counts[1], counts[2])

    def test_smooth_drift_severity_is_monotone(self):
        errors = [np.linalg.norm(apply_corruption(self.vertices, self.faces, self.neighbors, "ast002", "smooth_drift", level, 1)[0] - self.vertices) for level in (1, 2, 3)]
        self.assertLess(errors[0], errors[1])
        self.assertLess(errors[1], errors[2])

    def test_duplicate_mapping_reduces_unique_target_coverage(self):
        mapped, changed, affected, _, mapped_index, mapping_valid = apply_corruption(
            self.vertices, self.faces, self.neighbors, "ast001", "duplicate_mapping", 2, 19)
        self.assertTrue(np.all(mapping_valid))
        self.assertLess(len(np.unique(mapped_index)) / len(mapped_index), 1.0)
        np.testing.assert_array_equal(mapped, self.vertices[mapped_index])
        np.testing.assert_array_equal(affected, np.any(changed[self.faces], axis=1))
        metrics = fast_local_jacobian(self.vertices, mapped, self.faces)
        alerts = face_alerts(metrics, mapped, self.faces, mapping_valid, mapped_index)
        tp = np.sum(alerts["duplicate_mapping"] & affected)
        fp = np.sum(alerts["duplicate_mapping"] & ~affected)
        self.assertEqual(tp, np.sum(affected))
        self.assertEqual(fp, 0)

    def test_coverage_missing_is_localized_and_geometry_error_is_not_applicable(self):
        mapped, changed, affected, _, mapped_index, mapping_valid = apply_corruption(
            self.vertices, self.faces, self.neighbors, "ast001", "coverage_missing", 2, 23)
        row, localization, _, _, _ = evaluate_mapping(
            "ast001", "test", "development", self.vertices, self.vertices, mapped, self.faces,
            changed, affected, "corruption", "coverage_missing", 2,
            mapped_index=mapped_index, mapping_valid=mapping_valid)
        self.assertLess(row["mapping_coverage_fraction"], 1.0)
        self.assertFalse(row["correspondence_error_applicable"])
        self.assertIsNone(row["correspondence_vertex_error_mean_native"])
        by_probe = {item["probe"]: item for item in localization}
        self.assertEqual(by_probe["coverage_missing"]["affected_face_recall"], 1.0)
        self.assertTrue(by_probe["coverage_missing"]["probe_applicable"])
        self.assertFalse(by_probe["strain"]["probe_applicable"])
        self.assertIsNone(by_probe["strain"]["affected_face_recall"])
        self.assertTrue(by_probe["duplicate_mapping"]["probe_applicable"])

    def test_full_local_fold_reverses_face_orientation(self):
        mapped, changed, affected, _, _, _ = apply_corruption(
            self.vertices, self.faces, self.neighbors, "ast001", "local_fold", 3, 3)
        metrics = local_jacobian(self.vertices, mapped, self.faces)
        self.assertTrue(np.any((metrics["normal_dot"] < 0) & affected))

    def test_fast_jacobian_matches_s1_reference(self):
        rng = np.random.default_rng(81)
        target = self.vertices + rng.normal(scale=.08, size=self.vertices.shape)
        reference = local_jacobian(self.vertices, target, self.faces)
        fast = fast_local_jacobian(self.vertices, target, self.faces)
        for key in ("sigma_min", "sigma_max", "area_ratio", "normal_dot", "normal_angle_rad",
                    "edge_ratio_min", "edge_ratio_max", "reference_area", "target_area", "valid"):
            np.testing.assert_allclose(fast[key], reference[key], rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
