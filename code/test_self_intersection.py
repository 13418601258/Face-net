"""Small known-case and BVH-vs-brute-force tests for self-intersection."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from self_intersection import brute_force_intersections, detect_self_intersections  # noqa: E402


class SelfIntersectionTests(unittest.TestCase):
    def test_bvh_matches_brute_force_on_crossing_nonadjacent_triangles(self):
        vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0],
                             [.2, .2, -1], [.2, .2, 1], [.8, .8, 0]])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        brute = brute_force_intersections(vertices, faces)
        bvh = detect_self_intersections(vertices, faces)
        self.assertEqual(len(brute["proper_crossing"]), 1)
        self.assertEqual(len(bvh["proper_crossing"]), 1)

    def test_shared_edge_is_not_a_self_intersection(self):
        vertices = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        result = detect_self_intersections(vertices, faces)
        self.assertEqual(result["candidate_pairs"], 0)
        self.assertEqual(len(result["proper_crossing"]), 0)

    def test_coplanar_nonadjacent_overlap_is_separate_class(self):
        vertices = np.array([[0., 0, 0], [2, 0, 0], [0, 2, 0],
                             [.5, .5, 0], [1.5, .5, 0], [.5, 1.5, 0]])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        result = detect_self_intersections(vertices, faces)
        self.assertEqual(len(result["coplanar_overlap"]), 1)
        self.assertEqual(len(result["proper_crossing"]), 0)

    def test_disjoint_triangles_are_not_reported(self):
        vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0],
                             [3., 3, 3], [4, 3, 3], [3, 4, 3]])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        result = detect_self_intersections(vertices, faces)
        self.assertEqual(len(result["proper_crossing"]), 0)
        self.assertEqual(result["candidate_pairs"], 0)

    def test_bvh_candidate_and_classification_sets_match_brute_force(self):
        rng = np.random.default_rng(29)
        vertices, faces = [], []
        for _ in range(12):
            center = rng.uniform(-1, 1, size=3)
            local = rng.normal(scale=.2, size=(3, 3)) + center
            offset = len(vertices)
            vertices.extend(local)
            faces.append([offset, offset + 1, offset + 2])
        vertices, faces = np.asarray(vertices), np.asarray(faces)
        expected = brute_force_intersections(vertices, faces)
        actual = detect_self_intersections(vertices, faces)
        for category in ("proper_crossing", "coplanar_overlap", "touch_only"):
            self.assertEqual({tuple(pair) for pair in expected[category]},
                             {tuple(pair) for pair in actual[category]})


if __name__ == "__main__":
    unittest.main()
