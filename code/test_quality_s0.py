"""Focused input-contract tests, independent of the real dataset."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_io import ASTFaceReader, Mesh, read_obj, rigid_transform, topology_hash
from run_quality_s0 import connectivity_stats


class S0Tests(unittest.TestCase):
    def test_obj_negative_indices_and_no_silent_triangulation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mesh.obj"
            header = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
            path.write_text(header + "f -4/1 -3/2 -2/3\n", encoding="utf-8")
            np.testing.assert_array_equal(read_obj(path).faces, [[0, 1, 2]])
            path.write_text(header + "f 1 2 3 4\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_obj(path)
            path.write_text(header + "f 0 2 3\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_obj(path)

    def test_hash_keeps_orientation_but_not_coordinates(self):
        vertices = np.eye(3)
        mesh = Mesh(vertices, np.array([[0, 1, 2]]))
        self.assertEqual(topology_hash(mesh), topology_hash(Mesh(vertices + 2, mesh.faces)))
        self.assertNotEqual(topology_hash(mesh), topology_hash(Mesh(vertices, np.array([[0, 2, 1]]))))

    def test_proper_rigid_fit_and_inverse_preserve_scale(self):
        source = np.array([[0., 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]])
        expected = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        target = source @ expected + [3, 4, 5]
        r, t = rigid_transform(source, target)
        np.testing.assert_allclose(source @ r + t, target, atol=1e-13)
        np.testing.assert_allclose((source @ r + t) @ r.T - t @ r.T, source, atol=1e-13)
        mirrored = source * [-1, 1, 1]
        r, t = rigid_transform(source, mirrored)
        self.assertAlmostEqual(np.linalg.det(r), 1.)
        self.assertGreater(np.linalg.norm(source @ r + t - mirrored), .1)

    def test_open_boundary_and_winding_are_distinct(self):
        vertices = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        stats = connectivity_stats(Mesh(vertices, np.array([[0, 1, 2], [0, 2, 3]])))
        self.assertEqual(stats["boundary_edges"], 4)
        self.assertEqual(stats["boundary_components"], 1)
        self.assertEqual(stats["nonmanifold_edges"], 0)
        self.assertEqual(stats["inconsistent_winding_edges"], 0)
        bad = connectivity_stats(Mesh(vertices, np.array([[0, 1, 2], [0, 3, 2]])))
        self.assertEqual(bad["inconsistent_winding_edges"], 1)

    def test_archives_rejected_and_alias_collisions_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "data.zip"
            archive.touch()
            with self.assertRaises(ValueError):
                ASTFaceReader(archive)
            meshes = root / "ast001" / "Standardized Meshes"
            meshes.mkdir(parents=True)
            (meshes / "ast001_disguest.obj").touch()
            (meshes / "ast001_disgust.obj").touch()
            with self.assertRaises(ValueError):
                ASTFaceReader(root).inventory(["ast001"])


if __name__ == "__main__":
    unittest.main()
