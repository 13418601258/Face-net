"""Mathematical tests independent of the real-mesh diagnostic grid."""
import unittest
import numpy as np
from stage36_constraints import exact_sparse, observation, similarity


class ConstraintTests(unittest.TestCase):
    def test_underdetermined_keeps_unobserved_component(self):
        rng = np.random.default_rng(7)
        q = rng.normal(size=(20, 6))
        d = np.eye(6)[:4]
        y = rng.normal(size=(20, 4))
        result = exact_sparse(q, d, y)
        np.testing.assert_allclose(result[:, :4], y)
        np.testing.assert_allclose(result[:, 4:], q[:, 4:])

    def test_full_rank_recovers_noisy_target_even_at_zero_residual(self):
        rng = np.random.default_rng(8)
        d = rng.normal(size=(6, 6))
        q = rng.normal(size=(20, 6)); y = rng.normal(size=(20, 6))*.01
        result = exact_sparse(q, d, y)
        a, g = observation(result, d, y)
        self.assertLess(a, 1e-25)
        self.assertGreater(np.linalg.norm(result), .001)
        self.assertLess(np.max(np.abs(g)), 1e-12)

    def test_overdetermined_normal_equation_and_gradient(self):
        rng = np.random.default_rng(9)
        d = rng.normal(size=(12, 6)); q = rng.normal(size=(20, 6)); y = rng.normal(size=(20, 12))
        result = exact_sparse(q, d, y)
        a, gradient = observation(result, d, y)
        self.assertGreater(a, 0)
        self.assertLess(np.max(np.abs(gradient)), 1e-12)
        direction = rng.normal(size=q.shape)
        analytic = np.sum(observation(q, d, y)[1]*direction)
        h = 1e-6
        numeric = (observation(q+h*direction, d, y)[0]-observation(q-h*direction, d, y)[0])/(2*h)
        self.assertAlmostEqual(analytic, numeric, places=7)

    def test_similarity_uses_proper_rotation_and_holdout(self):
        rng = np.random.default_rng(10); source = rng.normal(size=(36, 3))
        angle = .7
        r = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0,0,1]])
        target = .012*source@r+np.array([.2, -.1, .5])
        fitted_r, s, t = similarity(source[::2], target[::2])
        np.testing.assert_allclose(s*source[1::2]@fitted_r+t, target[1::2], atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(fitted_r), 1.)


if __name__ == '__main__':
    unittest.main()
