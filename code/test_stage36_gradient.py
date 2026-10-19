"""Independent numerical tests for S3.6 gradient and constrained updates."""
import unittest
import numpy as np
from stage36_gradient import (helmert,PatchEnergy,points,closest_uv,projected_step,make_patch,errors,variance)
from ecomp import ecomp,covariance_spectrum
from metrics import _paired_triangle_distance


class GradientTests(unittest.TestCase):
    def test_helmert_orthonormal_and_centered(self):
        for n in [3,5,20]:
            h=helmert(n)
            np.testing.assert_allclose(h.T@h,np.eye(n-1),atol=1e-14)
            np.testing.assert_allclose(h.T@np.ones(n),0,atol=1e-14)

    def test_full_spectrum_matches_original_including_zero_rank(self):
        random=np.random.default_rng(4)
        x=random.normal(size=(5,16,3))
        for y in [x,np.repeat(x[[0]],5,axis=0)]:
            energy=PatchEnergy(y,np.arange(16),3.)
            reference=ecomp(covariance_spectrum(y),n_components=4,reference_variance=3.)
            self.assertAlmostEqual(energy.evaluate(y),reference,places=7)

    def test_partial_patch_gradient_all_coordinates(self):
        rng=np.random.default_rng(3);base=rng.normal(size=(5,12,3));idx=np.array([2,5,9]);patch=base[:,idx]+rng.normal(size=(5,3,3))*.1
        energy=PatchEnergy(base,idx,7.);value,grad=energy.evaluate(patch,True)
        full=base.copy();full[:,idx]=patch
        self.assertAlmostEqual(value,ecomp(covariance_spectrum(full),n_components=4,reference_variance=7.),places=10)
        finite=np.empty_like(grad);eps=1e-5
        for index in np.ndindex(patch.shape):
            plus=patch.copy();minus=patch.copy();plus[index]+=eps;minus[index]-=eps
            finite[index]=(energy.evaluate(plus)-energy.evaluate(minus))/(2*eps)
        np.testing.assert_allclose(grad,finite,rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(grad.sum(axis=0),0,atol=1e-14)

    def test_closest_triangle_projection_oblique_metric(self):
        origin=np.zeros((2,3,3));edge=np.tile(np.array([[2.,.8],[0.,.3],[0.,0.]])[None,None],(2,3,1,1))
        patch=dict(origin=origin,edges=edge)
        query=np.array([[-1.,0,2],[3.,1,0],[.6,.1,1.]])[None].repeat(2,axis=0)
        uv=closest_uv(patch,query);p=points(patch,uv)
        tri=np.stack((origin,origin+edge[...,0],origin+edge[...,1]),axis=-2)
        expected=_paired_triangle_distance(query.reshape(-1,3),tri.reshape(-1,3,3))
        np.testing.assert_allclose(np.linalg.norm(p-query,axis=-1).ravel(),expected,atol=1e-12)
        self.assertTrue(np.all(uv>=0) and np.all(uv.sum(axis=-1)<=1+1e-14))

    def test_true_recovery_and_negative_gradient_sign_distinct(self):
        # A compactness descent can point away from known correspondence truth.
        origin=np.zeros((2,1,3));edge=np.tile(np.array([[1.,0.],[0.,1.],[0.,0.]])[None,None],(2,1,1,1))
        patch=dict(origin=origin,edges=edge,diagonals=np.ones(2));uv=np.full((2,1,2),.1)
        gradient=-np.ones((2,1,3));moved,_=projected_step(patch,uv,gradient,.03)
        self.assertGreater(errors(patch,points(patch,moved))[0],errors(patch,points(patch,uv))[0])
        good,_=projected_step(patch,uv,-gradient,.03)
        self.assertLess(errors(patch,points(patch,good))[0],errors(patch,points(patch,uv))[0])

    def test_covariance_shrink_derivative(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(5,10,3));mean=x.mean(axis=0);idx=np.arange(10)
        energy=PatchEnergy(x,idx,8.)
        y=mean+.4*(x-mean);_,g=energy.evaluate(y,True)
        dot=np.sum(g*(mean-y));self.assertLess(dot,0)
        self.assertLess(variance(y),variance(x))


if __name__=='__main__':unittest.main()
