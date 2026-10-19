"""Protocol v2 mathematical tests; empirical direction is tested on real data."""
import itertools
import unittest
import numpy as np
from ecomp import covariance_spectrum, ecomp, ecomp_from_pca, delta_sensitivity
from model import fit_pca
from index_protocol import perturb_group, permutation, permute_mesh


class ECompTests(unittest.TestCase):
    def setUp(self):
        # Actual (3 samples, 2 vertices, 3 coordinates); only x varies.
        self.x=np.zeros((3,2,3))
        self.x[:,:,0]=[[0,0],[1,0],[0,1]]
        self.delta=1e-9
        self.reference=2/3

    def score(self,x):
        return ecomp(covariance_spectrum(x),n_components=2,
                     reference_variance=self.reference,delta=self.delta)

    def test_hand_covariance_and_score(self):
        z=self.x.reshape(3,6)-self.x.reshape(3,6).mean(axis=0)
        expected=np.zeros((6,6));expected[np.ix_([0,3],[0,3])]=[[1/3,-1/6],[-1/6,1/3]]
        m=fit_pca(self.x)
        np.testing.assert_allclose(z.T@z/2,expected,atol=1e-14)
        np.testing.assert_allclose(m.full_spectrum_,[.5,1/6],atol=1e-14)
        np.testing.assert_allclose(covariance_spectrum(self.x),[.5,1/6],atol=1e-14)
        self.assertAlmostEqual(self.score(self.x),np.log(.75+self.delta)+np.log(.25+self.delta),12)
        np.testing.assert_allclose((m.components_.T*m.eigenvalues_)@m.components_,expected,atol=1e-14)

    def test_rank_drop_keeps_zero_term(self):
        y=self.x.copy();y[1]=y[1,::-1]
        expected=np.log(.5+self.delta)+np.log(self.delta)
        self.assertAlmostEqual(self.score(y),expected,12)
        self.assertLess(self.score(y),self.score(self.x))

    def test_all_sample_orders(self):
        for order in itertools.permutations(range(3)):
            self.assertAlmostEqual(self.score(self.x[list(order)]),self.score(self.x),12)

    def test_common_permutation(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(8,30,3))
        y=perturb_group(x,[str(i) for i in range(8)],mode='common',strength=.5,seed=0)
        np.testing.assert_allclose(covariance_spectrum(x),covariance_spectrum(y),rtol=1e-12,atol=1e-12)

    def test_reconstruction_truncation_does_not_truncate_score(self):
        full=fit_pca(self.x);small=fit_pca(self.x,n_components=1)
        self.assertAlmostEqual(ecomp_from_pca(full,reference_variance=self.reference),
                               ecomp_from_pca(small,reference_variance=self.reference),12)
        self.assertGreater(small.reconstruction_error(self.x),full.reconstruction_error(self.x))

    def test_zero_variance_is_finite(self):
        x=np.zeros((3,2,3))
        self.assertAlmostEqual(self.score(x),2*np.log(self.delta),12)
        self.assertAlmostEqual(ecomp_from_pca(fit_pca(x),reference_variance=self.reference),self.score(x),12)
        np.testing.assert_array_equal(fit_pca(x).reconstruct(x),x)

    def test_fixed_scale_retains_variance_and_units_can_be_converted(self):
        self.assertGreater(self.score(self.x*2),self.score(self.x))
        converted=ecomp(covariance_spectrum(self.x*100),n_components=2,
                        reference_variance=self.reference*100**2,delta=self.delta)
        self.assertAlmostEqual(converted,self.score(self.x),12)

    def test_shared_rigid_transform(self):
        rng=np.random.default_rng(0);x=rng.normal(size=(5,20,3))
        rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        np.testing.assert_allclose(covariance_spectrum(x),covariance_spectrum(x@rotation+5),atol=1e-12)

    def test_delta_and_base(self):
        scores=delta_sensitivity([.5,1/6],[1e-12,1e-9,1e-3],n_components=2,reference_variance=self.reference)
        self.assertTrue((np.diff(scores)>0).all())
        self.assertAlmostEqual(ecomp([.5,1/6],n_components=2,reference_variance=self.reference,base=2),
                               self.score(self.x)/np.log(2),12)

    def test_input_guards(self):
        for kw in [dict(n_components=0,reference_variance=1),dict(n_components=1,reference_variance=1),
                   dict(n_components=2,reference_variance=0),dict(n_components=2,reference_variance=1,delta=0)]:
            with self.assertRaises(ValueError): ecomp([1,1],**kw)
        with self.assertRaises(ValueError): covariance_spectrum(np.full((3,2,3),np.nan))

    def test_independent_streams_reproducible_and_order_independent(self):
        x=np.tile(np.arange(60).reshape(1,20,3),(3,1,1)); ids=['a','b','c']
        args=dict(mode='independent',strength=.5,seed=2)
        y=perturb_group(x,ids,**args)
        self.assertFalse(np.array_equal(y[0],y[1]))
        np.testing.assert_array_equal(y,perturb_group(x,ids,**args))
        np.testing.assert_array_equal(y[::-1],perturb_group(x[::-1],ids[::-1],**args))
        z=perturb_group(x,ids,mode='single',target='b',strength=.5,seed=2)
        np.testing.assert_array_equal(z[[0,2]],x[[0,2]])

    def test_permutations_are_nested_and_preserve_surface(self):
        p=permutation(100,.1,7);q=permutation(100,.2,7);indexes=np.arange(100)
        self.assertEqual((p!=indexes).sum(),10)
        self.assertTrue(set(np.flatnonzero(p!=indexes))<=set(np.flatnonzero(q!=indexes)))
        v=np.arange(300).reshape(100,3); f=np.array([[0,1,2],[3,4,5]])
        vv,ff=permute_mesh(v,f,p)
        np.testing.assert_array_equal(vv[ff],v[f])


if __name__=='__main__':
    unittest.main(verbosity=2)
