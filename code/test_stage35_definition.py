"""Meaningful invariance, analytic shrink, threshold and protocol regression tests."""
import unittest
import numpy as np
from stage35_definition import (fixed_score, DELTA, graph_from_faces, disjoint_blocks,
    block_permutation, certify_permutation, restoration_permutation, conditional_bounds,
    curve_stats, geometry, ORIGINAL_LEVELS, SMOOTH_LEVELS, LOCAL_LEVELS)
from ecomp import covariance_spectrum,ecomp


class Stage35Tests(unittest.TestCase):
    def test_geometry_retains_perturbation_and_sampling_seeds(self):
        faces=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
        mesh=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
        x=np.repeat(mesh[None],20,axis=0)
        summary,rows=geometry(x,x+.01,faces,dict(task='T2',seed=2))
        self.assertEqual(len(rows),20)
        self.assertTrue(all(r['seed']==2 and r['geometry_seed']==0 for r in rows))
        self.assertGreater(summary['geometry_loss'],0)

    def test_analytic_shrink_retains_all_k_and_zero_floor(self):
        x=np.random.default_rng(6).normal(size=(5,30,3));mean=x.mean(axis=0);spec=covariance_spectrum(x)
        for t in [1,.6,.1,0]:
            y=mean+t*(x-mean)
            expected=ecomp(t*t*spec,n_components=4,reference_variance=12.,delta=DELTA)
            self.assertAlmostEqual(fixed_score(y,12.),expected,places=9)
        self.assertAlmostEqual(fixed_score(np.repeat(x[[0]],5,axis=0),12.),4*np.log(DELTA),places=9)

    def test_local_blocks_and_exact_surface_certificate(self):
        n=100;faces=np.array([[i,i+1,i+2] for i in range(n-2)])
        x=np.random.default_rng(4).normal(size=(n,3));graph=graph_from_faces(faces,n)
        for width in [2,3]:
            blocks=disjoint_blocks(graph,4,width,max_fraction=.3)
            previous=set()
            for level in [0,.03,.1,.2]:
                p=block_permutation(n,blocks,level);changed=set(np.flatnonzero(p!=np.arange(n)))
                self.assertTrue(previous<=changed);previous=changed
                self.assertTrue(certify_permutation(x,faces,p)['surface_identity_certified'])
            if width==2:
                for u,v in blocks:self.assertIn(v,graph[u])
        with self.assertRaises(ValueError):certify_permutation(x,faces,np.zeros(n,dtype=int))

    def test_restoration_nested_exact_endpoint(self):
        previous=set(range(100))
        for t in [0,.2,.4,.6,.8,1]:
            p=restoration_permutation(100,5,t);changed=set(np.flatnonzero(p!=np.arange(100)))
            self.assertTrue(changed<=previous);previous=changed
        np.testing.assert_array_equal(p,np.arange(100))

    def test_weight_bound_direction_for_both_conventions(self):
        lower,upper=conditional_bounds(-4.,2.)
        self.assertEqual(lower,2.);self.assertEqual(upper,.5)
        self.assertGreaterEqual(-4+lower*2,0)
        self.assertGreaterEqual(2+upper*(-4),0)
        self.assertLess(-4+.99*lower*2,0)
        self.assertEqual(conditional_bounds(-1,0),(None,0.))

    def test_negative_steps_are_not_positive_detection(self):
        rr=[dict(family='smooth',state='x',seed=s,strength=level,delta_per_k=-level*(1+.01*s)) for level in [1.,2.] for s in range(3)]
        stats=curve_stats(rr,[1.,2.],['family','state'])
        self.assertFalse(any(r['step_gt_sd'] or r['detected'] for r in stats))
        self.assertTrue(all(r['signed_step']<0 for r in stats))

    def test_paired_step_uses_seed_covariance(self):
        rr=[dict(family='index',state='x',seed=s,strength=level,delta_per_k=10*s+level) for level in [1.,2.] for s in range(3)]
        stats=curve_stats(rr,[1.,2.],['family','state'])
        self.assertEqual(stats[1]['paired_step_sd'],0.)
        self.assertTrue(stats[1]['paired_step_gt_3sd'])

    def test_original_failures_and_local_levels_preserved(self):
        self.assertTrue(set(ORIGINAL_LEVELS)<=set(SMOOTH_LEVELS))
        self.assertEqual(min(LOCAL_LEVELS),.001);self.assertEqual(max(LOCAL_LEVELS),.05)


if __name__=='__main__':unittest.main()
