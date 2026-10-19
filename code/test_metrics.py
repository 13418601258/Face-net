"""Analytic surface, holes, orientation and degenerate-input tests."""
import unittest
import numpy as np
from metrics import TriangleSurface,evaluate_geometry,_paired_triangle_distance
from index_protocol import permutation,permute_mesh


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.v=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]])
        self.f=np.array([[0,1,2],[0,2,3]])
    def metric(self,v,f=None,**kw):
        return evaluate_geometry(self.v,self.f,v,self.f if f is None else f,
                                 tolerance=.01,sample_count=2048,seed=0,**kw)
    def test_interior_edge_and_vertex_distance(self):
        surface=TriangleSurface(self.v[:3],np.array([[0,1,2]]))
        d=surface.distances([[.75,.25,2],[2,0,0],[-1,-1,0]])
        np.testing.assert_allclose(d,[2,1,np.sqrt(2)],atol=1e-12)
    def test_identical_and_permuted_surface(self):
        a=self.metric(self.v)
        v,f=permute_mesh(self.v,self.f,permutation(4,1,0));b=self.metric(v,f)
        self.assertEqual(a['coverage'],1);self.assertEqual(b['coverage'],1)
        self.assertLess(b['symmetric_mean'],1e-12);self.assertEqual(b['normal_reversals'],0)
    def test_parallel_translation_distance(self):
        r=self.metric(self.v+[0,0,.2])
        self.assertAlmostEqual(r['symmetric_mean'],.2,12)
        self.assertEqual(r['coverage'],0)
    def test_hole_coverage_and_mask(self):
        v=self.v.copy();v[1]=np.nan
        r=self.metric(v,valid_mask=np.array([True,False,True,True]))
        self.assertEqual(r['usable_faces'],1);self.assertEqual(r['valid_vertex_fraction'],.75)
        self.assertTrue(.45<r['coverage']<.6)
        self.assertGreater(r['reference_to_candidate_mean'],.05)
        self.assertLess(r['candidate_to_reference_mean'],1e-12)
    def test_reversed_winding_is_detected(self):
        f=self.f.copy();f[0]=f[0,::-1]
        r=self.metric(self.v,f)
        self.assertEqual(r['normal_reversals'],1);self.assertEqual(r['coverage'],1)
    def test_empty_and_degenerate_surface(self):
        r=self.metric(np.zeros_like(self.v))
        self.assertEqual(r['degenerate_faces'],2);self.assertEqual(r['coverage'],0)
        self.assertIsNone(r['symmetric_mean']);self.assertIsNone(r['normal_reversal_rate'])
    def test_global_rigid_frame_invariance(self):
        v=self.v+[0,0,.2];R=np.array([[0,0,1],[0,1,0],[-1,0,0]])
        a=self.metric(v)
        b=evaluate_geometry(self.v@R+10,self.f,v@R+10,self.f,tolerance=.01,sample_count=2048,seed=0)
        self.assertAlmostEqual(a['symmetric_mean'],b['symmetric_mean'],12)
    def test_acceleration_matches_all_triangles(self):
        rng=np.random.default_rng(5);v=rng.normal(size=(60,3));f=np.arange(60).reshape(-1,3)
        points=rng.normal(size=(12,3));surf=TriangleSurface(v,f)
        brute=np.array([_paired_triangle_distance(np.tile(p,(len(f),1)),v[f]).min() for p in points])
        np.testing.assert_allclose(surf.distances(points),brute,atol=1e-12)
    def test_input_guards(self):
        with self.assertRaises(ValueError):self.metric(self.v,valid_mask=np.ones(2,dtype=bool))
        with self.assertRaises(ValueError):TriangleSurface(self.v,np.array([[0,1,6]]))


if __name__=='__main__':
    unittest.main(verbosity=2)
