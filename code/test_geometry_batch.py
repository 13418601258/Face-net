import unittest
import numpy as np
from geometry_batch import BinnedSurface,evaluate_batch_geometry,identical_geometry
from metrics import TriangleSurface,evaluate_geometry,_paired_triangle_distance


class BatchGeometryTests(unittest.TestCase):
    def test_mixed_radius_search_matches_bruteforce(self):
        rng=np.random.default_rng(12)
        triangles=rng.normal(size=(60,3,3))*np.geomspace(.001,20,60)[:,None,None]
        v=triangles.reshape(-1,3);f=np.arange(len(v)).reshape(-1,3)
        p=rng.normal(size=(90,3))*3
        expected=np.array([_paired_triangle_distance(np.repeat(x[None],len(f),axis=0),triangles).min() for x in p])
        np.testing.assert_allclose(BinnedSurface(v,f).distances(p),expected,atol=1e-12)

    def test_metrics_match_existing_including_missing(self):
        rng=np.random.default_rng(4);v=rng.normal(size=(60,3));f=np.arange(60).reshape(-1,3)
        for missing in (False,True):
            y=v+rng.normal(scale=.03,size=v.shape)
            if missing:y[:3]=np.nan
            args=dict(tolerance=.05,sample_count=90,seed=2)
            a=evaluate_geometry(v,f,y,f,**args);b=evaluate_batch_geometry(v,f,y,f,**args)
            for key,value in a.items():
                if isinstance(value,(float,int)):
                    self.assertAlmostEqual(value,b[key],places=12,msg=key)

    def test_identity_is_geometric_not_vertex_fraction(self):
        v=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]]);f=np.array([[0,1,2],[0,1,3]])
        r=identical_geometry(v,f,tolerance=.001,seed=0)
        self.assertEqual(r['sample_count'],0)
        self.assertEqual(r['coverage'],1)
        self.assertEqual(r['symmetric_mean'],0)
        self.assertEqual(r['reference_area'],1)


if __name__=='__main__':unittest.main(verbosity=2)
