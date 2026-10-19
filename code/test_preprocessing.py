import unittest
import numpy as np
from preprocessing import rigid_gpa,proper_rotation,apply_rigid_group,match_group_frame,score_status
from ecomp import covariance_spectrum


class GPATests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(17)
        self.x=rng.normal(size=(40,3))[None]+rng.normal(scale=.02,size=(5,40,3))
        self.ids=list('abcde')
    def test_independent_pose_and_common_pose(self):
        a=rigid_gpa(self.x)
        for mode in ('common','independent'):
            y,_,_=apply_rigid_group(self.x,self.ids,mode=mode,seed=3,strength=1)
            b=rigid_gpa(y)
            self.assertTrue(a.converged and b.converged)
            np.testing.assert_allclose(covariance_spectrum(a.aligned),covariance_spectrum(b.aligned),atol=1e-10)
            np.testing.assert_allclose(match_group_frame(b.aligned,a.mean),a.aligned,atol=1e-8)
    def test_preserves_distances_size_and_no_mirror(self):
        result=rigid_gpa(self.x)
        np.testing.assert_allclose(np.linalg.det(result.rotations),1,atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(np.diff(self.x,axis=1),axis=2),
                                   np.linalg.norm(np.diff(result.aligned,axis=1),axis=2),atol=1e-12)
        reflected=self.x[0]*[-1,1,1]
        a=self.x[0]-self.x[0].mean(0);b=reflected-reflected.mean(0)
        r=proper_rotation(a,b)
        self.assertAlmostEqual(np.linalg.det(r),1,12)
        self.assertGreater(np.linalg.norm(a@r-b),1.)
        scaled=self.x.copy();scaled[2]*=2
        result=rigid_gpa(scaled)
        np.testing.assert_allclose(np.linalg.norm(scaled-scaled.mean(1)[:,None],axis=(1,2)),
                                   np.linalg.norm(result.aligned,axis=(1,2)),atol=1e-12)
    def test_reconstruct_transforms_and_repeat(self):
        a=rigid_gpa(self.x);b=rigid_gpa(self.x)
        np.testing.assert_array_equal(a.aligned,b.aligned)
        np.testing.assert_allclose(np.einsum('nvi,nij->nvj',self.x-a.centers[:,None],a.rotations),a.aligned,atol=1e-12)
    def test_common_index_and_sample_order(self):
        a=rigid_gpa(self.x)
        b=rigid_gpa(self.x[:,::-1])
        c=rigid_gpa(self.x[::-1])
        for r in (b,c):np.testing.assert_allclose(covariance_spectrum(a.aligned),covariance_spectrum(r.aligned),atol=1e-9)
    def test_missing_rejected_and_explicit_status(self):
        x=self.x.copy();x[0,0]=np.nan
        with self.assertRaises(ValueError):rigid_gpa(x)
        self.assertEqual(score_status(x,'geometry_only'),'not_applicable_missing_coordinates')
        with self.assertRaises(ValueError):score_status(x,'fill_zero')
    def test_nonconvergence_reported(self):
        a=rigid_gpa(self.x,max_iterations=1,tolerance=1e-16)
        self.assertFalse(a.converged)
    def test_common_is_identical_transform(self):
        _,r,t=apply_rigid_group(self.x,self.ids,mode='common',seed=4,strength=1)
        for i in range(1,5):
            np.testing.assert_array_equal(r[0],r[i]);np.testing.assert_array_equal(t[0],t[i])


if __name__=='__main__':unittest.main(verbosity=2)
