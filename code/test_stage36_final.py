"""Mathematical checks for final joint-spectrum and observation-only selection."""
import unittest
import numpy as np
from stage36_final import JointEnergy,SingleEnergy,BiLaplacian,center_tensor,mode_grams,log_score,select_gain,select_candidates

class FinalTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(365);self.base=self.rng.normal(size=(7,20,12,3));self.indices=np.array([1,3,6,8])
    def test_grand_centering_does_not_imply_mode_rank_minus_one(self):
        g=mode_grams(center_tensor(self.base.reshape(7,20,-1)))
        self.assertEqual(np.linalg.matrix_rank(g[0]),20);self.assertEqual(np.linalg.matrix_rank(g[1]),7)
    def test_reduced_and_full_joint_energy(self):
        e=JointEnergy(self.base,self.indices);candidate=self.base.copy();candidate[:,:,self.indices]+=.02
        actual=e.evaluate(candidate[:,:,self.indices],False)
        full=sum(log_score(g,r,e.delta)[0] for g,r in zip(mode_grams(center_tensor(candidate.reshape(7,20,-1))),e.references))
        self.assertAlmostEqual(actual,full,places=12)
    def test_joint_gradient(self):
        e=JointEnergy(self.base,self.indices);p=self.base[:,:,self.indices]+self.rng.normal(size=(7,20,4,3))*.01
        _,g=e.evaluate(p);d=self.rng.normal(size=p.shape);h=1e-6
        numeric=(e.evaluate(p+h*d,False)-e.evaluate(p-h*d,False))/(2*h)
        self.assertAlmostEqual(numeric,float(np.sum(g*d)),places=7)
    def test_identity_common_permutation(self):
        e=JointEnergy(self.base,self.indices);p=self.rng.permutation(20)
        x=center_tensor(self.base.reshape(7,20,-1));a=mode_grams(x);b=mode_grams(x[:,p])
        before=sum(log_score(g,r,e.delta)[0] for g,r in zip(a,e.references))
        after=sum(log_score(g,r,e.delta)[0] for g,r in zip(b,e.references))
        self.assertAlmostEqual(before,after,places=12)
    def test_single_gradient(self):
        e=SingleEnergy(self.base,self.indices,2.);p=self.base[:,:,self.indices]+.01;_,g=e.evaluate(p)
        d=self.rng.normal(size=p.shape);h=1e-6
        numeric=(e.evaluate(p+h*d,False)-e.evaluate(p-h*d,False))/(2*h)
        self.assertAlmostEqual(numeric,float(np.sum(g*d)),places=6)
    def test_bilaplacian_gradient_and_not_truth_centered(self):
        base=self.rng.normal(size=(3,4,3));faces=np.array([[0,1,2],[0,2,3]]);ix=np.array([1,2])
        e=BiLaplacian(base,faces,ix);p=base[:,ix]+.1;value,g=e.evaluate(p);d=self.rng.normal(size=p.shape);h=1e-6
        numeric=(e.evaluate(p+h*d,False)-e.evaluate(p-h*d,False))/(2*h)
        self.assertAlmostEqual(numeric,float(np.sum(g*d)),places=8)
        self.assertGreater(np.linalg.norm(e.evaluate(base[:,ix])[1]),0)
    def test_identity_level_selection_not_140_independent(self):
        reference=np.ones(140);candidate=reference.copy();candidate.reshape(7,20)[:,0]-=.2
        self.assertFalse(select_gain(reference,candidate)[0])
        self.assertTrue(select_gain(reference,reference*.9)[0])
    def test_selector_ignores_truth(self):
        def candidate(name,family,loss):return dict(row=dict(name=name,family=family,cap=.01,accepted=True,nrmse=999),selection=np.full(140,loss))
        pool=[candidate('stay','stay',1),candidate('reg','regularizer',.8),candidate('j','joint',.6)]
        self.assertEqual(select_candidates(pool,.01)['joint']['row']['name'],'j')
        pool[2]['row']['nrmse']=1e10
        self.assertEqual(select_candidates(pool,.01)['joint']['row']['name'],'j')
    def test_rank_deficient_spectrum_keeps_zeros(self):
        base=np.tile(self.base[:1,:1],(7,20,1,1))+np.arange(20)[None,:,None,None]
        e=JointEnergy(base,self.indices)
        self.assertTrue(np.isfinite(e.evaluate(base[:,:,self.indices],False)))
        self.assertEqual([d['terms'] for d in e.diagnostics()],[20,7])

if __name__=='__main__':unittest.main()
