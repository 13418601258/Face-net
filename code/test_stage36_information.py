"""Test subspace algebra, correction prior and observable-only selection."""
import unittest
import numpy as np
from stage36_information import subspaces, smooth_terms, observation_each, selection_gain, select_policies


class InformationTests(unittest.TestCase):
    def test_exact_null_and_weak_are_distinct(self):
        d=np.diag([1.,.5,.1,.001,0.,0.])
        null,weak,ratios=subspaces(d,.05)
        self.assertAlmostEqual(np.trace(null),2)
        self.assertAlmostEqual(np.trace(weak),3)
        np.testing.assert_allclose(d@null,0,atol=1e-15)
        np.testing.assert_allclose(weak@weak,weak,atol=1e-15)
        np.testing.assert_allclose(weak.T,weak,atol=1e-15)
        np.testing.assert_allclose(weak@null,null,atol=1e-15)

    def test_rectangular_design_padding(self):
        d=np.eye(6)[:4]
        null,weak,_=subspaces(d)
        self.assertAlmostEqual(np.trace(null),2)
        np.testing.assert_allclose(null,weak)

    def test_full_rank_does_not_remove_weak_directions(self):
        d=np.diag([1.,1.,.5,.5,.0001,.0001])
        null,weak,_=subspaces(d)
        self.assertAlmostEqual(np.trace(null),0)
        self.assertAlmostEqual(np.trace(weak),2)

    def test_smooth_prior_is_about_input_not_zero(self):
        rng=np.random.default_rng(364);q_input=rng.normal(size=(20,6))
        a=rng.normal(size=(6,6));gram=a.T@a
        value,gradient=smooth_terms(q_input,q_input,gram)
        self.assertEqual(value,0);np.testing.assert_array_equal(gradient,0)
        self.assertGreater(smooth_terms(np.zeros_like(q_input),q_input,gram)[0],0)
        shift=rng.normal(size=q_input.shape);q=q_input+.2*shift
        v,g=smooth_terms(q,q_input,gram)
        vv,gg=smooth_terms(q+shift,q_input+shift,gram)
        self.assertAlmostEqual(v,vv);np.testing.assert_allclose(g,gg,atol=1e-13)
        direction=rng.normal(size=q.shape);h=1e-6
        numeric=(smooth_terms(q+h*direction,q_input,gram)[0]-smooth_terms(q-h*direction,q_input,gram)[0])/(2*h)
        self.assertAlmostEqual(numeric,float(np.sum(g*direction)),places=7)

    def test_selection_rejects_mean_noise_and_accepts_clear_gain(self):
        a=np.ones(20)
        self.assertFalse(selection_gain(a,a)[0])
        self.assertTrue(selection_gain(a,a*.9)[0])
        b=a.copy();b[0]-=.5;b[1]+=.49
        self.assertFalse(selection_gain(a,b)[0])

    def test_selector_cannot_read_dense_outcomes(self):
        def c(name,arm,loss,error):
            return dict(row=dict(candidate=name,arm=arm,step=.01,accepted=True,nrmse=error),selection=np.full(20,loss))
        candidates=[c('stay','stay',1.,0.),c('smooth','smooth',.8,10.),c('energy','energy',.5,100.)]
        first=select_policies(candidates,.01)
        self.assertEqual(first['augmented']['row']['candidate'],'energy')
        for candidate in candidates:candidate['row']['nrmse']=-999
        second=select_policies(candidates,.01)
        self.assertEqual(second['augmented']['row']['candidate'],'energy')

    def test_energy_needs_to_beat_selected_smooth(self):
        def c(name,arm,loss):return dict(row=dict(candidate=name,arm=arm,step=.01,accepted=True),selection=np.full(20,loss))
        chosen=select_policies([c('stay','stay',1.),c('smooth','smooth',.5),c('energy','energy',.6)],.01)
        self.assertEqual(chosen['augmented']['row']['candidate'],'smooth')
        self.assertEqual(chosen['energy']['row']['candidate'],'energy')


if __name__=='__main__':unittest.main()
