"""Nontrivial checks of v2 sparse anchors, compact support, gate and face guard."""
import unittest
import numpy as np
from stage36_smooth import smooth_basis,anchor_terms,gate_weight,local_health,SmoothPatch


class SmoothTests(unittest.TestCase):
    def test_compact_boundary(self):
        uv=np.array([[0.,0.],[1.,0.],[1.2,0.]])
        b=smooth_basis(uv,1)
        self.assertGreater(np.linalg.norm(b[0]),0)
        np.testing.assert_array_equal(b[1:],0)
        eps=1e-4
        self.assertLess(np.linalg.norm(smooth_basis(np.array([[1-eps,0.]]),1))/eps,1e-9)

    def test_sparse_anchor_gradient(self):
        rng=np.random.default_rng(15);q=rng.normal(size=(3,6));b=smooth_basis(np.array([[.1,.2],[-.3,.1]]),1)
        _,g=anchor_terms(q,b);direction=rng.normal(size=q.shape);h=1e-6
        numeric=(anchor_terms(q+h*direction,b)[0]-anchor_terms(q-h*direction,b)[0])/(2*h)
        self.assertAlmostEqual(numeric,float(np.sum(g*direction)),places=7)

    def test_anchor_nullspace(self):
        b=smooth_basis(np.array([[0.,0.],[.4,.2]]),1)
        _,_,v=np.linalg.svd(b.reshape(4,6),full_matrices=True)
        q=v[-1:];value,gradient=anchor_terms(q,b)
        self.assertLess(value,1e-25);self.assertLess(np.linalg.norm(gradient),1e-12)
        self.assertGreater(np.linalg.norm(np.einsum('jak,nk->nja',smooth_basis(np.array([[.1,-.3]]),1),q)),.001)

    def test_gate_zero_is_constructed(self):
        self.assertEqual(gate_weight(0,'anchor_gated_e'),0)
        self.assertEqual(gate_weight(0,'anchor_fixed_e'),.1)
        self.assertAlmostEqual(gate_weight(1e-4,'anchor_gated_e'),.05)

    def test_face_quality_detects_flip_and_collapse(self):
        tri=np.array([[[[0.,0,0],[1,0,0],[0,1,0]]]])
        self.assertTrue(local_health(tri,tri,np.array([2.]))['valid'])
        self.assertFalse(local_health(tri,tri[:,:,[0,2,1]],np.array([2.]))['valid'])
        tiny=tri.copy();tiny[:,:,2]*=.1
        self.assertFalse(local_health(tri,tiny,np.array([2.]))['valid'])

    def test_original_connectivity_chart(self):
        uv=np.array([(x,y) for y in np.linspace(-1,1,11) for x in np.linspace(-1,1,11)])
        faces=[]
        for j in range(10):
            for i in range(10):
                a=j*11+i;faces.extend([[a,a+1,a+11],[a+1,a+12,a+11]])
        x=np.column_stack((uv,.05*(uv**2).sum(1)))
        base=np.array([x,x*np.array([1.01,.99,1.])])
        indices=np.flatnonzero(np.linalg.norm(uv,axis=1)<.85)
        m=SmoothPatch(base,np.array(faces),indices,60,1.)
        result=m.evaluate(np.zeros((2,6)))
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result['points'],base[:,m.indices],atol=1e-12)


if __name__=='__main__':unittest.main()
