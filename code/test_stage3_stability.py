import unittest
import numpy as np
from stage3_stability import spearman,monotonic,seed_table,identity_table,SCORED_PANELS


class StabilityTests(unittest.TestCase):
    def test_rank_and_monotonic_guards(self):
        self.assertAlmostEqual(spearman([1,2,4],[10,20,30]),1.)
        self.assertLess(spearman([1,2,4],[30,20,10]),-.9)
        self.assertIsNone(spearman([1,1],[2,3]))
        self.assertTrue(monotonic([0.,1.,2.]))
        self.assertFalse(monotonic([0.,2.,1.]))

    def test_seed_noise_and_identity_schema(self):
        rows=[]
        for cohort,n,k in [('base15',15,14),('new5',5,4)]:
            for kind,mode,levels in SCORED_PANELS:
                for strength in levels:
                    values=[strength*(1+seed*.01) for seed in range(3)]
                    for seed,v in enumerate(values):
                        rows.append(dict(cohort=cohort,N=n,K=k,reference_variance=1.,state='au0',kind=kind,mode=mode,seed=seed,strength=strength,
                            score=v,change=v,delta_per_k=v/k,status='scored',iterations=1))
        table=seed_table(rows)
        self.assertTrue(any(r['signal_gt_seed_noise'] for r in table if r['strength']==.1))
        self.assertTrue(all(r['signal_check_applicable'] for r in table if r['kind']=='index' and r['mode']=='independent' and r['step_from_previous_mean_per_K'] is not None))
        all20=[dict(r,cohort='all20',N=20,K=19) for r in rows if r['cohort']=='new5']
        identity=identity_table([r for r in rows if r['cohort']=='new5']+all20,[r for r in rows if r['cohort']=='base15'])
        self.assertEqual(len(identity),14)
        self.assertTrue(all('spearman_rho' in r for r in identity))


if __name__=='__main__':unittest.main(verbosity=2)
