import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import validate_protocol_v3 as protocol
import stage3_analysis as stage3


class Stage3Tests(unittest.TestCase):
    def test_replay_join_missing_and_seed_aggregation(self):
        cfg=json.loads(Path(__file__).with_name('protocol_config.json').read_text(encoding='utf-8'))
        cfg.update(subjects='1-3',states=['au0'],strengths=[.25],rigid_strengths=[1.],geometry_surface_samples=32)
        cfg['geometry_operations']={k:[v[0]] for k,v in cfg['geometry_operations'].items()}
        cfg['geometry_operations']['random_missing']=[.1]
        rng=np.random.default_rng(42);base=rng.normal(size=(40,3))
        shapes=np.array([base+rng.normal(scale=.1,size=base.shape) for _ in range(3)])
        faces=np.arange(39).reshape(-1,3)
        class Reader:
            def __init__(self,*args):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read_sample(self,subject,state,modalities):
                return SimpleNamespace(mesh=SimpleNamespace(vertices=shapes[int(subject[-3:])-1],faces=faces,path=f'{subject}/{state}.obj'))
        with tempfile.TemporaryDirectory() as tmp,patch.object(protocol,'ASTFaceReader',Reader),patch.object(protocol,'plot'),contextlib.redirect_stdout(io.StringIO()),contextlib.ExitStack() as cleanup:
            def close_maps():
                for arrays in stage3.WORK.get('arrays',{}).values():
                    for a in arrays:a._mmap.close()
                stage3.WORK.clear()
            cleanup.callback(close_maps)
            out=Path(tmp);protocol.run(out,cfg,out)
            with np.load(out/'transforms.npz') as z:
                baseline=np.einsum('nvi,nij->nvj',shapes-z['au0_baseline_centers'][:,None],z['au0_baseline_rotations'])
            for name,array in [('raw',shapes),('baseline',baseline),('faces',faces)]:np.save(out/f'au0_{name}.npy',array)
            scores=stage3.read_csv(out/'scores.csv');tasks=stage3.build_tasks(scores,cfg)
            self.assertEqual(len(tasks),27)
            with self.assertRaises(ValueError):stage3.build_tasks(scores+scores[:1],cfg)
            with self.assertRaises(ValueError):stage3.build_tasks(scores[:-1],cfg)
            stage3.init_worker(out,out,cfg)
            results=[stage3.process_experiment(row) for row in tasks]
            rows=[stage3.aggregate(r) for r in results]
            self.assertTrue(all(r['geometry_subject_count']==3 for r in rows))
            self.assertEqual(sum(r['kind']=='baseline' for r in rows),3)
            for result in results:
                row=stage3.aggregate(result)
                self.assertAlmostEqual(row['coverage_mean'],np.mean([d['coverage'] for d in result['geometry']]))
                if row['kind']=='random_missing':self.assertEqual(row['score'],'')
                if row['kind']=='index':
                    self.assertEqual(row['symmetric_mean_max'],0)
                    self.assertEqual(row['coverage_min'],1)
                    self.assertTrue(all(d['sample_count']==0 for d in result['geometry']))
            curves=stage3.make_curve_table(rows,cfg)
            missing=[r for r in curves if r['kind']=='random_missing' and r['metric']=='score']
            self.assertTrue(all(r['mean'] is None and r['n_seeds']==0 for r in missing))
            self.assertTrue(all(r['n_seeds']==3 for r in curves if r['metric']=='coverage_mean'))


if __name__=='__main__':unittest.main(verbosity=2)
