"""Small end-to-end regression including exported missing-data rows."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import validate_protocol_v3 as protocol


class ProtocolIntegrationTests(unittest.TestCase):
    def test_export_and_missing_policy(self):
        cfg=json.loads(Path(__file__).with_name('protocol_config.json').read_text(encoding='utf-8'))
        cfg.update(subjects='1-3',states=['au0'],seeds=[0],strengths=[.25],
                   rigid_strengths=[1.],geometry_surface_samples=32)
        cfg['geometry_operations']={k:[v[0]] for k,v in cfg['geometry_operations'].items()}
        cfg['geometry_operations']['random_missing']=[.1]
        rng=np.random.default_rng(42)
        base=rng.normal(size=(40,3))
        shapes=[base+rng.normal(scale=.1,size=base.shape) for _ in range(3)]
        faces=np.arange(39).reshape(-1,3)

        class Reader:
            def __init__(self,*args):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read_sample(self,subject,state,modalities):
                i=int(subject[-3:])-1
                return SimpleNamespace(mesh=SimpleNamespace(vertices=shapes[i],faces=faces,path=f'{subject}/{state}.obj'))

        with tempfile.TemporaryDirectory() as tmp, patch.object(protocol,'ASTFaceReader',Reader), \
                patch.object(protocol,'plot'), contextlib.redirect_stdout(io.StringIO()):
            out=Path(tmp)
            result=protocol.run(Path(tmp),cfg,out)
            summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(result,0,summary['failures'])
            self.assertEqual(summary['rows'],11)
            self.assertEqual(summary['geometry_rows'],6)
            self.assertTrue(summary['all_protocol_checks_passed'])
            with (out/'scores.csv').open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            missing=[r for r in rows if r['kind']=='random_missing']
            self.assertEqual(len(missing),1)
            self.assertEqual(missing[0]['score'],'')
            self.assertEqual(missing[0]['change'],'')
            self.assertEqual(missing[0]['status'],'not_applicable_missing_coordinates')
            with np.load(out/'transforms.npz') as transforms:
                self.assertIn('au0_baseline_rotations',transforms.files)
            reference=json.loads((out/'reference.json').read_text(encoding='utf-8'))
            self.assertEqual(len(reference['sources']),3)
            self.assertTrue(reference['code_sha256'])


if __name__=='__main__':unittest.main(verbosity=2)
