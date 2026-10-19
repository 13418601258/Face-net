"""Validate S3.5 complete grids, identities, analytical controls and hashes."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from stage35_definition import (STATES,SEEDS,COHORTS,SMOOTH_LEVELS,GLOBAL_LEVELS,LOCAL_LEVELS,
    RESTORE_LEVELS,SHRINK_LEVELS,SUBJECTS,ATOL,RTOL)


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='code/reports/stage3_s35_v2')
    out=Path(p.parse_args().input);checks=[]
    def check(name,condition):
        checks.append(dict(check=name,passed=bool(condition)))
        if not condition:raise AssertionError(name)
    rows={t:read(out/f'{t.lower()}_results.csv') for t in ['T2','T3','T4','T5']}
    expected={(c,s,seed,v) for c in COHORTS for s in STATES for seed in SEEDS for v in SMOOTH_LEVELS}
    actual=[(r['cohort'],r['state'],int(r['seed']),float(r['strength'])) for r in rows['T2']]
    check('T2 complete unique 441-condition grid',len(actual)==len(set(actual)) and set(actual)==expected)
    expected={(f,s,seed,v) for f in ['global_index','local_cycle','adjacent_swap'] for s in STATES for seed in SEEDS for v in (GLOBAL_LEVELS if f=='global_index' else LOCAL_LEVELS)}
    actual=[(r['family'],r['state'],int(r['seed']),float(r['strength'])) for r in rows['T3']]
    check('T3 complete unique 462-condition grid',len(actual)==len(set(actual)) and set(actual)==expected)
    expected={(s,seed,t) for s in STATES for seed in SEEDS for t in RESTORE_LEVELS}
    actual=[(r['state'],int(r['seed']),float(r['t'])) for r in rows['T4']]
    check('T4 complete unique 126-condition grid',len(actual)==len(set(actual)) and set(actual)==expected)
    expected={(s,'shrink',t) for s in STATES for t in SHRINK_LEVELS}|{(s,'copy_subject0',None) for s in STATES}
    actual=[(r['state'],r['variant'],float(r['t']) if r['t'] else None) for r in rows['T5']]
    check('T5 complete unique 56-condition grid',len(actual)==len(set(actual)) and set(actual)==expected)
    check('Every score is finite and K equals N minus one',all(np.isfinite(float(r['score'])) and int(r['K'])==int(r['N'])-1 for rr in rows.values() for r in rr))
    check('T3/T4 inverse-face certificates all pass',all(r['surface_identity_certified']=='True' and float(r['native_surface_distance'])==0 for t in ['T3','T4'] for r in rows[t]))
    check('T4 all endpoints exact and within numeric tolerance',all(r['exact_index_endpoint']=='True' and r['endpoint_within_tolerance']=='True' and float(r['changed_fraction'])==0 for r in rows['T4'] if float(r['t'])==1))
    check('T5 all analytic spectrum scores agree',all(float(r['analytic_error'])<=ATOL+RTOL*abs(float(r['analytic_score'])) for r in rows['T5']))
    geometry=read(out/'geometry_subjects.csv');groups=defaultdict(list)
    for r in geometry:
        key=(r['task'],r['state'],int(r['seed']),float(r['strength'])) if r['task']=='T2' else (r['task'],r['state'],r['variant'],r['t'])
        groups[key].append(r)
    check('4060 geometry rows cover 203 conditions and every identity',len(geometry)==4060 and len(groups)==203 and all(len(rr)==20 and {r['subject'] for r in rr}==set(SUBJECTS) for rr in groups.values()))
    check('Every sampled geometry row uses 2048 samples and seed zero',all(int(r['sample_count'])==2048 and int(r['geometry_seed'])==0 for r in geometry if int(r['sample_count'])!=0))
    check('No missing or nonfinite geometry measurements',all(all(np.isfinite(float(r[k])) for k in ['coverage','symmetric_mean','normal_reversal_rate','tolerance']) for r in geometry))
    for task in ['T2','T5']:
        for r in rows[task]:
            if task=='T2' and r['cohort']!='all20':continue
            key=(task,r['state'],int(r['seed']),float(r['strength'])) if task=='T2' else (task,r['state'],r['variant'],r['t'])
            rr=groups[key]
            g=1-np.mean([float(v['coverage']) for v in rr])+np.mean([float(v['symmetric_mean'])/float(v['tolerance']) for v in rr])+np.mean([float(v['normal_reversal_rate']) for v in rr])
            if not np.isclose(g,float(r['geometry_loss']),atol=1e-12,rtol=1e-12):raise AssertionError('geometry aggregation mismatch '+str(key))
    check('Every aggregate geometry loss matches its twenty subjects',True)
    replay=read(out/'replay_checks.csv')
    check('All 252 frozen old scores replay within tolerance',len(replay)==252 and all(r['passed']=='True' for r in replay))
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    weight=summary['T5']['max_lambda_geom_min']
    check('Conditional bound prevents every tested negative candidate beating baseline',all(float(r['delta_per_k'])+weight*float(r['geometry_loss'])>=-ATOL for r in rows['T5']))
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    check('All frozen output hashes match',all(sha(out/name)==digest for name,digest in manifest['outputs'].items()))
    check('All checkpoint hashes match',all(sha(out/'checkpoints'/name)==digest for name,digest in manifest['checkpoints'].items()))
    protocol=json.loads((out/'protocol_config_s35.json').read_text(encoding='utf-8'))
    check('Scoring and preprocessing source hashes match protocol',all(sha(Path('code')/name)==digest for name,digest in protocol['code_sha256'].items()))
    figure=json.loads((out/'figure_manifest.json').read_text(encoding='utf-8'))
    check('All nine figures exist in PNG and PDF',len(figure)==9 and all((out/(r['name']+'.'+ext)).stat().st_size>1000 for r in figure for ext in r['formats']))
    result=dict(passed=True,checks=checks,score_rows=sum(len(v) for v in rows.values()),geometry_rows=len(geometry),figures=len(figure))
    (out/'verification.json').write_text(json.dumps(result,indent=2)+chr(10),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
