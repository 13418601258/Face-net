"""Audit S3.6 grids, actual updates, surface constraints and reproducibility."""
import argparse,csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from stage36_gradient import STATES,SEEDS,LEVELS,STEPS,FD_STEPS,PRIMARY_STEP


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='code/reports/stage3_s36_v1');args=p.parse_args()
    out=Path(args.input);checks=[]
    def check(name,condition):
        checks.append(dict(check=name,passed=bool(condition)))
        if not condition:raise AssertionError(name)
    curves=read(out/'gradient_curves.csv');updates=read(out/'projected_updates.csv');subjects=read(out/'subject_updates.csv');numeric=read(out/'finite_difference_checks.csv')
    key=lambda r:(r['state'],int(r['seed']),float(r['t']))
    expected={(s,seed,t) for s in STATES for seed in SEEDS for t in LEVELS}
    check('All 231 path conditions unique and complete',len(curves)==231 and {key(r) for r in curves}==expected)
    check('All 693 step conditions unique and complete',len(updates)==693 and {(key(r),float(r['step_fraction'])) for r in updates}=={(k,step) for k in expected for step in STEPS})
    check('All 4620 identity updates unique and complete',len(subjects)==4620 and len({(key(r),r['subject']) for r in subjects})==4620)
    check('All 630 finite differences present and within frozen tolerance',len(numeric)==630 and all(r['passed']=='True' for r in numeric))
    check('Full native K=19 retained',all(int(r['N'])==20 and int(r['K'])==19 for r in curves))
    check('No nonfinite score or RMSE values',all(np.isfinite(float(r[k])) for r in curves for k in ['score','patch_nrmse','dE_dt','variance_ratio','local_variance_ratio']))
    check('All updated points lie inside their triangles',all(float(r['barycentric_min'])>=-1e-12 and float(r['barycentric_sum_max'])<=1+1e-12 and float(r['surface_certificate_max_distance'])<1e-10 for r in updates))
    check('Actual slide stays within 1 percent diagonal cap',max(float(r['max_displacement_over_diagonal']) for r in curves)<=.01+1e-12)
    by=defaultdict(list)
    for r in subjects:by[key(r)].append(r)
    check('Each condition contains all twenty identities',all({r['subject'] for r in rr}=={f'ast{i:03d}' for i in range(1,21)} for rr in by.values()))
    check('Group RMSE matches independent subject aggregation',all(np.isclose(float(u['patch_nrmse_after']),np.mean([float(r['patch_nrmse_after']) for r in by[key(u)]]),rtol=1e-10,atol=1e-13) for u in updates if float(u['step_fraction'])==PRIMARY_STEP))
    check('RMSE improvement flags match actual measurements',all((r['rmse_improves']=='True')==(float(r['patch_nrmse_after'])<float(r['patch_nrmse_before'])-1e-12) for r in updates))
    check('Energy descent flags match actual measurements',all((r['energy_decreases']=='True')==(float(r['score_after'])<float(r['score_before'])-1e-10) for r in updates))
    g=read(out/'geometry_subjects.csv');gkeys={(r['state'],int(r['seed']),float(r['t']),r['variant'],r['subject']) for r in g}
    gexpected={(s,seed,t,v,f'ast{i:03d}') for s in STATES for seed in SEEDS for t in [.1,.5,1.] for v in ['before','after'] for i in range(1,21)}
    check('All 2520 geometry records complete and unique',len(g)==2520 and gkeys==gexpected)
    check('Geometry uses 2048 samples and separate fixed sampling seed',all(int(r['sample_count'])==2048 and int(r['geometry_seed'])==0 for r in g))
    gates=read(out/'gate_conditions.csv');summary=json.loads((out/'summary.json').read_text())
    check('Gate decision equals all predeclared positive conditions',len(gates)==210 and summary['first_gate_pass']==all(r['passed']=='True' for r in gates))
    protocol=json.loads((out/'protocol_config_s36.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    check('Source code matches frozen hashes',all(sha(Path('code')/n)==h for n,h in protocol['code_sha256'].items()))
    check('All output hashes match',all(sha(out/n)==h for n,h in manifest['outputs'].items()))
    check('All checkpoints and patch hashes match',all(sha(out/n)==h for n,h in manifest['checkpoints'].items()))
    figures=json.loads((out/'figure_manifest.json').read_text())
    check('Five PNG and PDF figure pairs exist',len(figures)==5 and all((out/(f['name']+'.'+ext)).stat().st_size>1000 for f in figures for ext in ['png','pdf']))
    result=dict(passed=True,checks=checks,first_gate_pass=summary['first_gate_pass'],note='Integrity PASS is distinct from experimental gate PASS')
    (out/'verification.json').write_text(json.dumps(result,indent=2)+chr(10),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
