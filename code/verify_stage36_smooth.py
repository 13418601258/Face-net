"""Verify v2 trial completeness, sparse-anchor contracts and accepted geometry."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='code/reports/stage3_s36_v2_harmonic');args=p.parse_args()
    out=Path(args.input);checks=[]
    def check(name,condition):
        checks.append(dict(check=name,passed=bool(condition)))
        if not condition:raise AssertionError(name)
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    frozen=json.loads((out/'frozen_protocol.json').read_text(encoding='utf-8'))
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    cfg=frozen['protocol'];root=Path(__file__).resolve().parent.parent
    jobs=[json.loads(p.read_text(encoding='utf-8')) for p in (out/'checkpoints').glob('*.json')]
    expected_jobs={(s,k) for s in cfg['states'] for k in cfg['seeds']}
    check('All 21 chart attempts are present, including any rejected chart',len(jobs)==21 and {(j['state'],j['seed']) for j in jobs}==expected_jobs)
    valid=[j for j in jobs if j['status']=='complete']
    check('No chart rejection is silently dropped',len(valid)==summary['valid_charts'] and len(summary['invalid_charts'])+len(valid)==21)
    rows=read(out/'endpoints.csv');traces=read(out/'trajectories.csv');subjects=read(out/'subject_endpoints.csv');pairs=read(out/'paired_endpoints.csv')
    key=lambda r:(r['state'],int(r['seed']),float(r['t']),r['arm'])
    expected={(j['state'],j['seed'],t,a) for j in valid for t in cfg['levels'] for a in cfg['arms']}
    check('All four arms and five levels are complete and unique',len(rows)==len(expected) and {key(r) for r in rows}==expected)
    check('Twenty unique subject outcomes per endpoint',len(subjects)==len(rows)*20 and len({(key(r),r['subject']) for r in subjects})==len(subjects))
    check('Three pairwise comparisons per baseline condition',len(pairs)==len(valid)*len(cfg['levels'])*3)
    check('Finite nonnegative error and anchor results',all(np.isfinite(float(r[k])) and float(r[k])>=0 for r in rows for k in ['initial_nonanchor_nrmse','final_nonanchor_nrmse','initial_anchor','final_anchor']))
    check('Every accepted trajectory point passes affected-face guards',all(int(r['normal_reversals'])==0 and float(r['min_area_ratio'])>=.2 for r in traces))
    check('All endpoint displacements respect one percent cap',all(float(r['final_max_displacement_ratio'])<=.01+1e-12 for r in rows))
    check('All analytic surface-field derivative checks pass',all(v['passed'] for j in valid for v in j['finite_difference']))
    check('Two sparse anchors leave coefficient nullspace',all(j['anchor_rank']<6 for j in valid))
    check('Iteration budget exhaustion never counts as convergence',all(r['solver_converged']=='False' for r in rows if r['termination']=='max_iterations'))
    check('Gate is zero at true baseline without forced recovery',all(float(r['final_anchor'])==0 and float(r['final_nonanchor_nrmse'])<1e-12 for r in rows if r['arm']=='anchor_gated_e' and float(r['t'])==0))
    by_subject={}
    for r in subjects:by_subject.setdefault(key(r),[]).append(float(r['final_nrmse']))
    check('Group results equal independently aggregated subject means',all(np.isclose(float(r['final_nonanchor_nrmse']),np.mean(by_subject[key(r)]),rtol=1e-10,atol=1e-14) for r in rows))
    by_trace={}
    for r in traces:by_trace.setdefault(key(r),[]).append(r)
    check('Endpoint equals last recorded iterate',all(np.isclose(float(r['final_nonanchor_nrmse']),float(by_trace[key(r)][-1]['nonanchor_nrmse']),rtol=1e-10,atol=1e-14) for r in rows))
    check('No neural release from synthetic anchor experiments',summary['network_eligible'] is False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    check('Result hashes match',all(sha(out/n)==h for n,h in manifest['files'].items()))
    check('Code hashes match',all(sha(root/'code'/n)==h for n,h in manifest['code'].items()))
    provenance=frozen['provenance']
    check('Frozen model source matches',sha(root/'code/stage36_smooth.py')==provenance['code_sha256'] and sha(root/'code/stage36_gradient.py')==provenance['core_sha256'])
    check('Original cached surfaces and patches are unmodified',all(sha(root/n)==h for group in ['source_sha256','patch_sha256'] for n,h in provenance[group].items()))
    check('All three PNG and PDF plot pairs exist',all((out/(n+'.'+e)).stat().st_size>1000 for n in ['v2_endpoints','v2_closed_loop','v2_paired_gain'] for e in ['png','pdf']))
    if (out/'diagnostic_manifest.json').exists():
        dm=json.loads((out/'diagnostic_manifest.json').read_text(encoding='utf-8'))
        diagnosis=json.loads((out/'mechanism_diagnosis.json').read_text(encoding='utf-8'))
        certificate=read(out/'anchor_nullspace_certificate.csv');probes=read(out/'nullspace_energy_probes.csv')
        check('Diagnostic source and output hashes match',sha(out/'manifest.json')==dm['source_manifest_sha256'] and sha(root/'code/analyze_stage36_smooth.py')==dm['code_sha256'] and all(sha(out/n)==h for n,h in dm['outputs'].items()))
        check('Nullspace certificates cover every valid chart once',len(certificate)==len(valid) and {(r['state'],int(r['seed'])) for r in certificate}=={(j['state'],j['seed']) for j in valid})
        check('Both post-hoc probes cover each valid chart',len(probes)==2*len(valid) and len({(r['state'],r['seed'],r['start']) for r in probes})==len(probes))
        check('Accepted probes preserve sparse anchors and decrease energy',all(float(r['anchor_after'])<1e-20 and float(r['energy_after'])<float(r['energy_before']) for r in probes if r['accepted']=='True'))
        for start in ['truth','exact_anchor_minimizer']:
            rr=[r for r in probes if r['start']==start and r['accepted']=='True'];counts=diagnosis['nullspace_probes'][start]
            check('Probe summary independently matches '+start,len(rr)==counts['accepted'] and sum(float(r['nrmse_after'])<float(r['nrmse_before'])-1e-12 for r in rr)==counts['correspondence_improves'] and sum(float(r['nrmse_after'])>float(r['nrmse_before'])+1e-12 for r in rr)==counts['correspondence_worsens'])
    result=dict(passed=True,checks=checks,network_eligible=False,note='Integrity passes do not establish recovery, convergence or independent-anchor validity')
    (out/'verification.json').write_text(json.dumps(result,indent=2)+chr(10),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
