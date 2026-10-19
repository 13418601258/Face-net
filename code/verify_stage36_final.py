"""Independently replay final selection and numerical/geometry evidence."""
import json
import numpy as np
from stage36_final import OUT,ROOT,CACHE,STATES,build_model,sample_valid,measure,observation_each
from stage36_smooth import sha,write_json

def main():
    checks=[]
    def check(name,ok,details=None):
        checks.append(dict(name=name,passed=bool(ok),details=details))
        if not ok:print('FAIL',name,details,flush=True)
    frozen=json.loads((OUT/'frozen_protocol.json').read_text(encoding='utf-8'))
    for path,digest in frozen['hashes'].items():check('input:'+path,sha(ROOT/path)==digest)
    manifest=json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
    for path,digest in manifest['files'].items():check('output:'+path,sha(OUT/path)==digest)
    base=np.load(OUT/'joint_baseline.npy');faces=np.load(CACHE/'au0_faces.npy')
    all_starts=[];all_policies=[];all_subjects=[];selected_replayed=0;measured=0
    for seed in range(3):
        job=json.loads((OUT/'checkpoints'/f'seed_{seed}.json').read_text(encoding='utf-8'))
        model,chart=build_model(base,faces,seed)
        check(f'chart:{seed}',chart==job['chart']);check(f'numeric:{seed}',all(r['passed'] for r in job['numeric']))
        check(f'structure:{seed}',job['structure']['passed'])
        check(f'spectra:{seed}',[r['terms'] for r in job['structure']['modes']]==[20,7])
        all_starts+=job['starts'];all_policies+=job['policies'];all_subjects+=job['subjects']
        with np.load(OUT/'parameters'/f'seed_{seed}.npz') as z:
            mask=z['evaluation_mask'];sel=z['selection_indices'];audit=z['audit_indices']
            sd=model.basis[sel].reshape(-1,6);ad=model.basis[audit].reshape(-1,6)
            fit_union=np.r_[z['fit_two'],z['fit_six']]
            check(f'disjoint:{seed}',not np.intersect1d(sel,audit).size and not np.intersect1d(fit_union,np.r_[sel,audit]).size and not mask[np.r_[fit_union,sel,audit]].any())
            for start in job['starts']:
                case=start['case'];q0=z[case+'_start'];y=z[case+'_fit'];ys=z[case+'_selection'];ya=z[case+'_audit']
                design=model.basis[z['fit_two'] if start['layout']=='two_v2' else z['fit_six']].reshape(-1,6)
                valid=sample_valid(model,q0) is not None;check('start:'+case,valid==start['feasible'])
                if not valid:continue
                candidates=[dict(name='stay',family='stay',cap=0.,accepted=True,parameter_key=case+'_start')]+[r for r in job['candidates'] if r['case']==case]
                losses={c['name']:observation_each(z[c['parameter_key']],sd,ys) for c in candidates}
                def choose(pool,reference):
                    if not pool:return reference
                    winner=min(pool,key=lambda c:(float(losses[c['name']].mean()),c['name']))
                    gain=(losses[reference['name']]-losses[winner['name']]).reshape(7,20).mean(0)
                    return winner if gain.mean()>max(1e-12,2*gain.std(ddof=1)/np.sqrt(20)) else reference
                for cap in [.0025,.01]:
                    valid_candidates=[c for c in candidates if c['cap']==cap and c['accepted']]
                    reg=choose([c for c in valid_candidates if c['family']=='regularizer'],candidates[0])
                    selected={'regularizer':reg,'single':choose([c for c in valid_candidates if c['family']=='single'],reg),'joint':choose([c for c in valid_candidates if c['family']=='joint'],reg)}
                    for policy,c in selected.items():
                        row=next(r for r in job['policies'] if r['case']==case and r['cap']==cap and r['policy']==policy)
                        check(f'selection:{case}:{cap}:{policy}',row['chosen']==c['name']);selected_replayed+=1
                        if cap==.01:
                            mm=measure(model,z[c['parameter_key']],mask,sd,ys,ad,ya)
                            check(f'measure:{case}:{policy}',abs(mm['metrics']['nrmse']-row['nrmse'])<1e-14 and mm['metrics']['geometry_valid']);measured+=1
                            subject=[r for r in job['subjects'] if r['case']==case and r['cap']==cap and r['policy']==policy]
                            check(f'subjects:{case}:{policy}',len(subject)==140 and np.allclose([r['nrmse'] for r in subject],mm['each'],atol=1e-14,rtol=0))
                budget=2*(start['sigma']/.002)**2+1e-12;before=observation_each(q0,design,y)
                for c in candidates[1:]:
                    q=z[c['parameter_key']];increase=observation_each(q,design,y)-before
                    check('constraints:'+case+':'+c['name'],sample_valid(model,q) is not None and increase.max()<=budget+1e-14)
                    if not c['accepted']:check('rejection_stays:'+case+':'+c['name'],np.array_equal(q,q0))
        print(f'Independent replay seed {seed} complete',flush=True)
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    check('30_unique_starts',len(all_starts)==30 and len({r['case'] for r in all_starts})==30)
    check('6_controls',sum(r['pattern']=='control' for r in all_starts)==6)
    check('policy_rows',len(all_policies)==summary['counts']['policies'])
    check('subject_rows',len(all_subjects)==summary['counts']['subject_records'])
    for d in summary['decisions']:
        check('gate:'+d['layout'],d['passed']==all(d['tests'].values()))
        rows=[r for r in all_policies if r['policy']=='joint' and r['cap']==.01 and r['layout']==d['layout'] and r['pattern']!='control']
        check('median:'+d['layout'],abs(float(np.median([r['relative_gain'] for r in rows]))-d['median_relative_gain'])<1e-14)
        check('selection_count:'+d['layout'],sum(r['chosen_family']=='joint' for r in rows)==d['joint_selected'])
    check('network_closed_if_gate_failed',summary['local_gate_passed'] or (not summary['network_eligible'] and summary['closed_loop_status']=='skipped_by_frozen_gate'))
    result=dict(passed=all(c['passed'] for c in checks),check_count=len(checks),failed=[c for c in checks if not c['passed']],selection_replayed=selected_replayed,selected_endpoints_remeasured=measured,checks=checks)
    write_json(OUT/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False),flush=True)
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
