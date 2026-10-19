"""Independently audit v4 candidate selection, geometry and result provenance."""
import argparse
import csv
import json
from collections import defaultdict
import numpy as np
from stage36_information import OUT, ROOT, CACHE, OLD, V2, CONFIG
from stage36_information import subspaces, smooth_gram, smooth_terms, holdout_vertices, observation_each
from stage36_constraints import layouts, geometry_metrics
from stage36_smooth import SmoothPatch, sha, write_json


def independent_pick(pool,reference):
    if not pool:return reference
    best=min(pool,key=lambda c:(float(np.mean(c['selection'])),c['name']))
    differences=reference['selection']-best['selection']
    margin=max(1e-12,2*float(np.std(differences,ddof=1))/np.sqrt(len(differences)))
    return best if float(np.mean(differences))>margin else reference


def main():
    protocol=json.loads(CONFIG.read_text(encoding='utf-8'))
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    frozen=json.loads((OUT/'frozen_protocol.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/'checkpoints').glob('*.json'))]
    checks=[]
    def check(name,condition):checks.append(dict(name=name,passed=bool(condition)))
    check('frozen_source_and_inputs_match',all(sha(ROOT/p)==h for p,h in frozen['hashes'].items()))
    check('original_output_hashes_match',all(sha(OUT/p)==h for p,h in manifest['files'].items()))
    check('all_21_checkpoints',len(records)==21 and len({(r['state'],r['seed']) for r in records})==21)
    rows={name:[r for c in records for r in c[name]] for name in ['starts','candidates','policies','subjects','spectra']}
    key=lambda r: (r['state'],r['seed'],r['layout'],r['sigma'],r['replicate'])
    check('441_unique_starts',len(rows['starts'])==441 and len({key(r) for r in rows['starts']})==441)
    feasible={key(r) for r in rows['starts'] if r['status']=='feasible'}
    check('twelve_candidates_per_feasible_start',len(rows['candidates'])==12*len(feasible) and len({key(r)+(r['candidate'],) for r in rows['candidates']})==12*len(feasible))
    check('eight_policy_records_per_feasible_start',len(rows['policies'])==8*len(feasible))
    check('no_invalid_start_silently_repaired',all(key(r) in feasible for r in rows['candidates']))
    check('exact_sparse_normal_equation',max(r['normal_equation_inf'] for r in rows['starts'])<1e-10)
    check('all_exported_candidates_geometry_valid',all(r['geometry_valid'] and r['normal_reversals']==0 and r['min_area_ratio']>=.2 and r['max_displacement_ratio']<=.01 for r in rows['candidates']))
    check('observation_bounds_respected',all(r['maximum_anchor_mse_increase']<=r['anchor_increase_budget']+1e-14 for r in rows['candidates']))
    check('rejected_candidates_stay_put',all(abs(r['delta_nrmse'])<1e-14 for r in rows['candidates'] if not r['accepted']))
    check('energy_only_accepted_updates_lower_energy',all(r['energy_after']<=r['energy_before']+1e-12 for r in rows['candidates'] if r['accepted'] and r['arm']=='energy'))
    check('smooth_only_accepted_updates_lower_smoothness',all(r['smooth_after']<=r['smooth_before']+1e-12 for r in rows['candidates'] if r['accepted'] and r['arm']=='smooth'))
    check('selected_candidates_pass_geometry',all(r['geometry_valid'] for r in rows['policies']))
    check('no_network_release',summary['network_eligible'] is False)
    check('local_decisions_equal_frozen_checks',all(r['local_followup_supported']==all(r['tests'].values()) for r in summary['decisions']))
    subject_groups=defaultdict(list)
    for r in rows['subjects']:subject_groups[key(r)+(r['step'],r['policy'])].append(r)
    check('twenty_distinct_identities_per_policy',all(len(v)==20 and len({r['subject'] for r in v})==20 for v in subject_groups.values()))
    check('group_errors_equal_identity_means',all(np.isclose(r['nrmse'],np.mean([s['nrmse'] for s in subject_groups[key(r)+(r['step'],r['policy'])]]),rtol=1e-12,atol=1e-15) for r in rows['policies']))
    check('63_design_records',len(rows['spectra'])==63)
    check('holdout_sets_are_disjoint',all(not set(r['fit_indices'])&set(r['selection_indices']) and not set(r['fit_indices'])&set(r['audit_indices']) and not set(r['selection_indices'])&set(r['audit_indices']) for r in rows['spectra']))
    check('selection_and_audit_have_12_points',all(len(r['selection_indices'])==12 and len(r['audit_indices'])==12 for r in rows['spectra']))
    # Replay every selector from saved parameters and freshly regenerated noisy observations.
    selector_matches=True;parameter_bounds=True;geometry_replays=True;gradients_valid=True
    replay_count=0;geometry_count=0;gradient_errors=[]
    for record in records:
        state,seed=record['state'],record['seed']
        base=np.load(CACHE/f'all20_{state}_baseline.npy');faces=np.load(CACHE/f'{state}_faces.npy')
        with np.load(OLD/'patches'/f'{state}_{seed}.npz') as z:indices=z['indices'];center=int(z['center'])
        model=SmoothPatch(base,faces,indices,center,109.16412439918614)
        ls=layouts(model);union=np.unique(np.concatenate(list(ls.values())))
        sel,audit=holdout_vertices(model,union)
        mask=np.ones(len(model.indices),bool);mask[np.r_[union,sel,audit]]=False
        sd=model.basis[sel].reshape(-1,6)
        with np.load(V2/'parameters'/f'{state}_{seed}.npz') as original:q_input=original['initial_q']
        gram=smooth_gram(model)
        # Check spatial-prior Gram is PSD and ignores absolute parameter origin.
        check(f'correction_prior_{state}_{seed}',np.linalg.eigvalsh(gram).min()>=-1e-12 and smooth_terms(q_input,q_input,gram)[0]==0)
        # Independent centered difference away from the zero chart's triangle boundaries.
        q=.37*q_input
        rng=np.random.default_rng(993+seed);direction=rng.normal(size=q.shape);direction/=np.linalg.norm(direction)
        h=1e-6
        def dense_loss(qq):
            points=model.sample(qq)[0]
            squared=np.sum((points-model.origin)**2,axis=-1)/model.diagonals[:,None]**2
            return float(squared[:,mask].mean())
        from stage36_information import dense_gradient
        derivative=float(np.sum(dense_gradient(model,q,mask)*direction))
        numeric=(dense_loss(q+h*direction)-dense_loss(q-h*direction))/(2*h)
        gradient_errors.append(abs(derivative-numeric))
        gradients_valid &= abs(derivative-numeric)<=1e-11+1e-3*abs(numeric)
        candidates_by_start=defaultdict(list)
        for r in record['candidates']:candidates_by_start[key(r)].append(r)
        policies_by_start=defaultdict(list)
        for r in record['policies']:policies_by_start[key(r)].append(r)
        with np.load(OUT/'parameters'/f'{state}_{seed}.npz') as parameters:
            for start in record['starts']:
                if start['status']!='feasible':continue
                layout,sigma,rep=start['layout'],start['sigma'],start['replicate']
                param_prefix=f'{layout}_{sigma}_{rep}'
                qstart=parameters[param_prefix+'_start']
                noise_rng=np.random.default_rng(364000+protocol['states'].index(state)*1000+seed*100+rep)
                targets=noise_rng.normal(size=(20,len(model.indices),2))*(sigma/.002)
                ys=targets[:,sel].reshape(20,-1)
                design=model.basis[ls[layout]].reshape(-1,6);y=targets[:,ls[layout]].reshape(20,-1)
                selection=lambda q:np.sum((q@sd.T-ys)**2,axis=1)/len(sel)
                stay=dict(name='stay',selection=selection(qstart),arm='stay',q=qstart)
                pool=[]
                for r in candidates_by_start[key(start)]:
                    qtrial=parameters[r['parameter_key']]
                    parameter_bounds &= np.max(np.abs(qtrial-qstart))<=r['step']+1e-12
                    increases=observation_each(qtrial,design,y)-observation_each(qstart,design,y)
                    parameter_bounds &= float(increases.max())<=r['anchor_increase_budget']+1e-14
                    if not r['accepted']:parameter_bounds &= np.array_equal(qtrial,qstart)
                    if r['accepted']:pool.append(dict(name=r['candidate'],arm=r['arm'],step=r['step'],selection=selection(qtrial),q=qtrial))
                for step in protocol['coefficient_step_caps']:
                    current=[c for c in pool if c['step']==step]
                    baseline=independent_pick([c for c in current if c['arm']=='smooth'],stay)
                    energy=independent_pick([c for c in current if c['arm']=='energy'],stay)
                    augmented=independent_pick([c for c in current if c['arm'] in ('energy','smooth_energy')],baseline)
                    choices=dict(stay=stay,smooth=baseline,energy=energy,augmented=augmented)
                    for r in policies_by_start[key(start)]:
                        if r['step']!=step:continue
                        selector_matches &= r['chosen']==choices[r['policy']]['name'];replay_count+=1
                    # Re-evaluate selected primary updates for every layout, noise and replicate.
                    if step==protocol['primary_step']:
                        mm,each=geometry_metrics(model,augmented['q'],mask)
                        r=next(r for r in policies_by_start[key(start)] if r['step']==step and r['policy']=='augmented')
                        geometry_replays &= mm['geometry_valid'] and np.isclose(mm['nrmse'],r['nrmse'],rtol=1e-12,atol=1e-15)
                        geometry_count+=1
        print(f'reviewed {state}/{seed}',flush=True)
    check('all_selectors_reproduced_without_truth',selector_matches)
    check('all_candidate_parameter_and_observation_bounds_replayed',parameter_bounds)
    check('selected_geometry_and_dense_error_replayed',geometry_replays)
    check('dense_diagnostic_gradient_matches_finite_difference',gradients_valid)
    result=dict(passed=all(c['passed'] for c in checks),count=len(checks),checks=checks,
                selector_replays=replay_count,geometry_replays=geometry_count,max_dense_gradient_fd_error=max(gradient_errors),
                reviewer_sha256=sha(__file__),network_eligible=False)
    write_json(OUT/'verification.json',result)
    if not result['passed']:
        print(json.dumps(result,indent=2));raise SystemExit(1)
    # Compact supplementary statistics; no parameter tuning from results.
    by_layout=[]
    for layout in protocol['layouts']:
        dd=[r for r in rows['spectra'] if r['layout']==layout]
        conditions=[r for r in rows['starts'] if r['layout']==layout and r['status']=='feasible']
        policies=[r for r in rows['policies'] if r['layout']==layout and r['sigma']>0 and r['step']==.01]
        by_layout.append(dict(layout=layout,null_dimensions=sorted({r['nullity'] for r in dd}),
                              weak_dimensions=sorted({r['weak_dim'] for r in dd}),
                              median_energy_null_fraction=float(np.median([r['energy_null_fraction'] for r in conditions])),
                              median_energy_weak_fraction=float(np.median([r['energy_weak_fraction'] for r in conditions])),
                              policies=[dict(policy=p,count=len(rr),stay=sum(r['chosen']=='stay' for r in rr),
                                             median_nrmse=float(np.median([r['nrmse'] for r in rr])),
                                             median_delta_vs_smooth=float(np.median([r['delta_vs_smooth'] for r in rr])))
                                        for p in ['stay','smooth','energy','augmented'] for rr in [[r for r in policies if r['policy']==p]]]))
    possibility=[]
    candidate_groups=defaultdict(list)
    for r in rows['candidates']:candidate_groups[key(r)+(r['step'],)].append(r)
    policy_lookup={key(r)+(r['step'],r['policy']):r for r in rows['policies']}
    for layout in protocol['layouts']:
        selected=[r for r in rows['policies'] if r['layout']==layout and r['sigma']>0 and r['step']==protocol['primary_step'] and r['policy']=='augmented']
        opportunities=0;captured=0;harm=0;oracle_gains=[];selected_e=0;material_e=0
        for r in selected:
            reference=policy_lookup[key(r)+(r['step'],'smooth')]
            pool=[c for c in candidate_groups[key(r)+(r['step'],)] if c['accepted'] and c['arm'] in ('energy','smooth_energy')]
            oracle=min([reference['nrmse']]+[c['nrmse'] for c in pool])
            margin=max(1e-9,.01*reference['nrmse'])
            exists=oracle<reference['nrmse']-margin
            improvement=r['nrmse']<reference['nrmse']-margin
            opportunities+=exists;captured+=bool(exists and improvement)
            harm+=r['nrmse']>reference['nrmse']+margin
            selected_e+=r['chosen_arm'] in ('energy','smooth_energy')
            material_e+=bool(r['chosen_arm'] in ('energy','smooth_energy') and improvement)
            oracle_gains.append((reference['nrmse']-oracle)/max(reference['nrmse'],1e-12))
        possibility.append(dict(layout=layout,feasible_cases=len(selected),
                                diagnostic_oracle_material_opportunities=int(opportunities),
                                selector_captures_material_opportunities=int(captured),
                                selector_energy_cases=int(selected_e),selector_material_successes=int(material_e),
                                selector_material_harms=int(harm),
                                oracle_median_relative_gain=float(np.median(oracle_gains))))
    write_json(OUT/'review.json',dict(by_layout=by_layout,possibility_vs_selectability=possibility,verification_count=len(checks),
                oracle_warning='Oracle uses dense truth after all decisions solely to bound opportunities among the fixed candidate pool. It is not an executable method or a fair performance claim.',
                interpretation='Observation-based selector is replayed without dense truth. Selected audit anchors were never used for updates or selection. Previously viewed identities, synthetic observations and fixed local parameterization limit transfer.'))
    plot(rows,summary)
    write_json(OUT/'review_manifest.json',dict(reviewer_sha256=sha(__file__),source_manifest_sha256=sha(OUT/'manifest.json'),
        files={p:sha(OUT/p) for p in ['verification.json','review.json','information_review.png','information_review.pdf']}))
    print(json.dumps(dict(passed=True,checks=len(checks),selector_replays=replay_count,geometry_replays=geometry_count,by_layout=by_layout),indent=2))


def plot(rows,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    names=['two_v2','three_clustered','six_spread'];labels=['2 anchors','3 clustered','6 spread'];xx=np.arange(3)
    feasible=[]
    for name in names:
        rr=[r for r in rows['starts'] if r['layout']==name and r['sigma']>0]
        feasible.append(100*sum(r['status']=='feasible' for r in rr)/len(rr))
    axes[0,0].bar(xx,feasible,color='#4c78a8');axes[0,0].set(xticks=xx,xticklabels=labels,ylim=(0,105),ylabel='Feasible noisy starts (%)',title='Invalid starts remain in the denominator')
    for i,policy in enumerate(['stay','smooth','energy','augmented']):
        values=[np.median([r['nrmse'] for r in rows['policies'] if r['layout']==name and r['sigma']>0 and r['step']==.01 and r['policy']==policy]) for name in names]
        axes[0,1].bar(xx+(i-1.5)*.18,values,.18,label=policy)
    axes[0,1].set(xticks=xx,xticklabels=labels,yscale='log',ylabel='Median dense NRMSE',title='Observable policies; feasible starts only');axes[0,1].legend(fontsize=8)
    for i,(label,field,sign) in enumerate([('dense improvement','delta_vs_smooth',-1),('unused-anchor improvement','audit_gain_vs_smooth',1)]):
        values=[sum(sign*r[field]>0 for r in rows['policies'] if r['layout']==name and r['sigma']>0 and r['step']==.01 and r['policy']=='augmented') for name in names]
        axes[1,0].bar(xx+(i-.5)*.3,values,.3,label=label)
    axes[1,0].set(xticks=xx,xticklabels=labels,ylabel='Positive paired cases (ties excluded)',title='Added E candidates versus selected smooth');axes[1,0].legend(fontsize=8)
    for name,label in zip(names,labels):
        rr=[r for r in rows['policies'] if r['layout']==name and r['sigma']>0 and r['step']==.01 and r['policy']=='augmented']
        axes[1,1].scatter([r['audit_gain_vs_smooth'] for r in rr],[-r['delta_vs_smooth'] for r in rr],s=15,alpha=.5,label=label)
    axes[1,1].axhline(0,color='black',linewidth=.7);axes[1,1].axvline(0,color='black',linewidth=.7)
    axes[1,1].set(xlabel='Unused-anchor MSE improvement',ylabel='Dense NRMSE improvement',title='Agreement checked after selection');axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('S3.6 v4: local correction + separate selection / audit / dense evaluation')
    fig.text(.5,.01,'Primary coefficient cap=0.01. Synthetic observations, previously viewed identities; no convergence or neural-training claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.95))
    for suffix in ['png','pdf']:fig.savefig(OUT/f'information_review.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
