"""S3.6 v4: additional correction information and truth-free update selection."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
from stage36_smooth import ROOT, CACHE, OLD, OUT as V2, SmoothPatch, sha, write_json, write_csv
from stage36_constraints import layouts, exact_sparse, observation, geometry_metrics

OUT = ROOT/'code/reports/stage3_s36_v4_information'
CONFIG = ROOT/'code/stage36_information_protocol.json'


def subspaces(design, cutoff=.05):
    _, singular, vt = np.linalg.svd(design, full_matrices=True)
    values = np.zeros(design.shape[1]); values[:len(singular)] = singular
    ratios = values/values[0]
    null = vt[ratios <= 1e-12]
    weak = vt[ratios <= cutoff]
    return null.T@null, weak.T@weak, ratios


def holdout_vertices(model, fitting_union, count=24):
    possible = np.flatnonzero(np.linalg.norm(model.start/model.radius, axis=1)<.7)
    possible = np.setdiff1d(possible, fitting_union)
    chosen = [int(possible[np.argmin(np.linalg.norm(model.start[possible], axis=1))])]
    while len(chosen)<count:
        candidates = np.setdiff1d(possible, chosen)
        distance = np.linalg.norm(model.start[candidates,None]-model.start[chosen][None],axis=-1).min(1)
        chosen.append(int(candidates[np.argmax(distance)]))
    return np.array(chosen[::2]), np.array(chosen[1::2])


def smooth_gram(model):
    edges = np.unique(np.sort(np.vstack((model.faces[:,[0,1]],model.faces[:,[1,2]],model.faces[:,[2,0]])),axis=1),axis=0)
    lookup = np.full(model.base.shape[1],-1,int); lookup[model.indices] = np.arange(len(model.indices))
    b = np.zeros((model.base.shape[1],2,6)); b[model.indices] = model.basis
    delta = b[edges[:,0]]-b[edges[:,1]]
    gram = np.einsum('eak,eal->kl',delta,delta)/len(edges)
    gram /= np.trace(gram)/6
    return gram


def smooth_terms(q, q_input, gram):
    correction = q-q_input
    return float(np.sum((correction@gram)*correction)/len(q)), 2*correction@gram/len(q)


def observation_each(q, design, target):
    residual = q@design.T-target
    return np.sum(residual**2,axis=1)/(len(design)/2)


def selection_gain(reference, candidate):
    gain = np.asarray(reference)-np.asarray(candidate)
    mean = float(np.mean(gain)); se = float(np.std(gain,ddof=1)/np.sqrt(len(gain)))
    return mean > max(1e-12,2*se), mean, se


def select_policies(candidates, step):
    stay = candidates[0]
    valid = [c for c in candidates[1:] if c['row']['step']==step and c['row']['accepted']]
    def pick(pool, reference):
        if not pool:return reference
        winner = min(pool, key=lambda c:(float(c['selection'].mean()),c['row']['candidate']))
        return winner if selection_gain(reference['selection'],winner['selection'])[0] else reference
    smooth = pick([c for c in valid if c['row']['arm']=='smooth'],stay)
    energy = pick([c for c in valid if c['row']['arm']=='energy'],stay)
    augmented = pick([c for c in valid if c['row']['arm'] in ('energy','smooth_energy')],smooth)
    return dict(stay=stay, smooth=smooth, energy=energy, augmented=augmented)


def dense_gradient(model, q, evaluation_mask):
    points,jac,_ = model.sample(q)
    diff = (points-model.origin)/model.diagonals[:,None,None]**2
    diff[:,~evaluation_mask] = 0
    guv = np.einsum('njca,njc->nja',jac,diff)
    return 2*model.scale*np.einsum('jak,nja->nk',model.basis,guv)/(len(q)*evaluation_mask.sum())


def propose(model, q, q_input, gram, design, target, sigma_q, projection, arm, cap):
    current = model.evaluate(q)
    r0,gs = smooth_terms(q,q_input,gram)
    ge = current['ge']; ps=gs@projection; pe=ge@projection
    ns,ne = np.linalg.norm(ps),np.linalg.norm(pe)
    ws = 1/max(ns,1e-30) if arm in ('smooth','smooth_energy') and ns>1e-14 else 0.
    we = 1/max(ne,1e-30) if arm in ('energy','smooth_energy') and ne>1e-14 else 0.
    gradient = ws*ps+we*pe
    norm = float(np.max(np.abs(gradient)))
    information = dict(accepted=False,reason='no_direction',halvings=0,
                       smooth_norm=float(ns),energy_norm=float(ne),weighted_smooth=ws,weighted_energy=we,
                       energy_before=current['energy'],energy_after=current['energy'],smooth_before=r0,smooth_after=r0)
    if norm<=1e-14:return q.copy(),information,np.zeros_like(q)
    direction=-cap*gradient/norm
    slope=float(np.sum((ws*gs+we*ge)*direction))
    value=ws*r0+we*current['energy']
    anchor_before=observation_each(q,design,target)
    information['reason']='rejected'
    for attempt in range(12):
        factor=2.**(-attempt); trial=q+factor*direction
        anchor_after=observation_each(trial,design,target)
        if np.any(anchor_after>anchor_before+2*sigma_q**2+1e-12):continue
        candidate=model.evaluate(trial)
        if candidate is None:continue
        r1,_=smooth_terms(trial,q_input,gram)
        if ws*r1+we*candidate['energy']<=value+1e-4*factor*slope:
            information.update(accepted=True,reason='accepted',halvings=attempt,
                               energy_after=candidate['energy'],smooth_after=r1)
            return trial,information,direction
    information['halvings']=12
    return q.copy(),information,direction


def assess(model,q,mask,sel_design,sel_target,audit_design,audit_target):
    metrics,each=geometry_metrics(model,q,mask)
    return metrics,each,observation_each(q,sel_design,sel_target),observation_each(q,audit_design,audit_target)


def run_case(state,seed,protocol):
    destination=OUT/'checkpoints'/f'{state}_{seed}.json'
    if destination.exists():return json.loads(destination.read_text(encoding='utf-8'))
    base=np.load(CACHE/f'all20_{state}_baseline.npy');faces=np.load(CACHE/f'{state}_faces.npy')
    with np.load(OLD/'patches'/f'{state}_{seed}.npz') as z:indices=z['indices'];center=int(z['center'])
    model=SmoothPatch(base,faces,indices,center,109.16412439918614)
    with np.load(V2/'parameters'/f'{state}_{seed}.npz') as z:q_input=z['initial_q']
    ls=layouts(model);union=np.unique(np.concatenate(list(ls.values())))
    sel,audit=holdout_vertices(model,union)
    mask=np.ones(len(model.indices),bool);mask[np.r_[union,sel,audit]]=False
    sd=model.basis[sel].reshape(-1,6);ad=model.basis[audit].reshape(-1,6)
    gram=smooth_gram(model)
    rows=[];starts=[];policies=[];subjects=[];spectra=[];params={}
    for layout in protocol['layouts']:
        anchors=ls[layout];design=model.basis[anchors].reshape(-1,6)
        pnull,pweak,ratios=subspaces(design,protocol['weak_singular_ratio'])
        spectra.append(dict(state=state,seed=seed,layout=layout,rank=int(np.sum(ratios>1e-12)),
                            nullity=int(round(np.trace(pnull))),weak_dim=int(round(np.trace(pweak))),
                            weak_dims_sensitivity=[int(np.sum(ratios<=t)) for t in protocol['weak_rank_sensitivity']],
                            relative_singular_values=ratios.tolist(),fit_indices=model.indices[anchors].tolist(),
                            selection_indices=model.indices[sel].tolist(),audit_indices=model.indices[audit].tolist(),
                            evaluation_vertices=int(mask.sum())))
        for noise in protocol['noise']:
            sigma=noise['sigma'];sq=sigma/.002
            for replicate in noise['replicates']:
                rng=np.random.default_rng(364000+protocol['states'].index(state)*1000+seed*100+replicate)
                targets=rng.normal(size=(20,len(model.indices),2))*sq
                y=targets[:,anchors].reshape(20,-1);ys=targets[:,sel].reshape(20,-1);ya=targets[:,audit].reshape(20,-1)
                q=exact_sparse(q_input,design,y)
                common=dict(state=state,seed=seed,layout=layout,sigma=sigma,replicate=replicate)
                key=f'{layout}_{sigma}_{replicate}'
                params[key+'_start']=q
                m,each,selection,audit_value=assess(model,q,mask,sd,ys,ad,ya)
                current=model.evaluate(q)
                start=dict(**common,**m,status='feasible' if current else 'invalid_start',
                           normal_equation_inf=float(np.max(np.abs(observation(q,design,y)[1]))),
                           selection_residual=float(selection.mean()),audit_residual=float(audit_value.mean()))
                if current is None:
                    starts.append(start);continue
                ge=current['ge'];norm=float(np.linalg.norm(ge))
                start.update(energy_null_fraction=float(np.linalg.norm(ge@pnull)/max(norm,1e-30)),
                             energy_weak_fraction=float(np.linalg.norm(ge@pweak)/max(norm,1e-30)))
                starts.append(start)
                baseline=dict(row=dict(**common,candidate='stay',arm='stay',space='none',step=0.,accepted=True,**m),
                              q=q,each=each,selection=selection,audit=audit_value)
                candidates=[baseline];gd=dense_gradient(model,q,mask)
                for space,projection in [('null_and_weak',pweak),('full',np.eye(6))]:
                    for arm in protocol['arms']:
                        for cap in protocol['coefficient_step_caps']:
                            trial,detail,direction=propose(model,q,q_input,gram,design,y,sq,projection,arm,cap)
                            mm,ee,ss,aa=assess(model,trial,mask,sd,ys,ad,ya)
                            candidate=f'{space}:{arm}:{cap}'
                            param_key=key+'_'+candidate
                            params[param_key]=trial
                            drift=observation_each(trial,design,y)-observation_each(q,design,y)
                            row=dict(**common,candidate=candidate,arm=arm,space=space,step=cap,**detail,**mm,
                                     initial_nrmse=m['nrmse'],delta_nrmse=mm['nrmse']-m['nrmse'],
                                     evaluation_directional_derivative=float(np.sum(gd*direction)),
                                     maximum_anchor_mse_increase=float(drift.max()),anchor_increase_budget=2*sq**2+1e-12,
                                     selection_residual=float(ss.mean()),audit_residual=float(aa.mean()),parameter_key=param_key)
                            rows.append(row)
                            candidates.append(dict(row=row,q=trial,each=ee,selection=ss,audit=aa))
                for cap in protocol['coefficient_step_caps']:
                    chosen=select_policies(candidates,cap)
                    reference=chosen['smooth']
                    for name,result in chosen.items():
                        error=float(result['each'].mean());ref_error=float(reference['each'].mean())
                        policies.append(dict(**common,step=cap,policy=name,chosen=result['row']['candidate'],
                                             chosen_arm=result['row']['arm'],nrmse=error,initial_nrmse=m['nrmse'],
                                             delta_vs_smooth=error-ref_error,relative_gain_vs_smooth=(ref_error-error)/max(ref_error,1e-12),
                                             audit_residual=float(result['audit'].mean()),audit_gain_vs_smooth=float((reference['audit']-result['audit']).mean()),
                                             selection_residual=float(result['selection'].mean()),geometry_valid=result['row']['geometry_valid']))
                        for n in range(20):
                            subjects.append(dict(**common,step=cap,policy=name,subject=f'ast{n+1:03}',
                                                 nrmse=float(result['each'][n]),smooth_nrmse=float(reference['each'][n]),
                                                 initial_nrmse=float(each[n]),audit_residual=float(result['audit'][n])))
    output=dict(state=state,seed=seed,starts=starts,candidates=rows,policies=policies,subjects=subjects,spectra=spectra)
    parameter_path=OUT/'parameters'/f'{state}_{seed}.npz';parameter_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(parameter_path,**params)
    write_json(destination,output)
    print(f'completed {state}/{seed}: {len(starts)} starts, {len(rows)} proposals',flush=True)
    return output


def summarize(records,protocol):
    all_rows={name:[r for c in records for r in c[name]] for name in ['starts','candidates','policies','subjects','spectra']}
    for name,rr in all_rows.items():
        # Some invalid starts omit gradient fractions; keep a stable full schema.
        fields=list(dict.fromkeys(k for r in rr for k in r))
        write_csv(OUT/f'{name}.csv',[{k:r.get(k) for k in fields} for r in rr])
    decisions=[];comparisons=[]
    for layout in protocol['layouts']:
        for space in protocol['spaces']:
            for cap in protocol['coefficient_step_caps']:
                cc=[r for r in all_rows['candidates'] if r['layout']==layout and r['space']==space and r['step']==cap and r['sigma']>0]
                lookup={(r['state'],r['seed'],r['sigma'],r['replicate'],r['arm']):r for r in cc}
                for arm in ['energy','smooth_energy']:
                    pairs=[(r,lookup[(r['state'],r['seed'],r['sigma'],r['replicate'],'smooth')]) for r in cc if r['arm']==arm]
                    differences=[a['nrmse']-b['nrmse'] for a,b in pairs]
                    comparisons.append(dict(layout=layout,space=space,step=cap,arm=arm,count=len(pairs),
                        accepted=sum(a['accepted'] for a,b in pairs),
                        improves_smooth=sum(a['nrmse']<b['nrmse']-max(1e-9,.01*b['nrmse']) for a,b in pairs),
                        worsens_smooth=sum(a['nrmse']>b['nrmse']+max(1e-9,.01*b['nrmse']) for a,b in pairs),
                        median_delta_vs_smooth=float(np.median(differences)) if differences else None))
        starts=[r for r in all_rows['starts'] if r['layout']==layout and r['sigma']>0]
        rr=[r for r in all_rows['policies'] if r['layout']==layout and r['sigma']>0 and r['step']==protocol['primary_step'] and r['policy']=='augmented']
        alt=[r for r in all_rows['policies'] if r['layout']==layout and r['sigma']>0 and r['step']==.0025 and r['policy']=='augmented']
        sr=[r for r in all_rows['subjects'] if r['layout']==layout and r['sigma']>0 and r['step']==protocol['primary_step'] and r['policy']=='augmented']
        median=lambda a:float(np.median(a)) if a else 0.
        e_uses=sum(r['chosen_arm'] in ('energy','smooth_energy') for r in rr)
        state_positive=sum(median([-r['delta_vs_smooth'] for r in rr if r['state']==s])>0 for s in protocol['states'])
        seed_positive=sum(median([-r['delta_vs_smooth'] for r in rr if r['seed']==s])>0 for s in protocol['seeds'])
        harmed=sum(r['nrmse']>r['smooth_nrmse']+max(1e-9,.01*r['smooth_nrmse']) for r in sr)
        truth_rows=[r for r in all_rows['policies'] if r['layout']==layout and r['sigma']==0 and r['policy']=='augmented' and layout!='two_v2']
        tests=dict(feasible_coverage=len(rr)>=.8*len(starts),nontrivial_energy_use=e_uses>=.2*len(rr) and bool(rr),
                   median_relative_gain=median([r['relative_gain_vs_smooth'] for r in rr])>.01,
                   state_repeatability=state_positive>=5,seed_repeatability=seed_positive>=2,
                   audit_improves=median([r['audit_gain_vs_smooth'] for r in rr])>0,
                   identity_harm_limited=harmed<=.2*len(sr),alternate_step_support=median([-r['delta_vs_smooth'] for r in alt])>0,
                   no_full_rank_truth_drift=all(r['nrmse']<1e-9 for r in truth_rows),
                   geometry_valid=all(r['geometry_valid'] for r in rr))
        decisions.append(dict(layout=layout,expected_noisy=len(starts),feasible_noisy=len(rr),energy_selected=e_uses,
                              median_relative_gain_vs_smooth=median([r['relative_gain_vs_smooth'] for r in rr]),
                              median_delta_vs_smooth=median([r['delta_vs_smooth'] for r in rr]),
                              materially_improves=sum(r['delta_vs_smooth']<-max(1e-9,.01*(r['nrmse']-r['delta_vs_smooth'])) for r in rr),
                              materially_worsens=sum(r['delta_vs_smooth']>max(1e-9,.01*(r['nrmse']-r['delta_vs_smooth'])) for r in rr),
                              positive_states=state_positive,positive_seeds=seed_positive,
                              harmed_identities=harmed,total_identity_records=len(sr),
                              median_audit_gain=median([r['audit_gain_vs_smooth'] for r in rr]),
                              tests=tests,local_followup_supported=all(tests.values())))
    summary=dict(complete=True,counts={k:len(v) for k,v in all_rows.items()},
                 feasible_starts=sum(r['status']=='feasible' for r in all_rows['starts']),
                 invalid_starts=sum(r['status']!='feasible' for r in all_rows['starts']),
                 matched_candidate_comparisons=comparisons,decisions=decisions,
                 any_local_followup_supported=any(d['local_followup_supported'] for d in decisions),network_eligible=False,
                 limitations=protocol['scope']+'; smoothing penalizes corrections from the known initial mapping, not absolute parameters. Noise levels are synthetic and known. Same previously viewed identities in all conditions.')
    write_json(OUT/'summary.json',summary)
    return all_rows,summary


def freeze(protocol):
    paths=[Path(__file__),CONFIG,ROOT/'code/stage36_constraints.py',ROOT/'code/stage36_smooth.py',ROOT/'code/stage36_gradient.py']
    for state in protocol['states']:
        paths.extend([CACHE/f'all20_{state}_baseline.npy',CACHE/f'{state}_faces.npy'])
        for seed in protocol['seeds']:
            paths.extend([OLD/'patches'/f'{state}_{seed}.npz',V2/'parameters'/f'{state}_{seed}.npz'])
    document=dict(protocol=protocol,hashes={str(p.relative_to(ROOT)):sha(p) for p in paths})
    path=OUT/'frozen_protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=document:
        raise ValueError('Frozen code/input changed; create a new version instead of overwriting outcomes.')
    if not path.exists():write_json(path,document)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args();protocol=json.loads(CONFIG.read_text(encoding='utf-8'))
    OUT.mkdir(parents=True,exist_ok=True);freeze(protocol)
    cases=[(s,i) for s in protocol['states'] for i in protocol['seeds']]
    if args.workers==1:records=[run_case(s,i,protocol) for s,i in cases]
    else:
        from concurrent.futures import ProcessPoolExecutor,as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_case,s,i,protocol) for s,i in cases]
            records=[f.result() for f in as_completed(futures)]
    records.sort(key=lambda r:(protocol['states'].index(r['state']),r['seed']))
    _,summary=summarize(records,protocol)
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in ('manifest.json','verification.json'))
    write_json(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):sha(p) for p in files}))
    print(json.dumps({k:v for k,v in summary.items() if k!='matched_candidate_comparisons'},indent=2))


if __name__=='__main__':main()
