"""S2.6 regression driver. Called by validate_protocol.py with v3 config.

Regenerate perturbations reproducibly in memory, preserving all v2 outputs.
No score for missing coordinates. Geometry rows cover one declared subject,
while E_COMP always uses the full 15-subject group.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from astface_loader import ASTFaceReader,parse_subjects
from preprocessing import rigid_gpa,match_group_frame,apply_rigid_group,score_status
from ecomp import covariance_spectrum,ecomp
from index_protocol import perturb_group,sample_seed
from perturb import coordinate_noise,smooth_local_deformation,random_missing
from metrics import evaluate_geometry


def dump(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def table(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def run(data_dir,cfg,out):
    out.mkdir(parents=True,exist_ok=True)
    states=cfg['states'];subjects=parse_subjects(cfg['subjects'])
    gpa_kw={k:cfg['gpa'][k] for k in ['max_iterations','tolerance']}
    if cfg['gpa']['scale']!='preserve' or cfg['gpa']['reflection'] or cfg['gpa']['initialization']!='first_centered_sample':
        raise ValueError('unreviewed GPA settings')
    if cfg['missing_policy']!='geometry_only':raise ValueError('unreviewed missing policy')
    data={};faces={};baselines={};sources=[];digest=hashlib.sha256()
    gpa_logs=[];transforms={};spectra={}
    def align(x,key):
        result=rigid_gpa(x,**gpa_kw)
        gpa_logs.append(dict(key=key,iterations=result.iterations,converged=result.converged,
            relative_change=result.relative_change,objective=result.objective,
            min_rotation_determinant=float(np.linalg.det(result.rotations).min())))
        transforms[key+'_centers']=result.centers
        transforms[key+'_rotations']=result.rotations
        return result
    with ASTFaceReader(data_dir) as reader:
        for state in states:
            shapes=[]
            for subject in subjects:
                m=reader.read_sample(subject,state,modalities=('mesh',)).mesh
                if state in faces and not np.array_equal(faces[state],m.faces):raise ValueError('topology mismatch')
                faces[state]=m.faces
                shapes.append(m.vertices);digest.update(m.vertices.tobytes());digest.update(m.faces.tobytes())
                sources.append(dict(subject=subject,state=state,path=m.path))
            data[state]=np.asarray(shapes,float)
            baselines[state]=align(data[state],state+'_baseline')
            print('GPA baseline',state,baselines[state].iterations,flush=True)
    reference=float(covariance_spectrum(np.concatenate([b.aligned for b in baselines.values()])).sum())
    raw_reference=float(covariance_spectrum(np.concatenate(list(data.values()))).sum())
    k=min(len(subjects)-1,data[states[0]].shape[1]*3)
    score=lambda s:ecomp(s,n_components=k,reference_variance=reference,delta=cfg['delta'])
    oldscore=lambda s:ecomp(s,n_components=k,reference_variance=raw_reference,delta=cfg['delta'])
    baseline_rows=[];rows=[];geometry=[];checks=[];scales=[]
    def check(name,key,passed,value):checks.append(dict(check=name,key=key,passed=bool(passed),value=float(value)))
    for state in states:
        x=data[state];b=baselines[state];ids=[f'{s}/{state}' for s in subjects]
        raw_s=covariance_spectrum(x);bs=covariance_spectrum(b.aligned)
        spectra[state+'_baseline']=bs
        centered=x-x.mean(axis=1)[:,None]
        radii=np.sqrt(np.mean(np.sum(centered**2,axis=2),axis=1))
        # Diagnostic ablation only; the production policy PRESERVES size.
        equal_radius=b.aligned/radii[:,None,None]*radii.mean()
        scales.append(dict(state=state,raw_trace=float(raw_s.sum()),
            centered_trace=float(covariance_spectrum(centered).sum()),
            rigid_GPA_trace=float(bs.sum()),
            equal_RMS_radius_trace_diagnostic_only=float(covariance_spectrum(equal_radius).sum()),
            radius_min=float(radii.min()),radius_max=float(radii.max()),radius_cv=float(radii.std()/radii.mean())))
        baseline_rows.append(dict(state=state,raw_v2_score=oldscore(raw_s),
            raw_with_v3_reference=score(raw_s),aligned_with_v2_reference=oldscore(bs),
            aligned_v3_score=score(bs),alignment_effect_fixed_v3_reference=score(bs)-score(raw_s)))
        repeat=rigid_gpa(x,**gpa_kw)
        check('baseline_repeat',state,abs(score(covariance_spectrum(repeat.aligned))-score(bs))<=cfg['baseline_repeat_score_atol'],
              score(covariance_spectrum(repeat.aligned))-score(bs))
        def record(kind,mode,seed,strength,y,raw_spec,aligned_result,key):
            s=covariance_spectrum(aligned_result.aligned)
            spectra[key]=s
            row=dict(state=state,kind=kind,mode=mode,seed=seed,strength=strength,K=k,
                status='scored',score=score(s),change=score(s)-score(bs),
                raw_v2_change=oldscore(raw_spec)-oldscore(raw_s),
                raw_fixed_v3_change=score(raw_spec)-score(raw_s),
                reference_variance=reference,delta=cfg['delta'],
                iterations=aligned_result.iterations,converged=aligned_result.converged,
                total_variance=float(s.sum()))
            for d in cfg['delta_checks']:
                score_d=lambda v:ecomp(v,n_components=k,reference_variance=reference,delta=d)
                row[f'change_delta_{d:g}']=score_d(s)-score_d(bs)
            rows.append(row)
            return row,s
        for mode in cfg['modes']:
            for seed in cfg['seeds']:
                for strength in [0.]+cfg['strengths']:
                    key=f'{state}_index_{mode}_{seed}_{strength:g}'
                    y=x if strength==0 else perturb_group(x,ids,mode=mode,strength=strength,seed=seed,target=f"{cfg['single_subject']}/{state}")
                    a=b if strength==0 else align(y,key)
                    row,s=record('index',mode,seed,strength,y,covariance_spectrum(y),a,key)
                    if mode=='common':
                        check('common_index',key,np.allclose(s,bs,atol=cfg['invariance_atol'],rtol=cfg['invariance_rtol']),abs(s-bs).max())
                    if mode=='independent' and strength>0:
                        retention=row['change']/row['raw_fixed_v3_change']
                        check('index_signal_retained',key,retention>=cfg['independent_signal_retention_min'],retention)
        print('Index regression',state,flush=True)
        gi=subjects.index(cfg['geometry_subject'])
        ref=b.aligned[gi];f=faces[state]
        def geom(kind,mode,seed,strength,a,mask=None,native=False):
            # Candidate and reference get only a COMMON frame correction after
            # GPA. Missing-only tests have no added pose and use native frame.
            ref_v=x[gi] if native else ref
            candidate=a[gi] if native else match_group_frame(a,b.mean)[gi]
            tol=(float(np.linalg.norm(np.ptp(ref_v,axis=0)))*cfg['geometry_tolerance_diagonal_fraction'])
            r=evaluate_geometry(ref_v,f,candidate,f,valid_mask=mask,tolerance=tol,
                                sample_count=cfg['geometry_surface_samples'],seed=seed)
            geometry.append(dict(state=state,subject=subjects[gi],kind=kind,mode=mode,
                strength=strength,frame='native' if native else 'GPA_then_common_frame',**r))
            if kind in ('baseline','rigid'):
                key=f'{state}_{kind}_{mode}_{seed}_{strength:g}'
                relative_distance=r['symmetric_mean']/tol
                check('geometry_rigid_invariance',key,
                      relative_distance<=1e-5 and r['coverage']==1. and r['normal_reversals']==0,
                      relative_distance)
        geom('baseline','none',0,0,b.aligned)
        for mode in ('common','independent'):
            for seed in cfg['seeds']:
                for strength in cfg['rigid_strengths']:
                    key=f'{state}_rigid_{mode}_{seed}_{strength:g}'
                    y,r,t=apply_rigid_group(x,ids,mode=mode,seed=seed,strength=strength)
                    transforms[key+'_injected_rotations']=r;transforms[key+'_injected_translations']=t
                    a=align(y,key);row,s=record('rigid',mode,seed,strength,y,covariance_spectrum(y),a,key)
                    check('rigid_'+mode,key,abs(row['change'])<=cfg['pose_score_atol'],row['change'])
                    if mode=='common':
                        check('raw_common_rigid',key,abs(row['raw_fixed_v3_change'])<=cfg['pose_score_atol'],row['raw_fixed_v3_change'])
                    geom('rigid',mode,seed,strength,a.aligned)
        functions=dict(coordinate_noise=coordinate_noise,smooth_local_deformation=smooth_local_deformation,random_missing=random_missing)
        for kind,levels in cfg['geometry_operations'].items():
            for seed in cfg['seeds']:
                for strength in levels:
                    key=f'{state}_{kind}_{seed}_{strength:g}'
                    perturbed=[functions[kind](shape,f,strength,seed=sample_seed(seed,identity+'/'+kind)) for shape,identity in zip(x,ids)]
                    y=np.array([p.vertices for p in perturbed],float)
                    status=score_status(y,cfg['missing_policy'])
                    if status!='finite':
                        rows.append(dict(state=state,kind=kind,mode='independent',seed=seed,strength=strength,
                            K=k,status=status,score=None,change=None,raw_v2_change=None,raw_fixed_v3_change=None,
                            reference_variance=reference,delta=cfg['delta'],iterations=None,converged=None,total_variance=None))
                        geom(kind,'independent',seed,strength,y,perturbed[gi].valid_mask,native=True)
                    else:
                        a=align(y,key);record(kind,'independent',seed,strength,y,covariance_spectrum(y),a,key)
                        geom(kind,'independent',seed,strength,a.aligned)
        print('Geometry and pose regression',state,flush=True)
        table(out/'scores.csv',rows);table(out/'geometry.csv',geometry)
        table(out/'alignment_log.csv',gpa_logs);table(out/'acceptance_checks.csv',checks)
    trends=[]
    for state in states:
        for mode in ('single','independent'):
            for seed in cfg['seeds']:
                rr=sorted([r for r in rows if r['state']==state and r['kind']=='index' and r['mode']==mode and r['seed']==seed],key=lambda r:r['strength'])
                diff=np.diff([r['change'] for r in rr]);key=f'{state}_{mode}_{seed}'
                passed=bool((diff>=-cfg['permutation_nondecreasing_atol']).all() and all(r['change']>0 for r in rr[1:]))
                check('index_monotonicity',key,passed,diff.min())
                trends.append(dict(state=state,mode=mode,seed=seed,passed=passed,min_step=float(diff.min())))
    for r in gpa_logs:check('GPA_convergence',r['key'],r['converged'],r['relative_change'])
    expected_index=len(states)*len(cfg['modes'])*len(cfg['seeds'])*(1+len(cfg['strengths']))
    expected_rigid=len(states)*2*len(cfg['seeds'])*len(cfg['rigid_strengths'])
    expected_geometric=len(states)*len(cfg['seeds'])*sum(map(len,cfg['geometry_operations'].values()))
    expected_missing=len(states)*len(cfg['seeds'])*len(cfg['geometry_operations'].get('random_missing',[]))
    keys=[(r['state'],r['kind'],r['mode'],r['seed'],r['strength']) for r in rows]
    check('row_count','scores',len(rows)==expected_index+expected_rigid+expected_geometric,len(rows))
    check('unique_rows','scores',len(set(keys))==len(rows),len(set(keys)))
    check('row_count','geometry',len(geometry)==len(states)+expected_rigid+expected_geometric,len(geometry))
    missing=[r for r in rows if r['kind']=='random_missing']
    check('missing_policy','geometry_only',len(missing)==expected_missing and all(
        r['score'] is None and r['change'] is None and r['status']=='not_applicable_missing_coordinates'
        for r in missing),len(missing))
    check('finite_scores','scored',all(np.isfinite(r['score']) and np.isfinite(r['change'])
        for r in rows if r['status']=='scored'),sum(r['status']=='scored' for r in rows))
    # A negative E_COMP change with degraded geometry is reported, not hidden.
    warnings=[]
    lookup={(r['state'],r['kind'],r['mode'],r['seed'],r['strength']):r for r in rows}
    for g in geometry:
        r=lookup.get((g['state'],g['kind'],g['mode'],g['seed'],g['strength']))
        if r and r['change'] is not None and r['change']<0 and g['symmetric_mean'] and g['symmetric_mean']>1e-8:
            warnings.append(dict(state=g['state'],kind=g['kind'],seed=g['seed'],strength=g['strength'],
                change=r['change'],coverage=g['coverage'],distance=g['symmetric_mean']))
    reference_data=dict(version=cfg['version'],config=cfg,K=k,N=len(subjects),D=int(np.prod(data[states[0]].shape[1:])),
        reference_variance=reference,raw_v2_reference_variance=raw_reference,data_sha256=digest.hexdigest(),sources=sources,
        reference_rule='concatenate per-state GPA aligned unperturbed cohorts',
        numpy=np.__version__,code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).parent/n for n in ['preprocessing.py','validate_protocol_v3.py','validate_protocol.py','index_protocol.py','ecomp.py','metrics.py','perturb.py']]})
    dump(out/'reference.json',reference_data)
    np.savez_compressed(out/'transforms.npz',**transforms)
    np.savez_compressed(out/'spectra.npz',**spectra)
    table(out/'baseline_comparison.csv',baseline_rows);table(out/'scale_ablation.csv',scales)
    table(out/'monotonicity.csv',trends);table(out/'acceptance_checks.csv',checks)
    dump(out/'geometry_score_disagreements.json',warnings)
    failures=[c for c in checks if not c['passed']]
    summary=dict(version=cfg['version'],rows=len(rows),geometry_rows=len(geometry),
        scored_rows=sum(r['score'] is not None for r in rows),missing_geometry_only_rows=sum(r['score'] is None for r in rows),
        fixed_K=k,reference_variance=reference,
        all_protocol_checks_passed=not failures,failures=failures,
        max_iterations_used=max(r['iterations'] for r in gpa_logs),
        max_common_rigid_delta=max(abs(r['change']) for r in rows if r['kind']=='rigid' and r['mode']=='common'),
        max_independent_rigid_delta=max(abs(r['change']) for r in rows if r['kind']=='rigid' and r['mode']=='independent'),
        max_common_index_delta=max(abs(r['change']) for r in rows if r['kind']=='index' and r['mode']=='common'),
        min_independent_signal_retention=min(c['value'] for c in checks if c['check']=='index_signal_retained'),
        max_rigid_geometry_distance_over_tolerance=max(c['value'] for c in checks if c['check']=='geometry_rigid_invariance'),
        acceptance_scope='v3 processing checks; raw baseline unchanged condition assessed separately in baseline_comparison.csv',
        raw_to_aligned_baseline_change=[dict(state=r['state'],delta=r['alignment_effect_fixed_v3_reference']) for r in baseline_rows],
        geometry_score_disagreement_count=len(warnings),
        limits=['GPA uses candidate correspondences; regression is empirical, not universal invariance',
                'baseline raw-to-aligned score changes are not treated as zero; baseline is reprocessed',
                'scale preserved by user decision; missing data is geometry-only and NOT scored',
                'geometry computed for configured subject only, group score uses all 15 identities',
                'negative score with degraded geometry is possible; not validated as neural training loss'])
    dump(out/'summary.json',summary)
    plot(rows,states,cfg,out)
    print(json.dumps(summary,ensure_ascii=True,indent=2),flush=True)
    return 0 if not failures else 1


def plot(rows,states,cfg,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,mode in zip(axes,cfg['modes']):
        for state in states:
            grid=[0.]+cfg['strengths']
            values=np.array([[next(r['change'] for r in rows if r['state']==state and r['kind']=='index' and r['mode']==mode and r['seed']==seed and r['strength']==s) for s in grid] for seed in cfg['seeds']])
            ax.plot(np.array(grid)*100,values.mean(0),marker='o',label=state)
            ax.fill_between(np.array(grid)*100,values.min(0),values.max(0),alpha=.12)
        ax.set(title=mode,xlabel='Permuted vertices (%)',ylabel='Delta E_COMP after rigid GPA');ax.grid(alpha=.25)
        if mode=='common':ax.set_ylim(-1e-8,1e-8)
    axes[-1].legend(fontsize=8);fig.savefig(out/'permutation_regression.png',dpi=180);fig.savefig(out/'permutation_regression.pdf');plt.close(fig)
    pose=[r for r in rows if r['kind']=='rigid']
    fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    for mode in ('common','independent'):
        rr=[r for r in pose if r['mode']==mode]
        ax.scatter([r['raw_fixed_v3_change'] for r in rr],[r['change'] for r in rr],label=mode,alpha=.6)
    ax.set(xlabel='Delta before GPA (fixed v3 reference)',ylabel='Delta after GPA',title='Pose regression: 7 states x 3 seeds x 3 strengths')
    ax.legend();ax.grid(alpha=.25);fig.savefig(out/'pose_regression.png',dpi=180);plt.close(fig)
