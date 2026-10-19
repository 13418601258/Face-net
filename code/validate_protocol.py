"""Run S2.5 real-data contrasts and S2.4 geometry checks; write review artifacts.

Use extracted AST-Face data only. Legacy perturbation NPZ files are not used:
their common seeds generate common permutations, not independent errors.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from astface_loader import ASTFaceReader,parse_subjects
from ecomp import covariance_spectrum,ecomp
from index_protocol import perturb_group,permutation,permute_mesh
from metrics import evaluate_geometry
from perturb import coordinate_noise,smooth_local_deformation,random_missing


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',default='data/ASTFace_Public')
    parser.add_argument('--config',default='code/protocol_config.json')
    parser.add_argument('--output-dir',default=None)
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    if cfg['version']=='rigid-gpa-v3':
        from validate_protocol_v3 import run
        return run(Path(args.data_dir),cfg,Path(args.output_dir or 'code/reports/protocol_v3'))
    out=Path(args.output_dir or 'code/reports/protocol_v2');out.mkdir(parents=True,exist_ok=True)
    subjects=parse_subjects(cfg['subjects']);states=cfg['states']
    data={};face_map={};sources=[];data_digest=hashlib.sha256()
    with ASTFaceReader(args.data_dir) as reader:
        for state in states:
            shapes=[]
            for subject in subjects:
                sample=reader.read_sample(subject,state,modalities=('mesh',))
                v,f=sample.mesh.vertices,sample.mesh.faces
                shapes.append(v)
                if state not in face_map:face_map[state]=f
                if not np.array_equal(f,face_map[state]):raise ValueError('inconsistent topology')
                data_digest.update(v.tobytes());data_digest.update(f.tobytes())
                sources.append(dict(subject=subject,state=state,path=sample.mesh.path))
            data[state]=np.asarray(shapes,dtype=float)
            print('Loaded',state,len(shapes),'meshes',flush=True)
    all_baseline=np.concatenate(list(data.values()))
    reference_variance=float(covariance_spectrum(all_baseline).sum())
    k=min(len(subjects)-1,all_baseline.shape[1]*3)
    reference=dict(reference_variance=reference_variance,K=k,N=len(subjects),
        D=int(np.prod(all_baseline.shape[1:])),baseline_samples=len(all_baseline),
        data_sha256=data_digest.hexdigest(),config=cfg,sources=sources,
        preprocessing='published coordinates; no per-sample scaling or alignment',
        python=platform.python_version(),numpy=np.__version__,
        interpretation='within-cohort compactness surrogate; no cross-cohort comparability claim')
    reference['code_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__).parent/name for name in
                  ['ecomp.py','model.py','index_protocol.py','metrics.py','validate_protocol.py'])}
    write_json(out/'reference.json',reference)
    rows=[];checks=[];spectra={};base_spectra={}
    score=lambda v,d=cfg['delta']:ecomp(v,n_components=k,reference_variance=reference_variance,delta=d)
    for state,x in data.items():
        identities=[f'{s}/{state}' for s in subjects];target=f"{cfg['single_subject']}/{state}"
        base=covariance_spectrum(x);base_spectra[state]=base
        for mode in cfg['modes']:
            for seed in cfg['seeds']:
                for strength in [0.]+cfg['strengths']:
                    y=x if strength==0 else perturb_group(x,identities,mode=mode,
                        strength=strength,seed=seed,target=target)
                    v=base if strength==0 else covariance_spectrum(y)
                    key=f'{state}_{mode}_{seed}_{strength:g}';spectra[key]=v
                    row=dict(state=state,mode=mode,seed=seed,strength=strength,
                        changed_fraction=(0. if strength==0 else max(2,round(x.shape[1]*strength))/x.shape[1]),
                        K=len(v),reference_variance=reference_variance,delta=cfg['delta'],
                        total_variance=float(v.sum()),score=score(v),baseline_score=score(base),
                        change=score(v)-score(base),
                        adaptive_trace_score=float(np.log(v/v.sum()+cfg['delta']).sum()),
                        raw_score=float(np.log(v+cfg['delta']).sum()))
                    for d in cfg['delta_checks']:row[f'change_delta_{d:g}']=score(v,d)-score(base,d)
                    rows.append(row)
                    if mode=='common':
                        passed=np.allclose(base,v,rtol=cfg['invariance_rtol'],atol=cfg['invariance_atol'])
                        checks.append(dict(check='common_spectrum',state=state,seed=seed,strength=strength,
                                           passed=bool(passed),max_error=float(abs(base-v).max())))
        print('Scored',state,'3 protocols x 3 seeds x 5 levels',flush=True)
    write_csv(out/'permutation_scores.csv',rows)
    np.savez_compressed(out/'spectra.npz',**spectra)
    trends=[]
    for state in states:
        for mode in cfg['modes']:
            for seed in cfg['seeds']:
                group=sorted([r for r in rows if r['state']==state and r['mode']==mode and r['seed']==seed],key=lambda r:r['strength'])
                increments=np.diff([r['score'] for r in group])
                trend=dict(state=state,mode=mode,seed=seed,
                    nondecreasing=bool((increments>=-cfg['invariance_atol']).all()),
                    min_step=float(increments.min()),max_change=float(max(abs(r['change']) for r in group)),
                    all_positive=(bool(all(r['change']>cfg['invariance_atol'] for r in group[1:])) if mode!='common' else None))
                for d in cfg['delta_checks']:
                    vals=[r[f'change_delta_{d:g}'] for r in group]
                    trend[f'nondecreasing_delta_{d:g}']=bool((np.diff(vals)>=-cfg['invariance_atol']).all())
                trends.append(trend)
    write_csv(out/'monotonicity.csv',trends)
    geometry=[]
    # Real mesh checks in two states; fixed reference-derived threshold for each.
    for state in [s for s in ['au0','happy'] if s in data]:
        v=data[state][0];f=face_map[state]
        threshold=cfg['geometry_tolerance_diagonal_fraction']*float(np.linalg.norm(np.ptp(v,axis=0)))
        cases=[('baseline',v,f,None)]
        pv,pf=permute_mesh(v,f,permutation(len(v),.25,0));cases.append(('shuffle25',pv,pf,None))
        for name,func,amount in [('noise',coordinate_noise,.003),('smooth',smooth_local_deformation,.03),('missing',random_missing,.2)]:
            p=func(v,f,amount,seed=0);cases.append((name,p.vertices,p.faces,p.valid_mask))
        reversed_f=f.copy();reversed_f[:100]=reversed_f[:100,::-1]
        cases.append(('reverse100',v,reversed_f,None))
        R=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);t=np.array([2.,3,4])
        restored=((v@R.T+t)-t)@R
        cases.append(('rigid_inverse_known',restored,f,None))
        for name,vv,ff,mask in cases:
            result=evaluate_geometry(v,f,vv,ff,valid_mask=mask,tolerance=threshold,
                                     sample_count=cfg['geometry_surface_samples'],seed=0)
            geometry.append(dict(subject=subjects[0],state=state,case=name,**result))
            print('Geometry',state,name,'coverage',round(result['coverage'],4),flush=True)
    write_csv(out/'geometry_checks.csv',geometry)
    geometry_pass=bool(geometry)
    for state in {r['state'] for r in geometry}:
        by={r['case']:r for r in geometry if r['state']==state}
        for name in ['baseline','shuffle25','rigid_inverse_known']:
            r=by[name];geometry_pass &= r['coverage']==1 and r['symmetric_mean']<1e-8 and r['normal_reversals']==0
        geometry_pass &= by['reverse100']['normal_reversals']==100
        geometry_pass &= by['noise']['symmetric_mean']>by['baseline']['symmetric_mean']+1e-8
        geometry_pass &= by['missing']['coverage']<by['baseline']['coverage']
    failures=[t for t in trends if t['mode']!='common' and (not t['nondecreasing'] or not t['all_positive'])]
    summary=dict(protocol='v2',groups=len(rows),K=k,reference_variance=reference_variance,
        common_invariance_passed=all(c['passed'] for c in checks),
        common_max_eigenvalue_error=max(c['max_error'] for c in checks),
        fixed_K_passed=all(r['K']==k for r in rows),
        perturbation_trends_tested=sum(t['mode']!='common' for t in trends),
        perturbation_trend_failures=failures,geometry_checks_passed=bool(geometry_pass),
        delta_sensitivity_failures=[t for t in trends if t['mode']!='common' and any(not t[f'nondecreasing_delta_{d:g}'] for d in cfg['delta_checks'])],
        limits=['15 subjects, 7 states, 3 seeds; not independent external validation',
                'single protocol perturbs ast001 only; additional target identities remain untested',
                'geometry coverage is sampled; normal reversals are not self-intersections',
                'legacy NPZ seeds are common; v2 permutations generated in memory',
                'missing-coordinate E_COMP undefined: no implicit filling/dropping',
                'independent poses need an explicit alignment protocol before PCA'])
    write_json(out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,mode in zip(axes,cfg['modes']):
        for state in states:
            grid=[0.]+cfg['strengths']
            values=np.array([[next(r['change'] for r in rows if r['state']==state and r['mode']==mode and r['seed']==seed and r['strength']==s) for s in grid] for seed in cfg['seeds']])
            ax.plot(np.array(grid)*100,values.mean(axis=0),marker='o',label=state)
            ax.fill_between(np.array(grid)*100,values.min(axis=0),values.max(axis=0),alpha=.12)
        ax.set(title=mode,xlabel='Permuted vertices (%)',ylabel='E_COMP minus baseline')
        ax.grid(alpha=.25)
        if mode=='common':ax.set_ylim(-1e-8,1e-8)
    axes[-1].legend(fontsize=8)
    fig.suptitle('Fixed K=14; frozen reference variance; shading = min/max of 3 seeds')
    fig.savefig(out/'monotonicity.png',dpi=180);fig.savefig(out/'monotonicity.pdf');plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)
    return 0 if (summary['common_invariance_passed'] and summary['fixed_K_passed'] and not failures and geometry_pass) else 1


if __name__=='__main__':
    raise SystemExit(main())
