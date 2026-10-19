"""S3.1/S3.2: full-cohort geometry, joined long table, seed-errorbar curves.

Reuse v3 scores and saved GPA transforms after hash verification. Geometry
covers every subject. Resume only checkpoints with an identical manifest.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

from astface_loader import ASTFaceReader,parse_subjects
from ecomp import covariance_spectrum,ecomp
from geometry_batch import evaluate_batch_geometry,identical_geometry
from index_protocol import permutation,perturb_group,sample_seed
from perturb import coordinate_noise,smooth_local_deformation,random_missing
from preprocessing import match_group_frame


METRICS=['coverage','reference_to_candidate_mean','candidate_to_reference_mean',
    'symmetric_mean','reference_to_candidate_p95','normal_reversals','comparable_faces',
    'normal_reversal_rate','valid_vertex_fraction','usable_faces','degenerate_faces',
    'reference_area','candidate_area','tolerance','distance_over_tolerance']
PANELS=[('index','common'),('index','single'),('index','independent'),
    ('rigid','common'),('rigid','independent'),('coordinate_noise','independent'),
    ('smooth_local_deformation','independent'),('random_missing','independent')]
TITLES=['Common index','Single-subject index','Independent index','Common rigid',
    'Independent rigid','Coordinate noise','Smooth deformation','Missing vertices']
WORK={}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write_csv(path,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def write_json(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def experiment_id(row):
    return f"{row['state']}_{row['kind']}_{row['mode']}_{int(row['seed'])}_{float(row['strength']):g}"


def transform_key(row):
    state,kind,mode=row['state'],row['kind'],row['mode']
    if kind=='baseline':return state+'_baseline'
    if kind in ('index','rigid'):return experiment_id(row)
    return f"{state}_{kind}_{int(row['seed'])}_{float(row['strength']):g}"


def build_tasks(scores,cfg):
    keys=[experiment_id(r) for r in scores]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate v3 score keys')
    tasks=[]
    for state in cfg['states']:
        for seed in cfg['seeds']:
            base=next(r for r in scores if r['state']==state and r['kind']=='index'
                      and r['mode']=='common' and int(r['seed'])==seed and float(r['strength'])==0)
            base=dict(base,kind='baseline',mode='none')
            tasks.append(base)
    tasks.extend(r for r in scores if float(r['strength'])>0)
    expected={(state,kind,mode,seed,float(level)) for state in cfg['states']
        for kind,mode in PANELS for seed in cfg['seeds'] for level in (
            cfg['strengths'] if kind=='index' else cfg['rigid_strengths'] if kind=='rigid'
            else cfg['geometry_operations'][kind])}
    actual={(r['state'],r['kind'],r['mode'],int(r['seed']),float(r['strength'])) for r in tasks if r['kind']!='baseline'}
    if actual!=expected:raise ValueError('score grid does not match configuration')
    return sorted(tasks,key=lambda r:(cfg['states'].index(r['state']),r['kind'],r['mode'],int(r['seed']),float(r['strength'])))


def init_worker(cache,source,cfg):
    WORK.update(cache=Path(cache),cfg=cfg,subjects=parse_subjects(cfg['subjects']),arrays={})
    with np.load(Path(source)/'transforms.npz') as z:WORK['transforms']={k:z[k] for k in z.files}
    with np.load(Path(source)/'spectra.npz') as z:WORK['spectra']={k:z[k] for k in z.files}


def state_arrays(state):
    if state not in WORK['arrays']:
        WORK['arrays'][state]=tuple(np.load(WORK['cache']/f'{state}_{name}.npy',mmap_mode='r')
                                   for name in ('raw','baseline','faces'))
    return WORK['arrays'][state]


def process_experiment(row):
    cfg=WORK['cfg'];subjects=WORK['subjects'];tr=WORK['transforms']
    state,kind,mode=row['state'],row['kind'],row['mode']
    seed=int(row['seed']);strength=float(row['strength']);key=transform_key(row)
    x,b,f=state_arrays(state);identities=[f'{s}/{state}' for s in subjects]
    masks=[None]*len(subjects);candidate_faces=[f]*len(subjects)
    derived_seeds=[None]*len(subjects)
    if kind=='baseline':
        y=x;aligned=b;frame='native_identical_surface'
    elif kind=='index':
        y=perturb_group(x,identities,mode=mode,strength=strength,seed=seed,target=f"{cfg['single_subject']}/{state}")
        for i,identity in enumerate(identities):
            active=mode!='single' or subjects[i]==cfg['single_subject']
            derived_seeds[i]=seed if mode=='common' else sample_seed(seed,identity)
            p=permutation(x.shape[1],strength,derived_seeds[i]) if active else np.arange(x.shape[1])
            inverse=np.argsort(p);candidate_faces[i]=inverse[f]
            if not np.array_equal(y[i],x[i,p]) or not np.array_equal(p[candidate_faces[i]],f):
                raise ValueError('index perturbation changed triangle geometry')
        aligned=np.einsum('nvi,nij->nvj',y-tr[key+'_centers'][:,None],tr[key+'_rotations'])
        frame='native_identical_surface'
    elif kind=='rigid':
        y=np.einsum('nvi,nij->nvj',x,tr[key+'_injected_rotations'])+tr[key+'_injected_translations'][:,None]
        aligned=np.einsum('nvi,nij->nvj',y-tr[key+'_centers'][:,None],tr[key+'_rotations'])
        frame='GPA_then_common_frame'
    else:
        fn={'coordinate_noise':coordinate_noise,'smooth_local_deformation':smooth_local_deformation,
            'random_missing':random_missing}[kind]
        derived_seeds=[sample_seed(seed,identity+'/'+kind) for identity in identities]
        pp=[fn(v,f,strength,seed=s) for v,s in zip(x,derived_seeds)]
        y=np.array([p.vertices for p in pp],float);masks=[p.valid_mask for p in pp]
        aligned=None if kind=='random_missing' else np.einsum('nvi,nij->nvj',y-tr[key+'_centers'][:,None],tr[key+'_rotations'])
        frame='native' if aligned is None else 'GPA_then_common_frame'
    score_error=None
    if aligned is not None:
        spectrum=covariance_spectrum(aligned)
        expected=WORK['spectra'][key]
        if not np.allclose(spectrum,expected,atol=1e-8,rtol=1e-10):raise ValueError('saved spectrum mismatch '+key)
        value=ecomp(spectrum,n_components=int(row['K']),reference_variance=float(row['reference_variance']),delta=float(row['delta']))
        score_error=abs(value-float(row['score']))
        if score_error>1e-8:raise ValueError('saved score mismatch '+key)
    candidate=match_group_frame(aligned,b.mean(axis=0)) if frame=='GPA_then_common_frame' else y
    detail=[]
    for i,subject in enumerate(subjects):
        ref=b[i] if frame=='GPA_then_common_frame' else x[i]
        tolerance=float(np.linalg.norm(np.ptp(ref,axis=0)))*cfg['geometry_tolerance_diagonal_fraction']
        if kind in ('baseline','index'):
            result=identical_geometry(ref,f,tolerance=tolerance,seed=seed)
        else:
            result=evaluate_batch_geometry(ref,f,candidate[i],candidate_faces[i],valid_mask=masks[i],
                tolerance=tolerance,sample_count=cfg['geometry_surface_samples'],seed=seed)
        result['distance_over_tolerance']=result['symmetric_mean']/tolerance if result['symmetric_mean'] is not None else None
        detail.append(dict(experiment_id=experiment_id(row),state=state,kind=kind,mode=mode,
            strength=strength,subject=subject,frame=frame,derived_seed=derived_seeds[i],
            score_transform_key=key if aligned is not None else '',**result))
    return dict(row=row,geometry=detail,score_replay_error=score_error)


def aggregate(result):
    row=dict(result['row']);detail=result['geometry']
    row.update(experiment_id=experiment_id(row),geometry_subject_count=len(detail),
        geometry_frame=detail[0]['frame'],score_replay_error=result['score_replay_error'])
    for metric in METRICS:
        values=[d[metric] for d in detail if d[metric] is not None]
        row[metric+'_n_valid']=len(values)
        for label,fn in [('mean',np.mean),('min',np.min),('max',np.max)]:
            row[metric+'_'+label]=float(fn(values)) if values else None
    row['normal_reversals_total']=sum(d['normal_reversals'] for d in detail)
    row['comparable_faces_total']=sum(d['comparable_faces'] for d in detail)
    row['normal_reversal_rate_pooled']=row['normal_reversals_total']/row['comparable_faces_total'] if row['comparable_faces_total'] else None
    change=None if row['change']=='' else float(row['change'])
    degraded=[d['coverage']<1-1e-6 or (d['symmetric_mean'] is not None and d['symmetric_mean']>1e-8)
              or d['normal_reversals']>0 for d in detail]
    row['geometry_degraded_subjects']=sum(degraded)
    row['ecomp_geometry_disagreement']=change is not None and change < -1e-8 and any(degraded)
    row['baseline_experiment_id']=f"{row['state']}_baseline_none_{int(row['seed'])}_0"
    return row


def make_curve_table(rows,cfg):
    result=[]
    metrics=['score','change','coverage_mean','distance_over_tolerance_mean','normal_reversal_rate_mean']
    baseline={(r['state'],int(r['seed'])):r for r in rows if r['kind']=='baseline'}
    for state in cfg['states']:
        for kind,mode in PANELS:
            selected=[r for r in rows if r['state']==state and r['kind']==kind and r['mode']==mode]
            levels=sorted({float(r['strength']) for r in selected})
            for level in [0.]+levels:
                cohort=[baseline[state,seed] for seed in cfg['seeds']] if level==0 else [r for r in selected if float(r['strength'])==level]
                if len(cohort)!=len(cfg['seeds']):raise ValueError('incomplete seeds')
                for metric in metrics:
                    values=[float(r[metric]) for r in cohort if r[metric] is not None and r[metric]!='']
                    if kind=='random_missing' and metric in ('score','change'):values=[]
                    result.append(dict(state=state,kind=kind,mode=mode,strength=level,metric=metric,
                        n_seeds=len(values),mean=float(np.mean(values)) if values else None,
                        std=float(np.std(values,ddof=1)) if len(values)>1 else None,
                        minimum=min(values) if values else None,maximum=max(values) if values else None,
                        errorbar='sample standard deviation across perturbation seeds; not CI'))
    return result


def plots(curves,cfg,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=plt.get_cmap('tab10').colors
    def draw(ax,kind,mode,metric,scale=1.):
        for i,state in enumerate(cfg['states']):
            rr=sorted([r for r in curves if r['state']==state and r['kind']==kind and r['mode']==mode and r['metric']==metric and r['mean'] is not None],key=lambda r:r['strength'])
            if not rr:continue
            x=np.array([r['strength'] for r in rr])*(1 if kind=='rigid' else 100)
            ax.errorbar(x,np.array([r['mean'] for r in rr])*scale,
                        yerr=np.array([r['std'] for r in rr])*scale,
                        marker='o',markersize=3,linewidth=1.1,capsize=3,elinewidth=.9,
                        label=state,color=colors[i])
            ax.set_xticks(x)
        ax.grid(alpha=.22)
        ax.set_xlabel('Rigid strength' if kind=='rigid' else 'Vertices (%)' if kind in ('index','random_missing') else 'Amplitude / box diagonal (%)',fontsize=8)
        ax.tick_params(labelsize=8)
    for metric,name,label in [('score','ecomp_curves','E_COMP'),('change','ecomp_delta_curves','Delta E_COMP')]:
        fig,axes=plt.subplots(2,4,figsize=(17,8),constrained_layout=True)
        for ax,(kind,mode),title in zip(axes.flat,PANELS,TITLES):
            ax.set_title(title,fontsize=11)
            if kind=='random_missing':
                ax.text(.5,.5,'E_COMP: not applicable\nMissing coordinates are not imputed.\nSee geometry curves.',ha='center',va='center',transform=ax.transAxes)
                ax.set_axis_off();continue
            draw(ax,kind,mode,metric);ax.set_ylabel(label)
            if metric=='change' and ((kind=='index' and mode=='common') or kind=='rigid'):ax.set_ylim(-1e-8,1e-8)
        axes.flat[0].legend(fontsize=8,ncol=2)
        fig.suptitle('15 identities per state; mean +/- SD over 3 perturbation seeds',fontsize=13)
        fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'));plt.close(fig)
    fig,axes=plt.subplots(3,8,figsize=(28,10),constrained_layout=True)
    for j,((kind,mode),title) in enumerate(zip(PANELS,TITLES)):
        for i,(metric,label,scale) in enumerate([
            ('coverage_mean','Mean subject coverage (%)',100.),
            ('distance_over_tolerance_mean','Mean distance / tolerance',1.),
            ('normal_reversal_rate_mean','Mean normal reversal (%)',100.)]):
            ax=axes[i,j];draw(ax,kind,mode,metric,scale)
            if i==0:ax.set_title(title,fontsize=10)
            if j==0:ax.set_ylabel(label,fontsize=9)
            if kind in ('index','rigid'):
                ax.set_ylim((99.9,100.1) if i==0 else (-1e-7,1e-7))
            elif i==0:ax.set_ylim(0,102)
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('Geometry over all 15 identities: subject mean, then mean +/- seed SD; tolerance = 0.1% reference box diagonal',fontsize=13)
    fig.savefig(out/'geometry_curves.png',dpi=180);fig.savefig(out/'geometry_curves.pdf');plt.close(fig)


def run(args):
    source=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cache=out/'cache';cache.mkdir(exist_ok=True);checkpoints=out/'checkpoints';checkpoints.mkdir(exist_ok=True)
    ref=json.loads((source/'reference.json').read_text(encoding='utf-8'));cfg=ref['config']
    for name,digest in ref['code_sha256'].items():
        if sha(Path(__file__).parent/name)!=digest:raise ValueError('v3 source code changed: '+name)
    if not json.loads((source/'summary.json').read_text(encoding='utf-8'))['all_protocol_checks_passed']:
        raise ValueError('v3 processing checks failed')
    scores=read_csv(source/'scores.csv');tasks=build_tasks(scores,cfg)
    subjects=parse_subjects(cfg['subjects'])
    manifest=dict(version='stage3-v3-all-subjects',source=str(source.resolve()),config=cfg,
        geometry_subjects=subjects,subjects_per_experiment=len(subjects),experiments=len(tasks),
        data_sha256=ref['data_sha256'],source_sha256={n:sha(source/n) for n in
            ['reference.json','summary.json','scores.csv','geometry.csv','transforms.npz','spectra.npz']},
        code_sha256={n:sha(Path(__file__).parent/n) for n in
            ['stage3_analysis.py','geometry_batch.py','astface_loader.py','metrics.py','index_protocol.py','preprocessing.py','perturb.py','ecomp.py']},
        aggregation='equal weight per subject; error bars sample SD across seeds within each state',
        geometry_frames='native for baseline/index/missing; GPA_then_common_frame for rigid/noise/deformation',
        missing_policy='geometry_only; score blank; missing rows retained',
        index_geometry='exact identity certificate after inverse-remapping triangle indices; sample_count=0',
        distance_method='same sampled triangle distance metric as v3; conservative radius-binned acceleration',
        negative_score_tolerance=1e-8,geometry_distance_tolerance=1e-8)
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if old!=manifest:raise ValueError('checkpoint provenance changed; use a new output directory')
    write_json(out/'manifest.json',manifest)
    digest=hashlib.sha256()
    with ASTFaceReader(args.data_dir) as reader,np.load(source/'transforms.npz') as tr:
        for state in cfg['states']:
            shapes=[];faces=None
            for subject in subjects:
                mesh=reader.read_sample(subject,state,modalities=('mesh',)).mesh
                if faces is not None and not np.array_equal(faces,mesh.faces):raise ValueError('topology mismatch')
                faces=mesh.faces;shapes.append(mesh.vertices)
                digest.update(mesh.vertices.tobytes());digest.update(mesh.faces.tobytes())
            x=np.asarray(shapes,float)
            baseline=np.einsum('nvi,nij->nvj',x-tr[state+'_baseline_centers'][:,None],tr[state+'_baseline_rotations'])
            for name,array in [('raw',x),('baseline',baseline),('faces',faces)]:np.save(cache/f'{state}_{name}.npy',array)
            print('Prepared',state,flush=True)
    if digest.hexdigest()!=ref['data_sha256']:raise ValueError('raw data differs from validated v3')
    pending=[r for r in tasks if not (checkpoints/(experiment_id(r)+'.json')).exists()]
    print(f'Experiments {len(tasks)}; pending {len(pending)}; subjects {len(subjects)}; workers {args.workers}',flush=True)
    started=time.perf_counter();done=len(tasks)-len(pending)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(str(cache),str(source),cfg)) as pool:
            futures={pool.submit(process_experiment,r):r for r in pending}
            for future in as_completed(futures):
                result=future.result();write_json(checkpoints/(experiment_id(result['row'])+'.json'),result)
                done+=1
                if done%10==0 or done==len(tasks):print(f'Completed {done}/{len(tasks)}; elapsed {time.perf_counter()-started:.0f}s',flush=True)
    results=[json.loads((checkpoints/(experiment_id(r)+'.json')).read_text(encoding='utf-8')) for r in tasks]
    long_rows=[aggregate(r) for r in results];detail=[d for r in results for d in r['geometry']]
    if len(detail)!=len(tasks)*len(subjects):raise ValueError('incomplete geometry')
    for r in results:
        if {d['subject'] for d in r['geometry']}!=set(subjects):raise ValueError('subject coverage mismatch')
    original={(r['state'],r['kind'],r['mode'],int(r['seed']),float(r['strength'])):r
              for r in read_csv(source/'geometry.csv') if r['kind']!='baseline'}
    max_error=0.;n_matches=0
    for d in detail:
        key=(d['state'],d['kind'],d['mode'],d['seed'],d['strength'])
        if d['subject']!=cfg['geometry_subject'] or key not in original:continue
        old=original[key];n_matches+=1
        for metric in METRICS[:-1]:
            if old[metric]=='':continue
            error=abs(d[metric]-float(old[metric]));max_error=max(max_error,error)
            if not np.isclose(d[metric],float(old[metric]),atol=1e-10,rtol=1e-10):
                raise ValueError(f'v3 geometry replay mismatch: {key} {metric}')
    if n_matches!=len(original):raise ValueError('incomplete v3 geometry replay')
    curves=make_curve_table(long_rows,cfg)
    warnings=[r for r in long_rows if r['ecomp_geometry_disagreement']]
    write_csv(out/'experiments.csv',long_rows);write_csv(out/'geometry_subjects.csv',detail)
    write_csv(out/'curve_summary.csv',curves)
    write_csv(out/'disagreements.csv',warnings,fields=list(long_rows[0]))
    plots(curves,cfg,out)
    summary=dict(version=manifest['version'],s31_complete=True,s32_complete=True,
        experiments=len(tasks),baseline_experiments=sum(r['kind']=='baseline' for r in tasks),
        subjects=len(subjects),states=len(cfg['states']),geometry_rows=len(detail),
        nonbaseline_experiments=sum(r['kind']!='baseline' for r in tasks),
        missing_ecomp_rows=sum(r['status']=='not_applicable_missing_coordinates' for r in tasks),
        all_subjects_per_experiment=True,input_hash_verified=True,v3_code_hashes_verified=True,
        max_score_replay_error=max(r['score_replay_error'] or 0 for r in results),
        v3_geometry_replayed=n_matches,max_v3_geometry_absolute_difference=max_error,
        disagreements=len(warnings),
        geometry_degraded_subjects_in_disagreements=[int(r['geometry_degraded_subjects']) for r in warnings],
        limits=['7 configured states; not all 16 expressions/AUs',
            'no new independent cohort; S3.3 identity stability not completed',
            'errorbars are seed SD, not confidence intervals',
            'all geometry metrics descriptive; no self-intersection detection',
            'S3.1/S3.2 completion does not validate E_COMP as standalone training loss'])
    write_json(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=True,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',default='data/ASTFace_Public')
    p.add_argument('--source',default='code/reports/protocol_v3')
    p.add_argument('--output',default='code/reports/stage3_v3')
    p.add_argument('--workers',type=int,default=min(6,os.cpu_count() or 1))
    args=p.parse_args()
    if args.workers<1:p.error('--workers must be positive')
    run(args)


if __name__=='__main__':main()
