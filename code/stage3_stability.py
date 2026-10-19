"""S3.3 seed and identity stability checks with explicit K accounting.

The existing 15-identity v3 results are reused verbatim (K=14). New
identities are evaluated as a five-subject cohort (K=4) and as a combined
20-subject cohort (K=19). Absolute E_COMP values across these cohorts are
never compared; comparisons use baseline-centered delta divided by K.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from astface_loader import ASTFaceReader,parse_subjects
from ecomp import covariance_spectrum,ecomp
from index_protocol import perturb_group,sample_seed
from perturb import coordinate_noise,smooth_local_deformation,random_missing
from preprocessing import rigid_gpa,apply_rigid_group

STATES=['au0','angry','disgust','fear','happy','sad','surprise']
SEEDS=[0,1,2]
INDEX_LEVELS=[.01,.05,.1,.25]
RIGID_LEVELS=[.25,.5,1.]
GEOM_LEVELS={'coordinate_noise':[.0001,.0005,.001,.003],
             'smooth_local_deformation':[.001,.003,.01,.03],
             'random_missing':[.01,.05,.1,.2]}
SCORED_PANELS=[('index','common',INDEX_LEVELS),('index','single',INDEX_LEVELS),
               ('index','independent',INDEX_LEVELS),('rigid','common',RIGID_LEVELS),
               ('rigid','independent',RIGID_LEVELS),
               ('coordinate_noise','independent',GEOM_LEVELS['coordinate_noise']),
               ('smooth_local_deformation','independent',GEOM_LEVELS['smooth_local_deformation'])]
ALL_PANELS=SCORED_PANELS+[('random_missing','independent',GEOM_LEVELS['random_missing'])]


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def csv_write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def json_write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def ranks(values):
    order=np.argsort(np.argsort(values,kind='mergesort'),kind='mergesort')
    return order.astype(float)


def spearman(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(np.corrcoef(ranks(a),ranks(b))[0,1])


def monotonic(values,tolerance=1e-8):
    v=np.asarray(values,float)
    return bool((np.diff(v)>=-tolerance).all() and (v[1:]>tolerance).all())


def read_base15(source):
    """Convert S3.1 rows to the same per-seed delta/K schema."""
    out=[]
    for r in csv_read(Path(source)/'experiments.csv'):
        score=None if r['score']=='' else float(r['score'])
        change=None if r['change']=='' else float(r['change'])
        out.append(dict(cohort='base15',N=15,K=int(r['K']),reference_variance=float(r['reference_variance']),
            state=r['state'],kind=r['kind'],mode=r['mode'],seed=int(r['seed']),strength=float(r['strength']),
            score=score,change=change,delta_per_k=None if change is None else change/int(r['K']),
            status=r['status'],iterations=int(r['iterations']) if r['iterations'] else None))
    return out


def load_data(data_dir,subjects,states):
    data={};faces={};digest=hashlib.sha256();topology={}
    with ASTFaceReader(data_dir) as reader:
        for state in states:
            shapes=[];face=None
            for subject in subjects:
                mesh=reader.read_sample(subject,state,modalities=('mesh',)).mesh
                if face is None:face=mesh.faces
                if not np.array_equal(face,mesh.faces):raise ValueError(f'topology mismatch {state}/{subject}')
                shapes.append(np.asarray(mesh.vertices,float));digest.update(mesh.vertices.tobytes());digest.update(mesh.faces.tobytes())
            data[state]=np.asarray(shapes);faces[state]=face
            topology[state]=hashlib.sha256(face.tobytes()).hexdigest()
    return data,faces,digest.hexdigest(),topology


def score_cohort(data,faces,subjects,cohort,*,states=STATES):
    n=len(subjects);k=n-1
    identities_by_state={state:[f'{s}/{state}' for s in subjects] for state in states}
    baselines={};logs=[]
    for state in states:
        result=rigid_gpa(data[state],max_iterations=100,tolerance=1e-9)
        if not result.converged:raise ValueError(f'GPA did not converge {cohort}/{state}')
        baselines[state]=result;logs.append(dict(cohort=cohort,state=state,iterations=result.iterations,
            relative_change=result.relative_change,objective=result.objective,converged=result.converged))
    reference=float(covariance_spectrum(np.concatenate([b.aligned for b in baselines.values()])).sum())
    rows=[]
    def append(state,kind,mode,seed,strength,aligned,status='scored',raw_score=None):
        if aligned is None:
            rows.append(dict(cohort=cohort,N=n,K=k,reference_variance=reference,state=state,kind=kind,mode=mode,
                seed=seed,strength=strength,score=None,change=None,delta_per_k=None,status=status,iterations=None));return
        spec=covariance_spectrum(aligned);value=ecomp(spec,n_components=k,reference_variance=reference,delta=1e-9)
        baseline=ecomp(covariance_spectrum(baselines[state].aligned),n_components=k,reference_variance=reference,delta=1e-9)
        rows.append(dict(cohort=cohort,N=n,K=k,reference_variance=reference,state=state,kind=kind,mode=mode,
            seed=seed,strength=strength,score=value,change=value-baseline,delta_per_k=(value-baseline)/k,
            status=status,iterations=logs[-1]['iterations']))
    for state in states:
        x=data[state];b=baselines[state];ids=identities_by_state[state]
        for seed in SEEDS:append(state,'baseline','none',seed,0.,b.aligned)
        for kind,mode,levels in ALL_PANELS:
            for seed in SEEDS:
                for strength in levels:
                    key=f'{state}_{kind}_{mode}_{seed}_{strength:g}'
                    if kind=='index':
                        y=perturb_group(x,ids,mode=mode,strength=strength,seed=seed,target=ids[0])
                        a=rigid_gpa(y,max_iterations=100,tolerance=1e-9)
                        append(state,kind,mode,seed,strength,a.aligned)
                    elif kind=='rigid':
                        y,_,_=apply_rigid_group(x,ids,mode=mode,seed=seed,strength=strength)
                        a=rigid_gpa(y,max_iterations=100,tolerance=1e-9)
                        append(state,kind,mode,seed,strength,a.aligned)
                    else:
                        fn={'coordinate_noise':coordinate_noise,'smooth_local_deformation':smooth_local_deformation,
                            'random_missing':random_missing}[kind]
                        pp=[fn(v,faces[state],strength,seed=sample_seed(seed,identity+'/'+kind)) for v,identity in zip(x,ids)]
                        y=np.array([p.vertices for p in pp],float)
                        if kind=='random_missing':append(state,kind,mode,seed,strength,None,'not_applicable_missing_coordinates')
                        else:
                            a=rigid_gpa(y,max_iterations=100,tolerance=1e-9)
                            append(state,kind,mode,seed,strength,a.aligned)
                    if len(rows)%30==0:print('Computed',cohort,len(rows),flush=True)
    return rows,logs,reference


def seed_table(rows):
    grouped={}
    for r in rows:
        grouped.setdefault((r['cohort'],r['N'],r['K'],r['state'],r['kind'],r['mode'],r['strength']),[]).append(r)
    out=[]
    for key,values in grouped.items():
        vals=[r['delta_per_k'] for r in values if r['delta_per_k'] is not None]
        mean=float(np.mean(vals)) if vals else None;sd=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        out.append(dict(cohort=key[0],N=key[1],K=key[2],state=key[3],kind=key[4],mode=key[5],strength=key[6],
            n_seeds=len(vals),mean_delta_per_K=mean,seed_sd_per_K=sd,
            seed_cv=None if not mean else abs(sd/mean),status='scored' if vals else 'not_applicable_missing_coordinates'))
    out.sort(key=lambda r:(r['cohort'],r['state'],r['kind'],r['mode'],r['strength']))
    baseline_lookup={(r['cohort'],r['state']):r for r in out if r['kind']=='baseline'}
    # Add signal/noise metrics after the mean table exists.
    for r in out:
        earlier=[x for x in out if x['cohort']==r['cohort'] and x['state']==r['state'] and x['kind']==r['kind']
                 and x['mode']==r['mode'] and x['strength']<r['strength'] and x['mean_delta_per_K'] is not None]
        previous=max(earlier,key=lambda x:x['strength']) if earlier else baseline_lookup.get((r['cohort'],r['state']))
        if previous and r['mean_delta_per_K'] is not None:
            step=r['mean_delta_per_K']-previous['mean_delta_per_K'];noise=r['seed_sd_per_K'] or 0.
            r['step_from_previous_mean_per_K']=step
            r['seed_sd_over_step']=None if abs(step)<1e-15 else noise/abs(step)
            r['signal_gt_seed_noise']=bool(abs(step)>noise)
            r['signal_gt_half_seed_noise']=bool(abs(step)>2*noise)
        else:
            r['step_from_previous_mean_per_K']=None;r['seed_sd_over_step']=None
            r['signal_gt_seed_noise']=None;r['signal_gt_half_seed_noise']=None
        r['signal_check_applicable']=bool(
            r['strength']>0 and
            ((r['kind']=='index' and r['mode'] in ('single','independent')) or
             r['kind'] in ('coordinate_noise','smooth_local_deformation')) and
            r['mean_delta_per_K'] is not None and r['step_from_previous_mean_per_K'] is not None)
    return out


def identity_table(all_rows,base_rows):
    cohorts=['base15','new5','all20'];source={'base15':base_rows,'new5':all_rows[:0]}
    pairs=[('base15','new5'),('base15','all20')]
    states=sorted(set(r['state'] for r in base_rows) & set(r['state'] for r in all_rows))
    out=[]
    for left,right in pairs:
        for state in states:
            for kind,mode,levels in SCORED_PANELS:
                lv=[];rv=[]
                for strength in levels:
                    l=[r['delta_per_k'] for r in base_rows if r['cohort']==left and r['state']==state and r['kind']==kind and r['mode']==mode and r['strength']==strength]
                    rr=[r['delta_per_k'] for r in all_rows if r['cohort']==right and r['state']==state and r['kind']==kind and r['mode']==mode and r['strength']==strength]
                    if len(l)!=3 or len(rr)!=3:raise ValueError('incomplete identity comparison')
                    lv.append(np.mean(l));rv.append(np.mean(rr))
                rho=spearman(lv,rv);lm=monotonic(lv);rm=monotonic(rv)
                scale=max(max(abs(np.asarray(lv))),max(abs(np.asarray(rv))),1e-12)
                rmse=float(np.sqrt(np.mean(((np.asarray(lv)-np.asarray(rv))/scale)**2)))
                signs=bool(np.all(np.sign(lv)==np.sign(rv)))
                signal_panel=((kind=='index' and mode in ('single','independent')) or
                              kind in ('coordinate_noise','smooth_local_deformation'))
                out.append(dict(left_cohort=left,right_cohort=right,state=state,kind=kind,mode=mode,K_left=14 if left=='base15' else None,
                    K_right=4 if right=='new5' else 19,levels=';'.join(map(str,levels)),
                    left_delta_per_K=';'.join(f'{x:.12g}' for x in lv),right_delta_per_K=';'.join(f'{x:.12g}' for x in rv),
                    spearman_rho=rho,shape_nrmse=rmse,sign_consistent=signs,left_monotonic=lm,right_monotonic=rm,
                    applicable=bool(signal_panel and rho is not None),
                    identity_stable=bool(signal_panel and rho is not None and rho>=.8 and signs and lm and rm)))
    return out


def stability_summary(seed,identity,rows,logs,ref_info):
    applicable=[x for x in identity if x['applicable']]
    applicable_seed=[x for x in seed if x['signal_check_applicable']]
    signal=[x for x in applicable_seed if x['signal_gt_seed_noise'] is True]
    stronger=[x for x in applicable_seed if x['signal_gt_half_seed_noise'] is True]
    return dict(version='stage3-s3.3-v1',cohorts=ref_info,seed_rows=len(seed),identity_comparisons=len(identity),
        seed_signal_checks=len(signal),seed_signal_gt_noise_fraction=len(signal)/max(1,len(applicable_seed)),
        seed_signal_gt_half_noise_fraction=len(stronger)/max(1,len(applicable_seed)),
        identity_applicable_comparisons=len(applicable),identity_stable_count=sum(x['identity_stable'] for x in applicable),
        identity_stable_fraction=sum(x['identity_stable'] for x in applicable)/max(1,len(applicable)),
        identity_rho_min=min(x['spearman_rho'] for x in applicable),identity_rho_median=float(np.median([x['spearman_rho'] for x in applicable])),
        criteria=dict(seed_signal_gt_noise='adjacent mean step > seed SD; descriptive, not a universal pass gate',
                      identity_stable='Spearman rho >= 0.8, same signs, and both cohort curves nondecreasing',
                      comparison='delta E_COMP divided by native K; absolute E_COMP not compared across N'),
        limits=['new5 native K=4 is rank-limited and noisier than base15 K=14',
            'all20 native K=19 is a new reference scale and is not mixed with v3 absolute scores',
            'three seeds are perturbation repeats, not independent subjects',
            'identity check covers seven configured states and the five added identities',
            'this is S3.3 stability evidence, not validation of a neural training loss'])


def plot(seed,identity,out):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    colors=plt.get_cmap('tab10').colors
    cohorts=['base15','new5','all20']
    panels=[('index','independent','Index independent'),('index','single','Index single'),
            ('coordinate_noise','independent','Coordinate noise'),('smooth_local_deformation','independent','Smooth deformation')]
    fig,axes=plt.subplots(4,3,figsize=(15,16),constrained_layout=True)
    for i,(kind,mode,title) in enumerate(panels):
        for j,cohort in enumerate(cohorts):
            ax=axes[i,j]
            for si,state in enumerate(STATES):
                rr=sorted([r for r in seed if r['cohort']==cohort and r['state']==state and r['kind']==kind and r['mode']==mode and r['mean_delta_per_K'] is not None],key=lambda r:r['strength'])
                if not rr:continue
                x=[r['strength']*100 if kind!='rigid' else r['strength'] for r in rr]
                y=[r['mean_delta_per_K'] for r in rr];e=[r['seed_sd_per_K'] or 0 for r in rr]
                ax.errorbar(x,y,yerr=e,label=state,marker='o',markersize=3,color=colors[si],capsize=2,linewidth=.9)
            ax.set_title(f'{title} · {cohort}',fontsize=10);ax.grid(alpha=.2);ax.set_xlabel('strength (%)');ax.set_ylabel('ΔE_COMP / K')
            if i==0 and j==2:ax.legend(fontsize=7,ncol=2)
    fig.suptitle('S3.3 seed stability: mean ± sample SD across three seeds',fontsize=13)
    fig.savefig(out/'seed_stability_curves.png',dpi=180);fig.savefig(out/'seed_stability_curves.pdf');plt.close(fig)

    # Identity heatmap: use the nontrivial independent-index/noise/smooth panels.
    selected=[x for x in identity if x['right_cohort']=='new5' and x['mode']=='independent' and x['applicable']]
    labels=[f"{x['state']}\n{x['kind']}" for x in selected];vals=np.array([x['spearman_rho'] for x in selected])
    fig,ax=plt.subplots(figsize=(10,max(5,len(vals)*.18)),constrained_layout=True)
    image=ax.imshow(vals[:,None],vmin=-1,vmax=1,cmap='coolwarm',aspect='auto')
    ax.set_yticks(np.arange(len(labels)),labels,fontsize=7);ax.set_xticks([0],['base15 vs new5']);fig.colorbar(image,ax=ax,label='Spearman rho')
    for i,v in enumerate(vals):ax.text(0,i,f'{v:.2f}',ha='center',va='center',fontsize=7)
    ax.set_title('S3.3 identity curve-shape stability (normalized delta)')
    fig.savefig(out/'identity_stability_heatmap.png',dpi=180);fig.savefig(out/'identity_stability_heatmap.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',default='data/ASTFace_Public');p.add_argument('--source',default='code/reports/stage3_v3')
    p.add_argument('--output',default='code/reports/stage3_stability_v1')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((Path(args.source)/'manifest.json').read_text(encoding='utf-8'))['config']
    if cfg['states']!=STATES or cfg['seeds']!=SEEDS:raise ValueError('S3.3 expects the reviewed v3 states and seeds')
    base=read_base15(args.source)
    groups={'new5':parse_subjects('16-20'),'all20':parse_subjects('1-20')}
    group_rows=[];logs=[];refs={}
    for cohort,subjects in groups.items():
        data,faces,digest,topology=load_data(args.data_dir,subjects,STATES)
        rows,gpa,reference=score_cohort(data,faces,subjects,cohort)
        group_rows.extend(rows);logs.extend(gpa);refs[cohort]=dict(subjects=list(subjects),N=len(subjects),K=len(subjects)-1,
            reference_variance=reference,data_sha256=digest,topology_hashes=topology)
        print('Finished',cohort,'N',len(subjects),'K',len(subjects)-1,'reference',reference,flush=True)
    allrows=group_rows
    seed=seed_table(base+allrows)
    identity=identity_table(allrows,base)
    summary=stability_summary(seed,identity,base+allrows,logs,refs)
    manifest=dict(version='stage3-s3.3-v1',source=str(Path(args.source).resolve()),data_dir=str(Path(args.data_dir).resolve()),
        cohorts=refs,base15_reference_source='stage3_v3/experiments.csv',K_policy='native K=N-1; normalized comparison delta/K',
        base15_K=14,new5_K=4,all20_K=19,criteria=summary['criteria'],
        code_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['stage3_stability.py','rigid_gpa.py'] if (Path(__file__).parent/n).exists()},
        input_source_sha256={n:hashlib.sha256((Path(args.source)/n).read_bytes()).hexdigest() for n in ['experiments.csv','manifest.json','summary.json']})
    csv_write(out/'cohort_scores.csv',base+allrows);csv_write(out/'seed_stability.csv',seed);csv_write(out/'identity_stability.csv',identity)
    json_write(out/'manifest.json',manifest);json_write(out/'summary.json',summary);json_write(out/'gpa_log.json',logs)
    plot(seed,identity,out)
    print(json.dumps(summary,ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':main()
