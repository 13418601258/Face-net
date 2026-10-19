"""S3.6 first gate: continuous triangle-constrained correspondence gradients.

No network is trained until the declared strict gate passes. Source meshes,
reference variance, full K and original results remain frozen. NumPy computes
an exact reduced-Gram logdet gradient, checked against finite differences.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from ecomp import covariance_spectrum,ecomp
from index_protocol import sample_seed
from preprocessing import rigid_gpa
from geometry_batch import evaluate_batch_geometry
from metrics import _paired_triangle_distance
from stage35_definition import write_json,write_csv,sha

STATES=['au0','angry','disgust','fear','happy','sad','surprise']
SEEDS=[0,1,2]
LEVELS=np.linspace(0,1,11).tolist()
STEPS=[.01,.03,.1]
FD_STEPS=[.01,.001,.0001]
PATCH_SIZE=512
DELTA=1e-9
PRIMARY_STEP=.03
VARIANCE_FLOORS=[.99,.995,.999]


def helmert(n):
    """Columns form an orthonormal basis orthogonal to the sample mean."""
    q=np.zeros((n,n-1))
    for j in range(n-1):
        q[:j+1,j]=1/np.sqrt((j+1)*(j+2))
        q[j+1,j]=-(j+1)/np.sqrt((j+1)*(j+2))
    return q


class PatchEnergy:
    """Full-shape score with only selected vertices varying; no PCA truncation."""
    def __init__(self,base,indices,reference,delta=DELTA):
        self.base=base;self.indices=np.asarray(indices);self.reference=reference;self.delta=delta
        self.n=len(base);self.k=self.n-1;self.h=helmert(self.n)
        self.scale=self.k*reference
        flat=base.reshape(self.n,-1);reduced=self.h.T@flat
        reduced_patch=self.h.T@base[:,self.indices].reshape(self.n,-1)
        self.fixed_gram=(reduced@reduced.T-reduced_patch@reduced_patch.T)/self.scale

    def evaluate(self,patch,gradient=False):
        z=self.h.T@patch.reshape(self.n,-1)
        a=self.fixed_gram+z@z.T/self.scale+self.delta*np.eye(self.k)
        a=(a+a.T)/2
        sign,value=np.linalg.slogdet(a)
        if sign<=0:raise ArithmeticError('Nonpositive regularized Gram determinant')
        if not gradient:return float(value)
        grad=(2/self.scale)*(self.h@np.linalg.solve(a,z))
        return float(value),grad.reshape(patch.shape)


def make_patch(base,faces,seed,size=PATCH_SIZE):
    """A spatially local common index patch; independent incident-face rays."""
    mean=base.mean(axis=0);n,v,_=base.shape
    rng=np.random.default_rng(sample_seed(seed,'s36/shared_patch'))
    # Central 80% in each coordinate avoids deliberately choosing border tips.
    lo,hi=np.quantile(mean,[.1,.9],axis=0)
    eligible=np.flatnonzero(((mean>=lo)&(mean<=hi)).all(axis=1))
    center=int(rng.choice(eligible));distance=np.linalg.norm(mean-mean[center],axis=1)
    indices=np.argsort(distance,kind='stable')[:min(size,v)]
    incidence=[[] for _ in range(v)]
    for fi,face in enumerate(faces):
        for vertex in face:incidence[vertex].append(fi)
    endpoints=np.empty((n,len(indices),2));edges=np.empty((n,len(indices),3,2));chosen=np.empty((n,len(indices)),int)
    diagonals=np.linalg.norm(np.ptp(base,axis=1),axis=1)
    for i,x in enumerate(base):
        random=np.random.default_rng(sample_seed(seed,f'ast{i+1:03d}/s36/incident_faces'))
        for j,vertex in enumerate(indices):
            found=False
            for fi in random.permutation(incidence[vertex]):
                face=faces[fi];neighbors=face[face!=vertex]
                edge=(x[neighbors]-x[vertex]).T
                singular=np.linalg.svd(edge,compute_uv=False)
                if len(singular)<2 or singular[-1]<=singular[0]*1e-5:continue
                mix=random.uniform(.25,.75)
                uv=.65*np.array([mix,1-mix])
                length=np.linalg.norm(edge@uv)
                uv*=min(1.,.01*diagonals[i]/length)
                edges[i,j]=edge;endpoints[i,j]=uv;chosen[i,j]=fi;found=True;break
            if not found:raise ValueError(f'No nondegenerate incident face for vertex {vertex}')
    origin=base[:,indices].copy()
    return dict(indices=indices,origin=origin,edges=edges,endpoints=endpoints,face_ids=chosen,
        diagonals=diagonals,center=np.array(center))


def points(patch,uv):
    return patch['origin']+np.einsum('nmij,nmj->nmi',patch['edges'],uv)


def closest_uv(patch,query):
    """Euclidean projection onto each fixed triangle, including its edges."""
    a=patch['origin'];e=patch['edges'];d=query-a
    aa=np.sum(e[...,0]**2,axis=-1);bb=np.sum(e[...,0]*e[...,1],axis=-1);cc=np.sum(e[...,1]**2,axis=-1)
    da=np.sum(d*e[...,0],axis=-1);db=np.sum(d*e[...,1],axis=-1);det=aa*cc-bb*bb
    uv=np.stack(((cc*da-bb*db)/det,(aa*db-bb*da)/det),axis=-1)
    interior=(uv>=0).all(axis=-1)&(uv.sum(axis=-1)<=1)
    candidates=[uv]
    zero=np.zeros_like(aa)
    candidates.append(np.stack((np.clip(da/aa,0,1),zero),axis=-1))
    candidates.append(np.stack((zero,np.clip(db/cc,0,1)),axis=-1))
    bc=e[...,1]-e[...,0]
    t=np.clip(np.sum((d-e[...,0])*bc,axis=-1)/np.sum(bc**2,axis=-1),0,1)
    candidates.append(np.stack((1-t,t),axis=-1))
    errors=[np.sum((points(patch,c)-query)**2,axis=-1) for c in candidates]
    errors[0]=np.where(interior,errors[0],np.inf)
    best=np.argmin(np.stack(errors,axis=-1),axis=-1)
    return np.take_along_axis(np.stack(candidates,axis=-2),best[...,None,None],axis=-2)[...,0,:]


def projected_step(patch,uv,gradient,fraction,reference_rmse=None):
    current=points(patch,uv);edge=patch['edges'];normal=np.cross(edge[...,0],edge[...,1])
    normal/=np.linalg.norm(normal,axis=-1)[...,None]
    tangent=gradient-normal*np.sum(gradient*normal,axis=-1)[...,None]
    rms=np.sqrt(np.mean(np.sum((current-patch['origin'])**2,axis=-1))) if reference_rmse is None else reference_rmse
    norm=np.sqrt(np.mean(np.sum(tangent**2,axis=-1)))
    eta=fraction*rms/max(norm,np.finfo(float).tiny)
    return closest_uv(patch,current-eta*tangent),float(eta)


def variance(x):
    centered=x-x.mean(axis=0)
    return float(np.sum(centered**2)/(len(x)-1))


def errors(patch,point):
    diff=point-patch['origin'];mse=np.mean(np.sum(diff**2,axis=-1),axis=1)
    normalized=np.sqrt(mse)/patch['diagonals']
    return float(np.sqrt(np.mean(mse))),float(np.mean(normalized)),normalized


def face_health(base,candidate,faces,indices):
    selected=faces[np.isin(faces,indices).any(axis=1)]
    a=base[:,selected];b=candidate[:,selected]
    na=np.cross(a[:,:,1]-a[:,:,0],a[:,:,2]-a[:,:,0]);nb=np.cross(b[:,:,1]-b[:,:,0],b[:,:,2]-b[:,:,0])
    size=np.linalg.norm(na,axis=-1);size_b=np.linalg.norm(nb,axis=-1)
    good=size>np.finfo(float).eps*np.linalg.norm(np.ptp(base,axis=1),axis=1)[:,None]**2
    flip=(np.sum(na*nb,axis=-1)<0)&good
    deg=(size_b<=np.finfo(float).eps*np.linalg.norm(np.ptp(base,axis=1),axis=1)[:,None]**2)&good
    return dict(affected_faces=len(selected),normal_reversals=int(flip.sum()),new_degenerate_faces=int(deg.sum()),
        normal_reversal_rate=float(flip.sum()/max(good.sum(),1)))


def full_shape(base,indices,patch):
    y=base.copy();y[:,indices]=patch;return y


def regpa_score(y,reference):
    result=rigid_gpa(y)
    if not result.converged:raise ArithmeticError('Re-GPA did not converge')
    value=ecomp(covariance_spectrum(result.aligned),n_components=len(y)-1,reference_variance=reference,delta=DELTA)
    return value


def run_condition(state,seed,source,output,reference):
    out=Path(output);dest=out/'checkpoints'/f'{state}_{seed}.json'
    if dest.exists():return state,seed,'cached'
    start=time.perf_counter();base=np.load(Path(source)/'cache'/f'all20_{state}_baseline.npy')
    faces=np.load(Path(source)/'cache'/f'{state}_faces.npy')
    patch=make_patch(base,faces,seed);indices=patch['indices'];energy=PatchEnergy(base,indices,reference)
    np.savez_compressed(out/'patches'/f'{state}_{seed}.npz',**patch)
    baseline=energy.evaluate(patch['origin']);original=ecomp(covariance_spectrum(base),n_components=19,reference_variance=reference,delta=DELTA)
    if abs(original-baseline)>1e-8:raise ArithmeticError('Reduced Gram score differs from v3')
    base_var=variance(base);local_var=variance(patch['origin']);full_rms=errors(patch,points(patch,patch['endpoints']))[0]
    rows=[];updates=[];subjects=[];numeric=[]
    for t in LEVELS:
        uv=t*patch['endpoints'];current=points(patch,uv)
        value,g=energy.evaluate(current,True);dq=np.einsum('nmij,nmi->nmj',patch['edges'],g)
        slope=float(np.sum(dq*patch['endpoints']));directional=-t*slope
        rms,nrmse,each=errors(patch,current);y=full_shape(base,indices,current)
        row=dict(state=state,seed=seed,t=t,N=20,K=19,patch_vertices=len(indices),score=value,change=value-baseline,
            delta_per_k=(value-baseline)/19,dE_dt=slope,gradient_dot_restore=directional,
            patch_rmse=rms,patch_nrmse=nrmse,global_nrmse=nrmse*np.sqrt(len(indices)/base.shape[1]),
            variance_ratio=variance(y)/base_var,local_variance_ratio=variance(current)/local_var,
            max_displacement_over_diagonal=float(np.max(np.linalg.norm(current-patch['origin'],axis=-1)/patch['diagonals'][:,None])),
            regpa_score=regpa_score(y,reference),**face_health(base,y,faces,indices))
        if t>0:
            fds=[];gpa_fds=[]
            for h in FD_STEPS:
                plus=points(patch,(t+h)*patch['endpoints']);minus=points(patch,(t-h)*patch['endpoints'])
                fd=(energy.evaluate(plus)-energy.evaluate(minus))/(2*h);fds.append(fd)
                numeric.append(dict(state=state,seed=seed,t=t,h=h,analytic_dE_dt=slope,finite_difference=fd,
                    abs_error=abs(fd-slope),passed=bool(abs(fd-slope)<=1e-7+1e-4*abs(slope))))
            for h in [.001,.0001]:
                plus=full_shape(base,indices,points(patch,(t+h)*patch['endpoints']))
                minus=full_shape(base,indices,points(patch,(t-h)*patch['endpoints']))
                gpa_fds.append((regpa_score(plus,reference)-regpa_score(minus,reference))/(2*h))
            row.update(fd_dE_dt=fds[-1],fd_numerical_pass=bool(max(abs(v-slope) for v in fds)<=1e-7+1e-4*abs(slope)),
                restoring_direction_pass=bool(slope>1e-8 and min(fds)>1e-8),
                regpa_restoring_direction_pass=bool(min(gpa_fds)>1e-8),regpa_fd_dE_dt=gpa_fds[-1])
        else:
            row.update(fd_dE_dt=None,fd_numerical_pass=None,restoring_direction_pass=None,regpa_restoring_direction_pass=None,regpa_fd_dE_dt=None)
        rows.append(row)
        for step in STEPS:
            moved,eta=projected_step(patch,uv,g,step,reference_rmse=full_rms*.1 if t==0 else None)
            candidate=points(patch,moved);score=energy.evaluate(candidate);new_rms,new_nrmse,new_each=errors(patch,candidate)
            z=full_shape(base,indices,candidate);vr=variance(z)/base_var;local=variance(candidate)/local_var
            health=face_health(base,z,faces,indices)
            gpa_updated=regpa_score(z,reference) if step==PRIMARY_STEP else None
            triangle=np.stack((patch['origin'],patch['origin']+patch['edges'][...,0],patch['origin']+patch['edges'][...,1]),axis=-2)
            surface_error=float(_paired_triangle_distance(candidate.reshape(-1,3),triangle.reshape(-1,3,3)).max())
            updates.append(dict(state=state,seed=seed,t=t,step_fraction=step,eta=eta,score_before=value,score_after=score,
                energy_decreases=bool(score<value-1e-10),patch_nrmse_before=nrmse,patch_nrmse_after=new_nrmse,
                rmse_improves=bool(new_nrmse<nrmse-1e-12),relative_rmse_change=(new_nrmse/nrmse-1) if t>0 else None,
                variance_ratio=vr,local_variance_ratio=local,variance_pass=bool(min(vr,local)>=.99),
                subjects_improved=int(np.sum(new_each<each-1e-12)),surface_certificate_max_distance=surface_error,
                barycentric_min=float(moved.min()),barycentric_sum_max=float(moved.sum(axis=-1).max()),
                regpa_score_after=gpa_updated,regpa_energy_decreases=(bool(gpa_updated<row['regpa_score']-1e-10) if gpa_updated is not None else None),**health))
            if step==PRIMARY_STEP:
                for i in range(20):
                    subjects.append(dict(state=state,seed=seed,t=t,subject=f'ast{i+1:03d}',patch_nrmse_before=float(each[i]),
                        patch_nrmse_after=float(new_each[i]),rmse_improves=bool(new_each[i]<each[i]-1e-12)))
        print(f'{state}/{seed} t={t:.1f} slope={slope:.6g}',flush=True)
    write_json(dest,dict(state=state,seed=seed,rows=rows,updates=updates,subjects=subjects,numerical=numeric,
        baseline_score_error=abs(original-baseline),seconds=time.perf_counter()-start))
    return state,seed,f'{time.perf_counter()-start:.1f}s'


def geometry_job(state,seed,t,source,output,reference):
    out=Path(output);dest=out/'geometry_checkpoints'/f'{state}_{seed}_{t:g}.json'
    if dest.exists():return state,seed,t,'cached'
    base=np.load(Path(source)/'cache'/f'all20_{state}_baseline.npy');faces=np.load(Path(source)/'cache'/f'{state}_faces.npy')
    with np.load(out/'patches'/f'{state}_{seed}.npz') as f:patch={k:f[k] for k in f.files}
    energy=PatchEnergy(base,patch['indices'],reference);uv=t*patch['endpoints'];current=points(patch,uv)
    _,g=energy.evaluate(current,True);moved,_=projected_step(patch,uv,g,PRIMARY_STEP)
    rows=[]
    for name,p in [('before',current),('after',points(patch,moved))]:
        y=full_shape(base,patch['indices'],p)
        for i,(x,z) in enumerate(zip(base,y)):
            metrics=evaluate_batch_geometry(x,faces,z,faces,tolerance=.001*patch['diagonals'][i],sample_count=2048,seed=0)
            metrics['geometry_seed']=metrics.pop('seed')
            rows.append(dict(state=state,seed=seed,t=t,variant=name,subject=f'ast{i+1:03d}',**metrics))
    write_json(dest,dict(rows=rows));return state,seed,t,'done'


def aggregate(out,protocol):
    parts=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((out/'checkpoints').glob('*.json'))]
    tables={name:[r for d in parts for r in d[name]] for name in ['rows','updates','subjects','numerical']}
    expected={'rows':231,'updates':693,'subjects':4620,'numerical':630}
    for name,rows in tables.items():
        if len(rows)!=expected[name]:raise ValueError('Incomplete '+name)
        write_csv(out/({'rows':'gradient_curves','updates':'projected_updates','subjects':'subject_updates','numerical':'finite_difference_checks'}[name]+'.csv'),rows)
    curves=[r for r in tables['rows'] if r['t']>0]
    updates=[r for r in tables['updates'] if r['t']>0 and r['step_fraction']==PRIMARY_STEP]
    lookup={(r['state'],r['seed'],r['t']):r for r in updates};gate=[]
    for r in curves:
        u=lookup[r['state'],r['seed'],r['t']]
        passed=all([r['fd_numerical_pass'],r['restoring_direction_pass'],r['regpa_restoring_direction_pass'],
            u['energy_decreases'],u['rmse_improves'],u['variance_pass'],u['regpa_energy_decreases']])
        gate.append(dict(state=r['state'],seed=r['seed'],t=r['t'],direction=r['restoring_direction_pass'],
            regpa_direction=r['regpa_restoring_direction_pass'],energy=u['energy_decreases'],rmse=u['rmse_improves'],variance=u['variance_pass'],passed=passed))
    write_csv(out/'gate_conditions.csv',gate)
    geometry_rows=[r for p in sorted((out/'geometry_checkpoints').glob('*.json')) for r in json.loads(p.read_text(encoding='utf-8'))['rows']]
    if len(geometry_rows)!=2520:raise ValueError('Incomplete geometry diagnostics')
    write_csv(out/'geometry_subjects.csv',geometry_rows)
    gstats=[]
    for t in [.1,.5,1.]:
        for variant in ['before','after']:
            rr=[r for r in geometry_rows if abs(r['t']-t)<1e-10 and r['variant']==variant]
            gstats.append(dict(t=t,variant=variant,subjects=len(rr),coverage_mean=float(np.mean([r['coverage'] for r in rr])),
                coverage_min=min(r['coverage'] for r in rr),surface_distance_over_tolerance=float(np.mean([r['symmetric_mean']/r['tolerance'] for r in rr])),
                normal_reversal_rate_mean=float(np.mean([r['normal_reversal_rate'] for r in rr]))))
    write_csv(out/'geometry_summary.csv',gstats)
    baseline_updates=[r for r in tables['updates'] if r['t']==0 and r['step_fraction']==PRIMARY_STEP]
    counts=dict(conditions=210,finite_difference_pass=sum(r['fd_numerical_pass'] for r in curves),
        restoring_direction=sum(r['restoring_direction_pass'] for r in curves),regpa_restoring_direction=sum(r['regpa_restoring_direction_pass'] for r in curves),
        update_energy_decreases=sum(r['energy_decreases'] for r in updates),update_rmse_improves=sum(r['rmse_improves'] for r in updates),
        variance_preserved=sum(r['variance_pass'] for r in updates),all_criteria=sum(r['passed'] for r in gate))
    summary=dict(version=protocol['version'],complete=True,first_gate_pass=all(r['passed'] for r in gate),counts=counts,
        network_status='eligible_for_separate_preregistered_pilot' if all(r['passed'] for r in gate) else 'not_started_gate_failed_as_planned',
        baseline_drift=dict(conditions=len(baseline_updates),energy_decreases=sum(r['energy_decreases'] for r in baseline_updates),
            positive_rmse_after=sum(r['patch_nrmse_after']>1e-12 for r in baseline_updates)),
        step_sensitivity={str(step):dict(conditions=210,energy_decreases=sum(r['energy_decreases'] for r in tables['updates'] if r['t']>0 and r['step_fraction']==step),
            rmse_improves=sum(r['rmse_improves'] for r in tables['updates'] if r['t']>0 and r['step_fraction']==step)) for step in STEPS},
        variance_sensitivity={str(floor):sum(min(r['variance_ratio'],r['local_variance_ratio'])>=floor for r in updates) for floor in VARIANCE_FLOORS},
        local_variance_min=min(r['local_variance_ratio'] for r in updates),global_variance_min=min(r['variance_ratio'] for r in updates),
        fd_error_max=max(r['abs_error'] for r in tables['numerical']),baseline_score_error_max=max(d['baseline_score_error'] for d in parts),
        actual_max_displacement_over_diagonal=max(r['max_displacement_over_diagonal'] for r in tables['rows']),
        rmse_relative_change_median=float(np.median([r['relative_rmse_change'] for r in updates])),
        surface_certificate_max_distance=max(r['surface_certificate_max_distance'] for r in updates),
        geometry=gstats,limitations=protocol['limitations'])
    write_json(out/'summary.json',summary)
    from plot_stage36 import plot_all
    plot_all(out)
    write_json(out/'manifest.json',dict(version=protocol['version'],protocol_sha256=sha(out/'protocol_config_s36.json'),
        outputs={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ('manifest.json','verification.json')},
        checkpoints={str(p.relative_to(out)):sha(p) for folder in ['checkpoints','geometry_checkpoints','patches'] for p in sorted((out/folder).iterdir())}))
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default='code/reports/stage3_s35_v2');p.add_argument('--output',default='code/reports/stage3_s36_v1');p.add_argument('--workers',type=int,default=3)
    args=p.parse_args();source=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for name in ['checkpoints','geometry_checkpoints','patches']:(out/name).mkdir(exist_ok=True)
    meta=json.loads((source/'cache/metadata.json').read_text(encoding='utf-8'))
    input_hashes={}
    for state in STATES:
        for name in [f'all20_{state}_baseline.npy',f'{state}_faces.npy']:
            actual=sha(source/'cache'/name)
            if actual!=meta['cache_sha256'][name]:raise ValueError('Changed input cache '+name)
            input_hashes[name]=actual
    reference=meta['cohorts']['all20']['reference_variance']
    protocol=dict(version='s36-gradient-v1',states=STATES,seeds=SEEDS,levels=LEVELS,N=20,K=19,reference_variance=reference,delta=DELTA,
        input_digest=meta['input_digest'],cache_sha256=input_hashes,source_manifest_sha256=sha(source/'manifest.json'),
        patch_size=PATCH_SIZE,patch_selection='512 nearest mean-shape vertices to seed-fixed interior center; common indices',
        rays='independent incident faces per identity; interior barycentric sum=.65; displacement capped at .01*individual diagonal',
        primary_alignment='frozen baseline rigid GPA, preserve individual scale',secondary_alignment='fresh rigid GPA directional finite differences and primary-update scores',
        energy='logdet(H.T X X.T H/((N-1)*reference)+delta I); H orthonormal to ones; full K=N-1; exact derivative',
        fd_steps=FD_STEPS,fd_tolerance='abs error <= 1e-7+1e-4*abs(analytic derivative); positive slope >1e-8',
        update='negative Cartesian gradient projected to each triangle plane, normalized RMS then exact closest-triangle projection; no true recovery vector used',
        step_fractions=STEPS,primary_step=PRIMARY_STEP,baseline_drift_step='0.1*endpoint RMS reference',
        variance_floors=VARIANCE_FLOORS,primary_variance_floor=.99,
        gate='All 210 positive conditions: correct frozen+reGPA direction, numeric checks, primary step lowers E and RMSE, global/local variance >= .99 baseline, E after reGPA lower',
        geometry='2048 surface samples each direction, all20, t=.1/.5/1 before+after primary step; full affected-face normals all conditions',
        geometry_samples=2048,geometry_tolerance_fraction=.001,
        limitations=['Exploratory gate on previously observed identities, no blind test or trained network',
          'Same-face displacement cap may be far below 1% diagonal; no claim to cross-face or large slides',
          'A point stays on the original surface, but reconnecting moved points changes mesh faces',
          'Frozen aligned gradient is not the derivative through re-GPA; secondary re-GPA path checks address only path sensitivity',
          'A strict gate failure limits this setup, not all neural parameterizations and not automatic evaluation validity',
          'Three seeds are perturbation repeats; anatomy truth is the standardized AST correspondence proxy',
          'Variance and geometry tolerances are declared diagnostic thresholds, not significance tests'],
        code_sha256={name:sha(Path('code')/name) for name in ['stage36_gradient.py','ecomp.py','preprocessing.py','geometry_batch.py','metrics.py','index_protocol.py']})
    dest=out/'protocol_config_s36.json'
    if dest.exists() and json.loads(dest.read_text(encoding='utf-8'))!=protocol:raise ValueError('Changed protocol/code: use new output directory')
    write_json(dest,protocol);print('Protocol frozen; 21 gradient jobs followed by 63 geometry jobs',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(run_condition,s,seed,str(source),str(out),reference) for s in STATES for seed in SEEDS]
        for i,f in enumerate(as_completed(fs),1):print('Gradient',i,'/21',f.result(),flush=True)
        fs=[pool.submit(geometry_job,s,seed,t,str(source),str(out),reference) for s in STATES for seed in SEEDS for t in [.1,.5,1.]]
        for i,f in enumerate(as_completed(fs),1):print('Geometry',i,'/63',f.result(),flush=True)
    aggregate(out,protocol)


if __name__=='__main__':main()
