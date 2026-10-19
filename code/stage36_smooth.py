"""S3.6 v2 mechanism pilot; sparse proxy anchors are NOT real landmarks.

Frozen v1 data are read-only. Synthetic truth constructs corruptions, sparse
observations and evaluation, never dense optimization targets or step lengths.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.tri import Triangulation
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve
from stage36_gradient import PatchEnergy

ROOT=Path(__file__).resolve().parent.parent
CACHE=ROOT/'code/reports/stage3_s35_v2/cache'
OLD=ROOT/'code/reports/stage3_s36_v1'
CONFIG=ROOT/'code/stage36_v2_protocol.json'
OUT=ROOT/'code/reports/stage3_s36_v2_harmonic'
STATES=['au0','angry','disgust','fear','happy','sad','surprise']
ARMS=['anchor_only','anchor_fixed_e','anchor_gated_e','e_only']
LEVELS=[0.,.05,.1,.5,1.]
WEIGHT=.1
TAU=1e-4
MAX_ITER=160


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+chr(10),encoding='utf-8')


def write_csv(path,rows):
    if not rows:raise ValueError('Cannot export an empty result table')
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def smooth_basis(uv,radius):
    """Two coordinate directions times three affine modes, compact C1 support."""
    p=uv/radius;r=np.linalg.norm(p,axis=1)
    w=np.maximum(1-r,0)**4*(4*r+1)
    a=w[:,None]*np.column_stack((np.ones(len(p)),p))
    b=np.zeros((len(p),2,6));b[:,0,:3]=a;b[:,1,3:]=a
    return b


def anchor_terms(q,design):
    """Only the two sparse observed UV residuals enter this objective."""
    residual=np.einsum('jak,nk->nja',design,q)
    a=float(np.mean(np.sum(residual**2,axis=-1)))
    grad=2*np.einsum('jak,nja->nk',design,residual)/(len(q)*len(design))
    return a,grad


def gate_weight(a,arm):
    if arm=='anchor_only':return 0.
    if arm=='e_only':return 1.
    return WEIGHT*(a/(a+TAU) if arm=='anchor_gated_e' else 1.)


def local_health(reference,candidate,diagonals):
    n0=np.cross(reference[:,:,1]-reference[:,:,0],reference[:,:,2]-reference[:,:,0])
    n1=np.cross(candidate[:,:,1]-candidate[:,:,0],candidate[:,:,2]-candidate[:,:,0])
    a0=np.linalg.norm(n0,axis=-1);a1=np.linalg.norm(n1,axis=-1)
    good=a0>np.finfo(float).eps*diagonals[:,None]**2
    dots=np.sum(n0*n1,axis=-1)
    ratios=np.divide(a1,a0,out=np.ones_like(a1),where=good)
    flips=int(np.sum((dots<=0)&good));minimum=float(ratios[good].min())
    return dict(valid=bool(flips==0 and minimum>=.2),normal_reversals=flips,min_area_ratio=minimum)


def harmonic_chart(mean,faces,center):
    """Uniform positive-weight harmonic disk, fixed length-spaced circular boundary."""
    region=np.argsort(np.linalg.norm(mean-mean[center],axis=1),kind='stable')[:2048]
    chosen=faces[np.isin(faces,region).all(1)]
    used=np.unique(chosen);mapping=np.full(len(mean),-1,int);mapping[used]=np.arange(len(used))
    edges=np.sort(np.vstack((chosen[:,[0,1]],chosen[:,[1,2]],chosen[:,[2,0]])),axis=1)
    edges,counts=np.unique(edges,axis=0,return_counts=True)
    boundary_edges=edges[counts==1];boundary,degree=np.unique(boundary_edges,return_counts=True)
    if np.any(counts>2) or np.any(degree!=2) or len(used)-len(edges)+len(chosen)!=1:
        raise ValueError('Local chart topology is not a manifold disk')
    neighbors={int(v):[] for v in boundary}
    for a,b in boundary_edges:neighbors[int(a)].append(int(b));neighbors[int(b)].append(int(a))
    order=[int(boundary.min())];previous=None
    while True:
        choices=sorted(v for v in neighbors[order[-1]] if v!=previous)
        nxt=choices[0]
        if nxt==order[0]:break
        if nxt in order:raise ValueError('Boundary contains multiple cycles')
        previous=order[-1];order.append(nxt)
        if len(order)>len(boundary):raise ValueError('Boundary walk did not close')
    if len(order)!=len(boundary):raise ValueError('Boundary has disconnected components')
    order=np.array(order);length=np.linalg.norm(mean[np.roll(order,-1)]-mean[order],axis=1)
    if np.any(length<=0):raise ValueError('Zero-length chart boundary edge')
    angle=2*np.pi*np.r_[0,np.cumsum(length[:-1])]/length.sum();radius=length.sum()/(2*np.pi)
    local_edges=mapping[edges]
    rows=np.r_[local_edges[:,0],local_edges[:,1]];cols=np.r_[local_edges[:,1],local_edges[:,0]]
    adjacency=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(used),len(used))).tocsr()
    lap=diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    fixed=mapping[order];interior=np.setdiff1d(np.arange(len(used)),fixed)
    uv=np.zeros((len(used),2));uv[fixed]=radius*np.column_stack((np.cos(angle),np.sin(angle)))
    uv[interior]=spsolve(lap[interior][:,interior],-(lap[interior][:,fixed]@uv[fixed]))
    if not np.isfinite(uv).all():raise ValueError('Nonfinite harmonic solution')
    full=np.zeros((len(mean),2));full[used]=uv-uv[mapping[center]]
    return full,chosen,used,order


class SmoothPatch:
    def __init__(self,base,faces,indices,center,reference):
        self.base=base
        self.diagonals=np.linalg.norm(np.ptp(base,axis=1),axis=1)
        mean=base.mean(0);offset=mean-mean[center]
        self.uv,chosen,region,boundary=harmonic_chart(mean,faces,center)
        if not np.isin(indices,region).all():raise ValueError('v1 support falls outside chart')
        radius=min(float(np.max(np.linalg.norm(self.uv[indices],axis=1))),.8*float(np.min(np.linalg.norm(self.uv[boundary],axis=1))))
        if radius<=1e-6:raise ValueError('No usable interior chart support')
        indices=region[np.linalg.norm(self.uv[region],axis=1)<radius*(1-1e-12)]
        if len(indices)<30:raise ValueError('Insufficient vertices inside smooth compact support')
        self.indices=indices;self.origin=base[:,indices];self.start=self.uv[indices];self.radius=radius
        p=self.uv[chosen]
        signed=(p[:,1,0]-p[:,0,0])*(p[:,2,1]-p[:,0,1])-(p[:,1,1]-p[:,0,1])*(p[:,2,0]-p[:,0,0])
        if np.any(np.abs(signed)<1e-14) or not (np.all(signed>0) or np.all(signed<0)):
            raise ValueError('Harmonic chart contains degenerate or reversed triangles')
        if signed[0]<0:chosen=chosen[:,[0,2,1]]
        used=np.unique(chosen);mapping=np.full(len(mean),-1,int);mapping[used]=np.arange(len(used))
        self.triangles=chosen
        self.finder=Triangulation(self.uv[used,0],self.uv[used,1],mapping[chosen]).get_trifinder()
        e=np.stack((self.uv[chosen[:,1]]-self.uv[chosen[:,0]],self.uv[chosen[:,2]]-self.uv[chosen[:,0]]),axis=-1)
        self.inverse=np.linalg.inv(e)
        self.basis=smooth_basis(self.start,self.radius)
        self.scale=.002*float(np.linalg.norm(np.ptp(mean,axis=0)))
        candidates=np.flatnonzero(np.linalg.norm(self.start/self.radius,axis=1)<.65)
        if len(candidates)<2:raise ValueError('Insufficient interior samples')
        a=int(candidates[np.argmax(np.linalg.norm(self.basis[candidates],axis=(1,2)))])
        b=int(candidates[np.argmax(np.linalg.norm(self.start[candidates]-self.start[a],axis=1))])
        self.anchors=np.array([a,b]);self.design=self.basis[self.anchors]
        design_flat=self.design.reshape(-1,6)
        self.anchor_rank=int(np.linalg.matrix_rank(design_flat))
        self.preconditioner=np.linalg.inv(2*design_flat.T@design_flat/2+.1*np.eye(6))*len(base)
        self.energy=PatchEnergy(base,indices,reference)
        self.faces=faces[np.isin(faces,indices).any(1)]
        self.face_reference=base[:,self.faces].copy()
        lookup=np.full(len(mean),-1,int);lookup[indices]=np.arange(len(indices))
        self.face_slots=lookup[self.faces];self.face_mask=self.face_slots>=0
        self.nonanchor=np.ones(len(indices),bool);self.nonanchor[self.anchors]=False
        self.base_var=float(np.sum((base-base.mean(0))**2)/(len(base)-1))
        self.local_var=float(np.sum((self.origin-self.origin.mean(0))**2)/(len(base)-1))
        z=self.sample(np.zeros((len(base),6)))
        if z is None or np.max(np.abs(z[0]-self.origin))>1e-9:
            raise ValueError('Chart does not reconstruct baseline vertices')

    def sample(self,q):
        uv=self.start[None]+self.scale*np.einsum('jak,nk->nja',self.basis,q)
        ids=np.asarray(self.finder(uv[...,0],uv[...,1]))
        if np.any(ids<0):return None
        tri=self.triangles[ids]
        rhs=uv-self.uv[tri[:,:,0]]
        weights=np.einsum('njab,njb->nja',self.inverse[ids],rhs)
        x=self.base[np.arange(len(q))[:,None,None],tri]
        edge=np.stack((x[:,:,1]-x[:,:,0],x[:,:,2]-x[:,:,0]),axis=-1)
        points=x[:,:,0]+np.einsum('njca,nja->njc',edge,weights)
        jac=np.einsum('njca,njab->njcb',edge,self.inverse[ids])
        return points,jac,weights

    def evaluate(self,q):
        sample=self.sample(q)
        if sample is None:return None
        points,jac,_=sample
        if np.max(np.linalg.norm(points-self.origin,axis=-1)/self.diagonals[:,None])>.01:return None
        f=self.face_reference.copy();f[:,self.face_mask]=points[:,self.face_slots[self.face_mask]]
        health=local_health(self.face_reference,f,self.diagonals)
        if not health['valid']:return None
        energy,gradient=self.energy.evaluate(points,True)
        gradient_uv=np.einsum('njca,njc->nja',jac,gradient)
        gradient_q=self.scale*np.einsum('jak,nja->nk',self.basis,gradient_uv)/19
        a,ga=anchor_terms(q,self.design)
        return dict(points=points,energy=energy/19,ge=gradient_q,anchor=a,ga=ga,**health)

    def metrics(self,e):
        d=np.linalg.norm(e['points']-self.origin,axis=-1)**2
        each=np.sqrt(d[:,self.nonanchor].mean(1))/self.diagonals
        all_each=np.sqrt(d.mean(1))/self.diagonals
        local=float(np.sum((e['points']-e['points'].mean(0))**2)/(len(self.base)-1))
        return dict(nonanchor_nrmse=float(each.mean()),patch_nrmse=float(all_each.mean()),
                    local_variance_ratio=local/self.local_var,
                    global_variance_ratio=(self.base_var+local-self.local_var)/self.base_var,
                    max_displacement_ratio=float(np.max(np.sqrt(d)/self.diagonals[:,None])),
                    normal_reversals=e['normal_reversals'],min_area_ratio=e['min_area_ratio']),each


def solve(model,q0,arm,max_iterations=MAX_ITER):
    q=q0.copy();e=model.evaluate(q)
    if e is None:raise ValueError('Initial state violates declared geometry bounds')
    history=[];rejected=0;reason='max_iterations';initial_g=None
    for iteration in range(max_iterations+1):
        weight=gate_weight(e['anchor'],arm)
        ga=np.zeros_like(q) if arm=='e_only' else e['ga']
        gradient=ga+weight*e['ge']
        norm=float(np.max(np.abs(gradient)));m,_=model.metrics(e)
        history.append(dict(iteration=iteration,energy=e['energy'],anchor=e['anchor'],weight=weight,
                            gradient_inf=norm,rejected_geometry=rejected,**m))
        if initial_g is None:
            initial_g=dict(anchor_gradient_norm=float(np.linalg.norm(e['ga'])),energy_gradient_norm=float(np.linalg.norm(e['ge'])))
        if norm<=1e-7:reason='gradient_tolerance';break
        if iteration==max_iterations:break
        direction=-gradient@model.preconditioner
        cap=np.max(np.abs(direction))
        if cap>.25:direction*=.25/cap
        slope=float(np.sum(gradient*direction));factor=1.;accepted=False
        objective=(0 if arm=='e_only' else e['anchor'])+weight*e['energy']
        for _ in range(24):
            trial=q+factor*direction;candidate=model.evaluate(trial)
            if candidate is None:rejected+=1
            else:
                value=(0 if arm=='e_only' else candidate['anchor'])+weight*candidate['energy']
                if value<=objective+1e-4*factor*slope:
                    q=trial;e=candidate;accepted=True;break
            factor*=.5
        if not accepted:reason='line_search_failed';break
        if np.max(np.abs(factor*direction))<=1e-8:
            reason='step_tolerance'
            m,_=model.metrics(e)
            history.append(dict(iteration=iteration+1,energy=e['energy'],anchor=e['anchor'],weight=gate_weight(e['anchor'],arm),gradient_inf=norm,rejected_geometry=rejected,**m))
            break
    final,each=model.metrics(e)
    return dict(q=q,history=history,each=each,reason=reason,final=final,anchor=e['anchor'],energy=e['energy'],initial_gradients=initial_g)


def run_condition(state,seed,out_path):
    out=Path(out_path);dest=out/'checkpoints'/f'{state}_{seed}.json'
    if dest.exists():return state,seed,'cached'
    base=np.load(CACHE/f'all20_{state}_baseline.npy');faces=np.load(CACHE/f'{state}_faces.npy')
    with np.load(OLD/'patches'/f'{state}_{seed}.npz') as z:indices=z['indices'];center=int(z['center'])
    try:model=SmoothPatch(base,faces,indices,center,109.16412439918614)
    except (ValueError,RuntimeError) as exc:
        write_json(dest,dict(state=state,seed=seed,status='invalid_chart',reason=str(exc)))
        return state,seed,'invalid_chart: '+str(exc)
    rng=np.random.default_rng(36000+seed);q0=rng.normal(size=(20,6))
    maximum=np.max(np.linalg.norm(np.einsum('jak,nk->nja',model.basis,q0),axis=-1));q0/=maximum
    halvings=0
    while any(model.evaluate(t*q0) is None for t in LEVELS):
        q0*=.5;halvings+=1
        if halvings>12:raise ArithmeticError('No geometry-valid initial amplitude')
    fd=[];q=.5*q0;e=model.evaluate(q)
    for _ in range(3):
        direction=rng.normal(size=q.shape);direction/=np.linalg.norm(direction)
        analytic=float(np.sum(e['ge']*direction))
        errors=[]
        for h in [1e-5,1e-6]:
            plus=model.sample(q+h*direction);minus=model.sample(q-h*direction)
            numeric=(model.energy.evaluate(plus[0])-model.energy.evaluate(minus[0]))/(2*h*19)
            errors.append(abs(numeric-analytic))
        fd.append(dict(analytic=analytic,errors=errors,passed=min(errors)<=1e-6+1e-3*abs(analytic)))
    if not all(v['passed'] for v in fd):raise ArithmeticError('Smooth-field derivative check failed')
    summaries=[];traces=[];subjects=[];final_q=[]
    for t in LEVELS:
        initial=model.evaluate(t*q0);im,ie=model.metrics(initial)
        for arm in ARMS:
            result=solve(model,t*q0,arm);key=dict(state=state,seed=seed,t=t,arm=arm)
            row=dict(**key,iterations=result['history'][-1]['iteration'],termination=result['reason'],
                     solver_converged=result['reason'] in ['gradient_tolerance','step_tolerance'],
                     initial_nonanchor_nrmse=im['nonanchor_nrmse'],final_nonanchor_nrmse=result['final']['nonanchor_nrmse'],
                     initial_energy=initial['energy'],final_energy=result['energy'],initial_anchor=initial['anchor'],final_anchor=result['anchor'],
                     local_variance_ratio=result['final']['local_variance_ratio'],global_variance_ratio=result['final']['global_variance_ratio'],
                     normal_reversals=result['final']['normal_reversals'],min_area_ratio=result['final']['min_area_ratio'],
                     subjects_improved=int(np.sum(result['each']<ie-1e-12)),initial_max_displacement_ratio=im['max_displacement_ratio'],
                     final_max_displacement_ratio=result['final']['max_displacement_ratio'],**result['initial_gradients'])
            summaries.append(row);traces.extend([dict(**key,**h) for h in result['history']]);final_q.append(result['q'])
            subjects.extend([dict(**key,subject=f'ast{i+1:03d}',initial_nrmse=float(ie[i]),final_nrmse=float(result['each'][i])) for i in range(20)])
            print(f'{state}/{seed} t={t:g} {arm}: {row["termination"]}, {row["initial_nonanchor_nrmse"]:.3g} -> {row["final_nonanchor_nrmse"]:.3g}',flush=True)
    np.savez_compressed(out/'parameters'/f'{state}_{seed}.npz',initial_q=q0,final_q=np.array(final_q),
                        indices=model.indices,anchors=model.anchors,basis=model.basis,chart_uv=model.uv,
                        chart_triangles=model.triangles,scale=model.scale,radius=model.radius)
    write_json(dest,dict(state=state,seed=seed,status='complete',anchor_rank=model.anchor_rank,patch_vertices=len(model.indices),
                         amplitude_halvings=halvings,finite_difference=fd,rows=summaries,traces=traces,subjects=subjects))
    return state,seed,'complete'


def aggregate(out):
    jobs=[json.loads((out/'checkpoints'/f'{s}_{seed}.json').read_text(encoding='utf-8')) for s in STATES for seed in range(3)]
    complete=[j for j in jobs if j['status']=='complete'];rows=sum([j['rows'] for j in complete],[])
    traces=sum([j['traces'] for j in complete],[]);subjects=sum([j['subjects'] for j in complete],[])
    if not rows:raise RuntimeError('All charts invalid; inspect checkpoints, no outcome claim possible')
    for name,data in [('endpoints',rows),('trajectories',traces),('subject_endpoints',subjects)]:write_csv(out/f'{name}.csv',data)
    lookup={(r['state'],r['seed'],r['t'],r['arm']):r for r in rows};pairs=[]
    for r in rows:
        if r['arm']=='anchor_only':continue
        control=lookup[r['state'],r['seed'],r['t'],'anchor_only']
        pairs.append(dict(state=r['state'],seed=r['seed'],t=r['t'],arm=r['arm'],
                          delta_nrmse_vs_anchor=r['final_nonanchor_nrmse']-control['final_nonanchor_nrmse'],
                          improves_over_anchor=r['final_nonanchor_nrmse']<control['final_nonanchor_nrmse']-1e-12,
                          solver_converged=r['solver_converged']))
    write_csv(out/'paired_endpoints.csv',pairs)
    by=[]
    for t in LEVELS:
        for arm in ARMS:
            rr=[r for r in rows if r['t']==t and r['arm']==arm]
            by.append(dict(t=t,arm=arm,count=len(rr),median_initial_nrmse=float(np.median([r['initial_nonanchor_nrmse'] for r in rr])),
                           median_final_nrmse=float(np.median([r['final_nonanchor_nrmse'] for r in rr])),
                           improves_initial=sum(r['final_nonanchor_nrmse']<r['initial_nonanchor_nrmse']-1e-12 for r in rr),
                           solver_converged=sum(r['solver_converged'] for r in rr),
                           max_iterations=sum(r['termination']=='max_iterations' for r in rr),
                           line_search_failed=sum(r['termination']=='line_search_failed' for r in rr),
                           median_anchor_residual=float(np.median([r['final_anchor'] for r in rr])),
                           beats_anchor=sum(r['improves_over_anchor'] for r in pairs if r['t']==t and r['arm']==arm)))
    summary=dict(version='s36-v2-smooth-proxy-harmonic',complete=len(jobs)==21,valid_charts=len(complete),invalid_charts=[dict(state=j['state'],seed=j['seed'],reason=j['reason']) for j in jobs if j['status']!='complete'],
                 endpoint_count=len(rows),subject_count=len(subjects),trajectory_records=len(traces),
                 all_accepted_geometry_valid=all(r['normal_reversals']==0 and r['min_area_ratio']>=.2 for r in traces),
                 finite_difference_pass=all(v['passed'] for j in complete for v in j['finite_difference']),
                 by_level_arm=by,network_eligible=False,
                 interpretation='Synthetic sparse-anchor mechanism test only. Zero-baseline gating is constructed; independent anchor validity and held-out gains remain untested. All iteration-limit exits are explicitly incomplete optimization.')
    write_json(out/'summary.json',summary)
    from plot_stage36_smooth import plot_all
    plot_all(out)
    files=[p for p in out.rglob('*') if p.is_file() and p.name not in ('manifest.json','verification.json')]
    write_json(out/'manifest.json',dict(files={p.relative_to(out).as_posix():sha(p) for p in sorted(files)},code={n:sha(ROOT/'code'/n) for n in ['stage36_smooth.py','plot_stage36_smooth.py','stage36_v2_protocol.json']}))
    print(json.dumps({k:v for k,v in summary.items() if k!='by_level_arm'},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=3);p.add_argument('--out',default=str(OUT));p.add_argument('--pilot',action='store_true');p.add_argument('--aggregate-only',action='store_true');args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    for folder in ['checkpoints','parameters']:(out/folder).mkdir(exist_ok=True)
    original=json.loads(CONFIG.read_text(encoding='utf-8'))
    source={'protocol_sha256':sha(CONFIG),'code_sha256':sha(__file__),
            'core_sha256':sha(ROOT/'code/stage36_gradient.py'),
            'source_sha256':{str(p.relative_to(ROOT)):sha(p) for s in STATES for p in [CACHE/f'all20_{s}_baseline.npy',CACHE/f'{s}_faces.npy']},
            'patch_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((OLD/'patches').glob('*.npz'))}}
    frozen=out/'frozen_protocol.json'
    if frozen.exists():
        if json.loads(frozen.read_text(encoding='utf-8'))!=dict(protocol=original,provenance=source):raise RuntimeError('Protocol/code/input changed; use a new output directory')
    else:write_json(frozen,dict(protocol=original,provenance=source))
    if args.aggregate_only:aggregate(out);return
    tasks=[('happy',0)] if args.pilot else [(s,seed) for s in STATES for seed in range(3)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_condition,s,seed,str(out)) for s,seed in tasks]
        for f in as_completed(futures):print('Finished',f.result(),flush=True)
    if not args.pilot:aggregate(out)


if __name__=='__main__':main()
