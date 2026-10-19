"""Final bounded S3.6 joint-spectrum experiment; cached extracted meshes only."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time
import numpy as np
from scipy.sparse import coo_matrix,diags,eye
from stage36_smooth import ROOT,CACHE,OLD,STATES,SmoothPatch,local_health,sha,write_json,write_csv
from stage36_gradient import PatchEnergy
from stage36_constraints import layouts,exact_sparse,observation
from stage36_information import holdout_vertices,subspaces,observation_each
from preprocessing import rigid_gpa
from pathlib import Path

OUT=ROOT/'code/reports/stage3_s36_v5_final'
CONFIG=ROOT/'code/stage36_final_protocol.json'

def center_tensor(x):return x-x.mean(axis=(0,1),keepdims=True)

def mode_grams(x):
    s,n,_=x.shape
    return np.einsum('sid,sjd->ij',x,x)/s,np.einsum('sid,tid->st',x,x)/n

def log_score(gram,reference,delta):
    a=(gram+gram.T)/(2*reference)+delta*np.eye(len(gram))
    sign,value=np.linalg.slogdet(a)
    if sign<=0:raise ArithmeticError('Nonpositive regularized mode Gram')
    return float(value/len(gram)),a

class JointEnergy:
    def __init__(self,base,indices,delta=1e-9):
        self.shape=base.shape;self.indices=np.asarray(indices);self.delta=delta
        s,n=base.shape[:2];self.s=s;self.n=n
        x=center_tensor(base.reshape(s,n,-1));g=mode_grams(x)
        self.references=tuple(float(np.trace(a)/len(a)) for a in g)
        if min(self.references)<=0:raise ValueError('Degenerate mode reference')
        patch=center_tensor(base[:,:,indices].reshape(s,n,-1));p=mode_grams(patch)
        self.fixed=tuple(a-b for a,b in zip(g,p));self.baseline_grams=g
    def evaluate(self,points,gradient=True):
        z=center_tensor(points.reshape(self.s,self.n,-1));gp=mode_grams(z)
        grams=[a+b for a,b in zip(self.fixed,gp)]
        parts=[log_score(a,r,self.delta) for a,r in zip(grams,self.references)]
        value=sum(p[0] for p in parts)
        if not gradient:return value
        ai=np.linalg.inv(parts[0][1]);ae=np.linalg.inv(parts[1][1])
        gi=2*np.einsum('ij,sjd->sid',ai,z)/(self.n*self.s*self.references[0])
        ge=2*np.einsum('st,tid->sid',ae,z)/(self.s*self.n*self.references[1])
        return value,center_tensor(gi+ge).reshape(points.shape)
    def diagnostics(self):
        result=[]
        for label,g,r in zip(('identity','expression'),self.baseline_grams,self.references):
            values=np.linalg.eigvalsh((g+g.T)/2)[::-1]
            result.append(dict(mode=label,terms=len(g),rank=int(np.sum(values>values[0]*1e-12)),reference=r,eigenvalues=values.tolist()))
        return result

class SingleEnergy:
    def __init__(self,base,indices,reference):
        self.models=[PatchEnergy(b,indices,reference) for b in base];self.s,self.n=base.shape[:2]
    def evaluate(self,points,gradient=True):
        shaped=points.reshape(self.s,self.n,-1,3);values=[];grads=[]
        for m,p in zip(self.models,shaped):
            result=m.evaluate(p,gradient)
            if gradient:values.append(result[0]);grads.append(result[1])
            else:values.append(result)
        value=float(np.mean(values)/(self.n-1))
        if not gradient:return value
        return value,np.stack(grads).reshape(points.shape)/(self.s*(self.n-1))

def double_umbrella(faces,n):
    edges=np.unique(np.sort(np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    rr=np.r_[edges[:,0],edges[:,1]];cc=np.r_[edges[:,1],edges[:,0]]
    adj=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(n,n)).tocsr()
    degree=np.asarray(adj.sum(1)).ravel()
    if np.any(degree==0):raise ValueError('Isolated vertices in umbrella operator')
    u=diags(1/degree)@adj-eye(n,format='csr')
    return (u@u).tocsr()

class BiLaplacian:
    def __init__(self,base,faces,indices):
        n,v,_=base.shape;operator=double_umbrella(faces,v)
        rows=np.unique(operator[:,indices].nonzero()[0]);local=operator[rows]
        self.variable=local[:,indices].tocsr();self.origin=base[:,indices].copy()
        self.b0=np.stack([local@x for x in base])
        self.denominator=n*v*float(np.mean(np.linalg.norm(np.ptp(base,axis=1),axis=1)))**2
        self.affected_rows=len(rows)
    def evaluate(self,points,gradient=True):
        d=points-self.origin;n,p,_=d.shape
        change=(self.variable@d.transpose(1,0,2).reshape(p,-1)).reshape(-1,n,3).transpose(1,0,2)
        y=self.b0+change;value=float(np.sum(y*y)/self.denominator)
        if not gradient:return value
        g=(self.variable.T@y.transpose(1,0,2).reshape(y.shape[1],-1)).reshape(p,n,3).transpose(1,0,2)
        return value,2*g/self.denominator

def point_to_q(model,jac,gradient):
    uv=np.einsum('njca,njc->nja',jac,gradient)
    return model.scale*np.einsum('jak,nja->nk',model.basis,uv)

def sample_valid(model,q):
    sampled=model.sample(q)
    if sampled is None:return None
    points,jac,_=sampled
    maximum=float(np.max(np.linalg.norm(points-model.origin,axis=-1)/model.diagonals[:,None]))
    if maximum>.01:return None
    faces=model.face_reference.copy();faces[:,model.face_mask]=points[:,model.face_slots[model.face_mask]]
    health=local_health(model.face_reference,faces,model.diagonals)
    if not health['valid']:return None
    return dict(points=points,jac=jac,max_displacement_ratio=maximum,**health)

def select_gain(reference,candidate):
    gain=(reference-candidate).reshape(7,20).mean(axis=0)
    mean=float(gain.mean());se=float(gain.std(ddof=1)/np.sqrt(20))
    return mean>max(1e-12,2*se),mean,se

def select_candidates(candidates,cap):
    stay=candidates[0]
    valid=[c for c in candidates[1:] if c['row']['cap']==cap and c['row']['accepted']]
    def pick(pool,reference):
        if not pool:return reference
        winner=min(pool,key=lambda c:(float(c['selection'].mean()),c['row']['name']))
        return winner if select_gain(reference['selection'],winner['selection'])[0] else reference
    regularizer=pick([c for c in valid if c['row']['family']=='regularizer'],stay)
    single=pick([c for c in valid if c['row']['family']=='single'],regularizer)
    joint=pick([c for c in valid if c['row']['family']=='joint'],regularizer)
    return dict(regularizer=regularizer,single=single,joint=joint)

def measure(model,q,mask,sd,ys,ad,ya):
    sampled=model.sample(q)
    if sampled is None:return None
    points=sampled[0];difference=np.linalg.norm(points-model.origin,axis=-1)/model.diagonals[:,None]
    each=np.sqrt(np.mean(difference[:,mask]**2,axis=1))
    p=points.reshape(7,20,-1,3);origin=model.origin.reshape(7,20,-1,3)
    local=np.sum((p-p.mean(1,keepdims=True))**2,axis=(1,2,3))
    original=np.sum((origin-origin.mean(1,keepdims=True))**2,axis=(1,2,3))
    total=model.final_total_variance
    health=sample_valid(model,q)
    return dict(each=each,selection=observation_each(q,sd,ys),audit=observation_each(q,ad,ya),
                metrics=dict(nrmse=float(each.mean()),geometry_valid=health is not None,
                             min_local_variance_ratio=float(np.min(local/original)),
                             min_global_variance_ratio=float(np.min((total+local-original)/total)),
                             max_displacement_ratio=float(difference.max()),
                             normal_reversals=health['normal_reversals'] if health else None,
                             min_area_ratio=health['min_area_ratio'] if health else None))

def propose(model,regularizer,energy,q,projection,design,target,sigma_q,ratio,cap):
    initial=sample_valid(model,q)
    rv,rg=regularizer.evaluate(initial['points']);gr=point_to_q(model,initial['jac'],rg)
    if energy is None:ev=0.;ge=np.zeros_like(q)
    else:
        ev,eg=energy.evaluate(initial['points']);ge=point_to_q(model,initial['jac'],eg)
    pr=gr@projection;pe=ge@projection;nr=float(np.linalg.norm(pr));ne=float(np.linalg.norm(pe))
    wr=1/nr if nr>1e-14 else 0.;we=ratio/ne if ne>1e-14 else 0.
    gradient=wr*pr+we*pe;norm=float(np.max(np.abs(gradient)))
    detail=dict(accepted=False,reason='no_direction',halvings=0,regularizer_before=rv,regularizer_after=rv,
                energy_before=ev,energy_after=ev,regularizer_weight=wr,energy_weight=we,
                regularizer_projected_norm=nr,energy_projected_norm=ne,max_anchor_increase=0.)
    if norm<=1e-14:return q.copy(),detail
    direction=-cap*gradient/norm;slope=float(np.sum((wr*gr+we*ge)*direction))
    objective=wr*rv+we*ev;before=observation_each(q,design,target);detail['reason']='rejected'
    for k in range(12):
        factor=2.**(-k);trial=q+factor*direction
        increase=observation_each(trial,design,target)-before
        if np.any(increase>2*sigma_q**2+1e-12):continue
        candidate=sample_valid(model,trial)
        if candidate is None:continue
        r1=regularizer.evaluate(candidate['points'],False)
        e1=0. if energy is None else energy.evaluate(candidate['points'],False)
        if wr*r1+we*e1<=objective+1e-4*factor*slope:
            detail.update(accepted=True,reason='accepted',halvings=k,regularizer_after=r1,energy_after=e1,max_anchor_increase=float(increase.max()))
            return trial,detail
    return q.copy(),detail

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[CONFIG,Path(__file__),ROOT/'code/test_stage36_final.py',ROOT/'code/verify_stage36_final.py',
           ROOT/'code/stage36_smooth.py',ROOT/'code/stage36_gradient.py',ROOT/'code/stage36_constraints.py',
           ROOT/'code/stage36_information.py',ROOT/'code/preprocessing.py',ROOT/'code/ecomp.py',CACHE/'metadata.json']
    paths += [CACHE/f'{s}_raw.npy' for s in STATES]+[CACHE/f'{s}_faces.npy' for s in STATES]
    paths += [OLD/'patches'/f'au0_{seed}.npz' for seed in range(3)]
    value=dict(protocol=json.loads(CONFIG.read_text(encoding='utf-8')),hashes={p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    dest=OUT/'frozen_protocol.json'
    if dest.exists():
        if json.loads(dest.read_text(encoding='utf-8'))!=value:raise ValueError('Frozen source/input changed: do not overwrite results')
    else:write_json(dest,value)
    return value['protocol']

def prepare():
    dest=OUT/'joint_baseline.npy'
    if dest.exists():return np.load(dest),np.load(CACHE/'au0_faces.npy')
    raw=np.stack([np.load(CACHE/f'{s}_raw.npy') for s in STATES]);shape=raw.shape
    if shape[:2]!=(7,20):raise ValueError(f'Unexpected input shape {shape}')
    faces=np.load(CACHE/'au0_faces.npy')
    for s in STATES:
        if not np.array_equal(faces,np.load(CACHE/f'{s}_faces.npy')):raise ValueError('Cross-state topology mismatch')
    result=rigid_gpa(raw.reshape(-1,*shape[2:]))
    if not result.converged:raise ArithmeticError('Joint rigid GPA did not converge')
    base=result.aligned.reshape(shape);np.save(dest,base)
    np.savez_compressed(OUT/'rigid_alignment.npz',rotations=result.rotations,centers=result.centers)
    write_json(OUT/'alignment.json',dict(shape=list(shape),iterations=result.iterations,relative_change=result.relative_change,
               scale_normalization=False,converged=result.converged,input='cached extracted raw meshes',sha256=sha(dest)))
    print('Joint rigid GPA complete',result.iterations,flush=True)
    return base,faces

def build_model(base,faces,seed):
    with np.load(OLD/'patches'/f'au0_{seed}.npz') as z:indices=z['indices'];center=int(z['center'])
    model=SmoothPatch(base.reshape(-1,*base.shape[2:]),faces,indices,center,109.16412439918614)
    all_edges=np.sort(np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    unique,counts=np.unique(all_edges,axis=0,return_counts=True);boundary=np.unique(unique[counts==1])
    ring=np.unique(faces[np.isin(faces,boundary).any(axis=1)])
    fixed=np.isin(model.indices,ring);model.basis[fixed]=0.
    model.final_total_variance=np.sum((base-base.mean(1,keepdims=True))**2,axis=(1,2,3))
    return model,dict(seed=seed,center=center,vertices=len(model.indices),fixed_boundary_vertices=int(fixed.sum()),global_boundary_ring=len(ring))

def numeric_checks(model,joint,single,reg,seed):
    rng=np.random.default_rng(365900+seed);q=rng.normal(size=(140,6))*.0123
    current=sample_valid(model,q)
    if current is None:raise ArithmeticError('Numerical probe outside valid patch')
    checks=[]
    for label,term in [('joint',joint),('single',single),('bilaplacian',reg)]:
        value,g=term.evaluate(current['points']);gq=point_to_q(model,current['jac'],g)
        for j in range(2):
            direction=rng.normal(size=q.shape);direction/=np.linalg.norm(direction)
            analytic=float(np.sum(gq*direction));errors=[];numeric=[]
            for h in [1e-4,1e-5,1e-6]:
                plus=model.sample(q+h*direction)[0];minus=model.sample(q-h*direction)[0]
                num=(term.evaluate(plus,False)-term.evaluate(minus,False))/(2*h)
                numeric.append(float(num));errors.append(abs(num-analytic))
            tolerance=2e-8+2e-3*abs(analytic)
            checks.append(dict(term=label,direction=j,analytic=analytic,numeric=numeric,errors=errors,tolerance=tolerance,passed=min(errors)<=tolerance))
    if not all(c['passed'] for c in checks):raise ArithmeticError('Real-data gradient check failed')
    return checks

def structure_checks(base,model,joint,single,seed):
    x=center_tensor(base.reshape(7,20,-1))
    def score(a):return sum(log_score(g,r,joint.delta)[0] for g,r in zip(mode_grams(a),joint.references))
    baseline=score(x);local=joint.evaluate(model.origin,False)
    rng=np.random.default_rng(365700+seed);common=rng.permutation(20)
    common_score=score(x[:,common]);perms=np.array([rng.permutation(20) for _ in range(7)])
    broken=np.array([x[s,perms[s]] for s in range(7)]);broken_score=score(broken)
    old=single.evaluate(model.origin,False)
    def single_full(a):
        value=[]
        for b in a:
            z=b-b.mean(0);g=z@z.T/(19*109.16412439918614)
            eig=np.linalg.eigvalsh((g+g.T)/2)[1:]
            value.append(float(np.log(np.maximum(eig,0)+1e-9).mean()))
        return float(np.mean(value))
    before_single=single_full(x);broken_single=single_full(broken)
    q=rng.normal(size=(140,6))*.015;points=model.sample(q)[0]
    candidate=base.copy();candidate[:,:,model.indices]=points.reshape(7,20,-1,3)
    direct=score(center_tensor(candidate.reshape(7,20,-1)));reduced=joint.evaluate(points,False)
    result=dict(seed=seed,joint_baseline=baseline,joint_common_permutation=common_score,
                joint_broken_pairing=broken_score,joint_pairing_delta=broken_score-baseline,
                single_baseline=before_single,single_broken_pairing=broken_single,
                joint_baseline_reduced_error=abs(baseline-local),joint_update_reduced_error=abs(direct-reduced),
                single_reduced_error=abs(before_single-old),joint_common_permutation_error=abs(baseline-common_score),
                single_broken_permutation_error=abs(before_single-broken_single),modes=joint.diagnostics())
    keys=['joint_baseline_reduced_error','joint_update_reduced_error','single_reduced_error','joint_common_permutation_error','single_broken_permutation_error']
    result['passed']=all(result[k]<1e-7 for k in keys)
    if not result['passed']:raise ArithmeticError('Structure/equivalence checks failed')
    return result

def run_seed(base,faces,seed,protocol):
    destination=OUT/'checkpoints'/f'seed_{seed}.json'
    if destination.exists():return json.loads(destination.read_text(encoding='utf-8'))
    model,chart=build_model(base,faces,seed)
    joint=JointEnergy(base,model.indices,protocol['delta']);single=SingleEnergy(base,model.indices,protocol['single_reference'])
    reg=BiLaplacian(model.base,faces,model.indices)
    ls=layouts(model);union=np.unique(np.concatenate(list(ls.values())))
    selection,audit=holdout_vertices(model,union)
    mask=np.ones(len(model.indices),bool);mask[np.r_[union,selection,audit]]=False
    sd=model.basis[selection].reshape(-1,6);ad=model.basis[audit].reshape(-1,6)
    structure=structure_checks(base,model,joint,single,seed);numeric=numeric_checks(model,joint,single,reg,seed)
    print(f'Seed {seed}: 20/7 mode spectra, gradient and permutation checks passed',flush=True)
    patterns={};amplitudes={}
    for j,pattern in enumerate(protocol['patterns']):
        rng=np.random.default_rng(365000+seed*10+j)
        q=rng.normal(size=(140,6)) if pattern=='independent' else np.tile(rng.normal(size=(20,6)),(7,1))
        q/=np.max(np.linalg.norm(np.einsum('jak,nk->nja',model.basis,q),axis=-1))
        halvings=0
        while sample_valid(model,q) is None:
            q*=.5;halvings+=1
            if halvings>12:raise ArithmeticError('No feasible corruption amplitude')
        patterns[pattern]=q;amplitudes[pattern]=halvings
    starts=[];rows=[];policies=[];subjects=[];parameters={}
    tests=[(p,s,l) for p in protocol['patterns'] for s in protocol['noise_sigma_diagonal'] for l in protocol['layouts']]
    tests += [('control',0.,l) for l in protocol['layouts']]
    for pattern,sigma,layout in tests:
        case=f'{seed}_{pattern}_{sigma:g}_{layout}';q_input=np.zeros((140,6)) if pattern=='control' else patterns[pattern]
        anchors=ls[layout];design=model.basis[anchors].reshape(-1,6);sigma_q=sigma/.002
        rng=np.random.default_rng(365300+seed);noise=rng.normal(size=(140,len(model.indices),2))*sigma_q
        y=noise[:,anchors].reshape(140,-1);ys=noise[:,selection].reshape(140,-1);ya=noise[:,audit].reshape(140,-1)
        q=exact_sparse(q_input,design,y);null,_,_=subspaces(design)
        projection=null if sigma==0 else np.eye(6)
        parameters[case+'_input']=q_input;parameters[case+'_start']=q;parameters[case+'_fit']=y;parameters[case+'_selection']=ys;parameters[case+'_audit']=ya
        initial=measure(model,q,mask,sd,ys,ad,ya)
        common=dict(case=case,seed=seed,pattern=pattern,sigma=sigma,layout=layout)
        feasible=sample_valid(model,q) is not None
        starts.append(dict(**common,feasible=feasible,nrmse=initial['metrics']['nrmse'] if initial else None,
                           normal_equation_inf=float(np.max(np.abs(observation(q,design,y)[1]))),
                           rank=int(np.linalg.matrix_rank(design)),nullity=int(round(np.trace(null)))))
        if not feasible:
            print(case+' invalid sparse start (retained)',flush=True);continue
        stay=dict(row=dict(**common,name='stay',family='stay',cap=0.,ratio=0.,accepted=True,**initial['metrics']),q=q,**initial)
        candidates=[stay]
        for family,ratio in [('regularizer',0.),('single',.1),('single',1.),('joint',.1),('joint',1.)]:
            term=None if family=='regularizer' else single if family=='single' else joint
            for cap in protocol['caps']:
                trial,detail=propose(model,reg,term,q,projection,design,y,sigma_q,ratio,cap)
                result=measure(model,trial,mask,sd,ys,ad,ya)
                name=f'{family}_{ratio:g}_{cap:g}';key=case+'_'+name;parameters[key]=trial
                row=dict(**common,name=name,family=family,cap=cap,ratio=ratio,parameter_key=key,**detail,**result['metrics'],
                         initial_nrmse=initial['metrics']['nrmse'],selection_residual=float(result['selection'].mean()),audit_residual=float(result['audit'].mean()))
                rows.append(row);candidates.append(dict(row=row,q=trial,**result))
        for cap in protocol['caps']:
            chosen=select_candidates(candidates,cap);reference=chosen['regularizer'];single_policy=chosen['single']
            ref=reference['metrics']['nrmse']
            for policy,c in chosen.items():
                error=c['metrics']['nrmse'];gain=ref-error
                policies.append(dict(**common,cap=cap,policy=policy,chosen=c['row']['name'],chosen_family=c['row']['family'],
                    **c['metrics'],initial_nrmse=initial['metrics']['nrmse'],baseline_nrmse=ref,absolute_gain=gain,
                    relative_gain=gain/max(ref,1e-12),gain_vs_single=single_policy['metrics']['nrmse']-error,
                    audit_gain=float((reference['audit']-c['audit']).mean()),selection_residual=float(c['selection'].mean()),
                    anchor_budget=2*sigma_q**2+1e-12,parameter_key=case+'_start' if c['row']['name']=='stay' else c['row']['parameter_key']))
                for state_index,state in enumerate(STATES):
                    for identity in range(20):
                        idx=state_index*20+identity
                        subjects.append(dict(**common,cap=cap,policy=policy,state=state,subject=f'ast{identity+1:03d}',
                          nrmse=float(c['each'][idx]),baseline_nrmse=float(reference['each'][idx]),
                          single_nrmse=float(single_policy['each'][idx]),initial_nrmse=float(initial['each'][idx]),
                          audit_residual=float(c['audit'][idx]),baseline_audit=float(reference['audit'][idx])))
        result=[r for r in policies if r['case']==case and r['policy']=='joint' and r['cap']==.01][0]
        print(f'{case}: joint={result["chosen"]}, gain={100*result["relative_gain"]:.5f}%',flush=True)
    parameters.update(indices=model.indices,basis=model.basis,fit_two=ls['two_v2'],fit_six=ls['six_spread'],selection_indices=selection,audit_indices=audit,evaluation_mask=mask)
    (OUT/'parameters').mkdir(exist_ok=True);np.savez_compressed(OUT/'parameters'/f'seed_{seed}.npz',**parameters)
    result=dict(seed=seed,chart=chart,structure=structure,numeric=numeric,regularizer_rows=reg.affected_rows,amplitude_halvings=amplitudes,starts=starts,candidates=rows,policies=policies,subjects=subjects)
    write_json(destination,result);return result

def aggregate(jobs,protocol):
    starts=sum([j['starts'] for j in jobs],[]);candidates=sum([j['candidates'] for j in jobs],[])
    policies=sum([j['policies'] for j in jobs],[]);subjects=sum([j['subjects'] for j in jobs],[])
    decisions=[];g=protocol['local_gate']
    def med(items):return float(np.median(items)) if items else 0.
    for layout in protocol['layouts']:
        rr=[r for r in policies if r['policy']=='joint' and r['cap']==.01 and r['layout']==layout and r['pattern']!='control']
        ss=[r for r in subjects if r['policy']=='joint' and r['cap']==.01 and r['layout']==layout and r['pattern']!='control']
        alternate=[r for r in policies if r['policy']=='joint' and r['cap']==.0025 and r['layout']==layout and r['pattern']!='control']
        controls=[r for r in policies if r['policy']=='joint' and r['layout']==layout and r['pattern']=='control']
        selected=sum(r['chosen_family']=='joint' for r in rr)
        harm=sum(r['nrmse']-r['baseline_nrmse']>max(1e-9,.01*r['baseline_nrmse']) for r in ss)/max(len(ss),1)
        tests=dict(feasible_fraction=len(rr)/12>=g['feasible_fraction_min'],
                   joint_used=selected/max(len(rr),1)>=g['selected_joint_fraction_min'],
                   relative_gain=med([r['relative_gain'] for r in rr])>g['median_relative_gain_min'],
                   absolute_gain=med([r['absolute_gain'] for r in rr])>g['median_absolute_gain_min'],
                   each_pattern=all(med([r['absolute_gain'] for r in rr if r['pattern']==p])>0 for p in protocol['patterns']),
                   noisy_gain=med([r['absolute_gain'] for r in rr if r['sigma']>0])>0,
                   audit_gain=med([r['audit_gain'] for r in rr])>0,
                   positive_states=sum(med([r['baseline_nrmse']-r['nrmse'] for r in ss if r['state']==s])>0 for s in STATES)>=5,
                   positive_seeds=sum(med([r['absolute_gain'] for r in rr if r['seed']==s])>0 for s in [0,1,2])>=2,
                   alternate_cap=med([r['absolute_gain'] for r in alternate])>0,
                   identity_harm=harm<=.2,geometry=all(r['geometry_valid'] for r in rr+controls),
                   variance=all(r['min_local_variance_ratio']>=.99 and r['min_global_variance_ratio']>=.99 for r in rr+controls),
                   control_drift=len(controls)==6 and all(r['nrmse']<=1e-9 for r in controls))
        opportunities=0;oracle_gains=[]
        for r in rr:
            pool=[c for c in candidates if c['case']==r['case'] and c['family']=='joint' and c['cap']==.01 and c['accepted']]
            best=min([c['nrmse'] for c in pool]+[r['baseline_nrmse']]);gain=r['baseline_nrmse']-best
            opportunities+=int(gain>max(1e-9,.01*r['baseline_nrmse']));oracle_gains.append(gain/max(r['baseline_nrmse'],1e-12))
        decisions.append(dict(layout=layout,expected=12,feasible=len(rr),joint_selected=selected,
                              median_relative_gain=med([r['relative_gain'] for r in rr]),median_absolute_gain=med([r['absolute_gain'] for r in rr]),
                              median_gain_vs_single=med([r['gain_vs_single'] for r in rr]),harmed_identity_fraction=harm,
                              oracle_material_opportunities=opportunities,oracle_median_gain=med(oracle_gains),tests=tests,passed=all(tests.values())))
    local_pass=any(r['passed'] for r in decisions)
    summary=dict(complete=True,counts=dict(starts=len(starts),feasible=sum(r['feasible'] for r in starts),candidates=len(candidates),policies=len(policies),subject_records=len(subjects)),
        decisions=decisions,local_gate_passed=local_pass,closed_loop_status='required_not_yet_run' if local_pass else 'skipped_by_frozen_gate',
        network_eligible=False,branch_status='pending_closed_loop' if local_pass else 'closed_no_further_investment',
        decision_zh='局部门槛通过，仍须完成冻结闭环条件；当前不可进入第二关。' if local_pass else '最后一次局部验算未通过：结束 MDL 驱动对应支线，不进入第二关 MLP，不再追加调参或扩数据。',
        result_zh=f'已运行{len(starts)}个队列，{sum(r["feasible"] for r in starts)}个初值几何有效；保留全部失败或拒绝情况。联合身份/表情分别使用20和7项完整谱。',
        limitations=protocol['limitations'],oracle_warning='Dense-truth oracle is posthoc and bounded only by the fixed candidate pool, never used for updates or selection.')
    for name,data in [('starts',starts),('candidates',candidates),('policies',policies),('subjects',subjects)]:write_csv(OUT/f'{name}.csv',data)
    write_json(OUT/'summary.json',summary);write_json(OUT/'structure_checks.json',[j['structure'] for j in jobs]);write_json(OUT/'gradient_checks.json',[dict(seed=j['seed'],checks=j['numeric']) for j in jobs])
    return summary

def plot(jobs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    policies=sum([j['policies'] for j in jobs],[]);rows=sum([j['candidates'] for j in jobs],[])
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for index,layout in enumerate(['two_v2','six_spread']):
        for k,family in enumerate(['single','joint']):
            pp=[r for r in policies if r['layout']==layout and r['policy']==family and r['cap']==.01 and r['pattern']!='control']
            x=index+(k-.5)*.22;values=[100*r['relative_gain'] for r in pp]
            axes[0].scatter(np.full(len(values),x),values,s=22,alpha=.65,label=family if index==0 else None)
            axes[0].plot([x-.07,x+.07],[np.median(values)]*2,color='black')
    axes[0].set(xticks=[0,1],xticklabels=['2 anchors','6 spread'],ylabel='Selected added gain vs regularizer (%)',title='Observable selection');axes[0].axhline(1,color='red',ls='--',lw=1);axes[0].legend()
    for family in ['regularizer','single','joint']:
        rr=[r for r in rows if r['family']==family and r['cap']==.01 and r['pattern']!='control' and r['accepted']]
        gains=[(r['initial_nrmse']-r['nrmse'])/max(r['initial_nrmse'],1e-12)*100 for r in rr]
        axes[1].scatter(np.full(len(gains),['regularizer','single','joint'].index(family)),gains,s=14,alpha=.6)
    axes[1].set(xticks=[0,1,2],xticklabels=['regularizer','single','joint'],ylabel='Raw one-step recovery gain (%)',title='Accepted candidates before selection');axes[1].axhline(0,color='black',lw=.8)
    control=[r for r in rows if r['pattern']=='control' and r['cap']==.01]
    for i,family in enumerate(['regularizer','single','joint']):
        values=[r['nrmse'] for r in control if r['family']==family]
        axes[2].scatter(np.full(len(values),i),np.maximum(values,1e-16),s=22)
    axes[2].set(xticks=[0,1,2],xticklabels=['regularizer','single','joint'],yscale='log',ylabel='Unselected no-corruption NRMSE',title='Baseline drift is still audited')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('S3.6 final: synthetic proxy observations, 20 previously viewed identities');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'final_decision.{ext}',dpi=170)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,choices=[0,1,2]);parser.add_argument('--aggregate',action='store_true');args=parser.parse_args()
    protocol=freeze();start=time.perf_counter()
    if args.aggregate:
        jobs=[json.loads((OUT/'checkpoints'/f'seed_{s}.json').read_text(encoding='utf-8')) for s in range(3)]
    else:
        base,faces=prepare();seeds=[args.seed] if args.seed is not None else [0,1,2]
        jobs=[run_seed(base,faces,s,protocol) for s in seeds]
        if args.seed is not None:return
    summary=aggregate(jobs,protocol);plot(jobs)
    write_json(OUT/'run_manifest.json',dict(elapsed_seconds=time.perf_counter()-start,files={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.npz','.npy','.png','.pdf') and p.name not in ('run_manifest.json','verification.json')}))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
