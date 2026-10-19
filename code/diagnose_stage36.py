"""Post-hoc explanation of failed S3.6 directions; never changes gate criteria."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from stage36_gradient import PatchEnergy,points,errors,full_shape,face_health,projected_step
from stage35_definition import write_json,write_csv,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='code/reports/stage3_s36_v1')
    parser.add_argument('--source',default='code/reports/stage3_s35_v2')
    args=parser.parse_args();out=Path(args.input);source=Path(args.source)
    protocol=json.loads((out/'protocol_config_s36.json').read_text(encoding='utf-8'))
    summaries=[];curve=[];clean_controls=[]
    for f in sorted((out/'checkpoints').glob('*.json')):
        data=json.loads(f.read_text(encoding='utf-8'))
        failed=[r for r in data['rows'] if r['t']>0 and not r['restoring_direction_pass']]
        if not failed:continue
        state,seed=data['state'],data['seed'];base=np.load(source/'cache'/f'all20_{state}_baseline.npy')
        with np.load(out/'patches'/f'{state}_{seed}.npz') as datafile:patch={k:datafile[k] for k in datafile.files}
        energy=PatchEnergy(base,patch['indices'],protocol['reference_variance'])
        def evaluate(t):
            p=points(patch,t*patch['endpoints']);value,g=energy.evaluate(p,True)
            derivative=np.sum(g*np.einsum('nmij,nmj->nmi',patch['edges'],patch['endpoints']))
            return value,float(derivative),errors(patch,p)[1]
        baseline=evaluate(0)[0]
        # Only inspect the already observed failed path and adjacent two grid cells.
        hi=min(1,max(r['t'] for r in failed)+.1)
        grid=np.linspace(0,hi,101)
        for t in grid:
            value,derivative,nrmse=evaluate(t)
            curve.append(dict(state=state,seed=seed,t=float(t),change=value-baseline,dE_dt=derivative,patch_nrmse=nrmse))
        signs=[evaluate(t)[1] for t in grid];roots=[]
        for i in range(len(grid)-1):
            if signs[i]<0<signs[i+1]:roots.append(float(brentq(lambda t:evaluate(t)[1],grid[i],grid[i+1],xtol=1e-12)))
        for r in failed:
            update=next(u for u in data['updates'] if u['t']==r['t'] and u['step_fraction']==.03)
            summaries.append(dict(state=state,seed=seed,t=r['t'],path_energy_minimum_t=roots,
                score_change_at_path_minimum=[evaluate(t)[0]-baseline for t in roots],
                rmse_at_path_minimum=[evaluate(t)[2] for t in roots],
                derivative=r['dE_dt'],regpa_derivative=r['regpa_fd_dE_dt'],
                energy_change_actual_update=update['score_after']-update['score_before'],
                relative_rmse_change=update['relative_rmse_change'],subjects_improved=update['subjects_improved'],
                local_variance_ratio=update['local_variance_ratio'],global_variance_ratio=update['variance_ratio'],
                normal_reversals_before=r['normal_reversals'],normal_reversals_after=update['normal_reversals']))
            # Post-hoc confounding diagnostic only: mask all patch vertices touching
            # an inverted/degenerate face at the observed failing strength.
            faces=np.load(source/'cache'/f'{state}_faces.npy')
            y=full_shape(base,patch['indices'],points(patch,r['t']*patch['endpoints']))
            a=base[:,faces];b=y[:,faces]
            na=np.cross(a[:,:,1]-a[:,:,0],a[:,:,2]-a[:,:,0]);nb=np.cross(b[:,:,1]-b[:,:,0],b[:,:,2]-b[:,:,0])
            bad=(np.sum(na*nb,axis=-1)<0)|(np.linalg.norm(nb,axis=-1)<1e-14)
            bad_vertices=np.unique(faces[bad.any(axis=0)])
            keep=~np.isin(patch['indices'],bad_vertices)
            clean={k:v.copy() for k,v in patch.items()}
            for k in ['origin','edges','endpoints','face_ids']:clean[k]=clean[k][:,keep]
            clean['indices']=clean['indices'][keep]
            ce=PatchEnergy(base,clean['indices'],protocol['reference_variance'])
            uv=r['t']*clean['endpoints'];cp=points(clean,uv);cv,cg=ce.evaluate(cp,True)
            derivative=float(np.sum(cg*np.einsum('nmij,nmj->nmi',clean['edges'],clean['endpoints'])))
            step,_=projected_step(clean,uv,cg,.03);newp=points(clean,step)
            clean_controls.append(dict(state=state,seed=seed,t=r['t'],post_hoc=True,
                removed_vertices=int((~keep).sum()),retained_vertices=int(keep.sum()),dE_dt=derivative,
                energy_step=ce.evaluate(newp)-cv,relative_rmse_change=errors(clean,newp)[1]/errors(clean,cp)[1]-1,
                before_health=face_health(base,full_shape(base,clean['indices'],cp),faces,clean['indices']),
                after_health=face_health(base,full_shape(base,clean['indices'],newp),faces,clean['indices'])))
    write_csv(out/'failure_zoom.csv',curve)
    write_json(out/'failure_diagnosis.json',dict(status='post_hoc_explanation_only',gate_unchanged=True,
        explanation='An energy minimum displaced from the truth explains why low-amplitude restoration can increase compactness energy. This is not numerical noise or proof against every shared neural parameterization.',
        failed_paths=summaries,clean_face_controls=clean_controls,source_protocol_sha256=sha(out/'protocol_config_s36.json'),code_sha256=sha(Path(__file__))))
    if curve:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        for state,seed in sorted({(r['state'],r['seed']) for r in curve}):
            rr=[r for r in curve if r['state']==state and r['seed']==seed]
            for ax,key in zip(axs,['change','dE_dt','patch_nrmse']):
                ax.plot([r['t'] for r in rr],[r[key] for r in rr],label=f'{state} seed {seed}');ax.set(xlabel='slide t',ylabel=key);ax.grid(alpha=.2)
        axs[1].axhline(0,color='black',lw=.7);axs[0].legend();fig.suptitle('Post-hoc failure diagnosis: energy minimum differs from correspondence truth')
        for ext in ['png','pdf']:fig.savefig(out/f'failure_zoom.{ext}',dpi=160)
        plt.close(fig)
    print(json.dumps(dict(failed_paths=summaries,clean_face_controls=clean_controls),indent=2))


if __name__=='__main__':main()
