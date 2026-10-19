"""Post-hoc audit of proxy-anchor identifiability and v2 effect-size limits.

No changed gate or optimized outcome is fed back into the original trials.
The anchor optimum is obtained from sparse design alone, not dense truth.
"""
import argparse,json,csv
from pathlib import Path
import numpy as np
from stage36_smooth import SmoothPatch,CACHE,OLD,OUT,STATES,sha,write_csv,write_json


def analyze(out):
    certificate=[];probes=[]
    for state in STATES:
        base=np.load(CACHE/f'all20_{state}_baseline.npy');faces=np.load(CACHE/f'{state}_faces.npy')
        for seed in range(3):
            checkpoint=out/'checkpoints'/f'{state}_{seed}.json'
            if not checkpoint.exists():continue
            j=json.loads(checkpoint.read_text(encoding='utf-8'))
            if j['status']!='complete':continue
            with np.load(OLD/'patches'/f'{state}_{seed}.npz') as z:indices=z['indices'];center=int(z['center'])
            model=SmoothPatch(base,faces,indices,center,109.16412439918614)
            with np.load(out/'parameters'/f'{state}_{seed}.npz') as z:q0=z['initial_q']
            matrix=model.design.reshape(-1,6)
            projection=np.eye(6)-np.linalg.pinv(matrix)@matrix
            optimum=q0@projection.T
            initial=model.evaluate(q0);endpoint=model.evaluate(optimum)
            feasible_path=all(model.evaluate((1-t)*q0+t*optimum) is not None for t in np.linspace(0,1,21))
            row=dict(state=state,seed=seed,design_rank=int(np.linalg.matrix_rank(matrix)),nullspace_per_identity=6-int(np.linalg.matrix_rank(matrix)),
                     initial_anchor=initial['anchor'],exact_anchor_optimum_residual=float(np.mean(np.sum(np.einsum('jak,nk->nja',model.design,optimum)**2,axis=-1))),
                     feasible_endpoint=endpoint is not None,feasible_21_point_path=feasible_path,
                     initial_nonanchor_nrmse=model.metrics(initial)[0]['nonanchor_nrmse'],
                     optimum_nonanchor_nrmse=model.metrics(endpoint)[0]['nonanchor_nrmse'] if endpoint is not None else None,
                     energy_gradient_at_optimum=float(np.linalg.norm(endpoint['ge'])) if endpoint is not None else None)
            certificate.append(row)
            for start_name,start in [('truth',np.zeros_like(optimum)),('exact_anchor_minimizer',optimum)]:
                current=model.evaluate(start)
                if current is None:
                    probes.append(dict(state=state,seed=seed,start=start_name,status='infeasible_start',
                                       accepted=False,energy_before=None,energy_after=None,anchor_after=None,
                                       nrmse_before=None,nrmse_after=None,step_fraction=None,projected_gradient_inf=None))
                    continue
                gradient=current['ge']@projection.T
                norm=float(np.max(np.abs(gradient)));direction=-gradient/max(norm,np.finfo(float).tiny)*.01
                slope=float(np.sum(current['ge']*direction));factor=1.;candidate=None
                for attempt in range(24):
                    trial=model.evaluate(start+factor*direction)
                    if trial is not None and trial['energy']<=current['energy']+1e-4*factor*slope:
                        candidate=trial;break
                    factor*=.5
                probes.append(dict(state=state,seed=seed,start=start_name,status='accepted' if candidate is not None else 'line_search_failed',
                                   accepted=candidate is not None,energy_before=current['energy'],energy_after=candidate['energy'] if candidate else None,
                                   anchor_after=candidate['anchor'] if candidate else None,nrmse_before=model.metrics(current)[0]['nonanchor_nrmse'],
                                   nrmse_after=model.metrics(candidate)[0]['nonanchor_nrmse'] if candidate else None,
                                   step_fraction=factor,projected_gradient_inf=norm))
    write_csv(out/'anchor_nullspace_certificate.csv',certificate)
    write_csv(out/'nullspace_energy_probes.csv',probes)
    with (out/'endpoints.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    with (out/'paired_endpoints.csv').open(encoding='utf-8-sig',newline='') as f:pairs=list(csv.DictReader(f))
    diagnostics=[]
    for arm in ['anchor_fixed_e','anchor_gated_e']:
        rr=[r for r in rows if r['arm']==arm and float(r['t'])==1]
        ratios=[]
        for r in rr:
            a=float(r['initial_anchor']);w=.1*(a/(a+1e-4) if arm=='anchor_gated_e' else 1)
            ratios.append(w*float(r['energy_gradient_norm'])/float(r['anchor_gradient_norm']))
        pp=[r for r in pairs if r['arm']==arm and float(r['t'])>0]
        diagnostics.append(dict(arm=arm,positive_conditions=len(pp),strictly_beats_anchor=sum(r['improves_over_anchor']=='True' for r in pp),
                               median_signed_nrmse_difference=float(np.median([float(r['delta_nrmse_vs_anchor']) for r in pp])),
                               t1_initial_weighted_gradient_ratio_median=float(np.median(ratios)),
                               t1_initial_weighted_gradient_ratio_min=float(min(ratios)),t1_initial_weighted_gradient_ratio_max=float(max(ratios))))
    checkpoint_rows=[json.loads(p.read_text(encoding='utf-8')) for p in (out/'checkpoints').glob('*.json')]
    complete=[j for j in checkpoint_rows if j['status']=='complete']
    report=dict(status='post_hoc_diagnostic_not_new_gate',certificate_cases=len(certificate),
                feasible_zero_anchor_nonzero_correspondence=sum(r['feasible_endpoint'] and r['exact_anchor_optimum_residual']<1e-20 and r['optimum_nonanchor_nrmse']>1e-10 for r in certificate),
                feasible_path_cases=sum(r['feasible_21_point_path'] for r in certificate),
                median_exact_anchor_optimum_nrmse=float(np.median([r['optimum_nonanchor_nrmse'] for r in certificate if r['feasible_endpoint']])),
                amplitude_halvings=[dict(state=j['state'],seed=j['seed'],halvings=j['amplitude_halvings'],vertices=j['patch_vertices']) for j in complete],
                arm_diagnostics=diagnostics,
                nullspace_probes={name:dict(conditions=len([r for r in probes if r['start']==name]),
                                          accepted=sum(r['accepted'] for r in probes if r['start']==name),
                                          energy_decreases=sum(r['accepted'] and r['energy_after']<r['energy_before']-1e-12 for r in probes if r['start']==name),
                                          correspondence_improves=sum(r['accepted'] and r['nrmse_after']<r['nrmse_before']-1e-12 for r in probes if r['start']==name),
                                          correspondence_worsens=sum(r['accepted'] and r['nrmse_after']>r['nrmse_before']+1e-12 for r in probes if r['start']==name),
                                          max_anchor_residual=max([r['anchor_after'] for r in probes if r['start']==name and r['accepted']],default=None)) for name in ['truth','exact_anchor_minimizer']},
                nullspace_probe_protocol='Post-hoc single-step E/K descent in exact sparse-anchor nullspace; coefficient infinity step 0.01, Armijo 1e-4, 24 halvings and unchanged geometry guard. Uses no true recovery direction or dense error for step. Not a convergence or independent generalization experiment.',
                interpretation=['A=0 and gated gradient=0 are insufficient for dense correspondence; sparse-anchor nullspace remains.',
                                'The exact anchor minimizer uses only the sparse observation design; dense truth is used only to measure its remaining error.',
                                'Small observed combined-arm differences cannot establish utility: fixed lambda was exploratory and actual gradient ratios must be considered.',
                                'Initial amplitudes differ after geometry-only halving; t is relative within a path, not a shared physical corruption size.',
                                'Solver iteration limits are not convergence; this diagnostic does not replace or relabel any original endpoint.'])
    write_json(out/'mechanism_diagnosis.json',report)
    files=['anchor_nullspace_certificate.csv','nullspace_energy_probes.csv','mechanism_diagnosis.json']
    write_json(out/'diagnostic_manifest.json',dict(code_sha256=sha(__file__),source_manifest_sha256=sha(out/'manifest.json'),
                                                outputs={n:sha(out/n) for n in files}))
    print(json.dumps({k:v for k,v in report.items() if k!='amplitude_halvings'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default=str(OUT));analyze(Path(p.parse_args().input))
