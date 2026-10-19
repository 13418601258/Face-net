"""Supplement the frozen v3 run with correctly remapped topology and plots.

landmark_audit.json compared raw full-BFM face indices directly with cropped
release indices. Its false equality flags are not evidence of different
topology. Candidate landmark error calculations already used the correct
full-to-cropped lookup and are unaffected. Preserve original files explicitly.
"""
import csv
import json
from pathlib import Path
import numpy as np
from astface_loader import ASTFaceReader
from stage36_smooth import ROOT, sha, write_json

OUT = ROOT/'code/reports/stage3_s36_v3_constraints'


def read(name):
    with (OUT/name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def main():
    rows, designs, probes = (read(n) for n in ('constraint_results.csv','designs.csv','scaled_energy_steps.csv'))
    lines = (OUT/'sources/cut_bfm.txt').read_text().splitlines()
    retained = np.array([int(v)-1 for v in lines[0].split(':')[1].split(',') if v.strip()])
    faces = np.array([[int(v)-1 for v in line.split()[1:]] for line in lines if line.startswith('f ')])
    lookup = np.full(retained.max()+1, -1, dtype=int)
    lookup[retained] = np.arange(len(retained))
    remapped = lookup[faces]
    release = ASTFaceReader(ROOT/'data/ASTFace_Public').read_mesh('ast001','au0')
    exact = bool(np.array_equal(remapped, release.faces))
    assert exact and np.all(remapped >= 0)
    layout_names = ['two_v2','three_spread','six_spread','three_clustered']
    conditions = []
    summaries = []
    for layout in layout_names:
        dd = [r for r in designs if r['layout']==layout]
        values = [float(r['condition']) for r in dd if r['condition']]
        conditions.append(dict(layout=layout, ranks=sorted(set(int(r['rank']) for r in dd)),
                               condition_median=float(np.median(values)) if values else None))
        for sigma in [0.,1e-6,1e-5,1e-4]:
            # Noiseless repeats contain no new information: retain exactly one.
            rr = [r for r in rows if r['layout']==layout and float(r['sigma'])==sigma and (sigma>0 or r['replicate']=='0')]
            sampled = [r for r in rr if r['sampled']=='True']
            valid = [r for r in rr if r['geometry_valid']=='True']
            summaries.append(dict(layout=layout, sigma=sigma, records=len(rr), sampled=len(sampled), valid=len(valid),
                                  median_all_sampleable_nrmse=float(np.median([float(r['nrmse']) for r in sampled])),
                                  median_valid_nrmse=float(np.median([float(r['nrmse']) for r in valid])) if valid else None))
    report = dict(topology=dict(full_template_vertices_retained=len(retained), faces=len(faces),
                  full_bfm_face_index_range=[int(faces.min()),int(faces.max())],
                  remapped_face_index_range=[int(remapped.min()),int(remapped.max())],
                  exact_release_topology_match=exact,
                  correction='Original landmark_audit.json false equality flags compare different index domains. After crop reindexing the faces match exactly; candidate landmark errors already used the correct lookup and remain unchanged.'),
                  designs=conditions, noise_results=summaries, recorded_conditions=len(rows),
                  distinct_conditions_without_noiseless_repeats=sum(s['records'] for s in summaries),
                  interpretation=['Noiseless three-replicate controls are duplicates, not independent observations.',
                    'Noisy geometry-invalid optima are not successful recoveries; display invalid counts and separate sampleable/valid medians.',
                    'At sigma=1e-5, three-spread interpolation has zero observation residual despite dense error; six-spread least-squares has a nonzero noise floor, so residual-only gate remains active.',
                    'Gradient-balanced E probes are single-step, not a converged noise-robust benefit claim.',
                    'Template crop indexing is resolved; BFM keypoint equivalence and per-sample release coordinate transform are not.'],
                  network_eligible=False)
    write_json(OUT/'review.json', report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(13,8))
    for name in layout_names:
        rr = [r for r in summaries if r['layout']==name and r['sigma']>0]
        sigmas=[r['sigma'] for r in rr]
        axes[0,0].plot(sigmas,[r['median_all_sampleable_nrmse'] for r in rr],marker='o',label=name)
        axes[0,1].plot(sigmas,[100*r['valid']/r['records'] for r in rr],marker='o',label=name)
        xx=[.01,.1,1.]
        yy=[np.median([float(r['delta_vs_anchor']) for r in probes if r['layout']==name and float(r['requested_gradient_ratio'])==v]) for v in xx]
        axes[1,1].plot(xx,yy,marker='o',label=name)
    axes[0,0].set(xscale='log',yscale='log',xlabel='UV noise sigma / mean mesh diagonal',ylabel='Median NRMSE',title='All sampleable exact sparse optima')
    axes[0,1].set(xscale='log',xlabel='UV noise sigma / mean mesh diagonal',ylabel='Geometry-valid groups (%)',ylim=(-3,103),title='Invalid optima are failures, not recoveries')
    rr=[r for r in rows if float(r['sigma'])==1e-5]
    xx=np.arange(4);width=.35
    at=[np.median([float(r['gate_at_truth']) for r in rr if r['layout']==n]) for n in layout_names]
    opt=[np.median([float(r['gate_at_sparse_optimum']) for r in rr if r['layout']==n]) for n in layout_names]
    axes[1,0].bar(xx-width/2,at,width,label='At proxy truth')
    axes[1,0].bar(xx+width/2,opt,width,label='At exact sparse optimum')
    axes[1,0].set(xticks=xx,xticklabels=['2 old','3 spread','6 spread','3 clustered'],ylabel='Residual-based gate',title='Noise sigma=1e-5: residual is not correctness')
    axes[1,0].legend(fontsize=8)
    axes[1,1].set(xscale='log',xlabel='Initial weighted E / anchor gradient',ylabel='NRMSE difference vs anchor step',title='Single-step effect (positive = worse)')
    axes[1,1].axhline(0,color='black',linewidth=.7)
    axes[0,0].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('S3.6 v3: 21 paths x 3 noise draws; synthetic constraints, no neural release')
    fig.text(.5,.01,'Error medians include sampleable invalid geometry; consult validity panel. At highest clustered noise only 23/63 remain sampleable.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.96))
    for suffix in ('png','pdf'):fig.savefig(OUT/f'constraint_review.{suffix}',dpi=180)
    plt.close(fig)
    write_json(OUT/'review_manifest.json', dict(code_sha256=sha(__file__),
        source_manifest_sha256=sha(OUT/'manifest.json'),
        files={p:sha(OUT/p) for p in ['review.json','constraint_review.png','constraint_review.pdf']}))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
