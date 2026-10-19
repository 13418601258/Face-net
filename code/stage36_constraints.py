"""S3.6 v3: sparse constraint rank/noise, scaled E probes, landmark provenance.

Independent output directory; v1/v2 are read-only. Exact synthetic observations
are explicitly not independent anatomical truth. No archive reads.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import ast
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from stage36_smooth import (SmoothPatch, CACHE, OLD, OUT as V2, ROOT, STATES,
                            local_health, sha, write_csv, write_json)
from astface_loader import ASTFaceReader

OUT = ROOT / 'code/reports/stage3_s36_v3_constraints'
CONFIG = ROOT / 'code/stage36_constraints_protocol.json'


def layouts(model):
    candidates = np.flatnonzero(np.linalg.norm(model.start/model.radius, axis=1) < .65)
    selected = list(model.anchors)
    saved = {'two_v2': np.array(selected)}
    scalar = model.basis[:, 0, :3]
    while len(selected) < 6:
        remaining = [int(i) for i in candidates if i not in selected]
        best = max(remaining, key=lambda i: np.linalg.svd(scalar[selected+[i]], compute_uv=False)[-1])
        selected.append(best)
        if len(selected) == 3:
            saved['three_spread'] = np.array(selected)
    saved['six_spread'] = np.array(selected)
    saved['three_clustered'] = np.argsort(np.linalg.norm(model.start, axis=1), kind='stable')[:3]
    return saved


def observation(q, design, target):
    residual = q @ design.T - target
    a = float(np.mean(np.sum(residual**2, axis=1))/(design.shape[0]/2))
    ga = 2*residual@design/(len(q)*(design.shape[0]/2))
    return a, ga


def exact_sparse(q0, design, target):
    inverse = np.linalg.pinv(design, rcond=1e-12)
    projection = np.eye(design.shape[1])-inverse@design
    return target@inverse.T + q0@projection.T


def geometry_metrics(model, q, evaluation_mask):
    sample = model.sample(q)
    if sample is None:
        return dict(sampled=False, geometry_valid=False, normal_reversals=None,
                    min_area_ratio=None, max_displacement_ratio=None, nrmse=None), None
    points = sample[0]
    faces = model.face_reference.copy()
    faces[:, model.face_mask] = points[:, model.face_slots[model.face_mask]]
    health = local_health(model.face_reference, faces, model.diagonals)
    errors = np.linalg.norm(points-model.origin, axis=-1)/model.diagonals[:, None]
    each = np.sqrt(np.mean(errors[:, evaluation_mask]**2, axis=1))
    maximum = float(errors.max())
    return dict(sampled=True, geometry_valid=bool(health['valid'] and maximum <= .01),
                normal_reversals=health['normal_reversals'], min_area_ratio=health['min_area_ratio'],
                max_displacement_ratio=maximum, nrmse=float(each.mean())), each


def step_probe(model, q, design, weight):
    e = model.evaluate(q)
    target = np.zeros((len(q), len(design)))
    a, ga = observation(q, design, target)
    gradient = ga+weight*e['ge']
    direction = -gradient/max(float(np.max(np.abs(gradient))), np.finfo(float).tiny)*.01
    slope = float(np.sum(gradient*direction))
    objective = a+weight*e['energy']
    for k in range(24):
        trial = q+(2.**(-k))*direction
        candidate = model.evaluate(trial)
        if candidate is None:
            continue
        new_a, _ = observation(trial, design, target)
        if new_a+weight*candidate['energy'] <= objective+1e-4*(2.**(-k))*slope:
            return trial, dict(accepted=True, halvings=k, anchor_after=new_a,
                               energy_after=candidate['energy'])
    return q.copy(), dict(accepted=False, halvings=24, anchor_after=a, energy_after=e['energy'])


def run_constraints(out):
    rows, designs, probes, subjects = [], [], [], []
    protocol = json.loads(CONFIG.read_text(encoding='utf-8'))
    for si, state in enumerate(STATES):
        base = np.load(CACHE/f'all20_{state}_baseline.npy')
        faces = np.load(CACHE/f'{state}_faces.npy')
        for seed in range(3):
            with np.load(OLD/'patches'/f'{state}_{seed}.npz') as z:
                indices, center = z['indices'], int(z['center'])
            model = SmoothPatch(base, faces, indices, center, 109.16412439918614)
            with np.load(V2/'parameters'/f'{state}_{seed}.npz') as z:
                q0 = z['initial_q']
            ls = layouts(model)
            mask = np.ones(len(model.indices), bool)
            mask[np.unique(np.concatenate(list(ls.values())))] = False
            initial, _ = geometry_metrics(model, q0, mask)
            for layout, anchors in ls.items():
                design = model.basis[anchors].reshape(-1, 6)
                singular = np.linalg.svd(design, compute_uv=False)
                rank = int(np.sum(singular > singular[0]*1e-12))
                designs.append(dict(state=state, seed=seed, layout=layout, anchors=len(anchors),
                                    rank=rank, nullity=6-rank,
                                    condition=float(singular[0]/singular[-1]) if rank==6 else None,
                                    observed_condition=float(singular[0]/singular[rank-1]),
                                    indices=' '.join(map(str, model.indices[anchors]))))
                for sigma in protocol['noise_sigma_chart_diag']:
                    for replicate in protocol['noise_replicates']:
                        # Draw at every patch vertex before selecting anchors: shared-point noise is paired.
                        rng = np.random.default_rng(363000+si*1000+seed*100+replicate)
                        noise = rng.normal(size=(20, len(model.indices), 2))
                        target = (noise[:, anchors]*(sigma/.002)).reshape(20, -1)
                        optimum = exact_sparse(q0, design, target)
                        a, ga = observation(optimum, design, target)
                        at, _ = observation(np.zeros_like(q0), design, target)
                        result, each = geometry_metrics(model, optimum, mask)
                        row = dict(state=state, seed=seed, layout=layout, sigma=sigma, replicate=replicate,
                                   rank=rank, nullity=6-rank, initial_nrmse=initial['nrmse'],
                                   anchor_residual=a, normal_equation_inf=float(np.max(np.abs(ga))),
                                   gate_at_sparse_optimum=a/(a+1e-4), gate_at_truth=at/(at+1e-4), **result)
                        rows.append(row)
                        if each is not None:
                            for n, value in enumerate(each, 1):
                                subjects.append(dict(state=state, seed=seed, layout=layout, sigma=sigma,
                                                     replicate=replicate, subject=f'ast{n:03}',
                                                     nrmse=float(value), group_geometry_valid=result['geometry_valid']))
                target = np.zeros((20, len(design)))
                a, ga = observation(q0, design, target)
                e = model.evaluate(q0)
                anchor_q, control = step_probe(model, q0, design, 0.)
                control_metrics, _ = geometry_metrics(model, anchor_q, mask)
                for ratio in [0., .01, .1, 1.]:
                    weight = ratio*np.linalg.norm(ga)/np.linalg.norm(e['ge'])
                    candidate, detail = step_probe(model, q0, design, weight)
                    m, _ = geometry_metrics(model, candidate, mask)
                    probes.append(dict(state=state, seed=seed, layout=layout, requested_gradient_ratio=ratio,
                                       weight=float(weight), initial_nrmse=initial['nrmse'], initial_anchor=a,
                                       initial_energy=e['energy'], control_nrmse=control_metrics['nrmse'],
                                       delta_vs_anchor=m['nrmse']-control_metrics['nrmse'], **detail, **m))
            print(f'constraint case complete: {state}/{seed}', flush=True)
    write_csv(out/'designs.csv', designs)
    write_csv(out/'constraint_results.csv', rows)
    write_csv(out/'constraint_subjects.csv', subjects)
    write_csv(out/'scaled_energy_steps.csv', probes)
    summary = []
    for layout in protocol['layouts']:
        for sigma in protocol['noise_sigma_chart_diag']:
            rr = [r for r in rows if r['layout']==layout and r['sigma']==sigma]
            mm = [r['nrmse'] for r in rr if r['sampled']]
            summary.append(dict(layout=layout, sigma=sigma, count=len(rr), sampled=len(mm),
                                geometry_valid=sum(r['geometry_valid'] for r in rr),
                                median_nrmse=float(np.median(mm)) if mm else None,
                                median_gate_at_truth=float(np.median([r['gate_at_truth'] for r in rr])),
                                median_gate_at_optimum=float(np.median([r['gate_at_sparse_optimum'] for r in rr])),
                                median_anchor_residual=float(np.median([r['anchor_residual'] for r in rr]))))
    energy_summary = []
    for layout in protocol['layouts']:
        for ratio in [.01, .1, 1.]:
            rr = [r for r in probes if r['layout']==layout and r['requested_gradient_ratio']==ratio]
            energy_summary.append(dict(layout=layout, gradient_ratio=ratio, count=len(rr),
                                       accepted=sum(r['accepted'] for r in rr),
                                       beats_anchor=sum(r['delta_vs_anchor'] < -1e-12 for r in rr),
                                       worsens_vs_anchor=sum(r['delta_vs_anchor'] > 1e-12 for r in rr),
                                       median_delta_vs_anchor=float(np.median([r['delta_vs_anchor'] for r in rr])),
                                       median_weight=float(np.median([r['weight'] for r in rr]))))
    write_json(out/'constraint_summary.json', dict(complete=True, cases=len(rows), designs=len(designs),
               energy_steps=len(probes), by_layout_noise=summary, energy_steps_summary=energy_summary,
               max_normal_equation_inf=max(r['normal_equation_inf'] for r in rows), network_eligible=False,
               interpretation='Synthetic rank/noise and single-step weight diagnostics; no converged E comparison or independent landmark validation.'))
    return rows, designs, probes


def similarity(source, target):
    x = source-source.mean(0); y = target-target.mean(0)
    u, singular, vt = np.linalg.svd(x.T@y)
    correction = np.ones(3); correction[-1] = np.linalg.det(u@vt)
    rotation = (u*correction)@vt
    scale = float(np.sum(singular*correction)/np.sum(x*x))
    translation = target.mean(0)-scale*source.mean(0)@rotation
    return rotation, scale, translation


def landmark_audit(out):
    sources = out/'sources'
    reader = ASTFaceReader(ROOT/'data/ASTFace_Public')
    lines = (sources/'cut_bfm.txt').read_text().splitlines()
    kept = np.array([int(s) for s in lines[0].split(':')[1].split(',') if s.strip()])
    mapping_faces = np.array([[int(s)-1 for s in line.split()[1:]] for line in lines if line.startswith('f ')])
    candidate_indices = np.loadtxt(sources/'bfm_landmarks_withear.txt', dtype=float).astype(int)
    tree = ast.parse((sources/'astface_edge_nicp.py').read_text(encoding='utf-8'))
    mappings = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('landmark_id_84', 'landmark_id_68')}
    lookup = {int(v-1): i for i, v in enumerate(kept)}
    mesh0 = reader.read_mesh('ast001', 'au0')
    topology_equal = bool(mapping_faces.shape == mesh0.faces.shape and np.array_equal(mapping_faces, mesh0.faces))
    set_equal = bool(mapping_faces.shape == mesh0.faces.shape and
                     set(map(tuple, np.sort(mapping_faces, axis=1))) == set(map(tuple, np.sort(mesh0.faces, axis=1))))
    rows = []
    for n in range(1, 21):
        subject = f'ast{n:03}'
        for state in STATES:
            mesh = reader.read_mesh(subject, state)
            landmarks = reader.read_landmarks(subject, state)
            diag = float(np.linalg.norm(np.ptp(mesh.vertices, axis=0)))
            raw_distance = cKDTree(mesh.vertices).query(landmarks)[0]
            for index_base in [0, 1]:
                original = candidate_indices[mappings['landmark_id_68']]-index_base
                available = np.array([int(v) in lookup for v in original])
                a = landmarks[np.array(mappings['landmark_id_84'])[available]]
                b = mesh.vertices[[lookup[int(v)] for v in original[available]]]
                fit = np.arange(len(a)) % 2 == 0
                rotation, scale, translation = similarity(a[fit], b[fit])
                transformed = scale*a@rotation+translation
                errors = np.linalg.norm(transformed-b, axis=1)/diag
                rows.append(dict(subject=subject, state=state, candidate_index_base=index_base,
                                 mesh_diag=diag, landmark_diag=float(np.linalg.norm(np.ptp(landmarks, axis=0))),
                                 raw_nearest_vertex_median=float(np.median(raw_distance)), available_pairs=int(available.sum()),
                                 fitted_scale=scale, rotation_determinant=float(np.linalg.det(rotation)),
                                 fit_nrmse=float(np.sqrt(np.mean(errors[fit]**2))),
                                 heldout_nrmse=float(np.sqrt(np.mean(errors[~fit]**2))),
                                 heldout_max_error=float(errors[~fit].max())))
        print(f'landmark audit complete: {subject}', flush=True)
    write_csv(out/'landmark_candidates.csv', rows)
    result = dict(sample_count=140, candidate_rows=len(rows), crop_vertices=len(kept),
                  crop_faces=len(mapping_faces), exact_faces_match=topology_equal, unoriented_faces_match=set_equal,
                  candidate_keypoint_count=len(candidate_indices),
                  candidates=[dict(index_base=b, count=len(rr), median_heldout_nrmse=float(np.median([r['heldout_nrmse'] for r in rr])),
                                   median_fit_nrmse=float(np.median([r['fit_nrmse'] for r in rr])),
                                   fitted_scale_min=min(r['fitted_scale'] for r in rr), fitted_scale_max=max(r['fitted_scale'] for r in rr),
                                   available_pairs_min=min(r['available_pairs'] for r in rr))
                              for b in [0,1] for rr in [[r for r in rows if r['candidate_index_base']==b]]],
                  status='candidate_coordinate_audit_only_not_independent_validation',
                  blockers=['BFM.mat kpt_ind equivalence to public text index list is not established.',
                            'Sample-specific release similarity transforms and original fitted templates are absent from extracted subject folders.',
                            'Landmarks participate in original registration; agreement with registered meshes is not independent anatomical accuracy.',
                            'No fitted scale is applied to the experimental baseline or used for downstream recovery.'],
                  source_manifest_sha256=sha(sources/'sources.json'))
    write_json(out/'landmark_audit.json', result)
    return result


def plot(out, rows, designs, probes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    order = ['two_v2', 'three_spread', 'six_spread', 'three_clustered']
    for name in order:
        rr = [r for r in rows if r['layout']==name]
        sigmas = sorted(set(r['sigma'] for r in rr))
        yy = [np.median([r['nrmse'] for r in rr if r['sigma']==s and r['sampled']]) for s in sigmas]
        axes[0].plot(sigmas, np.maximum(yy, 1e-16), marker='o', label=name)
        xx = [.01, .1, 1.]
        yy = [np.median([r['delta_vs_anchor'] for r in probes if r['layout']==name and r['requested_gradient_ratio']==s]) for s in xx]
        axes[2].plot(xx, yy, marker='o', label=name)
    axes[0].set_xscale('symlog', linthresh=1e-6); axes[0].set_yscale('log')
    axes[0].set(xlabel='UV noise sigma / mean mesh diagonal', ylabel='Median non-anchor NRMSE', title='Exact sparse fit: noise sensitivity')
    values = [[r['condition'] for r in designs if r['layout']==name and r['condition'] is not None] for name in order[1:]]
    axes[1].boxplot(values, tick_labels=['3 spread', '6 spread', '3 clustered'])
    axes[1].set(yscale='log', ylabel='Design condition number', title='Full rank is not sufficient')
    axes[2].set_xscale('log'); axes[2].axhline(0, color='black', linewidth=.8)
    axes[2].set(xlabel='Initial weighted E / anchor gradient', ylabel='NRMSE difference vs anchor step', title='Single-step audit (positive = worse)')
    for ax in axes: ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('S3.6 v3: synthetic observations; no network eligibility', fontsize=12)
    fig.tight_layout()
    for suffix in ['png','pdf']: fig.savefig(out/f'constraint_diagnostics.{suffix}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['all','constraints','landmarks'], default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [Path(__file__), CONFIG, ROOT/'code/stage36_smooth.py', ROOT/'code/stage36_gradient.py',
              V2/'manifest.json', OUT/'sources/sources.json']
    inputs += sorted((V2/'parameters').glob('*.npz'))
    frozen = dict(protocol=json.loads(CONFIG.read_text(encoding='utf-8')),
                  hashes={str(p.relative_to(ROOT)): sha(p) for p in inputs})
    freeze = OUT/'frozen_protocol.json'
    if freeze.exists() and json.loads(freeze.read_text(encoding='utf-8')) != frozen:
        raise ValueError('Frozen source changed; use a new experiment output/version.')
    if not freeze.exists(): write_json(freeze, frozen)
    if args.mode in ('all','constraints'):
        rows, designs, probes = run_constraints(OUT)
        plot(OUT, rows, designs, probes)
    if args.mode in ('all','landmarks'): landmark_audit(OUT)
    outputs = [p for p in OUT.iterdir() if p.is_file() and p.name not in ('manifest.json','verification.json')]
    write_json(OUT/'manifest.json', dict(files={p.name:sha(p) for p in outputs}))


if __name__ == '__main__':
    main()
