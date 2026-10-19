"""S3.5 v2: reproducible, resumable boundary experiments (extracted meshes only).

Scores retain the frozen v3 convention. Geometry uses 2048 surface samples
on every all20 identity/condition; subcohorts supply the T2 stability checks.
Run from the project root. Checkpoints are rejected if code/config changes.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from astface_loader import ASTFaceReader
from ecomp import covariance_spectrum, ecomp
from geometry_batch import evaluate_batch_geometry
from index_protocol import permutation, sample_seed
from perturb import smooth_local_deformation
from preprocessing import rigid_gpa, match_group_frame

STATES = ['au0', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']
SEEDS = [0, 1, 2]
ORIGINAL_LEVELS = [.001, .003, .010, .030]
ADDED_LEVELS = [.014, .020, .045]
SMOOTH_LEVELS = sorted(ORIGINAL_LEVELS + ADDED_LEVELS)
GLOBAL_LEVELS = [.005, .01, .02, .04, .08, .16, .32, .64]
LOCAL_LEVELS = [.001, .003, .005, .01, .02, .03, .05]
RESTORE_LEVELS = [0., .2, .4, .6, .8, 1.]
SHRINK_LEVELS = [1., .8, .6, .4, .2, .1, 0.]
SUBJECTS = [f'ast{i:03d}' for i in range(1, 21)]
COHORTS = {'base15': (0, 15), 'new5': (15, 20), 'all20': (0, 20)}
SAMPLES = 2048
DELTA = 1e-9
ATOL = 1e-8
RTOL = 1e-10
VERSION = 'stage3-s35-v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def fixed_score(x, reference):
    return ecomp(covariance_spectrum(x), n_components=len(x)-1, reference_variance=reference, delta=DELTA)


def prepare(data_dir, out):
    """Hash all 140 actual source files; cache raw/aligned arrays once."""
    cache = out / 'cache'; cache.mkdir(exist_ok=True)
    with ASTFaceReader(data_dir) as reader:
        paths = {}
        for state in STATES:
            for subject in SUBJECTS:
                # Resolve through the unified reader, including spelling aliases.
                key = (subject, state)
                paths[key] = reader.read_sample(subject, state, modalities=('mesh',)).mesh
        inputs = []
        for (subject, state), mesh in paths.items():
            path = reader._root / mesh.path
            inputs.append(dict(subject=subject, state=state, path=mesh.path, sha256=sha(path)))
        digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        marker = cache / 'metadata.json'
        if marker.exists():
            meta = json.loads(marker.read_text(encoding='utf-8'))
            if meta['input_digest'] != digest:
                raise ValueError('Input data changed; use a new output directory')
            for filename, expected in meta['cache_sha256'].items():
                if sha(cache / filename) != expected:
                    raise ValueError('Cache mismatch: ' + filename)
            return meta
        for state in STATES:
            meshes = [paths[(s, state)] for s in SUBJECTS]
            faces = np.asarray(meshes[0].faces, dtype=np.int64)
            if not all(np.array_equal(faces, m.faces) for m in meshes):
                raise ValueError('Topology mismatch: ' + state)
            np.save(cache / f'{state}_faces.npy', faces)
            np.save(cache / f'{state}_raw.npy', np.array([m.vertices for m in meshes], dtype=float))
        meta = dict(input_digest=digest, inputs=inputs, cohorts={}, gpa=[])
        for name, (lo, hi) in COHORTS.items():
            aligned = []
            for state in STATES:
                x = np.load(cache / f'{state}_raw.npy')[lo:hi]
                gpa = rigid_gpa(x)
                if not gpa.converged:
                    raise ValueError(f'Baseline GPA did not converge: {name}/{state}')
                np.save(cache / f'{name}_{state}_baseline.npy', gpa.aligned)
                aligned.append(gpa.aligned)
                meta['gpa'].append(dict(cohort=name, state=state, iterations=gpa.iterations, relative_change=gpa.relative_change))
            reference = float(covariance_spectrum(np.concatenate(aligned)).sum())
            meta['cohorts'][name] = dict(N=hi-lo, K=hi-lo-1, subjects=SUBJECTS[lo:hi], reference_variance=reference,
                baseline_scores={s: fixed_score(x, reference) for s, x in zip(STATES, aligned)})
            print(f'Prepared {name}, K={hi-lo-1}, reference={reference:.12g}', flush=True)
        meta['cache_sha256'] = {p.name: sha(p) for p in sorted(cache.glob('*.npy'))}
        write_json(marker, meta)
    return meta


def config(meta, source):
    names = ['stage35_definition.py', 'astface_loader.py', 'preprocessing.py', 'ecomp.py',
             'perturb.py', 'index_protocol.py', 'geometry_batch.py', 'metrics.py']
    return dict(version=VERSION, states=STATES, seeds=SEEDS, cohorts=meta['cohorts'],
        input_digest=meta['input_digest'], original_levels=ORIGINAL_LEVELS, added_levels=ADDED_LEVELS,
        global_levels=GLOBAL_LEVELS, local_levels=LOCAL_LEVELS, restoration_levels=RESTORE_LEVELS, shrink_levels=SHRINK_LEVELS,
        geometry_samples=SAMPLES, geometry_seed=0, geometry_tolerance_fraction=.001,
        geometry_scope='T2 and T5: every all20 identity and every condition; T2 subcohorts: scores only',
        preprocessing='T2-T4: rigid GPA, no scale/reflection; T5: frozen baseline GPA frame, no re-GPA',
        smooth_seed='sample_seed(seed, subject/state/smooth_local_deformation), identical to v3',
        index_seed='sample_seed(seed, subject/state/family), nested sets within each seed',
        K_policy='K=N-1, all eigenvalues including zeros; delta/K does not remove cohort/rank effects',
        T1='Keep individual negative counterexamples; add mean<0 and -mean>2*sampleSD reproducibility label; not a significance test',
        T2='Positive signed adjacent step > SD or 2SD; report old, added and full grids separately; identity rho>=.8, signs match, both nondecreasing',
        T3='Detectable: mean delta>2SD and every seed positive; robust step: paired adjacent mean>3*pairedSD; report each state',
        T4='Every seed checked for nonincreasing score; exact identity endpoint, tolerance atol=1e-8+rtol=1e-10*abs(baseline)',
        T5='L=delta_E/K + lambda_geom*G; G=(1-coverage)+symmetric_distance/tolerance+normal_rate; lambda>=(-delta_E/K)/G for tested negative candidates only',
        T5_equivalent='For L=G+mu_comp*delta_E/K, mu_comp<=G/(-delta_E/K); bounds cover finite sampled paths only, not all collapse',
        T5_geometry_threshold='No external acceptability threshold invented; compare each sampled candidate with baseline',
        numerical_tolerance=dict(atol=ATOL, rtol=RTOL), delta=DELTA,
        source_hashes={str(p): sha(p) for p in [source/'experiments.csv', source/'geometry_subjects.csv', source.parent/'stage3_stability_v1/cohort_scores.csv']},
        code_sha256={n: sha(Path(__file__).parent/n) for n in names},
        limitations=['three perturbation seeds are not independent identity replicates', 'all20 overlaps base15 and new5',
                     'T3/T4 discrete permutation ordering does not demonstrate a useful continuous training gradient',
                     'standardized AST meshes are a proxy for correct correspondence, not external anatomical ground truth'])


def t1(source):
    groups = defaultdict(list); individuals = []
    for r in read_csv(source/'experiments.csv'):
        if r['kind'] != 'smooth_local_deformation' or r['status'] != 'scored':
            continue
        groups[(r['state'], r['mode'], float(r['strength']))].append(r)
        if float(r['change']) < -ATOL and int(r['geometry_degraded_subjects']) > 0:
            individuals.append(r)
    out = []
    for (state, mode, strength), rr in sorted(groups.items()):
        values = [float(r['change']) for r in rr]
        mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
        degraded = all(int(r['geometry_degraded_subjects']) > 0 for r in rr)
        out.append(dict(state=state, mode=mode, strength=strength, n_seeds=len(rr),
            mean_change=mean, seed_sd=sd, all_seeds_geometry_degraded=degraded,
            reproducible_negative=bool(mean < -ATOL and -mean > 2*sd and degraded),
            negative_seeds=sum(v < -ATOL for v in values)))
    return out, individuals


def align(y, base):
    gpa = rigid_gpa(y)
    if not gpa.converged:
        raise ValueError('Candidate GPA did not converge')
    return match_group_frame(gpa.aligned, base.mean(axis=0)), gpa.iterations


def geometry(refs, ys, faces, labels):
    rows = []
    for subject, ref, y in zip(SUBJECTS, refs, ys):
        tol = float(np.linalg.norm(np.ptp(ref, axis=0))*.001)
        if np.array_equal(ref, y):
            g = dict(coverage=1., symmetric_mean=0., normal_reversal_rate=0., tolerance=tol, sample_count=0)
        else:
            g = evaluate_batch_geometry(ref, faces, y, faces, tolerance=tol, sample_count=SAMPLES, seed=0)
        if 'seed' in g:
            g['geometry_seed'] = g.pop('seed')
        rows.append(dict(**labels, subject=subject, **g))
    cov = float(np.mean([r['coverage'] for r in rows]))
    dist = float(np.mean([r['symmetric_mean']/r['tolerance'] for r in rows]))
    normal = float(np.mean([r['normal_reversal_rate'] for r in rows]))
    return dict(coverage_mean=cov, distance_over_tolerance_mean=dist, normal_reversal_rate_mean=normal,
        geometry_loss=1-cov+dist+normal, geometry_subjects=len(rows), geometry_samples=SAMPLES), rows


def graph_from_faces(faces, n):
    sets = [set() for _ in range(n)]
    for a, b, c in faces:
        sets[a].update((int(b), int(c))); sets[b].update((int(a), int(c))); sets[c].update((int(a), int(b)))
    return [np.array(sorted(v), dtype=np.int64) for v in sets]


def disjoint_blocks(graph, seed, width, max_fraction=.05):
    """Fix ordered disjoint edge swaps or triangles around a center once."""
    rng = np.random.default_rng(seed); n = len(graph)
    used = np.zeros(n, dtype=bool); result = []
    target = int(np.ceil(n*max_fraction/width))
    for u in rng.permutation(n):
        if used[u]:
            continue
        choices = graph[u][~used[graph[u]]]
        if len(choices) < width-1:
            continue
        block = [int(u)] + rng.choice(choices, width-1, replace=False).tolist()
        used[block] = True; result.append(block)
        if len(result) >= target:
            break
    if len(result) < target:
        raise ValueError('Insufficient disjoint local blocks')
    return np.asarray(result, dtype=np.int64)


def block_permutation(n, blocks, fraction):
    width = blocks.shape[1]
    count = 0 if fraction == 0 else max(1, int(round(n*fraction/width)))
    if count > len(blocks):
        raise ValueError('Insufficient blocks')
    selected = blocks[:count]
    p = np.arange(n)
    if len(selected):
        p[selected] = np.roll(selected, 1, axis=1)
    return p


def restoration_permutation(n, seed, restored):
    order = np.random.default_rng(seed).permutation(n)
    pairs = order[:2*int(round(n*.25/2))].reshape(-1, 2)
    active = int(round(len(pairs)*(1-restored)))
    p = np.arange(n)
    p[pairs[:active]] = pairs[:active, ::-1]
    return p


def certify_permutation(x, faces, p):
    if not np.array_equal(np.sort(p), np.arange(len(x))):
        raise ValueError('Not a bijection')
    inverse = np.empty_like(p); inverse[p] = np.arange(len(p))
    remapped_faces = inverse[faces]
    if not np.array_equal(p[remapped_faces], faces):
        raise ValueError('Inverse face remapping failed')
    # Equality of the index composition proves exact equality of all triangles.
    movement = np.linalg.norm(x[p]-x, axis=1)
    diagonal = float(np.linalg.norm(np.ptp(x, axis=0)))
    return dict(changed_fraction=float(np.mean(p != np.arange(len(p)))),
                correspondence_rmse_over_diagonal=float(np.sqrt(np.mean(movement**2))/diagonal),
                correspondence_mean_over_diagonal=float(movement.mean()/diagonal), surface_identity_certified=True)


def conditional_bounds(change_per_k, geom):
    if change_per_k >= 0:
        return 0., None
    if geom <= 0:
        return None, 0.
    return -change_per_k/geom, geom/(-change_per_k)


def run_job(job, output, meta):
    out = Path(output); cache = out/'cache'
    task, state, seed, family = job
    key = f'{task}_{state}_{seed}_{family}'
    checkpoint = out/'checkpoints'/f'{key}.json'
    if checkpoint.exists():
        return key, 'cached'
    started = time.perf_counter()
    x = np.load(cache/f'{state}_raw.npy'); f = np.load(cache/f'{state}_faces.npy')
    base = np.load(cache/f'all20_{state}_baseline.npy')
    info = meta['cohorts']['all20']; reference = info['reference_variance']; base_score = info['baseline_scores'][state]
    rows, geo_rows, audit = [], [], []
    if task == 'T2':
        bases = {name: np.load(cache/f'{name}_{state}_baseline.npy') for name in COHORTS}
        for strength in SMOOTH_LEVELS:
            y = np.array([smooth_local_deformation(v, f, strength, seed=sample_seed(seed, f'{s}/{state}/smooth_local_deformation')).vertices for s, v in zip(SUBJECTS, x)], dtype=float)
            for cohort, (lo, hi) in COHORTS.items():
                ci = meta['cohorts'][cohort]
                aligned, it = align(y[lo:hi], bases[cohort])
                score = fixed_score(aligned, ci['reference_variance']); change = score-ci['baseline_scores'][state]
                row = dict(cohort=cohort, state=state, seed=seed, strength=strength, N=hi-lo, K=hi-lo-1,
                    reference_variance=ci['reference_variance'], score=score, change=change, delta_per_k=change/(hi-lo-1), iterations=it,
                    level_set='original' if strength in ORIGINAL_LEVELS else 'added', geometry_status='not_measured_subcohort')
                if cohort == 'all20':
                    g, gg = geometry(bases[cohort], aligned, f, dict(task=task, state=state, seed=seed, strength=strength))
                    row.update(g); row['geometry_status']='measured_all_identities'; geo_rows.extend(gg)
                rows.append(row)
    elif task in ('T3', 'T4'):
        n = x.shape[1]
        blocks = None
        if task == 'T3' and family != 'global_index':
            graph = graph_from_faces(f, n)
            blocks = [disjoint_blocks(graph, sample_seed(seed, f'{s}/{state}/{family}'), 3 if family=='local_cycle' else 2) for s in SUBJECTS]
        levels = RESTORE_LEVELS if task=='T4' else GLOBAL_LEVELS if family=='global_index' else LOCAL_LEVELS
        for level in levels:
            ps = []
            for i, s in enumerate(SUBJECTS):
                ss = sample_seed(seed, f'{s}/{state}/{family}')
                p = restoration_permutation(n, ss, level) if task=='T4' else permutation(n, level, ss) if family=='global_index' else block_permutation(n, blocks[i], level)
                ps.append(p)
            checks = [certify_permutation(v, f, p) for v, p in zip(x, ps)]
            y = np.array([v[p] for v,p in zip(x,ps)])
            aligned, it = align(y, base)
            score = fixed_score(aligned, reference); change = score-base_score
            row = dict(cohort='all20', state=state, family=family, seed=seed, N=20, K=19,
                score=score, change=change, delta_per_k=change/19, iterations=it,
                changed_fraction=float(np.mean([c['changed_fraction'] for c in checks])),
                correspondence_rmse_over_diagonal=float(np.mean([c['correspondence_rmse_over_diagonal'] for c in checks])),
                surface_identity_certified=all(c['surface_identity_certified'] for c in checks),
                native_coverage=1., native_surface_distance=0., geometry_frame='native, before correspondence-dependent GPA')
            row['t' if task=='T4' else 'strength'] = level
            if task=='T4' and level==1:
                row['exact_index_endpoint'] = all(np.array_equal(p, np.arange(n)) for p in ps)
                row['endpoint_within_tolerance'] = bool(abs(change)<=ATOL+RTOL*abs(base_score))
                if not row['exact_index_endpoint'] or not row['endpoint_within_tolerance']:
                    raise ArithmeticError('Restoration endpoint failed')
            rows.append(row)
    elif task=='T5':
        mean = base.mean(axis=0); spec = covariance_spectrum(base)
        variants = [('copy_subject0', None)] + [('shrink', t) for t in SHRINK_LEVELS]
        for variant, t in variants:
            y = np.repeat(base[[0]], 20, axis=0) if variant=='copy_subject0' else base.copy() if t==1 else mean[None] + t*(base-mean[None])
            score = fixed_score(y, reference); change = score-base_score
            expected = 19*np.log(DELTA) if variant=='copy_subject0' else ecomp(t*t*spec, n_components=19, reference_variance=reference, delta=DELTA)
            error = abs(score-expected)
            if error > ATOL+RTOL*abs(expected):
                raise ArithmeticError('Analytical collapse spectrum check failed')
            g, gg = geometry(base, y, f, dict(task=task, state=state, seed=0, variant=variant, t=t))
            lower, upper = conditional_bounds(change/19, g['geometry_loss'])
            rows.append(dict(cohort='all20', state=state, variant=variant, t=t, N=20, K=19, score=score, change=change,
                delta_per_k=change/19, analytic_score=expected, analytic_error=error,
                lambda_geom_min_tested_candidate=lower, mu_comp_max_tested_candidate=upper, **g))
            geo_rows.extend(gg)
    else:
        raise ValueError(task)
    write_json(checkpoint, dict(key=key, rows=rows, geometry_rows=geo_rows, audit=audit, seconds=time.perf_counter()-started))
    return key, f'{len(rows)} rows, {len(geo_rows)} geometry, {time.perf_counter()-started:.1f}s'


def curve_stats(rows, levels, fields, xkey='strength'):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[f] for f in fields)].append(r)
    out = []
    for key, rr in sorted(groups.items()):
        previous = np.zeros(3)
        for level in levels:
            values = np.array([next(r['delta_per_k'] for r in rr if r[xkey]==level and r['seed']==s) for s in SEEDS])
            mean, sd = float(values.mean()), float(values.std(ddof=1))
            differences = values-previous
            step, step_sd = float(differences.mean()), float(differences.std(ddof=1))
            out.append(dict(zip(fields,key), **{xkey:level}, mean_delta_per_k=mean, seed_sd_per_k=sd,
                signed_step=step, paired_step_sd=step_sd, step_gt_sd=bool(step>sd), step_gt_2sd=bool(step>2*sd),
                paired_step_gt_3sd=bool(step>3*step_sd), detected=bool(mean>2*sd and np.all(values>ATOL)),
                all_seeds_positive=bool(np.all(values>ATOL)), all_seed_steps_positive=bool(np.all(differences>ATOL))))
            previous=values
    return out


def identity_stats(rows, levels, label):
    result=[]
    for left,right in [('base15','new5'),('base15','all20')]:
        for state in STATES:
            curves=[]
            for cohort in (left,right):
                curves.append(np.array([np.mean([r['delta_per_k'] for r in rows if r['cohort']==cohort and r['state']==state and r['strength']==v]) for v in levels]))
            a,b=curves; rho=float(spearmanr(a,b).statistic)
            sign=bool(np.array_equal(np.sign(a),np.sign(b))); mono=bool(np.all(np.diff(a)>=-ATOL) and np.all(np.diff(b)>=-ATOL))
            result.append(dict(level_set=label,left=left,right=right,state=state,rho=rho,same_sign=sign,both_nondecreasing=mono,stable=bool(rho>=.8 and sign and mono),
                independent_identity_groups=(right=='new5')))
    return result


def aggregate(out, source, meta, protocol):
    tasks=defaultdict(list); geometry_rows=[]
    for path in sorted((out/'checkpoints').glob('*.json')):
        d=json.loads(path.read_text(encoding='utf-8')); tasks[d['key'].split('_')[0]].extend(d['rows']); geometry_rows.extend(d['geometry_rows'])
    expected={'T2':441,'T3':462,'T4':126,'T5':56}
    for task,count in expected.items():
        if len(tasks[task])!=count:
            raise ValueError(f'Incomplete {task}: {len(tasks[task])}/{count}')
    if len(geometry_rows)!=4060:
        raise ValueError(f'Incomplete all-identity geometry: {len(geometry_rows)}/4060')
    for task, rows in tasks.items(): write_csv(out/f'{task.lower()}_results.csv',rows)
    write_csv(out/'geometry_subjects.csv',geometry_rows)
    t1rows, individual=t1(source)
    write_csv(out/'t1_seed_groups.csv', t1rows);write_csv(out/'t1_original_counterexamples.csv',individual)
    # Full replay of all 252 old smooth scores must agree before merging evidence.
    old={(r['cohort'],r['state'],int(r['seed']),float(r['strength'])):r for r in read_csv(source.parent/'stage3_stability_v1/cohort_scores.csv') if r['kind']=='smooth_local_deformation'}
    replay=[]
    for r in tasks['T2']:
        if r['strength'] in ORIGINAL_LEVELS:
            prior=old[(r['cohort'],r['state'],r['seed'],r['strength'])]
            err=abs(r['score']-float(prior['score']))
            replay.append(dict(cohort=r['cohort'],state=r['state'],seed=r['seed'],strength=r['strength'],abs_score_error=err,passed=bool(err<=ATOL+RTOL*abs(r['score']))))
    if not all(r['passed'] for r in replay):raise ArithmeticError('v3 replay mismatch')
    write_csv(out/'replay_checks.csv',replay)
    all_seed=[]; all_identity=[]; t2_summary={}
    for label,levels in [('original',ORIGINAL_LEVELS),('added',ADDED_LEVELS),('full',SMOOTH_LEVELS)]:
        sr=curve_stats(tasks['T2'],levels,['cohort','state'])
        for r in sr:r['level_set']=label
        ir=identity_stats(tasks['T2'],levels,label)
        all_seed.extend(sr);all_identity.extend(ir)
        t2_summary[label]=dict(seed_checks=len(sr),positive_step_gt_sd=sum(r['step_gt_sd'] for r in sr),
            positive_step_gt_2sd=sum(r['step_gt_2sd'] for r in sr),identity_checks=len(ir),identity_stable=sum(r['stable'] for r in ir),
            by_cohort={c:dict(checks=sum(r['cohort']==c for r in sr),positive_step_gt_sd=sum(r['step_gt_sd'] for r in sr if r['cohort']==c)) for c in COHORTS})
    write_csv(out/'t2_seed_stability.csv',all_seed);write_csv(out/'t2_identity_stability.csv',all_identity)
    t3_stats=[];t3_resolution=[];t3_summary={}
    for family in ['global_index','local_cycle','adjacent_swap']:
        rows=[r for r in tasks['T3'] if r['family']==family]
        levels=GLOBAL_LEVELS if family=='global_index' else LOCAL_LEVELS
        sr=curve_stats(rows,levels,['family','state']);t3_stats.extend(sr)
        for state in STATES:
            ss=[r for r in sr if r['state']==state]
            detected=[r['strength'] for r in ss if r['detected']]
            mono_seeds=sum(all(np.diff([next(r['delta_per_k'] for r in rows if r['state']==state and r['seed']==seed and r['strength']==v) for v in levels])>=-ATOL) for seed in SEEDS)
            t3_resolution.append(dict(family=family,state=state,minimum_tested_detectable=min(detected) if detected else None,
                detected_levels=len(detected),tested_levels=len(levels),monotone_seed_curves=mono_seeds,
                robust_positive_steps=sum(r['paired_step_gt_3sd'] for r in ss)))
        t3_summary[family]=dict(checks=len(sr),detected=sum(r['detected'] for r in sr),paired_positive_steps_gt_3sd=sum(r['paired_step_gt_3sd'] for r in sr),
            monotone_seed_curves=sum(r['monotone_seed_curves'] for r in t3_resolution if r['family']==family),seed_curves=21)
    write_csv(out/'t3_seed_stability.csv',t3_stats);write_csv(out/'t3_resolution_limits.csv',t3_resolution)
    restore=[]
    for state in STATES:
        for seed in SEEDS:
            rr=sorted([r for r in tasks['T4'] if r['state']==state and r['seed']==seed],key=lambda r:r['t'])
            restore.append(dict(state=state,seed=seed,score_nondecreasing_violations=int(np.sum(np.diff([r['score'] for r in rr])>ATOL)),
                monotone_decreasing=bool(np.all(np.diff([r['score'] for r in rr])<=ATOL)),
                correspondence_rmse_decreases=bool(np.all(np.diff([r['correspondence_rmse_over_diagonal'] for r in rr])<=ATOL)),
                endpoint_delta=rr[-1]['change'],exact_index_endpoint=rr[-1]['exact_index_endpoint'],endpoint_pass=rr[-1]['endpoint_within_tolerance']))
    write_csv(out/'t4_seed_checks.csv',restore)
    harmful=[r for r in tasks['T5'] if r['delta_per_k'] < -ATOL]
    bound=max(r['lambda_geom_min_tested_candidate'] for r in harmful)
    t5=dict(harmful_candidates=len(harmful),tested_candidates=len(tasks['T5']),max_lambda_geom_min=bound,
        equivalent_mu_comp_max=1/bound,analytic_error_max=max(r['analytic_error'] for r in tasks['T5']),
        zero_spectrum_score=19*np.log(DELTA),loss_convention=protocol['T5'],limitation='Finite sampled paths only; no global safe weight or training guarantee')
    summary=dict(version=VERSION,complete=True,rows={k:len(v) for k,v in tasks.items()},geometry_subject_rows=len(geometry_rows),
        cohorts=meta['cohorts'],T1=dict(original_individual_counterexamples=len(individual),seed_groups=len(t1rows),reproducible_negative_groups=sum(r['reproducible_negative'] for r in t1rows)),
        T2=t2_summary,T3=t3_summary,T4=dict(seed_curves=len(restore),monotone=sum(r['monotone_decreasing'] for r in restore),
        endpoint_pass=sum(r['endpoint_pass'] for r in restore),endpoint_abs_max=max(abs(r['endpoint_delta']) for r in restore)),T5=t5,
        replay=dict(checks=len(replay),passed=sum(r['passed'] for r in replay),max_score_error=max(r['abs_score_error'] for r in replay)),
        geometry_scope=protocol['geometry_scope'],limits=protocol['limitations'])
    write_json(out/'summary.json',summary)
    from plot_stage35 import plot_all
    plot_all(out)
    write_json(out/'manifest.json',dict(version=VERSION,complete=True,protocol_sha256=sha(out/'protocol_config_s35.json'),input_digest=meta['input_digest'],
        outputs={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'},
        checkpoints={p.name:sha(p) for p in sorted((out/'checkpoints').glob('*.json'))}))
    print(json.dumps(summary,ensure_ascii=True,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',default='data/ASTFace_Public')
    parser.add_argument('--source',default='code/reports/stage3_v3')
    parser.add_argument('--output',default='code/reports/stage3_s35_v2')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    source=Path(args.source)
    t1rows,individual=t1(source)
    write_csv(out/'t1_seed_groups.csv',t1rows);write_csv(out/'t1_original_counterexamples.csv',individual)
    meta=prepare(Path(args.data_dir),out)
    protocol=config(meta,source);config_path=out/'protocol_config_s35.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8'))!=protocol:
        raise ValueError('Code/config/source changed; use a new output directory')
    write_json(config_path,protocol)
    jobs=[('T2',s,seed,'smooth') for s in STATES for seed in SEEDS]
    jobs += [('T3',s,seed,f) for f in ['global_index','local_cycle','adjacent_swap'] for s in STATES for seed in SEEDS]
    jobs += [('T4',s,seed,'restore') for s in STATES for seed in SEEDS]
    jobs += [('T5',s,0,'collapse') for s in STATES]
    print(f'Frozen protocol; {len(jobs)} jobs, {args.workers} workers; checkpoints enabled',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(run_job,j,str(out),meta):j for j in jobs}
        for i,future in enumerate(as_completed(pending),1):
            key,status=future.result()
            print(f'[{i}/{len(jobs)}] {key}: {status}',flush=True)
    aggregate(out,source,meta,protocol)


if __name__=='__main__':
    main()
