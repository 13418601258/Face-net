"""Independent checks for S3.6 v3 evidence and frozen source integrity."""
import csv
import json
from pathlib import Path
import numpy as np
from stage36_smooth import ROOT, sha, write_json

OUT = ROOT/'code/reports/stage3_s36_v3_constraints'


def main():
    checks = []
    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
    def read(name):
        with (OUT/name).open(encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))
    rows = read('constraint_results.csv'); designs = read('designs.csv')
    probes = read('scaled_energy_steps.csv'); landmarks = read('landmark_candidates.csv')
    summary = json.loads((OUT/'constraint_summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    frozen = json.loads((OUT/'frozen_protocol.json').read_text(encoding='utf-8'))
    sources = json.loads((OUT/'sources/sources.json').read_text(encoding='utf-8'))
    check('constraint_grid_complete', len(rows)==21*4*4*3 and len({(r['state'],r['seed'],r['layout'],r['sigma'],r['replicate']) for r in rows})==len(rows))
    check('designs_complete', len(designs)==84)
    check('energy_grid_complete', len(probes)==21*4*4)
    check('landmarks_all_140_samples_two_hypotheses', len(landmarks)==280 and len({(r['subject'],r['state']) for r in landmarks})==140)
    check('normal_equations_solved', max(float(r['normal_equation_inf']) for r in rows)<1e-10)
    check('gates_in_range', all(0<=float(r[k])<=1 for r in rows for k in ['gate_at_truth','gate_at_sparse_optimum']))
    check('noiseless_truth_gate_zero', all(float(r['gate_at_truth'])==0 for r in rows if float(r['sigma'])==0))
    check('full_rank_noiseless_recovery', all(float(r['nrmse'])<1e-10 for r in rows if float(r['sigma'])==0 and int(r['rank'])==6))
    check('two_anchor_nullity_two', all(int(r['nullity'])==2 for r in designs if r['layout']=='two_v2'))
    check('accepted_probes_valid_geometry', all(r['geometry_valid']=='True' for r in probes if r['accepted']=='True'))
    check('zero_weight_matches_control', all(abs(float(r['delta_vs_anchor']))<1e-14 for r in probes if float(r['requested_gradient_ratio'])==0))
    check('sparse_anchor_step_decreases_anchor', all(float(r['anchor_after'])<=float(r['initial_anchor'])+1e-12 for r in probes if float(r['requested_gradient_ratio'])==0))
    check('no_network_release', summary['network_eligible'] is False)
    check('sources_unchanged', all(sha(ROOT/p)==h for p,h in frozen['hashes'].items()))
    check('outputs_unchanged', all(sha(OUT/p)==h for p,h in manifest['files'].items()))
    check('external_source_hashes', all(sha(OUT/'sources'/r['file'])==r['sha256'] for r in sources['files']))
    subject_rows = read('constraint_subjects.csv')
    groups = {}
    keys = ['state','seed','layout','sigma','replicate']
    for row in subject_rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    check('twenty_subjects_per_sampleable_case', all(len(groups.get(tuple(r[k] for k in keys), []))==20
          for r in rows if r['sampled']=='True'))
    check('subject_aggregate_equals_group', all(np.isclose(float(r['nrmse']),
          np.mean([float(x['nrmse']) for x in groups[tuple(r[k] for k in keys)]]), rtol=1e-12, atol=1e-15)
          for r in rows if r['sampled']=='True'))
    check('no_subject_values_for_unsampleable', all(tuple(r[k] for k in keys) not in groups
          for r in rows if r['sampled']=='False'))
    review = json.loads((OUT/'review.json').read_text(encoding='utf-8'))
    review_manifest = json.loads((OUT/'review_manifest.json').read_text(encoding='utf-8'))
    check('crop_reindexing_confirmed', review['topology']['exact_release_topology_match'] is True)
    check('noise_free_duplicates_not_independent', review['distinct_conditions_without_noiseless_repeats']==840)
    check('review_hashes_match', review_manifest['code_sha256']==sha(ROOT/'code/review_stage36_constraints.py')
          and review_manifest['source_manifest_sha256']==sha(OUT/'manifest.json')
          and all(sha(OUT/p)==h for p,h in review_manifest['files'].items()))
    for group in summary['by_layout_noise']:
        rr = [r for r in rows if r['layout']==group['layout'] and float(r['sigma'])==group['sigma']]
        check('aggregate_'+group['layout']+'_'+str(group['sigma']), len(rr)==group['count'] and sum(r['geometry_valid']=='True' for r in rr)==group['geometry_valid'])
    result = dict(passed=all(c['passed'] for c in checks), checks=checks, count=len(checks))
    write_json(OUT/'verification.json', result)
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
