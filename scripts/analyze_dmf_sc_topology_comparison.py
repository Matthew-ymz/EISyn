#!/usr/bin/env python3
"""Exact per-person SC triplet/quartet ranks against frozen natural SPT counts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import networkx as nx
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_dmf_critical_coalitions_all93 import (
    M, METRICS, mask_members, rank_indices, word_mask,
)

BASE = ROOT / 'results/dmf_schaefer100/critical_coalitions'
INPUT = ROOT / 'results/dmf_schaefer100/subject_curves_93_dense/inputs.npz'
OUTPUT = ROOT / 'results/dmf_schaefer100/sc_topology_comparison'
METRIC_NAMES = ['internal_mean', 'clique_intensity', 'modularity_contribution']
CANDIDATES = {'C1': (9, 10, 11), 'C2': (58, 59, 60), 'C3': (61, 62, 63, 67)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def combinations(n, k):
    size = math.comb(n, k)
    a = np.fromiter((v for g in itertools.combinations(range(n), k) for v in g),
                    dtype=np.uint8, count=size * k).reshape(size, k)
    if a.shape != (size, k) or not np.all(np.diff(a.astype(np.int16), axis=1) > 0):
        raise ValueError('Invalid lexicographic combination enumeration')
    return a


def combination_index(members, n=100):
    k = len(members)
    if list(members) != sorted(set(members)) or not all(0 <= v < n for v in members):
        raise ValueError('Invalid exact coalition')
    return sum(math.comb(n - v - 1, k - j - 1)
               for j, x in enumerate(members)
               for v in range(0 if j == 0 else members[j-1]+1, x))


def score_arrays(c, groups):
    """Three fixed metrics, native positive weights; zero edges remain zero."""
    size, k = groups.shape
    edges = k * (k-1) // 2
    internal = np.zeros(size, dtype=np.float64)
    product = np.ones(size, dtype=np.float64)
    complete = np.ones(size, dtype=bool)
    for i, j in itertools.combinations(range(k), 2):
        w = c[groups[:, i], groups[:, j]]
        internal += w
        product *= w
        complete &= w > 0
    if np.any(complete & (product == 0)):
        raise ArithmeticError('Nonzero clique product underflow; use a log-domain evaluation')
    np.power(product, 1 / edges, out=product)
    strength = c.sum(axis=1)
    total = strength.sum() / 2
    if total <= 0:
        raise ValueError('Modularity is undefined for an all-zero SC')
    contribution = np.zeros(size, dtype=np.float64)
    for j in range(k):
        contribution += strength[groups[:, j]]
    contribution /= 2 * total
    np.square(contribution, out=contribution)
    contribution *= -1
    contribution += internal / total
    internal /= edges
    if not all(np.isfinite(x).all() for x in (internal, product, contribution)):
        raise ArithmeticError('Nonfinite SC metric')
    if not np.array_equal(product == 0, ~complete):
        raise ArithmeticError('Zero-edge clique semantics violated')
    return internal, product, contribution


def rank_percentile(scores):
    """(descending average rank - .5)/N; exact floating-point score ties."""
    n = len(scores)
    idx = np.argsort(scores, kind='stable')
    ordered = scores[idx]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], n]
    ranked = np.empty(n, dtype=np.float64)
    ranked[idx] = np.repeat(1 - (starts + ends) / (2 * n), ends - starts)
    if abs(float(ranked.mean()) - .5) > 1e-12:
        raise ArithmeticError('Average tie ranks do not have mean .5')
    return ranked


def verify_metrics():
    rng = np.random.default_rng(51)
    a = rng.uniform(0, 2, size=(7, 7))
    c = (a + a.T) / 2
    c[c < .7] = 0
    np.fill_diagonal(c, 0)
    g = nx.from_numpy_array(c)
    strengths = c.sum(1)
    total = c.sum() / 2
    checks = 0
    for k in (3, 4):
        groups = combinations(7, k)
        scores = score_arrays(c, groups)
        for z, members in enumerate(groups):
            vertices = members.astype(int).tolist()
            assert combination_index(vertices, 7) == z
            edges = [c[u, v] for u, v in itertools.combinations(vertices, 2)]
            assert np.isclose(scores[0][z], np.mean(edges), rtol=1e-13)
            assert np.isclose(scores[1][z], np.prod(edges)**(1/len(edges)), rtol=1e-13)
            direct = sum(c[u, v] - strengths[u]*strengths[v]/(2*total)
                         for u in vertices for v in vertices) / (2*total)
            assert np.isclose(scores[2][z], direct, atol=1e-14)
            rest = [v for v in range(7) if v not in vertices]
            rest_q = sum(c[u, v] - strengths[u]*strengths[v]/(2*total)
                         for u in rest for v in rest) / (2*total)
            assert np.isclose(scores[2][z] + rest_q,
                              nx.community.modularity(g, [set(vertices), set(rest)], weight='weight'),
                              atol=1e-14)
            checks += 4
        for values in scores:
            expected = (stats.rankdata(-values, method='average') - .5) / len(values)
            np.testing.assert_allclose(rank_percentile(values), expected, atol=2e-16)
            checks += 1
        scaled = score_arrays(c * 3, groups)
        np.testing.assert_allclose(scaled[0], scores[0] * 3)
        np.testing.assert_allclose(scaled[1], scores[1] * 3)
        np.testing.assert_allclose(scaled[2], scores[2], atol=1e-14)
    for x in [np.zeros(8), np.array([0, 0, 1, 1, 4, 5, 5, 5]), np.arange(9.)]:
        np.testing.assert_allclose(rank_percentile(x), (stats.rankdata(-x)-.5)/len(x))
        checks += 1
    return {'status': 'passed', 'direct_checks': checks,
            'checks': 'Exact enumeration; direct means/clique products; full modularity matrix and NetworkX partition; average ties including all-equal; uniform scaling'}


def load_data():
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in [INPUT, BASE/'all93_registry.npz',
        BASE/'all93_summary.json', BASE/'all93_display.npz', BASE/'contract.json', BASE/'all93_extension_contract.json']}
    summary = json.loads((BASE/'all93_summary.json').read_text())
    contract = json.loads((BASE/'contract.json').read_text())
    if summary['status'] != 'complete' or summary['subject_count'] != 93:
        raise ValueError('Incomplete original SPT cohort')
    if summary['scientific_contract_sha256'] != sha(BASE/'contract.json'):
        raise ValueError('SPT scientific contract changed')
    if summary['extension_contract_sha256'] != sha(BASE/'all93_extension_contract.json'):
        raise ValueError('SPT extension contract changed')
    with np.load(BASE/'all93_registry.npz') as r:
        words, orders, counts = [r[k].copy() for k in ['member_mask_words', 'orders', 'counts']]
        subjects = r['subject_ids'].tolist()
        if json.loads(str(r['metrics_json'])) != METRICS or subjects != summary['subject_ids']:
            raise ValueError('SPT registry metric/subject mismatch')
        if str(r['scientific_contract_sha256']) != summary['scientific_contract_sha256']:
            raise ValueError('SPT registry scientific provenance mismatch')
        if str(r['extension_contract_sha256']) != summary['extension_contract_sha256']:
            raise ValueError('SPT registry extension provenance mismatch')
    with np.load(INPUT) as data:
        source_ids = data['subject_ids'].tolist()
        if len(source_ids) != 94 or source_ids[-1] != 'group_mean_93':
            raise ValueError('Individual SC input layout changed')
        if len(set(source_ids)) != 94 or set(source_ids[:-1]) != set(subjects):
            raise ValueError('Unpaired SC subjects')
        c = data['connectivity'][[source_ids.index(s) for s in subjects]].copy()
    if (c.shape != (93, 100, 100) or not np.isfinite(c).all() or (c < 0).any()
            or not np.array_equal(c, c.transpose(0, 2, 1))
            or np.diagonal(c, axis1=1, axis2=2).any()):
        raise ValueError('Invalid native individual SC')
    with np.load(BASE/'all93_display.npz') as display:
        raw = display['raw_syn_nats']
        if display['subject_ids'].tolist() != subjects:
            raise ValueError('Selected display is not paired')
        finite = raw[np.isfinite(raw)]
        tol = contract['syn_tolerance_nats']
        violations = int((finite < -tol).sum())
        if np.isinf(raw).any() or violations:
            raise ArithmeticError(f'Invalid Syn minimum={finite.min()}, threshold={-tol}, affected={violations}')
        syn_audit = {'tolerance_nats': tol, 'finite_selected_values': len(finite),
            'minimum_selected_syn_nats': float(finite.min()),
            'tolerance_negative_count': int(((finite < 0) & (finite >= -tol)).sum()),
            'violation_count': violations, 'clipping': False}
    return c, subjects, words, orders, counts, summary, fingerprints, syn_audit


def run_order(k, c, subjects, words, orders, counts, fingerprints, output):
    start = time.perf_counter()
    groups = combinations(100, k)
    n = len(groups)
    registered = np.flatnonzero(orders == k)
    reg_members = [mask_members(word_mask(words[i])) for i in registered]
    reg_indices = np.array([combination_index(a) for a in reg_members], dtype=np.int64)
    np.testing.assert_array_equal(groups[reg_indices], reg_members)
    reg_mapping = {int(x): j for j, x in enumerate(reg_indices)}
    sum_percent = np.zeros((n, 3), dtype=np.float64)
    top1_counts = np.zeros((n, 3), dtype=np.uint8)
    reg_percent = np.empty((93, len(registered), 3), dtype=np.float64)
    reg_scores = np.empty_like(reg_percent)
    individual_audit = []
    for si, (subject, matrix) in enumerate(zip(subjects, c)):
        values = score_arrays(matrix, groups)
        audit = {'subject': subject, 'order': k, 'enumerated_sets': n, 'metrics': {}}
        for mi, scores in enumerate(values):
            p = rank_percentile(scores)
            sum_percent[:, mi] += p
            top1_counts[:, mi] += (p <= .01).astype(np.uint8)
            reg_percent[si, :, mi] = p[reg_indices]
            reg_scores[si, :, mi] = scores[reg_indices]
            audit['metrics'][METRIC_NAMES[mi]] = {'minimum': float(scores.min()),
                'maximum': float(scores.max()), 'zero_count': int((scores == 0).sum()),
                'rank_mean': float(p.mean()), 'top1_count': int((p <= .01).sum())}
            del p
        individual_audit.append(audit)
        del values, scores
        if si == 0 or (si + 1) % 10 == 0 or si == 92:
            print(f'order={k} subject={si+1}/93 elapsed={time.perf_counter()-start:.1f}s', flush=True)
    mean_percent = sum_percent / 93
    del sum_percent
    spt_global = rank_indices(words, orders, counts, 'strict', k, k, len(registered))
    spt_full_indices = np.array([combination_index(mask_members(word_mask(words[i])))
                                 for i in spt_global], dtype=np.int64)
    spt_rank = {int(i): j+1 for j, i in enumerate(spt_full_indices)}
    leading = []
    sc_boundary = []
    for mi in range(3):
        v = mean_percent[:, mi]
        bound = np.partition(v, 19)[19]
        tied_shortlist = np.flatnonzero(v <= bound)
        leading.append(tied_shortlist[np.lexsort((tied_shortlist, v[tied_shortlist]))][:20])
        sc_boundary.append({'mean_rank_fraction': float(bound), 'tie_count': int((v == bound).sum())})
    union_indices = np.unique(np.r_[reg_indices, spt_full_indices[:20], *leading])
    union_members = groups[union_indices].copy()
    union_mean = mean_percent[union_indices].copy()
    union_top1 = top1_counts[union_indices].copy()
    cohort_midrank = np.empty_like(union_mean)
    cohort_ordinal = np.empty(union_mean.shape, dtype=np.int64)
    for mi in range(3):
        sorted_indices = np.argsort(mean_percent[:, mi], kind='stable')
        sorted_values = mean_percent[sorted_indices, mi]
        left = np.searchsorted(sorted_values, union_mean[:, mi], side='left')
        right = np.searchsorted(sorted_values, union_mean[:, mi], side='right')
        cohort_midrank[:, mi] = (left + right + 1) / 2
        inverse = np.empty(n, dtype=np.int64)
        inverse[sorted_indices] = np.arange(1, n+1)
        cohort_ordinal[:, mi] = inverse[union_indices]
        del sorted_indices, sorted_values, inverse
    union_map = {int(i): j for j, i in enumerate(union_indices)}
    reg_union = np.array([union_map[int(i)] for i in reg_indices])
    union_counts = np.zeros((len(union_indices), 2, len(METRICS)), dtype=np.uint16)
    has_spt = np.zeros(len(union_indices), dtype=bool)
    for index in union_indices:
        if int(index) in reg_mapping:
            ui = union_map[int(index)]
            has_spt[ui] = True
            union_counts[ui] = counts[registered[reg_mapping[int(index)]]]

    def row(index):
        ui = union_map[int(index)]
        members = union_members[ui].astype(int).tolist()
        label = next((name for name, a in CANDIDATES.items() if tuple(members) == a), None)
        r = {'members_one_based': [v+1 for v in members], 'order': k, 'display_id': label,
             'ever_natural_node': bool(has_spt[ui]), 'spt_same_order_display_rank': spt_rank.get(int(index)),
             'sc': {}}
        for mi, name in enumerate(METRIC_NAMES):
            r['sc'][name] = {'mean_individual_rank_percent': float(union_mean[ui, mi]*100),
                'cohort_midrank': float(cohort_midrank[ui, mi]),
                'cohort_display_rank': int(cohort_ordinal[ui, mi]),
                'individual_top1_count': int(union_top1[ui, mi])}
            if int(index) in reg_mapping:
                pi = reg_mapping[int(index)]
                q = np.quantile(reg_percent[:, pi, mi] * 100, [.25, .5, .75])
                r['sc'][name]['individual_rank_percent_q25_median_q75'] = q.tolist()
        for mode, vals in zip(['natural_presence', 'above_background'], union_counts[ui]):
            r[mode] = {name: int(vals[j]) for j, name in enumerate(METRICS)}
            r[mode]['mean_critical_fraction'] = int(vals[M['critical_point_count']]) / (93*6)
            r[mode]['mean_outside_fraction'] = int(vals[M['outside_point_count']]) / (93*35)
        return r

    correlations, overlap = {}, {}
    y = counts[registered, 1, M['critical_ge4']]
    spt_boundary_count = counts[spt_global[19], 1, M['critical_ge4']]
    boundary_tuple = (int(spt_boundary_count),
        int(counts[spt_global[19], 1, M['low_any']] + counts[spt_global[19], 1, M['high_any']]),
        int(counts[spt_global[19], 0, M['critical_ge4']]))
    equal_primary = int((y == spt_boundary_count).sum())
    equal_all = sum((int(counts[i, 1, M['critical_ge4']]),
        int(counts[i, 1, M['low_any']] + counts[i, 1, M['high_any']]),
        int(counts[i, 0, M['critical_ge4']])) == boundary_tuple for i in registered)
    for mi, name in enumerate(METRIC_NAMES):
        # Higher structural quality = negative ascending rank percentile.
        rho = stats.spearmanr(-mean_percent[reg_indices, mi], y).statistic
        correlations[name] = {'spearman_rho': float(rho), 'registered_sets': len(registered),
            'scope': 'Conditional on ever selected as a natural node, not the full coalition universe; no p value'}
        common = set(map(int, leading[mi])) & set(map(int, spt_full_indices[:20]))
        overlap[name] = {'top20_intersection_count': len(common), 'jaccard': len(common)/(40-len(common)),
            'common_members_one_based': [[int(v)+1 for v in groups[i]] for i in sorted(common)],
            'sc_boundary': sc_boundary[mi], 'spt_boundary_primary_ge4_count': int(spt_boundary_count),
            'spt_boundary_primary_tie_count': equal_primary, 'spt_boundary_all_criteria_tie_count': equal_all}
    result = {'order': k, 'possible_sets': n, 'registered_spt_sets': len(registered),
        'evaluated_individual_sets': 93*n, 'elapsed_seconds': time.perf_counter()-start,
        'sc_top20': {name: [row(int(i)) for i in leading[mi]] for mi, name in enumerate(METRIC_NAMES)},
        'spt_top20': [row(int(i)) for i in spt_full_indices[:20]],
        'candidates': [row(combination_index(a)) for a in CANDIDATES.values() if len(a) == k],
        'correlations': correlations, 'overlap': overlap,
        'individual_metric_audit': individual_audit}
    np.savez_compressed(output/f'order{k}.npz',
        subject_ids=np.array(subjects), metric_names=np.array(METRIC_NAMES),
        registered_members=np.asarray(reg_members, dtype=np.uint8),
        registered_percentiles=reg_percent, registered_scores=reg_scores,
        registered_counts=counts[registered], registered_union_indices=reg_union,
        union_members=union_members, union_mean_percentiles=union_mean,
        union_top1_counts=union_top1, union_cohort_midrank=cohort_midrank,
        union_cohort_ordinal_rank=cohort_ordinal, union_counts=union_counts,
        union_ever_natural_node=has_spt,
        sc_top20_union_indices=np.array([[union_map[int(i)] for i in lead] for lead in leading]),
        spt_top20_union_indices=np.array([union_map[int(i)] for i in spt_full_indices[:20]]),
        provenance_json=json.dumps(fingerprints, sort_keys=True))
    save_json(output/f'order{k}_summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    verified = verify_metrics()
    print(json.dumps(verified), flush=True)
    if args.verify_only:
        return
    c, subjects, words, orders, counts, original, fingerprints, syn_audit = load_data()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    fingerprints['analysis_implementation_sha256'] = sha(Path(__file__))
    started = time.perf_counter()
    run_contract = {'version': 'individual-sc-all-triplets-quartets-v1',
        'metric_names': METRIC_NAMES, 'cohort_ranking': 'Mean of 93 per-subject full-universe midrank percentiles; ascending',
        'tie_rule': 'Exact computed float equality, average ranks; lexicographic members only for display',
        'top_fraction': .01, 'spt_ranking': 'Frozen strict same-order ge4, endpoints, raw ge4, exact membership',
        'preprocessing': 'Native nonnegative individual SC, no thresholding, no normalization, no group mean',
        'null_test': False, 'provenance': fingerprints,
        'created_utc': datetime.now(timezone.utc).isoformat()}
    previous = output/'contract.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        if {k:v for k,v in old.items() if k != 'created_utc'} != {k:v for k,v in run_contract.items() if k != 'created_utc'}:
            raise ValueError('Existing comparison cache has different provenance; choose another output')
        run_contract = old
    else:
        save_json(previous, run_contract)
    results = {}
    for k in (3, 4):
        path = output/f'order{k}_summary.json'
        if path.exists() and (output/f'order{k}.npz').exists():
            with np.load(output/f'order{k}.npz') as a:
                if json.loads(str(a['provenance_json'])) != fingerprints:
                    raise ValueError('Order cache provenance mismatch')
            results[str(k)] = json.loads(path.read_text())
            print(f'order={k} reused completed exact rank cache', flush=True)
        else:
            results[str(k)] = run_order(k, c, subjects, words, orders, counts, fingerprints, output)
    for relative, value in fingerprints.items():
        if relative != 'analysis_implementation_sha256' and sha(ROOT/relative) != value:
            raise ValueError(f'Original inputs changed during comparison: {relative}')
    summary = {'status': 'complete', 'subject_count': 93, 'subject_ids': subjects,
        'evaluated_individual_sets': sum(r['evaluated_individual_sets'] for r in results.values()),
        'verification': verified, 'selected_syn_audit': syn_audit, 'contract': run_contract,
        'orders': results, 'elapsed_this_invocation_seconds': time.perf_counter()-started,
        'manuscript': {'parent_key': 'P6UJCVG8', 'main_attachment': 'DXGC7JEA',
            'supplement_attachment': 'MWIWKSVG', 'checked_date': '2026-10-08',
            'version_date': None, 'ambiguity': 'No explicit manuscript date or revision number',
            'locations': 'Main Methods equations 10–12 pp16–17; SI S5 pp8–9; S12.2.1 p23'},
        'inference': 'Exploratory descriptive paired ranks; no random-network significance, heldout evaluation, or pure-order causal mechanism claim'}
    save_json(output/'summary.json', summary)
    print(json.dumps({'status':'complete', 'evaluated_individual_sets':summary['evaluated_individual_sets'],
                      'elapsed_seconds':summary['elapsed_this_invocation_seconds']}), flush=True)


if __name__ == '__main__':
    main()
