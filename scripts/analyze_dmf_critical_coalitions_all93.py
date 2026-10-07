#!/usr/bin/env python3
"""Streaming exact-member registration and descriptive all-cohort candidate ranking."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_dmf_critical_coalitions import BASE, TOL
from scripts.run_dmf_subject_consistency import atomic_json, digest
from scripts.run_dmf_subject_curve_baselines import atomic_savez
from scripts.analyze_dmf_critical_coalitions import wilson
from scripts.run_dmf_paired_spt_pilot import paired_sources, noise_seed

MODES = ['natural_presence', 'above_background']
METRICS = ['critical_ge1', 'critical_ge3', 'critical_ge4', 'low_any', 'high_any',
    'low_stage_ge2', 'high_stage_ge2', 'outside_any', 'exclusive_ge4',
    'critical_point_count', 'low_point_count', 'high_point_count', 'outside_point_count',
    'total_point_count', 'seed_node_count', 'narrow_both', 'exclusive_ge1',
    'critical_ge1_no_endpoints', 'critical_ge3_no_endpoints',
    'critical_ge4_no_endpoints', 'presence_subjects']
M = {name: i for i, name in enumerate(METRICS)}
POPCOUNT = np.array([0, 1, 1, 2, 1, 2, 2, 3], dtype=np.uint8)


def members_mask(members):
    """Injective 100-bit identity, not a probabilistic hash or overlap criterion."""
    if len(set(members)) != len(members) or any(i < 0 or i >= 100 for i in members):
        raise ValueError('Invalid complete ROI membership')
    return sum(1 << i for i in members)


def mask_members(mask):
    return [i for i in range(100) if mask & (1 << i)]


def freeze_analysis(base):
    contract = dict(version='all93-compact-exact-registration-v1',
        scientific_contract_sha256=digest(base / 'contract.json'),
        extension_contract_sha256=digest(base / 'all93_extension_contract.json'),
        analysis_sha256=digest(Path(__file__)), modes=MODES, metrics=METRICS,
        registry='Two uint64 words encode exact zero-based ROI membership; aligned uint16 counters, no clipping or fuzzy matching',
        inference='Descriptive selection across all 93 SCs; no held-out validation or post-selection uncorrected p values',
        intervals='Wilson95% subject proportions, descriptive after candidate selection, not selection-adjusted',
        ranking='No weighted score; ordered count/leakage criteria from the extension contract',
        created_utc=None)
    path = base / 'all93_analysis_contract.json'
    if path.exists():
        old = json.loads(path.read_text())
        contract['created_utc'] = old['created_utc']
        if old != contract:
            raise ValueError('Frozen analysis changed; incompatible compact registry')
    else:
        contract['created_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(path, contract)
    return contract


def read_subject(base, c, subject, thresholds, sha, audit, source_hashes):
    """Only one individual's sparse seed masks are resident at a time."""
    events = {}
    for gi, g in enumerate(c['G']):
        for si, seed in enumerate(c['seeds']):
            path = base / 'trees' / f'{subject}_G{g:.2f}_seed{seed}.json'
            r = json.loads(path.read_text())
            if (r['contract_sha256'] != sha or r['subject'] != subject
                    or r['G'] != g or r['seed'] != seed or r['background']):
                raise ValueError(f'Condition identity/provenance mismatch: {path}')
            t = r['tree']; seen = set()
            density_path = base / 'density' / (path.stem + '.npz')
            with np.load(density_path) as density:
                if (str(density['contract_sha256']) != sha
                        or str(density['source_sha256']) != source_hashes[seed]
                        or int(density['noise_seed']) != noise_seed(seed)):
                    raise ValueError(f'Density/source/noise provenance mismatch: {density_path}')
                diagnostics = json.loads(str(density['diagnostics_json']))
            if diagnostics['outside_state_count']:
                raise ArithmeticError(f'State boundary violation: {density_path}: {diagnostics}')
            audit['density_count'] += 1
            audit['outside_state_count'] += diagnostics['outside_state_count']
            for node in t['nodes']:
                mask = members_mask(node['members'])
                left, right = members_mask(node['left']), members_mask(node['right'])
                if mask in seen or mask.bit_count() != node['order'] or left & right or left | right != mask:
                    raise ValueError(f'Invalid tree partition: {path}')
                seen.add(mask)
                value = node['syn_nats_raw']
                if not np.isfinite(value) or value < -TOL:
                    raise ArithmeticError(f'Invalid Syn: minimum={value}, threshold={-TOL}, affected=1: {path}')
                if 2 <= node['order'] <= 99:
                    a = events.setdefault(mask, np.zeros((2, len(c['G'])), dtype=np.uint8))
                    a[0, gi] |= 1 << si
                    threshold = thresholds[str(node['order'])]['threshold_nats']
                    if threshold is not None and value > threshold:
                        a[1, gi] |= 1 << si
            if len(seen) != 99 or members_mask(range(100)) not in seen:
                raise ValueError(f'Incomplete natural tree: {path}')
            if (not np.isfinite(t['closure_error_nats'])
                    or not np.isfinite(t['root_cross_roi_nats'])
                    or not np.isfinite(r['dense_xi_difference_nats'])):
                raise ArithmeticError(f'Nonfinite closure/root/dense comparison: {path}')
            recomputed_closure = sum(node['syn_nats_raw'] for node in t['nodes']) - t['root_cross_roi_nats']
            if abs(recomputed_closure) > TOL:
                raise ArithmeticError(f'Recomputed tree budget does not close: {path}')
            ca = t['candidate_audit']
            if not np.isfinite(ca['minimum_candidate_syn']) or ca['minimum_candidate_syn'] < -TOL:
                raise ArithmeticError(f'Invalid candidate Syn audit: {path}')
            for kind in ['selected_syn_audit', 'pair_audit', 'query_audit']:
                if t[kind]['violation_count'] or t[kind]['tolerance_nats'] != TOL:
                    raise ArithmeticError(f'Invalid nonnegativity audit {kind}: {path}')
            audit['tree_count'] += 1
            audit['internal_node_count'] += len(seen)
            audit['candidate_count'] += ca['candidate_count']
            audit['minimum_candidate_syn_nats'] = min(audit['minimum_candidate_syn_nats'], ca['minimum_candidate_syn'])
            audit['candidate_tolerance_negative_count'] += ca['tolerance_zero_count']
            audit['selected_tolerance_negative_count'] += t['selected_syn_audit']['tolerance_negative_count']
            audit['pair_tolerance_negative_count'] += t['pair_audit']['tolerance_negative_count']
            audit['query_tolerance_negative_count'] += t['query_audit']['tolerance_negative_count']
            audit['max_closure_error_nats'] = max(audit['max_closure_error_nats'], abs(t['closure_error_nats']))
            audit['max_dense_xi_difference_nats'] = max(audit['max_dense_xi_difference_nats'], r['dense_xi_difference_nats'])
            audit['imported_density_count'] += int(r['imported_density'])
            for name in ['elapsed_seconds', 'simulation_seconds', 'tree_seconds']:
                if not np.isfinite(r[name]) or r[name] < 0:
                    raise ValueError(f'Invalid timings: {path}')
                audit['summed_' + name] += r[name]
    return events


def subject_counts(seed_masks, c, subject):
    """One subject per count; seeds supply repeatability, never extra subjects."""
    seed_counts = POPCOUNT[seed_masks]
    repeat = seed_counts >= c['repeat_seed_count']
    win = c['windows'][subject]['grid_indices']
    outside = np.ones(len(c['G']), bool); outside[win] = False
    lo, hi = c['windows'][subject]['transition']['interval']
    narrow = np.flatnonzero(np.isclose(c['G'], lo) | np.isclose(c['G'], hi))
    if len(narrow) != 2 or len(win) != 6:
        raise ValueError('Expected original transition bracket and six-point window')
    critical = repeat[:, win].sum(1)
    low = repeat[:, c['low_grid_indices']].sum(1)
    high = repeat[:, c['high_grid_indices']].sum(1)
    out = repeat[:, outside].sum(1)
    rows = np.array([critical >= 1, critical >= 3, critical >= 4, low > 0, high > 0,
        low >= 2, high >= 2, out > 0, (critical >= 4) & (out == 0),
        critical, low, high, out, repeat.sum(1), seed_counts.sum(1),
        repeat[:, narrow].all(1), (critical >= 1) & (out == 0),
        (critical >= 1) & (low == 0) & (high == 0),
        (critical >= 3) & (low == 0) & (high == 0),
        (critical >= 4) & (low == 0) & (high == 0), seed_counts.any(1)], dtype=np.uint16).T
    return rows


def empty_audit():
    return dict(tree_count=0, density_count=0, outside_state_count=0,
        internal_node_count=0, candidate_count=0,
        minimum_candidate_syn_nats=float('inf'), candidate_tolerance_negative_count=0,
        selected_tolerance_negative_count=0, pair_tolerance_negative_count=0,
        query_tolerance_negative_count=0, max_closure_error_nats=0.,
        max_dense_xi_difference_nats=0., imported_density_count=0,
        summed_elapsed_seconds=0., summed_simulation_seconds=0., summed_tree_seconds=0.)


def register(base, c, subjects, background):
    sha = digest(base / 'contract.json'); totals = {}; audit = empty_audit()
    source_hashes = {seed: hashlib.sha256(np.concatenate(
        paired_sources(seed, c['sample_count'], 100), axis=1).tobytes()).hexdigest() for seed in c['seeds']}
    for subject in subjects:
        events = read_subject(base, c, subject, background['thresholds'], sha, audit, source_hashes)
        for mask, values in events.items():
            counts = subject_counts(values, c, subject)
            if mask in totals:
                totals[mask] += counts
            else:
                totals[mask] = counts
        print(json.dumps(dict(event='registered_subject', subject=subject,
            completed_trees=audit['tree_count'], registered_exact_sets=len(totals))), flush=True)
    if audit['max_closure_error_nats'] > TOL or audit['max_dense_xi_difference_nats'] > TOL:
        raise ArithmeticError('Tree closure/dense reproduction failed in persisted trees')
    keys = sorted(totals)
    words = np.array([[k & ((1 << 64) - 1), k >> 64] for k in keys], dtype=np.uint64)
    orders = np.array([k.bit_count() for k in keys], dtype=np.uint8)
    counts = np.stack([totals[k] for k in keys])
    audit['mean_condition_seconds'] = audit['summed_elapsed_seconds'] / audit['tree_count']
    audit['mean_tree_seconds'] = audit['summed_tree_seconds'] / audit['tree_count']
    return words, orders, counts, audit


def word_mask(words):
    return int(words[0]) | (int(words[1]) << 64)


def rank_indices(words, orders, counts, ranking, minimum_order=3, maximum_order=99, limit=5):
    eligible = np.flatnonzero((orders >= minimum_order) & (orders <= maximum_order))
    strong, raw = counts[:, 1].astype(np.int32), counts[:, 0].astype(np.int32)
    leaks = strong[:, M['low_any']] + strong[:, M['high_any']]
    if ranking == 'strict':
        fields = [-strong[:, M['critical_ge4']], leaks, -raw[:, M['critical_ge4']], orders]
    elif ranking == 'relaxed':
        fields = [-strong[:, M['critical_ge1']], leaks, -strong[:, M['critical_ge3']],
                  -strong[:, M['critical_ge4']], orders]
    elif ranking == 'exclusive':
        fields = [-strong[:, M['exclusive_ge4']], -strong[:, M['critical_ge4']], leaks, orders]
    else:
        raise ValueError(ranking)
    if not len(eligible):
        return []
    ordered = eligible[np.lexsort(tuple(f[eligible] for f in reversed(fields)))]
    cut = min(limit, len(ordered)) - 1
    # Only decode exact member tuples in the small group tied at the display boundary.
    boundary = tuple(f[ordered[cut]] for f in fields)
    end = cut + 1
    while end < len(ordered) and tuple(f[ordered[end]] for f in fields) == boundary:
        end += 1
    shortlist = ordered[:end].tolist()
    shortlist.sort(key=lambda i: tuple(f[i] for f in fields) + (tuple(mask_members(word_mask(words[i]))),))
    return shortlist[:limit]


def row_record(i, words, orders, counts, n, thresholds):
    members = mask_members(word_mask(words[i])); row = dict(members=members,
        members_one_based=[v + 1 for v in members], order=int(orders[i]),
        threshold_nats=thresholds[str(orders[i])]['threshold_nats'])
    for mode, vals in zip(MODES, counts[i]):
        d = {name: int(vals[j]) for j, name in enumerate(METRICS)}
        d['critical_ge4_ci95_wilson'] = wilson(d['critical_ge4'], n)
        d['critical_ge1_ci95_wilson'] = wilson(d['critical_ge1'], n)
        d['mean_critical_fraction'] = d['critical_point_count'] / (n * 6)
        d['mean_low_fraction'] = d['low_point_count'] / (n * 4)
        d['mean_high_fraction'] = d['high_point_count'] / (n * 4)
        d['mean_outside_fraction'] = d['outside_point_count'] / (n * 35)
        d['critical_minus_endpoint_fraction'] = d['mean_critical_fraction'] - (d['mean_low_fraction'] + d['mean_high_fraction']) / 2
        d['critical_minus_outside_fraction'] = d['mean_critical_fraction'] - d['mean_outside_fraction']
        row[mode] = d
    return row


def selected_values(base, c, subjects, selected, thresholds):
    index = {members_mask(r['members']): i for i, r in enumerate(selected)}
    values = np.full((len(selected), len(subjects), len(c['G']), len(c['seeds'])), np.nan)
    for pi, subject in enumerate(subjects):
        for gi, g in enumerate(c['G']):
            for si, seed in enumerate(c['seeds']):
                r = json.loads((base / 'trees' / f'{subject}_G{g:.2f}_seed{seed}.json').read_text())
                for node in r['tree']['nodes']:
                    mask = members_mask(node['members'])
                    if mask in index:
                        values[index[mask], pi, gi, si] = node['syn_nats_raw']
    raw = np.isfinite(values).sum(3) >= c['repeat_seed_count']
    strong = np.zeros_like(raw)
    for i, row in enumerate(selected):
        threshold = thresholds[str(row['order'])]['threshold_nats']
        if threshold is not None:
            strong[i] = (values[i] > threshold).sum(2) >= c['repeat_seed_count']
    return values, raw, strong


def analyze(base, development_only=False):
    started = time.perf_counter()
    c = json.loads((base / 'contract.json').read_text())
    extension = json.loads((base / 'all93_extension_contract.json').read_text())
    sha = digest(base / 'contract.json')
    if extension['scientific_contract_sha256'] != sha:
        raise ValueError('Extension scientific provenance mismatch')
    if not development_only:
        freeze_analysis(base)
    subjects = c['development_subject_ids'] if development_only else c['subject_ids']
    prefix = 'all93_devcheck' if development_only else 'all93'
    bg = json.loads((base / 'background_reference.json').read_text())
    if bg['contract_sha256'] != sha:
        raise ValueError('Background reference provenance mismatch')
    words, orders, counts, audit = register(base, c, subjects, bg)
    rankings = {}
    for name in ['strict', 'relaxed', 'exclusive']:
        for scope, maximum in [('preferred', 10), ('all_orders', 99)]:
            ids = rank_indices(words, orders, counts, name, maximum_order=maximum)
            rankings[name + '_' + scope] = [row_record(i, words, orders, counts, len(subjects), bg['thresholds']) for i in ids]
    ids = rank_indices(words, orders, counts, 'strict', minimum_order=2, maximum_order=2, limit=3)
    rankings['pair_reference'] = [row_record(i, words, orders, counts, len(subjects), bg['thresholds']) for i in ids]
    by_order = []
    for order in range(2, 100):
        ix = orders == order
        by_order.append(dict(order=order, registered_sets=int(ix.sum()),
            maxima={mode: {name: int(counts[ix, mi, M[name]].max(initial=0))
                for name in ['critical_ge1', 'critical_ge3', 'critical_ge4', 'exclusive_ge4']}
                for mi, mode in enumerate(MODES)}))
    higher = orders >= 3
    strong = counts[:, 1]
    gate = ((strong[:, M['critical_ge4']] >= int(np.ceil(.8 * len(subjects))))
        & (strong[:, M['low_any']] <= int(np.floor(.1 * len(subjects))))
        & (strong[:, M['high_any']] <= int(np.floor(.1 * len(subjects)))) & higher)
    selected = []; roles = {}
    for name in ['strict_preferred', 'relaxed_preferred', 'exclusive_preferred']:
        for row in rankings[name][:3]:
            key = tuple(row['members'])
            if key not in roles:
                selected.append(row); roles[key] = []
            roles[key].append(name)
    for row in rankings['strict_all_orders'][:1] + rankings['relaxed_all_orders'][:1]:
        key = tuple(row['members'])
        if key not in roles:
            selected.append(row); roles[key] = ['all_order_reference']
    for ci, row in enumerate(selected):
        row['display_id'] = f'C{ci+1}'; row['ranking_roles'] = roles[tuple(row['members'])]
    values, raw, repeated = selected_values(base, c, subjects, selected, bg['thresholds'])
    result = dict(status='complete', phase='all93-exploratory' if not development_only else 'development-regression-check',
        subject_ids=subjects, subject_count=len(subjects), G=c['G'], seeds=c['seeds'],
        condition_count=audit['tree_count'], scientific_contract_sha256=sha,
        extension_contract_sha256=digest(base / 'all93_extension_contract.json'),
        background_reference_sha256=digest(base / 'background_reference.json'),
        analysis_implementation_sha256=digest(Path(__file__)),
        registered_exact_sets=len(words), registered_high_order_sets=int(higher.sum()),
        registered_preferred_sets=int(((orders >= 3) & (orders <= 10)).sum()),
        maximum_strong_critical_ge4=int(strong[higher, M['critical_ge4']].max(initial=0)),
        maximum_strong_critical_ge3=int(strong[higher, M['critical_ge3']].max(initial=0)),
        maximum_strong_critical_ge1=int(strong[higher, M['critical_ge1']].max(initial=0)),
        maximum_strong_exclusive_ge4=int(strong[higher, M['exclusive_ge4']].max(initial=0)),
        comparison_80percent_10percent=dict(critical_min=int(np.ceil(.8 * len(subjects))),
            low_max=int(np.floor(.1 * len(subjects))), high_max=int(np.floor(.1 * len(subjects))),
            matching_count=int(gate.sum()), window_exclusive_count=int((gate & (strong[:, M['exclusive_ge4']] >= np.ceil(.8 * len(subjects)))).sum()),
            role='Historical frequency target shown for comparison; never a stopping rule'),
        rankings=rankings, selected=selected, by_order=by_order, audit=audit,
        native_syn_tolerance_nats=TOL, background_tree_count=bg['tree_count'],
        inference='Entire cohort used for exploratory selection. No untouched validation subjects; all displayed intervals are descriptive after selection',
        analysis_elapsed_seconds=time.perf_counter()-started)
    atomic_savez(base / (prefix + '_registry.npz'), member_mask_words=words, orders=orders,
        counts=counts, metrics_json=json.dumps(METRICS), modes_json=json.dumps(MODES),
        subject_ids=subjects, scientific_contract_sha256=sha,
        extension_contract_sha256=result['extension_contract_sha256'],
        analysis_implementation_sha256=result['analysis_implementation_sha256'])
    atomic_savez(base / (prefix + '_display.npz'), members_json=json.dumps([r['members'] for r in selected]),
        subject_ids=subjects, G=c['G'], seeds=c['seeds'], raw_syn_nats=values,
        repeated_presence=raw, repeated_above_background=repeated,
        scientific_contract_sha256=sha, extension_contract_sha256=result['extension_contract_sha256'])
    atomic_json(base / (prefix + '_summary.json'), result)
    print(json.dumps({k: result[k] for k in ['phase', 'subject_count','condition_count',
        'registered_high_order_sets','maximum_strong_critical_ge4', 'maximum_strong_critical_ge1']}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=BASE)
    p.add_argument('--development-only', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    if a.prepare_only:
        freeze_analysis(a.output_dir)
    else:
        analyze(a.output_dir, a.development_only)


if __name__ == '__main__':
    main()
