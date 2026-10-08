#!/usr/bin/env python3
"""Independent native-data checks for the completed SC/SPT comparison cache."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/dmf_schaefer100/sc_topology_comparison'
BASE = ROOT / 'results/dmf_schaefer100/critical_coalitions'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_scores(matrix, members):
    """Edge-list means/products and full modularity matrix, independently."""
    k = members.shape[1]
    u, v = np.triu_indices(k, 1)
    edge_values = matrix[members[:, u], members[:, v]]
    internal = edge_values.sum(axis=1)
    strength = matrix.sum(axis=1)
    total = matrix.sum() / 2
    modularity_matrix = matrix - np.outer(strength, strength) / (2 * total)
    q = modularity_matrix[members[:, :, None], members[:, None, :]].sum(axis=(1, 2)) / (2 * total)
    return np.column_stack((internal / len(u),
        edge_values.prod(axis=1) ** (1 / len(u)), q))


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    provenance = summary['contract']['provenance']
    for path, expected in provenance.items():
        actual_path = (ROOT / 'scripts/analyze_dmf_sc_topology_comparison.py'
                       if path == 'analysis_implementation_sha256' else ROOT / path)
        assert digest(actual_path) == expected, path
    subjects = summary['subject_ids']
    with np.load(ROOT / 'results/dmf_schaefer100/subject_curves_93_dense/inputs.npz') as src:
        ids = src['subject_ids'].tolist()
        matrices = src['connectivity'][[ids.index(s) for s in subjects]]
    contract = json.loads((BASE / 'contract.json').read_text())
    with np.load(BASE / 'all93_registry.npz') as registry:
        metric_index = {name: j for j, name in enumerate(json.loads(str(registry['metrics_json'])))}
        registry_counts = {}
        for k in (3, 4):
            rows = np.flatnonzero(registry['orders'] == k)
            words = registry['member_mask_words'][rows]
            identities = [tuple(i for i in range(100)
                                if (int(lo) | (int(hi) << 64)) & (1 << i)) for lo, hi in words]
            assert len(set(identities)) == len(identities)
            registry_counts[k] = dict(zip(identities, registry['counts'][rows]))
    checks = {'status': 'passed', 'source_hash_checks': len(provenance),
        'registered_score_values_checked': 0, 'independent_full_universe_rank_checks': [],
        'candidate_display_count_checks': 0, 'frozen_spt_top20_checks': 0,
        'registered_counts_matched_to_source': 0, 'zero_clique_candidates': {}}
    candidates = {3: [(9, 10, 11), (58, 59, 60)], 4: [(61, 62, 63, 67)]}
    with np.load(BASE / 'all93_display.npz') as display:
        display_members = json.loads(str(display['members_json']))
        for k in (3, 4):
            with np.load(OUT / f'order{k}.npz') as cache:
                members = cache['registered_members'].astype(int)
                stored = cache['registered_scores']
                p = cache['registered_percentiles']
                assert cache['subject_ids'].tolist() == subjects
                for si, matrix in enumerate(matrices):
                    np.testing.assert_allclose(direct_scores(matrix, members), stored[si],
                                               rtol=1e-12, atol=1e-15)
                checks['registered_score_values_checked'] += int(stored.size)
                ru = cache['registered_union_indices']
                np.testing.assert_allclose(p.mean(axis=0), cache['union_mean_percentiles'][ru],
                                           rtol=0, atol=1e-15)
                np.testing.assert_array_equal((p <= .01).sum(axis=0), cache['union_top1_counts'][ru])
                np.testing.assert_array_equal(cache['registered_counts'], cache['union_counts'][ru])
                np.testing.assert_array_equal(cache['registered_counts'],
                    [registry_counts[k][tuple(row)] for row in members])
                checks['registered_counts_matched_to_source'] += len(members)
                def strict_key(group):
                    counts = registry_counts[k][group]
                    strong, raw = counts[1], counts[0]
                    return (-int(strong[metric_index['critical_ge4']]),
                            int(strong[metric_index['low_any']]) + int(strong[metric_index['high_any']]),
                            -int(raw[metric_index['critical_ge4']]), group)
                spt_order = sorted(registry_counts[k], key=strict_key)
                np.testing.assert_array_equal(cache['union_members'][cache['spt_top20_union_indices']],
                                              spt_order[:20])
                checks['frozen_spt_top20_checks'] += 20
                for group in candidates[k]:
                    ci = next(i for i, row in enumerate(members) if tuple(row) == group)
                    label = next(row['display_id'] for row in summary['orders'][str(k)]['candidates']
                                 if row['members_one_based'] == [v + 1 for v in group])
                    zeros = np.flatnonzero(stored[:, ci, 1] == 0)
                    checks['zero_clique_candidates'][label] = [
                        {'subject': subjects[si], 'rank_percent': float(p[si, ci, 1] * 100),
                         'zero_edges_one_based': [[u + 1, v + 1] for u, v in itertools.combinations(group, 2)
                                                 if matrices[si, u, v] == 0]}
                        for si in zeros]
                    di = display_members.index(list(group))
                    for mode, name in enumerate(('repeated_presence', 'repeated_above_background')):
                        counts_in = []
                        counts_out = []
                        for si, subject in enumerate(subjects):
                            window = contract['windows'][subject]['grid_indices']
                            outside = sorted(set(range(len(contract['G']))) - set(window))
                            counts_in.append(int(display[name][di, si, window].sum()))
                            counts_out.append(int(display[name][di, si, outside].sum()))
                        expected = summary['orders'][str(k)]['candidates'][candidates[k].index(group)][
                            ('natural_presence', 'above_background')[mode]]
                        assert sum(v >= 4 for v in counts_in) == expected['critical_ge4']
                        assert sum(v >= 1 for v in counts_in) == expected['critical_ge1']
                        assert sum(counts_in) == expected['critical_point_count']
                        assert sum(counts_out) == expected['outside_point_count']
                        checks['candidate_display_count_checks'] += 4
                # Native first participant: independent enumeration and direct
                # counting ranks, rather than the analysis sorting algorithm.
                universe = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(100), k)),
                                       dtype=np.uint8, count=math.comb(100, k) * k).reshape(-1, k)
                full = direct_scores(matrices[0], universe.astype(int))
                for group in candidates[k]:
                    gi = next(i for i, row in enumerate(members) if tuple(row) == group)
                    ui = np.flatnonzero(np.all(universe == group, axis=1))[0]
                    observed = []
                    for mi in range(3):
                        target = full[ui, mi]
                        greater = int((full[:, mi] > target).sum())
                        equal = int((full[:, mi] == target).sum())
                        rank_fraction = (greater + equal / 2) / len(universe)
                        np.testing.assert_allclose(rank_fraction, p[0, gi, mi],
                                                   rtol=0, atol=2e-16)
                        observed.append(float(rank_fraction))
                    checks['independent_full_universe_rank_checks'].append(
                        {'subject': subjects[0], 'members_one_based': [v + 1 for v in group],
                         'universe_size': len(universe), 'rank_fractions': observed,
                         'rank_fraction_absolute_tolerance': 2e-16})
                del universe, full
    checks['verification_implementation_sha256'] = digest(Path(__file__))
    checks['verified_cache_sha256'] = {name: digest(OUT / name)
                                      for name in ('summary.json', 'order3.npz', 'order4.npz')}
    (OUT / 'verification.json').write_text(json.dumps(checks, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == '__main__':
    main()
