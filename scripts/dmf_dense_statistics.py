"""Frozen, paired subject-level endpoints for the 93-person dense DMF sweep."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

from scripts.dmf_subject_consistency import transition_interval


def holm_two(pvalues):
    p = np.asarray(pvalues, float)
    if p.shape != (2,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected two finite p-values in [0,1]')
    order = np.argsort(p, kind='stable')
    adjusted = np.empty(2)
    adjusted[order[0]] = min(1., 2*p[order[0]])
    adjusted[order[1]] = max(adjusted[order[0]], p[order[1]])
    return adjusted.tolist()


def paired_hits(xi, comparator, bootstrap_seed=20261003):
    """Exact directional McNemar via Binomial(discordant, .5).

    The denominator is subjects, not seeds or G points. Bootstrap subjects jointly
    for a descriptive, unadjusted two-sided 95% CI on the success-rate difference.
    """
    x, y = np.asarray(xi, bool), np.asarray(comparator, bool)
    if x.ndim != 1 or x.shape != y.shape or not len(x):
        raise ValueError('Expected matching nonempty subject outcome vectors')
    wins, losses = int(np.count_nonzero(x & ~y)), int(np.count_nonzero(~x & y))
    difference = x.astype(int)-y.astype(int)
    rng = np.random.default_rng(bootstrap_seed)
    draws = difference[rng.integers(0, len(x), size=(10000, len(x)))].mean(1)
    return dict(subject_count=len(x), xi_hits=int(x.sum()), comparator_hits=int(y.sum()),
                xi_only=wins, comparator_only=losses, both_hits=int(np.count_nonzero(x & y)),
                both_misses=int(np.count_nonzero(~x & ~y)),
                rate_difference=float(difference.mean()),
                difference_ci95_percentile=np.quantile(draws, [.025, .975]).tolist(),
                p_one_sided_exact=float(binomtest(wins, wins+losses, .5, alternative='greater').pvalue) if wins+losses else 1.,
                alternative='Xi hit probability exceeds comparator hit probability',
                bootstrap_draws=10000, bootstrap_seed=bootstrap_seed)


def subject_landmarks(g, rates, metric_curves):
    """Same independent interval for all methods; boundary metric maxima miss."""
    g = np.asarray(g, float)
    rates = np.asarray(rates, float)
    if rates.shape != g.shape or not np.isfinite(rates).all():
        raise ValueError('Invalid independent order curve')
    t = transition_interval(g, rates)
    result = dict(transition=t, methods={})
    for name, curve in metric_curves.items():
        y = np.asarray(curve, float)
        if y.shape != g.shape or not np.isfinite(y).all():
            raise ValueError(f'Invalid metric curve: {name}')
        flat = float(np.ptp(y)) <= 1e-12 * max(1., float(np.abs(y).max()))
        k = int(np.argmin(y) if name == 'wms' else np.argmax(y))
        interior = bool(not flat and 0 < k < len(g)-1)
        point = None if flat else float(g[k])
        distance = None
        if t['located'] and point is not None:
            distance = float(max(t['interval'][0]-point, 0., point-t['interval'][1]))
        result['methods'][name] = dict(extremum_G=point, flat=flat, interior=interior,
            hit=(bool(interior and distance is not None and distance <= 1e-12) if t['located'] else None),
            distance_to_interval_G=distance)
    return result


def compare_cohort(rows):
    eligible = [r for r in rows if r['transition']['located']]
    excluded = [r['subject'] for r in rows if not r['transition']['located']]
    if not eligible:
        return dict(subject_count=len(rows), eligible_count=0, unlocated_subjects=excluded,
                    methods={}, comparisons={}, both_comparisons_significant=False)
    methods = {}
    for name in ('xi', 'phi_r', 'wms'):
        m = [r['methods'][name] for r in eligible]
        finite_distances = [v['distance_to_interval_G'] for v in m if v['distance_to_interval_G'] is not None]
        methods[name] = dict(hit_count=sum(v['hit'] for v in m), denominator=len(m),
                            hit_rate=sum(v['hit'] for v in m)/len(m),
                            flat_count=sum(v['flat'] for v in m),
                            boundary_extremum_count=sum(not v['interior'] and not v['flat'] for v in m),
                            mean_distance_to_interval_G=float(np.mean(finite_distances)) if finite_distances else None)
    x = [r['methods']['xi']['hit'] for r in eligible]
    comparisons = {name: paired_hits(x, [r['methods'][name]['hit'] for r in eligible]) for name in ('phi_r', 'wms')}
    adjusted = holm_two([v['p_one_sided_exact'] for v in comparisons.values()])
    for v, p in zip(comparisons.values(), adjusted):
        v['p_holm_two'] = p
        v['significant_superiority'] = bool(v['rate_difference'] > 0 and p < .05)
    return dict(subject_count=len(rows), eligible_count=len(eligible), unlocated_subjects=excluded,
                methods=methods, comparisons=comparisons,
                both_comparisons_significant=all(v['significant_superiority'] for v in comparisons.values()))
