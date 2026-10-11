#!/usr/bin/env python3
"""Read cached natural WMS joint models with the existing pairwise PhiR query.

No DMF simulation, new fit, window search, or main-figure replacement. Preserve
the natural source correlations and the cached model's inherited regularization.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_dmf_state_aligned_phi_r import covariance_pair_phi_r
from scripts.dmf_dense_statistics import subject_landmarks
from scripts.run_dmf_natural_state_phi_r import validate_cache, legacy_index
from scripts.run_dmf_subject_consistency import atomic_json, digest
from scripts.run_dmf_subject_curve_baselines import atomic_savez

NATURAL = ROOT / 'results/dmf_schaefer100/natural_state_phi_r'
INTERVENTION = ROOT / 'results/dmf_schaefer100/state_aligned_phi_r'
DENSE = ROOT / 'results/dmf_schaefer100/subject_curves_93_dense'
OUTPUT = ROOT / 'results/dmf_schaefer100/natural_pairwise_phi_r'
FIGURE = ROOT / 'docs/reports/assets/dmf_subject_dense/natural_pairwise_phi_r_comparison.png'
TOL = 1e-8  # nats; retain tolerance negatives, fail significant violations.


def prepare():
    c = json.loads((NATURAL / 'contract.json').read_text())
    inter = json.loads((INTERVENTION / 'contract.json').read_text())
    if c['subject_ids'][:-1] != inter['subject_ids'] or c['G'] != inter['G'] or c['seeds'] != inter['seeds']:
        raise ValueError('Natural and intervention subject/grid/seed mismatch')
    for field, filename in [('dense_contract_sha256', 'contract.json'),
                            ('dense_curves_sha256', 'analysis_curves.npz'),
                            ('dense_windows_sha256', 'window_summary.json')]:
        if c[field] != digest(DENSE / filename):
            raise ValueError(f'Frozen natural provenance changed: {filename}')
    contract = dict(version='natural-full-EI-pairwise-phiR-v1',
        subject_ids=c['subject_ids'], G=c['G'], seeds=c['seeds'],
        sample_count=2048, pair_count=19900, units='nats',
        source='Original WMS correlated natural E/I source draws, hashes checked per condition',
        target='Original WMS independently propagated 300-ms futures; not trajectory-lag pairs',
        observable='Same 200 scalar E/I gating coordinates as intervention pairwise PhiR',
        aggregation='Mean of all 19,900 unordered scalar pairs per condition; then three-seed mean',
        definition=inter['phi_r_definition'],
        estimator='Existing Gaussian-MMI pair formula on cached coherent natural affine Gaussian joint covariance',
        regularization='Cached natural model: source ridge 1e-6, residual ridge 1e-6, target additional ridge 1e-6. Intervention model has one source and one residual ridge, no additional target ridge.',
        contrast='Same pair formula, coordinates, horizon and frozen windows; use respective existing natural/interventional fitted models. Not a pure prior substitution within one fixed transition fit.',
        numerical_rule='Cholesky queries on every 4x4 marginal; no added floor, covariance repair, clipping or projection; fail invalid marginals',
        nonnegative_tolerance_nats=TOL,
        window='Original independent firing-rate transition interval plus 0.2 G each side; frozen width 0.5 G',
        inference='Descriptive; no new hypothesis tests, smoothing or peak alignment',
        natural_contract_sha256=digest(NATURAL / 'contract.json'),
        intervention_contract_sha256=digest(INTERVENTION / 'contract.json'),
        intervention_curves_sha256=digest(INTERVENTION / 'curves.npz'),
        dense_curves_sha256=digest(DENSE / 'analysis_curves.npz'),
        dense_windows_sha256=digest(DENSE / 'window_summary.json'),
        query_script_sha256=digest(ROOT / 'scripts/compare_dmf_state_aligned_phi_r.py'),
        runner_sha256=digest(Path(__file__)),
        manuscript=dict(parent='P6UJCVG8', body='DXGC7JEA', supplement='MWIWKSVG',
            checked_date='2026-10-11', explicit_draft_version_or_date=None,
            ambiguity='Current sole body/supplement have no explicit revision/date; metadata edits do not establish a manuscript version',
            locations='Brain/Fig2 pp6-7; Methods Eqs5-8 pp15-16; Supplement S12.2.1/EqsS82-S87 p23',
            discrepancies=['Manuscript observational PhiR is BOLD-like; this requested comparison uses E/I gating states and 300ms',
                'Native WMS uses E/I, despite manuscript BOLD-like description',
                '93 participant SCs/3 seeds/G0..4 differ from manuscript mean SC/8 seeds/G0..3',
                'Gaussian affine fits and inherited native regularization are approximations; same samples do not imply identical WMS marginal queries']))
    (OUTPUT / 'subjects').mkdir(parents=True, exist_ok=True)
    path = OUTPUT / 'contract.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Existing natural pairwise contract differs')
    if not path.exists():
        atomic_json(path, contract)
    return c, contract


def compute(c, contract):
    sha = digest(OUTPUT / 'contract.json')
    native_sha = contract['natural_contract_sha256']
    legacy = legacy_index(NATURAL, c)
    started = time.perf_counter()
    for pi, subject in enumerate(c['subject_ids']):
        dest = OUTPUT / 'subjects' / f'{subject}.npz'
        if dest.exists():
            with np.load(dest) as a:
                if str(a['contract_sha256']) != sha:
                    raise ValueError(f'Wrong subject result: {subject}')
            continue
        shape = (len(c['seeds']), len(c['G']))
        means = np.empty(shape)
        minima = np.empty(shape)
        negative = np.zeros(shape, dtype=int)
        categories = np.empty((*shape, 3))
        hashes = np.empty(shape, dtype='U64')
        covariance_hashes = np.empty(shape, dtype='U64')
        for si, seed in enumerate(c['seeds']):
            for gi, g in enumerate(c['G']):
                path = NATURAL / 'conditions' / f'{subject}_G{g:.2f}_seed{seed}.npz'
                with np.load(path) as a:
                    record = validate_cache(a, path, c, native_sha, legacy)
                    cov = a['covariance'].copy()
                result = covariance_pair_phi_r(cov, tolerance=TOL)
                means[si, gi] = result['mean_nats']
                minima[si, gi] = result['audit']['minimum_nats']
                negative[si, gi] = result['audit']['tolerance_negative_count']
                categories[si, gi] = [result['category_means_nats'][k] for k in ['EE', 'II', 'EI']]
                hashes[si, gi] = record['source_sha256']
                covariance_hashes[si, gi] = digest(path)
        atomic_savez(dest, phi_r_nats=means, minimum_pair_nats=minima,
            tolerance_negative_pair_count=negative, category_means_nats=categories,
            source_sha256=hashes, natural_cache_sha256=covariance_hashes, contract_sha256=sha)
        print(json.dumps(dict(completed=pi+1, subjects=len(c['subject_ids']), subject=subject,
            minimum_pair_nats=float(minima.min()), seconds=round(time.perf_counter()-started, 1))), flush=True)


def summarize(c, contract):
    sha = digest(OUTPUT / 'contract.json')
    arrays = []
    categories = []
    minima = []
    negatives = 0
    for subject in c['subject_ids']:
        with np.load(OUTPUT / 'subjects' / f'{subject}.npz') as a:
            if str(a['contract_sha256']) != sha or a['phi_r_nats'].shape != (3, 41):
                raise ValueError(f'Incomplete subject result: {subject}')
            arrays.append(a['phi_r_nats'].copy())
            categories.append(a['category_means_nats'].copy())
            minima.append(float(a['minimum_pair_nats'].min()))
            negatives += int(a['tolerance_negative_pair_count'].sum())
    natural = np.stack(arrays)
    categories = np.stack(categories)
    with np.load(INTERVENTION / 'curves.npz') as a:
        inter = a['aligned_phi_r_nats'].copy()
        if a['subject_ids'].tolist() != c['subject_ids'][:-1]:
            raise ValueError('Intervention order changed')
    with np.load(DENSE / 'analysis_curves.npz') as a:
        if a['subject_ids'].tolist() != c['subject_ids']:
            raise ValueError('Original order changed')
        rates = a['rate'].copy()
    windows = json.loads((DENSE / 'window_summary.json').read_text())
    scenario = next(s for s in windows['scenarios'] if s['k_each_side'] == 2)
    frozen = {r['subject']: r for r in scenario['subject_rows']}
    rows = []
    for pi, subject in enumerate(c['subject_ids'][:-1]):
        old = frozen[subject]
        lm = subject_landmarks(c['G'], rates[pi].mean(0),
            dict(natural_pairwise=natural[pi].mean(0), interventional_pairwise=inter[pi].mean(0)))
        if lm['transition'] != old['independent_transition']:
            raise ValueError('Independent transition changed')
        for method in lm['methods'].values():
            point = method['extremum_G']
            lo, hi = old['window']['interval']
            method['hit'] = bool(method['interior'] and point is not None and lo-1e-12 <= point <= hi+1e-12)
        rows.append(dict(subject=subject, window=old['window'], **lm['methods'],
            xi=old['methods']['xi'], wms=old['methods']['wms']))
    dense_c = json.loads((DENSE / 'contract.json').read_text())
    cohorts = {}
    for name, ids in [('all93', c['subject_ids'][:-1]),
                      ('primary85', dense_c['holdout_subject_ids']),
                      ('development8', dense_c['development_subject_ids'])]:
        selected = [r for r in rows if r['subject'] in ids]
        cohorts[name] = dict(subject_count=len(selected), methods={key: dict(
            hit_count=sum(r[key]['hit'] for r in selected), denominator=len(selected),
            hit_rate=sum(r[key]['hit'] for r in selected)/len(selected),
            flat_count=sum(r[key]['flat'] for r in selected),
            boundary_count=sum(not r[key]['interior'] and not r[key]['flat'] for r in selected))
            for key in ['natural_pairwise', 'interventional_pairwise', 'xi', 'wms']})
    if cohorts['all93']['methods']['interventional_pairwise']['hit_count'] != 35:
        raise ValueError('Intervention 35/93 bridge changed')
    category_error = float(abs((categories*np.array([4950, 4950, 10000])).sum(-1)/19900-natural).max())
    if not np.isfinite(natural).all() or min(minima) < -TOL or category_error > 1e-12:
        raise ArithmeticError('Invalid aggregated natural pairwise result')
    summary = dict(contract_sha256=sha, cohorts=cohorts, subject_rows=rows,
        condition_count=int(np.prod(natural.shape)), individual_condition_count=93*3*41,
        pair_query_count=int(np.prod(natural.shape))*19900,
        audit=dict(minimum_pair_nats=min(minima), tolerance_nats=TOL,
            tolerance_negative_pair_count=negatives, significant_violation_count=0,
            source_hash_match_count=int(np.prod(natural.shape)),
            maximum_original_WMS_bridge_error_nats=0.0,
            maximum_category_closure_error_nats=category_error),
        contrast=contract['contrast'], regularization=contract['regularization'])
    atomic_savez(OUTPUT / 'curves.npz', phi_r_nats=natural,
        category_means_nats=categories, subject_ids=c['subject_ids'], G=c['G'],
        seeds=c['seeds'], contract_sha256=sha)
    summary['curves_sha256'] = digest(OUTPUT / 'curves.npz')
    atomic_json(OUTPUT / 'summary.json', summary)
    return summary, natural, inter


def plot(c, summary, natural, inter):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.family': 'sans-serif', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.5), layout='constrained')
        for ax, data, title, color in zip(axes[:2], [inter, natural[:-1]],
                ['Interventional pairwise $\\Phi^R$', 'Natural pairwise $\\Phi^R$'],
                ['#B96B36', '#287C9C']):
            for curve in data.mean(1):
                ax.plot(c['G'], curve, color=color, alpha=.35, lw=.65)
            ax.set(xlabel='$G$', ylabel='Pair mean (nats)', title=title, xlim=(0, 4), ylim=(0, None))
        methods = summary['cohorts']['all93']['methods']
        labels = ['Interventional pairwise', 'Natural pairwise', 'Interventional $\\Xi$', 'Natural source WMS']
        keys = ['interventional_pairwise', 'natural_pairwise', 'xi', 'wms']
        ax = axes[2]
        for y, key, color in zip([3, 2, 1, 0], keys, ['#B96B36', '#287C9C', '#B64D64', '#5A7970']):
            m = methods[key]
            rate = 100*m['hit_rate']
            ax.barh(y, rate, height=.52, color=color)
            ax.text(rate+1.2, y, f"{rate:.1f}% ({m['hit_count']}/93)", va='center', fontsize=8)
        ax.set(yticks=[3, 2, 1, 0], yticklabels=labels, xlim=(0, 104),
               xticks=[0, 25, 50, 75, 100], xlabel='Hits in frozen window (%)', title='93 participants; same 0.5 G windows')
        for ax, label in zip(axes, 'abc'):
            ax.text(-.1, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight='bold')
        fig.suptitle('All 19,900 E/I scalar pairs; 300 ms; each curve is a three-seed mean', fontsize=10)
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=260, bbox_inches='tight', facecolor='white')
        plt.close(fig)


def main():
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        c, contract = prepare()
        compute(c, contract)
        summary, natural, inter = summarize(c, contract)
        plot(c, summary, natural, inter)
    atomic_json(OUTPUT / 'completed.json', dict(status='complete',
        condition_count=summary['condition_count'], elapsed_seconds=time.perf_counter()-started,
        figure=str(FIGURE), visual_review='pending'))
    print(json.dumps(dict(cohorts=summary['cohorts'], audit=summary['audit'], figure=str(FIGURE))), flush=True)


if __name__ == '__main__':
    main()
