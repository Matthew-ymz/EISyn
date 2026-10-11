#!/usr/bin/env python3
"""Compare pairwise PhiR with Xi on the same cached interventional E/I model.

All queries marginalize ONE full-system affine-TM density. PhiR still reads the
future pair, whereas Xi reads the complete future; this definitional difference
is retained and disclosed. Original BOLD results and frozen windows are read-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dmf_joint_readout import CommonTargetGame, audit_nonnegative
from scripts.dmf_dense_statistics import subject_landmarks, paired_hits
from scripts.run_dmf_subject_consistency import atomic_json, digest
from scripts.run_dmf_subject_curve_baselines import atomic_savez

DENSE = ROOT / 'results/dmf_schaefer100/subject_curves_93_dense'
DENSITIES = ROOT / 'results/dmf_schaefer100/critical_coalitions'
OUTPUT = ROOT / 'results/dmf_schaefer100/state_aligned_phi_r'
FIGURES = ROOT / 'docs/reports/assets/dmf_subject_dense'
TOL = 1e-8  # nats; raw values retained, significant violations fail.
W = {}


def batch_logdet(matrix):
    cholesky = np.linalg.cholesky(matrix)
    return 2 * np.log(np.diagonal(cholesky, axis1=-2, axis2=-1)).sum(-1)


def covariance_pair_phi_r(covariance, tolerance=TOL):
    """Gaussian-MMI PhiR over all scalar pairs; no new floors or projections.

    Covariance order: [past E(100), past I(100), future E(100), future I(100)].
    PhiR_ij = I(Xij;Yij) - I(Xi;Yi) - I(Xj;Yj)
              + min{I(Xi;Yi),I(Xi;Yj),I(Xj;Yi),I(Xj;Yj)}.
    Units are nats. The existing TM's ridge remains the sole regularization.
    """
    cov = np.asarray(covariance, float)
    if (cov.ndim != 2 or cov.shape[0] != cov.shape[1] or len(cov) % 4
            or not np.isfinite(cov).all()
            or not np.allclose(cov, cov.T, atol=1e-12, rtol=0)):
        raise ValueError('Expected finite symmetric [past E,I; future E,I] covariance')
    d = len(cov) // 2
    scales = np.sqrt(np.diag(cov))
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Invalid covariance diagonal')
    correlation = cov / scales[:, None] / scales[None, :]
    left, right = np.triu_indices(d, 1)
    indices = np.column_stack((left, right, left + d, right + d))
    blocks = correlation[indices[:, :, None], indices[:, None, :]]
    joint_mi = .5 * (batch_logdet(blocks[:, :2, :2])
                     + batch_logdet(blocks[:, 2:, 2:]) - batch_logdet(blocks))
    cross = blocks[:, :2, 2:]
    gap = 1 - cross ** 2
    if np.any(gap <= 0):
        raise ArithmeticError(f'Nonpositive scalar conditional variance: {gap.min()}')
    scalar_mi = -.5 * np.log1p(-cross ** 2)
    redundancy = scalar_mi.min(axis=(1, 2))
    values = joint_mi - scalar_mi[:, 0, 0] - scalar_mi[:, 1, 1] + redundancy
    audit = audit_nonnegative(values, tolerance=tolerance)
    n = d // 2
    categories = {
        'EE': (left < n) & (right < n),
        'II': (left >= n) & (right >= n),
        'EI': (left < n) & (right >= n),
    }
    return dict(mean_nats=float(values.mean()), pair_values_nats=values,
                pair_indices=np.column_stack((left, right)), audit=audit,
                category_means_nats={k: float(values[m].mean()) for k, m in categories.items()},
                category_counts={k: int(m.sum()) for k, m in categories.items()})


def prepare(output):
    old = json.loads((DENSE / 'contract.json').read_text())
    density = json.loads((DENSITIES / 'contract.json').read_text())
    for key in ['G', 'seeds', 'sample_count', 'support', 'horizon_steps',
                'dt_seconds', 'sigma', 'ridge', 'JFIC', 'estimator']:
        if old[key] != density[key]:
            raise ValueError(f'Incompatible density setting: {key}')
    if old['subject_ids'][:-1] != density['subject_ids']:
        raise ValueError('Density and curve subject order differs')
    if density['dense_contract_sha256'] != digest(DENSE / 'contract.json'):
        raise ValueError('Density cache was generated for a different dense protocol')
    for relative, expected in density['implementation_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f'Frozen density implementation changed: {relative}')
    missing = [f'{s}_G{g:.2f}_seed{seed}.npz' for s in density['subject_ids']
               for g in density['G'] for seed in density['seeds']
               if not (DENSITIES / 'density' / f'{s}_G{g:.2f}_seed{seed}.npz').exists()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing density caches; first: {missing[0]}')
    c = dict(version='same-intervention-ei-state-pairwise-phi-r-v1',
        subject_ids=density['subject_ids'], G=density['G'], seeds=density['seeds'],
        sample_count=old['sample_count'], support=old['support'], horizon_steps=300,
        dt_seconds=.001, estimator=old['estimator'], units='nats',
        observable='200 scalar E/I synaptic-gating coordinates; same as scalar-fine Xi',
        pair_count=19900, aggregation='mean over every unordered scalar pair, then mean over seeds',
        phi_r_definition='I(Xij;Yij)-I(Xi;Yi)-I(Xj;Yj)+minimum of four scalar cross-time MIs',
        intervention='same factorized independent uniform input draws and moment-matched Gaussian prior as Xi',
        numerical_rule='No added covariance floor, MI clipping, or PhiR projection; keep tolerance negatives',
        nonnegative_tolerance_nats=TOL,
        preserved_definition_difference='PhiR target is the future pair; Xi target is all 200 future coordinates. Pair averaging is not whole-system PhiID.',
        contrast='Exploratory bundled alignment: observable, prior, horizon, estimator and pair pool change together; cannot isolate BOLD conversion alone',
        landmarks='Original global maxima; first tie; flat or boundary maxima miss',
        window='Frozen independent firing-rate maximum-slope interval plus 0.2 G per side; width 0.5 G, no reselection',
        inference='Descriptive all93; original development8/primary85 reported separately; paired subject bootstrap and two-sided exact McNemar are exploratory',
        dense_contract_sha256=digest(DENSE / 'contract.json'),
        density_contract_sha256=digest(DENSITIES / 'contract.json'),
        frozen_curves_sha256=digest(DENSE / 'analysis_curves.npz'),
        frozen_windows_sha256=digest(DENSE / 'window_summary.json'),
        script_sha256=digest(Path(__file__)),
        manuscript=dict(parent='P6UJCVG8', title='Emergent hierarchical organization of causal interactions in complex systems',
            attachments=['DXGC7JEA', 'MWIWKSVG'], checked_date='2026-10-11',
            explicit_version_or_date=None,
            version_ambiguity='One body and one supplement; neither has an explicit draft date or revision. Metadata edits do not establish a new manuscript version.',
            locations='Main Brain/Fig2 pp6-7; Methods Eqs5-8 pp15-16; Supplement S12.2.1/EqsS82-S87 p23',
            discrepancies=['Main calls both native baselines BOLD-like, but native WMS uses full E/I states',
                'This interventional neural-state PhiR is an additional matched comparator, not the manuscript observational BOLD comparator',
                'Individual SC, three seeds and G=0..4 differ from the manuscript mean SC, eight seeds and G=0..3',
                'Affine-TM Gaussian moments approximate uniform-input nonlinear dynamics; no finite-sample correction']))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'subjects').mkdir(exist_ok=True)
    path = output / 'contract.json'
    if path.exists() and json.loads(path.read_text()) != c:
        raise ValueError('Existing aligned-analysis contract differs; use a new output directory')
    if not path.exists():
        atomic_json(path, c)
    return c


def initialize(output):
    global W
    output = Path(output)
    c = json.loads((output / 'contract.json').read_text())
    if digest(Path(__file__)) != c['script_sha256']:
        raise ValueError('Analysis implementation changed after contract was saved')
    W = dict(output=output, c=c, sha=digest(output / 'contract.json'),
             limiter=threadpool_limits(limits=1))


def subject_job(subject):
    w = W; c = w['c']
    path = w['output'] / 'subjects' / f'{subject}.npz'
    if path.exists():
        with np.load(path) as a:
            if str(a['contract_sha256']) != w['sha']:
                raise ValueError(f'Incompatible aligned subject cache: {subject}')
        return dict(subject=subject, reused=True)
    shape = (len(c['seeds']), len(c['G']))
    values = np.empty(shape); minima = np.empty(shape)
    negative_counts = np.zeros(shape, int); bridges = np.empty(shape)
    categories = np.empty((*shape, 3)); hashes = np.empty(shape, dtype='U64')
    started = time.perf_counter()
    for si, seed in enumerate(c['seeds']):
        for gi, g in enumerate(c['G']):
            name = f'{subject}_G{g:.2f}_seed{seed}.npz'
            with np.load(DENSITIES / 'density' / name) as a:
                if str(a['contract_sha256']) != c['density_contract_sha256']:
                    raise ValueError(f'Density provenance mismatch: {name}')
                cov = a['covariance'].copy()
                source_hash = str(a['source_sha256'])
                if int(a['noise_seed']) != 100000 * seed + 31017:
                    raise ValueError(f'Density future-noise seed mismatch: {name}')
                diagnostics = json.loads(str(a['diagnostics_json']))
                if diagnostics['outside_state_count'] or diagnostics['abnormal_rate_count']:
                    raise ArithmeticError(f'Invalid original simulation: {name}')
            with np.load(DENSE / 'conditions' / name) as a:
                if (str(a['contract_sha256']) != c['dense_contract_sha256']
                        or str(a['intervention_source_sha256']) != source_hash):
                    raise ValueError(f'Unpaired input samples: {name}')
                reference = json.loads(str(a['metrics_json']))
            game = CommonTargetGame(cov)
            totals = game.totals()
            audit_nonnegative([*game.audited, totals['xi_nats']], tolerance=TOL)
            bridge = max(abs(totals[k] - reference[k]) for k in ['xi_nats', 'whole_ei_nats'])
            if bridge > TOL:
                raise ArithmeticError(f'Xi density bridge failed: {name}: delta={bridge} nats')
            result = covariance_pair_phi_r(cov)
            values[si, gi] = result['mean_nats']
            minima[si, gi] = result['audit']['minimum_nats']
            negative_counts[si, gi] = result['audit']['tolerance_negative_count']
            bridges[si, gi] = bridge
            categories[si, gi] = [result['category_means_nats'][k] for k in ['EE', 'II', 'EI']]
            hashes[si, gi] = source_hash
    elapsed = time.perf_counter() - started
    atomic_savez(path, subject=subject, G=c['G'], seeds=c['seeds'],
        phi_r_nats=values, pair_minimum_nats=minima, tolerance_negative_pair_count=negative_counts,
        xi_bridge_error_nats=bridges, category_means_nats=categories,
        source_sha256=hashes, contract_sha256=w['sha'], elapsed_seconds=elapsed)
    return dict(subject=subject, reused=False, elapsed_seconds=elapsed,
                minimum_phi_r_pair_nats=float(minima.min()), maximum_xi_bridge_error_nats=float(bridges.max()))


def summarize(output, c):
    with np.load(DENSE / 'analysis_curves.npz') as a:
        if a['subject_ids'].tolist()[:-1] != c['subject_ids']:
            raise ValueError('Frozen curve subject order changed')
        old = {k: a[k][:-1].copy() for k in ['xi', 'phi_r', 'rate']}
    aligned = []; audits = []; category_values = []
    for subject in c['subject_ids']:
        with np.load(output / 'subjects' / f'{subject}.npz') as a:
            if str(a['contract_sha256']) != digest(output / 'contract.json'):
                raise ValueError(f'Incompatible subject result: {subject}')
            aligned.append(a['phi_r_nats'].copy())
            category_values.append(a['category_means_nats'].copy())
            audits.append(dict(subject=subject, minimum_pair_nats=float(a['pair_minimum_nats'].min()),
                tolerance_negative_pair_count=int(a['tolerance_negative_pair_count'].sum()),
                maximum_xi_bridge_error_nats=float(a['xi_bridge_error_nats'].max())))
    aligned = np.stack(aligned)
    categories = np.stack(category_values)
    if aligned.shape != old['xi'].shape or not np.isfinite(aligned).all():
        raise ValueError('Incomplete aligned curves')
    windows = json.loads((DENSE / 'window_summary.json').read_text())
    scenario = next(s for s in windows['scenarios'] if s['k_each_side'] == 2)
    frozen = {r['subject']: r for r in scenario['subject_rows']}
    rows = []
    for i, subject in enumerate(c['subject_ids']):
        landmark = subject_landmarks(c['G'], old['rate'][i].mean(0),
                                     {'phi_r': aligned[i].mean(0)})
        expected = frozen[subject]
        if landmark['transition'] != expected['independent_transition']:
            raise ValueError(f'Independent transition changed: {subject}')
        method = landmark['methods']['phi_r']
        point = method['extremum_G']; lo, hi = expected['window']['interval']
        method['hit'] = bool(method['interior'] and point is not None
                             and lo - 1e-12 <= point <= hi + 1e-12)
        method['distance_to_window_G'] = None if point is None else max(lo - point, 0., point - hi)
        rows.append(dict(subject=subject, transition=landmark['transition'], window=expected['window'],
            xi=expected['methods']['xi'], old_bold_phi_r=expected['methods']['phi_r'],
            aligned_phi_r=method))
    old_contract = json.loads((DENSE / 'contract.json').read_text())
    cohorts = {}
    for name, ids in [('all93', c['subject_ids']),
                      ('primary85', old_contract['holdout_subject_ids']),
                      ('development8', old_contract['development_subject_ids'])]:
        cohort = [r for r in rows if r['subject'] in ids]
        outcome = {k: np.array([r[k]['hit'] for r in cohort], bool)
                   for k in ['xi', 'old_bold_phi_r', 'aligned_phi_r']}
        comparisons = {}
        for label, first, second in [('xi_vs_aligned', 'xi', 'aligned_phi_r'),
                                     ('aligned_vs_old_bold', 'aligned_phi_r', 'old_bold_phi_r')]:
            p = paired_hits(outcome[first], outcome[second])
            n = p['xi_only'] + p['comparator_only']
            p['p_two_sided_exact'] = float(binomtest(p['xi_only'], n, .5).pvalue) if n else 1.
            p['first_method'] = first; p['second_method'] = second
            p['alternative'] = 'two-sided hit-probability difference; exploratory, no multiplicity correction'
            p.pop('p_one_sided_exact')
            # Generic names avoid attributing the first method to Xi in both contrasts.
            p['first_hits'] = p.pop('xi_hits'); p['first_only'] = p.pop('xi_only')
            comparisons[label] = p
        cohorts[name] = dict(subject_count=len(cohort),
            methods={k: dict(hit_count=int(v.sum()), denominator=len(v), hit_rate=float(v.mean()),
                flat_count=sum(r[k]['flat'] for r in cohort),
                boundary_count=sum(not r[k]['interior'] and not r[k]['flat'] for r in cohort))
                for k, v in outcome.items()}, comparisons=comparisons)
    summary = dict(contract_sha256=digest(output / 'contract.json'), cohorts=cohorts,
        subject_rows=rows, pair_count=19900, condition_count=int(np.prod(aligned.shape)),
        pair_query_count=int(np.prod(aligned.shape)) * 19900,
        audit=dict(minimum_pair_nats=min(a['minimum_pair_nats'] for a in audits),
            tolerance_nats=TOL, tolerance_negative_pair_count=sum(a['tolerance_negative_pair_count'] for a in audits),
            significant_violation_count=0,
            maximum_xi_bridge_error_nats=max(a['maximum_xi_bridge_error_nats'] for a in audits)),
        category_pair_counts=dict(EE=4950, II=4950, EI=10000),
        categories_diagnostic_only=True, inference=c['inference'], contrast=c['contrast'],
        preserved_definition_difference=c['preserved_definition_difference'])
    atomic_savez(output / 'curves.npz', subject_ids=c['subject_ids'], G=c['G'], seeds=c['seeds'],
        aligned_phi_r_nats=aligned, category_means_nats=categories,
        contract_sha256=digest(output / 'contract.json'))
    atomic_json(output / 'summary.json', summary)
    return summary, old, aligned


def plot(c, summary, old, aligned):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    with np.load(DENSE / 'inputs.npz') as a:
        rho = a['spectral_radius'][:-1].copy()
    FIGURES.mkdir(parents=True, exist_ok=True)
    style = {'font.family': 'sans-serif', 'font.size': 10,
             'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.5), layout='constrained')
        norm = Normalize(float(rho.min()), float(rho.max()))
        colors = plt.cm.viridis(norm(rho))
        series = [old['xi'].mean(1), old['phi_r'].mean(1), aligned.mean(1)]
        titles = [r'Integrated EI $\Xi$: E/I, intervention, 300 ms',
                  r'Original $\Phi^R$: BOLD-like, observation, 1 ms',
                  r'Aligned $\Phi^R$: E/I, intervention, 300 ms']
        curve_axes = list(axes.ravel()[:3])
        for ax, curves, title, letter in zip(curve_axes, series, titles, 'abc'):
            for i, y in enumerate(curves):
                ax.plot(c['G'], y, color=colors[i], lw=.7, alpha=.55)
            ax.set(xlim=(0, 4), xlabel='$G$', ylabel='Information (nats)', title=title)
            ax.text(-.11, 1.05, letter, transform=ax.transAxes, fontweight='bold', fontsize=12)
        curve_axes[0].set_ylim(bottom=0)
        curve_axes[2].set_ylim(bottom=0)
        ax = axes.ravel()[3]
        methods = summary['cohorts']['all93']['methods']
        for y, name, color in [(2, 'xi', '#B64D64'), (1, 'old_bold_phi_r', '#287C9C'),
                                (0, 'aligned_phi_r', '#D08A2E')]:
            m = methods[name]; rate = 100 * m['hit_rate']
            ax.barh(y, rate, height=.45, color=color)
            ax.text(rate + 1.3, y, f"{rate:.1f}%  ({m['hit_count']}/93)", va='center', fontsize=9)
        ax.set(xlim=(0, 117), xticks=[0, 25, 50, 75, 100], yticks=[2, 1, 0],
               yticklabels=[r'$\Xi$', r'Original $\Phi^R$', r'Aligned $\Phi^R$'],
               xlabel='Hits in own transition window (%)',
               title='All 93 participants; frozen 0.5 G window')
        ax.text(-.11, 1.05, 'd', transform=ax.transAxes, fontweight='bold', fontsize=12)
        fig.legend(handles=[Line2D([], [], color=plt.cm.viridis(.5), lw=1,
                        label='93 individual SCs; each curve is a 3-seed mean')],
                   loc='outside upper center', frameon=False)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=curve_axes,
                          orientation='horizontal', shrink=.60, aspect=45, pad=.04)
        cb.set_label('Native SC spectral radius; same individual colors in all curve panels')
        path = FIGURES / 'state_aligned_phi_r_comparison.png'
        fig.savefig(path, dpi=260, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit-subjects', type=int)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    c = prepare(args.output)
    if args.prepare_only:
        print(json.dumps(dict(subjects=len(c['subject_ids']), conditions=11439)))
        return
    subjects = c['subject_ids'][:args.limit_subjects] if args.limit_subjects else c['subject_ids']
    started = time.perf_counter()
    if args.workers == 1:
        initialize(args.output)
        for s in subjects:
            print(json.dumps(subject_job(s)), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
                initargs=(str(args.output),)) as pool:
            jobs = [pool.submit(subject_job, s) for s in subjects]
            for i, future in enumerate(as_completed(jobs), 1):
                print(json.dumps(dict(completed_subjects=i, total_subjects=len(subjects),
                                      **future.result())), flush=True)
    if not args.limit_subjects:
        summary, old, aligned = summarize(args.output, c)
        figure = plot(c, summary, old, aligned)
        atomic_json(args.output / 'completed.json', dict(status='complete',
            subjects=93, conditions=summary['condition_count'], elapsed_seconds=time.perf_counter() - started,
            figure=str(figure), contract_sha256=digest(args.output / 'contract.json'),
            visual_review='pending'))
        print(json.dumps(dict(cohorts=summary['cohorts'], audit=summary['audit'], figure=str(figure))), flush=True)


if __name__ == '__main__':
    main()
