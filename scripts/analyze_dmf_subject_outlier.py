#!/usr/bin/env python3
"""Lightweight diagnosis of the minimum-spectral-radius native DMF subject.

Read frozen all-93 caches; do not simulate, refit EI, or change hit classifications.
Curves are seed 3/4/5 means in nats. G*rho(C) changes the horizontal coordinate
only, retaining amplitudes and native sample nodes. The effective-coupling curve
panel shows 0--2; each curve ends at its own cached support without extrapolation.
Ranks below are ascending strict ranks (1 + count of smaller values).
The manuscript was rechecked through Zotero on 2026-10-08: P6UJCVG8,
DXGC7JEA (Brain/Fig.2, Methods Eqs.5/7/8), MWIWKSVG (S1.2, S12.2.1).
Neither attachment establishes an explicit manuscript date or revision number.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/dmf_schaefer100/subject_curves_93_dense'
OUTPUT = ROOT / 'docs/reports/assets/dmf_subject_dense/minimum_rho_diagnosis.png'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose():
    contract = json.loads((BASE / 'contract.json').read_text())
    summary = json.loads((BASE / 'summary.json').read_text())
    fingerprint = digest(BASE / 'contract.json')
    assert summary['contract_sha256'] == fingerprint
    with np.load(BASE / 'inputs.npz') as data:
        assert json.loads(str(data['contract_json'])) == contract
        ids = data['subject_ids'][:-1].copy()
        matrices = data['connectivity'][:-1].copy()
        group = data['connectivity'][-1].copy()
        rho = data['spectral_radius'][:-1].copy()
    with np.load(BASE / 'analysis_curves.npz') as data:
        assert str(data['contract_sha256']) == fingerprint
        assert data['subject_ids'].tolist() == contract['subject_ids']
        g = data['G'].copy()
        xi = data['xi'][:-1].copy()
        rates = data['rate'][:-1].copy()
        closure = float(np.abs(data['whole_ei'] - data['partial_ei_sum'] - data['xi']).max())
    assert len(ids) == 93 and xi.shape == (93, 3, 41)
    assert contract['seeds'] == [3, 4, 5]
    assert np.array_equal(g, np.asarray(contract['G']))
    assert np.allclose(matrices.mean(0), group, atol=1e-12, rtol=0)
    assert np.isfinite(xi).all() and np.isfinite(rates).all()
    tolerance = contract['syn_tolerance_nats']
    violations = int(np.count_nonzero(xi < -tolerance))
    if violations:
        raise ArithmeticError(f'Xi violation: min={xi.min()} nats, threshold={-tolerance} nats, count={violations}')
    assert closure <= 1e-8
    target = int(np.argmin(rho))
    native = ROOT / 'data/neuromodulator_receptor_sc_100/CON_SC_1mio' / f'{ids[target]}.csv'
    assert digest(native) == contract['subject_hashes'][str(ids[target])]
    assert np.array_equal(np.loadtxt(native, delimiter=','), matrices[target])
    assert np.allclose(np.linalg.eigvalsh(matrices)[:, -1], rho, atol=1e-12, rtol=0)
    upper = np.triu_indices(100, 1)
    edges = matrices[:, upper[0], upper[1]]
    correlations = np.asarray([np.corrcoef(e, group[upper])[0, 1] for e in edges])
    normalized = matrices / np.linalg.norm(matrices, axis=(1, 2))[:, None, None]
    distances = np.linalg.norm(normalized - group / np.linalg.norm(group), axis=(1, 2))
    mean_xi = xi.mean(1)
    peaks = g[mean_xi.argmax(-1)]
    transitions = np.asarray([r['transition']['midpoint'] for r in summary['subject_rows']])
    assert [r['subject'] for r in summary['subject_rows']] == ids.tolist()
    assert np.array_equal(peaks, [r['methods']['xi']['extremum_G'] for r in summary['subject_rows']])
    features = {
        'spectral_radius': rho,
        'mean_weighted_strength': matrices.sum(-1).mean(-1),
        'nonzero_offdiagonal_density': (edges > 0).mean(-1),
        'edge_correlation_group': correlations,
        'normalized_frobenius_group_distance': distances,
        'xi_peak_G': peaks,
        'xi_peak_G_rho': peaks * rho,
        'rate_transition_G': transitions,
        'rate_transition_G_rho': transitions * rho,
        'xi_peak_height_nats': mean_xi.max(-1),
    }
    # Unnormalized weighted graph Laplacian. This is a structural comparison,
    # not a replacement of the simulated direct-coupling operator.
    laplacian = -matrices.copy()
    diagonal = np.arange(matrices.shape[1])
    laplacian[:, diagonal, diagonal] += matrices.sum(-1)
    assert np.allclose(laplacian.sum(-1), 0, atol=1e-12, rtol=0)
    laplacian_eigenvalues = np.linalg.eigvalsh(laplacian)
    assert laplacian_eigenvalues.min() >= -1e-12
    laplacian_radius = laplacian_eigenvalues[:, -1]
    features.update(laplacian_radius=laplacian_radius,
                    xi_peak_G_laplacian_radius=peaks * laplacian_radius,
                    rate_transition_G_laplacian_radius=transitions * laplacian_radius)
    statistics = {
        key: dict(target=float(v[target]), cohort_median=float(np.median(v)),
                  cohort_q25_q75=np.percentile(v, [25, 75]).tolist(),
                  ascending_rank=int(np.count_nonzero(v < v[target])) + 1)
        for key, v in features.items()
    }
    alpha = float(np.vdot(matrices[target], group) / np.vdot(group, group))
    target_audit = dict(condition_count=0, intervention_outside_count=0,
                        intervention_abnormal_rate_count=0, native_future_outside_count=0,
                        diagnostic_nonzero_boundary_count=0, native_boundary_hit_count=0,
                        xi_tolerance_negative_count=0, xi_violation_count=0,
                        natural_stabilization_not_detected_count=0)
    for path in sorted((BASE / 'conditions').glob(f'{ids[target]}_*.npz')):
        with np.load(path) as data:
            assert str(data['contract_sha256']) == fingerprint
            intervention = json.loads(str(data['intervention_diagnostics_json']))
            future = json.loads(str(data['native_future_diagnostics_json']))
            natural = json.loads(str(data['native_natural_json']))
            syn = json.loads(str(data['xi_audit_json']))
            target_audit['condition_count'] += 1
            target_audit['intervention_outside_count'] += intervention['outside_state_count']
            target_audit['intervention_abnormal_rate_count'] += intervention['abnormal_rate_count']
            target_audit['native_future_outside_count'] += future['outside_state_count']
            target_audit['diagnostic_nonzero_boundary_count'] += int(float(data['boundary_fraction']) != 0)
            target_audit['native_boundary_hit_count'] += natural['boundary_hit_count']
            target_audit['xi_tolerance_negative_count'] += syn['tolerance_negative_count']
            target_audit['xi_violation_count'] += syn['violation_count']
            target_audit['natural_stabilization_not_detected_count'] += int(not natural['stabilization_detected'])
    assert target_audit['condition_count'] == 123
    if target_audit['xi_violation_count']:
        raise ArithmeticError(f'Target cache Syn nonnegativity violations: {target_audit}')
    result = dict(
        subject=str(ids[target]), subject_count=93, features=statistics,
        xi_peak_G_per_seed=g[xi[target].argmax(-1)].tolist(),
        xi_SD_at_mean_peak_nats=float(xi[target, :, mean_xi[target].argmax()].std(ddof=1)),
        scalar_fit_to_group=alpha,
        scalar_fit_relative_residual=float(np.linalg.norm(matrices[target] - alpha * group) / np.linalg.norm(matrices[target])),
        rho_vs_peak_G_spearman=float(spearmanr(rho, peaks).statistic),
        rho_vs_transition_G_spearman=float(spearmanr(rho, transitions).statistic),
        peak_G_cv=float(peaks.std() / peaks.mean()),
        peak_G_rho_cv=float((peaks * rho).std() / (peaks * rho).mean()),
        laplacian_comparison=dict(
            definition='L=diag(C@1)-C; unnormalized, symmetric weighted Laplacian',
            rho_C_vs_rho_L_spearman=float(spearmanr(rho, laplacian_radius).statistic),
            peak_G_rho_L_cv=float((peaks * laplacian_radius).std() / (peaks * laplacian_radius).mean()),
            transition_G_rho_C_cv=float((transitions * rho).std() / (transitions * rho).mean()),
            transition_G_rho_L_cv=float((transitions * laplacian_radius).std() / (transitions * laplacian_radius).mean()),
            rho_L_over_rho_C_median=float(np.median(laplacian_radius / rho)),
            interpretation='Descriptive rescaling only; no new dynamics or local-stability calculation'),
        audit=dict(xi_minimum_nats=float(xi.min()), tolerance_nats=tolerance,
                   tolerance_negative_count=int(np.count_nonzero((xi < 0) & (xi >= -tolerance))),
                   violation_count=violations, EI_closure_max_nats=closure,
                   target_cache=target_audit),
        source_sha256={name: digest(BASE / name) for name in ('inputs.npz', 'analysis_curves.npz', 'contract.json')},
        new_simulations=0, new_estimates=0, hypothesis_tests=0,
    )
    return g, mean_xi, rho, peaks, target, result


def draw(g, xi, rho, peaks, target):
    target_color, cohort_color = '#482475', '#239b9b'
    others = np.arange(len(rho)) != target
    style = {'font.family': 'sans-serif', 'font.size': 10,
             'axes.spines.top': False, 'axes.spines.right': False}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.7), layout='constrained')
        for col, scaled in enumerate((False, True)):
            ax = axes[0, col]
            for i in np.flatnonzero(others):
                x = g * rho[i] if scaled else g
                ax.plot(x, xi[i], lw=.65, alpha=.30, color=cohort_color)
            x = g * rho[target] if scaled else g
            ax.plot(x, xi[target], color=target_color, lw=2.0, zorder=3)
            ax.scatter(x[xi[target].argmax()], xi[target].max(), s=40,
                       color=target_color, edgecolors='white', linewidths=.6, zorder=4)
            ax.set(xlabel=r'Effective coupling $G\rho(\mathbf{C})$' if scaled else 'Global coupling $G$',
                   ylabel=r'Integrated EI $\Xi$ (nats)', ylim=(0, 23),
                   xlim=(0, 2) if scaled else (0, 4))
            ax.set_title('Spectral scale (0–2 view)' if scaled else 'Native coupling', fontsize=11)
            ax = axes[1, col]
            y = peaks * rho if scaled else peaks
            ax.scatter(rho[others], y[others], s=21, color=cohort_color, alpha=.65, linewidths=0)
            ax.scatter(rho[target], y[target], s=85, color=target_color, marker='D',
                       edgecolors='white', linewidths=.7, zorder=4)
            ax.set(xlabel=r'Native SC spectral radius $\rho(\mathbf{C})$',
                   ylabel=r'$\Xi$ peak: $G\rho(\mathbf{C})$' if scaled else r'$\Xi$ peak: $G$',
                   xlim=(.30, 1.34), ylim=(0, 1.7) if scaled else (0, 3.0))
            rank = int(np.count_nonzero(y < y[target])) + 1
            ax.set_title(f'Target peak: ascending rank {rank}/93', fontsize=11)
        for letter, ax in zip('abcd', axes.flat):
            ax.text(-.12, 1.03, letter, transform=ax.transAxes, fontweight='bold', fontsize=13)
        handles = [Line2D([], [], color=target_color, lw=2, label='sub-10377 (minimum spectral radius)'),
                   Line2D([], [], color=cohort_color, lw=1, label='Other 92 participants (3-seed means)')]
        fig.legend(handles=handles, loc='outside upper center', frameon=False, ncol=2, fontsize=9)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT, dpi=240, bbox_inches='tight', facecolor='white')
        plt.close(fig)


def main():
    g, xi, rho, peaks, target, result = diagnose()
    draw(g, xi, rho, peaks, target)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(OUTPUT)


if __name__ == '__main__':
    main()
