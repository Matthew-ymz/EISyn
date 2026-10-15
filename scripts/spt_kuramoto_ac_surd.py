"""Joint-target interventional SURD using an explicitly approximate posterior TM.

Fit one local affine Gaussian posterior transport per target using normalized
importance particles. Evaluate its marginal cross-entropies on an independent
importance batch, allocate SURD at each target, then average. The full posterior
KL defect and agreement with the common-joint EI table audit this additional
approximation; it is never called an exact SURD computation of the SDE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_kuramoto_ac_selected import SelectedCase
from scripts.spt_kuramoto_finite_tm import (
    FullTargetIntegrator, posterior_proposal, proposal_sample, proposal_log_prob,
    TWO_PI, LOG_TWO_PI, DEFENSIVE, WIDE_FRACTION, WIDE_SCALE, check_time, save_json,
)
from scripts.spt_local_hyperedge_kuramoto import wrap
from scripts.surd_batch_allocation import allocate_batch
from scripts.compare_spt_surd_three_cases import maximum_weight_tree


def posterior_moments(model, targets, center, chol, *, outer, particles, seed, deadline, stream_offset=0):
    """Two nested particle levels from one normalized posterior IS stream."""
    levels = (particles // 2, particles)
    # Finite Sobol grids can include exactly zero. Use cell midpoints rather
    # than clipping a normal quantile or silently discarding that particle.
    net = qmc.Sobol(10, scramble=True, seed=seed, bits=30)
    if stream_offset:
        net.fast_forward(stream_offset * particles)
    points = (net.random(outer * particles) + .5 / 2**30).reshape(outer, particles, 10)
    result = {p: {'mean': np.empty((outer, 9)), 'covariance': np.empty((outer, 9, 9)),
                  'direct_specific': np.empty(outer), 'ess': np.empty(outer),
                  'maximum_weight': np.empty(outer), 'skew': np.empty(outer),
                  'excess_kurtosis': np.empty(outer)} for p in levels}
    batch = max(1, 65536 // particles)
    for start in range(0, outer, batch):
        check_time(deadline)
        stop = min(outer, start + batch)
        u = points[start:stop]
        c = center[start:stop]
        l = chol[start:stop]
        local = proposal_sample(c, l, u[:, :, 1:])
        wide = proposal_sample(c, WIDE_SCALE * l, u[:, :, 1:])
        sample = np.where(((u[:, :, 0] >= DEFENSIVE) &
                          (u[:, :, 0] < DEFENSIVE + WIDE_FRACTION))[:, :, None], wide, local)
        sample = np.where((u[:, :, 0] < DEFENSIVE)[:, :, None],
                          u[:, :, 1:] * TWO_PI - np.pi, sample)
        logq = np.logaddexp(np.logaddexp(
            np.log(1 - DEFENSIVE - WIDE_FRACTION) + proposal_log_prob(sample, c, l),
            np.log(WIDE_FRACTION) + proposal_log_prob(sample, c, WIDE_SCALE * l)),
            np.log(DEFENSIVE) - 9 * LOG_TWO_PI)
        y = np.repeat(targets[start:stop], particles, axis=0)
        likelihood = model.log_prob(sample.reshape(-1, 9), y).reshape(stop - start, particles)
        weights = likelihood - 9 * LOG_TWO_PI - logq
        delta = wrap(sample - c[:, None, :])
        for p, data in result.items():
            normalizer = logsumexp(weights[:, :p], axis=1)
            w = np.exp(weights[:, :p] - normalizer[:, None])
            mean = np.einsum('np,npi->ni', w, delta[:, :p])
            centered = delta[:, :p] - mean[:, None]
            covariance = np.einsum('np,npi,npj->nij', w, centered, centered)
            if np.any(np.linalg.eigvalsh(covariance) <= 0):
                raise ArithmeticError('Posterior covariance not positive definite; no ridge added')
            variance = np.diagonal(covariance, axis1=1, axis2=2)
            third = np.einsum('np,npi->ni', w, centered ** 3) / variance ** 1.5
            fourth = np.einsum('np,npi->ni', w, centered ** 4) / variance ** 2 - 3
            data['mean'][start:stop] = mean
            data['covariance'][start:stop] = covariance
            data['direct_specific'][start:stop] = np.sum(w * likelihood[:, :p], axis=1) - (normalizer - np.log(p))
            data['ess'][start:stop] = 1 / np.sum(w * w, axis=1)
            data['maximum_weight'][start:stop] = w.max(axis=1)
            data['skew'][start:stop] = abs(third).max(axis=1)
            data['excess_kurtosis'][start:stop] = abs(fourth).max(axis=1)
    return result


def marginal_specific(fit_mean, fit_covariance, evaluation_mean, evaluation_covariance):
    """log uniform volume + independently evaluated Gaussian log density."""
    n = len(fit_mean)
    output = np.zeros((n, 512))
    for mask in range(1, 512):
        idx = np.array([i for i in range(9) if mask >> i & 1])
        covariance = fit_covariance[:, idx[:, None], idx]
        delta = evaluation_mean[:, idx] - fit_mean[:, idx]
        second = evaluation_covariance[:, idx[:, None], idx] + delta[:, :, None] * delta[:, None, :]
        sign, logdet = np.linalg.slogdet(covariance)
        if np.any(sign <= 0):
            raise ArithmeticError('A Gaussian marginal covariance is not positive definite')
        trace = np.trace(np.linalg.solve(covariance, second), axis1=1, axis2=2)
        output[:, mask] = .5 * (len(idx) * LOG_TWO_PI - logdet - trace)
    return output


def specific_audit(specific, defect):
    margins = []
    masks = np.arange(512)
    for bit in range(9):
        absent = masks[(masks & (1 << bit)) == 0]
        margins.append(float(np.min(specific[:, absent | (1 << bit)] - specific[:, absent])))
    if min(margins) < -1e-8 or specific.min() < -1e-8:
        raise ArithmeticError(f'Specific KL monotonicity/nonnegativity failed: {min(margins)}, {specific.min()}')
    adjacent = []
    for order in range(1, 10):
        group = [m for m in range(1, 512) if m.bit_count() == order]
        if len(group) > 1:
            gaps = np.diff(np.sort(specific[:, group], axis=1), axis=1)
            adjacent.append(float(np.mean(gaps < abs(defect[:, None]))))
    return {'minimum_specific_nats': float(specific.min()),
            'minimum_inclusion_margin_nats': min(margins),
            'full_posterior_gaussian_kl_mean_nats': float(defect.mean()),
            'full_posterior_gaussian_kl_minimum_nats': float(defect.min()),
            'full_posterior_gaussian_kl_95percentile_nats': float(np.quantile(defect, .95)),
            'negative_kl_diagnostic_count': int(np.count_nonzero(defect < 0)),
            'kl_diagnostic_below_minus_01_count': int(np.count_nonzero(defect < -.01)),
            'ambiguous_adjacent_order_fraction': adjacent}


def run(case, outer=1024, particles=8192, scrambles=4):
    context = SelectedCase(case)
    folder = context.folder
    output = folder / f'surd_selected_n{outer}_p{particles}.json'
    model, _, digest = context.load_model()
    protocol = {'case': case, 'model_sha256': digest, 'outer': outer, 'particles': particles,
                'scrambles': scrambles, 'units': 'nats', 'allocation': 'statewise then expectation',
                'posterior_approximation': 'independent-IS fit/evaluation of circular local affine Gaussian TM',
                'tree': 'fixed global maximum sum of selected SURD clade weights',
                'source': 'independent uniform nine phases', 'target': 'complete future nine phases',
                'density_approximation_target_nats': .01, 'clipping': False}
    protocol['posterior_uniforms'] = '30-bit scrambled Sobol cell midpoints; open interval'
    data = json.loads(output.read_text()) if output.exists() else {'protocol': protocol, 'records': [], 'status': 'running'}
    if data['protocol'] != protocol:
        raise ValueError('SURD cache protocol mismatch')
    started = time.monotonic()
    previous = data.get('elapsed_seconds', 0.)
    def save():
        data['elapsed_seconds'] = previous + time.monotonic() - started
        save_json(output, data)
    try:
        for r in range(scrambles):
            if any(record['scramble'] == r for record in data['records']):
                continue
            seed = 20261015 + 1000 * r
            outer_stream = FullTargetIntegrator(model, outer=outer, max_particles=1,
                                                seed=seed, deadline=context.deadline())
            center, chol, inversion = posterior_proposal(model, np.zeros((outer, 9)), outer_stream.target,
                                                        np.arange(9), np.arange(9), iterations=16)
            fit = posterior_moments(model, outer_stream.target, center, chol, outer=outer,
                particles=particles, seed=seed + 300000, deadline=context.deadline())
            evaluation = posterior_moments(model, outer_stream.target, center, chol, outer=outer,
                particles=particles, seed=seed + 400000, deadline=context.deadline())
            rows = []
            native = {}
            for p in (particles // 2, particles):
                f, e = fit[p], evaluation[p]
                specific = marginal_specific(f['mean'], f['covariance'], e['mean'], e['covariance'])
                defect = e['direct_specific'] - specific[:, 511]
                audit = specific_audit(specific, defect)
                for n in sorted({min(256, outer), outer}):
                    atoms = allocate_batch(specific[:n], tolerance=1e-8)
                    synergy = atoms['synergy_sum'] / n
                    output_tree = maximum_weight_tree(synergy)
                    metrics = context.scored(output_tree['root_split'], output_tree['clades'])
                    rows.append({'outer': n, 'particles': p,
                                 'synergy_atoms_nats': synergy.tolist(),
                                 'redundancy_atoms_nats': (atoms['redundancy_sum'] / n).tolist(),
                                 'mean_specific_nats': (atoms['mi_sum'] / n).tolist(),
                                 'metrics': metrics, 'tree': output_tree, 'allocation_audit': atoms['audit']})
                native[f'specific_p{p}'] = specific
                native[f'gaussian_kl_p{p}'] = defect
                native[f'fit_covariance_p{p}'] = f['covariance']
                native[f'evaluation_covariance_p{p}'] = e['covariance']
                rows[-1]['specific_audit'] = audit
            np.savez_compressed(folder / f'surd_posterior_n{outer}_p{particles}_r{r}.npz', **native)
            high = evaluation[particles]
            chart_bound = np.max(abs(high['mean']) + 6 * np.sqrt(np.diagonal(high['covariance'], axis1=1, axis2=2)))
            data['records'].append({'scramble': r, 'layouts': rows,
                'proposal_audit': {'minimum_ess': float(min(fit[particles]['ess'].min(), high['ess'].min())),
                    'maximum_normalized_weight': float(max(fit[particles]['maximum_weight'].max(), high['maximum_weight'].max())),
                    'maximum_six_sd_chart_extent_rad': float(chart_bound),
                    'maximum_standardized_skew': float(high['skew'].max()),
                    'maximum_absolute_excess_kurtosis': float(high['excess_kurtosis'].max()), **inversion}})
            save()
            print(json.dumps({'case': case, 'stage': 'SURD posterior TM', 'scramble': r,
                              'f1': rows[-1]['metrics']['clade_f1'],
                              'gaussian_kl_mean': rows[-1]['specific_audit']['full_posterior_gaussian_kl_mean_nats']}), flush=True)
        high = [next(z for z in r['layouts'] if (z['outer'], z['particles']) == (outer, particles)) for r in data['records']]
        weights = np.array([r['synergy_atoms_nats'] for r in high])
        tree = maximum_weight_tree(weights.mean(0))
        metrics = context.scored(tree['root_split'], tree['clades'])
        data.update(status='complete', synergy_atoms_nats=weights.mean(0).tolist(),
                    metrics=metrics, tree=tree, numerical_replicate_metrics=[r['metrics'] for r in high],
                    maximum_atom_sem_nats=float((weights.std(0, ddof=1) / np.sqrt(scrambles)).max()) if scrambles > 1 else None,
                    approximation_audits=[r['specific_audit'] for r in high])
        data['gaussian_approximation_validated'] = all(
            abs(a['full_posterior_gaussian_kl_mean_nats']) < .01
            and a['kl_diagnostic_below_minus_01_count'] == 0 for a in data['approximation_audits'])
        save()
    except (TimeoutError, ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
        data.update(status='budget_exhausted' if isinstance(error, TimeoutError) else 'numerical_failure', error=str(error))
        save()
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('A', 'C'), required=True)
    parser.add_argument('--outer', type=int, default=1024)
    parser.add_argument('--particles', type=int, default=8192)
    parser.add_argument('--scrambles', type=int, default=4)
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        result = run(args.case, args.outer, args.particles, args.scrambles)
    print(json.dumps({'case': args.case, 'status': result['status']}))
