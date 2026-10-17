"""Reproducible precision audit of saved full-joint EI, with a zero-mean control.

Raw cache files and their estimates are immutable. The control coefficient is one
by analytic derivation; no F1 or favorable node strengths enter its selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

from scripts.refine_spt_third_case import OUT, LADDER, SEEDS, TRUTH, read, fixed_tree_shares, shares
from scripts.kuramoto_six_topology_core import MLPChannel, save, spt_from_ei, recovery, features
from scripts.spt_bounded_joint_integration import zero_mean_projection_controls


def controlled_points(model, outer, particles, seed):
    source = OUT/f'ei_N{outer}_P{particles}_R{seed}.npz'
    corrected = OUT/f'ei_control_N{outer}_P{particles}_R{seed}.npz'
    if corrected.exists():
        data = np.load(corrected)
        return data['pointwise_ei'], data['raw_pointwise_ei']
    raw = np.load(source)['pointwise_ei']
    u = qmc.Sobol(12, scramble=True, seed=seed).random_base2(int(np.log2(outer)))
    x, z = (u[:, :6]-.5)*2*np.pi, ndtri(u[:, 6:])
    control = zero_mean_projection_controls(model, x, z)
    points = raw+control
    np.savez_compressed(corrected, pointwise_ei=points, raw_pointwise_ei=raw,
                        zero_mean_control=control, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    return points, raw


def fit_fourier_control(model):
    """Independent numerical pilot; choose variance model on held-out MSE only."""
    path = OUT/'fourier_control.json'
    if path.exists():
        return read(path)
    pilot = OUT/'ei_N512_P256_R9721.npz'
    if not pilot.exists():
        return None
    raw = np.load(pilot)['pointwise_ei']
    u = qmc.Sobol(12, scramble=True, seed=9721).random_base2(9)
    x, z = (u[:, :6]-.5)*2*np.pi, ndtri(u[:, 6:])
    target = (raw+zero_mean_projection_controls(model, x, z)).T
    candidates = []
    def fit(f, y, alpha):
        fm, ym = f.mean(0), y.mean(0)
        centered = f-fm
        beta = np.linalg.solve(centered.T@centered+alpha*np.eye(f.shape[1]), centered.T@(y-ym))
        return beta, ym-fm@beta
    for kind in ('node', 'universal_fourier'):
        f = features(x, kind=kind)
        for alpha in (1., 10., 100., 1000.):
            beta, intercept = fit(f[:256], target[:256], alpha)
            predicted = f[256:]@beta+intercept
            candidates.append(dict(feature_kind=kind, alpha=alpha,
                validation_mse=float(np.mean((target[256:]-predicted)**2))))
    selected = min(candidates, key=lambda r: r['validation_mse'])
    baseline = float(np.mean((target[256:]-target[:256].mean(0))**2))
    f = features(x, kind=selected['feature_kind'])
    beta, _ = fit(f, target, selected['alpha'])
    result = dict(pilot_scramble=9721, pilot_outer=512, pilot_particles=256,
        candidates=candidates, selection=selected, baseline_validation_mse=baseline,
        enabled=selected['validation_mse']<baseline, coefficients=beta.tolist(),
        feature_mean_under_prior='exactly zero for every nonconstant integer-frequency sine/cosine',
        selection_metric='held-out pilot MSE over all 63 EIs; no recovery score',
        independent_from_precision_scrambles=True)
    save(path, result)
    return result


def summarize(array, previous, diagnostics):
    mean, sem = array.mean(0), array.std(0, ddof=1)/np.sqrt(len(array))
    result = spt_from_ei(dict(zip(range(1, 64), mean)))
    reps = [spt_from_ei(dict(zip(range(1, 64), row))) for row in array]
    clades = [recovery(r['tree'], TRUTH)['scored_clades'] if r['tree'] else None for r in reps]
    stable = all(c is not None for c in clades) and len({json.dumps(sorted(c)) for c in clades}) == 1
    change = float(abs(mean-previous).max()) if previous is not None else None
    result.update(mean_ei_nats=mean.tolist(), maximum_ei_sem_nats=float(sem.max()), grid_change_nats=change,
                  minimum_ess=min(d['ess_min'] for d in diagnostics),
                  maximum_weight=max(d['maximum_weight'] for d in diagnostics),
                  scramble_module_tree_stable=stable,
                  replicate_module_f1=[recovery(r['tree'], TRUTH)['module_f1'] if r['tree'] else None for r in reps],
                  replicate_audits=[r['audit'] for r in reps])
    if result['tree']:
        result['recovery'] = recovery(result['tree'], TRUTH)
        result['node_shares_percent'] = shares(result['tree'])
        node_rows = [fixed_tree_shares(result['tree'], row) for row in array]
        share_sem = {key: float(np.std([r[key] for r in node_rows], ddof=1)/np.sqrt(len(array)))
                     for key in result['node_shares_percent']}
        result['node_share_sem_percentage_points'] = share_sem
        result['maximum_share_sem_percentage_points'] = max(share_sem.values())
        old_shares = fixed_tree_shares(result['tree'], previous) if previous is not None else None
        result['maximum_share_grid_change_percentage_points'] = (
            max(abs(value-old_shares[key]) for key, value in result['node_shares_percent'].items())
            if old_shares else None)
    result['numerically_resolved'] = bool(result['status']=='valid' and change is not None
        and change < .01 and sem.max() < .01 and result['minimum_ess'] >= 10 and result['maximum_weight'] <= .25
        and stable and result.get('maximum_share_sem_percentage_points', float('inf')) <= 1.
        and result.get('maximum_share_grid_change_percentage_points', float('inf')) <= 1.)
    return result


def run():
    model = MLPChannel(read(OUT/'model.json'))
    fourier = fit_fourier_control(model)
    previous_raw = previous_controlled = None
    levels = []
    for outer, particles in LADDER:
        if not all((OUT/f'ei_N{outer}_P{particles}_R{seed}.json').exists()
                   and (OUT/f'ei_N{outer}_P{particles}_R{seed}.npz').exists() for seed in SEEDS):
            break
        controlled, original, diagnostics, provenance = [], [], [], []
        for seed in SEEDS:
            points, raw = controlled_points(model, outer, particles, seed)
            if fourier and fourier['enabled']:
                u = qmc.Sobol(12, scramble=True, seed=seed).random_base2(int(np.log2(outer)))
                x = (u[:, :6]-.5)*2*np.pi
                points = points-(features(x, kind=fourier['selection']['feature_kind'])@
                                 np.array(fourier['coefficients'])).T
            controlled.append(points.mean(1))
            original.append(raw.mean(1))
            path = OUT/f'ei_N{outer}_P{particles}_R{seed}.json'
            row = read(path)
            diagnostics.extend(row['diagnostics'].values())
            provenance.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        controlled, original = np.array(controlled), np.array(original)
        cr = summarize(controlled, previous_controlled, diagnostics)
        rr = summarize(original, previous_raw, diagnostics)
        cr.update(outer=outer, particles=particles, replicate_count=4, provenance=provenance,
                  estimator='full-joint TM importance EI plus zero-mean projection and independent-pilot Fourier controls',
                  control_coefficient=1., raw_precision={k: rr.get(k) for k in
                      ('maximum_ei_sem_nats', 'grid_change_nats', 'maximum_share_sem_percentage_points',
                       'maximum_share_grid_change_percentage_points', 'numerically_resolved')},
                  share_precision_basis='same mean-selected tree cuts evaluated in every replicate/grid')
        cr['fourier_control_enabled'] = bool(fourier and fourier['enabled'])
        rr.update(outer=outer, particles=particles, replicate_count=4)
        previous_raw, previous_controlled = original.mean(0), controlled.mean(0)
        levels.append(dict(controlled=cr, raw=rr))
        save(OUT/'spt_projection_levels.json', levels)
        save(OUT/'spt_projection.json', cr)
        save(OUT/'spt_raw.json', rr)
        print(json.dumps({k: cr.get(k) for k in ('outer', 'particles', 'maximum_ei_sem_nats', 'grid_change_nats',
              'minimum_ess', 'maximum_weight', 'maximum_share_sem_percentage_points',
              'maximum_share_grid_change_percentage_points', 'numerically_resolved')}), flush=True)
    return levels


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
