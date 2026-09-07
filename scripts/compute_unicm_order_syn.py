"""Exhaustive minimum-bipartition Syn, paired across checkpoints and leads."""
from pathlib import Path
import sys
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.unicm_peid_syn_analysis import MODE_NAMES, sample_full_history_mode_inputs

OUTPUT = ROOT / 'results/unicm_exhaustive_order_syn/atoms.npz'
TOLERANCE = 1e-8  # Native bits; shared by all synchronized UniCM Syn analyses.
ESTIMATOR_VERSION = 'affine_tm_independent_uniform_v1'


def fit_independent_source_affine_channel(histories, targets):
    """Fit an affine channel once while leaving the intervention prior explicit."""
    history = np.asarray(histories, dtype=float)
    y = np.asarray(targets, dtype=float)
    if history.ndim != 3 or history.shape[2] != len(MODE_NAMES):
        raise ValueError('Expected sample x history x 11 modes')
    x = history.reshape(len(history), -1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2 or len(y) != len(x) or len(x) <= x.shape[1]:
        raise ValueError('Invalid target shape or insufficient samples for affine fit')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite samples')
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    covariance = x.T @ x / (len(x) - 1)
    weights = np.linalg.solve(covariance, x.T @ y / (len(x) - 1))
    residual = y - x @ weights
    residual_covariance = residual.T @ residual / (len(x) - 1)
    return weights, residual_covariance, {
        'source_covariance_condition': float(np.linalg.cond(covariance)),
    }


def independent_source_ei_table_from_channel(
    weights,
    residual_covariance,
    *,
    target_indices=None,
    intervention_bound=4.,
    jitter=1e-6,
):
    """Evaluate a fitted channel under the known product-uniform source prior."""
    weights = np.asarray(weights, dtype=float)
    noise = np.asarray(residual_covariance, dtype=float)
    if target_indices is not None:
        indices = np.asarray(target_indices, dtype=int)
        weights = weights[:, indices]
        noise = noise[np.ix_(indices, indices)]
    if weights.ndim != 2 or noise.shape != (weights.shape[1], weights.shape[1]):
        raise ValueError('Incompatible affine-channel shapes')
    if not np.isfinite(intervention_bound) or intervention_bound <= 0 or not np.isfinite(jitter) or jitter < 0:
        raise ValueError('Invalid intervention bound or jitter')
    ridge = jitter * max(float(np.trace(noise) / len(noise)), 1.)
    noise = noise + ridge * np.eye(len(noise))
    np.linalg.cholesky(noise)
    variance = intervention_bound**2 / 3.
    contributions = [variance * weights[i::len(MODE_NAMES)].T @ weights[i::len(MODE_NAMES)]
                     for i in range(len(MODE_NAMES))]
    conditional = [noise]
    logdets = np.empty(1 << len(MODE_NAMES))
    for mask in range(1 << len(MODE_NAMES)):
        if mask:
            bit = mask & -mask
            conditional.append(conditional[mask ^ bit] + contributions[bit.bit_length()-1])
        factor = np.linalg.cholesky(conditional[mask])
        logdets[mask] = 2 * np.log(np.diag(factor)).sum()
    full = len(logdets)-1
    names = tuple(MODE_NAMES)
    table = {tuple(names[i] for i in range(len(names)) if mask >> i & 1):
             float((logdets[full] - logdets[full ^ mask]) / (2*np.log(2)))
             for mask in range(1, full+1)}
    audit = dict(source_variance=variance, residual_ridge=ridge,
                 residual_min_eigenvalue=float(np.linalg.eigvalsh(noise).min()))
    return table, audit


def independent_source_ei_table(histories, targets, *, intervention_bound=4., jitter=1e-6):
    """Fit one affine channel, then evaluate it under the known product prior.

    X covariance is bound**2 / 3 times identity. Rebuild Y covariance and
    every marginal from that same channel; never combine this prior with
    the empirical Y covariance. This is a Gaussian/degree-1 TM readout
    approximation to uniform interventions, not their exact nonlinear MI.
    Jitter is added once to the residual covariance, in target units.
    """
    weights, noise, fit_audit = fit_independent_source_affine_channel(histories, targets)
    table, audit = independent_source_ei_table_from_channel(
        weights,
        noise,
        intervention_bound=intervention_bound,
        jitter=jitter,
    )
    audit.update(fit_audit)
    return table, audit


def minimum_syn(table):
    """EI(S)-max_A[EI(A)+EI(S minus A)]; singleton terms cancel from Xi."""
    names = tuple(MODE_NAMES)
    ei = np.zeros(1 << len(names))
    for subset, value in table.items():
        ei[sum(1 << names.index(name) for name in subset)] = value
    masks = np.array([m for m in range(1, len(ei)) if m.bit_count() >= 2])
    values = []
    for mask in masks:
        anchor = int(mask) & -int(mask)
        left = (int(mask) - 1) & int(mask)
        best = -np.inf
        while left:
            if left & anchor:
                best = max(best, ei[left] + ei[int(mask) ^ left])
            left = (left - 1) & int(mask)
        values.append(ei[mask] - best)
    return np.asarray(values), masks


def load_order_means(path=OUTPUT):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata'])) if 'metadata' in data else {}
        if metadata.get('estimator') != ESTIMATOR_VERSION:
            raise ValueError('Recompute cache with independent-source affine TM')
        values = data['values_bits']
        sizes = data['orders']
        tolerance = float(data['tolerance_bits'])
        if not np.isfinite(tolerance) or tolerance != TOLERANCE:
            raise ValueError('Cache tolerance must match the fixed native-bit tolerance')
        if values.shape != (3, 24, 2036) or not np.isfinite(values).all():
            raise ValueError('Incomplete exhaustive checkpoint/lead/coalition cache')
        bad = values < -tolerance
        if bad.any():
            raise ValueError(f'Syn violation: minimum={values.min()}, threshold={-tolerance}, affected_count={bad.sum()}')
        return np.stack([values[:, :, sizes == k].mean(axis=2).mean(axis=0) for k in range(2, 12)])


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    histories = sample_full_history_mode_inputs(n_samples=16384, intervention_bound=4., seed=20260901)
    result = []
    audits = []
    metadata = dict(estimator=ESTIMATOR_VERSION, approximation='affine degree-1 TM / Gaussian readout',
                    jitter=1e-6, regularization='once on residual covariance', n_samples=16384,
                    sampling_seed=20260901, intervention_bound=4., start_month=0,
                    source_prior='independent; covariance=(16/3)*I',
                    aggregation='all coalitions within order, then equal checkpoint mean', projection='none')
    for checkpoint in (1, 2, 3):
        path = ROOT / f'results/unicm_xi_hierarchy_uniform_n16384/cache/checkpoint{checkpoint}_samples16384_sampling20260901_bound4_fullhist12_start0_cpu.npz'
        with np.load(path) as data:
            predictions = data['all_mode_targets']
        if predictions.shape != (16384, 24, 11) or not np.isfinite(predictions).all():
            raise ValueError(f'Invalid prediction cache: {path}')
        rows = []
        for lead in range(1, 25):
            table, fit_audit = independent_source_ei_table(histories, predictions[:, lead-1, :])
            values, masks = minimum_syn(table)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite Syn')
            bad = values < -TOLERANCE
            diagnostic = dict(n_samples=16384, checkpoint=checkpoint, lead=lead,
                              minimum_bits=float(values.min()), threshold_bits=-TOLERANCE,
                              affected_count=int(bad.sum()), coalition_count=len(values),
                              tolerance_negative_count=int(((values < 0) & ~bad).sum()), **fit_audit)
            audits.append(diagnostic)
            print(f'checkpoint={checkpoint} lead={lead} minimum={values.min():.8g} violations={bad.sum()}', flush=True)
            if bad.any():
                (OUTPUT.parent / 'independent_source_audit.json').write_text(json.dumps(dict(status='failed',metadata=metadata,audits=audits),indent=2))
                raise ValueError(f'Syn violation: minimum={values.min()}, threshold={-TOLERANCE}, affected_count={bad.sum()}')
            rows.append(values)
        result.append(rows)
    values = np.asarray(result)
    metadata.update(tolerance_negative_count=int(((values < 0) & (values >= -TOLERANCE)).sum()), minimum_bits=float(values.min()))
    np.savez_compressed(OUTPUT, values_bits=values, masks=masks, orders=np.array([int(m).bit_count() for m in masks]), checkpoints=[1,2,3], leads=np.arange(1,25), tolerance_bits=TOLERANCE, metadata=json.dumps(metadata))
    load_order_means()
    (OUTPUT.parent / 'independent_source_audit.json').write_text(json.dumps(dict(status='complete',metadata=metadata,audits=audits),indent=2))
    from scripts.plot_earth_system_main_figures import configure_matplotlib, plot_unicm_figure
    configure_matplotlib()
    plot_unicm_figure(ROOT/'fig/earth_unicm_hierarchical_ei')

if __name__ == '__main__':
    main()
