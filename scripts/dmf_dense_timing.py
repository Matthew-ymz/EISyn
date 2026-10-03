"""Wall/CPU accounting around unchanged native calls, with cache exclusion."""
from contextlib import contextmanager, ExitStack
from functools import wraps
import time

import numpy as np


@contextmanager
def stage(records, name):
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        yield
    finally:
        row = records.setdefault(name, dict(wall_seconds=0., cpu_seconds=0.))
        row['wall_seconds'] += time.perf_counter()-wall
        row['cpu_seconds'] += time.process_time()-cpu


@contextmanager
def timed_call(owner, name, records, label):
    original = getattr(owner, name)
    @wraps(original)
    def wrapped(*args, **kwargs):
        with stage(records, label):
            return original(*args, **kwargs)
    setattr(owner, name, wrapped)
    try:
        yield
    finally:
        setattr(owner, name, original)


def native_condition_timed(sc, jf, g, seed, dmf, parameters, stabilization, records):
    # Each worker runs one condition at a time. Patches are process-local and
    # restored even on failure; the authoritative function and math are intact.
    from scripts import run_dmf_subject_curve_baselines as native
    with ExitStack() as scope:
        for owner, name, label in [
            (dmf, 'simulate_dmf', 'natural_shared'),
            (dmf, 'transform_rates_to_bold', 'bold_conversion'),
            (native, 'pairwise_gaussian_mmi_phi_r', 'phi_estimation'),
            (native, 'phi_floor_audit', 'phi_audit'),
            (native, 'simulate_sources', 'wms_future'),
            (native.wms_module, 'standardize', 'wms_standardization'),
            (native, 'native_wms', 'wms_estimation')]:
            scope.enter_context(timed_call(owner, name, records, label))
        with stage(records, 'native_total'):
            return native.condition(sc, jf, g, seed, dmf, parameters, stabilization)


COST_GROUPS = dict(
    xi_estimator=['xi_density_fit', 'xi_queries_audit'],
    phi_r_estimator=['phi_estimation', 'phi_audit'],
    wms_estimator=['wms_standardization', 'wms_estimation'],
    xi_pipeline=['xi_future', 'xi_density_fit', 'xi_queries_audit'],
    phi_r_pipeline=['natural_shared', 'bold_conversion', 'phi_estimation', 'phi_audit'],
    wms_pipeline=['natural_shared', 'wms_future', 'wms_standardization', 'wms_estimation'],
    natural_shared=['natural_shared'], xi_future=['xi_future'],
    wms_future=['wms_future'], bold_conversion=['bold_conversion'],
    independent_diagnostic=['independent_diagnostic'])

COMPLEXITY = dict(
    symbols='R ROIs, d=2R source/future state dimensions, N future samples, T natural timepoints, H future integration steps',
    xi_estimator='O(N d^2 + d^3): affine-TM moment fit, Schur complement and dense logdet; scalar marginals O(d)',
    phi_r_estimator='O(T R^2 + R^2): dense lag covariance and all R(R-1)/2 pairs with fixed-size 2/4-dimensional blocks',
    wms_estimator='O(N d^2 + d^3): least squares, covariance/pseudoinverse/logdet; singleton rank-one updates use a shared inverse, total O(d^3), not d separate dense inversions',
    future_simulation='O(H N R^2) for each dense SC future batch; Xi and WMS each require their own batch',
    natural_shared='O(T R^2), one natural trajectory shared by native PhiR and WMS; BOLD conversion O(T R)',
    diagnostic='O(L R^2), L=5000 independent diagnostic steps, shared evaluation overhead',
    caveat='Analytic leading orders of these fixed implementations, not measured scaling exponents. Timing at R=100,N=2048 cannot determine asymptotic order; different protocols and constants matter.')


def summarize_cost(records):
    eligible = [r for r in records if r['subject'] != 'group_mean_93' and r['eligible']]
    if not eligible:
        raise ValueError('No freshly computed individual conditions with stage timing')
    result = dict(condition_count=len(eligible),
        excluded_cached_individual_conditions=sum(r['subject'] != 'group_mean_93' and not r['eligible'] for r in records),
        subject_count=len(set(r['subject'] for r in eligible)), groups={}, complexity=COMPLEXITY,
        scope='One condition = one subject/G/seed; four concurrent workers, BLAS=1. Wall includes contention; CPU is per worker process. Cache reads excluded. Group-mean reference excluded.',
        shared_cost='Natural trajectory is shared: per-method pipeline costs include its dependency and must not be added. Independent order diagnostic is separate. Setup/input I/O/source draws/cache writes/plotting are excluded.',
        coverage='Fresh conditions only; development subjects lack timing on original coarse-grid cache points. Condition-weighted and equal-subject means both reported.')
    for group, stages in COST_GROUPS.items():
        result['groups'][group] = {}
        for clock in ('wall_seconds', 'cpu_seconds'):
            vals = np.asarray([sum(r['stages'][name][clock] for name in stages) for r in eligible])
            if not np.isfinite(vals).all() or np.any(vals < 0):
                raise ArithmeticError(f'Invalid runtime: {group}/{clock}')
            subject_means = [float(vals[[r['subject'] == subject for r in eligible]].mean())
                             for subject in sorted(set(r['subject'] for r in eligible))]
            result['groups'][group][clock] = dict(mean=float(vals.mean()), median=float(np.median(vals)),
                sd=float(vals.std(ddof=1)) if len(vals)>1 else None,
                q10=float(np.quantile(vals, .1)), q90=float(np.quantile(vals, .9)),
                equal_subject_mean=float(np.mean(subject_means)), subject_mean_min=min(subject_means), subject_mean_max=max(subject_means))
    return result
