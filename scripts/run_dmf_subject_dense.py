#!/usr/bin/env python3
"""Prepare, smoke-test, run and resume the frozen all-93-person DMF curve sweep.

One native NPZ per subject/G/seed contains Xi, native PhiR/WMS and independent
order diagnostics. This workflow never retunes an estimator or peak criterion.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dmf_joint_readout import CommonTargetGame, audit_nonnegative
from scripts.dmf_response_benchmark import fit_affine_joint, simulate_sources
from scripts.run_dmf_paired_spt_pilot import paired_sources, noise_seed
from scripts.run_dmf_schaefer100_critical_horizon import stochastic_rate_trace
from scripts.run_dmf_subject_consistency import BASE as PILOT, atomic_json, digest
from scripts.run_dmf_subject_curve_baselines import atomic_savez
from scripts.dmf_dense_timing import stage, native_condition_timed
from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module

BASE = ROOT/'results/dmf_schaefer100/subject_curves_93_dense'
W = {}


def prepare(base, step=.1, maximum=4.):
    if step < .01 or maximum <= 0 or not np.isclose(round(maximum/step)*step, maximum):
        raise ValueError('Positive G_max must be a multiple of G_step; step >= .01')
    g = np.round(np.arange(round(maximum/step)+1)*step, 2).tolist()
    if len(set(g)) != len(g):
        raise ValueError('G grid collides at cache filename precision')
    pilot = json.loads((PILOT/'contract.json').read_text())
    native = json.loads((PILOT/'curve_native_contract.json').read_text())
    if native['pilot_contract'] != pilot:
        raise ValueError('Pilot and native protocols disagree')
    for rel, sha in native['implementation_sha256'].items():
        if digest(ROOT/rel) != sha:
            raise ValueError(f'Original native implementation changed: {rel}')
    rows = []
    for path in sorted((ROOT/'data/neuromodulator_receptor_sc_100/CON_SC_1mio').glob('sub-*.csv')):
        sc = np.loadtxt(path, delimiter=',')
        if sc.shape != (100, 100) or not np.isfinite(sc).all() or sc.min() < 0 or not np.allclose(sc, sc.T, atol=1e-12, rtol=0):
            raise ValueError(f'Invalid SC: {path}')
        rows.append((float(np.linalg.eigvalsh(sc)[-1]), path.stem, sc, digest(path)))
    if len(rows) != 93:
        raise ValueError(f'Expected 93 SC matrices, found {len(rows)}')
    rows.sort(key=lambda r: (r[0], r[1]))
    source = ROOT/'results/dmf_schaefer100/source/group_mean_native_mean_rate.npz'
    if digest(source) != pilot['reference_source_sha256']:
        raise ValueError('Original group/JFIC source changed')
    with np.load(source) as a:
        mean_sc = a['connectivity'].copy()
        jf = a['j_fic'][int(np.flatnonzero(np.isclose(a['G'], 1.))[0])].copy()
        if not np.allclose(a['j_fic'], jf, atol=0, rtol=0):
            raise ValueError('JFIC is not frozen')
    if not np.allclose(np.mean([r[2] for r in rows], axis=0), mean_sc, atol=1e-12, rtol=0):
        raise ValueError('Reference SC is not the same 93-person mean')
    ids = [r[1] for r in rows]+['group_mean_93']
    development = pilot['subject_ids'][:-1]
    files = ['scripts/run_dmf_subject_dense.py', 'scripts/analyze_dmf_subject_dense.py',
             'scripts/dmf_dense_statistics.py', 'scripts/dmf_curve_shape.py', 'scripts/dmf_dense_timing.py',
             'scripts/dmf_joint_readout.py', 'scripts/dmf_response_benchmark.py',
             'scripts/run_dmf_paired_spt_pilot.py', 'scripts/run_dmf_fixed_uniform_multihorizon.py',
             'scripts/run_dmf_schaefer100_critical_horizon.py',
             'exp/TM/transport_map_density.py']
    contract = dict(version='all93-native-dense-curves-v2-timing', subject_ids=ids,
        subject_hashes={r[1]: r[3] for r in rows}, reference_source_sha256=digest(source),
        subject_selection='All 93 native SC matrices, no normalization; spectral-radius order only for display',
        G=g, seeds=pilot['seeds'], sample_count=2048, support=[.3, .7],
        horizon_steps=300, dt_seconds=.001, sigma=.01, ridge=1e-6,
        estimator=pilot['estimator'], JFIC=pilot['JFIC'], syn_tolerance_nats=1e-8,
        intervention_seed_rules=dict(source='100000*nominal_seed+31000', noise='100000*nominal_seed+31017'),
        diagnostic_seeds=[103, 104, 105], diagnostic_steps=5000, diagnostic_burn_steps=3000,
        native_protocol=native, development_subject_ids=development,
        holdout_subject_ids=[s for s in ids[:-1] if s not in development],
        timing=dict(clocks=['perf_counter wall', 'process_time CPU'], workers=4, BLAS_threads=1,
            cache_policy='Exclude imported and resumed caches from runtime measurements',
            accounting='Estimator includes native audit; sample/trajectory preparation separate; natural trajectory shared',
            instrumentation='Scoped function wrappers around unchanged authoritative native_condition; no math changes',
            complexity='Analytic operation count of current dense implementations; no empirical asymptotic claim'),
        analysis=dict(primary_cohort='Unseen 85 subjects; all 93 reported separately',
            transition='Maximum positive independent mean-E-rate slope interval; boundary maximum unlocated',
            metric_landmark='Global maximum Xi/PhiR, global minimum signed WMS; no interpolation',
            hit='Interior metric extremum lies in independent interval, endpoints included; flat/edge metric extrema count as misses',
            tie_rule='First observed global extremum, no distance-based tie selection',
            test='One-sided exact paired McNemar via binomtest on discordant subjects; Holm correction of Xi-vs-PhiR and Xi-vs-WMS',
            alpha=.05, effect='Paired hit-rate difference; 10000 paired-subject bootstrap percentile CI, unadjusted two-sided 95%',
            bootstrap_seed=20261003, success='Positive paired difference and Holm p<.05 against BOTH comparators in primary cohort',
            secondary='Distance to interval; U and Q; 0/1/5-percent peak prominence; per-seed endpoint sensitivity',
            interpretation='Within this fixed model, SC cohort and original native protocols; not formula-only comparison or confirmed phase-transition truth',
            absence_controls='None independently confirmed; false-positive rate not estimable',
            result_based_changes='None; failure to beat baselines is retained'),
        manuscript=dict(parent='P6UJCVG8', attachment='DXGC7JEA', pages=19, version_date=None,
            sections='Brain/Fig.2; Methods Eqs.5,7,8', supplementary_appendices='unavailable',
            discrepancy='Main text calls both observational baselines BOLD-like; original native WMS uses full E/I states'),
        implementation_sha256={p: digest(ROOT/p) for p in files})
    base.mkdir(parents=True, exist_ok=True)
    path = base/'contract.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Frozen dense contract changed: use a different output directory')
    if not path.exists():
        atomic_json(path, contract)
    path = base/'inputs.npz'
    matrices = np.asarray([r[2] for r in rows]+[mean_sc])
    if path.exists():
        with np.load(path) as a:
            if json.loads(str(a['contract_json'])) != contract or not np.array_equal(a['connectivity'], matrices) or not np.array_equal(a['j_fic'], jf):
                raise ValueError('Frozen dense input mismatch')
    else:
        atomic_savez(path, subject_ids=ids, connectivity=matrices, j_fic=jf,
                     spectral_radius=[r[0] for r in rows]+[float(np.linalg.eigvalsh(mean_sc)[-1])],
                     contract_json=json.dumps(contract))
    return contract


def initialize(base):
    global W
    base = Path(base)
    c = json.loads((base/'contract.json').read_text())
    for rel, sha in c['implementation_sha256'].items():
        if digest(ROOT/rel) != sha:
            raise ValueError(f'Frozen dense implementation changed: {rel}')
    limiter = threadpool_limits(limits=1)
    dmf = load_dmf_module()
    with np.load(base/'inputs.npz') as a:
        if json.loads(str(a['contract_json'])) != c:
            raise ValueError('Inputs/contract mismatch')
        matrices, jf = a['connectivity'].copy(), a['j_fic'].copy()
    W = dict(base=base, c=c, digest=digest(base/'contract.json'), matrices=matrices, jf=jf, dmf=dmf,
        p=dmf.DMFParameters(dt=.001, sigma=.01),
        native_p=dmf.DMFParameters(**c['native_protocol']['native_parameters']),
        stabilization=dmf.StabilizationParameters(**c['native_protocol']['stabilization']),
        sources={s: np.concatenate(paired_sources(s, 2048, 100), 1) for s in c['seeds']},
        limiter=limiter)


def job(indices):
    pi, gi, si = indices
    w = W; c = w['c']; subject = c['subject_ids'][pi]; g = c['G'][gi]; seed = c['seeds'][si]
    sc = w['matrices'][pi]; path = w['base']/'conditions'/f'{subject}_G{g:.2f}_seed{seed}.npz'
    if path.exists():
        with np.load(path) as a:
            if str(a['contract_sha256']) != w['digest']:
                raise ValueError(f'Dense condition contract mismatch: {path.name}')
        return dict(condition=path.stem, reused=True, elapsed_seconds=0.)
    started = time.perf_counter()
    timing = {}
    original = PILOT/'conditions'/path.name
    original_native = PILOT/'curve_native_conditions'/path.name
    pilot_reused = bool(subject in c['development_subject_ids']+['group_mean_93'] and original.exists() and original_native.exists())
    if pilot_reused:
        with np.load(original) as a:
            if json.loads(str(a['contract_json'])) != c['native_protocol']['pilot_contract']:
                raise ValueError('Original intervention cache incompatible')
            game = CommonTargetGame(a['covariance'])
            diagnostic = json.loads(str(a['diagnostics_json']))
        with np.load(original_native) as a:
            if json.loads(str(a['contract_json'])) != c['native_protocol']:
                raise ValueError('Original native cache incompatible')
            native = {key: a[key].copy() for key in a.files if key != 'contract_json'}
        old_si = c['diagnostic_seeds'].index(100+seed)
        old_g = c['native_protocol']['pilot_contract']['G'].index(g)
        with np.load(PILOT/'dynamics'/f'{subject}.npz') as a:
            if json.loads(str(a['contract_json'])) != c['native_protocol']['pilot_contract']:
                raise ValueError('Original independent diagnostic cache incompatible')
            rate = float(a['mean_rate_hz'][old_si, old_g]); susceptibility = float(a['susceptibility'][old_si, old_g])
            outside = float(a['boundary_fraction'][old_si, old_g])
    else:
        with stage(timing, 'xi_future'):
            y, diagnostic = simulate_sources(w['dmf'], w['sources'][seed], sc, w['jf'], g, w['p'], noise_seed(seed))
        with stage(timing, 'xi_density_fit'):
            game = CommonTargetGame(fit_affine_joint(w['sources'][seed], y, .2, ridge=1e-6))
        native = native_condition_timed(sc, w['jf'], g, seed, w['dmf'], w['native_p'], w['stabilization'], timing)
        with stage(timing, 'independent_diagnostic'):
            rates, outside = stochastic_rate_trace(w['dmf'], initial_se=np.full(100, w['p'].init_se),
                initial_si=np.full(100, w['p'].init_si), connectivity=sc, coupling_g=g,
                j_fic=w['jf'], parameters=w['p'], steps=5000, seed=100+seed)
            late = rates[3000:]; rate = float(late.mean()); susceptibility = float(100*np.var(late.mean(1), ddof=1))
            if not np.isfinite(rates).all() or rates.min() < 0 or rates.max() > 500:
                raise ArithmeticError('Invalid independent diagnostic rates')
    if diagnostic['outside_state_count'] or diagnostic['abnormal_rate_count'] or outside:
        raise ArithmeticError(f'State/rate violation: {path.stem}')
    with stage(timing, 'xi_queries_audit'):
        metrics = game.totals(); metrics['partial_ei_sum_nats'] = float(game.scalar_ei.sum())
        if abs(metrics['whole_ei_nats']-metrics['partial_ei_sum_nats']-metrics['xi_nats']) > 1e-8:
            raise ArithmeticError('EI component closure failed')
        audit = audit_nonnegative([*game.audited, metrics['xi_nats']])
    source_sha = hashlib.sha256(w['sources'][seed].tobytes()).hexdigest()
    native = {'native_'+k: v for k, v in native.items()}
    atomic_savez(path, contract_sha256=w['digest'], subject=subject, G=g, seed=seed,
        metrics_json=json.dumps(metrics, allow_nan=False), intervention_diagnostics_json=json.dumps(diagnostic, allow_nan=False),
        xi_audit_json=json.dumps(audit, allow_nan=False), scalar_ei_nats=game.scalar_ei,
        intervention_source_sha256=source_sha, diagnostic_seed=100+seed,
        mean_rate_hz=rate, susceptibility=susceptibility, boundary_fraction=outside,
        timing_json=json.dumps(dict(eligible=not pilot_reused, stages=timing), allow_nan=False),
        pilot_cache_reused=pilot_reused, elapsed_seconds=time.perf_counter()-started, **native)
    return dict(condition=path.stem, reused=False, pilot_cache_reused=pilot_reused,
        elapsed_seconds=time.perf_counter()-started, xi_nats=metrics['xi_nats'], mean_rate_hz=rate)


def execute(base, workers=4, smoke=False):
    c = json.loads((base/'contract.json').read_text())
    if not 1 <= workers <= 4:
        raise ValueError('Use 1-4 workers, each with one BLAS thread')
    cache = base/'conditions'; cache.mkdir(exist_ok=True)
    jobs = [(pi, gi, si) for pi in range(len(c['subject_ids'])) for gi in range(len(c['G'])) for si in range(len(c['seeds']))]
    if smoke:
        target = lambda s, g: (c['subject_ids'].index(s), c['G'].index(g), 0)
        heldout = c['holdout_subject_ids'][len(c['holdout_subject_ids'])//2]
        jobs = [target('group_mean_93', 1.3), target(c['subject_ids'][0], c['G'][-1]),
                target(c['subject_ids'][-2], c['G'][-1]), target(heldout, 1.3)]
    planned = len(jobs); started = time.perf_counter(); results = []
    status_path = base/('smoke_status.json' if smoke else 'run_status.json')
    atomic_json(status_path, dict(status='running', planned_conditions=planned, workers=workers, started_at=time.time()))
    print(f'launch {"smoke" if smoke else "full"}: {planned} conditions, {workers} workers', flush=True)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize, initargs=(str(base),)) as pool:
            futures = {pool.submit(job, j): j for j in jobs}
            try:
                for future in as_completed(futures):
                    row = future.result(); results.append(row)
                    if not row['reused']:
                        print(json.dumps(row, allow_nan=False), flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    except BaseException as exc:
        atomic_json(status_path, dict(status='failed', error=str(exc), finished_conditions=len(results),
            planned_conditions=planned, elapsed_seconds=time.perf_counter()-started, ended_at=time.time()))
        raise
    finish = dict(status='smoke_complete' if smoke else 'simulations_complete',
        finished_conditions=len(results), planned_conditions=planned, workers=workers,
        new_conditions=sum(not r['reused'] for r in results), elapsed_seconds=time.perf_counter()-started,
        contract_sha256=digest(base/'contract.json'), ended_at=time.time())
    if smoke:
        finish['conditions'] = results
    atomic_json(status_path, finish)
    print(json.dumps(finish, allow_nan=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['prepare', 'smoke', 'run', 'analyze'], required=True)
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--g-step', type=float, default=.1)
    p.add_argument('--g-max', type=float, default=4.)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    with threadpool_limits(limits=1):
        if a.phase == 'prepare':
            c = prepare(a.base, a.g_step, a.g_max)
            print(json.dumps(dict(subjects=93, holdout=len(c['holdout_subject_ids']), G_count=len(c['G']),
                condition_count=len(c['subject_ids'])*len(c['G'])*len(c['seeds']),
                contract_sha256=digest(a.base/'contract.json')), indent=2))
        elif a.phase == 'analyze':
            from scripts.analyze_dmf_subject_dense import main as analyze
            analyze(a.base)
        else:
            execute(a.base, a.workers, smoke=a.phase == 'smoke')
            if a.phase == 'run':
                from scripts.analyze_dmf_subject_dense import main as analyze
                try:
                    analyze(a.base)
                except BaseException as exc:
                    status_path = a.base/'run_status.json'
                    status = json.loads(status_path.read_text())
                    atomic_json(status_path, dict(status, status='analysis_failed', error=str(exc), ended_at=time.time()))
                    raise


if __name__ == '__main__':
    main()
