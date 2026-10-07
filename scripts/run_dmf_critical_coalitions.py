#!/usr/bin/env python3
"""Frozen, resumable exact-member search in individual-SC natural SPTs (nats)."""
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
from scripts.dmf_subject_consistency import ResidualOracle, pair_values
from scripts.dmf_response_benchmark import fit_affine_joint, simulate_sources
from scripts.run_dmf_paired_spt_pilot import noise_seed, paired_sources
from scripts.run_dmf_subject_consistency import atomic_json, digest
from scripts.run_dmf_subject_curve_baselines import atomic_savez
from scripts.spt import SPTConfig, build_spt, flatten_nodes, spectral_candidate_selector
from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module

BASE = ROOT / 'results/dmf_schaefer100/critical_coalitions'
DENSE = ROOT / 'results/dmf_schaefer100/subject_curves_93_dense'
PILOT = ROOT / 'results/dmf_schaefer100/subject_consistency_pilot'
TOL = 1e-8
W = {}


def standard_tree(game):
    """Supplement S5: exact <=10, otherwise six fixed prefix orderings."""
    affinity, pair_audit = pair_values(game)
    result = build_spt(tuple(range(game.n)), ResidualOracle(game),
        config=SPTConfig(syn_tolerance=TOL),
        candidate_selector=spectral_candidate_selector(affinity, exact_max_size=10))
    nodes = [dict(members=list(n.sources), order=n.size, syn_nats_raw=n.syn_value,
                  left=list(n.children[0].sources), right=list(n.children[1].sources),
                  search=n.split_kind, depth=n.depth)
             for n in flatten_nodes(result.root) if n.children]
    if len(nodes) != game.n - 1 or abs(result.closure_error) > TOL:
        raise ArithmeticError(f'Tree size/closure failure: {len(nodes)}, {result.closure_error}')
    return dict(nodes=nodes, root_cross_roi_nats=result.root.xi_value,
        closure_error_nats=result.closure_error, candidate_audit=asdict(result.audit),
        selected_syn_audit=audit_nonnegative([n['syn_nats_raw'] for n in nodes]),
        pair_audit=pair_audit, query_audit=audit_nonnegative(game.audited))


def prepare(base):
    old = json.loads((DENSE / 'contract.json').read_text())
    summary = json.loads((DENSE / 'summary.json').read_text())
    rows = {r['subject']: r['transition'] for r in summary['subject_rows']}
    ids = old['subject_ids'][:-1]
    if len(ids) != 93 or any(not rows[s]['located'] for s in ids):
        raise ValueError('Expected 93 independently located transitions')
    windows = {}
    grid = np.array(old['G'])
    for s in ids:
        lo, hi = rows[s]['interval']
        index = np.flatnonzero((grid >= lo - .2 - 1e-12) & (grid <= hi + .2 + 1e-12))
        if len(index) != 6:
            raise ValueError(f'Expected six inclusive grid points: {s}: {index}')
        windows[s] = dict(interval=[lo - .2, hi + .2], transition=rows[s],
                          grid_indices=index.tolist(), required_hits=len(index)//2 + 1)
    dependencies = ['scripts/run_dmf_critical_coalitions.py', 'scripts/spt.py',
        'scripts/dmf_subject_consistency.py', 'scripts/dmf_joint_readout.py',
        'scripts/dmf_response_benchmark.py', 'scripts/run_dmf_paired_spt_pilot.py',
        'scripts/run_dmf_fixed_uniform_multihorizon.py', 'exp/TM/transport_map_density.py',
        'exp/brain/dmf_fig6.py', 'scripts/run_dmf_diffusive_fullstate_control.py']
    c = dict(version='individual-natural-spt-exact-members-v1', subject_ids=ids,
        development_subject_ids=old['development_subject_ids'],
        holdout_subject_ids=old['holdout_subject_ids'], subject_hashes=old['subject_hashes'],
        reference_source_sha256=old['reference_source_sha256'],
        input_sha256=digest(DENSE / 'inputs.npz'), dense_contract_sha256=digest(DENSE / 'contract.json'),
        transition_summary_sha256=digest(DENSE / 'summary.json'),
        G=old['G'], seeds=[3, 4, 5], sample_count=2048, support=[.3, .7],
        horizon_steps=300, dt_seconds=.001, sigma=.01, ridge=1e-6,
        JFIC=old['JFIC'], estimator=old['estimator'], units='nats', syn_tolerance_nats=TOL,
        source_seed_rule='100000*seed+31000', noise_seed_rule='100000*seed+31017',
        pairing='identical input samples and noise by seed across G and subjects',
        windows=windows, low_grid_indices=[0, 1, 2, 3], high_grid_indices=[37, 38, 39, 40],
        repeat_seed_count=2, critical_rule='strict majority of inclusive window points (4/6)',
        endpoint_stage_required_hits=2, endpoint_leakage_rule='any reproducible grid point',
        outside_window_rule='report every reproducible point outside the frozen window; any such point prevents strict window-only classification',
        search=dict(exact_max_size=10, large_nodes='Supplement S5 spectral, average-linkage, original-order prefix cuts',
                    extra_random_candidates=0, include_all_singletons=False, objective='raw residual',
                    complete_to_singletons=True, priors=None),
        candidate_orders=list(range(3, 100)), preferred_display_orders=list(range(3, 11)),
        background=dict(seeds=list(range(1000, 1100)), G=0.,
            repeat_unit='independent intervention/simulation seed, not SC',
            threshold='0.95 quantile of per-tree maximum raw Syn at each observed order',
            quantile_method='linear', minimum_observed_trees_per_order=20,
            uncalibrated_order='missing threshold; cannot qualify as strong'),
        discovery_gate=dict(minimum_critical_subjects=7, maximum_low_leak_subjects=0,
            maximum_high_leak_subjects=0,
            rationale='ceil(0.80*8), floor(0.10*8); window exclusivity evaluated separately'),
        validation_gate=dict(minimum_critical_subjects=68, maximum_low_leak_subjects=8,
                             maximum_high_leak_subjects=8),
        statistical_protocol=dict(unit='individual SC', tests='one-sided exact paired McNemar',
            multiplicity='Holm across both comparisons for every frozen candidate',
            intervals='Wilson 95% occurrence; exact one-sided95% leakage upper; paired-subject bootstrap95%',
            bootstrap_replicates=10000, bootstrap_seed=20261007,
            discovery_inference='descriptive selected estimates; no confirmatory p values'),
        manuscript=dict(checked_date='2026-10-07', parent='P6UJCVG8',
            title='Emergent hierarchical organization of causal interactions in complex systems',
            attachments=['DXGC7JEA', 'MWIWKSVG'], explicit_version_or_date=None,
            locations='Main Brain pp6-7; Methods Eqs5-12 pp15-17; Supplement S1.2 pp3-4, S5 pp8-9, S12.2.1 p23',
            ambiguity='Only one body and one supplement; no explicit revision/date. Metadata edits do not date the draft.',
            discrepancies=['Individual SC and unconstrained trees replace mean SC and Yeo constraints',
                'G=0..4 rather than manuscript G=0..3',
                'Uniform sampled inputs approximated by moment-matched factorized Gaussian affine-TM',
                'Inherited ROI label order remains inferred',
                'Rate transition is an operational landmark, not independently established criticality']),
        plan_correction='A width-0.5 inclusive interval has six 0.1-grid points, not five; preserve strict majority',
        implementation_sha256={p: digest(ROOT / p) for p in dependencies})
    base.mkdir(parents=True, exist_ok=True)
    for name in ['density', 'trees', 'background_density', 'background_trees']:
        (base / name).mkdir(exist_ok=True)
    path = base / 'contract.json'
    if path.exists() and json.loads(path.read_text()) != c:
        raise ValueError('Frozen coalition protocol changed; use a new result directory')
    if not path.exists():
        atomic_json(path, c)
    return c


def initialize(base):
    global W
    base = Path(base)
    c = json.loads((base / 'contract.json').read_text())
    for p, sha in c['implementation_sha256'].items():
        if digest(ROOT / p) != sha:
            raise ValueError(f'Frozen implementation changed: {p}')
    limiter = threadpool_limits(limits=1)
    with np.load(DENSE / 'inputs.npz') as a:
        matrices, jf = a['connectivity'][:-1].copy(), a['j_fic'].copy()
    W = dict(base=base, c=c, sha=digest(base / 'contract.json'), matrices=matrices,
        jf=jf, limiter=limiter, dmf=load_dmf_module(), sources={})
    W['p'] = W['dmf'].DMFParameters(dt=.001, sigma=.01)


def job(task):
    subject, gi, seed, background = task
    w = W; c = w['c']; g = c['G'][gi]
    key = f'background_seed{seed}' if background else f'{subject}_G{g:.2f}_seed{seed}'
    path = w['base'] / ('background_trees' if background else 'trees') / (key + '.json')
    if path.exists():
        r = json.loads(path.read_text())
        if r['contract_sha256'] != w['sha']:
            raise ValueError(f'Tree contract mismatch: {key}')
        return dict(condition=key, reused=True, elapsed_seconds=0.)
    started = time.perf_counter()
    density_path = w['base'] / ('background_density' if background else 'density') / (key + '.npz')
    imported = False; simulation_seconds = 0.
    if density_path.exists():
        with np.load(density_path) as a:
            if str(a['contract_sha256']) != w['sha']:
                raise ValueError(f'Density contract mismatch: {key}')
            covariance = a['covariance'].copy()
    else:
        old = PILOT / 'conditions' / (key + '.npz')
        if old.exists() and not background:
            with np.load(old) as a:
                pc = json.loads(str(a['contract_json']))
                for field in ['sample_count', 'support', 'horizon_steps', 'dt_seconds', 'sigma', 'ridge', 'estimator', 'JFIC']:
                    if pc[field] != c[field]:
                        raise ValueError(f'Imported density protocol mismatch: {field}')
                if pc['subject_hashes'][subject] != c['subject_hashes'][subject] or pc['reference_source_sha256'] != c['reference_source_sha256']:
                    raise ValueError('Imported density data mismatch')
                covariance = a['covariance'].copy()
                x = np.concatenate(paired_sources(seed, c['sample_count'], 100), axis=1)
                if str(a['source_sha256']) != hashlib.sha256(x.tobytes()).hexdigest() or int(a['noise_seed']) != noise_seed(seed):
                    raise ValueError('Imported density source/noise mismatch')
                diagnostics = json.loads(str(a['diagnostics_json']))
                imported = True
        else:
            if seed not in w['sources']:
                w['sources'][seed] = np.concatenate(paired_sources(seed, c['sample_count'], 100), axis=1)
            x = w['sources'][seed]
            sc = w['matrices'][0 if background else c['subject_ids'].index(subject)]
            t = time.perf_counter()
            y, diagnostics = simulate_sources(w['dmf'], x, sc, w['jf'], g, w['p'], noise_seed(seed))
            simulation_seconds = time.perf_counter() - t
            covariance = fit_affine_joint(x, y, .2, ridge=c['ridge'])
            if background:
                del w['sources'][seed]
        if diagnostics['outside_state_count']:
            raise ArithmeticError(f'State boundary violation: {key}: {diagnostics}')
        atomic_savez(density_path, covariance=covariance, contract_sha256=w['sha'],
            source_sha256=hashlib.sha256(x.tobytes()).hexdigest(), noise_seed=noise_seed(seed),
            imported_from=str(old) if imported else '', diagnostics_json=json.dumps(diagnostics))
    game = CommonTargetGame(covariance)
    totals = game.totals()
    if not background:
        with np.load(DENSE / 'conditions' / (key + '.npz')) as a:
            dense_metrics = json.loads(str(a['metrics_json']))
        delta = abs(totals['xi_nats'] - dense_metrics['xi_nats'])
        if delta > 1e-8:
            raise ArithmeticError(f'Dense-protocol reproduction failed: {key}: Xi delta={delta}')
    else:
        delta = None
    t = time.perf_counter()
    tree = standard_tree(game)
    tree_seconds = time.perf_counter() - t
    record = dict(subject=subject, G=g, seed=seed, background=background,
        contract_sha256=w['sha'], tree=tree, totals=totals,
        dense_xi_difference_nats=delta, imported_density=imported,
        simulation_seconds=simulation_seconds, tree_seconds=tree_seconds,
        elapsed_seconds=time.perf_counter() - started)
    atomic_json(path, record)
    return dict(condition=key, reused=False, elapsed_seconds=record['elapsed_seconds'],
                tree_seconds=tree_seconds, simulation_seconds=simulation_seconds)


def run(base, phase, workers, limit=None):
    c = prepare(base)
    subjects = c['development_subject_ids'] if phase == 'discovery' else c['holdout_subject_ids']
    if phase == 'background':
        tasks = [(None, 0, s, True) for s in c['background']['seeds']]
    else:
        if phase == 'validation':
            frozen = json.loads((base / 'frozen_candidates.json').read_text())
            if not frozen['candidates'] or frozen['contract_sha256'] != digest(base / 'contract.json'):
                raise ValueError('Validation requires nonempty frozen discovery candidates')
        tasks = [(p, gi, s, False) for p in subjects for gi in range(len(c['G'])) for s in c['seeds']]
    # Resume without rerunning expensive completed trees.
    folder = 'background_trees' if phase == 'background' else 'trees'
    def key(t):
        p, gi, s, bg = t
        return f'background_seed{s}' if bg else f'{p}_G{c["G"][gi]:.2f}_seed{s}'
    tasks = [t for t in tasks if not (base / folder / (key(t) + '.json')).exists()]
    if limit:
        tasks = tasks[:limit]
    started = time.perf_counter(); completed = 0
    atomic_json(base / (phase + '_launch.json'), dict(phase=phase, pid=os.getpid(),
        scheduled_new_conditions=len(tasks), workers=workers, limit=limit, contract_sha256=digest(base / 'contract.json')))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
        initializer=initialize, initargs=(str(base),)) as pool:
        futures = {pool.submit(job, t): t for t in tasks}
        for f in as_completed(futures):
            r = f.result(); completed += 1
            print(json.dumps(dict(completed=completed, scheduled=len(tasks), **r)), flush=True)
    atomic_json(base / (phase + ('_smoke_complete.json' if limit else '_complete.json')),
        dict(status='complete', new_conditions=completed, elapsed_seconds=time.perf_counter()-started,
             contract_sha256=digest(base / 'contract.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['prepare', 'background', 'discovery', 'validation'], required=True)
    p.add_argument('--output-dir', type=Path, default=BASE)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    if a.workers < 1 or (a.limit is not None and a.limit < 1):
        p.error('workers and limit must be positive')
    if a.phase == 'prepare':
        prepare(a.output_dir)
    else:
        run(a.output_dir, a.phase, a.workers, a.limit)


if __name__ == '__main__':
    main()
