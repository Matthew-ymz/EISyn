"""Fixed weak bridges and full-joint EI refinement under a user-approved deadline."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import itertools
import json
import os
import multiprocessing
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.special import ndtr
from threadpoolctl import threadpool_limits

from scripts.kuramoto_six_topology_core import (
    MLPChannel, LinearChannel, configuration, dataset, fit_mlp, channel_diagnostics,
    save, check, recovery, spt_from_ei, mask_of,
)
from scripts.spt import nontrivial_bipartitions
from scripts.spt_bounded_joint_integration import BoundedJointIntegrator, DoublePrecisionMLPChannel, MPSDensityMLPChannel
from scripts.compare_kuramoto_short_trajectory_clustering import (
    standardize, distance_correlation_matrix, GroupMI, hierarchy_result, mic_tree,
)

OUT = ROOT / 'results/spt_third_case_weak_coupling'
LOG = ROOT / 'docs/log/spt_third_case_weak_coupling'
TRUTH = [[1, 2, 3], [1, 2, 3, 4], [5, 6]]
METHODS = ('ward', 'correlation_hac', 'dcor_hac', 'mi_hac', 'mic')
SEEDS = (880, 1880, 2880, 3880)
LADDER = ((512, 256), (2048, 512), (4096, 512), (8192, 1024))


def read(path):
    return json.loads(Path(path).read_text())


def config():
    cfg = configuration('B', 2)
    cfg['module_truth'] = copy.deepcopy(TRUTH)
    cfg['supports'] += [[1, 5], [4, 6]]
    cfg['weights'] += [.02, .02]
    cfg['id'] = 'B_2_bridges15_46_w0p02'
    cfg['base_average_coupling_rms'] = cfg['average_coupling_rms']
    cfg['average_coupling_rms'] = float(np.sqrt(sum(len(e)*w*w for e, w in zip(cfg['supports'], cfg['weights']))/12))
    cfg.pop('mechanism_hash')
    cfg['mechanism_hash'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    return cfg


def stage(name, **extra):
    payload = dict(phase=name, updated_epoch=time.time(), pid=os.getpid(), **extra)
    save(OUT/'status.json', payload)
    print(json.dumps(payload), flush=True)


def baselines(x, y, deadline):
    views = {}
    for view, raw in (('increment', (y-x)[:, :, None]),
                      ('transition', np.stack((np.sin(x), np.cos(x), y-x), axis=2))):
        check(deadline)
        stage('baselines', view=view, samples=len(x))
        blocks = standardize(raw)
        n, _, dimensions = blocks.shape
        jittered = blocks + np.random.default_rng(20261017).normal(0, 1e-10, blocks.shape)
        profiles = blocks.transpose(1, 0, 2).reshape(6, -1)
        corr = profiles @ profiles.T / (n*dimensions)
        np.fill_diagonal(corr, 1)
        dcor = distance_correlation_matrix(blocks)
        info = GroupMI(jittered)
        mi = np.zeros((6, 6))
        for i, j in itertools.combinations(range(6), 2):
            check(deadline)
            mi[i, j] = mi[j, i] = info([i+1], [j+1])
        scale = max(float(abs(mi).max()), 1e-12)
        methods = dict(ward=hierarchy_result(linkage(profiles, method='ward', optimal_ordering=True)))
        for name, distance in (('correlation_hac', 1-abs(corr)), ('dcor_hac', 1-dcor), ('mi_hac', 1-mi/scale)):
            np.fill_diagonal(distance, 0)
            methods[name] = hierarchy_result(linkage(squareform(distance, checks=False), method='average', optimal_ordering=True))
        methods['mic'] = mic_tree(info, dimensions)
        for row in methods.values():
            row['recovery'] = recovery(row['tree'], TRUTH)
        views[view] = dict(methods=methods, group_mi_queries=info.records(),
                           matrices=dict(correlation=corr.tolist(), dcor=dcor.tolist(), pair_mi_nats=mi.tolist()))
        save(OUT/'baselines_partial.json', dict(views=views))
    return dict(views=views, best_hierarchical_f1=max(
        v['methods'][m]['recovery']['module_f1'] for v in views.values() for m in METHODS))


def preflight(deadline):
    stage('bounded_transport_preflight')
    model = LinearChannel(np.eye(6), .01*np.eye(6))
    engine = BoundedJointIntegrator(model, 512, 256, 711, deadline)
    e, points, diagnostic = engine.information()
    py = np.log(ndtr((engine.y+np.pi)/.1)-ndtr((engine.y-np.pi)/.1))-np.log(2*np.pi)
    conditional = -.5*((engine.y-engine.x)/.1)**2-np.log(.1*np.sqrt(2*np.pi))
    exact = np.array([sum((conditional[:, i]-py[:, i]) for i in range(6) if mask >> i & 1)
                      for mask in range(1, 64)])
    error = float(abs(points-exact).mean(1).max())
    mean_error = float(abs(points.mean(1)-exact.mean(1)).max())
    audit = spt_from_ei(e)['audit']
    result = dict(maximum_mean_ei_error_nats=mean_error, maximum_pointwise_mae_nats=error,
                  audit=audit, passed=bool(mean_error < .01 and audit['violation_count']==0
                                           and audit['xi_violation_count']==0))
    save(OUT/'preflight.json', result)
    if not result['passed']:
        raise ArithmeticError(f'Independent analytic control failed: {result}')


def shares(tree):
    def walk(n):
        return ([n] if n['children'] else []) + [k for c in n['children'] for k in walk(c)]
    ns = walk(tree)
    total = sum(n['syn_nats'] for n in ns)
    return {''.join(map(str, n['sources'])): 100*n['syn_nats']/total for n in ns}


def fixed_tree_shares(tree, ei):
    """Compare numerical strength on a common path, even if tied pairs swap."""
    total = ei[62] - sum(ei[(1 << i)-1] for i in range(6))
    result = {}
    def visit(node):
        if not node['children']:
            return
        left, right = node['children']
        residual = (ei[mask_of(node['sources'])-1] - ei[mask_of(left['sources'])-1]
                    - ei[mask_of(right['sources'])-1])
        result[''.join(map(str, node['sources']))] = 100*residual/total
        visit(left)
        visit(right)
    visit(tree)
    return result


def density_channel():
    audit_path = OUT/'mps_ei_equivalence.json'
    use_gpu = audit_path.exists() and read(audit_path).get('passed', False)
    kind = MPSDensityMLPChannel if use_gpu else DoublePrecisionMLPChannel
    return kind(read(OUT/'model.json'))


def one_integral(arguments):
    outer, particles, seed, deadline = arguments
    path = OUT/f'ei_N{outer}_P{particles}_R{seed}.json'
    if path.exists():
        return read(path)
    check(deadline)
    with threadpool_limits(limits=2):
        model = density_channel()
        started = time.time()
        engine = BoundedJointIntegrator(model, outer, particles, seed, deadline)
        values, points, diagnostics = engine.information()
        row = dict(outer=outer, particles=particles, scramble=seed,
                   ei_nats={str(k): v for k, v in values.items()}, diagnostics=diagnostics,
                   seconds=time.time()-started,
                   proposal='65/30/5 local, 2x-wide bounded Gaussian, uniform',
                   density_backend='MPS float32 forward / CPU float64 density, independently validated'
                       if isinstance(model, MPSDensityMLPChannel) else 'CPU float64')
        save(path, row)
        np.savez_compressed(path.with_suffix('.npz'), pointwise_ei=points)
        return row


def integrate(model, deadline):
    levels = []
    for outer, particles in LADDER:
        pending = [(outer, particles, seed, deadline) for seed in SEEDS
                   if not (OUT/f'ei_N{outer}_P{particles}_R{seed}.json').exists()]
        if pending:
            stage('full_joint_ei_batch', outer=outer, particles=particles,
                  pending_scrambles=[p[2] for p in pending], concurrent_jobs=2)
            # These are the same four planned numerical repeats, not extra trials.
            # Spawned children inherit the worker's process group, so the absolute
            # budget supervisor terminates the complete batch when required.
            with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = [pool.submit(one_integral, arguments) for arguments in pending]
                for future in as_completed(futures):
                    row = future.result()
                    stage('integral_saved', outer=outer, particles=particles, scramble=row['scramble'],
                          seconds=row['seconds'])
        rows = []
        for seed in SEEDS:
            check(deadline)
            path = OUT/f'ei_N{outer}_P{particles}_R{seed}.json'
            row = read(path)
            rows.append(row)
            # Save completed work immediately; an incomplete level is never accepted.
            save(OUT/'completed_integrations.json', dict(outer=outer, particles=particles,
                                                       completed=len(rows), expected=len(SEEDS)))
        array = np.array([[r['ei_nats'][str(i)] for i in range(1, 64)] for r in rows])
        mean, sem = array.mean(0), array.std(0, ddof=1)/np.sqrt(len(rows))
        result = spt_from_ei(dict(zip(range(1, 64), mean)))
        diagnostics = [d for r in rows for d in r['diagnostics'].values()]
        rep = [spt_from_ei(dict(zip(range(1, 64), a))) for a in array]
        scored = [recovery(r['tree'], TRUTH)['scored_clades'] if r['tree'] else None for r in rep]
        stable = all(s is not None for s in scored) and len({json.dumps(sorted(s)) for s in scored}) == 1
        change = float(abs(mean-np.array(levels[-1]['mean_ei_nats'])).max()) if levels else None
        result.update(outer=outer, particles=particles, replicate_count=len(rows),
                      mean_ei_nats=mean.tolist(), maximum_ei_sem_nats=float(sem.max()),
                      grid_change_nats=change, minimum_ess=min(d['ess_min'] for d in diagnostics),
                      maximum_weight=max(d['maximum_weight'] for d in diagnostics),
                      scramble_module_tree_stable=stable)
        if result['tree']:
            result['recovery'] = recovery(result['tree'], TRUTH)
            result['node_shares_percent'] = shares(result['tree'])
            rep_shares = [fixed_tree_shares(result['tree'], a) for a in array]
            keys = set(result['node_shares_percent'])
            result['maximum_share_sem_percentage_points'] = max(
                np.std([s[key] for s in rep_shares], ddof=1)/np.sqrt(len(rep_shares)) for key in keys)
            old = fixed_tree_shares(result['tree'], np.array(levels[-1]['mean_ei_nats'])) if levels else {}
            result['maximum_share_grid_change_percentage_points'] = max(
                abs(value-old[key]) for key, value in result['node_shares_percent'].items()) if old else None
            result['share_precision_basis'] = 'same mean-selected tree cuts evaluated in every replicate/grid; no absent-node-as-zero'
        result['numerically_resolved'] = bool(result['status']=='valid' and change is not None
            and change < .01 and sem.max() < .01 and result['minimum_ess'] >= 10
            and result['maximum_weight'] <= .25 and stable
            and result.get('maximum_share_sem_percentage_points', float('inf')) <= 1.
            and result.get('maximum_share_grid_change_percentage_points', float('inf')) <= 1.)
        levels.append(result)
        save(OUT/'spt_levels.json', levels)
        save(OUT/'spt.json', result)
        stage('integration_level_complete', outer=outer, particles=particles,
              resolved=result['numerically_resolved'], maximum_ei_sem=result['maximum_ei_sem_nats'],
              grid_change=change, minimum_ess=result['minimum_ess'], maximum_weight=result['maximum_weight'],
              recovery=result.get('recovery'))
        # An independently zero-mean control changes variance, not the target.
        # Keep the raw result alongside it and apply every original precision
        # gate plus the additional percentage gates to the controlled oracle.
        from scripts.summarize_spt_weak_coupling import run as summarize_completed
        controlled_levels = summarize_completed()
        if controlled_levels and controlled_levels[-1]['controlled']['numerically_resolved']:
            controlled = controlled_levels[-1]['controlled']
            save(OUT/'control_converged.json', dict(outer=outer, particles=particles,
                 resolved_epoch=time.time(), estimator=controlled['estimator'],
                 numerical_thresholds_unchanged=True))
            return controlled
        if result['numerically_resolved']:
            break
    return levels[-1]


def worker():
    deadline = read(OUT/'budget.json')['deadline_epoch']
    check(deadline)
    cfg = config()
    save(OUT/'configuration.json', cfg)
    save(OUT/'protocol.json', dict(fixed_cross_edges=[[1, 5], [4, 6]], weights=[.02, .02],
        within_weights_unchanged=True, source='iid uniform(-pi,pi)^6', target='complete unwrapped future phase',
        module_truth=TRUTH, no_search=True, physical_seed=0, split=[24576, 4096, 4096],
        numerical_seeds=SEEDS, integration_ladder=LADDER, tolerance_nats=.01,
        precision=dict(max_ei_sem_nats=.01, max_grid_change_nats=.01, minimum_ess=10, maximum_weight=.25,
                       max_share_sem_percentage_points=1., max_share_grid_change_percentage_points=1.),
        manuscript=dict(parent='P6UJCVG8', main='DXGC7JEA', si='MWIWKSVG',
                        sections='Methods5-12; SI S1.2/S5', version='no explicit manuscript version/date')))
    stage('generate_fixed_weak_bridge_data')
    x, y, manifest = dataset(cfg, 0, blocks=4)
    save(OUT/'data_manifest.json', manifest)
    if not (OUT/'model.json').exists():
        stage('fit_shared_data_model')
        model, fit = fit_mlp(x, y, 0, deadline, feature_kind='universal_fourier', split=(24576, 4096, 4096))
        save(OUT/'model.json', model.payload)
        save(OUT/'fit.json', fit)
        save(OUT/'diagnostics.json', channel_diagnostics(model, x, y, cfg, 0, split=(24576, 4096, 4096)))
    model = DoublePrecisionMLPChannel(read(OUT/'model.json'))
    probes = np.random.default_rng(4615).uniform(-np.pi, np.pi, (4096, 6))
    reference = MLPChannel(model.payload)
    error = float(abs(reference.mean(probes)-model.mean(probes)).max())
    save(OUT/'double_precision_equivalence.json', dict(maximum_mean_error_rad=error,
                                                      device='CPU', dtype='float64', passed=error < 1e-12))
    if error >= 1e-12:
        raise ArithmeticError('Accelerated double-precision channel differs from frozen model')
    if not (OUT/'baselines_raw.json').exists():
        save(OUT/'baselines_raw.json', baselines(x[:24576], y[:24576], deadline))
    if not read(OUT/'diagnostics.json')['fit_eligible']:
        raise ArithmeticError('New learned channel failed unchanged prediction/coverage gate')
    preflight(deadline)
    result = integrate(model, deadline)
    stage('complete', numerically_resolved=result['numerically_resolved'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        try:
            with threadpool_limits(limits=2):
                worker()
        except Exception as error:
            save(OUT/'failure.json', dict(error=str(error), traceback=traceback.format_exc()))
            stage('failed', error=str(error))
            raise
        return
    LOG.mkdir(parents=True, exist_ok=True)
    budget = read(OUT/'budget.json')
    remaining = budget['deadline_epoch'] - time.time()
    if remaining <= 0:
        raise TimeoutError('Approved budget exhausted; it cannot be reset')
    if (OUT/'exit.json').exists():
        (OUT/'exit.json').replace(OUT/'previous_worker_exit.json')
    with (LOG/'run.log').open('a') as logfile:
        process = subprocess.Popen([sys.executable, __file__, '--worker'], cwd=ROOT,
                                   stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        save(OUT/'process.json', dict(worker_pid=process.pid, deadline_epoch=budget['deadline_epoch'],
                                    log=str(LOG/'run.log')))
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
            stage('budget_exhausted')
        save(OUT/'exit.json', dict(code=code, finished_epoch=time.time(),
                                 elapsed_seconds=time.time()-budget['started_epoch']))
    if code:
        sys.exit(code)


if __name__ == '__main__':
    main()
