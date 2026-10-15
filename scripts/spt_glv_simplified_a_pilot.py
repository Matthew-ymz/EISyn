"""Approved 15-minute feasibility pilot: HO-gLV on frozen simplified A.

The launcher waits for neither experiment: it refuses to launch until both
original workflows have ended. Its child is bounded by one persisted deadline.
No complete EI table, tree, F1, or baseline comparison is claimed by this pilot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spt_local_hyperedge_glv import LocalHyperedgeGLV, abundance, product_moments
from scripts.spt_kuramoto_simplified_ac import SimplifiedKuramoto
from scripts.spt_glv_finite_tm import (
    RealConditionalTM, RealFullTargetIntegrator, posterior_proposal,
    test_diagnostics, transition_with_increments,
)
from scripts.spt_kuramoto_finite_tm import (
    check_time, digest, pilot_masks, save_json, summarize_precision,
)

FROZEN = ROOT / 'results/spt_local_kuramoto_tm/ac_comparison/A/frozen_model.json'
OUTPUT = ROOT / 'results/spt_matched_three_case/glv_simplified_A/pilot.json'
LAYOUTS = ((256, 64), (256, 256), (1024, 256), (4096, 256))
TOLERANCE = .01


class MatchedSimplifiedGLV(LocalHyperedgeGLV):
    """Same physical hyperedges and coordinate-RMS weights as frozen Kuramoto A."""

    def __init__(self, weights):
        super().__init__('A')
        expected = SimplifiedKuramoto('A')
        weights = {tuple(edge): float(weight) for edge, weight in weights.items()}
        if weights != expected.weights:
            raise ValueError('Expected exact simplified-A members and inherited weights')
        self.supports = tuple(weights)
        self.weights = weights
        self.levels = dict(expected.levels)
        self.case = 'A_simplified'
        self.growth = np.full(9, self.self_limit * np.sinh(1.)) + self.frequencies / np.pi
        self.coefficients = {}
        for edge, weight in self.weights.items():
            mean, variance = product_moments(len(edge))
            coefficient = -weight / (np.pi * np.sqrt(2 * variance))
            self.coefficients[edge] = float(coefficient)
            self.growth[np.asarray(edge) - 1] -= coefficient * mean


def read(path):
    return json.loads(Path(path).read_text())


def original_workflows_ended():
    glv = read(ROOT / 'results/spt_local_glv_tm/a_comparison/comparison_1024.json')
    coordinator = read(ROOT / 'results/spt_local_kuramoto_tm/ac_comparison/natural_stage.json')
    if glv.get('status') in (None, 'running') or str(glv.get('status')).startswith('running'):
        raise RuntimeError('Original HO-gLV computation has not ended')
    if coordinator.get('status') not in ('all_local_jobs_ended', 'orchestration_failed'):
        raise RuntimeError('Original Kuramoto workflow has not ended')
    names = {'spt_glv_a_comparison.py', 'run_spt_kuramoto_natural_stage.py',
             'spt_kuramoto_ac_selected.py', 'spt_kuramoto_ac_surd.py',
             'spt_kuramoto_ac_block.py', 'refine_spt_kuramoto_ac_block.py',
             'refine_spt_kuramoto_ac_surd.py', 'spt_kuramoto_natural_baselines.py'}
    active = []
    for line in subprocess.check_output(['ps', '-axo', 'pid=,args='], text=True).splitlines():
        try:
            pid, command = line.strip().split(maxsplit=1)
            argv = shlex.split(command)
        except ValueError:
            continue
        if any(Path(argument).name in names for argument in argv):
            active.append((pid, command))
    if active:
        raise RuntimeError(f'Original workflow processes still active: {active}')
    return {'glv_status': glv['status'], 'kuramoto_status': coordinator['status']}


def syn_audit(estimates):
    """Audit every bipartition whose three EI queries are actually available."""
    estimates = {int(k): float(v) for k, v in estimates.items()}
    if not np.isfinite(list(estimates.values())).all():
        raise ArithmeticError('Nonfinite EI; no Syn audit or hierarchy is valid')
    values = []
    for parent in sorted(estimates):
        anchor = parent & -parent
        left = (parent - 1) & parent
        while left:
            right = parent ^ left
            if left & anchor and left in estimates and right in estimates:
                values.append(estimates[parent] - estimates[left] - estimates[right])
            left = (left - 1) & parent
    return {'tolerance_nats': TOLERANCE, 'clipping': False, 'split_count': len(values),
            'minimum_syn_nats': min(values) if values else None,
            'tolerance_negative_count': sum(-TOLERANCE <= v < 0 for v in values),
            'significant_violation_count': sum(v < -TOLERANCE for v in values),
            'passed': bool(values) and all(v >= -TOLERANCE for v in values)}


def require_valid_syn(audit):
    if audit['significant_violation_count']:
        raise ArithmeticError(f"Syn nonnegativity violation: minimum={audit['minimum_syn_nats']} "
                              f"nats; threshold={-TOLERANCE}; affected={audit['significant_violation_count']}")


def step_check(model):
    rng = np.random.default_rng(20261015)
    source = model.sample_source(512, rng)
    noise = np.sqrt(.00125) * rng.normal(size=(160, 512, 9))
    states = [transition_with_increments(model, source,
              noise.reshape(160 // block, block, 512, 9).sum(1), .00125 * block)
              for block in (8, 4, 2, 1)]
    rms = [float(np.sqrt(np.mean((a - b) ** 2))) for a, b in zip(states, states[1:])]
    return {'paired_brownian': True, 'samples': 512, 'dt': [.01, .005, .0025, .00125],
            'rms_difference_coordinate': rms, 'passed': rms[-1] < 1e-4}


def precision_result(rows, masks):
    summary = summarize_precision(rows, masks, 'A') if rows else []
    complete = {tuple(layout): [r for r in rows if (r['outer'], r['particles']) == layout]
                for layout in LAYOUTS}
    if any(len(group) != 4 for group in complete.values()):
        return {'summary': summary, 'passed': False, 'reason': 'Incomplete registered pilot grid'}
    means = [np.mean([[r['ei_nats'][str(m)] for m in masks] for r in complete[layout]], axis=0)
             for layout in LAYOUTS]
    changes = [float(np.max(abs(a - b))) for a, b in zip(means, means[1:])]
    diagnostic = [v for r in complete[LAYOUTS[-1]] for v in
                  [r['diagnostics']['denominator'], *r['diagnostics']['queries'].values()]
                  if v.get('ess_minimum') is not None]
    audit = {'minimum_ess': min(v['ess_minimum'] for v in diagnostic),
             'maximum_normalized_weight': max(v['maximum_normalized_weight'] for v in diagnostic),
             'maximum_projected_step_coordinate': max(v['max_projected_remaining_step_coordinate']
                                                       for v in diagnostic)}
    passed = (changes[0] < .01 and changes[-1] < .01 and summary[-1]['maximum_ei_sem_nats'] < .01
              and audit['minimum_ess'] >= 10 and audit['maximum_normalized_weight'] < .25
              and audit['maximum_projected_step_coordinate'] < 1e-3)
    return {'summary': summary, 'changes_nats': changes, 'proposal_audit': audit, 'passed': passed}


def worker(output):
    ledger = read(output.with_name('budget.json'))
    started = time.monotonic()
    deadline = started + ledger['deadline_epoch'] - time.time()
    reference = read(FROZEN)
    weights = {tuple(row['edge']): row['weight'] for row in reference['network']}
    generator = MatchedSimplifiedGLV(weights)
    payload = {'status': 'running', 'stage': 'step_precision', 'process_id': os.getpid(),
               'protocol': {'seed': 0, 'train': 16384, 'validation': 4096, 'test': 4096,
                            'tau': .2, 'dt': .0025, 'noise': .08, 'self_limit': 1.,
                            'source': 'iid Uniform(-pi,pi)^9 in u=pi log abundance',
                            'target': 'complete future nine log-abundance coordinates',
                            'units': 'nats', 'syn_tolerance_nats': TOLERANCE, 'clipping': False,
                            'layouts': LAYOUTS, 'scrambles': 4, 'budget': ledger,
                            'frozen_kuramoto_sha256': hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
                            'manuscript': {'parent': 'P6UJCVG8', 'main': 'DXGC7JEA', 'si': 'MWIWKSVG',
                                           'checked': '2026-10-15', 'version': 'no explicit date/version'},
                            'scope': 'feasibility only; no complete table, tree, F1, or baselines'},
               'generator': generator.metadata(), 'precision': []}

    def save():
        payload['elapsed_wall_seconds'] = time.time() - ledger['started_epoch']
        save_json(output, payload)

    save()
    try:
        payload['step_precision'] = step_check(generator)
        if not payload['step_precision']['passed']:
            payload['status'] = 'step_precision_failed'
            save()
            return
        check_time(deadline)
        payload['stage'] = 'paired_data_and_fit'
        save()
        streams = np.random.SeedSequence([20261015, 0]).spawn(8)
        source = generator.sample_source(20480, np.random.default_rng(streams[0]))
        paired_target = SimplifiedKuramoto('A').step(source, np.random.default_rng(streams[1]), dt=.0025)
        paired = (digest(source[:16384], paired_target[:16384]) == reference['training_digest']
                  and digest(source[16384:], paired_target[16384:]) == reference['validation_digest'])
        if not paired:
            raise ValueError('Frozen Kuramoto training/validation digest mismatch')
        payload['pairing'] = {'frozen_training_validation_verified': True, 'source_digest': digest(source)}
        target = generator.step(source, np.random.default_rng(streams[1]), dt=.0025)
        tx = generator.sample_source(4096, np.random.default_rng(streams[2]))
        ty = generator.step(tx, np.random.default_rng(streams[3]), dt=.0025)
        rx = generator.sample_source(256, np.random.default_rng(streams[4]))
        ry = np.array([generator.step(rx, np.random.default_rng(
            np.random.SeedSequence([20261015, 0, 500 + i])), dt=.0025) for i in range(16)])
        payload.update(training_digest=digest(source[:16384], target[:16384]),
                       validation_digest=digest(source[16384:], target[16384:]),
                       minimum_observed_abundance=float(min(abundance(a).min() for a in (source, target, tx, ty, rx, ry.reshape(-1, 9)))))
        save()
        fitted, history = RealConditionalTM.fit(source[:16384], target[:16384],
                                              source[16384:], target[16384:], deadline=deadline)
        payload.update(model=fitted.payload(), fit=history, heldout=test_diagnostics(fitted, tx, ty, rx, ry), stage='inverse_check')
        save()
        check_time(deadline)
        cx = np.random.default_rng(20261016).uniform(-np.pi, np.pi, (128, 9))
        cy = fitted.sample(cx, np.random.default_rng(20261017).uniform(size=(128, 9)))
        a = posterior_proposal(fitted, cx, cy, np.arange(9), np.arange(9), iterations=16)
        b = posterior_proposal(fitted, cx, cy, np.arange(9), np.arange(9), iterations=16, initial_offset=.15)
        _, jac = fitted.mean(cx, jacobian=True)
        payload['inverse_check'] = {'maximum_center_difference_coordinate': float(np.max(abs(a[0] - b[0]))),
                                    'nonpositive_jacobian_count': int(np.count_nonzero(np.linalg.det(jac) <= 0))}
        if payload['inverse_check']['maximum_center_difference_coordinate'] >= 1e-4 or payload['inverse_check']['nonpositive_jacobian_count']:
            payload['status'] = 'inverse_proposal_failed'
            save()
            return
        masks = pilot_masks('A')
        payload.update(stage='representative_ei', masks=masks)
        save()
        for outer, particles in LAYOUTS:
            for scramble in range(4):
                check_time(deadline)
                t = time.monotonic()
                integrator = RealFullTargetIntegrator(fitted, outer=outer, max_particles=256,
                    seed=20261015 + 1000 * scramble, deadline=deadline)
                estimates, _, diagnostic = integrator.information(masks, particles)
                audit = syn_audit(estimates)
                payload['precision'].append({'outer': outer, 'particles': particles, 'scramble': scramble,
                    'ei_nats': {str(k): v for k, v in estimates.items()}, 'diagnostics': diagnostic,
                    'syn_audit': audit, 'elapsed_seconds': time.monotonic() - t})
                save()
                require_valid_syn(audit)
                print(json.dumps({'stage': 'representative_ei', 'outer': outer, 'particles': particles,
                                  'scramble': scramble, 'syn_audit': audit}), flush=True)
        payload['precision_result'] = precision_result(payload['precision'], masks)
        payload['status'] = 'feasibility_passed' if payload['precision_result']['passed'] else 'precision_failed'
        payload['stage'] = 'ended'
        save()
    except TimeoutError as error:
        payload.update(status='budget_exhausted', error=str(error))
        save()
    except Exception as error:
        payload.update(status='failed', error=f'{type(error).__name__}: {error}')
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--budget-seconds', type=float, default=900.)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        worker(args.output)
        return
    if not 0 < args.budget_seconds <= 900:
        raise ValueError('Approval covers at most 900 seconds')
    ended = original_workflows_ended()
    budget = args.output.with_name('budget.json')
    if args.output.exists() or budget.exists():
        raise ValueError('Existing trial budget/results cannot be reset or overwritten')
    budget.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ledger = {'started_epoch': started, 'deadline_epoch': started + args.budget_seconds,
              'budget_seconds': args.budget_seconds, 'original_workflows': ended,
              'code_sha256': {name: hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest()
                              for name in ('spt_glv_simplified_a_pilot.py', 'spt_local_hyperedge_glv.py',
                                           'spt_kuramoto_simplified_ac.py', 'spt_glv_finite_tm.py',
                                           'spt_glv_native.py', 'spt_glv_mean_kernel.cpp',
                                           'spt_kuramoto_finite_tm.py')},
              'authorization': 'User approved 15-minute feasibility trial in supervision chat, 2026-10-15'}
    # A goal continuation and a heartbeat may arrive together. Exclusive creation
    # claims this one approved run before either can launch a child.
    with budget.open('x') as stream:
        json.dump(ledger, stream, indent=2, allow_nan=False)
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'VECLIB_MAXIMUM_THREADS': '1'}
    with args.output.with_suffix('.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                  '--worker', '--output', str(args.output)], cwd=ROOT,
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
        ledger['child_pid'] = child.pid
        save_json(budget, ledger)
        try:
            child.wait(timeout=max(.001, ledger['deadline_epoch'] - time.time()))
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            payload = read(args.output) if args.output.exists() else {}
            payload.update(status='budget_exhausted', error='Approved hard 900-second-or-shorter deadline reached',
                           elapsed_wall_seconds=time.time() - started)
            save_json(args.output, payload)
    ledger.update(exit_code=child.returncode, finished_epoch=time.time())
    save_json(budget, ledger)
    result = read(args.output)
    print(json.dumps({'status': result['status'], 'elapsed_wall_seconds': result.get('elapsed_wall_seconds'),
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
