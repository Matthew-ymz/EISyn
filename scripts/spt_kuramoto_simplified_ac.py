"""Bounded A/C topology diagnostic; no tree or recovery ranking is certified.

Replace only the original >=4-node forces by pairwise bridges, inheriting each
replaced force's coefficient. Fit the same 502-candidate conditional TM from data.
All outputs use a separate cache from the ongoing B comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spt_local_hyperedge_kuramoto import LocalHyperedgeKuramoto, wrap
from scripts.spt_kuramoto_finite_tm import (
    DT, SYN_TOLERANCE, FullTargetIntegrator, PeriodicConditionalTM, check_time,
    digest, posterior_proposal, save_json, summarize_precision,
    test_diagnostics, transition_with_increments,
)
from scripts.spt_kuramoto_flow_tm import PeriodicFlowConditionalTM


REPLACEMENTS = {
    'A': {(1, 2, 3, 4, 5): (3, 4), (6, 7, 8, 9): (7, 8),
          tuple(range(1, 10)): (5, 6)},
    'C': {(1, 2, 3, 4, 5, 6): (3, 4), tuple(range(1, 10)): (6, 7)},
}
LAYOUTS = ((256, 64), (256, 256), (1024, 256))


class SimplifiedKuramoto(LocalHyperedgeKuramoto):
    def __init__(self, case):
        super().__init__(case)
        if case not in REPLACEMENTS:
            raise ValueError('This diagnostic changes A/C only')
        replacements = REPLACEMENTS[case]
        self.weights = {replacements.get(edge, edge): weight
                        for edge, weight in self.weights.items()}
        self.levels = {replacements.get(edge, edge): level
                       for edge, level in self.levels.items()}
        self.supports = tuple(self.weights)


def queries(case):
    left = 31 if case == 'A' else 63
    return sorted({511, left, 511 ^ left, 1, 2, 3, 4, 7})


def step_check(model):
    rng = np.random.default_rng(20261015)
    source = model.sample_source(512, rng)
    noise = np.sqrt(.00125) * rng.normal(size=(160, 512, 9))
    states = [transition_with_increments(model, source,
              noise.reshape(160 // block, block, 512, 9).sum(1), .00125 * block)
              for block in (8, 4, 2, 1)]
    differences = [wrap(a - b) for a, b in zip(states, states[1:])]
    rms = [float(np.sqrt(np.mean(d ** 2))) for d in differences]
    return {'dt': [.01, .005, .0025, .00125], 'samples': 512,
            'paired_brownian': True, 'rms_difference_rad': rms,
            'selected_dt': DT, 'passed': rms[-1] < 1e-4}


def syn_audit(rows, case):
    left = 31 if case == 'A' else 63
    splits = ((511, left, 511 ^ left), (3, 1, 2), (7, 3, 4))
    values = np.array([[r['ei_nats'][str(a)] - r['ei_nats'][str(l)]
                        - r['ei_nats'][str(b)] for a, l, b in splits] for r in rows])
    result = {'splits': [list(s) for s in splits], 'units': 'nats',
              'tolerance_nats': SYN_TOLERANCE, 'raw_syn_nats': values.tolist(),
              'minimum_raw_syn_nats': float(values.min()),
              'tolerance_negative_count': int(np.count_nonzero(
                  (values < 0) & (values >= -SYN_TOLERANCE))),
              'significant_violation_count': int(np.count_nonzero(values < -SYN_TOLERANCE))}
    if len(rows) > 1:
        result.update(mean_syn_nats=values.mean(0).tolist(),
                      syn_sem_nats=(values.std(0, ddof=1) / np.sqrt(len(rows))).tolist())
    return result


def summarize(record):
    rows = record.get('precision', [])
    if not rows:
        return
    case = record['case']
    summaries = summarize_precision(rows, queries(case), case)
    record['precision_summary'] = summaries
    record['critical_syn_audits'] = []
    for outer, particles in LAYOUTS:
        matching = [r for r in rows if (r['outer'], r['particles']) == (outer, particles)]
        if matching:
            record['critical_syn_audits'].append(
                {'outer': outer, 'particles': particles, 'replicates': len(matching),
                 **syn_audit(matching, case)})
    record['full_precision_certified'] = False
    record['hierarchy_status'] = 'not evaluated: representative queries only; no 4096-sample gate'
    if len(summaries) != 3 or any(r['replicates'] != 4 for r in summaries):
        return
    changes = [max(abs(a['mean_ei_nats'][str(m)] - b['mean_ei_nats'][str(m)])
                   for m in queries(case)) for a, b in zip(summaries, summaries[1:])]
    high = [r for r in rows if (r['outer'], r['particles']) == LAYOUTS[-1]]
    diagnostics = [d for r in high for d in
                   [r['diagnostics']['denominator'], *r['diagnostics']['queries'].values()]
                   if d['ess_minimum'] is not None]
    proposal = {'minimum_ess': min(d['ess_minimum'] for d in diagnostics),
                'maximum_normalized_weight': max(d['maximum_normalized_weight'] for d in diagnostics),
                'maximum_remaining_newton_step_rad': max(d['max_remaining_step_rad'] for d in diagnostics)}
    audits = record['critical_syn_audits']
    syn_changes = [float(np.max(abs(np.array(a['mean_syn_nats']) - b['mean_syn_nats'])))
                   for a, b in zip(audits, audits[1:])]
    passed = (max(changes) < .01 and max(syn_changes) < .01
              and summaries[-1]['maximum_ei_sem_nats'] < .01
              and max(audits[-1]['syn_sem_nats']) < .01
              and proposal['minimum_ess'] >= 10
              and proposal['maximum_normalized_weight'] < .25
              and proposal['maximum_remaining_newton_step_rad'] < 1e-3
              and sum(a['significant_violation_count'] for a in audits) == 0)
    record.update(precision_changes_nats=changes, syn_changes_nats=syn_changes,
                  proposal_audit=proposal, lightweight_checks_passed=passed)


def run(output, budget_seconds=300.):
    started = time.monotonic()
    deadline = started + budget_seconds
    original_path = ROOT / 'results/spt_local_kuramoto_tm/flow_mixture_dt0025.json'
    original = json.loads(original_path.read_text())
    payload = {'protocol': {
        'date': '2026-10-15', 'seed': 0, 'cases': ['A', 'C'],
        'train_samples': 16384, 'validation_samples': 4096, 'test_samples': 4096,
        'source': 'iid Uniform(-pi,pi)^9', 'target': 'complete future nine phases',
        'tau': .2, 'dt': DT, 'coupling': 1., 'decay': .5, 'noise': .08,
        'layouts': LAYOUTS, 'scrambles': 4, 'budget_seconds': budget_seconds,
        'syn_tolerance_nats': SYN_TOLERANCE, 'clipping': False,
        'change': 'replace >=4-node forces by sparse pairs; inherit each original coefficient',
        'limitation': 'order, support, and resulting force magnitude change jointly; exploratory diagnostic',
        'fit': 'same 502-candidate data-only flow TM; no planted edges supplied',
        'scope': '8 representative EIs, 3 bipartitions; no full-table or six-method evaluation',
        'manuscript': {'parent': 'P6UJCVG8', 'main': 'DXGC7JEA', 'si': 'MWIWKSVG',
            'version': 'sole main/SI attachment group rechecked 2026-10-15; no explicit manuscript date/version',
            'locations': 'Methods (5)-(12), SI S1.2 (S7)-(S15), S5.1 (S49)-(S50), S7 (S67)-(S71)'},
        'original_cache_sha256': hashlib.sha256(original_path.read_bytes()).hexdigest(),
    }, 'records': [], 'status': 'running'}
    if output.exists():
        raise ValueError('Separate output required; existing results are never overwritten')
    def save():
        payload['elapsed_compute_seconds'] = time.monotonic() - started
        save_json(output, payload)
    save()
    record = None
    try:
        # Fit both cases before integrating: a bounded run covers both topologies.
        for case in ('A', 'C'):
            check_time(deadline)
            model = SimplifiedKuramoto(case)
            reference = next(r for r in original['records'] if r['case'] == case)
            record = {'case': case, 'status': 'fitting', 'precision': [],
                      'network': [{'edge': list(e), 'weight': w} for e, w in model.weights.items()],
                      'replacements': [{'old': list(e), 'new': list(r),
                          'weight': LocalHyperedgeKuramoto(case).weights[e]}
                          for e, r in REPLACEMENTS[case].items()],
                      'step_precision': step_check(model)}
            payload['records'].append(record)
            save()
            if not record['step_precision']['passed']:
                record['status'] = 'step_precision_failed'
                continue
            streams = np.random.SeedSequence([20261015, 0]).spawn(8)
            source = model.sample_source(20480, np.random.default_rng(streams[0]))
            target = model.step(source, np.random.default_rng(streams[1]), dt=DT)
            tx = model.sample_source(4096, np.random.default_rng(streams[2]))
            ty = model.step(tx, np.random.default_rng(streams[3]), dt=DT)
            rx = model.sample_source(256, np.random.default_rng(streams[4]))
            ry = np.array([model.step(rx, np.random.default_rng(
                np.random.SeedSequence([20261015, 0, 500 + i])), dt=DT) for i in range(16)])
            # Reproduce the original saved input/output digest, establishing actual
            # sample/noise pairing rather than relying on identical seed labels.
            old_target = LocalHyperedgeKuramoto(case).step(
                source, np.random.default_rng(streams[1]), dt=DT)
            paired = (digest(source[:16384], old_target[:16384]) == reference['training_digest']
                      and digest(source[16384:], old_target[16384:]) == reference['validation_digest'])
            if not paired:
                raise ValueError('Original data digest mismatch: paired comparison invalid')
            fitted, fit = PeriodicFlowConditionalTM.fit(source[:16384], target[:16384],
                source[16384:], target[16384:], deadline=deadline)
            record.update(model=fitted.payload(), fit=fit,
                          original_data_pairing_verified=paired,
                          training_digest=digest(source[:16384], target[:16384]),
                          validation_digest=digest(source[16384:], target[16384:]),
                          heldout=test_diagnostics(fitted, tx, ty, rx, ry))
            cx = np.random.default_rng(20261016).uniform(-np.pi, np.pi, (128, 9))
            cy = fitted.sample(cx, np.random.default_rng(20261017).uniform(size=(128, 9)))
            a = posterior_proposal(fitted, cx, cy, np.arange(9), np.arange(9), iterations=16)
            b = posterior_proposal(fitted, cx, cy, np.arange(9), np.arange(9),
                                   iterations=16, initial_offset=.15)
            _, jac = fitted.mean(cx, jacobian=True)
            record['inverse_check'] = {
                'max_center_difference_rad': float(np.max(abs(wrap(a[0] - b[0])))),
                'jacobian_nonpositive_determinant_count': int(np.count_nonzero(np.linalg.det(jac) <= 0)),
                'jacobian_minimum_singular_value': float(np.linalg.svd(jac, compute_uv=False).min())}
            ok = (record['inverse_check']['max_center_difference_rad'] < 1e-4
                  and record['inverse_check']['jacobian_nonpositive_determinant_count'] == 0)
            record['status'] = 'fitted' if ok else 'inverse_proposal_failed'
            save()
            print(json.dumps({'stage': 'fit_complete', 'case': case,
                              'active_edges': fit['active_candidate_edges']}), flush=True)
        for outer, particles in LAYOUTS:
            for record in payload['records']:
                if record['status'] not in ('fitted', 'precision_running'):
                    continue
                fitted = PeriodicConditionalTM.load(record['model'])
                for scramble in range(4):
                    integral = FullTargetIntegrator(fitted, outer=outer, max_particles=256,
                        seed=20261015 + 1000 * scramble, deadline=deadline)
                    values, _, diagnostics = integral.information(queries(record['case']), particles)
                    row = {'outer': outer, 'particles': particles, 'scramble': scramble,
                           'ei_nats': {str(k): v for k, v in values.items()}, 'diagnostics': diagnostics}
                    record['precision'].append(row)
                    audit = syn_audit([row], record['case'])
                    if audit['significant_violation_count']:
                        record.update(status='nonnegativity_failed', failure_audit=audit)
                        save()
                        raise ArithmeticError(f"Syn violation: minimum={audit['minimum_raw_syn_nats']}; "
                            f"threshold={-SYN_TOLERANCE}; count={audit['significant_violation_count']}")
                    record['status'] = 'precision_running'
                    save()
                print(json.dumps({'stage': 'precision', 'case': record['case'],
                                  'outer': outer, 'particles': particles}), flush=True)
        for record in payload['records']:
            summarize(record)
            if record['status'] == 'precision_running':
                record['status'] = 'lightweight_complete'
        payload['status'] = 'complete'
    except (TimeoutError, ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
        payload['status'] = 'budget_exhausted' if isinstance(error, TimeoutError) else 'numerical_failure'
        payload['error'] = str(error)
        if record is not None:
            record['interrupted_status'] = record['status']
        for record in payload['records']:
            summarize(record)
    save()
    return payload


def refine_roots(output, budget_seconds=300.):
    """Use only the unspent diagnostic budget for 4096-sample root queries."""
    payload = json.loads(output.read_text())
    previous = payload['elapsed_compute_seconds']
    started = time.monotonic()
    deadline = started + budget_seconds - previous
    payload['protocol']['root_refinement'] = {'outer': 4096, 'particles': 256,
        'scrambles': 4, 'scope': 'full source and two fixed root children only'}
    payload['status'] = 'root_refinement_running'
    def save():
        payload['elapsed_compute_seconds'] = previous + time.monotonic() - started
        save_json(output, payload)
    def summarize_roots():
        for record in payload['records']:
            rows = record.get('root_precision', [])
            if len(rows) != 4:
                continue
            left = 31 if record['case'] == 'A' else 63
            masks = (511, left, 511 ^ left)
            values = np.array([[r['ei_nats'][str(m)] for m in masks] for r in rows])
            coarse = np.array([[r['ei_nats'][str(m)] for m in masks]
                for r in record['precision'] if (r['outer'], r['particles']) == (1024, 256)])
            syn = values[:, 0] - values[:, 1] - values[:, 2]
            ds = [d for r in rows for d in [r['diagnostics']['denominator'],
                *r['diagnostics']['queries'].values()] if d['ess_minimum'] is not None]
            record['root_refinement_summary'] = {
                'outer': 4096, 'particles': 256, 'scrambles': 4,
                'root_syn_inner_change_nats': float(abs(
                    record['critical_syn_audits'][0]['mean_syn_nats'][0]
                    - record['critical_syn_audits'][1]['mean_syn_nats'][0])),
                'maximum_ei_change_from_1024_nats': float(np.max(abs(values.mean(0) - coarse.mean(0)))),
                'maximum_ei_sem_nats': float(np.max(values.std(0, ddof=1) / 2)),
                'root_syn_mean_nats': float(syn.mean()), 'root_syn_sem_nats': float(syn.std(ddof=1) / 2),
                'root_syn_change_from_1024_nats': float(abs(syn.mean() -
                    (coarse[:, 0] - coarse[:, 1] - coarse[:, 2]).mean())),
                'minimum_raw_syn_nats': float(syn.min()),
                'tolerance_negative_count': int(np.count_nonzero((syn < 0) & (syn >= -SYN_TOLERANCE))),
                'significant_violation_count': int(np.count_nonzero(syn < -SYN_TOLERANCE)),
                'minimum_ess': min(d['ess_minimum'] for d in ds),
                'maximum_normalized_weight': max(d['maximum_normalized_weight'] for d in ds),
                'maximum_remaining_newton_step_rad': max(d['max_remaining_step_rad'] for d in ds)}
            diagnostic = record['root_refinement_summary']
            record['root_precision_passed'] = (
                diagnostic['maximum_ei_change_from_1024_nats'] < .01
                and diagnostic['maximum_ei_sem_nats'] < .01
                and diagnostic['root_syn_sem_nats'] < .01
                and diagnostic['root_syn_change_from_1024_nats'] < .01
                and diagnostic['root_syn_inner_change_nats'] < .01
                and diagnostic['significant_violation_count'] == 0
                and diagnostic['minimum_ess'] >= 10
                and diagnostic['maximum_normalized_weight'] < .25
                and diagnostic['maximum_remaining_newton_step_rad'] < 1e-3
                and record['precision_changes_nats'][0] < .01
                and record['step_precision']['passed']
                and record['inverse_check']['max_center_difference_rad'] < 1e-4
                and record['inverse_check']['jacobian_nonpositive_determinant_count'] == 0)
    save()
    try:
        for scramble in range(4):
            for record in payload['records']:
                if not record.get('model') or len(record.get('precision', [])) != 12:
                    continue
                check_time(deadline)
                rows = record.setdefault('root_precision', [])
                if any(r['scramble'] == scramble for r in rows):
                    continue
                left = 31 if record['case'] == 'A' else 63
                model = PeriodicConditionalTM.load(record['model'])
                integral = FullTargetIntegrator(model, outer=4096, max_particles=256,
                    seed=20261015 + 1000 * scramble, deadline=deadline)
                values, _, diagnostics = integral.information((511, left, 511 ^ left), 256)
                row = {'outer': 4096, 'particles': 256, 'scramble': scramble,
                    'ei_nats': {str(k): v for k, v in values.items()}, 'diagnostics': diagnostics}
                rows.append(row)
                syn = values[511] - values[left] - values[511 ^ left]
                if syn < -SYN_TOLERANCE:
                    payload['failure_audit'] = {'minimum_raw_syn_nats': syn,
                        'threshold_nats': -SYN_TOLERANCE, 'significant_violation_count': 1}
                    save()
                    raise ArithmeticError(f'Syn violation: minimum={syn}; threshold={-SYN_TOLERANCE}; count=1')
                save()
                print(json.dumps({'stage': 'root_refinement', 'case': record['case'],
                                  'scramble': scramble, 'raw_syn_nats': syn}), flush=True)
        payload['status'] = 'complete'
    except (TimeoutError, ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
        payload['status'] = 'budget_exhausted' if isinstance(error, TimeoutError) else 'numerical_failure'
        payload['error'] = str(error)
    summarize_roots()
    save()
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        default=ROOT / 'results/spt_local_kuramoto_tm/simplified_ac_light.json')
    parser.add_argument('--budget-seconds', type=float, default=300.)
    parser.add_argument('--refine-roots', action='store_true')
    args = parser.parse_args()
    result = (refine_roots if args.refine_roots else run)(args.output, args.budget_seconds)
    print(json.dumps({'status': result['status'], 'seconds': result['elapsed_compute_seconds']}))
