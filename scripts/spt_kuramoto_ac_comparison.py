"""Resumable six-method comparison of the frozen simplified A/C joint models.

The existing B evaluators are reused through process-local case/score adapters;
their source files, B caches, dynamics, and estimator parameters are untouched.
One worker process handles exactly one case. A joint six-hour wall-clock limit
is shared by both case workers, including all refinements.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spt_kuramoto_finite_tm import (
    FullTargetIntegrator, PeriodicConditionalTM, SYN_TOLERANCE,
    masks_to_table, save_json, check_time,
)
from scripts.spt_continuous_hierarchy_pilot import audit_table
from scripts.spt_continuous_phase_benchmark import score
from scripts.spt_local_hyperedge_kuramoto import SUPPORTS

FOLDER = ROOT / 'results/spt_local_kuramoto_tm/ac_comparison'
PILOT = ROOT / 'results/spt_local_kuramoto_tm/simplified_ac_light.json'
SHARED = ('spt_kuramoto_finite_tm.py', 'spt_kuramoto_flow_tm.py',
          'spt_kuramoto_group_tm.py', 'spt_kuramoto_b_group_comparison.py',
          'spt_kuramoto_b_future_mi.py', 'analyze_spt_kuramoto_b.py',
          'validate_spt_kuramoto_b_full.py', 'validate_spt_kuramoto_b_groups.py',
          'spt_kuramoto_ac_comparison.py', 'spt.py',
          'spt_continuous_hierarchy_pilot.py', 'spt_continuous_phase_benchmark.py',
          'spt_multilevel_parity_case.py', 'spt_heterogeneous_hierarchy_case.py')
_ACTIVE_CASE = None


def read(path):
    return json.loads(Path(path).read_text())


def shared_hashes():
    return {name: hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest()
            for name in SHARED}


def initialize(hours=6.):
    path = FOLDER / 'budget.json'
    if path.exists():
        previous = read(path)
        if previous['shared_source_sha256'] != shared_hashes():
            if any((FOLDER / case / 'run.json').exists() for case in ('A', 'C')):
                raise ValueError('Cannot silently change evaluator fingerprints after execution begins')
            previous['shared_source_sha256'] = shared_hashes()
            save_json(path, previous)
        return previous
    now = time.time()
    payload = {'start_epoch_seconds': now, 'deadline_epoch_seconds': now + hours * 3600,
               'wall_clock_budget_hours': hours, 'cases': ['A', 'C'],
               'pilot_sha256': hashlib.sha256(PILOT.read_bytes()).hexdigest(),
               'shared_source_sha256': shared_hashes(),
               'physical_datasets_per_case': 1,
               'scoring': 'predeclared retained module hierarchy; not overlapping bridge-edge recovery',
               'source': 'Uniform(-pi,pi)^9', 'target': 'future phases, same frozen joint',
               'syn_tolerance_nats': SYN_TOLERANCE, 'clipping': False}
    save_json(path, payload)
    return payload


class Case:
    def __init__(self, case):
        if case not in ('A', 'C'):
            raise ValueError('A/C only; the B experiment is separate')
        self.case = case
        self.folder = FOLDER / case
        self.budget = read(FOLDER / 'budget.json')
        if self.budget['pilot_sha256'] != hashlib.sha256(PILOT.read_bytes()).hexdigest():
            raise ValueError('Frozen fitted-model cache changed')
        if self.budget['shared_source_sha256'] != shared_hashes():
            raise ValueError('Shared evaluator changed since run initialization; review before reusing results')
        self.pilot = next(r for r in read(PILOT)['records'] if r['case'] == case)
        if not self.pilot.get('root_precision_passed'):
            raise ValueError('Simplified root pilot must pass first')
        self.digest = hashlib.sha256(json.dumps(self.pilot['model'], sort_keys=True).encode()).hexdigest()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.pilot_path = self.folder / 'frozen_model.json'
        # The existing refinement evaluator reads pilot['precision']. Provide
        # one compatible list, retaining actual layouts and matching streams.
        self.compatible_pilot = {**self.pilot, 'precision': [*self.pilot['precision'],
                                                        *self.pilot['root_precision']]}
        if not self.pilot_path.exists():
            save_json(self.pilot_path, self.compatible_pilot)

    def deadline(self, folder=None):
        return time.monotonic() + self.budget['deadline_epoch_seconds'] - time.time()

    def load_model(self):
        return PeriodicConditionalTM.load(self.pilot['model']), self.compatible_pilot, self.digest

    def scored(self, root, clades):
        # These are fixed nested *modules*, inherited from the pre-simplification
        # design. They are not the new, overlapping physical edge collection.
        return score({'status': 'exploratory_numerical', 'root_split': root,
                      'clades': clades}, SUPPORTS[self.case])

    def configure_evaluators(self):
        """Inject the case in this worker only; never edit shared B source."""
        global _ACTIVE_CASE
        if _ACTIVE_CASE is not None and _ACTIVE_CASE != self.case:
            raise ValueError('A/C adapters require separate worker processes')
        _ACTIVE_CASE = self.case
        base = importlib.import_module('scripts.spt_kuramoto_b_comparison')
        base.load_model = self.load_model
        base.budget_deadline = self.deadline
        base.FOLDER = self.folder
        base.PILOT = self.pilot_path
        groups = importlib.import_module('scripts.spt_kuramoto_b_group_comparison')
        groups.load_model = self.load_model
        groups.budget_deadline = self.deadline
        groups.scored = self.scored
        groups.FOLDER = self.folder
        original = groups.Comparison
        context = self
        class CaseComparison(original):
            def __init__(self, folder, outer, particles):
                super().__init__(folder, outer, particles)
                self.result['protocol'].update(case=context.case,
                    scoring='retained nested module hierarchy; no overlapping-edge score')
        groups.Comparison = CaseComparison
        future = importlib.import_module('scripts.spt_kuramoto_b_future_mi')
        future.load_model = self.load_model
        future.budget_deadline = self.deadline
        from scripts.spt_multilevel_parity_case import graph_tree
        future.tree = lambda matrix: self.scored(*graph_tree(matrix))
        analyze = importlib.import_module('scripts.analyze_spt_kuramoto_b')
        analyze.scored = self.scored
        refine = importlib.import_module('scripts.validate_spt_kuramoto_b_full')
        refine.load_model = self.load_model
        refine.budget_deadline = self.deadline
        refine.scored = self.scored
        refine_groups = importlib.import_module('scripts.validate_spt_kuramoto_b_groups')
        refine_groups.Comparison = CaseComparison
        refine_groups.scored = self.scored
        return groups, future, analyze, refine, refine_groups

    def full_table(self, outer=1024, particles=256):
        path = self.folder / f'full_n{outer}_p{particles}.json'
        protocol = {'case': self.case, 'model_sha256': self.digest, 'outer': outer,
                    'particles': particles, 'scrambles': 4, 'max_particles': 256,
                    'source': 'Uniform(-pi,pi)^9', 'target': 'all nine future phases',
                    'syn_tolerance_nats': SYN_TOLERANCE, 'seed_base': 20261015,
                    'seed_stride': 1000}
        data = read(path) if path.exists() else {'protocol': protocol, 'rows': [], 'status': 'running'}
        if data['protocol'] != protocol:
            raise ValueError('Full EI table protocol mismatch')
        if data['status'] == 'complete':
            return data
        started = time.monotonic()
        previous = data.get('elapsed_seconds', 0.)
        def save():
            data['elapsed_seconds'] = previous + time.monotonic() - started
            save_json(path, data)
        try:
            model, _, _ = self.load_model()
            for scramble in range(4):
                row = next((r for r in data['rows'] if r['scramble'] == scramble), None)
                if row is None:
                    old = next((r for r in self.compatible_pilot['precision'] if
                                (r['outer'], r['particles'], r['scramble']) ==
                                (outer, particles, scramble)), None)
                    row = {'scramble': scramble, 'ei_nats': dict(old['ei_nats']) if old else {},
                           'diagnostics': dict(old['diagnostics']['queries']) if old else {}}
                    if old:
                        row['denominator'] = old['diagnostics']['denominator']
                    data['rows'].append(row)
                    save()
                if len(row['ei_nats']) == 511:
                    continue
                integral = FullTargetIntegrator(model, outer=outer, max_particles=256,
                    seed=20261015 + 1000 * scramble, deadline=self.deadline())
                for mask in range(1, 512):
                    if str(mask) in row['ei_nats']:
                        continue
                    values, _, diagnostic = integral.information([mask], particles)
                    row['ei_nats'][str(mask)] = values[mask]
                    row['diagnostics'][str(mask)] = diagnostic['queries'][str(mask)]
                    row['denominator'] = diagnostic['denominator']
                    save()
                    if len(row['ei_nats']) % 64 == 0:
                        print(json.dumps({'stage': 'full EI', 'case': self.case,
                            'scramble': scramble, 'subsets': len(row['ei_nats'])}), flush=True)
            values = np.array([[r['ei_nats'][str(m)] for m in range(1, 512)] for r in data['rows']])
            mean = values.mean(0)
            sem = values.std(0, ddof=1) / 2
            audits = [audit_table(masks_to_table(dict(zip(range(1, 512), v))), SYN_TOLERANCE)
                      for v in (mean, *values)]
            data.update(status='complete', mean_ei_nats=dict(zip(map(str, range(1, 512)), mean.tolist())),
                        sem_ei_nats=dict(zip(map(str, range(1, 512)), sem.tolist())),
                        audit=audits[0], replicate_audits=audits[1:])
            count = sum(a['significant_violation_count'] for a in audits)
            if count:
                data.update(status='significant_syn_violation', error=
                    f"Syn minimum={min(a['minimum_syn_nats'] for a in audits)}; "
                    f"threshold={-SYN_TOLERANCE}; count={count}")
        except (TimeoutError, ArithmeticError, np.linalg.LinAlgError) as error:
            data.update(status='budget_exhausted' if isinstance(error, TimeoutError) else 'numerical_failure',
                        error=str(error))
        save()
        return data

    def run(self, phase='all'):
        groups, future, analyze, refine, refine_groups = self.configure_evaluators()
        state_path = self.folder / 'run.json'
        state = read(state_path) if state_path.exists() else {
            'case': self.case, 'model_sha256': self.digest, 'phases': {}, 'status': 'running'}
        if state['case'] != self.case or state['model_sha256'] != self.digest:
            raise ValueError('Worker state belongs to a different frozen model')
        sequence = ('full', 'analysis', 'future', 'group', 'refine_full', 'refine_group')
        phases = sequence if phase == 'all' else (phase,)
        for step in phases:
            if phase == 'all' and state['phases'].get(step, {}).get('finished'):
                continue
            state.update(status='running', active_phase=step)
            save_json(state_path, state)
            print(json.dumps({'case': self.case, 'starting_phase': step}), flush=True)
            try:
                check_time(self.deadline())
                if step == 'full':
                    result = self.full_table()
                elif step == 'analysis':
                    result = analyze.analyze(self.folder)
                elif step == 'future':
                    result = future.run(self.folder, power_high=18)
                    if result.get('status') == 'complete' and not result.get('precision_passed'):
                        result = future.run(self.folder, power_high=20)
                elif step == 'group':
                    result = groups.Comparison(self.folder, 256, 256).run()
                elif step == 'refine_full':
                    result = refine.run(self.folder)
                elif step == 'refine_group':
                    result = refine_groups.run(self.folder)
                else:
                    raise ValueError(step)
            except (TimeoutError, ArithmeticError, ValueError, KeyError, np.linalg.LinAlgError) as error:
                result = {'status': 'budget_exhausted' if isinstance(error, TimeoutError)
                          else 'numerical_failure', 'error': str(error)}
            status = result.get('status')
            finished = status in ('complete', 'exploratory_single_dataset')
            state['phases'][step] = {'status': status, 'finished': finished}
            save_json(state_path, state)
            print(json.dumps({'case': self.case, 'finished_phase': step, 'status': status}), flush=True)
            if not finished and phase == 'all':
                # A significant full-Syn failure blocks SPT, but independent
                # future/group baselines still have scientific value. Numerical
                # budget failures stop the worker; no missing method becomes 0.
                if status in ('budget_exhausted', 'numerical_failure'):
                    state.update(status=status, error=result.get('error'))
                    save_json(state_path, state)
                    return state
        state['status'] = 'complete' if all(state['phases'].get(s, {}).get('finished') for s in sequence) else 'partial'
        state['completion_scope'] = 'query stages only; per-method numerical validity requires final analysis'
        save_json(state_path, state)
        return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--hours', type=float, default=6.)
    parser.add_argument('--case', choices=('A', 'C'))
    parser.add_argument('--phase', default='all', choices=(
        'all', 'full', 'analysis', 'future', 'group', 'refine_full', 'refine_group'))
    args = parser.parse_args()
    if args.initialize:
        print(json.dumps(initialize(args.hours)))
    elif args.case:
        print(json.dumps(Case(args.case).run(args.phase)))
    else:
        parser.error('Choose --initialize or --case')
