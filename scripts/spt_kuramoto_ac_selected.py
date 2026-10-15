"""Selected comparison: SPT, SURD max-weight tree, and two Block PhiR trees.

This entry isolates A/C from concurrent changes to the B analysis wrappers.
The expensive full-target EI cache is resumed without changing its protocol.
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
from scripts.spt_kuramoto_ac_comparison import Case as FullTableWorker, FOLDER, PILOT
from scripts.spt_kuramoto_finite_tm import (
    FullTargetIntegrator, PeriodicConditionalTM, SYN_TOLERANCE, masks_to_table, save_json, check_time,
)
from scripts.spt import SPTConfig, build_spt_from_ei_table, flatten_nodes, nontrivial_bipartitions
from scripts.spt_continuous_hierarchy_pilot import audit_table
from scripts.spt_continuous_phase_benchmark import score
from scripts.spt_local_hyperedge_kuramoto import SUPPORTS


def mask(nodes):
    return sum(1 << (i - 1) for i in nodes)


def labels(value):
    return tuple(i + 1 for i in range(9) if value >> i & 1)


def read(path):
    return json.loads(Path(path).read_text())


class SelectedCase(FullTableWorker):
    def __init__(self, case):
        self.case = case
        self.folder = FOLDER / case
        self.budget = read(FOLDER / 'budget.json')
        if self.budget['pilot_sha256'] != hashlib.sha256(PILOT.read_bytes()).hexdigest():
            raise ValueError('Frozen A/C model cache changed')
        required = ('spt_kuramoto_finite_tm.py', 'spt_kuramoto_flow_tm.py', 'spt.py',
                    'spt_continuous_hierarchy_pilot.py', 'spt_continuous_phase_benchmark.py',
                    'spt_multilevel_parity_case.py', 'spt_heterogeneous_hierarchy_case.py')
        for name in required:
            current = hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest()
            if current != self.budget['shared_source_sha256'][name]:
                raise ValueError(f'Core estimator/scoring changed: {name}')
        self.pilot = next(r for r in read(PILOT)['records'] if r['case'] == case)
        self.compatible_pilot = {**self.pilot, 'precision': [*self.pilot['precision'],
                                                        *self.pilot['root_precision']]}
        self.digest = hashlib.sha256(json.dumps(self.pilot['model'], sort_keys=True).encode()).hexdigest()

    def scored(self, root, clades):
        return score({'status': 'computed', 'root_split': root, 'clades': clades}, SUPPORTS[self.case])

    def table_tree(self, values):
        table = masks_to_table({int(k): v for k, v in values.items()})
        if len(table) != 511:
            raise ValueError('Complete SPT requires every source EI')
        audit = audit_table(table, SYN_TOLERANCE)
        xi = [v - sum(table[(i,)] for i in s) for s, v in table.items() if len(s) > 1]
        audit.update(minimum_xi_nats=min(xi),
                     significant_xi_violation_count=sum(v < -SYN_TOLERANCE for v in xi),
                     tolerance_negative_xi_count=sum(-SYN_TOLERANCE <= v < 0 for v in xi))
        if audit['significant_violation_count'] or audit['significant_xi_violation_count']:
            raise ArithmeticError(f"Syn/Xi minimum={min(audit['minimum_syn_nats'], min(xi))}; "
                f"threshold={-SYN_TOLERANCE}; count="
                f"{audit['significant_violation_count'] + audit['significant_xi_violation_count']}")
        tree = build_spt_from_ei_table(tuple(range(1, 10)), table,
                                    config=SPTConfig(syn_tolerance=SYN_TOLERANCE))
        nodes = flatten_nodes(tree.root)
        if abs(tree.closure_error) > SYN_TOLERANCE:
            raise ArithmeticError(f'SPT closure failed: {tree.closure_error} nats')
        return {'metrics': self.scored(tuple(c.sources for c in tree.root.children),
                    [n.sources for n in nodes if 1 < len(n.sources) < 9]),
                'audit': audit, 'closure_error_nats': tree.closure_error,
                'tolerance_zero_count': tree.audit.tolerance_zero_count,
                'nodes': [{'sources': list(n.sources), 'children': [list(c.sources) for c in n.children],
                           'syn_nats': n.syn_value, 'xi_nats': n.xi_value} for n in nodes]}

    def spt(self):
        path = self.folder / 'spt_selected.json'
        result = {'case': self.case, 'model_sha256': self.digest, 'status': 'running',
                  'units': 'nats', 'syn_tolerance_nats': SYN_TOLERANCE,
                  'scope': 'SPT plus the user-selected three baselines only'}
        try:
            data = self.full_table()
            if data['status'] != 'complete':
                result.update(status=data['status'], error=data.get('error'))
                save_json(path, result)
                return result
            mean = self.table_tree(data['mean_ei_nats'])
            replicates = [self.table_tree(r['ei_nats']) for r in data['rows']]
            result.update(mean, status='coarse_complete', numerical_replicate_metrics=[r['metrics'] for r in replicates])
            save_json(path, result)
            print(json.dumps({'case': self.case, 'stage': 'SPT complete',
                              'f1': mean['metrics']['clade_f1']}), flush=True)
            result['refinement'] = self.refine_spt(data, mean)
            result['status'] = 'complete' if result['refinement']['status'] == 'complete' else 'coarse_complete_refinement_unresolved'
        except (TimeoutError, ArithmeticError, ValueError, KeyError, np.linalg.LinAlgError) as error:
            result.update(status='budget_exhausted' if isinstance(error, TimeoutError) else 'numerical_failure', error=str(error))
        save_json(path, result)
        return result

    def refine_spt(self, coarse, tree):
        raw = np.array([[r['ei_nats'][str(m)] for m in range(1, 512)] for r in coarse['rows']])
        desired = {511, *[1 << i for i in range(9)]}
        registered = set()
        for node in tree['nodes']:
            if len(node['sources']) < 2:
                continue
            parent = mask(node['sources'])
            registered.add(parent)
            desired.add(parent)
            candidates = []
            for left, right in nontrivial_bipartitions(node['sources']):
                l, r = mask(left), mask(right)
                residual = raw[:, parent - 1] - raw[:, l - 1] - raw[:, r - 1]
                candidates.append((float(residual.mean()), l, r, residual))
            candidates.sort(key=lambda c: (c[0], labels(c[1]), labels(c[2])))
            best = candidates[0]
            for index, candidate in enumerate(candidates):
                difference = candidate[3] - best[3]
                if index < 2 or difference.mean() <= 8 * difference.std(ddof=1) / 2:
                    desired.update(candidate[1:3])
        path = self.folder / 'spt_selected_refined.json'
        protocol = {'model_sha256': self.digest, 'outer': 4096, 'particles': 256,
                    'scrambles': 4, 'initial_source_masks': sorted(desired),
                    'screening': 'coarse top two plus within 8 numerical SEM; new parents fully expanded'}
        data = read(path) if path.exists() else {'protocol': protocol, 'rows': [], 'status': 'running'}
        if data['protocol'] != protocol:
            raise ValueError('SPT refinement protocol differs')
        model, _, _ = self.load_model()
        for scramble in range(4):
            if not any(r['scramble'] == scramble for r in data['rows']):
                old = next(r for r in self.pilot['root_precision'] if r['scramble'] == scramble)
                data['rows'].append({'scramble': scramble, 'ei_nats': dict(old['ei_nats']),
                                    'denominator': old['diagnostics']['denominator'],
                                    'diagnostics': dict(old['diagnostics']['queries'])})
        def fill():
            for row in data['rows']:
                integral = FullTargetIntegrator(model, outer=4096, max_particles=256,
                    seed=20261015 + 1000 * row['scramble'], deadline=self.deadline())
                for source in sorted(desired):
                    if str(source) in row['ei_nats']:
                        continue
                    values, _, diagnostics = integral.information([source], 256)
                    row['ei_nats'][str(source)] = values[source]
                    row['diagnostics'][str(source)] = diagnostics['queries'][str(source)]
                    save_json(path, data)
                print(json.dumps({'case': self.case, 'stage': 'SPT refine',
                                  'scramble': row['scramble'], 'queries': len(row['ei_nats'])}), flush=True)
        while True:
            check_time(self.deadline())
            fill()
            pending = set()
            partition_values = {}
            def build(parent, replicate=None):
                if parent.bit_count() == 1:
                    return (), []
                candidates = []
                for left, right in nontrivial_bipartitions(labels(parent)):
                    l, r = mask(left), mask(right)
                    if l not in desired or r not in desired:
                        if parent not in registered:
                            pending.update((l, r))
                        continue
                    values = np.array([z['ei_nats'][str(parent)] - z['ei_nats'][str(l)]
                                       - z['ei_nats'][str(r)] for z in data['rows']])
                    partition_values[parent, l, r] = values
                    candidates.append((float(values.mean() if replicate is None else values[replicate]), l, r))
                if parent not in registered and pending:
                    return (), []
                if not candidates:
                    raise ArithmeticError(f'No represented high-grid split: {parent}')
                _, left, right = min(candidates, key=lambda c: (c[0], labels(c[1]), labels(c[2])))
                _, lc = build(left, replicate)
                _, rc = build(right, replicate)
                return (labels(left), labels(right)), [labels(g) for g in (left, right) if g.bit_count() > 1] + lc + rc
            trees = [build(511, r) for r in (None, 0, 1, 2, 3)]
            if pending:
                desired.update(pending)
                continue
            values = np.concatenate(list(partition_values.values()))
            count = int(np.count_nonzero(values < -SYN_TOLERANCE))
            audit = {'minimum_raw_syn_nats': float(values.min()), 'threshold_nats': -SYN_TOLERANCE,
                     'significant_violation_count': count,
                     'tolerance_negative_count': int(np.count_nonzero((values < 0) & (values >= -SYN_TOLERANCE))),
                     'unique_partition_count': len(partition_values)}
            xi = [row['ei_nats'][str(m)] - sum(row['ei_nats'][str(1 << i)]
                   for i in range(9) if m >> i & 1) for row in data['rows']
                   for m in desired if m.bit_count() > 1]
            audit.update(minimum_xi_nats=min(xi),
                         significant_xi_violation_count=sum(v < -SYN_TOLERANCE for v in xi),
                         tolerance_negative_xi_count=sum(-SYN_TOLERANCE <= v < 0 for v in xi))
            count += audit['significant_xi_violation_count']
            if count:
                data.update(status='significant_syn_violation', audit=audit)
            else:
                data.update(status='complete', audit=audit, metrics=self.scored(*trees[0]),
                    numerical_replicate_metrics=[self.scored(*t) for t in trees[1:]],
                    source_masks=sorted(desired), scope='coarse full enumeration plus high-grid candidate-pool validation')
            save_json(path, data)
            return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('A', 'C'), required=True)
    parser.add_argument('--task', choices=('spt',), default='spt')
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        result = SelectedCase(args.case).spt()
    print(json.dumps({'case': args.case, 'status': result['status']}))
