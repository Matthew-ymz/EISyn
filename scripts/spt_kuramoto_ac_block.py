"""Case-isolated, cached Block PhiR and normalized Block PhiR comparisons."""
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
from scripts.spt_kuramoto_ac_selected import SelectedCase, labels, mask
from scripts.spt_kuramoto_finite_tm import SYN_TOLERANCE, save_json, check_time
from scripts.spt_kuramoto_ac_group_tm import GroupTargetIntegrator
from scripts.spt import nontrivial_bipartitions


def normalized_cancelled(context):
    selection = context.folder.parent / 'method_selection.json'
    return selection.exists() and 'normalized_block_phi_r' in json.loads(selection.read_text()).get('cancelled', [])


class PersistentGroup(GroupTargetIntegrator):
    def __init__(self, context, outer, particles, scramble):
        model, _, _ = context.load_model()
        super().__init__(model, outer=outer, particles=particles, max_particles=max(1024, particles),
                         seed=20261015 + 1000 * scramble, deadline=context.deadline())
        self.path = context.folder / f'block_marginal_n{outer}_p{particles}_r{scramble}.jsonl'
        protocol = {'case': context.case, 'model_sha256': context.digest,
                    'outer': outer, 'particles': particles, 'max_particles': max(1024, particles),
                    'scramble': scramble, 'seed': 20261015 + 1000 * scramble,
                    'estimator_sha256': hashlib.sha256((ROOT / 'scripts/spt_kuramoto_ac_group_tm.py').read_bytes()).hexdigest()}
        if self.path.exists():
            with self.path.open() as f:
                if json.loads(next(f))['protocol'] != protocol:
                    raise ValueError('Block marginal protocol mismatch')
                for line in f:
                    row = json.loads(line)
                    self.cache[row['source'], row['target']] = np.asarray(row['log_marginal']), row['diagnostics']
        else:
            self.path.write_text(json.dumps({'protocol': protocol}) + '\n')

    def marginal(self, source, target):
        key = source, target
        if key in self.cache:
            return self.cache[key]
        values, diagnostic = super().marginal(source, target)
        with self.path.open('a') as f:
            f.write(json.dumps({'source': source, 'target': target,
                                'log_marginal': values.tolist(), 'diagnostics': diagnostic}, allow_nan=False) + '\n')
        return values, diagnostic


class BlockComparison:
    def __init__(self, case, outer=256, particles=256):
        self.context = SelectedCase(case)
        self.outer = outer
        self.particles = particles
        self.path = self.context.folder / f'block_selected_n{outer}_p{particles}.json'
        self.integrators = [PersistentGroup(self.context, outer, particles, r) for r in range(4)]
        self.information_cache = {}
        self.phi_cache = {}
        self.payload = {'protocol': {'case': case, 'model_sha256': self.context.digest,
                        'outer': outer, 'particles': particles, 'scrambles': 4,
                        'syn_tolerance_nats': SYN_TOLERANCE, 'clipping': False,
                        'source': 'independent uniform nine phases',
                        'point_estimate': 'average EI before nonlinear MMI minimum'},
                        'methods': {}, 'status': 'running'}

    def save(self):
        diagnostics = [d for integral in self.integrators for _, d in integral.cache.values()
                       if d.get('ess_minimum') is not None]
        if diagnostics:
            self.payload['proposal_audit'] = {
                'minimum_ess': min(d['ess_minimum'] for d in diagnostics),
                'maximum_normalized_weight': max(d['maximum_normalized_weight'] for d in diagnostics),
                'low_ess_marginal_count': sum(d['ess_minimum'] < 10 for d in diagnostics),
                'large_weight_marginal_count': sum(d['maximum_normalized_weight'] >= .25 for d in diagnostics)}
        self.payload['information_query_count'] = len(self.information_cache)
        save_json(self.path, self.payload)

    def info(self, source, target):
        key = source, target
        if key not in self.information_cache:
            values, lower, prefix = [], [], []
            for integral in self.integrators:
                value, pointwise, _ = integral.information(source, target)
                numerator, nd = integral.marginal(source, target)
                denominator, dd = integral.marginal(0, target)
                values.append(value)
                lower.append(float(pointwise[:self.outer // 2].mean()))
                if target == 511:
                    n, _ = integral.full.marginal(source, min(64, self.particles))
                    d, _ = integral.full.marginal(0, min(64, self.particles))
                else:
                    n = np.asarray(nd.get('prefix_log_marginal', numerator))
                    d = np.asarray(dd.get('prefix_log_marginal', denominator))
                prefix.append(float(np.mean(n - d)))
            self.information_cache[key] = {'values': np.asarray(values),
                                           'prefix': np.asarray(prefix), 'lower': np.asarray(lower)}
        return self.information_cache[key]

    def phi(self, parent, left, right):
        key = parent, left, right
        if key not in self.phi_cache:
            whole = self.info(parent, parent)
            ll = self.info(left, left)
            rr = self.info(right, right)
            terms = [self.info(a, b) for a in (left, right) for b in (left, right)]
            item = {k: whole[k] - ll[k] - rr[k] + np.min([v[k] for v in terms], axis=0)
                    for k in ('values', 'prefix', 'lower')}
            for k, output in (('values', 'estimate'), ('prefix', 'prefix_estimate'), ('lower', 'lower_estimate')):
                item[output] = float(whole[k].mean() - ll[k].mean() - rr[k].mean() + min(t[k].mean() for t in terms))
            self.phi_cache[key] = item
        return self.phi_cache[key]

    def tree(self, normalize=False):
        candidate_cache = {}
        selected = []
        all_values = {}
        def visit(parent, replicate=None):
            if parent.bit_count() == 1:
                return (), []
            if parent not in candidate_cache:
                choices = []
                for left, right in nontrivial_bipartitions(labels(parent)):
                    l, r = mask(left), mask(right)
                    value = self.phi(parent, l, r)
                    all_values[parent, l, r] = np.r_[value['values'], value['estimate']]
                    divisor = min(len(left), len(right)) if normalize else 1
                    choices.append((l, r, value, divisor))
                candidate_cache[parent] = choices
                print(json.dumps({'case': self.context.case, 'stage': 'Block candidates',
                                  'normalized': normalize, 'parent': parent, 'choices': len(choices)}), flush=True)
                self.save()
            choices = candidate_cache[parent]
            def objective(row):
                v = row[2]['estimate'] if replicate is None else row[2]['values'][replicate]
                return float(v / row[3])
            ordered = sorted(choices, key=lambda row: (objective(row), labels(row[0]), labels(row[1])))
            l, r, item, divisor = ordered[0]
            if replicate is None:
                runner = ordered[1] if len(ordered) > 1 else None
                gap = runner[2]['values'] / runner[3] - item['values'] / divisor if runner else None
                selected.append({'parent': parent, 'left': l, 'right': r, 'candidate_count': len(ordered),
                    'objective': objective(ordered[0]), 'objective_units': 'nats/node' if normalize else 'nats',
                    'runner_up_gap': objective(runner) - objective(ordered[0]) if runner else None,
                    'runner_up_gap_sem': float(gap.std(ddof=1) / 2) if gap is not None else None,
                    'maximum_inner_change': max(abs(z[2]['estimate'] - z[2]['prefix_estimate']) / z[3] for z in ordered),
                    'maximum_outer_change': max(abs(z[2]['estimate'] - z[2]['lower_estimate']) / z[3] for z in ordered)})
            _, lc = visit(l, replicate)
            _, rc = visit(r, replicate)
            return (labels(l), labels(r)), [labels(g) for g in (l, r) if g.bit_count() > 1] + lc + rc
        root, clades = visit(511)
        metrics = self.context.scored(root, clades)
        repeats = [self.context.scored(*visit(511, r)) for r in range(4)]
        raw = np.concatenate(list(all_values.values()))
        count = int(np.count_nonzero(raw < -SYN_TOLERANCE))
        record = {'status': 'complete' if count == 0 else 'significant_phi_violation',
                  'selection_diagnostics': selected,
                  'audit': {'minimum_raw_phi_nats': float(raw.min()), 'threshold_nats': -SYN_TOLERANCE,
                            'significant_violation_count': count,
                            'tolerance_negative_count': int(np.count_nonzero((raw < 0) & (raw >= -SYN_TOLERANCE))),
                            'unique_partition_count': len(all_values), 'draws_per_partition': 5}}
        if count:
            record['error'] = f'PhiR minimum={raw.min()}; threshold={-SYN_TOLERANCE}; count={count}'
        else:
            record.update(metrics=metrics, numerical_replicate_metrics=repeats)
        return record

    def run(self):
        started = time.monotonic()
        self.save()
        try:
            for name, normalize in (('block_phi_r', False), ('normalized_block_phi_r', True)):
                if normalize and normalized_cancelled(self.context):
                    self.payload['methods'][name] = {'status': 'cancelled_by_user', 'metrics': None}
                    self.save()
                    continue
                self.payload['methods'][name] = self.tree(normalize)
                self.save()
                print(json.dumps({'case': self.context.case, 'method': name,
                    'f1': self.payload['methods'][name].get('metrics', {}).get('clade_f1')}), flush=True)
            self.payload['status'] = 'complete'
        except (TimeoutError, ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
            self.payload.update(status='budget_exhausted' if isinstance(error, TimeoutError) else 'numerical_failure', error=str(error))
        self.payload['elapsed_seconds'] = time.monotonic() - started
        self.save()
        return self.payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('A', 'C'), required=True)
    parser.add_argument('--outer', type=int, default=256)
    parser.add_argument('--particles', type=int, default=256)
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        result = BlockComparison(args.case, args.outer, args.particles).run()
    print(json.dumps({'case': args.case, 'status': result['status']}))
