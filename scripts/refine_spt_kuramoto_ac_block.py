"""Validate coarse Block candidates on N512/P512, expanding newly visited parents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_kuramoto_ac_block import BlockComparison, normalized_cancelled
from scripts.spt_kuramoto_ac_selected import SelectedCase, labels, mask
from scripts.spt import nontrivial_bipartitions
from scripts.spt_kuramoto_finite_tm import SYN_TOLERANCE, save_json, check_time


def run(case):
    coarse = BlockComparison(case)
    previous = json.loads(coarse.path.read_text())
    if previous.get('status') != 'complete':
        raise ValueError('Finish coarse Block comparison first')
    # Reconstructing cached candidates must preserve the completed coarse
    # artifact; tree() saves diagnostics while visiting its parents.
    coarse.payload = previous
    high = BlockComparison(case, outer=512, particles=512)
    output = high.context.folder / 'block_selected_refinement.json'
    data = {'case': case, 'status': 'running', 'methods': {}, 'outer': 512,
            'informed_particles': 512, 'prior_only_particles': 1024, 'full_target_particles': 256,
            'screening': 'coarse top two plus within eight paired numerical SEM; new parents fully expanded'}
    try:
        for name, normalize in (('block_phi_r', False), ('normalized_block_phi_r', True)):
            if normalize and normalized_cancelled(high.context):
                data['methods'][name] = {'status': 'cancelled_by_user', 'metrics': None}
                save_json(output, data)
                continue
            if previous['methods'][name]['status'] != 'complete':
                data['methods'][name] = {'status': 'coarse_numerical_failure'}
                continue
            # Recreate every visited coarse candidate from the persistent marginal cache.
            coarse.tree(normalize)
            raw_cache = dict(coarse.phi_cache)
            registered = {p for p, _, _ in raw_cache}
            pools = {}
            for parent in registered:
                candidates = []
                for (p, l, r), value in raw_cache.items():
                    if p != parent:
                        continue
                    divisor = min(l.bit_count(), r.bit_count()) if normalize else 1
                    candidates.append((value['estimate'] / divisor, l, r, value, divisor))
                candidates.sort(key=lambda c: (c[0], labels(c[1]), labels(c[2])))
                best = candidates[0]
                pools[parent] = []
                for i, candidate in enumerate(candidates):
                    difference = candidate[3]['values'] / candidate[4] - best[3]['values'] / best[4]
                    if i < 2 or candidate[0] - best[0] <= 8 * difference.std(ddof=1) / 2:
                        pools[parent].append(candidate[1:3])
            audit_values = {}
            selections = []
            def visit(parent, replicate=None):
                if parent.bit_count() == 1:
                    return (), []
                if parent not in pools:
                    pools[parent] = [(mask(l), mask(r)) for l, r in nontrivial_bipartitions(labels(parent))]
                choices = []
                for l, r in pools[parent]:
                    item = high.phi(parent, l, r)
                    audit_values[parent, l, r] = np.r_[item['values'], item['estimate']]
                    divisor = min(l.bit_count(), r.bit_count()) if normalize else 1
                    objective = item['estimate'] if replicate is None else item['values'][replicate]
                    choices.append((float(objective / divisor), l, r, item, divisor))
                choices.sort(key=lambda c: (c[0], labels(c[1]), labels(c[2])))
                best = choices[0]
                if replicate is None:
                    low = coarse.phi(parent, best[1], best[2]) if (parent, best[1], best[2]) in raw_cache else None
                    gap = choices[1][3]['values'] / choices[1][4] - best[3]['values'] / best[4] if len(choices) > 1 else None
                    selections.append({'parent': parent, 'left': best[1], 'right': best[2], 'candidate_count': len(choices),
                        'objective': best[0], 'coarse_change_objective': abs(best[0] - low['estimate'] / best[4]) if low else None,
                        'runner_gap_objective': choices[1][0] - best[0] if len(choices) > 1 else None,
                        'runner_gap_sem_objective': float(gap.std(ddof=1) / 2) if gap is not None else None})
                _, left = visit(best[1], replicate)
                _, right = visit(best[2], replicate)
                return (labels(best[1]), labels(best[2])), [labels(m) for m in best[1:3] if m.bit_count() > 1] + left + right
            trees = [visit(511, r) for r in (None, 0, 1, 2, 3)]
            values = np.concatenate(list(audit_values.values()))
            count = int(np.count_nonzero(values < -SYN_TOLERANCE))
            record = {'status': 'complete' if not count else 'significant_phi_violation',
                'audit': {'minimum_raw_phi_nats': float(values.min()), 'threshold_nats': -SYN_TOLERANCE,
                    'significant_violation_count': count,
                    'tolerance_negative_count': int(np.count_nonzero((values < 0) & (values >= -SYN_TOLERANCE)))},
                'selection_diagnostics': selections}
            if not count:
                record.update(metrics=high.context.scored(*trees[0]),
                    numerical_replicate_metrics=[high.context.scored(*t) for t in trees[1:]])
            else:
                record['error'] = f'PhiR minimum={values.min()}; threshold={-SYN_TOLERANCE}; count={count}'
            data['methods'][name] = record
            high.save()
            data['proposal_audit'] = high.payload.get('proposal_audit', {})
            save_json(output, data)
            print(json.dumps({'case': case, 'method': name, 'status': record['status'],
                              'f1': record.get('metrics', {}).get('clade_f1')}), flush=True)
        data['status'] = 'complete'
    except (TimeoutError, ValueError, ArithmeticError, np.linalg.LinAlgError) as e:
        data.update(status='budget_exhausted' if isinstance(e, TimeoutError) else 'numerical_failure', error=str(e))
    save_json(output, data)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('A', 'C'), required=True)
    parser.add_argument('--wait-for-coarse', action='store_true')
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if args.wait_for_coarse:
            context = SelectedCase(args.case)
            coarse_path = context.folder / 'block_selected_n256_p256.json'
            while True:
                check_time(context.deadline())
                state = json.loads(coarse_path.read_text()) if coarse_path.exists() else {}
                if state.get('status') == 'complete':
                    break
                if state.get('status') not in (None, 'running'):
                    raise RuntimeError(f"Coarse Block stage ended: {state.get('status')}")
                time.sleep(30)
        result = run(args.case)
    print(json.dumps({'case': args.case, 'status': result['status']}))
