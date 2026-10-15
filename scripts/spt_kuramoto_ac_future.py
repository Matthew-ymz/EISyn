"""Run independent future-MI work while full-target A/C tables are computing.

Validate only this phase's actual sources, so unrelated concurrent edits of B's
SPT analysis do not affect the frozen future-MI model or evaluator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_kuramoto_finite_tm import PeriodicConditionalTM
from scripts.spt_multilevel_parity_case import graph_tree
from scripts.spt_continuous_phase_benchmark import score
from scripts.spt_local_hyperedge_kuramoto import SUPPORTS
from scripts import spt_kuramoto_b_future_mi as future


def run(case):
    folder = ROOT / 'results/spt_local_kuramoto_tm/ac_comparison'
    budget = json.loads((folder / 'budget.json').read_text())
    required = ('spt_kuramoto_finite_tm.py', 'spt_kuramoto_flow_tm.py',
                'spt_kuramoto_b_future_mi.py', 'spt_multilevel_parity_case.py',
                'spt_continuous_phase_benchmark.py', 'spt_heterogeneous_hierarchy_case.py')
    for name in required:
        if hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest() != budget['shared_source_sha256'][name]:
            raise ValueError(f'Future-MI evaluator changed: {name}')
    record = json.loads((folder / case / 'frozen_model.json').read_text())
    digest = hashlib.sha256(json.dumps(record['model'], sort_keys=True).encode()).hexdigest()
    future.load_model = lambda: (PeriodicConditionalTM.load(record['model']), record, digest)
    future.budget_deadline = lambda _: time.monotonic() + budget['deadline_epoch_seconds'] - time.time()
    def scored(matrix):
        root, clades = graph_tree(matrix)
        return score({'status': 'exploratory_numerical', 'root_split': root,
                      'clades': clades}, SUPPORTS[case])
    future.tree = scored
    result = future.run(folder / case, power_high=18)
    if result.get('status') == 'complete' and not result.get('precision_passed'):
        result = future.run(folder / case, power_high=20)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('A', 'C'), required=True)
    args = parser.parse_args()
    result = run(args.case)
    print(json.dumps({k: result.get(k) for k in ('status', 'precision_passed', 'metrics')}))
