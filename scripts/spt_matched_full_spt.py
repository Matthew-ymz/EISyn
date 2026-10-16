"""Complete paired-network gLV EI tables, raw Syn audits, and exact-node SPT."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_matched_common import FOLDER, Oracle, deadline, fitted_model, model_sha, read, state, guarded, TOLERANCE
from scripts.analyze_spt_glv_full_table import tree_record


def worker():
    end = deadline()
    model = fitted_model()
    path = FOLDER/'spt.json'
    result = read(path) if path.exists() else {'status': 'running', 'levels': []}
    result.update(status='running', model_sha=model_sha(model), independent_training_datasets=1,
                  syn_tolerance_nats=TOLERANCE, clipping=False,
                  protocol='all511 EI, all9330 bipartitions, fixed complete future, N1024/4096 P256 four scrambles')
    state('spt', result)
    try:
        for outer in (1024, 4096):
            oracle = Oracle(model, outer, end)
            level = next((v for v in result['levels'] if v['outer'] == outer), None)
            if level is None:
                level = {'outer': outer, 'subsets': 0}
                result['levels'].append(level)
            result['stage'] = f'full_ei_n{outer}'
            state('spt', result)
            for mask in range(1, 512):
                oracle.query(mask, 511)
                level['subsets'] = mask
                if mask % 16 == 0:
                    state('spt', result)
                    print(f'N={outer} complete subsets={mask}/511', flush=True)
            rows = [oracle.records[f'{mask}:511'] for mask in range(1, 512)]
            samples = np.array([r['values_nats'] for r in rows]).T
            level.update(complete=True, replicate_information_nats=[dict(zip(map(str, range(1,512)), r.tolist())) for r in samples],
                         mean_information_nats=dict(zip(map(str, range(1,512)), samples.mean(0).tolist())),
                         maximum_sem_nats=float(samples.std(0, ddof=1).max()/2),
                         failed_queries={r['key']: r['issues'] for r in rows if not r['valid']})
            tables = [tree_record(t, 'A', TOLERANCE) for t in level['replicate_information_nats']]
            canonical = tree_record(level['mean_information_nats'], 'A', TOLERANCE)
            level.update(mean_tree=canonical, replicate_trees=tables)
            nonnegative = canonical['status'] == 'exploratory_only' and all(t['status']=='exploratory_only' for t in tables)
            level['nonnegativity_passed'] = nonnegative
            state('spt', result)
            if not nonnegative:
                result.update(status='significant_nonnegativity_violation', metrics=None,
                              error='A complete raw Syn/Xi table violated the .01 nats tolerance; inspect per-table audits')
                state('spt', result)
                return
        low, high = result['levels']
        changes = [abs(low['mean_information_nats'][str(m)]-high['mean_information_nats'][str(m)]) for m in range(1,512)]
        failed = dict(high['failed_queries'])
        if max(changes) >= .01:
            failed['outer_grid'] = [f'Maximum1024_to4096_change={max(changes)}>=.01']
        high_tree = high['mean_tree']['metrics']
        repeats = [r['metrics'] for r in high['replicate_trees']]
        result.update(status='screen_passed' if not failed else 'numerically_unresolved',
                      maximum_outer_change_nats=max(changes), failed_checks=failed,
                      metrics=high_tree if not failed else None, diagnostic_metrics=high_tree,
                      integration_replicates=repeats, f1_range=[min(r['clade_f1'] for r in repeats),max(r['clade_f1'] for r in repeats)],
                      stage='ended', inference='one frozen fitted dataset; four numerical repeats, no population significance')
        state('spt', result)
    except Exception as error:
        result.update(status='budget_exhausted' if isinstance(error, TimeoutError) else 'failed',
                      error=f'{type(error).__name__}: {error}', metrics=None)
        state('spt', result)
        if not isinstance(error, TimeoutError):
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    worker() if args.worker else guarded('spt', __file__, worker, resume=args.resume)
