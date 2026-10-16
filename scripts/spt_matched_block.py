"""Unnormalized Block-PhiR, exact node search, fixed predeclared two-grid check."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_matched_common import Oracle, FOLDER, deadline, fitted_model, read, state, guarded, quality, labels, splits, TOLERANCE


def tree(oracle, repeat=None):
    clades, raw_values, gaps = [], [], []
    def info(a,b):
        values=oracle.query(a,b)
        return float(values.mean() if repeat is None else values[repeat])
    def visit(parent):
        if parent.bit_count()==1:
            return
        aa=info(parent,parent)
        candidates=[]
        for left,right in splits(parent):
            ll,rr=info(left,left),info(right,right)
            phi=aa-ll-rr+min(ll,rr,info(left,right),info(right,left))
            raw_values.append(phi)
            candidates.append((phi,labels(left),labels(right),left,right))
        bad=sum(x[0] < -TOLERANCE for x in candidates)
        if bad:
            raise ArithmeticError(f'Phi minimum={min(x[0] for x in candidates)} nats; threshold={-TOLERANCE}; affected={bad}; block={labels(parent)}')
        candidates.sort()
        selected=candidates[0]
        gaps.append({'block':labels(parent),'selected':selected[1:3],
                     'runner_up_gap_nats':candidates[1][0]-selected[0] if len(candidates)>1 else None})
        for child in selected[3:5]:
            if child.bit_count()>1:
                clades.append(labels(child))
            visit(child)
        return selected[1:3]
    root=visit(511)
    return quality(root,clades),{'minimum_phi_nats':min(raw_values),'threshold_nats':-TOLERANCE,
        'significant_violation_count':sum(v < -TOLERANCE for v in raw_values),
        'tolerance_negative_count':sum(-TOLERANCE<=v<0 for v in raw_values),'clipping':False,'selected_gaps':gaps}


def worker():
    end=deadline();model=fitted_model();path=FOLDER/'block.json'
    result=read(path) if path.exists() else {'status':'running','levels':[]}
    result.update(status='running',protocol='ordinary raw Block-PhiR; exact candidate enumeration; N256/P256 then N512/P512; four scrambles; no normalization',syn_tolerance_nats=TOLERANCE)
    state('block',result)
    try:
        for outer,particles in ((256,256),(512,512)):
            oracle=Oracle(model,outer,end,particles)
            level={'outer':outer,'particles':particles}
            result['stage']=f'block_n{outer}_p{particles}';state('block',result)
            try:
                mean,audit=tree(oracle);repeats=[];audits=[]
                for repeat in range(4):
                    metric,check=tree(oracle,repeat);repeats.append(metric);audits.append(check)
                level.update(metrics=mean,integration_replicates=repeats,audit=audit,replicate_audits=audits,
                             failed_queries={k:oracle.records[k]['issues'] for k in oracle.used if not oracle.records[k]['valid']},
                             information_nats={k:oracle.records[k]['mean_nats'] for k in oracle.used},
                             queried_information_count=len(oracle.used))
                level['status']='screen_passed' if not level['failed_queries'] else 'numerically_unresolved'
            except ArithmeticError as error:
                level.update(status='significant_phi_violation',error=str(error),metrics=None)
            result['levels']=[v for v in result['levels'] if (v['outer'],v['particles'])!=(outer,particles)]+[level]
            state('block',result)
        low,high=result['levels'];issues=[]
        common=set(low.get('information_nats',{}))&set(high.get('information_nats',{}))
        change=max((abs(low['information_nats'][k]-high['information_nats'][k]) for k in common),default=None)
        if change is None or change>=.01:
            issues.append('two_grid_mean_change_not_resolved')
        if high['status']!='screen_passed':
            issues.append(high['status'])
        result.update(status='screen_passed' if not issues else 'numerically_unresolved',stage='ended',
                      metrics=high.get('metrics') if not issues else None,diagnostic_metrics=high.get('metrics'),
                      failed_checks=issues,maximum_grid_change_nats=change,
                      integration_replicates=high.get('integration_replicates'))
        state('block',result)
    except Exception as error:
        result.update(status='budget_exhausted' if isinstance(error,TimeoutError) else 'failed',error=f'{type(error).__name__}: {error}',metrics=None)
        state('block',result)
        if not isinstance(error,TimeoutError):raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    worker() if args.worker else guarded('block',__file__,worker,resume=args.resume)
