"""Budgeted extension of passing circular pilots to all 511 source EIs.

This is the 1024x256 nonnegativity screen. It does not certify an SPT or report
F1: all-query high-resolution/ranking checks and group-target baselines remain
separate stages. Partial expensive results are reusable, never filled with zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.spt_kuramoto_finite_tm import (
    FullTargetIntegrator,PeriodicConditionalTM,SYN_TOLERANCE,check_time,save_json,masks_to_table,
)
from scripts.spt_continuous_hierarchy_pilot import audit_table


def run(pilot_path,output,*,prior_seconds=1271.1770462496200,total_budget=3518.):
    pilot=json.loads(pilot_path.read_text())
    if pilot['status']!='pilot_complete':
        raise ValueError('Complete the A/B/C pilot before extending passing cases')
    started=time.monotonic()
    previous=prior_seconds+pilot['elapsed_compute_seconds']
    protocol={'pilot_sha256':hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
        'outer':1024,'particles':256,'scrambles':4,'syn_tolerance_nats':SYN_TOLERANCE,
        'prior_seconds':prior_seconds,'total_experiment_budget_seconds':total_budget,
        'ordering':'A,B,C; passing critical-query pilots only',
        'scope':'complete EI/nonnegativity screen; no F1 or six-method comparison certification'}
    payload={'protocol':protocol,'records':[],'status':'running'}
    if output.exists():
        payload=json.loads(output.read_text())
        if payload['protocol']!=protocol: raise ValueError('Full-table cache protocol mismatch')
        previous=max(previous,payload['elapsed_main_compute_seconds'])
    deadline=started+total_budget-previous
    def save():
        payload['elapsed_main_compute_seconds']=previous+time.monotonic()-started
        save_json(output,payload)
    save()
    record=None; scramble=None; mask=None
    try:
        for source_record in pilot['records']:
            case=source_record['case']
            record=next((r for r in payload['records'] if r['case']==case),None)
            if record and record['status'] in ('pilot_failed','screen_complete'): continue
            if record is None:
                record={'case':case,'status':'preflight','replicates':[]}
                payload['records'].append(record)
            if not source_record.get('precision_passed'):
                record.update(status='pilot_failed',pilot_status=source_record['status']); save(); continue
            model=PeriodicConditionalTM.load(source_record['model'])
            for scramble in range(4):
                check_time(deadline)
                row=next((r for r in record['replicates'] if r['scramble']==scramble),None)
                if row is None:
                    cached=next(r for r in source_record['precision']
                                if (r['outer'],r['particles'],r['scramble'])==(1024,256,scramble))
                    row={'scramble':scramble,'ei_nats':dict(cached['ei_nats']),
                         'diagnostics':dict(cached['diagnostics']['queries']),
                         'denominator':cached['diagnostics']['denominator']}
                    record['replicates'].append(row)
                integral=FullTargetIntegrator(model,outer=1024,max_particles=256,
                    seed=20261015+1000*scramble,deadline=deadline)
                for mask in range(1,512):
                    if str(mask) in row['ei_nats']: continue
                    check_time(deadline)
                    values,_,diagnostic=integral.information([mask],256)
                    row['ei_nats'][str(mask)]=values[mask]
                    row['diagnostics'][str(mask)]=diagnostic['queries'][str(mask)]
                    record['status']='screen_running'; save()
                    if len(row['ei_nats'])%64==0:
                        print(json.dumps({'stage':'all_subsets','case':case,'scramble':scramble,
                            'completed_subsets':len(row['ei_nats'])}),flush=True)
            data=np.array([[r['ei_nats'][str(m)] for m in range(1,512)] for r in record['replicates']])
            means=data.mean(0); sem=data.std(0,ddof=1)/2
            audit=audit_table(masks_to_table(dict(zip(range(1,512),means))),SYN_TOLERANCE)
            replicate_audits=[audit_table(masks_to_table({int(k):v for k,v in r['ei_nats'].items()}),SYN_TOLERANCE)
                              for r in record['replicates']]
            diagnostics=[v for r in record['replicates'] for v in
                [r['denominator'],*r['diagnostics'].values()] if v['ess_minimum'] is not None]
            minimum_ess=min(v['ess_minimum'] for v in diagnostics)
            maximum_weight=max(v['maximum_normalized_weight'] for v in diagnostics)
            maximum_step=max(v['max_remaining_step_rad'] for v in diagnostics)
            violations=sum(a['significant_violation_count'] for a in replicate_audits)
            reasons=[]
            if sem.max()>=.01: reasons.append('maximum EI scramble SEM >= .01 nats')
            if minimum_ess<10: reasons.append('minimum ESS < 10')
            if maximum_weight>=.25: reasons.append('maximum normalized weight >= .25')
            if maximum_step>=1e-3: reasons.append('remaining Newton step >= 1e-3 rad')
            if audit['significant_violation_count'] or violations:
                minimum=min(a['minimum_syn_nats'] for a in [audit,*replicate_audits])
                reasons.append(f'Syn minimum={minimum}, threshold={-SYN_TOLERANCE}, affected={violations}')
            record.update(status='screen_complete',screen_passed=not reasons,failed_checks=reasons,
                ei_nats={str(m):float(v) for m,v in zip(range(1,512),means)},
                ei_sem_nats={str(m):float(v) for m,v in zip(range(1,512),sem)},
                audit=audit,replicate_audits=replicate_audits,
                proposal_audit={'minimum_ess':minimum_ess,'maximum_normalized_weight':maximum_weight,
                    'maximum_remaining_step_rad':maximum_step},
                hierarchy_status='not_evaluated: high-grid, split-ranking and partial-target checks pending')
            save()
    except TimeoutError as error:
        if record is not None: record.update(status='budget_exhausted',error=str(error))
        payload['status']='budget_exhausted'; save(); return payload
    except (ArithmeticError,np.linalg.LinAlgError) as error:
        if record is not None: record.update(status='numerical_failure',error=str(error),failed_scramble=scramble,failed_mask=mask)
        payload['status']='numerical_failure'; save(); return payload
    payload['status']='screen_complete'; save(); return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,default=ROOT/'results/spt_local_kuramoto_tm/flow_mixture_dt0025.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/spt_local_kuramoto_tm/full_table.json')
    args=parser.parse_args()
    result=run(args.pilot,args.output)
    print(json.dumps({'status':result['status'],'elapsed_main_compute_seconds':result['elapsed_main_compute_seconds']}))
