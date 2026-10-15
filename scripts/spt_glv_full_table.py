"""Extend passing HO-gLV pilots to the complete EI/nonnegativity screen.

The 1024x256 screen uses the already registered grid and all four scrambles.
It does not certify a hierarchy or report F1: the all-query high-grid/ranking
checks and partial-target baselines are separate, potentially costly stages.
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
from scripts.spt_glv_finite_tm import RealConditionalTM,RealFullTargetIntegrator
from scripts.spt_kuramoto_finite_tm import check_time,save_json
from scripts.spt_continuous_hierarchy_pilot import audit_table


def run(pilot_path,output,budget_seconds=3600.,prior_compute_seconds=0.):
    pilot=json.loads(pilot_path.read_text())
    if pilot['status']!='pilot_complete':
        raise ValueError('Finish the A/B/C pilot before extending its passing cases')
    started=time.monotonic(); previous=max(pilot['elapsed_compute_seconds'],prior_compute_seconds)
    protocol={'pilot_sha256':hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
        'outer':1024,'particles':256,'scrambles':4,'precision_target_nats':.01,
        'syn_tolerance_nats':pilot['protocol']['syn_tolerance_nats'],
        'budget_seconds':budget_seconds,'ordering':'A, B, C; passing pilots only',
        'scope':'complete EI and 9330-residual screen; no F1 without high-grid/ranking validation'}
    payload={'protocol':protocol,'records':[],'status':'running'}
    if output.exists():
        payload=json.loads(output.read_text())
        if payload['protocol']!=protocol: raise ValueError('Full-table cache protocol mismatch')
        previous=max(previous,payload.get('elapsed_compute_seconds',previous))
    deadline=started+budget_seconds-previous
    def save():
        payload['elapsed_compute_seconds']=previous+time.monotonic()-started
        save_json(output,payload)
    save()
    try:
        for source_record in pilot['records']:
            case=source_record['case']
            record=next((r for r in payload['records'] if r['case']==case),None)
            if record is not None and record['status'] in ('pilot_failed','screen_complete'): continue
            if record is None:
                record={'case':case,'status':'preflight','replicates':[]}
                payload['records'].append(record)
            if not source_record.get('precision_passed'):
                record.update(status='pilot_failed',pilot_status=source_record['status']); save(); continue
            model=RealConditionalTM.load(source_record['model'])
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
                integrator=RealFullTargetIntegrator(model,outer=1024,max_particles=256,
                    seed=20261015+1000*scramble,deadline=deadline)
                for mask in range(1,512):
                    if str(mask) in row['ei_nats']: continue
                    check_time(deadline)
                    values,_,diagnostic=integrator.information([mask],256)
                    row['ei_nats'][str(mask)]=values[mask]
                    row['diagnostics'][str(mask)]=diagnostic['queries'][str(mask)]
                    record['status']='screen_running'; save()
                    if len(row['ei_nats'])%64==0:
                        print(json.dumps({'stage':'full_table','case':case,'scramble':scramble,
                                          'subsets':len(row['ei_nats'])}),flush=True)
            masks=list(range(1,512))
            samples=np.asarray([[r['ei_nats'][str(m)] for m in masks] for r in record['replicates']])
            means=samples.mean(0); sem=samples.std(0,ddof=1)/2
            coalitions={tuple(i+1 for i in range(9) if m>>i&1):float(means[m-1]) for m in masks}
            audit=audit_table(coalitions,protocol['syn_tolerance_nats'])
            diagnostic=[d for r in record['replicates'] for d in
                [r['denominator'],*r['diagnostics'].values()] if d.get('ess_minimum') is not None]
            minimum_ess=min(d['ess_minimum'] for d in diagnostic)
            maximum_weight=max(d['maximum_normalized_weight'] for d in diagnostic)
            maximum_step=max(d['max_projected_remaining_step_coordinate'] for d in diagnostic)
            reasons=[]
            if sem.max()>=.01: reasons.append('maximum EI SEM >= .01 nats')
            if minimum_ess<10: reasons.append('minimum ESS < 10')
            if maximum_weight>=.25: reasons.append('maximum normalized weight >= .25')
            if maximum_step>=1e-3: reasons.append('projected Newton step >= 1e-3')
            if audit['significant_violation_count']:
                reasons.append(f"Syn minimum={audit['minimum_syn_nats']:.9g} nats, threshold=-{protocol['syn_tolerance_nats']}, affected={audit['significant_violation_count']}")
            record.update(status='screen_complete',ei_nats={str(m):float(means[m-1]) for m in masks},
                ei_sem_nats={str(m):float(sem[m-1]) for m in masks},maximum_ei_sem_nats=float(sem.max()),
                audit=audit,proposal_audit={'minimum_ess':minimum_ess,
                    'maximum_normalized_weight':maximum_weight,
                    'maximum_projected_newton_step_coordinate':maximum_step},
                screen_passed=not reasons,failed_checks=reasons,
                hierarchy_status='not_evaluated: full high-grid and ranking validation pending')
            save(); print(json.dumps({'stage':'screen_complete','case':case,
                'screen_passed':not reasons,'failed_checks':reasons,'audit':audit}),flush=True)
    except TimeoutError as error:
        record['status']='budget_exhausted'; record['error']=str(error)
        payload['status']='budget_exhausted'; save(); return payload
    except (ArithmeticError,np.linalg.LinAlgError) as error:
        record['status']='numerical_failure'; record['error']=str(error)
        record['failed_scramble']=scramble; record['failed_mask']=mask
        payload['status']='numerical_failure'; save(); return payload
    payload['status']='screen_complete'; save(); return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,default=ROOT/'results/spt_local_glv_tm/pilot.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/spt_local_glv_tm/full_table.json')
    parser.add_argument('--prior-compute-seconds',type=float,default=0.)
    args=parser.parse_args()
    result=run(args.pilot,args.output,prior_compute_seconds=args.prior_compute_seconds)
    print(json.dumps({'status':result['status'],'seconds':result['elapsed_compute_seconds']}))
