"""B-only, common-fitted-joint comparison; every expensive query is resumable.

The four-hour follow-up is separate from the exhausted A/B/C pilot budget.
No topology or scoring truth enters the information evaluator.
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
    FullTargetIntegrator,PeriodicConditionalTM,SYN_TOLERANCE,save_json,masks_to_table,
)
from scripts.spt_continuous_hierarchy_pilot import audit_table

FOLDER=ROOT/'results/spt_local_kuramoto_tm/b_comparison'
PILOT=ROOT/'results/spt_local_kuramoto_tm/flow_mixture_dt0025.json'


def read(path): return json.loads(path.read_text())


def budget_deadline(folder=FOLDER):
    budget=read(folder/'budget.json')
    return time.monotonic()+budget['deadline_epoch_seconds']-time.time()


def load_model():
    pilot=read(PILOT)
    record=next(r for r in pilot['records'] if r['case']=='B')
    if not record['precision_passed']: raise ValueError('B pilot did not pass')
    model_payload=record['model']
    digest=hashlib.sha256(json.dumps(model_payload,sort_keys=True).encode()).hexdigest()
    return PeriodicConditionalTM.load(model_payload),record,digest


def full_table(folder=FOLDER,outer=1024,particles=256,scrambles=4):
    model,pilot,digest=load_model()
    path=folder/f'full_n{outer}_p{particles}.json'
    protocol={'case':'B','model_sha256':digest,'outer':outer,'particles':particles,
        'max_particles':256,'scrambles':scrambles,'seed_base':20261015,'seed_stride':1000,
        'syn_tolerance_nats':SYN_TOLERANCE,'target':'all nine future phases',
        'pilot_sha256':hashlib.sha256(PILOT.read_bytes()).hexdigest()}
    data=read(path) if path.exists() else {'protocol':protocol,'rows':[],'status':'running'}
    if data['protocol']!=protocol: raise ValueError('Full query cache protocol mismatch')
    deadline=budget_deadline(folder)
    started=time.monotonic(); prior=data.get('elapsed_seconds',0.)
    def save():
        data['elapsed_seconds']=prior+time.monotonic()-started
        save_json(path,data)
    try:
        for scramble in range(scrambles):
            row=next((r for r in data['rows'] if r['scramble']==scramble),None)
            if row is None:
                old=next((r for r in pilot['precision'] if
                          (r['outer'],r['particles'],r['scramble'])==(outer,particles,scramble)),None)
                row={'scramble':scramble,'ei_nats':{},'diagnostics':{}}
                if old:
                    row.update(ei_nats=dict(old['ei_nats']),diagnostics=dict(old['diagnostics']['queries']),
                               denominator=old['diagnostics']['denominator'])
                previous=ROOT/'results/spt_local_kuramoto_tm/full_table.json'
                if previous.exists() and (outer,particles)==(1024,256):
                    previous_payload=read(previous)
                    if previous_payload['protocol']['pilot_sha256']!=protocol['pilot_sha256']:
                        raise ValueError('Prior table came from another pilot')
                    prior_record=next(r for r in previous_payload['records'] if r['case']=='B')
                    prior_row=next((r for r in prior_record['replicates'] if r['scramble']==scramble),None)
                    if prior_row:
                        row['ei_nats'].update(prior_row['ei_nats'])
                        row['diagnostics'].update(prior_row['diagnostics'])
                        row['reused_prior_subsets']=len(prior_row['ei_nats'])
                data['rows'].append(row); save()
            if len(row['ei_nats'])==511: continue
            integral=FullTargetIntegrator(model,outer=outer,max_particles=256,
                seed=20261015+1000*scramble,deadline=deadline)
            for mask in range(1,512):
                if str(mask) in row['ei_nats']: continue
                values,_,diagnostic=integral.information([mask],particles)
                row['ei_nats'][str(mask)]=values[mask]
                row['diagnostics'][str(mask)]=diagnostic['queries'][str(mask)]
                row['denominator']=diagnostic['denominator']
                save()
                if len(row['ei_nats'])%64==0:
                    print(json.dumps({'stage':'B full EI','outer':outer,'particles':particles,
                        'scramble':scramble,'completed_subsets':len(row['ei_nats'])}),flush=True)
        values=np.array([[r['ei_nats'][str(m)] for m in range(1,512)] for r in data['rows']])
        mean=values.mean(0); sem=values.std(0,ddof=1)/np.sqrt(scrambles)
        audit=audit_table(masks_to_table(dict(zip(range(1,512),mean))),SYN_TOLERANCE)
        replicate_audits=[audit_table(masks_to_table({int(k):v for k,v in r['ei_nats'].items()}),SYN_TOLERANCE)
                          for r in data['rows']]
        data.update(status='complete',mean_ei_nats={str(m):float(v) for m,v in zip(range(1,512),mean)},
            sem_ei_nats={str(m):float(v) for m,v in zip(range(1,512),sem)},
            audit=audit,replicate_audits=replicate_audits)
        all_audits=[audit,*replicate_audits]
        violations=sum(a['significant_violation_count'] for a in all_audits)
        if violations:
            data.update(status='significant_syn_violation',error=
                f"Syn minimum={min(a['minimum_syn_nats'] for a in all_audits)}, "
                f"threshold={-SYN_TOLERANCE}, affected={violations}")
        save(); return data
    except (TimeoutError,ArithmeticError,np.linalg.LinAlgError) as error:
        data.update(status='budget_exhausted' if isinstance(error,TimeoutError) else 'numerical_failure',
                    error=str(error)); save(); return data


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer',type=int,default=1024)
    parser.add_argument('--particles',type=int,default=256)
    args=parser.parse_args()
    result=full_table(outer=args.outer,particles=args.particles)
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}))
