"""Refine numerically competitive B SPT source queries."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.spt_kuramoto_b_comparison import FOLDER,PILOT,load_model,budget_deadline
from scripts.spt_kuramoto_finite_tm import FullTargetIntegrator,save_json,SYN_TOLERANCE
from scripts.spt_kuramoto_b_group_comparison import nodes,mask,scored
from scripts.spt import nontrivial_bipartitions
from scripts.spt_multilevel_parity_case import graph_tree


def audit_xi_queries(rows,desired):
    values=[]
    for a in desired:
        if a.bit_count()<2: continue
        repeats=[row['ei_nats'][str(a)]-sum(row['ei_nats'][str(1<<i)]
                    for i in range(9) if a>>i&1) for row in rows]
        values.extend(repeats); values.append(float(np.mean(repeats)))
    raw=np.asarray(values)
    return {'minimum_raw_xi_nats':float(raw.min()),'native_tolerance_nats':SYN_TOLERANCE,
        'tolerance_negative_count':int(np.count_nonzero((raw<0)&(raw>=-SYN_TOLERANCE))),
        'significant_violation_count':int(np.count_nonzero(raw < -SYN_TOLERANCE)),
        'scope':'all registered high-grid source coalitions; all repeats and means'}


def run(folder=FOLDER):
    low=json.loads((folder/'full_analysis.json').read_text())
    if not low.get('refinement_source_masks'):
        return {'status':'no_valid_coarse_tree'}
    desired=low['refinement_source_masks']
    model,pilot,digest=load_model(); deadline=budget_deadline(folder)
    path=folder/'full_refined_n4096_p256.json'
    protocol={'model_sha256':digest,'outer':4096,'particles':256,'scrambles':4,'source_masks':desired,
              'scope':'numerically competitive SPT splits; not all high-grid 9330 splits'}
    data=json.loads(path.read_text()) if path.exists() else {'protocol':protocol,'rows':[],'status':'running'}
    if data['protocol']!=protocol: raise ValueError('Refinement cache protocol mismatch')
    start=time.monotonic(); previous=data.get('elapsed_seconds',0.)
    def save():
        data['elapsed_seconds']=previous+time.monotonic()-start; save_json(path,data)
    try:
        for scramble in range(4):
            row=next((r for r in data['rows'] if r['scramble']==scramble),None)
            if row is None:
                old=next(r for r in pilot['precision'] if
                         (r['outer'],r['particles'],r['scramble'])==(4096,256,scramble))
                row={'scramble':scramble,'ei_nats':dict(old['ei_nats']),
                     'diagnostics':dict(old['diagnostics']['queries']),'denominator':old['diagnostics']['denominator']}
                data['rows'].append(row); save()
            integral=FullTargetIntegrator(model,outer=4096,max_particles=256,
                seed=20261015+1000*scramble,deadline=deadline)
            for a in desired:
                if str(a) in row['ei_nats']: continue
                value,_,diagnostic=integral.information([a],256)
                row['ei_nats'][str(a)]=value[a]
                row['diagnostics'][str(a)]=diagnostic['queries'][str(a)]
                save()
            print(json.dumps({'stage':'B refined full-target','scramble':scramble,'queries':len(row['ei_nats'])}),flush=True)
        data['xi_audit']=audit_xi_queries(data['rows'],desired)
        if data['xi_audit']['significant_violation_count']:
            data.update(status='significant_xi_violation',error=
                f"Xi minimum={data['xi_audit']['minimum_raw_xi_nats']}, threshold={-SYN_TOLERANCE}, affected={data['xi_audit']['significant_violation_count']}")
            save(); return data
        available=set(desired); checks=[]; residuals=[]
        for node in low['methods']['spt']['selection_diagnostics']:
            parent=node['parent']; choices=[]
            for left,right in nontrivial_bipartitions(nodes(parent)):
                l,r=mask(left),mask(right)
                if l not in available or r not in available: continue
                syn=np.array([z['ei_nats'][str(parent)]-z['ei_nats'][str(l)]-z['ei_nats'][str(r)] for z in data['rows']])
                residuals.extend(syn); choices.append((float(syn.mean()),l,r,syn))
            choices.sort(key=lambda z:(z[0],nodes(z[1]),nodes(z[2])))
            best=choices[0]; gap=choices[1][3]-best[3] if len(choices)>1 else None
            checks.append({'parent':parent,'coarse_selected':node['selected'],'refined_selected':[best[1],best[2]],
                'matches_coarse':set(node['selected'])=={best[1],best[2]},
                'evaluated_high_grid_candidates':len(choices),'total_candidates':node['candidate_count'],
                'runner_up_gap_nats':float(gap.mean()) if gap is not None else None,
                'runner_up_gap_sem_nats':float(gap.std(ddof=1)/2) if gap is not None else None})
        raw=np.asarray(residuals)
        audit={'native_tolerance_nats':SYN_TOLERANCE,'minimum_raw_syn_nats':float(raw.min()),
            'tolerance_negative_count':int(np.count_nonzero((raw<0)&(raw>=-SYN_TOLERANCE))),
            'significant_violation_count':int(np.count_nonzero(raw < -SYN_TOLERANCE)),
            'scope':'available high-grid candidate pools only'}
        data.update(status='complete',spt_selection_checks=checks,syn_audit=audit,
                    all_refined_spt_choices_match=all(c['matches_coarse'] for c in checks))
        if audit['significant_violation_count']:
            data.update(status='significant_syn_violation',error=
                f"Syn minimum={raw.min()}, threshold={-SYN_TOLERANCE}, affected={audit['significant_violation_count']}")
    except (TimeoutError,ArithmeticError,np.linalg.LinAlgError) as error:
        data.update(status='budget_exhausted' if isinstance(error,TimeoutError) else 'numerical_failure',error=str(error))
    save(); return data


if __name__=='__main__':
    result=run()
    print(json.dumps({'status':result['status'],'elapsed_seconds':result.get('elapsed_seconds')}))
