"""Read-only-cache sensitivity audit; no model fit or new EI integration.

Delete-block comparisons of Sobol points are perturbation diagnostics, not
independent replications, bootstrap confidence intervals, or a convergence proof.
The original .01-nat tolerance and original acceptance result remain unchanged.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.kuramoto_six_topology_core import mask_of,recovery,save,spt_from_ei
from scripts.spt import nontrivial_bipartitions

FOLDER=ROOT/'results/kuramoto_six_topology_search/generic_fourier_x4/B_2_seed0'
SUPPORTS=[[1,2,3],[1,2,3,4],[5,6]]
NODES=((1,2,3,4,5,6),(1,2,3,4),(1,2,3))


def cut_ranking(ei,sources):
    cuts=list(nontrivial_bipartitions(sources))
    score=np.array([ei[mask_of(left)-1]+ei[mask_of(right)-1] for left,right in cuts])
    order=np.argsort(-score,kind='stable');a,b=order[:2]
    result=dict(sources=list(sources),best=[list(s) for s in cuts[a]],runner_up=[list(s) for s in cuts[b]],
                margin_nats=float(score[a]-score[b]))
    expected={NODES[0]:((1,2,3,4),(5,6)),NODES[1]:((1,2,3),(4,))}.get(tuple(sources))
    if expected:
        index=cuts.index(expected)
        result['planted_cut_margin_nats']=float(score[index]-np.delete(score,index).max())
    return result


def row(ei,**meta):
    spt=spt_from_ei(dict(zip(range(1,64),ei)))
    return dict(**meta,status=spt['status'],audit=spt['audit'],
                recovery=recovery(spt['tree'],SUPPORTS) if spt['tree'] else None,
                cuts=[cut_ranking(ei,s) for s in NODES])


def audit():
    output=dict(kind='post-hoc cached numerical sensitivity audit',new_simulations=0,new_models=0,new_integrations=0,
                tolerance_nats=.01,original_acceptance_unchanged=True,
                caveat='Delete blocks of correlated Sobol points are sensitivity diagnostics, not independent trials or confidence intervals.',
                manuscript=dict(parent='P6UJCVG8',main='DXGC7JEA',si='MWIWKSVG',rechecked='2026-10-17',
                                sections='Methods5-12; SI S5/S7',version='no explicit draft date or version'),
                provenance=[],individual=[],level_means=[],delete_block=[])
    high=[]
    for n,p in ((512,256),(2048,512)):
        arrays=[]
        for seed in (880,1880):
            path=FOLDER/f'ei_N{n}_P{p}_R{seed}.npz'
            a=np.load(path)['pointwise_ei'];assert a.shape==(63,n) and np.isfinite(a).all()
            output['provenance'].append(dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            arrays.append(a);output['individual'].append(row(a.mean(1),outer=n,particles=p,scramble=seed))
        arrays=np.stack(arrays)
        output['level_means'].append(row(arrays.mean((0,2)),outer=n,particles=p))
        if n==2048:high=arrays
    for block in range(16):
        ids=np.arange(128*block,128*(block+1))
        for r,seed in enumerate((880,1880)):
            deleted=np.delete(high[r],ids,axis=1).mean(1)
            output['delete_block'].append(row(deleted,scope='individual',scramble=seed,deleted_block=block))
            combined=.5*(deleted+high[1-r].mean(1))
            output['delete_block'].append(row(combined,scope='combined_single_scramble_deleted',scramble=seed,deleted_block=block))
        output['delete_block'].append(row(np.delete(high,ids,axis=2).mean((0,2)),scope='combined_both_scrambles_deleted',deleted_block=block))
    summaries=[]
    for scope,seed in (('individual',880),('individual',1880),('combined_single_scramble_deleted',None),('combined_both_scrambles_deleted',None)):
        rows=[r for r in output['delete_block'] if r['scope']==scope and (seed is None or r.get('scramble')==seed)]
        summaries.append(dict(scope=scope,scramble=seed,count=len(rows),valid_count=sum(r['status']=='valid' for r in rows),
                              full_module_recovery_count=sum(r['recovery'] is not None and r['recovery']['module_f1']==1 for r in rows),
                              minimum_123_retention_margin_nats=min(r['cuts'][1]['planted_cut_margin_nats'] for r in rows)))
    output['summary']=summaries
    save(FOLDER/'candidate_stability_audit.json',output)
    return output


if __name__=='__main__':
    result=audit();print(json.dumps(result['summary'],indent=2))
