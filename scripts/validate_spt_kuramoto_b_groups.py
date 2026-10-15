"""Refine all pair-PhiR affinities and coarse-competitive Block partitions."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.spt_kuramoto_b_comparison import FOLDER
from scripts.spt_kuramoto_b_group_comparison import Comparison,mask,nodes,scored
from scripts.spt_kuramoto_finite_tm import SYN_TOLERANCE,save_json
from scripts.spt import nontrivial_bipartitions


def cached_coarse_information(folder,outer=128):
    marginals=[]; prefixes=[]
    for r in range(4):
        items={}; lower={}
        with (folder/f'group_n{outer}_p256_r{r}.jsonl').open() as f:
            next(f)
            for line in f:
                row=json.loads(line)
                items[row['source'],row['target']]=float(np.mean(row['log_marginal']))
                lower[row['source'],row['target']]=float(np.mean(
                    row['diagnostics'].get('prefix_log_marginal',row['log_marginal'])))
        marginals.append(items)
        prefixes.append(lower)
    def info(a,b): return np.array([d[a,b]-d[0,b] for d in marginals])
    def phi(parent,l,r):
        whole=info(parent,parent); ll=info(l,l); rr=info(r,r)
        four=[info(a,b) for a in (l,r) for b in (l,r)]
        def prefix_info(a,b): return np.array([d[a,b]-d[0,b] for d in prefixes])
        pw=prefix_info(parent,parent); pl=prefix_info(l,l); pr=prefix_info(r,r)
        pf=[prefix_info(a,b) for a in (l,r) for b in (l,r)]
        return {'values':whole-ll-rr+np.min(four,axis=0),
                'estimate':float(whole.mean()-ll.mean()-rr.mean()+min(v.mean() for v in four)),
                'particle_check_estimate':float(pw.mean()-pl.mean()-pr.mean()+min(v.mean() for v in pf))}
    return phi


def run(folder=FOLDER):
    low=json.loads((folder/'group_results_n128_p256.json').read_text())
    if low['status']!='complete': return {'status':'coarse_group_stage_incomplete'}
    coarse_phi=cached_coarse_information(folder)
    pools={}; query_set=set()
    for name in ('block_phi_r','normalized_block_phi_r'):
        normalize=name.startswith('normalized')
        per_parent=[]
        for item in low['methods'][name]['selection_diagnostics']:
            parent=item['parent']; candidates=[]
            for left,right in nontrivial_bipartitions(nodes(parent)):
                l,r=mask(left),mask(right); divisor=min(len(left),len(right)) if normalize else 1
                item=coarse_phi(parent,l,r); values=item['values']/divisor
                candidates.append((item['estimate']/divisor,l,r,values,
                    abs(item['estimate']-item['particle_check_estimate'])/divisor))
            candidates.sort(key=lambda c:(c[0],nodes(c[1]),nodes(c[2])))
            best=candidates[0]; chosen=[]
            for index,candidate in enumerate(candidates):
                difference=candidate[3]-best[3]; sem=float(difference.std(ddof=1)/2)
                if index<2 or candidate[0]-best[0]<=8*sem+candidate[4]+best[4]:
                    chosen.append((candidate[1],candidate[2]))
            per_parent.append({'parent':parent,'coarse_selected':[best[1],best[2]],
                'candidates':chosen,'total_candidates':len(candidates)})
            query_set.add((parent,parent))
            for l,r in chosen:
                query_set.update([(l,l),(r,r),(l,r),(r,l)])
        pools[name]=per_parent
    output=folder/'group_refinement.json'
    result={'status':'running','protocol':{'outer':256,'informed_particles':512,
        'prior_only_particles':1024,'full_target_particles':256,'scrambles':4,
        'coarse_outer':128,'coarse_informed_particles':256,'query_count':len(query_set),
        'screening':'top two and within 8 paired numerical SEM plus observed particle-check changes at visited nodes; heuristic, not simultaneous confidence bounds',
        'physical_datasets':1},'candidate_pools':pools,'methods':{}}
    comparison=Comparison(folder,256,512); start=time.monotonic()
    try:
        for index,(a,b) in enumerate(sorted(query_set)):
            comparison.info(a,b)
            if (index+1)%32==0:
                print(json.dumps({'stage':'B group refinement','completed_queries':index+1,
                    'total_queries':len(query_set)}),flush=True)
        for name,parent_pools in pools.items():
            normalize=name.startswith('normalized'); checks=[]; all_raw=[]; selections={}
            for pool in parent_pools:
                parent=pool['parent']; candidates=[]
                for l,r in pool['candidates']:
                    item=comparison.phi(parent,l,r); values=item['values']; all_raw.extend(values); all_raw.append(item['estimate'])
                    divisor=min(l.bit_count(),r.bit_count()) if normalize else 1
                    candidates.append((item['estimate']/divisor,l,r,values/divisor))
                candidates.sort(key=lambda c:(c[0],nodes(c[1]),nodes(c[2])))
                best=candidates[0]; selections[parent]=candidates
                gap=candidates[1][3]-best[3] if len(candidates)>1 else None
                checks.append({'parent':parent,'coarse_selected':pool['coarse_selected'],
                    'refined_selected':[best[1],best[2]],
                    'matches_coarse':set(pool['coarse_selected'])=={best[1],best[2]},
                    'high_grid_candidates':len(candidates),'all_candidates':pool['total_candidates'],
                    'gap_objective':candidates[1][0]-best[0] if gap is not None else None,
                    'gap_sem_objective':float(gap.std(ddof=1)/2) if gap is not None else None,
                    'objective_units':'nats/node' if normalize else 'nats'})
            audit=Comparison.audit(all_raw)
            record={'selection_checks':checks,'audit':audit,
                'all_mean_choices_match_coarse':all(c['matches_coarse'] for c in checks)}
            def build(parent,replicate=None):
                if parent.bit_count()==1: return (),[]
                if parent not in selections: raise KeyError(parent)
                candidates=selections[parent]
                row=min(candidates,key=lambda c:((c[0] if replicate is None else c[3][replicate]),nodes(c[1]),nodes(c[2])))
                l,r=row[1:3]; _,lc=build(l,replicate); _,rc=build(r,replicate)
                clades=[nodes(c) for c in (l,r) if c.bit_count()>1]+lc+rc
                return (nodes(l),nodes(r)),clades
            if audit['significant_violation_count']:
                record.update(status='significant_phi_violation',error=
                    f"PhiR minimum={audit['minimum_raw_nats']}, threshold={-SYN_TOLERANCE}, affected={audit['significant_violation_count']}")
            else:
                try:
                    record.update(status='exploratory_refined_candidate_pool',metrics=scored(*build(511)))
                except KeyError as error:
                    record.update(status='new_parent_requires_expansion',missing_parent=int(error.args[0]))
                replicas=[]
                for r in range(4):
                    try: replicas.append(scored(*build(511,r)))
                    except KeyError: replicas.append({'status':'unresolved_new_parent'})
                record['numerical_replicate_metrics']=replicas
            result['methods'][name]=record
        comparison.result['status']='selected_query_validation_complete'
        comparison.save()
        result['proposal_audit']=comparison.result.get('proposal_audit')
        result['status']='complete'
    except (TimeoutError,ArithmeticError,np.linalg.LinAlgError,KeyError) as error:
        result.update(status='budget_exhausted' if isinstance(error,TimeoutError) else 'numerical_failure',error=str(error))
    result['elapsed_seconds']=time.monotonic()-start; save_json(output,result)
    return result


if __name__=='__main__':
    result=run()
    print(json.dumps({'status':result['status'],'elapsed_seconds':result.get('elapsed_seconds')}))
