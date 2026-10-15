"""Analyze complete B EI tables; prepare truth-free numerical refinement queries."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.spt_kuramoto_b_comparison import FOLDER
from scripts.spt_kuramoto_finite_tm import SYN_TOLERANCE,save_json,masks_to_table
from scripts.spt import SPTConfig,build_spt_from_ei_table,flatten_nodes,nontrivial_bipartitions
from scripts.spt_continuous_hierarchy_pilot import audit_table
from scripts.spt_kuramoto_b_group_comparison import scored,mask,nodes
from scripts.spt_multilevel_parity_case import graph_tree


def audit_and_tree(values):
    table=masks_to_table({int(k):v for k,v in values.items()})
    if len(table)!=511: raise ValueError('No SPT from incomplete EI table')
    audit=audit_table(table,SYN_TOLERANCE)
    xi=[v-sum(table[(i,)] for i in s) for s,v in table.items() if len(s)>1]
    audit['minimum_xi_nats']=min(xi)
    audit['significant_xi_violation_count']=sum(v < -SYN_TOLERANCE for v in xi)
    if audit['significant_violation_count'] or audit['significant_xi_violation_count']:
        raise ArithmeticError(f"Syn/Ξ minimum={min(audit['minimum_syn_nats'],min(xi))}, "
            f"threshold={-SYN_TOLERANCE}, affected={audit['significant_violation_count']+audit['significant_xi_violation_count']}")
    tree=build_spt_from_ei_table(tuple(range(1,10)),table,config=SPTConfig(syn_tolerance=SYN_TOLERANCE))
    if abs(tree.closure_error)>SYN_TOLERANCE: raise ArithmeticError('SPT closure failed')
    records=flatten_nodes(tree.root)
    metrics=scored(tuple(c.sources for c in tree.root.children),[n.sources for n in records if 1<len(n.sources)<9])
    return {'metrics':metrics,'audit':audit,'closure_error_nats':tree.closure_error,
        'tolerance_zero_count':tree.audit.tolerance_zero_count,
        'nodes':[{'sources':n.sources,'children':[c.sources for c in n.children],
                  'syn_nats':n.syn_value,'xi_nats':n.xi_value} for n in records]}


def analyze(folder=FOLDER,outer=1024):
    path=folder/f'full_n{outer}_p256.json'
    data=json.loads(path.read_text())
    if data['status'] not in ('complete','significant_syn_violation'):
        return {'status':'waiting_for_full_table'}
    result={'status':'exploratory_single_dataset','methods':{},'protocol':data['protocol']}
    try:
        mean=audit_and_tree(data['mean_ei_nats'])
        replicates=[audit_and_tree(r['ei_nats']) for r in data['rows']]
        result['methods']['spt']={'status':'exploratory_numerical',**mean,
            'numerical_replicate_metrics':[r['metrics'] for r in replicates]}
        raw=np.array([[r['ei_nats'][str(m)] for m in range(1,512)] for r in data['rows']])
        means=raw.mean(0); masks=set([511,*[1<<i for i in range(9)]])
        diagnostics=[]
        for node in mean['nodes']:
            if len(node['sources'])<2: continue
            parent=mask(node['sources']); masks.add(parent)
            rows=[]
            for left,right in nontrivial_bipartitions(node['sources']):
                l,r=mask(left),mask(right)
                residual=raw[:,parent-1]-raw[:,l-1]-raw[:,r-1]
                rows.append((float(residual.mean()),l,r,residual))
            rows.sort(key=lambda r:(r[0],nodes(r[1]),nodes(r[2])))
            canonical={mask(c) for c in node['children']}
            best=next(row for row in rows if {row[1],row[2]}==canonical)
            selected=[]
            for row in rows:
                difference=row[3]-best[3]
                sem=float(difference.std(ddof=1)/2)
                if row is best or float(difference.mean())<=8*sem:
                    masks.update(row[1:3]); selected.append([row[1],row[2]])
            if len(rows)>1:
                runner_up=next(row for row in rows if row is not best)
                masks.update(runner_up[1:3]); gap=runner_up[3]-best[3]
            else: gap=None
            diagnostics.append({'parent':parent,'selected':[best[1],best[2]],'candidate_count':len(rows),
                'selected_syn_nats':best[0],'runner_up_gap_nats':float(gap.mean()) if gap is not None else None,
                'runner_up_gap_sem_nats':float(gap.std(ddof=1)/2) if gap is not None else None,
                'refinement_candidate_pool':selected})
        result['methods']['spt']['selection_diagnostics']=diagnostics
        result['refinement_source_masks']=sorted(masks)
        result['refinement_rule']='singletons, chosen nodes, top two splits, plus candidates within 8 paired numerical SEM of the coarse best; heuristic screening, not a simultaneous confidence bound'
        diagnostics=[d for r in data['rows'] for d in [r['denominator'],*r['diagnostics'].values()]
                     if d.get('ess_minimum') is not None]
        result['proposal_audit']={'minimum_ess':min(d['ess_minimum'] for d in diagnostics),
            'maximum_normalized_weight':max(d['maximum_normalized_weight'] for d in diagnostics),
            'maximum_remaining_newton_step_rad':max(d['max_remaining_step_rad'] for d in diagnostics)}
    except ArithmeticError as error:
        result.update(status='significant_nonnegativity_failure',error=str(error))
    save_json(folder/('full_analysis.json' if outer==1024 else f'full_analysis_n{outer}.json'),result)
    return result


if __name__=='__main__':
    result=analyze()
    print(json.dumps({'status':result['status'],'methods':{k:v.get('metrics') for k,v in result.get('methods',{}).items()},
        'refinement_query_count':len(result.get('refinement_source_masks',[]))}))
