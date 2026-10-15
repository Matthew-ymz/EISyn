"""Exploratory SPT only for complete, numerically passing EI screens.

No new data/model/integration. Keep this distinct from certified high-grid
and multi-data-seed validation, and from any baseline superiority claim.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.spt import SPTConfig,SPTNonnegativityError,build_spt_from_ei_table,flatten_nodes
from scripts.spt_continuous_hierarchy_pilot import audit_table
from scripts.spt_continuous_phase_benchmark import score
from scripts.spt_local_hyperedge_kuramoto import SOURCES,SUPPORTS


def tree_record(values,case,tolerance):
    table={tuple(i+1 for i in range(9) if int(mask)>>i&1):float(value)
           for mask,value in values.items()}
    if len(table)!=511: raise ValueError('No SPT from an incomplete EI table')
    audit=audit_table(table,tolerance)
    if audit['significant_violation_count']:
        return {'status':'significant_syn_violation','audit':audit}
    xi={s:v-sum(table[(i,)] for i in s) for s,v in table.items() if len(s)>1}
    audit['minimum_xi_nats']=min(xi.values())
    audit['xi_tolerance_negative_count']=sum(-tolerance<=v<0 for v in xi.values())
    audit['significant_xi_violation_count']=sum(v < -tolerance for v in xi.values())
    if audit['significant_xi_violation_count']:
        return {'status':'significant_xi_violation','audit':audit}
    try:
        tree=build_spt_from_ei_table(SOURCES,table,config=SPTConfig(syn_tolerance=tolerance))
    except SPTNonnegativityError as error:
        return {'status':'builder_nonnegativity_failed','error':str(error),'audit':audit}
    if abs(tree.closure_error)>tolerance:
        return {'status':'closure_failed','closure_error_nats':tree.closure_error,'audit':audit}
    nodes=flatten_nodes(tree.root)
    root=[c.sources for c in tree.root.children]
    clades=[n.sources for n in nodes if 1<len(n.sources)<9]
    result=score({'status':'resolved','root_split':root,'clades':clades},SUPPORTS[case])
    return {'status':'exploratory_only','metrics':result,'audit':audit,
        'closure_error_nats':tree.closure_error,
        'tolerance_zero_count':tree.audit.tolerance_zero_count,
        'nodes':[{'sources':n.sources,'children':[c.sources for c in n.children],
                  'xi_nats':n.xi_value,'syn_nats':n.syn_value} for n in nodes]}


def analyze(data):
    output={'status':'exploratory_only','pilot_sha256':data['protocol'].get('pilot_sha256'),
            'scope':'passing full 1024x256x4 screens; no high-grid/full-baseline/multiple-data-seed certification',
            'syn_tolerance_nats':data['protocol']['syn_tolerance_nats'],'records':[]}
    for record in data['records']:
        if record.get('status')!='screen_complete' or not record.get('screen_passed'):
            output['records'].append({'case':record['case'],'status':'not_evaluated',
                'reason':record['status'],'failed_checks':record.get('failed_checks',[])})
            continue
        mean=tree_record(record['ei_nats'],record['case'],output['syn_tolerance_nats'])
        replicates=[{'scramble':r['scramble'],**tree_record(r['ei_nats'],record['case'],output['syn_tolerance_nats'])}
                    for r in record['replicates']]
        result={'case':record['case'],'mean_table':mean,'integration_replicates':replicates}
        if mean['status']=='exploratory_only':
            expected={tuple(c) for c in mean['metrics']['clades']}
            result['all_replicate_clades_match_mean']=all(r['status']=='exploratory_only' and
                {tuple(c) for c in r['metrics']['clades']}==expected for r in replicates)
            expectedroot={tuple(c) for c in mean['metrics']['root_split']}
            result['all_replicate_roots_match_mean']=all(r['status']=='exploratory_only' and
                {tuple(c) for c in r['metrics']['root_split']}==expectedroot for r in replicates)
            primitive=[e for e in SUPPORTS[record['case']]
                       if not any(set(g)<set(e) for g in SUPPORTS[record['case']])]
            def evaluated_clades(metrics):
                return {tuple(c) for c in metrics['clades'] if not any(set(c)<set(e) for e in primitive)}
            expected_evaluated=evaluated_clades(mean['metrics'])
            result['all_replicate_evaluated_clades_match_mean']=all(r['status']=='exploratory_only' and
                evaluated_clades(r['metrics'])==expected_evaluated for r in replicates)
        output['records'].append(result)
    return output


if __name__=='__main__':
    directory=ROOT/'results/spt_local_glv_tm'
    result=analyze(json.loads((directory/'full_table.json').read_text()))
    (directory/'exploratory_spt.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'records':[{'case':r['case'],'status':r.get('status',r.get('mean_table',{}).get('status')),
        'metrics':r.get('mean_table',{}).get('metrics')} for r in result['records']]}))
