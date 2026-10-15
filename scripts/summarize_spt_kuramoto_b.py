"""Collect B scores with their actual numerical-evidence status; never fill NA."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.spt_kuramoto_b_comparison import FOLDER
from scripts.spt_kuramoto_finite_tm import save_json

ORDER=('spt','block_phi_r','normalized_block_phi_r')


def read(path): return json.loads(path.read_text()) if path.exists() else {}


def same_tree(a,b):
    return {tuple(x) for x in a.get('clades',[])}=={tuple(x) for x in b.get('clades',[])}


def summarize(folder=FOLDER):
    full=read(folder/'full_analysis.json')
    group=read(folder/'group_results_n128_p256.json')
    high_full=read(folder/'full_refined_n4096_p256.json')
    full_high=read(folder/'full_analysis_n4096.json')
    high_group=read(folder/'group_refinement.json')
    result={'status':'in_progress','physical_dataset_count':1,'physical_dataset_seed':0,
        'has_population_significance_test':False,'numerical_repetitions':4,
        'score_semantics':'support/clade F1 of tree from averaged information; separate numerical replicate trees',
        'methods':{},'higher_resolution_scope':'SPT complete high-grid table if finished; Block high-grid competitive-candidate checks if finished; ranking stability reported separately'}
    for name in ORDER:
        source=full.get('methods',{}).get(name) or group.get('methods',{}).get(name) or {}
        record={'status':source.get('status','pending'),'source_resolution':'coarse',
            'precision_status':'not_certified','metrics':source.get('metrics'),
            'numerical_replicate_metrics':source.get('numerical_replicate_metrics',[])}
        if name=='spt' and record['metrics']:
            replicas=record['numerical_replicate_metrics']
            record['precision_status']='coarse_replicates_stable' if all(same_tree(r,record['metrics']) for r in replicas) else 'coarse_numerically_unstable'
            if high_full.get('status')=='complete':
                record['refined_selection_checks']=high_full.get('spt_selection_checks')
                record['precision_status']='candidate_checks_match_coarse' if high_full.get('all_refined_spt_choices_match') else 'choice_changed_on_refinement'
            high=full_high.get('methods',{}).get('spt',{})
            if high.get('metrics'):
                record.update(status=high['status'],metrics=high['metrics'],
                    numerical_replicate_metrics=high.get('numerical_replicate_metrics',[]),
                    source_resolution='complete N4096 P256 table, 511 subsets')
                stable=all(same_tree(r,high['metrics']) for r in record['numerical_replicate_metrics'])
                record['precision_status']='high_grid_replicates_stable' if stable else 'high_grid_numerically_unstable'
                record['coarse_tree_matches_high_grid']=same_tree(source['metrics'],high['metrics'])
                record['selection_diagnostics']=high.get('selection_diagnostics')
        if name in ('block_phi_r','normalized_block_phi_r') and high_group.get('status')=='complete':
            high=high_group.get('methods',{}).get(name,{})
            if high.get('status')=='significant_phi_violation':
                record.update(status=high['status'],metrics=None,
                    numerical_replicate_metrics=[],precision_status='failed_refinement_nonnegativity_audit')
            if high.get('metrics'):
                record.update(status=high['status'],metrics=high['metrics'],
                    numerical_replicate_metrics=high.get('numerical_replicate_metrics',[]),
                    source_resolution='refined group queries / competitive candidates',precision_status='refined_numerical_candidate_checks')
                if source.get('metrics') and not same_tree(source['metrics'],high['metrics']):
                    record['precision_status']='tree_changed_on_refinement'
        valid_replicas=[r for r in record['numerical_replicate_metrics'] if 'clade_f1' in r]
        if valid_replicas:
            record['numerical_f1_range']=[min(r['clade_f1'] for r in valid_replicas),max(r['clade_f1'] for r in valid_replicas)]
            record['complete_numerical_replicate_count']=len(valid_replicas)
        result['methods'][name]=record
    if all(result['methods'][name]['metrics'] is not None for name in ORDER):
        result['status']='three_scores_available_single_dataset'
    spt=result['methods']['spt']['metrics']
    if spt:
        result['observed_spt_minus_baseline_f1']={name:spt['clade_f1']-record['metrics']['clade_f1']
            for name,record in result['methods'].items() if name!='spt' and record['metrics']}
    save_json(folder/'comparison_summary.json',result)
    return result


if __name__=='__main__':
    data=summarize()
    print(json.dumps({'status':data['status'],'scores':{k:(v['metrics'] or {}).get('clade_f1') for k,v in data['methods'].items()}}))
