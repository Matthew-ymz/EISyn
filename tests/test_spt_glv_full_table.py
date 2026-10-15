"""The full-table screen must preserve negative residual failures and coverage."""
import json

import pytest

from scripts import spt_glv_full_table as screen


@pytest.mark.parametrize('interaction,violates',[(.1,False),(-.02,True)])
def test_all_subsets_and_residual_audit(tmp_path,monkeypatch,interaction,violates):
    def ei(mask): return mask.bit_count()+interaction*((mask&3)==3)
    diagnostic={'ess_minimum':64.,'maximum_normalized_weight':.02,
                'max_projected_remaining_step_coordinate':0.}
    precision=[{'outer':1024,'particles':256,'scramble':s,'ei_nats':{'511':ei(511)},
                'diagnostics':{'queries':{'511':{'ess_minimum':None}},'denominator':diagnostic}}
               for s in range(4)]
    pilot=tmp_path/'pilot.json'
    pilot.write_text(json.dumps({'status':'pilot_complete','elapsed_compute_seconds':10.,
        'protocol':{'syn_tolerance_nats':.01},'records':[{'case':'A',
        'precision_passed':True,'model':{},'precision':precision}]}))
    class Integrator:
        def __init__(self,*a,**k): pass
        def information(self,masks,particles):
            return ({m:ei(m) for m in masks},{},{'denominator':diagnostic,
                    'queries':{str(m):diagnostic for m in masks}})
    monkeypatch.setattr(screen.RealConditionalTM,'load',lambda _:None)
    monkeypatch.setattr(screen,'RealFullTargetIntegrator',Integrator)
    monkeypatch.setattr(screen,'save_json',lambda *a:None)
    result=screen.run(pilot,tmp_path/'screen.json')
    record=result['records'][0]
    assert len(record['ei_nats'])==511
    assert record['audit']['partition_count']==9330
    assert bool(record['audit']['significant_violation_count'])==violates
    assert record['screen_passed']!=violates
    if violates:
        assert record['audit']['minimum_syn_nats']==pytest.approx(-.02)
        assert any('Syn minimum=' in s for s in record['failed_checks'])
    assert 'methods' not in record
    assert record['hierarchy_status'].startswith('not_evaluated')


def test_exploratory_tree_requires_passing_complete_screen():
    from scripts.analyze_spt_glv_full_table import analyze
    from scripts.spt_local_hyperedge_kuramoto import SUPPORTS
    values={str(m):float(m.bit_count()+sum(all(m>>(i-1)&1 for i in edge) for edge in SUPPORTS['A']))
            for m in range(1,512)}
    data={'protocol':{'syn_tolerance_nats':.01},'records':[
        {'case':'A','status':'screen_complete','screen_passed':True,'ei_nats':values,
         'replicates':[{'scramble':s,'ei_nats':values} for s in range(4)]},
        {'case':'B','status':'budget_exhausted'},
        {'case':'C','status':'screen_complete','screen_passed':False,'failed_checks':['Syn failure']}]}
    result=analyze(data)
    a,b,c=result['records']
    assert a['mean_table']['status']=='exploratory_only'
    assert a['mean_table']['metrics']['clade_f1']==1.
    assert a['all_replicate_clades_match_mean']
    assert b['status']==c['status']=='not_evaluated'


def test_accumulated_negative_xi_stops_tree_even_when_each_syn_is_small():
    from scripts.analyze_spt_glv_full_table import tree_record
    values={str(m):m.bit_count()-.004*(m.bit_count()-1) for m in range(1,512)}
    result=tree_record(values,'A',.01)
    assert result['audit']['significant_violation_count']==0
    assert result['audit']['minimum_xi_nats']==pytest.approx(-.032)
    assert result['status']=='significant_xi_violation'
    assert 'metrics' not in result
