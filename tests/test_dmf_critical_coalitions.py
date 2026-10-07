"""Scientific safeguards for exact membership and repeated occurrence."""
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from scripts.analyze_dmf_critical_coalitions import (
    candidate_arrays, holm, leakage_upper, occurrence_stats, paired_test,
)
from scripts.dmf_joint_readout import CommonTargetGame, audit_nonnegative
from scripts.run_dmf_critical_coalitions import standard_tree
from scripts.spt import nontrivial_bipartitions
from scripts.analyze_dmf_critical_coalitions_all93 import (
    members_mask, mask_members, subject_counts, rank_indices, M, METRICS,
)


def protocol():
    return dict(G=np.arange(41)*.1, seeds=[3,4,5], repeat_seed_count=2,
        low_grid_indices=[0,1,2,3],high_grid_indices=[37,38,39,40],
        windows={'p':dict(grid_indices=[10,11,12,13,14,15],required_hits=4,
                         transition={'interval':[1.2,1.3]})})


def test_members_are_exact_and_missing_is_not_zero():
    c=protocol(); bg={'thresholds':{'3':{'threshold_nats':.1}}}
    # A superset containing the triplet is not the triplet's occurrence.
    events={(0,1,2,3):{(0,12,0):4.},(0,1,2):{(0,12,0):.2,(0,12,1):.3}}
    v,raw,strong=candidate_arrays((0,1,2),events,c,['p'],bg)
    assert raw[0,12] and strong[0,12]
    assert np.isnan(v[0,12,2]) and np.isnan(v[0,0]).all()
    v,raw,strong=candidate_arrays((0,1,3),events,c,['p'],bg)
    assert np.isnan(v).all() and not raw.any() and not strong.any()


def test_background_and_seed_repetition_are_separate():
    c=protocol();events={(0,1,2):{(0,12,0):.2,(0,12,1):.09,(0,12,2):.09}}
    bg={'thresholds':{'3':{'threshold_nats':.1}}}
    _,raw,strong=candidate_arrays((0,1,2),events,c,['p'],bg)
    assert raw[0,12] and not strong[0,12]
    bg['thresholds']['3']['threshold_nats']=None
    _,raw,strong=candidate_arrays((0,1,2),events,c,['p'],bg)
    assert raw[0,12] and not strong.any()


def test_inclusive_six_point_window_requires_four_hits():
    c=protocol();r=np.zeros((1,41),bool);r[0,10:13]=True
    stats,*_=occurrence_stats(r,c,['p'])
    assert stats['critical']['count']==0
    r[0,13]=True;stats,*_=occurrence_stats(r,c,['p'])
    assert stats['critical']['count']==1 and stats['window_exclusive']['count']==1
    r[0,0]=True;stats,*_=occurrence_stats(r,c,['p'])
    assert stats['low_any_leak']['count']==1 and stats['low_stage']['count']==0
    assert stats['window_exclusive']['count']==0


def test_small_n_tree_matches_exhaustive_node_optima_and_closes():
    rng=np.random.default_rng(4);d=12
    coefficient=rng.normal(size=(d,d))
    cov=np.block([[np.eye(d),coefficient],
                  [coefficient.T,coefficient.T@coefficient+np.eye(d)]])
    with threadpool_limits(limits=1):
        game=CommonTargetGame(cov);t=standard_tree(game)
        for n in t['nodes']:
            residuals=[game.u(n['members'])-game.u(l)-game.u(r)
                       for l,r in nontrivial_bipartitions(n['members'])]
            assert n['syn_nats_raw']==pytest.approx(min(residuals),abs=1e-12)
        assert len(t['nodes'])==5 and abs(t['closure_error_nats'])<1e-10
        assert sum(n['syn_nats_raw'] for n in t['nodes'])==pytest.approx(game.u(range(6)))


def test_significant_negative_syn_fails_and_tiny_negative_is_recorded():
    with pytest.raises(ArithmeticError,match='affected=1'):
        audit_nonnegative([-.001],tolerance=1e-8)
    r=audit_nonnegative([-1e-9,1.],tolerance=1e-8)
    assert r['tolerance_negative_count']==1 and r['minimum_nats']==-1e-9


def test_inference_respects_subject_pairs_and_multiplicity():
    r=paired_test([True]*8,[False]*8)
    assert r['p_value']==pytest.approx(1/256) and r['difference']==1.
    assert np.allclose(holm([.02,.01,.04]),[.04,.03,.04])
    # Zero leaks in eight people does not establish a population rate <=10%.
    assert leakage_upper(0,8)>.1


def test_compact_identity_preserves_members_across_uint64_boundary():
    m=members_mask([99,64,63,0])
    assert mask_members(m)==[0,63,64,99]
    assert m!=members_mask([0,63,64])
    assert m==members_mask([0,63,64,99])
    with pytest.raises(ValueError):
        members_mask([0,0,1])
    with pytest.raises(ValueError):
        members_mask([100])


def test_relaxed_occurrence_does_not_replace_strict_or_seed_rule():
    c=protocol();a=np.zeros((2,41),np.uint8)
    a[0,10:14]=3  # seeds 0 and 1, four points
    a[1,10:13]=3  # above reference at three points
    a[:,15]=1     # a single seed is never repeated
    out=subject_counts(a,c,'p')
    assert out[0,M['critical_ge4']]==1
    assert out[1,M['critical_ge4']]==0
    assert out[1,M['critical_ge3']]==1 and out[1,M['critical_ge1']]==1
    assert out[1,M['exclusive_ge1']]==1 and out[1,M['exclusive_ge4']]==0
    a[:,0]=3
    out=subject_counts(a,c,'p')
    assert out[1,M['low_any']]==1 and out[1,M['low_stage_ge2']]==0
    assert out[1,M['outside_any']]==1 and out[1,M['exclusive_ge1']]==0
    assert out[1,M['critical_ge3_no_endpoints']]==0


def test_registry_counter_capacity_for_all_subjects_and_seeds():
    c=protocol();a=np.full((2,41),7,np.uint8)
    out=subject_counts(a,c,'p')
    total=out.astype(np.uint16)*93
    assert total[0,M['seed_node_count']]==93*41*3
    assert total[0,M['total_point_count']]==93*41
    assert total[0,M['critical_point_count']]==93*6
    assert total[0,M['outside_point_count']]==93*35
    assert total[0,M['exclusive_ge4']]==0


def test_rankings_expose_different_criteria_without_unsigned_wraparound():
    sets=[[0,1,2],[3,4,5],[6,7,8]]
    words=np.array([[members_mask(v),0] for v in sets],np.uint64)
    counts=np.zeros((3,2,len(METRICS)),np.uint16)
    counts[:,1,M['critical_ge4']]=[4,4,1]
    counts[:,1,M['critical_ge1']]=[4,5,8]
    counts[:,1,M['low_any']]=[1,0,0]
    counts[:,0,M['critical_ge4']]=[4,4,1]
    orders=np.array([3,3,3])
    assert rank_indices(words,orders,counts,'strict',limit=1)==[1]
    assert rank_indices(words,orders,counts,'relaxed',limit=1)==[2]
    counts[0,1,M['exclusive_ge4']]=2
    assert rank_indices(words,orders,counts,'exclusive',limit=1)==[0]
