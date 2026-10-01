import math

import numpy as np
import pytest

from scripts.analyze_dmf_unconstrained_spt import (
    ROIXiOracle, TOL, clades, make_selector, mass_distance, retained_core,
    rf_distance, weakest_split,
)
from scripts.spt import SPTConfig, SPTNode, TableXiOracle, build_spt, pairwise_syn_affinity


def test_roi_budget_removes_internal_ei_synergy_and_closes():
    # Two independent ROI, each with correlated E/I: all scalar TC is within ROI.
    cov=np.eye(4)
    cov[0,2]=cov[2,0]=0.7
    cov[1,3]=cov[3,1]=0.4
    oracle=ROIXiOracle(cov)
    scalar_tc=(np.log(np.diag(cov)).sum()-np.linalg.slogdet(cov)[1])/(2*math.log(2))
    assert scalar_tc>0.5
    assert oracle.xi((0,1))==pytest.approx(0,abs=1e-12)
    result=build_spt((0,1),oracle,config=SPTConfig(syn_tolerance=TOL))
    assert result.closure_error==pytest.approx(0,abs=1e-12)


def test_block_oracle_agrees_with_independent_logdet_and_exact_core():
    rng=np.random.default_rng(10)
    base=rng.normal(size=(8,8))
    cov=base@base.T+np.eye(8)
    oracle=ROIXiOracle(cov)
    blocks=[(i,i+4) for i in range(4)]
    expected=(sum(np.linalg.slogdet(cov[np.ix_(b,b)])[1] for b in blocks)-np.linalg.slogdet(cov)[1])/(2*math.log(2))
    assert oracle.xi(range(4))==pytest.approx(expected)
    result=build_spt(tuple(range(4)),oracle)
    weak=weakest_split(oracle,tuple(range(4)))
    assert weak['bits_raw']==pytest.approx(result.root.syn_value)
    assert weak['candidate_count']==7


def test_invalid_covariance_fails_without_projection():
    with pytest.raises(np.linalg.LinAlgError):
        ROIXiOracle(np.array([[1,2],[2,1]]))
    with pytest.raises(ValueError):
        ROIXiOracle(np.array([[1,np.nan],[np.nan,1]]))


def leaf(i):
    return SPTNode((i,),0,0,2,'leaf')


def test_rf_is_invariant_to_branch_orientation_and_tv_includes_root():
    ab=SPTNode((0,1),1,1,1,'exact',(leaf(0),leaf(1)))
    abc=SPTNode((0,1,2),2,1,0,'exact',(ab,leaf(2)))
    flipped=SPTNode((0,1,2),2,1,0,'exact',(leaf(2),ab))
    bc=SPTNode((1,2),1,1,1,'exact',(leaf(1),leaf(2)))
    changed=SPTNode((0,1,2),2,1,0,'exact',(leaf(0),bc))
    assert rf_distance(abc,flipped)==0
    assert mass_distance(abc,flipped)==0
    assert rf_distance(abc,changed)==1
    assert mass_distance(abc,changed)==0.5
    assert clades(abc)=={(0,1)}


def test_core_follows_information_instead_of_larger_branch():
    high=SPTNode((0,1),4,4,1,'exact',(leaf(0),leaf(1)))
    low=SPTNode((2,3,4),1,1,1,'exact',(leaf(2),SPTNode((3,4),0,0,2,'exact',(leaf(3),leaf(4)))))
    root=SPTNode((0,1,2,3,4),6,1,0,'exact',(high,low))
    assert retained_core(root,3)['members']==[0,1]
    assert retained_core(root,1)['members']==[]


def test_search_augmentation_is_nested_reproducible_and_has_every_singleton():
    cov=np.eye(24)
    cov+=0.03*np.ones((24,24))
    oracle=ROIXiOracle(cov)
    affinity,_=pairwise_syn_affinity(oracle,12,tolerance=TOL)
    sources=tuple(range(12))
    _,base=make_selector(affinity,oracle,0,[])(sources)
    _,expanded=make_selector(affinity,oracle,64,[])(sources)
    _,again=make_selector(affinity,oracle,64,[])(sources)
    assert set(base)<=set(expanded)
    assert expanded==again
    singleton_sides={s for split in expanded for s in split if len(s)==1}
    assert singleton_sides=={(i,) for i in sources}


def test_exact_core_audit_reports_all_significant_violations():
    oracle=TableXiOracle({(0,):0,(1,):0,(2,):0,
                         (0,1):1,(0,2):1,(1,2):1,(0,1,2):0},(0,1,2))
    from scripts.spt import SPTNonnegativityError
    with pytest.raises(SPTNonnegativityError,match='affected_count=3'):
        weakest_split(oracle,(0,1,2))


def test_paired_pilot_sources_are_reproducible_and_seed_specific():
    from scripts.run_dmf_paired_spt_pilot import paired_sources
    a,b=paired_sources(4,12,3)
    aa,bb=paired_sources(4,12,3)
    other,_=paired_sources(5,12,3)
    assert np.array_equal(a,aa) and np.array_equal(b,bb)
    assert not np.array_equal(a,other)
    assert np.all((a>=.3)&(a<=.7)) and np.all((b>=.3)&(b<=.7))


def test_paired_rollout_uses_identical_noise_at_different_G():
    from types import SimpleNamespace
    from scripts.run_dmf_paired_spt_pilot import noise_seed,paired_sources
    from scripts.run_dmf_diffusive_fullstate_control import rollout
    class RecordingRng:
        def __init__(self):
            self.rng=np.random.default_rng(noise_seed(4)); self.draws=[]
        def standard_normal(self,shape):
            value=self.rng.standard_normal(shape); self.draws.append(value.copy()); return value
    dmf=SimpleNamespace(transfer_function=lambda value,**kwargs:value)
    params=SimpleNamespace(dt=.001,w_e=1,i0=.3,w_plus=1,j_nmda=.1,w_i=1,
                           gain_e=1,threshold_e=0,shape_e=1,gain_i=1,threshold_i=0,shape_i=1,
                           sigma=.01,tau_e=.1,gamma_e=1,tau_i=.1)
    se,si=paired_sources(4,12,3)
    rng_a,rng_b=RecordingRng(),RecordingRng()
    results=[]
    for g,rng in [(1.2,rng_a),(1.4,rng_b)]:
        results.append(rollout(dmf,se,si,connectivity=np.ones((3,3)),coupling_g=g,
                               j_fic=np.ones(3),parameters=params,mode='direct',state_boundary='none',horizon=4,rng=rng))
    assert len(rng_a.draws)==len(rng_b.draws)==8
    assert all(np.array_equal(a,b) for a,b in zip(rng_a.draws,rng_b.draws))
    assert not np.array_equal(results[0][0],results[1][0])


def test_wide_grid_spans_whole_range_and_preserves_every_coarse_point():
    from scripts.run_dmf_unconstrained_spt_sweep import scan_grid
    gs=scan_grid()
    assert len(gs)==55
    assert gs[0]==0 and gs[-1]==3
    assert set(np.round(np.arange(0,3.01,.1),2))<=set(gs)
    assert set(np.round(np.arange(1.1,1.701,.02),2))<=set(gs)


def test_tangent_operator_agrees_with_existing_full_drift_jacobian():
    from scripts.check_dmf_spt_dynamics import drift_and_tangent
    from scripts.run_dmf_schaefer100_critical_horizon import drift_jacobian,rates_and_drift
    from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module
    dmf=load_dmf_module();p=dmf.DMFParameters(dt=.001)
    rng=np.random.default_rng(42)
    se=rng.uniform(.3,.7,3); si=rng.uniform(.3,.7,3)
    sc=rng.uniform(0,.2,(3,3));jf=np.ones(3);g=1.3;v=rng.normal(size=6)
    dx,dy,vx,vy,re=drift_and_tangent(dmf,se,si,v[:3],v[3:],sc,g,jf,p)
    jac=drift_jacobian(dmf,se=se,si=si,connectivity=sc,coupling_g=g,j_fic=jf,parameters=p)
    expected_re,_,expected_dx,expected_dy=rates_and_drift(dmf,se,si,connectivity=sc,coupling_g=g,j_fic=jf,parameters=p)
    np.testing.assert_allclose(np.concatenate((vx,vy)),jac@v,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(dx,expected_dx);np.testing.assert_allclose(dy,expected_dy)
    np.testing.assert_allclose(re,expected_re)


def test_exact_membership_statistics_keep_pairs_and_correct_family():
    from scripts.analyze_dmf_spt_membership import exact_paired_p,mcnemar_exact,bh_q,membership_statistics
    assert exact_paired_p(np.ones(8))==pytest.approx(2/256)
    assert exact_paired_p(np.zeros(8))==1
    assert mcnemar_exact(np.zeros(8),np.ones(8))==(8,0,2/256)
    assert mcnemar_exact(np.ones(8),np.ones(8))==(0,0,1)
    np.testing.assert_allclose(bh_q([.01,.04,.02]),[.03,.04,.03])
    from scripts.run_dmf_unconstrained_spt_sweep import scan_grid
    gs=scan_grid();chosen=np.zeros((8,55,100),dtype=bool)
    chosen[:,:,0]=True
    gi=int(np.flatnonzero(np.isclose(gs,1.8))[0]);chosen[:,gi,0]=False;chosen[:,gi,1]=True
    stats=membership_statistics(gs,np.arange(3,11),chosen)
    assert stats['same_center_contrast']['small_changed_ROI']==[0]*8
    assert stats['same_center_contrast']['large_changed_ROI']==[2]*8
    assert len(stats['ROI_anchor_tests'])==300
    assert stats['adjacent_entering_by_seed'][0][gi-1]==1
    assert stats['adjacent_leaving_by_seed'][0][gi-1]==1


def test_sample_increase_retains_each_original_noise_draw_and_source():
    from scripts.validate_dmf_spt_sample_size import NestedNoise,nested_sources
    from scripts.run_dmf_paired_spt_pilot import paired_sources,noise_seed
    se,si=nested_sources(4,count=6,dimension=3)
    original_e,original_i=paired_sources(4,6,3)
    np.testing.assert_array_equal(se[:6],original_e);np.testing.assert_array_equal(si[:6],original_i)
    rng=NestedNoise(4,prefix_count=6);original=np.random.default_rng(noise_seed(4))
    for _ in range(10):
        np.testing.assert_array_equal(rng.standard_normal((12,3))[:6],original.standard_normal((6,3)))
