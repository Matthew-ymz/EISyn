"""Controls for the continuous oracle, including null and rejection behavior."""
import numpy as np
import pytest

from scripts.spt_continuous_phase_benchmark import (
    InformationOracle, PhaseKernel, SOURCES, channel_integral, counterfactuals,
    evaluate, score,
)
from scripts.spt_continuous_precision_diagnostic import density_prefixes, residual_audit
from scripts.spt_continuous_hierarchy_pilot import ContinuousMap, MarginalOracle


def test_independent_density_and_entropy_mi_agree_across_grids():
    values = [channel_integral(n) for n in (128,512,1024)]
    for item in values:
        assert abs(item['normalization_error']) < 1e-12
        assert item['direct_mi_nats'] == pytest.approx(item['weight_nats'],abs=1e-12)
        assert item['weight_nats'] > 0
    assert np.ptp([i['weight_nats'] for i in values]) < 1e-12


def test_pair_matched_counterfactuals_keep_distinct_higher_order_supports():
    for family, variants in counterfactuals().items():
        matrices = []
        for supports in variants:
            oracle = InformationOracle(PhaseKernel(supports))
            record = evaluate(oracle.info,oracle.future_mi)
            matrices.append(record['matrices_nats'])
            for method in ('spt','block_phi_r'):
                result = score(record['methods'][method],supports)
                assert result['clade_f1'] == 1
                assert result['upper_support_recall'] == 1
                assert result['correct_root']
        for matrix in matrices[1:]:
            for method in matrix:
                np.testing.assert_array_equal(matrix[method],matrices[0][method])
        if family=='C':
            assert all(np.count_nonzero(v)==0 for v in matrices[0].values())


def test_null_has_no_measured_pair_hierarchy_and_raw_roundoff_is_audited():
    oracle = InformationOracle(PhaseKernel(counterfactuals()['A'][0],kappa=0))
    record = evaluate(oracle.info,oracle.future_mi)
    assert abs(record['xi_nats']) < 1e-8
    assert record['audit']['significant_violation_count']==0
    for method in ('pair_phi_r_hac','pair_xi_hac','future_mi_hac'):
        assert record['methods'][method]['status']=='no_structure_information'
        assert not record['methods'][method]['clades']


def test_significant_syn_violation_stops_before_tree_output():
    def invalid_info(source,target=SOURCES):
        return -.1 if len(source)>1 else 0.
    with pytest.raises(ArithmeticError,match=r'Syn violation: minimum=.*threshold=.*affected_count='):
        evaluate(invalid_info,lambda i,j:0.)


def test_precision_prefixes_match_original_conditional_tm():
    large = MarginalOracle(ContinuousMap('A'),128,256,0)
    small = MarginalOracle(ContinuousMap('A'),64,64,0)
    np.testing.assert_array_equal(large.x[:64],small.x)
    np.testing.assert_array_equal(large.base[:64],small.base)
    for bits in (0,34,511):
        densities = density_prefixes(large,bits,((64,64),(128,256)))
        np.testing.assert_allclose(densities[64]['jackknife'][:64],small.marginal(bits),atol=1e-12)


def test_diagnostic_reports_native_threshold_and_affected_count():
    result = residual_audit({'failed':-.031,'roundoff':-.001,'boundary':-.03})
    assert result['status']=='invalid_syn_estimate'
    assert result['significant_violation_count']==1
    assert result['tolerance_negative_count']==2
    assert 'minimum=-0.031' in result['error']
    assert 'threshold=-0.03' in result['error']
    assert 'affected_count=1' in result['error']


def test_broadcast_tm_matches_independent_sufficient_statistic_information():
    from scripts.spt_continuous_broadcast_case import GaussianBroadcast, validate
    result = validate(GaussianBroadcast())
    assert result['joint_rank']==24
    assert result['minimum_joint_eigenvalue']>.36
    assert result['all_225_group_target_mi_max_error_nats']<1e-12
    assert result['conditional_density_max_error_nats']<1e-12


def test_broadcast_counterexample_survives_every_label_permutation():
    from scripts.spt_continuous_broadcast_case import run
    result = run()
    assert len(result['records'])==24
    for record in result['records']:
        assert record['audit']['partition_count']==25
        assert record['audit']['significant_violation_count']==0
        assert abs(record['audit']['closure_error_nats'])<1e-12
        assert record['methods']['spt']['clade_f1']==1
        assert record['methods']['block_phi_r']['clade_f1']==.5
        assert not record['methods']['block_phi_r']['correct_root']
        assert record['normalized_block_control']['clade_f1']==1
        assert record['normalized_block_control']['correct_root']
        assert record['methods']['spt']['upper_support_recall'] is None


def test_broadcast_wrong_root_has_a_strict_gap_and_normalization_repairs_it():
    from scripts.spt_continuous_broadcast_case import run
    candidates = run()['root_candidates']
    truth = next(c for c in candidates if set(c['left'])=={1,2} and set(c['right'])=={3,4})
    singleton = [c for c in candidates if min(len(c['left']),len(c['right']))==1]
    assert len(singleton)==4
    assert truth['syn_nats']==pytest.approx(0,abs=1e-12)
    for wrong in singleton:
        assert truth['block_phi_r_nats']-wrong['block_phi_r_nats']==pytest.approx(.212833907712842)
        assert wrong['syn_nats']>truth['syn_nats']+.1
    assert truth['normalized_block_phi_r']<min(c['normalized_block_phi_r'] for c in candidates if c is not truth)


def test_weighted_broadcast_sampler_tm_and_all_group_target_queries_agree():
    from scripts.spt_continuous_broadcast_case import GaussianBroadcast, WEIGHTED_GAINS, validate
    result = validate(GaussianBroadcast(WEIGHTED_GAINS))
    assert result['joint_rank']==24
    assert result['all_225_group_target_mi_max_error_nats']<1e-12
    assert result['conditional_density_max_error_nats']<1e-12


def test_weighted_broadcast_defeats_normalized_block_for_every_labeling():
    from scripts.spt_continuous_broadcast_case import run, WEIGHTED_GAINS
    result = run(gains=WEIGHTED_GAINS)
    assert len(result['all_block_candidates'])==25
    assert len(result['records'])==24
    for record in result['records']:
        assert record['audit']['significant_violation_count']==0
        assert abs(record['audit']['closure_error_nats'])<1e-12
        assert record['methods']['spt']['clade_f1']==1
        assert record['methods']['pair_xi_hac']['clade_f1']==1
        for method in ('block_phi_r','normalized_block_phi_r','pair_phi_r_hac'):
            assert record['methods'][method]['clade_f1']==0
            assert not record['methods'][method]['correct_root']
    candidates = result['root_candidates']
    truth = next(c for c in candidates if set(c['left'])=={1,2} and set(c['right'])=={3,4})
    balanced_wrong = next(c for c in candidates if set(c['left'])=={1,3} and set(c['right'])=={2,4})
    minimum = min(c['normalized_block_phi_r'] for c in candidates)
    assert truth['normalized_block_phi_r']==pytest.approx(.195039639387394)
    assert minimum==pytest.approx(.117673826777910)
    assert truth['normalized_block_phi_r']-minimum>.077
    # Even restricting candidates to balanced 2|2 splits does not rescue it.
    assert balanced_wrong['normalized_block_phi_r']<truth['normalized_block_phi_r']-.074
    assert truth['syn_nats']==pytest.approx(0,abs=1e-12)
    assert min(c['syn_nats'] for c in candidates if c is not truth)>.038


def test_normalized_phase_control_matches_independent_crossing_channel_counts():
    from scripts.plot_spt_continuous_refinement import normalized_phase_controls
    from scripts.spt import nontrivial_bipartitions
    # Use one identity-label record per variant: all architecture variants,
    # without relying on ignored result caches being present in the checkout.
    variants = counterfactuals()
    records = [{'case':case,'variant':i,'permutation_id':0,'hyperedges':supports,'methods':{}}
               for case in 'BC' for i,supports in enumerate(variants[case])]
    data = {'protocol':{'grid':512,'permutations':[SOURCES]},'records':records}
    for record in normalized_phase_controls(data):
        method = record['methods']['normalized_block_phi_r']
        assert method['correct_root']
        if record['case']=='C':
            assert method['clade_f1']==1
        # Independently, local EI cancels and cross-block MI is zero. The
        # objective is a common channel weight times crossing-edge count / min
        # size. Tied B cascade refinements need not recover every true clade.
        clades = {frozenset(c) for c in method['clades']}
        for group in [frozenset(SOURCES),*(c for c in clades if len(c)>2)]:
            children = [c for c in clades if c<group and not any(c<d<group for d in clades)]
            covered = set().union(*children) if children else set()
            children += [frozenset((i,)) for i in group-covered]
            assert len(children)==2
            def crossing_ratio(left,right):
                return sum(set(e)<=group and bool(set(e)&set(left)) and bool(set(e)&set(right))
                           for e in record['hyperedges'])/min(len(left),len(right))
            best = min(crossing_ratio(l,r) for l,r in nontrivial_bipartitions(tuple(sorted(group))))
            assert crossing_ratio(*children)==pytest.approx(best,abs=1e-12)


def test_nested_nine_node_extension_keeps_real_supports_and_strict_normalized_failure():
    from scripts.spt_nested_broadcast_case import run, SOURCES as NINE
    result = run()
    valid = result['validation']
    assert valid['joint_rank']==valid['joint_dimension']==252
    assert valid['source_target_size_queries']==511*9
    assert valid['tm_formula_max_error_nats']<1e-11
    assert valid['full_joint_max_error_nats']<1e-11
    assert valid['all_future_pairs_max_error_nats']<1e-11
    assert len(result['records'])==20
    assert result['records'][0]['permutation']==NINE
    assert len(result['root_candidates'])==255
    for record in result['records']:
        assert record['audit']['partition_count']==9330
        assert record['audit']['minimum_syn_nats']>0
        assert record['audit']['significant_violation_count']==0
        assert abs(record['audit']['closure_error_nats'])<1e-12
        assert record['methods']['spt']['clade_f1']==1
        assert record['methods']['spt']['upper_support_recall']==1
        for method in ('block_phi_r','normalized_block_phi_r'):
            assert record['methods'][method]['clade_f1']==0
            assert not record['methods'][method]['correct_root']
    truth = next(c for c in result['root_candidates'] if c['left']==(1,2,3,4,5) and c['right']==(6,7,8,9))
    minimum = min(c['normalized_block_phi_r'] for c in result['root_candidates'])
    assert truth['normalized_block_phi_r']-minimum>.09


def test_fitted_tm_uses_product_prior_and_consistent_full_joint_marginals():
    from scripts.spt_finite_data_tm import FittedConditionalTM
    rng=np.random.default_rng(1729)
    x=rng.normal(size=(4096,9,1))
    coefficient=np.diag(np.linspace(.1,.9,9)); coefficient[0,1]=.4
    y=(x.reshape(-1,9)@coefficient.T+.7*rng.normal(size=(4096,9))).reshape(-1,9,1)
    fitted=FittedConditionalTM.fit(x,y,domain='real')
    assert np.max(abs(fitted.coefficient-coefficient))<.06
    np.testing.assert_array_equal(fitted.density.covariance[:9,:9],np.eye(9))
    covariance=fitted.density.covariance
    for source,target in [((1,2),tuple(range(1,10))),((1,4),(2,3)),((1,2,8),(1,2,5))]:
        a=tuple(i-1 for i in source); b=tuple(9+i-1 for i in target)
        direct=.5*(fitted.logdet(covariance[np.ix_(a,a)])+fitted.logdet(covariance[np.ix_(b,b)])
                   -fitted.logdet(covariance[np.ix_(a+b,a+b)]))
        assert fitted.info(source,target)==pytest.approx(direct,abs=1e-12)


def test_phase_chart_density_jacobian_reproduces_independent_uniform_law():
    from scripts.spt_finite_data_tm import FittedConditionalTM,encode
    rng=np.random.default_rng(42)
    x=rng.uniform(0,2*np.pi,size=(256,9,1)); y=rng.uniform(0,2*np.pi,size=(256,9,1))
    fitted=FittedConditionalTM(np.zeros((9,9)),np.zeros(9),np.eye(9),256,'phase',1)
    result=fitted.heldout(x,y,np.full(256,-9*np.log(2*np.pi)))
    assert abs(result['excess_nll_nats'])<1e-12
    assert abs(result['full_source_log_ratio_nats'])<1e-12
    with pytest.raises(ValueError,match='no silent boundary clipping'):
        encode(np.array([[0.]]),'phase')


def test_finite_sample_fitted_tm_spt_pipeline_and_paired_phase_inputs():
    from scripts.spt_finite_data_tm import sample_dataset,analyze_fit
    a=sample_dataset('A',1024,256,77)
    result=analyze_fit('A',77,1024,*a)
    assert len(result['ei_nats'])==511
    assert result['audit']['partition_count']==9330
    assert result['audit']['significant_violation_count']==0
    assert abs(result['audit']['closure_error_nats'])<1e-12
    assert result['noise_minimum_eigenvalue']>0
    assert np.isfinite(result['heldout']['excess_nll_nats'])
    assert set(result['methods'])=={'spt','block_phi_r','normalized_block_phi_r','pair_phi_r_hac','pair_xi_hac','future_mi_hac'}
    b=sample_dataset('B',1024,256,77); c=sample_dataset('C',1024,256,77)
    np.testing.assert_array_equal(b[0],c[0])
    np.testing.assert_array_equal(b[2],c[2])
