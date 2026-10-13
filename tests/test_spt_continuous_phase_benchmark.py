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
