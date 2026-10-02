import numpy as np
import pytest

from scripts.dmf_joint_readout import (CommonTargetGame,audit_nonnegative,exact_spt,
    exact_shapley,label_statistics,fit_decoder,posterior,decoder_metrics,ld)
from scripts.run_dmf_joint_readout import analytic_controls


def test_common_target_scalar_roi_budgets_and_exact_organization():
    rng = np.random.default_rng(51); n = 4
    transition = rng.normal(size=(2*n,2*n))*.4
    prior = np.eye(2*n)
    future = transition.T@transition+np.eye(2*n)
    cov = np.block([[prior,transition],[transition.T,future]])
    game = CommonTargetGame(cov)
    for s in ((0,),(0,2),tuple(range(n))):
        _,d = game.dims(s); y = tuple(range(2*n,4*n))
        direct = .5*(ld(cov[np.ix_(d,d)])+ld(future)-ld(cov[np.ix_(d+y,d+y)]))
        assert game.ei(s) == pytest.approx(direct,abs=1e-12)
        assert game.v(s) == pytest.approx(game.u(s)+game.local_xi[list(s)].sum(),abs=1e-12)
    total = game.totals()
    assert total['closure_error_nats'] < 1e-12
    assert exact_spt(game,range(n))['closure_error_nats'] < 1e-12
    shapley = exact_shapley(game,range(n))
    assert sum(shapley['cross_shapley_nats']) == pytest.approx(total['cross_roi_nats'],abs=1e-12)
    audit_nonnegative(game.audited)


def test_nonfactorized_prior_rejected_and_negative_syn_fails():
    cov = np.eye(8); cov[0,1] = cov[1,0] = .1
    with pytest.raises(ValueError,match='factorized'):
        CommonTargetGame(cov)
    raw = np.array([-.5e-8,1.])
    assert audit_nonnegative(raw)['tolerance_negative_count'] == 1
    assert raw[0] < 0
    with pytest.raises(ArithmeticError,match='affected=2'):
        audit_nonnegative([-2e-8,-3e-8])


def test_exact_noisy_xor_detects_multisource_without_claiming_recovery():
    for r in analytic_controls():
        n = r['n']
        assert sum(r['shapley']['cross_shapley_nats']) == pytest.approx(r['xi_nats'],abs=1e-12)
        assert r['pair_syn_sum_nats'] == pytest.approx(r['xi_nats'] if n==2 else 0.,abs=1e-12)
        assert r['spt']['mass_by_roi_order'][n] == pytest.approx(r['xi_nats'],abs=1e-12)
        assert r['bayes_joint_mode_accuracy'] <= .5


def test_nested_decoder_contains_exact_pairwise_joint_equivalence_at_k2():
    assert np.array_equal(label_statistics(2,2),label_statistics(2,4))
    rng = np.random.default_rng(61)
    x = np.c_[rng.normal(size=(512,3)),np.ones(512)]
    bits = np.c_[x[:,0]>0,x[:,1]>0].astype(int)
    codes = bits @ np.array([1,2])
    for order in (1,2):
        model = fit_decoder(x,codes,2,order,.001)
        prob = posterior(model,x)
        assert np.allclose(prob.sum(1),1.)
        assert decoder_metrics(prob,codes,2)['mode_accuracy'] > .9
        if order == 1:
            # Factorized log-linear model is an exact product distribution.
            assert np.allclose(prob[:,0]*prob[:,3],prob[:,1]*prob[:,2],atol=1e-12)


def test_spt_maximizes_retained_child_information():
    class Game:
        def u(self,s):
            s=set(s)
            return float({0,1}<=s)+2*float({2,3}<=s)
    tree = exact_spt(Game(),range(4))
    root = tree['nodes'][0]
    assert {tuple(root['left']),tuple(root['right'])} == {(0,1),(2,3)}
    assert root['syn_nats'] == 0.
    assert tree['mass_by_roi_order'][2] == 3.


def test_joint_decoder_recovers_parity_structure_that_pairwise_cannot_encode():
    k=4; labels=np.tile(np.arange(1<<k),16)
    parity=(((labels[:,None]>>np.arange(k))&1).sum(1)%2)*2-1
    features=np.c_[parity,np.ones(len(labels))]
    for order,accuracy in ((1,1/16),(2,1/16),(4,1/8)):
        model=fit_decoder(features,labels,k,order,.001)
        prob=posterior(model,features)
        assert decoder_metrics(prob,labels,k)['mode_accuracy'] == pytest.approx(accuracy)
