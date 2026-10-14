import time

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from scripts.spt_finite_replacement_bc import (
    LearnedMonotoneTM, LearnedConditionalTM, ReplacementBroadcast,
    evaluate_labels, sample_dataset, acceptance,
)
from scripts.spt_finite_data_tm import FittedConditionalTM
from scripts.spt_continuous_phase_benchmark import SOURCES


def test_monotone_inverse_linear_tails_and_jacobian():
    rng = np.random.default_rng(44)
    tm = LearnedMonotoneTM.fit(np.arcsinh(rng.normal(size=(4096,3))*4))
    values = np.array([[-10.,-8.,-6.],[0.,.2,.4],[6.,8.,10.]])
    z,jac = tm.transform(values,jacobian=True)
    np.testing.assert_allclose(tm.inverse(z),values,atol=1e-12)
    shifted = tm.transform(values+1e-6)
    derivatives = (shifted-z)/1e-6
    assert np.all(derivatives>0)
    np.testing.assert_allclose(np.log(derivatives).sum(1),jac,atol=1e-7)
    assert z[0,0]<tm.scores[0] and z[-1,0]>tm.scores[-1]


def test_learned_density_normalizes_on_whole_real_line():
    tm = LearnedMonotoneTM.fit(np.arcsinh(np.random.default_rng(17).normal(size=(2048,1))*3))
    def density(y):
        z,jac = tm.transform(np.array([[y]]),jacobian=True)
        return float(norm.pdf(z[0,0])*np.exp(jac[0]))
    knots = tm.raw_knots[:,0]
    integral = quad(density,-np.inf,knots[0])[0]+quad(density,knots[-1],np.inf)[0]
    integral += sum(quad(density,a,b)[0] for a,b in zip(knots[:-1],knots[1:]))
    assert abs(integral-1)<1e-9


def test_tied_training_knots_fail_instead_of_clipping():
    with pytest.raises(ArithmeticError):
        LearnedMonotoneTM.fit(np.ones((1024,2)))


@pytest.mark.parametrize('case',['B','C'])
def test_generator_formula_agrees_with_full_coherent_gaussian_joint(case):
    model = ReplacementBroadcast(case)
    exact = FittedConditionalTM(model.transition,np.zeros(model.dimension),
                                model.noise_covariance,1000,'real',model.coordinates_per_node)
    for source,target in (((1,2,6),SOURCES),((1,7),(2,3,8)),((2,3,4),SOURCES)):
        assert abs(exact.info(source,target)-model.formula_info(source,target))<1e-11
        assert abs(model.info(source,target)-model.formula_info(source,target))<1e-11
    assert abs(exact.future_mi(1,7)-model.future_mi(1,7))<1e-11
    exact.information.cache_clear(); exact.target_logdet.cache_clear()


def test_c_sampler_and_true_density_include_nonlinear_jacobian():
    c = ReplacementBroadcast('C')
    latent = ReplacementBroadcast('C'); latent.case='B'
    x = np.random.default_rng(12).normal(size=(1024,9,c.coordinates_per_node))
    y = c.step(x,np.random.default_rng(13))
    ly = latent.step(x,np.random.default_rng(13))
    np.testing.assert_allclose(np.sinh(y[:,:,c.channels:]),ly[:,:,c.channels:],atol=1e-12)
    jac = np.sum(np.logaddexp(y[:,:,c.channels:],-y[:,:,c.channels:])-np.log(2),axis=(1,2))
    np.testing.assert_allclose(c.true_log_conditional(x,y)-latent.true_log_conditional(x,ly),jac,atol=1e-10)


def test_conditional_learned_tm_heldout_density_is_raw_space():
    x,y,tx,ty,true_log,model = sample_dataset('C',901,2048,128)
    fitted = LearnedConditionalTM.fit(x,y,nonlinear=True)
    z,jac = fitted.encoder.transform(ty.reshape(128,-1),jacobian=True)
    latent = fitted.gaussian.heldout(tx,z.reshape(ty.shape),true_log-jac)
    raw = fitted.heldout(tx,ty,true_log)
    assert abs(raw['nll_nats']-(latent['nll_nats']-jac.mean()))<1e-10
    assert abs(raw['excess_nll_nats']-latent['excess_nll_nats'])<1e-10
    assert raw['inverse_max_abs_error']<1e-10
    joint = fitted.gaussian.density.covariance if hasattr(fitted.gaussian.density,'covariance') else None
    if joint is not None:
        np.testing.assert_allclose(joint[:model.dimension,:model.dimension],np.eye(model.dimension))


def test_relabeling_scores_preserve_mechanistic_support_not_node_ids():
    model = ReplacementBroadcast('B')
    oracle = FittedConditionalTM(model.transition,np.zeros(model.dimension),model.noise_covariance,
                                 1000,'real',model.coordinates_per_node)
    rows = evaluate_labels(oracle,model.supports,'B',901,2048,deadline=time.monotonic()+60)
    assert len(rows)==20
    assert all(r['methods']['spt']['clade_f1']==1 for r in rows)
    assert all(r['methods']['spt']['correct_root'] for r in rows)
    assert all(r['audit']['significant_violation_count']==0 for r in rows)
    oracle.information.cache_clear(); oracle.target_logdet.cache_clear()


def test_incomplete_or_invalid_results_do_not_pass_acceptance():
    assert not acceptance([])['replace_main_figure']
