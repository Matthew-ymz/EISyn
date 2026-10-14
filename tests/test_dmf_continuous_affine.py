"""Structural and signal controls for the experimental continuous-G TM."""
import numpy as np
import pytest

from scripts.dmf_continuous_affine import fit_continuous_affine, sample_moments
from scripts.dmf_joint_readout import CommonTargetGame, audit_nonnegative
from scripts.dmf_response_benchmark import fit_dmf_affine_joint
from scripts.dmf_response_benchmark import fit_affine_joint
from scripts.dmf_continuous_curve_refresh import recover_moments, fit_continuous_batched


def samples(seed=31, n=4, count=1024):
    rng = np.random.default_rng(seed)
    x = rng.uniform(.3, .7, (count, 2*n))
    e = rng.normal(0, .04, x.shape)
    y = x.copy()
    y[:, :n] += .6*x[:, n:]
    y[:, n:] += .3*x[:, :n]
    return x, y, e


def test_zero_matches_independent_roi_refit_and_local_budget():
    x, y, e = samples()
    sc = np.ones((4, 4))-np.eye(4)
    model = fit_continuous_affine(sample_moments(x, y+e), sc, 0, .1, .1)
    reference = fit_dmf_affine_joint(x, y+e, .2, connectivity=sc, coupling=0)
    np.testing.assert_allclose(model.covariance, reference, atol=1e-13, rtol=1e-13)
    game = CommonTargetGame(model.covariance)
    assert abs(game.totals()['cross_roi_nats']) < 1e-8
    assert game.totals()['roi_local_xi_nats'] > .01
    audit_nonnegative(game.audited, 1e-8)


def test_near_zero_parameters_bounded_and_metrics_continuous():
    x, y, e = samples()
    sc = np.ones((4, 4))-np.eye(4)
    matrices, norms = [], []
    for g in [0., 1e-8, 1e-6, 1e-4]:
        target = y+e
        target[:, 0] += g*x[:, 1]
        model = fit_continuous_affine(sample_moments(x, target), sc, g, .1, .1)
        matrices.append(model.covariance)
        norms.append(model.diagnostics['mean_unscaled_cross_norm'])
        assert model.diagnostics['factor_covariance_error'] < 1e-14
        audit_nonnegative(CommonTargetGame(model.covariance).audited, 1e-8)
    np.testing.assert_allclose(matrices[0], matrices[1], atol=1e-8, rtol=0)
    assert norms[-1] < .01
    assert norms[1] < norms[2] < norms[3]


def test_preserves_indirect_real_signal_and_disconnected_components():
    x, y, e = samples(n=4)
    y[:, 2] += 1.2*x[:, 0]  # two-step SC reachability; not a direct edge
    e[:, 2] += .5*e[:, 0]
    sc = np.zeros((4, 4)); sc[1, 0] = sc[2, 1] = 1
    model = fit_continuous_affine(sample_moments(x, y+e), sc, 1.3, .01, .01)
    assert model.coefficient[0, 2] > .2
    disconnected = [3, 7]
    connected = [0, 1, 2, 4, 5, 6]
    np.testing.assert_array_equal(model.coefficient[np.ix_(disconnected, connected)], 0)
    np.testing.assert_array_equal(model.noise[np.ix_(disconnected, connected)], 0)
    game = CommonTargetGame(model.covariance)
    assert game.u((0, 1, 2)) > .01
    assert abs(game.ei(range(4))-game.ei((0, 1, 2))-game.ei((3,))) < 1e-8


@pytest.mark.parametrize('penalty', [0, -1, float('nan')])
def test_invalid_penalty_fails(penalty):
    x, y, e = samples()
    with pytest.raises(ValueError):
        fit_continuous_affine(sample_moments(x, y+e), np.ones((4, 4)), .1, penalty, .1)


def test_reduces_false_positive_without_erasing_known_positive_control():
    x, y, e = samples(count=512)
    sc = np.ones((4, 4))-np.eye(4)
    zero = fit_continuous_affine(sample_moments(x, y+e), sc, 0, .1, .1)
    tiny = fit_continuous_affine(sample_moments(x, y+e), sc, 1e-6, .1, .1)
    signal = y+e; signal[:, 0] += 1.2*x[:, 1]
    positive = fit_continuous_affine(sample_moments(x, signal), sc, 1.3, .1, .1)
    assert abs(CommonTargetGame(zero.covariance).u(range(4))) < 1e-8
    assert abs(CommonTargetGame(tiny.covariance).u(range(4))) < 1e-8
    assert CommonTargetGame(positive.covariance).u(range(4)) > .05


@pytest.mark.parametrize('g',[0.,1e-6,.01,.1,1.3,4.])
@pytest.mark.parametrize('disconnected',[False,True])
def test_recovered_moments_and_batched_solver_match_raw_reference(g,disconnected):
    x,y,e=samples()
    y=y+e; y[:,2]+=g*.4*x[:,0]
    sc=np.ones((4,4))-np.eye(4)
    if disconnected:
        sc[3]=0; sc[:,3]=0
    m=sample_moments(x,y)
    reference=fit_continuous_affine(m,sc,g,.1,10.)
    recovered=recover_moments(fit_affine_joint(x,y,.2),x)
    fitted=fit_continuous_batched(recovered,sc,g)
    np.testing.assert_allclose(recovered.xx,m.xx,atol=1e-12,rtol=0)
    np.testing.assert_allclose(recovered.xy,m.xy,atol=1e-12,rtol=0)
    np.testing.assert_allclose(recovered.yy,m.yy,atol=1e-12,rtol=0)
    np.testing.assert_allclose(fitted.covariance,reference.covariance,atol=2e-12,rtol=0)
    assert CommonTargetGame(fitted.covariance).totals()['xi_nats']==pytest.approx(
        CommonTargetGame(reference.covariance).totals()['xi_nats'],abs=1e-9)
    with pytest.raises(ValueError,match='Native target'):
        fitted.scores(x,y)
