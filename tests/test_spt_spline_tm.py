"""Density, support and integration checks independent of planted hierarchies."""
import math
import time

import numpy as np
import pytest
import torch

from scripts.spt_spline_tm import ConditionalSplineTM, MarginalIntegrator


@pytest.fixture(autouse=True)
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def nonlinear_model(dimension, domain):
    torch.manual_seed(7)
    model = ConditionalSplineTM(dimension, domain).double()
    with torch.no_grad():
        for head in (model.conditioner.context[-1], model.conditioner.history[-1]):
            head.weight.normal_(0, .035)
            head.bias.normal_(0, .02)
    return model


@pytest.mark.parametrize('domain', ['real', 'phase'])
def test_inverse_density_jacobian_and_triangularity(domain):
    model = nonlinear_model(3, domain)
    x = torch.tensor([[.3, .8, 1.2]], dtype=torch.float64)
    reference = torch.tensor([[.2, .4, .7]], dtype=torch.float64)
    y = model.sample(x, reference)
    prepared = model.prepare(x, y)
    z, logjac = model.spline(prepared, model.conditioner(x, prepared))
    assert torch.max(abs(z-reference)) < 1e-10
    def transform(values):
        inputs = model.prepare(x, values.reshape(1, -1))
        return model.spline(inputs, model.conditioner(x, inputs))[0].reshape(-1)
    jacobian = torch.autograd.functional.jacobian(transform, y[0])
    assert torch.max(abs(torch.triu(jacobian, diagonal=1))) < 1e-12
    assert torch.all(jacobian.diag() > 0)
    _, logdet = torch.linalg.slogdet(jacobian)
    if domain == 'phase':
        expected = logdet
    else:
        expected = -.5*(z*z+math.log(2*math.pi)).sum()+logdet
    assert abs(float(model.log_prob(x, y))-float(expected)) < 1e-10


def test_periodic_conditional_density_normalization_and_seams():
    model = nonlinear_model(2, 'phase')
    grid = (torch.arange(16384, dtype=torch.float64)+.5)/16384
    x = torch.tensor([[.4, 1.3]], dtype=torch.float64).repeat(len(grid), 1)
    y = torch.column_stack((torch.full_like(grid, .31), grid))
    parameters = model.conditioner(x, y)
    _, logjac = model.spline(y[:, 1], parameters[:, 1])
    assert abs(float(logjac.exp().mean())-1) < 2e-7
    # Full joint density is periodic in every source/history/output coordinate.
    xx = torch.tensor([[0., 1.3], [2*math.pi, 1.3]], dtype=torch.float64)
    yy = torch.tensor([[.8, 2.1], [.8, 2.1]], dtype=torch.float64)
    assert abs(float(torch.diff(model.log_prob(xx, yy)))) < 1e-12
    for j in range(2):
        yy = torch.tensor([[.8, 2.1], [.8, 2.1]], dtype=torch.float64)
        yy[0, j], yy[1, j] = 0., 2*math.pi
        assert abs(float(torch.diff(model.log_prob(xx, yy)))) < 1e-12


def test_identity_circular_map_is_full_uniform_density():
    model = ConditionalSplineTM(3, 'phase').double()
    x = torch.full((4, 3), .3, dtype=torch.float64)
    y = torch.linspace(.2, 5.9, 12, dtype=torch.float64).reshape(4, 3)
    assert torch.max(abs(model.log_prob(x, y)+3*math.log(2*math.pi))) < 1e-12
    with pytest.raises(ValueError, match='outside'):
        model.log_prob(x, y-10)


def test_gaussian_posterior_proposal_corrects_nonzero_affine_channel():
    # One coordinate per source, arbitrary dense learned channel: no mechanism
    # support. Exact affine limit makes all importance correction weights one.
    from scripts.spt_finite_data_tm import FittedConditionalTM
    rng = np.random.default_rng(13)
    coefficient = rng.normal(0, .16, (9, 9))
    noise = np.eye(9)*.7
    oracle = FittedConditionalTM(coefficient, np.zeros(9), noise, 100, 'real', 1)
    model = ConditionalSplineTM(9, 'real', coefficient=coefficient,
                                cholesky=np.linalg.cholesky(noise))
    integrator = MarginalIntegrator(model, node_width=1, outer=16,
        max_particles=32, seed=12, deadline=time.monotonic()+60)
    for mask in (0, 1, 31, 480, 511):
        numerical, diagnostic = integrator.marginal(mask, 32)
        indices = integrator.columns(mask)
        missing = integrator.columns(511^mask)
        covariance = noise+coefficient[:, missing]@coefficient[:, missing].T
        residual = integrator.target-integrator.source[:, indices]@coefficient[:, indices].T
        chol = np.linalg.cholesky(covariance)
        z = np.linalg.solve(chol, residual.T).T
        expected = -.5*(z*z+math.log(2*math.pi)).sum(1)-np.log(np.diag(chol)).sum()
        assert np.max(abs(numerical-expected)) < 4e-6
        if mask != 511:
            assert diagnostic['ess_minimum'] > 31.999
    assert oracle.info((1,)) > 0


def test_uniform_nonlinear_family_null_has_zero_ei_and_syn():
    from scripts.spt_continuous_hierarchy_pilot import audit_table
    from scripts.spt import all_nonempty_subsets
    model = ConditionalSplineTM(9, 'phase')
    integrator = MarginalIntegrator(model, node_width=1, outer=8,
        max_particles=8, seed=17, deadline=time.monotonic()+60)
    masks = [sum(1 << (i-1) for i in s) for s in all_nonempty_subsets(tuple(range(1,10)))]
    values, _ = integrator.information(masks, 8)
    ei = {tuple(i+1 for i in range(9) if mask >> i & 1): value
          for mask, value in values.items()}
    assert max(abs(v) for v in values.values()) < 1e-12
    audit = audit_table(ei, .03)
    assert audit['significant_violation_count'] == 0
    assert audit['partition_count'] == 9330


def test_nonidentity_importance_weights_match_independent_quadrature():
    from numpy.polynomial.hermite import hermgauss
    from scipy.special import logsumexp
    model = ConditionalSplineTM(9, 'real')
    with torch.no_grad():
        for parameter in model.conditioner.parameters():
            parameter.zero_()
        # Genuine nonlinear source-dependent shape, without planted grouping.
        model.conditioner.context[0].weight[0,0] = 1.
        model.conditioner.context[2].weight[0,0] = 1.
        model.conditioner.context[-1].weight[0,0] = .3
    integrator = MarginalIntegrator(model, node_width=1, outer=8,
        max_particles=512, seed=41, deadline=time.monotonic()+60)
    assert not integrator.identity_spline
    numerical, _ = integrator.marginal(0,512)
    points, weights = hermgauss(96)
    source = np.zeros((len(points),9));source[:,0]=points*np.sqrt(2)
    expected=[]
    for y in integrator.target:
        with torch.no_grad():
            lp = model.log_prob(torch.tensor(source,dtype=torch.float32),
                torch.tensor(np.tile(y,(len(points),1)),dtype=torch.float32)).numpy()
        expected.append(logsumexp(lp+np.log(weights/np.sqrt(np.pi))))
    assert np.max(abs(numerical-np.array(expected))) < .002


def test_significant_syn_violation_stops_tree_and_reports_count():
    from scripts.run_spt_spline_tm import build_checked_tree
    values={m:0. for m in range(1,512)}
    values[3]=-.1
    result=build_checked_tree(values,((1,2),tuple(range(1,10))))
    assert result['status']=='invalid_syn'
    assert 'spt' not in result
    assert result['audit']['minimum_syn_nats']==-.1
    assert result['audit']['significant_violation_count']>0
    assert 'threshold=-0.03' in result['nonnegativity_error']
    assert 'affected_count=' in result['nonnegativity_error']
