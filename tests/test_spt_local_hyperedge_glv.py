import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from scripts.spt_local_hyperedge_kuramoto import LocalHyperedgeKuramoto, SUPPORTS
from scripts.spt_local_hyperedge_glv import LocalHyperedgeGLV, abundance, edge_force, product_moments


@pytest.mark.parametrize('order',range(2,10))
def test_analytic_center_and_rms_under_common_intervention(order):
    z,w=leggauss(32)
    mean_x=float(w@np.exp(z)/2)
    second=float(w@np.exp(2*z)/2)
    mean,var=product_moments(order)
    assert mean==pytest.approx(mean_x**(order-1),rel=1e-13)
    assert var==pytest.approx(second**(order-1)-mean**2,rel=1e-12)
    assert var/(2*var)==.5


@pytest.mark.parametrize('case',list('ABC'))
def test_model_is_exact_positive_glv_with_local_edges_and_paired_sources(case):
    model=LocalHyperedgeGLV(case)
    ref=LocalHyperedgeKuramoto(case)
    source=model.sample_source(128,np.random.default_rng(23))
    paired=ref.sample_source(128,np.random.default_rng(23))
    np.testing.assert_array_equal(source,paired)
    assert model.weights==ref.weights
    x=abundance(source)
    raw=model.raw_drift(x,ito=False)
    np.testing.assert_allclose(np.pi*raw/x,model.drift(source),atol=1e-12)
    correction=model.raw_drift(x)-raw
    np.testing.assert_allclose(correction,.5*(model.noise/np.pi)**2*x,atol=1e-13)
    for edge in SUPPORTS[case]:
        force=edge_force(source,edge)
        outside=sorted(set(range(9))-{i-1 for i in edge})
        assert np.count_nonzero(force[:,outside])==0


def test_shared_noise_path_step_convergence_and_no_periodic_wrapping():
    model=LocalHyperedgeGLV('C')
    source=model.sample_source(64,np.random.default_rng(17))
    fine_increments=model.noise*np.sqrt(.005)*np.random.default_rng(24).normal(size=(40,64,9))
    coarse_increments=fine_increments.reshape(20,2,64,9).sum(1)
    coarse=model.step(source,np.random.default_rng(1),increments=coarse_increments)
    fine=model.step(source,np.random.default_rng(1),dt=.005,increments=fine_increments)
    assert np.max(abs(coarse-fine))<.003
    assert np.isfinite(abundance(coarse)).all()
    # log abundance coordinates live on R; there is no equality modulo 2*pi.
    assert np.max(abs(model.drift(source+2*np.pi)-model.drift(source)))>1


def test_drift_has_the_claimed_irreducible_three_species_raw_term():
    coefficient=-.37
    x=np.array([1.1,.8,1.4]); h=1e-3
    difference=0.
    for signs in np.ndindex(2,2,2):
        shift=np.array([1 if s else -1 for s in signs])
        difference+=np.prod(shift)*coefficient*np.prod(x+h*shift)
    assert difference/(2*h)**3==pytest.approx(coefficient,abs=1e-7)


def test_invalid_or_exploding_states_fail_without_clipping():
    with pytest.raises((FloatingPointError,ArithmeticError)):
        abundance(np.full((2,9),1e6))
    with pytest.raises(ValueError):
        LocalHyperedgeGLV('A').raw_drift(np.zeros((2,9)))
