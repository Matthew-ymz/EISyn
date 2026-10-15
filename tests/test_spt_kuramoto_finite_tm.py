import time

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import i0e, i1e

from scripts.spt_kuramoto_finite_tm import (
    PeriodicConditionalTM, FullTargetIntegrator, TWO_PI, LOG_TWO_PI,
    dictionary, normal_interval_logprob, proposal_log_prob, proposal_sample,
    vm_log_density, posterior_proposal, step_precision,
)


def identity_model(kappa=1000.):
    return PeriodicConditionalTM([np.empty((0,9)) for _ in range(9)],
        [np.empty(0) for _ in range(9)], [np.empty(0) for _ in range(9)],
        np.zeros(9), np.full(9,kappa))


def test_unrestricted_dictionary_is_periodic_rotational_and_contains_nine_body():
    for receiver in range(9):
        vectors=dictionary(receiver)
        assert vectors.shape==(765,9)
        np.testing.assert_array_equal(vectors.sum(1),0)
        np.testing.assert_array_equal(vectors,np.rint(vectors))
        assert np.count_nonzero(np.count_nonzero(vectors,axis=1)==9)==3


def test_stable_vm_density_is_normalized_at_high_concentration():
    for kappa in (2.,1000.,1e5):
        # Rescale the integration variable to resolve the narrow density.
        mass=quad(lambda z:np.exp(vm_log_density(z/np.sqrt(kappa),kappa))/np.sqrt(kappa),
                  -min(14.,np.pi*np.sqrt(kappa)),min(14.,np.pi*np.sqrt(kappa)),epsabs=1e-10)[0]
        assert mass==pytest.approx(1,abs=1e-9)


def test_mean_jacobian_matches_finite_difference_and_payload_roundtrip():
    vectors=[np.empty((0,9)) for _ in range(9)]
    vectors[0]=np.array([[-2.,1.,1.,0,0,0,0,0,0]])
    sine=[np.empty(0) for _ in range(9)]; sine[0]=np.array([.1])
    cosine=[np.empty(0) for _ in range(9)]; cosine[0]=np.array([.02])
    model=PeriodicConditionalTM(vectors,sine,cosine,np.zeros(9),np.ones(9)*1000)
    x=np.random.default_rng(1).uniform(-np.pi,np.pi,(5,9))
    mean,jac=model.mean(x,jacobian=True)
    for j in range(9):
        delta=np.zeros(9); delta[j]=1e-5
        expected=(model.mean(x+delta)-model.mean(x-delta))/2e-5
        np.testing.assert_allclose(jac[:,:,j],expected,atol=2e-10)
    np.testing.assert_array_equal(PeriodicConditionalTM.load(model.payload()).mean(x),mean)


def test_truncated_normal_proposal_normalization_and_torus_boundary():
    center=np.array([[np.pi-.01]])
    chol=np.array([[[.7]]])
    mass=quad(lambda x:np.exp(proposal_log_prob(np.array([[[x]]]),center,chol)[0,0]),
              -np.pi,np.pi,points=[center.item()],epsabs=1e-9)[0]
    assert mass==pytest.approx(1,abs=1e-8)
    values=proposal_sample(center,chol,np.linspace(.001,.999,1000).reshape(1,1000,1))
    assert np.all((-np.pi<=values)&(values<np.pi))
    assert np.any(values<0) and np.any(values>0)
    assert np.isfinite(proposal_log_prob(values,center,chol)).all()
    assert np.isfinite(normal_interval_logprob(np.array([-100.,99.]),np.array([-99.,100.]))).all()


def test_posterior_proposal_does_not_read_missing_source_values():
    model=identity_model()
    x=np.zeros((3,9)); y=np.ones((3,9))*.2
    missing=np.array([1,2,5])
    a=posterior_proposal(model,x,y,missing,np.arange(9))
    x[:,missing]=2.9
    b=posterior_proposal(model,x,y,missing,np.arange(9))
    np.testing.assert_array_equal(a[0],b[0])
    np.testing.assert_array_equal(a[1],b[1])


def test_correlated_two_dimensional_proposal_normalizes_with_truncation():
    from scripts.spt_local_hyperedge_kuramoto import wrap
    center=np.array([[3.12,-3.1]])
    covariance=np.array([[2.8,1.8],[1.8,2.4]])
    chol=np.linalg.cholesky(covariance)[None]
    axis=np.linspace(-np.pi+1e-10,np.pi-1e-10,501)
    u,v=np.meshgrid(axis,axis,indexing='ij')
    values=wrap(center[:,None,:]+np.c_[u.ravel(),v.ravel()][None])
    density=np.exp(proposal_log_prob(values,center,chol)).reshape(501,501)
    mass=np.trapz(np.trapz(density,axis,axis=1),axis,axis=0)
    assert mass==pytest.approx(1,abs=2e-5)


def test_nonlinear_cross_coupled_marginal_agrees_with_direct_quadrature():
    vectors=[np.empty((0,9)) for _ in range(9)]
    a=np.array([[-1.,1.,0,0,0,0,0,0,0]])
    vectors[0]=a; vectors[1]=-a
    sine=[np.empty(0) for _ in range(9)]; sine[0]=np.array([.15]); sine[1]=np.array([.12])
    cosine=[np.zeros(len(v)) for v in vectors]
    model=PeriodicConditionalTM(vectors,sine,cosine,np.zeros(9),np.ones(9)*20)
    integral=FullTargetIntegrator(model,outer=16,max_particles=1024,seed=7,deadline=time.monotonic()+60)
    estimate,_=integral.marginal(510,1024)
    exact=[]
    for i in range(16):
        x=integral.source[i].copy(); y=integral.target[i:i+1]
        center=integral.proposals[510][0][i,0]
        x[0]=center
        offset=model.log_prob(x[None],y).item()
        def integrand(t):
            x[0]=t
            return np.exp(model.log_prob(x[None],y).item()-offset)/TWO_PI
        mass=quad(integrand,-np.pi,np.pi,points=[center],epsabs=1e-10)[0]
        exact.append(np.log(mass)+offset)
    assert np.max(abs(estimate-exact))<.015


def test_importance_marginals_agree_with_exact_independent_vm_channel():
    model=identity_model()
    integral=FullTargetIntegrator(model,outer=256,max_particles=512,seed=42,deadline=time.monotonic()+60)
    for mask in (0,1,15,255,511):
        estimate,diagnostic=integral.marginal(mask,512)
        kept=[i for i in range(9) if mask>>i&1]
        expected=np.full(256,-(9-len(kept))*LOG_TWO_PI)
        if kept:
            expected+=vm_log_density(integral.target[:,kept]-integral.source[:,kept],model.kappa[kept]).sum(1)
        assert np.max(abs(estimate-expected))<.04
        assert abs(np.mean(estimate-expected))<.003


def test_stochastic_step_precision_uses_identical_brownian_paths():
    result=step_precision('B',n=32)
    assert result['paired_brownian'] and result['passed']
    assert result['rms_difference_rad'][1]<result['rms_difference_rad'][0]


def test_integrated_mean_flow_source_and_parameter_sensitivities():
    from scripts.spt_kuramoto_flow_tm import PeriodicFlowConditionalTM,candidate_edges
    assert len(candidate_edges())==502
    model=PeriodicFlowConditionalTM([(0,1),(0,1,2)],np.array([.8,.6]),np.linspace(-.2,.2,9),np.ones(9)*800)
    source=np.random.default_rng(9).uniform(-np.pi,np.pi,(8,9))
    mean,jac=model.mean(source,jacobian=True)
    independent_mean,independent_jac=model.propagate(source,source_jacobian=True)
    np.testing.assert_allclose(mean,independent_mean,atol=2e-14,rtol=0)
    np.testing.assert_allclose(jac,independent_jac,atol=2e-14,rtol=0)
    for j in range(9):
        h=np.zeros(9); h[j]=1e-5
        difference=(model.mean(source+h)-model.mean(source-h))/2e-5
        np.testing.assert_allclose(jac[:,:,j],difference,atol=2e-10)
    _,parameter_jac=model.propagate(source,parameter_jacobian=True)
    for j in range(2):
        original=model.weights[j]
        model.weights[j]=original+1e-5; plus=model.mean(source)
        model.weights[j]=original-1e-5; minus=model.mean(source)
        model.weights[j]=original
        np.testing.assert_allclose(parameter_jac[:,:,j],(plus-minus)/2e-5,atol=2e-10)
    np.testing.assert_array_equal(PeriodicConditionalTM.load(model.payload()).mean(source),mean)
    assert np.all(np.linalg.det(jac)>0)


def test_critical_syn_audit_records_and_rejects_significant_negative_values():
    from scripts.spt_kuramoto_finite_tm import critical_syn_audit,pilot_masks
    rows=[{'outer':outer,'particles':particles,'scramble':scramble,
           'ei_nats':{str(mask):float(mask.bit_count()) for mask in pilot_masks('A')}}
          for outer,particles in ((256,64),(256,256),(1024,256),(4096,256))
          for scramble in range(4)]
    accepted=critical_syn_audit(rows)
    assert accepted['passed'] and accepted['split_count']>0
    for row in rows:
        if row['outer']==4096:
            row['ei_nats']['3']-=.02
    rejected=critical_syn_audit(rows)
    assert not rejected['passed']
    high=rejected['summary'][-1]
    assert high['significant_violation_count']>=4
    assert high['minimum_raw_syn_nats']==pytest.approx(-.02)
