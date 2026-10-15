import time

import numpy as np
from scipy.integrate import quad
from scipy.special import i0e,i1e

from scripts.spt_kuramoto_finite_tm import PeriodicConditionalTM,vm_log_density,LOG_TWO_PI
from scripts.spt_kuramoto_group_tm import GroupTargetIntegrator


def model(coupling=0.):
    vectors=[np.empty((0,9)) for _ in range(9)]
    sine=[np.empty(0) for _ in range(9)]
    cosine=[np.empty(0) for _ in range(9)]
    if coupling:
        v=np.zeros(9); v[0]=-1; v[1]=1
        vectors[0]=v[None,:]; sine[0]=np.array([coupling]); cosine[0]=np.array([0.])
    return PeriodicConditionalTM(vectors,sine,cosine,np.zeros(9),np.full(9,2.))


def integrator(fitted,n=256,p=1024):
    return GroupTargetIntegrator(fitted,outer=n,particles=p,max_particles=max(1024,p),
        seed=2017,deadline=time.monotonic()+30)


def test_group_identity_overlap_and_disjoint_information():
    fitted=model(); estimator=integrator(fitted)
    expected=2*i1e(2)/i0e(2)-np.log(i0e(2))-2
    value,_,_=estimator.information(1,3)
    assert abs(value-expected)<.005
    disjoint,_,_=estimator.information(1,2)
    assert abs(disjoint)<1e-12


def test_group_missing_source_truth_never_enters_integral():
    estimator=integrator(model(.18),n=32)
    first=estimator.marginal(1,3)[0].copy()
    estimator.source[:,1:]+=1.12
    estimator.cache.clear()
    second=estimator.marginal(1,3)[0]
    np.testing.assert_allclose(first,second,atol=1e-12,rtol=0)


def test_diagonal_group_marginal_against_independent_quadrature():
    fitted=model(.18); estimator=integrator(fitted,n=32)
    actual=estimator.marginal(1,1)[0]
    expected=[]
    for x,y in zip(estimator.source,estimator.target):
        density=quad(lambda u:np.exp(vm_log_density(y[0]-x[0]-.18*np.sin(u-x[0]),2)),
                     -np.pi,np.pi,epsabs=1e-12)[0]/(2*np.pi)
        expected.append(np.log(density))
    np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=0)
    marginal,_=estimator.marginal(0,1)
    np.testing.assert_allclose(marginal,-LOG_TWO_PI,atol=.004,rtol=0)


def test_mixed_group_importance_weights_against_two_dimensional_quadrature():
    fitted=model()
    for receiver,coupling in [(0,.3),(1,.4)]:
        vector=np.zeros(9); vector[receiver]=-1; vector[2]=1
        fitted.vectors[receiver]=vector[None,:]
        fitted.sine[receiver]=np.array([coupling]); fitted.cosine[receiver]=np.array([0.])
    estimator=integrator(fitted,n=8,p=4096)
    actual=estimator.marginal(1,3)[0]
    phase=np.arange(256)*2*np.pi/256-np.pi
    x1,x2=np.meshgrid(phase,phase,indexing='ij')
    expected=[]
    for x,y in zip(estimator.source,estimator.target):
        first=vm_log_density(y[0]-x[0]-.3*np.sin(x2-x[0]),2)
        second=vm_log_density(y[1]-x1-.4*np.sin(x2-x1),2)
        expected.append(np.log(np.exp(first+second).mean()))
    np.testing.assert_allclose(actual,expected,atol=.004,rtol=0)


def test_phi_point_estimate_averages_information_before_nonlinear_minimum():
    from scripts.spt_kuramoto_b_group_comparison import Comparison
    values={(3,3):[5,5,5,5],(1,1):[1,1,1,1],(2,2):[1,1,1,1],
            (1,2):[0,1,0,1],(2,1):[1,0,1,0]}
    comparison=Comparison.__new__(Comparison)
    comparison.info=lambda a,b:{k:np.asarray(values[a,b],dtype=float)
                                for k in ('values','prefix','lower_outer')}
    result=comparison.phi(3,1,2)
    assert result['estimate']==3.5
    np.testing.assert_array_equal(result['values'],[3,3,3,3])


def test_refined_pair_xi_violation_is_audited_outside_spt_nodes():
    from scripts.validate_spt_kuramoto_b_full import audit_xi_queries
    singles={str(1<<i):3. for i in range(9)}
    rows=[{'ei_nats':{**singles,'3':5.98}} for _ in range(4)]
    result=audit_xi_queries(rows,[*[1<<i for i in range(9)],3])
    assert result['significant_violation_count']==5
    assert abs(result['minimum_raw_xi_nats']+.02)<1e-12


def test_numpy_ei_summary_produces_serializable_audit_counts():
    import json
    from scripts.spt_kuramoto_finite_tm import masks_to_table
    from scripts.spt_continuous_hierarchy_pilot import audit_table
    table=masks_to_table({1:np.float64(3),2:np.float64(3),3:np.float64(6.1)})
    result=audit_table(table,.01)
    assert json.loads(json.dumps(result))['significant_violation_count']==0


def test_smc_constant_likelihood_has_exact_normalizer_without_prior_volume():
    from scripts.spt_kuramoto_group_smc import smc_evidence
    fitted=model(); x=np.zeros((3,9)); y=np.full((3,9),.1)
    logz,diag=smc_evidence(fitted,x,y,np.array([1]),np.array([0]),particles=256,
        rng=np.random.default_rng(13),fisher=np.zeros((3,1)),deadline=time.monotonic()+15)
    expected=fitted.log_prob(x,y,[0])
    np.testing.assert_allclose(logz,expected,atol=1e-12,rtol=0)
    assert abs(diag['pre_resampling_ess_minimum']-256)<1e-9


def test_smc_sharp_circular_likelihood_agrees_with_quadrature():
    from scripts.spt_kuramoto_group_smc import smc_evidence
    fitted=model(1.2); fitted.kappa[:]=100.
    x=np.zeros((4,9)); y=np.zeros((4,9)); y[:,0]=[.1,.35,.7,1.05]
    reference=np.array([np.log(quad(lambda u:np.exp(vm_log_density(v-1.2*np.sin(u),100)),
        -np.pi,np.pi,epsabs=1e-11,points=[0])[0]/(2*np.pi)) for v in y[:,0]])
    estimates=[]
    for r in range(4):
        logz,diag=smc_evidence(fitted,x,y,np.array([1]),np.array([0]),particles=2048,
            rng=np.random.default_rng(r+217),fisher=np.full((4,1),144.),deadline=time.monotonic()+20,
            layers=8,moves=1)
        estimates.append(np.exp(logz))
        assert 0<diag['local_acceptance']<1
    # Evidence, rather than log evidence, is the unbiased SMC quantity.
    np.testing.assert_allclose(np.log(np.mean(estimates,axis=0)),reference,atol=.06,rtol=0)


def test_group_outer_prefix_reuse_matches_fresh_nested_streams():
    fitted=model(1.2); fitted.kappa[:]=100.
    small=integrator(fitted,n=64,p=256); large=integrator(fitted,n=128,p=256)
    for a,b in [(1,1),(0,3),(2,1)]:
        sv,_=small.marginal(a,b); lv,_=large.marginal(a,b)
        np.testing.assert_allclose(sv,lv[:64],atol=1e-12,rtol=0)
