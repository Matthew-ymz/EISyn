import time

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from scripts.spt_glv_finite_tm import (
    RealConditionalTM,RealFullTargetIntegrator,proposal_sample,proposal_log_prob,
    posterior_proposal,dictionary,
)


def identity_model(sd=.1):
    return RealConditionalTM([np.empty((0,9)) for _ in range(9)],
                             [np.empty(0) for _ in range(9)],np.zeros(9),np.full(9,sd**2))


def test_conditional_density_tm_and_analytic_jacobian():
    vectors=[np.eye(9)[[i]]/np.pi for i in range(9)]
    model=RealConditionalTM(vectors,[np.array([.1]) for _ in range(9)],np.zeros(9),np.full(9,.04))
    x=np.random.default_rng(2).uniform(-3,3,(10,9))
    mean,jac=model.mean(x,jacobian=True)
    for i in range(9):
        shift=np.zeros(9); shift[i]=1e-6
        numeric=(model.mean(x+shift)-model.mean(x-shift))/2e-6
        np.testing.assert_allclose(numeric,jac[:,:,i],atol=1e-8)
    y=model.sample(x,np.random.default_rng(3).uniform(size=(10,9)))
    z=model.conditional_tm(x,y)
    expected=-.5*np.sum(z*z+np.log(2*np.pi),axis=1)-np.log(np.sqrt(model.variance)).sum()
    np.testing.assert_allclose(model.log_prob(x,y),expected,atol=1e-12)


def test_absolute_truncated_proposal_normalizes_near_boundary():
    center=np.array([[3.1]]); chol=np.array([[[.3]]])
    def density(x):
        return float(np.exp(proposal_log_prob(np.array([[[x]]]),center,chol))[0,0])
    assert abs(quad(density,-np.pi,np.pi,epsabs=1e-10)[0]-1)<1e-9
    u=np.random.default_rng(3).uniform(size=(1,4096,1))
    x=proposal_sample(center,chol,u)
    assert np.all((x>=-np.pi)&(x<np.pi))
    assert x.mean()>2.5


def test_fused_proposal_preserves_points_and_exact_density():
    center=np.array([[0.,0.,0.],[3.1,-3.1,.5],[3.65,0.,0.],[-3.65,0.,0.]])
    chol=np.broadcast_to(np.diag([.1,.2,.3]),(4,3,3)).copy()
    chol[:,2,1]=.08
    u=np.random.default_rng(14).uniform(.0001,.9999,size=(4,256,3))
    x=proposal_sample(center,chol,u)
    fused,density=proposal_sample(center,chol,u,return_log_prob=True)
    np.testing.assert_array_equal(x,fused)
    np.testing.assert_allclose(density,proposal_log_prob(x,center,chol),rtol=0,atol=1e-9)


def test_remote_truncated_tail_uses_stable_inverse_without_clipping():
    center=np.array([[30.]])
    chol=np.array([[[1.]]])
    u=np.linspace(.001,.999,64).reshape(1,64,1)
    value,lp=proposal_sample(center,chol,u,return_log_prob=True)
    assert np.isfinite(value).all() and np.isfinite(lp).all()
    assert np.all((value>-np.pi)&(value<np.pi))
    np.testing.assert_allclose(lp,proposal_log_prob(value,center,chol),atol=1e-9)


def test_real_integrator_agrees_with_exact_identity_gaussian_marginal():
    model=identity_model()
    estimator=RealFullTargetIntegrator(model,outer=256,max_particles=256,seed=17,deadline=time.monotonic()+60)
    masks=[1,7,511]
    estimate,pointwise,_=estimator.information(masks,256)
    y=estimator.target
    marginal=np.log((ndtr((y+np.pi)/.1)-ndtr((y-np.pi)/.1))/(2*np.pi))
    conditional=-.5*((y-estimator.source)/.1)**2-np.log(.1)-.5*np.log(2*np.pi)
    for mask in masks:
        columns=[i for i in range(9) if mask>>i&1]
        exact=(conditional[:,columns]-marginal[:,columns]).sum(1)
        assert abs(estimate[mask]-exact.mean())<.01


def test_real_boundary_convergence_uses_projected_stationarity():
    model=identity_model()
    source=np.zeros((1,9)); target=np.full((1,9),3.3)
    center,chol,diagnostics=posterior_proposal(model,source,target,np.arange(9),np.arange(9))
    assert diagnostics['max_remaining_step_coordinate']>.1
    assert diagnostics['max_projected_remaining_step_coordinate']<1e-9
    assert np.all(center<np.pi)


def test_mean_dictionary_contains_all_candidate_supports_without_case_argument():
    vectors=dictionary()
    assert vectors.shape==(1533,9)
    assert len({tuple(np.flatnonzero(v)) for v in vectors})==511


def test_gram_acceleration_preserves_the_fixed_lasso_objective():
    from sklearn.linear_model import Lasso
    rng=np.random.default_rng(15)
    design=rng.normal(size=(1024,48)); design[:,1]+=.8*design[:,0]
    design-=design.mean(0); design=np.asfortranarray(design)
    y=design[:,0]+.3*design[:,2]+rng.normal(scale=.03,size=1024)
    shared=dict(alpha=.0001,fit_intercept=True,max_iter=2500,tol=1e-7,selection='cyclic')
    a=Lasso(**shared,precompute=False).fit(design,y)
    b=Lasso(**shared,precompute=design.T@design).fit(design,y)
    np.testing.assert_allclose(a.coef_,b.coef_,atol=1e-10)
    np.testing.assert_allclose(a.predict(design),b.predict(design),atol=1e-10)


def test_compiled_product_kernel_matches_original_mean_and_jacobian():
    rng=np.random.default_rng(25); v=dictionary()
    selected=[v[rng.choice(len(v),40,replace=False)] for _ in range(9)]
    beta=[rng.normal(scale=1e-5,size=40) for _ in range(9)]
    model=RealConditionalTM(selected,beta,rng.normal(size=9),np.full(9,.001))
    x=rng.uniform(-np.pi,np.pi,(512,9))
    a,ja=model.mean(x,jacobian=True); b,jb=model.mean_numpy(x,jacobian=True)
    np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(ja,jb,rtol=1e-12,atol=1e-10)
    selected,derivative=model.mean_selected(x,[8,0,3],[7,1,3])
    np.testing.assert_array_equal(selected,a[:,[8,0,3]])
    np.testing.assert_array_equal(derivative,ja[:,[8,0,3]][:,:,[7,1,3]])
