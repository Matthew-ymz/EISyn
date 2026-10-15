import time

import numpy as np

from scripts.spt_glv_finite_tm import RealConditionalTM
from scripts.spt_glv_natural_baselines import NaturalInformation,shap_profile,fixed_target_wms,source_tm


def identity_model():
    return RealConditionalTM([np.empty((0,9)) for _ in range(9)],[np.empty(0) for _ in range(9)],
        np.zeros(9),np.full(9,.04))


def test_correlated_natural_marginal_matches_exact_gaussian_density(tmp_path):
    cov=np.eye(9)*.1; cov[0,1]=cov[1,0]=.08; mean=np.linspace(-.1,.1,9)
    e=NaturalInformation(identity_model(),mean,cov,outer=64,seed=6,deadline=time.monotonic()+60,cache_dir=tmp_path)
    for mask in (0,1,3,15,511):
        known=np.array([i for i in range(9) if mask>>i&1],dtype=int)
        missing=np.array([i for i in range(9) if not mask>>i&1],dtype=int)
        mu=e.source.copy(); conditional_cov=np.zeros((9,9))
        if len(missing):
            if len(known):
                gain=np.linalg.solve(cov[np.ix_(known,known)],cov[np.ix_(known,missing)]).T
                mu[:,missing]=mean[missing]+(e.source[:,known]-mean[known])@gain.T
                cm=cov[np.ix_(missing,missing)]-gain@cov[np.ix_(known,missing)]
            else: mu[:]=mean; cm=cov
            conditional_cov[np.ix_(missing,missing)]=cm
        ch=np.linalg.cholesky(conditional_cov+np.eye(9)*.04)
        white=np.linalg.solve(ch,(e.target-mu).T).T
        exact=-.5*(np.sum(white**2,axis=1)+9*np.log(2*np.pi))-np.log(np.diag(ch)).sum()
        actual,d=e.marginal(mask,64)
        np.testing.assert_allclose(actual,exact,atol=1e-11)
        if d['ess_minimum'] is not None: assert d['ess_minimum']>63.99


def test_shap_nonlinear_game_retains_joint_background_and_exact_efficiency():
    m=identity_model(); m.vectors[0]=np.array([[1,1,0,0,0,0,0,0,0]])/np.pi; m.coefficients[0]=np.array([.7])
    x=np.array([[.1,.4,0,0,0,0,0,0,0],[-.4,.3,0,0,0,0,0,0,0]])
    b=np.zeros((3,9)); b[:,0]=[-1.,0.,1.]; b[:,1]=b[:,0]
    p,a=shap_profile(m,x,b,time.monotonic()+60)
    joint=np.exp((b[:,0]+b[:,1])/np.pi).mean()
    e0=np.exp(b[:,0]/np.pi).mean(); e1=np.exp(b[:,1]/np.pi).mean()
    v0=joint; vx0=np.exp(x[:,0]/np.pi)*e1; vx1=e0*np.exp(x[:,1]/np.pi); vx01=np.exp(x[:,:2].sum(1)/np.pi)
    phi0=x[:,0]-b[:,0].mean()+.35*(vx0-v0+vx01-vx1)
    phi1=.35*(vx1-v0+vx01-vx0)
    np.testing.assert_allclose(p[0,0],abs(phi0).mean(),atol=1e-12)
    np.testing.assert_allclose(p[1,0],abs(phi1).mean(),atol=1e-12)
    assert a['efficiency_error_output_units']<1e-12


def test_signed_natural_wms_is_kept_and_telescopes():
    covariance=np.eye(9); covariance[0,1]=covariance[1,0]=.8
    posterior=np.linalg.inv(np.linalg.inv(covariance)+np.eye(9)*25)
    table={}
    for mask in range(1,512):
        idx=[i for i in range(9) if mask>>i&1]
        table[mask]=.5*(np.linalg.slogdet(covariance[np.ix_(idx,idx)])[1]-np.linalg.slogdet(posterior[np.ix_(idx,idx)])[1])
    _,audit=fixed_target_wms(table)
    assert audit['minimum_selected_search_wms_nats']<-.01
    assert audit['negative_values_are_valid'] and not audit['clipping']
    assert abs(audit['closure_error_nats'])<1e-10


def test_natural_source_tm_retains_off_diagonal_covariance():
    rng=np.random.default_rng(31); states=rng.normal(size=(513,64,9))*.05
    states[:,:,1]+=.7*states[:,:,0]
    _,cov,audit=source_tm(states)
    assert audit['passed'] and audit['maximum_absolute_source_correlation']>.5
    assert cov[0,1]>.001
