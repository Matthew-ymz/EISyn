import time

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from scripts.spt_glv_finite_tm import RealConditionalTM
from scripts.spt_glv_group_tm import RealGroupIntegrator,informed_variables


def model(sd=.1):
    return RealConditionalTM([np.empty((0,9)) for _ in range(9)],
        [np.empty(0) for _ in range(9)],np.zeros(9),np.full(9,sd**2))


def test_identity_group_information_and_disjoint_source():
    m=model()
    e=RealGroupIntegrator(m,outer=256,max_particles=256,seed=13,deadline=time.monotonic()+60)
    y=e.target
    marginal=np.log((ndtr((y+np.pi)/.1)-ndtr((y-np.pi)/.1))/(2*np.pi))
    conditional=-.5*((y-e.source)/.1)**2-np.log(.1)-.5*np.log(2*np.pi)
    for a,b in [(1,1),(1,3),(7,3),(2,1),(0,1)]:
        shared=[i for i in range(9) if a&b&(1<<i)]
        exact=(conditional[:,shared]-marginal[:,shared]).sum(1).mean()
        estimate,_=e.information(a,b,256)
        assert abs(estimate-exact)<.01
    estimate,_=e.future_information(0,1)
    assert abs(estimate)<.01


def test_nonlinear_missing_input_is_guided_and_matches_quadrature():
    m=model()
    m.vectors[0]=np.eye(9)[[1]]/np.pi
    m.coefficients[0]=np.array([.8])
    guided,nuisance=informed_variables(m,1,1)
    assert guided.tolist()==[1]
    assert 0 not in nuisance
    e=RealGroupIntegrator(m,outer=64,max_particles=256,seed=27,deadline=time.monotonic()+60)
    estimate,diagnostic=e.marginal(1,1,256)
    def reference(x,y):
        return np.log(quad(lambda v: np.exp(-.5*((y-x-.8*np.exp(v/np.pi))/.1)**2)
            /(.1*np.sqrt(2*np.pi)*2*np.pi),-np.pi,np.pi,epsabs=1e-11)[0])
    exact=np.array([reference(x,y) for x,y in zip(e.source[:,0],e.target[:,0])])
    assert abs((estimate-exact).mean())<.01
    assert diagnostic['max_projected_step']<1e-3


def test_query_cache_reuses_exact_density_and_provenance(tmp_path):
    m=model()
    make=lambda:RealGroupIntegrator(m,outer=16,max_particles=64,seed=14,
        deadline=time.monotonic()+60,cache_dir=tmp_path)
    first=make(); value,meta=first.marginal(1,3,64)
    second=make(); loaded,loaded_meta=second.marginal(1,3,64)
    np.testing.assert_array_equal(value,loaded)
    assert meta==loaded_meta and second.query_count==0
    m.intercept[0]=.2
    changed=make(); altered,new_meta=changed.marginal(1,3,64)
    assert meta['model_sha']!=new_meta['model_sha']
    assert len(list(tmp_path.glob('*.npz')))==2


def test_no_true_outer_missing_values_are_used():
    m=model(); e=RealGroupIntegrator(m,outer=32,max_particles=64,seed=44,deadline=time.monotonic()+60)
    a,_=e.marginal(1,3,64)
    e.cache.clear(); e.source[:,1:]=100.
    b,_=e.marginal(1,3,64)
    np.testing.assert_array_equal(a,b)


def test_full_missing_proposal_integrates_unidentified_directions():
    m=model()
    e=RealGroupIntegrator(m,outer=128,max_particles=256,seed=13,deadline=time.monotonic()+60,proposal='full')
    y=e.target[:,0]
    marginal=np.log((ndtr((y+np.pi)/.1)-ndtr((y-np.pi)/.1))/(2*np.pi))
    estimate,d=e.marginal(0,1,256)
    assert abs((estimate-marginal).mean())<.01
    assert set(d['guided'])==set(range(9)) and not d['nuisance']


def test_fresh_row_refinement_preserves_identity_marginal():
    m=model()
    e=RealGroupIntegrator(m,outer=16,max_particles=256,seed=13,deadline=time.monotonic()+60,proposal='full')
    row=0; y=e.target[row,0]
    exact=np.log((ndtr((y+np.pi)/.1)-ndtr((y-np.pi)/.1))/(2*np.pi))
    estimate,ess,weight,p=e.refine_row(0,1,row,np.arange(9),256)
    assert abs(estimate-exact)<.02
    assert ess>=10 and weight<.25 and p>=1024
