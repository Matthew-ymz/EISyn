import time

import numpy as np
from scipy.special import ndtr
from scipy.stats import truncnorm

from scripts.spt_glv_finite_tm import RealConditionalTM
from scripts.spt_glv_surd import SpecificInformation,audit_and_allocate


def identity_model():
    return RealConditionalTM([np.empty((0,9)) for _ in range(9)],
        [np.empty(0) for _ in range(9)],np.zeros(9),np.full(9,.01))


def test_specific_kl_against_exact_bounded_gaussian_posterior(tmp_path):
    e=SpecificInformation(identity_model(),outer=32,posterior_particles=64,inner_particles=64,
        seed=17,deadline=time.monotonic()+60,cache_dir=tmp_path)
    y=e.base.target
    exact=np.log(2*np.pi)-truncnorm.entropy((-np.pi-y)/.1,(np.pi-y)/.1,loc=y,scale=.1)
    for mask in [1,3,511]:
        value,diagnostic=e.specific(mask)
        columns=[j for j in range(9) if mask>>j&1]
        assert abs((value-exact[:,columns].sum(1)).mean())<.025
        assert not diagnostic['issues']
    loaded,_=e.specific(1)
    first,_=e.specific(1)
    np.testing.assert_array_equal(loaded,first)


def test_surd_allocates_before_target_average():
    # The preferred source varies by state. Allocating mean MI loses this.
    specific=np.zeros((2,512))
    membership=((np.arange(512)[:,None]>>np.arange(9))&1)
    specific[0]=membership[:,0]*1.+membership[:,1]*3.+membership[:,2:].sum(1)*.01
    specific[1]=membership[:,0]*3.+membership[:,1]*1.+membership[:,2:].sum(1)*.01
    red,syn,audit=audit_and_allocate(specific)
    r2,s2,_=audit_and_allocate(specific.mean(0,keepdims=True))
    assert not np.allclose(syn,s2)
    assert abs(red.sum()+syn.sum()-specific[:,-1].mean())<1e-10
    assert audit['clipping'] is False


def test_significant_specific_kl_violation_fails():
    import pytest
    specific=np.ones((2,512)); specific[:,0]=0.; specific[0,1]=-.02
    with pytest.raises(ArithmeticError,match='threshold=-0.01; affected_count=1'):
        audit_and_allocate(specific)


def test_inner_refinement_requires_all_weight_checks_even_when_means_converge(tmp_path,monkeypatch):
    import scripts.spt_glv_surd as module
    monkeypatch.setattr(module,'load_model',lambda:None)
    class Estimate:
        bad=True
        def __init__(self,*args,outer,**kwargs):
            self.outer=outer; self.common_diagnostics={}
        def specific(self,mask):
            return np.full(self.outer,mask*.01),dict(issues=['ESS<10'] if self.bad else [])
    monkeypatch.setattr(module,'SpecificInformation',Estimate)
    def references(n,p,k):
        return [dict(outer=n,posterior=p,inner=k,scramble=s,mi_nats={str(m):m*.01 for m in module.PILOT_MASKS}) for s in range(4)]
    initial=dict(records=references(64,64,256))
    refined=dict(status='precision_failed',failed_checks=['ESS<10'],records=references(128,128,64))
    failed=module.refine_inner_pilot(tmp_path/'bad',time.monotonic()+60,initial,refined)
    assert failed['status']=='precision_failed' and failed['failed_checks']==['ESS<10']
    assert max(failed['grid_maximum_changes_nats'].values())<1e-12
    Estimate.bad=False
    passed=module.refine_inner_pilot(tmp_path/'good',time.monotonic()+60,initial,refined)
    assert passed['status']=='precision_passed' and passed['full_grid']['inner']==256
    assert len(passed['records'])==8


def test_provisional_presentation_requires_complete_audited_coalition_tables():
    import pytest
    from scripts.summarize_spt_glv_a_comparison import provisional_surd_result
    atoms=np.zeros(512); atoms[7]=1.; atoms[24]=.9; atoms[96]=.8; atoms[384]=.7
    row=dict(scramble=0,subsets=510,complete=False,synergy_nats=atoms.tolist(),diagnostics={},audit=dict(
        significant_specific_violation_count=0,significant_atom_violation_count=0,
        clipping=False,maximum_statewise_closure_error_nats=0.))
    full=dict(status='budget_exhausted',grid=dict(outer=128,posterior=128,inner=256),replicates=[row])
    assert provisional_surd_result(full) is None
    row.update(subsets=511,complete=True)
    result=provisional_surd_result(full)
    assert result['provisional'] and result['completed_integration_replicates']==1
    assert result['selected_scrambles']==[0] and result['source_status']=='budget_exhausted'
    atoms[7]=-.02; row['synergy_nats']=atoms.tolist()
    with pytest.raises(ArithmeticError,match='threshold=-0.01'):
        provisional_surd_result(full)


def test_absolute_budget_uses_wall_clock_when_monotonic_clock_drifts(monkeypatch):
    import pytest
    import scripts.spt_glv_finite_tm as module
    monkeypatch.setattr(module.time,'monotonic',lambda:1.)
    monkeypatch.setattr(module.time,'time',lambda:201.)
    module.check_time(3.)  # Existing elapsed-time pilots keep their semantics.
    with pytest.raises(TimeoutError,match='absolute wall-clock'):
        module.check_time(module.WallClockDeadline(200.))
