import numpy as np
import pytest

from scripts.spt_local_hyperedge_kuramoto import (
    LocalHyperedgeKuramoto, SUPPORTS, edge_force, nesting_levels,
)


@pytest.mark.parametrize('case',list('ABC'))
def test_each_hyperedge_is_local_periodic_and_rotation_invariant(case):
    rng=np.random.default_rng(17)
    phases=rng.uniform(-np.pi,np.pi,(64,9))
    for edge in SUPPORTS[case]:
        force=edge_force(phases,edge)
        outside=sorted(set(range(9))-{i-1 for i in edge})
        assert np.count_nonzero(force[:,outside])==0
        shifted=phases+2*np.pi*rng.integers(-2,3,(64,9))
        np.testing.assert_allclose(edge_force(shifted,edge),force,atol=1e-13)
        np.testing.assert_allclose(edge_force(phases+.731,edge),force,atol=1e-13)
        # Every listed node actually enters the kernel.
        for node in edge:
            altered=phases.copy(); altered[:,node-1]+=.137
            assert np.max(abs(edge_force(altered,edge)-force))>1e-3


def test_pair_kernel_is_kuramoto_and_triad_is_not_a_sum_of_pair_forces():
    phases=np.array([[.23,.78,-.46,0,0,0,0,0,0]])
    pair=edge_force(phases,(1,2))
    assert pair[0,0]==pytest.approx(np.sin(.78-.23))
    assert pair[0,1]==pytest.approx(-pair[0,0])
    # A sum g(theta1,theta2)+h(theta1,theta3) has zero mixed derivative
    # with respect to theta2 and theta3. The triad's derivative is nonzero.
    h=1e-4
    values=[]
    for a,b in ((1,1),(1,-1),(-1,1),(-1,-1)):
        altered=phases.copy(); altered[0,1]+=a*h; altered[0,2]+=b*h
        values.append(edge_force(altered,(1,2,3))[0,0])
    mixed=(values[0]-values[1]-values[2]+values[3])/(4*h*h)
    expected=-np.sin(.78-.46-2*.23)
    assert abs(expected)>.1
    assert mixed==pytest.approx(expected,abs=1e-7)


def test_strength_schedule_depends_only_on_declared_nesting():
    expected_roots={'A':2,'B':4,'C':2}
    for case in 'ABC':
        model=LocalHyperedgeKuramoto(case)
        root=tuple(range(1,10))
        assert nesting_levels(model.supports)[root]==expected_roots[case]
        assert model.weights[root]==.5**expected_roots[case]
        assert all(weight>0 for weight in model.weights.values())


def test_nine_phase_transition_is_reproducible_wrapped_and_step_convergent():
    model=LocalHyperedgeKuramoto('C')
    x=model.sample_source(128,np.random.default_rng(1))
    a=model.step(x,np.random.default_rng(2))
    b=model.step(x,np.random.default_rng(2))
    np.testing.assert_array_equal(a,b)
    assert a.shape==(128,9) and np.all((-np.pi<=a)&(a<np.pi))
    noiseless=LocalHyperedgeKuramoto('C',noise=0)
    coarse=noiseless.step(x,np.random.default_rng(2),dt=.01)
    fine=noiseless.step(x,np.random.default_rng(2),dt=.005)
    difference=np.angle(np.exp(1j*(coarse-fine)))
    assert np.max(abs(difference))<1e-6


def test_bad_phase_states_and_time_steps_fail():
    model=LocalHyperedgeKuramoto('A')
    with pytest.raises(ValueError):
        model.drift(np.zeros((10,9,2)))
    with pytest.raises(ValueError):
        model.step(np.zeros((10,9)),np.random.default_rng(0),dt=.03)
