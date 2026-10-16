import time

import numpy as np
import pytest
from scipy.integrate import trapezoid

from scripts.spt_kuramoto_finite_tm import PeriodicConditionalTM
from scripts.spt_kuramoto_natural_mlp import NaturalMLPConditionalTM, chain_partitions
from scripts.run_spt_natural_mlp_pilot import launch


def test_pair_mlp_equivariance_jacobian_and_serialization():
    from scripts.spt_kuramoto_pair_mlp import PairMLPConditionalTM
    rng = np.random.default_rng(875)
    m = PairMLPConditionalTM([rng.normal(0,.08,s) for s in [(144,7),(7,7),(7,9)]],
        [rng.normal(0,.03,s) for s in [7,7,9]],rng.normal(0,.1,144),
        rng.uniform(.5,2,144),rng.normal(0,.02,9),rng.uniform(.1,.3,9),np.full(9,50.))
    x = rng.uniform(-np.pi,np.pi,(3,9))
    mean,jac = m.mean(x,jacobian=True)
    for k in range(9):
        d = np.zeros_like(x)
        d[:,k] = 1e-5
        np.testing.assert_allclose(jac[:,:,k],(m.mean(x+d)-m.mean(x-d))/2e-5,atol=1e-9)
    np.testing.assert_allclose(m.mean(x+.83),mean+.83,atol=1e-12)
    shifted = x.copy()
    shifted[:,4] += 2*np.pi
    np.testing.assert_allclose(np.exp(1j*m.mean(shifted)),np.exp(1j*mean),atol=1e-12)
    loaded = PeriodicConditionalTM.load(m.payload())
    assert loaded.sha256() == m.sha256()
    np.testing.assert_array_equal(loaded.mean(x),mean)


def test_multi_initial_pool_uses_disjoint_complete_trajectories():
    from scripts.run_spt_kuramoto_A_fit_improvement import partitions
    states = np.arange(5*16384*9,dtype=float).reshape(5,16384,9)
    data,manifest = partitions(states)
    groups = [set(v['trajectories']) for v in manifest['splits'].values()]
    assert len(set.union(*groups)) == 16384
    assert all(not a&b for i,a in enumerate(groups) for b in groups[i+1:])
    assert [len(a[0]) for a in data.values()] == [49152,8192,8192]
    for x,y in data.values():
        np.testing.assert_array_equal(y-x,np.full_like(x,16384*9))


def test_fourfold_extension_preserves_every_original_split_assignment():
    from scripts.run_spt_kuramoto_A_sample_scaling import nested_partitions
    states = np.arange(5*64*9,dtype=float).reshape(5,64,9)
    old = {'splits':{'train':{'trajectories':list(range(12))},
                     'validation':{'trajectories':[12,13]},'test':{'trajectories':[14,15]}}}
    data,manifest = nested_partitions(states,old)
    groups = {k:set(v['trajectories']) for k,v in manifest['splits'].items()}
    assert len(set.union(*groups.values())) == 64
    assert all(not groups[a]&groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    assert [len(v[0]) for v in data.values()] == [192,32,32]
    for name,g in groups.items():
        assert g&set(range(16)) == set(old['splits'][name]['trajectories'])
        x,y = data[name]
        np.testing.assert_array_equal(y-x,np.full_like(x,64*9))
    bad = {'splits':{**old['splits'],'test':{'trajectories':[13,15]}}}
    with pytest.raises(ValueError,match='disjoint'):
        nested_partitions(states,bad)


def model():
    rng = np.random.default_rng(48)
    return NaturalMLPConditionalTM([rng.normal(0, .07, size=s) for s in [(18, 7), (7, 7), (7, 9)]],
                                   [rng.normal(0, .03, size=s) for s in [7, 7, 9]], np.full(9, 20.))


def test_mean_jacobian_and_periodicity():
    m = model()
    x = np.random.default_rng(1).uniform(-np.pi, np.pi, (3, 9))
    mean, jac = m.mean(x, jacobian=True)
    for k in range(9):
        delta = np.zeros_like(x)
        delta[:, k] = 1e-5
        numerical = (m.mean(x + delta) - m.mean(x - delta)) / 2e-5
        np.testing.assert_allclose(jac[:, :, k], numerical, atol=1e-9)
    shifted = x.copy()
    shifted[:, 2] += 2 * np.pi
    np.testing.assert_allclose(np.exp(1j * m.mean(shifted)), np.exp(1j * mean), atol=1e-12)


def test_conditional_density_normalizes_and_generic_loader_roundtrips():
    m = model()
    grid = np.linspace(-np.pi, np.pi, 8193)
    x = np.full((len(grid), 9), .3)
    y = np.zeros_like(x)
    y[:, 3] = grid
    integral = trapezoid(np.exp(m.log_prob(x, y, targets=[3])), grid)
    assert abs(integral - 1) < 1e-10
    loaded = PeriodicConditionalTM.load(m.payload())
    assert loaded.sha256() == m.sha256()
    np.testing.assert_array_equal(loaded.mean(x[:4]), m.mean(x[:4]))


def test_chain_split_has_no_shared_states_or_transition_pairs():
    states = np.arange(513 * 64 * 9, dtype=float).reshape(513, 64, 9)
    data, manifest = chain_partitions(states)
    chains = [set(v['chains']) for v in manifest['splits'].values()]
    assert len(set.union(*chains)) == 64
    assert all(not a & b for i, a in enumerate(chains) for b in chains[i + 1:])
    for name, pair in data.items():
        x, y = pair
        np.testing.assert_array_equal(y - x, np.full_like(x, 64 * 9))
        assert len(x) == manifest['splits'][name]['pairs']
    assert [len(pair[0]) for pair in data.values()] == [24576, 4096, 4096]


def test_expired_absolute_budget_cannot_relaunch(tmp_path):
    import json
    start = time.time() - 1000
    (tmp_path / 'budget.json').write_text(json.dumps(dict(started_epoch=start, deadline_epoch=start + 900, budget_seconds=900)))
    with pytest.raises(TimeoutError, match='cannot be reset'):
        launch(tmp_path)
    assert not (tmp_path / 'process.json').exists()


def test_watchdog_kills_its_real_worker_at_absolute_deadline(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys
    import scripts.run_spt_natural_mlp_pilot as pilot

    real_popen = subprocess.Popen
    children = []
    def controlled_worker(*args, **kwargs):
        child = real_popen([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(pilot.subprocess, 'Popen', controlled_worker)
    monkeypatch.setattr(pilot, 'write_report', lambda folder: None)
    start = time.time()
    (tmp_path / 'budget.json').write_text(json.dumps(dict(started_epoch=start, deadline_epoch=start + .2, budget_seconds=.2)))
    assert pilot.launch(tmp_path) == 124
    assert time.time() - start < 2
    assert children[0].poll() is not None
    assert json.loads((tmp_path / 'pilot.json').read_text())['status'] == 'budget_exhausted'


def test_completed_pilot_is_reused_without_spawning(tmp_path, monkeypatch):
    import json
    import scripts.run_spt_natural_mlp_pilot as pilot

    start = time.time() - 1000
    (tmp_path / 'budget.json').write_text(json.dumps(dict(started_epoch=start, deadline_epoch=start + 900, budget_seconds=900)))
    (tmp_path / 'pilot.json').write_text(json.dumps(dict(status='complete')))
    monkeypatch.setattr(pilot, 'write_report', lambda folder: None)
    def forbidden(*args, **kwargs):
        raise AssertionError('A completed pilot must not launch another worker')
    monkeypatch.setattr(pilot.subprocess, 'Popen', forbidden)
    assert pilot.launch(tmp_path) == 0


def test_real_mlp_jacobian_and_gaussian_density_normalization():
    from scripts.spt_real_natural_mlp import NaturalRealMLPConditionalTM
    rng = np.random.default_rng(73)
    m = NaturalRealMLPConditionalTM(
        [rng.normal(0, .04, size=s) for s in [(9, 7), (7, 7), (7, 9)]],
        [np.zeros(n) for n in [7, 7, 9]], np.linspace(-.1, .1, 9), np.linspace(.4, 1.3, 9), np.full(9, .2))
    x = rng.normal(size=(3, 9))
    _, jac = m.mean(x, jacobian=True)
    for j in range(9):
        step = np.zeros_like(x)
        step[:, j] = 1e-5
        np.testing.assert_allclose(jac[:, :, j], (m.mean(x + step) - m.mean(x - step)) / 2e-5, atol=1e-9)
    grid = np.linspace(-10, 10, 8193)
    source, target = np.zeros((len(grid), 9)), np.zeros((len(grid), 9))
    target[:, 2] = grid
    assert abs(trapezoid(np.exp(m.log_prob(source, target, targets=[2])), grid) - 1) < 1e-10
    assert NaturalRealMLPConditionalTM.load(m.payload()).sha256() == m.sha256()


def test_full_tree_audit_rejects_significant_negative_syn_without_scoring():
    from scripts.run_spt_natural_mlp_three_cases import evaluated_tree
    values = np.array([m.bit_count() for m in range(1, 512)], dtype=float)
    assert evaluated_tree(values, 'A')['status'] == 'exploratory_only'
    values[-1] -= .1
    failed = evaluated_tree(values, 'A')
    assert failed['status'] == 'significant_syn_violation'
    assert 'metrics' not in failed
    assert 'threshold=-0.01' in failed['error']
    assert failed['audit']['significant_violation_count'] > 0


def test_parallel_integrator_construction_preserves_serial_sobol_points():
    from concurrent.futures import ThreadPoolExecutor
    from scipy.special import ndtri
    from scripts.run_spt_natural_mlp_three_cases import make_integrator
    class Toy:
        def sample(self, x, u):
            return x + .02 * ndtri(u)
    def create(pair):
        family, repeat = pair
        i = make_integrator(Toy(), family, 16, 16, repeat, time.monotonic() + 30)
        return i.source, i.target, i.inner
    pairs = [('circular', 0), ('real', 0), ('circular', 1)]
    expected = [create(p) for p in pairs]
    with ThreadPoolExecutor(max_workers=3) as executor:
        actual = list(executor.map(create, pairs))
    for serial, parallel in zip(expected, actual):
        for a, b in zip(serial, parallel):
            np.testing.assert_array_equal(a, b)
