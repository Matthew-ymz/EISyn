import numpy as np
import pytest

from scripts.dmf_response_benchmark import (
    MIOracle, score, full_phi_lu, check_syn, weaken_return, responses,
)
from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module


def oracle(k, seed=10):
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(2*k, 2*k)) * .15
    d = np.eye(2*k)
    covariance = np.block([[d, b], [b.T, b.T @ b + np.eye(2*k)]])
    return MIOracle(covariance)


@pytest.mark.parametrize('k', [2, 3, 4])
@pytest.mark.parametrize('seed', [11, 12, 13, 14])
def test_product_lattice_and_scalar(k, seed):
    q = oracle(k, seed); a = tuple(range(k))
    value, closure, atoms = full_phi_lu(q, a)
    assert atoms == {2:16, 3:324, 4:27556}[k]
    assert abs(closure) < 1e-10
    assert value == pytest.approx(score(q, a, 'phi_r_lu'), abs=1e-10)
    assert score(q, a, 'xi') == pytest.approx(score(q, a, 'xi_fast'), abs=1e-10)
    check_syn([score(q, a, 'xi')])
    if k == 2:
        assert value == pytest.approx(score(q, a, 'pair_phi_r'), abs=1e-10)
        assert score(q, a, 'xi') == pytest.approx(score(q, a, 'o_increment'), abs=1e-10)


def test_permutation_and_empty_single_block():
    q = oracle(4)
    permutation = np.array([2, 0, 3, 1])
    dims = np.r_[permutation, permutation+4, permutation+8, permutation+12]
    permuted = MIOracle(q.cov[np.ix_(dims, dims)])
    for method in ('xi', 'phi_r_lu', 'wms', 'o_increment', 'weak_phi_r', 'pair_phi_r'):
        assert score(q, range(4), method) == pytest.approx(score(permuted, range(4), method), abs=1e-10)
        assert score(q, (), method) == 0
        assert score(q, (0,), method) == pytest.approx(0, abs=1e-10)


def test_syn_rule_reports_and_retains_raw():
    raw = np.array([-.5e-8, 1.])
    assert check_syn(raw) == 1
    assert raw[0] < 0
    with pytest.raises(ArithmeticError, match='affected=2'):
        check_syn([-1e-7, -2e-7])


def test_return_direction_budget_and_tonic_compensation():
    sc = np.arange(1, 17, dtype=float).reshape(4, 4); np.fill_diagonal(sc, 0)
    a = (0, 2); modified, cut, alpha, strength = weaken_return(sc, a, 5.)
    assert cut.sum() == pytest.approx(5.)
    assert np.all(cut[[1, 3]] == 0)  # Outputs from A to others are intact.
    assert np.all(cut[:, [0, 2]] == 0)  # Internal connections intact.
    ref = np.array([.1, .2, .3, .4])
    assert modified @ ref + cut @ ref == pytest.approx(sc @ ref)
    assert alpha * strength == pytest.approx(5.)


def test_response_zero_budget_and_sham():
    dmf = load_dmf_module(); p = dmf.DMFParameters(dt=.001, sigma=.01)
    sc = np.array([[0., .2], [.2, 0.]])
    initial = (np.full((2, 2), .02), np.full((2, 2), .025))
    q = responses(dmf, sc, np.ones(2), 1., p, initial, 0, [(0,)], 0., .00382, 999)
    assert np.array_equal(q['rms'][0], q['rms'][1])
    assert np.all(q['loss'] == 0)
    q = responses(dmf, sc, np.ones(2), 1., p, initial, 0, [(0,)], .05, 0., 999)
    assert np.all(q['rms'] == 0)
    assert np.all(q['loss'] == 0)
