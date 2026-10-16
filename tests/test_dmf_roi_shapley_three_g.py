"""Scientific controls for native-unit, paired three-state attribution."""
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compute_dmf_roi_shapley import prepare_game, permutation_contributions
from scripts.dmf_joint_readout import CommonTargetGame
from scripts.plot_dmf_roi_shapley_three_g import sampling_diagnostics, estimate


def test_prefix_shapley_equals_common_density_ei_game_in_native_nats():
    rng = np.random.default_rng(18)
    prior = np.diag(rng.uniform(.5, 2, 6))
    coefficient = rng.normal(size=(6, 6))
    cross = prior @ coefficient
    covariance = np.block([[prior, cross], [cross.T, coefficient.T @ cross + np.eye(6)]])
    game = CommonTargetGame(covariance)
    c, logs, total = prepare_game(game.conditional[None])
    for order in permutations(range(3)):
        increments = permutation_contributions(c, logs, order)[0] * np.log(2)
        for pos, roi in enumerate(order):
            expected = game.v(order[:pos+1]) - game.v(order[:pos])
            np.testing.assert_allclose(increments[roi], expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(increments.sum(), game.v(range(3)), atol=1e-12, rtol=0)
    np.testing.assert_allclose(total[0] * np.log(2), game.v(range(3)), atol=1e-12, rtol=0)


def test_per_condition_normalization_and_shared_mc_error():
    rng = np.random.default_rng(21)
    xi = np.array([[1., 2., 4.], [2., 3., 7.], [4., 5., 8.]])
    draws = rng.normal(.1, .01, (32, 3, 1, 4))
    draws = np.broadcast_to(draws, (32, 3, 3, 4)).copy()
    samples = draws * xi[None, ..., None]
    pp, condition_se, mean_se, _ = sampling_diagnostics(samples, xi)
    np.testing.assert_allclose(pp, 100 * draws)
    # Perfectly shared seed errors cannot decrease by sqrt(seed_count).
    np.testing.assert_allclose(mean_se, condition_se[:, 0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(pp.mean(0).mean(1), (100 * draws).mean((0, 2)))
    independent_draws = samples.copy()
    independent_draws[:, :, 0] *= 2
    true = (100 * independent_draws / xi[None, ..., None]).mean((0, 2))
    wrong = 100 * independent_draws.mean((0, 2)) / xi.mean(1)[:, None]
    assert np.max(np.abs(true - wrong)) > .1


def test_roi_independence_converges_without_ranking_gate_or_clipping():
    c = np.eye(6)
    for i in range(3):
        c[i, i+3] = c[i+3, i] = .3
    local = np.full((3, 3, 3), -.5 * np.log(1 - .3**2))
    xi = local.sum(-1)
    samples, audit = estimate(np.tile(c, (9, 1, 1)), xi, local, min_pairs=4, max_pairs=8)
    assert audit['converged'] and audit['independent_pairs'] == 4
    np.testing.assert_allclose(samples.mean(0), local, atol=1e-12, rtol=0)
    assert audit['significant_nonnegativity_violation_count'] == 0


def test_nonnegative_violation_reports_native_threshold_and_count(monkeypatch):
    c = np.eye(6)
    for i in range(3):
        c[i, i+3] = c[i+3, i] = .3
    xi = np.full((3, 3), -.5 * 3 * np.log(1 - .3**2))
    def bad_marginals(cov, logs, order):
        value = np.zeros_like(logs)
        value[:, 0] = -1e-5 / np.log(2)
        return value
    monkeypatch.setattr('scripts.plot_dmf_roi_shapley_three_g.permutation_contributions', bad_marginals)
    with pytest.raises(ArithmeticError, match=r'min=-1e-05 nats; threshold=-1e-08 nats; affected=9'):
        estimate(np.tile(c, (9, 1, 1)), xi, np.zeros((3, 3, 3)), min_pairs=4, max_pairs=4)


def test_spt_order_bands_match_exact_coalition_shapley_and_recombine():
    from scripts.plot_dmf_roi_order_attribution import block_marginals, tree_budget
    rng = np.random.default_rng(31)
    coefficient = rng.normal(size=(8, 8))
    prior = np.eye(8)
    game = CommonTargetGame(np.block([[prior, coefficient],
                                     [coefficient.T, coefficient.T @ coefficient + np.eye(8)]]))
    def node(members, children=()):
        return dict(indices=list(members), children=list(children),
                    xi_bits=game.v(members)/np.log(2))
    pair = node([0, 1], [node([0]), node([1])])
    triple = node([0, 1, 2], [pair, node([2])])
    tree = node(range(4), [triple, node([3])])
    blocks, totals, audit = tree_budget(game, tree, 3)
    assert blocks == [(0, 1), (2,), (3,)]
    assert audit['high_order_node_count'] == 2  # Includes exactly order N.
    c, logs, _ = prepare_game(game.conditional[None])
    lows, highs = [], []
    for p in permutations(range(4)):
        low = block_marginals(c[0], logs[0], p, blocks)
        full = permutation_contributions(c, logs, p)[0]*np.log(2)
        high = full-low
        def low_game(s):
            return sum(game.v(set(s) & set(b)) for b in blocks)
        for pos, roi in enumerate(p):
            low_delta = low_game(p[:pos+1])-low_game(p[:pos])
            high_delta = game.v(p[:pos+1])-game.v(p[:pos])-low_delta
            np.testing.assert_allclose([low[roi], high[roi]], [low_delta, high_delta], atol=1e-12)
        assert min(low.min(), high.min()) >= -1e-12
        np.testing.assert_allclose([low.sum(), high.sum()], totals, atol=1e-12)
        lows.append(low); highs.append(high)
    full_mean = np.mean([permutation_contributions(c, logs, p)[0]*np.log(2)
                        for p in permutations(range(4))], axis=0)
    np.testing.assert_allclose(np.mean(lows, axis=0)+np.mean(highs, axis=0), full_mean, atol=1e-12)
    # Full-coalition high mass is the sum of the original 3- and 4-ROI splits.
    expected = ((game.v(range(4))-game.v([0,1,2])-game.v([3]))
                +(game.v([0,1,2])-game.v([0,1])-game.v([2])))
    np.testing.assert_allclose(totals[1], expected, atol=1e-12)
    assert tree_budget(game, tree, 5)[1][1] == 0
    np.testing.assert_allclose(tree_budget(game, tree, 2)[1][0], game.local_xi.sum(), atol=1e-12)


def test_spt_frontier_rejects_nonpartition_and_invalid_threshold():
    from scripts.plot_dmf_roi_order_attribution import frontier_blocks
    leaf = dict(indices=[0], children=[])
    with pytest.raises(ValueError, match='integer >= 2'):
        frontier_blocks(leaf, 1)
    malformed = dict(indices=[0, 1], children=[leaf, leaf])
    with pytest.raises(ValueError, match='binary partition'):
        frontier_blocks(malformed, 3)


def test_order_band_percentages_normalize_each_seed_and_leave_zero_budget_undefined():
    from scripts.plot_dmf_roi_order_attribution import band_percentages
    raw = np.array([[[[1., 1.], [3., 1.]]], [[[0., 0.], [0., 0.]]]])
    totals = raw.sum(-1)
    shares, valid, audit = band_percentages(raw, totals)
    np.testing.assert_allclose(shares[0, 0].mean(0), [62.5, 37.5])
    assert not np.allclose(shares[0, 0].mean(0), 100*raw[0, 0].mean(0)/totals[0, 0].mean())
    np.testing.assert_allclose(shares[valid].sum(-1), 100.)
    assert valid[0].all() and not valid[1].any()
    assert np.isnan(shares[1]).all() and audit['undefined_condition_count'] == 2
    with pytest.raises(ArithmeticError, match=r'threshold=-1e-08 nats; affected=1'):
        band_percentages(np.array([[-1e-5, 1.]]), np.array([1.-1e-5]))
    with pytest.raises(ArithmeticError, match='does not sum'):
        band_percentages(raw, totals+1.)
