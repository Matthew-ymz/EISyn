from __future__ import annotations

import itertools
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_dmf_roi_shapley import Moments, permutation_contributions, prepare_game
from scripts.analyze_dmf_schaefer100_xi_hierarchy_tree import ConditionalBlockXiOracle


def test_cholesky_increments_match_original_coalition_game():
    rng = np.random.default_rng(17)
    a = rng.normal(size=(6, 6))
    c = a @ a.T + np.eye(6)
    cov, logs, totals = prepare_game(c[None])
    oracle = ConditionalBlockXiOracle(c, [(i, i+3) for i in range(3)])
    increments = []
    exact = np.zeros(3)
    for p in itertools.permutations(range(3)):
        values = permutation_contributions(cov, logs, p)[0]
        increments.append(values)
        for position, roi in enumerate(p):
            reference = oracle.xi(p[:position+1]) - oracle.xi(p[:position])
            np.testing.assert_allclose(values[roi], reference, atol=1e-12)
            exact[roi] += reference / 6
        np.testing.assert_allclose(values.sum(), totals[0], atol=1e-12)
    np.testing.assert_allclose(np.mean(increments, axis=0), exact, atol=1e-12)


def test_independent_roi_game_retains_local_ei_xi():
    c = np.eye(6)
    for i in range(3):
        c[i, i+3] = c[i+3, i] = 0.6
    cov, logs, totals = prepare_game(c[None])
    local_bits = -.5 * np.log2(1 - 0.6**2)
    np.testing.assert_allclose(totals, 3 * local_bits, atol=1e-12)
    values = permutation_contributions(cov, logs, [2, 0, 1])
    np.testing.assert_allclose(values, local_bits, atol=1e-12)
    np.testing.assert_allclose(values - local_bits, 0, atol=1e-12)


def test_exchangeable_game_has_equal_exact_shares():
    c = 0.3 * np.ones((6, 6)) + 0.7 * np.eye(6)
    cov, logs, totals = prepare_game(c[None])
    mean = np.mean([permutation_contributions(cov, logs, p)[0]
                    for p in itertools.permutations(range(3))], axis=0)
    np.testing.assert_allclose(mean, totals[0]/3, atol=1e-12)


def test_moments_report_sampling_se():
    data = np.random.default_rng(1).normal(size=(20, 3))
    moments = Moments((3,))
    for row in data:
        moments.add(row)
    np.testing.assert_allclose(moments.mean, data.mean(axis=0))
    np.testing.assert_allclose(moments.se(), data.std(axis=0, ddof=1)/np.sqrt(20))


def test_floor_active_covariance_fails_instead_of_changing_game():
    with pytest.raises(ValueError, match="floor is active"):
        prepare_game(np.diag([1.0, 1e-14])[None])


def test_invalid_permutation_fails():
    c, logs, _ = prepare_game(np.eye(6)[None])
    with pytest.raises(ValueError, match="permutation"):
        permutation_contributions(c, logs, [1, 1, 0])
