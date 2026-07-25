from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_greedy_hierarchy_kuramoto import (
    FREQUENCIES,
    PLANTED_MODULES,
    coupling_matrix,
    is_planted_split,
    kuramoto_derivative,
)


def test_coupling_matrix_has_two_normalized_planted_communities() -> None:
    matrix = coupling_matrix(within_coupling=1.5, cross_coupling=0.3)

    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert np.allclose(matrix[:3, :3][~np.eye(3, dtype=bool)], 0.75)
    assert np.allclose(matrix[:3, 3:], 0.1)


def test_zero_cross_coupling_blocks_cross_module_influence() -> None:
    phases = np.array([[0.1, -0.4, 0.8, -1.0, 0.3, 1.2]])
    changed = phases.copy()
    changed[:, 3:] += np.array([0.7, -0.2, 1.1])

    baseline = kuramoto_derivative(phases, within_coupling=1.5, cross_coupling=0.0)
    perturbed = kuramoto_derivative(changed, within_coupling=1.5, cross_coupling=0.0)

    assert np.allclose(baseline[:, :3], perturbed[:, :3])


def test_vector_field_is_equivariant_to_global_phase_shift() -> None:
    phases = np.array([[0.1, -0.4, 0.8, -1.0, 0.3, 1.2]])

    baseline = kuramoto_derivative(phases, within_coupling=1.5, cross_coupling=0.4)
    shifted = kuramoto_derivative(phases + 1.37, within_coupling=1.5, cross_coupling=0.4)

    assert np.allclose(baseline, shifted)
    assert np.allclose(baseline.mean(axis=1), FREQUENCIES.mean())


def test_planted_split_is_order_invariant() -> None:
    assert is_planted_split(PLANTED_MODULES[0], PLANTED_MODULES[1])
    assert is_planted_split(PLANTED_MODULES[1], PLANTED_MODULES[0])
    assert not is_planted_split(("theta1",), tuple(name for name in sum(PLANTED_MODULES, ()) if name != "theta1"))
