from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_mixed_order_kuramoto_hierarchy import (
    PAIRWISE_MODULE,
    TRIADIC_MODULE,
    greedy_decision_trace,
    mixed_order_derivative,
    phase_state_features,
    phase_source_blocks,
    phase_transport_context,
    response_observable,
    simulate_future_phases,
)


def test_pairwise_and_triadic_terms_have_disjoint_planted_support() -> None:
    phases = np.array([[0.1, -0.4, 0.8, -1.0, 0.3]])
    base = mixed_order_derivative(
        phases, pairwise_coupling=1.8, triadic_coupling=2.0
    )

    changed_pair = phases.copy()
    changed_pair[:, :2] += np.array([0.5, -0.2])
    pair_perturbed = mixed_order_derivative(
        changed_pair, pairwise_coupling=1.8, triadic_coupling=2.0
    )
    assert not np.allclose(base[:, :2], pair_perturbed[:, :2])
    assert np.allclose(base[:, 2:], pair_perturbed[:, 2:])

    changed_triad = phases.copy()
    changed_triad[:, 2:] += np.array([0.5, -0.2, 0.1])
    triad_perturbed = mixed_order_derivative(
        changed_triad, pairwise_coupling=1.8, triadic_coupling=2.0
    )
    assert np.allclose(base[:, :2], triad_perturbed[:, :2])
    assert not np.allclose(base[:, 2:], triad_perturbed[:, 2:])


def test_weak_cross_coupling_changes_both_planted_blocks() -> None:
    phases = np.array([[0.1, -0.4, 0.8, -1.0, 0.3]])
    disconnected = mixed_order_derivative(
        phases,
        pairwise_coupling=1.8,
        triadic_coupling=2.0,
        cross_coupling=0.0,
    )
    weakly_connected = mixed_order_derivative(
        phases,
        pairwise_coupling=1.8,
        triadic_coupling=2.0,
        cross_coupling=0.04,
    )
    assert not np.allclose(disconnected[:, :2], weakly_connected[:, :2])
    assert not np.allclose(disconnected[:, 2:], weakly_connected[:, 2:])


def test_vector_field_is_equivariant_to_global_phase_shift() -> None:
    phases = np.array([[0.1, -0.4, 0.8, -1.0, 0.3]])
    baseline = mixed_order_derivative(
        phases, pairwise_coupling=1.8, triadic_coupling=2.0
    )
    shifted = mixed_order_derivative(
        phases + 1.37, pairwise_coupling=1.8, triadic_coupling=2.0
    )
    assert np.allclose(baseline, shifted)


def test_response_aggregates_both_mechanism_channels() -> None:
    phases = np.array([[0.1, -0.4, 0.8, -1.0, 0.3]])
    derivative = mixed_order_derivative(
        phases, pairwise_coupling=1.8, triadic_coupling=2.0
    )
    response = response_observable(derivative)
    assert response.shape == (1, 1)
    assert np.allclose(response[:, 0], derivative[:, 0] + derivative[:, 4])


def test_source_blocks_encode_every_oscillator_on_the_unit_circle() -> None:
    phases = np.array([[0.1, -0.4, 0.8, -1.0, 0.3]])
    blocks = phase_source_blocks(phases)
    assert tuple(blocks) == PAIRWISE_MODULE + TRIADIC_MODULE
    assert np.allclose(blocks["theta1"][0], [np.cos(0.1), np.sin(0.1)])
    assert np.allclose(
        blocks["theta5"][0],
        [np.cos(0.3), np.sin(0.3), np.cos(0.6), np.sin(0.6)],
    )


def test_full_future_target_contains_every_oscillator() -> None:
    phases = np.array([[0.1, -0.4, 0.8, -1.0, 0.3]])
    future = simulate_future_phases(
        phases,
        pairwise_coupling=1.8,
        triadic_coupling=2.0,
        cross_coupling=0.12,
        process_noise=0.0,
        tau=0.2,
        dt=0.01,
        seed=0,
    )
    encoded = phase_state_features(future)
    assert future.shape == (1, 5)
    assert encoded.shape == (1, 10)
    assert np.allclose(
        np.linalg.norm(encoded.reshape(1, 5, 2), axis=2),
        1.0,
    )


def test_greedy_trace_uses_the_same_maximum_capture_rule() -> None:
    names = ("a", "b", "c")
    table = {
        ("a",): 0.0,
        ("b",): 0.0,
        ("c",): 0.0,
        ("a", "b"): 0.4,
        ("a", "c"): 0.0,
        ("b", "c"): 0.0,
        ("a", "b", "c"): 1.0,
    }
    trace = greedy_decision_trace(names, table, {name: 0.0 for name in names})

    assert trace["action"] == "split"
    assert trace["selected_left"] == ["a", "b"]
    assert trace["selected_right"] == ["c"]
    assert np.isclose(trace["residual_bits"], 0.6)
    assert trace["children"][0]["action"] == "split"
    assert all(child["action"] == "terminal" for child in trace["children"][0]["children"])


def test_joint_tm_context_is_fixed_for_all_subset_marginals() -> None:
    phases = np.array(
        [
            [0.1, -0.4, 0.8, -1.0, 0.3],
            [-0.2, 0.5, -0.7, 0.9, -0.1],
        ]
    )
    context = phase_transport_context(phases)

    assert context.shape == (2, 16)
    assert np.allclose(context[:, :10], phase_state_features(phases))
    assert np.allclose(context[:, 10:13], np.cos(2.0 * phases[:, 2:]))
    assert np.allclose(context[:, 13:], np.sin(2.0 * phases[:, 2:]))
