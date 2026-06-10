from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.coupled_standard_map_peid import (
    StandardMapConfig,
    analytic_interaction_strength,
    analytic_pairwise_strengths,
    build_trajectory_dataset,
    coupled_impulses,
    evaluate_trajectory_gate,
    periodic_features,
    run_experiment,
    wrap_angle,
)


def test_wrap_angle_and_periodic_features_are_periodic() -> None:
    values = np.array([[-np.pi, 0.2, np.pi - 0.1, -0.7]])
    shifted = values + 2.0 * np.pi

    assert np.all(wrap_angle(values) >= -np.pi)
    assert np.all(wrap_angle(values) < np.pi)
    np.testing.assert_allclose(periodic_features(values), periodic_features(shifted), atol=1e-12)


def test_coupled_impulses_match_equations_and_ground_truth() -> None:
    config = StandardMapConfig(k=1.5, coupling=0.8, noise_std=0.0)
    states = np.array([[0.2, -0.3, 1.1, 0.4]])
    impulses = coupled_impulses(states, config=config)

    expected_1 = 1.5 * np.sin(0.2) + 0.8 * np.sin(1.1 - 0.2)
    expected_2 = 1.5 * np.sin(1.1) + 0.8 * np.sin(0.2 - 1.1)
    np.testing.assert_allclose(impulses[0], [expected_1, expected_2])
    assert np.isclose(analytic_interaction_strength(config), 0.8**2 / 2.0)
    strengths = analytic_pairwise_strengths(config)
    assert np.isclose(strengths["q1->I1"], (1.5**2 + 0.8**2) / 2.0)
    assert np.isclose(strengths["q2->I1"], 0.8**2 / 2.0)
    assert strengths["p1->I1"] == 0.0


def test_trajectory_splits_are_disjoint() -> None:
    dataset = build_trajectory_dataset(
        StandardMapConfig(noise_std=0.01),
        trajectory_count=8,
        steps_per_trajectory=40,
        seed=4,
    )

    train_ids = set(dataset.train.trajectory_ids.tolist())
    validation_ids = set(dataset.validation.trajectory_ids.tolist())
    test_ids = set(dataset.test.trajectory_ids.tolist())
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)


def test_gate_requires_every_preregistered_condition() -> None:
    passing = {
        "r2": 0.995,
        "nrmse": 0.04,
        "circular_mae": 0.03,
        "spearman": 0.95,
        "true_pair_top_both": True,
        "true_pair_max_relative_error": 0.15,
        "momentum_max_ei": 0.015,
    }
    assert evaluate_trajectory_gate(passing)["passed"] is True

    failing = dict(passing)
    failing["spearman"] = 0.89
    result = evaluate_trajectory_gate(failing)
    assert result["passed"] is False
    assert "spearman" in result["failed_checks"]


def test_smoke_experiment_creates_artifacts(tmp_path: Path) -> None:
    result = run_experiment(
        mode="smoke",
        seed=2,
        result_dir=tmp_path / "results",
        figure_dir=tmp_path / "figures",
        report_path=tmp_path / "report.md",
    )

    assert Path(result["summary_path"]).exists()
    assert Path(result["figure_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert result["trajectory"]["peid"]["intervention_digest"] == result["oracle"]["intervention_digest"]
    assert len(result["oracle"]["hyperedges"]) == 12
    assert result["mixed_ran"] is (not result["trajectory_gate"]["passed"])
