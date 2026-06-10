from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lorenz3d_multihorizon_peid import (
    LorenzConfig,
    _full_rho_grid,
    build_natural_dataset,
    compute_dynamics_diagnostics,
    evaluate_conditional_wing_peid,
    evaluate_matched_peid,
    fit_direct_mlp,
    lorenz_field,
    run_experiment,
    simulate_transition,
    summarize_two_source_synergy,
)


def test_lorenz_field_and_horizon_transition_match_protocol() -> None:
    config = LorenzConfig(rho=28.0, dt=0.01, tau=0.05)
    states = np.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 8.0]])

    expected = np.column_stack(
        [
            10.0 * (states[:, 1] - states[:, 0]),
            states[:, 0] * (28.0 - states[:, 2]) - states[:, 1],
            states[:, 0] * states[:, 1] - (8.0 / 3.0) * states[:, 2],
        ]
    )
    transitioned = simulate_transition(states, config=config)

    assert config.integration_steps == 5
    assert np.allclose(lorenz_field(states, rho=28.0), expected)
    assert transitioned.shape == states.shape
    assert np.isfinite(transitioned).all()
    assert not np.array_equal(transitioned, states)


def test_horizon_must_be_an_integer_multiple_of_dt() -> None:
    config = LorenzConfig(rho=28.0, dt=0.01, tau=0.015)

    try:
        _ = config.integration_steps
    except ValueError as exc:
        assert "integer multiple" in str(exc)
    else:
        raise AssertionError("invalid horizon should raise ValueError")


def test_natural_dataset_uses_disjoint_trajectory_splits() -> None:
    dataset = build_natural_dataset(
        rho=28.0,
        tau=0.05,
        dt=0.01,
        trajectory_count=12,
        burnin_steps=12,
        record_steps=30,
        max_samples=(80, 30, 30),
        seed=7,
    )

    assert dataset.train.inputs.shape[1] == 3
    assert dataset.train.inputs.shape == dataset.train.targets.shape
    assert dataset.validation.inputs.shape == dataset.validation.targets.shape
    assert dataset.test.inputs.shape == dataset.test.targets.shape
    assert set(dataset.train.trajectory_ids).isdisjoint(dataset.validation.trajectory_ids)
    assert set(dataset.train.trajectory_ids).isdisjoint(dataset.test.trajectory_ids)
    assert set(dataset.validation.trajectory_ids).isdisjoint(dataset.test.trajectory_ids)
    assert set(np.unique(dataset.train.trajectory_ids)) == set(range(8))
    assert set(np.unique(dataset.validation.trajectory_ids)) == {8, 9}
    assert set(np.unique(dataset.test.trajectory_ids)) == {10, 11}
    expected_targets = simulate_transition(
        dataset.test.inputs,
        config=LorenzConfig(rho=28.0, dt=0.01, tau=0.05),
    )
    assert np.allclose(dataset.test.targets, expected_targets)


def test_direct_mlp_is_horizon_specific_and_reports_prediction_baselines() -> None:
    short_data = build_natural_dataset(
        rho=28.0,
        tau=0.01,
        burnin_steps=12,
        record_steps=36,
        max_samples=(120, 30, 30),
        seed=11,
    )
    long_data = build_natural_dataset(
        rho=28.0,
        tau=0.05,
        burnin_steps=12,
        record_steps=36,
        max_samples=(120, 30, 30),
        seed=11,
    )

    short_model = fit_direct_mlp(
        short_data,
        rho=28.0,
        tau=0.01,
        seed=3,
        hidden_widths=(16, 16),
        epochs=40,
        patience=8,
        batch_size=32,
    )
    long_model = fit_direct_mlp(
        long_data,
        rho=28.0,
        tau=0.05,
        seed=3,
        hidden_widths=(16, 16),
        epochs=40,
        patience=8,
        batch_size=32,
    )
    short_prediction = short_model.predict(short_data.test.inputs)

    assert short_model.tau == 0.01
    assert long_model.tau == 0.05
    assert short_prediction.shape == short_data.test.targets.shape
    assert np.isfinite(short_prediction).all()
    for key in ("nrmse", "r2", "correlation", "constant_mse_ratio", "linear_mse_ratio"):
        assert key in short_model.metrics
        assert np.isfinite(short_model.metrics[key])
    assert short_model.best_epoch >= 0


def test_two_source_synergy_preserves_negative_estimates() -> None:
    left = np.arange(12, dtype=float).reshape(-1, 1)
    right = left + 1.0
    target = left - 2.0

    def estimator(source: np.ndarray, _: np.ndarray) -> float:
        return 0.2 if source.shape[1] == 2 else 0.7

    summary = summarize_two_source_synergy(left, right, target, estimator=estimator)

    assert summary["left_ei"] == 0.7
    assert summary["right_ei"] == 0.7
    assert np.isclose(summary["joint_ei"], 0.2)
    assert np.isclose(summary["synergy"], -1.2)


def test_matched_peid_enumerates_all_source_pairs_and_targets() -> None:
    dataset = build_natural_dataset(
        rho=28.0,
        tau=0.05,
        burnin_steps=12,
        record_steps=40,
        max_samples=(140, 35, 35),
        seed=13,
    )
    model = fit_direct_mlp(
        dataset,
        rho=28.0,
        tau=0.05,
        seed=5,
        hidden_widths=(16, 16),
        epochs=35,
        patience=8,
        batch_size=32,
    )
    bounds = np.array([[-18.0, 18.0], [-24.0, 24.0], [0.0, 50.0]])

    result = evaluate_matched_peid(
        config=LorenzConfig(rho=28.0, tau=0.05),
        model=model,
        intervention_bounds=bounds,
        samples=240,
        seed=17,
        bootstrap_replicates=0,
    )

    assert result["intervention_sample_count"] == 240
    assert len(result["rows"]) == 9
    assert len({(row["sources"], row["target"]) for row in result["rows"]}) == 9
    assert result["oracle_intervention_digest"] == result["mlp_intervention_digest"]
    for row in result["rows"]:
        assert row["sources"] in {"x+y", "x+z", "y+z"}
        assert row["target"] in {"x_tau", "y_tau", "z_tau"}
        for prefix in ("oracle", "mlp"):
            for suffix in ("left_ei", "right_ei", "joint_ei", "synergy"):
                assert np.isfinite(row[f"{prefix}_{suffix}"])
        for field in ("mlp_ablation_score", "mlp_response_interaction", "mlp_jacobian_strength"):
            assert np.isfinite(row[field])
    assert -1.0 <= result["recovery"]["spearman"] <= 1.0
    assert 0.0 <= result["recovery"]["top2_precision"] <= 1.0
    assert result["recovery"]["top2_precision"] == result["recovery"]["top2_recall"]


def test_dynamics_diagnostics_and_full_grid_are_pre_registered() -> None:
    dataset = build_natural_dataset(
        rho=28.0,
        tau=0.05,
        burnin_steps=12,
        record_steps=40,
        max_samples=(140, 35, 35),
        seed=23,
    )

    diagnostics = compute_dynamics_diagnostics(
        dataset.train,
        config=LorenzConfig(rho=28.0, tau=0.05),
        lyapunov_samples=12,
    )

    assert set(diagnostics) == {"state_variance", "wing_switch_rate", "finite_time_lyapunov"}
    assert all(np.isfinite(value) for value in diagnostics.values())
    assert 0.0 <= diagnostics["wing_switch_rate"] <= 1.0
    grid = set(_full_rho_grid())
    assert {0.5, 5.0, 15.0, 22.0, 24.0, 24.5, 25.0, 28.0, 35.0, 45.0, 55.0}.issubset(grid)


def test_conditional_wing_peid_uses_two_disjoint_intervention_boxes() -> None:
    dataset = build_natural_dataset(
        rho=28.0,
        tau=0.01,
        burnin_steps=12,
        record_steps=40,
        max_samples=(140, 35, 35),
        seed=29,
    )
    model = fit_direct_mlp(
        dataset,
        rho=28.0,
        tau=0.01,
        seed=7,
        hidden_widths=(16, 16),
        epochs=30,
        patience=6,
        batch_size=32,
    )

    result = evaluate_conditional_wing_peid(
        config=LorenzConfig(rho=28.0, tau=0.01),
        model=model,
        intervention_bounds=np.array([[-18.0, 18.0], [-24.0, 24.0], [0.0, 50.0]]),
        samples=120,
        seed=31,
        bootstrap_replicates=0,
    )

    assert set(result) == {"left", "right"}
    assert result["left"]["intervention_bounds"][0][1] == 0.0
    assert result["right"]["intervention_bounds"][0][0] == 0.0
    assert len(result["left"]["rows"]) == len(result["right"]["rows"]) == 9


def test_smoke_experiment_writes_artifacts_and_reuses_cache(tmp_path: Path) -> None:
    kwargs = dict(
        rhos=(28.0,),
        horizons=(0.01, 0.05),
        training_seeds=(0,),
        representative_rhos=(28.0,),
        result_dir=tmp_path / "results",
        figure_dir=tmp_path / "figures",
        report_path=tmp_path / "report.md",
        burnin_steps=12,
        record_steps=40,
        max_samples=(140, 35, 35),
        hidden_widths=(16, 16),
        epochs=30,
        patience=6,
        batch_size=32,
        peid_samples=220,
        bootstrap_replicates=0,
        seed=19,
    )

    first = run_experiment(**kwargs)
    second = run_experiment(**kwargs)

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert Path(first["summary_path"]).exists()
    assert Path(first["prediction_cache_path"]).suffix == ".npz"
    assert Path(first["prediction_cache_path"]).exists()
    assert Path(first["report_path"]).exists()
    assert len(first["figure_paths"]) >= 3
    assert all(Path(path).suffix == ".png" and Path(path).exists() for path in first["figure_paths"])
    payload = json.loads(Path(first["summary_path"]).read_text(encoding="utf-8"))
    assert len(payload["prediction_rows"]) == 2
    assert len(payload["peid_runs"]) == 2
    report = Path(first["report_path"]).read_text(encoding="utf-8")
    assert "Lorenz-3D" in report
    assert "有限分辨率" in report


def test_cli_can_be_invoked_by_script_path() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "lorenz3d_multihorizon_peid.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--mode" in completed.stdout
