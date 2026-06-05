from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def test_load_lorzen_transition_csvs_preserves_headerless_rows(tmp_path: Path) -> None:
    from exp.TM.lorzen_tm_ei import load_lorzen_transition_csvs

    current = pd.DataFrame(np.arange(24, dtype=float).reshape(3, 8))
    next_state = current + 0.5
    current.to_csv(tmp_path / "yt.csv", header=False, index=False)
    next_state.to_csv(tmp_path / "yt+1.csv", header=False, index=False)

    source, target = load_lorzen_transition_csvs(tmp_path)

    assert source.shape == (3, 8)
    assert target.shape == (3, 8)
    assert list(source.columns) == [f"M{i}" for i in range(1, 9)]
    assert float(source.iloc[0, 0]) == 0.0
    assert float(target.iloc[-1, -1]) == 23.5


def test_lorenz96_groundtruth_uses_source_to_target_orientation() -> None:
    from exp.TM.lorzen_tm_ei import lorenz96_groundtruth_adjacency

    adjacency = lorenz96_groundtruth_adjacency(8)

    assert adjacency.shape == (8, 8)
    assert adjacency.iloc[0].tolist() == [1, 1, 0, 0, 0, 0, 1, 1]
    assert adjacency.iloc[2].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert adjacency.iloc[7].tolist() == [1, 0, 0, 0, 0, 1, 1, 1]


def test_prepare_lagged_transition_samples_reconstructs_longer_gap(tmp_path: Path) -> None:
    from exp.TM.lorzen_tm_ei import load_lorzen_transition_csvs, prepare_lagged_transition_samples

    trajectory = pd.DataFrame(np.arange(48, dtype=float).reshape(6, 8))
    trajectory.iloc[:-1].to_csv(tmp_path / "yt.csv", header=False, index=False)
    trajectory.iloc[1:].to_csv(tmp_path / "yt+1.csv", header=False, index=False)
    source, target = load_lorzen_transition_csvs(tmp_path)

    lagged_source, lagged_target = prepare_lagged_transition_samples(source, target, lag=3)

    assert lagged_source.shape == (3, 8)
    assert lagged_target.shape == (3, 8)
    assert float(lagged_source.iloc[0, 0]) == 0.0
    assert float(lagged_target.iloc[0, 0]) == 24.0
    assert float(lagged_source.iloc[-1, -1]) == 23.0
    assert float(lagged_target.iloc[-1, -1]) == 47.0


def test_run_lorzen_tm_ei_experiment_exports_summary_artifacts(tmp_path: Path) -> None:
    from exp.TM.lorzen_tm_ei import run_lorzen_tm_ei_experiment

    rng = np.random.default_rng(4)
    x = rng.normal(size=(80, 8))
    y = np.zeros_like(x)
    for target_index in range(8):
        y[:, target_index] = (
            0.8 * x[:, target_index]
            + 0.4 * x[:, (target_index - 1) % 8]
            - 0.2 * x[:, (target_index - 2) % 8]
            + 0.3 * x[:, (target_index + 1) % 8]
            + 0.03 * rng.normal(size=x.shape[0])
        )
    pd.DataFrame(x).to_csv(tmp_path / "yt.csv", header=False, index=False)
    pd.DataFrame(y).to_csv(tmp_path / "yt+1.csv", header=False, index=False)

    output_dir = tmp_path / "out"
    result = run_lorzen_tm_ei_experiment(tmp_path, output_dir=output_dir, top_k=4)

    assert result["ei_matrix"].shape == (8, 8)
    assert result["edge_table"].shape[0] == 64
    assert result["metrics"]["top_k"] == 4
    assert result["metrics"]["lag"] == 1
    assert np.isfinite(result["ei_matrix"].to_numpy(dtype=float)).all()
    assert (output_dir / "lorzen_tm_ei_matrix.csv").exists()
    assert (output_dir / "lorzen_tm_ei_edges.csv").exists()
    assert (output_dir / "lorzen_tm_ei_summary.json").exists()
    assert (output_dir / "lorzen_tm_ei_heatmap.png").exists()


def test_delta_target_mode_removes_raw_next_state_diagonal_dominance(tmp_path: Path) -> None:
    from exp.TM.lorzen_tm_ei import run_lorzen_tm_ei_experiment

    rng = np.random.default_rng(12)
    trajectory = np.zeros((101, 8), dtype=float)
    trajectory[0] = rng.normal(size=8)
    for time_index in range(100):
        current = trajectory[time_index]
        delta = np.zeros(8, dtype=float)
        for target_index in range(8):
            delta[target_index] = (
                0.08 * current[(target_index - 1) % 8]
                - 0.07 * current[(target_index - 2) % 8]
                + 0.06 * current[(target_index + 1) % 8]
                + 0.002 * rng.normal()
            )
        trajectory[time_index + 1] = current + delta
    pd.DataFrame(trajectory[:-1]).to_csv(tmp_path / "yt.csv", header=False, index=False)
    pd.DataFrame(trajectory[1:]).to_csv(tmp_path / "yt+1.csv", header=False, index=False)

    next_result = run_lorzen_tm_ei_experiment(tmp_path, output_dir=tmp_path / "next", target_mode="next")
    delta_result = run_lorzen_tm_ei_experiment(tmp_path, output_dir=tmp_path / "delta", target_mode="delta")

    next_matrix = next_result["ei_matrix"].to_numpy(dtype=float)
    delta_matrix = delta_result["ei_matrix"].to_numpy(dtype=float)
    off_diagonal_mask = ~np.eye(8, dtype=bool)

    next_diag_ratio = np.diag(next_matrix).mean() / next_matrix[off_diagonal_mask].mean()
    delta_diag_ratio = np.diag(delta_matrix).mean() / delta_matrix[off_diagonal_mask].mean()

    assert next_diag_ratio > 1.0
    assert delta_diag_ratio < next_diag_ratio
    assert delta_result["metrics"]["target_mode"] == "delta"


def test_surrogate_intervention_uses_box_width_and_sample_count(tmp_path: Path) -> None:
    from exp.TM.lorzen_tm_ei import run_lorzen_tm_ei_experiment

    rng = np.random.default_rng(21)
    x = rng.normal(size=(120, 8))
    delta = np.zeros_like(x)
    for target_index in range(8):
        delta[:, target_index] = (
            x[:, (target_index - 1) % 8] * x[:, (target_index + 1) % 8]
            - 0.5 * x[:, target_index]
            + 0.03 * rng.normal(size=x.shape[0])
        )
    pd.DataFrame(x).to_csv(tmp_path / "yt.csv", header=False, index=False)
    pd.DataFrame(x + delta).to_csv(tmp_path / "yt+1.csv", header=False, index=False)

    result = run_lorzen_tm_ei_experiment(
        tmp_path,
        output_dir=tmp_path / "surrogate",
        estimator_mode="surrogate_intervention",
        target_mode="delta",
        box_width=1.5,
        sample_count=96,
        seed=5,
    )

    assert result["metrics"]["estimator_mode"] == "surrogate_intervention"
    assert result["metrics"]["box_width"] == 1.5
    assert result["metrics"]["sample_count"] == 96
    assert np.isfinite(result["ei_matrix"].to_numpy(dtype=float)).all()


def test_run_lorzen_tm_ei_experiment_accepts_lagged_next_target(tmp_path: Path) -> None:
    from exp.TM.lorzen_tm_ei import run_lorzen_tm_ei_experiment

    rng = np.random.default_rng(31)
    trajectory = rng.normal(size=(14, 8)).cumsum(axis=0)
    pd.DataFrame(trajectory[:-1]).to_csv(tmp_path / "yt.csv", header=False, index=False)
    pd.DataFrame(trajectory[1:]).to_csv(tmp_path / "yt+1.csv", header=False, index=False)

    result = run_lorzen_tm_ei_experiment(tmp_path, output_dir=tmp_path / "lagged", lag=4)

    assert result["metrics"]["lag"] == 4
    assert result["metrics"]["sample_count"] == 10
    assert result["source"].shape == (10, 8)
    assert result["target"].shape == (10, 8)


def test_simulate_lorenz96_time_series_is_reproducible() -> None:
    from exp.TM.lorzen_tm_ei import simulate_lorenz96_time_series

    first = simulate_lorenz96_time_series(node_count=8, steps=32, dt=0.01, forcing=8.0, seed=9)
    second = simulate_lorenz96_time_series(node_count=8, steps=32, dt=0.01, forcing=8.0, seed=9)

    assert first.shape == (33, 8)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second)


def test_run_lorzen_mlp_tm_ei_experiment_exports_artifacts(tmp_path: Path) -> None:
    from exp.TM.lorzen_tm_ei import run_lorzen_mlp_tm_ei_experiment

    result = run_lorzen_mlp_tm_ei_experiment(
        output_dir=tmp_path / "mlp",
        node_count=8,
        steps=220,
        burn_in=20,
        dt=0.01,
        lag=3,
        train_sample_count=160,
        intervention_sample_count=128,
        box_width=1.5,
        seed=14,
        hidden_layer_sizes=(32,),
        max_iter=250,
    )

    assert result["metrics"]["estimator_mode"] == "mlp_intervention"
    assert result["metrics"]["lag"] == 3
    assert result["metrics"]["intervention_sample_count"] == 128
    assert np.isfinite(result["metrics"]["train_r2"])
    assert np.isfinite(result["metrics"]["test_r2"])
    assert result["ei_matrix"].shape == (8, 8)
    assert result["edge_table"].shape[0] == 64
    assert (tmp_path / "mlp" / "lorzen_mlp_training_history.csv").exists()
    assert (tmp_path / "mlp" / "lorzen_tm_ei_heatmap.png").exists()
    assert (tmp_path / "mlp" / "lorzen_ei_causal_graph.png").exists()


def test_reference_style_strength_uses_raw_nonnegative_ei_topk_not_absolute_threshold(tmp_path: Path) -> None:
    from exp.TM.lorzen_tm_ei import (
        build_reference_style_causal_strength,
        lorenz96_groundtruth_adjacency,
        plot_reference_style_ei_causal_graph,
    )

    labels = [f"M{i}" for i in range(1, 9)]
    groundtruth = lorenz96_groundtruth_adjacency(8)
    matrix = pd.DataFrame(1e-5, index=labels, columns=labels)
    for target in labels:
        true_sources = list(groundtruth.columns[groundtruth.loc[target].astype(bool)])
        for rank, source in enumerate(true_sources):
            matrix.loc[target, source] = 1.0 / (rank + 1)
    matrix.loc["M1", "M8"] = 0.005

    strength, graph = build_reference_style_causal_strength(matrix, top_k=4)
    metrics = plot_reference_style_ei_causal_graph(
        matrix,
        groundtruth,
        output_path=tmp_path / "causal_graph.png",
        top_k=4,
    )

    assert graph.equals(groundtruth)
    assert float(strength.min().min()) >= 0.0
    assert strength.loc["M1", "M8"] == matrix.loc["M1", "M8"]
    assert strength.loc["M1", "M5"] == 0.0
    assert metrics["f1"] == 1.0
    assert (tmp_path / "causal_graph.png").exists()
