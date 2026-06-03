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
    assert np.isfinite(result["ei_matrix"].to_numpy(dtype=float)).all()
    assert (output_dir / "lorzen_tm_ei_matrix.csv").exists()
    assert (output_dir / "lorzen_tm_ei_edges.csv").exists()
    assert (output_dir / "lorzen_tm_ei_summary.json").exists()
    assert (output_dir / "lorzen_tm_ei_heatmap.png").exists()


def test_delta_target_mode_removes_raw_next_state_diagonal_dominance(tmp_path: Path) -> None:
    from exp.TM.lorzen_tm_ei import run_lorzen_tm_ei_experiment

    rng = np.random.default_rng(12)
    x = rng.normal(size=(100, 8))
    delta = np.zeros_like(x)
    for target_index in range(8):
        delta[:, target_index] = (
            0.9 * x[:, (target_index - 1) % 8]
            - 0.7 * x[:, (target_index - 2) % 8]
            + 0.6 * x[:, (target_index + 1) % 8]
            + 0.02 * rng.normal(size=x.shape[0])
        )
    y_next = x + delta
    pd.DataFrame(x).to_csv(tmp_path / "yt.csv", header=False, index=False)
    pd.DataFrame(y_next).to_csv(tmp_path / "yt+1.csv", header=False, index=False)

    next_result = run_lorzen_tm_ei_experiment(tmp_path, output_dir=tmp_path / "next", target_mode="next")
    delta_result = run_lorzen_tm_ei_experiment(tmp_path, output_dir=tmp_path / "delta", target_mode="delta")

    next_matrix = next_result["ei_matrix"].to_numpy(dtype=float)
    delta_matrix = delta_result["ei_matrix"].to_numpy(dtype=float)
    off_diagonal_mask = ~np.eye(8, dtype=bool)

    assert np.diag(next_matrix).mean() > next_matrix[off_diagonal_mask].mean()
    assert np.diag(delta_matrix).mean() < delta_matrix[off_diagonal_mask].mean()
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
