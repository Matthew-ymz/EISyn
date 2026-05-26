from pathlib import Path
from tempfile import TemporaryDirectory
import json

import numpy as np
import pandas as pd

from exp.network_revival.microbiome import MicrobiomeParameters
from exp.network_revival.microbiome_ei import (
    MicrobiomeEIConfig,
    MicrobiomePhiEIDSweepConfig,
    MicrobiomeStateSpaceEIConfig,
    compare_ei_to_recovery,
    estimate_microbiome_whole_system_phi_eid,
    estimate_single_node_mean_response_ei,
    plot_microbiome_phi_eid_summary,
    plot_microbiome_ei_comparison,
    plot_microbiome_state_space_ei_comparison,
    run_microbiome_phi_eid_sweep,
    run_microbiome_state_space_node_ei,
    run_microbiome_single_node_ei,
    simulate_microbiome_state_space_final_states,
    simulate_single_node_mean_response_samples,
)


def test_single_node_mean_response_samples_have_node_delta_shape():
    adjacency = np.array([[0.0, 0.6], [0.4, 0.0]], dtype=float)
    params = MicrobiomeParameters(dt=0.2, t_max=0.4)
    deltas = np.array([0.0, 1.0, 2.0], dtype=float)

    samples = simulate_single_node_mean_response_samples(
        adjacency,
        params,
        node_indices=(0, 1),
        deltas=deltas,
    )

    assert samples.shape == (2, 3)
    assert np.isfinite(samples).all()


def test_microbiome_state_space_simulation_returns_full_final_states():
    adjacency = np.array([[0.0, 0.6], [0.4, 0.0]], dtype=float)
    params = MicrobiomeParameters(dt=0.1, t_max=0.2)
    initial_states = np.array(
        [
            [0.1, 0.2],
            [1.0, 0.5],
            [2.0, 1.5],
        ],
        dtype=float,
    )

    final_states = simulate_microbiome_state_space_final_states(
        adjacency,
        params,
        initial_states=initial_states,
        tau=0.2,
        dt=0.1,
        batch_size=2,
    )

    assert final_states.shape == initial_states.shape
    assert np.isfinite(final_states).all()
    assert np.all(final_states >= 0.0)


def test_whole_system_phi_eid_estimator_returns_finite_positive_coupled_score():
    rng = np.random.default_rng(12)
    source = rng.normal(size=(80, 2))
    target = np.column_stack(
        [
            source[:, 0] + source[:, 1],
            source[:, 0] - source[:, 1],
        ]
    )

    result = estimate_microbiome_whole_system_phi_eid(
        source,
        target,
        target_noise_fraction=0.01,
        seed=17,
    )

    assert np.isfinite(result["whole_ei"])
    assert np.isfinite(result["singleton_ei_sum"])
    assert np.isfinite(result["phi_eid"])
    assert result["whole_ei"] > 0.0
    assert result["phi_eid"] > 0.0


def test_run_microbiome_phi_eid_sweep_writes_reloads_and_ranks_grid():
    adjacency = np.array([[0.0, 0.6], [0.4, 0.0]], dtype=float)
    active_indices = np.array([10, 11], dtype=int)

    with TemporaryDirectory() as tmpdir:
        config = MicrobiomePhiEIDSweepConfig(
            sample_counts=(24, 32),
            tau_values=(0.0, 0.2),
            state_low=0.0,
            state_high=2.0,
            dt=0.1,
            seed=7,
            batch_size=8,
            output_dir=Path(tmpdir),
            params=MicrobiomeParameters(dt=0.1, t_max=0.2),
            show_progress=False,
        )

        result = run_microbiome_phi_eid_sweep(
            config,
            adjacency=adjacency,
            active_indices=active_indices,
            force_recompute=True,
        )

        grid = result["grid"]
        top = result["top_conditions"]
        assert len(grid) == 4
        assert len(top) == 4
        assert top["phi_eid"].is_monotonic_decreasing
        assert result["cache_paths"]["grid_csv"].exists()
        assert result["cache_paths"]["top_conditions_csv"].exists()
        assert result["cache_paths"]["manifest_json"].exists()
        assert result["manifest"]["experiment"] == "network_revival_microbiome_phi_eid_sweep"

        loaded = run_microbiome_phi_eid_sweep(
            config,
            adjacency=adjacency,
            active_indices=active_indices,
            force_recompute=False,
        )
        pd.testing.assert_frame_equal(loaded["grid"], grid)
        pd.testing.assert_frame_equal(loaded["top_conditions"], top)


def test_plot_microbiome_phi_eid_summary_writes_heatmaps():
    grid = pd.DataFrame(
        {
            "sample_count": [5000, 5000, 10000, 10000],
            "tau": [1.0, 5.0, 1.0, 5.0],
            "whole_ei": [1.0, 2.0, 3.0, 4.0],
            "singleton_ei_sum": [0.5, 1.1, 1.0, 1.5],
            "phi_eid": [0.5, 0.9, 2.0, 2.5],
            "phi_ratio": [0.5, 0.45, 0.667, 0.625],
        }
    )

    with TemporaryDirectory() as tmpdir:
        paths = plot_microbiome_phi_eid_summary(grid, Path(tmpdir))

        assert set(paths) == {"phi_heatmap", "phi_ratio_heatmap"}
        assert all(path.exists() and path.stat().st_size > 1000 for path in paths.values())


def test_run_microbiome_state_space_node_ei_writes_and_reloads_cache():
    adjacency = np.array([[0.0, 0.6], [0.4, 0.0]], dtype=float)
    active_indices = np.array([10, 11], dtype=int)
    ranked = pd.DataFrame(
        {
            "rank": [1, 2],
            "active_node": [0, 1],
            "species_index": [10, 11],
            "species_name": ["a", "b"],
            "tree_size": [2, 1],
            "state": [7.0, 0.2],
            "success": [True, False],
        }
    )

    with TemporaryDirectory() as tmpdir:
        config = MicrobiomeStateSpaceEIConfig(
            sample_count=24,
            state_low=0.0,
            state_high=2.0,
            tau=0.2,
            dt=0.1,
            seed=7,
            batch_size=8,
            output_dir=Path(tmpdir),
            params=MicrobiomeParameters(dt=0.1, t_max=0.2),
        )

        result = run_microbiome_state_space_node_ei(
            config,
            adjacency=adjacency,
            active_indices=active_indices,
            recovery_ranked=ranked,
            force_recompute=True,
        )

        assert result["initial_states"].shape == (24, 2)
        assert result["final_states"].shape == (24, 2)
        assert result["final_mean_activity"].shape == (24,)
        assert result["node_ei"].shape == (2,)
        assert "ei_final_state" in result["summary"].columns
        assert result["manifest"]["source_variable"] == "initial_node_state_x_i_0"
        assert result["manifest"]["target_variable"] == "whole_system_state_at_tau"
        assert result["cache_paths"]["samples_npz"].exists()
        assert result["cache_paths"]["summary_csv"].exists()
        assert result["cache_paths"]["comparison_csv"].exists()
        assert result["cache_paths"]["manifest_json"].exists()

        loaded = run_microbiome_state_space_node_ei(
            config,
            adjacency=adjacency,
            active_indices=active_indices,
            recovery_ranked=ranked,
            force_recompute=False,
        )
        np.testing.assert_allclose(loaded["initial_states"], result["initial_states"])
        np.testing.assert_allclose(loaded["final_states"], result["final_states"])
        np.testing.assert_allclose(loaded["node_ei"], result["node_ei"])


def test_ei_estimator_ranks_dependent_response_above_shuffled_response():
    rng = np.random.default_rng(3)
    deltas = np.linspace(0.0, 10.0, 80)
    dependent = deltas + 0.1 * rng.normal(size=deltas.shape)
    shuffled = dependent.copy()
    rng.shuffle(shuffled)
    samples = np.vstack([dependent, shuffled])

    result = estimate_single_node_mean_response_ei(
        deltas,
        samples,
        target_noise_fraction=0.01,
        seed=9,
    )

    assert result["node_ei"][0] > result["node_ei"][1]


def test_run_microbiome_single_node_ei_writes_and_reloads_cache():
    adjacency = np.array([[0.0, 0.6], [0.4, 0.0]], dtype=float)
    active_indices = np.array([10, 11], dtype=int)
    ranked = pd.DataFrame(
        {
            "rank": [1, 2],
            "active_node": [0, 1],
            "species_index": [10, 11],
            "species_name": ["a", "b"],
            "tree_size": [2, 1],
            "state": [7.0, 0.2],
            "success": [True, False],
        }
    )

    with TemporaryDirectory() as tmpdir:
        config = MicrobiomeEIConfig(
            n_delta=5,
            delta_max=2.0,
            seed=5,
            output_dir=Path(tmpdir),
            node_indices=(0, 1),
            params=MicrobiomeParameters(dt=0.2, t_max=0.4),
        )
        result = run_microbiome_single_node_ei(
            config,
            adjacency=adjacency,
            active_indices=active_indices,
            recovery_ranked=ranked,
            force_recompute=True,
        )

        assert result["mean_response_samples"].shape == (2, 5)
        assert result["node_ei"].shape == (2,)
        assert result["cache_paths"]["samples_npz"].exists()
        assert result["cache_paths"]["summary_csv"].exists()
        assert result["cache_paths"]["comparison_csv"].exists()
        assert result["cache_paths"]["manifest_json"].exists()

        loaded = run_microbiome_single_node_ei(
            config,
            adjacency=adjacency,
            active_indices=active_indices,
            recovery_ranked=ranked,
            force_recompute=False,
        )
        np.testing.assert_allclose(loaded["node_ei"], result["node_ei"])
        np.testing.assert_allclose(loaded["mean_response_samples"], result["mean_response_samples"])


def test_run_microbiome_single_node_ei_recomputes_when_cache_manifest_differs(monkeypatch):
    adjacency = np.array([[0.0, 0.6], [0.4, 0.0]], dtype=float)
    active_indices = np.array([10, 11], dtype=int)
    ranked = pd.DataFrame(
        {
            "rank": [1, 2],
            "active_node": [0, 1],
            "species_index": [10, 11],
            "species_name": ["a", "b"],
            "tree_size": [2, 1],
            "state": [7.0, 0.2],
            "success": [True, False],
        }
    )

    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        np.savez_compressed(
            output_dir / "microbiome_single_node_ei_samples.npz",
            deltas=np.linspace(0.0, 1.0, 3),
            node_indices=np.array([0, 1], dtype=int),
            active_indices=active_indices,
            mean_response_samples=np.zeros((2, 3)),
            node_ei=np.zeros(2),
            target_noise_sigma=np.zeros(2),
            bias_correction=np.zeros(2),
        )
        pd.DataFrame({"active_node": [0, 1], "species_index": [10, 11], "ei_mean_response": [0.0, 0.0]}).to_csv(
            output_dir / "microbiome_single_node_ei_summary.csv",
            index=False,
        )
        ranked.assign(ei_rank=[1, 2], ei_mean_response=[0.0, 0.0], ei_residual_vs_tree_size=[0.0, 0.0]).to_csv(
            output_dir / "microbiome_ei_vs_recovery_comparison.csv",
            index=False,
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "experiment": "network_revival_microbiome_single_node_ei",
                    "source_variable": "point_ignition_strength_delta_i",
                    "target_variable": "whole_system_final_mean_activation",
                    "release_after_forcing": False,
                    "delta_max": 99.0,
                    "n_delta": 3,
                    "seed": 5,
                    "target_noise_fraction": 0.01,
                    "microbiome_parameters": {"dt": 0.2, "t_max": 0.4},
                    "node_count": 2,
                    "evaluated_node_count": 2,
                    "comparison_metrics": {},
                }
            ),
            encoding="utf-8",
        )

        calls = {"count": 0}

        def fake_samples(adjacency, params, *, node_indices, deltas):
            calls["count"] += 1
            return np.tile(np.asarray(deltas, dtype=float), (len(node_indices), 1))

        monkeypatch.setattr("exp.network_revival.microbiome_ei.simulate_single_node_mean_response_samples", fake_samples)
        config = MicrobiomeEIConfig(
            n_delta=5,
            delta_max=2.0,
            seed=5,
            output_dir=output_dir,
            node_indices=(0, 1),
            params=MicrobiomeParameters(dt=0.2, t_max=0.4),
        )

        result = run_microbiome_single_node_ei(
            config,
            adjacency=adjacency,
            active_indices=active_indices,
            recovery_ranked=ranked,
            force_recompute=False,
        )

        assert calls["count"] == 1
        assert result["manifest"]["delta_max"] == 2.0
        assert result["mean_response_samples"].shape == (2, 5)


def test_compare_ei_to_recovery_reports_success_prediction_metrics():
    summary = pd.DataFrame(
        {
            "active_node": [0, 1, 2, 3],
            "ei_mean_response": [0.9, 0.8, 0.1, 0.0],
        }
    )
    ranked = pd.DataFrame(
        {
            "rank": [1, 2, 3, 4],
            "active_node": [0, 1, 2, 3],
            "tree_size": [1, 1, 4, 5],
            "state": [8.0, 7.0, 0.2, 0.3],
            "success": [True, True, False, False],
        }
    )

    comparison, metrics = compare_ei_to_recovery(summary, ranked, k_values=(1, 2))

    assert "ei_rank" in comparison.columns
    assert "tree_size_rank" in comparison.columns
    assert metrics["ei_success_auroc"] == 1.0
    assert metrics["ei_precision_at_2"] == 1.0
    assert metrics["tree_size_precision_at_2"] == 0.0


def test_compare_ei_to_recovery_handles_nan_ei_in_logistic_metrics():
    summary = pd.DataFrame(
        {
            "active_node": [0, 1, 2, 3, 4],
            "ei_mean_response": [0.9, np.nan, 0.8, 0.1, 0.0],
        }
    )
    ranked = pd.DataFrame(
        {
            "rank": [1, 2, 3, 4, 5],
            "active_node": [0, 1, 2, 3, 4],
            "tree_size": [1, 3, 2, 4, 5],
            "state": [8.0, 6.0, 7.0, 0.2, 0.3],
            "success": [True, True, True, False, False],
        }
    )

    _, metrics = compare_ei_to_recovery(summary, ranked, k_values=(2,))

    assert np.isfinite(metrics["logistic_tree_size_auc"])
    assert np.isfinite(metrics["logistic_ei_auc"])
    assert np.isfinite(metrics["logistic_tree_size_plus_ei_auc"])


def test_plot_microbiome_ei_comparison_writes_expected_figures():
    comparison = pd.DataFrame(
        {
            "active_node": [0, 1, 2, 3],
            "ei_mean_response": [0.9, 0.8, 0.1, 0.0],
            "ei_rank": [1, 2, 3, 4],
            "rank": [1, 2, 3, 4],
            "tree_size_rank": [3, 4, 2, 1],
            "tree_size": [1, 1, 4, 5],
            "state": [8.0, 7.0, 0.2, 0.3],
            "success": [True, True, False, False],
            "ei_residual_vs_tree_size": [0.4, 0.3, -0.2, -0.3],
        }
    )
    metrics = {
        "ei_success_auroc": 1.0,
        "tree_size_success_auroc": 0.0,
        "ei_precision_at_2": 1.0,
        "tree_size_precision_at_2": 0.0,
    }

    with TemporaryDirectory() as tmpdir:
        paths = plot_microbiome_ei_comparison(comparison, metrics, Path(tmpdir))

        assert set(paths) == {
            "ei_ranking_curve",
            "ei_vs_recovery",
            "success_enrichment",
            "success_prediction",
            "ei_specific_residual",
        }
        assert all(path.exists() and path.stat().st_size > 1000 for path in paths.values())


def test_plot_microbiome_state_space_ei_comparison_writes_expected_figures():
    comparison = pd.DataFrame(
        {
            "active_node": [0, 1, 2, 3],
            "ei_mean_response": [0.9, 0.8, 0.1, 0.0],
            "ei_rank": [1, 2, 3, 4],
            "rank": [1, 2, 3, 4],
            "tree_size_rank": [3, 4, 2, 1],
            "tree_size": [1, 1, 4, 5],
            "state": [8.0, 7.0, 0.2, 0.3],
            "success": [True, True, False, False],
            "ei_residual_vs_tree_size": [0.4, 0.3, -0.2, -0.3],
        }
    )
    metrics = {
        "ei_success_auroc": 1.0,
        "tree_size_success_auroc": 0.0,
        "ei_precision_at_2": 1.0,
        "tree_size_precision_at_2": 0.0,
    }

    with TemporaryDirectory() as tmpdir:
        paths = plot_microbiome_state_space_ei_comparison(comparison, metrics, Path(tmpdir))

        assert set(paths) == {
            "ei_ranking_curve",
            "ei_vs_recovery",
            "success_enrichment",
            "success_prediction",
            "ei_specific_residual",
        }
        assert all("state_space" in path.name for path in paths.values())
        assert all(path.exists() and path.stat().st_size > 1000 for path in paths.values())


def test_microbiome_ei_notebook_contains_cache_first_workflow():
    notebook_path = Path(__file__).resolve().parents[1] / "exp" / "network_revival" / "notebook_microbiome_ei.ipynb"
    notebook_text = notebook_path.read_text(encoding="utf-8")

    assert "Microbiome State-space EI" in notebook_text
    assert "MicrobiomeStateSpaceEIConfig" in notebook_text
    assert "MicrobiomePhiEIDSweepConfig" in notebook_text
    assert "run_microbiome_state_space_node_ei" in notebook_text
    assert "run_microbiome_phi_eid_sweep" in notebook_text
    assert "Phi^EID parameter sweep" in notebook_text
    assert "network_revival_microbiome_state_space_ei" in notebook_text
    assert "network_revival_microbiome_phi_eid_sweep" in notebook_text
    assert "plot_microbiome_state_space_ei_comparison" in notebook_text
    assert "plot_microbiome_phi_eid_summary" in notebook_text
    assert "run_microbiome_single_node_ei" not in notebook_text
    assert "force_recompute=False" in notebook_text
