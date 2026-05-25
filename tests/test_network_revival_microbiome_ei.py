from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from exp.network_revival.microbiome import MicrobiomeParameters
from exp.network_revival.microbiome_ei import (
    MicrobiomeEIConfig,
    compare_ei_to_recovery,
    estimate_single_node_mean_response_ei,
    plot_microbiome_ei_comparison,
    run_microbiome_single_node_ei,
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


def test_microbiome_ei_notebook_contains_cache_first_workflow():
    notebook_path = Path(__file__).resolve().parents[1] / "exp" / "network_revival" / "notebook_microbiome_ei.ipynb"
    notebook_text = notebook_path.read_text(encoding="utf-8")

    assert "Microbiome Single-Ignition EI" in notebook_text
    assert "MicrobiomeEIConfig" in notebook_text
    assert "run_microbiome_single_node_ei" in notebook_text
    assert "network_revival_microbiome_ei" in notebook_text
    assert "force_recompute=False" in notebook_text
