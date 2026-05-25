import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from exp.network_revival.microbiome import (
    DEFAULT_ARTICLE_ZIP,
    MicrobiomeParameters,
    build_signed_microbiome_adjacency,
    filter_active_nodes,
    load_microbiome_networks,
    rank_nodes_by_positive_reach,
    solve_ecological_steady_state,
)


ROOT = Path(__file__).resolve().parents[1]


def test_load_microbiome_networks_from_article_zip_has_expected_numeric_shapes():
    data = load_microbiome_networks(DEFAULT_ARTICLE_ZIP)

    assert data.competition.shape == (838, 838)
    assert data.complementarity.shape == (838, 838)
    assert data.norm_import.shape == (838, 283)
    assert np.isfinite(data.competition).all()
    assert np.isfinite(data.complementarity).all()


def test_signed_microbiome_adjacency_uses_paper_weights_and_has_both_edge_signs():
    data = load_microbiome_networks(DEFAULT_ARTICLE_ZIP)
    params = MicrobiomeParameters(positive_weight=30.0, competition_weight=1.0)

    adjacency = build_signed_microbiome_adjacency(data, params)

    assert adjacency.shape == (838, 838)
    np.testing.assert_allclose(adjacency, 30.0 * data.complementarity - data.competition)
    assert np.any(adjacency > 0.0)
    assert np.any(adjacency < 0.0)


def test_filter_active_nodes_returns_active_subgraph_and_original_indices():
    adjacency = np.arange(25, dtype=float).reshape(5, 5)
    upper_state = np.array([0.5, 2.1, 10.0, 1.9, 3.0])

    filtered = filter_active_nodes(adjacency, upper_state, threshold=2.0)

    np.testing.assert_array_equal(filtered.active_indices, np.array([1, 2, 4]))
    np.testing.assert_allclose(filtered.active_adjacency, adjacency[np.ix_([1, 2, 4], [1, 2, 4])])
    np.testing.assert_allclose(filtered.active_upper_state, np.array([2.1, 10.0, 3.0]))


def test_default_upper_state_reproduces_reference_active_species_count():
    params = MicrobiomeParameters(dt=0.05, t_max=20.0)
    data = load_microbiome_networks(DEFAULT_ARTICLE_ZIP)
    adjacency = build_signed_microbiome_adjacency(data, params)

    upper_state = solve_ecological_steady_state(adjacency, params, free_value=params.upper_free_value)
    active = filter_active_nodes(adjacency, upper_state, threshold=params.active_threshold)
    rows = rank_nodes_by_positive_reach(
        active.active_adjacency,
        active_indices=active.active_indices,
        states=np.zeros(len(active.active_indices)),
        threshold=params.reach_threshold,
    )

    assert len(active.active_indices) == 568
    assert [row["tree_size"] for row in rows[:5]] == [360, 360, 360, 360, 360]


def test_rank_nodes_by_positive_reach_matches_transposed_directed_spread_rule():
    adjacency = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
        ]
    )
    states = np.array([0.2, 6.0, 8.0, 1.0])

    rows = rank_nodes_by_positive_reach(
        adjacency,
        active_indices=np.array([10, 11, 12, 13]),
        states=states,
        threshold=1.122,
        species_names={10: "a", 11: "b", 12: "c", 13: "d"},
    )

    assert [row["active_node"] for row in rows] == [0, 1, 2, 3]
    assert [row["tree_size"] for row in rows] == [3, 2, 1, 1]
    assert rows[0]["species_index"] == 10
    assert rows[0]["species_name"] == "a"
    assert rows[1]["success"] is True
    assert rows[0]["success"] is False


def test_cli_smoke_run_writes_cache_figures_and_paper_parameters():
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "reproduce_network_revival_microbiome.py"),
            "--article-zip",
            str(DEFAULT_ARTICLE_ZIP),
            "--output-dir",
            str(output_dir),
            "--max-nodes",
            "3",
            "--dt",
            "0.2",
            "--t-max",
            "1.0",
            "--force",
        ]

        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

        results_dir = output_dir / "results" / "network_revival_microbiome"
        fig_dir = output_dir / "fig" / "network_revival_microbiome"
        assert (results_dir / "active_network.npz").exists()
        assert (results_dir / "node_ignition_states.csv").exists()
        assert (results_dir / "node_ignition_ranked.csv").exists()
        assert (results_dir / "metadata.json").exists()
        assert (fig_dir / "ranked_ignition_success.png").exists()
        assert (fig_dir / "ranked_ignition_success.png").stat().st_size > 1000

        metadata = json.loads((results_dir / "metadata.json").read_text())
        assert metadata["parameters"]["positive_weight"] == 30.0
        assert metadata["parameters"]["competition_weight"] == 1.0
        assert metadata["parameters"]["delta"] == 10.0
        assert metadata["parameters"]["active_threshold"] == 2.0
        assert metadata["parameters"]["success_threshold"] == 5.0
        assert metadata["parameters"]["reach_threshold"] == 1.122
        assert metadata["evaluated_node_count"] == 3
