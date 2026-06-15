import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp.network_revival.dynamics import get_model
from exp.network_revival.effective_information import (
    DynamicEIConfig,
    EIIgnitionThresholdConfig,
    PairIgnitionCostConfig,
    StateSpacePairSynergyConfig,
    StateSpaceEIConfig,
    binary_search_critical_delta,
    evaluate_fixed_node_ignition,
    evaluate_fixed_pair_ignition,
    estimate_node_mean_activity_ei,
    estimate_state_space_pair_synergy,
    estimate_state_space_node_ei,
    load_fig5_brain_modular_adjacency,
    run_pair_ignition_cost_experiment,
    run_ei_ignition_threshold_experiment,
    run_node_mean_activity_ei,
    run_state_space_pair_synergy,
    run_state_space_node_ei,
    sample_uniform_state_space,
    sample_random_node_pairs,
    select_ei_stratified_nodes,
    simulate_state_space_final_mean_activity,
    simulate_state_space_final_states,
    simulate_single_ignition_trajectory,
)


def test_brain_adjacency_builder_returns_matching_community_masks():
    data = load_fig5_brain_modular_adjacency(win=20.0, wout=5.0)

    adjacency = data["adjacency"]
    comm1 = data["comm1_mask"]
    comm2 = data["comm2_mask"]

    assert adjacency.ndim == 2
    assert adjacency.shape[0] == adjacency.shape[1]
    assert comm1.shape == (adjacency.shape[0],)
    assert comm2.shape == (adjacency.shape[0],)
    assert np.array_equal(comm2, ~comm1)
    assert np.isfinite(adjacency).all()
    assert np.all(adjacency >= 0.0)
    assert data["win"] == 20.0
    assert data["wout"] == 5.0


def test_batched_runner_writes_cache_with_expected_shapes_and_manifest():
    with TemporaryDirectory() as tmpdir:
        config = DynamicEIConfig(
            delta_max=2.0,
            n_delta=6,
            tau_grid=(0.0, 0.2),
            t_ignite=0.2,
            dt=0.1,
            seed=7,
            output_dir=Path(tmpdir),
            node_indices=(0, 1, 2),
        )

        result = run_node_mean_activity_ei(config, force_recompute=True)

        assert result["mean_activity_samples"].shape == (3, 6, 2)
        assert result["node_ei_by_tau"].shape == (3, 2)
        assert np.isfinite(result["mean_activity_samples"]).all()
        assert np.isfinite(result["node_ei_by_tau"]).all()

        cache_paths = result["cache_paths"]
        assert cache_paths["samples_npz"].exists()
        assert cache_paths["long_csv"].exists()
        assert cache_paths["summary_csv"].exists()
        assert cache_paths["manifest_json"].exists()

        manifest = json.loads(cache_paths["manifest_json"].read_text())
        assert manifest["win"] == 20.0
        assert manifest["wout"] == 5.0
        assert manifest["delta_max"] == 2.0
        assert manifest["tau_grid"] == [0.0, 0.2]
        assert manifest["noise_policy"] == "none"
        assert manifest["transport_backend"] == "affine_triangular_transport_map"


def test_forced_node_is_held_during_ignition_and_released_afterward():
    model = get_model("Neural", mu=2.0, delta=1.0)
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

    trajectory = simulate_single_ignition_trajectory(
        adjacency,
        model,
        source_node=0,
        delta=3.0,
        tau_grid=(0.0, 0.2),
        t_ignite=0.2,
        dt=0.1,
    )

    assert trajectory["forced_source_values"].shape == (3,)
    np.testing.assert_allclose(trajectory["forced_source_values"], 3.0)
    assert trajectory["states_by_tau"].shape == (2, 2)
    assert trajectory["states_by_tau"][0, 0] == 3.0
    assert trajectory["states_by_tau"][1, 0] < 3.0


def test_transport_map_ei_is_higher_for_dependent_target_than_shuffled_target():
    rng = np.random.default_rng(11)
    deltas = np.linspace(0.0, 2.0, 80)
    dependent = deltas[:, None] + 0.05 * rng.normal(size=(80, 1))
    shuffled = dependent.copy()
    rng.shuffle(shuffled[:, 0])

    dependent_summary = estimate_node_mean_activity_ei(
        deltas,
        dependent[None, :, :],
        tau_grid=(0.0,),
        seed=3,
    )
    shuffled_summary = estimate_node_mean_activity_ei(
        deltas,
        shuffled[None, :, :],
        tau_grid=(0.0,),
        seed=3,
    )

    assert dependent_summary["node_ei_by_tau"][0, 0] > shuffled_summary["node_ei_by_tau"][0, 0]


def test_uniform_state_space_sampling_is_seeded_and_bounded():
    sample_a = sample_uniform_state_space(
        node_count=4,
        sample_count=8,
        state_low=0.0,
        state_high=30.0,
        seed=13,
    )
    sample_b = sample_uniform_state_space(
        node_count=4,
        sample_count=8,
        state_low=0.0,
        state_high=30.0,
        seed=13,
    )

    assert sample_a.shape == (8, 4)
    np.testing.assert_allclose(sample_a, sample_b)
    assert np.all(sample_a >= 0.0)
    assert np.all(sample_a <= 30.0)


def test_batched_state_space_simulation_returns_finite_final_means():
    model = get_model("Neural", mu=2.0, delta=1.0)
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    initial_states = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
        ],
        dtype=float,
    )

    final_mean = simulate_state_space_final_mean_activity(
        adjacency,
        model,
        initial_states=initial_states,
        tau=0.2,
        dt=0.1,
        batch_size=2,
    )

    assert final_mean.shape == (3,)
    assert np.isfinite(final_mean).all()
    assert np.all(final_mean >= 0.0)


def test_batched_state_space_simulation_returns_full_final_states():
    model = get_model("Neural", mu=2.0, delta=1.0)
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    initial_states = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
        ],
        dtype=float,
    )

    final_states = simulate_state_space_final_states(
        adjacency,
        model,
        initial_states=initial_states,
        tau=0.2,
        dt=0.1,
        batch_size=2,
    )

    assert final_states.shape == initial_states.shape
    assert np.isfinite(final_states).all()
    assert np.all(final_states >= 0.0)


def test_state_space_runner_writes_cache_with_expected_shapes_and_manifest():
    with TemporaryDirectory() as tmpdir:
        config = StateSpaceEIConfig(
            sample_count=24,
            state_low=0.0,
            state_high=30.0,
            tau=0.2,
            dt=0.1,
            seed=17,
            batch_size=8,
            output_dir=Path(tmpdir),
        )

        result = run_state_space_node_ei(config, force_recompute=True)

        node_count = result["initial_states"].shape[1]
        assert result["initial_states"].shape == (24, node_count)
        assert result["final_states"].shape == (24, node_count)
        assert result["final_mean_activity"].shape == (24,)
        assert result["node_ei"].shape == (node_count,)
        assert np.isfinite(result["final_states"]).all()
        assert np.isfinite(result["node_ei"]).all()

        cache_paths = result["cache_paths"]
        assert cache_paths["samples_npz"].exists()
        assert cache_paths["summary_csv"].exists()
        assert cache_paths["manifest_json"].exists()

        loaded = run_state_space_node_ei(config, force_recompute=False)
        np.testing.assert_allclose(loaded["initial_states"], result["initial_states"])
        np.testing.assert_allclose(loaded["final_states"], result["final_states"])
        np.testing.assert_allclose(loaded["node_ei"], result["node_ei"])

        manifest = json.loads(cache_paths["manifest_json"].read_text())
        assert manifest["experiment"] == "network_revival_state_space_node_ei"
        assert manifest["source_variable"] == "initial_node_state_x_i_0"
        assert manifest["target_variable"] == "whole_system_state_at_tau"
        assert manifest["sampling_mode"] == "independent_uniform_state_space"
        assert manifest["sample_count"] == 24
        assert manifest["state_low"] == 0.0
        assert manifest["state_high"] == 30.0
        assert manifest["tau"] == 0.2


def test_state_space_runner_recomputes_when_cache_manifest_differs(monkeypatch):
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        np.savez_compressed(
            output_dir / "state_space_ei_samples.npz",
            initial_states=np.zeros((2, 2)),
            final_states=np.zeros((2, 2)),
            final_mean_activity=np.zeros(2),
            node_ei=np.zeros(2),
            target_noise_sigma=np.zeros(2),
            bias_correction=np.zeros(2),
        )
        (output_dir / "state_space_node_ei_summary.csv").write_text(
            "node,community,ei_final_state,rank_ei_final_state\n0,M1,0.0,1\n1,M2,0.0,2\n",
            encoding="utf-8",
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "experiment": "network_revival_state_space_node_ei",
                    "source_variable": "initial_node_state_x_i_0",
                    "target_variable": "whole_system_state_at_tau",
                    "sampling_mode": "independent_uniform_state_space",
                    "noise_policy": "none",
                    "transport_backend": "affine_triangular_transport_map",
                    "sample_count": 2,
                    "state_low": 0.0,
                    "state_high": 1.0,
                    "tau": 99.0,
                    "dt": 0.1,
                    "seed": 3,
                    "batch_size": 2,
                    "target_noise_fraction": 0.0,
                    "show_progress": False,
                    "win": 1.0,
                    "wout": 1.0,
                    "node_count": 2,
                    "brain_source": "test",
                }
            ),
            encoding="utf-8",
        )

        import exp.network_revival.effective_information as ei_module

        monkeypatch.setattr(
            ei_module,
            "load_fig5_brain_modular_adjacency",
            lambda win, wout: {
                "adjacency": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
                "adjacency_sparse": None,
                "comm1_mask": np.array([True, False]),
                "source": "test",
                "win": win,
                "wout": wout,
            },
        )

        config = StateSpaceEIConfig(
            sample_count=12,
            state_low=0.0,
            state_high=1.0,
            tau=0.2,
            dt=0.1,
            seed=3,
            batch_size=4,
            target_noise_fraction=0.0,
            show_progress=False,
            win=1.0,
            wout=1.0,
            output_dir=output_dir,
        )

        result = run_state_space_node_ei(config, force_recompute=False)

        assert result["manifest"]["tau"] == 0.2
        assert result["initial_states"].shape == (12, 2)


def test_state_space_node_ei_is_higher_for_dependent_node_than_shuffled_target():
    rng = np.random.default_rng(23)
    initial_states = rng.uniform(0.0, 1.0, size=(100, 3))
    dependent_target = np.column_stack(
        [
            initial_states[:, 0] + 0.03 * rng.normal(size=100),
            rng.normal(size=100),
        ]
    )
    shuffled_target = dependent_target.copy()
    rng.shuffle(shuffled_target[:, 0])

    dependent_summary = estimate_state_space_node_ei(
        initial_states,
        dependent_target,
        target_noise_fraction=0.0,
        seed=5,
    )
    shuffled_summary = estimate_state_space_node_ei(
        initial_states,
        shuffled_target,
        target_noise_fraction=0.0,
        seed=5,
    )

    assert dependent_summary["node_ei"][0] > shuffled_summary["node_ei"][0]


def test_zero_target_noise_fraction_adds_no_artificial_ei_noise():
    rng = np.random.default_rng(29)
    initial_states = rng.uniform(0.0, 1.0, size=(80, 3))
    final_states = np.column_stack([initial_states[:, 0], initial_states[:, 1] ** 2])
    mean_activity = final_states[:, [0]].T[:, :, None]

    node_a = estimate_state_space_node_ei(
        initial_states,
        final_states,
        target_noise_fraction=0.0,
        seed=5,
    )
    node_b = estimate_state_space_node_ei(
        initial_states,
        final_states,
        target_noise_fraction=0.0,
        seed=17,
    )
    pair = estimate_state_space_pair_synergy(
        initial_states,
        final_states,
        pairs=[(0, 1)],
        target_noise_fraction=0.0,
        seed=23,
    )
    mean_ei = estimate_node_mean_activity_ei(
        initial_states[:, 0],
        mean_activity,
        tau_grid=(1.0,),
        target_noise_fraction=0.0,
        seed=31,
    )

    np.testing.assert_allclose(node_a["target_noise_sigma"], 0.0)
    np.testing.assert_allclose(node_b["target_noise_sigma"], 0.0)
    np.testing.assert_allclose(node_a["node_ei"], node_b["node_ei"])
    np.testing.assert_allclose(pair["target_noise_sigma"], 0.0)
    np.testing.assert_allclose(mean_ei["target_noise_sigma"], 0.0)


def test_random_node_pair_sampler_is_seeded_unique_unordered_and_self_free():
    pairs_a = sample_random_node_pairs(node_count=12, pair_count=20, seed=17)
    pairs_b = sample_random_node_pairs(node_count=12, pair_count=20, seed=17)

    assert pairs_a == pairs_b
    assert len(pairs_a) == 20
    assert len(set(pairs_a)) == 20
    assert all(left < right for left, right in pairs_a)


def test_lifted_pair_synergy_is_positive_for_product_target():
    rng = np.random.default_rng(31)
    initial_states = rng.uniform(-1.0, 1.0, size=(400, 3))
    target = (initial_states[:, [0]] * initial_states[:, [1]]) + 0.03 * rng.normal(size=(400, 1))

    summary = estimate_state_space_pair_synergy(
        initial_states,
        target,
        pairs=[(0, 1)],
        target_noise_fraction=0.0,
        seed=7,
    )

    assert summary["pair_rows"][0]["synergy"] > 0.2
    assert summary["pair_rows"][0]["joint_ei"] > summary["pair_rows"][0]["left_ei"]
    assert summary["pair_rows"][0]["joint_ei"] > summary["pair_rows"][0]["right_ei"]


def test_lifted_pair_synergy_stays_low_for_additive_single_source_target():
    rng = np.random.default_rng(37)
    initial_states = rng.uniform(-1.0, 1.0, size=(400, 3))
    target = initial_states[:, [0]] + 0.03 * rng.normal(size=(400, 1))

    summary = estimate_state_space_pair_synergy(
        initial_states,
        target,
        pairs=[(0, 1)],
        target_noise_fraction=0.0,
        seed=7,
    )

    assert summary["pair_rows"][0]["synergy"] < 0.1


def test_state_space_pair_synergy_runner_writes_and_reloads_cache(monkeypatch):
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "pair_synergy"
        state_space_dir = Path(tmpdir) / "state_space"
        state_space_dir.mkdir()
        rng = np.random.default_rng(41)
        initial_states = rng.uniform(-1.0, 1.0, size=(80, 4))
        final_states = np.column_stack(
            [
                initial_states[:, 0] * initial_states[:, 1],
                initial_states[:, 2],
            ]
        )
        np.savez_compressed(
            state_space_dir / "state_space_ei_samples.npz",
            initial_states=initial_states,
            final_states=final_states,
            final_mean_activity=final_states.mean(axis=1),
            node_ei=np.zeros(4),
            target_noise_sigma=np.zeros(2),
            bias_correction=np.zeros(4),
        )

        import exp.network_revival.effective_information as ei_module

        monkeypatch.setattr(
            ei_module,
            "load_fig5_brain_modular_adjacency",
            lambda win, wout: {
                "adjacency": np.ones((4, 4), dtype=float) - np.eye(4),
                "adjacency_sparse": None,
                "comm1_mask": np.array([True, True, False, False]),
                "comm2_mask": np.array([False, False, True, True]),
                "source": "test",
                "win": win,
                "wout": wout,
            },
        )
        config = StateSpacePairSynergyConfig(
            state_space_run_id="test_state",
            pair_count=3,
            pair_seed=13,
            target_noise_fraction=0.0,
            output_dir=output_dir,
            state_space_output_dir=state_space_dir,
        )

        result = run_state_space_pair_synergy(config, force_recompute=True)

        assert len(result["pair_rows"]) == 3
        assert result["pairs"].shape == (3, 2)
        assert result["cache_paths"]["summary_csv"].exists()
        assert result["cache_paths"]["samples_npz"].exists()
        assert result["cache_paths"]["manifest_json"].exists()

        loaded = run_state_space_pair_synergy(config, force_recompute=False)
        assert loaded["pair_rows"] == result["pair_rows"]
        np.testing.assert_array_equal(loaded["pairs"], result["pairs"])


def test_binary_search_critical_delta_recovers_monotone_threshold():
    result = binary_search_critical_delta(
        lambda delta: {"success": bool(delta >= 7.5), "recovered_modules": 2},
        delta_low=0.0,
        delta_high=30.0,
        binary_steps=10,
    )

    assert result["threshold_status"] == "finite"
    assert abs(result["critical_delta"] - 7.5) <= 30.0 / (2**10)


def test_binary_search_critical_delta_marks_censored_when_high_fails():
    result = binary_search_critical_delta(
        lambda delta: {"success": False, "recovered_modules": 0},
        delta_low=0.0,
        delta_high=30.0,
        binary_steps=10,
    )

    assert result["threshold_status"] == "censored_above_delta_max"
    assert np.isnan(result["critical_delta"])
    assert result["recovered_modules_at_delta_max"] == 0


def test_select_ei_stratified_nodes_returns_unique_top_middle_bottom():
    rows = [
        {"node": node, "community": "M1", "ei_final_state": float(100 - node), "rank_ei_final_state": node + 1}
        for node in range(12)
    ]

    selected = select_ei_stratified_nodes(rows, per_stratum=2)

    assert len(selected) == 6
    assert len({row["node"] for row in selected}) == 6
    assert [row["ei_stratum"] for row in selected].count("top") == 2
    assert [row["ei_stratum"] for row in selected].count("middle") == 2
    assert [row["ei_stratum"] for row in selected].count("bottom") == 2
    assert {row["node"] for row in selected if row["ei_stratum"] == "top"} == {0, 1}
    assert {row["node"] for row in selected if row["ei_stratum"] == "bottom"} == {10, 11}


def test_fixed_node_ignition_reports_recovered_modules_on_toy_network():
    model = {
        "M0": lambda x: np.zeros_like(x),
        "M1": lambda x: np.ones_like(x),
        "M2": lambda x: x,
    }
    adjacency = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    result = evaluate_fixed_node_ignition(
        adjacency,
        model,
        source_node=0,
        delta=6.0,
        comm1_mask=np.array([True, True, False, False]),
        comm2_mask=np.array([False, False, True, True]),
        success_threshold=5.0,
        t_force=1.0,
        dt=0.1,
        tol_ss=0.0,
    )

    assert result["recovered_modules"] == 2
    assert result["success"] is True
    assert result["module1_mean"] > 5.0
    assert result["module2_mean"] > 5.0


def test_fixed_pair_ignition_splits_total_cost_across_two_nodes():
    model = {
        "M0": lambda x: np.zeros_like(x),
        "M1": lambda x: np.ones_like(x),
        "M2": lambda x: x,
    }
    adjacency = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    result = evaluate_fixed_pair_ignition(
        adjacency,
        model,
        source_pair=(0, 1),
        total_cost=12.0,
        comm1_mask=np.array([True, True, False, False]),
        comm2_mask=np.array([False, False, True, True]),
        success_threshold=5.0,
        t_force=1.0,
        dt=0.1,
        tol_ss=0.0,
    )

    assert result["delta_i"] == 6.0
    assert result["delta_j"] == 6.0
    assert result["success"] is True
    assert result["recovered_modules"] == 2


def test_pair_ignition_cost_runner_writes_cache_and_singleton_baselines(monkeypatch):
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "pair_cost"
        pair_csv = Path(tmpdir) / "pair_synergy.csv"
        pair_csv.write_text(
            "pair_i,pair_j,community_i,community_j,left_ei,right_ei,joint_ei,synergy,synergy_ratio,rank_synergy\n"
            "0,1,M1,M1,0.2,0.1,0.6,0.3,0.5,1\n"
            "2,3,M2,M2,0.1,0.1,0.25,0.05,0.2,2\n",
            encoding="utf-8",
        )

        import exp.network_revival.effective_information as ei_module

        monkeypatch.setattr(
            ei_module,
            "load_fig5_brain_modular_adjacency",
            lambda win, wout: {
                "adjacency": np.ones((4, 4), dtype=float) - np.eye(4),
                "adjacency_sparse": None,
                "comm1_mask": np.array([True, True, False, False]),
                "comm2_mask": np.array([False, False, True, True]),
                "source": "test",
                "win": win,
                "wout": wout,
            },
        )
        monkeypatch.setattr(
            ei_module,
            "evaluate_fixed_pair_ignition",
            lambda adjacency, model, source_pair, total_cost, comm1_mask, comm2_mask, **kwargs: {
                "success": bool(total_cost >= 8.0 + sum(source_pair)),
                "recovered_modules": 2 if total_cost >= 8.0 + sum(source_pair) else 0,
                "module1_mean": float(total_cost),
                "module2_mean": float(total_cost),
                "delta_i": float(total_cost) / 2.0,
                "delta_j": float(total_cost) / 2.0,
            },
        )
        monkeypatch.setattr(
            ei_module,
            "evaluate_fixed_node_ignition",
            lambda adjacency, model, source_node, delta, comm1_mask, comm2_mask, **kwargs: {
                "success": bool(delta >= 10.0 + int(source_node)),
                "recovered_modules": 2 if delta >= 10.0 + int(source_node) else 0,
                "module1_mean": float(delta),
                "module2_mean": float(delta),
            },
        )
        config = PairIgnitionCostConfig(
            pair_synergy_run_id="test_pairs",
            cost_high=30.0,
            single_delta_high=30.0,
            binary_steps=4,
            output_dir=output_dir,
            show_progress=False,
        )

        result = run_pair_ignition_cost_experiment(
            config,
            pair_synergy_csv=pair_csv,
            force_recompute=True,
        )

        assert len(result["cost_rows"]) == 2
        assert result["cost_rows"][0]["single_min_cost"] > result["cost_rows"][0]["critical_total_cost"]
        assert result["cost_rows"][0]["cost_saving"] > 0.0
        assert result["cache_paths"]["cost_csv"].exists()
        assert result["cache_paths"]["samples_npz"].exists()
        assert result["cache_paths"]["manifest_json"].exists()

        loaded = run_pair_ignition_cost_experiment(
            config,
            pair_synergy_csv=pair_csv,
            force_recompute=False,
        )
        assert loaded["cost_rows"] == result["cost_rows"]


def test_ei_ignition_threshold_runner_writes_and_reloads_cache(monkeypatch):
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "thresholds"
        ei_summary_csv = Path(tmpdir) / "ei_summary.csv"
        ei_summary_csv.write_text(
            "node,community,ei_final_state,rank_ei_final_state\n"
            "0,M1,3.0,1\n1,M1,2.0,2\n2,M2,1.0,3\n3,M2,0.5,4\n",
            encoding="utf-8",
        )

        import exp.network_revival.effective_information as ei_module

        monkeypatch.setattr(
            ei_module,
            "load_fig5_brain_modular_adjacency",
            lambda win, wout: {
                "adjacency": np.ones((4, 4), dtype=float) - np.eye(4),
                "adjacency_sparse": None,
                "comm1_mask": np.array([True, True, False, False]),
                "comm2_mask": np.array([False, False, True, True]),
                "source": "test",
                "win": win,
                "wout": wout,
            },
        )
        monkeypatch.setattr(
            ei_module,
            "evaluate_fixed_node_ignition",
            lambda adjacency, model, source_node, delta, comm1_mask, comm2_mask, **kwargs: {
                "success": bool(delta >= 4.0 + int(source_node)),
                "recovered_modules": 2 if delta >= 4.0 + int(source_node) else 0,
                "module1_mean": float(delta),
                "module2_mean": float(delta),
            },
        )

        config = EIIgnitionThresholdConfig(
            output_dir=output_dir,
            state_space_run_id="test_run",
            per_stratum=1,
            delta_high=10.0,
            binary_steps=4,
            t_force=0.2,
            dt=0.1,
            tol_ss=0.0,
            show_progress=False,
        )
        result = run_ei_ignition_threshold_experiment(
            config,
            ei_summary_csv=ei_summary_csv,
            force_recompute=True,
        )

        assert len(result["threshold_rows"]) == 3
        assert result["cache_paths"]["threshold_csv"].exists()
        assert result["cache_paths"]["samples_npz"].exists()
        assert result["cache_paths"]["manifest_json"].exists()
        assert result["threshold_rows"][0]["threshold_status"] == "finite"

        loaded = run_ei_ignition_threshold_experiment(
            config,
            ei_summary_csv=ei_summary_csv,
            force_recompute=False,
        )
        assert loaded["threshold_rows"] == result["threshold_rows"]


def test_notebook_contains_state_space_ei_section_and_external_legend():
    notebook_path = Path(__file__).resolve().parents[1] / "exp" / "network_revival" / "notebook_neural.ipynb"
    notebook_text = notebook_path.read_text(encoding="utf-8")
    report_module = (
        Path(__file__).resolve().parents[1] / "exp" / "network_revival" / "notebook_reports.py"
    ).read_text(encoding="utf-8")

    assert "State-space EI" in notebook_text
    assert "network_revival_state_space_ei" in report_module
    assert "samples_npz" in notebook_text
    assert "make_state_space_config" in notebook_text
    assert "EI-ranked ignition thresholds" in notebook_text
    assert "run_ei_ignition_threshold_experiment" in notebook_text
    assert "Pair state-space synergy" in notebook_text
    assert "Pair co-ignition cost validation" in notebook_text
    assert "run_state_space_pair_synergy" in notebook_text
    assert "run_pair_ignition_cost_experiment" in notebook_text
    assert "make_pair_synergy_config" in notebook_text
    assert "make_pair_ignition_cost_config" in notebook_text
    assert "plot_pair_synergy_report" in notebook_text
    assert "plot_pair_ignition_cost_report" in notebook_text
    assert "run_node_mean_activity_ei" not in notebook_text
    assert "Fig. 5 reproduction is intentionally not rerun" in notebook_text
    assert "bbox_to_anchor=(1.02, 0.5)" in report_module
    assert "state_space_pair_synergy" in report_module
    assert "pair_ignition_cost" in report_module


def test_three_node_synergy_notebook_contains_sensitivity_workflow():
    notebook_path = Path(__file__).resolve().parents[1] / "exp" / "network_revival" / "notebook_three_node_synergy.ipynb"
    module_path = Path(__file__).resolve().parents[1] / "exp" / "network_revival" / "sin_synergy_ignition.py"
    notebook_text = notebook_path.read_text(encoding="utf-8")
    module_text = module_path.read_text(encoding="utf-8")

    assert "network_revival_three_node_synergy" in notebook_text
    assert "ei_sample_count" in module_text
    assert "t_force" in module_text
    assert "bbox_to_anchor=(1.02, 0.5)" in notebook_text
    assert "Eco" not in notebook_text
    assert "Neural" not in notebook_text
    assert '"MM"' not in notebook_text
    assert "make_sin_synergy_ignition_model" in notebook_text
    assert "evaluate_sin_synergy_ignition" in notebook_text
    assert "sin_synergy_ignition_response.csv" in notebook_text
    assert "estimate_sin_synergy_ei_decomposition" in notebook_text
    assert "sin_synergy_ei_decomposition.csv" in notebook_text
    assert "sin_synergy_pair_duration_surface.csv" in notebook_text
    assert "sin_synergy_ei_time_curve.csv" in notebook_text
    assert "build_sin_synergy_pair_duration_surface" in notebook_text
    assert "estimate_sin_synergy_ei_time_curve" in notebook_text
    assert "SIN_GAIN = 1.0" in notebook_text
    assert "SIN_SOURCE_DECAY = 1.0" in notebook_text
    assert "SIN_TARGET_DECAY = 1.0" in notebook_text
    assert "SIN_SOURCE_WEIGHT = 1.0" in notebook_text
    assert "SIN_SECONDARY_SOURCE_WEIGHT = 0.01" in notebook_text
    assert "SIN_ALPHA_LOW = 0.05" in notebook_text
    assert "SIN_ALPHA_HIGH = 0.9" in notebook_text
    assert "SIN_EI_COST_LOW = 0.0" in notebook_text
    assert "SIN_EI_COST_HIGH = 2.0" in notebook_text
    assert "SIN_EI_TARGET_NOISE_FRACTION = 0.1" in notebook_text
    assert "monotone_synergy_nonlinearity" in notebook_text
    assert "np.log1p(np.maximum" in module_text
    assert "log1p_product" in notebook_text
    assert "sin_synergy_alpha_sweep.csv" in notebook_text
    assert "build_sin_synergy_alpha_sweep" in notebook_text
    assert "pair_gain_ratio" in notebook_text
    assert "SynRatio < 0.1" in notebook_text
    assert "SynRatio > 0.6" in notebook_text
    assert "ei_target_noise_fraction=SIN_EI_TARGET_NOISE_FRACTION" in notebook_text
    assert "单源 EI" in notebook_text
    assert "协同分解" in notebook_text
    assert "estimate_state_space_pair_synergy" in module_text
    assert "多节点关键节点对联合点火" in notebook_text
    assert "MultiNodeSynergyIgnitionConfig" in notebook_text
    assert "estimate_multi_node_pair_synergy" in notebook_text
    assert "build_multi_node_pair_ignition_table" in notebook_text
    assert "multi_node_pair_synergy.csv" in notebook_text
    assert "multi_node_pair_ignition_response.csv" in notebook_text
    assert "multi_node_pair_summary.csv" in notebook_text
    assert "multi_node_synergy_manifest.json" in notebook_text
    assert "multi_node_synergy_vs_pair_response" in notebook_text
    assert "multi_node_network_topology" in notebook_text
    assert "multi_node_network_topology" in module_text
    assert "Control pair" not in module_text
    assert "Embedded pair" not in module_text
    assert "feedback loop" in notebook_text


def test_multi_node_synergy_experiment_scores_embedded_pairs_and_writes_cache():
    from exp.network_revival.sin_synergy_ignition import (
        MultiNodeSynergyIgnitionConfig,
        build_multi_node_pair_ignition_table,
        estimate_multi_node_pair_synergy,
        load_or_run_multi_node_synergy_experiment,
        multi_node_output_paths,
        simulate_multi_node_fixed_sources,
    )

    with TemporaryDirectory() as tmpdir:
        config = MultiNodeSynergyIgnitionConfig(
            output_dir=Path(tmpdir),
            cost_grid=np.linspace(0.0, 4.0, 9),
            ei_sample_count=450,
        )

        paths = multi_node_output_paths(config)
        result = load_or_run_multi_node_synergy_experiment(config, force=True)
        pair_synergy = result["pair_synergy"]
        ignition = result["pair_ignition_response"]
        summary = result["pair_summary"]
        model = result["model"]

        assert len(config.source_nodes) == 5
        assert config.target_node == 5
        assert model["adjacency"].shape == (6, 6)
        assert np.all(model["adjacency"].sum(axis=1) > 0.0)
        assert model["feedback_loop_edges"] == [(0, 1), (1, 2), (2, 0)]
        assert len(pair_synergy) == 10
        assert len(summary) == 10
        assert set(paths) >= {
            "pair_synergy_csv",
            "pair_ignition_csv",
            "pair_summary_csv",
            "manifest_json",
        }
        assert all(path.exists() for path in paths.values())

        simulated = simulate_multi_node_fixed_sources(
            fixed_values={0: 2.0, 1: 2.0},
            model=result["model"],
            t_force=config.t_force,
            dt=config.dt,
        )
        assert simulated["final_state"].shape == (6,)
        assert simulated["final_state"][2] > 0.0
        assert simulated["final_state"][3] > 0.0
        assert simulated["final_state"][4] > 0.0
        assert simulated["target_end"] > 0.0

        strong = summary.loc[(summary["pair_i"].eq(0)) & (summary["pair_j"].eq(1))].iloc[0]
        control = summary.loc[(summary["pair_i"].eq(0)) & (summary["pair_j"].eq(2))].iloc[0]
        assert strong["is_embedded_synergy_pair"] is True
        assert control["is_embedded_synergy_pair"] is False
        assert strong["synergy"] > control["synergy"]
        assert strong["pair_surplus_at_max_cost"] > control["pair_surplus_at_max_cost"]
        assert strong["pair_response_at_max_cost"] > control["pair_response_at_max_cost"]
        assert summary["spearman_synergy_pair_response"].notna().all()
        assert set(ignition["ignition"].unique()) >= {"none", "single_i", "single_j", "pair"}

        cached_synergy = estimate_multi_node_pair_synergy(config, force=False)
        cached_ignition = build_multi_node_pair_ignition_table(config, force=False)
        assert len(cached_synergy) == 10
        assert len(cached_ignition) == len(ignition)


def test_joint_required_latch_critical_input_matches_saddle_node():
    from exp.network_revival.joint_required_ignition import critical_saddle_input

    theta = 0.4
    critical = critical_saddle_input(theta)
    z_minus = (1.0 + theta - np.sqrt(1.0 - theta + theta**2)) / 3.0
    drift = z_minus * (1.0 - z_minus) * (z_minus - theta)
    derivative = -3.0 * z_minus**2 + 2.0 * (1.0 + theta) * z_minus - theta

    assert critical > 0.0
    assert np.isclose(drift + critical, 0.0, atol=1e-12)
    assert np.isclose(derivative, 0.0, atol=1e-12)


def test_joint_required_instance_makes_singletons_fail_and_only_valid_pairs_switch():
    from exp.network_revival.joint_required_ignition import (
        JointRequiredIgnitionConfig,
        build_threshold_latch_instance,
        simulate_isolated_pair_intervention,
    )

    config = JointRequiredIgnitionConfig(source_count=8, amplitude_levels=8, dt=0.04)
    instance = build_threshold_latch_instance(config, seed=19)
    critical = float(instance["critical_input"])
    weights = np.asarray(instance["weights"], dtype=float)

    assert np.all(weights < critical)
    positive_pairs = [tuple(pair) for pair in instance["switchable_pairs"]]
    negative_pairs = [tuple(pair) for pair in instance["nonswitchable_pairs"]]
    assert positive_pairs
    assert negative_pairs
    for seed in range(50):
        generated = build_threshold_latch_instance(config, seed=seed)
        generated_weights = np.asarray(generated["weights"], dtype=float)
        generated_ratios = np.asarray(generated["pair_input_ratios"], dtype=float)
        assert np.all(generated_weights < generated["critical_input"])
        assert np.all(np.abs(generated_ratios - 1.0) >= config.pair_margin)
        assert generated["switchable_pairs"]
        assert generated["nonswitchable_pairs"]

    for node in range(config.source_count):
        single = simulate_isolated_pair_intervention(
            instance,
            pair=(node, (node + 1) % config.source_count),
            delta_i=1e6,
            delta_j=0.0,
            t_force=50.0,
            release_time=20.0,
            dt=config.dt,
        )
        assert single["basin_label"] == 0

    valid = simulate_isolated_pair_intervention(
        instance,
        pair=positive_pairs[0],
        delta_i=1e6,
        delta_j=1e6,
        t_force=50.0,
        release_time=20.0,
        dt=config.dt,
    )
    invalid = simulate_isolated_pair_intervention(
        instance,
        pair=negative_pairs[0],
        delta_i=1e6,
        delta_j=1e6,
        t_force=50.0,
        release_time=20.0,
        dt=config.dt,
    )
    assert valid["basin_label"] == 1
    assert invalid["basin_label"] == 0


def test_joint_required_basin_peid_matches_conditional_mi_and_separates_pairs():
    from exp.network_revival.joint_required_ignition import (
        JointRequiredIgnitionConfig,
        build_threshold_latch_instance,
        compute_pair_basin_peid,
    )

    config = JointRequiredIgnitionConfig(source_count=8, amplitude_levels=12, dt=0.04)
    instance = build_threshold_latch_instance(config, seed=23)
    pairs = compute_pair_basin_peid(instance, config)

    assert len(pairs) == 28
    assert set(pairs["max_pair_basin_label"].unique()) <= {0, 1}
    assert np.array_equal(
        pairs["max_pair_basin_label"].to_numpy(dtype=int),
        pairs["analytic_switchable"].to_numpy(dtype=int),
    )
    assert np.array_equal(
        pairs["max_pair_basin_label"].to_numpy(dtype=int),
        (pairs["synergy"].to_numpy() > 1e-12).astype(int),
    )
    np.testing.assert_allclose(pairs["synergy"], pairs["conditional_mi"], atol=1e-10)
    assert np.all(pairs["synergy"] >= -1e-10)
    assert pairs.loc[pairs["analytic_switchable"], "synergy"].max() > 0.0
    assert np.allclose(pairs.loc[~pairs["analytic_switchable"], "synergy"], 0.0, atol=1e-10)
    assert bool(pairs.sort_values("synergy", ascending=False).iloc[0]["analytic_switchable"])


def test_joint_required_and_gate_control_ranks_true_pair_first():
    from exp.network_revival.joint_required_ignition import (
        JointRequiredIgnitionConfig,
        compute_and_gate_control_peid,
    )

    config = JointRequiredIgnitionConfig(source_count=6, amplitude_levels=12, dt=0.04)
    control = compute_and_gate_control_peid(config, true_pair=(1, 4))
    top = control.sort_values("synergy", ascending=False).iloc[0]

    assert (int(top["pair_i"]), int(top["pair_j"])) == (1, 4)
    assert top["synergy"] > 0.0
    assert np.allclose(
        control.loc[~control["is_true_pair"], "synergy"],
        0.0,
        atol=1e-10,
    )


def test_joint_required_ensemble_writes_cache_and_reloads_reproducibly():
    from exp.network_revival.joint_required_ignition import (
        JointRequiredIgnitionConfig,
        run_joint_required_ensemble,
    )

    with TemporaryDirectory() as tmpdir:
        config = JointRequiredIgnitionConfig(
            output_dir=Path(tmpdir),
            source_count=6,
            amplitude_levels=8,
            ensemble_size=3,
            sample_sizes=(64,),
            label_noise_levels=(0.0,),
            dt=0.04,
        )
        first = run_joint_required_ensemble(config, force=True)
        second = run_joint_required_ensemble(config, force=False)

        assert first["cache_paths"]["summary_json"].exists()
        assert first["cache_paths"]["pairs_jsonl"].exists()
        assert first["cache_paths"]["arrays_npz"].exists()
        assert first["cache_paths"]["manifest_json"].exists()
        assert len(first["representative_pairs"]) == 15
        assert not first["metrics"].empty
        assert first["metrics"].equals(second["metrics"])
        assert first["summary"]["support_correspondence"]["all_instances_exact_match"]

        manifest = json.loads(first["cache_paths"]["manifest_json"].read_text())
        assert manifest["config"]["seed"] == config.seed
        assert manifest["config"]["t_force"] == config.t_force
        assert manifest["config"]["amplitude_levels"] == config.amplitude_levels
        assert "w_i + w_j > u_SN" in manifest["analytic_truth"]
        assert "I(delta_i,delta_j; basin)" in manifest["peid_definition"]


def test_joint_required_notebook_and_part3_reference_persisted_figures():
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "exp" / "network_revival" / "notebook_joint_required_ignition.ipynb"
    report_path = root / "docs" / "reports" / "Part3.md"
    assets = (
        "part3_joint_required_mechanism.png",
        "part3_joint_required_pair_screening.png",
        "part3_joint_required_ensemble_performance.png",
    )

    notebook = json.loads(notebook_path.read_text())
    notebook_text = json.dumps(notebook, ensure_ascii=False)
    report_text = report_path.read_text()

    assert "run_joint_required_ensemble" in notebook_text
    assert "plot_joint_required_results" in notebook_text
    assert "最终 basin 标签" in notebook_text
    assert "claim_gate_passed" in notebook_text
    assert all("execution_count" in cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert not any(
        output.get("output_type") == "error"
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    )
    for asset in assets:
        assert f"assets/{asset}" in report_text
        assert (root / "docs" / "reports" / "assets" / asset).exists()
