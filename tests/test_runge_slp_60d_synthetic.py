import unittest

import numpy as np
import pandas as pd

from exp.runge_slp_60d_visualization.synthetic_runge_gru import (
    GruTrainingConfig,
    MlpTrainingConfig,
    SyntheticRungeConfig,
    compute_causal_role_scores,
    compute_series_diagnostics,
    build_synergy_necessity_ablation,
    discrete_effective_information,
    estimate_gru_peid_graphs,
    estimate_model_peid_graphs,
    estimate_gru_tm_ei_graphs,
    compute_gateway_scores,
    ground_truth_strength_matrix,
    ground_truth_pairwise_edges,
    ground_truth_synergy_edges,
    make_supervised_windows,
    predict_gru_windows,
    predict_mlp_windows,
    run_gateway_synergy_benchmark,
    simulate_runge_like_dynamics,
    train_gru_forecaster,
    train_mlp_forecaster,
)


class SyntheticRungeGruTests(unittest.TestCase):
    def test_synthetic_dynamics_returns_runge_like_component_frame(self) -> None:
        config = SyntheticRungeConfig(n_components=8, n_steps=96, burn_in=24, seed=11)

        frame, metadata = simulate_runge_like_dynamics(config)

        self.assertEqual(frame.shape, (96, 8))
        self.assertEqual(pd.infer_freq(frame.index), "W-THU")
        self.assertEqual(frame.columns[0], "component_01")
        self.assertEqual(metadata["equation"], "mixed_sparse_lagged_hopf_map")
        self.assertLess(abs(float(frame.mean().mean())), 0.15)
        self.assertGreater(float(frame.std(ddof=1).mean()), 0.4)

    def test_sparse_var12_dynamics_exports_readable_parent_edges(self) -> None:
        config = SyntheticRungeConfig(
            dynamics_kind="sparse_var12",
            n_components=12,
            n_steps=96,
            burn_in=24,
            seed=11,
        )

        frame, metadata = simulate_runge_like_dynamics(config)
        pairwise = ground_truth_pairwise_edges(metadata, source_mode="latest")

        self.assertEqual(frame.shape, (96, 12))
        self.assertEqual(metadata["equation"], "sparse_var12")
        self.assertEqual(metadata["coordinate_space"], "mechanism")
        self.assertIn("parent_edges", metadata)
        self.assertEqual(len(metadata["parent_edges"]), len(pairwise))
        self_edges = {
            (row["source_index"], row["target_index"])
            for row in metadata["parent_edges"]
            if row["edge_kind"] == "self_memory"
        }
        self.assertEqual(self_edges, {(idx, idx) for idx in range(12)})
        cross_edges = {
            (row["source_index"], row["target_index"])
            for row in metadata["parent_edges"]
            if row["edge_kind"] == "cross_causal"
        }
        self.assertGreaterEqual(len(cross_edges), 8)
        self.assertTrue(cross_edges <= set(pairwise[["source_index", "target_index"]].itertuples(index=False, name=None)))

    def test_ground_truth_heatmap_uses_coefficient_strength(self) -> None:
        pairwise_matrix = np.zeros((3, 3), dtype=float)
        ground_truth_edges = pd.DataFrame(
            [
                {"source_index": 0, "target_index": 1, "coefficient": -0.42},
                {"source_index": 2, "target_index": 0, "coefficient": 0.35},
            ]
        )

        truth = ground_truth_strength_matrix(pairwise_matrix, ground_truth_edges)

        self.assertAlmostEqual(float(truth[0, 1]), 0.42)
        self.assertAlmostEqual(float(truth[2, 0]), 0.35)
        self.assertEqual(float(truth[1, 1]), 0.0)

    def test_sparse_var12_requires_twelve_components(self) -> None:
        config = SyntheticRungeConfig(dynamics_kind="sparse_var12", n_components=10)

        with self.assertRaisesRegex(ValueError, "sparse_var12 requires n_components=12"):
            simulate_runge_like_dynamics(config)

    def test_nonlinear_gateway_synergy12_exports_gateway_truth(self) -> None:
        config = SyntheticRungeConfig(
            dynamics_kind="nonlinear_gateway_synergy12",
            n_components=12,
            n_steps=128,
            burn_in=64,
            noise_scale=0.02,
            nonlinear_strength=0.18,
            synergy_strength=0.24,
            seed=23,
            apply_observation_mixing=False,
        )

        frame, metadata = simulate_runge_like_dynamics(config)
        pairwise = ground_truth_pairwise_edges(metadata, source_mode="latest")
        synergy = ground_truth_synergy_edges(metadata)

        self.assertEqual(frame.shape, (128, 12))
        self.assertTrue(np.isfinite(frame.to_numpy(dtype=float)).all())
        self.assertEqual(metadata["equation"], "nonlinear_gateway_synergy12")
        self.assertEqual(metadata["coordinate_space"], "mechanism")
        self.assertEqual(metadata["gateway_node"], "component_01")
        self.assertEqual(metadata["gateway_expected_rank"], 1)
        self.assertEqual(metadata["module_labels"]["component_01"], "A")
        self.assertEqual(metadata["module_labels"]["component_07"], "B")
        self.assertIn("parent_edges", metadata)
        self.assertIn("synergy_edges", metadata)
        self.assertFalse(pairwise.empty)
        self.assertFalse(synergy.empty)

        cross_gateway_edges = {
            (int(row["source_index"]), int(row["target_index"]))
            for row in metadata["parent_edges"]
            if row["edge_kind"] == "gateway_cross_module"
        }
        self.assertGreaterEqual(len(cross_gateway_edges), 5)
        self.assertTrue(all(source == 0 and target >= 6 for source, target in cross_gateway_edges))
        self.assertTrue(any(0 in (int(row.source_i), int(row.source_j)) for row in synergy.itertuples()))

    def test_nonlinear_gateway_synergy12_ground_truth_ranks_component_01_first(self) -> None:
        _, metadata = simulate_runge_like_dynamics(
            SyntheticRungeConfig(
                dynamics_kind="nonlinear_gateway_synergy12",
                n_components=12,
                n_steps=96,
                burn_in=48,
                seed=29,
                apply_observation_mixing=False,
            )
        )
        scores = compute_gateway_scores(metadata)

        self.assertEqual(str(scores.iloc[0]["component"]), "component_01")
        self.assertEqual(int(scores.iloc[0]["rank"]), 1)
        self.assertGreater(float(scores.iloc[0]["path_effect"]), float(scores.iloc[1]["path_effect"]))
        self.assertGreater(float(scores.iloc[0]["cross_module_out_degree"]), 0.0)
        self.assertGreater(
            float(scores.attrs["cross_module_path_effect_full"]),
            float(scores.attrs["cross_module_path_effect_without_gateway"]) * 2.0,
        )

    def test_causal_role_scores_rank_ground_truth_gateway_with_synergy(self) -> None:
        _, metadata = simulate_runge_like_dynamics(
            SyntheticRungeConfig(
                dynamics_kind="nonlinear_gateway_synergy12",
                n_components=12,
                n_steps=96,
                burn_in=48,
                seed=31,
                apply_observation_mixing=False,
            )
        )
        pairwise = ground_truth_pairwise_edges(metadata, source_mode="latest")
        synergy = ground_truth_synergy_edges(metadata)

        scores = compute_causal_role_scores(
            pairwise_edges=pairwise,
            synergy_edges=synergy,
            n_components=12,
            module_labels=metadata["module_labels"],
        )

        self.assertEqual(str(scores.iloc[0]["component"]), "component_01")
        self.assertEqual(int(scores.iloc[0]["gateway_rank"]), 1)
        gateway_row = scores.loc[scores["component"] == "component_01"].iloc[0]
        self.assertGreater(float(gateway_row["synergy_source_score"]), 0.0)
        self.assertGreater(float(gateway_row["gateway_score"]), float(scores.iloc[1]["gateway_score"]))
        mediator_candidates = scores.sort_values("mediator_score", ascending=False)
        self.assertGreater(float(mediator_candidates.iloc[0]["mediator_score"]), 0.0)

    def test_causal_role_scores_rank_middle_node_as_mediator_in_chain(self) -> None:
        pairwise = pd.DataFrame(
            [
                {"source_index": 0, "target_index": 1, "ei": 0.9},
                {"source_index": 1, "target_index": 2, "ei": 0.8},
                {"source_index": 0, "target_index": 2, "ei": 0.1},
            ]
        )

        scores = compute_causal_role_scores(pairwise_edges=pairwise, n_components=3)

        mediator_row = scores.sort_values("mediator_score", ascending=False).iloc[0]
        self.assertEqual(str(mediator_row["component"]), "component_02")
        self.assertEqual(int(mediator_row["mediator_rank"]), 1)
        self.assertGreater(float(mediator_row["incoming_path_effect"]), 0.0)
        self.assertGreater(float(mediator_row["outgoing_path_effect"]), 0.0)

    def test_causal_role_scores_clips_learned_negative_synergy_for_roles(self) -> None:
        pairwise = pd.DataFrame(
            [
                {"source_index": 0, "target_index": 1, "ei": 0.5},
                {"source_index": 1, "target_index": 2, "ei": 0.4},
            ]
        )
        synergy = pd.DataFrame(
            [
                {"source_i": 0, "source_j": 1, "target_index": 2, "synergy": -0.25},
            ]
        )

        scores = compute_causal_role_scores(pairwise_edges=pairwise, synergy_edges=synergy, n_components=3)

        source_row = scores.loc[scores["component"] == "component_01"].iloc[0]
        target_row = scores.loc[scores["component"] == "component_03"].iloc[0]
        self.assertAlmostEqual(float(source_row["synergy_source_score"]), 0.0)
        self.assertAlmostEqual(float(target_row["synergy_target_score"]), 0.0)

    def test_synergy_necessity_ablation_reports_pairwise_and_synergy_aware_ranks(self) -> None:
        _, metadata = simulate_runge_like_dynamics(
            SyntheticRungeConfig(
                dynamics_kind="nonlinear_gateway_synergy12",
                n_components=12,
                n_steps=96,
                burn_in=48,
                seed=33,
                apply_observation_mixing=False,
            )
        )
        pairwise = ground_truth_pairwise_edges(metadata, source_mode="latest")
        synergy = ground_truth_synergy_edges(metadata)

        ablation = build_synergy_necessity_ablation(
            estimators=[("Ground-truth dynamics", pairwise, synergy)],
            n_components=12,
            module_labels=metadata["module_labels"],
        )

        row = ablation.loc["Ground-truth dynamics"]
        self.assertIn("pairwise_only_component_01_gateway_rank", ablation.columns)
        self.assertIn("synergy_aware_component_01_gateway_rank", ablation.columns)
        self.assertGreaterEqual(int(row["positive_synergy_edges_with_component_01"]), 3)
        self.assertEqual(str(row["synergy_aware_top_gateway"]), "component_01")
        self.assertTrue(bool(row["synergy_aware_top_is_component_01"]))

    def test_causal_role_scores_can_rank_gateway_from_learned_gru_peid(self) -> None:
        frame, metadata = simulate_runge_like_dynamics(
            SyntheticRungeConfig(
                dynamics_kind="nonlinear_gateway_synergy12",
                n_components=12,
                n_steps=260,
                burn_in=96,
                noise_scale=0.018,
                nonlinear_strength=0.18,
                synergy_strength=0.26,
                seed=37,
                apply_observation_mixing=False,
            )
        )
        windows = make_supervised_windows(frame, history=4, horizon=1)
        result = train_gru_forecaster(
            windows,
            GruTrainingConfig(
                hidden_dim=32,
                epochs=14,
                batch_size=48,
                learning_rate=2.0e-3,
                patience=6,
                seed=37,
            ),
        )
        truth_synergy = ground_truth_synergy_edges(metadata)
        peid = estimate_gru_peid_graphs(
            result,
            windows,
            intervention_samples=384,
            bins=5,
            top_synergy_sources_per_target=4,
            extra_synergy_candidates=truth_synergy,
            null_reps=0,
            source_mode="latest",
            seed=37,
        )

        scores = compute_causal_role_scores(
            pairwise_edges=peid["pairwise_edges"],
            synergy_edges=peid["synergy_edges"],
            n_components=12,
            module_labels=metadata["module_labels"],
            top_pairwise_edges=38,
        )
        gateway_rank = int(scores.loc[scores["component"] == "component_01", "gateway_rank"].iloc[0])

        self.assertLess(result.metrics["test_rmse"], result.metrics["persistence_test_rmse"])
        self.assertLessEqual(gateway_rank, 3)

    def test_gateway_synergy_benchmark_pipeline_returns_role_summary(self) -> None:
        result = run_gateway_synergy_benchmark(
            synthetic_config=SyntheticRungeConfig(
                dynamics_kind="nonlinear_gateway_synergy12",
                n_components=12,
                n_steps=220,
                burn_in=80,
                noise_scale=0.018,
                nonlinear_strength=0.18,
                synergy_strength=0.26,
                seed=43,
                apply_observation_mixing=False,
            ),
            gru_config=GruTrainingConfig(
                hidden_dim=24,
                epochs=10,
                batch_size=48,
                learning_rate=2.0e-3,
                patience=5,
                seed=43,
            ),
            mlp_config=MlpTrainingConfig(
                hidden_dims=(32,),
                epochs=8,
                batch_size=48,
                learning_rate=2.0e-3,
                patience=4,
                seed=44,
            ),
            history=4,
            horizon=1,
            intervention_samples=256,
            bins=5,
        )

        role_summary = result["role_summary"]
        self.assertEqual(str(role_summary.loc["Ground-truth dynamics", "top_gateway"]), "component_01")
        self.assertLessEqual(int(role_summary.loc["GRU + PEID", "component_01_gateway_rank"]), 3)
        self.assertIn("gateway_gru_role_scores", result)
        self.assertIn("synergy_necessity_ablation", result)
        self.assertIn("pairwise_only_component_01_gateway_rank", result["synergy_necessity_ablation"].columns)
        self.assertFalse(result["gateway_truth_synergy"].empty)

    def test_unmixed_synthetic_metadata_exports_sparse_ground_truth_edges(self) -> None:
        config = SyntheticRungeConfig(
            n_components=8,
            n_steps=96,
            burn_in=24,
            seed=11,
            apply_observation_mixing=False,
            synergy_strength=0.2,
            xor_synergy_targets=2,
        )

        _, metadata = simulate_runge_like_dynamics(config)
        pairwise = ground_truth_pairwise_edges(metadata)
        synergy = ground_truth_synergy_edges(metadata)

        target0_sources = set(pairwise.loc[pairwise["target_index"] == 0, "source_index"])
        self.assertIn(0, target0_sources)
        self.assertIn(1, target0_sources)
        self.assertIn(int(metadata["parents_a"][0]) - 1, target0_sources)
        self.assertIn(int(metadata["parents_b"][0]) - 1, target0_sources)
        self.assertEqual(metadata["coordinate_space"], "mechanism")
        self.assertEqual(metadata["synergy_strength"], 0.2)
        self.assertEqual(metadata["xor_synergy_targets"], 2)
        source_i, source_j = sorted((int(metadata["parents_a"][0]) - 1, int(metadata["parents_b"][0]) - 1))
        self.assertIn((source_i, source_j, 0), set(synergy[["source_i", "source_j", "target_index"]].itertuples(index=False, name=None)))

    def test_supervised_windows_use_history_and_next_target(self) -> None:
        frame = pd.DataFrame(
            np.arange(30, dtype=np.float32).reshape(10, 3),
            columns=["component_01", "component_02", "component_03"],
        )

        windows = make_supervised_windows(frame, history=4, horizon=1)

        self.assertEqual(windows.X.shape, (6, 4, 3))
        self.assertEqual(windows.y.shape, (6, 3))
        np.testing.assert_allclose(windows.X[0], frame.iloc[0:4].to_numpy())
        np.testing.assert_allclose(windows.y[0], frame.iloc[4].to_numpy())

    def test_gru_learns_synthetic_dynamics_better_than_persistence(self) -> None:
        frame, _ = simulate_runge_like_dynamics(
            SyntheticRungeConfig(
                n_components=6,
                n_steps=180,
                burn_in=48,
                noise_scale=0.025,
                nonlinear_strength=0.24,
                seed=17,
            )
        )
        windows = make_supervised_windows(frame, history=4, horizon=1)

        result = train_gru_forecaster(
            windows,
            GruTrainingConfig(
                hidden_dim=16,
                epochs=18,
                batch_size=32,
                learning_rate=2.0e-3,
                patience=8,
                seed=17,
            ),
        )

        self.assertLess(result.metrics["test_nrmse"], 1.0)
        self.assertLess(result.metrics["test_rmse"], result.metrics["persistence_test_rmse"])
        pred = predict_gru_windows(result, windows.X[:5])
        self.assertEqual(pred.shape, (5, frame.shape[1]))

    def test_mlp_forecaster_supports_model_peid_estimator(self) -> None:
        frame, metadata = simulate_runge_like_dynamics(
            SyntheticRungeConfig(
                dynamics_kind="sparse_var12",
                n_components=12,
                n_steps=160,
                burn_in=48,
                noise_scale=0.025,
                seed=19,
            )
        )
        windows = make_supervised_windows(frame, history=4, horizon=1)

        result = train_mlp_forecaster(
            windows,
            MlpTrainingConfig(
                hidden_dims=(24,),
                epochs=8,
                batch_size=32,
                learning_rate=2.0e-3,
                patience=4,
                seed=19,
            ),
        )
        pred = predict_mlp_windows(result, windows.X[:7])
        peid = estimate_model_peid_graphs(
            result,
            windows,
            prediction_fn=predict_mlp_windows,
            intervention_samples=96,
            bins=4,
            top_synergy_sources_per_target=2,
            null_reps=0,
            source_mode="latest",
            seed=19,
        )
        truth = ground_truth_pairwise_edges(metadata, source_mode="latest")

        self.assertEqual(pred.shape, (7, frame.shape[1]))
        self.assertLess(result.metrics["test_nrmse"], 1.5)
        self.assertEqual(peid["pairwise_matrix"].shape, (12, 12))
        self.assertEqual(len(peid["pairwise_edges"]), 144)
        self.assertFalse(truth.empty)

    def test_diagnostics_compare_lag_and_cross_correlation_scales(self) -> None:
        rng = np.random.default_rng(3)
        frame = pd.DataFrame(rng.normal(size=(40, 5)), columns=[f"component_{i + 1:02d}" for i in range(5)])

        diagnostics = compute_series_diagnostics(frame)

        self.assertIn("lag1_autocorr_mean", diagnostics)
        self.assertIn("offdiag_corr_abs_mean", diagnostics)
        self.assertEqual(diagnostics["n_components"], 5)

    def test_discrete_peid_pair_synergy_is_positive_for_xor(self) -> None:
        x = np.array([0, 0, 1, 1])
        y = np.array([0, 1, 0, 1])
        z = np.logical_xor(x, y).astype(int)

        ei_x = discrete_effective_information(x, z)
        ei_y = discrete_effective_information(y, z)
        ei_xy = discrete_effective_information(np.column_stack([x, y]), z)

        self.assertAlmostEqual(ei_x, 0.0)
        self.assertAlmostEqual(ei_y, 0.0)
        self.assertAlmostEqual(ei_xy, 1.0)
        self.assertGreater(ei_xy - ei_x - ei_y, 0.9)

    def test_gru_tm_ei_accepts_continuous_states(self) -> None:
        class CopyResult:
            model = object()
            x_mean = np.zeros((1, 2, 2), dtype=np.float32)
            x_std = np.ones((1, 2, 2), dtype=np.float32)
            y_mean = np.zeros((1, 2), dtype=np.float32)
            y_std = np.ones((1, 2), dtype=np.float32)

        rng = np.random.default_rng(11)
        windows = type(
            "Windows",
            (),
            {
                "X": rng.normal(size=(96, 2, 2)).astype(np.float32),
                "y": rng.normal(size=(96, 2)).astype(np.float32),
                "target_time": None,
            },
        )()

        def copy_latest(result, intervention_windows):
            return np.asarray(intervention_windows[:, -1, :], dtype=np.float32)

        tm_result = estimate_gru_tm_ei_graphs(
            CopyResult(),
            windows,
            prediction_fn=copy_latest,
            intervention_samples=128,
            source_mode="latest",
            seed=5,
        )

        matrix = tm_result["pairwise_matrix"]
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(len(tm_result["pairwise_edges"]), 4)
        self.assertGreater(float(matrix[0, 0]), float(matrix[0, 1]))


if __name__ == "__main__":
    unittest.main()
