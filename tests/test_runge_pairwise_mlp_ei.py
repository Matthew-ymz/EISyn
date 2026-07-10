import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_runge_pairwise_mlp_ei import (
    DEFAULT_COMPONENT_SCORES,
    PairwiseMlpEiConfig,
    build_lagged_dataset,
    compare_ei_to_linear_coefficients,
    compute_ei_path_effects,
    discrete_effective_information,
    fit_ridge_linear_map,
    gateway_scores_from_matrix,
    estimate_pairwise_tm_ei_matrix,
    train_or_load_mlp,
    sparsify_ei_graph,
)


ROOT = Path(__file__).resolve().parents[1]


class RungePairwiseMlpEiTests(unittest.TestCase):
    def test_default_component_scores_use_1948_2026_data(self) -> None:
        self.assertEqual(
            DEFAULT_COMPONENT_SCORES,
            Path("results/runge_slp_daily_1948_2026_20260628/results/runge/2015_gateways/component_weekly_scores.csv"),
        )

    def test_build_lagged_dataset_keeps_temporal_order(self) -> None:
        frame = pd.DataFrame(
            {
                "component_01": [1.0, 2.0, 3.0, 4.0],
                "component_02": [10.0, 20.0, 30.0, 40.0],
            }
        )

        features, targets = build_lagged_dataset(frame, lag=2, horizon=1)

        self.assertEqual(features.shape, (2, 4))
        self.assertEqual(targets.shape, (2, 2))
        np.testing.assert_allclose(features[0], [1.0, 10.0, 2.0, 20.0])
        np.testing.assert_allclose(targets[0], [3.0, 30.0])

    def test_ridge_linear_map_recovers_scaled_linear_transition(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.normal(size=(128, 5)).astype(np.float32)
        true_weight = rng.normal(size=(5, 3)).astype(np.float32)
        true_bias = rng.normal(size=(1, 3)).astype(np.float32)
        y = x @ true_weight + true_bias

        weight, bias = fit_ridge_linear_map(x, y, alpha=0.0)
        prediction = x @ weight.T + bias

        np.testing.assert_allclose(prediction, y, atol=1.0e-5)

    def test_train_or_load_mlp_records_validation_best_checkpoint(self) -> None:
        rng = np.random.default_rng(29)
        x = rng.normal(size=(90, 6)).astype(np.float32)
        y = np.column_stack(
            [
                0.6 * x[:, 0] - 0.2 * x[:, 1],
                np.sin(x[:, 2]) + 0.1 * x[:, 3],
            ]
        ).astype(np.float32)
        splits = {
            "train": (x[:54], y[:54]),
            "val": (x[54:72], y[54:72]),
            "test": (x[72:], y[72:]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "mlp_transition.pt"
            config = PairwiseMlpEiConfig(
                hidden_dim=16,
                epochs=8,
                batch_size=16,
                seed=5,
                force_retrain=True,
            )
            model, _, loss_history, reused = train_or_load_mlp(
                splits,
                config,
                model_path,
                config_hash="unit-test",
            )

            self.assertFalse(reused)
            self.assertEqual(len(loss_history), 8)
            training = getattr(model, "training_summary")
            self.assertIn("val_loss_history", training)
            self.assertIn("best_val_loss", training)
            self.assertIn("best_epoch", training)
            self.assertIn("residual_scale", training)
            self.assertGreaterEqual(float(training["residual_scale"]), -0.5)
            self.assertLessEqual(float(training["residual_scale"]), 1.0)
            self.assertLessEqual(float(training["best_val_loss"]), float(training["val_loss_history"][-1]) + 1.0e-12)

            import torch

            payload = torch.load(model_path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["best_epoch"], training["best_epoch"])
            self.assertIn("val_loss_history", payload)

    def test_discrete_effective_information_detects_deterministic_binary_copy(self) -> None:
        source_states = np.array([0, 0, 1, 1])
        target_states = np.array([0, 0, 1, 1])

        ei = discrete_effective_information(source_states, target_states)

        self.assertAlmostEqual(ei, 1.0, places=9)

    def test_gateway_scores_average_outgoing_and_incoming_ei(self) -> None:
        matrix = np.array(
            [
                [0.0, 0.2, 0.4],
                [0.1, 0.0, 0.3],
                [0.0, 0.5, 0.0],
            ]
        )

        scores = gateway_scores_from_matrix(matrix, ["C1", "C2", "C3"]).set_index("component")

        self.assertAlmostEqual(float(scores.loc["C1", "gateway_ei"]), 0.3)
        self.assertAlmostEqual(float(scores.loc["C2", "susceptibility_ei"]), 0.35)
        self.assertEqual(int(scores.loc["C3", "out_rank"]), 2)

    def test_sparsify_ei_graph_keeps_source_topk_edges_and_removes_self_edges(self) -> None:
        matrix = np.array(
            [
                [0.9, 0.4, 0.2, 0.1],
                [0.5, 0.8, 0.3, 0.7],
                [0.1, 0.6, 0.9, 0.2],
                [0.4, 0.2, 0.5, 0.7],
            ]
        )

        sparse = sparsify_ei_graph(matrix, mode="source_topk", topk=2)

        expected = np.array(
            [
                [0.0, 0.4, 0.2, 0.0],
                [0.5, 0.0, 0.0, 0.7],
                [0.0, 0.6, 0.0, 0.2],
                [0.4, 0.0, 0.5, 0.0],
            ]
        )
        np.testing.assert_allclose(sparse, expected)

    def test_sparsify_ei_graph_keeps_bidirectional_topk_edges(self) -> None:
        matrix = np.array(
            [
                [0.9, 0.4, 0.2, 0.1],
                [0.5, 0.8, 0.3, 0.7],
                [0.1, 0.6, 0.9, 0.2],
                [0.4, 0.2, 0.5, 0.7],
            ]
        )

        sparse = sparsify_ei_graph(matrix, mode="bidirectional_topk", topk=1)

        expected = np.array(
            [
                [0.0, 0.4, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.7],
                [0.0, 0.6, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0],
            ]
        )
        np.testing.assert_allclose(sparse, expected)

    def test_ei_path_effects_accumulate_total_effects_and_mediator_scores(self) -> None:
        direct = np.array(
            [
                [0.0, 0.5, 0.1],
                [0.0, 0.0, 0.4],
                [0.0, 0.0, 0.0],
            ]
        )

        effects = compute_ei_path_effects(direct, ["C1", "C2", "C3"], path_alpha=1.0).gateway_scores
        mediator = compute_ei_path_effects(direct, ["C1", "C2", "C3"], path_alpha=1.0).mediator_scores

        gateway = effects.set_index("component")
        mediated = mediator.set_index("component")
        self.assertAlmostEqual(float(gateway.loc["C1", "ace"]), 0.4, places=9)
        self.assertAlmostEqual(float(gateway.loc["C3", "acs"]), 0.35, places=9)
        self.assertAlmostEqual(float(mediated.loc["C2", "amce"]), 0.1, places=9)

    def test_transport_map_ei_matrix_accepts_continuous_states(self) -> None:
        class CopyModel:
            def eval(self) -> None:
                return None

            def __call__(self, tensor):
                import torch

                return torch.stack([tensor[:, -2], tensor[:, -1]], dim=1)

        rng = np.random.default_rng(11)
        train_features = rng.normal(size=(96, 4))
        scalers = {
            "x_mean": np.zeros((1, 4), dtype=np.float32),
            "x_std": np.ones((1, 4), dtype=np.float32),
            "y_mean": np.zeros((1, 2), dtype=np.float32),
            "y_std": np.ones((1, 2), dtype=np.float32),
        }

        matrix, edges = estimate_pairwise_tm_ei_matrix(
            CopyModel(),
            scalers,
            train_features,
            n_components=2,
            lag=2,
            intervention_samples=128,
            low_q=0.05,
            high_q=0.95,
            source_mode="latest",
            seed=5,
        )

        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(len(edges), 4)
        self.assertGreater(float(matrix[0, 0]), float(matrix[0, 1]))

    def test_compare_ei_to_linear_coefficients_keeps_one_row_per_matrix_element(self) -> None:
        ei = np.array([[0.8, 0.2], [0.1, 0.4]])
        linear = np.array([[0.7, -0.3], [0.0, 0.5]])

        rows, summary = compare_ei_to_linear_coefficients(ei, linear, ["component_01", "component_02"])

        self.assertEqual(len(rows), 4)
        self.assertEqual(int(summary["n_elements"]), 4)
        self.assertEqual(int(summary["n_off_diagonal_elements"]), 2)
        self.assertIn("abs_linear_coefficient", rows.columns)
        row = rows.set_index(["source_index", "target_index"]).loc[(0, 1)]
        self.assertAlmostEqual(float(row["pairwise_ei"]), 0.2)
        self.assertAlmostEqual(float(row["linear_coefficient"]), -0.3)
        self.assertAlmostEqual(float(row["abs_linear_coefficient"]), 0.3)

    def test_cli_smoke_writes_cached_mlp_and_pairwise_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            times = pd.date_range("2001-01-01", periods=90, freq="7D")
            rng = np.random.default_rng(7)
            x0 = rng.normal(size=len(times))
            x1 = np.roll(x0, 1) + 0.1 * rng.normal(size=len(times))
            x2 = -0.5 * np.roll(x1, 1) + 0.1 * rng.normal(size=len(times))
            frame = pd.DataFrame(
                {
                    "time": times,
                    "component_01": x0,
                    "component_02": x1,
                    "component_03": x2,
                }
            )
            input_path = base / "components.csv"
            frame.to_csv(input_path, index=False)
            linear_path = base / "linear.csv"
            pd.DataFrame(
                np.array(
                    [
                        [0.2, 0.1, 0.0],
                        [0.0, 0.3, 0.4],
                        [0.5, 0.0, 0.1],
                    ]
                ),
                index=["component_01", "component_02", "component_03"],
                columns=["component_01", "component_02", "component_03"],
            ).to_csv(linear_path)

            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_runge_pairwise_mlp_ei.py"),
                "--component-scores",
                str(input_path),
                "--linear-coefficients",
                str(linear_path),
                "--output-dir",
                str(base / "out"),
                "--lag",
                "2",
                "--horizon",
                "1",
                "--epochs",
                "2",
                "--hidden-dim",
                "8",
                "--intervention-samples",
                "128",
                "--bins",
                "4",
                "--seed",
                "3",
            ]

            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

            result_dir = base / "out" / "results" / "runge" / "pairwise_mlp_ei"
            fig_dir = base / "out" / "fig" / "runge" / "pairwise_mlp_ei"
            manifest = json.loads((result_dir / "manifest.json").read_text())
            edge_frame = pd.read_csv(result_dir / "pairwise_ei_edges.csv")
            gateway_frame = pd.read_csv(result_dir / "gateway_scores.csv")
            comparison_frame = pd.read_csv(result_dir / "ei_linear_coefficient_comparison.csv")
            comparison_summary = json.loads((result_dir / "ei_linear_coefficient_comparison_summary.json").read_text())

            self.assertEqual(manifest["config"]["lag"], 2)
            self.assertEqual(manifest["n_components"], 3)
            self.assertTrue((result_dir / "mlp_transition.pt").exists())
            self.assertTrue((result_dir / "pairwise_ei_matrix.csv").exists())
            self.assertEqual(len(edge_frame), 9)
            self.assertEqual(len(comparison_frame), 9)
            self.assertEqual(comparison_summary["n_elements"], 9)
            self.assertEqual(manifest["ei_linear_comparison"]["n_elements"], 9)
            self.assertIn("gateway_ei", gateway_frame.columns)
            self.assertTrue((fig_dir / "pairwise_ei_heatmap.png").exists())
            self.assertTrue((fig_dir / "gateway_ranking.png").exists())

    def test_cli_smoke_with_transport_map_estimator_writes_separate_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            times = pd.date_range("2001-01-01", periods=80, freq="7D")
            rng = np.random.default_rng(13)
            x0 = rng.normal(size=len(times))
            x1 = np.roll(x0, 1) + 0.1 * rng.normal(size=len(times))
            frame = pd.DataFrame({"time": times, "component_01": x0, "component_02": x1})
            input_path = base / "components.csv"
            frame.to_csv(input_path, index=False)

            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_runge_pairwise_mlp_ei.py"),
                "--component-scores",
                str(input_path),
                "--output-dir",
                str(base / "out"),
                "--lag",
                "2",
                "--horizon",
                "1",
                "--epochs",
                "2",
                "--hidden-dim",
                "8",
                "--intervention-samples",
                "96",
                "--ei-estimator",
                "tm",
                "--seed",
                "3",
            ]

            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

            result_dir = base / "out" / "results" / "runge" / "pairwise_mlp_tm_ei"
            fig_dir = base / "out" / "fig" / "runge" / "pairwise_mlp_tm_ei"
            manifest = json.loads((result_dir / "manifest.json").read_text())
            edge_frame = pd.read_csv(result_dir / "pairwise_ei_edges.csv")

            self.assertEqual(manifest["config"]["ei_estimator"], "tm")
            self.assertTrue((base / "out" / "results" / "runge" / "pairwise_mlp_ei" / "mlp_transition.pt").exists())
            self.assertEqual(len(edge_frame), 4)
            self.assertIn("bias_correction", edge_frame.columns)
            self.assertTrue((fig_dir / "pairwise_ei_heatmap.png").exists())

    def test_cli_smoke_with_ensemble_ridge_alphas_records_member_caches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            times = pd.date_range("2001-01-01", periods=72, freq="7D")
            rng = np.random.default_rng(19)
            x0 = rng.normal(size=len(times))
            x1 = np.roll(x0, 1) + 0.1 * rng.normal(size=len(times))
            frame = pd.DataFrame({"time": times, "component_01": x0, "component_02": x1})
            input_path = base / "components.csv"
            frame.to_csv(input_path, index=False)

            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_runge_pairwise_mlp_ei.py"),
                "--component-scores",
                str(input_path),
                "--output-dir",
                str(base / "out"),
                "--lag",
                "2",
                "--horizon",
                "1",
                "--epochs",
                "2",
                "--hidden-dim",
                "8",
                "--intervention-samples",
                "96",
                "--ensemble-ridge-alphas",
                "1,10",
                "--seed",
                "3",
            ]

            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

            result_dir = base / "out" / "results" / "runge" / "pairwise_mlp_ei"
            manifest = json.loads((result_dir / "manifest.json").read_text())

            self.assertEqual(manifest["config"]["ensemble_ridge_alphas"], [1.0, 10.0])
            self.assertEqual(len(manifest["model_caches"]), 2)
            self.assertEqual(len(manifest["ensemble_members"]), 2)
            for cache_path in manifest["model_caches"]:
                self.assertTrue(Path(cache_path).exists())

    def test_cli_smoke_with_validation_linear_blend_records_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            times = pd.date_range("2001-01-01", periods=72, freq="7D")
            rng = np.random.default_rng(23)
            x0 = rng.normal(size=len(times))
            x1 = np.roll(x0, 1) + 0.1 * rng.normal(size=len(times))
            frame = pd.DataFrame({"time": times, "component_01": x0, "component_02": x1})
            input_path = base / "components.csv"
            frame.to_csv(input_path, index=False)

            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_runge_pairwise_mlp_ei.py"),
                "--component-scores",
                str(input_path),
                "--output-dir",
                str(base / "out"),
                "--lag",
                "2",
                "--horizon",
                "1",
                "--epochs",
                "2",
                "--hidden-dim",
                "8",
                "--intervention-samples",
                "96",
                "--ensemble-ridge-alphas",
                "1,10",
                "--linear-blend-grid-steps",
                "5",
                "--seed",
                "3",
            ]

            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

            result_dir = base / "out" / "results" / "runge" / "pairwise_mlp_ei"
            manifest = json.loads((result_dir / "manifest.json").read_text())
            metrics = pd.read_csv(result_dir / "mlp_metrics.csv")

            self.assertEqual(manifest["config"]["linear_blend_grid_steps"], 5)
            self.assertTrue(manifest["linear_blend"]["enabled"])
            self.assertGreaterEqual(manifest["linear_blend"]["mlp_weight"], 0.0)
            self.assertLessEqual(manifest["linear_blend"]["mlp_weight"], 1.0)
            self.assertIn("test", set(metrics["split"]))

    def test_cli_smoke_with_path_effect_gateway_mode_writes_runge_style_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            times = pd.date_range("2001-01-01", periods=80, freq="7D")
            rng = np.random.default_rng(17)
            x0 = rng.normal(size=len(times))
            x1 = np.roll(x0, 1) + 0.1 * rng.normal(size=len(times))
            x2 = np.roll(x1, 1) + 0.1 * rng.normal(size=len(times))
            frame = pd.DataFrame({"time": times, "component_01": x0, "component_02": x1, "component_03": x2})
            input_path = base / "components.csv"
            frame.to_csv(input_path, index=False)

            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_runge_pairwise_mlp_ei.py"),
                "--component-scores",
                str(input_path),
                "--output-dir",
                str(base / "out"),
                "--lag",
                "2",
                "--horizon",
                "1",
                "--epochs",
                "2",
                "--hidden-dim",
                "8",
                "--intervention-samples",
                "96",
                "--ei-estimator",
                "tm",
                "--gateway-mode",
                "path_effect",
                "--graph-topk",
                "1",
                "--path-alpha",
                "0.8",
                "--seed",
                "3",
            ]

            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

            result_dir = base / "out" / "results" / "runge" / "pairwise_mlp_tm_ei_path_effects"
            fig_dir = base / "out" / "fig" / "runge" / "pairwise_mlp_tm_ei_path_effects"
            manifest = json.loads((result_dir / "manifest.json").read_text())
            gateways = pd.read_csv(result_dir / "gateway_scores.csv")
            mediators = pd.read_csv(result_dir / "mediator_scores.csv")

            self.assertEqual(manifest["config"]["gateway_mode"], "path_effect")
            self.assertEqual(manifest["config"]["graph_topk"], 1)
            self.assertIn("ace", gateways.columns)
            self.assertIn("amce", mediators.columns)
            self.assertTrue((result_dir / "direct_effects.csv").exists())
            self.assertTrue((result_dir / "total_effects.csv").exists())
            self.assertTrue((result_dir / "mediated_path_effects.csv").exists())
            self.assertTrue((fig_dir / "gateway_ranking.png").exists())
            self.assertTrue((fig_dir / "mediator_ranking.png").exists())

    def test_hypergraph_cli_accepts_ensemble_and_writes_order2_rankings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            times = pd.date_range("2001-01-01", periods=72, freq="7D")
            rng = np.random.default_rng(31)
            x0 = rng.normal(size=len(times))
            x1 = 0.7 * np.roll(x0, 1) + 0.2 * rng.normal(size=len(times))
            frame = pd.DataFrame({"time": times, "component_01": x0, "component_02": x1})
            input_path = base / "components.csv"
            frame.to_csv(input_path, index=False)
            pairwise_gateway_path = base / "pairwise_gateway.csv"
            pairwise_mediator_path = base / "pairwise_mediator.csv"
            pd.DataFrame(
                {
                    "component": ["component_01", "component_02"],
                    "component_index": [0, 1],
                    "ace": [0.2, 0.1],
                    "acs": [0.1, 0.2],
                }
            ).to_csv(pairwise_gateway_path, index=False)
            pd.DataFrame(
                {
                    "component": ["component_01", "component_02"],
                    "component_index": [0, 1],
                    "amce": [0.03, 0.02],
                }
            ).to_csv(pairwise_mediator_path, index=False)

            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_runge_peid_hypergraph.py"),
                "--component-scores",
                str(input_path),
                "--output-dir",
                str(base / "out"),
                "--lag",
                "2",
                "--horizon",
                "1",
                "--epochs",
                "2",
                "--hidden-dim",
                "8",
                "--intervention-samples",
                "64",
                "--order-max",
                "2",
                "--candidate-top-sources",
                "2",
                "--candidate-target-topk",
                "2",
                "--null-reps",
                "1",
                "--pairwise-gateway-path",
                str(pairwise_gateway_path),
                "--pairwise-mediator-path",
                str(pairwise_mediator_path),
                "--ensemble-ridge-alphas",
                "1,10",
                "--seed",
                "5",
            ]

            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

            result_dir = base / "out" / "results" / "runge" / "peid_hypergraph"
            manifest = json.loads((result_dir / "manifest.json").read_text())
            gateways = pd.read_csv(result_dir / "hyper_gateway_scores.csv")
            mediators = pd.read_csv(result_dir / "hyper_mediator_scores.csv")

            self.assertEqual(manifest["config"]["ensemble_ridge_alphas"], [1.0, 10.0])
            self.assertEqual(len(manifest["model_caches"]), 2)
            self.assertEqual(manifest["candidate_counts"]["order_3"], 0)
            self.assertIn("hyper_ace_order2", gateways.columns)
            self.assertIn("mediator_synergy_order2", mediators.columns)


if __name__ == "__main__":
    unittest.main()
