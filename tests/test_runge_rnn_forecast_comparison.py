import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_runge_rnn_forecast_comparison import (
    forecast_recursive,
    parse_int_tuple,
)


class RungeRnnForecastComparisonTests(unittest.TestCase):
    def test_recursive_forecast_rolls_predictions_without_future_truth(self) -> None:
        def predict_next(features: np.ndarray) -> np.ndarray:
            window = features.reshape(len(features), 3, 1)
            return window[:, -1, :] + 1.0

        initial = np.array([[0.0, 1.0, 2.0]])

        forecasts = forecast_recursive(initial, predict_next, lag=3, n_components=1, max_horizon=4)

        np.testing.assert_allclose(forecasts[1], [[3.0]])
        np.testing.assert_allclose(forecasts[2], [[4.0]])
        np.testing.assert_allclose(forecasts[3], [[5.0]])
        np.testing.assert_allclose(forecasts[4], [[6.0]])

    def test_parse_int_tuple_rejects_empty_and_nonpositive_horizons(self) -> None:
        self.assertEqual(parse_int_tuple("1,2,4,8"), (1, 2, 4, 8))
        with self.assertRaises(ValueError):
            parse_int_tuple("")
        with self.assertRaises(ValueError):
            parse_int_tuple("1,0,4")

    def test_cli_smoke_writes_rnn_mlp_and_linear_multistep_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            times = pd.date_range("2001-01-01", periods=96, freq="7D")
            rng = np.random.default_rng(41)
            x0 = np.zeros(len(times), dtype=float)
            x1 = np.zeros(len(times), dtype=float)
            for idx in range(2, len(times)):
                x0[idx] = 0.55 * x0[idx - 1] - 0.15 * x0[idx - 2] + 0.08 * rng.normal()
                x1[idx] = np.tanh(x0[idx - 1]) + 0.35 * x1[idx - 1] + 0.08 * rng.normal()
            frame = pd.DataFrame({"time": times, "component_01": x0, "component_02": x1})
            input_path = base / "components.csv"
            frame.to_csv(input_path, index=False)

            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_runge_rnn_forecast_comparison.py"),
                "--component-scores",
                str(input_path),
                "--output-dir",
                str(base / "out"),
                "--lag",
                "4",
                "--horizons",
                "1,2,4",
                "--epochs",
                "2",
                "--mlp-epochs",
                "2",
                "--hidden-dim",
                "8",
                "--mlp-hidden-dim",
                "8",
                "--batch-size",
                "16",
                "--ridge-alphas",
                "0.1,1,10",
                "--seed",
                "7",
            ]

            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

            result_dir = base / "out" / "results" / "runge" / "rnn_forecast_comparison"
            manifest = json.loads((result_dir / "manifest.json").read_text())
            metrics = pd.read_csv(result_dir / "multistep_metrics.csv")

            self.assertEqual(manifest["config"]["lag"], 4)
            self.assertEqual(manifest["config"]["horizons"], [1, 2, 4])
            self.assertEqual(manifest["rnn_training"]["training_objective"], "direct_multihorizon")
            self.assertIn("best_linear_alpha", manifest)
            self.assertEqual({1, 2, 4}, set(metrics["horizon"]))
            self.assertIn("RNN", set(metrics["model"]))
            self.assertIn("MLP", set(metrics["model"]))
            self.assertIn("TunedRidge", set(metrics["model"]))
            self.assertTrue((result_dir / "summary.md").exists())
            self.assertTrue((result_dir / "prediction_significance.json").exists())

    def test_cli_smoke_accepts_gru_rollout_multistep_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            times = pd.date_range("2001-01-01", periods=84, freq="7D")
            rng = np.random.default_rng(47)
            x0 = np.zeros(len(times), dtype=float)
            x1 = np.zeros(len(times), dtype=float)
            for idx in range(2, len(times)):
                x0[idx] = 0.45 * x0[idx - 1] - 0.1 * x0[idx - 2] + 0.06 * rng.normal()
                x1[idx] = np.sin(x0[idx - 1]) + 0.25 * x1[idx - 1] + 0.06 * rng.normal()
            input_path = base / "components.csv"
            pd.DataFrame({"time": times, "component_01": x0, "component_02": x1}).to_csv(input_path, index=False)

            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_runge_rnn_forecast_comparison.py"),
                "--component-scores",
                str(input_path),
                "--output-dir",
                str(base / "out"),
                "--lag",
                "4",
                "--horizons",
                "1,2,4",
                "--rnn-type",
                "gru",
                "--rnn-objective",
                "rollout_multistep",
                "--epochs",
                "2",
                "--mlp-epochs",
                "2",
                "--hidden-dim",
                "8",
                "--mlp-hidden-dim",
                "8",
                "--batch-size",
                "16",
                "--ridge-alphas",
                "0.1,1,10",
                "--seed",
                "11",
            ]

            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

            result_dir = base / "out" / "results" / "runge" / "rnn_forecast_comparison"
            manifest = json.loads((result_dir / "manifest.json").read_text())
            metrics = pd.read_csv(result_dir / "multistep_metrics.csv")
            significance = json.loads((result_dir / "prediction_significance.json").read_text())

            self.assertEqual(manifest["config"]["rnn_type"], "gru")
            self.assertEqual(manifest["rnn_training"]["training_objective"], "rollout_multistep")
            self.assertEqual(manifest["rnn_training"]["horizons"], [1, 2, 4])
            self.assertEqual({1, 2, 4}, set(metrics["horizon"]))
            self.assertIn("1", significance["horizons"])


if __name__ == "__main__":
    unittest.main()
