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
    candidate_run_name,
    parse_history_grid,
    rank_leaderboard,
)


class RungeRnnHistorySweepTests(unittest.TestCase):
    def test_parse_history_grid_rejects_empty_and_nonpositive_values(self) -> None:
        self.assertEqual(parse_history_grid("4,1,8,4"), (1, 4, 8))
        with self.assertRaises(ValueError):
            parse_history_grid("")
        with self.assertRaises(ValueError):
            parse_history_grid("1,0,4")

    def test_candidate_run_name_is_unique_for_history_config_and_seed(self) -> None:
        first = candidate_run_name(history=4, hidden_dim=192, dropout=0.0, weight_decay=1.0e-4, seed=42)
        second = candidate_run_name(history=8, hidden_dim=192, dropout=0.0, weight_decay=1.0e-4, seed=42)
        third = candidate_run_name(history=4, hidden_dim=128, dropout=0.1, weight_decay=1.0e-3, seed=43)

        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertIn("history_04", first)
        self.assertIn("seed42", first)

    def test_rank_leaderboard_uses_validation_metric_not_test_metric(self) -> None:
        frame = pd.DataFrame(
            [
                {"candidate": "test_best", "val_avg_rmse": 0.30, "test_avg_rmse": 0.10},
                {"candidate": "val_best", "val_avg_rmse": 0.20, "test_avg_rmse": 0.50},
            ]
        )

        ranked = rank_leaderboard(frame, rank_metric="val_avg_rmse")

        self.assertEqual(list(ranked["candidate"]), ["val_best", "test_best"])
        self.assertEqual(list(ranked["rank"]), [1, 2])

    def test_cli_smoke_writes_history_sweep_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            times = pd.date_range("2001-01-01", periods=72, freq="7D")
            rng = np.random.default_rng(53)
            x0 = np.zeros(len(times), dtype=float)
            x1 = np.zeros(len(times), dtype=float)
            for idx in range(2, len(times)):
                x0[idx] = 0.5 * x0[idx - 1] - 0.12 * x0[idx - 2] + 0.05 * rng.normal()
                x1[idx] = np.tanh(x0[idx - 1]) + 0.25 * x1[idx - 1] + 0.05 * rng.normal()
            input_path = base / "components.csv"
            pd.DataFrame({"time": times, "component_01": x0, "component_02": x1}).to_csv(input_path, index=False)

            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_runge_rnn_forecast_comparison.py"),
                "--component-scores",
                str(input_path),
                "--output-dir",
                str(base / "out"),
                "--history-grid",
                "2,4",
                "--candidate-output-root",
                "results/runge/rnn_history_sweep",
                "--horizons",
                "1,2",
                "--rnn-type",
                "gru",
                "--rnn-objective",
                "rollout_multistep",
                "--rnn-linear-blend-grid-steps",
                "3",
                "--epochs",
                "2",
                "--mlp-epochs",
                "2",
                "--hidden-dim",
                "8",
                "--hidden-dim-grid",
                "8",
                "--dropout-grid",
                "0",
                "--weight-decay-grid",
                "0.0001",
                "--top-k-refine",
                "0",
                "--final-seeds",
                "1",
                "--mlp-hidden-dim",
                "8",
                "--batch-size",
                "16",
                "--ridge-alphas",
                "0.1,1,10",
                "--bootstrap-reps",
                "2",
                "--seed",
                "1",
            ]

            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

            sweep_dir = base / "out" / "results" / "runge" / "rnn_history_sweep"
            fig_dir = base / "out" / "fig" / "runge" / "rnn_history_sweep"
            leaderboard = pd.read_csv(sweep_dir / "leaderboard.csv")
            final_metrics = pd.read_csv(sweep_dir / "final_test_metrics.csv")
            significance = json.loads((sweep_dir / "final_prediction_significance.json").read_text())

            self.assertEqual({2, 4}, set(leaderboard["history"]))
            self.assertEqual(list(leaderboard.sort_values("rank")["rank"]), [1, 2])
            self.assertIn("BestBaseline", set(final_metrics["model"]))
            self.assertIn("1", significance["horizons"])
            self.assertTrue((fig_dir / "history_sweep_rmse.png").exists())
            self.assertTrue((fig_dir / "final_multistep_rmse.png").exists())


if __name__ == "__main__":
    unittest.main()
