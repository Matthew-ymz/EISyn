import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from yrd.analysis import write_markdown_summary
from yrd.train import _predict_numpy, ensure_output_layout, rebuild_joint_model_from_checkpoint, train_joint_model_with_history


class YRDTrainLayoutTests(unittest.TestCase):
    def test_ensure_output_layout_creates_cache_and_results_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ensure_output_layout(Path(tmpdir))
            self.assertTrue(paths["cache_dir"].exists())
            self.assertTrue(paths["results_dir"].exists())


class YRDMarkdownSummaryTests(unittest.TestCase):
    def test_write_markdown_summary_persists_explanatory_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "summary.md"
            write_markdown_summary(
                out,
                title="Smoke Summary",
                bullets=["Joint model beats persistence on 24h horizon."],
            )
            text = out.read_text()
            self.assertIn("Smoke Summary", text)
            self.assertIn("24h horizon", text)


class YRDCliTests(unittest.TestCase):
    def test_cli_supports_help_output(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/run_yrd_experiment.py", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("train", result.stdout)
        self.assertIn("analyze", result.stdout)

    def test_air_search_cli_supports_help_output(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/run_air_search.py", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("coarse", result.stdout)
        self.assertIn("refine", result.stdout)
        self.assertIn("report", result.stdout)


class YRDTrainingHistoryTests(unittest.TestCase):
    def test_train_joint_model_with_history_records_losses_and_best_epoch(self) -> None:
        rng = np.random.default_rng(0)
        x_train = rng.normal(size=(8, 4, 2, 3)).astype(np.float32)
        x_val = rng.normal(size=(4, 4, 2, 3)).astype(np.float32)
        y_train = {
            1: rng.normal(size=(8, 4)).astype(np.float32),
            24: rng.normal(size=(8, 4)).astype(np.float32),
        }
        y_val = {
            1: rng.normal(size=(4, 4)).astype(np.float32),
            24: rng.normal(size=(4, 4)).astype(np.float32),
        }

        result = train_joint_model_with_history(
            n_stations=2,
            n_features=3,
            history_hours=4,
            target_dim=4,
            hidden_dim=8,
            horizons=(1, 24),
            learning_rate=1e-3,
            weight_decay=0.0,
            batch_size=2,
            epochs=3,
            early_stopping_patience=2,
            seed=0,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
        )

        self.assertEqual(len(result["train_loss_history"]), 3)
        self.assertEqual(len(result["val_loss_history"]), 3)
        self.assertIn(result["best_epoch"], (1, 2, 3))
        self.assertIn("model", result)
        self.assertIn("model_kwargs", result)
        self.assertIn("stopped_early", result)
        self.assertIn("early_stopping_patience", result)

    def test_rebuild_joint_model_from_checkpoint_restores_output_shapes(self) -> None:
        payload = {
            "state_dict": JointCheckpointFixtures.state_dict(),
            "model_kwargs": {
                "n_stations": 2,
                "n_features": 3,
                "history_hours": 4,
                "target_dim": 4,
                "hidden_dim": 8,
                "horizons": (1, 24),
                "model_name": "resmlp",
                "num_layers": 3,
                "dropout": 0.1,
                "norm_type": "layernorm",
                "activation": "silu",
            },
        }
        model = rebuild_joint_model_from_checkpoint(payload)
        outputs = model(torch.randn(1, 4, 2, 3))
        self.assertEqual(outputs[1].shape, (1, 4))
        self.assertEqual(outputs[24].shape, (1, 4))

    def test_train_joint_model_with_history_supports_one_step_single_horizon_training(self) -> None:
        rng = np.random.default_rng(42)
        x_train = rng.normal(size=(8, 2, 3)).astype(np.float32)
        x_val = rng.normal(size=(4, 2, 3)).astype(np.float32)
        y_train = {
            1: rng.normal(size=(8, 4)).astype(np.float32),
        }
        y_val = {
            1: rng.normal(size=(4, 4)).astype(np.float32),
        }

        result = train_joint_model_with_history(
            n_stations=2,
            n_features=3,
            history_hours=1,
            target_dim=4,
            hidden_dim=8,
            horizons=(1,),
            learning_rate=1e-3,
            weight_decay=0.0,
            batch_size=2,
            epochs=3,
            early_stopping_patience=2,
            seed=0,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            model_name="resmlp",
            num_layers=2,
            dropout=0.05,
            norm_type="layernorm",
            activation="silu",
        )

        self.assertEqual(set(result["model_kwargs"]["horizons"]), {1})
        self.assertEqual(result["model_kwargs"]["history_hours"], 1)
        self.assertEqual(result["model_kwargs"]["model_name"], "resmlp")
        predictions = _predict_numpy(result["model"], x_val, (1,))
        self.assertEqual(predictions[1].shape, (4, 4))

    def test_rebuild_joint_model_from_checkpoint_restores_one_step_model(self) -> None:
        fixture = train_joint_model_with_history(
            n_stations=2,
            n_features=3,
            history_hours=1,
            target_dim=4,
            hidden_dim=8,
            horizons=(1,),
            learning_rate=1e-3,
            weight_decay=0.0,
            batch_size=2,
            epochs=2,
            early_stopping_patience=2,
            seed=0,
            x_train=np.random.default_rng(11).normal(size=(8, 2, 3)).astype(np.float32),
            y_train={1: np.random.default_rng(12).normal(size=(8, 4)).astype(np.float32)},
            x_val=np.random.default_rng(13).normal(size=(4, 2, 3)).astype(np.float32),
            y_val={1: np.random.default_rng(14).normal(size=(4, 4)).astype(np.float32)},
            model_name="resmlp",
            num_layers=2,
            dropout=0.05,
            norm_type="layernorm",
            activation="silu",
        )
        payload = {
            "state_dict": fixture["model"].state_dict(),
            "model_kwargs": fixture["model_kwargs"],
        }

        model = rebuild_joint_model_from_checkpoint(payload)
        outputs = model(torch.randn(1, 2, 3))
        self.assertEqual(set(outputs), {1})
        self.assertEqual(outputs[1].shape, (1, 4))


class JointCheckpointFixtures:
    @staticmethod
    def state_dict() -> dict[str, torch.Tensor]:
        result = train_joint_model_with_history(
            n_stations=2,
            n_features=3,
            history_hours=4,
            target_dim=4,
            hidden_dim=8,
            horizons=(1, 24),
            learning_rate=1e-3,
            weight_decay=0.0,
            batch_size=2,
            epochs=2,
            early_stopping_patience=2,
            seed=0,
            x_train=np.random.default_rng(1).normal(size=(8, 4, 2, 3)).astype(np.float32),
            y_train={
                1: np.random.default_rng(2).normal(size=(8, 4)).astype(np.float32),
                24: np.random.default_rng(3).normal(size=(8, 4)).astype(np.float32),
            },
            x_val=np.random.default_rng(4).normal(size=(4, 4, 2, 3)).astype(np.float32),
            y_val={
                1: np.random.default_rng(5).normal(size=(4, 4)).astype(np.float32),
                24: np.random.default_rng(6).normal(size=(4, 4)).astype(np.float32),
            },
            model_name="resmlp",
            num_layers=3,
            dropout=0.1,
            norm_type="layernorm",
            activation="silu",
        )
        return result["model"].state_dict()


if __name__ == "__main__":
    unittest.main()
