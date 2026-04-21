import unittest

import torch

from yrd.models import JointStationMLP, PersistenceBaseline


class PersistenceBaselineTests(unittest.TestCase):
    def test_forward_repeats_last_pollutant_state_for_each_horizon(self) -> None:
        model = PersistenceBaseline(target_dim=4, horizons=(1, 24))
        x = torch.arange(1 * 2 * 3 * 5, dtype=torch.float32).reshape(1, 2, 3, 5)
        outputs = model(x)
        self.assertEqual(outputs[1].shape, (1, 4))
        self.assertEqual(outputs[24].shape, (1, 4))

    def test_forward_supports_one_step_joint_snapshots(self) -> None:
        model = PersistenceBaseline(target_dim=4, horizons=(1,))
        x = torch.tensor(
            [
                [
                    [1.0, 10.0, 100.0],
                    [2.0, 20.0, 200.0],
                ]
            ]
        )
        outputs = model(x)
        self.assertEqual(outputs[1].shape, (1, 4))
        self.assertTrue(torch.equal(outputs[1], torch.tensor([[1.0, 10.0, 2.0, 20.0]])))


class JointStationMLPTests(unittest.TestCase):
    def test_joint_station_mlp_returns_one_prediction_per_horizon(self) -> None:
        model = JointStationMLP(
            n_stations=3,
            n_features=10,
            history_hours=24,
            target_dim=6,
            hidden_dim=32,
            horizons=(1, 24),
        )
        x = torch.randn(2, 24, 3, 10)
        outputs = model(x)
        self.assertEqual(outputs[1].shape, (2, 6))
        self.assertEqual(outputs[24].shape, (2, 6))

    def test_joint_station_mlp_resmlp_returns_one_prediction_per_horizon(self) -> None:
        model = JointStationMLP(
            n_stations=3,
            n_features=10,
            history_hours=24,
            target_dim=6,
            hidden_dim=64,
            horizons=(1, 24),
            model_name="resmlp",
            num_layers=3,
            dropout=0.1,
            norm_type="layernorm",
            activation="silu",
        )
        x = torch.randn(2, 24, 3, 10)
        outputs = model(x)
        self.assertEqual(outputs[1].shape, (2, 6))
        self.assertEqual(outputs[24].shape, (2, 6))

    def test_joint_station_mlp_baseline_mode_remains_available(self) -> None:
        model = JointStationMLP(
            n_stations=3,
            n_features=10,
            history_hours=24,
            target_dim=6,
            hidden_dim=32,
            horizons=(1, 24),
            model_name="baseline",
            num_layers=2,
            dropout=0.0,
            norm_type="layernorm",
            activation="relu",
        )
        outputs = model(torch.randn(2, 24, 3, 10))
        self.assertEqual(set(outputs), {1, 24})

    def test_joint_station_mlp_accepts_one_step_joint_snapshots_with_resmlp(self) -> None:
        model = JointStationMLP(
            n_stations=3,
            n_features=10,
            history_hours=1,
            target_dim=6,
            hidden_dim=32,
            horizons=(1,),
            model_name="resmlp",
            num_layers=2,
            dropout=0.05,
            norm_type="layernorm",
            activation="silu",
        )
        outputs = model(torch.randn(2, 3, 10))
        self.assertEqual(set(outputs), {1})
        self.assertEqual(outputs[1].shape, (2, 6))


if __name__ == "__main__":
    unittest.main()
