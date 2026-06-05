import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scripts.reproduce_runge2015_gateways import (
    LaggedEdge,
    RungeConfig,
    apply_paper_component_labels,
    compute_sem_effects,
    detrend_time_axis,
    ensure_causal_backend_available,
    rotated_component_order,
    save_ranking_figure,
    varimax,
    weekly_aggregate,
)


ROOT = Path(__file__).resolve().parents[1]


class Runge2015GatewaysTests(unittest.TestCase):
    def test_varimax_preserves_shape_and_returns_finite_rotation(self) -> None:
        rng = np.random.default_rng(0)
        loadings = rng.normal(size=(18, 4))

        rotated, rotation = varimax(loadings)

        self.assertEqual(rotated.shape, loadings.shape)
        self.assertEqual(rotation.shape, (4, 4))
        self.assertTrue(np.isfinite(rotated).all())
        self.assertTrue(np.isfinite(rotation).all())
        self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(4), atol=1e-6))

    def test_rotated_component_order_uses_rotated_covariance_diagonal(self) -> None:
        rotation = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        eigenvalues = np.array([5.0, 2.0, 3.0])

        order, diagonal = rotated_component_order(rotation, eigenvalues)

        np.testing.assert_array_equal(order, np.array([1, 2, 0]))
        np.testing.assert_allclose(diagonal, np.array([2.0, 5.0, 3.0]))

    def test_weekly_aggregate_returns_monotonic_week_start_index(self) -> None:
        dates = pd.date_range("2001-01-01", periods=15, freq="D")
        frame = pd.DataFrame({"component_01": np.arange(15.0)}, index=dates)

        weekly = weekly_aggregate(frame)

        self.assertTrue(weekly.index.is_monotonic_increasing)
        self.assertEqual(list(weekly.index), list(pd.date_range("2001-01-01", periods=2, freq="7D")))
        self.assertAlmostEqual(float(weekly.iloc[0, 0]), 3.0)
        self.assertAlmostEqual(float(weekly.iloc[1, 0]), 10.0)

    def test_detrend_time_axis_removes_linear_gridpoint_trend(self) -> None:
        times = pd.date_range("2001-01-01", periods=8, freq="D")
        t = np.arange(len(times), dtype=float)
        values = (2.0 * t[:, None, None] + np.array([[[1.0, -1.0]]])).astype(float)
        field = xr.DataArray(values, coords={"time": times, "lat": [0.0], "lon": [0.0, 90.0]}, dims=("time", "lat", "lon"))

        detrended = detrend_time_axis(field)

        matrix = detrended.values.reshape(len(times), -1)
        slopes = [np.polyfit(t, matrix[:, col], deg=1)[0] for col in range(matrix.shape[1])]
        self.assertTrue(np.allclose(slopes, 0.0, atol=1.0e-10))

    def test_sem_effects_recover_total_and_mediated_effect_on_tiny_dag(self) -> None:
        edges = [
            LaggedEdge(source=0, target=1, lag=1, coefficient=0.5, p_value=0.01),
            LaggedEdge(source=1, target=2, lag=1, coefficient=0.4, p_value=0.01),
            LaggedEdge(source=0, target=2, lag=2, coefficient=0.1, p_value=0.02),
        ]

        effects = compute_sem_effects(edges, n_components=3, max_lag=2)

        gateway = effects.gateway_scores.set_index("component")
        mediator = effects.mediator_scores.set_index("component")
        path = effects.path_effects.set_index(["source", "mediator", "target"])
        self.assertAlmostEqual(float(gateway.loc[0, "ace"]), 0.4, places=9)
        self.assertAlmostEqual(float(gateway.loc[2, "acs"]), 0.35, places=9)
        self.assertAlmostEqual(float(path.loc[(0, 1, 2), "mce_max_abs"]), 0.2, places=9)
        self.assertAlmostEqual(float(mediator.loc[1, "amce"]), 0.2, places=9)
        self.assertAlmostEqual(float(mediator.loc[1, "mediated_fraction"]), 0.5, places=9)

    def test_sem_effects_follow_runge_lag_resolved_recursion_and_averages(self) -> None:
        edges = [
            LaggedEdge(source=0, target=1, lag=1, coefficient=0.5, p_value=0.01),
            LaggedEdge(source=1, target=2, lag=1, coefficient=0.4, p_value=0.01),
            LaggedEdge(source=0, target=2, lag=2, coefficient=0.1, p_value=0.02),
        ]

        effects = compute_sem_effects(edges, n_components=3, max_lag=2)

        gateway = effects.gateway_scores.set_index("component")
        mediator = effects.mediator_scores.set_index("component")
        total = effects.total_effects.set_index(["source", "target", "lag"])
        path = effects.path_effects.set_index(["source", "mediator", "target"])
        self.assertAlmostEqual(float(total.loc[(0, 2, 2), "total_effect"]), 0.3, places=9)
        self.assertAlmostEqual(float(gateway.loc[0, "ace"]), 0.4, places=9)
        self.assertAlmostEqual(float(gateway.loc[2, "acs"]), 0.35, places=9)
        self.assertAlmostEqual(float(path.loc[(0, 1, 2), "mce_max_abs"]), 0.2, places=9)
        self.assertAlmostEqual(float(mediator.loc[1, "amce"]), 0.2, places=9)

    def test_default_lag_matches_runge_figure4_weekly_tau_max(self) -> None:
        self.assertEqual(RungeConfig().max_lag, 4)

    def test_paper_component_labels_relabel_internal_west_pacific_mode(self) -> None:
        frame = pd.DataFrame(
            {
                "component": [7, 18, 8, 26, 21, 48, 0],
                "ace": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.5],
            }
        )

        labelled = apply_paper_component_labels(frame)

        self.assertEqual(list(labelled["component"]), [7, 18, 8, 26, 21, 48, 0])
        self.assertEqual(list(labelled["paper_component"]), [18, 7, 26, 8, 48, 21, 0])
        self.assertEqual(int(labelled.sort_values("ace", ascending=False).iloc[0]["paper_component"]), 18)

    def test_ranking_plot_writes_png_with_outside_legend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "ranking.png"
            frame = pd.DataFrame(
                {
                    "component": [0, 1, 2],
                    "ace": [0.4, 0.3, 0.1],
                    "acs": [0.2, 0.5, 0.1],
                }
            )

            path = save_ranking_figure(frame, output, title="Synthetic ranking")

            self.assertEqual(path, output)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)

    def test_tigramite_backend_reports_missing_dependency_before_full_run(self) -> None:
        if importlib.util.find_spec("tigramite") is not None:
            self.skipTest("tigramite is installed in this environment")
        with self.assertRaisesRegex(RuntimeError, "tigramite is required"):
            ensure_causal_backend_available("tigramite")

    def test_cli_smoke_run_writes_cache_figures_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            data_dir = base / "data"
            daily_dir = data_dir / "daily"
            daily_dir.mkdir(parents=True)
            times = pd.date_range("2001-01-01", periods=60, freq="D")
            lat = np.array([-30.0, 0.0, 30.0], dtype=np.float32)
            lon = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32)
            t = np.arange(len(times), dtype=np.float32)
            field = (
                1000.0
                + 0.2 * t[:, None, None]
                + lat[None, :, None] / 90.0
                + np.cos(np.deg2rad(lon))[None, None, :]
            ).astype(np.float32)
            ds = xr.Dataset(
                {"slp": (("time", "lat", "lon"), field)},
                coords={"time": times, "lat": lat, "lon": lon},
            )
            ds.to_netcdf(daily_dir / "slp.2001.nc")

            command = [
                sys.executable,
                str(ROOT / "scripts" / "reproduce_runge2015_gateways.py"),
                "--mode",
                "smoke",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(base / "out"),
                "--start-year",
                "2001",
                "--end-year",
                "2001",
                "--n-components",
                "3",
                "--max-lag",
                "2",
                "--pc-alpha",
                "0.05",
                "--causal-backend",
                "regression",
            ]

            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

            result_dir = base / "out" / "results" / "runge" / "2015_gateways"
            fig_dir = base / "out" / "fig" / "runge" / "2015_gateways"
            manifest = json.loads((result_dir / "manifest.json").read_text())
            edges = pd.read_csv(result_dir / "causal_edges.csv")
            gateways = pd.read_csv(result_dir / "gateway_scores.csv")
            mediators = pd.read_csv(result_dir / "mediator_scores.csv")

            self.assertEqual(manifest["config"]["n_components"], 3)
            self.assertEqual(manifest["config"]["causal_backend"], "regression")
            self.assertIn("n_edges", manifest)
            self.assertEqual(manifest["n_linear_matrix_elements"], 9)
            self.assertIn("data_files", manifest)
            self.assertIn("dependency_versions", manifest)
            self.assertIn("component", gateways.columns)
            self.assertIn("component", mediators.columns)
            self.assertIn("source", edges.columns)
            self.assertIn("lag", pd.read_csv(result_dir / "total_effects.csv").columns)
            self.assertTrue((result_dir / "summary.md").exists())
            self.assertIn("causal gateways", (result_dir / "summary.md").read_text())
            self.assertTrue((result_dir / "linear_coefficient_matrix.csv").exists())
            self.assertTrue((result_dir / "lagged_linear_matrices.npz").exists())
            self.assertTrue((fig_dir / "causal_network.png").exists())
            self.assertTrue((fig_dir / "gateway_ranking.png").exists())
            self.assertTrue((fig_dir / "mediator_ranking.png").exists())


if __name__ == "__main__":
    unittest.main()
