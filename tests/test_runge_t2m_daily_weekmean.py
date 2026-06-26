import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_runge_t2m_daily_weekmean import (
    build_future_mean_dataset,
    standardize_t2m_daily_anomalies,
)


class RungeT2MDailyWeekMeanTests(unittest.TestCase):
    def test_standardize_t2m_daily_anomalies_drops_feb29_and_uses_365_day_calendar(self) -> None:
        times = pd.to_datetime(
            [
                "2000-02-28",
                "2000-02-29",
                "2000-03-01",
                "2001-02-28",
                "2001-03-01",
            ]
        )
        values = np.array([[[280.0]], [[999.0]], [[284.0]], [[282.0]], [[288.0]]])
        field = xr.DataArray(values, coords={"time": times, "lat": [0.0], "lon": [0.0]}, dims=("time", "lat", "lon"))

        standardized = standardize_t2m_daily_anomalies(field)

        kept_times = pd.to_datetime(standardized["time"].values)
        self.assertNotIn(pd.Timestamp("2000-02-29"), set(kept_times))
        self.assertEqual(len(kept_times), 4)
        self.assertTrue(np.isfinite(standardized.values).all())
        feb28 = standardized.sel(time=["2000-02-28", "2001-02-28"]).values[:, 0, 0]
        mar01 = standardized.sel(time=["2000-03-01", "2001-03-01"]).values[:, 0, 0]
        np.testing.assert_allclose(feb28, np.array([-1.0, 1.0]))
        np.testing.assert_allclose(mar01, np.array([-1.0, 1.0]))

    def test_build_future_mean_dataset_starts_target_after_input_window(self) -> None:
        dates = pd.date_range("2001-01-01", periods=40, freq="D")
        frame = pd.DataFrame(
            {
                "component_01": np.arange(40.0),
                "component_02": np.arange(100.0, 140.0),
            },
            index=dates,
        )

        dataset = build_future_mean_dataset(frame, history_days=28, target_days=7, lead_days=1)

        self.assertEqual(dataset.features.shape, (6, 56))
        self.assertEqual(dataset.targets.shape, (6, 2))
        np.testing.assert_allclose(dataset.features[0, :4], [0.0, 100.0, 1.0, 101.0])
        np.testing.assert_allclose(dataset.features[0, -4:], [26.0, 126.0, 27.0, 127.0])
        np.testing.assert_allclose(dataset.targets[0], [31.0, 131.0])
        self.assertEqual(dataset.sample_times[0], pd.Timestamp("2001-01-28"))
        self.assertEqual(dataset.target_start_times[0], pd.Timestamp("2001-01-29"))
        self.assertEqual(dataset.target_end_times[0], pd.Timestamp("2001-02-04"))

    def test_cli_smoke_manifest_records_daily_future_week_frequency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            data_dir = base / "daily_2m"
            data_dir.mkdir(parents=True)
            times = pd.date_range("2000-01-01", "2002-12-31", freq="D")
            lat = np.array([-30.0, 0.0, 30.0], dtype=np.float32)
            lon = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32)
            t = np.arange(len(times), dtype=np.float32)
            field = (
                280.0
                + 0.03 * t[:, None, None]
                + lat[None, :, None] / 120.0
                + np.cos(np.deg2rad(lon))[None, None, :]
            ).astype(np.float32)
            for year in [2000, 2001, 2002]:
                mask = times.year == year
                ds = xr.Dataset(
                    {"air": (("time", "lat", "lon"), field[mask])},
                    coords={"time": times[mask], "lat": lat, "lon": lon},
                )
                ds.to_netcdf(data_dir / f"air.2m.gauss.{year}.nc")

            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_runge_t2m_daily_weekmean.py"),
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(base / "out"),
                "--start-year",
                "2000",
                "--end-year",
                "2002",
                "--n-components",
                "4",
                "--history-days",
                "14",
                "--target-days",
                "7",
                "--epochs",
                "1",
                "--hidden-dim",
                "8",
                "--batch-size",
                "32",
                "--intervention-samples",
                "32",
                "--skip-peid",
            ]

            completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)

            self.assertEqual(completed.returncode, 0)
            manifest_path = base / "out" / "results" / "runge_t2m_daily_weekmean" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["frequency"], "daily_to_future_7d_mean")
            self.assertEqual(manifest["history_days"], 14)
            self.assertEqual(manifest["target_days"], 7)
            self.assertTrue((manifest_path.parent / "component_daily_scores.csv").exists())
            self.assertTrue((manifest_path.parent / "future_7d_mean_component_targets.csv").exists())
            self.assertTrue((manifest_path.parent / "mlp_metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
