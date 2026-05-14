import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scripts import bthsa_pm25_deweather_shap as bthsa_shap


class BthsaPm25DeweatherShapTests(unittest.TestCase):
    def test_load_bthsa_station_metadata_matches_expected_scope(self) -> None:
        metadata = bthsa_shap.load_bthsa_station_metadata(Path("data/stations_bthsa.csv"))

        self.assertEqual(len(metadata), 228)
        self.assertEqual(metadata["city_en"].nunique(), 28)
        self.assertTrue(metadata["station_id"].is_monotonic_increasing)

    def test_compute_relative_humidity_is_clipped_to_physical_range(self) -> None:
        rh = bthsa_shap.compute_relative_humidity(
            t2m_k=np.array([293.15, 293.15, 293.15]),
            d2m_k=np.array([293.15, 303.15, 193.15]),
        )

        self.assertTrue(np.all(rh >= 0.0))
        self.assertTrue(np.all(rh <= 100.0))
        self.assertAlmostEqual(float(rh[0]), 100.0, places=5)

    def test_add_derived_features_builds_rh_wind_and_delta_temperature(self) -> None:
        frame = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=26, freq="h").tolist() * 2,
                "city_en": ["a"] * 26 + ["b"] * 26,
                "t2m": np.r_[np.arange(273.15, 299.15), np.arange(280.15, 306.15)],
                "d2m": np.r_[np.arange(270.15, 296.15), np.arange(276.15, 302.15)],
                "u100": 3.0,
                "v100": 4.0,
            }
        )

        result = bthsa_shap.add_derived_features(frame)

        for column in ["temp_c", "dewpoint_c", "RH", "WS", "WD_sin", "WD_cos", "delta_temp_24h"]:
            self.assertIn(column, result.columns)
        self.assertTrue(np.allclose(result["WS"], 5.0))
        self.assertTrue(np.isfinite(result["WD_sin"]).all())
        self.assertTrue(np.isfinite(result["WD_cos"]).all())
        self.assertTrue(pd.isna(result.loc[result["city_en"] == "a", "delta_temp_24h"]).iloc[:24].all())
        self.assertAlmostEqual(
            float(result.loc[result["city_en"] == "a", "delta_temp_24h"].iloc[24]),
            24.0,
        )

    def test_build_city_hourly_features_returns_unique_city_time_rows(self) -> None:
        times = pd.date_range("2020-01-01", periods=4, freq="h")
        stations = ["s1", "s2", "s3"]
        ds = xr.Dataset(
            {
                "PM2.5": (("time", "station"), np.arange(12, dtype=float).reshape(4, 3)),
                "O3": (("time", "station"), np.ones((4, 3))),
                "t2m": (("time", "station"), np.full((4, 3), 293.15)),
                "d2m": (("time", "station"), np.full((4, 3), 283.15)),
                "sp": (("time", "station"), np.full((4, 3), 101325.0)),
                "tp": (("time", "station"), np.zeros((4, 3))),
                "blh": (("time", "station"), np.full((4, 3), 500.0)),
                "msdwswrf": (("time", "station"), np.full((4, 3), 100.0)),
                "u100": (("time", "station"), np.full((4, 3), 1.0)),
                "v100": (("time", "station"), np.full((4, 3), 0.0)),
                "meic_PM25": (("time", "station"), np.full((4, 3), 2.0)),
                "meic_PM10": (("time", "station"), np.full((4, 3), 3.0)),
                "meic_NOx": (("time", "station"), np.full((4, 3), 4.0)),
                "meic_SO2": (("time", "station"), np.full((4, 3), 5.0)),
                "meic_VOC": (("time", "station"), np.full((4, 3), 6.0)),
                "meic_NH3": (("time", "station"), np.full((4, 3), 7.0)),
            },
            coords={"time": times, "station": stations},
        )
        metadata = pd.DataFrame(
            {
                "station_id": stations,
                "city_en": ["alpha", "alpha", "beta"],
                "city": ["A", "A", "B"],
            }
        )

        result = bthsa_shap.build_city_hourly_features(ds, metadata)

        self.assertFalse(result.duplicated(["time", "city_en"]).any())
        self.assertEqual(len(result), len(times) * 2)
        alpha_first = result[(result["time"] == times[0]) & (result["city_en"] == "alpha")].iloc[0]
        self.assertAlmostEqual(float(alpha_first["PM2.5"]), 0.5)

    def test_assign_time_split_uses_fixed_year_boundaries(self) -> None:
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(["2021-12-31 23:00", "2022-01-01 00:00", "2023-01-01 00:00"]),
                "city_en": ["a", "a", "a"],
            }
        )

        result = bthsa_shap.assign_time_split(frame)

        self.assertEqual(result["split"].tolist(), ["train", "val", "test"])


if __name__ == "__main__":
    unittest.main()
