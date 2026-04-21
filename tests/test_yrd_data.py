import unittest

import numpy as np
import pandas as pd
import xarray as xr

from yrd.config import YRDExperimentConfig
from yrd.data import (
    build_one_step_samples,
    build_time_splits,
    select_station_metadata,
    standardize_dataset,
)


class YRDConfigTests(unittest.TestCase):
    def test_default_horizons_and_variables_match_design(self) -> None:
        cfg = YRDExperimentConfig()
        self.assertEqual(cfg.horizons, (1, 24))
        self.assertIn("O3", cfg.target_variables)
        self.assertIn("PM2.5", cfg.target_variables)
        self.assertEqual(cfg.history_hours, 24)

    def test_one_step_config_can_express_single_horizon_hourly_setup(self) -> None:
        cfg = YRDExperimentConfig(sample_mode="one_step", horizons=(1,))
        self.assertEqual(cfg.sample_mode, "one_step")
        self.assertEqual(cfg.horizons, (1,))
        self.assertEqual(cfg.history_hours, 24)


class YRDSplitTests(unittest.TestCase):
    def test_time_splits_match_year_boundaries(self) -> None:
        cfg = YRDExperimentConfig()
        splits = build_time_splits(cfg)
        self.assertEqual(splits["train_end"], pd.Timestamp("2021-12-31 23:00:00"))
        self.assertEqual(splits["val_end"], pd.Timestamp("2022-12-31 23:00:00"))
        self.assertEqual(splits["test_end"], pd.Timestamp("2023-12-31 23:00:00"))


class YRDStandardizationTests(unittest.TestCase):
    def test_standardize_dataset_uses_training_statistics(self) -> None:
        time = np.array(["2021-12-31T23", "2022-01-01T00"], dtype="datetime64[h]")
        station = np.array(["s1"], dtype=object)
        ds = xr.Dataset(
            {"O3": (("time", "station"), np.array([[10.0], [14.0]]))},
            coords={"time": time, "station": station},
        )
        scaled, stats = standardize_dataset(ds, train_end=np.datetime64("2021-12-31T23"))
        self.assertAlmostEqual(float(scaled["O3"].isel(time=0, station=0)), 0.0, places=6)
        self.assertAlmostEqual(float(stats["O3"]["mean"]), 10.0, places=6)


class YRDStationSelectionTests(unittest.TestCase):
    def test_select_station_metadata_filters_city_and_dataset_intersection(self) -> None:
        metadata = pd.DataFrame(
            {
                "station_id": ["s1", "s2", "s3", "s4"],
                "city_en": ["shanghai", "nanjing", "shanghai", "shanghai"],
            }
        )

        selected = select_station_metadata(
            metadata,
            available_station_ids=["s1", "s3", "s5"],
            city_en="shanghai",
        )

        self.assertEqual(selected["station_id"].tolist(), ["s1", "s3"])

    def test_select_station_metadata_applies_smoke_limit_after_filtering(self) -> None:
        metadata = pd.DataFrame(
            {
                "station_id": ["s1", "s2", "s3"],
                "city_en": ["shanghai", "shanghai", "shanghai"],
            }
        )

        selected = select_station_metadata(
            metadata,
            available_station_ids=["s1", "s2", "s3"],
            city_en="shanghai",
            station_limit=2,
        )

        self.assertEqual(selected["station_id"].tolist(), ["s1", "s2"])


class YRDOneStepSampleTests(unittest.TestCase):
    def test_build_one_step_samples_returns_current_hour_joint_snapshots(self) -> None:
        time = np.array(
            [
                "2021-12-31T22",
                "2021-12-31T23",
                "2022-01-01T00",
                "2023-01-01T00",
            ],
            dtype="datetime64[h]",
        )
        station = np.array(["s1", "s2"], dtype=object)
        ds = xr.Dataset(
            {
                "O3": (("time", "station"), np.arange(8, dtype=float).reshape(4, 2)),
                "PM2.5": (("time", "station"), np.arange(10, 18, dtype=float).reshape(4, 2)),
                "t2m": (("time", "station"), np.arange(20, 28, dtype=float).reshape(4, 2)),
            },
            coords={"time": time, "station": station},
        )
        metadata = pd.DataFrame(
            {
                "station_id": ["s1", "s2"],
                "city_en": ["shanghai", "shanghai"],
            }
        )
        cfg = YRDExperimentConfig(
            sample_mode="one_step",
            horizons=(1,),
            target_variables=("O3", "PM2.5"),
            meteorology_variables=("t2m",),
            train_end=pd.Timestamp("2021-12-31 23:00:00"),
            val_end=pd.Timestamp("2022-12-31 23:00:00"),
            test_end=pd.Timestamp("2023-12-31 23:00:00"),
        )

        result = build_one_step_samples(ds, metadata, cfg)

        self.assertEqual(result["n_stations"], 2)
        self.assertEqual(result["n_features"], 3)
        self.assertEqual(result["splits"]["train"]["X"].shape, (1, 2, 3))
        self.assertEqual(result["splits"]["val"]["X"].shape, (1, 2, 3))
        self.assertEqual(set(result["splits"]["train"]["targets"]), {1})
        self.assertEqual(result["splits"]["train"]["targets"][1].shape, (1, 4))
        self.assertEqual(result["splits"]["val"]["targets"][1].shape, (1, 4))

    def test_build_one_step_samples_keeps_target_timestamp_for_split_assignment(self) -> None:
        time = np.array(
            [
                "2021-12-31T23",
                "2022-01-01T00",
                "2023-01-01T00",
            ],
            dtype="datetime64[h]",
        )
        station = np.array(["s1"], dtype=object)
        ds = xr.Dataset(
            {
                "O3": (("time", "station"), np.array([[1.0], [2.0], [3.0]])),
                "PM2.5": (("time", "station"), np.array([[4.0], [5.0], [6.0]])),
                "t2m": (("time", "station"), np.array([[7.0], [8.0], [9.0]])),
            },
            coords={"time": time, "station": station},
        )
        metadata = pd.DataFrame({"station_id": ["s1"], "city_en": ["shanghai"]})
        cfg = YRDExperimentConfig(
            sample_mode="one_step",
            horizons=(1,),
            target_variables=("O3", "PM2.5"),
            meteorology_variables=("t2m",),
            train_end=pd.Timestamp("2021-12-31 23:00:00"),
            val_end=pd.Timestamp("2022-12-31 23:00:00"),
            test_end=pd.Timestamp("2023-12-31 23:00:00"),
        )

        result = build_one_step_samples(ds, metadata, cfg)

        self.assertEqual(result["splits"]["train"]["X"].shape[0], 0)
        self.assertEqual(result["splits"]["val"]["X"].shape[0], 1)
        self.assertEqual(result["splits"]["test"]["X"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
