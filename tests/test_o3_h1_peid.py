from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from scripts.validate_o3_h1_peid import (
    FEATURE_COLUMNS,
    H1Config,
    assign_station_groups,
    build_o3_h1_artifact_manifest,
    build_peak_o3_feature_table,
    build_station_mean_feature_table,
    select_stations,
    build_sillman_reference_surface,
    build_mlp_regressor,
    build_nox_voc_poly_regressor,
    compute_relative_humidity,
    metrics_for_splits,
    source_synergy_bits,
    surface_regime_diagnostics,
)


def test_compute_relative_humidity_reaches_saturation_when_dewpoint_equals_temperature() -> None:
    rh = compute_relative_humidity(np.array([298.15]), np.array([298.15]))

    assert np.allclose(rh, np.array([100.0]), atol=1e-6)


def test_assign_station_groups_uses_nox_voc_quantile_labels() -> None:
    stations = pd.DataFrame(
        {
            "station_id": ["low_low", "low_high", "high_low", "high_high"],
            "lon": [116.40, 116.41, 116.90, 116.91],
            "lat": [39.90, 39.91, 40.40, 40.41],
            "mean_nox": [1.0, 2.0, 10.0, 20.0],
            "mean_voc": [1.0, 20.0, 2.0, 30.0],
        }
    )

    grouped = assign_station_groups(stations)

    labels = dict(zip(grouped["station_id"], grouped["station_group"]))
    assert labels["low_low"] == "nox_low_voc_low"
    assert labels["high_high"] == "nox_high_voc_high"


def test_select_stations_metadata_all_keeps_multiple_cities() -> None:
    stations = pd.DataFrame(
        {
            "station_id": ["1001A", "2001A", "3001A"],
            "station_name": ["A", "B", "C"],
            "city_en": ["beijing", "tianjin", "jinan"],
            "lon": [116.3, 117.2, 117.0],
            "lat": [39.9, 39.1, 36.7],
        }
    )

    all_stations = select_stations(stations, H1Config(station_scope="metadata_all"))
    beijing = select_stations(stations, H1Config(station_scope="city", city_en="beijing"))

    assert set(all_stations["city_en"]) == {"beijing", "tianjin", "jinan"}
    assert beijing["city_en"].tolist() == ["beijing"]


def test_source_synergy_bits_is_positive_for_xor_mechanism() -> None:
    states = pd.DataFrame(
        {
            "a_src": [0, 0, 1, 1] * 20,
            "b_src": [0, 1, 0, 1] * 20,
        }
    )
    states["y_tgt"] = states["a_src"] ^ states["b_src"]

    result = source_synergy_bits(
        states,
        source_cols=["a_src", "b_src"],
        target_col="y_tgt",
        target_bins=2,
        alpha=0.0,
        min_source_count=1,
    )

    assert result["synergy"] > 0.9


def test_sillman_surrogate_has_ridge_and_both_sensitivity_regimes() -> None:
    surface, ridge = build_sillman_reference_surface(grid_size=60)
    diagnostics = surface_regime_diagnostics(
        surface,
        x_col="nox",
        y_col="voc",
        z_col="o3_response",
    )

    assert {"nox", "voc", "o3_response"}.issubset(surface.columns)
    assert {"voc", "ridge_nox", "ridge_o3_response"}.issubset(ridge.columns)
    assert diagnostics["has_positive_ridge"]
    assert diagnostics["has_nox_sensitive_regime"]
    assert diagnostics["has_voc_sensitive_regime"]


def _synthetic_peak_o3_dataset(periods: int = 48) -> xr.Dataset:
    times = pd.date_range("2020-07-01", periods=periods, freq="h")
    stations = ["1001A"]
    values = np.arange(periods, dtype=float).reshape(periods, 1)
    return xr.Dataset(
        {
            "PM2.5": (("time", "station"), 20.0 + values),
            "O3": (("time", "station"), values),
            "t2m": (("time", "station"), np.full((periods, 1), 303.15)),
            "d2m": (("time", "station"), np.full((periods, 1), 293.15)),
            "sp": (("time", "station"), np.full((periods, 1), 101325.0)),
            "tp": (("time", "station"), np.zeros((periods, 1))),
            "blh": (("time", "station"), np.full((periods, 1), 900.0)),
            "msdwswrf": (("time", "station"), np.full((periods, 1), 700.0)),
            "u100": (("time", "station"), np.full((periods, 1), 3.0)),
            "v100": (("time", "station"), np.full((periods, 1), 4.0)),
            "meic_PM25": (("time", "station"), np.full((periods, 1), 1.0)),
            "meic_PM10": (("time", "station"), np.full((periods, 1), 2.0)),
            "meic_NOx": (("time", "station"), np.full((periods, 1), 3.0)),
            "meic_SO2": (("time", "station"), np.full((periods, 1), 4.0)),
            "meic_VOC": (("time", "station"), np.full((periods, 1), 8.0)),
            "meic_NH3": (("time", "station"), np.full((periods, 1), 5.0)),
        },
        coords={"time": times, "station": stations},
    )


def test_build_peak_o3_feature_table_returns_station_day_peak_records() -> None:
    stations = ["1001A"]
    ds = _synthetic_peak_o3_dataset(periods=48)
    metadata = pd.DataFrame(
        {
            "station_id": stations,
            "station_name": ["Wanliu"],
            "city_en": ["beijing"],
            "lon": [116.36],
            "lat": [39.88],
        }
    )

    frame = build_peak_o3_feature_table(ds, metadata, H1Config(max_samples=100))

    assert len(frame) == 1
    assert float(frame["O3_peak"].iloc[0]) == 42.0
    assert float(frame["WS"].iloc[0]) == 5.0
    assert "PM25_mean" in frame.columns
    assert "PM25_peak" in frame.columns
    assert "lag1_O3_peak" in frame.columns
    assert "lag1_PM25_mean" in frame.columns
    assert "WD_sin" in frame.columns
    assert "meic_PM25" in frame.columns
    assert "city_code" in frame.columns
    assert frame["split"].iloc[0] == "train"


def test_all_station_feature_table_expands_nox_voc_support_and_groups_by_quantiles() -> None:
    stations = ["1001A", "2001A"]
    ds = _synthetic_peak_o3_dataset(periods=48).sel(station=["1001A"])
    values = np.stack([np.arange(48, dtype=float), np.arange(48, dtype=float) + 5.0], axis=1)
    ds = xr.Dataset(
        {
            "PM2.5": (("time", "station"), 20.0 + values),
            "O3": (("time", "station"), values),
            "t2m": (("time", "station"), np.full((48, 2), 303.15)),
            "d2m": (("time", "station"), np.full((48, 2), 293.15)),
            "sp": (("time", "station"), np.full((48, 2), 101325.0)),
            "tp": (("time", "station"), np.zeros((48, 2))),
            "blh": (("time", "station"), np.full((48, 2), 900.0)),
            "msdwswrf": (("time", "station"), np.full((48, 2), 700.0)),
            "u100": (("time", "station"), np.full((48, 2), 3.0)),
            "v100": (("time", "station"), np.full((48, 2), 4.0)),
            "meic_PM25": (("time", "station"), np.ones((48, 2))),
            "meic_PM10": (("time", "station"), np.ones((48, 2)) * 2.0),
            "meic_NOx": (("time", "station"), np.array([[1.0, 10.0]] * 48)),
            "meic_SO2": (("time", "station"), np.ones((48, 2)) * 4.0),
            "meic_VOC": (("time", "station"), np.array([[2.0, 20.0]] * 48)),
            "meic_NH3": (("time", "station"), np.ones((48, 2)) * 5.0),
        },
        coords={"time": ds["time"].to_numpy(), "station": stations},
    )
    metadata = pd.DataFrame(
        {
            "station_id": stations,
            "station_name": ["A", "B"],
            "city_en": ["beijing", "tianjin"],
            "lon": [116.36, 117.20],
            "lat": [39.88, 39.12],
        }
    )

    frame = build_peak_o3_feature_table(ds, metadata, H1Config(max_samples=100))

    assert frame["station_id"].nunique() == 2
    assert frame["city_en"].nunique() == 2
    assert frame["meic_NOx"].max() > frame["meic_NOx"].min()
    assert frame["meic_VOC"].max() > frame["meic_VOC"].min()
    assert set(frame["station_group"]) == {"nox_low_voc_low", "nox_high_voc_high"}


def test_station_mean_feature_table_aggregates_one_row_per_station() -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["s1", "s1", "s2", "s2"],
            "station": ["s1", "s1", "s2", "s2"],
            "station_name": ["A", "A", "B", "B"],
            "city_en": ["beijing", "beijing", "tianjin", "tianjin"],
            "station_group": ["nox_low_voc_low", "nox_low_voc_low", "nox_high_voc_high", "nox_high_voc_high"],
            "nox_group": ["low", "low", "high", "high"],
            "voc_group": ["low", "low", "high", "high"],
            "O3_peak": [100.0, 120.0, 140.0, 160.0],
        }
    )
    for idx, column in enumerate(FEATURE_COLUMNS):
        frame[column] = np.array([1.0, 3.0, 10.0, 14.0]) + idx

    station_mean = build_station_mean_feature_table(frame, H1Config(random_state=0))

    assert len(station_mean) == 2
    assert station_mean["station_id"].is_unique
    assert float(station_mean.loc[station_mean["station_id"] == "s1", "O3_peak"].iloc[0]) == 110.0
    assert float(station_mean.loc[station_mean["station_id"] == "s2", "meic_NOx"].iloc[0]) == 12.0
    assert set(station_mean["surface_scope"]) == {"station_mean"}
    assert set(station_mean["split"]) == {"train"}


def test_feature_columns_use_lagged_pollution_without_same_day_o3_leakage() -> None:
    assert "O3" not in FEATURE_COLUMNS
    assert "O3_peak" not in FEATURE_COLUMNS
    assert "lag1_O3_peak" in FEATURE_COLUMNS
    assert "PM25_mean" in FEATURE_COLUMNS
    assert "meic_NH3" in FEATURE_COLUMNS
    assert "city_code" in FEATURE_COLUMNS


def test_mlp_pipeline_fits_and_metrics_include_model_rows() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train"] * 8 + ["val"] * 4,
            "O3_peak": np.linspace(80.0, 140.0, 12),
        }
    )
    for idx, column in enumerate(FEATURE_COLUMNS):
        frame[column] = np.linspace(0.0, 1.0, 12) + idx * 0.01

    mlp = build_mlp_regressor(H1Config(random_state=0, mlp_max_iter=50, mlp_early_stopping=False))
    mlp.fit(frame.loc[frame["split"] == "train", list(FEATURE_COLUMNS)], frame.loc[frame["split"] == "train", "O3_peak"])
    pred = mlp.predict(frame[list(FEATURE_COLUMNS)])
    assert np.isfinite(pred).all()

    metrics = metrics_for_splits(frame, {"mlp": mlp}, H1Config())
    assert {"mlp", "train_mean"} == set(metrics["model"])


def test_nox_voc_poly_regressor_fits_feature_table_and_predicts_finite_values() -> None:
    frame = pd.DataFrame({"O3_peak": np.linspace(80.0, 140.0, 16)})
    for idx, column in enumerate(FEATURE_COLUMNS):
        if column == "meic_NOx":
            frame[column] = np.geomspace(0.1, 10.0, 16)
        elif column == "meic_VOC":
            frame[column] = np.geomspace(0.2, 20.0, 16)
        else:
            frame[column] = np.linspace(0.0, 1.0, 16) + idx

    model = build_nox_voc_poly_regressor()
    model.fit(frame[list(FEATURE_COLUMNS)], frame["O3_peak"])

    assert np.isfinite(model.predict(frame[list(FEATURE_COLUMNS)])).all()


def test_artifact_manifest_excludes_failed_display_outputs_but_keeps_diagnostics() -> None:
    artifacts = build_o3_h1_artifact_manifest(
        peid_enabled_by_model={"rf": False, "mlp": False},
        station_mean_shape_by_model={"rf": False, "mlp": False, "poly_nox_voc": True},
    )

    assert "response_surface_diagnostics" in artifacts
    assert "response_surface_grid_rf" not in artifacts
    assert "response_surface_grid_mlp" not in artifacts
    assert "all_stations_response_surface_grid_rf" not in artifacts
    assert "all_stations_response_surface_grid_mlp" not in artifacts
    assert "station_mean_response_surface_grid_rf" not in artifacts
    assert "station_mean_response_surface_grid_mlp" not in artifacts
    assert "peid_graph_png_rf" not in artifacts
    assert "station_mean_response_surface_png_rf" not in artifacts
    assert "station_mean_response_surface_png_poly_nox_voc" in artifacts
    assert "station_mean_response_surface_grid_poly_nox_voc" in artifacts
