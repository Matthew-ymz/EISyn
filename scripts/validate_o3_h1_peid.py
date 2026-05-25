#!/usr/bin/env python3
"""Validate the O3 H1 NOx-VOC synergy hypothesis with ML and PEID graphs."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "dataset_bthsa_yrd_aqi_mete_emis.nc"
STATION_PATH = ROOT / "data" / "stations_bthsa.csv"
RESULTS_DIR = ROOT / "results" / "o3_h1_peid"
CACHE_DIR = ROOT / "exp" / "cache" / "o3_h1_peid"

SOURCE_VARIABLES = ("NOx", "VOC", "Temp", "RH", "SSR", "BLH", "WS")
FEATURE_COLUMNS = (
    "meic_NOx",
    "meic_VOC",
    "meic_PM25",
    "meic_PM10",
    "meic_SO2",
    "meic_NH3",
    "PM25_mean",
    "PM25_peak",
    "temp_c",
    "dewpoint_c",
    "RH",
    "msdwswrf",
    "blh",
    "WS",
    "u100",
    "v100",
    "WD_sin",
    "WD_cos",
    "sp",
    "tp",
    "dayofyear_sin",
    "dayofyear_cos",
    "lag1_O3_peak",
    "lag1_PM25_mean",
    "lag1_PM25_peak",
    "city_code",
    "lon",
    "lat",
)
SOURCE_TO_FEATURE = {
    "NOx": "meic_NOx",
    "VOC": "meic_VOC",
    "Temp": "temp_c",
    "RH": "RH",
    "SSR": "msdwswrf",
    "BLH": "blh",
    "WS": "WS",
}


def build_sillman_reference_surface(
    *,
    grid_size: int = 96,
    voc_min: float = 1.0,
    voc_max: float = 100.0,
    nox_min: float = 0.1,
    nox_max: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a documented Sillman-style isopleth shape surrogate.

    This is not a chemical box model. It encodes the qualitative Sillman
    structure: O3 response increases with NOx in the NOx-sensitive region,
    decreases with NOx in the NOx-saturated region, and has a log-log ridge
    that separates those regimes.
    """

    voc_grid = np.geomspace(voc_min, voc_max, grid_size)
    nox_grid = np.geomspace(nox_min, nox_max, grid_size)
    rows: list[dict[str, float]] = []
    for voc in voc_grid:
        ridge_nox = 0.28 * voc**0.82
        for nox in nox_grid:
            ratio = nox / ridge_nox
            ridge_response = 8.5 + 8.0 * np.log10(voc)
            response = ridge_response * ratio * np.exp(1.0 - ratio)
            response += 0.9 * np.log10(voc / voc_min)
            rows.append({"voc": float(voc), "nox": float(nox), "o3_response": float(max(response, 0.0))})
    surface = pd.DataFrame(rows)
    ridge_rows = []
    for voc, group in surface.groupby("voc", sort=True):
        best = group.loc[group["o3_response"].idxmax()]
        ridge_rows.append(
            {
                "voc": float(voc),
                "ridge_nox": float(best["nox"]),
                "ridge_o3_response": float(best["o3_response"]),
            }
        )
    return surface, pd.DataFrame(ridge_rows)


def _nearest_surface_value(surface: pd.DataFrame, x_col: str, y_col: str, z_col: str, x: float, y: float) -> float:
    dist = (np.log(surface[x_col].astype(float)) - np.log(float(x))) ** 2
    dist += (np.log(surface[y_col].astype(float)) - np.log(float(y))) ** 2
    return float(surface.loc[dist.idxmin(), z_col])


def surface_regime_diagnostics(
    surface: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    z_col: str,
) -> dict[str, bool | float]:
    """Check for the Sillman ridge and both qualitative sensitivity regimes."""

    clean = surface[[x_col, y_col, z_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or clean[x_col].nunique() < 4 or clean[y_col].nunique() < 4:
        return {
            "has_positive_ridge": False,
            "has_nox_sensitive_regime": False,
            "has_voc_sensitive_regime": False,
            "passes_sillman_shape_check": False,
            "z_range": 0.0,
        }

    x_low, x_mid, x_high = np.quantile(clean[x_col], [0.15, 0.5, 0.85])
    y_low, y_mid, y_high = np.quantile(clean[y_col], [0.15, 0.5, 0.85])
    z_range = max(float(clean[z_col].max() - clean[z_col].min()), 1e-9)
    threshold = 0.02 * z_range

    nox_sensitive_delta = _nearest_surface_value(clean, x_col, y_col, z_col, x_mid, y_high) - _nearest_surface_value(
        clean, x_col, y_col, z_col, x_low, y_high
    )
    nox_saturated_delta = _nearest_surface_value(clean, x_col, y_col, z_col, x_high, y_low) - _nearest_surface_value(
        clean, x_col, y_col, z_col, x_mid, y_low
    )
    voc_sensitive_delta = _nearest_surface_value(clean, x_col, y_col, z_col, x_high, y_mid) - _nearest_surface_value(
        clean, x_col, y_col, z_col, x_high, y_low
    )

    ridge = clean.loc[clean.groupby(y_col, observed=True)[z_col].idxmax()].sort_values(y_col)
    x_min = float(clean[x_col].min())
    x_max = float(clean[x_col].max())
    internal_ridge_share = float(((ridge[x_col] > x_min) & (ridge[x_col] < x_max)).mean())
    if len(ridge) > 2 and float(np.std(np.log(ridge[x_col]))) > 0.0 and float(np.std(np.log(ridge[y_col]))) > 0.0:
        ridge_corr = float(np.corrcoef(np.log(ridge[y_col]), np.log(ridge[x_col]))[0, 1])
    else:
        ridge_corr = 0.0
    has_positive_ridge = bool(internal_ridge_share >= 0.35 and ridge_corr > 0.25)
    has_nox_sensitive_regime = bool(nox_sensitive_delta > threshold)
    has_voc_sensitive_regime = bool(nox_saturated_delta < -threshold and voc_sensitive_delta > threshold)
    return {
        "has_positive_ridge": has_positive_ridge,
        "has_nox_sensitive_regime": has_nox_sensitive_regime,
        "has_voc_sensitive_regime": has_voc_sensitive_regime,
        "passes_sillman_shape_check": bool(has_positive_ridge and has_nox_sensitive_regime and has_voc_sensitive_regime),
        "z_range": float(z_range),
        "nox_sensitive_delta": float(nox_sensitive_delta),
        "nox_saturated_delta": float(nox_saturated_delta),
        "voc_sensitive_delta": float(voc_sensitive_delta),
        "ridge_internal_share": float(internal_ridge_share),
        "ridge_loglog_corr": float(ridge_corr),
    }


@dataclass(frozen=True)
class H1Config:
    station_scope: str = "metadata_all"
    city_en: str = "beijing"
    city_center_lon: float = 116.397
    city_center_lat: float = 39.908
    months: tuple[int, ...] = (5, 6, 7, 8, 9)
    daylight_hours: tuple[int, ...] = (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
    afternoon_hours: tuple[int, ...] = (12, 13, 14, 15, 16, 17, 18)
    clear_sky_ssr_min: float = 90.0
    min_daylight_hours: int = 6
    min_afternoon_hours: int = 3
    max_samples: int = 90000
    random_state: int = 42
    n_estimators: int = 260
    min_samples_leaf: int = 20
    mlp_hidden_layer_sizes: tuple[int, ...] = (128, 64)
    mlp_alpha: float = 1e-4
    mlp_max_iter: int = 1000
    mlp_early_stopping: bool = True
    peid_bins: int = 3
    peid_alpha: float = 0.5
    peid_min_source_count: int = 40
    sillman_grid_size: int = 96


def compute_relative_humidity(t2m_k: np.ndarray | pd.Series, d2m_k: np.ndarray | pd.Series) -> np.ndarray:
    temp_c = np.asarray(t2m_k, dtype=float) - 273.15
    dewpoint_c = np.asarray(d2m_k, dtype=float) - 273.15
    rh = 100.0 * np.exp(
        (17.625 * dewpoint_c) / (243.04 + dewpoint_c)
        - (17.625 * temp_c) / (243.04 + temp_c)
    )
    return np.clip(rh, 0.0, 100.0)


def haversine_km(lon: np.ndarray, lat: np.ndarray, center_lon: float, center_lat: float) -> np.ndarray:
    radius_km = 6371.0088
    lon1 = np.deg2rad(np.asarray(lon, dtype=float))
    lat1 = np.deg2rad(np.asarray(lat, dtype=float))
    lon2 = math.radians(center_lon)
    lat2 = math.radians(center_lat)
    dlon = lon1 - lon2
    dlat = lat1 - lat2
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * math.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(a))


def assign_station_groups(stations: pd.DataFrame) -> pd.DataFrame:
    result = stations.copy()
    nox_threshold = float(result["mean_nox"].median())
    voc_threshold = float(result["mean_voc"].median())
    result["nox_group"] = np.where(result["mean_nox"] >= nox_threshold, "high", "low")
    result["voc_group"] = np.where(result["mean_voc"] >= voc_threshold, "high", "low")
    result["station_group"] = "nox_" + result["nox_group"] + "_voc_" + result["voc_group"]
    return result


def entropy_bits(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


def conditional_total_correlation_bits(posterior: np.ndarray, source_keys: list[tuple[int, ...]]) -> float:
    posterior = np.asarray(posterior, dtype=float)
    if posterior.size == 0:
        return 0.0
    source_array = np.asarray(source_keys)
    if source_array.ndim != 2 or source_array.shape[1] < 2:
        return 0.0
    joint_entropy = entropy_bits(posterior)
    marginal_entropy_sum = 0.0
    for col_idx in range(source_array.shape[1]):
        values = source_array[:, col_idx]
        marginal_probs = np.array([posterior[values == value].sum() for value in np.unique(values)])
        marginal_entropy_sum += entropy_bits(marginal_probs)
    return float(marginal_entropy_sum - joint_entropy)


def source_synergy_bits(
    states: pd.DataFrame,
    *,
    source_cols: list[str],
    target_col: str,
    target_bins: int,
    alpha: float,
    min_source_count: int,
) -> dict[str, float]:
    grouped = states.groupby(source_cols + [target_col], observed=True).size().rename("count").reset_index()
    row_counts = grouped.groupby(source_cols, observed=True)["count"].sum().rename("row_total").reset_index()
    row_counts = row_counts[row_counts["row_total"] >= min_source_count]
    if len(row_counts) < 2:
        return {"synergy": 0.0, "source_support": int(len(row_counts)), "total_count": 0}

    filtered = grouped.merge(row_counts[source_cols], on=source_cols, how="inner")
    source_keys = [tuple(int(row[col]) for col in source_cols) for _, row in row_counts.iterrows()]
    key_to_idx = {key: idx for idx, key in enumerate(source_keys)}
    counts = np.zeros((len(source_keys), target_bins), dtype=float)
    for row in filtered.itertuples(index=False):
        source_key = tuple(int(getattr(row, col)) for col in source_cols)
        target_state = int(getattr(row, target_col))
        counts[key_to_idx[source_key], target_state] += float(getattr(row, "count"))

    smoothed = counts + float(alpha)
    probs = smoothed / smoothed.sum(axis=1, keepdims=True)
    target_probs = probs.mean(axis=0)
    synergy = 0.0
    for target_idx, target_prob in enumerate(target_probs):
        if target_prob <= 0:
            continue
        posterior = probs[:, target_idx] / probs[:, target_idx].sum()
        synergy += float(target_prob) * conditional_total_correlation_bits(posterior, source_keys)
    return {
        "synergy": float(synergy),
        "source_support": int(len(source_keys)),
        "total_count": int(row_counts["row_total"].sum()),
    }


def effective_information_bits(
    states: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    target_bins: int,
    alpha: float,
    min_source_count: int,
) -> dict[str, float]:
    grouped = states.groupby([source_col, target_col], observed=True).size().rename("count").reset_index()
    row_counts = grouped.groupby(source_col, observed=True)["count"].sum().rename("row_total").reset_index()
    row_counts = row_counts[row_counts["row_total"] >= min_source_count]
    if len(row_counts) < 2:
        return {"ei": 0.0, "source_support": int(len(row_counts)), "total_count": 0}

    filtered = grouped.merge(row_counts[[source_col]], on=source_col, how="inner")
    source_values = [int(value) for value in row_counts[source_col].tolist()]
    value_to_idx = {value: idx for idx, value in enumerate(source_values)}
    counts = np.zeros((len(source_values), target_bins), dtype=float)
    for row in filtered.itertuples(index=False):
        counts[value_to_idx[int(getattr(row, source_col))], int(getattr(row, target_col))] += float(
            getattr(row, "count")
        )
    probs = (counts + float(alpha)) / (counts + float(alpha)).sum(axis=1, keepdims=True)
    target_probs = probs.mean(axis=0)
    ei = entropy_bits(target_probs) - float(np.apply_along_axis(entropy_bits, 1, probs).mean())
    return {"ei": float(ei), "source_support": int(len(source_values)), "total_count": int(row_counts["row_total"].sum())}


def _cyclic(values: pd.Series, period: float) -> tuple[np.ndarray, np.ndarray]:
    radians = 2.0 * np.pi * values.astype(float).to_numpy() / float(period)
    return np.sin(radians), np.cos(radians)


def build_peak_o3_feature_table(ds: xr.Dataset, stations: pd.DataFrame, config: H1Config) -> pd.DataFrame:
    station_ids = stations["station_id"].astype(str).tolist()
    selected = ds.sel(station=station_ids)
    selected = selected.sel(time=selected["time"].dt.month.isin(config.months))
    selected = selected.sel(time=selected["time"].dt.hour.isin(config.daylight_hours))
    variables = [
        "PM2.5",
        "O3",
        "t2m",
        "d2m",
        "sp",
        "tp",
        "blh",
        "msdwswrf",
        "u100",
        "v100",
        "meic_PM25",
        "meic_PM10",
        "meic_NOx",
        "meic_SO2",
        "meic_VOC",
        "meic_NH3",
    ]
    hourly = selected[variables].to_dataframe().reset_index()
    hourly = hourly.merge(
        stations[["station_id", "station_name", "city_en", "lon", "lat"]],
        left_on="station",
        right_on="station_id",
        how="inner",
    )
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly["date"] = hourly["time"].dt.floor("D")
    hourly["hour"] = hourly["time"].dt.hour
    hourly["temp_c"] = hourly["t2m"].astype(float) - 273.15
    hourly["dewpoint_c"] = hourly["d2m"].astype(float) - 273.15
    hourly["RH"] = compute_relative_humidity(hourly["t2m"], hourly["d2m"])
    hourly["WS"] = np.sqrt(hourly["u100"].astype(float) ** 2 + hourly["v100"].astype(float) ** 2)
    direction_rad = np.arctan2(hourly["u100"].astype(float), hourly["v100"].astype(float))
    hourly["WD_sin"] = np.sin(direction_rad)
    hourly["WD_cos"] = np.cos(direction_rad)
    hourly = hourly.replace([np.inf, -np.inf], np.nan)

    predictor_cols = [
        "meic_NOx",
        "meic_VOC",
        "meic_PM25",
        "meic_PM10",
        "meic_SO2",
        "meic_NH3",
        "temp_c",
        "dewpoint_c",
        "RH",
        "msdwswrf",
        "blh",
        "WS",
        "u100",
        "v100",
        "WD_sin",
        "WD_cos",
        "sp",
        "tp",
        "lon",
        "lat",
    ]
    daily_rows: list[dict[str, object]] = []
    for (station_id, date), group in hourly.groupby(["station_id", "date"], sort=True):
        day = group.dropna(subset=["O3", *predictor_cols])
        afternoon = day[day["hour"].isin(config.afternoon_hours)]
        if len(day) < config.min_daylight_hours or len(afternoon) < config.min_afternoon_hours:
            continue
        if float(day["msdwswrf"].mean()) < config.clear_sky_ssr_min:
            continue
        payload: dict[str, object] = {
            "station": station_id,
            "station_id": station_id,
            "station_name": str(day["station_name"].iloc[0]),
            "city_en": str(day["city_en"].iloc[0]),
            "date": pd.Timestamp(date),
            "O3_peak": float(afternoon["O3"].max()),
            "PM25_mean": float(day["PM2.5"].mean()),
            "PM25_peak": float(afternoon["PM2.5"].max()),
            "n_daylight_hours": int(len(day)),
            "n_afternoon_hours": int(len(afternoon)),
        }
        for col in predictor_cols:
            payload[col] = float(day[col].mean())
        daily_rows.append(payload)

    frame = pd.DataFrame(daily_rows)
    if frame.empty:
        raise ValueError("No station-day records remain after summer clear-sky daytime filtering.")
    time = pd.to_datetime(frame["date"])
    frame["dayofyear_sin"], frame["dayofyear_cos"] = _cyclic(time.dt.dayofyear, 366.0)
    frame = frame.sort_values(["station_id", "date"]).reset_index(drop=True)
    frame["lag1_O3_peak"] = frame.groupby("station_id", sort=False)["O3_peak"].shift(1)
    frame["lag1_PM25_mean"] = frame.groupby("station_id", sort=False)["PM25_mean"].shift(1)
    frame["lag1_PM25_peak"] = frame.groupby("station_id", sort=False)["PM25_peak"].shift(1)

    city_codes = {city: idx for idx, city in enumerate(sorted(stations["city_en"].dropna().unique().tolist()))}
    station_stats = (
        frame.groupby("station_id", sort=False)
        .agg(mean_nox=("meic_NOx", "mean"), mean_voc=("meic_VOC", "mean"))
        .reset_index()
    )
    station_groups = stations.merge(station_stats, on="station_id", how="inner")
    station_groups["city_code"] = station_groups["city_en"].map(city_codes).astype(float)
    station_groups = assign_station_groups(station_groups)
    frame = frame.merge(
        station_groups[["station_id", "city_code", "nox_group", "voc_group", "station_group"]],
        on="station_id",
        how="left",
    )
    frame = (
        frame.dropna(subset=["O3_peak", *FEATURE_COLUMNS, "station_group"])
        .sort_values(["date", "station_id"])
        .reset_index(drop=True)
    )
    frame["split"] = np.select(
        [
            pd.to_datetime(frame["date"]) <= pd.Timestamp("2021-12-31"),
            pd.to_datetime(frame["date"]) <= pd.Timestamp("2022-12-31"),
        ],
        ["train", "val"],
        default="test",
    )
    if len(frame) > config.max_samples:
        frame = (
            frame.groupby(["split", "station_group"], group_keys=False, observed=True)
            .sample(frac=min(1.0, config.max_samples / len(frame)), random_state=config.random_state)
            .sort_values(["date", "station_id"])
            .reset_index(drop=True)
        )
    return frame.reset_index(drop=True)


def select_stations(stations: pd.DataFrame, config: H1Config) -> pd.DataFrame:
    if config.station_scope == "metadata_all":
        selected = stations.copy()
    elif config.station_scope == "city":
        selected = stations[stations["city_en"] == config.city_en].copy()
    else:
        raise ValueError("station_scope must be 'metadata_all' or 'city'.")
    selected["station_id"] = selected["station_id"].astype(str)
    return selected.dropna(subset=["station_id", "city_en", "lon", "lat"]).reset_index(drop=True)


def build_feature_table(config: H1Config) -> pd.DataFrame:
    stations = pd.read_csv(STATION_PATH)
    stations = select_stations(stations, config)
    if stations.empty:
        raise ValueError(f"No stations found for station_scope={config.station_scope!r}, city_en={config.city_en!r}.")
    with xr.open_dataset(DATASET_PATH) as ds:
        available = set(ds["station"].astype(str).values.tolist())
        stations = stations[stations["station_id"].isin(available)].copy()
        if stations.empty:
            raise ValueError("No selected stations are present in the NetCDF dataset.")
        return build_peak_o3_feature_table(ds, stations, config)


def fit_random_forest(frame: pd.DataFrame, config: H1Config):
    from sklearn.ensemble import RandomForestRegressor

    train = frame[frame["split"] == "train"]
    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
        n_jobs=-1,
    )
    model.fit(train[list(FEATURE_COLUMNS)], train["O3_peak"])
    return model


def build_mlp_regressor(config: H1Config):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=config.mlp_hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=config.mlp_alpha,
            early_stopping=config.mlp_early_stopping,
            max_iter=config.mlp_max_iter,
            random_state=config.random_state,
        ),
    )


def fit_models(frame: pd.DataFrame, config: H1Config) -> dict[str, object]:
    train = frame[frame["split"] == "train"]
    models = {
        "rf": fit_random_forest(frame, config),
        "mlp": build_mlp_regressor(config),
    }
    models["mlp"].fit(train[list(FEATURE_COLUMNS)], train["O3_peak"])
    return models


def metrics_for_splits(frame: pd.DataFrame, models, config: H1Config) -> pd.DataFrame:
    if not isinstance(models, dict):
        models = {"rf": models}
    rows = []
    train_mean = float(frame.loc[frame["split"] == "train", "O3_peak"].mean())
    for split, part in frame.groupby("split", sort=False):
        y = part["O3_peak"].to_numpy(dtype=float)
        baseline = np.full_like(y, train_mean, dtype=float)
        model_predictions = [(name, model.predict(part[list(FEATURE_COLUMNS)])) for name, model in models.items()]
        for name, values in [*model_predictions, ("train_mean", baseline)]:
            diff = values - y
            rows.append(
                {
                    "split": split,
                    "model": name,
                    "n_samples": int(len(part)),
                    "rmse": float(np.sqrt(np.mean(diff * diff))),
                    "mae": float(np.mean(np.abs(diff))),
                    "corr": float(np.corrcoef(y, values)[0, 1]) if np.std(values) > 0 and np.std(y) > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def discretize_quantile(series: pd.Series, bins: int) -> pd.Series:
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=bins, labels=False, duplicates="drop").astype(int)


def compute_peid_tables(frame: pd.DataFrame, config: H1Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    states = pd.DataFrame(index=frame.index)
    for source_name, feature_name in SOURCE_TO_FEATURE.items():
        states[f"{source_name}_src"] = discretize_quantile(frame[feature_name], config.peid_bins)
    states["O3hat_tgt"] = discretize_quantile(frame["O3_pred"], config.peid_bins)

    pairwise_rows = []
    for source_name in SOURCE_VARIABLES:
        result = effective_information_bits(
            states,
            source_col=f"{source_name}_src",
            target_col="O3hat_tgt",
            target_bins=config.peid_bins,
            alpha=config.peid_alpha,
            min_source_count=config.peid_min_source_count,
        )
        pairwise_rows.append({"source": source_name, "target": "O3hat", **result})

    synergy_rows = []
    source_sets = list(itertools.combinations(SOURCE_VARIABLES, 2))
    source_sets.append(("NOx", "VOC", "Temp"))
    for source_set in source_sets:
        result = source_synergy_bits(
            states,
            source_cols=[f"{source}_src" for source in source_set],
            target_col="O3hat_tgt",
            target_bins=config.peid_bins,
            alpha=config.peid_alpha,
            min_source_count=config.peid_min_source_count,
        )
        synergy_rows.append(
            {
                "sources": "+".join(source_set),
                "target": "O3hat",
                "source_order": len(source_set),
                "synergy_raw": result["synergy"],
                "source_support": result["source_support"],
                "total_count": result["total_count"],
            }
        )
    pairwise = pd.DataFrame(pairwise_rows).sort_values("ei", ascending=False).reset_index(drop=True)
    synergy = pd.DataFrame(synergy_rows).sort_values("synergy_raw", ascending=False).reset_index(drop=True)
    return pairwise, synergy


def response_surface(model, frame: pd.DataFrame, config: H1Config, *, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = ["all_stations", *sorted(frame["station_group"].dropna().unique().tolist())]
    rows = []
    diag_rows = []
    for group in groups:
        if group == "all_stations":
            subset = frame
            surface_scope = "all_stations"
        else:
            subset = frame[frame["station_group"] == group]
            surface_scope = "station_group"
        if subset.empty:
            continue
        med = subset[list(FEATURE_COLUMNS)].median(numeric_only=True)
        nox_grid = np.quantile(subset["meic_NOx"], np.linspace(0.05, 0.95, 36))
        voc_grid = np.quantile(subset["meic_VOC"], np.linspace(0.05, 0.95, 36))
        nox_grid = np.unique(nox_grid)
        voc_grid = np.unique(voc_grid)
        grid_rows = []
        for nox in nox_grid:
            for voc in voc_grid:
                payload = med.to_dict()
                payload["meic_NOx"] = float(nox)
                payload["meic_VOC"] = float(voc)
                grid_rows.append(payload)
        grid = pd.DataFrame(grid_rows)[list(FEATURE_COLUMNS)]
        pred = model.predict(grid)
        for idx, payload in enumerate(grid_rows):
            rows.append(
                {
                    "surface_scope": surface_scope,
                    "station_group": group,
                    "model": model_name,
                    "meic_NOx": payload["meic_NOx"],
                    "meic_VOC": payload["meic_VOC"],
                    "O3_pred": float(pred[idx]),
                }
            )

        surface = pd.DataFrame(rows).query("surface_scope == @surface_scope and station_group == @group")
        low_nox, mid_nox, high_nox = np.quantile(surface["meic_NOx"], [0.1, 0.5, 0.9])
        low_voc, mid_voc, high_voc = np.quantile(surface["meic_VOC"], [0.1, 0.5, 0.9])

        def nearest_value(nox: float, voc: float) -> float:
            dist = (surface["meic_NOx"] - nox) ** 2 + (surface["meic_VOC"] - voc) ** 2
            return float(surface.loc[dist.idxmin(), "O3_pred"])

        diag_rows.append(
            {
                "surface_scope": surface_scope,
                "station_group": group,
                "model": model_name,
                "voc_effect_at_mid_nox": nearest_value(mid_nox, high_voc) - nearest_value(mid_nox, low_voc),
                "nox_low_to_mid_at_mid_voc": nearest_value(mid_nox, mid_voc) - nearest_value(low_nox, mid_voc),
                "nox_mid_to_high_at_mid_voc": nearest_value(high_nox, mid_voc) - nearest_value(mid_nox, mid_voc),
                "surface_min": float(surface["O3_pred"].min()),
                "surface_max": float(surface["O3_pred"].max()),
            }
        )
    surface_frame = pd.DataFrame(rows)
    for group in groups:
        if group == "all_stations":
            surface_scope = "all_stations"
        else:
            surface_scope = "station_group"
        subset = surface_frame[
            (surface_frame["surface_scope"] == surface_scope) & (surface_frame["station_group"] == group)
        ]
        if subset.empty:
            continue
        diagnostics = surface_regime_diagnostics(
            subset,
            x_col="meic_NOx",
            y_col="meic_VOC",
            z_col="O3_pred",
        )
        for key, value in diagnostics.items():
            match_idx = [
                idx
                for idx, row in enumerate(diag_rows)
                if row["surface_scope"] == surface_scope and row["station_group"] == group
            ][0]
            diag_rows[match_idx][key] = value
    return surface_frame, pd.DataFrame(diag_rows)


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
        }
    )


def plot_sillman_reference_surface(surface: pd.DataFrame, ridge: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib()
    pivot = surface.pivot_table(index="nox", columns="voc", values="o3_response", aggfunc="mean")
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.4, 4.8), constrained_layout=True)
    levels = np.linspace(float(surface["o3_response"].min()), float(surface["o3_response"].max()), 14)
    image = ax.contourf(x, y, pivot.to_numpy(), levels=levels, cmap="viridis")
    ax.contour(x, y, pivot.to_numpy(), levels=levels[2::2], colors="white", linewidths=0.55, alpha=0.72)
    ax.plot(ridge["voc"], ridge["ridge_nox"], color="#0B3DDB", lw=2.2, ls="--", label="Ridge")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("VOC")
    ax.set_ylabel("NOx")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.colorbar(image, ax=ax, label="O3 response surrogate")
    fig.savefig(out_dir / "sillman_reference_surface.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / "sillman_reference_surface.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_response_surface(surface: pd.DataFrame, out_dir: Path, *, output_stem: str) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib()
    groups = surface["station_group"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(groups), figsize=(5.1 * len(groups), 4.4), constrained_layout=True, squeeze=False)
    vmin = float(surface["O3_pred"].min())
    vmax = float(surface["O3_pred"].max())
    image = None
    for ax, group in zip(axes[0], groups):
        subset = surface[surface["station_group"] == group]
        pivot = subset.pivot_table(index="meic_NOx", columns="meic_VOC", values="O3_pred", aggfunc="mean")
        x = pivot.columns.to_numpy(dtype=float)
        y = pivot.index.to_numpy(dtype=float)
        image = ax.contourf(x, y, pivot.to_numpy(), levels=16, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.contour(x, y, pivot.to_numpy(), levels=8, colors="white", linewidths=0.5, alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("MEIC VOC emissions proxy")
        ax.set_ylabel("MEIC NOx emissions proxy")
        ax.set_title(group, fontsize=10, pad=8)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Predicted afternoon peak O3")
    fig.savefig(out_dir / f"{output_stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_peid_graph(pairwise: pd.DataFrame, synergy: pd.DataFrame, out_dir: Path, *, model_name: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    configure_matplotlib()
    if pairwise.empty or synergy.empty:
        fig, ax = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
        ax.set_axis_off()
        ax.text(
            0.5,
            0.55,
            f"PEID skipped ({model_name})",
            ha="center",
            va="center",
            fontsize=15,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.42,
            "Empirical surface did not pass the Sillman shape check.",
            ha="center",
            va="center",
            fontsize=9.5,
            transform=ax.transAxes,
        )
        fig.savefig(out_dir / f"peid_o3_h1_causal_graph_{model_name}.png", dpi=320, bbox_inches="tight")
        fig.savefig(out_dir / f"peid_o3_h1_causal_graph_{model_name}.pdf", bbox_inches="tight")
        plt.close(fig)
        return

    top_sources = pairwise.sort_values("ei", ascending=False)["source"].head(6).tolist()
    if "NOx" not in top_sources:
        top_sources = ["NOx", *top_sources[:5]]
    if "VOC" not in top_sources:
        top_sources = ["VOC", *top_sources[:5]]
    top_sources = list(dict.fromkeys(top_sources))[:6]

    y_positions = np.linspace(0.84, 0.16, len(top_sources))
    source_pos = {source: (0.12, float(y)) for source, y in zip(top_sources, y_positions)}
    target_pos = (0.84, 0.5)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.set_axis_off()

    def box(center: tuple[float, float], label: str, face: str) -> None:
        w, h = 0.16, 0.075
        ax.add_patch(
            FancyBboxPatch(
                (center[0] - w / 2, center[1] - h / 2),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.012",
                linewidth=0.8,
                edgecolor="#2f3b45",
                facecolor=face,
                transform=ax.transAxes,
                zorder=3,
            )
        )
        ax.text(center[0], center[1], label, ha="center", va="center", fontsize=9, transform=ax.transAxes, zorder=4)

    for source, pos in source_pos.items():
        box(pos, source, "#e7f0f7")
    box(target_pos, "O3hat", "#fff0d6")

    max_ei = max(float(pairwise["ei"].max()), 1e-12)
    for row in pairwise.itertuples(index=False):
        if row.source not in source_pos:
            continue
        start = (source_pos[row.source][0] + 0.08, source_pos[row.source][1])
        end = (target_pos[0] - 0.08, target_pos[1])
        width = 0.6 + 3.2 * float(row.ei) / max_ei
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=width,
                color="#4C78A8",
                alpha=0.72,
                connectionstyle="arc3,rad=0.05",
                transform=ax.transAxes,
                zorder=1,
            )
        )

    hyper = synergy[synergy["sources"] == "NOx+VOC"]
    if not hyper.empty and {"NOx", "VOC"}.issubset(source_pos):
        value = float(hyper.iloc[0]["synergy_raw"])
        max_syn = max(float(synergy["synergy_raw"].max()), 1e-12)
        junction = (0.47, 0.5)
        for source in ("NOx", "VOC"):
            start = (source_pos[source][0] + 0.08, source_pos[source][1])
            ax.plot([start[0], junction[0]], [start[1], junction[1]], color="#D17B0F", lw=1.2, ls="--", transform=ax.transAxes)
        ax.scatter([junction[0]], [junction[1]], s=42, color="white", edgecolor="#D17B0F", linewidth=1.2, transform=ax.transAxes, zorder=5)
        ax.add_patch(
            FancyArrowPatch(
                junction,
                (target_pos[0] - 0.08, target_pos[1]),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=0.8 + 4.0 * value / max_syn,
                color="#D17B0F",
                alpha=0.9,
                transform=ax.transAxes,
                zorder=2,
            )
        )
        ax.text(junction[0], junction[1] + 0.055, "NOx+VOC", color="#9A5A00", ha="center", fontsize=8.5, transform=ax.transAxes)

    ax.text(0.02, 0.04, "Blue: pairwise EI. Orange: PEID source synergy.", fontsize=8.5, transform=ax.transAxes)
    fig.savefig(out_dir / f"peid_o3_h1_causal_graph_{model_name}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"peid_o3_h1_causal_graph_{model_name}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_notes(
    out_dir: Path,
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    synergy: pd.DataFrame,
    *,
    reference_diagnostics: dict[str, bool | float],
    peid_enabled_by_model: dict[str, bool],
) -> None:
    test_rows = metrics[metrics["split"] == "test"].set_index("model")
    metric_lines = []
    for model_name in ("rf", "mlp", "train_mean"):
        if model_name in test_rows.index:
            row = test_rows.loc[model_name]
            metric_lines.append(f"{model_name}: RMSE={row['rmse']:.3f}, MAE={row['mae']:.3f}, corr={row['corr']:.3f}")
    h1_rows = synergy[synergy["sources"] == "NOx+VOC"]
    h1_lines = []
    for row in h1_rows.itertuples(index=False):
        value = getattr(row, "synergy_raw", float("nan"))
        h1_lines.append(f"{row.model}={value:.4f} bits")
    if not h1_lines:
        h1_lines.append("none")
    lines = [
        "# Figure Notes",
        "",
        "## Sillman Reference Surface",
        "",
        "The reference panel is a parametric shape surrogate, not a full photochemical box model. It is designed to reproduce the qualitative ridge separating VOC-sensitive and NOx-sensitive regimes described by Sillman (1999).",
        "",
        "## O3 NOx-VOC Response Surface",
        "",
        "The empirical surfaces show RF and MLP fitted responses of station-day afternoon peak O3 to MEIC NOx and VOC emissions proxies while holding all other features at full-sample or station-group medians.",
        "",
        "Feature controls include meteorology, other MEIC emissions, PM2.5 daylight/afternoon summaries, station location, seasonality, and previous-day O3/PM2.5 state. Because lag features are included, each surface is conditional on prior-day pollution state.",
        "",
        "## PEID O3 H1 Causal Graph",
        "",
        "The graph summarizes pairwise effective information from each source variable to model-predicted peak O3 and highlights the NOx+VOC PEID source synergy hyperedge when the empirical surface passes the Sillman shape check.",
        "",
        "## Numeric Summary",
        "",
        f"- Test metrics: {'; '.join(metric_lines)}.",
        f"- Reference surface shape check: {bool(reference_diagnostics['passes_sillman_shape_check'])}.",
        f"- PEID enabled after empirical shape check: {peid_enabled_by_model}.",
        f"- NOx+VOC PEID source synergy to O3hat: {'; '.join(h1_lines)}.",
    ]
    for row in diagnostics.itertuples(index=False):
        lines.append(
        f"- {row.model}/{row.surface_scope}/{row.station_group}: VOC effect={row.voc_effect_at_mid_nox:.3f}, "
            f"NOx low-to-mid={row.nox_low_to_mid_at_mid_voc:.3f}, "
            f"NOx mid-to-high={row.nox_mid_to_high_at_mid_voc:.3f}."
        )
    (out_dir / "figure_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: H1Config) -> dict[str, str]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    reference_surface, reference_ridge = build_sillman_reference_surface(grid_size=config.sillman_grid_size)
    reference_diagnostics = surface_regime_diagnostics(
        reference_surface,
        x_col="nox",
        y_col="voc",
        z_col="o3_response",
    )
    reference_surface.to_csv(RESULTS_DIR / "sillman_reference_surface.csv", index=False)
    reference_ridge.to_csv(RESULTS_DIR / "sillman_reference_ridge.csv", index=False)
    plot_sillman_reference_surface(reference_surface, reference_ridge, RESULTS_DIR)

    frame = build_feature_table(config)
    frame.to_csv(CACHE_DIR / "peak_o3_feature_sample.csv", index=False)

    models = fit_models(frame, config)
    metrics = metrics_for_splits(frame, models, config)

    surface_frames = []
    diagnostic_frames = []
    pairwise_frames = []
    synergy_frames = []
    peid_enabled_by_model: dict[str, bool] = {}
    for model_name, model in models.items():
        model_frame = frame.copy()
        model_frame["O3_pred"] = model.predict(model_frame[list(FEATURE_COLUMNS)])
        eval_frame = model_frame[model_frame["split"].isin(["val", "test"])].copy()
        surface, diagnostics = response_surface(model, model_frame, config, model_name=model_name)
        global_surface = surface[surface["surface_scope"] == "all_stations"].copy()
        grouped_surface = surface[surface["surface_scope"] == "station_group"].copy()
        global_surface.to_csv(RESULTS_DIR / f"all_stations_peak_o3_response_surface_{model_name}.csv", index=False)
        grouped_surface.to_csv(RESULTS_DIR / f"beijing_peak_o3_response_surface_{model_name}.csv", index=False)
        plot_response_surface(
            global_surface,
            RESULTS_DIR,
            output_stem=f"all_stations_peak_o3_response_surface_{model_name}",
        )
        if not grouped_surface.empty:
            plot_response_surface(
                grouped_surface,
                RESULTS_DIR,
                output_stem=f"beijing_peak_o3_response_surface_{model_name}",
            )

        global_diagnostics = diagnostics[diagnostics["surface_scope"] == "all_stations"]
        peid_enabled = bool(global_diagnostics["passes_sillman_shape_check"].fillna(False).any())
        peid_enabled_by_model[model_name] = peid_enabled
        if peid_enabled:
            pairwise, synergy = compute_peid_tables(eval_frame, config)
            pairwise.insert(0, "model", model_name)
            synergy.insert(0, "model", model_name)
        else:
            pairwise = pd.DataFrame(columns=["model", "source", "target", "ei", "source_support", "total_count"])
            synergy = pd.DataFrame(
                columns=[
                    "model",
                    "sources",
                    "target",
                    "source_order",
                    "synergy_raw",
                    "source_support",
                    "total_count",
                    "skipped_reason",
                ]
            )
            synergy.loc[0] = {
                "model": model_name,
                "sources": "NOx+VOC",
                "target": "O3hat",
                "source_order": 2,
                "synergy_raw": np.nan,
                "source_support": 0,
                "total_count": 0,
                "skipped_reason": "empirical_surface_failed_sillman_shape_check",
            }
        plot_peid_graph(pairwise, synergy, RESULTS_DIR, model_name=model_name)
        surface_frames.append(surface)
        diagnostic_frames.append(diagnostics)
        pairwise_frames.append(pairwise)
        synergy_frames.append(synergy)

    surface = pd.concat(surface_frames, ignore_index=True)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    pairwise = pd.concat(pairwise_frames, ignore_index=True)
    synergy = pd.concat(synergy_frames, ignore_index=True)

    metrics.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    pairwise.to_csv(RESULTS_DIR / "peid_pairwise_edges.csv", index=False)
    synergy.to_csv(RESULTS_DIR / "peid_synergy_hyperedges.csv", index=False)
    surface.to_csv(RESULTS_DIR / "beijing_peak_o3_response_surface.csv", index=False)
    diagnostics.to_csv(RESULTS_DIR / "response_surface_diagnostics.csv", index=False)

    write_notes(
        RESULTS_DIR,
        metrics,
        diagnostics,
        synergy,
        reference_diagnostics=reference_diagnostics,
        peid_enabled_by_model=peid_enabled_by_model,
    )

    manifest = {
        "config": asdict(config),
        "data_scope": {
            "station_scope": config.station_scope,
            "city_en": config.city_en,
            "n_rows": int(len(frame)),
            "n_stations": int(frame["station"].nunique()),
            "n_cities": int(frame["city_en"].nunique()),
            "cities": sorted(frame["city_en"].dropna().unique().tolist()),
            "station_groups": sorted(frame["station_group"].dropna().unique().tolist()),
            "source_proxy_note": "meic_NOx and meic_VOC are emissions proxies, not ground precursor concentrations.",
        },
        "artifacts": {
            "feature_sample": str(CACHE_DIR / "peak_o3_feature_sample.csv"),
            "sillman_reference_surface_csv": str(RESULTS_DIR / "sillman_reference_surface.csv"),
            "sillman_reference_ridge_csv": str(RESULTS_DIR / "sillman_reference_ridge.csv"),
            "sillman_reference_surface_png": str(RESULTS_DIR / "sillman_reference_surface.png"),
            "sillman_reference_surface_pdf": str(RESULTS_DIR / "sillman_reference_surface.pdf"),
            "model_metrics": str(RESULTS_DIR / "model_metrics.csv"),
            "peid_pairwise_edges": str(RESULTS_DIR / "peid_pairwise_edges.csv"),
            "peid_synergy_hyperedges": str(RESULTS_DIR / "peid_synergy_hyperedges.csv"),
            "response_surface_grid": str(RESULTS_DIR / "beijing_peak_o3_response_surface.csv"),
            "response_surface_grid_rf": str(RESULTS_DIR / "beijing_peak_o3_response_surface_rf.csv"),
            "response_surface_grid_mlp": str(RESULTS_DIR / "beijing_peak_o3_response_surface_mlp.csv"),
            "all_stations_response_surface_grid_rf": str(
                RESULTS_DIR / "all_stations_peak_o3_response_surface_rf.csv"
            ),
            "all_stations_response_surface_grid_mlp": str(
                RESULTS_DIR / "all_stations_peak_o3_response_surface_mlp.csv"
            ),
            "response_surface_diagnostics": str(RESULTS_DIR / "response_surface_diagnostics.csv"),
            "response_surface_png_rf": str(RESULTS_DIR / "beijing_peak_o3_response_surface_rf.png"),
            "response_surface_pdf_rf": str(RESULTS_DIR / "beijing_peak_o3_response_surface_rf.pdf"),
            "response_surface_png_mlp": str(RESULTS_DIR / "beijing_peak_o3_response_surface_mlp.png"),
            "response_surface_pdf_mlp": str(RESULTS_DIR / "beijing_peak_o3_response_surface_mlp.pdf"),
            "all_stations_response_surface_png_rf": str(
                RESULTS_DIR / "all_stations_peak_o3_response_surface_rf.png"
            ),
            "all_stations_response_surface_pdf_rf": str(
                RESULTS_DIR / "all_stations_peak_o3_response_surface_rf.pdf"
            ),
            "all_stations_response_surface_png_mlp": str(
                RESULTS_DIR / "all_stations_peak_o3_response_surface_mlp.png"
            ),
            "all_stations_response_surface_pdf_mlp": str(
                RESULTS_DIR / "all_stations_peak_o3_response_surface_mlp.pdf"
            ),
            "peid_graph_png_rf": str(RESULTS_DIR / "peid_o3_h1_causal_graph_rf.png"),
            "peid_graph_pdf_rf": str(RESULTS_DIR / "peid_o3_h1_causal_graph_rf.pdf"),
            "peid_graph_png_mlp": str(RESULTS_DIR / "peid_o3_h1_causal_graph_mlp.png"),
            "peid_graph_pdf_mlp": str(RESULTS_DIR / "peid_o3_h1_causal_graph_mlp.pdf"),
            "figure_notes": str(RESULTS_DIR / "figure_notes.md"),
        },
        "reference_surface_diagnostics": reference_diagnostics,
        "empirical_surface_passes_sillman_shape_check": peid_enabled_by_model,
    }
    (RESULTS_DIR / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest["artifacts"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate O3 H1 NOx-VOC synergy with ML response surfaces and PEID.")
    parser.add_argument("--station-scope", choices=("metadata_all", "city"), default=H1Config.station_scope)
    parser.add_argument("--city-en", default=H1Config.city_en)
    parser.add_argument("--max-samples", type=int, default=H1Config.max_samples)
    parser.add_argument("--n-estimators", type=int, default=H1Config.n_estimators)
    parser.add_argument("--mlp-max-iter", type=int, default=H1Config.mlp_max_iter)
    parser.add_argument("--no-mlp-early-stopping", action="store_true")
    parser.add_argument("--random-state", type=int, default=H1Config.random_state)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = H1Config(
        station_scope=args.station_scope,
        city_en=args.city_en,
        max_samples=args.max_samples,
        n_estimators=args.n_estimators,
        mlp_max_iter=args.mlp_max_iter,
        mlp_early_stopping=not args.no_mlp_early_stopping,
        random_state=args.random_state,
    )
    artifacts = run(config)
    print(json.dumps(artifacts, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
