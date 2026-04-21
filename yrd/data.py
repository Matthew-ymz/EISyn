from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .config import YRDExperimentConfig


def build_time_splits(cfg: YRDExperimentConfig) -> dict[str, pd.Timestamp]:
    return {
        "train_end": cfg.train_end,
        "val_end": cfg.val_end,
        "test_end": cfg.test_end,
    }


def load_station_metadata(cfg: YRDExperimentConfig) -> pd.DataFrame:
    path = cfg.root_dir / cfg.station_path
    frame = pd.read_csv(path)
    return frame.sort_values("station_id").reset_index(drop=True)


def select_station_metadata(
    metadata: pd.DataFrame,
    *,
    available_station_ids: list[str],
    city_en: str | None = None,
    station_limit: int | None = None,
) -> pd.DataFrame:
    selected = metadata[metadata["station_id"].isin(available_station_ids)].copy()
    if city_en is not None:
        selected = selected[selected["city_en"].str.lower() == city_en.lower()]
    selected = selected.sort_values("station_id").reset_index(drop=True)
    if station_limit is not None:
        selected = selected.head(station_limit).reset_index(drop=True)
    return selected


def load_dataset(
    cfg: YRDExperimentConfig,
    *,
    smoke: bool = False,
    city_en: str | None = None,
) -> tuple[xr.Dataset, pd.DataFrame]:
    metadata = load_station_metadata(cfg)
    ds = xr.open_dataset(cfg.root_dir / cfg.dataset_path)
    ds = ds[list(cfg.input_variables)].transpose("time", "station")

    station_limit = cfg.smoke_station_count if smoke else None
    metadata = select_station_metadata(
        metadata,
        available_station_ids=ds["station"].values.tolist(),
        city_en=city_en,
        station_limit=station_limit,
    )
    ds = ds.sel(station=metadata["station_id"].tolist())
    return ds, metadata


def standardize_dataset(
    ds: xr.Dataset,
    *,
    train_end: np.datetime64 | pd.Timestamp,
) -> tuple[xr.Dataset, dict[str, dict[str, float]]]:
    train = ds.sel(time=slice(None, train_end))
    stats: dict[str, dict[str, float]] = {}
    scaled = ds.copy()
    for name, da in ds.data_vars.items():
        mean = float(train[name].mean().item())
        std = float(train[name].std().item())
        if std == 0.0:
            std = 1.0
        scaled[name] = (da - mean) / std
        stats[name] = {"mean": mean, "std": std}
    return scaled, stats


def _target_names(metadata: pd.DataFrame, target_variables: tuple[str, ...]) -> list[str]:
    names: list[str] = []
    for _, row in metadata.iterrows():
        for variable in target_variables:
            names.append(f"{row['city_en']}__{row['station_id']}__{variable}")
    return names


def _future_split_name(
    future_time: pd.Timestamp,
    *,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    test_end: pd.Timestamp,
) -> str | None:
    if future_time <= train_end:
        return "train"
    if future_time <= val_end:
        return "val"
    if future_time <= test_end:
        return "test"
    return None


def build_windowed_samples(
    ds: xr.Dataset,
    metadata: pd.DataFrame,
    cfg: YRDExperimentConfig,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    scaled, stats = standardize_dataset(ds, train_end=cfg.train_end)
    feature_values = np.stack(
        [scaled[name].values.astype(np.float32) for name in cfg.input_variables],
        axis=-1,
    )
    target_values = np.stack(
        [scaled[name].values.astype(np.float32) for name in cfg.target_variables],
        axis=-1,
    )

    times = pd.to_datetime(ds["time"].values)
    max_horizon = max(cfg.horizons)
    n_time, n_stations, _ = feature_values.shape

    split_data: dict[str, dict[str, Any]] = {
        split: {"X": [], "times": [], "targets": {h: [] for h in cfg.horizons}}
        for split in ("train", "val", "test")
    }

    for end_index in range(cfg.history_hours - 1, n_time - max_horizon):
        x_window = feature_values[end_index - cfg.history_hours + 1 : end_index + 1]
        future_time = pd.Timestamp(times[end_index + max_horizon])
        split_name = _future_split_name(
            future_time,
            train_end=cfg.train_end,
            val_end=cfg.val_end,
            test_end=cfg.test_end,
        )
        if split_name is None:
            continue

        if smoke and len(split_data[split_name]["X"]) >= cfg.smoke_samples_per_split:
            continue

        split_data[split_name]["X"].append(x_window)
        split_data[split_name]["times"].append(future_time.isoformat())
        for horizon in cfg.horizons:
            target = target_values[end_index + horizon].reshape(-1)
            split_data[split_name]["targets"][horizon].append(target)

    target_names = _target_names(metadata, cfg.target_variables)
    target_dim = n_stations * len(cfg.target_variables)
    for split_name, payload in split_data.items():
        payload["X"] = _stack_or_empty(
            payload["X"],
            shape=(0, cfg.history_hours, n_stations, len(cfg.input_variables)),
        )
        payload["times"] = list(payload["times"])
        payload["targets"] = {
            horizon: _stack_or_empty(values, shape=(0, target_dim))
            for horizon, values in payload["targets"].items()
        }

    return {
        "splits": split_data,
        "stats": stats,
        "target_names": target_names,
        "station_ids": metadata["station_id"].tolist(),
        "city_names": metadata["city_en"].tolist(),
        "n_stations": n_stations,
        "n_features": len(cfg.input_variables),
    }


def _stack_or_empty(values: list[np.ndarray], *, shape: tuple[int, ...]) -> np.ndarray:
    if values:
        return np.stack(values, axis=0).astype(np.float32)
    return np.empty(shape, dtype=np.float32)


def build_one_step_samples(
    ds: xr.Dataset,
    metadata: pd.DataFrame,
    cfg: YRDExperimentConfig,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    scaled, stats = standardize_dataset(ds, train_end=cfg.train_end)
    feature_values = np.stack(
        [scaled[name].values.astype(np.float32) for name in cfg.input_variables],
        axis=-1,
    )
    target_values = np.stack(
        [scaled[name].values.astype(np.float32) for name in cfg.target_variables],
        axis=-1,
    )

    times = pd.to_datetime(ds["time"].values)
    n_time, n_stations, n_features = feature_values.shape
    horizons = tuple(cfg.horizons)
    if horizons != (1,):
        raise ValueError("build_one_step_samples currently supports horizons=(1,) only.")

    split_data: dict[str, dict[str, Any]] = {
        split: {"X": [], "times": [], "targets": {1: []}}
        for split in ("train", "val", "test")
    }

    for current_index in range(n_time - 1):
        future_time = pd.Timestamp(times[current_index + 1])
        split_name = _future_split_name(
            future_time,
            train_end=cfg.train_end,
            val_end=cfg.val_end,
            test_end=cfg.test_end,
        )
        if split_name is None:
            continue

        if smoke and len(split_data[split_name]["X"]) >= cfg.smoke_samples_per_split:
            continue

        split_data[split_name]["X"].append(feature_values[current_index])
        split_data[split_name]["times"].append(future_time.isoformat())
        split_data[split_name]["targets"][1].append(target_values[current_index + 1].reshape(-1))

    target_names = _target_names(metadata, cfg.target_variables)
    target_dim = n_stations * len(cfg.target_variables)
    for split_name, payload in split_data.items():
        payload["X"] = _stack_or_empty(payload["X"], shape=(0, n_stations, n_features))
        payload["times"] = list(payload["times"])
        payload["targets"] = {
            1: _stack_or_empty(payload["targets"][1], shape=(0, target_dim)),
        }

    return {
        "splits": split_data,
        "stats": stats,
        "target_names": target_names,
        "station_ids": metadata["station_id"].tolist(),
        "city_names": metadata["city_en"].tolist(),
        "n_stations": n_stations,
        "n_features": len(cfg.input_variables),
    }


def flatten_input_group_indices(
    cfg: YRDExperimentConfig,
    *,
    n_stations: int,
    station_index: int = 0,
) -> dict[str, list[int]]:
    n_features = len(cfg.input_variables)
    groups: dict[str, list[int]] = {
        "local_o3_history": [],
        "local_pm25_history": [],
        "local_meteorology_history": [],
        "cross_station_pollutants": [],
    }

    for hour_index in range(cfg.history_hours):
        for local_feature_name, group_name in (
            ("O3", "local_o3_history"),
            ("PM2.5", "local_pm25_history"),
        ):
            feature_index = cfg.input_variables.index(local_feature_name)
            flat_index = ((hour_index * n_stations) + station_index) * n_features + feature_index
            groups[group_name].append(flat_index)

        for met_name in cfg.meteorology_variables:
            feature_index = cfg.input_variables.index(met_name)
            flat_index = ((hour_index * n_stations) + station_index) * n_features + feature_index
            groups["local_meteorology_history"].append(flat_index)

        for other_station in range(n_stations):
            if other_station == station_index:
                continue
            for pollutant_name in cfg.target_variables:
                feature_index = cfg.input_variables.index(pollutant_name)
                flat_index = ((hour_index * n_stations) + other_station) * n_features + feature_index
                groups["cross_station_pollutants"].append(flat_index)

    return groups
