"""BTHSA city-level PM2.5 deweathering and SHAP helpers.

The module keeps the reusable parts of the notebook-first experiment small:
city aggregation, feature engineering, Random Forest fitting, deweathering,
SHAP extraction, and publication-friendly plot exports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


DATASET_PATH = Path("data/dataset_bthsa_yrd_aqi_mete_emis.nc")
STATION_PATH = Path("data/stations_bthsa.csv")
CACHE_DIR = Path("exp/cache/bthsa_pm25_deweather_shap")
FIG_DIR = Path("fig/bthsa_pm25_deweather_shap")

BASE_DATASET_VARIABLES = (
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
)

METEOROLOGY_FEATURES = (
    "temp_c",
    "dewpoint_c",
    "RH",
    "WS",
    "WD_sin",
    "WD_cos",
    "sp",
    "tp",
    "blh",
    "msdwswrf",
    "delta_temp_24h",
)
EMISSION_FEATURES = (
    "meic_PM25",
    "meic_PM10",
    "meic_NOx",
    "meic_SO2",
    "meic_VOC",
    "meic_NH3",
)
TIME_FEATURES = (
    "year",
    "dayofyear_sin",
    "dayofyear_cos",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
)
POLLUTION_FEATURES = ("O3",)
FEATURE_COLUMNS = POLLUTION_FEATURES + METEOROLOGY_FEATURES + EMISSION_FEATURES + TIME_FEATURES
TARGET_COLUMN = "PM2.5"

TRAIN_END = pd.Timestamp("2021-12-31 23:00:00")
VAL_END = pd.Timestamp("2022-12-31 23:00:00")
TEST_END = pd.Timestamp("2023-12-31 23:00:00")


@dataclass(frozen=True)
class ExperimentConfig:
    root_dir: Path = Path(".")
    dataset_path: Path = DATASET_PATH
    station_path: Path = STATION_PATH
    cache_dir: Path = CACHE_DIR
    fig_dir: Path = FIG_DIR
    random_state: int = 0
    n_estimators: int = 200
    min_samples_leaf: int = 5
    n_jobs: int = -1
    shap_sample_per_city: int = 5000
    deweather_samples: int = 1000
    smoke: bool = False
    smoke_cities: tuple[str, ...] = ("beijing", "tianjin")
    smoke_hours_per_city: int = 2000

    @property
    def resolved_dataset_path(self) -> Path:
        return self.root_dir / self.dataset_path

    @property
    def resolved_station_path(self) -> Path:
        return self.root_dir / self.station_path

    @property
    def resolved_cache_dir(self) -> Path:
        return self.root_dir / self.cache_dir

    @property
    def resolved_fig_dir(self) -> Path:
        return self.root_dir / self.fig_dir


def ensure_output_dirs(config: ExperimentConfig) -> None:
    config.resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    config.resolved_fig_dir.mkdir(parents=True, exist_ok=True)


def require_shap() -> Any:
    try:
        import shap  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "This experiment requires SHAP. Install a compatible version with: "
            "python -m pip install 'shap==0.44.1' 'numpy<2' 'pandas<3'"
        ) from exc
    return shap


def load_bthsa_station_metadata(path: Path = STATION_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = {"station_id", "city", "city_en", "lon", "lat"}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Station metadata is missing columns: {sorted(missing)}")
    return frame.sort_values("station_id").reset_index(drop=True)


def open_bthsa_dataset(path: Path = DATASET_PATH) -> xr.Dataset:
    try:
        ds = xr.open_dataset(path, engine="h5netcdf")
    except Exception:
        ds = xr.open_dataset(path)
    missing = [name for name in BASE_DATASET_VARIABLES if name not in ds.data_vars]
    if missing:
        raise ValueError(f"Dataset is missing required variables: {missing}")
    return ds[list(BASE_DATASET_VARIABLES)].transpose("time", "station")


def compute_relative_humidity(t2m_k: np.ndarray | pd.Series, d2m_k: np.ndarray | pd.Series) -> np.ndarray:
    temp_c = np.asarray(t2m_k, dtype=float) - 273.15
    dewpoint_c = np.asarray(d2m_k, dtype=float) - 273.15
    rh = 100.0 * np.exp(
        (17.625 * dewpoint_c) / (243.04 + dewpoint_c)
        - (17.625 * temp_c) / (243.04 + temp_c)
    )
    return np.clip(rh, 0.0, 100.0)


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["time"] = pd.to_datetime(result["time"])
    result = result.sort_values(["city_en", "time"]).reset_index(drop=True)
    result["temp_c"] = result["t2m"].astype(float) - 273.15
    result["dewpoint_c"] = result["d2m"].astype(float) - 273.15
    result["RH"] = compute_relative_humidity(result["t2m"], result["d2m"])
    u = result["u100"].astype(float).to_numpy()
    v = result["v100"].astype(float).to_numpy()
    result["WS"] = np.sqrt(u * u + v * v)
    direction_rad = np.arctan2(u, v)
    result["WD_sin"] = np.sin(direction_rad)
    result["WD_cos"] = np.cos(direction_rad)
    result["delta_temp_24h"] = result.groupby("city_en", sort=False)["temp_c"].diff(24)
    return result


def _cyclic(series: pd.Series, period: float) -> tuple[np.ndarray, np.ndarray]:
    radians = 2.0 * np.pi * series.astype(float).to_numpy() / float(period)
    return np.sin(radians), np.cos(radians)


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    time = pd.to_datetime(result["time"])
    result["year"] = time.dt.year.astype(float)
    result["dayofyear_sin"], result["dayofyear_cos"] = _cyclic(time.dt.dayofyear, 366.0)
    result["hour_sin"], result["hour_cos"] = _cyclic(time.dt.hour, 24.0)
    result["weekday_sin"], result["weekday_cos"] = _cyclic(time.dt.weekday, 7.0)
    result["month_sin"], result["month_cos"] = _cyclic(time.dt.month, 12.0)
    return result


def assign_time_split(
    frame: pd.DataFrame,
    *,
    train_end: pd.Timestamp = TRAIN_END,
    val_end: pd.Timestamp = VAL_END,
    test_end: pd.Timestamp = TEST_END,
) -> pd.DataFrame:
    result = frame.copy()
    time = pd.to_datetime(result["time"])
    result["split"] = np.select(
        [time <= train_end, time <= val_end, time <= test_end],
        ["train", "val", "test"],
        default="out_of_range",
    )
    return result


def build_city_hourly_features(ds: xr.Dataset, metadata: pd.DataFrame) -> pd.DataFrame:
    station_ids = [str(value) for value in metadata["station_id"].tolist()]
    available = {str(value) for value in ds["station"].values.tolist()}
    missing = sorted(set(station_ids).difference(available))
    if missing:
        raise ValueError(f"{len(missing)} BTHSA stations are missing from dataset, first={missing[:5]}")

    frames: list[pd.DataFrame] = []
    for city_en in sorted(metadata["city_en"].unique()):
        city_meta = metadata[metadata["city_en"] == city_en].sort_values("station_id")
        city_ds = ds.sel(station=city_meta["station_id"].tolist()).mean("station", skipna=True)
        city_frame = city_ds.to_dataframe().reset_index()
        city_frame["city_en"] = city_en
        city_frame["city"] = str(city_meta["city"].iloc[0])
        city_frame["station_count"] = int(len(city_meta))
        frames.append(city_frame)

    city_frame = pd.concat(frames, ignore_index=True)
    city_frame = add_derived_features(city_frame)
    city_frame = add_time_features(city_frame)
    city_frame = assign_time_split(city_frame)
    city_frame = city_frame.replace([np.inf, -np.inf], np.nan)
    columns = ["time", "city_en", "city", "station_count", TARGET_COLUMN, *FEATURE_COLUMNS, "split"]
    city_frame = city_frame[columns].sort_values(["city_en", "time"]).reset_index(drop=True)
    return city_frame


def drop_model_missing_rows(frame: pd.DataFrame) -> pd.DataFrame:
    needed = [TARGET_COLUMN, *FEATURE_COLUMNS, "split"]
    return frame.dropna(subset=needed).reset_index(drop=True)


def apply_smoke_subset(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    if not config.smoke:
        return frame
    selected = frame[frame["city_en"].isin(config.smoke_cities)].copy()
    parts = []
    for _, city_frame in selected.groupby("city_en", sort=True):
        city_frame = city_frame.sort_values("time").reset_index(drop=True)
        if len(city_frame) > config.smoke_hours_per_city:
            indices = np.linspace(0, len(city_frame) - 1, config.smoke_hours_per_city).round().astype(int)
            city_frame = city_frame.iloc[np.unique(indices)].copy()
        parts.append(city_frame)
    return pd.concat(parts, ignore_index=True) if parts else selected


def prepare_city_feature_table(config: ExperimentConfig) -> pd.DataFrame:
    ensure_output_dirs(config)
    metadata = load_bthsa_station_metadata(config.resolved_station_path)
    if config.smoke:
        metadata = metadata[metadata["city_en"].isin(config.smoke_cities)].reset_index(drop=True)
    with open_bthsa_dataset(config.resolved_dataset_path) as ds:
        city_frame = build_city_hourly_features(ds, metadata)
    city_frame = apply_smoke_subset(city_frame, config)
    city_frame = drop_model_missing_rows(city_frame)
    city_frame.to_csv(config.resolved_cache_dir / "city_hourly_features.csv", index=False)
    return city_frame


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    if y_true.size > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = float("nan")
    return {"rmse": rmse, "mae": mae, "corr": corr}


def build_random_forest(config: ExperimentConfig):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=config.n_estimators,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )


def fit_city_models(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    models: dict[str, Any] = {}
    for city_en, city_frame in frame.groupby("city_en", sort=True):
        train = city_frame[city_frame["split"] == "train"]
        val = city_frame[city_frame["split"] == "val"]
        test = city_frame[city_frame["split"] == "test"]
        if train.empty or val.empty or test.empty:
            raise ValueError(f"City {city_en} does not have train/val/test rows after filtering.")

        model = build_random_forest(config)
        model.fit(train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN])
        models[str(city_en)] = model
        for split_name, split_frame in (("train", train), ("val", val), ("test", test)):
            pred = model.predict(split_frame[list(FEATURE_COLUMNS)])
            payload = _metrics(split_frame[TARGET_COLUMN].to_numpy(), pred)
            payload.update({"city_en": city_en, "split": split_name, "n_samples": int(len(split_frame))})
            rows.append(payload)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(config.resolved_cache_dir / "model_metrics.csv", index=False)
    return models, metrics


def refit_city_models_on_all_data(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for city_en, city_frame in frame.groupby("city_en", sort=True):
        model = build_random_forest(config)
        model.fit(city_frame[list(FEATURE_COLUMNS)], city_frame[TARGET_COLUMN])
        models[str(city_en)] = model
    return models


def deweather_city_pm25(
    frame: pd.DataFrame,
    models: dict[str, Any],
    config: ExperimentConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_state)
    rows: list[pd.DataFrame] = []
    feature_columns = list(FEATURE_COLUMNS)
    meteo_columns = list(METEOROLOGY_FEATURES)

    for city_en, city_frame in frame.groupby("city_en", sort=True):
        model = models[str(city_en)]
        base = city_frame[feature_columns].copy()
        meteo_pool = city_frame[meteo_columns].to_numpy()
        pred_sum = np.zeros(len(city_frame), dtype=float)
        for _ in range(int(config.deweather_samples)):
            sampled = base.copy()
            indices = rng.integers(0, len(meteo_pool), size=len(city_frame))
            sampled.loc[:, meteo_columns] = meteo_pool[indices]
            pred_sum += model.predict(sampled)
        deweathered = pred_sum / float(config.deweather_samples)
        city_out = city_frame[["time", "city_en", "city", "split", TARGET_COLUMN]].copy()
        city_out = city_out.rename(columns={TARGET_COLUMN: "PM2.5_obs"})
        city_out["PM2.5_deweathered"] = deweathered
        city_out["meteo_component"] = city_out["PM2.5_obs"] - city_out["PM2.5_deweathered"]
        rows.append(city_out)

    result = pd.concat(rows, ignore_index=True)
    result.to_csv(config.resolved_cache_dir / "deweathered_pm25.csv", index=False)
    return result


def compute_shap_values(
    frame: pd.DataFrame,
    models: dict[str, Any],
    config: ExperimentConfig,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    shap = require_shap()
    rng = np.random.default_rng(config.random_state)
    shap_payload: dict[str, np.ndarray] = {"feature_names": np.asarray(FEATURE_COLUMNS, dtype=object)}
    importance_rows: list[dict[str, object]] = []

    for city_en, city_frame in frame.groupby("city_en", sort=True):
        sample_size = min(int(config.shap_sample_per_city), len(city_frame))
        sample_indices = np.sort(rng.choice(len(city_frame), size=sample_size, replace=False))
        sample = city_frame.iloc[sample_indices].copy()
        x_sample = sample[list(FEATURE_COLUMNS)]
        explainer = shap.TreeExplainer(models[str(city_en)])
        values = np.asarray(explainer.shap_values(x_sample), dtype=np.float32)
        shap_payload[f"{city_en}__values"] = values
        shap_payload[f"{city_en}__features"] = x_sample.to_numpy(dtype=np.float32)
        shap_payload[f"{city_en}__times"] = sample["time"].astype(str).to_numpy(dtype=object)

        mean_abs = np.abs(values).mean(axis=0)
        mean_signed = values.mean(axis=0)
        for feature, abs_value, signed_value in zip(FEATURE_COLUMNS, mean_abs, mean_signed):
            importance_rows.append(
                {
                    "city_en": city_en,
                    "feature": feature,
                    "mean_abs_shap": float(abs_value),
                    "mean_shap": float(signed_value),
                    "n_samples": int(sample_size),
                }
            )

    np.savez(config.resolved_cache_dir / "shap_values_sampled.npz", **shap_payload)
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(config.resolved_cache_dir / "shap_global_importance.csv", index=False)
    return shap_payload, importance


def save_model_performance_plot(metrics: pd.DataFrame, config: ExperimentConfig) -> Path:
    import matplotlib.pyplot as plt

    test = metrics[metrics["split"] == "test"].sort_values("rmse")
    fig, ax = plt.subplots(figsize=(10.8, 6.4), constrained_layout=True)
    ax.barh(test["city_en"], test["rmse"], color="#4C78A8")
    ax.set_xlabel("Test RMSE (ug/m3)")
    ax.set_ylabel("City")
    path = config.resolved_fig_dir / "model_performance_rmse.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_deweathered_timeseries_plot(deweathered: pd.DataFrame, config: ExperimentConfig) -> Path:
    import matplotlib.pyplot as plt

    frame = deweathered.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    regional = frame.groupby("time", as_index=False)[["PM2.5_obs", "PM2.5_deweathered"]].mean()
    fig, ax = plt.subplots(figsize=(12.8, 4.8), constrained_layout=True)
    ax.plot(regional["time"], regional["PM2.5_obs"], linewidth=0.8, label="Observed PM2.5", color="#4C78A8")
    ax.plot(
        regional["time"],
        regional["PM2.5_deweathered"],
        linewidth=0.8,
        label="Deweathered PM2.5",
        color="#F58518",
    )
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    path = config.resolved_fig_dir / "regional_pm25_deweathered_timeseries.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_shap_bar_plot(importance: pd.DataFrame, config: ExperimentConfig, *, top_n: int = 18) -> Path:
    import matplotlib.pyplot as plt

    global_importance = (
        importance.groupby("feature", as_index=False)["mean_abs_shap"]
        .mean()
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .sort_values("mean_abs_shap")
    )
    fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
    ax.barh(global_importance["feature"], global_importance["mean_abs_shap"], color="#2F7D63")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("Feature")
    path = config.resolved_fig_dir / "shap_global_top_features.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_city_feature_heatmap(importance: pd.DataFrame, config: ExperimentConfig, *, top_n: int = 18) -> Path:
    import matplotlib.pyplot as plt

    top_features = (
        importance.groupby("feature")["mean_abs_shap"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    matrix = (
        importance[importance["feature"].isin(top_features)]
        .pivot(index="city_en", columns="feature", values="mean_abs_shap")
        .reindex(columns=top_features)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(12.0, 8.0), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.set_xlabel("Feature")
    ax.set_ylabel("City")
    fig.colorbar(image, ax=ax, label="Mean |SHAP|")
    path = config.resolved_fig_dir / "city_feature_shap_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_dependence_plots(
    shap_payload: dict[str, np.ndarray],
    config: ExperimentConfig,
    *,
    features: tuple[str, ...] = ("RH", "temp_c", "blh", "WS", "msdwswrf", "O3", "meic_NOx"),
) -> list[Path]:
    import matplotlib.pyplot as plt

    feature_names = list(shap_payload["feature_names"])
    paths: list[Path] = []
    for feature in features:
        if feature not in feature_names:
            continue
        feature_index = feature_names.index(feature)
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for key, values in shap_payload.items():
            if not key.endswith("__values"):
                continue
            city = key.removesuffix("__values")
            feature_matrix = shap_payload[f"{city}__features"]
            xs.append(feature_matrix[:, feature_index])
            ys.append(values[:, feature_index])
        if not xs:
            continue
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        if len(x) > 12000:
            rng = np.random.default_rng(config.random_state)
            idx = rng.choice(len(x), size=12000, replace=False)
            x = x[idx]
            y = y[idx]
        fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
        ax.scatter(x, y, s=7, alpha=0.25, color="#4C78A8", edgecolors="none")
        ax.axhline(0.0, color="#666666", linewidth=0.8)
        ax.set_xlabel(feature)
        ax.set_ylabel("SHAP value")
        path = config.resolved_fig_dir / f"shap_dependence_{feature}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def save_all_plots(
    *,
    metrics: pd.DataFrame,
    deweathered: pd.DataFrame,
    importance: pd.DataFrame,
    shap_payload: dict[str, np.ndarray],
    config: ExperimentConfig,
) -> list[Path]:
    paths = [
        save_model_performance_plot(metrics, config),
        save_deweathered_timeseries_plot(deweathered, config),
        save_shap_bar_plot(importance, config),
        save_city_feature_heatmap(importance, config),
    ]
    paths.extend(save_dependence_plots(shap_payload, config))
    return paths


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    ensure_output_dirs(config)
    features = prepare_city_feature_table(config)
    eval_models, metrics = fit_city_models(features, config)
    _ = eval_models
    final_models = refit_city_models_on_all_data(features, config)
    deweathered = deweather_city_pm25(features, final_models, config)
    shap_payload, importance = compute_shap_values(features, final_models, config)
    plot_paths = save_all_plots(
        metrics=metrics,
        deweathered=deweathered,
        importance=importance,
        shap_payload=shap_payload,
        config=config,
    )
    return {
        "features": features,
        "metrics": metrics,
        "models": final_models,
        "deweathered": deweathered,
        "shap_payload": shap_payload,
        "importance": importance,
        "plot_paths": plot_paths,
    }


def build_config(*, root_dir: Path = Path("."), smoke: bool = False) -> ExperimentConfig:
    if smoke:
        return ExperimentConfig(
            root_dir=root_dir,
            smoke=True,
            n_estimators=20,
            min_samples_leaf=3,
            shap_sample_per_city=200,
            deweather_samples=20,
        )
    return ExperimentConfig(root_dir=root_dir)
