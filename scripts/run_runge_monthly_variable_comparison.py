#!/usr/bin/env python3
"""Run monthly Runge-style MLP-TM-EI gateway maps across climate variables."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reproduce_runge2015_gateways import (
    dependency_versions,
    latitude_area_weights,
    rotated_component_order,
    varimax,
)


RESULT_SUBDIR = Path("results/runge_monthly_variable_comparison")
FIG_SUBDIR = Path("fig/runge_monthly_variable_comparison")


@dataclass(frozen=True)
class MonthlyDatasetConfig:
    name: str
    path: Path
    variable: str
    level: float | None = None
    display_name: str | None = None


@dataclass(frozen=True)
class DatasetArtifacts:
    dataset_dir: Path
    component_scores: Path
    component_maps: Path
    manifest: Path
    pairwise_output_dir: Path
    pairwise_result_dir: Path


DATASETS: dict[str, MonthlyDatasetConfig] = {
    "slp_monthly": MonthlyDatasetConfig(
        name="slp_monthly",
        path=Path("data/ncep_reanalysis_slp/monthly/slp.mon.mean.nc"),
        variable="slp",
        display_name="SLP",
    ),
    "t2m_monthly": MonthlyDatasetConfig(
        name="t2m_monthly",
        path=Path("data/ncep_reanalysis_runge_validation/air.2m.mon.mean.nc"),
        variable="air",
        display_name="2m air temperature",
    ),
    "air1000_monthly": MonthlyDatasetConfig(
        name="air1000_monthly",
        path=Path("data/ncep_reanalysis_runge_validation/air.1000hPa.mon.mean.nc"),
        variable="air",
        level=1000.0,
        display_name="1000hPa air temperature",
    ),
    "sst_monthly": MonthlyDatasetConfig(
        name="sst_monthly",
        path=Path("data/noaa_ersst_v5/sst.mnmean.1948_2026.nc"),
        variable="sst",
        display_name="SST",
    ),
}


def _repo_root() -> Path:
    return ROOT


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    root_candidate = _repo_root() / candidate
    return root_candidate if root_candidate.exists() else candidate.resolve()


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def parse_dataset_names(text: str) -> list[str]:
    names = [part.strip() for part in str(text).split(",") if part.strip()]
    if not names or names == ["all"]:
        return list(DATASETS)
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        raise ValueError(f"unknown dataset(s): {', '.join(unknown)}")
    return names


def load_monthly_field(config: MonthlyDatasetConfig) -> xr.DataArray:
    path = _resolve_path(config.path)
    with xr.open_dataset(path) as ds:
        if config.variable not in ds:
            raise ValueError(f"{path} does not contain variable {config.variable!r}.")
        field = ds[config.variable].load()
    if config.level is not None:
        if "level" not in field.dims:
            raise ValueError(f"{path} does not contain a level dimension.")
        field = field.sel(level=float(config.level)).squeeze(drop=True)
    else:
        for dim in tuple(field.dims):
            if dim not in {"time", "lat", "lon"} and int(field.sizes[dim]) == 1:
                field = field.squeeze(dim, drop=True)
    missing_dims = {"time", "lat", "lon"} - set(field.dims)
    if missing_dims:
        raise ValueError(f"{path} is missing required dimensions: {sorted(missing_dims)}")
    field = field.transpose("time", "lat", "lon").sortby("time").sortby("lat")
    if not np.all(np.diff(pd.to_datetime(field["time"].values).astype("int64")) > 0):
        raise ValueError(f"{path} time coordinate must be strictly increasing.")
    return field


def standardize_monthly_anomalies(field: xr.DataArray) -> xr.DataArray:
    all_missing = field.isnull().all("time")
    month = field["time"].dt.month
    climatology = field.groupby(month).mean("time", skipna=True)
    anomaly = field.groupby(month) - climatology
    scale = anomaly.groupby(month).std("time", skipna=True)
    scale = scale.where(np.isfinite(scale) & (scale > 0.0), 1.0)
    standardized = anomaly.groupby(month) / scale
    standardized = standardized.where(np.isfinite(standardized), 0.0)
    return standardized.where(~all_missing)


def detrend_valid_cells(field: xr.DataArray) -> xr.DataArray:
    values = np.asarray(field.values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("monthly field must have shape [time, lat, lon].")
    n_time = values.shape[0]
    matrix = values.reshape(n_time, -1)
    valid = np.all(np.isfinite(matrix), axis=0)
    detrended = np.full_like(matrix, np.nan, dtype=np.float64)
    if np.any(valid):
        x = np.arange(n_time, dtype=np.float64)
        x_centered = x - x.mean()
        denom = float(np.sum(x_centered**2))
        valid_values = matrix[:, valid]
        slopes = (x_centered[:, None] * valid_values).sum(axis=0) / denom if denom > 0.0 else np.zeros(valid_values.shape[1])
        intercepts = valid_values.mean(axis=0) - slopes * x.mean()
        trend = intercepts[None, :] + slopes[None, :] * x[:, None]
        detrended[:, valid] = valid_values - trend
    return xr.DataArray(
        detrended.reshape(values.shape),
        coords=field.coords,
        dims=field.dims,
        attrs=field.attrs,
        name=field.name,
    )


def fit_valid_cell_varimax_components(
    field: xr.DataArray,
    *,
    n_components: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(field.values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("monthly field must have shape [time, lat, lon].")
    n_time, n_lat, n_lon = values.shape
    matrix_raw = values.reshape(n_time, n_lat * n_lon)
    valid = np.all(np.isfinite(matrix_raw), axis=0)
    valid_count = int(np.sum(valid))
    if valid_count < 2:
        raise ValueError("monthly field has fewer than two valid spatial cells.")
    if n_components < 1 or n_components > min(n_time, valid_count):
        raise ValueError("n_components must be between 1 and min(time, valid_cell_count).")

    weights_2d = latitude_area_weights(field["lat"].values)[:, None] * np.ones((n_lat, n_lon), dtype=np.float64)
    weighted = values * weights_2d[None, :, :]
    matrix = weighted.reshape(n_time, n_lat * n_lon)[:, valid]
    matrix = matrix - matrix.mean(axis=0, keepdims=True)

    pca = PCA(n_components=int(n_components), svd_solver="full", random_state=int(seed))
    pca_scores = pca.fit_transform(matrix)
    loadings = pca.components_.T
    rotated_loadings, rotation = varimax(loadings)
    order, rotated_diagonal = rotated_component_order(rotation, np.asarray(pca.explained_variance_, dtype=float))
    rotated_loadings = rotated_loadings[:, order]
    rotated_scores = (pca_scores @ rotation)[:, order]
    rotated_scores = (rotated_scores - rotated_scores.mean(axis=0, keepdims=True)) / np.maximum(
        rotated_scores.std(axis=0, ddof=1, keepdims=True), 1.0e-12
    )

    maps = np.full((n_lat * n_lon, int(n_components)), np.nan, dtype=np.float64)
    maps[valid, :] = rotated_loadings
    maps = maps.reshape(n_lat, n_lon, int(n_components))
    total_variance = float(np.sum(pca.explained_variance_))
    explained = rotated_diagonal[order] / total_variance if total_variance > 0.0 else np.zeros_like(rotated_diagonal[order])
    columns = [f"component_{index + 1:02d}" for index in range(int(n_components))]
    scores = pd.DataFrame(rotated_scores, index=pd.to_datetime(field["time"].values), columns=columns)
    scores.index.name = "time"
    return scores, maps, np.asarray(explained, dtype=float), valid.reshape(n_lat, n_lon)


def sign_normalized_map(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr
    masked = np.where(finite, np.abs(arr), -np.inf)
    idx = np.unravel_index(int(np.nanargmax(masked)), arr.shape)
    sign = 1.0 if arr[idx] >= 0.0 else -1.0
    return sign * arr


def component_center(values: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[float, float]:
    arr = sign_normalized_map(values)
    positive = np.where(np.isfinite(arr), np.maximum(arr, 0.0), 0.0)
    if float(np.nanmax(positive)) <= 0.0:
        finite = np.isfinite(arr)
        idx = np.unravel_index(int(np.nanargmax(np.where(finite, arr, -np.inf))), arr.shape)
        return float(((lon[idx[1]] + 180.0) % 360.0) - 180.0), float(lat[idx[0]])
    threshold = max(float(np.nanpercentile(positive[positive > 0.0], 97.5)), 0.58 * float(np.nanmax(positive)))
    weights = np.where(positive >= threshold, positive, 0.0)
    if float(np.sum(weights)) <= 0.0:
        idx = np.unravel_index(int(np.nanargmax(positive)), positive.shape)
        return float(((lon[idx[1]] + 180.0) % 360.0) - 180.0), float(lat[idx[0]])
    lon180 = ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0
    lon_rad = np.radians(lon180)
    lon_weights = np.sum(weights, axis=0)
    x = float(np.sum(lon_weights * np.cos(lon_rad)))
    y = float(np.sum(lon_weights * np.sin(lon_rad)))
    center_lon = float(np.degrees(np.arctan2(y, x)))
    lat_weights = np.sum(weights, axis=1)
    center_lat = float(np.sum(lat_weights * np.asarray(lat, dtype=float)) / np.sum(lat_weights))
    return center_lon, center_lat


def component_centers_from_scores(
    scores: pd.DataFrame,
    component_maps: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    metric: str,
    top_k: int,
) -> pd.DataFrame:
    if metric not in scores.columns:
        raise ValueError(f"score frame does not contain metric {metric!r}.")
    rows: list[dict[str, object]] = []
    ranked = scores.sort_values(metric, ascending=False).head(int(top_k))
    for _, row in ranked.iterrows():
        component = parse_component_index(row["component"])
        center_lon, center_lat = component_center(component_maps[..., component], lat, lon)
        rows.append(
            {
                "component": component,
                "label": f"C{component + 1}",
                "lon": center_lon,
                "lat": center_lat,
                "score": float(row[metric]),
                "metric": metric,
            }
        )
    return pd.DataFrame(rows)


def parse_component_index(value: object) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    text = str(value).strip()
    lower = text.lower()
    for prefix in ("component_", "component", "c"):
        if lower.startswith(prefix):
            suffix = lower[len(prefix) :].strip()
            if suffix.isdigit():
                return int(suffix) - 1
    return int(text)


def preprocess_dataset(
    config: MonthlyDatasetConfig,
    *,
    output_dir: Path,
    n_components: int,
    seed: int,
    force: bool,
) -> DatasetArtifacts:
    dataset_dir = Path(output_dir) / config.name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    scores_path = dataset_dir / "component_monthly_scores.csv"
    maps_path = dataset_dir / "component_maps.npz"
    manifest_path = dataset_dir / "manifest.json"
    pairwise_output_dir = dataset_dir / "mlp_tm_ei"
    pairwise_result_dir = pairwise_output_dir / "results" / "runge" / "pairwise_mlp_tm_ei_path_effects"

    if scores_path.exists() and maps_path.exists() and manifest_path.exists() and not force:
        return DatasetArtifacts(dataset_dir, scores_path, maps_path, manifest_path, pairwise_output_dir, pairwise_result_dir)

    raw = load_monthly_field(config)
    standardized = standardize_monthly_anomalies(raw)
    detrended = detrend_valid_cells(standardized)
    scores, maps, explained, valid_mask = fit_valid_cell_varimax_components(
        detrended,
        n_components=int(n_components),
        seed=int(seed),
    )
    scores.to_csv(scores_path, index_label="time")
    np.savez_compressed(
        maps_path,
        component_maps=maps,
        explained_variance_ratio=explained,
        lat=np.asarray(detrended["lat"].values, dtype=float),
        lon=np.asarray(detrended["lon"].values, dtype=float),
        valid_mask=valid_mask,
    )
    manifest = {
        "dataset": asdict(config),
        "input_path": str(_resolve_path(config.path)),
        "preprocessing": {
            "monthly_deseasonalized": True,
            "monthly_standardized": True,
            "linear_detrended": True,
            "valid_cell_count": int(np.sum(valid_mask)),
            "total_cell_count": int(valid_mask.size),
        },
        "n_monthly_samples": int(scores.shape[0]),
        "time_start": str(scores.index.min().date()),
        "time_end": str(scores.index.max().date()),
        "n_components": int(n_components),
        "dependency_versions": dependency_versions(),
    }
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    _write_dataset_summary(dataset_dir / "summary.md", manifest)
    return DatasetArtifacts(dataset_dir, scores_path, maps_path, manifest_path, pairwise_output_dir, pairwise_result_dir)


def _write_dataset_summary(path: Path, manifest: dict[str, object]) -> None:
    dataset = manifest["dataset"]
    assert isinstance(dataset, dict)
    preprocessing = manifest["preprocessing"]
    assert isinstance(preprocessing, dict)
    lines = [
        f"# {dataset.get('display_name') or dataset['name']} Monthly Runge Components",
        "",
        "This preprocessing uses calendar-month deseasonalization, monthly anomaly standardization, linear detrending, and valid-cell latitude-weighted Varimax PCA.",
        "",
        f"- Input: `{manifest['input_path']}`",
        f"- Variable: `{dataset['variable']}`",
        f"- Level: `{dataset.get('level')}`",
        f"- Time range: `{manifest['time_start']}` to `{manifest['time_end']}`",
        f"- Monthly samples: `{manifest['n_monthly_samples']}`",
        f"- Components: `{manifest['n_components']}`",
        f"- Valid cells: `{preprocessing['valid_cell_count']}` / `{preprocessing['total_cell_count']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pairwise_mlp_tm_ei(artifacts: DatasetArtifacts, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_runge_pairwise_mlp_ei.py"),
        "--component-scores",
        str(artifacts.component_scores),
        "--output-dir",
        str(artifacts.pairwise_output_dir),
        "--lag",
        str(args.lag),
        "--horizon",
        str(args.horizon),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--batch-size",
        str(args.batch_size),
        "--weight-decay",
        str(args.weight_decay),
        "--ridge-alpha",
        str(args.ridge_alpha),
        "--ensemble-ridge-alphas",
        str(args.ensemble_ridge_alphas),
        "--linear-blend-grid-steps",
        str(args.linear_blend_grid_steps),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--scheduler-patience",
        str(args.scheduler_patience),
        "--gradient-clip-norm",
        str(args.gradient_clip_norm),
        "--intervention-samples",
        str(args.intervention_samples),
        "--ei-estimator",
        "tm",
        "--gateway-mode",
        "path_effect",
        "--source-mode",
        "latest",
        "--seed",
        str(args.seed),
    ]
    if bool(args.force_retrain):
        command.append("--force-retrain")
    subprocess.run(command, cwd=ROOT, check=True)


def write_comparison_summary(path: Path, artifacts: Sequence[DatasetArtifacts], args: argparse.Namespace) -> None:
    rows = []
    for artifact in artifacts:
        pair_manifest_path = artifact.pairwise_result_dir / "manifest.json"
        pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8")) if pair_manifest_path.exists() else {}
        rows.append(
            {
                "dataset": artifact.dataset_dir.name,
                "component_scores": str(artifact.component_scores),
                "component_maps": str(artifact.component_maps),
                "pairwise_result_dir": str(artifact.pairwise_result_dir),
                "n_components": pair_manifest.get("n_components"),
                "n_lagged_samples": pair_manifest.get("n_lagged_samples"),
                "top_gateway": (pair_manifest.get("top_gateways") or [{}])[0].get("component") if pair_manifest.get("top_gateways") else None,
                "top_mediator": (pair_manifest.get("top_mediators") or [{}])[0].get("component") if pair_manifest.get("top_mediators") else None,
            }
        )
    frame = pd.DataFrame(rows)
    lines = [
        "# Monthly Runge MLP-TM-EI Variable Comparison",
        "",
        "All datasets are processed with calendar-month deseasonalization before detrending and component extraction.",
        "",
        f"- Horizon: `{args.horizon}` month",
        f"- Lag: `{args.lag}` months",
        f"- Components: `{args.n_components}`",
        "",
        frame.to_markdown(index=False),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--output-dir", type=Path, default=RESULT_SUBDIR)
    parser.add_argument("--fig-dir", type=Path, default=FIG_SUBDIR)
    parser.add_argument("--n-components", type=int, default=60)
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--ridge-alpha", type=float, default=1000.0)
    parser.add_argument("--ensemble-ridge-alphas", default="10,100,1000,3000")
    parser.add_argument("--linear-blend-grid-steps", type=int, default=101)
    parser.add_argument("--early-stopping-patience", type=int, default=80)
    parser.add_argument("--scheduler-patience", type=int, default=20)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--intervention-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--causal-backend", choices=["regression", "tigramite"], default="regression", help="Accepted for smoke-command compatibility; MLP-TM-EI does not use this option.")
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--skip-figure", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--save-svg", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dataset_names = parse_dataset_names(args.datasets)
    output_dir = Path(args.output_dir)
    artifacts: list[DatasetArtifacts] = []
    for name in dataset_names:
        artifact = preprocess_dataset(
            DATASETS[name],
            output_dir=output_dir,
            n_components=int(args.n_components),
            seed=int(args.seed),
            force=bool(args.force_preprocess),
        )
        artifacts.append(artifact)
        if not args.skip_mlp:
            run_pairwise_mlp_tm_ei(artifact, args)

    if not args.skip_figure and not args.skip_mlp:
        from scripts.plot_runge_monthly_variable_gateway_map import plot_monthly_variable_gateway_map

        plot_monthly_variable_gateway_map(
            output_dir,
            Path(args.fig_dir) / "gateway_mediator_centers.png",
            dataset_names=dataset_names,
            top_k=5,
            save_svg=bool(args.save_svg),
        )
    write_comparison_summary(output_dir / "summary.md", artifacts, args)
    print(json.dumps({"datasets": dataset_names, "summary": str(output_dir / "summary.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
