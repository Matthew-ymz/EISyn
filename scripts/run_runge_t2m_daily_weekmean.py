#!/usr/bin/env python3
"""Run daily 2m-temperature components for future-week mean prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reproduce_runge2015_gateways import (
    dependency_versions,
    detrend_time_axis,
    fit_projected_varimax_components,
    save_component_map_figure,
    standardize_daily_anomalies,
)
from scripts.run_runge_pairwise_mlp_ei import (
    PairwiseMlpEiConfig,
    compute_ei_path_effects,
    estimate_pairwise_tm_ei_matrix,
    fit_ridge_linear_map,
    gateway_scores_from_matrix,
    predict_mlp,
    regression_metrics,
    sample_max_entropy_features,
    save_gateway_ranking,
    save_heatmap,
    save_mediator_ranking,
    save_path_gateway_ranking,
    split_temporal_arrays,
    train_or_load_mlp,
)


DEFAULT_DATA_DIR = Path("data/ncep_reanalysis_air/daily_2m")
RESULT_SUBDIR = Path("results/runge_t2m_daily_weekmean")
FIG_SUBDIR = Path("fig/runge_t2m_daily_weekmean")


@dataclass(frozen=True)
class DailyT2MConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    variable: str = "air"
    output_dir: Path = Path(".")
    start_year: int = 1948
    end_year: int = 2025
    n_components: int = 60
    history_days: int = 28
    target_days: int = 7
    lead_days: int = 1
    detrend: bool = True
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    epochs: int = 80
    learning_rate: float = 1.0e-3
    batch_size: int = 256
    weight_decay: float = 1.0e-3
    ridge_alpha: float = 1000.0
    early_stopping_patience: int = 20
    scheduler_patience: int = 8
    gradient_clip_norm: float = 1.0
    intervention_samples: int = 4096
    quantile_low: float = 0.05
    quantile_high: float = 0.95
    source_mode: str = "latest"
    candidate_top_sources: int = 4
    seed: int = 42
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    skip_peid: bool = False
    force_retrain: bool = False


@dataclass(frozen=True)
class FutureMeanDataset:
    features: np.ndarray
    targets: np.ndarray
    sample_times: pd.DatetimeIndex
    target_start_times: pd.DatetimeIndex
    target_end_times: pd.DatetimeIndex
    component_names: list[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    root_candidate = _repo_root() / candidate
    return root_candidate if root_candidate.exists() else candidate.resolve()


def standardize_t2m_daily_anomalies(field: xr.DataArray) -> xr.DataArray:
    """Return standardized daily T2M anomalies on a 365-day calendar."""

    standardized = standardize_daily_anomalies(field)
    return standardized.fillna(0.0)


def load_daily_t2m(config: DailyT2MConfig) -> xr.DataArray:
    data_dir = _resolve_path(config.data_dir)
    paths = [data_dir / f"air.2m.gauss.{year}.nc" for year in range(int(config.start_year), int(config.end_year) + 1)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(f"Missing NCEP daily 2m temperature file(s): {preview}")

    arrays: list[xr.DataArray] = []
    for path in paths:
        with xr.open_dataset(path) as ds:
            if config.variable not in ds:
                raise ValueError(f"{path} does not contain variable {config.variable!r}.")
            arrays.append(ds[config.variable].load())
    field = xr.concat(arrays, dim="time").sortby("time")
    if "time" not in field.dims or "lat" not in field.dims or "lon" not in field.dims:
        raise ValueError("input variable must have time, lat, and lon dimensions.")
    return field.sel(time=slice(f"{int(config.start_year)}-01-01", f"{int(config.end_year)}-12-31"))


def build_future_mean_dataset(
    frame: pd.DataFrame,
    *,
    history_days: int,
    target_days: int,
    lead_days: int,
) -> FutureMeanDataset:
    if int(history_days) < 1:
        raise ValueError("history_days must be positive.")
    if int(target_days) < 1:
        raise ValueError("target_days must be positive.")
    if int(lead_days) < 1:
        raise ValueError("lead_days must be positive.")
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()
    values = frame.to_numpy(dtype=float)
    n_time, n_components = values.shape
    n_samples = n_time - int(history_days) - int(lead_days) - int(target_days) + 2
    if n_samples <= 0:
        raise ValueError("not enough rows for requested history, lead, and target windows.")

    features = np.empty((n_samples, int(history_days) * n_components), dtype=float)
    targets = np.empty((n_samples, n_components), dtype=float)
    sample_times = []
    target_start_times = []
    target_end_times = []
    index = pd.DatetimeIndex(frame.index)
    for sample_idx in range(n_samples):
        history_start = sample_idx
        history_end = sample_idx + int(history_days)
        target_start = history_end - 1 + int(lead_days)
        target_end = target_start + int(target_days)
        features[sample_idx] = values[history_start:history_end].reshape(-1)
        targets[sample_idx] = values[target_start:target_end].mean(axis=0)
        sample_times.append(index[history_end - 1])
        target_start_times.append(index[target_start])
        target_end_times.append(index[target_end - 1])

    return FutureMeanDataset(
        features=features,
        targets=targets,
        sample_times=pd.DatetimeIndex(sample_times),
        target_start_times=pd.DatetimeIndex(target_start_times),
        target_end_times=pd.DatetimeIndex(target_end_times),
        component_names=list(frame.columns),
    )


def build_feature_diagnostics(dataset: FutureMeanDataset, history_days: int) -> pd.DataFrame:
    n_components = len(dataset.component_names)
    features = dataset.features.reshape(len(dataset.features), int(history_days), n_components)
    rows = {
        "sample_time": dataset.sample_times,
        "target_start_time": dataset.target_start_times,
        "target_end_time": dataset.target_end_times,
    }
    x = np.arange(int(history_days), dtype=float)
    x_centered = x - x.mean()
    denom = float(np.sum(x_centered**2))
    for comp_idx, name in enumerate(dataset.component_names):
        history = features[:, :, comp_idx]
        rows[f"{name}_last7_mean"] = history[:, -min(7, int(history_days)) :].mean(axis=1)
        start_8_14 = max(0, int(history_days) - 14)
        end_8_14 = max(0, int(history_days) - 7)
        rows[f"{name}_days8_14_mean"] = history[:, start_8_14:end_8_14].mean(axis=1)
        rows[f"{name}_days15_28_mean"] = history[:, : max(1, int(history_days) - 14)].mean(axis=1)
        if denom > 0.0:
            rows[f"{name}_history_trend"] = (history * x_centered[None, :]).sum(axis=1) / denom
        else:
            rows[f"{name}_history_trend"] = np.zeros(len(history), dtype=float)
    return pd.DataFrame(rows)


def _target_frame(dataset: FutureMeanDataset) -> pd.DataFrame:
    frame = pd.DataFrame(dataset.targets, columns=dataset.component_names)
    frame.insert(0, "target_end_time", dataset.target_end_times)
    frame.insert(0, "target_start_time", dataset.target_start_times)
    frame.insert(0, "sample_time", dataset.sample_times)
    return frame


def _prediction_metrics(
    model_name: str,
    split_name: str,
    prediction: np.ndarray,
    target: np.ndarray,
    names: Sequence[str],
) -> pd.DataFrame:
    frame = regression_metrics(prediction, target, names)
    frame.insert(0, "split", split_name)
    frame.insert(0, "model", model_name)
    return frame


def _linear_predict(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) @ np.asarray(weight, dtype=float).T + np.asarray(bias, dtype=float)


def estimate_second_order_peid_candidates(
    model: object,
    scalers: dict[str, np.ndarray],
    train_features: np.ndarray,
    pairwise_matrix: np.ndarray,
    names: Sequence[str],
    *,
    history_days: int,
    intervention_samples: int,
    low_q: float,
    high_q: float,
    source_mode: str,
    candidate_top_sources: int,
    seed: int,
) -> pd.DataFrame:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from exp.TM.transport_map_density import estimate_mutual_information_transport_map

    n_components = len(names)
    intervention_features = sample_max_entropy_features(
        train_features,
        n_components=n_components,
        lag=int(history_days),
        samples=int(intervention_samples),
        low_q=float(low_q),
        high_q=float(high_q),
        seed=int(seed) + 2003,
    )
    predictions = predict_mlp(model, scalers, intervention_features)
    rows = []
    max_sources = max(2, min(int(candidate_top_sources), n_components))
    for target in range(n_components):
        source_scores = np.asarray(pairwise_matrix[:, target], dtype=float).copy()
        source_scores[target] = -np.inf
        candidate_sources = [int(idx) for idx in np.argsort(source_scores)[::-1][:max_sources] if np.isfinite(source_scores[idx])]
        for left_pos, source_a in enumerate(candidate_sources):
            for source_b in candidate_sources[left_pos + 1 :]:
                cols = []
                for source in (source_a, source_b):
                    lagged_cols = [lag_idx * n_components + source for lag_idx in range(int(history_days))]
                    if source_mode == "latest":
                        cols.append(lagged_cols[-1])
                    else:
                        cols.extend(lagged_cols)
                source_state = intervention_features[:, cols]
                target_state = predictions[:, [target]]
                summary = estimate_mutual_information_transport_map(source_state, target_state)
                joint_mi = max(0.0, float(summary["mi_hat"]))
                best_single = max(float(pairwise_matrix[source_a, target]), float(pairwise_matrix[source_b, target]))
                rows.append(
                    {
                        "order": 2,
                        "subset_str": f"{names[source_a]}+{names[source_b]}",
                        "source_a": names[source_a],
                        "source_b": names[source_b],
                        "target": names[target],
                        "source_a_index": source_a,
                        "source_b_index": source_b,
                        "target_index": target,
                        "joint_mi": joint_mi,
                        "best_single_mi": best_single,
                        "delta_K": float(joint_mi - best_single),
                        "bias_correction": float(summary["bias_correction"]),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("delta_K", key=lambda series: series.abs(), ascending=False).reset_index(drop=True)


def _config_hash(config: DailyT2MConfig, dataset: FutureMeanDataset) -> str:
    payload = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "n_samples": int(len(dataset.features)),
        "n_components": int(len(dataset.component_names)),
        "target_start": str(dataset.target_start_times[0].date()),
        "target_end": str(dataset.target_end_times[-1].date()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def run(config: DailyT2MConfig) -> dict[str, Path]:
    result_dir = Path(config.output_dir) / RESULT_SUBDIR
    fig_dir = Path(config.output_dir) / FIG_SUBDIR
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    field = load_daily_t2m(config)
    standardized = standardize_t2m_daily_anomalies(field)
    component_field = detrend_time_axis(standardized) if bool(config.detrend) else standardized
    scores, component_maps, explained = fit_projected_varimax_components(
        component_field,
        component_field,
        n_components=int(config.n_components),
        seed=int(config.seed),
    )
    scores.index.name = "time"
    scores_path = result_dir / "component_daily_scores.csv"
    scores.to_csv(scores_path, index_label="time")
    np.savez_compressed(
        result_dir / "component_maps.npz",
        component_maps=component_maps,
        explained_variance_ratio=explained,
        lat=np.asarray(field["lat"].values),
        lon=np.asarray(field["lon"].values),
    )
    save_component_map_figure(component_maps, fig_dir / "component_maps.png")

    dataset = build_future_mean_dataset(
        scores,
        history_days=int(config.history_days),
        target_days=int(config.target_days),
        lead_days=int(config.lead_days),
    )
    target_path = result_dir / "future_7d_mean_component_targets.csv"
    _target_frame(dataset).to_csv(target_path, index=False)
    diagnostics_path = result_dir / "feature_diagnostics.csv"
    build_feature_diagnostics(dataset, int(config.history_days)).to_csv(diagnostics_path, index=False)

    splits = split_temporal_arrays(
        dataset.features,
        dataset.targets,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
    )
    pairwise_config = PairwiseMlpEiConfig(
        output_dir=Path(config.output_dir),
        lag=int(config.history_days),
        horizon=1,
        hidden_dim=int(config.hidden_dim),
        num_layers=int(config.num_layers),
        dropout=float(config.dropout),
        epochs=int(config.epochs),
        learning_rate=float(config.learning_rate),
        batch_size=int(config.batch_size),
        weight_decay=float(config.weight_decay),
        ridge_alpha=float(config.ridge_alpha),
        early_stopping_patience=int(config.early_stopping_patience),
        scheduler_patience=int(config.scheduler_patience),
        gradient_clip_norm=float(config.gradient_clip_norm),
        intervention_samples=int(config.intervention_samples),
        ei_estimator="tm",
        gateway_mode="path_effect",
        quantile_low=float(config.quantile_low),
        quantile_high=float(config.quantile_high),
        source_mode=str(config.source_mode),
        seed=int(config.seed),
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        force_retrain=bool(config.force_retrain),
    )
    model_path = result_dir / "mlp_future_week_transition.pt"
    model, scalers, loss_history, reused = train_or_load_mlp(
        splits,
        pairwise_config,
        model_path,
        config_hash=_config_hash(config, dataset),
    )

    metric_frames = []
    linear_weight, linear_bias = fit_ridge_linear_map(splits["train"][0], splits["train"][1], alpha=float(config.ridge_alpha))
    for split_name, (x_split, y_split) in splits.items():
        metric_frames.append(
            _prediction_metrics(
                "mlp",
                split_name,
                predict_mlp(model, scalers, x_split),
                y_split,
                dataset.component_names,
            )
        )
        metric_frames.append(
            _prediction_metrics(
                "ridge",
                split_name,
                _linear_predict(x_split, linear_weight, linear_bias),
                y_split,
                dataset.component_names,
            )
        )
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics_path = result_dir / "mlp_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame({"epoch": np.arange(1, len(loss_history) + 1), "train_loss": loss_history}).to_csv(
        result_dir / "mlp_training_history.csv",
        index=False,
    )

    train_features = splits["train"][0]
    matrix, edges = estimate_pairwise_tm_ei_matrix(
        model,
        scalers,
        train_features,
        n_components=len(dataset.component_names),
        lag=int(config.history_days),
        intervention_samples=int(config.intervention_samples),
        low_q=float(config.quantile_low),
        high_q=float(config.quantile_high),
        source_mode=str(config.source_mode),
        seed=int(config.seed),
    )
    pairwise_dir = result_dir / "pairwise_tm_ei"
    pairwise_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=dataset.component_names, columns=dataset.component_names).to_csv(pairwise_dir / "pairwise_ei_matrix.csv")
    edges.to_csv(pairwise_dir / "pairwise_ei_edges.csv", index=False)
    gateway = gateway_scores_from_matrix(matrix, dataset.component_names)
    gateway.to_csv(pairwise_dir / "gateway_scores.csv", index=False)
    save_heatmap(matrix, fig_dir / "pairwise_tm_ei_heatmap.png", title="Daily T2M future-week pairwise TM-EI")
    save_gateway_ranking(gateway, fig_dir / "pairwise_tm_ei_gateway_ranking.png", title="Daily T2M future-week gateway ranking")

    path_effects = compute_ei_path_effects(matrix, dataset.component_names)
    path_dir = result_dir / "pairwise_tm_ei_path_effects"
    path_dir.mkdir(parents=True, exist_ok=True)
    path_effects.direct_effects.to_csv(path_dir / "direct_effects.csv", index=False)
    path_effects.total_effects.to_csv(path_dir / "total_effects.csv", index=False)
    path_effects.path_effects.to_csv(path_dir / "mediated_path_effects.csv", index=False)
    path_effects.gateway_scores.to_csv(path_dir / "gateway_scores.csv", index=False)
    path_effects.mediator_scores.to_csv(path_dir / "mediator_scores.csv", index=False)
    pd.DataFrame(path_effects.scaled_direct_matrix, index=dataset.component_names, columns=dataset.component_names).to_csv(
        path_dir / "scaled_direct_effect_matrix.csv"
    )
    pd.DataFrame(path_effects.total_matrix, index=dataset.component_names, columns=dataset.component_names).to_csv(path_dir / "total_effect_matrix.csv")
    save_path_gateway_ranking(path_effects.gateway_scores, fig_dir / "path_effect_gateway_ranking.png", title="Daily T2M future-week path gateways")
    save_mediator_ranking(path_effects.mediator_scores, fig_dir / "path_effect_mediator_ranking.png", title="Daily T2M future-week mediators")

    peid_status = "skipped"
    if not bool(config.skip_peid):
        peid_dir = result_dir / "peid_hypergraph"
        peid_dir.mkdir(parents=True, exist_ok=True)
        peid_edges = estimate_second_order_peid_candidates(
            model,
            scalers,
            train_features,
            matrix,
            dataset.component_names,
            history_days=int(config.history_days),
            intervention_samples=int(config.intervention_samples),
            low_q=float(config.quantile_low),
            high_q=float(config.quantile_high),
            source_mode=str(config.source_mode),
            candidate_top_sources=int(config.candidate_top_sources),
            seed=int(config.seed),
        )
        peid_edges.to_csv(peid_dir / "peid_hyperedges.csv", index=False)
        peid_status = "candidate_second_order_tm_mi"

    manifest = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "input_data_dir": str(_resolve_path(config.data_dir)),
        "variable": config.variable,
        "frequency": "daily_to_future_7d_mean",
        "time_start": str(pd.to_datetime(field["time"].values[0]).date()),
        "time_end": str(pd.to_datetime(field["time"].values[-1]).date()),
        "n_daily_samples": int(scores.shape[0]),
        "n_supervised_samples": int(len(dataset.features)),
        "n_components": int(config.n_components),
        "history_days": int(config.history_days),
        "target_days": int(config.target_days),
        "lead_days": int(config.lead_days),
        "model_reused": bool(reused),
        "peid_status": peid_status,
        "dependency_versions": dependency_versions(),
    }
    manifest_path = result_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    summary = [
        "# Daily 2m Temperature Future-Week Mean",
        "",
        "This experiment predicts future 7-day mean standardized 2m-temperature component anomalies from recent daily component history.",
        "",
        f"- Frequency: `{manifest['frequency']}`",
        f"- Time range: `{manifest['time_start']}` to `{manifest['time_end']}`",
        f"- Components: `{manifest['n_components']}`",
        f"- History days: `{manifest['history_days']}`",
        f"- Target days: `{manifest['target_days']}`",
        f"- Supervised samples: `{manifest['n_supervised_samples']}`",
        "",
    ]
    (result_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {
        "result_dir": result_dir,
        "fig_dir": fig_dir,
        "manifest": manifest_path,
        "component_scores": scores_path,
        "targets": target_path,
        "metrics": metrics_path,
    }


def parse_args(argv: Sequence[str] | None = None) -> DailyT2MConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--variable", default="air")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--start-year", type=int, default=1948)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--n-components", type=int, default=60)
    parser.add_argument("--history-days", type=int, default=28)
    parser.add_argument("--target-days", type=int, default=7)
    parser.add_argument("--lead-days", type=int, default=1)
    parser.add_argument("--no-detrend", dest="detrend", action="store_false")
    parser.set_defaults(detrend=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--ridge-alpha", type=float, default=1000.0)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--intervention-samples", type=int, default=4096)
    parser.add_argument("--quantile-low", type=float, default=0.05)
    parser.add_argument("--quantile-high", type=float, default=0.95)
    parser.add_argument("--source-mode", choices=["latest", "history"], default="latest")
    parser.add_argument("--candidate-top-sources", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--skip-peid", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    return DailyT2MConfig(**vars(parser.parse_args(argv)))


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
