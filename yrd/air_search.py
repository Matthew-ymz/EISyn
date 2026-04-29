from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .analysis import save_json
from .config import YRDExperimentConfig
from .coupling import (
    build_one_step_station_pollutant_feature_groups,
    build_one_step_station_source_groups,
    compute_station_level_ei_summary,
    compute_station_level_nis_summary,
    compute_station_pollutant_pair_synergy_summary,
    estimate_residual_covariance,
    jacobian_for_target_subset,
    summarize_global_station_coupling,
    summarize_global_station_pollutant_synergy,
    summarize_global_station_single_pollutant_ei,
)
from .data import build_one_step_samples, load_dataset
from .intervention_sampling import (
    collapse_support_cover_box_profile_to_global_max,
    compute_training_input_center,
    estimate_support_cover_box_profile,
    sample_uniform_box_inputs,
)
from .models import PersistenceBaseline
from .shanghai_notebook import build_self_loop_node_strengths, draw_station_causal_graph
from .train import _compute_metrics, _predict_numpy, rebuild_joint_model_from_checkpoint, set_seed, train_joint_model_with_history

_YRD_CITIES = frozenset({"shanghai", "nanjing", "hangzhou"})
_BTHSA_CITIES = frozenset({"beijing"})
DEFAULT_TM_NONNEGATIVE_VARIABLES = (
    "O3",
    "PM2.5",
    "t2m",
    "d2m",
    "sp",
    "tp",
    "blh",
    "msdwswrf",
)
DEFAULT_COARSE_SAMPLE_COUNT = 16
DEFAULT_COARSE_SAMPLE_COUNT_SMOKE = 4
DEFAULT_NEGATIVE_RATIO_THRESHOLD = 0.10


@dataclass(frozen=True)
class CityScope:
    city_en: str
    dataset_path: Path
    station_path: Path


def _to_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def resolve_city_scope(city_en: str) -> CityScope:
    city_key = str(city_en).strip().lower()
    if city_key in _YRD_CITIES:
        return CityScope(
            city_en=city_key,
            dataset_path=Path("data/dataset_yrd.nc"),
            station_path=Path("data/stations_yrd.csv"),
        )
    if city_key in _BTHSA_CITIES:
        return CityScope(
            city_en=city_key,
            dataset_path=Path("data/dataset_bthsa.nc"),
            station_path=Path("data/stations_bthsa.csv"),
        )
    supported = sorted(_YRD_CITIES | _BTHSA_CITIES)
    raise ValueError(f"Unsupported city_en={city_en!r}. Supported values: {supported}")


def build_air_search_config(
    root_dir: Path,
    *,
    city_en: str,
    horizon: int,
    test_mode: bool,
) -> YRDExperimentConfig:
    scope = resolve_city_scope(city_en)
    horizon_int = int(horizon)
    if horizon_int <= 0:
        raise ValueError("horizon must be a positive integer number of hours.")
    return replace(
        YRDExperimentConfig(
            root_dir=Path(root_dir),
            dataset_path=scope.dataset_path,
            station_path=scope.station_path,
        ),
        sample_mode="one_step",
        history_hours=1,
        horizons=(horizon_int,),
        model_name="resmlp",
        hidden_dim=16 if test_mode else 96,
        num_layers=2 if test_mode else 3,
        dropout=0.0 if test_mode else 0.05,
        norm_type="layernorm",
        activation="silu",
        learning_rate=5e-4,
        weight_decay=0.0 if test_mode else 1e-5,
        batch_size=8 if test_mode else 64,
        epochs=2 if test_mode else 12,
        max_epochs=4 if test_mode else 60,
        early_stopping_patience=2 if test_mode else 8,
        seed=0,
    )


def build_air_search_artifact_paths(
    *,
    root_dir: Path,
    city_en: str,
    horizon: int,
    run_tag: str,
    use_smoke: bool,
) -> dict[str, Path]:
    scope = resolve_city_scope(city_en)
    horizon_int = int(horizon)
    if horizon_int <= 0:
        raise ValueError("horizon must be a positive integer number of hours.")
    horizon_label = f"{horizon_int}h"
    base_root = Path(tempfile.gettempdir()) / "eisyn" if use_smoke else Path(root_dir)
    cache_dir = (
        base_root / "exp" / "cache" / "yrd_coupling" / "air_search" / scope.city_en / horizon_label / run_tag
    )
    results_dir = base_root / "fig" / "yrd_air_search" / scope.city_en / horizon_label / run_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return {
        "cache_dir": cache_dir,
        "results_dir": results_dir,
        "config": cache_dir / "config.json",
        "checkpoint": cache_dir / "joint_model_checkpoint.pt",
        "loss_history": cache_dir / "loss_history.json",
        "predictions": cache_dir / "test_predictions.npz",
        "metrics": cache_dir / "metrics_summary.json",
        "coarse_summary": cache_dir / "coarse_summary.json",
        "refine_summary": cache_dir / "refine_summary.json",
        "leaderboard_row": cache_dir / "leaderboard_row.json",
        "run_manifest": results_dir / "run_manifest.json",
        "o3_pairwise_graph": results_dir / "o3_pairwise_graph.png",
        "pm25_to_o3_pairwise_graph": results_dir / "pm25_to_o3_pairwise_graph.png",
        "o3_pm25_synergy_graph": results_dir / "o3_pm25_synergy_graph.png",
        "report_manifest": results_dir / "report_manifest.json",
    }


def inverse_transform_targets(
    array: np.ndarray,
    target_names: list[str],
    stats: dict[str, dict[str, float]],
) -> np.ndarray:
    restored = np.asarray(array, dtype=np.float32).copy()
    for index, name in enumerate(target_names):
        variable = name.split("__")[-1]
        restored[:, index] = restored[:, index] * stats[variable]["std"] + stats[variable]["mean"]
    return restored


def _single_horizon(cfg: YRDExperimentConfig) -> int:
    if len(cfg.horizons) != 1:
        raise ValueError("Air search bundle helpers currently expect exactly one forecast horizon per run.")
    return int(cfg.horizons[0])


def resolve_data_root(root_dir: Path, *, dataset_path: Path, station_path: Path) -> Path:
    start = Path(root_dir).resolve()
    for candidate in (start, *start.parents):
        if (candidate / dataset_path).exists() and (candidate / station_path).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate data root containing {dataset_path} and {station_path} starting from {start}."
    )


def _resolve_target_dim(splits: dict[str, dict[str, object]], *, horizon: int) -> int:
    for split_name in ("train", "val", "test"):
        payload = splits[split_name]
        targets = payload["targets"]
        if not isinstance(targets, dict):
            continue
        values = targets.get(horizon)
        if isinstance(values, np.ndarray) and values.ndim == 2 and values.shape[1] > 0:
            return int(values.shape[1])
    raise ValueError(f"Could not resolve target_dim for horizon={horizon}.")


def prepare_air_search_bundle(
    *,
    cfg: YRDExperimentConfig,
    city_en: str,
    run_tag: str,
    use_smoke: bool,
) -> dict[str, object]:
    horizon = _single_horizon(cfg)
    artifact_paths = build_air_search_artifact_paths(
        root_dir=cfg.root_dir,
        city_en=city_en,
        horizon=horizon,
        run_tag=run_tag,
        use_smoke=use_smoke,
    )
    data_cfg = replace(
        cfg,
        root_dir=resolve_data_root(
            cfg.root_dir,
            dataset_path=cfg.dataset_path,
            station_path=cfg.station_path,
        ),
    )
    ds, metadata = load_dataset(data_cfg, smoke=use_smoke, city_en=city_en)
    sample_bundle = build_one_step_samples(ds, metadata, cfg, smoke=use_smoke)
    splits = sample_bundle["splits"]
    stats = sample_bundle["stats"]
    target_names = sample_bundle["target_names"]
    station_ids = sample_bundle["station_ids"]
    target_width = len(cfg.target_variables)
    target_dim = _resolve_target_dim(splits, horizon=horizon)

    x_train = splits["train"]["X"]
    x_val = splits["val"]["X"]
    x_test = splits["test"]["X"]
    y_train_scaled = splits["train"]["targets"]
    y_val_scaled = splits["val"]["targets"]
    y_test_scaled = splits["test"]["targets"]
    effective_input_dim = sample_bundle["n_stations"] * sample_bundle["n_features"]
    run_context = {
        "run_tag": run_tag,
        "city_en": str(city_en),
        "data_root": str(data_cfg.root_dir),
        "test_mode": bool(use_smoke),
        "use_smoke": bool(use_smoke),
        "sample_mode": cfg.sample_mode,
        "data_resolution_hours": 1,
        "history_hours": cfg.history_hours,
        "horizons": list(cfg.horizons),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "hidden_dim": cfg.hidden_dim,
        "model_name": cfg.model_name,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "norm_type": cfg.norm_type,
        "activation": cfg.activation,
        "station_count": len(station_ids),
        "input_shape": [sample_bundle["n_stations"], sample_bundle["n_features"]],
        "effective_input_dim": int(effective_input_dim),
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "test_samples": int(x_test.shape[0]),
    }
    save_json(artifact_paths["config"], _to_jsonable(run_context))
    return {
        "cfg": cfg,
        "artifact_paths": artifact_paths,
        "sample_bundle": sample_bundle,
        "city_metadata": metadata,
        "splits": splits,
        "stats": stats,
        "target_names": target_names,
        "station_ids": station_ids,
        "target_width": target_width,
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train_scaled": y_train_scaled,
        "y_val_scaled": y_val_scaled,
        "y_test_scaled": y_test_scaled,
        "target_dim": target_dim,
        "effective_input_dim": int(effective_input_dim),
        "run_context": run_context,
    }


def _build_metrics_payload(
    *,
    cfg: YRDExperimentConfig,
    y_test_original: dict[int, np.ndarray],
    baseline_original_predictions: dict[int, np.ndarray],
    joint_original_predictions: dict[int, np.ndarray],
) -> dict[str, object]:
    baseline_metrics: dict[str, dict[str, float]] = {}
    joint_metrics: dict[str, dict[str, float]] = {}
    for horizon in cfg.horizons:
        baseline_metrics[str(horizon)] = _compute_metrics(
            y_test_original[horizon],
            baseline_original_predictions[horizon],
        )
        joint_metrics[str(horizon)] = _compute_metrics(
            y_test_original[horizon],
            joint_original_predictions[horizon],
        )
    return {
        "baseline_test": baseline_metrics,
        "joint_test": joint_metrics,
    }


def _prediction_payload(
    *,
    cfg: YRDExperimentConfig,
    y_test_scaled: dict[int, np.ndarray],
    baseline_scaled_predictions: dict[int, np.ndarray],
    joint_scaled_predictions: dict[int, np.ndarray],
    y_test_original: dict[int, np.ndarray],
    baseline_original_predictions: dict[int, np.ndarray],
    joint_original_predictions: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for horizon in cfg.horizons:
        horizon_label = f"{int(horizon)}h"
        payload[f"y_test_scaled_{horizon_label}"] = y_test_scaled[horizon]
        payload[f"baseline_scaled_{horizon_label}"] = baseline_scaled_predictions[horizon]
        payload[f"joint_scaled_{horizon_label}"] = joint_scaled_predictions[horizon]
        payload[f"y_test_original_{horizon_label}"] = y_test_original[horizon]
        payload[f"baseline_original_{horizon_label}"] = baseline_original_predictions[horizon]
        payload[f"joint_original_{horizon_label}"] = joint_original_predictions[horizon]
    return payload


def run_or_load_air_search_predictions(
    bundle: dict[str, object],
    *,
    force_retrain: bool,
) -> dict[str, object]:
    cfg = bundle["cfg"]
    artifact_paths = bundle["artifact_paths"]
    target_dim = bundle["target_dim"]
    target_names = bundle["target_names"]
    stats = bundle["stats"]
    x_test = bundle["x_test"]
    y_test_scaled = bundle["y_test_scaled"]

    assert isinstance(cfg, YRDExperimentConfig)
    assert isinstance(artifact_paths, dict)
    assert isinstance(target_dim, int)
    assert isinstance(target_names, list)
    assert isinstance(stats, dict)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_test_scaled, dict)

    set_seed(cfg.seed)
    baseline_model = PersistenceBaseline(target_dim=target_dim, horizons=cfg.horizons)
    checkpoint_path = Path(artifact_paths["checkpoint"])
    loss_history_path = Path(artifact_paths["loss_history"])

    if checkpoint_path.exists() and loss_history_path.exists() and not force_retrain:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        if "model_kwargs" not in checkpoint_payload:
            raise RuntimeError(
                "Checkpoint is missing model_kwargs metadata. Set force_retrain=True to rebuild the cache."
            )
        joint_model = rebuild_joint_model_from_checkpoint(checkpoint_payload)
        loss_history_payload = load_json(loss_history_path)
    else:
        training_result = train_joint_model_with_history(
            n_stations=bundle["sample_bundle"]["n_stations"],
            n_features=bundle["sample_bundle"]["n_features"],
            history_hours=cfg.history_hours,
            target_dim=target_dim,
            hidden_dim=cfg.hidden_dim,
            horizons=cfg.horizons,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            max_epochs=cfg.max_epochs,
            early_stopping_patience=cfg.early_stopping_patience,
            seed=cfg.seed,
            x_train=bundle["x_train"],
            y_train=bundle["y_train_scaled"],
            x_val=bundle["x_val"],
            y_val=bundle["y_val_scaled"],
            model_name=cfg.model_name,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            norm_type=cfg.norm_type,
            activation=cfg.activation,
        )
        joint_model = training_result["model"]
        checkpoint_payload = {
            "state_dict": joint_model.state_dict(),
            "best_epoch": training_result["best_epoch"],
            "best_val_loss": training_result["best_val_loss"],
            "train_loss_history": training_result["train_loss_history"],
            "val_loss_history": training_result["val_loss_history"],
            "model_kwargs": training_result["model_kwargs"],
            "run_context": bundle["run_context"],
        }
        torch.save(checkpoint_payload, checkpoint_path)
        loss_history_payload = {
            "best_epoch": training_result["best_epoch"],
            "best_val_loss": training_result["best_val_loss"],
            "train_loss_history": training_result["train_loss_history"],
            "val_loss_history": training_result["val_loss_history"],
        }
        save_json(loss_history_path, _to_jsonable(loss_history_payload))

    joint_model.eval()
    baseline_scaled_predictions = _predict_numpy(baseline_model, x_test, cfg.horizons)
    joint_scaled_predictions = _predict_numpy(joint_model, x_test, cfg.horizons)
    y_test_original = {
        horizon: inverse_transform_targets(y_test_scaled[horizon], target_names, stats)
        for horizon in cfg.horizons
    }
    baseline_original_predictions = {
        horizon: inverse_transform_targets(baseline_scaled_predictions[horizon], target_names, stats)
        for horizon in cfg.horizons
    }
    joint_original_predictions = {
        horizon: inverse_transform_targets(joint_scaled_predictions[horizon], target_names, stats)
        for horizon in cfg.horizons
    }
    np.savez(
        artifact_paths["predictions"],
        **_prediction_payload(
            cfg=cfg,
            y_test_scaled=y_test_scaled,
            baseline_scaled_predictions=baseline_scaled_predictions,
            joint_scaled_predictions=joint_scaled_predictions,
            y_test_original=y_test_original,
            baseline_original_predictions=baseline_original_predictions,
            joint_original_predictions=joint_original_predictions,
        ),
    )
    metrics_payload = _build_metrics_payload(
        cfg=cfg,
        y_test_original=y_test_original,
        baseline_original_predictions=baseline_original_predictions,
        joint_original_predictions=joint_original_predictions,
    )
    save_json(Path(artifact_paths["metrics"]), _to_jsonable(metrics_payload))
    run_manifest = {
        "cache_dir": str(Path(artifact_paths["config"]).parent),
        "results_dir": str(Path(artifact_paths["run_manifest"]).parent),
        "config": str(artifact_paths["config"]),
        "checkpoint": str(checkpoint_path),
        "loss_history": str(loss_history_path),
        "predictions": str(artifact_paths["predictions"]),
        "metrics": str(artifact_paths["metrics"]),
    }
    save_json(Path(artifact_paths["run_manifest"]), run_manifest)
    return {
        "baseline_model": baseline_model,
        "joint_model": joint_model,
        "checkpoint_payload": checkpoint_payload,
        "loss_history_payload": loss_history_payload,
        "baseline_scaled_predictions": baseline_scaled_predictions,
        "joint_scaled_predictions": joint_scaled_predictions,
        "y_test_original": y_test_original,
        "baseline_original_predictions": baseline_original_predictions,
        "joint_original_predictions": joint_original_predictions,
        "metrics_payload": metrics_payload,
        "run_manifest": run_manifest,
    }


def summarize_air_search_station_pollutant_effects(
    *,
    sample_summaries: list[dict[str, object]],
    station_ids: list[str],
    pairwise_feature_name: str = "PM2.5",
) -> dict[str, object]:
    return {
        "conditional_synergy": summarize_global_station_pollutant_synergy(
            sample_summaries,
            station_ids=station_ids,
        ),
        "single_pollutant_pairwise": summarize_global_station_single_pollutant_ei(
            sample_summaries,
            station_ids=station_ids,
            feature_name=pairwise_feature_name,
        ),
    }


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def parse_float_csv(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


def ensure_air_tuning_state(root_dir: Path) -> dict[str, Path]:
    log_dir = Path(root_dir) / "docs" / "log" / "air_tuning"
    log_dir.mkdir(parents=True, exist_ok=True)
    return {
        "log_dir": log_dir,
        "run_history": log_dir / "run_history.jsonl",
        "coarse_leaderboard": log_dir / "coarse_leaderboard.json",
        "refine_results": log_dir / "refine_results.json",
        "report_manifest": log_dir / "report_manifest.json",
    }


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_jsonable(payload), ensure_ascii=False) + "\n")


def build_station_variable_index_map(target_names: list[str], variable: str) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    suffix = f"__{variable}"
    for index, name in enumerate(target_names):
        if not name.endswith(suffix):
            continue
        _, station_id, _ = name.split("__")
        mapping.setdefault(station_id, []).append(index)
    return mapping


def _subset_metric(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    suffix: str,
) -> dict[str, float]:
    indices = [index for index, name in enumerate(target_names) if name.endswith(f"__{suffix}")]
    if not indices:
        raise ValueError(f"Could not find targets ending with __{suffix}.")
    return _compute_metrics(y_true[:, indices], y_pred[:, indices])


def _nonself_edge_frame(edge_rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(edge_rows)
    if frame.empty or "source_station_id" not in frame.columns or "target_station_id" not in frame.columns:
        return frame
    filtered = frame[frame["source_station_id"] != frame["target_station_id"]].copy()
    return filtered if not filtered.empty else frame


def summarize_edge_distribution(edge_rows: list[dict[str, object]]) -> dict[str, float]:
    frame = _nonself_edge_frame(edge_rows)
    if frame.empty:
        return {"mean": 0.0, "positive_mean": 0.0, "negative_ratio": 0.0, "count": 0.0}
    values = frame["mean"].astype(float).to_numpy()
    positive_values = values[values > 0.0]
    return {
        "mean": float(values.mean()),
        "positive_mean": float(positive_values.mean()) if positive_values.size else 0.0,
        "negative_ratio": float(np.mean(values < 0.0)),
        "count": float(values.size),
    }


def build_coarse_row(
    *,
    city_en: str,
    horizon: int,
    o3_rmse: float,
    baseline_o3_rmse: float,
    syn_mean: float,
    syn_negative_ratio: float,
    pm25_to_o3_mean: float,
    pm25_negative_ratio: float,
) -> dict[str, object]:
    passes_accuracy_gate = float(o3_rmse) < float(baseline_o3_rmse)
    return {
        "city_en": str(city_en),
        "horizon": int(horizon),
        "o3_rmse": float(o3_rmse),
        "baseline_o3_rmse": float(baseline_o3_rmse),
        "passes_accuracy_gate": bool(passes_accuracy_gate),
        "primary_syn_mean": float(syn_mean),
        "primary_syn_abs_mean": abs(float(syn_mean)),
        "syn_negative_ratio": float(syn_negative_ratio),
        "pm25_to_o3_mean": float(pm25_to_o3_mean),
        "pm25_negative_ratio": float(pm25_negative_ratio),
    }


def choose_tm_gamma(
    rows: list[dict[str, object]],
    *,
    negative_ratio_threshold: float = DEFAULT_NEGATIVE_RATIO_THRESHOLD,
) -> dict[str, object]:
    if not rows:
        raise ValueError("rows must not be empty.")
    eligible = [
        row for row in rows
        if float(row.get("syn_negative_ratio", 1.0)) <= float(negative_ratio_threshold)
    ]
    candidates = eligible or rows
    return sorted(
        candidates,
        key=lambda row: (
            float(row.get("gamma", math.inf)),
            float(row.get("syn_negative_ratio", math.inf)),
            -abs(float(row.get("syn_mean", row.get("primary_syn_mean", 0.0)))),
        ),
    )[0]


def build_report_manifest(
    *,
    city_en: str,
    horizon: int,
    selected_refine_run: dict[str, object],
    graph_paths: dict[str, str],
) -> dict[str, object]:
    return {
        "city_en": str(city_en),
        "horizon": int(horizon),
        "selected_refine_run": _to_jsonable(selected_refine_run),
        "graphs": dict(graph_paths),
    }


def _horizon_label(horizon: int) -> str:
    return f"{int(horizon)}h"


def _coarse_run_tag(*, use_smoke: bool) -> str:
    return "coarse_smoke" if use_smoke else "coarse"


def _refine_run_tag(*, gamma: float, sample_count: int, seed: int, use_smoke: bool) -> str:
    gamma_label = str(f"{float(gamma):.2f}").replace(".", "p")
    prefix = "refine_smoke" if use_smoke else "refine"
    return f"{prefix}_tm_g{gamma_label}_m{int(sample_count)}_seed{int(seed)}"


def _sort_coarse_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            not bool(row["passes_accuracy_gate"]),
            -abs(float(row["primary_syn_mean"])),
            float(row["syn_negative_ratio"]),
            -float(row["pm25_to_o3_mean"]),
            str(row["city_en"]),
            int(row["horizon"]),
        ),
    )


def _station_groups(bundle: dict[str, object]) -> tuple[dict[str, list[int]], dict[str, dict[str, list[int]]]]:
    cfg = bundle["cfg"]
    sample_bundle = bundle["sample_bundle"]
    station_ids = bundle["station_ids"]
    assert isinstance(cfg, YRDExperimentConfig)
    assert isinstance(sample_bundle, dict)
    assert isinstance(station_ids, list)
    station_source_groups = build_one_step_station_source_groups(
        n_stations=sample_bundle["n_stations"],
        n_features=sample_bundle["n_features"],
        station_ids=station_ids,
    )
    station_pollutant_feature_groups = build_one_step_station_pollutant_feature_groups(
        n_stations=sample_bundle["n_stations"],
        n_features=sample_bundle["n_features"],
        pollutant_feature_indices={
            "O3": cfg.input_variables.index("O3"),
            "PM2.5": cfg.input_variables.index("PM2.5"),
        },
        station_ids=station_ids,
    )
    return station_source_groups, station_pollutant_feature_groups


def _o3_target_indices(bundle: dict[str, object], *, target_variable: str = "O3") -> dict[str, list[int]]:
    target_names = bundle["target_names"]
    assert isinstance(target_names, list)
    return build_station_variable_index_map(target_names, target_variable)


def compute_air_search_nis_summary(
    bundle: dict[str, object],
    predictions: dict[str, object],
    *,
    coupling_sample_count: int,
    sampling_seed: int,
    target_variable: str = "O3",
) -> dict[str, object]:
    cfg = bundle["cfg"]
    x_train = bundle["x_train"]
    station_ids = bundle["station_ids"]
    assert isinstance(cfg, YRDExperimentConfig)
    assert isinstance(x_train, np.ndarray)
    assert isinstance(station_ids, list)
    horizon = _single_horizon(cfg)
    center = compute_training_input_center(x_train)
    synthetic_inputs = sample_uniform_box_inputs(
        center=center,
        box_size=float(cfg.box_size),
        sample_count=int(coupling_sample_count),
        seed=int(sampling_seed),
    )
    station_source_groups, station_pollutant_feature_groups = _station_groups(bundle)
    target_indices_by_station = _o3_target_indices(bundle, target_variable=target_variable)
    sigma_eps = estimate_residual_covariance(
        bundle["y_test_scaled"][horizon],
        predictions["joint_scaled_predictions"][horizon],
    )
    joint_model = predictions["joint_model"]
    joint_model.eval()

    sample_summaries: list[dict[str, object]] = []
    for sample_id, synthetic_input in enumerate(synthetic_inputs):
        sample_x = torch.from_numpy(synthetic_input[None, ...]).to(dtype=torch.float32)
        flat_sample = sample_x.reshape(-1).detach().clone().requires_grad_(True)

        def horizon_model(tensor: torch.Tensor) -> torch.Tensor:
            shaped = tensor.reshape(1, bundle["sample_bundle"]["n_stations"], bundle["sample_bundle"]["n_features"])
            return joint_model(shaped)[horizon].reshape(-1)

        for target_station_id, target_indices in target_indices_by_station.items():
            jacobian = jacobian_for_target_subset(
                horizon_model,
                flat_sample,
                target_indices=target_indices,
            ).detach().cpu().numpy()
            sample_summaries.append(
                {
                    "sample_id": int(sample_id),
                    "target_station_id": target_station_id,
                    **compute_station_level_nis_summary(
                        jacobian=jacobian,
                        sigma_eps=sigma_eps,
                        station_source_groups=station_source_groups,
                        target_indices=target_indices,
                        box_size=float(cfg.box_size),
                    ),
                    **compute_station_pollutant_pair_synergy_summary(
                        jacobian=jacobian,
                        sigma_eps=sigma_eps,
                        station_pollutant_feature_groups=station_pollutant_feature_groups,
                        target_indices=target_indices,
                        method="nis",
                        box_size=float(cfg.box_size),
                    ),
                }
            )

    return {
        "method": "nis",
        "horizon": int(horizon),
        "coupling_sample_count": int(coupling_sample_count),
        "sampling_seed": int(sampling_seed),
        "sample_summaries": sample_summaries,
        "o3_pairwise": summarize_global_station_coupling(sample_summaries, station_ids=station_ids),
        **summarize_air_search_station_pollutant_effects(
            sample_summaries=sample_summaries,
            station_ids=station_ids,
            pairwise_feature_name="PM2.5",
        ),
    }


def compute_air_search_tm_summary(
    bundle: dict[str, object],
    predictions: dict[str, object],
    *,
    sample_count: int,
    sampling_seed: int,
    gamma: float,
    target_variable: str = "O3",
    nonnegative_variables: tuple[str, ...] = DEFAULT_TM_NONNEGATIVE_VARIABLES,
    box_mode: str = "per_variable",
    global_box_size_override: float | None = None,
) -> dict[str, object]:
    cfg = bundle["cfg"]
    station_ids = bundle["station_ids"]
    assert isinstance(cfg, YRDExperimentConfig)
    assert isinstance(station_ids, list)
    horizon = _single_horizon(cfg)
    profile = estimate_support_cover_box_profile(
        x_train=bundle["x_train"],
        input_variables=cfg.input_variables,
        gamma=float(gamma),
        stats=bundle["stats"],
        nonnegative_variables=tuple(nonnegative_variables),
    )
    if box_mode == "global_max":
        profile = collapse_support_cover_box_profile_to_global_max(
            profile,
            global_box_size_override=global_box_size_override,
        )
    elif box_mode != "per_variable":
        raise ValueError(f"Unsupported box_mode={box_mode!r}.")
    synthetic_inputs = sample_uniform_box_inputs(
        center=np.asarray(profile["center"], dtype=np.float32),
        box_size=np.asarray(profile["box_size_by_feature"], dtype=np.float32),
        sample_count=int(sample_count),
        seed=int(sampling_seed),
        lower_bounds=None if profile["lower_bounds"] is None else np.asarray(profile["lower_bounds"], dtype=np.float32),
    )
    flat_source_samples = synthetic_inputs.reshape(synthetic_inputs.shape[0], -1)
    joint_model = predictions["joint_model"]
    joint_model.eval()
    with torch.no_grad():
        predicted_next = joint_model(torch.from_numpy(synthetic_inputs).to(dtype=torch.float32))[horizon].detach().cpu().numpy()
    station_source_groups, station_pollutant_feature_groups = _station_groups(bundle)
    target_indices_by_station = _o3_target_indices(bundle, target_variable=target_variable)
    sample_summaries: list[dict[str, object]] = []
    for target_station_id, target_indices in target_indices_by_station.items():
        target_samples = predicted_next[:, target_indices]
        sample_summaries.append(
            {
                "target_station_id": target_station_id,
                **compute_station_level_ei_summary(
                    method="tm",
                    station_source_groups=station_source_groups,
                    source_samples=flat_source_samples,
                    target_samples=target_samples,
                ),
                **compute_station_pollutant_pair_synergy_summary(
                    method="tm",
                    station_pollutant_feature_groups=station_pollutant_feature_groups,
                    source_samples=flat_source_samples,
                    target_samples=target_samples,
                ),
            }
        )
    return {
        "method": "tm",
        "horizon": int(horizon),
        "gamma": float(gamma),
        "sample_count": int(sample_count),
        "sampling_seed": int(sampling_seed),
        "box_mode": str(profile.get("box_mode", box_mode)),
        "profile": profile,
        "sample_summaries": sample_summaries,
        "o3_pairwise": summarize_global_station_coupling(sample_summaries, station_ids=station_ids),
        **summarize_air_search_station_pollutant_effects(
            sample_summaries=sample_summaries,
            station_ids=station_ids,
            pairwise_feature_name="PM2.5",
        ),
    }


def _graph_edge_frame(summary: dict[str, object], key: str) -> pd.DataFrame:
    return pd.DataFrame(summary.get(key, []))


def export_air_search_graphs(
    *,
    bundle: dict[str, object],
    summary: dict[str, object],
) -> dict[str, str]:
    artifact_paths = bundle["artifact_paths"]
    station_positions = bundle["city_metadata"][["station_id", "lon", "lat"]]
    station_ids = bundle["station_ids"]
    horizon_label = _horizon_label(_single_horizon(bundle["cfg"]))

    o3_pairwise_df = _graph_edge_frame(summary["o3_pairwise"], "pairwise_edges")
    pm25_pairwise_df = _graph_edge_frame(summary["single_pollutant_pairwise"], "pairwise_edges")
    synergy_df = _graph_edge_frame(summary["conditional_synergy"], "conditional_synergy_edges")
    o3_display = o3_pairwise_df[
        (o3_pairwise_df["source_station_id"] != o3_pairwise_df["target_station_id"])
        & (o3_pairwise_df["mean"] > 0.0)
    ].copy() if not o3_pairwise_df.empty else pd.DataFrame()
    draw_station_causal_graph(
        station_positions=station_positions,
        pairwise_edges=o3_display,
        horizon_label=horizon_label,
        out_path=artifact_paths["o3_pairwise_graph"],
        title=f"{bundle['run_context']['city_en'].title()} O3 -> O3 pairwise graph ({horizon_label})",
        strength_col="mean",
        node_self_strengths=build_self_loop_node_strengths(o3_pairwise_df, station_ids=station_ids),
    )
    pm25_display = pm25_pairwise_df[
        (pm25_pairwise_df["source_station_id"] != pm25_pairwise_df["target_station_id"])
        & (pm25_pairwise_df["mean"] > 0.0)
    ].copy() if not pm25_pairwise_df.empty else pd.DataFrame()
    draw_station_causal_graph(
        station_positions=station_positions,
        pairwise_edges=pm25_display,
        horizon_label=horizon_label,
        out_path=artifact_paths["pm25_to_o3_pairwise_graph"],
        title=f"{bundle['run_context']['city_en'].title()} PM2.5 -> O3 pairwise graph ({horizon_label})",
        strength_col="mean",
        node_self_strengths=build_self_loop_node_strengths(pm25_pairwise_df, station_ids=station_ids),
        legend_label="PM2.5 -> O3 edge",
    )
    synergy_display = synergy_df[synergy_df["source_station_id"] != synergy_df["target_station_id"]].copy() if not synergy_df.empty else pd.DataFrame()
    if not synergy_display.empty:
        synergy_display["abs_mean"] = synergy_display["mean"].abs()
    draw_station_causal_graph(
        station_positions=station_positions,
        pairwise_edges=synergy_display,
        horizon_label=horizon_label,
        out_path=artifact_paths["o3_pm25_synergy_graph"],
        title=f"{bundle['run_context']['city_en'].title()} O3+PM2.5 -> O3 synergy graph ({horizon_label})",
        strength_col="abs_mean",
        positive_color="#2F7D63",
        negative_color="#B04A5A",
        legend_label="Synergy edge",
        node_self_strengths=build_self_loop_node_strengths(synergy_df, station_ids=station_ids),
        node_colorbar_label="Self Syn",
    )
    return {
        "o3_pairwise": str(artifact_paths["o3_pairwise_graph"]),
        "pm25_to_o3_pairwise": str(artifact_paths["pm25_to_o3_pairwise_graph"]),
        "o3_pm25_synergy": str(artifact_paths["o3_pm25_synergy_graph"]),
    }


def _coarse_result_row(
    *,
    city_en: str,
    horizon: int,
    bundle: dict[str, object],
    predictions: dict[str, object],
    coarse_summary: dict[str, object],
) -> dict[str, object]:
    o3_metrics = _subset_metric(
        y_true=predictions["y_test_original"][horizon],
        y_pred=predictions["joint_original_predictions"][horizon],
        target_names=bundle["target_names"],
        suffix="O3",
    )
    baseline_o3_metrics = _subset_metric(
        y_true=predictions["y_test_original"][horizon],
        y_pred=predictions["baseline_original_predictions"][horizon],
        target_names=bundle["target_names"],
        suffix="O3",
    )
    syn_stats = summarize_edge_distribution(coarse_summary["conditional_synergy"]["conditional_synergy_edges"])
    pm25_stats = summarize_edge_distribution(coarse_summary["single_pollutant_pairwise"]["pairwise_edges"])
    row = build_coarse_row(
        city_en=city_en,
        horizon=horizon,
        o3_rmse=o3_metrics["rmse"],
        baseline_o3_rmse=baseline_o3_metrics["rmse"],
        syn_mean=syn_stats["mean"],
        syn_negative_ratio=syn_stats["negative_ratio"],
        pm25_to_o3_mean=pm25_stats["mean"],
        pm25_negative_ratio=pm25_stats["negative_ratio"],
    )
    row.update(
        {
            "stage": "coarse",
            "run_tag": str(bundle["run_context"]["run_tag"]),
            "artifact_dir": str(bundle["artifact_paths"]["results_dir"]),
        }
    )
    return row


def run_coarse_stage(
    *,
    root_dir: Path,
    cities: list[str],
    horizons: list[int],
    smoke: bool,
    force_retrain: bool = False,
    force_recompute_coupling: bool = False,
    coupling_sample_count: int | None = None,
) -> dict[str, object]:
    state_paths = ensure_air_tuning_state(root_dir)
    effective_sample_count = (
        int(coupling_sample_count)
        if coupling_sample_count is not None
        else (DEFAULT_COARSE_SAMPLE_COUNT_SMOKE if smoke else DEFAULT_COARSE_SAMPLE_COUNT)
    )
    rows: list[dict[str, object]] = []
    for city_en in cities:
        for horizon in horizons:
            cfg = build_air_search_config(root_dir, city_en=city_en, horizon=horizon, test_mode=smoke)
            bundle = prepare_air_search_bundle(
                cfg=cfg,
                city_en=city_en,
                run_tag=_coarse_run_tag(use_smoke=smoke),
                use_smoke=smoke,
            )
            predictions = run_or_load_air_search_predictions(bundle, force_retrain=force_retrain)
            coarse_summary_path = Path(bundle["artifact_paths"]["coarse_summary"])
            if coarse_summary_path.exists() and not force_recompute_coupling:
                coarse_summary = load_json(coarse_summary_path)
            else:
                coarse_summary = compute_air_search_nis_summary(
                    bundle,
                    predictions,
                    coupling_sample_count=effective_sample_count,
                    sampling_seed=cfg.seed,
                )
                save_json(coarse_summary_path, _to_jsonable(coarse_summary))
            row = _coarse_result_row(
                city_en=city_en,
                horizon=horizon,
                bundle=bundle,
                predictions=predictions,
                coarse_summary=coarse_summary,
            )
            save_json(Path(bundle["artifact_paths"]["leaderboard_row"]), _to_jsonable(row))
            append_jsonl(Path(state_paths["run_history"]), row)
            rows.append(row)
    leaderboard = {"rows": _sort_coarse_rows(rows)}
    save_json(Path(state_paths["coarse_leaderboard"]), _to_jsonable(leaderboard))
    return leaderboard


def _load_leaderboard_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = load_json(path)
    rows = payload.get("rows", [])
    return list(rows) if isinstance(rows, list) else []


def _merge_unique_rows(
    existing_rows: list[dict[str, object]],
    new_rows: list[dict[str, object]],
    *,
    key_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    merged: dict[tuple[object, ...], dict[str, object]] = {}
    for row in existing_rows + new_rows:
        key = tuple(row.get(field) for field in key_fields)
        merged[key] = row
    return list(merged.values())


def _group_refine_rows_by_gamma(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[float, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(float(row["gamma"]), []).append(row)
    aggregated: list[dict[str, object]] = []
    for gamma, gamma_rows in grouped.items():
        aggregated.append(
            {
                "gamma": float(gamma),
                "syn_negative_ratio": float(np.mean([float(row["syn_negative_ratio"]) for row in gamma_rows])),
                "syn_mean": float(np.mean([float(row["syn_mean"]) for row in gamma_rows])),
            }
        )
    return sorted(aggregated, key=lambda row: float(row["gamma"]))


def run_refine_stage(
    *,
    root_dir: Path,
    cities: list[str],
    horizons: list[int],
    smoke: bool,
    top_k: int,
    force_retrain: bool = False,
    force_recompute_coupling: bool = False,
    tm_sample_counts: list[int] | None = None,
    tm_seeds: list[int] | None = None,
    tm_gammas: list[float] | None = None,
) -> dict[str, object]:
    state_paths = ensure_air_tuning_state(root_dir)
    existing_payload = (
        load_json(Path(state_paths["refine_results"]))
        if Path(state_paths["refine_results"]).exists()
        else {"rows": [], "reports": []}
    )
    shortlist = [
        row for row in _load_leaderboard_rows(Path(state_paths["coarse_leaderboard"]))
        if row.get("city_en") in cities and int(row.get("horizon", -1)) in horizons
    ]
    shortlist = _sort_coarse_rows(shortlist)[: max(1, int(top_k))]
    tm_sample_counts = tm_sample_counts or ([64] if smoke else [512])
    tm_seeds = tm_seeds or [0]
    tm_gammas = tm_gammas or [1.0, 1.1, 1.2]
    refine_rows: list[dict[str, object]] = []
    report_manifests: list[dict[str, object]] = []

    for coarse_row in shortlist:
        city_en = str(coarse_row["city_en"])
        horizon = int(coarse_row["horizon"])
        cfg = build_air_search_config(root_dir, city_en=city_en, horizon=horizon, test_mode=smoke)
        candidate_records: list[dict[str, object]] = []
        for gamma in tm_gammas:
            for sample_count in tm_sample_counts:
                for seed in tm_seeds:
                    bundle = prepare_air_search_bundle(
                        cfg=cfg,
                        city_en=city_en,
                        run_tag=_refine_run_tag(
                            gamma=float(gamma),
                            sample_count=int(sample_count),
                            seed=int(seed),
                            use_smoke=smoke,
                        ),
                        use_smoke=smoke,
                    )
                    predictions = run_or_load_air_search_predictions(bundle, force_retrain=force_retrain)
                    refine_summary_path = Path(bundle["artifact_paths"]["refine_summary"])
                    if refine_summary_path.exists() and not force_recompute_coupling:
                        refine_summary = load_json(refine_summary_path)
                    else:
                        refine_summary = compute_air_search_tm_summary(
                            bundle,
                            predictions,
                            sample_count=int(sample_count),
                            sampling_seed=int(seed),
                            gamma=float(gamma),
                        )
                        save_json(refine_summary_path, _to_jsonable(refine_summary))
                    syn_stats = summarize_edge_distribution(
                        refine_summary["conditional_synergy"]["conditional_synergy_edges"]
                    )
                    pm25_stats = summarize_edge_distribution(
                        refine_summary["single_pollutant_pairwise"]["pairwise_edges"]
                    )
                    record = {
                        "stage": "refine",
                        "city_en": city_en,
                        "horizon": int(horizon),
                        "gamma": float(gamma),
                        "sample_count": int(sample_count),
                        "seed": int(seed),
                        "syn_mean": float(syn_stats["mean"]),
                        "syn_negative_ratio": float(syn_stats["negative_ratio"]),
                        "pm25_to_o3_mean": float(pm25_stats["mean"]),
                        "pm25_negative_ratio": float(pm25_stats["negative_ratio"]),
                        "run_tag": str(bundle["run_context"]["run_tag"]),
                        "artifact_dir": str(bundle["artifact_paths"]["results_dir"]),
                        "summary": refine_summary,
                    }
                    append_jsonl(
                        Path(state_paths["run_history"]),
                        {key: value for key, value in record.items() if key != "summary"},
                    )
                    candidate_records.append(record)
                    refine_rows.append({key: value for key, value in record.items() if key != "summary"})

        gamma_rows = _group_refine_rows_by_gamma(candidate_records)
        winner = choose_tm_gamma(gamma_rows)
        chosen_candidates = [
            row for row in candidate_records
            if float(row["gamma"]) == float(winner["gamma"])
        ]
        selected = sorted(
            chosen_candidates,
            key=lambda row: (-int(row["sample_count"]), int(row["seed"])),
        )[0]
        graph_paths = export_air_search_graphs(
            bundle=prepare_air_search_bundle(
                cfg=cfg,
                city_en=city_en,
                run_tag=str(selected["run_tag"]),
                use_smoke=smoke,
            ),
            summary=selected["summary"],
        )
        manifest = build_report_manifest(
            city_en=city_en,
            horizon=horizon,
            selected_refine_run={key: value for key, value in selected.items() if key != "summary"},
            graph_paths=graph_paths,
        )
        manifest_path = build_air_search_artifact_paths(
            root_dir=root_dir,
            city_en=city_en,
            horizon=horizon,
            run_tag=str(selected["run_tag"]),
            use_smoke=smoke,
        )["report_manifest"]
        save_json(Path(manifest_path), _to_jsonable(manifest))
        report_manifests.append(manifest)

    payload = {
        "rows": _merge_unique_rows(
            list(existing_payload.get("rows", [])),
            refine_rows,
            key_fields=("city_en", "horizon", "gamma", "sample_count", "seed", "run_tag"),
        ),
        "reports": _merge_unique_rows(
            list(existing_payload.get("reports", [])),
            report_manifests,
            key_fields=("city_en", "horizon"),
        ),
    }
    save_json(Path(state_paths["refine_results"]), _to_jsonable(payload))
    save_json(Path(state_paths["report_manifest"]), _to_jsonable({"reports": payload["reports"]}))
    return payload


def run_report_stage(
    *,
    root_dir: Path,
) -> dict[str, object]:
    state_paths = ensure_air_tuning_state(root_dir)
    refine_payload = load_json(Path(state_paths["refine_results"])) if Path(state_paths["refine_results"]).exists() else {"rows": [], "reports": []}
    reports = list(refine_payload.get("reports", []))
    payload = {"reports": reports}
    save_json(Path(state_paths["report_manifest"]), _to_jsonable(payload))
    return payload
