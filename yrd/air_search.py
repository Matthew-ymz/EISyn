from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .analysis import save_json
from .config import YRDExperimentConfig
from .coupling import summarize_global_station_pollutant_synergy, summarize_global_station_single_pollutant_ei
from .data import build_one_step_samples, load_dataset
from .models import PersistenceBaseline
from .train import _compute_metrics, _predict_numpy, rebuild_joint_model_from_checkpoint, set_seed, train_joint_model_with_history

_YRD_CITIES = frozenset({"shanghai", "nanjing", "hangzhou"})
_BTHSA_CITIES = frozenset({"beijing"})


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
        "leaderboard_row": cache_dir / "leaderboard_row.json",
        "run_manifest": results_dir / "run_manifest.json",
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
