from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .analysis import coupling_by_horizon_bullets, metrics_bullets, save_json, write_markdown_summary
from .config import YRDExperimentConfig
from .coupling import compute_subset_nis_summary, estimate_residual_covariance, save_coupling_summary
from .data import build_windowed_samples, flatten_input_group_indices, load_dataset
from .models import JointStationMLP, PersistenceBaseline


def joint_model_kwargs(
    *,
    n_stations: int,
    n_features: int,
    history_hours: int,
    target_dim: int,
    hidden_dim: int,
    horizons: tuple[int, ...],
    model_name: str = "baseline",
    num_layers: int = 2,
    dropout: float = 0.0,
    norm_type: str = "layernorm",
    activation: str = "relu",
) -> dict[str, Any]:
    return {
        "n_stations": n_stations,
        "n_features": n_features,
        "history_hours": history_hours,
        "target_dim": target_dim,
        "hidden_dim": hidden_dim,
        "horizons": tuple(horizons),
        "model_name": model_name,
        "num_layers": num_layers,
        "dropout": dropout,
        "norm_type": norm_type,
        "activation": activation,
    }


def rebuild_joint_model_from_checkpoint(payload: dict[str, Any]) -> JointStationMLP:
    model = JointStationMLP(**payload["model_kwargs"])
    model.load_state_dict(payload["state_dict"])
    return model


def ensure_output_layout(root: Path) -> dict[str, Path]:
    cache_dir = root / "exp" / "cache" / "yrd_coupling"
    results_dir = root / "fig" / "yrd_shanghai" / "artifacts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return {"cache_dir": cache_dir, "results_dir": results_dir}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).to(dtype=torch.float32)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_true - y_pred
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    corr = float(np.corrcoef(y_true.reshape(-1), y_pred.reshape(-1))[0, 1]) if y_true.size > 1 else 1.0
    return {"rmse": rmse, "mae": mae, "corr": corr}


def _predict_numpy(model: torch.nn.Module, x: np.ndarray, horizons: tuple[int, ...]) -> dict[int, np.ndarray]:
    model.eval()
    with torch.no_grad():
        outputs = model(_to_tensor(x))
    return {horizon: tensor.cpu().numpy() for horizon, tensor in outputs.items() if horizon in horizons}


def train_joint_model_with_history(
    *,
    n_stations: int,
    n_features: int,
    history_hours: int,
    target_dim: int,
    hidden_dim: int,
    horizons: tuple[int, ...],
    learning_rate: float,
    weight_decay: float = 0.0,
    batch_size: int,
    epochs: int,
    max_epochs: int | None = None,
    early_stopping_patience: int | None = None,
    seed: int,
    x_train: np.ndarray,
    y_train: dict[int, np.ndarray],
    x_val: np.ndarray,
    y_val: dict[int, np.ndarray],
    model_name: str = "baseline",
    num_layers: int = 2,
    dropout: float = 0.0,
    norm_type: str = "layernorm",
    activation: str = "relu",
) -> dict[str, Any]:
    set_seed(seed)
    model_kwargs = joint_model_kwargs(
        n_stations=n_stations,
        n_features=n_features,
        history_hours=history_hours,
        target_dim=target_dim,
        hidden_dim=hidden_dim,
        horizons=horizons,
        model_name=model_name,
        num_layers=num_layers,
        dropout=dropout,
        norm_type=norm_type,
        activation=activation,
    )
    model = JointStationMLP(**model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_train_tensor = _to_tensor(x_train)
    y_train_tensor = {h: _to_tensor(values) for h, values in y_train.items()}
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    stopped_early = False
    epochs_without_improvement = 0
    effective_epochs = max_epochs if max_epochs is not None else epochs

    for epoch in range(effective_epochs):
        model.train()
        permutation = torch.randperm(x_train_tensor.shape[0])
        batch_losses: list[float] = []
        for start in range(0, x_train_tensor.shape[0], batch_size):
            batch_indices = permutation[start : start + batch_size]
            batch_x = x_train_tensor[batch_indices]
            predictions = model(batch_x)
            loss = sum(
                torch.nn.functional.mse_loss(predictions[horizon], y_train_tensor[horizon][batch_indices])
                for horizon in horizons
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        train_loss_history.append(float(np.mean(batch_losses)))

        val_predictions = _predict_numpy(model, x_val, horizons)
        val_loss = float(sum(np.mean((val_predictions[horizon] - y_val[horizon]) ** 2) for horizon in horizons))
        val_loss_history.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "model": model,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "best_state_dict": best_state,
        "model_kwargs": model_kwargs,
        "stopped_early": stopped_early,
        "early_stopping_patience": early_stopping_patience,
    }


def _train_joint_model(
    cfg: YRDExperimentConfig,
    *,
    x_train: np.ndarray,
    y_train: dict[int, np.ndarray],
    x_val: np.ndarray,
    y_val: dict[int, np.ndarray],
    target_dim: int,
    n_stations: int,
    n_features: int,
) -> JointStationMLP:
    result = train_joint_model_with_history(
        n_stations=n_stations,
        n_features=n_features,
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
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        model_name=cfg.model_name,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        norm_type=cfg.norm_type,
        activation=cfg.activation,
    )
    return result["model"]


def run_smoke_pipeline(cfg: YRDExperimentConfig) -> dict[str, Any]:
    layout = ensure_output_layout(cfg.root_dir)
    ds, metadata = load_dataset(cfg, smoke=True)
    sample_bundle = build_windowed_samples(ds, metadata, cfg, smoke=True)
    splits = sample_bundle["splits"]

    x_train = splits["train"]["X"]
    x_val = splits["val"]["X"]
    x_test = splits["test"]["X"]
    y_train = splits["train"]["targets"]
    y_val = splits["val"]["targets"]
    y_test = splits["test"]["targets"]

    target_dim = y_train[cfg.horizons[0]].shape[1]
    baseline = PersistenceBaseline(target_dim=target_dim, horizons=cfg.horizons)
    model = _train_joint_model(
        cfg,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        target_dim=target_dim,
        n_stations=sample_bundle["n_stations"],
        n_features=sample_bundle["n_features"],
    )

    baseline_predictions = _predict_numpy(baseline, x_test, cfg.horizons)
    joint_predictions = _predict_numpy(model, x_test, cfg.horizons)

    metrics = {
        "baseline_test": {
            str(horizon): _compute_metrics(y_test[horizon], baseline_predictions[horizon])
            for horizon in cfg.horizons
        },
        "joint_test": {
            str(horizon): _compute_metrics(y_test[horizon], joint_predictions[horizon])
            for horizon in cfg.horizons
        },
    }

    source_groups = flatten_input_group_indices(
        cfg,
        n_stations=sample_bundle["n_stations"],
        station_index=0,
    )
    sample_x = _to_tensor(x_test[:1]).reshape(-1).detach().clone().requires_grad_(True)
    coupling_summary: dict[str, dict[str, Any]] = {}
    for horizon in cfg.horizons:
        horizon_model = (
            lambda tensor, target_horizon=horizon: model(
                tensor.reshape(1, cfg.history_hours, sample_bundle["n_stations"], sample_bundle["n_features"])
            )[target_horizon].reshape(-1)
        )
        jacobian = torch.autograd.functional.jacobian(horizon_model, sample_x).detach().cpu().numpy()
        sigma_eps = estimate_residual_covariance(y_test[horizon], joint_predictions[horizon])
        coupling_summary[str(horizon)] = compute_subset_nis_summary(
            jacobian=jacobian,
            sigma_eps=sigma_eps,
            source_groups=source_groups,
            target_indices=list(range(jacobian.shape[0])),
            box_size=cfg.box_size,
        )

    metrics_path = layout["cache_dir"] / "smoke_metrics.json"
    coupling_path = layout["cache_dir"] / "smoke_coupling_summary.json"
    prediction_path = layout["cache_dir"] / "smoke_predictions.npz"
    save_json(metrics_path, metrics)
    save_coupling_summary(coupling_summary, coupling_path)
    np.savez(
        prediction_path,
        y_test_1h=y_test[cfg.horizons[0]],
        pred_test_1h=joint_predictions[cfg.horizons[0]],
        y_test_24h=y_test[cfg.horizons[1]],
        pred_test_24h=joint_predictions[cfg.horizons[1]],
    )

    metrics_md_path = layout["results_dir"] / "smoke_metrics_summary.md"
    coupling_md_path = layout["results_dir"] / "smoke_coupling_summary.md"
    write_markdown_summary(
        metrics_md_path,
        title="YRD Smoke Metrics Summary",
        intro="这一文件解释了 smoke 级别多站点 MLP 预测实验的主要误差指标。",
        bullets=metrics_bullets(metrics),
    )
    write_markdown_summary(
        coupling_md_path,
        title="YRD Smoke Coupling Summary",
        intro="这一文件解释了基于局部 Jacobian 与残差协方差得到的连续耦合摘要。",
        bullets=coupling_by_horizon_bullets(coupling_summary),
    )

    return {
        "metrics": metrics,
        "coupling_summary": coupling_summary,
        "metrics_path": metrics_path,
        "coupling_path": coupling_path,
        "metrics_md_path": metrics_md_path,
        "coupling_md_path": coupling_md_path,
        "target_names": sample_bundle["target_names"],
    }
