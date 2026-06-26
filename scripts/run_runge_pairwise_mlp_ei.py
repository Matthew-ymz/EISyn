#!/usr/bin/env python3
"""Pairwise PEID-EI gateway readout on Runge-style component dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

RESULT_SUBDIR = Path("results/runge/pairwise_mlp_ei")
FIG_SUBDIR = Path("fig/runge/pairwise_mlp_ei")
TM_RESULT_SUBDIR = Path("results/runge/pairwise_mlp_tm_ei")
TM_FIG_SUBDIR = Path("fig/runge/pairwise_mlp_tm_ei")
TM_PATH_RESULT_SUBDIR = Path("results/runge/pairwise_mlp_tm_ei_path_effects")
TM_PATH_FIG_SUBDIR = Path("fig/runge/pairwise_mlp_tm_ei_path_effects")
DEFAULT_COMPONENT_SCORES = Path("results/runge/2015_gateways/component_weekly_scores.csv")

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


@dataclass(frozen=True)
class PairwiseMlpEiConfig:
    component_scores: Path = DEFAULT_COMPONENT_SCORES
    linear_coefficients: Path | None = None
    output_dir: Path = Path(".")
    lag: int = 4
    horizon: int = 1
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.05
    epochs: int = 160
    learning_rate: float = 1.0e-3
    batch_size: int = 256
    weight_decay: float = 1.0e-4
    ridge_alpha: float = 1.0
    ensemble_ridge_alphas: tuple[float, ...] = ()
    linear_blend_grid_steps: int = 0
    use_linear_skip: bool = True
    freeze_linear_skip: bool = True
    residual_shrinkage: bool = True
    residual_gamma_min: float = -0.5
    residual_gamma_max: float = 0.5
    residual_gamma_steps: int = 101
    early_stopping_patience: int = 40
    min_delta: float = 1.0e-5
    scheduler_patience: int = 12
    gradient_clip_norm: float = 5.0
    intervention_samples: int = 4096
    bins: int = 8
    ei_estimator: str = "discrete"
    gateway_mode: str = "pairwise"
    graph_sparsify: str = "source_topk"
    graph_topk: int = 5
    graph_quantile: float = 0.95
    path_alpha: float = 0.8
    quantile_low: float = 0.05
    quantile_high: float = 0.95
    source_mode: str = "latest"
    seed: int = 42
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    force_retrain: bool = False


class ResidualMLPTransitionNet:
    def __new__(
        cls,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        *,
        num_layers: int,
        dropout: float,
        use_linear_skip: bool,
    ):
        import torch

        class _Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.skip = torch.nn.Linear(input_dim, output_dim) if use_linear_skip else None
                self.residual_scale = 1.0
                layers: list[torch.nn.Module] = []
                current_dim = input_dim
                for _ in range(max(1, int(num_layers))):
                    layers.extend(
                        [
                            torch.nn.Linear(current_dim, hidden_dim),
                            torch.nn.LayerNorm(hidden_dim),
                            torch.nn.SiLU(),
                            torch.nn.Dropout(float(dropout)),
                        ]
                    )
                    current_dim = hidden_dim
                self.residual = torch.nn.Sequential(*layers)
                self.residual_head = torch.nn.Linear(hidden_dim, output_dim)
                if self.skip is not None:
                    torch.nn.init.zeros_(self.residual_head.weight)
                    torch.nn.init.zeros_(self.residual_head.bias)

            def forward(self, x):
                output = self.residual_head(self.residual(x))
                output = output * float(self.residual_scale)
                if self.skip is not None:
                    output = output + self.skip(x)
                return output

        return _Net()


class MLPTransition:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        *,
        num_layers: int = 3,
        dropout: float = 0.05,
        use_linear_skip: bool = True,
    ):
        self.net = ResidualMLPTransitionNet(
            input_dim,
            output_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_linear_skip=use_linear_skip,
        )


class AveragedTransition:
    def __init__(self, models: Sequence[object]) -> None:
        self.models = list(models)
        self.training_summary = {
            "ensemble_size": len(self.models),
            "members": [getattr(model, "training_summary", {}) for model in self.models],
        }

    def eval(self) -> None:
        for model in self.models:
            model.eval()

    def __call__(self, tensor):
        if not self.models:
            raise ValueError("AveragedTransition requires at least one model.")
        prediction = self.models[0](tensor)
        for model in self.models[1:]:
            prediction = prediction + model(tensor)
        return prediction / len(self.models)


class WeightedAveragedTransition:
    def __init__(
        self,
        models: Sequence[object],
        weights: Sequence[float],
        *,
        training_summary: dict[str, object] | None = None,
    ) -> None:
        if len(models) != len(weights):
            raise ValueError("WeightedAveragedTransition requires matching models and weights.")
        if not models:
            raise ValueError("WeightedAveragedTransition requires at least one model.")
        total = float(np.sum(weights))
        if not np.isfinite(total) or abs(total) < 1.0e-12:
            raise ValueError("WeightedAveragedTransition weights must have a nonzero finite sum.")
        self.models = list(models)
        self.weights = [float(weight) / total for weight in weights]
        self.training_summary = training_summary or {
            "ensemble_size": len(self.models),
            "weights": list(self.weights),
            "members": [getattr(model, "training_summary", {}) for model in self.models],
        }

    def eval(self) -> None:
        for model in self.models:
            model.eval()

    def __call__(self, tensor):
        prediction = self.models[0](tensor) * self.weights[0]
        for model, weight in zip(self.models[1:], self.weights[1:]):
            prediction = prediction + model(tensor) * weight
        return prediction


def build_scaled_ridge_transition(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    scalers: dict[str, np.ndarray],
    *,
    ridge_alpha: float,
) -> object:
    import torch

    x_train, y_train = splits["train"]
    x_scaled = (np.asarray(x_train, dtype=np.float32) - scalers["x_mean"]) / scalers["x_std"]
    y_scaled = (np.asarray(y_train, dtype=np.float32) - scalers["y_mean"]) / scalers["y_std"]
    weight, bias = fit_ridge_linear_map(x_scaled, y_scaled, alpha=float(ridge_alpha))

    class _ScaledRidgeTransition(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(x_scaled.shape[1], y_scaled.shape[1])
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor(weight, dtype=self.linear.weight.dtype))
                self.linear.bias.copy_(torch.tensor(bias, dtype=self.linear.bias.dtype))
            for parameter in self.linear.parameters():
                parameter.requires_grad = False
            self.training_summary = {
                "type": "scaled_ridge_transition",
                "ridge_alpha": float(ridge_alpha),
            }

        def forward(self, tensor):
            return self.linear(tensor)

    return _ScaledRidgeTransition()


def select_linear_blend_weight(
    mlp_model: object,
    ridge_model: object,
    scalers: dict[str, np.ndarray],
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    names: Sequence[str],
    *,
    grid_steps: int,
) -> dict[str, float]:
    steps = max(2, int(grid_steps))
    x_val, y_val = splits["val"]
    mlp_pred = predict_mlp(mlp_model, scalers, x_val)
    ridge_pred = predict_mlp(ridge_model, scalers, x_val)
    best: dict[str, float] | None = None
    for weight in np.linspace(0.0, 1.0, steps):
        pred = float(weight) * mlp_pred + (1.0 - float(weight)) * ridge_pred
        row = regression_metrics(pred, y_val, names).query("component == 'overall'").iloc[0]
        candidate = {
            "mlp_weight": float(weight),
            "ridge_weight": float(1.0 - float(weight)),
            "val_rmse": float(row["rmse"]),
            "val_mae": float(row["mae"]),
            "val_corr": float(row["corr"]),
        }
        if best is None or candidate["val_rmse"] < best["val_rmse"]:
            best = candidate
    assert best is not None
    return best


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    root_candidate = _repo_root() / candidate
    return root_candidate if root_candidate.exists() else candidate.resolve()


def _jsonable_config(config: PairwiseMlpEiConfig) -> dict[str, object]:
    data = asdict(config)
    data["component_scores"] = str(config.component_scores)
    data["linear_coefficients"] = str(config.linear_coefficients) if config.linear_coefficients is not None else None
    data["output_dir"] = str(config.output_dir)
    return data


def _frame_content_hash(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update(hashed.tobytes())
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    return digest.hexdigest()[:16]


def _model_config_hash(config: PairwiseMlpEiConfig, *, n_components: int, n_rows: int, data_hash: str) -> str:
    payload = _jsonable_config(config).copy()
    for key in (
        "force_retrain",
        "intervention_samples",
        "bins",
        "ei_estimator",
        "gateway_mode",
        "graph_sparsify",
        "graph_topk",
        "graph_quantile",
        "path_alpha",
        "quantile_low",
        "quantile_high",
        "source_mode",
        "linear_coefficients",
    ):
        payload.pop(key, None)
    payload["n_components"] = int(n_components)
    payload["n_rows"] = int(n_rows)
    payload["data_hash"] = str(data_hash)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_component_scores(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(_resolve_path(path))
    if "time" in frame.columns:
        frame["time"] = pd.to_datetime(frame["time"])
        frame = frame.sort_values("time").set_index("time")
    numeric = frame.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] < 2:
        raise ValueError("component score file must contain at least two numeric component columns.")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if len(numeric) < 10:
        raise ValueError("component score file has too few complete rows.")
    return numeric.astype(float)


def build_lagged_dataset(frame: pd.DataFrame, *, lag: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    if lag < 1:
        raise ValueError("lag must be positive.")
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    values = frame.to_numpy(dtype=float)
    n_time, n_components = values.shape
    n_samples = n_time - int(lag) - int(horizon) + 1
    if n_samples <= 0:
        raise ValueError("not enough rows for requested lag and horizon.")
    features = np.empty((n_samples, int(lag) * n_components), dtype=float)
    targets = np.empty((n_samples, n_components), dtype=float)
    for sample_idx in range(n_samples):
        features[sample_idx] = values[sample_idx : sample_idx + int(lag)].reshape(-1)
        targets[sample_idx] = values[sample_idx + int(lag) + int(horizon) - 1]
    return features, targets


def split_temporal_arrays(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    n = len(features)
    if not 0.1 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.1, 1.0).")
    if not 0.0 <= val_fraction < 0.8:
        raise ValueError("val_fraction must be in [0.0, 0.8).")
    train_end = max(1, min(n - 2, int(round(n * train_fraction))))
    val_end = max(train_end + 1, min(n - 1, int(round(n * (train_fraction + val_fraction)))))
    return {
        "train": (features[:train_end], targets[:train_end]),
        "val": (features[train_end:val_end], targets[train_end:val_end]),
        "test": (features[val_end:], targets[val_end:]),
    }


def _standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std > 1.0e-8, std, 1.0)
    return (values - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def fit_ridge_linear_map(x: np.ndarray, y: np.ndarray, *, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.ndim != 2 or y_values.ndim != 2:
        raise ValueError("x and y must be two-dimensional arrays.")
    if x_values.shape[0] != y_values.shape[0]:
        raise ValueError("x and y must contain the same number of rows.")
    design = np.concatenate([x_values, np.ones((x_values.shape[0], 1), dtype=np.float64)], axis=1)
    if float(alpha) <= 0.0:
        coefficients, *_ = np.linalg.lstsq(design, y_values, rcond=None)
    else:
        penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
        penalty[-1, -1] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y_values)
    weight = coefficients[:-1].T.astype(np.float32)
    bias = coefficients[-1].astype(np.float32)
    return weight, bias


def initialize_linear_skip(model: object, x_scaled: np.ndarray, y_scaled: np.ndarray, *, alpha: float) -> None:
    import torch

    skip = getattr(model, "skip", None)
    if skip is None:
        return
    weight, bias = fit_ridge_linear_map(x_scaled, y_scaled, alpha=float(alpha))
    with torch.no_grad():
        skip.weight.copy_(torch.tensor(weight, dtype=skip.weight.dtype))
        skip.bias.copy_(torch.tensor(bias, dtype=skip.bias.dtype))


def predict_ridge_linear_baseline(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    ridge_alpha: float,
) -> dict[str, np.ndarray]:
    x_train, y_train = splits["train"]
    x_train_scaled, x_mean, x_std = _standardize(x_train, x_train)
    y_train_scaled, y_mean, y_std = _standardize(y_train, y_train)
    weight, bias = fit_ridge_linear_map(x_train_scaled, y_train_scaled, alpha=float(ridge_alpha))
    predictions: dict[str, np.ndarray] = {}
    for split_name, (features, _) in splits.items():
        x_scaled = (np.asarray(features, dtype=np.float32) - x_mean) / x_std
        pred_scaled = x_scaled @ weight.T + bias
        predictions[split_name] = pred_scaled * y_std + y_mean
    return predictions


def train_or_load_mlp(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    config: PairwiseMlpEiConfig,
    model_path: Path,
    *,
    config_hash: str,
) -> tuple[object, dict[str, np.ndarray], list[float], bool]:
    import torch

    x_train, y_train = splits["train"]
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]

    if model_path.exists() and not config.force_retrain:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        if payload.get("config_hash") == config_hash:
            model = MLPTransition(
                input_dim,
                output_dim,
                int(payload["hidden_dim"]),
                num_layers=int(payload.get("num_layers", config.num_layers)),
                dropout=float(payload.get("dropout", config.dropout)),
                use_linear_skip=bool(payload.get("use_linear_skip", config.use_linear_skip)),
            )
            model.net.load_state_dict(payload["model_state_dict"])
            cached_loss_history = list(payload.get("loss_history", []))
            model.net.training_summary = {
                "train_loss_history": cached_loss_history,
                "val_loss_history": list(payload.get("val_loss_history", [])),
                "best_epoch": int(payload.get("best_epoch", len(cached_loss_history))),
                "best_train_loss": float(payload.get("best_train_loss", cached_loss_history[-1] if cached_loss_history else float("nan"))),
                "best_val_loss": float(payload.get("best_val_loss", float("nan"))),
                "stopped_early": bool(payload.get("stopped_early", False)),
                "residual_scale": float(payload.get("residual_scale", 1.0)),
            }
            model.net.residual_scale = float(payload.get("residual_scale", 1.0))
            scalers = {
                "x_mean": np.asarray(payload["x_mean"], dtype=np.float32),
                "x_std": np.asarray(payload["x_std"], dtype=np.float32),
                "y_mean": np.asarray(payload["y_mean"], dtype=np.float32),
                "y_std": np.asarray(payload["y_std"], dtype=np.float32),
            }
            return model.net, scalers, list(payload.get("loss_history", [])), True

    torch.manual_seed(int(config.seed))
    torch.set_num_threads(1)
    x_train_scaled, x_mean, x_std = _standardize(x_train, x_train)
    y_train_scaled, y_mean, y_std = _standardize(y_train, y_train)
    x_val, y_val = splits["val"]
    x_val_scaled = (np.asarray(x_val, dtype=np.float32) - x_mean) / x_std
    y_val_scaled = (np.asarray(y_val, dtype=np.float32) - y_mean) / y_std
    model = MLPTransition(
        input_dim,
        output_dim,
        int(config.hidden_dim),
        num_layers=int(config.num_layers),
        dropout=float(config.dropout),
        use_linear_skip=bool(config.use_linear_skip),
    )
    initialize_linear_skip(model.net, x_train_scaled, y_train_scaled, alpha=float(config.ridge_alpha))
    if bool(config.freeze_linear_skip):
        skip = getattr(model.net, "skip", None)
        if skip is not None:
            for parameter in skip.parameters():
                parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.net.parameters() if parameter.requires_grad],
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, int(config.scheduler_patience)),
    )
    loss_fn = torch.nn.MSELoss()
    x_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
    batch_size = max(1, min(int(config.batch_size), len(x_tensor)))
    generator = torch.Generator().manual_seed(int(config.seed))
    loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_state = {key: value.detach().clone() for key, value in model.net.state_dict().items()}
    best_epoch = 0
    best_train_loss = float("inf")
    model.net.eval()
    with torch.no_grad():
        best_val_loss = float(loss_fn(model.net(x_val_tensor), y_val_tensor).item())
    epochs_without_improvement = 0
    stopped_early = False
    for _ in range(int(config.epochs)):
        order = torch.randperm(len(x_tensor), generator=generator)
        losses: list[float] = []
        model.net.train()
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model.net(x_tensor[batch]), y_tensor[batch])
            loss.backward()
            if float(config.gradient_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.net.parameters(), float(config.gradient_clip_norm))
            optimizer.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses))
        loss_history.append(train_loss)
        model.net.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model.net(x_val_tensor), y_val_tensor).item())
        val_loss_history.append(val_loss)
        scheduler.step(val_loss)
        if val_loss < best_val_loss - float(config.min_delta):
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = len(loss_history)
            best_state = {key: value.detach().clone() for key, value in model.net.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if int(config.early_stopping_patience) > 0 and epochs_without_improvement >= int(config.early_stopping_patience):
            stopped_early = True
            break

    final_state = {key: value.detach().clone() for key, value in model.net.state_dict().items()}
    residual_scale = 1.0
    if bool(config.residual_shrinkage) and getattr(model.net, "skip", None) is not None:
        model.net.load_state_dict(final_state)
        gamma_values = np.linspace(
            float(config.residual_gamma_min),
            float(config.residual_gamma_max),
            max(2, int(config.residual_gamma_steps)),
        )
        best_gamma_val = float("inf")
        best_gamma = 0.0
        model.net.eval()
        with torch.no_grad():
            for gamma in gamma_values:
                model.net.residual_scale = float(gamma)
                gamma_loss = float(loss_fn(model.net(x_val_tensor), y_val_tensor).item())
                if gamma_loss < best_gamma_val:
                    best_gamma_val = gamma_loss
                    best_gamma = float(gamma)
        if best_gamma_val < best_val_loss:
            best_state = final_state
            best_val_loss = best_gamma_val
            best_epoch = len(loss_history)
            best_train_loss = loss_history[-1] if loss_history else best_train_loss
            residual_scale = best_gamma

    model.net.load_state_dict(best_state)
    model.net.residual_scale = float(residual_scale)
    model.net.training_summary = {
        "train_loss_history": loss_history,
        "val_loss_history": val_loss_history,
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "initial_val_loss": float(val_loss_history[0]) if val_loss_history else float(best_val_loss),
        "stopped_early": bool(stopped_early),
        "early_stopping_patience": int(config.early_stopping_patience),
        "residual_scale": float(residual_scale),
    }

    scalers = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config_hash": config_hash,
            "hidden_dim": int(config.hidden_dim),
            "num_layers": int(config.num_layers),
            "dropout": float(config.dropout),
            "use_linear_skip": bool(config.use_linear_skip),
            "model_state_dict": model.net.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "loss_history": loss_history,
            "val_loss_history": val_loss_history,
            "best_epoch": int(best_epoch),
            "best_train_loss": float(best_train_loss),
            "best_val_loss": float(best_val_loss),
            "stopped_early": bool(stopped_early),
            "residual_scale": float(residual_scale),
        },
        model_path,
    )
    return model.net, scalers, loss_history, False


def predict_mlp(model: object, scalers: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    import torch

    values = np.asarray(features, dtype=np.float32)
    scaled = (values - scalers["x_mean"]) / scalers["x_std"]
    model.eval()
    with torch.no_grad():
        pred_tensor = model(torch.tensor(scaled.tolist(), dtype=torch.float32)).cpu()
        pred = np.asarray(pred_tensor.tolist(), dtype=np.float32)
    return pred * scalers["y_std"] + scalers["y_mean"]


def regression_metrics(pred: np.ndarray, target: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    rows = []
    for idx, name in enumerate(names):
        y = np.asarray(target[:, idx], dtype=float)
        p = np.asarray(pred[:, idx], dtype=float)
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        mae = float(np.mean(np.abs(y - p)))
        corr = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 1.0e-12 and np.std(p) > 1.0e-12 else 0.0
        rows.append({"component": name, "rmse": rmse, "mae": mae, "corr": corr})
    overall = {
        "component": "overall",
        "rmse": float(np.sqrt(np.mean((np.asarray(target) - np.asarray(pred)) ** 2))),
        "mae": float(np.mean(np.abs(np.asarray(target) - np.asarray(pred)))),
        "corr": float(np.corrcoef(np.asarray(target).reshape(-1), np.asarray(pred).reshape(-1))[0, 1]),
    }
    return pd.DataFrame([overall, *rows])


def _discretize(values: np.ndarray, bins: int) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(-1)
    if len(flat) == 0:
        return np.asarray([], dtype=int)
    unique = np.unique(np.round(flat, decimals=10))
    if len(unique) <= 1:
        return np.zeros(len(flat), dtype=int)
    if 1 < len(unique) <= bins:
        mapping = {value: idx for idx, value in enumerate(sorted(unique))}
        rounded = np.round(flat, decimals=10)
        return np.asarray([mapping[value] for value in rounded], dtype=int)
    quantiles = np.quantile(flat, np.linspace(0.0, 1.0, int(bins) + 1))
    edges = np.unique(quantiles)
    if len(edges) <= 2:
        ranks = pd.Series(flat).rank(method="first").to_numpy()
        edges = np.quantile(ranks, np.linspace(0.0, 1.0, int(bins) + 1))
        return np.clip(np.digitize(ranks, edges[1:-1], right=False), 0, int(bins) - 1).astype(int)
    return np.clip(np.digitize(flat, edges[1:-1], right=False), 0, len(edges) - 2).astype(int)


def _state_codes(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=int)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    _, inverse = np.unique(values, axis=0, return_inverse=True)
    return inverse.astype(int)


def _entropy_bits(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    probs = probs[probs > 0.0]
    if len(probs) == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


def discrete_effective_information(source_states: np.ndarray, target_states: np.ndarray) -> float:
    source_inverse = _state_codes(source_states)
    target_inverse = _state_codes(target_states)
    n_source = int(source_inverse.max()) + 1
    n_target = int(target_inverse.max()) + 1
    counts = np.zeros((n_source, n_target), dtype=float)
    for source, target in zip(source_inverse, target_inverse):
        counts[int(source), int(target)] += 1.0
    row_totals = counts.sum(axis=1)
    observed = row_totals > 0.0
    if int(observed.sum()) < 2:
        return 0.0
    conditional = counts[observed] / row_totals[observed, None]
    target_probs = conditional.mean(axis=0)
    return float(_entropy_bits(target_probs) - np.mean([_entropy_bits(row) for row in conditional]))


def sample_max_entropy_features(
    train_features: np.ndarray,
    *,
    n_components: int,
    lag: int,
    samples: int,
    low_q: float,
    high_q: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 1009)
    features = np.empty((int(samples), int(lag) * int(n_components)), dtype=float)
    for col in range(features.shape[1]):
        values = np.asarray(train_features[:, col], dtype=float)
        low = float(np.quantile(values, float(low_q)))
        high = float(np.quantile(values, float(high_q)))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.min(values))
            high = float(np.max(values))
        if high <= low:
            features[:, col] = low
        else:
            features[:, col] = rng.uniform(low, high, size=int(samples))
    return features


def estimate_pairwise_ei_matrix(
    model: object,
    scalers: dict[str, np.ndarray],
    train_features: np.ndarray,
    *,
    n_components: int,
    lag: int,
    intervention_samples: int,
    bins: int,
    low_q: float,
    high_q: float,
    source_mode: str,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    if source_mode not in {"latest", "history"}:
        raise ValueError("source_mode must be 'latest' or 'history'.")
    intervention_features = sample_max_entropy_features(
        train_features,
        n_components=n_components,
        lag=lag,
        samples=intervention_samples,
        low_q=low_q,
        high_q=high_q,
        seed=seed,
    )
    predictions = predict_mlp(model, scalers, intervention_features)
    source_states = []
    for source in range(int(n_components)):
        lagged_cols = [lag_idx * int(n_components) + source for lag_idx in range(int(lag))]
        if source_mode == "latest":
            source_states.append(_discretize(intervention_features[:, lagged_cols[-1]], int(bins)))
        else:
            source_states.append(np.column_stack([_discretize(intervention_features[:, col], int(bins)) for col in lagged_cols]))
    target_states = [_discretize(predictions[:, target], int(bins)) for target in range(int(n_components))]

    matrix = np.zeros((int(n_components), int(n_components)), dtype=float)
    rows = []
    for source in range(int(n_components)):
        for target in range(int(n_components)):
            ei = discrete_effective_information(source_states[source], target_states[target])
            matrix[source, target] = float(ei)
            rows.append(
                {
                    "source": f"component_{source + 1:02d}",
                    "target": f"component_{target + 1:02d}",
                    "source_index": source,
                    "target_index": target,
                    "ei": float(ei),
                }
            )
    states = pd.DataFrame(rows).sort_values("ei", ascending=False).reset_index(drop=True)
    return matrix, states


def estimate_pairwise_tm_ei_matrix(
    model: object,
    scalers: dict[str, np.ndarray],
    train_features: np.ndarray,
    *,
    n_components: int,
    lag: int,
    intervention_samples: int,
    low_q: float,
    high_q: float,
    source_mode: str,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from exp.TM.transport_map_density import estimate_mutual_information_transport_map

    if source_mode not in {"latest", "history"}:
        raise ValueError("source_mode must be 'latest' or 'history'.")
    intervention_features = sample_max_entropy_features(
        train_features,
        n_components=n_components,
        lag=lag,
        samples=intervention_samples,
        low_q=low_q,
        high_q=high_q,
        seed=seed,
    )
    predictions = predict_mlp(model, scalers, intervention_features)

    source_states = []
    for source in range(int(n_components)):
        lagged_cols = [lag_idx * int(n_components) + source for lag_idx in range(int(lag))]
        if source_mode == "latest":
            source_states.append(intervention_features[:, [lagged_cols[-1]]])
        else:
            source_states.append(intervention_features[:, lagged_cols])

    matrix = np.zeros((int(n_components), int(n_components)), dtype=float)
    rows = []
    for source in range(int(n_components)):
        for target in range(int(n_components)):
            target_state = predictions[:, [target]]
            summary = estimate_mutual_information_transport_map(source_states[source], target_state)
            ei = max(0.0, float(summary["mi_hat"]))
            matrix[source, target] = ei
            rows.append(
                {
                    "source": f"component_{source + 1:02d}",
                    "target": f"component_{target + 1:02d}",
                    "source_index": source,
                    "target_index": target,
                    "ei": ei,
                    "bias_correction": float(summary["bias_correction"]),
                }
            )
    states = pd.DataFrame(rows).sort_values("ei", ascending=False).reset_index(drop=True)
    return matrix, states


def gateway_scores_from_matrix(matrix: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square.")
    n = values.shape[0]
    off_diag = values.copy()
    np.fill_diagonal(off_diag, np.nan)
    gateway = np.nanmean(off_diag, axis=1)
    susceptibility = np.nanmean(off_diag, axis=0)
    rows = []
    out_order = pd.Series(gateway).rank(ascending=False, method="min").astype(int).to_numpy()
    in_order = pd.Series(susceptibility).rank(ascending=False, method="min").astype(int).to_numpy()
    for idx, name in enumerate(names):
        rows.append(
            {
                "component": name,
                "component_index": idx,
                "gateway_ei": float(gateway[idx]),
                "susceptibility_ei": float(susceptibility[idx]),
                "self_memory_ei": float(values[idx, idx]),
                "out_rank": int(out_order[idx]),
                "in_rank": int(in_order[idx]),
            }
        )
    return pd.DataFrame(rows).sort_values("gateway_ei", ascending=False).reset_index(drop=True)


def compare_ei_to_linear_coefficients(
    ei_matrix: np.ndarray,
    linear_matrix: np.ndarray,
    names: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    ei = np.asarray(ei_matrix, dtype=float)
    linear = np.asarray(linear_matrix, dtype=float)
    if ei.ndim != 2 or ei.shape[0] != ei.shape[1]:
        raise ValueError("ei_matrix must be square.")
    if linear.shape != ei.shape:
        raise ValueError("linear coefficient matrix must have the same shape as the EI matrix.")
    if len(names) != ei.shape[0]:
        raise ValueError("names length must match matrix dimensions.")

    rows = []
    for source in range(ei.shape[0]):
        for target in range(ei.shape[1]):
            coefficient = float(linear[source, target])
            rows.append(
                {
                    "source": names[source],
                    "target": names[target],
                    "source_index": source,
                    "target_index": target,
                    "pairwise_ei": float(ei[source, target]),
                    "linear_coefficient": coefficient,
                    "abs_linear_coefficient": abs(coefficient),
                    "same_nonzero_support": bool((ei[source, target] > 0.0) == (abs(coefficient) > 0.0)),
                }
            )
    frame = pd.DataFrame(rows)
    off_diag = frame["source_index"] != frame["target_index"]
    summary: dict[str, float | int] = {
        "n_elements": int(len(frame)),
        "n_off_diagonal_elements": int(off_diag.sum()),
        "ei_nonzero_elements": int((frame["pairwise_ei"] > 0.0).sum()),
        "linear_nonzero_elements": int((frame["abs_linear_coefficient"] > 0.0).sum()),
        "support_match_fraction": float(frame["same_nonzero_support"].mean()) if len(frame) else 0.0,
        "off_diagonal_support_match_fraction": float(frame.loc[off_diag, "same_nonzero_support"].mean()) if off_diag.any() else 0.0,
    }
    if len(frame) >= 2:
        summary["pearson_abs_linear_vs_ei"] = float(frame["pairwise_ei"].corr(frame["abs_linear_coefficient"], method="pearson"))
        summary["spearman_abs_linear_vs_ei"] = float(frame["pairwise_ei"].corr(frame["abs_linear_coefficient"], method="spearman"))
    if off_diag.sum() >= 2:
        summary["off_diagonal_pearson_abs_linear_vs_ei"] = float(
            frame.loc[off_diag, "pairwise_ei"].corr(frame.loc[off_diag, "abs_linear_coefficient"], method="pearson")
        )
        summary["off_diagonal_spearman_abs_linear_vs_ei"] = float(
            frame.loc[off_diag, "pairwise_ei"].corr(frame.loc[off_diag, "abs_linear_coefficient"], method="spearman")
        )
    return frame.sort_values("pairwise_ei", ascending=False).reset_index(drop=True), summary


def load_linear_coefficient_matrix(path: str | Path, names: Sequence[str]) -> np.ndarray:
    frame = pd.read_csv(_resolve_path(path), index_col=0)
    if list(frame.index) == list(names) and list(frame.columns) == list(names):
        return frame.to_numpy(dtype=float)
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape == (len(names), len(names)):
        return numeric.to_numpy(dtype=float)
    raise ValueError("linear coefficient matrix must be an N x N CSV aligned with component names.")


PAPER_COMPONENT_LABEL_MAP: dict[int, int] = {
    7: 18,
    18: 7,
    8: 26,
    26: 8,
    21: 48,
    48: 21,
}


def paper_component_label(index: object) -> str:
    paper_index = PAPER_COMPONENT_LABEL_MAP.get(int(index), int(index))
    return f"No.{paper_index}"


def parse_float_tuple(text: str | Sequence[float] | None) -> tuple[float, ...]:
    if text is None:
        return ()
    if isinstance(text, str):
        if not text.strip():
            return ()
        return tuple(float(part.strip()) for part in text.split(",") if part.strip())
    return tuple(float(value) for value in text)


def _finite_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


@dataclass(frozen=True)
class EiPathEffects:
    direct_effects: pd.DataFrame
    total_effects: pd.DataFrame
    path_effects: pd.DataFrame
    gateway_scores: pd.DataFrame
    mediator_scores: pd.DataFrame
    scaled_direct_matrix: np.ndarray
    total_matrix: np.ndarray
    scale_factor: float


def sparsify_ei_graph(
    matrix: np.ndarray,
    *,
    mode: str = "source_topk",
    topk: int = 5,
    quantile: float = 0.95,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square.")
    if mode not in {"none", "source_topk", "target_topk", "bidirectional_topk", "global_quantile"}:
        raise ValueError("mode must be one of none, source_topk, target_topk, bidirectional_topk, global_quantile.")
    sparse = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).copy()
    sparse = np.maximum(sparse, 0.0)
    np.fill_diagonal(sparse, 0.0)
    if mode == "none":
        return sparse
    if mode in {"source_topk", "bidirectional_topk"}:
        keep = np.zeros_like(sparse, dtype=bool)
        k = max(0, min(int(topk), sparse.shape[1] - 1))
        if k == 0:
            return np.zeros_like(sparse)
        for source in range(sparse.shape[0]):
            candidates = np.flatnonzero(sparse[source] > 0.0)
            if len(candidates) == 0:
                continue
            order = candidates[np.argsort(sparse[source, candidates])[::-1][:k]]
            keep[source, order] = True
        if mode == "source_topk":
            return np.where(keep, sparse, 0.0)
    if mode in {"target_topk", "bidirectional_topk"}:
        if mode == "target_topk":
            keep = np.zeros_like(sparse, dtype=bool)
        k = max(0, min(int(topk), sparse.shape[0] - 1))
        if k == 0:
            return np.zeros_like(sparse)
        for target in range(sparse.shape[1]):
            candidates = np.flatnonzero(sparse[:, target] > 0.0)
            if len(candidates) == 0:
                continue
            order = candidates[np.argsort(sparse[candidates, target])[::-1][:k]]
            keep[order, target] = True
        return np.where(keep, sparse, 0.0)
    off_diag = sparse[~np.eye(sparse.shape[0], dtype=bool)]
    positive = off_diag[off_diag > 0.0]
    if len(positive) == 0:
        return np.zeros_like(sparse)
    threshold = float(np.quantile(positive, float(quantile)))
    return np.where(sparse >= threshold, sparse, 0.0)


def compute_ei_path_effects(
    direct_matrix: np.ndarray,
    names: Sequence[str],
    *,
    path_alpha: float = 0.8,
    max_path_length: int | None = None,
) -> EiPathEffects:
    direct = np.asarray(direct_matrix, dtype=float)
    if direct.ndim != 2 or direct.shape[0] != direct.shape[1]:
        raise ValueError("direct_matrix must be square.")
    n = direct.shape[0]
    if len(names) != n:
        raise ValueError("names length must match direct_matrix dimensions.")
    if not 0.0 < float(path_alpha) <= 1.0:
        raise ValueError("path_alpha must be in (0, 1].")

    direct = np.nan_to_num(direct, nan=0.0, posinf=0.0, neginf=0.0).copy()
    direct = np.maximum(direct, 0.0)
    np.fill_diagonal(direct, 0.0)
    if np.any(direct):
        radius = float(np.max(np.abs(np.linalg.eigvals(direct))))
    else:
        radius = 0.0
    scale_factor = float(path_alpha) / radius if radius > 1.0e-12 and radius > float(path_alpha) else 1.0
    scaled = direct * scale_factor

    total = np.zeros_like(scaled)
    power = scaled.copy()
    path_count = int(max_path_length) if max_path_length is not None else n
    for _ in range(max(1, path_count)):
        total += power
        power = power @ scaled

    direct_rows = []
    total_rows = []
    path_rows = []
    for source in range(n):
        for target in range(n):
            if source == target:
                continue
            if scaled[source, target] > 0.0:
                direct_rows.append(
                    {
                        "source": names[source],
                        "target": names[target],
                        "source_index": source,
                        "target_index": target,
                        "direct_effect": float(scaled[source, target]),
                        "raw_ei": float(direct[source, target]),
                    }
                )
            if total[source, target] > 1.0e-12:
                total_rows.append(
                    {
                        "source": names[source],
                        "target": names[target],
                        "source_index": source,
                        "target_index": target,
                        "total_effect": float(total[source, target]),
                    }
                )
            for mediator in range(n):
                if mediator in (source, target):
                    continue
                mediated = float(scaled[source, mediator] * total[mediator, target])
                if mediated > 1.0e-12:
                    path_rows.append(
                        {
                            "source": names[source],
                            "mediator": names[mediator],
                            "target": names[target],
                            "source_index": source,
                            "mediator_index": mediator,
                            "target_index": target,
                            "amce": mediated,
                        }
                    )

    gateway_rows = []
    mediator_rows = []
    denom = max(1, n - 1)
    mediator_denom = max(1, (n - 1) * (n - 2))
    mediated_total = float(sum(row["amce"] for row in path_rows))
    for idx, name in enumerate(names):
        ace = float(np.sum(total[idx, :]) / denom)
        acs = float(np.sum(total[:, idx]) / denom)
        mediated = float(sum(row["amce"] for row in path_rows if int(row["mediator_index"]) == idx) / mediator_denom)
        gateway_rows.append(
            {
                "component": name,
                "component_index": idx,
                "ace": ace,
                "acs": acs,
                "direct_out_strength": float(np.sum(scaled[idx, :])),
                "direct_in_strength": float(np.sum(scaled[:, idx])),
                "out_rank": 0,
                "in_rank": 0,
            }
        )
        mediator_rows.append(
            {
                "component": name,
                "component_index": idx,
                "amce": mediated,
                "mediated_fraction": float((mediated * mediator_denom) / mediated_total) if mediated_total > 0.0 else 0.0,
            }
        )
    gateway_scores = pd.DataFrame(gateway_rows)
    gateway_scores["out_rank"] = gateway_scores["ace"].rank(ascending=False, method="min").astype(int)
    gateway_scores["in_rank"] = gateway_scores["acs"].rank(ascending=False, method="min").astype(int)
    gateway_scores = gateway_scores.sort_values("ace", ascending=False).reset_index(drop=True)
    mediator_scores = pd.DataFrame(mediator_rows).sort_values("amce", ascending=False).reset_index(drop=True)
    return EiPathEffects(
        direct_effects=pd.DataFrame(direct_rows),
        total_effects=pd.DataFrame(total_rows),
        path_effects=pd.DataFrame(path_rows),
        gateway_scores=gateway_scores,
        mediator_scores=mediator_scores,
        scaled_direct_matrix=scaled,
        total_matrix=total,
        scale_factor=scale_factor,
    )


def save_heatmap(matrix: np.ndarray, output_path: str | Path, *, title: str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("target component No.")
    ax.set_ylabel("source component No.")
    step = max(1, matrix.shape[0] // 12)
    ticks = np.arange(0, matrix.shape[0], step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(int(t)) for t in ticks], rotation=45, ha="right")
    ax.set_yticklabels([str(int(t)) for t in ticks])
    fig.colorbar(image, ax=ax, label="pairwise EI (bits)")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_gateway_ranking(scores: pd.DataFrame, output_path: str | Path, *, title: str = "Pairwise MLP-EI gateway ranking") -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = scores.head(20).copy()
    x = np.arange(len(plot_frame))
    fig, ax = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
    ax.bar(x - 0.18, plot_frame["gateway_ei"], width=0.36, label="Gateway EI", color="#4c78a8")
    ax.bar(x + 0.18, plot_frame["susceptibility_ei"], width=0.36, label="Susceptibility EI", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels([paper_component_label(index) for index in plot_frame["component_index"]], rotation=50, ha="right")
    ax.set_ylabel("mean pairwise EI (bits)")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_path_gateway_ranking(scores: pd.DataFrame, output_path: str | Path, *, title: str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = scores.head(20).copy()
    x = np.arange(len(plot_frame))
    fig, ax = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
    ax.bar(x - 0.18, plot_frame["ace"], width=0.36, label="ACE", color="#4c78a8")
    ax.bar(x + 0.18, plot_frame["acs"], width=0.36, label="ACS", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels([paper_component_label(index) for index in plot_frame["component_index"]], rotation=50, ha="right")
    ax.set_ylabel("mean total effect")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_mediator_ranking(scores: pd.DataFrame, output_path: str | Path, *, title: str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = scores.head(20).copy()
    x = np.arange(len(plot_frame))
    fig, ax = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
    ax.bar(x, plot_frame["amce"], width=0.62, label="AMCE", color="#54a24b")
    ax.set_xticks(x)
    ax.set_xticklabels([paper_component_label(index) for index in plot_frame["component_index"]], rotation=50, ha="right")
    ax.set_ylabel("mean mediated effect")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def write_summary(path: str | Path, manifest: dict[str, object], scores: pd.DataFrame, metrics: pd.DataFrame) -> Path:
    output = Path(path)
    top = scores.head(10)
    lines = [
        "# Runge component pairwise MLP-EI gateway readout",
        "",
        "This experiment trains a cached MLP transition model on Varimax/PCA component score dynamics and reads out pairwise effective information under independent maximum-entropy interventions.",
        "",
        "## Scope",
        "",
        f"- Source unit: `{manifest['config']['source_mode']}` component state read from the lagged MLP input.",
        "- Target unit: one next-horizon component.",
        "- Synergy and mediator blocking are intentionally not computed in this run.",
        "",
        "## Run",
        "",
        f"- Components: {manifest['n_components']}",
        f"- Rows: {manifest['n_rows']}",
        f"- Lag: {manifest['config']['lag']}",
        f"- Horizon: {manifest['config']['horizon']}",
        f"- Intervention samples: {manifest['config']['intervention_samples']}",
        f"- EI estimator: {manifest['config']['ei_estimator']}",
        f"- MLP cache reused: {manifest['model_cache_reused']}",
        f"- Overall test RMSE: {float(metrics.iloc[0]['rmse']):.6g}",
        f"- Overall test corr: {float(metrics.iloc[0]['corr']):.6g}",
        "",
        "## Top gateways",
        "",
        "| paper_component | gateway_ei | susceptibility_ei | self_memory_ei |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in top.to_dict("records"):
        lines.append(
            f"| {paper_component_label(row['component_index'])} | {row['gateway_ei']:.6g} | {row['susceptibility_ei']:.6g} | {row['self_memory_ei']:.6g} |"
        )
    if "ei_linear_comparison" in manifest:
        comparison = manifest["ei_linear_comparison"]
        lines.extend(
            [
                "",
                "## Pairwise EI vs linear coefficient matrix",
                "",
                f"- Compared elements: {comparison['n_elements']}",
                f"- Off-diagonal elements: {comparison['n_off_diagonal_elements']}",
                f"- Support match fraction: {comparison['support_match_fraction']:.6g}",
                f"- Spearman(abs linear coefficient, EI): {comparison.get('spearman_abs_linear_vs_ei', float('nan')):.6g}",
                "- Per-element comparison: `ei_linear_coefficient_comparison.csv`.",
            ]
        )
    output.write_text("\n".join(lines) + "\n")
    return output


def write_path_summary(
    path: str | Path,
    manifest: dict[str, object],
    gateway_scores: pd.DataFrame,
    mediator_scores: pd.DataFrame,
    metrics: pd.DataFrame,
) -> Path:
    output = Path(path)
    top_gateways = gateway_scores.head(10)
    top_mediators = mediator_scores.head(10)
    lines = [
        "# Runge component MLP-TM-EI path-effect gateway readout",
        "",
        "This experiment keeps the MLP/TM-EI intervention readout, but computes Runge-style path effects on a sparsified EI causal graph.",
        "",
        "## Run",
        "",
        f"- Components: {manifest['n_components']}",
        f"- Rows: {manifest['n_rows']}",
        f"- Lag: {manifest['config']['lag']}",
        f"- Horizon: {manifest['config']['horizon']}",
        f"- EI estimator: {manifest['config']['ei_estimator']}",
        f"- Gateway mode: {manifest['config']['gateway_mode']}",
        f"- Graph sparsify: {manifest['config']['graph_sparsify']}",
        f"- Graph top-k: {manifest['config']['graph_topk']}",
        f"- Graph quantile: {manifest['config']['graph_quantile']}",
        f"- Path alpha: {manifest['config']['path_alpha']}",
        f"- Path scale factor: {manifest['path_scale_factor']:.6g}",
        f"- Direct EI edges: {manifest['n_direct_edges']}",
        f"- Total-effect paths: {manifest['n_total_effects']}",
        f"- Mediated paths: {manifest['n_mediated_paths']}",
        f"- MLP cache reused: {manifest['model_cache_reused']}",
        f"- Overall test RMSE: {float(metrics.iloc[0]['rmse']):.6g}",
        f"- Overall test corr: {float(metrics.iloc[0]['corr']):.6g}",
        "",
        "## Top gateways",
        "",
        "| paper_component | ace | acs | direct_out_strength | direct_in_strength |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in top_gateways.to_dict("records"):
        lines.append(
            f"| {paper_component_label(row['component_index'])} | {row['ace']:.6g} | {row['acs']:.6g} | {row['direct_out_strength']:.6g} | {row['direct_in_strength']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Top mediators",
            "",
            "| paper_component | amce | mediated_fraction |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in top_mediators.to_dict("records"):
        lines.append(f"| {paper_component_label(row['component_index'])} | {row['amce']:.6g} | {row['mediated_fraction']:.6g} |")
    if "ei_linear_comparison" in manifest:
        comparison = manifest["ei_linear_comparison"]
        lines.extend(
            [
                "",
                "## Pairwise EI vs linear coefficient matrix",
                "",
                f"- Compared elements: {comparison['n_elements']}",
                f"- Off-diagonal elements: {comparison['n_off_diagonal_elements']}",
                f"- Support match fraction: {comparison['support_match_fraction']:.6g}",
                f"- Spearman(abs linear coefficient, EI): {comparison.get('spearman_abs_linear_vs_ei', float('nan')):.6g}",
                "- Per-element comparison: `ei_linear_coefficient_comparison.csv`.",
            ]
        )
    output.write_text("\n".join(lines) + "\n")
    return output


def dependency_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "matplotlib", "torch"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def run(config: PairwiseMlpEiConfig) -> dict[str, Path]:
    component_scores_path = _resolve_path(config.component_scores)
    frame = load_component_scores(component_scores_path)
    names = list(frame.columns)
    features, targets = build_lagged_dataset(frame, lag=int(config.lag), horizon=int(config.horizon))
    splits = split_temporal_arrays(
        features,
        targets,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
    )

    if config.ei_estimator not in {"discrete", "tm"}:
        raise ValueError("ei_estimator must be 'discrete' or 'tm'.")
    if config.gateway_mode not in {"pairwise", "path_effect"}:
        raise ValueError("gateway_mode must be 'pairwise' or 'path_effect'.")
    if config.graph_sparsify not in {"none", "source_topk", "target_topk", "bidirectional_topk", "global_quantile"}:
        raise ValueError("graph_sparsify must be one of none, source_topk, target_topk, bidirectional_topk, global_quantile.")
    if config.gateway_mode == "path_effect" and config.ei_estimator != "tm":
        raise ValueError("path_effect gateway_mode currently requires --ei-estimator tm.")
    if config.gateway_mode == "path_effect":
        result_dir = Path(config.output_dir) / TM_PATH_RESULT_SUBDIR
        fig_dir = Path(config.output_dir) / TM_PATH_FIG_SUBDIR
    else:
        result_dir = Path(config.output_dir) / (TM_RESULT_SUBDIR if config.ei_estimator == "tm" else RESULT_SUBDIR)
        fig_dir = Path(config.output_dir) / (TM_FIG_SUBDIR if config.ei_estimator == "tm" else FIG_SUBDIR)
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_hash = _frame_content_hash(frame)
    config_hash = _model_config_hash(config, n_components=len(names), n_rows=len(frame), data_hash=data_hash)
    ensemble_alphas = tuple(config.ensemble_ridge_alphas) if config.ensemble_ridge_alphas else (float(config.ridge_alpha),)
    model_paths: list[Path] = []
    models: list[object] = []
    member_summaries: list[dict[str, object]] = []
    scalers = None
    cache_reused = True
    loss_history: list[float] = []
    model_dir = Path(config.output_dir) / RESULT_SUBDIR
    model_dir.mkdir(parents=True, exist_ok=True)
    for member_index, alpha in enumerate(ensemble_alphas):
        member_config = replace(config, ridge_alpha=float(alpha), ensemble_ridge_alphas=())
        member_hash = _model_config_hash(member_config, n_components=len(names), n_rows=len(frame), data_hash=data_hash)
        if len(ensemble_alphas) == 1:
            member_path = model_dir / "mlp_transition.pt"
        else:
            alpha_label = str(float(alpha)).replace("-", "m").replace(".", "p")
            member_path = model_dir / f"mlp_transition_alpha{alpha_label}_{member_hash}.pt"
        member_model, member_scalers, member_loss_history, member_cache_reused = train_or_load_mlp(
            splits,
            member_config,
            member_path,
            config_hash=member_hash,
        )
        model_paths.append(member_path)
        models.append(member_model)
        scalers = member_scalers if scalers is None else scalers
        cache_reused = cache_reused and bool(member_cache_reused)
        if member_index == 0:
            loss_history = member_loss_history
        member_summaries.append(
            {
                "ridge_alpha": float(alpha),
                "model_cache": str(member_path),
                "model_config_hash": member_hash,
                "cache_reused": bool(member_cache_reused),
                "training": getattr(member_model, "training_summary", {}),
            }
        )
    model = models[0] if len(models) == 1 else AveragedTransition(models)
    assert scalers is not None
    linear_blend: dict[str, object] = {"enabled": False}
    if int(config.linear_blend_grid_steps) > 1:
        ridge_transition = build_scaled_ridge_transition(
            splits,
            scalers,
            ridge_alpha=float(config.ridge_alpha),
        )
        blend = select_linear_blend_weight(
            model,
            ridge_transition,
            scalers,
            splits,
            names,
            grid_steps=int(config.linear_blend_grid_steps),
        )
        model = WeightedAveragedTransition(
            [model, ridge_transition],
            [float(blend["mlp_weight"]), float(blend["ridge_weight"])],
            training_summary={
                "type": "validation_linear_blend",
                "base_model": getattr(model, "training_summary", {}),
                "ridge_model": getattr(ridge_transition, "training_summary", {}),
                **blend,
            },
        )
        linear_blend = {
            "enabled": True,
            "ridge_alpha": float(config.ridge_alpha),
            "grid_steps": int(config.linear_blend_grid_steps),
            **blend,
        }
    training_summary = getattr(model, "training_summary", {})

    metrics_frames = []
    for split_name, (split_x, split_y) in splits.items():
        pred = predict_mlp(model, scalers, split_x)
        split_metrics = regression_metrics(pred, split_y, names)
        split_metrics.insert(0, "split", split_name)
        metrics_frames.append(split_metrics)
    metrics = pd.concat(metrics_frames, ignore_index=True)
    metrics.to_csv(result_dir / "mlp_metrics.csv", index=False)
    linear_baseline_predictions = predict_ridge_linear_baseline(splits, ridge_alpha=float(config.ridge_alpha))
    reference_linear_predictions = predict_ridge_linear_baseline(splits, ridge_alpha=1.0)
    linear_metric_frames = []
    reference_linear_metric_frames = []
    for split_name, (_, split_y) in splits.items():
        split_metrics = regression_metrics(linear_baseline_predictions[split_name], split_y, names)
        split_metrics.insert(0, "split", split_name)
        linear_metric_frames.append(split_metrics)
        reference_metrics = regression_metrics(reference_linear_predictions[split_name], split_y, names)
        reference_metrics.insert(0, "split", split_name)
        reference_linear_metric_frames.append(reference_metrics)
    linear_metrics = pd.concat(linear_metric_frames, ignore_index=True)
    reference_linear_metrics = pd.concat(reference_linear_metric_frames, ignore_index=True)
    linear_metrics.to_csv(result_dir / "linear_baseline_metrics.csv", index=False)
    reference_linear_metrics.to_csv(result_dir / "reference_linear_alpha1_metrics.csv", index=False)

    if config.ei_estimator == "tm":
        matrix, edges = estimate_pairwise_tm_ei_matrix(
            model,
            scalers,
            splits["train"][0],
            n_components=len(names),
            lag=int(config.lag),
            intervention_samples=int(config.intervention_samples),
            low_q=float(config.quantile_low),
            high_q=float(config.quantile_high),
            seed=int(config.seed),
            source_mode=str(config.source_mode),
        )
    else:
        matrix, edges = estimate_pairwise_ei_matrix(
            model,
            scalers,
            splits["train"][0],
            n_components=len(names),
            lag=int(config.lag),
            intervention_samples=int(config.intervention_samples),
            bins=int(config.bins),
            low_q=float(config.quantile_low),
            high_q=float(config.quantile_high),
            seed=int(config.seed),
            source_mode=str(config.source_mode),
        )
    matrix_frame = pd.DataFrame(matrix, index=names, columns=names)
    matrix_frame.to_csv(result_dir / "pairwise_ei_matrix.csv")
    edges.to_csv(result_dir / "pairwise_ei_edges.csv", index=False)

    title = "Pairwise TM-EI under MLP interventions" if config.ei_estimator == "tm" else "Pairwise EI under MLP interventions"
    save_heatmap(matrix, fig_dir / "pairwise_ei_heatmap.png", title=title)

    manifest = {
        "config": _jsonable_config(config),
        "config_hash": config_hash,
        "model_config_hash": config_hash,
        "component_scores_hash": data_hash,
        "component_scores": str(component_scores_path),
        "n_rows": int(len(frame)),
        "n_components": int(len(names)),
        "n_lagged_samples": int(len(features)),
        "splits": {key: int(len(value[0])) for key, value in splits.items()},
        "model_cache": str(model_paths[0]) if len(model_paths) == 1 else str(model_paths[0]),
        "model_caches": [str(path) for path in model_paths],
        "model_cache_reused": bool(cache_reused),
        "ensemble_members": member_summaries,
        "final_train_loss": float(loss_history[-1]) if loss_history else None,
        "best_epoch": int(training_summary["best_epoch"]) if "best_epoch" in training_summary else None,
        "best_train_loss": _finite_or_none(training_summary.get("best_train_loss")),
        "best_val_loss": _finite_or_none(training_summary.get("best_val_loss")),
        "stopped_early": bool(training_summary.get("stopped_early", False)),
        "model_training": training_summary,
        "linear_blend": linear_blend,
        "linear_baseline": {
            split_name: linear_metrics[
                (linear_metrics["split"] == split_name) & (linear_metrics["component"] == "overall")
            ].iloc[0][["rmse", "mae", "corr"]].to_dict()
            for split_name in splits
        },
        "reference_linear_alpha1": {
            split_name: reference_linear_metrics[
                (reference_linear_metrics["split"] == split_name) & (reference_linear_metrics["component"] == "overall")
            ].iloc[0][["rmse", "mae", "corr"]].to_dict()
            for split_name in splits
        },
        "dependency_versions": dependency_versions(),
    }
    if config.linear_coefficients is not None:
        linear_matrix = load_linear_coefficient_matrix(config.linear_coefficients, names)
        comparison, comparison_summary = compare_ei_to_linear_coefficients(matrix, linear_matrix, names)
        comparison.to_csv(result_dir / "ei_linear_coefficient_comparison.csv", index=False)
        (result_dir / "ei_linear_coefficient_comparison_summary.json").write_text(
            json.dumps(comparison_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        manifest["linear_coefficients"] = str(_resolve_path(config.linear_coefficients))
        manifest["ei_linear_comparison"] = comparison_summary
    test_metrics = metrics[metrics["split"] == "test"]
    if config.gateway_mode == "path_effect":
        sparse = sparsify_ei_graph(
            matrix,
            mode=str(config.graph_sparsify),
            topk=int(config.graph_topk),
            quantile=float(config.graph_quantile),
        )
        path_effects = compute_ei_path_effects(sparse, names, path_alpha=float(config.path_alpha))
        pd.DataFrame(sparse, index=names, columns=names).to_csv(result_dir / "direct_ei_matrix.csv")
        pd.DataFrame(path_effects.scaled_direct_matrix, index=names, columns=names).to_csv(result_dir / "scaled_direct_effect_matrix.csv")
        pd.DataFrame(path_effects.total_matrix, index=names, columns=names).to_csv(result_dir / "total_effect_matrix.csv")
        path_effects.direct_effects.to_csv(result_dir / "direct_effects.csv", index=False)
        path_effects.total_effects.to_csv(result_dir / "total_effects.csv", index=False)
        path_effects.path_effects.to_csv(result_dir / "mediated_path_effects.csv", index=False)
        path_effects.gateway_scores.to_csv(result_dir / "gateway_scores.csv", index=False)
        path_effects.mediator_scores.to_csv(result_dir / "mediator_scores.csv", index=False)
        save_heatmap(sparse, fig_dir / "direct_ei_heatmap.png", title="Sparsified TM-EI causal graph")
        save_heatmap(path_effects.total_matrix, fig_dir / "total_effect_heatmap.png", title="Runge-style total effects on TM-EI graph")
        save_path_gateway_ranking(path_effects.gateway_scores, fig_dir / "gateway_ranking.png", title="MLP-TM-EI path-effect gateway ranking")
        save_mediator_ranking(path_effects.mediator_scores, fig_dir / "mediator_ranking.png", title="MLP-TM-EI mediated-effect ranking")
        manifest.update(
            {
                "path_scale_factor": float(path_effects.scale_factor),
                "n_direct_edges": int(len(path_effects.direct_effects)),
                "n_total_effects": int(len(path_effects.total_effects)),
                "n_mediated_paths": int(len(path_effects.path_effects)),
                "top_gateways": path_effects.gateway_scores.head(10).to_dict("records"),
                "top_mediators": path_effects.mediator_scores.head(10).to_dict("records"),
            }
        )
        (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
        write_path_summary(result_dir / "summary.md", manifest, path_effects.gateway_scores, path_effects.mediator_scores, test_metrics)
    else:
        gateway_scores = gateway_scores_from_matrix(matrix, names)
        gateway_scores.to_csv(result_dir / "gateway_scores.csv", index=False)
        ranking_title = "Pairwise MLP-TM-EI gateway ranking" if config.ei_estimator == "tm" else "Pairwise MLP-EI gateway ranking"
        save_gateway_ranking(gateway_scores, fig_dir / "gateway_ranking.png", title=ranking_title)
        (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
        write_summary(result_dir / "summary.md", manifest, gateway_scores, test_metrics)
    return {
        "result_dir": result_dir,
        "fig_dir": fig_dir,
        "model_path": model_paths[0],
        "manifest": result_dir / "manifest.json",
    }


def parse_args(argv: Sequence[str] | None = None) -> PairwiseMlpEiConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-scores", type=Path, default=DEFAULT_COMPONENT_SCORES)
    parser.add_argument("--linear-coefficients", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ensemble-ridge-alphas", default="")
    parser.add_argument("--linear-blend-grid-steps", type=int, default=0)
    parser.add_argument("--disable-linear-skip", dest="use_linear_skip", action="store_false")
    parser.set_defaults(use_linear_skip=True)
    parser.add_argument("--train-linear-skip", dest="freeze_linear_skip", action="store_false")
    parser.set_defaults(freeze_linear_skip=True)
    parser.add_argument("--disable-residual-shrinkage", dest="residual_shrinkage", action="store_false")
    parser.set_defaults(residual_shrinkage=True)
    parser.add_argument("--residual-gamma-min", type=float, default=-0.5)
    parser.add_argument("--residual-gamma-max", type=float, default=0.5)
    parser.add_argument("--residual-gamma-steps", type=int, default=101)
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=1.0e-5)
    parser.add_argument("--scheduler-patience", type=int, default=12)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--intervention-samples", type=int, default=4096)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--ei-estimator", choices=["discrete", "tm"], default="discrete")
    parser.add_argument("--gateway-mode", choices=["pairwise", "path_effect"], default="pairwise")
    parser.add_argument(
        "--graph-sparsify",
        choices=["none", "source_topk", "target_topk", "bidirectional_topk", "global_quantile"],
        default="source_topk",
    )
    parser.add_argument("--graph-topk", type=int, default=5)
    parser.add_argument("--graph-quantile", type=float, default=0.95)
    parser.add_argument("--path-alpha", type=float, default=0.8)
    parser.add_argument("--quantile-low", type=float, default=0.05)
    parser.add_argument("--quantile-high", type=float, default=0.95)
    parser.add_argument("--source-mode", choices=["latest", "history"], default="latest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args(argv)
    args.ensemble_ridge_alphas = parse_float_tuple(args.ensemble_ridge_alphas)
    return PairwiseMlpEiConfig(**vars(args))


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    artifacts = run(config)
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
