#!/usr/bin/env python3
"""Compare Runge RNN, MLP, and tuned linear multi-step forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import Callable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_runge_pairwise_mlp_ei as pairwise  # noqa: E402


RESULT_SUBDIR = Path("results/runge/rnn_forecast_comparison")
FIG_SUBDIR = Path("fig/runge/rnn_forecast_comparison")
SWEEP_RESULT_SUBDIR = Path("results/runge/rnn_history_sweep")
SWEEP_FIG_SUBDIR = Path("fig/runge/rnn_history_sweep")

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
class RungeRnnForecastConfig:
    component_scores: Path = pairwise.DEFAULT_COMPONENT_SCORES
    output_dir: Path = Path(".")
    result_subdir: Path = RESULT_SUBDIR
    fig_subdir: Path = FIG_SUBDIR
    lag: int = 4
    horizons: tuple[int, ...] = (1, 2, 4, 8)
    rnn_type: str = "gru"
    rnn_objective: str = "direct_multihorizon"
    device: str = "auto"
    hidden_dim: int = 192
    num_layers: int = 1
    dropout: float = 0.0
    transformer_nhead: int = 4
    transformer_dim_feedforward: int = 384
    transformer_pooling: str = "last"
    transformer_positional_encoding: str = "learned"
    epochs: int = 160
    learning_rate: float = 8.0e-4
    batch_size: int = 256
    weight_decay: float = 1.0e-4
    ridge_alphas: tuple[float, ...] = (10.0, 100.0, 1000.0, 3000.0)
    use_linear_skip: bool = True
    freeze_linear_skip: bool = True
    residual_shrinkage: bool = True
    residual_gamma_min: float = -0.5
    residual_gamma_max: float = 0.5
    residual_gamma_steps: int = 101
    rnn_linear_blend_grid_steps: int = 0
    early_stopping_patience: int = 40
    min_delta: float = 1.0e-5
    scheduler_patience: int = 12
    gradient_clip_norm: float = 5.0
    mlp_hidden_dim: int = 128
    mlp_num_layers: int = 1
    mlp_dropout: float = 0.5
    mlp_epochs: int = 120
    bootstrap_reps: int = 1000
    bootstrap_block_size: int = 26
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    seed: int = 42
    force_retrain: bool = False
    history_grid: tuple[int, ...] | None = None
    candidate_output_root: Path = SWEEP_RESULT_SUBDIR
    rank_metric: str = "val_avg_rmse"
    top_k_refine: int = 3
    final_seeds: tuple[int, ...] = (42, 43, 44)
    hidden_dim_grid: tuple[int, ...] = (64, 128, 192, 256)
    dropout_grid: tuple[float, ...] = (0.0, 0.1, 0.2)
    weight_decay_grid: tuple[float, ...] = (1.0e-5, 1.0e-4, 1.0e-3)


def _jsonable_config(config: RungeRnnForecastConfig) -> dict[str, object]:
    data = asdict(config)
    data["component_scores"] = str(config.component_scores)
    data["output_dir"] = str(config.output_dir)
    data["result_subdir"] = str(config.result_subdir)
    data["fig_subdir"] = str(config.fig_subdir)
    data["horizons"] = list(config.horizons)
    data["ridge_alphas"] = list(config.ridge_alphas)
    data["history_grid"] = list(config.history_grid) if config.history_grid is not None else None
    data["candidate_output_root"] = str(config.candidate_output_root)
    data["final_seeds"] = list(config.final_seeds)
    data["hidden_dim_grid"] = list(config.hidden_dim_grid)
    data["dropout_grid"] = list(config.dropout_grid)
    data["weight_decay_grid"] = list(config.weight_decay_grid)
    return data


def primary_model_name(config: RungeRnnForecastConfig | None = None, *, rnn_type: str | None = None) -> str:
    value = str(rnn_type if rnn_type is not None else getattr(config, "rnn_type", "rnn")).lower()
    return "Transformer" if value == "transformer" else "RNN"


def resolve_torch_device(requested: str):
    import torch

    value = str(requested).lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if value == "mps" and not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        return torch.device("cpu")
    if value not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be 'auto', 'cpu', 'mps', or 'cuda'.")
    return torch.device(value)


def _state_dict_cpu(model: object) -> dict[str, object]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _model_device(model: object):
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def parse_int_tuple(text: str | Sequence[int] | None) -> tuple[int, ...]:
    if text is None:
        raise ValueError("horizons must not be empty.")
    if isinstance(text, str):
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if not parts:
            raise ValueError("horizons must not be empty.")
        values = tuple(int(part) for part in parts)
    else:
        values = tuple(int(value) for value in text)
    if not values or any(value < 1 for value in values):
        raise ValueError("horizons must contain positive integers.")
    return tuple(sorted(set(values)))


def parse_history_grid(text: str | Sequence[int] | None) -> tuple[int, ...]:
    try:
        return parse_int_tuple(text)
    except ValueError as exc:
        raise ValueError("history grid must contain positive integers.") from exc


def parse_float_tuple(text: str | Sequence[float] | None) -> tuple[float, ...]:
    if text is None:
        raise ValueError("ridge_alphas must not be empty.")
    if isinstance(text, str):
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if not parts:
            raise ValueError("ridge_alphas must not be empty.")
        values = tuple(float(part) for part in parts)
    else:
        values = tuple(float(value) for value in text)
    if not values or any(value < 0.0 for value in values):
        raise ValueError("ridge_alphas must contain nonnegative values.")
    return values


def _format_float_token(value: float) -> str:
    text = f"{float(value):.0e}" if abs(float(value)) < 1.0e-3 and float(value) != 0.0 else f"{float(value):.4f}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def candidate_run_name(*, history: int, hidden_dim: int, dropout: float, weight_decay: float, seed: int) -> str:
    return (
        f"history_{int(history):02d}_h{int(hidden_dim)}"
        f"_do{_format_float_token(float(dropout))}"
        f"_wd{_format_float_token(float(weight_decay))}"
        f"_seed{int(seed)}"
    )


def _content_hash(frame: pd.DataFrame, config: RungeRnnForecastConfig) -> str:
    digest = hashlib.sha256()
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    digest.update(hashed.tobytes())
    digest.update(json.dumps(_jsonable_config(config), sort_keys=True).encode("utf-8"))
    digest.update(b"rnn_forecast_objectives_v2")
    return digest.hexdigest()[:16]


def build_multistep_lagged_dataset(
    frame: pd.DataFrame,
    *,
    lag: int,
    horizons: Sequence[int],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    horizons = parse_int_tuple(horizons)
    values = frame.to_numpy(dtype=float)
    n_time, n_components = values.shape
    max_horizon = max(horizons)
    n_samples = n_time - int(lag) - int(max_horizon) + 1
    if n_samples <= 0:
        raise ValueError("not enough rows for requested lag and horizons.")
    features = np.empty((n_samples, int(lag) * n_components), dtype=float)
    targets = {horizon: np.empty((n_samples, n_components), dtype=float) for horizon in horizons}
    for sample_idx in range(n_samples):
        features[sample_idx] = values[sample_idx : sample_idx + int(lag)].reshape(-1)
        for horizon in horizons:
            targets[int(horizon)][sample_idx] = values[sample_idx + int(lag) + int(horizon) - 1]
    return features, targets


def split_multistep_arrays(
    features: np.ndarray,
    targets_by_horizon: dict[int, np.ndarray],
    *,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, tuple[np.ndarray, dict[int, np.ndarray]]]:
    first_horizon = min(targets_by_horizon)
    base_splits = pairwise.split_temporal_arrays(
        features,
        targets_by_horizon[first_horizon],
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    splits: dict[str, tuple[np.ndarray, dict[int, np.ndarray]]] = {}
    offset = 0
    for split_name, (split_x, _) in base_splits.items():
        size = len(split_x)
        split_targets = {horizon: values[offset : offset + size] for horizon, values in targets_by_horizon.items()}
        splits[split_name] = (split_x, split_targets)
        offset += size
    return splits


def _one_step_splits(
    splits: dict[str, tuple[np.ndarray, dict[int, np.ndarray]]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {name: (features, targets[min(targets)]) for name, (features, targets) in splits.items()}


def _concat_horizon_targets(targets: dict[int, np.ndarray], horizons: Sequence[int]) -> np.ndarray:
    return np.concatenate([np.asarray(targets[int(horizon)], dtype=float) for horizon in horizons], axis=1)


def _multi_horizon_splits(
    splits: dict[str, tuple[np.ndarray, dict[int, np.ndarray]]],
    horizons: Sequence[int],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {name: (features, _concat_horizon_targets(targets, horizons)) for name, (features, targets) in splits.items()}


def forecast_recursive(
    initial_features: np.ndarray,
    predict_next: Callable[[np.ndarray], np.ndarray],
    *,
    lag: int,
    n_components: int,
    max_horizon: int,
) -> dict[int, np.ndarray]:
    window = np.asarray(initial_features, dtype=float).reshape(len(initial_features), int(lag), int(n_components)).copy()
    forecasts: dict[int, np.ndarray] = {}
    for horizon in range(1, int(max_horizon) + 1):
        next_values = np.asarray(predict_next(window.reshape(len(window), int(lag) * int(n_components))), dtype=float)
        if next_values.shape != (len(window), int(n_components)):
            raise ValueError("predict_next returned an array with the wrong shape.")
        forecasts[horizon] = next_values.copy()
        window = np.concatenate([window[:, 1:, :], next_values[:, None, :]], axis=1)
    return forecasts


class RnnTransition:
    def __new__(
        cls,
        *,
        input_size: int,
        output_dim: int,
        lag: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        rnn_type: str,
        use_linear_skip: bool,
    ):
        import torch

        rnn_type_lower = str(rnn_type).lower()
        if rnn_type_lower not in {"gru", "rnn"}:
            raise ValueError("rnn_type must be 'gru' or 'rnn'.")
        recurrent_cls = torch.nn.GRU if rnn_type_lower == "gru" else torch.nn.RNN

        class _Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lag = int(lag)
                self.input_size = int(input_size)
                self.residual_scale = 1.0
                self.rnn = recurrent_cls(
                    input_size=int(input_size),
                    hidden_size=int(hidden_dim),
                    num_layers=max(1, int(num_layers)),
                    dropout=float(dropout) if int(num_layers) > 1 else 0.0,
                    batch_first=True,
                )
                self.head = torch.nn.Linear(int(hidden_dim), int(output_dim))
                self.skip = torch.nn.Linear(int(lag) * int(input_size), int(output_dim)) if use_linear_skip else None
                if self.skip is not None:
                    torch.nn.init.zeros_(self.head.weight)
                    torch.nn.init.zeros_(self.head.bias)

            def forward(self, x):
                sequence = x.reshape(x.shape[0], self.lag, self.input_size)
                recurrent, _ = self.rnn(sequence)
                output = self.head(recurrent[:, -1, :]) * float(self.residual_scale)
                if self.skip is not None:
                    output = output + self.skip(x)
                return output

        return _Net()


class TransformerTransition:
    def __new__(
        cls,
        *,
        input_size: int,
        output_dim: int,
        lag: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        transformer_nhead: int,
        transformer_dim_feedforward: int,
        transformer_pooling: str,
        transformer_positional_encoding: str,
        use_linear_skip: bool,
    ):
        import math
        import torch

        if int(transformer_nhead) < 1:
            raise ValueError("transformer_nhead must be positive.")
        if int(hidden_dim) % int(transformer_nhead) != 0:
            raise ValueError("hidden_dim must be divisible by transformer_nhead.")
        if str(transformer_pooling) not in {"last", "mean"}:
            raise ValueError("transformer_pooling must be 'last' or 'mean'.")
        if str(transformer_positional_encoding) not in {"learned", "sinusoidal"}:
            raise ValueError("transformer_positional_encoding must be 'learned' or 'sinusoidal'.")
        feedforward_dim = int(transformer_dim_feedforward) if int(transformer_dim_feedforward) > 0 else int(hidden_dim) * 4

        class _Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lag = int(lag)
                self.input_size = int(input_size)
                self.pooling = str(transformer_pooling)
                self.positional_encoding_kind = str(transformer_positional_encoding)
                self.residual_scale = 1.0
                self.input_projection = torch.nn.Linear(int(input_size), int(hidden_dim))
                if self.positional_encoding_kind == "learned":
                    self.position = torch.nn.Parameter(torch.zeros(1, int(lag), int(hidden_dim)))
                    torch.nn.init.normal_(self.position, mean=0.0, std=0.02)
                else:
                    positions = torch.arange(int(lag), dtype=torch.float32).unsqueeze(1)
                    div_term = torch.exp(
                        torch.arange(0, int(hidden_dim), 2, dtype=torch.float32)
                        * (-math.log(10000.0) / max(1, int(hidden_dim)))
                    )
                    encoding = torch.zeros(int(lag), int(hidden_dim), dtype=torch.float32)
                    encoding[:, 0::2] = torch.sin(positions * div_term)
                    if int(hidden_dim) > 1:
                        encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
                    self.register_buffer("position", encoding.unsqueeze(0), persistent=False)
                layer = torch.nn.TransformerEncoderLayer(
                    d_model=int(hidden_dim),
                    nhead=int(transformer_nhead),
                    dim_feedforward=feedforward_dim,
                    dropout=float(dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = torch.nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))
                self.output_norm = torch.nn.LayerNorm(int(hidden_dim))
                self.head = torch.nn.Linear(int(hidden_dim), int(output_dim))
                self.skip = torch.nn.Linear(int(lag) * int(input_size), int(output_dim)) if use_linear_skip else None
                if self.skip is not None:
                    torch.nn.init.zeros_(self.head.weight)
                    torch.nn.init.zeros_(self.head.bias)

            def forward(self, x):
                sequence = x.reshape(x.shape[0], self.lag, self.input_size)
                tokens = self.input_projection(sequence) + self.position[:, : self.lag, :].to(sequence.device)
                encoded = self.encoder(tokens)
                if self.pooling == "mean":
                    pooled = encoded.mean(dim=1)
                else:
                    pooled = encoded[:, -1, :]
                output = self.head(self.output_norm(pooled)) * float(self.residual_scale)
                if self.skip is not None:
                    output = output + self.skip(x)
                return output

        return _Net()


def build_transition_model(
    *,
    config: RungeRnnForecastConfig,
    input_size: int,
    output_dim: int,
) -> object:
    rnn_type_lower = str(config.rnn_type).lower()
    if rnn_type_lower == "transformer":
        return TransformerTransition(
            input_size=int(input_size),
            output_dim=int(output_dim),
            lag=int(config.lag),
            hidden_dim=int(config.hidden_dim),
            num_layers=int(config.num_layers),
            dropout=float(config.dropout),
            transformer_nhead=int(config.transformer_nhead),
            transformer_dim_feedforward=int(config.transformer_dim_feedforward),
            transformer_pooling=str(config.transformer_pooling),
            transformer_positional_encoding=str(config.transformer_positional_encoding),
            use_linear_skip=bool(config.use_linear_skip),
        )
    return RnnTransition(
        input_size=int(input_size),
        output_dim=int(output_dim),
        lag=int(config.lag),
        hidden_dim=int(config.hidden_dim),
        num_layers=int(config.num_layers),
        dropout=float(config.dropout),
        rnn_type=str(config.rnn_type),
        use_linear_skip=bool(config.use_linear_skip),
    )


def initialize_linear_skip(model: object, x_scaled: np.ndarray, y_scaled: np.ndarray, *, alpha: float) -> None:
    import torch

    skip = getattr(model, "skip", None)
    if skip is None:
        return
    weight, bias = pairwise.fit_ridge_linear_map(x_scaled, y_scaled, alpha=float(alpha))
    with torch.no_grad():
        skip.weight.copy_(torch.tensor(weight, dtype=skip.weight.dtype, device=skip.weight.device))
        skip.bias.copy_(torch.tensor(bias, dtype=skip.bias.dtype, device=skip.bias.device))


def _scale_rollout_targets(targets: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, *, n_components: int) -> np.ndarray:
    target_values = np.asarray(targets, dtype=np.float32)
    if target_values.shape[1] % int(n_components) != 0:
        raise ValueError("rollout target width must be a multiple of n_components.")
    pieces = []
    for start in range(0, target_values.shape[1], int(n_components)):
        pieces.append((target_values[:, start : start + int(n_components)] - y_mean) / y_std)
    return np.concatenate(pieces, axis=1).astype(np.float32)


def _rollout_loss(
    model: object,
    x_tensor: object,
    y_tensor: object,
    *,
    horizons: Sequence[int],
    lag: int,
    n_components: int,
    loss_fn: object,
):
    import torch

    window = x_tensor.reshape(x_tensor.shape[0], int(lag), int(n_components))
    losses = []
    max_horizon = max(int(horizon) for horizon in horizons)
    horizon_positions = {int(horizon): idx for idx, horizon in enumerate(horizons)}
    for step in range(1, max_horizon + 1):
        prediction = model(window.reshape(window.shape[0], int(lag) * int(n_components)))
        if step in horizon_positions:
            pos = horizon_positions[step]
            target = y_tensor[:, pos * int(n_components) : (pos + 1) * int(n_components)]
            losses.append(loss_fn(prediction, target))
        window = torch.cat([window[:, 1:, :], prediction.unsqueeze(1)], dim=1)
    if not losses:
        raise ValueError("rollout loss requires at least one horizon.")
    return torch.stack(losses).mean()


def train_or_load_rnn(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    config: RungeRnnForecastConfig,
    model_path: Path,
    *,
    n_components: int,
    horizons: Sequence[int],
    ridge_alpha: float,
    config_hash: str,
) -> tuple[object, dict[str, np.ndarray], list[float], bool]:
    import torch

    device = resolve_torch_device(str(config.device))
    x_train, y_train = splits["train"]
    rollout_objective = str(config.rnn_objective) == "rollout_multistep"
    output_dim = int(n_components) if rollout_objective else y_train.shape[1]
    if model_path.exists() and not config.force_retrain:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        if payload.get("config_hash") == config_hash:
            model_config = replace(
                config,
                hidden_dim=int(payload["hidden_dim"]),
                num_layers=int(payload["num_layers"]),
                dropout=float(payload["dropout"]),
                rnn_type=str(payload["rnn_type"]),
                use_linear_skip=bool(payload["use_linear_skip"]),
                transformer_nhead=int(payload.get("transformer_nhead", config.transformer_nhead)),
                transformer_dim_feedforward=int(
                    payload.get("transformer_dim_feedforward", config.transformer_dim_feedforward)
                ),
                transformer_pooling=str(payload.get("transformer_pooling", config.transformer_pooling)),
                transformer_positional_encoding=str(
                    payload.get("transformer_positional_encoding", config.transformer_positional_encoding)
                ),
            )
            model = build_transition_model(
                config=model_config,
                input_size=int(n_components),
                output_dim=output_dim,
            )
            model.load_state_dict(payload["model_state_dict"])
            model.residual_scale = float(payload.get("residual_scale", 1.0))
            model.training_summary = dict(payload.get("training_summary", {}))
            model.to(device)
            scalers = {
                "x_mean": np.asarray(payload["x_mean"], dtype=np.float32),
                "x_std": np.asarray(payload["x_std"], dtype=np.float32),
                "y_mean": np.asarray(payload["y_mean"], dtype=np.float32),
                "y_std": np.asarray(payload["y_std"], dtype=np.float32),
            }
            return model, scalers, list(payload.get("loss_history", [])), True

    torch.manual_seed(int(config.seed))
    if device.type == "cpu":
        torch.set_num_threads(1)
    x_train_scaled, x_mean, x_std = pairwise._standardize(x_train, x_train)
    if rollout_objective:
        y_train_base = np.asarray(y_train[:, : int(n_components)], dtype=np.float32)
        _, y_mean, y_std = pairwise._standardize(y_train_base, y_train_base)
        y_train_scaled = _scale_rollout_targets(y_train, y_mean, y_std, n_components=int(n_components))
    else:
        y_train_scaled, y_mean, y_std = pairwise._standardize(y_train, y_train)
    x_val, y_val = splits["val"]
    x_val_scaled = (np.asarray(x_val, dtype=np.float32) - x_mean) / x_std
    if rollout_objective:
        y_val_scaled = _scale_rollout_targets(y_val, y_mean, y_std, n_components=int(n_components))
    else:
        y_val_scaled = (np.asarray(y_val, dtype=np.float32) - y_mean) / y_std
    model = build_transition_model(
        config=config,
        input_size=int(n_components),
        output_dim=output_dim,
    )
    model.to(device)
    skip_targets = y_train_scaled[:, : int(n_components)] if rollout_objective else y_train_scaled
    initialize_linear_skip(model, x_train_scaled, skip_targets, alpha=float(ridge_alpha))
    if bool(config.freeze_linear_skip) and getattr(model, "skip", None) is not None:
        for parameter in model.skip.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
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
    x_tensor = torch.tensor(x_train_scaled, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train_scaled, dtype=torch.float32, device=device)
    x_val_tensor = torch.tensor(x_val_scaled, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32, device=device)
    batch_size = max(1, min(int(config.batch_size), len(x_tensor)))
    generator = torch.Generator().manual_seed(int(config.seed))
    best_state = _state_dict_cpu(model)
    best_epoch = 0
    best_train_loss = float("inf")
    model.eval()
    with torch.no_grad():
        if rollout_objective:
            best_val_loss = float(
                _rollout_loss(
                    model,
                    x_val_tensor,
                    y_val_tensor,
                    horizons=horizons,
                    lag=int(config.lag),
                    n_components=int(n_components),
                    loss_fn=loss_fn,
                ).item()
            )
        else:
            best_val_loss = float(loss_fn(model(x_val_tensor), y_val_tensor).item())
    loss_history: list[float] = []
    val_loss_history: list[float] = []
    epochs_without_improvement = 0
    stopped_early = False
    for _ in range(int(config.epochs)):
        order = torch.randperm(len(x_tensor), generator=generator)
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            if rollout_objective:
                loss = _rollout_loss(
                    model,
                    x_tensor[batch],
                    y_tensor[batch],
                    horizons=horizons,
                    lag=int(config.lag),
                    n_components=int(n_components),
                    loss_fn=loss_fn,
                )
            else:
                loss = loss_fn(model(x_tensor[batch]), y_tensor[batch])
            loss.backward()
            if float(config.gradient_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip_norm))
            optimizer.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses))
        loss_history.append(train_loss)
        model.eval()
        with torch.no_grad():
            if rollout_objective:
                val_loss = float(
                    _rollout_loss(
                        model,
                        x_val_tensor,
                        y_val_tensor,
                        horizons=horizons,
                        lag=int(config.lag),
                        n_components=int(n_components),
                        loss_fn=loss_fn,
                    ).item()
                )
            else:
                val_loss = float(loss_fn(model(x_val_tensor), y_val_tensor).item())
        val_loss_history.append(val_loss)
        scheduler.step(val_loss)
        if val_loss < best_val_loss - float(config.min_delta):
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = len(loss_history)
            best_state = _state_dict_cpu(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if int(config.early_stopping_patience) > 0 and epochs_without_improvement >= int(config.early_stopping_patience):
            stopped_early = True
            break

    final_state = _state_dict_cpu(model)
    residual_scale = 1.0
    if bool(config.residual_shrinkage) and getattr(model, "skip", None) is not None:
        gamma_values = np.linspace(
            float(config.residual_gamma_min),
            float(config.residual_gamma_max),
            max(2, int(config.residual_gamma_steps)),
        )
        best_gamma_val = float("inf")
        best_gamma = 0.0
        model.load_state_dict(final_state)
        model.eval()
        with torch.no_grad():
            for gamma in gamma_values:
                model.residual_scale = float(gamma)
                if rollout_objective:
                    gamma_loss = float(
                        _rollout_loss(
                            model,
                            x_val_tensor,
                            y_val_tensor,
                            horizons=horizons,
                            lag=int(config.lag),
                            n_components=int(n_components),
                            loss_fn=loss_fn,
                        ).item()
                    )
                else:
                    gamma_loss = float(loss_fn(model(x_val_tensor), y_val_tensor).item())
                if gamma_loss < best_gamma_val:
                    best_gamma_val = gamma_loss
                    best_gamma = float(gamma)
        if best_gamma_val < best_val_loss:
            best_state = final_state
            best_val_loss = best_gamma_val
            best_epoch = len(loss_history)
            best_train_loss = loss_history[-1] if loss_history else best_train_loss
            residual_scale = best_gamma

    model.load_state_dict(best_state)
    model.residual_scale = float(residual_scale)
    model.training_summary = {
        "type": f"{config.rnn_type}_transition",
        "training_objective": str(config.rnn_objective),
        "device": str(device),
        "horizons": [int(horizon) for horizon in horizons],
        "train_loss_history": loss_history,
        "val_loss_history": val_loss_history,
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "stopped_early": bool(stopped_early),
        "residual_scale": float(residual_scale),
        "ridge_alpha": float(ridge_alpha),
    }
    scalers = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config_hash": config_hash,
            "hidden_dim": int(config.hidden_dim),
            "num_layers": int(config.num_layers),
            "dropout": float(config.dropout),
            "rnn_type": str(config.rnn_type),
            "transformer_nhead": int(config.transformer_nhead),
            "transformer_dim_feedforward": int(config.transformer_dim_feedforward),
            "transformer_pooling": str(config.transformer_pooling),
            "transformer_positional_encoding": str(config.transformer_positional_encoding),
            "horizons": [int(horizon) for horizon in horizons],
            "use_linear_skip": bool(config.use_linear_skip),
            "model_state_dict": model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "loss_history": loss_history,
            "residual_scale": float(residual_scale),
            "training_summary": model.training_summary,
        },
        model_path,
    )
    return model, scalers, loss_history, False


def predict_torch_model(model: object, scalers: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    import torch

    values = np.asarray(features, dtype=np.float32)
    scaled = (values - scalers["x_mean"]) / scalers["x_std"]
    device = _model_device(model)
    model.eval()
    with torch.no_grad():
        pred_tensor = model(torch.tensor(scaled, dtype=torch.float32, device=device)).detach().cpu()
        pred = np.asarray(pred_tensor.tolist(), dtype=np.float32)
    return pred * scalers["y_std"] + scalers["y_mean"]


def predict_direct_horizons(
    model: object,
    scalers: dict[str, np.ndarray],
    features: np.ndarray,
    *,
    horizons: Sequence[int],
    n_components: int,
) -> dict[int, np.ndarray]:
    raw = predict_torch_model(model, scalers, features)
    predictions: dict[int, np.ndarray] = {}
    for idx, horizon in enumerate(horizons):
        start = idx * int(n_components)
        end = start + int(n_components)
        predictions[int(horizon)] = raw[:, start:end]
    return predictions


def select_best_ridge_alpha(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    names: Sequence[str],
    alphas: Sequence[float],
) -> tuple[float, dict[float, dict[str, float]]]:
    summaries: dict[float, dict[str, float]] = {}
    best_alpha: float | None = None
    best_rmse = float("inf")
    for alpha in alphas:
        predictions = pairwise.predict_ridge_linear_baseline(splits, ridge_alpha=float(alpha))
        row = pairwise.regression_metrics(predictions["val"], splits["val"][1], names).query("component == 'overall'").iloc[0]
        summaries[float(alpha)] = {"val_rmse": float(row["rmse"]), "val_mae": float(row["mae"]), "val_corr": float(row["corr"])}
        if float(row["rmse"]) < best_rmse:
            best_rmse = float(row["rmse"])
            best_alpha = float(alpha)
    assert best_alpha is not None
    return best_alpha, summaries


def evaluate_multistep(
    model_name: str,
    forecasts: dict[int, np.ndarray],
    targets_by_horizon: dict[int, np.ndarray],
    names: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for horizon in sorted(targets_by_horizon):
        metrics = pairwise.regression_metrics(forecasts[int(horizon)], targets_by_horizon[int(horizon)], names)
        overall = metrics.query("component == 'overall'").iloc[0]
        rows.append(
            {
                "model": model_name,
                "horizon": int(horizon),
                "component": "overall",
                "rmse": float(overall["rmse"]),
                "mae": float(overall["mae"]),
                "corr": float(overall["corr"]),
            }
        )
        for row in metrics.query("component != 'overall'").to_dict("records"):
            rows.append(
                {
                    "model": model_name,
                    "horizon": int(horizon),
                    "component": row["component"],
                    "rmse": float(row["rmse"]),
                    "mae": float(row["mae"]),
                    "corr": float(row["corr"]),
                }
            )
    return pd.DataFrame(rows)


def select_horizon_blend_weights(
    rnn_forecasts: dict[int, np.ndarray],
    ridge_forecasts: dict[int, np.ndarray],
    targets_by_horizon: dict[int, np.ndarray],
    *,
    grid_steps: int,
) -> dict[int, dict[str, float]]:
    steps = max(2, int(grid_steps))
    selected: dict[int, dict[str, float]] = {}
    for horizon, target in targets_by_horizon.items():
        best: dict[str, float] | None = None
        for weight in np.linspace(0.0, 1.0, steps):
            prediction = float(weight) * rnn_forecasts[int(horizon)] + (1.0 - float(weight)) * ridge_forecasts[int(horizon)]
            rmse = float(np.sqrt(np.mean((np.asarray(target) - prediction) ** 2)))
            mae = float(np.mean(np.abs(np.asarray(target) - prediction)))
            corr = float(np.corrcoef(np.asarray(target).reshape(-1), prediction.reshape(-1))[0, 1])
            candidate = {
                "rnn_weight": float(weight),
                "ridge_weight": float(1.0 - float(weight)),
                "val_rmse": rmse,
                "val_mae": mae,
                "val_corr": corr,
            }
            if best is None or candidate["val_rmse"] < best["val_rmse"]:
                best = candidate
        assert best is not None
        selected[int(horizon)] = best
    return selected


def apply_horizon_blend(
    rnn_forecasts: dict[int, np.ndarray],
    ridge_forecasts: dict[int, np.ndarray],
    weights: dict[int, dict[str, float]],
) -> dict[int, np.ndarray]:
    return {
        int(horizon): float(weights[int(horizon)]["rnn_weight"]) * rnn_forecasts[int(horizon)]
        + float(weights[int(horizon)]["ridge_weight"]) * ridge_forecasts[int(horizon)]
        for horizon in weights
    }


def _rmse_from_rows(prediction: np.ndarray, target: np.ndarray, rows: np.ndarray | None = None) -> float:
    pred = np.asarray(prediction, dtype=float)
    truth = np.asarray(target, dtype=float)
    if rows is not None:
        pred = pred[rows]
        truth = truth[rows]
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def _circular_block_indices(n_rows: int, *, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")
    size = max(1, min(int(block_size), int(n_rows)))
    starts = rng.integers(0, int(n_rows), size=int(np.ceil(n_rows / size)))
    chunks = [(start + np.arange(size)) % int(n_rows) for start in starts]
    return np.concatenate(chunks)[: int(n_rows)].astype(int)


def compute_prediction_significance(
    forecasts: dict[str, dict[int, np.ndarray]],
    targets_by_horizon: dict[int, np.ndarray],
    *,
    reps: int,
    block_size: int,
    seed: int,
    primary_model: str = "RNN",
) -> dict[str, object]:
    rng = np.random.default_rng(int(seed) + 2909)
    horizon_results: dict[str, object] = {}
    comparisons = ["MLP", "TunedRidge", "BestBaseline"]
    for horizon in sorted(targets_by_horizon):
        target = np.asarray(targets_by_horizon[int(horizon)], dtype=float)
        primary_pred = np.asarray(forecasts[str(primary_model)][int(horizon)], dtype=float)
        baseline_preds = {
            "MLP": np.asarray(forecasts["MLP"][int(horizon)], dtype=float),
            "TunedRidge": np.asarray(forecasts["TunedRidge"][int(horizon)], dtype=float),
        }
        baseline_rmse = {name: _rmse_from_rows(pred, target) for name, pred in baseline_preds.items()}
        best_name = min(baseline_rmse, key=baseline_rmse.get)
        baseline_preds["BestBaseline"] = baseline_preds[best_name]
        baseline_rmse["BestBaseline"] = baseline_rmse[best_name]
        observed_primary_rmse = _rmse_from_rows(primary_pred, target)
        rows_by_rep = [
            _circular_block_indices(len(target), block_size=int(block_size), rng=rng)
            for _ in range(max(1, int(reps)))
        ]
        per_comparison: dict[str, object] = {}
        for comparison in comparisons:
            improvements = np.asarray(
                [
                    _rmse_from_rows(baseline_preds[comparison], target, rows)
                    - _rmse_from_rows(primary_pred, target, rows)
                    for rows in rows_by_rep
                ],
                dtype=float,
            )
            per_comparison[comparison] = {
                "baseline_model": best_name if comparison == "BestBaseline" else comparison,
                "primary_model": str(primary_model),
                "primary_rmse": float(observed_primary_rmse),
                "rnn_rmse": float(observed_primary_rmse),
                "baseline_rmse": float(baseline_rmse[comparison]),
                "rmse_improvement": float(baseline_rmse[comparison] - observed_primary_rmse),
                "bootstrap_ci95": [
                    float(np.quantile(improvements, 0.025)),
                    float(np.quantile(improvements, 0.975)),
                ],
                "bootstrap_p_improvement_le_0": float(np.mean(improvements <= 0.0)),
            }
        horizon_results[str(int(horizon))] = per_comparison
    return {
        "method": "paired_circular_block_bootstrap",
        "primary_model": str(primary_model),
        "reps": int(reps),
        "block_size": int(block_size),
        "horizons": horizon_results,
    }


def save_metric_plot(metrics: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overall = metrics[metrics["component"] == "overall"].copy()
    colors = {
        "RNN": "#4c78a8",
        "Transformer": "#b279a2",
        "MLP": "#f58518",
        "TunedRidge": "#54a24b",
        "BestBaseline": "#666666",
    }
    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    preferred_order = [
        "RNN",
        "Transformer",
        "TransformerEnsembleTop2",
        "TransformerEnsembleTop3",
        "TransformerEnsembleTop5",
        "TransformerHorizonSelector",
        "MLP",
        "TunedRidge",
    ]
    model_order = [name for name in preferred_order if name in set(overall["model"])]
    model_order.extend([name for name in overall["model"].dropna().unique() if name not in set(model_order) and name != "BestBaseline"])
    for idx, model_name in enumerate(model_order):
        frame = overall[overall["model"] == model_name].sort_values("horizon")
        if frame.empty:
            continue
        ax.plot(
            frame["horizon"],
            frame["rmse"],
            marker="o",
            linewidth=1.8,
            markersize=4.0,
            label=model_name,
            color=colors.get(model_name, f"C{idx}"),
        )
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("Test RMSE")
    ax.set_xticks(sorted(overall["horizon"].unique()))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def add_best_baseline_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = [metrics]
    overall = metrics[metrics["component"] == "overall"]
    for horizon in sorted(overall["horizon"].unique()):
        horizon_rows = overall[overall["horizon"] == horizon]
        baselines = horizon_rows[horizon_rows["model"].isin(["MLP", "TunedRidge"])]
        if baselines.empty:
            continue
        best_model = str(baselines.sort_values("rmse").iloc[0]["model"])
        best_rows = metrics[(metrics["horizon"] == horizon) & (metrics["model"] == best_model)].copy()
        best_rows["model"] = "BestBaseline"
        rows.append(best_rows)
    return pd.concat(rows, ignore_index=True)


def split_sample_indices(n_samples: int, *, train_fraction: float, val_fraction: float) -> dict[str, np.ndarray]:
    dummy = np.arange(int(n_samples))
    splits = pairwise.split_temporal_arrays(
        dummy[:, None],
        dummy[:, None],
        train_fraction=float(train_fraction),
        val_fraction=float(val_fraction),
    )
    return {name: values[1].reshape(-1).astype(int) for name, values in splits.items()}


def save_forecast_arrays(
    output_path: Path,
    *,
    forecasts_by_split: dict[str, dict[str, dict[int, np.ndarray]]],
    targets_by_split: dict[str, dict[int, np.ndarray]],
    sample_indices_by_split: dict[str, np.ndarray],
    horizons: Sequence[int],
    lag: int,
) -> Path:
    arrays: dict[str, np.ndarray] = {}
    for split_name, targets in targets_by_split.items():
        sample_indices = np.asarray(sample_indices_by_split[split_name], dtype=np.int64)
        for horizon in horizons:
            arrays[f"{split_name}_target_h{int(horizon)}"] = np.asarray(targets[int(horizon)], dtype=np.float32)
            arrays[f"{split_name}_target_index_h{int(horizon)}"] = (
                sample_indices + int(lag) + int(horizon) - 1
            ).astype(np.int64)
        for model_name, model_forecasts in forecasts_by_split[split_name].items():
            safe_name = str(model_name).replace(" ", "")
            for horizon in horizons:
                arrays[f"{split_name}_{safe_name}_h{int(horizon)}"] = np.asarray(
                    model_forecasts[int(horizon)],
                    dtype=np.float32,
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return output_path


def _primary_overall_summary(metrics: pd.DataFrame, *, prefix: str) -> dict[str, float]:
    model_names = [name for name in metrics["model"].dropna().unique() if name not in {"MLP", "TunedRidge", "BestBaseline"}]
    model_name = "RNN" if "RNN" in model_names else str(model_names[0])
    overall = metrics[(metrics["component"] == "overall") & (metrics["model"] == model_name)].copy()
    summary = {
        f"{prefix}_avg_rmse": float(overall["rmse"].mean()),
        f"{prefix}_avg_corr": float(overall["corr"].mean()),
    }
    for horizon in (1, 2, 4, 8):
        rows = overall[overall["horizon"] == horizon]
        summary[f"{prefix}_h{horizon}_rmse"] = float(rows.iloc[0]["rmse"]) if not rows.empty else float("nan")
    return summary


def _rnn_overall_summary(metrics: pd.DataFrame, *, prefix: str) -> dict[str, float]:
    return _primary_overall_summary(metrics, prefix=prefix)


def rank_leaderboard(frame: pd.DataFrame, *, rank_metric: str = "val_avg_rmse") -> pd.DataFrame:
    if rank_metric not in frame.columns:
        raise ValueError(f"rank metric {rank_metric!r} is not present in the leaderboard.")
    ranked = frame.copy()
    sort_columns = [rank_metric]
    ascending = [True]
    for column in ("val_h4_rmse", "val_h8_rmse"):
        if column in ranked.columns and column not in sort_columns:
            sort_columns.append(column)
            ascending.append(True)
    if "val_avg_corr" in ranked.columns:
        sort_columns.append("val_avg_corr")
        ascending.append(False)
    ranked = ranked.sort_values(sort_columns, ascending=ascending, kind="mergesort").reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=int))
    return ranked


def _read_candidate_row(
    *,
    candidate: str,
    stage: str,
    history: int,
    hidden_dim: int,
    dropout: float,
    weight_decay: float,
    seed: int,
    artifacts: dict[str, Path],
) -> dict[str, object]:
    validation_metrics = pd.read_csv(artifacts["validation_metrics"])
    test_metrics = pd.read_csv(artifacts["metrics"])
    manifest = json.loads(Path(artifacts["manifest"]).read_text(encoding="utf-8"))
    row: dict[str, object] = {
        "candidate": candidate,
        "stage": stage,
        "history": int(history),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "result_dir": str(artifacts["result_dir"]),
        "fig_dir": str(artifacts["fig_dir"]),
        "manifest": str(artifacts["manifest"]),
        "best_linear_alpha": float(manifest["best_linear_alpha"]),
    }
    row.update(_rnn_overall_summary(validation_metrics, prefix="val"))
    row.update(_rnn_overall_summary(test_metrics, prefix="test"))
    return row


def append_candidate_log(row: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def save_history_sweep_plot(leaderboard: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    stage_colors = {"history": "#4c78a8", "refine": "#f58518", "final_seed": "#54a24b"}
    for stage, frame in leaderboard.groupby("stage", sort=False):
        ax.scatter(
            frame["history"],
            frame["val_avg_rmse"],
            s=32,
            label=str(stage),
            color=stage_colors.get(str(stage), "#777777"),
            alpha=0.88,
        )
    best = leaderboard.sort_values("rank").head(1)
    if not best.empty:
        ax.scatter(best["history"], best["val_avg_rmse"], s=72, facecolors="none", edgecolors="#222222", linewidths=1.4)
    ax.set_xlabel("History length (weeks)")
    ax.set_ylabel("Validation average RMSE")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def dependency_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "matplotlib", "torch"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _json_sanitize(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def write_summary(path: Path, manifest: dict[str, object], metrics: pd.DataFrame) -> Path:
    overall = metrics[metrics["component"] == "overall"].sort_values(["horizon", "model"])
    lines = [
        "# Runge RNN forecast comparison",
        "",
        "This run compares recursive multi-step forecasts from an RNN-class transition model, the existing MLP-style transition model, and the validation-selected Ridge baseline.",
        "",
        "## Run",
        "",
        f"- Lag: {manifest['config']['lag']}",
        f"- Horizons: {manifest['config']['horizons']}",
        f"- Components: {manifest['n_components']}",
        f"- Best linear alpha: {manifest['best_linear_alpha']}",
        f"- RNN residual scale: {manifest['rnn_training'].get('residual_scale')}",
        "",
        "## Test metrics",
        "",
        "| model | horizon | rmse | mae | corr |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in overall.to_dict("records"):
        lines.append(f"| {row['model']} | {int(row['horizon'])} | {row['rmse']:.6g} | {row['mae']:.6g} | {row['corr']:.6g} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(config: RungeRnnForecastConfig) -> dict[str, Path]:
    if int(config.lag) < 1:
        raise ValueError("lag must be positive.")
    if config.rnn_objective not in {"direct_multihorizon", "one_step_recursive", "rollout_multistep"}:
        raise ValueError("rnn_objective must be 'direct_multihorizon', 'one_step_recursive', or 'rollout_multistep'.")
    if str(config.rnn_type).lower() not in {"gru", "rnn", "transformer"}:
        raise ValueError("rnn_type must be 'gru', 'rnn', or 'transformer'.")
    if str(config.transformer_pooling) not in {"last", "mean"}:
        raise ValueError("transformer_pooling must be 'last' or 'mean'.")
    if str(config.transformer_positional_encoding) not in {"learned", "sinusoidal"}:
        raise ValueError("transformer_positional_encoding must be 'learned' or 'sinusoidal'.")
    if str(config.rnn_type).lower() == "transformer" and int(config.hidden_dim) % int(config.transformer_nhead) != 0:
        raise ValueError("hidden_dim must be divisible by transformer_nhead.")
    horizons = parse_int_tuple(config.horizons)
    model_name = primary_model_name(config)
    component_scores_path = pairwise._resolve_path(config.component_scores)
    frame = pairwise.load_component_scores(component_scores_path)
    names = list(frame.columns)
    features, targets_by_horizon = build_multistep_lagged_dataset(frame, lag=int(config.lag), horizons=horizons)
    multi_splits = split_multistep_arrays(
        features,
        targets_by_horizon,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
    )
    one_step_splits = _one_step_splits(multi_splits)
    rnn_splits = _multi_horizon_splits(multi_splits, horizons)
    if config.rnn_objective == "one_step_recursive":
        rnn_splits = one_step_splits
        rnn_train_horizons = (1,)
    else:
        rnn_train_horizons = horizons

    result_dir = (Path(config.output_dir) / Path(config.result_subdir)).resolve()
    fig_dir = (Path(config.output_dir) / Path(config.fig_subdir)).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _content_hash(frame, config)

    best_linear_alpha, linear_search = select_best_ridge_alpha(one_step_splits, names, config.ridge_alphas)
    x_train, y_train = one_step_splits["train"]
    x_train_scaled, x_mean, x_std = pairwise._standardize(x_train, x_train)
    y_train_scaled, y_mean, y_std = pairwise._standardize(y_train, y_train)
    ridge_model = pairwise.build_scaled_ridge_transition(
        one_step_splits,
        {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std},
        ridge_alpha=best_linear_alpha,
    )
    ridge_scalers = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}

    rnn_path = result_dir / f"rnn_transition_{config_hash}.pt"
    rnn_model, rnn_scalers, _, rnn_cache_reused = train_or_load_rnn(
        rnn_splits,
        config,
        rnn_path,
        n_components=len(names),
        horizons=rnn_train_horizons,
        ridge_alpha=best_linear_alpha,
        config_hash=config_hash,
    )

    mlp_config = pairwise.PairwiseMlpEiConfig(
        component_scores=config.component_scores,
        output_dir=config.output_dir,
        lag=int(config.lag),
        horizon=1,
        hidden_dim=int(config.mlp_hidden_dim),
        num_layers=int(config.mlp_num_layers),
        dropout=float(config.mlp_dropout),
        epochs=int(config.mlp_epochs),
        learning_rate=float(config.learning_rate),
        batch_size=int(config.batch_size),
        weight_decay=float(config.weight_decay),
        ridge_alpha=float(best_linear_alpha),
        early_stopping_patience=int(config.early_stopping_patience),
        min_delta=float(config.min_delta),
        scheduler_patience=int(config.scheduler_patience),
        gradient_clip_norm=float(config.gradient_clip_norm),
        intervention_samples=64,
        seed=int(config.seed),
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        force_retrain=bool(config.force_retrain),
    )
    mlp_hash = f"rnn-comparison-{config_hash}"
    mlp_path = result_dir / f"mlp_transition_{config_hash}.pt"
    mlp_model, mlp_scalers, _, mlp_cache_reused = pairwise.train_or_load_mlp(
        one_step_splits,
        mlp_config,
        mlp_path,
        config_hash=mlp_hash,
    )

    x_val, y_val_by_horizon = multi_splits["val"]
    x_test, y_test_by_horizon = multi_splits["test"]
    max_horizon = max(horizons)
    if config.rnn_objective in {"one_step_recursive", "rollout_multistep"}:
        rnn_val_forecasts = forecast_recursive(
            x_val,
            lambda x: predict_torch_model(rnn_model, rnn_scalers, x),
            lag=int(config.lag),
            n_components=len(names),
            max_horizon=max_horizon,
        )
        rnn_test_forecasts = forecast_recursive(
            x_test,
            lambda x: predict_torch_model(rnn_model, rnn_scalers, x),
            lag=int(config.lag),
            n_components=len(names),
            max_horizon=max_horizon,
        )
    else:
        rnn_val_forecasts = predict_direct_horizons(rnn_model, rnn_scalers, x_val, horizons=horizons, n_components=len(names))
        rnn_test_forecasts = predict_direct_horizons(rnn_model, rnn_scalers, x_test, horizons=horizons, n_components=len(names))
    ridge_val_forecasts = forecast_recursive(
        x_val,
        lambda x: pairwise.predict_mlp(ridge_model, ridge_scalers, x),
        lag=int(config.lag),
        n_components=len(names),
        max_horizon=max_horizon,
    )
    ridge_test_forecasts = forecast_recursive(
        x_test,
        lambda x: pairwise.predict_mlp(ridge_model, ridge_scalers, x),
        lag=int(config.lag),
        n_components=len(names),
        max_horizon=max_horizon,
    )
    rnn_linear_blend: dict[str, object] = {"enabled": False}
    rnn_val_metric_forecasts = {horizon: rnn_val_forecasts[horizon] for horizon in horizons}
    if int(config.rnn_linear_blend_grid_steps) > 1:
        weights = select_horizon_blend_weights(
            {horizon: rnn_val_forecasts[horizon] for horizon in horizons},
            {horizon: ridge_val_forecasts[horizon] for horizon in horizons},
            y_val_by_horizon,
            grid_steps=int(config.rnn_linear_blend_grid_steps),
        )
        rnn_test_forecasts = apply_horizon_blend(
            {horizon: rnn_test_forecasts[horizon] for horizon in horizons},
            {horizon: ridge_test_forecasts[horizon] for horizon in horizons},
            weights,
        )
        rnn_val_metric_forecasts = apply_horizon_blend(
            {horizon: rnn_val_forecasts[horizon] for horizon in horizons},
            {horizon: ridge_val_forecasts[horizon] for horizon in horizons},
            weights,
        )
        rnn_linear_blend = {"enabled": True, "grid_steps": int(config.rnn_linear_blend_grid_steps), "weights": weights}

    mlp_val_forecasts = forecast_recursive(
        x_val,
        lambda x: pairwise.predict_mlp(mlp_model, mlp_scalers, x),
        lag=int(config.lag),
        n_components=len(names),
        max_horizon=max_horizon,
    )
    val_forecasts = {
        model_name: rnn_val_metric_forecasts,
        "MLP": mlp_val_forecasts,
        "TunedRidge": ridge_val_forecasts,
    }
    validation_metric_frames = [
        evaluate_multistep(model_name, {horizon: values[horizon] for horizon in horizons}, y_val_by_horizon, names)
        for model_name, values in val_forecasts.items()
    ]
    validation_metrics = pd.concat(validation_metric_frames, ignore_index=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    validation_metrics.to_csv(result_dir / "validation_metrics.csv", index=False)

    forecasts = {
        model_name: rnn_test_forecasts,
        "MLP": forecast_recursive(
            x_test,
            lambda x: pairwise.predict_mlp(mlp_model, mlp_scalers, x),
            lag=int(config.lag),
            n_components=len(names),
            max_horizon=max_horizon,
        ),
        "TunedRidge": ridge_test_forecasts,
    }
    metric_frames = [
        evaluate_multistep(model_name, {horizon: values[horizon] for horizon in horizons}, y_test_by_horizon, names)
        for model_name, values in forecasts.items()
    ]
    metrics = pd.concat(metric_frames, ignore_index=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(result_dir / "multistep_metrics.csv", index=False)
    save_metric_plot(metrics, fig_dir / "multistep_rmse.png")
    significance = compute_prediction_significance(
        forecasts,
        y_test_by_horizon,
        reps=int(config.bootstrap_reps),
        block_size=int(config.bootstrap_block_size),
        seed=int(config.seed),
        primary_model=model_name,
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "prediction_significance.json").write_text(
        json.dumps(_json_sanitize(significance), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    manifest = {
        "config": _jsonable_config(config),
        "config_hash": config_hash,
        "component_scores": str(component_scores_path),
        "n_rows": int(len(frame)),
        "n_components": int(len(names)),
        "n_lagged_samples": int(len(features)),
        "splits": {key: int(len(value[0])) for key, value in multi_splits.items()},
        "best_linear_alpha": float(best_linear_alpha),
        "linear_alpha_search": {str(alpha): values for alpha, values in linear_search.items()},
        "rnn_cache": str(rnn_path),
        "rnn_cache_reused": bool(rnn_cache_reused),
        "primary_model": model_name,
        "rnn_training": getattr(rnn_model, "training_summary", {}),
        "rnn_linear_blend": rnn_linear_blend,
        "prediction_significance": significance,
        "mlp_cache": str(mlp_path),
        "mlp_cache_reused": bool(mlp_cache_reused),
        "mlp_training": getattr(mlp_model, "training_summary", {}),
        "dependency_versions": dependency_versions(),
    }
    manifest = _json_sanitize(manifest)
    result_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = split_sample_indices(
        len(features),
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
    )
    forecast_arrays_path = save_forecast_arrays(
        result_dir / "forecast_arrays.npz",
        forecasts_by_split={
            "val": val_forecasts,
            "test": forecasts,
        },
        targets_by_split={
            "val": y_val_by_horizon,
            "test": y_test_by_horizon,
        },
        sample_indices_by_split={
            "val": sample_indices["val"],
            "test": sample_indices["test"],
        },
        horizons=horizons,
        lag=int(config.lag),
    )
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    write_summary(result_dir / "summary.md", manifest, metrics)
    return {
        "result_dir": result_dir,
        "fig_dir": fig_dir,
        "manifest": result_dir / "manifest.json",
        "metrics": result_dir / "multistep_metrics.csv",
        "validation_metrics": result_dir / "validation_metrics.csv",
        "forecast_arrays": forecast_arrays_path,
    }


def run_history_sweep(config: RungeRnnForecastConfig) -> dict[str, Path]:
    if config.history_grid is None:
        raise ValueError("history_grid is required for the history sweep.")
    histories = parse_history_grid(config.history_grid)
    final_seeds = parse_history_grid(config.final_seeds)
    hidden_grid = parse_history_grid(config.hidden_dim_grid)
    dropout_grid = parse_float_tuple(config.dropout_grid)
    weight_decay_grid = parse_float_tuple(config.weight_decay_grid)
    sweep_result_dir = (Path(config.output_dir) / Path(config.candidate_output_root)).resolve()
    sweep_fig_dir = (Path(config.output_dir) / SWEEP_FIG_SUBDIR).resolve()
    sweep_result_dir.mkdir(parents=True, exist_ok=True)
    sweep_fig_dir.mkdir(parents=True, exist_ok=True)
    candidate_log_path = (Path(config.output_dir) / "docs" / "log" / "runge_rnn_history_sweep_candidates.csv").resolve()

    rows: list[dict[str, object]] = []
    seen: dict[str, dict[str, object]] = {}

    def run_candidate(*, stage: str, history: int, hidden_dim: int, dropout: float, weight_decay: float, seed: int) -> dict[str, object]:
        candidate = candidate_run_name(
            history=history,
            hidden_dim=hidden_dim,
            dropout=dropout,
            weight_decay=weight_decay,
            seed=seed,
        )
        if candidate in seen:
            return seen[candidate]
        candidate_config = replace(
            config,
            lag=int(history),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            weight_decay=float(weight_decay),
            seed=int(seed),
            rnn_type=str(config.rnn_type),
            rnn_objective="rollout_multistep",
            rnn_linear_blend_grid_steps=int(config.rnn_linear_blend_grid_steps)
            if int(config.rnn_linear_blend_grid_steps) > 0
            else 101,
            result_subdir=Path(config.candidate_output_root) / candidate,
            fig_subdir=SWEEP_FIG_SUBDIR / candidate,
            history_grid=None,
        )
        artifacts = run(candidate_config)
        row = _read_candidate_row(
            candidate=candidate,
            stage=stage,
            history=history,
            hidden_dim=hidden_dim,
            dropout=dropout,
            weight_decay=weight_decay,
            seed=seed,
            artifacts=artifacts,
        )
        seen[candidate] = row
        rows.append(row)
        append_candidate_log(row, candidate_log_path)
        return row

    for history in histories:
        run_candidate(
            stage="history",
            history=int(history),
            hidden_dim=int(config.hidden_dim),
            dropout=float(config.dropout),
            weight_decay=float(config.weight_decay),
            seed=int(config.seed),
        )

    initial = rank_leaderboard(pd.DataFrame(rows), rank_metric=str(config.rank_metric))
    refine_count = max(0, int(config.top_k_refine))
    if refine_count > 0:
        for history in initial.head(refine_count)["history"].astype(int).tolist():
            for hidden_dim in hidden_grid:
                for dropout in dropout_grid:
                    for weight_decay in weight_decay_grid:
                        run_candidate(
                            stage="refine",
                            history=int(history),
                            hidden_dim=int(hidden_dim),
                            dropout=float(dropout),
                            weight_decay=float(weight_decay),
                            seed=int(config.seed),
                        )

    refined = rank_leaderboard(pd.DataFrame(rows), rank_metric=str(config.rank_metric))
    final_specs = refined.head(2).to_dict("records")
    for spec in final_specs:
        for seed in final_seeds:
            run_candidate(
                stage="final_seed",
                history=int(spec["history"]),
                hidden_dim=int(spec["hidden_dim"]),
                dropout=float(spec["dropout"]),
                weight_decay=float(spec["weight_decay"]),
                seed=int(seed),
            )

    leaderboard = rank_leaderboard(pd.DataFrame(rows), rank_metric=str(config.rank_metric))
    leaderboard_path = sweep_result_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    save_history_sweep_plot(leaderboard, sweep_fig_dir / "history_sweep_rmse.png")

    best = leaderboard.iloc[0]
    best_result_dir = Path(str(best["result_dir"]))
    final_metrics = add_best_baseline_rows(pd.read_csv(best_result_dir / "multistep_metrics.csv"))
    final_metrics_path = sweep_result_dir / "final_test_metrics.csv"
    final_metrics.to_csv(final_metrics_path, index=False)
    final_significance = json.loads((best_result_dir / "prediction_significance.json").read_text(encoding="utf-8"))
    final_significance_path = sweep_result_dir / "final_prediction_significance.json"
    final_significance_path.write_text(
        json.dumps(_json_sanitize(final_significance), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    selected_manifest = json.loads((best_result_dir / "manifest.json").read_text(encoding="utf-8"))
    sweep_manifest = {
        "selection_rule": {
            "rank_metric": str(config.rank_metric),
            "selected_candidate": str(best["candidate"]),
            "selected_rank": int(best["rank"]),
            "uses_test_for_selection": False,
        },
        "history_grid": [int(value) for value in histories],
        "top_k_refine": int(config.top_k_refine),
        "final_seeds": [int(value) for value in final_seeds],
        "selected_candidate_manifest": selected_manifest,
        "dependency_versions": dependency_versions(),
    }
    manifest_path = sweep_result_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_json_sanitize(sweep_manifest), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    save_metric_plot(final_metrics, sweep_fig_dir / "final_multistep_rmse.png")
    return {
        "result_dir": sweep_result_dir,
        "fig_dir": sweep_fig_dir,
        "manifest": manifest_path,
        "leaderboard": leaderboard_path,
        "final_test_metrics": final_metrics_path,
        "final_prediction_significance": final_significance_path,
        "history_sweep_plot": sweep_fig_dir / "history_sweep_rmse.png",
        "final_metric_plot": sweep_fig_dir / "final_multistep_rmse.png",
    }


def parse_args(argv: Sequence[str] | None = None) -> RungeRnnForecastConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-scores", type=Path, default=pairwise.DEFAULT_COMPONENT_SCORES)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--history-grid", default=None)
    parser.add_argument("--candidate-output-root", type=Path, default=SWEEP_RESULT_SUBDIR)
    parser.add_argument("--rank-metric", default="val_avg_rmse")
    parser.add_argument("--top-k-refine", type=int, default=3)
    parser.add_argument("--final-seeds", default="42,43,44")
    parser.add_argument("--hidden-dim-grid", default="64,128,192,256")
    parser.add_argument("--dropout-grid", default="0,0.1,0.2")
    parser.add_argument("--weight-decay-grid", default="0.00001,0.0001,0.001")
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--horizons", default="1,2,4,8")
    parser.add_argument("--rnn-type", choices=["gru", "rnn", "transformer"], default="gru")
    parser.add_argument(
        "--rnn-objective",
        choices=["direct_multihorizon", "one_step_recursive", "rollout_multistep"],
        default="direct_multihorizon",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--transformer-nhead", type=int, default=4)
    parser.add_argument("--transformer-dim-feedforward", type=int, default=384)
    parser.add_argument("--transformer-pooling", choices=["last", "mean"], default="last")
    parser.add_argument("--transformer-positional-encoding", choices=["learned", "sinusoidal"], default="learned")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--ridge-alphas", default="10,100,1000,3000")
    parser.add_argument("--disable-linear-skip", dest="use_linear_skip", action="store_false")
    parser.set_defaults(use_linear_skip=True)
    parser.add_argument("--train-linear-skip", dest="freeze_linear_skip", action="store_false")
    parser.set_defaults(freeze_linear_skip=True)
    parser.add_argument("--disable-residual-shrinkage", dest="residual_shrinkage", action="store_false")
    parser.set_defaults(residual_shrinkage=True)
    parser.add_argument("--residual-gamma-min", type=float, default=-0.5)
    parser.add_argument("--residual-gamma-max", type=float, default=0.5)
    parser.add_argument("--residual-gamma-steps", type=int, default=101)
    parser.add_argument("--rnn-linear-blend-grid-steps", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=1.0e-5)
    parser.add_argument("--scheduler-patience", type=int, default=12)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-num-layers", type=int, default=1)
    parser.add_argument("--mlp-dropout", type=float, default=0.5)
    parser.add_argument("--mlp-epochs", type=int, default=120)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--bootstrap-block-size", type=int, default=26)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args(argv)
    args.history_grid = parse_history_grid(args.history_grid) if args.history_grid is not None else None
    args.horizons = parse_int_tuple(args.horizons)
    args.ridge_alphas = parse_float_tuple(args.ridge_alphas)
    args.final_seeds = parse_history_grid(args.final_seeds)
    args.hidden_dim_grid = parse_history_grid(args.hidden_dim_grid)
    args.dropout_grid = parse_float_tuple(args.dropout_grid)
    args.weight_decay_grid = parse_float_tuple(args.weight_decay_grid)
    return RungeRnnForecastConfig(**vars(args))


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    artifacts = run_history_sweep(config) if config.history_grid is not None else run(config)
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
