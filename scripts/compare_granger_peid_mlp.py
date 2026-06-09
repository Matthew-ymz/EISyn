#!/usr/bin/env python3
"""Compare lag-ablation Granger graphs with PEID graphs learned by an MLP."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_RESULT_DIR = ROOT / "results" / "granger_peid_mlp_comparison"
DEFAULT_FIGURE_DIR = ROOT / "fig" / "granger_peid_mlp_comparison"
DEFAULT_REPORT_PATH = ROOT / "docs" / "reports" / "granger_peid_mlp_comparison.md"
VARIABLE_NAMES = ("x", "y", "z", "w")


@dataclass(frozen=True)
class SimConfig:
    mechanism: str = "xor_synergy"
    n_samples: int = 1500
    noise: float = 0.05
    seed: int = 0
    lag: int = 1
    synergy_strength: float = 1.0
    hidden_dim: int = 32
    mlp_epochs: int = 100
    learning_rate: float = 0.01
    batch_size: int = 256
    weight_decay: float = 1e-4
    intervention_samples: int = 1024
    bins: int = 4
    common_driver_strength: float = 1.0
    quantile_low: float = 0.05
    quantile_high: float = 0.95
    variable_names: tuple[str, ...] = VARIABLE_NAMES

    def __post_init__(self) -> None:
        if self.mechanism not in {
            "linear_additive",
            "xor_synergy",
            "multiplicative_gate",
            "product_memory_synergy",
            "common_driver_sine_synergy",
            "redundant_common_driver",
        }:
            raise ValueError(f"Unknown mechanism {self.mechanism!r}.")
        if self.n_samples <= self.lag + 10:
            raise ValueError("n_samples must be larger than lag + 10.")
        if self.lag < 1:
            raise ValueError("lag must be positive.")
        if self.noise < 0.0:
            raise ValueError("noise must be nonnegative.")
        if self.intervention_samples < 16:
            raise ValueError("intervention_samples must be at least 16.")
        if self.bins < 2:
            raise ValueError("bins must be at least 2.")
        if not 0.0 <= self.common_driver_strength <= 1.0:
            raise ValueError("common_driver_strength must be between 0 and 1.")


@dataclass
class TrainedMLPTransition:
    net: object
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    variable_names: tuple[str, ...]
    lag: int
    loss_history: list[float]

    def predict(self, features: np.ndarray) -> np.ndarray:
        import torch

        values = np.asarray(features, dtype=np.float32)
        scaled = (values - self.x_mean) / self.x_std
        self.net.eval()
        with torch.no_grad():
            pred_tensor = self.net(torch.tensor(scaled.tolist(), dtype=torch.float32)).cpu()
            pred = np.asarray(pred_tensor.tolist(), dtype=np.float32)
        return pred * self.y_std + self.y_mean


@dataclass(frozen=True)
class PeidGraph:
    pairwise_edges: pd.DataFrame
    synergy_edges: pd.DataFrame
    intervention_states: pd.DataFrame


@dataclass(frozen=True)
class ShapReadout:
    feature_attributions: pd.DataFrame
    shap_interaction_terms: pd.DataFrame
    interaction_terms: pd.DataFrame


@dataclass(frozen=True)
class ConditionalShapReadout:
    feature_attributions: pd.DataFrame


def simulate_system(config: SimConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    """Generate a controlled four-variable time series and its causal ground truth."""

    rng = np.random.default_rng(config.seed)
    n = int(config.n_samples)
    names = tuple(config.variable_names)
    if names != VARIABLE_NAMES:
        raise ValueError("This experiment expects variables ('x', 'y', 'z', 'w').")

    data = np.zeros((n, len(names)), dtype=float)
    truth_pairwise: set[tuple[str, str]] = set()
    truth_hyperedges: set[tuple[str, str, str]] = set()

    if config.mechanism == "xor_synergy":
        data[:, 0] = rng.integers(0, 2, size=n)
        data[:, 1] = rng.integers(0, 2, size=n)
        data[:, 3] = rng.integers(0, 2, size=n)
        xor = np.logical_xor(data[:-1, 0] > 0.5, data[:-1, 1] > 0.5).astype(float)
        if config.noise > 0.0:
            flip = rng.random(size=n - 1) < config.noise
            xor = np.where(flip, 1.0 - xor, xor)
        data[1:, 2] = xor
        truth_hyperedges.add(("x", "y", "z"))

    elif config.mechanism == "multiplicative_gate":
        data[:, 0] = rng.uniform(-1.0, 1.0, size=n)
        data[:, 1] = rng.uniform(-1.0, 1.0, size=n)
        data[:, 3] = rng.normal(0.0, 0.5, size=n)
        signal = np.tanh(config.synergy_strength * data[:-1, 0] * data[:-1, 1])
        data[1:, 2] = signal + rng.normal(0.0, config.noise, size=n - 1)
        truth_hyperedges.add(("x", "y", "z"))

    elif config.mechanism == "product_memory_synergy":
        data[0, 0] = rng.normal(0.0, 0.8)
        data[0, 1] = rng.normal(0.0, 0.8)
        data[0, 2] = rng.normal(0.0, 0.2)
        data[0, 3] = rng.normal(0.0, 0.5)
        for t in range(n - 1):
            data[t + 1, 0] = 0.72 * data[t, 0] + rng.normal(0.0, 0.55)
            data[t + 1, 1] = 0.68 * data[t, 1] + rng.normal(0.0, 0.55)
            data[t + 1, 3] = 0.55 * data[t, 3] + rng.normal(0.0, 0.45)
            product_signal = np.tanh(config.synergy_strength * data[t, 0] * data[t, 1])
            data[t + 1, 2] = (
                0.25 * data[t, 2]
                + product_signal
                + rng.normal(0.0, config.noise)
            )
        truth_hyperedges.add(("x", "y", "z"))

    elif config.mechanism == "common_driver_sine_synergy":
        beta = float(config.common_driver_strength)
        private_scale = float(np.sqrt(max(0.0, 1.0 - beta**2)))
        data[0, 0] = rng.normal(0.0, 0.4)
        data[0, 1] = rng.normal(0.0, 0.4)
        data[0, 2] = rng.normal(0.0, 0.2)
        data[0, 3] = rng.normal(0.0, 0.8)
        for t in range(n - 1):
            data[t + 1, 3] = 0.78 * data[t, 3] + rng.normal(0.0, 0.45)
            data[t + 1, 0] = (
                0.42 * data[t, 0]
                + 0.82 * (beta * data[t, 3] + private_scale * rng.normal(0.0, 0.55))
                + rng.normal(0.0, 0.25)
            )
            data[t + 1, 1] = (
                0.38 * data[t, 1]
                + 0.76 * (beta * data[t, 3] + private_scale * rng.normal(0.0, 0.55))
                + rng.normal(0.0, 0.25)
            )
            sine_signal = config.synergy_strength * np.sin(data[t, 0] * data[t, 1])
            data[t + 1, 2] = (
                0.22 * data[t, 2]
                + sine_signal
                + rng.normal(0.0, config.noise)
            )
        truth_pairwise.update({("w", "x"), ("w", "y")})
        truth_hyperedges.add(("x", "y", "z"))

    elif config.mechanism == "linear_additive":
        data[:, 0] = rng.normal(0.0, 1.0, size=n)
        data[:, 1] = rng.normal(0.0, 1.0, size=n)
        data[:, 3] = rng.normal(0.0, 1.0, size=n)
        data[1:, 2] = (
            0.85 * data[:-1, 0]
            - 0.65 * data[:-1, 1]
            + rng.normal(0.0, config.noise, size=n - 1)
        )
        truth_pairwise.update({("x", "z"), ("y", "z")})

    elif config.mechanism == "redundant_common_driver":
        driver = rng.normal(0.0, 1.0, size=n)
        data[:, 3] = driver
        data[1:, 0] = 0.9 * driver[:-1] + rng.normal(0.0, config.noise, size=n - 1)
        data[1:, 1] = -0.8 * driver[:-1] + rng.normal(0.0, config.noise, size=n - 1)
        data[1:, 2] = 0.7 * driver[:-1] + rng.normal(0.0, config.noise, size=n - 1)
        truth_pairwise.update({("w", "x"), ("w", "y"), ("w", "z")})

    frame = pd.DataFrame(data, columns=names)
    truth = {
        "pairwise_edges": sorted(truth_pairwise),
        "hyperedges": sorted(truth_hyperedges),
        "variables": names,
    }
    return frame, truth


def make_lagged_dataset(series: pd.DataFrame, *, lag: int = 1) -> tuple[np.ndarray, np.ndarray]:
    values = series.to_numpy(dtype=float)
    if lag < 1:
        raise ValueError("lag must be positive.")
    if len(values) <= lag:
        raise ValueError("series is too short for lag.")
    features = np.asarray([values[idx : idx + lag].reshape(-1) for idx in range(len(values) - lag)])
    targets = values[lag:]
    return features.astype(float), targets.astype(float)


def train_mlp_transition_model(
    features: np.ndarray,
    targets: np.ndarray,
    config: SimConfig,
) -> TrainedMLPTransition:
    """Fit a compact Torch MLP one-step transition surrogate."""

    import torch

    torch.manual_seed(int(config.seed))
    torch.set_num_threads(1)

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std = np.where(x_std > 1e-8, x_std, 1.0)
    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True)
    y_std = np.where(y_std > 1e-8, y_std, 1.0)
    x_scaled = (x - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std

    net = torch.nn.Sequential(
        torch.nn.Linear(x_scaled.shape[1], config.hidden_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(config.hidden_dim, config.hidden_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(config.hidden_dim, y_scaled.shape[1]),
    )
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    loss_fn = torch.nn.MSELoss()
    x_tensor = torch.tensor(x_scaled.tolist(), dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled.tolist(), dtype=torch.float32)
    batch_size = max(1, min(int(config.batch_size), len(x_tensor)))

    generator = torch.Generator().manual_seed(int(config.seed))
    loss_history: list[float] = []
    for _ in range(int(config.mlp_epochs)):
        order = torch.randperm(len(x_tensor), generator=generator)
        epoch_losses: list[float] = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net(x_tensor[batch]), y_tensor[batch])
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        loss_history.append(float(np.mean(epoch_losses)))

    return TrainedMLPTransition(
        net=net,
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
        variable_names=tuple(config.variable_names),
        lag=int(config.lag),
        loss_history=loss_history,
    )


def estimate_granger_graph(
    model: TrainedMLPTransition,
    features: np.ndarray,
    targets: np.ndarray,
    config: SimConfig,
) -> pd.DataFrame:
    """Estimate pairwise directed scores by source-lag ablation in the learned MLP."""

    names = tuple(config.variable_names)
    n_vars = len(names)
    base_pred = model.predict(features)
    rows: list[dict[str, object]] = []
    for source_idx, source in enumerate(names):
        ablated = np.asarray(features, dtype=float).copy()
        for lag_idx in range(config.lag):
            col = lag_idx * n_vars + source_idx
            ablated[:, col] = float(np.mean(ablated[:, col]))
        ablated_pred = model.predict(ablated)
        for target_idx, target in enumerate(names):
            base_mse = float(np.mean((targets[:, target_idx] - base_pred[:, target_idx]) ** 2))
            ablated_mse = float(np.mean((targets[:, target_idx] - ablated_pred[:, target_idx]) ** 2))
            score = max(0.0, ablated_mse - base_mse)
            relative = score / (base_mse + 1e-12)
            rows.append(
                {
                    "method": "granger_ablation",
                    "source": source,
                    "target": target,
                    "score": score,
                    "relative_score": float(relative),
                    "base_mse": base_mse,
                    "ablated_mse": ablated_mse,
                }
            )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def _entropy_bits(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


def _state_codes(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(states, dtype=int)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    unique_rows, inverse = np.unique(values, axis=0, return_inverse=True)
    return unique_rows, inverse


def _effective_information_from_states(source_states: np.ndarray, target_states: np.ndarray) -> float:
    _, source_inverse = _state_codes(source_states)
    target = np.asarray(target_states, dtype=int).reshape(-1)
    n_source = int(source_inverse.max()) + 1
    n_target = int(target.max()) + 1
    counts = np.zeros((n_source, n_target), dtype=float)
    for source_idx, target_idx in zip(source_inverse, target):
        counts[int(source_idx), int(target_idx)] += 1.0
    row_totals = counts.sum(axis=1)
    observed = row_totals > 0.0
    if int(observed.sum()) < 2:
        return 0.0
    probs = counts[observed] / row_totals[observed, None]
    target_probs = probs.mean(axis=0)
    return float(_entropy_bits(target_probs) - np.mean([_entropy_bits(row) for row in probs]))


def _mutual_information_from_states(source_states: np.ndarray, target_states: np.ndarray) -> float:
    """Compute empirical mutual information in bits from discrete states."""

    _, source_codes = _state_codes(source_states)
    _, target_codes = _state_codes(target_states)
    _, joint_codes = _state_codes(np.column_stack([source_codes, target_codes]))

    def state_entropy(codes: np.ndarray) -> float:
        counts = np.bincount(np.asarray(codes, dtype=int))
        return _entropy_bits(counts / max(float(counts.sum()), 1.0))

    return float(state_entropy(source_codes) + state_entropy(target_codes) - state_entropy(joint_codes))


def _discretize_vector(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    unique = np.unique(np.round(values, decimals=10))
    if 1 < len(unique) <= bins:
        mapping = {value: idx for idx, value in enumerate(sorted(unique))}
        rounded = np.round(values, decimals=10)
        return np.asarray([mapping[value] for value in rounded], dtype=int)
    if len(unique) <= 1:
        return np.zeros(len(values), dtype=int)
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(quantiles)
    if len(edges) <= 2:
        ranks = pd.Series(values).rank(method="first").to_numpy()
        edges = np.quantile(ranks, np.linspace(0.0, 1.0, bins + 1))
        return np.clip(np.digitize(ranks, edges[1:-1], right=False), 0, bins - 1).astype(int)
    return np.clip(np.digitize(values, edges[1:-1], right=False), 0, len(edges) - 2).astype(int)


def _sample_intervention_sources(series: pd.DataFrame, config: SimConfig) -> pd.DataFrame:
    rng = np.random.default_rng(int(config.seed) + 1009)
    rows: dict[str, np.ndarray] = {}
    for name in config.variable_names:
        values = series[name].to_numpy(dtype=float)
        unique = np.unique(np.round(values, decimals=10))
        if 1 < len(unique) <= max(8, config.bins):
            rows[name] = rng.choice(unique.astype(float), size=config.intervention_samples, replace=True)
            continue
        low = float(np.quantile(values, config.quantile_low))
        high = float(np.quantile(values, config.quantile_high))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.min(values)), float(np.max(values))
        if high <= low:
            rows[name] = np.full(config.intervention_samples, low, dtype=float)
        else:
            rows[name] = rng.uniform(low, high, size=config.intervention_samples)
    return pd.DataFrame(rows)


def _intervention_features(samples: pd.DataFrame, config: SimConfig) -> np.ndarray:
    current = samples[list(config.variable_names)].to_numpy(dtype=float)
    return np.tile(current, (1, int(config.lag)))


def estimate_peid_graph(
    model: TrainedMLPTransition,
    series: pd.DataFrame,
    config: SimConfig,
) -> PeidGraph:
    """Estimate PEID pairwise edges and second-order synergy hyperedges from MLP interventions."""

    samples = _sample_intervention_sources(series, config)
    predictions = model.predict(_intervention_features(samples, config))
    names = tuple(config.variable_names)

    source_states = {
        name: _discretize_vector(samples[name].to_numpy(dtype=float), config.bins)
        for name in names
    }
    target_states = {
        name: _discretize_vector(predictions[:, idx], config.bins)
        for idx, name in enumerate(names)
    }

    pair_rows: list[dict[str, object]] = []
    single_lookup: dict[tuple[str, str], float] = {}
    for source, target in product(names, names):
        ei = _effective_information_from_states(source_states[source], target_states[target])
        single_lookup[(source, target)] = ei
        pair_rows.append({"method": "peid_pairwise", "source": source, "target": target, "ei": float(ei)})

    syn_rows: list[dict[str, object]] = []
    for source_a, source_b in combinations(names, 2):
        joint_sources = np.column_stack([source_states[source_a], source_states[source_b]])
        for target in names:
            joint_ei = _effective_information_from_states(joint_sources, target_states[target])
            single_a = single_lookup[(source_a, target)]
            single_b = single_lookup[(source_b, target)]
            synergy = float(joint_ei - single_a - single_b)
            syn_rows.append(
                {
                    "method": "peid_synergy",
                    "sources": f"{source_a}+{source_b}",
                    "target": target,
                    "source_order": 2,
                    "joint_ei": float(joint_ei),
                    "single_ei_sum": float(single_a + single_b),
                    "best_single_ei": float(max(single_a, single_b)),
                    "synergy": synergy,
                }
            )

    states = samples.copy()
    for idx, name in enumerate(names):
        states[f"{name}_pred"] = predictions[:, idx]
        states[f"{name}_src_state"] = source_states[name]
        states[f"{name}_tgt_state"] = target_states[name]

    return PeidGraph(
        pairwise_edges=pd.DataFrame(pair_rows).sort_values("ei", ascending=False).reset_index(drop=True),
        synergy_edges=pd.DataFrame(syn_rows).sort_values("synergy", ascending=False).reset_index(drop=True),
        intervention_states=states,
    )


def _shapley_kernel_weight(subset_size: int, n_features: int) -> float:
    from math import factorial

    return float(
        factorial(subset_size)
        * factorial(n_features - subset_size - 1)
        / factorial(n_features)
    )


def _subset_prediction_mean(
    model: TrainedMLPTransition,
    foreground: np.ndarray,
    background: np.ndarray,
    *,
    subset: tuple[int, ...],
    target_idx: int,
) -> np.ndarray:
    """Interventional value function v(S) using empirical background replacement."""

    subset_set = set(subset)
    values: list[float] = []
    for row in foreground:
        tiled = np.repeat(background.copy(), repeats=1, axis=0)
        for feature_idx in subset_set:
            tiled[:, feature_idx] = row[feature_idx]
        values.append(float(np.mean(model.predict(tiled)[:, target_idx])))
    return np.asarray(values, dtype=float)


def estimate_shap_readout(
    model: TrainedMLPTransition,
    features: np.ndarray,
    series: pd.DataFrame,
    config: SimConfig,
    *,
    foreground_samples: int = 96,
    background_samples: int = 96,
) -> ShapReadout:
    """Read out SHAP-style single features and OLS product interactions from the same MLP."""

    rng = np.random.default_rng(int(config.seed) + 4049)
    feature_values = np.asarray(features, dtype=float)
    if config.lag != 1:
        raise ValueError("SHAP readout is currently defined for lag=1 examples.")
    if len(feature_values) == 0:
        raise ValueError("features must not be empty.")
    foreground_idx = rng.choice(
        len(feature_values),
        size=min(int(foreground_samples), len(feature_values)),
        replace=False,
    )
    background_idx = rng.choice(
        len(feature_values),
        size=min(int(background_samples), len(feature_values)),
        replace=False,
    )
    foreground = feature_values[foreground_idx]
    background = feature_values[background_idx]
    names = tuple(config.variable_names)
    n_features = len(names)

    subset_cache: dict[tuple[tuple[int, ...], int], np.ndarray] = {}

    def value(subset: tuple[int, ...], target_idx: int) -> np.ndarray:
        key = (tuple(sorted(subset)), int(target_idx))
        if key not in subset_cache:
            subset_cache[key] = _subset_prediction_mean(
                model,
                foreground,
                background,
                subset=key[0],
                target_idx=target_idx,
            )
        return subset_cache[key]

    attribution_rows: list[dict[str, object]] = []
    all_indices = tuple(range(n_features))
    for target_idx, target in enumerate(names):
        for source_idx, source in enumerate(names):
            phi = np.zeros(len(foreground), dtype=float)
            remaining = tuple(idx for idx in all_indices if idx != source_idx)
            for subset_size in range(len(remaining) + 1):
                for subset in combinations(remaining, subset_size):
                    with_source = tuple(sorted((*subset, source_idx)))
                    weight = _shapley_kernel_weight(subset_size, n_features)
                    phi += weight * (value(with_source, target_idx) - value(tuple(subset), target_idx))
            attribution_rows.append(
                {
                    "method": "interventional_shap",
                    "source": source,
                    "target": target,
                    "mean_abs_phi": float(np.mean(np.abs(phi))),
                    "mean_phi": float(np.mean(phi)),
                }
            )

    shap_interaction_rows: list[dict[str, object]] = []
    for target_idx, target in enumerate(names):
        for source_a_idx, source_b_idx in combinations(range(n_features), 2):
            interaction = np.zeros(len(foreground), dtype=float)
            remaining = tuple(
                idx for idx in all_indices if idx not in {source_a_idx, source_b_idx}
            )
            for subset_size in range(len(remaining) + 1):
                for subset in combinations(remaining, subset_size):
                    subset_with_a = tuple(sorted((*subset, source_a_idx)))
                    subset_with_b = tuple(sorted((*subset, source_b_idx)))
                    subset_with_ab = tuple(sorted((*subset, source_a_idx, source_b_idx)))
                    delta = (
                        value(subset_with_ab, target_idx)
                        - value(subset_with_a, target_idx)
                        - value(subset_with_b, target_idx)
                        + value(tuple(subset), target_idx)
                    )
                    weight = _shapley_kernel_weight(subset_size, n_features - 1)
                    interaction += weight * delta
            shap_interaction_rows.append(
                {
                    "method": "interventional_shap_interaction",
                    "sources": f"{names[source_a_idx]}+{names[source_b_idx]}",
                    "target": target,
                    "term": f"{names[source_a_idx]}:{names[source_b_idx]}",
                    "mean_abs_interaction": float(np.mean(np.abs(interaction))),
                    "mean_interaction": float(np.mean(interaction)),
                }
            )

    samples = _sample_intervention_sources(series, config)
    intervention_features = _intervention_features(samples, config)
    predictions = model.predict(intervention_features)
    source_matrix = samples[list(names)].to_numpy(dtype=float)
    source_mean = source_matrix.mean(axis=0, keepdims=True)
    source_std = source_matrix.std(axis=0, keepdims=True)
    source_std = np.where(source_std > 1e-8, source_std, 1.0)
    source_z = (source_matrix - source_mean) / source_std
    main_design = np.column_stack([np.ones(len(source_z)), source_z])
    interaction_rows: list[dict[str, object]] = []
    for target_idx, target in enumerate(names):
        y = predictions[:, target_idx]
        _, base_pred, _, base_r2 = _fit_ols_with_intercept(source_z, y)
        _ = base_pred
        for source_a_idx, source_b_idx in combinations(range(n_features), 2):
            interaction = (source_z[:, source_a_idx] * source_z[:, source_b_idx]).reshape(-1, 1)
            design = np.column_stack([main_design, interaction])
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            predicted = design @ coefficients
            mse = float(np.mean((y - predicted) ** 2))
            baseline_mse = float(np.mean((y - float(np.mean(y))) ** 2))
            r2 = 1.0 - mse / (baseline_mse + 1e-12)
            interaction_rows.append(
                {
                    "method": "standardized_product_probe",
                    "sources": f"{names[source_a_idx]}+{names[source_b_idx]}",
                    "target": target,
                    "term": f"{names[source_a_idx]}:{names[source_b_idx]}",
                    "main_effect_r2": float(base_r2),
                    "with_interaction_r2": float(r2),
                    "incremental_r2": float(max(0.0, r2 - base_r2)),
                    "interaction_coef": float(coefficients[-1]),
                }
            )

    return ShapReadout(
        feature_attributions=pd.DataFrame(attribution_rows).sort_values(
            "mean_abs_phi", ascending=False
        ).reset_index(drop=True),
        shap_interaction_terms=pd.DataFrame(shap_interaction_rows).sort_values(
            "mean_abs_interaction", ascending=False
        ).reset_index(drop=True),
        interaction_terms=pd.DataFrame(interaction_rows).sort_values(
            "incremental_r2", ascending=False
        ).reset_index(drop=True),
    )


def estimate_conditional_shap_readout(
    model: TrainedMLPTransition,
    features: np.ndarray,
    config: SimConfig,
    *,
    target: str = "y",
    foreground_samples: int = 64,
    background_samples: int = 384,
    neighbors: int = 64,
) -> ConditionalShapReadout:
    """Approximate observational SHAP by conditioning missing features with nearest neighbors."""

    rng = np.random.default_rng(int(config.seed) + 7079)
    feature_values = np.asarray(features, dtype=float)
    if config.lag != 1:
        raise ValueError("conditional SHAP readout is currently defined for lag=1 examples.")
    if len(feature_values) == 0:
        raise ValueError("features must not be empty.")
    names = tuple(config.variable_names)
    if target not in names:
        raise ValueError(f"unknown target {target!r}.")
    target_idx = names.index(target)
    foreground_idx = rng.choice(
        len(feature_values),
        size=min(int(foreground_samples), len(feature_values)),
        replace=False,
    )
    background_idx = rng.choice(
        len(feature_values),
        size=min(int(background_samples), len(feature_values)),
        replace=False,
    )
    foreground = feature_values[foreground_idx]
    background = feature_values[background_idx]
    bg_mean = background.mean(axis=0, keepdims=True)
    bg_std = background.std(axis=0, keepdims=True)
    bg_std = np.where(bg_std > 1e-8, bg_std, 1.0)
    background_z = (background - bg_mean) / bg_std
    foreground_z = (foreground - bg_mean) / bg_std
    n_features = len(names)
    all_indices = tuple(range(n_features))
    k = max(1, min(int(neighbors), len(background)))
    value_cache: dict[tuple[int, tuple[int, ...]], float] = {}

    def conditional_value(row_idx: int, subset: tuple[int, ...]) -> float:
        key = (int(row_idx), tuple(sorted(subset)))
        if key in value_cache:
            return value_cache[key]
        row = foreground[row_idx]
        if not key[1]:
            conditional_samples = background.copy()
        else:
            cols = list(key[1])
            distances = np.sum((background_z[:, cols] - foreground_z[row_idx, cols]) ** 2, axis=1)
            nearest_idx = np.argsort(distances)[:k]
            conditional_samples = background[nearest_idx].copy()
            for feature_idx in key[1]:
                conditional_samples[:, feature_idx] = row[feature_idx]
        prediction = float(np.mean(model.predict(conditional_samples)[:, target_idx]))
        value_cache[key] = prediction
        return prediction

    rows: list[dict[str, object]] = []
    for source_idx, source in enumerate(names):
        phi = np.zeros(len(foreground), dtype=float)
        remaining = tuple(idx for idx in all_indices if idx != source_idx)
        for row_idx in range(len(foreground)):
            value_sum = 0.0
            for subset_size in range(len(remaining) + 1):
                for subset in combinations(remaining, subset_size):
                    with_source = tuple(sorted((*subset, source_idx)))
                    weight = _shapley_kernel_weight(subset_size, n_features)
                    value_sum += weight * (
                        conditional_value(row_idx, with_source)
                        - conditional_value(row_idx, tuple(subset))
                    )
            phi[row_idx] = value_sum
        rows.append(
            {
                "method": "conditional_shap",
                "source": source,
                "target": target,
                "mean_abs_phi": float(np.mean(np.abs(phi))),
                "mean_phi": float(np.mean(phi)),
            }
        )
    return ConditionalShapReadout(
        feature_attributions=pd.DataFrame(rows).sort_values(
            "mean_abs_phi", ascending=False
        ).reset_index(drop=True)
    )


def _fit_ols_with_intercept(features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    y = np.asarray(target, dtype=float).reshape(-1)
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    predictions = design @ coefficients
    mse = float(np.mean((y - predictions) ** 2))
    baseline_mse = float(np.mean((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - mse / (baseline_mse + 1e-12)
    return coefficients, predictions, mse, float(r2)


def _directed_edge_key(source: str, target: str, metric: str) -> str:
    return f"{metric}_{source}_to_{target}"


def _fit_small_multitask_mlp(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    hidden_dim: int = 32,
    epochs: int = 400,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
) -> dict[str, object]:
    import torch

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)

    x = np.asarray(features, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    y = np.asarray(targets, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std = np.where(x_std > 1e-8, x_std, 1.0)
    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True)
    y_std = np.where(y_std > 1e-8, y_std, 1.0)
    x_scaled = (x - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std

    net = torch.nn.Sequential(
        torch.nn.Linear(x_scaled.shape[1], int(hidden_dim)),
        torch.nn.Tanh(),
        torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
        torch.nn.Tanh(),
        torch.nn.Linear(int(hidden_dim), y_scaled.shape[1]),
    )
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    x_tensor = torch.tensor(x_scaled.tolist(), dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled.tolist(), dtype=torch.float32)
    loss_history: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        prediction = net(x_tensor)
        loss = torch.mean((prediction - y_tensor) ** 2)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().item()))

    def predict(new_features: np.ndarray) -> np.ndarray:
        values = np.asarray(new_features, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        scaled = (values - x_mean) / x_std
        net.eval()
        with torch.no_grad():
            predicted = np.asarray(
                net(torch.tensor(scaled.tolist(), dtype=torch.float32)).tolist(),
                dtype=np.float32,
            )
        return predicted * y_std + y_mean

    train_prediction = predict(x)
    train_mse_by_target = np.mean((train_prediction - y) ** 2, axis=0)
    return {
        "predict": predict,
        "train_mse": float(np.mean(train_mse_by_target)),
        "train_mse_by_target": [float(value) for value in train_mse_by_target],
        "loss_history": loss_history,
    }


def run_lagged_proxy_common_driver_experiment(
    *,
    n_samples: int = 5000,
    noise: float = 0.05,
    seed: int = 0,
    bins: int = 8,
    intervention_samples: int = 4096,
) -> dict[str, object]:
    """Show that lagged proxy Granger can be spurious in a fitted-MLP PEID setup.

    The generative mechanism is w_t -> x_{t+1} and w_t -> y_{t+2}; there is no
    direct x_{t+1} -> y_{t+2} mechanism. Pairwise Granger sees x as a proxy for
    the hidden driver w_t. PEID is computed on one fitted multitask MLP rather
    than by evaluating the known structural equations directly.
    """

    if n_samples < 32:
        raise ValueError("n_samples must be at least 32.")
    if noise < 0.0:
        raise ValueError("noise must be nonnegative.")
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    if intervention_samples < 16:
        raise ValueError("intervention_samples must be at least 16.")

    rng = np.random.default_rng(int(seed))
    w = rng.normal(0.0, 1.0, size=n_samples)
    x = rng.normal(0.0, noise, size=n_samples)
    y = rng.normal(0.0, noise, size=n_samples)
    x[1:] = 0.9 * w[:-1] + rng.normal(0.0, noise, size=n_samples - 1)
    y[2:] = 0.7 * w[:-2] + rng.normal(0.0, noise, size=n_samples - 2)

    # Align as (x_{t+1}, w_t) -> y_{t+2}. Pairwise Granger omits w_t.
    x_proxy = x[1:-1]
    w_driver = w[:-2]
    y_future = y[2:]

    variables = {
        "w": w_driver,
        "x": x_proxy,
        "y": y_future,
    }

    pairwise_linear_edges: dict[str, float] = {}
    pairwise_linear_r2: dict[str, float] = {}
    for source, target in product(variables, variables):
        if source == target:
            continue
        target_values = variables[target]
        baseline_target_mse = float(np.mean((target_values - float(np.mean(target_values))) ** 2))
        _, _, target_mse, target_r2 = _fit_ols_with_intercept(variables[source], target_values)
        pairwise_linear_edges[f"{source}->{target}"] = float(max(0.0, baseline_target_mse - target_mse))
        pairwise_linear_r2[f"{source}->{target}"] = float(target_r2)

    baseline_mse = float(np.mean((y_future - float(np.mean(y_future))) ** 2))
    pair_coef, _, pairwise_mse, pairwise_r2 = _fit_ols_with_intercept(x_proxy, y_future)
    w_coef, _, w_only_mse, w_only_r2 = _fit_ols_with_intercept(w_driver, y_future)
    causal_coef, _, causal_mse, causal_r2 = _fit_ols_with_intercept(
        np.column_stack([x_proxy, w_driver]),
        y_future,
    )
    pairwise_granger_score = max(0.0, baseline_mse - pairwise_mse)
    conditional_x_incremental_score = max(0.0, w_only_mse - causal_mse)

    intervention_rng = np.random.default_rng(int(seed) + 2003)

    def sample_uniform_like(values: np.ndarray) -> np.ndarray:
        low = float(np.quantile(values, 0.05))
        high = float(np.quantile(values, 0.95))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.min(values)), float(np.max(values))
        return intervention_rng.uniform(low, high, size=int(intervention_samples))

    lagged_series = _make_lagged_proxy_series(n_samples, noise, seed)
    mlp_features, mlp_targets = _make_reverse_lag_dataset(lagged_series, max_lag=2)
    peid_mlp = _fit_small_multitask_mlp(
        mlp_features,
        mlp_targets,
        seed=int(seed) + 3001,
    )

    names = ("w", "x", "y")
    base_prediction = peid_mlp["predict"](mlp_features)
    pairwise_granger_edges: dict[str, float] = {}
    for source_idx, source in enumerate(names):
        ablated_features = np.asarray(mlp_features, dtype=float).copy()
        for lag_idx in range(2):
            col = lag_idx * len(names) + source_idx
            ablated_features[:, col] = float(np.mean(mlp_features[:, col]))
        ablated_prediction = peid_mlp["predict"](ablated_features)
        for target_idx, target in enumerate(names):
            if source == target:
                continue
            base_mse = float(np.mean((mlp_targets[:, target_idx] - base_prediction[:, target_idx]) ** 2))
            ablated_mse = float(np.mean((mlp_targets[:, target_idx] - ablated_prediction[:, target_idx]) ** 2))
            pairwise_granger_edges[f"{source}->{target}"] = float(max(0.0, ablated_mse - base_mse))

    intervention_features = np.zeros((int(intervention_samples), mlp_features.shape[1]), dtype=float)
    intervention_states_by_source: dict[str, np.ndarray] = {}
    for source_idx, source in enumerate(("w", "x", "y")):
        lag_states: list[np.ndarray] = []
        for lag_idx in range(2):
            col = lag_idx * 3 + source_idx
            intervention_features[:, col] = sample_uniform_like(mlp_features[:, col])
            lag_states.append(_discretize_vector(intervention_features[:, col], bins))
        intervention_states_by_source[source] = np.column_stack(lag_states)
    mlp_predictions = peid_mlp["predict"](intervention_features)
    target_states_by_target = {
        name: _discretize_vector(mlp_predictions[:, idx], bins)
        for idx, name in enumerate(names)
    }
    peid_ei_edges: dict[str, float] = {}
    for source, target in product(names, names):
        if source == target:
            continue
        peid_ei_edges[f"{source}->{target}"] = float(
            max(
                0.0,
                _effective_information_from_states(
                    intervention_states_by_source[source],
                    target_states_by_target[target],
                ),
            )
        )
    peid_ei_x_to_y = peid_ei_edges["x->y"]
    peid_ei_w_to_y = peid_ei_edges["w->y"]

    result: dict[str, object] = {
        "n_samples": float(n_samples),
        "noise": float(noise),
        "seed": float(seed),
        "pairwise_granger_edges": pairwise_granger_edges,
        "pairwise_linear_edges": pairwise_linear_edges,
        "pairwise_linear_r2": pairwise_linear_r2,
        "peid_ei_edges": peid_ei_edges,
        "pairwise_granger_x_to_y_score": float(pairwise_granger_edges["x->y"]),
        "pairwise_linear_x_to_y_score": float(pairwise_granger_score),
        "pairwise_linear_x_to_y_r2": float(pairwise_r2),
        "pairwise_granger_x_to_y_coef": float(pair_coef[1]),
        "conditional_x_incremental_score_given_w": float(conditional_x_incremental_score),
        "causal_state_r2": float(causal_r2),
        "causal_state_coef_x_proxy": float(causal_coef[1]),
        "causal_state_coef_w_driver": float(causal_coef[2]),
        "peid_mlp_train_mse": float(peid_mlp["train_mse"]),
        "peid_mlp_w_train_mse": float(peid_mlp["train_mse_by_target"][0]),
        "peid_mlp_x_train_mse": float(peid_mlp["train_mse_by_target"][1]),
        "peid_mlp_y_train_mse": float(peid_mlp["train_mse_by_target"][2]),
        "w_only_r2": float(w_only_r2),
        "w_only_coef": float(w_coef[1]),
        "peid_ei_x_to_y": float(peid_ei_x_to_y),
        "peid_ei_w_to_y": float(peid_ei_w_to_y),
        "peid_proxy_to_driver_ratio": float(peid_ei_x_to_y / (peid_ei_w_to_y + 1e-12)),
    }
    for edge, value in pairwise_granger_edges.items():
        source, target = edge.split("->")
        result[_directed_edge_key(source, target, "pairwise_granger")] = value
    for edge, value in peid_ei_edges.items():
        source, target = edge.split("->")
        result[_directed_edge_key(source, target, "peid_ei")] = value
    return result


def _make_lagged_proxy_series(n_samples: int, noise: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    w = rng.normal(0.0, 1.0, size=n_samples)
    x = rng.normal(0.0, noise, size=n_samples)
    y = rng.normal(0.0, noise, size=n_samples)
    x[1:] = 0.9 * w[:-1] + rng.normal(0.0, noise, size=n_samples - 1)
    y[2:] = 0.7 * w[:-2] + rng.normal(0.0, noise, size=n_samples - 2)
    return pd.DataFrame({"w": w, "x": x, "y": y})


def _make_reverse_lag_dataset(series: pd.DataFrame, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    values = series[["w", "x", "y"]].to_numpy(dtype=float)
    features = np.asarray(
        [values[t - max_lag : t][::-1].reshape(-1) for t in range(max_lag, len(values))],
        dtype=float,
    )
    targets = values[max_lag:]
    return features, targets


def _fitted_mlp_lagged_proxy_readout(
    series: pd.DataFrame,
    *,
    max_lag: int,
    seed: int,
    bins: int,
    intervention_samples: int,
) -> dict[str, object]:
    if max_lag < 1:
        raise ValueError("max_lag must be positive.")
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    if intervention_samples < 16:
        raise ValueError("intervention_samples must be at least 16.")

    names = ("w", "x", "y")
    features, targets = _make_reverse_lag_dataset(series, max_lag=max_lag)
    fitted_mlp = _fit_small_multitask_mlp(
        features,
        targets,
        seed=int(seed) + 3001 + 101 * int(max_lag),
    )
    base_prediction = fitted_mlp["predict"](features)
    granger_edges: dict[str, float] = {}
    for source_idx, source in enumerate(names):
        ablated_features = np.asarray(features, dtype=float).copy()
        for lag_idx in range(max_lag):
            col = lag_idx * len(names) + source_idx
            ablated_features[:, col] = float(np.mean(features[:, col]))
        ablated_prediction = fitted_mlp["predict"](ablated_features)
        for target_idx, target in enumerate(names):
            if source == target:
                continue
            base_mse = float(np.mean((targets[:, target_idx] - base_prediction[:, target_idx]) ** 2))
            ablated_mse = float(np.mean((targets[:, target_idx] - ablated_prediction[:, target_idx]) ** 2))
            granger_edges[f"{source}->{target}"] = float(max(0.0, ablated_mse - base_mse))

    intervention_rng = np.random.default_rng(int(seed) + 2003 + 101 * int(max_lag))

    def sample_uniform_like(values: np.ndarray) -> np.ndarray:
        low = float(np.quantile(values, 0.05))
        high = float(np.quantile(values, 0.95))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.min(values)), float(np.max(values))
        return intervention_rng.uniform(low, high, size=int(intervention_samples))

    intervention_features = np.zeros((int(intervention_samples), features.shape[1]), dtype=float)
    intervention_states_by_source: dict[str, np.ndarray] = {}
    for source_idx, source in enumerate(names):
        lag_states: list[np.ndarray] = []
        for lag_idx in range(max_lag):
            col = lag_idx * len(names) + source_idx
            intervention_features[:, col] = sample_uniform_like(features[:, col])
            lag_states.append(_discretize_vector(intervention_features[:, col], bins))
        intervention_states_by_source[source] = np.column_stack(lag_states)
    intervention_predictions = fitted_mlp["predict"](intervention_features)
    target_states_by_target = {
        name: _discretize_vector(intervention_predictions[:, idx], bins)
        for idx, name in enumerate(names)
    }
    peid_ei_edges: dict[str, float] = {}
    for source, target in product(names, names):
        if source == target:
            continue
        peid_ei_edges[f"{source}->{target}"] = float(
            _effective_information_from_states(
                intervention_states_by_source[source],
                target_states_by_target[target],
            )
        )

    return {
        "max_lag": int(max_lag),
        "granger_edges": granger_edges,
        "peid_ei_edges": peid_ei_edges,
        "mlp_train_mse": float(fitted_mlp["train_mse"]),
        "mlp_train_mse_by_target": list(fitted_mlp["train_mse_by_target"]),
    }


def run_lag_sensitivity_lagged_proxy_experiment(
    *,
    n_samples: int = 5000,
    noise: float = 0.05,
    seed: int = 0,
    bins: int = 8,
    intervention_samples: int = 4096,
) -> dict[str, object]:
    if n_samples < 64:
        raise ValueError("n_samples must be at least 64.")
    if noise < 0.0:
        raise ValueError("noise must be nonnegative.")
    series = _make_lagged_proxy_series(n_samples, noise, seed)
    by_lag = {
        max_lag: _fitted_mlp_lagged_proxy_readout(
            series,
            max_lag=max_lag,
            seed=seed,
            bins=bins,
            intervention_samples=intervention_samples,
        )
        for max_lag in (1, 2)
    }
    return {
        "n_samples": int(n_samples),
        "noise": float(noise),
        "seed": int(seed),
        "by_lag": by_lag,
    }


def _fit_componentwise_neural_granger(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    target_idx: int,
    max_lag: int,
    variable_names: Sequence[str],
    group_lasso: float,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    """Fit one cMLP target model and read first-layer source-group norms."""

    import torch

    names = tuple(str(name) for name in variable_names)
    if not names:
        raise ValueError("variable_names must not be empty.")
    if max_lag < 1:
        raise ValueError("max_lag must be positive.")

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets[:, target_idx : target_idx + 1], dtype=np.float32)
    expected_features = int(max_lag) * len(names)
    if x.ndim != 2 or x.shape[1] != expected_features:
        raise ValueError(
            f"features must have {expected_features} columns for {len(names)} variables and max_lag={max_lag}."
        )
    if targets.shape[1] != len(names):
        raise ValueError("targets width must match variable_names.")
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std = np.where(x_std > 1e-8, x_std, 1.0)
    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True)
    y_std = np.where(y_std > 1e-8, y_std, 1.0)
    x_scaled = (x - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std

    net = torch.nn.Sequential(
        torch.nn.Linear(x_scaled.shape[1], int(hidden_dim)),
        torch.nn.Tanh(),
        torch.nn.Linear(int(hidden_dim), 1),
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=float(learning_rate))
    x_tensor = torch.tensor(x_scaled.tolist(), dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled.tolist(), dtype=torch.float32)
    loss_value = 0.0
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        prediction = net(x_tensor)
        mse = torch.mean((prediction - y_tensor) ** 2)
        first_layer = net[0].weight
        penalty = sum(
            torch.linalg.vector_norm(
                first_layer[:, [lag_idx * len(names) + source_idx for lag_idx in range(max_lag)]]
            )
            for source_idx in range(len(names))
        )
        loss = mse + float(group_lasso) * penalty
        loss.backward()
        optimizer.step()
        loss_value = float(mse.detach().item())

    first_layer_weights = np.asarray(net[0].weight.detach().tolist(), dtype=float)
    group_norms: dict[str, float] = {}
    lag_norms: dict[str, list[float]] = {}
    for source_idx, source in enumerate(names):
        cols = [lag_idx * len(names) + source_idx for lag_idx in range(max_lag)]
        group_norms[source] = float(np.linalg.norm(first_layer_weights[:, cols]))
        lag_norms[source] = [
            float(np.linalg.norm(first_layer_weights[:, lag_idx * len(names) + source_idx]))
            for lag_idx in range(max_lag)
        ]
    return {
        "group_norms": group_norms,
        "lag_norms": lag_norms,
        "scaled_mse": loss_value,
    }


def run_neural_granger_readout(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    variable_names: Sequence[str],
    max_lag: int,
    group_lasso: float = 0.03,
    hidden_dim: int = 12,
    epochs: int = 120,
    learning_rate: float = 0.01,
    seed: int = 0,
) -> dict[str, object]:
    """Run component-wise cMLP Neural Granger and return ranked source norms."""

    names = tuple(str(name) for name in variable_names)
    if not names:
        raise ValueError("variable_names must not be empty.")
    if int(max_lag) < 1:
        raise ValueError("max_lag must be positive.")
    if float(group_lasso) < 0.0:
        raise ValueError("group_lasso must be nonnegative.")

    rows: list[dict[str, object]] = []
    for target_idx, target in enumerate(names):
        fit = _fit_componentwise_neural_granger(
            features,
            targets,
            target_idx=target_idx,
            max_lag=int(max_lag),
            variable_names=names,
            group_lasso=float(group_lasso),
            hidden_dim=int(hidden_dim),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            seed=int(seed) + target_idx,
        )
        group_norms = dict(fit["group_norms"])
        lag_norms = dict(fit["lag_norms"])
        ordered_sources = sorted(group_norms, key=lambda name: group_norms[name], reverse=True)
        for rank, source in enumerate(ordered_sources, start=1):
            source_lag_norms = [float(value) for value in lag_norms[source]]
            strongest_lag_idx = int(np.argmax(source_lag_norms)) + 1
            rows.append(
                {
                    "method": "neural_granger",
                    "source": source,
                    "target": target,
                    "rank": int(rank),
                    "group_norm": float(group_norms[source]),
                    "strongest_lag": int(strongest_lag_idx),
                    "strongest_lag_norm": float(max(source_lag_norms)),
                    "lag_norms": source_lag_norms,
                    "scaled_mse": float(fit["scaled_mse"]),
                }
            )
    return {
        "variable_names": list(names),
        "max_lag": int(max_lag),
        "group_lasso": float(group_lasso),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "rows": rows,
    }


def run_neural_granger_lagged_proxy_experiment(
    *,
    n_samples: int = 3000,
    noise: float = 0.05,
    seed: int = 0,
    model_seed: int = 1,
    max_lags: Sequence[int] = (1, 2),
    group_lasso: float = 0.03,
    hidden_dim: int = 12,
    epochs: int = 250,
    learning_rate: float = 0.01,
) -> dict[str, object]:
    """Run a mainstream cMLP neural-Granger readout on the lagged-proxy example."""

    if n_samples < 64:
        raise ValueError("n_samples must be at least 64.")
    if noise < 0.0:
        raise ValueError("noise must be nonnegative.")
    if group_lasso < 0.0:
        raise ValueError("group_lasso must be nonnegative.")

    series = _make_lagged_proxy_series(n_samples, noise, seed)
    rows: list[dict[str, object]] = []
    for max_lag in max_lags:
        if max_lag < 1:
            raise ValueError("max_lags must contain positive integers.")
        features, targets = _make_reverse_lag_dataset(series, int(max_lag))
        readout = run_neural_granger_readout(
            features,
            targets,
            variable_names=("w", "x", "y"),
            max_lag=int(max_lag),
            group_lasso=float(group_lasso),
            hidden_dim=int(hidden_dim),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            seed=int(model_seed + 17 * max_lag),
        )
        for row in readout["rows"]:
            rows.append({**row, "max_lag": int(max_lag)})

    def top_source(max_lag: int, target: str) -> str:
        candidates = [
            row for row in rows if row["max_lag"] == max_lag and row["target"] == target and row["rank"] == 1
        ]
        return str(candidates[0]["source"]) if candidates else ""

    return {
        "n_samples": int(n_samples),
        "noise": float(noise),
        "seed": int(seed),
        "model_seed": int(model_seed),
        "group_lasso": float(group_lasso),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "rows": rows,
        "lag1_y_top_source": top_source(1, "y"),
        "lag2_y_top_source": top_source(2, "y"),
        "truth": {
            "w->x": "lag 1",
            "w->y": "lag 2",
            "x->y": "absent",
        },
    }


def _f1_scores(predicted: set[tuple[str, ...]], truth: set[tuple[str, ...]]) -> dict[str, float]:
    if not truth:
        false_positive_rate = 1.0 if predicted else 0.0
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "false_positive_rate": false_positive_rate, "miss_rate": 0.0}
    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    miss_rate = 1.0 - recall if truth else 0.0
    false_positive_rate = (len(predicted - truth) / len(predicted)) if predicted else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(false_positive_rate),
        "miss_rate": float(miss_rate),
    }


def _top_pairwise(edges: pd.DataFrame, score_col: str, k: int) -> set[tuple[str, str]]:
    if k <= 0 or edges.empty:
        return set()
    top = edges.sort_values(score_col, ascending=False).head(k)
    return {(str(row.source), str(row.target)) for row in top.itertuples(index=False)}


def _top_hyperedges(edges: pd.DataFrame, k: int) -> set[tuple[str, str, str]]:
    if k <= 0 or edges.empty:
        return set()
    top = edges.sort_values("synergy", ascending=False).head(k)
    parsed: set[tuple[str, str, str]] = set()
    for row in top.itertuples(index=False):
        sources = str(row.sources).split("+")
        if len(sources) == 2:
            parsed.add((sources[0], sources[1], str(row.target)))
    return parsed


def _evaluate_run(
    truth: dict[str, object],
    granger_edges: pd.DataFrame,
    peid: PeidGraph,
) -> dict[str, float]:
    truth_pairwise = {tuple(edge) for edge in truth["pairwise_edges"]}
    truth_hyperedges = {tuple(edge) for edge in truth["hyperedges"]}
    granger_pred = _top_pairwise(granger_edges, "score", len(truth_pairwise))
    peid_pair_pred = _top_pairwise(peid.pairwise_edges, "ei", len(truth_pairwise))
    peid_hyper_pred = _top_hyperedges(peid.synergy_edges, len(truth_hyperedges))
    granger_metrics = _f1_scores(granger_pred, truth_pairwise)
    peid_pair_metrics = _f1_scores(peid_pair_pred, truth_pairwise)
    peid_hyper_metrics = _f1_scores(peid_hyper_pred, truth_hyperedges)
    union = granger_pred | peid_pair_pred
    overlap = len(granger_pred & peid_pair_pred) / len(union) if union else 1.0
    return {
        "granger_pairwise_f1": granger_metrics["f1"],
        "granger_pairwise_precision": granger_metrics["precision"],
        "granger_pairwise_recall": granger_metrics["recall"],
        "granger_pairwise_miss_rate": granger_metrics["miss_rate"],
        "granger_pairwise_false_positive_rate": granger_metrics["false_positive_rate"],
        "peid_pairwise_f1": peid_pair_metrics["f1"],
        "peid_pairwise_precision": peid_pair_metrics["precision"],
        "peid_pairwise_recall": peid_pair_metrics["recall"],
        "peid_hyperedge_f1": peid_hyper_metrics["f1"],
        "peid_hyperedge_recall": peid_hyper_metrics["recall"],
        "peid_advantage": peid_hyper_metrics["f1"] - granger_metrics["f1"],
        "disagreement_score": float(1.0 - overlap),
    }


def _edge_records(
    run_id: str,
    config: SimConfig,
    granger_edges: pd.DataFrame,
    peid: PeidGraph,
    shap_readout: ShapReadout | None = None,
    conditional_shap_readout: ConditionalShapReadout | None = None,
    neural_granger_readout: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    base = {
        "run_id": run_id,
        "mechanism": config.mechanism,
        "seed": int(config.seed),
        "noise": float(config.noise),
        "n_samples": int(config.n_samples),
        "synergy_strength": float(config.synergy_strength),
    }
    rows: list[dict[str, object]] = []
    for row in granger_edges.to_dict("records"):
        rows.append({**base, "edge_type": "granger_pairwise", **row})
    for row in peid.pairwise_edges.to_dict("records"):
        rows.append({**base, "edge_type": "peid_pairwise", **row})
    for row in peid.synergy_edges.to_dict("records"):
        rows.append({**base, "edge_type": "peid_synergy", **row})
    if shap_readout is not None:
        for row in shap_readout.feature_attributions.to_dict("records"):
            rows.append({**base, "edge_type": "interventional_shap", **row})
        for row in shap_readout.shap_interaction_terms.to_dict("records"):
            rows.append({**base, "edge_type": "interventional_shap_interaction", **row})
        for row in shap_readout.interaction_terms.to_dict("records"):
            rows.append({**base, "edge_type": "product_interaction_probe", **row})
    if conditional_shap_readout is not None:
        for row in conditional_shap_readout.feature_attributions.to_dict("records"):
            rows.append({**base, "edge_type": "conditional_shap", **row})
    if neural_granger_readout is not None:
        for row in list(neural_granger_readout.get("rows", [])):
            rows.append({**base, "edge_type": "neural_granger", **dict(row)})
    return rows


def _plot_summary(runs: list[dict[str, object]], figure_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(runs)
    grouped = (
        frame.groupby("mechanism", as_index=False)[
            ["granger_pairwise_f1", "peid_pairwise_f1", "peid_hyperedge_f1", "peid_advantage"]
        ]
        .mean()
        .sort_values("mechanism")
    )
    x = np.arange(len(grouped))
    width = 0.22
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    ax.bar(x - width, grouped["granger_pairwise_f1"], width, label="Granger pairwise F1", color="#5b8db8")
    ax.bar(x, grouped["peid_pairwise_f1"], width, label="PEID pairwise F1", color="#d99a48")
    ax.bar(x + width, grouped["peid_hyperedge_f1"], width, label="PEID hyperedge F1", color="#6aa36f")
    for idx, value in enumerate(grouped["peid_advantage"]):
        ax.text(idx + width, min(1.05, grouped["peid_hyperedge_f1"].iloc[idx] + 0.03), f"adv {value:+.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("Mean score")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["mechanism"], rotation=20, ha="right")
    ax.set_title("Granger pairwise edges vs PEID pairwise and synergy hyperedges")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    path = figure_dir / "granger_vs_peid_summary.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_sine_readout_summary(edge_rows: list[dict[str, object]], figure_dir: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    edge_frame = pd.DataFrame(edge_rows)
    sine_runs = edge_frame.loc[
        edge_frame["mechanism"].eq("common_driver_sine_synergy"), "run_id"
    ].dropna()
    if sine_runs.empty:
        return None
    run_id = str(sine_runs.iloc[0])
    run_edges = edge_frame[edge_frame["run_id"] == run_id].copy()

    directed_edges = [("w", "x"), ("w", "y"), ("w", "z"), ("x", "z"), ("y", "z")]
    edge_labels = [f"{source}->{target}" for source, target in directed_edges]
    method_specs = [
        ("Granger", "granger_pairwise", "score"),
        ("Neural Granger", "neural_granger", "group_norm"),
        ("SHAP", "interventional_shap", "mean_abs_phi"),
        ("PEID", "peid_pairwise", "ei"),
    ]
    single_values = np.full((len(method_specs), len(directed_edges)), np.nan, dtype=float)
    for method_idx, (_, edge_type, score_col) in enumerate(method_specs):
        for edge_idx, (source, target) in enumerate(directed_edges):
            subset = run_edges[
                (run_edges["edge_type"] == edge_type)
                & (run_edges["source"] == source)
                & (run_edges["target"] == target)
            ]
            if not subset.empty and score_col in subset:
                single_values[method_idx, edge_idx] = float(subset.iloc[0][score_col])

    interaction_pairs = [("x+y", "x:y"), ("x+w", "w:x"), ("y+w", "w:y")]
    interaction_values = np.full((2, len(interaction_pairs)), np.nan, dtype=float)
    for pair_idx, (sources, _) in enumerate(interaction_pairs):
        shap_subset = run_edges[
            (run_edges["edge_type"] == "interventional_shap_interaction")
            & (run_edges["sources"] == sources)
            & (run_edges["target"] == "z")
        ]
        if not shap_subset.empty:
            interaction_values[0, pair_idx] = float(shap_subset.iloc[0]["mean_abs_interaction"])
        subset = run_edges[
            (run_edges["edge_type"] == "product_interaction_probe")
            & (run_edges["sources"] == sources)
            & (run_edges["target"] == "z")
        ]
        if not subset.empty:
            interaction_values[1, pair_idx] = float(subset.iloc[0]["incremental_r2"])

    peid_decomp_specs = [
        ("EI x->z", "peid_pairwise", "x", "z", "ei"),
        ("EI y->z", "peid_pairwise", "y", "z", "ei"),
        ("joint EI", "peid_synergy", "x+y", "z", "joint_ei"),
        ("synergy", "peid_synergy", "x+y", "z", "synergy"),
    ]
    peid_decomp = []
    for _, edge_type, source_or_sources, target, score_col in peid_decomp_specs:
        if edge_type == "peid_pairwise":
            subset = run_edges[
                (run_edges["edge_type"] == edge_type)
                & (run_edges["source"] == source_or_sources)
                & (run_edges["target"] == target)
            ]
        else:
            subset = run_edges[
                (run_edges["edge_type"] == edge_type)
                & (run_edges["sources"] == source_or_sources)
                & (run_edges["target"] == target)
            ]
        peid_decomp.append(float(subset.iloc[0][score_col]) if not subset.empty else np.nan)

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
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.6, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0])

    ax_single = fig.add_subplot(gs[:, 0])
    row_scaled = single_values.copy()
    for row_idx in range(row_scaled.shape[0]):
        row_max = np.nanmax(row_scaled[row_idx])
        if np.isfinite(row_max) and row_max > 0.0:
            row_scaled[row_idx] = row_scaled[row_idx] / row_max
    im = ax_single.imshow(row_scaled, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax_single.set_xticks(np.arange(len(edge_labels)))
    ax_single.set_xticklabels(edge_labels, rotation=35, ha="right")
    ax_single.set_yticks(np.arange(len(method_specs)))
    ax_single.set_yticklabels([spec[0] for spec in method_specs])
    ax_single.set_title("Single-source readouts", fontsize=9, fontweight="bold", pad=6)
    ax_single.set_xlabel("Directed source-target readout")
    for row_idx in range(single_values.shape[0]):
        for col_idx in range(single_values.shape[1]):
            value = single_values[row_idx, col_idx]
            if np.isfinite(value):
                color = "white" if row_scaled[row_idx, col_idx] > 0.62 else "#202020"
                ax_single.text(col_idx, row_idx, f"{value:.3g}", ha="center", va="center", fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=ax_single, fraction=0.046, pad=0.02)
    cbar.set_label("Row-normalized intensity", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    ax_interaction = fig.add_subplot(gs[0, 1])
    row_scaled_interactions = interaction_values.copy()
    for row_idx in range(row_scaled_interactions.shape[0]):
        row_max = np.nanmax(row_scaled_interactions[row_idx])
        if np.isfinite(row_max) and row_max > 0.0:
            row_scaled_interactions[row_idx] = row_scaled_interactions[row_idx] / row_max
    im2 = ax_interaction.imshow(row_scaled_interactions, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax_interaction.set_xticks(np.arange(len(interaction_pairs)))
    ax_interaction.set_xticklabels([label for _, label in interaction_pairs])
    ax_interaction.set_yticks([0, 1])
    ax_interaction.set_yticklabels(["SHAP |I|", "probe R2"])
    ax_interaction.set_title("Pair interaction readouts to z", fontsize=9, fontweight="bold", pad=6)
    for row_idx in range(interaction_values.shape[0]):
        for col_idx, value in enumerate(interaction_values[row_idx]):
            if np.isfinite(value):
                ax_interaction.text(
                    col_idx,
                    row_idx,
                    f"{value:.3g}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if row_scaled_interactions[row_idx, col_idx] > 0.62 else "#202020",
                )
    cbar2 = fig.colorbar(im2, ax=ax_interaction, fraction=0.08, pad=0.03)
    cbar2.set_label("Row-normalized intensity", fontsize=7)
    cbar2.ax.tick_params(labelsize=7)

    ax_peid = fig.add_subplot(gs[1, 1])
    y_pos = np.arange(len(peid_decomp))
    colors = ["#9bb7d4", "#9bb7d4", "#5f8f6b", "#2f6f4e"]
    ax_peid.barh(y_pos, peid_decomp, color=colors)
    ax_peid.set_yticks(y_pos)
    ax_peid.set_yticklabels([spec[0] for spec in peid_decomp_specs])
    ax_peid.invert_yaxis()
    ax_peid.set_xlabel("bits")
    ax_peid.set_title("PEID decomposition to z", fontsize=9, fontweight="bold", pad=6)
    xmax = max(value for value in peid_decomp if np.isfinite(value))
    ax_peid.set_xlim(0.0, xmax * 1.18)
    for y_idx, value in enumerate(peid_decomp):
        if np.isfinite(value):
            ax_peid.text(value + xmax * 0.025, y_idx, f"{value:.3g}", va="center", ha="left", fontsize=7)

    path = figure_dir / "sine_readout_2d_summary.png"
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return path


def run_sine_alpha_sweep(
    *,
    alpha_values: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    n_samples: int = 1300,
    noise: float = 0.05,
    seed: int = 0,
    mlp_epochs: int = 100,
    intervention_samples: int = 768,
    bins: int = 4,
    neural_granger_epochs: int = 120,
) -> list[dict[str, float]]:
    from yrd.transport_map import (
        clip_nonnegative_ei,
        estimate_mutual_information_transport_map,
        summarize_two_source_synergy_transport_map,
    )

    rows: list[dict[str, float]] = []
    for alpha in alpha_values:
        config = SimConfig(
            mechanism="common_driver_sine_synergy",
            n_samples=int(n_samples),
            noise=float(noise),
            seed=int(seed),
            synergy_strength=float(alpha),
            mlp_epochs=int(mlp_epochs),
            intervention_samples=int(intervention_samples),
            bins=int(bins),
        )
        series, _ = simulate_system(config)
        features, targets = make_lagged_dataset(series, lag=config.lag)
        model = train_mlp_transition_model(features, targets, config)
        granger = estimate_granger_graph(model, features, targets, config)
        peid = estimate_peid_graph(model, series, config)
        neural_granger = run_neural_granger_readout(
            features,
            targets,
            variable_names=config.variable_names,
            max_lag=config.lag,
            epochs=int(neural_granger_epochs),
            seed=int(seed) + 9201 + int(round(float(alpha) * 1000)),
        )
        tm_peid_xy_z = summarize_two_source_synergy_transport_map(
            peid.intervention_states[["x"]].to_numpy(dtype=float),
            peid.intervention_states[["y"]].to_numpy(dtype=float),
            peid.intervention_states[["z_pred"]].to_numpy(dtype=float),
        )
        tm_peid_w_to_z = clip_nonnegative_ei(
            float(
                estimate_mutual_information_transport_map(
                    peid.intervention_states[["w"]].to_numpy(dtype=float),
                    peid.intervention_states[["z_pred"]].to_numpy(dtype=float),
                )["mi_hat"]
            )
        )
        shap_readout = estimate_shap_readout(
            model,
            features,
            series,
            config,
            foreground_samples=72,
            background_samples=72,
        )
        peid_xy_z = peid.synergy_edges[
            (peid.synergy_edges["sources"] == "x+y")
            & (peid.synergy_edges["target"] == "z")
        ].iloc[0]
        shap_xy_z = shap_readout.shap_interaction_terms[
            (shap_readout.shap_interaction_terms["sources"] == "x+y")
            & (shap_readout.shap_interaction_terms["target"] == "z")
        ].iloc[0]
        shap_single_lookup = {
            str(row["source"]): float(row["mean_abs_phi"])
            for row in shap_readout.feature_attributions[
                shap_readout.feature_attributions["target"] == "z"
            ].to_dict("records")
        }
        product_xy_z = shap_readout.interaction_terms[
            (shap_readout.interaction_terms["sources"] == "x+y")
            & (shap_readout.interaction_terms["target"] == "z")
        ].iloc[0]
        granger_lookup = {
            str(row["source"]): float(row["score"])
            for row in granger[granger["target"] == "z"].to_dict("records")
        }
        neural_granger_lookup = {
            str(row["source"]): float(row["group_norm"])
            for row in list(neural_granger["rows"])
            if str(row["target"]) == "z"
        }
        rows.append(
            {
                "alpha": float(alpha),
                "final_train_loss": float(model.loss_history[-1]) if model.loss_history else float("nan"),
                "shap_x_to_z_mean_abs": float(shap_single_lookup.get("x", 0.0)),
                "shap_y_to_z_mean_abs": float(shap_single_lookup.get("y", 0.0)),
                "shap_w_to_z_mean_abs": float(shap_single_lookup.get("w", 0.0)),
                "shap_xy_mean_abs_interaction": float(shap_xy_z["mean_abs_interaction"]),
                "shap_xy_mean_interaction": float(shap_xy_z["mean_interaction"]),
                "product_xy_incremental_r2": float(product_xy_z["incremental_r2"]),
                "product_xy_coef": float(product_xy_z["interaction_coef"]),
                "granger_x_to_z": float(granger_lookup.get("x", 0.0)),
                "granger_y_to_z": float(granger_lookup.get("y", 0.0)),
                "granger_w_to_z": float(granger_lookup.get("w", 0.0)),
                "neural_granger_x_to_z": float(neural_granger_lookup.get("x", 0.0)),
                "neural_granger_y_to_z": float(neural_granger_lookup.get("y", 0.0)),
                "neural_granger_w_to_z": float(neural_granger_lookup.get("w", 0.0)),
                "peid_xy_joint_ei": float(peid_xy_z["joint_ei"]),
                "peid_xy_synergy": float(peid_xy_z["synergy"]),
                "tm_peid_xy_joint_ei": float(tm_peid_xy_z["joint_ei"]),
                "tm_peid_xy_synergy": float(tm_peid_xy_z["syn"]),
                "tm_peid_x_to_z": float(tm_peid_xy_z["left_ei"]),
                "tm_peid_y_to_z": float(tm_peid_xy_z["right_ei"]),
                "tm_peid_w_to_z": float(tm_peid_w_to_z),
                "peid_x_to_z": float(
                    peid.pairwise_edges[
                        (peid.pairwise_edges["source"] == "x")
                        & (peid.pairwise_edges["target"] == "z")
                    ].iloc[0]["ei"]
                ),
                "peid_y_to_z": float(
                    peid.pairwise_edges[
                        (peid.pairwise_edges["source"] == "y")
                        & (peid.pairwise_edges["target"] == "z")
                    ].iloc[0]["ei"]
                ),
            }
        )
    return rows


def _plot_sine_alpha_sweep(alpha_rows: list[dict[str, float]], figure_dir: Path) -> Path | None:
    if not alpha_rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(alpha_rows).sort_values("alpha")
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
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.0), constrained_layout=True)

    ax = axes[0]
    ax.plot(
        frame["alpha"],
        frame["shap_x_to_z_mean_abs"],
        marker="o",
        color="#4f7ca8",
        linewidth=1.8,
        label="SHAP x->z",
    )
    ax.plot(
        frame["alpha"],
        frame["shap_y_to_z_mean_abs"],
        marker="s",
        color="#c48f65",
        linewidth=1.8,
        label="SHAP y->z",
    )
    ax.plot(
        frame["alpha"],
        frame["shap_w_to_z_mean_abs"],
        marker="^",
        color="#8c8c8c",
        linewidth=1.4,
        label="SHAP w->z",
    )
    ax.plot(
        frame["alpha"],
        frame["shap_xy_mean_abs_interaction"],
        marker="D",
        color="#7b6aa8",
        linewidth=1.8,
        label="SHAP interaction (x,y)->z",
    )
    ax.set_xlabel("alpha in alpha * sin(x y)")
    ax.set_ylabel("mean |SHAP value|")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax = axes[1]
    ax.plot(
        frame["alpha"],
        frame["granger_x_to_z"],
        marker="o",
        color="#4f7ca8",
        linewidth=1.8,
        label="Granger x->z",
    )
    ax.plot(
        frame["alpha"],
        frame["granger_y_to_z"],
        marker="s",
        color="#c48f65",
        linewidth=1.8,
        label="Granger y->z",
    )
    ax.plot(
        frame["alpha"],
        frame["granger_w_to_z"],
        marker="^",
        color="#8c8c8c",
        linewidth=1.4,
        label="Granger w->z",
    )
    ax.set_xlabel("alpha in alpha * sin(x y)")
    ax.set_ylabel("ablation score")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax = axes[2]
    ax.plot(
        frame["alpha"],
        frame["tm_peid_xy_joint_ei"],
        marker="o",
        color="#5f8f6b",
        linewidth=1.8,
        label="TM PEID joint EI {x,y}->z",
    )
    ax.plot(
        frame["alpha"],
        frame["tm_peid_xy_synergy"],
        marker="s",
        color="#2f6f4e",
        linewidth=1.8,
        label="TM PEID synergy {x,y}->z",
    )
    ax.plot(
        frame["alpha"],
        frame["tm_peid_x_to_z"],
        marker="^",
        color="#9bb7d4",
        linewidth=1.2,
        label="TM PEID x->z",
    )
    ax.plot(
        frame["alpha"],
        frame["tm_peid_y_to_z"],
        marker="v",
        color="#c4a07a",
        linewidth=1.2,
        label="TM PEID y->z",
    )
    ax.plot(
        frame["alpha"],
        frame["tm_peid_w_to_z"],
        marker="P",
        color="#8c8c8c",
        linewidth=1.2,
        label="TM PEID w->z",
    )
    ax.set_xlabel("alpha in alpha * sin(x y)")
    ax.set_ylabel("Information (bits)")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    path = figure_dir / "sine_alpha_shap_peid_sweep.png"
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_sine_alpha_neural_granger_sweep(alpha_rows: list[dict[str, float]], figure_dir: Path) -> Path | None:
    if not alpha_rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(alpha_rows).sort_values("alpha")
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
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    for col in ("neural_granger_x_to_z", "neural_granger_y_to_z", "neural_granger_w_to_z"):
        if col not in frame:
            frame[col] = 0.0

    ax.plot(
        frame["alpha"],
        frame["neural_granger_x_to_z"],
        marker="o",
        color="#4f7ca8",
        linewidth=1.8,
        label="Neural Granger x->z",
    )
    ax.plot(
        frame["alpha"],
        frame["neural_granger_y_to_z"],
        marker="s",
        color="#c48f65",
        linewidth=1.8,
        label="Neural Granger y->z",
    )
    ax.plot(
        frame["alpha"],
        frame["neural_granger_w_to_z"],
        marker="^",
        color="#8c8c8c",
        linewidth=1.4,
        label="Neural Granger w->z",
    )
    ax.set_xlabel("alpha in alpha * sin(x y)")
    ax.set_ylabel("first-layer group norm")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    path = figure_dir / "sine_alpha_neural_granger_sweep.png"
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return path


def adjust_fdr_bh(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg correction preserving input order."""

    values = np.asarray(list(p_values), dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=float)
    values = np.where(np.isfinite(values), np.clip(values, 0.0, 1.0), 1.0)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def run_pcmci_cmiknn_readout(
    series: pd.DataFrame,
    *,
    variable_names: Sequence[str] = VARIABLE_NAMES,
    tau_max: int = 1,
    pc_alpha: float = 0.05,
    q_threshold: float = 0.05,
    knn: float = 0.10,
    sig_samples: int = 30,
    workers: int = -1,
) -> dict[str, object]:
    """Run nonlinear PCMCI-CMIknn and return cross-variable lagged pair scores."""

    from tigramite import data_processing as pp
    from tigramite.independence_tests.cmiknn import CMIknn
    from tigramite.pcmci import PCMCI

    names = tuple(variable_names)
    test = CMIknn(
        knn=float(knn),
        significance="shuffle_test",
        sig_samples=int(sig_samples),
        workers=int(workers),
    )
    pcmci = PCMCI(
        dataframe=pp.DataFrame(series[list(names)].to_numpy(dtype=float), var_names=list(names)),
        cond_ind_test=test,
        verbosity=0,
    )
    result = pcmci.run_pcmci(
        tau_min=1,
        tau_max=int(tau_max),
        pc_alpha=float(pc_alpha),
        alpha_level=float(q_threshold),
    )
    p_matrix = np.asarray(result["p_matrix"], dtype=float)
    val_matrix = np.asarray(result["val_matrix"], dtype=float)
    rows: list[dict[str, float | int | str | bool]] = []
    for source_idx, source in enumerate(names):
        for target_idx, target in enumerate(names):
            if source == target:
                continue
            for lag in range(1, int(tau_max) + 1):
                rows.append(
                    {
                        "method": "PCMCI-CMIknn",
                        "source": str(source),
                        "target": str(target),
                        "lag": int(lag),
                        "score": float(abs(val_matrix[source_idx, target_idx, lag])),
                        "signed_score": float(val_matrix[source_idx, target_idx, lag]),
                        "p_value": float(p_matrix[source_idx, target_idx, lag]),
                    }
                )
    q_values = adjust_fdr_bh([float(row["p_value"]) for row in rows])
    for row, q_value in zip(rows, q_values):
        row["q_value"] = float(q_value)
        row["selected"] = bool(float(q_value) <= float(q_threshold))
    rows.sort(key=lambda row: (float(row["q_value"]), -float(row["score"])))
    return {
        "method": "PCMCI-CMIknn",
        "config": {
            "tau_max": int(tau_max),
            "pc_alpha": float(pc_alpha),
            "q_threshold": float(q_threshold),
            "knn": float(knn),
            "sig_samples": int(sig_samples),
        },
        "rows": rows,
    }


def _observational_wms(
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    *,
    bins: int = 4,
) -> dict[str, float]:
    """Compute signed whole-minus-sum information on aligned observational samples."""

    left_array = _discretize_vector(left, int(bins)).reshape(-1, 1)
    right_array = _discretize_vector(right, int(bins)).reshape(-1, 1)
    target_array = _discretize_vector(target, int(bins)).reshape(-1, 1)
    if not (len(left_array) == len(right_array) == len(target_array)):
        raise ValueError("left, right, and target must have matching sample counts.")

    x_mi = _mutual_information_from_states(left_array, target_array)
    y_mi = _mutual_information_from_states(right_array, target_array)
    joint_mi = _mutual_information_from_states(np.column_stack([left_array, right_array]), target_array)
    return {
        "x_mi": x_mi,
        "y_mi": y_mi,
        "joint_mi": joint_mi,
        "wms": float(joint_mi - x_mi - y_mi),
    }


def run_sine_beta_common_driver_sweep(
    *,
    beta_values: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    seeds: Sequence[int] = (0, 1, 2, 3),
    n_samples: int = 1100,
    alpha: float = 1.0,
    noise: float = 0.05,
    mlp_epochs: int = 90,
    intervention_samples: int = 640,
    bins: int = 4,
    neural_granger_epochs: int = 120,
    pcmci_cmiknn_sig_samples: int = 30,
    pcmci_cmiknn_knn: float = 0.10,
) -> dict[str, object]:
    from yrd.transport_map import summarize_two_source_synergy_transport_map
    from scripts.reproduce_surd_synergistic_collider import decompose_surd_2source_transport_map

    rows: list[dict[str, float]] = []
    for beta in beta_values:
        for seed in seeds:
            config = SimConfig(
                mechanism="common_driver_sine_synergy",
                n_samples=int(n_samples),
                noise=float(noise),
                seed=int(seed),
                synergy_strength=float(alpha),
                common_driver_strength=float(beta),
                mlp_epochs=int(mlp_epochs),
                intervention_samples=int(intervention_samples),
                bins=int(bins),
            )
            series, _ = simulate_system(config)
            pcmci_cmiknn = run_pcmci_cmiknn_readout(
                series,
                variable_names=config.variable_names,
                tau_max=config.lag,
                pc_alpha=0.05,
                q_threshold=0.05,
                knn=float(pcmci_cmiknn_knn),
                sig_samples=int(pcmci_cmiknn_sig_samples),
                workers=-1,
            )
            features, targets = make_lagged_dataset(series, lag=config.lag)
            model = train_mlp_transition_model(features, targets, config)
            peid = estimate_peid_graph(model, series, config)
            neural_granger = run_neural_granger_readout(
                features,
                targets,
                variable_names=config.variable_names,
                max_lag=config.lag,
                epochs=int(neural_granger_epochs),
                seed=int(seed) + 9301 + int(round(float(beta) * 1000)),
            )
            shap_readout = estimate_shap_readout(
                model,
                features,
                series,
                config,
                foreground_samples=64,
                background_samples=64,
            )
            intervention_samples_frame = _sample_intervention_sources(series, config)
            intervention_predictions = model.predict(
                _intervention_features(intervention_samples_frame, config)
            )
            tm_peid_xy_z = summarize_two_source_synergy_transport_map(
                intervention_samples_frame[["x"]].to_numpy(dtype=float),
                intervention_samples_frame[["y"]].to_numpy(dtype=float),
                intervention_predictions[:, [config.variable_names.index("z")]],
            )
            oracle_z = (
                0.22 * intervention_samples_frame["z"].to_numpy(dtype=float)
                + float(alpha)
                * np.sin(
                    intervention_samples_frame["x"].to_numpy(dtype=float)
                    * intervention_samples_frame["y"].to_numpy(dtype=float)
                )
            ).reshape(-1, 1)
            oracle_peid_xy_z = summarize_two_source_synergy_transport_map(
                intervention_samples_frame[["x"]].to_numpy(dtype=float),
                intervention_samples_frame[["y"]].to_numpy(dtype=float),
                oracle_z,
            )
            surd_xy_z = decompose_surd_2source_transport_map(
                series["x"].to_numpy(dtype=float)[:-1],
                series["y"].to_numpy(dtype=float)[:-1],
                series["z"].to_numpy(dtype=float)[1:],
                degree=3,
                target_anchors=128,
                conditional_samples=64,
                seed=int(seed),
            )
            observational_wms = _observational_wms(
                series["x"].to_numpy(dtype=float)[:-1],
                series["y"].to_numpy(dtype=float)[:-1],
                series["z"].to_numpy(dtype=float)[1:],
                bins=int(config.bins),
            )
            peid_xy_z = peid.synergy_edges[
                (peid.synergy_edges["sources"] == "x+y")
                & (peid.synergy_edges["target"] == "z")
            ].iloc[0]
            shap_xy_z = shap_readout.shap_interaction_terms[
                (shap_readout.shap_interaction_terms["sources"] == "x+y")
                & (shap_readout.shap_interaction_terms["target"] == "z")
            ].iloc[0]
            product_xy_z = shap_readout.interaction_terms[
                (shap_readout.interaction_terms["sources"] == "x+y")
                & (shap_readout.interaction_terms["target"] == "z")
            ].iloc[0]
            shap_single_lookup = {
                str(row["source"]): float(row["mean_abs_phi"])
                for row in shap_readout.feature_attributions[
                    shap_readout.feature_attributions["target"] == "z"
                ].to_dict("records")
            }
            neural_granger_lookup = {
                str(row["source"]): float(row["group_norm"])
                for row in list(neural_granger["rows"])
                if str(row["target"]) == "z"
            }
            pcmci_lookup = {
                str(row["source"]): float(row["score"])
                for row in list(pcmci_cmiknn["rows"])
                if str(row["target"]) == "z" and int(row["lag"]) == 1
            }
            pcmci_q_lookup = {
                str(row["source"]): float(row["q_value"])
                for row in list(pcmci_cmiknn["rows"])
                if str(row["target"]) == "z" and int(row["lag"]) == 1
            }
            rows.append(
                {
                    "run_id": f"beta={float(beta):.2f}|seed={int(seed)}",
                    "beta": float(beta),
                    "seed": float(seed),
                    "xy_observed_corr": float(series[["x", "y"]].corr().iloc[0, 1]),
                    "observational_x_to_z_mi": float(observational_wms["x_mi"]),
                    "observational_y_to_z_mi": float(observational_wms["y_mi"]),
                    "observational_xy_to_z_joint_mi": float(observational_wms["joint_mi"]),
                    "observational_wms": float(observational_wms["wms"]),
                    "final_train_loss": float(model.loss_history[-1]) if model.loss_history else float("nan"),
                    "shap_x_to_z_mean_abs": float(shap_single_lookup.get("x", 0.0)),
                    "shap_y_to_z_mean_abs": float(shap_single_lookup.get("y", 0.0)),
                    "shap_xy_mean_abs_interaction": float(shap_xy_z["mean_abs_interaction"]),
                    "shap_xy_mean_interaction": float(shap_xy_z["mean_interaction"]),
                    "product_xy_incremental_r2": float(product_xy_z["incremental_r2"]),
                    "neural_granger_x_to_z": float(neural_granger_lookup.get("x", 0.0)),
                    "neural_granger_y_to_z": float(neural_granger_lookup.get("y", 0.0)),
                    "neural_granger_w_to_z": float(neural_granger_lookup.get("w", 0.0)),
                    "neural_granger_xy_to_z": float(
                        neural_granger_lookup.get("x", 0.0) + neural_granger_lookup.get("y", 0.0)
                    ),
                    "pcmci_cmiknn_x_to_z": float(pcmci_lookup.get("x", 0.0)),
                    "pcmci_cmiknn_y_to_z": float(pcmci_lookup.get("y", 0.0)),
                    "pcmci_cmiknn_w_to_z": float(pcmci_lookup.get("w", 0.0)),
                    "pcmci_cmiknn_xy_to_z": float(
                        pcmci_lookup.get("x", 0.0) + pcmci_lookup.get("y", 0.0)
                    ),
                    "pcmci_cmiknn_x_to_z_q": float(pcmci_q_lookup.get("x", 1.0)),
                    "pcmci_cmiknn_y_to_z_q": float(pcmci_q_lookup.get("y", 1.0)),
                    "pcmci_cmiknn_w_to_z_q": float(pcmci_q_lookup.get("w", 1.0)),
                    "peid_xy_joint_ei": float(peid_xy_z["joint_ei"]),
                    "peid_xy_synergy": float(peid_xy_z["synergy"]),
                    "tm_peid_xy_joint_ei": float(tm_peid_xy_z["joint_ei"]),
                    "tm_peid_xy_left_ei": float(tm_peid_xy_z["left_ei"]),
                    "tm_peid_xy_right_ei": float(tm_peid_xy_z["right_ei"]),
                    "tm_peid_xy_synergy": float(tm_peid_xy_z["syn"]),
                    "surd_redundancy": float(surd_xy_z["redundancy"]),
                    "surd_unique_x": float(surd_xy_z["unique_x"]),
                    "surd_unique_y": float(surd_xy_z["unique_y"]),
                    "surd_xy_synergy": float(surd_xy_z["synergy"]),
                    "surd_xy_joint": float(surd_xy_z["joint_ei"]),
                    "mlp_peid_redundancy": 0.0,
                    "mlp_peid_unique_x": float(tm_peid_xy_z["left_ei"]),
                    "mlp_peid_unique_y": float(tm_peid_xy_z["right_ei"]),
                    "mlp_peid_xy_synergy": float(tm_peid_xy_z["syn"]),
                    "mlp_peid_xy_joint": float(tm_peid_xy_z["joint_ei"]),
                    "oracle_peid_redundancy": 0.0,
                    "oracle_peid_unique_x": float(oracle_peid_xy_z["left_ei"]),
                    "oracle_peid_unique_y": float(oracle_peid_xy_z["right_ei"]),
                    "oracle_peid_xy_synergy": float(oracle_peid_xy_z["syn"]),
                    "oracle_peid_xy_joint": float(oracle_peid_xy_z["joint_ei"]),
                    "peid_x_to_z": float(
                        peid.pairwise_edges[
                            (peid.pairwise_edges["source"] == "x")
                            & (peid.pairwise_edges["target"] == "z")
                        ].iloc[0]["ei"]
                    ),
                    "peid_y_to_z": float(
                        peid.pairwise_edges[
                            (peid.pairwise_edges["source"] == "y")
                            & (peid.pairwise_edges["target"] == "z")
                        ].iloc[0]["ei"]
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("beta", as_index=False)
        .agg(
            xy_observed_corr_mean=("xy_observed_corr", "mean"),
            xy_observed_corr_std=("xy_observed_corr", "std"),
            observational_x_to_z_mi_mean=("observational_x_to_z_mi", "mean"),
            observational_x_to_z_mi_std=("observational_x_to_z_mi", "std"),
            observational_y_to_z_mi_mean=("observational_y_to_z_mi", "mean"),
            observational_y_to_z_mi_std=("observational_y_to_z_mi", "std"),
            observational_xy_to_z_joint_mi_mean=("observational_xy_to_z_joint_mi", "mean"),
            observational_xy_to_z_joint_mi_std=("observational_xy_to_z_joint_mi", "std"),
            observational_wms_mean=("observational_wms", "mean"),
            observational_wms_std=("observational_wms", "std"),
            shap_xy_mean_abs_interaction_mean=("shap_xy_mean_abs_interaction", "mean"),
            shap_xy_mean_abs_interaction_std=("shap_xy_mean_abs_interaction", "std"),
            shap_x_to_z_mean_abs_mean=("shap_x_to_z_mean_abs", "mean"),
            shap_x_to_z_mean_abs_std=("shap_x_to_z_mean_abs", "std"),
            shap_y_to_z_mean_abs_mean=("shap_y_to_z_mean_abs", "mean"),
            shap_y_to_z_mean_abs_std=("shap_y_to_z_mean_abs", "std"),
            product_xy_incremental_r2_mean=("product_xy_incremental_r2", "mean"),
            product_xy_incremental_r2_std=("product_xy_incremental_r2", "std"),
            neural_granger_x_to_z_mean=("neural_granger_x_to_z", "mean"),
            neural_granger_x_to_z_std=("neural_granger_x_to_z", "std"),
            neural_granger_y_to_z_mean=("neural_granger_y_to_z", "mean"),
            neural_granger_y_to_z_std=("neural_granger_y_to_z", "std"),
            neural_granger_w_to_z_mean=("neural_granger_w_to_z", "mean"),
            neural_granger_w_to_z_std=("neural_granger_w_to_z", "std"),
            neural_granger_xy_to_z_mean=("neural_granger_xy_to_z", "mean"),
            neural_granger_xy_to_z_std=("neural_granger_xy_to_z", "std"),
            pcmci_cmiknn_x_to_z_mean=("pcmci_cmiknn_x_to_z", "mean"),
            pcmci_cmiknn_x_to_z_std=("pcmci_cmiknn_x_to_z", "std"),
            pcmci_cmiknn_y_to_z_mean=("pcmci_cmiknn_y_to_z", "mean"),
            pcmci_cmiknn_y_to_z_std=("pcmci_cmiknn_y_to_z", "std"),
            pcmci_cmiknn_w_to_z_mean=("pcmci_cmiknn_w_to_z", "mean"),
            pcmci_cmiknn_w_to_z_std=("pcmci_cmiknn_w_to_z", "std"),
            pcmci_cmiknn_xy_to_z_mean=("pcmci_cmiknn_xy_to_z", "mean"),
            pcmci_cmiknn_xy_to_z_std=("pcmci_cmiknn_xy_to_z", "std"),
            pcmci_cmiknn_x_to_z_q_mean=("pcmci_cmiknn_x_to_z_q", "mean"),
            pcmci_cmiknn_y_to_z_q_mean=("pcmci_cmiknn_y_to_z_q", "mean"),
            pcmci_cmiknn_w_to_z_q_mean=("pcmci_cmiknn_w_to_z_q", "mean"),
            peid_xy_joint_ei_mean=("peid_xy_joint_ei", "mean"),
            peid_xy_joint_ei_std=("peid_xy_joint_ei", "std"),
            peid_xy_synergy_mean=("peid_xy_synergy", "mean"),
            peid_xy_synergy_std=("peid_xy_synergy", "std"),
            tm_peid_xy_joint_ei_mean=("tm_peid_xy_joint_ei", "mean"),
            tm_peid_xy_joint_ei_std=("tm_peid_xy_joint_ei", "std"),
            tm_peid_xy_synergy_mean=("tm_peid_xy_synergy", "mean"),
            tm_peid_xy_synergy_std=("tm_peid_xy_synergy", "std"),
            tm_peid_xy_left_ei_mean=("tm_peid_xy_left_ei", "mean"),
            tm_peid_xy_left_ei_std=("tm_peid_xy_left_ei", "std"),
            tm_peid_xy_right_ei_mean=("tm_peid_xy_right_ei", "mean"),
            tm_peid_xy_right_ei_std=("tm_peid_xy_right_ei", "std"),
            surd_redundancy_mean=("surd_redundancy", "mean"),
            surd_redundancy_std=("surd_redundancy", "std"),
            surd_unique_x_mean=("surd_unique_x", "mean"),
            surd_unique_x_std=("surd_unique_x", "std"),
            surd_unique_y_mean=("surd_unique_y", "mean"),
            surd_unique_y_std=("surd_unique_y", "std"),
            surd_xy_synergy_mean=("surd_xy_synergy", "mean"),
            surd_xy_synergy_std=("surd_xy_synergy", "std"),
            surd_xy_joint_mean=("surd_xy_joint", "mean"),
            surd_xy_joint_std=("surd_xy_joint", "std"),
            mlp_peid_redundancy_mean=("mlp_peid_redundancy", "mean"),
            mlp_peid_redundancy_std=("mlp_peid_redundancy", "std"),
            mlp_peid_unique_x_mean=("mlp_peid_unique_x", "mean"),
            mlp_peid_unique_x_std=("mlp_peid_unique_x", "std"),
            mlp_peid_unique_y_mean=("mlp_peid_unique_y", "mean"),
            mlp_peid_unique_y_std=("mlp_peid_unique_y", "std"),
            mlp_peid_xy_synergy_mean=("mlp_peid_xy_synergy", "mean"),
            mlp_peid_xy_synergy_std=("mlp_peid_xy_synergy", "std"),
            mlp_peid_xy_joint_mean=("mlp_peid_xy_joint", "mean"),
            mlp_peid_xy_joint_std=("mlp_peid_xy_joint", "std"),
            oracle_peid_redundancy_mean=("oracle_peid_redundancy", "mean"),
            oracle_peid_redundancy_std=("oracle_peid_redundancy", "std"),
            oracle_peid_unique_x_mean=("oracle_peid_unique_x", "mean"),
            oracle_peid_unique_x_std=("oracle_peid_unique_x", "std"),
            oracle_peid_unique_y_mean=("oracle_peid_unique_y", "mean"),
            oracle_peid_unique_y_std=("oracle_peid_unique_y", "std"),
            oracle_peid_xy_synergy_mean=("oracle_peid_xy_synergy", "mean"),
            oracle_peid_xy_synergy_std=("oracle_peid_xy_synergy", "std"),
            oracle_peid_xy_joint_mean=("oracle_peid_xy_joint", "mean"),
            oracle_peid_xy_joint_std=("oracle_peid_xy_joint", "std"),
            peid_x_to_z_mean=("peid_x_to_z", "mean"),
            peid_y_to_z_mean=("peid_y_to_z", "mean"),
        )
        .sort_values("beta")
        .reset_index(drop=True)
    )
    trend = _beta_sweep_trend_stats(frame)
    return {
        "config": {
            "beta_values": [float(value) for value in beta_values],
            "seeds": [int(value) for value in seeds],
            "n_samples": int(n_samples),
            "alpha": float(alpha),
            "noise": float(noise),
            "mlp_epochs": int(mlp_epochs),
            "intervention_samples": int(intervention_samples),
            "neural_granger_epochs": int(neural_granger_epochs),
            "pcmci_cmiknn_sig_samples": int(pcmci_cmiknn_sig_samples),
            "pcmci_cmiknn_knn": float(pcmci_cmiknn_knn),
        },
        "units": {
            "observational_wms": "bits",
            "shap": "mean absolute SHAP readout",
            "neural_granger": "first-layer source-group norm",
            "pcmci_cmiknn": "absolute CMIknn dependence statistic",
            "surd": "bits",
            "mlp_peid": "bits",
            "oracle_peid": "bits",
        },
        "runs": rows,
        "summary": summary.to_dict("records"),
        "trend": trend,
    }


def _linear_slope(values: pd.DataFrame, y_col: str) -> float:
    finite = values[["beta", y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if finite["beta"].nunique() < 2:
        return float("nan")
    slope, _intercept = np.polyfit(finite["beta"].to_numpy(dtype=float), finite[y_col].to_numpy(dtype=float), deg=1)
    return float(slope)


def _bootstrap_slope_ci(
    values: pd.DataFrame,
    y_col: str,
    *,
    seed: int,
    n_boot: int = 1000,
) -> tuple[float, float]:
    finite = values[["beta", y_col]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(finite) < 4 or finite["beta"].nunique() < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    slopes = np.empty(int(n_boot), dtype=float)
    for idx in range(int(n_boot)):
        sample_idx = rng.integers(0, len(finite), size=len(finite))
        sample = finite.iloc[sample_idx]
        slopes[idx] = _linear_slope(sample, y_col)
    lo, hi = np.nanpercentile(slopes, [2.5, 97.5])
    return (float(lo), float(hi))


def _beta_sweep_trend_stats(frame: pd.DataFrame) -> dict[str, float]:
    observational_wms_slope = _linear_slope(frame, "observational_wms")
    shap_slope = _linear_slope(frame, "shap_xy_mean_abs_interaction")
    neural_granger_xy_slope = _linear_slope(frame, "neural_granger_xy_to_z")
    pcmci_cmiknn_xy_slope = _linear_slope(frame, "pcmci_cmiknn_xy_to_z")
    peid_slope = _linear_slope(frame, "peid_xy_synergy")
    tm_peid_slope = _linear_slope(frame, "tm_peid_xy_synergy")
    surd_slope = _linear_slope(frame, "surd_xy_synergy")
    oracle_slope = _linear_slope(frame, "oracle_peid_xy_synergy")
    product_slope = _linear_slope(frame, "product_xy_incremental_r2")
    corr_slope = _linear_slope(frame, "xy_observed_corr")
    observational_wms_ci = _bootstrap_slope_ci(frame, "observational_wms", seed=17009)
    shap_ci = _bootstrap_slope_ci(frame, "shap_xy_mean_abs_interaction", seed=17001)
    neural_granger_xy_ci = _bootstrap_slope_ci(frame, "neural_granger_xy_to_z", seed=17007)
    pcmci_cmiknn_xy_ci = _bootstrap_slope_ci(frame, "pcmci_cmiknn_xy_to_z", seed=17008)
    peid_ci = _bootstrap_slope_ci(frame, "peid_xy_synergy", seed=17002)
    tm_peid_ci = _bootstrap_slope_ci(frame, "tm_peid_xy_synergy", seed=17004)
    surd_ci = _bootstrap_slope_ci(frame, "surd_xy_synergy", seed=17005)
    oracle_ci = _bootstrap_slope_ci(frame, "oracle_peid_xy_synergy", seed=17006)
    product_ci = _bootstrap_slope_ci(frame, "product_xy_incremental_r2", seed=17003)
    return {
        "xy_observed_corr_slope": corr_slope,
        "observational_wms_slope": observational_wms_slope,
        "observational_wms_slope_ci_low": observational_wms_ci[0],
        "observational_wms_slope_ci_high": observational_wms_ci[1],
        "shap_interaction_slope": shap_slope,
        "shap_interaction_slope_ci_low": shap_ci[0],
        "shap_interaction_slope_ci_high": shap_ci[1],
        "neural_granger_xy_to_z_slope": neural_granger_xy_slope,
        "neural_granger_xy_to_z_slope_ci_low": neural_granger_xy_ci[0],
        "neural_granger_xy_to_z_slope_ci_high": neural_granger_xy_ci[1],
        "pcmci_cmiknn_xy_to_z_slope": pcmci_cmiknn_xy_slope,
        "pcmci_cmiknn_xy_to_z_slope_ci_low": pcmci_cmiknn_xy_ci[0],
        "pcmci_cmiknn_xy_to_z_slope_ci_high": pcmci_cmiknn_xy_ci[1],
        "product_probe_r2_slope": product_slope,
        "product_probe_r2_slope_ci_low": product_ci[0],
        "product_probe_r2_slope_ci_high": product_ci[1],
        "peid_synergy_slope": peid_slope,
        "peid_synergy_slope_ci_low": peid_ci[0],
        "peid_synergy_slope_ci_high": peid_ci[1],
        "tm_peid_synergy_slope": tm_peid_slope,
        "tm_peid_synergy_slope_ci_low": tm_peid_ci[0],
        "tm_peid_synergy_slope_ci_high": tm_peid_ci[1],
        "surd_synergy_slope": surd_slope,
        "surd_synergy_slope_ci_low": surd_ci[0],
        "surd_synergy_slope_ci_high": surd_ci[1],
        "oracle_peid_synergy_slope": oracle_slope,
        "oracle_peid_synergy_slope_ci_low": oracle_ci[0],
        "oracle_peid_synergy_slope_ci_high": oracle_ci[1],
        "slope_difference_shap_minus_peid": float(shap_slope - peid_slope),
        "slope_difference_shap_minus_tm_peid": float(shap_slope - tm_peid_slope),
    }


def _plot_sine_beta_sweep(beta_result: dict[str, object], figure_dir: Path) -> Path | None:
    summary_rows = beta_result.get("summary", [])
    if not summary_rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(summary_rows).sort_values("beta")
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
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14.4, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 6)
    flat_axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[1, 4:6]),
    ]

    def line_with_band(ax, y_col: str, std_col: str, *, label: str, color: str, marker: str = "o") -> None:
        x = frame["beta"].to_numpy(dtype=float)
        y = frame[y_col].to_numpy(dtype=float)
        std = frame[std_col].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, marker=marker, color=color, linewidth=1.7, label=label)
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.16, linewidth=0.0)

    line_with_band(
        flat_axes[0],
        "observational_wms_mean",
        "observational_wms_std",
        label="observational WMS",
        color="#8c564b",
    )
    flat_axes[0].axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.9)
    flat_axes[0].set_ylabel("Information (bits)")
    flat_axes[0].set_title("Whole-minus-sum")

    for y_col, std_col, label, color, marker in [
        ("shap_x_to_z_mean_abs_mean", "shap_x_to_z_mean_abs_std", "SHAP x->z", "#4c78a8", "^"),
        ("shap_y_to_z_mean_abs_mean", "shap_y_to_z_mean_abs_std", "SHAP y->z", "#f58518", "v"),
        (
            "shap_xy_mean_abs_interaction_mean",
            "shap_xy_mean_abs_interaction_std",
            "SHAP interaction x:y->z",
            "#e45756",
            "o",
        ),
    ]:
        line_with_band(flat_axes[1], y_col, std_col, label=label, color=color, marker=marker)
    flat_axes[1].set_ylabel("Mean absolute SHAP readout")
    flat_axes[1].set_title("MLP+SHAP")

    for y_col, std_col, label, color, marker in [
        ("surd_redundancy_mean", "surd_redundancy_std", r"SURD $R_{xy}$", "#7f8c8d", "s"),
        ("surd_unique_x_mean", "surd_unique_x_std", r"SURD $U_x$", "#4c78a8", "^"),
        ("surd_unique_y_mean", "surd_unique_y_std", r"SURD $U_y$", "#f58518", "v"),
        ("surd_xy_synergy_mean", "surd_xy_synergy_std", r"SURD $S_{xy}$", "#e45756", "o"),
    ]:
        line_with_band(flat_axes[2], y_col, std_col, label=label, color=color, marker=marker)
    flat_axes[2].set_ylabel("Information (bits)")
    flat_axes[2].set_title("Observational SURD")

    for y_col, std_col, label, color, marker in [
        ("pcmci_cmiknn_x_to_z_mean", "pcmci_cmiknn_x_to_z_std", "PCMCI x->z", "#4c78a8", "^"),
        ("pcmci_cmiknn_y_to_z_mean", "pcmci_cmiknn_y_to_z_std", "PCMCI y->z", "#f58518", "v"),
        ("pcmci_cmiknn_w_to_z_mean", "pcmci_cmiknn_w_to_z_std", "PCMCI w->z", "#6b7280", "s"),
    ]:
        if y_col in frame and std_col in frame:
            line_with_band(flat_axes[3], y_col, std_col, label=label, color=color, marker=marker)
    flat_axes[3].set_ylabel("Absolute CMIknn statistic")
    flat_axes[3].set_title("PCMCI-CMIknn")

    for y_col, std_col, label, color, marker in [
        ("neural_granger_x_to_z_mean", "neural_granger_x_to_z_std", "NG x->z", "#4c78a8", "^"),
        ("neural_granger_y_to_z_mean", "neural_granger_y_to_z_std", "NG y->z", "#f58518", "v"),
    ]:
        line_with_band(flat_axes[4], y_col, std_col, label=label, color=color, marker=marker)
    flat_axes[4].set_ylabel("First-layer group norm")
    flat_axes[4].set_title("Neural Granger")

    for y_col, std_col, label, color, marker in [
        ("mlp_peid_unique_x_mean", "mlp_peid_unique_x_std", r"PEID $U_x$", "#4c78a8", "^"),
        ("mlp_peid_unique_y_mean", "mlp_peid_unique_y_std", r"PEID $U_y$", "#f58518", "v"),
        ("mlp_peid_xy_synergy_mean", "mlp_peid_xy_synergy_std", r"PEID $S_{xy}$", "#e45756", "o"),
    ]:
        line_with_band(flat_axes[5], y_col, std_col, label=label, color=color, marker=marker)
    flat_axes[5].set_ylabel("Information (bits)")
    flat_axes[5].set_title("MLP+PEID")

    for ax in flat_axes:
        ax.set_xlabel("beta: common-driver strength")
        ax.set_xticks(frame["beta"].to_numpy(dtype=float))
        ax.grid(alpha=0.18, linewidth=0.5)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    path = figure_dir / "sine_beta_unified_readout_sweep.png"
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_sine_beta_validation(beta_result: dict[str, object], figure_dir: Path) -> Path | None:
    summary_rows = beta_result.get("summary", [])
    if not summary_rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(summary_rows).sort_values("beta")
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)
    axes[0].plot(
        frame["beta"],
        frame["mlp_peid_xy_synergy_mean"],
        marker="o",
        color="#1b9e77",
        label="MLP+PEID synergy",
    )
    axes[0].plot(
        frame["beta"],
        frame["oracle_peid_xy_synergy_mean"],
        marker="s",
        color="#7570b3",
        label="Oracle+PEID synergy",
    )
    axes[0].set(xlabel="beta: common-driver strength", ylabel="Information (bits)")
    axes[0].set_title("MLP+PEID versus Oracle+PEID")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    q1_path = ROOT / "results" / "surd_original_synergistic_collider" / "summary_q1.json"
    if q1_path.exists():
        q1 = json.loads(q1_path.read_text(encoding="utf-8"))["q1"]["normalized_atoms"]
        names = list(q1)
        colors = ["#607d8b" if name.startswith("R") else "#e57373" if name.startswith("U") else "#fdb462" for name in names]
        axes[1].bar(np.arange(len(names)), [q1[name] for name in names], color=colors)
        axes[1].set_xticks(np.arange(len(names)), names, rotation=60, ha="right", fontsize=7)
        axes[1].set_ylabel("Normalized SURD atom")
        axes[1].set_title("SURD Q1 transport-map reproduction")
    else:
        axes[1].text(0.5, 0.5, "Q1 reproduction summary unavailable", ha="center", va="center")
        axes[1].axis("off")
    path = figure_dir / "sine_beta_method_validation.png"
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return path


def _proxy_y_readout_values(edge_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    edge_frame = pd.DataFrame(edge_rows)
    sine_runs = edge_frame.loc[
        edge_frame["mechanism"].eq("common_driver_sine_synergy"), "run_id"
    ].dropna()
    if sine_runs.empty:
        return []
    run_id = str(sine_runs.iloc[0])
    run_edges = edge_frame[edge_frame["run_id"] == run_id].copy()
    methods = [
        ("SHAP", "interventional_shap", "mean_abs_phi"),
        ("PEID EI", "peid_pairwise", "ei"),
        ("Granger", "granger_pairwise", "score"),
    ]
    sources = ["w", "x", "y", "z"]
    rows: list[dict[str, object]] = []
    for method_label, edge_type, score_col in methods:
        for source in sources:
            subset = run_edges[
                (run_edges["edge_type"] == edge_type)
                & (run_edges["source"] == source)
                & (run_edges["target"] == "y")
            ]
            value = float(subset.iloc[0][score_col]) if not subset.empty else 0.0
            rows.append(
                {
                    "method": method_label,
                    "source": source,
                    "target": "y",
                    "value": value,
                }
            )
    return rows


def _plot_proxy_y_readout_summary(edge_rows: list[dict[str, object]], figure_dir: Path) -> Path | None:
    rows = _proxy_y_readout_values(edge_rows)
    if not rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(rows)
    method_order = ["SHAP", "PEID EI", "Granger"]
    source_order = ["w", "x", "y", "z"]
    matrix = np.zeros((len(method_order), len(source_order)), dtype=float)
    for row_idx, method in enumerate(method_order):
        for col_idx, source in enumerate(source_order):
            subset = frame[(frame["method"] == method) & (frame["source"] == source)]
            matrix[row_idx, col_idx] = float(subset.iloc[0]["value"]) if not subset.empty else 0.0
    row_scaled = matrix.copy()
    for row_idx in range(row_scaled.shape[0]):
        row_max = np.nanmax(row_scaled[row_idx])
        if np.isfinite(row_max) and row_max > 0.0:
            row_scaled[row_idx] = row_scaled[row_idx] / row_max
    ratios = []
    for row_idx, method in enumerate(method_order):
        driver = matrix[row_idx, source_order.index("w")]
        proxy = matrix[row_idx, source_order.index("x")]
        ratios.append(proxy / (driver + 1e-12))

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
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7), constrained_layout=True, width_ratios=[1.45, 0.9])

    ax = axes[0]
    im = ax.imshow(row_scaled, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(source_order)))
    ax.set_xticklabels([f"{source}->y" for source in source_order], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(method_order)))
    ax.set_yticklabels(method_order)
    ax.set_title("Target y readouts on the same MLP", fontsize=9, fontweight="bold", pad=6)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            color = "white" if row_scaled[row_idx, col_idx] > 0.62 else "#202020"
            ax.text(col_idx, row_idx, f"{value:.3g}", ha="center", va="center", fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.025)
    cbar.set_label("Row-normalized intensity", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    ax = axes[1]
    y_pos = np.arange(len(method_order))
    colors = ["#4f7ca8", "#5f8f6b", "#8c8c8c"]
    ax.barh(y_pos, ratios, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_order)
    ax.invert_yaxis()
    ax.set_xlabel("proxy / driver ratio")
    ax.set_title("x->y relative to w->y", fontsize=9, fontweight="bold", pad=6)
    xmax = max(max(ratios), 0.05)
    ax.set_xlim(0.0, xmax * 1.25)
    for idx, value in enumerate(ratios):
        ax.text(value + xmax * 0.025, idx, f"{value:.3g}", va="center", ha="left", fontsize=7)

    path = figure_dir / "proxy_y_shap_peid_readout.png"
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw_lagged_proxy_panel(
    ax,
    *,
    title: str,
    edges: Sequence[dict[str, object]],
    note: str = "",
) -> None:
    from matplotlib.patches import Circle, FancyArrowPatch

    positions = {
        "w": (0.18, 0.70),
        "x": (0.50, 0.70),
        "y": (0.82, 0.70),
    }
    labels = {
        "w": r"$w_t$",
        "x": r"$x_{t+1}$",
        "y": r"$y_{t+2}$",
    }
    ax.set_title(title, fontsize=8.2, pad=5, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        if source not in positions or target not in positions:
            continue
        start = positions[source]
        end = positions[target]
        color = str(edge.get("color", "#4f7ca8"))
        linewidth = float(edge.get("linewidth", 1.4))
        linestyle = str(edge.get("linestyle", "-"))
        rad = float(edge.get("rad", 0.0))
        label = str(edge.get("label", ""))
        label_offset = tuple(edge.get("label_offset", (0.0, 0.08)))
        label_pos = edge.get("label_pos")
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
            shrinkA=18,
            shrinkB=18,
            connectionstyle=f"arc3,rad={rad}",
            alpha=float(edge.get("alpha", 1.0)),
        )
        ax.add_patch(arrow)
        if label:
            if label_pos is None:
                mid_x = (start[0] + end[0]) / 2.0 + float(label_offset[0])
                mid_y = (start[1] + end[1]) / 2.0 + float(label_offset[1])
            else:
                mid_x = float(label_pos[0])
                mid_y = float(label_pos[1])
            ax.text(
                mid_x,
                mid_y,
                label,
                ha="center",
                va="center",
                fontsize=6.5,
                color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.35, "alpha": 0.92},
            )

    for name, (x_pos, y_pos) in positions.items():
        node = Circle((x_pos, y_pos), 0.065, facecolor="white", edgecolor="#222222", linewidth=1.0, zorder=3)
        ax.add_patch(node)
        ax.text(x_pos, y_pos, labels[name], ha="center", va="center", fontsize=7.6, fontweight="bold", zorder=4)
    if note:
        ax.text(0.5, 0.25, note, ha="center", va="top", fontsize=6.7, color="#333333", wrap=True)


def _lagged_proxy_weight_edges(
    weights: dict[str, float],
    *,
    color: str,
    weak_color: str,
    weak_threshold: float,
) -> list[dict[str, object]]:
    style = {
        ("w", "x"): {"rad": -0.18, "label_pos": (0.34, 0.83)},
        ("x", "w"): {"rad": 0.34, "label_pos": (0.34, 0.55)},
        ("x", "y"): {"rad": -0.18, "label_pos": (0.66, 0.83)},
        ("y", "x"): {"rad": 0.34, "label_pos": (0.66, 0.55)},
        ("w", "y"): {"rad": -0.30, "label_pos": (0.50, 0.91)},
        ("y", "w"): {"rad": 0.30, "label_pos": (0.50, 0.45)},
    }
    edges: list[dict[str, object]] = []
    max_weight = max([abs(value) for value in weights.values()] + [1e-12])
    for edge, value in sorted(weights.items()):
        source, target = edge.split("->")
        edge_style = style.get((source, target), {})
        is_weak = abs(value) < weak_threshold
        edges.append(
            {
                "source": source,
                "target": target,
                "label": f"{value:.3g}",
                "color": weak_color if is_weak else color,
                "linestyle": "--" if is_weak else "-",
                "linewidth": 0.75 + 1.65 * min(1.0, abs(value) / max_weight),
                "alpha": 0.72 if is_weak else 0.96,
                **edge_style,
            }
        )
    return edges


def _plot_lagged_proxy_causal_graph(result: dict[str, object], figure_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 2.65), constrained_layout=True)
    true_color = "#4f7ca8"
    false_color = "#c95f43"
    peid_color = "#5f9a64"
    muted = "#9a9a9a"
    granger_edges = dict(result["pairwise_granger_edges"])
    peid_edges = dict(result["peid_ei_edges"])

    _draw_lagged_proxy_panel(
        axes[0],
        title="Ground truth",
        edges=[
            {"source": "w", "target": "x", "label": "0.9", "color": true_color, "rad": 0.0, "label_pos": (0.34, 0.80)},
            {"source": "w", "target": "y", "label": "0.7", "color": true_color, "rad": -0.30, "label_pos": (0.50, 0.91)},
        ],
        note="No direct proxy-target structural term.",
    )
    _draw_lagged_proxy_panel(
        axes[1],
        title="Pairwise Granger score",
        edges=_lagged_proxy_weight_edges(
            granger_edges,
            color=false_color,
            weak_color=muted,
            weak_threshold=1e-3,
        ),
        note="Each edge is a one-source predictive gain.",
    )
    _draw_lagged_proxy_panel(
        axes[2],
        title=r"Condition on $w_t$",
        edges=[
            {
                "source": "x",
                "target": "y",
                "label": f"{result['causal_state_coef_x_proxy']:.3f}",
                "color": muted,
                "linestyle": "--",
                "linewidth": 1.2,
                "label_pos": (0.66, 0.80),
            },
            {
                "source": "w",
                "target": "y",
                "label": f"{result['causal_state_coef_w_driver']:.3f}",
                "color": true_color,
                "linewidth": 2.0,
                "rad": -0.30,
                "label_pos": (0.50, 0.91),
            },
        ],
        note=f"x increment after w: {result['conditional_x_incremental_score_given_w']:.1e}.",
    )
    _draw_lagged_proxy_panel(
        axes[3],
        title="PEID / EI",
        edges=_lagged_proxy_weight_edges(
            peid_edges,
            color=peid_color,
            weak_color=muted,
            weak_threshold=0.05,
        ),
        note=f"proxy / driver EI: {result['peid_proxy_to_driver_ratio']:.4f}.",
    )
    fig.suptitle("Lagged common-driver counterexample: all directed pairwise weights", fontsize=9.8)
    path = figure_dir / "lagged_proxy_causal_graph.png"
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return path


def _node_positions() -> dict[str, tuple[float, float]]:
    return {
        "x": (0.18, 0.72),
        "y": (0.18, 0.28),
        "z": (0.78, 0.50),
        "w": (0.50, 0.88),
    }


def _draw_graph_panel(
    ax,
    *,
    title: str,
    pairwise_edges: Sequence[tuple[str, str]],
    hyperedges: Sequence[tuple[str, str, str]] = (),
) -> None:
    from matplotlib.patches import Circle, FancyArrowPatch

    positions = _node_positions()
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    for source, target in pairwise_edges:
        if source not in positions or target not in positions:
            continue
        start = positions[source]
        end = positions[target]
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.2,
            color="#4f7ca8",
            shrinkA=14,
            shrinkB=14,
            connectionstyle="arc3,rad=0.05",
        )
        ax.add_patch(arrow)

    for source_a, source_b, target in hyperedges:
        if source_a not in positions or source_b not in positions or target not in positions:
            continue
        a = np.asarray(positions[source_a], dtype=float)
        b = np.asarray(positions[source_b], dtype=float)
        t = np.asarray(positions[target], dtype=float)
        center = (a + b + t) / 3.0
        ax.plot([a[0], center[0], b[0]], [a[1], center[1], b[1]], color="#6aa36f", linewidth=1.6)
        arrow = FancyArrowPatch(
            tuple(center),
            tuple(t),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.6,
            color="#6aa36f",
            shrinkA=6,
            shrinkB=14,
        )
        ax.add_patch(arrow)
        ax.text(center[0], center[1] - 0.06, "synergy", ha="center", va="top", fontsize=6.5, color="#3f7448")

    for name, (x_pos, y_pos) in positions.items():
        node = Circle((x_pos, y_pos), 0.065, facecolor="white", edgecolor="#222222", linewidth=1.0, zorder=3)
        ax.add_patch(node)
        ax.text(x_pos, y_pos, name, ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)


def _top_edges_for_run(edge_frame: pd.DataFrame, *, run_id: str, edge_type: str, score_col: str, k: int) -> list[tuple[str, str]]:
    if k <= 0:
        return []
    subset = edge_frame[(edge_frame["run_id"] == run_id) & (edge_frame["edge_type"] == edge_type)].copy()
    if subset.empty or score_col not in subset:
        return []
    subset = subset[subset["source"] != subset["target"]]
    subset = subset.sort_values(score_col, ascending=False).head(k)
    return [(str(row["source"]), str(row["target"])) for _, row in subset.iterrows()]


def _top_hyperedges_for_run(edge_frame: pd.DataFrame, *, run_id: str, k: int) -> list[tuple[str, str, str]]:
    if k <= 0:
        return []
    subset = edge_frame[(edge_frame["run_id"] == run_id) & (edge_frame["edge_type"] == "peid_synergy")].copy()
    if subset.empty:
        return []
    subset = subset.sort_values("synergy", ascending=False).head(k)
    rows: list[tuple[str, str, str]] = []
    for _, row in subset.iterrows():
        sources = str(row["sources"]).split("+")
        if len(sources) == 2:
            rows.append((sources[0], sources[1], str(row["target"])))
    return rows


def _plot_representative_causal_graphs(
    runs: list[dict[str, object]],
    edge_rows: list[dict[str, object]],
    figure_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    run_frame = pd.DataFrame(runs)
    edge_frame = pd.DataFrame(edge_rows)
    mechanisms = sorted(run_frame["mechanism"].unique())
    representative_rows = (
        run_frame.sort_values(["mechanism", "noise", "seed", "n_samples", "synergy_strength"])
        .groupby("mechanism", as_index=False)
        .head(1)
        .to_dict("records")
    )
    n_cols = max(1, len(representative_rows))
    fig, axes = plt.subplots(2, n_cols, figsize=(3.0 * n_cols, 4.4), constrained_layout=True)
    axes_array = np.asarray(axes).reshape(2, n_cols)
    for col_idx, run in enumerate(representative_rows):
        run_id = str(run["run_id"])
        mechanism = str(run["mechanism"])
        truth_pairwise = [tuple(edge) for edge in run.get("truth_pairwise_edges", [])]
        truth_hyperedges = [tuple(edge) for edge in run.get("truth_hyperedges", [])]
        pair_k = len(truth_pairwise) if truth_pairwise else 2
        hyper_k = len(truth_hyperedges)
        granger_edges = _top_edges_for_run(
            edge_frame,
            run_id=run_id,
            edge_type="granger_pairwise",
            score_col="score",
            k=pair_k,
        )
        peid_pairwise = _top_edges_for_run(
            edge_frame,
            run_id=run_id,
            edge_type="peid_pairwise",
            score_col="ei",
            k=len(truth_pairwise),
        )
        peid_hyperedges = _top_hyperedges_for_run(edge_frame, run_id=run_id, k=hyper_k)
        _draw_graph_panel(
            axes_array[0, col_idx],
            title=f"{mechanism}\nGranger top pairwise",
            pairwise_edges=granger_edges,
        )
        _draw_graph_panel(
            axes_array[1, col_idx],
            title="PEID pairwise + hyperedge",
            pairwise_edges=peid_pairwise,
            hyperedges=peid_hyperedges,
        )
    fig.suptitle("Representative learned causal graphs", fontsize=10)
    path = figure_dir / "representative_causal_graphs.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw_loss_panel(ax, *, title: str, loss_history: Sequence[float]) -> None:
    values = np.asarray(loss_history, dtype=float)
    ax.set_title(title, fontsize=8, pad=4)
    if values.size:
        ax.plot(np.arange(1, values.size + 1), values, color="#333333", linewidth=1.5)
        ax.scatter([values.size], [values[-1]], color="#333333", s=12, zorder=3)
    ax.set_xlabel("epoch", fontsize=7)
    ax.set_ylabel("MSE", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.18, linewidth=0.5)


def _representative_run_rows(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    run_frame = pd.DataFrame(runs)
    if run_frame.empty:
        return []
    return (
        run_frame.sort_values(["mechanism", "noise", "seed", "n_samples", "synergy_strength"])
        .groupby("mechanism", as_index=False)
        .head(1)
        .to_dict("records")
    )


def _plot_report_panels(
    runs: list[dict[str, object]],
    edge_rows: list[dict[str, object]],
    figure_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )
    representative_rows = _representative_run_rows(runs)
    edge_frame = pd.DataFrame(edge_rows)
    n_rows = max(1, len(representative_rows))
    fig, axes = plt.subplots(n_rows, 4, figsize=(12.2, 3.2 * n_rows), constrained_layout=True)
    axes_array = np.asarray(axes).reshape(n_rows, 4)

    for row_idx, run in enumerate(representative_rows):
        run_id = str(run["run_id"])
        mechanism = str(run["mechanism"])
        truth_pairwise = [tuple(edge) for edge in run.get("truth_pairwise_edges", [])]
        truth_hyperedges = [tuple(edge) for edge in run.get("truth_hyperedges", [])]
        pair_k = len(truth_pairwise) if truth_pairwise else 2
        hyper_k = len(truth_hyperedges)
        granger_edges = _top_edges_for_run(
            edge_frame,
            run_id=run_id,
            edge_type="granger_pairwise",
            score_col="score",
            k=pair_k,
        )
        peid_pairwise = _top_edges_for_run(
            edge_frame,
            run_id=run_id,
            edge_type="peid_pairwise",
            score_col="ei",
            k=len(truth_pairwise),
        )
        peid_hyperedges = _top_hyperedges_for_run(edge_frame, run_id=run_id, k=hyper_k)
        _draw_graph_panel(
            axes_array[row_idx, 0],
            title=f"{mechanism}\nGround truth",
            pairwise_edges=truth_pairwise,
            hyperedges=truth_hyperedges,
        )
        _draw_loss_panel(
            axes_array[row_idx, 1],
            title="MLP learning curve",
            loss_history=run.get("loss_history", []),
        )
        _draw_graph_panel(
            axes_array[row_idx, 2],
            title="time lag / Granger",
            pairwise_edges=granger_edges,
        )
        _draw_graph_panel(
            axes_array[row_idx, 3],
            title="PEID",
            pairwise_edges=peid_pairwise,
            hyperedges=peid_hyperedges,
        )

    fig.suptitle("Experiment examples: ground truth, MLP learning, Granger, and PEID", fontsize=11)
    path = figure_dir / "experiment_report_panels.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _relative_markdown_path(target: Path, markdown_path: Path) -> str:
    return os.path.relpath(target, start=markdown_path.parent).replace(os.sep, "/")


def _write_chinese_report(
    runs: list[dict[str, object]],
    *,
    edge_rows: list[dict[str, object]],
    lagged_proxy_result: dict[str, object],
    lag_sensitivity_result: dict[str, object],
    neural_granger_result: dict[str, object],
    lagged_proxy_figure_path: Path,
    summary_figure_path: Path,
    graph_figure_path: Path,
    report_figure_path: Path,
    sine_readout_figure_path: Path | None,
    proxy_y_figure_path: Path | None,
    alpha_sweep_figure_path: Path | None,
    alpha_neural_granger_figure_path: Path | None,
    alpha_sweep_rows: list[dict[str, float]],
    beta_sweep_figure_path: Path | None,
    beta_validation_figure_path: Path | None,
    beta_sweep_result: dict[str, object],
    report_path: Path,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = (
        pd.DataFrame(runs)
        .groupby("mechanism", as_index=False)[
            ["granger_pairwise_f1", "peid_pairwise_f1", "peid_hyperedge_f1", "peid_advantage"]
        ]
        .mean()
        .sort_values("mechanism")
    )
    if aggregate.empty:
        table = "_无结果_"
    else:
        table_lines = [
            "| mechanism | granger_pairwise_f1 | peid_pairwise_f1 | peid_hyperedge_f1 | peid_advantage |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in aggregate.to_dict("records"):
            table_lines.append(
                "| {mechanism} | {granger:.3f} | {pairwise:.3f} | {hyper:.3f} | {advantage:.3f} |".format(
                    mechanism=row["mechanism"],
                    granger=float(row["granger_pairwise_f1"]),
                    pairwise=float(row["peid_pairwise_f1"]),
                    hyper=float(row["peid_hyperedge_f1"]),
                    advantage=float(row["peid_advantage"]),
                )
            )
        table = "\n".join(table_lines)

    proxy_table = "_未生成 common-driver 代表性边权表_"
    sine_system_table = "_未生成共同驱动 sine 协同动力系统代表性结果表_"
    product_memory_table = "_未生成二阶协同动力系统代表性结果表_"
    if runs and edge_rows:
        run_frame = pd.DataFrame(runs)
        edge_frame = pd.DataFrame(edge_rows)
        common_rows = run_frame[run_frame["mechanism"] == "redundant_common_driver"]
        if not common_rows.empty and not edge_frame.empty:
            common_run = (
                common_rows.sort_values(["noise", "seed", "n_samples", "synergy_strength"])
                .head(1)
                .iloc[0]
            )
            common_run_id = str(common_run["run_id"])

            def edge_score(edge_type: str, score_col: str, source: str, target: str) -> float:
                subset = edge_frame[
                    (edge_frame["run_id"] == common_run_id)
                    & (edge_frame["edge_type"] == edge_type)
                    & (edge_frame["source"] == source)
                    & (edge_frame["target"] == target)
                ]
                if subset.empty or score_col not in subset:
                    return 0.0
                return float(subset.iloc[0][score_col])

            proxy_lines = [
                "| method | true `w -> z` | proxy `x -> z` | proxy `y -> z` | max proxy / true |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
            for method_name, edge_type, score_col in [
                ("Granger/ablation score", "granger_pairwise", "score"),
                ("PEID pairwise EI", "peid_pairwise", "ei"),
            ]:
                true_score = edge_score(edge_type, score_col, "w", "z")
                x_proxy = edge_score(edge_type, score_col, "x", "z")
                y_proxy = edge_score(edge_type, score_col, "y", "z")
                ratio = max(x_proxy, y_proxy) / true_score if true_score > 0.0 else 0.0
                proxy_lines.append(
                    f"| {method_name} | {true_score:.4g} | {x_proxy:.4g} | {y_proxy:.4g} | {ratio:.4g} |"
                )
            proxy_table = "\n".join(proxy_lines)
        sine_rows = run_frame[run_frame["mechanism"] == "common_driver_sine_synergy"]
        if not sine_rows.empty and not edge_frame.empty:
            sine_run = (
                sine_rows.sort_values(["noise", "seed", "n_samples", "synergy_strength"])
                .head(1)
                .iloc[0]
            )
            sine_run_id = str(sine_run["run_id"])

            def sine_pair_score(edge_type: str, score_col: str, source: str, target: str) -> float:
                subset = edge_frame[
                    (edge_frame["run_id"] == sine_run_id)
                    & (edge_frame["edge_type"] == edge_type)
                    & (edge_frame["source"] == source)
                    & (edge_frame["target"] == target)
                ]
                if subset.empty or score_col not in subset:
                    return 0.0
                return float(subset.iloc[0][score_col])

            def sine_synergy_score(sources: str, target: str, score_col: str) -> float:
                subset = edge_frame[
                    (edge_frame["run_id"] == sine_run_id)
                    & (edge_frame["edge_type"] == "peid_synergy")
                    & (edge_frame["sources"] == sources)
                    & (edge_frame["target"] == target)
                ]
                if subset.empty or score_col not in subset:
                    return 0.0
                return float(subset.iloc[0][score_col])

            def sine_shap_score(source: str, target: str, score_col: str) -> float:
                subset = edge_frame[
                    (edge_frame["run_id"] == sine_run_id)
                    & (edge_frame["edge_type"] == "interventional_shap")
                    & (edge_frame["source"] == source)
                    & (edge_frame["target"] == target)
                ]
                if subset.empty or score_col not in subset:
                    return 0.0
                return float(subset.iloc[0][score_col])

            def sine_shap_interaction_score(sources: str, target: str, score_col: str) -> float:
                subset = edge_frame[
                    (edge_frame["run_id"] == sine_run_id)
                    & (edge_frame["edge_type"] == "interventional_shap_interaction")
                    & (edge_frame["sources"] == sources)
                    & (edge_frame["target"] == target)
                ]
                if subset.empty or score_col not in subset:
                    return 0.0
                return float(subset.iloc[0][score_col])

            def sine_interaction_score(sources: str, target: str, score_col: str) -> float:
                subset = edge_frame[
                    (edge_frame["run_id"] == sine_run_id)
                    & (edge_frame["edge_type"] == "product_interaction_probe")
                    & (edge_frame["sources"] == sources)
                    & (edge_frame["target"] == target)
                ]
                if subset.empty or score_col not in subset:
                    return 0.0
                return float(subset.iloc[0][score_col])

            sine_system_table = "\n".join(
                [
                    "| quantity | value |",
                    "| --- | ---: |",
                    f"| fitted MLP final training loss | {float(sine_run['final_train_loss']):.4g} |",
                    f"| Granger/ablation `w -> x` | {sine_pair_score('granger_pairwise', 'score', 'w', 'x'):.4g} |",
                    f"| Granger/ablation `w -> y` | {sine_pair_score('granger_pairwise', 'score', 'w', 'y'):.4g} |",
                    f"| Granger/ablation `w -> z` | {sine_pair_score('granger_pairwise', 'score', 'w', 'z'):.4g} |",
                    f"| Granger/ablation `x -> z` | {sine_pair_score('granger_pairwise', 'score', 'x', 'z'):.4g} |",
                    f"| Granger/ablation `y -> z` | {sine_pair_score('granger_pairwise', 'score', 'y', 'z'):.4g} |",
                    f"| Neural Granger `w -> x` | {sine_pair_score('neural_granger', 'group_norm', 'w', 'x'):.4g} |",
                    f"| Neural Granger `w -> y` | {sine_pair_score('neural_granger', 'group_norm', 'w', 'y'):.4g} |",
                    f"| Neural Granger `w -> z` | {sine_pair_score('neural_granger', 'group_norm', 'w', 'z'):.4g} |",
                    f"| Neural Granger `x -> z` | {sine_pair_score('neural_granger', 'group_norm', 'x', 'z'):.4g} |",
                    f"| Neural Granger `y -> z` | {sine_pair_score('neural_granger', 'group_norm', 'y', 'z'):.4g} |",
                    f"| SHAP mean abs `w -> x` | {sine_shap_score('w', 'x', 'mean_abs_phi'):.4g} |",
                    f"| SHAP mean abs `w -> y` | {sine_shap_score('w', 'y', 'mean_abs_phi'):.4g} |",
                    f"| SHAP mean abs `w -> z` | {sine_shap_score('w', 'z', 'mean_abs_phi'):.4g} |",
                    f"| SHAP mean abs `x -> z` | {sine_shap_score('x', 'z', 'mean_abs_phi'):.4g} |",
                    f"| SHAP mean abs `y -> z` | {sine_shap_score('y', 'z', 'mean_abs_phi'):.4g} |",
                    f"| SHAP interaction mean abs `x:y -> z` | {sine_shap_interaction_score('x+y', 'z', 'mean_abs_interaction'):.4g} |",
                    f"| product interaction `x:y -> z` incremental `R^2` | {sine_interaction_score('x+y', 'z', 'incremental_r2'):.4g} |",
                    f"| product interaction `x:y -> z` coefficient | {sine_interaction_score('x+y', 'z', 'interaction_coef'):.4g} |",
                    f"| product interaction `w:x -> z` incremental `R^2` | {sine_interaction_score('x+w', 'z', 'incremental_r2'):.4g} |",
                    f"| product interaction `w:y -> z` incremental `R^2` | {sine_interaction_score('y+w', 'z', 'incremental_r2'):.4g} |",
                    f"| PEID pairwise EI `w -> x` | {sine_pair_score('peid_pairwise', 'ei', 'w', 'x'):.4g} |",
                    f"| PEID pairwise EI `w -> y` | {sine_pair_score('peid_pairwise', 'ei', 'w', 'y'):.4g} |",
                    f"| PEID pairwise EI `w -> z` | {sine_pair_score('peid_pairwise', 'ei', 'w', 'z'):.4g} |",
                    f"| PEID pairwise EI `x -> z` | {sine_pair_score('peid_pairwise', 'ei', 'x', 'z'):.4g} |",
                    f"| PEID pairwise EI `y -> z` | {sine_pair_score('peid_pairwise', 'ei', 'y', 'z'):.4g} |",
                    f"| PEID joint EI `{{x, y}} -> z` | {sine_synergy_score('x+y', 'z', 'joint_ei'):.4g} |",
                    f"| PEID synergy `{{x, y}} -> z` | {sine_synergy_score('x+y', 'z', 'synergy'):.4g} |",
                ]
            )
        product_rows = run_frame[run_frame["mechanism"] == "product_memory_synergy"]
        if not product_rows.empty and not edge_frame.empty:
            product_run = (
                product_rows.sort_values(["noise", "seed", "n_samples", "synergy_strength"])
                .head(1)
                .iloc[0]
            )
            product_run_id = str(product_run["run_id"])

            def product_pair_score(edge_type: str, score_col: str, source: str, target: str) -> float:
                subset = edge_frame[
                    (edge_frame["run_id"] == product_run_id)
                    & (edge_frame["edge_type"] == edge_type)
                    & (edge_frame["source"] == source)
                    & (edge_frame["target"] == target)
                ]
                if subset.empty or score_col not in subset:
                    return 0.0
                return float(subset.iloc[0][score_col])

            def product_synergy_score(sources: str, target: str, score_col: str) -> float:
                subset = edge_frame[
                    (edge_frame["run_id"] == product_run_id)
                    & (edge_frame["edge_type"] == "peid_synergy")
                    & (edge_frame["sources"] == sources)
                    & (edge_frame["target"] == target)
                ]
                if subset.empty or score_col not in subset:
                    return 0.0
                return float(subset.iloc[0][score_col])

            product_memory_table = "\n".join(
                [
                    "| quantity | value |",
                    "| --- | ---: |",
                    f"| fitted MLP final training loss | {float(product_run['final_train_loss']):.4g} |",
                    f"| Granger/ablation `x -> z` | {product_pair_score('granger_pairwise', 'score', 'x', 'z'):.4g} |",
                    f"| Granger/ablation `y -> z` | {product_pair_score('granger_pairwise', 'score', 'y', 'z'):.4g} |",
                    f"| Granger/ablation `w -> z` | {product_pair_score('granger_pairwise', 'score', 'w', 'z'):.4g} |",
                    f"| PEID pairwise EI `x -> z` | {product_pair_score('peid_pairwise', 'ei', 'x', 'z'):.4g} |",
                    f"| PEID pairwise EI `y -> z` | {product_pair_score('peid_pairwise', 'ei', 'y', 'z'):.4g} |",
                    f"| PEID joint EI `{{x, y}} -> z` | {product_synergy_score('x+y', 'z', 'joint_ei'):.4g} |",
                    f"| PEID synergy `{{x, y}} -> z` | {product_synergy_score('x+y', 'z', 'synergy'):.4g} |",
                ]
            )
    lagged_proxy_table = "\n".join(
        [
            "| quantity | value |",
            "| --- | ---: |",
            f"| fitted-MLP Granger/ablation `x -> y` score | {lagged_proxy_result['pairwise_granger_x_to_y_score']:.4g} |",
            f"| one-source linear proxy `x_{{t+1}} -> y_{{t+2}}` score | {lagged_proxy_result['pairwise_linear_x_to_y_score']:.4g} |",
            f"| one-source linear proxy `R^2` | {lagged_proxy_result['pairwise_linear_x_to_y_r2']:.4g} |",
            f"| incremental score of `x_{{t+1}}` after conditioning on `w_t` | {lagged_proxy_result['conditional_x_incremental_score_given_w']:.4g} |",
            f"| conditional linear coefficient on proxy `x_{{t+1}}` | {lagged_proxy_result['causal_state_coef_x_proxy']:.4g} |",
            f"| conditional linear coefficient on driver `w_t` | {lagged_proxy_result['causal_state_coef_w_driver']:.4g} |",
            f"| fitted MLP `y` train MSE | {lagged_proxy_result['peid_mlp_y_train_mse']:.4g} |",
            f"| PEID EI `x_{{t+1}} -> y_{{t+2}}` on fitted MLP | {lagged_proxy_result['peid_ei_x_to_y']:.4g} |",
            f"| PEID EI `w_t -> y_{{t+2}}` on fitted MLP | {lagged_proxy_result['peid_ei_w_to_y']:.4g} |",
            f"| PEID proxy / driver EI ratio | {lagged_proxy_result['peid_proxy_to_driver_ratio']:.4g} |",
        ]
    )
    lagged_proxy_edge_lines = [
        "| edge | fitted-MLP Granger/ablation score | fitted-MLP PEID EI |",
        "| --- | ---: | ---: |",
    ]
    granger_edge_weights = dict(lagged_proxy_result["pairwise_granger_edges"])
    peid_edge_weights = dict(lagged_proxy_result["peid_ei_edges"])
    for edge in ("w->x", "w->y", "x->w", "x->y", "y->w", "y->x"):
        lagged_proxy_edge_lines.append(
            f"| `{edge}` | {float(granger_edge_weights[edge]):.4g} | {float(peid_edge_weights[edge]):.4g} |"
        )
    lagged_proxy_edge_table = "\n".join(lagged_proxy_edge_lines)
    lag_sensitivity_lines = [
        "| MLP max lag | Granger `w->y` | Granger `x->y` | PEID EI `w->y` | PEID EI `x->y` | interpretation |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for max_lag in sorted(lag_sensitivity_result["by_lag"]):
        row = lag_sensitivity_result["by_lag"][max_lag]
        granger_edges = dict(row["granger_edges"])
        peid_edges = dict(row["peid_ei_edges"])
        if int(max_lag) == 1:
            interpretation = "欠滞后，MLP 看不到真实 `w_{t-2}`，两种读出都转向代理 `x`"
        else:
            interpretation = "lag 足够，MLP 能看到真实 driver，主要权重回到 `w`"
        lag_sensitivity_lines.append(
            "| {max_lag} | {g_w:.4g} | {g_x:.4g} | {p_w:.4g} | {p_x:.4g} | {interpretation} |".format(
                max_lag=int(max_lag),
                g_w=float(granger_edges["w->y"]),
                g_x=float(granger_edges["x->y"]),
                p_w=float(peid_edges["w->y"]),
                p_x=float(peid_edges["x->y"]),
                interpretation=interpretation,
            )
        )
    lag_sensitivity_table = "\n".join(lag_sensitivity_lines)
    neural_granger_rows = list(neural_granger_result["rows"])
    neural_granger_lines = [
        "| max lag | target | source | rank | first-layer group norm | strongest lag | strongest lag norm |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in neural_granger_rows:
        if str(row["target"]) not in {"x", "y"}:
            continue
        if int(row["rank"]) > 2:
            continue
        neural_granger_lines.append(
            "| {max_lag} | `{target}` | `{source}` | {rank} | {group_norm:.4g} | {strongest_lag} | {strongest_lag_norm:.4g} |".format(
                max_lag=int(row["max_lag"]),
                target=str(row["target"]),
                source=str(row["source"]),
                rank=int(row["rank"]),
                group_norm=float(row["group_norm"]),
                strongest_lag=int(row["strongest_lag"]),
                strongest_lag_norm=float(row["strongest_lag_norm"]),
            )
        )
    neural_granger_table = "\n".join(neural_granger_lines)
    lagged_proxy_rel = _relative_markdown_path(lagged_proxy_figure_path, report_path)
    report_rel = _relative_markdown_path(report_figure_path, report_path)
    summary_rel = _relative_markdown_path(summary_figure_path, report_path)
    graph_rel = _relative_markdown_path(graph_figure_path, report_path)
    sine_readout_rel = (
        _relative_markdown_path(sine_readout_figure_path, report_path)
        if sine_readout_figure_path is not None
        else ""
    )
    sine_readout_block = (
        f"![同一 MLP 上的二维读出对照]({sine_readout_rel})\n\n"
        if sine_readout_rel
        else ""
    )
    proxy_y_rel = (
        _relative_markdown_path(proxy_y_figure_path, report_path)
        if proxy_y_figure_path is not None
        else ""
    )
    proxy_y_rows = _proxy_y_readout_values(edge_rows)
    proxy_y_block = ""
    if proxy_y_rel and proxy_y_rows:
        proxy_frame = pd.DataFrame(proxy_y_rows)
        proxy_table_lines = [
            "| method | `w->y` true driver | `x->y` proxy | `y->y` memory | `x/w` ratio |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for method in ["SHAP", "PEID EI", "Granger"]:
            values_by_source: dict[str, float] = {}
            for source in ["w", "x", "y"]:
                subset = proxy_frame[
                    (proxy_frame["method"] == method) & (proxy_frame["source"] == source)
                ]
                values_by_source[source] = float(subset.iloc[0]["value"]) if not subset.empty else 0.0
            ratio = values_by_source["x"] / (values_by_source["w"] + 1e-12)
            proxy_table_lines.append(
                f"| {method} | {values_by_source['w']:.4g} | {values_by_source['x']:.4g} | {values_by_source['y']:.4g} | {ratio:.4g} |"
            )
        proxy_y_block = (
            "## 第二章：代理变量情形：`x` 作为 `w -> y` 的 proxy\n\n"
            "同一个动力系统还包含一个不需要额外造数的代理变量实验。对目标 `y_{t+1}`，结构方程中有直接项 `w_t -> y_{t+1}` 与自回归项 `y_t -> y_{t+1}`，但没有 `x_t -> y_{t+1}`。不过 `x_t` 由自相关的 `w` 驱动，因此在观测分布上是 `w_t` 的代理变量。\n\n"
            f"![target y 的代理变量读出]({proxy_y_rel})\n\n"
            + "\n".join(proxy_table_lines)
            + "\n\n"
            "这里不再区分不同 SHAP 口径，只保留当前应用最常见的背景替换式 SHAP 基线。该读出在同一 fitted MLP 上计算 mean absolute attribution，用来表示特征对预测输出的平均贡献；PEID 使用最大熵独立干预读出，主要保留直接 driver `w->y` 与自回归 `y->y`，而不是把观测 proxy 当作强机制边。\n\n"
        )
    alpha_sweep_rel = (
        _relative_markdown_path(alpha_sweep_figure_path, report_path)
        if alpha_sweep_figure_path is not None
        else ""
    )
    alpha_neural_granger_rel = (
        _relative_markdown_path(alpha_neural_granger_figure_path, report_path)
        if alpha_neural_granger_figure_path is not None
        else ""
    )
    alpha_sweep_block = ""
    if alpha_sweep_rel:
        alpha_table_lines = [
            "| alpha | SHAP `x->z` | SHAP `y->z` | SHAP `w->z` | SHAP interaction `|x:y|` | Granger `x->z` | Granger `y->z` | Granger `w->z` | Neural Granger `x->z` | Neural Granger `y->z` | Neural Granger `w->z` | TM PEID joint EI `{x,y}->z` | TM PEID synergy `{x,y}->z` | TM PEID `w->z` |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in alpha_sweep_rows:
            alpha_table_lines.append(
                "| {alpha:.2f} | {shap_x:.4g} | {shap_y:.4g} | {shap_w:.4g} | {shap_interaction:.4g} | {granger_x:.4g} | {granger_y:.4g} | {granger_w:.4g} | {ng_x:.4g} | {ng_y:.4g} | {ng_w:.4g} | {joint:.4g} | {syn:.4g} | {tm_w:.4g} |".format(
                    alpha=float(row["alpha"]),
                    shap_x=float(row["shap_x_to_z_mean_abs"]),
                    shap_y=float(row["shap_y_to_z_mean_abs"]),
                    shap_w=float(row["shap_w_to_z_mean_abs"]),
                    shap_interaction=float(row["shap_xy_mean_abs_interaction"]),
                    granger_x=float(row["granger_x_to_z"]),
                    granger_y=float(row["granger_y_to_z"]),
                    granger_w=float(row["granger_w_to_z"]),
                    ng_x=float(row["neural_granger_x_to_z"]),
                    ng_y=float(row["neural_granger_y_to_z"]),
                    ng_w=float(row["neural_granger_w_to_z"]),
                    joint=float(row["tm_peid_xy_joint_ei"]),
                    syn=float(row["tm_peid_xy_synergy"]),
                    tm_w=float(row["tm_peid_w_to_z"]),
                )
            )
        alpha_sweep_block = (
            "### alpha 扫描：SHAP 交互与 PEID 协同\n\n"
            f"![alpha 扫描下的 SHAP 与 PEID 对照]({alpha_sweep_rel})\n\n"
            + (
                f'<img src="{alpha_neural_granger_rel}" alt="alpha 扫描下的 Neural Granger 单独读出" width="420">\n\n'
                "Neural Granger 单图单独展示 target-wise cMLP 的 first-layer source-group norm。"
                "该读数在 `alpha=0.2` 和 `alpha=0.8` 处把 sine 协同响应投影到 `x->z`、`y->z` 两条 pairwise 边，"
                "而 `w->z` 保持较低，说明它更像预测结构读出而不是源集合干预语义下的协同分解。\n\n"
                if alpha_neural_granger_rel
                else ""
            )
            + "\n".join(alpha_table_lines)
            + "\n\n"
            "这里的 `alpha` 是 sine 项前面的强度系数。`alpha=0` 时，`z` 只剩自身记忆与噪声，"
            "SHAP 二阶交互接近零；TM PEID 仅保留少量连续估计底噪。随着 `alpha` 增大，"
            "SHAP 单源 `x->z`、`y->z` 与 SHAP interaction 同时上升，但单源项是对协同响应的归因分摊，不是结构边；"
            "Granger/ablation 的 `x->z`、`y->z` 也会随 `alpha` 上升，因为它衡量单源置换对 fitted MLP 预测误差的影响；"
            "Neural Granger 的 cMLP group norm 同样是 pairwise 预测结构读出，会把 sine 协同响应投影到 `x->z` 与 `y->z`；"
            "这里的 PEID 曲线改用连续 transport-map EI，在同一最大熵联合干预样本上直接读出 `{x,y}` 对连续目标预测的机制信息约束。"
            "\n\n"
        )
    beta_sweep_rel = (
        _relative_markdown_path(beta_sweep_figure_path, report_path)
        if beta_sweep_figure_path is not None
        else ""
    )
    beta_validation_rel = (
        _relative_markdown_path(beta_validation_figure_path, report_path)
        if beta_validation_figure_path is not None
        else ""
    )
    beta_sweep_block = ""
    beta_summary_rows = list(beta_sweep_result.get("summary", []))
    beta_trend = dict(beta_sweep_result.get("trend", {}))
    if beta_sweep_rel and beta_summary_rows:
        beta_table_lines = [
            "| beta | corr(`x`,`y`) | observational WMS | SHAP `x` | SHAP `y` | SHAP `x:y` | Neural Granger `x/y->z` | PCMCI-CMIknn `x/y/w->z` | SURD R/Ux/Uy/S | MLP+PEID Ux/Uy/S |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
        for row in beta_summary_rows:
            beta_table_lines.append(
                "| {beta:.2f} | {corr:.4g} | {wms:.4g} | {shap_x:.4g} | {shap_y:.4g} | {shap_xy:.4g} | {ng_x:.4g}/{ng_y:.4g} | {pcmci_x:.4g}/{pcmci_y:.4g}/{pcmci_w:.4g} | {surd_r:.4g}/{surd_ux:.4g}/{surd_uy:.4g}/{surd_s:.4g} | {peid_ux:.4g}/{peid_uy:.4g}/{peid_s:.4g} |".format(
                    beta=float(row["beta"]),
                    corr=float(row["xy_observed_corr_mean"]),
                    wms=float(row["observational_wms_mean"]),
                    shap_x=float(row["shap_x_to_z_mean_abs_mean"]),
                    shap_y=float(row["shap_y_to_z_mean_abs_mean"]),
                    shap_xy=float(row["shap_xy_mean_abs_interaction_mean"]),
                    ng_x=float(row["neural_granger_x_to_z_mean"]),
                    ng_y=float(row["neural_granger_y_to_z_mean"]),
                    pcmci_x=float(row.get("pcmci_cmiknn_x_to_z_mean", float("nan"))),
                    pcmci_y=float(row.get("pcmci_cmiknn_y_to_z_mean", float("nan"))),
                    pcmci_w=float(row.get("pcmci_cmiknn_w_to_z_mean", float("nan"))),
                    surd_r=float(row["surd_redundancy_mean"]),
                    surd_ux=float(row["surd_unique_x_mean"]),
                    surd_uy=float(row["surd_unique_y_mean"]),
                    surd_s=float(row["surd_xy_synergy_mean"]),
                    peid_ux=float(row["mlp_peid_unique_x_mean"]),
                    peid_uy=float(row["mlp_peid_unique_y_mean"]),
                    peid_s=float(row["mlp_peid_xy_synergy_mean"]),
                )
            )
        beta_sweep_block = (
            "### beta 扫描：共同驱动增强但结构协同固定\n\n"
            "这里固定 `alpha=1`，只改变 `x,y` 的共同驱动强度 `beta`。生成式中 `beta` 增大只让 `x` 与 `y` 在观测轨迹上更相关，"
            "并没有增强 `z_{t+1}` 中的 `sin(x_t y_t)` 结构项。因此理论预期是：`{x,y}->z` 的 PEID 协同不应因为 `beta` 增大而单调增加。\n\n"
            "beta 扫描对应的动力学为\n\n"
            "$$\n"
            "\\begin{aligned}\n"
            "w_{t+1} &= 0.78w_t + \\eta^w_t,\\\\\n"
            "x_{t+1} &= 0.42x_t + 0.82\\left(\\beta w_t + \\sqrt{1-\\beta^2}\\,\\xi^x_t\\right) + \\eta^x_t,\\\\\n"
            "y_{t+1} &= 0.38y_t + 0.76\\left(\\beta w_t + \\sqrt{1-\\beta^2}\\,\\xi^y_t\\right) + \\eta^y_t,\\\\\n"
            "z_{t+1} &= 0.22z_t + \\sin\\left(x_t y_t\\right) + \\eta^z_t.\n"
            "\\end{aligned}\n"
            "$$\n\n"
            "其中 `beta=0` 时，`x` 与 `y` 主要由各自私有扰动驱动；`beta=1` 时，它们的新增驱动完全共享同一个 `w_t`。"
            "`\\sqrt{1-\\beta^2}` 是 `beta` 的互补私有驱动权重，使共享驱动项和私有驱动项的平方权重和保持为 1；"
            "这样 beta 扫描主要改变源变量之间的观测相关性，而不是简单放大或缩小 `x,y` 的总驱动强度。"
            "`z` 的结构项始终是同一个 `sin(x_t y_t)`，因此 beta 不改变二源机制本身。\n\n"
            f"![beta 扫描统一方法对照]({beta_sweep_rel})\n\n"
            + "\n".join(beta_table_lines)
            + "\n\n"
            "每个 `beta × seed` 只生成一次轨迹并训练一个 MLP。Observational SURD 直接作用于这条自然轨迹；"
            "左上角 WMS 也直接使用该自然轨迹上对齐的 `(x_t,y_t,z_{t+1})`，计算 "
            "`I([x_t,y_t];z_{t+1}) - I(x_t;z_{t+1}) - I(y_t;z_{t+1})`。"
            "三个 MI 均由相同的四分位离散经验联合分布直接计算，并保留 WMS 负值；"
            "MLP+SHAP 与 MLP+PEID 共享同一个 fitted MLP，PEID 与 Oracle+PEID 共享同一组独立干预源样本。"
            "Neural Granger 在同一自然轨迹上训练 target-wise cMLP，并以 first-layer source-group norm 作为 pairwise 读出。"
            "PCMCI-CMIknn 在同一自然轨迹上运行非线性条件独立检验，图中显示 lag-1 pairwise 依赖强度的绝对值。"
            "SURD 与 PEID 的 transport-map 输入均为原始源变量，信息量单位统一为 bits。"
            "SHAP、Neural Granger 与 PCMCI-CMIknn 保留自身原始读出尺度，不与信息量绝对值直接比较。\n\n"
            "线性趋势读数显示，observational WMS 的 beta 斜率为 "
            f"{float(beta_trend.get('observational_wms_slope', float('nan'))):.4g} "
            f"(bootstrap 95% CI [{float(beta_trend.get('observational_wms_slope_ci_low', float('nan'))):.4g}, "
            f"{float(beta_trend.get('observational_wms_slope_ci_high', float('nan'))):.4g}])；"
            "SHAP interaction 的 beta 斜率为 "
            f"{float(beta_trend.get('shap_interaction_slope', float('nan'))):.4g} "
            f"(bootstrap 95% CI [{float(beta_trend.get('shap_interaction_slope_ci_low', float('nan'))):.4g}, "
            f"{float(beta_trend.get('shap_interaction_slope_ci_high', float('nan'))):.4g}])；"
            "Observational SURD synergy 的 beta 斜率为 "
            f"{float(beta_trend.get('surd_synergy_slope', float('nan'))):.4g} "
            f"(bootstrap 95% CI [{float(beta_trend.get('surd_synergy_slope_ci_low', float('nan'))):.4g}, "
            f"{float(beta_trend.get('surd_synergy_slope_ci_high', float('nan'))):.4g}])；"
            "PCMCI-CMIknn `x/y->z` 合计强度的 beta 斜率为 "
            f"{float(beta_trend.get('pcmci_cmiknn_xy_to_z_slope', float('nan'))):.4g} "
            f"(bootstrap 95% CI [{float(beta_trend.get('pcmci_cmiknn_xy_to_z_slope_ci_low', float('nan'))):.4g}, "
            f"{float(beta_trend.get('pcmci_cmiknn_xy_to_z_slope_ci_high', float('nan'))):.4g}])；"
            "MLP+PEID synergy 的 beta 斜率为 "
            f"{float(beta_trend.get('tm_peid_synergy_slope', float('nan'))):.4g} "
            f"(bootstrap 95% CI [{float(beta_trend.get('tm_peid_synergy_slope_ci_low', float('nan'))):.4g}, "
            f"{float(beta_trend.get('tm_peid_synergy_slope_ci_high', float('nan'))):.4g}])。\n\n"
            + (
                f"![Oracle+PEID 与 SURD Q1 验证]({beta_validation_rel})\n\n"
                if beta_validation_rel
                else ""
            )
            + "验证图中的 Oracle+PEID 只用于检查 learned MLP 的 PEID 趋势是否偏离真实转移方程；"
            "SURD Q1 原子用于确认原论文 specific-MI transport-map 复现入口。二者都不进入主方法排名。\n\n"
        )

    text = f"""# 统一动力系统：共同驱动 + sine 协同

这个例子把两种容易混淆的结构放进同一个动力系统：一方面，`w` 是 `x`、`y` 背后的共同原因；另一方面，`x`、`y` 对 `z` 的作用不是两条可分离的 pairwise 边，而是一个二源协同项。系统为

$$
\\begin{{aligned}}
w_{{t+1}} &= 0.78w_t + \\eta^w_t,\\\\
x_{{t+1}} &= 0.42x_t + 0.82w_t + \\eta^x_t,\\\\
y_{{t+1}} &= 0.38y_t + 0.76w_t + \\eta^y_t,\\\\
z_{{t+1}} &= 0.22z_t + \\alpha\\sin\\left(x_t y_t\\right) + \\eta^z_t.
\\end{{aligned}}
$$

其中 `w` 不直接进入 `z` 的结构方程。真实机制应读成两层：

- pairwise 层面：`w -> x`、`w -> y`；
- 高阶层面：`{{x, y}} -> z`；
- 非结构边：`w -> z` 不是直接机制边，单独的 `x -> z`、`y -> z` 也只是 sine 协同项的 pairwise 投影。

## 读出方式

同一条模拟时间序列先用于训练一个 MLP 一步转移模型，输入为 `[x_t, y_t, z_t, w_t]`，输出为 `[x_{{t+1}}, y_{{t+1}}, z_{{t+1}}, w_{{t+1}}]`。随后在同一轨迹或固定 MLP 上读出几类量：

- Granger/ablation：把某个 source 的输入列替换为均值，记录目标预测 MSE 的增量。它回答“去掉这个变量会不会损害预测”。
- Neural Granger：对每个 target 单独训练带 group-lasso 的 cMLP，并读取第一层按 source lag group 聚合的权重范数。它回答“target-wise 非线性预测器是否使用这个 source 的历史输入”，仍是 pairwise 预测结构读出。
- SHAP 类归因：在同一 fitted MLP 上只保留一个常用背景替换式 SHAP 基线，用经验背景替换未给定特征。单特征 SHAP 报告 mean absolute attribution；二阶 SHAP interaction 报告 `x:y` 的 mean absolute interaction。前者回答“某个特征分到多少预测贡献”，后者回答“两个特征的非加性预测贡献有多大”。
- 交互项 probe：在同一 fitted MLP 的最大熵干预预测面上，用标准化主效应加一个二阶乘积项拟合目标输出，并记录该乘积项相对于主效应模型的 incremental `R^2`。它回答“固定这个预测器时，响应面是否含有可由 `x:y` 近似的二阶非加性形状”。
- PEID：先做最大熵独立干预，再计算 single-source EI、joint EI 和 synergy：

$$
\\mathrm{{Syn}}(\\{{x,y\\}}\\to z)
= EI(\\{{x,y\\}}\\to z)-EI(x\\to z)-EI(y\\to z).
$$

- Observational SURD：直接在自然轨迹的 `(x_t,y_t,z_{{t+1}})` 上，按原论文方式先用 transport map 估计逐目标状态的 specific MI：

$$
R_{{xy}}(z)=\\min\\{{i_x(z),i_y(z)\\}},\\quad
U_x(z)=i_x(z)-R_{{xy}}(z),\\quad
U_y(z)=i_y(z)-R_{{xy}}(z),\\quad
S_{{xy}}(z)=i_{{xy}}(z)-\\max\\{{i_x(z),i_y(z)\\}}.
$$

最后对目标状态积分得到 `Rxy/Ux/Uy/Sxy`，满足 `Rxy + Ux + Uy + Sxy = I({{x,y}};z)`。SURD 描述观测分布中的冗余、特有与协同；PEID 描述独立干预后机制映射的信息约束，两者回答的问题不同。PEID 当前定义不单独分配冗余原子，因此报告中其 redundancy 显式记为零。

独立入口 `scripts/reproduce_surd_synergistic_collider.py` 保留用于原论文 Q1 的 11 原子复现；Q1 的主导原子应为 `S23`。该验证只确认 SURD specific-MI transport-map 实现，不进入共同驱动 sine 主方法排名。

## 第一章：二源协同情形：`{{x,y}} -> z`

### 代表性结果

{sine_system_table}

{sine_readout_block}图中左侧热图把 Granger、Neural Granger、SHAP 和 PEID 的单源读出放在同一组边上比较。因为各行单位不同，颜色只在每一行内部归一化，格子里的数字才是原始读数。右上角显示标准化乘积项对 `z` 的增量解释度：`x:y` 明显高于 `w:x` 与 `w:y`。右下角显示 PEID 对 `z` 的信息分解，联合 EI 与 synergy 高于单源 EI。

{alpha_sweep_block}

{beta_sweep_block}

### 解释

`w -> x` 和 `w -> y` 在 Granger/ablation、Neural Granger 与 PEID pairwise EI 中都很强，说明预测器学到了共同驱动结构。`w -> z` 很小，符合结构方程中 `w` 不直接进入 `z` 的设定；若某些归因方法给出非零 `w -> z`，应解释为 `w` 通过诱导 `x,y` 相关性形成的代理贡献，而不是直接结构边。

对 `z` 来说，Granger/ablation 和 Neural Granger 会给出明显的 `x -> z` 与 `y -> z`，SHAP 类单特征归因也会倾向把 sine 项拆成单变量贡献。交互项 probe 则能进一步指出 fitted MLP 的响应面中确实存在强 `x:y` 二阶非加性项，因此它比纯单特征 SHAP 更接近“有交互”的诊断；但它仍然是响应面形状分析，不是源侧最大熵干预语义下的机制信息分解。这些读出有预测解释价值，但它们把

$$
\\alpha\\sin(x_t y_t)
$$

投影成了 pairwise 贡献或低阶乘积项，不能单独表达“只有联合给定 `x_t` 和 `y_t` 时才稳定确定目标响应”的机制事实。

PEID 的关键读数是 `EI({{x, y}} -> z)` 与 `Syn({{x, y}} -> z)` 均显著高于单源投影。它说明联合干预 `{{x,y}}` 后，目标分布的约束远超过两个单源 EI 的加和。因此这个例子的结论不是“PEID 消除了所有代理效应”，而是：在同一个 learned transition surrogate 上，PEID 可以同时保留 `w -> x,y` 的共同驱动边，以及 `{{x,y}} -> z` 的协同超边；Granger、Neural Granger 和 SHAP 单特征方法主要给出预测贡献的 pairwise 投影，交互项 probe 可以提示 `x:y` 非加性存在，但 PEID 才把这个非加性读成源集合到目标的协同有效信息。

{proxy_y_block}
"""
    report_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return report_path


def _resolve_report_path(report_path: Path | str | None, result_dir: Path, figure_dir: Path) -> Path:
    if report_path is not None:
        return Path(report_path)
    if result_dir.resolve() == DEFAULT_RESULT_DIR.resolve() and figure_dir.resolve() == DEFAULT_FIGURE_DIR.resolve():
        return DEFAULT_REPORT_PATH
    return result_dir / "granger_peid_mlp_comparison.md"


def run_comparison_grid(
    *,
    mode: str = "smoke",
    mechanisms: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    noise_values: Sequence[float] | None = None,
    sample_values: Sequence[int] | None = None,
    synergy_values: Sequence[float] | None = None,
    result_dir: Path | str = DEFAULT_RESULT_DIR,
    figure_dir: Path | str = DEFAULT_FIGURE_DIR,
    report_path: Path | str | None = None,
    include_diagnostic_sweeps: bool | None = None,
) -> dict[str, str]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")

    default_mechanisms = (
        "linear_additive",
        "xor_synergy",
        "multiplicative_gate",
        "product_memory_synergy",
        "common_driver_sine_synergy",
        "redundant_common_driver",
    )
    mechanisms = tuple(mechanisms or default_mechanisms)
    seeds = tuple(int(seed) for seed in (seeds or ((0, 1) if mode == "smoke" else tuple(range(10)))))
    noise_values = tuple(float(value) for value in (noise_values or ((0.05, 0.20) if mode == "smoke" else (0.01, 0.05, 0.10, 0.20, 0.40))))
    sample_values = tuple(int(value) for value in (sample_values or ((1500,) if mode == "smoke" else (1000, 3000, 10000))))
    synergy_values = tuple(float(value) for value in (synergy_values or ((1.0,) if mode == "smoke" else (0.25, 0.5, 1.0, 2.0))))
    result_dir = Path(result_dir)
    figure_dir = Path(figure_dir)
    report_path = _resolve_report_path(report_path, result_dir, figure_dir)
    run_diagnostic_sweeps = (
        (report_path.resolve() == DEFAULT_REPORT_PATH.resolve()) or mode == "full"
        if include_diagnostic_sweeps is None
        else bool(include_diagnostic_sweeps)
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    for mechanism, seed, noise, n_samples, synergy_strength in product(
        mechanisms,
        seeds,
        noise_values,
        sample_values,
        synergy_values,
    ):
        if mechanism not in {
            "multiplicative_gate",
            "product_memory_synergy",
            "common_driver_sine_synergy",
            "xor_synergy",
        } and synergy_strength != synergy_values[0]:
            continue
        config = SimConfig(
            mechanism=str(mechanism),
            n_samples=int(n_samples),
            noise=float(noise),
            seed=int(seed),
            synergy_strength=float(synergy_strength),
            mlp_epochs=80 if mode == "smoke" else 140,
            intervention_samples=512 if mode == "smoke" else 1024,
            bins=2 if mechanism == "xor_synergy" else 4,
        )
        series, truth = simulate_system(config)
        features, targets = make_lagged_dataset(series, lag=config.lag)
        model = train_mlp_transition_model(features, targets, config)
        granger_edges = estimate_granger_graph(model, features, targets, config)
        peid = estimate_peid_graph(model, series, config)
        is_report_sine_run = (
            mechanism == "common_driver_sine_synergy"
            and int(seed) == int(seeds[0])
            and float(noise) == float(noise_values[0])
            and int(n_samples) == int(sample_values[0])
            and float(synergy_strength) == float(synergy_values[0])
        )
        shap_readout = (
            estimate_shap_readout(model, features, series, config)
            if is_report_sine_run
            else None
        )
        conditional_shap_readout = (
            estimate_conditional_shap_readout(model, features, config, target="y")
            if is_report_sine_run
            else None
        )
        neural_granger_readout = (
            run_neural_granger_readout(
                features,
                targets,
                variable_names=config.variable_names,
                max_lag=config.lag,
                seed=int(seed) + 9101,
            )
            if is_report_sine_run
            else None
        )
        metrics = _evaluate_run(truth, granger_edges, peid)
        run_id = f"{mechanism}_seed{seed}_n{n_samples}_noise{noise:g}_syn{synergy_strength:g}"
        runs.append(
            {
                "run_id": run_id,
                "mechanism": str(mechanism),
                "seed": int(seed),
                "noise": float(noise),
                "n_samples": int(n_samples),
                "synergy_strength": float(synergy_strength),
                "truth_pairwise_edges": [list(edge) for edge in truth["pairwise_edges"]],
                "truth_hyperedges": [list(edge) for edge in truth["hyperedges"]],
                "loss_history": [float(value) for value in model.loss_history],
                "final_train_loss": float(model.loss_history[-1]) if model.loss_history else float("nan"),
                **metrics,
            }
        )
        edge_rows.extend(
            _edge_records(
                run_id,
                config,
                granger_edges,
                peid,
                shap_readout,
                conditional_shap_readout,
                neural_granger_readout,
            )
        )

    summary_path = result_dir / "summary.json"
    edge_table_path = result_dir / "edge_table.jsonl"
    lagged_proxy_result = run_lagged_proxy_common_driver_experiment()
    lag_sensitivity_result = run_lag_sensitivity_lagged_proxy_experiment()
    neural_granger_result = run_neural_granger_lagged_proxy_experiment()
    lagged_proxy_figure_path = _plot_lagged_proxy_causal_graph(lagged_proxy_result, figure_dir)
    figure_path = _plot_summary(runs, figure_dir)
    graph_figure_path = _plot_representative_causal_graphs(runs, edge_rows, figure_dir)
    report_figure_path = _plot_report_panels(runs, edge_rows, figure_dir)
    sine_readout_figure_path = _plot_sine_readout_summary(edge_rows, figure_dir)
    proxy_y_figure_path = _plot_proxy_y_readout_summary(edge_rows, figure_dir)
    alpha_sweep_rows = (
        run_sine_alpha_sweep()
        if run_diagnostic_sweeps and "common_driver_sine_synergy" in set(mechanisms)
        else []
    )
    alpha_sweep_figure_path = _plot_sine_alpha_sweep(alpha_sweep_rows, figure_dir)
    alpha_neural_granger_figure_path = _plot_sine_alpha_neural_granger_sweep(alpha_sweep_rows, figure_dir)
    beta_sweep_result = (
        run_sine_beta_common_driver_sweep()
        if run_diagnostic_sweeps and "common_driver_sine_synergy" in set(mechanisms)
        else {"runs": [], "summary": [], "trend": {}}
    )
    beta_sweep_figure_path = _plot_sine_beta_sweep(beta_sweep_result, figure_dir)
    beta_validation_figure_path = _plot_sine_beta_validation(beta_sweep_result, figure_dir)
    report_markdown_path = _write_chinese_report(
        runs,
        edge_rows=edge_rows,
        lagged_proxy_result=lagged_proxy_result,
        lag_sensitivity_result=lag_sensitivity_result,
        neural_granger_result=neural_granger_result,
        lagged_proxy_figure_path=lagged_proxy_figure_path,
        summary_figure_path=figure_path,
        graph_figure_path=graph_figure_path,
        report_figure_path=report_figure_path,
        sine_readout_figure_path=sine_readout_figure_path,
        proxy_y_figure_path=proxy_y_figure_path,
        alpha_sweep_figure_path=alpha_sweep_figure_path,
        alpha_neural_granger_figure_path=alpha_neural_granger_figure_path,
        alpha_sweep_rows=alpha_sweep_rows,
        beta_sweep_figure_path=beta_sweep_figure_path,
        beta_validation_figure_path=beta_validation_figure_path,
        beta_sweep_result=beta_sweep_result,
        report_path=report_path,
    )

    aggregate = (
        pd.DataFrame(runs)
        .groupby("mechanism", as_index=False)[
            ["granger_pairwise_f1", "peid_pairwise_f1", "peid_hyperedge_f1", "peid_advantage", "disagreement_score"]
        ]
        .mean()
        .to_dict("records")
        if runs
        else []
    )
    summary = {
        "mode": mode,
        "config": {
            "mechanisms": list(mechanisms),
            "seeds": list(seeds),
            "noise_values": list(noise_values),
            "sample_values": list(sample_values),
            "synergy_values": list(synergy_values),
        },
        "runs": runs,
        "aggregate_by_mechanism": aggregate,
        "figure_path": str(figure_path),
        "graph_figure_path": str(graph_figure_path),
        "report_figure_path": str(report_figure_path),
        "sine_readout_figure_path": str(sine_readout_figure_path) if sine_readout_figure_path else None,
        "proxy_y_figure_path": str(proxy_y_figure_path) if proxy_y_figure_path else None,
        "alpha_sweep_figure_path": str(alpha_sweep_figure_path) if alpha_sweep_figure_path else None,
        "alpha_neural_granger_figure_path": (
            str(alpha_neural_granger_figure_path) if alpha_neural_granger_figure_path else None
        ),
        "beta_sweep_figure_path": str(beta_sweep_figure_path) if beta_sweep_figure_path else None,
        "beta_validation_figure_path": str(beta_validation_figure_path) if beta_validation_figure_path else None,
        "lagged_proxy_figure_path": str(lagged_proxy_figure_path),
        "report_markdown_path": str(report_markdown_path),
        "edge_table_path": str(edge_table_path),
        "sine_alpha_sweep": alpha_sweep_rows,
        "sine_beta_common_driver_sweep": beta_sweep_result,
        "lagged_proxy_common_driver": lagged_proxy_result,
        "lag_sensitivity_lagged_proxy": lag_sensitivity_result,
        "neural_granger_lagged_proxy": neural_granger_result,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with edge_table_path.open("w", encoding="utf-8") as handle:
        for row in edge_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "summary_path": str(summary_path),
        "edge_table_path": str(edge_table_path),
        "figure_path": str(figure_path),
        "graph_figure_path": str(graph_figure_path),
        "report_figure_path": str(report_figure_path),
        "sine_readout_figure_path": str(sine_readout_figure_path) if sine_readout_figure_path else None,
        "proxy_y_figure_path": str(proxy_y_figure_path) if proxy_y_figure_path else None,
        "alpha_sweep_figure_path": str(alpha_sweep_figure_path) if alpha_sweep_figure_path else None,
        "alpha_neural_granger_figure_path": (
            str(alpha_neural_granger_figure_path) if alpha_neural_granger_figure_path else None
        ),
        "beta_sweep_figure_path": str(beta_sweep_figure_path) if beta_sweep_figure_path else None,
        "beta_validation_figure_path": str(beta_validation_figure_path) if beta_validation_figure_path else None,
        "lagged_proxy_figure_path": str(lagged_proxy_figure_path),
        "report_markdown_path": str(report_markdown_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true", help="Run the quick smoke grid.")
    group.add_argument("--full", action="store_true", help="Run the full comparison grid.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = "full" if args.full else "smoke"
    output = run_comparison_grid(mode=mode, result_dir=args.result_dir, figure_dir=args.figure_dir)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
