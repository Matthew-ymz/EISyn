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
DEFAULT_REPORT_PATH = ROOT / "docs" / "granger_peid_mlp_comparison.md"
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
        data[0, 0] = rng.normal(0.0, 0.4)
        data[0, 1] = rng.normal(0.0, 0.4)
        data[0, 2] = rng.normal(0.0, 0.2)
        data[0, 3] = rng.normal(0.0, 0.8)
        for t in range(n - 1):
            data[t + 1, 3] = 0.78 * data[t, 3] + rng.normal(0.0, 0.45)
            data[t + 1, 0] = (
                0.42 * data[t, 0]
                + 0.82 * data[t, 3]
                + rng.normal(0.0, 0.25)
            )
            data[t + 1, 1] = (
                0.38 * data[t, 1]
                + 0.76 * data[t, 3]
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
            synergy = max(0.0, float(joint_ei - single_a - single_b))
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
    group_lasso: float,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    """Fit one cMLP target model and read first-layer source-group norms."""

    import torch

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets[:, target_idx : target_idx + 1], dtype=np.float32)
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
            torch.linalg.vector_norm(first_layer[:, [lag_idx * 3 + source_idx for lag_idx in range(max_lag)]])
            for source_idx in range(3)
        )
        loss = mse + float(group_lasso) * penalty
        loss.backward()
        optimizer.step()
        loss_value = float(mse.detach().item())

    first_layer_weights = np.asarray(net[0].weight.detach().tolist(), dtype=float)
    group_norms: dict[str, float] = {}
    lag_norms: dict[str, list[float]] = {}
    for source_idx, source in enumerate(("w", "x", "y")):
        cols = [lag_idx * 3 + source_idx for lag_idx in range(max_lag)]
        group_norms[source] = float(np.linalg.norm(first_layer_weights[:, cols]))
        lag_norms[source] = [
            float(np.linalg.norm(first_layer_weights[:, lag_idx * 3 + source_idx]))
            for lag_idx in range(max_lag)
        ]
    return {
        "group_norms": group_norms,
        "lag_norms": lag_norms,
        "scaled_mse": loss_value,
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
        for target_idx, target in enumerate(("w", "x", "y")):
            fit = _fit_componentwise_neural_granger(
                features,
                targets,
                target_idx=target_idx,
                max_lag=int(max_lag),
                group_lasso=float(group_lasso),
                hidden_dim=int(hidden_dim),
                epochs=int(epochs),
                learning_rate=float(learning_rate),
                seed=int(model_seed + 17 * max_lag + target_idx),
            )
            group_norms = dict(fit["group_norms"])
            lag_norms = dict(fit["lag_norms"])
            ordered_sources = sorted(group_norms, key=lambda name: group_norms[name], reverse=True)
            for rank, source in enumerate(ordered_sources, start=1):
                source_lag_norms = [float(value) for value in lag_norms[source]]
                strongest_lag_idx = int(np.argmax(source_lag_norms)) + 1
                rows.append(
                    {
                        "max_lag": int(max_lag),
                        "target": target,
                        "source": source,
                        "rank": int(rank),
                        "group_norm": float(group_norms[source]),
                        "strongest_lag": int(strongest_lag_idx),
                        "strongest_lag_norm": float(max(source_lag_norms)),
                        "lag_norms": source_lag_norms,
                        "scaled_mse": float(fit["scaled_mse"]),
                    }
                )

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
) -> list[dict[str, float]]:
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
        peid = estimate_peid_graph(model, series, config)
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
                "peid_xy_joint_ei": float(peid_xy_z["joint_ei"]),
                "peid_xy_synergy": float(peid_xy_z["synergy"]),
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
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0), constrained_layout=True)

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
    ax.set_xlabel("alpha in alpha * sin(x y)")
    ax.set_ylabel("mean |SHAP value|")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax = axes[1]
    ax.plot(
        frame["alpha"],
        frame["shap_xy_mean_abs_interaction"],
        marker="o",
        color="#4f7ca8",
        linewidth=1.8,
        label="SHAP interaction |x:y|",
    )
    ax.plot(
        frame["alpha"],
        frame["product_xy_incremental_r2"],
        marker="s",
        color="#7b6aa8",
        linewidth=1.5,
        label="product probe incremental R2",
    )
    ax.set_xlabel("alpha in alpha * sin(x y)")
    ax.set_ylabel("SHAP/probe scale")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax = axes[2]
    ax.plot(
        frame["alpha"],
        frame["peid_xy_joint_ei"],
        marker="o",
        color="#5f8f6b",
        linewidth=1.8,
        label="PEID joint EI",
    )
    ax.plot(
        frame["alpha"],
        frame["peid_xy_synergy"],
        marker="s",
        color="#2f6f4e",
        linewidth=1.8,
        label="PEID synergy",
    )
    ax.plot(
        frame["alpha"],
        frame["peid_x_to_z"],
        marker="^",
        color="#9bb7d4",
        linewidth=1.2,
        label="PEID x->z",
    )
    ax.plot(
        frame["alpha"],
        frame["peid_y_to_z"],
        marker="v",
        color="#c4a07a",
        linewidth=1.2,
        label="PEID y->z",
    )
    ax.set_xlabel("alpha in alpha * sin(x y)")
    ax.set_ylabel("bits")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    path = figure_dir / "sine_alpha_shap_peid_sweep.png"
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
    alpha_sweep_figure_path: Path | None,
    alpha_sweep_rows: list[dict[str, float]],
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
    alpha_sweep_rel = (
        _relative_markdown_path(alpha_sweep_figure_path, report_path)
        if alpha_sweep_figure_path is not None
        else ""
    )
    alpha_sweep_block = ""
    if alpha_sweep_rel:
        alpha_table_lines = [
            "| alpha | SHAP `x->z` | SHAP `y->z` | SHAP `w->z` | SHAP interaction `|x:y|` | product probe incremental `R^2` | PEID joint EI `{x,y}->z` | PEID synergy `{x,y}->z` |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in alpha_sweep_rows:
            alpha_table_lines.append(
                "| {alpha:.2f} | {shap_x:.4g} | {shap_y:.4g} | {shap_w:.4g} | {shap_interaction:.4g} | {r2:.4g} | {joint:.4g} | {syn:.4g} |".format(
                    alpha=float(row["alpha"]),
                    shap_x=float(row["shap_x_to_z_mean_abs"]),
                    shap_y=float(row["shap_y_to_z_mean_abs"]),
                    shap_w=float(row["shap_w_to_z_mean_abs"]),
                    shap_interaction=float(row["shap_xy_mean_abs_interaction"]),
                    r2=float(row["product_xy_incremental_r2"]),
                    joint=float(row["peid_xy_joint_ei"]),
                    syn=float(row["peid_xy_synergy"]),
                )
            )
        alpha_sweep_block = (
            "## alpha 扫描：SHAP 交互与 PEID 协同\n\n"
            f"![alpha 扫描下的 SHAP 与 PEID 对照]({alpha_sweep_rel})\n\n"
            + "\n".join(alpha_table_lines)
            + "\n\n"
            "这里的 `alpha` 是 sine 项前面的强度系数。`alpha=0` 时，`z` 只剩自身记忆与噪声，"
            "SHAP 二阶交互和产品项增量解释度接近零；PEID 仍保留少量估计底噪和分箱残差。随着 `alpha` 增大，"
            "SHAP 单源 `x->z`、`y->z` 与 SHAP interaction 同时上升，但单源项是对协同响应的归因分摊，不是结构边；"
            "SHAP interaction 与产品项 probe 反映的是 fitted MLP 响应面的二阶非加性形状；"
            "PEID joint EI 与 synergy 反映的是在最大熵联合干预下 `{x,y}` 对目标分布施加的机制信息约束。"
            "\n\n"
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

同一条模拟时间序列先用于训练一个 MLP 一步转移模型，输入为 `[x_t, y_t, z_t, w_t]`，输出为 `[x_{{t+1}}, y_{{t+1}}, z_{{t+1}}, w_{{t+1}}]`。随后在固定 MLP 上读出四类量：

- Granger/ablation：把某个 source 的输入列替换为均值，记录目标预测 MSE 的增量。它回答“去掉这个变量会不会损害预测”。
- SHAP 类归因：在同一 fitted MLP 上做 interventional Shapley 读出，用经验背景替换未给定特征。单特征 SHAP 报告 mean absolute attribution；二阶 SHAP interaction 报告 `x:y` 的 mean absolute interaction。前者回答“某个特征分到多少预测贡献”，后者回答“两个特征的非加性预测贡献有多大”。
- 交互项 probe：在同一 fitted MLP 的最大熵干预预测面上，用标准化主效应加一个二阶乘积项拟合目标输出，并记录该乘积项相对于主效应模型的 incremental `R^2`。它回答“固定这个预测器时，响应面是否含有可由 `x:y` 近似的二阶非加性形状”。
- PEID：先做最大熵独立干预，再计算 single-source EI、joint EI 和 synergy：

$$
\\mathrm{{Syn}}(\\{{x,y\\}}\\to z)
= \\max\\left(0,\\; EI(\\{{x,y\\}}\\to z)-EI(x\\to z)-EI(y\\to z)\\right).
$$

## 代表性结果

{sine_system_table}

{sine_readout_block}图中左侧热图把 Granger、SHAP 和 PEID 的单源读出放在同一组边上比较。因为三行的单位不同，颜色只在每一行内部归一化，格子里的数字才是原始读数。右上角显示标准化乘积项对 `z` 的增量解释度：`x:y` 明显高于 `w:x` 与 `w:y`。右下角显示 PEID 对 `z` 的信息分解，联合 EI 与 synergy 高于单源 EI。

{alpha_sweep_block}

## 解释

`w -> x` 和 `w -> y` 在 Granger/ablation 与 PEID pairwise EI 中都很强，说明 fitted MLP 学到了共同驱动结构。`w -> z` 很小，符合结构方程中 `w` 不直接进入 `z` 的设定；若某些归因方法给出非零 `w -> z`，应解释为 `w` 通过诱导 `x,y` 相关性形成的代理贡献，而不是直接结构边。

对 `z` 来说，Granger/ablation 会给出明显的 `x -> z` 与 `y -> z`，SHAP 类单特征归因也会倾向把 sine 项拆成单变量贡献。交互项 probe 则能进一步指出 fitted MLP 的响应面中确实存在强 `x:y` 二阶非加性项，因此它比纯单特征 SHAP 更接近“有交互”的诊断；但它仍然是响应面形状分析，不是源侧最大熵干预语义下的机制信息分解。这些读出有预测解释价值，但它们把

$$
\\alpha\\sin(x_t y_t)
$$

投影成了 pairwise 贡献或低阶乘积项，不能单独表达“只有联合给定 `x_t` 和 `y_t` 时才稳定确定目标响应”的机制事实。

PEID 的关键读数是 `EI({{x, y}} -> z)` 与 `Syn({{x, y}} -> z)` 均显著高于单源投影。它说明联合干预 `{{x,y}}` 后，目标分布的约束远超过两个单源 EI 的加和。因此这个例子的结论不是“PEID 消除了所有代理效应”，而是：在同一个 learned transition surrogate 上，PEID 可以同时保留 `w -> x,y` 的共同驱动边，以及 `{{x,y}} -> z` 的协同超边；Granger 和 SHAP 单特征方法主要给出预测贡献的 pairwise 投影，交互项 probe 可以提示 `x:y` 非加性存在，但 PEID 才把这个非加性读成源集合到目标的协同有效信息。
"""
    report_path.write_text(text, encoding="utf-8")
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
        edge_rows.extend(_edge_records(run_id, config, granger_edges, peid, shap_readout))

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
    alpha_sweep_rows = (
        run_sine_alpha_sweep()
        if "common_driver_sine_synergy" in set(mechanisms)
        else []
    )
    alpha_sweep_figure_path = _plot_sine_alpha_sweep(alpha_sweep_rows, figure_dir)
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
        alpha_sweep_figure_path=alpha_sweep_figure_path,
        alpha_sweep_rows=alpha_sweep_rows,
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
        "alpha_sweep_figure_path": str(alpha_sweep_figure_path) if alpha_sweep_figure_path else None,
        "lagged_proxy_figure_path": str(lagged_proxy_figure_path),
        "report_markdown_path": str(report_markdown_path),
        "edge_table_path": str(edge_table_path),
        "sine_alpha_sweep": alpha_sweep_rows,
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
        "alpha_sweep_figure_path": str(alpha_sweep_figure_path) if alpha_sweep_figure_path else None,
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
