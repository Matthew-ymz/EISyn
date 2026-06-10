#!/usr/bin/env python3
"""Direct multihorizon MLP prediction and PEID for the Lorenz-3D system."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.TM.transport_map_density import estimate_mutual_information_transport_map


STATE_NAMES = ("x", "y", "z")
SOURCE_PAIRS = ((0, 1), (0, 2), (1, 2))
DEFAULT_RESULT_DIR = ROOT / "results" / "lorenz3d_multihorizon_peid"
DEFAULT_FIGURE_DIR = ROOT / "fig" / "lorenz3d_multihorizon_peid"
DEFAULT_REPORT_PATH = ROOT / "docs" / "reports" / "lorenz3d_multihorizon_mlp_peid.md"


@dataclass(frozen=True)
class LorenzConfig:
    rho: float
    dt: float = 0.01
    tau: float = 0.01
    sigma: float = 10.0
    beta: float = 8.0 / 3.0

    @property
    def integration_steps(self) -> int:
        steps = int(round(float(self.tau) / float(self.dt)))
        if steps <= 0 or not np.isclose(steps * self.dt, self.tau):
            raise ValueError("tau must be a positive integer multiple of dt.")
        return steps


@dataclass(frozen=True)
class DatasetSplit:
    inputs: np.ndarray
    targets: np.ndarray
    trajectory_ids: np.ndarray


@dataclass(frozen=True)
class NaturalDataset:
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit


@dataclass
class FittedDirectMLP:
    net: object
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    rho: float
    tau: float
    seed: int
    hidden_widths: tuple[int, ...]
    best_epoch: int
    metrics: dict[str, float]

    def predict(self, states: np.ndarray) -> np.ndarray:
        import torch

        values = np.asarray(states, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("states must have shape (n, 3).")
        scaled = (values - self.x_mean) / self.x_std
        self.net.eval()
        with torch.no_grad():
            output = np.asarray(
                self.net(torch.tensor(scaled.tolist(), dtype=torch.float32)).cpu().tolist(),
                dtype=float,
            )
        return output * self.y_std + self.y_mean


def lorenz_field(
    states: np.ndarray,
    *,
    rho: float,
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("states must have shape (n, 3).")
    x, y, z = values.T
    return np.column_stack(
        [
            float(sigma) * (y - x),
            x * (float(rho) - z) - y,
            x * y - float(beta) * z,
        ]
    )


def _rk4_step(states: np.ndarray, *, config: LorenzConfig) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    field = lambda rows: lorenz_field(
        rows,
        rho=config.rho,
        sigma=config.sigma,
        beta=config.beta,
    )
    k1 = field(values)
    k2 = field(values + 0.5 * config.dt * k1)
    k3 = field(values + 0.5 * config.dt * k2)
    k4 = field(values + config.dt * k3)
    return values + (config.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_transition(states: np.ndarray, *, config: LorenzConfig) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("states must have shape (n, 3).")
    result = values.copy()
    for _ in range(config.integration_steps):
        result = _rk4_step(result, config=config)
        if not np.isfinite(result).all():
            raise FloatingPointError("Lorenz transition diverged.")
    return result


def _initial_state_bank(count: int, *, seed: int) -> np.ndarray:
    if count < 1:
        raise ValueError("trajectory_count must be positive.")
    rng = np.random.default_rng(int(seed))
    return np.column_stack(
        [
            rng.uniform(-20.0, 20.0, size=count),
            rng.uniform(-20.0, 20.0, size=count),
            rng.uniform(0.0, 60.0, size=count),
        ]
    )


def _trajectory_pairs(
    initial_state: np.ndarray,
    *,
    config: LorenzConfig,
    burnin_steps: int,
    record_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    if burnin_steps < 0 or record_steps <= config.integration_steps:
        raise ValueError("record_steps must exceed the horizon and burnin_steps must be nonnegative.")
    state = np.asarray(initial_state, dtype=float).reshape(1, 3)
    one_step = LorenzConfig(
        rho=config.rho,
        dt=config.dt,
        tau=config.dt,
        sigma=config.sigma,
        beta=config.beta,
    )
    for _ in range(int(burnin_steps)):
        state = simulate_transition(state, config=one_step)
    rows = [state[0].copy()]
    for _ in range(int(record_steps)):
        state = simulate_transition(state, config=one_step)
        rows.append(state[0].copy())
    trajectory = np.asarray(rows)
    steps = config.integration_steps
    return trajectory[:-steps], trajectory[steps:]


def _subsample_split(
    inputs: np.ndarray,
    targets: np.ndarray,
    trajectory_ids: np.ndarray,
    *,
    limit: int,
    seed: int,
) -> DatasetSplit:
    if limit <= 0:
        raise ValueError("sample limits must be positive.")
    if len(inputs) > limit:
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(len(inputs), size=int(limit), replace=False))
        inputs = inputs[indices]
        targets = targets[indices]
        trajectory_ids = trajectory_ids[indices]
    return DatasetSplit(inputs=inputs, targets=targets, trajectory_ids=trajectory_ids)


def build_natural_dataset(
    *,
    rho: float,
    tau: float,
    dt: float = 0.01,
    trajectory_count: int = 12,
    burnin_steps: int = 5000,
    record_steps: int = 20000,
    max_samples: tuple[int, int, int] = (50000, 10000, 10000),
    seed: int = 0,
) -> NaturalDataset:
    if trajectory_count != 12:
        raise ValueError("the registered protocol requires exactly 12 trajectories.")
    config = LorenzConfig(rho=float(rho), dt=float(dt), tau=float(tau))
    states = _initial_state_bank(trajectory_count, seed=int(seed))
    split_rows: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for trajectory_id, initial_state in enumerate(states):
        inputs, targets = _trajectory_pairs(
            initial_state,
            config=config,
            burnin_steps=int(burnin_steps),
            record_steps=int(record_steps),
        )
        name = "train" if trajectory_id < 8 else "validation" if trajectory_id < 10 else "test"
        ids = np.full(len(inputs), trajectory_id, dtype=int)
        split_rows[name].append((inputs, targets, ids))

    results: dict[str, DatasetSplit] = {}
    for index, name in enumerate(("train", "validation", "test")):
        inputs = np.vstack([row[0] for row in split_rows[name]])
        targets = np.vstack([row[1] for row in split_rows[name]])
        ids = np.concatenate([row[2] for row in split_rows[name]])
        results[name] = _subsample_split(
            inputs,
            targets,
            ids,
            limit=int(max_samples[index]),
            seed=int(seed) + 100 + index,
        )
    return NaturalDataset(
        train=results["train"],
        validation=results["validation"],
        test=results["test"],
    )


def _prediction_metrics(
    expected: np.ndarray,
    predicted: np.ndarray,
    *,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    test_inputs: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(expected, dtype=float)
    prediction = np.asarray(predicted, dtype=float)
    mse = float(np.mean((truth - prediction) ** 2))
    scale = max(float(np.std(truth)), 1e-12)
    nrmse = float(np.sqrt(mse) / scale)
    denominator = max(float(np.sum((truth - truth.mean(axis=0, keepdims=True)) ** 2)), 1e-12)
    r2 = float(1.0 - np.sum((truth - prediction) ** 2) / denominator)
    correlations: list[float] = []
    for index in range(truth.shape[1]):
        if np.std(truth[:, index]) < 1e-12 or np.std(prediction[:, index]) < 1e-12:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(truth[:, index], prediction[:, index])[0, 1]))
    constant = np.repeat(train_targets.mean(axis=0, keepdims=True), len(truth), axis=0)
    constant_mse = max(float(np.mean((truth - constant) ** 2)), 1e-12)
    train_design = np.column_stack([np.ones(len(train_inputs)), train_inputs])
    coefficients, *_ = np.linalg.lstsq(train_design, train_targets, rcond=None)
    linear = np.column_stack([np.ones(len(test_inputs)), test_inputs]) @ coefficients
    linear_mse = max(float(np.mean((truth - linear) ** 2)), 1e-12)
    return {
        "mse": mse,
        "nrmse": nrmse,
        "r2": r2,
        "correlation": float(np.mean(correlations)),
        "constant_mse_ratio": float(mse / constant_mse),
        "linear_mse_ratio": float(mse / linear_mse),
    }


def fit_direct_mlp(
    dataset: NaturalDataset,
    *,
    rho: float,
    tau: float,
    seed: int,
    hidden_widths: tuple[int, ...] = (64, 64),
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 300,
    patience: int = 25,
    batch_size: int = 512,
) -> FittedDirectMLP:
    import torch

    if not hidden_widths or any(width <= 0 for width in hidden_widths):
        raise ValueError("hidden_widths must contain positive values.")
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    train_x = np.asarray(dataset.train.inputs, dtype=np.float32)
    train_y = np.asarray(dataset.train.targets, dtype=np.float32)
    validation_x = np.asarray(dataset.validation.inputs, dtype=np.float32)
    validation_y = np.asarray(dataset.validation.targets, dtype=np.float32)
    x_mean = train_x.mean(axis=0, keepdims=True)
    x_std = np.maximum(train_x.std(axis=0, keepdims=True), 1e-6)
    y_mean = train_y.mean(axis=0, keepdims=True)
    y_std = np.maximum(train_y.std(axis=0, keepdims=True), 1e-6)
    train_xn = (train_x - x_mean) / x_std
    train_yn = (train_y - y_mean) / y_std
    validation_xn = (validation_x - x_mean) / x_std
    validation_yn = (validation_y - y_mean) / y_std
    layers: list[object] = []
    input_width = 3
    for width in hidden_widths:
        layers.extend([torch.nn.Linear(input_width, int(width)), torch.nn.SiLU()])
        input_width = int(width)
    layers.append(torch.nn.Linear(input_width, 3))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*", category=UserWarning)
        net = torch.nn.Sequential(*layers)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    xt = torch.tensor(train_xn, dtype=torch.float32)
    yt = torch.tensor(train_yn, dtype=torch.float32)
    xv = torch.tensor(validation_xn, dtype=torch.float32)
    yv = torch.tensor(validation_yn, dtype=torch.float32)
    generator = torch.Generator().manual_seed(int(seed) + 17)
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(net.state_dict())
    stale_epochs = 0
    for epoch in range(int(epochs)):
        net.train()
        order = torch.randperm(len(xt), generator=generator)
        for start in range(0, len(order), int(batch_size)):
            indices = order[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((net(xt[indices]) - yt[indices]) ** 2)
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            validation_loss = float(torch.mean((net(xv) - yv) ** 2).item())
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(net.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break
    net.load_state_dict(best_state)
    model = FittedDirectMLP(
        net=net,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        rho=float(rho),
        tau=float(tau),
        seed=int(seed),
        hidden_widths=tuple(int(width) for width in hidden_widths),
        best_epoch=int(best_epoch),
        metrics={},
    )
    test_prediction = model.predict(dataset.test.inputs)
    model.metrics = _prediction_metrics(
        dataset.test.targets,
        test_prediction,
        train_inputs=dataset.train.inputs,
        train_targets=dataset.train.targets,
        test_inputs=dataset.test.inputs,
    )
    return model


def _transport_ei(source: np.ndarray, target: np.ndarray) -> float:
    summary = estimate_mutual_information_transport_map(source, target, degree=3)
    return float(summary["mi_hat"])


def summarize_two_source_synergy(
    left_source: np.ndarray,
    right_source: np.ndarray,
    target: np.ndarray,
    *,
    estimator: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> dict[str, float]:
    left = np.asarray(left_source, dtype=float)
    right = np.asarray(right_source, dtype=float)
    target_values = np.asarray(target, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or target_values.ndim != 2:
        raise ValueError("sources and target must be 2D arrays.")
    if left.shape[1] != 1 or right.shape[1] != 1:
        raise ValueError("each source must contain one column.")
    if not (len(left) == len(right) == len(target_values)):
        raise ValueError("sources and target must share the sample axis.")
    estimate = _transport_ei if estimator is None else estimator
    left_ei = float(estimate(left, target_values))
    right_ei = float(estimate(right, target_values))
    joint_ei = float(estimate(np.column_stack([left, right]), target_values))
    return {
        "left_ei": left_ei,
        "right_ei": right_ei,
        "joint_ei": joint_ei,
        "synergy": float(joint_ei - left_ei - right_ei),
    }


def _array_digest(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()[:16]


def _bootstrap_synergy_interval(
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    *,
    indices: list[np.ndarray],
) -> tuple[float, float]:
    if not indices:
        return float("nan"), float("nan")
    estimates = [
        summarize_two_source_synergy(left[index], right[index], target[index])["synergy"]
        for index in indices
    ]
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def _rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _recovery_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    oracle = np.asarray([row["oracle_synergy"] for row in rows], dtype=float)
    learned = np.asarray([row["mlp_synergy"] for row in rows], dtype=float)
    if np.std(oracle) < 1e-12 or np.std(learned) < 1e-12:
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(_rank_values(oracle), _rank_values(learned))[0, 1])
    oracle_top2 = set(np.argsort(oracle)[-2:].tolist())
    learned_top2 = set(np.argsort(learned)[-2:].tolist())
    overlap = len(oracle_top2 & learned_top2) / 2.0
    return {
        "spearman": spearman,
        "top2_precision": float(overlap),
        "top2_recall": float(overlap),
        "synergy_mae": float(np.mean(np.abs(oracle - learned))),
    }


def _learned_response_comparisons(
    model: FittedDirectMLP,
    interventions: np.ndarray,
) -> dict[tuple[int, int, int], dict[str, float]]:
    values = np.asarray(interventions, dtype=float)
    baseline = values.mean(axis=0)
    full_prediction = model.predict(values)
    derivative_cache: dict[int, np.ndarray] = {}
    spans = np.maximum(np.ptp(values, axis=0), 1.0)
    for source_index in range(3):
        step = 1e-3 * spans[source_index]
        plus = values.copy()
        minus = values.copy()
        plus[:, source_index] += step
        minus[:, source_index] -= step
        derivative_cache[source_index] = (model.predict(plus) - model.predict(minus)) / (2.0 * step)
    result: dict[tuple[int, int, int], dict[str, float]] = {}
    for left_index, right_index in SOURCE_PAIRS:
        left_removed = values.copy()
        right_removed = values.copy()
        both_removed = values.copy()
        left_removed[:, left_index] = baseline[left_index]
        right_removed[:, right_index] = baseline[right_index]
        both_removed[:, [left_index, right_index]] = baseline[[left_index, right_index]]
        left_prediction = model.predict(left_removed)
        right_prediction = model.predict(right_removed)
        both_prediction = model.predict(both_removed)
        interaction = full_prediction - left_prediction - right_prediction + both_prediction
        ablated_prediction = both_prediction
        for target_index in range(3):
            result[(left_index, right_index, target_index)] = {
                "mlp_ablation_score": float(
                    np.mean((full_prediction[:, target_index] - ablated_prediction[:, target_index]) ** 2)
                ),
                "mlp_response_interaction": float(np.mean(np.abs(interaction[:, target_index]))),
                "mlp_jacobian_strength": float(
                    np.mean(
                        np.sqrt(
                            derivative_cache[left_index][:, target_index] ** 2
                            + derivative_cache[right_index][:, target_index] ** 2
                        )
                    )
                ),
            }
    return result


def evaluate_matched_peid(
    *,
    config: LorenzConfig,
    model: FittedDirectMLP,
    intervention_bounds: np.ndarray,
    samples: int,
    seed: int,
    bootstrap_replicates: int = 200,
) -> dict[str, object]:
    bounds = np.asarray(intervention_bounds, dtype=float)
    if bounds.shape != (3, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("intervention_bounds must have shape (3, 2) with increasing limits.")
    if samples < 20:
        raise ValueError("samples must be at least 20.")
    rng = np.random.default_rng(int(seed))
    interventions = rng.uniform(bounds[:, 0], bounds[:, 1], size=(int(samples), 3))
    oracle_targets = simulate_transition(interventions, config=config)
    mlp_targets = model.predict(interventions)
    response_comparisons = _learned_response_comparisons(model, interventions)
    digest = _array_digest(interventions)
    bootstrap_indices = [
        rng.integers(0, int(samples), size=int(samples))
        for _ in range(int(bootstrap_replicates))
    ]
    rows: list[dict[str, object]] = []
    for target_index, target_name in enumerate(STATE_NAMES):
        for left_index, right_index in SOURCE_PAIRS:
            left = interventions[:, [left_index]]
            right = interventions[:, [right_index]]
            oracle = summarize_two_source_synergy(
                left,
                right,
                oracle_targets[:, [target_index]],
            )
            learned = summarize_two_source_synergy(
                left,
                right,
                mlp_targets[:, [target_index]],
            )
            oracle_ci = _bootstrap_synergy_interval(
                left,
                right,
                oracle_targets[:, [target_index]],
                indices=bootstrap_indices,
            )
            learned_ci = _bootstrap_synergy_interval(
                left,
                right,
                mlp_targets[:, [target_index]],
                indices=bootstrap_indices,
            )
            row: dict[str, object] = {
                "sources": f"{STATE_NAMES[left_index]}+{STATE_NAMES[right_index]}",
                "target": f"{target_name}_tau",
                **response_comparisons[(left_index, right_index, target_index)],
            }
            for prefix, values, interval in (
                ("oracle", oracle, oracle_ci),
                ("mlp", learned, learned_ci),
            ):
                row.update({f"{prefix}_{key}": float(value) for key, value in values.items()})
                row[f"{prefix}_synergy_ci_low"] = interval[0]
                row[f"{prefix}_synergy_ci_high"] = interval[1]
            rows.append(row)
    result = {
        "rho": float(config.rho),
        "tau": float(config.tau),
        "intervention_bounds": bounds.tolist(),
        "intervention_sample_count": int(samples),
        "oracle_intervention_digest": digest,
        "mlp_intervention_digest": digest,
        "bootstrap_replicates": int(bootstrap_replicates),
        "rows": rows,
    }
    result["recovery"] = _recovery_summary(rows)
    return result


def evaluate_conditional_wing_peid(
    *,
    config: LorenzConfig,
    model: FittedDirectMLP,
    intervention_bounds: np.ndarray,
    samples: int,
    seed: int,
    bootstrap_replicates: int = 200,
) -> dict[str, dict[str, object]]:
    bounds = np.asarray(intervention_bounds, dtype=float)
    if bounds.shape != (3, 2) or not (bounds[0, 0] < 0.0 < bounds[0, 1]):
        raise ValueError("the global intervention box must span both Lorenz wings.")
    left_bounds = bounds.copy()
    right_bounds = bounds.copy()
    left_bounds[0, 1] = 0.0
    right_bounds[0, 0] = 0.0
    return {
        "left": evaluate_matched_peid(
            config=config,
            model=model,
            intervention_bounds=left_bounds,
            samples=int(samples),
            seed=int(seed),
            bootstrap_replicates=int(bootstrap_replicates),
        ),
        "right": evaluate_matched_peid(
            config=config,
            model=model,
            intervention_bounds=right_bounds,
            samples=int(samples),
            seed=int(seed) + 1,
            bootstrap_replicates=int(bootstrap_replicates),
        ),
    }


def compute_dynamics_diagnostics(
    split: DatasetSplit,
    *,
    config: LorenzConfig,
    lyapunov_samples: int = 32,
) -> dict[str, float]:
    values = np.asarray(split.inputs, dtype=float)
    state_variance = float(np.mean(np.var(values, axis=0)))
    switches = 0
    transitions = 0
    for trajectory_id in np.unique(split.trajectory_ids):
        trajectory = values[split.trajectory_ids == trajectory_id]
        if len(trajectory) < 2:
            continue
        signs = trajectory[:, 0] >= 0.0
        switches += int(np.count_nonzero(signs[1:] != signs[:-1]))
        transitions += len(signs) - 1
    wing_switch_rate = float(switches / max(transitions, 1))
    count = min(int(lyapunov_samples), len(values))
    indices = np.linspace(0, len(values) - 1, count, dtype=int)
    base = values[indices]
    epsilon = 1e-6
    direction = np.array([1.0, -1.0, 1.0], dtype=float)
    direction /= np.linalg.norm(direction)
    perturbed = base + epsilon * direction
    base_future = simulate_transition(base, config=config)
    perturbed_future = simulate_transition(perturbed, config=config)
    distances = np.linalg.norm(perturbed_future - base_future, axis=1)
    finite_time_lyapunov = float(np.mean(np.log(np.maximum(distances, 1e-15) / epsilon) / config.tau))
    return {
        "state_variance": state_variance,
        "wing_switch_rate": wing_switch_rate,
        "finite_time_lyapunov": finite_time_lyapunov,
    }


def _protocol_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _global_intervention_bounds(training_states: list[np.ndarray]) -> np.ndarray:
    values = np.vstack(training_states)
    low = np.quantile(values, 0.005, axis=0)
    high = np.quantile(values, 0.995, axis=0)
    width = np.maximum(high - low, 2.0)
    return np.column_stack([low - 0.1 * width, high + 0.1 * width])


def _job_label(rho: float, tau: float, seed: int) -> str:
    def token(value: float) -> str:
        return f"{value:g}".replace("-", "m").replace(".", "p")

    return f"rho_{token(rho)}_tau_{token(tau)}_seed_{seed}"


def _aggregate_grid(
    rows: list[dict[str, object]],
    *,
    rhos: tuple[float, ...],
    horizons: tuple[float, ...],
    field: str,
) -> np.ndarray:
    grid = np.full((len(rhos), len(horizons)), np.nan, dtype=float)
    for rho_index, rho in enumerate(rhos):
        for tau_index, tau in enumerate(horizons):
            values = [
                float(row[field])
                for row in rows
                if np.isclose(float(row["rho"]), rho) and np.isclose(float(row["tau"]), tau)
            ]
            if values:
                grid[rho_index, tau_index] = float(np.mean(values))
    return grid


def _save_heatmap(
    grid: np.ndarray,
    *,
    rhos: tuple[float, ...],
    horizons: tuple[float, ...],
    title: str,
    colorbar_label: str,
    path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6.5, 0.75 * len(horizons)), max(3.8, 0.28 * len(rhos))), constrained_layout=True)
    image = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(horizons)), [f"{value:g}" for value in horizons])
    ax.set_yticks(range(len(rhos)), [f"{value:g}" for value in rhos])
    ax.set_xlabel("Prediction horizon tau")
    ax.set_ylabel("rho")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _mechanism_anchor_rows(peid_runs: list[dict[str, object]]) -> list[dict[str, float]]:
    anchors = {("x+z", "y_tau"), ("x+y", "z_tau")}
    result: list[dict[str, float]] = []
    for run in peid_runs:
        rows = [row for row in run["rows"] if (row["sources"], row["target"]) in anchors]
        if not rows:
            continue
        result.append(
            {
                "rho": float(run["rho"]),
                "tau": float(run["tau"]),
                "seed": float(run["training_seed"]),
                "oracle_synergy": float(np.mean([row["oracle_synergy"] for row in rows])),
                "mlp_synergy": float(np.mean([row["mlp_synergy"] for row in rows])),
                "peid_mae": float(np.mean([abs(row["mlp_synergy"] - row["oracle_synergy"]) for row in rows])),
                "prediction_nrmse": float(run["prediction_nrmse"]),
            }
        )
    return result


def _save_error_scatter(anchor_rows: list[dict[str, float]], *, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    if anchor_rows:
        scatter = ax.scatter(
            [row["prediction_nrmse"] for row in anchor_rows],
            [row["peid_mae"] for row in anchor_rows],
            c=[row["rho"] for row in anchor_rows],
            s=55,
            cmap="plasma",
            alpha=0.85,
        )
        fig.colorbar(scatter, ax=ax, label="rho")
    ax.set_xlabel("Test NRMSE")
    ax.set_ylabel("Anchor synergy MAE")
    ax.set_title("Prediction accuracy versus PEID recovery")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_horizon_curves(anchor_rows: list[dict[str, float]], *, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    for rho in sorted({row["rho"] for row in anchor_rows}):
        selected = [row for row in anchor_rows if np.isclose(row["rho"], rho)]
        taus = sorted({row["tau"] for row in selected})
        oracle = [np.mean([row["oracle_synergy"] for row in selected if np.isclose(row["tau"], tau)]) for tau in taus]
        learned = [np.mean([row["mlp_synergy"] for row in selected if np.isclose(row["tau"], tau)]) for tau in taus]
        ax.plot(taus, oracle, marker="o", label=f"Oracle rho={rho:g}")
        ax.plot(taus, learned, marker="s", linestyle="--", label=f"MLP rho={rho:g}")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Prediction horizon tau")
    ax.set_ylabel("Mean anchor synergy (bits)")
    ax.set_title("Mechanism-anchor synergy across horizons")
    if anchor_rows:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    *,
    path: Path,
    protocol: dict[str, object],
    prediction_rows: list[dict[str, object]],
    anchor_rows: list[dict[str, float]],
    figure_paths: list[Path],
) -> None:
    best = min(prediction_rows, key=lambda row: float(row["nrmse"])) if prediction_rows else None
    lines = [
        "# Lorenz-3D 多步 MLP+PEID 实验",
        "",
        "本实验仅使用自然轨迹训练每个 horizon 的直接预测 MLP，并在独立均匀干预盒上比较 Oracle 与 MLP+PEID。",
        "",
        "## 协议",
        "",
        f"- rho: `{protocol['rhos']}`",
        f"- horizon: `{protocol['horizons']}`",
        f"- training seeds: `{protocol['training_seeds']}`",
        f"- intervention samples: `{protocol['peid_samples']}`",
        "- PEID 解释：确定性连续映射上的有限样本、有限分辨率估计，不解释为精确连续 EI。",
        "",
        "## 预测摘要",
        "",
    ]
    if best is not None:
        lines.append(
            f"最低测试 NRMSE 为 {float(best['nrmse']):.4f}，对应 rho={float(best['rho']):g}、tau={float(best['tau']):g}。"
        )
        lines.extend(["", "| rho | tau | NRMSE | R2 |", "| ---: | ---: | ---: | ---: |"])
        for rho in sorted({float(row["rho"]) for row in prediction_rows}):
            selected = [row for row in prediction_rows if np.isclose(float(row["rho"]), rho)]
            for row in sorted(selected, key=lambda item: float(item["tau"])):
                lines.append(
                    f"| {rho:g} | {float(row['tau']):g} | {float(row['nrmse']):.4f} | {float(row['r2']):.4f} |"
                )
    if anchor_rows:
        mean_error = float(np.mean([row["peid_mae"] for row in anchor_rows]))
        lines.extend(["", "## PEID 摘要", "", f"机制锚点的平均 Oracle--MLP 协同绝对误差为 {mean_error:.4f} bits。"])
    lines.extend(["", "## 图表", ""])
    for figure in figure_paths:
        relative = Path(os.path.relpath(figure, path.parent))
        lines.append(f"![{figure.stem}]({relative.as_posix()})")
        lines.append("")
    lines.extend(
        [
            "## 结论边界",
            "",
            "短 horizon 的 `{x,z}->y` 与 `{x,y}->z` 是机制锚点；长 horizon 的协同表示有限时间流映射的联合状态约束。预测误差低不自动意味着干预域中的机制恢复准确。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    *,
    rhos: tuple[float, ...],
    horizons: tuple[float, ...],
    training_seeds: tuple[int, ...],
    representative_rhos: tuple[float, ...],
    result_dir: str | Path = DEFAULT_RESULT_DIR,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    burnin_steps: int = 5000,
    record_steps: int = 20000,
    max_samples: tuple[int, int, int] = (50000, 10000, 10000),
    hidden_widths: tuple[int, ...] = (64, 64),
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 300,
    patience: int = 25,
    batch_size: int = 512,
    peid_samples: int = 5000,
    bootstrap_replicates: int = 200,
    enable_conditional_wings: bool = False,
    seed: int = 0,
    force: bool = False,
) -> dict[str, object]:
    if not rhos or not horizons or not training_seeds:
        raise ValueError("rhos, horizons, and training_seeds must be nonempty.")
    result_path = Path(result_dir)
    figure_path = Path(figure_dir)
    report = Path(report_path)
    result_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)
    protocol: dict[str, object] = {
        "rhos": [float(value) for value in rhos],
        "horizons": [float(value) for value in horizons],
        "training_seeds": [int(value) for value in training_seeds],
        "representative_rhos": [float(value) for value in representative_rhos],
        "dt": 0.01,
        "burnin_steps": int(burnin_steps),
        "record_steps": int(record_steps),
        "max_samples": list(max_samples),
        "hidden_widths": list(hidden_widths),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "epochs": int(epochs),
        "patience": int(patience),
        "batch_size": int(batch_size),
        "peid_samples": int(peid_samples),
        "bootstrap_replicates": int(bootstrap_replicates),
        "enable_conditional_wings": bool(enable_conditional_wings),
        "seed": int(seed),
        "estimator": "polynomial_triangular_transport_map_degree_3",
    }
    protocol_id = _protocol_hash(protocol)
    summary_path = result_path / "lorenz3d_multihorizon_summary.json"
    prediction_cache_path = result_path / "lorenz3d_prediction_cache.npz"
    if not force and summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        existing_figures = [Path(value) for value in existing.get("figure_paths", [])]
        if (
            existing.get("protocol_id") == protocol_id
            and prediction_cache_path.exists()
            and report.exists()
            and all(path.exists() for path in existing_figures)
        ):
            return {
                "cache_hit": True,
                "summary_path": str(summary_path),
                "prediction_cache_path": str(prediction_cache_path),
                "report_path": str(report),
                "figure_paths": [str(path) for path in existing_figures],
            }

    pilot_states: list[np.ndarray] = []
    shortest_horizon = min(horizons)
    for rho in rhos:
        pilot = build_natural_dataset(
            rho=float(rho),
            tau=float(shortest_horizon),
            burnin_steps=int(burnin_steps),
            record_steps=int(record_steps),
            max_samples=max_samples,
            seed=int(seed),
        )
        pilot_states.append(pilot.train.inputs)
    intervention_bounds = _global_intervention_bounds(pilot_states)

    prediction_rows: list[dict[str, object]] = []
    peid_runs: list[dict[str, object]] = []
    conditional_wing_runs: list[dict[str, object]] = []
    prediction_arrays: dict[str, np.ndarray] = {}
    for rho in rhos:
        for tau in horizons:
            dataset = build_natural_dataset(
                rho=float(rho),
                tau=float(tau),
                burnin_steps=int(burnin_steps),
                record_steps=int(record_steps),
                max_samples=max_samples,
                seed=int(seed),
            )
            diagnostics = compute_dynamics_diagnostics(
                dataset.train,
                config=LorenzConfig(rho=float(rho), tau=float(tau)),
            )
            for training_seed in training_seeds:
                model = fit_direct_mlp(
                    dataset,
                    rho=float(rho),
                    tau=float(tau),
                    seed=int(training_seed),
                    hidden_widths=hidden_widths,
                    learning_rate=float(learning_rate),
                    weight_decay=float(weight_decay),
                    epochs=int(epochs),
                    patience=int(patience),
                    batch_size=int(batch_size),
                )
                prediction = model.predict(dataset.test.inputs)
                row: dict[str, object] = {
                    "rho": float(rho),
                    "tau": float(tau),
                    "training_seed": int(training_seed),
                    "best_epoch": int(model.best_epoch),
                    **diagnostics,
                    **{key: float(value) for key, value in model.metrics.items()},
                }
                prediction_rows.append(row)
                label = _job_label(float(rho), float(tau), int(training_seed))
                prediction_arrays[f"{label}_expected"] = dataset.test.targets
                prediction_arrays[f"{label}_predicted"] = prediction
                if any(np.isclose(float(rho), value) for value in representative_rhos):
                    peid = evaluate_matched_peid(
                        config=LorenzConfig(rho=float(rho), tau=float(tau)),
                        model=model,
                        intervention_bounds=intervention_bounds,
                        samples=int(peid_samples),
                        seed=int(seed) + 10000 + int(training_seed),
                        bootstrap_replicates=int(bootstrap_replicates),
                    )
                    peid["training_seed"] = int(training_seed)
                    peid["prediction_nrmse"] = float(model.metrics["nrmse"])
                    peid_runs.append(peid)
                    if (
                        enable_conditional_wings
                        and any(np.isclose(float(rho), value) for value in (28.0, 45.0))
                        and float(model.metrics["linear_mse_ratio"]) < 1.0
                        and float(model.metrics["r2"]) >= 0.5
                    ):
                        wing_result = evaluate_conditional_wing_peid(
                            config=LorenzConfig(rho=float(rho), tau=float(tau)),
                            model=model,
                            intervention_bounds=intervention_bounds,
                            samples=int(peid_samples),
                            seed=int(seed) + 20000 + int(training_seed),
                            bootstrap_replicates=int(bootstrap_replicates),
                        )
                        conditional_wing_runs.append(
                            {
                                "rho": float(rho),
                                "tau": float(tau),
                                "training_seed": int(training_seed),
                                "wings": wing_result,
                            }
                        )
    np.savez_compressed(prediction_cache_path, **prediction_arrays)

    anchor_rows = _mechanism_anchor_rows(peid_runs)
    prediction_grid = _aggregate_grid(
        prediction_rows,
        rhos=rhos,
        horizons=horizons,
        field="nrmse",
    )
    anchor_grid = _aggregate_grid(
        anchor_rows,
        rhos=tuple(value for value in rhos if any(np.isclose(value, rep) for rep in representative_rhos)),
        horizons=horizons,
        field="mlp_synergy",
    )
    oracle_anchor_grid = _aggregate_grid(
        anchor_rows,
        rhos=tuple(value for value in rhos if any(np.isclose(value, rep) for rep in representative_rhos)),
        horizons=horizons,
        field="oracle_synergy",
    )
    representative_grid_rhos = tuple(
        value for value in rhos if any(np.isclose(value, rep) for rep in representative_rhos)
    )
    prediction_figure = figure_path / "lorenz3d_prediction_nrmse.png"
    synergy_figure = figure_path / "lorenz3d_mlp_anchor_synergy.png"
    oracle_synergy_figure = figure_path / "lorenz3d_oracle_anchor_synergy.png"
    scatter_figure = figure_path / "lorenz3d_prediction_vs_peid.png"
    curve_figure = figure_path / "lorenz3d_anchor_horizon_curves.png"
    _save_heatmap(
        prediction_grid,
        rhos=rhos,
        horizons=horizons,
        title="Direct MLP prediction error",
        colorbar_label="Test NRMSE",
        path=prediction_figure,
    )
    if representative_grid_rhos:
        _save_heatmap(
            anchor_grid,
            rhos=representative_grid_rhos,
            horizons=horizons,
            title="MLP mechanism-anchor synergy",
            colorbar_label="Mean synergy (bits)",
            path=synergy_figure,
        )
        _save_heatmap(
            oracle_anchor_grid,
            rhos=representative_grid_rhos,
            horizons=horizons,
            title="Oracle mechanism-anchor synergy",
            colorbar_label="Mean synergy (bits)",
            path=oracle_synergy_figure,
        )
    else:
        _save_heatmap(
            np.zeros((1, len(horizons))),
            rhos=(0.0,),
            horizons=horizons,
            title="No representative rho selected",
            colorbar_label="Mean synergy (bits)",
            path=synergy_figure,
        )
        _save_heatmap(
            np.zeros((1, len(horizons))),
            rhos=(0.0,),
            horizons=horizons,
            title="No representative rho selected",
            colorbar_label="Mean synergy (bits)",
            path=oracle_synergy_figure,
        )
    _save_error_scatter(anchor_rows, path=scatter_figure)
    _save_horizon_curves(anchor_rows, path=curve_figure)
    figure_paths = [
        prediction_figure,
        oracle_synergy_figure,
        synergy_figure,
        scatter_figure,
        curve_figure,
    ]
    _write_report(
        path=report,
        protocol=protocol,
        prediction_rows=prediction_rows,
        anchor_rows=anchor_rows,
        figure_paths=figure_paths,
    )
    payload = {
        "protocol_id": protocol_id,
        "protocol": protocol,
        "intervention_bounds": intervention_bounds.tolist(),
        "prediction_rows": prediction_rows,
        "peid_runs": peid_runs,
        "conditional_wing_runs": conditional_wing_runs,
        "anchor_rows": anchor_rows,
        "prediction_cache_path": str(prediction_cache_path),
        "figure_paths": [str(path) for path in figure_paths],
        "report_path": str(report),
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    return {
        "cache_hit": False,
        "summary_path": str(summary_path),
        "prediction_cache_path": str(prediction_cache_path),
        "report_path": str(report),
        "figure_paths": [str(path) for path in figure_paths],
    }


def _full_rho_grid() -> tuple[float, ...]:
    coarse = np.arange(0.0, 60.0 + 0.1, 2.0)
    dense = np.concatenate([np.array([0.5, 1.0, 1.5]), np.arange(22.0, 30.0 + 0.1, 0.5)])
    representatives = np.array([0.5, 5.0, 15.0, 22.0, 24.0, 24.5, 25.0, 28.0, 35.0, 45.0, 55.0])
    return tuple(float(value) for value in np.unique(np.concatenate([coarse, dense, representatives])))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    horizons = (0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
    if args.mode == "smoke":
        options = dict(
            rhos=(15.0, 28.0, 45.0),
            horizons=horizons,
            training_seeds=(0,),
            representative_rhos=(15.0, 28.0, 45.0),
            burnin_steps=200,
            record_steps=500,
            max_samples=(2400, 600, 600),
            hidden_widths=(32, 32),
            epochs=60,
            patience=10,
            batch_size=128,
            peid_samples=500,
            bootstrap_replicates=0,
            seed=0,
        )
    else:
        options = dict(
            rhos=_full_rho_grid(),
            horizons=horizons,
            training_seeds=(0, 1, 2, 3, 4),
            representative_rhos=(0.5, 5.0, 15.0, 22.0, 24.0, 24.5, 25.0, 28.0, 35.0, 45.0, 55.0),
            enable_conditional_wings=True,
            seed=0,
        )
    result = run_experiment(
        **options,
        result_dir=args.result_dir,
        figure_dir=args.figure_dir,
        report_path=args.report_path,
        force=args.force,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
