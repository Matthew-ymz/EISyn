#!/usr/bin/env python3
"""MLP prediction and matched PEID for two coupled standard maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import sys
import warnings

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATE_NAMES = ("q1", "p1", "q2", "p2")
TARGET_NAMES = ("I1", "I2")
TRUE_PAIR = "q1+q2"
DEFAULT_RESULT_DIR = ROOT / "results" / "coupled_standard_map_peid"
DEFAULT_FIGURE_DIR = ROOT / "fig" / "coupled_standard_map_peid"
DEFAULT_REPORT_PATH = ROOT / "docs" / "reports" / "coupled_standard_map_peid.md"


@dataclass(frozen=True)
class StandardMapConfig:
    k: float = 1.5
    coupling: float = 0.8
    noise_std: float = 0.05


@dataclass(frozen=True)
class DatasetSplit:
    states: np.ndarray
    impulses: np.ndarray
    next_states: np.ndarray
    trajectory_ids: np.ndarray


@dataclass(frozen=True)
class TrajectoryDataset:
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit


@dataclass
class FittedImpulseMLP:
    net: object
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    residual_std: np.ndarray
    best_epoch: int

    def predict_mean(self, states: np.ndarray) -> np.ndarray:
        import torch

        features = periodic_features(states).astype(np.float32)
        scaled = (features - self.x_mean) / self.x_std
        self.net.eval()
        with torch.no_grad():
            prediction = np.asarray(
                self.net(torch.tensor(scaled.tolist(), dtype=torch.float32)).cpu().tolist(),
                dtype=float,
            )
        return prediction * self.y_std + self.y_mean


def wrap_angle(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return (array + np.pi) % (2.0 * np.pi) - np.pi


def periodic_features(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("states must have shape (n, 4).")
    return np.concatenate([np.sin(values), np.cos(values)], axis=1)


def coupled_impulses(
    states: np.ndarray,
    *,
    config: StandardMapConfig,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("states must have shape (n, 4).")
    q1, _, q2, _ = values.T
    coupling_12 = float(config.coupling) * np.sin(q2 - q1)
    means = np.column_stack(
        [
            float(config.k) * np.sin(q1) + coupling_12,
            float(config.k) * np.sin(q2) - coupling_12,
        ]
    )
    if noise is None:
        return means
    errors = np.asarray(noise, dtype=float)
    if errors.shape != means.shape:
        raise ValueError("noise must match the impulse shape.")
    return means + errors


def transition_from_impulses(states: np.ndarray, impulses: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    kicks = np.asarray(impulses, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or kicks.shape != (len(values), 2):
        raise ValueError("states and impulses have incompatible shapes.")
    q1, p1, q2, p2 = values.T
    p1_next = wrap_angle(p1 + kicks[:, 0])
    p2_next = wrap_angle(p2 + kicks[:, 1])
    q1_next = wrap_angle(q1 + p1_next)
    q2_next = wrap_angle(q2 + p2_next)
    return np.column_stack([q1_next, p1_next, q2_next, p2_next])


def analytic_interaction_strength(config: StandardMapConfig) -> float:
    return float(config.coupling) ** 2 / 2.0


def analytic_pairwise_strengths(config: StandardMapConfig) -> dict[str, float]:
    own_angle = (float(config.k) ** 2 + float(config.coupling) ** 2) / 2.0
    cross_angle = float(config.coupling) ** 2 / 2.0
    return {
        "q1->I1": own_angle,
        "q1->I2": cross_angle,
        "q2->I1": cross_angle,
        "q2->I2": own_angle,
        "p1->I1": 0.0,
        "p1->I2": 0.0,
        "p2->I1": 0.0,
        "p2->I2": 0.0,
    }


def _simulate_trajectory(
    *,
    config: StandardMapConfig,
    steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = rng.uniform(-np.pi, np.pi, size=(1, 4))
    states: list[np.ndarray] = []
    impulses: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    for _ in range(int(steps)):
        noise = rng.normal(0.0, float(config.noise_std), size=(1, 2))
        kick = coupled_impulses(state, config=config, noise=noise)
        next_state = transition_from_impulses(state, kick)
        states.append(state[0].copy())
        impulses.append(kick[0].copy())
        next_states.append(next_state[0].copy())
        state = next_state
    return np.asarray(states), np.asarray(impulses), np.asarray(next_states)


def _merge_trajectories(records: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]) -> DatasetSplit:
    return DatasetSplit(
        states=np.concatenate([row[1] for row in records], axis=0),
        impulses=np.concatenate([row[2] for row in records], axis=0),
        next_states=np.concatenate([row[3] for row in records], axis=0),
        trajectory_ids=np.concatenate(
            [np.full(len(row[1]), row[0], dtype=int) for row in records], axis=0
        ),
    )


def build_trajectory_dataset(
    config: StandardMapConfig,
    *,
    trajectory_count: int,
    steps_per_trajectory: int,
    seed: int,
) -> TrajectoryDataset:
    if trajectory_count < 5 or steps_per_trajectory < 2:
        raise ValueError("trajectory_count must be at least 5 and steps_per_trajectory at least 2.")
    rng = np.random.default_rng(int(seed))
    records = []
    for trajectory_id in range(int(trajectory_count)):
        states, impulses, next_states = _simulate_trajectory(
            config=config,
            steps=steps_per_trajectory,
            rng=rng,
        )
        records.append((trajectory_id, states, impulses, next_states))
    train_end = max(1, int(round(0.6 * trajectory_count)))
    validation_end = max(train_end + 1, int(round(0.8 * trajectory_count)))
    validation_end = min(validation_end, trajectory_count - 1)
    return TrajectoryDataset(
        train=_merge_trajectories(records[:train_end]),
        validation=_merge_trajectories(records[train_end:validation_end]),
        test=_merge_trajectories(records[validation_end:]),
    )


def _augment_with_uniform_states(
    split: DatasetSplit,
    *,
    config: StandardMapConfig,
    seed: int,
) -> DatasetSplit:
    rng = np.random.default_rng(int(seed))
    count = len(split.states)
    states = rng.uniform(-np.pi, np.pi, size=(count, 4))
    noise = rng.normal(0.0, float(config.noise_std), size=(count, 2))
    impulses = coupled_impulses(states, config=config, noise=noise)
    next_states = transition_from_impulses(states, impulses)
    return DatasetSplit(
        states=np.concatenate([split.states, states], axis=0),
        impulses=np.concatenate([split.impulses, impulses], axis=0),
        next_states=np.concatenate([split.next_states, next_states], axis=0),
        trajectory_ids=np.concatenate([split.trajectory_ids, np.full(count, -1, dtype=int)]),
    )


def fit_impulse_mlp(
    train: DatasetSplit,
    validation: DatasetSplit,
    *,
    seed: int,
    epochs: int,
    hidden_width: int,
) -> FittedImpulseMLP:
    import torch

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    x = periodic_features(train.states).astype(np.float32)
    y = np.asarray(train.impulses, dtype=np.float32)
    xv = periodic_features(validation.states).astype(np.float32)
    yv = np.asarray(validation.impulses, dtype=np.float32)
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = np.maximum(x.std(axis=0, keepdims=True), 1e-6)
    y_mean = y.mean(axis=0, keepdims=True)
    y_std = np.maximum(y.std(axis=0, keepdims=True), 1e-6)
    x = (x - x_mean) / x_std
    xv = (xv - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std
    yv_scaled = (yv - y_mean) / y_std
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*", category=UserWarning)
        net = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], int(hidden_width)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_width), int(hidden_width)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_width), 2),
        )
    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-6)
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y_scaled, dtype=torch.float32)
    xvt = torch.tensor(xv, dtype=torch.float32)
    yvt = torch.tensor(yv_scaled, dtype=torch.float32)
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    patience = 50
    stale = 0
    batch_size = min(512, len(xt))
    generator = torch.Generator().manual_seed(int(seed) + 11)
    for epoch in range(int(epochs)):
        net.train()
        permutation = torch.randperm(len(xt), generator=generator)
        for start in range(0, len(xt), batch_size):
            idx = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((net(xt[idx]) - yt[idx]) ** 2)
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            validation_loss = float(torch.mean((net(xvt) - yvt) ** 2))
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("MLP training did not produce a checkpoint.")
    net.load_state_dict(best_state)
    fitted = FittedImpulseMLP(net, x_mean, x_std, y_mean, y_std, np.zeros(2), best_epoch)
    residual = validation.impulses - fitted.predict_mean(validation.states)
    fitted.residual_std = np.maximum(residual.std(axis=0), 1e-4)
    return fitted


def _circular_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(wrap_angle(np.asarray(prediction) - np.asarray(target)))


def prediction_metrics(model: FittedImpulseMLP, split: DatasetSplit) -> dict[str, float]:
    prediction = model.predict_mean(split.states)
    error = prediction - split.impulses
    mse = float(np.mean(error**2))
    baseline = float(np.mean((split.impulses - split.impulses.mean(axis=0, keepdims=True)) ** 2))
    variance = float(np.mean((split.impulses - split.impulses.mean(axis=0, keepdims=True)) ** 2))
    scale = float(np.sqrt(np.mean(split.impulses**2)))
    reconstructed = transition_from_impulses(split.states, prediction)
    return {
        "mse": mse,
        "r2": float(1.0 - mse / max(variance, 1e-12)),
        "nrmse": float(np.sqrt(mse) / max(scale, 1e-12)),
        "baseline_mse": baseline,
        "circular_mae": float(np.mean(_circular_error(reconstructed, split.next_states))),
    }


def _discretize_equal_width(values: np.ndarray, *, bins: int, low: float, high: float) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    scaled = (np.clip(array, low, high) - low) / max(high - low, 1e-12)
    return np.minimum((scaled * bins).astype(int), bins - 1)


def _discretize_quantile(values: np.ndarray, *, bins: int) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    edges = np.unique(np.quantile(array, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) <= 2:
        return np.zeros(len(array), dtype=int)
    return np.digitize(array, edges[1:-1], right=False).astype(int)


def _entropy(codes: np.ndarray) -> float:
    _, counts = np.unique(np.asarray(codes), axis=0, return_counts=True)
    probabilities = counts.astype(float) / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _mi(source_codes: np.ndarray, target_codes: np.ndarray) -> float:
    source = np.asarray(source_codes)
    if source.ndim == 1:
        source = source[:, None]
    target = np.asarray(target_codes).reshape(-1, 1)
    return _entropy(source) + _entropy(target) - _entropy(np.concatenate([source, target], axis=1))


def _bias_corrected_mi(
    source_codes: np.ndarray,
    target_codes: np.ndarray,
    *,
    permutations: np.ndarray,
) -> float:
    observed = _mi(source_codes, target_codes)
    null = np.mean([_mi(source_codes, target_codes[index]) for index in permutations], dtype=float)
    return float(max(0.0, observed - null))


def _array_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()[:16]


def evaluate_peid(
    *,
    states: np.ndarray,
    targets: np.ndarray,
    bins: int,
    permutation_count: int,
    seed: int,
) -> dict[str, object]:
    values = np.asarray(states, dtype=float)
    outputs = np.asarray(targets, dtype=float)
    rng = np.random.default_rng(int(seed))
    permutations = np.asarray([rng.permutation(len(values)) for _ in range(int(permutation_count))])
    source_codes = [
        _discretize_equal_width(values[:, idx], bins=bins, low=-np.pi, high=np.pi)
        for idx in range(4)
    ]
    target_codes = [_discretize_quantile(outputs[:, idx], bins=bins) for idx in range(2)]
    single_rows = []
    single_lookup: dict[tuple[int, int], float] = {}
    for source_idx, source_name in enumerate(STATE_NAMES):
        for target_idx, target_name in enumerate(TARGET_NAMES):
            score = _bias_corrected_mi(
                source_codes[source_idx], target_codes[target_idx], permutations=permutations
            )
            single_lookup[(source_idx, target_idx)] = score
            single_rows.append({"source": source_name, "target": target_name, "ei": score})
    hyperedges = []
    for left_idx, right_idx in combinations(range(4), 2):
        pair_name = "+".join(sorted((STATE_NAMES[left_idx], STATE_NAMES[right_idx])))
        joint_codes = np.column_stack([source_codes[left_idx], source_codes[right_idx]])
        for target_idx, target_name in enumerate(TARGET_NAMES):
            joint = _bias_corrected_mi(joint_codes, target_codes[target_idx], permutations=permutations)
            left = single_lookup[(left_idx, target_idx)]
            right = single_lookup[(right_idx, target_idx)]
            hyperedges.append(
                {
                    "sources": pair_name,
                    "target": target_name,
                    "joint_ei": joint,
                    "left_ei": left,
                    "right_ei": right,
                    "syn": float(joint - left - right),
                }
            )
    return {
        "single_sources": single_rows,
        "hyperedges": hyperedges,
        "intervention_digest": _array_digest(values),
        "bins": int(bins),
        "permutation_count": int(permutation_count),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = _rank(np.asarray(left, dtype=float))
    b = _rank(np.asarray(right, dtype=float))
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compare_peid(oracle: dict[str, object], learned: dict[str, object]) -> dict[str, object]:
    oracle_rows = {(row["sources"], row["target"]): row for row in oracle["hyperedges"]}
    learned_rows = {(row["sources"], row["target"]): row for row in learned["hyperedges"]}
    keys = sorted(oracle_rows)
    oracle_values = np.asarray([oracle_rows[key]["syn"] for key in keys], dtype=float)
    learned_values = np.asarray([learned_rows[key]["syn"] for key in keys], dtype=float)
    true_errors = []
    true_pair_top = []
    for target in TARGET_NAMES:
        true_key = (TRUE_PAIR, target)
        oracle_value = float(oracle_rows[true_key]["syn"])
        learned_value = float(learned_rows[true_key]["syn"])
        true_errors.append(abs(learned_value - oracle_value) / max(abs(oracle_value), 1e-12))
        target_rows = [row for row in learned["hyperedges"] if row["target"] == target]
        strongest = max(target_rows, key=lambda row: float(row["syn"]))
        true_pair_top.append(strongest["sources"] == TRUE_PAIR)
    momentum_ei = [
        float(row["ei"])
        for row in learned["single_sources"]
        if str(row["source"]).startswith("p")
    ]
    return {
        "spearman": _spearman(oracle_values, learned_values),
        "true_pair_top_both": bool(all(true_pair_top)),
        "true_pair_max_relative_error": float(max(true_errors)),
        "momentum_max_ei": float(max(momentum_ei)),
    }


def evaluate_trajectory_gate(metrics: dict[str, object]) -> dict[str, object]:
    checks = {
        "r2": float(metrics["r2"]) >= 0.99,
        "nrmse": float(metrics["nrmse"]) <= 0.05,
        "circular_mae": float(metrics["circular_mae"]) <= 0.05,
        "spearman": float(metrics["spearman"]) >= 0.9,
        "true_pair_top_both": bool(metrics["true_pair_top_both"]),
        "true_pair_max_relative_error": float(metrics["true_pair_max_relative_error"]) <= 0.20,
        "momentum_max_ei": float(metrics["momentum_max_ei"]) <= 0.02,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def _evaluate_model(
    model: FittedImpulseMLP,
    *,
    test: DatasetSplit,
    intervention_states: np.ndarray,
    intervention_noise: np.ndarray,
    bins: int,
    permutation_count: int,
    seed: int,
) -> dict[str, object]:
    means = model.predict_mean(intervention_states)
    targets = means + intervention_noise * model.residual_std.reshape(1, -1)
    peid = evaluate_peid(
        states=intervention_states,
        targets=targets,
        bins=bins,
        permutation_count=permutation_count,
        seed=seed,
    )
    return {
        "prediction": prediction_metrics(model, test),
        "residual_std": model.residual_std.tolist(),
        "best_epoch": int(model.best_epoch),
        "peid": peid,
    }


def _plot_summary(summary: dict[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    regimes = ["trajectory"] + (["mixed"] if summary["mixed_ran"] else [])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    prediction_labels = ["R2", "1-NRMSE", "1-circ.MAE"]
    x = np.arange(len(prediction_labels))
    width = 0.34
    for index, regime in enumerate(regimes):
        metrics = summary[regime]["prediction"]
        values = [metrics["r2"], 1.0 - metrics["nrmse"], 1.0 - metrics["circular_mae"]]
        axes[0].bar(x + (index - (len(regimes) - 1) / 2) * width, values, width, label=regime)
    axes[0].set_xticks(x, prediction_labels)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Prediction quality")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    oracle_rows = summary["oracle"]["hyperedges"]
    keys = [(row["sources"], row["target"]) for row in oracle_rows]
    oracle_values = [row["syn"] for row in oracle_rows]
    axes[1].scatter(oracle_values, [summary["trajectory"]["peid"]["hyperedges"][keys.index(key)]["syn"] for key in keys], label="trajectory")
    if summary["mixed_ran"]:
        axes[1].scatter(oracle_values, [summary["mixed"]["peid"]["hyperedges"][keys.index(key)]["syn"] for key in keys], label="mixed")
    limit = max(max(np.abs(oracle_values)), 0.1)
    axes[1].plot([-limit, limit], [-limit, limit], color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Oracle synergy (bit)", ylabel="MLP synergy (bit)", title="Matched PEID")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    labels = ["I1", "I2"]
    oracle_true = [next(row["syn"] for row in oracle_rows if row["sources"] == TRUE_PAIR and row["target"] == target) for target in labels]
    trajectory_true = [next(row["syn"] for row in summary["trajectory"]["peid"]["hyperedges"] if row["sources"] == TRUE_PAIR and row["target"] == target) for target in labels]
    positions = np.arange(2)
    axes[2].bar(positions - 0.18, oracle_true, 0.34, label="Oracle")
    axes[2].bar(positions + 0.18, trajectory_true, 0.34, label="Trajectory-MLP")
    if summary["mixed_ran"]:
        mixed_true = [next(row["syn"] for row in summary["mixed"]["peid"]["hyperedges"] if row["sources"] == TRUE_PAIR and row["target"] == target) for target in labels]
        axes[2].plot(positions, mixed_true, marker="o", color="#D55E00", label="Mixed-MLP")
    axes[2].set_xticks(positions, labels)
    axes[2].set_ylabel("Synergy (bit)")
    axes[2].set_title("True structural pair q1+q2")
    axes[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(summary: dict[str, object], path: Path, figure_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trajectory = summary["trajectory"]
    gate = summary["trajectory_gate"]
    relative_figure = os.path.relpath(figure_path, path.parent)
    lines = [
        "# Coupled Standard Map MLP+PEID",
        "",
        f"![Coupled standard map summary]({relative_figure})",
        "",
        "## Protocol",
        "",
        f"- K: `{summary['config']['k']}`",
        f"- coupling J: `{summary['config']['coupling']}`",
        f"- impulse noise: `{summary['config']['noise_std']}`",
        f"- analytic structural interaction J^2/2: `{summary['analytic_interaction_strength']:.6f}`",
        f"- finite-resolution PEID bins: `{summary['protocol']['bins']}`",
        "",
        "## Trajectory-MLP",
        "",
        f"- test R2: `{trajectory['prediction']['r2']:.6f}`",
        f"- test NRMSE: `{trajectory['prediction']['nrmse']:.6f}`",
        f"- next-state circular MAE: `{trajectory['prediction']['circular_mae']:.6f}` rad",
        f"- PEID Spearman: `{trajectory['comparison']['spearman']:.6f}`",
        f"- true-pair maximum relative error: `{trajectory['comparison']['true_pair_max_relative_error']:.6f}`",
        f"- maximum momentum-source EI: `{trajectory['comparison']['momentum_max_ei']:.6f}` bit",
        f"- preregistered gate passed: `{gate['passed']}`",
        f"- failed checks: `{', '.join(gate['failed_checks']) or 'none'}`",
        "",
        "## Causal Strengths",
        "",
        "### Analytic squared-Jacobian ground truth",
        "",
        "| relation | strength |",
        "| --- | ---: |",
    ]
    for relation, strength in summary["analytic_pairwise_strengths"].items():
        lines.append(f"| {relation} | {strength:.6f} |")
    lines.extend(
        [
        "",
        "### Single-source EI",
        "",
        "| source | target | Oracle EI | Trajectory-MLP EI |",
        "| --- | --- | ---: | ---: |",
        ]
    )
    oracle_singles = {
        (row["source"], row["target"]): row for row in summary["oracle"]["single_sources"]
    }
    trajectory_singles = {
        (row["source"], row["target"]): row for row in trajectory["peid"]["single_sources"]
    }
    for key in sorted(oracle_singles):
        lines.append(
            f"| {key[0]} | {key[1]} | {oracle_singles[key]['ei']:.6f} | "
            f"{trajectory_singles[key]['ei']:.6f} |"
        )
    lines.extend(
        [
            "",
            "### Two-source PEID residual",
            "",
            "| sources | target | structural truth | Oracle Syn | Trajectory-MLP Syn |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    oracle_hyperedges = {
        (row["sources"], row["target"]): row for row in summary["oracle"]["hyperedges"]
    }
    trajectory_hyperedges = {
        (row["sources"], row["target"]): row for row in trajectory["peid"]["hyperedges"]
    }
    for key in sorted(oracle_hyperedges):
        truth = "true" if key[0] == TRUE_PAIR else "null"
        lines.append(
            f"| {key[0]} | {key[1]} | {truth} | {oracle_hyperedges[key]['syn']:.6f} | "
            f"{trajectory_hyperedges[key]['syn']:.6f} |"
        )
    lines.extend(
        [
        "",
        "## Conditional Mixed-MLP",
        "",
        f"Mixed-MLP executed: `{summary['mixed_ran']}`.",
        ]
    )
    if summary["mixed_ran"]:
        mixed = summary["mixed"]
        lines.extend(
            [
                "",
                f"- test R2: `{mixed['prediction']['r2']:.6f}`",
                f"- test NRMSE: `{mixed['prediction']['nrmse']:.6f}`",
                f"- next-state circular MAE: `{mixed['prediction']['circular_mae']:.6f}` rad",
                f"- PEID Spearman: `{mixed['comparison']['spearman']:.6f}`",
                f"- true-pair maximum relative error: `{mixed['comparison']['true_pair_max_relative_error']:.6f}`",
                f"- maximum momentum-source EI: `{mixed['comparison']['momentum_max_ei']:.6f}` bit",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "The analytic mixed derivative is the structural ground truth. Oracle PEID is the numerical information-theoretic ground truth under the stated uniform intervention, noise level, discretization, and permutation-bias correction. A positive PEID residual is not itself proof of an explicit product term.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    *,
    mode: str,
    seed: int,
    result_dir: Path = DEFAULT_RESULT_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    config = StandardMapConfig()
    if mode == "smoke":
        trajectory_count, steps, epochs, width = 8, 80, 80, 32
        intervention_samples, bins, permutation_count = 2500, 8, 2
    else:
        trajectory_count, steps, epochs, width = 16, 2500, 400, 96
        intervention_samples, bins, permutation_count = 50000, 12, 5
    dataset = build_trajectory_dataset(
        config,
        trajectory_count=trajectory_count,
        steps_per_trajectory=steps,
        seed=seed,
    )
    trajectory_model = fit_impulse_mlp(
        dataset.train,
        dataset.validation,
        seed=seed,
        epochs=epochs,
        hidden_width=width,
    )
    rng = np.random.default_rng(int(seed) + 1000)
    intervention_states = rng.uniform(-np.pi, np.pi, size=(intervention_samples, 4))
    intervention_noise = rng.normal(size=(intervention_samples, 2))
    oracle_targets = coupled_impulses(
        intervention_states,
        config=config,
        noise=intervention_noise * float(config.noise_std),
    )
    oracle = evaluate_peid(
        states=intervention_states,
        targets=oracle_targets,
        bins=bins,
        permutation_count=permutation_count,
        seed=seed + 2000,
    )
    trajectory = _evaluate_model(
        trajectory_model,
        test=dataset.test,
        intervention_states=intervention_states,
        intervention_noise=intervention_noise,
        bins=bins,
        permutation_count=permutation_count,
        seed=seed + 2000,
    )
    trajectory["comparison"] = compare_peid(oracle, trajectory["peid"])
    gate_inputs = {**trajectory["prediction"], **trajectory["comparison"]}
    gate = evaluate_trajectory_gate(gate_inputs)
    mixed = None
    if not gate["passed"]:
        mixed_train = _augment_with_uniform_states(dataset.train, config=config, seed=seed + 3000)
        mixed_validation = _augment_with_uniform_states(dataset.validation, config=config, seed=seed + 4000)
        mixed_model = fit_impulse_mlp(
            mixed_train,
            mixed_validation,
            seed=seed + 1,
            epochs=epochs,
            hidden_width=width,
        )
        mixed = _evaluate_model(
            mixed_model,
            test=dataset.test,
            intervention_states=intervention_states,
            intervention_noise=intervention_noise,
            bins=bins,
            permutation_count=permutation_count,
            seed=seed + 2000,
        )
        mixed["comparison"] = compare_peid(oracle, mixed["peid"])
    result_dir = Path(result_dir)
    figure_dir = Path(figure_dir)
    report_path = Path(report_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "coupled_standard_map_peid.png"
    summary_path = result_dir / "summary.json"
    summary: dict[str, object] = {
        "config": config.__dict__,
        "protocol": {
            "mode": mode,
            "seed": int(seed),
            "trajectory_count": trajectory_count,
            "steps_per_trajectory": steps,
            "intervention_samples": intervention_samples,
            "bins": bins,
            "permutation_count": permutation_count,
        },
        "analytic_interaction_strength": analytic_interaction_strength(config),
        "analytic_pairwise_strengths": analytic_pairwise_strengths(config),
        "oracle": oracle,
        "trajectory": trajectory,
        "trajectory_gate": gate,
        "mixed_ran": mixed is not None,
        "mixed": mixed,
        "summary_path": str(summary_path),
        "figure_path": str(figure_path),
        "report_path": str(report_path),
    }
    _plot_summary(summary, figure_path)
    _write_report(summary, report_path, figure_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_experiment(mode=args.mode, seed=args.seed)
    print(json.dumps({"summary_path": result["summary_path"], "mixed_ran": result["mixed_ran"]}, indent=2))


if __name__ == "__main__":
    main()
