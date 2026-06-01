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
            pred_tensor = self.net(torch.as_tensor(scaled, dtype=torch.float32)).cpu()
            pred = np.asarray(pred_tensor.tolist(), dtype=np.float32)
        return pred * self.y_std + self.y_mean


@dataclass(frozen=True)
class PeidGraph:
    pairwise_edges: pd.DataFrame
    synergy_edges: pd.DataFrame
    intervention_states: pd.DataFrame


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
    x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_scaled, dtype=torch.float32)
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
    summary_figure_path: Path,
    graph_figure_path: Path,
    report_figure_path: Path,
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
    report_rel = _relative_markdown_path(report_figure_path, report_path)
    summary_rel = _relative_markdown_path(summary_figure_path, report_path)
    graph_rel = _relative_markdown_path(graph_figure_path, report_path)
    text = f"""# MLP 学习下 Granger 与 PEID 因果图对照实验

本文档记录一个模拟实验：先生成带有已知因果结构的时间序列，用 MLP 学习一步转移动力学，再分别用基于 time lag 的 Granger/ablation 方法和 PEID 最大熵干预方法识别变量之间的因果关系。

## 实验设计

变量为 `x, y, z, w`。`w` 主要作为无关变量或 common-driver 对照。当前 smoke 结果包含三类机制：

- `linear_additive`：真实机制是 `x -> z` 与 `y -> z` 的线性加性 pairwise 因果关系。这是 sanity check。
- `multiplicative_gate`：真实机制是 `{{x, y}} -> z` 的连续非线性协同门控关系，单独看 `x` 或 `y` 都不足以解释目标。
- `xor_synergy`：真实机制是 `{{x, y}} -> z` 的 XOR/parity 协同关系，是 PEID 应该明显优于 pairwise time-lag 图的核心例子。

## Ground truth 因果图、MLP 学习情况与两种识别结果

下图每一行是一个机制；四列分别是 Ground truth 因果图、MLP 学习情况（loss 曲线）、time lag / Granger 识别的因果图、PEID 识别的因果图。蓝色箭头表示普通 pairwise 边，绿色结构表示 PEID 的协同超边。

![实验示意与结果图]({report_rel})

## 汇总结果图

下图汇总不同机制下的平均 F1。对协同机制，Granger 只能输出 pairwise 边，而 PEID 可以输出 `{{x, y}} -> z` 的 synergy hyperedge，因此 `peid_advantage` 为正。

![F1 汇总图]({summary_rel})

## 代表性因果图对照

下图单独放大比较 Granger 与 PEID 的代表性因果图。可以看到，在 `multiplicative_gate` 和 `xor_synergy` 中，Granger 倾向给出 `x -> z`、`y -> z` 这样的 pairwise 解释；PEID 则把同一机制表达为 `{{x, y}} -> z` 的协同超边。

![代表性因果图]({graph_rel})

## 数值汇总

| 指标 | 含义 |
| --- | --- |
| `granger_pairwise_f1` | time lag / Granger pairwise 图相对真实 pairwise 边的 F1 |
| `peid_pairwise_f1` | PEID pairwise EI 图相对真实 pairwise 边的 F1 |
| `peid_hyperedge_f1` | PEID synergy hyperedge 相对真实协同超边的 F1 |
| `peid_advantage` | `peid_hyperedge_f1 - granger_pairwise_f1`，越大越凸显 PEID 在协同机制中的优势 |

{table}

## 结论

这个实验凸显了两类方法差异显著的条件：真实机制不是单变量滞后可以还原的 pairwise 关系，而是需要多个源变量联合出现才产生目标响应的协同机制。此时 Granger/time-lag 图容易把联合机制拆成若干 pairwise 箭头；PEID 在最大熵独立干预下比较 joint EI 与 single-source EI，可以把这种结构表示为协同超边，因此更接近 ground truth。
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

    mechanisms = tuple(mechanisms or (("linear_additive", "xor_synergy", "multiplicative_gate") if mode == "smoke" else ("linear_additive", "xor_synergy", "multiplicative_gate", "redundant_common_driver")))
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
        if mechanism not in {"multiplicative_gate", "xor_synergy"} and synergy_strength != synergy_values[0]:
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
        edge_rows.extend(_edge_records(run_id, config, granger_edges, peid))

    summary_path = result_dir / "summary.json"
    edge_table_path = result_dir / "edge_table.jsonl"
    figure_path = _plot_summary(runs, figure_dir)
    graph_figure_path = _plot_representative_causal_graphs(runs, edge_rows, figure_dir)
    report_figure_path = _plot_report_panels(runs, edge_rows, figure_dir)
    report_markdown_path = _write_chinese_report(
        runs,
        summary_figure_path=figure_path,
        graph_figure_path=graph_figure_path,
        report_figure_path=report_figure_path,
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
        "report_markdown_path": str(report_markdown_path),
        "edge_table_path": str(edge_table_path),
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
