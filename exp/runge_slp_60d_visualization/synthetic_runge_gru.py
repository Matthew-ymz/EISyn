from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticRungeConfig:
    dynamics_kind: str = "hopf"
    n_components: int = 60
    n_steps: int = 3339
    burn_in: int = 1000
    lag: int = 4
    noise_scale: float = 0.001
    nonlinear_strength: float = 0.03
    coupling_strength: float = 0.03
    synergy_strength: float = 0.0
    xor_synergy_targets: int = 0
    seed: int = 42
    start: str = "1948-01-01"
    freq: str = "7D"
    apply_observation_mixing: bool = True


@dataclass(frozen=True)
class GruTrainingConfig:
    hidden_dim: int = 96
    num_layers: int = 1
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    patience: int = 16
    gradient_clip_norm: float = 1.0
    use_linear_skip: bool = False
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    seed: int = 42
    num_threads: int = 1


@dataclass(frozen=True)
class MlpTrainingConfig:
    hidden_dims: tuple[int, ...] = (128, 64)
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    patience: int = 16
    gradient_clip_norm: float = 1.0
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    seed: int = 42
    num_threads: int = 1


@dataclass(frozen=True)
class WindowedDataset:
    X: np.ndarray
    y: np.ndarray
    target_time: pd.DatetimeIndex | None


@dataclass(frozen=True)
class GruExperimentResult:
    metrics: dict[str, float]
    history: pd.DataFrame
    y_true: np.ndarray
    y_pred: np.ndarray
    persistence_pred: np.ndarray
    split_summary: pd.DataFrame
    model: Any | None = field(default=None, repr=False)
    x_mean: np.ndarray | None = field(default=None, repr=False)
    x_std: np.ndarray | None = field(default=None, repr=False)
    y_mean: np.ndarray | None = field(default=None, repr=False)
    y_std: np.ndarray | None = field(default=None, repr=False)


def load_component_scores(path: str | Path) -> pd.DataFrame:
    scores = pd.read_csv(path, parse_dates=["time"]).set_index("time").sort_index()
    component_cols = [col for col in scores.columns if col.startswith("component_")]
    if not component_cols:
        raise ValueError(f"{path} does not contain component_* columns.")
    return scores[component_cols].astype(float)


def _parents(n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n_components)
    return (idx + 3) % n_components, (idx + 17) % n_components, (idx + 29) % n_components


def _sparse_var12_transition_matrix() -> tuple[np.ndarray, list[dict[str, Any]]]:
    d = 12
    matrix = np.zeros((d, d), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    self_coefficients = [0.64, 0.58, 0.62, 0.60, 0.57, 0.63, 0.59, 0.61, 0.56, 0.60, 0.58, 0.62]
    for idx, coefficient in enumerate(self_coefficients):
        matrix[idx, idx] = float(coefficient)
        rows.append(
            {
                "source": f"component_{idx + 1:02d}",
                "target": f"component_{idx + 1:02d}",
                "source_index": idx,
                "target_index": idx,
                "coefficient": float(coefficient),
                "edge_kind": "self_memory",
            }
        )

    cross_edges = [
        (0, 3, 0.48),
        (1, 4, -0.42),
        (2, 5, 0.45),
        (3, 6, 0.40),
        (4, 7, -0.44),
        (5, 8, 0.43),
        (6, 9, 0.39),
        (7, 10, 0.41),
        (8, 11, -0.40),
        (9, 0, 0.34),
        (10, 1, -0.36),
        (11, 2, 0.35),
    ]
    for source, target, coefficient in cross_edges:
        matrix[target, source] = float(coefficient)
        rows.append(
            {
                "source": f"component_{source + 1:02d}",
                "target": f"component_{target + 1:02d}",
                "source_index": int(source),
                "target_index": int(target),
                "coefficient": float(coefficient),
                "edge_kind": "cross_causal",
            }
        )

    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if spectral_radius >= 0.92:
        matrix *= 0.92 / spectral_radius
        for row in rows:
            row["coefficient"] = float(matrix[int(row["target_index"]), int(row["source_index"])])
    return matrix, rows


def _simulate_sparse_var12_dynamics(config: SyntheticRungeConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(int(config.seed))
    d = int(config.n_components)
    if d != 12:
        raise ValueError("sparse_var12 requires n_components=12.")

    transition, parent_edges = _sparse_var12_transition_matrix()
    total = int(config.burn_in) + int(config.n_steps) + 1
    values = np.zeros((total, d), dtype=np.float64)
    values[0] = rng.normal(scale=0.5, size=d)
    noise_scale = float(config.noise_scale)
    if noise_scale <= 0.0:
        noise_scale = 0.01
    for t in range(total - 1):
        innovation = rng.normal(scale=noise_scale, size=d)
        values[t + 1] = transition @ values[t] + innovation

    observed = values[int(config.burn_in) : int(config.burn_in) + int(config.n_steps)]
    observed = observed - observed.mean(axis=0, keepdims=True)
    observed = observed / np.maximum(observed.std(axis=0, ddof=1, keepdims=True), 1.0e-12)
    target_scale = 0.78 + 0.04 * np.exp(-np.arange(d) / max(d / 4.0, 1.0))
    observed = observed * target_scale[None, :]

    columns = [f"component_{i + 1:02d}" for i in range(d)]
    index = pd.date_range(str(config.start), periods=int(config.n_steps), freq=str(config.freq), name="time")
    frame = pd.DataFrame(observed.astype(np.float32), index=index, columns=columns)
    metadata: dict[str, Any] = {
        "equation": "sparse_var12",
        "dynamics_kind": "sparse_var12",
        "n_components": d,
        "lag": 1,
        "coordinate_space": "mechanism",
        "transition_matrix": transition.astype(float).tolist(),
        "parent_edges": parent_edges,
        "noise_scale": float(config.noise_scale),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(transition)))),
    }
    return frame, metadata


def _nonlinear_gateway_synergy12_truth() -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    d = 12
    matrix = np.zeros((d, d), dtype=np.float64)
    rows: list[dict[str, Any]] = []

    def add_edge(source: int, target: int, coefficient: float, edge_kind: str) -> None:
        matrix[int(target), int(source)] = float(coefficient)
        rows.append(
            {
                "source": f"component_{source + 1:02d}",
                "target": f"component_{target + 1:02d}",
                "source_index": int(source),
                "target_index": int(target),
                "coefficient": float(coefficient),
                "edge_kind": str(edge_kind),
            }
        )

    self_coefficients = [0.46, 0.42, 0.41, 0.40, 0.39, 0.38, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35]
    for idx, coefficient in enumerate(self_coefficients):
        add_edge(idx, idx, coefficient, "self_memory")

    for source, target, coefficient in [
        (1, 0, 0.34),
        (2, 0, -0.31),
        (3, 0, 0.27),
        (4, 0, 0.23),
        (5, 0, -0.21),
        (1, 2, 0.17),
        (2, 3, -0.16),
        (3, 4, 0.15),
        (4, 5, 0.14),
        (5, 1, -0.13),
        (6, 7, 0.15),
        (7, 8, -0.14),
        (8, 9, 0.14),
        (9, 10, 0.13),
        (10, 11, -0.13),
        (11, 6, 0.12),
    ]:
        add_edge(source, target, coefficient, "within_module")

    for target, coefficient in [(6, 0.58), (7, -0.54), (8, 0.51), (9, 0.48), (10, -0.45), (11, 0.42)]:
        add_edge(0, target, coefficient, "gateway_cross_module")

    for source, target, coefficient in [(2, 4, 0.19), (5, 3, -0.18), (7, 10, 0.17), (8, 11, -0.16)]:
        add_edge(source, target, coefficient, "single_source_nonlinear")

    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if spectral_radius >= 0.88:
        matrix *= 0.88 / spectral_radius
        for row in rows:
            row["coefficient"] = float(matrix[int(row["target_index"]), int(row["source_index"])])

    synergy_edges = [
        {"source_i": 0, "source_j": 1, "target_index": 6, "coefficient": 0.42, "edge_kind": "gateway_synergy"},
        {"source_i": 0, "source_j": 2, "target_index": 7, "coefficient": -0.38, "edge_kind": "gateway_synergy"},
        {"source_i": 0, "source_j": 5, "target_index": 10, "coefficient": 0.36, "edge_kind": "gateway_synergy"},
        {"source_i": 0, "source_j": 6, "target_index": 8, "coefficient": 0.34, "edge_kind": "gateway_receiver_synergy"},
        {"source_i": 0, "source_j": 8, "target_index": 11, "coefficient": -0.32, "edge_kind": "gateway_receiver_synergy"},
    ]
    for row in synergy_edges:
        source_i = int(row["source_i"])
        source_j = int(row["source_j"])
        target = int(row["target_index"])
        row["source_i_name"] = f"component_{source_i + 1:02d}"
        row["source_j_name"] = f"component_{source_j + 1:02d}"
        row["sources"] = f"component_{source_i + 1:02d}+component_{source_j + 1:02d}"
        row["target"] = f"component_{target + 1:02d}"
    return matrix, rows, synergy_edges


def _simulate_nonlinear_gateway_synergy12_dynamics(config: SyntheticRungeConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(int(config.seed))
    d = int(config.n_components)
    if d != 12:
        raise ValueError("nonlinear_gateway_synergy12 requires n_components=12.")

    transition, parent_edges, synergy_edges = _nonlinear_gateway_synergy12_truth()
    total = int(config.burn_in) + int(config.n_steps) + 1
    values = np.zeros((total, d), dtype=np.float64)
    values[0] = rng.normal(scale=0.45, size=d)
    noise_scale = max(float(config.noise_scale), 1.0e-6)
    nonlinear_strength = float(config.nonlinear_strength)
    synergy_strength = float(config.synergy_strength) if float(config.synergy_strength) != 0.0 else 0.22

    for t in range(total - 1):
        x = values[t]
        candidate = transition @ x
        single_nonlinear = np.zeros(d, dtype=np.float64)
        single_nonlinear[4] += 0.55 * np.tanh(1.4 * x[2])
        single_nonlinear[3] -= 0.48 * np.sin(0.9 * x[5])
        single_nonlinear[10] += 0.42 * np.tanh(1.2 * x[7])
        single_nonlinear[11] -= 0.38 * np.sin(1.1 * x[8])
        gateway_drive = np.tanh(1.3 * x[0])
        single_nonlinear[6] += 0.36 * gateway_drive
        single_nonlinear[7] -= 0.32 * gateway_drive

        pair_nonlinear = np.zeros(d, dtype=np.float64)
        pair_nonlinear[6] += 0.80 * np.tanh(1.6 * x[0] * x[1])
        pair_nonlinear[7] -= 0.72 * np.sin(0.9 * x[0] * x[2])
        pair_nonlinear[10] += 0.68 * np.tanh(1.4 * x[0] * x[5])
        pair_nonlinear[8] += 0.62 * np.tanh(1.3 * x[0] * x[6])
        pair_nonlinear[11] -= 0.58 * np.sin(0.8 * x[0] * x[8])

        innovation = rng.normal(scale=noise_scale, size=d)
        values[t + 1] = np.clip(candidate + nonlinear_strength * single_nonlinear + synergy_strength * pair_nonlinear + innovation, -4.5, 4.5)

    observed = values[int(config.burn_in) : int(config.burn_in) + int(config.n_steps)]
    observed = observed - observed.mean(axis=0, keepdims=True)
    observed = observed / np.maximum(observed.std(axis=0, ddof=1, keepdims=True), 1.0e-12)
    target_scale = 0.78 + 0.04 * np.exp(-np.arange(d) / max(d / 4.0, 1.0))
    observed = observed * target_scale[None, :]

    columns = [f"component_{i + 1:02d}" for i in range(d)]
    index = pd.date_range(str(config.start), periods=int(config.n_steps), freq=str(config.freq), name="time")
    frame = pd.DataFrame(observed.astype(np.float32), index=index, columns=columns)
    metadata: dict[str, Any] = {
        "equation": "nonlinear_gateway_synergy12",
        "dynamics_kind": "nonlinear_gateway_synergy12",
        "n_components": d,
        "lag": 1,
        "coordinate_space": "mechanism",
        "transition_matrix": transition.astype(float).tolist(),
        "parent_edges": parent_edges,
        "synergy_edges": synergy_edges,
        "gateway_node": "component_01",
        "gateway_index": 0,
        "gateway_expected_rank": 1,
        "module_labels": {f"component_{idx + 1:02d}": ("A" if idx < 6 else "B") for idx in range(d)},
        "noise_scale": float(config.noise_scale),
        "nonlinear_strength": nonlinear_strength,
        "synergy_strength": synergy_strength,
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(transition)))),
    }
    return frame, metadata


def simulate_runge_like_dynamics(config: SyntheticRungeConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate a known component-score process."""

    dynamics_kind = str(config.dynamics_kind)
    if dynamics_kind == "sparse_var12":
        return _simulate_sparse_var12_dynamics(config)
    if dynamics_kind == "nonlinear_gateway_synergy12":
        return _simulate_nonlinear_gateway_synergy12_dynamics(config)
    if dynamics_kind != "hopf":
        raise ValueError("dynamics_kind must be 'hopf', 'sparse_var12', or 'nonlinear_gateway_synergy12'.")

    rng = np.random.default_rng(int(config.seed))
    d = int(config.n_components)
    if d < 2:
        raise ValueError("n_components must be at least 2.")
    if d % 2 != 0:
        raise ValueError("n_components must be even for paired Hopf modes.")
    if int(config.lag) < 4:
        raise ValueError("lag must be at least 4 for the Runge-like delayed terms.")

    total = int(config.burn_in) + int(config.n_steps) + int(config.lag) + 1
    values = np.zeros((total, d), dtype=np.float64)
    values[: int(config.lag)] = rng.normal(scale=0.4, size=(int(config.lag), d))

    idx = np.arange(d, dtype=int)
    parent_a, parent_b, parent_c = _parents(d)
    n_pairs = d // 2
    theta = 0.35 + 0.30 * rng.random(n_pairs)
    growth = 0.04 + 0.01 * rng.random(n_pairs)
    saturation = 0.12
    nonlinear_sign = rng.choice([-1.0, 1.0], size=d)
    mixing, _ = np.linalg.qr(rng.normal(size=(d, d)))

    for t in range(int(config.lag), total - 1):
        x1 = values[t]
        x2 = values[t - 1]
        x4 = values[t - 3]
        next_values = np.zeros(d, dtype=np.float64)
        for pair in range(n_pairs):
            i = 2 * pair
            j = i + 1
            a = x1[i]
            b = x1[j]
            radius_sq = a * a + b * b
            next_values[i] = (1.0 + growth[pair]) * a - theta[pair] * b - saturation * radius_sq * a
            next_values[j] = theta[pair] * a + (1.0 + growth[pair]) * b - saturation * radius_sq * b
        lagged_memory = -0.08 * x2 + 0.04 * x4
        sparse_cross = float(config.coupling_strength) * (0.50 * x1[parent_a] + 0.20 * x2[parent_b])
        nonlinear = float(config.nonlinear_strength) * (
            0.30 * np.tanh(x1[parent_a] - x2[parent_b])
            + 0.20 * nonlinear_sign * np.sin(x1[parent_a] * x2[parent_b])
        )
        synergy_gate = float(config.synergy_strength) * nonlinear_sign * np.tanh(2.0 * x1[parent_a] * x2[parent_b])
        innovation = rng.normal(scale=float(config.noise_scale), size=d)
        candidate_next = next_values + lagged_memory + sparse_cross + nonlinear + synergy_gate + innovation
        if int(config.xor_synergy_targets) > 0:
            target_idx = np.arange(min(int(config.xor_synergy_targets), d))
            xor_signal = np.sign(x1[parent_a[target_idx]]) * np.sign(x1[parent_b[target_idx]])
            xor_signal = np.where(xor_signal == 0.0, 1.0, xor_signal)
            candidate_next[target_idx] = 0.85 * xor_signal + 0.05 * rng.normal(size=len(target_idx))
        values[t + 1] = np.clip(candidate_next, -5.0, 5.0)

    observed = values[int(config.burn_in) + int(config.lag) : int(config.burn_in) + int(config.lag) + int(config.n_steps)]
    if bool(config.apply_observation_mixing):
        observed = observed @ mixing
    observed = observed - observed.mean(axis=0, keepdims=True)
    observed = observed / np.maximum(observed.std(axis=0, ddof=1, keepdims=True), 1.0e-12)
    target_scale = 0.78 + 0.04 * np.exp(-np.arange(d) / max(d / 4.0, 1.0))
    target_scale += 0.02 * np.sin(2.0 * np.pi * np.arange(d) / max(d, 1))
    observed = observed * target_scale[None, :]

    columns = [f"component_{i + 1:02d}" for i in range(d)]
    index = pd.date_range(str(config.start), periods=int(config.n_steps), freq=str(config.freq), name="time")
    frame = pd.DataFrame(observed.astype(np.float32), index=index, columns=columns)
    metadata: dict[str, Any] = {
        "equation": "mixed_sparse_lagged_hopf_map",
        "dynamics_kind": "hopf",
        "n_components": d,
        "lag": int(config.lag),
        "parents_a": (parent_a + 1).astype(int).tolist(),
        "parents_b": (parent_b + 1).astype(int).tolist(),
        "parents_c": (parent_c + 1).astype(int).tolist(),
        "pair_partner": (np.where(idx % 2 == 0, idx + 1, idx - 1) + 1).astype(int).tolist(),
        "coordinate_space": "observed_mixed" if bool(config.apply_observation_mixing) else "mechanism",
        "theta_mean": float(np.mean(theta)),
        "growth_mean": float(np.mean(growth)),
        "noise_scale": float(config.noise_scale),
        "nonlinear_strength": float(config.nonlinear_strength),
        "coupling_strength": float(config.coupling_strength),
        "synergy_strength": float(config.synergy_strength),
        "xor_synergy_targets": int(config.xor_synergy_targets),
    }
    return frame, metadata


def ground_truth_pairwise_edges(metadata: dict[str, Any], *, include_self: bool = True, source_mode: str = "history") -> pd.DataFrame:
    """Return the sparse mechanism-coordinate parent graph implied by the generator."""

    if source_mode not in {"history", "latest"}:
        raise ValueError("source_mode must be 'history' or 'latest'.")
    if "parent_edges" in metadata:
        rows = []
        for edge in metadata["parent_edges"]:
            source = int(edge["source_index"])
            target = int(edge["target_index"])
            if not include_self and source == target:
                continue
            rows.append(
                {
                    "source": str(edge.get("source", f"component_{source + 1:02d}")),
                    "target": str(edge.get("target", f"component_{target + 1:02d}")),
                    "source_index": source,
                    "target_index": target,
                    "edge_kind": str(edge.get("edge_kind", "ground_truth_parent")),
                    "coefficient": float(edge.get("coefficient", 1.0)),
                }
            )
        return pd.DataFrame(rows)

    n_components = int(metadata["n_components"])
    parents_a = [int(value) - 1 for value in metadata["parents_a"]]
    parents_b = [int(value) - 1 for value in metadata["parents_b"]]
    pair_partner = [int(value) - 1 for value in metadata["pair_partner"]]
    rows: list[dict[str, Any]] = []
    for target in range(n_components):
        sources: set[int] = {pair_partner[target], parents_a[target]}
        if source_mode == "history":
            sources.add(parents_b[target])
        if include_self:
            sources.add(target)
        for source in sorted(sources):
            rows.append(
                {
                    "source": f"component_{source + 1:02d}",
                    "target": f"component_{target + 1:02d}",
                    "source_index": int(source),
                    "target_index": int(target),
                    "edge_kind": "ground_truth_parent",
                }
            )
    return pd.DataFrame(rows)


def ground_truth_synergy_edges(metadata: dict[str, Any]) -> pd.DataFrame:
    """Return designed nonlinear source-pair interactions in mechanism coordinates."""

    if "synergy_edges" in metadata:
        rows = []
        for edge in metadata["synergy_edges"]:
            source_i = int(edge["source_i"])
            source_j = int(edge["source_j"])
            target = int(edge["target_index"])
            left, right = sorted((source_i, source_j))
            rows.append(
                {
                    "source_i": left,
                    "source_j": right,
                    "target_index": target,
                    "sources": str(edge.get("sources", f"component_{left + 1:02d}+component_{right + 1:02d}")),
                    "target": str(edge.get("target", f"component_{target + 1:02d}")),
                    "coefficient": float(edge.get("coefficient", 1.0)),
                    "edge_kind": str(edge.get("edge_kind", "ground_truth_synergy")),
                }
            )
        return pd.DataFrame(rows)

    n_components = int(metadata["n_components"])
    parents_a = [int(value) - 1 for value in metadata["parents_a"]]
    parents_b = [int(value) - 1 for value in metadata["parents_b"]]
    pair_partner = [int(value) - 1 for value in metadata["pair_partner"]]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for target in range(n_components):
        candidates = [
            (*sorted((parents_a[target], parents_b[target])), "ground_truth_teleconnection_pair"),
            (*sorted((target, pair_partner[target])), "ground_truth_hopf_pair"),
        ]
        for source_i, source_j, edge_kind in candidates:
            key = (int(source_i), int(source_j), target)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "source_i": int(source_i),
                    "source_j": int(source_j),
                    "target_index": int(target),
                    "sources": f"component_{source_i + 1:02d}+component_{source_j + 1:02d}",
                    "target": f"component_{target + 1:02d}",
                    "edge_kind": str(edge_kind),
                }
            )
    return pd.DataFrame(rows)


def _ground_truth_adjacency(metadata: dict[str, Any], *, include_self: bool = False) -> np.ndarray:
    n_components = int(metadata["n_components"])
    adjacency = np.zeros((n_components, n_components), dtype=float)
    for edge in metadata.get("parent_edges", []):
        source = int(edge["source_index"])
        target = int(edge["target_index"])
        if not include_self and source == target:
            continue
        coefficient = abs(float(edge.get("coefficient", 1.0)))
        adjacency[source, target] = max(float(adjacency[source, target]), coefficient)
    return adjacency


def compute_gateway_scores(metadata: dict[str, Any], *, max_path_length: int = 4, damping: float = 0.65) -> pd.DataFrame:
    """Rank components by designed outgoing path effect in the metadata graph."""

    adjacency = _ground_truth_adjacency(metadata, include_self=False)
    if adjacency.size == 0:
        return pd.DataFrame(columns=["rank", "component", "out_degree", "cross_module_out_degree", "direct_out_strength", "path_effect"])
    max_row_sum = max(float(np.max(adjacency.sum(axis=1))), 1.0e-12)
    normalized = adjacency / max_row_sum
    path_effect_matrix = np.zeros_like(normalized, dtype=float)
    power = normalized.copy()
    for length in range(1, int(max_path_length) + 1):
        path_effect_matrix += (float(damping) ** (length - 1)) * power
        power = power @ normalized

    labels = metadata.get("module_labels", {})
    names = [f"component_{idx + 1:02d}" for idx in range(adjacency.shape[0])]
    rows = []
    for source, name in enumerate(names):
        source_module = labels.get(name)
        cross_out = 0
        for target, target_name in enumerate(names):
            if adjacency[source, target] > 0.0 and source_module is not None and labels.get(target_name) != source_module:
                cross_out += 1
        rows.append(
            {
                "component": name,
                "source_index": int(source),
                "out_degree": float(np.count_nonzero(adjacency[source] > 0.0)),
                "cross_module_out_degree": float(cross_out),
                "direct_out_strength": float(adjacency[source].sum()),
                "path_effect": float(path_effect_matrix[source].sum()),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["path_effect", "direct_out_strength"], ascending=[False, False]).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1, dtype=int))

    gateway_index = int(metadata.get("gateway_index", 0))
    module_a = [idx for idx, name in enumerate(names) if labels.get(name) == "A"]
    module_b = [idx for idx, name in enumerate(names) if labels.get(name) == "B"]
    full_cross = float(path_effect_matrix[np.ix_(module_a, module_b)].sum()) if module_a and module_b else float("nan")
    without_gateway_adjacency = normalized.copy()
    if 0 <= gateway_index < without_gateway_adjacency.shape[0]:
        without_gateway_adjacency[gateway_index, :] = 0.0
        without_gateway_adjacency[:, gateway_index] = 0.0
    reduced = np.zeros_like(without_gateway_adjacency, dtype=float)
    power = without_gateway_adjacency.copy()
    for length in range(1, int(max_path_length) + 1):
        reduced += (float(damping) ** (length - 1)) * power
        power = power @ without_gateway_adjacency
    reduced_cross = float(reduced[np.ix_(module_a, module_b)].sum()) if module_a and module_b else float("nan")
    frame.attrs["cross_module_path_effect_full"] = full_cross
    frame.attrs["cross_module_path_effect_without_gateway"] = reduced_cross
    return frame


def _edge_weight(row: Any) -> float:
    if hasattr(row, "ei"):
        return max(0.0, float(getattr(row, "ei")))
    if hasattr(row, "coefficient"):
        return abs(float(getattr(row, "coefficient")))
    if hasattr(row, "weight"):
        return max(0.0, float(getattr(row, "weight")))
    return 1.0


def _path_effect_matrix(adjacency: np.ndarray, *, max_path_length: int, damping: float) -> np.ndarray:
    values = np.asarray(adjacency, dtype=float)
    max_row_sum = max(float(np.max(values.sum(axis=1))), 1.0e-12)
    normalized = values / max_row_sum
    path_effect = np.zeros_like(normalized, dtype=float)
    power = normalized.copy()
    for length in range(1, int(max_path_length) + 1):
        path_effect += (float(damping) ** (length - 1)) * power
        power = power @ normalized
    return path_effect


def compute_causal_role_scores(
    *,
    pairwise_edges: pd.DataFrame,
    synergy_edges: pd.DataFrame | None = None,
    n_components: int | None = None,
    module_labels: dict[str, str] | None = None,
    top_pairwise_edges: int | None = None,
    max_path_length: int = 4,
    damping: float = 0.65,
) -> pd.DataFrame:
    """Rank gateway and mediator roles from pairwise PEID/EI edges plus optional Syn edges.

    Pairwise EI is interpreted as directed probability-causal strength under the
    intervention distribution. Source-side synergy contributes to gateway-like
    broadcast roles; target-side synergy contributes to mediator roles only when
    the receiving node also has downstream causal reach. Learned PEID Syn must
    be nonnegative; negative estimates are rejected rather than clipped.
    Designed ground-truth coefficients use absolute strength.
    """

    if pairwise_edges.empty and n_components is None:
        return pd.DataFrame()
    if n_components is None:
        n_components = int(pairwise_edges[["source_index", "target_index"]].to_numpy(dtype=int).max()) + 1
    names = [f"component_{idx + 1:02d}" for idx in range(int(n_components))]
    edge_frame = pairwise_edges.copy()
    if not edge_frame.empty:
        edge_frame["role_weight"] = [_edge_weight(row) for row in edge_frame.itertuples()]
        edge_frame = edge_frame[edge_frame["role_weight"] > 0.0]
        if top_pairwise_edges is not None:
            edge_frame = edge_frame.sort_values("role_weight", ascending=False).head(int(top_pairwise_edges))

    adjacency = np.zeros((int(n_components), int(n_components)), dtype=float)
    for row in edge_frame.itertuples():
        source = int(row.source_index)
        target = int(row.target_index)
        if source == target:
            continue
        if 0 <= source < int(n_components) and 0 <= target < int(n_components):
            adjacency[source, target] = max(float(adjacency[source, target]), float(row.role_weight))
    path_effect = _path_effect_matrix(adjacency, max_path_length=int(max_path_length), damping=float(damping))

    labels = module_labels or {}
    cross_module_path_out = np.zeros(int(n_components), dtype=float)
    cross_module_direct_out = np.zeros(int(n_components), dtype=float)
    for source, source_name in enumerate(names):
        source_module = labels.get(source_name)
        if source_module is None:
            continue
        targets = [target for target, target_name in enumerate(names) if labels.get(target_name) not in {None, source_module}]
        if targets:
            cross_module_path_out[source] = float(path_effect[source, targets].sum())
            cross_module_direct_out[source] = float(adjacency[source, targets].sum())

    synergy_source = np.zeros(int(n_components), dtype=float)
    synergy_target = np.zeros(int(n_components), dtype=float)
    if synergy_edges is not None and not synergy_edges.empty:
        for row in synergy_edges.itertuples():
            if hasattr(row, "synergy"):
                weight = float(getattr(row, "synergy"))
                if weight < 0.0:
                    raise ValueError(
                        "Syn is nonnegative by definition; fix negative estimates upstream."
                    )
            elif hasattr(row, "coefficient"):
                weight = abs(float(getattr(row, "coefficient")))
            elif hasattr(row, "joint_ei"):
                weight = max(0.0, float(getattr(row, "joint_ei")))
            else:
                weight = 1.0
            if weight <= 0.0:
                continue
            source_i = int(getattr(row, "source_i"))
            source_j = int(getattr(row, "source_j"))
            target = int(getattr(row, "target_index"))
            for source in (source_i, source_j):
                if 0 <= source < int(n_components):
                    synergy_source[source] += float(weight)
            if 0 <= target < int(n_components):
                synergy_target[target] += float(weight)

    outgoing_path = path_effect.sum(axis=1)
    incoming_path = path_effect.sum(axis=0)
    direct_out = adjacency.sum(axis=1)
    direct_in = adjacency.sum(axis=0)
    gateway_score = outgoing_path + cross_module_path_out + synergy_source
    mediator_score = (incoming_path + synergy_target) * outgoing_path

    rows = []
    for idx, name in enumerate(names):
        rows.append(
            {
                "component": name,
                "source_index": int(idx),
                "module": labels.get(name, ""),
                "direct_in_strength": float(direct_in[idx]),
                "direct_out_strength": float(direct_out[idx]),
                "incoming_path_effect": float(incoming_path[idx]),
                "outgoing_path_effect": float(outgoing_path[idx]),
                "cross_module_direct_out_strength": float(cross_module_direct_out[idx]),
                "cross_module_outgoing_path_effect": float(cross_module_path_out[idx]),
                "synergy_source_score": float(synergy_source[idx]),
                "synergy_target_score": float(synergy_target[idx]),
                "gateway_score": float(gateway_score[idx]),
                "mediator_score": float(mediator_score[idx]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["gateway_rank"] = frame["gateway_score"].rank(method="first", ascending=False).astype(int)
    frame["mediator_rank"] = frame["mediator_score"].rank(method="first", ascending=False).astype(int)
    return frame.sort_values(["gateway_score", "mediator_score"], ascending=[False, False]).reset_index(drop=True)


def _positive_synergy_count_for_component(synergy_edges: pd.DataFrame, *, component_index: int) -> int:
    if synergy_edges.empty:
        return 0
    count = 0
    for row in synergy_edges.itertuples():
        if hasattr(row, "synergy"):
            weight = float(getattr(row, "synergy"))
            if weight < 0.0:
                raise ValueError(
                    "Syn is nonnegative by definition; fix negative estimates upstream."
                )
        elif hasattr(row, "coefficient"):
            weight = abs(float(getattr(row, "coefficient")))
        elif hasattr(row, "joint_ei"):
            weight = max(0.0, float(getattr(row, "joint_ei")))
        else:
            weight = 1.0
        if weight <= 0.0:
            continue
        sources = {int(getattr(row, "source_i")), int(getattr(row, "source_j"))}
        if int(component_index) in sources:
            count += 1
    return count


def build_synergy_necessity_ablation(
    *,
    estimators: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    n_components: int,
    module_labels: dict[str, str] | None = None,
    component: str = "component_01",
    top_pairwise_edges: int | None = None,
) -> pd.DataFrame:
    """Compare pairwise-only and synergy-aware causal role scores for a target component."""

    component_index = int(component.rsplit("_", 1)[-1]) - 1
    rows: list[dict[str, Any]] = []
    for estimator, pairwise_edges, synergy_edges in estimators:
        pairwise_only = compute_causal_role_scores(
            pairwise_edges=pairwise_edges,
            synergy_edges=None,
            n_components=int(n_components),
            module_labels=module_labels,
            top_pairwise_edges=top_pairwise_edges,
        )
        synergy_aware = compute_causal_role_scores(
            pairwise_edges=pairwise_edges,
            synergy_edges=synergy_edges,
            n_components=int(n_components),
            module_labels=module_labels,
            top_pairwise_edges=top_pairwise_edges,
        )
        pairwise_row = pairwise_only.loc[pairwise_only["component"] == str(component)].iloc[0]
        synergy_row = synergy_aware.loc[synergy_aware["component"] == str(component)].iloc[0]
        synergy_top = str(synergy_aware.iloc[0]["component"])
        rows.append(
            {
                "Estimator": str(estimator),
                "pairwise_only_component_01_gateway_rank": int(pairwise_row["gateway_rank"]),
                "synergy_aware_component_01_gateway_rank": int(synergy_row["gateway_rank"]),
                "pairwise_only_top_gateway": str(pairwise_only.iloc[0]["component"]),
                "synergy_aware_top_gateway": synergy_top,
                "synergy_aware_top_is_component_01": bool(synergy_top == str(component)),
                "positive_synergy_edges_with_component_01": _positive_synergy_count_for_component(
                    synergy_edges,
                    component_index=component_index,
                ),
                "component_01_synergy_source_score": float(synergy_row["synergy_source_score"]),
                "component_01_synergy_target_score": float(synergy_row["synergy_target_score"]),
            }
        )
    return pd.DataFrame(rows).set_index("Estimator")


def _component_gateway_summary(role_scores: pd.DataFrame, *, component: str = "component_01") -> dict[str, Any]:
    row = role_scores.loc[role_scores["component"] == str(component)].iloc[0]
    return {
        "component_01_gateway_rank": int(row["gateway_rank"]),
        "component_01_gateway_score": float(row["gateway_score"]),
        "top_gateway": str(role_scores.iloc[0]["component"]),
    }


def run_gateway_synergy_benchmark(
    *,
    synthetic_config: SyntheticRungeConfig,
    gru_config: GruTrainingConfig,
    mlp_config: MlpTrainingConfig,
    history: int = 4,
    horizon: int = 1,
    intervention_samples: int = 2048,
    bins: int = 5,
    top_synergy_sources_per_target: int = 3,
    seed_offset: int = 100,
) -> dict[str, Any]:
    """Run the nonlinear gateway benchmark as a notebook-facing orchestration helper."""

    scores, metadata = simulate_runge_like_dynamics(synthetic_config)
    truth_pairwise = ground_truth_pairwise_edges(metadata, source_mode="latest")
    truth_synergy = ground_truth_synergy_edges(metadata)
    topology_scores = compute_gateway_scores(metadata)
    windows = make_supervised_windows(scores, history=int(history), horizon=int(horizon))

    gru_result = train_gru_forecaster(windows, gru_config)
    mlp_result = train_mlp_forecaster(windows, mlp_config)
    gru_peid = estimate_gru_peid_graphs(
        gru_result,
        windows,
        intervention_samples=int(intervention_samples),
        bins=int(bins),
        top_synergy_sources_per_target=int(top_synergy_sources_per_target),
        extra_synergy_candidates=truth_synergy,
        null_reps=0,
        source_mode="latest",
        seed=int(gru_config.seed) + int(seed_offset),
    )
    mlp_peid = estimate_model_peid_graphs(
        mlp_result,
        windows,
        prediction_fn=predict_mlp_windows,
        intervention_samples=int(intervention_samples),
        bins=int(bins),
        top_synergy_sources_per_target=int(top_synergy_sources_per_target),
        extra_synergy_candidates=truth_synergy,
        null_reps=0,
        source_mode="latest",
        seed=int(mlp_config.seed) + int(seed_offset),
    )

    top_graph_edges = len(truth_pairwise)
    gru_recovery = pairwise_recovery_summary(gru_peid["pairwise_edges"], truth_pairwise, top_k=top_graph_edges)
    mlp_recovery = pairwise_recovery_summary(mlp_peid["pairwise_edges"], truth_pairwise, top_k=top_graph_edges)
    metric_frame = pd.concat(
        [
            metrics_table(gru_result).assign(Model="GRU"),
            metrics_table(mlp_result).assign(Model="MLP"),
        ],
        ignore_index=True,
    ).pivot(index="Metric", columns="Model", values="Value")
    recovery_frame = pd.DataFrame(
        [
            {"Estimator": "GRU + discrete PEID", **gru_recovery},
            {"Estimator": "MLP + discrete PEID", **mlp_recovery},
        ]
    ).set_index("Estimator").T

    module_labels = metadata.get("module_labels", {})
    truth_role_scores = compute_causal_role_scores(
        pairwise_edges=truth_pairwise,
        synergy_edges=truth_synergy,
        n_components=int(synthetic_config.n_components),
        module_labels=module_labels,
    )
    gru_role_scores = compute_causal_role_scores(
        pairwise_edges=gru_peid["pairwise_edges"],
        synergy_edges=gru_peid["synergy_edges"],
        n_components=int(synthetic_config.n_components),
        module_labels=module_labels,
        top_pairwise_edges=top_graph_edges,
    )
    mlp_role_scores = compute_causal_role_scores(
        pairwise_edges=mlp_peid["pairwise_edges"],
        synergy_edges=mlp_peid["synergy_edges"],
        n_components=int(synthetic_config.n_components),
        module_labels=module_labels,
        top_pairwise_edges=top_graph_edges,
    )
    role_summary = pd.DataFrame(
        [
            {"Estimator": "Ground-truth dynamics", **_component_gateway_summary(truth_role_scores)},
            {"Estimator": "GRU + PEID", **_component_gateway_summary(gru_role_scores)},
            {"Estimator": "MLP + PEID", **_component_gateway_summary(mlp_role_scores)},
        ]
    ).set_index("Estimator")
    synergy_necessity_ablation = build_synergy_necessity_ablation(
        estimators=[
            ("Ground-truth dynamics", truth_pairwise, truth_synergy),
            ("GRU + PEID", gru_peid["pairwise_edges"], gru_peid["synergy_edges"]),
            ("MLP + PEID", mlp_peid["pairwise_edges"], mlp_peid["synergy_edges"]),
        ],
        n_components=int(synthetic_config.n_components),
        module_labels=module_labels,
        top_pairwise_edges=top_graph_edges,
    )

    return {
        "gateway_scores": scores,
        "gateway_metadata": metadata,
        "gateway_windows": windows,
        "gateway_gru_result": gru_result,
        "gateway_mlp_result": mlp_result,
        "gateway_gru_peid": gru_peid,
        "gateway_mlp_peid": mlp_peid,
        "gateway_truth_pairwise": truth_pairwise,
        "gateway_truth_synergy": truth_synergy,
        "gateway_topology_scores": topology_scores,
        "gateway_metric_frame": metric_frame,
        "gateway_recovery_frame": recovery_frame,
        "gateway_truth_role_scores": truth_role_scores,
        "gateway_gru_role_scores": gru_role_scores,
        "gateway_mlp_role_scores": mlp_role_scores,
        "role_summary": role_summary,
        "synergy_necessity_ablation": synergy_necessity_ablation,
        "top_graph_edges": top_graph_edges,
    }


def make_supervised_windows(frame: pd.DataFrame, *, history: int, horizon: int = 1) -> WindowedDataset:
    values = frame.to_numpy(dtype=np.float32)
    n_samples = len(frame) - int(history) - int(horizon) + 1
    if n_samples <= 0:
        raise ValueError("not enough rows for requested history and horizon.")
    X = np.empty((n_samples, int(history), values.shape[1]), dtype=np.float32)
    y = np.empty((n_samples, values.shape[1]), dtype=np.float32)
    target_positions = []
    for start in range(n_samples):
        stop = start + int(history)
        target = stop + int(horizon) - 1
        X[start] = values[start:stop]
        y[start] = values[target]
        target_positions.append(target)
    target_time = None
    if isinstance(frame.index, pd.DatetimeIndex):
        target_time = pd.DatetimeIndex(frame.index[target_positions], name="target_time")
    return WindowedDataset(X=X, y=y, target_time=target_time)


def _temporal_split(n_samples: int, train_fraction: float, val_fraction: float) -> dict[str, slice]:
    train_end = int(round(n_samples * float(train_fraction)))
    val_end = train_end + int(round(n_samples * float(val_fraction)))
    train_end = max(1, min(train_end, n_samples - 2))
    val_end = max(train_end + 1, min(val_end, n_samples - 1))
    return {"train": slice(0, train_end), "val": slice(train_end, val_end), "test": slice(val_end, n_samples)}


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def train_gru_forecaster(windows: WindowedDataset, config: GruTrainingConfig) -> GruExperimentResult:
    import torch

    warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*", category=UserWarning)
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    if int(config.num_threads) > 0:
        torch.set_num_threads(int(config.num_threads))

    X = np.asarray(windows.X, dtype=np.float32)
    y = np.asarray(windows.y, dtype=np.float32)
    splits = _temporal_split(len(X), config.train_fraction, config.val_fraction)

    train_slice = splits["train"]
    x_mean = X[train_slice].mean(axis=(0, 1), keepdims=True)
    x_std = np.maximum(X[train_slice].std(axis=(0, 1), keepdims=True), 1.0e-6)
    y_mean = y[train_slice].mean(axis=0, keepdims=True)
    y_std = np.maximum(y[train_slice].std(axis=0, keepdims=True), 1.0e-6)
    Xn = (X - x_mean) / x_std
    yn = (y - y_mean) / y_std

    class _GruNet(torch.nn.Module):
        def __init__(self, input_dim: int, history: int) -> None:
            super().__init__()
            self.gru = torch.nn.GRU(
                input_size=input_dim,
                hidden_size=int(config.hidden_dim),
                num_layers=int(config.num_layers),
                dropout=0.0 if int(config.num_layers) == 1 else 0.05,
                batch_first=True,
            )
            self.head = torch.nn.Linear(int(config.hidden_dim), input_dim)
            self.skip = torch.nn.Linear(input_dim * int(history), input_dim) if bool(config.use_linear_skip) else None
            if self.skip is not None:
                torch.nn.init.zeros_(self.head.weight)
                torch.nn.init.zeros_(self.head.bias)

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            out, _ = self.gru(batch)
            pred = self.head(out[:, -1, :])
            if self.skip is not None:
                pred = pred + self.skip(batch.reshape(batch.shape[0], -1))
            return pred

    device = torch.device("cpu")
    model = _GruNet(X.shape[2], X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    loss_fn = torch.nn.MSELoss()

    x_train = torch.as_tensor(Xn[splits["train"]], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(yn[splits["train"]], dtype=torch.float32, device=device)
    x_val = torch.as_tensor(Xn[splits["val"]], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(yn[splits["val"]], dtype=torch.float32, device=device)

    history_rows = []
    best_state = None
    best_val = float("inf")
    stale = 0
    rng = np.random.default_rng(int(config.seed))
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        order = rng.permutation(len(x_train))
        batch_losses = []
        for start in range(0, len(order), int(config.batch_size)):
            batch_idx = torch.as_tensor(order[start : start + int(config.batch_size)], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_train[batch_idx])
            loss = loss_fn(pred, y_train[batch_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip_norm))
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val).detach().cpu())
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1.0e-6:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(config.patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_tensor = model(torch.as_tensor(Xn[splits["test"]], dtype=torch.float32, device=device)).detach().cpu()
        pred_n = np.asarray(pred_tensor.tolist(), dtype=np.float32)
    y_pred = pred_n * y_std + y_mean
    y_true = y[splits["test"]]
    persistence_pred = X[splits["test"], -1, :]

    test_rmse = _rmse(y_true, y_pred)
    persistence_rmse = _rmse(y_true, persistence_pred)
    target_std = float(np.std(y_true, ddof=1))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    metrics = {
        "test_rmse": test_rmse,
        "test_nrmse": test_rmse / max(target_std, 1.0e-12),
        "test_r2": 1.0 - ss_res / max(ss_tot, 1.0e-12),
        "persistence_test_rmse": persistence_rmse,
        "persistence_test_nrmse": persistence_rmse / max(target_std, 1.0e-12),
        "rmse_improvement_vs_persistence": persistence_rmse - test_rmse,
        "best_val_loss": float(best_val),
        "epochs_ran": float(len(history_rows)),
    }

    split_summary_rows = []
    for name, slc in splits.items():
        size = len(range(*slc.indices(len(X))))
        row: dict[str, Any] = {"split": name, "samples": size}
        if windows.target_time is not None and size:
            times = windows.target_time[slc]
            row["start"] = times.min()
            row["end"] = times.max()
        split_summary_rows.append(row)

    return GruExperimentResult(
        metrics=metrics,
        history=pd.DataFrame(history_rows),
        y_true=y_true,
        y_pred=y_pred.astype(np.float32),
        persistence_pred=persistence_pred,
        split_summary=pd.DataFrame(split_summary_rows),
        model=model,
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
    )


def predict_gru_windows(result: GruExperimentResult, windows: np.ndarray) -> np.ndarray:
    """Predict next-state values from retained GRU result and raw history windows."""

    if result.model is None or result.x_mean is None or result.x_std is None or result.y_mean is None or result.y_std is None:
        raise ValueError("GruExperimentResult does not retain a fitted model and scalers.")
    import torch

    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("windows must have shape (samples, history, components).")
    scaled = (values - result.x_mean) / result.x_std
    result.model.eval()
    with torch.no_grad():
        pred_tensor = result.model(torch.as_tensor(scaled, dtype=torch.float32))
        pred = np.asarray(pred_tensor.detach().cpu().tolist(), dtype=np.float32)
    return pred * result.y_std + result.y_mean


def train_mlp_forecaster(windows: WindowedDataset, config: MlpTrainingConfig) -> GruExperimentResult:
    import torch

    warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*", category=UserWarning)
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    if int(config.num_threads) > 0:
        torch.set_num_threads(int(config.num_threads))

    X = np.asarray(windows.X, dtype=np.float32)
    y = np.asarray(windows.y, dtype=np.float32)
    splits = _temporal_split(len(X), config.train_fraction, config.val_fraction)

    train_slice = splits["train"]
    x_mean = X[train_slice].mean(axis=(0, 1), keepdims=True)
    x_std = np.maximum(X[train_slice].std(axis=(0, 1), keepdims=True), 1.0e-6)
    y_mean = y[train_slice].mean(axis=0, keepdims=True)
    y_std = np.maximum(y[train_slice].std(axis=0, keepdims=True), 1.0e-6)
    Xn = (X - x_mean) / x_std
    yn = (y - y_mean) / y_std

    class _MlpNet(torch.nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            super().__init__()
            layers: list[torch.nn.Module] = []
            current_dim = int(input_dim)
            for hidden_dim in tuple(int(value) for value in config.hidden_dims):
                layers.append(torch.nn.Linear(current_dim, hidden_dim))
                layers.append(torch.nn.ReLU())
                current_dim = hidden_dim
            layers.append(torch.nn.Linear(current_dim, int(output_dim)))
            self.net = torch.nn.Sequential(*layers)

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            return self.net(batch.reshape(batch.shape[0], -1))

    device = torch.device("cpu")
    model = _MlpNet(X.shape[1] * X.shape[2], X.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    loss_fn = torch.nn.MSELoss()

    x_train = torch.as_tensor(Xn[splits["train"]], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(yn[splits["train"]], dtype=torch.float32, device=device)
    x_val = torch.as_tensor(Xn[splits["val"]], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(yn[splits["val"]], dtype=torch.float32, device=device)

    history_rows = []
    best_state = None
    best_val = float("inf")
    stale = 0
    rng = np.random.default_rng(int(config.seed))
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        order = rng.permutation(len(x_train))
        batch_losses = []
        for start in range(0, len(order), int(config.batch_size)):
            batch_idx = torch.as_tensor(order[start : start + int(config.batch_size)], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_train[batch_idx])
            loss = loss_fn(pred, y_train[batch_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip_norm))
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val).detach().cpu())
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1.0e-6:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(config.patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_tensor = model(torch.as_tensor(Xn[splits["test"]], dtype=torch.float32, device=device)).detach().cpu()
        pred_n = np.asarray(pred_tensor.tolist(), dtype=np.float32)
    y_pred = pred_n * y_std + y_mean
    y_true = y[splits["test"]]
    persistence_pred = X[splits["test"], -1, :]

    test_rmse = _rmse(y_true, y_pred)
    persistence_rmse = _rmse(y_true, persistence_pred)
    target_std = float(np.std(y_true, ddof=1))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    metrics = {
        "test_rmse": test_rmse,
        "test_nrmse": test_rmse / max(target_std, 1.0e-12),
        "test_r2": 1.0 - ss_res / max(ss_tot, 1.0e-12),
        "persistence_test_rmse": persistence_rmse,
        "persistence_test_nrmse": persistence_rmse / max(target_std, 1.0e-12),
        "rmse_improvement_vs_persistence": persistence_rmse - test_rmse,
        "best_val_loss": float(best_val),
        "epochs_ran": float(len(history_rows)),
    }

    split_summary_rows = []
    for name, slc in splits.items():
        size = len(range(*slc.indices(len(X))))
        row: dict[str, Any] = {"split": name, "samples": size}
        if windows.target_time is not None and size:
            times = windows.target_time[slc]
            row["start"] = times.min()
            row["end"] = times.max()
        split_summary_rows.append(row)

    return GruExperimentResult(
        metrics=metrics,
        history=pd.DataFrame(history_rows),
        y_true=y_true,
        y_pred=y_pred.astype(np.float32),
        persistence_pred=persistence_pred,
        split_summary=pd.DataFrame(split_summary_rows),
        model=model,
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
    )


def predict_mlp_windows(result: GruExperimentResult, windows: np.ndarray) -> np.ndarray:
    """Predict next-state values from retained MLP result and raw history windows."""

    if result.model is None or result.x_mean is None or result.x_std is None or result.y_mean is None or result.y_std is None:
        raise ValueError("GruExperimentResult does not retain a fitted model and scalers.")
    import torch

    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("windows must have shape (samples, history, components).")
    scaled = (values - result.x_mean) / result.x_std
    result.model.eval()
    with torch.no_grad():
        pred_tensor = result.model(torch.as_tensor(scaled, dtype=torch.float32))
        pred = np.asarray(pred_tensor.detach().cpu().tolist(), dtype=np.float32)
    return pred * result.y_std + result.y_mean


def _discretize(values: np.ndarray, bins: int) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(-1)
    if len(flat) == 0:
        return np.asarray([], dtype=int)
    unique = np.unique(np.round(flat, decimals=10))
    if len(unique) <= 1:
        return np.zeros(len(flat), dtype=int)
    if 1 < len(unique) <= int(bins):
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
    """Discrete EI under a uniform source-state intervention over observed source states."""

    source_inverse = _state_codes(source_states)
    target_inverse = _state_codes(target_states)
    if len(source_inverse) != len(target_inverse):
        raise ValueError("source_states and target_states must have the same sample count.")
    if len(source_inverse) == 0:
        return 0.0
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


def sample_independent_intervention_windows(
    reference_windows: np.ndarray,
    *,
    samples: int,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    seed: int = 42,
) -> np.ndarray:
    """Sample independent uniform interventions for every lag/component coordinate."""

    reference = np.asarray(reference_windows, dtype=float)
    if reference.ndim != 3:
        raise ValueError("reference_windows must have shape (samples, history, components).")
    rng = np.random.default_rng(int(seed) + 2003)
    out = np.empty((int(samples), reference.shape[1], reference.shape[2]), dtype=np.float32)
    for lag_idx in range(reference.shape[1]):
        for component in range(reference.shape[2]):
            values = reference[:, lag_idx, component]
            low = float(np.quantile(values, float(quantile_low)))
            high = float(np.quantile(values, float(quantile_high)))
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low = float(np.min(values))
                high = float(np.max(values))
            if high <= low:
                out[:, lag_idx, component] = low
            else:
                out[:, lag_idx, component] = rng.uniform(low, high, size=int(samples))
    return out


def _component_source_states(intervention_windows: np.ndarray, *, bins: int, source_mode: str = "history") -> list[np.ndarray]:
    windows = np.asarray(intervention_windows, dtype=float)
    if source_mode not in {"history", "latest"}:
        raise ValueError("source_mode must be 'history' or 'latest'.")
    states: list[np.ndarray] = []
    for source in range(windows.shape[2]):
        if source_mode == "latest":
            states.append(_discretize(windows[:, -1, source], int(bins)))
        else:
            lag_states = [_discretize(windows[:, lag_idx, source], int(bins)) for lag_idx in range(windows.shape[1])]
            states.append(np.column_stack(lag_states))
    return states


def estimate_model_peid_graphs(
    result: GruExperimentResult,
    windows: WindowedDataset,
    *,
    prediction_fn: Callable[[GruExperimentResult, np.ndarray], np.ndarray],
    intervention_samples: int = 4096,
    bins: int = 5,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    top_synergy_sources_per_target: int = 8,
    extra_synergy_candidates: pd.DataFrame | None = None,
    null_reps: int = 64,
    source_mode: str = "history",
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate pairwise EI edges and second-order PEID interactions from model predictions."""

    intervention_windows = sample_independent_intervention_windows(
        windows.X,
        samples=int(intervention_samples),
        quantile_low=float(quantile_low),
        quantile_high=float(quantile_high),
        seed=int(seed),
    )
    predictions = np.asarray(prediction_fn(result, intervention_windows), dtype=float)
    if predictions.ndim != 2:
        raise ValueError("prediction_fn must return an array with shape (samples, components).")
    n_components = int(predictions.shape[1])
    names = [f"component_{idx + 1:02d}" for idx in range(n_components)]
    source_states = _component_source_states(intervention_windows, bins=int(bins), source_mode=str(source_mode))
    target_states = [_discretize(predictions[:, target], int(bins)) for target in range(n_components)]

    matrix = np.zeros((n_components, n_components), dtype=float)
    pairwise_rows: list[dict[str, Any]] = []
    for source in range(n_components):
        for target in range(n_components):
            ei = discrete_effective_information(source_states[source], target_states[target])
            matrix[source, target] = float(ei)
            pairwise_rows.append(
                {
                    "source": names[source],
                    "target": names[target],
                    "source_index": source,
                    "target_index": target,
                    "ei": float(ei),
                }
            )
    pairwise = pd.DataFrame(pairwise_rows).sort_values("ei", ascending=False).reset_index(drop=True)

    candidates: set[tuple[int, int, int]] = set()
    for target in range(n_components):
        ranked = np.argsort(matrix[:, target])[::-1][: max(2, int(top_synergy_sources_per_target))]
        for source_i, source_j in itertools.combinations(sorted(int(x) for x in ranked), 2):
            candidates.add((source_i, source_j, target))
    if extra_synergy_candidates is not None and not extra_synergy_candidates.empty:
        for row in extra_synergy_candidates.itertuples():
            source_i = int(getattr(row, "source_i"))
            source_j = int(getattr(row, "source_j"))
            target = int(getattr(row, "target_index"))
            candidates.add((*sorted((source_i, source_j)), target))

    rng = np.random.default_rng(int(seed) + 3001)
    synergy_rows: list[dict[str, Any]] = []
    for source_i, source_j, target in sorted(candidates):
        joint_source = np.column_stack([source_states[source_i], source_states[source_j]])
        joint_ei = discrete_effective_information(joint_source, target_states[target])
        synergy = float(joint_ei - matrix[source_i, target] - matrix[source_j, target])
        null_values: list[float] = []
        if int(null_reps) > 0:
            for _ in range(int(null_reps)):
                shuffled = rng.permutation(target_states[target])
                null_joint = discrete_effective_information(joint_source, shuffled)
                null_values.append(float(null_joint - matrix[source_i, target] - matrix[source_j, target]))
        p_value = float((1 + sum(value >= synergy for value in null_values)) / (1 + len(null_values))) if null_values else float("nan")
        synergy_rows.append(
            {
                "source_i": int(source_i),
                "source_j": int(source_j),
                "target_index": int(target),
                "sources": f"{names[source_i]}+{names[source_j]}",
                "target": names[target],
                "joint_ei": float(joint_ei),
                "source_i_ei": float(matrix[source_i, target]),
                "source_j_ei": float(matrix[source_j, target]),
                "synergy": synergy,
                "p_value": p_value,
            }
        )
    synergy_frame = pd.DataFrame(synergy_rows)
    if not synergy_frame.empty:
        synergy_frame = synergy_frame.sort_values(["p_value", "synergy"], ascending=[True, False]).reset_index(drop=True)
    return {
        "intervention_windows": intervention_windows,
        "predictions": predictions,
        "pairwise_matrix": matrix,
        "pairwise_edges": pairwise,
        "synergy_edges": synergy_frame,
    }


def estimate_gru_peid_graphs(
    result: GruExperimentResult,
    windows: WindowedDataset,
    *,
    intervention_samples: int = 4096,
    bins: int = 5,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    top_synergy_sources_per_target: int = 8,
    extra_synergy_candidates: pd.DataFrame | None = None,
    null_reps: int = 64,
    source_mode: str = "history",
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate pairwise EI edges and second-order PEID interactions from a GRU."""

    return estimate_model_peid_graphs(
        result,
        windows,
        prediction_fn=predict_gru_windows,
        intervention_samples=intervention_samples,
        bins=bins,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
        top_synergy_sources_per_target=top_synergy_sources_per_target,
        extra_synergy_candidates=extra_synergy_candidates,
        null_reps=null_reps,
        source_mode=source_mode,
        seed=seed,
    )


def estimate_gru_tm_ei_graphs(
    result: GruExperimentResult,
    windows: WindowedDataset,
    *,
    prediction_fn: Callable[[GruExperimentResult, np.ndarray], np.ndarray] | None = None,
    intervention_samples: int = 4096,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    source_mode: str = "history",
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate pairwise EI from GRU predictions with affine transport-map MI."""

    from exp.TM.transport_map_density import estimate_mutual_information_transport_map

    intervention_windows = sample_independent_intervention_windows(
        windows.X,
        samples=int(intervention_samples),
        quantile_low=float(quantile_low),
        quantile_high=float(quantile_high),
        seed=int(seed),
    )
    predictor = predict_gru_windows if prediction_fn is None else prediction_fn
    predictions = np.asarray(predictor(result, intervention_windows), dtype=float)
    if predictions.ndim != 2:
        raise ValueError("prediction_fn must return an array with shape (samples, components).")
    n_components = int(predictions.shape[1])
    names = [f"component_{idx + 1:02d}" for idx in range(n_components)]

    if source_mode not in {"history", "latest"}:
        raise ValueError("source_mode must be 'history' or 'latest'.")
    source_states: list[np.ndarray] = []
    for source in range(n_components):
        if source_mode == "latest":
            source_states.append(intervention_windows[:, -1, [source]])
        else:
            source_states.append(intervention_windows[:, :, source])

    matrix = np.zeros((n_components, n_components), dtype=float)
    rows: list[dict[str, Any]] = []
    for source in range(n_components):
        for target in range(n_components):
            summary = estimate_mutual_information_transport_map(source_states[source], predictions[:, [target]])
            ei = max(0.0, float(summary["mi_hat"]))
            matrix[source, target] = ei
            rows.append(
                {
                    "source": names[source],
                    "target": names[target],
                    "source_index": source,
                    "target_index": target,
                    "ei": ei,
                    "bias_correction": float(summary["bias_correction"]),
                }
            )

    return {
        "intervention_windows": intervention_windows,
        "predictions": predictions,
        "pairwise_matrix": matrix,
        "pairwise_edges": pd.DataFrame(rows).sort_values("ei", ascending=False).reset_index(drop=True),
    }


def pairwise_recovery_summary(pairwise_edges: pd.DataFrame, ground_truth_edges: pd.DataFrame, *, top_k: int = 240) -> dict[str, float]:
    truth = set(ground_truth_edges[["source_index", "target_index"]].itertuples(index=False, name=None))
    ranked = pairwise_edges.sort_values("ei", ascending=False).head(int(top_k))
    predicted = set(ranked[["source_index", "target_index"]].itertuples(index=False, name=None))
    true_positive = len(predicted & truth)
    precision = true_positive / max(len(predicted), 1)
    recall = true_positive / max(len(truth), 1)
    return {
        "top_k": float(int(top_k)),
        "ground_truth_edges": float(len(truth)),
        "true_positive_edges": float(true_positive),
        "precision_at_k": float(precision),
        "recall_at_k": float(recall),
    }


def ground_truth_strength_matrix(pairwise_matrix: np.ndarray, ground_truth_edges: pd.DataFrame) -> np.ndarray:
    values = np.asarray(pairwise_matrix, dtype=float)
    truth = np.zeros_like(values, dtype=float)
    for row in ground_truth_edges.itertuples():
        coefficient = float(getattr(row, "coefficient", 1.0))
        truth[int(row.source_index), int(row.target_index)] = abs(coefficient)
    return truth


def plot_pairwise_recovery_heatmap(pairwise_matrix: np.ndarray, ground_truth_edges: pd.DataFrame) -> plt.Figure:
    values = np.asarray(pairwise_matrix, dtype=float)
    truth = ground_truth_strength_matrix(values, ground_truth_edges)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    truth_vmax = max(float(np.nanmax(truth)), 1.0e-12)
    im0 = axes[0].imshow(truth.T, cmap="Greys", vmin=0.0, vmax=truth_vmax, interpolation="nearest", aspect="auto")
    axes[0].set_xlabel("Source component")
    axes[0].set_ylabel("Target component")
    axes[0].set_title("Ground-truth strength")
    fig.colorbar(im0, ax=axes[0], shrink=0.80, label="|VAR coefficient|")

    vmax = float(np.nanquantile(values, 0.995)) if values.size else 1.0
    im1 = axes[1].imshow(values.T, cmap="viridis", vmin=0.0, vmax=max(vmax, 1.0e-12), interpolation="nearest", aspect="auto")
    axes[1].set_xlabel("Source component")
    axes[1].set_ylabel("Target component")
    axes[1].set_title("GRU + PEID pairwise EI")
    fig.colorbar(im1, ax=axes[1], shrink=0.80, label="EI (bit)")
    for ax in axes:
        ticks = np.arange(0, values.shape[0], 10)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks + 1)
        ax.set_yticklabels(ticks + 1)
    return fig


def plot_pairwise_recovery_triptych(
    gru_matrix: np.ndarray,
    mlp_matrix: np.ndarray,
    ground_truth_edges: pd.DataFrame,
) -> plt.Figure:
    """Show ground truth, GRU+PEID, and MLP+PEID matrices with comparable axes."""

    gru_values = np.asarray(gru_matrix, dtype=float)
    mlp_values = np.asarray(mlp_matrix, dtype=float)
    truth = ground_truth_strength_matrix(gru_values, ground_truth_edges)
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4), constrained_layout=True)
    truth_vmax = max(float(np.nanmax(truth)), 1.0e-12)
    im0 = axes[0].imshow(truth.T, cmap="Greys", vmin=0.0, vmax=truth_vmax, interpolation="nearest", aspect="auto")
    axes[0].set_title("Ground truth")
    fig.colorbar(im0, ax=axes[0], shrink=0.80, label="|VAR coefficient|")

    vmax = max(float(np.nanquantile(np.concatenate([gru_values.ravel(), mlp_values.ravel()]), 0.995)), 1.0e-12)
    for ax, values, title in (
        (axes[1], gru_values, "GRU + PEID"),
        (axes[2], mlp_values, "MLP + PEID"),
    ):
        im = ax.imshow(values.T, cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest", aspect="auto")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.80, label="EI (bit)")

    for ax in axes:
        ticks = np.arange(0, gru_values.shape[0], 2 if gru_values.shape[0] <= 16 else 10)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks + 1)
        ax.set_yticklabels(ticks + 1)
        ax.set_xlabel("Source component")
    axes[0].set_ylabel("Target component")
    return fig


def plot_pairwise_causal_graph_comparison(
    pairwise_edges: pd.DataFrame,
    ground_truth_edges: pd.DataFrame,
    *,
    top_k: int = 80,
    n_components: int | None = None,
    seed: int = 42,
) -> plt.Figure:
    import networkx as nx

    recovered = pairwise_edges.sort_values("ei", ascending=False).head(int(top_k)).copy()
    truth_pairs = set(ground_truth_edges[["source_index", "target_index"]].itertuples(index=False, name=None))
    recovered_pairs = set(recovered[["source_index", "target_index"]].itertuples(index=False, name=None))
    selected_pairs = recovered_pairs
    if n_components is None:
        n_components = int(max(max(pair) for pair in selected_pairs)) + 1 if selected_pairs else 0
    nodes = sorted(set(itertools.chain.from_iterable(selected_pairs)))
    labels = {node: f"C{node + 1}" for node in nodes}
    graph_for_layout = nx.DiGraph()
    graph_for_layout.add_nodes_from(nodes)
    graph_for_layout.add_edges_from(selected_pairs)
    pos = nx.spring_layout(graph_for_layout, seed=int(seed), k=0.9, iterations=200) if nodes else {}

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), constrained_layout=True)
    panels = [
        (axes[0], ground_truth_edges.copy(), "Ground truth parents", "#555555", "ground_truth"),
        (axes[1], recovered, f"Top {int(top_k)} GRU + PEID edges", "#3B73B9", "recovered"),
    ]
    for ax, edges, title, color, mode in panels:
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for row in edges.itertuples():
            pair = (int(row.source_index), int(row.target_index))
            if (mode == "ground_truth" and pair[0] in nodes and pair[1] in nodes) or (mode != "ground_truth" and pair in selected_pairs):
                graph.add_edge(pair[0], pair[1], weight=float(getattr(row, "ei", 1.0)))
        weights = np.asarray([graph[u][v].get("weight", 1.0) for u, v in graph.edges()], dtype=float)
        if len(weights):
            widths = 0.6 + 2.8 * weights / max(float(weights.max()), 1.0e-12)
        else:
            widths = []
        edge_colors = []
        for u, v in graph.edges():
            if mode == "recovered" and (u, v) in truth_pairs:
                edge_colors.append("#2E7D32")
            elif mode == "recovered":
                edge_colors.append("#9E9E9E")
            else:
                edge_colors.append(color)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=260, node_color="#F4F4F4", edgecolors="#333333", linewidths=0.7)
        nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=6)
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            width=widths,
            edge_color=edge_colors,
            arrows=True,
            arrowsize=8,
            node_size=260,
            connectionstyle="arc3,rad=0.08",
            alpha=0.82,
        )
        ax.set_title(title)
        ax.set_axis_off()
    return fig


def plot_pairwise_causal_graph_triptych(
    gru_edges: pd.DataFrame,
    mlp_edges: pd.DataFrame,
    ground_truth_edges: pd.DataFrame,
    *,
    top_k: int = 80,
    n_components: int | None = None,
    seed: int = 42,
) -> plt.Figure:
    """Place ground-truth, GRU+PEID, and MLP+PEID causal graphs in one figure."""

    import networkx as nx

    if n_components is None:
        max_index = int(
            max(
                ground_truth_edges[["source_index", "target_index"]].to_numpy(dtype=int).ravel().max(),
                gru_edges[["source_index", "target_index"]].to_numpy(dtype=int).ravel().max(),
                mlp_edges[["source_index", "target_index"]].to_numpy(dtype=int).ravel().max(),
            )
        )
        n_components = max_index + 1

    truth_pairs = set(ground_truth_edges[["source_index", "target_index"]].itertuples(index=False, name=None))
    nodes = list(range(int(n_components)))
    labels = {node: f"C{node + 1}" for node in nodes}
    layout_graph = nx.DiGraph()
    layout_graph.add_nodes_from(nodes)
    layout_graph.add_edges_from(truth_pairs)
    pos = nx.circular_layout(layout_graph) if int(n_components) <= 16 else nx.spring_layout(layout_graph, seed=int(seed), k=0.9, iterations=200)

    recovered_gru = gru_edges.sort_values("ei", ascending=False).head(int(top_k)).copy()
    recovered_mlp = mlp_edges.sort_values("ei", ascending=False).head(int(top_k)).copy()
    panels = [
        (ground_truth_edges.copy(), "Ground truth", "truth"),
        (recovered_gru, f"Top {int(top_k)} GRU + PEID edges", "recovered"),
        (recovered_mlp, f"Top {int(top_k)} MLP + PEID edges", "recovered"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4), constrained_layout=True)
    for ax, edges, title, mode in zip(axes, [p[0] for p in panels], [p[1] for p in panels], [p[2] for p in panels]):
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for row in edges.itertuples():
            graph.add_edge(int(row.source_index), int(row.target_index), weight=float(getattr(row, "ei", abs(float(getattr(row, "coefficient", 1.0))))))
        weights = np.asarray([graph[u][v].get("weight", 1.0) for u, v in graph.edges()], dtype=float)
        widths = 0.7 + 2.5 * weights / max(float(weights.max()), 1.0e-12) if len(weights) else []
        edge_colors = []
        for u, v in graph.edges():
            if mode == "truth":
                edge_colors.append("#555555")
            elif (u, v) in truth_pairs:
                edge_colors.append("#2E7D32")
            else:
                edge_colors.append("#9E9E9E")

        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=250, node_color="#F7F7F7", edgecolors="#333333", linewidths=0.7)
        nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=6)
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            width=widths,
            edge_color=edge_colors,
            arrows=True,
            arrowsize=8,
            node_size=250,
            connectionstyle="arc3,rad=0.10",
            alpha=0.82,
        )
        ax.set_title(title)
        ax.set_axis_off()
    return fig


def plot_gateway_topology(metadata: dict[str, Any]) -> plt.Figure:
    """Plot the designed ground-truth topology and highlight the causal gateway."""

    import networkx as nx

    edges = ground_truth_pairwise_edges(metadata, include_self=False, source_mode="latest")
    n_components = int(metadata["n_components"])
    labels = metadata.get("module_labels", {})
    gateway_index = int(metadata.get("gateway_index", 0))
    nodes = list(range(n_components))
    node_labels = {node: f"C{node + 1}" for node in nodes}
    pos = {}
    for idx in range(6):
        pos[idx] = (-1.0, 1.0 - 2.0 * idx / 5.0)
        pos[idx + 6] = (1.0, 1.0 - 2.0 * idx / 5.0)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for row in edges.itertuples():
        graph.add_edge(int(row.source_index), int(row.target_index), weight=abs(float(getattr(row, "coefficient", 1.0))), edge_kind=str(getattr(row, "edge_kind", "")))

    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    node_colors = []
    for node in nodes:
        name = f"component_{node + 1:02d}"
        if node == gateway_index:
            node_colors.append("#D62728")
        elif labels.get(name) == "A":
            node_colors.append("#A6CEE3")
        else:
            node_colors.append("#B2DF8A")
    edge_colors = ["#D62728" if graph[u][v].get("edge_kind") == "gateway_cross_module" else "#6E6E6E" for u, v in graph.edges()]
    weights = np.asarray([graph[u][v].get("weight", 1.0) for u, v in graph.edges()], dtype=float)
    widths = 0.7 + 2.4 * weights / max(float(weights.max()), 1.0e-12) if len(weights) else []

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=430, node_color=node_colors, edgecolors="#333333", linewidths=0.8)
    nx.draw_networkx_labels(graph, pos, labels=node_labels, ax=ax, font_size=8)
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        width=widths,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=10,
        node_size=430,
        connectionstyle="arc3,rad=0.08",
        alpha=0.82,
    )
    ax.text(-1.0, 1.22, "Module A", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(1.0, 1.22, "Module B", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Designed nonlinear gateway topology")
    ax.set_axis_off()
    return fig


def plot_top_synergy_hypergraph(
    synergy_edges: pd.DataFrame,
    ground_truth_synergy: pd.DataFrame,
    *,
    top_k: int = 12,
    p_threshold: float = 0.05,
) -> plt.Figure:
    truth = set(ground_truth_synergy[["source_i", "source_j", "target_index"]].itertuples(index=False, name=None))
    frame = synergy_edges.copy()
    if "p_value" in frame:
        selected = frame[(frame["synergy"] > 0.0) & (frame["p_value"] <= float(p_threshold))]
        if selected.empty:
            selected = frame[frame["synergy"] > 0.0]
    else:
        selected = frame[frame["synergy"] > 0.0]
    selected = selected.sort_values(["p_value", "synergy"] if "p_value" in selected else ["synergy"], ascending=[True, False] if "p_value" in selected else [False]).head(int(top_k))

    fig, ax = plt.subplots(figsize=(9.6, max(4.4, 0.38 * max(len(selected), 1) + 1.8)), constrained_layout=True)
    if selected.empty:
        ax.text(0.5, 0.5, "No positive synergy edge selected", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    labels = [f"{{C{int(row.source_i) + 1}, C{int(row.source_j) + 1}}} -> C{int(row.target_index) + 1}" for row in selected.itertuples()]
    values = selected["synergy"].to_numpy(dtype=float)
    y = np.arange(len(selected))
    colors = [
        "#2E7D32" if tuple(sorted((int(row.source_i), int(row.source_j))) + [int(row.target_index)]) in truth else "#C66A2E"
        for row in selected.itertuples()
    ]
    ax.barh(y, values, color=colors, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Second-order PEID interaction, Delta({source pair} -> target) [bit]")
    ax.set_title("Top positive GRU + PEID synergies")
    max_value = max(float(values.max()), 1.0e-12)
    for idx, row in enumerate(selected.itertuples()):
        p_text = f"p={float(row.p_value):.3f}" if hasattr(row, "p_value") and np.isfinite(float(row.p_value)) else ""
        ax.text(float(row.synergy) + 0.02 * max_value, idx, p_text, va="center", fontsize=6.5)
    ax.set_xlim(0.0, max_value * 1.22)
    return fig


def compute_series_diagnostics(frame: pd.DataFrame) -> dict[str, float]:
    values = frame.to_numpy(dtype=float)
    diagnostics: dict[str, float] = {
        "n_samples": float(values.shape[0]),
        "n_components": float(values.shape[1]),
        "std_mean": float(np.std(values, axis=0, ddof=1).mean()),
        "abs_p99": float(np.percentile(np.abs(values), 99.0)),
        "abs_max": float(np.max(np.abs(values))),
    }
    for lag in (1, 2, 4, 8):
        if len(values) <= lag:
            diagnostics[f"lag{lag}_autocorr_mean"] = float("nan")
            diagnostics[f"lag{lag}_autocorr_abs_median"] = float("nan")
            continue
        ac = []
        for col in range(values.shape[1]):
            left = values[:-lag, col]
            right = values[lag:, col]
            if np.std(left) > 0.0 and np.std(right) > 0.0:
                ac.append(float(np.corrcoef(left, right)[0, 1]))
        diagnostics[f"lag{lag}_autocorr_mean"] = float(np.mean(ac))
        diagnostics[f"lag{lag}_autocorr_abs_median"] = float(np.median(np.abs(ac)))
    corr = np.corrcoef(values, rowvar=False)
    offdiag = corr[~np.eye(corr.shape[0], dtype=bool)]
    diagnostics["offdiag_corr_abs_mean"] = float(np.mean(np.abs(offdiag)))
    diagnostics["offdiag_corr_abs_q95"] = float(np.quantile(np.abs(offdiag), 0.95))
    return diagnostics


def diagnostics_table(real_frame: pd.DataFrame, synthetic_frame: pd.DataFrame) -> pd.DataFrame:
    real = compute_series_diagnostics(real_frame)
    synthetic = compute_series_diagnostics(synthetic_frame)
    rows = []
    labels = {
        "std_mean": "Mean component standard deviation",
        "lag1_autocorr_mean": "Mean lag-1 autocorrelation",
        "lag4_autocorr_mean": "Mean lag-4 autocorrelation",
        "offdiag_corr_abs_mean": "Mean off-diagonal |correlation|",
        "offdiag_corr_abs_q95": "95th percentile off-diagonal |correlation|",
        "abs_p99": "99th percentile |score|",
        "abs_max": "Maximum |score|",
    }
    for key, label in labels.items():
        rows.append({"Metric": label, "Real Runge-NCEP": real[key], "Synthetic dynamics": synthetic[key], "Absolute difference": abs(real[key] - synthetic[key])})
    return pd.DataFrame(rows)


def metrics_table(result: GruExperimentResult) -> pd.DataFrame:
    keys = [
        "test_rmse",
        "test_nrmse",
        "test_r2",
        "persistence_test_rmse",
        "persistence_test_nrmse",
        "rmse_improvement_vs_persistence",
        "best_val_loss",
        "epochs_ran",
    ]
    labels = {
        "test_rmse": "Test RMSE",
        "test_nrmse": "Test NRMSE",
        "test_r2": "Test R2",
        "persistence_test_rmse": "Persistence test RMSE",
        "persistence_test_nrmse": "Persistence test NRMSE",
        "rmse_improvement_vs_persistence": "RMSE improvement vs persistence",
        "best_val_loss": "Best validation loss",
        "epochs_ran": "Training epochs",
    }
    return pd.DataFrame({"Metric": [labels[key] for key in keys], "Value": [result.metrics[key] for key in keys]})


def plot_real_synthetic_overview(real_frame: pd.DataFrame, synthetic_frame: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
    for ax, frame, title in zip(axes, (real_frame, synthetic_frame), ("Real Runge-NCEP 60D", "Known synthetic dynamics")):
        values = frame.to_numpy(dtype=float).T
        vlim = float(np.nanpercentile(np.abs(values), 99.0))
        im = ax.imshow(values, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax.set_title(title)
        ax.set_xlabel("Week index")
        ax.set_ylabel("Component")
        ax.set_yticks(np.arange(0, frame.shape[1], 10))
        ax.set_yticklabels(np.arange(1, frame.shape[1] + 1, 10))
        fig.colorbar(im, ax=ax, shrink=0.82, label="Score value")
    return fig


def plot_training_history(result: GruExperimentResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 3.4), constrained_layout=True)
    ax.plot(result.history["epoch"], result.history["train_loss"], label="Train", color="#4C78A8")
    ax.plot(result.history["epoch"], result.history["val_loss"], label="Validation", color="#D62728")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE on standardized target")
    ax.set_title("GRU learning curve")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return fig


def plot_prediction_quality(result: GruExperimentResult, *, max_points: int = 3500, seed: int = 42) -> plt.Figure:
    y_true = result.y_true.reshape(-1)
    y_pred = result.y_pred.reshape(-1)
    if len(y_true) > int(max_points):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(len(y_true), size=int(max_points), replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    fig, ax = plt.subplots(figsize=(4.7, 4.2), constrained_layout=True)
    ax.scatter(y_true, y_pred, s=7, alpha=0.28, color="#4C78A8", linewidths=0)
    limit = float(max(np.max(np.abs(y_true)), np.max(np.abs(y_pred))))
    ax.plot([-limit, limit], [-limit, limit], color="0.25", lw=1.0, label="Ideal prediction")
    ax.set_xlabel("True next-step score")
    ax.set_ylabel("GRU predicted next-step score")
    ax.set_title("Test one-step prediction scatter")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return fig
