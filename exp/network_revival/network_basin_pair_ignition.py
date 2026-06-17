"""Whole-network basin switching by pair ignition on synthetic networks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from itertools import combinations
from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from exp.network_revival.dynamics import get_model
from exp.network_revival.effective_information import estimate_mutual_information_transport_map
from exp.network_revival.joint_required_ignition import (
    _conditional_mi,
    _matrix_from_pairs,
    _mutual_information,
    _save_figure,
)
from exp.network_revival.network import build_er, largest_connected_component
from exp.network_revival.simulate import _rk4_step


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "network_revival_network_basin_pair_ignition"
DEFAULT_INITIAL_STATE_PROXY_OUTPUT_DIR = REPO_ROOT / "results" / "network_revival_initial_state_syn_proxy"
DEFAULT_TM_INITIAL_STATE_OUTPUT_DIR = REPO_ROOT / "results" / "network_revival_transport_map_initial_state_syn"

__all__ = [
    "NetworkBasinPairIgnitionConfig",
    "InitialStateSynProxyConfig",
    "TransportMapInitialStateSynConfig",
    "build_candidate_network",
    "simulate_released_ignition_batch",
    "run_network_basin_pair_ensemble",
    "plot_network_basin_results",
    "run_initial_state_syn_proxy_experiment",
    "plot_initial_state_syn_proxy_results",
    "run_transport_map_initial_state_syn_experiment",
    "plot_transport_map_initial_state_syn_results",
]


@dataclass(frozen=True)
class NetworkBasinPairIgnitionConfig:
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    node_count: int = 14
    model_names: tuple[str, ...] = ("Neural", "Eco")
    network_kinds: tuple[str, ...] = ("ER", "WS")
    accepted_instances_per_group: int = 10
    candidate_seed_count: int = 80
    er_avg_degree: float = 3.5
    ws_degree: int = 4
    ws_rewire_probability: float = 0.2
    coupling_scales: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.8, 2.6, 3.8, 5.5, 8.0)
    delta_max_values: tuple[float, ...] = (0.2, 0.5, 1.0, 1.5, 2.5, 4.0, 6.0, 8.0)
    delta_levels: int = 8
    t_force: float = 4.0
    t_free: float = 12.0
    dt: float = 0.08
    state_clip: float = 50.0
    min_basin_separation: float = 0.15
    min_successful_pairs: int = 3
    min_success_rate_std: float = 1e-6
    seed: int = 20260616
    neural_parameters: dict[str, float] = field(default_factory=lambda: {"mu": 3.0, "delta": 1.0})
    eco_parameters: dict[str, float] = field(default_factory=dict)
    high_initial_by_model: dict[str, float] = field(default_factory=lambda: {"Neural": 12.0, "Eco": 3.0})

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.node_count < 4:
            raise ValueError("node_count must be at least four.")
        if self.accepted_instances_per_group < 1 or self.candidate_seed_count < 1:
            raise ValueError("instance counts must be positive.")
        if self.delta_levels < 3:
            raise ValueError("delta_levels must be at least three.")
        if self.t_force < 0.0 or self.t_free < 0.0 or self.dt <= 0.0:
            raise ValueError("time parameters must be nonnegative and dt must be positive.")
        if self.state_clip <= 0.0:
            raise ValueError("state_clip must be positive.")
        if self.ws_degree < 2 or self.ws_degree >= self.node_count or self.ws_degree % 2 != 0:
            raise ValueError("ws_degree must be even and lie in [2, node_count).")
        if not self.model_names or not self.network_kinds:
            raise ValueError("model_names and network_kinds must be nonempty.")


@dataclass(frozen=True)
class InitialStateSynProxyConfig:
    output_dir: Path = field(default_factory=lambda: DEFAULT_INITIAL_STATE_PROXY_OUTPUT_DIR)
    source_arrays_npz: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "representative_arrays.npz")
    model_names: tuple[str, ...] = ("Neural", "Eco")
    network_kinds: tuple[str, ...] = ("ER", "WS")
    sample_count: int = 2048
    source_bins: int = 4
    t_short: float = 2.0
    dt: float = 0.08
    state_clip: float = 50.0
    seed: int = 20260616
    neural_parameters: dict[str, float] = field(default_factory=lambda: {"mu": 3.0, "delta": 1.0})
    eco_parameters: dict[str, float] = field(default_factory=dict)
    initial_state_high_by_model: dict[str, float] = field(default_factory=lambda: {"Neural": 8.0, "Eco": 3.0})
    min_valid_fraction: float = 0.95

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "source_arrays_npz", Path(self.source_arrays_npz))
        if self.sample_count < 8:
            raise ValueError("sample_count must be at least eight.")
        if self.source_bins < 2:
            raise ValueError("source_bins must be at least two.")
        if self.t_short <= 0.0 or self.dt <= 0.0:
            raise ValueError("t_short and dt must be positive.")
        if self.state_clip <= 0.0:
            raise ValueError("state_clip must be positive.")
        if not 0.0 < self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must lie in (0, 1].")
        if not self.model_names or not self.network_kinds:
            raise ValueError("model_names and network_kinds must be nonempty.")


@dataclass(frozen=True)
class TransportMapInitialStateSynConfig:
    output_dir: Path = field(default_factory=lambda: DEFAULT_TM_INITIAL_STATE_OUTPUT_DIR)
    source_arrays_npz: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "representative_arrays.npz")
    model_names: tuple[str, ...] = ("Neural", "Eco")
    network_kinds: tuple[str, ...] = ("ER", "WS")
    sample_count: int = 512
    pair_count_per_group: int = 30
    t_short: float = 2.0
    dt: float = 0.08
    state_clip: float = 50.0
    seed: int = 20260616
    transport_degree: int = 3
    clip_negative_ei: bool = True
    neural_parameters: dict[str, float] = field(default_factory=lambda: {"mu": 3.0, "delta": 1.0})
    eco_parameters: dict[str, float] = field(default_factory=dict)
    initial_state_high_by_model: dict[str, float] = field(default_factory=lambda: {"Neural": 8.0, "Eco": 3.0})
    min_valid_fraction: float = 0.95

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "source_arrays_npz", Path(self.source_arrays_npz))
        if self.sample_count < 16:
            raise ValueError("sample_count must be at least sixteen.")
        if self.pair_count_per_group < 1:
            raise ValueError("pair_count_per_group must be positive.")
        if self.t_short <= 0.0 or self.dt <= 0.0:
            raise ValueError("t_short and dt must be positive.")
        if self.state_clip <= 0.0:
            raise ValueError("state_clip must be positive.")
        if self.transport_degree < 1:
            raise ValueError("transport_degree must be positive.")
        if not 0.0 < self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must lie in (0, 1].")
        if not self.model_names or not self.network_kinds:
            raise ValueError("model_names and network_kinds must be nonempty.")


def _model_for_name(model_name: str, config: NetworkBasinPairIgnitionConfig) -> dict[str, Any]:
    if model_name == "Neural":
        return get_model("Neural", **config.neural_parameters)
    if model_name == "Eco":
        return get_model("Eco", **config.eco_parameters)
    raise ValueError(f"Unsupported network basin model {model_name!r}.")


def _proxy_model_for_name(model_name: str, config: InitialStateSynProxyConfig) -> dict[str, Any]:
    if model_name == "Neural":
        return get_model("Neural", **config.neural_parameters)
    if model_name == "Eco":
        return get_model("Eco", **config.eco_parameters)
    raise ValueError(f"Unsupported initial-state proxy model {model_name!r}.")


def _tm_model_for_name(model_name: str, config: TransportMapInitialStateSynConfig) -> dict[str, Any]:
    if model_name == "Neural":
        return get_model("Neural", **config.neural_parameters)
    if model_name == "Eco":
        return get_model("Eco", **config.eco_parameters)
    raise ValueError(f"Unsupported transport-map initial-state model {model_name!r}.")


def build_candidate_network(
    network_kind: str,
    config: NetworkBasinPairIgnitionConfig,
    *,
    seed: int,
    coupling_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a connected synthetic network and normalize mean weighted degree."""

    kind = str(network_kind)
    rng = np.random.default_rng(int(seed))
    adjacency: np.ndarray | None = None
    for attempt in range(50):
        if kind == "ER":
            candidate = build_er(config.node_count, config.er_avg_degree, rng=rng)
            graph = nx.from_numpy_array(candidate)
            if nx.is_connected(graph):
                adjacency = candidate
                break
        elif kind == "WS":
            graph = nx.watts_strogatz_graph(
                config.node_count,
                config.ws_degree,
                config.ws_rewire_probability,
                seed=int(seed) + attempt,
            )
            if nx.is_connected(graph):
                adjacency = nx.to_numpy_array(graph, dtype=float)
                break
        else:
            raise ValueError(f"Unsupported network kind {network_kind!r}.")

    if adjacency is None:
        if kind == "ER":
            candidate = build_er(config.node_count, config.er_avg_degree, rng=rng)
            adjacency, _ = largest_connected_component(candidate)
        else:
            graph = nx.connected_watts_strogatz_graph(
                config.node_count,
                config.ws_degree,
                config.ws_rewire_probability,
                tries=100,
                seed=int(seed),
            )
            adjacency = nx.to_numpy_array(graph, dtype=float)

    mean_degree = float(adjacency.sum(axis=1).mean())
    if mean_degree <= 0.0:
        raise RuntimeError("candidate network has zero mean degree.")
    adjacency = np.asarray(adjacency, dtype=float) / mean_degree * float(coupling_scale)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency, {
        "network_kind": kind,
        "seed": int(seed),
        "node_count": int(adjacency.shape[0]),
        "coupling_scale": float(coupling_scale),
        "mean_weighted_degree": float(adjacency.sum(axis=1).mean()),
    }


def simulate_released_ignition_batch(
    adjacency: np.ndarray,
    model: dict[str, Any],
    fixed_mask: np.ndarray,
    fixed_values: np.ndarray,
    *,
    t_force: float,
    t_free: float,
    dt: float,
    state_clip: float,
    initial_states: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Batch simulate fixed ignition followed by free release."""

    adjacency = np.asarray(adjacency, dtype=float)
    fixed_mask = np.asarray(fixed_mask, dtype=bool)
    fixed_values = np.asarray(fixed_values, dtype=float)
    if fixed_mask.shape != fixed_values.shape:
        raise ValueError("fixed_mask and fixed_values must have matching shape.")
    if fixed_mask.ndim != 2 or fixed_mask.shape[1] != adjacency.shape[0]:
        raise ValueError("fixed arrays must have shape [condition, node].")
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square.")

    batch_count, node_count = fixed_mask.shape
    if initial_states is None:
        x = np.zeros((batch_count, node_count), dtype=float)
    else:
        x = np.asarray(initial_states, dtype=float).copy()
        if x.shape == (node_count,):
            x = np.repeat(x[None, :], batch_count, axis=0)
        if x.shape != fixed_mask.shape:
            raise ValueError("initial_states must have shape [condition, node] or [node].")
    x = np.where(fixed_mask, fixed_values, x)
    valid = np.ones(batch_count, dtype=bool)
    m0 = model["M0"]
    m1 = model["M1"]
    m2 = model["M2"]

    def _interaction(states: np.ndarray) -> np.ndarray:
        return m2(states) @ adjacency.T

    def _mark_and_clip(states: np.ndarray) -> np.ndarray:
        nonlocal valid
        finite = np.isfinite(states).all(axis=1)
        bounded = np.nanmax(np.abs(np.nan_to_num(states, nan=0.0, posinf=np.inf, neginf=np.inf)), axis=1) <= float(state_clip)
        valid &= finite & bounded
        return np.clip(np.nan_to_num(states, nan=0.0, posinf=float(state_clip), neginf=0.0), 0.0, float(state_clip))

    def rhs_forced(_, states: np.ndarray) -> np.ndarray:
        forced = np.where(fixed_mask, fixed_values, states)
        dx = m0(forced) + m1(forced) * _interaction(forced)
        return np.where(fixed_mask, 0.0, dx)

    def rhs_free(_, states: np.ndarray) -> np.ndarray:
        return m0(states) + m1(states) * _interaction(states)

    t = 0.0
    while t < float(t_force) - 1e-12:
        step = min(float(dt), float(t_force) - t)
        x = _rk4_step(rhs_forced, t, x, step)
        x = np.where(fixed_mask, fixed_values, x)
        x = _mark_and_clip(x)
        t += step

    release_t = 0.0
    while release_t < float(t_free) - 1e-12:
        step = min(float(dt), float(t_free) - release_t)
        x = _rk4_step(rhs_free, release_t, x, step)
        x = _mark_and_clip(x)
        release_t += step

    return {
        "final_states": x,
        "final_mean": x.mean(axis=1),
        "valid": valid,
    }


def _free_attractor_means(
    adjacency: np.ndarray,
    model: dict[str, Any],
    model_name: str,
    config: NetworkBasinPairIgnitionConfig,
) -> dict[str, float | bool]:
    node_count = adjacency.shape[0]
    masks = np.zeros((2, node_count), dtype=bool)
    values = np.zeros((2, node_count), dtype=float)
    initial = np.vstack(
        [
            np.zeros(node_count, dtype=float),
            np.full(node_count, float(config.high_initial_by_model.get(model_name, 8.0)), dtype=float),
        ]
    )
    result = simulate_released_ignition_batch(
        adjacency,
        model,
        masks,
        values,
        t_force=0.0,
        t_free=config.t_free,
        dt=config.dt,
        state_clip=config.state_clip,
        initial_states=initial,
    )
    means = np.asarray(result["final_mean"], dtype=float)
    valid = bool(np.all(result["valid"]))
    low = float(min(means))
    high = float(max(means))
    return {
        "valid": valid and high - low >= float(config.min_basin_separation),
        "low_mean": low,
        "high_mean": high,
        "basin_threshold": 0.5 * (low + high),
    }


def _fixed_conditions(node_count: int, sources: list[tuple[int, ...]], values: list[tuple[float, ...]]) -> tuple[np.ndarray, np.ndarray]:
    masks = np.zeros((len(sources), node_count), dtype=bool)
    fixed_values = np.zeros_like(masks, dtype=float)
    for row, (nodes, node_values) in enumerate(zip(sources, values, strict=True)):
        for node, value in zip(nodes, node_values, strict=True):
            masks[row, int(node)] = True
            fixed_values[row, int(node)] = float(value)
    return masks, fixed_values


def _screen_candidate_instance(
    model_name: str,
    network_kind: str,
    seed: int,
    config: NetworkBasinPairIgnitionConfig,
) -> dict[str, Any] | None:
    model = _model_for_name(model_name, config)
    for scale in config.coupling_scales:
        adjacency, network_meta = build_candidate_network(network_kind, config, seed=seed, coupling_scale=float(scale))
        basin = _free_attractor_means(adjacency, model, model_name, config)
        if not bool(basin["valid"]):
            continue
        pairs = list(combinations(range(adjacency.shape[0]), 2))
        for delta_max in config.delta_max_values:
            single_sources = [(node,) for node in range(adjacency.shape[0])]
            single_values = [(2.0 * float(delta_max),) for _ in single_sources]
            pair_sources = [tuple(pair) for pair in pairs]
            pair_values = [(float(delta_max), float(delta_max)) for _ in pair_sources]
            sources = single_sources + pair_sources
            values = single_values + pair_values
            masks, fixed_values = _fixed_conditions(adjacency.shape[0], sources, values)
            result = simulate_released_ignition_batch(
                adjacency,
                model,
                masks,
                fixed_values,
                t_force=config.t_force,
                t_free=config.t_free,
                dt=config.dt,
                state_clip=config.state_clip,
            )
            labels = (result["valid"] & (result["final_mean"] > float(basin["basin_threshold"]))).astype(int)
            single_labels = labels[: adjacency.shape[0]]
            max_pair_labels = labels[adjacency.shape[0] :]
            if np.any(single_labels):
                continue
            successful_pairs = int(max_pair_labels.sum())
            if successful_pairs < int(config.min_successful_pairs):
                continue
            if successful_pairs >= len(max_pair_labels):
                continue
            return {
                "instance": {
                    "model_name": model_name,
                    "network_kind": network_kind,
                    "seed": int(seed),
                    "adjacency": adjacency,
                    "coupling_scale": float(scale),
                    "delta_max": float(delta_max),
                    "basin_threshold": float(basin["basin_threshold"]),
                    "low_mean": float(basin["low_mean"]),
                    "high_mean": float(basin["high_mean"]),
                    **network_meta,
                },
                "pairs": pairs,
                "single_labels": single_labels.astype(int),
                "max_pair_labels": max_pair_labels.astype(int),
            }
    return None


def _pair_response_grid(
    model: dict[str, Any],
    instance: dict[str, Any],
    pair: tuple[int, int],
    config: NetworkBasinPairIgnitionConfig,
) -> dict[str, np.ndarray]:
    amplitudes = np.linspace(0.0, float(instance["delta_max"]), int(config.delta_levels))
    left_index, right_index = np.meshgrid(np.arange(config.delta_levels), np.arange(config.delta_levels), indexing="ij")
    left, right = int(pair[0]), int(pair[1])
    sources = [(left, right)] * left_index.size
    values = [(float(amplitudes[i]), float(amplitudes[j])) for i, j in zip(left_index.ravel(), right_index.ravel(), strict=True)]
    masks, fixed_values = _fixed_conditions(np.asarray(instance["adjacency"]).shape[0], sources, values)
    result = simulate_released_ignition_batch(
        np.asarray(instance["adjacency"], dtype=float),
        model,
        masks,
        fixed_values,
        t_force=config.t_force,
        t_free=config.t_free,
        dt=config.dt,
        state_clip=config.state_clip,
    )
    labels = (result["valid"] & (result["final_mean"] > float(instance["basin_threshold"]))).astype(int)
    return {
        "amplitudes": amplitudes,
        "source_i": left_index.ravel(),
        "source_j": right_index.ravel(),
        "delta_i": amplitudes[left_index.ravel()],
        "delta_j": amplitudes[right_index.ravel()],
        "labels": labels,
        "final_mean": np.asarray(result["final_mean"], dtype=float),
    }


def _peid_from_grid_labels(source_i: np.ndarray, source_j: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    left_ei = _mutual_information(source_i, labels)
    right_ei = _mutual_information(source_j, labels)
    joint_ei = _mutual_information(np.column_stack([source_i, source_j]), labels)
    synergy = float(joint_ei - left_ei - right_ei)
    conditional_mi = _conditional_mi(source_i, source_j, labels)
    return {
        "left_ei": float(left_ei),
        "right_ei": float(right_ei),
        "single_ei_sum": float(left_ei + right_ei),
        "joint_ei": float(joint_ei),
        "synergy": synergy,
        "conditional_mi": float(conditional_mi),
        "synergy_ratio": float(synergy / joint_ei) if abs(joint_ei) > 1e-12 else 0.0,
    }


def _safe_spearman(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    frame = pd.DataFrame({"left": np.asarray(left, dtype=float), "right": np.asarray(right, dtype=float)}).dropna()
    if len(frame) < 2 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return float("nan")
    return float(frame["left"].corr(frame["right"], method="spearman"))


def _cache_paths(config: NetworkBasinPairIgnitionConfig) -> dict[str, Path]:
    return {
        "summary_json": config.output_dir / "summary.json",
        "pairs_jsonl": config.output_dir / "pair_scores.jsonl",
        "arrays_npz": config.output_dir / "representative_arrays.npz",
        "manifest_json": config.output_dir / "manifest.json",
    }


def _jsonable_config(config: NetworkBasinPairIgnitionConfig) -> dict[str, Any]:
    values = asdict(config)
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
    return values


def _read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line.strip()])


def _matrix_with_diagonal(frame: pd.DataFrame, value: str, source_count: int, diagonal: np.ndarray | None = None) -> np.ndarray:
    matrix = _matrix_from_pairs(frame, value, source_count)
    if diagonal is not None:
        np.fill_diagonal(matrix, np.asarray(diagonal, dtype=float))
    return matrix


def _group_prefix(model_name: str, network_kind: str) -> str:
    return f"{model_name}_{network_kind}".lower().replace("|", "_")


def _top_k_recall(frame: pd.DataFrame) -> float:
    labels = frame["max_pair_basin_label"].to_numpy(dtype=bool)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-frame["synergy"].to_numpy(dtype=float), kind="mergesort")
    return float(np.mean(labels[order[:positives]]))


def _top_k_recall_by_score(frame: pd.DataFrame, score_column: str) -> float:
    labels = frame["max_pair_basin_label"].to_numpy(dtype=bool)
    positives = int(labels.sum())
    if positives == 0 or score_column not in frame:
        return float("nan")
    scores = frame[score_column].to_numpy(dtype=float)
    finite = np.isfinite(scores)
    if finite.sum() < positives:
        return float("nan")
    candidate = frame.loc[finite].copy()
    labels = candidate["max_pair_basin_label"].to_numpy(dtype=bool)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-candidate[score_column].to_numpy(dtype=float), kind="mergesort")
    return float(np.mean(labels[order[:positives]]))


def _initial_state_proxy_cache_paths(config: InitialStateSynProxyConfig) -> dict[str, Path]:
    return {
        "summary_json": config.output_dir / "summary.json",
        "pairs_jsonl": config.output_dir / "pair_scores.jsonl",
        "arrays_npz": config.output_dir / "representative_arrays.npz",
        "manifest_json": config.output_dir / "manifest.json",
    }


def _tm_initial_state_cache_paths(config: TransportMapInitialStateSynConfig) -> dict[str, Path]:
    return {
        "summary_json": config.output_dir / "summary.json",
        "pairs_jsonl": config.output_dir / "pair_scores.jsonl",
        "arrays_npz": config.output_dir / "representative_arrays.npz",
        "manifest_json": config.output_dir / "manifest.json",
    }


def _equal_frequency_bins(values: np.ndarray, *, bin_count: int) -> np.ndarray | None:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if values.size < int(bin_count) or not np.isfinite(values).all():
        return None
    if np.nanmax(values) - np.nanmin(values) <= 1e-12:
        return None
    order = np.argsort(values, kind="mergesort")
    labels = np.empty(values.size, dtype=int)
    labels[order] = np.minimum((np.arange(values.size) * int(bin_count)) // values.size, int(bin_count) - 1)
    if np.unique(labels).size < 2:
        return None
    return labels


def _binary_median_labels(values: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        return None
    threshold = float(np.median(values))
    labels = (values > threshold).astype(int)
    if np.unique(labels).size < 2:
        return None
    return labels


def _load_initial_state_proxy_cache(paths: dict[str, Path]) -> dict[str, Any]:
    summary = json.loads(paths["summary_json"].read_text())
    pair_rows = _read_jsonl(paths["pairs_jsonl"])
    arrays = dict(np.load(paths["arrays_npz"], allow_pickle=False))
    manifest = json.loads(paths["manifest_json"].read_text())
    return {
        "summary": summary,
        "pair_rows": pair_rows,
        "representative_arrays": arrays,
        "manifest": manifest,
        "cache_paths": paths,
    }


def _load_tm_initial_state_cache(paths: dict[str, Path]) -> dict[str, Any]:
    summary = json.loads(paths["summary_json"].read_text())
    pair_rows = _read_jsonl(paths["pairs_jsonl"])
    arrays = dict(np.load(paths["arrays_npz"], allow_pickle=False))
    manifest = json.loads(paths["manifest_json"].read_text())
    return {
        "summary": summary,
        "pair_rows": pair_rows,
        "representative_arrays": arrays,
        "manifest": manifest,
        "cache_paths": paths,
    }


def _source_array_key(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"


def _stratified_pairs_from_matrix(success_rate_matrix: np.ndarray, *, pair_count: int) -> list[tuple[int, int]]:
    matrix = np.asarray(success_rate_matrix, dtype=float)
    node_count = int(matrix.shape[0])
    rows = [
        (left, right, float(matrix[left, right]))
        for left in range(node_count)
        for right in range(left + 1, node_count)
        if np.isfinite(matrix[left, right])
    ]
    if not rows:
        return []
    rows.sort(key=lambda item: (item[2], item[0], item[1]))
    if int(pair_count) >= len(rows):
        return [(left, right) for left, right, _ in rows]
    indices = np.linspace(0, len(rows) - 1, int(pair_count), dtype=int)
    selected = [rows[int(index)] for index in indices]
    return [(left, right) for left, right, _ in selected]


def _transport_map_mi(source: np.ndarray, target: np.ndarray, *, degree: int, clip_negative: bool) -> tuple[float, str]:
    summary = estimate_mutual_information_transport_map(source, target, degree=int(degree))
    value = float(summary["mi_hat"])
    if bool(clip_negative):
        value = max(0.0, value)
    return value, str(summary.get("backend", "transport_map"))


def _compute_tm_initial_state_group(
    model_name: str,
    network_kind: str,
    source_arrays: dict[str, np.ndarray],
    config: TransportMapInitialStateSynConfig,
    *,
    group_index: int,
) -> dict[str, Any]:
    group = f"{model_name}|{network_kind}"
    prefix = _group_prefix(model_name, network_kind)
    adjacency_key = _source_array_key(prefix, "adjacency")
    success_key = _source_array_key(prefix, "success_rate_matrix")
    if adjacency_key not in source_arrays or success_key not in source_arrays:
        return {"group": group, "valid": False, "reason": "missing representative adjacency or success matrix"}

    adjacency = np.asarray(source_arrays[adjacency_key], dtype=float)
    success_rate_matrix = np.asarray(source_arrays[success_key], dtype=float)
    node_count = int(adjacency.shape[0])
    pairs = _stratified_pairs_from_matrix(success_rate_matrix, pair_count=int(config.pair_count_per_group))
    if not pairs:
        return {"group": group, "valid": False, "reason": "no finite candidate pairs"}

    rng = np.random.default_rng(int(config.seed + 1009 * group_index))
    state_high = float(config.initial_state_high_by_model.get(model_name, 3.0))
    initial_states = rng.uniform(0.0, state_high, size=(int(config.sample_count), node_count))
    masks = np.zeros_like(initial_states, dtype=bool)
    fixed_values = np.zeros_like(initial_states, dtype=float)
    result = simulate_released_ignition_batch(
        adjacency,
        _tm_model_for_name(model_name, config),
        masks,
        fixed_values,
        t_force=0.0,
        t_free=config.t_short,
        dt=config.dt,
        state_clip=config.state_clip,
        initial_states=initial_states,
    )
    valid = np.asarray(result["valid"], dtype=bool)
    valid_fraction = float(valid.mean()) if valid.size else 0.0
    if valid_fraction < float(config.min_valid_fraction):
        return {"group": group, "valid": False, "reason": "too many invalid trajectories", "valid_fraction": valid_fraction}

    initial_valid = initial_states[valid]
    delta_mean = np.asarray(result["final_mean"], dtype=float)[valid] - initial_valid.mean(axis=1)
    if np.nanmax(delta_mean) - np.nanmin(delta_mean) <= 1e-12:
        return {"group": group, "valid": False, "reason": "degenerate continuous target", "valid_fraction": valid_fraction}
    target = delta_mean[:, None]

    singleton_cache: dict[int, tuple[float, str]] = {}

    def _single(node: int) -> tuple[float, str]:
        if int(node) not in singleton_cache:
            singleton_cache[int(node)] = _transport_map_mi(
                initial_valid[:, [int(node)]],
                target,
                degree=int(config.transport_degree),
                clip_negative=bool(config.clip_negative_ei),
            )
        return singleton_cache[int(node)]

    rows: list[dict[str, Any]] = []
    backend = "transport_map"
    max_basin_matrix = np.asarray(source_arrays.get(_source_array_key(prefix, "max_basin_matrix"), np.full((node_count, node_count), np.nan)), dtype=float)
    full_syn_matrix = np.asarray(source_arrays.get(_source_array_key(prefix, "synergy_matrix"), np.full((node_count, node_count), np.nan)), dtype=float)
    for left, right in pairs:
        left_ei, left_backend = _single(left)
        right_ei, right_backend = _single(right)
        joint_ei, joint_backend = _transport_map_mi(
            initial_valid[:, [left, right]],
            target,
            degree=int(config.transport_degree),
            clip_negative=bool(config.clip_negative_ei),
        )
        backend = joint_backend or left_backend or right_backend
        synergy = float(joint_ei - left_ei - right_ei)
        rows.append(
            {
                "model_name": model_name,
                "network_kind": network_kind,
                "group": group,
                "pair_i": int(left),
                "pair_j": int(right),
                "tm_initial_syn": synergy,
                "tm_joint_ei": float(joint_ei),
                "tm_unique_i": float(left_ei),
                "tm_unique_j": float(right_ei),
                "tm_single_ei_sum": float(left_ei + right_ei),
                "success_rate": float(success_rate_matrix[left, right]),
                "max_pair_basin_label": int(max_basin_matrix[left, right]) if np.isfinite(max_basin_matrix[left, right]) else 0,
                "full_basin_syn": float(full_syn_matrix[left, right]),
                "valid_fraction": valid_fraction,
                "target_std": float(np.std(delta_mean, ddof=1)),
                "backend": backend,
            }
        )

    frame = pd.DataFrame(rows)
    summary = {
        "model_name": model_name,
        "network_kind": network_kind,
        "valid": True,
        "censored": False,
        "node_count": node_count,
        "pair_count": int(len(frame)),
        "valid_fraction": valid_fraction,
        "target_std": float(np.std(delta_mean, ddof=1)),
        "spearman_tm_syn_success": _safe_spearman(frame["tm_initial_syn"], frame["success_rate"]),
        "spearman_tm_syn_full_basin_syn": _safe_spearman(frame["tm_initial_syn"], frame["full_basin_syn"]),
        "top_k_recall": _top_k_recall_by_score(frame, "tm_initial_syn"),
        "backend": backend,
    }
    matrix_frame = frame.rename(columns={"tm_initial_syn": "value"})
    arrays = {
        f"{prefix}_adjacency": adjacency,
        f"{prefix}_tm_initial_syn_matrix": _matrix_from_pairs(matrix_frame, "value", node_count),
        f"{prefix}_source_success_rate_matrix": success_rate_matrix,
        f"{prefix}_source_basin_syn_matrix": full_syn_matrix,
        f"{prefix}_sampled_pairs": np.asarray(pairs, dtype=int),
        f"{prefix}_delta_mean": delta_mean.astype(float),
    }
    return {"group": group, "valid": True, "summary": summary, "pair_rows": frame, "arrays": arrays}


def run_transport_map_initial_state_syn_experiment(
    config: TransportMapInitialStateSynConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    paths = _tm_initial_state_cache_paths(config)
    if not force and all(path.exists() for path in paths.values()):
        return _load_tm_initial_state_cache(paths)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if not config.source_arrays_npz.exists():
        raise FileNotFoundError(f"Missing source representative arrays: {config.source_arrays_npz}")

    source_arrays = dict(np.load(config.source_arrays_npz, allow_pickle=False))
    all_rows: list[pd.DataFrame] = []
    groups: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    group_index = 0
    for model_name in config.model_names:
        for network_kind in config.network_kinds:
            computed = _compute_tm_initial_state_group(
                model_name,
                network_kind,
                source_arrays,
                config,
                group_index=group_index,
            )
            group_index += 1
            group = str(computed["group"])
            if not bool(computed.get("valid", False)):
                groups[group] = {
                    "model_name": model_name,
                    "network_kind": network_kind,
                    "valid": False,
                    "censored": True,
                    "reason": str(computed.get("reason", "invalid")),
                    "pair_count": 0,
                    "spearman_tm_syn_success": float("nan"),
                    "spearman_tm_syn_full_basin_syn": float("nan"),
                    "top_k_recall": float("nan"),
                }
                continue
            groups[group] = dict(computed["summary"])
            all_rows.append(computed["pair_rows"])
            arrays.update(computed["arrays"])

    pair_frame = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    pooled = {
        "spearman_tm_syn_success": _safe_spearman(pair_frame["tm_initial_syn"], pair_frame["success_rate"]) if not pair_frame.empty else float("nan"),
        "spearman_tm_syn_full_basin_syn": _safe_spearman(pair_frame["tm_initial_syn"], pair_frame["full_basin_syn"]) if not pair_frame.empty else float("nan"),
        "top_k_recall": _top_k_recall_by_score(pair_frame, "tm_initial_syn") if not pair_frame.empty else float("nan"),
    }
    summary = {
        "experiment": "transport_map_initial_state_syn",
        "source_variable": "continuous initial node states",
        "target_variable": "continuous short-time whole-network mean growth",
        "source_basin_experiment": str(config.source_arrays_npz),
        "groups": groups,
        "pooled": pooled,
    }
    manifest = {
        "experiment": "transport_map_initial_state_syn",
        "config": _jsonable_config(config),
        "peid_definition": "I(x_i(0),x_j(0); delta_mean)-I(x_i(0); delta_mean)-I(x_j(0); delta_mean)",
        "target_definition": "delta_mean=mean(x(t_short))-mean(x(0))",
        "pair_sampling": "success-rate-stratified representative pairs per model-network group",
        "cache_paths": {key: str(path) for key, path in paths.items()},
    }

    paths["summary_json"].write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n")
    with paths["pairs_jsonl"].open("w") as handle:
        for row in pair_frame.to_dict("records"):
            handle.write(json.dumps(row, allow_nan=True) + "\n")
    np.savez_compressed(paths["arrays_npz"], **arrays)

    return {
        "summary": summary,
        "pair_rows": pair_frame,
        "representative_arrays": arrays,
        "manifest": manifest,
        "cache_paths": paths,
    }


def _compute_initial_state_proxy_group(
    model_name: str,
    network_kind: str,
    source_arrays: dict[str, np.ndarray],
    config: InitialStateSynProxyConfig,
    *,
    group_index: int,
) -> dict[str, Any]:
    group = f"{model_name}|{network_kind}"
    prefix = _group_prefix(model_name, network_kind)
    adjacency_key = _source_array_key(prefix, "adjacency")
    if adjacency_key not in source_arrays:
        return {"group": group, "valid": False, "reason": "missing representative adjacency"}

    adjacency = np.asarray(source_arrays[adjacency_key], dtype=float)
    node_count = int(adjacency.shape[0])
    rng = np.random.default_rng(int(config.seed + 1009 * group_index))
    state_high = float(config.initial_state_high_by_model.get(model_name, 3.0))
    initial_states = rng.uniform(0.0, state_high, size=(int(config.sample_count), node_count))
    masks = np.zeros_like(initial_states, dtype=bool)
    fixed_values = np.zeros_like(initial_states, dtype=float)
    result = simulate_released_ignition_batch(
        adjacency,
        _proxy_model_for_name(model_name, config),
        masks,
        fixed_values,
        t_force=0.0,
        t_free=config.t_short,
        dt=config.dt,
        state_clip=config.state_clip,
        initial_states=initial_states,
    )
    valid = np.asarray(result["valid"], dtype=bool)
    valid_fraction = float(valid.mean()) if valid.size else 0.0
    if valid_fraction < float(config.min_valid_fraction):
        return {"group": group, "valid": False, "reason": "too many invalid trajectories", "valid_fraction": valid_fraction}

    initial_valid = initial_states[valid]
    final_mean = np.asarray(result["final_mean"], dtype=float)[valid]
    delta_mean = final_mean - initial_valid.mean(axis=1)
    target_labels = _binary_median_labels(delta_mean)
    if target_labels is None:
        return {"group": group, "valid": False, "reason": "degenerate median-binarized target", "valid_fraction": valid_fraction}

    node_bins: list[np.ndarray] = []
    for node in range(node_count):
        labels = _equal_frequency_bins(initial_valid[:, node], bin_count=int(config.source_bins))
        if labels is None:
            return {
                "group": group,
                "valid": False,
                "reason": f"degenerate source bins for node {node}",
                "valid_fraction": valid_fraction,
            }
        node_bins.append(labels)

    rows: list[dict[str, Any]] = []
    for left, right in combinations(range(node_count), 2):
        peid = _peid_from_grid_labels(node_bins[left], node_bins[right], target_labels)
        success_rate_matrix = np.asarray(source_arrays.get(_source_array_key(prefix, "success_rate_matrix"), np.full((node_count, node_count), np.nan)), dtype=float)
        max_basin_matrix = np.asarray(source_arrays.get(_source_array_key(prefix, "max_basin_matrix"), np.full((node_count, node_count), np.nan)), dtype=float)
        full_syn_matrix = np.asarray(source_arrays.get(_source_array_key(prefix, "synergy_matrix"), np.full((node_count, node_count), np.nan)), dtype=float)
        rows.append(
            {
                "model_name": model_name,
                "network_kind": network_kind,
                "group": group,
                "pair_i": int(left),
                "pair_j": int(right),
                "cheap_initial_syn": float(peid["synergy"]),
                "cheap_joint_ei": float(peid["joint_ei"]),
                "cheap_unique_i": float(peid["left_ei"]),
                "cheap_unique_j": float(peid["right_ei"]),
                "cheap_single_ei_sum": float(peid["single_ei_sum"]),
                "cheap_conditional_mi": float(peid["conditional_mi"]),
                "cheap_synergy_ratio": float(peid["synergy_ratio"]),
                "success_rate": float(success_rate_matrix[left, right]),
                "max_pair_basin_label": int(max_basin_matrix[left, right]) if np.isfinite(max_basin_matrix[left, right]) else 0,
                "full_basin_syn": float(full_syn_matrix[left, right]),
                "valid_fraction": valid_fraction,
                "target_positive_rate": float(target_labels.mean()),
            }
        )

    frame = pd.DataFrame(rows)
    summary = {
        "model_name": model_name,
        "network_kind": network_kind,
        "valid": True,
        "censored": False,
        "node_count": node_count,
        "pair_count": int(len(frame)),
        "valid_fraction": valid_fraction,
        "target_positive_rate": float(target_labels.mean()),
        "spearman_proxy_success": _safe_spearman(frame["cheap_initial_syn"], frame["success_rate"]),
        "spearman_proxy_full_basin_syn": _safe_spearman(frame["cheap_initial_syn"], frame["full_basin_syn"]),
        "top_k_recall": _top_k_recall_by_score(frame, "cheap_initial_syn"),
    }
    arrays = {
        f"{prefix}_adjacency": adjacency,
        f"{prefix}_cheap_initial_syn_matrix": _matrix_from_pairs(frame.rename(columns={"cheap_initial_syn": "value"}), "value", node_count),
        f"{prefix}_cheap_joint_ei_matrix": _matrix_from_pairs(frame.rename(columns={"cheap_joint_ei": "value"}), "value", node_count),
        f"{prefix}_cheap_conditional_mi_matrix": _matrix_from_pairs(frame.rename(columns={"cheap_conditional_mi": "value"}), "value", node_count),
        f"{prefix}_source_success_rate_matrix": np.asarray(source_arrays.get(_source_array_key(prefix, "success_rate_matrix"), np.full((node_count, node_count), np.nan)), dtype=float),
        f"{prefix}_source_max_basin_matrix": np.asarray(source_arrays.get(_source_array_key(prefix, "max_basin_matrix"), np.full((node_count, node_count), np.nan)), dtype=float),
        f"{prefix}_source_basin_syn_matrix": np.asarray(source_arrays.get(_source_array_key(prefix, "synergy_matrix"), np.full((node_count, node_count), np.nan)), dtype=float),
        f"{prefix}_target_labels": target_labels.astype(int),
        f"{prefix}_delta_mean": delta_mean.astype(float),
    }
    return {"group": group, "valid": True, "summary": summary, "pair_rows": frame, "arrays": arrays}


def run_initial_state_syn_proxy_experiment(
    config: InitialStateSynProxyConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    paths = _initial_state_proxy_cache_paths(config)
    if not force and all(path.exists() for path in paths.values()):
        return _load_initial_state_proxy_cache(paths)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if not config.source_arrays_npz.exists():
        raise FileNotFoundError(f"Missing source representative arrays: {config.source_arrays_npz}")

    source_arrays = dict(np.load(config.source_arrays_npz, allow_pickle=False))
    all_rows: list[pd.DataFrame] = []
    groups: dict[str, dict[str, Any]] = {}
    manifest_groups: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    group_index = 0
    for model_name in config.model_names:
        for network_kind in config.network_kinds:
            computed = _compute_initial_state_proxy_group(
                model_name,
                network_kind,
                source_arrays,
                config,
                group_index=group_index,
            )
            group_index += 1
            group = str(computed["group"])
            if not bool(computed.get("valid", False)):
                groups[group] = {
                    "model_name": model_name,
                    "network_kind": network_kind,
                    "valid": False,
                    "censored": True,
                    "reason": str(computed.get("reason", "invalid")),
                    "pair_count": 0,
                    "spearman_proxy_success": float("nan"),
                    "spearman_proxy_full_basin_syn": float("nan"),
                    "top_k_recall": float("nan"),
                }
                manifest_groups[group] = dict(groups[group])
                continue
            groups[group] = dict(computed["summary"])
            manifest_groups[group] = dict(computed["summary"])
            all_rows.append(computed["pair_rows"])
            arrays.update(computed["arrays"])

    pair_frame = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    pooled = {
        "spearman_proxy_success": _safe_spearman(pair_frame["cheap_initial_syn"], pair_frame["success_rate"]) if not pair_frame.empty else float("nan"),
        "spearman_proxy_full_basin_syn": _safe_spearman(pair_frame["cheap_initial_syn"], pair_frame["full_basin_syn"]) if not pair_frame.empty else float("nan"),
        "top_k_recall": _top_k_recall_by_score(pair_frame, "cheap_initial_syn") if not pair_frame.empty else float("nan"),
    }
    summary = {
        "experiment": "initial_state_syn_proxy",
        "source_variable": "equal-frequency bins of initial node states",
        "target_variable": "median-binarized short-time whole-network mean growth",
        "source_basin_experiment": str(config.source_arrays_npz),
        "groups": groups,
        "pooled": pooled,
    }
    manifest = {
        "experiment": "initial_state_syn_proxy",
        "config": _jsonable_config(config),
        "peid_definition": "I(bin(x_i(0)),bin(x_j(0)); Y)-I(bin(x_i(0));Y)-I(bin(x_j(0));Y)",
        "target_definition": "Y=1[mean(x(t_short))-mean(x(0)) is above its sample median]",
        "groups": manifest_groups,
        "cache_paths": {key: str(path) for key, path in paths.items()},
    }

    paths["summary_json"].write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n")
    with paths["pairs_jsonl"].open("w") as handle:
        for row in pair_frame.to_dict("records"):
            handle.write(json.dumps(row, allow_nan=True) + "\n")
    np.savez_compressed(paths["arrays_npz"], **arrays)

    return {
        "summary": summary,
        "pair_rows": pair_frame,
        "representative_arrays": arrays,
        "manifest": manifest,
        "cache_paths": paths,
    }


def run_network_basin_pair_ensemble(
    config: NetworkBasinPairIgnitionConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    paths = _cache_paths(config)
    if not force and all(path.exists() for path in paths.values()):
        summary = json.loads(paths["summary_json"].read_text())
        pair_rows = _read_jsonl(paths["pairs_jsonl"])
        arrays = dict(np.load(paths["arrays_npz"], allow_pickle=False))
        return {
            "summary": summary,
            "pair_rows": pair_rows,
            "representative_arrays": arrays,
            "cache_paths": paths,
        }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    representatives: dict[str, dict[str, Any]] = {}
    candidate_counts: dict[str, int] = {}

    for model_name in config.model_names:
        model = _model_for_name(model_name, config)
        for network_kind in config.network_kinds:
            group = f"{model_name}|{network_kind}"
            accepted = 0
            candidate_counts[group] = 0
            for offset in range(config.candidate_seed_count):
                if accepted >= config.accepted_instances_per_group:
                    break
                seed = int(config.seed + 10000 * list(config.model_names).index(model_name) + 1000 * list(config.network_kinds).index(network_kind) + offset)
                candidate_counts[group] += 1
                screened = _screen_candidate_instance(model_name, network_kind, seed, config)
                if screened is None:
                    continue
                instance = dict(screened["instance"])
                pairs = [tuple(pair) for pair in screened["pairs"]]
                single_labels = np.asarray(screened["single_labels"], dtype=int)
                instance_id = accepted
                pair_rows: list[dict[str, Any]] = []
                for pair in pairs:
                    response = _pair_response_grid(model, instance, pair, config)
                    labels = np.asarray(response["labels"], dtype=int)
                    peid = _peid_from_grid_labels(response["source_i"], response["source_j"], labels)
                    row = {
                        "model_name": model_name,
                        "network_kind": network_kind,
                        "group": group,
                        "instance": int(instance_id),
                        "instance_seed": int(instance["seed"]),
                        "pair_i": int(pair[0]),
                        "pair_j": int(pair[1]),
                        "max_pair_basin_label": int(labels[-1]),
                        "success_rate": float(labels.mean()),
                        "mean_final_mean": float(np.mean(response["final_mean"])),
                        "coupling_scale": float(instance["coupling_scale"]),
                        "delta_max": float(instance["delta_max"]),
                        "basin_threshold": float(instance["basin_threshold"]),
                        "single_failure_rate": float(1.0 - single_labels.mean()),
                        **peid,
                    }
                    pair_rows.append(row)
                frame = pd.DataFrame(pair_rows)
                if frame["success_rate"].std(ddof=0) <= float(config.min_success_rate_std):
                    continue
                frame["rank_synergy"] = frame["synergy"].rank(method="first", ascending=False).astype(int)
                frame["instance_spearman_synergy_success"] = _safe_spearman(frame["synergy"], frame["success_rate"])
                all_rows.extend(frame.to_dict("records"))
                if group not in representatives:
                    representatives[group] = {
                        "instance": instance,
                        "pair_rows": frame.copy(),
                        "single_labels": single_labels,
                    }
                accepted += 1

    pair_frame = pd.DataFrame(all_rows)
    groups: dict[str, dict[str, Any]] = {}
    for model_name in config.model_names:
        for network_kind in config.network_kinds:
            group = f"{model_name}|{network_kind}"
            subset = pair_frame.loc[pair_frame["group"].eq(group)].copy() if not pair_frame.empty else pd.DataFrame()
            accepted_instances = int(subset["instance"].nunique()) if not subset.empty else 0
            groups[group] = {
                "model_name": model_name,
                "network_kind": network_kind,
                "accepted_instances": accepted_instances,
                "candidate_count": int(candidate_counts.get(group, 0)),
                "censored": bool(accepted_instances < config.accepted_instances_per_group),
                "single_failure_rate": float(subset["single_failure_rate"].mean()) if not subset.empty else float("nan"),
                "successful_pair_count": int(subset["max_pair_basin_label"].sum()) if not subset.empty else 0,
                "pooled_spearman_synergy_success": _safe_spearman(subset["synergy"], subset["success_rate"]) if not subset.empty else float("nan"),
                "mean_instance_spearman_synergy_success": float(subset.groupby("instance")["instance_spearman_synergy_success"].first().mean()) if not subset.empty else float("nan"),
                "top_k_recall": _top_k_recall(subset) if not subset.empty else float("nan"),
            }

    summary = {
        "experiment": "network_basin_pair_ignition",
        "target": "released whole-network final mean basin label",
        "models": list(config.model_names),
        "network_kinds": list(config.network_kinds),
        "groups": groups,
    }
    manifest = {
        "experiment": "network_basin_pair_ignition",
        "config": _jsonable_config(config),
        "peid_definition": "I(delta_i,delta_j; released whole-network basin)-I(delta_i; basin)-I(delta_j; basin)",
        "screening_rule": "all singleton 2*delta_max releases fail and at least three max-strength pairs release to the high basin",
        "zotero_context": "Local Zotero contains PEID preprint item MYATYWAJ; implementation uses repository MI helpers.",
        "cache_paths": {key: str(path) for key, path in paths.items()},
    }

    paths["summary_json"].write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n")
    with paths["pairs_jsonl"].open("w") as handle:
        for row in pair_frame.to_dict("records"):
            handle.write(json.dumps(row, allow_nan=True) + "\n")

    arrays: dict[str, np.ndarray] = {}
    for group, item in representatives.items():
        model_name, network_kind = group.split("|")
        prefix = _group_prefix(model_name, network_kind)
        frame = item["pair_rows"]
        single_labels = np.asarray(item["single_labels"], dtype=int)
        arrays[f"{prefix}_adjacency"] = np.asarray(item["instance"]["adjacency"], dtype=float)
        arrays[f"{prefix}_max_basin_matrix"] = _matrix_with_diagonal(frame, "max_pair_basin_label", config.node_count, single_labels)
        arrays[f"{prefix}_synergy_matrix"] = _matrix_with_diagonal(frame, "synergy", config.node_count)
        arrays[f"{prefix}_success_rate_matrix"] = _matrix_with_diagonal(frame, "success_rate", config.node_count)
        arrays[f"{prefix}_single_labels"] = single_labels
    np.savez_compressed(paths["arrays_npz"], **arrays)

    return {
        "summary": summary,
        "pair_rows": pair_frame,
        "representative_arrays": arrays,
        "cache_paths": paths,
    }


def plot_network_basin_results(
    results: dict[str, Any],
    config: NetworkBasinPairIgnitionConfig,
    *,
    figure_dir: Path | None = None,
    report_asset_dir: Path | None = None,
) -> dict[str, dict[str, Path]]:
    figure_dir = config.output_dir / "figures" if figure_dir is None else Path(figure_dir)
    arrays = results.get("representative_arrays", {})
    pair_rows = results["pair_rows"].copy()
    summary = results["summary"]
    paths: dict[str, dict[str, Path]] = {}

    fig, axes = plt.subplots(1, len(config.network_kinds), figsize=(4.2 * len(config.network_kinds), 3.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, network_kind in zip(axes, config.network_kinds, strict=False):
        key = None
        for model_name in config.model_names:
            candidate = f"{_group_prefix(model_name, network_kind)}_adjacency"
            if candidate in arrays:
                key = candidate
                break
        axis.set_title(str(network_kind), fontsize=9)
        axis.axis("off")
        if key is None:
            axis.text(0.5, 0.5, "no accepted instance", ha="center", va="center")
            continue
        adjacency = np.asarray(arrays[key], dtype=float)
        graph = nx.from_numpy_array((adjacency > 0.0).astype(float))
        pos = nx.spring_layout(graph, seed=42)
        degrees = adjacency.sum(axis=1)
        nx.draw_networkx_edges(graph, pos, ax=axis, edge_color="0.75", width=1.0)
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=axis,
            node_color=degrees,
            cmap="Blues",
            node_size=260,
            edgecolors="black",
            linewidths=0.6,
        )
        nx.draw_networkx_labels(graph, pos, ax=axis, font_size=7)
    paths["network_structure"] = _save_figure(fig, figure_dir, "network_basin_network_structure")

    groups = [group for group in summary["groups"] if f"{_group_prefix(*group.split('|'))}_max_basin_matrix" in arrays]
    groups = groups[: max(1, min(4, len(groups)))]
    fig, axes = plt.subplots(len(groups), 3, figsize=(9.6, 3.0 * max(1, len(groups))), constrained_layout=True)
    axes = np.asarray(axes).reshape(len(groups), 3)
    for row, group in enumerate(groups):
        model_name, network_kind = group.split("|")
        prefix = _group_prefix(model_name, network_kind)
        matrices = [
            (np.asarray(arrays[f"{prefix}_max_basin_matrix"], dtype=float), "Max basin label", "Greys"),
            (np.asarray(arrays[f"{prefix}_synergy_matrix"], dtype=float), "PEID Syn", "YlOrBr"),
            (np.asarray(arrays[f"{prefix}_success_rate_matrix"], dtype=float), "Success rate", "viridis"),
        ]
        for col, (matrix, label, cmap) in enumerate(matrices):
            image = axes[row, col].imshow(matrix, cmap=cmap, vmin=0.0)
            axes[row, col].set_title(f"{model_name} {network_kind}: {label}", fontsize=8)
            axes[row, col].set_xlabel("Source node")
            axes[row, col].set_ylabel("Source node")
            axes[row, col].set_xticks(range(config.node_count))
            axes[row, col].set_yticks(range(config.node_count))
            fig.colorbar(image, ax=axes[row, col], shrink=0.74)
    paths["representative_heatmaps"] = _save_figure(fig, figure_dir, "network_basin_representative_heatmaps")

    fig, axis = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    palette = {
        "Neural|ER": "#4C78A8",
        "Neural|WS": "#72B7B2",
        "Eco|ER": "#59A14F",
        "Eco|WS": "#F28E2B",
    }
    if not pair_rows.empty:
        for group, subset in pair_rows.groupby("group"):
            axis.scatter(
                subset["synergy"],
                subset["success_rate"],
                s=26,
                alpha=0.72,
                edgecolor="white",
                linewidth=0.3,
                color=palette.get(group, "0.4"),
                label=group,
            )
    axis.set_xlabel("PEID Syn (bits)")
    axis.set_ylabel("Grid ignition success rate")
    axis.set_ylim(-0.04, 1.04)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    paths["success_scatter"] = _save_figure(fig, figure_dir, "network_basin_success_scatter")

    metric_rows = []
    for group, values in summary["groups"].items():
        metric_rows.append(
            {
                "group": group,
                "spearman": float(values["pooled_spearman_synergy_success"]),
                "top_k_recall": float(values["top_k_recall"]),
            }
        )
    metric_frame = pd.DataFrame(metric_rows)
    fig, axis = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    x = np.arange(len(metric_frame))
    width = 0.36
    axis.bar(x - width / 2, metric_frame["spearman"], width=width, color="#4C78A8", label="Spearman")
    axis.bar(x + width / 2, metric_frame["top_k_recall"], width=width, color="#F28E2B", label="Top-k recall")
    axis.set_xticks(x, labels=metric_frame["group"], rotation=25, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Metric")
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    paths["summary_metrics"] = _save_figure(fig, figure_dir, "network_basin_summary_metrics")

    if report_asset_dir is not None:
        report_asset_dir = Path(report_asset_dir)
        report_asset_dir.mkdir(parents=True, exist_ok=True)
        for item in paths.values():
            target = report_asset_dir / f"part3_{item['png'].name}"
            shutil.copyfile(item["png"], target)
            item["report_png"] = target
    return paths


def plot_initial_state_syn_proxy_results(
    results: dict[str, Any],
    config: InitialStateSynProxyConfig,
    *,
    figure_dir: Path | None = None,
    report_asset_dir: Path | None = None,
) -> dict[str, dict[str, Path]]:
    figure_dir = config.output_dir / "figures" if figure_dir is None else Path(figure_dir)
    arrays = results.get("representative_arrays", {})
    pair_rows = results["pair_rows"].copy()
    summary = results["summary"]
    paths: dict[str, dict[str, Path]] = {}

    valid_groups = [
        group
        for group, values in summary["groups"].items()
        if bool(values.get("valid", False)) and f"{_group_prefix(*group.split('|'))}_cheap_initial_syn_matrix" in arrays
    ]
    valid_groups = valid_groups[: max(1, min(4, len(valid_groups)))]
    if valid_groups:
        fig, axes = plt.subplots(len(valid_groups), 3, figsize=(9.8, 3.0 * len(valid_groups)), constrained_layout=True)
        axes = np.asarray(axes).reshape(len(valid_groups), 3)
        for row, group in enumerate(valid_groups):
            model_name, network_kind = group.split("|")
            prefix = _group_prefix(model_name, network_kind)
            matrices = [
                (np.asarray(arrays[f"{prefix}_cheap_initial_syn_matrix"], dtype=float), "Initial-state Syn", "YlOrBr"),
                (np.asarray(arrays[f"{prefix}_source_success_rate_matrix"], dtype=float), "Ignition success rate", "viridis"),
                (np.asarray(arrays[f"{prefix}_source_basin_syn_matrix"], dtype=float), "Basin-label Syn", "YlGnBu"),
            ]
            node_count = int(matrices[0][0].shape[0])
            for col, (matrix, label, cmap) in enumerate(matrices):
                image = axes[row, col].imshow(matrix, cmap=cmap, vmin=0.0)
                axes[row, col].set_title(f"{model_name} {network_kind}: {label}", fontsize=8)
                axes[row, col].set_xlabel("Source node")
                axes[row, col].set_ylabel("Source node")
                axes[row, col].set_xticks(range(node_count))
                axes[row, col].set_yticks(range(node_count))
                fig.colorbar(image, ax=axes[row, col], shrink=0.74)
    else:
        fig, axis = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)
        axis.axis("off")
        axis.text(0.5, 0.5, "no valid proxy group", ha="center", va="center")
    paths["heatmaps"] = _save_figure(fig, figure_dir, "initial_state_syn_heatmaps")

    fig, axis = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    palette = {
        "Neural|ER": "#4C78A8",
        "Neural|WS": "#72B7B2",
        "Eco|ER": "#59A14F",
        "Eco|WS": "#F28E2B",
    }
    if not pair_rows.empty:
        for group, subset in pair_rows.groupby("group"):
            axis.scatter(
                subset["cheap_initial_syn"],
                subset["success_rate"],
                s=28,
                alpha=0.74,
                edgecolor="white",
                linewidth=0.3,
                color=palette.get(group, "0.4"),
                label=group,
            )
    axis.set_xlabel("Initial-state proxy Syn (bits)")
    axis.set_ylabel("Grid ignition success rate")
    axis.set_ylim(-0.04, 1.04)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    paths["success_scatter"] = _save_figure(fig, figure_dir, "initial_state_syn_vs_success")

    metric_rows = []
    for group, values in summary["groups"].items():
        if not bool(values.get("valid", False)):
            continue
        metric_rows.append(
            {
                "group": group,
                "proxy_success": float(values["spearman_proxy_success"]),
                "proxy_basin_syn": float(values["spearman_proxy_full_basin_syn"]),
                "top_k_recall": float(values["top_k_recall"]),
            }
        )
    metric_frame = pd.DataFrame(metric_rows)
    fig, axis = plt.subplots(figsize=(8.0, 3.8), constrained_layout=True)
    if not metric_frame.empty:
        x = np.arange(len(metric_frame))
        width = 0.26
        axis.bar(x - width, metric_frame["proxy_success"], width=width, color="#4C78A8", label="rho(proxy, success)")
        axis.bar(x, metric_frame["proxy_basin_syn"], width=width, color="#72B7B2", label="rho(proxy, basin Syn)")
        axis.bar(x + width, metric_frame["top_k_recall"], width=width, color="#F28E2B", label="Top-k recall")
        axis.set_xticks(x, labels=metric_frame["group"], rotation=25, ha="right")
    axis.axhline(0.0, color="0.25", linewidth=0.8)
    axis.set_ylim(-1.05, 1.05)
    axis.set_ylabel("Metric")
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    paths["summary"] = _save_figure(fig, figure_dir, "initial_state_syn_summary")

    if report_asset_dir is not None:
        report_asset_dir = Path(report_asset_dir)
        report_asset_dir.mkdir(parents=True, exist_ok=True)
        for item in paths.values():
            target = report_asset_dir / f"part3_{item['png'].name}"
            shutil.copyfile(item["png"], target)
            item["report_png"] = target
    return paths


def plot_transport_map_initial_state_syn_results(
    results: dict[str, Any],
    config: TransportMapInitialStateSynConfig,
    *,
    figure_dir: Path | None = None,
    report_asset_dir: Path | None = None,
) -> dict[str, dict[str, Path]]:
    figure_dir = config.output_dir / "figures" if figure_dir is None else Path(figure_dir)
    pair_rows = results["pair_rows"].copy()
    summary = results["summary"]
    paths: dict[str, dict[str, Path]] = {}

    fig, axis = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    palette = {
        "Neural|ER": "#4C78A8",
        "Neural|WS": "#72B7B2",
        "Eco|ER": "#59A14F",
        "Eco|WS": "#F28E2B",
    }
    if not pair_rows.empty:
        for group, subset in pair_rows.groupby("group"):
            axis.scatter(
                subset["tm_initial_syn"],
                subset["success_rate"],
                s=34,
                alpha=0.78,
                edgecolor="white",
                linewidth=0.3,
                color=palette.get(group, "0.4"),
                label=group,
            )
    axis.axvline(0.0, color="0.35", linewidth=0.8)
    axis.set_xlabel("Transport-map initial-state Syn (bits)")
    axis.set_ylabel("Grid ignition success rate")
    axis.set_ylim(-0.04, 1.04)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    paths["success_scatter"] = _save_figure(fig, figure_dir, "transport_map_initial_state_syn_vs_success")

    metric_rows = []
    for group, values in summary["groups"].items():
        if not bool(values.get("valid", False)):
            continue
        metric_rows.append(
            {
                "group": group,
                "tm_success": float(values["spearman_tm_syn_success"]),
                "tm_basin_syn": float(values["spearman_tm_syn_full_basin_syn"]),
                "top_k_recall": float(values["top_k_recall"]),
            }
        )
    metric_frame = pd.DataFrame(metric_rows)
    fig, axis = plt.subplots(figsize=(8.0, 3.8), constrained_layout=True)
    if not metric_frame.empty:
        x = np.arange(len(metric_frame))
        width = 0.26
        axis.bar(x - width, metric_frame["tm_success"], width=width, color="#4C78A8", label="rho(TM Syn, success)")
        axis.bar(x, metric_frame["tm_basin_syn"], width=width, color="#72B7B2", label="rho(TM Syn, basin Syn)")
        axis.bar(x + width, metric_frame["top_k_recall"], width=width, color="#F28E2B", label="Top-k recall")
        axis.set_xticks(x, labels=metric_frame["group"], rotation=25, ha="right")
    axis.axhline(0.0, color="0.25", linewidth=0.8)
    axis.set_ylim(-1.05, 1.05)
    axis.set_ylabel("Metric")
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    paths["summary"] = _save_figure(fig, figure_dir, "transport_map_initial_state_syn_summary")

    if report_asset_dir is not None:
        report_asset_dir = Path(report_asset_dir)
        report_asset_dir.mkdir(parents=True, exist_ok=True)
        for item in paths.values():
            target = report_asset_dir / f"part3_{item['png'].name}"
            shutil.copyfile(item["png"], target)
            item["report_png"] = target
    return paths


def main() -> None:
    config = NetworkBasinPairIgnitionConfig()
    results = run_network_basin_pair_ensemble(config, force=False)
    plot_network_basin_results(results, config, report_asset_dir=REPO_ROOT / "docs" / "reports" / "assets")
    proxy_config = InitialStateSynProxyConfig()
    proxy_results = run_initial_state_syn_proxy_experiment(proxy_config, force=False)
    plot_initial_state_syn_proxy_results(proxy_results, proxy_config, report_asset_dir=REPO_ROOT / "docs" / "reports" / "assets")
    tm_config = TransportMapInitialStateSynConfig()
    tm_results = run_transport_map_initial_state_syn_experiment(tm_config, force=False)
    plot_transport_map_initial_state_syn_results(tm_results, tm_config, report_asset_dir=REPO_ROOT / "docs" / "reports" / "assets")
    print(
        json.dumps(
            {
                "network_basin": results["summary"],
                "initial_state_proxy": proxy_results["summary"],
                "transport_map_initial_state": tm_results["summary"],
            },
            indent=2,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()
