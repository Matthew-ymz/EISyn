from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import pandas as pd

from .effective_information import estimate_state_space_pair_synergy


@dataclass(frozen=True)
class SinSynergyIgnitionConfig:
    output_dir: Path
    nonlinearity: str = "log1p_product"
    alpha_low: float = 0.05
    alpha_high: float = 0.9
    alpha: float = 0.9
    alpha_grid: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    )
    gain: float = 1.0
    source_decay: float = 1.0
    target_decay: float = 1.0
    source_weight: float = 1.0
    secondary_source_weight: float = 0.01
    t_force: float = 8.0
    dt: float = 0.02
    cost_grid: np.ndarray = field(default_factory=lambda: np.linspace(0.0, 4.0, 33))
    duration_grid: np.ndarray = field(default_factory=lambda: np.linspace(0.25, 12.0, 32))
    ei_time_grid: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]))
    ei_sample_count: int = 1200
    ei_cost_low: float = 0.0
    ei_cost_high: float = 2.0
    ei_seed: int = 314
    ei_target_noise_fraction: float = 0.1


@dataclass(frozen=True)
class MultiNodeSynergyIgnitionConfig:
    output_dir: Path
    source_nodes: tuple[int, ...] = (0, 1, 2, 3, 4)
    target_node: int = 5
    embedded_pair_weights: tuple[tuple[int, int, float], ...] = (
        (0, 1, 1.0),
        (2, 3, 0.65),
        (1, 4, 0.4),
    )
    network_edges: tuple[tuple[int, int, float], ...] = (
        (0, 1, 0.12),
        (1, 2, 0.12),
        (2, 0, 0.12),
        (0, 3, 0.08),
        (1, 3, 0.06),
        (2, 4, 0.08),
        (3, 4, 0.10),
        (3, 5, 0.08),
        (4, 5, 0.12),
    )
    feedback_loop_edges: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 0))
    single_source_weights: tuple[float, ...] = (1.0, 0.35, 0.75, 0.25, 0.45)
    alpha: float = 0.9
    gain: float = 1.0
    source_decay: float = 1.0
    target_decay: float = 1.0
    t_force: float = 8.0
    dt: float = 0.02
    cost_grid: np.ndarray = field(default_factory=lambda: np.linspace(0.0, 4.0, 33))
    ei_sample_count: int = 1200
    ei_cost_low: float = 0.0
    ei_cost_high: float = 2.0
    ei_seed: int = 2718
    ei_target_noise_fraction: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        source_nodes = tuple(int(node) for node in self.source_nodes)
        if len(source_nodes) < 2:
            raise ValueError("source_nodes must contain at least two nodes.")
        if len(set(source_nodes)) != len(source_nodes):
            raise ValueError("source_nodes must be unique.")
        if int(self.target_node) in source_nodes:
            raise ValueError("target_node must not be a source node.")
        if len(self.single_source_weights) != len(source_nodes):
            raise ValueError("single_source_weights must match source_nodes.")
        normalized_pairs = []
        valid_source_set = set(source_nodes)
        for left, right, weight in self.embedded_pair_weights:
            pair = tuple(sorted((int(left), int(right))))
            if pair[0] == pair[1]:
                raise ValueError("embedded synergy pairs must not be self-pairs.")
            if pair[0] not in valid_source_set or pair[1] not in valid_source_set:
                raise ValueError("embedded synergy pairs must use source nodes.")
            normalized_pairs.append((pair[0], pair[1], float(weight)))
        node_set = set(source_nodes) | {int(self.target_node)}
        normalized_edges = []
        for src, dst, weight in self.network_edges:
            src_i = int(src)
            dst_i = int(dst)
            if src_i == dst_i:
                raise ValueError("network_edges must not contain self loops.")
            if src_i not in node_set or dst_i not in node_set:
                raise ValueError("network_edges must use configured nodes.")
            normalized_edges.append((src_i, dst_i, float(weight)))
        feedback_edges = tuple((int(src), int(dst)) for src, dst in self.feedback_loop_edges)
        edge_pairs = {(src, dst) for src, dst, _ in normalized_edges}
        if not set(feedback_edges).issubset(edge_pairs):
            raise ValueError("feedback_loop_edges must be present in network_edges.")
        incoming_nodes = {dst for _, dst, _ in normalized_edges}
        if not node_set.issubset(incoming_nodes):
            raise ValueError("every configured node must have at least one incoming edge.")
        object.__setattr__(self, "source_nodes", source_nodes)
        object.__setattr__(self, "target_node", int(self.target_node))
        object.__setattr__(self, "embedded_pair_weights", tuple(normalized_pairs))
        object.__setattr__(self, "network_edges", tuple(normalized_edges))
        object.__setattr__(self, "feedback_loop_edges", feedback_edges)
        object.__setattr__(self, "single_source_weights", tuple(float(value) for value in self.single_source_weights))
        object.__setattr__(self, "cost_grid", np.asarray(self.cost_grid, dtype=float))
        if self.cost_grid.ndim != 1 or len(self.cost_grid) < 2:
            raise ValueError("cost_grid must be a one-dimensional grid with at least two values.")
        if self.ei_sample_count < 2:
            raise ValueError("ei_sample_count must be at least two.")
        if self.ei_cost_high <= self.ei_cost_low:
            raise ValueError("ei_cost_high must be greater than ei_cost_low.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.t_force < 0.0:
            raise ValueError("t_force must be nonnegative.")


IGNITION_LABELS = {
    "none": "No ignition",
    "node_0": "Node 0 ignition",
    "node_1": "Node 1 ignition",
    "pair_0_1": "Pair ignition",
}

IGNITION_COLORS = {
    "none": "0.45",
    "node_0": "#2563eb",
    "node_1": "#dc2626",
    "pair_0_1": "#16a34a",
}


def sin_output_paths(config: SinSynergyIgnitionConfig) -> dict[str, Path]:
    output_dir = Path(config.output_dir)
    return {
        "ignition_csv": output_dir / "sin_synergy_ignition_response.csv",
        "ei_csv": output_dir / "sin_synergy_ei_decomposition.csv",
        "ei_samples_csv": output_dir / "sin_synergy_ei_samples.csv",
        "pair_duration_surface_csv": output_dir / "sin_synergy_pair_duration_surface.csv",
        "ei_time_curve_csv": output_dir / "sin_synergy_ei_time_curve.csv",
        "alpha_sweep_csv": output_dir / "sin_synergy_alpha_sweep.csv",
        "manifest_json": output_dir / "sin_synergy_ignition_manifest.json",
    }


def multi_node_output_paths(config: MultiNodeSynergyIgnitionConfig) -> dict[str, Path]:
    output_dir = Path(config.output_dir)
    return {
        "pair_synergy_csv": output_dir / "multi_node_pair_synergy.csv",
        "pair_ignition_csv": output_dir / "multi_node_pair_ignition_response.csv",
        "pair_summary_csv": output_dir / "multi_node_pair_summary.csv",
        "manifest_json": output_dir / "multi_node_synergy_manifest.json",
    }


def sin_model_parameter_row(
    config: SinSynergyIgnitionConfig,
    alpha: float | None = None,
) -> dict[str, float | str]:
    return {
        "nonlinearity": config.nonlinearity,
        "alpha": float(config.alpha if alpha is None else alpha),
        "gain": float(config.gain),
        "source_decay": float(config.source_decay),
        "target_decay": float(config.target_decay),
        "source_weight": float(config.source_weight),
        "secondary_source_weight": float(config.secondary_source_weight),
    }


def multi_node_model_parameter_row(config: MultiNodeSynergyIgnitionConfig) -> dict[str, float | str]:
    return {
        "alpha": float(config.alpha),
        "gain": float(config.gain),
        "source_decay": float(config.source_decay),
        "target_decay": float(config.target_decay),
        "source_nodes": ",".join(str(node) for node in config.source_nodes),
        "target_node": int(config.target_node),
        "embedded_pair_weights": json.dumps(
            [[int(left), int(right), float(weight)] for left, right, weight in config.embedded_pair_weights]
        ),
        "network_edges": json.dumps(
            [[int(src), int(dst), float(weight)] for src, dst, weight in config.network_edges]
        ),
        "single_source_weights": json.dumps([float(value) for value in config.single_source_weights]),
    }


def sin_cache_matches_current_parameters(
    frame: pd.DataFrame,
    config: SinSynergyIgnitionConfig,
    *,
    alpha: float | None = None,
) -> bool:
    if frame.empty:
        return False
    for column, value in sin_model_parameter_row(config, alpha).items():
        if column not in frame.columns:
            return False
        if isinstance(value, str):
            if str(frame[column].iloc[0]) != value:
                return False
        elif not np.allclose(frame[column].iloc[0], value):
            return False
    return True


def sin_ei_cache_matches_current_parameters(
    frame: pd.DataFrame,
    config: SinSynergyIgnitionConfig,
    *,
    alpha: float | None = None,
) -> bool:
    return (
        sin_cache_matches_current_parameters(frame, config, alpha=alpha)
        and "target_noise_fraction" in frame.columns
        and np.allclose(frame["target_noise_fraction"].iloc[0], config.ei_target_noise_fraction)
        and "delta_low" in frame.columns
        and "delta_high" in frame.columns
        and np.allclose(frame["delta_low"].iloc[0], config.ei_cost_low)
        and np.allclose(frame["delta_high"].iloc[0], config.ei_cost_high)
    )


def multi_node_source_pairs(config: MultiNodeSynergyIgnitionConfig) -> list[tuple[int, int]]:
    nodes = tuple(int(node) for node in config.source_nodes)
    return [(left, right) for index, left in enumerate(nodes) for right in nodes[index + 1 :]]


def make_multi_node_synergy_model(config: MultiNodeSynergyIgnitionConfig) -> dict[str, Any]:
    source_nodes = tuple(int(node) for node in config.source_nodes)
    target_node = int(config.target_node)
    node_count = max((target_node, *source_nodes)) + 1
    embedded = {
        tuple(sorted((int(left), int(right)))): float(weight)
        for left, right, weight in config.embedded_pair_weights
    }
    single_weights = {
        int(node): float(weight)
        for node, weight in zip(source_nodes, config.single_source_weights, strict=True)
    }
    adjacency = np.zeros((node_count, node_count), dtype=float)
    for src, dst, weight in config.network_edges:
        adjacency[int(dst), int(src)] = float(weight)
    decay = np.full(node_count, float(config.source_decay), dtype=float)
    decay[target_node] = float(config.target_decay)
    return {
        "source_nodes": source_nodes,
        "target_node": target_node,
        "node_count": node_count,
        "adjacency": adjacency,
        "network_edges": [(int(src), int(dst), float(weight)) for src, dst, weight in config.network_edges],
        "feedback_loop_edges": [(int(src), int(dst)) for src, dst in config.feedback_loop_edges],
        "decay": decay,
        "embedded_pair_weights": embedded,
        "single_source_weights": single_weights,
        "alpha": float(config.alpha),
        "gain": float(config.gain),
        "source_decay": float(config.source_decay),
        "target_decay": float(config.target_decay),
    }


def multi_node_synergy_forcing(
    state: dict[int, float] | np.ndarray,
    model: dict[str, Any],
) -> float:
    if isinstance(state, np.ndarray):
        values = {int(node): float(state[int(node)]) for node in model["source_nodes"]}
    else:
        values = {int(node): float(value) for node, value in state.items()}
    joint = 0.0
    for (left, right), weight in model["embedded_pair_weights"].items():
        joint += float(weight) * float(monotone_synergy_nonlinearity(values.get(left, 0.0) * values.get(right, 0.0), model))
    single = sum(
        float(model["single_source_weights"][node]) * values.get(node, 0.0)
        for node in model["source_nodes"]
    )
    return float(float(model["alpha"]) * joint + (1.0 - float(model["alpha"])) * single)


def multi_node_target_value(
    fixed_values: dict[int, float],
    model: dict[str, Any],
    *,
    duration: float,
) -> float:
    return float(
        simulate_multi_node_fixed_sources(
            fixed_values=fixed_values,
            model=model,
            t_force=float(duration),
            dt=0.02,
        )["target_end"]
    )


def multi_node_rhs(x: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    state = np.asarray(x, dtype=float)
    target_node = int(model["target_node"])
    dx = -np.asarray(model["decay"], dtype=float) * state
    dx += np.asarray(model["adjacency"], dtype=float) @ state
    dx[target_node] += multi_node_synergy_forcing(state, model)
    return dx


def rk4_step_multi_node(
    x: np.ndarray,
    dt: float,
    model: dict[str, Any],
    fixed_values: dict[int, float],
) -> np.ndarray:
    def rhs_with_clamp(state: np.ndarray) -> np.ndarray:
        y = np.maximum(state.copy(), 0.0)
        for node, value in fixed_values.items():
            y[int(node)] = float(value)
        dy = multi_node_rhs(y, model)
        for node in fixed_values:
            dy[int(node)] = 0.0
        return dy

    k1 = rhs_with_clamp(x)
    k2 = rhs_with_clamp(x + 0.5 * dt * k1)
    k3 = rhs_with_clamp(x + 0.5 * dt * k2)
    k4 = rhs_with_clamp(x + dt * k3)
    next_x = np.maximum(x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0)
    for node, value in fixed_values.items():
        next_x[int(node)] = float(value)
    return next_x


def simulate_multi_node_fixed_sources(
    *,
    fixed_values: dict[int, float],
    model: dict[str, Any],
    t_force: float,
    dt: float,
) -> dict[str, Any]:
    clean_fixed = {int(node): float(value) for node, value in fixed_values.items()}
    node_count = int(model["node_count"])
    target_node = int(model["target_node"])
    if target_node in clean_fixed:
        raise ValueError("target node must not be fixed during multi-node ignition.")

    x = np.zeros(node_count, dtype=float)
    for node, value in clean_fixed.items():
        x[int(node)] = float(value)
    target_values = [float(x[target_node])]
    times = [0.0]
    t = 0.0
    while t < float(t_force) - 1e-12:
        step = min(float(dt), float(t_force) - t)
        x = rk4_step_multi_node(x, step, model, clean_fixed)
        t += step
        target_values.append(float(x[target_node]))
        times.append(float(t))
    target_series = np.asarray(target_values, dtype=float)
    time_grid = np.asarray(times, dtype=float)
    return {
        "fixed_values": clean_fixed,
        "final_state": x,
        "target_end": float(x[target_node]),
        "target_peak": float(np.max(target_series)),
        "target_auc": float(np.trapz(target_series, x=time_grid)),
    }


def _multi_node_pair_is_embedded(pair: tuple[int, int], model: dict[str, Any]) -> bool:
    return tuple(sorted((int(pair[0]), int(pair[1])))) in model["embedded_pair_weights"]


def estimate_multi_node_pair_synergy(
    config: MultiNodeSynergyIgnitionConfig,
    *,
    force: bool = False,
) -> pd.DataFrame:
    paths = multi_node_output_paths(config)
    if paths["pair_synergy_csv"].exists() and not force:
        cached = pd.read_csv(paths["pair_synergy_csv"])
        if {"pair_i", "pair_j", "joint_ei", "synergy", "synergy_ratio", "network_edges"}.issubset(cached.columns):
            return cached

    model = make_multi_node_synergy_model(config)
    rng = np.random.default_rng(config.ei_seed)
    source_nodes = tuple(model["source_nodes"])
    source_samples = rng.uniform(
        float(config.ei_cost_low),
        float(config.ei_cost_high),
        size=(int(config.ei_sample_count), len(source_nodes)),
    )
    source_index = {node: index for index, node in enumerate(source_nodes)}
    actual_pairs = multi_node_source_pairs(config)
    estimator_pairs = [(source_index[left], source_index[right]) for left, right in actual_pairs]
    target = np.array(
        [
            simulate_multi_node_fixed_sources(
                fixed_values={node: float(sample[source_index[node]]) for node in source_nodes},
                model=model,
                t_force=config.t_force,
                dt=config.dt,
            )["target_end"]
            for sample in source_samples
        ],
        dtype=float,
    )[:, None]

    estimated = estimate_state_space_pair_synergy(
        source_samples,
        target,
        pairs=estimator_pairs,
        target_noise_fraction=config.ei_target_noise_fraction,
        seed=config.ei_seed + 1,
    )
    rows = []
    inverse_source_index = {index: node for node, index in source_index.items()}
    for row in estimated["pair_rows"]:
        pair = (inverse_source_index[int(row["pair_i"])], inverse_source_index[int(row["pair_j"])])
        weight = float(model["embedded_pair_weights"].get(pair, 0.0))
        rows.append(
            {
                "pair_i": int(pair[0]),
                "pair_j": int(pair[1]),
                "left_ei": float(row["left_ei"]),
                "right_ei": float(row["right_ei"]),
                "joint_ei": float(row["joint_ei"]),
                "synergy": float(row["synergy"]),
                "synergy_ratio": float(row["synergy_ratio"]),
                "embedded_synergy_weight": weight,
                "is_embedded_synergy_pair": bool(weight > 0.0),
                "sample_count": int(config.ei_sample_count),
                "delta_low": float(config.ei_cost_low),
                "delta_high": float(config.ei_cost_high),
                "target_mean": float(target[:, 0].mean()),
                "target_std": float(target[:, 0].std(ddof=1)),
                "target_noise_fraction": float(config.ei_target_noise_fraction),
                **multi_node_model_parameter_row(config),
            }
        )
    synergy_df = pd.DataFrame(rows).sort_values(["synergy", "pair_i", "pair_j"], ascending=[False, True, True]).reset_index(drop=True)
    synergy_df["rank_synergy"] = np.arange(1, len(synergy_df) + 1, dtype=int)
    synergy_df["is_embedded_synergy_pair"] = synergy_df["is_embedded_synergy_pair"].astype(object)
    paths["pair_synergy_csv"].parent.mkdir(parents=True, exist_ok=True)
    synergy_df.to_csv(paths["pair_synergy_csv"], index=False)
    return synergy_df


def _evaluate_multi_node_pair_ignition(
    *,
    pair: tuple[int, int],
    ignition: str,
    total_cost: float,
    model: dict[str, Any],
    t_force: float,
    dt: float,
) -> dict[str, Any]:
    left, right = tuple(sorted((int(pair[0]), int(pair[1]))))
    if ignition == "none":
        fixed_values: dict[int, float] = {}
        delta_i, delta_j = 0.0, 0.0
    elif ignition == "single_i":
        fixed_values = {left: float(total_cost)}
        delta_i, delta_j = float(total_cost), 0.0
    elif ignition == "single_j":
        fixed_values = {right: float(total_cost)}
        delta_i, delta_j = 0.0, float(total_cost)
    elif ignition == "pair":
        delta_i = float(total_cost) / 2.0
        delta_j = float(total_cost) / 2.0
        fixed_values = {left: delta_i, right: delta_j}
    else:
        raise ValueError(f"Unknown ignition mode: {ignition}")
    result = simulate_multi_node_fixed_sources(
        fixed_values=fixed_values,
        model=model,
        t_force=t_force,
        dt=dt,
    )
    return {
        "pair_i": left,
        "pair_j": right,
        "ignition": ignition,
        "total_cost": float(total_cost),
        "delta_i": float(delta_i),
        "delta_j": float(delta_j),
        "target_end": float(result["target_end"]),
        "target_peak": float(result["target_peak"]),
        "target_auc": float(result["target_auc"]),
        "embedded_synergy_weight": float(model["embedded_pair_weights"].get((left, right), 0.0)),
        "is_embedded_synergy_pair": bool(_multi_node_pair_is_embedded((left, right), model)),
    }


def build_multi_node_pair_ignition_table(
    config: MultiNodeSynergyIgnitionConfig,
    *,
    force: bool = False,
) -> pd.DataFrame:
    paths = multi_node_output_paths(config)
    if paths["pair_ignition_csv"].exists() and not force:
        cached = pd.read_csv(paths["pair_ignition_csv"])
        if {"pair_i", "pair_j", "ignition", "total_cost", "target_end", "pair_gain_ratio", "network_edges"}.issubset(cached.columns):
            return cached

    model = make_multi_node_synergy_model(config)
    rows = []
    for pair in multi_node_source_pairs(config):
        for total_cost in config.cost_grid:
            for ignition in ("none", "single_i", "single_j", "pair"):
                rows.append(
                    _evaluate_multi_node_pair_ignition(
                        pair=pair,
                        ignition=ignition,
                        total_cost=float(total_cost),
                        model=model,
                        t_force=config.t_force,
                        dt=config.dt,
                    )
                )
    response = pd.DataFrame(rows).assign(**multi_node_model_parameter_row(config), t_force=float(config.t_force))
    comparison_frames = []
    for pair, group in response.groupby(["pair_i", "pair_j"], sort=False):
        wide = group.pivot(index="total_cost", columns="ignition", values="target_end")
        best_single = pd.concat([wide["single_i"], wide["single_j"]], axis=1).max(axis=1)
        comparison_frames.append(
            pd.DataFrame(
                {
                    "pair_i": pair[0],
                    "pair_j": pair[1],
                    "total_cost": wide.index.to_numpy(dtype=float),
                    "pair_surplus_target_end": wide["pair"] - wide["single_i"] - wide["single_j"] + wide["none"],
                    "best_single_target_end": best_single,
                    "pair_gain_ratio": wide["pair"] / best_single.replace(0.0, np.nan),
                }
            )
        )
    comparison = pd.concat(comparison_frames, ignore_index=True)
    response = response.merge(comparison, on=["pair_i", "pair_j", "total_cost"], how="left")
    response["is_embedded_synergy_pair"] = response["is_embedded_synergy_pair"].astype(object)
    paths["pair_ignition_csv"].parent.mkdir(parents=True, exist_ok=True)
    response.to_csv(paths["pair_ignition_csv"], index=False)
    return response


def build_multi_node_pair_summary(
    config: MultiNodeSynergyIgnitionConfig,
    *,
    force: bool = False,
) -> pd.DataFrame:
    paths = multi_node_output_paths(config)
    if paths["pair_summary_csv"].exists() and not force:
        cached = pd.read_csv(paths["pair_summary_csv"])
        if {"pair_i", "pair_j", "synergy", "pair_response_at_max_cost", "network_edges"}.issubset(cached.columns):
            return cached

    synergy = estimate_multi_node_pair_synergy(config, force=force)
    ignition = build_multi_node_pair_ignition_table(config, force=force)
    max_cost = float(np.max(config.cost_grid))
    pair_at_max = ignition.loc[
        ignition["ignition"].eq("pair") & np.isclose(ignition["total_cost"], max_cost),
        [
            "pair_i",
            "pair_j",
            "target_end",
            "pair_surplus_target_end",
            "best_single_target_end",
            "pair_gain_ratio",
        ],
    ].rename(
        columns={
            "target_end": "pair_response_at_max_cost",
            "pair_surplus_target_end": "pair_surplus_at_max_cost",
            "best_single_target_end": "best_single_response_at_max_cost",
            "pair_gain_ratio": "pair_gain_ratio_at_max_cost",
        }
    )
    summary = synergy.merge(pair_at_max, on=["pair_i", "pair_j"], how="left")
    finite_response = summary.replace([np.inf, -np.inf], np.nan).dropna(subset=["synergy", "pair_response_at_max_cost"])
    spearman_response = float(finite_response["synergy"].corr(finite_response["pair_response_at_max_cost"], method="spearman")) if len(finite_response) >= 2 else np.nan
    finite_gain = summary.replace([np.inf, -np.inf], np.nan).dropna(subset=["synergy", "pair_gain_ratio_at_max_cost"])
    spearman_gain = float(finite_gain["synergy"].corr(finite_gain["pair_gain_ratio_at_max_cost"], method="spearman")) if len(finite_gain) >= 2 else np.nan
    finite_surplus = summary.replace([np.inf, -np.inf], np.nan).dropna(subset=["synergy", "pair_surplus_at_max_cost"])
    spearman_surplus = float(finite_surplus["synergy"].corr(finite_surplus["pair_surplus_at_max_cost"], method="spearman")) if len(finite_surplus) >= 2 else np.nan
    summary["spearman_synergy_pair_response"] = spearman_response
    summary["spearman_synergy_pair_gain_ratio"] = spearman_gain
    summary["spearman_synergy_pair_surplus"] = spearman_surplus
    summary["is_embedded_synergy_pair"] = summary["is_embedded_synergy_pair"].astype(object)
    summary.to_csv(paths["pair_summary_csv"], index=False)
    return summary


def load_or_run_multi_node_synergy_experiment(
    config: MultiNodeSynergyIgnitionConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    model = make_multi_node_synergy_model(config)
    pair_synergy = estimate_multi_node_pair_synergy(config, force=force)
    pair_ignition_response = build_multi_node_pair_ignition_table(config, force=force)
    pair_summary = build_multi_node_pair_summary(config, force=force)
    paths = multi_node_output_paths(config)
    paths["manifest_json"].write_text(
        json.dumps(multi_node_synergy_manifest(config, paths), indent=2),
        encoding="utf-8",
    )
    return {
        "model": model,
        "pair_synergy": pair_synergy,
        "pair_ignition_response": pair_ignition_response,
        "pair_summary": pair_summary,
        "paths": paths,
    }


def summarize_multi_node_synergy_results(results: dict[str, Any]) -> dict[str, Any]:
    summary = results["pair_summary"]
    assert isinstance(summary, pd.DataFrame)
    top_syn = summary.sort_values("synergy", ascending=False).iloc[0]
    top_response = summary.sort_values("pair_response_at_max_cost", ascending=False).iloc[0]
    return {
        "top_synergy_pair": (int(top_syn["pair_i"]), int(top_syn["pair_j"])),
        "top_synergy": float(top_syn["synergy"]),
        "top_synergy_pair_response": float(top_syn["pair_response_at_max_cost"]),
        "top_response_pair": (int(top_response["pair_i"]), int(top_response["pair_j"])),
        "top_pair_response": float(top_response["pair_response_at_max_cost"]),
        "spearman_synergy_pair_response": float(summary["spearman_synergy_pair_response"].iloc[0]),
        "spearman_synergy_pair_gain_ratio": float(summary["spearman_synergy_pair_gain_ratio"].iloc[0]),
        "spearman_synergy_pair_surplus": float(summary["spearman_synergy_pair_surplus"].iloc[0]),
    }


def multi_node_synergy_manifest(
    config: MultiNodeSynergyIgnitionConfig,
    paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    if paths is None:
        paths = multi_node_output_paths(config)
    return {
        "experiment": "controlled_six_node_pair_synergy_ignition",
        "ode": "dx_i/dt=-lambda_i*x_i+sum_j A_ij*x_j plus target pair-synergy and singleton readout terms",
        "source_nodes": [int(node) for node in config.source_nodes],
        "target_node": int(config.target_node),
        "embedded_pair_weights": [[int(left), int(right), float(weight)] for left, right, weight in config.embedded_pair_weights],
        "network_edges": [[int(src), int(dst), float(weight)] for src, dst, weight in config.network_edges],
        "feedback_loop_edges": [[int(src), int(dst)] for src, dst in config.feedback_loop_edges],
        "single_source_weights": [float(value) for value in config.single_source_weights],
        "alpha": float(config.alpha),
        "gain": float(config.gain),
        "source_decay": float(config.source_decay),
        "target_decay": float(config.target_decay),
        "t_force": float(config.t_force),
        "dt": float(config.dt),
        "cost_grid": [float(value) for value in config.cost_grid],
        "ei_sample_count": int(config.ei_sample_count),
        "ei_delta_interval": [float(config.ei_cost_low), float(config.ei_cost_high)],
        "ei_target_noise_fraction": float(config.ei_target_noise_fraction),
        "pair_synergy_csv": str(paths["pair_synergy_csv"]),
        "pair_ignition_csv": str(paths["pair_ignition_csv"]),
        "pair_summary_csv": str(paths["pair_summary_csv"]),
    }


def make_sin_synergy_ignition_model(
    config: SinSynergyIgnitionConfig | None = None,
    *,
    alpha: float | None = None,
    gain: float | None = None,
    source_decay: float | None = None,
    target_decay: float | None = None,
    source_weight: float | None = None,
    secondary_source_weight: float | None = None,
) -> dict[str, float | str]:
    if config is None:
        config = SinSynergyIgnitionConfig(output_dir=Path("."))
    return {
        "nonlinearity": config.nonlinearity,
        "alpha": float(config.alpha if alpha is None else alpha),
        "gain": float(config.gain if gain is None else gain),
        "source_decay": float(config.source_decay if source_decay is None else source_decay),
        "target_decay": float(config.target_decay if target_decay is None else target_decay),
        "source_weight": float(config.source_weight if source_weight is None else source_weight),
        "secondary_source_weight": float(
            config.secondary_source_weight if secondary_source_weight is None else secondary_source_weight
        ),
    }


def monotone_synergy_nonlinearity(
    z: float | np.ndarray,
    model: dict[str, float | str],
) -> float | np.ndarray:
    return np.log1p(np.maximum(float(model["gain"]) * np.asarray(z, dtype=float), 0.0))


def sin_synergy_forcing(
    delta_node_0: float,
    delta_node_1: float,
    model: dict[str, float | str],
) -> float:
    joint_input = monotone_synergy_nonlinearity(float(delta_node_0) * float(delta_node_1), model)
    single_input = float(model["source_weight"]) * (
        float(delta_node_0) + float(model["secondary_source_weight"]) * float(delta_node_1)
    )
    return float(float(model["alpha"]) * joint_input + (1.0 - float(model["alpha"])) * single_input)


def sin_synergy_rhs(x: np.ndarray, model: dict[str, float | str]) -> np.ndarray:
    dx = np.empty(3, dtype=float)
    dx[0] = -float(model["source_decay"]) * x[0]
    dx[1] = -float(model["source_decay"]) * x[1]
    dx[2] = -float(model["target_decay"]) * x[2] + sin_synergy_forcing(float(x[0]), float(x[1]), model)
    return dx


def sin_synergy_target_value(
    *,
    delta_node_0: float,
    delta_node_1: float,
    duration: float,
    model: dict[str, float | str],
) -> float:
    forcing = sin_synergy_forcing(delta_node_0, delta_node_1, model)
    decay = float(model["target_decay"])
    return float((forcing / decay) * (1.0 - np.exp(-decay * float(duration))))


def rk4_step_sin_synergy(
    x: np.ndarray,
    dt: float,
    model: dict[str, float | str],
    fixed_values: dict[int, float],
) -> np.ndarray:
    def rhs_with_clamp(state: np.ndarray) -> np.ndarray:
        y = state.copy()
        for node, value in fixed_values.items():
            y[int(node)] = float(value)
        dy = sin_synergy_rhs(y, model)
        for node in fixed_values:
            dy[int(node)] = 0.0
        return dy

    k1 = rhs_with_clamp(x)
    k2 = rhs_with_clamp(x + 0.5 * dt * k1)
    k3 = rhs_with_clamp(x + 0.5 * dt * k2)
    k4 = rhs_with_clamp(x + dt * k3)
    next_x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    for node, value in fixed_values.items():
        next_x[int(node)] = float(value)
    return next_x


def simulate_sin_synergy_fixed_sources(
    *,
    delta_node_0: float,
    delta_node_1: float,
    model: dict[str, float | str],
    t_force: float,
    dt: float,
) -> dict[str, float]:
    fixed_values = {0: float(delta_node_0), 1: float(delta_node_1)}
    x = np.array([float(delta_node_0), float(delta_node_1), 0.0], dtype=float)
    target_values = [float(x[2])]
    times = [0.0]
    t = 0.0
    while t < float(t_force) - 1e-12:
        step = min(float(dt), float(t_force) - t)
        x = rk4_step_sin_synergy(x, step, model, fixed_values)
        t += step
        target_values.append(float(x[2]))
        times.append(float(t))
    target_series = np.asarray(target_values, dtype=float)
    time_grid = np.asarray(times, dtype=float)
    return {
        "delta_node_0": float(delta_node_0),
        "delta_node_1": float(delta_node_1),
        "target_end": float(x[2]),
        "target_peak": float(np.max(target_series)),
        "target_auc": float(np.trapz(target_series, x=time_grid)),
    }


def evaluate_sin_synergy_ignition(
    *,
    ignition: str,
    total_cost: float,
    model: dict[str, float | str],
    t_force: float,
    dt: float,
) -> dict[str, float | str]:
    if ignition == "none":
        delta_node_0, delta_node_1 = 0.0, 0.0
    elif ignition == "node_0":
        delta_node_0, delta_node_1 = float(total_cost), 0.0
    elif ignition == "node_1":
        delta_node_0, delta_node_1 = 0.0, float(total_cost)
    elif ignition == "pair_0_1":
        delta_node_0, delta_node_1 = float(total_cost) / 2.0, float(total_cost) / 2.0
    else:
        raise ValueError(f"Unknown ignition mode: {ignition}")

    result = simulate_sin_synergy_fixed_sources(
        delta_node_0=delta_node_0,
        delta_node_1=delta_node_1,
        model=model,
        t_force=t_force,
        dt=dt,
    )
    return {"ignition": ignition, "total_cost": float(total_cost), **result}


def add_ignition_comparison_columns(response: pd.DataFrame) -> pd.DataFrame:
    single_lookup = response.pivot(index="total_cost", columns="ignition", values="target_end")
    baseline = single_lookup["none"]
    pair_surplus = single_lookup["pair_0_1"] - single_lookup["node_0"] - single_lookup["node_1"] + baseline
    best_single = pd.concat([single_lookup["node_0"], single_lookup["node_1"]], axis=1).max(axis=1)
    pair_gain_ratio = single_lookup["pair_0_1"] / best_single.replace(0.0, np.nan)
    return response.merge(
        pd.DataFrame(
            {
                "pair_surplus_target_end": pair_surplus,
                "best_single_target_end": best_single,
                "pair_gain_ratio": pair_gain_ratio,
            }
        ),
        left_on="total_cost",
        right_index=True,
        how="left",
    )


def build_sin_synergy_ignition_table(
    config: SinSynergyIgnitionConfig,
    *,
    force: bool = False,
) -> pd.DataFrame:
    paths = sin_output_paths(config)
    if paths["ignition_csv"].exists() and not force:
        cached = pd.read_csv(paths["ignition_csv"])
        if sin_cache_matches_current_parameters(cached, config) and {
            "pair_gain_ratio",
            "best_single_target_end",
        }.issubset(cached.columns):
            return cached
    model = make_sin_synergy_ignition_model(config)
    rows = []
    for total_cost in config.cost_grid:
        for ignition in ("none", "node_0", "node_1", "pair_0_1"):
            rows.append(
                evaluate_sin_synergy_ignition(
                    ignition=ignition,
                    total_cost=float(total_cost),
                    model=model,
                    t_force=config.t_force,
                    dt=config.dt,
                )
            )
    response = pd.DataFrame(rows).assign(**sin_model_parameter_row(config), t_force=float(config.t_force))
    response = add_ignition_comparison_columns(response)
    paths["ignition_csv"].parent.mkdir(parents=True, exist_ok=True)
    response.to_csv(paths["ignition_csv"], index=False)
    return response


def build_sin_synergy_pair_duration_surface(
    config: SinSynergyIgnitionConfig,
    *,
    force: bool = False,
) -> pd.DataFrame:
    paths = sin_output_paths(config)
    if paths["pair_duration_surface_csv"].exists() and not force:
        cached = pd.read_csv(paths["pair_duration_surface_csv"])
        if {"total_cost", "duration", "target_value"}.issubset(cached.columns) and sin_cache_matches_current_parameters(cached, config):
            return cached
    model = make_sin_synergy_ignition_model(config)
    rows = []
    for total_cost in config.cost_grid:
        delta_each = float(total_cost) / 2.0
        for duration in config.duration_grid:
            rows.append(
                {
                    "total_cost": float(total_cost),
                    "duration": float(duration),
                    "delta_node_0": delta_each,
                    "delta_node_1": delta_each,
                    "target_value": sin_synergy_target_value(
                        delta_node_0=delta_each,
                        delta_node_1=delta_each,
                        duration=float(duration),
                        model=model,
                    ),
                }
            )
    surface_df = pd.DataFrame(rows).assign(**sin_model_parameter_row(config))
    surface_df.to_csv(paths["pair_duration_surface_csv"], index=False)
    return surface_df


def estimate_sin_pair_ei_for_sources(
    source_samples: np.ndarray,
    target: np.ndarray,
    *,
    config: SinSynergyIgnitionConfig,
    seed: int,
) -> dict[str, float]:
    pair_summary = estimate_state_space_pair_synergy(
        source_samples,
        target,
        pairs=[(0, 1)],
        target_noise_fraction=config.ei_target_noise_fraction,
        seed=int(seed),
    )["pair_rows"][0]
    return {
        "source_0_ei": float(pair_summary["left_ei"]),
        "source_1_ei": float(pair_summary["right_ei"]),
        "joint_ei": float(pair_summary["joint_ei"]),
        "synergy": float(pair_summary["synergy"]),
        "synergy_ratio": float(pair_summary["synergy_ratio"]),
    }


def estimate_sin_synergy_ei_time_curve(
    config: SinSynergyIgnitionConfig,
    *,
    force: bool = False,
) -> pd.DataFrame:
    paths = sin_output_paths(config)
    if paths["ei_time_curve_csv"].exists() and not force:
        cached = pd.read_csv(paths["ei_time_curve_csv"])
        if {"duration", "source_0_ei", "source_1_ei", "joint_ei", "synergy"}.issubset(cached.columns) and sin_ei_cache_matches_current_parameters(cached, config):
            return cached

    model = make_sin_synergy_ignition_model(config)
    rng = np.random.default_rng(config.ei_seed)
    source_samples = rng.uniform(config.ei_cost_low, config.ei_cost_high, size=(config.ei_sample_count, 2))
    rows = []
    for duration in config.ei_time_grid:
        target = np.array(
            [
                sin_synergy_target_value(
                    delta_node_0=float(delta0),
                    delta_node_1=float(delta1),
                    duration=float(duration),
                    model=model,
                )
                for delta0, delta1 in source_samples
            ],
            dtype=float,
        )[:, None]
        rows.append(
            {
                "duration": float(duration),
                "sample_count": config.ei_sample_count,
                "delta_low": config.ei_cost_low,
                "delta_high": config.ei_cost_high,
                **estimate_sin_pair_ei_for_sources(
                    source_samples,
                    target,
                    config=config,
                    seed=config.ei_seed + int(round(float(duration) * 100)),
                ),
                "target_mean": float(target[:, 0].mean()),
                "target_std": float(target[:, 0].std(ddof=1)),
                "target_noise_fraction": config.ei_target_noise_fraction,
                **sin_model_parameter_row(config),
            }
        )
    time_df = pd.DataFrame(rows)
    time_df.to_csv(paths["ei_time_curve_csv"], index=False)
    return time_df


def estimate_sin_synergy_ei_decomposition(
    config: SinSynergyIgnitionConfig,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sin_output_paths(config)
    if paths["ei_csv"].exists() and paths["ei_samples_csv"].exists() and not force:
        cached_summary = pd.read_csv(paths["ei_csv"])
        if (
            not cached_summary.empty
            and int(cached_summary.loc[0, "sample_count"]) == int(config.ei_sample_count)
            and sin_ei_cache_matches_current_parameters(cached_summary, config)
        ):
            return cached_summary, pd.read_csv(paths["ei_samples_csv"])

    model = make_sin_synergy_ignition_model(config)
    rng = np.random.default_rng(config.ei_seed)
    source_samples = rng.uniform(config.ei_cost_low, config.ei_cost_high, size=(config.ei_sample_count, 2))
    rows = [
        simulate_sin_synergy_fixed_sources(
            delta_node_0=float(delta0),
            delta_node_1=float(delta1),
            model=model,
            t_force=config.t_force,
            dt=config.dt,
        )
        for delta0, delta1 in source_samples
    ]
    sample_df = pd.DataFrame(rows)
    target = sample_df[["target_end"]].to_numpy(dtype=float)
    summary_df = pd.DataFrame(
        [
            {
                "sample_count": config.ei_sample_count,
                "delta_low": config.ei_cost_low,
                "delta_high": config.ei_cost_high,
                **estimate_sin_pair_ei_for_sources(
                    source_samples,
                    target,
                    config=config,
                    seed=config.ei_seed + 1,
                ),
                "target_mean": float(sample_df["target_end"].mean()),
                "target_std": float(sample_df["target_end"].std(ddof=1)),
                "target_noise_fraction": config.ei_target_noise_fraction,
                **sin_model_parameter_row(config),
            }
        ]
    )
    summary_df.to_csv(paths["ei_csv"], index=False)
    sample_df.to_csv(paths["ei_samples_csv"], index=False)
    return summary_df, sample_df


def build_sin_synergy_alpha_sweep(
    config: SinSynergyIgnitionConfig,
    *,
    force: bool = False,
) -> pd.DataFrame:
    paths = sin_output_paths(config)
    if paths["alpha_sweep_csv"].exists() and not force:
        cached = pd.read_csv(paths["alpha_sweep_csv"])
        required = {
            "alpha",
            "source_0_ei",
            "source_1_ei",
            "joint_ei",
            "synergy",
            "synergy_ratio",
            "best_single_response",
            "pair_response",
            "pair_gain_ratio",
            "pair_surplus",
        }
        high_alpha = cached.loc[cached["alpha"].eq(config.alpha_high)].head(1)
        if required.issubset(cached.columns) and sin_ei_cache_matches_current_parameters(
            high_alpha,
            config,
            alpha=config.alpha_high,
        ):
            return cached

    rng = np.random.default_rng(config.ei_seed)
    source_samples = rng.uniform(config.ei_cost_low, config.ei_cost_high, size=(config.ei_sample_count, 2))
    rows = []
    for alpha in config.alpha_grid:
        model = make_sin_synergy_ignition_model(config, alpha=float(alpha))
        target = np.array(
            [
                sin_synergy_target_value(
                    delta_node_0=float(delta0),
                    delta_node_1=float(delta1),
                    duration=config.t_force,
                    model=model,
                )
                for delta0, delta1 in source_samples
            ],
            dtype=float,
        )[:, None]
        response_rows = [
            evaluate_sin_synergy_ignition(
                ignition=ignition,
                total_cost=float(config.cost_grid.max()),
                model=model,
                t_force=config.t_force,
                dt=config.dt,
            )
            for ignition in ("none", "node_0", "node_1", "pair_0_1")
        ]
        response_at_max = add_ignition_comparison_columns(
            pd.DataFrame(response_rows).assign(**sin_model_parameter_row(config, float(alpha)), t_force=float(config.t_force))
        )
        pair_row = response_at_max.loc[response_at_max["ignition"].eq("pair_0_1")].iloc[0]
        ei_row = estimate_sin_pair_ei_for_sources(
            source_samples,
            target,
            config=config,
            seed=config.ei_seed + 10_000 + int(round(float(alpha) * 1000)),
        )
        rows.append(
            {
                "alpha": float(alpha),
                "sample_count": config.ei_sample_count,
                "delta_low": config.ei_cost_low,
                "delta_high": config.ei_cost_high,
                **ei_row,
                "best_single_response": float(pair_row["best_single_target_end"]),
                "pair_response": float(pair_row["target_end"]),
                "pair_gain_ratio": float(pair_row["pair_gain_ratio"]),
                "pair_surplus": float(pair_row["pair_surplus_target_end"]),
                "target_mean": float(target[:, 0].mean()),
                "target_std": float(target[:, 0].std(ddof=1)),
                "target_noise_fraction": config.ei_target_noise_fraction,
                **sin_model_parameter_row(config, float(alpha)),
            }
        )
    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(paths["alpha_sweep_csv"], index=False)
    return sweep_df


def load_or_run_sin_synergy_experiment(
    config: SinSynergyIgnitionConfig,
    *,
    force: bool = False,
) -> dict[str, pd.DataFrame | dict[str, Path]]:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    response = build_sin_synergy_ignition_table(config, force=force)
    surface = build_sin_synergy_pair_duration_surface(config, force=force)
    time_curve = estimate_sin_synergy_ei_time_curve(config, force=force)
    ei_decomposition, ei_samples = estimate_sin_synergy_ei_decomposition(config, force=force)
    alpha_sweep = build_sin_synergy_alpha_sweep(config, force=force)
    paths = sin_output_paths(config)
    paths["manifest_json"].write_text(json.dumps(sin_synergy_manifest(config, paths), indent=2), encoding="utf-8")
    return {
        "response": response,
        "pair_duration_surface": surface,
        "ei_time_curve": time_curve,
        "ei_decomposition": ei_decomposition,
        "ei_samples": ei_samples,
        "alpha_sweep": alpha_sweep,
        "paths": paths,
    }


def sin_synergy_manifest(
    config: SinSynergyIgnitionConfig,
    paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    if paths is None:
        paths = sin_output_paths(config)
    return {
        "experiment": "monotone_log1p_synergy_ode_ignition_response_and_ei_decomposition",
        "ode": "dx2/dt=-lambda_t*x2+alpha*log1p(max(g*x0*x1,0))+(1-alpha)*rho*(x0+epsilon*x1)",
        "nonlinearity": config.nonlinearity,
        "alpha": config.alpha,
        "alpha_low": config.alpha_low,
        "alpha_high": config.alpha_high,
        "alpha_grid": [float(value) for value in config.alpha_grid],
        "gain": config.gain,
        "source_decay": config.source_decay,
        "target_decay": config.target_decay,
        "source_weight": config.source_weight,
        "secondary_source_weight": config.secondary_source_weight,
        "t_force": config.t_force,
        "dt": config.dt,
        "cost_grid": [float(value) for value in config.cost_grid],
        "ei_sample_count": config.ei_sample_count,
        "ei_delta_interval": [config.ei_cost_low, config.ei_cost_high],
        "ei_target_noise_fraction": config.ei_target_noise_fraction,
        "duration_grid": [float(value) for value in config.duration_grid],
        "ei_time_grid": [float(value) for value in config.ei_time_grid],
        "response_csv": str(paths["ignition_csv"]),
        "ei_csv": str(paths["ei_csv"]),
        "ei_samples_csv": str(paths["ei_samples_csv"]),
        "pair_duration_surface_csv": str(paths["pair_duration_surface_csv"]),
        "ei_time_curve_csv": str(paths["ei_time_curve_csv"]),
        "alpha_sweep_csv": str(paths["alpha_sweep_csv"]),
    }


def summarize_sin_synergy_results(
    results: dict[str, pd.DataFrame | dict[str, Path]],
    config: SinSynergyIgnitionConfig,
) -> dict[str, Any]:
    response = results["response"]
    ei_decomposition = results["ei_decomposition"]
    alpha_sweep = results["alpha_sweep"]
    assert isinstance(response, pd.DataFrame)
    assert isinstance(ei_decomposition, pd.DataFrame)
    assert isinstance(alpha_sweep, pd.DataFrame)

    pair_surplus = (
        response.loc[response["ignition"].eq("pair_0_1"), ["total_cost", "pair_surplus_target_end", "pair_gain_ratio"]]
        .sort_values("pair_surplus_target_end", ascending=False)
        .iloc[0]
    )
    ei_row = ei_decomposition.iloc[0]
    low_alpha_row = alpha_sweep.loc[np.isclose(alpha_sweep["alpha"], config.alpha_low)].iloc[0]
    high_alpha_row = alpha_sweep.loc[np.isclose(alpha_sweep["alpha"], config.alpha_high)].iloc[0]
    return {
        "best_surplus_cost": float(pair_surplus["total_cost"]),
        "best_pair_surplus": float(pair_surplus["pair_surplus_target_end"]),
        "best_pair_gain_ratio": float(pair_surplus["pair_gain_ratio"]),
        "source_0_ei": float(ei_row["source_0_ei"]),
        "source_1_ei": float(ei_row["source_1_ei"]),
        "joint_ei": float(ei_row["joint_ei"]),
        "synergy": float(ei_row["synergy"]),
        "synergy_ratio": float(ei_row["synergy_ratio"]),
        "low_alpha_synergy_ratio": float(low_alpha_row["synergy_ratio"]),
        "low_alpha_pair_gain_ratio": float(low_alpha_row["pair_gain_ratio"]),
        "high_alpha_synergy_ratio": float(high_alpha_row["synergy_ratio"]),
        "high_alpha_pair_gain_ratio": float(high_alpha_row["pair_gain_ratio"]),
    }


def _save_figure(
    fig: plt.Figure,
    figure_dir: Path,
    name: str,
    *,
    dpi: int = 240,
) -> tuple[Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / f"{name}.png"
    pdf_path = figure_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def _multi_node_topology_positions(config: MultiNodeSynergyIgnitionConfig) -> dict[int, tuple[float, float]]:
    default_positions = {
        0: (0.00, 1.00),
        1: (0.95, 1.35),
        2: (1.90, 1.00),
        3: (0.75, 0.25),
        4: (1.90, 0.25),
        5: (2.95, 0.65),
    }
    nodes = tuple(config.source_nodes) + (config.target_node,)
    if all(int(node) in default_positions for node in nodes):
        return {int(node): default_positions[int(node)] for node in nodes}
    angles = np.linspace(0.0, 2.0 * np.pi, len(nodes), endpoint=False)
    return {
        int(node): (float(np.cos(angle)), float(np.sin(angle)))
        for node, angle in zip(nodes, angles, strict=True)
    }


def _multi_node_effective_topology_edges(config: MultiNodeSynergyIgnitionConfig) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = {
        (int(src), int(dst))
        for src, dst, _ in config.network_edges
    }
    target_node = int(config.target_node)
    for left, right, _ in config.embedded_pair_weights:
        for src in (int(left), int(right)):
            edges.add((src, target_node))
    return sorted(edges)


def plot_sin_synergy_figures(
    results: dict[str, pd.DataFrame | dict[str, Path]],
    config: SinSynergyIgnitionConfig,
    figure_dir: Path,
    *,
    dpi: int = 240,
) -> dict[str, tuple[Path, Path]]:
    response = results["response"]
    surface = results["pair_duration_surface"]
    ei_decomposition = results["ei_decomposition"]
    time_curve = results["ei_time_curve"]
    alpha_sweep = results["alpha_sweep"]
    assert isinstance(response, pd.DataFrame)
    assert isinstance(surface, pd.DataFrame)
    assert isinstance(ei_decomposition, pd.DataFrame)
    assert isinstance(time_curve, pd.DataFrame)
    assert isinstance(alpha_sweep, pd.DataFrame)

    figure_paths: dict[str, tuple[Path, Path]] = {}

    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    for ignition in ("none", "node_0", "node_1", "pair_0_1"):
        group = response.loc[response["ignition"].eq(ignition)].sort_values("total_cost")
        ax.plot(
            group["total_cost"],
            group["target_end"],
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=IGNITION_COLORS[ignition],
            label=IGNITION_LABELS[ignition],
        )
    ax.set_xlabel("Total ignition cost")
    ax.set_ylabel(r"Target response $x_2(T)$")
    ax.tick_params(direction="in")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure_paths["sin_synergy_ignition_target_response"] = _save_figure(fig, figure_dir, "sin_synergy_ignition_target_response", dpi=dpi)
    plt.close(fig)

    surface_matrix = surface.pivot_table(index="duration", columns="total_cost", values="target_value", aggfunc="mean").sort_index().sort_index(axis=1)
    surface_costs = surface_matrix.columns.to_numpy(dtype=float)
    surface_durations = surface_matrix.index.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    image = ax.imshow(
        surface_matrix.to_numpy(dtype=float),
        aspect="auto",
        origin="lower",
        extent=[
            float(surface_costs.min()),
            float(surface_costs.max()),
            float(surface_durations.min()),
            float(surface_durations.max()),
        ],
        cmap="viridis",
    )
    ax.set_xlabel("Total ignition cost")
    ax.set_ylabel("Ignition duration")
    ax.tick_params(direction="in")
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label(r"Target value $x_2(T)$")
    figure_paths["sin_synergy_pair_duration_surface"] = _save_figure(fig, figure_dir, "sin_synergy_pair_duration_surface", dpi=dpi)
    plt.close(fig)

    surplus_frame = (
        response.loc[response["ignition"].eq("pair_0_1"), ["total_cost", "pair_surplus_target_end"]]
        .sort_values("total_cost")
    )
    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    ax.plot(
        surplus_frame["total_cost"],
        surplus_frame["pair_surplus_target_end"],
        marker="o",
        markersize=3.5,
        linewidth=1.6,
        color="#7c2d12",
        label=r"$x_2^{01}-x_2^0-x_2^1+x_2^\varnothing$",
    )
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xlabel("Total ignition cost")
    ax.set_ylabel("Pair surplus response")
    ax.tick_params(direction="in")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure_paths["sin_synergy_ignition_pair_surplus"] = _save_figure(fig, figure_dir, "sin_synergy_ignition_pair_surplus", dpi=dpi)
    plt.close(fig)

    ei_row = ei_decomposition.iloc[0]
    ei_components = pd.DataFrame(
        {
            "component": [r"$EI_0$", r"$EI_1$", r"$EI_{01}$", r"$Syn_{01}$"],
            "value": [ei_row["source_0_ei"], ei_row["source_1_ei"], ei_row["joint_ei"], ei_row["synergy"]],
        }
    )
    fig, ax = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
    ax.bar(
        ei_components["component"],
        ei_components["value"],
        color=["#2563eb", "#dc2626", "#16a34a", "#7c2d12"],
    )
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_ylabel("nats")
    ax.tick_params(direction="in", axis="x", rotation=20)
    figure_paths["sin_synergy_ei_decomposition"] = _save_figure(fig, figure_dir, "sin_synergy_ei_decomposition", dpi=dpi)
    plt.close(fig)

    time_specs = [
        ("source_0_ei", r"$EI_0$", "#2563eb"),
        ("source_1_ei", r"$EI_1$", "#dc2626"),
        ("joint_ei", r"$EI_{01}$", "#16a34a"),
        ("synergy", r"$Syn_{01}$", "#7c2d12"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    for column, label, color in time_specs:
        ax.plot(
            time_curve["duration"],
            time_curve[column],
            marker="o",
            markersize=4.0,
            linewidth=1.6,
            color=color,
            label=label,
        )
    ax.set_xlabel("Evolution time")
    ax.set_ylabel("nats")
    ax.tick_params(direction="in")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure_paths["sin_synergy_ei_time_curve"] = _save_figure(fig, figure_dir, "sin_synergy_ei_time_curve", dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), constrained_layout=True)
    axes[0].plot(
        alpha_sweep["alpha"],
        alpha_sweep["synergy_ratio"],
        marker="o",
        linewidth=1.6,
        color="#7c2d12",
        label=r"$Syn_{01}/EI_{01}$",
    )
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel("Synergy ratio")
    axes[0].tick_params(direction="in")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    axes[1].plot(
        alpha_sweep["alpha"],
        alpha_sweep["pair_gain_ratio"],
        marker="s",
        linewidth=1.6,
        color="#16a34a",
        label="Pair / best single",
    )
    axes[1].axhline(1.0, color="0.25", linewidth=0.8)
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel("Ignition gain ratio")
    axes[1].tick_params(direction="in")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure_paths["sin_synergy_alpha_sweep_summary"] = _save_figure(fig, figure_dir, "sin_synergy_alpha_sweep_summary", dpi=dpi)
    plt.close(fig)

    return figure_paths


def plot_multi_node_synergy_figures(
    results: dict[str, Any],
    config: MultiNodeSynergyIgnitionConfig,
    figure_dir: Path,
    *,
    dpi: int = 240,
) -> dict[str, tuple[Path, Path]]:
    summary = results["pair_summary"]
    ignition = results["pair_ignition_response"]
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(ignition, pd.DataFrame)

    figure_paths: dict[str, tuple[Path, Path]] = {}
    source_nodes = list(config.source_nodes)
    node_to_pos = {node: pos for pos, node in enumerate(source_nodes)}

    synergy_matrix = np.full((len(source_nodes), len(source_nodes)), np.nan, dtype=float)
    ratio_matrix = np.full_like(synergy_matrix, np.nan)
    for row in summary.itertuples(index=False):
        i = node_to_pos[int(row.pair_i)]
        j = node_to_pos[int(row.pair_j)]
        synergy_matrix[i, j] = synergy_matrix[j, i] = float(row.synergy)
        ratio_matrix[i, j] = ratio_matrix[j, i] = float(row.synergy_ratio)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)
    for ax, matrix, label in [
        (axes[0], synergy_matrix, "Syn"),
        (axes[1], ratio_matrix, "Syn ratio"),
    ]:
        image = ax.imshow(matrix, cmap="viridis")
        ax.set_xticks(range(len(source_nodes)), labels=source_nodes)
        ax.set_yticks(range(len(source_nodes)), labels=source_nodes)
        ax.set_xlabel("Source node")
        ax.set_ylabel("Source node")
        ax.tick_params(direction="in")
        cbar = fig.colorbar(image, ax=ax, pad=0.02)
        cbar.set_label(label)
    figure_paths["multi_node_pair_synergy_heatmap"] = _save_figure(fig, figure_dir, "multi_node_pair_synergy_heatmap", dpi=dpi)
    plt.close(fig)

    positions = _multi_node_topology_positions(config)
    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    feedback_edges = {tuple(edge) for edge in config.feedback_loop_edges}
    for src, dst in _multi_node_effective_topology_edges(config):
        start = positions[int(src)]
        end = positions[int(dst)]
        rad = 0.12 if (int(src), int(dst)) in feedback_edges else 0.02
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.4,
            color="0.35",
            alpha=0.82,
            shrinkA=18,
            shrinkB=18,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(arrow)
    for node, (x, y) in positions.items():
        is_target = int(node) == int(config.target_node)
        circle = Circle(
            (x, y),
            radius=0.135,
            facecolor="#e5e7eb" if not is_target else "#dbeafe",
            edgecolor="0.15",
            linewidth=1.2,
            zorder=3,
        )
        ax.add_patch(circle)
        ax.text(x, y, str(node), ha="center", va="center", fontsize=10, zorder=4)
    xs = [xy[0] for xy in positions.values()]
    ys = [xy[1] for xy in positions.values()]
    ax.set_xlim(min(xs) - 0.35, max(xs) + 0.35)
    ax.set_ylim(min(ys) - 0.35, max(ys) + 0.35)
    ax.set_aspect("equal")
    ax.axis("off")
    figure_paths["multi_node_network_topology"] = _save_figure(fig, figure_dir, "multi_node_network_topology", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.scatter(
        summary["synergy"],
        summary["pair_response_at_max_cost"],
        s=54,
        color="#2563eb",
        edgecolor="white",
        linewidth=0.6,
    )
    annotation_offsets = {
        (0, 1): (-28, 4),
        (2, 4): (4, 12),
        (3, 4): (4, -4),
    }
    for row in summary.itertuples(index=False):
        pair = (int(row.pair_i), int(row.pair_j))
        offset = annotation_offsets.get(pair, (4, 4))
        ax.annotate(
            f"({pair[0]},{pair[1]})",
            (float(row.synergy), float(row.pair_response_at_max_cost)),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            ha="right" if offset[0] < 0 else "left",
        )
    ax.set_xlabel("Pair Syn")
    ax.set_ylabel(r"Pair ignition response $x_5(T)$ at max cost")
    ax.tick_params(direction="in")
    spearman = float(summary["spearman_synergy_pair_response"].iloc[0])
    ax.text(0.02, 0.96, rf"Spearman $\rho={spearman:.2f}$", transform=ax.transAxes, va="top")
    figure_paths["multi_node_synergy_vs_pair_response"] = _save_figure(fig, figure_dir, "multi_node_synergy_vs_pair_response", dpi=dpi)
    plt.close(fig)

    selected_pairs: list[tuple[int, int, str]] = []
    top_row = summary.sort_values("synergy", ascending=False).iloc[0]
    selected_pairs.append((int(top_row["pair_i"]), int(top_row["pair_j"]), "Top Syn"))
    weak = summary.loc[summary["pair_i"].eq(1) & summary["pair_j"].eq(4)]
    if not weak.empty:
        weak_row = weak.iloc[0]
        selected_pairs.append((int(weak_row["pair_i"]), int(weak_row["pair_j"]), "Reference pair"))
    low_row = summary.sort_values("synergy", ascending=True).iloc[0]
    low_pair = (int(low_row["pair_i"]), int(low_row["pair_j"]))
    if low_pair not in [(left, right) for left, right, _ in selected_pairs]:
        selected_pairs.append((low_pair[0], low_pair[1], "Low Syn"))

    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    colors = ["#16a34a", "#7c2d12", "0.45"]
    for (left, right, label), color in zip(selected_pairs, colors, strict=False):
        group = ignition.loc[
            ignition["pair_i"].eq(left) & ignition["pair_j"].eq(right) & ignition["ignition"].eq("pair")
        ].sort_values("total_cost")
        ax.plot(
            group["total_cost"],
            group["target_end"],
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=color,
            label=f"{label} ({left},{right})",
        )
    ax.set_xlabel("Total ignition cost")
    ax.set_ylabel(r"Target response $x_5(T)$")
    ax.tick_params(direction="in")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure_paths["multi_node_pair_response_curves"] = _save_figure(fig, figure_dir, "multi_node_pair_response_curves", dpi=dpi)
    plt.close(fig)

    ranked = summary.sort_values("synergy", ascending=True).copy()
    labels = [f"({int(row.pair_i)},{int(row.pair_j)})" for row in ranked.itertuples(index=False)]
    y = np.arange(len(ranked))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), sharey=True, constrained_layout=True)
    bar_color = "#2563eb"
    axes[0].barh(y, ranked["synergy"], color=bar_color)
    axes[0].set_yticks(y, labels=labels)
    axes[0].set_xlabel("Pair Syn")
    axes[0].tick_params(direction="in")
    axes[1].barh(y, ranked["pair_response_at_max_cost"], color=bar_color)
    axes[1].set_xlabel(r"Pair response $x_5(T)$")
    axes[1].tick_params(direction="in")
    figure_paths["multi_node_pair_summary_rankings"] = _save_figure(fig, figure_dir, "multi_node_pair_summary_rankings", dpi=dpi)
    plt.close(fig)

    return figure_paths


def write_sin_synergy_figure_manifest(
    config: SinSynergyIgnitionConfig,
    figure_paths: dict[str, tuple[Path, Path]],
    *,
    manifest_path: Path,
    notes_path: Path,
) -> None:
    paths = sin_output_paths(config)
    figure_manifest = {
        "figures": {
            name: {"png": str(paths_[0]), "pdf": str(paths_[1])}
            for name, paths_ in sorted(figure_paths.items())
        },
        "sin_ignition_csv": str(paths["ignition_csv"]),
        "sin_ei_csv": str(paths["ei_csv"]),
        "sin_pair_duration_surface_csv": str(paths["pair_duration_surface_csv"]),
        "sin_ei_time_curve_csv": str(paths["ei_time_curve_csv"]),
        "sin_alpha_sweep_csv": str(paths["alpha_sweep_csv"]),
    }
    manifest_path.write_text(json.dumps(figure_manifest, indent=2), encoding="utf-8")
    notes = """# Figure notes

- `sin_synergy_ignition_target_response`: endpoint response of target node 2 under no ignition, node 0 ignition, node 1 ignition, and equal-cost pair ignition.
- `sin_synergy_pair_duration_surface`: target value under pair ignition across total ignition cost and ignition duration.
- `sin_synergy_ignition_pair_surplus`: pair response surplus over the two singleton responses.
- `sin_synergy_ei_decomposition`: singleton EI, joint EI, and signed pair synergy under the random nonnegative two-source intervention distribution.
- `sin_synergy_ei_time_curve`: EI components as the target evolution time changes.
- `sin_synergy_alpha_sweep_summary`: alpha sweep linking the synergy ratio and the pair ignition gain ratio.
- `multi_node_pair_synergy_heatmap`: pair Syn and SynRatio over all ten source-node pairs in the controlled six-node experiment.
- `multi_node_network_topology`: directed effective topology used by the controlled six-node experiment, including both direct network edges and pair-readout source-to-target effects.
- `multi_node_synergy_vs_pair_response`: relationship between pair Syn and direct pair ignition response at the maximum total cost.
- `multi_node_pair_response_curves`: pair ignition response curves for representative high-, weak-, and low-Syn pairs.
- `multi_node_pair_summary_rankings`: pair Syn ranking compared with direct pair ignition response.
"""
    notes_path.write_text(notes, encoding="utf-8")
