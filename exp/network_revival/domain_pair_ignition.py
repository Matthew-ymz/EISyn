"""Domain-specific pair ignition controls for bistable target dynamics.

This module repeats the Part 3 joint-required ignition experiment on three
canonical one-dimensional bistable dynamics. Candidate source nodes are treated
as controlled input channels into the same target module, so source-source
network topology is deliberately outside the success criterion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from itertools import combinations
from pathlib import Path
import shutil
from typing import Any, Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from exp.network_revival.joint_required_ignition import (
    _average_precision,
    _conditional_mi,
    _matrix_from_pairs,
    _mutual_information,
    _save_figure,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "network_revival_domain_pair_ignition"

__all__ = [
    "DomainBistableModel",
    "DomainPairIgnitionConfig",
    "iter_domain_models",
    "get_domain_model",
    "build_domain_pair_instance",
    "simulate_domain_pair_intervention",
    "compute_domain_pair_basin_peid",
    "run_domain_pair_ensemble",
    "plot_domain_pair_results",
]


ArrayFn = Callable[[np.ndarray | float], np.ndarray]


@dataclass(frozen=True)
class DomainBistableModel:
    key: str
    display_name: str
    equation_label: str
    domain: str
    drift: ArrayFn
    drift_derivative: ArrayFn
    low_state: float
    basin_threshold: float
    high_state: float
    saddle_state: float
    critical_input_value: float
    physical_meaning: str
    trajectory_t_force: float = 20.0
    trajectory_release_time: float = 20.0
    state_plot_min: float = -0.05
    state_plot_max: float = 1.4

    def rhs(self, state: np.ndarray | float, input_value: np.ndarray | float) -> np.ndarray:
        return self.drift(state) + np.asarray(input_value, dtype=float)

    def critical_input(self) -> float:
        return float(self.critical_input_value)


@dataclass(frozen=True)
class DomainPairIgnitionConfig:
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    source_count: int = 8
    hill_coefficient: float = 4.0
    half_saturation: float = 1.0
    amplitude_max: float = 4.0
    amplitude_levels: int = 16
    t_force: float = 8.0
    release_time: float = 8.0
    dt: float = 0.04
    ensemble_size: int = 50
    sample_sizes: tuple[int, ...] = (128, 256, 512)
    label_noise_levels: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10)
    high_weight_fraction_range: tuple[float, float] = (0.62, 0.78)
    low_weight_fraction_range: tuple[float, float] = (0.10, 0.14)
    pair_margin: float = 0.06
    fixed_delta: float = 1.5
    seed: int = 20260616

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.source_count < 4:
            raise ValueError("source_count must be at least four.")
        if self.hill_coefficient <= 0.0 or self.half_saturation <= 0.0:
            raise ValueError("Hill parameters must be positive.")
        if self.amplitude_max <= 0.0 or self.amplitude_levels < 3:
            raise ValueError("amplitude grid must contain at least three nonnegative states.")
        if self.t_force <= 0.0 or self.release_time < 0.0 or self.dt <= 0.0:
            raise ValueError("time parameters must be positive except release_time may be zero.")
        if self.ensemble_size < 1:
            raise ValueError("ensemble_size must be positive.")
        if not self.sample_sizes or min(self.sample_sizes) < 2:
            raise ValueError("sample_sizes must contain values of at least two.")
        if not self.label_noise_levels or min(self.label_noise_levels) < 0.0 or max(self.label_noise_levels) > 0.5:
            raise ValueError("label_noise_levels must lie between zero and 0.5.")
        high_low, high_high = self.high_weight_fraction_range
        low_low, low_high = self.low_weight_fraction_range
        if not 0.0 < low_low <= low_high < high_low <= high_high < 1.0:
            raise ValueError("weight fraction ranges must be ordered within (0, 1).")
        if 2.0 * high_low <= 1.0 + self.pair_margin:
            raise ValueError("high-weight pairs must clear the analytic threshold margin.")
        if high_high + low_high >= 1.0 - self.pair_margin:
            raise ValueError("mixed-weight pairs must remain below the analytic threshold margin.")


def _asarray(value: np.ndarray | float) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _sigmoid(value: np.ndarray | float, *, gain: float, threshold: float) -> np.ndarray:
    arg = np.clip(-gain * (_asarray(value) - threshold), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(arg))


def _scan_roots(function: Callable[[np.ndarray | float], np.ndarray], lower: float, upper: float, *, points: int = 4000) -> list[float]:
    grid = np.linspace(float(lower), float(upper), int(points))
    values = np.asarray(function(grid), dtype=float)
    roots: list[float] = []
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:], strict=True):
        if not np.isfinite(f_left) or not np.isfinite(f_right):
            continue
        if abs(f_left) < 1e-10:
            roots.append(float(left))
        elif f_left * f_right < 0.0:
            roots.append(_bisect_root(function, float(left), float(right)))
    if abs(values[-1]) < 1e-10:
        roots.append(float(grid[-1]))
    unique: list[float] = []
    for root in roots:
        if not unique or abs(root - unique[-1]) > 1e-5:
            unique.append(root)
    return unique


def _bisect_root(function: Callable[[np.ndarray | float], np.ndarray], lower: float, upper: float) -> float:
    lo = float(lower)
    hi = float(upper)
    f_lo = float(function(lo))
    f_hi = float(function(hi))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = float(function(mid))
        if abs(f_mid) < 1e-14 or abs(hi - lo) < 1e-14:
            return mid
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def _wilson_cowan_model() -> DomainBistableModel:
    beta = 3.2
    gain = 4.0
    threshold = 1.0
    def drift(x: np.ndarray | float) -> np.ndarray:
        state = _asarray(x)
        return -state + _sigmoid(beta * state, gain=gain, threshold=threshold)

    def derivative(x: np.ndarray | float) -> np.ndarray:
        state = _asarray(x)
        response = _sigmoid(beta * state, gain=gain, threshold=threshold)
        return -1.0 + beta * gain * response * (1.0 - response)

    roots = _scan_roots(drift, 0.0, 1.2)
    extrema = _scan_roots(derivative, 0.0, 1.2)
    saddle = [root for root in extrema if roots[0] < root < roots[1]][0]
    return DomainBistableModel(
        key="wilson_cowan",
        display_name="Wilson-Cowan",
        equation_label=r"$\dot x=-x+S(\beta x)+u$",
        domain="neural population",
        drift=drift,
        drift_derivative=derivative,
        low_state=roots[0],
        basin_threshold=roots[1],
        high_state=roots[2],
        saddle_state=float(saddle),
        critical_input_value=float(-drift(saddle)),
        physical_meaning="minimum external current that removes the low-firing attractor",
        state_plot_min=-0.02,
        state_plot_max=1.08,
    )


def _allee_model() -> DomainBistableModel:
    r = 1.0
    allee = 0.35
    capacity = 1.0
    coeffs = np.poly1d([-r / (allee * capacity), r * (capacity + allee) / (allee * capacity), -r, 0.0])
    derivative_coeffs = np.polyder(coeffs)
    critical_candidates = [root.real for root in np.roots(derivative_coeffs) if abs(root.imag) < 1e-10 and 0.0 < root.real < allee]
    saddle = float(critical_candidates[0])

    def drift(x: np.ndarray | float) -> np.ndarray:
        state = _asarray(x)
        return r * state * (1.0 - state / capacity) * (state / allee - 1.0)

    def derivative(x: np.ndarray | float) -> np.ndarray:
        return np.polyval(derivative_coeffs, _asarray(x))

    return DomainBistableModel(
        key="allee",
        display_name="Allee effect",
        equation_label=r"$\dot x=rx(1-x/K)(x/A-1)+u$",
        domain="ecological recovery",
        drift=drift,
        drift_derivative=derivative,
        low_state=0.0,
        basin_threshold=allee,
        high_state=capacity,
        saddle_state=saddle,
        critical_input_value=float(-drift(saddle)),
        physical_meaning="minimum immigration or restocking rate that overcomes the strong Allee threshold",
        state_plot_min=-0.02,
        state_plot_max=1.08,
    )


def _schlogl_model() -> DomainBistableModel:
    time_scale = 10.0
    k0 = 0.08
    k1 = 3.0
    k2 = 1.0
    k3 = 1.0
    drift_coeffs = time_scale * np.poly1d([-k2, k1, -k3, k0])
    derivative_coeffs = np.polyder(drift_coeffs)
    roots = sorted(root.real for root in np.roots(drift_coeffs) if abs(root.imag) < 1e-10 and root.real >= 0.0)
    critical_candidates = [
        root.real
        for root in np.roots(derivative_coeffs)
        if abs(root.imag) < 1e-10 and roots[0] < root.real < roots[1]
    ]
    saddle = float(critical_candidates[0])

    def drift(x: np.ndarray | float) -> np.ndarray:
        return np.polyval(drift_coeffs, _asarray(x))

    def derivative(x: np.ndarray | float) -> np.ndarray:
        return np.polyval(derivative_coeffs, _asarray(x))

    return DomainBistableModel(
        key="schlogl",
        display_name="Schlögl",
        equation_label=r"$\dot x=k_0+k_1x^2-k_2x^3-k_3x+u$",
        domain="autocatalytic chemistry",
        drift=drift,
        drift_derivative=derivative,
        low_state=float(roots[0]),
        basin_threshold=float(roots[1]),
        high_state=float(roots[2]),
        saddle_state=saddle,
        critical_input_value=float(-drift(saddle)),
        physical_meaning="minimum feed flux that removes the low-concentration attractor",
        state_plot_min=0.0,
        state_plot_max=3.0,
    )


def iter_domain_models() -> tuple[DomainBistableModel, ...]:
    return (_wilson_cowan_model(), _allee_model(), _schlogl_model())


def get_domain_model(key: str) -> DomainBistableModel:
    for model in iter_domain_models():
        if model.key == key:
            return model
    raise ValueError(f"Unknown domain model {key!r}.")


def _source_pairs(source_count: int) -> list[tuple[int, int]]:
    return list(combinations(range(int(source_count)), 2))


def _hill(delta: np.ndarray | float, config: DomainPairIgnitionConfig) -> np.ndarray:
    values = np.maximum(_asarray(delta), 0.0)
    exponent = float(config.hill_coefficient)
    half = float(config.half_saturation)
    return values**exponent / (half**exponent + values**exponent + 1e-15)


def build_domain_pair_instance(
    model: DomainBistableModel,
    config: DomainPairIgnitionConfig,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    critical = model.critical_input()
    high_count = max(2, config.source_count // 2)
    high = rng.uniform(*config.high_weight_fraction_range, size=high_count)
    low = rng.uniform(*config.low_weight_fraction_range, size=config.source_count - high_count)
    fractions = np.concatenate([high, low])
    rng.shuffle(fractions)
    weights = critical * fractions

    pairs = _source_pairs(config.source_count)
    pair_ratios = np.asarray([(weights[i] + weights[j]) / critical for i, j in pairs], dtype=float)
    switchable = [pair for pair, ratio in zip(pairs, pair_ratios, strict=True) if ratio > 1.0]
    nonswitchable = [pair for pair, ratio in zip(pairs, pair_ratios, strict=True) if ratio < 1.0]
    if not switchable or not nonswitchable:
        raise RuntimeError("generated domain instance must contain both pair classes.")
    if np.any(np.abs(pair_ratios - 1.0) < config.pair_margin):
        raise RuntimeError("generated domain instance violates the pair threshold margin.")

    return {
        "model_key": model.key,
        "seed": int(seed),
        "critical_input": float(critical),
        "weights": weights,
        "weight_fractions": fractions,
        "pairs": pairs,
        "pair_input_ratios": pair_ratios,
        "switchable_pairs": switchable,
        "nonswitchable_pairs": nonswitchable,
    }


def _integrate_domain_input(
    model: DomainBistableModel,
    initial: np.ndarray | float,
    input_value: np.ndarray | float,
    *,
    duration: float,
    dt: float,
    record: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    state = _asarray(initial).copy()
    forcing = _asarray(input_value)
    times = [0.0] if record else None
    states = [float(state)] if record and state.ndim == 0 else None
    t = 0.0
    while t < float(duration) - 1e-12:
        step = min(float(dt), float(duration) - t)
        k1 = model.rhs(state, forcing)
        k2 = model.rhs(state + 0.5 * step * k1, forcing)
        k3 = model.rhs(state + 0.5 * step * k2, forcing)
        k4 = model.rhs(state + step * k3, forcing)
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += step
        if record and times is not None and states is not None:
            times.append(float(t))
            states.append(float(state))
    if record and times is not None and states is not None:
        return state, np.asarray(times, dtype=float), np.asarray(states, dtype=float)
    return state, None, None


def simulate_domain_pair_intervention(
    model: DomainBistableModel,
    instance: dict[str, Any],
    *,
    pair: tuple[int, int],
    delta_i: float,
    delta_j: float,
    t_force: float,
    release_time: float,
    dt: float,
) -> dict[str, Any]:
    left, right = sorted((int(pair[0]), int(pair[1])))
    weights = np.asarray(instance["weights"], dtype=float)
    pseudo_config = DomainPairIgnitionConfig(dt=dt)
    input_value = float(
        weights[left] * _hill(delta_i, pseudo_config)
        + weights[right] * _hill(delta_j, pseudo_config)
    )
    forced, force_times, force_states = _integrate_domain_input(
        model,
        model.low_state,
        input_value,
        duration=t_force,
        dt=dt,
        record=True,
    )
    released, release_times, release_states = _integrate_domain_input(
        model,
        forced,
        0.0,
        duration=release_time,
        dt=dt,
        record=True,
    )
    times = np.concatenate([force_times, float(t_force) + release_times[1:]])
    trajectory = np.concatenate([force_states, release_states[1:]])
    return {
        "pair": (left, right),
        "input_value": input_value,
        "force_end": float(forced),
        "final_state": float(released),
        "basin_label": int(float(released) > float(model.basin_threshold)),
        "times": times,
        "trajectory": trajectory,
    }


def _pair_response_grid(
    model: DomainBistableModel,
    instance: dict[str, Any],
    config: DomainPairIgnitionConfig,
    pair: tuple[int, int],
) -> dict[str, np.ndarray]:
    amplitudes = np.linspace(0.0, config.amplitude_max, config.amplitude_levels)
    delta_i, delta_j = np.meshgrid(amplitudes, amplitudes, indexing="ij")
    left, right = pair
    weights = np.asarray(instance["weights"], dtype=float)
    forcing = weights[left] * _hill(delta_i.ravel(), config) + weights[right] * _hill(delta_j.ravel(), config)
    labels = (forcing > float(instance["critical_input"])).astype(int)
    return {
        "amplitudes": amplitudes,
        "source_i": np.repeat(np.arange(config.amplitude_levels), config.amplitude_levels),
        "source_j": np.tile(np.arange(config.amplitude_levels), config.amplitude_levels),
        "delta_i": delta_i.ravel(),
        "delta_j": delta_j.ravel(),
        "input_value": np.asarray(forcing, dtype=float),
        "labels": labels,
    }


def _peid_row(source_i: np.ndarray, source_j: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    left_ei = _mutual_information(source_i, labels)
    right_ei = _mutual_information(source_j, labels)
    joint_ei = _mutual_information(np.column_stack([source_i, source_j]), labels)
    synergy = float(joint_ei - left_ei - right_ei)
    conditional_mi = _conditional_mi(source_i, source_j, labels)
    return {
        "left_ei": left_ei,
        "right_ei": right_ei,
        "single_ei_sum": float(left_ei + right_ei),
        "joint_ei": joint_ei,
        "synergy": synergy,
        "conditional_mi": conditional_mi,
        "synergy_ratio": float(synergy / joint_ei) if joint_ei > 1e-12 else 0.0,
    }


def _build_pair_response_cache(
    model: DomainBistableModel,
    instance: dict[str, Any],
    config: DomainPairIgnitionConfig,
) -> dict[tuple[int, int], dict[str, np.ndarray | float]]:
    cache: dict[tuple[int, int], dict[str, np.ndarray | float]] = {}
    for left, right in _source_pairs(config.source_count):
        response = _pair_response_grid(model, instance, config, (left, right))
        successful = response["labels"].astype(bool)
        costs = response["delta_i"] + response["delta_j"]
        fixed_input = (
            np.asarray(instance["weights"], dtype=float)[left] * _hill(config.fixed_delta, config)
            + np.asarray(instance["weights"], dtype=float)[right] * _hill(config.fixed_delta, config)
        )
        response["minimum_joint_cost"] = float(np.min(costs[successful])) if np.any(successful) else np.nan
        response["fixed_cost_response"] = float(fixed_input)
        response["grid_switch_rate"] = float(np.mean(successful))
        cache[(left, right)] = response
    return cache


def _single_node_total_strength_labels(
    model: DomainBistableModel,
    instance: dict[str, Any],
    config: DomainPairIgnitionConfig,
) -> np.ndarray:
    """Label singleton ignition at the same total drive as a max-strength pair."""

    weights = np.asarray(instance["weights"], dtype=float)
    total_drive = 2.0 * _hill(config.amplitude_max, config)
    forcing = weights * total_drive
    return (forcing > float(instance["critical_input"])).astype(int)


def _truth_matrix_with_single_node_diagonal(
    model: DomainBistableModel,
    frame: pd.DataFrame,
    config: DomainPairIgnitionConfig,
) -> np.ndarray:
    matrix = _matrix_from_pairs(frame, "max_pair_basin_label", config.source_count)
    if frame.empty:
        return matrix
    instance_seed = int(frame["instance_seed"].iloc[0])
    instance = build_domain_pair_instance(model, config, instance_seed)
    np.fill_diagonal(matrix, _single_node_total_strength_labels(model, instance, config))
    return matrix


def _representative_instance_for_model(
    model: DomainBistableModel,
    frame: pd.DataFrame,
    config: DomainPairIgnitionConfig,
) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("representative frame must contain at least one row.")
    instance_seed = int(frame["instance_seed"].iloc[0])
    return build_domain_pair_instance(model, config, instance_seed)


def _score_pair_response_cache(
    model: DomainBistableModel,
    instance: dict[str, Any],
    config: DomainPairIgnitionConfig,
    response_cache: dict[tuple[int, int], dict[str, np.ndarray | float]],
    *,
    sample_count: int | None,
    label_noise: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    weights = np.asarray(instance["weights"], dtype=float)
    critical = float(instance["critical_input"])
    rows: list[dict[str, Any]] = []
    for (left, right), response in response_cache.items():
        response_labels = np.asarray(response["labels"], dtype=int)
        if sample_count is None:
            indices = np.arange(len(response_labels))
        else:
            indices = rng.integers(0, len(response_labels), size=int(sample_count))
        labels = response_labels[indices].copy()
        if label_noise > 0.0:
            flips = rng.random(len(labels)) < float(label_noise)
            labels[flips] = 1 - labels[flips]
        values = _peid_row(
            np.asarray(response["source_i"])[indices],
            np.asarray(response["source_j"])[indices],
            labels,
        )
        rows.append(
            {
                "model_key": model.key,
                "model_name": model.display_name,
                "pair_i": left,
                "pair_j": right,
                "analytic_switchable": bool(weights[left] + weights[right] > critical),
                "max_pair_basin_label": int(response_labels[-1]),
                "pair_input_ratio": float((weights[left] + weights[right]) / critical),
                "grid_switch_rate": float(response["grid_switch_rate"]),
                "minimum_joint_cost": float(response["minimum_joint_cost"]),
                "fixed_cost_response": float(response["fixed_cost_response"]),
                "random_score": float(rng.random()),
                "sample_count": int(len(indices)),
                "label_noise": float(label_noise),
                **values,
            }
        )
    frame = pd.DataFrame(rows).sort_values(["synergy", "pair_i", "pair_j"], ascending=[False, True, True]).reset_index(drop=True)
    frame["rank_synergy"] = np.arange(1, len(frame) + 1)
    return frame


def compute_domain_pair_basin_peid(
    model: DomainBistableModel,
    instance: dict[str, Any],
    config: DomainPairIgnitionConfig,
    *,
    sample_count: int | None = None,
    label_noise: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame:
    cache = _build_pair_response_cache(model, instance, config)
    return _score_pair_response_cache(
        model,
        instance,
        config,
        cache,
        sample_count=sample_count,
        label_noise=label_noise,
        seed=config.seed if seed is None else int(seed),
    )


def _screening_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    labels = frame["analytic_switchable"].to_numpy(dtype=bool)
    positive_count = int(np.sum(labels))
    rows: list[dict[str, Any]] = []
    for score in ("synergy", "joint_ei", "single_ei_sum", "fixed_cost_response", "grid_switch_rate", "random_score"):
        values = frame[score].to_numpy(dtype=float)
        order = np.argsort(-values, kind="mergesort")
        top_k = labels[order[:positive_count]]
        if frame[score].nunique() >= 2 and frame["grid_switch_rate"].nunique() >= 2:
            success_spearman = float(frame[score].corr(frame["grid_switch_rate"], method="spearman"))
        else:
            success_spearman = np.nan
        rows.append(
            {
                "score": score,
                "auprc": _average_precision(values, labels),
                "top_k_recall": float(np.mean(top_k)) if positive_count else np.nan,
                "top1_hit": float(labels[order[0]]) if len(order) else np.nan,
                "success_rate_spearman": success_spearman,
            }
        )
    return rows


def _cache_paths(config: DomainPairIgnitionConfig) -> dict[str, Path]:
    return {
        "summary_json": config.output_dir / "summary.json",
        "pairs_jsonl": config.output_dir / "pair_scores.jsonl",
        "arrays_npz": config.output_dir / "representative_arrays.npz",
        "manifest_json": config.output_dir / "manifest.json",
    }


def _jsonable_config(config: DomainPairIgnitionConfig) -> dict[str, Any]:
    values = asdict(config)
    values["output_dir"] = str(config.output_dir)
    for key in ("sample_sizes", "label_noise_levels", "high_weight_fraction_range", "low_weight_fraction_range"):
        values[key] = list(values[key])
    return values


def _read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line.strip()])


def run_domain_pair_ensemble(
    config: DomainPairIgnitionConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    paths = _cache_paths(config)
    if not force and all(path.exists() for path in paths.values()):
        summary = json.loads(paths["summary_json"].read_text())
        pair_rows = _read_jsonl(paths["pairs_jsonl"])
        metrics = pd.DataFrame(summary["metrics"])
        representative = pair_rows.loc[pair_rows["condition"].eq("exact_grid") & pair_rows["instance"].eq(0)].copy()
        return {
            "summary": summary,
            "pair_rows": pair_rows,
            "metrics": metrics,
            "representative_pairs": representative,
            "cache_paths": paths,
        }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    all_pair_rows: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    representative_by_model: dict[str, pd.DataFrame] = {}
    summary_models: dict[str, dict[str, Any]] = {}

    for model_index, model in enumerate(iter_domain_models()):
        for instance_index in range(config.ensemble_size):
            instance_seed = config.seed + 1000 * model_index + instance_index
            instance = build_domain_pair_instance(model, config, instance_seed)
            response_cache = _build_pair_response_cache(model, instance, config)
            exact = _score_pair_response_cache(
                model,
                instance,
                config,
                response_cache,
                sample_count=None,
                label_noise=0.0,
                seed=instance_seed + 10000,
            )
            exact_meta = {
                "model_key": model.key,
                "instance": instance_index,
                "instance_seed": instance_seed,
                "condition": "exact_grid",
                "requested_sample_count": config.amplitude_levels**2,
                "label_noise": 0.0,
            }
            exact_with_meta = exact.assign(**exact_meta)
            if instance_index == 0:
                representative_by_model[model.key] = exact_with_meta.copy()
            all_pair_rows.extend(exact_with_meta.to_dict("records"))
            all_metrics.extend({**exact_meta, **row} for row in _screening_metrics(exact))

            for sample_count in config.sample_sizes:
                for noise_index, label_noise in enumerate(config.label_noise_levels):
                    condition_seed = instance_seed * 1000 + int(sample_count) * 10 + noise_index
                    sampled = _score_pair_response_cache(
                        model,
                        instance,
                        config,
                        response_cache,
                        sample_count=int(sample_count),
                        label_noise=float(label_noise),
                        seed=condition_seed,
                    )
                    metadata = {
                        "model_key": model.key,
                        "instance": instance_index,
                        "instance_seed": instance_seed,
                        "condition": "sampled",
                        "requested_sample_count": int(sample_count),
                        "label_noise": float(label_noise),
                    }
                    all_pair_rows.extend(sampled.assign(**metadata).to_dict("records"))
                    all_metrics.extend({**metadata, **row} for row in _screening_metrics(sampled))

    metrics = pd.DataFrame(all_metrics)
    pair_rows = pd.DataFrame(all_pair_rows)
    exact_pairs = pair_rows.loc[pair_rows["condition"].eq("exact_grid")].copy()
    exact_pairs["support_match"] = exact_pairs["max_pair_basin_label"].astype(bool).eq(exact_pairs["synergy"].gt(1e-12))
    for model in iter_domain_models():
        model_pairs = exact_pairs.loc[exact_pairs["model_key"].eq(model.key)].copy()
        model_metrics = metrics.loc[
            metrics["model_key"].eq(model.key)
            & metrics["condition"].eq("exact_grid")
            & metrics["score"].eq("synergy")
        ]
        instance_support_match = model_pairs.groupby("instance")["support_match"].all()
        summary_models[model.key] = {
            "display_name": model.display_name,
            "equation_label": model.equation_label,
            "critical_input": model.critical_input(),
            "saddle_state": model.saddle_state,
            "physical_meaning": model.physical_meaning,
            "support_correspondence": {
                "criterion": "max-pair final basin label equals 1 iff PEID synergy > 1e-12",
                "pairwise_match_rate": float(model_pairs["support_match"].mean()),
                "all_instances_exact_match": bool(instance_support_match.all()),
            },
            "exact_synergy": {
                "mean_auprc": float(model_metrics["auprc"].mean()),
                "mean_top_k_recall": float(model_metrics["top_k_recall"].mean()),
                "mean_top1_hit": float(model_metrics["top1_hit"].mean()),
                "mean_success_spearman": float(model_metrics["success_rate_spearman"].mean()),
            },
        }

    summary = {
        "experiment": "domain_pair_ignition",
        "models": summary_models,
        "metrics": metrics.to_dict("records"),
    }
    manifest = {
        "experiment": "domain_pair_ignition",
        "config": _jsonable_config(config),
        "model_keys": [model.key for model in iter_domain_models()],
        "analytic_truth": "pair is switchable iff summed saturated input exceeds the model saddle-node critical input; every singleton is below that input",
        "peid_definition": "I(delta_i,delta_j; basin)-I(delta_i; basin)-I(delta_j; basin)",
        "intervention_distribution": "independent discrete uniform amplitudes",
        "target": "physical-threshold final basin label for controlled target module",
        "cache_paths": {key: str(path) for key, path in paths.items()},
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n")
    with paths["pairs_jsonl"].open("w") as handle:
        for row in pair_rows.to_dict("records"):
            handle.write(json.dumps(row, allow_nan=True) + "\n")

    arrays: dict[str, np.ndarray] = {}
    for model_key, representative in representative_by_model.items():
        model = get_domain_model(model_key)
        arrays[f"{model_key}_truth_matrix"] = _truth_matrix_with_single_node_diagonal(model, representative, config)
        arrays[f"{model_key}_synergy_matrix"] = _matrix_from_pairs(representative, "synergy", config.source_count)
        arrays[f"{model_key}_success_rate_matrix"] = _matrix_from_pairs(representative, "grid_switch_rate", config.source_count)
    np.savez_compressed(paths["arrays_npz"], **arrays)

    representative_pairs = pair_rows.loc[pair_rows["condition"].eq("exact_grid") & pair_rows["instance"].eq(0)].copy()
    return {
        "summary": summary,
        "pair_rows": pair_rows,
        "metrics": metrics,
        "representative_pairs": representative_pairs,
        "cache_paths": paths,
    }


def plot_domain_pair_results(
    results: dict[str, Any],
    config: DomainPairIgnitionConfig,
    *,
    figure_dir: Path | None = None,
    report_asset_dir: Path | None = None,
) -> dict[str, dict[str, Path]]:
    figure_dir = config.output_dir / "figures" if figure_dir is None else Path(figure_dir)
    representative = results["representative_pairs"].copy()
    paths: dict[str, dict[str, Path]] = {}

    models = list(iter_domain_models())
    control_model = models[0]
    control_subset = representative.loc[representative["model_key"].eq(control_model.key)].copy()
    control_instance = _representative_instance_for_model(control_model, control_subset, config)
    control_labels = _single_node_total_strength_labels(control_model, control_instance, config)
    fractions = np.asarray(control_instance["weight_fractions"], dtype=float)
    graph = nx.DiGraph()
    target = "Target"
    for node in range(config.source_count):
        graph.add_edge(f"S{node}", target, weight=float(fractions[node]))

    fig, axis = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    source_y = np.linspace(1.0, -1.0, config.source_count)
    positions = {f"S{node}": (-1.2, float(source_y[node])) for node in range(config.source_count)}
    positions[target] = (1.15, 0.0)
    source_colors = ["#4C78A8" if label else "#B8B8B8" for label in control_labels]
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[f"S{node}" for node in range(config.source_count)],
        node_color=source_colors,
        node_size=420,
        edgecolors="black",
        linewidths=0.8,
        ax=axis,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[target],
        node_color="#F2C14E",
        node_size=1800,
        edgecolors="black",
        linewidths=1.0,
        ax=axis,
    )
    edge_widths = 1.0 + 4.0 * fractions
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=[(f"S{node}", target) for node in range(config.source_count)],
        width=edge_widths,
        edge_color="#555555",
        arrows=True,
        arrowsize=13,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.0",
        ax=axis,
    )
    labels = {f"S{node}": str(node) for node in range(config.source_count)}
    labels[target] = "target\nmodule\nY"
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=8, ax=axis)
    for node in range(config.source_count):
        axis.text(-0.72, source_y[node], f"{fractions[node]:.2f} $u_c$", va="center", ha="left", fontsize=7)
    axis.text(-1.55, -1.26, "candidate source nodes; no source-source edges", ha="left", va="center", fontsize=8)
    axis.text(0.35, -1.26, "edge width = controlled input weight", ha="left", va="center", fontsize=8)
    axis.set_xlim(-1.75, 1.75)
    axis.set_ylim(-1.45, 1.20)
    axis.axis("off")
    paths["control_structure"] = _save_figure(fig, figure_dir, "domain_pair_control_structure")

    fig, axes = plt.subplots(len(models), 2, figsize=(8.6, 8.2), constrained_layout=True)
    for row, model in enumerate(models):
        subset = representative.loc[representative["model_key"].eq(model.key)].copy()
        truth = _truth_matrix_with_single_node_diagonal(model, subset, config)
        synergy = _matrix_from_pairs(subset, "synergy", config.source_count)
        image_truth = axes[row, 0].imshow(truth, cmap="Greys", vmin=0.0, vmax=1.0)
        image_syn = axes[row, 1].imshow(synergy, cmap="YlOrBr", vmin=0.0)
        axes[row, 0].set_title(f"{model.display_name}: final basin label", fontsize=9)
        axes[row, 1].set_title(f"{model.display_name}: PEID Syn", fontsize=9)
        for axis in axes[row]:
            axis.set(xlabel="Source node", ylabel="Source node")
            axis.set_xticks(range(config.source_count))
            axis.set_yticks(range(config.source_count))
        for i in range(config.source_count):
            for j in range(config.source_count):
                truth_value = truth[i, j]
                axes[row, 0].text(j, i, f"{int(truth_value)}", ha="center", va="center", fontsize=6, color="white" if truth_value >= 0.5 else "black")
                if i == j:
                    continue
                syn_value = synergy[i, j]
                axes[row, 1].text(j, i, f"{syn_value:.2f}", ha="center", va="center", fontsize=5, color="white" if syn_value >= 0.18 else "black")
    fig.colorbar(image_truth, ax=axes[:, 0], shrink=0.72, label="Final basin label")
    fig.colorbar(image_syn, ax=axes[:, 1], shrink=0.72, label="Synergy (bits)")
    paths["screening"] = _save_figure(fig, figure_dir, "domain_pair_screening")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4), constrained_layout=True)
    colors = {"wilson_cowan": "#4C78A8", "allee": "#5F8F6B", "schlogl": "#D97732"}
    for axis, model in zip(axes, models, strict=True):
        subset = representative.loc[representative["model_key"].eq(model.key)].copy()
        axis.scatter(
            subset["synergy"],
            subset["grid_switch_rate"],
            c=subset["analytic_switchable"].map({True: colors[model.key], False: "0.75"}),
            edgecolor="black",
            linewidth=0.35,
            s=38,
        )
        axis.set_title(model.display_name, fontsize=9)
        axis.set(xlabel="PEID Syn (bits)", ylabel="Grid ignition success rate", ylim=(-0.03, 1.03))
        axis.text(
            0.02,
            0.96,
            f"$u_c$={model.critical_input():.3g}",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
        )
    paths["success_vs_synergy"] = _save_figure(fig, figure_dir, "domain_pair_success_vs_synergy")

    if report_asset_dir is not None:
        report_asset_dir = Path(report_asset_dir)
        report_asset_dir.mkdir(parents=True, exist_ok=True)
        for item in paths.values():
            target = report_asset_dir / f"part3_{item['png'].name}"
            shutil.copyfile(item["png"], target)
            item["report_png"] = target
    return paths


def main() -> None:
    config = DomainPairIgnitionConfig()
    results = run_domain_pair_ensemble(config, force=False)
    plot_domain_pair_results(results, config, report_asset_dir=REPO_ROOT / "docs" / "reports" / "assets")
    print(json.dumps(results["summary"]["models"], indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
