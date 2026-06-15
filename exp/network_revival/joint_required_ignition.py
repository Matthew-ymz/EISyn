"""Joint-required ignition in a bistable threshold latch.

The controlled system is designed so that every singleton source has a
saturated input below the lower saddle-node threshold, while selected source
pairs exceed that threshold. Pair PEID is evaluated against the released
steady-state basin label under independent discrete interventions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from itertools import combinations
from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "network_revival_joint_required_ignition"

__all__ = [
    "JointRequiredIgnitionConfig",
    "critical_saddle_input",
    "build_threshold_latch_instance",
    "simulate_isolated_pair_intervention",
    "compute_pair_basin_peid",
    "compute_and_gate_control_peid",
    "run_joint_required_ensemble",
    "plot_joint_required_results",
]


@dataclass(frozen=True)
class JointRequiredIgnitionConfig:
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    source_count: int = 8
    theta: float = 0.4
    kappa: float = 8.0
    hill_coefficient: float = 4.0
    half_saturation: float = 1.0
    amplitude_max: float = 4.0
    amplitude_levels: int = 16
    t_force: float = 8.0
    release_time: float = 8.0
    dt: float = 0.04
    ensemble_size: int = 50
    sample_sizes: tuple[int, ...] = (128, 256, 512, 1024)
    label_noise_levels: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10)
    high_weight_fraction_range: tuple[float, float] = (0.62, 0.78)
    low_weight_fraction_range: tuple[float, float] = (0.10, 0.14)
    pair_margin: float = 0.06
    fixed_delta: float = 1.5
    seed: int = 20260615

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.source_count < 4:
            raise ValueError("source_count must be at least four.")
        if not 0.0 < self.theta < 1.0:
            raise ValueError("theta must lie strictly between zero and one.")
        if self.kappa <= 0.0 or self.dt <= 0.0:
            raise ValueError("kappa and dt must be positive.")
        if self.hill_coefficient <= 0.0 or self.half_saturation <= 0.0:
            raise ValueError("Hill parameters must be positive.")
        if self.amplitude_max <= 0.0 or self.amplitude_levels < 3:
            raise ValueError("amplitude grid must contain at least three nonnegative states.")
        if self.t_force <= 0.0 or self.release_time < 0.0:
            raise ValueError("forcing time must be positive and release time nonnegative.")
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


def critical_saddle_input(theta: float) -> float:
    """Return the positive input that removes the lower stable latch state."""

    theta = float(theta)
    if not 0.0 < theta < 1.0:
        raise ValueError("theta must lie strictly between zero and one.")
    z_minus = (1.0 + theta - np.sqrt(1.0 - theta + theta**2)) / 3.0
    return float(-z_minus * (1.0 - z_minus) * (z_minus - theta))


def _source_pairs(source_count: int) -> list[tuple[int, int]]:
    return list(combinations(range(int(source_count)), 2))


def build_threshold_latch_instance(
    config: JointRequiredIgnitionConfig,
    seed: int,
) -> dict[str, Any]:
    """Build one instance with strict singleton failure and separated pair truth."""

    rng = np.random.default_rng(int(seed))
    critical = critical_saddle_input(config.theta)
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
        raise RuntimeError("generated latch instance must contain both pair classes.")
    if np.any(np.abs(pair_ratios - 1.0) < config.pair_margin):
        raise RuntimeError("generated latch instance violates the pair threshold margin.")

    return {
        "seed": int(seed),
        "theta": float(config.theta),
        "kappa": float(config.kappa),
        "hill_coefficient": float(config.hill_coefficient),
        "half_saturation": float(config.half_saturation),
        "critical_input": critical,
        "weights": weights,
        "weight_fractions": fractions,
        "pairs": pairs,
        "pair_input_ratios": pair_ratios,
        "switchable_pairs": switchable,
        "nonswitchable_pairs": nonswitchable,
    }


def _hill(delta: np.ndarray | float, instance: dict[str, Any]) -> np.ndarray:
    values = np.maximum(np.asarray(delta, dtype=float), 0.0)
    exponent = float(instance["hill_coefficient"])
    half = float(instance["half_saturation"])
    return values**exponent / (half**exponent + values**exponent + 1e-15)


def _latch_drift(z: np.ndarray | float, theta: float) -> np.ndarray:
    state = np.asarray(z, dtype=float)
    return state * (1.0 - state) * (state - float(theta))


def _integrate_constant_input(
    initial: np.ndarray | float,
    input_value: np.ndarray | float,
    *,
    theta: float,
    kappa: float,
    duration: float,
    dt: float,
    record: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    state = np.asarray(initial, dtype=float).copy()
    forcing = np.asarray(input_value, dtype=float)
    times = [0.0] if record else None
    states = [float(state)] if record and state.ndim == 0 else None
    t = 0.0

    def rhs(value: np.ndarray) -> np.ndarray:
        return float(kappa) * (_latch_drift(value, theta) + forcing)

    while t < float(duration) - 1e-12:
        step = min(float(dt), float(duration) - t)
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * step * k1)
        k3 = rhs(state + 0.5 * step * k2)
        k4 = rhs(state + step * k3)
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += step
        if record and times is not None and states is not None:
            times.append(float(t))
            states.append(float(state))
    if record and times is not None and states is not None:
        return state, np.asarray(times, dtype=float), np.asarray(states, dtype=float)
    return state, None, None


def simulate_isolated_pair_intervention(
    instance: dict[str, Any],
    *,
    pair: tuple[int, int],
    delta_i: float,
    delta_j: float,
    t_force: float,
    release_time: float,
    dt: float,
) -> dict[str, Any]:
    """Clamp one pair, release it, and report the final latch basin."""

    left, right = sorted((int(pair[0]), int(pair[1])))
    weights = np.asarray(instance["weights"], dtype=float)
    input_value = float(
        weights[left] * _hill(delta_i, instance)
        + weights[right] * _hill(delta_j, instance)
    )
    forced, force_times, force_states = _integrate_constant_input(
        0.0,
        input_value,
        theta=float(instance["theta"]),
        kappa=float(instance["kappa"]),
        duration=float(t_force),
        dt=float(dt),
        record=True,
    )
    released, release_times, release_states = _integrate_constant_input(
        forced,
        0.0,
        theta=float(instance["theta"]),
        kappa=float(instance["kappa"]),
        duration=float(release_time),
        dt=float(dt),
        record=True,
    )
    times = np.concatenate([force_times, float(t_force) + release_times[1:]])
    trajectory = np.concatenate([force_states, release_states[1:]])
    return {
        "pair": (left, right),
        "input_value": input_value,
        "force_end": float(forced),
        "final_z": float(released),
        "basin_label": int(float(released) > float(instance["theta"])),
        "times": times,
        "trajectory": trajectory,
    }


def _entropy_codes(values: np.ndarray) -> float:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    _, counts = np.unique(array, axis=0, return_counts=True)
    probabilities = counts.astype(float) / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _mutual_information(source: np.ndarray, target: np.ndarray) -> float:
    left = np.asarray(source)
    right = np.asarray(target)
    if left.ndim == 1:
        left = left[:, None]
    if right.ndim == 1:
        right = right[:, None]
    return float(_entropy_codes(left) + _entropy_codes(right) - _entropy_codes(np.column_stack([left, right])))


def _conditional_mi(left: np.ndarray, right: np.ndarray, target: np.ndarray) -> float:
    a = np.asarray(left).reshape(-1, 1)
    b = np.asarray(right).reshape(-1, 1)
    y = np.asarray(target).reshape(-1, 1)
    return float(
        _entropy_codes(np.column_stack([a, y]))
        + _entropy_codes(np.column_stack([b, y]))
        - _entropy_codes(y)
        - _entropy_codes(np.column_stack([a, b, y]))
    )


def _pair_response_grid(
    instance: dict[str, Any],
    config: JointRequiredIgnitionConfig,
    pair: tuple[int, int],
) -> dict[str, np.ndarray]:
    amplitudes = np.linspace(0.0, config.amplitude_max, config.amplitude_levels)
    delta_i, delta_j = np.meshgrid(amplitudes, amplitudes, indexing="ij")
    left, right = pair
    weights = np.asarray(instance["weights"], dtype=float)
    forcing = weights[left] * _hill(delta_i.ravel(), instance) + weights[right] * _hill(delta_j.ravel(), instance)
    force_end, _, _ = _integrate_constant_input(
        np.zeros_like(forcing),
        forcing,
        theta=config.theta,
        kappa=config.kappa,
        duration=config.t_force,
        dt=config.dt,
    )
    labels = (force_end > config.theta).astype(int)
    return {
        "amplitudes": amplitudes,
        "source_i": np.repeat(np.arange(config.amplitude_levels), config.amplitude_levels),
        "source_j": np.tile(np.arange(config.amplitude_levels), config.amplitude_levels),
        "delta_i": delta_i.ravel(),
        "delta_j": delta_j.ravel(),
        "force_end": np.asarray(force_end, dtype=float),
        "labels": labels,
    }


def _peid_row(
    source_i: np.ndarray,
    source_j: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
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
    instance: dict[str, Any],
    config: JointRequiredIgnitionConfig,
) -> dict[tuple[int, int], dict[str, np.ndarray | float]]:
    cache: dict[tuple[int, int], dict[str, np.ndarray | float]] = {}
    for left, right in _source_pairs(config.source_count):
        response = _pair_response_grid(instance, config, (left, right))
        successful = response["labels"].astype(bool)
        costs = response["delta_i"] + response["delta_j"]
        fixed = simulate_isolated_pair_intervention(
            instance,
            pair=(left, right),
            delta_i=config.fixed_delta,
            delta_j=config.fixed_delta,
            t_force=config.t_force,
            release_time=0.0,
            dt=config.dt,
        )
        response["minimum_joint_cost"] = float(np.min(costs[successful])) if np.any(successful) else np.nan
        response["fixed_cost_response"] = float(fixed["force_end"])
        cache[(left, right)] = response
    return cache


def _score_pair_response_cache(
    instance: dict[str, Any],
    config: JointRequiredIgnitionConfig,
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
                "pair_i": left,
                "pair_j": right,
                "analytic_switchable": bool(weights[left] + weights[right] > critical),
                "max_pair_basin_label": int(response_labels[-1]),
                "pair_input_ratio": float((weights[left] + weights[right]) / critical),
                "grid_switch_rate": float(np.mean(response_labels)),
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


def compute_pair_basin_peid(
    instance: dict[str, Any],
    config: JointRequiredIgnitionConfig,
    *,
    sample_count: int | None = None,
    label_noise: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Score every isolated pair against its released high-basin label."""

    cache = _build_pair_response_cache(instance, config)
    return _score_pair_response_cache(
        instance,
        config,
        cache,
        sample_count=sample_count,
        label_noise=label_noise,
        seed=config.seed if seed is None else int(seed),
    )


def compute_and_gate_control_peid(
    config: JointRequiredIgnitionConfig,
    *,
    true_pair: tuple[int, int] = (0, 1),
) -> pd.DataFrame:
    """Return an explicit pair-product gate as a PEID positive control."""

    true_pair = tuple(sorted((int(true_pair[0]), int(true_pair[1]))))
    instance = build_threshold_latch_instance(config, config.seed)
    amplitudes = np.linspace(0.0, config.amplitude_max, config.amplitude_levels)
    delta_i, delta_j = np.meshgrid(amplitudes, amplitudes, indexing="ij")
    source_i = np.repeat(np.arange(config.amplitude_levels), config.amplitude_levels)
    source_j = np.tile(np.arange(config.amplitude_levels), config.amplitude_levels)
    gate_gain = 1.4 * float(instance["critical_input"])
    rows: list[dict[str, Any]] = []
    for pair in _source_pairs(config.source_count):
        if pair == true_pair:
            forcing = gate_gain * _hill(delta_i.ravel(), instance) * _hill(delta_j.ravel(), instance)
            force_end, _, _ = _integrate_constant_input(
                np.zeros_like(forcing),
                forcing,
                theta=config.theta,
                kappa=config.kappa,
                duration=config.t_force,
                dt=config.dt,
            )
            labels = (force_end > config.theta).astype(int)
        else:
            labels = np.zeros(delta_i.size, dtype=int)
        rows.append(
            {
                "pair_i": pair[0],
                "pair_j": pair[1],
                "is_true_pair": pair == true_pair,
                **_peid_row(source_i, source_j, labels),
            }
        )
    return pd.DataFrame(rows).sort_values(["synergy", "pair_i", "pair_j"], ascending=[False, True, True]).reset_index(drop=True)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    truth = np.asarray(labels, dtype=bool)[order]
    positive_count = int(np.sum(truth))
    if positive_count == 0:
        return float("nan")
    precision = np.cumsum(truth) / np.arange(1, len(truth) + 1)
    return float(np.sum(precision[truth]) / positive_count)


def _screening_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    labels = frame["analytic_switchable"].to_numpy(dtype=bool)
    positive_count = int(np.sum(labels))
    finite_cost = frame.loc[frame["analytic_switchable"] & frame["minimum_joint_cost"].notna()]
    best_cost = float(finite_cost["minimum_joint_cost"].min()) if len(finite_cost) else np.nan
    rows: list[dict[str, Any]] = []
    for score in ("synergy", "joint_ei", "single_ei_sum", "fixed_cost_response", "random_score"):
        values = frame[score].to_numpy(dtype=float)
        order = np.argsort(-values, kind="mergesort")
        top_k = labels[order[:positive_count]]
        selected_cost = float(frame.iloc[order[0]]["minimum_joint_cost"])
        if len(finite_cost) >= 2 and finite_cost[score].nunique() >= 2 and finite_cost["minimum_joint_cost"].nunique() >= 2:
            cost_spearman = float(finite_cost[score].corr(finite_cost["minimum_joint_cost"], method="spearman"))
        else:
            cost_spearman = np.nan
        rows.append(
            {
                "score": score,
                "auprc": _average_precision(values, labels),
                "top_k_recall": float(np.mean(top_k)) if positive_count else np.nan,
                "top1_hit": float(labels[order[0]]) if len(order) else np.nan,
                "minimum_cost_spearman": cost_spearman,
                "selected_cost_regret": float(selected_cost - best_cost) if np.isfinite(selected_cost) else np.nan,
            }
        )
    return rows


def _cache_paths(config: JointRequiredIgnitionConfig) -> dict[str, Path]:
    return {
        "summary_json": config.output_dir / "summary.json",
        "pairs_jsonl": config.output_dir / "pair_scores.jsonl",
        "arrays_npz": config.output_dir / "representative_arrays.npz",
        "manifest_json": config.output_dir / "manifest.json",
    }


def _jsonable_config(config: JointRequiredIgnitionConfig) -> dict[str, Any]:
    values = asdict(config)
    values["output_dir"] = str(config.output_dir)
    for key in ("sample_sizes", "label_noise_levels", "high_weight_fraction_range", "low_weight_fraction_range"):
        values[key] = list(values[key])
    return values


def _read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line.strip()])


def _matrix_from_pairs(frame: pd.DataFrame, value: str, source_count: int) -> np.ndarray:
    matrix = np.full((source_count, source_count), np.nan, dtype=float)
    for row in frame.itertuples(index=False):
        left, right = int(row.pair_i), int(row.pair_j)
        matrix[left, right] = matrix[right, left] = float(getattr(row, value))
    return matrix


def run_joint_required_ensemble(
    config: JointRequiredIgnitionConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Run or load the parameter ensemble and screening benchmark."""

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
    representative_instance: dict[str, Any] | None = None
    representative_pairs: pd.DataFrame | None = None

    for instance_index in range(config.ensemble_size):
        instance_seed = config.seed + instance_index
        instance = build_threshold_latch_instance(config, instance_seed)
        response_cache = _build_pair_response_cache(instance, config)
        exact = _score_pair_response_cache(
            instance,
            config,
            response_cache,
            sample_count=None,
            label_noise=0.0,
            seed=instance_seed + 10000,
        )
        if instance_index == 0:
            representative_instance = instance
            representative_pairs = exact.copy()
        exact_meta = {
            "instance": instance_index,
            "instance_seed": instance_seed,
            "condition": "exact_grid",
            "requested_sample_count": config.amplitude_levels**2,
            "label_noise": 0.0,
        }
        all_pair_rows.extend(exact.assign(**exact_meta).to_dict("records"))
        all_metrics.extend({**exact_meta, **row} for row in _screening_metrics(exact))

        for sample_count in config.sample_sizes:
            for noise_index, label_noise in enumerate(config.label_noise_levels):
                condition_seed = instance_seed * 1000 + int(sample_count) * 10 + noise_index
                sampled = _score_pair_response_cache(
                    instance,
                    config,
                    response_cache,
                    sample_count=int(sample_count),
                    label_noise=float(label_noise),
                    seed=condition_seed,
                )
                metadata = {
                    "instance": instance_index,
                    "instance_seed": instance_seed,
                    "condition": "sampled",
                    "requested_sample_count": int(sample_count),
                    "label_noise": float(label_noise),
                }
                all_pair_rows.extend(sampled.assign(**metadata).to_dict("records"))
                all_metrics.extend({**metadata, **row} for row in _screening_metrics(sampled))

    assert representative_instance is not None and representative_pairs is not None
    and_control = compute_and_gate_control_peid(config, true_pair=(0, 1))
    metrics = pd.DataFrame(all_metrics)
    pair_rows = pd.DataFrame(all_pair_rows)
    exact_synergy = metrics.loc[
        metrics["condition"].eq("exact_grid") & metrics["score"].eq("synergy")
    ]
    noisy_synergy = metrics.loc[
        metrics["condition"].eq("sampled")
        & metrics["score"].eq("synergy")
        & metrics["requested_sample_count"].eq(512)
        & np.isclose(metrics["label_noise"], 0.05)
    ]
    gates = {
        "exact_mean_auprc": float(exact_synergy["auprc"].mean()),
        "exact_mean_top_k_recall": float(exact_synergy["top_k_recall"].mean()),
        "sample512_noise005_mean_auprc": float(noisy_synergy["auprc"].mean()) if len(noisy_synergy) else float("nan"),
    }
    gates["claim_gate_passed"] = bool(
        gates["exact_mean_auprc"] >= 0.90
        and gates["exact_mean_top_k_recall"] >= 0.90
        and gates["sample512_noise005_mean_auprc"] >= 0.80
    )
    exact_pairs = pair_rows.loc[pair_rows["condition"].eq("exact_grid")].copy()
    exact_pairs["support_match"] = exact_pairs["max_pair_basin_label"].astype(bool).eq(
        exact_pairs["synergy"].gt(1e-12)
    )
    instance_support_match = exact_pairs.groupby("instance")["support_match"].all()
    summary = {
        "experiment": "joint_required_threshold_latch",
        "claim_gates": gates,
        "support_correspondence": {
            "criterion": "max-pair final basin label equals 1 iff PEID synergy > 1e-12",
            "pairwise_match_rate": float(exact_pairs["support_match"].mean()),
            "all_instances_exact_match": bool(instance_support_match.all()),
        },
        "and_gate_control": {
            "true_pair": [0, 1],
            "top_synergy_pair": [
                int(and_control.iloc[0]["pair_i"]),
                int(and_control.iloc[0]["pair_j"]),
            ],
            "top_synergy": float(and_control.iloc[0]["synergy"]),
        },
        "metrics": metrics.to_dict("records"),
    }
    manifest = {
        "experiment": "joint_required_threshold_latch",
        "config": _jsonable_config(config),
        "critical_input": float(representative_instance["critical_input"]),
        "analytic_truth": "pair is switchable iff w_i + w_j > u_SN; every w_i < u_SN",
        "peid_definition": "I(delta_i,delta_j; basin)-I(delta_i; basin)-I(delta_j; basin)",
        "intervention_distribution": "independent discrete uniform amplitudes",
        "target": "released final basin label",
        "support_correspondence": "max-pair final basin label equals 1 iff PEID synergy > 1e-12",
        "cache_paths": {key: str(path) for key, path in paths.items()},
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n")
    with paths["pairs_jsonl"].open("w") as handle:
        for row in pair_rows.to_dict("records"):
            handle.write(json.dumps(row, allow_nan=True) + "\n")
    np.savez_compressed(
        paths["arrays_npz"],
        weights=np.asarray(representative_instance["weights"], dtype=float),
        truth_matrix=_matrix_from_pairs(representative_pairs, "analytic_switchable", config.source_count),
        synergy_matrix=_matrix_from_pairs(representative_pairs, "synergy", config.source_count),
        joint_ei_matrix=_matrix_from_pairs(representative_pairs, "joint_ei", config.source_count),
    )
    return {
        "summary": summary,
        "pair_rows": pair_rows,
        "metrics": metrics,
        "representative_pairs": representative_pairs.assign(
            instance=0,
            instance_seed=config.seed,
            condition="exact_grid",
            requested_sample_count=config.amplitude_levels**2,
            label_noise=0.0,
        ),
        "cache_paths": paths,
    }


def _save_figure(fig: plt.Figure, output_dir: Path, basename: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": output_dir / f"{basename}.png",
        "pdf": output_dir / f"{basename}.pdf",
    }
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_joint_required_results(
    results: dict[str, Any],
    config: JointRequiredIgnitionConfig,
    *,
    figure_dir: Path | None = None,
    report_asset_dir: Path | None = None,
) -> dict[str, dict[str, Path]]:
    """Create the mechanism, pair-screening, and ensemble-performance figures."""

    figure_dir = config.output_dir / "figures" if figure_dir is None else Path(figure_dir)
    representative = results["representative_pairs"].copy()
    instance = build_threshold_latch_instance(config, config.seed)
    paths: dict[str, dict[str, Path]] = {}

    strongest = int(np.argmax(np.asarray(instance["weights"])))
    valid_pair = tuple(representative.loc[representative["analytic_switchable"]].iloc[0][["pair_i", "pair_j"]].astype(int))
    single = simulate_isolated_pair_intervention(
        instance,
        pair=(strongest, (strongest + 1) % config.source_count),
        delta_i=1e6,
        delta_j=0.0,
        t_force=config.t_force,
        release_time=config.release_time,
        dt=config.dt,
    )
    pair = simulate_isolated_pair_intervention(
        instance,
        pair=valid_pair,
        delta_i=1e6,
        delta_j=1e6,
        t_force=config.t_force,
        release_time=config.release_time,
        dt=config.dt,
    )
    z = np.linspace(-0.05, 1.15, 400)
    critical = float(instance["critical_input"])
    single_u = float(np.max(instance["weights"]))
    pair_u = float(instance["weights"][valid_pair[0]] + instance["weights"][valid_pair[1]])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), constrained_layout=True)
    for value, label, color in (
        (0.0, "No input", "0.45"),
        (single_u, "Strongest singleton", "#4C78A8"),
        (pair_u, "Switchable pair", "#D97732"),
    ):
        axes[0].plot(z, config.kappa * (_latch_drift(z, config.theta) + value), label=label, color=color)
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].axvline(config.theta, color="0.3", lw=0.8, ls=":")
    axes[0].text(config.theta + 0.02, axes[0].get_ylim()[1] * 0.82, r"$z=\theta$", fontsize=8)
    axes[0].set(xlabel="Latch state z", ylabel=r"$\dot z$")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    axes[1].plot(single["times"], single["trajectory"], color="#4C78A8", label="Singleton")
    axes[1].plot(pair["times"], pair["trajectory"], color="#D97732", label="Pair")
    axes[1].axhline(config.theta, color="0.3", lw=0.8, ls=":")
    axes[1].axvline(config.t_force, color="0.3", lw=0.8, ls="--")
    axes[1].set(xlabel="Time", ylabel="Latch state z")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    paths["mechanism"] = _save_figure(fig, figure_dir, "joint_required_mechanism")

    truth = _matrix_from_pairs(representative, "max_pair_basin_label", config.source_count)
    synergy = _matrix_from_pairs(representative, "synergy", config.source_count)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), constrained_layout=True)
    image_truth = axes[0].imshow(truth, cmap="Greys", vmin=0.0, vmax=1.0)
    image_syn = axes[1].imshow(synergy, cmap="YlOrBr", vmin=0.0)
    axes[0].set_title("Final basin label after pair ignition", fontsize=9)
    axes[1].set_title("Basin-label PEID synergy", fontsize=9)
    for axis in axes:
        axis.set(xlabel="Source node", ylabel="Source node")
        axis.set_xticks(range(config.source_count))
        axis.set_yticks(range(config.source_count))
    for row in range(config.source_count):
        for column in range(config.source_count):
            if row == column:
                continue
            truth_value = truth[row, column]
            synergy_value = synergy[row, column]
            axes[0].text(
                column,
                row,
                f"{int(truth_value)}",
                ha="center",
                va="center",
                color="white" if truth_value >= 0.5 else "black",
                fontsize=7,
                fontweight="bold",
            )
            axes[1].text(
                column,
                row,
                f"{synergy_value:.2f}",
                ha="center",
                va="center",
                color="white" if synergy_value >= 0.18 else "black",
                fontsize=6,
            )
    truth_colorbar = fig.colorbar(image_truth, ax=axes[0], shrink=0.82, label="Final basin label")
    truth_colorbar.set_ticks([0.0, 1.0])
    fig.colorbar(image_syn, ax=axes[1], shrink=0.82, label="Synergy (bits)")
    paths["screening"] = _save_figure(fig, figure_dir, "joint_required_pair_screening")

    metrics = results["metrics"].copy()
    exact = metrics.loc[metrics["condition"].eq("exact_grid")].groupby("score", as_index=False)["auprc"].agg(["mean", "std"]).reset_index()
    score_order = ["synergy", "joint_ei", "single_ei_sum", "fixed_cost_response", "random_score"]
    exact = exact.set_index("score").reindex(score_order).reset_index()
    sample_size = 128 if 128 in config.sample_sizes else int(config.sample_sizes[0])
    robust = metrics.loc[
        metrics["condition"].eq("sampled") & metrics["requested_sample_count"].eq(sample_size)
    ].groupby(["label_noise", "score"], as_index=False)[["auprc", "top_k_recall"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.5), constrained_layout=True)
    axes[0].bar(np.arange(len(exact)), exact["mean"], yerr=exact["std"], color=["#D97732", "#7068A8", "#5F8F6B", "#4C78A8", "0.65"])
    axes[0].set_xticks(np.arange(len(exact)), ["Syn", "Joint EI", "Single EI sum", "Fixed response", "Random"], rotation=25, ha="right")
    axes[0].set(ylabel="AUPRC", ylim=(0.0, 1.05))
    colors = {"synergy": "#D97732", "joint_ei": "#7068A8", "single_ei_sum": "#5F8F6B", "fixed_cost_response": "#4C78A8", "random_score": "0.65"}
    labels = {"synergy": "PEID Syn", "joint_ei": "Joint EI", "single_ei_sum": "Single EI sum", "fixed_cost_response": "Fixed response", "random_score": "Random"}
    for score in score_order:
        subset = robust.loc[robust["score"].eq(score)].sort_values("label_noise")
        axes[1].plot(subset["label_noise"], subset["auprc"], marker="o", color=colors[score], label=labels[score])
        axes[2].plot(subset["label_noise"], subset["top_k_recall"], marker="o", color=colors[score], label=labels[score])
    axes[1].set(xlabel=f"Basin-label noise ({sample_size} samples)", ylabel="AUPRC", ylim=(0.0, 1.05))
    axes[2].set(xlabel=f"Basin-label noise ({sample_size} samples)", ylabel=r"Top-$k$ recall", ylim=(0.0, 1.05))
    axes[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    paths["performance"] = _save_figure(fig, figure_dir, "joint_required_ensemble_performance")

    if report_asset_dir is not None:
        report_asset_dir = Path(report_asset_dir)
        report_asset_dir.mkdir(parents=True, exist_ok=True)
        for key, item in paths.items():
            target = report_asset_dir / f"part3_{item['png'].name}"
            shutil.copyfile(item["png"], target)
            item["report_png"] = target
    return paths
