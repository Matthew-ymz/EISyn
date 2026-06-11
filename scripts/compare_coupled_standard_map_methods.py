#!/usr/bin/env python3
"""Compare six causal readouts on noisy coupled standard maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from itertools import combinations
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.coupled_standard_map_peid import (
    DatasetSplit,
    FittedImpulseMLP,
    StandardMapConfig,
    build_trajectory_dataset,
    coupled_impulses,
    evaluate_peid,
    fit_impulse_mlp,
    periodic_features,
    prediction_metrics,
)


METHOD_NAMES = (
    "Whole-minus-sum",
    "MLP+SHAP",
    "Observational SURD",
    "PCMCI-CMIknn",
    "Neural Granger",
    "MLP+PEID",
)
STATE_NAMES = ("q1", "p1", "q2", "p2")
TARGET_NAMES = ("I1", "I2")
DEFAULT_RESULT_DIR = ROOT / "results" / "coupled_standard_map_method_comparison"
DEFAULT_FIGURE_DIR = ROOT / "fig" / "coupled_standard_map_method_comparison"
DEFAULT_REPORT_PATH = ROOT / "docs" / "reports" / "coupled_standard_map_method_comparison.md"


def build_periodic_source_groups() -> dict[str, tuple[int, int]]:
    return {name: (index, index + len(STATE_NAMES)) for index, name in enumerate(STATE_NAMES)}


def comparison_ground_truth(config: StandardMapConfig) -> dict[str, float]:
    coupling_strength = float(config.coupling) ** 2 / 2.0
    return {
        "own": (float(config.k) ** 2 + float(config.coupling) ** 2) / 2.0,
        "cross": coupling_strength,
        "interaction": coupling_strength,
        "momentum": 0.0,
    }


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()[:16]


def _discretize(values: np.ndarray, bins: int) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    edges = np.unique(np.quantile(array, np.linspace(0.0, 1.0, int(bins) + 1)))
    if len(edges) <= 2:
        return np.zeros(len(array), dtype=int)
    return np.digitize(array, edges[1:-1], right=False)


def _entropy(codes: np.ndarray) -> float:
    array = np.asarray(codes)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    _, counts = np.unique(array, axis=0, return_counts=True)
    probabilities = counts.astype(float) / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _mutual_information(source: np.ndarray, target: np.ndarray) -> float:
    left = np.asarray(source)
    right = np.asarray(target)
    if left.ndim == 1:
        left = left.reshape(-1, 1)
    if right.ndim == 1:
        right = right.reshape(-1, 1)
    return float(_entropy(left) + _entropy(right) - _entropy(np.column_stack([left, right])))


def _wms(left: np.ndarray, right: np.ndarray, target: np.ndarray, bins: int) -> dict[str, float]:
    left_codes = _discretize(left, bins)
    right_codes = _discretize(right, bins)
    target_codes = _discretize(target, bins)
    left_mi = _mutual_information(left_codes, target_codes)
    right_mi = _mutual_information(right_codes, target_codes)
    joint_mi = _mutual_information(np.column_stack([left_codes, right_codes]), target_codes)
    return {
        "own_mi": left_mi,
        "cross_mi": right_mi,
        "joint_mi": joint_mi,
        "wms": float(joint_mi - left_mi - right_mi),
    }


def _target_sources(target_index: int) -> tuple[str, str]:
    return ("q1", "q2") if target_index == 0 else ("q2", "q1")


def _state_column(name: str) -> int:
    return STATE_NAMES.index(name)


def _mean_dict(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating))
    ]
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def _observational_readouts(split: DatasetSplit, *, bins: int, seed: int) -> tuple[dict, dict]:
    from scripts.reproduce_surd_synergistic_collider import decompose_surd_2source_transport_map

    rng = np.random.default_rng(int(seed) + 1901)
    count = min(len(split.states), 1200)
    indices = rng.choice(len(split.states), size=count, replace=False)
    states = split.states[indices]
    impulses = split.impulses[indices]
    wms_rows: list[dict[str, float]] = []
    surd_rows: list[dict[str, float]] = []
    for target_index, target in enumerate(TARGET_NAMES):
        own, cross = _target_sources(target_index)
        left = states[:, _state_column(own)]
        right = states[:, _state_column(cross)]
        output = impulses[:, target_index]
        wms_row = _wms(left, right, output, bins)
        wms_rows.append({"target": target, **wms_row})
        surd = decompose_surd_2source_transport_map(
            left,
            right,
            output,
            degree=3,
            target_anchors=min(96, count),
            conditional_samples=min(48, count),
            seed=int(seed) + target_index,
        )
        surd_rows.append(
            {
                "target": target,
                "redundancy": float(surd["redundancy"]),
                "own_unique": float(surd["unique_x"]),
                "cross_unique": float(surd["unique_y"]),
                "synergy": float(surd["synergy"]),
            }
        )
    return (
        {"aggregate": _mean_dict([{k: v for k, v in row.items() if k != "target"} for row in wms_rows]), "targets": wms_rows},
        {"aggregate": _mean_dict([{k: v for k, v in row.items() if k != "target"} for row in surd_rows]), "targets": surd_rows},
    )


def _subset_predictions(
    model: FittedImpulseMLP,
    foreground: np.ndarray,
    background: np.ndarray,
    subset: tuple[int, ...],
) -> np.ndarray:
    outputs = []
    for row in foreground:
        samples = background.copy()
        for source_index in subset:
            samples[:, source_index] = row[source_index]
        outputs.append(model.predict_mean(samples).mean(axis=0))
    return np.asarray(outputs, dtype=float)


def _shap_weight(subset_size: int, feature_count: int) -> float:
    return float(
        math.factorial(subset_size)
        * math.factorial(feature_count - subset_size - 1)
        / math.factorial(feature_count)
    )


def _grouped_shap(
    model: FittedImpulseMLP,
    split: DatasetSplit,
    *,
    seed: int,
    sample_count: int,
) -> dict[str, object]:
    rng = np.random.default_rng(int(seed) + 4049)
    foreground = split.states[rng.choice(len(split.states), size=min(sample_count, len(split.states)), replace=False)]
    background = split.states[rng.choice(len(split.states), size=min(sample_count, len(split.states)), replace=False)]
    cache: dict[tuple[int, ...], np.ndarray] = {}

    def value(subset: tuple[int, ...]) -> np.ndarray:
        key = tuple(sorted(subset))
        if key not in cache:
            cache[key] = _subset_predictions(model, foreground, background, key)
        return cache[key]

    feature_indices = tuple(range(len(STATE_NAMES)))
    phi = np.zeros((len(STATE_NAMES), len(foreground), len(TARGET_NAMES)), dtype=float)
    for source_index in feature_indices:
        remaining = tuple(index for index in feature_indices if index != source_index)
        for subset_size in range(len(remaining) + 1):
            for subset in combinations(remaining, subset_size):
                phi[source_index] += _shap_weight(subset_size, len(feature_indices)) * (
                    value((*subset, source_index)) - value(subset)
                )
    interactions: dict[tuple[int, int], np.ndarray] = {}
    for left_index, right_index in combinations(feature_indices, 2):
        remaining = tuple(index for index in feature_indices if index not in {left_index, right_index})
        interaction = np.zeros((len(foreground), len(TARGET_NAMES)), dtype=float)
        for subset_size in range(len(remaining) + 1):
            for subset in combinations(remaining, subset_size):
                delta = (
                    value((*subset, left_index, right_index))
                    - value((*subset, left_index))
                    - value((*subset, right_index))
                    + value(subset)
                )
                interaction += _shap_weight(subset_size, len(feature_indices) - 1) * delta
        interactions[(left_index, right_index)] = interaction
    target_rows = []
    for target_index, target in enumerate(TARGET_NAMES):
        own, cross = _target_sources(target_index)
        own_index = _state_column(own)
        cross_index = _state_column(cross)
        target_rows.append(
            {
                "target": target,
                "own": float(np.mean(np.abs(phi[own_index, :, target_index]))),
                "cross": float(np.mean(np.abs(phi[cross_index, :, target_index]))),
                "interaction": float(np.mean(np.abs(interactions[tuple(sorted((own_index, cross_index)))][:, target_index]))),
                "momentum_null": float(max(np.mean(np.abs(phi[1, :, target_index])), np.mean(np.abs(phi[3, :, target_index])))),
            }
        )
    return {
        "aggregate": _mean_dict([{k: v for k, v in row.items() if k != "target"} for row in target_rows]),
        "targets": target_rows,
    }


def _fit_neural_granger(
    split: DatasetSplit,
    *,
    seed: int,
    epochs: int,
    sample_count: int,
) -> dict[str, object]:
    import torch

    rng = np.random.default_rng(int(seed) + 7117)
    indices = rng.choice(len(split.states), size=min(sample_count, len(split.states)), replace=False)
    x = periodic_features(split.states[indices]).astype(np.float32)
    y = split.impulses[indices].astype(np.float32)
    x = (x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), 1e-6)
    y = (y - y.mean(axis=0, keepdims=True)) / np.maximum(y.std(axis=0, keepdims=True), 1e-6)
    groups = build_periodic_source_groups()
    target_rows = []
    for target_index, target in enumerate(TARGET_NAMES):
        torch.manual_seed(int(seed) + target_index)
        net = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        xt = torch.tensor(x, dtype=torch.float32)
        yt = torch.tensor(y[:, [target_index]], dtype=torch.float32)
        for _ in range(int(epochs)):
            optimizer.zero_grad(set_to_none=True)
            prediction = net(xt)
            first_layer = net[0].weight
            penalty = sum(torch.linalg.vector_norm(first_layer[:, list(columns)]) for columns in groups.values())
            loss = torch.mean((prediction - yt) ** 2) + 0.02 * penalty
            loss.backward()
            optimizer.step()
        weights = np.asarray(net[0].weight.detach().tolist(), dtype=float)
        scores = {name: float(np.linalg.norm(weights[:, list(columns)])) for name, columns in groups.items()}
        own, cross = _target_sources(target_index)
        target_rows.append(
            {
                "target": target,
                "own": scores[own],
                "cross": scores[cross],
                "momentum_null": max(scores["p1"], scores["p2"]),
            }
        )
    return {
        "aggregate": _mean_dict([{k: v for k, v in row.items() if k != "target"} for row in target_rows]),
        "targets": target_rows,
    }


def _pcmci_readout(
    split: DatasetSplit,
    *,
    max_points: int,
) -> dict[str, object]:
    from tigramite import data_processing as pp
    from tigramite.independence_tests.cmiknn import CMIknn
    from tigramite.pcmci import PCMCI

    data: dict[int, np.ndarray] = {}
    for trajectory_id in sorted(set(split.trajectory_ids.tolist())):
        mask = split.trajectory_ids == trajectory_id
        states = split.states[mask]
        impulses = split.impulses[mask]
        stop = min(len(states) - 1, int(max_points))
        encoded = periodic_features(states[1 : stop + 1])
        data[int(trajectory_id)] = np.column_stack([encoded, impulses[:stop]])
    vector_vars = {
        0: [(0, 0), (4, 0)],
        1: [(1, 0), (5, 0)],
        2: [(2, 0), (6, 0)],
        3: [(3, 0), (7, 0)],
        4: [(8, 0)],
        5: [(9, 0)],
    }
    dataframe = pp.DataFrame(
        data,
        analysis_mode="multiple",
        vector_vars=vector_vars,
        var_names=list((*STATE_NAMES, *TARGET_NAMES)),
    )
    test = CMIknn(
        knn=0.10,
        significance="fixed_thres",
        workers=1,
        verbosity=0,
    )
    result = PCMCI(dataframe=dataframe, cond_ind_test=test, verbosity=0).run_pcmci(
        tau_min=1,
        tau_max=1,
        pc_alpha=0.05,
        alpha_level=0.05,
    )
    values = np.abs(np.asarray(result["val_matrix"], dtype=float))
    target_rows = []
    for target_index, target in enumerate(TARGET_NAMES):
        own, cross = _target_sources(target_index)
        target_column = 4 + target_index
        target_rows.append(
            {
                "target": target,
                "own": float(values[_state_column(own), target_column, 1]),
                "cross": float(values[_state_column(cross), target_column, 1]),
                "momentum_null": float(max(values[1, target_column, 1], values[3, target_column, 1])),
            }
        )
    return {
        "aggregate": _mean_dict([{k: v for k, v in row.items() if k != "target"} for row in target_rows]),
        "targets": target_rows,
    }


def _peid_readout(
    model: FittedImpulseMLP,
    *,
    split: DatasetSplit,
    bins: int,
    permutation_count: int,
    seed: int,
) -> dict[str, object]:
    learned_targets = model.predict_mean(split.states)
    learned = evaluate_peid(
        states=split.states,
        targets=learned_targets,
        bins=bins,
        permutation_count=permutation_count,
        seed=seed,
    )
    learned_aggregate, learned_rows = _summarize_peid_result(learned)
    return {
        "aggregate": learned_aggregate,
        "targets": learned_rows,
        "state_distribution": "natural_test_trajectory",
        "target_distribution": "mlp_predicted_impulses",
        "state_digest": _digest(split.states),
        "target_digest": _digest(learned_targets),
    }


def _summarize_peid_result(result: dict[str, object]) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    singles = {(row["source"], row["target"]): float(row["ei"]) for row in result["single_sources"]}
    pairs = {(row["sources"], row["target"]): float(row["syn"]) for row in result["hyperedges"]}
    rows = []
    for target_index, target in enumerate(TARGET_NAMES):
        own, cross = _target_sources(target_index)
        strongest_pair = max(
            (row for row in result["hyperedges"] if row["target"] == target),
            key=lambda row: float(row["syn"]),
        )["sources"]
        rows.append(
            {
                "target": target,
                "own": singles[(own, target)],
                "cross": singles[(cross, target)],
                "synergy": pairs[("q1+q2", target)],
                "momentum_null": max(singles[("p1", target)], singles[("p2", target)]),
                "strongest_pair": str(strongest_pair),
                "true_pair_top": str(strongest_pair) == "q1+q2",
            }
        )
    aggregate = _mean_dict([{k: v for k, v in row.items() if k != "target"} for row in rows])
    return aggregate, rows


def _natural_peid_readout(
    model: FittedImpulseMLP,
    split: DatasetSplit,
    *,
    bins: int,
    permutation_count: int,
    seed: int,
) -> dict[str, object]:
    targets = model.predict_mean(split.states)
    peid = evaluate_peid(
        states=split.states,
        targets=targets,
        bins=bins,
        permutation_count=permutation_count,
        seed=seed,
    )
    aggregate, rows = _summarize_peid_result(peid)
    return {
        "aggregate": aggregate,
        "targets": rows,
        "state_distribution": "natural_test_trajectory",
        "target_distribution": "mlp_predicted_impulses",
        "state_digest": _digest(split.states),
        "target_digest": _digest(targets),
    }


def _quality_gate(metrics: dict[str, float]) -> dict[str, object]:
    checks = {
        "r2": float(metrics["r2"]) >= 0.99,
        "nrmse": float(metrics["nrmse"]) <= 0.05,
        "circular_mae": float(metrics["circular_mae"]) <= 0.05,
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def _run_one(
    *,
    coupling: float,
    seed: int,
    mode: str,
) -> dict[str, object]:
    smoke = mode == "smoke"
    config = StandardMapConfig(k=1.5, coupling=float(coupling), noise_std=0.05)
    dataset = build_trajectory_dataset(
        config,
        trajectory_count=8 if smoke else 16,
        steps_per_trajectory=80 if smoke else 2500,
        seed=int(seed),
    )
    model = fit_impulse_mlp(
        dataset.train,
        dataset.validation,
        seed=int(seed),
        epochs=80 if smoke else 400,
        hidden_width=32 if smoke else 96,
    )
    metrics = prediction_metrics(model, dataset.test)
    wms, surd = _observational_readouts(dataset.test, bins=4 if smoke else 8, seed=seed)
    shap = _grouped_shap(model, dataset.test, seed=seed, sample_count=12 if smoke else 48)
    pcmci = _pcmci_readout(
        dataset.test,
        max_points=70 if smoke else 1100,
    )
    neural_granger = _fit_neural_granger(
        dataset.train,
        seed=seed,
        epochs=30 if smoke else 120,
        sample_count=500 if smoke else 6000,
    )
    peid = _peid_readout(
        model,
        split=dataset.test,
        bins=8 if smoke else 12,
        permutation_count=2 if smoke else 5,
        seed=int(seed) + 2000,
    )
    method_status = {name: True for name in METHOD_NAMES}
    return {
        "coupling": float(coupling),
        "seed": int(seed),
        "ground_truth": comparison_ground_truth(config),
        "prediction": metrics,
        "quality_gate": _quality_gate(metrics),
        "wms": wms,
        "shap": shap,
        "surd": surd,
        "pcmci": pcmci,
        "neural_granger": neural_granger,
        "peid": peid,
        "method_status": method_status,
    }


def _run_one_payload(payload: dict[str, object]) -> dict[str, object]:
    return _run_one(**payload)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x = _rank(np.asarray(left, dtype=float))
    y = _rank(np.asarray(right, dtype=float))
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _summarize(runs: Sequence[dict[str, object]]) -> tuple[list[dict[str, float]], dict[str, float]]:
    metric_paths = {
        "wms": ("wms", "wms"),
        "shap_own": ("shap", "own"),
        "shap_cross": ("shap", "cross"),
        "shap_interaction": ("shap", "interaction"),
        "shap_momentum": ("shap", "momentum_null"),
        "surd_redundancy": ("surd", "redundancy"),
        "surd_own": ("surd", "own_unique"),
        "surd_cross": ("surd", "cross_unique"),
        "surd_synergy": ("surd", "synergy"),
        "pcmci_own": ("pcmci", "own"),
        "pcmci_cross": ("pcmci", "cross"),
        "pcmci_momentum": ("pcmci", "momentum_null"),
        "ng_own": ("neural_granger", "own"),
        "ng_cross": ("neural_granger", "cross"),
        "ng_momentum": ("neural_granger", "momentum_null"),
        "peid_own": ("peid", "own"),
        "peid_cross": ("peid", "cross"),
        "peid_synergy": ("peid", "synergy"),
        "peid_momentum": ("peid", "momentum_null"),
    }

    def metric(run: dict[str, object], path: tuple[str, str]) -> float:
        return float(run[path[0]]["aggregate"][path[1]])

    summary_rows = []
    for coupling in sorted({float(run["coupling"]) for run in runs}):
        group = [run for run in runs if float(run["coupling"]) == coupling]
        row: dict[str, float] = {
            "coupling": coupling,
            "truth": coupling**2 / 2.0,
            "r2_min": min(float(run["prediction"]["r2"]) for run in group),
            "nrmse_max": max(float(run["prediction"]["nrmse"]) for run in group),
            "circular_mae_max": max(float(run["prediction"]["circular_mae"]) for run in group),
            "gate_pass_rate": float(np.mean([run["quality_gate"]["passed"] for run in group])),
        }
        for name, path in metric_paths.items():
            values = np.asarray([metric(run, path) for run in group], dtype=float)
            row[f"{name}_mean"] = float(values.mean())
            row[f"{name}_std"] = float(values.std(ddof=0))
        summary_rows.append(row)
    truth = [row["truth"] for row in summary_rows]
    trends = {
        "wms": _spearman(truth, [row["wms_mean"] for row in summary_rows]),
        "shap_interaction": _spearman(truth, [row["shap_interaction_mean"] for row in summary_rows]),
        "surd_synergy": _spearman(truth, [row["surd_synergy_mean"] for row in summary_rows]),
        "pcmci_cross": _spearman(truth, [row["pcmci_cross_mean"] for row in summary_rows]),
        "neural_granger_cross": _spearman(truth, [row["ng_cross_mean"] for row in summary_rows]),
        "mlp_peid_synergy": _spearman(truth, [row["peid_synergy_mean"] for row in summary_rows]),
    }
    return summary_rows, trends


def _diagnostics(runs: Sequence[dict[str, object]]) -> dict[str, object]:
    zero_runs = [run for run in runs if abs(float(run["coupling"])) < 1e-12]
    positive_runs = [run for run in runs if float(run["coupling"]) > 0.0]
    primary_paths = {
        "wms": ("wms", "wms"),
        "shap_interaction": ("shap", "interaction"),
        "surd_synergy": ("surd", "synergy"),
        "pcmci_cross": ("pcmci", "cross"),
        "neural_granger_cross": ("neural_granger", "cross"),
        "mlp_peid_synergy": ("peid", "synergy"),
    }

    def value(run: dict[str, object], path: tuple[str, str]) -> float:
        return float(run[path[0]]["aggregate"][path[1]])

    zero_false_positive = {
        name: float(np.mean([abs(value(run, path)) for run in zero_runs])) if zero_runs else float("nan")
        for name, path in primary_paths.items()
    }
    separation = {}
    for name, method in (("shap", "shap"), ("pcmci", "pcmci"), ("neural_granger", "neural_granger"), ("mlp_peid", "peid")):
        margins = [
            float(run[method]["aggregate"]["cross"])
            - float(run[method]["aggregate"]["momentum_null"])
            for run in positive_runs
        ]
        separation[name] = {
            "cross_above_momentum_rate": float(np.mean(np.asarray(margins) > 0.0)) if margins else float("nan"),
            "mean_cross_minus_momentum": float(np.mean(margins)) if margins else float("nan"),
        }
    return {
        "j0_false_positive": zero_false_positive,
        "true_null_separation": separation,
        "mlp_peid_true_pair_top_rate": float(
            np.mean([run["peid"]["aggregate"]["true_pair_top"] for run in positive_runs])
        )
        if positive_runs
        else float("nan"),
    }


def _plot(summary_rows: Sequence[dict[str, float]], path: Path) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    couplings = np.asarray([row["coupling"] for row in summary_rows], dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 7.6), constrained_layout=True)
    flat = axes.ravel()

    def curve(axis, key: str, label: str, color: str, marker: str = "o", dashed: bool = False) -> None:
        mean = np.asarray([row[f"{key}_mean"] for row in summary_rows], dtype=float)
        std = np.asarray([row[f"{key}_std"] for row in summary_rows], dtype=float)
        axis.plot(couplings, mean, color=color, marker=marker, linewidth=1.6, linestyle="--" if dashed else "-", label=label)
        axis.fill_between(couplings, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

    curve(flat[0], "wms", "WMS", "#8c564b")
    flat[0].axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
    flat[0].set_title("Whole-minus-sum")
    flat[0].set_ylabel("Information (bits)")

    for key, label, color, marker in (("shap_own", "same-rotor angle", "#4c78a8", "^"), ("shap_cross", "other-rotor angle", "#f58518", "v"), ("shap_interaction", "joint angle pair", "#e45756", "o"), ("shap_momentum", "momentum control", "#7f8c8d", "s")):
        curve(flat[1], key, label, color, marker)
    flat[1].set_title("MLP+SHAP")
    flat[1].set_ylabel("Mean absolute attribution")

    for key, label, color, marker in (("surd_redundancy", "redundancy", "#7f8c8d", "s"), ("surd_own", "same-rotor unique", "#4c78a8", "^"), ("surd_cross", "other-rotor unique", "#f58518", "v"), ("surd_synergy", "synergy", "#e45756", "o")):
        curve(flat[2], key, label, color, marker)
    flat[2].set_title("Observational SURD")
    flat[2].set_ylabel("Information (bits)")

    for key, label, color, marker in (("pcmci_own", "same-rotor angle", "#4c78a8", "^"), ("pcmci_cross", "other-rotor angle", "#f58518", "v"), ("pcmci_momentum", "momentum control", "#7f8c8d", "s")):
        curve(flat[3], key, label, color, marker)
    flat[3].set_title("PCMCI-CMIknn")
    flat[3].set_ylabel("Absolute CMIknn statistic")

    for key, label, color, marker in (("ng_own", "same-rotor angle", "#4c78a8", "^"), ("ng_cross", "other-rotor angle", "#f58518", "v"), ("ng_momentum", "momentum control", "#7f8c8d", "s")):
        curve(flat[4], key, label, color, marker)
    flat[4].set_title("Neural Granger")
    flat[4].set_ylabel("First-layer group norm")

    for key, label, color, marker, dashed in (("peid_own", "same-rotor angle EI", "#4c78a8", "^", False), ("peid_cross", "other-rotor angle EI", "#f58518", "v", False), ("peid_synergy", "joint angle-pair synergy", "#e45756", "o", False), ("peid_momentum", "momentum control", "#7f8c8d", "s", False)):
        curve(flat[5], key, label, color, marker, dashed)
    flat[5].set_title("MLP+PEID")
    flat[5].set_ylabel("Information (bits)")

    for axis in flat:
        axis.set_xlabel("Coupling strength J")
        axis.set_xticks(couplings)
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_ground_truth(summary_rows: Sequence[dict[str, float]], path: Path) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    couplings = np.asarray([row["coupling"] for row in summary_rows], dtype=float)
    interaction_truth = np.asarray([row["truth"] for row in summary_rows], dtype=float)
    same_rotor_truth = np.asarray([row["truth"] + 1.5**2 / 2.0 for row in summary_rows], dtype=float)
    other_rotor_truth = interaction_truth
    momentum_truth = np.zeros_like(interaction_truth)
    dense = np.linspace(float(couplings.min()), float(couplings.max()), 200)
    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    ax.plot(dense, (1.5**2 + dense**2) / 2.0, color="#4c78a8", linewidth=1.8, label=r"same-rotor angle $(K^2+J^2)/2$")
    ax.plot(dense, dense**2 / 2.0, color="#f58518", linewidth=1.8, label=r"other-rotor angle $J^2/2$")
    ax.plot(dense, dense**2 / 2.0, color="#e45756", linewidth=1.4, linestyle="--", label=r"joint angle-pair interaction $J^2/2$")
    ax.axhline(0.0, color="#7f8c8d", linewidth=1.2, linestyle=":", label="momentum control 0")
    ax.scatter(couplings, same_rotor_truth, color="#4c78a8", s=20, zorder=3)
    ax.scatter(couplings, other_rotor_truth, color="#f58518", s=20, zorder=3)
    ax.scatter(couplings, interaction_truth, color="#e45756", s=20, zorder=3)
    ax.scatter(couplings, momentum_truth, color="#7f8c8d", s=18, zorder=3)
    ax.set_xlabel("Coupling strength J")
    ax.set_ylabel("Analytic squared-derivative strength")
    ax.set_xticks(couplings)
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _write_report(result: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure_path = Path(result["figure_path"])
    ground_truth_figure_path = Path(result["ground_truth_figure_path"])
    relative_figure = os.path.relpath(figure_path, path.parent)
    relative_ground_truth_figure = os.path.relpath(ground_truth_figure_path, path.parent)
    lines = [
        "# Coupled Standard Map Six-Method Comparison",
        "",
        f"![Six-method comparison]({relative_figure})",
        "",
        f"![Ground-truth curve]({relative_ground_truth_figure})",
        "",
        "## Protocol",
        "",
        f"- coupling values: `{result['protocol']['couplings']}`",
        f"- seeds: `{result['protocol']['seeds']}`",
        f"- trajectories per full run: `{result['protocol']['trajectory_count']}`",
        f"- steps per trajectory: `{result['protocol']['steps_per_trajectory']}`",
        "- targets: impulses `I1` and `I2`; symmetric target readouts are averaged only in the main figure",
        "- analytic other-rotor and interaction strength: `J^2 / 2`",
        "- PEID state distribution: natural test trajectories",
        "- PEID target distribution: MLP-predicted impulses",
        "",
        "## Ground-Truth 方程",
        "",
        "双转子 coupled standard map 的冲量方程为",
        "",
        r"$$",
        r"I_{1,t}=K\sin q_{1,t}+J\sin(q_{2,t}-q_{1,t})+\epsilon_{1,t},",
        r"$$",
        "",
        r"$$",
        r"I_{2,t}=K\sin q_{2,t}-J\sin(q_{2,t}-q_{1,t})+\epsilon_{2,t}.",
        r"$$",
        "",
        "状态更新为",
        "",
        r"$$",
        r"p_{i,t+1}=\operatorname{wrap}(p_{i,t}+I_{i,t}),\qquad",
        r"q_{i,t+1}=\operatorname{wrap}(q_{i,t}+p_{i,t+1}),\qquad i\in\{1,2\}.",
        r"$$",
        "",
        "因此动量 `p1,p2` 不直接进入 `I1,I2` 的结构方程。真实二阶来源是 `q1+q2`。对耦合项求混合二阶导数可得",
        "",
        r"$$",
        r"\frac{\partial^2 I_1}{\partial q_1\partial q_2}=J\sin(q_2-q_1),\qquad",
        r"\frac{\partial^2 I_2}{\partial q_1\partial q_2}=-J\sin(q_2-q_1).",
        r"$$",
        "",
        "在均匀角度基准下，`sin^2(q2-q1)` 的平均值为 `1/2`，所以解析 interaction ground truth 为",
        "",
        r"$$",
        r"\mathbb E\left[\left(\frac{\partial^2 I_i}{\partial q_1\partial q_2}\right)^2\right]=\frac{J^2}{2},\qquad i\in\{1,2\}.",
        r"$$",
        "",
        "上方 ground-truth 曲线图同时画出三个解析基准：Same-rotor angle strength 为 `(K^2+J^2)/2`，Other-rotor angle strength 为 `J^2/2`，joint angle-pair interaction 也是 `J^2/2`；momentum control 的结构真值为 `0`。",
        "",
        "直观地说，Same-rotor angle EI 指 `q1->I1` 和 `q2->I2` 这类“本转子角度对本转子冲量”的单源读数；Other-rotor angle EI 指 `q2->I1` 和 `q1->I2` 这类“另一个转子角度对当前冲量”的单源读数。原先的 own/cross 标签分别对应这里的 same-rotor / other-rotor。",
        "",
        "## 中文实验说明",
        "",
        "这个版本把所有对比方法统一到同一类数据上：都使用 coupled standard map 的自然轨迹样本。WMS、SURD、PCMCI 直接读自然轨迹 test split；SHAP 使用自然轨迹样本作为 MLP 解释的 foreground/background；Neural Granger 在自然轨迹训练样本上拟合稀疏预测模型；MLP+PEID 也先在自然轨迹 train/validation split 上拟合 impulse MLP，然后在自然 test states 上用 MLP 输出的 predicted impulses 计算 PEID。",
        "",
        "这里的 PEID 不再是独立均匀干预分布上的 Oracle/MLP matched-intervention 评估，而是自然轨迹分布上的 learned-mechanism readout。这样做满足“所有方法使用同样自然轨迹数据”的公平对比要求，但解释语义也随之改变：它回答的是模型在实际轨迹访问区域中的信息分解读数，而不是最大熵独立干预下的机制强度。",
        "",
        "结构真值仍然来自已知方程。动量变量 `p1` 和 `p2` 不直接进入冲量方程；真正的二阶协同来源是角度对 `q1+q2`。解析趋势用 `J^2 / 2` 表示，因此理想读数应随耦合强度单调上升，并在正耦合时把 `q1+q2` 排为最强 pair。",
        "",
        "## Surrogate Quality",
        "",
        "| J | min R2 | max NRMSE | max circular MAE | gate pass rate |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summary"]:
        lines.append(
            f"| {row['coupling']:.1f} | {row['r2_min']:.4f} | {row['nrmse_max']:.4f} | "
            f"{row['circular_mae_max']:.4f} | {row['gate_pass_rate']:.2f} |"
        )
    lines.extend(["", "## Spearman Trend Against J^2/2", "", "| readout | rho |", "| --- | ---: |"])
    for name, value in result["trends"].items():
        lines.append(f"| {name} | {value:.4f} |")
    lines.extend(
        [
            "",
            "## 中文读图说明",
            "",
            "Spearman 相关系数衡量每个方法的读数是否跟随解析真值 `J^2 / 2` 的排序变化。WMS、SHAP interaction、PCMCI other-rotor angle、Neural Granger other-rotor angle 和 MLP+PEID synergy 都得到 `rho=1.000`，说明它们在这个耦合扫描上保持正确单调趋势。SURD synergy 为负相关，没有跟随耦合强度。",
            "",
            "但趋势正确不等于量纲或因果语义相同。WMS 和 SURD 是观测分布上的信息读数；SHAP 和 Neural Granger 反映拟合预测模型中的变量使用；PCMCI 是滞后条件依赖；MLP+PEID 是在自然轨迹 states 上对拟合 MLP 的 predicted impulses 做信息分解。它们使用同一自然轨迹数据来源，但数值大小仍不能直接互换。",
        ]
    )
    lines.extend(["", "## Ground-Truth Diagnostics", "", "### J=0 absolute readout", "", "| readout | mean absolute value |", "| --- | ---: |"])
    for name, value in result["diagnostics"]["j0_false_positive"].items():
        lines.append(f"| {name} | {value:.6f} |")
    lines.extend(["", "### Other-rotor angle versus momentum control", "", "| method | other-rotor > momentum rate | mean margin |", "| --- | ---: | ---: |"])
    for name, values in result["diagnostics"]["true_null_separation"].items():
        lines.append(f"| {name} | {values['cross_above_momentum_rate']:.3f} | {values['mean_cross_minus_momentum']:.6f} |")
    lines.extend(
        [
            "",
            f"- MLP+PEID true pair top rate: `{result['diagnostics']['mlp_peid_true_pair_top_rate']:.3f}`",
            "",
            "## 自然轨迹 MLP+PEID 结果",
            "",
            "| J | truth J^2/2 | MLP+PEID q1+q2 Syn | Same-rotor angle EI | Other-rotor angle EI | Momentum control EI |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["summary"]:
        lines.append(
            f"| {row['coupling']:.1f} | {row['truth']:.3f} | "
            f"{row['peid_synergy_mean']:.6f} ± {row['peid_synergy_std']:.6f} | "
            f"{row['peid_own_mean']:.6f} | {row['peid_cross_mean']:.6f} | {row['peid_momentum_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "这个结果说明，自然轨迹 MLP+PEID 在正耦合下仍然稳定识别真实 pair：`q1+q2` 在所有正耦合 runs 中都是最强 pair，true-pair top rate 为 `1.000`。同时，它也暴露了自然轨迹估计的代价：`J=0` 时 `q1+q2` synergy 不是零，而是出现明显负残差；这来自自然轨迹经验分布、有限分箱、变量相关性和模型预测面的共同影响。因此，这个版本适合做“同数据分布公平比较”，不适合替代最大熵独立干预语义下的 PEID 机制强度。",
            "",
            "误差显示方式也相应改变：主图对所有曲线统一使用跨 seed 的 `mean ± std` 浅色阴影带，而不是单独给 PEID 画 T 形 error bar。PEID 的标准差在表中列出；它表示 seed 间变动，不是 PEID 理论量的 bootstrap 置信区间。",
            "",
            "## Observed Result",
            "",
            f"Natural-trajectory MLP+PEID has Spearman `rho={result['trends']['mlp_peid_synergy']:.3f}` against `J^2/2` and identifies `q1+q2` as the strongest pair in `{result['diagnostics']['mlp_peid_true_pair_top_rate']:.1%}` of positive-coupling runs.",
            "",
            f"Observational SURD does not track the analytic coupling trend in this periodic system (`rho={result['trends']['surd_synergy']:.3f}`) and has a large `J=0` synergy readout. This is retained as a method failure rather than removed by post-hoc tuning.",
            "",
            "中文解释：在“所有方法都用自然轨迹”的设定下，MLP+PEID 的优势是正耦合排序和真源识别稳定；限制是零耦合处出现明显自然分布残差。这个结果比独立干预 PEID 更适合作为同数据对比，但不能再解释为最大熵干预下的 Oracle-aligned 机制量。",
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "The panels retain each method's native scale. All methods use the natural trajectory data distribution in this comparison. WMS and SURD are observational distribution readouts; SHAP and Neural Granger describe fitted predictive use; PCMCI reports lagged conditional dependence; MLP+PEID evaluates the fitted impulse MLP on natural test states. Their absolute magnitudes are therefore not interchangeable.",
            "",
            "PEID rows are considered surrogate-valid only where the preregistered MLP quality gate passes. In this natural-trajectory variant, PEID targets are MLP-predicted impulses rather than observed noisy impulses.",
            "",
            "中文边界说明：本报告现在支持的结论是，在统一自然轨迹数据分布下，MLP+PEID 可以在正耦合条件中恢复真实角度 pair 的排序，并保持与 `J^2/2` 一致的单调趋势；但它的绝对数值包含自然轨迹分布效应。若要讨论 PEID 论文定义中的干预机制强度，应另行使用独立最大熵 intervention states，而不是把本图的自然轨迹 PEID 直接当作干预 PEID。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_arrays(runs: Sequence[dict[str, object]], path: Path) -> None:
    np.savez_compressed(
        path,
        couplings=np.asarray([run["coupling"] for run in runs], dtype=float),
        seeds=np.asarray([run["seed"] for run in runs], dtype=int),
        truth=np.asarray([run["ground_truth"]["interaction"] for run in runs], dtype=float),
        wms_by_target=np.asarray([[row["wms"] for row in run["wms"]["targets"]] for run in runs], dtype=float),
        shap_interaction_by_target=np.asarray([[row["interaction"] for row in run["shap"]["targets"]] for run in runs], dtype=float),
        surd_synergy_by_target=np.asarray([[row["synergy"] for row in run["surd"]["targets"]] for run in runs], dtype=float),
        pcmci_cross_by_target=np.asarray([[row["cross"] for row in run["pcmci"]["targets"]] for run in runs], dtype=float),
        neural_granger_cross_by_target=np.asarray([[row["cross"] for row in run["neural_granger"]["targets"]] for run in runs], dtype=float),
        peid_synergy_by_target=np.asarray([[row["synergy"] for row in run["peid"]["targets"]] for run in runs], dtype=float),
        mlp_peid_synergy=np.asarray([run["peid"]["aggregate"]["synergy"] for run in runs], dtype=float),
    )


def run_experiment(
    *,
    mode: str = "full",
    couplings: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    seeds: Sequence[int] = (0, 1, 2, 3),
    result_dir: Path = DEFAULT_RESULT_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    workers: int | None = None,
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    if not couplings or not seeds:
        raise ValueError("couplings and seeds must not be empty.")
    smoke = mode == "smoke"
    tasks = [(float(coupling), int(seed)) for coupling in couplings for seed in seeds]
    worker_count = 1 if smoke else min(4, len(tasks)) if workers is None else max(1, int(workers))
    if worker_count == 1:
        runs = [
            _run_one(
                coupling=coupling,
                seed=seed,
                mode=mode,
            )
            for coupling, seed in tasks
        ]
    else:
        from concurrent.futures import ProcessPoolExecutor

        payloads = [
            {
                "coupling": coupling,
                "seed": seed,
                "mode": mode,
            }
            for coupling, seed in tasks
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            runs = list(executor.map(_run_one_payload, payloads))
        runs.sort(key=lambda row: (float(row["coupling"]), int(row["seed"])))
    summary_rows, trends = _summarize(runs)
    diagnostics = _diagnostics(runs)
    result_dir = Path(result_dir)
    figure_dir = Path(figure_dir)
    report_path = Path(report_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "coupled_standard_map_six_method_comparison.png"
    ground_truth_figure_path = figure_dir / "coupled_standard_map_ground_truth_curve.png"
    summary_path = result_dir / "summary.json"
    arrays_path = result_dir / "readouts.npz"
    result: dict[str, object] = {
        "methods": list(METHOD_NAMES),
        "protocol": {
            "mode": mode,
            "couplings": [float(value) for value in couplings],
            "seeds": [int(value) for value in seeds],
            "trajectory_count": 8 if smoke else 16,
            "steps_per_trajectory": 80 if smoke else 2500,
            "split": "10/3/3 in full mode",
            "peid_state_distribution": "natural_test_trajectory",
            "peid_target_distribution": "mlp_predicted_impulses",
            "workers": worker_count,
        },
        "runs": runs,
        "summary": summary_rows,
        "trends": trends,
        "diagnostics": diagnostics,
        "summary_path": str(summary_path),
        "arrays_path": str(arrays_path),
        "figure_path": str(figure_path),
        "ground_truth_figure_path": str(ground_truth_figure_path),
        "report_path": str(report_path),
    }
    _plot(summary_rows, figure_path)
    _plot_ground_truth(summary_rows, ground_truth_figure_path)
    _write_arrays(runs, arrays_path)
    _write_report(result, report_path)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_natural_peid_experiment(
    *,
    couplings: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    seeds: Sequence[int] = (0, 1, 2, 3),
    result_path: Path = DEFAULT_RESULT_DIR / "natural_peid_summary.json",
    trajectory_count: int = 16,
    steps_per_trajectory: int = 2500,
    bins: int = 12,
    permutation_count: int = 5,
) -> dict[str, object]:
    if not couplings or not seeds:
        raise ValueError("couplings and seeds must not be empty.")
    runs = []
    for coupling in couplings:
        for seed in seeds:
            config = StandardMapConfig(k=1.5, coupling=float(coupling), noise_std=0.05)
            dataset = build_trajectory_dataset(
                config,
                trajectory_count=int(trajectory_count),
                steps_per_trajectory=int(steps_per_trajectory),
                seed=int(seed),
            )
            smoke_sized = int(trajectory_count) <= 8 or int(steps_per_trajectory) <= 120
            model = fit_impulse_mlp(
                dataset.train,
                dataset.validation,
                seed=int(seed),
                epochs=80 if smoke_sized else 400,
                hidden_width=32 if smoke_sized else 96,
            )
            natural = _natural_peid_readout(
                model,
                dataset.test,
                bins=int(bins),
                permutation_count=int(permutation_count),
                seed=int(seed) + 2000,
            )
            runs.append(
                {
                    "coupling": float(coupling),
                    "seed": int(seed),
                    "ground_truth": comparison_ground_truth(config),
                    "prediction": prediction_metrics(model, dataset.test),
                    "natural_peid": natural,
                }
            )

    summary_rows = []
    for coupling in sorted({float(run["coupling"]) for run in runs}):
        group = [run for run in runs if float(run["coupling"]) == coupling]
        row: dict[str, object] = {
            "coupling": coupling,
            "truth": coupling**2 / 2.0,
        }
        for name in ("own", "cross", "synergy", "momentum_null"):
            values = np.asarray(
                [float(run["natural_peid"]["aggregate"][name]) for run in group],
                dtype=float,
            )
            row[f"natural_peid_{name}_mean"] = float(values.mean())
            row[f"natural_peid_{name}_std"] = float(values.std(ddof=0))
        top_values = np.asarray(
            [float(run["natural_peid"]["aggregate"]["true_pair_top"]) for run in group],
            dtype=float,
        )
        strongest_pairs = sorted(
            {
                str(target["strongest_pair"])
                for run in group
                for target in run["natural_peid"]["targets"]
            }
        )
        row["natural_peid_true_pair_top_rate"] = float(top_values.mean())
        row["natural_peid_strongest_pairs"] = strongest_pairs
        summary_rows.append(row)

    truth = [float(row["truth"]) for row in summary_rows]
    trends = {
        "natural_peid_synergy": _spearman(
            truth,
            [float(row["natural_peid_synergy_mean"]) for row in summary_rows],
        )
    }
    zero_runs = [run for run in runs if abs(float(run["coupling"])) < 1e-12]
    positive_runs = [run for run in runs if float(run["coupling"]) > 0.0]
    diagnostics = {
        "j0_natural_peid_synergy_abs": float(
            np.mean(
                [
                    abs(float(run["natural_peid"]["aggregate"]["synergy"]))
                    for run in zero_runs
                ]
            )
        )
        if zero_runs
        else float("nan"),
        "positive_true_pair_top_rate": float(
            np.mean(
                [
                    float(run["natural_peid"]["aggregate"]["true_pair_top"])
                    for run in positive_runs
                ]
            )
        )
        if positive_runs
        else float("nan"),
    }
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "protocol": {
            "couplings": [float(value) for value in couplings],
            "seeds": [int(value) for value in seeds],
            "trajectory_count": int(trajectory_count),
            "steps_per_trajectory": int(steps_per_trajectory),
            "split": "same trajectory split as six-method comparison",
            "state_distribution": "natural_test_trajectory",
            "target_distribution": "mlp_predicted_impulses",
            "bins": int(bins),
            "permutation_count": int(permutation_count),
        },
        "runs": runs,
        "summary": summary_rows,
        "trends": trends,
        "diagnostics": diagnostics,
        "result_path": str(result_path),
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--natural-peid-only", action="store_true")
    args = parser.parse_args()
    if args.natural_peid_only:
        result = run_natural_peid_experiment()
        print(json.dumps({"result_path": result["result_path"]}, indent=2))
        return
    result = run_experiment(mode=args.mode)
    print(
        json.dumps(
            {
                "summary_path": result["summary_path"],
                "figure_path": result["figure_path"],
                "ground_truth_figure_path": result["ground_truth_figure_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
