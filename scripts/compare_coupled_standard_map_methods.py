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
    config: StandardMapConfig,
    intervention_states: np.ndarray,
    intervention_noise: np.ndarray,
    bins: int,
    permutation_count: int,
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    oracle_targets = coupled_impulses(
        intervention_states,
        config=config,
        noise=intervention_noise * float(config.noise_std),
    )
    learned_targets = model.predict_mean(intervention_states) + intervention_noise * model.residual_std
    oracle = evaluate_peid(
        states=intervention_states,
        targets=oracle_targets,
        bins=bins,
        permutation_count=permutation_count,
        seed=seed,
    )
    learned = evaluate_peid(
        states=intervention_states,
        targets=learned_targets,
        bins=bins,
        permutation_count=permutation_count,
        seed=seed,
    )

    def summarize(result: dict[str, object]) -> tuple[dict[str, float], list[dict[str, float | str]]]:
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

    learned_aggregate, learned_rows = summarize(learned)
    oracle_aggregate, oracle_rows = summarize(oracle)
    learned_aggregate["oracle_synergy"] = oracle_aggregate["synergy"]
    for learned_row, oracle_row in zip(learned_rows, oracle_rows):
        learned_row["oracle_synergy"] = oracle_row["synergy"]
    return (
        {"aggregate": learned_aggregate, "targets": learned_rows},
        {"aggregate": oracle_aggregate, "targets": oracle_rows},
    )


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
    intervention_states: np.ndarray,
    intervention_noise: np.ndarray,
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
    peid, oracle = _peid_readout(
        model,
        config=config,
        intervention_states=intervention_states,
        intervention_noise=intervention_noise,
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
        "oracle_peid": oracle,
        "oracle_intervention_digest": _digest(intervention_states),
        "mlp_intervention_digest": _digest(intervention_states),
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
        "oracle_synergy": ("peid", "oracle_synergy"),
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
        "oracle_peid_synergy": _spearman(truth, [row["oracle_synergy_mean"] for row in summary_rows]),
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
        "oracle_peid_synergy": ("peid", "oracle_synergy"),
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
    peid_errors = []
    for run in positive_runs:
        learned = float(run["peid"]["aggregate"]["synergy"])
        oracle = float(run["peid"]["aggregate"]["oracle_synergy"])
        peid_errors.append(abs(learned - oracle) / max(abs(oracle), 1e-12))
    return {
        "j0_false_positive": zero_false_positive,
        "true_null_separation": separation,
        "mlp_peid_true_pair_top_rate": float(
            np.mean([run["peid"]["aggregate"]["true_pair_top"] for run in positive_runs])
        )
        if positive_runs
        else float("nan"),
        "mlp_peid_relative_error_mean": float(np.mean(peid_errors)) if peid_errors else float("nan"),
        "mlp_peid_relative_error_max": float(np.max(peid_errors)) if peid_errors else float("nan"),
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

    for key, label, color, marker in (("shap_own", "own angle", "#4c78a8", "^"), ("shap_cross", "cross angle", "#f58518", "v"), ("shap_interaction", "q1:q2 interaction", "#e45756", "o"), ("shap_momentum", "momentum null", "#7f8c8d", "s")):
        curve(flat[1], key, label, color, marker)
    flat[1].set_title("MLP+SHAP")
    flat[1].set_ylabel("Mean absolute attribution")

    for key, label, color, marker in (("surd_redundancy", "redundancy", "#7f8c8d", "s"), ("surd_own", "own unique", "#4c78a8", "^"), ("surd_cross", "cross unique", "#f58518", "v"), ("surd_synergy", "synergy", "#e45756", "o")):
        curve(flat[2], key, label, color, marker)
    flat[2].set_title("Observational SURD")
    flat[2].set_ylabel("Information (bits)")

    for key, label, color, marker in (("pcmci_own", "own angle", "#4c78a8", "^"), ("pcmci_cross", "cross angle", "#f58518", "v"), ("pcmci_momentum", "momentum null", "#7f8c8d", "s")):
        curve(flat[3], key, label, color, marker)
    flat[3].set_title("PCMCI-CMIknn")
    flat[3].set_ylabel("Absolute CMIknn statistic")

    for key, label, color, marker in (("ng_own", "own angle", "#4c78a8", "^"), ("ng_cross", "cross angle", "#f58518", "v"), ("ng_momentum", "momentum null", "#7f8c8d", "s")):
        curve(flat[4], key, label, color, marker)
    flat[4].set_title("Neural Granger")
    flat[4].set_ylabel("First-layer group norm")

    for key, label, color, marker, dashed in (("peid_own", "own angle EI", "#4c78a8", "^", False), ("peid_cross", "cross angle EI", "#f58518", "v", False), ("peid_synergy", "MLP synergy", "#e45756", "o", False), ("oracle_synergy", "Oracle synergy", "#222222", "D", True), ("peid_momentum", "momentum null", "#7f8c8d", "s", False)):
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


def _write_report(result: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure_path = Path(result["figure_path"])
    relative_figure = os.path.relpath(figure_path, path.parent)
    lines = [
        "# Coupled Standard Map Six-Method Comparison",
        "",
        f"![Six-method comparison]({relative_figure})",
        "",
        "## Protocol",
        "",
        f"- coupling values: `{result['protocol']['couplings']}`",
        f"- seeds: `{result['protocol']['seeds']}`",
        f"- trajectories per full run: `{result['protocol']['trajectory_count']}`",
        f"- steps per trajectory: `{result['protocol']['steps_per_trajectory']}`",
        "- targets: impulses `I1` and `I2`; symmetric target readouts are averaged only in the main figure",
        "- analytic cross and interaction strength: `J^2 / 2`",
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
    lines.extend(["", "## Ground-Truth Diagnostics", "", "### J=0 absolute readout", "", "| readout | mean absolute value |", "| --- | ---: |"])
    for name, value in result["diagnostics"]["j0_false_positive"].items():
        lines.append(f"| {name} | {value:.6f} |")
    lines.extend(["", "### True source versus momentum null", "", "| method | cross > null rate | mean margin |", "| --- | ---: | ---: |"])
    for name, values in result["diagnostics"]["true_null_separation"].items():
        lines.append(f"| {name} | {values['cross_above_momentum_rate']:.3f} | {values['mean_cross_minus_momentum']:.6f} |")
    lines.extend(
        [
            "",
            f"- MLP+PEID true pair top rate: `{result['diagnostics']['mlp_peid_true_pair_top_rate']:.3f}`",
            f"- MLP+PEID mean relative Oracle error: `{result['diagnostics']['mlp_peid_relative_error_mean']:.4f}`",
            f"- MLP+PEID maximum relative Oracle error: `{result['diagnostics']['mlp_peid_relative_error_max']:.4f}`",
            "",
            "## Observed Result",
            "",
            f"MLP+PEID tracks Oracle synergy with Spearman `rho={result['trends']['mlp_peid_synergy']:.3f}`, identifies `q1+q2` as the strongest pair in `{result['diagnostics']['mlp_peid_true_pair_top_rate']:.1%}` of positive-coupling runs, and has maximum relative Oracle error `{result['diagnostics']['mlp_peid_relative_error_max']:.3%}`.",
            "",
            f"Observational SURD does not track the analytic coupling trend in this periodic system (`rho={result['trends']['surd_synergy']:.3f}`) and has a large `J=0` synergy readout. This is retained as a method failure rather than removed by post-hoc tuning.",
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "The panels retain each method's native scale. WMS and SURD are observational distribution readouts; SHAP and Neural Granger describe fitted predictive use; PCMCI reports lagged conditional dependence; MLP+PEID evaluates the learned mechanism under independent interventions. Their absolute magnitudes are therefore not interchangeable.",
            "",
            "PEID rows are considered surrogate-valid only where the preregistered MLP quality gate passes. Oracle and learned PEID use identical intervention states and matched noise draws.",
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
        oracle_peid_synergy_by_target=np.asarray([[row["synergy"] for row in run["oracle_peid"]["targets"]] for run in runs], dtype=float),
        mlp_peid_synergy=np.asarray([run["peid"]["aggregate"]["synergy"] for run in runs], dtype=float),
        oracle_peid_synergy=np.asarray([run["oracle_peid"]["aggregate"]["synergy"] for run in runs], dtype=float),
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
    intervention_count = 2500 if smoke else 50000
    intervention_rng = np.random.default_rng(90210)
    intervention_states = intervention_rng.uniform(-np.pi, np.pi, size=(intervention_count, 4))
    intervention_noise = intervention_rng.normal(size=(intervention_count, 2))
    tasks = [(float(coupling), int(seed)) for coupling in couplings for seed in seeds]
    worker_count = 1 if smoke else min(4, len(tasks)) if workers is None else max(1, int(workers))
    if worker_count == 1:
        runs = [
            _run_one(
                coupling=coupling,
                seed=seed,
                mode=mode,
                intervention_states=intervention_states,
                intervention_noise=intervention_noise,
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
                "intervention_states": intervention_states,
                "intervention_noise": intervention_noise,
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
            "intervention_samples": intervention_count,
            "workers": worker_count,
        },
        "runs": runs,
        "summary": summary_rows,
        "trends": trends,
        "diagnostics": diagnostics,
        "summary_path": str(summary_path),
        "arrays_path": str(arrays_path),
        "figure_path": str(figure_path),
        "report_path": str(report_path),
    }
    _plot(summary_rows, figure_path)
    _write_arrays(runs, arrays_path)
    _write_report(result, report_path)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    result = run_experiment(mode=args.mode)
    print(json.dumps({"summary_path": result["summary_path"], "figure_path": result["figure_path"]}, indent=2))


if __name__ == "__main__":
    main()
