#!/usr/bin/env python3
"""Fit 500->500 parcel dynamics and decompose parcel Phi only across Yeo7 boundaries."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_schaefer500_task_specific_regions import (
    DEFAULT_LABEL_FILE,
    DEFAULT_REST_ROOT,
    DEFAULT_TASK_ROOT,
    NETWORK_ORDER,
    STATE_LABELS,
    STATES,
    TASKS,
    discover_inputs,
    discover_rest_inputs,
    load_rest_series,
    load_task_pair,
    parse_schaefer_labels,
)
from scripts.phi_hierarchy import SIGNED, greedy_phi_atoms


DEFAULT_OUTPUT = ROOT / "results" / "hcp_schaefer500_parcel_phi_yeo7_hierarchy"
CONFIG_VERSION = "parcel500_target500_yeo7_boundaries_affine_tm_v1"
DISPLAY_NETWORKS = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Control", "Default")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def network_groups(labels: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    groups = {
        network: np.asarray([int(item["index"]) for item in labels if item["network"] == network], dtype=int)
        for network in NETWORK_ORDER
    }
    if sum(len(indices) for indices in groups.values()) != 500 or any(len(indices) == 0 for indices in groups.values()):
        raise ValueError("The seven Yeo networks must form a non-empty partition of all 500 parcels.")
    return groups


def load_series(path: Path, condition: str) -> np.ndarray:
    if condition == "REST":
        return load_rest_series(path)
    retained, _ = load_task_pair(path)
    return retained


def fit_transition(raw: np.ndarray, *, alpha: float) -> dict[str, Any]:
    values = np.asarray(raw, dtype=float)
    development_end = max(40, min(len(values) - 10, int(round(0.75 * len(values)))))
    development = values[:development_end]
    mean = development.mean(axis=0)
    scale = development.std(axis=0, ddof=1)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    standardized = (values - mean) / scale
    train_source = standardized[: development_end - 1]
    train_target = standardized[1:development_end]
    model = Ridge(alpha=float(alpha), fit_intercept=True).fit(train_source, train_target - train_source)
    transition = np.eye(500) + np.asarray(model.coef_, dtype=float)
    train_prediction = train_source + model.predict(train_source)
    residual = train_target - train_prediction
    noise = np.asarray(LedoitWolf(assume_centered=False).fit(residual).covariance_, dtype=float)
    noise += 1.0e-8 * np.eye(500)

    holdout_source = standardized[development_end - 1 : -1]
    holdout_target = standardized[development_end:]
    if len(holdout_source):
        holdout_prediction = holdout_source + model.predict(holdout_source)
        holdout_rmse = float(np.sqrt(np.mean((holdout_target - holdout_prediction) ** 2)))
        persistence_rmse = float(np.sqrt(np.mean((holdout_target - holdout_source) ** 2)))
    else:
        holdout_rmse = float("nan")
        persistence_rmse = float("nan")
    return {
        "transition": transition,
        "noise": noise,
        "development_end": development_end,
        "train_rmse": float(np.sqrt(np.mean(residual**2))),
        "holdout_rmse": holdout_rmse,
        "persistence_rmse": persistence_rmse,
        "skill_ratio": float(holdout_rmse / max(persistence_rmse, 1.0e-12)),
    }


def logdet_spd(matrix: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(np.asarray(matrix, dtype=float))
    if sign <= 0.0 or not np.isfinite(value):
        raise ValueError("Expected a positive-definite covariance matrix.")
    return float(value)


def decompose_transition(
    transition: np.ndarray,
    noise: np.ndarray,
    groups: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    names = tuple(NETWORK_ORDER)
    full_covariance = transition @ transition.T + noise
    full_logdet = logdet_spd(full_covariance)
    table: dict[tuple[str, ...], float] = {}
    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(names, size):
            selected = set(subset)
            complement = np.concatenate([groups[name] for name in names if name not in selected]) if size < len(names) else np.empty(0, dtype=int)
            conditional = noise.copy()
            if len(complement):
                conditional += transition[:, complement] @ transition[:, complement].T
            value = 0.5 * (full_logdet - logdet_spd(conditional)) / math.log(2.0)
            table[subset] = float(value)

    solved = np.linalg.solve(full_covariance, transition)
    leverage = np.sum(transition * solved, axis=0)
    leverage = np.clip(leverage, 0.0, 1.0 - 1.0e-12)
    parcel_singleton_ei = -0.5 * np.log1p(-leverage) / math.log(2.0)
    network_ei = {name: float(table[(name,)]) for name in names}
    within_atoms = {
        name: float(network_ei[name] - parcel_singleton_ei[groups[name]].sum()) for name in names
    }
    cross_phi = float(table[names] - sum(network_ei.values()))
    total_phi = float(table[names] - parcel_singleton_ei.sum())
    cross_atoms = greedy_phi_atoms(
        names,
        table,
        policy=SIGNED,
        singleton_ei=network_ei,
        eps=1.0e-8,
    )
    atom_rows = [
        {"sources": [name], "value": float(within_atoms[name]), "kind": "within_network", "depth": -1}
        for name in names
    ]
    atom_rows.extend(
        {
            "sources": list(atom.sources),
            "value": float(atom.value),
            "kind": str(atom.kind),
            "depth": int(atom.depth),
        }
        for atom in cross_atoms
    )

    network_contribution = dict(within_atoms)
    for atom in cross_atoms:
        share = float(atom.value) / len(atom.sources)
        for name in atom.sources:
            network_contribution[name] += share
    parcel_contribution = np.zeros(500, dtype=float)
    for name in names:
        parcel_contribution[groups[name]] = network_contribution[name] / len(groups[name])
    return {
        "whole_ei": float(table[names]),
        "parcel_singleton_ei_sum": float(parcel_singleton_ei.sum()),
        "total_phi": total_phi,
        "cross_network_phi": cross_phi,
        "network_ei": network_ei,
        "within_network_atoms": within_atoms,
        "atoms": atom_rows,
        "network_contribution": network_contribution,
        "parcel_contribution": parcel_contribution.tolist(),
        "atom_sum_error": float(sum(row["value"] for row in atom_rows) - total_phi),
        "network_contribution_sum_error": float(sum(network_contribution.values()) - total_phi),
        "parcel_contribution_sum_error": float(parcel_contribution.sum() - total_phi),
    }


def analyze_job(
    path: Path,
    *,
    subject: str,
    condition: str,
    alpha: float,
    groups: Mapping[str, np.ndarray],
    config_id: str,
) -> dict[str, Any]:
    raw = load_series(path, condition)
    fitted = fit_transition(raw, alpha=alpha)
    decomposition = decompose_transition(fitted.pop("transition"), fitted.pop("noise"), groups)
    return {
        "config_id": config_id,
        "subject": subject,
        "condition": condition,
        "n_timepoints": int(len(raw)),
        "model": fitted,
        **decomposition,
    }


def load_cache(path: Path, config_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    cached: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return cached
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("config_id") == config_id:
                cached[(str(row["subject"]), str(row["condition"]))] = row
    return cached


def append_cache(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def bootstrap_ci(values: np.ndarray, *, seed: int, repeats: int = 5000) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(repeats, len(array)))
    return [float(value) for value in np.quantile(array[indices].mean(axis=1), [0.025, 0.975])]


def loso_nearest_centroid(values: np.ndarray) -> tuple[float, np.ndarray]:
    n_states, n_subjects, _ = values.shape
    if n_subjects < 2:
        return float("nan"), np.zeros((n_states, n_states), dtype=int)
    confusion = np.zeros((n_states, n_states), dtype=int)
    for held_out in range(n_subjects):
        train = np.delete(values, held_out, axis=1)
        pooled = train.reshape(-1, train.shape[-1])
        mean = pooled.mean(axis=0)
        scale = pooled.std(axis=0, ddof=1)
        scale = np.where(scale > 1.0e-12, scale, 1.0)
        centroids = ((train - mean) / scale).mean(axis=1)
        test = (values[:, held_out] - mean) / scale
        predicted = np.argmin(np.linalg.norm(test[:, None, :] - centroids[None, :, :], axis=2), axis=1)
        for truth, guess in enumerate(predicted):
            confusion[truth, int(guess)] += 1
    return float(np.trace(confusion) / confusion.sum()), confusion


def permutation_accuracy_p(
    values: np.ndarray,
    observed: float,
    *,
    repeats: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(int(repeats)):
        permuted = np.empty_like(values)
        for subject in range(values.shape[1]):
            permuted[:, subject] = values[rng.permutation(values.shape[0]), subject]
        accuracy, _ = loso_nearest_centroid(permuted)
        exceed += int(accuracy >= observed)
    return float((1 + exceed) / (int(repeats) + 1))


def summarize(
    records: Mapping[tuple[str, str], Mapping[str, Any]],
    subjects: Sequence[str],
    groups: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    network = np.empty((len(STATES), len(subjects), len(NETWORK_ORDER)), dtype=float)
    parcel = np.empty((len(STATES), len(subjects), 500), dtype=float)
    phi = np.empty((len(STATES), len(subjects)), dtype=float)
    skill = np.empty_like(phi)
    for state_index, condition in enumerate(STATES):
        for subject_index, subject in enumerate(subjects):
            row = records[(subject, condition)]
            network[state_index, subject_index] = [row["network_contribution"][name] for name in NETWORK_ORDER]
            parcel[state_index, subject_index] = row["parcel_contribution"]
            phi[state_index, subject_index] = row["total_phi"]
            skill[state_index, subject_index] = row["model"]["skill_ratio"]
    if np.any(np.abs(phi) <= 1.0e-10):
        raise ValueError("Cannot run-normalize a state with near-zero total Phi.")
    network_share = network / phi[:, :, None]
    parcel_enrichment = 500.0 * parcel / phi[:, :, None]
    network_specificity = network_share - (
        network_share.sum(axis=0, keepdims=True) - network_share
    ) / (len(STATES) - 1)
    parcel_specificity = parcel_enrichment - (
        parcel_enrichment.sum(axis=0, keepdims=True) - parcel_enrichment
    ) / (len(STATES) - 1)
    raw_accuracy, raw_confusion = loso_nearest_centroid(network)
    share_accuracy, share_confusion = loso_nearest_centroid(network_share)
    task_raw_accuracy, _ = loso_nearest_centroid(network[1:])
    task_share_accuracy, _ = loso_nearest_centroid(network_share[1:])
    permutation_repeats = 2000
    share_accuracy_p = permutation_accuracy_p(
        network_share, share_accuracy, repeats=permutation_repeats, seed=2026071901
    )
    task_share_accuracy_p = permutation_accuracy_p(
        network_share[1:], task_share_accuracy, repeats=permutation_repeats, seed=2026071902
    )
    state_summary = {}
    for index, condition in enumerate(STATES):
        state_summary[condition] = {
            "phi_mean_bits": float(phi[index].mean()),
            "phi_bootstrap_95_ci": bootstrap_ci(phi[index], seed=5100 + index),
            "skill_ratio_mean": float(skill[index].mean()),
            "skill_ratio_below_one": int(np.sum(skill[index] < 1.0)),
            "network_share_mean": {
                name: float(network_share[index, :, j].mean()) for j, name in enumerate(NETWORK_ORDER)
            },
        }
    errors = [
        max(abs(float(row["atom_sum_error"])), abs(float(row["network_contribution_sum_error"])), abs(float(row["parcel_contribution_sum_error"])))
        for row in records.values()
    ]
    summary = {
        "n_subjects": len(subjects),
        "n_states": len(STATES),
        "state_summary": state_summary,
        "diagnostics": {
            "maximum_closure_error_bits": float(max(errors)),
            "nonpositive_phi_models": int(np.sum(phi <= 0.0)),
            "mean_skill_ratio": float(skill.mean()),
            "skill_ratio_below_one": int(np.sum(skill < 1.0)),
            "n_models": int(skill.size),
        },
        "discriminability": {
            "chance_eight_states": 1.0 / len(STATES),
            "raw_network_contribution_loso_accuracy": raw_accuracy,
            "run_normalized_network_share_loso_accuracy": share_accuracy,
            "chance_seven_tasks": 1.0 / len(TASKS),
            "task_only_raw_accuracy": task_raw_accuracy,
            "task_only_run_normalized_accuracy": task_share_accuracy,
            "permutation_repeats": permutation_repeats,
            "run_normalized_accuracy_empirical_p": share_accuracy_p,
            "task_only_run_normalized_accuracy_empirical_p": task_share_accuracy_p,
            "run_normalized_confusion": share_confusion.tolist(),
            "raw_confusion": raw_confusion.tolist(),
        },
        "network_sizes": {name: int(len(groups[name])) for name in NETWORK_ORDER},
    }
    arrays = {
        "network_contribution": network,
        "network_share": network_share,
        "network_specificity": network_specificity,
        "parcel_contribution": parcel,
        "parcel_enrichment": parcel_enrichment,
        "parcel_specificity": parcel_specificity,
        "phi": phi,
        "skill_ratio": skill,
    }
    return summary, arrays


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: Any, path: Path) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_results(
    summary: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    groups: Mapping[str, np.ndarray],
    labels: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    configure_style()
    parcel_specificity = arrays["parcel_specificity"].mean(axis=1)
    network_specificity = arrays["network_specificity"].mean(axis=1).T * 100.0
    phi = arrays["phi"]
    confusion = np.asarray(summary["discriminability"]["run_normalized_confusion"], dtype=float)
    confusion /= confusion.sum(axis=1, keepdims=True)
    parcel_limit = float(np.quantile(np.abs(parcel_specificity), 0.995))
    network_limit = float(np.quantile(np.abs(network_specificity), 0.995))

    figure = plt.figure(figsize=(12.2, 7.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(2.15, 1.0), height_ratios=(1.0, 1.0))
    axes = [figure.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    image = axes[0].imshow(parcel_specificity, aspect="auto", cmap="RdBu_r", vmin=-parcel_limit, vmax=parcel_limit)
    axes[0].set(yticks=np.arange(len(STATES)), yticklabels=STATE_LABELS, xlabel="Schaefer-500 parcel (atlas order)", ylabel="State")
    segments: list[tuple[int, int, str]] = []
    segment_start = 0
    segment_key = (str(labels[0]["hemisphere"]), str(labels[0]["network"]))
    for index, item in enumerate(labels[1:], start=1):
        key = (str(item["hemisphere"]), str(item["network"]))
        if key != segment_key:
            segments.append((segment_start, index - 1, f"{segment_key[0]} {DISPLAY_NETWORKS[NETWORK_ORDER.index(segment_key[1])] }"))
            segment_start = index
            segment_key = key
    segments.append((segment_start, len(labels) - 1, f"{segment_key[0]} {DISPLAY_NETWORKS[NETWORK_ORDER.index(segment_key[1])] }"))
    for start, end, label in segments:
        axes[0].axvline(float(end) + 0.5, color="#333333", linewidth=0.35)
        axes[0].text(0.5 * (start + end), -0.85, label, ha="center", va="bottom", fontsize=4.7, rotation=32)
    colorbar = figure.colorbar(image, ax=axes[0], shrink=0.82, pad=0.012)
    colorbar.set_label("State-specific run-normalized hierarchy contribution")

    image = axes[1].imshow(network_specificity, aspect="auto", cmap="RdBu_r", vmin=-network_limit, vmax=network_limit)
    axes[1].set(
        xticks=np.arange(len(STATES)), xticklabels=STATE_LABELS,
        yticks=np.arange(len(NETWORK_ORDER)), yticklabels=DISPLAY_NETWORKS,
        xlabel="State", ylabel="Yeo7 network",
    )
    axes[1].tick_params(axis="x", labelrotation=40, length=0)
    axes[1].tick_params(axis="y", length=0)
    for row in range(network_specificity.shape[0]):
        for column in range(network_specificity.shape[1]):
            value = network_specificity[row, column]
            axes[1].text(column, row, f"{value:+.1f}", ha="center", va="center", fontsize=5.0, color="white" if abs(value) > 0.58 * network_limit else "black")
    colorbar = figure.colorbar(image, ax=axes[1], shrink=0.82, pad=0.02)
    colorbar.set_label("State minus other states (percentage points)")

    positions = np.arange(len(STATES))
    means = phi.mean(axis=1)
    intervals = np.asarray([summary["state_summary"][condition]["phi_bootstrap_95_ci"] for condition in STATES])
    axes[2].errorbar(
        positions, means, yerr=np.vstack([means - intervals[:, 0], intervals[:, 1] - means]),
        fmt="o", color="#4C78A8", ecolor="#9CB7CF", capsize=2, markersize=4,
    )
    axes[2].set(xticks=positions, xticklabels=STATE_LABELS, ylabel=r"Parcel-level $\Phi^{EID}$ (bits)", xlabel="State")
    axes[2].tick_params(axis="x", labelrotation=35)

    image = axes[3].imshow(confusion, cmap="Blues", vmin=0.0, vmax=1.0)
    axes[3].set(
        xticks=np.arange(len(STATES)), xticklabels=STATE_LABELS,
        yticks=np.arange(len(STATES)), yticklabels=STATE_LABELS,
        xlabel="Predicted state", ylabel="True state",
    )
    axes[3].tick_params(axis="x", labelrotation=40, length=0)
    axes[3].tick_params(axis="y", length=0)
    for row in range(len(STATES)):
        for column in range(len(STATES)):
            value = confusion[row, column]
            axes[3].text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=4.5, color="white" if value > 0.55 else "black")
    colorbar = figure.colorbar(image, ax=axes[3], shrink=0.82, pad=0.02)
    colorbar.set_label("Fraction of subjects")
    for label, axis in zip("abcd", axes):
        axis.text(-0.10, 1.05, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    save_figure(figure, output_dir / "parcel500_yeo7_hierarchy_task_differences")


def write_report(summary: Mapping[str, Any], output_dir: Path) -> None:
    disc = summary["discriminability"]
    lines = [
        "# Schaefer-500 source/target 的 Yeo7 边界层级 Phi 分解",
        "",
        "原变量和下一时刻 target 均保留为 500 个 Schaefer parcel；层级搜索只允许在七个 Yeo 网络边界之间发生。同一网络内部不继续选择 parcel 子集，而将网络 EI 减去网络内 parcel 单源 EI 之和作为该网络的终端 atom。",
        "",
        "## 任务态可分性",
        "",
        f"- 八状态 raw network contribution LOSO 准确率：{100*disc['raw_network_contribution_loso_accuracy']:.2f}%（chance {100*disc['chance_eight_states']:.2f}%）。",
        f"- 八状态 run-normalized share LOSO 准确率：{100*disc['run_normalized_network_share_loso_accuracy']:.2f}% 。",
        f"- 八状态配对标签置换检验：p={disc['run_normalized_accuracy_empirical_p']:.6f}（{disc['permutation_repeats']}次）。",
        f"- 仅七任务 raw / normalized：{100*disc['task_only_raw_accuracy']:.2f}% / {100*disc['task_only_run_normalized_accuracy']:.2f}%（chance {100*disc['chance_seven_tasks']:.2f}%）；置换 p={disc['task_only_run_normalized_accuracy_empirical_p']:.6f}。",
        "",
        "## 状态特异的网络层级份额",
        "",
        "下表先在每个被试–状态内将七网络层级贡献除以该 run 的总 Phi，再计算该状态减其余七状态的配对组均值。正值表示该网络在该状态的 Phi 构成中相对富集。",
        "",
        "| State | Largest enrichment | Largest depletion |",
        "|---|---:|---:|",
    ]
    state_items = summary["state_summary"]
    for condition in STATES:
        differences = {}
        for network in NETWORK_ORDER:
            own = state_items[condition]["network_share_mean"][network]
            others = np.mean(
                [state_items[other]["network_share_mean"][network] for other in STATES if other != condition]
            )
            differences[network] = 100.0 * (own - others)
        positive = max(differences, key=differences.get)
        negative = min(differences, key=differences.get)
        lines.append(
            f"| {condition} | {positive} {differences[positive]:+.2f} pp | {negative} {differences[negative]:+.2f} pp |"
        )
    lines.extend(
        [
        "",
        "## 状态汇总",
        "",
        "| State | Mean Phi (bits) | 95% bootstrap CI | Mean skill ratio | skill < persistence |",
        "|---|---:|---:|---:|---:|",
        ]
    )
    for condition in STATES:
        item = summary["state_summary"][condition]
        lines.append(
            f"| {condition} | {item['phi_mean_bits']:.4f} | [{item['phi_bootstrap_95_ci'][0]:.4f}, {item['phi_bootstrap_95_ci'][1]:.4f}] | "
            f"{item['skill_ratio_mean']:.4f} | {item['skill_ratio_below_one']}/{summary['n_subjects']} |"
        )
    diagnostics = summary["diagnostics"]
    lines.extend(
        [
            "",
            "## 诊断与边界",
            "",
            f"最大层级闭合误差为 {diagnostics['maximum_closure_error_bits']:.3e} bits；非正 Phi 模型 {diagnostics['nonpositive_phi_models']}/{diagnostics['n_models']}；"
            f"预测优于 persistence 的模型为 {diagnostics['skill_ratio_below_one']}/{diagnostics['n_models']}。",
            "",
            "图中500列保留 atlas 顺序，但由于网络内部不再划分，同一 Yeo 网络中的 parcel 获得相同的每-parcel层级份额；因此该图识别的是网络边界层级差异，不是网络内部 parcel 定位。任务和 REST 的时间长度不同，跨状态比较仍可能包含样本长度差异。",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    task_subjects, task_paths = discover_inputs(args.task_root)
    rest_paths = discover_rest_inputs(args.rest_root)
    subjects = [subject for subject in task_subjects if subject in rest_paths]
    if args.max_subjects is not None:
        subjects = subjects[: int(args.max_subjects)]
    labels = parse_schaefer_labels(args.label_file)
    groups = network_groups(labels)
    config_id = f"{CONFIG_VERSION}|alpha={float(args.alpha):.12g}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "question": "What changes across HCP states when source/target stay at 500 parcels and hierarchy splits are restricted to Yeo7 boundaries?",
        "treatment_levels": list(STATES),
        "paired_unit": "subject",
        "source": "current standardized Schaefer-500 parcel state",
        "target": "next standardized Schaefer-500 parcel state",
        "estimator": "affine TM / Gaussian log-det EI from a one-step delta Ridge surrogate",
        "alpha": float(args.alpha),
        "hierarchy": "all 127 Yeo7 coalitions cached; signed greedy hierarchy; no within-network parcel split",
        "primary_metric": "run-normalized hierarchy contribution map and LOSO state accuracy",
        "known_confounds": ["state-specific run length", "state-specific surrogate fit", "LR runs only"],
    }
    atomic_json(output_dir / "experiment_contract.json", contract)
    cache_path = output_dir / "records.jsonl"
    records = load_cache(cache_path, config_id)
    jobs = []
    for subject in subjects:
        jobs.append((subject, "REST", rest_paths[subject]))
        jobs.extend((subject, task, task_paths[subject][task]) for task in TASKS)
    pending = [job for job in jobs if (job[0], job[1]) not in records]
    status_path = output_dir / "live_progress.json"
    bar = tqdm(pending, desc="Parcel500 Yeo7-boundary Phi", unit="model", mininterval=1.0)
    try:
        for completed, (subject, condition, path) in enumerate(bar, start=1):
            row = analyze_job(
                Path(path), subject=subject, condition=condition, alpha=float(args.alpha), groups=groups, config_id=config_id
            )
            records[(subject, condition)] = row
            append_cache(cache_path, row)
            elapsed = time.monotonic() - started
            done = len(jobs) - len(pending) + completed
            rate = done / elapsed if elapsed > 0.0 else 0.0
            atomic_json(
                status_path,
                {
                    "phase": "compute",
                    "current": done,
                    "total": len(jobs),
                    "unit": "model",
                    "elapsed_seconds": elapsed,
                    "eta_seconds": (len(jobs) - done) / rate if rate > 0.0 else None,
                    "metrics": {
                        "subject": subject,
                        "state": condition,
                        "phi_bits": row["total_phi"],
                        "skill_ratio": row["model"]["skill_ratio"],
                    },
                    "updated_at": time.time(),
                },
            )
            bar.set_postfix(state=condition, phi=f"{row['total_phi']:.2f}", skill=f"{row['model']['skill_ratio']:.2f}")
    except Exception as exc:
        atomic_json(status_path, {"phase": "failed", "message": f"{type(exc).__name__}: {exc}", "updated_at": time.time()})
        raise
    summary, arrays = summarize(records, subjects, groups)
    summary["config"] = {**contract, "config_id": config_id, "subjects": subjects}
    atomic_json(output_dir / "summary.json", summary)
    np.savez_compressed(
        output_dir / "arrays.npz",
        states=np.asarray(STATES),
        subjects=np.asarray(subjects),
        networks=np.asarray(NETWORK_ORDER),
        **arrays,
    )
    plot_results(summary, arrays, groups, labels, output_dir)
    write_report(summary, output_dir)
    atomic_json(
        status_path,
        {
            "phase": "complete",
            "current": len(jobs),
            "total": len(jobs),
            "unit": "model",
            "elapsed_seconds": time.monotonic() - started,
            "metrics": summary["discriminability"],
            "updated_at": time.time(),
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--rest-root", type=Path, default=DEFAULT_REST_ROOT)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args(argv)
    summary = run(args)
    print(json.dumps({"output_dir": str(args.output_dir), "diagnostics": summary["diagnostics"], "discriminability": summary["discriminability"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
