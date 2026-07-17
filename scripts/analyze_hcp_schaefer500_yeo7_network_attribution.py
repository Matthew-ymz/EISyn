#!/usr/bin/env python3
"""Attribute Yeo7 EI/Phi and summarize synergy cores across REST, MOTOR, and WM."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Polygon
from scipy.io import loadmat
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hcp_lausanne_phi_eid_pilot import ei_for_source_indices
from scripts.run_hcp_schaefer500_all_tasks_phi import development_end_for_length
from scripts.run_hcp_schaefer500_wm_yeo7_phi import load_task_series
from scripts.run_hcp_schaefer500_yeo7_module_phi_decomposition import (
    _block_phi,
    decompose_modules,
    network_history_indices,
)
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import (
    default_yeo7_labels,
    fit_yeo7_pc1,
    load_yeo7_groups,
)
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi


DEFAULT_REST_ROOT = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30"
DEFAULT_TASK_ROOT = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_30"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "hcp_schaefer500_yeo7_network_attribution"
NETWORK_ORDER = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
NETWORK_LABELS = {
    "Vis": "Vis",
    "SomMot": "SomMot",
    "DorsAttn": "DorsAttn",
    "SalVentAttn": "SalVentAttn",
    "Limbic": "Limbic",
    "Cont": "Control",
    "Default": "Default",
}
CONDITIONS = ("REST", "MOTOR", "WM")
CONDITION_LABELS = {"REST": "REST", "MOTOR": "Motor", "WM": "WM"}
CONDITION_COLORS = {"REST": "#4C78A8", "MOTOR": "#E3A857", "WM": "#D07A3A"}


def discover_inputs(rest_root: Path, task_root: Path) -> dict[str, dict[str, Path]]:
    rest = {
        path.parent.name: path
        for path in Path(rest_root).glob("sub-*/*REST1_LR*schaefer500-1000_yeo7.mat")
    }
    motor = {path.parent.name: path for path in Path(task_root).glob("sub-*/MOTOR_LR.mat")}
    wm = {path.parent.name: path for path in Path(task_root).glob("sub-*/WM_LR.mat")}
    common = sorted(set(rest) & set(motor) & set(wm))
    return {
        subject: {"REST": rest[subject], "MOTOR": motor[subject], "WM": wm[subject]}
        for subject in common
    }


def load_rest_series(path: Path, *, data_key: str = "Schaefer500") -> np.ndarray:
    payload = loadmat(path)
    if data_key not in payload:
        raise ValueError(f"MAT file {path} does not contain {data_key!r}.")
    values = np.asarray(payload[data_key], dtype=float)
    if values.ndim != 2 or values.shape[1] != 500 or not np.isfinite(values).all():
        raise ValueError(f"Expected finite [time, 500] REST data in {path}, got {values.shape}.")
    return values


def exact_shapley(
    names: Sequence[str],
    value: Callable[[tuple[str, ...]], float],
) -> dict[str, float]:
    ordered = tuple(names)
    n_players = len(ordered)
    if n_players < 1:
        return {}
    denominator = math.factorial(n_players)
    result: dict[str, float] = {}
    for player in ordered:
        others = tuple(name for name in ordered if name != player)
        contribution = 0.0
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(n_players - size - 1) / denominator
            for subset in itertools.combinations(others, size):
                with_player = tuple(name for name in ordered if name == player or name in subset)
                contribution += weight * (float(value(with_player)) - float(value(tuple(subset))))
        result[player] = float(contribution)
    return result


def transition_attribution(
    transition: np.ndarray,
    noise_covariance: np.ndarray,
    network_names: Sequence[str],
    *,
    order: int,
    ridge: float = 1.0e-6,
) -> dict[str, Any]:
    names = tuple(network_names)
    table, atoms = decompose_modules(transition, noise_covariance, names, order=order)
    indices = network_history_indices(names, order=order)
    scalar_ei = {
        name: float(
            sum(
                ei_for_source_indices(transition, noise_covariance, [index], ridge=ridge)
                for index in indices[name]
            )
        )
        for name in names
    }
    module_ei = {name: float(table[(name,)]) for name in names}
    within_phi = {name: float(module_ei[name] - scalar_ei[name]) for name in names}

    def cross_value(subset: tuple[str, ...]) -> float:
        if not subset:
            return 0.0
        ordered_subset = tuple(name for name in names if name in subset)
        return float(table[ordered_subset] - sum(module_ei[name] for name in ordered_subset))

    cross_shapley = exact_shapley(names, cross_value)
    total_contribution = {
        name: float(within_phi[name] + cross_shapley[name]) for name in names
    }
    full = tuple(names)
    joint_ei = float(table[full])
    scalar_ei_sum = float(sum(scalar_ei.values()))
    raw_phi = float(joint_ei - scalar_ei_sum)
    cross_phi = float(_block_phi(full, table))
    within_phi_sum = float(sum(within_phi.values()))
    ranked_atoms = sorted(
        (atom for atom in atoms if len(atom.sources) >= 2 and atom.value > 1.0e-9),
        key=lambda atom: atom.value,
        reverse=True,
    )
    return {
        "joint_ei": joint_ei,
        "scalar_ei_sum": scalar_ei_sum,
        "raw_phi": raw_phi,
        "within_phi_sum": within_phi_sum,
        "cross_network_phi": cross_phi,
        "module_ei": module_ei,
        "scalar_ei_by_network": scalar_ei,
        "within_network_phi": within_phi,
        "cross_network_shapley": cross_shapley,
        "total_phi_contribution": total_contribution,
        "cross_shapley_sum_error": float(sum(cross_shapley.values()) - cross_phi),
        "total_contribution_sum_error": float(sum(total_contribution.values()) - raw_phi),
        "atom_sum_error": float(sum(atom.value for atom in atoms) - cross_phi),
        "top_atoms": [
            {
                "sources": list(atom.sources),
                "value": float(atom.value),
                "kind": str(atom.kind),
                "depth": int(atom.depth),
            }
            for atom in ranked_atoms[:3]
        ],
    }


def analyze_series(
    raw: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    *,
    subject: str,
    condition: str,
    order: int,
    alpha: float,
) -> dict[str, Any]:
    development_end = development_end_for_length(len(raw))
    reduced = fit_yeo7_pc1(np.asarray(raw, dtype=float)[:development_end], groups).transform(raw)
    fitted = fit_delta_history_phi(reduced, alpha=alpha, order=order, development_end=development_end)
    attribution = transition_attribution(
        fitted["transition"], fitted["noise_covariance"], tuple(groups), order=order
    )
    development = np.asarray(reduced[:development_end], dtype=float)
    scale = np.where(development.std(axis=0, ddof=1) > 1.0e-12, development.std(axis=0, ddof=1), 1.0)
    max_z = float(np.max(np.abs((development - development.mean(axis=0)) / scale)))
    noise_condition = float(np.linalg.cond(np.asarray(fitted["noise_covariance"], dtype=float)))
    return {
        "subject": subject,
        "condition": condition,
        "n_timepoints": int(len(raw)),
        "development_end": int(development_end),
        "quality_diagnostics": {
            "max_abs_development_pc_zscore": max_z,
            "noise_covariance_condition": noise_condition,
            "quality_flag": bool(max_z > 10.0 or noise_condition > 500.0),
            "flag_rule": "max_abs_development_pc_zscore > 10 or noise_covariance_condition > 500",
        },
        **attribution,
    }


def _analyze_file(payload: tuple[Any, ...]) -> dict[str, Any]:
    path, groups, subject, condition, data_key, order, alpha = payload
    if condition == "REST":
        raw = load_rest_series(Path(path), data_key="Schaefer500")
    else:
        raw = load_task_series(Path(path), data_key=str(data_key), parcel_count=500)
    return analyze_series(
        raw,
        groups,
        subject=str(subject),
        condition=str(condition),
        order=int(order),
        alpha=float(alpha),
    )


def bootstrap_mean_interval(values: Sequence[float], *, seed: int, replicates: int = 10_000) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(replicates), len(array)))
    return [float(value) for value in np.quantile(array[indices].mean(axis=1), [0.025, 0.975])]


def summarize_cores(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        for atom in row["top_atoms"]:
            grouped[tuple(atom["sources"])].append(float(atom["value"]))
    result = [
        {
            "sources": list(sources),
            "top3_frequency": int(len(values)),
            "mean_atom_value_when_top": float(np.mean(values)),
            "median_atom_value_when_top": float(np.median(values)),
        }
        for sources, values in grouped.items()
    ]
    return sorted(
        result,
        key=lambda row: (-int(row["top3_frequency"]), -float(row["mean_atom_value_when_top"]), row["sources"]),
    )


def summarize(rows: Sequence[Mapping[str, Any]], *, bootstrap_replicates: int) -> list[dict[str, Any]]:
    condition_results = []
    metrics = ("joint_ei", "raw_phi", "within_phi_sum", "cross_network_phi")
    for condition_index, condition in enumerate(CONDITIONS):
        selected = [row for row in rows if row["condition"] == condition]
        network_summary = []
        for network_index, network in enumerate(NETWORK_ORDER):
            item: dict[str, Any] = {"network": network}
            for metric_index, metric in enumerate(
                ("module_ei", "within_network_phi", "cross_network_shapley", "total_phi_contribution")
            ):
                values = [float(row[metric][network]) for row in selected]
                item[metric] = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "bootstrap_95_ci": bootstrap_mean_interval(
                        values,
                        seed=2026072300 + 100 * condition_index + 10 * network_index + metric_index,
                        replicates=bootstrap_replicates,
                    ),
                }
            network_summary.append(item)
        system_summary = {}
        for metric_index, metric in enumerate(metrics):
            values = [float(row[metric]) for row in selected]
            system_summary[metric] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "bootstrap_95_ci": bootstrap_mean_interval(
                    values,
                    seed=2026072600 + 100 * condition_index + metric_index,
                    replicates=bootstrap_replicates,
                ),
            }
        condition_results.append(
            {
                "condition": condition,
                "n_subjects": len(selected),
                "network_summary": network_summary,
                "system_summary": system_summary,
                "core_summary": summarize_cores(selected),
                "quality_flags": [
                    {"subject": row["subject"], **row["quality_diagnostics"]}
                    for row in selected
                    if row["quality_diagnostics"]["quality_flag"]
                ],
            }
        )
    return condition_results


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def _save(fig: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _core_label(sources: Sequence[str]) -> str:
    selected = set(sources)
    if len(selected) == 7:
        return "All 7"
    missing = [NETWORK_LABELS[name] for name in NETWORK_ORDER if name not in selected]
    if len(missing) <= 2:
        return "No " + "/".join(missing)
    return "+".join(NETWORK_LABELS[name] for name in NETWORK_ORDER if name in selected)


def plot_summary(summary: Mapping[str, Any], destination: Path) -> None:
    _style()
    condition_results = {item["condition"]: item for item in summary["condition_results"]}
    all_total = [
        item["total_phi_contribution"]["mean"]
        for result in condition_results.values()
        for item in result["network_summary"]
    ]
    minimum, maximum = float(min(all_total)), float(max(all_total))
    if minimum < 0.0 < maximum:
        color_norm: Normalize = TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)
    else:
        color_norm = Normalize(vmin=min(0.0, minimum), vmax=max(maximum, 1.0e-9))
    node_cmap = mpl.colormaps["RdBu_r"]
    all_ei = [
        item["module_ei"]["mean"]
        for result in condition_results.values()
        for item in result["network_summary"]
    ]
    ei_min, ei_max = float(min(all_ei)), float(max(all_ei))

    core_union: set[tuple[str, ...]] = set()
    for result in condition_results.values():
        core_union.update(tuple(item["sources"]) for item in result["core_summary"][:3])
    lookup = {
        (condition, tuple(item["sources"])): item
        for condition, result in condition_results.items()
        for item in result["core_summary"]
    }
    cores = sorted(
        core_union,
        key=lambda core: (
            -sum(lookup.get((condition, core), {}).get("top3_frequency", 0) for condition in CONDITIONS),
            core,
        ),
    )
    contribution_values = [
        float(lookup[(condition, core)]["mean_atom_value_when_top"])
        for core in cores
        for condition in CONDITIONS
        if (condition, core) in lookup
    ]
    contribution_norm = Normalize(vmin=min(contribution_values), vmax=max(contribution_values))
    contribution_cmap = mpl.colormaps["YlOrBr"]

    fig = plt.figure(figsize=(11.2, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.05, 1.0), width_ratios=(1.05, 1.05, 1.0))
    top_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    ei_axis = fig.add_subplot(grid[1, 0])
    phi_axis = fig.add_subplot(grid[1, 1])
    core_axis = fig.add_subplot(grid[1, 2])

    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(NETWORK_ORDER), endpoint=False)
    positions = {name: np.array([np.cos(angle), np.sin(angle)]) for name, angle in zip(NETWORK_ORDER, angles)}
    for axis, condition in zip(top_axes, CONDITIONS):
        result = condition_results[condition]
        by_network = {item["network"]: item for item in result["network_summary"]}
        top_core = result["core_summary"][0]
        top_core_contribution = float(top_core["mean_atom_value_when_top"])
        core_color = contribution_cmap(contribution_norm(top_core_contribution))
        core_points = np.asarray([positions[name] for name in NETWORK_ORDER if name in set(top_core["sources"])])
        if len(core_points) >= 3:
            axis.add_patch(
                Polygon(
                    core_points,
                    closed=True,
                    facecolor=mpl.colors.to_rgba(core_color, alpha=0.30),
                    edgecolor=core_color,
                    linewidth=1.4,
                    zorder=1,
                )
            )
        elif len(core_points) == 2:
            axis.plot(core_points[:, 0], core_points[:, 1], color=core_color, linewidth=2.0)
        for network in NETWORK_ORDER:
            item = by_network[network]
            ei_value = float(item["module_ei"]["mean"])
            fraction = (ei_value - ei_min) / max(ei_max - ei_min, 1.0e-12)
            size = 80.0 + 260.0 * fraction
            phi_value = float(item["total_phi_contribution"]["mean"])
            x, y = positions[network]
            axis.scatter(
                [x],
                [y],
                s=size,
                color=node_cmap(color_norm(phi_value)),
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )
            label_x, label_y = 1.24 * positions[network]
            axis.text(label_x, label_y, NETWORK_LABELS[network], ha="center", va="center", fontsize=6.2)
        raw_phi = result["system_summary"]["raw_phi"]["mean"]
        cross_phi = result["system_summary"]["cross_network_phi"]["mean"]
        axis.set_title(f"{CONDITION_LABELS[condition]}\nraw $\\Phi$={raw_phi:.2f}, cross-network={cross_phi:.2f} bits", fontsize=7)
        axis.text(
            0.5,
            -0.04,
            f"top core: {_core_label(top_core['sources'])} "
            f"({top_core['top3_frequency']}/{result['n_subjects']}); atom={top_core_contribution:.2f} bits",
            transform=axis.transAxes,
            ha="center",
            fontsize=6.2,
        )
        axis.set(xlim=(-1.42, 1.42), ylim=(-1.38, 1.35), aspect="equal")
        axis.axis("off")
    top_axes[0].text(-0.07, 1.04, "a", transform=top_axes[0].transAxes, fontweight="bold", fontsize=9)
    top_axes[0].text(
        0.02,
        0.02,
        "node size: module EI\npolygon color: mean greedy atom contribution",
        transform=top_axes[0].transAxes,
        fontsize=6.2,
    )
    scalar = mpl.cm.ScalarMappable(norm=color_norm, cmap=node_cmap)
    colorbar = fig.colorbar(scalar, ax=top_axes, shrink=0.72, pad=0.015)
    colorbar.set_label("Mean total $\\Phi$ attribution (bits)")

    y = np.arange(len(NETWORK_ORDER))
    offsets = {"REST": -0.18, "MOTOR": 0.0, "WM": 0.18}
    for condition in CONDITIONS:
        result = condition_results[condition]
        by_network = {item["network"]: item for item in result["network_summary"]}
        for axis, metric in ((ei_axis, "module_ei"), (phi_axis, "total_phi_contribution")):
            means = np.asarray([by_network[name][metric]["mean"] for name in NETWORK_ORDER], dtype=float)
            intervals = np.asarray([by_network[name][metric]["bootstrap_95_ci"] for name in NETWORK_ORDER], dtype=float)
            axis.errorbar(
                means,
                y + offsets[condition],
                xerr=np.vstack((means - intervals[:, 0], intervals[:, 1] - means)),
                fmt="o",
                markersize=3.8,
                capsize=2,
                linewidth=0.9,
                color=CONDITION_COLORS[condition],
                label=CONDITION_LABELS[condition],
            )
    ei_axis.set(
        yticks=y,
        yticklabels=[NETWORK_LABELS[name] for name in NETWORK_ORDER],
        xlabel="Module EI (bits; mean and 95% bootstrap CI)",
    )
    ei_axis.invert_yaxis()
    ei_axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    phi_axis.axvline(0.0, color="#606060", linestyle="--", linewidth=0.7)
    phi_axis.set(
        yticks=y,
        yticklabels=[NETWORK_LABELS[name] for name in NETWORK_ORDER],
        xlabel="Total $\\Phi$ attribution (bits; mean and 95% bootstrap CI)",
    )
    phi_axis.invert_yaxis()
    ei_axis.text(-0.16, 1.04, "b", transform=ei_axis.transAxes, fontweight="bold", fontsize=9)
    phi_axis.text(-0.16, 1.04, "c", transform=phi_axis.transAxes, fontweight="bold", fontsize=9)

    for row_index, core in enumerate(cores):
        for column_index, condition in enumerate(CONDITIONS):
            item = lookup.get((condition, core))
            if item is None:
                continue
            frequency = int(item["top3_frequency"])
            contribution = float(item["mean_atom_value_when_top"])
            core_axis.scatter(
                column_index,
                row_index,
                s=24 + 7.0 * frequency,
                color=contribution_cmap(contribution_norm(contribution)),
                edgecolor="#6B4A2B",
                linewidth=0.5,
            )
            core_axis.text(column_index, row_index, str(frequency), ha="center", va="center", fontsize=6.2)
    core_axis.set(
        xticks=np.arange(len(CONDITIONS)),
        xticklabels=[CONDITION_LABELS[name] for name in CONDITIONS],
        yticks=np.arange(len(cores)),
        yticklabels=[_core_label(core) for core in cores],
        xlabel="Number = subjects with core in top-3\nColor = mean atom contribution (bits)",
        ylabel="Synergy core",
        xlim=(-0.6, len(CONDITIONS) - 0.4),
        ylim=(-0.6, len(cores) - 0.4),
    )
    core_axis.invert_yaxis()
    core_axis.tick_params(length=0)
    core_axis.spines["left"].set_visible(False)
    core_axis.spines["bottom"].set_visible(False)
    core_axis.text(-0.18, 1.04, "d", transform=core_axis.transAxes, fontweight="bold", fontsize=9)
    scalar_core = mpl.cm.ScalarMappable(norm=contribution_norm, cmap=contribution_cmap)
    core_colorbar = fig.colorbar(scalar_core, ax=core_axis, shrink=0.72, pad=0.02)
    core_colorbar.set_label("Mean greedy atom contribution (bits)\n(polygons and panel d)")
    _save(fig, destination)


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# REST–MOTOR–WM Yeo7 网络 EI、Phi 归因与协同核",
        "",
        "三条件使用29名共同被试、各自前75%时间点拟合的 Yeo7-PC1、共享预测最优的五阶 Δ-Ridge（p=5, alpha=10）和 Gaussian log-det EI。不同条件分别重拟合 PCA、标准化、Ridge 与残差协方差，因此结果是条件对照，不是仅改变任务标签的单因素因果效应。",
        "",
        "对网络 i 的五阶历史模块 H_i，module EI 定义为 EI(H_i; X_{t+1})。网络内部历史整合为 module EI 减去五个单滞后 EI 之和。跨网络 Phi 以七个网络模块为分解单位，对全部 128 个网络组合精确计算 Shapley 归因。每个网络的 total Phi attribution 等于其网络内部历史整合加上 Shapley 分得的跨网络 Phi；七网络归因之和逐被试严格等于原始35维 history-source Phi。协同核仍按多网络 greedy atom 展示，不解释为单一网络固有量。",
        "",
        "## 系统级汇总",
        "",
        "| 条件 | joint EI | raw Phi | 网络内历史整合和 | 跨网络 Phi |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in summary["condition_results"]:
        system = result["system_summary"]
        lines.append(
            f"| {CONDITION_LABELS[result['condition']]} | {system['joint_ei']['mean']:.6f} | "
            f"{system['raw_phi']['mean']:.6f} | {system['within_phi_sum']['mean']:.6f} | "
            f"{system['cross_network_phi']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 网络级汇总",
            "",
            "| 条件 | 网络 | module EI | 网络内历史整合 | 跨网络 Shapley Phi | total Phi attribution |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for result in summary["condition_results"]:
        for item in result["network_summary"]:
            lines.append(
                f"| {CONDITION_LABELS[result['condition']]} | {NETWORK_LABELS[item['network']]} | "
                f"{item['module_ei']['mean']:.6f} | {item['within_network_phi']['mean']:.6f} | "
                f"{item['cross_network_shapley']['mean']:.6f} | {item['total_phi_contribution']['mean']:.6f} |"
            )
    lines.extend(
        [
            "",
            "图 a 以节点大小编码 module EI、节点颜色编码 total Phi attribution；多边形标出每个条件最常进入被试 top-3 的协同核，其颜色编码该核进入 top-3 时的平均 greedy atom 贡献。图 b、c 给出网络级均值与被试 bootstrap 95% CI；图 d 同时显示主要协同核的 top-3 频率和平均 atom 贡献，并与多边形共享 atom 色标。",
            "",
            "REST 使用900个拟合时间点，MOTOR 与 WM 分别使用213和304个拟合时间点；不同有效样本长度与条件共同变化，跨条件差异不能解释为纯任务效应。Shapley 是满足效率与对称性的归因约定，不等于唯一的生物学归属。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    rest_root: Path,
    task_root: Path,
    labels_path: Path,
    output_dir: Path,
    *,
    subjects: Sequence[str] | None = None,
    data_key: str = "Schaefer500_taskRetained",
    order: int = 5,
    alpha: float = 10.0,
    workers: int = 4,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    discovered = discover_inputs(rest_root, task_root)
    if subjects is not None:
        wanted = set(subjects)
        missing = wanted - set(discovered)
        if missing:
            raise FileNotFoundError(f"Missing complete REST/MOTOR/WM inputs for {sorted(missing)}")
        discovered = {subject: discovered[subject] for subject in sorted(wanted)}
    if not discovered:
        raise FileNotFoundError("No common REST/MOTOR/WM subjects were found.")
    groups = load_yeo7_groups(labels_path, expected_parcels=500)
    payloads = [
        (str(paths[condition]), groups, subject, condition, data_key, order, alpha)
        for subject, paths in discovered.items()
        for condition in CONDITIONS
    ]
    rows: list[dict[str, Any]] = []
    progress = tqdm(total=len(payloads), desc="Yeo7 EI/Phi attribution", unit="fit", mininterval=1.0)
    try:
        if int(workers) <= 1:
            for payload in payloads:
                rows.append(_analyze_file(payload))
                progress.update(1)
        else:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                future_by_key = {
                    executor.submit(_analyze_file, payload): (str(payload[2]), str(payload[3]))
                    for payload in payloads
                }
                for future in as_completed(future_by_key):
                    rows.append(future.result())
                    progress.update(1)
    finally:
        progress.close()
    rows.sort(key=lambda row: (CONDITIONS.index(str(row["condition"])), str(row["subject"])))
    maximum_identity_error = max(
        max(
            abs(float(row["cross_shapley_sum_error"])),
            abs(float(row["total_contribution_sum_error"])),
            abs(float(row["atom_sum_error"])),
        )
        for row in rows
    )
    summary = {
        "config": {
            "conditions": list(CONDITIONS),
            "subjects": sorted(discovered),
            "n_subjects": len(discovered),
            "representation": "Yeo7 PC1 independently refitted per subject and condition on the first 75% of time points",
            "model": "delta Ridge independently refitted per subject and condition",
            "order": int(order),
            "alpha": float(alpha),
            "selection_rule": "shared lowest held-out delta-NRMSE across REST and seven tasks",
            "source_dimension": int(order * len(NETWORK_ORDER)),
            "target_dimension": len(NETWORK_ORDER),
            "estimator": "Gaussian log-det affine continuous-EI approximation",
            "attribution": "exact Shapley over all 2^7 Yeo7 network coalitions",
            "null_replicates": 0,
            "bootstrap_replicates": int(bootstrap_replicates),
            "workers": int(workers),
        },
        "identity_checks": {"maximum_absolute_error_bits": float(maximum_identity_error)},
        "rows": rows,
    }
    summary["condition_results"] = summarize(rows, bootstrap_replicates=bootstrap_replicates)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(summary, output_dir / "report.md")
    plot_summary(summary, output_dir / "rest_motor_wm_network_attribution")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rest-root", type=Path, default=DEFAULT_REST_ROOT)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--labels", type=Path, default=default_yeo7_labels(500))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subjects", default="", help="Comma-separated common subject IDs for smoke tests.")
    parser.add_argument("--data-key", default="Schaefer500_taskRetained")
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args(argv)
    subjects = tuple(value.strip() for value in args.subjects.split(",") if value.strip()) or None
    summary = run(
        args.rest_root,
        args.task_root,
        args.labels,
        args.output_dir,
        subjects=subjects,
        data_key=args.data_key,
        order=args.order,
        alpha=args.alpha,
        workers=args.workers,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(
        json.dumps(
            {
                "n_subjects": summary["config"]["n_subjects"],
                "maximum_identity_error_bits": summary["identity_checks"]["maximum_absolute_error_bits"],
                "system_summary": {
                    item["condition"]: item["system_summary"] for item in summary["condition_results"]
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
