#!/usr/bin/env python3
"""Attribute Yeo7 EI/Phi and compare synergy cores across REST and all HCP tasks."""

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
from scripts.run_hcp_schaefer500_all_tasks_phi import (
    CONDITION_ORDER,
    DISPLAY_NAMES,
    TASKS,
    development_end_for_length,
)
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
CONDITIONS = CONDITION_ORDER
CONDITION_LABELS = DISPLAY_NAMES


def discover_inputs(rest_root: Path, task_root: Path) -> dict[str, dict[str, Path]]:
    rest = {
        path.parent.name: path
        for path in Path(rest_root).glob("sub-*/*REST1_LR*schaefer500-1000_yeo7.mat")
    }
    task_paths = {
        task.removesuffix("_LR"): {
            path.parent.name: path for path in Path(task_root).glob(f"sub-*/{task}.mat")
        }
        for task in TASKS
    }
    common = set(rest)
    for paths in task_paths.values():
        common &= set(paths)
    return {
        subject: {
            "REST": rest[subject],
            **{condition: paths[subject] for condition, paths in task_paths.items()},
        }
        for subject in sorted(common)
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


def summarize_paired_differences(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_replicates: int
) -> list[dict[str, Any]]:
    by_key = {(str(row["subject"]), str(row["condition"])): row for row in rows}
    subjects = sorted({str(row["subject"]) for row in rows})
    comparisons = []
    for condition_index, condition in enumerate(CONDITIONS[1:], start=1):
        system_summary = {}
        for metric_index, metric in enumerate(("joint_ei", "raw_phi", "cross_network_phi")):
            differences = np.asarray(
                [
                    float(by_key[(subject, condition)][metric])
                    - float(by_key[(subject, "REST")][metric])
                    for subject in subjects
                ],
                dtype=float,
            )
            system_summary[metric] = {
                "mean": float(differences.mean()),
                "median": float(np.median(differences)),
                "std": float(differences.std(ddof=1)) if len(differences) > 1 else 0.0,
                "bootstrap_95_ci": bootstrap_mean_interval(
                    differences,
                    seed=2026072800 + 100 * condition_index + metric_index,
                    replicates=bootstrap_replicates,
                ),
                "n_positive": int(np.sum(differences > 0.0)),
                "n_negative": int(np.sum(differences < 0.0)),
            }
        network_summary = []
        for network_index, network in enumerate(NETWORK_ORDER):
            item: dict[str, Any] = {"network": network}
            for metric_index, metric in enumerate(("module_ei", "total_phi_contribution")):
                differences = np.asarray(
                    [
                        float(by_key[(subject, condition)][metric][network])
                        - float(by_key[(subject, "REST")][metric][network])
                        for subject in subjects
                    ],
                    dtype=float,
                )
                item[metric] = {
                    "mean": float(differences.mean()),
                    "median": float(np.median(differences)),
                    "std": float(differences.std(ddof=1)) if len(differences) > 1 else 0.0,
                    "bootstrap_95_ci": bootstrap_mean_interval(
                        differences,
                        seed=2026072900 + 100 * condition_index + 10 * network_index + metric_index,
                        replicates=bootstrap_replicates,
                    ),
                    "n_positive": int(np.sum(differences > 0.0)),
                    "n_negative": int(np.sum(differences < 0.0)),
                }
            network_summary.append(item)
        comparisons.append(
            {
                "condition": condition,
                "reference": "REST",
                "n_paired_subjects": len(subjects),
                "system_summary": system_summary,
                "network_summary": network_summary,
            }
        )
    return comparisons


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
    contribution_min = float(min(contribution_values))
    contribution_max = float(max(contribution_values))
    contribution_norm = Normalize(
        vmin=contribution_min,
        vmax=contribution_max if contribution_max > contribution_min else contribution_min + 1.0e-9,
    )
    contribution_cmap = mpl.colormaps["YlOrBr"]

    fig = plt.figure(figsize=(12.8, 9.6), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=(1.0, 1.0, 1.08))
    top_axes = [fig.add_subplot(grid[index // 4, index % 4]) for index in range(len(CONDITIONS))]
    bottom = grid[2, :].subgridspec(1, 3, width_ratios=(1.0, 1.0, 1.55))
    ei_axis = fig.add_subplot(bottom[0, 0])
    phi_axis = fig.add_subplot(bottom[0, 1])
    core_axis = fig.add_subplot(bottom[0, 2])

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
            size = 50.0 + 180.0 * fraction
            phi_value = float(item["total_phi_contribution"]["mean"])
            x, y = positions[network]
            axis.scatter(
                [x],
                [y],
                s=size,
                color=node_cmap(color_norm(phi_value)),
                edgecolor="white",
                linewidth=0.75,
                zorder=3,
            )
            label_x, label_y = 1.25 * positions[network]
            axis.text(label_x, label_y, NETWORK_LABELS[network], ha="center", va="center", fontsize=5.5)
        raw_phi = result["system_summary"]["raw_phi"]["mean"]
        cross_phi = result["system_summary"]["cross_network_phi"]["mean"]
        axis.set_title(
            f"{CONDITION_LABELS[condition]}\nraw $\\Phi$={raw_phi:.2f}; cross={cross_phi:.2f} bits",
            fontsize=6.4,
        )
        axis.text(
            0.5,
            -0.055,
            f"{_core_label(top_core['sources'])} "
            f"({top_core['top3_frequency']}/{result['n_subjects']}); atom={top_core_contribution:.2f}",
            transform=axis.transAxes,
            ha="center",
            fontsize=5.3,
        )
        axis.set(xlim=(-1.43, 1.43), ylim=(-1.40, 1.36), aspect="equal")
        axis.axis("off")
    top_axes[0].text(-0.08, 1.04, "a", transform=top_axes[0].transAxes, fontweight="bold", fontsize=9)
    top_axes[0].text(
        0.02,
        0.02,
        "node size: module EI\nnode color: total $\\Phi$ attribution\npolygon: leading greedy core",
        transform=top_axes[0].transAxes,
        fontsize=5.1,
    )
    scalar = mpl.cm.ScalarMappable(norm=color_norm, cmap=node_cmap)
    colorbar = fig.colorbar(scalar, ax=top_axes, shrink=0.78, pad=0.012)
    colorbar.set_label("Mean total $\\Phi$ attribution (bits)")

    module_matrix = np.asarray(
        [
            [
                next(
                    item["module_ei"]["mean"]
                    for item in condition_results[condition]["network_summary"]
                    if item["network"] == network
                )
                for condition in CONDITIONS
            ]
            for network in NETWORK_ORDER
        ],
        dtype=float,
    )
    phi_matrix = np.asarray(
        [
            [
                next(
                    item["total_phi_contribution"]["mean"]
                    for item in condition_results[condition]["network_summary"]
                    if item["network"] == network
                )
                for condition in CONDITIONS
            ]
            for network in NETWORK_ORDER
        ],
        dtype=float,
    )

    def draw_heatmap(axis: Any, matrix: np.ndarray, *, cmap: Any, norm: Normalize, label: str) -> None:
        image = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        axis.set(
            xticks=np.arange(len(CONDITIONS)),
            xticklabels=[CONDITION_LABELS[name] for name in CONDITIONS],
            yticks=np.arange(len(NETWORK_ORDER)),
            yticklabels=[NETWORK_LABELS[name] for name in NETWORK_ORDER],
            xlabel="State",
            ylabel="Yeo7 network",
        )
        axis.tick_params(axis="x", labelrotation=45, length=0)
        axis.tick_params(axis="y", length=0)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                red, green, blue, _ = cmap(norm(matrix[row_index, column_index]))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                color = "white" if luminance < 0.53 else "black"
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=4.7,
                    color=color,
                )
        cbar = fig.colorbar(image, ax=axis, shrink=0.78, pad=0.02)
        cbar.set_label(label)

    ei_norm = Normalize(vmin=ei_min, vmax=ei_max if ei_max > ei_min else ei_min + 1.0e-9)
    draw_heatmap(
        ei_axis,
        module_matrix,
        cmap=mpl.colormaps["Blues"],
        norm=ei_norm,
        label="Mean module EI (bits)",
    )
    draw_heatmap(
        phi_axis,
        phi_matrix,
        cmap=node_cmap,
        norm=color_norm,
        label="Mean total $\\Phi$ attribution (bits)",
    )
    ei_axis.text(-0.18, 1.04, "b", transform=ei_axis.transAxes, fontweight="bold", fontsize=9)
    phi_axis.text(-0.18, 1.04, "c", transform=phi_axis.transAxes, fontweight="bold", fontsize=9)

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
                s=20 + 4.2 * frequency,
                color=contribution_cmap(contribution_norm(contribution)),
                edgecolor="#6B4A2B",
                linewidth=0.45,
            )
            core_axis.text(column_index, row_index, str(frequency), ha="center", va="center", fontsize=5.3)
    core_axis.set(
        xticks=np.arange(len(CONDITIONS)),
        xticklabels=[CONDITION_LABELS[name] for name in CONDITIONS],
        yticks=np.arange(len(cores)),
        yticklabels=[_core_label(core) for core in cores],
        xlabel="State; number = subjects with core in top-3",
        ylabel="Synergy core",
        xlim=(-0.6, len(CONDITIONS) - 0.4),
        ylim=(-0.6, len(cores) - 0.4),
    )
    core_axis.invert_yaxis()
    core_axis.tick_params(length=0)
    core_axis.tick_params(axis="x", labelrotation=35)
    core_axis.spines["left"].set_visible(False)
    core_axis.spines["bottom"].set_visible(False)
    core_axis.text(-0.18, 1.04, "d", transform=core_axis.transAxes, fontweight="bold", fontsize=9)
    scalar_core = mpl.cm.ScalarMappable(norm=contribution_norm, cmap=contribution_cmap)
    core_colorbar = fig.colorbar(scalar_core, ax=core_axis, shrink=0.78, pad=0.02)
    core_colorbar.set_label("Mean greedy atom contribution when top (bits)")
    _save(fig, destination)


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# REST 与七任务态的 Yeo7 网络 EI、Phi 归因与协同核",
        "",
        f"本比较纳入 {summary['config']['n_subjects']} 名同时具有 REST 和七个 HCP 任务态数据的被试。状态是比较因素，被试是配对单位；八个状态固定使用相同的 Schaefer-500 分区、Yeo7 网络定义、前 75% 拟合段、五阶历史、Ridge 正则强度、Gaussian log-det EI、网络归因公式和绘图尺度。各状态分别重拟合 PCA、标准化、Ridge 与残差协方差，因此结果是配对状态对照，不是只改变任务标签的单因素因果效应。",
        "",
        "图中的节点是七个 Yeo7 功能网络，不是单个 Schaefer parcel。节点大小表示该网络五阶历史模块对下一时刻七网络状态的 EI；节点颜色表示分配给该网络的 total Phi attribution；多边形表示该状态中跨被试最常进入 greedy top-3 的多网络协同核。",
        "",
        "## 计算方法",
        "",
        "### 1. Yeo7 状态表示",
        "",
        r"对被试 $s$ 和状态 $c$，原始时间序列记为 $\mathbf{R}^{(s,c)}\in\mathbb{R}^{T_c\times 500}$。拟合段长度取 $D_c=\operatorname{round}(0.75T_c)$。在每个 Yeo7 网络内部，只用前 $D_c$ 个时间点拟合一维 PCA，再将同一变换应用到完整序列，得到七维状态向量 $\mathbf{x}_t\in\mathbb{R}^{7}$。PCA 按被试和状态独立拟合，避免把测试段信息用于降维。",
        "",
        "### 2. 五阶 Delta-Ridge 动力学",
        "",
        r"历史源向量为",
        "",
        "$$",
        r"\mathbf{h}_t=\left[\mathbf{x}_t^\top,\mathbf{x}_{t-1}^\top,\ldots,\mathbf{x}_{t-4}^\top\right]^\top\in\mathbb{R}^{35},",
        "$$",
        "",
        r"目标为 $\mathbf{y}_t=\mathbf{x}_{t+1}$。代码先用拟合段均值和样本标准差标准化每个历史滞后的七维状态，再对增量 $\Delta\mathbf{x}_t=\mathbf{x}_{t+1}-\mathbf{x}_t$ 单独标准化，拟合",
        "",
        "$$",
        r"\widehat{\mathbf{B}}=\arg\min_{\mathbf{B},\mathbf{b}}\sum_t\left\|\widetilde{\Delta\mathbf{x}}_t-\mathbf{B}\widetilde{\mathbf{h}}_t-\mathbf{b}\right\|_2^2+\alpha\|\mathbf{B}\|_F^2,",
        "$$",
        "",
        f"其中历史阶数 $p={summary['config']['order']}$，$\\alpha={summary['config']['alpha']:g}$。随后把增量模型还原为标准化状态坐标中的线性转移 $\\mathbf{{y}}_t=\\mathbf{{A}}\\mathbf{{h}}_t+\\boldsymbol{{\\varepsilon}}_t$，并由训练残差估计噪声协方差矩阵 $\\mathbf{{\\Sigma}}_\\varepsilon$。",
        "",
        "### 3. Gaussian log-det EI 与原始 Phi",
        "",
        r"对历史源索引集合 $S$，干预读出采用单位协方差、源维独立的 Gaussian 支持。记 $\mathbf{A}_S$ 为对应转移列，则",
        "",
        "$$",
        r"EI(S;\mathbf{y})=\frac{1}{2\ln 2}\left[\log\det\left(\mathbf{A}\mathbf{A}^\top+\mathbf{\Sigma}_\varepsilon\right)-\log\det\left(\mathbf{A}_{S^c}\mathbf{A}_{S^c}^\top+\mathbf{\Sigma}_\varepsilon\right)\right].",
        "$$",
        "",
        r"所有协方差 log-determinant 计算使用 $10^{-6}$ 的特征值下限。35 维历史源的原始整合量定义为",
        "",
        "$$",
        r"\Phi_{\mathrm{raw}}=EI(\{1,\ldots,35\};\mathbf{y})-\sum_{j=1}^{35}EI(\{j\};\mathbf{y}).",
        "$$",
        "",
        "这里沿用既有线性 Gaussian 分析而没有改用 TM：每个被试–状态都要对 127 个非空网络联盟求 EI，并进一步做精确 Shapley 归因；在当前 35 维历史源和 232 个配对模型上，TM 会显著扩大计算量，也会改变与已有结果的估计口径。代价是当前 EI 只刻画拟合线性动力学及其 Gaussian 干预支持，不覆盖一般非线性或非 Gaussian 机制。",
        "",
        "### 4. 网络内整合与跨网络 Shapley 归因",
        "",
        r"令 $\mathcal{N}=\{1,\ldots,7\}$ 表示 Yeo7 网络集合，$H_i$ 表示网络 $i$ 的五个历史滞后。对任意 $S\subseteq\mathcal{N}$，定义联盟函数 $F(S)=EI(\cup_{i\in S}H_i;\mathbf{y})$，并令 $F(\varnothing)=0$。网络 $i$ 的 module EI 为 $F(\{i\})$，网络内历史整合为",
        "",
        "$$",
        r"\Phi_i^{\mathrm{within}}=F(\{i\})-\sum_{\ell=0}^{4}EI(x_{t-\ell,i};\mathbf{y}).",
        "$$",
        "",
        r"跨网络联盟价值定义为 $G(S)=F(S)-\sum_{i\in S}F(\{i\})$。对全部 $2^7=128$ 个联盟精确枚举后，网络 $i$ 的 Shapley 份额为",
        "",
        "$$",
        r"\psi_i=\sum_{S\subseteq\mathcal{N}\setminus\{i\}}\frac{|S|!(7-|S|-1)!}{7!}\left[G(S\cup\{i\})-G(S)\right].",
        "$$",
        "",
        r"最终节点颜色对应 $a_i=\Phi_i^{\mathrm{within}}+\psi_i$。按 Shapley 效率性质，$\sum_i\psi_i=G(\mathcal{N})$；代码还逐被试验证 $\sum_i a_i=\Phi_{\mathrm{raw}}$。",
        "",
        "### 5. Greedy 协同核",
        "",
        r"协同核以七个网络模块为不可拆单位。对一个网络子集 $Q$，先计算 $\Phi(Q)=F(Q)-\sum_{i\in Q}F(\{i\})$，再枚举所有无序非平凡二分 $Q=L\cup R$。算法只接受残差 $r_Q=\Phi(Q)-\Phi(L)-\Phi(R)\geq-10^{-4}$ 的拆分，并优先选择使 $\Phi(L)+\Phi(R)$ 最大的拆分；正残差作为当前层 atom，随后递归处理 $L$ 和 $R$。每名被试保留正贡献最大的三个多网络 atom。状态级 leading core 先按进入被试 top-3 的频率排序，再按进入 top-3 时的平均 atom 贡献排序。该结果依赖 greedy 路径，不是所有层级分解的唯一 exhaustive 解。",
        "",
        "### 6. 聚合、配对差值与不确定性",
        "",
        f"每个状态的均值由同一组 {summary['config']['n_subjects']} 名被试计算。95% 区间使用 {summary['config']['bootstrap_replicates']:,} 次被试 bootstrap 的均值分位数区间。任务减 REST 的差值先在同一被试内计算，再对配对差值做 bootstrap；`n+/n-` 分别是差值严格大于和小于 0 的被试数。未在本图上对多个状态、网络和指标进行假设检验或多重校正。",
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
            "## 相对 REST 的配对系统差值",
            "",
            "| 任务态 | Δ raw Phi [95% CI] | n+/n- | Δ cross-network Phi [95% CI] | n+/n- |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for comparison in summary["paired_task_minus_rest"]:
        raw = comparison["system_summary"]["raw_phi"]
        cross = comparison["system_summary"]["cross_network_phi"]
        lines.append(
            f"| {CONDITION_LABELS[comparison['condition']]} | {raw['mean']:.6f} "
            f"[{raw['bootstrap_95_ci'][0]:.6f}, {raw['bootstrap_95_ci'][1]:.6f}] | "
            f"{raw['n_positive']}/{raw['n_negative']} | {cross['mean']:.6f} "
            f"[{cross['bootstrap_95_ci'][0]:.6f}, {cross['bootstrap_95_ci'][1]:.6f}] | "
            f"{cross['n_positive']}/{cross['n_negative']} |"
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
            "## 各状态 leading core",
            "",
            "| 状态 | leading core | top-3 频率 | top 时平均 atom（bits） |",
            "|---|---|---:|---:|",
        ]
    )
    for result in summary["condition_results"]:
        top = result["core_summary"][0]
        lines.append(
            f"| {CONDITION_LABELS[result['condition']]} | {_core_label(top['sources'])} | "
            f"{top['top3_frequency']}/{result['n_subjects']} | {top['mean_atom_value_when_top']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 图示与解释边界",
            "",
            "图 a 用统一节点大小、节点色标和多边形色标展示 REST 与七任务态；图 b、c 用共享尺度热图比较 module EI 与 total Phi attribution；图 d 同时编码主要协同核的 top-3 频率和进入 top-3 时的平均 atom 贡献。所有色标均跨八状态统一，不能在单一 panel 内重新拉伸。",
            "",
            "REST 和不同任务的时间序列长度不同，因而拟合样本数与状态同时变化；PCA 方向也按状态独立估计。跨状态差异不能解释为纯任务诱发的因果效应。Shapley 是满足效率与对称性的归因约定，不是唯一的生物学归属；节点归因也不应解释为单个脑区的重要性。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_reusable_rows(
    summary_path: Path | None,
    *,
    subjects: Sequence[str],
    order: int,
    alpha: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    if summary_path is None or not Path(summary_path).is_file():
        return {}
    cached = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    config = cached.get("config", {})
    if int(config.get("order", -1)) != int(order) or not np.isclose(float(config.get("alpha", np.nan)), alpha):
        raise ValueError(
            f"Cannot reuse {summary_path}: cached order/alpha do not match p={order}, alpha={alpha}."
        )
    wanted_subjects = set(subjects)
    reusable = {}
    for row in cached.get("rows", []):
        subject = str(row.get("subject", ""))
        condition = str(row.get("condition", ""))
        if subject in wanted_subjects and condition in CONDITIONS:
            reusable[(subject, condition)] = dict(row)
    return reusable


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
    reuse_summary: Path | None = None,
) -> dict[str, Any]:
    discovered = discover_inputs(rest_root, task_root)
    if subjects is not None:
        wanted = set(subjects)
        missing = wanted - set(discovered)
        if missing:
            raise FileNotFoundError(f"Missing complete REST/all-task inputs for {sorted(missing)}")
        discovered = {subject: discovered[subject] for subject in sorted(wanted)}
    if not discovered:
        raise FileNotFoundError("No subjects with complete REST and seven-task inputs were found.")
    groups = load_yeo7_groups(labels_path, expected_parcels=500)
    reusable = load_reusable_rows(
        reuse_summary,
        subjects=tuple(discovered),
        order=order,
        alpha=alpha,
    )
    payloads = [
        (str(paths[condition]), groups, subject, condition, data_key, order, alpha)
        for subject, paths in discovered.items()
        for condition in CONDITIONS
        if (subject, condition) not in reusable
    ]
    rows: list[dict[str, Any]] = list(reusable.values())
    progress = tqdm(total=len(payloads), desc="Yeo7 EI/Phi attribution", unit="fit", mininterval=1.0)
    try:
        if not payloads:
            pass
        elif int(workers) <= 1:
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
            "reused_rows": len(reusable),
            "computed_rows": len(payloads),
        },
        "identity_checks": {"maximum_absolute_error_bits": float(maximum_identity_error)},
        "rows": rows,
    }
    summary["condition_results"] = summarize(rows, bootstrap_replicates=bootstrap_replicates)
    summary["paired_task_minus_rest"] = summarize_paired_differences(
        rows, bootstrap_replicates=bootstrap_replicates
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(summary, output_dir / "report.md")
    plot_summary(summary, output_dir / "all_conditions_network_attribution")
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
    parser.add_argument(
        "--reuse-summary",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "summary.json",
        help="Reuse compatible subject-condition rows from an existing summary JSON.",
    )
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
        reuse_summary=args.reuse_summary,
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
