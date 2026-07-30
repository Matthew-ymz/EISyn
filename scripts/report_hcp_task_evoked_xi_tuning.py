#!/usr/bin/env python3
"""Create final figures and report for the HCP task-evoked Xi parameter search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_task_evoked_pc2_xi_hierarchy import STATES, TASKS, all_atom_subsets
from scripts.analyze_hcp_schaefer500_yeo7_network_attribution import NETWORK_ORDER
from scripts.run_hcp_schaefer500_all_tasks_phi import DISPLAY_NAMES


DEFAULT_ROOT = ROOT / "results" / "hcp_schaefer500_task_evoked_xi_tuning"
DEFAULT_OUTPUT = DEFAULT_ROOT / "final"
DEFAULT_COGNITION_SCORES = ROOT / "results" / "hcp_single_group_sem_full_1206" / "selected_29_sem_results.csv"
DEFAULT_COGNITION_EXHAUSTIVE_ROOT = (
    ROOT / "results" / "hcp_cognition_exhaustive_targeted_greedy"
)
BASELINE = "k1_p5_a10"
SELECTED = "k1_p3_a1"
COMPARATORS = (BASELINE, "k1_p5_a1", "k1_p5_a0.3", "k1_p3_a0.3", SELECTED, "k2_p3_a2")
NETWORK_LABELS = {
    "Vis": "Visual",
    "SomMot": "SomMot",
    "DorsAttn": "DorsAttn",
    "SalVentAttn": "SalVentAttn",
    "Limbic": "Limbic",
    "Cont": "Control",
    "Default": "Default",
}
COGNITION_SCORES = ("g_score", "cry_score", "mem_score", "spd_score")
COGNITION_LABELS = ("General", "Crystallized", "Memory", "Speed")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: Any, path: Path) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def collect_search_summaries(root: Path) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for stage in ("screen", "refine"):
        for path in (root / stage).glob("*/summary.json"):
            row = load_json(path)
            rows[str(row["config_id"])] = row
    return list(rows.values())


def short_atom(name: str) -> str:
    mapping = {
        "Vis": "Vis",
        "SomMot": "Som",
        "DorsAttn": "DAN",
        "SalVentAttn": "SVAN",
        "Limbic": "Lim",
        "Cont": "Cont",
        "Default": "Def",
    }
    return "+".join(mapping[item] for item in name.split("+"))


def compact_atom(name: str) -> str:
    """Compact labels for dense hierarchy heatmaps."""
    mapping = {
        "Vis": "V",
        "SomMot": "SM",
        "DorsAttn": "DAN",
        "SalVentAttn": "VAN",
        "Limbic": "Lim",
        "Cont": "FPN",
        "Default": "DMN",
    }
    return "+".join(mapping[item] for item in name.split("+"))


def plot_selected(root: Path, output: Path) -> None:
    archive = np.load(root / "full" / SELECTED / "arrays.npz")
    network_mean = archive["network_share"].mean(axis=1).T * 100.0
    atom_share_mean = archive["atom_share"].mean(axis=1)
    atom_value = (
        archive["atom_value"]
        if "atom_value" in archive.files
        else archive["atom_share"] * archive["cross_xi"][:, :, None]
    )
    atom_mean = atom_value.mean(axis=1)
    atom_names = archive["atom_names"].astype(str)
    selected = np.argsort(atom_share_mean.mean(axis=0))[::-1][:12]
    atom_panel = atom_mean[:, selected].T

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), constrained_layout=True)
    lower, upper = float(np.floor(network_mean.min())), float(np.ceil(network_mean.max()))
    image = axes[0].imshow(network_mean, cmap="YlGnBu", vmin=lower, vmax=upper, aspect="auto")
    axes[0].set(
        xticks=np.arange(8),
        xticklabels=[DISPLAY_NAMES[state] for state in STATES],
        yticks=np.arange(7),
        yticklabels=[NETWORK_LABELS[name] for name in NETWORK_ORDER],
        xlabel="State (each column sums to 100%)",
        ylabel="Yeo7 network",
    )
    axes[0].tick_params(axis="x", labelrotation=35, length=0)
    axes[0].tick_params(axis="y", length=0)
    axes[0].axvline(0.5, color="#333333", linewidth=0.9)
    for row in range(7):
        for column in range(8):
            value = network_mean[row, column]
            normalized = (value - lower) / max(upper - lower, 1.0e-12)
            axes[0].text(column, row, f"{value:.1f}%", ha="center", va="center", fontsize=5.0, color="white" if normalized > 0.6 else "black")
    figure.colorbar(image, ax=axes[0], shrink=0.80, pad=0.02).set_label(r"Compositional share of system-level $\Xi$ (%)")
    atom_upper = max(float(np.quantile(atom_panel, 0.995)), 0.1)
    image = axes[1].imshow(atom_panel, cmap="magma_r", vmin=0.0, vmax=atom_upper, aspect="auto")
    axes[1].set(
        xticks=np.arange(8),
        xticklabels=[DISPLAY_NAMES[state] for state in STATES],
        yticks=np.arange(len(selected)),
        yticklabels=[short_atom(atom_names[index]) for index in selected],
        xlabel="State",
        ylabel="Greedy hierarchy atom",
    )
    axes[1].tick_params(axis="x", labelrotation=35, length=0)
    axes[1].tick_params(axis="y", length=0)
    axes[1].axvline(0.5, color="#F0F0F0", linewidth=0.9)
    for row in range(atom_panel.shape[0]):
        for column in range(8):
            value = atom_panel[row, column]
            axes[1].text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=4.1, color="white" if value > 0.38 * atom_upper else "black")
    figure.colorbar(image, ax=axes[1], shrink=0.80, pad=0.02).set_label(r"Mean hierarchy-atom contribution (bits)")
    for label, axis in zip("ab", axes):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    save_figure(figure, output / "selected_xi_state_distributions")


def load_cognition_data(path: Path, subjects: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Return the ordered z-score panel and scores aligned to the model subjects."""
    scores = pd.read_csv(path, dtype={"Subject": str})
    missing_columns = {"Subject", *COGNITION_SCORES}.difference(scores.columns)
    if missing_columns:
        raise ValueError(f"Cognition table is missing columns: {sorted(missing_columns)}")
    scores["Subject"] = scores["Subject"].str.removeprefix("sub-")
    model_subjects = {str(subject).removeprefix("sub-") for subject in subjects}
    score_subjects = set(scores["Subject"])
    if model_subjects != score_subjects:
        raise ValueError(
            "Cognition and Xi subject sets differ: "
            f"missing scores={sorted(model_subjects - score_subjects)}, "
            f"extra scores={sorted(score_subjects - model_subjects)}"
        )
    subject_order = [str(subject).removeprefix("sub-") for subject in subjects]
    aligned = scores.set_index("Subject").loc[subject_order, list(COGNITION_SCORES)]
    ordered = scores.sort_values("g_score", ascending=False).reset_index(drop=True)
    values = ordered.loc[:, COGNITION_SCORES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(aligned.to_numpy(dtype=float)).all():
        raise ValueError("Cognition scores contain non-finite values")
    panel = (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)
    return panel, aligned


def plot_main_combined(
    root: Path,
    output: Path,
    cognition_scores: Path,
    cognition_exhaustive_root: Path | None,
    *,
    language_story_math_root: Path | None = None,
    language_behavior_scores: Path | None = None,
    wm_back_condition_root: Path | None = None,
    wm_fixed_candidate_metrics: Path | None = None,
) -> None:
    """Combine Xi, attribution, cognition, and the selected behavioral evidence."""
    summary = load_json(root / "full" / SELECTED / "summary.json")
    archive = np.load(root / "full" / SELECTED / "arrays.npz")
    use_language_behavior = language_story_math_root is not None
    if use_language_behavior != (language_behavior_scores is not None):
        raise ValueError(
            "Language Story/Math result root and behavior table must be provided together"
        )
    if use_language_behavior:
        language_summary = load_json(language_story_math_root / "summary.json")
        language_metrics = np.load(
            language_story_math_root / "fixed_candidates_schaefer1000.npz"
        )
        exhaustive_summary = None
        exhaustive_metrics = None
    else:
        if cognition_exhaustive_root is None:
            raise ValueError("Exploratory cognition results are required for legacy layout")
        exhaustive_summary = load_json(cognition_exhaustive_root / "summary.json")
        exhaustive_metrics = np.load(cognition_exhaustive_root / "metrics.npz")
        language_summary = None
        language_metrics = None
    use_wm_back_condition = wm_back_condition_root is not None
    if use_wm_back_condition != (wm_fixed_candidate_metrics is not None):
        raise ValueError(
            "WM back-condition result root and fixed metrics must be provided together"
        )
    if use_wm_back_condition and not use_language_behavior:
        raise ValueError("WM back-condition panel requires the direct-behavior layout")
    if use_wm_back_condition:
        wm_back_summary = load_json(wm_back_condition_root / "summary.json")
        wm_fixed_metrics = np.load(wm_fixed_candidate_metrics)
    else:
        wm_back_summary = None
        wm_fixed_metrics = None
    states = archive["states"].astype(str).tolist()
    values = np.asarray(archive["system_xi"], dtype=float).T
    network_mean = archive["network_share"].mean(axis=1).T * 100.0
    atom_share_mean = archive["atom_share"].mean(axis=1)
    atom_value = (
        archive["atom_value"]
        if "atom_value" in archive.files
        else archive["atom_share"] * archive["cross_xi"][:, :, None]
    )
    atom_mean = atom_value.mean(axis=1)
    atom_names = archive["atom_names"].astype(str)
    selected = np.argsort(atom_share_mean.mean(axis=0))[::-1][:12]
    atom_panel = atom_mean[:, selected].T
    cognition_panel, cognition_aligned = load_cognition_data(
        cognition_scores, archive["subjects"].astype(str)
    )
    model_subjects = [
        str(subject).removeprefix("sub-") for subject in archive["subjects"].astype(str)
    ]
    if use_language_behavior:
        language_subjects = [
            str(subject).removeprefix("sub-")
            for subject in language_metrics["subjects"].astype(str)
        ]
        if language_subjects != model_subjects:
            raise ValueError("Main-figure and Story/Math subject orders differ")
        if use_wm_back_condition:
            wm_subjects = [
                str(subject).removeprefix("sub-")
                for subject in wm_fixed_metrics["subjects"].astype(str)
            ]
            if wm_subjects != model_subjects:
                raise ValueError("Main-figure and WM subject orders differ")
    else:
        exhaustive_subjects = [
            str(subject).removeprefix("sub-")
            for subject in exhaustive_metrics["subjects"].astype(str)
        ]
        if exhaustive_subjects != model_subjects:
            raise ValueError("Main-figure and exhaustive-search subject orders differ")

    # Asymmetric evidence hierarchy: e-f are the primary cognition panels,
    # while the bottom row contains either exploratory leads or direct task behavior.
    figure = plt.figure(figsize=(14.8, 9.4), constrained_layout=False)
    figure.subplots_adjust(left=0.052, right=0.988, top=0.975, bottom=0.065)
    outer_grid = figure.add_gridspec(
        3,
        1,
        height_ratios=(0.62, 1.12, 0.82),
        hspace=0.34,
    )
    top_grid = outer_grid[0, 0].subgridspec(
        1,
        2,
        width_ratios=(4.15, 1.25),
        wspace=0.18,
    )
    middle_grid = outer_grid[1, 0].subgridspec(
        1,
        4,
        width_ratios=(1.82, 1.20, 0.94, 0.94),
        wspace=0.50,
    )
    overall_axis = figure.add_subplot(top_grid[0, 0])
    cognition_axis = figure.add_subplot(top_grid[0, 1])
    atom_axis = figure.add_subplot(middle_grid[0, 0])
    network_axis = figure.add_subplot(middle_grid[0, 1])
    language_axis = figure.add_subplot(middle_grid[0, 2])
    motor_axis = figure.add_subplot(middle_grid[0, 3], sharey=language_axis)
    bottom_columns = (
        3 if use_wm_back_condition else (2 if use_language_behavior else 3)
    )
    exploratory_grid = outer_grid[2, 0].subgridspec(
        1, bottom_columns, wspace=0.18 if use_language_behavior else 0.16
    )
    domain_axes = tuple(
        figure.add_subplot(exploratory_grid[0, column])
        for column in range(bottom_columns)
    )

    rest_color, task_color = "#4C78A8", "#D07A3A"
    colors = [rest_color] + [task_color] * (len(states) - 1)
    positions = np.arange(len(states), dtype=float)
    box = overall_axis.boxplot(
        values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#303030", "linewidth": 1.1},
        whiskerprops={"color": "#7B8490", "linewidth": 0.75},
        capprops={"color": "#7B8490", "linewidth": 0.75},
    )
    for box_patch, color in zip(box["boxes"], colors):
        box_patch.set(facecolor=color, alpha=0.18, edgecolor=color, linewidth=1.0)
    rng = np.random.default_rng(20260719)
    for index, color in enumerate(colors):
        jitter = rng.uniform(-0.13, 0.13, size=values.shape[0])
        overall_axis.scatter(positions[index] + jitter, values[:, index], s=14, color=color, alpha=0.76, linewidths=0, zorder=3)
    overall_axis.scatter(positions, values.mean(axis=0), marker="D", s=21, facecolor="white", edgecolor="#303030", linewidth=0.7, zorder=4)
    tests = {str(row["task"]): row for row in summary["rest_system_tests"]}
    data_min, data_max = float(values.min()), float(values.max())
    span = max(data_max - data_min, 1.0)
    star_y = data_max + 0.075 * span
    for index, state in enumerate(states[1:], start=1):
        overall_axis.text(index, star_y, significance_stars(float(tests[state]["q"])), ha="center", va="bottom", fontsize=8.5)
    overall_axis.axvline(0.5, color="#A7ADB5", linewidth=0.75, linestyle="--", zorder=0)
    overall_axis.set(
        xticks=positions,
        xticklabels=[DISPLAY_NAMES[state] for state in states],
        xlim=(-0.55, len(states) - 0.45),
        ylim=(data_min - 0.08 * span, star_y + 0.11 * span),
        ylabel=r"System-level $\Xi$ (bits)",
        xlabel="State",
    )
    overall_axis.tick_params(axis="x", labelrotation=22, labelsize=8.0)
    overall_axis.tick_params(axis="y", labelsize=8.0)
    overall_axis.xaxis.label.set_size(8.4)
    overall_axis.yaxis.label.set_size(8.4)
    overall_axis.text(
        0.99,
        1.025,
        f"paired n={values.shape[0]} · vs REST: Wilcoxon, BH-corrected · white diamond: mean",
        transform=overall_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#454545",
        clip_on=False,
    )
    overall_axis.text(0.01, 0.02, "*** q<0.001   ** q<0.01   * q<0.05", transform=overall_axis.transAxes, ha="left", va="bottom", fontsize=7.1, color="#454545")

    lower, upper = float(np.floor(network_mean.min())), float(np.ceil(network_mean.max()))
    image = network_axis.imshow(network_mean, cmap="YlGnBu", vmin=lower, vmax=upper, aspect="auto")
    network_axis.set(xticks=np.arange(8), xticklabels=[DISPLAY_NAMES[state] for state in STATES], yticks=np.arange(7), yticklabels=[NETWORK_LABELS[name] for name in NETWORK_ORDER], xlabel="State (each column sums to 100%)", ylabel="Yeo7 network")
    network_axis.tick_params(axis="x", labelrotation=34, length=0, labelsize=7.3)
    network_axis.tick_params(axis="y", length=0, labelsize=7.3)
    network_axis.xaxis.label.set_size(8.0)
    network_axis.yaxis.label.set_size(8.0)
    network_axis.axvline(0.5, color="#333333", linewidth=0.9)
    for row in range(7):
        for column in range(8):
            value = network_mean[row, column]
            normalized = (value - lower) / max(upper - lower, 1.0e-12)
            network_axis.text(column, row, f"{value:.1f}%", ha="center", va="center", fontsize=6.1, color="white" if normalized > 0.6 else "black")

    atom_upper = max(float(np.quantile(atom_panel, 0.995)), 0.1)
    image = atom_axis.imshow(atom_panel, cmap="magma_r", vmin=0.0, vmax=atom_upper, aspect="auto")
    atom_axis.set(xticks=np.arange(8), xticklabels=[DISPLAY_NAMES[state] for state in STATES], yticks=np.arange(len(selected)), yticklabels=[compact_atom(atom_names[index]) for index in selected], xlabel="State", ylabel="")
    atom_axis.tick_params(axis="x", labelrotation=34, length=0, labelsize=7.1)
    atom_axis.tick_params(axis="y", length=0, labelsize=6.2, pad=1.5)
    atom_axis.xaxis.label.set_size(8.0)
    atom_axis.text(
        0.0,
        1.015,
        "Greedy hierarchy atom",
        transform=atom_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#303030",
    )
    atom_axis.axvline(0.5, color="#F0F0F0", linewidth=0.9)
    for row in range(atom_panel.shape[0]):
        for column in range(8):
            value = atom_panel[row, column]
            atom_axis.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=5.3, color="white" if value > 0.38 * atom_upper else "black")
    atom_colorbar = figure.colorbar(
        image, ax=atom_axis, fraction=0.032, pad=0.022, aspect=32
    )
    atom_colorbar.set_label("Contribution (bits)", fontsize=7.2)
    atom_colorbar.ax.tick_params(labelsize=6.8)

    cognition_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "cognition_diverging", ("#5B8FB9", "#F4F1EF", "#B65F3C")
    )
    image = cognition_axis.imshow(
        cognition_panel,
        cmap=cognition_cmap,
        vmin=-2.5,
        vmax=2.5,
        aspect="auto",
        interpolation="nearest",
    )
    cognition_axis.set(
        xticks=np.arange(len(COGNITION_LABELS)),
        xticklabels=COGNITION_LABELS,
        yticks=[],
        ylabel="Subjects: high → low general cognition",
    )
    cognition_axis.tick_params(axis="x", labelrotation=32, length=0, labelsize=7.4)
    cognition_axis.yaxis.label.set_size(7.8)
    cognition_axis.set_xticks(np.arange(-0.5, len(COGNITION_LABELS), 1), minor=True)
    cognition_axis.set_yticks(np.arange(-0.5, cognition_panel.shape[0], 1), minor=True)
    cognition_axis.grid(which="minor", color="white", linewidth=0.45)
    cognition_axis.tick_params(which="minor", bottom=False, left=False)
    cognition_axis.text(
        0.0,
        1.015,
        "n=29 · each column standardized within sample",
        transform=cognition_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.1,
        color="#454545",
    )
    cognition_colorbar = figure.colorbar(
        image, ax=cognition_axis, fraction=0.045, pad=0.035, aspect=24
    )
    cognition_colorbar.set_label("Within-29 score z", fontsize=7.2)
    cognition_colorbar.ax.tick_params(labelsize=6.8)

    all_network_atom = "+".join(NETWORK_ORDER)
    matching_atoms = np.flatnonzero(atom_names == all_network_atom)
    if len(matching_atoms) != 1:
        raise ValueError(f"Expected one all-network atom, found {len(matching_atoms)}")
    association_x = cognition_aligned["g_score"].to_numpy(dtype=float)
    association_values = {
        state: atom_value[states.index(state), :, int(matching_atoms[0])]
        for state in ("LANGUAGE", "MOTOR")
    }
    guide_x = np.linspace(float(association_x.min()), float(association_x.max()), 200)
    shared_min = min(float(values.min()) for values in association_values.values())
    shared_max = max(float(values.max()) for values in association_values.values())
    shared_pad = 0.06 * max(shared_max - shared_min, 0.1)
    for association_axis, state in zip(
        (language_axis, motor_axis), ("LANGUAGE", "MOTOR"), strict=True
    ):
        association_y = association_values[state]
        association = spearmanr(association_x, association_y)
        association_axis.scatter(
            association_x,
            association_y,
            s=27,
            color="#B65F3C",
            edgecolor="white",
            linewidth=0.5,
            alpha=0.92,
            zorder=3,
        )
        slope, intercept = np.polyfit(association_x, association_y, deg=1)
        association_axis.plot(
            guide_x,
            slope * guide_x + intercept,
            color="#4B5563",
            linewidth=1.0,
            linestyle="--",
            zorder=2,
        )
        association_axis.set(
            xlabel="General cognition score",
            xlim=(float(association_x.min()) - 0.08, float(association_x.max()) + 0.08),
            ylim=(shared_min - shared_pad, shared_max + shared_pad),
        )
        association_axis.set_title(
            state,
            loc="right",
            fontsize=7.8,
            fontweight="bold",
            color="#454545",
            pad=3,
        )
        association_axis.text(
            0.03,
            0.97,
            f"Spearman $\\rho$ = {float(association.statistic):+.3f}\n"
            f"raw two-sided $p$ = {float(association.pvalue):.5f}\n"
            f"n = {len(association_x)}",
            transform=association_axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.4,
        )
        association_axis.tick_params(labelsize=7.7)
        association_axis.xaxis.label.set_size(8.3)
        association_axis.yaxis.label.set_size(8.3)
    language_axis.set_ylabel("All-network hierarchy-atom contribution (bits)")
    motor_axis.tick_params(axis="y", labelleft=False)

    if use_language_behavior:
        coalition = "SomMot+Limbic+Cont"
        coalition_names = language_metrics["coalitions"].astype(str).tolist()
        candidate_index = coalition_names.index(coalition)
        y_values = np.asarray(language_metrics["values"], dtype=float)[
            candidate_index
        ]
        behavior = pd.read_csv(language_behavior_scores, dtype={"Subject": str})
        behavior["Subject"] = behavior["Subject"].str.removeprefix("sub-")
        behavior = behavior.set_index("Subject").loc[model_subjects]
        behavior_specs = (
            (
                "Language_Task_Story_Acc",
                "Story accuracy (%)",
                "#D66A4E",
                "Story",
            ),
            (
                "Language_Task_Math_Acc",
                "Math accuracy (%)",
                "#4C78A8",
                "Math",
            ),
        )
        candidate = next(
            row
            for row in language_summary["schaefer1000"]
            if row["coalition"] == coalition
        )
        shared_min = float(y_values.min())
        shared_max = float(y_values.max())
        shared_pad = 0.08 * max(shared_max - shared_min, 0.1)
        jitter_rng = np.random.default_rng(2026072903)
        for index, (
            domain_axis,
            (column, score_label, color, condition),
        ) in enumerate(zip(domain_axes[:2], behavior_specs, strict=True)):
            x_values = behavior[column].to_numpy(dtype=float)
            association = spearmanr(x_values, y_values)
            expected_rho = float(
                candidate["story_rho"]
                if condition == "Story"
                else candidate["math_rho"]
            )
            expected_p = float(
                candidate["story_p_permutation"]
                if condition == "Story"
                else candidate["math_p_permutation"]
            )
            if not np.isclose(
                float(association.statistic), expected_rho, atol=1.0e-12
            ):
                raise AssertionError(f"{condition} panel rho mismatch")
            jitter_width = 0.45 if condition == "Story" else 0.20
            displayed_x = x_values + jitter_rng.uniform(
                -jitter_width, jitter_width, size=len(x_values)
            )
            domain_axis.scatter(
                displayed_x,
                y_values,
                s=25,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.84,
                zorder=3,
            )
            guide_x = np.linspace(
                float(x_values.min()), float(x_values.max()), 200
            )
            slope, intercept = np.polyfit(x_values, y_values, deg=1)
            domain_axis.plot(
                guide_x,
                slope * guide_x + intercept,
                color=color,
                linewidth=1.0,
                linestyle="--",
                zorder=2,
            )
            x_pad = 0.04 * max(float(np.ptp(x_values)), 1.0)
            domain_axis.set(
                xlabel=score_label,
                xlim=(
                    float(x_values.min()) - x_pad,
                    float(x_values.max()) + x_pad,
                ),
                ylim=(shared_min - shared_pad, shared_max + shared_pad),
            )
            domain_axis.set_title(
                f"LANGUAGE · {short_atom(coalition)} · {condition}",
                loc="right",
                fontsize=7.6,
                fontweight="bold",
                color="#454545",
                pad=3,
            )
            domain_axis.text(
                0.02,
                0.97,
                f"Spearman $\\rho$ = {expected_rho:+.3f}\n"
                f"pointwise permutation $p$ = {expected_p:.4f}\n"
                f"n = {len(x_values)}",
                transform=domain_axis.transAxes,
                ha="left",
                va="top",
                fontsize=7.3,
                color="#303942",
            )
            domain_axis.grid(color="#E8EBED", linewidth=0.55, zorder=0)
            domain_axis.tick_params(labelsize=7.7)
            domain_axis.xaxis.label.set_size(8.3)
            domain_axis.yaxis.label.set_size(8.3)
        domain_axes[0].set_ylabel("Fixed-coalition synergy (bits)")
        domain_axes[1].tick_params(axis="y", labelleft=False)
        if use_wm_back_condition:
            wm_axis = domain_axes[2]
            validation = wm_back_summary["schaefer1000_validation"]
            wm_states = wm_fixed_metrics["states"].astype(str).tolist()
            wm_coalitions = wm_fixed_metrics["coalitions"].astype(str).tolist()
            wm_index = wm_states.index("WM")
            if wm_coalitions[wm_index] != "Cont+Default":
                raise ValueError("Unexpected frozen Schaefer-1000 WM coalition")
            wm_values_all = np.asarray(
                wm_fixed_metrics["values"][wm_index], dtype=float
            )
            wm_complete = behavior["WM_Task_0bk_Acc"].notna() & behavior[
                "WM_Task_2bk_Acc"
            ].notna()
            wm_x = behavior.loc[wm_complete, "WM_Task_0bk_Acc"].to_numpy(
                dtype=float
            )
            wm_y = wm_values_all[np.asarray(wm_complete, dtype=bool)]
            wm_association = spearmanr(wm_x, wm_y)
            if not np.isclose(
                float(wm_association.statistic),
                float(validation["rho_0back"]),
                atol=1.0e-12,
            ):
                raise AssertionError("WM 0-back panel rho mismatch")
            wm_jitter = np.random.default_rng(2026073101).uniform(
                -0.24, 0.24, size=len(wm_x)
            )
            wm_color = "#8B6BAE"
            wm_axis.scatter(
                wm_x + wm_jitter,
                wm_y,
                s=25,
                color=wm_color,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.86,
                zorder=3,
            )
            wm_guide_x = np.linspace(float(wm_x.min()), float(wm_x.max()), 200)
            wm_slope, wm_intercept = np.polyfit(wm_x, wm_y, deg=1)
            wm_axis.plot(
                wm_guide_x,
                wm_slope * wm_guide_x + wm_intercept,
                color=wm_color,
                linewidth=1.0,
                linestyle="--",
                zorder=2,
            )
            wm_x_pad = 0.04 * max(float(np.ptp(wm_x)), 1.0)
            wm_y_pad = 0.08 * max(float(np.ptp(wm_y)), 0.1)
            wm_axis.set(
                xlabel="0-back accuracy (%)",
                ylabel="Fixed-coalition synergy (bits)",
                xlim=(
                    float(wm_x.min()) - wm_x_pad,
                    float(wm_x.max()) + wm_x_pad,
                ),
                ylim=(
                    float(wm_y.min()) - wm_y_pad,
                    float(wm_y.max()) + wm_y_pad,
                ),
            )
            wm_axis.set_title(
                "WM · Cont+Default · 0-back",
                loc="right",
                fontsize=7.6,
                fontweight="bold",
                color="#454545",
                pad=3,
            )
            wm_axis.text(
                0.98,
                0.97,
                f"0-back $\\rho$ = {float(validation['rho_0back']):+.3f}\n"
                f"two-condition max-T $p$ = "
                f"{float(validation['p_max_t_0back']):.4f}\n"
                f"n = {len(wm_x)}",
                transform=wm_axis.transAxes,
                ha="right",
                va="top",
                fontsize=7.3,
                color="#303942",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 1.5,
                },
            )
            wm_axis.text(
                0.02,
                0.72,
                f"2-back $\\rho$ = {float(validation['rho_2back']):+.3f}\n"
                f"$\\Delta\\rho$ = "
                f"{float(validation['delta_rho_2back_minus_0back']):+.3f}"
                f" · paired $p$ = "
                f"{float(validation['p_condition_difference']):.4f}",
                transform=wm_axis.transAxes,
                ha="left",
                va="top",
                fontsize=6.9,
                color="#4B5563",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 1.5,
                },
            )
            wm_axis.grid(color="#E8EBED", linewidth=0.55, zorder=0)
            wm_axis.tick_params(labelsize=7.7)
            wm_axis.xaxis.label.set_size(8.3)
            wm_axis.yaxis.label.set_size(8.3)
    else:
        exhaustive_states = exhaustive_metrics["states"].astype(str).tolist()
        exhaustive_coalitions = (
            exhaustive_metrics["coalitions"].astype(str).tolist()
        )
        domain_specs = (
            ("cry_score", "Crystallized cognition"),
            ("mem_score", "Memory"),
            ("spd_score", "Processing speed"),
        )
        for domain_axis, (score_name, score_label) in zip(
            domain_axes, domain_specs, strict=True
        ):
            candidate = exhaustive_summary["scores"][score_name][
                "selected_full_sample"
            ]
            metric = str(candidate["metric"])
            state = str(candidate["state"])
            coalition = str(candidate["coalition"])
            x_values = cognition_aligned[score_name].to_numpy(dtype=float)
            y_values = np.asarray(exhaustive_metrics[metric], dtype=float)[
                exhaustive_states.index(state),
                :,
                exhaustive_coalitions.index(coalition),
            ]
            association = spearmanr(x_values, y_values)
            if not np.isclose(
                float(association.statistic), float(candidate["rho"]), atol=1.0e-12
            ):
                raise AssertionError(f"Domain-panel rho mismatch for {score_name}")
            domain_axis.set_facecolor("#FAFAFA")
            domain_axis.scatter(
                x_values,
                y_values,
                s=19,
                color="#C58A70",
                edgecolor="white",
                linewidth=0.4,
                alpha=0.78,
                zorder=3,
            )
            guide_x = np.linspace(float(x_values.min()), float(x_values.max()), 200)
            slope, intercept = np.polyfit(x_values, y_values, deg=1)
            domain_axis.plot(
                guide_x,
                slope * guide_x + intercept,
                color="#7B8490",
                linewidth=0.8,
                linestyle="--",
                zorder=2,
            )
            x_pad = 0.06 * max(float(np.ptp(x_values)), 0.1)
            y_pad = 0.08 * max(float(np.ptp(y_values)), 0.1)
            domain_axis.set(
                xlabel=score_label,
                xlim=(
                    float(x_values.min()) - x_pad,
                    float(x_values.max()) + x_pad,
                ),
                ylim=(
                    float(y_values.min()) - y_pad,
                    float(y_values.max()) + y_pad,
                ),
            )
            domain_axis.set_title(
                f"{state} · {short_atom(coalition)}",
                loc="right",
                fontsize=7.3,
                fontweight="normal",
                color="#525A65",
                pad=2,
            )
            domain_axis.text(
                0.02,
                0.97,
                f"$\\rho$ = {float(candidate['rho']):+.3f} · "
                f"raw $p$ = {float(candidate['p_raw_two_sided']):.3g} · "
                f"perm $p$ = {float(candidate['p_permutation_pointwise']):.3g}\n"
                "n = 29",
                transform=domain_axis.transAxes,
                ha="left",
                va="top",
                fontsize=7.0,
                color="#4B5563",
            )
            domain_axis.tick_params(labelsize=7.7)
            domain_axis.xaxis.label.set_size(8.3)
            domain_axis.yaxis.label.set_size(8.3)
        domain_axes[0].set_ylabel(r"$\mathrm{Syn}_{EID}$ (bits)")

    axes = (
        overall_axis,
        atom_axis,
        network_axis,
        cognition_axis,
        language_axis,
        motor_axis,
        *domain_axes,
    )
    panel_x_positions = (
        (-0.05, -0.09, -0.09, -0.17, -0.09, -0.09)
        + tuple(-0.05 if use_language_behavior else -0.09 for _ in domain_axes)
    )
    for label, axis, x_position in zip(
        "abcdefghijklmnopqrstuvwxyz"[:len(axes)],
        axes,
        panel_x_positions,
        strict=True,
    ):
        axis.text(
            x_position,
            1.025,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10.5,
        )
    save_figure(figure, output / "task_evoked_xi_main_combined")


def significance_stars(q_value: float) -> str:
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return "ns"


def plot_overall_phi(root: Path, output: Path) -> None:
    """Plot the historical overall-Phi quantity, denoted system-level Xi here."""
    summary = load_json(root / "full" / SELECTED / "summary.json")
    archive = np.load(root / "full" / SELECTED / "arrays.npz")
    states = archive["states"].astype(str).tolist()
    values = np.asarray(archive["system_xi"], dtype=float).T
    if values.shape[1] != len(states):
        raise ValueError("system_xi must have one column per state after transposition")

    rest_color = "#4C78A8"
    task_color = "#D07A3A"
    colors = [rest_color] + [task_color] * (len(states) - 1)
    positions = np.arange(len(states), dtype=float)
    figure, axis = plt.subplots(figsize=(7.8, 4.2), constrained_layout=True)
    box = axis.boxplot(
        values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#303030", "linewidth": 1.1},
        whiskerprops={"color": "#7B8490", "linewidth": 0.75},
        capprops={"color": "#7B8490", "linewidth": 0.75},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set(facecolor=color, alpha=0.18, edgecolor=color, linewidth=1.0)

    rng = np.random.default_rng(20260719)
    for index, color in enumerate(colors):
        jitter = rng.uniform(-0.13, 0.13, size=values.shape[0])
        axis.scatter(
            positions[index] + jitter,
            values[:, index],
            s=13,
            color=color,
            alpha=0.76,
            linewidths=0,
            zorder=3,
        )
    axis.scatter(
        positions,
        values.mean(axis=0),
        marker="D",
        s=19,
        facecolor="white",
        edgecolor="#303030",
        linewidth=0.7,
        zorder=4,
    )

    tests = {str(row["task"]): row for row in summary["rest_system_tests"]}
    data_min = float(values.min())
    data_max = float(values.max())
    span = max(data_max - data_min, 1.0)
    star_y = data_max + 0.075 * span
    for index, state in enumerate(states[1:], start=1):
        axis.text(index, star_y, significance_stars(float(tests[state]["q"])), ha="center", va="bottom", fontsize=8)

    axis.axvline(0.5, color="#A7ADB5", linewidth=0.75, linestyle="--", zorder=0)
    axis.set(
        xticks=positions,
        xticklabels=[DISPLAY_NAMES[state] for state in states],
        xlim=(-0.55, len(states) - 0.45),
        ylim=(data_min - 0.08 * span, star_y + 0.11 * span),
        ylabel=r"Overall $\Phi$ / system-level $\Xi$ (bits)",
        xlabel="State",
    )
    axis.tick_params(axis="x", labelrotation=32)
    axis.set_title(
        f"paired n={values.shape[0]} · vs REST: Wilcoxon, BH-corrected · white diamond: mean",
        loc="right",
        fontsize=6.3,
        color="#454545",
        pad=5,
    )
    axis.text(
        0.01,
        0.02,
        "*** q<0.001   ** q<0.01   * q<0.05",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#454545",
    )
    save_figure(figure, output / "overall_phi_rest_task_scatter_box")

    note_lines = [
        "# Overall Phi / system-level Xi scatter-box figure",
        "",
        f"Selected configuration: `{SELECTED}`; paired subjects: n={values.shape[0]}.",
        "",
        "The plotted quantity is the historical overall-Phi output, denoted system-level Xi in the current manuscript. Boxes show the interquartile range and median; points are subjects; white diamonds are arithmetic means. Tests are paired two-sided Wilcoxon signed-rank tests comparing REST with each task, followed by Benjamini-Hochberg correction across seven contrasts.",
        "",
        "| Task | REST minus task (bits) | BH q |",
        "|---|---:|---:|",
    ]
    for state in states[1:]:
        row = tests[state]
        note_lines.append(f"| {DISPLAY_NAMES[state]} | {float(row['rest_minus_task_mean_bits']):.4f} | {float(row['q']):.3g} |")
    (output / "overall_phi_rest_task_scatter_box_notes.md").write_text("\n".join(note_lines) + "\n", encoding="utf-8")


def plot_tuning(root: Path, output: Path) -> None:
    search = collect_search_summaries(root)
    full = {identifier: load_json(root / "full" / identifier / "summary.json") for identifier in COMPARATORS}
    confirm = {identifier: load_json(root / "confirm" / identifier / "summary.json") for identifier in (BASELINE, SELECTED, "k1_p3_a0.3")}
    colors = {1: "#4C78A8", 2: "#E08B45", 3: "#70A37F", 4: "#9A77B6"}
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), constrained_layout=True)

    for k in sorted({int(row["params"]["k"]) for row in search}):
        subset = [row for row in search if int(row["params"]["k"]) == k]
        axes[0, 0].scatter(
            [row["network_all"]["between_within_ratio"] for row in subset],
            [row["atom_all"]["between_within_ratio"] for row in subset],
            s=22,
            marker=markers[k],
            color=colors[k],
            alpha=0.72,
            label=f"k={k}",
        )
    lookup = {row["config_id"]: row for row in search}
    for identifier, text, color in ((BASELINE, "baseline", "#333333"), (SELECTED, "selected", "#C83E4D")):
        row = lookup[identifier]
        x = row["network_all"]["between_within_ratio"]
        y = row["atom_all"]["between_within_ratio"]
        axes[0, 0].scatter([x], [y], s=70, facecolor="none", edgecolor=color, linewidth=1.5, zorder=5)
        axes[0, 0].annotate(text, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=6.2, color=color)
    axes[0, 0].set(xlabel="Network between/within TV ratio", ylabel="Atom between/within TV ratio")
    axes[0, 0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    identifiers = list(COMPARATORS)
    positions = np.arange(len(identifiers))
    width = 0.34
    axes[0, 1].bar(positions - width / 2, [full[x]["network_all"]["between_within_ratio"] for x in identifiers], width, color="#4C78A8", label="Network")
    axes[0, 1].bar(positions + width / 2, [full[x]["atom_all"]["between_within_ratio"] for x in identifiers], width, color="#E08B45", label="Hierarchy atom")
    axes[0, 1].set(xticks=positions, xticklabels=identifiers, ylabel="Between/within TV ratio")
    axes[0, 1].tick_params(axis="x", labelrotation=35)
    axes[0, 1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    base, chosen = full[BASELINE], full[SELECTED]
    categories = ["Network\nall", "Network\ntasks", "Atom\nall", "Atom\ntasks"]
    base_counts = [base["significance"][key] for key in ("network_pairs_all", "network_pairs_tasks", "atom_pairs_all", "atom_pairs_tasks")]
    chosen_counts = [chosen["significance"][key] for key in ("network_pairs_all", "network_pairs_tasks", "atom_pairs_all", "atom_pairs_tasks")]
    x = np.arange(4)
    axes[1, 0].bar(x - width / 2, base_counts, width, color="#B7C1CC", label="Baseline")
    axes[1, 0].bar(x + width / 2, chosen_counts, width, color="#C83E4D", label="Selected")
    axes[1, 0].set(xticks=x, xticklabels=categories, ylabel="Significant state pairs (BH q<0.05)", ylim=(0, 28))
    axes[1, 0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    for identifier, color, marker in ((BASELINE, "#777777", "o"), (SELECTED, "#C83E4D", "s"), ("k1_p3_a0.3", "#4C78A8", "^")):
        row = confirm[identifier]
        axes[1, 1].scatter(row["rest_min_mean_margin_bits"], row["network_all"]["between_within_ratio"], s=42, color=color, marker=marker, label=identifier)
    axes[1, 1].axvline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    axes[1, 1].set(xlabel=r"Minimum REST $-$ task $\Xi$ (bits)", ylabel="Network between/within TV ratio")
    axes[1, 1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    for label, axis in zip("abcd", axes.ravel()):
        axis.text(-0.13, 1.05, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    save_figure(figure, output / "parameter_tuning_comparison")


def percentage_gain(selected: float, baseline: float) -> float:
    return 100.0 * (selected / baseline - 1.0)


def write_report(root: Path, output: Path) -> None:
    baseline = load_json(root / "full" / BASELINE / "summary.json")
    selected = load_json(root / "full" / SELECTED / "summary.json")
    confirm_base = load_json(root / "confirm" / BASELINE / "summary.json")
    confirm_selected = load_json(root / "confirm" / SELECTED / "summary.json")
    lines = [
        "# HCP 任务诱发 PCA–Xi 参数搜索结果",
        "",
        "最终选择 `k=1, p=3, alpha=1`。参数首先在固定8名被试上筛选，再在未参与筛选的21名被试上独立确认；最后以29名完整配对被试汇总效应。LOSO 未进入目标函数或结论。",
        "",
        "EI 使用线性高斯 affine-TM/log-det 连续估计。任务态 PCA 在 `taskRetained-taskRegressed` 上拟合并投影 `taskRetained`；REST 在自身时序上拟合与投影。除 k、p、alpha 外，数据、分割、Yeo7 边界、EI 估计和层级算法均固定。",
        "",
        "## 独立确认",
        "",
        f"在21名未参与筛选的被试中，network ratio 从 {confirm_base['network_all']['between_within_ratio']:.3f} 提高到 {confirm_selected['network_all']['between_within_ratio']:.3f}，atom ratio 从 {confirm_base['atom_all']['between_within_ratio']:.3f} 提高到 {confirm_selected['atom_all']['between_within_ratio']:.3f}。网络显著状态对由 {confirm_base['significance']['network_pairs_all']}/28 增至 {confirm_selected['significance']['network_pairs_all']}/28，atom 显著状态对由 {confirm_base['significance']['atom_pairs_all']}/28 增至 {confirm_selected['significance']['atom_pairs_all']}/28。",
        "",
        "## 29人完整汇总",
        "",
        "| Metric | Baseline k1-p5-a10 | Selected k1-p3-a1 | Change |",
        "|---|---:|---:|---:|",
    ]
    metrics = [
        ("Network ratio, all states", "network_all", "between_within_ratio"),
        ("Network ratio, tasks", "network_tasks", "between_within_ratio"),
        ("Atom ratio, all states", "atom_all", "between_within_ratio"),
        ("Atom ratio, tasks", "atom_tasks", "between_within_ratio"),
    ]
    for label, group, key in metrics:
        old, new = float(baseline[group][key]), float(selected[group][key])
        lines.append(f"| {label} | {old:.3f} | {new:.3f} | {percentage_gain(new, old):+.1f}% |")
    lines.extend(
        [
            f"| Network significant pairs, all | {baseline['significance']['network_pairs_all']}/28 | {selected['significance']['network_pairs_all']}/28 | +{selected['significance']['network_pairs_all']-baseline['significance']['network_pairs_all']} |",
            f"| Atom significant pairs, all | {baseline['significance']['atom_pairs_all']}/28 | {selected['significance']['atom_pairs_all']}/28 | +{selected['significance']['atom_pairs_all']-baseline['significance']['atom_pairs_all']} |",
            f"| Mean held-out skill ratio | {baseline['heldout_skill_ratio_mean']:.3f} | {selected['heldout_skill_ratio_mean']:.3f} | {selected['heldout_skill_ratio_mean']-baseline['heldout_skill_ratio_mean']:+.3f} |",
            "",
            "## 系统 Xi 与 REST 约束",
            "",
            "| State | Mean Xi (bits) | REST minus task | BH q |",
            "|---|---:|---:|---:|",
            f"| REST | {selected['system_xi_mean_bits']['REST']:.4f} | — | — |",
        ]
    )
    rest_lookup = {row["task"]: row for row in selected["rest_system_tests"]}
    for task in TASKS:
        row = rest_lookup[task]
        lines.append(f"| {DISPLAY_NAMES[task]} | {selected['system_xi_mean_bits'][task]:.4f} | {row['rest_minus_task_mean_bits']:+.4f} | {row['q']:.3g} |")
    lines.extend(
        [
            "",
            "REST 均值高于全部七任务，且七个被试内 Wilcoxon 对比经 BH 校正后均达到 q<0.05，因此保留了 REST 整体 Xi 显著高于所有任务态的原结论。",
            "",
            "## 解释边界",
            "",
            f"最大 Xi 分解闭合误差为 {selected['diagnostics']['max_identity_error_bits']:.3e} bits；network/atom share 闭合误差均低于 1e-15。选择配置的 held-out skill ratio 均值为 {selected['heldout_skill_ratio_mean']:.3f}，{selected['models_better_than_persistence']}/{selected['n_models']} 个模型优于 persistence。",
            "",
            "降低 alpha 会增强状态分离，但同时削弱 REST 相对任务的总 Xi 余量并略降低预测稳定性；alpha=0.3 虽有更大的分离 ratio，却不能保持 REST–Social 显著。最终 alpha=1 因此是分离度、REST结论和预测诊断之间的折中，而不是单纯最大化图面差异。greedy atom 仍具有路径依赖性，显著组合应解释为当前固定层级算法下可重复的组合归因，而不是唯一真实脑网络层级。",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_tuning(args.results_root, args.output_dir)
    plot_selected(args.results_root, args.output_dir)
    plot_overall_phi(args.results_root, args.output_dir)
    plot_main_combined(
        args.results_root,
        args.output_dir,
        args.cognition_scores,
        args.cognition_exhaustive_root,
    )
    write_report(args.results_root, args.output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cognition-scores", type=Path, default=DEFAULT_COGNITION_SCORES)
    parser.add_argument(
        "--cognition-exhaustive-root",
        type=Path,
        default=DEFAULT_COGNITION_EXHAUSTIVE_ROOT,
    )
    args = parser.parse_args(argv)
    run(args)
    print(json.dumps({"output_dir": str(args.output_dir), "selected": SELECTED}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
