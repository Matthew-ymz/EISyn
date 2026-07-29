#!/usr/bin/env python3
"""Visualize five HCP task-performance scores for the 29 REST-task subjects.

Figure contract
---------------
Core conclusion: subject-level performance and the equal-weight overall score
can be compared without discarding individual variation.
Archetype: quantitative grid.
Backend: Python/matplotlib.
Hero evidence: five-task subject heatmap.
Supporting evidence: aligned overall-score dot plot.
Reviewer risk: Social is derived rather than an official overall-accuracy field;
the definition is stated directly on the figure.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_FILE = ROOT / "Data" / "unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
SUBJECT_ARCHIVE = (
    ROOT
    / "results"
    / "hcp_schaefer500_fixed_hierarchy_tm_peid"
    / "fixed_hierarchy_tm_peid.npz"
)
OUTPUT_DIR = ROOT / "fig" / "hcp_task_behavior"
OUTPUT_STEM = OUTPUT_DIR / "hcp_29_subject_task_performance"

TASK_LABELS = ("Emotion", "Language", "Relational", "Social*", "Working\nmemory")
DIRECT_FIELDS = (
    "Emotion_Task_Acc",
    "Language_Task_Acc",
    "Relational_Task_Acc",
    "WM_Task_Acc",
)


def load_scores() -> tuple[list[str], np.ndarray]:
    archive = np.load(SUBJECT_ARCHIVE)
    subjects = [str(value).removeprefix("sub-") for value in archive["common_subjects"]]

    with BEHAVIOR_FILE.open(newline="", encoding="utf-8-sig") as handle:
        rows = {
            row["Subject"]: row
            for row in csv.DictReader(handle)
            if row["Subject"] in set(subjects)
        }
    missing = [subject for subject in subjects if subject not in rows]
    if missing:
        raise ValueError(f"Behavioral rows missing for subjects: {missing}")

    scores = np.empty((len(subjects), len(TASK_LABELS)), dtype=float)
    for index, subject in enumerate(subjects):
        row = rows[subject]
        scores[index, 0] = float(row[DIRECT_FIELDS[0]])
        scores[index, 1] = float(row[DIRECT_FIELDS[1]])
        scores[index, 2] = float(row[DIRECT_FIELDS[2]])
        scores[index, 3] = 0.5 * (
            float(row["Social_Task_Random_Perc_Random"])
            + float(row["Social_Task_TOM_Perc_TOM"])
        )
        scores[index, 4] = float(row[DIRECT_FIELDS[3]])

    if not np.isfinite(scores).all():
        raise ValueError("The five selected task scores must be finite for all 29 subjects.")
    return subjects, scores


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.2,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def plot(subjects: list[str], scores: np.ndarray) -> plt.Figure:
    overall = scores.mean(axis=1)
    order = np.argsort(-overall, kind="stable")
    sorted_scores = scores[order]
    sorted_overall = overall[order]
    sorted_subjects = [subjects[index] for index in order]

    figure = plt.figure(figsize=(7.2, 8.0))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(4.9, 2.0),
        left=0.15,
        right=0.97,
        bottom=0.095,
        top=0.89,
        wspace=0.13,
    )
    heatmap_ax = figure.add_subplot(grid[0, 0])
    overall_ax = figure.add_subplot(grid[0, 1], sharey=heatmap_ax)

    vmin, vmax = 40.0, 100.0
    colormap = mpl.colormaps["YlGnBu"]
    image = heatmap_ax.imshow(
        sorted_scores,
        aspect="auto",
        interpolation="nearest",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
    )
    heatmap_ax.set_xticks(np.arange(len(TASK_LABELS)), TASK_LABELS)
    heatmap_ax.xaxis.tick_top()
    heatmap_ax.tick_params(axis="x", length=0, pad=7)
    heatmap_ax.set_yticks(
        np.arange(len(sorted_subjects)),
        [f"sub-{subject}" for subject in sorted_subjects],
    )
    heatmap_ax.tick_params(axis="y", length=0, pad=4)
    heatmap_ax.set_ylabel("Subject")
    heatmap_ax.set_xticks(np.arange(-0.5, len(TASK_LABELS), 1), minor=True)
    heatmap_ax.set_yticks(np.arange(-0.5, len(sorted_subjects), 1), minor=True)
    heatmap_ax.grid(which="minor", color="white", linewidth=0.8)
    heatmap_ax.tick_params(which="minor", bottom=False, left=False)
    for spine in heatmap_ax.spines.values():
        spine.set_visible(False)

    normalized = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    for row in range(sorted_scores.shape[0]):
        for column in range(sorted_scores.shape[1]):
            value = sorted_scores[row, column]
            text_color = "white" if normalized(value) > 0.61 else "#1F2933"
            heatmap_ax.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=5.4,
            )

    y_positions = np.arange(len(sorted_subjects))
    overall_ax.axvline(
        float(overall.mean()),
        color="#6B7280",
        linestyle=(0, (3, 2)),
        linewidth=0.9,
        zorder=1,
    )
    overall_ax.scatter(
        sorted_overall,
        y_positions,
        c=sorted_overall,
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        s=24,
        edgecolor="#263238",
        linewidth=0.35,
        zorder=3,
    )
    for y_position, value in zip(y_positions, sorted_overall, strict=True):
        overall_ax.text(
            value + 0.7,
            y_position,
            f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=5.7,
            color="#263238",
        )
    lower = max(55.0, np.floor(sorted_overall.min() / 5.0) * 5.0 - 2.0)
    overall_ax.set_xlim(lower, 103.0)
    overall_ax.set_xlabel("Overall performance (%)")
    overall_ax.xaxis.set_label_position("top")
    overall_ax.xaxis.tick_top()
    overall_ax.tick_params(axis="x", pad=4)
    overall_ax.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    overall_ax.grid(axis="x", color="#D9DEE3", linewidth=0.55)
    overall_ax.set_axisbelow(True)
    overall_ax.spines["left"].set_visible(False)
    overall_ax.spines["bottom"].set_visible(False)
    overall_ax.text(
        float(overall.mean()),
        0.985,
        f"mean {overall.mean():.1f}",
        transform=overall_ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=6,
        color="#59636E",
    )

    colorbar_ax = figure.add_axes((0.19, 0.935, 0.38, 0.014))
    colorbar = figure.colorbar(image, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_label("Task performance (%)", labelpad=2)
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.tick_params(length=2, pad=1)
    colorbar.outline.set_linewidth(0.5)

    heatmap_ax.text(
        -0.14, 1.04, "a", transform=heatmap_ax.transAxes, weight="bold", fontsize=8
    )
    overall_ax.text(
        -0.14, 1.04, "b", transform=overall_ax.transAxes, weight="bold", fontsize=8
    )
    figure.text(
        0.15,
        0.025,
        "Social* = mean of condition-congruent response percentages. "
        "Overall = unweighted mean of the five task scores. n = 29.",
        ha="left",
        va="bottom",
        fontsize=6,
        color="#4B5563",
    )
    return figure


def main() -> None:
    configure_style()
    subjects, scores = load_scores()
    figure = plot(subjects, scores)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(f"{OUTPUT_STEM}.png", dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(f"{OUTPUT_STEM}.svg", bbox_inches="tight", facecolor="white")
    figure.savefig(f"{OUTPUT_STEM}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved {OUTPUT_STEM}.png/.svg/.pdf")


if __name__ == "__main__":
    main()
