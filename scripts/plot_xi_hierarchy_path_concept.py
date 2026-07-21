#!/usr/bin/env python3
"""Draw a four-source schematic for the current PEID Xi hierarchy."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fig" / "xi_hierarchy_visual_path"

INK = "#252A34"
MUTED = "#727984"
LIGHT = "#E8EBEF"
BLUE = "#4C78A8"
BLUE_LIGHT = "#DCE8F3"
GREEN = "#6F9F73"
GREEN_LIGHT = "#E1EFE2"
ORANGE = "#E08B45"
ORANGE_LIGHT = "#F8E6D6"
PURPLE = "#8064A2"


def rounded_box(axis, xy, width, height, *, facecolor, edgecolor, linewidth=1.4, radius=0.025):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    axis.add_patch(box)
    return box


def arrow(axis, start, end, *, color=MUTED, linewidth=1.3, linestyle="-", mutation=10):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    axis.add_patch(patch)
    return patch


def node(axis, center, label, *, color=INK, radius=0.052, fontsize=8):
    axis.add_patch(Circle(center, radius, facecolor=color, edgecolor="white", linewidth=1.2, zorder=3))
    axis.text(*center, label, ha="center", va="center", color="white", fontsize=fontsize, fontweight="bold", zorder=4)


def panel_label(axis, label):
    axis.text(-0.03, 1.02, label, transform=axis.transAxes, fontsize=15, fontweight="bold", va="top", color=INK)


def setup_axis(axis):
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")


def draw_panel_a(axis):
    panel_label(axis, "A")
    axis.text(0.50, 0.94, "Joint effect from four sources", ha="center", va="top", fontsize=11, fontweight="bold", color=INK)

    rounded_box(axis, (0.07, 0.32), 0.43, 0.46, facecolor="#F6F7F9", edgecolor="#AAB0B8", linewidth=1.2)
    axis.text(0.285, 0.74, "four sources act together", ha="center", va="center", fontsize=8, color=INK)
    positions = {"1": (0.18, 0.61), "2": (0.39, 0.61), "3": (0.18, 0.43), "4": (0.39, 0.43)}
    for label, position in positions.items():
        node(axis, position, label, color="#3D6B3D")

    rounded_box(axis, (0.68, 0.47), 0.23, 0.18, facecolor=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=1.5)
    axis.text(0.795, 0.575, "future target", ha="center", va="center", fontsize=7, color=MUTED)
    axis.text(0.795, 0.525, "brain state", ha="center", va="center", fontsize=9, color=INK, fontweight="bold")
    arrow(axis, (0.50, 0.55), (0.68, 0.55), color=INK, linewidth=2.0, mutation=12)
    axis.text(0.59, 0.605, "together", ha="center", va="bottom", fontsize=8, color=INK, fontweight="bold")

    rounded_box(axis, (0.08, 0.07), 0.83, 0.17, facecolor="white", edgecolor=LIGHT, linewidth=1.0)
    axis.text(0.12, 0.190, "Together", ha="left", va="center", fontsize=7.4, color=INK, fontweight="bold")
    rounded_box(axis, (0.30, 0.172), 0.50, 0.036, facecolor=INK, edgecolor=INK, linewidth=0.8, radius=0.008)
    axis.text(0.12, 0.125, "Singles added", ha="left", va="center", fontsize=7.4, color=MUTED)
    for index, color in enumerate((BLUE, GREEN, PURPLE, "#C96B65")):
        rounded_box(axis, (0.30 + index * 0.086, 0.107), 0.068, 0.036, facecolor=color, edgecolor=color, linewidth=0.8, radius=0.008)
    axis.plot([0.66, 0.80], [0.125, 0.125], color=ORANGE, linewidth=2.2)
    axis.plot([0.80, 0.80], [0.125, 0.190], color=ORANGE, linewidth=1.4)
    axis.text(0.82, 0.156, r"joint-only effect  $\Xi$", ha="left", va="center", fontsize=7.3, color=ORANGE, fontweight="bold")
    axis.text(0.50, 0.035, "what the group adds beyond the four sources separately", ha="center", va="center", fontsize=7, color=MUTED)


def tree_box(axis, center, width, height, title, subtitle, *, facecolor, edgecolor):
    x, y = center
    rounded_box(axis, (x - width / 2, y - height / 2), width, height, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.35)
    axis.text(x, y + 0.018, title, ha="center", va="center", fontsize=8.5, color=INK, fontweight="bold")
    axis.text(x, y - 0.034, subtitle, ha="center", va="center", fontsize=7, color=MUTED)


def draw_panel_b(axis):
    panel_label(axis, "B")
    axis.text(0.50, 0.94, "Split the joint effect into visual blocks", ha="center", va="top", fontsize=11, fontweight="bold", color=INK)

    tree_box(axis, (0.50, 0.82), 0.40, 0.12, "All four sources", r"system joint effect  $\Xi$", facecolor="#EEF1F5", edgecolor=INK)
    tree_box(axis, (0.27, 0.53), 0.29, 0.12, "Group 1 + 2", "joint effect left inside", facecolor=BLUE_LIGHT, edgecolor=BLUE)
    tree_box(axis, (0.73, 0.53), 0.29, 0.12, "Group 3 + 4", "joint effect left inside", facecolor=GREEN_LIGHT, edgecolor=GREEN)

    arrow(axis, (0.45, 0.76), (0.31, 0.60), color=BLUE, linewidth=1.6)
    arrow(axis, (0.55, 0.76), (0.69, 0.60), color=GREEN, linewidth=1.6)
    axis.text(0.57, 0.68, "keep the most effect inside both groups", ha="center", va="center", fontsize=6.8, color=MUTED)

    rounded_box(axis, (0.005, 0.585), 0.205, 0.09, facecolor=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=1.2)
    axis.text(0.1075, 0.642, "cross-group synergy", ha="center", va="center", fontsize=6.2, color=ORANGE, fontweight="bold")
    axis.text(0.1075, 0.605, "belongs only to both groups", ha="center", va="center", fontsize=5.5, color=INK)
    arrow(axis, (0.39, 0.78), (0.21, 0.65), color=ORANGE, linewidth=1.1, linestyle="--", mutation=8)

    leaf_positions = [0.14, 0.39, 0.61, 0.86]
    for index, x in enumerate(leaf_positions, start=1):
        node(axis, (x, 0.25), str(index), color=BLUE if index <= 2 else GREEN, radius=0.043, fontsize=7)
        axis.text(x, 0.17, "single source", ha="center", va="center", fontsize=6.2, color=MUTED)
    arrow(axis, (0.23, 0.47), (0.15, 0.30), color=BLUE)
    arrow(axis, (0.31, 0.47), (0.38, 0.30), color=BLUE)
    arrow(axis, (0.69, 0.47), (0.62, 0.30), color=GREEN)
    arrow(axis, (0.77, 0.47), (0.85, 0.30), color=GREEN)

    rounded_box(axis, (0.18, 0.33), 0.18, 0.072, facecolor=BLUE_LIGHT, edgecolor=BLUE, linewidth=1.1)
    axis.text(0.27, 0.366, "within-pair synergy", ha="center", va="center", fontsize=6.6, color=BLUE, fontweight="bold")
    rounded_box(axis, (0.64, 0.33), 0.18, 0.072, facecolor=GREEN_LIGHT, edgecolor=GREEN, linewidth=1.1)
    axis.text(0.73, 0.366, "within-pair synergy", ha="center", va="center", fontsize=6.6, color=GREEN, fontweight="bold")

    rounded_box(axis, (0.16, 0.025), 0.68, 0.11, facecolor="white", edgecolor=LIGHT, linewidth=1.0)
    axis.text(0.50, 0.115, r"system joint effect  $\Xi$", ha="center", va="center", fontsize=7.6, color=INK, fontweight="bold")
    rounded_box(axis, (0.22, 0.055), 0.19, 0.035, facecolor=ORANGE, edgecolor=ORANGE, linewidth=0.8, radius=0.007)
    rounded_box(axis, (0.41, 0.055), 0.19, 0.035, facecolor=BLUE, edgecolor=BLUE, linewidth=0.8, radius=0.007)
    rounded_box(axis, (0.60, 0.055), 0.19, 0.035, facecolor=GREEN, edgecolor=GREEN, linewidth=0.8, radius=0.007)
    axis.text(0.315, 0.072, "cross-group", ha="center", va="center", fontsize=5.8, color="white", fontweight="bold")
    axis.text(0.505, 0.072, "pair 1+2", ha="center", va="center", fontsize=5.8, color="white", fontweight="bold")
    axis.text(0.695, 0.072, "pair 3+4", ha="center", va="center", fontsize=5.8, color="white", fontweight="bold")


def draw_source_module(axis, y, name, color):
    rounded_box(axis, (0.04, y - 0.037), 0.23, 0.074, facecolor="white", edgecolor=color, linewidth=1.0, radius=0.018)
    axis.text(0.075, y, name, ha="left", va="center", fontsize=6.5, color=INK, fontweight="bold")
    for offset in (0.155, 0.19, 0.225):
        axis.add_patch(Circle((offset, y), 0.0105, facecolor=color, edgecolor="none"))


def draw_panel_c(axis):
    panel_label(axis, "C")
    axis.text(0.50, 0.94, r"Current HCP $\Xi$ hierarchy", ha="center", va="top", fontsize=11, fontweight="bold", color=INK)

    labels = ("Vis", "SomMot", "DAN", "SVAN", "Limbic", "Control", "Default")
    colors = ("#4C78A8", "#63A6A0", "#7187C6", "#9A77B6", "#D8A343", "#C96B65", "#6F9F73")
    ys = [0.79, 0.70, 0.61, 0.52, 0.43, 0.34, 0.25]
    for y, label, color in zip(ys, labels, colors):
        draw_source_module(axis, y, label, color)
        axis.plot([0.27, 0.30], [y, y], color="#AAB0B8", linewidth=0.8)
    axis.plot([0.30, 0.30], [ys[-1], ys[0]], color="#AAB0B8", linewidth=0.8)
    axis.text(0.155, 0.175, "three lagged signals per network", ha="center", va="center", fontsize=6.4, color=MUTED)

    tree_box(axis, (0.47, 0.69), 0.25, 0.14, "within networks", "interaction across 3 lags", facecolor=BLUE_LIGHT, edgecolor=BLUE)
    tree_box(axis, (0.47, 0.43), 0.25, 0.14, "between networks", "interaction across networks", facecolor=ORANGE_LIGHT, edgecolor=ORANGE)
    arrow(axis, (0.30, 0.61), (0.34, 0.69), color=BLUE, linewidth=1.5)
    arrow(axis, (0.30, 0.47), (0.34, 0.43), color=ORANGE, linewidth=1.5)

    rounded_box(axis, (0.67, 0.31), 0.27, 0.25, facecolor="#FBF6F1", edgecolor=ORANGE, linewidth=1.35)
    axis.text(0.805, 0.525, "split network groups", ha="center", va="center", fontsize=8, color=INK, fontweight="bold")
    axis.text(0.805, 0.485, "repeat the same visual split", ha="center", va="center", fontsize=6.3, color=MUTED)
    axis.plot([0.805, 0.76, 0.72], [0.45, 0.40, 0.35], color=INK, linewidth=1.2)
    axis.plot([0.805, 0.85, 0.89], [0.45, 0.40, 0.35], color=INK, linewidth=1.2)
    for x, y, size in ((0.805, 0.45, 0.019), (0.76, 0.40, 0.016), (0.85, 0.40, 0.016), (0.72, 0.35, 0.013), (0.89, 0.35, 0.013)):
        axis.add_patch(Circle((x, y), size, facecolor=ORANGE if y > 0.35 else MUTED, edgecolor="white", linewidth=0.6, zorder=3))
    axis.text(0.805, 0.325, "residual synergy blocks", ha="center", va="center", fontsize=7, color=ORANGE, fontweight="bold")
    arrow(axis, (0.60, 0.43), (0.67, 0.43), color=ORANGE, linewidth=1.7)

    rounded_box(axis, (0.34, 0.055), 0.60, 0.14, facecolor="white", edgecolor=LIGHT, linewidth=1.0)
    axis.text(0.64, 0.168, r"system joint effect  $\Xi$", ha="center", va="center", fontsize=8.0, color=INK, fontweight="bold")
    rounded_box(axis, (0.40, 0.100), 0.20, 0.038, facecolor=BLUE, edgecolor=BLUE, linewidth=0.8, radius=0.008)
    axis.text(0.50, 0.119, "within-network", ha="center", va="center", fontsize=5.8, color="white", fontweight="bold")
    for index, width in enumerate((0.09, 0.075, 0.065, 0.055)):
        x = 0.60 + sum((0.09, 0.075, 0.065, 0.055)[:index])
        rounded_box(axis, (x, 0.100), width, 0.038, facecolor=ORANGE, edgecolor="white", linewidth=0.6, radius=0.006)
    axis.text(0.742, 0.119, "cross-network blocks", ha="center", va="center", fontsize=5.5, color="white", fontweight="bold")
    axis.text(0.64, 0.078, "21 signals  →  7 networks  →  additive visual blocks", ha="center", va="center", fontsize=6.4, color=MUTED)
    axis.text(0.98, 0.02, "schematic · split path varies by subject/state", ha="right", va="bottom", fontsize=6.1, color=MUTED)


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure = plt.figure(figsize=(15.8, 5.7), constrained_layout=True, facecolor="white")
    grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 1.25, 1.40), wspace=0.05)
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    for axis in axes:
        setup_axis(axis)
    draw_panel_a(axes[0])
    draw_panel_b(axes[1])
    draw_panel_c(axes[2])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(OUTPUT.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
