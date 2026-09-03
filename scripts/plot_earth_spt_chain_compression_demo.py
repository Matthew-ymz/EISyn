#!/usr/bin/env python3
"""Show chain compression on the Earth SLP H=1 synergy partition tree."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_runge_slp_pc60_xi_hierarchy import (
    EDGE_COLOR,
    INK,
    SPLIT_COLOR,
    SYN_NONNEGATIVE_TOLERANCE_BITS,
    _blend_with_white,
    _compact_positions,
    _dominant_spine,
    flatten_nodes,
)
from scripts.compare_runge_slp_pc60_xi_horizons import _node_from_record


INPUT = ROOT / "results/runge/slp_pc60_xi_hierarchy/H001/summary.json"
OUTPUT = ROOT / "fig/earth_slp_pc60_xi_hierarchy_H001_chain_compressed.png"
LEAF_FACE = "#F2F4F6"
LEAF_EDGE = "#7A8792"
ACCENT = "#D08745"


def _display_syn(value: float) -> float:
    if value < -SYN_NONNEGATIVE_TOLERANCE_BITS:
        raise RuntimeError(
            f"Significant negative Syn {value:.12g} bits; "
            f"tolerance={SYN_NONNEGATIVE_TOLERANCE_BITS:.12g} bits."
        )
    return max(0.0, float(value))


def _syn_style(value: float, maximum: float) -> tuple[float, str, str, float]:
    relative = min(1.0, _display_syn(value) / max(maximum, 1.0e-15))
    return (
        26.0 + 75.0 * math.sqrt(relative),
        _blend_with_white(SPLIT_COLOR, 0.16 + 0.70 * relative),
        _blend_with_white(SPLIT_COLOR, 0.58 + 0.42 * relative),
        0.7 + 1.25 * relative,
    )


def _draw_internal(ax: plt.Axes, x: float, y: float, node, maximum: float, label: str) -> None:
    area, face, edge, width = _syn_style(node.syn_bits, maximum)
    ax.scatter([x], [y], s=area, facecolor=face, edgecolor=edge, linewidth=width, zorder=5)
    if label:
        ax.text(
            x + 0.035,
            y,
            label,
            ha="left",
            va="center",
            color=INK,
            fontsize=7.0,
            linespacing=1.15,
            zorder=6,
        )


def _draw_leaf(ax: plt.Axes, x: float, y: float, label: str) -> None:
    ax.scatter([x], [y], s=26, facecolor=LEAF_FACE, edgecolor=LEAF_EDGE, linewidth=0.75, zorder=5)
    ax.text(x, y - 0.035, label, ha="center", va="top", color="#53606B", fontsize=6.5)


def _draw_full_tree(ax: plt.Axes, root, maximum: float) -> None:
    positions, _ = _compact_positions(root)
    internal = [node for node in flatten_nodes(root) if node.children]
    leaves = [node for node in flatten_nodes(root) if not node.children]
    scale = root.size - 1

    for node in internal:
        _, parent_y = positions[node.indices]
        child_points = [positions[child.indices] for child in node.children]
        xs = [point[0] / scale for point in child_points]
        ys = [point[1] for point in child_points]
        ax.plot(xs, [parent_y, parent_y], color=EDGE_COLOR, linewidth=0.55, zorder=1)
        for child_x, child_y in zip(xs, ys, strict=True):
            ax.plot([child_x, child_x], [child_y, parent_y], color=EDGE_COLOR, linewidth=0.55, zorder=1)

    for node in internal:
        x, y = positions[node.indices]
        area, face, edge, width = _syn_style(node.syn_bits, maximum)
        ax.scatter(
            [x / scale], [y], s=0.35 * area, facecolor=face, edgecolor=edge,
            linewidth=0.65 * width, zorder=3,
        )
    for node in leaves:
        x, y = positions[node.indices]
        ax.scatter([x / scale], [y], s=7, facecolor=LEAF_FACE, edgecolor=LEAF_EDGE, linewidth=0.4, zorder=3)

    focus = next(
        node
        for node in internal
        if node.size == 7 and set(node.indices) == {1, 3, 4, 15, 16, 21, 51}
    )
    focus_nodes = flatten_nodes(focus)
    focus_x = [positions[node.indices][0] / scale for node in focus_nodes]
    focus_y = [positions[node.indices][1] for node in focus_nodes]
    x0, x1 = min(focus_x) - 0.018, max(focus_x) + 0.018
    y0, y1 = -0.025, max(focus_y) + 0.035
    ax.add_patch(
        Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor=_blend_with_white(ACCENT, 0.10), edgecolor=ACCENT,
            linewidth=1.1, linestyle=(0, (3, 2)), zorder=0,
        )
    )
    ax.annotate(
        "Expanded focus",
        xy=((x0 + x1) / 2, y1), xytext=(0.18, 0.30),
        arrowprops={"arrowstyle": "-", "color": ACCENT, "linewidth": 0.9},
        color=ACCENT, fontsize=7.2, ha="left", va="bottom",
    )
    ax.text(0.0, 1.06, "a  Full H=1 hierarchy", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold", color=INK)
    ax.text(0.0, 1.005, "60 leaves · 59 internal splits", transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="#5C6873")
    ax.set_xlim(-0.035, 1.025)
    ax.set_ylim(-0.06, 1.05)
    ax.axis("off")


def _draw_compressed_tree(ax: plt.Axes, root, maximum: float) -> None:
    spine, sides = _dominant_spine(root)
    by_size = {node.size: node for node in spine}
    side_by_size = {node.size: side for node, side in zip(spine, sides, strict=True)}

    # The retained leaves keep the same left-to-right order as the full hierarchy.
    leaf_x = {
        1: 0.05, 3: 0.15, 4: 0.27, 16: 0.43, 21: 0.54,
        15: 0.72, 51: 0.88, 56: 0.98,
    }
    xs = {2: 0.10, 3: 0.157, 5: 0.302, 6: 0.372, 7: 0.445, 59: 0.79, 60: 0.80}
    ys = {2: 0.12, 3: 0.21, 5: 0.33, 6: 0.42, 7: 0.51, 59: 0.88, 60: 0.96}
    pair_x, pair_y = 0.485, 0.16

    def split_edge(parent_size: int, child_x: float, child_y: float, side_x: float) -> None:
        parent_y = ys[parent_size]
        ax.plot([child_x, side_x], [parent_y, parent_y], color=EDGE_COLOR, linewidth=1.0, zorder=1)
        ax.plot([child_x, child_x], [child_y, parent_y], color=EDGE_COLOR, linewidth=1.0, zorder=1)
        ax.plot([side_x, side_x], [0.0, parent_y], color=EDGE_COLOR, linewidth=1.0, zorder=1)

    split_edge(60, xs[59], ys[59], leaf_x[56])
    split_edge(7, xs[6], ys[6], leaf_x[51])
    split_edge(6, xs[5], ys[5], leaf_x[15])
    split_edge(3, xs[2], ys[2], leaf_x[4])

    # The n=5 split is the only multi-PC side branch.
    ax.plot([xs[3], pair_x], [ys[5], ys[5]], color=EDGE_COLOR, linewidth=1.05, zorder=1)
    ax.plot([xs[3], xs[3]], [ys[3], ys[5]], color=EDGE_COLOR, linewidth=1.0, zorder=1)
    ax.plot([pair_x, pair_x], [pair_y, ys[5]], color=EDGE_COLOR, linewidth=1.0, zorder=1)
    ax.plot([leaf_x[16], leaf_x[21]], [pair_y, pair_y], color=EDGE_COLOR, linewidth=0.95, zorder=1)
    for index in (16, 21):
        ax.plot([leaf_x[index], leaf_x[index]], [0.0, pair_y], color=EDGE_COLOR, linewidth=0.95, zorder=1)

    # Terminal pair No.1 + No.3.
    ax.plot([leaf_x[1], leaf_x[3]], [ys[2], ys[2]], color=EDGE_COLOR, linewidth=0.95, zorder=1)
    for index in (1, 3):
        ax.plot([leaf_x[index], leaf_x[index]], [0.0, ys[2]], color=EDGE_COLOR, linewidth=0.95, zorder=1)

    # Keep only the omitted diagonal chain and its ellipsis.
    ax.plot(
        [xs[7], xs[59]], [ys[7], ys[59]],
        color=ACCENT, linewidth=1.25, linestyle=(0, (3, 2)), zorder=2,
    )
    midpoint_x = (xs[7] + xs[59]) / 2
    midpoint_y = (ys[7] + ys[59]) / 2
    ax.text(
        midpoint_x, midpoint_y + 0.018, r"$\cdots$",
        ha="center", va="center", fontsize=18, color=ACCENT,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8}, zorder=5,
    )
    for index in (1, 3, 4, 16, 21, 15, 51, 56):
        _draw_leaf(ax, leaf_x[index], 0.0, f"No.{index}")

    for size in (60, 59, 7, 6, 5, 3, 2):
        label = ""
        if size in (60, 59, 7, 5, 2):
            label = f"n={size}"
        _draw_internal(ax, xs[size], ys[size], by_size[size], maximum, label)
    branch = side_by_size[5]
    _draw_internal(
        ax, pair_x, pair_y, branch, maximum,
        "n=2",
    )

    ax.text(0.0, 1.06, "b  Chain-compressed view", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold", color=INK)
    ax.text(0.0, 1.005, "Original tree geometry retained", transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="#5C6873")
    ax.set_xlim(0.0, 1.06)
    ax.set_ylim(-0.08, 1.03)
    ax.axis("off")


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "savefig.facecolor": "white",
        }
    )
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    root = _node_from_record(payload["tree"])
    internal = [node for node in flatten_nodes(root) if node.children]
    maximum = max(_display_syn(node.syn_bits) for node in internal)

    figure = plt.figure(figsize=(12.2, 5.5), layout="constrained", facecolor="white")
    grid = figure.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.06)
    left = figure.add_subplot(grid[0, 0])
    right = figure.add_subplot(grid[0, 1])
    _draw_full_tree(left, root, maximum)
    _draw_compressed_tree(right, root, maximum)
    figure.suptitle(
        rf"Earth SLP synergy partition tree ($H=1$; $\Xi$ = {root.xi_bits:.2f} bits)",
        x=0.5, y=1.01, fontsize=10, color=INK,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
