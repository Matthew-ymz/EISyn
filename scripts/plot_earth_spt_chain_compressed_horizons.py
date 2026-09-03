#!/usr/bin/env python3
"""Draw chain-compressed Earth SLP SPTs for H=1, 10, and 60."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_runge_slp_pc60_xi_hierarchy import (
    EDGE_COLOR,
    INK,
    SPLIT_COLOR,
    SYN_NONNEGATIVE_TOLERANCE_BITS,
    _blend_with_white,
    _dominant_spine,
    flatten_nodes,
)
from scripts.compare_runge_slp_pc60_xi_horizons import _node_from_record


RESULT_ROOT = ROOT / "results/runge/slp_pc60_xi_hierarchy"
OUTPUT = ROOT / "fig/earth_slp_pc60_xi_hierarchy_chain_compressed_horizons.png"
HORIZONS = (1, 10, 60)
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


def _node_style(value: float, maximum: float) -> tuple[float, str, str, float]:
    relative = min(1.0, _display_syn(value) / max(maximum, 1.0e-15))
    return (
        22.0 + 70.0 * math.sqrt(relative),
        _blend_with_white(SPLIT_COLOR, 0.14 + 0.72 * relative),
        _blend_with_white(SPLIT_COLOR, 0.58 + 0.42 * relative),
        0.7 + 1.25 * relative,
    )


def _leaf_order(node) -> list[int]:
    if not node.children:
        return [int(node.indices[0])]
    return [index for child in node.children for index in _leaf_order(child)]


def _focus_positions(focus) -> tuple[dict[tuple[int, ...], tuple[float, float]], list[int]]:
    order = _leaf_order(focus)
    leaf_x = {
        index: 0.055 + position * (0.755 / max(1, len(order) - 1))
        for position, index in enumerate(order)
    }
    positions: dict[tuple[int, ...], tuple[float, float]] = {}

    def visit(node) -> tuple[float, float]:
        if node.indices in positions:
            return positions[node.indices]
        if not node.children:
            point = (leaf_x[int(node.indices[0])], 0.0)
        else:
            children = [visit(child) for child in node.children]
            x_value = sum(
                point[0] * child.size
                for point, child in zip(children, node.children, strict=True)
            ) / node.size
            y_value = 0.49 * math.log2(node.size) / math.log2(focus.size)
            point = (x_value, y_value)
        positions[node.indices] = point
        return point

    visit(focus)
    return positions, order


def _draw_panel(ax: plt.Axes, root, horizon: int, maximum: float, panel: str) -> None:
    spine, sides = _dominant_spine(root)
    by_size = {node.size: node for node in spine}
    side_by_size = {node.size: side for node, side in zip(spine, sides, strict=True)}
    focus = by_size[7]
    positions, leaf_order = _focus_positions(focus)

    # Fully expanded seven-PC focus subtree.
    for node in flatten_nodes(focus):
        if not node.children:
            continue
        _, parent_y = positions[node.indices]
        child_points = [positions[child.indices] for child in node.children]
        ax.plot(
            [child_points[0][0], child_points[1][0]],
            [parent_y, parent_y],
            color=EDGE_COLOR, linewidth=0.95, zorder=1,
        )
        for child_x, child_y in child_points:
            ax.plot(
                [child_x, child_x], [child_y, parent_y],
                color=EDGE_COLOR, linewidth=0.95, zorder=1,
            )

    for node in flatten_nodes(focus):
        x_value, y_value = positions[node.indices]
        if node.children:
            area, face, edge, width = _node_style(node.syn_bits, maximum)
            ax.scatter(
                [x_value], [y_value], s=area, facecolor=face,
                edgecolor=edge, linewidth=width, zorder=4,
            )
        else:
            ax.scatter(
                [x_value], [0.0], s=22, facecolor=LEAF_FACE,
                edgecolor=LEAF_EDGE, linewidth=0.75, zorder=4,
            )
            ax.text(
                x_value, -0.035, f"No.{node.indices[0]}",
                ha="center", va="top", fontsize=6.2, color="#53606B",
            )

    # Root context: show the first peel-off, then compress n=59 down to n=7.
    root_x, root_y = 0.825, 0.96
    n59_x, n59_y = 0.815, 0.88
    first_side = side_by_size[60]
    first_leaf_x = 0.98
    ax.plot([n59_x, first_leaf_x], [root_y, root_y], color=EDGE_COLOR, linewidth=1.0, zorder=1)
    ax.plot([n59_x, n59_x], [n59_y, root_y], color=EDGE_COLOR, linewidth=1.0, zorder=1)
    ax.plot([first_leaf_x, first_leaf_x], [0.0, root_y], color=EDGE_COLOR, linewidth=1.0, zorder=1)
    ax.scatter(
        [first_leaf_x], [0.0], s=22, facecolor=LEAF_FACE,
        edgecolor=LEAF_EDGE, linewidth=0.75, zorder=4,
    )
    ax.text(
        first_leaf_x, -0.035, f"No.{first_side.indices[0]}",
        ha="center", va="top", fontsize=6.2, color="#53606B",
    )

    focus_x, focus_y = positions[focus.indices]
    ax.plot(
        [focus_x, n59_x], [focus_y, n59_y],
        color=ACCENT, linewidth=1.3, linestyle=(0, (3, 2)), zorder=2,
    )
    midpoint_x = (focus_x + n59_x) / 2
    midpoint_y = (focus_y + n59_y) / 2
    ax.text(
        midpoint_x, midpoint_y, r"$\cdots$",
        ha="center", va="center", fontsize=18, color=ACCENT,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7}, zorder=5,
    )

    for node, x_value, y_value in (
        (root, root_x, root_y),
        (by_size[59], n59_x, n59_y),
    ):
        area, face, edge, width = _node_style(node.syn_bits, maximum)
        ax.scatter(
            [x_value], [y_value], s=area, facecolor=face,
            edgecolor=edge, linewidth=width, zorder=4,
        )
    ax.text(root_x + 0.025, root_y, "n=60", ha="left", va="center", fontsize=6.3, color=INK)
    ax.text(n59_x + 0.025, n59_y, "n=59", ha="left", va="center", fontsize=6.3, color=INK)
    ax.text(focus_x + 0.025, focus_y, "n=7", ha="left", va="center", fontsize=6.3, color=INK)

    ax.text(
        0.0, 1.04, f"{panel}  H = {horizon}",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=8.5, fontweight="bold", color=INK,
    )
    ax.text(
        0.0, 0.992, rf"$\Xi$ = {root.xi_bits:.3f} bits",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=6.8, color="#5C6873",
    )
    ax.set_xlim(0.0, 1.08)
    ax.set_ylim(-0.085, 1.04)
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
    roots = []
    for horizon in HORIZONS:
        payload = json.loads(
            (RESULT_ROOT / f"H{horizon:03d}/summary.json").read_text(encoding="utf-8")
        )
        roots.append(_node_from_record(payload["tree"]))
    maximum = max(
        _display_syn(node.syn_bits)
        for root in roots
        for node in flatten_nodes(root)
        if node.children
    )

    figure, axes = plt.subplots(
        1, 3, figsize=(13.2, 4.7), layout="constrained", facecolor="white",
    )
    for panel, horizon, root, axis in zip("abc", HORIZONS, roots, axes, strict=True):
        _draw_panel(axis, root, horizon, maximum, panel)
    color_map = mpl.colors.LinearSegmentedColormap.from_list(
        "syn_strength",
        [
            _blend_with_white(SPLIT_COLOR, 0.14),
            _blend_with_white(SPLIT_COLOR, 0.86),
        ],
    )
    color_norm = mpl.colors.Normalize(vmin=0.0, vmax=maximum)
    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=color_norm, cmap=color_map),
        ax=axes, location="right", shrink=0.76, aspect=28, pad=0.018,
        ticks=[0.0, 0.1, 0.2, maximum],
    )
    colorbar.set_label("Syn (bits)", fontsize=7.2, color=INK, labelpad=7)
    colorbar.ax.tick_params(labelsize=6.5, width=0.6, length=2.5, colors=INK)
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.set_yticklabels(["0.00", "0.10", "0.20", f"{maximum:.2f}"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
