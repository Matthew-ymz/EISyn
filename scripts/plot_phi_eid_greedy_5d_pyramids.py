#!/usr/bin/env python3
"""Visualize the five-node Phi EID greedy decomposition as 3D synergy pyramids."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


FIG_DIR = Path("docs/ref/assets/phi_eid_greedy_5d")


def _setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def _add_pyramid(
    ax,
    base_xy: list[tuple[float, float]],
    *,
    height: float,
    color: str,
    edgecolor: str,
    alpha: float,
    label: str = "",
    apex_shift: tuple[float, float] = (0.0, 0.0),
) -> None:
    base = np.asarray([[x, y, 0.0] for x, y in base_xy], dtype=float)
    centroid = base[:, :2].mean(axis=0) + np.asarray(apex_shift, dtype=float)
    apex = np.asarray([centroid[0], centroid[1], float(height)], dtype=float)

    faces = []
    if len(base) == 2:
        faces.append([base[0].tolist(), base[1].tolist(), apex.tolist()])
    elif len(base) >= 3:
        faces.append(base.tolist())
        for idx in range(len(base)):
            faces.append([base[idx].tolist(), base[(idx + 1) % len(base)].tolist(), apex.tolist()])
    else:
        raise ValueError("A synergy shape needs at least two source nodes.")

    poly = Poly3DCollection(
        faces,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=1.0,
        alpha=alpha,
    )
    ax.add_collection3d(poly)
    ax.scatter([apex[0]], [apex[1]], [apex[2]], s=34, color=edgecolor, depthshade=False)
    if label:
        ax.text(apex[0], apex[1], apex[2] + 0.10, label, ha="center", va="bottom", fontsize=8, color=edgecolor)


def _plot_base_network(ax, positions: dict[int, tuple[float, float]]) -> None:
    base_edges = [(1, 2), (2, 3), (3, 1), (4, 5)]
    for left, right in base_edges:
        x0, y0 = positions[left]
        x1, y1 = positions[right]
        ax.plot([x0, x1], [y0, y1], [0, 0], color="#9aa3ad", linewidth=1.2, alpha=0.75)


def _draw_nodes(ax, positions: dict[int, tuple[float, float]]) -> None:
    xs = [positions[i][0] for i in positions]
    ys = [positions[i][1] for i in positions]
    zs = np.full(len(xs), 0.035)
    ax.scatter(xs, ys, zs, s=82, color="#fdfdfd", edgecolor="#333333", linewidth=0.9, depthshade=False, zorder=20)
    for node, (x, y) in positions.items():
        offset_x = 0.10 if x <= 1.0 else -0.10
        offset_y = 0.08 if y <= 0 else -0.08
        ax.text(
            x + offset_x,
            y + offset_y,
            0.085,
            str(node),
            ha="center",
            va="center",
            fontsize=8,
            color="#111111",
            zorder=21,
        )


def build_figure() -> plt.Figure:
    _setup_style()
    fig = plt.figure(figsize=(8.2, 3.9), constrained_layout=True)
    grid = GridSpec(2, 3, figure=fig, width_ratios=[1.25, 1.25, 1.0], height_ratios=[1.0, 1.0])

    ax3d = fig.add_subplot(grid[:, :2], projection="3d")
    ax_order = fig.add_subplot(grid[0, 2])
    ax_burden = fig.add_subplot(grid[1, 2])

    positions = {
        1: (-0.65, 0.45),
        2: (0.05, -0.55),
        3: (0.75, 0.45),
        4: (1.75, -0.28),
        5: (2.45, 0.42),
    }
    _plot_base_network(ax3d, positions)

    _add_pyramid(
        ax3d,
        [positions[1], positions[2], positions[3]],
        height=1.0,
        color="#f58518",
        edgecolor="#b75b00",
        alpha=0.28,
        label="1.00",
    )
    _add_pyramid(
        ax3d,
        [positions[4], positions[5]],
        height=1.0,
        color="#4c78a8",
        edgecolor="#24527a",
        alpha=0.28,
        label="1.00",
    )
    _draw_nodes(ax3d, positions)

    ax3d.set_xlim(-1.2, 2.8)
    ax3d.set_ylim(-1.05, 1.05)
    ax3d.set_zlim(0.0, 1.25)
    ax3d.view_init(elev=24, azim=-58)
    ax3d.set_axis_off()
    ax3d.text2D(0.02, 0.96, "a", transform=ax3d.transAxes, fontsize=10, fontweight="bold")
    ax3d.text2D(0.08, 0.96, "Synergy surfaces over the source network", transform=ax3d.transAxes, fontsize=9)

    order_labels = ["order 2", "order 3", "cross-block"]
    order_values = [0.50, 0.50, 0.0]
    order_colors = ["#4c78a8", "#f58518", "#b7bec7"]
    ax_order.barh(order_labels, order_values, color=order_colors, height=0.58)
    ax_order.set_xlim(0, 0.6)
    ax_order.set_xlabel("fraction of Phi EID")
    ax_order.set_title("b  Order distribution", loc="left", fontsize=9, fontweight="bold")
    ax_order.set_xticks([0, 0.25, 0.50])
    ax_order.set_xticklabels(["0", "0.25", "0.50"])
    ax_order.grid(axis="x", color="#e4e7eb", linewidth=0.7)
    for idx, value in enumerate(order_values):
        ax_order.text(value + 0.02, idx, f"{value:.0%}", va="center", fontsize=7)

    burden_labels = ["X1", "X2", "X3", "X4", "X5"]
    burden_values = [1 / 6, 1 / 6, 1 / 6, 1 / 4, 1 / 4]
    burden_colors = ["#f58518", "#f58518", "#f58518", "#4c78a8", "#4c78a8"]
    ax_burden.bar(burden_labels, burden_values, color=burden_colors, width=0.62)
    ax_burden.set_ylim(0, 0.30)
    ax_burden.set_ylabel("burden")
    ax_burden.set_title("c  Node participation", loc="left", fontsize=9, fontweight="bold")
    ax_burden.set_yticks([0, 1 / 6, 1 / 4])
    ax_burden.set_yticklabels(["0", "1/6", "1/4"])
    ax_burden.grid(axis="y", color="#e4e7eb", linewidth=0.7)

    fig.suptitle("Five-node Phi EID decomposition: 2 bits split across two local modules", fontsize=10)
    return fig


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    base = FIG_DIR / "phi_eid_greedy_5d_distribution"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(base.with_suffix(".png"))
    print(base.with_suffix(".svg"))
    print(base.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
