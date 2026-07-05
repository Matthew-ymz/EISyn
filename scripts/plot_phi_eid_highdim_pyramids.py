#!/usr/bin/env python3
"""High-dimensional synergy-surface visualization and greedy-search timing."""

from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


OUT_DIR = Path("docs/ref/assets/phi_eid_highdim")


@dataclass(frozen=True)
class SynergyModule:
    sources: tuple[int, ...]
    weight: float
    color: str
    edgecolor: str


MODULES: tuple[SynergyModule, ...] = (
    SynergyModule((1, 2, 3, 4), 1.00, "#f58518", "#a95500"),
    SynergyModule((4, 5, 6), 0.75, "#54a24b", "#2f6f2a"),
    SynergyModule((7, 8, 9), 0.80, "#b279a2", "#7b4f70"),
    SynergyModule((6, 10, 12), 0.60, "#00a6a6", "#006f70"),
    SynergyModule((10, 11), 0.45, "#4c78a8", "#24527a"),
    SynergyModule((2, 12), 0.35, "#e45756", "#9a2e2d"),
)


def _setup_style() -> None:
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


def synergy_field(subset: Iterable[int], modules: Iterable[SynergyModule]) -> float:
    source_set = set(int(x) for x in subset)
    return float(sum(module.weight for module in modules if set(module.sources).issubset(source_set)))


def nontrivial_bipartitions(nodes: tuple[int, ...]) -> Iterable[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Yield each unordered bipartition once."""

    ordered = tuple(sorted(nodes))
    if len(ordered) <= 1:
        return
    first = ordered[0]
    rest = ordered[1:]
    all_set = set(ordered)
    for mask in range(1 << len(rest)):
        left = {first}
        for idx, node in enumerate(rest):
            if mask & (1 << idx):
                left.add(node)
        if len(left) == len(ordered):
            continue
        right = all_set - left
        yield tuple(sorted(left)), tuple(sorted(right))


def greedy_tree(nodes: tuple[int, ...], modules: tuple[SynergyModule, ...], *, eps: float = 1e-12) -> dict[str, object]:
    """Top-down split that maximizes child synergy captured by the partition."""

    nodes = tuple(sorted(nodes))
    own = synergy_field(nodes, modules)
    if len(nodes) <= 1 or own <= eps:
        return {"nodes": nodes, "synergy": own, "terminal": True, "children": [], "residual": own}

    best: tuple[float, tuple[int, ...], tuple[int, ...]] | None = None
    checked = 0
    for left, right in nontrivial_bipartitions(nodes):
        checked += 1
        captured = synergy_field(left, modules) + synergy_field(right, modules)
        if best is None or captured > best[0]:
            best = (captured, left, right)

    if best is None or best[0] <= eps:
        return {
            "nodes": nodes,
            "synergy": own,
            "terminal": True,
            "children": [],
            "residual": own,
            "checked_bipartitions": checked,
        }

    captured, left, right = best
    residual = max(0.0, own - captured)
    return {
        "nodes": nodes,
        "synergy": own,
        "terminal": False,
        "split": [left, right],
        "residual": residual,
        "checked_bipartitions": checked,
        "children": [greedy_tree(left, modules, eps=eps), greedy_tree(right, modules, eps=eps)],
    }


def benchmark_split_search() -> list[dict[str, float]]:
    """Measure exhaustive split-search overhead for small synthetic fields."""

    rows: list[dict[str, float]] = []
    for n in (8, 10, 12, 14, 16):
        modules = tuple(
            SynergyModule(tuple(range(start, min(start + width, n + 1))), 1.0, "#000000", "#000000")
            for start, width in ((1, 4), (4, 3), (7, 3), (10, 2), (2, 2))
            if start + width - 1 <= n
        )
        t0 = time.perf_counter()
        tree = greedy_tree(tuple(range(1, n + 1)), modules)
        elapsed = time.perf_counter() - t0
        checked = _sum_checked(tree)
        rows.append(
            {
                "n": float(n),
                "seconds": float(elapsed),
                "checked_bipartitions": float(checked),
                "root_bipartitions": float((2 ** (n - 1)) - 1),
                "all_order_le_4_subsets": float(sum(math.comb(n, k) for k in range(1, min(4, n) + 1))),
            }
        )
    return rows


def _sum_checked(tree: dict[str, object]) -> int:
    total = int(tree.get("checked_bipartitions", 0) or 0)
    for child in tree.get("children", []) or []:
        total += _sum_checked(child)  # type: ignore[arg-type]
    return total


def _node_positions(n: int = 12) -> dict[int, tuple[float, float]]:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions: dict[int, tuple[float, float]] = {}
    for idx, angle in enumerate(angles, start=1):
        radius = 1.0 + (0.18 if idx % 2 == 0 else -0.04)
        x = 1.45 * radius * np.cos(angle)
        y = 1.00 * radius * np.sin(angle)
        positions[idx] = (float(x), float(y))
    return positions


def _plot_base(ax, positions: dict[int, tuple[float, float]]) -> None:
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 10), (10, 11), (10, 12), (2, 12), (7, 8), (8, 9), (9, 7)]
    for left, right in edges:
        x0, y0 = positions[left]
        x1, y1 = positions[right]
        ax.plot([x0, x1], [y0, y1], [0, 0], color="#aeb7c2", linewidth=0.75, alpha=0.62)


def _add_surface(ax, base_xy: list[tuple[float, float]], module: SynergyModule, *, max_height: float) -> None:
    base = np.asarray([[x, y, 0.0] for x, y in base_xy], dtype=float)
    centroid = base[:, :2].mean(axis=0)
    height = 1.35 * float(module.weight) / float(max_height)
    apex = np.asarray([centroid[0], centroid[1], height], dtype=float)

    faces: list[list[list[float]]] = []
    if len(base) == 2:
        faces.append([base[0].tolist(), base[1].tolist(), apex.tolist()])
    else:
        faces.append(base.tolist())
        for idx in range(len(base)):
            faces.append([base[idx].tolist(), base[(idx + 1) % len(base)].tolist(), apex.tolist()])

    ax.add_collection3d(
        Poly3DCollection(faces, facecolor=module.color, edgecolor=module.edgecolor, linewidth=0.85, alpha=0.24)
    )
    ax.scatter([apex[0]], [apex[1]], [apex[2]], s=24, color=module.edgecolor, depthshade=False)
    ax.text(
        apex[0],
        apex[1],
        apex[2] + 0.105,
        f"{module.weight:.2f}",
        ha="center",
        va="bottom",
        fontsize=6.6,
        color=module.edgecolor,
        bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.62},
    )


def _draw_nodes(ax, positions: dict[int, tuple[float, float]]) -> None:
    xs = [positions[i][0] for i in sorted(positions)]
    ys = [positions[i][1] for i in sorted(positions)]
    zs = np.full(len(xs), 0.035)
    ax.scatter(xs, ys, zs, s=42, color="#ffffff", edgecolor="#333333", linewidth=0.75, depthshade=False)
    for node in sorted(positions):
        x, y = positions[node]
        offset_x = 0.08 if x <= 0 else -0.08
        offset_y = 0.06 if y <= 0 else -0.06
        ax.text(
            x + offset_x,
            y + offset_y,
            0.095,
            str(node),
            ha="center",
            va="center",
            fontsize=6.8,
            color="#111111",
        )


def build_figure(benchmark_rows: list[dict[str, float]]) -> plt.Figure:
    _setup_style()
    total_phi = sum(module.weight for module in MODULES)
    order_totals: dict[int, float] = {}
    burden = {node: 0.0 for node in range(1, 13)}
    for module in MODULES:
        order_totals[len(module.sources)] = order_totals.get(len(module.sources), 0.0) + module.weight
        for node in module.sources:
            burden[node] += module.weight / len(module.sources) / total_phi

    fig = plt.figure(figsize=(9.0, 5.2), constrained_layout=True)
    grid = GridSpec(2, 3, figure=fig, width_ratios=[1.42, 1.42, 1.0], height_ratios=[1.0, 1.0])
    ax3d = fig.add_subplot(grid[:, :2], projection="3d")
    ax_order = fig.add_subplot(grid[0, 2])
    ax_runtime = fig.add_subplot(grid[1, 2])

    positions = _node_positions(12)
    _plot_base(ax3d, positions)
    max_height = max(module.weight for module in MODULES)
    # Larger modules first; smaller two-source surfaces stay visible on top.
    for module in sorted(MODULES, key=lambda item: (len(item.sources), item.weight), reverse=True):
        _add_surface(ax3d, [positions[node] for node in module.sources], module, max_height=max_height)
    _draw_nodes(ax3d, positions)
    ax3d.set_xlim(-2.1, 2.1)
    ax3d.set_ylim(-1.65, 1.65)
    ax3d.set_zlim(0.0, 1.60)
    ax3d.view_init(elev=28, azim=-58)
    ax3d.set_axis_off()
    ax3d.text2D(0.02, 0.96, "a", transform=ax3d.transAxes, fontsize=10, fontweight="bold")
    ax3d.text2D(0.08, 0.96, "12-node synergy surfaces", transform=ax3d.transAxes, fontsize=9)

    orders = [2, 3, 4]
    order_values = [order_totals.get(order, 0.0) / total_phi for order in orders]
    colors = ["#4c78a8", "#54a24b", "#f58518"]
    ax_order.barh([f"order {order}" for order in orders], order_values, color=colors, height=0.56)
    ax_order.set_xlim(0.0, 0.60)
    ax_order.set_xlabel("fraction of Phi EID")
    ax_order.set_title("b  Order distribution", loc="left", fontsize=9, fontweight="bold")
    ax_order.grid(axis="x", color="#e4e7eb", linewidth=0.7)
    for idx, value in enumerate(order_values):
        ax_order.text(value + 0.015, idx, f"{value:.0%}", va="center", fontsize=7)

    runtime_n = [row["n"] for row in benchmark_rows]
    runtime_seconds = [max(row["seconds"], 1e-5) for row in benchmark_rows]
    ax_runtime.plot(runtime_n, runtime_seconds, color="#5b6770", marker="o", linewidth=1.2)
    ax_runtime.set_yscale("log")
    ax_runtime.set_xlabel("number of source nodes")
    ax_runtime.set_ylabel("split-search seconds")
    ax_runtime.set_title("c  Greedy split overhead", loc="left", fontsize=9, fontweight="bold")
    ax_runtime.grid(axis="y", color="#e4e7eb", linewidth=0.7)

    fig.suptitle("High-dimensional local integration map: overlapping synergy surfaces on a 12-node base graph", fontsize=10)
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tree = greedy_tree(tuple(range(1, 13)), MODULES)
    tree_elapsed = time.perf_counter() - t0
    benchmark_rows = benchmark_split_search()

    fig = build_figure(benchmark_rows)
    base = OUT_DIR / "phi_eid_highdim_synergy_surfaces"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    summary = {
        "node_count": 12,
        "module_count": len(MODULES),
        "phi_eid_bits": sum(module.weight for module in MODULES),
        "greedy_tree_seconds": tree_elapsed,
        "greedy_tree_checked_bipartitions": _sum_checked(tree),
        "benchmark_rows": benchmark_rows,
        "outputs": {
            "png": str(base.with_suffix(".png")),
            "svg": str(base.with_suffix(".svg")),
            "pdf": str(base.with_suffix(".pdf")),
        },
    }
    (OUT_DIR / "phi_eid_highdim_runtime_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
