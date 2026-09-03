#!/usr/bin/env python3
"""Generate Kuramoto coupling-network and PEID-tree assets for the report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_kuramoto_xi_hierarchy_trees import (
    DEFAULT_CROSS_COUPLINGS,
    DEFAULT_SUMMARY,
    _condition_name,
    _condition_rows,
    render_all,
)
from scripts.validate_greedy_hierarchy_kuramoto import NAMES, coupling_matrix


DEFAULT_OUTPUT_ROOT = ROOT / "docs/reports/assets/kuramoto_hierarchy"
DEFAULT_COMBINED_FIGURE = DEFAULT_OUTPUT_ROOT / "kuramoto_all_kout_complete_spt.png"
POSITIONS = {
    "theta1": (-1.10, 0.72),
    "theta2": (-1.25, -0.62),
    "theta3": (-0.30, -0.05),
    "theta4": (0.30, -0.05),
    "theta5": (1.25, -0.62),
    "theta6": (1.10, 0.72),
}


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
        }
    )


def render_network(
    *,
    within_coupling: float,
    cross_coupling: float,
    output_path: Path,
    dpi: int = 600,
) -> Path:
    """Draw one weighted six-oscillator network with a fixed layout."""
    matrix = coupling_matrix(
        within_coupling=float(within_coupling),
        cross_coupling=float(cross_coupling),
    )
    graph = nx.Graph()
    graph.add_nodes_from(NAMES)
    for left_index, left in enumerate(NAMES):
        for right_index in range(left_index + 1, len(NAMES)):
            weight = float(matrix[left_index, right_index])
            if weight > 0.0:
                graph.add_edge(left, NAMES[right_index], weight=weight)

    _configure_style()
    figure, axis = plt.subplots(figsize=(4.8, 3.6), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    within_edges = [
        (left, right)
        for left, right in graph.edges()
        if (left in NAMES[:3]) == (right in NAMES[:3])
    ]
    cross_edges = [edge for edge in graph.edges() if edge not in within_edges]

    nx.draw_networkx_edges(
        graph,
        POSITIONS,
        edgelist=within_edges,
        width=[1.0 + 4.0 * float(graph[left][right]["weight"]) for left, right in within_edges],
        edge_color="#4F7FA3",
        alpha=0.78,
        ax=axis,
    )
    if cross_edges:
        nx.draw_networkx_edges(
            graph,
            POSITIONS,
            edgelist=cross_edges,
            width=[0.55 + 4.0 * float(graph[left][right]["weight"]) for left, right in cross_edges],
            edge_color="#D07A35",
            alpha=[min(0.82, 0.22 + float(graph[left][right]["weight"])) for left, right in cross_edges],
            ax=axis,
        )
    nx.draw_networkx_nodes(
        graph,
        POSITIONS,
        nodelist=list(NAMES[:3]),
        node_color="#DCEAF3",
        edgecolors="#355F7A",
        linewidths=1.5,
        node_size=720,
        ax=axis,
    )
    nx.draw_networkx_nodes(
        graph,
        POSITIONS,
        nodelist=list(NAMES[3:]),
        node_color="#F5E2D1",
        edgecolors="#9B5C2B",
        linewidths=1.5,
        node_size=720,
        ax=axis,
    )
    labels = {name: rf"$\theta_{{{index}}}$" for index, name in enumerate(NAMES, start=1)}
    nx.draw_networkx_labels(graph, POSITIONS, labels=labels, font_size=9, font_color="#24313C", ax=axis)
    axis.set_xlim(-1.62, 1.62)
    axis.set_ylim(-1.00, 1.02)
    axis.set_aspect("equal")
    axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def render_report_assets(
    summary_path: Path,
    output_root: Path,
    *,
    seed: int = 0,
    dpi: int = 600,
) -> dict[float, dict[str, Path]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _condition_rows(summary, seed=seed)
    tree_outputs = render_all(summary_path, output_root / "trees", seed=seed, dpi=dpi)
    assets: dict[float, dict[str, Path]] = {}
    for row, tree_path in zip(rows, tree_outputs, strict=True):
        cross_coupling = float(row["cross_coupling"])
        network_path = output_root / "networks" / f"kuramoto_network_{_condition_name(cross_coupling)}.png"
        render_network(
            within_coupling=float(row["within_coupling"]),
            cross_coupling=cross_coupling,
            output_path=network_path,
            dpi=dpi,
        )
        assets[cross_coupling] = {"network": network_path, "tree": tree_path}
    if tuple(assets) != DEFAULT_CROSS_COUPLINGS:
        raise ValueError(f"Unexpected rendered conditions: {tuple(assets)}")
    return assets


def _crop_white_margin(image: np.ndarray, *, pad: int = 24) -> np.ndarray:
    rgb = np.asarray(image)[..., :3]
    occupied = np.any(rgb < 0.985, axis=2)
    if not np.any(occupied):
        return image
    rows, columns = np.where(occupied)
    top = max(0, int(rows.min()) - int(pad))
    bottom = min(image.shape[0], int(rows.max()) + int(pad) + 1)
    left = max(0, int(columns.min()) - int(pad))
    right = min(image.shape[1], int(columns.max()) + int(pad) + 1)
    return image[top:bottom, left:right]


def render_combined_figure(
    assets: dict[float, dict[str, Path]],
    output_path: Path = DEFAULT_COMBINED_FIGURE,
    *,
    dpi: int = 300,
) -> Path:
    """Combine all four network/tree pairs into one vertically aligned figure."""
    _configure_style()
    figure, axes = plt.subplots(4, 2, figsize=(12.0, 17.0), constrained_layout=True)
    figure.patch.set_facecolor("white")
    for row_index, cross_coupling in enumerate(DEFAULT_CROSS_COUPLINGS):
        pair = assets[float(cross_coupling)]
        for column_index, key in enumerate(("network", "tree")):
            axis = axes[row_index, column_index]
            axis.imshow(_crop_white_margin(plt.imread(pair[key])))
            axis.axis("off")
        axes[row_index, 0].text(
            -0.04,
            0.5,
            rf"$K_{{\mathrm{{out}}}}={cross_coupling:g}$",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=13,
            color="#24313C",
        )
    axes[0, 0].set_title("Coupling network", fontsize=14, pad=12, color="#24313C")
    axes[0, 1].set_title("Complete SPT hierarchy", fontsize=14, pad=12, color="#24313C")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets = render_report_assets(args.summary, args.output_root, seed=args.seed)
    render_combined_figure(assets, args.output_root / DEFAULT_COMBINED_FIGURE.name)


if __name__ == "__main__":
    main()
