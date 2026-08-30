#!/usr/bin/env python3
"""Render a representative 100-ROI Xi hierarchy for the Schaefer100 DMF experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_dmf_critical_phi_hierarchy_topology import conditional_total_correlation
from scripts.analyze_runge_slp_pc60_xi_hierarchy import (
    EDGE_COLOR,
    INK,
    SPLIT_COLOR,
    ScalableHierarchyNode,
    _node_record,
    build_scalable_hierarchy,
    flatten_nodes,
    pairwise_syn_affinity,
)


DEFAULT_INPUT = ROOT / "results/dmf_schaefer100/full/critical_yeo7.npz"
DEFAULT_OUTPUT_DIR = ROOT / "results/dmf_schaefer100/xi_hierarchy_tree"
DEFAULT_FIGURE = ROOT / "fig/brain_dmf_schaefer100_xi_hierarchy_G130_seed04.png"
SYN_NONNEGATIVE_TOLERANCE_BITS = 1.0e-8
NETWORK_COLORS = (
    "#6A51A3",
    "#3182BD",
    "#31A354",
    "#E6550D",
    "#9C6B30",
    "#E6AB02",
    "#66A61E",
)


class ConditionalBlockXiOracle:
    """Cached conditional total correlation for arbitrary sets of ROI blocks."""

    def __init__(self, conditional: np.ndarray, blocks: Sequence[Sequence[int]]):
        self.conditional = np.asarray(conditional, dtype=np.float64)
        self.blocks = tuple(tuple(map(int, block)) for block in blocks)
        self._cache: dict[tuple[int, ...], float] = {
            (index,): 0.0 for index in range(len(self.blocks))
        }
        self.evaluations = len(self.blocks)

    def xi(self, indices: Iterable[int]) -> float:
        key = tuple(sorted(int(index) for index in indices))
        if not key or len(key) == 1:
            return 0.0
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        source_indices = [source for index in key for source in self.blocks[index]]
        local = self.conditional[np.ix_(source_indices, source_indices)]
        local_blocks = tuple((2 * position, 2 * position + 1) for position in range(len(key)))
        value = conditional_total_correlation(local, local_blocks)
        self._cache[key] = float(value)
        self.evaluations += 1
        return float(value)


def _blend_with_white(color: str, strength: float) -> str:
    base = to_rgb(color)
    amount = min(1.0, max(0.0, float(strength)))
    return to_hex(tuple(1.0 - amount * (1.0 - channel) for channel in base))


def _leaf_order(root: ScalableHierarchyNode) -> list[int]:
    if not root.children:
        return [root.indices[0]]
    return [index for child in root.children for index in _leaf_order(child)]


def _short_roi_label(label: str) -> str:
    text = str(label)
    for prefix, replacement in (
        ("7Networks_LH_", "L-"),
        ("7Networks_RH_", "R-"),
        ("ctx-lh-", "L-"),
        ("ctx-rh-", "R-"),
    ):
        text = text.replace(prefix, replacement)
    return text.replace("SalVentAttn", "SalVent").replace("Default", "DMN")


def _tree_metrics(root: ScalableHierarchyNode) -> dict[str, float | int]:
    internal = [node for node in flatten_nodes(root) if node.children]
    leaf_depths = [node.depth for node in flatten_nodes(root) if not node.children]
    spine_splits = 0
    node = root
    while node.children:
        spine_splits += 1
        node = max(node.children, key=lambda child: (child.size, -min(child.indices)))
    colless = sum(abs(node.children[0].size - node.children[1].size) for node in internal)
    maximum_colless = (root.size - 1) * (root.size - 2) / 2.0
    return {
        "internal_node_count": len(internal),
        "maximum_depth": max(leaf_depths),
        "mean_leaf_depth": float(np.mean(leaf_depths)),
        "dominant_spine_split_count": spine_splits,
        "dominant_spine_fraction": float(spine_splits / len(internal)),
        "normalized_colless_imbalance": float(colless / maximum_colless),
    }


def render_tree(
    root: ScalableHierarchyNode,
    output: Path,
    *,
    labels: Sequence[str],
    network_membership: np.ndarray,
    network_names: Sequence[str],
    full_xi: float,
    within_roi_xi: float,
    seed: int,
    coupling_g: float,
    dpi: int,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    all_nodes = flatten_nodes(root)
    internal = [node for node in all_nodes if node.children]
    order = _leaf_order(root)
    leaf_x = {index: float(position) for position, index in enumerate(order)}
    positions: dict[tuple[int, ...], tuple[float, float]] = {}

    def position(node: ScalableHierarchyNode) -> tuple[float, float]:
        if node.indices in positions:
            return positions[node.indices]
        if not node.children:
            point = (leaf_x[node.indices[0]], 0.0)
        else:
            children = [position(child) for child in node.children]
            x_value = sum(point[0] * child.size for point, child in zip(children, node.children, strict=True)) / node.size
            point = (x_value, math.log2(node.size) / math.log2(root.size))
        positions[node.indices] = point
        return point

    position(root)
    max_syn = max(float(node.syn_bits) for node in internal)
    figure, axis = plt.subplots(figsize=(15.2, 8.2), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")

    for node in internal:
        px, py = positions[node.indices]
        child_points = [positions[child.indices] for child in node.children]
        axis.plot(
            [child_points[0][0], child_points[1][0]], [py, py],
            color=EDGE_COLOR, linewidth=0.75, solid_capstyle="round", zorder=1,
        )
        for cx, cy in child_points:
            axis.plot([cx, cx], [cy, py], color=EDGE_COLOR, linewidth=0.75, zorder=1)

    ranked = sorted(
        (node for node in internal if node is not root and node.size >= 4),
        key=lambda node: (node.depth <= 2, node.syn_bits),
        reverse=True,
    )
    labeled: set[tuple[int, ...]] = set()
    label_points: list[tuple[float, float]] = []
    for node in ranked:
        x_value, y_value = positions[node.indices]
        if all(abs(x_value - old_x) >= 4.0 or abs(y_value - old_y) >= 0.075 for old_x, old_y in label_points):
            labeled.add(node.indices)
            label_points.append((x_value, y_value))
        if len(labeled) >= 12:
            break
    for node in internal:
        x_value, y_value = positions[node.indices]
        syn = 0.0 if -SYN_NONNEGATIVE_TOLERANCE_BITS <= node.syn_bits < 0.0 else float(node.syn_bits)
        relative = min(1.0, syn / max_syn) if max_syn > 0.0 else 0.0
        face = _blend_with_white(SPLIT_COLOR, 0.12 + 0.62 * relative)
        edge = _blend_with_white(SPLIT_COLOR, 0.55 + 0.45 * relative)
        if node.indices in labeled:
            axis.text(
                x_value, y_value, f"n={node.size}\nSyn {syn:.2f}",
                ha="center", va="center", fontsize=5.5, color=INK, linespacing=1.12,
                bbox={"boxstyle": "round,pad=0.27", "facecolor": face, "edgecolor": edge,
                      "linewidth": 0.8 + 1.7 * relative}, zorder=4,
            )
        else:
            axis.scatter(
                [x_value], [y_value], s=7.0 + 23.0 * relative,
                facecolor=face, edgecolor=edge, linewidth=0.6 + relative, zorder=3,
            )

    for index in order:
        x_value = leaf_x[index]
        color = NETWORK_COLORS[int(network_membership[index]) % len(NETWORK_COLORS)]
        axis.scatter([x_value], [-0.008], s=9, color=color, zorder=4, clip_on=False)
        axis.text(
            x_value, -0.027, _short_roi_label(labels[index]), rotation=90,
            ha="right", va="top", fontsize=3.7, color="#55616D", clip_on=False,
        )

    metrics = _tree_metrics(root)
    root_x, _ = positions[root.indices]
    axis.text(
        root_x, 1.115,
        rf"Cross-ROI $\Xi$ = {root.xi_bits:.2f} bits   |   full $\Xi$ = {full_xi:.2f} bits   |   within-ROI = {within_roi_xi:.2f} bits",
        ha="center", va="center", fontsize=9.0, color=INK,
    )
    axis.text(
        1.015, 0.98,
        f"DMF  G={coupling_g:.1f}  seed={seed}\n"
        f"100 ROI blocks (E/I paired)\n"
        f"max depth  {metrics['maximum_depth']}\n"
        f"dominant spine  {metrics['dominant_spine_fraction']:.1%}\n"
        f"Colless imbalance  {metrics['normalized_colless_imbalance']:.3f}",
        transform=axis.transAxes, ha="left", va="top", fontsize=7.0, color=INK,
        linespacing=1.5,
    )
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=5,
               markerfacecolor=NETWORK_COLORS[i], markeredgecolor="none", label=str(name))
        for i, name in enumerate(network_names)
    ]
    axis.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.48),
        frameon=False, fontsize=6.4, handletextpad=0.45,
    )
    axis.set_xlim(-1.0, root.size)
    axis.set_ylim(-0.30, 1.16)
    axis.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_analysis(
    input_path: Path,
    output_dir: Path,
    figure_path: Path,
    *,
    coupling_g: float,
    exact_max_size: int,
    dpi: int,
) -> dict[str, object]:
    with np.load(input_path, allow_pickle=True) as archive:
        g_values = np.asarray(archive["G"], dtype=float)
        g_index = int(np.flatnonzero(np.isclose(g_values, coupling_g))[0])
        seeds = np.asarray(archive["seeds"], dtype=int)
        cross_roi = np.asarray(archive["cross_roi"], dtype=float)[:, g_index]
        seed_index = int(np.argmin(np.abs(cross_roi - cross_roi.mean())))
        seed = int(seeds[seed_index])
        conditional = np.asarray(archive["conditional_covariance"], dtype=float)[seed_index, g_index]
        labels = [str(value) for value in np.asarray(archive["region_labels"], dtype=object)]
        network_names = [str(value) for value in np.asarray(archive["network_names"], dtype=object)]
        membership = np.asarray(archive["network_membership"], dtype=int)
        full_xi = float(np.asarray(archive["fine_phi"], dtype=float)[seed_index, g_index])
        within_roi_xi = float(np.asarray(archive["within_roi"], dtype=float)[seed_index, g_index])

    roi_count = len(labels)
    roi_blocks = tuple((index, index + roi_count) for index in range(roi_count))
    oracle = ConditionalBlockXiOracle(conditional, roi_blocks)
    affinity, pair_tolerance_zero_count = pairwise_syn_affinity(
        oracle, roi_count, tolerance=SYN_NONNEGATIVE_TOLERANCE_BITS,
    )
    audit = {"candidate_count": 0, "tolerance_zero_count": 0}
    tree = build_scalable_hierarchy(
        tuple(range(roi_count)), oracle, affinity,
        exact_max_size=int(exact_max_size), tolerance=SYN_NONNEGATIVE_TOLERANCE_BITS,
        audit=audit,
    )
    internal = [node for node in flatten_nodes(tree) if node.children]
    closure_error = float(sum(node.syn_bits for node in internal) - tree.xi_bits)
    if abs(closure_error) > 1.0e-8:
        raise RuntimeError(f"Hierarchy closure failed: error={closure_error:.12g} bits")
    render_tree(
        tree, figure_path, labels=labels, network_membership=membership,
        network_names=network_names, full_xi=full_xi, within_roi_xi=within_roi_xi,
        seed=seed, coupling_g=coupling_g, dpi=dpi,
    )
    metrics = _tree_metrics(tree)
    payload: dict[str, object] = {
        "experiment": "Schaefer100 DMF representative 100-ROI Xi hierarchy",
        "status": "approximate spectral-candidate hierarchy; exact for coalitions at or below the declared size",
        "selection": "At G=1.3, choose the seed whose cross-ROI Xi is nearest the eight-seed mean; topology is not averaged",
        "coupling_g": float(coupling_g),
        "seed": seed,
        "seed_cross_roi_xi_bits": float(tree.xi_bits),
        "eight_seed_cross_roi_xi_mean_bits": float(cross_roi.mean()),
        "full_xi_bits": full_xi,
        "within_roi_xi_bits": within_roi_xi,
        "roi_count": roi_count,
        "syn_nonnegative_tolerance_bits": SYN_NONNEGATIVE_TOLERANCE_BITS,
        "pair_tolerance_zero_count": int(pair_tolerance_zero_count),
        "split_tolerance_zero_count": int(audit["tolerance_zero_count"]),
        "significant_nonnegativity_violation_count": 0,
        "candidate_split_count": int(audit["candidate_count"]),
        "coalition_evaluation_count": int(oracle.evaluations),
        "exact_search_max_coalition_size": int(exact_max_size),
        "closure_error_bits": closure_error,
        "figure": str(figure_path),
        "tree_metrics": metrics,
        "tree": _node_record(tree),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--coupling-g", type=float, default=1.3)
    parser.add_argument("--exact-max-size", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_analysis(
        args.input, args.output_dir, args.figure,
        coupling_g=args.coupling_g, exact_max_size=args.exact_max_size, dpi=args.dpi,
    )
    print(
        f"[done] seed={payload['seed']} cross-ROI Xi={payload['seed_cross_roi_xi_bits']:.6f} bits; "
        f"spine={payload['tree_metrics']['dominant_spine_fraction']:.3f}; "
        f"imbalance={payload['tree_metrics']['normalized_colless_imbalance']:.3f}"
    )


if __name__ == "__main__":
    main()
