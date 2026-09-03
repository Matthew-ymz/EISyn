#!/usr/bin/env python3
"""Controlled H=1/10/60 comparison of the Runge SLP 60-PC Xi hierarchy."""

from __future__ import annotations

import argparse
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
    DEFAULT_ROLLOUT,
    DEFAULT_SOURCE,
    SYN_NONNEGATIVE_TOLERANCE_BITS,
    ScalableHierarchyNode,
    EDGE_COLOR,
    INK,
    SPLIT_COLOR,
    _blend_with_white,
    _compact_positions,
    _dominant_spine,
    flatten_nodes,
    render_compact_tree,
    run_analysis,
)
from scripts.phi_hierarchy import ALL_ORDER_CROSS_DENSITY, RAW_RESIDUAL


DEFAULT_OUTPUT_ROOT = ROOT / "results/runge/slp_pc60_xi_hierarchy"
DEFAULT_FIGURE_ROOT = ROOT / "fig"
DEFAULT_HORIZONS = (1, 10, 60)
DEFAULT_COMPARISON_FIGURE = DEFAULT_FIGURE_ROOT / "earth_slp_pc60_xi_hierarchy_vertical_comparison.png"
LEAF_FACE = "#E7EBEF"
LEAF_EDGE = "#73808C"


def _node_from_record(record: dict[str, object]) -> ScalableHierarchyNode:
    return ScalableHierarchyNode(
        sources=tuple(int(index) for index in record["indices"]),
        xi_value=float(record["xi_bits"]),
        syn_value=float(record["syn_bits_raw"]),
        depth=int(record["depth"]),
        split_kind=str(record["search_kind"]),
        children=tuple(_node_from_record(dict(child)) for child in record["children"]),
    )


def _figure_path(figure_root: Path, horizon: int, suffix: str = "") -> Path:
    return figure_root / f"earth_slp_pc60_xi_hierarchy_H{int(horizon):03d}{suffix}.png"


def _summary_row(tree: ScalableHierarchyNode, payload: dict[str, object]) -> dict[str, object]:
    internal = [node for node in flatten_nodes(tree) if node.children]
    leaves = [node for node in flatten_nodes(tree) if not node.children]
    spine, side = _dominant_spine(tree)
    multi_node_side = [branch for branch in side if branch is not None and branch.size > 1]
    peeled = [
        int(branch.indices[0])
        for branch in side
        if branch is not None and branch.size == 1
    ]
    top = max(internal, key=lambda node: float(node.syn_bits))
    colless = sum(abs(node.children[0].size - node.children[1].size) for node in internal)
    maximum_colless = (tree.size - 1) * (tree.size - 2) / 2.0
    return {
        "horizon": int(payload["horizon"]),
        "xi_bits": float(tree.xi_bits),
        "root_syn_bits": float(tree.syn_bits),
        "maximum_syn_bits": float(top.syn_bits),
        "maximum_syn_coalition_size": int(top.size),
        "maximum_syn_indices": list(top.indices),
        "dominant_spine_split_count": len(spine) - 1,
        "dominant_spine_fraction": float((len(spine) - 1) / len(internal)),
        "maximum_depth": max(node.depth for node in leaves),
        "normalized_colless_imbalance": float(colless / maximum_colless),
        "first_ten_peeled_indices": peeled[:10],
        "terminal_pair": list(spine[-2].indices) if len(spine) >= 2 else list(spine[-1].indices),
        "multi_node_side_branches": [
            {
                "indices": list(branch.indices),
                "size": branch.size,
                "syn_bits": float(branch.syn_bits),
            }
            for branch in multi_node_side
        ],
        "candidate_split_count": int(payload["candidate_split_count"]),
        "coalition_evaluation_count": int(payload["coalition_evaluation_count"]),
        "closure_error_bits": float(payload["closure_error_bits"]),
        "pair_tolerance_zero_count": int(payload["pair_tolerance_zero_count"]),
        "split_tolerance_zero_count": int(payload["split_tolerance_zero_count"]),
        "significant_nonnegativity_violation_count": int(
            payload["significant_nonnegativity_violation_count"]
        ),
        "figure": str(payload["figure"]),
    }


def _syn_style(
    value: float,
    *,
    scale_max: float,
    tolerance: float,
) -> tuple[float, str, str, float]:
    """Map one nonnegative Syn value to point area, fill, edge, and edge width."""
    if value < -float(tolerance):
        raise RuntimeError(
            f"Cannot render significant negative Syn {value:.12g} bits; "
            f"tolerance={tolerance:.12g} bits."
        )
    display_value = 0.0 if value < 0.0 else float(value)
    relative = min(1.0, display_value / max(float(scale_max), 1.0e-15))
    area = 9.0 + 72.0 * math.sqrt(relative)
    face = _blend_with_white(SPLIT_COLOR, 0.10 + 0.78 * relative)
    edge = _blend_with_white(SPLIT_COLOR, 0.52 + 0.48 * relative)
    edge_width = 0.45 + 1.35 * relative
    return area, face, edge, edge_width


def _draw_side_subtree(
    axis: plt.Axes,
    node: ScalableHierarchyNode,
    *,
    x_value: float,
    y_value: float,
    direction: float,
    level_gap: float,
    syn_scale_max: float,
    tolerance: float,
) -> None:
    """Draw a compact text-free side subtree attached to the dominant spine."""
    if not node.children:
        axis.scatter(
            [x_value],
            [y_value],
            s=8.0,
            facecolor=LEAF_FACE,
            edgecolor=LEAF_EDGE,
            linewidth=0.55,
            zorder=4,
        )
        return

    area, face, edge, edge_width = _syn_style(
        float(node.syn_bits),
        scale_max=syn_scale_max,
        tolerance=tolerance,
    )
    axis.scatter(
        [x_value],
        [y_value],
        s=area,
        facecolor=face,
        edgecolor=edge,
        linewidth=edge_width,
        zorder=4,
    )
    child_y = y_value - 0.52 * level_gap
    spread = min(0.055, 0.018 + 0.004 * node.size)
    child_x = [x_value - direction * spread, x_value + direction * spread]
    axis.plot(
        [x_value, x_value],
        [y_value, child_y],
        color=EDGE_COLOR,
        linewidth=0.65,
        solid_capstyle="round",
        zorder=1,
    )
    axis.plot(
        child_x,
        [child_y, child_y],
        color=EDGE_COLOR,
        linewidth=0.65,
        solid_capstyle="round",
        zorder=1,
    )
    for child, child_x_value in zip(node.children, child_x, strict=True):
        _draw_side_subtree(
            axis,
            child,
            x_value=child_x_value,
            y_value=child_y,
            direction=direction,
            level_gap=0.70 * level_gap,
            syn_scale_max=syn_scale_max,
            tolerance=tolerance,
        )


def draw_vertical_point_tree(
    axis: plt.Axes,
    root: ScalableHierarchyNode,
    *,
    horizon: int,
    syn_scale_max: float,
    tolerance: float,
) -> None:
    """Draw a dense hierarchy as a Brain-style triangular point tree."""
    positions, leaf_order = _compact_positions(root)
    internal = [node for node in flatten_nodes(root) if node.children]
    leaves = [node for node in flatten_nodes(root) if not node.children]

    for node in internal:
        _, parent_y = positions[node.indices]
        child_points = [positions[child.indices] for child in node.children]
        axis.plot(
            [child_points[0][0], child_points[1][0]],
            [parent_y, parent_y],
            color=EDGE_COLOR,
            linewidth=0.60,
            solid_capstyle="round",
            zorder=1,
        )
        for child_x, child_y in child_points:
            axis.plot(
                [child_x, child_x], [child_y, parent_y],
                color=EDGE_COLOR, linewidth=0.60,
                solid_capstyle="round", zorder=1,
            )

    for node in internal:
        x_value, y_value = positions[node.indices]
        area, face, edge, edge_width = _syn_style(
            float(node.syn_bits),
            scale_max=syn_scale_max,
            tolerance=tolerance,
        )
        axis.scatter(
            [x_value], [y_value], s=0.72 * area,
            facecolor=face, edgecolor=edge,
            linewidth=0.85 * edge_width, zorder=4,
        )

    for node in leaves:
        x_value, y_value = positions[node.indices]
        axis.scatter(
            [x_value], [y_value], s=7.0,
            facecolor=LEAF_FACE, edgecolor=LEAF_EDGE,
            linewidth=0.50, zorder=4,
        )

    axis.text(
        0.01,
        1.04,
        rf"$H={int(horizon)}$",
        transform=axis.transAxes,
        ha="left",
        va="center",
        color=INK,
        fontsize=8.8,
        fontweight="semibold",
    )
    axis.text(
        0.10,
        1.04,
        rf"$\Xi={root.xi_bits:.2f}$ bits",
        transform=axis.transAxes,
        ha="left",
        va="center",
        color="#596570",
        fontsize=7.2,
    )
    axis.set_xlim(-1.0, len(leaf_order))
    axis.set_ylim(-0.035, 1.15)
    axis.axis("off")


def render_vertical_point_comparison(
    trees: list[ScalableHierarchyNode],
    horizons: tuple[int, ...],
    output_path: Path,
    *,
    tolerance: float,
    syn_scale_max: float,
    dpi: int = 600,
) -> Path:
    """Render aligned triangular trees with one shared Syn visual scale."""
    if len(trees) != len(horizons):
        raise ValueError("trees and horizons must have the same length")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(
        len(trees),
        1,
        figsize=(13.8, 9.4),
        constrained_layout=True,
        squeeze=False,
    )
    figure.patch.set_facecolor("white")
    for axis, tree, horizon in zip(axes[:, 0], trees, horizons, strict=True):
        axis.set_facecolor("white")
        draw_vertical_point_tree(
            axis,
            tree,
            horizon=int(horizon),
            syn_scale_max=float(syn_scale_max),
            tolerance=float(tolerance),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def run_comparison(
    source_path: Path,
    rollout_path: Path,
    output_root: Path,
    figure_root: Path,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    exact_max_size: int = 14,
    covariance_ridge: float = 1.0e-6,
    split_objective: str = RAW_RESIDUAL,
    candidate_strategy: str = "stratified_random_local",
    initial_candidate_budget: int = 8_000,
    total_candidate_budget: int = 10_000,
    local_search_top_k: int = 8,
    search_seed: int = 0,
    figure_suffix: str = "",
    dpi: int = 600,
) -> dict[str, object]:
    payloads: list[dict[str, object]] = []
    trees: list[ScalableHierarchyNode] = []
    for horizon in horizons:
        output_dir = output_root / f"H{int(horizon):03d}"
        figure_path = _figure_path(figure_root, horizon, figure_suffix)
        payload = run_analysis(
            source_path,
            rollout_path,
            output_dir,
            figure_path,
            horizon=int(horizon),
            exact_max_size=int(exact_max_size),
            covariance_ridge=float(covariance_ridge),
            split_objective=split_objective,
            candidate_strategy=candidate_strategy,
            initial_candidate_budget=int(initial_candidate_budget),
            total_candidate_budget=int(total_candidate_budget),
            local_search_top_k=int(local_search_top_k),
            search_seed=int(search_seed),
            dpi=int(dpi),
        )
        payloads.append(payload)
        trees.append(_node_from_record(dict(payload["tree"])))

    shared_syn_scale = max(
        float(node.syn_bits)
        for tree in trees
        for node in flatten_nodes(tree)
        if node.children
    )
    for tree, payload in zip(trees, payloads, strict=True):
        figure_path = Path(str(payload["figure"]))
        render_compact_tree(
            tree,
            figure_path,
            tolerance=SYN_NONNEGATIVE_TOLERANCE_BITS,
            syn_scale_max=shared_syn_scale,
            dpi=dpi,
        )

    comparison_figure = figure_root / f"{DEFAULT_COMPARISON_FIGURE.stem}{figure_suffix}.png"
    render_vertical_point_comparison(
        trees,
        horizons,
        comparison_figure,
        tolerance=SYN_NONNEGATIVE_TOLERANCE_BITS,
        syn_scale_max=shared_syn_scale,
        dpi=dpi,
    )

    summary = {
        "experiment": "Controlled Runge SLP PC60 Xi hierarchy horizon comparison",
        "scientific_question": "What changes when only forecast horizon H changes?",
        "treatment_factor": "forecast horizon H",
        "horizons": list(horizons),
        "paired_intervention_rows": 4096,
        "fixed_source_dimension": 60,
        "fixed_target_dimension": 60,
        "fixed_estimator": "affine degree-1 TM / linear-Gaussian log-det equivalent",
        "fixed_covariance_ridge": float(covariance_ridge),
        "fixed_exact_search_max_coalition_size": int(exact_max_size),
        "candidate_strategy": str(candidate_strategy),
        "initial_candidate_budget_per_large_node": int(initial_candidate_budget),
        "total_candidate_budget_per_large_node": int(total_candidate_budget),
        "local_search_top_k": int(local_search_top_k),
        "search_seed": int(search_seed),
        "split_objective": str(split_objective),
        "split_objective_denominator": (
            "(2^|A| - 1)(2^|B| - 1)"
            if split_objective == ALL_ORDER_CROSS_DENSITY
            else "1"
        ),
        "reported_node_value": "raw unnormalized Syn residual in bits",
        "syn_nonnegative_tolerance_bits": SYN_NONNEGATIVE_TOLERANCE_BITS,
        "shared_syn_visual_scale_max_bits": float(shared_syn_scale),
        "comparison_figure": str(comparison_figure),
        "rows": [_summary_row(tree, payload) for tree, payload in zip(trees, payloads, strict=True)],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "horizon_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--rollout", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    parser.add_argument("--horizons", default="1,10,60")
    parser.add_argument("--exact-max-size", type=int, default=14)
    parser.add_argument("--covariance-ridge", type=float, default=1.0e-6)
    parser.add_argument(
        "--split-objective",
        choices=(RAW_RESIDUAL, ALL_ORDER_CROSS_DENSITY),
        default=RAW_RESIDUAL,
    )
    parser.add_argument("--figure-suffix", default="")
    parser.add_argument(
        "--candidate-strategy",
        choices=("spectral", "stratified_random_local"),
        default="stratified_random_local",
    )
    parser.add_argument("--initial-candidate-budget", type=int, default=8_000)
    parser.add_argument("--total-candidate-budget", type=int, default=10_000)
    parser.add_argument("--local-search-top-k", type=int, default=8)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = tuple(int(value.strip()) for value in str(args.horizons).split(",") if value.strip())
    summary = run_comparison(
        args.source,
        args.rollout,
        args.output_root,
        args.figure_root,
        horizons=horizons,
        exact_max_size=args.exact_max_size,
        covariance_ridge=args.covariance_ridge,
        split_objective=args.split_objective,
        candidate_strategy=args.candidate_strategy,
        initial_candidate_budget=args.initial_candidate_budget,
        total_candidate_budget=args.total_candidate_budget,
        local_search_top_k=args.local_search_top_k,
        search_seed=args.search_seed,
        figure_suffix=str(args.figure_suffix),
        dpi=args.dpi,
    )
    values = ", ".join(
        f"H={row['horizon']}: Xi={row['xi_bits']:.4f}" for row in summary["rows"]
    )
    print(f"[done] {values}")


if __name__ == "__main__":
    main()
