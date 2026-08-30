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
    _dominant_spine,
    flatten_nodes,
    render_backbone_tree,
    render_compact_tree,
    run_analysis,
)


DEFAULT_OUTPUT_ROOT = ROOT / "results/runge/slp_pc60_xi_hierarchy"
DEFAULT_FIGURE_ROOT = ROOT / "fig"
DEFAULT_HORIZONS = (1, 10, 60)
DEFAULT_COMPARISON_FIGURE = DEFAULT_FIGURE_ROOT / "earth_slp_pc60_xi_hierarchy_vertical_comparison.png"
LEAF_FACE = "#E7EBEF"
LEAF_EDGE = "#73808C"


def _node_from_record(record: dict[str, object]) -> ScalableHierarchyNode:
    return ScalableHierarchyNode(
        indices=tuple(int(index) for index in record["indices"]),
        xi_bits=float(record["xi_bits"]),
        syn_bits=float(record["syn_bits_raw"]),
        depth=int(record["depth"]),
        search_kind=str(record["search_kind"]),
        children=tuple(_node_from_record(dict(child)) for child in record["children"]),
    )


def _figure_path(figure_root: Path, horizon: int) -> Path:
    return figure_root / f"earth_slp_pc60_xi_hierarchy_H{int(horizon):03d}.png"


def _summary_row(tree: ScalableHierarchyNode, payload: dict[str, object]) -> dict[str, object]:
    internal = [node for node in flatten_nodes(tree) if node.children]
    spine, side = _dominant_spine(tree)
    multi_node_side = [branch for branch in side if branch is not None and branch.size > 1]
    peeled = [
        int(branch.indices[0])
        for branch in side
        if branch is not None and branch.size == 1
    ]
    top = max(internal, key=lambda node: float(node.syn_bits))
    return {
        "horizon": int(payload["horizon"]),
        "xi_bits": float(tree.xi_bits),
        "root_syn_bits": float(tree.syn_bits),
        "maximum_syn_bits": float(top.syn_bits),
        "maximum_syn_coalition_size": int(top.size),
        "maximum_syn_indices": list(top.indices),
        "dominant_spine_split_count": len(spine) - 1,
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
    """Draw a dense hierarchy as a vertical, label-free point tree."""
    spine, side_branches = _dominant_spine(root)
    top, bottom = 0.965, 0.035
    level_gap = (top - bottom) / max(1, len(spine) - 1)
    y_values = [top - position * level_gap for position in range(len(spine))]
    center_x = 0.50

    if len(spine) > 1:
        axis.plot(
            [center_x, center_x],
            [y_values[-1], y_values[0]],
            color=EDGE_COLOR,
            linewidth=0.70,
            solid_capstyle="round",
            zorder=1,
        )

    for position, (node, side) in enumerate(zip(spine, side_branches, strict=True)):
        y_value = y_values[position]
        if node.children:
            area, face, edge, edge_width = _syn_style(
                float(node.syn_bits),
                scale_max=syn_scale_max,
                tolerance=tolerance,
            )
            axis.scatter(
                [center_x],
                [y_value],
                s=area,
                facecolor=face,
                edgecolor=edge,
                linewidth=edge_width,
                zorder=4,
            )
        else:
            axis.scatter(
                [center_x],
                [y_value],
                s=8.0,
                facecolor=LEAF_FACE,
                edgecolor=LEAF_EDGE,
                linewidth=0.55,
                zorder=4,
            )

        if side is None or position + 1 >= len(y_values):
            continue
        branch_y = y_values[position + 1]
        direction = -1.0 if position % 2 == 0 else 1.0
        branch_x = center_x + direction * (0.27 if side.size == 1 else 0.30)
        axis.plot(
            [center_x, branch_x],
            [branch_y, branch_y],
            color=EDGE_COLOR,
            linewidth=0.62,
            solid_capstyle="round",
            zorder=1,
        )
        _draw_side_subtree(
            axis,
            side,
            x_value=branch_x,
            y_value=branch_y,
            direction=direction,
            level_gap=level_gap,
            syn_scale_max=syn_scale_max,
            tolerance=tolerance,
        )

    axis.text(
        0.50,
        1.075,
        rf"$H={int(horizon)}$",
        ha="center",
        va="center",
        color=INK,
        fontsize=8.8,
        fontweight="semibold",
    )
    axis.text(
        0.50,
        1.025,
        rf"$\Xi={root.xi_bits:.2f}$ bits",
        ha="center",
        va="center",
        color="#596570",
        fontsize=7.2,
    )
    axis.set_xlim(0.05, 0.95)
    axis.set_ylim(0.0, 1.11)
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
    """Render aligned text-free trees with one shared Syn visual scale."""
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
        1,
        len(trees),
        figsize=(7.2, 8.8),
        constrained_layout=True,
        squeeze=False,
    )
    figure.patch.set_facecolor("white")
    for axis, tree, horizon in zip(axes[0], trees, horizons, strict=True):
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
    exact_max_size: int = 10,
    covariance_ridge: float = 1.0e-6,
    dpi: int = 600,
) -> dict[str, object]:
    payloads: list[dict[str, object]] = []
    trees: list[ScalableHierarchyNode] = []
    for horizon in horizons:
        output_dir = output_root / f"H{int(horizon):03d}"
        figure_path = _figure_path(figure_root, horizon)
        payload = run_analysis(
            source_path,
            rollout_path,
            output_dir,
            figure_path,
            horizon=int(horizon),
            exact_max_size=int(exact_max_size),
            covariance_ridge=float(covariance_ridge),
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
        spine, _ = _dominant_spine(tree)
        internal_count = sum(bool(node.children) for node in flatten_nodes(tree))
        if len(spine) - 1 >= 0.65 * internal_count:
            render_backbone_tree(
                tree,
                figure_path,
                tolerance=SYN_NONNEGATIVE_TOLERANCE_BITS,
                syn_scale_max=shared_syn_scale,
                dpi=dpi,
            )
        else:
            render_compact_tree(
                tree,
                figure_path,
                tolerance=SYN_NONNEGATIVE_TOLERANCE_BITS,
                syn_scale_max=shared_syn_scale,
                dpi=dpi,
            )

    comparison_figure = figure_root / DEFAULT_COMPARISON_FIGURE.name
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
    parser.add_argument("--exact-max-size", type=int, default=10)
    parser.add_argument("--covariance-ridge", type=float, default=1.0e-6)
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
        dpi=args.dpi,
    )
    values = ", ".join(
        f"H={row['horizon']}: Xi={row['xi_bits']:.4f}" for row in summary["rows"]
    )
    print(f"[done] {values}")


if __name__ == "__main__":
    main()
