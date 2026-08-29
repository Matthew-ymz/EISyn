#!/usr/bin/env python3
"""Controlled H=1/10/60 comparison of the Runge SLP 60-PC Xi hierarchy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_runge_slp_pc60_xi_hierarchy import (
    DEFAULT_ROLLOUT,
    DEFAULT_SOURCE,
    SYN_NONNEGATIVE_TOLERANCE_BITS,
    ScalableHierarchyNode,
    _dominant_spine,
    flatten_nodes,
    render_backbone_tree,
    render_compact_tree,
    run_analysis,
)


DEFAULT_OUTPUT_ROOT = ROOT / "results/runge/slp_pc60_xi_hierarchy"
DEFAULT_FIGURE_ROOT = ROOT / "fig"
DEFAULT_HORIZONS = (1, 10, 60)


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
