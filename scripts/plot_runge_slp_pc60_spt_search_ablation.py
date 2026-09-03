#!/usr/bin/env python3
"""Compare spectral and stratified-random-local SPT search at SLP H=60."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from scripts.analyze_runge_slp_pc60_xi_hierarchy import (
    SYN_NONNEGATIVE_TOLERANCE_BITS,
    flatten_nodes,
)
from scripts.compare_runge_slp_pc60_xi_horizons import (
    _node_from_record,
    draw_vertical_point_tree,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results/runge/slp_pc60_xi_hierarchy/H060/summary.json"
PROPOSED = ROOT / "results/runge/slp_pc60_xi_hierarchy_stratified_random_local/H060_seed0/summary.json"
OUTPUT = ROOT / "fig/earth_slp_pc60_spt_search_ablation_H060.png"


def main() -> None:
    records = [
        json.loads(BASELINE.read_text(encoding="utf-8")),
        json.loads(PROPOSED.read_text(encoding="utf-8")),
    ]
    trees = [_node_from_record(record["tree"]) for record in records]
    scale = max(
        float(node.syn_bits)
        for tree in trees
        for node in flatten_nodes(tree)
        if node.children
    )
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), constrained_layout=True)
    labels = (
        ("a", "Spectral candidates"),
        ("b", "Stratified random + local search"),
    )
    for axis, tree, (panel, method) in zip(axes, trees, labels, strict=True):
        draw_vertical_point_tree(
            axis,
            tree,
            horizon=60,
            syn_scale_max=scale,
            tolerance=SYN_NONNEGATIVE_TOLERANCE_BITS,
        )
        axis.text(
            0.01, 1.13, panel, transform=axis.transAxes,
            ha="left", va="center", fontsize=9.0, fontweight="bold",
        )
        axis.text(
            0.055, 1.13, method, transform=axis.transAxes,
            ha="left", va="center", fontsize=8.0, fontweight="semibold",
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
