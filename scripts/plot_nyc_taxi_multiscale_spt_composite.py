#!/usr/bin/env python3
"""Build a unified NYC Taxi multiscale-synergy and SPT main figure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_nyc_taxi_mgstn_spt import (
    TIME_COLORS,
    deserialize_tree,
    flatten_nodes,
    plot_tree,
    representative_condition,
)
from plot_nyc_taxi_mgstn_ei import (
    DATA,
    FINITE,
    add_map_panels,
    add_synergy_flow_panel,
    add_temporal_share_panel,
    add_time_series_panel,
)


SPT_RUN = (
    ROOT
    / "results/nyc_taxi_mgstn_spt"
    / "affine_regression_v3_time_block_search_n4096_pc2_r1e-06_exact8"
)
OUTPUT = ROOT / "fig/nyc_taxi_multiscale_spt_composite.png"


def _load_spt() -> tuple[object, np.ndarray, mpl.colors.Normalize, int, str]:
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(SPT_RUN.glob("seed_*/*/affine_summary.json"))
    ]
    if not summaries:
        raise RuntimeError("No cached NYC Taxi SPT summaries were found.")
    seed, state = representative_condition(summaries)
    chosen = next(
        row for row in summaries if row["seed"] == seed and row["state"] == state
    )
    roots = {
        name: deserialize_tree(chosen[name]["tree"])
        for name in ("unconstrained", "time_prior")
    }
    norm = mpl.colors.SymLogNorm(
        linthresh=0.001,
        linscale=0.5,
        vmin=0,
        vmax=max(
            node.syn_bits
            for root in roots.values()
            for node in flatten_nodes(root)
        ),
        base=10,
    )
    zone_ids = np.asarray(
        [chosen["unconstrained"]["tree"]["members"][index * 3]["zone_id"] for index in range(66)]
    )
    return roots["unconstrained"], zone_ids, norm, seed, state


def main() -> None:
    saved = np.load(DATA, allow_pickle=False)
    flow = saved["flow"]
    data_zone_ids = saved["zone_ids"]
    finite = json.loads(FINITE.read_text(encoding="utf-8"))
    if finite["nonnegative_audit"]["hurdle"]["violation_count"]:
        raise RuntimeError("Formal estimator failed its declared nonnegative audit.")
    root, zone_ids, norm, seed, state = _load_spt()

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "savefig.facecolor": "white",
        }
    )

    figure = plt.figure(figsize=(12.0, 12.2), layout="constrained", facecolor="white")
    outer = figure.add_gridspec(
        3,
        1,
        height_ratios=(0.60, 1.05, 1.25),
        hspace=0.018,
    )

    top = outer[0].subgridspec(
        1,
        3,
        width_ratios=(1.72, 1.0, 0.82),
        wspace=0.11,
    )
    add_time_series_panel(
        figure.add_subplot(top[0, 0]),
        flow,
        data_zone_ids,
        panel="a",
    )
    add_synergy_flow_panel(
        figure.add_subplot(top[0, 1]),
        finite,
        flow,
        data_zone_ids,
        panel="b",
    )
    add_temporal_share_panel(
        figure.add_subplot(top[0, 2]),
        finite,
        panel="c",
    )

    middle = outer[1].subgridspec(
        2,
        2,
        height_ratios=(0.10, 0.90),
        width_ratios=(1.0, 0.018),
        wspace=0.015,
        hspace=0.0,
    )
    header = figure.add_subplot(middle[0, :])
    header.axis("off")
    header.text(
        0.0,
        0.55,
        "d",
        ha="left",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#111111",
    )
    header.text(
        0.025,
        0.55,
        (
            f"Unconstrained SPT | seed {seed}, {state.replace('_', ' ')} | "
            f"joint-target affine Xi = {root.xi_bits:.2f} bits"
        ),
        ha="left",
        va="center",
        fontsize=8.0,
        color="#24313C",
    )
    handles = [
        mpl.lines.Line2D([], [], marker="o", ls="", ms=4, color=color, label=name)
        for name, color in TIME_COLORS.items()
    ]
    header.legend(
        handles=handles,
        loc="center right",
        ncol=3,
        frameon=False,
        title="Source atom time scale",
        borderaxespad=0.0,
    )

    tree_axis = figure.add_subplot(middle[1, 0])
    plot_tree(
        tree_axis,
        root,
        label="",
        zone_ids=zone_ids,
        norm=norm,
        visual_scale=1.22,
    )
    colorbar_axis = figure.add_subplot(middle[1, 1])
    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="YlGnBu"),
        cax=colorbar_axis,
        ticks=(0, 0.001, 0.01, 0.1, 1, 10),
    )
    colorbar.ax.set_yticklabels(("0", ".001", ".01", ".1", "1", "10"))
    colorbar.set_label("Local Syn (bits; logarithmic color above .001)")
    colorbar.outline.set_linewidth(0.6)

    add_map_panels(figure, outer[2], finite, panel="e")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
