#!/usr/bin/env python3
"""Compare four cached Kuramoto structures using the squarified SPT renderer.

Evidence: increasing inter-module coupling moves Syn mass from within-module
atoms into larger coalitions. This is a visualization of seed 0 cached point
estimates, with no new simulation, fitting, averaging, or uncertainty estimate.
Every panel has its own area normalization; color uses a shared scale in bits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_kuramoto_xi_hierarchy_trees import DEFAULT_SUMMARY, _condition_rows, _tree
from scripts.synergy_hierarchy_treemap import draw_synergy_treemap
from scripts.spt import flatten_nodes


def render_preview(summary_path: Path, output: Path, *, dpi=450):
    rows = _condition_rows(json.loads(summary_path.read_text()), seed=0)
    if len(rows) != 4:
        raise ValueError("Expected all four cached Kuramoto conditions for seed 0")
    trees = [_tree(row) for row in rows]
    scale = max(node.residual for tree in trees for node in flatten_nodes(tree))
    labels = {f"theta{i}": rf"$\theta_{{{i}}}$" for i in range(1, 7)}
    with mpl.rc_context({"font.family": "sans-serif", "font.size": 9}):
        fig = plt.figure(figsize=(15, 9))
        grid = fig.add_gridspec(2, 4, width_ratios=[1, .48, 1, .48],
                               left=.025, right=.985, bottom=.16, top=.94,
                               hspace=.38, wspace=.08)
        audits = []
        for i, (row, tree) in enumerate(zip(rows, trees, strict=True)):
            ax = fig.add_subplot(grid[i // 2, 2 * (i % 2)])
            audit = draw_synergy_treemap(ax, tree, source_labels=labels,
                                         syn_scale_max=scale)
            audits.append(audit)
            ax.text(0, 1.08,
                    rf"$K_{{\mathrm{{out}}}}={float(row['cross_coupling']):g}$"
                    + rf"   |   $\Xi={tree.phi_value:.3f}$ bits",
                    transform=ax.transAxes, fontsize=12, color="#24313C", weight="bold")
        bar_ax = fig.add_axes([.025, .086, .23, .017])
        cmap = mpl.colors.LinearSegmentedColormap.from_list("syn_teal", ["#EAF3F1", "#267A70"])
        colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, scale), cmap=cmap),
                               cax=bar_ax, orientation="horizontal")
        colorbar.set_label("Local Syn (bits); shared color scale", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
        colorbar.outline.set_visible(False)
        fig.text(.30, .09, "Area = local Syn / panel total Syn; dark frames = nested subtrees.\n"
                 "Zero-area atoms remain in the ledger; singleton membership is retained in coalition labels.",
                 fontsize=9, va="center", color="#24313C", linespacing=1.7)
        fig.text(.30, .038, "Cached seed 0 estimates · tolerance 1e-10 bits · "
                 f"{sum(a['tolerance_zero_count'] for a in audits)} negatives displayed as zero",
                 fontsize=8, color="#566573")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, facecolor="white", metadata={
            "Source": str(summary_path.relative_to(ROOT)) if summary_path.is_relative_to(ROOT) else str(summary_path),
            "SPT numerical audits": json.dumps(audits),
            "Description": "Four nested squarified SPTs from cached Kuramoto estimates. "
                           "As coupling increases, larger coalitions account for more Syn mass. "
                           "Area is normalized within each panel; color has a common bits scale."})
        plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=ROOT / "fig/spt_squarified_kuramoto.png")
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()
    render_preview(args.summary, args.output, dpi=args.dpi)
