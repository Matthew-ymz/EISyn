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
    audit_nonnegative,
    AFFINE_TOLERANCE_BITS,
    TIME_NAMES,
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


def plot_compact_tree(axis, root, zone_ids, norm):
    """Collapse only singleton-peeling runs; preserve every nontrivial subtree."""
    audit_nonnegative([n.syn_bits for n in flatten_nodes(root) if n.children],
                      AFFINE_TOLERANCE_BITS, "compact figure")
    tips = []
    def build(node):
        view = {"node": node, "children": []}
        current, omitted = node, []
        while current.children:
            small, large = sorted(current.children, key=lambda child: child.size)
            if small.size != 1 or not large.children:
                break
            omitted.append(small)
            current = large
        if len(omitted) >= 4:
            view["children"] = [build(current)]
            view["collapsed"] = len(omitted)
        elif node.children:
            view["children"] = [build(child) for child in node.children]
        else:
            view["x"] = len(tips)
            tips.append(view)
        view["y"] = np.log2(node.size)
        if view["children"]:
            view["x"] = np.average([child["x"] for child in view["children"]],
                                     weights=[child["node"].size for child in view["children"]])
            if "collapsed" in view:
                view["x"] += 8
        return view
    tree = build(root)
    represented = set()
    def collect(view):
        represented.add(tuple(view["node"].indices))
        for child in view["children"]:
            collect(child)
    collect(tree)
    required = [n for n in flatten_nodes(root) if n.children and
                (n.size == 2 or min(c.size for c in n.children) > 1)]
    assert all(tuple(n.indices) in represented for n in required)
    print(f"Preserved all {sum(n.size == 2 for n in required)} paired subtrees")
    def draw(view):
        node = view["node"]
        if view["children"]:
            collapsed = view.get("collapsed", 0)
            for child in view["children"]:
                if collapsed:
                    axis.plot([view["x"], child["x"]], [view["y"], child["y"]],
                              color="#DA8438", lw=1.35, ls=(0, (4, 3)), zorder=2)
                    mx, my = (view["x"] + child["x"])/2, (view["y"] + child["y"])/2
                    axis.text(mx, my, r"$\cdots$", color="#DA8438", fontsize=14,
                              ha="center", va="center", bbox=dict(facecolor="white", edgecolor="none", pad=0.1), zorder=5)
                    axis.annotate(f"{collapsed} singleton splits omitted", (mx, my),
                                  xytext=(-12, 22), textcoords="offset points", fontsize=7, ha="right",
                                  color="#B86D2D", va="center")
                    for endpoint in (view, child):
                        axis.annotate(f"n={endpoint['node'].size}", (endpoint["x"], endpoint["y"]),
                                      xytext=(5, 5) if endpoint is view else (-5, 6),
                                      ha="left" if endpoint is view else "right",
                                      textcoords="offset points", fontsize=6.5)
                else:
                    axis.plot([view["x"], child["x"], child["x"]],
                              [view["y"], view["y"], child["y"]],
                              color="#AAB7BF", lw=.65, zorder=1)
                draw(child)
            value = 0.0 if node.syn_bits < 0 else node.syn_bits
            axis.scatter(view["x"], view["y"], s=24,
                         color=mpl.colormaps["YlGnBu"](norm(value)),
                         edgecolor="#35515D", linewidth=.4, zorder=3)
        else:
            atom = node.indices[0]
            axis.scatter(view["x"], 0, s=18, color=TIME_COLORS[TIME_NAMES[atom % 3]], zorder=4)
            view["label"] = f"{zone_ids[atom // 3]} {'RDW'[atom % 3]}"
    draw(tree)
    axis.set_xticks([t["x"] for t in tips], [t["label"] for t in tips], fontsize=5, rotation=90)
    axis.tick_params(axis="x", length=0)
    ticks = [1, 2, 4, 8, 16, 32, 66, 132, 198]
    axis.set_yticks(np.log2(ticks), [str(t) for t in ticks])
    axis.set_ylim(-.12, np.log2(root.size) + .65)
    axis.set_xlim(-.6, len(tips) - .4)
    axis.set_ylabel("Coalition size (log scale)")
    axis.set_xlabel("Source atoms: Zone ID × time scale (R: recent; D: daily; W: weekly)", fontsize=8)
    axis.spines[["top", "right", "bottom"]].set_visible(False)
    print(f"Compact tree: {root.size} atoms → {len(tips)} displayed tips")


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

    figure = plt.figure(figsize=(12.0, 10.8), layout="constrained", facecolor="white")
    outer = figure.add_gridspec(
        3,
        1,
        height_ratios=(0.60, 0.98, 1.18),
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
    share_axis = figure.add_subplot(top[0, 2])
    add_temporal_share_panel(share_axis, finite, panel="c")
    share_axis.texts[-1].set_position((-0.32, 1.01))

    middle = outer[1].subgridspec(
        2,
        2,
        height_ratios=(0.15, 0.85),
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
        0.045,
        0.72,
        (
            f"Unconstrained SPT | {state.replace('_', ' ')}\n"
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
        loc="lower left",
        ncol=3,
        frameon=False,
        title=None,
        borderaxespad=0.0,
        bbox_to_anchor=(0.035, -0.08),
    )

    tree_axis = figure.add_subplot(middle[1, 0])
    plot_compact_tree(tree_axis, root, zone_ids, norm)
    colorbar_axis = figure.add_subplot(middle[1, 1])
    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="YlGnBu"),
        cax=colorbar_axis,
        ticks=(0, 0.001, 0.01, 0.1, 1, 10),
    )
    colorbar.ax.set_yticklabels(("0", ".001", ".01", ".1", "1", "10"))
    colorbar.set_label("Local Syn (bits)", fontsize=8)
    colorbar.outline.set_linewidth(0.6)

    add_map_panels(figure, outer[2], finite, panel="e")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
