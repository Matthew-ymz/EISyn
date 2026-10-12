#!/usr/bin/env python3
"""Dendrograms with consecutive singleton exits sharing one displayed fork.

Only G=1.3 is condensed by default. The original binary SPT is untouched;
all individual ROI leaves and all other forks remain. A condensed fork's
color is the peak original local Syn in its run, never a new multiway Syn.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_dmf_schaefer100_xi_hierarchy_tree import (
    EDGE_COLOR, INK, SPLIT_COLOR, _blend_with_white, _leaf_order,
    _short_roi_label, _tree_metrics, render_tree,
)
from scripts.compare_runge_slp_pc60_xi_horizons import _node_from_record
from scripts.plot_dmf_schaefer100_tree_examples import NETWORK_COLORS, audit_tree
from scripts.plot_dmf_spt_peel_groups import ORIGINAL_SUMMARY, PAIRED_SUMMARY, YEO_METADATA, load_comparison
from scripts.spt import flatten_nodes


@dataclass(frozen=True)
class DisplayNode:
    indices: tuple[int, ...]
    represented: tuple = ()
    children: tuple["DisplayNode", ...] = ()

    @property
    def size(self):
        return len(self.indices)

    @property
    def collapsed(self):
        return len(self.represented) >= 2


def display_tree(root, *, collapse=True):
    """Contract maximal >=2-step singleton runs, retaining terminal pairs."""
    order = _leaf_order(root)
    leaf_x = {index: position for position, index in enumerate(order)}

    def build(node):
        if not node.children:
            return DisplayNode(node.indices)
        current, steps, exits = node, [], []
        if collapse:
            while current.children and current.size > 2:
                large, small = sorted(current.children, key=lambda c: (-c.size, c.indices))
                if small.size != 1:
                    break
                steps.append(current)
                exits.append(small)
                current = large
        if len(steps) >= 2:
            children = [build(current), *(build(leaf) for leaf in exits)]
            children.sort(key=lambda child: min(leaf_x[i] for i in child.indices))
            return DisplayNode(node.indices, tuple(steps), tuple(children))
        return DisplayNode(node.indices, (node,), tuple(build(c) for c in node.children))

    display = build(root)
    nodes, positions, levels = [], {}, {}

    def visit(node, depth):
        nodes.append(node)
        levels[id(node)] = depth
        if node.children:
            child_indices = [i for child in node.children for i in child.indices]
            if sorted(child_indices) != sorted(node.indices) or len(set(child_indices)) != node.size:
                raise ValueError("Displayed children must partition their original source set")
            points = [visit(child, depth + 1) for child in node.children]
            height = 1 + max(p[1] for p in points)
        else:
            if node.size != 1:
                raise ValueError("Every displayed leaf must remain one ROI")
            height = 0
        point = (float(np.mean([leaf_x[i] for i in node.indices])), height)
        positions[id(node)] = point
        return point

    visit(display, 0)
    internal = [n for n in flatten_nodes(root) if n.children]
    represented = [n for d in nodes for n in d.represented]
    if len(represented) != len(internal) or {id(n) for n in represented} != {id(n) for n in internal}:
        raise ValueError("Each original binary Syn must be represented exactly once")
    leaves = [n.indices[0] for n in nodes if not n.children]
    if sorted(leaves) != sorted(root.indices) or len(leaves) != root.size:
        raise ValueError("Lost or duplicated an ROI leaf")
    root_height = positions[id(display)][1]
    positions = {key: (x, h / root_height) for key, (x, h) in positions.items()}
    return display, nodes, order, positions, levels


def layout_audit(root, *, collapse=True):
    original = audit_tree(root, root.xi_bits)
    _, nodes, order, positions, levels = display_tree(root, collapse=collapse)
    internal = [n for n in nodes if n.children]
    return {**original, "original_tree_metrics": _tree_metrics(root),
            "collapsed": collapse, "display_internal_node_count": len(internal),
            "display_leaf_count": sum(not n.children for n in nodes),
            "display_maximum_depth": max(levels.values()),
            "collapsed_run_count": sum(n.collapsed for n in internal),
            "collapsed_run_lengths": [len(n.represented) for n in internal if n.collapsed],
            "multi_roi_fork_count": sum(all(c.size >= 2 for c in n.children)
                                        for n in internal if not n.collapsed),
            "terminal_pair_count": sum(n.size == 2 for n in internal),
            "roi_order": order,
            "height_rule": "retained subtree height" if collapse else "original log2 ROI count",
            "nodes": [{"indices": list(n.indices), "child_indices": [list(c.indices) for c in n.children],
                       "collapsed": n.collapsed, "display_depth": levels[id(n)],
                       "original_nodes": [{"indices": list(o.indices), "syn_bits_raw": o.syn_bits,
                                           "original_depth": o.depth} for o in n.represented]}
                      for n in internal]}


def add_colorbar(figure, axis, maximum, fontsize=7):
    cmap = LinearSegmentedColormap.from_list("original_local_syn", [
        _blend_with_white(SPLIT_COLOR, .12), _blend_with_white(SPLIT_COLOR, .74)])
    bar = figure.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0, maximum), cmap=cmap),
                         cax=axis, orientation="horizontal", ticks=[0, maximum], format="%.2f")
    bar.set_label(r"Local Syn / overall $\Xi$ (%)", fontsize=fontsize, labelpad=2)
    axis.xaxis.set_label_position("top")
    axis.tick_params(labelsize=fontsize - .3, length=2, pad=1)
    bar.outline.set_linewidth(.5)


def render_collapsed_tree(root, *, axis, labels, network_membership, network_names, seed, coupling_g,
                          network_colors=NETWORK_COLORS, color_max=3., show_heading=True,
                          show_colorbar=False, show_roi_labels=True, node_label_limit=12):
    audit = layout_audit(root)
    _, nodes, order, positions, levels = display_tree(root)
    tolerance = audit["syn_nonnegative_tolerance_bits"]
    membership = np.asarray(network_membership, dtype=int)
    if (len(membership) != root.size or len(labels) != root.size
            or np.any(membership < 0) or np.any(membership >= len(network_names))
            or len(network_colors) != len(network_names)):
        raise ValueError("Labels and network metadata must cover every ROI")
    if root.xi_bits <= 0 or not math.isfinite(root.xi_bits):
        raise ValueError("Syn shares require positive finite overall Xi")

    def share(node):
        raw = max(n.syn_bits for n in node.represented)
        # Original values passed the native-unit audit; only the declared
        # numerical-zero band may be represented as zero.
        if -tolerance <= raw < 0:
            raw = 0.
        return 100 * raw / root.xi_bits

    internal = [n for n in nodes if n.children]
    if not math.isfinite(color_max) or color_max <= 0 or max(map(share, internal)) > color_max + 1e-10:
        raise ValueError("Shared Syn guide must cover all original local values")
    axis.set(xlim=(-1, root.size), ylim=(-.26, 1.06))
    axis.axis("off")
    for node in internal:
        _, py = positions[id(node)]
        children = [positions[id(c)] for c in node.children]
        axis.plot([min(p[0] for p in children), max(p[0] for p in children)], [py, py],
                  color="#718A87" if node.collapsed else EDGE_COLOR,
                  linestyle=(0, (2.5, 2)) if node.collapsed else "-",
                  linewidth=.9 if node.collapsed else .75, zorder=1)
        for cx, cy in children:
            axis.plot([cx, cx], [cy, py], color=EDGE_COLOR, linewidth=.65, zorder=1)
    for node in internal:
        x, y = positions[id(node)]
        strength = share(node) / color_max
        axis.scatter(x, y, s=9 + 23 * strength, marker="D" if node.collapsed else "o",
                     facecolor=_blend_with_white(SPLIT_COLOR, .12 + .62 * strength),
                     edgecolor=_blend_with_white(SPLIT_COLOR, .55 + .45 * strength),
                     linewidth=.8, zorder=3)
    # Labels favor retained modules; compression never removes their geometry.
    ranked = sorted((n for n in internal if n.size >= 4),
                    key=lambda n: (n.size >= 20, all(c.size >= 2 for c in n.children), share(n)), reverse=True)
    chosen = []
    font = min(7.5, max(5.5, axis.get_position().width * axis.figure.get_figwidth() * 1.05))
    for node in ranked:
        x, y = positions[id(node)]
        if len(chosen) >= node_label_limit:
            break
        if any(abs(x - px) < 5.5 and abs(y - py) < .085 for px, py in chosen):
            continue
        chosen.append((x, y))
        value = f"peak {share(node):.2f}%" if node.collapsed else f"{share(node):.2f}%"
        axis.text(x + .8, y + .009, f"n={node.size}\n{value}", fontsize=font,
                  ha="left", va="bottom", linespacing=1.05, color=INK, zorder=5,
                  bbox={"boxstyle": "round,pad=.17", "facecolor": "white",
                        "edgecolor": _blend_with_white(SPLIT_COLOR, .55), "linewidth": .7})
    for position, index in enumerate(order):
        color = network_colors[membership[index]]
        axis.scatter(position, 0, s=15, facecolor="white", edgecolor=color, linewidth=1, zorder=4)
        axis.add_patch(Rectangle((position - .48, -.033), .96, .017,
                                 facecolor=color, edgecolor="white", linewidth=.28,
                                 clip_on=False, zorder=4))
        if show_roi_labels:
            axis.text(position, -.042, _short_roi_label(labels[index]), rotation=90,
                      ha="right", va="top", fontsize=3.7, color="#55616D", clip_on=False)
    if show_heading:
        axis.text(.5, 1.015, rf"$G={coupling_g:g}$, seed {seed}  |  $\Xi={root.xi_bits * math.log(2):.2f}$ nats",
                  transform=axis.transAxes, ha="center", va="bottom", fontsize=7, color=INK)
    if show_colorbar:
        add_colorbar(axis.figure, axis.inset_axes([.79, 1.115, .2, .018]), color_max, fontsize=5.8)
    audit["display_label_count"] = len(chosen)
    return audit


def render_figure(conditions, labels, membership, names, output, *, dpi=400):
    with mpl.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "savefig.facecolor": "white"}):
        single = len(conditions) == 1
        fig, axes = plt.subplots(1, len(conditions), figsize=(12 if single else 15.6, 7), squeeze=False)
        fig.subplots_adjust(left=.028, right=.985, top=.80, bottom=.10 if single else .13, wspace=.13)
        audits = []
        for axis, (tree, source) in zip(axes[0], conditions, strict=True):
            condense = math.isclose(source["coupling_g"], 1.3)
            if condense:
                audit = render_collapsed_tree(tree, axis=axis, labels=labels, network_membership=membership,
                                              network_names=names, seed=source["seed"], coupling_g=source["coupling_g"],
                                              show_roi_labels=single, node_label_limit=12 if single else 8)
            else:
                render_tree(tree, None, axis=axis, labels=labels, network_membership=membership,
                            network_names=names, network_colors=NETWORK_COLORS, seed=source["seed"],
                            coupling_g=source["coupling_g"], dpi=dpi, information_unit="nats",
                            node_value_mode="root_share", syn_color_max=3, show_colorbar=False,
                            show_roi_labels=False, show_network_strip_label=False, node_label_limit=6)
                audit = layout_audit(tree, collapse=False)
            audits.append({**audit, "source": source})
            if not single:
                axis.set_ylim(-.08, 1.10)
            axis.text(.5, 1.075, f"Multi-ROI forks: {audit['multi_roi_fork_count']}  |  "
                      f"Display depth: {audit['display_maximum_depth']}"
                      + (f" (from {audit['original_tree_metrics']['maximum_depth']})" if condense else ""),
                      transform=axis.transAxes, fontsize=8, color="#53646C", ha="center")
        handles = [Patch(facecolor=c, label=n) for c, n in zip(NETWORK_COLORS, names)]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.022, .995), ncol=4,
                   fontsize=9, frameon=False, handlelength=1, columnspacing=1.3)
        fig.legend(handles=[Line2D([0], [0], color="#718A87", linestyle="--", marker="D", markersize=4,
                                  label="Condensed singleton exits (color = peak local Syn)")],
                   loc="upper left", bbox_to_anchor=(.022, .914), frameon=False, fontsize=9)
        add_colorbar(fig, fig.add_axes([.76, .92, .21, .02]), 3., fontsize=8)
        fig.text(.028, .035, "All 100 ROI leaves retained. Dashed forks share one display level; peak labels refer to original binary steps.\n"
                 "Vertical spacing shows the displayed hierarchy, not information amount or elapsed time. Yeo-7 colors are post hoc.",
                 fontsize=8, color="#53646C", linespacing=1.55)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white",
                    metadata={"Collapsed SPT audits": json.dumps(audits),
                              "Description": "G=1.3 singleton runs share one displayed fork; other G retain full binary trees."})
        plt.close(fig)
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=PAIRED_SUMMARY)
    parser.add_argument("--original", action="store_true")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--g-values", nargs="+", type=float, default=[0., 1.3, 10.])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    if args.original:
        record = json.loads(ORIGINAL_SUMMARY.read_text())
        conditions = [(_node_from_record(record["tree"]), {"seed": record["seed"],
                       "coupling_g": record["coupling_g"], "cache": str(ORIGINAL_SUMMARY),
                       "sha256": hashlib.sha256(ORIGINAL_SUMMARY.read_bytes()).hexdigest()})]
        output = args.output or ROOT / "docs/reports/assets/dmf_tree_collapsed.png"
    else:
        conditions = load_comparison(args.summary, args.seed, args.g_values)
        output = args.output or ROOT / "docs/reports/assets/dmf_trees_G000_130_1000_seed04.png"
    with np.load(YEO_METADATA, allow_pickle=True) as a:
        membership, names, labels = a["network_membership"], a["network_names"].tolist(), a["region_labels"].tolist()
    audits = render_figure(conditions, labels, membership, names, output, dpi=args.dpi)
    print(json.dumps([{k: v for k, v in a.items() if k not in {"nodes", "roi_order"}} for a in audits], indent=2))


if __name__ == "__main__":
    main()
