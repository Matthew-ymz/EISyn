#!/usr/bin/env python3
"""Show cached DMF SPT subtrees as nested modules, without binary connectors.

Widths encode ROI counts, not information mass. Only header color encodes each
retained node's original local Syn; collapsing ancestors never sums their Syn
into a displayed node. Yeo labels are post hoc and do not select modules.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_dmf_schaefer100_xi_hierarchy_tree import INK, SPLIT_COLOR, _blend_with_white
from scripts.compare_runge_slp_pc60_xi_horizons import _node_from_record
from scripts.plot_dmf_schaefer100_tree_examples import NETWORK_COLORS, audit_tree
from scripts.spt import flatten_nodes

MIN_MODULE_SIZE = 4
SIZE_RATIO = 0.72


def module_layout(root):
    """Retain scale landmarks and both sides of substantial branch points.

    On each largest-child path retain the highest-Syn node below successive
    size limits (0.85 * root size initially, then 0.72 * last landmark size).
    Retain forks with both children >= 4 ROI and their children, and apply the
    same rule inside side subtrees of >= 4 ROI. These are display choices only.
    """
    retained = {root.indices: root}

    def select(branch):
        spine = []
        node = branch
        while node.children:
            spine.append(node)
            children = sorted(node.children, key=lambda c: (-c.size, tuple(c.indices)))
            if all(child.size >= MIN_MODULE_SIZE for child in children):
                for item in (node, *children):
                    retained[item.indices] = item
            for side in children[1:]:
                if side.size >= MIN_MODULE_SIZE:
                    retained[side.indices] = side
                    select(side)
            node = children[0]
        limit = branch.size * 0.85
        while True:
            candidates = [n for n in spine if MIN_MODULE_SIZE <= n.size <= limit]
            if not candidates:
                break
            landmark = max(candidates, key=lambda n: (n.syn_bits, -n.size))
            retained[landmark.indices] = landmark
            limit = landmark.size * SIZE_RATIO

    select(root)
    order, intervals, levels = [], {}, {}

    def visit(node, level):
        start = len(order)
        shown = node.indices in retained
        if shown:
            levels[node.indices] = level
        if node.children:
            for child in sorted(node.children, key=lambda c: (-c.size, tuple(c.indices))):
                visit(child, level + int(shown))
        else:
            order.extend(node.indices)
        intervals[node.indices] = (start, len(order))

    visit(root, 0)
    if len(order) != root.size or set(order) != set(root.indices):
        raise ValueError("Module display must retain each ROI exactly once")
    for key, node in retained.items():
        start, stop = intervals[key]
        if stop - start != node.size or set(order[start:stop]) != set(node.indices):
            raise ValueError("Module interval differs from its original SPT subtree")
    return list(retained.values()), order, intervals, levels


def render_module_hierarchy(
    root, *, axis, network_membership, network_names, seed, coupling_g,
    network_colors=NETWORK_COLORS, node_value_mode="root_share", information_unit="nats",
    show_colorbar=True,
):
    """Draw nested frames, local-Syn headers and a complete ROI membership strip."""
    audit = audit_tree(root, root.xi_bits)
    if node_value_mode not in {"absolute", "root_share"}:
        raise ValueError("Module values must be absolute or root_share")
    if information_unit not in {"bits", "nats"}:
        raise ValueError("Module information unit must be bits or nats")
    membership = np.asarray(network_membership, dtype=int)
    if (len(membership) != root.size or np.any(membership < 0)
            or np.any(membership >= len(network_names))
            or len(network_colors) != len(network_names)):
        raise ValueError("Network metadata must match all ROI leaves")
    unit_scale = math.log(2) if information_unit == "nats" else 1.0
    percentage = node_value_mode == "root_share"
    if percentage and root.xi_bits <= 0:
        raise ValueError("Root shares require positive overall Xi")
    scale = 100 / root.xi_bits if percentage else unit_scale
    nodes, order, intervals, levels = module_layout(root)
    tolerance = audit["syn_nonnegative_tolerance_bits"]
    # audit_tree has already rejected significant negatives; this declared rule
    # absorbs only values in [-tolerance, 0), and its count stays in the audit.
    values = {n.indices: (0.0 if -tolerance <= n.syn_bits < 0 else n.syn_bits) * scale
              for n in flatten_nodes(root) if n.children}
    maximum = max(values.values(), default=0.0)
    norm = Normalize(0, maximum if maximum > 0 else 1)
    cmap = LinearSegmentedColormap.from_list(
        "module_local_syn", [_blend_with_white(SPLIT_COLOR, .12),
                             _blend_with_white(SPLIT_COLOR, .74)])
    last = max(levels.values())
    bottom = last + 1.40
    axis.set(xlim=(-.4, root.size + .4), ylim=(bottom + 1.1, -.1))
    axis.axis("off")
    axis.figure.canvas.draw()
    width_inches = axis.get_window_extent().width / axis.figure.dpi
    font = min(8.3, max(5.3, width_inches * .78))
    for node in sorted(nodes, key=lambda n: (levels[n.indices], -n.size)):
        start, stop = intervals[node.indices]
        level = levels[node.indices]
        x, width = start + .12, stop - start - .24
        # Shared bottom makes containment explicit; outline and header carry
        # hierarchy. Background area has no information-mass interpretation.
        axis.add_patch(Rectangle((x, level), width, bottom - level,
                                 facecolor="#F7FAF9", edgecolor="#ADBEB9",
                                 linewidth=.65, zorder=level + 1))
        header = Rectangle((x, level), width, .82,
                           facecolor=cmap(norm(values[node.indices])),
                           edgecolor="none", zorder=level + 2)
        axis.add_patch(header)
        suffix = "%" if percentage else ""
        value = values[node.indices]
        if width / root.size * width_inches >= 1.12:
            label = f"{node.size} ROI   ·   {value:.2f}{suffix}"
        else:
            label = f"{node.size} ROI\n{value:.2f}{suffix}"
        text = axis.text(x + width / 2, level + .41, label, ha="center", va="center",
                         fontsize=font, color=INK, linespacing=1.05,
                         zorder=level + 3)
        text.set_clip_path(header)

    # Every member is retained, including all collapsed singleton side branches.
    for position, index in enumerate(order):
        axis.add_patch(Rectangle((position + .10, bottom - .48), .80, .30,
                                 facecolor=network_colors[membership[index]],
                                 edgecolor="none", zorder=last + 5))
    axis.text(0, bottom + .16, "Yeo-7 membership (post hoc)  ·  one tile per ROI",
              fontsize=font * .90, color="#596A70", va="top")
    axis.text(0, bottom + .68,
              "Width = ROI count   ·   Nested frames = retained SPT subtrees   ·   Intermediate splits collapsed",
              fontsize=font * .86, color="#596A70", va="top")
    axis.text(0, 1.035,
              rf"$G={coupling_g:g}$, seed {seed}   |   $\Xi={unit_scale * root.xi_bits:.2f}$ {information_unit}",
              transform=axis.transAxes, fontsize=font * 1.10, color=INK, va="bottom")
    if show_colorbar:
        color_axis = axis.inset_axes([.78, 1.145, .22, .025])
        colorbar = axis.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                                       cax=color_axis, orientation="horizontal",
                                       ticks=[0, maximum], format="%.2f")
        colorbar.set_label(r"Local Syn / overall $\Xi$ (%)" if percentage
                           else f"Local Syn ({information_unit})",
                           fontsize=font * .85, labelpad=3)
        color_axis.xaxis.set_label_position("top")
        color_axis.tick_params(labelsize=font * .85, length=2, pad=2)
        colorbar.outline.set_linewidth(.4)
    return {**audit, "retained_module_count": len(nodes),
            "retained_module_sizes": [n.size for n in nodes],
            "display_levels": last + 1, "roi_order": order,
            "retained_module_indices": [list(n.indices) for n in nodes],
            "display_rule": {"minimum_module_size": MIN_MODULE_SIZE,
                             "initial_size_ratio": .85, "successive_size_ratio": SIZE_RATIO},
            "width_encoding": "ROI count; not Syn or Xi mass",
            "value_encoding": "original local binary-split Syn; no aggregation across collapsed nodes"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-summary", type=Path,
                        default=ROOT / "results/dmf_schaefer100/xi_hierarchy_tree/summary.json")
    parser.add_argument("--yeo7", type=Path, default=ROOT / "results/dmf_schaefer100/full/critical_yeo7.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/assets/dmf_tree_modules.png")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    payload = json.loads(args.tree_summary.read_text())
    tree = _node_from_record(payload["tree"])
    with np.load(args.yeo7, allow_pickle=True) as archive:
        membership = archive["network_membership"]
        names = [str(n) for n in archive["network_names"]]
        si = np.flatnonzero(archive["seeds"] == payload["seed"])
        gi = np.flatnonzero(np.isclose(archive["G"], payload["coupling_g"]))
        if len(si) != 1 or len(gi) != 1:
            raise ValueError("Cached seed/G condition is missing or ambiguous")
        audit_tree(tree, float(archive["fine_phi"][si[0], gi[0]]))
    with mpl.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "savefig.facecolor": "white"}):
        fig, axis = plt.subplots(figsize=(12, 4.6))
        fig.subplots_adjust(left=.025, right=.99, top=.79, bottom=.03)
        audit = render_module_hierarchy(tree, axis=axis, network_membership=membership,
                                        network_names=names, seed=payload["seed"],
                                        coupling_g=payload["coupling_g"])
        axis.legend(handles=[Patch(facecolor=c, label=n) for c, n in zip(NETWORK_COLORS, names)],
                    loc="lower left", bbox_to_anchor=(0, 1.12), ncol=4,
                    fontsize=8, frameon=False, handlelength=1.1, columnspacing=1.2)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white",
                    metadata={"Source": str(args.tree_summary), "Module display audit": json.dumps(audit)})
        plt.close(fig)
    print(json.dumps({k: v for k, v in audit.items()
                      if k not in {"roi_order", "retained_module_indices"}}, indent=2))


if __name__ == "__main__":
    main()
