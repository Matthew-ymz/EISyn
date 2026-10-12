#!/usr/bin/env python3
"""Condense consecutive singleton exits while preserving every other SPT branch.

Dashed display groups contain singleton ROIs peeled consecutively from one
largest-child path. They are visual groups, not newly estimated communities or
coalition Syn. Every original internal node is represented exactly once: by a
retained fork, a terminal pair, or one step in a condensed path. Teal encodes
the maximum original local Syn among those represented steps, not their sum.
Widths count ROIs; area and vertical spacing do not encode information.
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
from matplotlib.patches import Patch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_dmf_schaefer100_xi_hierarchy_tree import INK, SPLIT_COLOR, _blend_with_white, _tree_metrics
from scripts.compare_runge_slp_pc60_xi_horizons import _node_from_record
from scripts.plot_dmf_schaefer100_tree_examples import NETWORK_COLORS, audit_tree, condition_config
from scripts.spt import flatten_nodes

PAIRED_SUMMARY = ROOT / "results/dmf_schaefer100/xi_hierarchy_tree/examples/wide_G1000/summary.json"
ORIGINAL_SUMMARY = ROOT / "results/dmf_schaefer100/xi_hierarchy_tree/summary.json"
YEO_METADATA = ROOT / "results/dmf_schaefer100/full/critical_yeo7.npz"


@dataclass(frozen=True)
class DisplayModule:
    indices: tuple[int, ...]
    kind: str
    represented: tuple = ()
    children: tuple["DisplayModule", ...] = ()

    @property
    def size(self):
        return len(self.indices)


def grouped_hierarchy(root):
    """Contract only maximal runs of >=2 consecutive one-ROI splits."""
    def build(node):
        if not node.children:
            return DisplayModule(node.indices, "roi")
        if node.size == 2:
            return DisplayModule(node.indices, "pair", (node,))
        current, escaped, steps = node, [], []
        while current.children and current.size > 2:
            large, small = sorted(current.children, key=lambda c: (-c.size, c.indices))
            if small.size != 1:
                break
            escaped.append(int(small.indices[0]))
            steps.append(current)
            current = large
        if escaped:
            pooled = DisplayModule(tuple(escaped), "peel_group" if len(escaped) >= 2 else "roi")
            return DisplayModule(node.indices, "run" if len(escaped) >= 2 else "split",
                                 tuple(steps), (build(current), pooled))
        return DisplayModule(node.indices, "split", (node,),
                             tuple(build(c) for c in sorted(node.children, key=lambda c: (-c.size, c.indices))))

    display = build(root)
    all_modules, order, intervals, levels = [], [], {}, {}

    def visit(module, depth):
        all_modules.append(module)
        levels[id(module)] = depth
        start = len(order)
        if module.children:
            for child in module.children:
                visit(child, depth + 1)
            child_sets = [set(child.indices) for child in module.children]
            if child_sets[0] & child_sets[1] or set.union(*child_sets) != set(module.indices):
                raise ValueError("Displayed children do not partition their original parent")
        else:
            order.extend(module.indices)
        intervals[id(module)] = (start, len(order))
        if set(order[start:]) != set(module.indices) or len(order) - start != module.size:
            raise ValueError("Displayed module interval lost or duplicated an ROI")

    visit(display, 0)
    internal = [n for n in flatten_nodes(root) if n.children]
    represented = [n for module in all_modules for n in module.represented]
    if len(represented) != len(internal) or {id(n) for n in represented} != {id(n) for n in internal}:
        raise ValueError("Each original local Syn must be represented exactly once")
    if sorted(order) != sorted(root.indices) or len(order) != root.size:
        raise ValueError("Every ROI must remain in the grouped display")
    return display, all_modules, order, intervals, levels


def peak_share(module, root, tolerance):
    value = max((n.syn_bits for n in module.represented), default=0.)
    # The raw tree has already passed the native-unit audit. No significant
    # negative is projected; only the declared numerical-zero band is used.
    if -tolerance <= value < 0:
        value = 0.
    return 100 * value / root.xi_bits


def render_peel_hierarchy(root, *, axis, network_membership, network_names, seed, coupling_g,
                          network_colors=NETWORK_COLORS, color_max=3., common_depth=None,
                          show_heading=True, show_colorbar=False):
    audit = audit_tree(root, root.xi_bits)
    if root.xi_bits <= 0:
        raise ValueError("Local Syn percentages require positive root Xi")
    display, modules, order, intervals, levels = grouped_hierarchy(root)
    membership = np.asarray(network_membership, dtype=int)
    if (len(membership) != root.size or np.any(membership < 0)
            or np.any(membership >= len(network_names)) or len(network_colors) != len(network_names)):
        raise ValueError("Yeo metadata must describe all ROI leaves")
    tolerance = audit["syn_nonnegative_tolerance_bits"]
    actual_max = 100 * max(n.syn_bits for n in flatten_nodes(root) if n.children) / root.xi_bits
    if not math.isfinite(color_max) or color_max <= 0 or color_max < actual_max - 1e-10:
        raise ValueError("Shared color scale must cover all original local Syn")
    maximum_depth = max(levels.values())
    depth = maximum_depth if common_depth is None else int(common_depth)
    if depth < maximum_depth:
        raise ValueError("Common display depth cannot truncate a branch")
    bottom = depth + 1.55
    axis.set(xlim=(-.25, root.size + .25), ylim=(bottom + .2, -.1))
    axis.axis("off")
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    width_inches = axis.get_window_extent().width / axis.figure.dpi
    font = min(8.6, max(6.5, width_inches * 1.5))
    cmap = LinearSegmentedColormap.from_list("peel_peak_syn", [
        _blend_with_white(SPLIT_COLOR, .12), _blend_with_white(SPLIT_COLOR, .74)])
    norm = Normalize(0, color_max)
    texts = []
    for module in sorted(modules, key=lambda m: levels[id(m)]):
        start, stop = intervals[id(module)]
        level = levels[id(module)]
        x, width = start + .07, stop - start - .14
        y = float(level)
        if module.children:
            axis.add_patch(Rectangle((x, y), width, bottom - y, facecolor="#F8FBFA",
                                     edgecolor="#ABBDB8", linewidth=.65, zorder=level + 1))
            header = Rectangle((x, y), width, .77,
                               facecolor=cmap(norm(peak_share(module, root, tolerance))),
                               edgecolor="none", zorder=level + 2)
            axis.add_patch(header)
            value = peak_share(module, root, tolerance)
            label = f"{module.size} ROI  ·  {value:.2f}%" if width >= 24 else str(module.size)
            text = axis.text(x + width / 2, y + .385, label,
                             ha="center", va="center", color=INK, fontsize=font,
                             zorder=level + 3)
            texts.append((text, header))
        elif module.kind == "peel_group":
            box = Rectangle((x, y), width, bottom - .50 - y,
                            facecolor="#F1F3F4", edgecolor="#8A969C",
                            linewidth=.75, linestyle=(0, (3, 2)), zorder=level + 1)
            axis.add_patch(box)
            label = f"{module.size} ROI\nsequential exits" if width >= 25 else str(module.size)
            text = axis.text(x + width / 2, y + (bottom - .50 - y) / 2,
                             label, ha="center", va="center", color="#637178",
                             fontsize=font, linespacing=1.3, zorder=level + 2)
            texts.append((text, box))
        else:
            box = Rectangle((x, y), width, bottom - .50 - y,
                            facecolor=cmap(norm(peak_share(module, root, tolerance)))
                            if module.kind == "pair" else "#F8FBFA",
                            edgecolor="#A6B7B1", linewidth=.55, zorder=level + 1)
            axis.add_patch(box)
            text = axis.text(x + width / 2, y + .38, str(module.size), ha="center", va="center",
                             fontsize=font, color=INK, zorder=level + 2)
            texts.append((text, box))
    for position, index in enumerate(order):
        axis.add_patch(Rectangle((position + .10, bottom - .39), .80, .30,
                                 facecolor=network_colors[membership[index]],
                                 edgecolor="none", zorder=depth + 5))
    # Keep even two-ROI and single-ROI labels visible without spilling into a
    # neighboring module. Fit text to its own frame; geometries are never dropped.
    label_count = 0
    for text, patch in texts:
        box = patch.get_window_extent(renderer)
        for size in (font, 6.5, 6., 5.5, 5., 4.5):
            text.set_fontsize(min(font, size))
            bounds = text.get_window_extent(renderer)
            if bounds.width < box.width - 1 and bounds.height < box.height - 1:
                label_count += 1
                break
        else:
            text.set_text("")
        text.set_clip_path(patch)
    if show_heading:
        axis.text(0, 1.065, rf"$G={coupling_g:g}$  |  $\Xi={root.xi_bits * math.log(2):.2f}$ nats",
                  transform=axis.transAxes, fontsize=11, color=INK, va="bottom")
    if show_colorbar:
        add_colorbar(axis.figure, axis.inset_axes([.78, 1.145, .22, .022]), color_max, fontsize=6.)
    groups = [m for m in modules if m.kind == "peel_group"]
    represented_forks = [m for m in modules if m.kind == "split" and len(m.represented) == 1
                         and all(c.size >= 2 for c in m.represented[0].children)]
    records = [{"kind": m.kind, "indices": list(m.indices),
                "represented_original_indices": [list(n.indices) for n in m.represented],
                "represented_local_syn_bits": [n.syn_bits for n in m.represented],
                "peak_local_syn_share_percent": peak_share(m, root, tolerance),
                "display_level": levels[id(m)]} for m in modules]
    return {**audit, "seed": seed, "coupling_g": coupling_g,
            "overall_xi_nats": root.xi_bits * math.log(2), "tree_metrics": _tree_metrics(root),
            "original_internal_nodes_represented": sum(len(m.represented) for m in modules),
            "display_module_count": len(modules), "visible_module_label_count": label_count,
            "display_maximum_depth": maximum_depth, "multi_roi_fork_count": len(represented_forks),
            "terminal_pair_count": sum(m.kind == "pair" for m in modules),
            "peel_group_count": len(groups), "peel_group_sizes": [m.size for m in groups],
            "roi_order": order, "modules": records,
            "color_encoding": "maximum original local binary Syn among represented steps / root Xi (%)",
            "width_encoding": "ROI count; areas and heights are not information mass",
            "group_rule": "maximal runs of >=2 consecutive one-ROI exits; stop at any multi-ROI fork or terminal pair"}


def add_colorbar(figure, color_axis, maximum, *, fontsize=9.):
    cmap = LinearSegmentedColormap.from_list("peel_peak_syn_guide", [
        _blend_with_white(SPLIT_COLOR, .12), _blend_with_white(SPLIT_COLOR, .74)])
    bar = figure.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0, maximum), cmap=cmap),
                         cax=color_axis, orientation="horizontal", ticks=np.linspace(0, maximum, 4))
    bar.set_label(r"Peak local Syn / overall $\Xi$ (%)", fontsize=fontsize, labelpad=4)
    color_axis.xaxis.set_label_position("top")
    color_axis.tick_params(labelsize=fontsize * .89, length=2)
    bar.outline.set_linewidth(.4)


def load_comparison(summary_path, seed, couplings):
    summary = json.loads(summary_path.read_text())
    results = []
    for g in couplings:
        rows = [r for r in summary["rows"] if r["seed"] == seed and np.isclose(r["coupling_g"], g)]
        if len(rows) != 1:
            raise ValueError(f"Expected one cached condition for seed={seed}, G={g}")
        path = Path(rows[0]["cache"])
        record = json.loads(path.read_text())
        if record["seed"] != seed or not np.isclose(record["coupling_g"], g):
            raise ValueError("Condition cache has wrong seed or G")
        if condition_config(record["config"], seed=seed, coupling_g=g) != condition_config(
                summary["config"], seed=seed, coupling_g=g):
            raise ValueError("Condition provenance differs from comparison protocol")
        tree = _node_from_record(record["tree"])
        audit_tree(tree, rows[0]["overall_xi_bits"])
        results.append((tree, {"cache": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "seed": seed, "coupling_g": g}))
    paired_inputs = summary["config"]["paired_covariance_inputs"]
    names = {f"seed{seed:02d}_G{g:.2f}.npz" for g in couplings}
    inputs = [p for p in paired_inputs if Path(p["path"]).name in names]
    if (len(inputs) != len(couplings)
            or len({p["paired_input_sha256"] for p in inputs}) != 1
            or any(p["simulation_config"] != inputs[0]["simulation_config"] for p in inputs)):
        raise ValueError("Three G conditions must have paired inputs and the same simulation protocol")
    for item in inputs:
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Paired covariance cache changed since tree construction")
    return results


def render_figure(conditions, membership, names, output, *, dpi=400):
    trees = [tree for tree, record in conditions]
    depth = max(max(grouped_hierarchy(tree)[4].values()) for tree in trees)
    peak = max(100 * n.syn_bits / tree.xi_bits for tree in trees
               for n in flatten_nodes(tree) if n.children)
    color_max = max(3., math.ceil(peak * 2) / 2)
    with mpl.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "savefig.facecolor": "white"}):
        figure_width = 12. if len(trees) == 1 else 5.1 * len(trees)
        fig, axes = plt.subplots(1, len(trees), figsize=(figure_width, 6.9), squeeze=False)
        fig.subplots_adjust(left=.025, right=.985, top=.78, bottom=.19, wspace=.11)
        audits = []
        for axis, (tree, provenance) in zip(axes[0], conditions, strict=True):
            audit = render_peel_hierarchy(tree, axis=axis, network_membership=membership,
                                         network_names=names, seed=provenance["seed"],
                                         coupling_g=provenance["coupling_g"], color_max=color_max,
                                         common_depth=depth)
            audits.append({**audit, "source": provenance})
            axis.text(0, -.035,
                      f"Multi-ROI forks: {audit['multi_roi_fork_count']}  ·  "
                      f"Pairs: {audit['terminal_pair_count']}  ·  Exit groups: {audit['peel_group_count']}",
                      transform=axis.transAxes, fontsize=8.2, va="top", color=INK)
        fig.legend(handles=[Patch(facecolor=c, label=n) for c, n in zip(NETWORK_COLORS, names)],
                   loc="upper left", bbox_to_anchor=(.025, .972), ncol=4, fontsize=9,
                   frameon=False, handlelength=1.1, columnspacing=1.3)
        fig.text(.025, .86, f"Seed {conditions[0][1]['seed']}  ·  consecutive singleton exits grouped",
                 color="#53646C", fontsize=10)
        color_axis = fig.add_axes([.72, .909, .25, .024])
        add_colorbar(fig, color_axis, color_max)
        fig.text(.025, .098,
                 "Dashed boxes = consecutive singleton exit groups; solid frames = retained branching modules.\n"
                 "Widths and numbers = ROI counts; bottom tiles = Yeo-7 membership (post hoc). All 100 ROI retained.\n"
                 "Teal = peak original local Syn in each condensed path; groups have no newly estimated Syn.\n"
                 "Each panel uses its own hierarchy order; vertical position is display depth.",
                 fontsize=8.7, va="center", color="#53646C", linespacing=1.55)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white", metadata={
            "Grouped SPT audits": json.dumps(audits),
            "Description": "Consecutive singleton exits pooled for display; all original binary Syn atoms represented once."})
        plt.close(fig)
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=PAIRED_SUMMARY)
    parser.add_argument("--original", action="store_true", help="Use the original single-condition Figure Q2b cache")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--g-values", type=float, nargs="+", default=[0., 1.3, 10.])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    if args.original:
        payload = json.loads(ORIGINAL_SUMMARY.read_text())
        conditions = [(_node_from_record(payload["tree"]), {"seed": payload["seed"],
                       "coupling_g": payload["coupling_g"], "cache": str(ORIGINAL_SUMMARY),
                       "sha256": hashlib.sha256(ORIGINAL_SUMMARY.read_bytes()).hexdigest()})]
        output = args.output or ROOT / "docs/reports/assets/dmf_tree_peel_groups.png"
    else:
        conditions = load_comparison(args.summary, args.seed, args.g_values)
        output = args.output or ROOT / "docs/reports/assets/dmf_peel_groups_G000_130_1000_seed04.png"
    with np.load(YEO_METADATA, allow_pickle=True) as a:
        membership, names = a["network_membership"], a["network_names"].tolist()
    audits = render_figure(conditions, membership, names, output, dpi=args.dpi)
    print(json.dumps([{k: v for k, v in a.items() if k not in {"modules", "roi_order"}}
                      for a in audits], indent=2))


if __name__ == "__main__":
    main()
