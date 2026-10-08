#!/usr/bin/env python3
"""Describe the paired fine-G SPT scan and render its scientific figures.

Uses exact ROI-set identities, raw Syn, and fixed reference splits. No smoothing,
background subtraction, topology regularization, or interpolation between trees.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analyze_dmf_schaefer100_xi_hierarchy_tree import (
    flatten_nodes, render_tree, _blend_with_white, SPLIT_COLOR, INK, _leaf_order,
)
from scripts.compare_runge_slp_pc60_xi_horizons import _node_from_record
from scripts.plot_dmf_schaefer100_tree_examples import (
    CachedScalarLogdetXiOracle, NETWORK_COLORS, audit_tree,
)
from scripts.run_dmf_spt_G_resolution import OUTPUT, TOL, write_json
from scripts.spt import audit_syn_value

FIGURES = ROOT / "fig/dmf_schaefer100/spt_G_resolution_seed04"
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
         "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.linewidth": .7, "savefig.facecolor": "white"}
CURVE_COLORS = ("#267A70", "#3A65A0", "#C16A30", "#9E4E82", "#71822F", "#66889C")


def nodes_by_set(tree):
    return {n.indices: n for n in flatten_nodes(tree) if n.children}


def split(node):
    return tuple(child.indices for child in node.children)


def unlabelled_shape(node):
    """Only branching shape; exclude ROI IDs, left/right order, and Syn weights."""
    if not node.children:
        return "L"
    return "("+"".join(sorted(unlabelled_shape(child) for child in node.children))+")"


def runs(mask, gs):
    result = []
    start = None
    for i, present in enumerate([*mask, False]):
        if present and start is None:
            start = i
        if not present and start is not None:
            end = i - 1
            result.append({"start_index": start, "end_index": end,
                           "first_G": float(gs[start]), "last_G": float(gs[end]),
                           "points": i-start, "left_censored": start == 0,
                           "right_censored": end == len(gs)-1})
            start = None
    return result


def load(input_dir, preview=False):
    summary_path = input_dir / "summary.json"
    if summary_path.exists():
        meta = json.loads(summary_path.read_text())
        if not preview and (meta["status"] != "complete" or len(meta["records"]) != 171):
            raise ValueError("The approved 171-condition scan is incomplete")
        records = [json.loads((ROOT / r["tree_path"]).read_text()) for r in meta["records"]]
    elif preview:
        with np.load(ROOT / "results/dmf_schaefer100/full/critical_yeo7.npz", allow_pickle=True) as a:
            meta = {"labels": a["region_labels"].tolist(), "network_names": a["network_names"].tolist(),
                    "network_membership": a["network_membership"].tolist()}
        records = [json.loads(p.read_text()) for p in sorted((input_dir / "trees").glob("*.json"))]
    else:
        raise ValueError("Missing final summary.json; wait for the scientific scan")
    if not records:
        raise ValueError("No trees to plot")
    records.sort(key=lambda r: r["G"])
    if not preview and (any(r["seed"] != 4 for r in records)
                        or not np.array_equal([r["G"] for r in records], np.arange(171)/100.)):
        raise ValueError("Expected exactly seed 4 at G=0..1.70, step 0.01")
    configs = [r["config"] for r in records]
    if any(c != configs[0] for c in configs):
        raise ValueError("Scientific configurations differ between conditions")
    trees = [_node_from_record(r["tree"]) for r in records]
    for record, tree in zip(records, trees, strict=True):
        audit_tree(tree, tree.xi_bits)
    return meta, records, trees


def choose_tracked(maps, gs):
    """Exploratory, deterministic selection: 12 strongest persistent small clades.

    Eligibility is at least five consecutive sampled G points and 2..20 ROI.
    Rank by peak selected local Syn; this is not a significance criterion.
    """
    keys = set().union(*(set(m) for m in maps))
    candidates = []
    for key in keys:
        if not 2 <= len(key) <= 20:
            continue
        mask = np.asarray([key in m for m in maps], dtype=bool)
        intervals = runs(mask, gs)
        longest = max(r["points"] for r in intervals)
        if longest < min(5, len(gs)):
            continue
        present = [(i, m[key]) for i, m in enumerate(maps) if key in m]
        index, ref = max(present, key=lambda x: (x[1].syn_bits, -x[0]))
        candidates.append({"members": list(key), "size": len(key), "mask": mask.tolist(),
                           "intervals": intervals, "longest_run_points": longest,
                           "selected_points": int(mask.sum()), "reference_index": index,
                           "reference_G": float(gs[index]), "reference_split": [list(x) for x in split(ref)],
                           "maximum_selected_syn_nats": ref.syn_bits * math.log(2)})
    chosen = sorted(candidates, key=lambda r: (-r["maximum_selected_syn_nats"], tuple(r["members"])))[:12]
    for rank, row in enumerate(chosen, 1):
        row["strength_rank"] = rank
    chosen.sort(key=lambda r: (r["intervals"][0]["first_G"], r["size"], tuple(r["members"])))
    for index, row in enumerate(chosen, 1):
        row["id"] = f"C{index:02d}"
    return chosen


def original_reference():
    original = json.loads((ROOT / "results/dmf_schaefer100/xi_hierarchy_tree/summary.json").read_text())
    node = _node_from_record(original["tree"])
    spine = []
    while node.children:
        spine.append(node)
        node = max(node.children, key=lambda x: (x.size, -min(x.indices)))
    desired = (6, 9, 15, 24, 40, 59, 85)
    lookup = {n.size: n for n in spine}
    return [{"id": f"O{size:02d}", "size": size, "members": list(lookup[size].indices), "reference_G": 1.3,
             "reference_split": [list(x) for x in split(lookup[size])],
             "original_selected_syn_nats": lookup[size].syn_bits * math.log(2)}
            for size in desired if size in lookup]


def analyze(meta, records, trees, input_dir, preview=False):
    gs = np.asarray([r["G"] for r in records])
    maps = [nodes_by_set(t) for t in trees]
    shapes = [unlabelled_shape(t) for t in trees]
    clades = [set(m)-{t.indices} for m, t in zip(maps, trees, strict=True)]
    small = [{k for k in c if len(k) <= 20} for c in clades]
    tracked = choose_tracked(maps, gs)
    original = original_reference()
    for row in original:
        key = tuple(row["members"])
        mask = np.asarray([key in m for m in maps])
        row.update(mask=mask.tolist(), selected_points=int(mask.sum()), intervals=runs(mask, gs))
    events = []
    for i in range(1, len(gs)):
        differing = [maps[i-1][k] for k in set(maps[i-1]) & set(maps[i])
                     if split(maps[i-1][k]) != split(maps[i][k])]
        first = min(differing, key=lambda n: (n.depth, n.indices)) if differing else None
        common_small = len(small[i-1] & small[i])
        row = {"previous_G": float(gs[i-1]), "G": float(gs[i]),
               "same_unlabelled_shape": shapes[i-1] == shapes[i],
               "shared_nonroot_clades": len(clades[i-1] & clades[i]),
               "new_nonroot_clades": len(clades[i] - clades[i-1]),
               "lost_nonroot_clades": len(clades[i-1] - clades[i]),
               "small_clades_before": len(small[i-1]), "small_clades_after": len(small[i]),
               "shared_small_clades": common_small,
               "new_small_clades": len(small[i]-small[i-1]),
               "lost_small_clades": len(small[i-1]-small[i])}
        if first:
            other = maps[i][first.indices]
            row["first_changed_parent"] = list(first.indices)
            row["first_changed_parent_size"] = first.size
            row["previous_split"] = [list(x) for x in split(first)]
            row["new_split"] = [list(x) for x in split(other)]
            for label, position in (("previous", i-1), ("new", i)):
                gap = next(r for r in records[position]["candidate_gaps"] if tuple(r["indices"]) == first.indices)
                row[label+"_candidate_gap_nats"] = None if gap["best_second_gap_bits"] is None else gap["best_second_gap_bits"]*math.log(2)
        events.append(row)
    n_negative = 0
    minimum = math.inf
    # Freeze each tracked coalition AND its reference child partition.
    # Missing from the independently selected tree remains a valid evaluated coalition.
    for row in tracked + original:
        row["fixed_syn_nats"] = []
        row["fixed_cross_roi_xi_nats"] = []
    for i, record in enumerate(records):
        path = ROOT / record["covariance"]["path"]
        with np.load(path) as a:
            oracle = CachedScalarLogdetXiOracle(a["conditional_covariance"], [(j, j+100) for j in range(100)])
        for row in tracked + original:
            members = tuple(row["members"])
            left, right = row["reference_split"]
            syn = float(oracle.xi(members)-oracle.xi(left)-oracle.xi(right))
            minimum = min(minimum, syn)
            n_negative += int(audit_syn_value(syn, tolerance=TOL, context=f"G={gs[i]} {row['id']} frozen split"))
            value = float(oracle.xi(members)-sum(oracle.xi((j,)) for j in members))
            n_negative += int(audit_syn_value(value, tolerance=TOL, context=f"G={gs[i]} {row['id']} frozen coalition Xi"))
            row["fixed_syn_nats"].append(syn*math.log(2))
            row["fixed_cross_roi_xi_nats"].append(value*math.log(2))
        # Compare the two selected partitions on the SAME parent at both Gs.
        # This does not increase candidate budget or construct any replacement tree.
        for event_index in (i-1, i):
            if not 0 <= event_index < len(events):
                continue
            event = events[event_index]
            if "first_changed_parent" not in event:
                continue
            side = "at_previous_G" if event_index == i else "at_new_G"
            parent = event["first_changed_parent"]
            for name in ("previous_split", "new_split"):
                left, right = event[name]
                value = float(oracle.xi(parent)-oracle.xi(left)-oracle.xi(right))
                minimum = min(minimum, value)
                n_negative += int(audit_syn_value(value, tolerance=TOL, context=f"G={gs[i]} cross-score {name}"))
                event.setdefault("cross_scores", {}).setdefault(side, {})[name+"_syn_nats"] = value*math.log(2)
    tolerance_nats = TOL*math.log(2)
    for event in events:
        if "cross_scores" not in event:
            event["comparison_kind"] = "same topology"
            continue
        before = event["cross_scores"]["at_previous_G"]
        after = event["cross_scores"]["at_new_G"]
        first_margin = before["new_split_syn_nats"]-before["previous_split_syn_nats"]
        second_margin = after["previous_split_syn_nats"]-after["new_split_syn_nats"]
        event["previous_winner_margin_nats"] = first_margin
        event["new_winner_margin_nats"] = second_margin
        event["comparison_kind"] = ("selected split scores cross" if min(first_margin, second_margin) >= -tolerance_nats
                                    else "candidate-set change matters")
        if event["comparison_kind"] == "candidate-set change matters" and event["first_changed_parent_size"] <= 8:
            raise RuntimeError("An exact-search node selected a worse admissible split; audit before interpreting")
    stats = {
        "condition_count": len(gs), "adjacent_pair_count": len(events),
        "identical_full_tree_pairs": sum(e["new_nonroot_clades"] == 0 for e in events),
        "identical_unlabelled_shape_pairs": sum(e["same_unlabelled_shape"] for e in events),
        "identical_small_clade_sets_pairs": sum(e["new_small_clades"]+e["lost_small_clades"] == 0 for e in events),
        "new_full_clades_median": float(np.median([e["new_nonroot_clades"] for e in events])) if events else None,
        "new_full_clades_range": [min(e["new_nonroot_clades"] for e in events), max(e["new_nonroot_clades"] for e in events)] if events else None,
        "score_crossing_first_changes": sum(e["comparison_kind"] == "selected split scores cross" for e in events),
        "candidate_set_first_changes": sum(e["comparison_kind"] == "candidate-set change matters" for e in events),
        "peak_overall_xi_G": float(gs[np.argmax([r["overall_xi_nats"] for r in records])]),
        "peak_overall_xi_nats": max(r["overall_xi_nats"] for r in records),
        "maximum_closure_error_bits": max(abs(r["validation"]["closure_error_bits"]) for r in records),
        "selected_tree_tolerance_negative_count": sum(r["validation"]["tolerance_negative_count"] for r in records),
        "search_tolerance_negative_count": sum(r["search_audit"]["tolerance_zero_count"]+r["pair_tolerance_zero_count"] for r in records),
        "fixed_evaluation_tolerance_negative_count": n_negative,
        "minimum_fixed_or_cross_scored_syn_bits": minimum if np.isfinite(minimum) else None,
        "significant_nonnegativity_violations": 0,
        "nonnegative_tolerance_bits": TOL,
    }
    payload = {"status": "preview" if preview else "complete", "G": gs.tolist(), "seed": 4,
               "selection_rule": "12 clades with 2..20 ROI and >=5 consecutive grid points; rank peak selected local Syn; exploratory",
               "stats": stats, "tracked": tracked, "original_reference": original, "events": events,
               "labels": meta["labels"], "config": records[0]["config"],
               "fixed_split_definition": "hold ROI members and reference two-child partition fixed at every G; values retained when clade is absent from SPT"}
    write_json(input_dir / ("analysis_preview.json" if preview else "analysis.json"), payload)
    return payload


def axis_G(ax, gs):
    ax.set_xlim(gs[0], gs[-1] if len(gs)>1 else gs[0]+.01)
    ax.set_xlabel("Global coupling G")
    ax.tick_params(direction="out", length=3)


def presence_panel(ax, rows, gs, *, labels):
    data = np.asarray([r["mask"] for r in rows], dtype=int)
    if not len(rows):
        ax.text(.5, .5, "No eligible clades", transform=ax.transAxes, ha="center")
        return
    edges = np.r_[gs-.005, gs[-1]+.005]
    ax.pcolormesh(edges, np.arange(len(rows)+1)-.5, data,
                  cmap=ListedColormap(["#F2F2F2", "#267A70"]), vmin=0, vmax=1, shading="flat")
    ax.set_yticks(np.arange(len(rows)), labels=labels)
    ax.set_ylim(len(rows)-.5, -.5)
    ax.tick_params(axis="y", length=0)
    axis_G(ax, gs)


def figure_overview(meta, records, analysis, output):
    gs = np.asarray(analysis["G"])
    events = analysis["events"]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(14.8, 13.4), layout="constrained")
        grid = fig.add_gridspec(3, 2, height_ratios=[.95, 1.15, 1.2], width_ratios=[1, 1])
        ax = fig.add_subplot(grid[0, 0])
        for key, color, label, style in (("overall_xi_nats", INK, "Overall Xi", "-"),
                                       ("cross_roi_xi_nats", "#267A70", "Cross-ROI Xi", "-"),
                                       ("within_roi_xi_nats", "#A57950", "Within-ROI E/I Xi", "--")):
            ax.plot(gs, [r[key] for r in records], color=color, lw=1.5, label=label, linestyle=style)
        ax.set_ylabel("Integrated effective information (nats)")
        ax.set_title("a  Information budget", loc="left", fontweight="bold")
        ax.set_ylim(bottom=0)
        axis_G(ax, gs)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.25), frameon=False, fontsize=8, ncol=2)

        ax = fig.add_subplot(grid[0, 1])
        eg = [e["G"] for e in events]
        ax.plot(eg, [e["new_nonroot_clades"] for e in events], color="#9EA7B0", lw=.95, label="New non-root clades (of 98)")
        ax.plot(eg, [e["new_small_clades"] for e in events], color="#267A70", lw=1.1, label="New clades with <=20 ROI")
        ax.set(ylabel="Newly selected clades vs previous G", ylim=(0, 100))
        ax.set_title("b  Discrete changes at dG = 0.01", loc="left", fontweight="bold")
        axis_G(ax, gs)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.25), frameon=False, fontsize=8)

        ax = fig.add_subplot(grid[1, :])
        rows = analysis["tracked"]
        presence_panel(ax, rows, gs, labels=[f"{r['id']}  (n={r['size']})" for r in rows])
        ax.set_title("c  Persistent small coalitions selected by the independent trees", loc="left", fontweight="bold")
        ax.legend(handles=[Patch(facecolor="#267A70", label="Selected"), Patch(facecolor="#F2F2F2", label="Absent from selected tree")],
                  loc="lower center", bbox_to_anchor=(.5, 1.025), ncol=2, frameon=False, fontsize=8)

        ax = fig.add_subplot(grid[2, 0])
        rows = analysis["original_reference"]
        presence_panel(ax, rows, gs, labels=[f"Original n={r['size']}" for r in rows])
        ax.set_title("d  Original G=1.3 groups (different input sample)", loc="left", fontweight="bold")

        ax = fig.add_subplot(grid[2, 1])
        selected = sorted(analysis["tracked"], key=lambda r: r["strength_rank"])[:4]
        for r, color in zip(selected, CURVE_COLORS, strict=False):
            ax.plot(gs, r["fixed_syn_nats"], color=color, lw=1.4, label=f"{r['id']} (n={r['size']})")
        ax.set_ylabel("Syn of a frozen two-child split (nats)")
        ax.set_title("e  Fixed groups change strength outside the tree", loc="left", fontweight="bold")
        axis_G(ax, gs)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.25), frameon=False, ncol=2, fontsize=8)
        fig.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(fig)


def shared_guides(fig, meta, vmax):
    handles = [Patch(facecolor=color, label=name) for color, name in zip(NETWORK_COLORS, meta["network_names"], strict=True)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.39, .993), ncol=4,
               frameon=False, fontsize=8, handlelength=1, columnspacing=1.3)
    cax = fig.add_axes([.79, .945, .18, .012])
    cmap = LinearSegmentedColormap.from_list("syn", [_blend_with_white(SPLIT_COLOR, .12), _blend_with_white(SPLIT_COLOR, .74)])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0, vmax), cmap=cmap), cax=cax,
                      orientation="horizontal", ticks=[0, vmax])
    cb.set_label("Local Syn / overall Xi (%)", fontsize=8, labelpad=3)
    cax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=8)


def representatives(meta, records, trees, analysis, output):
    gs = np.asarray(analysis["G"])
    wanted = (0., .3, .5, .8, 1.3, 1.7)
    indices = sorted(set(int(np.argmin(abs(gs-g))) for g in wanted))
    vmax = math.ceil(max(r["maximum_local_syn_share_percent"] for r in records)*2)/2
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))
        fig.subplots_adjust(left=.02, right=.985, bottom=.035, top=.91, hspace=.22, wspace=.1)
        shared_guides(fig, meta, vmax)
        for ax, i in zip(axes.ravel(), indices, strict=False):
            render_tree(trees[i], None, labels=meta["labels"], network_membership=np.asarray(meta["network_membership"]),
                        network_names=meta["network_names"], seed=4, coupling_g=float(gs[i]), dpi=170, axis=ax,
                        network_colors=NETWORK_COLORS, information_unit="nats", node_value_mode="root_share",
                        syn_color_max=vmax, show_colorbar=False, show_roi_labels=False,
                        show_network_strip_label=False, node_label_limit=4)
            ax.set_ylim(-.08, 1.06)
        for ax in axes.ravel()[len(indices):]:
            ax.axis("off")
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return vmax


def animation(meta, records, trees, analysis, output, vmax, preview=False):
    gs = np.asarray(analysis["G"])
    frames = []
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(14.8, 8.8), dpi=110, facecolor="white")
        ax = fig.add_axes([.025, .30, .95, .60])
        timeline = fig.add_axes([.12, .055, .82, .16])
        shared_guides(fig, meta, vmax)
        rows = analysis["tracked"]
        presence_panel(timeline, rows, gs, labels=[f"{r['id']} (n={r['size']})" for r in rows])
        timeline.tick_params(labelsize=6)
        cursor = timeline.axvline(gs[0], color="#B34B37", lw=1.2)
        fig.text(.025, .238, "Exact coalition identities are tracked below; each tree uses its own readable leaf order.",
                 fontsize=8, color="#56616C")
        for i, (r, tree) in enumerate(zip(records, trees, strict=True)):
            ax.clear()
            render_tree(tree, None, labels=meta["labels"], network_membership=np.asarray(meta["network_membership"]),
                        network_names=meta["network_names"], seed=4, coupling_g=r["G"], dpi=110, axis=ax,
                        network_colors=NETWORK_COLORS, information_unit="nats", node_value_mode="root_share",
                        syn_color_max=vmax, show_colorbar=False, show_roi_labels=True,
                        show_network_strip_label=False, node_label_limit=4)
            ax.set_ylim(-.19, 1.06)
            cursor.set_xdata([r["G"], r["G"]])
            fig.canvas.draw()
            rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB")
            frames.append(rgb.convert("P", palette=Image.Palette.ADAPTIVE, colors=192))
            if not preview and (i+1) % 30 == 0:
                print(f"rendered animation {i+1}/{len(records)}", flush=True)
        duration = [350]*len(frames)
        duration[0] = duration[-1] = 1500
        frames[0].save(output, save_all=True, append_images=frames[1:], loop=0,
                       duration=duration, optimize=False, disposal=2)
        plt.close(fig)


def interactive(meta, records, trees, analysis, output, vmax):
    """Self-contained canvas figure, accessible controls, exact memberships/data."""
    frames = []
    for r, tree in zip(records, trees, strict=True):
        order = _leaf_order(tree)
        frames.append({"G": r["G"], "xi": r["overall_xi_nats"], "depth": r["tree_metrics"]["maximum_depth"],
                       "order": order, "nodes": [{"members": list(n.indices), "children": [list(c.indices) for c in n.children],
                                                    "syn": n.syn_bits*math.log(2), "xi": n.xi_bits*math.log(2)} for n in flatten_nodes(tree)]})
    data = {"frames": frames, "labels": meta["labels"], "networks": meta["network_names"],
            "membership": meta["network_membership"], "colors": NETWORK_COLORS,
            "tracked": analysis["tracked"], "original": analysis["original_reference"], "vmax": vmax,
            "tolerance_nats": TOL*math.log(2)}
    template = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>平均 SC：SPT 随 G 的变化</title>
<style>body{font:15px system-ui,sans-serif;margin:24px;color:#24313c;background:white}main{max-width:1500px;margin:auto}h1{font-size:22px}button,select,input{font:inherit}button{padding:6px 12px;border:1px solid #9ca8ac;background:white;border-radius:4px;cursor:pointer}input[type=range]{flex:1;min-width:180px}#controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}canvas{display:block;width:100%;border:1px solid #e3e7e9;margin-top:15px}#legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;margin-top:16px}.sw{display:inline-block;width:12px;height:12px;margin-right:5px}#selected{padding:12px;background:#f4f7f7;line-height:1.65}.note{font-size:13px;color:#56616c}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:8px;border-bottom:1px solid #e3e7e9;vertical-align:top}summary{cursor:pointer;padding:14px 0}a{color:#267a70}#scale{height:12px;width:200px;background:linear-gradient(to right,#e5efed,#5e9b95);display:inline-block}#status{min-width:220px}#clade{max-width:530px}</style>
<main><h1>平均 SC：团簇在什么 G 下被选中、保留或替换？</h1>
<p>seed 4 · 100 ROI · G 步长 0.01 · 各 G 独立构树 · 共同输入与噪声 · 原图切分搜索</p>
<div id="controls"><button id="prev" aria-label="前一个 G">←</button><button id="play">播放</button><button id="next" aria-label="后一个 G">→</button><input id="slider" type="range" min="0" max="170" value="130" aria-label="G 扫描位置"><select id="gselect" aria-label="选择 G"></select><span id="status" aria-live="polite"></span></div>
<div id="legend"></div><p class="note"><span id="scale"></span> 局部 Syn / 整体 Ξ：0–<span id="vmax"></span>%（全部帧共用）。树高为 ROI 数的对数刻度。</p>
<canvas id="tree" width="1800" height="900" role="img" aria-label="当前 G 的完整 SPT；成员及原始数值也可在下方表格读取"></canvas>
<p class="note">每棵树保留自己的可读叶序，横向移动不能解释为 ROI 的移动。点击内部节点查看完整成员。播放只切换真实采样帧，不做树形插值。</p>
<div id="nodeDetail" class="note" aria-live="polite"></div>
<label for="clade">追踪联盟：</label> <select id="clade"></select><div id="selected"></div>
<canvas id="curves" width="1800" height="490" role="img" aria-label="固定成员和固定参考二分的 Syn 曲线；选入树的 G 区间在下方色带表示"></canvas>
<p class="note">曲线冻结成员和参考二分，未被当前树选中时仍计算 Syn。色带只表示选入树，不是显著性判据。G=0 含有限样本背景。单 seed 轨迹不能确认动力学相变。</p>
<details><summary>读取当前树的全部内部节点与数值</summary><div id="nodeTable"></div></details>
<details><summary>追踪联盟的完整成员、逐 G 数值与保留区间</summary><div id="cladeTable"></div></details>
<p><button id="download">下载内嵌原始图数据 JSON</button> · <a href="spt_G_resolution_overview.png">静态总览图</a> · <a href="spt_G_resolution.gif">完整动画</a></p>
<p class="note">持久小团簇为探索性选择：2–20 ROI、至少连续 5 个采样点，再按峰值局部 Syn 选前 12 个。“原图”联盟来自用户给出的未配对 G=1.3 树；本扫描使用共同输入，成员差异也可能包含抽样差异。</p></main>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),F=D.frames,all=[...D.tracked,...D.original];let index=Math.min(130,F.length-1),timer=null,chosen=0,hit=[];
const $=id=>document.getElementById(id), key=a=>a.join(','), esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const short=s=>s.replace('7Networks_LH_','L-').replace('7Networks_RH_','R-').replace('SalVentAttn','SalVent').replace('Default','DMN');
$('slider').max=F.length-1;$('vmax').textContent=D.vmax.toFixed(2);
F.forEach((f,i)=>$('gselect').add(new Option('G = '+f.G.toFixed(2),i)));
D.networks.forEach((n,i)=>$('legend').insertAdjacentHTML('beforeend','<span><i class="sw" style="background:'+D.colors[i]+'"></i>'+esc(n)+'</span>'));
all.forEach((r,i)=>$('clade').add(new Option((r.id.startsWith('O')?'原图 ':'')+r.id+' · n='+r.size,i)));
function show(i){index=Math.max(0,Math.min(F.length-1,i));$('slider').value=index;$('gselect').value=index;let f=F[index];$('status').textContent='Ξ = '+f.xi.toFixed(3)+' nats · 深度 '+f.depth;draw();detail();}
function draw(){let f=F[index],c=$('tree').getContext('2d'),W=1800,H=900;c.clearRect(0,0,W,H);let map=new Map(f.nodes.map(n=>[key(n.members),n])),pos=new Map(),leaf=new Map(f.order.map((r,i)=>[r,38+i*17.3]));
 function place(n){let k=key(n.members);if(pos.has(k))return pos.get(k);let y=690-610*Math.log2(n.members.length)/Math.log2(100),x;if(!n.children.length)x=leaf.get(n.members[0]);else x=n.children.reduce((a,k)=>a+place(map.get(key(k)))[0]*k.length,0)/n.members.length;pos.set(k,[x,y]);return [x,y]}
 f.nodes.forEach(place);c.strokeStyle='#b2bac3';c.lineWidth=1.1;
 f.nodes.filter(n=>n.children.length).forEach(n=>{let [x,y]=pos.get(key(n.members)),p=n.children.map(k=>pos.get(key(k)));c.beginPath();c.moveTo(p[0][0],p[0][1]);c.lineTo(p[0][0],y);c.lineTo(p[1][0],y);c.lineTo(p[1][0],p[1][1]);c.stroke()});
 hit=[];let selected=all[chosen],sk=selected?key(selected.members):'',found=false;
 f.nodes.forEach(n=>{let [x,y]=pos.get(key(n.members));if(n.children.length){let raw=n.syn;if(raw< -D.tolerance_nats)throw Error('Significant negative Syn');let v=raw<0?0:raw,q=100*v/f.xi/D.vmax;c.fillStyle='rgb('+[38,122,112].map(v=>Math.round(255-(255-v)*(.12+.62*q))).join(',')+')';c.strokeStyle='#5e9b95';c.lineWidth=1+2*q;c.beginPath();c.arc(x,y,3+4*q,0,2*Math.PI);c.fill();c.stroke();hit.push({x,y,n});if(key(n.members)===sk){found=true;c.strokeStyle='#b34b37';c.lineWidth=3;c.beginPath();c.arc(x,y,12,0,2*Math.PI);c.stroke()}}else{let roi=n.members[0];c.strokeStyle=D.colors[D.membership[roi]];c.fillStyle='white';c.lineWidth=2;c.beginPath();c.arc(x,y,4,0,2*Math.PI);c.fill();c.stroke();c.fillStyle=D.colors[D.membership[roi]];c.fillRect(x-6,707,12,8);c.save();c.translate(x,726);c.rotate(Math.PI/2);c.font='9px sans-serif';c.fillStyle='#56616c';c.fillText(short(D.labels[roi]),0,0);c.restore()}});
 c.font='20px sans-serif';c.fillStyle='#24313c';c.textAlign='center';c.fillText('G = '+f.G.toFixed(2)+'  |  Ξ = '+f.xi.toFixed(3)+' nats',900,35);c.textAlign='left';if(selected){c.font='14px sans-serif';c.fillText(selected.id+' (n='+selected.size+')：'+(found?'红圈标出当前选中的联盟':'当前树未选中该精确联盟'),30,880)}
 let nodes=f.nodes.filter(n=>n.children.length).sort((a,b)=>a.members.length-b.members.length||a.members[0]-b.members[0]);$('nodeTable').innerHTML='<table><tr><th>ROI 数</th><th>局部 Syn / nats</th><th>整体子集 Ξ / nats</th><th>完整成员（ROI 从 1 编号）</th></tr>'+nodes.map(n=>'<tr><td>'+n.members.length+'</td><td>'+n.syn.toFixed(7)+'</td><td>'+n.xi.toFixed(7)+'</td><td>'+n.members.map(i=>(i+1)+': '+esc(short(D.labels[i]))).join(', ')+'</td></tr>').join('')+'</table>';
}
function detail(){let r=all[chosen];if(!r)return;$('selected').innerHTML='<b>'+esc(r.id)+' · n='+r.size+'</b> — 当前 '+(r.mask[index]?'已选中':'未选中')+'；本扫描 '+r.selected_points+'/'+F.length+' 个格点选中。<br>'+r.members.map(i=>'<span><i class="sw" style="background:'+D.colors[D.membership[i]]+'"></i>'+(i+1)+': '+esc(short(D.labels[i]))+'</span>').join(' · ')+'<br><small>冻结参考二分：G='+r.reference_G.toFixed(2)+'；两组 ROI 编号为 '+r.reference_split.map(a=>a.map(j=>j+1).join(', ')).join(' | ')+'。当前树节点的二分可能不同。</small>';curve(r);$('cladeTable').innerHTML='<p>区间：'+(r.intervals.map(s=>s.first_G.toFixed(2)+'–'+s.last_G.toFixed(2)+' ('+s.points+' 点)').join('；')||'没有选中格点')+'</p><table><tr><th>G</th><th>选入树</th><th>固定二分 Syn / nats</th><th>固定联盟跨 ROI Ξ / nats</th></tr>'+F.map((f,i)=>'<tr><td>'+f.G.toFixed(2)+'</td><td>'+(r.mask[i]?'是':'否')+'</td><td>'+r.fixed_syn_nats[i].toFixed(8)+'</td><td>'+r.fixed_cross_roi_xi_nats[i].toFixed(8)+'</td></tr>').join('')+'</table>';}
function curve(r){let c=$('curves').getContext('2d'),W=1800,H=490;c.clearRect(0,0,W,H);let x=g=>90+(g-F[0].G)/(F.at(-1).G-F[0].G)*1650,max=Math.max(...r.fixed_syn_nats)*1.12||1,y=v=>350-v/max*285;c.strokeStyle='#849198';c.lineWidth=1;c.beginPath();c.moveTo(90,55);c.lineTo(90,350);c.lineTo(1740,350);c.stroke();c.fillStyle='#24313c';c.font='14px sans-serif';for(let j=0;j<=4;j++){let v=max*j/4;c.fillText(v.toFixed(3),20,y(v)+5)}for(let tick=0;tick<=17;tick++){let g=tick/10;if(g<F[0].G||g>F.at(-1).G)continue;c.fillText(g.toFixed(1),x(g)-8,380)}c.font='17px sans-serif';c.fillText(r.id+'：固定成员与 G='+r.reference_G.toFixed(2)+' 参考二分的 Syn（nats）',90,30);c.strokeStyle='#267a70';c.lineWidth=3;c.beginPath();F.forEach((f,i)=>i?c.lineTo(x(f.G),y(r.fixed_syn_nats[i])):c.moveTo(x(f.G),y(r.fixed_syn_nats[i])));c.stroke();let step=F.length>1?x(F[1].G)-x(F[0].G):5;F.forEach((f,i)=>{c.fillStyle=r.mask[i]?'#267a70':'#f2f2f2';c.fillRect(x(f.G)-step/2,404,step+0.3,22)});c.strokeStyle='#b34b37';c.lineWidth=2;c.beginPath();c.moveTo(x(F[index].G),55);c.lineTo(x(F[index].G),427);c.stroke();c.fillStyle='#24313c';c.fillText('色带：该精确联盟是否选入树',90,460);}
$('slider').oninput=e=>show(+e.target.value);$('gselect').onchange=e=>show(+e.target.value);$('prev').onclick=()=>show(index-1);$('next').onclick=()=>show(index+1);$('clade').onchange=e=>{chosen=+e.target.value;draw();detail()};$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放'}else{timer=setInterval(()=>show((index+1)%F.length),350);$('play').textContent='暂停'}};document.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)&&e.target.tagName!=='SELECT'){e.preventDefault();show(index+(e.key==='ArrowLeft'?-1:1))}});
$('tree').onclick=e=>{let box=e.target.getBoundingClientRect(),x=(e.clientX-box.left)*1800/box.width,y=(e.clientY-box.top)*900/box.height,nearest=hit.filter(h=>Math.hypot(h.x-x,h.y-y)<14).sort((a,b)=>Math.hypot(a.x-x,a.y-y)-Math.hypot(b.x-x,b.y-y))[0];if(nearest){let n=nearest.n;$('nodeDetail').innerHTML='<b>点击的当前树节点 · n='+n.members.length+' · Syn='+n.syn.toFixed(7)+' nats</b><br>'+n.members.map(i=>(i+1)+': '+esc(short(D.labels[i]))).join(' · ')+'<br>下方曲线对应追踪联盟下拉框的选择。'}};
$('download').onclick=()=>{let url=URL.createObjectURL(new Blob([JSON.stringify(D)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='spt_G_resolution_figure_data.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000)};show(index);
</script></html>'''
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    output.write_text(template.replace("__DATA__", text))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=OUTPUT)
    p.add_argument("--figure-dir", type=Path, default=FIGURES)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--skip-animation", action="store_true")
    args = p.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        meta, records, trees = load(args.input_dir, args.preview)
        analysis = analyze(meta, records, trees, args.input_dir, args.preview)
        figure_overview(meta, records, analysis, args.figure_dir / "spt_G_resolution_overview.png")
        vmax = representatives(meta, records, trees, analysis, args.figure_dir / "spt_G_resolution_representative_trees.png")
        if not args.skip_animation:
            animation(meta, records, trees, analysis, args.figure_dir / "spt_G_resolution.gif", vmax, args.preview)
        interactive(meta, records, trees, analysis, args.figure_dir / "spt_G_resolution_explorer.html", vmax)
        print(json.dumps(analysis["stats"], indent=2), flush=True)
        print("FIGURES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
