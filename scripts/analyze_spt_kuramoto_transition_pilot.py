#!/usr/bin/env python3
"""Audit and plot paired pilot caches; failed EI never enters tree analysis."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_spt_kuramoto_transition_pilot import GRID, TRACKS

ASSETS = ROOT / "docs/reports/assets/kuramoto_hierarchy"
COLORS = {TRACKS[0]: "#256BA6", TRACKS[1]: "#CB7637"}
STYLES = {0: "-", 1: "--", 2: ":"}


def load_records(directories):
    records = {}
    for directory in directories:
        for path in sorted(directory.glob("kout_*_seed_*.json")):
            r = json.loads(path.read_text())
            key = (r["track"], r["seed"], r["k_out"])
            if key in records and records[key] != r:
                raise RuntimeError(f"Conflicting cached condition {key}")
            records[key] = r
    for (track, seed, k), r in records.items():
        partner = records.get((TRACKS[1] if track == TRACKS[0] else TRACKS[0], seed, k))
        if partner is None:
            raise RuntimeError(f"Missing paired track at K={k}, seed={seed}")
        if partner["shared_channel_hashes"] != r["shared_channel_hashes"]:
            raise RuntimeError(f"Unpaired arrays at K={k}, seed={seed}")
        if r["status"] != "valid":
            continue
        if len(r["ei_table"]) != 63 or len(r["partition_candidates"]) != 301 or len(r["nodes"]) != 11:
            raise RuntimeError(f"Incomplete table/tree at {key}")
        scalars = [v["ei_bits"] for v in r["ei_table"]]
        scalars += [v["syn_bits"] for v in r["partition_candidates"]]
        scalars += [r["root_best_second_gap_bits"], r["closure_error_bits"]]
        if not np.isfinite(scalars).all():
            raise RuntimeError(f"Nonfinite estimate at {key}")
        audit = r["nonnegative_audit"]
        if audit["violating_partition_count"] or audit["violating_xi_count"]:
            raise RuntimeError(f"Invalid Syn labeled valid at {key}")
        if abs(r["closure_error_bits"]) > 1e-8:
            raise RuntimeError(f"Closure failed at {key}")
    return sorted(records.values(), key=lambda r: (r["track"], r["seed"], r["k_out"]))


def split_label(r):
    return " | ".join("".join(s.removeprefix("theta") for s in block) for block in r["root_split"])


def clades(r):
    return frozenset(tuple(c) for c in r["nontrivial_clades"])


def first_event(rows, predicate):
    for i, r in enumerate(rows):
        if r["status"] == "valid" and predicate(r):
            previous = rows[i - 1] if i else None
            previous_valid = next((q for q in reversed(rows[:i]) if q["status"] == "valid"), None)
            return {"low": previous["k_out"] if previous else None, "high": r["k_out"],
                    "lower_valid": previous is not None and previous["status"] == "valid",
                    "last_valid_unchanged_k": previous_valid["k_out"] if previous_valid else None}
    return None


def events(records):
    results = []
    for track in TRACKS:
        for seed in sorted({r["seed"] for r in records}):
            rows = [r for r in records if r["track"] == track and r["seed"] == seed]
            if not rows:
                continue
            baseline = next((r for r in rows if r["k_out"] == 0 and r["status"] == "valid"), None)
            topology = first_event(rows, lambda r: clades(r) != clades(baseline)) if baseline else None
            component = next((r["k_out"] for r in rows
                if len(r["dynamics_diagnostics"]["learned_dependency_components"]) == 1), None)
            loss = first_event(rows, lambda r: not r["planted_root_split"])
            chain = first_event(rows, lambda r: r["sackin_index"] == 20)
            crossings, chain_crossings = [], []
            for left, right in zip(rows, rows[1:]):
                if left["status"] == right["status"] == "valid" and left["planted_root_split"] != right["planted_root_split"]:
                    crossings.append([left["k_out"], right["k_out"]])
                if left["status"] == right["status"] == "valid" and (left["sackin_index"] == 20) != (right["sackin_index"] == 20):
                    chain_crossings.append({"low": left["k_out"], "high": right["k_out"],
                                            "enters_chain": right["sackin_index"] == 20})
            results.append({"track": track, "seed": seed, "zero_baseline_available": baseline is not None,
                            "first_topology_change": topology, "first_one_component_k": component,
                            "first_module_loss": loss, "first_chain": chain,
                            "all_planted_root_crossings": crossings,
                            "all_chain_crossings": chain_crossings})
    return results


def refinement_intervals(records):
    """All adjacent coarse intervals with a clade change, grouped without gaps."""
    spans = set()
    for track in TRACKS:
        for seed in (0, 1, 2):
            rows = [r for r in records if r["track"] == track and r["seed"] == seed
                    and r["k_out"] in GRID]
            for a, b in zip(rows, rows[1:]):
                if a["status"] == b["status"] == "valid" and clades(a) != clades(b):
                    spans.add((a["k_out"], b["k_out"]))
    return sorted(spans)


def refinement_plan(records):
    """Batch midpoint refinement of all observed boundaries, without one-root monotonicity."""
    spans = set()
    priorities = {}
    for seed in (0, 1, 2):
        rows = sorted((r for r in records if r["track"] == TRACKS[0] and r["seed"] == seed),
                      key=lambda r: r["k_out"])
        for left, right in zip(rows, rows[1:]):
            if left["status"] == right["status"] == "valid" and clades(left) != clades(right):
                spans.add((left["k_out"], right["k_out"]))
            if len(left["estimator_components"]) != len(right["estimator_components"]):
                span = (left["k_out"], right["k_out"])
                spans.add(span)
                priorities[span] = 0
        for event in events(rows):
            for rank, field in enumerate(("first_topology_change", "first_module_loss", "first_chain"), start=1):
                value = event[field]
                if value and value["low"] is not None and value["lower_valid"]:
                    span = (value["low"], value["high"])
                    spans.add(span)
                    priorities[span] = min(priorities.get(span, 99), rank)
        # This threshold triggers exploration only; it is not a significance test.
        for left, right in zip(rows, rows[1:]):
            if left["status"] == right["status"] == "valid" and min(
                    left["root_best_second_gap_bits"], right["root_best_second_gap_bits"]) <= .02:
                spans.add((left["k_out"], right["k_out"]))
    ordered = sorted((span for span in spans if span[1] - span[0] > .05 + 1e-9),
                     key=lambda span: (priorities.get(span, 4), span))
    existing = {r["k_out"] for r in records}
    points = [round((a + b) / 2, 8) for a, b in ordered if round((a + b) / 2, 8) not in existing]
    primary = [round((a + b) / 2, 8) for a, b in ordered if priorities.get((a, b), 4) < 4
               and round((a + b) / 2, 8) not in existing]
    return {"maximum_grid_width": .05, "small_gap_exploration_trigger_bits": .02,
            "ordered_intervals": ordered, "k_out_priority_order": points,
            "first_event_midpoints": primary,
            "planned_extra_records": len(points) * 6,
            "strategy": "batch midpoints, recompute all observed boundaries after each round; equal-endpoint hidden re-entry remains unexcluded",
            "budget_policy": "first events first, then other changed/small-gap intervals; stop new pairs at remaining pilot budget"}


def state(r):
    if r is None or r["status"] != "valid":
        return 4
    if r["planted_root_split"]:
        return 0
    small = min(map(len, r["root_split"]))
    return {3: 1, 2: 2, 1: 3}[small]


def overview(records, path):
    # Seed curves are individual experimental repeats, not confidence intervals.
    palette = ["#56A28B", "#E5A158", "#75A5C9", "#AB88BA", "#D6D9DD"]
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(3, 2, figsize=(14.2, 10.5), constrained_layout=True)
        ax = axes[0, 0]
        matrix = [[state(next((r for r in records if r["track"] == track and r["seed"] == seed
                              and r["k_out"] == k), None)) for k in GRID]
                  for track in TRACKS for seed in (0, 1, 2)]
        ax.imshow(matrix, aspect="auto", cmap=ListedColormap(palette), vmin=-.5, vmax=4.5,
                  interpolation="nearest")
        ax.set_xticks(range(len(GRID)), [f"{k:g}" for k in GRID], rotation=60)
        ax.set_yticks(range(6), [f"{prefix} seed {seed}" for prefix in ("Components", "Full") for seed in (0, 1, 2)])
        ax.set_xlabel(r"Sampled $K_{\mathrm{out}}$ (equal-spaced columns)")
        ax.set_title("a  Root partition at coarse-grid conditions", loc="left", weight="bold")
        ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(palette,
                    ["123 | 456", "Other 3 | 3", "2 | 4", "1 | 5", "Invalid EI / missing"])],
                  loc="center left", bbox_to_anchor=(1.02, .5), frameon=False, fontsize=8)
        fields = [("maximum_cross_module_effect", "Permutation effect / output SD", "b  Learned channel connectivity"),
                  ("planted_vs_best_other_margin_bits", r"$\Delta_P$ (bits)", "c  Original root versus best alternative"),
                  ("root_syn_fraction", "Raw root Syn estimate / Ξ (%)", "d  Root synergy allocation"),
                  ("sackin_index", "Sum of leaf depths", "e  Tree shape"),
                  ("root_best_second_gap_bits", "Best–second root score gap (bits)", "f  Root selection margin")]
        for ax, (field, ylabel, title) in zip(axes.flat[1:], fields):
            for track in TRACKS:
                if field == "maximum_cross_module_effect" and track == TRACKS[1]:
                    continue
                for seed in (0, 1, 2):
                    rows = [r for r in records if r["track"] == track and r["seed"] == seed]
                    x = [r["k_out"] for r in rows]
                    y = [(r["dynamics_diagnostics"][field] if field == "maximum_cross_module_effect" else
                          r.get(field) if r["status"] == "valid" else None) for r in rows]
                    y = np.array([float(v) if v is not None else np.nan for v in y])
                    if field == "root_syn_fraction":
                        y *= 100
                    ax.plot(x, y, color=COLORS[track], linestyle=STYLES[seed], marker="o" if track == TRACKS[0] else "x",
                            markersize=2.8, linewidth=1.25, alpha=.9)
            if field == "maximum_cross_module_effect":
                ax.axhline(.25, color="#555555", linestyle="-.", linewidth=.8)
            if field == "planted_vs_best_other_margin_bits":
                ax.axhline(0, color="#777777", linewidth=.7)
            if field == "sackin_index":
                ax.set_yticks([16, 17, 18, 19, 20])
                ax.set_ylim(15.6, 20.4)
            ax.set(xlabel=r"$K_{\mathrm{out}}$", ylabel=ylabel, xlim=(-.05, 5.05))
            ax.set_title(title, loc="left", weight="bold")
        handles = [Line2D([], [], color=COLORS[t], marker="o" if t == TRACKS[0] else "x",
                          label="Components" if t == TRACKS[0] else "Full channel") for t in TRACKS]
        handles += [Line2D([], [], color="#555555", linestyle=STYLES[s], label=f"Seed {s}") for s in (0, 1, 2)]
        handles += [Line2D([], [], color="#555555", linestyle="-.", label="Threshold 0.25")]
        axes[0, 1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, .5), frameon=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def representative_trees(records, path):
    rows = [r for r in records if r["track"] == TRACKS[0] and r["seed"] == 0 and r["status"] == "valid"
            and r["k_out"] in GRID]
    if not rows:
        return
    selected = [rows[0]]
    for predicate in (lambda r: clades(r) != clades(rows[0]),
                      lambda r: not r["planted_root_split"], lambda r: r["sackin_index"] == 20):
        r = next((r for r in rows if predicate(r)), None)
        if r is not None and r not in selected:
            selected.append(r)
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9}):
        fig, axes = plt.subplots(1, len(selected), figsize=(4.3 * len(selected), 5), squeeze=False,
                                 constrained_layout=True)
        for ax, record in zip(axes[0], selected):
            lookup = {tuple(n["sources"]): n for n in record["nodes"]}
            positions = {}
            counter = [0]

            def layout(sources):
                node = lookup[tuple(sources)]
                if node["children"]:
                    x = np.mean([layout(child) for child in node["children"]])
                else:
                    x = counter[0]
                    counter[0] += 1
                positions[tuple(sources)] = (x, -node["depth"])
                return x

            layout(record["nodes"][0]["sources"])
            for node in record["nodes"]:
                x, y = positions[tuple(node["sources"])]
                for child in node["children"]:
                    cx, cy = positions[tuple(child)]
                    ax.plot([x, cx], [y, cy], color="#ABB5BE", linewidth=1, zorder=0)
                label = ",".join(s.removeprefix("theta") for s in node["sources"])
                if node["children"]:
                    label += f"\nSyn {100 * node['syn_bits'] / record['root_xi_bits']:.1f}%"
                ax.text(x, y, label, ha="center", va="center", fontsize=8,
                        bbox={"boxstyle": "round,pad=.3", "facecolor": "#E0EFEB" if node["children"] else "#F5F6F8",
                              "edgecolor": "#468E82" if node["children"] else "#A2ADB7"})
            ax.set(xlim=(-1, 6), ylim=(-5.7, .65))
            ax.axis("off")
            ax.set_title("Components · seed 0\n" + rf"$K_{{\mathrm{{out}}}} = {record['k_out']:g}$"
                         + f" · Sackin = {record['sackin_index']}", fontsize=10)
        fig.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, nargs="+", default=[ROOT / "results/spt_kuramoto_transition_pilot"])
    p.add_argument("--output", type=Path, default=ROOT / "results/spt_kuramoto_transition_pilot/analysis.json")
    args = p.parse_args()
    records = load_records(args.input)
    analysis = {"condition_count": len(records), "valid_count": sum(r["status"] == "valid" for r in records),
                "failed_count": sum(r["status"] != "valid" for r in records), "events": events(records),
                "coarse_changed_intervals": refinement_intervals(records),
                "refinement_plan": refinement_plan(records),
                "failure_conditions": [{"k_out": r["k_out"], "seed": r["seed"], "track": r["track"], "error": r.get("error"),
                                        "diagnostics": r.get("nonnegative_audit", r["component_marginal_diagnostics"])}
                                       for r in records if r["status"] != "valid"],
                "raw_tolerance_negative_count_valid": sum(r["nonnegative_audit"]["tolerance_negative_partition_count"]
                                                          for r in records if r["status"] == "valid"),
                "zero_point_controls": [{"seed": r["seed"], "track": r["track"], "status": r["status"],
                                         "control": r.get("zero_point_control", "unavailable")}
                                        for r in records if r["k_out"] == 0]}
    valid = [r for r in records if r["status"] == "valid"]
    analysis["coverage"] = {"paired_k_seed_conditions": len(records) // 2,
        "unique_k_count": len({r["k_out"] for r in records}),
        "points_by_seed": {str(s): len({r["k_out"] for r in records if r["seed"] == s}) for s in (0, 1, 2)},
        "shared_full_track_records": sum(r.get("reused_from") == TRACKS[0] for r in records),
        "paired_hashes_verified": True}
    analysis["valid_audit"] = {"tolerance_bits": .1, "raw_values_retained": True, "clipping": False,
        "maximum_absolute_closure_error_bits": max(abs(r["closure_error_bits"]) for r in valid),
        "minimum_partition_syn_bits": min(r["nonnegative_audit"]["minimum_partition_syn_bits"] for r in valid),
        "tolerance_negative_xi_count": sum(r["nonnegative_audit"]["tolerance_negative_xi_count"] for r in valid),
        "selected_tolerance_negative_count": sum(r["selected_tolerance_negative_count"] for r in valid)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
    overview(records, ASSETS / "spt_kuramoto_transition_pilot.png")
    representative_trees(records, ASSETS / "spt_kuramoto_transition_pilot_trees.png")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
