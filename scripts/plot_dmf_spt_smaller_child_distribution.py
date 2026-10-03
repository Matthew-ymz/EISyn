#!/usr/bin/env python3
"""Count disjoint clusters peeled along the larger-child spine of cached trees.

Each peeled child is counted once and is never subdivided in this statistic.
The final singleton is included, so size times frequency sums to all 100 ROI.
No Syn weighting, simulation, EI fit, or tree search is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_runge_slp_pc60_xi_horizons import _node_from_record
from scripts.plot_dmf_schaefer100_tree_examples import audit_tree

DEFAULT_INPUT = ROOT / "results/dmf_schaefer100/xi_hierarchy_tree/examples/wide/summary.json"
SINGLETON_COLOR = "#9AA7B2"
MULTI_COLOR = "#34877B"
INK = "#26343E"


def split_size_statistics(tree) -> dict:
    """Partition ROI along the larger-child spine; enumerate equal-size choices.

    At a tie, continue the child with the lexicographically smaller sorted ROI
    tuple. Alternative choices are evaluated without selecting a favorable one.
    Ties between singletons have identical size distributions and need no fork.
    """
    leaves = []

    def validate(node):
        if node.size != len(set(node.indices)):
            raise ValueError("Repeated ROI indices within a node")
        if not node.children:
            if node.size != 1:
                raise ValueError("Expected singleton ROI-block leaves")
            leaves.append(node.indices[0])
            return
        if len(node.children) != 2:
            raise ValueError("Smaller-child statistics require binary splits")
        left, right = node.children
        if set(left.indices) & set(right.indices) or set(left.indices) | set(right.indices) != set(node.indices):
            raise ValueError("Children do not partition their parent ROI set")
        validate(left)
        validate(right)

    validate(tree)
    if len(leaves) != tree.size or len(set(leaves)) != tree.size or set(leaves) != set(tree.indices):
        raise ValueError("Tree is not a complete binary hierarchy with unique ROI leaves")

    def options(node):
        if not node.children:
            return [([{"source_node_depth": node.depth, "roi_indices": sorted(node.indices),
                       "roi_count": node.size, "kind": "terminal_remainder",
                       "remaining_roi_count": 0}], [])]
        ordered = sorted(node.children, key=lambda child: (-child.size, tuple(sorted(child.indices))))
        continued, removed = ordered
        equal_size = continued.size == removed.size
        choices = [(continued, removed)]
        if equal_size and continued.size > 1:
            choices.append((removed, continued))
        result = []
        for continued, removed in choices:
            cluster = {"source_node_depth": node.depth, "roi_indices": sorted(removed.indices),
                       "roi_count": removed.size,
                       "kind": "peeled_equal_child" if equal_size else "peeled_smaller_child",
                       "remaining_roi_count": continued.size}
            tie = [{"depth": node.depth, "child_roi_count": continued.size,
                    "continued_roi_indices": sorted(continued.indices),
                    "removed_roi_indices": sorted(removed.indices)}] if equal_size else []
            for groups, downstream_ties in options(continued):
                result.append(([cluster, *groups], [*tie, *downstream_ties]))
        return result

    support = np.arange(1, max(1, tree.size // 2) + 1)

    def summarize(clusters):
        flattened = [index for cluster in clusters for index in cluster["roi_indices"]]
        if len(flattened) != tree.size or len(set(flattened)) != tree.size or set(flattened) != set(tree.indices):
            raise ValueError("Peeled clusters do not cover each ROI exactly once")
        sizes = np.asarray([cluster["roi_count"] for cluster in clusters], dtype=int)
        if np.any(sizes < 1) or np.any(sizes > support[-1]):
            raise ValueError("Peeled-cluster size falls outside the declared ROI support")
        counts = np.bincount(sizes, minlength=support[-1] + 1)[1:]
        probabilities = counts / len(clusters)
        roi_mass = support * counts
        positive = probabilities[probabilities > 0]
        entropy = float(np.sum(-positive * np.log2(positive)))
        if int(counts.sum()) != len(clusters) or int(roi_mass.sum()) != tree.size or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("Cluster frequency or ROI mass does not close")
        multi_roi_count = int(roi_mass[1:].sum())
        return {
            "roi_count": tree.size, "cluster_count": len(clusters),
            "spine_split_count": len(clusters) - 1, "terminal_remainder_roi_count": sizes[-1].item(),
            "support_roi_counts": support.tolist(), "counts": counts.tolist(),
            "probabilities": probabilities.tolist(), "roi_mass": roi_mass.tolist(),
            "roi_mass_sum": int(roi_mass.sum()),
            "singleton_cluster_count": int(counts[0]),
            "multi_roi_cluster_count": int(counts[1:].sum()),
            "multi_roi_cluster_fraction": float(counts[1:].sum() / len(clusters)),
            "multi_roi_coverage_count": multi_roi_count,
            "multi_roi_coverage_fraction": multi_roi_count / tree.size,
            "observed_size_count": int(np.count_nonzero(counts)),
            "mean_cluster_roi_count": float(sizes.mean()),
            "maximum_cluster_roi_count": int(sizes.max()),
            "size_distribution_entropy_bits": entropy,
        }

    partitions = options(tree)
    clusters, tie_decisions = partitions[0]
    canonical = summarize(clusters)
    variants = []
    seen = set()
    for alternative_clusters, decisions in partitions:
        alternative = summarize(alternative_clusters)
        signature = tuple(alternative["counts"])
        if signature not in seen:
            seen.add(signature)
            variants.append({**alternative, "tie_decisions": decisions})
    return {**canonical, "clusters": clusters, "tie_decisions": tie_decisions,
            "equal_size_paths_evaluated": len(partitions), "distinct_distribution_count": len(variants),
            "tie_variants": variants,
            "tie_sensitivity": {name: {"minimum": min(v[name] for v in variants),
                                       "maximum": max(v[name] for v in variants)}
                                for name in ("cluster_count", "multi_roi_coverage_fraction",
                                             "size_distribution_entropy_bits")}}


def group_statistics(rows) -> list[dict]:
    groups = []
    for coupling_g in sorted({row["coupling_g"] for row in rows}):
        group = [row for row in rows if row["coupling_g"] == coupling_g]
        result = {"coupling_g": coupling_g, "seed_count": len(group),
                  "seeds": [row["seed"] for row in group]}
        for name in ("cluster_count", "multi_roi_cluster_fraction", "multi_roi_coverage_fraction",
                     "observed_size_count", "mean_cluster_roi_count", "size_distribution_entropy_bits"):
            values = np.asarray([row[name] for row in group])
            result[name] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                            "minimum": float(values.min()), "maximum": float(values.max())}
        groups.append(result)
    return groups


def draw_distribution(axis, row, *, x_max, tail_y_max):
    support = np.asarray(row["support_roi_counts"])
    counts = np.asarray(row["counts"])
    colors = [SINGLETON_COLOR if size == 1 else MULTI_COLOR for size in support]
    axis.bar(support, counts, width=0.72, color=colors, linewidth=0, zorder=2)
    axis.set_xlim(0.35, x_max + 0.65)
    axis.set_ylim(0, 105)
    axis.set_xticks([1, 2, 4, 6, 8, 10, 12] if x_max == 12 else np.arange(1, x_max + 1))
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_ylabel("Number of clusters")
    axis.set_xlabel("Peeled-cluster size (ROI blocks)")
    axis.grid(axis="y", color="#E8ECEF", linewidth=0.6, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title(rf"$G={row['coupling_g']:g}$, seed {row['seed']}  |  {row['cluster_count']} clusters", pad=10)
    axis.text(1, counts[0] + 1.2, str(counts[0]),
              ha="center", va="bottom", color=INK, fontsize=8)
    axis.text(0.97, 0.96,
              f"H = {row['size_distribution_entropy_bits']:.3f} bits\n"
              f"ROIs in clusters > 1: {row['multi_roi_coverage_count']}/{row['roi_count']}\n"
              f"Size × count total = {row['roi_mass_sum']} ROI",
              transform=axis.transAxes, ha="right", va="top", color=INK,
              fontsize=8, linespacing=1.55)

    # Inset lies wholly above the small tail bars, in an empty data region.
    # It shows the same raw counts, without renormalizing the tail.
    inset = axis.inset_axes([0.34, 0.33, 0.62, 0.39])
    tail = (support >= 2) & (support <= x_max)
    inset.bar(support[tail], counts[tail], width=0.72, color=MULTI_COLOR, linewidth=0)
    inset.set_xlim(1.4, x_max + 0.6)
    inset.set_ylim(0, tail_y_max)
    inset.set_xticks([2, 4, 6, 8, 10, 12] if x_max == 12 else np.arange(2, x_max + 1))
    inset.set_yticks([0, 2, 4])
    inset.set_ylabel("Clusters", fontsize=6.5)
    inset.set_title(r"$k\geq2$ detail; raw counts", fontsize=7, pad=5)
    inset.tick_params(labelsize=6.5, length=2, pad=2)
    inset.grid(axis="y", color="#E8ECEF", linewidth=0.5)
    inset.spines[["top", "right"]].set_visible(False)
    for size in support[tail & (counts > 0)]:
        inset.text(size, counts[size - 1] + 0.1, str(counts[size - 1]),
                   ha="center", va="bottom", fontsize=6, color=INK)
    if row["multi_roi_cluster_count"] == 0:
        inset.text(0.5, 0.5, "No clusters with k > 1", transform=inset.transAxes,
                   ha="center", va="center", fontsize=7, color=INK)


def add_legend(figure):
    figure.legend(handles=[Patch(facecolor=SINGLETON_COLOR, label="Singleton clusters"),
                           Patch(facecolor=MULTI_COLOR, label="Clusters with two or more ROI")],
                  loc="upper center", bbox_to_anchor=(0.5, 0.998), ncol=2,
                  frameon=False, fontsize=9, handlelength=1.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dpi", type=int, default=260)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    rows = []
    for original in source["rows"]:
        path = Path(original["cache"])
        payload = json.loads(path.read_text())
        if payload["seed"] != original["seed"] or not np.isclose(payload["coupling_g"], original["coupling_g"]):
            raise ValueError("Input tree condition differs from its summary row")
        tree = _node_from_record(payload["tree"])
        validation = audit_tree(tree, original["overall_xi_bits"])
        rows.append({"seed": original["seed"], "coupling_g": original["coupling_g"],
                     "tree_cache": str(path), "tree_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "input_syn_audit": validation, **split_size_statistics(tree)})
    rows.sort(key=lambda row: (row["seed"], row["coupling_g"]))
    seeds = sorted({row["seed"] for row in rows})
    couplings = sorted({row["coupling_g"] for row in rows})
    conditions = {(row["seed"], row["coupling_g"]) for row in rows}
    if len(conditions) != len(rows) or len(rows) != len(seeds) * len(couplings):
        raise ValueError("Distribution grid has missing or repeated conditions")
    x_max = max(row["maximum_cluster_roi_count"] for row in rows)
    tail_max = max(max(row["counts"][1:]) for row in rows)
    tail_y_max = max(5.0, math.ceil(tail_max + 0.5))
    figure_dir = Path(source["figure"]).parent
    overview = figure_dir / "smaller_child_size_distributions_G000_130_300_seeds03_04_05.png"
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.7,
                         "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#62717C"}):
        figure, axes = plt.subplots(len(seeds), len(couplings), figsize=(15.2, 11.8), squeeze=False)
        figure.subplots_adjust(left=0.06, right=0.985, bottom=0.075, top=0.92, wspace=0.22, hspace=0.35)
        for index, row in enumerate(rows):
            axis = axes[index // len(couplings), index % len(couplings)]
            draw_distribution(axis, row, x_max=x_max, tail_y_max=tail_y_max)
            axis.text(-0.13, 1.045, chr(97 + index), transform=axis.transAxes,
                      fontsize=12, fontweight="bold", va="bottom")
        add_legend(figure)
        figure.text(0.06, 0.018,
                    "Follow the larger child only; count each peeled cluster once and include the final singleton. Size x count sums to 100.\n"
                    f"One ROI block contains paired E/I states. Ties follow the lexicographically smaller ROI tuple. Sizes > {x_max} are unobserved.",
                    fontsize=8, linespacing=1.5)
        figure.savefig(overview, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        for row in rows:
            output = figure_dir / f"smaller_child_sizes_G{round(row['coupling_g'] * 100):03d}_seed{row['seed']:02d}.png"
            figure, axis = plt.subplots(figsize=(6.2, 4.7))
            figure.subplots_adjust(left=0.12, right=0.96, bottom=0.16, top=0.84)
            draw_distribution(axis, row, x_max=x_max, tail_y_max=tail_y_max)
            add_legend(figure)
            figure.savefig(output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
            plt.close(figure)
            row["figure"] = str(output)
    summary = {
        "source_summary": str(args.input),
        "source_summary_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "metric_version": "main_spine_disjoint_partition_v1",
        "definition": "Count the smaller child once, follow only the larger child, then include the final singleton",
        "weighting": "One count per disjoint peeled cluster and terminal remainder; no Syn weights",
        "closure": "sum_k k * n(k) = 100; each ROI belongs to exactly one recorded cluster",
        "denominator": "For entropy, probabilities are raw cluster counts divided by the per-tree cluster count",
        "equal_size_rule": "Continue the child with the lexicographically smaller sorted ROI tuple; enumerate alternative size distributions",
        "source_granularity": "Each ROI is one leaf block with paired E/I state; scalar E/I counts are twice ROI counts",
        "entropy": "Shannon entropy of per-tree cluster-size probabilities in bits; zero-frequency terms omitted; not EI or Syn",
        "inset": "Same raw cluster counts; restricted to k >= 2 with a common enlarged y scale",
        "displayed_size_range": [1, x_max], "full_size_support": [1, 50],
        "figure": str(overview), "rows": rows, "groups": group_statistics(rows),
        "limitations": "Clusters from one tree are dependent. Equal-size continuation can change the distribution; all such size variants are recorded. Three paired seeds support descriptive comparisons, not a critical-transition test. G=0 retains finite-sample Gaussian estimation background.",
    }
    (args.input.parent / "smaller_child_size_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(overview), "groups": summary["groups"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
