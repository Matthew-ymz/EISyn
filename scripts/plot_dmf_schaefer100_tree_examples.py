#!/usr/bin/env python3
"""Compare a small, declared set of DMF trees using the main figure's search.

Only cached Gaussian conditional covariances are read; no trajectories or EI
models are refitted. Near-peak mode reuses the original G=1.3 / seed=4 tree;
wide mode uses the existing paired-input covariance caches throughout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_dmf_schaefer100_xi_hierarchy_tree import (
    ConditionalBlockXiOracle,
    DEFAULT_INPUT,
    INK,
    SPLIT_COLOR,
    SYN_NONNEGATIVE_TOLERANCE_BITS as TOLERANCE,
    _blend_with_white,
    _tree_metrics,
    build_scalable_hierarchy,
    flatten_nodes,
    pairwise_syn_affinity,
    render_tree,
)
from scripts.analyze_runge_slp_pc60_xi_hierarchy import _node_record
from scripts.analyze_dmf_critical_phi_hierarchy_topology import logdet_minor, safe_logdet_psd
from scripts.compare_runge_slp_pc60_xi_horizons import _node_from_record

DEFAULT_OUTPUT = ROOT / "results/dmf_schaefer100/xi_hierarchy_tree/examples"
DEFAULT_FIGURES = ROOT / "fig/dmf_schaefer100/tree_examples"
# Match the current Figure 1 Yeo palette, including its red VAN / orange FPN.
NETWORK_COLORS = ("#6A3D9A", "#1F78B4", "#33A02C", "#E31A1C",
                  "#B15928", "#FF7F00", "#66A61E")


class CachedScalarLogdetXiOracle(ConditionalBlockXiOracle):
    """Same formula and summation order; compute each scalar minor only once."""

    def __init__(self, conditional, blocks):
        super().__init__(conditional, blocks)
        self.scalar_logdets = [logdet_minor(self.conditional, (i,))
                              for i in range(self.conditional.shape[0])]
        legacy = ConditionalBlockXiOracle(conditional, blocks)
        # Verify the optimization across small and large source coalitions.
        errors = [abs(self.xi(range(size)) - legacy.xi(range(size)))
                  for size in (2, 8, 10, 25, 60, len(blocks))]
        self.legacy_maximum_error_bits = max(errors)
        if self.legacy_maximum_error_bits > 1.0e-12:
            raise RuntimeError(f"Cached scalar determinant differs from the original oracle: {max(errors)} bits")

    def xi(self, indices):
        key = tuple(sorted(map(int, indices)))
        if not key:
            return 0.0
        if key in self._cache:
            return self._cache[key]
        sources = [source for index in key for source in self.blocks[index]]
        local = self.conditional[np.ix_(sources, sources)]
        value = 0.5 * (sum(self.scalar_logdets[source] for source in sources)
                       - safe_logdet_psd(local)) / math.log(2.0)
        self._cache[key] = float(value)
        self.evaluations += 1
        return float(value)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_tree(tree, expected_xi: float) -> dict:
    nodes = flatten_nodes(tree)
    internal = [node for node in nodes if node.children]
    leaves = [node for node in nodes if not node.children]
    values = np.asarray([node.syn_bits for node in internal] + [node.xi_bits for node in leaves])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite selected Syn / leaf Xi")
    violations = values < -TOLERANCE
    if violations.any():
        raise RuntimeError(f"Syn nonnegativity violation: minimum={values.min():.12g} bits, "
                           f"threshold={-TOLERANCE:.12g} bits, affected_count={violations.sum()}")
    if sorted(node.indices[0] for node in leaves) != list(range(tree.size)):
        raise ValueError("Tree does not contain each ROI leaf exactly once")
    if len(internal) != tree.size - 1:
        raise ValueError("Tree is not a complete binary ROI hierarchy")
    local_errors = []
    for node in internal:
        if len(node.children) != 2:
            raise ValueError("Nonbinary split")
        left, right = (set(child.indices) for child in node.children)
        if left & right or left | right != set(node.indices):
            raise ValueError("Children do not partition their parent")
        local_errors.append(node.syn_bits + sum(child.xi_bits for child in node.children) - node.xi_bits)
    closure = float(values.sum() - tree.xi_bits)
    root_error = float(tree.xi_bits - expected_xi)
    maximum_error = max(abs(closure), abs(root_error), max(map(abs, local_errors)))
    if maximum_error > TOLERANCE:
        raise RuntimeError(f"Xi closure / input consistency failed: maximum error={maximum_error:.12g} bits")
    return {
        "syn_nonnegative_tolerance_bits": TOLERANCE,
        "selected_internal_node_count": len(internal),
        "leaf_count": len(leaves),
        "minimum_selected_syn_bits": float(min(node.syn_bits for node in internal)),
        "minimum_leaf_xi_bits": float(min(node.xi_bits for node in leaves)),
        "tolerance_negative_count": int(((values < 0.0) & ~violations).sum()),
        "significant_nonnegativity_violation_count": 0,
        "closure_error_bits": closure,
        "root_input_error_bits": root_error,
        "maximum_local_closure_error_bits": float(max(map(abs, local_errors))),
    }


def nontrivial_clades(tree) -> set[tuple[int, ...]]:
    """Exclude the identical root and singleton leaves, and ignore leaf order."""
    return {tuple(sorted(node.indices)) for node in flatten_nodes(tree)
            if node.children and node is not tree}


def lca_sizes(tree) -> np.ndarray:
    """For each fixed ROI pair, size of its smallest shared subtree."""
    matrix = np.zeros((tree.size, tree.size), dtype=int)
    for node in flatten_nodes(tree):
        if node.children:
            left, right = node.children
            matrix[np.ix_(left.indices, right.indices)] = node.size
            matrix[np.ix_(right.indices, left.indices)] = node.size
    return matrix[np.triu_indices(tree.size, k=1)]


def comparison(first, second) -> dict:
    left, right = nontrivial_clades(first), nontrivial_clades(second)
    first_metrics, second_metrics = _tree_metrics(first), _tree_metrics(second)
    return {
        "shared_nontrivial_clade_count": len(left & right),
        "nontrivial_clades_per_tree": len(left),
        "shared_nontrivial_clade_fraction": len(left & right) / len(left),
        "roi_pair_lca_size_spearman_rho": float(spearmanr(lca_sizes(first), lca_sizes(second)).statistic),
        "absolute_maximum_depth_difference": abs(first_metrics["maximum_depth"] - second_metrics["maximum_depth"]),
        "absolute_colless_difference": abs(first_metrics["normalized_colless_imbalance"] - second_metrics["normalized_colless_imbalance"]),
        "absolute_overall_xi_difference_nats": abs(first.xi_bits - second.xi_bits) * math.log(2),
    }


def summarize_difference_scales(pairs, couplings) -> list[dict]:
    """Descriptive means/ranges; overlapping tree pairs are not replicates."""
    groups = [(f"seed changes at G={g}", [p for p in pairs
               if p["comparison_type"] == "same_G_change_seed" and p["first"][1] == g])
              for g in couplings]
    for i, first_g in enumerate(couplings):
        for second_g in couplings[i + 1:]:
            groups.append((f"G={first_g} vs {second_g}", [p for p in pairs
                if p["comparison_type"] == "same_seed_change_G"
                and sorted([p["first"][1], p["second"][1]]) == [first_g, second_g]]))
    summaries = []
    for name, group in groups:
        row = {"comparison": name, "pair_count": len(group)}
        for field in ("absolute_maximum_depth_difference", "absolute_colless_difference",
                      "absolute_overall_xi_difference_nats", "shared_nontrivial_clade_count",
                      "roi_pair_lca_size_spearman_rho"):
            values = [p[field] for p in group]
            row[field] = {"mean": float(np.mean(values)), "min": float(min(values)), "max": float(max(values))}
        summaries.append(row)
    return summaries


def load_or_build(conditional, *, seed, coupling_g, expected_xi, config, output_dir, reference,
                  use_original_reference=True):
    path = output_dir / f"seed{seed:02d}_G{coupling_g:.2f}.json"
    reused = False
    if use_original_reference and seed == 4 and np.isclose(coupling_g, 1.3):
        payload = reference
        tree = _node_from_record(payload["tree"])
        reused = True
    elif path.exists() and (payload := json.loads(path.read_text()))["config"] == config:
        tree = _node_from_record(payload["tree"])
        reused = True
    else:
        started = time.monotonic()
        roi_count = conditional.shape[0] // 2
        oracle = CachedScalarLogdetXiOracle(conditional, [(i, i + roi_count) for i in range(roi_count)])
        affinity, pair_zero_count = pairwise_syn_affinity(oracle, roi_count, tolerance=TOLERANCE)
        search_audit = {"candidate_count": 0, "tolerance_zero_count": 0}
        tree = build_scalable_hierarchy(tuple(range(roi_count)), oracle, affinity,
                                       exact_max_size=8, tolerance=TOLERANCE, audit=search_audit)
        payload = {
            "config": config, "seed": seed, "coupling_g": coupling_g,
            "elapsed_seconds": time.monotonic() - started,
            "candidate_split_count": search_audit["candidate_count"],
            "coalition_evaluation_count": oracle.evaluations,
            "pair_tolerance_zero_count": pair_zero_count,
            "split_tolerance_zero_count": search_audit["tolerance_zero_count"],
            "scalar_logdet_cache_legacy_maximum_error_bits": oracle.legacy_maximum_error_bits,
            "tree": _node_record(tree),
        }
        payload["validation"] = audit_tree(tree, expected_xi)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    validation = audit_tree(tree, expected_xi)
    internal = [node for node in flatten_nodes(tree) if node.children]
    strongest = max(internal, key=lambda node: node.syn_bits)
    row = {
        "seed": seed, "coupling_g": coupling_g,
        "overall_xi_bits": tree.xi_bits, "overall_xi_nats": tree.xi_bits * math.log(2),
        "tree_metrics": _tree_metrics(tree), "validation": validation,
        "internal_syn_share_percent": 100 * sum(node.syn_bits for node in internal) / tree.xi_bits,
        "maximum_local_syn_nats": strongest.syn_bits * math.log(2),
        "maximum_local_syn_share_percent": 100 * strongest.syn_bits / tree.xi_bits,
        "maximum_local_syn_roi_count": strongest.size,
        "candidate_split_count": payload["candidate_split_count"],
        "coalition_evaluation_count": payload["coalition_evaluation_count"],
        "pair_tolerance_zero_count": payload["pair_tolerance_zero_count"],
        "split_tolerance_zero_count": payload["split_tolerance_zero_count"],
        "cache": str(ROOT / "results/dmf_schaefer100/xi_hierarchy_tree/summary.json"
                     if use_original_reference and seed == 4 and np.isclose(coupling_g, 1.3) else path),
    }
    print(f"[{'cached' if reused else 'built'}] G={coupling_g:g}, seed={seed}, "
          f"depth={row['tree_metrics']['maximum_depth']}", flush=True)
    return tree, row


def plot_grid(trees, rows, *, seeds, couplings, labels, membership, network_names, color_max, output, dpi):
    figure, axes = plt.subplots(len(seeds), len(couplings), figsize=(23, 15), squeeze=False)
    # A reserved top strip keeps the shared legend and scale outside every tree.
    figure.subplots_adjust(left=0.025, right=0.98, bottom=0.05, top=0.89, wspace=0.07, hspace=0.21)
    for index, (tree, row) in enumerate(zip(trees, rows, strict=True)):
        axis = axes[index // len(couplings), index % len(couplings)]
        render_tree(tree, None, labels=labels, network_membership=membership,
                    network_names=network_names, seed=row["seed"], coupling_g=row["coupling_g"],
                    dpi=dpi, axis=axis, network_colors=NETWORK_COLORS, information_unit="nats",
                    node_value_mode="root_share", syn_color_max=color_max, show_colorbar=False,
                    show_roi_labels=False, show_network_strip_label=False, node_label_limit=7)
        axis.set_ylim(-0.07, 1.06)
        metrics = row["tree_metrics"]
        axis.text(0.0, 1.025, chr(97 + index), transform=axis.transAxes,
                  fontsize=10, fontweight="bold", color=INK)
        axis.text(0.5, -0.025,
                  f"Depth {metrics['maximum_depth']}  |  Spine {metrics['dominant_spine_fraction']:.1%}"
                  f"  |  Colless {metrics['normalized_colless_imbalance']:.3f}",
                  transform=axis.transAxes, ha="center", va="top", fontsize=8, color=INK)
    handles = [Patch(facecolor=color, edgecolor="none", label=name)
               for color, name in zip(NETWORK_COLORS, network_names, strict=True)]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.10, 0.982),
                  ncol=4, frameon=False, fontsize=9, handlelength=1, columnspacing=1.4)
    color_axis = figure.add_axes([0.785, 0.948, 0.18, 0.015])
    cmap = LinearSegmentedColormap.from_list("shared_local_syn",
           [_blend_with_white(SPLIT_COLOR, 0.12), _blend_with_white(SPLIT_COLOR, 0.74)])
    colorbar = figure.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0, color_max), cmap=cmap),
                              cax=color_axis, orientation="horizontal", ticks=[0, color_max])
    colorbar.set_label(r"Local Syn / overall $\Xi$ (%)", fontsize=9, labelpad=4)
    color_axis.xaxis.set_label_position("top")
    color_axis.tick_params(labelsize=8, length=2)
    colorbar.outline.set_linewidth(0.5)
    figure.text(0.025, 0.017,
                f"Rows: seeds {', '.join(map(str, seeds))}. Columns: G = {', '.join(f'{g:g}' for g in couplings)}.  "
                "Height: log ROI count. Yeo-7 colors: post hoc. Individual trees retain ROI labels.",
                fontsize=9, color=INK)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--wide", action="store_true",
                        help="Compare G=0,1.3,3 with consistently paired source/noise caches")
    args = parser.parse_args()
    if args.wide:
        args.output_dir = args.output_dir / "wide"
        args.figure_dir = args.figure_dir / "wide"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    seeds, couplings = (3, 4, 5), ((0.0, 1.3, 3.0) if args.wide else (1.2, 1.3, 1.4))
    config = {
        "input_sha256": sha256(args.input), "exact_max_size": 8,
        "search": "same spectral candidates as Figure 1b; no singleton supplementation or random candidates",
        "tolerance_bits": TOLERANCE,
        "search_source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
            (ROOT / "scripts/spt.py", ROOT / "scripts/analyze_runge_slp_pc60_xi_hierarchy.py",
             ROOT / "scripts/analyze_dmf_schaefer100_xi_hierarchy_tree.py",
             ROOT / "scripts/analyze_dmf_critical_phi_hierarchy_topology.py")},
    }
    reference_path = ROOT / "results/dmf_schaefer100/xi_hierarchy_tree/summary.json"
    reference = json.loads(reference_path.read_text())
    if reference["seed"] != 4 or not np.isclose(reference["coupling_g"], 1.3) or reference["exact_search_max_coalition_size"] != 8:
        raise ValueError("Original reference condition / search differs from Figure 1b")
    trees, rows = [], []
    with np.load(args.input, allow_pickle=True) as archive:
        labels = list(map(str, archive["region_labels"]))
        network_names = list(map(str, archive["network_names"]))
        membership = np.asarray(archive["network_membership"], dtype=int)
        conditions = []
        for seed in seeds:
            seed_index = int(np.flatnonzero(archive["seeds"] == seed)[0])
            for coupling_g in couplings:
                if args.wide:
                    shard = "seeds3_4" if seed in (3, 4) else "seeds5_6"
                    path = ROOT / f"results/dmf_schaefer100/unconstrained_spt_wide/shards/{shard}/covariance/seed{seed:02d}_G{coupling_g:.2f}.npz"
                    with np.load(path) as cache:
                        simulation_config = json.loads(str(cache["config_json"].item()))
                        if int(cache["seed"].item()) != seed or not np.isclose(cache["G"].item(), coupling_g):
                            raise ValueError(f"Wrong covariance condition in {path}")
                        if not simulation_config["paired_inputs_across_G"] or not simulation_config["paired_noise_across_G"]:
                            raise ValueError("Wide comparison requires paired intervention and noise")
                        conditional = cache["conditional_covariance"].copy()
                        input_hash = str(cache["input_sha256"].item())
                    expected_xi = ConditionalBlockXiOracle(conditional, [(i, i + 100) for i in range(100)]).xi(range(100))
                    conditions.append((seed, coupling_g, conditional, expected_xi,
                                       {"path": str(path), "sha256": sha256(path),
                                        "simulation_config": simulation_config, "paired_input_sha256": input_hash}))
                else:
                    g_index = int(np.flatnonzero(np.isclose(archive["G"], coupling_g))[0])
                    conditions.append((seed, coupling_g, archive["conditional_covariance"][seed_index, g_index],
                                       float(archive["fine_phi"][seed_index, g_index]), None))
    if args.wide:
        configs = [condition[4]["simulation_config"] for condition in conditions]
        if any(value != configs[0] for value in configs):
            raise ValueError("Wide conditions have different simulation protocols")
        for seed in seeds:
            hashes = {condition[4]["paired_input_sha256"] for condition in conditions if condition[0] == seed}
            if len(hashes) != 1:
                raise ValueError("Source samples differ across G within one seed")
        config["paired_covariance_inputs"] = [condition[4] for condition in conditions]
    for seed, coupling_g, conditional, expected_xi, provenance in conditions:
        tree, row = load_or_build(conditional, seed=seed, coupling_g=coupling_g, expected_xi=expected_xi,
                                 config=config, output_dir=args.output_dir, reference=reference,
                                 use_original_reference=not args.wide)
        if provenance:
            row["covariance_provenance"] = provenance
        trees.append(tree)
        rows.append(row)
    baseline = trees[4]
    for tree, row in zip(trees, rows, strict=True):
        row["comparison_to_G130_seed04"] = comparison(tree, baseline)
    color_max = math.ceil(max(row["maximum_local_syn_share_percent"] for row in rows) * 2) / 2
    pairs = []
    for i, first in enumerate(rows):
        for j in range(i + 1, len(rows)):
            second = rows[j]
            if first["seed"] == second["seed"] or first["coupling_g"] == second["coupling_g"]:
                pairs.append({"first": [first["seed"], first["coupling_g"]],
                              "second": [second["seed"], second["coupling_g"]],
                              "comparison_type": "same_seed_change_G" if first["seed"] == second["seed"] else "same_G_change_seed",
                              **comparison(trees[i], trees[j])})
    g_tag = "_".join(f"{round(g * 100):03d}" for g in couplings)
    overview = args.figure_dir / f"tree_examples_G{g_tag}_seeds03_04_05.png"
    for tree, row in zip(trees, rows, strict=True):
        path = args.figure_dir / f"tree_G{round(row['coupling_g'] * 100):03d}_seed{row['seed']:02d}.png"
        render_tree(tree, path, labels=labels, network_membership=membership, network_names=network_names,
                    seed=row["seed"], coupling_g=row["coupling_g"], dpi=args.dpi,
                    network_colors=NETWORK_COLORS, information_unit="nats", node_value_mode="root_share",
                    syn_color_max=color_max)
        row["figure"] = str(path)
    plot_grid(trees, rows, seeds=seeds, couplings=couplings, labels=labels, membership=membership,
              network_names=network_names, color_max=color_max, output=overview, dpi=args.dpi)
    summary = {
        "selection": f"Declared small grid: G={couplings} and seeds={seeds}; not selected by topology",
        "input": str(args.input), "config": config,
        "original_reference_sha256": sha256(reference_path),
        "estimator": "Existing high-dimensional Gaussian conditional-covariance approximation; no new EI fit or dynamics",
        "source_blocks": "100 E/I-paired ROI blocks; leaf Xi includes within-ROI E/I increments",
        "normalization": "Each local split Syn / same-condition overall scalar-source Xi * 100",
        "shared_color_range_percent": [0.0, color_max],
        "clade_comparison": "Exact ROI membership of the 98 internal nodes excluding root; insensitive to leaf ordering",
        "lca_comparison": "Spearman correlation of smallest shared subtree ROI counts for the same 4,950 ROI pairs; descriptive, no p-value",
        "limitations": "Three G and three seeds at fixed SC, horizon, sample size, estimator and approximate search; not a full stability study",
        "sampling": "Paired across G; G=1.3 is rebuilt from paired draws and differs from the original Figure 1b samples" if args.wide else "Original independently drawn seed-G conditions",
        "figure": str(overview), "rows": rows, "pairwise_comparisons": pairs,
        "difference_scale_summary": summarize_difference_scales(pairs, couplings),
        "difference_scale_interpretation": "Descriptive averages of overlapping tree comparisons; pairs are not independent statistical replicates",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {overview}", flush=True)


if __name__ == "__main__":
    main()
