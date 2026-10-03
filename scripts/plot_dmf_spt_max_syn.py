#!/usr/bin/env python3
"""Plot cached SPT node maxima by ROI order and coupling, without weighting.

The heatmap is the observed maximum across all eight seeds, not an average
of maxima. Panel b compares low- and high-order peak strengths per seed. These are selected-path
hierarchical synergies, not exhaustive coalition maxima or pure-order PID atoms.
No dynamics, estimator fits, or partition searches are run.
Absent orders are assigned zero in the heatmap at the user's request;
the source node statistics retain their occurrence information.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_dmf_spt_order_distribution import (
    INPUT, ORDERS, SEEDS, STYLE, TOLERANCE, bin_edges, load_analysis,
    setup_G_axis,
)

FIGURE = ROOT / "fig/dmf_schaefer100/dmf_spt_max_syn_by_order.png"
HEATMAP = ROOT / "fig/dmf_schaefer100/dmf_spt_max_syn_heatmap.png"
SUMMARY = ROOT / "results/dmf_schaefer100/spt_order_distribution/max_syn_summary.json"
REFERENCE = ROOT / "results/dmf_schaefer100/xi_hierarchy_tree/summary.json"
NATS_PER_BIT = np.log(2.0)
BLUE, ORANGE, PURPLE = "#206A9B", "#D56A28", "#71509A"
LOW_ORDER_CUTOFF = 10  # Existing C10 scale; never tuned to sharpen this crossover.


def max_by_order(data):
    """First maximize within each tree/order; retain absent orders as NaN."""
    maxima = np.full(data["counts"].shape, np.nan)
    for si, seed in enumerate(data["samples"]):
        for gi, row in enumerate(seed):
            for oi, values in enumerate(row):
                if len(values):
                    maxima[si, gi, oi] = np.max(values)
    if not np.array_equal(np.isfinite(maxima), data["counts"] > 0):
        raise ValueError("Missing maxima and absent orders disagree")
    if np.any(maxima[np.isfinite(maxima)] < 0):
        raise ValueError("Audited Syn values must be nonnegative")
    # All-negative-infinity columns are absent, then restored to NaN.
    envelope = np.max(np.where(np.isfinite(maxima), maxima, -np.inf), axis=0)
    envelope[~np.isfinite(envelope)] = np.nan
    winner_index = np.nanargmax(maxima, axis=2)
    peak = np.take_along_axis(maxima, winner_index[:, :, None], axis=2)[:, :, 0]
    orders = ORDERS[winner_index]
    winning_seed_index = np.argmax(peak, axis=0)
    pooled_orders = orders[winning_seed_index, np.arange(peak.shape[1])]
    pooled_peak = peak[winning_seed_index, np.arange(peak.shape[1])]
    np.testing.assert_allclose(pooled_peak, np.nanmax(envelope, axis=1), rtol=0, atol=0)
    # Ties are explicit; display picks the smallest order / smallest seed.
    tie_counts = np.sum(maxima == peak[:, :, None], axis=2)
    return dict(maxima=maxima, envelope=envelope, peak=peak, orders=orders,
                pooled_orders=pooled_orders, pooled_peak=pooled_peak,
                pooled_seed=SEEDS[winning_seed_index], winning_order_tie_counts=tie_counts)


def low_order_dominance(maxima, cutoff=LOW_ORDER_CUTOFF):
    """D=(L-H)/(L+H), comparing maxima at orders <=cutoff and >cutoff.

    D is a dimensionless signed comparison, not a Syn estimate. Both order
    groups are required; denominators <=2*tolerance stay undefined, not clipped.
    """
    if cutoff < 2 or cutoff >= 100:
        raise ValueError("Low-order cutoff must be between 2 and 99")
    values = np.asarray(maxima, dtype=float)
    if values.ndim != 3 or values.shape[-1] != len(ORDERS):
        raise ValueError("Expected seed x G x 99-order maxima in bits")
    if np.any(values[np.isfinite(values)] < 0) or np.isinf(values).any():
        raise ValueError("Dominance requires audited nonnegative finite Syn or missing orders")
    lo_values, hi_values = values[:, :, ORDERS <= cutoff], values[:, :, ORDERS > cutoff]
    if np.any(~np.isfinite(lo_values).any(axis=2)) or np.any(~np.isfinite(hi_values).any(axis=2)):
        raise ValueError("Both low- and high-order groups must have selected nodes")
    low, high = np.nanmax(lo_values, axis=2), np.nanmax(hi_values, axis=2)
    total = low + high
    dominance = np.divide(low - high, total, out=np.full_like(total, np.nan), where=total > 2 * TOLERANCE)
    return dict(low=low, high=high, dominance=dominance, cutoff=cutoff,
                mean=dominance.mean(axis=0), sd=dominance.std(axis=0, ddof=1),
                positive_seed_count=np.sum(dominance > 0, axis=0))


def load_reference(path):
    """Audit the Fig. 1b tree; its leaves include within-ROI E/I increments."""
    raw = path.read_bytes()
    record = json.loads(raw)
    if record["syn_nonnegative_tolerance_bits"] != TOLERANCE:
        raise ValueError("Reference tree uses a different Syn tolerance")
    nodes, leaves = [], []

    def visit(node):
        indices = set(node["indices"])
        if len(indices) != len(node["indices"]):
            raise ValueError("Duplicate reference ROI indices")
        values = [node["xi_bits"], node["syn_bits_raw"]]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite reference information")
        children = node["children"]
        if not children:
            if len(indices) != 1:
                raise ValueError("Reference leaves must each be one ROI")
            leaves.append(node)
            return
        if len(children) != 2:
            raise ValueError("Reference tree must be binary")
        left, right = (set(c["indices"]) for c in children)
        if left & right or left | right != indices:
            raise ValueError("Reference children do not partition their parent")
        residual = node["xi_bits"] - sum(c["xi_bits"] for c in children)
        if abs(residual - node["syn_bits_raw"]) > TOLERANCE:
            raise ValueError("Reference node does not close")
        nodes.append(node)
        for child in children:
            visit(child)

    visit(record["tree"])
    if len(nodes) != 99 or set(record["tree"]["indices"]) != set(range(100)):
        raise ValueError("Expected a complete 100-ROI reference tree")
    values = np.asarray([n["syn_bits_raw"] for n in nodes])
    bad = values < -TOLERANCE
    if bad.any():
        raise ValueError(f"Reference Syn violation: minimum={values.min():.12g}, "
                         f"threshold={-TOLERANCE:.12g}, affected_count={int(bad.sum())}")
    zero_count = int(np.sum(values < 0))
    adjusted = values.copy()
    adjusted[adjusted < 0] = 0.0  # Only audited tolerance-scale negatives.
    profile = np.full(len(ORDERS), np.nan)
    for node, value in zip(nodes, adjusted, strict=True):
        oi = len(node["indices"]) - 2
        profile[oi] = value if np.isnan(profile[oi]) else max(profile[oi], value)
    winner = int(np.nanargmax(profile))
    closure = values.sum() + sum(n["xi_bits"] for n in leaves) - record["tree"]["xi_bits"]
    if abs(closure) > TOLERANCE:
        raise ValueError("Reference tree budget fails to close")
    return dict(profile=profile, G=float(record["coupling_g"]), seed=int(record["seed"]),
                winner_order=int(ORDERS[winner]), winner_syn_bits=float(profile[winner]),
                max_order2_bits=float(profile[0]), exact_max_size=record["exact_search_max_coalition_size"],
                audit=dict(selected_node_count=len(nodes), minimum_raw_syn_bits=float(values.min()),
                           syn_tolerance_bits=TOLERANCE, tolerance_zero_count=zero_count,
                           significant_negative_count=0, closure_error_bits=float(closure)),
                fingerprint=hashlib.sha256(raw).hexdigest())


def panel_title(ax, text, pad=31):
    ax.set_title(text, loc="left", fontweight="bold", fontsize=10, pad=pad)


def outside_key(ax, **kwargs):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), borderaxespad=0,
              frameon=False, fontsize=8, **kwargs)


def draw_order_heatmap(fig, ax, data, statistics):
    """Assign absent order/G cells to zero, sharing the original linear scale."""
    envelope = statistics["envelope"]
    if np.isinf(envelope).any():
        raise ValueError("Infinite heatmap Syn values")
    values = np.where(np.isnan(envelope), 0.0, envelope) * NATS_PER_BIT
    heat = ax.pcolormesh(bin_edges(data["G"]), np.arange(1.5, 101.5), values.T,
                         cmap="viridis", vmin=0,
                         vmax=float(statistics["peak"].max() * NATS_PER_BIT),
                         shading="flat", rasterized=True)
    setup_G_axis(ax, data["G"])
    ax.set(ylim=(1.5, 100.5), ylabel="Synergy order (ROI count)", yticks=[2, 20, 40, 60, 80, 100])
    panel_title(ax, "a  Maximum node Syn at each order", pad=23)
    ax.text(0, 1.025, "Maximum across 8 seeds; absolute values; absent order = 0",
            transform=ax.transAxes, fontsize=8, color="#444444")
    colorbar = fig.colorbar(heat, ax=ax, fraction=.035, pad=.02)
    colorbar.set_label("Maximum node Syn (nats)")


def plot_heatmap(data, statistics, output):
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
        draw_order_heatmap(fig, ax, data, statistics)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=320, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def plot(data, statistics, reference, output):
    gs = data["G"]
    si = int(np.flatnonzero(SEEDS == reference["seed"])[0])
    gi = int(np.flatnonzero(np.isclose(gs, reference["G"]))[0])
    nats = statistics["maxima"] * NATS_PER_BIT
    peak = statistics["peak"] * NATS_PER_BIT
    with mpl.rc_context({**STYLE, "axes.spines.top": False, "axes.spines.right": False}):
        fig = plt.figure(figsize=(12.8, 7.6), layout="constrained")
        grid = fig.add_gridspec(2, 2, width_ratios=(1.8, 1), height_ratios=(1.2, 1))
        ax = fig.add_subplot(grid[0, 0])
        draw_order_heatmap(fig, ax, data, statistics)

        ax = fig.add_subplot(grid[1, 0])
        dominance = low_order_dominance(statistics["maxima"])
        ax.axhline(0, color="#666666", lw=.8, linestyle="--")
        ax.axvspan(.5, .6, color="#D5D9DE", alpha=.3, linewidth=0)
        for seed_index, values in enumerate(dominance["dominance"]):
            ax.plot(gs, values, color="#A1A7AD", lw=.7, alpha=.65,
                    label="Individual seeds" if seed_index == 0 else None)
        mean, sd = dominance["mean"], dominance["sd"]
        ax.fill_between(gs, mean - sd, mean + sd, color=BLUE, alpha=.15, linewidth=0)
        ax.plot(gs, mean, color=BLUE, lw=1.6, label="8-seed mean ± SD")
        setup_G_axis(ax, gs)
        ax.set(ylim=(-1, 1), ylabel=r"Low-order dominance, $D_{10}$", yticks=[-1, -.5, 0, .5, 1])
        ax.grid(axis="y", color="#E8EAED", lw=.5)
        panel_title(ax, "b  Low-order dominance: $(L-H)/(L+H)$")
        outside_key(ax, ncol=2, columnspacing=1.25)

        ax = fig.add_subplot(grid[0, 1])
        old = reference["profile"] * NATS_PER_BIT
        new = nats[si, gi]
        ax.scatter(ORDERS, old, color=PURPLE, s=17, alpha=.8, label="Fig. 1b tree")
        ax.scatter(ORDERS, new, color=BLUE, marker="s", s=13, alpha=.8, label="Paired-scan tree")
        old_oi, new_oi = int(np.nanargmax(old)), int(np.nanargmax(new))
        ax.scatter([ORDERS[old_oi], ORDERS[new_oi]], [old[old_oi], new[new_oi]],
                   marker="*", s=105, c=[PURPLE, BLUE], edgecolors="white", linewidths=.5, zorder=5)
        ax.set(xlim=(-1, 103), ylim=(0, .51), xlabel="Synergy order (ROI count)",
               ylabel="Maximum node Syn (nats)", xticks=[2, 20, 40, 60, 80, 100])
        panel_title(ax, f"c  Two caches: $G={reference['G']:g}$, seed {reference['seed']}")
        outside_key(ax, ncol=2, columnspacing=1)
        # Exact maxima above the data; annotations do not conceal either profile.
        ax.text(.02, .97, f"Fig. 1b: {ORDERS[old_oi]} ROI, {old[old_oi]:.3f} nats\n"
                f"Scan: {ORDERS[new_oi]} ROI, {new[new_oi]:.3f} nats", transform=ax.transAxes,
                va="top", fontsize=8, color="#333333")
        ax.grid(axis="y", color="#E8EAED", lw=.5)

        ax = fig.add_subplot(grid[1, 1])
        for values, color, label in ((peak, BLUE, "Maximum over all orders"),
                                     (nats[:, :, 0], ORANGE, "Maximum at order 2")):
            if not np.isfinite(values).all():
                raise ValueError("Magnitude comparison requires all 8 seeds")
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            ax.fill_between(gs, mean - sd, mean + sd, color=color, alpha=.13, linewidth=0)
            ax.plot(gs, mean, color=color, lw=1.5, label=label)
        setup_G_axis(ax, gs)
        ax.set(ylim=(0, .325), ylabel="Maximum node Syn (nats)")
        panel_title(ax, "d  Strength of the maximum (mean ± SD)")
        outside_key(ax, ncol=1, labelspacing=.25)
        ax.grid(axis="y", color="#E8EAED", lw=.5)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=320, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    parser.add_argument("--heatmap", type=Path, default=HEATMAP)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()
    data = load_analysis(args.input_dir)
    statistics = max_by_order(data)
    reference = load_reference(args.reference)
    dominance = low_order_dominance(statistics["maxima"])
    plot(data, statistics, reference, args.figure)
    plot_heatmap(data, statistics, args.heatmap)
    peak, orders = statistics["peak"], statistics["orders"]
    mean = peak.mean(axis=0)
    si, gi = np.unravel_index(np.argmax(peak), peak.shape)
    reference_si = int(np.flatnonzero(SEEDS == reference["seed"])[0])
    reference_gi = int(np.flatnonzero(np.isclose(data["G"], reference["G"]))[0])
    payload = dict(
        figure=str(args.figure.relative_to(ROOT)) if args.figure.is_relative_to(ROOT) else str(args.figure),
        unit="nats", order_definition="Number of ROI source blocks; each block contains its E/I pair",
        within_tree_definition="Maximum selected local split Syn at each order; absent orders remain missing",
        heatmap_definition="Maximum over all selected nodes of an order in all 8 seeds; absent orders assigned zero; no weights or normalization",
        heatmap_absent_order_value_nats=0.0,
        heatmap_figure=str(args.heatmap.relative_to(ROOT)) if args.heatmap.is_relative_to(ROOT) else str(args.heatmap),
        winner_definition="Maximize within each tree before reporting its winning order; never maximize a seed-averaged profile",
        magnitude_definition="Equal-seed mean and sample SD of each tree's maximum; SD is not a confidence interval",
        tie_rule="Exact equality; smallest order, then smallest seed for displayed winner",
        limitations="Selected-path maxima only; different G may select different coalitions. Reference and scan caches differ in both covariance samples and partition-search contracts.",
        seeds=SEEDS.tolist(), G=data["G"].tolist(), orders=ORDERS.tolist(),
        per_seed_winner_order=orders.tolist(), per_seed_max_syn_nats=(peak * NATS_PER_BIT).tolist(),
        low_order_cutoff=LOW_ORDER_CUTOFF,
        dominance_definition="D10=(L-H)/(L+H); L=max selected Syn at orders 2..10, H=max at orders 11..100; signed dimensionless comparison, not Syn",
        per_seed_low_order_dominance=dominance["dominance"].tolist(),
        mean_low_order_dominance=dominance["mean"].tolist(), sd_low_order_dominance=dominance["sd"].tolist(),
        low_order_dominant_seed_count=dominance["positive_seed_count"].tolist(),
        per_seed_max_order2_syn_nats=(statistics["maxima"][:, :, 0] * NATS_PER_BIT).tolist(),
        pooled_winner_order=statistics["pooled_orders"].tolist(), pooled_winner_seed=statistics["pooled_seed"].tolist(),
        pooled_max_syn_nats=(statistics["pooled_peak"] * NATS_PER_BIT).tolist(),
        mean_max_syn_nats=(mean * NATS_PER_BIT).tolist(), sd_max_syn_nats=(peak.std(axis=0, ddof=1) * NATS_PER_BIT).tolist(),
        mean_max_order2_syn_nats=(statistics["maxima"][:, :, 0].mean(axis=0) * NATS_PER_BIT).tolist(),
        pooled_heatmap_absent_cells=int(np.sum(~np.isfinite(statistics["envelope"]))),
        per_seed_winning_order_tie_counts=statistics["winning_order_tie_counts"].tolist(),
        second_order_winner_conditions=int(np.sum(orders == 2)), higher_order_winner_conditions=int(np.sum(orders > 2)),
        scan_largest=dict(G=float(data["G"][gi]), seed=int(SEEDS[si]), order=int(orders[si, gi]),
                          max_syn_nats=float(peak[si, gi] * NATS_PER_BIT)),
        mean_maximum_peak=dict(G=float(data["G"][np.argmax(mean)]), max_syn_nats=float(mean.max() * NATS_PER_BIT)),
        reference_comparison=dict(G=reference["G"], seed=reference["seed"],
            figure_1b_order=reference["winner_order"], figure_1b_max_syn_nats=reference["winner_syn_bits"] * NATS_PER_BIT,
            figure_1b_max_order2_syn_nats=reference["max_order2_bits"] * NATS_PER_BIT,
            paired_scan_order=int(orders[reference_si, reference_gi]),
            paired_scan_max_syn_nats=float(peak[reference_si, reference_gi] * NATS_PER_BIT)),
        syn_tolerance_nats=TOLERANCE * NATS_PER_BIT,
        simulation=data["simulation"], search=data["search"], audit=data["audit"], reference_audit=reference["audit"],
        input_tree_fingerprint_sha256=data["fingerprint"], reference_fingerprint_sha256=reference["fingerprint"],
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:payload[k] for k in ("scan_largest", "mean_maximum_peak", "reference_comparison",
                                           "second_order_winner_conditions", "higher_order_winner_conditions")}, indent=2))
    print(f"Figure: {args.figure}")


if __name__ == "__main__":
    main()
