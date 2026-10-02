#!/usr/bin/env python3
"""Cache-only DMF SPT order means, conserved mass, and Syn distributions.

No simulations, estimator refits, or new partition searches are performed.
The 440 paired ROI-block trees supply 99 selected internal nodes each. These
are selected-path statistics, not exhaustive same-order coalition statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/dmf_schaefer100/unconstrained_spt_wide"
SUMMARY = ROOT / "results/dmf_schaefer100/spt_order_distribution/summary.json"
STATISTICS = ROOT / "results/dmf_schaefer100/spt_order_distribution/order_statistics.npz"
FIGURE = ROOT / "fig/dmf_schaefer100/dmf_spt_order_summary.png"
DISTRIBUTION_FIGURE = ROOT / "fig/dmf_schaefer100/dmf_spt_syn_distribution_by_G.png"
TOLERANCE = 1.0e-8  # Native bits; fixed before reading/aggregating selected Syn.
ORDERS = np.arange(2, 101)
SEEDS = np.arange(3, 11)
# Illustrative orders chosen by scale coverage, never by observed peaks/ranks.
DISPLAY_ORDERS = (2, 3, 5, 10, 50, 100)
GROUPS = ((2, 10), (11, 20), (21, 50), (51, 100))
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
         "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
         "xtick.labelsize": 8, "ytick.labelsize": 8, "savefig.facecolor": "white"}


def coupling_grid():
    return np.array(sorted(set(np.round(np.arange(0, 3.00001, .1), 2)) |
                           set(np.round(np.arange(1.1, 1.70001, .02), 2))))


def bin_edges(coordinates):
    """Midpoint cell boundaries retain the actual, nonuniform G coordinates."""
    x = np.asarray(coordinates, dtype=float)
    if x.ndim != 1 or len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("Expected at least two strictly increasing coordinates")
    return np.r_[x[0] - (x[1] - x[0]) / 2, (x[:-1] + x[1:]) / 2,
                 x[-1] + (x[-1] - x[-2]) / 2]


def selected_nodes(tree):
    """Validate the complete binary partition and return its internal nodes."""
    indices = tuple(tree["indices"])
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Empty/duplicate ROI membership")
    if not np.isfinite([tree["xi_bits"], tree["syn_bits_raw"]]).all():
        raise ValueError("Nonfinite tree information value")
    children = tree["children"]
    if not children:
        if len(indices) != 1 or abs(tree["xi_bits"]) > TOLERANCE:
            raise ValueError("Expected singleton ROI leaves with zero cross-ROI Xi")
        return []
    if len(children) != 2:
        raise ValueError("Expected binary partition")
    left, right = (set(c["indices"]) for c in children)
    if left & right or left | right != set(indices):
        raise ValueError("Children must be disjoint and partition their parent")
    residual = tree["xi_bits"] - sum(c["xi_bits"] for c in children)
    if abs(residual - tree["syn_bits_raw"]) > TOLERANCE:
        raise ValueError("Local split does not close to its recorded raw Syn")
    return [tree] + [n for child in children for n in selected_nodes(child)]


def aggregate_values(values_by_order, tolerance=TOLERANCE):
    """Keep absence distinct from zero and record any numerical-zero treatment."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative in bits")
    raw = np.array([v for values in values_by_order for v in values], dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite selected Syn")
    bad = raw < -tolerance
    if bad.any():
        raise ValueError(f"Syn violation: minimum={raw.min():.12g}, "
                         f"threshold={-tolerance:.12g}, affected_count={int(bad.sum())}")
    adjusted = [np.array(values, dtype=float) for values in values_by_order]
    zero_count = int(np.sum(raw < 0))
    for values in adjusted:
        values[values < 0] = 0.0  # Only tolerance-scale negatives, audited above.
    counts = np.array([len(values) for values in adjusted], dtype=int)
    mass = np.array([values.sum() for values in adjusted])
    means = np.divide(mass, counts, out=np.full_like(mass, np.nan), where=counts > 0)
    return mass, counts, means, adjusted, zero_count


def present_seed_mean(values):
    """Equal seed weight among seeds where the selected order exists."""
    count = np.isfinite(values).sum(axis=0)
    return np.divide(np.nansum(values, axis=0), count,
                     out=np.full(values.shape[1:], np.nan), where=count > 0)


def distribution_by_G(samples, edges):
    """Per-seed empirical bin probabilities, then equal mean of present seeds.

    samples is seed x G, with each entry holding same-order selected-node Syn.
    Absent orders remain NaN; no synthetic zero observations are introduced.
    """
    probability = np.full((len(edges) - 1, len(samples[0])), np.nan)
    present = np.zeros(len(samples[0]), dtype=int)
    for gi in range(len(samples[0])):
        rows = []
        for seed_samples in samples:
            values = np.asarray(seed_samples[gi])
            if not len(values):
                continue
            counts, _ = np.histogram(values, bins=edges)
            if counts.sum() != len(values):
                raise ValueError("Distribution bins exclude selected Syn observations")
            rows.append(counts / len(values))
        present[gi] = len(rows)
        if rows:
            probability[:, gi] = np.mean(rows, axis=0)
    if not np.allclose(np.nansum(probability, axis=0)[present > 0], 1, atol=1e-12):
        raise ValueError("Distribution columns do not sum to one")
    return probability, present


def load_analysis(input_dir):
    gs = coupling_grid()
    records = {}
    fingerprint = hashlib.sha256()
    paths = sorted((input_dir / "shards").glob("seeds*/trees/seed*.json"))
    if not paths:
        raise ValueError(f"No paired tree caches found in {input_dir}")
    simulation = search = None
    for path in paths:
        raw = path.read_bytes()
        record = json.loads(raw)
        key = (int(record["seed"]), float(record["G"]))
        if key in records:
            raise ValueError(f"Duplicate seed/G cache: {key}")
        records[key] = record
        fingerprint.update(str(path.relative_to(input_dir)).encode() + b"\0" + raw)
        if simulation is None:
            simulation, search = record["simulation_config"], record["search_config"]
        if record["simulation_config"] != simulation or record["search_config"] != search:
            raise ValueError("Simulation/search contract changed between cached conditions")
    expected = {(int(s), float(g)) for s in SEEDS for g in gs}
    if set(records) != expected:
        raise ValueError(f"Expected 440 paired conditions: missing={len(expected - set(records))}, "
                         f"extra={len(set(records) - expected)}")
    if search["tolerance_bits"] != TOLERANCE:
        raise ValueError("Cached tree tolerance differs from this analysis")
    if not simulation["paired_inputs_across_G"] or not simulation["paired_noise_across_G"]:
        raise ValueError("Require paired intervention samples and noise across G")

    shape = (len(SEEDS), len(gs), len(ORDERS))
    mass = np.zeros(shape)
    counts = np.zeros(shape, dtype=int)
    means = np.full(shape, np.nan)
    xi = np.empty(shape[:2])
    samples = [[[None for _ in ORDERS] for _ in gs] for _ in SEEDS]
    minimum = float("inf")
    zero_count = 0
    closure_errors = []
    raw_closure_errors = []
    search_kinds = {}
    for si, seed in enumerate(SEEDS):
        for gi, g in enumerate(gs):
            record = records[int(seed), float(g)]
            tree = record["tree"]
            if set(tree["indices"]) != set(range(100)):
                raise ValueError("Expected all 100 ROI in each root")
            nodes = selected_nodes(tree)
            if len(nodes) != 99:
                raise ValueError("Expected 99 internal nodes per complete ROI tree")
            if abs(tree["xi_bits"] - record["cross_roi_xi_bits"]) > TOLERANCE:
                raise ValueError("Root value differs from cached cross-ROI Xi")
            by_order = [[] for _ in ORDERS]
            for node in nodes:
                value = float(node["syn_bits_raw"])
                by_order[len(node["indices"]) - 2].append(value)
                minimum = min(minimum, value)
                kind = node["search_kind"]
                search_kinds[kind] = search_kinds.get(kind, 0) + 1
            m, n, mean, adjusted, zeros = aggregate_values(by_order)
            mass[si, gi], counts[si, gi], means[si, gi] = m, n, mean
            samples[si][gi] = adjusted
            xi[si, gi] = tree["xi_bits"]
            zero_count += zeros
            raw_closure_errors.append(sum(sum(v) for v in by_order) - xi[si, gi])
            closure_errors.append(m.sum() - xi[si, gi])
    max_error = max(abs(v) for v in closure_errors)
    if max_error > TOLERANCE or np.any(xi <= TOLERANCE):
        raise ValueError(f"Invalid mass/share budget: maximum closure error={max_error:.12g}")
    share = mass / xi[:, :, None]  # Normalize each tree BEFORE averaging seeds.
    if not np.allclose(share.sum(axis=2), 1.0, atol=1e-12, rtol=0):
        raise ValueError("Within-tree order shares do not sum to one")
    return dict(G=gs, mass=mass, counts=counts, node_means=means, xi=xi, share=share,
                mean_mass=mass.mean(axis=0), mass_sd=mass.std(axis=0, ddof=1),
                mean_share=share.mean(axis=0), mean_node_syn=present_seed_mean(means),
                present_seed_count=(counts > 0).sum(axis=0), samples=samples,
                audit=dict(condition_count=len(records), selected_node_count=int(counts.sum()),
                           syn_tolerance_bits=TOLERANCE, minimum_selected_raw_syn_bits=minimum,
                           tolerance_zero_count=zero_count, significant_negative_count=0,
                           maximum_raw_closure_error_bits=max(abs(v) for v in raw_closure_errors),
                           maximum_treated_closure_error_bits=max_error,
                           selected_search_kind_counts=search_kinds),
                simulation=simulation, search=search, fingerprint=fingerprint.hexdigest())


def setup_G_axis(ax, gs):
    ax.set_xlim(bin_edges(gs)[[0, -1]])
    ax.set_xticks(np.arange(0, 3.01, .5))
    ax.set_xlabel(r"Global coupling, $G$")


def order_distributions(data, n_bins):
    """Calculate distributions for ALL 99 orders, not just the six shown."""
    if n_bins < 2:
        raise ValueError("Use at least two histogram bins")
    gs = data["G"]
    probabilities = np.full((len(ORDERS), n_bins, len(gs)), np.nan)
    all_edges = np.empty((len(ORDERS), n_bins + 1))
    spread = np.full((len(gs), len(ORDERS)), np.nan)
    quantiles = np.full((len(gs), len(ORDERS), 3), np.nan)
    for oi, order in enumerate(ORDERS):
        samples = [[row[oi] for row in seed] for seed in data["samples"]]
        maximum = max(v.max() for seed in samples for v in seed if len(v))
        edges = np.linspace(0, np.nextafter(maximum, np.inf), n_bins + 1)
        probability, present = distribution_by_G(samples, edges)
        np.testing.assert_array_equal(present, data["present_seed_count"][:, oi])
        probabilities[oi], all_edges[oi] = probability, edges
        for gi in range(len(gs)):
            selected = [seed[gi] for seed in samples if len(seed[gi])]
            if not selected:
                continue
            values = np.concatenate(selected)
            weights = np.concatenate([np.full(len(v), 1 / (len(selected) * len(v)))
                                      for v in selected])
            mean = float(weights @ values)
            if not np.isclose(mean, data["mean_node_syn"][gi, oi], atol=1e-12, rtol=0):
                raise ValueError("Distribution and present-node means use different seed weights")
            spread[gi, oi] = np.sqrt(weights @ ((values - mean) ** 2))
            # Discrete empirical quantile: smallest observed value with CDF >= q.
            index = np.argsort(values, kind="stable")
            cdf = np.cumsum(weights[index])
            quantiles[gi, oi] = values[index][np.searchsorted(cdf, [.25, .5, .75])]
    return dict(probability=probabilities, edges=all_edges, weighted_sd=spread,
                weighted_quantiles=quantiles, n_bins=n_bins)


def plot_summary(data, output):
    gs = data["G"]
    panels = ((data["mean_mass"], "a  Mean order mass", "Order sum (bits)", "viridis"),
              (100 * data["mean_share"], "b  Within-tree order share", "Share (%)", "viridis"),
              (data["mean_node_syn"], "c  Mean Syn of present nodes", "Mean node Syn (bits)", "viridis"),
              (data["present_seed_count"], "d  Order occurrence", "Seeds with this order (of 8)", "Blues"))
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), layout="constrained")
        for ax, (values, title, label, cmap_name) in zip(axes.flat, panels):
            cmap = mpl.colormaps[cmap_name].copy()
            cmap.set_bad("#dedede")
            norm = (mpl.colors.BoundaryNorm(np.arange(-.5, 9.5), cmap.N)
                    if title.startswith("d") else mpl.colors.Normalize(0, float(np.nanmax(values))))
            image = ax.pcolormesh(bin_edges(gs), np.arange(1.5, 101.5),
                                  np.ma.masked_invalid(values.T), cmap=cmap, norm=norm,
                                  shading="flat", rasterized=True)
            setup_G_axis(ax, gs)
            ax.set(ylim=(1.5, 100.5), ylabel="SPT node order (ROI count)",
                   yticks=[2, 10, 20, 40, 60, 80, 100])
            ax.set_title(title, loc="left", pad=8, fontweight="bold")
            colorbar = fig.colorbar(image, ax=ax, fraction=.046, pad=.025)
            colorbar.set_label(label)
            if title.startswith("d"):
                colorbar.set_ticks([0, 2, 4, 6, 8])
        fig.suptitle("440 paired ROI-block SPTs; 8 seeds per G; linear colour scales\n"
                     "Order mass sums to cross-ROI Xi. Grey in c: no selected nodes.", fontsize=9)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=350, bbox_inches="tight")
        plt.close(fig)


def plot_distributions(data, distributions, output):
    n_bins = distributions["n_bins"]
    gs = data["G"]
    bin_records = {}
    with mpl.rc_context(STYLE):
        # An explicit colourbar column preserves alignment of heatmaps/strips.
        fig = plt.figure(figsize=(11.2, 7.1))
        grid = fig.add_gridspec(2, 4, width_ratios=(1, 1, 1, .045),
                               left=.075, right=.94, bottom=.10, top=.865,
                               hspace=.43, wspace=.35)
        for i, order in enumerate(DISPLAY_ORDERS):
            subgrid = grid[i // 3, i % 3].subgridspec(2, 1, height_ratios=(12, 1), hspace=.03)
            ax = fig.add_subplot(subgrid[0])
            availability_ax = fig.add_subplot(subgrid[1], sharex=ax)
            oi = order - 2
            edges = distributions["edges"][oi]
            probability = distributions["probability"][oi]
            present = data["present_seed_count"][:, oi]
            cmap = mpl.colormaps["magma"].copy()
            cmap.set_bad("#dedede")
            image = ax.pcolormesh(bin_edges(gs), edges, np.ma.masked_invalid(probability),
                                  cmap=cmap, vmin=0, vmax=1, shading="flat", rasterized=True)
            ax.set(ylim=(0, edges[-1]), ylabel="Selected-node Syn (bits)")
            ax.set_title(f"{'abcdef'[i]}  Order {order}", loc="left", pad=6, fontweight="bold")
            ax.tick_params(axis="x", labelbottom=False, bottom=False)
            availability_ax.pcolormesh(bin_edges(gs), [0, 1], present[None, :],
                                      cmap="Blues", vmin=0, vmax=8, shading="flat")
            availability_ax.set(yticks=[], ylabel="n")
            availability_ax.yaxis.label.set_rotation(0)
            availability_ax.yaxis.label.set_size(8)
            setup_G_axis(availability_ax, gs)
            bin_records[str(order)] = dict(edges_bits=edges.tolist(), present_seed_count=present.tolist(),
                                            selected_node_count_by_G=data["counts"][:, :, oi].sum(0).tolist())
        colorbar = fig.colorbar(image, cax=fig.add_subplot(grid[:, 3]))
        colorbar.set_label("Probability per bin (equal weight per present seed)")
        fig.suptitle(f"Fixed {n_bins}-bin empirical distributions; no smoothing; Syn axes differ by order\n"
                     "Grey: order absent. Blue strips: present seeds (white = 0, dark blue = 8).",
                     fontsize=9)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=350, bbox_inches="tight")
        plt.close(fig)
    return bin_records


def summary_payload(data, distributions, bins, figure, distribution_figure, statistics):
    gs = data["G"]
    rows = []
    for gi, g in enumerate(gs):
        group_stats = []
        for low, high in GROUPS:
            index = (ORDERS >= low) & (ORDERS <= high)
            values = data["mass"][:, gi, index].sum(axis=1)
            shares = data["share"][:, gi, index].sum(axis=1)
            group_stats.append(dict(orders=[low, high], mass_mean_bits=float(values.mean()),
                                    mass_sd_bits=float(values.std(ddof=1)),
                                    share_mean_percent=float(100 * shares.mean()),
                                    share_sd_percentage_points=float(100 * shares.std(ddof=1))))
        rows.append(dict(G=float(g), cross_roi_xi_mean_bits=float(data["xi"][:, gi].mean()),
                         cross_roi_xi_sd_bits=float(data["xi"][:, gi].std(ddof=1)),
                         peak_mass_order=int(ORDERS[np.argmax(data["mean_mass"][gi])]),
                         groups=group_stats))
    anchor_orders = []
    for g in (0., .8, 1.3, 2., 3.):
        gi = int(np.flatnonzero(np.isclose(gs, g))[0])
        orders = []
        for oi, order in enumerate(ORDERS):
            value = data["mean_node_syn"][gi, oi]
            orders.append(dict(order=int(order), mass_mean_bits=float(data["mean_mass"][gi, oi]),
                               mass_sd_bits=float(data["mass_sd"][gi, oi]),
                               share_mean_percent=float(100 * data["mean_share"][gi, oi]),
                               present_node_mean_syn_bits=float(value) if np.isfinite(value) else None,
                               present_node_distribution_sd_bits=float(distributions["weighted_sd"][gi, oi])
                               if np.isfinite(value) else None,
                               present_node_quartiles_bits=distributions["weighted_quantiles"][gi, oi].tolist()
                               if np.isfinite(value) else None,
                               present_seed_count=int(data["present_seed_count"][gi, oi]),
                               selected_node_count=int(data["counts"][:, gi, oi].sum())))
        anchor_orders.append(dict(G=g, orders=orders))
    peaks = []
    for oi, order in enumerate(ORDERS):
        gi = int(np.argmax(data["mean_mass"][:, oi]))
        peaks.append(dict(order=int(order), G=float(gs[gi]), mass_bits=float(data["mean_mass"][gi, oi])))
    return dict(status="complete", analysis="Selected SPT internal-node statistics; not all coalitions",
                seeds=SEEDS.tolist(), G=gs.tolist(), orders=ORDERS.tolist(),
                aggregation=dict(mass="Within-tree order sum, then equal 8-seed mean; absent order = 0",
                                 share="Within-tree order mass / root cross-ROI Xi, then equal 8-seed mean",
                                 node_mean="Within-tree present-node mean, then equal mean of present seeds; absent = missing",
                                 histogram="Within-tree order histogram normalized to 1, then equal present-seed mean",
                                 node_distribution_sd="Weighted empirical population spread of selected nodes; "
                                                      "equal present-seed weight, equal within-seed node weight",
                                 node_quartiles="Discrete weighted empirical quantiles at CDF >= 0.25, 0.5, 0.75",
                                 uncertainty="Sample SD across simulation seeds, not population confidence intervals",
                                 G_cells="Midpoint boundaries on actual nonuniform G grid; no interpolation"),
                audit=data["audit"], simulation_config=data["simulation"], search_config=data["search"],
                estimator="Inherited Gaussian conditional covariance, ridge=1e-6; no refit; "
                          "existing high-dimensional exception to TM-first policy; captures second-order dependencies",
                input_tree_fingerprint_sha256=data["fingerprint"],
                figures=[str(figure), str(distribution_figure)], statistics_file=str(statistics),
                histogram_bins=bins, distribution_order_count=len(ORDERS),
                histogram_bin_count=distributions["n_bins"],
                by_G=rows, anchor_order_statistics=anchor_orders, peak_mass_by_order=peaks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--statistics", type=Path, default=STATISTICS)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    parser.add_argument("--distribution-figure", type=Path, default=DISTRIBUTION_FIGURE)
    parser.add_argument("--bins", type=int, default=24)
    args = parser.parse_args()
    data = load_analysis(args.input_dir)
    distributions = order_distributions(data, args.bins)
    plot_summary(data, args.figure)
    bins = plot_distributions(data, distributions, args.distribution_figure)
    payload = summary_payload(data, distributions, bins, args.figure, args.distribution_figure, args.statistics)
    args.statistics.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.statistics, G=data["G"], seeds=SEEDS, orders=ORDERS,
                        mass_bits=data["mass"], mass_share=data["share"], node_counts=data["counts"],
                        present_node_mean_bits=data["node_means"], mean_node_syn_bits=data["mean_node_syn"],
                        cross_roi_xi_bits=data["xi"], present_seed_count=data["present_seed_count"],
                        distribution_probability=distributions["probability"],
                        distribution_edges_bits=distributions["edges"],
                        distribution_sd_bits=distributions["weighted_sd"],
                        distribution_quartiles_bits=distributions["weighted_quantiles"],
                        quantile_levels=np.array([.25, .5, .75]),
                        metadata=json.dumps({k: payload[k] for k in ("aggregation", "audit", "estimator",
                                                                     "input_tree_fingerprint_sha256")},
                                            allow_nan=False))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload["audit"], indent=2))
    print(f"Figures: {args.figure}\n{args.distribution_figure}")


if __name__ == "__main__":
    main()
