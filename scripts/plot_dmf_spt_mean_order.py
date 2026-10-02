#!/usr/bin/env python3
"""One DMF order-distribution heatmap with its mean order overlaid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_dmf_spt_order_distribution import (
    INPUT, ORDERS, SEEDS, STYLE, bin_edges, load_analysis, setup_G_axis,
)

FIGURE = ROOT / "fig/dmf_schaefer100/dmf_spt_order_distribution_mean_order.png"
SUMMARY = ROOT / "results/dmf_schaefer100/spt_order_distribution/mean_order_summary.json"


def mean_orders(distribution, orders=ORDERS):
    """Expected order for each seed/G; normalize before averaging seeds."""
    distribution = np.asarray(distribution, dtype=float)
    orders = np.asarray(orders, dtype=float)
    if distribution.ndim != 3 or distribution.shape[-1] != len(orders):
        raise ValueError("Expected seed x G x order probabilities")
    if not np.isfinite(distribution).all() or np.any(distribution < 0):
        raise ValueError("Order probabilities must be finite and nonnegative")
    if not np.allclose(distribution.sum(axis=-1), 1, atol=1e-12, rtol=0):
        raise ValueError("Order probabilities must sum to one for every seed/G")
    per_seed = distribution @ orders
    mean = per_seed.mean(axis=0)
    np.testing.assert_allclose(mean, distribution.mean(axis=0) @ orders, atol=1e-12, rtol=0)
    return per_seed, mean, per_seed.std(axis=0, ddof=1)


def plot(data, distribution, mean, output, weighting):
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.3, 4.8), layout="constrained")
        image = ax.pcolormesh(bin_edges(data["G"]), np.arange(1.5, 101.5),
                              100 * distribution.mean(axis=0).T,
                              cmap="viridis", vmin=0,
                              vmax=float(100 * distribution.mean(axis=0).max()),
                              shading="flat", rasterized=True)
        line, = ax.plot(data["G"], mean, color="white", linewidth=1.6,
                        path_effects=[path_effects.Stroke(linewidth=2.7, foreground="#252525"),
                                      path_effects.Normal()],
                        label="Syn-weighted mean order" if weighting == "syn" else "Mean node order")
        setup_G_axis(ax, data["G"])
        ax.set(ylim=(1.5, 100.5), ylabel="Synergy order (ROI count)",
               yticks=[2, 10, 20, 40, 60, 80, 100])
        # Keep the key outside the heatmap, including its overlaid mean curve.
        ax.legend(handles=[line], loc="lower left", bbox_to_anchor=(0, 1.025),
                  frameon=False, borderaxespad=0, handlelength=2.7)
        ax.text(1, 1.04, "Within-tree normalization; 8-seed mean",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#444444")
        colorbar = fig.colorbar(image, ax=ax, fraction=.035, pad=.025)
        colorbar.set_label("Share of cross-ROI Syn (%)" if weighting == "syn" else "Share of SPT nodes (%)")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=400, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--weighting", choices=("syn", "nodes"), default="syn")
    args = parser.parse_args()
    data = load_analysis(args.input_dir)
    node_frequency = data["counts"] / data["counts"].sum(axis=2, keepdims=True)
    distributions = {"syn": data["share"], "nodes": node_frequency}
    results = {key: mean_orders(distribution) for key, distribution in distributions.items()}
    per_seed, mean, sd = results[args.weighting]
    plot(data, distributions[args.weighting], mean, args.figure, args.weighting)
    minimum, maximum = int(np.argmin(mean)), int(np.argmax(mean))
    payload = dict(
        weighting=args.weighting, figure=str(args.figure),
        mean_definition="Sum of order times within-tree normalized order mass; then equal seed mean"
                        if args.weighting == "syn" else
                        "Sum of order times within-tree normalized node counts; then equal seed mean",
        heatmap_definition="Within-tree order probabilities, then equal seed mean; every column sums to one",
        sd_definition="Sample SD of each seed's mean order; not an order-distribution spread or confidence interval",
        orders=ORDERS.tolist(), seeds=SEEDS.tolist(), G=data["G"].tolist(),
        mean_order=mean.tolist(), mean_order_sd=sd.tolist(), per_seed_mean_order=per_seed.tolist(),
        syn_weighted_mean_order=results["syn"][1].tolist(),
        node_count_mean_order=results["nodes"][1].tolist(),
        minimum=dict(G=float(data["G"][minimum]), mean_order=float(mean[minimum])),
        maximum=dict(G=float(data["G"][maximum]), mean_order=float(mean[maximum])),
        audit=data["audit"], input_tree_fingerprint_sha256=data["fingerprint"],
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for g in (0., .8, 1.3, 2., 3.):
        i = int(np.flatnonzero(np.isclose(data["G"], g))[0])
        print(f"G={g:g}: mean order={mean[i]:.6f}, seed SD={sd[i]:.6f}")
    print(f"Figure: {args.figure}")


if __name__ == "__main__":
    main()
