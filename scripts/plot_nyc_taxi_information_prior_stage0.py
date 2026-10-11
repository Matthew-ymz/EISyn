#!/usr/bin/env python3
"""Stage-0 diagnostic figure; no test estimates or confirmatory ranking."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/nyc_taxi_information_prior_calibration/smoke"
OUTPUT = ROOT / "docs/reports/assets/nyc_taxi_information_prior_stage0.png"
LABELS = dict(xi_shapley=r"$\Xi$-Shapley", ei_shapley="EI-Shapley", pair_only="Pair-only PEID",
              phi_r=r"$\Phi^R$-Shapley", phi_wms=r"$\Phi^{WMS}$-Shapley", surd="Gaussian SURD",
              o_shapley="Target O-Shapley", correlation="Correlation", uniform="Uniform ridge",
              univariate="Univariate", frozen="Frozen seed 0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = json.loads((args.input / "manifest.json").read_text())
    if manifest["status"] != "passed" or manifest["stage"] != "smoke":
        raise ValueError("Need a passed stage-0 manifest")
    selection = json.loads((args.input / "validation_selection.json").read_text())["selection"]
    with np.load(args.input / "priors_and_fv_predictions.npz") as saved:
        raw = saved["prior_xi_shapley"]
    with np.load(ROOT / "data/nyc_taxi_mgstn_2023/nyc_taxi_mgstn_hourly.npz") as saved:
        zones = saved["zone_ids"]
    total = raw.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("Source shares undefined for zero-Xi zones; do not silently use uniform shares")
    shares = raw / total
    methods = list(LABELS)
    colors = ("#0072B2", "#E69F00", "#009E73")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.25, 1]})
        ax = axes[0]
        bottom = np.zeros(66)
        for j, (label, color) in enumerate(zip(("Recent", "Daily", "Weekly"), colors)):
            ax.bar(np.arange(66), shares[:, j] * 100, bottom=bottom, width=.9, color=color, label=label)
            bottom += shares[:, j] * 100
        ax.set(xlim=(-1, 66), ylim=(0, 100), xlabel="Taxi zone ID", ylabel=r"Share of local $\Xi$ attribution (%)")
        positions = np.r_[np.arange(0, 60, 8), 65]
        ax.set_xticks(positions, zones[positions], rotation=45)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False,
                  handlelength=1.2, columnspacing=1.3)
        ax.text(-.1, 1.1, "a", transform=ax.transAxes, weight="bold", fontsize=12)
        ax = axes[1]
        y = np.arange(len(methods))
        values = np.array([selection[m]["macro_nrmse"] for m in methods])
        point_colors = ["#009E73" if m == "xi_shapley" else "#555F6D" for m in methods]
        ax.scatter(values, y, c=point_colors, s=30, zorder=3)
        for value, row in zip(values, y):
            ax.annotate(f"{value:.4f}", (value, row), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8)
        ax.set_yticks(y, [LABELS[m] for m in methods])
        ax.invert_yaxis()
        span = max(float(np.ptp(values)), .01)
        ax.set(xlim=(max(0, values.min() - span * .12), values.max() + span * .3),
               xlabel="Validation macro nRMSE (lower is better)")
        ax.axvline(selection["uniform"]["macro_nrmse"], color="#8D97A1", ls=":", lw=.8, zorder=0)
        ax.text(-.1, 1.1, "b", transform=ax.transAxes, weight="bold", fontsize=12)
        fig.suptitle("Stage 0: 512 interventions, one checkpoint, one context\nF/V diagnostics only; SHAP and test evaluation pending", fontsize=10)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=220, bbox_inches="tight")
        plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
