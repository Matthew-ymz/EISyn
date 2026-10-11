#!/usr/bin/env python3
"""Verified 2023 exploration figure from locked predictions and paired caches."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/nyc_taxi_information_prior_calibration/exploration2023"
OUTPUT = ROOT / "docs/reports/assets/nyc_taxi_information_prior_exploration2023.png"
LABELS = dict(xi_shapley=r"$\Xi$-Shapley", uniform="Uniform ridge", ei_shapley="EI-Shapley",
              pair_only="Pair-only PEID", phi_r=r"$\Phi^R$-Shapley", phi_wms=r"$\Phi^{WMS}$-Shapley",
              surd="Gaussian SURD", o_shapley="Target O-Shapley", correlation="Correlation",
              grouped_shap="Grouped SHAP", univariate="Univariate", frozen="Frozen ensemble")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    summary = json.loads((args.input / "summary.json").read_text())
    manifest = json.loads((args.input / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Need completed exploration manifest")
    saved = np.load(args.input / "T23_predictions.npz", allow_pickle=False)
    bstar = summary["B_star"]
    primary = ["uniform", "ei_shapley", bstar]
    colors = ("#555F6D", "#0072B2", "#D55E00")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.8), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.2, 1]})
        ax = axes[0, 0]
        methods = list(LABELS)
        for y, m in enumerate(methods):
            point = summary["metrics"][m]["macro_nrmse"]
            interval = np.percentile(saved[f"bootstrap_{m}"], [2.5, 97.5])
            color = "#009E73" if m == "xi_shapley" else "#555F6D"
            # Percentile CI may miss its point estimate, so draw the interval directly.
            ax.plot(interval, [y, y], color=color, lw=1.3)
            ax.scatter([point], [y], color=color, s=23, zorder=3)
        ax.set_yticks(np.arange(len(methods)), [LABELS[m]+(" *" if m == "grouped_shap" and not summary['SHAP_precision']['converged'] else "") for m in methods])
        ax.invert_yaxis()
        ax.set_xlabel("Test macro nRMSE (lower is better)")
        ax.text(-.12, 1.02, "a", transform=ax.transAxes, weight="bold", fontsize=12)
        ax = axes[0, 1]
        for y, (method, color) in enumerate(zip(primary, colors)):
            record = summary["comparison"][method]
            interval, point = record["relative_gain_ci95"], record["relative_gain_percent"]
            ax.plot(interval, [y, y], color=color, lw=2)
            ax.scatter([point], [y], color=color, s=35, zorder=3)
        ax.axvline(0, color="#8D97A1", ls=":", lw=.8, zorder=0)
        ax.set_yticks(np.arange(3), [LABELS[m]+(" (B*)" if m == bstar else "") for m in primary])
        ax.invert_yaxis()
        ax.set_ylim(2.7, -.7)
        ax.set_xlabel(r"$\Xi$ error reduction relative to baseline (%)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.text(-.12, 1.02, "b", transform=ax.transAxes, weight="bold", fontsize=12)
        ax = axes[1, 0]
        real = summary["metrics"]["xi_shapley"]["macro_nrmse"]
        null = (saved["null_nrmse"] - real) * 1e6
        if np.ptp(null) < 1e-12:
            center = float(null[0])
            bins = np.linspace(center-.1, center+.1, 15)
        else:
            bins = 20
        ax.hist(null, bins=bins, color="#9BACCA", edgecolor="white")
        ax.axvline(0., color="#009E73", lw=1.8, label=r"Unshuffled $\Xi$")
        ax.set(xlabel=r"Shuffled $-$ unshuffled test macro nRMSE ($\times 10^{-6}$)"+"\n200 independently tuned shuffles",
               ylabel="Number of shuffled priors")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False)
        ax.text(-.12, 1.1, "c", transform=ax.transAxes, weight="bold", fontsize=12)
        ax = axes[1, 1]
        rows = {r["method"]: r for r in summary["stratification"]["comparisons"]}
        for method, color, marker in zip(primary, colors, ("o", "s", "^")):
            ax.plot(np.arange(1, 6), rows[method]["quintile_gain_percent"], color=color, marker=marker,
                    ms=4, lw=1.2, label=LABELS[method]+(" (B*)" if method == bstar else ""))
        ax.axhline(0, color="#8D97A1", ls=":", lw=.8)
        ax.set(xlabel="D0 traffic-volume quintile (low to high)", ylabel=r"$\Xi$ error reduction (%)", xticks=np.arange(1, 6))
        ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), frameon=False, handlelength=1.6)
        ax.text(-.12, 1.02, "d", transform=ax.transAxes, weight="bold", fontsize=12)
        note = "SHAP precision passed" if summary["SHAP_precision"]["converged"] else "* SHAP precision failed; excluded from B* selection"
        fig.suptitle("NYC Taxi 2023 exploration: 760 test hours, three frozen checkpoints\n"
                     "Paired 7-day block bootstrap (95% intervals); "+note, fontsize=10)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=220, bbox_inches="tight")
        plt.close(fig)
    saved.close()
    print(args.output)


if __name__ == "__main__":
    main()
