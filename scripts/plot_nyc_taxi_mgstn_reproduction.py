#!/usr/bin/env python3
"""Publication-ready audit figure for the three-seed MGSTN reproduction.

Figure contract
---------------
Claim: all four reproduced taxi forecasting metrics match the paper within 5%.
Evidence: absolute metric comparison, relative-gap acceptance band, and validation
trajectories for all three independent seeds. Python/matplotlib is the exclusive
rendering backend; PNG, SVG, and PDF are exported from the same figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "nyc_taxi_mgstn" / "summary.json"
OUTPUT = ROOT / "fig" / "nyc_taxi_mgstn_reproduction"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def main() -> None:
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    keys = ["inflow_mae", "inflow_rmse", "outflow_mae", "outflow_rmse"]
    labels = ["Inflow\nMAE", "Inflow\nRMSE", "Outflow\nMAE", "Outflow\nRMSE"]
    paper_sd = np.asarray([0.026, 0.078, 0.047, 0.079])
    paper = np.asarray([payload["summary"][key]["paper"] for key in keys])
    reproduced = np.asarray([payload["summary"][key]["mean"] for key in keys])
    reproduced_sd = np.asarray([payload["summary"][key]["sd"] for key in keys])
    relative = 100 * (reproduced / paper - 1)

    colors = {"paper": "#7A8795", "ours": "#2878B5", "accent": "#D97706"}
    fig = plt.figure(figsize=(10.2, 3.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.35])

    ax = fig.add_subplot(grid[0, 0])
    x = np.arange(len(keys))
    offset = 0.14
    ax.errorbar(
        x - offset,
        paper,
        yerr=paper_sd,
        fmt="o",
        color=colors["paper"],
        markersize=5,
        capsize=2.5,
        linewidth=1.2,
        label="Paper (mean ± SD, n=3)",
    )
    ax.errorbar(
        x + offset,
        reproduced,
        yerr=reproduced_sd,
        fmt="o",
        color=colors["ours"],
        markersize=5,
        capsize=2.5,
        linewidth=1.2,
        label="Reproduction (mean ± SD, n=3)",
    )
    for index, value in enumerate(reproduced):
        ax.text(index + offset, value + reproduced_sd[index] + 0.35, f"{value:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Error (rides per zone-hour; lower is better)")
    ax.set_ylim(0, max(reproduced + reproduced_sd) * 1.18)
    ax.text(-0.17, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)

    ax = fig.add_subplot(grid[0, 1])
    bar_colors = [colors["ours"] if abs(value) <= 5 else colors["accent"] for value in relative]
    ax.axhspan(-5, 5, color="#DDEFE2", alpha=0.75, zorder=0)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.bar(x, relative, width=0.58, color=bar_colors, edgecolor="none")
    for index, value in enumerate(relative):
        ax.text(index, value + (0.25 if value >= 0 else -0.25), f"{value:+.1f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Reproduction − paper (%)")
    ax.set_ylim(-5.5, 6.2)
    ax.text(0.02, 0.93, "±5% acceptance band", transform=ax.transAxes, color="#3E6B48", fontsize=7)
    ax.text(-0.17, 1.02, "b", transform=ax.transAxes, fontweight="bold", fontsize=10)

    ax = fig.add_subplot(grid[0, 2])
    seed_colors = ["#2878B5", "#6F4E9C", "#C44E52"]
    for run, color in zip(payload["runs"], seed_colors):
        epochs = np.asarray([row["epoch"] for row in run["history"]])
        values = np.asarray([row["normalized_mse"] for row in run["history"]])
        ax.plot(epochs, values, color=color, linewidth=1.25, label=f"seed {run['seed']}")
        best = int(run["best_epoch"])
        best_value = values[epochs == best][0]
        ax.scatter([best], [best_value], color=color, s=20, zorder=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation normalized MSE")
    ax.set_ylim(0.23, 0.44)
    ax.text(-0.17, 1.02, "c", transform=ax.transAxes, fontweight="bold", fontsize=10)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in [("png", {"dpi": 400}), ("svg", {}), ("pdf", {})]:
        fig.savefig(OUTPUT.with_suffix(f".{suffix}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
