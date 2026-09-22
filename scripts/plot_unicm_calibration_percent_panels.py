#!/usr/bin/env python3
"""Redraw Figure 4 panels i/j with percentage x-axes and direct value labels.

Reads the two authoritative summary.json files:
- results/unicm_synergy_regularized_forecast_normfit_1980_2003/summary.json
- results/unicm_target_xi_shapley_prior_normfit_1980_2003_n16384/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = (
    ROOT
    / "results"
    / "unicm_synergy_regularized_forecast_normfit_1980_2003"
    / "summary.json"
)
XI = (
    ROOT
    / "results"
    / "unicm_target_xi_shapley_prior_normfit_1980_2003_n16384"
    / "summary.json"
)
OUTPUT = ROOT / "tmp" / "review" / "figure4ij_percent_preview.png"

INK = "#172033"
MID_GREY = "#8D96A5"
LIGHT_GREY = "#E9EDF2"
BLUE = "#3F6F9F"
XI_COLOR = "#287D76"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def main() -> None:
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    xi_calibration = json.loads(XI.read_text(encoding="utf-8"))

    metrics = calibration["test_metrics"]
    frozen = float(metrics["frozen"]["mean_cell_nrmse"])
    univariate = float(metrics["univariate"]["mean_cell_nrmse"])
    uniform = float(metrics["uniform"]["mean_cell_nrmse"])
    xi = float(xi_calibration["test_nrmse"]["target_xi_shapley_prior"])

    keys = ("frozen", "univariate", "uniform", "xi")
    labels = ("Frozen", "Univariate", "Uniform ridge", r"$\Xi$-Shapley prior")
    values = np.asarray([frozen, univariate, uniform, xi])
    colors = (MID_GREY, "#91A7CF", BLUE, XI_COLOR)
    reductions = (frozen - values) / frozen * 100.0

    fig, (ax_i, ax_j) = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 1.15), "wspace": 0.10},
    )

    # --- Panel i: RMSE reduction relative to the frozen ensemble ----------
    y = np.arange(len(keys))[::-1]
    ax_i.scatter(
        reductions,
        y,
        s=31,
        color=colors,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    for reduction, nrmse, y_value in zip(reductions, values, y):
        text = f"{reduction:.1f}% ({nrmse:.3f})"
        ax_i.text(
            reduction + 0.22,
            y_value,
            text,
            va="center",
            ha="left",
            fontsize=5.6,
            color=INK,
        )
    ax_i.set_yticks(y, labels)
    ax_i.set_xlabel("Test RMSE reduction vs frozen ensemble (%)")
    ax_i.set_xlim(-0.5, 13.5)
    ax_i.set_ylim(-0.55, 3.55)
    ax_i.grid(axis="x", color=LIGHT_GREY, linewidth=0.5)
    ax_i.text(
        -0.18, 1.04, "i", transform=ax_i.transAxes,
        fontsize=8.5, fontweight="bold", color=INK, ha="left", va="bottom",
    )

    # --- Panel j: gain over uniform ridge, in percent ---------------------
    random_scores = np.asarray(
        xi_calibration["shuffled_xi_control"]["scores"], dtype=float
    )
    repeats = int(xi_calibration["shuffled_xi_control"]["repeats"])
    p_value = float(
        xi_calibration["shuffled_xi_control"]["fraction_null_at_least_as_good"]
    )
    null_gains = (uniform - random_scores) / uniform * 100.0
    xi_gain = (uniform - xi) / uniform * 100.0
    display_min = -0.006 / uniform * 100.0
    visible = null_gains[null_gains >= display_min]
    rng = np.random.default_rng(20260728)
    null_y = 1.0 + rng.uniform(-0.18, 0.18, size=len(visible))

    ax_j.axvline(0, color="#686F78", linewidth=0.7, linestyle="--", zorder=1)
    ax_j.axhline(1.0, color=LIGHT_GREY, linewidth=0.5, zorder=0)
    ax_j.axhline(0.0, color=LIGHT_GREY, linewidth=0.5, zorder=0)
    ax_j.scatter(
        visible, null_y, s=14, color="#B4BFCC", edgecolor="white",
        linewidth=0.25, alpha=0.62, zorder=2,
    )
    ax_j.scatter(
        [xi_gain], [0.0], s=32, marker="o", color=XI_COLOR,
        edgecolor="white", linewidth=0.5, zorder=3,
    )
    ax_j.text(
        0.02, 0.92, f"{len(visible)}/{repeats} null draws shown",
        transform=ax_j.transAxes, ha="left", va="top",
        fontsize=5.5, color="#657080",
    )
    ax_j.text(
        0.98, 0.92, rf"$P={p_value:.3f}$",
        transform=ax_j.transAxes, ha="right", va="top",
        fontsize=5.6, color=INK,
    )
    ax_j.text(
        xi_gain - 0.12, 0.13, f"{xi_gain:+.2f}%",
        ha="right", va="bottom", fontsize=6.0, color=XI_COLOR,
        fontweight="bold",
    )
    ax_j.set_yticks((1.0, 0.0), (r"Shuffled $\Xi$ priors", r"$\Xi$-Shapley prior"))
    ax_j.set_xlim(display_min, max(3.2, xi_gain + 0.5))
    ax_j.set_ylim(-0.42, 1.42)
    ax_j.grid(axis="x", color=LIGHT_GREY, linewidth=0.5)
    ax_j.set_xlabel("RMSE gain over uniform ridge (%)")
    ax_j.text(
        -0.18, 1.04, "j", transform=ax_j.transAxes,
        fontsize=8.5, fontweight="bold", color=INK, ha="left", va="bottom",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUTPUT}")
    print(f"reductions vs frozen (%): {np.round(reductions, 2)}")
    print(f"xi gain vs uniform: {xi_gain:.2f}%; visible nulls: {len(visible)}/{repeats}")


if __name__ == "__main__":
    main()
