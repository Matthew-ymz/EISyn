#!/usr/bin/env python3
"""Plot the corrected NYC Taxi Ridge Xi decomposition from cached JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "nyc_taxi_global_ridge_xi.json"
OUTPUT = ROOT / "fig" / "nyc_taxi_corrected_xi_decomposition"

COLORS = {
    "singleton": "#AAB7C4",
    "cross": "#3D6FA6",
    "within": "#D98C5F",
    "ink": "#263442",
    "muted": "#687684",
    "grid": "#DDE3E8",
}

ZONE_NAMES = {
    246: "West Chelsea",
    161: "Midtown Center",
    230: "Times Sq",
    237: "UES South",
    48: "Clinton East",
    236: "UES North",
    162: "Midtown East",
    186: "Penn Station",
    43: "Central Park",
    100: "Garment District",
    142: "Lincoln Sq East",
    79: "East Village",
    239: "UWS South",
    68: "East Chelsea",
    231: "TriBeCa/Civic Ctr",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.05,
        label,
        transform=axis.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLORS["ink"],
    )


def decomposition_panel(axis: plt.Axes, decomposition: dict[str, float]) -> None:
    singleton = float(decomposition["scalar_singleton_ei_sum"])
    cross = float(decomposition["cross_region_xi"])
    within = float(decomposition["within_region_xi_sum"])
    joint = float(decomposition["joint_ei"])
    system = float(decomposition["system_xi"])

    y_joint, y_xi = 1.0, 0.0
    axis.barh(y_joint, singleton, height=0.48, color=COLORS["singleton"])
    axis.barh(y_joint, cross, left=singleton, height=0.48, color=COLORS["cross"])
    axis.barh(y_joint, within, left=singleton + cross, height=0.48, color=COLORS["within"])
    axis.barh(y_xi, cross, height=0.48, color=COLORS["cross"])
    axis.barh(y_xi, within, left=cross, height=0.48, color=COLORS["within"])

    axis.text(singleton / 2, y_joint, f"singletons\n{singleton:.2f}", ha="center", va="center", color="white", fontsize=6.2)
    axis.text(singleton + cross / 2, y_joint, f"cross\n{cross:.2f}", ha="center", va="center", color="white", fontsize=6.2)
    axis.text(cross / 2, y_xi, f"cross-region  {cross:.2f}", ha="center", va="center", color="white", fontsize=6.2)
    axis.annotate(
        f"within {within:.2f}",
        xy=(cross + within / 2, y_xi),
        xytext=(cross + within + 2.0, y_xi + 0.30),
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": COLORS["muted"]},
        va="center",
        fontsize=6.2,
        color=COLORS["ink"],
    )
    axis.text(joint + 0.5, y_joint, f"{joint:.2f} bits", va="center", fontsize=6.4, color=COLORS["ink"])
    axis.text(system + 0.5, y_xi - 0.22, f"total {system:.2f} bits", va="center", fontsize=6.4, color=COLORS["ink"])

    axis.set_yticks([y_xi, y_joint], [r"System $\Xi$", "Joint EI"])
    axis.set_xlabel("Information (bits)")
    axis.set_xlim(0, 34)
    axis.grid(axis="x", color=COLORS["grid"], linewidth=0.5, zorder=0)
    axis.set_axisbelow(True)
    axis.text(
        0.02,
        1.0,
        rf"$\Xi$/joint EI = {100 * system / joint:.1f}%",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
        color=COLORS["muted"],
    )


def target_panel(axis: plt.Axes, rows: list[dict[str, float]], top_n: int = 12) -> None:
    ranked = sorted(rows, key=lambda row: float(row["system_xi_bits"]), reverse=True)[:top_n]
    ranked = ranked[::-1]
    cross = np.array([float(row["cross_region_xi_bits"]) for row in ranked])
    within = np.array([float(row["within_region_xi_bits"]) for row in ranked])
    labels = [ZONE_NAMES.get(int(row["target_zone_id"]), f"Zone {int(row['target_zone_id'])}") for row in ranked]
    y = np.arange(len(ranked))
    axis.barh(y, cross, color=COLORS["cross"], height=0.68, label=r"Cross-region $\Xi$")
    axis.barh(y, within, left=cross, color=COLORS["within"], height=0.68, label=r"Within-region $\Xi$")
    for index, total in enumerate(cross + within):
        axis.text(total + 0.006, index, f"{total:.2f}", va="center", fontsize=5.7, color=COLORS["ink"])
    axis.set_yticks(y, labels)
    axis.set_xlabel(r"Target-resolved $\Xi$ (bits)")
    axis.set_xlim(0, 0.47)
    axis.grid(axis="x", color=COLORS["grid"], linewidth=0.5)
    axis.set_axisbelow(True)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        fontsize=6.2,
        handlelength=1.4,
        columnspacing=1.2,
    )


def lag_panel(axis: plt.Axes, lag_sums: dict[str, float]) -> None:
    lag_labels = {"1": "0.5 h", "2": "1 h", "3": "1.5 h", "6": "3 h", "12": "6 h", "48": "24 h", "336": "168 h"}
    keys = list(lag_sums)
    values = np.array([float(lag_sums[key]) for key in keys])
    x = np.arange(len(keys))
    bars = axis.bar(x, values, color=COLORS["cross"], width=0.68)
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.10, f"{value:.2f}", ha="center", va="bottom", fontsize=5.7)
    axis.set_xticks(x, [lag_labels[key] for key in keys], rotation=35, ha="right")
    axis.set_ylabel("Singleton EI sum (bits)")
    axis.set_xlabel("Lag")
    axis.set_ylim(0, 6.25)
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    axis.set_axisbelow(True)
    axis.text(0.98, 0.96, "Across 69 zones", transform=axis.transAxes, ha="right", va="top", fontsize=6.1, color=COLORS["muted"])


def sensitivity_panel(axis: plt.Axes, rows: list[dict[str, float]]) -> None:
    ridge = np.array([float(row["covariance_ridge"]) for row in rows])
    system = np.array([float(row["system_xi_bits"]) for row in rows])
    cross = np.array([float(row["cross_region_xi_bits"]) for row in rows])
    axis.plot(ridge, system, "o-", color=COLORS["ink"], lw=1.2, ms=3.3)
    axis.plot(ridge, cross, "s-", color=COLORS["cross"], lw=1.2, ms=3.1)
    axis.set_xscale("log")
    axis.set_xlabel("Residual-covariance ridge")
    axis.set_ylabel(r"$\Xi$ (bits)")
    axis.set_ylim(7.2, 8.35)
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    axis.set_axisbelow(True)
    axis.annotate("System", (ridge[-1], system[-1]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=6.2, color=COLORS["ink"])
    axis.annotate("Cross-region", (ridge[-1], cross[-1]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=6.2, color=COLORS["cross"])
    max_system_drift = 100 * (system.max() - system.min()) / system[2]
    axis.text(0.03, 0.06, f"Maximum drift: {max_system_drift:.2f}%", transform=axis.transAxes, fontsize=6.2, color=COLORS["muted"])
    axis.set_xlim(ridge.min() / 1.7, ridge.max() * 2.6)


def main() -> None:
    configure()
    payload = json.loads(INPUT.read_text(encoding="utf-8"))

    figure = plt.figure(figsize=(7.2, 5.25), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.0, 1.18), height_ratios=(1.0, 1.0))
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
    ]

    decomposition_panel(axes[0], payload["decomposition_bits"])
    target_panel(axes[1], payload["target_resolved"])
    lag_panel(axes[2], payload["lag_scalar_ei_sums_bits"])
    sensitivity_panel(axes[3], payload["covariance_ridge_sensitivity"])
    for axis, label in zip(axes, "abcd", strict=True):
        panel_label(axis, label)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
