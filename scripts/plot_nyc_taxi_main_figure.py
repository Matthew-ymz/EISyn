#!/usr/bin/env python3
"""Create the concise main figure for the NYC Taxi experiment report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCREEN_JSON = ROOT / "results" / "nyc_taxi_synergy_model_screen_metrics.json"
XI_JSON = ROOT / "results" / "nyc_taxi_global_ridge_xi.json"
DATA_CACHE = ROOT / "data" / "nyc_taxi_yellow_2023_30min_manhattan.npz"
OUTPUT = ROOT / "fig" / "nyc_taxi_main_results"

INK = "#263442"
MUTED = "#697887"
GRID = "#DDE4EA"
LOCAL = "#AAB7C4"
GLOBAL = "#3D6FA6"
INTERACTION = "#2A9D8F"
WITHIN = "#D98C5F"


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
        -0.14,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
    )


def prediction_panel(axis: plt.Axes, models: dict[str, object]) -> None:
    names = ["Local Ridge", "Global Ridge", "Interaction Ridge"]
    values = np.array([float(models[name]["test"]["log_scaled_rmse"]) for name in names])
    colors = [LOCAL, GLOBAL, INTERACTION]
    y = np.arange(len(names))[::-1]

    axis.hlines(y, 0.533, values, color=GRID, lw=1.1, zorder=1)
    axis.scatter(values, y, c=colors, s=45, zorder=3, edgecolor="white", linewidth=0.5)
    point_labels = [
        f"{values[0]:.4f}",
        f"{values[1]:.4f}  (−3.29%)",
        f"{values[2]:.4f}  (≈ Global)",
    ]
    for value, ypos, point_label in zip(values, y, point_labels, strict=True):
        axis.text(value + 0.00055, ypos, point_label, va="center", fontsize=6.2, color=INK)
    axis.set_yticks(y, names)
    axis.set_xlim(0.533, 0.561)
    axis.set_xlabel("Test log-scaled RMSE  ↓")
    axis.grid(axis="x", color=GRID, lw=0.5)
    axis.set_axisbelow(True)


def zone_consistency_panel(axis: plt.Axes, models: dict[str, object]) -> None:
    from scripts.nyc_taxi_synergy_model_screen import make_design

    cached = np.load(DATA_CACHE, allow_pickle=False)
    design = make_design(cached["counts"], cached["zone_ids"], smoke=False)
    active = design.counts[design.train_mask].mean(axis=0) >= 1.0
    local_rmse = np.asarray(models["Local Ridge"]["test"]["per_zone_log_rmse"], dtype=float)
    global_rmse = np.asarray(models["Global Ridge"]["test"]["per_zone_log_rmse"], dtype=float)
    gain = 100 * (local_rmse[active] - global_rmse[active]) / local_rmse[active]

    rng = np.random.default_rng(20260819)
    jitter = rng.uniform(-0.085, 0.085, size=len(gain))
    axis.scatter(gain, jitter, s=13, color=GLOBAL, alpha=0.78, edgecolor="white", linewidth=0.25, zorder=3)
    axis.boxplot(
        gain,
        orientation="horizontal",
        positions=[0],
        widths=0.32,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "none", "edgecolor": INK, "linewidth": 0.8},
        medianprops={"color": WITHIN, "linewidth": 1.3},
        whiskerprops={"color": INK, "linewidth": 0.7},
        capprops={"color": INK, "linewidth": 0.7},
    )
    axis.axvline(0, color=MUTED, lw=0.8, ls="--")
    axis.set_yticks([])
    axis.set_xlim(-13, 19)
    axis.set_ylim(-0.28, 0.28)
    axis.set_xlabel("Per-zone RMSE improvement (%)  →")
    axis.grid(axis="x", color=GRID, lw=0.5)
    axis.set_axisbelow(True)
    improved = int(np.count_nonzero(gain > 0))
    axis.text(
        0.03,
        0.94,
        f"{improved}/{len(gain)} active zones improved",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        0.03,
        0.78,
        f"Median improvement: {np.median(gain):.2f}%",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.1,
        color=MUTED,
    )


def xi_panel(axis: plt.Axes, decomposition: dict[str, float]) -> None:
    singleton = float(decomposition["scalar_singleton_ei_sum"])
    cross = float(decomposition["cross_region_xi"])
    within = float(decomposition["within_region_xi_sum"])
    joint = float(decomposition["joint_ei"])
    system = float(decomposition["system_xi"])

    y_joint, y_system = 1.0, 0.0
    axis.barh(y_joint, singleton, height=0.48, color=LOCAL)
    axis.barh(y_joint, cross, left=singleton, height=0.48, color=GLOBAL)
    axis.barh(y_joint, within, left=singleton + cross, height=0.48, color=WITHIN)
    axis.barh(y_system, cross, height=0.48, color=GLOBAL)
    axis.barh(y_system, within, left=cross, height=0.48, color=WITHIN)

    axis.text(singleton / 2, y_joint, f"single-source EI\n{singleton:.2f}", ha="center", va="center", color="white", fontsize=6.3)
    axis.text(singleton + cross / 2, y_joint, f"cross\n{cross:.2f}", ha="center", va="center", color="white", fontsize=6.2)
    axis.text(cross / 2, y_system, f"cross-region  {cross:.2f}", ha="center", va="center", color="white", fontsize=6.3)
    axis.annotate(
        f"within-region\n{within:.2f}",
        xy=(cross + within / 2, y_system),
        xytext=(10.0, 0.34),
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": MUTED},
        fontsize=6.1,
        color=INK,
        ha="left",
        va="center",
    )
    axis.text(joint + 0.45, y_joint, f"{joint:.2f} bits", va="center", fontsize=6.4, color=INK)
    axis.text(system + 0.45, y_system - 0.22, f"{system:.2f} bits", va="center", fontsize=6.4, color=INK)
    axis.set_yticks([y_system, y_joint], [r"System $\Xi$", "Joint EI"])
    axis.set_xlim(0, 34)
    axis.set_xlabel("Information (bits)")
    axis.grid(axis="x", color=GRID, lw=0.5)
    axis.set_axisbelow(True)
    axis.text(
        0.02,
        1.01,
        f"Joint-information share = {100 * system / joint:.1f}%; cross-region share = {100 * cross / system:.1f}%",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color=MUTED,
    )


def main() -> None:
    configure()
    screen = json.loads(SCREEN_JSON.read_text(encoding="utf-8"))
    xi = json.loads(XI_JSON.read_text(encoding="utf-8"))

    figure = plt.figure(figsize=(7.2, 3.65), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(0.92, 1.25), height_ratios=(1, 1))
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[1, 0])
    ax_c = figure.add_subplot(grid[:, 1])

    prediction_panel(ax_a, screen["models"])
    zone_consistency_panel(ax_b, screen["models"])
    xi_panel(ax_c, xi["decomposition_bits"])
    for axis, label in zip((ax_a, ax_b, ax_c), "abc", strict=True):
        panel_label(axis, label)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
