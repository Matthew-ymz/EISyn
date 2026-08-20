#!/usr/bin/env python3
"""Create the NYC Taxi main figure from the finite quadratic-TM audit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from plot_nyc_taxi_temporal_coupling_map import load_geojson, outer_rings, zone_id


ROOT = Path(__file__).resolve().parents[1]
REPRODUCTION = ROOT / "results/nyc_taxi_mgstn/summary.json"
FINITE = ROOT / "results/nyc_taxi_mgstn_ei/finite_quadratic_tm/quadratic_tm_full_summary.json"
OUTPUT = ROOT / "fig/nyc_taxi_social_multiscale_main"

STATES = ["weekday_peak", "weekend_midday", "rainy_high_demand"]
STATE_LABELS = ["Weekday peak", "Weekend midday", "Rainy high-demand"]
SPARSE_ZONE_IDS = {120, 127, 128, 153, 194, 202}
TOLERANCE_BITS = 0.05

BLUE = "#477AA8"
TEAL = "#4F918B"
DARK_GRAY = "#59656D"
STATE_COLORS = ["#547A99", "#72A9A1", "#D4A06A"]


def interpreted(value: float) -> float:
    return 0.0 if -TOLERANCE_BITS <= value < 0 else value


def seed_zone_values(payload: dict, state: str) -> dict[int, dict[int, float]]:
    values: dict[int, dict[int, float]] = {}
    for unit in payload["units"]:
        if unit["state"] != state:
            continue
        values.setdefault(int(unit["model_seed"]), {})[int(unit["zone_id"])] = interpreted(
            float(unit["hurdle_temporal"]["syn_bits_raw"])
        )
    return values


def mean_zone_values(payload: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    zone_ids = np.asarray(sorted({int(unit["zone_id"]) for unit in payload["units"]}), dtype=int)
    values = {}
    for state in STATES:
        by_seed = seed_zone_values(payload, state)
        values[state] = np.asarray([
            np.mean([by_seed[seed][int(location_id)] for seed in sorted(by_seed)])
            for location_id in zone_ids
        ])
    return zone_ids, values


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def add_map_panels(fig, spec, finite: dict) -> None:
    geojson = load_geojson()
    features = geojson["features"]
    feature_by_id = {zone_id(feature): feature for feature in features}
    selected_ids, state_values = mean_zone_values(finite)
    missing = sorted(set(selected_ids.tolist()) - set(feature_by_id))
    if missing:
        raise RuntimeError(f"missing Taxi Zone polygons: {missing}")

    selected_rings = [
        ring for location_id in selected_ids for ring in outer_rings(feature_by_id[int(location_id)])
    ]
    points = np.vstack(selected_rings)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.018 * (y_max - y_min)

    all_values = np.concatenate([state_values[state] for state in STATES])
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "time_synergy", ["#F0F2EE", "#C9E1DB", "#78AFA6", "#2F6F69"]
    )
    norm = mpl.colors.Normalize(vmin=0.0, vmax=float(all_values.max()))
    grid = spec.subgridspec(1, 3, wspace=0.025)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]

    for index, (ax, state, state_label) in enumerate(zip(axes, STATES, STATE_LABELS)):
        background_patches = [
            Polygon(ring, closed=True)
            for feature in features
            for ring in outer_rings(feature)
        ]
        ax.add_collection(PatchCollection(
            background_patches, facecolor="#E7EBED", edgecolor="white", linewidth=0.22, zorder=0
        ))

        foreground_patches, foreground_values = [], []
        lookup = dict(zip(selected_ids.tolist(), state_values[state].tolist()))
        for location_id in selected_ids:
            rings = outer_rings(feature_by_id[int(location_id)])
            foreground_patches.extend(Polygon(ring, closed=True) for ring in rings)
            foreground_values.extend([lookup[int(location_id)]] * len(rings))
        foreground = PatchCollection(
            foreground_patches, cmap=cmap, norm=norm,
            edgecolor="#263238", linewidth=0.42, zorder=1,
        )
        foreground.set_array(np.asarray(foreground_values))
        ax.add_collection(foreground)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.text(0.5, -0.005, state_label, transform=ax.transAxes,
                ha="center", va="top", fontsize=9.5)
        if index == 0:
            panel_label(ax, "d", -0.04, 1.02)

    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
        fraction=0.018, pad=0.012, aspect=28,
    )
    colorbar.set_label("Time-scale synergy (bits)")
    colorbar.outline.set_linewidth(0.6)


def main() -> None:
    reproduction = json.loads(REPRODUCTION.read_text(encoding="utf-8"))
    finite = json.loads(FINITE.read_text(encoding="utf-8"))
    if finite["nonnegative_audit"]["hurdle"]["violation_count"]:
        raise RuntimeError("formal estimator failed its declared nonnegative audit")
    if len(finite["units"]) != 66 * 3 * 3:
        raise RuntimeError(f"expected 594 units, found {len(finite['units'])}")

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(13.2, 9.4), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[0.72, 1.55], hspace=0.045)
    top = outer[0].subgridspec(1, 3, width_ratios=[1.72, 0.82, 1.05], wspace=0.16)

    # a: reproduced forecasting accuracy and training stability.
    validation = top[0, 0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.23)
    ax_error = fig.add_subplot(validation[0, 0])
    keys = ["inflow_mae", "inflow_rmse", "outflow_mae", "outflow_rmse"]
    labels = ["Inflow\nMAE", "Inflow\nRMSE", "Outflow\nMAE", "Outflow\nRMSE"]
    x_metric = np.arange(4)
    run_values = np.asarray([[run["test"][key] for key in keys] for run in reproduction["runs"]])
    means, deviations = run_values.mean(axis=0), run_values.std(axis=0, ddof=1)
    for seed_index, values in enumerate(run_values):
        ax_error.scatter(x_metric + (seed_index - 1) * 0.055, values, s=20,
                         facecolor="white", edgecolor=BLUE, linewidth=0.7, zorder=3)
    ax_error.errorbar(x_metric, means, yerr=deviations, fmt="o", ms=5.5, color=BLUE,
                      ecolor=BLUE, elinewidth=1.0, capsize=2.5, zorder=4)
    for index, value in enumerate(means):
        ax_error.text(index, value + 0.55, f"{value:.2f}", ha="center", va="bottom",
                      fontsize=7.5, fontweight="bold")
    ax_error.set_xticks(x_metric, labels)
    ax_error.set_ylim(0, 18.2)
    ax_error.set_ylabel("Error (rides per zone-hour)")
    panel_label(ax_error, "a", -0.20)

    ax_curve = fig.add_subplot(validation[0, 1])
    seed_colors = ["#2B7BBB", "#7667A8", "#D45B52"]
    for color, run in zip(seed_colors, reproduction["runs"]):
        epochs = np.asarray([row["epoch"] for row in run["history"]])
        losses = np.asarray([row["normalized_mse"] for row in run["history"]])
        ax_curve.plot(epochs, losses, color=color, lw=1.25)
        ax_curve.scatter(int(run["best_epoch"]), float(run["best_validation_normalized_mse"]),
                         s=24, color=color, edgecolor="white", linewidth=0.5, zorder=3)
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Validation error")
    ax_curve.set_ylim(0.235, 0.42)

    summaries = {row["state"]: row for row in finite["summary"] if row["method"] == "hurdle"}
    x_state = np.arange(3)
    rng = np.random.default_rng(19)

    # b: temporal share without method terminology on the figure.
    ax_share = fig.add_subplot(top[0, 1])
    mean_shares = 100 * np.asarray([summaries[state]["temporal_share_mean"] for state in STATES])
    ax_share.bar(x_state, mean_shares, width=0.52, color=TEAL)
    for index, state in enumerate(STATES):
        points = 100 * np.asarray([row["temporal_share"] for row in summaries[state]["seed_rows"]])
        ax_share.scatter(np.full(len(points), index) + rng.uniform(-0.035, 0.035, len(points)),
                         points, s=20, facecolor="white", edgecolor=DARK_GRAY,
                         linewidth=0.65, zorder=3)
        ax_share.text(index, mean_shares[index] - 1.25, f"{mean_shares[index]:.1f}%",
                      ha="center", va="top", fontsize=7.5, color="white", fontweight="bold")
    ax_share.set_xticks(x_state, ["Weekday", "Weekend", "Rainy"])
    ax_share.set_ylim(65, 88)
    ax_share.set_ylabel("Share of synergy across time scales (%)")
    panel_label(ax_share, "b", -0.22)

    # c: region-level distribution, elevated to a main evidence panel.
    ax_regions = fig.add_subplot(top[0, 2])
    _, values_by_state = mean_zone_values(finite)
    for state_index, (state, color) in enumerate(zip(STATES, STATE_COLORS)):
        values = values_by_state[state]
        ax_regions.scatter(np.full(len(values), state_index) + rng.uniform(-0.14, 0.14, len(values)),
                           values, s=16, color=color, alpha=0.72,
                           edgecolor="white", linewidth=0.3)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax_regions.vlines(state_index, q1, q3, color="#20282C", lw=4.5, zorder=4)
        ax_regions.scatter(state_index, median, marker="_", s=105,
                           color="white", linewidth=1.2, zorder=5)
    ax_regions.set_xticks(x_state, ["Weekday", "Weekend", "Rainy"])
    ax_regions.set_ylabel("Time-scale synergy (bits)")
    ax_regions.set_ylim(bottom=-0.02)
    panel_label(ax_regions, "c", -0.16)

    # d: three state maps occupy the hero region.
    add_map_panels(fig, outer[1], finite)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(OUTPUT.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
