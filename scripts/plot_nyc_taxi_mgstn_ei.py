#!/usr/bin/env python3
"""Create the NYC Taxi main figure from the finite quadratic-TM audit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPRODUCTION = ROOT / "results/nyc_taxi_mgstn/summary.json"
FINITE = ROOT / "results/nyc_taxi_mgstn_ei/finite_quadratic_tm/quadratic_tm_full_summary.json"
OUTPUT = ROOT / "fig/nyc_taxi_social_multiscale_main"
STATES = ["weekday_peak", "weekend_midday", "rainy_high_demand"]
STATE_LABELS = ["Weekday\npeak", "Weekend\nmidday", "Rainy\nhigh-demand"]
METHODS = ["hurdle"]
METHOD_LABELS = ["Quadratic hurdle TM"]
SPARSE_ZONE_IDS = {120, 127, 128, 153, 194, 202}
TOLERANCE_BITS = 0.05
BLUE, TEAL, GOLD = "#477AA8", "#4F918B", "#D7AF62"
DARK_GRAY = "#59656D"
STATE_COLORS = ["#355C7D", "#4F918B", "#C7894A"]


def interpreted(value: float) -> float:
    return 0.0 if -TOLERANCE_BITS <= value < 0 else value


def seed_zone_values(payload: dict, state: str, method: str) -> dict[int, dict[int, float]]:
    values: dict[int, dict[int, float]] = {}
    for unit in payload["units"]:
        if unit["state"] != state:
            continue
        values.setdefault(int(unit["model_seed"]), {})[int(unit["zone_id"])] = interpreted(
            float(unit[f"{method}_temporal"]["syn_bits_raw"])
        )
    return values


def panel_label(ax, label: str, x: float = -0.14) -> None:
    ax.text(x, 1.08, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")


def main() -> None:
    reproduction = json.loads(REPRODUCTION.read_text(encoding="utf-8"))
    finite = json.loads(FINITE.read_text(encoding="utf-8"))
    if finite["nonnegative_audit"]["hurdle"]["violation_count"]:
        raise RuntimeError("formal hurdle quadratic TM failed its declared nonnegative audit")
    if len(finite["units"]) != 66 * 3 * 3:
        raise RuntimeError(f"expected 594 units, found {len(finite['units'])}")

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.5, "axes.labelsize": 10, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(13.4, 7.4), constrained_layout=True)
    outer = fig.add_gridspec(2, 2, width_ratios=[1.42, 1.0], wspace=0.08, hspace=0.08)

    # a: reproduced forecasting accuracy and training stability.
    validation = outer[0, 0].subgridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.20)
    ax_error = fig.add_subplot(validation[0, 0])
    keys = ["inflow_mae", "inflow_rmse", "outflow_mae", "outflow_rmse"]
    labels = ["Inflow\nMAE", "Inflow\nRMSE", "Outflow\nMAE", "Outflow\nRMSE"]
    x_metric = np.arange(4)
    run_values = np.asarray([[run["test"][key] for key in keys] for run in reproduction["runs"]])
    means, deviations = run_values.mean(axis=0), run_values.std(axis=0, ddof=1)
    for seed_index, values in enumerate(run_values):
        ax_error.scatter(x_metric + (seed_index - 1) * 0.055, values, s=24,
                         facecolor="white", edgecolor=BLUE, linewidth=0.8, zorder=3)
    ax_error.errorbar(x_metric, means, yerr=deviations, fmt="o", ms=6.5, color=BLUE,
                      ecolor=BLUE, elinewidth=1.1, capsize=3, zorder=4)
    for index, value in enumerate(means):
        ax_error.text(index, value + 0.55, f"{value:.2f}", ha="center", va="bottom",
                      fontsize=8.5, fontweight="bold")
    ax_error.set_xticks(x_metric, labels)
    ax_error.set_ylim(0, 18.2)
    ax_error.set_ylabel("Error (rides per zone-hour)")
    panel_label(ax_error, "a", -0.18)

    ax_curve = fig.add_subplot(validation[0, 1])
    for color, run in zip(["#2B7BBB", "#7667A8", "#D45B52"], reproduction["runs"]):
        epochs = np.asarray([row["epoch"] for row in run["history"]])
        losses = np.asarray([row["normalized_mse"] for row in run["history"]])
        ax_curve.plot(epochs, losses, color=color, lw=1.5, label=f"seed {run['seed']}")
        ax_curve.scatter(int(run["best_epoch"]), float(run["best_validation_normalized_mse"]),
                         s=32, color=color, edgecolor="white", linewidth=0.6, zorder=3)
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Validation normalized MSE")
    ax_curve.set_ylim(0.235, 0.42)
    ax_curve.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    # b: temporal share under two finite-amplitude representations.
    ax_share = fig.add_subplot(outer[0, 1])
    x_state, width = np.arange(3), 0.48
    rng = np.random.default_rng(19)
    summaries_all = {(row["method"], row["state"]): row for row in finite["summary"]}
    for method, label, color in zip(METHODS, METHOD_LABELS, [TEAL]):
        xpos = x_state
        means_share = 100 * np.asarray([summaries_all[method, state]["temporal_share_mean"] for state in STATES])
        ax_share.bar(xpos, means_share, width=width * 0.88, color=color, label=label)
        for index, state in enumerate(STATES):
            points = 100 * np.asarray([row["temporal_share"] for row in summaries_all[method, state]["seed_rows"]])
            ax_share.scatter(np.full(len(points), xpos[index]) + rng.uniform(-0.025, 0.025, len(points)),
                             points, s=23, facecolor="white", edgecolor=DARK_GRAY, linewidth=0.7, zorder=3)
    ax_share.set_xticks(x_state, STATE_LABELS)
    ax_share.set_ylim(65, 95)
    ax_share.set_ylabel("Temporal share of target-wise synergy (%)")
    ax_share.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    panel_label(ax_share, "b", -0.13)

    # c: one point per region after averaging the three model seeds.
    ax_regions = fig.add_subplot(outer[1, 0])
    for state_index, (state, color) in enumerate(zip(STATES, STATE_COLORS)):
        by_seed = seed_zone_values(finite, state, "hurdle")
        zone_ids = sorted(next(iter(by_seed.values())))
        values = np.asarray([np.mean([by_seed[seed][zone] for seed in sorted(by_seed)]) for zone in zone_ids])
        ax_regions.scatter(np.full(len(values), state_index) + rng.uniform(-0.16, 0.16, len(values)),
                           values, s=18, color=color, alpha=0.68, edgecolor="white", linewidth=0.35)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax_regions.vlines(state_index, q1, q3, color="#20282C", lw=5, zorder=4)
        ax_regions.scatter(state_index, median, marker="_", s=130, color="white", linewidth=1.4, zorder=5)
    ax_regions.set_xticks(x_state, STATE_LABELS)
    ax_regions.set_ylabel("Region-level recent–macro synergy (bits)")
    ax_regions.set_ylim(bottom=-0.02)
    panel_label(ax_regions, "c", -0.09)

    # d: contribution of six predeclared sparse zones to the temporal total.
    ax_sparse = fig.add_subplot(outer[1, 1])
    for method, label, color in zip(METHODS, METHOD_LABELS, [TEAL]):
        xpos = x_state
        contributions = 100 * np.asarray([summaries_all[method, state]["sparse_temporal_share_mean"] for state in STATES])
        ax_sparse.bar(xpos, contributions, width=width * 0.88, color=color, label=label)
        for index, state in enumerate(STATES):
            points = 100 * np.asarray([row["sparse_temporal_share"] for row in summaries_all[method, state]["seed_rows"]])
            ax_sparse.scatter(np.full(len(points), xpos[index]) + rng.uniform(-0.025, 0.025, len(points)),
                              points, s=23, facecolor="white", edgecolor=DARK_GRAY, linewidth=0.7, zorder=3)
    ax_sparse.axhline(100 * len(SPARSE_ZONE_IDS) / 66, color=DARK_GRAY, lw=1.0, ls=(0, (4, 3)))
    ax_sparse.set_xticks(x_state, STATE_LABELS)
    ax_sparse.set_ylim(0, 10)
    ax_sparse.set_ylabel("Contribution from six sparse zones (%)")
    panel_label(ax_sparse, "d", -0.13)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(OUTPUT.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
