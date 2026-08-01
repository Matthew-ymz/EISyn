#!/usr/bin/env python3
"""Draw the six-panel Schaefer-1000 validation overview for the 57-subject cohort."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "hcp_schaefer1000_57_validation_suite"
OUTPUT = SOURCE / "final"
NETWORKS = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
TASKS = ("EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")


def load(name: str, filename: str = "summary.json") -> dict:
    return json.loads((SOURCE / name / filename).read_text(encoding="utf-8"))


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def annotate_heatmap(ax: plt.Axes, values: np.ndarray, fmt: str, threshold: float) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            color = "white" if abs(values[row, col]) >= threshold else "black"
            ax.text(col, row, format(values[row, col], fmt), ha="center", va="center", fontsize=7, color=color)


def core_name(sources: list[str]) -> str:
    missing = [name for name in NETWORKS if name not in sources]
    if not missing:
        return "All 7"
    if len(missing) <= 2:
        return "−" + ",".join(missing)
    return "+".join(sources)


def main() -> None:
    null = load("null")
    module = load("module")
    robustness = load("robustness")
    prediction = load("prediction", "prediction_error_summary.json")
    tevf = load("tevf")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(3, 2, figsize=(11.4, 10.2), constrained_layout=True)

    ax = axes[0, 0]
    delta = np.asarray(
        [row["null_comparison"]["observed_minus_null_mean"] for row in null["rows"]], dtype=float
    )
    rng = np.random.default_rng(20260801)
    ax.violinplot(delta, positions=[0], widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    ax.scatter(rng.normal(0, 0.055, len(delta)), delta, s=17, color="#2878B5", alpha=0.72, edgecolors="none")
    ax.scatter([0], [delta.mean()], marker="D", s=48, facecolor="white", edgecolor="#222222", zorder=4)
    ax.axhline(0, color="#555555", linestyle="--", linewidth=0.9)
    ax.set(xticks=[0], xticklabels=["REST"], ylabel=r"Observed $-$ circular-shift null mean (bits)")
    ax.set_title(f"REST null: {np.sum(delta > 0)}/{len(delta)} above null mean")
    panel_label(ax, "a")

    ax = axes[0, 1]
    cores = module["core_summary"][:6]
    matched = module["null_rank_comparison"][:6]
    observed = np.asarray([item["top_frequency"] for item in cores], dtype=float)
    null_mean = np.asarray([item["null_frequency_mean"] for item in matched], dtype=float)
    null_min = np.asarray([min(item["frequency_by_replicate"]) for item in matched], dtype=float)
    null_max = np.asarray([max(item["frequency_by_replicate"]) for item in matched], dtype=float)
    positions = np.arange(len(cores))
    ax.bar(positions - 0.18, observed, width=0.36, color="#D95F02", label="Observed")
    ax.bar(positions + 0.18, null_mean, width=0.36, color="#7F8C8D", label="Matched-null mean")
    ax.errorbar(positions + 0.18, null_mean, yerr=[null_mean - null_min, null_max - null_mean], fmt="none", ecolor="#333333", capsize=2, linewidth=0.8)
    ax.set(xticks=positions, xticklabels=[core_name(item["sources"]) for item in cores], ylabel="Top-3 frequency (subjects)")
    ax.tick_params(axis="x", rotation=28)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_title("Greedy module cores versus matched null")
    panel_label(ax, "b")

    orders = [1, 2, 3, 5, 8]
    alphas = [0.1, 1, 10, 100, 1000]
    ax = axes[1, 0]
    margin = np.full((len(orders), len(alphas)), np.nan)
    for item in robustness["grid_points"]:
        margin[orders.index(int(item["order"])), alphas.index(float(item["alpha"]))] = item["minimum_rest_minus_task_mean"]
    bound = float(np.max(np.abs(margin)))
    image = ax.imshow(margin, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    annotate_heatmap(ax, margin, ".2f", 0.58 * bound)
    ax.set(xticks=range(5), xticklabels=[f"{v:g}" for v in alphas], yticks=range(5), yticklabels=orders, xlabel=r"Ridge $\alpha$", ylabel="History order $p$")
    ax.set_title("Minimum REST−task mean margin")
    fig.colorbar(image, ax=ax, shrink=0.78, pad=0.02, label="bits")
    panel_label(ax, "c")

    ax = axes[1, 1]
    error = np.full((len(orders), len(alphas)), np.nan)
    for item in prediction["grid_points"]:
        error[orders.index(int(item["order"])), alphas.index(float(item["alpha"]))] = item["overall"]["test_delta_nrmse"]
    image = ax.imshow(error, cmap="magma_r", aspect="auto")
    annotate_heatmap(ax, error, ".3f", float(np.quantile(np.abs(error), 0.7)))
    ax.set(xticks=range(5), xticklabels=[f"{v:g}" for v in alphas], yticks=range(5), yticklabels=orders, xlabel=r"Ridge $\alpha$", ylabel="History order $p$")
    ax.set_title("Held-out prediction error")
    fig.colorbar(image, ax=ax, shrink=0.78, pad=0.02, label="delta-NRMSE")
    panel_label(ax, "d")

    ax = axes[2, 0]
    means = 100 * np.asarray([tevf["mean_task_evoked_fraction"][task] for task in TASKS])
    ax.bar(np.arange(7), means, color="#3A9D8F")
    ax.set(xticks=np.arange(7), xticklabels=[name.title() for name in TASKS], ylabel="Mean TEVF (%)")
    ax.tick_params(axis="x", rotation=28)
    ax.set_title("Task-evoked variance fraction")
    panel_label(ax, "e")

    ax = axes[2, 1]
    accuracies = 100 * np.asarray([
        tevf["classification"]["parcel"]["accuracy"],
        tevf["classification"]["network"]["accuracy"],
        tevf["rest_comparison"]["classification"]["parcel"]["accuracy"],
        tevf["rest_comparison"]["classification"]["network"]["accuracy"],
    ])
    chance = np.asarray([100 / 7, 100 / 7, 12.5, 12.5])
    positions = np.arange(4)
    ax.bar(positions, accuracies, color=["#4C78A8", "#9ECAE1", "#E45756", "#F2A7A0"])
    ax.scatter(positions, chance, marker="_", s=340, linewidth=2, color="#222222", label="Chance")
    ax.set(xticks=positions, xticklabels=["TEVF\nparcels", "TEVF\nYeo7", "8 states\nparcels", "8 states\nYeo7"], ylabel="LOSO accuracy (%)", ylim=(0, 100))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_title("Spatial-map discriminability")
    panel_label(ax, "f")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUTPUT / f"hcp_schaefer1000_validation_overview_57.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
