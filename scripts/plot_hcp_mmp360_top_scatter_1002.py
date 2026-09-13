#!/usr/bin/env python3
"""Plot the two strongest corrected MMP360 same-task associations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.screen_hcp_mmp360_same_task_scores_1002 import (
    CACHE,
    SUMMARY,
    TASKS,
    compact,
    design_for,
    load_endpoints,
    residualize,
    task_mask,
)


OUTPUT = ROOT / "results/hcp_mmp360_same_task_score_screen_1002"
FIGURE = OUTPUT / "top_two_same_task_scatter_1002.png"
SELECTED_TASKS = ("RELATIONAL", "LANGUAGE")


def standardized_rank_residual(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    residual = residualize(rankdata(values), design)
    scale = residual.std(ddof=1)
    if not np.isfinite(scale) or scale <= 1.0e-12:
        raise ValueError("Cannot standardize a constant residualized variable.")
    return residual / scale


def main() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    winners = {row["task"]: row for row in payload["task_winners"]}
    with np.load(CACHE, allow_pickle=False) as archive:
        subjects = archive["subjects"].astype(str)
        tasks = archive["tasks"].astype(str)
        coalitions = archive["coalitions"].astype(str)
        synergy = archive["synergy_bits"].astype(float)

    endpoints, age, sex = load_endpoints(subjects)
    colors = {"RELATIONAL": "#66788A", "LANGUAGE": "#4C78A8"}
    panel_data = []
    for task in SELECTED_TASKS:
        winner = winners[task]
        task_index = int(np.flatnonzero(tasks == task)[0])
        coalition_index = int(np.flatnonzero(coalitions == winner["coalition"])[0])
        specs = endpoints[task]
        spec = specs[int(winner["endpoint_index"])]
        mask = task_mask(specs, age, sex)
        design = design_for(spec, age, sex, mask)
        x = standardized_rank_residual(synergy[task_index, mask, coalition_index], design)
        y = standardized_rank_residual(spec["values"][mask], design)
        observed = float(np.corrcoef(x, y)[0, 1])
        if not np.isclose(observed, winner["rho"], atol=1.0e-10):
            raise AssertionError(f"Plotted rho {observed} differs from summary {winner['rho']}.")
        panel_data.append((task, winner, x, y))

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.45), sharex=True, sharey=True, layout="constrained")
    for panel_index, (axis, (task, winner, x, y)) in enumerate(zip(axes, panel_data)):
        color = colors[task]
        axis.scatter(
            x,
            y,
            s=10,
            alpha=0.27,
            color=color,
            edgecolors="none",
            rasterized=True,
        )
        x_line = np.linspace(-3.5, 3.5, 200)
        axis.plot(x_line, winner["rho"] * x_line, color=color, linewidth=2.0)
        axis.axhline(0, color="#D9DDE2", linewidth=0.7, zorder=0)
        axis.axvline(0, color="#D9DDE2", linewidth=0.7, zorder=0)
        ci = winner["bootstrap_95_ci"]
        annotation = (
            rf"$n$ = {winner['n']:,}" "\n"
            rf"adjusted $\rho$ = {winner['rho']:+.3f}" "\n"
            rf"95% CI [{ci[0]:+.3f}, {ci[2]:+.3f}]" "\n"
            rf"task-family max-$T$ $p$ = {winner['p_max_t_task_family']:.5f}" "\n"
            rf"seven-task Holm $p$ = {winner['p_holm_across_task_winners']:.5f}"
        )
        axis.text(
            0.04,
            0.96,
            annotation,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )
        axis.set_title(
            f"{task.title()} · {winner['endpoint']}\n{compact(winner['coalition'])}",
            loc="left",
            fontsize=9.5,
            fontweight="bold",
            pad=8,
        )
        axis.text(-0.12, 1.08, chr(ord("a") + panel_index), transform=axis.transAxes, fontsize=12, fontweight="bold")
        axis.set_xlim(-3.6, 3.6)
        axis.set_ylim(-3.6, 3.6)
        axis.set_xlabel("Coalition Syn adjusted rank residual (SD)")
    axes[0].set_ylabel("Behavior adjusted rank residual (SD)")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(FIGURE)


if __name__ == "__main__":
    main()
