#!/usr/bin/env python3
"""Plot the two selection-corrected exploratory MMP360 associations."""

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

from scripts.validate_hcp_mmp360_behavior_main_1002 import (
    CACHE,
    OUTPUT,
    SUMMARY,
    TASKS,
    design_for,
    endpoint_for,
    load_behavior,
    load_subjects,
    residualize,
)


FIGURE = OUTPUT / "hcp_mmp360_exploratory_winners_scatter_1002.png"


def standardized_rank_residuals(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    residuals = residualize(rankdata(values), design)
    return (residuals - residuals.mean()) / residuals.std(ddof=1)


def regression_band(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(np.quantile(x, 0.005), np.quantile(x, 0.995), 240)
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ beta
    residual_sd = np.sqrt(np.sum((y - fitted) ** 2) / (len(x) - 2))
    centered = x - x.mean()
    se = residual_sd * np.sqrt(1.0 / len(x) + (grid - x.mean()) ** 2 / np.sum(centered**2))
    mean = beta[0] + beta[1] * grid
    return grid, mean, 1.96 * se


def main() -> None:
    subjects, _ = load_subjects()
    behavior = load_behavior(subjects)
    with np.load(CACHE, allow_pickle=False) as archive:
        cached_subjects = archive["subjects"].astype(str)
        names = archive["coalitions"].astype(str)
        matrices = archive["synergy_bits"].astype(float)
    if not np.array_equal(subjects, cached_subjects):
        raise ValueError("Subject order differs between the cache and behavior table.")

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    winners = summary["samples"]["full_1002_mmp360"]["winners"]
    specs = (
        ("social", "SOCIAL", "social_dprime", "Limbic+Cont", "Social discrimination $d'$", "#397A70"),
        (
            "emotion",
            "EMOTION",
            "face_speed",
            "Vis+DorsAttn+SalVentAttn+Default",
            "Face-matching speed\n(adjusted for shape speed)",
            "#5B6FA8",
        ),
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), layout="constrained")
    task_index = {task: index for index, task in enumerate(TASKS)}

    for panel, (ax, (key, task, endpoint_name, coalition, endpoint_label, color)) in enumerate(
        zip(axes, specs, strict=True)
    ):
        if winners[key]["coalition"] != coalition:
            raise ValueError(f"Unexpected {key} winner: {winners[key]['coalition']}")
        coalition_index = int(np.flatnonzero(names == coalition)[0])
        take = np.arange(len(subjects))
        design, _ = design_for(endpoint_name, behavior, take, None)
        x = standardized_rank_residuals(matrices[task_index[task], :, coalition_index], design)
        y = standardized_rank_residuals(endpoint_for(endpoint_name, behavior, take), design)
        rho = float(np.corrcoef(x, y)[0, 1])
        if not np.isclose(rho, winners[key]["rho"], atol=1.0e-12):
            raise ValueError(f"Plotted {key} rho {rho} does not match summary {winners[key]['rho']}.")

        ax.scatter(x, y, s=8.0, color=color, alpha=0.25, edgecolors="none", rasterized=True)
        grid, mean, half_width = regression_band(x, y)
        ax.fill_between(grid, mean - half_width, mean + half_width, color=color, alpha=0.13, linewidth=0)
        ax.plot(grid, mean, color=color, linewidth=1.5)
        ax.axhline(0, color="#D9DEE2", linewidth=0.6, zorder=0)
        ax.axvline(0, color="#D9DEE2", linewidth=0.6, zorder=0)
        ax.set_xlabel(f"{coalition.replace('DorsAttn', 'DAN').replace('SalVentAttn', 'SVAN').replace('Default', 'DMN')} Syn\n(residualized rank, SD)")
        ax.set_ylabel(f"{endpoint_label}\n(residualized rank, SD)")
        ax.set_title(f"{'ab'[panel]}  {task.title()}", loc="left", fontweight="bold")
        ax.text(
            0.97,
            0.96,
            rf"$\rho_{{adj}}={rho:.3f}$" + "\n" + rf"max-$T$ $p={winners[key]['p_max_t_120']:.3f}$" + "\n" + r"$n=1{,}002$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
        )
        ax.set_xlim(-3.9, 3.9)
        ax.set_ylim(-3.9, 3.9)

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(FIGURE)


if __name__ == "__main__":
    main()
