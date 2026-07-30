#!/usr/bin/env python3
"""Prespecified 2-back versus 0-back association with WM Cont+Default synergy."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results/hcp_cognition_exhaustive_targeted_greedy/metrics.npz"
METRICS_1000 = (
    ROOT
    / "results/hcp_task_score_synergy_schaefer1000_validation"
    / "fixed_candidates_schaefer1000.npz"
)
BEHAVIOR = ROOT / "Data/unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
OUTPUT = ROOT / "results/hcp_wm_back_condition_correlations"

STATE = "WM"
COALITION = "Cont+Default"
CONDITIONS = (
    ("2-back", "WM_Task_2bk_Acc", "#4477AA"),
    ("0-back", "WM_Task_0bk_Acc", "#CC6677"),
)
PERMUTATIONS = 100_000
BOOTSTRAPS = 20_000
SEED = 20260731


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def age_midpoint(value: str) -> float:
    if value == "36+":
        return 38.0
    low, high = value.split("-")
    return 0.5 * (float(low) + float(high))


def unit_rank(values: np.ndarray, design: np.ndarray | None = None) -> np.ndarray:
    ranked = rankdata(np.asarray(values, dtype=float), method="average")
    if design is not None:
        ranked = ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]
    else:
        ranked = ranked - ranked.mean()
    norm = float(np.linalg.norm(ranked))
    if norm <= 1.0e-12:
        raise ValueError("Cannot correlate a constant vector")
    return ranked / norm


def permutation_correlations(
    brain: np.ndarray,
    scores: tuple[np.ndarray, np.ndarray],
    *,
    design: np.ndarray | None,
    rng: np.random.Generator,
) -> dict[str, Any]:
    brain_rank = unit_rank(brain, design)
    score_ranks = tuple(unit_rank(score, design) for score in scores)
    observed = np.asarray([brain_rank @ score for score in score_ranks])
    exceed = np.zeros(2, dtype=np.int64)
    exceed_max = np.zeros(2, dtype=np.int64)
    chunk_size = 2_000
    for start in range(0, PERMUTATIONS, chunk_size):
        size = min(chunk_size, PERMUTATIONS - start)
        indices = np.vstack([rng.permutation(len(brain)) for _ in range(size)])
        null = np.column_stack([
            brain_rank[indices] @ score for score in score_ranks
        ])
        exceed += np.sum(np.abs(null) >= np.abs(observed), axis=0)
        maxima = np.max(np.abs(null), axis=1)
        exceed_max += np.asarray([
            np.count_nonzero(maxima >= abs(value)) for value in observed
        ])
    return {
        "rho": observed,
        "p_permutation": (1 + exceed) / (PERMUTATIONS + 1),
        "p_max_t": (1 + exceed_max) / (PERMUTATIONS + 1),
    }


def paired_difference_test(
    brain: np.ndarray,
    score_2back: np.ndarray,
    score_0back: np.ndarray,
    *,
    design: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[float, float]:
    brain_rank = unit_rank(brain, design)
    difference = unit_rank(score_2back, design) - unit_rank(score_0back, design)
    observed = float(brain_rank @ difference)
    exceed = 0
    chunk_size = 5_000
    for start in range(0, PERMUTATIONS, chunk_size):
        size = min(chunk_size, PERMUTATIONS - start)
        signs = rng.choice((-1.0, 1.0), size=(size, len(brain)))
        null = (signs * difference) @ brain_rank
        exceed += int(np.count_nonzero(np.abs(null) >= abs(observed)))
    return observed, float((1 + exceed) / (PERMUTATIONS + 1))


def bootstrap_statistics(
    brain: np.ndarray,
    score_2back: np.ndarray,
    score_0back: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    estimates = np.empty((BOOTSTRAPS, 3), dtype=float)
    n = len(brain)
    for index in range(BOOTSTRAPS):
        sample = rng.integers(0, n, size=n)
        rho_2 = float(spearmanr(brain[sample], score_2back[sample]).statistic)
        rho_0 = float(spearmanr(brain[sample], score_0back[sample]).statistic)
        estimates[index] = rho_2, rho_0, rho_2 - rho_0
    return {
        "rho_2back_ci95": np.quantile(estimates[:, 0], [0.025, 0.975]).tolist(),
        "rho_0back_ci95": np.quantile(estimates[:, 1], [0.025, 0.975]).tolist(),
        "delta_ci95": np.quantile(estimates[:, 2], [0.025, 0.975]).tolist(),
    }


def leave_one_out(
    brain: np.ndarray, score_2back: np.ndarray, score_0back: np.ndarray
) -> dict[str, list[float]]:
    estimates = np.asarray([
        [
            spearmanr(np.delete(brain, i), np.delete(score_2back, i)).statistic,
            spearmanr(np.delete(brain, i), np.delete(score_0back, i)).statistic,
        ]
        for i in range(len(brain))
    ])
    return {
        "rho_2back_range": [float(estimates[:, 0].min()), float(estimates[:, 0].max())],
        "rho_0back_range": [float(estimates[:, 1].min()), float(estimates[:, 1].max())],
        "delta_range": [
            float((estimates[:, 0] - estimates[:, 1]).min()),
            float((estimates[:, 0] - estimates[:, 1]).max()),
        ],
    }


def plot_scatter(
    ax: plt.Axes,
    score: np.ndarray,
    brain: np.ndarray,
    *,
    label: str,
    color: str,
    rho: float,
    p_value: float,
    n: int,
) -> None:
    ax.scatter(
        score, brain, s=19, color=color, alpha=0.78,
        edgecolor="white", linewidth=0.4, zorder=3,
    )
    grid = np.linspace(float(score.min()), float(score.max()), 100)
    slope, intercept = np.polyfit(score, brain, 1)
    ax.plot(grid, intercept + slope * grid, color="#687386", lw=1.0, ls="--")
    ax.set_xlabel(f"{label} accuracy (%)")
    ax.set_ylabel("Control+Default synergy (bits)")
    ax.text(
        0.03, 0.97,
        rf"$\rho={rho:+.3f}$" + f"\nmax-T $p={p_value:.4f}$\n$n={n}$",
        transform=ax.transAxes, va="top", ha="left", color="#25364A",
    )


def make_figure(
    brain: np.ndarray,
    score_2back: np.ndarray,
    score_0back: np.ndarray,
    summary: dict[str, Any],
) -> None:
    configure_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.35),
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.82]},
        constrained_layout=True,
    )
    raw = summary["primary_common_case"]
    plot_scatter(
        axes[0], score_2back, brain, label="2-back", color=CONDITIONS[0][2],
        rho=raw["rho_2back"], p_value=raw["p_max_t_2back"], n=len(brain),
    )
    plot_scatter(
        axes[1], score_0back, brain, label="0-back", color=CONDITIONS[1][2],
        rho=raw["rho_0back"], p_value=raw["p_max_t_0back"], n=len(brain),
    )

    estimates = np.asarray([raw["rho_2back"], raw["rho_0back"]])
    intervals = np.asarray([
        raw["rho_2back_ci95"], raw["rho_0back_ci95"],
    ])
    y = np.asarray([1.0, 0.0])
    for index, color in enumerate((CONDITIONS[0][2], CONDITIONS[1][2])):
        axes[2].errorbar(
            estimates[index], y[index],
            xerr=np.asarray([
                [estimates[index] - intervals[index, 0]],
                [intervals[index, 1] - estimates[index]],
            ]),
            fmt="o", color=color, ecolor=color, markersize=5,
            elinewidth=1.2, capsize=2.5,
        )
    axes[2].axvline(0.0, color="#B8BEC7", lw=0.8, ls=":")
    axes[2].set_yticks(y, ["2-back", "0-back"])
    axes[2].set_xlim(-1.0, 0.55)
    axes[2].set_xlabel("Spearman correlation\n(95% paired bootstrap CI)")
    axes[2].text(
        0.04, 0.50,
        rf"$\Delta\rho={raw['delta_rho_2back_minus_0back']:+.3f}$"
        + f"\npaired $p={raw['p_condition_difference']:.4f}$",
        transform=axes[2].transAxes, va="center", ha="left",
    )

    for letter, ax in zip(("a", "b", "c"), axes, strict=True):
        ax.text(
            -0.17, 1.04, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom",
        )
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(
            OUTPUT / f"wm_2back_0back_correlation_comparison.{suffix}",
            bbox_inches="tight", **kwargs,
        )
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive = np.load(METRICS)
    states = archive["states"].astype(str).tolist()
    coalitions = archive["coalitions"].astype(str).tolist()
    subjects = [
        str(value).removeprefix("sub-") for value in archive["subjects"].astype(str)
    ]
    brain_all = np.asarray(
        archive["fixed_block_synergy"][
            states.index(STATE), :, coalitions.index(COALITION)
        ],
        dtype=float,
    )
    archive_1000 = np.load(METRICS_1000)
    subjects_1000 = [
        str(value).removeprefix("sub-")
        for value in archive_1000["subjects"].astype(str)
    ]
    if subjects_1000 != subjects:
        raise ValueError("Schaefer-500 and Schaefer-1000 subject orders differ")
    states_1000 = archive_1000["states"].astype(str).tolist()
    coalitions_1000 = archive_1000["coalitions"].astype(str).tolist()
    brain_1000_all = np.asarray(
        archive_1000["values"][
            states_1000.index(STATE),
        ],
        dtype=float,
    )
    if coalitions_1000[states_1000.index(STATE)] != COALITION:
        raise ValueError("Unexpected Schaefer-1000 WM coalition")
    with BEHAVIOR.open(newline="", encoding="utf-8-sig") as handle:
        table = {str(row["Subject"]): row for row in csv.DictReader(handle)}

    complete = np.asarray([
        all(table[subject][field].strip() for _, field, _ in CONDITIONS)
        for subject in subjects
    ])
    common_subjects = [subject for subject, keep in zip(subjects, complete) if keep]
    missing_subjects = [subject for subject, keep in zip(subjects, complete) if not keep]
    brain = brain_all[complete]
    brain_1000 = brain_1000_all[complete]
    score_2back = np.asarray([
        float(table[subject][CONDITIONS[0][1]]) for subject in common_subjects
    ])
    score_0back = np.asarray([
        float(table[subject][CONDITIONS[1][1]]) for subject in common_subjects
    ])
    if len(brain) != 28 or missing_subjects != ["104012"]:
        raise ValueError(
            f"Unexpected complete-case set: n={len(brain)}, missing={missing_subjects}"
        )
    design = np.column_stack([
        np.ones(len(brain)),
        [age_midpoint(table[subject]["Age"]) for subject in common_subjects],
        [table[subject]["Gender"] == "M" for subject in common_subjects],
    ]).astype(float)

    raw = permutation_correlations(
        brain, (score_2back, score_0back),
        design=None, rng=np.random.default_rng(SEED),
    )
    delta, delta_p = paired_difference_test(
        brain, score_2back, score_0back,
        design=None, rng=np.random.default_rng(SEED + 1),
    )
    adjusted = permutation_correlations(
        brain, (score_2back, score_0back),
        design=design, rng=np.random.default_rng(SEED + 2),
    )
    adjusted_delta, adjusted_delta_p = paired_difference_test(
        brain, score_2back, score_0back,
        design=design, rng=np.random.default_rng(SEED + 3),
    )
    bootstrap = bootstrap_statistics(
        brain, score_2back, score_0back, np.random.default_rng(SEED + 4)
    )
    loo = leave_one_out(brain, score_2back, score_0back)

    design_specific = np.column_stack([design, rankdata(score_0back)])
    brain_specific = unit_rank(brain, design_specific)
    score_specific = unit_rank(score_2back, design_specific)
    specific_rho = float(brain_specific @ score_specific)
    exceed = 0
    rng = np.random.default_rng(SEED + 5)
    for _ in range(PERMUTATIONS):
        exceed += int(
            abs(float(brain_specific[rng.permutation(len(brain))] @ score_specific))
            >= abs(specific_rho)
        )
    specific_p = float((1 + exceed) / (PERMUTATIONS + 1))

    raw_1000 = permutation_correlations(
        brain_1000, (score_2back, score_0back),
        design=None, rng=np.random.default_rng(SEED + 100),
    )
    delta_1000, delta_p_1000 = paired_difference_test(
        brain_1000, score_2back, score_0back,
        design=None, rng=np.random.default_rng(SEED + 101),
    )
    adjusted_1000 = permutation_correlations(
        brain_1000, (score_2back, score_0back),
        design=design, rng=np.random.default_rng(SEED + 102),
    )
    adjusted_delta_1000, adjusted_delta_p_1000 = paired_difference_test(
        brain_1000, score_2back, score_0back,
        design=design, rng=np.random.default_rng(SEED + 103),
    )
    bootstrap_1000 = bootstrap_statistics(
        brain_1000, score_2back, score_0back,
        np.random.default_rng(SEED + 104),
    )
    loo_1000 = leave_one_out(brain_1000, score_2back, score_0back)
    brain_specific_1000 = unit_rank(brain_1000, design_specific)
    score_specific_1000 = unit_rank(score_2back, design_specific)
    specific_rho_1000 = float(brain_specific_1000 @ score_specific_1000)
    exceed_1000 = 0
    rng_1000 = np.random.default_rng(SEED + 105)
    for _ in range(PERMUTATIONS):
        exceed_1000 += int(
            abs(
                float(
                    brain_specific_1000[
                        rng_1000.permutation(len(brain_1000))
                    ]
                    @ score_specific_1000
                )
            )
            >= abs(specific_rho_1000)
        )
    specific_p_1000 = float((1 + exceed_1000) / (PERMUTATIONS + 1))

    summary = {
        "experiment": "Prespecified WM 2-back versus 0-back association",
        "brain_measure": "WM Cont+Default fixed-coalition total synergy (Schaefer-500)",
        "brain_measure_is_condition_specific": False,
        "subjects_total": 29,
        "subjects_common_case": len(brain),
        "missing_condition_score_subjects": missing_subjects,
        "permutations": PERMUTATIONS,
        "bootstraps": BOOTSTRAPS,
        "primary_common_case": {
            "rho_2back": float(raw["rho"][0]),
            "p_permutation_2back": float(raw["p_permutation"][0]),
            "p_max_t_2back": float(raw["p_max_t"][0]),
            "rho_0back": float(raw["rho"][1]),
            "p_permutation_0back": float(raw["p_permutation"][1]),
            "p_max_t_0back": float(raw["p_max_t"][1]),
            "delta_rho_2back_minus_0back": delta,
            "p_condition_difference": delta_p,
            **bootstrap,
            **loo,
        },
        "age_sex_adjusted": {
            "rho_2back": float(adjusted["rho"][0]),
            "p_permutation_2back": float(adjusted["p_permutation"][0]),
            "p_max_t_2back": float(adjusted["p_max_t"][0]),
            "rho_0back": float(adjusted["rho"][1]),
            "p_permutation_0back": float(adjusted["p_permutation"][1]),
            "p_max_t_0back": float(adjusted["p_max_t"][1]),
            "delta_rho_2back_minus_0back": adjusted_delta,
            "p_condition_difference": adjusted_delta_p,
        },
        "working_memory_specific_diagnostic": {
            "definition": "2-back adjusted for 0-back accuracy, age, and gender",
            "partial_spearman_rho": specific_rho,
            "p_permutation": specific_p,
        },
        "schaefer1000_validation": {
            "rho_2back": float(raw_1000["rho"][0]),
            "p_permutation_2back": float(raw_1000["p_permutation"][0]),
            "p_max_t_2back": float(raw_1000["p_max_t"][0]),
            "rho_0back": float(raw_1000["rho"][1]),
            "p_permutation_0back": float(raw_1000["p_permutation"][1]),
            "p_max_t_0back": float(raw_1000["p_max_t"][1]),
            "delta_rho_2back_minus_0back": delta_1000,
            "p_condition_difference": delta_p_1000,
            "age_sex_adjusted": {
                "rho_2back": float(adjusted_1000["rho"][0]),
                "p_max_t_2back": float(adjusted_1000["p_max_t"][0]),
                "rho_0back": float(adjusted_1000["rho"][1]),
                "p_max_t_0back": float(adjusted_1000["p_max_t"][1]),
                "delta_rho_2back_minus_0back": adjusted_delta_1000,
                "p_condition_difference": adjusted_delta_p_1000,
            },
            "working_memory_specific_diagnostic": {
                "partial_spearman_rho": specific_rho_1000,
                "p_permutation": specific_p_1000,
            },
            **bootstrap_1000,
            **loo_1000,
        },
        "behavior_diagnostics": {
            "rho_2back_0back": float(spearmanr(score_2back, score_0back).statistic),
            "score_2back_range": [float(score_2back.min()), float(score_2back.max())],
            "score_0back_range": [float(score_0back.min()), float(score_0back.max())],
            "score_2back_unique": int(len(np.unique(score_2back))),
            "score_0back_unique": int(len(np.unique(score_0back))),
        },
        "interpretation_boundary": (
            "The same combined-WM brain metric is used for both behavioral scores. "
            "This tests behavioral condition specificity, not block-specific neural "
            "reconfiguration."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_figure(brain, score_2back, score_0back, summary)

    primary = summary["primary_common_case"]
    adjusted_summary = summary["age_sex_adjusted"]
    specific = summary["working_memory_specific_diagnostic"]
    validation_1000 = summary["schaefer1000_validation"]
    lines = [
        "# WM 2-back versus 0-back condition comparison",
        "",
        "The `Cont+Default` coalition and all analysis choices were frozen before "
        "condition-specific testing. Both primary correlations use the same 28 "
        "complete-case subjects; subject 104012 lacks a 2-back score.",
        "",
        "| Analysis | 2-back | 0-back | Δρ (2-back − 0-back) | Difference p |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Primary Spearman | {primary['rho_2back']:+.3f} "
            f"(max-T p={primary['p_max_t_2back']:.4f}) | "
            f"{primary['rho_0back']:+.3f} "
            f"(max-T p={primary['p_max_t_0back']:.4f}) | "
            f"{primary['delta_rho_2back_minus_0back']:+.3f} | "
            f"{primary['p_condition_difference']:.4f} |"
        ),
        (
            f"| Age/sex-adjusted rank correlation | "
            f"{adjusted_summary['rho_2back']:+.3f} "
            f"(max-T p={adjusted_summary['p_max_t_2back']:.4f}) | "
            f"{adjusted_summary['rho_0back']:+.3f} "
            f"(max-T p={adjusted_summary['p_max_t_0back']:.4f}) | "
            f"{adjusted_summary['delta_rho_2back_minus_0back']:+.3f} | "
            f"{adjusted_summary['p_condition_difference']:.4f} |"
        ),
        (
            f"| Schaefer-1000 fixed validation | "
            f"{validation_1000['rho_2back']:+.3f} "
            f"(max-T p={validation_1000['p_max_t_2back']:.4f}) | "
            f"{validation_1000['rho_0back']:+.3f} "
            f"(max-T p={validation_1000['p_max_t_0back']:.4f}) | "
            f"{validation_1000['delta_rho_2back_minus_0back']:+.3f} | "
            f"{validation_1000['p_condition_difference']:.4f} |"
        ),
        "",
        "## Working-memory-specific diagnostic",
        "",
        "After additionally adjusting 2-back accuracy for 0-back accuracy, age, "
        f"and gender, partial Spearman ρ={specific['partial_spearman_rho']:+.3f}, "
        f"permutation p={specific['p_permutation']:.4f}.",
        "",
        "## Interpretation",
        "",
        "The negative association is concentrated in 0-back performance and "
        "replicates across Schaefer-500 and Schaefer-1000. The paired condition "
        "difference is significant in the frozen Schaefer-1000 validation, but "
        "not in Schaefer-500. The current brain metric was estimated from the "
        "combined WM run; block-specific neural claims still require separate "
        "0-back and 2-back brain-metric estimation.",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
