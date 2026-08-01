#!/usr/bin/env python3
"""Validate LANGUAGE coalition synergy against adaptive Math difficulty."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
VALUES = (
    ROOT
    / "results"
    / "hcp_schaefer1000_panels_e_i_57"
    / "panel_values_57.npz"
)
OLD_VALUES = (
    ROOT
    / "results"
    / "hcp_language_story_math_candidates_schaefer1000_replication"
    / "fixed_candidates_schaefer1000.npz"
)
BEHAVIOR = ROOT / "data" / "unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
OUTPUT = ROOT / "results" / "hcp_language_math_difficulty_57"
SEED = 20260731
PERMUTATIONS = 100_000
BOOTSTRAPS = 20_000

BRAIN_KEYS = ("language_all_network_atom", "language_fixed_coalition")
BRAIN_LABELS = ("All-network atom", "SomMot+Limbic+Cont")
BEHAVIOR_COLUMNS = (
    "Language_Task_Acc",
    "Language_Task_Story_Acc",
    "Language_Task_Math_Acc",
    "Language_Task_Story_Minus_Math_Acc",
    "Language_Task_Story_Avg_Difficulty_Level",
    "Language_Task_Math_Avg_Difficulty_Level",
)
BEHAVIOR_LABELS = (
    "Total accuracy",
    "Story accuracy",
    "Math accuracy",
    "Story-minus-Math accuracy",
    "Story average difficulty",
    "Math average difficulty",
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def standardized_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    result = np.empty_like(values)
    for column in range(values.shape[1]):
        ranks = rankdata(values[:, column], method="average")
        scale = float(ranks.std(ddof=0))
        if scale <= 0:
            raise ValueError("Ranks have zero variance.")
        result[:, column] = (ranks - ranks.mean()) / scale
    return result


def blocked_permutation(
    x: np.ndarray,
    y: np.ndarray,
    cohorts: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    ranks_x = standardized_ranks(x)[:, 0]
    ranks_y = standardized_ranks(y)[:, 0]
    observed = float(np.mean(ranks_x * ranks_y))
    null = np.empty(permutations, dtype=float)
    cohort_indices = [
        np.flatnonzero(cohorts == cohort) for cohort in np.unique(cohorts)
    ]
    for draw in range(permutations):
        permuted = ranks_y.copy()
        for indices in cohort_indices:
            permuted[indices] = ranks_y[rng.permutation(indices)]
        null[draw] = float(np.mean(ranks_x * permuted))
    p_value = float(
        (1 + np.count_nonzero(np.abs(null) >= abs(observed)))
        / (permutations + 1)
    )
    return observed, p_value


def language_family_max_t(
    brain: np.ndarray,
    behavior: np.ndarray,
    cohorts: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    brain_ranks = standardized_ranks(brain)
    behavior_ranks = standardized_ranks(behavior)
    observed = brain_ranks.T @ behavior_ranks / len(cohorts)
    maximum_null = np.empty(permutations, dtype=float)
    cohort_indices = [
        np.flatnonzero(cohorts == cohort) for cohort in np.unique(cohorts)
    ]
    for draw in range(permutations):
        permutation = np.arange(len(cohorts))
        for indices in cohort_indices:
            permutation[indices] = rng.permutation(indices)
        permuted = brain_ranks.T @ behavior_ranks[permutation] / len(cohorts)
        maximum_null[draw] = float(np.max(np.abs(permuted)))
    adjusted = np.empty_like(observed)
    for row in range(observed.shape[0]):
        for column in range(observed.shape[1]):
            adjusted[row, column] = (
                1
                + np.count_nonzero(
                    maximum_null >= abs(float(observed[row, column]))
                )
            ) / (permutations + 1)
    return observed, adjusted


def bootstrap_primary(
    x: np.ndarray,
    y: np.ndarray,
    cohorts: np.ndarray,
    *,
    bootstraps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    values = np.empty(bootstraps, dtype=float)
    cohort_indices = [
        np.flatnonzero(cohorts == cohort) for cohort in np.unique(cohorts)
    ]
    for draw in range(bootstraps):
        sampled_groups = [
            rng.choice(indices, size=len(indices), replace=True)
            for indices in cohort_indices
        ]
        sampled = np.concatenate(sampled_groups)
        values[draw] = float(
            spearmanr(x[sampled], y[sampled]).statistic
        )
    return {
        "rho_ci95": np.quantile(values, [0.025, 0.975]).tolist(),
    }


def leave_one_out(
    x: np.ndarray, y: np.ndarray, cohorts: np.ndarray
) -> list[float]:
    values = []
    for index in range(len(x)):
        keep = np.arange(len(x)) != index
        values.append(float(spearmanr(x[keep], y[keep]).statistic))
    return values


def partial_spearman(
    x: np.ndarray, y: np.ndarray, covariate: np.ndarray
) -> float:
    ranks_x = rankdata(x, method="average")
    ranks_y = rankdata(y, method="average")
    ranks_c = rankdata(covariate, method="average")
    design = np.column_stack([np.ones(len(x)), ranks_c])
    residual_x = ranks_x - design @ np.linalg.lstsq(
        design, ranks_x, rcond=None
    )[0]
    residual_y = ranks_y - design @ np.linalg.lstsq(
        design, ranks_y, rcond=None
    )[0]
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    with np.load(VALUES, allow_pickle=False) as archive:
        subjects = archive["subjects"].astype(str)
        fixed = archive["language_fixed_coalition"].astype(float)
        brain = np.column_stack(
            [archive[key].astype(float) for key in BRAIN_KEYS]
        )
    with np.load(OLD_VALUES, allow_pickle=False) as archive:
        old_subjects = archive["subjects"].astype(str)
    if not np.array_equal(subjects[:29], old_subjects):
        raise ValueError("The first 29 subjects do not match the frozen cohort.")
    if len(subjects) != 57 or len(set(subjects.tolist())) != 57:
        raise ValueError("Expected 57 unique subjects.")
    behavior = pd.read_csv(BEHAVIOR, dtype={"Subject": str})
    behavior["Subject"] = behavior["Subject"].str.removeprefix("sub-")
    ordered = behavior.set_index("Subject").loc[
        [subject.removeprefix("sub-") for subject in subjects]
    ].copy()
    ordered["Language_Task_Story_Minus_Math_Acc"] = (
        ordered["Language_Task_Story_Acc"]
        - ordered["Language_Task_Math_Acc"]
    )
    behavior_values = ordered.loc[:, BEHAVIOR_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(fixed).all():
        raise ValueError("Fixed-coalition values must be finite.")
    if not np.isfinite(brain).all() or not np.isfinite(behavior_values).all():
        raise ValueError("LANGUAGE validation variables must be finite.")
    cohorts = np.r_[np.zeros(29, dtype=int), np.ones(28, dtype=int)]
    frame = pd.DataFrame(
        {
            "subject": subjects,
            "cohort": np.where(
                cohorts == 0, "original_29", "supplementary_28"
            ),
            "math_accuracy": ordered["Language_Task_Math_Acc"].to_numpy(
                dtype=float
            ),
            "math_average_difficulty": ordered[
                "Language_Task_Math_Avg_Difficulty_Level"
            ].to_numpy(dtype=float),
            "fixed_coalition_synergy_bits": fixed,
        }
    )
    return frame, brain, behavior_values


def analyze(
    frame: pd.DataFrame, brain: np.ndarray, behavior: np.ndarray
) -> dict[str, Any]:
    cohorts = np.r_[np.zeros(29, dtype=int), np.ones(28, dtype=int)]
    synergy = frame["fixed_coalition_synergy_bits"].to_numpy(dtype=float)
    difficulty = frame["math_average_difficulty"].to_numpy(dtype=float)
    accuracy = frame["math_accuracy"].to_numpy(dtype=float)
    primary_rho, primary_p = blocked_permutation(
        synergy,
        difficulty,
        cohorts,
        permutations=PERMUTATIONS,
        rng=np.random.default_rng(SEED),
    )
    family_rho, family_p = language_family_max_t(
        brain,
        behavior,
        cohorts,
        permutations=PERMUTATIONS,
        rng=np.random.default_rng(SEED + 1),
    )
    target_row = BRAIN_KEYS.index("language_fixed_coalition")
    target_column = BEHAVIOR_COLUMNS.index(
        "Language_Task_Math_Avg_Difficulty_Level"
    )
    bootstrap = bootstrap_primary(
        synergy,
        difficulty,
        cohorts,
        bootstraps=BOOTSTRAPS,
        rng=np.random.default_rng(SEED + 2),
    )
    loo = leave_one_out(synergy, difficulty, cohorts)
    cohort_results: dict[str, Any] = {}
    for cohort, label in ((0, "original_29"), (1, "supplementary_28")):
        mask = cohorts == cohort
        association = spearmanr(synergy[mask], difficulty[mask])
        accuracy_association = spearmanr(synergy[mask], accuracy[mask])
        cohort_results[label] = {
            "n": int(mask.sum()),
            "rho_math_difficulty": float(association.statistic),
            "p_math_difficulty_raw": float(association.pvalue),
            "rho_math_accuracy": float(accuracy_association.statistic),
            "p_math_accuracy_raw": float(accuracy_association.pvalue),
            "partial_rho_difficulty_controlling_accuracy": partial_spearman(
                synergy[mask], difficulty[mask], accuracy[mask]
            ),
        }
    pooled_difficulty = spearmanr(synergy, difficulty)
    pooled_accuracy = spearmanr(synergy, accuracy)
    accuracy_difficulty = spearmanr(accuracy, difficulty)
    return {
        "analysis": "LANGUAGE SomMot+Limbic+Cont synergy versus adaptive Math difficulty",
        "exploratory": True,
        "n": 57,
        "cohorts": cohort_results,
        "primary": {
            "rho_pooled": primary_rho,
            "p_blocked_permutation_two_sided": primary_p,
            "permutations": PERMUTATIONS,
            "p_pooled_raw": float(pooled_difficulty.pvalue),
            "language_family_max_t_p": float(
                family_p[target_row, target_column]
            ),
            "language_family_size": int(family_p.size),
            **bootstrap,
            "leave_one_out_rho_range": [
                float(np.min(loo)),
                float(np.max(loo)),
            ],
        },
        "diagnostics": {
            "rho_math_accuracy_pooled": float(pooled_accuracy.statistic),
            "p_math_accuracy_pooled": float(pooled_accuracy.pvalue),
            "rho_accuracy_difficulty": float(accuracy_difficulty.statistic),
            "p_accuracy_difficulty": float(accuracy_difficulty.pvalue),
            "partial_rho_difficulty_controlling_accuracy_pooled": (
                partial_spearman(synergy, difficulty, accuracy)
            ),
            "family_rho": family_rho.tolist(),
            "family_max_t_p": family_p.tolist(),
            "brain_labels": list(BRAIN_LABELS),
            "behavior_labels": list(BEHAVIOR_LABELS),
        },
        "limitations": [
            "The endpoint was selected after inspection of the expanded data.",
            "The supplementary cohort was enriched for behavioral diversity.",
            "The brain metric uses the full LANGUAGE run, not Math-only blocks.",
        ],
    }


def plot_validation(frame: pd.DataFrame, summary: dict[str, Any]) -> None:
    configure_style()
    figure, axis = plt.subplots(figsize=(3.55, 2.75), constrained_layout=True)
    jitter_rng = np.random.default_rng(SEED + 3)
    x_all = frame["math_average_difficulty"].to_numpy(dtype=float)
    y = frame["fixed_coalition_synergy_bits"].to_numpy(dtype=float)
    jitter_width = 0.012 * max(float(np.ptp(x_all)), 1.0)
    displayed_x = x_all + jitter_rng.uniform(
        -jitter_width, jitter_width, size=len(x_all)
    )
    axis.scatter(
        displayed_x,
        y,
        s=25,
        color="#D66A4E",
        marker="o",
        edgecolor="white",
        linewidth=0.45,
        alpha=0.86,
        zorder=3,
    )
    guide_x = np.linspace(float(x_all.min()), float(x_all.max()), 200)
    slope, intercept = np.polyfit(x_all, y, deg=1)
    axis.plot(
        guide_x,
        slope * guide_x + intercept,
        color="#4B5563",
        linewidth=1.0,
        linestyle="--",
        zorder=2,
    )
    x_pad = 0.045 * max(float(np.ptp(x_all)), 1.0)
    y_span = max(float(np.ptp(y)), 0.1)
    axis.set(
        xlabel="Math average difficulty level",
        ylabel="Fixed-coalition synergy (bits)",
        xlim=(float(x_all.min()) - x_pad, float(x_all.max()) + x_pad),
        ylim=(float(y.min()) - 0.07 * y_span, float(y.max()) + 0.34 * y_span),
    )
    primary = summary["primary"]
    axis.text(
        0.02,
        0.98,
        f"Spearman $\\rho$ = "
        f"{float(primary['rho_pooled']):+.3f}\n"
        f"cohort-blocked permutation $p$ = "
        f"{float(primary['p_blocked_permutation_two_sided']):.4f}\n"
        f"LANGUAGE-family max-T $p$ = "
        f"{float(primary['language_family_max_t_p']):.4f}\n"
        "n = 57",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color="#303942",
    )
    axis.grid(color="#E8EBED", linewidth=0.55, zorder=0)
    save_figure(figure, OUTPUT / "language_math_difficulty_synergy_57")


def write_notes(summary: dict[str, Any]) -> None:
    primary = summary["primary"]
    text = f"""# LANGUAGE Math-difficulty validation

Across all 57 subjects, the frozen `SomMot+Limbic+Cont` synergy is positively
associated with adaptive Math difficulty (Spearman
rho={primary['rho_pooled']:+.3f}). The two-sided cohort-blocked permutation
p={primary['p_blocked_permutation_two_sided']:.5f}
and LANGUAGE-family max-T p={primary['language_family_max_t_p']:.5f}.

This is an exploratory validation because the behavioral endpoint was selected
after inspecting the expanded data. The brain metric represents the full
LANGUAGE run rather than Math-only blocks.
"""
    (OUTPUT / "figure_notes.md").write_text(text, encoding="utf-8")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, brain, behavior = load_data()
    summary = analyze(frame, brain, behavior)
    frame.to_csv(OUTPUT / "source_data.tsv", sep="\t", index=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_validation(frame, summary)
    write_notes(summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
