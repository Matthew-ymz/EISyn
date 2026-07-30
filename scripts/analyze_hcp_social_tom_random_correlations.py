#!/usr/bin/env python3
"""SOCIAL-run fixed-coalition synergy correlations with TOM and Random accuracy.

The previously selected SOCIAL coalition is tested first. If it does not show
condition-wise evidence, all 120 fixed Yeo7 coalitions are screened as a
separate exploratory family with joint TOM/Random multiplicity control.
"""

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
METRICS_500 = (
    ROOT / "results/hcp_cognition_exhaustive_targeted_greedy/metrics.npz"
)
METRICS_1000 = (
    ROOT
    / "results/hcp_task_score_synergy_schaefer1000_validation"
    / "fixed_candidates_schaefer1000.npz"
)
BEHAVIOR = ROOT / "Data/unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
OUTPUT = ROOT / "results/hcp_social_tom_random_correlations"

STATE = "SOCIAL"
PRIMARY_COALITION = "Vis+SomMot+DorsAttn+SalVentAttn"
CONDITIONS = (
    ("TOM", "Social_Task_TOM_Perc_TOM", "#4477AA"),
    ("Random", "Social_Task_Random_Perc_Random", "#CC6677"),
)
PERMUTATIONS = 100_000
SEED = 20260801
ALPHA = 0.05
SYN_TOLERANCE_BITS = 1.0e-10

SHORT = {
    "Vis": "Vis",
    "SomMot": "Som",
    "DorsAttn": "DAN",
    "SalVentAttn": "SVAN",
    "Limbic": "Lim",
    "Cont": "Cont",
    "Default": "Def",
}


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


def short_name(value: str) -> str:
    return "+".join(SHORT[item] for item in value.split("+"))


def age_midpoint(value: str) -> float:
    if value == "36+":
        return 38.0
    low, high = value.split("-")
    return 0.5 * (float(low) + float(high))


def unit_rank(values: np.ndarray, design: np.ndarray | None = None) -> np.ndarray:
    ranked = rankdata(np.asarray(values, dtype=float), method="average")
    if design is None:
        ranked = ranked - ranked.mean()
    else:
        ranked = ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]
    norm = float(np.linalg.norm(ranked))
    if norm <= 1.0e-12:
        raise ValueError("Cannot correlate a constant vector")
    return ranked / norm


def holm(values: list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate(
        np.minimum(1.0, ranked * (len(p) - np.arange(len(p))))
    )
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def bh(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(1.0, adjusted_ranked)
    return adjusted


def permutation_matrix(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack([rng.permutation(n) for _ in range(PERMUTATIONS)])


def test_two_conditions(
    brain: np.ndarray,
    scores: tuple[np.ndarray, np.ndarray],
    permutations: np.ndarray,
    design: np.ndarray | None,
) -> dict[str, Any]:
    brain_rank = unit_rank(brain, design)
    score_rank = np.column_stack([unit_rank(score, design) for score in scores])
    observed = brain_rank @ score_rank
    null = brain_rank[permutations] @ score_rank
    p_raw = np.asarray([
        (1 + np.count_nonzero(np.abs(null[:, index]) >= abs(observed[index])))
        / (PERMUTATIONS + 1)
        for index in range(2)
    ])
    null_max = np.max(np.abs(null), axis=1)
    p_max = np.asarray([
        (1 + np.count_nonzero(null_max >= abs(value))) / (PERMUTATIONS + 1)
        for value in observed
    ])
    return {
        "rho_tom": float(observed[0]),
        "p_tom": float(p_raw[0]),
        "p_tom_max_t_two_conditions": float(p_max[0]),
        "rho_random": float(observed[1]),
        "p_random": float(p_raw[1]),
        "p_random_max_t_two_conditions": float(p_max[1]),
        "holm_p_tom": float(holm(p_raw.tolist())[0]),
        "holm_p_random": float(holm(p_raw.tolist())[1]),
    }


def condition_difference_test(
    brain: np.ndarray,
    tom: np.ndarray,
    random: np.ndarray,
    design: np.ndarray | None,
    seed: int,
) -> tuple[float, float]:
    brain_rank = unit_rank(brain, design)
    difference = unit_rank(tom, design) - unit_rank(random, design)
    observed = float(brain_rank @ difference)
    rng = np.random.default_rng(seed)
    exceed = 0
    chunk = 5_000
    for start in range(0, PERMUTATIONS, chunk):
        size = min(chunk, PERMUTATIONS - start)
        signs = rng.choice((-1.0, 1.0), size=(size, len(brain)))
        null = (signs * difference) @ brain_rank
        exceed += int(np.count_nonzero(np.abs(null) >= abs(observed)))
    return observed, float((1 + exceed) / (PERMUTATIONS + 1))


def leave_one_out(
    brain: np.ndarray, tom: np.ndarray, random: np.ndarray
) -> dict[str, list[float]]:
    values = np.asarray([
        [
            spearmanr(np.delete(brain, index), np.delete(tom, index)).statistic,
            spearmanr(np.delete(brain, index), np.delete(random, index)).statistic,
        ]
        for index in range(len(brain))
    ])
    return {
        "tom_rho_range": [float(values[:, 0].min()), float(values[:, 0].max())],
        "random_rho_range": [float(values[:, 1].min()), float(values[:, 1].max())],
    }


def partial_rank_test(
    brain: np.ndarray,
    target: np.ndarray,
    control: np.ndarray,
    covariates: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    design = np.column_stack([covariates, rankdata(control)])
    brain_rank = unit_rank(brain, design)
    target_rank = unit_rank(target, design)
    observed = float(brain_rank @ target_rank)
    rng = np.random.default_rng(seed)
    exceed = 0
    chunk = 5_000
    for start in range(0, PERMUTATIONS, chunk):
        size = min(chunk, PERMUTATIONS - start)
        indices = np.vstack([rng.permutation(len(brain)) for _ in range(size)])
        null = brain_rank[indices] @ target_rank
        exceed += int(np.count_nonzero(np.abs(null) >= abs(observed)))
    return observed, float((1 + exceed) / (PERMUTATIONS + 1))


def screen_candidates(
    brain_matrix: np.ndarray,
    coalitions: list[str],
    scores: tuple[np.ndarray, np.ndarray],
    permutations: np.ndarray,
    design: np.ndarray | None,
) -> list[dict[str, Any]]:
    brain_rank = np.column_stack([
        unit_rank(brain_matrix[:, index], design)
        for index in range(brain_matrix.shape[1])
    ])
    score_rank = np.column_stack([unit_rank(score, design) for score in scores])
    observed = brain_rank.T @ score_rank
    exceed = np.zeros_like(observed, dtype=np.int64)
    exceed_max = np.zeros_like(observed, dtype=np.int64)
    global_max_parts = []
    chunk = 2_000
    for start in range(0, PERMUTATIONS, chunk):
        indices = permutations[start:start + chunk]
        null_tom = score_rank[:, 0][indices] @ brain_rank
        null_random = score_rank[:, 1][indices] @ brain_rank
        null = np.stack([null_tom, null_random], axis=2)
        exceed += np.sum(np.abs(null) >= np.abs(observed)[None, :, :], axis=0)
        global_max_parts.append(np.max(np.abs(null), axis=(1, 2)))
    global_max = np.concatenate(global_max_parts)
    p = (1 + exceed) / (PERMUTATIONS + 1)
    q = bh(p.ravel()).reshape(p.shape)
    p_max = np.empty_like(p)
    for row in range(p.shape[0]):
        for column in range(p.shape[1]):
            p_max[row, column] = (
                1 + np.count_nonzero(global_max >= abs(observed[row, column]))
            ) / (PERMUTATIONS + 1)
    rows = []
    for index, coalition in enumerate(coalitions):
        rows.append({
            "coalition": coalition,
            "short_coalition": short_name(coalition),
            "rho_tom": float(observed[index, 0]),
            "p_tom": float(p[index, 0]),
            "q_tom_global_240": float(q[index, 0]),
            "p_tom_global_max_t_240": float(p_max[index, 0]),
            "rho_random": float(observed[index, 1]),
            "p_random": float(p[index, 1]),
            "q_random_global_240": float(q[index, 1]),
            "p_random_global_max_t_240": float(p_max[index, 1]),
        })
    return rows


def plot_scatter(
    axis: plt.Axes,
    score: np.ndarray,
    brain: np.ndarray,
    label: str,
    color: str,
    rho: float,
    p_value: float,
) -> None:
    axis.scatter(
        score, brain, s=22, color=color, alpha=0.82,
        edgecolor="white", linewidth=0.4, zorder=3,
    )
    grid = np.linspace(float(score.min()), float(score.max()), 100)
    slope, intercept = np.polyfit(score, brain, 1)
    axis.plot(grid, intercept + slope * grid, color="#687386", lw=1.0, ls="--")
    axis.set_xlabel(f"{label} accuracy (%)")
    axis.set_ylabel("Fixed-coalition synergy (bits)")
    axis.text(
        0.03, 0.97,
        rf"$\rho={rho:+.3f}$" + f"\ntwo-condition max-$T$ $p={p_value:.4f}$\n$n=29$",
        transform=axis.transAxes, va="top", ha="left", color="#25364A",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.2},
    )


def save_figure(figure: plt.Figure, stem: Path) -> None:
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        figure.savefig(
            stem.with_suffix(f".{suffix}"), bbox_inches="tight",
            facecolor="white", **kwargs,
        )
    plt.close(figure)


def make_primary_figure(
    brain: np.ndarray,
    tom: np.ndarray,
    random: np.ndarray,
    primary: dict[str, Any],
    primary_1000: dict[str, Any],
) -> None:
    configure_style()
    figure, axes = plt.subplots(
        1, 3, figsize=(7.4, 2.45),
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.86]},
        constrained_layout=True,
    )
    plot_scatter(
        axes[0], tom, brain, "TOM", CONDITIONS[0][2],
        primary["rho_tom"], primary["p_tom_max_t_two_conditions"],
    )
    plot_scatter(
        axes[1], random, brain, "Random", CONDITIONS[1][2],
        primary["rho_random"], primary["p_random_max_t_two_conditions"],
    )
    atlas_labels = ["S500", "S1000"]
    y = np.asarray([1.0, 0.0])
    offsets = np.asarray([0.09, -0.09])
    for condition, color, offset in zip(
        ("tom", "random"), (CONDITIONS[0][2], CONDITIONS[1][2]), offsets,
        strict=True,
    ):
        values = [
            primary[f"rho_{condition}"],
            primary_1000[f"rho_{condition}"],
        ]
        axes[2].scatter(
            values, y + offset, s=28, color=color, edgecolor="white",
            linewidth=0.4, label=condition.upper() if condition == "tom" else "Random",
        )
        for row, value in enumerate(values):
            axes[2].plot([0.0, value], [y[row] + offset] * 2, color=color, lw=1.0)
    axes[2].axvline(0.0, color="#AEB6C0", lw=0.8, ls=":")
    axes[2].set_yticks(y, atlas_labels)
    axes[2].set_xlim(-0.35, 0.55)
    axes[2].set_xlabel(r"Spearman $\rho$")
    axes[2].legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
    )
    for letter, axis in zip(("a", "b", "c"), axes, strict=True):
        axis.text(
            -0.17, 1.04, letter, transform=axis.transAxes,
            fontsize=9, fontweight="bold", va="bottom",
        )
    save_figure(figure, OUTPUT / "social_primary_tom_random_correlations")


def make_exploration_figure(
    rows: list[dict[str, Any]], primary_coalition: str
) -> None:
    configure_style()
    top_tom = sorted(rows, key=lambda item: abs(item["rho_tom"]), reverse=True)[:8]
    top_random = sorted(
        rows, key=lambda item: abs(item["rho_random"]), reverse=True
    )[:8]
    selected_names = []
    for item in top_tom + top_random:
        if item["coalition"] not in selected_names:
            selected_names.append(item["coalition"])
    if primary_coalition not in selected_names:
        selected_names.append(primary_coalition)
    lookup = {item["coalition"]: item for item in rows}
    selected = [lookup[name] for name in selected_names]
    selected.sort(
        key=lambda item: max(abs(item["rho_tom"]), abs(item["rho_random"])),
        reverse=True,
    )
    matrix = np.asarray([
        [item["rho_tom"], item["rho_random"]] for item in selected
    ])
    figure, axis = plt.subplots(
        figsize=(4.8, max(3.2, 0.24 * len(selected) + 0.8)),
        constrained_layout=True,
    )
    image = axis.imshow(
        matrix, cmap="RdBu_r", vmin=-0.65, vmax=0.65,
        aspect="auto", interpolation="nearest",
    )
    axis.set_xticks([0, 1], ["TOM", "Random"])
    axis.xaxis.tick_top()
    labels = [
        item["short_coalition"]
        + ("  [prespecified]" if item["coalition"] == primary_coalition else "")
        for item in selected
    ]
    axis.set_yticks(range(len(selected)), labels)
    axis.tick_params(length=0)
    axis.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(selected), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for row_index, item in enumerate(selected):
        for column, condition in enumerate(("tom", "random")):
            rho = item[f"rho_{condition}"]
            marker = (
                "†" if item[f"p_{condition}_global_max_t_240"] < ALPHA
                else ("*" if item[f"q_{condition}_global_240"] < ALPHA else "")
            )
            axis.text(
                column, row_index, f"{rho:+.2f}{marker}",
                ha="center", va="center",
                color="white" if abs(rho) >= 0.40 else "#263238",
                fontsize=6.4, fontweight="bold",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    colorbar.set_label(r"Spearman $\rho$")
    save_figure(figure, OUTPUT / "social_candidate_tom_random_screen")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = np.load(METRICS_500)
    states = source["states"].astype(str).tolist()
    subjects = [
        str(value).removeprefix("sub-") for value in source["subjects"].astype(str)
    ]
    coalitions = source["coalitions"].astype(str).tolist()
    brain_matrix = np.asarray(
        source["fixed_block_synergy"][states.index(STATE)], dtype=float
    )
    if brain_matrix.shape != (29, 120):
        raise ValueError(f"Unexpected SOCIAL synergy shape: {brain_matrix.shape}")
    violation = brain_matrix < -SYN_TOLERANCE_BITS
    near_zero_negative = (
        (brain_matrix < 0.0) & (brain_matrix >= -SYN_TOLERANCE_BITS)
    )
    if np.any(violation):
        raise ValueError(
            "PEID Syn nonnegativity violation: "
            f"minimum={brain_matrix.min():.12g} bits, "
            f"threshold={-SYN_TOLERANCE_BITS:.12g} bits, "
            f"count={np.count_nonzero(violation)}"
        )

    with BEHAVIOR.open(newline="", encoding="utf-8-sig") as handle:
        table = {str(row["Subject"]): row for row in csv.DictReader(handle)}
    missing = [subject for subject in subjects if subject not in table]
    if missing:
        raise ValueError(f"Missing behavior rows: {missing}")
    tom = np.asarray([
        float(table[subject][CONDITIONS[0][1]]) for subject in subjects
    ])
    random = np.asarray([
        float(table[subject][CONDITIONS[1][1]]) for subject in subjects
    ])
    design = np.column_stack([
        np.ones(29),
        [age_midpoint(table[subject]["Age"]) for subject in subjects],
        [table[subject]["Gender"] == "M" for subject in subjects],
    ]).astype(float)
    permutations = permutation_matrix(29, SEED)
    primary_brain = brain_matrix[:, coalitions.index(PRIMARY_COALITION)]

    primary = test_two_conditions(
        primary_brain, (tom, random), permutations, design=None
    )
    primary_adjusted = test_two_conditions(
        primary_brain, (tom, random), permutations, design=design
    )
    delta, delta_p = condition_difference_test(
        primary_brain, tom, random, None, SEED + 1
    )
    delta_adjusted, delta_adjusted_p = condition_difference_test(
        primary_brain, tom, random, design, SEED + 2
    )
    primary.update({
        "delta_rho_tom_minus_random": delta,
        "p_condition_difference": delta_p,
        **leave_one_out(primary_brain, tom, random),
    })
    primary_adjusted.update({
        "delta_rho_tom_minus_random": delta_adjusted,
        "p_condition_difference": delta_adjusted_p,
    })

    target = np.load(METRICS_1000)
    if target["subjects"].astype(str).tolist() != source["subjects"].astype(str).tolist():
        raise ValueError("Schaefer-500 and Schaefer-1000 subject order differs")
    target_states = target["states"].astype(str).tolist()
    target_coalitions = target["coalitions"].astype(str).tolist()
    target_index = target_states.index(STATE)
    if target_coalitions[target_index] != PRIMARY_COALITION:
        raise ValueError("Unexpected Schaefer-1000 SOCIAL coalition")
    brain_1000 = np.asarray(target["values"][target_index], dtype=float)
    primary_1000 = test_two_conditions(
        brain_1000, (tom, random), permutations, design=None
    )
    primary_1000_adjusted = test_two_conditions(
        brain_1000, (tom, random), permutations, design=design
    )

    partial_tom, partial_tom_p = partial_rank_test(
        primary_brain, tom, random, design, SEED + 3
    )
    partial_random, partial_random_p = partial_rank_test(
        primary_brain, random, tom, design, SEED + 4
    )

    exploratory = screen_candidates(
        brain_matrix, coalitions, (tom, random), permutations, design=None
    )
    exploratory_adjusted = screen_candidates(
        brain_matrix, coalitions, (tom, random), permutations, design=design
    )
    top = {
        "unadjusted_tom": max(exploratory, key=lambda item: abs(item["rho_tom"])),
        "unadjusted_random": max(
            exploratory, key=lambda item: abs(item["rho_random"])
        ),
        "adjusted_tom": max(
            exploratory_adjusted, key=lambda item: abs(item["rho_tom"])
        ),
        "adjusted_random": max(
            exploratory_adjusted, key=lambda item: abs(item["rho_random"])
        ),
    }

    summary = {
        "experiment": "SOCIAL fixed-coalition synergy: TOM versus Random accuracy",
        "subjects": 29,
        "state": STATE,
        "brain_measure_is_condition_specific": False,
        "primary_coalition": PRIMARY_COALITION,
        "permutations": PERMUTATIONS,
        "syn_nonnegative_tolerance_bits": SYN_TOLERANCE_BITS,
        "syn_values_in_tolerance_negative_range": int(
            np.count_nonzero(near_zero_negative)
        ),
        "primary_schaefer500": primary,
        "primary_schaefer500_age_sex_adjusted": primary_adjusted,
        "primary_schaefer1000_fixed_validation": primary_1000,
        "primary_schaefer1000_age_sex_adjusted": primary_1000_adjusted,
        "condition_specific_diagnostics_schaefer500": {
            "tom_adjusted_for_random_age_sex": {
                "partial_rho": partial_tom,
                "p_permutation": partial_tom_p,
            },
            "random_adjusted_for_tom_age_sex": {
                "partial_rho": partial_random,
                "p_permutation": partial_random_p,
            },
        },
        "behavior_diagnostics": {
            "tom_unique_values": int(len(np.unique(tom))),
            "tom_range": [float(tom.min()), float(tom.max())],
            "random_unique_values": int(len(np.unique(random))),
            "random_range": [float(random.min()), float(random.max())],
            "tom_random_rho": float(spearmanr(tom, random).statistic),
        },
        "exploratory_family": {
            "coalitions": 120,
            "tests_per_analysis": 240,
            "multiplicity": "BH FDR and permutation max-T jointly over 120 coalitions x 2 conditions",
            "top_candidates": top,
            "unadjusted": exploratory,
            "age_sex_adjusted": exploratory_adjusted,
        },
        "interpretation_boundary": (
            "All neural values come from the combined SOCIAL run, not separate "
            "TOM and Random blocks. The 120-coalition screen is exploratory because "
            "selection and effect estimation use the same 29 subjects."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT / "candidate_associations.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for item, adjusted_item in zip(
            exploratory, exploratory_adjusted, strict=True
        ):
            handle.write(json.dumps({
                "unadjusted": item,
                "age_sex_adjusted": adjusted_item,
            }, ensure_ascii=False) + "\n")

    contract = f"""# SOCIAL TOM-versus-Random experiment contract

## Scientific question

For the same 29 subjects and the same SOCIAL-run brain metric, does the
previously selected coalition covary specifically with TOM accuracy or with
Random-control accuracy?

## Frozen primary analysis

- Subjects: the same 29 common imaging subjects.
- Brain state: full SOCIAL run.
- Primary coalition: `{PRIMARY_COALITION}`.
- Brain metric: fixed-coalition total synergy in native bits.
- Behavior: TOM percent correct and Random percent correct, tested separately.
- Statistic: two-sided Spearman correlation.
- Covariate sensitivity: rank residualization for age-bin midpoint and gender.
- Randomization: {PERMUTATIONS:,} shared subject-label permutations.
- Primary multiplicity: max-T and Holm correction across the two conditions.
- Cross-atlas check: the same fixed coalition in Schaefer-1000.
- Syn nonnegative tolerance: {SYN_TOLERANCE_BITS:g} bits; values below the
  negative tolerance fail explicitly.

## Exploratory fallback

All 120 fixed Yeo7 coalitions are screened against both conditions. BH FDR and
permutation max-T are computed jointly across all 240 tests, separately for
unadjusted and age/sex-adjusted analyses.

## Interpretation boundary

The brain metric is estimated from the combined SOCIAL run. Behavioral
condition specificity does not establish block-specific neural dynamics.
Candidate screening and effect estimation use the same cohort.
"""
    (OUTPUT / "experiment_contract.md").write_text(contract, encoding="utf-8")

    lines = [
        "# SOCIAL synergy correlations with TOM and Random accuracy",
        "",
        f"Primary coalition: `{PRIMARY_COALITION}`; n=29.",
        "",
        "| Analysis | TOM | Random | Δρ (TOM−Random) | Difference p |",
        "|---|---:|---:|---:|---:|",
        (
            f"| S500 | {primary['rho_tom']:+.3f} "
            f"(two-condition max-T p={primary['p_tom_max_t_two_conditions']:.4f}) | "
            f"{primary['rho_random']:+.3f} "
            f"(two-condition max-T p={primary['p_random_max_t_two_conditions']:.4f}) | "
            f"{primary['delta_rho_tom_minus_random']:+.3f} | "
            f"{primary['p_condition_difference']:.4f} |"
        ),
        (
            f"| S500 age/sex-adjusted | {primary_adjusted['rho_tom']:+.3f} "
            f"(max-T p={primary_adjusted['p_tom_max_t_two_conditions']:.4f}) | "
            f"{primary_adjusted['rho_random']:+.3f} "
            f"(max-T p={primary_adjusted['p_random_max_t_two_conditions']:.4f}) | "
            f"{primary_adjusted['delta_rho_tom_minus_random']:+.3f} | "
            f"{primary_adjusted['p_condition_difference']:.4f} |"
        ),
        (
            f"| S1000 fixed validation | {primary_1000['rho_tom']:+.3f} "
            f"(max-T p={primary_1000['p_tom_max_t_two_conditions']:.4f}) | "
            f"{primary_1000['rho_random']:+.3f} "
            f"(max-T p={primary_1000['p_random_max_t_two_conditions']:.4f}) | — | — |"
        ),
        "",
        "## Condition-specific partial diagnostics",
        "",
        (
            f"- TOM adjusted for Random, age, and sex: "
            f"partial rho={partial_tom:+.3f}, permutation p={partial_tom_p:.4f}."
        ),
        (
            f"- Random adjusted for TOM, age, and sex: "
            f"partial rho={partial_random:+.3f}, permutation p={partial_random_p:.4f}."
        ),
        "",
        "## Exploratory 120-coalition fallback",
        "",
        "| Analysis | Condition | Top coalition | rho | Raw p | Global q | Global max-T p |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for key, label, condition in (
        ("unadjusted_tom", "Unadjusted", "tom"),
        ("unadjusted_random", "Unadjusted", "random"),
        ("adjusted_tom", "Age/sex-adjusted", "tom"),
        ("adjusted_random", "Age/sex-adjusted", "random"),
    ):
        item = top[key]
        lines.append(
            f"| {label} | {condition.upper() if condition == 'tom' else 'Random'} | "
            f"{item['short_coalition']} | {item[f'rho_{condition}']:+.3f} | "
            f"{item[f'p_{condition}']:.4f} | "
            f"{item[f'q_{condition}_global_240']:.4f} | "
            f"{item[f'p_{condition}_global_max_t_240']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        summary["interpretation_boundary"],
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_primary_figure(primary_brain, tom, random, primary, primary_1000)
    make_exploration_figure(exploratory, PRIMARY_COALITION)
    print(json.dumps({
        "primary": primary,
        "primary_adjusted": primary_adjusted,
        "primary_1000": primary_1000,
        "top_candidates": top,
    }, indent=2))


if __name__ == "__main__":
    main()
