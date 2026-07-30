#!/usr/bin/env python3
"""Compare Language Story and Math accuracy associations with coalition synergy."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr, t as student_t


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "hcp_cognition_exhaustive_targeted_greedy" / "metrics.npz"
VALIDATION_1000_PATH = (
    ROOT
    / "results"
    / "hcp_task_score_synergy_schaefer1000_validation"
    / "fixed_candidates_schaefer1000.npz"
)
BEHAVIOR_PATH = ROOT / "Data" / "unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
OUTPUT_DIR = ROOT / "results" / "hcp_language_story_math_synergy"

STATE = "LANGUAGE"
PRIMARY_COALITION = "Limbic+Default"
PERMUTATIONS = 50_000
BOOTSTRAPS = 20_000
SEED = 20260729
ALPHA = 0.05
SHORT = {
    "Vis": "Vis",
    "SomMot": "Som",
    "DorsAttn": "DAN",
    "SalVentAttn": "SVAN",
    "Limbic": "Lim",
    "Cont": "Cont",
    "Default": "Def",
}


def short_name(value: str) -> str:
    return "+".join(SHORT[item] for item in value.split("+"))


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=float), method="average")
    centered = ranks - ranks.mean()
    norm = float(np.sqrt(np.sum(centered**2)))
    if norm <= 1e-12:
        raise ValueError("Constant vector")
    return centered / norm


def bh_adjust(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def holm_adjust(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate(
        np.minimum(1.0, ranked * (len(p) - np.arange(len(p))))
    )
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def williams_test(
    shared: float, first: float, second: float, n: int
) -> tuple[float, float]:
    """Williams test for two dependent correlations sharing one variable.

    `first` and `second` are correlations between the common brain variable and
    Story/Math ranks; `shared` is the Story–Math rank correlation.
    """
    determinant = (
        1
        - first**2
        - second**2
        - shared**2
        + 2 * first * second * shared
    )
    denominator_squared = (
        2 * ((n - 1) / (n - 3)) * determinant
        + ((first + second) ** 2 / 4) * (1 - shared) ** 3
    )
    if denominator_squared <= 0:
        return float("nan"), float("nan")
    statistic = (
        (first - second)
        * np.sqrt((n - 1) * (1 + shared))
        / np.sqrt(denominator_squared)
    )
    p_value = float(2 * student_t.sf(abs(statistic), df=n - 3))
    return float(statistic), p_value


def load_data() -> dict[str, Any]:
    archive = np.load(METRICS_PATH)
    states = archive["states"].astype(str).tolist()
    subjects = archive["subjects"].astype(str).tolist()
    coalitions = archive["coalitions"].astype(str).tolist()
    synergy = np.asarray(
        archive["fixed_block_synergy"][states.index(STATE)], dtype=float
    )
    if synergy.shape != (29, 120) or not np.isfinite(synergy).all():
        raise ValueError("Expected a complete 29 × 120 Language synergy matrix")
    with BEHAVIOR_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = {str(row["Subject"]): row for row in csv.DictReader(handle)}
    clean_subjects = [subject.removeprefix("sub-") for subject in subjects]
    story = np.asarray([
        float(rows[subject]["Language_Task_Story_Acc"])
        for subject in clean_subjects
    ])
    math = np.asarray([
        float(rows[subject]["Language_Task_Math_Acc"])
        for subject in clean_subjects
    ])
    overall = np.asarray([
        float(rows[subject]["Language_Task_Acc"])
        for subject in clean_subjects
    ])
    if not all(np.isfinite(values).all() for values in (story, math, overall)):
        raise ValueError("Non-finite Language behavior values")
    synergy_1000 = None
    if VALIDATION_1000_PATH.is_file():
        validation = np.load(VALIDATION_1000_PATH)
        validation_subjects = validation["subjects"].astype(str).tolist()
        validation_states = validation["states"].astype(str).tolist()
        validation_coalitions = validation["coalitions"].astype(str).tolist()
        if validation_subjects != subjects:
            raise ValueError("Schaefer-500/1000 subject order mismatch")
        validation_index = validation_states.index(STATE)
        if validation_coalitions[validation_index] != PRIMARY_COALITION:
            raise ValueError("Unexpected Schaefer-1000 Language coalition")
        synergy_1000 = np.asarray(
            validation["values"][validation_index], dtype=float
        )
    return {
        "subjects": subjects,
        "coalitions": coalitions,
        "synergy": synergy,
        "story": story,
        "math": math,
        "overall": overall,
        "synergy_1000": synergy_1000,
    }


def paired_bootstrap_difference(
    brain: np.ndarray,
    story: np.ndarray,
    math: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Paired subject bootstrap on full-sample rank transforms."""
    brain_ranks = rankdata(brain, method="average")
    story_ranks = rankdata(story, method="average")
    math_ranks = rankdata(math, method="average")
    differences = np.empty(BOOTSTRAPS, dtype=float)
    chunk = 2_000
    for start in range(0, BOOTSTRAPS, chunk):
        stop = min(start + chunk, BOOTSTRAPS)
        indices = rng.integers(0, len(brain), size=(stop - start, len(brain)))
        y = brain_ranks[indices]
        s = story_ranks[indices]
        m = math_ranks[indices]
        y = y - y.mean(axis=1, keepdims=True)
        s = s - s.mean(axis=1, keepdims=True)
        m = m - m.mean(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_story = np.sum(y * s, axis=1) / np.sqrt(
                np.sum(y**2, axis=1) * np.sum(s**2, axis=1)
            )
            r_math = np.sum(y * m, axis=1) / np.sqrt(
                np.sum(y**2, axis=1) * np.sum(m**2, axis=1)
            )
        differences[start:stop] = r_story - r_math
    finite_differences = differences[np.isfinite(differences)]
    if finite_differences.size < int(0.99 * BOOTSTRAPS):
        raise ValueError(
            "Too many invalid paired bootstrap resamples: "
            f"{BOOTSTRAPS - finite_differences.size}"
        )
    return {
        "bootstrap_repeats": BOOTSTRAPS,
        "valid_bootstrap_repeats": int(finite_differences.size),
        "invalid_constant_resamples": int(BOOTSTRAPS - finite_differences.size),
        "ci95_low": float(np.quantile(finite_differences, 0.025)),
        "ci95_median": float(np.median(finite_differences)),
        "ci95_high": float(np.quantile(finite_differences, 0.975)),
        "fraction_above_zero": float(np.mean(finite_differences > 0)),
    }


def analyze(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    story = data["story"]
    math = data["math"]
    synergy = data["synergy"]
    coalitions = data["coalitions"]
    n = len(story)
    story_rank = normalized_ranks(story)
    math_rank = normalized_ranks(math)
    brain_rank = np.column_stack([
        normalized_ranks(synergy[:, index]) for index in range(synergy.shape[1])
    ])
    observed_story = np.asarray(story_rank @ brain_rank, dtype=float)
    observed_math = np.asarray(math_rank @ brain_rank, dtype=float)
    score_rho = float(story_rank @ math_rank)

    rng = np.random.default_rng(SEED)
    permutation_indices = np.vstack([
        rng.permutation(n) for _ in range(PERMUTATIONS)
    ])
    null_story = np.asarray(
        story_rank[permutation_indices] @ brain_rank, dtype=np.float32
    )
    null_math = np.asarray(
        math_rank[permutation_indices] @ brain_rank, dtype=np.float32
    )
    story_max = np.max(np.abs(null_story), axis=1)
    math_max = np.max(np.abs(null_math), axis=1)
    global_max = np.maximum(story_max, math_max)

    rows: list[dict[str, Any]] = []
    for index, coalition in enumerate(coalitions):
        story_rho = float(observed_story[index])
        math_rho = float(observed_math[index])
        p_story = float(
            (1 + np.count_nonzero(np.abs(null_story[:, index]) >= abs(story_rho)))
            / (PERMUTATIONS + 1)
        )
        p_math = float(
            (1 + np.count_nonzero(np.abs(null_math[:, index]) >= abs(math_rho)))
            / (PERMUTATIONS + 1)
        )
        statistic, difference_p = williams_test(
            score_rho, story_rho, math_rho, n
        )
        rows.append({
            "coalition": coalition,
            "coalition_short": short_name(coalition),
            "coalition_size": coalition.count("+") + 1,
            "n_subjects": n,
            "primary_coalition": coalition == PRIMARY_COALITION,
            "story_rho": story_rho,
            "story_p_permutation": p_story,
            "story_q_120": None,
            "story_p_max_t_120": float(
                (1 + np.count_nonzero(story_max >= abs(story_rho)))
                / (PERMUTATIONS + 1)
            ),
            "story_global_q_240": None,
            "story_global_p_max_t_240": float(
                (1 + np.count_nonzero(global_max >= abs(story_rho)))
                / (PERMUTATIONS + 1)
            ),
            "math_rho": math_rho,
            "math_p_permutation": p_math,
            "math_q_120": None,
            "math_p_max_t_120": float(
                (1 + np.count_nonzero(math_max >= abs(math_rho)))
                / (PERMUTATIONS + 1)
            ),
            "math_global_q_240": None,
            "math_global_p_max_t_240": float(
                (1 + np.count_nonzero(global_max >= abs(math_rho)))
                / (PERMUTATIONS + 1)
            ),
            "rho_story_minus_math": story_rho - math_rho,
            "williams_t": statistic,
            "williams_p": difference_p,
            "difference_q_120": None,
        })

    story_q = bh_adjust([row["story_p_permutation"] for row in rows])
    math_q = bh_adjust([row["math_p_permutation"] for row in rows])
    difference_q = bh_adjust([row["williams_p"] for row in rows])
    global_q = bh_adjust(
        [row["story_p_permutation"] for row in rows]
        + [row["math_p_permutation"] for row in rows]
    )
    for index, row in enumerate(rows):
        row["story_q_120"] = float(story_q[index])
        row["math_q_120"] = float(math_q[index])
        row["difference_q_120"] = float(difference_q[index])
        row["story_global_q_240"] = float(global_q[index])
        row["math_global_q_240"] = float(global_q[len(rows) + index])

    primary = next(row for row in rows if row["primary_coalition"])
    primary_raw = [
        primary["story_p_permutation"],
        primary["math_p_permutation"],
        primary["williams_p"],
    ]
    primary_holm = holm_adjust(primary_raw)
    bootstrap = paired_bootstrap_difference(
        synergy[:, coalitions.index(PRIMARY_COALITION)],
        story,
        math,
        rng,
    )
    cross_atlas_validation = None
    if data["synergy_1000"] is not None:
        validation_rank = normalized_ranks(data["synergy_1000"])
        validation_story_rho = float(story_rank @ validation_rank)
        validation_math_rho = float(math_rank @ validation_rank)
        validation_story_null = story_rank[permutation_indices] @ validation_rank
        validation_math_null = math_rank[permutation_indices] @ validation_rank
        validation_story_p = float(
            (
                1
                + np.count_nonzero(
                    np.abs(validation_story_null) >= abs(validation_story_rho)
                )
            )
            / (PERMUTATIONS + 1)
        )
        validation_math_p = float(
            (
                1
                + np.count_nonzero(
                    np.abs(validation_math_null) >= abs(validation_math_rho)
                )
            )
            / (PERMUTATIONS + 1)
        )
        validation_t, validation_difference_p = williams_test(
            score_rho, validation_story_rho, validation_math_rho, n
        )
        validation_holm = holm_adjust(
            [validation_story_p, validation_math_p, validation_difference_p]
        )
        cross_atlas_validation = {
            "atlas": "Schaefer-1000",
            "coalition": PRIMARY_COALITION,
            "story_rho": validation_story_rho,
            "story_p_permutation": validation_story_p,
            "math_rho": validation_math_rho,
            "math_p_permutation": validation_math_p,
            "rho_story_minus_math": validation_story_rho - validation_math_rho,
            "williams_t": validation_t,
            "williams_p": validation_difference_p,
            "holm_p_across_three_tests": {
                "story_correlation": float(validation_holm[0]),
                "math_correlation": float(validation_holm[1]),
                "correlation_difference": float(validation_holm[2]),
            },
            "difference_bootstrap": paired_bootstrap_difference(
                data["synergy_1000"], story, math, rng
            ),
        }
    summary = {
        "experiment": "Language Story vs Math accuracy–synergy comparison",
        "subjects": n,
        "state": STATE,
        "primary_coalition": PRIMARY_COALITION,
        "permutations": PERMUTATIONS,
        "score_descriptives": {
            "story": {
                "minimum": float(story.min()),
                "maximum": float(story.max()),
                "mean": float(story.mean()),
                "unique_values": int(np.unique(story).size),
            },
            "math": {
                "minimum": float(math.min()),
                "maximum": float(math.max()),
                "mean": float(math.mean()),
                "unique_values": int(np.unique(math).size),
            },
            "story_math_spearman_rho": score_rho,
        },
        "primary": {
            **primary,
            "holm_p_across_three_primary_tests": {
                "story_correlation": float(primary_holm[0]),
                "math_correlation": float(primary_holm[1]),
                "correlation_difference": float(primary_holm[2]),
            },
            "difference_bootstrap": bootstrap,
        },
        "cross_atlas_validation": cross_atlas_validation,
        "exploratory_counts": {
            "story_raw_p_lt_0_05": int(sum(row["story_p_permutation"] < ALPHA for row in rows)),
            "story_q_120_lt_0_05": int(sum(row["story_q_120"] < ALPHA for row in rows)),
            "story_max_t_120_lt_0_05": int(sum(row["story_p_max_t_120"] < ALPHA for row in rows)),
            "math_raw_p_lt_0_05": int(sum(row["math_p_permutation"] < ALPHA for row in rows)),
            "math_q_120_lt_0_05": int(sum(row["math_q_120"] < ALPHA for row in rows)),
            "math_max_t_120_lt_0_05": int(sum(row["math_p_max_t_120"] < ALPHA for row in rows)),
            "difference_raw_p_lt_0_05": int(sum(row["williams_p"] < ALPHA for row in rows)),
            "difference_q_120_lt_0_05": int(sum(row["difference_q_120"] < ALPHA for row in rows)),
        },
        "top_story": sorted(rows, key=lambda row: -abs(row["story_rho"]))[:10],
        "top_math": sorted(rows, key=lambda row: -abs(row["math_rho"]))[:10],
        "top_difference": sorted(
            rows, key=lambda row: -abs(row["rho_story_minus_math"])
        )[:10],
        "interpretation_boundary": (
            "Limbic+Default is the prespecified primary hypothesis. All other "
            "coalition rankings are exploratory and require multiplicity control."
        ),
    }
    return rows, summary


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def save_figure(figure: plt.Figure, stem: str) -> None:
    for extension in ("png", "svg", "pdf"):
        figure.savefig(
            OUTPUT_DIR / f"{stem}.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )


def scatter_panel(
    axis: plt.Axes,
    score: np.ndarray,
    brain: np.ndarray,
    *,
    color: str,
    label: str,
    rho: float,
    p_value: float,
    holm_p: float,
) -> None:
    axis.scatter(
        score,
        brain,
        s=22,
        color=color,
        alpha=0.84,
        edgecolor="white",
        linewidth=0.35,
    )
    y_range = float(np.ptp(brain))
    axis.set_ylim(
        float(brain.min() - 0.08 * y_range),
        float(brain.max() + 0.27 * y_range),
    )
    axis.set_xlabel(f"{label} accuracy (%)")
    axis.set_ylabel("Limbic+Default synergy (bits)")
    axis.text(
        0.02,
        0.98,
        (
            rf"$\rho$={rho:+.3f} · perm. $p$={p_value:.4f}"
            "\n"
            rf"primary Holm $p$={holm_p:.4f} · $n$=29"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#45515C",
    )


def plot_results(
    data: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    configure_style()
    primary = summary["primary"]
    primary_index = data["coalitions"].index(PRIMARY_COALITION)
    brain = data["synergy"][:, primary_index]
    figure = plt.figure(figsize=(7.4, 5.15), constrained_layout=True)
    grid = figure.add_gridspec(
        2, 3, height_ratios=(1.05, 1.25), width_ratios=(1, 1, 0.82)
    )
    story_axis = figure.add_subplot(grid[0, 0])
    math_axis = figure.add_subplot(grid[0, 1], sharey=story_axis)
    comparison_axis = figure.add_subplot(grid[0, 2])
    heatmap_axis = figure.add_subplot(grid[1, :])

    scatter_panel(
        story_axis,
        data["story"],
        brain,
        color="#4C78A8",
        label="Story",
        rho=primary["story_rho"],
        p_value=primary["story_p_permutation"],
        holm_p=primary["holm_p_across_three_primary_tests"]["story_correlation"],
    )
    scatter_panel(
        math_axis,
        data["math"],
        brain,
        color="#E07A5F",
        label="Math",
        rho=primary["math_rho"],
        p_value=primary["math_p_permutation"],
        holm_p=primary["holm_p_across_three_primary_tests"]["math_correlation"],
    )
    math_axis.set_ylabel("")
    math_axis.tick_params(labelleft=False)
    story_axis.set_title("a  Story", loc="left", weight="bold")
    math_axis.set_title("b  Math", loc="left", weight="bold")

    comparison_axis.axvline(0, color="#B5BDC4", lw=0.8)
    comparison_axis.scatter(
        [primary["story_rho"], primary["math_rho"]],
        [1.06, 0.06],
        s=42,
        marker="o",
        color="#4C78A8",
        zorder=3,
    )
    comparison_axis.plot(
        [primary["math_rho"], primary["story_rho"]],
        [0.06, 1.06],
        color="#4C78A8",
        lw=1,
        alpha=0.55,
        zorder=1,
    )
    validation = summary["cross_atlas_validation"]
    if validation is not None:
        comparison_axis.scatter(
            [validation["story_rho"], validation["math_rho"]],
            [0.94, -0.06],
            s=38,
            marker="D",
            color="#8064A2",
            zorder=3,
        )
        comparison_axis.plot(
            [validation["math_rho"], validation["story_rho"]],
            [-0.06, 0.94],
            color="#8064A2",
            lw=1,
            alpha=0.55,
            zorder=1,
        )
        comparison_axis.text(
            primary["story_rho"] + 0.02,
            1.06,
            "S500",
            ha="left",
            va="center",
            fontsize=5.8,
            color="#4C78A8",
        )
        comparison_axis.text(
            validation["story_rho"] - 0.02,
            0.94,
            "S1000",
            ha="right",
            va="center",
            fontsize=5.8,
            color="#8064A2",
        )
    comparison_axis.set_yticks([1, 0], ["Story", "Math"])
    comparison_axis.set_xlim(-0.1, 0.62)
    comparison_axis.set_xlabel(r"Spearman $\rho$")
    comparison_axis.set_title("c  Cross-atlas coefficients", loc="left", weight="bold")
    bootstrap = primary["difference_bootstrap"]
    comparison_axis.text(
        1.04,
        0.50,
        (
            rf"S500 $\Delta\rho$={primary['rho_story_minus_math']:+.3f}"
            "\n"
            rf"Williams $p$={primary['williams_p']:.3f}"
            + (
                "\n"
                rf"S1000 $\Delta\rho$={validation['rho_story_minus_math']:+.3f}"
                "\n"
                rf"Williams $p$={validation['williams_p']:.3f}"
                if validation is not None
                else ""
            )
        ),
        transform=comparison_axis.transAxes,
        ha="left",
        va="center",
        fontsize=6.1,
        color="#45515C",
    )

    ranked = sorted(
        rows,
        key=lambda row: -max(
            abs(row["story_rho"]),
            abs(row["math_rho"]),
            abs(row["rho_story_minus_math"]),
        ),
    )
    selected = ranked[:11]
    if not any(row["primary_coalition"] for row in selected):
        selected[-1] = primary
    selected = sorted(
        {row["coalition"]: row for row in selected}.values(),
        key=lambda row: (
            not row["primary_coalition"],
            -max(
                abs(row["story_rho"]),
                abs(row["math_rho"]),
                abs(row["rho_story_minus_math"]),
            ),
        ),
    )
    matrix = np.asarray([
        [row["story_rho"], row["math_rho"], row["rho_story_minus_math"]]
        for row in selected
    ])
    image = heatmap_axis.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-0.68,
        vmax=0.68,
        aspect="auto",
        interpolation="nearest",
    )
    heatmap_axis.set_xticks(
        [0, 1, 2],
        [r"Story $\rho$", r"Math $\rho$", r"$\Delta\rho$ (Story − Math)"],
    )
    heatmap_axis.xaxis.tick_top()
    heatmap_axis.set_yticks(
        np.arange(len(selected)),
        [
            ("P  " if row["primary_coalition"] else "") + row["coalition_short"]
            for row in selected
        ],
    )
    heatmap_axis.tick_params(length=0)
    heatmap_axis.set_title(
        "d  Exploratory coalition comparison (P = prespecified primary)",
        loc="left",
        weight="bold",
        pad=24,
    )
    for row_index, row in enumerate(selected):
        values = (
            row["story_rho"],
            row["math_rho"],
            row["rho_story_minus_math"],
        )
        p_values = (
            row["story_q_120"],
            row["math_q_120"],
            row["difference_q_120"],
        )
        for column_index, (value, q_value) in enumerate(zip(values, p_values)):
            marker = "*" if q_value < ALPHA else ""
            heatmap_axis.text(
                column_index,
                row_index,
                f"{value:+.2f}{marker}",
                ha="center",
                va="center",
                fontsize=6.2,
                weight="bold",
                color="white" if abs(value) > 0.40 else "#263238",
            )
    colorbar = figure.colorbar(
        image,
        ax=heatmap_axis,
        orientation="horizontal",
        fraction=0.07,
        pad=0.08,
    )
    colorbar.set_label(r"Spearman correlation or paired difference in $\rho$")
    save_figure(figure, "language_story_math_synergy")
    plt.close(figure)


def write_report(summary: dict[str, Any]) -> None:
    primary = summary["primary"]
    holm = primary["holm_p_across_three_primary_tests"]
    bootstrap = primary["difference_bootstrap"]
    lines = [
        "# Language Story vs Math synergy analysis",
        "",
        "All tests use the same 29 subjects and the same Language-state fixed-coalition "
        "synergy values. `Limbic+Default` was frozen as the primary coalition.",
        "",
        "## Primary result",
        "",
        "| Comparison | rho or delta-rho | raw p | Holm p across 3 primary tests |",
        "|---|---:|---:|---:|",
        (
            f"| Story ACC vs Lim+Def | {primary['story_rho']:+.3f} | "
            f"{primary['story_p_permutation']:.5f} | "
            f"{holm['story_correlation']:.4f} |"
        ),
        (
            f"| Math ACC vs Lim+Def | {primary['math_rho']:+.3f} | "
            f"{primary['math_p_permutation']:.5f} | "
            f"{holm['math_correlation']:.4f} |"
        ),
        (
            f"| Story rho − Math rho | {primary['rho_story_minus_math']:+.3f} | "
            f"{primary['williams_p']:.5f} | "
            f"{holm['correlation_difference']:.4f} |"
        ),
        "",
        (
            f"The paired bootstrap 95% interval for the correlation difference is "
            f"[{bootstrap['ci95_low']:+.3f}, {bootstrap['ci95_high']:+.3f}]."
        ),
        "",
        "## Fixed Schaefer-1000 validation",
        "",
    ]
    validation = summary["cross_atlas_validation"]
    if validation is not None:
        validation_holm = validation["holm_p_across_three_tests"]
        lines += [
            "| Comparison | rho or delta-rho | raw p | Holm p across 3 tests |",
            "|---|---:|---:|---:|",
            (
                f"| Story ACC vs Lim+Def | {validation['story_rho']:+.3f} | "
                f"{validation['story_p_permutation']:.5f} | "
                f"{validation_holm['story_correlation']:.4f} |"
            ),
            (
                f"| Math ACC vs Lim+Def | {validation['math_rho']:+.3f} | "
                f"{validation['math_p_permutation']:.5f} | "
                f"{validation_holm['math_correlation']:.4f} |"
            ),
            (
                f"| Story rho − Math rho | "
                f"{validation['rho_story_minus_math']:+.3f} | "
                f"{validation['williams_p']:.5f} | "
                f"{validation_holm['correlation_difference']:.4f} |"
            ),
        ]
    else:
        lines.append("The Schaefer-1000 fixed candidate cache was unavailable.")
    lines += [
        "",
        "## Exploratory 120-coalition scan",
        "",
        "| Family | raw p < 0.05 | FDR q < 0.05 | max-T p < 0.05 |",
        "|---|---:|---:|---:|",
        (
            f"| Story correlations | "
            f"{summary['exploratory_counts']['story_raw_p_lt_0_05']} | "
            f"{summary['exploratory_counts']['story_q_120_lt_0_05']} | "
            f"{summary['exploratory_counts']['story_max_t_120_lt_0_05']} |"
        ),
        (
            f"| Math correlations | "
            f"{summary['exploratory_counts']['math_raw_p_lt_0_05']} | "
            f"{summary['exploratory_counts']['math_q_120_lt_0_05']} | "
            f"{summary['exploratory_counts']['math_max_t_120_lt_0_05']} |"
        ),
        (
            f"| Story–Math differences | "
            f"{summary['exploratory_counts']['difference_raw_p_lt_0_05']} | "
            f"{summary['exploratory_counts']['difference_q_120_lt_0_05']} | n/a |"
        ),
        "",
        "## Interpretation boundary",
        "",
        "A significant Story association combined with a nonsignificant Math "
        "association does not by itself prove that the two correlations differ. "
        "The Williams test and paired bootstrap directly evaluate that contrast. "
        "Story accuracy has only four distinct values and a strong ceiling effect.",
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    rows, summary = analyze(data)
    with (OUTPUT_DIR / "associations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(summary)
    plot_results(data, rows, summary)
    print(json.dumps({
        "output_dir": str(OUTPUT_DIR),
        "subjects": summary["subjects"],
        "primary": summary["primary"],
        "exploratory_counts": summary["exploratory_counts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
