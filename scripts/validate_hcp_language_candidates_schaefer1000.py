#!/usr/bin/env python3
"""Validate three frozen Language Story-vs-Math candidates at Schaefer-1000."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_language_story_math_synergy import (
    BOOTSTRAPS,
    PERMUTATIONS,
    holm_adjust,
    normalized_ranks,
    paired_bootstrap_difference,
    williams_test,
)
from scripts.analyze_hcp_schaefer500_yeo7_network_attribution import (
    NETWORK_ORDER,
    discover_inputs,
)
from scripts.analyze_hcp_task_evoked_pc2_xi_hierarchy import network_module_indices
from scripts.phi_hierarchy import subset_phi_raw
from scripts.run_hcp_schaefer500_yeo7_module_phi_decomposition import module_ei_table
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import load_yeo7_groups
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi
from scripts.tune_hcp_task_evoked_xi_hierarchy import prepare_projection


OUTPUT_DIR = (
    ROOT
    / "results"
    / "hcp_language_story_math_candidates_schaefer1000_replication"
)
CACHE_PATH = OUTPUT_DIR / "fixed_candidates_schaefer1000.npz"
METRICS_500 = ROOT / "results" / "hcp_cognition_exhaustive_targeted_greedy" / "metrics.npz"
ARRAYS_1000 = (
    ROOT
    / "results"
    / "hcp_schaefer1000_task_evoked_xi_replication"
    / "full"
    / "k1_p3_a1"
    / "arrays.npz"
)
BEHAVIOR_PATH = ROOT / "Data" / "unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
REST_ROOT = ROOT / "Data" / "hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30"
TASK_ROOT = ROOT / "Data" / "hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_30"
LABELS_1000 = REST_ROOT / "_atlas_labels" / "Schaefer2018_1000Parcels_7Networks_order.txt"

STATE = "LANGUAGE"
SEED = 2026072902
CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("SomMot", "Limbic", "Cont"),
    ("SalVentAttn", "Limbic", "Default"),
    ("SomMot", "Cont"),
)
SHORT = {
    "Vis": "Vis",
    "SomMot": "Som",
    "DorsAttn": "DAN",
    "SalVentAttn": "SVAN",
    "Limbic": "Lim",
    "Cont": "Cont",
    "Default": "Def",
}


def candidate_key(candidate: Sequence[str]) -> str:
    return "+".join(candidate)


def short_name(candidate: Sequence[str] | str) -> str:
    names = candidate.split("+") if isinstance(candidate, str) else candidate
    return "+".join(SHORT[name] for name in names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Compute the three metrics for only the first subject without caching.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore an existing complete Schaefer-1000 metric cache.",
    )
    return parser.parse_args()


def load_subjects() -> list[str]:
    arrays = np.load(ARRAYS_1000)
    subjects = arrays["subjects"].astype(str).tolist()
    if len(subjects) != 29 or len(set(subjects)) != 29:
        raise ValueError("Expected 29 unique Schaefer-1000 subjects")
    return subjects


def compute_values(
    subjects: list[str], *, smoke: bool = False, recompute: bool = False
) -> np.ndarray:
    keys = np.asarray([candidate_key(candidate) for candidate in CANDIDATES])
    if CACHE_PATH.is_file() and not smoke and not recompute:
        cache = np.load(CACHE_PATH)
        if (
            np.array_equal(cache["subjects"].astype(str), np.asarray(subjects))
            and np.array_equal(cache["coalitions"].astype(str), keys)
            and str(cache["state"]) == STATE
        ):
            values = np.asarray(cache["values"], dtype=float)
            if values.shape == (3, 29) and np.isfinite(values).all():
                return values

    discovered = discover_inputs(REST_ROOT, TASK_ROOT)
    groups = load_yeo7_groups(LABELS_1000, expected_parcels=1000)
    indices = network_module_indices(NETWORK_ORDER, n_components=1, order=3)
    selected_subjects = subjects[:1] if smoke else subjects
    values = np.zeros((3, len(selected_subjects)), dtype=float)
    for subject_index, subject in enumerate(
        tqdm(selected_subjects, desc="Schaefer-1000 Language", unit="subject")
    ):
        projections, _, development_end = prepare_projection(
            Path(discovered[subject][STATE]),
            groups,
            state=STATE,
            max_components=2,
            task_retained_key="Schaefer1000_taskRetained",
            task_regressed_key="Schaefer1000_taskRegressed",
            expected_parcels=1000,
        )
        fitted = fit_delta_history_phi(
            projections[1], alpha=1.0, order=3, development_end=development_end
        )
        table = module_ei_table(
            fitted["transition"], fitted["noise_covariance"], indices, ridge=1.0e-6
        )
        singleton = {name: float(table[(name,)]) for name in NETWORK_ORDER}
        for candidate_index, candidate in enumerate(CANDIDATES):
            values[candidate_index, subject_index] = subset_phi_raw(
                candidate, table, singleton
            )
    if smoke:
        return values
    if values.shape != (3, 29) or not np.isfinite(values).all():
        raise ValueError("Incomplete Schaefer-1000 candidate matrix")
    np.savez_compressed(
        CACHE_PATH,
        values=values,
        subjects=np.asarray(subjects),
        state=np.asarray(STATE),
        coalitions=keys,
    )
    return values


def load_behavior(subjects: list[str]) -> tuple[np.ndarray, np.ndarray]:
    with BEHAVIOR_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = {str(row["Subject"]): row for row in csv.DictReader(handle)}
    clean = [subject.removeprefix("sub-") for subject in subjects]
    story = np.asarray([
        float(rows[subject]["Language_Task_Story_Acc"]) for subject in clean
    ])
    math = np.asarray([
        float(rows[subject]["Language_Task_Math_Acc"]) for subject in clean
    ])
    if not np.isfinite(story).all() or not np.isfinite(math).all():
        raise ValueError("Non-finite Story or Math accuracy")
    return story, math


def load_values_500(subjects: list[str]) -> np.ndarray:
    archive = np.load(METRICS_500)
    if archive["subjects"].astype(str).tolist() != subjects:
        raise ValueError("Schaefer-500/1000 subject order mismatch")
    states = archive["states"].astype(str).tolist()
    coalitions = archive["coalitions"].astype(str).tolist()
    values = np.column_stack([
        archive["fixed_block_synergy"][
            states.index(STATE), :, coalitions.index(candidate_key(candidate))
        ]
        for candidate in CANDIDATES
    ]).T
    if values.shape != (3, 29) or not np.isfinite(values).all():
        raise ValueError("Incomplete Schaefer-500 candidate matrix")
    return values


def atlas_statistics(
    atlas: str,
    values: np.ndarray,
    story: np.ndarray,
    math: np.ndarray,
    permutations: np.ndarray,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    story_rank = normalized_ranks(story)
    math_rank = normalized_ranks(math)
    score_rho = float(story_rank @ math_rank)
    results = []
    for index, candidate in enumerate(CANDIDATES):
        brain = values[index]
        brain_rank = normalized_ranks(brain)
        story_rho = float(story_rank @ brain_rank)
        math_rho = float(math_rank @ brain_rank)
        story_null = story_rank[permutations] @ brain_rank
        math_null = math_rank[permutations] @ brain_rank
        story_p = float(
            (1 + np.count_nonzero(np.abs(story_null) >= abs(story_rho)))
            / (PERMUTATIONS + 1)
        )
        math_p = float(
            (1 + np.count_nonzero(np.abs(math_null) >= abs(math_rho)))
            / (PERMUTATIONS + 1)
        )
        statistic, difference_p = williams_test(
            score_rho, story_rho, math_rho, len(story)
        )
        results.append({
            "atlas": atlas,
            "coalition": candidate_key(candidate),
            "coalition_short": short_name(candidate),
            "n_subjects": len(story),
            "story_rho": story_rho,
            "story_p_permutation": story_p,
            "story_holm_p_3": None,
            "math_rho": math_rho,
            "math_p_permutation": math_p,
            "math_holm_p_3": None,
            "rho_story_minus_math": story_rho - math_rho,
            "williams_t": statistic,
            "williams_p": difference_p,
            "difference_holm_p_3": None,
            "difference_bootstrap": paired_bootstrap_difference(
                brain, story, math, rng
            ),
        })
    story_holm = holm_adjust([item["story_p_permutation"] for item in results])
    math_holm = holm_adjust([item["math_p_permutation"] for item in results])
    difference_holm = holm_adjust([item["williams_p"] for item in results])
    for index, item in enumerate(results):
        item["story_holm_p_3"] = float(story_holm[index])
        item["math_holm_p_3"] = float(math_holm[index])
        item["difference_holm_p_3"] = float(difference_holm[index])
    return results


def analyze(
    subjects: list[str],
    values_500: np.ndarray,
    values_1000: np.ndarray,
    story: np.ndarray,
    math: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    permutations = np.vstack([
        rng.permutation(len(subjects)) for _ in range(PERMUTATIONS)
    ])
    results_500 = atlas_statistics(
        "Schaefer-500", values_500, story, math, permutations, rng
    )
    results_1000 = atlas_statistics(
        "Schaefer-1000", values_1000, story, math, permutations, rng
    )
    validation = []
    for index, candidate in enumerate(CANDIDATES):
        discovery = results_500[index]
        replication = results_1000[index]
        direction_preserved = bool(
            replication["story_rho"] > 0
            and replication["math_rho"] < 0
            and replication["rho_story_minus_math"] > 0
        )
        strict = bool(
            direction_preserved and replication["difference_holm_p_3"] < 0.05
        )
        validation.append({
            "coalition": candidate_key(candidate),
            "coalition_short": short_name(candidate),
            "cross_atlas_metric_rho": float(
                spearmanr(values_500[index], values_1000[index]).statistic
            ),
            "direction_pattern_preserved": direction_preserved,
            "strict_replication": strict,
            "delta_rho_500": discovery["rho_story_minus_math"],
            "delta_rho_1000": replication["rho_story_minus_math"],
            "delta_retention_ratio": (
                replication["rho_story_minus_math"]
                / discovery["rho_story_minus_math"]
            ),
        })
    brain_500 = values_500[0]
    brain_1000 = values_1000[0]
    story_rank = rankdata(story)
    math_rank = rankdata(math)

    def residualize(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones(len(covariate)), covariate])
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        return values - design @ coefficients

    def partial_spearman(
        brain: np.ndarray, score: np.ndarray, covariate: np.ndarray
    ) -> tuple[float, float]:
        brain_rank = rankdata(brain)
        score_rank = rankdata(score)
        covariate_rank = rankdata(covariate)
        result = pearsonr(
            residualize(brain_rank, covariate_rank),
            residualize(score_rank, covariate_rank),
        )
        return float(result.statistic), float(result.pvalue)

    def leave_one_out(brain: np.ndarray) -> dict[str, list[float]]:
        story_coefficients = []
        math_coefficients = []
        differences = []
        for excluded in range(len(brain)):
            retained = np.arange(len(brain)) != excluded
            story_coefficient = float(
                spearmanr(brain[retained], story[retained]).statistic
            )
            math_coefficient = float(
                spearmanr(brain[retained], math[retained]).statistic
            )
            story_coefficients.append(story_coefficient)
            math_coefficients.append(math_coefficient)
            differences.append(story_coefficient - math_coefficient)
        return {
            "story_rho_range": [
                float(min(story_coefficients)),
                float(max(story_coefficients)),
            ],
            "math_rho_range": [
                float(min(math_coefficients)),
                float(max(math_coefficients)),
            ],
            "delta_rho_range": [
                float(min(differences)),
                float(max(differences)),
            ],
        }

    partial_500_story = partial_spearman(brain_500, story, math)
    partial_500_math = partial_spearman(brain_500, math, story)
    partial_1000_story = partial_spearman(brain_1000, story, math)
    partial_1000_math = partial_spearman(brain_1000, math, story)
    score_association = spearmanr(story, math)
    atlas_association = spearmanr(brain_500, brain_1000)
    sensitivity = {
        "coalition": candidate_key(CANDIDATES[0]),
        "behavior": {
            "story_unique_values": int(len(np.unique(story))),
            "story_range": [float(story.min()), float(story.max())],
            "story_median": float(np.median(story)),
            "math_unique_values": int(len(np.unique(math))),
            "math_range": [float(math.min()), float(math.max())],
            "math_median": float(np.median(math)),
            "story_math_rho": float(score_association.statistic),
            "story_math_p": float(score_association.pvalue),
        },
        "cross_atlas": {
            "rho": float(atlas_association.statistic),
            "p": float(atlas_association.pvalue),
        },
        "partial_spearman_post_hoc": {
            "schaefer500_story_given_math": {
                "rho": partial_500_story[0],
                "p": partial_500_story[1],
            },
            "schaefer500_math_given_story": {
                "rho": partial_500_math[0],
                "p": partial_500_math[1],
            },
            "schaefer1000_story_given_math": {
                "rho": partial_1000_story[0],
                "p": partial_1000_story[1],
            },
            "schaefer1000_math_given_story": {
                "rho": partial_1000_math[0],
                "p": partial_1000_math[1],
            },
        },
        "leave_one_subject_out": {
            "schaefer500": leave_one_out(brain_500),
            "schaefer1000": leave_one_out(brain_1000),
        },
    }

    return {
        "experiment": "Schaefer-1000 replication of three frozen Language candidates",
        "subjects": 29,
        "state": STATE,
        "candidates": [candidate_key(candidate) for candidate in CANDIDATES],
        "permutations": PERMUTATIONS,
        "bootstraps": BOOTSTRAPS,
        "primary_family": "three Williams Story-minus-Math difference tests",
        "strict_replication_rule": (
            "Story rho > 0, Math rho < 0, delta rho > 0, "
            "and Holm-adjusted Williams p < 0.05"
        ),
        "schaefer500": results_500,
        "schaefer1000": results_1000,
        "validation": validation,
        "som_lim_cont_sensitivity": sensitivity,
        "strictly_replicated_candidates": [
            item["coalition"] for item in validation if item["strict_replication"]
        ],
    }


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def plot_results(summary: dict[str, Any]) -> None:
    configure_style()
    rows = []
    for index, candidate in enumerate(CANDIDATES):
        rows.append(summary["schaefer500"][index])
        rows.append(summary["schaefer1000"][index])
    matrix = np.asarray([
        [
            row["story_rho"],
            row["math_rho"],
            row["rho_story_minus_math"],
        ]
        for row in rows
    ])
    figure, (heatmap, forest) = plt.subplots(
        1,
        2,
        figsize=(7.3, 3.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.5, 1.05)},
    )
    image = heatmap.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-0.68,
        vmax=0.68,
        aspect="auto",
        interpolation="nearest",
    )
    heatmap.set_xticks(
        [0, 1, 2],
        [r"Story $\rho$", r"Math $\rho$", r"$\Delta\rho$"],
    )
    heatmap.xaxis.tick_top()
    labels = []
    for index, candidate in enumerate(CANDIDATES):
        labels.extend([
            f"{short_name(candidate)} · S500",
            f"{short_name(candidate)} · S1000",
        ])
    heatmap.set_yticks(range(len(rows)), labels)
    heatmap.tick_params(length=0)
    heatmap.set_title("a  Fixed-coalition coefficients", loc="left", weight="bold", pad=24)
    for row_index, row in enumerate(rows):
        values = (
            row["story_rho"],
            row["math_rho"],
            row["rho_story_minus_math"],
        )
        p_values = (
            row["story_holm_p_3"],
            row["math_holm_p_3"],
            row["difference_holm_p_3"],
        )
        for column_index, (value, p_value) in enumerate(zip(values, p_values)):
            marker = "*" if p_value < 0.05 else ""
            heatmap.text(
                column_index,
                row_index,
                f"{value:+.2f}{marker}",
                ha="center",
                va="center",
                fontsize=6.4,
                weight="bold",
                color="white" if abs(value) > 0.40 else "#263238",
            )
    colorbar = figure.colorbar(
        image,
        ax=heatmap,
        orientation="horizontal",
        fraction=0.08,
        pad=0.10,
    )
    colorbar.set_label(r"Spearman correlation or Story − Math difference")

    colors = {"Schaefer-500": "#4C78A8", "Schaefer-1000": "#8064A2"}
    offsets = {"Schaefer-500": -0.10, "Schaefer-1000": 0.10}
    y = np.arange(len(CANDIDATES))
    forest.axvline(0, color="#B5BDC4", lw=0.8)
    for atlas_key in ("schaefer500", "schaefer1000"):
        atlas_rows = summary[atlas_key]
        atlas = atlas_rows[0]["atlas"]
        for index, row in enumerate(atlas_rows):
            bootstrap = row["difference_bootstrap"]
            forest.errorbar(
                row["rho_story_minus_math"],
                y[index] + offsets[atlas],
                xerr=np.asarray([[
                    row["rho_story_minus_math"] - bootstrap["ci95_low"]
                ], [
                    bootstrap["ci95_high"] - row["rho_story_minus_math"]
                ]]),
                fmt="o" if atlas == "Schaefer-500" else "D",
                color=colors[atlas],
                ecolor=colors[atlas],
                capsize=2.2,
                markersize=4.5,
                lw=1,
                label=atlas if index == 0 else None,
            )
    forest.set_yticks(y, [short_name(candidate) for candidate in CANDIDATES])
    forest.invert_yaxis()
    forest.set_xlabel(r"$\Delta\rho$ (Story − Math), bootstrap 95% CI")
    forest.set_title("b  Cross-atlas replication", loc="left", weight="bold")
    forest.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    for extension in ("png", "svg", "pdf"):
        figure.savefig(
            OUTPUT_DIR / f"story_math_candidate_replication.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def plot_som_lim_cont_detail(
    summary: dict[str, Any],
    values_500: np.ndarray,
    values_1000: np.ndarray,
    story: np.ndarray,
    math: np.ndarray,
) -> None:
    """Plot Schaefer-1000 behavior associations and cross-atlas stability."""

    configure_style()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.3, 2.55),
        constrained_layout=True,
    )
    story_color = "#D66A4E"
    math_color = "#4C78A8"
    atlas_color = "#8064A2"
    rng = np.random.default_rng(2026072903)
    story_jitter = rng.uniform(-0.45, 0.45, size=len(story))
    math_jitter = rng.uniform(-0.20, 0.20, size=len(math))

    def scatter_with_guide(
        axis: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        *,
        color: str,
        annotation: str,
        x_jitter: np.ndarray | None = None,
        identity: bool = False,
    ) -> None:
        displayed_x = x if x_jitter is None else x + x_jitter
        axis.scatter(
            displayed_x,
            y,
            s=22,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        lower, upper = float(np.min(x)), float(np.max(x))
        guide_x = np.linspace(lower, upper, 100)
        slope, intercept = np.polyfit(x, y, 1)
        axis.plot(
            guide_x,
            slope * guide_x + intercept,
            color=color,
            lw=1.1,
            ls=(0, (3, 2)),
            zorder=2,
        )
        if identity:
            limits = [
                min(float(np.min(x)), float(np.min(y))),
                max(float(np.max(x)), float(np.max(y))),
            ]
            axis.plot(
                limits,
                limits,
                color="#C2C7CC",
                lw=0.8,
                zorder=1,
            )
        axis.text(
            0.04,
            0.95,
            annotation,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.3,
            color="#263238",
        )
        axis.grid(color="#E8EBED", linewidth=0.55, zorder=0)

    statistics = summary["schaefer1000"][0]
    scatter_with_guide(
        axes[0],
        story,
        values_1000[0],
        color=story_color,
        annotation=(
            rf"$\rho={statistics['story_rho']:+.3f}$"
            f"\nperm. $p={statistics['story_p_permutation']:.4f}$"
        ),
        x_jitter=story_jitter,
    )
    scatter_with_guide(
        axes[1],
        math,
        values_1000[0],
        color=math_color,
        annotation=(
            rf"$\rho={statistics['math_rho']:+.3f}$"
            f"\nperm. $p={statistics['math_p_permutation']:.4f}$"
        ),
        x_jitter=math_jitter,
    )
    axes[0].set_ylabel("Schaefer-1000\ncoalition synergy (bits)")
    axes[0].set_xlabel("Story accuracy (%)")
    axes[1].set_xlabel("Math accuracy (%)")
    axes[0].set_ylim(0.04, 1.19)
    axes[1].set_ylim(0.04, 1.19)

    sensitivity = summary["som_lim_cont_sensitivity"]
    cross_atlas = sensitivity["cross_atlas"]
    scatter_with_guide(
        axes[2],
        values_500[0],
        values_1000[0],
        color=atlas_color,
        annotation=(
            rf"$\rho={cross_atlas['rho']:+.3f}$"
            f"\n$p={cross_atlas['p']:.2g}$"
        ),
        identity=True,
    )
    axes[2].set_xlabel("Schaefer-500 synergy (bits)")
    axes[2].set_ylabel("Schaefer-1000 synergy (bits)")
    axes[2].set_xlim(0.04, 1.19)
    axes[2].set_ylim(0.04, 1.19)

    panel_titles = (
        "a  Story association",
        "b  Math association",
        "c  Cross-atlas agreement",
    )
    for axis, title in zip(axes, panel_titles):
        axis.set_title(title, loc="left", weight="bold")

    for extension in ("png", "svg", "pdf"):
        figure.savefig(
            OUTPUT_DIR / f"som_lim_cont_story_math_detail.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Schaefer-1000 replication of three Language candidates",
        "",
        "The three candidates were frozen from the Schaefer-500 Story-minus-Math "
        "exploratory analysis. The primary replication family contains three "
        "two-sided Williams difference tests, Holm-adjusted across candidates.",
        "",
        "| Coalition | Atlas | Story rho (p) | Math rho (p) | Delta rho | "
        "Williams p | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for index, candidate in enumerate(CANDIDATES):
        for atlas_key in ("schaefer500", "schaefer1000"):
            row = summary[atlas_key][index]
            lines.append(
                f"| {short_name(candidate)} | {row['atlas'].replace('Schaefer-', 'S')} | "
                f"{row['story_rho']:+.3f} ({row['story_p_permutation']:.4f}) | "
                f"{row['math_rho']:+.3f} ({row['math_p_permutation']:.4f}) | "
                f"{row['rho_story_minus_math']:+.3f} | "
                f"{row['williams_p']:.4f} | {row['difference_holm_p_3']:.4f} |"
            )
    lines += [
        "",
        "## Replication audit",
        "",
        "| Coalition | Metric rho across atlases | Direction pattern preserved | "
        "Strictly replicated |",
        "|---|---:|---|---|",
    ]
    for item in summary["validation"]:
        lines.append(
            f"| {item['coalition_short']} | {item['cross_atlas_metric_rho']:+.3f} | "
            f"{'yes' if item['direction_pattern_preserved'] else 'no'} | "
            f"{'yes' if item['strict_replication'] else 'no'} |"
        )
    lines += [
        "",
        "Strict replication requires Story rho > 0, Math rho < 0, positive delta "
        "rho, and Holm-adjusted Williams p < 0.05 in Schaefer-1000.",
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subjects = load_subjects()
    values_1000 = compute_values(
        subjects, smoke=args.smoke, recompute=args.recompute
    )
    if args.smoke:
        print(json.dumps({
            "subject": subjects[0],
            "coalitions": [candidate_key(candidate) for candidate in CANDIDATES],
            "values": values_1000[:, 0].tolist(),
            "all_finite_positive": bool(
                np.isfinite(values_1000).all() and np.all(values_1000 > 0)
            ),
        }, indent=2))
        return
    values_500 = load_values_500(subjects)
    story, math = load_behavior(subjects)
    summary = analyze(subjects, values_500, values_1000, story, math)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(summary)
    plot_results(summary)
    plot_som_lim_cont_detail(
        summary, values_500, values_1000, story, math
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
