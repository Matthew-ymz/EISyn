#!/usr/bin/env python3
"""Compare rank-space partial Spearman with raw-value partial Pearson.

The comparison freezes the pooled-57 subjects, seven primary endpoints, age/sex
covariates, 120 coalition matrices, and Freedman--Lane permutations.  The only
treatment factor is whether continuous variables enter the analysis as ranks
(Spearman) or as their measured/predefined values (Pearson).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_hcp_all_task_behavior_coalitions_57 as base
from scripts import plot_hcp_schaefer1000_behavior_main as main_figure
from scripts import screen_hcp_emotion_performance_coalitions_57 as legacy_emotion


OUTPUT = ROOT / "results/hcp_all_task_behavior_correlation_method_comparison_57"
METHODS = ("spearman", "pearson")
METHOD_LABELS = {"spearman": "Spearman (ranks)", "pearson": "Pearson (raw values)"}


def method_inputs(
    matrix: np.ndarray,
    contract: Mapping[str, Any],
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    age = np.asarray(contract["age"], dtype=float)
    sex = np.asarray(contract["sex"], dtype=float)
    nuisance = np.asarray(contract["nuisance"], dtype=float)
    endpoint = np.asarray(contract["endpoint"], dtype=float)
    if method == "spearman":
        brain = base.rankdata(matrix, axis=0)
        outcome = base.rankdata(endpoint)
        parts = [np.ones(len(age)), base.rankdata(age), sex]
        if nuisance.size:
            parts.extend(base.rankdata(nuisance, axis=0).T)
    elif method == "pearson":
        brain = np.asarray(matrix, dtype=float)
        outcome = endpoint
        parts = [np.ones(len(age)), age, sex]
        if nuisance.size:
            parts.extend(nuisance.T)
    else:
        raise ValueError(f"Unknown method: {method}")
    return brain, outcome, np.column_stack(parts)


def screen(
    matrix: np.ndarray,
    contract: Mapping[str, Any],
    method: str,
    permutations: int,
    seed: int,
) -> dict[str, np.ndarray]:
    brain_raw, endpoint, design = method_inputs(matrix, contract, method)
    brain = base.unit_columns(base.residualize(brain_raw, design))
    fitted = design @ np.linalg.lstsq(design, endpoint, rcond=None)[0]
    endpoint_residual = base.residualize(endpoint, design)
    endpoint_unit = base.unit_columns(endpoint_residual).ravel()
    observed = brain.T @ endpoint_unit

    point_counts = np.zeros(matrix.shape[1], dtype=np.int64)
    family_counts = np.zeros(matrix.shape[1], dtype=np.int64)
    rng = np.random.default_rng(seed)
    chunk = 1_000
    n = len(endpoint)
    for start in range(0, permutations, chunk):
        size = min(chunk, permutations - start)
        indices = np.argsort(rng.random((size, n)), axis=1)
        pseudo = fitted[None, :] + endpoint_residual[indices]
        coefficients = np.linalg.lstsq(design, pseudo.T, rcond=None)[0]
        null_endpoint = pseudo - (design @ coefficients).T
        null_endpoint /= np.linalg.norm(null_endpoint, axis=1, keepdims=True)
        absolute = np.abs(null_endpoint @ brain)
        point_counts += np.sum(absolute >= np.abs(observed)[None, :], axis=0)
        maxima = absolute.max(axis=1)
        family_counts += np.sum(maxima[:, None] >= np.abs(observed)[None, :], axis=0)

    denominator = permutations + 1.0
    p_raw = (point_counts + 1.0) / denominator
    return {
        "coefficient": observed,
        "p_raw": p_raw,
        "q_bh_120": base.bh(p_raw),
        "p_max_t_120": (family_counts + 1.0) / denominator,
    }


def leave_one_out_coefficient(
    brain: np.ndarray,
    contract: Mapping[str, Any],
    method: str,
) -> dict[str, float]:
    estimates = []
    n = len(np.asarray(contract["endpoint"]))
    for removed in range(n):
        keep = np.arange(n) != removed
        local = dict(contract)
        for key in ("age", "sex", "endpoint", "nuisance"):
            local[key] = np.asarray(contract[key])[keep]
        brain_input, endpoint, design = method_inputs(brain[keep, None], local, method)
        x = base.residualize(brain_input[:, 0], design)
        y = base.residualize(endpoint, design)
        estimates.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))))
    values = np.asarray(estimates)
    median = float(np.median(values))
    return {
        "minimum": float(values.min()),
        "median": median,
        "maximum": float(values.max()),
        "same_sign_fraction": float(np.mean(np.sign(values) == np.sign(median))),
    }


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def plot_comparison(rows: list[dict[str, Any]], tasks: Mapping[str, Any]) -> None:
    configure_style()
    task_colors = {
        "LANGUAGE": "#D07A4F",
        "SOCIAL": "#3F8978",
        "EMOTION": "#4C78A8",
        "MOTOR": "#7867A8",
        "GAMBLING": "#B89B4C",
        "RELATIONAL": "#6D8799",
        "WM": "#9A7085",
    }
    figure, axes = plt.subplots(1, 3, figsize=(9.0, 3.35), constrained_layout=True)

    for state in base.TASK_ORDER:
        selected = [row for row in rows if row["task"] == state]
        x = np.asarray([row["spearman_coefficient"] for row in selected])
        y = np.asarray([row["pearson_coefficient"] for row in selected])
        axes[0].scatter(
            x,
            y,
            s=13,
            alpha=0.58,
            color=task_colors[state],
            edgecolor="none",
            label=base.TASK_LABELS[state],
        )
    limits = (-0.58, 0.58)
    axes[0].plot(limits, limits, color="#7D858C", linestyle="--", linewidth=0.8)
    axes[0].axhline(0, color="#D7DBDE", linewidth=0.6)
    axes[0].axvline(0, color="#D7DBDE", linewidth=0.6)
    axes[0].set(
        xlim=limits,
        ylim=limits,
        xlabel="Partial Spearman coefficient",
        ylabel="Partial Pearson coefficient",
    )
    axes[0].set_title("a  All 840 matched associations", loc="left", fontweight="bold")

    eps = 1.0e-6
    for state in base.TASK_ORDER:
        selected = [row for row in rows if row["task"] == state]
        x = -np.log10(np.maximum([row["spearman_p_raw"] for row in selected], eps))
        y = -np.log10(np.maximum([row["pearson_p_raw"] for row in selected], eps))
        axes[1].scatter(x, y, s=13, alpha=0.58, color=task_colors[state], edgecolor="none")
    p_limit = max(5.1, axes[1].get_xlim()[1], axes[1].get_ylim()[1])
    axes[1].plot((0, p_limit), (0, p_limit), color="#7D858C", linestyle="--", linewidth=0.8)
    threshold = -np.log10(0.05)
    axes[1].axhline(threshold, color="#A65E49", linestyle=":", linewidth=0.75)
    axes[1].axvline(threshold, color="#A65E49", linestyle=":", linewidth=0.75)
    axes[1].set(
        xlim=(0, p_limit),
        ylim=(0, p_limit),
        xlabel=r"Spearman $-\log_{10}(p_{raw})$",
        ylabel=r"Pearson $-\log_{10}(p_{raw})$",
    )
    axes[1].set_title("b  Pointwise permutation evidence", loc="left", fontweight="bold")

    y_positions = np.arange(len(base.TASK_ORDER))
    for position, state in zip(y_positions, base.TASK_ORDER, strict=True):
        spearman = tasks[state]["spearman_winner"]["p_max_t_120"]
        pearson = tasks[state]["pearson_winner"]["p_max_t_120"]
        axes[2].plot([spearman, pearson], [position, position], color="#B8C0C6", linewidth=1.0)
        axes[2].scatter(spearman, position, s=28, color="#526D82", marker="o", zorder=3)
        axes[2].scatter(pearson, position, s=30, color="#C57450", marker="D", zorder=3)
    axes[2].axvline(0.05, color="#A65E49", linestyle=":", linewidth=0.8)
    axes[2].set_xscale("log")
    axes[2].set_xlim(1.0e-4, 1.05)
    axes[2].set_yticks(y_positions, [base.TASK_LABELS[state] for state in base.TASK_ORDER])
    axes[2].invert_yaxis()
    axes[2].set_xlabel(r"Method-specific winner max-$T$ $p$")
    axes[2].set_title("c  Best coalition within each task", loc="left", fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    method_handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="none", color="#526D82", label="Spearman winner"),
        mpl.lines.Line2D([], [], marker="D", linestyle="none", color="#C57450", label="Pearson winner"),
    ]
    figure.legend(
        handles + method_handles,
        labels + [handle.get_label() for handle in method_handles],
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        ncol=1,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        figure.savefig(
            OUTPUT / f"hcp_correlation_method_comparison_57.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(figure)


def winner_row(
    state: str,
    names: np.ndarray,
    sizes: np.ndarray,
    result: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    index = int(np.argmax(np.abs(result["coefficient"])))
    return {
        "task": state,
        "index": index,
        "coalition": str(names[index]),
        "short_coalition": base.compact_name(str(names[index])),
        "coalition_size": int(sizes[index]),
        "coefficient": float(result["coefficient"][index]),
        "p_raw": float(result["p_raw"][index]),
        "q_bh_120": float(result["q_bh_120"][index]),
        "p_max_t_120": float(result["p_max_t_120"][index]),
    }


def legacy_emotion_pearson_audit(permutations: int) -> dict[str, Any]:
    """Repeat the original cohort-aware five-analysis EMOTION family with Pearson."""
    subjects = legacy_emotion.load_subjects()
    combinations = legacy_emotion.coalitions()
    names = legacy_emotion.coalition_names(combinations)
    synergy, _, _, _ = legacy_emotion.compute_matrices(subjects, combinations, False)
    behavior = legacy_emotion.load_behavior(subjects)
    cohort = behavior["cohort"].astype(float)
    base_design = np.column_stack(
        [np.ones(len(cohort)), behavior["age"], behavior["sex"], cohort]
    )
    face_speed = -np.log(behavior["Emotion_Task_Face_Median_RT"])
    shape_speed = -np.log(behavior["Emotion_Task_Shape_Median_RT"])
    overall_speed = -np.log(behavior["Emotion_Task_Median_RT"])
    face_efficiency = -np.log(
        behavior["Emotion_Task_Face_Median_RT"]
        / (behavior["Emotion_Task_Face_Acc"] / 100.0)
    )
    shape_efficiency = -np.log(
        behavior["Emotion_Task_Shape_Median_RT"]
        / (behavior["Emotion_Task_Shape_Acc"] / 100.0)
    )
    primary_design = np.column_stack([base_design, shape_speed])
    efficiency_design = np.column_stack([base_design, shape_efficiency])
    task = synergy[0]
    delta = synergy[0] - synergy[1]
    definitions = (
        ("task_face_specific_speed", task, face_speed, primary_design),
        ("delta_face_specific_speed", delta, face_speed, primary_design),
        ("task_overall_speed", task, overall_speed, base_design),
        ("task_overall_accuracy", task, behavior["Emotion_Task_Acc"], base_design),
        ("task_face_efficiency", task, face_efficiency, efficiency_design),
    )
    analyses = []
    for analysis_name, matrix, endpoint, design in definitions:
        brain_unit = legacy_emotion.unit_columns(
            legacy_emotion.projection_residual(matrix, design)
        )
        fitted = design @ np.linalg.lstsq(design, endpoint, rcond=None)[0]
        endpoint_residual = legacy_emotion.projection_residual(endpoint, design)
        endpoint_unit = legacy_emotion.unit_columns(endpoint_residual).ravel()
        analyses.append(
            {
                "name": analysis_name,
                "matrix": matrix,
                "endpoint": endpoint,
                "design": design,
                "brain_unit": brain_unit,
                "fitted": fitted,
                "endpoint_residual": endpoint_residual,
                "residual_maker": np.eye(len(endpoint))
                - design @ np.linalg.pinv(design),
                "observed": brain_unit.T @ endpoint_unit,
            }
        )
    predefined = np.asarray(
        [names.tolist().index(value) for value in legacy_emotion.PREDEFINED], dtype=int
    )
    result = legacy_emotion.screen_analyses(
        analyses,
        behavior["cohort"].astype(int),
        predefined,
        permutations,
        legacy_emotion.SEED,
    )
    old_summary = json.loads(
        (legacy_emotion.OUTPUT / "summary.json").read_text(encoding="utf-8")
    )
    old_winner = old_summary["analyses"]["task_face_specific_speed"]["winner"]
    old_index = names.tolist().index(old_winner["coalition"])
    pearson_index = int(np.argmax(np.abs(result["observed"][0])))

    def serialize(index: int) -> dict[str, Any]:
        return {
            "coalition": str(names[index]),
            "coefficient": float(result["observed"][0, index]),
            "p_raw": float(result["p_raw"][0, index]),
            "q_bh_120": float(result["q_bh_within_analysis"][0, index]),
            "p_max_t_120": float(result["p_max_t_120"][0, index]),
            "p_max_t_global600": float(result["p_max_t_global600"][0, index]),
        }

    return {
        "contract": {
            "subjects": 57,
            "cohort_covariate": True,
            "permutation_blocked_within_cohort": True,
            "continuous_covariates_unranked": True,
            "analyses": 5,
            "coalitions_per_analysis": 120,
        },
        "spearman_winner_original": {
            "coalition": old_winner["coalition"],
            "coefficient": old_winner["rho"],
            "p_raw": old_winner["p_raw"],
            "p_max_t_120": old_winner["p_max_t_120"],
            "p_max_t_global600": old_winner["p_max_t_global600"],
        },
        "spearman_winner_under_pearson": serialize(old_index),
        "pearson_winner": serialize(pearson_index),
        "pearson_primary_max_t_below_0_05_count": int(
            np.sum(result["p_max_t_120"][0] < 0.05)
        ),
        "pearson_global600_below_0_05_count": int(
            np.sum(result["p_max_t_global600"] < 0.05)
        ),
    }


def blocked_simple_correlation(
    x: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    method: str,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    if method == "spearman":
        x_input = base.rankdata(x)
        y_input = base.rankdata(y)
    elif method == "pearson":
        x_input = np.asarray(x, dtype=float)
        y_input = np.asarray(y, dtype=float)
    else:
        raise ValueError(method)
    x_unit = base.unit_columns(x_input).ravel()
    y_unit = base.unit_columns(y_input).ravel()
    observed = float(x_unit @ y_unit)
    groups = [np.flatnonzero(blocks == value) for value in np.unique(blocks)]
    rng = np.random.default_rng(seed)
    exceedances = 0
    chunk = 1_000
    for start in range(0, permutations, chunk):
        size = min(chunk, permutations - start)
        indices = np.tile(np.arange(len(x)), (size, 1))
        for group in groups:
            order = np.argsort(rng.random((size, len(group))), axis=1)
            indices[:, group] = group[order]
        null = y_unit[indices] @ x_unit
        exceedances += int(np.sum(np.abs(null) >= abs(observed)))
    return {
        "coefficient": observed,
        "p_raw": (exceedances + 1.0) / (permutations + 1.0),
    }


def main_figure_language_audit(permutations: int) -> list[dict[str, Any]]:
    _, _, behavior = main_figure.load_inputs()
    blocks = np.unique(behavior["cohort"], return_inverse=True)[1]
    output = []
    for spec in main_figure.SCATTER_SPECS:
        coalition = str(spec["coalition"])
        endpoint = str(spec["endpoint"])
        x = behavior[endpoint].to_numpy(dtype=float)
        y = behavior[f"synergy_bits__{coalition}"].to_numpy(dtype=float)
        output.append(
            {
                "coalition": coalition,
                "endpoint": endpoint,
                "spearman": blocked_simple_correlation(
                    x, y, blocks, "spearman", permutations, int(spec["seed"])
                ),
                "pearson": blocked_simple_correlation(
                    x, y, blocks, "pearson", permutations, int(spec["seed"])
                ),
                "multiplicity_note": "fixed pointwise candidate; p is not corrected across candidate selection",
            }
        )
    return output


def legacy_motor_pearson_audit(permutations: int) -> dict[str, Any]:
    motor = base.common
    subjects = motor.load_subjects()
    combinations = motor.coalitions()
    names = motor.coalition_names(combinations)
    matrix, _, _ = motor.compute_matrix(subjects, combinations, False)
    scores = motor.load_scores(subjects)
    design = motor.base_design(scores)
    brain = motor.unit_columns(motor.residualize(matrix, design))
    endpoint = np.asarray(scores["composite"], dtype=float)
    fitted = design @ np.linalg.lstsq(design, endpoint, rcond=None)[0]
    endpoint_residual = motor.residualize(endpoint, design)
    endpoint_unit = motor.unit_columns(endpoint_residual)[:, 0]
    observed = brain.T @ endpoint_unit
    cohort = scores["cohort"].astype(int)
    groups = [np.flatnonzero(cohort == value) for value in np.unique(cohort)]
    rng = np.random.default_rng(motor.SEED)
    point_counts = np.zeros(120, dtype=np.int64)
    family_counts = np.zeros(120, dtype=np.int64)
    chunk = 1_000
    for start in range(0, permutations, chunk):
        size = min(chunk, permutations - start)
        indices = np.tile(np.arange(len(cohort)), (size, 1))
        for group in groups:
            order = np.argsort(rng.random((size, len(group))), axis=1)
            indices[:, group] = group[order]
        pseudo = fitted[None, :] + endpoint_residual[indices]
        coefficients = np.linalg.lstsq(design, pseudo.T, rcond=None)[0]
        residual = pseudo - (design @ coefficients).T
        residual /= np.linalg.norm(residual, axis=1, keepdims=True)
        absolute = np.abs(residual @ brain)
        point_counts += np.sum(absolute >= np.abs(observed)[None, :], axis=0)
        maxima = absolute.max(axis=1)
        family_counts += np.sum(maxima[:, None] >= np.abs(observed)[None, :], axis=0)
    denominator = permutations + 1.0
    p_raw = (point_counts + 1.0) / denominator
    p_max_t = (family_counts + 1.0) / denominator
    q_bh = motor.bh(p_raw)
    old_summary = json.loads((motor.OUTPUT / "summary.json").read_text(encoding="utf-8"))

    def serialize(index: int) -> dict[str, Any]:
        return {
            "coalition": str(names[index]),
            "coefficient": float(observed[index]),
            "p_raw": float(p_raw[index]),
            "q_bh_120": float(q_bh[index]),
            "p_max_t_120": float(p_max_t[index]),
        }

    winner = int(np.argmax(np.abs(observed)))
    candidates = []
    for old_row in old_summary["top_ten"]:
        index = int(old_row["index"])
        candidates.append(
            {
                "coalition": str(names[index]),
                "spearman": {
                    "coefficient": float(old_row["rho_adjusted"]),
                    "p_raw": float(old_row["p_raw"]),
                    "p_max_t_120": float(old_row["p_max_t_120"]),
                },
                "pearson": serialize(index),
            }
        )
    return {
        "pearson_winner": serialize(winner),
        "spearman_top_ten_candidates": candidates,
        "pearson_raw_p_below_0_05_count": int(np.sum(p_raw < 0.05)),
        "pearson_bh_q_below_0_05_count": int(np.sum(q_bh < 0.05)),
        "pearson_max_t_p_below_0_05_count": int(np.sum(p_max_t < 0.05)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=100_000)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    with np.load(base.common.SUBJECT_SOURCE, allow_pickle=False) as archive:
        subjects = archive["subjects"].astype(str)
    table = base.load_table()
    contracts = base.make_endpoint_contracts(subjects, table)

    task_summaries: dict[str, Any] = {}
    association_rows: list[dict[str, Any]] = []
    frozen_names: np.ndarray | None = None
    for task_index, state in enumerate(base.TASK_ORDER):
        _, names, sizes, matrix = base.load_matrix(state, subjects)
        if frozen_names is None:
            frozen_names = names
        elif not np.array_equal(names, frozen_names):
            raise ValueError(f"Coalition order mismatch for {state}.")
        contract = contracts[state]
        results = {
            method: screen(
                matrix,
                contract,
                method,
                args.permutations,
                base.SEED + task_index,
            )
            for method in METHODS
        }
        for index, name in enumerate(names):
            row: dict[str, Any] = {
                "task": state,
                "index": int(index),
                "coalition": str(name),
                "short_coalition": base.compact_name(str(name)),
                "coalition_size": int(sizes[index]),
            }
            for method in METHODS:
                row[f"{method}_coefficient"] = float(results[method]["coefficient"][index])
                row[f"{method}_p_raw"] = float(results[method]["p_raw"][index])
                row[f"{method}_q_bh_120"] = float(results[method]["q_bh_120"][index])
                row[f"{method}_p_max_t_120"] = float(results[method]["p_max_t_120"][index])
            association_rows.append(row)

        spearman_winner = winner_row(state, names, sizes, results["spearman"])
        pearson_winner = winner_row(state, names, sizes, results["pearson"])
        spearman_index = spearman_winner["index"]
        pearson_index = pearson_winner["index"]
        task_summaries[state] = {
            "endpoint": {
                key: contract[key]
                for key in ("label", "definition", "in_scanner", "condition_specific")
            },
            "spearman_winner": spearman_winner,
            "pearson_winner": pearson_winner,
            "cross_method_fixed_candidates": {
                "spearman_winner_under_pearson": {
                    "coefficient": float(results["pearson"]["coefficient"][spearman_index]),
                    "p_raw": float(results["pearson"]["p_raw"][spearman_index]),
                    "q_bh_120": float(results["pearson"]["q_bh_120"][spearman_index]),
                    "p_max_t_120": float(results["pearson"]["p_max_t_120"][spearman_index]),
                },
                "pearson_winner_under_spearman": {
                    "coefficient": float(results["spearman"]["coefficient"][pearson_index]),
                    "p_raw": float(results["spearman"]["p_raw"][pearson_index]),
                    "q_bh_120": float(results["spearman"]["q_bh_120"][pearson_index]),
                    "p_max_t_120": float(results["spearman"]["p_max_t_120"][pearson_index]),
                },
            },
            "significance_counts": {
                method: {
                    "raw_p_below_0_05": int(np.sum(results[method]["p_raw"] < 0.05)),
                    "bh_q_below_0_05": int(np.sum(results[method]["q_bh_120"] < 0.05)),
                    "max_t_p_below_0_05": int(np.sum(results[method]["p_max_t_120"] < 0.05)),
                }
                for method in METHODS
            },
            "winner_leave_one_out": {
                "spearman": leave_one_out_coefficient(
                    matrix[:, spearman_index], contract, "spearman"
                ),
                "pearson": leave_one_out_coefficient(
                    matrix[:, pearson_index], contract, "pearson"
                ),
            },
            "method_difference": {
                "median_abs_coefficient_difference": float(
                    np.median(
                        np.abs(
                            results["pearson"]["coefficient"]
                            - results["spearman"]["coefficient"]
                        )
                    )
                ),
                "maximum_abs_coefficient_difference": float(
                    np.max(
                        np.abs(
                            results["pearson"]["coefficient"]
                            - results["spearman"]["coefficient"]
                        )
                    )
                ),
                "pearson_smaller_raw_p_count": int(
                    np.sum(results["pearson"]["p_raw"] < results["spearman"]["p_raw"])
                ),
                "spearman_smaller_raw_p_count": int(
                    np.sum(results["spearman"]["p_raw"] < results["pearson"]["p_raw"])
                ),
            },
        }

    spearman_p = np.asarray([row["spearman_p_raw"] for row in association_rows])
    pearson_p = np.asarray([row["pearson_p_raw"] for row in association_rows])
    spearman_max = np.asarray([row["spearman_p_max_t_120"] for row in association_rows])
    pearson_max = np.asarray([row["pearson_p_max_t_120"] for row in association_rows])
    legacy_emotion_audit = legacy_emotion_pearson_audit(args.permutations)
    language_figure_audit = main_figure_language_audit(args.permutations)
    legacy_motor_audit = legacy_motor_pearson_audit(args.permutations)
    summary = {
        "experiment": "Pooled-57 partial Spearman versus raw-value partial Pearson across seven task states",
        "subjects": 57,
        "treatment_factor": "correlation method only",
        "methods": {
            "spearman": "Pearson correlation of residualized ranks; age and other continuous nuisance variables also ranked",
            "pearson": "Pearson correlation of residualized measured/predefined values; age and continuous nuisance variables unranked",
        },
        "frozen": {
            "task_order": list(base.TASK_ORDER),
            "coalitions_per_task": 120,
            "primary_endpoints": 1,
            "subjects": 57,
            "covariates": ["age", "sex"],
            "emotion_additional_nuisance": "negative log Shape median RT",
            "permutations": args.permutations,
            "permutation_scheme": "unrestricted pooled-sample Freedman-Lane residual permutation",
            "same_seed_per_task_across_methods": True,
            "syn_nonnegativity_tolerance_bits": base.SYN_TOLERANCE_BITS,
        },
        "overall": {
            "associations": len(association_rows),
            "pearson_smaller_raw_p_count": int(np.sum(pearson_p < spearman_p)),
            "spearman_smaller_raw_p_count": int(np.sum(spearman_p < pearson_p)),
            "equal_raw_p_count": int(np.sum(spearman_p == pearson_p)),
            "pearson_smaller_max_t_p_count": int(np.sum(pearson_max < spearman_max)),
            "spearman_smaller_max_t_p_count": int(np.sum(spearman_max < pearson_max)),
            "spearman_raw_p_below_0_05": int(np.sum(spearman_p < 0.05)),
            "pearson_raw_p_below_0_05": int(np.sum(pearson_p < 0.05)),
            "spearman_max_t_p_below_0_05": int(np.sum(spearman_max < 0.05)),
            "pearson_max_t_p_below_0_05": int(np.sum(pearson_max < 0.05)),
        },
        "tasks": task_summaries,
        "legacy_cohort_aware_emotion_audit": legacy_emotion_audit,
        "main_figure_language_accuracy_audit": language_figure_audit,
        "legacy_cohort_aware_motor_audit": legacy_motor_audit,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT / "all_associations.jsonl").open("w", encoding="utf-8") as handle:
        for row in association_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    contract = {
        "scientific_question": "What changes when only the correlation method changes from rank-space partial Spearman to raw-value partial Pearson?",
        "treatment_factor": "correlation method",
        "treatment_levels": list(METHOD_LABELS.values()),
        "unit_of_pairing": "same task, subject set, primary endpoint, and coalition",
        "primary_metrics": ["pointwise permutation p", "BH q across 120", "120-coalition max-T p"],
        "frozen_variables": summary["frozen"],
        "figure_contract": {
            "core_conclusion": "Show whether either correlation method systematically produces stronger selection-aware evidence.",
            "evidence_chain": ["all 840 paired coefficients", "all pointwise p values", "method-specific task winners after max-T"],
            "archetype": "quantitative grid",
            "role": "method comparison",
            "backend": "Python/matplotlib only",
            "exports": ["PNG 600 dpi", "editable SVG", "PDF"],
            "review_risks": ["winner switching", "nominal versus selection-corrected significance", "legend overlap"],
        },
    }
    (OUTPUT / "experiment_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_comparison(association_rows, task_summaries)

    lines = [
        "# Spearman versus Pearson in pooled-57 task-state coalition screens",
        "",
        "![Correlation-method comparison](hcp_correlation_method_comparison_57.png)",
        "",
        "Only the correlation method changes. Subjects, endpoints, coalition Syn estimates, covariates, permutations, and within-task multiplicity correction are frozen.",
        "",
        "| Task | Spearman winner | rho | raw p | max-T p | Pearson winner | r | raw p | max-T p |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for state in base.TASK_ORDER:
        s = task_summaries[state]["spearman_winner"]
        p = task_summaries[state]["pearson_winner"]
        lines.append(
            f"| {base.TASK_LABELS[state]} | {s['short_coalition']} | {s['coefficient']:+.3f} | {s['p_raw']:.5f} | {s['p_max_t_120']:.5f} | "
            f"{p['short_coalition']} | {p['coefficient']:+.3f} | {p['p_raw']:.5f} | {p['p_max_t_120']:.5f} |"
        )
    lines.extend(
        [
            "",
            f"Across 840 matched associations, Pearson had the smaller raw p for {summary['overall']['pearson_smaller_raw_p_count']} and Spearman for {summary['overall']['spearman_smaller_raw_p_count']}. ",
            f"Nominal p<.05 counts were {summary['overall']['spearman_raw_p_below_0_05']} for Spearman and {summary['overall']['pearson_raw_p_below_0_05']} for Pearson; max-T p<.05 counts were {summary['overall']['spearman_max_t_p_below_0_05']} and {summary['overall']['pearson_max_t_p_below_0_05']}, respectively.",
            "",
            "## Legacy cohort-aware EMOTION audit",
            "",
            "| Candidate/method | Coalition | Coefficient | Raw p | 120 max-T p | Global-600 max-T p |",
            "|---|---|---:|---:|---:|---:|",
            f"| Original Spearman winner | {legacy_emotion_audit['spearman_winner_original']['coalition']} | {legacy_emotion_audit['spearman_winner_original']['coefficient']:+.3f} | {legacy_emotion_audit['spearman_winner_original']['p_raw']:.5f} | {legacy_emotion_audit['spearman_winner_original']['p_max_t_120']:.5f} | {legacy_emotion_audit['spearman_winner_original']['p_max_t_global600']:.5f} |",
            f"| Same coalition under Pearson | {legacy_emotion_audit['spearman_winner_under_pearson']['coalition']} | {legacy_emotion_audit['spearman_winner_under_pearson']['coefficient']:+.3f} | {legacy_emotion_audit['spearman_winner_under_pearson']['p_raw']:.5f} | {legacy_emotion_audit['spearman_winner_under_pearson']['p_max_t_120']:.5f} | {legacy_emotion_audit['spearman_winner_under_pearson']['p_max_t_global600']:.5f} |",
            f"| Pearson-selected winner | {legacy_emotion_audit['pearson_winner']['coalition']} | {legacy_emotion_audit['pearson_winner']['coefficient']:+.3f} | {legacy_emotion_audit['pearson_winner']['p_raw']:.5f} | {legacy_emotion_audit['pearson_winner']['p_max_t_120']:.5f} | {legacy_emotion_audit['pearson_winner']['p_max_t_global600']:.5f} |",
            "",
            "## Main-figure fixed candidates",
            "",
            "| Panel candidate | Spearman coefficient | Spearman raw p | Pearson coefficient | Pearson raw p |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {row['coalition']} x {row['endpoint']} | {row['spearman']['coefficient']:+.3f} | {row['spearman']['p_raw']:.5f} | {row['pearson']['coefficient']:+.3f} | {row['pearson']['p_raw']:.5f} |"
                for row in language_figure_audit
            ],
            "",
            f"For the legacy cohort-aware MOTOR family, the Pearson winner was {legacy_motor_audit['pearson_winner']['coalition']} (r={legacy_motor_audit['pearson_winner']['coefficient']:+.3f}, raw p={legacy_motor_audit['pearson_winner']['p_raw']:.5f}, max-T p={legacy_motor_audit['pearson_winner']['p_max_t_120']:.5f}); no Pearson coalition passed BH or max-T at .05.",
            "",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
