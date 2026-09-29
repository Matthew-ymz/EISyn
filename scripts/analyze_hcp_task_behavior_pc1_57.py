#!/usr/bin/env python3
"""Behavior-only task PC1 audit and matched fixed-coalition association screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.analyze_hcp_all_task_behavior_coalitions_57 as base
import scripts.screen_hcp_gambling_reward_valuation_57 as gambling
import scripts.screen_hcp_motor_composite_scores_57 as motor
import scripts.screen_hcp_social_composite_scores_57 as social


OUTPUT = base.ROOT / "results/hcp_task_behavior_pc1_57"
SEED = 20260929
TASKS = base.TASK_ORDER


def raw_column(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.asarray(
        [float(row[field]) if row[field] not in ("", "NA") else np.nan for row in rows],
        dtype=float,
    )


def task_inputs(subjects: np.ndarray, table: dict[str, dict[str, str]]) -> dict[str, tuple[list[str], np.ndarray]]:
    rows = [table[str(subject).removeprefix("sub-")] for subject in subjects]
    column = lambda field: raw_column(rows, field)
    social_scores = social.load_scores(subjects)
    motor_scores = motor.load_scores(subjects)
    gambling_scores = gambling.load_scores(subjects)
    return {
        "LANGUAGE": (
            ["Corrected Story difficulty", "Corrected Math difficulty"],
            np.column_stack([
                column("Language_Task_Math_Avg_Difficulty_Level"),
                column("Language_Task_Story_Avg_Difficulty_Level"),
            ]),
        ),
        "SOCIAL": (
            ["TOM hit rate", "Random correct-rejection rate"],
            np.column_stack([social_scores["hit_percent"], social_scores["correct_reject_percent"]]),
        ),
        "EMOTION": (
            ["Face speed (-log RT)", "Shape speed (-log RT)"],
            np.column_stack([
                -np.log(column("Emotion_Task_Face_Median_RT")),
                -np.log(column("Emotion_Task_Shape_Median_RT")),
            ]),
        ),
        "MOTOR": (list(motor.COMPONENT_FIELDS), motor_scores["components_raw"]),
        "GAMBLING": (list(gambling.COMPONENT_FIELDS), gambling_scores["components_raw"]),
        "RELATIONAL": (
            ["Relational accuracy", "Match accuracy"],
            np.column_stack([
                column("Relational_Task_Rel_Acc"),
                column("Relational_Task_Match_Acc"),
            ]),
        ),
        "WM": (
            ["2-back accuracy", "0-back accuracy"],
            np.column_stack([
                column("WM_Task_2bk_Acc"),
                column("WM_Task_0bk_Acc"),
            ]),
        ),
    }


def compute_pc1(values: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    keep = np.isfinite(values).all(axis=1)
    complete = values[keep]
    if len(complete) < 3 or np.any(complete.std(axis=0, ddof=1) <= 0):
        raise ValueError("Insufficient nonconstant, complete behavior scores for PCA.")
    standardized = (complete - complete.mean(axis=0)) / complete.std(axis=0, ddof=1)
    correlation = np.corrcoef(standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    loadings = eigenvectors[:, 0].copy()
    if abs(loadings.sum()) < 1.0e-10:
        sign = np.sign(loadings[0])
    else:
        sign = np.sign(loadings.sum())
    loadings *= sign
    scores = np.full(len(values), np.nan)
    scores[keep] = standardized @ loadings
    explained = float(eigenvalues[0] / eigenvalues.sum())
    off_diagonal = correlation[np.triu_indices(correlation.shape[0], k=1)]
    # This screen is fixed using behavior data only, before inspecting brain associations.
    interpretable = bool(explained >= 0.70 and np.all(loadings > 0) and np.all(off_diagonal >= 0.30))
    return scores, {
        "n_complete": int(keep.sum()),
        "explained_variance_ratio": explained,
        "loadings": loadings.tolist(),
        "item_correlation_matrix": correlation.tolist(),
        "behavior_only_interpretability_gate": interpretable,
    }


def joint_screen(
    brain_values: np.ndarray,
    endpoints: np.ndarray,
    design: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Freedman-Lane, with 120-coalition and 240-comparison max-T control."""
    brain = base.unit_columns(base.residualize(rankdata(brain_values, axis=0), design))
    ranked = rankdata(endpoints, axis=0)
    fitted = design @ np.linalg.lstsq(design, ranked, rcond=None)[0]
    residual = base.residualize(ranked, design)
    observed = brain.T @ base.unit_columns(residual.copy())
    point_counts = np.zeros((120, 2), dtype=np.int64)
    within_counts = np.zeros((120, 2), dtype=np.int64)
    joint_counts = np.zeros((120, 2), dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = len(endpoints)
    for start in range(0, permutations, 1000):
        size = min(1000, permutations - start)
        indices = np.argsort(rng.random((size, n)), axis=1)
        pseudo = fitted[None, :, :] + residual[indices]
        coefficients = np.linalg.lstsq(
            design, pseudo.transpose(1, 0, 2).reshape(n, -1), rcond=None
        )[0]
        null = pseudo - (design @ coefficients).reshape(n, size, 2).transpose(1, 0, 2)
        null /= np.linalg.norm(null, axis=1, keepdims=True)
        absolute = np.abs(np.einsum("nc,bne->bce", brain, null, optimize=True))
        threshold = np.abs(observed)[None, :, :]
        point_counts += np.sum(absolute >= threshold, axis=0)
        within_counts += np.sum(absolute.max(axis=1)[:, None, :] >= threshold, axis=0)
        joint_counts += np.sum(absolute.max(axis=(1, 2))[:, None, None] >= threshold, axis=0)
    denominator = permutations + 1
    return {
        "rho": observed,
        "p_point": (point_counts + 1) / denominator,
        "p_max_t_120": (within_counts + 1) / denominator,
        "p_max_t_240": (joint_counts + 1) / denominator,
    }


def winner(values: dict[str, np.ndarray], names: np.ndarray, endpoint: int) -> dict[str, object]:
    index = int(np.argmax(np.abs(values["rho"][:, endpoint])))
    return {
        "coalition": str(names[index]),
        "adjusted_rho": float(values["rho"][index, endpoint]),
        "p_point": float(values["p_point"][index, endpoint]),
        "p_max_t_120": float(values["p_max_t_120"][index, endpoint]),
        "p_max_t_240": float(values["p_max_t_240"][index, endpoint]),
    }


def write_report(summary: dict[str, object], output: Path) -> None:
    lines = [
        "# HCP 57: behavior PC1 sensitivity analysis",
        "",
        "The sole changed factor is behavior endpoint construction. Each task uses oriented,",
        "non-duplicated component scores. PCA is fit on complete subjects after z-scoring",
        "the components; no brain values are used to determine PC1 or interpretability.",
        "The behavior-only gate requires PC1 variance share >= 70%, all positive loadings,",
        "and all component Pearson correlations >= 0.30. This is an exploratory sensitivity",
        "analysis, not a replacement for frozen task-specific primary endpoints.",
        "",
        "| Task | n | PC1 variance | Component loadings | Gate | Spearman with original endpoint |",
        "| --- | ---: | ---: | --- | --- | ---: |",
    ]
    for task in TASKS:
        item = summary["tasks"][task]
        loadings = ", ".join(
            f"{label} {value:+.2f}"
            for label, value in zip(item["components"], item["pca"]["loadings"])
        )
        lines.append(
            f"| {task} | {item['pca']['n_complete']} | "
            f"{item['pca']['explained_variance_ratio']:.1%} | {loadings} | "
            f"{'pass' if item['pca']['behavior_only_interpretability_gate'] else 'fail'} | "
            f"{item['spearman_pc1_original']:+.3f} |"
        )
    lines += [
        "",
        "## Matched association comparison",
        "",
        "Both endpoints use the same complete subjects, age-rank and sex covariates,",
        "the cached 120-coalition TM Syn matrix, and the same Freedman-Lane permutations.",
        "Within-task max-T covers 120 coalitions; joint max-T covers both endpoints",
        "and all 120 coalitions. EMOTION here uses age/sex only for both endpoints,",
        "so its old-endpoint comparator differs from the original Face|Shape analysis.",
        "EMOTION PC1 measures general speed, not face-specific performance.",
        "WM has one missing 2-back score, so both endpoints use the same 56 people.",
        "",
        "| Task | Endpoint | Winner coalition | Adjusted rho | max-T 120 p | max-T 240 p |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        item = summary["tasks"][task]
        if "associations" not in item:
            continue
        for label in ("original", "pc1"):
            result = item["associations"][label]
            lines.append(
                f"| {task} | {label} | {result['coalition']} | "
                f"{result['adjusted_rho']:+.3f} | {result['p_max_t_120']:.4f} | "
                f"{result['p_max_t_240']:.4f} |"
            )
    lines += [
        "",
        "PC1 scoring and these brain associations were inspected on the same 57-person",
        "cohort. A smaller p value here is exploratory evidence, not an independent",
        "confirmation or a demonstration of predictive improvement.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with np.load(motor.SUBJECT_SOURCE, allow_pickle=False) as archive:
        subjects = archive["subjects"].astype(str)
    if subjects.shape != (57,):
        raise ValueError("Expected the frozen 57-person subject order.")
    table = base.load_table()
    contracts = base.make_endpoint_contracts(subjects, table)
    inputs = task_inputs(subjects, table)
    summary: dict[str, object] = {
        "scientific_question": "What changes when only the task behavior endpoint is replaced by an interpretable PC1?",
        "permutations": args.permutations,
        "syn_nonnegative_tolerance_bits": base.SYN_TOLERANCE_BITS,
        "tasks": {},
    }
    saved_scores: dict[str, np.ndarray] = {"subjects": subjects}
    for index, task in enumerate(TASKS):
        labels, raw = inputs[task]
        scores, audit = compute_pc1(raw)
        keep = np.isfinite(scores)
        old = np.asarray(contracts[task]["endpoint"])[keep]
        task_result: dict[str, object] = {
            "components": labels,
            "pca": audit,
            "spearman_pc1_original": float(spearmanr(scores[keep], old).statistic),
        }
        saved_scores[f"pc1_{task}"] = scores
        if audit["behavior_only_interpretability_gate"]:
            _, names, _, matrix = base.load_matrix(task, subjects)
            age = contracts[task]["age"][keep]
            sex = contracts[task]["sex"][keep]
            design = base.base_design(age, sex)
            endpoints = np.column_stack([old, scores[keep]])
            tested = joint_screen(matrix[keep], endpoints, design, args.permutations, SEED + index)
            task_result["associations"] = {
                "original": winner(tested, names, 0),
                "pc1": winner(tested, names, 1),
                "same_complete_subjects": int(keep.sum()),
            }
        summary["tasks"][task] = task_result
    np.savez_compressed(output / "pc1_scores.npz", **saved_scores)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(summary, output)
    print((output / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
