#!/usr/bin/env python3
"""Relate task-matched system-level Xi to one frozen behavior endpoint per HCP task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_all_task_behavior_coalitions_57 import (  # noqa: E402
    load_table,
    make_endpoint_contracts,
)


ARRAYS = ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/full/k1_p3_a1/arrays.npz"
OUTPUT = ROOT / "results/hcp_system_xi_task_behavior_57"
TASK_ORDER = ("EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")
TASK_LABELS = {
    "EMOTION": "Emotion",
    "GAMBLING": "Gambling",
    "LANGUAGE": "Language",
    "MOTOR": "Motor",
    "RELATIONAL": "Relational",
    "SOCIAL": "Social",
    "WM": "Working memory",
}
ENDPOINT_LABELS = {
    "EMOTION": "Face speed | Shape speed",
    "GAMBLING": "Delayed-reward valuation",
    "LANGUAGE": "Corrected Story difficulty",
    "MOTOR": "Broad motor score",
    "RELATIONAL": "Overall relational accuracy",
    "SOCIAL": "Corrected social d-prime",
    "WM": "Overall working-memory accuracy",
}
COLORS = {
    "EMOTION": "#5C85A6",
    "GAMBLING": "#8B8E91",
    "LANGUAGE": "#D08A58",
    "MOTOR": "#B25D56",
    "RELATIONAL": "#8B73A8",
    "SOCIAL": "#4D927D",
    "WM": "#5577A8",
}
PERMUTATIONS = 100_000
BOOTSTRAPS = 20_000
SEED = 20260804
XI_TOLERANCE_BITS = 1.0e-10


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 6.8,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array - design @ np.linalg.lstsq(design, array, rcond=None)[0]


def unit_vector(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-12:
        raise ValueError("Cannot normalize a constant residualized variable.")
    return array / norm


def partial_spearman(x: np.ndarray, y: np.ndarray, design: np.ndarray) -> float:
    x_residual = residualize(rankdata(x), design)
    y_residual = residualize(rankdata(y), design)
    return float(unit_vector(x_residual) @ unit_vector(y_residual))


def partial_pearson(x: np.ndarray, y: np.ndarray, design: np.ndarray) -> float:
    x_residual = residualize(x, design)
    y_residual = residualize(y, design)
    return float(unit_vector(x_residual) @ unit_vector(y_residual))


def bh_adjust(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array)
    ranked = array[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(array) / np.arange(1, len(array) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def raw_design(contract: Mapping[str, Any], sample: np.ndarray | None = None) -> np.ndarray:
    age = np.asarray(contract["age"], dtype=float)
    sex = np.asarray(contract["sex"], dtype=float)
    nuisance = np.asarray(contract["nuisance"], dtype=float)
    if sample is not None:
        age = age[sample]
        sex = sex[sample]
        nuisance = nuisance[sample]
    parts = [np.ones(len(age)), age, sex]
    if nuisance.size:
        if nuisance.ndim == 1:
            nuisance = nuisance[:, None]
        parts.extend(nuisance.T)
    return np.column_stack(parts)


def rank_design(contract: Mapping[str, Any], sample: np.ndarray | None = None) -> np.ndarray:
    age = np.asarray(contract["age"], dtype=float)
    sex = np.asarray(contract["sex"], dtype=float)
    nuisance = np.asarray(contract["nuisance"], dtype=float)
    if sample is not None:
        age = age[sample]
        sex = sex[sample]
        nuisance = nuisance[sample]
    parts = [np.ones(len(age)), rankdata(age), sex]
    if nuisance.size:
        if nuisance.ndim == 1:
            nuisance = nuisance[:, None]
        parts.extend(rankdata(nuisance, axis=0).T)
    return np.column_stack(parts)


def bootstrap_interval(
    x: np.ndarray,
    y: np.ndarray,
    contract: Mapping[str, Any],
    repeats: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    estimates = np.full(repeats, np.nan)
    n = len(x)
    for index in range(repeats):
        sample = rng.integers(0, n, size=n)
        estimates[index] = partial_spearman(
            x[sample], y[sample], rank_design(contract, sample)
        )
    return np.nanquantile(estimates, [0.025, 0.5, 0.975]).tolist()


def leave_one_out(
    x: np.ndarray, y: np.ndarray, contract: Mapping[str, Any]
) -> dict[str, float]:
    estimates = []
    n = len(x)
    for removed in range(n):
        keep = np.arange(n) != removed
        estimates.append(partial_spearman(x[keep], y[keep], rank_design(contract, keep)))
    array = np.asarray(estimates)
    return {
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
        "same_sign_fraction": float(np.mean(np.sign(array) == np.sign(np.median(array)))),
    }


def analyze(
    system_xi: Mapping[str, np.ndarray],
    contracts: Mapping[str, Mapping[str, Any]],
    permutations: int,
    bootstraps: int,
) -> list[dict[str, Any]]:
    prepared: dict[str, dict[str, np.ndarray | float]] = {}
    for state in TASK_ORDER:
        x = np.asarray(system_xi[state], dtype=float)
        y = np.asarray(contracts[state]["endpoint"], dtype=float)
        design = rank_design(contracts[state])
        x_unit = unit_vector(residualize(rankdata(x), design))
        y_rank = rankdata(y)
        y_fitted = design @ np.linalg.lstsq(design, y_rank, rcond=None)[0]
        y_residual = y_rank - y_fitted
        prepared[state] = {
            "x_unit": x_unit,
            "y_fitted": y_fitted,
            "y_residual": y_residual,
            "observed": float(x_unit @ unit_vector(y_residual)),
        }

    rng = np.random.default_rng(SEED)
    point_counts = np.zeros(len(TASK_ORDER), dtype=np.int64)
    max_counts = np.zeros(len(TASK_ORDER), dtype=np.int64)
    motor_negative_count = 0
    observed = np.asarray([prepared[state]["observed"] for state in TASK_ORDER], dtype=float)
    chunk_size = 1000
    n = len(next(iter(system_xi.values())))
    for start in range(0, permutations, chunk_size):
        size = min(chunk_size, permutations - start)
        indices = np.argsort(rng.random((size, n)), axis=1)
        null = np.empty((size, len(TASK_ORDER)), dtype=float)
        for task_index, state in enumerate(TASK_ORDER):
            item = prepared[state]
            design = rank_design(contracts[state])
            pseudo = item["y_fitted"][None, :] + item["y_residual"][indices]
            coefficients = np.linalg.lstsq(design, pseudo.T, rcond=None)[0]
            residuals = pseudo - (design @ coefficients).T
            residuals /= np.linalg.norm(residuals, axis=1, keepdims=True)
            null[:, task_index] = residuals @ item["x_unit"]
        absolute = np.abs(null)
        maxima = absolute.max(axis=1)
        point_counts += np.sum(absolute >= np.abs(observed)[None, :], axis=0)
        max_counts += np.sum(maxima[:, None] >= np.abs(observed)[None, :], axis=0)
        motor_index = TASK_ORDER.index("MOTOR")
        motor_negative_count += int(np.sum(null[:, motor_index] <= observed[motor_index]))

    denominator = permutations + 1.0
    p_two_sided = (point_counts + 1.0) / denominator
    p_max_t = (max_counts + 1.0) / denominator
    q_bh = bh_adjust(p_two_sided)
    rows = []
    for task_index, state in enumerate(TASK_ORDER):
        x = np.asarray(system_xi[state], dtype=float)
        y = np.asarray(contracts[state]["endpoint"], dtype=float)
        interval = bootstrap_interval(
            x, y, contracts[state], bootstraps, SEED + 100 + task_index
        )
        row = {
            "task": state,
            "task_label": TASK_LABELS[state],
            "endpoint_label": contracts[state]["label"],
            "endpoint_definition": contracts[state]["definition"],
            "n_subjects": int(len(x)),
            "partial_spearman_rho": float(observed[task_index]),
            "permutation_p_two_sided": float(p_two_sided[task_index]),
            "bh_q_across_7": float(q_bh[task_index]),
            "max_t_p_across_7": float(p_max_t[task_index]),
            "bootstrap_95_ci": [float(interval[0]), float(interval[2])],
            "bootstrap_median": float(interval[1]),
            "partial_pearson_sensitivity": partial_pearson(
                x, y, raw_design(contracts[state])
            ),
            "leave_one_out": leave_one_out(x, y, contracts[state]),
            "significant_positive_bh_0_05": bool(q_bh[task_index] < 0.05 and observed[task_index] > 0),
            "significant_negative_bh_0_05": bool(q_bh[task_index] < 0.05 and observed[task_index] < 0),
        }
        if state == "MOTOR":
            row["user_specified_negative_one_sided_p"] = float(
                (motor_negative_count + 1.0) / denominator
            )
        rows.append(row)
    return rows


def plot_results(
    rows: list[dict[str, Any]],
    system_xi: Mapping[str, np.ndarray],
    contracts: Mapping[str, Mapping[str, Any]],
) -> None:
    configure_style()
    figure, axes = plt.subplots(2, 4, figsize=(7.2, 4.85), constrained_layout=True)
    summary_axis = axes.flat[0]
    positions = np.arange(len(rows))[::-1]
    estimates = np.asarray([row["partial_spearman_rho"] for row in rows])
    intervals = np.asarray([row["bootstrap_95_ci"] for row in rows])
    errors = np.vstack([estimates - intervals[:, 0], intervals[:, 1] - estimates])
    summary_axis.axvline(0, color="#B8BEC4", linewidth=0.75, zorder=0)
    for index, row in enumerate(rows):
        summary_axis.errorbar(
            estimates[index],
            positions[index],
            xerr=errors[:, index : index + 1],
            fmt="o",
            markersize=4.0,
            color=COLORS[row["task"]],
            ecolor=COLORS[row["task"]],
            elinewidth=1.0,
            capsize=2.0,
        )
    summary_axis.set_yticks(positions, [row["task_label"] for row in rows])
    summary_axis.set_xlim(-0.55, 0.55)
    summary_axis.set_xlabel(r"Partial Spearman $\rho$ (95% CI)")
    summary_axis.set_title("a  Seven fixed system-level tests", loc="left", fontweight="bold")
    panel_letters = "bcdefgh"
    for panel_index, (axis, row) in enumerate(zip(axes.flat[1:], rows)):
        state = row["task"]
        x = np.asarray(contracts[state]["endpoint"], dtype=float)
        y = np.asarray(system_xi[state], dtype=float)
        design = rank_design(contracts[state])
        x_residual = residualize(rankdata(x), design)
        y_residual = residualize(rankdata(y), design)
        color = COLORS[state]
        axis.scatter(
            x_residual,
            y_residual,
            s=16,
            color=color,
            alpha=0.80,
            edgecolor="white",
            linewidth=0.3,
        )
        order = np.argsort(x_residual)
        coefficient = np.polyfit(x_residual, y_residual, 1)
        axis.plot(
            x_residual[order],
            np.polyval(coefficient, x_residual[order]),
            color=color,
            linewidth=0.9,
        )
        axis.axhline(0, color="#D7DBDF", linewidth=0.5, zorder=0)
        axis.axvline(0, color="#D7DBDF", linewidth=0.5, zorder=0)
        axis.set_title(
            f"{panel_letters[panel_index]}  {row['task_label']}",
            loc="left",
            fontweight="bold",
        )
        axis.text(
            0.03,
            0.97,
            rf"$\rho$={row['partial_spearman_rho']:+.3f}" + "\n" + rf"$q_{{BH}}$={row['bh_q_across_7']:.3f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.1,
        )
        axis.set_xlabel(ENDPOINT_LABELS[state] + "\n(rank residual)")
        axis.set_ylabel(r"System $\Xi$ (rank residual)")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "hcp_system_xi_task_behavior_57"
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_outputs(
    rows: list[dict[str, Any]],
    system_xi: Mapping[str, np.ndarray],
    permutations: int,
    bootstraps: int,
) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    motor = next(row for row in rows if row["task"] == "MOTOR")
    checked_xi = np.concatenate([np.asarray(system_xi[state]) for state in TASK_ORDER])
    summary = {
        "experiment": "Task-matched system-level Xi versus behavior in 57 HCP subjects",
        "subjects": 57,
        "primary_inference": "partial Spearman correlation controlling age rank and sex; EMOTION additionally controls Shape speed; two-sided Freedman-Lane permutation; BH FDR and max-T across seven tasks",
        "permutations": permutations,
        "bootstraps": bootstraps,
        "system_xi_estimator": "Schaefer-1000/Yeo7 network PC1, order-3 Ridge alpha=1, affine Gaussian TM",
        "nonnegativity_audit": {
            "tolerance_bits": XI_TOLERANCE_BITS,
            "checked_count": int(checked_xi.size),
            "minimum_bits": float(checked_xi.min()),
            "numerical_zero_count": int(np.sum((checked_xi < 0) & (checked_xi >= -XI_TOLERANCE_BITS))),
            "significant_violation_count": int(np.sum(checked_xi < -XI_TOLERANCE_BITS)),
        },
        "significant_positive_bh_0_05_count": int(sum(row["significant_positive_bh_0_05"] for row in rows)),
        "significant_negative_bh_0_05_count": int(sum(row["significant_negative_bh_0_05"] for row in rows)),
        "motor_directional_hypothesis": {
            "hypothesis": "system-level Xi is negatively associated with the broad motor score",
            "rho": motor["partial_spearman_rho"],
            "one_sided_permutation_p": motor["user_specified_negative_one_sided_p"],
            "status": "directionally consistent but not significant at 0.05",
        },
        "tasks": rows,
    }
    contract = {
        "scientific_question": "Within each of seven HCP task states, is subject-level system Xi associated with its single frozen behavior endpoint?",
        "family_size": 7,
        "frozen_endpoints": {row["task"]: row["endpoint_definition"] for row in rows},
        "covariates": "age rank and sex for all tasks; Shape speed additionally for EMOTION",
        "primary_statistic": "partial Spearman rho",
        "primary_test": "two-sided Freedman-Lane permutation",
        "multiplicity": "BH FDR and permutation max-T across seven tasks",
        "directional_supplement": "user-specified MOTOR negative one-sided permutation test",
        "sensitivity": "partial Pearson correlation on unranked variables",
        "figure_contract": {
            "core_conclusion": "Determine whether any fixed task-level system Xi has a stable positive or negative behavior association.",
            "evidence_chain": "one forest panel for all seven corrected tests plus one adjusted scatter panel per task",
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib",
            "exports": ["PNG 600 dpi", "editable SVG", "PDF"],
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "experiment_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# 七任务 system-level Xi 与行为表现",
        "",
        "主分析在 57 名被试内逐任务检验 system-level $\\Xi$ 与一个冻结行为端点的偏 Spearman 相关。所有任务控制年龄秩和性别，EMOTION 额外控制 Shape 速度；双侧 $p$ 来自 Freedman--Lane 置换，并在七项任务间同时报告 BH FDR 与 max-$T$ 校正。",
        "",
        "| 任务 | 偏 Spearman $\\rho$ | bootstrap 95% CI | 双侧置换 $p$ | 7 项 BH $q$ | 7 项 max-$T$ $p$ | Pearson 敏感性 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        low, high = row["bootstrap_95_ci"]
        lines.append(
            f"| {row['task_label']} | {row['partial_spearman_rho']:+.3f} | [{low:+.3f}, {high:+.3f}] | {row['permutation_p_two_sided']:.4f} | {row['bh_q_across_7']:.4f} | {row['max_t_p_across_7']:.4f} | {row['partial_pearson_sensitivity']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "七项中没有显著正相关或负相关通过 BH 或 max-$T$ 的 0.05 阈值。LANGUAGE 的正相关最大，但仍只是未校正候选。MOTOR 的方向与负相关假设一致；其单侧置换检验也未达到 0.05，因此不能据此声称运动表现越好、全系统整合有效信息越低。",
            "",
            "![Task-matched system Xi correlations](hcp_system_xi_task_behavior_57.png)",
            "",
            "图中散点和回归线基于协变量校正后的秩残差。汇总面板给出偏 Spearman 相关及被试 bootstrap 95% CI。区间用于描述估计不确定性；显著性结论以七任务校正后的置换检验为准。",
            "",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(ARRAYS, allow_pickle=False) as archive:
        states = archive["states"].astype(str).tolist()
        subjects = archive["subjects"].astype(str)
        values = archive["system_xi"].astype(float)
    if subjects.shape != (57,) or values.shape != (8, 57):
        raise ValueError("Expected eight states and the frozen 57-subject sample.")
    task_values = {state: values[states.index(state)] for state in TASK_ORDER}
    checked = np.concatenate(list(task_values.values()))
    violations = checked < -XI_TOLERANCE_BITS
    if np.any(violations):
        raise ValueError(
            "System Xi nonnegativity violation: "
            f"minimum={checked.min():.12g} bits, threshold={-XI_TOLERANCE_BITS:.1e}, "
            f"count={int(violations.sum())}"
        )
    contracts = make_endpoint_contracts(subjects, load_table())
    rows = analyze(task_values, contracts, args.permutations, args.bootstraps)
    write_outputs(rows, task_values, args.permutations, args.bootstraps)
    plot_results(rows, task_values, contracts)
    print(json.dumps({row["task"]: row for row in rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
