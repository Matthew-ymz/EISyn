#!/usr/bin/env python3
"""Test raw-value partial Pearson associations for seven task-level system Xi values.

This is the one-factor counterpart to ``analyze_hcp_system_xi_task_behavior_57.py``:
subjects, task-level Xi estimates, endpoints, nuisance variables, and the seven-test
family are frozen; only rank-space partial Spearman is replaced by raw-value
partial Pearson.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_hcp_system_xi_task_behavior_57 as spearman_analysis


OUTPUT = ROOT / "results/hcp_system_xi_task_behavior_pearson_57"
SEED = spearman_analysis.SEED


def pearson_components(
    x: np.ndarray,
    y: np.ndarray,
    contract: Mapping[str, Any],
    sample: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample is not None:
        x = x[sample]
        y = y[sample]
    design = spearman_analysis.raw_design(contract, sample)
    x_residual = spearman_analysis.residualize(x, design)
    y_residual = spearman_analysis.residualize(y, design)
    return x_residual, y_residual, design


def coefficient(
    x: np.ndarray,
    y: np.ndarray,
    contract: Mapping[str, Any],
    sample: np.ndarray | None = None,
) -> float:
    x_residual, y_residual, _ = pearson_components(x, y, contract, sample)
    return float(
        spearman_analysis.unit_vector(x_residual)
        @ spearman_analysis.unit_vector(y_residual)
    )


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
        try:
            estimates[index] = coefficient(x, y, contract, sample)
        except ValueError:
            continue
    return np.nanquantile(estimates, [0.025, 0.5, 0.975]).tolist()


def leave_one_out(
    x: np.ndarray,
    y: np.ndarray,
    contract: Mapping[str, Any],
) -> dict[str, float]:
    estimates = []
    for removed in range(len(x)):
        keep = np.arange(len(x)) != removed
        estimates.append(coefficient(x, y, contract, keep))
    values = np.asarray(estimates)
    median = float(np.median(values))
    return {
        "minimum": float(values.min()),
        "median": median,
        "maximum": float(values.max()),
        "same_sign_fraction": float(np.mean(np.sign(values) == np.sign(median))),
    }


def analyze(
    system_xi: Mapping[str, np.ndarray],
    contracts: Mapping[str, Mapping[str, Any]],
    permutations: int,
    bootstraps: int,
) -> list[dict[str, Any]]:
    prepared: dict[str, dict[str, np.ndarray | float]] = {}
    observed = []
    for state in spearman_analysis.TASK_ORDER:
        x = np.asarray(system_xi[state], dtype=float)
        y = np.asarray(contracts[state]["endpoint"], dtype=float)
        design = spearman_analysis.raw_design(contracts[state])
        x_unit = spearman_analysis.unit_vector(
            spearman_analysis.residualize(x, design)
        )
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - fitted
        value = float(x_unit @ spearman_analysis.unit_vector(residual))
        observed.append(value)
        prepared[state] = {
            "x_unit": x_unit,
            "fitted": fitted,
            "residual": residual,
        }

    observed_array = np.asarray(observed)
    point_counts = np.zeros(len(observed_array), dtype=np.int64)
    max_counts = np.zeros(len(observed_array), dtype=np.int64)
    rng = np.random.default_rng(SEED)
    n = len(next(iter(system_xi.values())))
    for start in range(0, permutations, 1_000):
        size = min(1_000, permutations - start)
        indices = np.argsort(rng.random((size, n)), axis=1)
        null = np.empty((size, len(observed_array)), dtype=float)
        for task_index, state in enumerate(spearman_analysis.TASK_ORDER):
            item = prepared[state]
            design = spearman_analysis.raw_design(contracts[state])
            pseudo = item["fitted"][None, :] + item["residual"][indices]
            coefficients = np.linalg.lstsq(design, pseudo.T, rcond=None)[0]
            residuals = pseudo - (design @ coefficients).T
            residuals /= np.linalg.norm(residuals, axis=1, keepdims=True)
            null[:, task_index] = residuals @ item["x_unit"]
        absolute = np.abs(null)
        maxima = absolute.max(axis=1)
        point_counts += np.sum(
            absolute >= np.abs(observed_array)[None, :], axis=0
        )
        max_counts += np.sum(
            maxima[:, None] >= np.abs(observed_array)[None, :], axis=0
        )

    denominator = permutations + 1.0
    p_raw = (point_counts + 1.0) / denominator
    p_max_t = (max_counts + 1.0) / denominator
    q_bh = spearman_analysis.bh_adjust(p_raw)
    rows = []
    for task_index, state in enumerate(spearman_analysis.TASK_ORDER):
        x = np.asarray(system_xi[state], dtype=float)
        y = np.asarray(contracts[state]["endpoint"], dtype=float)
        interval = bootstrap_interval(
            x, y, contracts[state], bootstraps, SEED + 100 + task_index
        )
        rows.append(
            {
                "task": state,
                "task_label": spearman_analysis.TASK_LABELS[state],
                "endpoint_label": contracts[state]["label"],
                "partial_pearson_r": float(observed_array[task_index]),
                "bootstrap_95_ci": [float(interval[0]), float(interval[2])],
                "bootstrap_median": float(interval[1]),
                "permutation_p_two_sided": float(p_raw[task_index]),
                "bh_q_across_7": float(q_bh[task_index]),
                "max_t_p_across_7": float(p_max_t[task_index]),
                "leave_one_out": leave_one_out(x, y, contracts[state]),
            }
        )
    return rows


def write_outputs(
    rows: list[dict[str, Any]],
    system_xi: Mapping[str, np.ndarray],
    permutations: int,
    bootstraps: int,
) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    old_summary = json.loads(
        (spearman_analysis.OUTPUT / "summary.json").read_text(encoding="utf-8")
    )
    old_rows = {row["task"]: row for row in old_summary["tasks"]}
    checked_xi = np.concatenate(
        [np.asarray(system_xi[state]) for state in spearman_analysis.TASK_ORDER]
    )
    summary = {
        "experiment": "Raw-value partial Pearson system-level Xi versus behavior",
        "scientific_question": "What changes when only rank-space partial Spearman is replaced by raw-value partial Pearson?",
        "subjects": 57,
        "treatment_factor": "correlation method only",
        "frozen": {
            "system_xi_estimator": "Schaefer-1000/Yeo7 network PC1, order-3 Ridge alpha=1, affine Gaussian TM",
            "endpoints_and_covariates": "identical to the seven-task Spearman analysis",
            "permutations": permutations,
            "bootstraps": bootstraps,
            "multiplicity": "BH and Freedman-Lane max-T across seven fixed tasks",
        },
        "nonnegativity_audit": {
            "tolerance_bits": spearman_analysis.XI_TOLERANCE_BITS,
            "checked_count": int(checked_xi.size),
            "minimum_bits": float(checked_xi.min()),
            "significant_violation_count": int(
                np.sum(checked_xi < -spearman_analysis.XI_TOLERANCE_BITS)
            ),
        },
        "pearson_bh_below_0_05_count": int(
            sum(row["bh_q_across_7"] < 0.05 for row in rows)
        ),
        "pearson_max_t_below_0_05_count": int(
            sum(row["max_t_p_across_7"] < 0.05 for row in rows)
        ),
        "tasks": [
            {
                **row,
                "spearman_reference": {
                    "rho": old_rows[row["task"]]["partial_spearman_rho"],
                    "permutation_p_two_sided": old_rows[row["task"]][
                        "permutation_p_two_sided"
                    ],
                    "bh_q_across_7": old_rows[row["task"]]["bh_q_across_7"],
                    "max_t_p_across_7": old_rows[row["task"]][
                        "max_t_p_across_7"
                    ],
                },
            }
            for row in rows
        ],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# 七任务 system-level Xi：Spearman 与 Pearson 对照",
        "",
        "唯一变化因素为相关方法：原分析使用秩空间偏 Spearman，本分析使用原始实测值偏 Pearson。其余样本、Xi、端点、协变量、置换和七任务校正均冻结。",
        "",
        "| 任务 | Spearman rho | Spearman BH q | Pearson r | Pearson 95% CI | Pearson raw p | Pearson BH q | Pearson max-T p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["tasks"]:
        old = row["spearman_reference"]
        low, high = row["bootstrap_95_ci"]
        lines.append(
            f"| {row['task_label']} | {old['rho']:+.3f} | {old['bh_q_across_7']:.4f} | "
            f"{row['partial_pearson_r']:+.3f} | [{low:+.3f}, {high:+.3f}] | "
            f"{row['permutation_p_two_sided']:.5f} | {row['bh_q_across_7']:.5f} | "
            f"{row['max_t_p_across_7']:.5f} |"
        )
    lines.extend(
        [
            "",
            f"Pearson 下通过七任务 BH 的任务数为 {summary['pearson_bh_below_0_05_count']}，通过七任务 max-T 的任务数为 {summary['pearson_max_t_below_0_05_count']}。",
            "",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--bootstraps", type=int, default=20_000)
    args = parser.parse_args()

    with np.load(spearman_analysis.ARRAYS, allow_pickle=False) as archive:
        states = archive["states"].astype(str).tolist()
        subjects = archive["subjects"].astype(str)
        values = archive["system_xi"].astype(float)
    if subjects.shape != (57,) or values.shape != (8, 57):
        raise ValueError("Expected eight states and the frozen 57-subject sample.")
    system_xi = {
        state: values[states.index(state)] for state in spearman_analysis.TASK_ORDER
    }
    contracts = spearman_analysis.make_endpoint_contracts(
        subjects, spearman_analysis.load_table()
    )
    rows = analyze(system_xi, contracts, args.permutations, args.bootstraps)
    write_outputs(rows, system_xi, args.permutations, args.bootstraps)
    print(json.dumps({row["task"]: row for row in rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
