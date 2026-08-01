#!/usr/bin/env python3
"""Evaluate held-out delta-Ridge prediction error over the HCP Phi hyperparameter grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hcp_schaefer500_all_tasks_phi import DISPLAY_NAMES, development_end_for_length
from scripts.run_hcp_schaefer500_phi_hyperparameter_robustness import (
    CONDITIONS,
    DEFAULT_ALPHAS,
    DEFAULT_ORDERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REST_ROOT,
    DEFAULT_TASK_ROOT,
    _grid_matrix,
    _save,
    _style,
    default_yeo7_labels,
    prepare_reduced_series,
)


DEFAULT_PHI_SUMMARY = DEFAULT_OUTPUT_DIR / "summary.json"


def _state_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, ddof=1, keepdims=True)
    return mean, np.where(scale > 1.0e-12, scale, 1.0)


def _history_samples(series: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(series, dtype=float)
    history = np.concatenate([values[order - 1 - lag : -1 - lag] for lag in range(order)], axis=1)
    return history, values[order:]


def fit_prediction_metrics(
    series: np.ndarray,
    *,
    order: int,
    alpha: float,
    development_end: int,
) -> dict[str, float | int]:
    """Refit the exact delta-Ridge protocol and evaluate only the held-out final 25%."""
    values = np.asarray(series, dtype=float)
    history, next_state = _history_samples(values, int(order))
    n_state = values.shape[1]
    train_rows = int(development_end) - int(order)
    train_history = history[:train_rows]
    test_history = history[train_rows:]
    train_next = next_state[:train_rows]
    test_next = next_state[train_rows:]

    state_mean, state_scale = _state_scale(train_history[:, :n_state])

    def transform_history(block: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                (block[:, start : start + n_state] - state_mean) / state_scale
                for start in range(0, block.shape[1], n_state)
            ],
            axis=1,
        )

    train_x = transform_history(train_history)
    test_x = transform_history(test_history)
    train_delta = train_next - train_history[:, :n_state]
    test_delta = test_next - test_history[:, :n_state]
    delta_mean, delta_scale = _state_scale(train_delta)
    train_delta_z = (train_delta - delta_mean) / delta_scale
    test_delta_z = (test_delta - delta_mean) / delta_scale

    model = Ridge(alpha=float(alpha), fit_intercept=True).fit(train_x, train_delta_z)
    train_prediction_z = model.predict(train_x)
    test_prediction_z = model.predict(test_x)
    train_error_z = train_prediction_z - train_delta_z
    test_error_z = test_prediction_z - test_delta_z
    train_nrmse = float(np.sqrt(np.mean(np.square(train_error_z))))
    test_nrmse = float(np.sqrt(np.mean(np.square(test_error_z))))

    test_prediction_delta = test_prediction_z * delta_scale + delta_mean
    model_state_error = (test_prediction_delta - test_delta) / state_scale
    persistence_state_error = test_delta / state_scale
    model_mse = float(np.mean(np.square(model_state_error)))
    persistence_mse = float(np.mean(np.square(persistence_state_error)))
    persistence_skill = float(1.0 - model_mse / persistence_mse) if persistence_mse > 0 else float("nan")
    return {
        "n_train_rows": int(len(train_x)),
        "n_test_rows": int(len(test_x)),
        "n_features": int(train_x.shape[1]),
        "train_delta_nrmse": train_nrmse,
        "test_delta_nrmse": test_nrmse,
        "generalization_gap": float(test_nrmse - train_nrmse),
        "test_state_nrmse": float(np.sqrt(model_mse)),
        "persistence_state_nrmse": float(np.sqrt(persistence_mse)),
        "persistence_skill": persistence_skill,
        "coefficient_frobenius_norm": float(np.linalg.norm(model.coef_)),
    }


def summarize_grid(
    rows: Sequence[Mapping[str, Any]],
    orders: Sequence[int],
    alphas: Sequence[float],
) -> list[dict[str, Any]]:
    grid = []
    for order in orders:
        for alpha in alphas:
            selected = [row for row in rows if int(row["order"]) == int(order) and float(row["alpha"]) == float(alpha)]
            condition_summary = {}
            for condition in CONDITIONS:
                condition_rows = [row for row in selected if row["condition"] == condition]
                condition_summary[condition] = {
                    metric: float(np.mean([float(row[metric]) for row in condition_rows]))
                    for metric in (
                        "train_delta_nrmse",
                        "test_delta_nrmse",
                        "generalization_gap",
                        "test_state_nrmse",
                        "persistence_skill",
                        "coefficient_frobenius_norm",
                    )
                }
            task_rows = [row for row in selected if row["condition"] != "REST"]
            grid.append(
                {
                    "order": int(order),
                    "alpha": float(alpha),
                    "overall": {
                        metric: float(np.mean([float(row[metric]) for row in selected]))
                        for metric in (
                            "train_delta_nrmse",
                            "test_delta_nrmse",
                            "generalization_gap",
                            "test_state_nrmse",
                            "persistence_skill",
                            "coefficient_frobenius_norm",
                        )
                    },
                    "task_average": {
                        metric: float(np.mean([float(row[metric]) for row in task_rows]))
                        for metric in (
                            "train_delta_nrmse",
                            "test_delta_nrmse",
                            "generalization_gap",
                            "test_state_nrmse",
                            "persistence_skill",
                            "coefficient_frobenius_norm",
                        )
                    },
                    "condition_summary": condition_summary,
                }
            )
    return grid


def add_phi_diagnostics(grid: list[dict[str, Any]], phi_summary_path: Path) -> dict[str, Any]:
    phi_summary = json.loads(Path(phi_summary_path).read_text(encoding="utf-8"))
    phi_lookup = {(int(item["order"]), float(item["alpha"])): item for item in phi_summary["grid_points"]}
    pair_rows = []
    for point in grid:
        key = (int(point["order"]), float(point["alpha"]))
        phi_point = phi_lookup[key]
        point["minimum_rest_minus_task_phi"] = float(phi_point["minimum_rest_minus_task_mean"])
        point["n_holm_significant_phi_contrasts"] = int(phi_point["n_holm_significant_rest_task_contrasts"])
        for contrast in phi_point["rest_task_contrasts"]:
            condition = str(contrast["condition"])
            pair_rows.append(
                {
                    "order": key[0],
                    "alpha": key[1],
                    "condition": condition,
                    "rest_minus_task_phi": float(contrast["rest_minus_task_mean"]),
                    "task_minus_rest_test_nrmse": float(
                        point["condition_summary"][condition]["test_delta_nrmse"]
                        - point["condition_summary"]["REST"]["test_delta_nrmse"]
                    ),
                    "task_minus_rest_generalization_gap": float(
                        point["condition_summary"][condition]["generalization_gap"]
                        - point["condition_summary"]["REST"]["generalization_gap"]
                    ),
                }
            )

    overall_error = np.asarray([point["overall"]["test_delta_nrmse"] for point in grid])
    overall_gap = np.asarray([point["overall"]["generalization_gap"] for point in grid])
    minimum_margin = np.asarray([point["minimum_rest_minus_task_phi"] for point in grid])
    pair_phi = np.asarray([row["rest_minus_task_phi"] for row in pair_rows])
    pair_error_delta = np.asarray([row["task_minus_rest_test_nrmse"] for row in pair_rows])
    pair_gap_delta = np.asarray([row["task_minus_rest_generalization_gap"] for row in pair_rows])

    def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        result = spearmanr(x, y)
        return {"spearman_rho": float(result.statistic), "p_value": float(result.pvalue)}

    within_order = {}
    for order in sorted({int(row["order"]) for row in pair_rows}):
        selected = [row for row in pair_rows if int(row["order"]) == order]
        order_phi = np.asarray([row["rest_minus_task_phi"] for row in selected])
        order_error = np.asarray([row["task_minus_rest_test_nrmse"] for row in selected])
        order_gap = np.asarray([row["task_minus_rest_generalization_gap"] for row in selected])
        within_order[str(order)] = {
            "phi_margin_vs_task_minus_rest_test_nrmse": correlation(order_phi, order_error),
            "phi_margin_vs_task_minus_rest_generalization_gap": correlation(order_phi, order_gap),
        }

    return {
        "grid_level_overall_test_nrmse_vs_minimum_phi_margin": correlation(overall_error, minimum_margin),
        "grid_level_overall_generalization_gap_vs_minimum_phi_margin": correlation(overall_gap, minimum_margin),
        "all_grid_task_pairs_phi_margin_vs_task_minus_rest_test_nrmse": correlation(pair_phi, pair_error_delta),
        "all_grid_task_pairs_phi_margin_vs_task_minus_rest_generalization_gap": correlation(pair_phi, pair_gap_delta),
        "within_order_correlations": within_order,
        "pair_rows": pair_rows,
    }


def aggregate_summary(
    grid: Sequence[Mapping[str, Any]], orders: Sequence[int], alphas: Sequence[float]
) -> dict[str, Any]:
    global_best = min(grid, key=lambda point: float(point["overall"]["test_delta_nrmse"]))
    best_overall_by_order = []
    best_by_condition = {condition: [] for condition in CONDITIONS}
    for order in orders:
        points = [point for point in grid if int(point["order"]) == int(order)]
        best = min(points, key=lambda point: float(point["overall"]["test_delta_nrmse"]))
        best_overall_by_order.append(
            {
                "order": int(order),
                "best_alpha": float(best["alpha"]),
                "test_delta_nrmse": float(best["overall"]["test_delta_nrmse"]),
            }
        )
        for condition in CONDITIONS:
            condition_best = min(
                points, key=lambda point: float(point["condition_summary"][condition]["test_delta_nrmse"])
            )
            best_by_condition[condition].append(
                {
                    "order": int(order),
                    "best_alpha": float(condition_best["alpha"]),
                    "test_delta_nrmse": float(condition_best["condition_summary"][condition]["test_delta_nrmse"]),
                }
            )
    alpha10_wins = sum(item["best_alpha"] == 10.0 for values in best_by_condition.values() for item in values)
    total_condition_orders = len(CONDITIONS) * len(orders)
    return {
        "pooled_global_best": {
            "order": int(global_best["order"]),
            "alpha": float(global_best["alpha"]),
            "test_delta_nrmse": float(global_best["overall"]["test_delta_nrmse"]),
            "minimum_rest_minus_task_phi": float(global_best["minimum_rest_minus_task_phi"]),
            "n_holm_significant_phi_contrasts": int(global_best["n_holm_significant_phi_contrasts"]),
        },
        "best_overall_alpha_by_order": best_overall_by_order,
        "best_alpha_by_condition_and_order": best_by_condition,
        "n_condition_order_cells_best_at_alpha_10": int(alpha10_wins),
        "n_condition_order_cells": int(total_condition_orders),
        "alpha_10_best_fraction": float(alpha10_wins / total_condition_orders),
    }


def plot_overview(
    grid: Sequence[Mapping[str, Any]], orders: Sequence[int], alphas: Sequence[float], destination: Path
) -> None:
    _style()
    test_error = _grid_matrix(grid, orders, alphas, lambda point: point["overall"]["test_delta_nrmse"])
    gap = _grid_matrix(grid, orders, alphas, lambda point: point["overall"]["generalization_gap"])
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), gridspec_kw={"width_ratios": (1.0, 1.0, 1.35)}, constrained_layout=True)

    image = axes[0].imshow(test_error, cmap="magma_r")
    axes[0].set(
        xticks=np.arange(len(alphas)), xticklabels=[f"{alpha:g}" for alpha in alphas],
        yticks=np.arange(len(orders)), yticklabels=[str(order) for order in orders],
        xlabel=r"Ridge $\alpha$", ylabel=r"History order $p$",
    )
    for row in range(len(orders)):
        for column in range(len(alphas)):
            color = "white" if test_error[row, column] > np.quantile(test_error, 0.75) else "black"
            axes[0].text(column, row, f"{test_error[row, column]:.3f}", ha="center", va="center", fontsize=5.8, color=color)
    colorbar = fig.colorbar(image, ax=axes[0], shrink=0.82, pad=0.03)
    colorbar.set_label("Held-out delta-NRMSE (lower is better)")

    limit = max(abs(float(gap.min())), abs(float(gap.max())), 0.01)
    image = axes[1].imshow(gap, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
    axes[1].set(
        xticks=np.arange(len(alphas)), xticklabels=[f"{alpha:g}" for alpha in alphas],
        yticks=np.arange(len(orders)), yticklabels=[str(order) for order in orders],
        xlabel=r"Ridge $\alpha$", ylabel=r"History order $p$",
    )
    for row in range(len(orders)):
        for column in range(len(alphas)):
            color = "white" if abs(gap[row, column]) > 0.72 * limit else "black"
            axes[1].text(column, row, f"{gap[row, column]:.3f}", ha="center", va="center", fontsize=5.8, color=color)
    colorbar = fig.colorbar(image, ax=axes[1], shrink=0.82, pad=0.03)
    colorbar.set_label("Test − train delta-NRMSE")

    colors = mpl.colormaps["tab10"](np.linspace(0.0, 0.75, len(CONDITIONS)))
    p8 = [point for point in grid if int(point["order"]) == 8]
    for condition, color in zip(CONDITIONS, colors):
        values = [point["condition_summary"][condition]["test_delta_nrmse"] for point in p8]
        axes[2].plot(alphas, values, marker="o", markersize=3.2, linewidth=1.0, color=color, label=DISPLAY_NAMES[condition])
    axes[2].set_xscale("log")
    axes[2].set(xlabel=r"Ridge $\alpha$", ylabel="Held-out delta-NRMSE", xticks=alphas)
    axes[2].get_xaxis().set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    axes[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title=r"$p=8$")
    for label, axis in zip("abc", axes):
        axis.text(-0.16, 1.05, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    _save(fig, destination)


def plot_condition_heatmaps(
    grid: Sequence[Mapping[str, Any]], orders: Sequence[int], alphas: Sequence[float], destination: Path
) -> None:
    _style()
    matrices = {
        condition: _grid_matrix(
            grid,
            orders,
            alphas,
            lambda point, condition=condition: point["condition_summary"][condition]["test_delta_nrmse"],
        )
        for condition in CONDITIONS
    }
    lower = min(float(matrix.min()) for matrix in matrices.values())
    upper = max(float(matrix.max()) for matrix in matrices.values())
    fig, axes = plt.subplots(2, 4, figsize=(10.0, 5.2), constrained_layout=True)
    image = None
    for axis, condition in zip(axes.flat, CONDITIONS):
        matrix = matrices[condition]
        image = axis.imshow(matrix, cmap="magma_r", vmin=lower, vmax=upper)
        axis.set(
            xticks=np.arange(len(alphas)), xticklabels=[f"{alpha:g}" for alpha in alphas],
            yticks=np.arange(len(orders)), yticklabels=[str(order) for order in orders],
            xlabel=r"Ridge $\alpha$", ylabel=r"History order $p$", title=DISPLAY_NAMES[condition],
        )
        for row in range(len(orders)):
            for column in range(len(alphas)):
                color = "white" if matrix[row, column] > lower + 0.72 * (upper - lower) else "black"
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=5.5, color=color)
    assert image is not None
    colorbar = fig.colorbar(image, ax=list(axes.flat), shrink=0.82, pad=0.02)
    colorbar.set_label("Held-out delta-NRMSE (lower is better)")
    _save(fig, destination)


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    n_subjects = int(summary["config"]["n_subjects"])
    aggregate = summary["aggregate"]
    p8 = [point for point in summary["grid_points"] if int(point["order"]) == 8]
    p8_weak = next(point for point in p8 if float(point["alpha"]) == 0.1)
    p8_mid = next(point for point in p8 if float(point["alpha"]) == 10.0)
    p8_strong = next(point for point in p8 if float(point["alpha"]) == 100.0)
    emotion_weak = p8_weak["condition_summary"]["EMOTION"]
    relational_weak = p8_weak["condition_summary"]["RELATIONAL"]
    p8_correlation = summary["phi_error_diagnostics"]["within_order_correlations"]["8"]
    lines = [
        "# REST–七任务预测误差超参数诊断",
        "",
        f"使用与 raw Phi 扫描完全相同的 {n_subjects} 名共同被试、Yeo7-PC1 表征和前 75%/后 25% 时间切分。每个状态和网格点独立重拟合 Δ-Ridge。主指标是按训练段 delta 标准差归一化的留出 delta-NRMSE；同时检查测试减训练误差与持久性基线技能。",
        "",
        "## 各 p 的整体最优 alpha",
        "",
        "| p | 最优 alpha | 留出 delta-NRMSE |",
        "|---:|---:|---:|",
    ]
    for item in aggregate["best_overall_alpha_by_order"]:
        lines.append(f"| {item['order']} | {item['best_alpha']:g} | {item['test_delta_nrmse']:.6f} |")
    lines.extend(
        [
            "",
            f"在 {aggregate['n_condition_order_cells']} 个独立的状态–阶数组合中，alpha=10 是留出误差最小值的 {aggregate['n_condition_order_cells_best_at_alpha_10']}/{aggregate['n_condition_order_cells']}。",
            f"若对八种状态与{n_subjects}名被试等权汇总，并以最低留出误差选择一组共享超参数，全网格最优点为 `p={aggregate['pooled_global_best']['order']}, alpha={aggregate['pooled_global_best']['alpha']:g}`（delta-NRMSE={aggregate['pooled_global_best']['test_delta_nrmse']:.6f}）；该点的最小 REST-minus-task Phi 边际为 {aggregate['pooled_global_best']['minimum_rest_minus_task_phi']:.6f} bits，七项 Phi 对比中 {aggregate['pooled_global_best']['n_holm_significant_phi_contrasts']}/7 经 Holm 校正显著。",
            "",
            "## p=8 的过拟合诊断",
            "",
            "| alpha | 训练 delta-NRMSE | 留出 delta-NRMSE | 泛化间隙 | 持久性基线技能 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for point in p8:
        item = point["overall"]
        lines.append(
            f"| {point['alpha']:g} | {item['train_delta_nrmse']:.6f} | {item['test_delta_nrmse']:.6f} | "
            f"{item['generalization_gap']:.6f} | {item['persistence_skill']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"`p=8, alpha=0.1` 的留出误差为 {p8_weak['overall']['test_delta_nrmse']:.6f}，高于 `alpha=10` 的 {p8_mid['overall']['test_delta_nrmse']:.6f} 与 `alpha=100` 的 {p8_strong['overall']['test_delta_nrmse']:.6f}；其泛化间隙为 {p8_weak['overall']['generalization_gap']:.6f}，持久性基线技能为 {p8_weak['overall']['persistence_skill']:.6f}。发生显著 Phi 反转的 EMOTION/RELATIONAL 在该点的留出误差分别为 {emotion_weak['test_delta_nrmse']:.6f}/{relational_weak['test_delta_nrmse']:.6f}，泛化间隙为 {emotion_weak['generalization_gap']:.6f}/{relational_weak['generalization_gap']:.6f}，持久性技能为 {emotion_weak['persistence_skill']:.6f}/{relational_weak['persistence_skill']:.6f}。",
            "",
            f"在固定 `p=8` 的 35 个网格-任务对比中，REST-minus-task Phi 边际与 task-minus-REST 留出误差差的 Spearman rho={p8_correlation['phi_margin_vs_task_minus_rest_test_nrmse']['spearman_rho']:.6f}（p={p8_correlation['phi_margin_vs_task_minus_rest_test_nrmse']['p_value']:.6g}），与泛化间隙差的 rho={p8_correlation['phi_margin_vs_task_minus_rest_generalization_gap']['spearman_rho']:.6f}（p={p8_correlation['phi_margin_vs_task_minus_rest_generalization_gap']['p_value']:.6g}）。这是与过拟合解释一致的描述性关联，不是因果证明。",
            "",
            "该诊断回答预测泛化，不把最低预测误差直接等同于最可靠 Phi。Phi 还依赖拟合系数和训练残差协方差；弱正则可能通过压低训练残差而放大 Phi，即使留出预测变差。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    task_root: Path,
    rest_root: Path,
    labels_path: Path,
    output_dir: Path,
    phi_summary_path: Path,
    *,
    orders: Sequence[int] = DEFAULT_ORDERS,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    task_data_key: str = "Schaefer500_taskRetained",
    rest_data_key: str = "Schaefer500",
    parcel_count: int = 500,
    max_subjects: int | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    subjects, reduced, lengths = prepare_reduced_series(
        task_root,
        rest_root,
        labels_path,
        output_dir / "reduced_series_cache.npz",
        task_data_key=task_data_key,
        rest_data_key=rest_data_key,
        parcel_count=parcel_count,
        max_subjects=max_subjects,
    )
    checkpoint_path = output_dir / "prediction_checkpoint.json"
    rows: list[dict[str, Any]] = []
    completed: set[tuple[int, float]] = set()
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        rows = list(checkpoint.get("rows", []))
        completed = {(int(p), float(a)) for p, a in checkpoint.get("completed_grid_points", [])}
    total = len(subjects) * len(CONDITIONS) * len(orders) * len(alphas)
    finished = len(completed) * len(subjects) * len(CONDITIONS)
    progress = tqdm(total=total, initial=finished, desc="prediction grid", unit="fit", mininterval=1.0)
    try:
        for order in orders:
            for alpha in alphas:
                if (int(order), float(alpha)) in completed:
                    continue
                for subject in subjects:
                    for condition in CONDITIONS:
                        series = reduced[(subject, condition)]
                        metrics = fit_prediction_metrics(
                            series,
                            order=int(order),
                            alpha=float(alpha),
                            development_end=development_end_for_length(len(series)),
                        )
                        rows.append(
                            {"subject": subject, "condition": condition, "order": int(order), "alpha": float(alpha), **metrics}
                        )
                        progress.update(1)
                completed.add((int(order), float(alpha)))
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text(
                    json.dumps(
                        {"completed_grid_points": sorted(completed), "rows": rows}, indent=2
                    ),
                    encoding="utf-8",
                )
    finally:
        progress.close()
    grid = summarize_grid(rows, orders, alphas)
    phi_diagnostics = add_phi_diagnostics(grid, phi_summary_path)
    summary = {
        "config": {
            "task_data_key": task_data_key,
            "rest_data_key": rest_data_key,
            "parcel_count": int(parcel_count),
            "subjects": list(subjects),
            "n_subjects": len(subjects),
            "conditions": list(CONDITIONS),
            "orders": [int(value) for value in orders],
            "alphas": [float(value) for value in alphas],
            "timepoints": lengths,
            "development_fraction": 0.75,
            "primary_metric": "held-out delta-NRMSE using training delta scale",
            "baseline": "persistence prediction x_(t+1)=x_t",
        },
        "aggregate": aggregate_summary(grid, orders, alphas),
        "phi_error_diagnostics": phi_diagnostics,
        "grid_points": grid,
        "rows": rows,
    }
    (output_dir / "prediction_error_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, output_dir / "prediction_error_report.md")
    plot_overview(grid, orders, alphas, output_dir / "prediction_error_overview")
    plot_condition_heatmaps(grid, orders, alphas, output_dir / "prediction_error_by_condition")
    return summary


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--rest-root", type=Path, default=DEFAULT_REST_ROOT)
    parser.add_argument("--labels", type=Path, default=default_yeo7_labels(500))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--phi-summary", type=Path, default=DEFAULT_PHI_SUMMARY)
    parser.add_argument("--orders", default=",".join(map(str, DEFAULT_ORDERS)))
    parser.add_argument("--alphas", default=",".join(map(str, DEFAULT_ALPHAS)))
    parser.add_argument("--task-data-key", default="Schaefer500_taskRetained")
    parser.add_argument("--rest-data-key", default="Schaefer500")
    parser.add_argument("--parcel-count", type=int, choices=(500, 1000), default=500)
    parser.add_argument("--max-subjects", type=int)
    args = parser.parse_args(argv)
    summary = run(
        args.task_root,
        args.rest_root,
        args.labels,
        args.output_dir,
        args.phi_summary,
        orders=_parse_ints(args.orders),
        alphas=_parse_floats(args.alphas),
        task_data_key=args.task_data_key,
        rest_data_key=args.rest_data_key,
        parcel_count=args.parcel_count,
        max_subjects=args.max_subjects,
    )
    diagnostics = {key: value for key, value in summary["phi_error_diagnostics"].items() if key != "pair_rows"}
    print(json.dumps({"aggregate": summary["aggregate"], "correlations": diagnostics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
