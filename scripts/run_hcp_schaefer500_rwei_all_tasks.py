#!/usr/bin/env python3
"""Compare length-matched one-step RW-EI across HCP REST and seven tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.TM.transport_map_density import (
    estimate_mutual_information_transport_map,
    fit_polynomial_triangular_transport_map_density,
)
from exp.reweighted_ei.reweighted_ei_experiment import (
    effective_sample_size,
    estimate_product_density_ratio_knn,
)
from scripts.run_hcp_schaefer500_all_tasks_phi import (
    CONDITION_ORDER,
    DISPLAY_NAMES,
    TASKS,
    discover_subject_task_files,
)
from scripts.run_hcp_schaefer500_wm_yeo7_phi import load_task_series
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import (
    DEFAULT_LABELS,
    load_hcp_series,
    load_yeo7_groups,
)


DEFAULT_REST_ROOT = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30"
DEFAULT_TASK_ROOT = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_30"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "hcp_schaefer500_rwei_conditional_tc_tm_weights"


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted p-values in the original order."""

    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted.tolist()


def reduce_yeo7_pc1(
    raw: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    *,
    n_timepoints: int,
) -> np.ndarray:
    """Fit deterministic randomized PC1s on one length-matched condition segment."""

    values = np.asarray(raw, dtype=float)[: int(n_timepoints)]
    if values.ndim != 2 or len(values) != int(n_timepoints):
        raise ValueError(f"Expected at least {n_timepoints} time points, got {values.shape}.")
    scores = []
    for indices in groups.values():
        model = PCA(n_components=1, svd_solver="randomized", random_state=0)
        projected = model.fit_transform(values[:, np.asarray(indices, dtype=int)])[:, 0]
        if float(model.components_[0].sum()) < 0.0:
            projected = -projected
        scores.append(projected)
    reduced = np.column_stack(scores)
    scale = reduced.std(axis=0, ddof=1)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return (reduced - reduced.mean(axis=0, keepdims=True)) / scale


def estimate_product_density_ratio_tm(
    source: np.ndarray,
    *,
    degree: int = 1,
    log_ratio_clip: float = 20.0,
) -> np.ndarray:
    """Estimate product-marginal / observed density ratio with TM densities.

    The numerator is structurally factorized as the product of separately fitted
    univariate TM marginals. The denominator is one multivariate TM density.
    """

    values = np.asarray(source, dtype=float)
    if values.ndim != 2 or len(values) < 4:
        raise ValueError("source must be a two-dimensional array with at least four rows.")
    observed_model = fit_polynomial_triangular_transport_map_density(values, degree=int(degree))
    log_product = np.zeros(len(values), dtype=float)
    for index in range(values.shape[1]):
        marginal = values[:, [index]]
        marginal_model = fit_polynomial_triangular_transport_map_density(
            marginal, degree=int(degree)
        )
        log_product += marginal_model.log_prob(marginal)
    log_ratio = log_product - observed_model.log_prob(values)
    log_ratio = np.clip(log_ratio, -float(log_ratio_clip), float(log_ratio_clip))
    log_ratio -= float(np.max(log_ratio))
    return np.exp(log_ratio)


def _weighted_covariance(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    normalized = np.asarray(weight, dtype=float).reshape(-1)
    normalized = normalized / normalized.sum()
    centered = array - np.sum(normalized[:, None] * array, axis=0)
    correction = 1.0 - float(np.sum(normalized**2))
    if correction <= 0.0:
        raise ValueError("Weights do not support a finite covariance estimate.")
    return (centered.T * normalized) @ centered / correction


def _gaussian_total_correlation(covariance: np.ndarray) -> float:
    matrix = np.asarray(covariance, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    ridge_scale = max(float(np.trace(matrix)) / matrix.shape[0], 1.0)
    matrix = matrix + 1.0e-9 * ridge_scale * np.eye(matrix.shape[0])
    scale = np.sqrt(np.diag(matrix))
    correlation = matrix / np.outer(scale, scale)
    sign, logdet = np.linalg.slogdet(correlation)
    if sign <= 0.0:
        raise ValueError("Covariance must be positive definite.")
    return max(0.0, float(-0.5 * logdet / np.log(2.0)))


def gaussian_source_dependence_diagnostics(
    source: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float]:
    """Decompose affine-TM Xi into conditional TC minus source TC."""

    source_array = np.asarray(source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    joint_covariance = _weighted_covariance(
        np.concatenate([source_array, target_array], axis=1), weight
    )
    width = source_array.shape[1]
    source_covariance = joint_covariance[:width, :width]
    target_covariance = joint_covariance[width:, width:]
    cross_covariance = joint_covariance[:width, width:]
    conditional_covariance = source_covariance - (
        cross_covariance @ np.linalg.pinv(target_covariance) @ cross_covariance.T
    )
    source_tc = _gaussian_total_correlation(source_covariance)
    conditional_tc = _gaussian_total_correlation(conditional_covariance)
    raw_source_tc = _gaussian_total_correlation(np.cov(source_array, rowvar=False, ddof=1))
    return {
        "raw_source_tc_bits": raw_source_tc,
        "weighted_source_tc_bits": source_tc,
        "conditional_source_tc_bits": conditional_tc,
        "tc_difference_bits": conditional_tc - source_tc,
    }


def estimate_rwei_phi(
    reduced: np.ndarray,
    *,
    knn_k: int,
    tm_degree: int,
    weight_seeds: Sequence[int],
    weight_estimator: str = "tm",
    ratio_tm_degree: int = 1,
    xi_estimator: str = "conditional_tc",
) -> dict[str, Any]:
    """Estimate one-step Xi directly as conditional total correlation."""

    values = np.asarray(reduced, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or len(values) < int(knn_k) + 2:
        raise ValueError("reduced must be [time, variable] with enough rows for kNN.")
    source, target = values[:-1], values[1:]
    estimator = str(weight_estimator).lower()
    if estimator not in {"knn", "tm"}:
        raise ValueError("weight_estimator must be 'knn' or 'tm'.")
    xi_method = str(xi_estimator).lower()
    if xi_method not in {"conditional_tc", "mi_difference"}:
        raise ValueError("xi_estimator must be 'conditional_tc' or 'mi_difference'.")
    seeds: Sequence[int | None] = weight_seeds if estimator == "knn" else (None,)
    replicate_rows = []
    for seed in seeds:
        if estimator == "knn":
            weight = estimate_product_density_ratio_knn(source, k=int(knn_k), seed=int(seed))
        else:
            weight = estimate_product_density_ratio_tm(source, degree=int(ratio_tm_degree))
        joint = float(
            estimate_mutual_information_transport_map(
                source,
                target,
                degree=int(tm_degree),
                sample_weight=weight,
                joint_order="source_first",
            )["mi_hat"]
        )
        singleton = [
            float(
                estimate_mutual_information_transport_map(
                    source[:, [index]],
                    target,
                    degree=int(tm_degree),
                    sample_weight=weight,
                    joint_order="source_first",
                )["mi_hat"]
            )
            for index in range(source.shape[1])
        ]
        singleton_sum = float(sum(singleton))
        dependence = gaussian_source_dependence_diagnostics(source, target, weight)
        mi_difference = joint - singleton_sum
        direct_conditional_tc = dependence["conditional_source_tc_bits"]
        replicate_rows.append(
            {
                "seed": None if seed is None else int(seed),
                "rwei_phi_bits": (
                    direct_conditional_tc if xi_method == "conditional_tc" else mi_difference
                ),
                "direct_conditional_tc_bits": direct_conditional_tc,
                "mi_difference_bits": mi_difference,
                "joint_rwei_bits": joint,
                "singleton_sum_bits": singleton_sum,
                "singleton_rwei_bits": singleton,
                "ess": effective_sample_size(weight),
                "ess_ratio": effective_sample_size(weight) / len(weight),
                "max_normalized_weight": float(weight.max() / weight.sum()),
                **dependence,
            }
        )
    return {
        "n_pairs": int(len(source)),
        "rwei_phi_bits": float(np.mean([row["rwei_phi_bits"] for row in replicate_rows])),
        "rwei_phi_sd": float(np.std([row["rwei_phi_bits"] for row in replicate_rows], ddof=1))
        if len(replicate_rows) > 1
        else 0.0,
        "joint_rwei_bits": float(np.mean([row["joint_rwei_bits"] for row in replicate_rows])),
        "singleton_sum_bits": float(np.mean([row["singleton_sum_bits"] for row in replicate_rows])),
        "ess_ratio": float(np.mean([row["ess_ratio"] for row in replicate_rows])),
        "max_normalized_weight": float(
            np.mean([row["max_normalized_weight"] for row in replicate_rows])
        ),
        "raw_source_tc_bits": float(
            np.mean([row["raw_source_tc_bits"] for row in replicate_rows])
        ),
        "weighted_source_tc_bits": float(
            np.mean([row["weighted_source_tc_bits"] for row in replicate_rows])
        ),
        "conditional_source_tc_bits": float(
            np.mean([row["conditional_source_tc_bits"] for row in replicate_rows])
        ),
        "direct_conditional_tc_bits": float(
            np.mean([row["direct_conditional_tc_bits"] for row in replicate_rows])
        ),
        "mi_difference_bits": float(
            np.mean([row["mi_difference_bits"] for row in replicate_rows])
        ),
        "tc_identity_max_abs_error_bits": float(
            max(abs(row["tc_difference_bits"] - row["mi_difference_bits"]) for row in replicate_rows)
        ),
        "xi_estimator": xi_method,
        "weight_estimator": estimator,
        "replicates": replicate_rows,
    }


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, replicates: int = 50_000) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    draw = rng.choice(array, size=(int(replicates), len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(draw, 0.025)), float(np.quantile(draw, 0.975))]


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_subject: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_subject.setdefault(str(row["subject"]), {})[str(row["condition"])] = row
    subjects = sorted(
        subject for subject, conditions in by_subject.items() if all(item in conditions for item in CONDITION_ORDER)
    )
    matrix = np.asarray(
        [[by_subject[subject][condition]["rwei_phi_bits"] for condition in CONDITION_ORDER] for subject in subjects],
        dtype=float,
    )
    ess = np.asarray(
        [[by_subject[subject][condition]["ess_ratio"] for condition in CONDITION_ORDER] for subject in subjects],
        dtype=float,
    )
    raw_source_tc = np.asarray(
        [
            [by_subject[subject][condition]["raw_source_tc_bits"] for condition in CONDITION_ORDER]
            for subject in subjects
        ],
        dtype=float,
    )
    weighted_source_tc = np.asarray(
        [
            [by_subject[subject][condition]["weighted_source_tc_bits"] for condition in CONDITION_ORDER]
            for subject in subjects
        ],
        dtype=float,
    )
    omnibus = friedmanchisquare(*[matrix[:, index] for index in range(matrix.shape[1])])
    contrasts = []
    for index, condition in enumerate(CONDITION_ORDER[1:], start=1):
        difference = matrix[:, index] - matrix[:, 0]
        statistic = wilcoxon(matrix[:, index], matrix[:, 0], alternative="two-sided", method="approx")
        contrasts.append(
            {
                "condition": condition,
                "task_minus_rest_mean": float(difference.mean()),
                "task_minus_rest_median": float(np.median(difference)),
                "bootstrap_95_ci": bootstrap_mean_ci(difference, seed=2026072200 + index),
                "task_above_rest": int(np.sum(difference > 0.0)),
                "task_below_rest": int(np.sum(difference < 0.0)),
                "wilcoxon_statistic": float(statistic.statistic),
                "wilcoxon_p_two_sided": float(statistic.pvalue),
            }
        )
    adjusted = benjamini_hochberg([row["wilcoxon_p_two_sided"] for row in contrasts])
    for row, q_value in zip(contrasts, adjusted):
        row["bh_adjusted_q"] = q_value
    diagnostic_rows = [row for row in rows if "max_normalized_weight" in row]
    weight_diagnostics = {}
    if diagnostic_rows:
        effective_n = np.asarray(
            [float(row["ess_ratio"]) * int(row["n_pairs"]) for row in diagnostic_rows], dtype=float
        )
        maximum_weight = np.asarray(
            [float(row["max_normalized_weight"]) for row in diagnostic_rows], dtype=float
        )
        identity_error = np.asarray(
            [float(row["tc_identity_max_abs_error_bits"]) for row in diagnostic_rows], dtype=float
        )
        weight_diagnostics = {
            "effective_n_mean": float(effective_n.mean()),
            "effective_n_median": float(np.median(effective_n)),
            "effective_n_min": float(effective_n.min()),
            "maximum_single_weight_mean": float(maximum_weight.mean()),
            "maximum_single_weight_max": float(maximum_weight.max()),
            "tc_identity_error_median_bits": float(np.median(identity_error)),
            "tc_identity_error_max_bits": float(identity_error.max()),
            "tc_identity_error_above_1e_3": int(np.sum(identity_error > 1.0e-3)),
        }
    return {
        "xi_estimator": str(rows[0].get("xi_estimator", "mi_difference")),
        "subjects": subjects,
        "condition_order": list(CONDITION_ORDER),
        "values_by_subject": [
            {
                "subject": subject,
                **{
                    condition: float(by_subject[subject][condition]["rwei_phi_bits"])
                    for condition in CONDITION_ORDER
                },
            }
            for subject in subjects
        ],
        "condition_summary": [
            {
                "condition": condition,
                "mean_rwei_phi_bits": float(matrix[:, index].mean()),
                "median_rwei_phi_bits": float(np.median(matrix[:, index])),
                "sd_rwei_phi_bits": float(matrix[:, index].std(ddof=1)) if len(matrix) > 1 else 0.0,
                "mean_ess_ratio": float(ess[:, index].mean()),
                "min_ess_ratio": float(ess[:, index].min()),
                "mean_raw_source_tc_bits": float(raw_source_tc[:, index].mean()),
                "mean_weighted_source_tc_bits": float(weighted_source_tc[:, index].mean()),
                "source_tc_reduction_bits": float(
                    raw_source_tc[:, index].mean() - weighted_source_tc[:, index].mean()
                ),
                "negative_subjects": int(np.sum(matrix[:, index] < 0.0)),
            }
            for index, condition in enumerate(CONDITION_ORDER)
        ],
        "friedman": {"statistic": float(omnibus.statistic), "p_value": float(omnibus.pvalue), "n": len(subjects)},
        "rest_pairwise_contrasts": contrasts,
        "rest_is_subject_maximum": int(np.sum(np.argmax(matrix, axis=1) == 0)),
        "weight_diagnostics": weight_diagnostics,
    }


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def plot_summary(summary: Mapping[str, Any], destination: Path) -> None:
    _style()
    conditions = list(summary["condition_order"])
    matrix = np.asarray(
        [[row[condition] for condition in conditions] for row in summary["values_by_subject"]], dtype=float
    )
    condition_rows = {row["condition"]: row for row in summary["condition_summary"]}
    contrasts = list(summary["rest_pairwise_contrasts"])
    rest_color, task_color = "#4C78A8", "#D98245"
    colors = [rest_color] + [task_color] * (len(conditions) - 1)
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(13.6, 3.45),
        gridspec_kw={"width_ratios": (1.75, 1.0, 0.9, 0.8)},
        constrained_layout=True,
    )

    box = axes[0].boxplot(matrix, widths=0.58, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set(facecolor=color, edgecolor=color, alpha=0.22, linewidth=0.9)
    for median in box["medians"]:
        median.set(color="#303030", linewidth=1.0)
    rng = np.random.default_rng(20260722)
    for index, color in enumerate(colors):
        axes[0].scatter(
            index + rng.uniform(-0.13, 0.13, len(matrix)),
            matrix[:, index],
            s=9,
            color=color,
            alpha=0.72,
            linewidths=0,
        )
    axes[0].axhline(0.0, color="#888888", linewidth=0.6, linestyle="--", zorder=0)
    axes[0].set(
        xticks=np.arange(1, len(conditions) + 1),
        xticklabels=[DISPLAY_NAMES[item] for item in conditions],
        ylabel=(
            "Direct conditional total correlation (bits)"
            if summary.get("xi_estimator") == "conditional_tc"
            else "One-step RW integrated EI (bits)"
        ),
    )
    axes[0].tick_params(axis="x", labelrotation=35)
    axes[0].text(0.02, 0.98, f"paired n={len(matrix)}", transform=axes[0].transAxes, va="top")

    means = np.asarray([row["task_minus_rest_mean"] for row in contrasts])
    intervals = np.asarray([row["bootstrap_95_ci"] for row in contrasts])
    y = np.arange(len(contrasts))
    axes[1].errorbar(
        means,
        y,
        xerr=np.vstack((means - intervals[:, 0], intervals[:, 1] - means)),
        fmt="o",
        color=task_color,
        ecolor=task_color,
        capsize=2,
        markersize=4,
    )
    axes[1].axvline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[1].set(
        yticks=y,
        yticklabels=[DISPLAY_NAMES[row["condition"]] for row in contrasts],
        xlabel="Task − REST (bits; mean and 95% CI)",
    )
    axes[1].invert_yaxis()
    left = min(0.0, float(intervals[:, 0].min()))
    evidence_right = max(0.0, float(intervals[:, 1].max()))
    span = max(evidence_right - left, 1.0)
    annotation_x = evidence_right + 0.16 * span
    axes[1].set_xlim(left - 0.08 * span, evidence_right + 0.36 * span)
    for position, row in enumerate(contrasts):
        axes[1].text(
            annotation_x,
            position,
            f"q={row['bh_adjusted_q']:.2g}",
            ha="left",
            va="center",
            fontsize=6.3,
        )

    x = np.arange(len(conditions))
    raw_tc = [condition_rows[item]["mean_raw_source_tc_bits"] for item in conditions]
    weighted_tc = [condition_rows[item]["mean_weighted_source_tc_bits"] for item in conditions]
    axes[2].plot(x, raw_tc, marker="o", color="#9C755F", linewidth=1.0, markersize=3.5, label="Before")
    axes[2].plot(x, weighted_tc, marker="o", color="#5B8E7D", linewidth=1.0, markersize=3.5, label="After")
    axes[2].set(
        xticks=x,
        xticklabels=[DISPLAY_NAMES[item] for item in conditions],
        ylabel="Source total correlation (bits)",
        ylim=(0.0, max(raw_tc) * 1.18),
    )
    axes[2].tick_params(axis="x", labelrotation=55)
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False)

    ess_mean = [condition_rows[item]["mean_ess_ratio"] for item in conditions]
    ess_min = [condition_rows[item]["min_ess_ratio"] for item in conditions]
    axes[3].plot(x, ess_mean, marker="o", color="#5B8E7D", linewidth=1.0, markersize=3.5, label="Mean")
    axes[3].scatter(x, ess_min, marker="v", color="#9C755F", s=18, label="Minimum")
    axes[3].set(
        xticks=x,
        xticklabels=[DISPLAY_NAMES[item] for item in conditions],
        ylabel="Weight ESS / N",
        ylim=(0.0, 1.0),
    )
    axes[3].tick_params(axis="x", labelrotation=55)
    axes[3].legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False)
    for label, axis in zip("abcd", axes):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 600}), (".svg", {}), (".pdf", {})):
        fig.savefig(destination.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_report(summary: Mapping[str, Any], config: Mapping[str, Any], path: Path) -> None:
    negative_conditions = [
        row["condition"] for row in summary["condition_summary"] if row["mean_rwei_phi_bits"] < 0.0
    ]
    significant_above = [
        row["condition"]
        for row in summary["rest_pairwise_contrasts"]
        if row["bh_adjusted_q"] < 0.05 and row["task_minus_rest_mean"] > 0.0
    ]
    significant_below = [
        row["condition"]
        for row in summary["rest_pairwise_contrasts"]
        if row["bh_adjusted_q"] < 0.05 and row["task_minus_rest_mean"] < 0.0
    ]
    direct_ctc = config["xi_estimator"] == "conditional_tc"
    if direct_ctc:
        conclusion = (
            "主指标改为直接估计高斯条件总相关，全部条件和被试均满足 $\\Xi_{\\mathrm{CTC}}\\geq0$。"
            if not negative_conditions
            else f"直接条件总相关仍出现 {len(negative_conditions)}/8 个条件均值为负，数值实现需要复查。"
        )
    else:
        conclusion = (
            f"TM 密度比下仍有 {len(negative_conditions)}/8 个状态的平均整合量为负，说明乘积干预尚未在有限加权样本中实现；当前结果不能作为有效的非负 $\\Xi$。"
            if negative_conditions
            else "TM 密度比下八状态平均整合量均非负，有限样本非负性诊断通过。"
        )
    state_finding = (
        f"显著高于 REST：{', '.join(significant_above)}；显著低于 REST：{', '.join(significant_below)}。"
        if significant_above or significant_below
        else "七个 REST–任务配对比较经 BH 校正后均不显著。"
    )
    lines = [
        "# HCP 静息态与七任务的一步 RW-EI 比较",
        "",
        "## 核心结论",
        "",
        conclusion + state_finding,
        "",
        "## 方法口径",
        "",
        f"主分析使用 {summary['friedman']['n']} 名完整配对被试。REST 和每个任务均截取前 {config['n_timepoints']} 个时间点，形成 {config['n_timepoints'] - 1} 个 `(x_t, x_(t+1))` 样本；不拟合动力学模型，也不堆叠多阶历史。每个状态独立拟合 Schaefer-500 到 Yeo7 的网络 PC1。密度比估计器为 {config['weight_estimator_description']}。条件总相关由加权条件协方差直接计算；{config['tm_degree']} 阶加权三角 TM 互信息仅用于恒等式诊断。",
        "",
        (
            "主指标直接定义为加权高斯分布下的条件总相关 `TC_RW(X_t | X_(t+1))`，单位为 bit；原来的 `I_RW(X_t; X_(t+1)) - sum_j I_RW(X_t,j; X_(t+1))` 仅保留作恒等式诊断。ESS/N 用于诊断重加权支持重叠。"
            if direct_ctc
            else "整体整合 RW-EI 定义为 `I_RW(X_t; X_(t+1)) - sum_j I_RW(X_t,j; X_(t+1))`，单位为 bit。ESS/N 用于诊断重加权支持重叠。"
        ),
        "",
        "## 条件汇总",
        "",
        "| 条件 | 直接条件 TC 均值 | 中位数 | 加权前源 TC | 加权后源 TC | 平均 ESS/N | 最小 ESS/N | 负值被试 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["condition_summary"]:
        lines.append(
            f"| {row['condition']} | {row['mean_rwei_phi_bits']:.6f} | {row['median_rwei_phi_bits']:.6f} | "
            f"{row['mean_raw_source_tc_bits']:.4f} | {row['mean_weighted_source_tc_bits']:.4f} | "
            f"{row['mean_ess_ratio']:.4f} | {row['min_ess_ratio']:.4f} | {row['negative_subjects']} |"
        )
    friedman = summary["friedman"]
    diagnostics = summary.get("weight_diagnostics", {})
    lines.extend(
        [
            "",
            f"八条件 Friedman 检验：$\\chi^2={friedman['statistic']:.6f}$，$p={friedman['p_value']:.6g}$。REST 在 {summary['rest_is_subject_maximum']}/{friedman['n']} 名被试中为八条件最大值。",
            (
                f"权重诊断：平均/中位有效样本数为 {diagnostics['effective_n_mean']:.2f}/{diagnostics['effective_n_median']:.2f}（原始 N={config['n_pairs']}），最小值 {diagnostics['effective_n_min']:.2f}；单点最大权重的跨模型均值为 {100.0 * diagnostics['maximum_single_weight_mean']:.1f}%，最极端为 {100.0 * diagnostics['maximum_single_weight_max']:.1f}%。"
                if diagnostics
                else ""
            ),
            "",
            "## REST 配对比较",
            "",
            "| 任务 | task − REST 均值差 | 95% bootstrap CI | task 高/低于 REST | Wilcoxon p | BH q |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rest_pairwise_contrasts"]:
        lines.append(
            f"| {row['condition']} | {row['task_minus_rest_mean']:.6f} | "
            f"[{row['bootstrap_95_ci'][0]:.6f}, {row['bootstrap_95_ci'][1]:.6f}] | "
            f"{row['task_above_rest']} / {row['task_below_rest']} | "
            f"{row['wilcoxon_p_two_sided']:.6g} | {row['bh_adjusted_q']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
        (
            "该结果是一步、Yeo7-PC1、仿射高斯口径的直接条件总相关估计。按定义取条件总相关保证了结构非负性，但这并不证明 TM 密度比已实现理想乘积干预。若加权后的源总相关仍明显大于零或 ESS 很低，该非负量只能视为加权观测分布的条件依赖，不能直接等同于理想干预下的 PEID 协同。任务长度已严格匹配，但 PC1 是按状态分别拟合的。"
            if direct_ctc
            else "该结果是一步、Yeo7-PC1、仿射 TM 口径的直接观测重加权估计。任务长度已严格匹配，但 PC1 是按状态分别拟合的。理论 $\\Xi$ 的非负性要求加权后的源总相关为零；因此负值是密度比/支持失配诊断，不能解释为有效的负协同。ESS 只诊断权重集中，也不能证明源变量已经独立。"
        ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    rest_root: Path,
    task_root: Path,
    labels_path: Path,
    output_dir: Path,
    *,
    subjects: Sequence[str] | None = None,
    n_timepoints: int = 176,
    knn_k: int = 20,
    tm_degree: int = 1,
    weight_estimator: str = "tm",
    ratio_tm_degree: int = 1,
    xi_estimator: str = "conditional_tc",
    weight_replicates: int = 5,
    seed: int = 20260722,
) -> dict[str, Any]:
    task_files = discover_subject_task_files(task_root, TASKS)
    rest_files = {path.parent.name: path for path in sorted(Path(rest_root).glob("sub-*/*.mat"))}
    common = sorted(set(task_files) & set(rest_files))
    if subjects is not None:
        wanted = set(subjects)
        missing = wanted - set(common)
        if missing:
            raise FileNotFoundError(f"Missing complete REST/task inputs for {sorted(missing)}")
        common = sorted(wanted)
    if not common:
        raise FileNotFoundError("No complete paired HCP subjects were found.")
    groups = load_yeo7_groups(labels_path, expected_parcels=500)
    weight_seeds = [int(seed) + replicate for replicate in range(int(weight_replicates))]
    rows = []
    reduced_cache: dict[str, np.ndarray] = {}
    for subject in common:
        raw_by_condition = {"REST": load_hcp_series(rest_files[subject])}
        raw_by_condition.update(
            {
                task.removesuffix("_LR"): load_task_series(
                    task_files[subject][task], data_key="Schaefer500_taskRetained", parcel_count=500
                )
                for task in TASKS
            }
        )
        for condition in CONDITION_ORDER:
            reduced = reduce_yeo7_pc1(raw_by_condition[condition], groups, n_timepoints=n_timepoints)
            reduced_cache[f"{subject}__{condition}"] = reduced.astype(np.float32)
            result = estimate_rwei_phi(
                reduced,
                knn_k=knn_k,
                tm_degree=tm_degree,
                weight_seeds=weight_seeds,
                weight_estimator=weight_estimator,
                ratio_tm_degree=ratio_tm_degree,
                xi_estimator=xi_estimator,
            )
            rows.append({"subject": subject, "condition": condition, **result})
    comparison = summarize(rows)
    config = {
        "representation": "condition-specific Schaefer-500 Yeo7 PC1",
        "transition": "one-step observed pairs x_t -> x_(t+1); no fitted dynamics and no history stacking",
        "n_timepoints": int(n_timepoints),
        "n_pairs": int(n_timepoints) - 1,
        "knn_k": int(knn_k),
        "weight_estimator": str(weight_estimator),
        "xi_estimator": str(xi_estimator),
        "ratio_tm_degree": int(ratio_tm_degree),
        "weight_estimator_description": (
            f"TM density ratio: product of seven univariate degree-{ratio_tm_degree} TM marginals divided by one seven-dimensional degree-{ratio_tm_degree} TM density"
            if str(weight_estimator) == "tm"
            else f"two-sample kNN density ratio with k={knn_k}"
        ),
        "target_intervention": "product of the seven empirical source marginals",
        "weight_replicates": int(weight_replicates),
        "effective_weight_replicates": int(weight_replicates) if str(weight_estimator) == "knn" else 1,
        "weight_seeds": weight_seeds if str(weight_estimator) == "knn" else [],
        "tm_degree": int(tm_degree),
        "tm_backend": f"polynomial triangular transport map degree {tm_degree}",
        "phi_definition": (
            "TC_RW(X_t | X_(t+1)) = 0.5 log2(prod_j Var_RW(X_t,j | X_(t+1)) / det Cov_RW(X_t | X_(t+1)))"
            if str(xi_estimator) == "conditional_tc"
            else "I_RW(X_t; X_(t+1)) - sum_j I_RW(X_t,j; X_(t+1))"
        ),
        "statistics": "paired two-sided Wilcoxon; Benjamini-Hochberg across seven REST contrasts; paired bootstrap mean CI",
    }
    payload = {"config": config, "rows": rows, "comparison": comparison}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(output_dir / "reduced_series.npz", **reduced_cache)
    plot_summary(comparison, output_dir / "rest_all_tasks_rwei")
    write_report(comparison, config, output_dir / "report.md")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rest-root", type=Path, default=DEFAULT_REST_ROOT)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subjects", default="")
    parser.add_argument("--n-timepoints", type=int, default=176)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--tm-degree", type=int, default=1)
    parser.add_argument("--weight-estimator", choices=("tm", "knn"), default="tm")
    parser.add_argument("--ratio-tm-degree", type=int, default=1)
    parser.add_argument(
        "--xi-estimator", choices=("conditional_tc", "mi_difference"), default="conditional_tc"
    )
    parser.add_argument("--weight-replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args(argv)
    subjects = [item.strip() for item in args.subjects.split(",") if item.strip()] or None
    payload = run(
        args.rest_root,
        args.task_root,
        args.labels,
        args.output_dir,
        subjects=subjects,
        n_timepoints=args.n_timepoints,
        knn_k=args.knn_k,
        tm_degree=args.tm_degree,
        weight_estimator=args.weight_estimator,
        ratio_tm_degree=args.ratio_tm_degree,
        xi_estimator=args.xi_estimator,
        weight_replicates=args.weight_replicates,
        seed=args.seed,
    )
    print(json.dumps(payload["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
