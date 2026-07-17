#!/usr/bin/env python3
"""Test whether REST raw Phi variance remains larger after exact task-length matching."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hcp_schaefer500_all_tasks_phi import DISPLAY_NAMES, analyze_series
from scripts.run_hcp_schaefer500_phi_hyperparameter_robustness import discover_rest_files
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import (
    default_yeo7_labels,
    load_hcp_series,
    load_yeo7_groups,
)


CONDITIONS = ("EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")
DEFAULT_REST_ROOT = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30"
DEFAULT_TASK_SUMMARY = ROOT / "results" / "hcp_schaefer500_all_tasks_phi" / "summary.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "hcp_schaefer500_length_matched_variance"


def evenly_spaced_window_starts(total_length: int, window_length: int, n_windows: int) -> np.ndarray:
    """Return deterministic integer starts spanning the full admissible REST interval."""
    total = int(total_length)
    length = int(window_length)
    count = int(n_windows)
    if count < 1:
        raise ValueError("n_windows must be positive.")
    if length < 1 or length > total:
        raise ValueError("window_length must be in [1, total_length].")
    maximum = total - length
    if count == 1:
        return np.asarray([maximum // 2], dtype=int)
    starts = np.rint(np.linspace(0, maximum, count)).astype(int)
    starts = np.unique(starts)
    if len(starts) != count:
        raise ValueError("n_windows exceeds the number of distinct integer window starts.")
    return starts


def _mad(values: np.ndarray, *, axis: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    median = np.median(array, axis=axis, keepdims=True)
    result = np.median(np.abs(array - median), axis=axis)
    return np.asarray(result, dtype=float)


def window_spread_summary(rest_windows: np.ndarray, task_values: np.ndarray) -> dict[str, Any]:
    """Summarize matched REST spread across subjects for every window position."""
    rest = np.asarray(rest_windows, dtype=float)
    task = np.asarray(task_values, dtype=float)
    if rest.ndim != 2 or task.ndim != 1 or len(rest) != len(task) or len(task) < 3:
        raise ValueError("Expected REST [subjects, windows] and paired task [subjects].")
    rest_variances = np.var(rest, axis=0, ddof=1)
    task_variance = float(np.var(task, ddof=1))
    rest_iqrs = np.quantile(rest, 0.75, axis=0) - np.quantile(rest, 0.25, axis=0)
    task_iqr = float(np.quantile(task, 0.75) - np.quantile(task, 0.25))
    rest_mads = _mad(rest, axis=0)
    task_mad = float(_mad(task))
    if min(task_variance, task_iqr, task_mad) <= 0.0:
        raise ValueError("Task spread must be positive for variance-ratio analysis.")
    variance_ratios = rest_variances / task_variance
    subject_mean = rest.mean(axis=1)
    return {
        "task_mean": float(task.mean()),
        "task_std": float(task.std(ddof=1)),
        "task_variance": task_variance,
        "task_iqr": task_iqr,
        "task_mad": task_mad,
        "rest_window_mean": float(rest.mean()),
        "rest_window_std_values": np.sqrt(rest_variances).tolist(),
        "rest_window_variance_values": rest_variances.tolist(),
        "rest_window_variance_ratio_values": variance_ratios.tolist(),
        "mean_window_variance": float(rest_variances.mean()),
        "mean_window_variance_ratio": float(variance_ratios.mean()),
        "median_window_variance_ratio": float(np.median(variance_ratios)),
        "minimum_window_variance_ratio": float(variance_ratios.min()),
        "maximum_window_variance_ratio": float(variance_ratios.max()),
        "windows_with_rest_variance_greater": int(np.sum(variance_ratios > 1.0)),
        "n_windows": int(rest.shape[1]),
        "mean_window_iqr_ratio": float(rest_iqrs.mean() / task_iqr),
        "mean_window_mad_ratio": float(rest_mads.mean() / task_mad),
        "subject_mean_rest_std": float(subject_mean.std(ddof=1)),
        "subject_mean_rest_variance_ratio": float(np.var(subject_mean, ddof=1) / task_variance),
    }


def hierarchical_bootstrap_spread_ratios(
    rest_windows: np.ndarray,
    task_values: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, list[float]]:
    """Bootstrap paired subjects and REST window positions for spread-ratio intervals."""
    rest = np.asarray(rest_windows, dtype=float)
    task = np.asarray(task_values, dtype=float)
    if rest.ndim != 2 or task.shape != (rest.shape[0],):
        raise ValueError("REST and task inputs are not paired by subject.")
    rng = np.random.default_rng(int(seed))
    n_subjects, n_windows = rest.shape
    target = int(replicates)
    variance_ratios = np.empty(target, dtype=float)
    iqr_ratios = np.empty(target, dtype=float)
    mad_ratios = np.empty(target, dtype=float)
    kept = 0
    attempts = 0
    maximum_attempts = max(100, target * 2)
    while kept < target and attempts < maximum_attempts:
        chunk = min(2_000, target - kept)
        attempts += chunk
        subject_indices = rng.integers(0, n_subjects, size=(chunk, n_subjects))
        window_indices = rng.integers(0, n_windows, size=(chunk, n_windows))
        sampled_rest = rest[subject_indices[:, :, None], window_indices[:, None, :]]
        sampled_task = task[subject_indices]
        task_variance = np.var(sampled_task, axis=1, ddof=1)
        task_quantiles = np.quantile(sampled_task, [0.25, 0.75], axis=1)
        task_iqr = task_quantiles[1] - task_quantiles[0]
        task_median = np.median(sampled_task, axis=1, keepdims=True)
        task_mad = np.median(np.abs(sampled_task - task_median), axis=1)
        valid = (task_variance > 1.0e-12) & (task_iqr > 1.0e-12) & (task_mad > 1.0e-12)
        if not np.any(valid):
            continue
        rest_variance = np.var(sampled_rest, axis=1, ddof=1).mean(axis=1)
        rest_quantiles = np.quantile(sampled_rest, [0.25, 0.75], axis=1)
        rest_iqr = (rest_quantiles[1] - rest_quantiles[0]).mean(axis=1)
        rest_median = np.median(sampled_rest, axis=1, keepdims=True)
        rest_mad = np.median(np.abs(sampled_rest - rest_median), axis=1).mean(axis=1)
        valid_indices = np.flatnonzero(valid)
        take = min(len(valid_indices), target - kept)
        selected = valid_indices[:take]
        destination = slice(kept, kept + take)
        variance_ratios[destination] = rest_variance[selected] / task_variance[selected]
        iqr_ratios[destination] = rest_iqr[selected] / task_iqr[selected]
        mad_ratios[destination] = rest_mad[selected] / task_mad[selected]
        kept += take
    if kept != target:
        raise RuntimeError("Could not obtain the requested number of nondegenerate bootstrap replicates.")

    def interval(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]

    return {
        "mean_window_variance_ratio_95_ci": interval(variance_ratios),
        "mean_window_iqr_ratio_95_ci": interval(iqr_ratios),
        "mean_window_mad_ratio_95_ci": interval(mad_ratios),
    }


def load_task_reference(
    path: Path, conditions: Sequence[str]
) -> tuple[list[str], dict[str, int], dict[str, np.ndarray], np.ndarray, Mapping[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    comparison = payload["comparison"]
    subjects = [str(subject) for subject in comparison["common_subjects"]]
    rows_by_subject = {str(row["subject"]): row for row in comparison["values_by_subject"]}
    if subjects != sorted(subjects) or set(subjects) != set(rows_by_subject):
        raise ValueError("Task summary common-subject ordering is inconsistent.")
    task_values = {
        condition: np.asarray([rows_by_subject[subject][condition] for subject in subjects], dtype=float)
        for condition in conditions
    }
    full_rest = np.asarray([rows_by_subject[subject]["REST"] for subject in subjects], dtype=float)
    lengths: dict[str, int] = {}
    for condition in conditions:
        observed_lengths = {
            int(row["n_timepoints"])
            for row in payload["rows"]
            if str(row["condition"]) == condition and str(row["subject"]) in set(subjects)
        }
        if len(observed_lengths) != 1:
            raise ValueError(f"Expected one shared task length for {condition}, got {sorted(observed_lengths)}.")
        lengths[condition] = observed_lengths.pop()
    return subjects, lengths, task_values, full_rest, payload


def _fit_subject_windows(
    subject: str,
    rest_path: str,
    labels_path: str,
    conditions: tuple[str, ...],
    lengths: tuple[int, ...],
    starts: tuple[tuple[int, ...], ...],
    order: int,
    alpha: float,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    raw = load_hcp_series(Path(rest_path), parcel_count=500, data_key="Schaefer500")
    groups = load_yeo7_groups(Path(labels_path), expected_parcels=500)
    n_windows = len(starts[0])
    phi = np.empty((len(conditions), n_windows), dtype=float)
    max_zscore = np.empty_like(phi)
    noise_condition = np.empty_like(phi)
    for condition_index, (length, condition_starts) in enumerate(zip(lengths, starts)):
        for window_index, start in enumerate(condition_starts):
            result = analyze_series(raw[start : start + length], groups, order=int(order), alpha=float(alpha))
            phi[condition_index, window_index] = result["observed_raw_phi"]
            diagnostics = result["quality_diagnostics"]
            max_zscore[condition_index, window_index] = diagnostics["max_abs_development_pc_zscore"]
            noise_condition[condition_index, window_index] = diagnostics["noise_covariance_condition"]
    return subject, phi, max_zscore, noise_condition


def _checkpoint_metadata(
    subjects: Sequence[str],
    conditions: Sequence[str],
    lengths: Mapping[str, int],
    starts: Mapping[str, np.ndarray],
    rest_paths: Mapping[str, Path],
    labels_path: Path,
    *,
    order: int,
    alpha: float,
) -> dict[str, Any]:
    return {
        "subjects": list(subjects),
        "conditions": list(conditions),
        "lengths": {condition: int(lengths[condition]) for condition in conditions},
        "starts": {condition: [int(value) for value in starts[condition]] for condition in conditions},
        "order": int(order),
        "alpha": float(alpha),
        "rest_sources": {subject: str(Path(rest_paths[subject]).resolve()) for subject in subjects},
        "labels_path": str(Path(labels_path).resolve()),
    }


def _save_checkpoint(
    path: Path,
    metadata: Mapping[str, Any],
    phi: np.ndarray,
    max_zscore: np.ndarray,
    noise_condition: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(
        temporary,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        phi=np.asarray(phi, dtype=float),
        max_zscore=np.asarray(max_zscore, dtype=float),
        noise_condition=np.asarray(noise_condition, dtype=float),
    )
    temporary.replace(path)


def _load_checkpoint(
    path: Path, metadata: Mapping[str, Any], shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    empty = tuple(np.full(shape, np.nan, dtype=float) for _ in range(3))
    if not path.is_file():
        return empty
    with np.load(path) as payload:
        cached_metadata = json.loads(str(payload["metadata"].item()))
        arrays = tuple(np.asarray(payload[key], dtype=float) for key in ("phi", "max_zscore", "noise_condition"))
    if cached_metadata != dict(metadata) or any(array.shape != shape for array in arrays):
        return empty
    return arrays


def fit_all_windows(
    rest_paths: Mapping[str, Path],
    labels_path: Path,
    subjects: Sequence[str],
    conditions: Sequence[str],
    lengths: Mapping[str, int],
    starts: Mapping[str, np.ndarray],
    checkpoint_path: Path,
    *,
    order: int,
    alpha: float,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata = _checkpoint_metadata(
        subjects,
        conditions,
        lengths,
        starts,
        rest_paths,
        labels_path,
        order=order,
        alpha=alpha,
    )
    shape = (len(subjects), len(conditions), len(starts[conditions[0]]))
    phi, max_zscore, noise_condition = _load_checkpoint(checkpoint_path, metadata, shape)
    pending = [index for index in range(len(subjects)) if not np.isfinite(phi[index]).all()]
    if not pending:
        return phi, max_zscore, noise_condition
    missing_paths = [subjects[index] for index in pending if subjects[index] not in rest_paths]
    if missing_paths:
        raise FileNotFoundError(f"Missing REST inputs for {missing_paths}.")
    condition_tuple = tuple(conditions)
    length_tuple = tuple(int(lengths[condition]) for condition in conditions)
    starts_tuple = tuple(tuple(int(value) for value in starts[condition]) for condition in conditions)
    subject_to_index = {subject: index for index, subject in enumerate(subjects)}
    progress = tqdm(
        total=len(pending) * len(conditions) * shape[2],
        desc="length-matched REST Phi",
        unit="fit",
        mininterval=1.0,
    )
    try:
        if int(workers) == 1:
            results = (
                _fit_subject_windows(
                    subjects[index],
                    str(rest_paths[subjects[index]]),
                    str(labels_path),
                    condition_tuple,
                    length_tuple,
                    starts_tuple,
                    int(order),
                    float(alpha),
                )
                for index in pending
            )
            for subject, subject_phi, subject_z, subject_condition in results:
                index = subject_to_index[subject]
                phi[index], max_zscore[index], noise_condition[index] = subject_phi, subject_z, subject_condition
                _save_checkpoint(checkpoint_path, metadata, phi, max_zscore, noise_condition)
                progress.update(len(conditions) * shape[2])
        else:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                futures = {
                    executor.submit(
                        _fit_subject_windows,
                        subjects[index],
                        str(rest_paths[subjects[index]]),
                        str(labels_path),
                        condition_tuple,
                        length_tuple,
                        starts_tuple,
                        int(order),
                        float(alpha),
                    ): subjects[index]
                    for index in pending
                }
                for future in as_completed(futures):
                    subject, subject_phi, subject_z, subject_condition = future.result()
                    index = subject_to_index[subject]
                    phi[index], max_zscore[index], noise_condition[index] = subject_phi, subject_z, subject_condition
                    _save_checkpoint(checkpoint_path, metadata, phi, max_zscore, noise_condition)
                    progress.update(len(conditions) * shape[2])
    finally:
        progress.close()
    return phi, max_zscore, noise_condition


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


def _save_figure(fig: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary: Mapping[str, Any], destination: Path) -> None:
    _style()
    conditions = list(summary["config"]["conditions"])
    items = {item["condition"]: item for item in summary["condition_results"]}
    rest_color = "#4C78A8"
    task_color = "#D07A3A"
    fig, axes_grid = plt.subplots(2, 4, figsize=(10.8, 6.0), sharey=True, constrained_layout=True)
    axes = list(axes_grid.flat)
    plotted_values = []
    for condition in conditions:
        paired = items[condition]["paired_subject_values"]
        plotted_values.extend(float(row["matched_rest_phi"]) for row in paired)
        plotted_values.extend(float(row["task_phi"]) for row in paired)
    lower = float(np.floor(min(plotted_values) - 0.4))
    upper = float(np.ceil(max(plotted_values) + 0.4))

    for condition_index, (axis, condition) in enumerate(zip(axes, conditions)):
        item = items[condition]
        paired = item["paired_subject_values"]
        subjects = [str(row["subject"]) for row in paired]
        rest_values = np.asarray([row["matched_rest_phi"] for row in paired], dtype=float)
        task_values = np.asarray([row["task_phi"] for row in paired], dtype=float)
        rng = np.random.default_rng(2026073100 + condition_index)
        jitter = rng.uniform(-0.07, 0.07, size=len(paired))
        for rest_value, task_value, offset in zip(rest_values, task_values, jitter):
            axis.plot(
                [offset, 1.0 + offset],
                [rest_value, task_value],
                color="#B7BEC8",
                linewidth=0.55,
                alpha=0.65,
                zorder=1,
            )
        box = axis.boxplot(
            [rest_values, task_values],
            positions=[0.0, 1.0],
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#303030", "linewidth": 1.0},
            whiskerprops={"color": "#7B8490", "linewidth": 0.7},
            capprops={"color": "#7B8490", "linewidth": 0.7},
        )
        for patch, color in zip(box["boxes"], [rest_color, task_color]):
            patch.set(facecolor=color, alpha=0.16, edgecolor=color, linewidth=0.9)
        axis.scatter(jitter, rest_values, s=11, color=rest_color, alpha=0.82, linewidths=0, zorder=3)
        axis.scatter(1.0 + jitter, task_values, s=11, color=task_color, alpha=0.82, linewidths=0, zorder=3)
        axis.set(
            xticks=[0.0, 1.0],
            xticklabels=["REST\nmatched", DISPLAY_NAMES[condition]],
            xlim=(-0.42, 1.42),
            ylim=(lower, upper),
            title=f"{DISPLAY_NAMES[condition]}  ·  {item['task_length']} time points",
        )
        axis.text(
            0.03,
            0.97,
            f"SD: REST {rest_values.std(ddof=1):.2f}  |  task {task_values.std(ddof=1):.2f}",
            transform=axis.transAxes,
            va="top",
            fontsize=6.2,
        )
        if condition == "WM" and "sub-103515" in subjects:
            subject_index = subjects.index("sub-103515")
            axis.annotate(
                "103515",
                xy=(1.0 + jitter[subject_index], task_values[subject_index]),
                xytext=(-3, -9),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=5.8,
                color="#4A4A4A",
            )
    axes[0].set_ylabel(r"Raw $\Phi^{EID}$ (bits)")
    axes[4].set_ylabel(r"Raw $\Phi^{EID}$ (bits)")

    legend_axis = axes[-1]
    legend_axis.axis("off")
    legend_axis.scatter([], [], s=18, color=rest_color, label="Matched REST")
    legend_axis.scatter([], [], s=18, color=task_color, label="Task")
    legend_axis.plot([], [], color="#B7BEC8", linewidth=0.7, label="Same subject")
    legend_axis.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), borderaxespad=0.0)
    legend_axis.text(
        0.02,
        0.58,
        "Each dot is one subject (n=29).\n"
        "REST is the subject mean across\n"
        "12 equal-length windows.\n"
        "All panels share the same y-axis.",
        transform=legend_axis.transAxes,
        va="top",
        linespacing=1.45,
    )
    _save_figure(fig, destination)


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    config = summary["config"]
    lines = [
        "# HCP REST–任务长度匹配 raw Phi 方差比较",
        "",
        "## 实验契约",
        "",
        "唯一处理因素是 REST 截窗长度：对每个任务使用完全相同的时间点数，并在每个 REST 窗口内重新拟合 Yeo7 PC1、标准化、八阶 Δ-Ridge 与残差协方差。任务态 raw Phi 复用原比较结果；模型结构固定为 `p=8, alpha=10`。未使用 null 模型。",
        "",
        f"共同被试 `n={config['n_subjects']}`；每种任务长度使用 {config['n_windows']} 个覆盖完整 REST1_LR run 的等距窗口。主统计量是各窗口位置跨被试方差的均值除以对应任务跨被试方差；95% 区间同时对配对被试和窗口位置分层 bootstrap（{config['bootstrap_replicates']:,} 次）。",
        "",
        "图中按任务作七组 REST–任务两两比较：每名被试的 REST 点是 12 个等长窗口 raw Phi 的均值，任务点是该被试的任务 raw Phi，灰线连接同一被试。所有面板共享纵轴，并直接标出图中两组分布的标准差。",
        "",
        "## 结果",
        "",
        "| 任务 | 长度 | task SD | 图中 REST SD | 单窗口 REST SD 范围 | 平均方差比 [95% CI] | REST 方差较大的窗口 | IQR 比 | MAD 比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["condition_results"]:
        rest_stds = np.asarray(item["rest_window_std_values"], dtype=float)
        ci = item["mean_window_variance_ratio_95_ci"]
        lines.append(
            f"| {item['condition']} | {item['task_length']} | {item['task_std']:.3f} | "
            f"{item['subject_mean_rest_std']:.3f} | "
            f"{rest_stds.min():.3f}–{rest_stds.max():.3f} | {item['mean_window_variance_ratio']:.3f} "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {item['windows_with_rest_variance_greater']}/{item['n_windows']} | "
            f"{item['mean_window_iqr_ratio']:.3f} | {item['mean_window_mad_ratio']:.3f} |"
        )
    larger = [
        item["condition"]
        for item in summary["condition_results"]
        if item["minimum_window_variance_ratio"] > 1.0
    ]
    ci_above = [
        item["condition"]
        for item in summary["condition_results"]
        if item["mean_window_variance_ratio_95_ci"][0] > 1.0
    ]
    robust_iqr_above = [
        item["condition"]
        for item in summary["condition_results"]
        if item["mean_window_iqr_ratio_95_ci"][0] > 1.0
    ]
    robust_mad_above = [
        item["condition"]
        for item in summary["condition_results"]
        if item["mean_window_mad_ratio_95_ci"][0] > 1.0
    ]
    sensitivity_lines = []
    for item in summary["condition_results"]:
        sensitivity = item.get("task_quality_flag_sensitivity")
        if sensitivity:
            excluded = ", ".join(sensitivity["excluded_subjects"])
            sensitivity_lines.append(
                f"{item['condition']} 排除预先质量规则标记的 {excluded} 后，task SD 从 {item['task_std']:.3f} 降至 "
                f"{sensitivity['task_std']:.3f}，REST/task 平均窗口方差比从 {item['mean_window_variance_ratio']:.3f} "
                f"升至 {sensitivity['mean_window_variance_ratio']:.3f}（95% CI "
                f"[{sensitivity['mean_window_variance_ratio_95_ci'][0]:.3f}, "
                f"{sensitivity['mean_window_variance_ratio_95_ci'][1]:.3f}]）。"
            )
    flags = summary["quality_diagnostics"]
    lines.extend(
        [
            "",
            "## 解释",
            "",
            f"全部 {config['n_windows']} 个窗口位置的 REST 方差都高于任务的条件：{', '.join(larger) if larger else '无'}。平均窗口方差比的 bootstrap 区间整体高于 1 的条件：{', '.join(ci_above) if ci_above else '无'}。",
            "",
            f"稳健离散度的 bootstrap 区间整体高于 1：IQR 为 {', '.join(robust_iqr_above) if robust_iqr_above else '无'}；MAD 为 {', '.join(robust_mad_above) if robust_mad_above else '无'}。",
            "",
            *sensitivity_lines,
            "" if sensitivity_lines else "",
            f"REST 窗口质量规则标记 {flags['n_flagged_windows']}/{flags['n_total_windows']} 个拟合；主分析保留全部窗口。",
            "",
            "这里沿用原始 Gaussian log-det history-source Phi 估计器，是为了只改变序列长度；改用 TM 会同时改变估计器，无法把差异归因于长度匹配。结果因此回答的是原分析口径下的方差稳健性，不是所有 EI/Phi 估计器下的普遍结论。重叠窗口用于覆盖 REST run，不应当作独立被试；推断中的实验单位始终是被试，窗口位置仅作为重复测量层级。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    rest_root: Path,
    task_summary_path: Path,
    labels_path: Path,
    output_dir: Path,
    *,
    conditions: Sequence[str] = CONDITIONS,
    n_windows: int = 12,
    order: int = 8,
    alpha: float = 10.0,
    bootstrap_replicates: int = 5_000,
    workers: int = 8,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subjects, lengths, task_values, full_rest, task_payload = load_task_reference(task_summary_path, conditions)
    rest_paths = discover_rest_files(rest_root)
    starts = {
        condition: evenly_spaced_window_starts(1200, lengths[condition], int(n_windows))
        for condition in conditions
    }
    contract = {
        "scientific_question": "What changes in across-subject raw Phi variance when only REST sequence length is changed to equal each task length?",
        "treatment_factor": "REST window length",
        "treatment_levels": {condition: int(lengths[condition]) for condition in conditions},
        "unit_of_pairing": "HCP subject",
        "primary_metric": "mean across window positions of Var_subject(REST-window Phi) / Var_subject(task Phi)",
        "secondary_diagnostics": ["IQR ratio", "MAD ratio", "per-window variance ratios", "quality flags"],
        "frozen": {
            "subjects": subjects,
            "preprocessing": "condition/window-local Yeo7 PC1 fitted on first 75% only",
            "source_target": "56-dimensional p=8 Yeo7 history to next 7-dimensional Yeo7 state",
            "train_fraction": 0.75,
            "window_positions": {condition: starts[condition].tolist() for condition in conditions},
            "estimator": "Gaussian log-det raw history-source PhiEID",
            "ridge_alpha": float(alpha),
            "null_model": "none",
            "task_reference": "existing task raw-Phi summary from the identical estimator pipeline",
        },
    }
    (output_dir / "experiment_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    phi, max_zscore, noise_condition = fit_all_windows(
        rest_paths,
        labels_path,
        subjects,
        conditions,
        lengths,
        starts,
        output_dir / "rest_window_phi.npz",
        order=order,
        alpha=alpha,
        workers=workers,
    )
    if not np.isfinite(phi).all():
        raise RuntimeError("Length-matched REST cache is incomplete.")
    condition_results = []
    task_flags = task_payload.get("quality_flags", [])
    for condition_index, condition in enumerate(conditions):
        result = window_spread_summary(phi[:, condition_index, :], task_values[condition])
        result.update(
            hierarchical_bootstrap_spread_ratios(
                phi[:, condition_index, :],
                task_values[condition],
                seed=2026072100 + condition_index,
                replicates=int(bootstrap_replicates),
            )
        )
        result.update(
            {
                "condition": condition,
                "task_length": int(lengths[condition]),
                "window_starts": starts[condition].tolist(),
                "paired_subject_values": [
                    {
                        "subject": subject,
                        "matched_rest_phi": float(phi[subject_index, condition_index, :].mean()),
                        "task_phi": float(task_values[condition][subject_index]),
                    }
                    for subject_index, subject in enumerate(subjects)
                ],
            }
        )
        flagged_task_subjects = sorted(
            {
                str(item["subject"])
                for item in task_flags
                if str(item["condition"]) == condition and str(item["subject"]) in set(subjects)
            }
        )
        if flagged_task_subjects:
            keep = np.asarray([subject not in set(flagged_task_subjects) for subject in subjects], dtype=bool)
            sensitivity = window_spread_summary(phi[keep, condition_index, :], task_values[condition][keep])
            sensitivity.update(
                hierarchical_bootstrap_spread_ratios(
                    phi[keep, condition_index, :],
                    task_values[condition][keep],
                    seed=2026072900 + condition_index,
                    replicates=int(bootstrap_replicates),
                )
            )
            result["task_quality_flag_sensitivity"] = {
                "excluded_subjects": flagged_task_subjects,
                "n_subjects": int(keep.sum()),
                **sensitivity,
            }
        condition_results.append(result)
    flagged = (max_zscore > 10.0) | (noise_condition > 500.0)
    summary = {
        "config": {
            "conditions": list(conditions),
            "subjects": subjects,
            "n_subjects": len(subjects),
            "n_windows": int(n_windows),
            "development_fraction": 0.75,
            "order": int(order),
            "alpha": float(alpha),
            "source_dimension": int(7 * order),
            "target_dimension": 7,
            "estimator": "Gaussian log-det raw history-source PhiEID",
            "estimator_reason": "kept identical to the original REST-task comparison so length is the only changed analysis factor",
            "bootstrap_replicates": int(bootstrap_replicates),
            "null_replicates": 0,
            "task_summary_config": task_payload.get("config", {}),
        },
        "experiment_contract": contract,
        "full_length_rest": {
            "mean": float(full_rest.mean()),
            "std": float(full_rest.std(ddof=1)),
            "variance": float(full_rest.var(ddof=1)),
        },
        "condition_results": condition_results,
        "quality_diagnostics": {
            "n_total_windows": int(flagged.size),
            "n_flagged_windows": int(flagged.sum()),
            "flag_rule": "max_abs_development_pc_zscore > 10 or noise_covariance_condition > 500",
            "max_abs_development_pc_zscore": float(max_zscore.max()),
            "maximum_noise_covariance_condition": float(noise_condition.max()),
            "flagged_rest_windows_by_condition": {
                condition: int(flagged[:, condition_index, :].sum())
                for condition_index, condition in enumerate(conditions)
            },
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, output_dir / "report.md")
    plot_summary(summary, output_dir / "length_matched_variance")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rest-root", type=Path, default=DEFAULT_REST_ROOT)
    parser.add_argument("--task-summary", type=Path, default=DEFAULT_TASK_SUMMARY)
    parser.add_argument("--labels", type=Path, default=default_yeo7_labels(500))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--n-windows", type=int, default=12)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    conditions = tuple(value.strip() for value in args.conditions.split(",") if value.strip())
    summary = run(
        args.rest_root,
        args.task_summary,
        args.labels,
        args.output_dir,
        conditions=conditions,
        n_windows=args.n_windows,
        order=args.order,
        alpha=args.alpha,
        bootstrap_replicates=args.bootstrap_replicates,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                item["condition"]: {
                    "variance_ratio": item["mean_window_variance_ratio"],
                    "ci": item["mean_window_variance_ratio_95_ci"],
                    "windows_rest_greater": item["windows_with_rest_variance_greater"],
                }
                for item in summary["condition_results"]
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
