#!/usr/bin/env python3
"""Test REST-versus-task raw PhiEID robustness over a fixed (p, alpha) grid."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
from scipy.stats import friedmanchisquare
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hcp_schaefer500_all_tasks_phi import (
    DISPLAY_NAMES,
    TASKS,
    development_end_for_length,
    discover_subject_task_files,
    holm_adjust,
)
from scripts.run_hcp_schaefer500_wm_yeo7_phi import load_task_series
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import (
    default_yeo7_labels,
    fit_yeo7_pc1,
    load_hcp_series,
    load_yeo7_groups,
)


CONDITIONS = ("REST",) + tuple(task.removesuffix("_LR") for task in TASKS)
DEFAULT_TASK_ROOT = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_30"
DEFAULT_REST_ROOT = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "hcp_schaefer500_phi_hyperparameter_robustness"
DEFAULT_ORDERS = (1, 2, 3, 5, 8)
DEFAULT_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def discover_rest_files(root: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for folder in sorted(Path(root).glob("sub-*")):
        candidates = sorted(folder.glob("*_rfMRI_REST1_LR_schaefer500-1000_yeo7.mat"))
        if len(candidates) == 1:
            discovered[folder.name] = candidates[0]
    return discovered


def _cache_key(subject: str, condition: str) -> str:
    return f"{subject.removeprefix('sub-')}__{condition}"


def prepare_reduced_series(
    task_root: Path,
    rest_root: Path,
    labels_path: Path,
    cache_path: Path,
    *,
    task_data_key: str,
) -> tuple[list[str], dict[tuple[str, str], np.ndarray], dict[str, int]]:
    task_files = discover_subject_task_files(task_root, TASKS)
    rest_files = discover_rest_files(rest_root)
    common = sorted(set(task_files) & set(rest_files))
    if not common:
        raise FileNotFoundError("No subjects are shared by complete REST and seven-task inputs.")
    expected_keys = {_cache_key(subject, condition) for subject in common for condition in CONDITIONS}
    if cache_path.is_file():
        with np.load(cache_path) as payload:
            if expected_keys == set(payload.files):
                reduced = {
                    (subject, condition): np.asarray(payload[_cache_key(subject, condition)], dtype=float)
                    for subject in common
                    for condition in CONDITIONS
                }
                lengths = {condition: int(len(reduced[(common[0], condition)])) for condition in CONDITIONS}
                return common, reduced, lengths

    groups = load_yeo7_groups(labels_path, expected_parcels=500)
    reduced: dict[tuple[str, str], np.ndarray] = {}
    progress = tqdm(total=len(common) * len(CONDITIONS), desc="Yeo7 PC1", unit="series", mininterval=1.0)
    try:
        for subject in common:
            rest_raw = load_hcp_series(rest_files[subject], parcel_count=500, data_key="Schaefer500")
            rest_end = development_end_for_length(len(rest_raw))
            reduced[(subject, "REST")] = fit_yeo7_pc1(rest_raw[:rest_end], groups).transform(rest_raw)
            progress.update(1)
            for task in TASKS:
                condition = task.removesuffix("_LR")
                task_raw = load_task_series(task_files[subject][task], data_key=task_data_key, parcel_count=500)
                task_end = development_end_for_length(len(task_raw))
                reduced[(subject, condition)] = fit_yeo7_pc1(task_raw[:task_end], groups).transform(task_raw)
                progress.update(1)
    finally:
        progress.close()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        **{_cache_key(subject, condition): reduced[(subject, condition)] for subject in common for condition in CONDITIONS},
    )
    lengths = {condition: int(len(reduced[(common[0], condition)])) for condition in CONDITIONS}
    return common, reduced, lengths


def paired_sign_flip_p(differences: np.ndarray, *, seed: int, replicates: int) -> float:
    values = np.asarray(differences, dtype=float)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(int(seed))
    extreme = 0
    remaining = int(replicates)
    while remaining:
        size = min(20_000, remaining)
        signs = rng.choice((-1.0, 1.0), size=(size, len(values)))
        extreme += int(np.sum(np.abs(np.mean(signs * values, axis=1)) >= observed - 1.0e-15))
        remaining -= size
    return float((extreme + 1) / (int(replicates) + 1))


def summarize_grid_point(
    values: np.ndarray,
    *,
    order: int,
    alpha: float,
    permutation_replicates: int,
    grid_index: int,
) -> dict[str, Any]:
    rest = values[:, 0]
    contrasts = []
    for condition_index, condition in enumerate(CONDITIONS[1:], start=1):
        task = values[:, condition_index]
        differences = task - rest
        contrasts.append(
            {
                "condition": condition,
                "task_minus_rest_mean": float(differences.mean()),
                "rest_minus_task_mean": float(-differences.mean()),
                "task_minus_rest_median": float(np.median(differences)),
                "positive_task_minus_rest_subjects": int(np.sum(differences > 0)),
                "negative_task_minus_rest_subjects": int(np.sum(differences < 0)),
                "paired_sign_flip_p_two_sided": paired_sign_flip_p(
                    differences,
                    seed=2026071900 + grid_index * 10 + condition_index,
                    replicates=permutation_replicates,
                ),
            }
        )
    adjusted = holm_adjust([item["paired_sign_flip_p_two_sided"] for item in contrasts])
    for item, adjusted_p in zip(contrasts, adjusted):
        item["holm_adjusted_p"] = float(adjusted_p)
        item["holm_significant_0_05"] = bool(adjusted_p < 0.05)
    means = values.mean(axis=0)
    omnibus = friedmanchisquare(*[values[:, index] for index in range(values.shape[1])])
    maxima = [CONDITIONS[int(index)] for index in np.argmax(values, axis=1)]
    return {
        "order": int(order),
        "alpha": float(alpha),
        "source_dimension": int(7 * order),
        "condition_means": {condition: float(means[index]) for index, condition in enumerate(CONDITIONS)},
        "rest_is_highest_group_mean": bool(means[0] > np.max(means[1:])),
        "minimum_rest_minus_task_mean": float(np.min(means[0] - means[1:])),
        "minimum_margin_task": CONDITIONS[1 + int(np.argmin(means[0] - means[1:]))],
        "n_holm_significant_rest_task_contrasts": int(sum(item["holm_significant_0_05"] for item in contrasts)),
        "all_seven_holm_significant": bool(all(item["holm_significant_0_05"] for item in contrasts)),
        "friedman": {"statistic": float(omnibus.statistic), "p_value": float(omnibus.pvalue)},
        "maximum_condition_counts": {condition: int(Counter(maxima).get(condition, 0)) for condition in CONDITIONS},
        "rest_task_contrasts": contrasts,
    }


def run_grid(
    reduced: Mapping[tuple[str, str], np.ndarray],
    subjects: Sequence[str],
    orders: Sequence[int],
    alphas: Sequence[float],
    *,
    permutation_replicates: int,
    checkpoint_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    grid: list[dict[str, Any]] = []
    total = len(subjects) * len(CONDITIONS) * len(orders) * len(alphas)
    progress = tqdm(total=total, desc="raw Phi grid", unit="fit", mininterval=1.0)
    try:
        for order_index, order in enumerate(orders):
            for alpha_index, alpha in enumerate(alphas):
                values = np.empty((len(subjects), len(CONDITIONS)), dtype=float)
                for subject_index, subject in enumerate(subjects):
                    for condition_index, condition in enumerate(CONDITIONS):
                        series = reduced[(subject, condition)]
                        development_end = development_end_for_length(len(series))
                        fitted = fit_delta_history_phi(
                            series,
                            alpha=float(alpha),
                            order=int(order),
                            development_end=development_end,
                        )
                        phi = float(fitted["phi"]["raw_phi"])
                        values[subject_index, condition_index] = phi
                        rows.append(
                            {
                                "subject": subject,
                                "condition": condition,
                                "order": int(order),
                                "alpha": float(alpha),
                                "development_end": int(development_end),
                                "raw_phi": phi,
                            }
                        )
                        progress.update(1)
                grid_index = order_index * len(alphas) + alpha_index
                grid.append(
                    summarize_grid_point(
                        values,
                        order=int(order),
                        alpha=float(alpha),
                        permutation_replicates=permutation_replicates,
                        grid_index=grid_index,
                    )
                )
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text(
                    json.dumps({"completed_grid_points": grid, "rows": rows}, indent=2), encoding="utf-8"
                )
    finally:
        progress.close()
    return rows, grid


def aggregate_grid(grid: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    weakest = min(grid, key=lambda item: float(item["minimum_rest_minus_task_mean"]))
    strongest = max(grid, key=lambda item: float(item["minimum_rest_minus_task_mean"]))
    contrast_margins = [
        (float(item["rest_minus_task_mean"]), int(point["order"]), float(point["alpha"]), str(item["condition"]))
        for point in grid
        for item in point["rest_task_contrasts"]
    ]
    global_weakest = min(contrast_margins)
    return {
        "n_grid_points": int(len(grid)),
        "n_rest_highest_group_mean": int(sum(bool(item["rest_is_highest_group_mean"]) for item in grid)),
        "n_all_seven_holm_significant": int(sum(bool(item["all_seven_holm_significant"]) for item in grid)),
        "minimum_n_holm_significant": int(min(int(item["n_holm_significant_rest_task_contrasts"]) for item in grid)),
        "weakest_grid_point": {
            "order": int(weakest["order"]),
            "alpha": float(weakest["alpha"]),
            "minimum_rest_minus_task_mean": float(weakest["minimum_rest_minus_task_mean"]),
            "minimum_margin_task": str(weakest["minimum_margin_task"]),
        },
        "strongest_grid_point_by_minimum_margin": {
            "order": int(strongest["order"]),
            "alpha": float(strongest["alpha"]),
            "minimum_rest_minus_task_mean": float(strongest["minimum_rest_minus_task_mean"]),
            "minimum_margin_task": str(strongest["minimum_margin_task"]),
        },
        "weakest_individual_contrast": {
            "rest_minus_task_mean": global_weakest[0],
            "order": global_weakest[1],
            "alpha": global_weakest[2],
            "condition": global_weakest[3],
        },
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
        }
    )


def _save(fig: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _grid_matrix(
    grid: Sequence[Mapping[str, Any]],
    orders: Sequence[int],
    alphas: Sequence[float],
    getter: Any,
) -> np.ndarray:
    lookup = {(int(item["order"]), float(item["alpha"])): item for item in grid}
    return np.asarray([[getter(lookup[(int(order), float(alpha))]) for alpha in alphas] for order in orders], dtype=float)


def _format_heatmap_axis(axis: Any, orders: Sequence[int], alphas: Sequence[float], *, title: str) -> None:
    axis.set(
        xticks=np.arange(len(alphas)),
        xticklabels=[f"{alpha:g}" for alpha in alphas],
        yticks=np.arange(len(orders)),
        yticklabels=[str(order) for order in orders],
        xlabel=r"Ridge $\alpha$",
        ylabel=r"History order $p$",
        title=title,
    )


def plot_overview(
    grid: Sequence[Mapping[str, Any]], orders: Sequence[int], alphas: Sequence[float], destination: Path
) -> None:
    _style()
    margins = _grid_matrix(grid, orders, alphas, lambda item: item["minimum_rest_minus_task_mean"])
    significant = _grid_matrix(grid, orders, alphas, lambda item: item["n_holm_significant_rest_task_contrasts"])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), constrained_layout=True)
    limit = max(abs(float(margins.min())), abs(float(margins.max())), 0.1)
    margin_image = axes[0].imshow(margins, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
    _format_heatmap_axis(axes[0], orders, alphas, title="Minimum REST advantage across seven tasks")
    for row in range(len(orders)):
        for column in range(len(alphas)):
            axes[0].text(column, row, f"{margins[row, column]:.2f}", ha="center", va="center", fontsize=6)
    colorbar = fig.colorbar(margin_image, ax=axes[0], shrink=0.83, pad=0.03)
    colorbar.set_label(r"min(REST $-$ task) raw $\Phi^{EID}$ (bits)")

    count_image = axes[1].imshow(
        significant,
        cmap="YlGn",
        norm=BoundaryNorm(np.arange(-0.5, 8.5, 1), ncolors=256),
        vmin=None,
        vmax=None,
    )
    _format_heatmap_axis(axes[1], orders, alphas, title="Holm-significant REST–task contrasts")
    for row in range(len(orders)):
        for column in range(len(alphas)):
            axes[1].text(column, row, f"{int(significant[row, column])}/7", ha="center", va="center", fontsize=6)
    colorbar = fig.colorbar(count_image, ax=axes[1], ticks=np.arange(0, 8), shrink=0.83, pad=0.03)
    colorbar.set_label("Number significant at 0.05")
    for label, axis in zip("ab", axes):
        axis.text(-0.14, 1.06, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    _save(fig, destination)


def plot_task_margins(
    grid: Sequence[Mapping[str, Any]], orders: Sequence[int], alphas: Sequence[float], destination: Path
) -> None:
    _style()
    matrices = {}
    for condition in CONDITIONS[1:]:
        matrices[condition] = _grid_matrix(
            grid,
            orders,
            alphas,
            lambda point, condition=condition: next(
                item["rest_minus_task_mean"] for item in point["rest_task_contrasts"] if item["condition"] == condition
            ),
        )
    limit = max(max(abs(float(values.min())), abs(float(values.max()))) for values in matrices.values())
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(2, 4, figsize=(10.0, 5.2), constrained_layout=True)
    image = None
    for axis, condition in zip(axes.flat, CONDITIONS[1:]):
        values = matrices[condition]
        image = axis.imshow(values, cmap="RdBu_r", norm=norm)
        _format_heatmap_axis(axis, orders, alphas, title=DISPLAY_NAMES[condition])
        for row in range(len(orders)):
            for column in range(len(alphas)):
                axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=5.6)
    axes.flat[-1].axis("off")
    assert image is not None
    colorbar = fig.colorbar(image, ax=list(axes.flat), shrink=0.82, pad=0.02)
    colorbar.set_label(r"REST $-$ task raw $\Phi^{EID}$ (bits)")
    _save(fig, destination)


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    aggregate = summary["aggregate"]
    lines = [
        "# REST–七任务 raw Phi 超参数鲁棒性",
        "",
        "同一组 `(p, alpha)` 同时用于 REST 和七种任务态；每名被试、每个状态独立重新拟合 PCA、标准化、Ridge 系数、截距与残差协方差。分析限于 29 名共同被试，各时间序列使用前 75%，不计算 null。",
        "",
        f"扫描 {aggregate['n_grid_points']} 组参数。REST 在 {aggregate['n_rest_highest_group_mean']}/{aggregate['n_grid_points']} 组中为群体均值最高状态；在 {aggregate['n_all_seven_holm_significant']}/{aggregate['n_grid_points']} 组中，七项 paired sign-flip 检验均在七任务内 Holm 校正后显著。",
        "",
        "## 网格结果",
        "",
        "| p | alpha | 最小 REST−任务均值差 | 最小边际对应任务 | Holm 显著数 | REST 均值最高 |",
        "|---:|---:|---:|---|---:|:---:|",
    ]
    for item in summary["grid_points"]:
        lines.append(
            f"| {item['order']} | {item['alpha']:g} | {item['minimum_rest_minus_task_mean']:.6f} | "
            f"{item['minimum_margin_task']} | {item['n_holm_significant_rest_task_contrasts']}/7 | "
            f"{'是' if item['rest_is_highest_group_mean'] else '否'} |"
        )
    weakest = aggregate["weakest_grid_point"]
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            f"最弱网格点为 `p={weakest['order']}, alpha={weakest['alpha']:g}`；最小边际对应任务为 {weakest['minimum_margin_task']}，REST−任务均值差为 {weakest['minimum_rest_minus_task_mean']:.6f} bits。",
            "",
            "`p` 改变 source 维数（7p），因此只应在相同 p、alpha 下做 REST–任务比较；不同 p 的 raw Phi 绝对值不是严格相同维度的 estimand。Holm 校正只在每个网格点的七任务内进行，没有把 25 个网格点再作为发现性假设族校正；网格显著数应作为敏感性描述，而不是 175 项独立发现。本结果验证的是所声明网格内的超参数鲁棒性，不证明所有可能超参数、时间截取、RL run 或预处理选择下均成立。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    task_root: Path,
    rest_root: Path,
    labels_path: Path,
    output_dir: Path,
    *,
    orders: Sequence[int] = DEFAULT_ORDERS,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    task_data_key: str = "Schaefer500_taskRetained",
    permutation_replicates: int = 200_000,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subjects, reduced, lengths = prepare_reduced_series(
        task_root,
        rest_root,
        labels_path,
        output_dir / "reduced_series_cache.npz",
        task_data_key=task_data_key,
    )
    rows, grid = run_grid(
        reduced,
        subjects,
        orders,
        alphas,
        permutation_replicates=permutation_replicates,
        checkpoint_path=output_dir / "checkpoint.json",
    )
    summary = {
        "config": {
            "task_data_key": task_data_key,
            "subjects": list(subjects),
            "n_subjects": len(subjects),
            "conditions": list(CONDITIONS),
            "orders": [int(value) for value in orders],
            "alphas": [float(value) for value in alphas],
            "timepoints": lengths,
            "development_fraction": 0.75,
            "model": "Yeo7-PC1 delta Ridge independently refitted per subject and condition",
            "estimator": "Gaussian log-det raw history-source PhiEID",
            "null_replicates": 0,
            "paired_test": "two-sided Monte Carlo sign-flip",
            "permutation_replicates": int(permutation_replicates),
            "multiplicity": "Holm correction across the seven REST-task contrasts within each grid point",
        },
        "aggregate": aggregate_grid(grid),
        "grid_points": grid,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, output_dir / "report.md")
    plot_overview(grid, orders, alphas, output_dir / "hyperparameter_robustness_overview")
    plot_task_margins(grid, orders, alphas, output_dir / "hyperparameter_task_margins")
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
    parser.add_argument("--orders", default=",".join(map(str, DEFAULT_ORDERS)))
    parser.add_argument("--alphas", default=",".join(map(str, DEFAULT_ALPHAS)))
    parser.add_argument("--task-data-key", default="Schaefer500_taskRetained")
    parser.add_argument("--permutation-replicates", type=int, default=200_000)
    args = parser.parse_args(argv)
    summary = run(
        args.task_root,
        args.rest_root,
        args.labels,
        args.output_dir,
        orders=_parse_ints(args.orders),
        alphas=_parse_floats(args.alphas),
        task_data_key=args.task_data_key,
        permutation_replicates=args.permutation_replicates,
    )
    print(json.dumps(summary["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
