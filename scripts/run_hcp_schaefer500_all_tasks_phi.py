#!/usr/bin/env python3
"""Compare observed raw Yeo7-PC1 PhiEID across all HCP task states and REST."""

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
from scipy.stats import friedmanchisquare
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hcp_schaefer500_wm_yeo7_phi import load_task_series, paired_summary
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import (
    default_yeo7_labels,
    fit_yeo7_pc1,
    load_yeo7_groups,
)


TASKS = ("EMOTION_LR", "GAMBLING_LR", "LANGUAGE_LR", "MOTOR_LR", "RELATIONAL_LR", "SOCIAL_LR", "WM_LR")
CONDITION_ORDER = ("REST",) + tuple(task.removesuffix("_LR") for task in TASKS)
DISPLAY_NAMES = {
    "REST": "REST",
    "EMOTION": "Emotion",
    "GAMBLING": "Gambling",
    "LANGUAGE": "Language",
    "MOTOR": "Motor",
    "RELATIONAL": "Relational",
    "SOCIAL": "Social",
    "WM": "WM",
}
DEFAULT_DATA_ROOT = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_30"
DEFAULT_REST_SUMMARY = ROOT / "results" / "hcp_schaefer500_yeo7_pc1_phi_null_all" / "summary.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "hcp_schaefer500_all_tasks_phi"


def development_end_for_length(length: int) -> int:
    return int(np.floor(0.75 * int(length) + 0.5))


def discover_subject_task_files(data_root: Path, tasks: Sequence[str]) -> dict[str, dict[str, Path]]:
    root = Path(data_root)
    discovered: dict[str, dict[str, Path]] = {}
    for folder in sorted(root.glob("sub-*")):
        if not folder.is_dir():
            continue
        paths = {task: folder / f"{task}.mat" for task in tasks}
        if all(path.is_file() for path in paths.values()):
            discovered[folder.name] = paths
    return discovered


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = [float(value) for value in p_values]
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def analyze_series(
    raw: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    *,
    order: int,
    alpha: float,
) -> dict[str, Any]:
    development_end = development_end_for_length(len(raw))
    reduced = fit_yeo7_pc1(np.asarray(raw, dtype=float)[:development_end], groups).transform(raw)
    fitted = fit_delta_history_phi(reduced, alpha=alpha, order=order, development_end=development_end)
    noise = np.asarray(fitted["noise_covariance"], dtype=float)
    development = np.asarray(reduced[:development_end], dtype=float)
    scale = np.where(development.std(axis=0, ddof=1) > 1.0e-12, development.std(axis=0, ddof=1), 1.0)
    max_zscore = float(np.max(np.abs((development - development.mean(axis=0)) / scale)))
    noise_condition = float(np.linalg.cond(noise))
    return {
        "n_timepoints": int(len(raw)),
        "development_end": development_end,
        "observed_raw_phi": float(fitted["phi"]["raw_phi"]),
        "joint_ei": float(fitted["phi"]["joint_ei"]),
        "singleton_ei_sum": float(fitted["phi"]["singleton_ei_sum"]),
        "quality_diagnostics": {
            "max_abs_development_pc_zscore": max_zscore,
            "noise_covariance_condition": noise_condition,
            "quality_flag": bool(max_zscore > 10.0 or noise_condition > 500.0),
            "flag_rule": "max_abs_development_pc_zscore > 10 or noise_covariance_condition > 500",
        },
    }


def build_comparison(rows: Sequence[Mapping[str, Any]], rest_summary_path: Path) -> dict[str, Any]:
    rest = json.loads(Path(rest_summary_path).read_text(encoding="utf-8"))
    rest_by_subject = {str(row["subject"]): float(row["observed_raw_phi"]) for row in rest["rows"]}
    task_by_subject: dict[str, dict[str, float]] = {}
    for row in rows:
        task_by_subject.setdefault(str(row["subject"]), {})[str(row["condition"])] = float(row["observed_raw_phi"])
    common = sorted(
        subject
        for subject, values in task_by_subject.items()
        if subject in rest_by_subject and all(condition in values for condition in CONDITION_ORDER[1:])
    )
    matrix = np.asarray(
        [
            [rest_by_subject[subject]] + [task_by_subject[subject][condition] for condition in CONDITION_ORDER[1:]]
            for subject in common
        ],
        dtype=float,
    )
    omnibus = friedmanchisquare(*[matrix[:, index] for index in range(matrix.shape[1])])
    contrasts = []
    for index, condition in enumerate(CONDITION_ORDER[1:], start=1):
        result = paired_summary(matrix[:, index], matrix[:, 0], seed=2026071800 + index)
        result["condition"] = condition
        contrasts.append(result)
    adjusted = holm_adjust([result["paired_sign_flip_p_two_sided"] for result in contrasts])
    for result, value in zip(contrasts, adjusted):
        result["holm_adjusted_p"] = value

    descending_ranks = np.empty_like(matrix, dtype=int)
    for row_index, values in enumerate(matrix):
        descending_ranks[row_index, np.argsort(-values)] = np.arange(1, len(values) + 1)
    maximum_conditions = [CONDITION_ORDER[int(index)] for index in np.argmax(matrix, axis=1)]
    return {
        "common_subjects": common,
        "task_only_subjects": sorted(set(task_by_subject) - set(rest_by_subject)),
        "rest_only_subjects": sorted(set(rest_by_subject) - set(task_by_subject)),
        "condition_order": list(CONDITION_ORDER),
        "values_by_subject": [
            {"subject": subject, **{condition: float(matrix[row_index, index]) for index, condition in enumerate(CONDITION_ORDER)}}
            for row_index, subject in enumerate(common)
        ],
        "condition_summary": [
            {
                "condition": condition,
                "mean": float(matrix[:, index].mean()),
                "median": float(np.median(matrix[:, index])),
                "std": float(matrix[:, index].std(ddof=1)),
            }
            for index, condition in enumerate(CONDITION_ORDER)
        ],
        "friedman": {"statistic": float(omnibus.statistic), "p_value": float(omnibus.pvalue), "n": len(common)},
        "rest_pairwise_contrasts": contrasts,
        "rest_rank": {
            "mean": float(descending_ranks[:, 0].mean()),
            "median": float(np.median(descending_ranks[:, 0])),
            "rank_counts": {str(rank): int(np.sum(descending_ranks[:, 0] == rank)) for rank in range(1, len(CONDITION_ORDER) + 1)},
        },
        "maximum_condition_counts": {condition: int(Counter(maximum_conditions).get(condition, 0)) for condition in CONDITION_ORDER},
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


def _save(fig: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_comparison(summary: Mapping[str, Any], destination: Path) -> None:
    _style()
    comparison = summary["comparison"]
    conditions = list(comparison["condition_order"])
    values = np.asarray(
        [[row[condition] for condition in conditions] for row in comparison["values_by_subject"]], dtype=float
    )
    rest_color = "#4C78A8"
    task_color = "#D07A3A"
    colors = [rest_color] + [task_color] * (len(conditions) - 1)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.2, 3.5),
        gridspec_kw={"width_ratios": (1.65, 0.9, 0.9)},
        constrained_layout=True,
    )

    box = axes[0].boxplot(values, positions=np.arange(len(conditions)), widths=0.58, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set(facecolor=color, alpha=0.22, edgecolor=color, linewidth=0.9)
    for median in box["medians"]:
        median.set(color="#303030", linewidth=1.0)
    rng = np.random.default_rng(20260718)
    for index, color in enumerate(colors):
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        axes[0].scatter(index + jitter, values[:, index], s=9, color=color, alpha=0.72, linewidths=0)
    axes[0].set(
        xticks=np.arange(len(conditions)),
        xticklabels=[DISPLAY_NAMES[condition] for condition in conditions],
        ylabel="Raw history-source $\\Phi^{EID}$ (bits)",
    )
    axes[0].tick_params(axis="x", labelrotation=35)
    axes[0].text(0.02, 0.98, f"paired n={len(values)}", transform=axes[0].transAxes, va="top")

    rest_values = values[:, conditions.index("REST")]
    wm_values = values[:, conditions.index("WM")]
    for rest_value, wm_value in zip(rest_values, wm_values):
        axes[1].plot([0, 1], [rest_value, wm_value], color="#B7BEC8", linewidth=0.6, zorder=1)
    axes[1].scatter(np.zeros(len(rest_values)), rest_values, color=rest_color, s=13, zorder=2)
    axes[1].scatter(np.ones(len(wm_values)), wm_values, color=task_color, s=13, zorder=2)
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["REST", "WM"],
        ylabel="Raw $\\Phi^{EID}$ (bits)",
        ylim=axes[0].get_ylim(),
    )
    wm_contrast = next(item for item in comparison["rest_pairwise_contrasts"] if item["condition"] == "WM")
    axes[1].text(
        0.02,
        0.98,
        f"paired n={len(rest_values)}\n$p$={wm_contrast['paired_sign_flip_p_two_sided']:.3g}",
        transform=axes[1].transAxes,
        va="top",
    )
    flagged_subjects = {
        str(item["subject"])
        for item in summary["quality_flags"]
        if str(item["condition"]) == "WM"
    }
    for row, wm_value in zip(comparison["values_by_subject"], wm_values):
        if str(row["subject"]) in flagged_subjects:
            axes[1].annotate(
                str(row["subject"]).removeprefix("sub-"),
                xy=(1, wm_value),
                xytext=(-4, 5),
                textcoords="offset points",
                ha="right",
                fontsize=6,
            )

    counts = np.asarray([comparison["maximum_condition_counts"][condition] for condition in conditions])
    axes[2].barh(np.arange(len(conditions)), counts, color=colors, alpha=0.85)
    axes[2].set(
        yticks=np.arange(len(conditions)),
        yticklabels=[DISPLAY_NAMES[condition] for condition in conditions],
        xlabel="Subjects with condition-specific maximum",
        xlim=(0, max(1, int(counts.max())) * 1.15),
    )
    axes[2].invert_yaxis()
    for position, count in enumerate(counts):
        axes[2].text(count + 0.2, position, str(int(count)), va="center", fontsize=6.5)
    for label, axis in zip("abc", axes):
        axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    _save(fig, destination)


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    comparison = summary["comparison"]
    lines = [
        "# HCP Schaefer-500 静息态与七任务 raw Phi 比较",
        "",
        "每个任务与每名被试均独立拟合 Yeo7 PC1、标准化、八阶 Δ-Ridge 系数和残差协方差；跨条件固定的是模型结构与 `p=8, alpha=10`，不是回归系数。任务态均使用各自前 75% 时间点，静息态直接读取既有结果。本轮不计算 circular-shift null。",
        "",
        "## 条件汇总",
        "",
        "| 条件 | raw Phi 均值 | 中位数 | 标准差 |",
        "|---|---:|---:|---:|",
    ]
    for item in comparison["condition_summary"]:
        lines.append(f"| {item['condition']} | {item['mean']:.6f} | {item['median']:.6f} | {item['std']:.6f} |")
    friedman = comparison["friedman"]
    lines.extend(
        [
            "",
            f"29 名共同被试的八条件 Friedman 检验：$\\chi^2={friedman['statistic']:.6f}$，$p={friedman['p_value']:.6g}$。",
            f"REST 的平均/中位降序排名为 {comparison['rest_rank']['mean']:.3f}/{comparison['rest_rank']['median']:.3f}；"
            f"REST 在 {comparison['maximum_condition_counts']['REST']}/{friedman['n']} 名被试中为八条件最大值。",
            "",
            "## REST 与各任务配对比较",
            "",
            "| 任务 | task − REST 均值差 | 95% bootstrap CI | 正/负差人数 | sign-flip p | Holm p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in comparison["rest_pairwise_contrasts"]:
        lines.append(
            f"| {item['condition']} | {item['mean_difference']:.6f} | "
            f"[{item['bootstrap_95_ci'][0]:.6f}, {item['bootstrap_95_ci'][1]:.6f}] | "
            f"{item['positive_differences']} / {item['negative_differences']} | "
            f"{item['paired_sign_flip_p_two_sided']:.6g} | {item['holm_adjusted_p']:.6g} |"
        )
    flagged = summary["quality_flags"]
    lines.extend(
        [
            "",
            f"质量规则标记 {len(flagged)} 个任务-被试组合：" + (", ".join(f"{item['subject']}/{item['condition']}" for item in flagged) if flagged else "无") + "。主分析保留全部被试。",
            "",
            "该比较只支持当前 Gaussian raw Phi 口径下的条件排序；未使用 null，因此不能判断条件差异是否超出各任务自身的边际时间结构。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    data_root: Path,
    rest_summary_path: Path,
    labels_path: Path,
    output_dir: Path,
    *,
    tasks: Sequence[str] = TASKS,
    subjects: Sequence[str] | None = None,
    data_key: str = "Schaefer500_taskRetained",
    order: int = 8,
    alpha: float = 10.0,
) -> dict[str, Any]:
    discovered = discover_subject_task_files(data_root, tasks)
    if subjects is not None:
        wanted = set(subjects)
        missing = wanted - set(discovered)
        if missing:
            raise FileNotFoundError(f"Missing complete task inputs for {sorted(missing)}")
        discovered = {subject: discovered[subject] for subject in sorted(wanted)}
    if not discovered:
        raise FileNotFoundError(f"No subjects with all requested task files below {data_root}.")
    groups = load_yeo7_groups(labels_path, expected_parcels=500)
    rows = []
    progress = tqdm(total=len(tasks) * len(discovered), desc="observed task Phi", unit="model", mininterval=1.0)
    try:
        for task in tasks:
            condition = task.removesuffix("_LR")
            for subject, paths in discovered.items():
                progress.set_postfix(task=condition, subject=subject.removeprefix("sub-"), refresh=False)
                raw = load_task_series(paths[task], data_key=data_key, parcel_count=500)
                row = analyze_series(raw, groups, order=order, alpha=alpha)
                rows.append({"subject": subject, "condition": condition, "task": task, **row})
                progress.update(1)
    finally:
        progress.close()
    comparison = build_comparison(rows, rest_summary_path)
    quality_flags = [
        {"subject": row["subject"], "condition": row["condition"], **row["quality_diagnostics"]}
        for row in rows
        if row["quality_diagnostics"]["quality_flag"]
    ]
    summary = {
        "config": {
            "tasks": list(tasks),
            "data_key": data_key,
            "subjects": list(discovered),
            "representation": "Yeo7 PC1 refitted per subject and task on the first 75% of time points",
            "model": "delta Ridge refitted per subject and task",
            "order": int(order),
            "alpha": float(alpha),
            "source_dimension": int(order * 7),
            "target_dimension": 7,
            "estimator": "Gaussian log-det raw history-source PhiEID",
            "null_replicates": 0,
        },
        "rows": rows,
        "comparison": comparison,
        "quality_flags": quality_flags,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(summary, output_dir / "report.md")
    plot_comparison(summary, output_dir / "rest_all_tasks_raw_phi")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--rest-summary", type=Path, default=DEFAULT_REST_SUMMARY)
    parser.add_argument("--labels", type=Path, default=default_yeo7_labels(500))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subjects", default="", help="Comma-separated subset for smoke tests.")
    parser.add_argument("--tasks", default=",".join(TASKS), help="Comma-separated task names without .mat.")
    parser.add_argument("--data-key", default="Schaefer500_taskRetained")
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=10.0)
    args = parser.parse_args(argv)
    subjects = tuple(value.strip() for value in args.subjects.split(",") if value.strip()) or None
    tasks = tuple(value.strip() for value in args.tasks.split(",") if value.strip())
    summary = run(
        args.data_root,
        args.rest_summary,
        args.labels,
        args.output_dir,
        tasks=tasks,
        subjects=subjects,
        data_key=args.data_key,
        order=args.order,
        alpha=args.alpha,
    )
    print(json.dumps({"friedman": summary["comparison"]["friedman"], "condition_summary": summary["comparison"]["condition_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
