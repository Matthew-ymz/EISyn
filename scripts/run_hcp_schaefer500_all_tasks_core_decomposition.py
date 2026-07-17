#!/usr/bin/env python3
"""Decompose Yeo7 PhiEID cores for all HCP task states using the pooled-optimal model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
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

from scripts.run_hcp_schaefer500_all_tasks_phi import (
    DISPLAY_NAMES,
    TASKS,
    development_end_for_length,
    discover_subject_task_files,
)
from scripts.run_hcp_schaefer500_wm_yeo7_phi import (
    NETWORK_ABBREVIATIONS,
    _atomic_json,
    analyze_subject,
    load_task_series,
)
from scripts.run_hcp_schaefer500_yeo7_module_phi_decomposition import (
    null_rank_frequency,
    summarize_cores,
    summarize_null_cores,
)
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import default_yeo7_labels, load_yeo7_groups


DEFAULT_DATA_ROOT = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_30"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "hcp_schaefer500_all_tasks_core_decomposition"
DEFAULT_STATUS = ROOT / "docs" / "log" / "hcp_schaefer500_all_tasks_core_decomposition" / "live_progress.json"
NETWORK_ORDER = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")


def _analyze_subject_file(payload: tuple[Any, ...]) -> dict[str, Any]:
    (
        path,
        groups,
        subject,
        data_key,
        order,
        alpha,
        null_replicates,
        seed,
        top_k,
    ) = payload
    raw = load_task_series(Path(path), data_key=str(data_key), parcel_count=500)
    return analyze_subject(
        raw,
        groups,
        subject=str(subject),
        development_end=development_end_for_length(len(raw)),
        order=int(order),
        alpha=float(alpha),
        null_replicates=int(null_replicates),
        seed=int(seed),
        top_k=int(top_k),
        on_model_complete=None,
    )


def _core_key(sources: Sequence[str]) -> tuple[str, ...]:
    selected = set(sources)
    return tuple(name for name in NETWORK_ORDER if name in selected)


def _core_label(sources: Sequence[str]) -> str:
    names = [NETWORK_ABBREVIATIONS[name] for name in sources]
    if len(names) <= 3:
        return " + ".join(names)
    split = (len(names) + 1) // 2
    return " + ".join(names[:split]) + "\n" + " + ".join(names[split:])


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


def summarize_task(task: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    core_summary = summarize_cores(rows)
    null_rank = [
        null_rank_frequency(rows, core["sources"], observed_frequency=int(core["top_frequency"]))
        for core in core_summary
    ]
    return {
        "task": task,
        "condition": task.removesuffix("_LR"),
        "n_subjects": len(rows),
        "core_summary": core_summary,
        "null_rank_comparison": null_rank,
        "null_core_summary": summarize_null_cores(rows),
        "quality_flags": [
            {"subject": row["subject"], **row["quality_diagnostics"]}
            for row in rows
            if row["quality_diagnostics"]["quality_flag"]
        ],
        "rows": list(rows),
    }


def build_cross_task_summary(task_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_condition = {str(result["condition"]): result for result in task_results}
    rows_by_condition_subject = {
        condition: {str(row["subject"]): row for row in result["rows"]}
        for condition, result in by_condition.items()
    }
    subjects = sorted(next(iter(rows_by_condition_subject.values())))
    union: set[tuple[str, ...]] = set()
    for result in task_results:
        union.update(_core_key(core["sources"]) for core in result["core_summary"][:3])

    frequency_lookup: dict[tuple[str, tuple[str, ...]], int] = {}
    contribution_lookup: dict[tuple[str, tuple[str, ...]], float | None] = {}
    null_lookup: dict[tuple[str, tuple[str, ...]], Mapping[str, Any]] = {}
    for condition, result in by_condition.items():
        for core in result["core_summary"]:
            key = _core_key(core["sources"])
            frequency_lookup[(condition, key)] = int(core["top_frequency"])
            contribution_lookup[(condition, key)] = float(core["mean_atom_value_when_top"])
        for comparison in result["null_rank_comparison"]:
            null_lookup[(condition, _core_key(comparison["sources"]))] = comparison

    top_three_by_task = {
        condition: {_core_key(core["sources"]) for core in result["core_summary"][:3]}
        for condition, result in by_condition.items()
    }
    selected = sorted(
        union,
        key=lambda core: (
            -sum(core in top_three_by_task[condition] for condition in by_condition),
            -sum(frequency_lookup.get((condition, core), 0) for condition in by_condition),
            -max(frequency_lookup.get((condition, core), 0) for condition in by_condition),
            core,
        ),
    )

    core_task_rows = []
    for core in selected:
        task_values = {}
        for condition in by_condition:
            comparison = null_lookup.get((condition, core))
            task_values[condition] = {
                "top_frequency": int(frequency_lookup.get((condition, core), 0)),
                "mean_atom_value_when_top": contribution_lookup.get((condition, core)),
                "null_rank_frequency_mean": float(comparison["null_frequency_mean"]) if comparison else None,
                "null_rank_frequency_max": int(comparison["null_frequency_max"]) if comparison else None,
                "null_rank_empirical_p": float(comparison["empirical_p"]) if comparison else None,
                "minimum_null_p_flag": bool(comparison and float(comparison["empirical_p"]) <= 1.0 / 21.0 + 1.0e-12),
            }
        subject_task_counts = []
        for subject in subjects:
            count = sum(
                any(_core_key(atom["sources"]) == core for atom in rows_by_condition_subject[condition][subject]["top_atoms"])
                for condition in by_condition
            )
            subject_task_counts.append(int(count))
        distribution = {
            str(count): int(sum(value == count for value in subject_task_counts))
            for count in range(len(by_condition) + 1)
            if any(value == count for value in subject_task_counts)
        }
        core_task_rows.append(
            {
                "sources": list(core),
                "n_tasks_in_per_task_top3": int(sum(core in values for values in top_three_by_task.values())),
                "total_top_frequency": int(sum(item["top_frequency"] for item in task_values.values())),
                "subject_task_consistency": {
                    "mean_tasks_per_subject": float(np.mean(subject_task_counts)),
                    "n_subjects_all_tasks": int(sum(value == len(by_condition) for value in subject_task_counts)),
                    "n_subjects_at_least_six_tasks": int(sum(value >= 6 for value in subject_task_counts)),
                    "n_subjects_at_least_four_tasks": int(sum(value >= 4 for value in subject_task_counts)),
                    "task_count_distribution": distribution,
                },
                "task_values": task_values,
            }
        )

    shared = [
        row
        for row in core_task_rows
        if sum(item["top_frequency"] >= 15 for item in row["task_values"].values()) >= 4
    ]
    exploratory_null_supported = []
    for condition, result in by_condition.items():
        comparison_by_core = {
            _core_key(item["sources"]): item for item in result["null_rank_comparison"]
        }
        for core in result["core_summary"]:
            key = _core_key(core["sources"])
            comparison = comparison_by_core[key]
            if int(core["top_frequency"]) >= 15 and float(comparison["empirical_p"]) <= 1.0 / 21.0 + 1.0e-12:
                exploratory_null_supported.append(
                    {
                        "condition": condition,
                        "sources": list(key),
                        "top_frequency": int(core["top_frequency"]),
                        "mean_atom_value_when_top": float(core["mean_atom_value_when_top"]),
                        "null_frequency_mean": float(comparison["null_frequency_mean"]),
                        "null_frequency_max": int(comparison["null_frequency_max"]),
                        "empirical_p": float(comparison["empirical_p"]),
                    }
                )

    participation = {}
    for condition, result in by_condition.items():
        counts = {network: 0 for network in NETWORK_ORDER}
        total_atoms = 0
        for row in result["rows"]:
            for atom in row["top_atoms"]:
                total_atoms += 1
                for network in atom["sources"]:
                    counts[network] += 1
        participation[condition] = {
            network: float(counts[network] / total_atoms) if total_atoms else 0.0 for network in NETWORK_ORDER
        }
    return {
        "selected_core_rule": "union of the three highest-frequency observed cores from each task, deduplicated",
        "selected_cores": core_task_rows,
        "shared_core_rule": "top-3 frequency >=15/30 in at least four tasks",
        "shared_cores": shared,
        "exploratory_null_supported_rule": "observed top-3 frequency >=15/30 and uncorrected 20-null rank p=1/21",
        "exploratory_null_supported_cores": exploratory_null_supported,
        "network_participation_fraction": participation,
    }


def plot_core_landscape(summary: Mapping[str, Any], destination: Path) -> None:
    _style()
    conditions = list(summary["config"]["conditions"])
    cores = summary["cross_task"]["selected_cores"]
    frequency = np.asarray(
        [[row["task_values"][condition]["top_frequency"] for condition in conditions] for row in cores], dtype=float
    )
    contribution = np.asarray(
        [
            [
                np.nan if row["task_values"][condition]["mean_atom_value_when_top"] is None
                else row["task_values"][condition]["mean_atom_value_when_top"]
                for condition in conditions
            ]
            for row in cores
        ],
        dtype=float,
    )
    flags = np.asarray(
        [[row["task_values"][condition]["minimum_null_p_flag"] for condition in conditions] for row in cores], dtype=bool
    )
    labels = [_core_label(row["sources"]) for row in cores]
    height = max(5.0, 0.43 * len(cores) + 1.6)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, height), constrained_layout=True, sharey=True)

    freq_image = axes[0].imshow(frequency, cmap="YlGnBu", vmin=0, vmax=30, aspect="auto")
    axes[0].set(
        xticks=np.arange(len(conditions)),
        xticklabels=[DISPLAY_NAMES[condition] for condition in conditions],
        yticks=np.arange(len(cores)),
        yticklabels=labels,
        xlabel="Task state",
        ylabel="Yeo7 synergy core",
    )
    axes[0].tick_params(axis="x", labelrotation=35)
    for row in range(len(cores)):
        for column in range(len(conditions)):
            color = "white" if frequency[row, column] >= 19 else "black"
            marker = "*" if flags[row, column] else ""
            axes[0].text(column, row, f"{int(frequency[row, column])}{marker}", ha="center", va="center", fontsize=6, color=color)
    colorbar = fig.colorbar(freq_image, ax=axes[0], shrink=0.84, pad=0.02)
    colorbar.set_label("Subjects with core in top-3 (n=30)")

    masked = np.ma.masked_invalid(contribution)
    contribution_image = axes[1].imshow(masked, cmap="magma", aspect="auto")
    axes[1].set(
        xticks=np.arange(len(conditions)),
        xticklabels=[DISPLAY_NAMES[condition] for condition in conditions],
        xlabel="Task state",
    )
    axes[1].tick_params(axis="x", labelrotation=35)
    axes[1].set_facecolor("#F1F3F5")
    midpoint = float(np.nanmedian(contribution))
    for row in range(len(cores)):
        for column in range(len(conditions)):
            if np.isfinite(contribution[row, column]):
                color = "white" if contribution[row, column] < midpoint else "black"
                axes[1].text(column, row, f"{contribution[row, column]:.2f}", ha="center", va="center", fontsize=6, color=color)
    colorbar = fig.colorbar(contribution_image, ax=axes[1], shrink=0.84, pad=0.02)
    colorbar.set_label("Mean greedy atom contribution when top (bits)")
    for label, axis in zip("ab", axes):
        axis.text(-0.12, 1.02, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    axes[0].text(0.0, -0.14, "* uncorrected 20-null rank p = 1/21", transform=axes[0].transAxes, fontsize=6.5)
    _save(fig, destination)


def plot_network_participation(summary: Mapping[str, Any], destination: Path) -> None:
    _style()
    conditions = list(summary["config"]["conditions"])
    participation = summary["cross_task"]["network_participation_fraction"]
    matrix = np.asarray([[participation[condition][network] for condition in conditions] for network in NETWORK_ORDER])
    fig, axis = plt.subplots(figsize=(6.3, 3.4), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set(
        xticks=np.arange(len(conditions)),
        xticklabels=[DISPLAY_NAMES[condition] for condition in conditions],
        yticks=np.arange(len(NETWORK_ORDER)),
        yticklabels=[NETWORK_ABBREVIATIONS[network] for network in NETWORK_ORDER],
        xlabel="Task state",
        ylabel="Yeo7 network",
    )
    axis.tick_params(axis="x", labelrotation=35)
    for row in range(len(NETWORK_ORDER)):
        for column in range(len(conditions)):
            color = "white" if matrix[row, column] >= 0.58 else "black"
            axis.text(column, row, f"{100 * matrix[row, column]:.0f}%", ha="center", va="center", fontsize=6.5, color=color)
    colorbar = fig.colorbar(image, ax=axis, shrink=0.86, pad=0.02)
    colorbar.set_label("Fraction of subject-level top-3 atoms containing network")
    _save(fig, destination)


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# HCP 七任务 Yeo7 协同核分解",
        "",
        "使用跨 REST 与七任务等权留出预测误差选择的共享模型 `p=5, alpha=10`。七任务的30名共同被试均独立重拟合 Yeo7-PC1、Δ-Ridge 与残差协方差；每个 Yeo7 网络的5个历史滞后绑定为不可拆模块。每名被试报告 greedy top-3，并对20个独立 circular-shift null 完整重拟合与重跑 greedy。",
        "",
        "核心重要性首先按进入被试 top-3 的频率排序，其次报告进入 top-3 时的平均 atom 贡献。null p 仅为20个配对 cohort 的未校正排名频率检验；最小值为1/21，不能解释为跨任务、跨核心多重校正后的显著性。",
    ]
    for task_result in summary["task_results"]:
        condition = task_result["condition"]
        comparison = {_core_key(item["sources"]): item for item in task_result["null_rank_comparison"]}
        lines.extend(
            [
                "",
                f"## {DISPLAY_NAMES[condition]}",
                "",
                "| 协同核 | top-3频率 | top时平均贡献（bits） | null频率均值；最大值 | null rank p |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for core in task_result["core_summary"][:8]:
            null = comparison[_core_key(core["sources"])]
            lines.append(
                f"| {' + '.join(core['sources'])} | {core['top_frequency']}/30 | "
                f"{core['mean_atom_value_when_top']:.6f} | {null['null_frequency_mean']:.2f}; "
                f"{null['null_frequency_max']} | {null['empirical_p']:.6f} |"
            )
        if task_result["quality_flags"]:
            lines.append(
                "\n质量规则标记：" + ", ".join(item["subject"] for item in task_result["quality_flags"]) + "；主分析保留。"
            )
    lines.extend(["", "## 跨任务摘要", ""])
    if summary["cross_task"]["shared_cores"]:
        lines.append("至少四个任务中达到15/30的共享高频核：")
        for core in summary["cross_task"]["shared_cores"]:
            consistency = core["subject_task_consistency"]
            lines.append(
                f"- {' + '.join(core['sources'])}；每名被试平均出现于 {consistency['mean_tasks_per_subject']:.2f}/7 个任务，"
                f"{consistency['n_subjects_all_tasks']}/30 出现于全部七任务，"
                f"{consistency['n_subjects_at_least_six_tasks']}/30 出现于至少六任务。"
            )
    else:
        lines.append("没有协同核在至少四个任务中同时达到15/30的预设共享高频阈值。")
    lines.extend(["", "达到15/30且获得最小未校正20-null排名 p=1/21 的任务-核心组合："])
    for item in summary["cross_task"]["exploratory_null_supported_cores"]:
        lines.append(
            f"- {DISPLAY_NAMES[item['condition']]}：{' + '.join(item['sources'])}，"
            f"{item['top_frequency']}/30，null均值 {item['null_frequency_mean']:.2f}/30。"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    data_root: Path,
    labels_path: Path,
    output_dir: Path,
    status_path: Path,
    *,
    tasks: Sequence[str] = TASKS,
    data_key: str = "Schaefer500_taskRetained",
    order: int = 5,
    alpha: float = 10.0,
    null_replicates: int = 20,
    seed: int = 20260720,
    top_k: int = 3,
    workers: int = 4,
) -> dict[str, Any]:
    discovered = discover_subject_task_files(data_root, tasks)
    if not discovered:
        raise FileNotFoundError(f"No subjects with complete task inputs below {data_root}.")
    subjects = sorted(discovered)
    groups = load_yeo7_groups(labels_path, expected_parcels=500)
    total = len(tasks) * len(subjects) * (int(null_replicates) + 1)
    completed = 0
    started = time.monotonic()
    task_results = []
    progress = tqdm(total=total, desc="all-task cores", unit="model", mininterval=1.0)

    def update(task: str, subject: str, increment: int) -> None:
        nonlocal completed
        completed += int(increment)
        progress.update(int(increment))
        if completed % 20 == 0 or completed >= total:
            elapsed = time.monotonic() - started
            _atomic_json(
                status_path,
                {
                    "phase": "running" if completed < total else "complete",
                    "current": completed,
                    "total": total,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": (total - completed) * elapsed / completed if completed else None,
                    "task": task,
                    "subject": subject,
                    "models_in_last_completed_subject": int(increment),
                },
            )

    try:
        for task in tasks:
            condition = task.removesuffix("_LR")
            rows = []
            payloads = [
                (
                    str(discovered[subject][task]),
                    groups,
                    subject,
                    data_key,
                    order,
                    alpha,
                    null_replicates,
                    seed,
                    top_k,
                )
                for subject in subjects
            ]
            if int(workers) <= 1:
                for payload in payloads:
                    row = _analyze_subject_file(payload)
                    rows.append(row)
                    update(condition, str(row["subject"]), int(null_replicates) + 1)
            else:
                with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                    future_by_subject = {
                        executor.submit(_analyze_subject_file, payload): str(payload[2]) for payload in payloads
                    }
                    for future in as_completed(future_by_subject):
                        row = future.result()
                        rows.append(row)
                        update(condition, str(row["subject"]), int(null_replicates) + 1)
            rows.sort(key=lambda row: str(row["subject"]))
            task_result = summarize_task(task, rows)
            task_results.append(task_result)
            _atomic_json(
                Path(output_dir) / "checkpoint.json",
                {
                    "config": {"order": order, "alpha": alpha, "null_replicates": null_replicates},
                    "completed_tasks": [result["condition"] for result in task_results],
                    "task_results": task_results,
                },
            )
    finally:
        progress.close()

    summary = {
        "config": {
            "tasks": list(tasks),
            "conditions": [task.removesuffix("_LR") for task in tasks],
            "subjects": subjects,
            "n_subjects": len(subjects),
            "data_key": data_key,
            "representation": "Yeo7 PC1 refitted per subject and task on the first 75% of time points",
            "model": "delta Ridge independently refitted per subject and task",
            "order": int(order),
            "alpha": float(alpha),
            "selection_rule": "lowest held-out delta-NRMSE pooled equally across REST, seven tasks, and 29 common subjects",
            "source_dimension": int(7 * order),
            "target_dimension": 7,
            "module_atoms": f"all {order} lags of one Yeo7 network remain inseparable",
            "estimator": "Gaussian log-det (affine Gaussian continuous-EI approximation)",
            "null": "independent non-zero circular shift of each Yeo7 PC1; model and greedy decomposition refitted",
            "null_replicates": int(null_replicates),
            "seed": int(seed),
            "top_k_per_subject": int(top_k),
            "workers": int(workers),
        },
        "task_results": task_results,
    }
    summary["cross_task"] = build_cross_task_summary(task_results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "summary.json", summary)
    write_report(summary, output_dir / "report.md")
    plot_core_landscape(summary, output_dir / "core_task_landscape")
    plot_network_participation(summary, output_dir / "network_participation")
    (output_dir / "checkpoint.json").unlink(missing_ok=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--labels", type=Path, default=default_yeo7_labels(500))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--data-key", default="Schaefer500_taskRetained")
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--null-replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tasks", default=",".join(TASKS))
    args = parser.parse_args(argv)
    summary = run(
        args.data_root,
        args.labels,
        args.output_dir,
        args.status,
        tasks=tuple(value.strip() for value in args.tasks.split(",") if value.strip()),
        data_key=args.data_key,
        order=args.order,
        alpha=args.alpha,
        null_replicates=args.null_replicates,
        seed=args.seed,
        top_k=args.top_k,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "shared_cores": summary["cross_task"]["shared_cores"],
                "exploratory_null_supported_cores": summary["cross_task"]["exploratory_null_supported_cores"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
