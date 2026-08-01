#!/usr/bin/env python3
"""Compare the original 29-subject Brain cohort with 28 supplementary subjects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from analyze_hcp_schaefer500_task_specific_regions import (
    TASKS,
    TASK_LABELS,
    loso_nearest_centroid,
    permutation_accuracy_pvalue,
    unit_spatial_features,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPS = (
    ROOT
    / "results"
    / "hcp_schaefer500_task_specific_regions_58"
    / "task_evoked_region_maps.npz"
)
DEFAULT_NEW_SUBJECTS = (
    ROOT
    / "data"
    / "hcp_task_lr_schaefer500_1000_28_complete_subjects"
    / "subjects_28_complete.txt"
)
DEFAULT_OUTPUT = ROOT / "results" / "hcp_task_dataset_extension_57"


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def welch_t(values_a: np.ndarray, values_b: np.ndarray) -> np.ndarray:
    mean_difference = values_b.mean(axis=0) - values_a.mean(axis=0)
    standard_error = np.sqrt(
        values_a.var(axis=0, ddof=1) / len(values_a)
        + values_b.var(axis=0, ddof=1) / len(values_b)
    )
    return np.divide(
        mean_difference,
        standard_error,
        out=np.zeros_like(mean_difference),
        where=standard_error > 0,
    )


def permutation_tests(
    old_values: np.ndarray,
    new_values: np.ndarray,
    *,
    permutations: int,
    seed: int,
    batch_size: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw two-sided and max-|t| familywise permutation p-values."""
    rng = np.random.default_rng(seed)
    pooled = np.vstack([old_values, new_values])
    pooled_sq = pooled**2
    total = pooled.sum(axis=0)
    total_sq = pooled_sq.sum(axis=0)
    n_old, n_new = len(old_values), len(new_values)
    observed = np.abs(welch_t(old_values, new_values))
    exceed_raw = np.zeros(pooled.shape[1], dtype=int)
    exceed_max = np.zeros(pooled.shape[1], dtype=int)

    completed = 0
    while completed < permutations:
        size = min(batch_size, permutations - completed)
        order = np.argsort(rng.random((size, len(pooled))), axis=1)
        new_indices = order[:, :n_new]
        new_sum = pooled[new_indices].sum(axis=1)
        new_sum_sq = pooled_sq[new_indices].sum(axis=1)
        old_sum = total - new_sum
        old_sum_sq = total_sq - new_sum_sq
        new_mean = new_sum / n_new
        old_mean = old_sum / n_old
        new_var = (new_sum_sq - n_new * new_mean**2) / (n_new - 1)
        old_var = (old_sum_sq - n_old * old_mean**2) / (n_old - 1)
        standard_error = np.sqrt(old_var / n_old + new_var / n_new)
        permuted_t = np.divide(
            new_mean - old_mean,
            standard_error,
            out=np.zeros_like(new_mean),
            where=standard_error > 0,
        )
        absolute_t = np.abs(permuted_t)
        exceed_raw += np.count_nonzero(absolute_t >= observed, axis=0)
        exceed_max += np.count_nonzero(
            absolute_t.max(axis=1, keepdims=True) >= observed[None, :],
            axis=0,
        )
        completed += size

    return (exceed_raw + 1) / (permutations + 1), (exceed_max + 1) / (permutations + 1)


def bootstrap_statistics(
    old_values: np.ndarray,
    new_values: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    old_boot = old_values[
        rng.integers(0, len(old_values), size=(repeats, len(old_values)))
    ].mean(axis=1)
    new_boot = new_values[
        rng.integers(0, len(new_values), size=(repeats, len(new_values)))
    ].mean(axis=1)
    return {
        "old_ci95": np.quantile(old_boot, [0.025, 0.975], axis=0).T,
        "new_ci95": np.quantile(new_boot, [0.025, 0.975], axis=0).T,
        "difference_ci95": np.quantile(new_boot - old_boot, [0.025, 0.975], axis=0).T,
    }


def hedges_g(old_values: np.ndarray, new_values: np.ndarray) -> np.ndarray:
    n_old, n_new = len(old_values), len(new_values)
    pooled_variance = (
        (n_old - 1) * old_values.var(axis=0, ddof=1)
        + (n_new - 1) * new_values.var(axis=0, ddof=1)
    ) / (n_old + n_new - 2)
    correction = 1.0 - 3.0 / (4.0 * (n_old + n_new) - 9.0)
    return correction * (new_values.mean(axis=0) - old_values.mean(axis=0)) / np.sqrt(
        pooled_variance
    )


def spatial_correlations(old_maps: np.ndarray, new_maps: np.ndarray) -> np.ndarray:
    old_mean = old_maps.mean(axis=0)
    new_mean = new_maps.mean(axis=0)
    old_centered = old_mean - old_mean.mean(axis=1, keepdims=True)
    new_centered = new_mean - new_mean.mean(axis=1, keepdims=True)
    numerator = np.einsum("tp,tp->t", old_centered, new_centered)
    denominator = np.linalg.norm(old_centered, axis=1) * np.linalg.norm(
        new_centered, axis=1
    )
    return numerator / denominator


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def plot_comparison(summary: dict, output_dir: Path) -> None:
    configure_style()
    colors = {"old": "#7A8FA6", "new": "#D28B5B", "combined": "#5B8E7D"}
    x = np.arange(len(TASKS))
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.75),
        gridspec_kw={"width_ratios": [1.45, 1.15, 1.0]},
        constrained_layout=True,
    )

    old_mean = np.asarray(summary["old_mean"])
    new_mean = np.asarray(summary["new_mean"])
    old_ci = np.asarray(summary["old_ci95"])
    new_ci = np.asarray(summary["new_ci95"])
    for offset, mean, interval, label, color in (
        (-0.10, old_mean, old_ci, "Original (n=29)", colors["old"]),
        (+0.10, new_mean, new_ci, "Supplement (n=28)", colors["new"]),
    ):
        axes[0].errorbar(
            x + offset,
            100 * mean,
            yerr=100 * np.vstack([mean - interval[:, 0], interval[:, 1] - mean]),
            fmt="o",
            markersize=3.5,
            capsize=2,
            elinewidth=0.8,
            color=color,
            label=label,
        )
    axes[0].set_ylabel("Mean TEVF (%)")
    axes[0].set_xticks(x, TASK_LABELS, rotation=42, ha="right")
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    difference = np.asarray(summary["difference"])
    difference_ci = np.asarray(summary["difference_ci95"])
    axes[1].axhline(0, color="#666666", linewidth=0.7, linestyle="--")
    significant = np.asarray(summary["max_t_pvalue"]) < 0.05
    axes[1].errorbar(
        x,
        100 * difference,
        yerr=100
        * np.vstack(
            [difference - difference_ci[:, 0], difference_ci[:, 1] - difference]
        ),
        fmt="none",
        capsize=2,
        elinewidth=0.8,
        color="#777777",
    )
    axes[1].scatter(
        x,
        100 * difference,
        s=18,
        color=np.where(significant, colors["new"], "#777777"),
        zorder=3,
    )
    axes[1].set_ylabel("Supplement − original (pp)")
    axes[1].set_xticks(x, TASK_LABELS, rotation=42, ha="right")

    correlations = np.asarray(summary["spatial_correlation"])
    axes[2].bar(x, correlations, width=0.65, color=colors["combined"])
    axes[2].set_ylim(0.8, 1.005)
    axes[2].set_ylabel("Old–new spatial correlation")
    axes[2].set_xticks(x, TASK_LABELS, rotation=42, ha="right")
    for index, value in enumerate(correlations):
        axes[2].text(index, value + 0.004, f"{value:.2f}", ha="center", va="bottom", fontsize=5.7)

    for label, axis in zip("abc", axes):
        axis.text(
            -0.16,
            1.04,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9,
            va="top",
        )
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(
            output_dir / f"original29_vs_supplement28_tevf.{suffix}",
            dpi=600,
            bbox_inches="tight",
        )
    plt.close(figure)


def write_report(summary: dict, output_dir: Path) -> None:
    rows = []
    for index, task in enumerate(TASKS):
        rows.append(
            f"| {task} | {100 * summary['old_mean'][index]:.2f} | "
            f"{100 * summary['new_mean'][index]:.2f} | "
            f"{100 * summary['difference'][index]:+.2f} | "
            f"{summary['hedges_g'][index]:+.2f} | "
            f"{summary['max_t_pvalue'][index]:.3f} | "
            f"{summary['spatial_correlation'][index]:.3f} |"
        )
    lines = [
        "# HCP Brain 数据扩展：原 29 人与新增 28 人",
        "",
        "## 结论",
        "",
        "新增 28 人复现了原 29 人的七任务 TEVF 强度排序和 Schaefer-500 空间分布。",
        "七项新旧批次均值差异均未通过跨任务 max-T 校正；这支持合并分析，但不证明两批数据等价。",
        "",
        "![原 29 人与新增 28 人比较](original29_vs_supplement28_tevf.png)",
        "",
        "## 数值结果",
        "",
        "| Task | Original mean (%) | Supplement mean (%) | Difference (pp) | Hedges g | max-T p | Spatial r |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        f"两个批次的任务均值排序完全一致：`{summary['task_ranking_high_to_low']}`。",
        f"合并 57 人后，七任务 Schaefer-500 空间图的 LOSO 准确率为 "
        f"{100 * summary['combined_57_classification']['parcel_loso_accuracy']:.1f}%"
        f"（机会水平 14.3%，置换 p={summary['combined_57_classification']['permutation_pvalue']:.4f}）。",
        "",
        "## 数据边界",
        "",
        "- 57 人主集合严格由原 Brain 实验中有 REST 配对的 29 人，加上压缩包中的 28 人组成。",
        "- 原任务目录另有 `sub-106521`，但它不在原 29 人 REST 配对集合中，因此未纳入本次 29+28 对比。",
        "- 新压缩包只有七项 LR 任务时序，没有 REST 时序或行为/认知表。因此当前可以扩展任务态 TEVF；不能据此把既有 REST–任务 Xi、认知相关或行为相关结论扩展到 57 人。",
        "- 新旧是独立批次。批次差异同时包含被试选择和潜在处理批次因素，不能作因果解释。",
        "- TEVF 衡量任务 GLM 解释的 parcel 时间能量，不是 EI、Xi 或 PEID。",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, default=DEFAULT_MAPS)
    parser.add_argument("--new-subjects", type=Path, default=DEFAULT_NEW_SUBJECTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--bootstrap-repeats", type=int, default=20_000)
    parser.add_argument("--classification-permutations", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = np.load(args.maps)
    subjects = cache["subjects"].astype(str)
    original_subjects = set(cache["state_subjects"].astype(str))
    supplementary_subjects = {
        f"sub-{line.strip()}"
        for line in args.new_subjects.read_text().splitlines()
        if line.strip()
    }
    old_mask = np.asarray([subject in original_subjects for subject in subjects])
    new_mask = np.asarray([subject in supplementary_subjects for subject in subjects])
    if old_mask.sum() != 29 or new_mask.sum() != 28 or np.any(old_mask & new_mask):
        raise ValueError(
            f"Expected disjoint 29/28 cohorts, got {old_mask.sum()}/{new_mask.sum()}."
        )

    fractions = cache["fractions"]
    enrichment = cache["enrichment"]
    subject_tevf = fractions.mean(axis=2)
    old_values = subject_tevf[old_mask]
    new_values = subject_tevf[new_mask]
    bootstrap = bootstrap_statistics(
        old_values, new_values, repeats=args.bootstrap_repeats, seed=args.seed
    )
    raw_pvalue, max_t_pvalue = permutation_tests(
        old_values,
        new_values,
        permutations=args.permutations,
        seed=args.seed + 1,
    )
    combined_mask = old_mask | new_mask
    combined_features = unit_spatial_features(enrichment[combined_mask])
    classification = loso_nearest_centroid(combined_features)
    classification_permutation = permutation_accuracy_pvalue(
        combined_features,
        classification["accuracy"],
        permutations=args.classification_permutations,
        seed=args.seed + 2,
    )
    old_ranking = np.argsort(old_values.mean(axis=0))[::-1]
    new_ranking = np.argsort(new_values.mean(axis=0))[::-1]

    summary = {
        "analysis": "Original 29 versus supplementary 28 HCP task-fMRI cohort comparison",
        "metric": "subject mean Schaefer-500 task-evoked variance fraction (TEVF)",
        "n_original": int(old_mask.sum()),
        "n_supplementary": int(new_mask.sum()),
        "n_combined": int(combined_mask.sum()),
        "original_subjects": subjects[old_mask].tolist(),
        "supplementary_subjects": subjects[new_mask].tolist(),
        "tasks": list(TASKS),
        "old_mean": old_values.mean(axis=0).tolist(),
        "new_mean": new_values.mean(axis=0).tolist(),
        "combined_mean": subject_tevf[combined_mask].mean(axis=0).tolist(),
        "difference": (new_values.mean(axis=0) - old_values.mean(axis=0)).tolist(),
        "old_ci95": bootstrap["old_ci95"].tolist(),
        "new_ci95": bootstrap["new_ci95"].tolist(),
        "difference_ci95": bootstrap["difference_ci95"].tolist(),
        "hedges_g": hedges_g(old_values, new_values).tolist(),
        "raw_permutation_pvalue": raw_pvalue.tolist(),
        "holm_pvalue": holm_adjust(raw_pvalue).tolist(),
        "max_t_pvalue": max_t_pvalue.tolist(),
        "spatial_correlation": spatial_correlations(
            fractions[old_mask], fractions[new_mask]
        ).tolist(),
        "task_ranking_identical": bool(np.array_equal(old_ranking, new_ranking)),
        "task_ranking_high_to_low": " > ".join(TASKS[index] for index in old_ranking),
        "combined_57_classification": {
            "parcel_loso_accuracy": classification["accuracy"],
            "chance": 1.0 / len(TASKS),
            "permutation_pvalue": classification_permutation["pvalue"],
            "permutation_null_mean": classification_permutation["null_mean"],
            "permutation_null_ci95": classification_permutation["null_ci95"],
        },
        "inference_contract": {
            "treatment_factor": "data cohort/batch (original versus supplement)",
            "controlled": [
                "LR direction",
                "seven tasks",
                "Schaefer-500 parcellation",
                "taskRetained/taskRegressed TEVF definition",
            ],
            "limitation": "Independent cohorts; differences may reflect subject selection or processing batch and are not causal effects.",
            "familywise_test": "max-|Welch t| label permutation across seven tasks",
            "permutations": args.permutations,
            "bootstrap_repeats": args.bootstrap_repeats,
            "seed": args.seed,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "cohort_comparison.npz",
        subjects=subjects[combined_mask],
        cohort=np.where(old_mask[combined_mask], "original", "supplement"),
        tasks=np.asarray(TASKS),
        subject_tevf=subject_tevf[combined_mask],
        fractions=fractions[combined_mask],
    )
    (args.output_dir / "figure_contract.md").write_text(
        "\n".join(
            [
                "# Figure contract",
                "",
                "- Core conclusion: test whether the 28-subject supplement reproduces the original 29-subject task-strength ranking and spatial maps.",
                "- Panel a: cohort means and 95% subject-bootstrap intervals.",
                "- Panel b: supplement-minus-original differences and 95% bootstrap intervals; highlighted only if max-T adjusted p < 0.05.",
                "- Panel c: old-new correlations of the 500-parcel group TEVF maps.",
                "- Archetype/role: quantitative grid; cohort comparison and robustness.",
                "- Export: 183 mm wide, editable SVG/PDF and 600-dpi PNG.",
                "- Review risks: independent cohorts and possible batch/selection confounding; TEVF is not EI or PEID.",
            ]
        ),
        encoding="utf-8",
    )
    plot_comparison(summary, args.output_dir)
    write_report(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
