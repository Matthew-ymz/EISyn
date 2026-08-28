#!/usr/bin/env python3
"""Strict REST-length control for the 57-subject Schaefer-1000 main analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# Small per-network SVDs are much faster without nested BLAS threading; subject-level
# parallelism is controlled explicitly below.
for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import wilcoxon
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_task_evoked_pc2_xi_hierarchy import (
    decompose_transition,
    fit_project_network_pca,
)
from scripts.analyze_hcp_schaefer500_yeo7_network_attribution import NETWORK_ORDER
from scripts.run_hcp_schaefer500_all_tasks_phi import development_end_for_length
from scripts.run_hcp_schaefer500_length_matched_variance import evenly_spaced_window_starts
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import load_yeo7_groups
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi


CONDITIONS = ("EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")
DISPLAY_NAMES = {
    "EMOTION": "Emotion",
    "GAMBLING": "Gambling",
    "LANGUAGE": "Language",
    "MOTOR": "Motor",
    "RELATIONAL": "Relational",
    "SOCIAL": "Social",
    "WM": "WM",
}
TASK_LENGTHS = {
    "EMOTION": 176,
    "GAMBLING": 253,
    "LANGUAGE": 316,
    "MOTOR": 284,
    "RELATIONAL": 232,
    "SOCIAL": 274,
    "WM": 405,
}
DEFAULT_REST_ROOT = (
    ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_57_brain"
)
DEFAULT_LABELS = (
    ROOT
    / "data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30"
    / "_atlas_labels/Schaefer2018_1000Parcels_7Networks_order.txt"
)
DEFAULT_MAIN = ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/full/k1_p3_a1/arrays.npz"
DEFAULT_RECORDS = ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/full/records.jsonl"
DEFAULT_OUTPUT = ROOT / "results/hcp_schaefer1000_length_matched_rest_57"
SYN_TOLERANCE_BITS = 1.0e-9


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def update_progress(
    output_dir: Path,
    *,
    state: str,
    completed_subjects: int,
    total_subjects: int,
    detail: str,
) -> None:
    atomic_json(
        output_dir / "live_progress.json",
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": state,
            "completed_subjects": int(completed_subjects),
            "total_subjects": int(total_subjects),
            "completed_fits": int(completed_subjects * len(CONDITIONS) * 12),
            "total_fits": int(total_subjects * len(CONDITIONS) * 12),
            "detail": detail,
        },
    )


def bh_adjust(values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def discover_rest_paths(rest_root: Path) -> dict[str, Path]:
    paths = {
        path.parent.name: path
        for path in Path(rest_root).glob("sub-*/*REST1_LR*schaefer500-1000_yeo7.mat")
    }
    return dict(sorted(paths.items()))


def load_main_reference(
    arrays_path: Path, records_path: Path
) -> tuple[list[str], dict[str, np.ndarray]]:
    archive = np.load(arrays_path)
    subjects = archive["subjects"].astype(str).tolist()
    states = archive["states"].astype(str).tolist()
    system_xi = np.asarray(archive["system_xi"], dtype=float)
    if system_xi.shape != (8, 57) or states != ["REST", *CONDITIONS]:
        raise ValueError("Main Schaefer-1000 cache does not match the frozen 57-subject analysis.")
    task_values = {
        condition: system_xi[states.index(condition)].copy() for condition in CONDITIONS
    }
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    observed_lengths = {
        condition: sorted(
            {
                int(row["n_timepoints"])
                for row in rows
                if row["state"] == condition and row["subject"] in set(subjects)
            }
        )
        for condition in CONDITIONS
    }
    expected = {condition: [TASK_LENGTHS[condition]] for condition in CONDITIONS}
    if observed_lengths != expected:
        raise ValueError(f"Task lengths differ from the frozen main analysis: {observed_lengths}")
    return subjects, task_values


def fit_window_xi(
    window: np.ndarray,
    groups: Mapping[str, Sequence[int]],
) -> tuple[float, float, float, float, float]:
    development_end = development_end_for_length(len(window))
    reduced, explained = fit_project_network_pca(
        window,
        window,
        groups,
        development_end=development_end,
        n_components=1,
    )
    fitted = fit_delta_history_phi(
        reduced,
        alpha=1.0,
        order=3,
        development_end=development_end,
    )
    decomposition = decompose_transition(
        fitted["transition"],
        fitted["noise_covariance"],
        tuple(NETWORK_ORDER),
        n_components=1,
        order=3,
    )
    value = float(decomposition["system_xi"])
    if value < -SYN_TOLERANCE_BITS:
        raise RuntimeError(
            f"Significant Syn nonnegativity violation: minimum={value:.12g} bits, "
            f"tolerance={SYN_TOLERANCE_BITS:.1e} bits, affected_count=1."
        )
    development = reduced[:development_end]
    scale = development.std(axis=0, ddof=1)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    max_z = float(np.max(np.abs((development - development.mean(axis=0)) / scale)))
    condition = float(np.linalg.cond(fitted["noise_covariance"]))
    explained_mean = float(np.mean([values[0] for values in explained.values()]))
    return value, float(fitted["heldout"]["skill_ratio"]), max_z, condition, explained_mean


def fit_subject(
    subject: str,
    rest_path: str,
    labels_path: str,
    starts: tuple[tuple[int, ...], ...],
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = loadmat(rest_path)
    if "Schaefer1000" not in payload:
        raise ValueError(f"{rest_path} lacks Schaefer1000")
    raw = np.asarray(payload["Schaefer1000"], dtype=float)
    if raw.shape != (1200, 1000) or not np.isfinite(raw).all():
        raise ValueError(f"Expected finite REST [1200,1000], got {raw.shape} for {subject}")
    groups = load_yeo7_groups(Path(labels_path), expected_parcels=1000)
    shape = (len(CONDITIONS), len(starts[0]))
    xi = np.empty(shape, dtype=float)
    skill = np.empty(shape, dtype=float)
    max_z = np.empty(shape, dtype=float)
    noise_condition = np.empty(shape, dtype=float)
    explained = np.empty(shape, dtype=float)
    for condition_index, condition in enumerate(CONDITIONS):
        length = TASK_LENGTHS[condition]
        for window_index, start in enumerate(starts[condition_index]):
            metrics = fit_window_xi(raw[start : start + length], groups)
            xi[condition_index, window_index] = metrics[0]
            skill[condition_index, window_index] = metrics[1]
            max_z[condition_index, window_index] = metrics[2]
            noise_condition[condition_index, window_index] = metrics[3]
            explained[condition_index, window_index] = metrics[4]
    return subject, xi, skill, max_z, noise_condition, explained


def checkpoint_metadata(
    subjects: Sequence[str], rest_paths: Mapping[str, Path], starts: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    return {
        "subjects": list(subjects),
        "conditions": list(CONDITIONS),
        "task_lengths": TASK_LENGTHS,
        "window_starts": {key: value.tolist() for key, value in starts.items()},
        "rest_paths": {key: str(value.resolve()) for key, value in rest_paths.items()},
        "parcellation": "Schaefer-1000 / Yeo-7",
        "components_per_network": 1,
        "history_order": 3,
        "ridge_alpha": 1.0,
        "development_fraction": 0.75,
        "estimator": "affine Gaussian TM system-level Xi",
        "syn_nonnegative_tolerance_bits": SYN_TOLERANCE_BITS,
    }


def save_checkpoint(
    path: Path,
    metadata: Mapping[str, Any],
    xi: np.ndarray,
    skill: np.ndarray,
    max_z: np.ndarray,
    noise_condition: np.ndarray,
    explained: np.ndarray,
) -> None:
    temporary = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(
        temporary,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        xi=xi,
        heldout_skill_ratio=skill,
        max_abs_development_pc_zscore=max_z,
        noise_covariance_condition=noise_condition,
        mean_pc1_explained=explained,
    )
    os.replace(temporary, path)


def load_checkpoint(
    path: Path, metadata: Mapping[str, Any], shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = [np.full(shape, np.nan, dtype=float) for _ in range(5)]
    if not path.is_file():
        return tuple(arrays)  # type: ignore[return-value]
    archive = np.load(path)
    saved_metadata = json.loads(str(archive["metadata"].item()))
    if saved_metadata != metadata:
        raise ValueError("Existing checkpoint metadata differs from the strict experiment contract.")
    keys = (
        "xi",
        "heldout_skill_ratio",
        "max_abs_development_pc_zscore",
        "noise_covariance_condition",
        "mean_pc1_explained",
    )
    loaded = tuple(np.asarray(archive[key], dtype=float) for key in keys)
    if any(array.shape != shape for array in loaded):
        raise ValueError("Existing checkpoint shape differs from the strict experiment contract.")
    return loaded  # type: ignore[return-value]


def run_fits(
    subjects: Sequence[str],
    rest_paths: Mapping[str, Path],
    labels_path: Path,
    starts: Mapping[str, np.ndarray],
    output_dir: Path,
    *,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    checkpoint_path = output_dir / "rest_window_xi.npz"
    metadata = checkpoint_metadata(subjects, rest_paths, starts)
    shape = (len(subjects), len(CONDITIONS), len(next(iter(starts.values()))))
    xi, skill, max_z, noise_condition, explained = load_checkpoint(
        checkpoint_path, metadata, shape
    )
    completed = np.all(np.isfinite(xi), axis=(1, 2))
    pending = np.flatnonzero(~completed).tolist()
    update_progress(
        output_dir,
        state="running" if pending else "cached",
        completed_subjects=int(completed.sum()),
        total_subjects=len(subjects),
        detail="strict window-local PCA and dynamics fitting",
    )
    starts_tuple = tuple(tuple(int(value) for value in starts[c]) for c in CONDITIONS)
    subject_to_index = {subject: index for index, subject in enumerate(subjects)}
    with tqdm(total=len(subjects), initial=int(completed.sum()), desc="Schaefer-1000 length control", unit="subject") as progress:
        if workers == 1:
            results = (
                fit_subject(subjects[index], str(rest_paths[subjects[index]]), str(labels_path), starts_tuple)
                for index in pending
            )
            for result in results:
                subject, a, b, c, d, e = result
                index = subject_to_index[subject]
                xi[index], skill[index], max_z[index], noise_condition[index], explained[index] = a, b, c, d, e
                save_checkpoint(checkpoint_path, metadata, xi, skill, max_z, noise_condition, explained)
                completed[index] = True
                progress.update(1)
                update_progress(output_dir, state="running", completed_subjects=int(completed.sum()), total_subjects=len(subjects), detail=f"completed {subject}")
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        fit_subject,
                        subjects[index],
                        str(rest_paths[subjects[index]]),
                        str(labels_path),
                        starts_tuple,
                    ): subjects[index]
                    for index in pending
                }
                for future in as_completed(futures):
                    subject, a, b, c, d, e = future.result()
                    index = subject_to_index[subject]
                    xi[index], skill[index], max_z[index], noise_condition[index], explained[index] = a, b, c, d, e
                    save_checkpoint(checkpoint_path, metadata, xi, skill, max_z, noise_condition, explained)
                    completed[index] = True
                    progress.update(1)
                    update_progress(output_dir, state="running", completed_subjects=int(completed.sum()), total_subjects=len(subjects), detail=f"completed {subject}")
    if not all(np.isfinite(array).all() for array in (xi, skill, max_z, noise_condition, explained)):
        raise RuntimeError("Strict length-control cache remains incomplete.")
    return xi, skill, max_z, noise_condition, explained


def summarize(
    subjects: Sequence[str],
    task_values: Mapping[str, np.ndarray],
    xi: np.ndarray,
    skill: np.ndarray,
    max_z: np.ndarray,
    noise_condition: np.ndarray,
    explained: np.ndarray,
    starts: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    p_values = []
    for condition_index, condition in enumerate(CONDITIONS):
        rest_windows = xi[:, condition_index, :]
        matched_rest = rest_windows.mean(axis=1)
        task = np.asarray(task_values[condition], dtype=float)
        difference = matched_rest - task
        result = wilcoxon(difference)
        window_margins = rest_windows.mean(axis=0) - task.mean()
        window_p = np.asarray([wilcoxon(rest_windows[:, j] - task).pvalue for j in range(rest_windows.shape[1])])
        row = {
            "condition": condition,
            "task_length": TASK_LENGTHS[condition],
            "window_starts": starts[condition].tolist(),
            "matched_rest_mean_bits": float(matched_rest.mean()),
            "matched_rest_sd_bits": float(matched_rest.std(ddof=1)),
            "task_mean_bits": float(task.mean()),
            "task_sd_bits": float(task.std(ddof=1)),
            "rest_minus_task_mean_bits": float(difference.mean()),
            "rest_greater_fraction": float(np.mean(difference > 0.0)),
            "paired_wilcoxon_p": float(result.pvalue),
            "per_window_mean_margin_bits": window_margins.tolist(),
            "minimum_window_mean_margin_bits": float(window_margins.min()),
            "maximum_window_mean_margin_bits": float(window_margins.max()),
            "windows_with_positive_mean_margin": int(np.sum(window_margins > 0.0)),
            "per_window_p": window_p.tolist(),
            "windows_nominally_significant": int(np.sum(window_p < 0.05)),
            "paired_subject_values": [
                {
                    "subject": subject,
                    "matched_rest_xi": float(matched_rest[index]),
                    "task_xi": float(task[index]),
                }
                for index, subject in enumerate(subjects)
            ],
        }
        rows.append(row)
        p_values.append(float(result.pvalue))
    for row, q_value in zip(rows, bh_adjust(p_values), strict=True):
        row["paired_wilcoxon_bh_q"] = float(q_value)
    tolerance_count = int(np.sum((xi < 0.0) & (xi >= -SYN_TOLERANCE_BITS)))
    summary = {
        "experiment": "Schaefer-1000 57-subject strict REST sequence-length control",
        "config": {
            "n_subjects": len(subjects),
            "n_windows_per_task_length": xi.shape[2],
            "n_rest_fits": int(xi.size),
            "parcellation": "Schaefer-1000 / Yeo-7",
            "components_per_network": 1,
            "history_order": 3,
            "ridge_alpha": 1.0,
            "development_fraction": 0.75,
            "estimator": "affine Gaussian TM system-level Xi",
            "task_reference": "frozen values from the 57-subject main analysis",
            "syn_nonnegative_tolerance_bits": SYN_TOLERANCE_BITS,
        },
        "condition_results": rows,
        "quality_diagnostics": {
            "minimum_xi_bits": float(xi.min()),
            "negative_within_tolerance_count": tolerance_count,
            "significant_nonnegativity_violation_count": int(np.sum(xi < -SYN_TOLERANCE_BITS)),
            "mean_heldout_skill_ratio": float(skill.mean()),
            "models_better_than_persistence": int(np.sum(skill < 1.0)),
            "n_models": int(skill.size),
            "maximum_abs_development_pc_zscore": float(max_z.max()),
            "maximum_noise_covariance_condition": float(noise_condition.max()),
            "mean_pc1_explained_variance": float(explained.mean()),
        },
        "conclusion_checks": {
            "all_seven_mean_margins_positive": bool(all(row["rest_minus_task_mean_bits"] > 0.0 for row in rows)),
            "all_seven_bh_significant": bool(all(row["paired_wilcoxon_bh_q"] < 0.05 for row in rows)),
            "all_84_window_mean_margins_positive": bool(all(row["windows_with_positive_mean_margin"] == 12 for row in rows)),
        },
    }
    return summary


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def significance_label(q_value: float) -> str:
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return "ns"


def plot(summary: Mapping[str, Any], output: Path) -> None:
    """Quantitative-grid robustness figure; one PNG is the manuscript asset."""
    configure_style()
    rest_color = "#4C78A8"
    task_color = "#D28E5B"
    line_color = "#B8C0CA"
    rows = summary["condition_results"]
    all_values = [
        value
        for row in rows
        for pair in row["paired_subject_values"]
        for value in (pair["matched_rest_xi"], pair["task_xi"])
    ]
    lower = np.floor((min(all_values) - 0.3) * 2.0) / 2.0
    upper = np.ceil((max(all_values) + 0.3) * 2.0) / 2.0
    fig, axes_grid = plt.subplots(2, 4, figsize=(10.8, 5.8), sharey=True, constrained_layout=True)
    axes = axes_grid.flat
    for index, (axis, row) in enumerate(zip(axes, rows, strict=False)):
        rest = np.asarray([pair["matched_rest_xi"] for pair in row["paired_subject_values"]])
        task = np.asarray([pair["task_xi"] for pair in row["paired_subject_values"]])
        rng = np.random.default_rng(2026082800 + index)
        jitter = rng.uniform(-0.065, 0.065, len(rest))
        for offset, rest_value, task_value in zip(jitter, rest, task, strict=True):
            axis.plot([offset, 1.0 + offset], [rest_value, task_value], color=line_color, lw=0.45, alpha=0.55, zorder=1)
        boxes = axis.boxplot(
            [rest, task],
            positions=[0, 1],
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#252525", "linewidth": 1.0},
            whiskerprops={"color": "#6E7781", "linewidth": 0.7},
            capprops={"color": "#6E7781", "linewidth": 0.7},
        )
        for box, color in zip(boxes["boxes"], (rest_color, task_color), strict=True):
            box.set(facecolor=color, edgecolor=color, alpha=0.18, linewidth=0.9)
        axis.scatter(jitter, rest, s=10, color=rest_color, alpha=0.78, linewidths=0, zorder=3)
        axis.scatter(1.0 + jitter, task, s=10, color=task_color, alpha=0.78, linewidths=0, zorder=3)
        axis.scatter([0, 1], [rest.mean(), task.mean()], marker="D", s=24, facecolor="white", edgecolor="#303030", linewidth=0.8, zorder=4)
        axis.set(
            xticks=[0, 1],
            xticklabels=["REST\nmatched", DISPLAY_NAMES[row["condition"]]],
            xlim=(-0.42, 1.42),
            ylim=(lower, upper),
        )
        axis.text(-0.34, upper - 0.08 * (upper - lower), f"{row['task_length']} frames", fontsize=6.2)
        axis.text(
            0.5,
            upper - 0.08 * (upper - lower),
            significance_label(row["paired_wilcoxon_bh_q"]),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
        axis.text(
            0.5,
            lower + 0.04 * (upper - lower),
            f"Δ={row['rest_minus_task_mean_bits']:+.2f} bits",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="#424A53",
        )
    axes[0].set_ylabel(r"System-level $\Xi$ (bits)")
    axes[4].set_ylabel(r"System-level $\Xi$ (bits)")
    legend_axis = axes[7]
    legend_axis.axis("off")
    legend_axis.scatter([], [], s=18, color=rest_color, label="Length-matched REST")
    legend_axis.scatter([], [], s=18, color=task_color, label="Task")
    legend_axis.plot([], [], color=line_color, lw=0.7, label="Same subject")
    legend_axis.scatter([], [], marker="D", s=24, facecolor="white", edgecolor="#303030", label="Mean")
    legend_axis.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), borderaxespad=0.0)
    legend_axis.text(
        0.02,
        0.50,
        "n = 57; 12 REST windows per task length\n"
        "Window-local PCA and dynamics refitting\n"
        "Two-sided paired Wilcoxon; BH across 7 tasks",
        transform=legend_axis.transAxes,
        va="top",
        linespacing=1.45,
        fontsize=6.4,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Schaefer-1000 57 人 REST 序列长度严格控制实验",
        "",
        "## 设计",
        "",
        "固定正文主分析的 Schaefer-1000/Yeo7、每网络 PC1、三阶历史、Ridge alpha=1 与 affine Gaussian TM Xi。唯一改变是把 REST1_LR 截成与各任务完全相同的长度。每个任务长度使用 12 个覆盖完整 REST run 的等距窗口；每个窗口均在自身前 75% 内重新拟合 PCA 和动力学。任务 Xi 直接复用 57 人正文主分析的冻结结果。",
        "",
        "## 结果",
        "",
        "| Task | Frames | Matched REST | Task | REST-task | REST>task | Wilcoxon p | BH q | Positive windows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["condition_results"]:
        lines.append(
            f"| {row['condition']} | {row['task_length']} | {row['matched_rest_mean_bits']:.3f} | "
            f"{row['task_mean_bits']:.3f} | {row['rest_minus_task_mean_bits']:+.3f} | "
            f"{row['rest_greater_fraction']:.1%} | {row['paired_wilcoxon_p']:.3g} | "
            f"{row['paired_wilcoxon_bh_q']:.3g} | {row['windows_with_positive_mean_margin']}/12 |"
        )
    checks = summary["conclusion_checks"]
    quality = summary["quality_diagnostics"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"七项平均差均为正：{checks['all_seven_mean_margins_positive']}；七项 BH 校正后均显著：{checks['all_seven_bh_significant']}；84 个具体窗口的群体均值差均为正：{checks['all_84_window_mean_margins_positive']}。",
            "",
            f"共拟合 {quality['n_models']} 个 REST 窗口模型，其中 {quality['models_better_than_persistence']} 个留出预测优于持久性基线。Syn 非负容差为 {SYN_TOLERANCE_BITS:.1e} bits；容差内负值 {quality['negative_within_tolerance_count']} 个，显著违反 {quality['significant_nonnegativity_violation_count']} 个。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rest-root", type=Path, default=DEFAULT_REST_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--main-arrays", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--main-records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-subjects", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.windows != 12:
        raise ValueError("The strict preregistered design uses exactly 12 REST windows per task length.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects, task_values = load_main_reference(args.main_arrays, args.main_records)
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]
        task_values = {condition: values[: args.max_subjects] for condition, values in task_values.items()}
    rest_paths = discover_rest_paths(args.rest_root)
    missing = sorted(set(subjects) - set(rest_paths))
    if missing:
        raise FileNotFoundError(f"Missing REST data for {missing}")
    rest_paths = {subject: rest_paths[subject] for subject in subjects}
    starts = {
        condition: evenly_spaced_window_starts(1200, TASK_LENGTHS[condition], args.windows)
        for condition in CONDITIONS
    }
    contract = {
        "scientific_question": "Does the 57-subject Schaefer-1000 REST>task Xi conclusion survive exact sequence-length matching?",
        "treatment_factor": "REST sequence length",
        "treatment_levels": TASK_LENGTHS,
        "pairing_unit": "HCP subject",
        "primary_metric": "subject-level mean across 12 matched REST windows minus frozen task Xi",
        "statistics": "two-sided paired Wilcoxon; Benjamini-Hochberg across seven tasks",
        "frozen": checkpoint_metadata(subjects, rest_paths, starts),
        "figure_contract": {
            "core_conclusion": "Exact length matching does or does not preserve the REST>task system-level Xi result.",
            "evidence_chain": "seven paired panels; margins; BH inference; 12-window sensitivity",
            "archetype": "quantitative grid",
            "role": "robustness",
            "backend": "Python",
            "output": "one manuscript-ready PNG",
            "review_risks": ["window-position dependence", "paired-subject visibility", "annotation overlap"],
        },
    }
    atomic_json(args.output_dir / "experiment_contract.json", contract)
    try:
        arrays = run_fits(
            subjects,
            rest_paths,
            args.labels,
            starts,
            args.output_dir,
            workers=args.workers,
        )
        summary = summarize(subjects, task_values, *arrays, starts)
        atomic_json(args.output_dir / "summary.json", summary)
        plot(summary, args.output_dir / "length_matched_rest_task_xi_57.png")
        write_report(summary, args.output_dir / "report.md")
        update_progress(args.output_dir, state="complete", completed_subjects=len(subjects), total_subjects=len(subjects), detail="summary, report, and figure complete")
    except Exception as error:
        update_progress(args.output_dir, state="failed", completed_subjects=0, total_subjects=len(subjects), detail=f"{type(error).__name__}: {error}")
        raise
    print(json.dumps(summary["conclusion_checks"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
