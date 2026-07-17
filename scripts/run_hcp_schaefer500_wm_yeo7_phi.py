#!/usr/bin/env python3
"""Run Schaefer-500 WM_LR Yeo7-PC1 Phi/null and module-core analyses."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hcp_lausanne_phi_eid_pilot import circular_shift_null
from scripts.run_hcp_schaefer500_yeo7_module_phi_decomposition import (
    _block_phi,
    decompose_modules,
    null_rank_frequency,
    summarize_cores,
    summarize_null_cores,
)
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import (
    _subject_seed,
    fit_delta_history_phi,
    summarize_null,
)
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null_all import aggregate_rows
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import (
    default_yeo7_labels,
    fit_yeo7_pc1,
    load_yeo7_groups,
)


DEFAULT_DATA_ROOT = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_30"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "hcp_schaefer500_wm_yeo7_phi"
DEFAULT_REST_PHI = ROOT / "results" / "hcp_schaefer500_yeo7_pc1_phi_null_all" / "summary.json"
DEFAULT_REST_MODULE = ROOT / "results" / "hcp_schaefer500_yeo7_module_phi_decomposition" / "summary.json"
DEFAULT_STATUS = ROOT / "docs" / "log" / "hcp_schaefer500_wm_yeo7_phi" / "live_progress.json"
NETWORK_ABBREVIATIONS = {
    "Vis": "Vis",
    "SomMot": "Som",
    "DorsAttn": "DAN",
    "SalVentAttn": "SVAN",
    "Limbic": "Lim",
    "Cont": "Cont",
    "Default": "Def",
}


def discover_task_files(data_root: Path, task: str) -> tuple[Path, ...]:
    files = tuple(sorted(Path(data_root).glob(f"sub-*/{task}.mat")))
    subjects = [path.parent.name for path in files]
    if len(subjects) != len(set(subjects)):
        raise ValueError(f"Task {task!r} resolves to duplicate subject files.")
    return files


def load_task_series(path: Path, *, data_key: str, parcel_count: int) -> np.ndarray:
    payload = loadmat(path)
    if data_key not in payload:
        raise ValueError(f"MAT file {path} does not contain {data_key!r}.")
    values = np.asarray(payload[data_key], dtype=float)
    if values.ndim != 2 or values.shape[1] != int(parcel_count) or values.shape[0] < 4 or not np.isfinite(values).all():
        raise ValueError(f"Expected finite [time, {parcel_count}] {data_key} data in {path}, got {values.shape}.")
    return values


def _atomic_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def write_status(
    path: Path,
    *,
    phase: str,
    current: int,
    total: int,
    started: float,
    metrics: Mapping[str, Any] | None = None,
    message: str | None = None,
) -> None:
    elapsed = max(0.0, time.monotonic() - float(started))
    rate = current / elapsed if elapsed > 0 else 0.0
    payload = {
        "phase": phase,
        "pid": os.getpid(),
        "current": int(current),
        "total": int(total),
        "unit": "model",
        "elapsed_seconds": elapsed,
        "eta_seconds": (total - current) / rate if rate > 0 and current < total else 0.0 if current >= total else None,
        "metrics": dict(metrics or {}),
        "updated_at": time.time(),
    }
    if message:
        payload["message"] = str(message)
    _atomic_json(path, payload)


def _rank_atoms(atoms: Sequence[Any], top_k: int) -> list[Any]:
    ranked = sorted(
        (atom for atom in atoms if len(atom.sources) >= 2 and atom.value > 1.0e-9),
        key=lambda atom: atom.value,
        reverse=True,
    )
    return ranked[: int(top_k)]


def analyze_subject(
    raw_series: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    *,
    subject: str,
    development_end: int,
    order: int,
    alpha: float,
    null_replicates: int,
    seed: int,
    top_k: int,
    on_model_complete: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    network_names = tuple(groups)
    reduced = fit_yeo7_pc1(np.asarray(raw_series, dtype=float)[:development_end], groups).transform(raw_series)
    observed_fit = fit_delta_history_phi(reduced, alpha=alpha, order=order, development_end=development_end)
    observed_noise = np.asarray(observed_fit["noise_covariance"], dtype=float)
    development_reduced = np.asarray(reduced[:development_end], dtype=float)
    reduced_scale = np.where(development_reduced.std(axis=0, ddof=1) > 1.0e-12, development_reduced.std(axis=0, ddof=1), 1.0)
    max_pc_zscore = float(
        np.max(np.abs((development_reduced - development_reduced.mean(axis=0)) / reduced_scale))
    )
    noise_condition = float(np.linalg.cond(observed_noise))
    quality_flag = bool(max_pc_zscore > 10.0 or noise_condition > 500.0)
    observed_table, observed_atoms = decompose_modules(
        observed_fit["transition"], observed_fit["noise_covariance"], network_names, order=order
    )
    selected_atoms = _rank_atoms(observed_atoms, top_k)
    if on_model_complete is not None:
        on_model_complete(subject, 0)

    observed = float(observed_fit["phi"]["raw_phi"])
    null_phi: list[float] = []
    null_top_atoms: list[list[dict[str, Any]]] = []
    null_blocks: dict[tuple[str, ...], list[float]] = {tuple(atom.sources): [] for atom in selected_atoms}
    for replicate in range(int(null_replicates)):
        shifted = circular_shift_null(
            reduced[:development_end],
            seed=_subject_seed(seed + int(subject.removeprefix("sub-")), replicate),
        )
        null_fit = fit_delta_history_phi(shifted, alpha=alpha, order=order, development_end=development_end)
        null_table, null_atoms = decompose_modules(
            null_fit["transition"], null_fit["noise_covariance"], network_names, order=order
        )
        null_phi.append(float(null_fit["phi"]["raw_phi"]))
        ranked_null = _rank_atoms(null_atoms, top_k)
        null_top_atoms.append(
            [
                {
                    "sources": list(atom.sources),
                    "value": float(atom.value),
                    "kind": atom.kind,
                    "depth": int(atom.depth),
                }
                for atom in ranked_null
            ]
        )
        for sources in null_blocks:
            null_blocks[sources].append(_block_phi(sources, null_table))
        if on_model_complete is not None:
            on_model_complete(subject, replicate + 1)

    top_atoms = []
    for atom in selected_atoms:
        sources = tuple(atom.sources)
        block_phi = _block_phi(sources, observed_table)
        values = np.asarray(null_blocks[sources], dtype=float)
        top_atoms.append(
            {
                "sources": list(sources),
                "value": float(atom.value),
                "kind": atom.kind,
                "depth": int(atom.depth),
                "block_phi": float(block_phi),
                "null_block_phi_mean": float(values.mean()),
                "empirical_p": float((1 + np.sum(values >= block_phi)) / (len(values) + 1)),
            }
        )

    return {
        "subject": subject,
        "n_timepoints": int(len(raw_series)),
        "observed_raw_phi": observed,
        "joint_ei": float(observed_fit["phi"]["joint_ei"]),
        "singleton_ei_sum": float(observed_fit["phi"]["singleton_ei_sum"]),
        "null_raw_phi": null_phi,
        "null_comparison": summarize_null(observed, np.asarray(null_phi, dtype=float)),
        "quality_diagnostics": {
            "max_abs_development_pc_zscore": max_pc_zscore,
            "noise_covariance_condition": noise_condition,
            "noise_covariance_min_eigenvalue": float(np.linalg.eigvalsh(observed_noise).min()),
            "transition_spectral_norm": float(np.linalg.norm(observed_fit["transition"], 2)),
            "quality_flag": quality_flag,
            "flag_rule": "max_abs_development_pc_zscore > 10 or noise_covariance_condition > 500",
        },
        "module_full_phi": float(_block_phi(tuple(network_names), observed_table)),
        "atom_sum": float(sum(atom.value for atom in observed_atoms)),
        "atoms": [
            {"sources": list(atom.sources), "value": float(atom.value), "kind": atom.kind, "depth": int(atom.depth)}
            for atom in observed_atoms
        ],
        "top_atoms": top_atoms,
        "null_top_atoms": null_top_atoms,
    }


def paired_summary(task: np.ndarray, rest: np.ndarray, *, seed: int, replicates: int = 200_000) -> dict[str, Any]:
    task_values = np.asarray(task, dtype=float)
    rest_values = np.asarray(rest, dtype=float)
    if task_values.shape != rest_values.shape or task_values.ndim != 1 or len(task_values) < 1:
        raise ValueError("paired task and rest vectors must be one-dimensional and equally sized.")
    differences = task_values - rest_values
    rng = np.random.default_rng(int(seed))
    observed = abs(float(differences.mean()))
    extreme = 0
    remaining = int(replicates)
    while remaining:
        size = min(20_000, remaining)
        signs = rng.choice((-1.0, 1.0), size=(size, len(differences)))
        extreme += int(np.sum(np.abs(np.mean(signs * differences, axis=1)) >= observed - 1.0e-15))
        remaining -= size
    bootstrap_means = np.empty(20_000, dtype=float)
    for start in range(0, len(bootstrap_means), 2_000):
        stop = min(start + 2_000, len(bootstrap_means))
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)
    standard_deviation = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    return {
        "n": int(len(differences)),
        "task_mean": float(task_values.mean()),
        "rest_mean": float(rest_values.mean()),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "bootstrap_95_ci": [float(value) for value in np.quantile(bootstrap_means, [0.025, 0.975])],
        "positive_differences": int(np.sum(differences > 0)),
        "negative_differences": int(np.sum(differences < 0)),
        "cohens_dz": float(differences.mean() / standard_deviation) if standard_deviation > 0 else math.inf,
        "paired_sign_flip_p_two_sided": float((extreme + 1) / (int(replicates) + 1)),
        "permutation_replicates": int(replicates),
    }


def compare_with_rest(rows: Sequence[Mapping[str, Any]], rest_phi_path: Path) -> dict[str, Any]:
    rest = json.loads(Path(rest_phi_path).read_text(encoding="utf-8"))
    task_by_subject = {str(row["subject"]): row for row in rows}
    rest_by_subject = {str(row["subject"]): row for row in rest["rows"]}
    common = sorted(set(task_by_subject) & set(rest_by_subject))
    task_raw = np.asarray([task_by_subject[subject]["observed_raw_phi"] for subject in common], dtype=float)
    rest_raw = np.asarray([rest_by_subject[subject]["observed_raw_phi"] for subject in common], dtype=float)
    return {
        "common_subjects": common,
        "task_only_subjects": sorted(set(task_by_subject) - set(rest_by_subject)),
        "rest_only_subjects": sorted(set(rest_by_subject) - set(task_by_subject)),
        "raw_phi": paired_summary(task_raw, rest_raw, seed=2026071701),
        "paired_values": [
            {
                "subject": subject,
                "wm_raw_phi": float(task_by_subject[subject]["observed_raw_phi"]),
                "rest_raw_phi": float(rest_by_subject[subject]["observed_raw_phi"]),
            }
            for subject in common
        ],
        "contrast": "WM observed raw Phi minus REST observed raw Phi; no null correction",
        "interpretation_limit": "WM uses 304 fitting points whereas the existing REST result uses 900; the raw-Phi paired differences are descriptive and confounded by effective sample length.",
    }


def wm_observed_vs_null_group(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    observed = np.asarray([row["observed_raw_phi"] for row in rows], dtype=float)
    null_means = np.asarray([row["null_comparison"]["null_mean"] for row in rows], dtype=float)
    result = paired_summary(observed, null_means, seed=seed)
    result["contrast"] = "WM observed minus subject-specific WM null mean"
    return result


def compare_core_presence_with_rest(
    rows: Sequence[Mapping[str, Any]],
    rest_module_path: Path,
    cores: Sequence[Sequence[str]],
) -> list[dict[str, Any]]:
    rest = json.loads(Path(rest_module_path).read_text(encoding="utf-8"))
    wm_by_subject = {
        str(row["subject"]): {tuple(atom["sources"]) for atom in row["top_atoms"]}
        for row in rows
    }
    rest_by_subject = {
        str(row["subject"]): {tuple(atom["sources"]) for atom in row["top_atoms"]}
        for row in rest["rows"]
    }
    common = sorted(set(wm_by_subject) & set(rest_by_subject))
    comparisons = []
    for sources in cores:
        core = tuple(sources)
        both = sum(core in wm_by_subject[subject] and core in rest_by_subject[subject] for subject in common)
        wm_only = sum(core in wm_by_subject[subject] and core not in rest_by_subject[subject] for subject in common)
        rest_only = sum(core not in wm_by_subject[subject] and core in rest_by_subject[subject] for subject in common)
        neither = len(common) - both - wm_only - rest_only
        discordant = wm_only + rest_only
        p_value = float(binomtest(wm_only, discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
        comparisons.append(
            {
                "sources": list(core),
                "n_common_subjects": len(common),
                "wm_frequency": both + wm_only,
                "rest_frequency": both + rest_only,
                "both": both,
                "wm_only": wm_only,
                "rest_only": rest_only,
                "neither": neither,
                "exact_mcnemar_p_two_sided": p_value,
            }
        )
    order = sorted(range(len(comparisons)), key=lambda index: comparisons[index]["exact_mcnemar_p_two_sided"])
    running = 0.0
    adjusted = [1.0] * len(comparisons)
    for rank, index in enumerate(order):
        value = min(1.0, (len(comparisons) - rank) * comparisons[index]["exact_mcnemar_p_two_sided"])
        running = max(running, value)
        adjusted[index] = running
    for comparison, value in zip(comparisons, adjusted):
        comparison["holm_adjusted_p"] = value
    return comparisons


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


def plot_phi_comparison(summary: Mapping[str, Any], destination: Path) -> None:
    _style()
    rows = summary["rows"]
    comparison = summary["rest_comparison"]
    fig, axis = plt.subplots(figsize=(3.8, 3.2), constrained_layout=True)

    flagged = [row for row in rows if row["quality_diagnostics"]["quality_flag"]]
    paired = comparison["paired_values"]
    rest_raw = np.asarray([row["rest_raw_phi"] for row in paired], dtype=float)
    wm_raw = np.asarray([row["wm_raw_phi"] for row in paired], dtype=float)
    for rest_value, wm_value in zip(rest_raw, wm_raw):
        axis.plot([0, 1], [rest_value, wm_value], color="#B7BEC8", linewidth=0.6, zorder=1)
    axis.scatter(np.zeros(len(rest_raw)), rest_raw, color="#4C78A8", s=13, zorder=2)
    axis.scatter(np.ones(len(wm_raw)), wm_raw, color="#D55E00", s=13, zorder=2)
    axis.set(xticks=[0, 1], xticklabels=["REST", "WM"], ylabel="$\\Phi^{EID}$ (bits)")
    p_value = float(comparison["raw_phi"]["paired_sign_flip_p_two_sided"])
    axis.text(0.02, 0.98, f"paired n={len(rest_raw)}\n$p$={p_value:.3g}", transform=axis.transAxes, va="top")
    flagged_subjects = {str(row["subject"]) for row in flagged}
    for row in paired:
        if str(row["subject"]) in flagged_subjects:
            axis.annotate(
                str(row["subject"]).removeprefix("sub-"),
                xy=(1, float(row["wm_raw_phi"])),
                xytext=(-4, 5),
                textcoords="offset points",
                ha="right",
                fontsize=6,
            )
    _save_figure(fig, destination)


def _core_label(sources: Sequence[str]) -> str:
    return "+".join(NETWORK_ABBREVIATIONS[name] for name in sources)


def plot_core_distribution(summary: Mapping[str, Any], rest_module_path: Path, destination: Path) -> None:
    _style()
    rest = json.loads(Path(rest_module_path).read_text(encoding="utf-8"))
    selected = list(summary["core_summary"][:8])
    rows = summary["rows"]
    labels = [f"C{index + 1}" for index in range(len(selected))]
    matrix = np.full((len(rows), len(selected)), np.nan)
    for row_index, row in enumerate(rows):
        atoms = {tuple(atom["sources"]): atom for atom in row["top_atoms"]}
        for column, core in enumerate(selected):
            atom = atoms.get(tuple(core["sources"]))
            if atom is not None:
                matrix[row_index, column] = float(atom["value"])

    fig = plt.figure(figsize=(11.2, 7.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.65, 1.0))
    heatmap_axis = fig.add_subplot(grid[0, 0])
    frequency_axis = fig.add_subplot(grid[0, 1])
    image = heatmap_axis.imshow(np.ma.masked_invalid(matrix), cmap="YlGnBu", aspect="auto")
    heatmap_axis.set(
        xticks=np.arange(len(labels)),
        xticklabels=labels,
        yticks=np.arange(len(rows)),
        yticklabels=[str(row["subject"]).removeprefix("sub-") for row in rows],
        ylabel="Subject",
    )
    heatmap_axis.tick_params(axis="x", labelsize=6.5, pad=5)
    heatmap_axis.tick_params(axis="y", labelsize=6.5)
    colorbar = fig.colorbar(image, ax=heatmap_axis, pad=0.02, shrink=0.72)
    colorbar.set_label("Greedy atom contribution (bits)")

    rest_frequency = {tuple(core["sources"]): int(core["top_frequency"]) for core in rest["core_summary"]}
    null_frequency = {
        tuple(item["sources"]): float(item["null_frequency_mean"])
        for item in summary["null_rank_comparison"]
    }
    positions = np.arange(len(selected))
    wm_values = np.asarray([int(core["top_frequency"]) for core in selected])
    rest_values = np.asarray([rest_frequency.get(tuple(core["sources"]), 0) for core in selected])
    null_values = np.asarray([null_frequency.get(tuple(core["sources"]), 0.0) for core in selected])
    frequency_axis.scatter(null_values, positions - 0.18, color="#9AA5B1", marker="x", s=28, label="WM null mean")
    frequency_axis.scatter(rest_values, positions, color="#4C78A8", s=24, label="REST")
    frequency_axis.scatter(wm_values, positions + 0.18, color="#D55E00", s=24, label="WM")
    frequency_axis.set(
        yticks=positions,
        yticklabels=[f"C{index + 1}  {_core_label(core['sources'])}" for index, core in enumerate(selected)],
        xlabel="Subjects with core in greedy top-3",
        xlim=(-0.8, 30.8),
    )
    frequency_axis.invert_yaxis()
    frequency_axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=3, frameon=False)
    heatmap_axis.text(-0.11, 1.02, "a", transform=heatmap_axis.transAxes, fontweight="bold", fontsize=9)
    frequency_axis.text(-0.16, 1.02, "b", transform=frequency_axis.transAxes, fontweight="bold", fontsize=9)
    _save_figure(fig, destination)


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    aggregate = summary["phi_aggregate"]
    raw = summary["rest_comparison"]["raw_phi"]
    wm_null = summary["wm_observed_vs_null_group"]
    quality = summary["quality_sensitivity"]
    lines = [
        "# HCP WM_LR Schaefer-500 Yeo7-PC1 Phi 与协同核",
        "",
        "## 整体 Phi",
        "",
        f"30 名被试的 observed $\\Phi^{{EID}}$ 均值/中位数为 {aggregate['observed_raw_phi_mean']:.6f}/{aggregate['observed_raw_phi_median']:.6f} bits；"
        f"observed-minus-null-mean 均值/中位数为 {aggregate['observed_minus_null_mean_mean']:.6f}/{aggregate['observed_minus_null_mean_median']:.6f} bits。"
        f"Observed 高于自身 null mean：{aggregate['subjects_observed_above_null_mean']}/{aggregate['n_subjects']}；"
        f"经验 $p<0.05$：{aggregate['subjects_empirical_p_lt_0_05']}/{aggregate['n_subjects']}。",
        f"受试者级 observed-minus-null 均值的 95% bootstrap CI 为 [{wm_null['bootstrap_95_ci'][0]:.6f}, {wm_null['bootstrap_95_ci'][1]:.6f}]，"
        f"paired sign-flip $p={wm_null['paired_sign_flip_p_two_sided']:.6g}$。",
        "",
        "## 与既有静息态结果的描述性对比",
        "",
        f"共同受试者为 {raw['n']} 名。WM-minus-REST raw Phi 均值差为 {raw['mean_difference']:.6f} bits，"
        f"95% bootstrap CI [{raw['bootstrap_95_ci'][0]:.6f}, {raw['bootstrap_95_ci'][1]:.6f}]，"
        f"paired sign-flip $p={raw['paired_sign_flip_p_two_sided']:.6g}$。",
        "",
        "该横向比较只使用 observed raw Phi，不引入 null 校正。它直接复用既有静息态结果：WM 使用前 304 点拟合，而静息态使用前 900 点。故差异同时包含任务条件和有效样本长度变化，只能作描述性条件对照。",
        "",
        f"QC 标记被试：{', '.join(quality['excluded_subjects'])}。排除后 WM-minus-REST raw Phi 均值差为 "
        f"{quality['rest_comparison']['raw_phi']['mean_difference']:.6f} bits（$p={quality['rest_comparison']['raw_phi']['paired_sign_flip_p_two_sided']:.6g}$）。"
        "该分析为事后敏感性检查，不替代全 30 人主结果。",
        "",
        "## WM 协同核",
        "",
        "| 核 | WM top-3 频率 | top 时贡献均值（bits） | matched-null 频率均值；经验 p |",
        "|---|---:|---:|---:|",
    ]
    rank_by_sources = {tuple(item["sources"]): item for item in summary["null_rank_comparison"]}
    for core in summary["core_summary"]:
        rank = rank_by_sources[tuple(core["sources"])]
        lines.append(
            f"| {' + '.join(core['sources'])} | {core['top_frequency']} / {aggregate['n_subjects']} | "
            f"{core['mean_atom_value_when_top']:.6f} | {rank['null_frequency_mean']:.3f} / {aggregate['n_subjects']}; {rank['empirical_p']:.6f} |"
        )
    lines.extend(
        [
            "",
            "20 个 null 的最小经验 p 为 $1/21=0.047619$。Greedy 核是模块级定位摘要，不是经完备多重比较确认的唯一生物学 atom。",
            "",
            "## 与静息态协同核频率的配对对比",
            "",
            "| 核 | WM / 共同被试 | REST / 共同被试 | exact McNemar p | Holm p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for comparison in summary["rest_core_comparison"]:
        lines.append(
            f"| {' + '.join(comparison['sources'])} | {comparison['wm_frequency']} / {comparison['n_common_subjects']} | "
            f"{comparison['rest_frequency']} / {comparison['n_common_subjects']} | {comparison['exact_mcnemar_p_two_sided']:.6f} | "
            f"{comparison['holm_adjusted_p']:.6f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    data_root: Path,
    labels: Path,
    output_dir: Path,
    *,
    task: str = "WM_LR",
    data_key: str = "Schaefer500_taskRetained",
    parcel_count: int = 500,
    development_end: int = 304,
    order: int = 8,
    alpha: float = 10.0,
    null_replicates: int = 20,
    seed: int = 20260717,
    top_k: int = 3,
    subjects: Sequence[str] | None = None,
    rest_phi_path: Path = DEFAULT_REST_PHI,
    rest_module_path: Path = DEFAULT_REST_MODULE,
    status_path: Path = DEFAULT_STATUS,
) -> dict[str, Any]:
    files = discover_task_files(data_root, task)
    if subjects is not None:
        wanted = set(subjects)
        files = tuple(path for path in files if path.parent.name in wanted)
        missing = wanted - {path.parent.name for path in files}
        if missing:
            raise FileNotFoundError(f"Missing {task} inputs for: {sorted(missing)}")
    if not files:
        raise FileNotFoundError(f"No {task}.mat files found below {data_root}.")
    groups = load_yeo7_groups(labels, expected_parcels=parcel_count)
    started = time.monotonic()
    total = len(files) * (int(null_replicates) + 1)
    completed = 0
    rows = []

    def update(subject: str, replicate: int) -> None:
        nonlocal completed
        completed += 1
        write_status(
            status_path,
            phase="running",
            current=completed,
            total=total,
            started=started,
            metrics={"subject": subject, "replicate": int(replicate)},
        )

    write_status(status_path, phase="running", current=0, total=total, started=started, metrics={"task": task})
    try:
        for index, path in enumerate(files, start=1):
            subject = path.parent.name
            print(f"[{index}/{len(files)}] {subject}", flush=True)
            raw = load_task_series(path, data_key=data_key, parcel_count=parcel_count)
            if development_end > len(raw):
                raise ValueError(f"development_end={development_end} exceeds {subject} length {len(raw)}.")
            rows.append(
                analyze_subject(
                    raw,
                    groups,
                    subject=subject,
                    development_end=development_end,
                    order=order,
                    alpha=alpha,
                    null_replicates=null_replicates,
                    seed=seed,
                    top_k=top_k,
                    on_model_complete=update,
                )
            )
            _atomic_json(Path(output_dir) / "partial_rows.json", {"config": {"task": task, "data_key": data_key}, "rows": rows})
        core_summary = summarize_cores(rows)
        null_rank_comparison = [
            null_rank_frequency(rows, core["sources"], observed_frequency=int(core["top_frequency"]))
            for core in core_summary
        ]
        flagged_subjects = [
            str(row["subject"]) for row in rows if row["quality_diagnostics"]["quality_flag"]
        ]
        sensitivity_rows = [row for row in rows if str(row["subject"]) not in set(flagged_subjects)]
        summary = {
            "config": {
                "task": task,
                "data_key": data_key,
                "parcel_count": int(parcel_count),
                "labels": str(labels),
                "network_sizes": {name: len(indices) for name, indices in groups.items()},
                "subjects": [path.parent.name for path in files],
                "n_timepoints": 405,
                "development_end": int(development_end),
                "model": "delta Ridge",
                "order": int(order),
                "alpha": float(alpha),
                "source_dimension": int(order * 7),
                "target_dimension": 7,
                "estimator": "Gaussian log-det",
                "null": "independent non-zero circular shift of each Yeo7 PC1; model refit",
                "null_replicates": int(null_replicates),
                "seed": int(seed),
                "top_k_per_subject": int(top_k),
            },
            "rows": rows,
            "phi_aggregate": aggregate_rows(rows),
            "wm_observed_vs_null_group": wm_observed_vs_null_group(rows, seed=2026071703),
            "core_summary": core_summary,
            "null_rank_comparison": null_rank_comparison,
            "null_core_summary": summarize_null_cores(rows),
            "rest_comparison": compare_with_rest(rows, rest_phi_path),
            "rest_core_comparison": compare_core_presence_with_rest(
                rows,
                rest_module_path,
                (
                    ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Cont", "Default"),
                    ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"),
                ),
            ),
            "quality_sensitivity": {
                "excluded_subjects": flagged_subjects,
                "wm_observed_vs_null_group": wm_observed_vs_null_group(sensitivity_rows, seed=2026071704),
                "rest_comparison": compare_with_rest(sensitivity_rows, rest_phi_path),
                "status": "post-hoc sensitivity analysis; primary results retain all subjects",
            },
        }
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(output_dir / "summary.json", summary)
        write_report(summary, output_dir / "report.md")
        plot_phi_comparison(summary, output_dir / "wm_rest_phi_comparison")
        plot_core_distribution(summary, rest_module_path, output_dir / "wm_core_distribution")
        (output_dir / "partial_rows.json").unlink(missing_ok=True)
        write_status(status_path, phase="complete", current=total, total=total, started=started, metrics={"subjects": len(rows)})
        return summary
    except Exception as error:
        write_status(
            status_path,
            phase="failed",
            current=completed,
            total=total,
            started=started,
            metrics={"subjects_complete": len(rows)},
            message=str(error),
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--labels", type=Path, default=default_yeo7_labels(500))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task", default="WM_LR")
    parser.add_argument("--data-key", default="Schaefer500_taskRetained")
    parser.add_argument("--development-end", type=int, default=304)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--null-replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--subjects", default="", help="Comma-separated subject IDs; defaults to all WM subjects.")
    parser.add_argument("--rest-phi", type=Path, default=DEFAULT_REST_PHI)
    parser.add_argument("--rest-module", type=Path, default=DEFAULT_REST_MODULE)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS)
    args = parser.parse_args(argv)
    subjects = tuple(value.strip() for value in args.subjects.split(",") if value.strip()) or None
    summary = run(
        args.data_root,
        args.labels,
        args.output_dir,
        task=args.task,
        data_key=args.data_key,
        development_end=args.development_end,
        order=args.order,
        alpha=args.alpha,
        null_replicates=args.null_replicates,
        seed=args.seed,
        top_k=args.top_k,
        subjects=subjects,
        rest_phi_path=args.rest_phi,
        rest_module_path=args.rest_module,
        status_path=args.status_path,
    )
    print(json.dumps({"phi_aggregate": summary["phi_aggregate"], "rest_comparison": summary["rest_comparison"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
