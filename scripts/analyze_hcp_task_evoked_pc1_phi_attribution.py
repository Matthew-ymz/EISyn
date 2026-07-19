#!/usr/bin/env python3
"""Compare retained-PCA with task-evoked-PCA for HCP Yeo7 Phi attribution."""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_schaefer500_yeo7_network_attribution import (
    DEFAULT_REST_ROOT,
    DEFAULT_TASK_ROOT,
    NETWORK_LABELS,
    NETWORK_ORDER,
    discover_inputs,
    transition_attribution,
)
from scripts.run_hcp_schaefer500_all_tasks_phi import DISPLAY_NAMES, development_end_for_length
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import (
    default_yeo7_labels,
    fit_yeo7_pc1,
    load_yeo7_groups,
)
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi


TASKS = ("EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")
DEFAULT_BASELINE = ROOT / "results" / "hcp_schaefer500_yeo7_network_attribution" / "summary.json"
DEFAULT_OUTPUT = ROOT / "results" / "hcp_schaefer500_task_evoked_pc1_phi_attribution"


def load_task_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = loadmat(path)
    retained = np.asarray(payload["Schaefer500_taskRetained"], dtype=float)
    regressed = np.asarray(payload["Schaefer500_taskRegressed"], dtype=float)
    if retained.shape != regressed.shape or retained.ndim != 2 or retained.shape[1] != 500:
        raise ValueError(f"Expected paired [time, 500] task arrays in {path}, got {retained.shape} and {regressed.shape}.")
    if not np.isfinite(retained).all() or not np.isfinite(regressed).all():
        raise ValueError(f"Non-finite task data in {path}.")
    return retained, regressed


def analyze_task(
    path: Path,
    groups: Mapping[str, Sequence[int]],
    *,
    subject: str,
    condition: str,
    order: int,
    alpha: float,
) -> dict[str, Any]:
    retained, regressed = load_task_pair(path)
    development_end = development_end_for_length(len(retained))
    task_evoked = retained - regressed
    reducer = fit_yeo7_pc1(task_evoked[:development_end], groups)
    reduced_retained = reducer.transform(retained)
    fitted = fit_delta_history_phi(
        reduced_retained,
        alpha=float(alpha),
        order=int(order),
        development_end=int(development_end),
    )
    attribution = transition_attribution(
        fitted["transition"], fitted["noise_covariance"], tuple(groups), order=int(order)
    )
    development = reduced_retained[:development_end]
    scale = development.std(axis=0, ddof=1)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return {
        "subject": subject,
        "condition": condition,
        "n_timepoints": int(len(retained)),
        "development_end": int(development_end),
        "pca_explained_variance_ratio": {
            name: float(model.explained_variance_ratio_[0])
            for name, model in zip(groups, reducer.models)
        },
        "quality_diagnostics": {
            "max_abs_development_pc_zscore": float(
                np.max(np.abs((development - development.mean(axis=0)) / scale))
            ),
            "noise_covariance_condition": float(np.linalg.cond(fitted["noise_covariance"])),
        },
        **attribution,
    }


def _worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    path, groups, subject, condition, order, alpha = payload
    return analyze_task(
        Path(path), groups, subject=str(subject), condition=str(condition), order=int(order), alpha=float(alpha)
    )


def load_baseline(path: Path, subjects: Sequence[str]) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = payload["config"]
    if int(config["order"]) != 5 or not np.isclose(float(config["alpha"]), 10.0):
        raise ValueError("Baseline must use p=5 and alpha=10.")
    wanted = set(subjects)
    rows = {
        (str(row["subject"]), str(row["condition"])): row
        for row in payload["rows"]
        if str(row["subject"]) in wanted and str(row["condition"]) in ("REST",) + TASKS
    }
    missing = [
        (subject, condition)
        for subject in subjects
        for condition in ("REST",) + TASKS
        if (subject, condition) not in rows
    ]
    if missing:
        raise ValueError(f"Baseline lacks {len(missing)} paired task rows; first missing={missing[0]}.")
    return rows


def load_reusable_new_rows(path: Path, subjects: Sequence[str]) -> dict[tuple[str, str], dict[str, Any]]:
    if not Path(path).is_file():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = payload.get("config", {})
    if int(config.get("order", -1)) != 5 or not np.isclose(float(config.get("alpha", np.nan)), 10.0):
        return {}
    wanted = set(subjects)
    rows = {
        (str(row["subject"]), str(row["condition"])): dict(row)
        for row in payload.get("rows", [])
        if str(row.get("subject")) in wanted and str(row.get("condition")) in TASKS
    }
    if any((subject, task) not in rows for subject in subjects for task in TASKS):
        return {}
    return rows


def matrices(
    rows: Mapping[tuple[str, str], Mapping[str, Any]], subjects: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contribution = np.asarray(
        [
            [
                [float(rows[(subject, task)]["total_phi_contribution"][network]) for network in NETWORK_ORDER]
                for subject in subjects
            ]
            for task in TASKS
        ],
        dtype=float,
    )
    phi = np.asarray(
        [[float(rows[(subject, task)]["raw_phi"]) for subject in subjects] for task in TASKS], dtype=float
    )
    if np.any(phi <= 1.0e-10):
        raise ValueError("Run normalization requires positive task Phi in every paired row.")
    share = contribution / phi[:, :, None]
    return contribution, share, phi


def loso(values: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    n_tasks, n_subjects, _ = values.shape
    confusion = np.zeros((n_tasks, n_tasks), dtype=int)
    predicted = np.empty((n_tasks, n_subjects), dtype=int)
    for held_out in range(n_subjects):
        train = np.delete(values, held_out, axis=1)
        pooled = train.reshape(-1, train.shape[-1])
        mean = pooled.mean(axis=0)
        scale = pooled.std(axis=0, ddof=1)
        scale = np.where(scale > 1.0e-12, scale, 1.0)
        centroids = ((train - mean) / scale).mean(axis=1)
        test = (values[:, held_out] - mean) / scale
        guess = np.argmin(np.linalg.norm(test[:, None, :] - centroids[None, :, :], axis=2), axis=1)
        predicted[:, held_out] = guess
        for truth, item in enumerate(guess):
            confusion[truth, int(item)] += 1
    return float(np.trace(confusion) / confusion.sum()), confusion, predicted


def separation_ratio(values: np.ndarray) -> float:
    pooled = values.reshape(-1, values.shape[-1])
    scale = pooled.std(axis=0, ddof=1)
    scaled = values / np.where(scale > 1.0e-12, scale, 1.0)
    centroids = scaled.mean(axis=1)
    between = [
        float(np.linalg.norm(centroids[left] - centroids[right]))
        for left in range(len(TASKS))
        for right in range(left + 1, len(TASKS))
    ]
    within = np.linalg.norm(scaled - centroids[:, None, :], axis=2)
    return float(np.mean(between) / max(float(within.mean()), 1.0e-12))


def cluster_bootstrap_delta(
    left_correct: np.ndarray,
    right_correct: np.ndarray,
    *,
    seed: int,
    repeats: int = 10_000,
) -> list[float]:
    per_subject = left_correct.mean(axis=0) - right_correct.mean(axis=0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(per_subject), size=(int(repeats), len(per_subject)))
    return [float(value) for value in np.quantile(per_subject[indices].mean(axis=1), [0.025, 0.975])]


def mean_ci(values: np.ndarray, *, seed: int, repeats: int = 10_000) -> tuple[float, list[float]]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(int(repeats), len(array)))
    return float(array.mean()), [float(v) for v in np.quantile(array[indices].mean(axis=1), [0.025, 0.975])]


def summarize(
    new_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    baseline_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    subjects: Sequence[str],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    base_raw, base_share, base_phi = matrices(baseline_rows, subjects)
    new_raw, new_share, new_phi = matrices(new_rows, subjects)
    rest_phi = np.asarray(
        [float(baseline_rows[(subject, "REST")]["raw_phi"]) for subject in subjects], dtype=float
    )
    rest_contribution = np.asarray(
        [
            [
                float(baseline_rows[(subject, "REST")]["total_phi_contribution"][network])
                for network in NETWORK_ORDER
            ]
            for subject in subjects
        ],
        dtype=float,
    )
    rest_share = rest_contribution / rest_phi[:, None]
    base_raw_acc, _, base_raw_pred = loso(base_raw)
    base_share_acc, _, base_share_pred = loso(base_share)
    new_raw_acc, new_raw_confusion, new_raw_pred = loso(new_raw)
    new_share_acc, new_share_confusion, new_share_pred = loso(new_share)
    truth = np.arange(len(TASKS))[:, None]
    base_raw_correct = base_raw_pred == truth
    base_share_correct = base_share_pred == truth
    new_raw_correct = new_raw_pred == truth
    new_share_correct = new_share_pred == truth
    mean_share = new_share.mean(axis=1)
    pairwise_total_variation = []
    for left in range(len(TASKS)):
        for right in range(left + 1, len(TASKS)):
            pairwise_total_variation.append(
                {
                    "left": TASKS[left],
                    "right": TASKS[right],
                    "total_variation": float(0.5 * np.abs(mean_share[left] - mean_share[right]).sum()),
                }
            )
    pca_ev = np.asarray(
        [
            [
                [float(new_rows[(subject, task)]["pca_explained_variance_ratio"][network]) for network in NETWORK_ORDER]
                for subject in subjects
            ]
            for task in TASKS
        ]
    )
    state_summary = {}
    for task_index, task in enumerate(TASKS):
        base_mean, base_ci = mean_ci(base_phi[task_index], seed=2026073100 + task_index)
        new_mean, new_ci = mean_ci(new_phi[task_index], seed=2026073200 + task_index)
        delta_mean, delta_ci = mean_ci(new_phi[task_index] - base_phi[task_index], seed=2026073300 + task_index)
        state_summary[task] = {
            "baseline_phi_mean_bits": base_mean,
            "baseline_phi_bootstrap_95_ci": base_ci,
            "task_evoked_pca_phi_mean_bits": new_mean,
            "task_evoked_pca_phi_bootstrap_95_ci": new_ci,
            "paired_phi_delta_mean_bits": delta_mean,
            "paired_phi_delta_bootstrap_95_ci": delta_ci,
            "task_evoked_pca_explained_variance_mean": {
                network: float(pca_ev[task_index, :, network_index].mean())
                for network_index, network in enumerate(NETWORK_ORDER)
            },
            "network_share_percent": {
                network: float(100.0 * mean_share[task_index, network_index])
                for network_index, network in enumerate(NETWORK_ORDER)
            },
        }
    rest_mean, rest_ci = mean_ci(rest_phi, seed=2026073299)
    rest_mean_share = rest_share.mean(axis=0)
    rest_contrasts = {}
    for task_index, task in enumerate(TASKS):
        difference = rest_phi - new_phi[task_index]
        difference_mean, difference_ci = mean_ci(difference, seed=2026073350 + task_index)
        rest_contrasts[task] = {
            "rest_minus_task_evoked_pca_phi_mean_bits": difference_mean,
            "bootstrap_95_ci": difference_ci,
            "n_rest_greater": int(np.sum(difference > 0.0)),
            "n_task_greater": int(np.sum(difference < 0.0)),
            "network_share_total_variation": float(
                0.5 * np.abs(rest_mean_share - mean_share[task_index]).sum()
            ),
        }
    closure = max(
        max(
            abs(float(row["cross_shapley_sum_error"])),
            abs(float(row["total_contribution_sum_error"])),
            abs(float(row["atom_sum_error"])),
        )
        for row in new_rows.values()
    )
    flagged_subjects = sorted(
        {
            subject
            for (subject, _), row in new_rows.items()
            if float(row["quality_diagnostics"]["max_abs_development_pc_zscore"]) > 10.0
            or float(row["quality_diagnostics"]["noise_covariance_condition"]) > 500.0
        }
    )
    sensitivity: dict[str, Any] | None = None
    if flagged_subjects and len(subjects) - len(flagged_subjects) >= 2:
        keep = np.asarray([subject not in set(flagged_subjects) for subject in subjects], dtype=bool)
        sensitivity = {
            "excluded_subjects": flagged_subjects,
            "n_subjects": int(keep.sum()),
            "baseline_raw_accuracy": loso(base_raw[:, keep])[0],
            "task_evoked_pca_raw_accuracy": loso(new_raw[:, keep])[0],
            "baseline_raw_separation_ratio": separation_ratio(base_raw[:, keep]),
            "task_evoked_pca_raw_separation_ratio": separation_ratio(new_raw[:, keep]),
            "baseline_run_normalized_accuracy": loso(base_share[:, keep])[0],
            "task_evoked_pca_run_normalized_accuracy": loso(new_share[:, keep])[0],
            "baseline_run_normalized_separation_ratio": separation_ratio(base_share[:, keep]),
            "task_evoked_pca_run_normalized_separation_ratio": separation_ratio(new_share[:, keep]),
        }
    summary = {
        "config": {
            "subjects": list(subjects),
            "n_subjects": len(subjects),
            "tasks": list(TASKS),
            "representation": "Yeo7 PC1 fitted on retained-minus-regressed training prefix and applied to retained full series",
            "baseline_representation": "Yeo7 PC1 fitted on retained training prefix and applied to retained full series",
            "order": 5,
            "alpha": 10.0,
            "estimator": "Gaussian log-det affine continuous-EI approximation",
            "rest_policy": "excluded from the primary controlled comparison because REST has no retained/regressed pair",
        },
        "discriminability": {
            "chance": 1.0 / len(TASKS),
            "baseline_raw_accuracy": base_raw_acc,
            "task_evoked_pca_raw_accuracy": new_raw_acc,
            "raw_accuracy_delta": new_raw_acc - base_raw_acc,
            "raw_accuracy_delta_cluster_bootstrap_95_ci": cluster_bootstrap_delta(
                new_raw_correct, base_raw_correct, seed=2026073401
            ),
            "baseline_run_normalized_accuracy": base_share_acc,
            "task_evoked_pca_run_normalized_accuracy": new_share_acc,
            "run_normalized_accuracy_delta": new_share_acc - base_share_acc,
            "run_normalized_accuracy_delta_cluster_bootstrap_95_ci": cluster_bootstrap_delta(
                new_share_correct, base_share_correct, seed=2026073402
            ),
            "baseline_raw_separation_ratio": separation_ratio(base_raw),
            "task_evoked_pca_raw_separation_ratio": separation_ratio(new_raw),
            "baseline_run_normalized_separation_ratio": separation_ratio(base_share),
            "task_evoked_pca_run_normalized_separation_ratio": separation_ratio(new_share),
            "task_evoked_pca_raw_confusion": new_raw_confusion.tolist(),
            "task_evoked_pca_run_normalized_confusion": new_share_confusion.tolist(),
        },
        "direct_network_share_comparison": {
            "mean_pairwise_total_variation": float(
                np.mean([item["total_variation"] for item in pairwise_total_variation])
            ),
            "minimum_pairwise_total_variation": min(
                pairwise_total_variation, key=lambda item: item["total_variation"]
            ),
            "maximum_pairwise_total_variation": max(
                pairwise_total_variation, key=lambda item: item["total_variation"]
            ),
            "network_share_range_percentage_points": {
                network: float(100.0 * (mean_share[:, index].max() - mean_share[:, index].min()))
                for index, network in enumerate(NETWORK_ORDER)
            },
            "pairwise_total_variation": pairwise_total_variation,
        },
        "diagnostics": {
            "maximum_decomposition_identity_error_bits": float(closure),
            "quality_flag_models": int(
                sum(
                    float(row["quality_diagnostics"]["max_abs_development_pc_zscore"]) > 10.0
                    or float(row["quality_diagnostics"]["noise_covariance_condition"]) > 500.0
                    for row in new_rows.values()
                )
            ),
            "n_models": len(new_rows),
            "excluding_quality_flag_subjects": sensitivity,
        },
        "rest_original_pca_reference": {
            "representation": "REST PC1 fitted on the REST development prefix and applied to the REST series",
            "phi_mean_bits": rest_mean,
            "phi_bootstrap_95_ci": rest_ci,
            "network_share_percent": {
                network: float(100.0 * rest_mean_share[index])
                for index, network in enumerate(NETWORK_ORDER)
            },
            "paired_rest_minus_task_evoked_pca": rest_contrasts,
            "comparability_limit": "REST and task states use different PCA-fitting signals; contrasts are descriptive external-reference comparisons, not a one-factor projection comparison."
        },
        "state_summary": state_summary,
    }
    arrays = {
        "baseline_contribution": base_raw,
        "baseline_share": base_share,
        "baseline_phi": base_phi,
        "rest_original_pca_phi": rest_phi,
        "rest_original_pca_contribution": rest_contribution,
        "rest_original_pca_share": rest_share,
        "task_evoked_pca_contribution": new_raw,
        "task_evoked_pca_share": new_share,
        "task_evoked_pca_phi": new_phi,
        "task_evoked_pca_explained_variance": pca_ev,
    }
    return summary, arrays


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
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot(summary: Mapping[str, Any], arrays: Mapping[str, np.ndarray], output: Path) -> None:
    configure_style()
    task_network_share = arrays["task_evoked_pca_share"].mean(axis=1).T * 100.0
    rest_network_share = arrays["rest_original_pca_share"].mean(axis=0)[:, None] * 100.0
    network_share = np.concatenate([rest_network_share, task_network_share], axis=1)
    confusion = np.asarray(summary["discriminability"]["task_evoked_pca_run_normalized_confusion"], dtype=float)
    confusion /= confusion.sum(axis=1, keepdims=True)
    share_min = float(np.floor(network_share.min()))
    share_max = float(np.ceil(network_share.max()))
    fig = plt.figure(figsize=(10.8, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1.0))
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]

    image = axes[0].imshow(network_share, cmap="YlGnBu", vmin=share_min, vmax=share_max, aspect="auto")
    axes[0].set(
        xticks=np.arange(len(TASKS) + 1), xticklabels=["REST"] + [DISPLAY_NAMES[t] for t in TASKS],
        yticks=np.arange(len(NETWORK_ORDER)), yticklabels=[NETWORK_LABELS[n] for n in NETWORK_ORDER],
        xlabel="Task", ylabel="Yeo7 network",
    )
    axes[0].tick_params(axis="x", labelrotation=35, length=0)
    axes[0].tick_params(axis="y", length=0)
    axes[0].axvline(0.5, color="#333333", linewidth=0.9)
    for row in range(len(NETWORK_ORDER)):
        for column in range(len(TASKS) + 1):
            value = network_share[row, column]
            normalized = (value - share_min) / max(share_max - share_min, 1.0e-12)
            axes[0].text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=5.2,
                         color="white" if normalized > 0.62 else "black")
    fig.colorbar(image, ax=axes[0], shrink=0.82, pad=0.02).set_label(r"Share of system-level $\Xi$ (%)")

    disc = summary["discriminability"]
    x = np.arange(2)
    width = 0.32
    baseline = [disc["baseline_raw_accuracy"], disc["baseline_run_normalized_accuracy"]]
    task_pca = [disc["task_evoked_pca_raw_accuracy"], disc["task_evoked_pca_run_normalized_accuracy"]]
    axes[1].bar(x - width / 2, baseline, width, color="#B7C9D6", label="Retained PCA")
    axes[1].bar(x + width / 2, task_pca, width, color="#D98B5F", label="Task-evoked PCA")
    axes[1].axhline(disc["chance"], color="#555555", linestyle="--", linewidth=0.8, label="Chance")
    axes[1].set(xticks=x, xticklabels=("Raw contribution", "Run-normalized share"), ylabel="Seven-task LOSO accuracy", ylim=(0, 0.7))
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    state_x = np.arange(len(TASKS) + 1)
    rest = summary["rest_original_pca_reference"]
    rest_mean = float(rest["phi_mean_bits"])
    rest_ci = np.asarray(rest["phi_bootstrap_95_ci"], dtype=float)
    baseline_means = np.asarray([summary["state_summary"][task]["baseline_phi_mean_bits"] for task in TASKS])
    baseline_cis = np.asarray([summary["state_summary"][task]["baseline_phi_bootstrap_95_ci"] for task in TASKS])
    new_means = np.asarray([summary["state_summary"][task]["task_evoked_pca_phi_mean_bits"] for task in TASKS])
    new_cis = np.asarray([summary["state_summary"][task]["task_evoked_pca_phi_bootstrap_95_ci"] for task in TASKS])
    axes[2].errorbar(
        [0.0], [rest_mean], yerr=np.asarray([[rest_mean - rest_ci[0]], [rest_ci[1] - rest_mean]]),
        fmt="D", color="#4C78A8", ecolor="#9CB7CF", capsize=2, markersize=4, label="REST: original PCA",
    )
    axes[2].errorbar(
        state_x[1:] - 0.08, baseline_means,
        yerr=np.vstack([baseline_means - baseline_cis[:, 0], baseline_cis[:, 1] - baseline_means]),
        fmt="s", color="#8CA9BC", ecolor="#B7C9D6", capsize=2, markersize=3.6, label="Tasks: retained PCA",
    )
    axes[2].errorbar(
        state_x[1:] + 0.08, new_means,
        yerr=np.vstack([new_means - new_cis[:, 0], new_cis[:, 1] - new_means]),
        fmt="o", color="#D46A3A", ecolor="#E7AF95", capsize=2, markersize=4, label="Tasks: task-evoked PCA",
    )
    axes[2].set(
        xticks=state_x,
        xticklabels=["REST"] + [DISPLAY_NAMES[t] for t in TASKS],
        ylabel=r"System-level $\Xi$ (bits)",
        xlabel="State",
    )
    axes[2].tick_params(axis="x", labelrotation=35)
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=3)

    image = axes[3].imshow(confusion, cmap="Blues", vmin=0.0, vmax=1.0)
    axes[3].set(
        xticks=np.arange(len(TASKS)), xticklabels=[DISPLAY_NAMES[t] for t in TASKS],
        yticks=np.arange(len(TASKS)), yticklabels=[DISPLAY_NAMES[t] for t in TASKS],
        xlabel="Predicted task", ylabel="True task",
    )
    axes[3].tick_params(axis="x", labelrotation=40, length=0)
    axes[3].tick_params(axis="y", length=0)
    for row in range(len(TASKS)):
        for column in range(len(TASKS)):
            value = confusion[row, column]
            axes[3].text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=4.7,
                         color="white" if value > 0.55 else "black")
    fig.colorbar(image, ax=axes[3], shrink=0.82, pad=0.02).set_label("Fraction of subjects")
    for label, axis in zip("abcd", axes):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    d = summary["discriminability"]
    lines = [
        "# 任务诱发 PC1 投影下的 Yeo7 Xi 层级归因",
        "",
        "本实验只改变 PCA 的拟合信号：基线在 taskRetained 上拟合每个 Yeo7 网络的 PC1；新方法在 taskRetained-taskRegressed 上拟合 PC1。两种方法都把所得载荷投影到完整 taskRetained 时序，随后使用完全相同的五阶 Delta-Ridge、Gaussian log-det EI、127 个网络联盟、精确 Shapley 归因与 greedy 层级分解。脚本历史字段 `raw_phi` 在本文统一记为系统级联合有效信息增量 $\\Xi$。",
        "",
        "REST 没有 retained/regressed 配对，因此不进入本轮任务投影方法的主受控比较；图中加入原 PCA 的 REST 作为外部参照。",
        "",
        "## Panel a 指标",
        "",
        r"对被试 $s$、状态 $c$ 和网络 $i$，先将该网络内部五阶历史的联合增量与跨网络 Shapley 份额相加，得到 $a_{sci}=\Xi^{\mathrm{within}}_{sci}+\psi^{\mathrm{cross}}_{sci}$，且 $\sum_i a_{sci}=\Xi_{sc}$。再定义网络份额 $p_{sci}=a_{sci}/\Xi_{sc}$。Panel a 直接显示 $100p_{sci}$ 的被试均值，不再减去其他任务；每个状态列的七网络份额合计为 100%。最左列是原 PCA REST，竖线右侧是任务诱发 PCA 的七任务。它表示系统级 $\Xi$ 的网络归因构成，不是单个网络的 EI。",
        "",
        "## 任务可分性",
        "",
        f"- Raw contribution LOSO：{100*d['baseline_raw_accuracy']:.2f}% -> {100*d['task_evoked_pca_raw_accuracy']:.2f}%（差 {100*d['raw_accuracy_delta']:+.2f} pp；被试聚类 bootstrap 95% CI [{100*d['raw_accuracy_delta_cluster_bootstrap_95_ci'][0]:+.2f}, {100*d['raw_accuracy_delta_cluster_bootstrap_95_ci'][1]:+.2f}] pp）。",
        f"- Run-normalized share LOSO：{100*d['baseline_run_normalized_accuracy']:.2f}% -> {100*d['task_evoked_pca_run_normalized_accuracy']:.2f}%（差 {100*d['run_normalized_accuracy_delta']:+.2f} pp；95% CI [{100*d['run_normalized_accuracy_delta_cluster_bootstrap_95_ci'][0]:+.2f}, {100*d['run_normalized_accuracy_delta_cluster_bootstrap_95_ci'][1]:+.2f}] pp）。",
        f"- Raw between/within separation ratio：{d['baseline_raw_separation_ratio']:.3f} -> {d['task_evoked_pca_raw_separation_ratio']:.3f}。",
        f"- Normalized between/within separation ratio：{d['baseline_run_normalized_separation_ratio']:.3f} -> {d['task_evoked_pca_run_normalized_separation_ratio']:.3f}。",
        f"- 七任务组均值分布的平均两两 total-variation 距离为 {100*summary['direct_network_share_comparison']['mean_pairwise_total_variation']:.2f}%；最大为 {100*summary['direct_network_share_comparison']['maximum_pairwise_total_variation']['total_variation']:.2f}%（{DISPLAY_NAMES[summary['direct_network_share_comparison']['maximum_pairwise_total_variation']['left']]} vs {DISPLAY_NAMES[summary['direct_network_share_comparison']['maximum_pairwise_total_variation']['right']]}），最小为 {100*summary['direct_network_share_comparison']['minimum_pairwise_total_variation']['total_variation']:.2f}%（{DISPLAY_NAMES[summary['direct_network_share_comparison']['minimum_pairwise_total_variation']['left']]} vs {DISPLAY_NAMES[summary['direct_network_share_comparison']['minimum_pairwise_total_variation']['right']]}）。",
        "",
        "## REST 原 PCA 外部参照",
        "",
        f"REST 原 PCA 的系统级 $\\Xi$ 为 {summary['rest_original_pca_reference']['phi_mean_bits']:.4f} bits，bootstrap 95% CI [{summary['rest_original_pca_reference']['phi_bootstrap_95_ci'][0]:.4f}, {summary['rest_original_pca_reference']['phi_bootstrap_95_ci'][1]:.4f}]。",
        "",
        "| Task | REST minus task-evoked-PCA Xi [95% CI] | REST greater / task greater | REST–task share TV |",
        "|---|---:|---:|---:|",
    ]
    for task in TASKS:
        contrast = summary["rest_original_pca_reference"]["paired_rest_minus_task_evoked_pca"][task]
        ci = contrast["bootstrap_95_ci"]
        lines.append(
            f"| {DISPLAY_NAMES[task]} | {contrast['rest_minus_task_evoked_pca_phi_mean_bits']:+.4f} "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {contrast['n_rest_greater']}/{contrast['n_task_greater']} | "
            f"{100*contrast['network_share_total_variation']:.2f}% |"
        )
    lines.extend(
        [
        "",
        "这些 REST–任务差值在描述上保留了 REST 整体 $\\Xi$ 更高的原结论；但 REST 的 PC1 来自 REST 自身，而任务 PC1 来自 retained-regressed，故不能把差值完全归因于状态本身。",
        "",
        "## 逐任务 Xi 变化",
        "",
        "| Task | Baseline Xi | Task-evoked PCA Xi | Paired delta [95% CI] | Largest network share | Smallest network share |",
        "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for task in TASKS:
        item = summary["state_summary"][task]
        shares = item["network_share_percent"]
        high = max(shares, key=shares.get)
        low = min(shares, key=shares.get)
        ci = item["paired_phi_delta_bootstrap_95_ci"]
        lines.append(
            f"| {DISPLAY_NAMES[task]} | {item['baseline_phi_mean_bits']:.4f} | {item['task_evoked_pca_phi_mean_bits']:.4f} | "
            f"{item['paired_phi_delta_mean_bits']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{NETWORK_LABELS[high]} {shares[high]:.2f}% | {NETWORK_LABELS[low]} {shares[low]:.2f}% |"
        )
    diag = summary["diagnostics"]
    sensitivity = diag.get("excluding_quality_flag_subjects")
    if sensitivity:
        lines.extend(
            [
                "",
                "## 质量敏感性",
                "",
                f"删除触发质量标记的被试（{', '.join(sensitivity['excluded_subjects'])}）后，raw LOSO 由 "
                f"{100*sensitivity['baseline_raw_accuracy']:.2f}% 提高到 {100*sensitivity['task_evoked_pca_raw_accuracy']:.2f}%，"
                f"raw separation ratio 由 {sensitivity['baseline_raw_separation_ratio']:.3f} 提高到 {sensitivity['task_evoked_pca_raw_separation_ratio']:.3f}；"
                f"run-normalized LOSO 由 {100*sensitivity['baseline_run_normalized_accuracy']:.2f}% 提高到 "
                f"{100*sensitivity['task_evoked_pca_run_normalized_accuracy']:.2f}%，normalized separation ratio 由 "
                f"{sensitivity['baseline_run_normalized_separation_ratio']:.3f} 提高到 {sensitivity['task_evoked_pca_run_normalized_separation_ratio']:.3f}。",
            ]
        )
    lines.extend(
        [
            "",
            "## 诊断与解释边界",
            "",
            f"层级与归因最大闭合误差为 {diag['maximum_decomposition_identity_error_bits']:.3e} bits；按 PC 训练段极值或噪声协方差条件数触发质量标记的模型为 {diag['quality_flag_models']}/{diag['n_models']}。",
            "",
            "该方法有意把表示学习偏向任务设计解释的空间方向，因此若任务识别提高，只能说明这些方向使拟合动力学的 $\\Xi$ 构成更具任务可分性；不能据此断言任务诱发成分本身产生了协同因果。PCA 载荷按被试和任务独立估计，任务间差异同时包含投影方向差异。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    discovered = discover_inputs(args.rest_root, args.task_root)
    subjects = sorted(discovered)
    if args.max_subjects is not None:
        subjects = subjects[: int(args.max_subjects)]
    groups = load_yeo7_groups(args.labels, expected_parcels=500)
    baseline = load_baseline(args.baseline_summary, subjects)
    reusable = load_reusable_new_rows(args.output_dir / "summary.json", subjects)
    payloads = [
        (str(discovered[subject][task]), groups, subject, task, args.order, args.alpha)
        for subject in subjects
        for task in TASKS
        if (subject, task) not in reusable
    ]
    rows: list[dict[str, Any]] = list(reusable.values())
    progress = tqdm(total=len(payloads), desc="Task-evoked PC1 Phi attribution", unit="fit", mininterval=1.0)
    try:
        if int(args.workers) <= 1:
            for payload in payloads:
                rows.append(_worker(payload))
                progress.update(1)
        else:
            with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
                futures = [executor.submit(_worker, payload) for payload in payloads]
                for future in as_completed(futures):
                    rows.append(future.result())
                    progress.update(1)
    finally:
        progress.close()
    row_map = {(str(row["subject"]), str(row["condition"])): row for row in rows}
    summary, arrays = summarize(row_map, baseline, subjects)
    summary["config"]["workers"] = int(args.workers)
    summary["config"]["reused_rows"] = len(reusable)
    summary["config"]["computed_rows"] = len(payloads)
    summary["rows"] = sorted(rows, key=lambda row: (TASKS.index(str(row["condition"])), str(row["subject"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "arrays.npz",
        tasks=np.asarray(TASKS), subjects=np.asarray(subjects), networks=np.asarray(NETWORK_ORDER), **arrays,
    )
    write_report(summary, args.output_dir / "report.md")
    plot(summary, arrays, args.output_dir / "task_evoked_pc1_phi_comparison")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rest-root", type=Path, default=DEFAULT_REST_ROOT)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--labels", type=Path, default=default_yeo7_labels(500))
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args(argv)
    if args.order != 5 or not np.isclose(args.alpha, 10.0):
        raise ValueError("The controlled comparison is fixed to p=5 and alpha=10.")
    result = run(args)
    print(json.dumps(result["discriminability"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
