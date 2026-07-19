#!/usr/bin/env python3
"""Post-hoc run-normalize HCP Yeo7 Phi attribution and audit task separation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "hcp_schaefer500_yeo7_network_attribution" / "summary.json"
DEFAULT_OUTPUT = ROOT / "results" / "hcp_schaefer500_phi_attribution_run_normalized"
CONDITIONS = ("REST", "EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")
CONDITION_LABELS = ("REST", "Emotion", "Gambling", "Language", "Motor", "Relational", "Social", "WM")
NETWORKS = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
NETWORK_LABELS = ("Visual", "Somatomotor", "Dorsal attention", "Salience/ventral attention", "Limbic", "Control", "Default")


def _load(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    lookup = {(str(row["subject"]), str(row["condition"])): row for row in payload["rows"]}
    subjects = sorted({subject for subject, _ in lookup})
    missing = [(subject, condition) for subject in subjects for condition in CONDITIONS if (subject, condition) not in lookup]
    if missing:
        raise ValueError(f"Input lacks {len(missing)} paired subject-condition rows; first={missing[0]}")
    raw = np.asarray(
        [
            [[float(lookup[(subject, condition)]["total_phi_contribution"][network]) for network in NETWORKS]
             for condition in CONDITIONS]
            for subject in subjects
        ],
        dtype=float,
    )
    phi = np.asarray(
        [[float(lookup[(subject, condition)]["raw_phi"]) for condition in CONDITIONS] for subject in subjects],
        dtype=float,
    )
    if np.any(phi <= 0.0):
        raise ValueError("Run normalization is undefined because at least one raw Phi is non-positive.")
    closure = np.max(np.abs(raw.sum(axis=2) - phi))
    if closure > 1.0e-6:
        raise ValueError(f"Attribution does not close to raw Phi (maximum error={closure:.3g} bits).")
    return subjects, raw, raw / phi[:, :, None]


def _loso_nearest_centroid(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_subjects, n_conditions, _ = values.shape
    predictions = np.empty((n_subjects, n_conditions), dtype=int)
    for held_out in range(n_subjects):
        train = np.delete(values, held_out, axis=0)
        pooled = train.reshape(-1, train.shape[-1])
        mean = pooled.mean(axis=0)
        scale = pooled.std(axis=0, ddof=1)
        scale = np.where(scale > 1.0e-12, scale, 1.0)
        centroids = ((train - mean) / scale).mean(axis=0)
        test = (values[held_out] - mean) / scale
        distances = np.linalg.norm(test[:, None, :] - centroids[None, :, :], axis=2)
        predictions[held_out] = np.argmin(distances, axis=1)
    confusion = np.zeros((n_conditions, n_conditions), dtype=int)
    for truth in range(n_conditions):
        for predicted in predictions[:, truth]:
            confusion[truth, predicted] += 1
    return predictions, confusion


def _accuracy(predictions: np.ndarray) -> float:
    truth = np.broadcast_to(np.arange(predictions.shape[1]), predictions.shape)
    return float(np.mean(predictions == truth))


def _separation_ratio(values: np.ndarray) -> float:
    centroids = values.mean(axis=0)
    grand = centroids.mean(axis=0)
    between = float(np.mean(np.sum((centroids - grand) ** 2, axis=1)))
    within = float(np.mean(np.sum((values - centroids[None, :, :]) ** 2, axis=2)))
    return float(np.sqrt(between / within)) if within > 0.0 else float("inf")


def _bootstrap_accuracy_delta(
    raw_predictions: np.ndarray,
    normalized_predictions: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    truth = np.broadcast_to(np.arange(raw_predictions.shape[1]), raw_predictions.shape)
    raw_subject = np.mean(raw_predictions == truth, axis=1)
    normalized_subject = np.mean(normalized_predictions == truth, axis=1)
    delta = normalized_subject - raw_subject
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(replicates, len(delta)))
    return [float(value) for value in np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])]


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )


def _annotated_heatmap(
    fig: Any,
    axis: Any,
    matrix: np.ndarray,
    *,
    cmap: Any,
    label: str,
    fmt: str,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Any | None = None,
) -> None:
    image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    axis.set(
        xticks=np.arange(len(CONDITIONS)),
        xticklabels=CONDITION_LABELS,
        yticks=np.arange(len(NETWORKS)),
        yticklabels=NETWORK_LABELS,
        xlabel="State",
        ylabel="Yeo7 network",
    )
    axis.tick_params(axis="x", labelrotation=40, length=0)
    axis.tick_params(axis="y", length=0)
    norm = image.norm
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            rgba = cmap(norm(matrix[row, column]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            axis.text(column, row, format(matrix[row, column], fmt), ha="center", va="center",
                      fontsize=4.8, color="white" if luminance < 0.50 else "black")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82, pad=0.02)
    colorbar.set_label(label)


def _plot(raw: np.ndarray, normalized: np.ndarray, metrics: dict[str, Any], output: Path) -> None:
    _style()
    raw_mean = raw.mean(axis=0).T
    normalized_mean = 100.0 * normalized.mean(axis=0).T
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)
    _annotated_heatmap(
        fig, axes[0, 0], raw_mean, cmap=mpl.colormaps["YlOrBr"],
        label="Mean total $\\Phi$ attribution (bits)", fmt=".2f",
    )
    _annotated_heatmap(
        fig, axes[0, 1], normalized_mean, cmap=mpl.colormaps["RdBu_r"],
        label="Mean share of run $\\Phi$ (%)", fmt=".1f",
        norm=mpl.colors.TwoSlopeNorm(
            vmin=float(normalized_mean.min()), vcenter=100.0 / len(NETWORKS), vmax=float(normalized_mean.max())
        ),
    )
    raw_accuracy = 100.0 * metrics["raw"]["loso_accuracy"]
    normalized_accuracy = 100.0 * metrics["run_normalized"]["loso_accuracy"]
    raw_task_accuracy = 100.0 * metrics["task_only"]["raw"]["loso_accuracy"]
    normalized_task_accuracy = 100.0 * metrics["task_only"]["run_normalized"]["loso_accuracy"]
    x = np.arange(2)
    width = 0.34
    raw_bars = axes[1, 0].bar(
        x - width / 2, [raw_accuracy, raw_task_accuracy], width=width,
        color="#B8B8B8", edgecolor="none", label="Raw bits",
    )
    normalized_bars = axes[1, 0].bar(
        x + width / 2, [normalized_accuracy, normalized_task_accuracy], width=width,
        color="#4C78A8", edgecolor="none", label="Run-normalized share",
    )
    axes[1, 0].axhline(12.5, color="#555555", linestyle="--", linewidth=0.8)
    axes[1, 0].axhline(100.0 / 7.0, color="#888888", linestyle=":", linewidth=0.8)
    axes[1, 0].set(
        xticks=x, xticklabels=["REST + 7 tasks", "7 tasks only"],
        ylabel="LOSO state accuracy (%)", ylim=(0, max(35.0, normalized_accuracy + 8.0)),
    )
    for bar, value in zip(
        [*raw_bars, *normalized_bars],
        [raw_accuracy, raw_task_accuracy, normalized_accuracy, normalized_task_accuracy],
    ):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}", ha="center", fontsize=6)
    axes[1, 0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False)
    confusion = np.asarray(metrics["run_normalized"]["confusion"], dtype=float)
    confusion /= confusion.sum(axis=1, keepdims=True)
    image = axes[1, 1].imshow(confusion, cmap=mpl.colormaps["Blues"], vmin=0.0, vmax=1.0)
    axes[1, 1].set(
        xticks=np.arange(len(CONDITIONS)), xticklabels=CONDITION_LABELS,
        yticks=np.arange(len(CONDITIONS)), yticklabels=CONDITION_LABELS,
        xlabel="Predicted state", ylabel="True state",
    )
    axes[1, 1].tick_params(axis="x", labelrotation=40, length=0)
    axes[1, 1].tick_params(axis="y", length=0)
    for row in range(len(CONDITIONS)):
        for column in range(len(CONDITIONS)):
            value = confusion[row, column]
            axes[1, 1].text(column, row, f"{value:.2f}", ha="center", va="center",
                            fontsize=4.7, color="white" if value > 0.55 else "black")
    colorbar = fig.colorbar(image, ax=axes[1, 1], shrink=0.82, pad=0.02)
    colorbar.set_label("Fraction of subjects")
    for label, axis in zip("abcd", axes.flat):
        axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(input_path: Path, output_dir: Path, *, bootstrap_replicates: int = 10_000) -> dict[str, Any]:
    subjects, raw, normalized = _load(input_path)
    raw_predictions, raw_confusion = _loso_nearest_centroid(raw)
    normalized_predictions, normalized_confusion = _loso_nearest_centroid(normalized)
    raw_task_predictions, raw_task_confusion = _loso_nearest_centroid(raw[:, 1:, :])
    normalized_task_predictions, normalized_task_confusion = _loso_nearest_centroid(normalized[:, 1:, :])
    metrics = {
        "contract": {
            "treatment": "divide each subject-state Yeo7 attribution vector by that run's raw Phi",
            "fixed": "subjects, states, fitted dynamics, estimator, hierarchy/Shapley attribution, network order",
            "normalization": "p_sck = a_sck / sum_j(a_scj) = a_sck / Phi_sc",
            "primary_metric": "paired LOSO nearest-centroid state accuracy",
            "secondary_metric": "between-state / within-state RMS separation ratio",
        },
        "n_subjects": len(subjects),
        "n_states": len(CONDITIONS),
        "chance_accuracy": 1.0 / len(CONDITIONS),
        "maximum_share_closure_error": float(np.max(np.abs(normalized.sum(axis=2) - 1.0))),
        "raw": {
            "loso_accuracy": _accuracy(raw_predictions),
            "separation_ratio": _separation_ratio(raw),
            "confusion": raw_confusion.tolist(),
        },
        "run_normalized": {
            "loso_accuracy": _accuracy(normalized_predictions),
            "separation_ratio": _separation_ratio(normalized),
            "confusion": normalized_confusion.tolist(),
            "condition_network_mean_share": normalized.mean(axis=0).tolist(),
        },
        "task_only": {
            "n_states": len(CONDITIONS) - 1,
            "chance_accuracy": 1.0 / (len(CONDITIONS) - 1),
            "raw": {
                "loso_accuracy": _accuracy(raw_task_predictions),
                "separation_ratio": _separation_ratio(raw[:, 1:, :]),
                "confusion": raw_task_confusion.tolist(),
            },
            "run_normalized": {
                "loso_accuracy": _accuracy(normalized_task_predictions),
                "separation_ratio": _separation_ratio(normalized[:, 1:, :]),
                "confusion": normalized_task_confusion.tolist(),
            },
        },
        "normalized_minus_raw_accuracy": float(_accuracy(normalized_predictions) - _accuracy(raw_predictions)),
        "accuracy_delta_subject_bootstrap_95_ci": _bootstrap_accuracy_delta(
            raw_predictions, normalized_predictions, replicates=bootstrap_replicates, seed=20260719
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    raw_acc = 100.0 * metrics["raw"]["loso_accuracy"]
    norm_acc = 100.0 * metrics["run_normalized"]["loso_accuracy"]
    raw_task_acc = 100.0 * metrics["task_only"]["raw"]["loso_accuracy"]
    norm_task_acc = 100.0 * metrics["task_only"]["run_normalized"]["loso_accuracy"]
    ci = metrics["accuracy_delta_subject_bootstrap_95_ci"]
    report = (
        "# Phi 层级贡献的 run 内归一化对照\n\n"
        "唯一改动是对每名被试、每个状态的七网络 Phi 归因做 `a / Phi`；原始动力模型和归因结果均直接复用。\n\n"
        f"- 原始 bits 的 LOSO 八状态识别率：{raw_acc:.2f}%（机会水平 12.50%）。\n"
        f"- run 内归一化后的识别率：{norm_acc:.2f}%。\n"
        f"- 配对提升：{norm_acc - raw_acc:+.2f} 个百分点；被试 bootstrap 95% CI "
        f"[{100*ci[0]:.2f}, {100*ci[1]:.2f}] 个百分点。\n"
        f"- between/within RMS 分离比：原始 {metrics['raw']['separation_ratio']:.4f}，"
        f"归一化 {metrics['run_normalized']['separation_ratio']:.4f}。\n\n"
        f"仅比较七个任务态时，识别率由 {raw_task_acc:.2f}% 变为 {norm_task_acc:.2f}%（机会水平 14.29%），"
        f"分离比由 {metrics['task_only']['raw']['separation_ratio']:.4f} 变为 "
        f"{metrics['task_only']['run_normalized']['separation_ratio']:.4f}。\n\n"
        "归一化图表示每个网络占该 run 总 Phi 的份额，不再使用 bits；它只能判断层级构成是否更具任务特异性，"
        "不能替代原始 Phi 强度结果。\n"
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    _plot(raw, normalized, metrics, output_dir / "run_normalization_comparison")
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.input, args.output_dir, bootstrap_replicates=args.bootstrap_replicates), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
