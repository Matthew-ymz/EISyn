#!/usr/bin/env python3
"""Measure PEID Syn for the pure product map z = alpha * x * y."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_ALPHAS = (-5.0, -1.0, -0.1, -0.01, -0.001, 0.0, 0.001, 0.01, 0.1, 1.0, 5.0)
DEFAULT_RESULT_PATH = ROOT / "results" / "product_scale_peid" / "product_scale_peid.json"
DEFAULT_FIGURE_PATH = ROOT / "fig" / "product_scale_peid" / "product_scale_peid.png"
DEFAULT_TRANSPORT_RESULT_PATH = ROOT / "results" / "product_scale_peid" / "product_scale_transport_peid.json"
DEFAULT_TRANSPORT_FIGURE_PATH = ROOT / "fig" / "product_scale_peid" / "product_scale_transport_peid.png"


def _discretize(values: np.ndarray, bins: int) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    scale = max(1.0, float(np.max(np.abs(vector))))
    if float(np.ptp(vector)) <= 1e-12 * scale:
        return np.zeros(len(vector), dtype=int)
    edges = np.unique(np.quantile(vector, np.linspace(0.0, 1.0, int(bins) + 1)))
    if len(edges) <= 2:
        return np.zeros(len(vector), dtype=int)
    return np.digitize(vector, edges[1:-1], right=False).astype(int)


def _entropy(codes: np.ndarray) -> float:
    values = np.asarray(codes)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    _, counts = np.unique(values, axis=0, return_counts=True)
    probabilities = counts.astype(float) / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _mutual_information(source: np.ndarray, target: np.ndarray) -> float:
    left = np.asarray(source)
    right = np.asarray(target)
    if left.ndim == 1:
        left = left.reshape(-1, 1)
    if right.ndim == 1:
        right = right.reshape(-1, 1)
    return float(_entropy(left) + _entropy(right) - _entropy(np.column_stack([left, right])))


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()[:16]


def estimate_product_peid(
    *,
    alphas: Sequence[float],
    samples: int,
    bins: int,
    seed: int,
) -> list[dict[str, float | int | str]]:
    """Estimate the finest-partition two-source PEID residual for one seed."""
    rng = np.random.default_rng(int(seed))
    x = rng.uniform(-1.0, 1.0, size=int(samples))
    y = rng.uniform(-1.0, 1.0, size=int(samples))
    x_codes = _discretize(x, bins)
    y_codes = _discretize(y, bins)
    source_digest = _digest(np.column_stack([x, y]))
    rows: list[dict[str, float | int | str]] = []
    for alpha in alphas:
        z = float(alpha) * x * y
        z_codes = _discretize(z, bins)
        x_ei = _mutual_information(x_codes, z_codes)
        y_ei = _mutual_information(y_codes, z_codes)
        joint_ei = _mutual_information(np.column_stack([x_codes, y_codes]), z_codes)
        rows.append(
            {
                "alpha": float(alpha),
                "seed": int(seed),
                "x_ei": x_ei,
                "y_ei": y_ei,
                "joint_ei": joint_ei,
                "syn": float(joint_ei - x_ei - y_ei),
                "source_digest": source_digest,
            }
        )
    return rows


def estimate_product_transport_peid(
    *,
    alphas: Sequence[float],
    samples: int,
    noise_std: float,
    seed: int,
) -> list[dict[str, float | int | str]]:
    """Estimate noisy product-map PEID with the repository transport-map estimator."""
    from yrd.transport_map import summarize_two_source_synergy_transport_map

    rng = np.random.default_rng(int(seed))
    x = rng.uniform(-1.0, 1.0, size=(int(samples), 1))
    y = rng.uniform(-1.0, 1.0, size=(int(samples), 1))
    noise = float(noise_std) * rng.normal(size=(int(samples), 1))
    source_digest = _digest(np.column_stack([x, y]))
    noise_digest = _digest(noise)
    rows: list[dict[str, float | int | str]] = []
    for alpha in alphas:
        z = float(alpha) * x * y + noise
        values = summarize_two_source_synergy_transport_map(x, y, z)
        rows.append(
            {
                "alpha": float(alpha),
                "seed": int(seed),
                "x_ei": float(values["left_ei"]),
                "y_ei": float(values["right_ei"]),
                "joint_ei": float(values["joint_ei"]),
                "syn": float(values["syn"]),
                "source_digest": source_digest,
                "noise_digest": noise_digest,
            }
        )
    return rows


def _summarize(rows: Sequence[dict[str, float | int | str]]) -> list[dict[str, float]]:
    summary: list[dict[str, float]] = []
    for alpha in sorted({float(row["alpha"]) for row in rows}):
        selected = [row for row in rows if float(row["alpha"]) == alpha]
        item = {"alpha": alpha}
        for key in ("x_ei", "y_ei", "joint_ei", "syn"):
            values = np.asarray([float(row[key]) for row in selected])
            item[f"{key}_mean"] = float(np.mean(values))
            item[f"{key}_std"] = float(np.std(values))
        summary.append(item)
    return summary


def _plot_summary(summary: Sequence[dict[str, float]], path: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    alphas = np.asarray([row["alpha"] for row in summary], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.7), constrained_layout=True)
    colors = {"syn": "#D97732", "joint_ei": "#7068A8", "x_ei": "#4C78A8", "y_ei": "#5F8F6B"}

    def plot_metric(axis: object, key: str, label: str, marker: str, linestyle: str = "-") -> None:
        mean = np.asarray([row[f"{key}_mean"] for row in summary], dtype=float)
        std = np.asarray([row[f"{key}_std"] for row in summary], dtype=float)
        axis.plot(
            alphas,
            mean,
            color=colors[key],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            markersize=4.2,
            label=label,
        )
        axis.fill_between(alphas, mean - std, mean + std, color=colors[key], alpha=0.14, linewidth=0)

    plot_metric(axes[0], "syn", "PEID Syn", "o")
    axes[0].set_ylabel("Synergy (bits)")
    axes[0].set_title("a  Product-map PEID Syn", loc="left", fontweight="bold")

    plot_metric(axes[1], "joint_ei", "Joint EI", "o")
    plot_metric(axes[1], "x_ei", "EI(x)", "s", "--")
    plot_metric(axes[1], "y_ei", "EI(y)", "^", "--")
    axes[1].set_ylabel("Information (bits)")
    axes[1].set_title("b  PEID components", loc="left", fontweight="bold")

    for axis in axes:
        axis.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
        axis.set_xscale("symlog", linthresh=0.001)
        axis.set_xlabel("Product coefficient alpha")
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_transport_summary(summary: Sequence[dict[str, float]], path: Path, *, noise_std: float) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    alphas = np.asarray([row["alpha"] for row in summary], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.7), constrained_layout=True)
    colors = {"syn": "#D97732", "joint_ei": "#7068A8", "x_ei": "#4C78A8", "y_ei": "#5F8F6B"}

    def plot_metric(axis: object, key: str, label: str, marker: str, linestyle: str = "-") -> None:
        mean = np.asarray([row[f"{key}_mean"] for row in summary], dtype=float)
        std = np.asarray([row[f"{key}_std"] for row in summary], dtype=float)
        axis.plot(
            alphas, mean, color=colors[key], marker=marker, linestyle=linestyle,
            linewidth=1.8, markersize=4.2, label=label,
        )
        axis.fill_between(alphas, mean - std, mean + std, color=colors[key], alpha=0.14, linewidth=0)

    plot_metric(axes[0], "syn", "Transport-map Syn", "o")
    axes[0].set_ylabel("Synergy (bits)")
    axes[0].set_title(f"a  Noisy product-map Syn (noise std={noise_std:g})", loc="left", fontweight="bold")
    plot_metric(axes[1], "joint_ei", "Joint EI", "o")
    plot_metric(axes[1], "x_ei", "EI(x)", "s", "--")
    plot_metric(axes[1], "y_ei", "EI(y)", "^", "--")
    axes[1].set_ylabel("Information (bits)")
    axes[1].set_title("b  Transport-map PEID components", loc="left", fontweight="bold")
    for axis in axes:
        axis.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
        axis.set_xscale("symlog", linthresh=0.001)
        axis.set_xlabel("Product coefficient alpha")
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_product_scale_experiment(
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    seeds: Sequence[int] = tuple(range(12)),
    samples: int = 50000,
    bins: int = 12,
    result_path: Path = DEFAULT_RESULT_PATH,
    figure_path: Path = DEFAULT_FIGURE_PATH,
) -> dict[str, object]:
    rows = [
        row
        for seed in seeds
        for row in estimate_product_peid(alphas=alphas, samples=samples, bins=bins, seed=int(seed))
    ]
    summary = _summarize(rows)
    result = {
        "system": "z=alpha*x*y",
        "source_distribution": "independent_uniform[-1,1]",
        "target": "deterministic_pure_product",
        "estimator": "quantile_histogram",
        "samples_per_seed": int(samples),
        "bins": int(bins),
        "alphas": [float(alpha) for alpha in alphas],
        "seeds": [int(seed) for seed in seeds],
        "hypothesis": "PEID components are zero at alpha=0 and invariant under every nonzero alpha.",
        "rows": rows,
        "summary": summary,
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_summary(summary, figure_path)
    return {**result, "result_path": str(result_path)}


def run_product_transport_experiment(
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    seeds: Sequence[int] = tuple(range(12)),
    samples: int = 50000,
    noise_std: float = 0.05,
    result_path: Path = DEFAULT_TRANSPORT_RESULT_PATH,
    figure_path: Path = DEFAULT_TRANSPORT_FIGURE_PATH,
) -> dict[str, object]:
    rows = [
        row
        for seed in seeds
        for row in estimate_product_transport_peid(
            alphas=alphas,
            samples=samples,
            noise_std=noise_std,
            seed=int(seed),
        )
    ]
    summary = _summarize(rows)
    result = {
        "system": "z=alpha*x*y+epsilon",
        "source_distribution": "independent_uniform[-1,1]",
        "target": "alpha*x*y+gaussian_process_noise",
        "estimator": "transport_map",
        "noise_distribution": "gaussian",
        "noise_std": float(noise_std),
        "samples_per_seed": int(samples),
        "alphas": [float(alpha) for alpha in alphas],
        "seeds": [int(seed) for seed in seeds],
        "hypothesis": "Fixed additive noise breaks exact scale invariance: Syn rises with absolute alpha and approaches a high-SNR plateau.",
        "rows": rows,
        "summary": summary,
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_transport_summary(summary, figure_path, noise_std=noise_std)
    return {**result, "result_path": str(result_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--transport-map", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.transport_map:
        result_path = args.result_path if args.result_path != DEFAULT_RESULT_PATH else DEFAULT_TRANSPORT_RESULT_PATH
        figure_path = args.figure_path if args.figure_path != DEFAULT_FIGURE_PATH else DEFAULT_TRANSPORT_FIGURE_PATH
        result = run_product_transport_experiment(
            seeds=tuple(args.seeds),
            samples=args.samples,
            noise_std=args.noise_std,
            result_path=result_path,
            figure_path=figure_path,
        )
    else:
        result = run_product_scale_experiment(
            seeds=tuple(args.seeds),
            samples=args.samples,
            bins=args.bins,
            result_path=args.result_path,
            figure_path=args.figure_path,
        )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
