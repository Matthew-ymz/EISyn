#!/usr/bin/env python3
"""Verify single-source EI invariance under invertible target transforms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Callable, Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.TM.transport_map_density import estimate_mutual_information_transport_map


SCALE_VALUES = (-10.0, -2.0, -0.1, 0.1, 2.0, 10.0)
DEFAULT_RESULT_PATH = ROOT / "results" / "ei_target_transform_invariance" / "ei_target_transform_invariance.json"
DEFAULT_FIGURE_PATH = ROOT / "fig" / "ei_target_transform_invariance" / "ei_target_transform_invariance.png"


def analytic_gaussian_ei(*, noise_std: float) -> float:
    """Return I(S; S + epsilon) in bits for unit-variance Gaussian S."""
    if noise_std <= 0.0:
        raise ValueError("noise_std must be positive.")
    return float(0.5 * np.log2(1.0 + 1.0 / float(noise_std) ** 2))


def _discretize_quantile(values: np.ndarray, bins: int) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
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
    left = np.asarray(source).reshape(-1, 1)
    right = np.asarray(target).reshape(-1, 1)
    return float(_entropy(left) + _entropy(right) - _entropy(np.column_stack([left, right])))


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()[:16]


def _transform_specs() -> list[tuple[str, str, str, float | None, Callable[[np.ndarray], np.ndarray]]]:
    specs: list[tuple[str, str, str, float | None, Callable[[np.ndarray], np.ndarray]]] = [
        ("identity", "Identity", "invertible", 1.0, lambda z: z),
    ]
    specs.extend(
        (
            f"scale_{scale:g}",
            f"c={scale:g}",
            "invertible_linear",
            float(scale),
            lambda z, scale=scale: float(scale) * z,
        )
        for scale in SCALE_VALUES
    )
    specs.extend(
        [
            (
                "cubic_monotone",
                "z + 0.2z^3",
                "invertible_nonlinear",
                None,
                lambda z: z + 0.2 * z**3,
            ),
            ("square_noninvertible", "z^2", "noninvertible", None, lambda z: z**2),
        ]
    )
    return specs


def estimate_transform_ei(
    *,
    samples: int,
    bins: int,
    noise_std: float,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(int(seed))
    source = rng.normal(size=(int(samples), 1))
    noise = float(noise_std) * rng.normal(size=(int(samples), 1))
    target = source + noise
    source_codes = _discretize_quantile(source, bins)
    sample_digest = _digest(np.column_stack([source, noise]))
    exact_ei = analytic_gaussian_ei(noise_std=noise_std)
    rows: list[dict[str, object]] = []
    for name, label, kind, scale, transform in _transform_specs():
        transformed = transform(target)
        target_codes = _discretize_quantile(transformed, bins)
        histogram_ei = _mutual_information(source_codes, target_codes)
        transport_ei = float(estimate_mutual_information_transport_map(source, transformed)["mi_hat"])
        rows.append(
            {
                "transform": name,
                "label": label,
                "kind": kind,
                "scale": scale,
                "seed": int(seed),
                "analytic_ei": exact_ei if kind != "noninvertible" else None,
                "histogram_ei": histogram_ei,
                "transport_ei": transport_ei,
                "sample_digest": sample_digest,
            }
        )
    return rows


def _summarize(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    order = [spec[0] for spec in _transform_specs()]
    summary: list[dict[str, object]] = []
    for name in order:
        selected = [row for row in rows if row["transform"] == name]
        first = selected[0]
        item: dict[str, object] = {
            "transform": name,
            "label": first["label"],
            "kind": first["kind"],
            "scale": first["scale"],
            "analytic_ei": first["analytic_ei"],
        }
        for key in ("histogram_ei", "transport_ei"):
            values = np.asarray([float(row[key]) for row in selected])
            item[f"{key}_mean"] = float(np.mean(values))
            item[f"{key}_std"] = float(np.std(values))
        summary.append(item)
    return summary


def _plot_summary(summary: Sequence[dict[str, object]], path: Path, *, exact_ei: float) -> None:
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
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8), constrained_layout=True)
    colors = {"analytic": "#555555", "histogram": "#4C78A8", "transport": "#D97732"}

    scales = [row for row in summary if row["kind"] == "invertible_linear"]
    x_values = np.asarray([float(row["scale"]) for row in scales])
    axes[0].axhline(exact_ei, color=colors["analytic"], linestyle="--", linewidth=1.6, label="Analytic EI")
    for prefix, label, color, marker in (
        ("histogram_ei", "Quantile histogram", colors["histogram"], "s"),
        ("transport_ei", "Transport map", colors["transport"], "o"),
    ):
        mean = np.asarray([float(row[f"{prefix}_mean"]) for row in scales])
        std = np.asarray([float(row[f"{prefix}_std"]) for row in scales])
        axes[0].plot(x_values, mean, color=color, marker=marker, linewidth=1.8, markersize=4.2, label=label)
        axes[0].fill_between(x_values, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
    axes[0].set_xscale("symlog", linthresh=0.1)
    axes[0].set_xlabel("Nonzero target scale c")
    axes[0].set_ylabel("EI (bits)")
    axes[0].set_title("a  Invertible linear target scaling", loc="left", fontweight="bold")

    selected_names = ("identity", "cubic_monotone", "square_noninvertible")
    selected = [next(row for row in summary if row["transform"] == name) for name in selected_names]
    positions = np.arange(len(selected), dtype=float)
    width = 0.24
    axes[1].bar(
        positions - width,
        [exact_ei, exact_ei, np.nan],
        width,
        color=colors["analytic"],
        label="Analytic EI",
        alpha=0.78,
    )
    for offset, prefix, label, color in (
        (0.0, "histogram_ei", "Quantile histogram", colors["histogram"]),
        (width, "transport_ei", "Transport map", colors["transport"]),
    ):
        axes[1].bar(
            positions + offset,
            [float(row[f"{prefix}_mean"]) for row in selected],
            width,
            yerr=[float(row[f"{prefix}_std"]) for row in selected],
            color=color,
            label=label,
            capsize=2.5,
        )
    axes[1].set_xticks(positions, ["Identity", "Invertible cubic", "Square"])
    axes[1].set_ylabel("EI (bits)")
    axes[1].set_title("b  Invertible and noninvertible transforms", loc="left", fontweight="bold")

    for axis in axes:
        axis.grid(True, axis="y", alpha=0.22, linewidth=0.7)
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    *,
    seeds: Sequence[int] = tuple(range(12)),
    samples: int = 50000,
    bins: int = 12,
    noise_std: float = 0.5,
    result_path: Path = DEFAULT_RESULT_PATH,
    figure_path: Path = DEFAULT_FIGURE_PATH,
) -> dict[str, object]:
    rows = [
        row
        for seed in seeds
        for row in estimate_transform_ei(samples=samples, bins=bins, noise_std=noise_std, seed=int(seed))
    ]
    summary = _summarize(rows)
    exact_ei = analytic_gaussian_ei(noise_std=noise_std)
    result = {
        "system": "gaussian_additive_channel",
        "equation": "Z=S+epsilon",
        "source_distribution": "S~N(0,1)",
        "noise_distribution": f"epsilon~N(0,{float(noise_std) ** 2:g})",
        "analytic_ei_bits": exact_ei,
        "samples_per_seed": int(samples),
        "bins": int(bins),
        "seeds": [int(seed) for seed in seeds],
        "scale_values": [float(scale) for scale in SCALE_VALUES],
        "rows": rows,
        "summary": summary,
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_summary(summary, figure_path, exact_ei=exact_ei)
    return {**result, "result_path": str(result_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        seeds=tuple(args.seeds),
        samples=args.samples,
        bins=args.bins,
        noise_std=args.noise_std,
        result_path=args.result_path,
        figure_path=args.figure_path,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
