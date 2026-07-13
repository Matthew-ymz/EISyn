#!/usr/bin/env python3
"""Controlled diagnostics for elevated Schaefer-500 circular-shift null PhiEID."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hcp_schaefer500_tuned_phi_null import (
    DEFAULT_DATA_ROOT,
    _load_subject_series,
    fit_state_phi,
    resolve_subject_mat,
)


DEFAULT_OUTPUT = ROOT / "results" / "hcp_schaefer500_tuned_phi_null_diagnostic" / "summary.json"
DEFAULT_REPORT = ROOT / "docs" / "log" / "hcp_schaefer500_tuned_phi_null_diagnostic" / "run_report.md"

def global_circular_shift(series: np.ndarray, *, seed: int) -> tuple[np.ndarray, int]:
    """Apply one shared non-zero temporal shift to every ROI."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("series must be a [time, roi] matrix with at least two rows.")
    offset = int(np.random.default_rng(seed).integers(1, values.shape[0]))
    return np.roll(values, offset, axis=0), offset


def independent_circular_shift(series: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply a distinct non-zero temporal shift to every ROI."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("series must be a [time, roi] matrix with at least two rows.")
    offsets = np.random.default_rng(seed).integers(1, values.shape[0], size=values.shape[1])
    shifted = np.empty_like(values)
    for index, offset in enumerate(offsets):
        shifted[:, index] = np.roll(values[:, index], int(offset))
    return shifted, offsets


def independent_phase_surrogate(series: np.ndarray, *, seed: int) -> np.ndarray:
    """Preserve each ROI Fourier amplitude spectrum with independent random phases."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 2:
        raise ValueError("series must be a [time, roi] matrix.")
    spectrum = np.fft.rfft(values, axis=0)
    randomized = spectrum.copy()
    upper = spectrum.shape[0] - (1 if values.shape[0] % 2 == 0 else 0)
    if upper > 1:
        phases = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, size=(upper - 1, values.shape[1]))
        randomized[1:upper] *= np.exp(1j * phases)
    return np.fft.irfft(randomized, n=values.shape[0], axis=0)


def covariance_diagnostics(covariance: np.ndarray) -> dict[str, float]:
    """Return stable spectral and correlation summaries for a covariance matrix."""
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square.")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.maximum(np.linalg.eigvalsh(symmetric), 1.0e-12)
    weights = eigenvalues / eigenvalues.sum()
    effective_rank = float(np.exp(-np.sum(weights * np.log(weights))))
    scale = np.sqrt(np.maximum(np.diag(symmetric), 1.0e-12))
    correlation = symmetric / np.outer(scale, scale)
    upper = correlation[np.triu_indices_from(correlation, k=1)]
    return {
        "effective_rank": effective_rank,
        "condition_number": float(eigenvalues.max() / eigenvalues.min()),
        "logdet_natural": float(np.log(eigenvalues).sum()),
        "mean_abs_offdiagonal_correlation": float(np.mean(np.abs(upper))) if len(upper) else 0.0,
    }


def _mean_lag1_autocorrelation(series: np.ndarray) -> float:
    values = np.asarray(series, dtype=float)
    left = values[:-1]
    right = values[1:]
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    denominator = np.sqrt(np.sum(left**2, axis=0) * np.sum(right**2, axis=0))
    correlation = np.divide(np.sum(left * right, axis=0), denominator, out=np.zeros_like(denominator), where=denominator > 1.0e-12)
    return float(correlation.mean())


def _fit_record(series: np.ndarray, *, alpha: float, development_end: int) -> dict[str, Any]:
    values = np.asarray(series, dtype=float)
    start = time.perf_counter()
    fit = fit_state_phi(values, alpha=alpha, development_end=development_end)
    elapsed = float(time.perf_counter() - start)
    source_covariance = np.cov(values[: development_end - 1], rowvar=False, bias=False)
    phi = fit["phi"]
    return {
        "raw_phi": float(phi["raw_phi"]),
        "joint_ei": float(phi["joint_ei"]),
        "singleton_ei_sum": float(phi["singleton_ei_sum"]),
        "mean_lag1_autocorrelation": _mean_lag1_autocorrelation(values[:development_end]),
        "state_covariance": covariance_diagnostics(source_covariance),
        "residual_covariance": covariance_diagnostics(fit["noise_covariance"]),
        "elapsed_seconds": elapsed,
    }


def _paired_surrogate_records(
    series: np.ndarray,
    *,
    alpha: float,
    development_end: int,
    seeds: Sequence[int],
    constructor: Callable[[np.ndarray], tuple[np.ndarray, object]],
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        surrogate, metadata = constructor(np.asarray(series, dtype=float), seed=int(seed))
        row = _fit_record(surrogate, alpha=alpha, development_end=development_end)
        row["seed"] = int(seed)
        row["surrogate_metadata"] = metadata.tolist() if isinstance(metadata, np.ndarray) else metadata
        rows.append(row)
    return rows


def _phase_constructor(series: np.ndarray, *, seed: int) -> tuple[np.ndarray, None]:
    return independent_phase_surrogate(series, seed=seed), None


def _paired_difference(observed: float, rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    values = np.asarray([row["raw_phi"] for row in rows], dtype=float)
    delta = float(observed - values.mean())
    return {
        "observed_minus_null_mean": delta,
        "null_mean": float(values.mean()),
        "null_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "sign_consistency_observed_greater": float(np.mean(observed > values)),
    }


def _write_report(summary: dict[str, Any], report: Path) -> None:
    observed = summary["observed"]
    lines = [
        "# Schaefer-500 circular-shift null PhiEID 单被试诊断",
        "",
        "数据为 `sub-100206` 的前 900 点；所有 surrogate 对比固定 Ridge alpha=1000，"
        "Gaussian log-det signed raw PhiEID 与训练预算完全一致。",
        "",
        "| 条件 | null Phi 均值（bits） | observed − null（bits） | observed > null 比例 |",
        "|---|---:|---:|---:|",
    ]
    for name, difference in summary["surrogate_comparison"]["observed_minus_surrogate"].items():
        lines.append(
            f"| {name} | {difference['null_mean']:.6f} | {difference['observed_minus_null_mean']:.6f} | "
            f"{difference['sign_consistency_observed_greater']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Observed raw Phi={observed['raw_phi']:.6f} bits；"
            f"mean parcel lag-1 autocorrelation={observed['mean_lag1_autocorrelation']:.6f}。",
            "",
            "维度和 alpha sweep 是单因素敏感性分析；单被试结果用于定位估计器/ surrogate 行为，"
            "不构成人群推断。完整数值、协方差谱与耗时见 JSON。",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostic(
    series: np.ndarray,
    *,
    alpha: float,
    development_end: int,
    surrogate_seeds: Sequence[int],
    dimensions: Sequence[int],
    alpha_values: Sequence[float],
    output: Path,
    report: Path,
) -> dict[str, Any]:
    values = np.asarray(series, dtype=float)
    if values.ndim != 2 or development_end > len(values):
        raise ValueError("series must include the requested development segment.")
    if max(dimensions) > values.shape[1]:
        raise ValueError("dimension sweep exceeds the available parcel count.")
    start = time.perf_counter()
    observed = _fit_record(values, alpha=alpha, development_end=development_end)
    conditions = {
        "global_circular_shift": _paired_surrogate_records(
            values, alpha=alpha, development_end=development_end, seeds=surrogate_seeds, constructor=global_circular_shift
        ),
        "independent_circular_shift": _paired_surrogate_records(
            values, alpha=alpha, development_end=development_end, seeds=surrogate_seeds, constructor=independent_circular_shift
        ),
        "independent_phase_surrogate": _paired_surrogate_records(
            values, alpha=alpha, development_end=development_end, seeds=surrogate_seeds, constructor=_phase_constructor
        ),
    }
    rng = np.random.default_rng(20260713)
    permutation = rng.permutation(values.shape[1])
    dimension_sweep = []
    independent_for_sweep, offsets = independent_circular_shift(values, seed=int(surrogate_seeds[0]))
    for dimension in dimensions:
        selected = permutation[: int(dimension)]
        observed_dimension = _fit_record(values[:, selected], alpha=alpha, development_end=development_end)
        null_dimension = _fit_record(independent_for_sweep[:, selected], alpha=alpha, development_end=development_end)
        dimension_sweep.append(
            {
                "dimension": int(dimension),
                "observed": observed_dimension,
                "independent_circular_shift": null_dimension,
                "observed_minus_null": float(observed_dimension["raw_phi"] - null_dimension["raw_phi"]),
                "nested_subset": True,
                "null_offset_summary": {"min": int(offsets[selected].min()), "max": int(offsets[selected].max())},
            }
        )
    alpha_sweep = []
    for value in alpha_values:
        observed_alpha = _fit_record(values, alpha=float(value), development_end=development_end)
        null_alpha = _fit_record(independent_for_sweep, alpha=float(value), development_end=development_end)
        alpha_sweep.append(
            {
                "alpha": float(value),
                "observed": observed_alpha,
                "independent_circular_shift": null_alpha,
                "observed_minus_null": float(observed_alpha["raw_phi"] - null_alpha["raw_phi"]),
                "paired_null_seed": int(surrogate_seeds[0]),
            }
        )
    summary = {
        "subject": "sub-100206",
        "config": {
            "development_end": int(development_end),
            "alpha_for_surrogate_and_dimension": float(alpha),
            "surrogate_seeds": [int(seed) for seed in surrogate_seeds],
            "dimensions": [int(dimension) for dimension in dimensions],
            "alpha_values": [float(value) for value in alpha_values],
            "test_segment_used": False,
            "phi_definition": "existing_500d_onestep_gaussian_logdet_signed_raw_phi",
        },
        "observed": observed,
        "surrogate_comparison": {
            **conditions,
            "observed_minus_surrogate": {
                name: _paired_difference(observed["raw_phi"], rows) for name, rows in conditions.items()
            },
        },
        "dimension_sweep": dimension_sweep,
        "alpha_sweep": alpha_sweep,
        "elapsed_seconds": float(time.perf_counter() - start),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(summary, Path(report))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--subject", default="sub-100206")
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--development-end", type=int, default=900)
    parser.add_argument("--surrogate-seeds", default="101,202,303")
    parser.add_argument("--dimensions", default="50,200,500")
    parser.add_argument("--alpha-values", default="100,1000,10000")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    series = _load_subject_series(resolve_subject_mat(args.data_root, args.subject))
    summary = run_diagnostic(
        series,
        alpha=args.alpha,
        development_end=args.development_end,
        surrogate_seeds=_parse_ints(args.surrogate_seeds),
        dimensions=_parse_ints(args.dimensions),
        alpha_values=_parse_floats(args.alpha_values),
        output=args.output,
        report=args.report,
    )
    print(json.dumps({"elapsed_seconds": summary["elapsed_seconds"], "observed_raw_phi": summary["observed"]["raw_phi"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
