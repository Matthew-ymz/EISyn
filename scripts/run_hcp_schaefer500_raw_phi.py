#!/usr/bin/env python3
"""Fit raw Schaefer-500 REST1 time series and compute observed PhiEID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.io import loadmat

from scripts.run_hcp_lausanne_phi_eid_pilot import fit_ridge_transition, make_lagged_samples


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30"
DEFAULT_OUTPUT = ROOT / "results" / "hcp_schaefer500_raw_phi" / "summary.json"


def load_subject_series(path: Path, *, expected_parcels: int = 500) -> np.ndarray:
    values = np.asarray(loadmat(path)["Schaefer500"])
    if values.ndim != 2 or values.shape[1] != expected_parcels:
        raise ValueError(f"Expected [time, {expected_parcels}] Schaefer500 data, got {values.shape} in {path}.")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite Schaefer500 values in {path}.")
    return values


def analyze_subject(
    series: np.ndarray,
    *,
    subject: str,
    ridge_alpha: float,
    ridge: float,
) -> dict[str, object]:
    source, target = make_lagged_samples(np.asarray(series), tau=1)
    fit = fit_ridge_transition(source, target, alpha=ridge_alpha, ridge=ridge)
    metrics = fit["metrics"]
    phi = fit["phi"]
    return {
        "subject": subject,
        "shape": list(series.shape),
        "preprocessing": "none_explicit",
        "model_internal_scaling": "train_segment_columnwise_zscore",
        "ridge_alpha": float(ridge_alpha),
        "covariance_diagonal_ridge": float(ridge),
        "raw_phi": float(phi["raw_phi"]),
        "joint_ei": float(phi["joint_ei"]),
        "singleton_ei_sum": float(phi["singleton_ei_sum"]),
        "validation_rmse": float(metrics["rmse"]),
        "persistence_rmse": float(metrics["persistence_rmse"]),
        "rmse_over_persistence": float(metrics["skill_ratio"]),
        "validation_correlation": float(metrics["corr"]),
    }


def run(data_root: Path, output: Path, *, ridge_alpha: float, ridge: float) -> dict[str, object]:
    files = sorted(data_root.glob("sub-*/*.mat"))
    if not files:
        raise FileNotFoundError(f"No subject MAT files found under {data_root}.")
    rows = []
    for index, path in enumerate(files, start=1):
        subject = path.parent.name
        print(f"[{index}/{len(files)}] {subject}", flush=True)
        rows.append(
            analyze_subject(
                load_subject_series(path),
                subject=subject,
                ridge_alpha=ridge_alpha,
                ridge=ridge,
            )
        )
    phi = np.asarray([row["raw_phi"] for row in rows], dtype=float)
    skill = np.asarray([row["rmse_over_persistence"] for row in rows], dtype=float)
    summary = {
        "data_root": str(data_root),
        "n_subjects": len(rows),
        "config": {
            "explicit_preprocessing": "none",
            "tau": 1,
            "model": "ridge_var1",
            "ridge_alpha": float(ridge_alpha),
            "covariance_diagonal_ridge": float(ridge),
        },
        "aggregate": {
            "raw_phi_mean": float(phi.mean()),
            "raw_phi_median": float(np.median(phi)),
            "raw_phi_min": float(phi.min()),
            "raw_phi_max": float(phi.max()),
            "rmse_over_persistence_mean": float(skill.mean()),
            "subjects_better_than_persistence": int(np.sum(skill < 1.0)),
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args.data_root, args.output, ridge_alpha=args.ridge_alpha, ridge=args.ridge)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
