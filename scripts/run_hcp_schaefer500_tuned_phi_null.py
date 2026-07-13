#!/usr/bin/env python3
"""Compare tuned Schaefer-500 state-model PhiEID to circular-shift nulls."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hcp_lausanne_phi_eid_pilot import (
    circular_shift_null,
    gaussian_phi_from_linear_transition,
    standardize_columns,
)


DEFAULT_DATA_ROOT = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30"
DEFAULT_HISTORY = ROOT / "docs" / "log" / "hcp_schaefer500_tuned_dynamics" / "run_history.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "hcp_schaefer500_tuned_phi_null" / "summary.json"
DEFAULT_REPORT = ROOT / "docs" / "log" / "hcp_schaefer500_tuned_phi_null" / "run_report.md"
DEFAULT_SUBJECTS = ("sub-100206", "sub-100307", "sub-100408", "sub-100610", "sub-101006")


def fit_state_phi(
    series: np.ndarray,
    *,
    alpha: float,
    development_end: int,
    covariance_ridge: float = 1.0e-6,
) -> dict[str, Any]:
    """Fit the selected one-step state Ridge model and compute its PhiEID."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 2 or development_end > values.shape[0] or development_end < 3:
        raise ValueError("series must contain a valid two-dimensional development segment.")
    source = values[: development_end - 1]
    target = values[1:development_end]
    source_z, mean, scale = standardize_columns(source)
    target_z = (target - mean.reshape(1, -1)) / scale.reshape(1, -1)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(source_z, target_z)
    residual = target_z - model.predict(source_z)
    noise_covariance = np.atleast_2d(np.cov(residual, rowvar=False, bias=False))
    phi = gaussian_phi_from_linear_transition(
        model.coef_, noise_covariance, ridge=float(covariance_ridge)
    )
    return {
        "transition": model.coef_,
        "noise_covariance": noise_covariance,
        "phi": phi,
    }


def summarize_null(*, observed: float, null_values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(null_values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("null_values must be a non-empty finite vector.")
    return {
        "n_null": int(len(values)),
        "null_mean": float(values.mean()),
        "null_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "observed_minus_null_mean": float(observed - values.mean()),
        "empirical_p_ge_observed": float((1 + np.sum(values >= observed)) / (len(values) + 1)),
    }


def load_state_ridge_selections(history: Path, *, subjects: Sequence[str]) -> dict[str, float]:
    """Load fixed p=1 state-Ridge alphas from completed tuning records."""
    wanted = set(subjects)
    selected: dict[str, float] = {}
    for line in Path(history).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        subject = record.get("run_name")
        if record.get("status") != "completed" or subject not in wanted:
            continue
        model = record["models"]["tuned_state_ridge_p1"]
        config = model["selected"]
        if int(config["order"]) != 1:
            raise ValueError(f"{subject} selected a non-state p=1 configuration.")
        selected[str(subject)] = float(config["alpha"])
    missing = wanted.difference(selected)
    if missing:
        raise ValueError(f"No completed tuned_state_ridge_p1 selection for: {sorted(missing)}")
    return {subject: selected[subject] for subject in subjects}


def _subject_seed(*, seed: int, subject: str, replicate: int) -> int:
    return int(np.random.SeedSequence([int(seed), int(subject.removeprefix("sub-")), int(replicate)]).generate_state(1)[0])


def _timed_state_phi(series: np.ndarray, *, alpha: float, development_end: int) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    result = fit_state_phi(series, alpha=alpha, development_end=development_end)
    return result, float(time.perf_counter() - start)


def _analyze_series(
    series: np.ndarray,
    *,
    subject: str,
    alpha: float,
    development_end: int,
    null_replicates: int,
    seed: int,
) -> dict[str, Any]:
    observed_fit, observed_seconds = _timed_state_phi(series, alpha=alpha, development_end=development_end)
    null_phi: list[float] = []
    null_seconds: list[float] = []
    for replicate in range(null_replicates):
        null_series = circular_shift_null(series, seed=_subject_seed(seed=seed, subject=subject, replicate=replicate))
        null_fit, elapsed = _timed_state_phi(null_series, alpha=alpha, development_end=development_end)
        null_phi.append(float(null_fit["phi"]["raw_phi"]))
        null_seconds.append(elapsed)
    observed = float(observed_fit["phi"]["raw_phi"])
    return {
        "subject": subject,
        "alpha": float(alpha),
        "order": 1,
        "development_end": int(development_end),
        "observed_raw_phi": observed,
        "null_raw_phi": null_phi,
        "null_comparison": summarize_null(observed=observed, null_values=np.asarray(null_phi)),
        "timing_seconds": {
            "observed_fit": observed_seconds,
            "null_fits": null_seconds,
            "mean_per_fit": float(np.mean([observed_seconds, *null_seconds])),
        },
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observed = np.asarray([row["observed_raw_phi"] for row in rows], dtype=float)
    differences = np.asarray([row["null_comparison"]["observed_minus_null_mean"] for row in rows], dtype=float)
    per_fit = np.asarray([row["timing_seconds"]["mean_per_fit"] for row in rows], dtype=float)
    median_per_fit = float(np.median(per_fit))
    return {
        "observed_raw_phi_mean": float(observed.mean()),
        "observed_raw_phi_median": float(np.median(observed)),
        "observed_minus_null_mean_mean": float(differences.mean()),
        "observed_minus_null_mean_median": float(np.median(differences)),
        "subjects_observed_above_null_mean": int(np.sum(differences > 0.0)),
        "median_seconds_per_phi_fit": median_per_fit,
        "runtime_estimate_seconds": {
            "five_subjects_20_nulls": float(len(rows) * 21 * median_per_fit),
            "five_subjects_100_nulls": float(len(rows) * 101 * median_per_fit),
        },
    }


def _write_report(summary: Mapping[str, Any], report: Path) -> None:
    rows = summary["rows"]
    lines = [
        "# Schaefer-500 调优状态模型 PhiEID 与 circular-shift null",
        "",
        "每名被试固定采用预测调优阶段选出的 `tuned_state_ridge_p1` alpha，在前 900 点重训。"
        "每个 null 对所有 ROI 分别进行 circular shift，且不在 null 或测试段重新选择超参数。",
        "",
        "| 被试 | alpha | observed raw Phi（bits） | null mean | observed − null mean | empirical p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        null = row["null_comparison"]
        lines.append(
            f"| {row['subject']} | {row['alpha']:.0f} | {row['observed_raw_phi']:.6f} | "
            f"{null['null_mean']:.6f} | {null['observed_minus_null_mean']:.6f} | "
            f"{null['empirical_p_ge_observed']:.6f} |"
        )
    aggregate = summary["aggregate"]
    lines.extend(
        [
            "",
            f"完成的 20-null pilot 总耗时：{summary['elapsed_seconds']:.2f} 秒。",
            f"中位单次 Phi 拟合时间：{aggregate['median_seconds_per_phi_fit']:.3f} 秒。",
            f"按实测速度，五名被试 100 null 的预计耗时："
            f"{aggregate['runtime_estimate_seconds']['five_subjects_100_nulls']:.1f} 秒。",
            "",
            "PhiEID 保持既有 500D one-step Gaussian log-det 定义；结果是初步 null screening，"
            "不替代运动/去趋势等独立敏感性分析。",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_series(
    series_by_subject: Mapping[str, np.ndarray],
    *,
    selections: Mapping[str, float],
    development_end: int,
    null_replicates: int,
    output: Path,
    report: Path,
    seed: int = 20260713,
) -> dict[str, Any]:
    if null_replicates < 1:
        raise ValueError("null_replicates must be positive.")
    start = time.perf_counter()
    rows = [
        _analyze_series(
            np.asarray(series_by_subject[subject], dtype=float),
            subject=subject,
            alpha=float(selections[subject]),
            development_end=development_end,
            null_replicates=null_replicates,
            seed=seed,
        )
        for subject in selections
    ]
    summary = {
        "n_subjects": len(rows),
        "config": {
            "model": "tuned_state_ridge_p1",
            "development_end": int(development_end),
            "null_model": "roi_wise_circular_shift",
            "null_replicates": int(null_replicates),
            "covariance_diagonal_ridge": 1.0e-6,
            "test_segment_used": False,
        },
        "rows": rows,
        "aggregate": _aggregate(rows),
        "elapsed_seconds": float(time.perf_counter() - start),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(summary, Path(report))
    return summary


def _load_subject_series(path: Path) -> np.ndarray:
    values = np.asarray(loadmat(path)["Schaefer500"], dtype=float)
    if values.ndim != 2 or values.shape[1] != 500 or not np.isfinite(values).all():
        raise ValueError(f"Expected finite [time, 500] Schaefer500 data in {path}.")
    return values


def resolve_subject_mat(data_root: Path, subject: str) -> Path:
    matches = sorted((Path(data_root) / subject).glob("*.mat"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one MAT file for {subject}, found {len(matches)}.")
    return matches[0]


def run(
    *,
    data_root: Path,
    history: Path,
    subjects: Sequence[str],
    null_replicates: int,
    output: Path,
    report: Path,
    seed: int,
) -> dict[str, Any]:
    selections = load_state_ridge_selections(history, subjects=subjects)
    series_by_subject = {
        subject: _load_subject_series(resolve_subject_mat(data_root, subject))
        for subject in subjects
    }
    return run_series(
        series_by_subject,
        selections=selections,
        development_end=900,
        null_replicates=null_replicates,
        output=output,
        report=report,
        seed=seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    parser.add_argument("--null-replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    subjects = tuple(item.strip() for item in args.subjects.split(",") if item.strip())
    summary = run(
        data_root=args.data_root,
        history=args.history,
        subjects=subjects,
        null_replicates=args.null_replicates,
        output=args.output,
        report=args.report,
        seed=args.seed,
    )
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    print(f"Saved: {args.output}")
    return 0


__all__ = ["circular_shift_null", "fit_state_phi", "load_state_ridge_selections", "resolve_subject_mat", "run_series", "summarize_null"]


if __name__ == "__main__":
    raise SystemExit(main())
