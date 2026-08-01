#!/usr/bin/env python3
"""Scan pairwise Gaussian-MMI PhiR on Schaefer100 DMF BOLD-like traces."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module, resolve_path


DEFAULT_SOURCE = ROOT / "results" / "dmf_schaefer100" / "source" / "group_mean_native_mean_rate.npz"
DEFAULT_OUTPUT = ROOT / "results" / "dmf_schaefer100" / "full" / "observational_phi_r.npz"
DEFAULT_SUMMARY = ROOT / "results" / "dmf_schaefer100" / "full" / "observational_phi_r_summary.json"
DEFAULT_STATUS = ROOT / "docs" / "log" / "dmf_schaefer100_phi_r_progress.json"
NONNEGATIVE_TOLERANCE_BITS = 1.0e-10


def atomic_savez(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_status(
    path: Path,
    *,
    phase: str,
    current: int,
    total: int,
    started: float,
    baseline: int = 0,
    metrics: dict[str, object] | None = None,
    message: str | None = None,
) -> None:
    elapsed = time.monotonic() - started
    completed = max(0, current - baseline)
    rate = completed / elapsed if elapsed > 0 else 0.0
    atomic_json(
        path,
        {
            "phase": phase,
            "current": current,
            "total": total,
            "unit": "seed-G condition",
            "elapsed_seconds": elapsed,
            "eta_seconds": (total - current) / rate if rate > 0 else None,
            "metrics": metrics or {},
            "message": message,
            "updated_at": time.time(),
        },
    )


def _safe_logdet_batch(matrices: np.ndarray, *, floor: float = 1.0e-10) -> np.ndarray:
    symmetric = 0.5 * (matrices + np.swapaxes(matrices, -1, -2))
    eigenvalues = np.linalg.eigvalsh(symmetric)
    return np.log(np.maximum(eigenvalues, floor)).sum(axis=-1)


def pairwise_gaussian_mmi_phi_r(
    bold: np.ndarray,
    *,
    tau: int = 1,
    tolerance_bits: float = NONNEGATIVE_TOLERANCE_BITS,
) -> dict[str, np.ndarray | int | float]:
    """Vectorized bivariate PhiR = PhiWMS + MMI double redundancy, in bits."""

    series = np.asarray(bold, dtype=float)
    if series.ndim != 2 or series.shape[0] <= tau + 3 or series.shape[1] < 2:
        raise ValueError("bold must have shape (time, regions) and enough lagged samples.")
    source = series[:-tau]
    target = series[tau:]
    joined = np.column_stack((source, target))
    scale = joined.std(axis=0, ddof=1)
    if np.any(scale <= 1.0e-12) or not np.isfinite(scale).all():
        bad = np.flatnonzero((scale <= 1.0e-12) | ~np.isfinite(scale))
        raise ValueError(f"Near-constant/nonfinite BOLD channels: {bad.tolist()}")
    joined = (joined - joined.mean(axis=0, keepdims=True)) / scale
    correlation = np.cov(joined, rowvar=False, bias=False)
    correlation = 0.5 * (correlation + correlation.T)

    n_regions = source.shape[1]
    left, right = np.triu_indices(n_regions, 1)
    pair_count = left.size
    batch = np.empty((pair_count, 4, 4), dtype=float)
    indices = np.column_stack((left, right, left + n_regions, right + n_regions))
    for row in range(4):
        for column in range(4):
            batch[:, row, column] = correlation[indices[:, row], indices[:, column]]

    logdet_source = _safe_logdet_batch(batch[:, :2, :2])
    logdet_target = _safe_logdet_batch(batch[:, 2:, 2:])
    logdet_joint = _safe_logdet_batch(batch)
    tdmi = np.maximum(0.0, 0.5 * (logdet_source + logdet_target - logdet_joint) / np.log(2.0))

    cross = batch[:, :2, 2:]
    singleton_mi = np.maximum(
        0.0,
        -0.5 * np.log(np.maximum(1.0 - np.square(cross), 1.0e-10)) / np.log(2.0),
    )
    self_left = singleton_mi[:, 0, 0]
    self_right = singleton_mi[:, 1, 1]
    redundancy = singleton_mi.min(axis=(1, 2))
    phi_wms = tdmi - self_left - self_right
    phi_r_raw = phi_wms + redundancy

    violation = phi_r_raw < -float(tolerance_bits)
    if np.any(violation):
        raise RuntimeError(
            "PhiR nonnegativity violation: "
            f"minimum={phi_r_raw.min():.12g} bits, tolerance={tolerance_bits:.3g}, "
            f"count={int(violation.sum())}/{pair_count}."
        )
    numerical_zero = (phi_r_raw < 0.0) & ~violation
    phi_r = phi_r_raw.copy()
    phi_r[numerical_zero] = 0.0
    return {
        "pair_indices": np.column_stack((left, right)),
        "phi_r": phi_r,
        "phi_r_raw": phi_r_raw,
        "phi_wms": phi_wms,
        "redundancy": redundancy,
        "pair_count": int(pair_count),
        "numerical_zero_count": int(numerical_zero.sum()),
        "minimum_raw_phi_r": float(phi_r_raw.min()),
    }


def validate_vectorization() -> float:
    """Check the vectorized estimator against the repository scalar Gaussian MI."""

    dmf = load_dmf_module()
    rng = np.random.default_rng(20260731)
    innovations = rng.normal(size=(600, 5))
    series = np.empty_like(innovations)
    series[0] = innovations[0]
    for step in range(1, len(series)):
        series[step] = 0.65 * series[step - 1] + innovations[step]
    vectorized = pairwise_gaussian_mmi_phi_r(series)
    source, target = series[:-1], series[1:]
    scalar: list[float] = []
    for left, right in np.asarray(vectorized["pair_indices"], dtype=int):
        covariance = np.cov(
            np.column_stack((source[:, left], source[:, right], target[:, left], target[:, right])),
            rowvar=False,
            bias=False,
        )
        tdmi = dmf.gaussian_mutual_information(
            covariance, sources=[0, 1], targets=[2, 3], log_base=2.0
        )
        self_left = dmf.gaussian_mutual_information(
            covariance, sources=[0], targets=[2], log_base=2.0
        )
        self_right = dmf.gaussian_mutual_information(
            covariance, sources=[1], targets=[3], log_base=2.0
        )
        redundancy = min(
            dmf.gaussian_mutual_information(
                covariance, sources=[source_index], targets=[target_index], log_base=2.0
            )
            for source_index in (0, 1)
            for target_index in (2, 3)
        )
        scalar.append(tdmi - self_left - self_right + redundancy)
    error = float(np.max(np.abs(np.asarray(scalar) - np.asarray(vectorized["phi_r_raw"]))))
    if error > 1.0e-9:
        raise AssertionError(f"Vectorized/scalar PhiR mismatch: {error:.3g} bits.")
    return error


SUMMARY_NAMES = (
    "phi_r_mean",
    "phi_r_sd_pairs",
    "phi_r_q05",
    "phi_r_median",
    "phi_r_q95",
    "phi_wms_mean",
    "redundancy_mean",
    "minimum_raw_phi_r",
    "numerical_zero_count",
    "pair_count",
    "mean_rate_hz",
    "stabilization_start_s",
)


def initialize(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    payload = {name: np.full(shape, np.nan, dtype=float) for name in SUMMARY_NAMES}
    payload["completed"] = np.zeros(shape, dtype=bool)
    return payload


def load_partial(path: Path, shape: tuple[int, int], resume: bool) -> dict[str, np.ndarray]:
    if not resume or not path.exists():
        return initialize(shape)
    with np.load(path) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    if payload["completed"].shape != shape:
        raise ValueError("Partial cache shape does not match the requested scan.")
    return payload


def run(args: argparse.Namespace) -> None:
    vectorization_error = validate_vectorization()
    dmf = load_dmf_module()
    source_path = resolve_path(args.source)
    with np.load(source_path) as archive:
        source_g = np.asarray(archive["G"], dtype=float)
        connectivity = np.asarray(archive["connectivity"], dtype=float)
        source_j_fic = np.asarray(archive["j_fic"], dtype=float)

    if args.mode == "smoke":
        selected = np.asarray([int(np.argmin(np.abs(source_g - float(args.smoke_g))))])
        seeds = np.asarray([int(args.seeds.split(",")[0])], dtype=int)
    else:
        selected = np.arange(source_g.size, dtype=int)
        seeds = np.asarray([int(value) for value in args.seeds.split(",")], dtype=int)
    g_values = source_g[selected]
    j_fic = source_j_fic[selected]

    output = resolve_path(args.output)
    summary_path = resolve_path(args.summary)
    status_path = resolve_path(args.status)
    if args.mode == "smoke":
        if output == DEFAULT_OUTPUT:
            output = output.with_name(output.stem + "_smoke.npz")
        if summary_path == DEFAULT_SUMMARY:
            summary_path = summary_path.with_name(summary_path.stem + "_smoke.json")
    partial = output.with_name(output.stem + ".partial.npz")
    shape = (len(seeds), len(g_values))
    payload = load_partial(partial, shape, resume=not args.no_resume)
    total = int(np.prod(shape))
    current = int(payload["completed"].sum())
    baseline = current
    started = time.monotonic()
    parameters = dmf.DMFParameters(
        t_total=float(args.natural_t_total),
        burn_in=float(args.natural_burn_in),
        dt=float(args.dt),
        sigma=float(args.sigma),
    )
    stabilization = dmf.StabilizationParameters(
        window=float(args.stabilization_window),
        tolerance_hz=float(args.stabilization_tolerance),
        confirm_windows=int(args.stabilization_confirm_windows),
    )
    write_status(
        status_path,
        phase="running",
        current=current,
        total=total,
        started=started,
        baseline=baseline,
        metrics={"vectorized_scalar_max_error_bits": vectorization_error},
    )

    try:
        for seed_index, seed in enumerate(seeds):
            for g_index, coupling in enumerate(g_values):
                if bool(payload["completed"][seed_index, g_index]):
                    continue
                source_index = int(selected[g_index])
                simulation = dmf.simulate_dmf(
                    connectivity,
                    float(coupling),
                    np.asarray(j_fic[g_index], dtype=float),
                    parameters=parameters,
                    stabilization_parameters=stabilization,
                    seed=int(seed) + source_index,
                    record_rate_trace=True,
                )
                start = int(float(simulation["stabilization_start_step"]))
                rates = np.asarray(simulation["region_rate_trace_hz"], dtype=float)[start:]
                bold = dmf.transform_rates_to_bold(rates, dt=float(args.dt))
                metrics = pairwise_gaussian_mmi_phi_r(
                    bold,
                    tau=int(args.tau),
                    tolerance_bits=float(args.nonnegative_tolerance),
                )
                phi_r = np.asarray(metrics["phi_r"], dtype=float)
                phi_wms = np.asarray(metrics["phi_wms"], dtype=float)
                redundancy = np.asarray(metrics["redundancy"], dtype=float)
                values = {
                    "phi_r_mean": float(phi_r.mean()),
                    "phi_r_sd_pairs": float(phi_r.std(ddof=1)),
                    "phi_r_q05": float(np.quantile(phi_r, 0.05)),
                    "phi_r_median": float(np.median(phi_r)),
                    "phi_r_q95": float(np.quantile(phi_r, 0.95)),
                    "phi_wms_mean": float(phi_wms.mean()),
                    "redundancy_mean": float(redundancy.mean()),
                    "minimum_raw_phi_r": float(metrics["minimum_raw_phi_r"]),
                    "numerical_zero_count": float(metrics["numerical_zero_count"]),
                    "pair_count": float(metrics["pair_count"]),
                    "mean_rate_hz": float(simulation["mean_rate_hz"]),
                    "stabilization_start_s": float(simulation["stabilization_start_time_s"]),
                }
                for name, value in values.items():
                    payload[name][seed_index, g_index] = value
                payload["completed"][seed_index, g_index] = True
                current += 1
                atomic_savez(partial, **payload)
                status_metrics = {
                    "seed": int(seed),
                    "G": float(coupling),
                    "phi_r_mean_bits": values["phi_r_mean"],
                    "phi_wms_mean_bits": values["phi_wms_mean"],
                    "redundancy_mean_bits": values["redundancy_mean"],
                    "minimum_raw_phi_r_bits": values["minimum_raw_phi_r"],
                }
                write_status(
                    status_path,
                    phase="running",
                    current=current,
                    total=total,
                    started=started,
                    baseline=baseline,
                    metrics=status_metrics,
                )
                print(
                    f"seed={seed} G={coupling:.2f} PhiR={values['phi_r_mean']:.6g} "
                    f"WMS={values['phi_wms_mean']:.6g} Rtr={values['redundancy_mean']:.6g}",
                    flush=True,
                )

        if not bool(np.all(payload["completed"])):
            raise RuntimeError("Scan ended with incomplete conditions.")
        atomic_savez(
            output,
            G=g_values,
            selected_g_indices=selected,
            seeds=seeds,
            dt=np.asarray(float(args.dt)),
            sigma=np.asarray(float(args.sigma)),
            natural_t_total=np.asarray(float(args.natural_t_total)),
            natural_burn_in=np.asarray(float(args.natural_burn_in)),
            tau=np.asarray(int(args.tau)),
            nonnegative_tolerance_bits=np.asarray(float(args.nonnegative_tolerance)),
            estimator=np.asarray("pairwise Gaussian MMI PhiID on Balloon-Windkessel BOLD-like signals"),
            bold_sampling_interval_s=np.asarray(float(args.dt)),
            j_fic_policy=np.asarray("fixed source calibration; one stored J_FIC vector per G"),
            vectorized_scalar_max_error_bits=np.asarray(vectorization_error),
            **payload,
        )
        peak_indices = np.argmax(payload["phi_r_mean"], axis=1)
        peak_g = g_values[peak_indices]
        curve_mean = payload["phi_r_mean"].mean(axis=0)
        curve_sd = payload["phi_r_mean"].std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(curve_mean)
        summary = {
            "output": str(output.relative_to(ROOT)),
            "condition_count": total,
            "seed_count": int(len(seeds)),
            "G_count": int(len(g_values)),
            "pair_count_per_condition": int(payload["pair_count"][0, 0]),
            "curve_mean_bits": curve_mean.tolist(),
            "curve_sd_across_seeds_bits": curve_sd.tolist(),
            "peak_G_by_seed": peak_g.tolist(),
            "peak_G_of_mean_curve": float(g_values[int(np.argmax(curve_mean))]),
            "peak_phi_r_mean_bits": float(curve_mean.max()),
            "minimum_raw_phi_r_bits": float(np.nanmin(payload["minimum_raw_phi_r"])),
            "numerical_zero_count_total": int(np.nansum(payload["numerical_zero_count"])),
            "nonnegative_tolerance_bits": float(args.nonnegative_tolerance),
            "vectorized_scalar_max_error_bits": vectorization_error,
        }
        atomic_json(summary_path, summary)
        if partial.exists():
            partial.unlink()
        write_status(
            status_path,
            phase="complete",
            current=total,
            total=total,
            started=started,
            baseline=baseline,
            metrics={"output": str(output), "peak_G": summary["peak_G_of_mean_curve"]},
        )
        print(f"Saved {output}", flush=True)
    except Exception as error:
        atomic_savez(partial, **payload)
        write_status(
            status_path,
            phase="failed",
            current=current,
            total=total,
            started=started,
            baseline=baseline,
            message=str(error),
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--seeds", default="3,4,5,6,7,8,9,10")
    parser.add_argument("--smoke-g", type=float, default=1.3)
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--natural-t-total", type=float, default=1.5)
    parser.add_argument("--natural-burn-in", type=float, default=0.3)
    parser.add_argument("--stabilization-window", type=float, default=0.05)
    parser.add_argument("--stabilization-tolerance", type=float, default=0.15)
    parser.add_argument("--stabilization-confirm-windows", type=int, default=2)
    parser.add_argument("--nonnegative-tolerance", type=float, default=NONNEGATIVE_TOLERANCE_BITS)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
