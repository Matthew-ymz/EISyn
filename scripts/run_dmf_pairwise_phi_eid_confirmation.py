#!/usr/bin/env python3
"""Paired all-ROI-pair PhiEID confirmation for the 83-region DMF model.

The experiment changes only the integration scale: each metric is the mean over
all unordered ROI pairs.  Every ROI contributes its (sE, sI) state as one source
block; pair PhiEID is the joint pair EI minus the two one-ROI EIs to the same
pair target.  Seeds, intervention support, G schedule, horizon, dynamics, and
noise stream match the existing full-system peak-alignment experiment.
"""

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

from scripts.run_dmf_diffusive_fullstate_control import dmf_step_batch_with_operator
from scripts.run_dmf_fixed_uniform_multihorizon import fixed_uniform_initial_state
from scripts.validate_dmf_83_region_oracle_phi_eid import (
    DEFAULT_SOURCE_RESULTS,
    gaussian_singleton_source_phi,
    load_dmf_module,
    resolve_path,
    standardize,
)


DEFAULT_OUTPUT = ROOT / "results" / "dmf_pairwise_phi_eid_confirmation" / "support030_070_tau400_n2048_seeds3_10.npz"
DEFAULT_STATUS = ROOT / "docs" / "log" / "dmf_pairwise_phi_eid_confirmation_live_progress.json"
DEFAULT_G = (1.0, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.2, 3.0)


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(raw).split(",") if item.strip())


def parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in str(raw).split(",") if item.strip())


def atomic_savez(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def write_status(path: Path, *, started: float, current: int, total: int, phase: str,
                 metrics: dict[str, object] | None = None, message: str | None = None) -> None:
    elapsed = time.monotonic() - started
    rate = current / elapsed if elapsed > 0.0 else 0.0
    payload: dict[str, object] = {
        "phase": phase,
        "current": current,
        "total": total,
        "unit": "seed-coupling condition",
        "elapsed_seconds": elapsed,
        "eta_seconds": (total - current) / rate if rate > 0.0 else None,
        "metrics": metrics or {},
        "updated_at": time.time(),
    }
    if message:
        payload["message"] = message
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def interpolate_j_fic(source_g: np.ndarray, source_j_fic: np.ndarray, requested_g: np.ndarray) -> np.ndarray:
    if requested_g.min() < source_g.min() or requested_g.max() > source_g.max():
        raise ValueError("Requested G values must lie within the calibrated J_FIC schedule.")
    return np.column_stack([
        np.interp(requested_g, source_g, source_j_fic[:, region])
        for region in range(source_j_fic.shape[1])
    ])


def pair_coordinates(node_count: int) -> np.ndarray:
    left, right = np.triu_indices(node_count, k=1)
    return np.column_stack((left, right, node_count + left, node_count + right)).astype(int)


def batch_logdet_psd(matrices: np.ndarray, *, floor: float = 1.0e-12) -> np.ndarray:
    symmetric = 0.5 * (matrices + np.swapaxes(matrices, -1, -2))
    return np.log(np.maximum(np.linalg.eigvalsh(symmetric), float(floor))).sum(axis=-1)


def block_ei(source_cov_emp: np.ndarray, target_cov_emp: np.ndarray, cross_cov: np.ndarray,
             source_coordinates: np.ndarray, target_coordinates: np.ndarray, *, ridge: float) -> np.ndarray:
    """Vectorized Gaussian EI matching gaussian_singleton_source_phi's estimator."""
    source_emp = source_cov_emp[source_coordinates[:, :, None], source_coordinates[:, None, :]]
    target_emp = target_cov_emp[target_coordinates[:, :, None], target_coordinates[:, None, :]]
    cross = cross_cov[source_coordinates[:, :, None], target_coordinates[:, None, :]]
    source_dim = source_coordinates.shape[1]
    target_dim = target_coordinates.shape[1]
    source_eye = np.eye(source_dim)[None, :, :]
    target_eye = np.eye(target_dim)[None, :, :]
    source_prior = source_eye * np.diagonal(source_emp, axis1=1, axis2=2)[:, None, :]
    source_prior = source_prior + float(ridge) * source_eye
    # np.linalg.lstsq(source, target) equals solve(C_xx, C_xy) for centered samples.
    coefficient = np.linalg.solve(source_emp, cross)
    transition = np.swapaxes(coefficient, -1, -2)
    residual = target_emp - transition @ cross
    residual = 0.5 * (residual + np.swapaxes(residual, -1, -2)) + float(ridge) * target_eye
    target_cov = transition @ source_prior @ np.swapaxes(transition, -1, -2) + residual
    target_cov = 0.5 * (target_cov + np.swapaxes(target_cov, -1, -2)) + float(ridge) * target_eye
    source_target = source_prior @ np.swapaxes(transition, -1, -2)
    conditional = source_prior - source_target @ np.linalg.inv(target_cov) @ np.swapaxes(source_target, -1, -2)
    conditional = 0.5 * (conditional + np.swapaxes(conditional, -1, -2)) + float(ridge) * source_eye
    return 0.5 * (batch_logdet_psd(source_prior) - batch_logdet_psd(conditional)) / np.log(2.0)


def all_pair_phi_eid(source_z: np.ndarray, target_z: np.ndarray, *, ridge: float) -> np.ndarray:
    feature_count = source_z.shape[1]
    node_count = feature_count // 2
    if source_z.shape != target_z.shape or 2 * node_count != feature_count:
        raise ValueError("Expected matching standardized full-state source and target arrays.")
    denominator = float(source_z.shape[0] - 1)
    source_cov = source_z.T @ source_z / denominator
    target_cov = target_z.T @ target_z / denominator
    cross_cov = source_z.T @ target_z / denominator
    pair = pair_coordinates(node_count)
    joint = block_ei(source_cov, target_cov, cross_cov, pair, pair, ridge=ridge)
    first = block_ei(source_cov, target_cov, cross_cov, pair[:, (0, 2)], pair, ridge=ridge)
    second = block_ei(source_cov, target_cov, cross_cov, pair[:, (1, 3)], pair, ridge=ridge)
    return joint - first - second


def scalar_pair_phi_eid(source_z: np.ndarray, target_z: np.ndarray, coordinates: np.ndarray, *, ridge: float) -> float:
    """Reference implementation used only for vectorization smoke verification."""
    pair_source = source_z[:, coordinates]
    pair_target = target_z[:, coordinates]
    joint = float(gaussian_singleton_source_phi(pair_source, pair_target, ridge=ridge)["joint_ei"])
    first = float(gaussian_singleton_source_phi(pair_source[:, (0, 2)], pair_target, ridge=ridge)["joint_ei"])
    second = float(gaussian_singleton_source_phi(pair_source[:, (1, 3)], pair_target, ridge=ridge)["joint_ei"])
    return joint - first - second


def verify_vectorization(*, ridge: float) -> None:
    rng = np.random.default_rng(7)
    source, _, _ = standardize(rng.normal(size=(512, 10)))
    target, _, _ = standardize(0.35 * source + rng.normal(size=(512, 10)))
    vectorized = all_pair_phi_eid(source, target, ridge=ridge)
    coordinates = pair_coordinates(5)
    reference = np.asarray([scalar_pair_phi_eid(source, target, item, ridge=ridge) for item in coordinates])
    if not np.allclose(vectorized, reference, rtol=1.0e-10, atol=1.0e-10):
        raise RuntimeError(f"Vectorized pair EI differs from scalar reference (max abs={np.max(np.abs(vectorized-reference)):.3e}).")


def make_payload(*, seeds: tuple[int, ...], g_values: np.ndarray, pair_count: int) -> dict[str, np.ndarray]:
    shape = (len(seeds), len(g_values))
    return {
        "pair_phi_eid": np.full(shape + (pair_count,), np.nan, dtype=float),
        "pair_mean_phi_eid": np.full(shape, np.nan, dtype=float),
        "pair_median_phi_eid": np.full(shape, np.nan, dtype=float),
        "pair_positive_fraction": np.full(shape, np.nan, dtype=float),
        "completed": np.zeros(shape, dtype=bool),
        "seeds": np.asarray(seeds, dtype=int),
        "G": np.asarray(g_values, dtype=float),
    }


def load_or_initialize(partial: Path, *, seeds: tuple[int, ...], g_values: np.ndarray, pair_count: int,
                       resume: bool) -> dict[str, np.ndarray]:
    if not resume or not partial.exists():
        return make_payload(seeds=seeds, g_values=g_values, pair_count=pair_count)
    with np.load(partial) as stored:
        if not np.array_equal(stored["seeds"], np.asarray(seeds, dtype=int)) or not np.array_equal(stored["G"], g_values):
            raise ValueError("Partial result does not match requested seeds or G schedule.")
        payload = {name: np.asarray(stored[name]) for name in stored.files}
    if payload["pair_phi_eid"].shape != (len(seeds), len(g_values), pair_count):
        raise ValueError("Partial result has a mismatched pair count.")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired all-ROI-pair PhiEID DMF confirmation.")
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--seeds", type=parse_int_list, default=tuple(range(3, 11)))
    parser.add_argument("--g-values", type=parse_float_list, default=DEFAULT_G)
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--support-low", type=float, default=0.3)
    parser.add_argument("--support-high", type=float, default=0.7)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.support_low) < float(args.support_high) <= 1.0:
        raise ValueError("Support must satisfy 0 <= low < high <= 1.")
    seeds = tuple(args.seeds)
    g_values = np.asarray(args.g_values, dtype=float)
    if not seeds or len(np.unique(g_values)) != len(g_values) or int(args.sample_count) < 8:
        raise ValueError("Seeds/G values must be non-empty and sample count must be at least eight.")
    verify_vectorization(ridge=float(args.ridge))
    dmf = load_dmf_module()
    with np.load(resolve_path(args.source_results)) as archive:
        source_g = np.asarray(archive["G"], dtype=float)
        connectivity = np.asarray(archive["connectivity"], dtype=float)
        source_j_fic = np.asarray(archive["j_fic"], dtype=float)
    j_fic = interpolate_j_fic(source_g, source_j_fic, g_values)
    node_count = connectivity.shape[0]
    pair_count = node_count * (node_count - 1) // 2
    output = resolve_path(args.output)
    partial = output.with_name(output.stem + ".partial.npz")
    status = resolve_path(args.status)
    payload = load_or_initialize(partial, seeds=seeds, g_values=g_values, pair_count=pair_count,
                                 resume=not bool(args.no_resume))
    started = time.monotonic()
    total = len(seeds) * len(g_values)
    current = int(payload["completed"].sum())
    write_status(status, started=started, current=current, total=total, phase="running",
                 message="Verified vectorized Gaussian pair-EI calculation; loading DMF inputs")
    parameters = dmf.DMFParameters(t_total=1.0, burn_in=0.0, dt=float(args.dt), sigma=float(args.sigma))
    try:
        for seed_index, seed in enumerate(seeds):
            for g_index, coupling_g in enumerate(g_values):
                if bool(payload["completed"][seed_index, g_index]):
                    continue
                key = int(seed) * 1_000_000 + int(round(float(coupling_g) * 100)) * 1_000
                source_rng = np.random.default_rng(key)
                source_se, source_si = fixed_uniform_initial_state(
                    source_rng, sample_count=int(args.sample_count), dimension=node_count, source_state="se_si",
                    se_low=float(args.support_low), se_high=float(args.support_high),
                    si_low=float(args.support_low), si_high=float(args.support_high),
                )
                if source_si is None:  # pragma: no cover
                    raise RuntimeError("Full-state intervention unexpectedly omitted sI.")
                source_z, _, _ = standardize(np.concatenate((source_se, source_si), axis=1))
                noise_rng = np.random.default_rng(key + 17)
                target_se, target_si = source_se, source_si
                for _ in range(int(args.horizon)):
                    target_se, target_si = dmf_step_batch_with_operator(
                        dmf, target_se, target_si, connectivity=connectivity, coupling_g=float(coupling_g),
                        j_fic=j_fic[g_index], parameters=parameters, mode="direct", state_boundary="none", rng=noise_rng,
                    )
                target_z, _, _ = standardize(np.concatenate((target_se, target_si), axis=1))
                values = all_pair_phi_eid(source_z, target_z, ridge=float(args.ridge))
                payload["pair_phi_eid"][seed_index, g_index] = values
                payload["pair_mean_phi_eid"][seed_index, g_index] = float(values.mean())
                payload["pair_median_phi_eid"][seed_index, g_index] = float(np.median(values))
                payload["pair_positive_fraction"][seed_index, g_index] = float(np.mean(values > 0.0))
                payload["completed"][seed_index, g_index] = True
                current += 1
                atomic_savez(partial, **payload)
                metrics = {"seed": int(seed), "G": float(coupling_g),
                           "mean_pair_phi_eid_bits": float(values.mean()),
                           "positive_pair_fraction": float(np.mean(values > 0.0))}
                write_status(status, started=started, current=current, total=total, phase="running", metrics=metrics)
                print(f"seed={seed} G={coupling_g:.2f} {metrics}", flush=True)
        atomic_savez(output, **payload, pair_count=np.asarray(pair_count), sample_count=np.asarray(int(args.sample_count)),
                     horizon=np.asarray(int(args.horizon)), support=np.asarray((args.support_low, args.support_high)),
                     source_state=np.asarray("se_si"), target_state=np.asarray("se_si"),
                     pair_definition=np.asarray("ROI=(sE,sI); joint EI minus one-ROI EI sum"),
                     j_fic_interpolation=np.asarray("linear"), state_boundary=np.asarray("none"), coupling_mode=np.asarray("direct"))
        write_status(status, started=started, current=total, total=total, phase="complete", metrics={"output": str(output)})
        print(f"Saved {output}", flush=True)
    except Exception as error:
        atomic_savez(partial, **payload)
        write_status(status, started=started, current=current, total=total, phase="failed", message=str(error))
        raise


if __name__ == "__main__":
    main()
