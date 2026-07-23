#!/usr/bin/env python3
"""Run observational WMS with every non-distribution setting aligned to the current PhiEID sweep."""

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

from scripts.run_dmf_diffusive_fullstate_control import rollout
from scripts.run_dmf_fixed_uniform_multihorizon import fixed_uniform_initial_state
from scripts.validate_dmf_83_region_oracle_phi_eid import (
    gaussian_singleton_source_phi,
    load_dmf_module,
    resolve_path,
    standardize,
)


DEFAULT_SOURCE = ROOT / "exp" / "brain" / "result_lausanne_fig6" / "count_00_fig6b_mean_rate.npz"
DEFAULT_REFERENCE = ROOT / "results" / "dmf_fullstate_uniform_support" / "confirm_c050_h020_tau300_n2048_no_clip_seeds3_10.npz"
DEFAULT_OUTPUT = ROOT / "results" / "dmf_83_whole_system_wms" / "aligned_observational_tau300_n2048_seeds3_10.npz"
DEFAULT_REUSE = DEFAULT_OUTPUT
DEFAULT_STATUS = ROOT / "docs" / "log" / "dmf_observational_wms_progress.json"


def atomic_savez(path: Path, **payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
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
    completed_this_run = max(0, current - int(baseline))
    rate = completed_this_run / elapsed if elapsed > 0.0 else 0.0
    payload = {
        "phase": phase,
        "current": current,
        "total": total,
        "unit": "seed-G condition",
        "elapsed_seconds": elapsed,
        "eta_seconds": (total - current) / rate if rate > 0.0 else None,
        "metrics": metrics or {},
        "message": message,
        "updated_at": time.time(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def initialize(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    payload = {
        name: np.full(shape, np.nan, dtype=float)
        for name in (
            "phi_wms",
            "whole_mi",
            "singleton_mi_sum",
            "source_condition_number",
            "noise_condition_number",
            "source_unique_count",
        )
    }
    payload["completed"] = np.zeros(shape, dtype=bool)
    return payload


def load_partial(path: Path, shape: tuple[int, int], resume: bool) -> dict[str, np.ndarray]:
    if not resume or not path.exists():
        return initialize(shape)
    with np.load(path) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    if payload["completed"].shape != shape:
        raise ValueError("Partial cache shape does not match the requested experiment.")
    return payload


def reuse_completed_conditions(
    payload: dict[str, np.ndarray],
    *,
    reuse_path: Path,
    seeds: np.ndarray,
    g_values: np.ndarray,
) -> int:
    if not reuse_path.exists():
        return 0
    reused = 0
    with np.load(reuse_path) as archive:
        old_seeds = np.asarray(archive["seeds"], dtype=int)
        old_g = np.asarray(archive["G"], dtype=float)
        for seed_index, seed in enumerate(seeds):
            old_seed_matches = np.flatnonzero(old_seeds == int(seed))
            if old_seed_matches.size != 1:
                continue
            old_seed_index = int(old_seed_matches[0])
            for g_index, coupling in enumerate(g_values):
                old_g_matches = np.flatnonzero(np.isclose(old_g, float(coupling), atol=1.0e-12))
                if old_g_matches.size != 1 or bool(payload["completed"][seed_index, g_index]):
                    continue
                old_g_index = int(old_g_matches[0])
                if "completed" in archive.files and not bool(archive["completed"][old_seed_index, old_g_index]):
                    continue
                for name in payload:
                    if name == "completed" or name not in archive.files:
                        continue
                    payload[name][seed_index, g_index] = archive[name][old_seed_index, old_g_index]
                payload["completed"][seed_index, g_index] = True
                reused += 1
    return reused


def run(args: argparse.Namespace) -> None:
    dmf = load_dmf_module()
    with np.load(resolve_path(args.reference)) as reference:
        reference_g = np.asarray(reference["G"], dtype=float)
        g_values = reference_g.copy()
        selected = np.asarray(reference["selected_g_indices"], dtype=int)
        seeds = np.asarray(reference["seeds"], dtype=int)
        horizon = int(np.asarray(reference["horizon"]).item())
        sample_count = int(np.asarray(reference["sample_count"]).item())
        ridge = float(args.ridge)
        support_low = float(np.asarray(reference["intervention_low"]).item())
        support_high = float(np.asarray(reference["intervention_high"]).item())
        state_boundary = str(np.asarray(reference["state_boundary"]).item())
        source_state = str(np.asarray(reference["source_state"]).item())
        reference_phi_eid = np.asarray(reference["phi_eid"], dtype=float)[0]
    if source_state != "se_si":
        raise ValueError("Reference experiment must use the full se_si source state.")

    with np.load(resolve_path(args.source)) as source_archive:
        source_g = np.asarray(source_archive["G"], dtype=float)
        connectivity = np.asarray(source_archive["connectivity"], dtype=float)
        source_j_fic = np.asarray(source_archive["j_fic"], dtype=float)

    if args.dense_g:
        g_values = source_g.copy()
        selected = np.arange(source_g.size, dtype=int)
        j_fic = source_j_fic.copy()
    else:
        j_fic = source_j_fic[selected]

    if args.smoke:
        seeds = seeds[:1]
        g_values = g_values[:1]
        selected = selected[:1]
        j_fic = j_fic[:1]
        sample_count = int(args.smoke_sample_count)

    output = resolve_path(args.output)
    if args.dense_g and output == DEFAULT_OUTPUT:
        output = output.with_name(output.stem + "_dense_g01.npz")
    if args.smoke:
        output = output.with_name(output.stem + "_smoke.npz")
    partial = output.with_name(output.stem + ".partial.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    status = resolve_path(args.status)
    shape = (len(seeds), len(g_values))
    payload = load_partial(partial, shape, resume=not args.no_resume)
    reused = reuse_completed_conditions(
        payload,
        reuse_path=resolve_path(args.reuse_cache),
        seeds=seeds,
        g_values=g_values,
    )
    total = int(np.prod(shape))
    current = int(payload["completed"].sum())
    baseline = current
    started = time.monotonic()
    write_status(
        status,
        phase="running",
        current=current,
        total=total,
        started=started,
        baseline=baseline,
        metrics={"reused_conditions": int(reused)},
    )

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
    rollout_parameters = dmf.DMFParameters(t_total=1.0, burn_in=0.0, dt=float(args.dt), sigma=float(args.sigma))

    try:
        for seed_index, seed in enumerate(seeds):
            for g_index, coupling in enumerate(g_values):
                if bool(payload["completed"][seed_index, g_index]):
                    continue
                simulation = dmf.simulate_dmf(
                    connectivity,
                    float(coupling),
                    np.asarray(j_fic[g_index], dtype=float),
                    parameters=parameters,
                    stabilization_parameters=stabilization,
                    seed=int(seed) + int(selected[g_index]),
                    record_state_trace=True,
                )
                start = int(simulation["stabilization_start_step"])
                se_trace = np.asarray(simulation["state_se_trace"], dtype=float)[start:]
                si_trace = np.asarray(simulation["state_si_trace"], dtype=float)[start:]
                if len(se_trace) < 4:
                    raise RuntimeError("Natural steady-state trace is too short.")

                schedule_index = int(np.rint((float(coupling) - 1.0) / 0.1))
                source_seed = int(seed) * 100_000 + schedule_index * 1_000
                source_rng = np.random.default_rng(source_seed)
                # Consume the same source RNG stream as the intervention reference.
                fixed_uniform_initial_state(
                    source_rng,
                    sample_count=sample_count,
                    dimension=connectivity.shape[0],
                    source_state="se_si",
                    low=support_low,
                    high=support_high,
                )
                indices = source_rng.integers(0, len(se_trace), size=sample_count)
                source_se = se_trace[indices].copy()
                source_si = si_trace[indices].copy()
                source = np.concatenate((source_se, source_si), axis=1)
                source_z, _, _ = standardize(source)

                noise_rng = np.random.default_rng(source_seed + 17)
                target_se, target_si = rollout(
                    dmf,
                    source_se,
                    source_si,
                    connectivity=connectivity,
                    coupling_g=float(coupling),
                    j_fic=np.asarray(j_fic[g_index], dtype=float),
                    parameters=rollout_parameters,
                    mode="direct",
                    state_boundary=state_boundary,
                    horizon=horizon,
                    rng=noise_rng,
                )
                target = np.concatenate((target_se, target_si), axis=1)
                target_z, _, _ = standardize(target)
                result = gaussian_singleton_source_phi(
                    source_z,
                    target_z,
                    ridge=ridge,
                    factorize_source_covariance=False,
                )
                payload["phi_wms"][seed_index, g_index] = float(result["raw_phi"])
                payload["whole_mi"][seed_index, g_index] = float(result["joint_ei"])
                payload["singleton_mi_sum"][seed_index, g_index] = float(result["singleton_ei_sum"])
                payload["source_condition_number"][seed_index, g_index] = float(result["source_condition_number"])
                payload["noise_condition_number"][seed_index, g_index] = float(result["noise_condition_number"])
                payload["source_unique_count"][seed_index, g_index] = float(np.unique(indices).size)
                payload["completed"][seed_index, g_index] = True
                current += 1
                atomic_savez(partial, **payload)
                metrics = {
                    "seed": int(seed),
                    "G": float(coupling),
                    "phi_wms_bits": float(result["raw_phi"]),
                    "unique_natural_states": int(np.unique(indices).size),
                }
                write_status(
                    status,
                    phase="running",
                    current=current,
                    total=total,
                    started=started,
                    baseline=baseline,
                    metrics=metrics,
                )
                print(
                    f"seed={seed} G={coupling:.2f} WMS={float(result['raw_phi']):.6g} "
                    f"unique={int(np.unique(indices).size)}/{sample_count}",
                    flush=True,
                )

        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_savez(
            output,
            G=g_values,
            selected_g_indices=selected,
            seeds=seeds,
            horizon=np.asarray(horizon),
            sample_count=np.asarray(sample_count),
            ridge=np.asarray(ridge),
            dt=np.asarray(float(args.dt)),
            sigma=np.asarray(float(args.sigma)),
            source_state=np.asarray("se_si"),
            target_state=np.asarray("se_si"),
            source_distribution=np.asarray("natural steady-state DMF"),
            estimator=np.asarray("linear-Gaussian ordinary mutual information"),
            factorized_source_covariance=np.asarray(False),
            state_boundary=np.asarray(state_boundary),
            coupling_mode=np.asarray("direct"),
            reference_phi_eid_G=reference_g,
            reference_phi_eid=reference_phi_eid,
            **payload,
        )
        if partial.exists():
            partial.unlink()
        write_status(
            status,
            phase="complete",
            current=total,
            total=total,
            started=started,
            baseline=baseline,
            metrics={"output": str(output)},
        )
        print(f"Saved {output}", flush=True)
    except Exception as error:
        atomic_savez(partial, **payload)
        write_status(
            status,
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
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--reuse-cache", type=Path, default=DEFAULT_REUSE)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--natural-t-total", type=float, default=1.5)
    parser.add_argument("--natural-burn-in", type=float, default=0.3)
    parser.add_argument("--stabilization-window", type=float, default=0.05)
    parser.add_argument("--stabilization-tolerance", type=float, default=0.15)
    parser.add_argument("--stabilization-confirm-windows", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-sample-count", type=int, default=64)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dense-g", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
