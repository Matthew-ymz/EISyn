#!/usr/bin/env python3
"""Paired DMF support-by-horizon sweep for mechanism-aligned PhiEID peaks.

The only experimental treatments are the fixed maximum-entropy intervention
support and prediction horizon.  Each support uses the same underlying uniform
draws and each horizon is read from the same 400-step rollout, so comparisons
are paired within every seed and coupling value.
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
from scripts.run_dmf_fixed_uniform_multihorizon import fixed_uniform_initial_state, target_diagnostics
from scripts.validate_dmf_83_region_oracle_phi_eid import (
    DEFAULT_SOURCE_RESULTS,
    gaussian_singleton_source_phi,
    load_dmf_module,
    resolve_path,
    standardize,
)


DEFAULT_OUTPUT = ROOT / "results" / "dmf_phi_eid_peak_alignment" / "paired_support_horizon.npz"
DEFAULT_STATUS = ROOT / "docs" / "log" / "dmf_phi_eid_peak_alignment_live_progress.json"
METRICS = (
    "whole_ei",
    "singleton_ei_sum",
    "phi_eid",
    "target_variance_retained",
    "target_spatial_sd",
    "target_mean_offdiag_correlation",
    "target_entropy",
    "joint_conditional_entropy",
    "singleton_conditional_entropy_sum",
)


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())


def parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())


def parse_supports(raw: str) -> tuple[tuple[float, float], ...]:
    supports: list[tuple[float, float]] = []
    for item in str(raw).split(","):
        low, high = (float(part.strip()) for part in item.split(":"))
        if not 0.0 <= low < high <= 1.0:
            raise argparse.ArgumentTypeError("Every support must satisfy 0 <= low < high <= 1.")
        supports.append((low, high))
    if not supports:
        raise argparse.ArgumentTypeError("At least one support is required.")
    return tuple(supports)


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_status(
    path: Path,
    *,
    started: float,
    current: int,
    total: int,
    phase: str,
    metrics: dict[str, object] | None = None,
    message: str | None = None,
) -> None:
    elapsed = time.monotonic() - started
    rate = current / elapsed if elapsed > 0.0 else 0.0
    payload: dict[str, object] = {
        "phase": phase,
        "current": current,
        "total": total,
        "unit": "support-seed-coupling condition",
        "elapsed_seconds": elapsed,
        "eta_seconds": (total - current) / rate if rate > 0.0 else None,
        "metrics": metrics or {},
        "updated_at": time.time(),
    }
    if message:
        payload["message"] = message
    atomic_write_json(path, payload)


def atomic_savez(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def interpolate_j_fic(source_g: np.ndarray, source_j_fic: np.ndarray, requested_g: np.ndarray) -> np.ndarray:
    if requested_g.min() < source_g.min() or requested_g.max() > source_g.max():
        raise ValueError("Requested G values must lie within the calibrated J_FIC schedule.")
    return np.column_stack([
        np.interp(requested_g, source_g, source_j_fic[:, region])
        for region in range(source_j_fic.shape[1])
    ])


def make_payload(
    *,
    g_values: np.ndarray,
    supports: tuple[tuple[float, float], ...],
    seeds: tuple[int, ...],
    horizons: tuple[int, ...],
) -> dict[str, np.ndarray]:
    shape = (len(supports), len(seeds), len(g_values), len(horizons))
    payload = {name: np.full(shape, np.nan, dtype=float) for name in METRICS}
    payload["completed"] = np.zeros(shape[:-1], dtype=bool)
    payload["G"] = np.asarray(g_values, dtype=float)
    payload["supports"] = np.asarray(supports, dtype=float)
    payload["seeds"] = np.asarray(seeds, dtype=int)
    payload["horizons"] = np.asarray(horizons, dtype=int)
    return payload


def load_or_initialize_payload(
    partial: Path,
    *,
    g_values: np.ndarray,
    supports: tuple[tuple[float, float], ...],
    seeds: tuple[int, ...],
    horizons: tuple[int, ...],
    resume: bool,
) -> dict[str, np.ndarray]:
    if not resume or not partial.exists():
        return make_payload(g_values=g_values, supports=supports, seeds=seeds, horizons=horizons)
    with np.load(partial) as stored:
        expected = {
            "G": np.asarray(g_values, dtype=float),
            "supports": np.asarray(supports, dtype=float),
            "seeds": np.asarray(seeds, dtype=int),
            "horizons": np.asarray(horizons, dtype=int),
        }
        for name, value in expected.items():
            if name not in stored or not np.array_equal(np.asarray(stored[name]), value):
                raise ValueError(f"Partial result does not match requested {name}; use a different output path.")
        return {name: np.asarray(stored[name]) for name in stored.files}


def persist_partial(partial: Path, payload: dict[str, np.ndarray]) -> None:
    atomic_savez(partial, **payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired DMF PhiEID peak-alignment confirmation.")
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--seeds", type=parse_int_list, default=tuple(range(3, 11)))
    parser.add_argument(
        "--g-values", type=parse_float_list,
        default=(1.0, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.2, 3.0),
    )
    parser.add_argument("--supports", type=parse_supports, default=((0.3, 0.7), (0.3, 0.5)))
    parser.add_argument("--horizons", type=parse_int_list, default=(300, 400))
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    g_values = np.asarray(args.g_values, dtype=float)
    supports = tuple(args.supports)
    seeds = tuple(args.seeds)
    horizons = tuple(sorted({int(value) for value in args.horizons}))
    if not horizons or horizons[0] < 1 or int(args.sample_count) < 4:
        raise ValueError("Horizons must be positive and sample count must be at least four.")
    if len(np.unique(g_values)) != len(g_values):
        raise ValueError("G values must be unique.")

    output = resolve_path(args.output)
    partial = output.with_name(output.stem + ".partial.npz")
    status = resolve_path(args.status)
    started = time.monotonic()
    total = len(supports) * len(seeds) * len(g_values)
    payload = load_or_initialize_payload(
        partial, g_values=g_values, supports=supports, seeds=seeds, horizons=horizons,
        resume=not bool(args.no_resume),
    )
    current = int(np.asarray(payload["completed"], dtype=bool).sum())
    write_status(status, started=started, current=current, total=total, phase="running", message="Loading DMF inputs")

    try:
        dmf = load_dmf_module()
        with np.load(resolve_path(args.source_results)) as archive:
            source_g = np.asarray(archive["G"], dtype=float)
            connectivity = np.asarray(archive["connectivity"], dtype=float)
            source_j_fic = np.asarray(archive["j_fic"], dtype=float)
        j_fic = interpolate_j_fic(source_g, source_j_fic, g_values)
        parameters = dmf.DMFParameters(t_total=1.0, burn_in=0.0, dt=float(args.dt), sigma=float(args.sigma))

        for support_index, (low, high) in enumerate(supports):
            for seed_index, seed in enumerate(seeds):
                for g_index, coupling_g in enumerate(g_values):
                    if bool(payload["completed"][support_index, seed_index, g_index]):
                        continue
                    source_rng = np.random.default_rng(int(seed) * 1_000_000 + int(round(coupling_g * 100)) * 1_000)
                    source_se, source_si = fixed_uniform_initial_state(
                        source_rng, sample_count=int(args.sample_count), dimension=connectivity.shape[0],
                        source_state="se_si", se_low=float(low), se_high=float(high),
                        si_low=float(low), si_high=float(high),
                    )
                    if source_si is None:  # pragma: no cover - guarded by source_state
                        raise RuntimeError("Full-state source unexpectedly omitted inhibitory coordinates.")
                    source_z, _, _ = standardize(np.concatenate((source_se, source_si), axis=1))
                    noise_rng = np.random.default_rng(int(seed) * 1_000_000 + int(round(coupling_g * 100)) * 1_000 + 17)
                    state_se = source_se
                    state_si = source_si
                    targets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
                    for step in range(1, horizons[-1] + 1):
                        state_se, state_si = dmf_step_batch_with_operator(
                            dmf, state_se, state_si, connectivity=connectivity, coupling_g=float(coupling_g),
                            j_fic=j_fic[g_index], parameters=parameters, mode="direct", state_boundary="none",
                            rng=noise_rng,
                        )
                        if step in horizons:
                            targets[step] = (state_se.copy(), state_si.copy())
                    phi_by_horizon: dict[str, float] = {}
                    for horizon_index, horizon in enumerate(horizons):
                        target_se, target_si = targets[horizon]
                        target_z, _, _ = standardize(np.concatenate((target_se, target_si), axis=1))
                        result = gaussian_singleton_source_phi(source_z, target_z, ridge=float(args.ridge))
                        diagnostics = target_diagnostics(source_se, target_se)
                        for name in ("whole_ei", "singleton_ei_sum", "phi_eid", "target_entropy", "joint_conditional_entropy", "singleton_conditional_entropy_sum"):
                            key = "joint_ei" if name == "whole_ei" else "raw_phi" if name == "phi_eid" else name
                            payload[name][support_index, seed_index, g_index, horizon_index] = float(result[key])
                        for name, value in diagnostics.items():
                            payload[name][support_index, seed_index, g_index, horizon_index] = float(value)
                        phi_by_horizon[f"phi_eid_tau{horizon}_bits"] = float(result["raw_phi"])
                    payload["completed"][support_index, seed_index, g_index] = True
                    current += 1
                    persist_partial(partial, payload)
                    write_status(
                        status, started=started, current=current, total=total, phase="running",
                        metrics={"support": f"U({low:.2f},{high:.2f})", "seed": int(seed), "G": float(coupling_g), **phi_by_horizon},
                    )
                    print(f"support=U({low:.2f},{high:.2f}) seed={seed} G={coupling_g:.2f} {phi_by_horizon}", flush=True)

        atomic_savez(
            output,
            **payload,
            source_results_g=source_g,
            j_fic_interpolation=np.asarray("linear"),
            sample_count=np.asarray(int(args.sample_count)),
            source_state=np.asarray("se_si"),
            target_state=np.asarray("se_si"),
            state_boundary=np.asarray("none"),
            coupling_mode=np.asarray("direct"),
        )
        write_status(status, started=started, current=total, total=total, phase="complete", metrics={"output": str(output)})
        print(f"Saved {output}", flush=True)
    except Exception as error:
        persist_partial(partial, payload)
        write_status(status, started=started, current=current, total=total, phase="failed", message=str(error))
        raise


if __name__ == "__main__":
    main()
