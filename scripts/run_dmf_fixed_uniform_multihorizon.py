from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_dmf_83_region_oracle_phi_eid import (
    DEFAULT_SOURCE_RESULTS,
    dmf_step_batch,
    gaussian_singleton_source_phi,
    load_dmf_module,
    resolve_path,
    standardize,
)

DEFAULT_OUTPUT = ROOT / "results" / "dmf_fixed_uniform_multihorizon" / "chunk.npz"


def fixed_uniform_sources(
    rng: np.random.Generator, *, sample_count: int, dimension: int, low: float = 0.0, high: float = 1.0,
) -> np.ndarray:
    if not 0.0 <= float(low) < float(high) <= 1.0:
        raise ValueError("Uniform intervention support must satisfy 0 <= low < high <= 1.")
    return rng.uniform(float(low), float(high), size=(int(sample_count), int(dimension)))


def fixed_uniform_initial_state(
    rng: np.random.Generator, *, sample_count: int, dimension: int, source_state: str,
    low: float = 0.0, high: float = 1.0,
    se_low: float | None = None, se_high: float | None = None,
    si_low: float | None = None, si_high: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    se_support = (float(low) if se_low is None else float(se_low), float(high) if se_high is None else float(se_high))
    si_support = (float(low) if si_low is None else float(si_low), float(high) if si_high is None else float(si_high))
    source_se = fixed_uniform_sources(
        rng, sample_count=sample_count, dimension=dimension, low=se_support[0], high=se_support[1],
    )
    if source_state == "se":
        return source_se, None
    if source_state == "se_si":
        return source_se, fixed_uniform_sources(
            rng, sample_count=sample_count, dimension=dimension, low=si_support[0], high=si_support[1],
        )
    raise ValueError(f"Unsupported source state: {source_state}")


def select_target_state(
    target_se: np.ndarray, target_si: np.ndarray, *, target_state: str,
) -> np.ndarray:
    if target_state == "se":
        return np.asarray(target_se, dtype=float)
    if target_state == "se_si":
        return np.concatenate((np.asarray(target_se, dtype=float), np.asarray(target_si, dtype=float)), axis=1)
    raise ValueError(f"Unsupported target state: {target_state}")


def background_indices(
    rng: np.random.Generator, *, trace_length: int, max_horizon: int, sample_count: int,
) -> np.ndarray:
    count = int(trace_length) - int(max_horizon) + 1
    if count <= 0:
        raise ValueError("Stable trace is shorter than the largest requested horizon.")
    return rng.integers(0, count, size=int(sample_count))


def rollout_to_horizons(
    source_se: np.ndarray,
    source_si: np.ndarray,
    *,
    horizons: Sequence[int],
    step: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    ordered = tuple(sorted({int(horizon) for horizon in horizons}))
    if not ordered or ordered[0] < 1:
        raise ValueError("horizons must contain positive integration-step counts.")
    se = np.asarray(source_se, dtype=float).copy()
    si = np.asarray(source_si, dtype=float).copy()
    targets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for step_index in range(1, ordered[-1] + 1):
        se, si = step(se, si)
        if step_index in ordered:
            targets[step_index] = (se.copy(), si.copy())
    return targets


def target_diagnostics(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    source_array = np.asarray(source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    source_variance = float(np.mean(np.var(source_array, axis=0, ddof=1)))
    target_variance = float(np.mean(np.var(target_array, axis=0, ddof=1)))
    centered = target_array - target_array.mean(axis=0, keepdims=True)
    scale = target_array.std(axis=0, ddof=1)
    valid = scale > 1.0e-12
    if int(np.sum(valid)) < 2:
        mean_offdiag_correlation = float("nan")
    else:
        normalized = centered[:, valid] / scale[valid]
        correlation = normalized.T @ normalized / float(len(target_array) - 1)
        mask = ~np.eye(correlation.shape[0], dtype=bool)
        mean_offdiag_correlation = float(np.mean(correlation[mask]))
    return {
        "target_variance_retained": target_variance / source_variance if source_variance > 0.0 else float("nan"),
        "target_spatial_sd": float(np.mean(np.std(target_array, axis=1, ddof=0))),
        "target_mean_offdiag_correlation": mean_offdiag_correlation,
    }


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired fixed-uniform DMF multi-horizon EI sweep.")
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=parse_int_list, default=(0,))
    parser.add_argument("--g-indices", type=parse_int_list, default=None)
    parser.add_argument("--horizons", type=parse_int_list, default=(1, 10, 50, 100, 300))
    parser.add_argument("--source-state", choices=("se", "se_si"), default="se")
    parser.add_argument("--target-state", choices=("se", "se_si"), default="se")
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--t-total", type=float, default=1.05)
    parser.add_argument("--burn-in", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--stabilization-window", type=float, default=0.05)
    parser.add_argument("--stabilization-tolerance", type=float, default=0.15)
    parser.add_argument("--stabilization-confirm-windows", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = tuple(sorted({int(value) for value in args.horizons}))
    if not horizons or horizons[0] < 1:
        raise ValueError("--horizons must contain positive step counts.")
    dmf = load_dmf_module()
    archive = np.load(resolve_path(args.source_results))
    all_g = np.asarray(archive["G"], dtype=float)
    selected = np.asarray(args.g_indices if args.g_indices is not None else np.arange(all_g.size), dtype=int)
    g_values = all_g[selected]
    connectivity = np.asarray(archive["connectivity"], dtype=float)
    j_fic = np.asarray(archive["j_fic"], dtype=float)[selected]
    parameters = dmf.DMFParameters(
        t_total=float(args.t_total), burn_in=float(args.burn_in), dt=float(args.dt), sigma=float(args.sigma),
    )
    stabilization = dmf.StabilizationParameters(
        window=float(args.stabilization_window),
        tolerance_hz=float(args.stabilization_tolerance),
        confirm_windows=int(args.stabilization_confirm_windows),
    )
    shape = (len(args.seeds), len(g_values), len(horizons))
    metrics = {name: np.full(shape, np.nan, dtype=float) for name in (
        "whole_ei", "singleton_ei_sum", "phi_eid", "target_variance_retained",
        "target_spatial_sd", "target_mean_offdiag_correlation",
        "target_entropy", "joint_conditional_entropy", "singleton_conditional_entropy_sum",
    )}
    clip_fraction = np.zeros(shape, dtype=float)

    for seed_index, seed in enumerate(args.seeds):
        initial_se = initial_si = None
        for g_index, coupling_g in enumerate(g_values):
            simulation = dmf.simulate_dmf(
                connectivity, float(coupling_g), np.asarray(j_fic[g_index], dtype=float),
                parameters=parameters, stabilization_parameters=stabilization,
                seed=int(seed) + int(selected[g_index]), initial_se=initial_se, initial_si=initial_si,
                record_rate_trace=False, record_state_trace=True,
            )
            initial_se = np.asarray(simulation["final_se"], dtype=float)
            initial_si = np.asarray(simulation["final_si"], dtype=float)
            start = int(float(simulation["stabilization_start_step"]))
            si_trace = np.asarray(simulation["state_si_trace"], dtype=float)[start:]
            rng = np.random.default_rng(int(seed) * 100_000 + int(selected[g_index]) * 1_000)
            source_se, source_si_fixed = fixed_uniform_initial_state(
                rng, sample_count=int(args.sample_count), dimension=connectivity.shape[0], source_state=args.source_state,
            )
            sampled_background_indices = background_indices(
                rng, trace_length=len(si_trace), max_horizon=horizons[-1], sample_count=int(args.sample_count),
            )
            source_si = si_trace[sampled_background_indices].copy() if source_si_fixed is None else source_si_fixed
            source_z, _, _ = standardize(select_target_state(source_se, source_si, target_state=args.source_state))

            def step(current_se: np.ndarray, current_si: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                return dmf_step_batch(
                    dmf, current_se, current_si, connectivity=connectivity, coupling_g=float(coupling_g),
                    j_fic=np.asarray(j_fic[g_index], dtype=float), parameters=parameters, rng=rng,
                )

            targets = rollout_to_horizons(source_se, source_si, horizons=horizons, step=step)
            for horizon_index, horizon in enumerate(horizons):
                target_se, target_si = targets[horizon]
                target_z, _, _ = standardize(
                    select_target_state(target_se, target_si, target_state=args.target_state)
                )
                ei = gaussian_singleton_source_phi(source_z, target_z, ridge=float(args.ridge))
                diagnostics = target_diagnostics(source_se, target_se)
                metrics["whole_ei"][seed_index, g_index, horizon_index] = float(ei["joint_ei"])
                metrics["singleton_ei_sum"][seed_index, g_index, horizon_index] = float(ei["singleton_ei_sum"])
                metrics["phi_eid"][seed_index, g_index, horizon_index] = float(ei["raw_phi"])
                metrics["target_entropy"][seed_index, g_index, horizon_index] = float(ei["target_entropy"])
                metrics["joint_conditional_entropy"][seed_index, g_index, horizon_index] = float(
                    ei["joint_conditional_entropy"]
                )
                metrics["singleton_conditional_entropy_sum"][seed_index, g_index, horizon_index] = float(
                    ei["singleton_conditional_entropy_sum"]
                )
                for name, value in diagnostics.items():
                    metrics[name][seed_index, g_index, horizon_index] = float(value)
            print(f"seed={seed} G={coupling_g:.1f} horizons={horizons} clip=0.00%", flush=True)

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output, G=g_values, selected_g_indices=selected, seeds=np.asarray(args.seeds, dtype=int),
        horizons=np.asarray(horizons, dtype=int), clip_fraction=clip_fraction,
        sample_count=int(args.sample_count), ridge=float(args.ridge), dt=float(args.dt),
        source_state=str(args.source_state), target_state=str(args.target_state),
        source_count=int(source_z.shape[1]), **metrics,
    )
    summary = {
        "source": f"fixed independent U(0,1)^{83 if args.source_state == 'se' else 166}, no clipping",
        "paired_background": True,
        "horizons": list(horizons), "source_state": str(args.source_state), "target_state": str(args.target_state),
        "seeds": list(args.seeds), "g_values": g_values.tolist(),
        "sample_count": int(args.sample_count), "max_clip_fraction": 0.0,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
