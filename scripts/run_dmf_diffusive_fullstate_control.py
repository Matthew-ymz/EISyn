from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_dmf_fixed_uniform_multihorizon import fixed_uniform_initial_state, target_diagnostics
from scripts.validate_dmf_83_region_oracle_phi_eid import (
    DEFAULT_SOURCE_RESULTS,
    gaussian_singleton_source_phi,
    load_dmf_module,
    resolve_path,
    standardize,
)


DEFAULT_OUTPUT = ROOT / "results" / "dmf_diffusive_coupling_control" / "smoke.npz"
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


def parse_str_list(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def row_normalize_connectivity(connectivity: np.ndarray) -> np.ndarray:
    matrix = np.asarray(connectivity, dtype=float).copy()
    row_sum = matrix.sum(axis=1, keepdims=True)
    isolated = np.flatnonzero(row_sum[:, 0] <= 0.0)
    if isolated.size:
        matrix[isolated, isolated] = 1.0
        row_sum = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sum


def long_range_term(state: np.ndarray, connectivity: np.ndarray, *, mode: str) -> np.ndarray:
    neighbor_input = np.asarray(state, dtype=float) @ np.asarray(connectivity, dtype=float).T
    if mode == "direct":
        return neighbor_input
    if mode == "diffusive":
        degree = np.asarray(connectivity, dtype=float).sum(axis=1, keepdims=True).T
        return neighbor_input - np.asarray(state, dtype=float) * degree
    raise ValueError(f"Unsupported coupling mode: {mode}")


def dmf_step_batch_with_operator(
    dmf,
    se: np.ndarray,
    si: np.ndarray,
    *,
    connectivity: np.ndarray,
    coupling_g: float,
    j_fic: np.ndarray,
    parameters,
    mode: str,
    state_boundary: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    dt = float(parameters.dt)
    input_e = (
        parameters.w_e * parameters.i0
        + parameters.w_plus * parameters.j_nmda * se
        + float(coupling_g) * parameters.j_nmda * long_range_term(se, connectivity, mode=mode)
        - np.asarray(j_fic, dtype=float).reshape(1, -1) * si
    )
    input_i = parameters.w_i * parameters.i0 + parameters.j_nmda * se - si
    rate_e = dmf.transfer_function(
        input_e, gain=parameters.gain_e, threshold=parameters.threshold_e, shape=parameters.shape_e,
    )
    rate_i = dmf.transfer_function(
        input_i, gain=parameters.gain_i, threshold=parameters.threshold_i, shape=parameters.shape_i,
    )
    noise_e = parameters.sigma * np.sqrt(dt) * rng.standard_normal(se.shape)
    noise_i = parameters.sigma * np.sqrt(dt) * rng.standard_normal(si.shape)
    dse = dt * (-se / parameters.tau_e + (1.0 - se) * parameters.gamma_e * rate_e) + noise_e
    dsi = dt * (-si / parameters.tau_i + rate_i) + noise_i
    next_se, next_si = se + dse, si + dsi
    if state_boundary == "none":
        return next_se, next_si
    if state_boundary == "clip":
        return np.clip(next_se, 0.0, 1.0), np.clip(next_si, 0.0, 1.0)
    raise ValueError(f"Unsupported state boundary mode: {state_boundary}")


def rollout(
    dmf,
    source_se: np.ndarray,
    source_si: np.ndarray,
    *,
    connectivity: np.ndarray,
    coupling_g: float,
    j_fic: np.ndarray,
    parameters,
    mode: str,
    state_boundary: str,
    horizon: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    se = np.asarray(source_se, dtype=float).copy()
    si = np.asarray(source_si, dtype=float).copy()
    for _ in range(int(horizon)):
        se, si = dmf_step_batch_with_operator(
            dmf, se, si, connectivity=connectivity, coupling_g=coupling_g,
            j_fic=j_fic, parameters=parameters, mode=mode, state_boundary=state_boundary, rng=rng,
        )
    return se, si


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired direct-versus-diffusive DMF full-state EI control.")
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=parse_int_list, default=(0,))
    parser.add_argument("--g-indices", type=parse_int_list, default=(0, 6, 8, 12, 20))
    parser.add_argument("--modes", type=parse_str_list, default=("direct", "diffusive"))
    parser.add_argument("--source-state", choices=("se", "se_si"), default="se_si")
    parser.add_argument("--intervention-low", type=float, default=0.0)
    parser.add_argument("--intervention-high", type=float, default=1.0)
    parser.add_argument("--se-low", type=float)
    parser.add_argument("--se-high", type=float)
    parser.add_argument("--si-low", type=float)
    parser.add_argument("--si-high", type=float)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--state-boundary", choices=("clip", "none"), default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    valid_modes = {"direct", "diffusive"}
    if not set(args.modes).issubset(valid_modes):
        raise ValueError(f"--modes must be a subset of {sorted(valid_modes)}")
    dmf = load_dmf_module()
    archive = np.load(resolve_path(args.source_results))
    all_g = np.asarray(archive["G"], dtype=float)
    selected = np.asarray(args.g_indices, dtype=int)
    g_values = all_g[selected]
    connectivity = np.asarray(archive["connectivity"], dtype=float)
    j_fic = np.asarray(archive["j_fic"], dtype=float)[selected]
    parameters = dmf.DMFParameters(t_total=1.0, burn_in=0.0, dt=float(args.dt), sigma=float(args.sigma))
    shape = (len(args.modes), len(args.seeds), len(g_values))
    metrics = {name: np.full(shape, np.nan, dtype=float) for name in METRICS}
    clip_fraction = np.zeros(shape, dtype=float)
    se_low = float(args.intervention_low) if args.se_low is None else float(args.se_low)
    se_high = float(args.intervention_high) if args.se_high is None else float(args.se_high)
    si_low = float(args.intervention_low) if args.si_low is None else float(args.si_low)
    si_high = float(args.intervention_high) if args.si_high is None else float(args.si_high)

    for seed_index, seed in enumerate(args.seeds):
        for g_position, coupling_g in enumerate(g_values):
            schedule_index = int(np.rint((float(coupling_g) - 1.0) / 0.1))
            source_rng = np.random.default_rng(int(seed) * 100_000 + schedule_index * 1_000)
            source_se, source_si_optional = fixed_uniform_initial_state(
                source_rng, sample_count=int(args.sample_count), dimension=connectivity.shape[0],
                source_state=args.source_state, low=float(args.intervention_low), high=float(args.intervention_high),
                se_low=se_low, se_high=se_high, si_low=si_low, si_high=si_high,
            )
            source_si = np.zeros_like(source_se) if source_si_optional is None else source_si_optional
            source = source_se if args.source_state == "se" else np.concatenate((source_se, source_si), axis=1)
            source_z, _, _ = standardize(source)
            for mode_index, mode in enumerate(args.modes):
                noise_rng = np.random.default_rng(
                    int(seed) * 100_000 + schedule_index * 1_000 + 17
                )
                target_se, target_si = rollout(
                    dmf, source_se, source_si, connectivity=connectivity, coupling_g=float(coupling_g),
                    j_fic=np.asarray(j_fic[g_position], dtype=float), parameters=parameters, mode=mode,
                    state_boundary=str(args.state_boundary), horizon=int(args.horizon), rng=noise_rng,
                )
                target = np.concatenate((target_se, target_si), axis=1)
                target_z, _, _ = standardize(target)
                result = gaussian_singleton_source_phi(source_z, target_z, ridge=float(args.ridge))
                diagnostics = target_diagnostics(source_se, target_se)
                metrics["whole_ei"][mode_index, seed_index, g_position] = float(result["joint_ei"])
                metrics["singleton_ei_sum"][mode_index, seed_index, g_position] = float(result["singleton_ei_sum"])
                metrics["phi_eid"][mode_index, seed_index, g_position] = float(result["raw_phi"])
                metrics["target_entropy"][mode_index, seed_index, g_position] = float(result["target_entropy"])
                metrics["joint_conditional_entropy"][mode_index, seed_index, g_position] = float(
                    result["joint_conditional_entropy"]
                )
                metrics["singleton_conditional_entropy_sum"][mode_index, seed_index, g_position] = float(
                    result["singleton_conditional_entropy_sum"]
                )
                for name, value in diagnostics.items():
                    metrics[name][mode_index, seed_index, g_position] = float(value)
            print(f"seed={seed} G={coupling_g:.1f} modes={args.modes} clip=0.00%", flush=True)

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    se_support = f"U({se_low:.2f},{se_high:.2f})"
    si_support = f"U({si_low:.2f},{si_high:.2f})"
    n_regions = int(connectivity.shape[0])
    source_target = (
        f"independent {se_support}^{n_regions} sE source with sI=0 background to full-state future target"
        if args.source_state == "se"
        else (
            f"independent {se_support}^{n_regions} sE and "
            f"{si_support}^{n_regions} sI source to full-state future target"
        )
    )
    np.savez(
        output,
        G=g_values,
        selected_g_indices=selected,
        modes=np.asarray(args.modes),
        seeds=np.asarray(args.seeds, dtype=int),
        horizon=int(args.horizon),
        sample_count=int(args.sample_count),
        source_count=connectivity.shape[0] if args.source_state == "se" else 2 * connectivity.shape[0],
        source_state=str(args.source_state),
        intervention_low=float(args.intervention_low),
        intervention_high=float(args.intervention_high),
        se_intervention_low=se_low,
        se_intervention_high=se_high,
        si_intervention_low=si_low,
        si_intervention_high=si_high,
        state_boundary=str(args.state_boundary),
        clip_fraction=clip_fraction,
        **metrics,
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "treatment": "long-range coupling operator",
                "modes": list(args.modes),
                "source_target": source_target,
                "horizon_steps": int(args.horizon),
                "state_boundary": str(args.state_boundary),
                "seeds": list(args.seeds),
                "g_values": g_values.tolist(),
                "sample_count": int(args.sample_count),
                "intervention_support": {
                    "sE": [se_low, se_high],
                    "sI": [si_low, si_high],
                },
                "max_clip_fraction": 0.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
