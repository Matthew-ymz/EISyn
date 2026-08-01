#!/usr/bin/env python3
"""Run paired Schaefer100 DMF critical-band or target-horizon experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_dmf_diffusive_fullstate_control import (
    dmf_step_batch_with_operator,
    rollout,
)
from scripts.run_dmf_fixed_uniform_multihorizon import (
    fixed_uniform_initial_state,
    rollout_to_horizons,
    target_diagnostics,
)
from scripts.validate_dmf_83_region_oracle_phi_eid import (
    gaussian_singleton_source_phi,
    load_dmf_module,
    standardize,
)


SOURCE = ROOT / "results" / "dmf_schaefer100" / "source" / "group_mean_native_mean_rate.npz"
CRITICAL_DIR = ROOT / "results" / "dmf_schaefer100" / "critical_diagnostics"
HORIZON_DIR = ROOT / "results" / "dmf_schaefer100" / "multihorizon"
SYN_TOLERANCE_BITS = 1.0e-8


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())


def parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("critical", "horizon"), required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", type=parse_int_list)
    parser.add_argument("--g-values", type=parse_float_list)
    parser.add_argument("--horizons", type=parse_int_list)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--trace-steps", type=int)
    parser.add_argument("--trace-burn-steps", type=int)
    parser.add_argument("--perturbation", type=float, default=1.0e-3)
    parser.add_argument("--response-steps", type=int, default=100)
    parser.add_argument("--fixed-point-steps", type=int, default=5000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    full = args.mode == "full"
    if args.seeds is None:
        args.seeds = tuple(range(3, 11)) if full else (3,)
    if args.sample_count is None:
        args.sample_count = 2048 if full else 256
    if args.experiment == "critical":
        if args.g_values is None:
            args.g_values = (
                tuple(np.round(np.arange(1.10, 1.7001, 0.02), 2))
                if full
                else (1.20, 1.30, 1.40)
            )
        if args.horizons is None:
            args.horizons = (300,) if full else (30,)
        if args.trace_steps is None:
            args.trace_steps = 6000 if full else 1200
        if args.trace_burn_steps is None:
            args.trace_burn_steps = 1000 if full else 200
        if args.output_dir is None:
            args.output_dir = CRITICAL_DIR / args.mode
    else:
        if args.g_values is None:
            args.g_values = (
                tuple(np.round(np.arange(0.0, 3.0001, 0.1), 1))
                if full
                else (1.20, 1.30, 1.40)
            )
        if args.horizons is None:
            args.horizons = (50, 100, 200, 300, 400, 500) if full else (10, 20, 30)
        if args.trace_steps is None:
            args.trace_steps = 0
        if args.trace_burn_steps is None:
            args.trace_burn_steps = 0
        if args.output_dir is None:
            args.output_dir = HORIZON_DIR / args.mode
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    args.source = args.source if args.source.is_absolute() else ROOT / args.source
    return args


def transfer_derivative(
    dmf,
    current: np.ndarray,
    *,
    gain: float,
    threshold: float,
    shape: float,
) -> np.ndarray:
    scale = 1.0e-6 * np.maximum(1.0, np.abs(current))
    upper = dmf.transfer_function(
        current + scale, gain=gain, threshold=threshold, shape=shape
    )
    lower = dmf.transfer_function(
        current - scale, gain=gain, threshold=threshold, shape=shape
    )
    return (upper - lower) / (2.0 * scale)


def rates_and_drift(
    dmf,
    se: np.ndarray,
    si: np.ndarray,
    *,
    connectivity: np.ndarray,
    coupling_g: float,
    j_fic: np.ndarray,
    parameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    input_e = (
        parameters.w_e * parameters.i0
        + parameters.w_plus * parameters.j_nmda * se
        + float(coupling_g) * parameters.j_nmda * (connectivity @ se)
        - j_fic * si
    )
    input_i = parameters.w_i * parameters.i0 + parameters.j_nmda * se - si
    rate_e = dmf.transfer_function(
        input_e,
        gain=parameters.gain_e,
        threshold=parameters.threshold_e,
        shape=parameters.shape_e,
    )
    rate_i = dmf.transfer_function(
        input_i,
        gain=parameters.gain_i,
        threshold=parameters.threshold_i,
        shape=parameters.shape_i,
    )
    drift_e = -se / parameters.tau_e + (1.0 - se) * parameters.gamma_e * rate_e
    drift_i = -si / parameters.tau_i + rate_i
    return rate_e, rate_i, drift_e, drift_i


def deterministic_fixed_point(
    dmf,
    *,
    initial_se: np.ndarray,
    initial_si: np.ndarray,
    connectivity: np.ndarray,
    coupling_g: float,
    j_fic: np.ndarray,
    parameters,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    se = np.asarray(initial_se, dtype=float).copy()
    si = np.asarray(initial_si, dtype=float).copy()
    drift_norm = np.inf
    for _ in range(int(steps)):
        _, _, drift_e, drift_i = rates_and_drift(
            dmf,
            se,
            si,
            connectivity=connectivity,
            coupling_g=coupling_g,
            j_fic=j_fic,
            parameters=parameters,
        )
        se += parameters.dt * drift_e
        si += parameters.dt * drift_i
        drift_norm = float(np.linalg.norm(np.concatenate((drift_e, drift_i))))
    return se, si, drift_norm


def drift_jacobian(
    dmf,
    *,
    se: np.ndarray,
    si: np.ndarray,
    connectivity: np.ndarray,
    coupling_g: float,
    j_fic: np.ndarray,
    parameters,
) -> np.ndarray:
    n = len(se)
    input_e = (
        parameters.w_e * parameters.i0
        + parameters.w_plus * parameters.j_nmda * se
        + float(coupling_g) * parameters.j_nmda * (connectivity @ se)
        - j_fic * si
    )
    input_i = parameters.w_i * parameters.i0 + parameters.j_nmda * se - si
    rate_e = dmf.transfer_function(
        input_e,
        gain=parameters.gain_e,
        threshold=parameters.threshold_e,
        shape=parameters.shape_e,
    )
    derivative_e = transfer_derivative(
        dmf,
        input_e,
        gain=parameters.gain_e,
        threshold=parameters.threshold_e,
        shape=parameters.shape_e,
    )
    derivative_i = transfer_derivative(
        dmf,
        input_i,
        gain=parameters.gain_i,
        threshold=parameters.threshold_i,
        shape=parameters.shape_i,
    )
    eye = np.eye(n)
    e_gain = (1.0 - se) * parameters.gamma_e * derivative_e
    ee_input = (
        parameters.w_plus * parameters.j_nmda * eye
        + float(coupling_g) * parameters.j_nmda * connectivity
    )
    jac_ee = (
        -eye / parameters.tau_e
        - np.diag(parameters.gamma_e * rate_e)
        + e_gain[:, None] * ee_input
    )
    jac_ei = e_gain[:, None] * (-np.diag(j_fic))
    jac_ie = derivative_i[:, None] * (parameters.j_nmda * eye)
    jac_ii = -eye / parameters.tau_i - derivative_i[:, None] * eye
    return np.block([[jac_ee, jac_ei], [jac_ie, jac_ii]])


def perturbation_susceptibility(
    dmf,
    *,
    se: np.ndarray,
    si: np.ndarray,
    connectivity: np.ndarray,
    coupling_g: float,
    j_fic: np.ndarray,
    parameters,
    perturbation: float,
    response_steps: int,
) -> tuple[float, float]:
    baseline_se = se[None, :].copy()
    baseline_si = si[None, :].copy()
    perturbed_se = baseline_se + float(perturbation)
    perturbed_si = baseline_si.copy()
    initial_norm = float(np.linalg.norm(perturbed_se - baseline_se))
    ratios = []
    zero_rng = np.random.default_rng(0)
    for _ in range(int(response_steps)):
        baseline_se, baseline_si = dmf_step_batch_with_operator(
            dmf,
            baseline_se,
            baseline_si,
            connectivity=connectivity,
            coupling_g=coupling_g,
            j_fic=j_fic,
            parameters=parameters,
            mode="direct",
            state_boundary="none",
            rng=zero_rng,
        )
        perturbed_se, perturbed_si = dmf_step_batch_with_operator(
            dmf,
            perturbed_se,
            perturbed_si,
            connectivity=connectivity,
            coupling_g=coupling_g,
            j_fic=j_fic,
            parameters=parameters,
            mode="direct",
            state_boundary="none",
            rng=zero_rng,
        )
        separation = np.concatenate(
            (perturbed_se - baseline_se, perturbed_si - baseline_si), axis=1
        )
        ratios.append(float(np.linalg.norm(separation) / initial_norm))
    return float(np.mean(ratios)), float(np.max(ratios))


def stochastic_rate_trace(
    dmf,
    *,
    initial_se: np.ndarray,
    initial_si: np.ndarray,
    connectivity: np.ndarray,
    coupling_g: float,
    j_fic: np.ndarray,
    parameters,
    steps: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    se = initial_se[None, :].copy()
    si = initial_si[None, :].copy()
    rates = np.empty((int(steps), connectivity.shape[0]), dtype=float)
    outside = 0
    rng = np.random.default_rng(int(seed))
    for index in range(int(steps)):
        rate_e, _, _, _ = rates_and_drift(
            dmf,
            se[0],
            si[0],
            connectivity=connectivity,
            coupling_g=coupling_g,
            j_fic=j_fic,
            parameters=parameters,
        )
        rates[index] = rate_e
        se, si = dmf_step_batch_with_operator(
            dmf,
            se,
            si,
            connectivity=connectivity,
            coupling_g=coupling_g,
            j_fic=j_fic,
            parameters=parameters,
            mode="direct",
            state_boundary="none",
            rng=rng,
        )
        outside += int(np.count_nonzero((se < 0.0) | (se > 1.0)))
        outside += int(np.count_nonzero((si < 0.0) | (si > 1.0)))
    denominator = 2 * int(steps) * connectivity.shape[0]
    return rates, float(outside / denominator)


def neural_rate_metastability(
    rates: np.ndarray,
    *,
    dt: float,
    burn_steps: int,
) -> tuple[float, float, float]:
    values = np.asarray(rates, dtype=float)[int(burn_steps) :]
    if values.shape[0] < 200:
        raise ValueError("Rate trace is too short for the metastability calculation.")
    sample_rate = 1.0 / float(dt)
    low_hz = max(1.0, 2.0 / (values.shape[0] * float(dt)))
    high_hz = min(40.0, 0.4 * sample_rate)
    if low_hz >= high_hz:
        raise ValueError("Invalid neural-rate band for metastability.")
    centered = values - values.mean(axis=0, keepdims=True)
    sos = butter(3, (low_hz, high_hz), btype="bandpass", fs=sample_rate, output="sos")
    filtered = sosfiltfilt(sos, centered, axis=0)
    phases = np.angle(hilbert(filtered, axis=0))
    order = np.abs(np.mean(np.exp(1j * phases), axis=1))
    global_rate = np.mean(values, axis=1)
    susceptibility = values.shape[1] * float(np.var(global_rate, ddof=1))
    return float(np.std(order, ddof=1)), float(np.mean(order)), susceptibility


def check_syn_nonnegative(values: np.ndarray, *, context: str) -> tuple[int, float]:
    array = np.asarray(values, dtype=float)
    invalid = array < -SYN_TOLERANCE_BITS
    minimum = float(np.nanmin(array))
    if np.any(invalid):
        raise RuntimeError(
            f"{context}: PEID Syn nonnegativity violation: minimum={minimum:.12g} bits, "
            f"threshold={-SYN_TOLERANCE_BITS:.12g} bits, affected={int(np.count_nonzero(invalid))}."
        )
    near_zero = (array < 0.0) & ~invalid
    return int(np.count_nonzero(near_zero)), minimum


def condition_rng_seed(seed: int, coupling_g: float, offset: int) -> int:
    return int(seed) * 1_000_000 + int(round((float(coupling_g) + 1.0) * 10_000)) + int(offset)


def critical_seed(
    *,
    dmf,
    seed: int,
    g_values: np.ndarray,
    connectivity: np.ndarray,
    j_fic: np.ndarray,
    parameters,
    sample_count: int,
    horizon: int,
    ridge: float,
    trace_steps: int,
    trace_burn_steps: int,
    fixed_states: tuple[np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    n = connectivity.shape[0]
    source_rng = np.random.default_rng(int(seed) * 1_000_000 + 101)
    source_se, source_si_optional = fixed_uniform_initial_state(
        source_rng,
        sample_count=sample_count,
        dimension=n,
        source_state="se_si",
        se_low=0.30,
        se_high=0.70,
        si_low=0.30,
        si_high=0.70,
    )
    assert source_si_optional is not None
    source_si = source_si_optional
    source = np.concatenate((source_se, source_si), axis=1)
    source_z, _, _ = standardize(source)
    phi = np.empty(len(g_values), dtype=float)
    whole = np.empty(len(g_values), dtype=float)
    singleton = np.empty(len(g_values), dtype=float)
    metastability = np.empty(len(g_values), dtype=float)
    mean_order = np.empty(len(g_values), dtype=float)
    rate_susceptibility = np.empty(len(g_values), dtype=float)
    boundary_fraction = np.empty(len(g_values), dtype=float)
    for index, coupling_g in enumerate(g_values):
        noise_rng = np.random.default_rng(int(seed) * 1_000_000 + 303)
        target_se, target_si = rollout(
            dmf,
            source_se,
            source_si,
            connectivity=connectivity,
            coupling_g=float(coupling_g),
            j_fic=j_fic,
            parameters=parameters,
            mode="direct",
            state_boundary="none",
            horizon=horizon,
            rng=noise_rng,
        )
        target = np.concatenate((target_se, target_si), axis=1)
        target_z, _, _ = standardize(target)
        result = gaussian_singleton_source_phi(source_z, target_z, ridge=ridge)
        whole[index] = float(result["joint_ei"])
        singleton[index] = float(result["singleton_ei_sum"])
        phi[index] = float(result["raw_phi"])
        rates, boundary_fraction[index] = stochastic_rate_trace(
            dmf,
            initial_se=fixed_states[0][index],
            initial_si=fixed_states[1][index],
            connectivity=connectivity,
            coupling_g=float(coupling_g),
            j_fic=j_fic,
            parameters=parameters,
            steps=trace_steps,
            seed=int(seed) * 1_000_000 + 707,
        )
        (
            metastability[index],
            mean_order[index],
            rate_susceptibility[index],
        ) = neural_rate_metastability(
            rates, dt=parameters.dt, burn_steps=trace_burn_steps
        )
        print(
            f"seed={seed} G={coupling_g:.2f} Xi={phi[index]:.6f} "
            f"metastability={metastability[index]:.6f}",
            flush=True,
        )
    check_syn_nonnegative(phi, context=f"critical seed={seed}")
    return {
        "phi_eid": phi,
        "whole_ei": whole,
        "singleton_ei_sum": singleton,
        "metastability": metastability,
        "mean_phase_order": mean_order,
        "rate_susceptibility": rate_susceptibility,
        "trace_boundary_fraction": boundary_fraction,
    }


def deterministic_diagnostics(
    *,
    dmf,
    g_values: np.ndarray,
    connectivity: np.ndarray,
    j_fic: np.ndarray,
    parameters,
    fixed_point_steps: int,
    perturbation: float,
    response_steps: int,
) -> dict[str, np.ndarray]:
    n = connectivity.shape[0]
    state_se = np.full(n, parameters.init_se, dtype=float)
    state_si = np.full(n, parameters.init_si, dtype=float)
    fixed_se = np.empty((len(g_values), n), dtype=float)
    fixed_si = np.empty((len(g_values), n), dtype=float)
    drift_norm = np.empty(len(g_values), dtype=float)
    jacobian_max_real = np.empty(len(g_values), dtype=float)
    susceptibility_mean = np.empty(len(g_values), dtype=float)
    susceptibility_peak = np.empty(len(g_values), dtype=float)
    deterministic_parameters = replace(parameters, sigma=0.0)
    for index, coupling_g in enumerate(g_values):
        state_se, state_si, drift_norm[index] = deterministic_fixed_point(
            dmf,
            initial_se=state_se,
            initial_si=state_si,
            connectivity=connectivity,
            coupling_g=float(coupling_g),
            j_fic=j_fic,
            parameters=deterministic_parameters,
            steps=fixed_point_steps,
        )
        fixed_se[index], fixed_si[index] = state_se, state_si
        jacobian = drift_jacobian(
            dmf,
            se=state_se,
            si=state_si,
            connectivity=connectivity,
            coupling_g=float(coupling_g),
            j_fic=j_fic,
            parameters=deterministic_parameters,
        )
        jacobian_max_real[index] = float(np.max(np.linalg.eigvals(jacobian).real))
        susceptibility_mean[index], susceptibility_peak[index] = perturbation_susceptibility(
            dmf,
            se=state_se,
            si=state_si,
            connectivity=connectivity,
            coupling_g=float(coupling_g),
            j_fic=j_fic,
            parameters=deterministic_parameters,
            perturbation=perturbation,
            response_steps=response_steps,
        )
    return {
        "fixed_se": fixed_se,
        "fixed_si": fixed_si,
        "fixed_point_drift_norm": drift_norm,
        "jacobian_max_real": jacobian_max_real,
        "susceptibility_mean": susceptibility_mean,
        "susceptibility_peak": susceptibility_peak,
    }


def horizon_seed(
    *,
    dmf,
    seed: int,
    g_values: np.ndarray,
    connectivity: np.ndarray,
    j_fic: np.ndarray,
    parameters,
    sample_count: int,
    horizons: tuple[int, ...],
    ridge: float,
) -> dict[str, np.ndarray]:
    shape = (len(g_values), len(horizons))
    names = (
        "phi_eid",
        "whole_ei",
        "singleton_ei_sum",
        "target_variance_retained",
        "target_spatial_sd",
        "target_mean_offdiag_correlation",
        "boundary_fraction",
    )
    values = {name: np.empty(shape, dtype=float) for name in names}
    n = connectivity.shape[0]
    for g_index, coupling_g in enumerate(g_values):
        source_rng = np.random.default_rng(condition_rng_seed(seed, coupling_g, 101))
        source_se, source_si_optional = fixed_uniform_initial_state(
            source_rng,
            sample_count=sample_count,
            dimension=n,
            source_state="se_si",
            se_low=0.30,
            se_high=0.70,
            si_low=0.30,
            si_high=0.70,
        )
        assert source_si_optional is not None
        source_si = source_si_optional
        source = np.concatenate((source_se, source_si), axis=1)
        source_z, _, _ = standardize(source)
        noise_rng = np.random.default_rng(condition_rng_seed(seed, coupling_g, 303))

        def step(current_se: np.ndarray, current_si: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return dmf_step_batch_with_operator(
                dmf,
                current_se,
                current_si,
                connectivity=connectivity,
                coupling_g=float(coupling_g),
                j_fic=j_fic,
                parameters=parameters,
                mode="direct",
                state_boundary="none",
                rng=noise_rng,
            )

        targets = rollout_to_horizons(
            source_se, source_si, horizons=horizons, step=step
        )
        for horizon_index, horizon in enumerate(horizons):
            target_se, target_si = targets[horizon]
            target = np.concatenate((target_se, target_si), axis=1)
            target_z, _, _ = standardize(target)
            result = gaussian_singleton_source_phi(source_z, target_z, ridge=ridge)
            values["whole_ei"][g_index, horizon_index] = float(result["joint_ei"])
            values["singleton_ei_sum"][g_index, horizon_index] = float(
                result["singleton_ei_sum"]
            )
            values["phi_eid"][g_index, horizon_index] = float(result["raw_phi"])
            diagnostics = target_diagnostics(source_se, target_se)
            for name, value in diagnostics.items():
                values[name][g_index, horizon_index] = float(value)
            outside = np.count_nonzero(
                (target_se < 0.0)
                | (target_se > 1.0)
                | (target_si < 0.0)
                | (target_si > 1.0)
            )
            values["boundary_fraction"][g_index, horizon_index] = float(
                outside / target.size
            )
        check_syn_nonnegative(values["phi_eid"][g_index], context=f"horizon seed={seed} G={coupling_g:.2f}")
        print(
            f"seed={seed} G={coupling_g:.2f} horizons={horizons} "
            f"Xi300={values['phi_eid'][g_index, int(np.argmin(np.abs(np.asarray(horizons) - 300)))]:.6f}",
            flush=True,
        )
    return values


def combine_chunks(
    chunk_paths: Iterable[Path],
    *,
    array_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    chunks = []
    for path in chunk_paths:
        with np.load(path) as archive:
            chunks.append({name: np.asarray(archive[name]) for name in array_names})
    return {
        name: np.stack([chunk[name] for chunk in chunks], axis=0)
        for name in array_names
    }


def main() -> None:
    args = resolve_defaults(parse_args())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "progress.json"
    started = time.monotonic()
    dmf = load_dmf_module()
    with np.load(args.source) as archive:
        connectivity = np.asarray(archive["connectivity"], dtype=float)
        j_fic = np.asarray(archive["j_fic"], dtype=float)[0]
    g_values = np.asarray(args.g_values, dtype=float)
    horizons = tuple(sorted({int(value) for value in args.horizons}))
    parameters = dmf.DMFParameters(
        t_total=1.0,
        burn_in=0.0,
        dt=float(args.dt),
        sigma=float(args.sigma),
    )
    deterministic: dict[str, np.ndarray] = {}
    if args.experiment == "critical":
        deterministic = deterministic_diagnostics(
            dmf=dmf,
            g_values=g_values,
            connectivity=connectivity,
            j_fic=j_fic,
            parameters=parameters,
            fixed_point_steps=int(args.fixed_point_steps),
            perturbation=float(args.perturbation),
            response_steps=int(args.response_steps),
        )
        deterministic_path = args.output_dir / "deterministic_diagnostics.npz"
        np.savez(deterministic_path, G=g_values, **deterministic)
    chunk_paths = []
    for seed_position, seed in enumerate(args.seeds):
        chunk_path = args.output_dir / f"seed_{int(seed):02d}.npz"
        chunk_paths.append(chunk_path)
        if chunk_path.exists() and not args.force:
            print(f"Reusing {chunk_path}", flush=True)
        else:
            if args.experiment == "critical":
                payload = critical_seed(
                    dmf=dmf,
                    seed=int(seed),
                    g_values=g_values,
                    connectivity=connectivity,
                    j_fic=j_fic,
                    parameters=parameters,
                    sample_count=int(args.sample_count),
                    horizon=int(horizons[0]),
                    ridge=float(args.ridge),
                    trace_steps=int(args.trace_steps),
                    trace_burn_steps=int(args.trace_burn_steps),
                    fixed_states=(deterministic["fixed_se"], deterministic["fixed_si"]),
                )
            else:
                payload = horizon_seed(
                    dmf=dmf,
                    seed=int(seed),
                    g_values=g_values,
                    connectivity=connectivity,
                    j_fic=j_fic,
                    parameters=parameters,
                    sample_count=int(args.sample_count),
                    horizons=horizons,
                    ridge=float(args.ridge),
                )
            np.savez(
                chunk_path,
                seed=int(seed),
                G=g_values,
                horizons=np.asarray(horizons, dtype=int),
                **payload,
            )
        elapsed = time.monotonic() - started
        completed = seed_position + 1
        rate = completed / elapsed if elapsed > 0 else 0.0
        atomic_json(
            status_path,
            {
                "phase": args.experiment,
                "current": completed,
                "total": len(args.seeds),
                "unit": "seed",
                "elapsed_seconds": elapsed,
                "eta_seconds": (len(args.seeds) - completed) / rate if rate > 0 else None,
                "metrics": {"last_seed": int(seed), "mode": args.mode},
                "updated_at": time.time(),
            },
        )
    if args.experiment == "critical":
        array_names = (
            "phi_eid",
            "whole_ei",
            "singleton_ei_sum",
            "metastability",
            "mean_phase_order",
            "rate_susceptibility",
            "trace_boundary_fraction",
        )
    else:
        array_names = (
            "phi_eid",
            "whole_ei",
            "singleton_ei_sum",
            "target_variance_retained",
            "target_spatial_sd",
            "target_mean_offdiag_correlation",
            "boundary_fraction",
        )
    combined = combine_chunks(chunk_paths, array_names=array_names)
    near_zero_count, minimum_syn = check_syn_nonnegative(
        combined["phi_eid"], context=f"{args.experiment} combined"
    )
    output = args.output_dir / "results.npz"
    np.savez(
        output,
        experiment=args.experiment,
        mode=args.mode,
        G=g_values,
        seeds=np.asarray(args.seeds, dtype=int),
        horizons=np.asarray(horizons, dtype=int),
        sample_count=int(args.sample_count),
        ridge=float(args.ridge),
        dt=float(args.dt),
        sigma=float(args.sigma),
        syn_nonnegative_tolerance_bits=SYN_TOLERANCE_BITS,
        syn_near_zero_count=near_zero_count,
        syn_minimum_bits=minimum_syn,
        **combined,
        **{
            name: value
            for name, value in deterministic.items()
            if name not in {"fixed_se", "fixed_si"}
        },
    )
    if args.experiment == "critical":
        peak_indices = np.argmax(combined["phi_eid"], axis=1)
        summary_extra = {
            "xi_peak_G_by_seed": g_values[peak_indices].tolist(),
            "xi_peak_G_mean": float(np.mean(g_values[peak_indices])),
            "susceptibility_peak_G": float(
                g_values[np.argmax(np.mean(combined["rate_susceptibility"], axis=0))]
            ),
            "metastability_peak_G_by_seed": g_values[
                np.argmax(combined["metastability"], axis=1)
            ].tolist(),
            "jacobian_closest_to_zero_G": float(
                g_values[np.argmin(np.abs(deterministic["jacobian_max_real"]))]
            ),
        }
    else:
        peak_indices = np.argmax(combined["phi_eid"], axis=1)
        summary_extra = {
            "xi_peak_G_by_seed_and_horizon": g_values[peak_indices].tolist(),
            "xi_peak_G_mean_by_horizon": np.mean(g_values[peak_indices], axis=0).tolist(),
        }
    summary = {
        "experiment": args.experiment,
        "mode": args.mode,
        "treatment": "global coupling G" if args.experiment == "critical" else "target horizon",
        "G": g_values.tolist(),
        "horizons": list(horizons),
        "seeds": list(args.seeds),
        "sample_count": int(args.sample_count),
        "source": "paired independent U(0.30,0.70)^200 full E/I intervention",
        "target": "complete 200-dimensional E/I future state",
        "state_boundary": "none",
        "estimator": f"Gaussian conditional covariance, ridge={args.ridge:g}",
        "syn_nonnegative_audit": {
            "tolerance_bits": SYN_TOLERANCE_BITS,
            "near_zero_count": near_zero_count,
            "minimum_bits": minimum_syn,
        },
        "maximum_boundary_fraction": float(
            np.nanmax(
                combined[
                    "trace_boundary_fraction"
                    if args.experiment == "critical"
                    else "boundary_fraction"
                ]
            )
        ),
        **summary_extra,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(
        status_path,
        {
            "phase": "complete",
            "current": len(args.seeds),
            "total": len(args.seeds),
            "unit": "seed",
            "elapsed_seconds": time.monotonic() - started,
            "eta_seconds": 0.0,
            "metrics": {"output": str(output.relative_to(ROOT))},
            "updated_at": time.time(),
        },
    )
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
