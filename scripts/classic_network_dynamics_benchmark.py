#!/usr/bin/env python3
"""Benchmark causal readouts on classical network dynamical systems."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_RESULT_DIR = ROOT / "results" / "classic_network_dynamics_benchmark"
DEFAULT_FIGURE_DIR = ROOT / "fig" / "classic_network_dynamics_benchmark"
DEFAULT_REPORT_PATH = ROOT / "docs" / "reports" / "granger_peid_mlp_comparison.md"
PART1_COMBINED_FIGURE_PATH = ROOT / "fig" / "part1_synergy_comparison" / "six_system_five_method_synergy_panels.png"
BENCHMARK_MODEL_NAMES = ("kuramoto", "coupled_rossler", "sis", "wilson_cowan")
LEGACY_MARKER = "## 附录：原共同驱动 sine 基准"
SIS_GATE_SWEEP_BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
KURAMOTO_COUPLING_VALUES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
KURAMOTO_FREQUENCY_DETUNING = 0.1
KURAMOTO_PHASE_POTENTIAL_STRENGTH = 0.2
KURAMOTO_PEID_DETAIL_COUPLINGS = (
    0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08, 0.1,
    0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0,
)
LARGE_KURAMOTO_COUPLINGS = (0.0, 0.4, 0.8, 1.1, 1.3, 1.5, 1.6, 1.7, 1.9, 2.2, 2.6, 3.2, 4.0)
LORENZ_RHO_VALUES = (10.0, 15.0, 20.0, 24.0, 28.0)
LORENZ_UNIFORM_TAU_VALUES = (0.01, 0.02, 0.05, 0.1, 0.2)
ROSSLER_COUPLING_VALUES = (0.0, 0.1, 0.25, 0.5, 0.75)
WILSON_COWAN_GAIN_VALUES = (1.0, 2.0, 3.5, 5.1, 7.5)
TRANSPORT_MAP_DEGREE = 3
TRANSPORT_MAP_JITTER = 1e-6
SURD_TM_TARGET_ANCHORS = 128
SURD_TM_CONDITIONAL_SAMPLES = 64


def _transport_map_config() -> dict[str, object]:
    return {
        "degree": TRANSPORT_MAP_DEGREE,
        "jitter": TRANSPORT_MAP_JITTER,
        "surd_target_anchors": SURD_TM_TARGET_ANCHORS,
        "surd_conditional_samples": SURD_TM_CONDITIONAL_SAMPLES,
        "applies_to": ["WMS", "SURD synergy", "MLP+PEID synergy", "Oracle PEID synergy"],
    }


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    state_names: tuple[str, ...]
    target_names: tuple[str, ...]
    equation: str
    dt: float
    warmup_steps: int
    intervention_bounds: np.ndarray
    truth_pairwise: tuple[tuple[str, str], ...]
    truth_hyperedges: tuple[tuple[str, str, str], ...]
    _vector_field: Callable[[np.ndarray], np.ndarray]
    _initial_state: Callable[[np.random.Generator], np.ndarray]
    clip_bounds: tuple[float, float] | None = None

    def vector_field(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(self.state_names):
            raise ValueError(f"states must have shape (n, {len(self.state_names)}).")
        result = np.asarray(self._vector_field(values), dtype=float)
        if result.shape != values.shape:
            raise ValueError("vector_field returned an invalid shape.")
        return result

    def simulate(self, *, seed: int, samples: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
        if samples < 2:
            raise ValueError("samples must be at least 2.")
        if noise < 0.0:
            raise ValueError("noise must be nonnegative.")
        rng = np.random.default_rng(int(seed))
        state = np.asarray(self._initial_state(rng), dtype=float).reshape(-1)
        rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        total = int(self.warmup_steps) + int(samples)
        for step in range(total):
            if step >= self.warmup_steps:
                rows.append(state.copy())
                derivative = self.vector_field(state)[0]
                targets.append(derivative + rng.normal(0.0, noise, size=len(state)))
            state = _rk4_step(self.vector_field, state, self.dt)
            if self.name.startswith("kuramoto"):
                state = (state + np.pi) % (2.0 * np.pi) - np.pi
            if self.clip_bounds is not None:
                state = np.clip(state, self.clip_bounds[0], self.clip_bounds[1])
            if noise > 0.0:
                state = state + rng.normal(0.0, noise * math.sqrt(self.dt), size=len(state))
                if self.clip_bounds is not None:
                    state = np.clip(state, self.clip_bounds[0], self.clip_bounds[1])
            if not np.isfinite(state).all():
                raise FloatingPointError(f"{self.name} simulation diverged.")
        return np.asarray(rows), np.asarray(targets)


def _rk4_step(field: Callable[[np.ndarray], np.ndarray], state: np.ndarray, dt: float) -> np.ndarray:
    y = np.asarray(state, dtype=float)
    k1 = field(y)[0]
    k2 = field(y + 0.5 * dt * k1)[0]
    k3 = field(y + 0.5 * dt * k2)[0]
    k4 = field(y + dt * k3)[0]
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def build_model_specs() -> dict[str, ModelSpec]:
    def kuramoto_field(values: np.ndarray) -> np.ndarray:
        x, y, w = values.T
        return np.column_stack(
            [
                1.0 + 0.2 * np.sin(w - x),
                1.1 + 0.2 * np.sin(w - y),
                np.full(len(values), 0.9),
            ]
        )

    def rossler_field(values: np.ndarray) -> np.ndarray:
        x0, y0, z0, x1, y1, z1 = values.T
        return np.column_stack(
            [
                -y0 - z0 + 0.5 * np.sin(x1 - x0),
                x0 + 0.165 * y0,
                2.0 + z0 * (x0 - 5.5),
                -y1 - z1 + 0.5 * np.sin(x0 - x1),
                x1 + 0.165 * y1,
                2.0 + z1 * (x1 - 5.5),
            ]
        )

    def sis_field(values: np.ndarray) -> np.ndarray:
        w, x, y = values.T
        return np.column_stack(
            [
                -0.8 * w + w * (1.0 - w),
                -1.0 * x + w * (1.0 - x),
                -1.2 * y + w * (1.0 - y),
            ]
        )

    def sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-5.1 * (values - 1.0)))

    def wilson_field(values: np.ndarray) -> np.ndarray:
        w, x, y = values.T
        drive = sigmoid(w)
        return np.column_stack([-w + drive, -x + drive, -y + drive])

    return {
        "kuramoto": ModelSpec(
            name="kuramoto",
            display_name="Kuramoto",
            state_names=("x", "y", "w"),
            target_names=("dx", "dy", "dw"),
            equation=r"\dot{x}_i=\omega_i+0.2\sum_j A_{ij}\sin(x_j-x_i)",
            dt=0.02,
            warmup_steps=400,
            intervention_bounds=np.array([[-np.pi, np.pi]] * 3),
            truth_pairwise=(("w", "dx"), ("w", "dy")),
            truth_hyperedges=(("w", "x", "dx"), ("w", "y", "dy")),
            _vector_field=kuramoto_field,
            _initial_state=lambda rng: rng.uniform(-np.pi, np.pi, size=3),
        ),
        "coupled_rossler": ModelSpec(
            name="coupled_rossler",
            display_name="Coupled Rössler",
            state_names=("x0", "y0", "z0", "x1", "y1", "z1"),
            target_names=("dx0", "dy0", "dz0", "dx1", "dy1", "dz1"),
            equation=(
                r"\dot{x}_i=-y_i-z_i+0.5\sum_jA_{ij}\sin(x_j-x_i),\;"
                r"\dot{y}_i=x_i+0.165y_i,\;\dot{z}_i=2+z_i(x_i-5.5)"
            ),
            dt=0.01,
            warmup_steps=2000,
            intervention_bounds=np.array(
                [[-7.0, 7.0], [-7.0, 7.0], [0.1, 9.0], [-7.0, 7.0], [-7.0, 7.0], [0.1, 9.0]]
            ),
            truth_pairwise=(("x0", "dy0"), ("x0", "dz0"), ("x1", "dy1"), ("x1", "dz1")),
            truth_hyperedges=(
                ("x0", "z0", "dz0"),
                ("x1", "z1", "dz1"),
                ("x0", "x1", "dx0"),
                ("x0", "x1", "dx1"),
            ),
            _vector_field=rossler_field,
            _initial_state=lambda rng: np.array(
                [rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0.1, 0.9),
                 rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0.1, 0.9)]
            ),
        ),
        "sis": ModelSpec(
            name="sis",
            display_name="SIS",
            state_names=("w", "x", "y"),
            target_names=("dw", "dx", "dy"),
            equation=r"\dot{x}_i=-\delta_i x_i+\sum_jA_{ij}x_j(1-x_i)",
            dt=0.02,
            warmup_steps=300,
            intervention_bounds=np.array([[0.02, 0.98]] * 3),
            truth_pairwise=(("w", "dx"), ("w", "dy")),
            truth_hyperedges=(("w", "x", "dx"), ("w", "y", "dy")),
            _vector_field=sis_field,
            _initial_state=lambda rng: rng.uniform(0.15, 0.75, size=3),
            clip_bounds=(0.0, 1.0),
        ),
        "wilson_cowan": ModelSpec(
            name="wilson_cowan",
            display_name="Wilson–Cowan",
            state_names=("w", "x", "y"),
            target_names=("dw", "dx", "dy"),
            equation=r"\dot{x}_i=-x_i+\sum_jA_{ij}[1+e^{-5.1(x_j-1)}]^{-1}",
            dt=0.02,
            warmup_steps=300,
            intervention_bounds=np.array([[0.0, 2.0]] * 3),
            truth_pairwise=(("w", "dx"), ("w", "dy")),
            truth_hyperedges=(),
            _vector_field=wilson_field,
            _initial_state=lambda rng: rng.uniform(0.0, 1.8, size=3),
            clip_bounds=(0.0, 5.0),
        ),
    }


def build_kuramoto_coupling_spec(coupling: float) -> ModelSpec:
    coupling = float(coupling)

    def kuramoto_field(values: np.ndarray) -> np.ndarray:
        theta1, theta2 = values.T
        return np.column_stack(
            [
                1.0
                + KURAMOTO_PHASE_POTENTIAL_STRENGTH * np.sin(theta1)
                + coupling * np.sin(theta2 - theta1),
                0.9 + KURAMOTO_PHASE_POTENTIAL_STRENGTH * np.sin(theta2),
            ]
        )

    return ModelSpec(
        name="kuramoto_phase_coupling",
        display_name=f"Kuramoto kappa={coupling:g}",
        state_names=("theta1", "theta2"),
        target_names=("dtheta1", "dtheta2"),
        equation=(
            rf"\dot\theta_1=1+0.2\sin\theta_1+{coupling:g}\sin(\theta_2-\theta_1),\;"
            r"\dot\theta_2=0.9+0.2\sin\theta_2"
        ),
        dt=0.02,
        warmup_steps=400,
        intervention_bounds=np.array([[-np.pi, np.pi]] * 2),
        truth_pairwise=(("theta1", "dtheta1"), ("theta2", "dtheta1")),
        truth_hyperedges=(("theta1", "theta2", "dtheta1"),),
        _vector_field=kuramoto_field,
        _initial_state=lambda rng: rng.uniform(-np.pi, np.pi, size=2),
    )


def build_coupled_rossler_spec(coupling: float) -> ModelSpec:
    coupling = float(coupling)

    def rossler_field(values: np.ndarray) -> np.ndarray:
        x0, y0, z0, x1, y1, z1 = values.T
        return np.column_stack(
            [
                -y0 - z0 + coupling * np.sin(x1 - x0),
                x0 + 0.165 * y0,
                2.0 + z0 * (x0 - 5.5),
                -y1 - z1 + coupling * np.sin(x0 - x1),
                x1 + 0.165 * y1,
                2.0 + z1 * (x1 - 5.5),
            ]
        )

    return ModelSpec(
        name="coupled_rossler_sweep",
        display_name=f"Coupled Rössler kappa={coupling:g}",
        state_names=("x0", "y0", "z0", "x1", "y1", "z1"),
        target_names=("dx0", "dy0", "dz0", "dx1", "dy1", "dz1"),
        equation=rf"\dot x_i=-y_i-z_i+{coupling:g}\sin(x_j-x_i)",
        dt=0.01,
        warmup_steps=2000,
        intervention_bounds=np.array(
            [[-7.0, 7.0], [-7.0, 7.0], [0.1, 9.0], [-7.0, 7.0], [-7.0, 7.0], [0.1, 9.0]]
        ),
        truth_pairwise=(("x1", "dx0"), ("x0", "dx1")),
        truth_hyperedges=(("x0", "x1", "dx0"), ("x0", "x1", "dx1")),
        _vector_field=rossler_field,
        _initial_state=lambda rng: np.array(
            [rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0.1, 0.9),
             rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0.1, 0.9)]
        ),
    )


def build_coupled_rossler_internal_product_spec(coupling: float) -> ModelSpec:
    spec = build_coupled_rossler_spec(coupling)
    return replace(
        spec,
        truth_pairwise=(("x0", "dz0"), ("z0", "dz0"), ("x1", "dz1"), ("z1", "dz1")),
        truth_hyperedges=(("x0", "z0", "dz0"), ("x1", "z1", "dz1")),
    )


def build_wilson_cowan_gain_spec(gain: float) -> ModelSpec:
    gain = float(gain)

    def sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-gain * (values - 1.0)))

    def wilson_field(values: np.ndarray) -> np.ndarray:
        w, x, y = values.T
        drive = sigmoid(w)
        return np.column_stack([-w + drive, -x + drive, -y + drive])

    return ModelSpec(
        name="wilson_cowan_gain_sweep",
        display_name=f"Wilson–Cowan gain={gain:g}",
        state_names=("w", "x", "y"),
        target_names=("dw", "dx", "dy"),
        equation=rf"\dot x_i=-x_i+[1+\exp(-{gain:g}(w-1))]^{{-1}}",
        dt=0.02,
        warmup_steps=300,
        intervention_bounds=np.array([[0.0, 2.0]] * 3),
        truth_pairwise=(("w", "dx"), ("w", "dy")),
        truth_hyperedges=(("w", "x", "dx"), ("w", "y", "dy")),
        _vector_field=wilson_field,
        _initial_state=lambda rng: rng.uniform(0.0, 1.8, size=3),
        clip_bounds=(0.0, 5.0),
    )


def build_sis_gate_spec(beta: float) -> ModelSpec:
    beta = float(beta)

    def sis_gate_field(values: np.ndarray) -> np.ndarray:
        w, x, y = values.T
        return np.column_stack(
            [
                -0.8 * w + w * (1.0 - w),
                -1.0 * x + beta * w * (1.0 - x),
                -1.2 * y + beta * w * (1.0 - y),
            ]
        )

    return ModelSpec(
        name="sis_gate_next_state",
        display_name=f"SIS beta={beta:g}",
        state_names=("w", "x", "y"),
        target_names=("w_tau", "x_tau", "y_tau"),
        equation=rf"\dot x=-x+{beta:g}w(1-x),\;\dot y=-1.2y+{beta:g}w(1-y)",
        dt=0.02,
        warmup_steps=300,
        intervention_bounds=np.array([[0.02, 0.98]] * 3),
        truth_pairwise=(("w", "x_tau"), ("w", "y_tau")),
        truth_hyperedges=(("w", "x", "x_tau"), ("w", "y", "y_tau")),
        _vector_field=sis_gate_field,
        _initial_state=lambda rng: rng.uniform(0.02, 0.98, size=3),
        clip_bounds=(0.0, 1.0),
    )


def build_lorenz_rho_spec(rho: float) -> ModelSpec:
    rho = float(rho)
    sigma = 10.0
    beta = 8.0 / 3.0

    def lorenz_field(values: np.ndarray) -> np.ndarray:
        x, y, z = values.T
        return np.column_stack(
            [
                sigma * (y - x),
                x * (rho - z) - y,
                x * y - beta * z,
            ]
        )

    return ModelSpec(
        name="lorenz_next_state",
        display_name=f"Lorenz rho={rho:g}",
        state_names=("x", "y", "z"),
        target_names=("x_tau", "y_tau", "z_tau"),
        equation=rf"\dot x=10(y-x),\;\dot y=x({rho:g}-z)-y,\;\dot z=xy-\frac{{8}}{{3}}z",
        dt=0.01,
        warmup_steps=2000,
        intervention_bounds=np.array([[-20.0, 20.0], [-30.0, 30.0], [0.0, 50.0]]),
        truth_pairwise=(("x", "y_tau"), ("z", "y_tau"), ("x", "z_tau"), ("y", "z_tau")),
        truth_hyperedges=(("x", "z", "y_tau"), ("x", "y", "z_tau")),
        _vector_field=lorenz_field,
        _initial_state=lambda rng: np.array(
            [rng.uniform(-15.0, 15.0), rng.uniform(-20.0, 20.0), rng.uniform(5.0, 35.0)]
        ),
    )


def _future_target_name(target: str) -> str:
    if target.endswith("_tau"):
        return target
    if target.startswith("d") and len(target) > 1:
        return f"{target[1:]}_tau"
    raise ValueError(f"Cannot infer a future-state target name from {target!r}.")


def build_future_state_spec(spec: ModelSpec) -> ModelSpec:
    target_names = tuple(f"{source}_tau" for source in spec.state_names)
    target_lookup = {
        old_target: _future_target_name(old_target)
        for old_target in spec.target_names
    }
    unknown = sorted(set(target_lookup.values()) - set(target_names))
    if unknown:
        raise ValueError(f"Future targets are not state targets for {spec.name}: {unknown}")
    return ModelSpec(
        name=f"{spec.name}_future_state",
        display_name=f"{spec.display_name} future state",
        state_names=spec.state_names,
        target_names=target_names,
        equation=spec.equation,
        dt=spec.dt,
        warmup_steps=spec.warmup_steps,
        intervention_bounds=spec.intervention_bounds,
        truth_pairwise=tuple((source, target_lookup[target]) for source, target in spec.truth_pairwise),
        truth_hyperedges=tuple(
            (left, right, target_lookup[target]) for left, right, target in spec.truth_hyperedges
        ),
        _vector_field=spec._vector_field,
        _initial_state=spec._initial_state,
        clip_bounds=spec.clip_bounds,
    )


def _rk4_step_batch(field: Callable[[np.ndarray], np.ndarray], states: np.ndarray, dt: float) -> np.ndarray:
    y = np.asarray(states, dtype=float)
    k1 = field(y)
    k2 = field(y + 0.5 * dt * k1)
    k3 = field(y + 0.5 * dt * k2)
    k4 = field(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _project_states(spec: ModelSpec, states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if spec.name.startswith("kuramoto"):
        values = (values + np.pi) % (2.0 * np.pi) - np.pi
    if spec.clip_bounds is not None:
        values = np.clip(values, spec.clip_bounds[0], spec.clip_bounds[1])
    return values


def simulate_natural_trajectory_pool(
    spec: ModelSpec,
    *,
    seed: int,
    trajectories: int,
    samples_per_trajectory: int,
    burnin_steps: int,
    noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    if trajectories <= 0 or samples_per_trajectory <= 0 or burnin_steps < 0 or noise < 0.0:
        raise ValueError("trajectory counts and samples must be positive; burnin and noise must be nonnegative.")
    rng = np.random.default_rng(int(seed))
    state_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for _ in range(int(trajectories)):
        state = np.asarray(spec._initial_state(rng), dtype=float).reshape(-1)
        for step in range(int(burnin_steps) + int(samples_per_trajectory)):
            if step >= burnin_steps:
                state_rows.append(state.copy())
                target_rows.append(spec.vector_field(state)[0] + rng.normal(0.0, noise, size=len(state)))
            state = _rk4_step(spec.vector_field, state, spec.dt)
            if spec.name.startswith("kuramoto"):
                state = (state + np.pi) % (2.0 * np.pi) - np.pi
            if noise > 0.0:
                state = state + rng.normal(0.0, noise * math.sqrt(spec.dt), size=len(state))
            if spec.clip_bounds is not None:
                state = np.clip(state, spec.clip_bounds[0], spec.clip_bounds[1])
            if not np.isfinite(state).all():
                raise FloatingPointError(f"{spec.name} simulation diverged.")
    return np.asarray(state_rows), np.asarray(target_rows)


def simulate_finite_time_next_states(
    spec: ModelSpec,
    initial_states: np.ndarray,
    *,
    tau: float = 1.0,
    process_noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    steps = int(round(float(tau) / float(spec.dt)))
    if steps <= 0 or not np.isclose(steps * spec.dt, tau):
        raise ValueError("tau must be a positive integer multiple of spec.dt.")
    rng = np.random.default_rng(int(seed))
    states = np.asarray(initial_states, dtype=float).copy()
    if states.ndim != 2 or states.shape[1] != len(spec.state_names):
        raise ValueError(f"initial_states must have shape (n, {len(spec.state_names)}).")
    for _ in range(steps):
        states = _rk4_step_batch(spec.vector_field, states, spec.dt)
        states = _project_states(spec, states)
        if process_noise > 0.0:
            states = states + rng.normal(0.0, float(process_noise) * math.sqrt(spec.dt), size=states.shape)
            states = _project_states(spec, states)
    return states


def simulate_sis_gate_next_states(
    spec: ModelSpec,
    initial_states: np.ndarray,
    *,
    tau: float = 1.0,
    process_noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    return simulate_finite_time_next_states(
        spec,
        initial_states,
        tau=tau,
        process_noise=process_noise,
        seed=seed,
    )


def _entropy(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    probs = probs[probs > 0.0]
    return float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0


def _discretize(values: np.ndarray, bins: int = 6) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if len(vector) == 0:
        return np.zeros(0, dtype=int)
    finite = vector[np.isfinite(vector)]
    if len(finite) == 0:
        return np.zeros(len(vector), dtype=int)
    scale = max(1.0, float(np.max(np.abs(finite))))
    if float(np.ptp(finite)) <= 1e-5 * scale:
        return np.zeros(len(vector), dtype=int)
    edges = np.unique(np.quantile(vector, np.linspace(0.0, 1.0, int(bins) + 1)))
    if len(edges) <= 2:
        return np.zeros(len(vector), dtype=int)
    return np.digitize(vector, edges[1:-1], right=False).astype(int)


def _entropy_discrete(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=int)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    _, codes = np.unique(array, axis=0, return_inverse=True)
    counts = np.bincount(codes)
    return _entropy(counts / counts.sum())


def _mi_discrete(sources: np.ndarray, target: np.ndarray) -> float:
    source = np.asarray(sources, dtype=int)
    if source.ndim == 1:
        source = source.reshape(-1, 1)
    target_codes = np.asarray(target, dtype=int).reshape(-1, 1)
    return (
        _entropy_discrete(source)
        + _entropy_discrete(target_codes)
        - _entropy_discrete(np.column_stack([source, target_codes]))
    )


def _conditional_mi_discrete(left: np.ndarray, right: np.ndarray, target: np.ndarray) -> float:
    a = np.asarray(left, dtype=int).reshape(-1, 1)
    b = np.asarray(right, dtype=int).reshape(-1, 1)
    t = np.asarray(target, dtype=int)
    if t.ndim == 1:
        t = t.reshape(-1, 1)
    value = (
        _entropy_discrete(np.column_stack([a, t]))
        + _entropy_discrete(np.column_stack([b, t]))
        - _entropy_discrete(t)
        - _entropy_discrete(np.column_stack([a, b, t]))
    )
    return float(max(0.0, value))


def _histogram_synergy(left: np.ndarray, right: np.ndarray, target: np.ndarray, bins: int) -> dict[str, float]:
    a = _discretize(left, bins)
    b = _discretize(right, bins)
    t = _discretize(target, bins)
    left_ei = _mi_discrete(a, t)
    right_ei = _mi_discrete(b, t)
    joint_ei = _mi_discrete(np.column_stack([a, b]), t)
    signed_residual = joint_ei - left_ei - right_ei
    source_tc = _mi_discrete(a, b)
    return {
        "left_ei": left_ei,
        "right_ei": right_ei,
        "joint_ei": joint_ei,
        "source_tc": source_tc,
        "signed_residual": signed_residual,
        "syn": _conditional_mi_discrete(a, b, t),
    }


def _discretize_columns(values: np.ndarray, bins: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return np.column_stack([_discretize(array[:, idx], bins) for idx in range(array.shape[1])])


def _histogram_synergy_matrix(left: np.ndarray, right: np.ndarray, target: np.ndarray, bins: int) -> dict[str, float]:
    a = _discretize(left, bins)
    b = _discretize(right, bins)
    t = _discretize_columns(target, bins)
    left_ei = _mi_discrete(a, t)
    right_ei = _mi_discrete(b, t)
    joint_ei = _mi_discrete(np.column_stack([a, b]), t)
    signed_residual = joint_ei - left_ei - right_ei
    source_tc = _mi_discrete(a, b)
    return {
        "left_ei": left_ei,
        "right_ei": right_ei,
        "joint_ei": joint_ei,
        "source_tc": source_tc,
        "signed_residual": signed_residual,
        "syn": _conditional_mi_discrete(a, b, t),
    }


def _transport_synergy(left: np.ndarray, right: np.ndarray, target: np.ndarray) -> dict[str, float]:
    from yrd import clip_nonnegative_ei, estimate_mutual_information_transport_map

    left_array = np.asarray(left, dtype=float).reshape(len(left), -1)
    right_array = np.asarray(right, dtype=float).reshape(len(right), -1)
    target_array = np.asarray(target, dtype=float).reshape(len(target), -1)
    left_ei = float(
        estimate_mutual_information_transport_map(
            left_array,
            target_array,
            jitter=TRANSPORT_MAP_JITTER,
            degree=TRANSPORT_MAP_DEGREE,
        )["mi_hat"]
    )
    right_ei = float(
        estimate_mutual_information_transport_map(
            right_array,
            target_array,
            jitter=TRANSPORT_MAP_JITTER,
            degree=TRANSPORT_MAP_DEGREE,
        )["mi_hat"]
    )
    joint_ei = float(
        estimate_mutual_information_transport_map(
            np.column_stack([left_array, right_array]),
            target_array,
            jitter=TRANSPORT_MAP_JITTER,
            degree=TRANSPORT_MAP_DEGREE,
        )["mi_hat"]
    )
    left_ei = clip_nonnegative_ei(left_ei)
    right_ei = clip_nonnegative_ei(right_ei)
    joint_ei = clip_nonnegative_ei(joint_ei)
    return {
        "left_ei": float(left_ei),
        "right_ei": float(right_ei),
        "joint_ei": float(joint_ei),
        "syn": float(joint_ei - left_ei - right_ei),
    }


def _transport_single(source: np.ndarray, target: np.ndarray) -> float:
    from yrd import clip_nonnegative_ei, estimate_mutual_information_transport_map

    estimate = estimate_mutual_information_transport_map(
        np.asarray(source, dtype=float).reshape(len(source), -1),
        np.asarray(target, dtype=float).reshape(len(target), -1),
        jitter=TRANSPORT_MAP_JITTER,
        degree=TRANSPORT_MAP_DEGREE,
    )
    return float(clip_nonnegative_ei(float(estimate["mi_hat"])))


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()[:16]


def _fitted_model_digest(model: object) -> str:
    digest = hashlib.sha256()
    net = getattr(model, "net")
    for name, value in sorted(net.state_dict().items()):
        digest.update(name.encode("utf-8"))
        tensor_values = np.asarray(value.detach().cpu().reshape(-1).tolist(), dtype=np.float32)
        digest.update(np.ascontiguousarray(tensor_values).view(np.uint8))
    for name in ("x_mean", "x_std", "y_mean", "y_std"):
        digest.update(np.ascontiguousarray(np.asarray(getattr(model, name), dtype=float)).view(np.uint8))
    return digest.hexdigest()[:16]


def _part1_fairness_audit(
    rows: Sequence[dict[str, object]],
    *,
    parameter_key: str,
    estimator: str,
    zero_values: Sequence[float] = (0.0,),
) -> dict[str, object]:
    expected_estimator = "transport_map" if estimator == "transport" else "histogram"

    def parameter_matched(field: str) -> bool:
        by_seed: dict[int, set[str]] = {}
        for row in rows:
            by_seed.setdefault(int(row["seed"]), set()).add(str(row[field]))
        return all(len(values) == 1 for values in by_seed.values())

    shared_readout = all(
        str(row["readout_state_digest"]) == str(row["peid_readout_state_digest"])
        for row in rows
    )
    shared_model = all(
        str(row["shap_mlp_model_digest"]) == str(row["peid_mlp_model_digest"])
        for row in rows
    )
    estimator_consistent = all(
        str(row[key]) == expected_estimator
        for row in rows
        for key in ("wms_estimator", "surd_estimator", "peid_estimator")
    )
    raw_fields_match = all(
        float(row[key]) == float(row[f"raw_{key}"])
        for row in rows
        for key in ("wms", "surd_synergy", "shap_interaction", "peid_synergy")
    )
    zero_rows = [
        row
        for row in rows
        if any(np.isclose(float(row[parameter_key]), float(value)) for value in zero_values)
    ]
    zero_parameter_uses_same_pipeline = bool(zero_rows) and raw_fields_match and estimator_consistent
    checks = {
        "zero_parameter_uses_same_pipeline": zero_parameter_uses_same_pipeline,
        "parameter_matched_train_states": parameter_matched("train_state_digest"),
        "parameter_matched_readout_states": parameter_matched("readout_state_digest"),
        "shared_readout_states_for_wms_surd_shap_peid": shared_readout,
        "same_fitted_mlp_for_shap_and_peid": shared_model,
        "information_estimator_consistent": estimator_consistent,
        "reported_zero_residuals_equal_raw_estimates": raw_fields_match,
    }
    return {
        "passed": bool(all(checks.values())),
        "expected_information_estimator": expected_estimator,
        "training_vs_readout": "same_registered_distribution_with_held_out_readout_samples",
        **checks,
    }


def _sample_intervention_states(spec: ModelSpec, *, samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return np.column_stack([rng.uniform(low, high, size=int(samples)) for low, high in spec.intervention_bounds])


def estimate_peid_from_samples(
    spec: ModelSpec,
    states: np.ndarray,
    targets: np.ndarray,
    *,
    estimator: str,
    bins: int = 6,
) -> dict[str, pd.DataFrame]:
    interventions = np.asarray(states, dtype=float)
    target_values = np.asarray(targets, dtype=float)
    if interventions.ndim != 2 or interventions.shape[1] != len(spec.state_names):
        raise ValueError(f"states must have shape (n, {len(spec.state_names)}).")
    if target_values.ndim != 2 or target_values.shape != (len(interventions), len(spec.target_names)):
        raise ValueError(f"targets must have shape (n, {len(spec.target_names)}).")
    pair_rows: list[dict[str, object]] = []
    pair_lookup: dict[tuple[int, int], float] = {}
    target_is_degenerate = [
        float(np.ptp(target_values[:, target_idx]))
        <= 1e-6 * max(1.0, float(np.max(np.abs(target_values[:, target_idx]))))
        for target_idx in range(len(spec.target_names))
    ]
    for source_idx, source in enumerate(spec.state_names):
        for target_idx, target in enumerate(spec.target_names):
            if target_is_degenerate[target_idx]:
                score = 0.0
            elif estimator == "transport":
                score = _transport_single(interventions[:, [source_idx]], target_values[:, [target_idx]])
            elif estimator == "histogram":
                score = _mi_discrete(
                    _discretize(interventions[:, source_idx], bins),
                    _discretize(target_values[:, target_idx], bins),
                )
            else:
                raise ValueError("estimator must be 'histogram' or 'transport'.")
            pair_lookup[(source_idx, target_idx)] = float(score)
            pair_rows.append({"source": source, "target": target, "score": float(score)})

    hyper_rows: list[dict[str, object]] = []
    for left_idx, right_idx in combinations(range(len(spec.state_names)), 2):
        for target_idx, target in enumerate(spec.target_names):
            if target_is_degenerate[target_idx]:
                values = {
                    "left_ei": 0.0,
                    "right_ei": 0.0,
                    "joint_ei": 0.0,
                    "syn": 0.0,
                    "signed_residual": 0.0,
                    "source_tc": 0.0,
                }
            elif estimator == "transport":
                values = _transport_synergy(
                    interventions[:, [left_idx]], interventions[:, [right_idx]], target_values[:, [target_idx]]
                )
            else:
                values = _histogram_synergy(
                    interventions[:, left_idx], interventions[:, right_idx], target_values[:, target_idx], bins
                )
            hyper_rows.append(
                {
                    "sources": "+".join(sorted((spec.state_names[left_idx], spec.state_names[right_idx]))),
                    "target": target,
                    "score": float(values["syn"]),
                    "raw_syn": float(values["syn"]),
                    "joint_ei": float(values["joint_ei"]),
                    "single_ei_sum": float(
                        values.get("left_ei", pair_lookup[(left_idx, target_idx)])
                        + values.get("right_ei", pair_lookup[(right_idx, target_idx)])
                    ),
                    "signed_residual": float(
                        values.get(
                            "signed_residual",
                            float(values["joint_ei"])
                            - pair_lookup[(left_idx, target_idx)]
                            - pair_lookup[(right_idx, target_idx)],
                        )
                    ),
                    "source_tc": float(values.get("source_tc", float("nan"))),
                }
            )
    return {"pairwise": pd.DataFrame(pair_rows), "hyperedges": pd.DataFrame(hyper_rows)}


def estimate_peid_for_joint_targets_from_samples(
    spec: ModelSpec,
    states: np.ndarray,
    targets: np.ndarray,
    *,
    joint_targets: Mapping[str, Sequence[str]],
    estimator: str,
    bins: int = 6,
) -> dict[str, object]:
    interventions = np.asarray(states, dtype=float)
    target_values = np.asarray(targets, dtype=float)
    if interventions.ndim != 2 or interventions.shape[1] != len(spec.state_names):
        raise ValueError(f"states must have shape (n, {len(spec.state_names)}).")
    if target_values.ndim != 2 or target_values.shape != (len(interventions), len(spec.target_names)):
        raise ValueError(f"targets must have shape (n, {len(spec.target_names)}).")

    target_index = {name: idx for idx, name in enumerate(spec.target_names)}
    grouped_targets: list[tuple[str, list[int]]] = []
    for name, members in joint_targets.items():
        indices = [target_index[member] for member in members]
        if not indices:
            raise ValueError(f"joint target {name!r} must contain at least one target.")
        grouped_targets.append((str(name), indices))

    pair_rows: list[dict[str, object]] = []
    pair_lookup: dict[tuple[int, int], float] = {}
    for source_idx, source in enumerate(spec.state_names):
        for group_idx, (target_name, target_indices) in enumerate(grouped_targets):
            target_block = target_values[:, target_indices]
            target_is_degenerate = float(np.ptp(target_block)) <= 1e-6 * max(
                1.0, float(np.max(np.abs(target_block)))
            )
            if target_is_degenerate:
                score = 0.0
            elif estimator == "transport":
                score = _transport_single(interventions[:, [source_idx]], target_block)
            elif estimator == "histogram":
                score = _mi_discrete(
                    _discretize(interventions[:, source_idx], bins),
                    _discretize_columns(target_block, bins),
                )
            else:
                raise ValueError("estimator must be 'histogram' or 'transport'.")
            pair_lookup[(source_idx, group_idx)] = float(score)
            pair_rows.append({"source": source, "target": target_name, "score": float(score)})

    hyper_rows: list[dict[str, object]] = []
    for left_idx, right_idx in combinations(range(len(spec.state_names)), 2):
        for group_idx, (target_name, target_indices) in enumerate(grouped_targets):
            target_block = target_values[:, target_indices]
            target_is_degenerate = float(np.ptp(target_block)) <= 1e-6 * max(
                1.0, float(np.max(np.abs(target_block)))
            )
            if target_is_degenerate:
                values = {
                    "left_ei": 0.0,
                    "right_ei": 0.0,
                    "joint_ei": 0.0,
                    "syn": 0.0,
                    "signed_residual": 0.0,
                    "source_tc": 0.0,
                }
            elif estimator == "transport":
                values = _transport_synergy(
                    interventions[:, [left_idx]], interventions[:, [right_idx]], target_block
                )
                values["signed_residual"] = float(values["syn"])
            else:
                values = _histogram_synergy_matrix(
                    interventions[:, left_idx], interventions[:, right_idx], target_block, bins
                )
            hyper_rows.append(
                {
                    "sources": "+".join(sorted((spec.state_names[left_idx], spec.state_names[right_idx]))),
                    "target": target_name,
                    "score": float(values["syn"]),
                    "raw_syn": float(values["syn"]),
                    "joint_ei": float(values["joint_ei"]),
                    "single_ei_sum": float(
                        values.get("left_ei", pair_lookup[(left_idx, group_idx)])
                        + values.get("right_ei", pair_lookup[(right_idx, group_idx)])
                    ),
                    "signed_residual": float(
                        values.get(
                            "signed_residual",
                            float(values["joint_ei"])
                            - pair_lookup[(left_idx, group_idx)]
                            - pair_lookup[(right_idx, group_idx)],
                        )
                    ),
                    "source_tc": float(values.get("source_tc", float("nan"))),
                }
            )
    return {
        "target_names": [name for name, _ in grouped_targets],
        "pairwise": pd.DataFrame(pair_rows),
        "hyperedges": pd.DataFrame(hyper_rows),
    }


def estimate_peid(
    spec: ModelSpec,
    predictor: Callable[[np.ndarray], np.ndarray],
    *,
    samples: int,
    seed: int,
    estimator: str,
    bins: int = 6,
) -> dict[str, pd.DataFrame]:
    interventions = _sample_intervention_states(spec, samples=samples, seed=seed)
    targets = np.asarray(predictor(interventions), dtype=float)
    return estimate_peid_from_samples(spec, interventions, targets, estimator=estimator, bins=bins)


def estimate_oracle_peid(
    spec: ModelSpec, *, samples: int = 2048, seed: int = 0, estimator: str = "transport"
) -> dict[str, pd.DataFrame]:
    return estimate_peid(spec, spec.vector_field, samples=samples, seed=seed, estimator=estimator)


@dataclass
class FittedMLP:
    net: object
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    train_mse: float
    baseline_mse: float

    def predict(self, states: np.ndarray) -> np.ndarray:
        import torch

        values = np.asarray(states, dtype=np.float32)
        scaled = (values - self.x_mean) / self.x_std
        self.net.eval()
        with torch.no_grad():
            pred = np.asarray(
                self.net(torch.tensor(scaled.tolist(), dtype=torch.float32)).cpu().tolist(),
                dtype=float,
            )
        return pred * self.y_std + self.y_mean


def fit_mlp(states: np.ndarray, targets: np.ndarray, *, seed: int, epochs: int) -> FittedMLP:
    import torch

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    x = np.asarray(states, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    split = max(32, int(0.8 * len(x)))
    x_mean = x[:split].mean(axis=0, keepdims=True)
    x_std = np.maximum(x[:split].std(axis=0, keepdims=True), 1e-6)
    y_mean = y[:split].mean(axis=0, keepdims=True)
    y_std = np.maximum(y[:split].std(axis=0, keepdims=True), 1e-6)
    xn = (x - x_mean) / x_std
    yn = (y - y_mean) / y_std
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*", category=UserWarning)
        net = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 48),
            torch.nn.SiLU(),
            torch.nn.Linear(48, 48),
            torch.nn.SiLU(),
            torch.nn.Linear(48, y.shape[1]),
        )
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
    xt = torch.tensor(xn[:split], dtype=torch.float32)
    yt = torch.tensor(yn[:split], dtype=torch.float32)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((net(xt) - yt) ** 2)
        loss.backward()
        optimizer.step()
    fitted = FittedMLP(net, x_mean, x_std, y_mean, y_std, 0.0, 0.0)
    prediction = fitted.predict(x[split:])
    fitted.train_mse = float(np.mean((prediction - y[split:]) ** 2))
    fitted.baseline_mse = float(np.mean((y[split:] - y[:split].mean(axis=0, keepdims=True)) ** 2))
    return fitted


def estimate_granger_ablation(
    model: FittedMLP, states: np.ndarray, targets: np.ndarray, spec: ModelSpec
) -> pd.DataFrame:
    base = model.predict(states)
    rows: list[dict[str, object]] = []
    for source_idx, source in enumerate(spec.state_names):
        ablated = np.asarray(states, dtype=float).copy()
        ablated[:, source_idx] = np.mean(ablated[:, source_idx])
        prediction = model.predict(ablated)
        for target_idx, target in enumerate(spec.target_names):
            base_mse = np.mean((targets[:, target_idx] - base[:, target_idx]) ** 2)
            ablated_mse = np.mean((targets[:, target_idx] - prediction[:, target_idx]) ** 2)
            rows.append({"source": source, "target": target, "score": float(max(0.0, ablated_mse - base_mse))})
    return pd.DataFrame(rows)


def estimate_neural_granger(
    states: np.ndarray, targets: np.ndarray, spec: ModelSpec, *, seed: int, epochs: int
) -> pd.DataFrame:
    import torch

    x = np.asarray(states, dtype=np.float32)
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = np.maximum(x.std(axis=0, keepdims=True), 1e-6)
    xn = (x - x_mean) / x_std
    rows: list[dict[str, object]] = []
    for target_idx, target in enumerate(spec.target_names):
        torch.manual_seed(int(seed) + target_idx)
        y = np.asarray(targets[:, target_idx], dtype=np.float32)
        yn = (y - y.mean()) / max(float(y.std()), 1e-6)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*", category=UserWarning)
            net = torch.nn.Sequential(
                torch.nn.Linear(x.shape[1], 24), torch.nn.Tanh(), torch.nn.Linear(24, 1)
            )
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        xt = torch.tensor(xn, dtype=torch.float32)
        yt = torch.tensor(yn[:, None], dtype=torch.float32)
        for _ in range(int(epochs)):
            optimizer.zero_grad(set_to_none=True)
            weights = net[0].weight
            group_penalty = torch.linalg.vector_norm(weights, dim=0).sum()
            loss = torch.mean((net(xt) - yt) ** 2) + 0.01 * group_penalty
            loss.backward()
            optimizer.step()
        norms = np.linalg.norm(np.asarray(net[0].weight.detach().tolist(), dtype=float), axis=0)
        for source, score in zip(spec.state_names, norms):
            rows.append({"source": source, "target": target, "score": float(score)})
    return pd.DataFrame(rows)


def estimate_shap_readout(
    model: FittedMLP, states: np.ndarray, spec: ModelSpec, *, samples: int, seed: int
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    count = min(int(samples), len(states))
    foreground = np.asarray(states, dtype=float)[rng.choice(len(states), size=count, replace=False)]
    background = np.mean(states, axis=0)
    baseline_rows = np.repeat(background[None, :], count, axis=0)
    baseline = model.predict(baseline_rows)
    feature_rows: list[dict[str, object]] = []
    for source_idx, source in enumerate(spec.state_names):
        modified = baseline_rows.copy()
        modified[:, source_idx] = foreground[:, source_idx]
        contribution = model.predict(modified) - baseline
        for target_idx, target in enumerate(spec.target_names):
            feature_rows.append(
                {"source": source, "target": target, "score": float(np.mean(np.abs(contribution[:, target_idx])))}
            )
    interaction_rows: list[dict[str, object]] = []
    for left_idx, right_idx in combinations(range(len(spec.state_names)), 2):
        both = baseline_rows.copy()
        left = baseline_rows.copy()
        right = baseline_rows.copy()
        both[:, [left_idx, right_idx]] = foreground[:, [left_idx, right_idx]]
        left[:, left_idx] = foreground[:, left_idx]
        right[:, right_idx] = foreground[:, right_idx]
        interaction = model.predict(both) - model.predict(left) - model.predict(right) + baseline
        for target_idx, target in enumerate(spec.target_names):
            interaction_rows.append(
                {
                    "sources": "+".join(sorted((spec.state_names[left_idx], spec.state_names[right_idx]))),
                    "target": target,
                    "score": float(np.mean(np.abs(interaction[:, target_idx]))),
                }
            )
    return {"pairwise": pd.DataFrame(feature_rows), "interactions": pd.DataFrame(interaction_rows)}


def observational_wms_surd(
    states: np.ndarray,
    targets: np.ndarray,
    spec: ModelSpec,
    *,
    bins: int = 6,
    estimator: str = "transport",
    seed: int = 0,
) -> pd.DataFrame:
    source_index = {name: idx for idx, name in enumerate(spec.state_names)}
    target_index = {name: idx for idx, name in enumerate(spec.target_names)}
    rows: list[dict[str, object]] = []
    relations = list(spec.truth_hyperedges)
    if spec.name == "wilson_cowan":
        relations = [("w", "x", "dx"), ("w", "y", "dy")]
    for left, right, target in relations:
        if estimator == "transport":
            left_values = states[:, [source_index[left]]]
            right_values = states[:, [source_index[right]]]
            target_values = targets[:, [target_index[target]]]
            values = _transport_synergy(left_values, right_values, target_values)
            from scripts.reproduce_surd_synergistic_collider import decompose_surd_2source_transport_map

            surd = decompose_surd_2source_transport_map(
                left_values,
                right_values,
                target_values,
                degree=TRANSPORT_MAP_DEGREE,
                target_anchors=SURD_TM_TARGET_ANCHORS,
                conditional_samples=SURD_TM_CONDITIONAL_SAMPLES,
                seed=int(seed),
            )
            wms = float(values["syn"])
            row_extra = {
                "tm_degree": TRANSPORT_MAP_DEGREE,
                "tm_jitter": TRANSPORT_MAP_JITTER,
                "surd_target_anchors": min(SURD_TM_TARGET_ANCHORS, len(target_values)),
                "surd_conditional_samples": SURD_TM_CONDITIONAL_SAMPLES,
            }
        elif estimator == "histogram":
            a = _discretize(states[:, source_index[left]], bins)
            b = _discretize(states[:, source_index[right]], bins)
            t = _discretize(targets[:, target_index[target]], bins)
            values = _histogram_synergy(a, b, t, bins)
            surd = _specific_information_surd(a, b, t)
            wms = float(values["syn"])
            row_extra = {"bins": int(bins)}
        else:
            raise ValueError("estimator must be 'transport' or 'histogram'.")
        rows.append(
            {
                "sources": "+".join(sorted((left, right))),
                "target": target,
                "wms": wms,
                "left_mi": float(values["left_ei"]),
                "right_mi": float(values["right_ei"]),
                "joint_mi": float(values["joint_ei"]),
                **surd,
                **row_extra,
            }
        )
    return pd.DataFrame(rows)


def _specific_information_surd(a: np.ndarray, b: np.ndarray, target: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    t = np.asarray(target, dtype=int)
    n = len(t)
    totals = {"redundancy": 0.0, "unique_left": 0.0, "unique_right": 0.0, "synergy": 0.0}
    for target_value in np.unique(t):
        mask = t == target_value
        pt = float(mask.mean())
        if pt <= 0.0:
            continue

        def specific(source: np.ndarray) -> float:
            value = 0.0
            for source_value in np.unique(source[mask]):
                joint_mask = mask & (source == source_value)
                p_source_given_t = float(joint_mask.sum() / mask.sum())
                p_t_given_source = float(joint_mask.sum() / max((source == source_value).sum(), 1))
                if p_source_given_t > 0.0 and p_t_given_source > 0.0:
                    value += p_source_given_t * math.log2(p_t_given_source / pt)
            return value

        ia = specific(a)
        ib = specific(b)
        joint_codes = a * (int(b.max()) + 1) + b
        iab = specific(joint_codes)
        redundancy = min(ia, ib)
        totals["redundancy"] += pt * redundancy
        totals["unique_left"] += pt * (ia - redundancy)
        totals["unique_right"] += pt * (ib - redundancy)
        totals["synergy"] += pt * (iab - max(ia, ib))
    return {key: float(value) for key, value in totals.items()}


def _frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def _mean_records(
    seed_payloads: list[dict[str, object]], path: tuple[str, ...], keys: Sequence[str]
) -> list[dict[str, object]]:
    frames: list[pd.DataFrame] = []
    for payload in seed_payloads:
        value: object = payload
        for part in path:
            value = value[part]  # type: ignore[index]
        frames.append(pd.DataFrame(value))
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return []
    numeric = [column for column in combined.select_dtypes(include=[np.number]).columns if column not in keys]
    return _frame_records(combined.groupby(list(keys), as_index=False)[numeric].mean())


def _aggregate_seed_payloads(seed_payloads: list[dict[str, object]]) -> dict[str, object]:
    metrics = pd.DataFrame([payload["mlp_metrics"] for payload in seed_payloads])
    return {
        "seed": "mean",
        "mlp_metrics": {key: float(value) for key, value in metrics.mean().items()},
        "oracle_peid": {
            "pairwise": _mean_records(seed_payloads, ("oracle_peid", "pairwise"), ("source", "target")),
            "hyperedges": _mean_records(seed_payloads, ("oracle_peid", "hyperedges"), ("sources", "target")),
        },
        "mlp_peid": {
            "pairwise": _mean_records(seed_payloads, ("mlp_peid", "pairwise"), ("source", "target")),
            "hyperedges": _mean_records(seed_payloads, ("mlp_peid", "hyperedges"), ("sources", "target")),
        },
        "granger_ablation": _mean_records(seed_payloads, ("granger_ablation",), ("source", "target")),
        "neural_granger": _mean_records(seed_payloads, ("neural_granger",), ("source", "target")),
        "shap": {
            "pairwise": _mean_records(seed_payloads, ("shap", "pairwise"), ("source", "target")),
            "interactions": _mean_records(seed_payloads, ("shap", "interactions"), ("sources", "target")),
        },
        "observational": _mean_records(seed_payloads, ("observational",), ("sources", "target")),
        "all_seeds": seed_payloads,
    }


def _truth_synergy_scores(frame: pd.DataFrame, spec: ModelSpec) -> list[float]:
    scores: list[float] = []
    for left, right, target in spec.truth_hyperedges:
        key = "+".join(sorted((left, right)))
        selected = frame[(frame["sources"] == key) & (frame["target"] == target)]
        if not selected.empty:
            scores.append(float(selected.iloc[0]["score"]))
    return scores


def _plot_model_result(spec: ModelSpec, payload: dict[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt

    oracle = pd.DataFrame(payload["oracle_peid"]["hyperedges"])
    learned = pd.DataFrame(payload["mlp_peid"]["hyperedges"])
    relations = list(spec.truth_hyperedges)
    if not relations:
        relations = [("w", "x", "dx"), ("w", "y", "dy")]
    labels = [f"{'+'.join(sorted((left, right)))}→{target}" for left, right, target in relations]
    positions = np.arange(len(labels))
    oracle_lookup = oracle.assign(label=oracle["sources"] + "→" + oracle["target"]).set_index("label")["score"]
    learned_lookup = learned.assign(label=learned["sources"] + "→" + learned["target"]).set_index("label")["score"]
    oracle_values = np.asarray([float(oracle_lookup.get(label, 0.0)) for label in labels])
    learned_values = np.asarray([float(learned_lookup.get(label, 0.0)) for label in labels])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, max(3.5, 0.55 * len(labels) + 2.0)), constrained_layout=True)
    axes[0].barh(positions - 0.16, oracle_values, height=0.28, label="Oracle + PEID", color="#4C78A8")
    axes[0].barh(positions + 0.16, learned_values, height=0.28, label="MLP + PEID", color="#E45756")
    axes[0].set_yticks(positions, labels, fontsize=7)
    axes[0].set_xlabel("Synergy (bits)")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
    pair_labels = [f"{source}→{target}" for source, target in spec.truth_pairwise]
    methods = {
        "Granger": pd.DataFrame(payload["granger_ablation"]),
        "Neural Granger": pd.DataFrame(payload["neural_granger"]),
        "SHAP": pd.DataFrame(payload["shap"]["pairwise"]),
        "MLP PEID": pd.DataFrame(payload["mlp_peid"]["pairwise"]),
    }
    matrix = np.zeros((len(methods), len(pair_labels)), dtype=float)
    raw = np.zeros_like(matrix)
    for method_idx, frame in enumerate(methods.values()):
        lookup = frame.set_index(["source", "target"])["score"]
        raw[method_idx] = [float(lookup.get(edge, 0.0)) for edge in spec.truth_pairwise]
        scale = float(np.max(np.abs(raw[method_idx])))
        matrix[method_idx] = raw[method_idx] / scale if scale > 0.0 else raw[method_idx]
    axes[1].imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    axes[1].set_xticks(np.arange(len(pair_labels)), pair_labels, rotation=25, ha="right", fontsize=7)
    axes[1].set_yticks(np.arange(len(methods)), list(methods), fontsize=7)
    for row in range(raw.shape[0]):
        for col in range(raw.shape[1]):
            axes[1].text(col, row, f"{raw[row, col]:.3g}", ha="center", va="center", fontsize=6.5)
    axes[1].set_xlabel("Within-method normalized color; cells show raw scores")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(payloads: dict[str, dict[str, object]], specs: dict[str, ModelSpec], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows: list[list[float]] = []
    labels: list[str] = []
    for name, spec in specs.items():
        payload = payloads[name]
        oracle = pd.DataFrame(payload["oracle_peid"]["hyperedges"])
        learned = pd.DataFrame(payload["mlp_peid"]["hyperedges"])
        oracle_truth = _truth_synergy_scores(oracle, spec)
        learned_truth = _truth_synergy_scores(learned, spec)
        rows.append(
            [
                float(np.mean(oracle_truth)) if oracle_truth else 0.0,
                float(np.mean(learned_truth)) if learned_truth else 0.0,
                float(oracle["score"].abs().max()) if spec.name == "wilson_cowan" else 0.0,
                float(payload["mlp_metrics"]["skill_ratio"]),
            ]
        )
        labels.append(spec.display_name)
    matrix = np.asarray(rows, dtype=float)
    normalized = matrix.copy()
    for col in range(matrix.shape[1] - 1):
        scale = np.max(np.abs(matrix[:, col]))
        normalized[:, col] = matrix[:, col] / scale if scale > 0 else matrix[:, col]
    normalized[:, -1] = 1.0 - np.clip(matrix[:, -1], 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
    image = ax.imshow(normalized, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(
        np.arange(4),
        ["Oracle truth synergy", "MLP truth synergy", "WC additive-control max |syn|", "MLP MSE / baseline"],
        rotation=18,
        ha="right",
        fontsize=7,
    )
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03, label="Column-normalized intensity")
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _legacy_appendix(report_path: Path) -> str:
    if not report_path.exists():
        return "原 sine 基准未在此输出目录中提供。"
    current = report_path.read_text(encoding="utf-8")
    if LEGACY_MARKER in current:
        legacy = current.split(LEGACY_MARKER, maxsplit=1)[1].lstrip()
        intro = "以下内容保留原人工系统，用于校准两个外部源到第三目标的纯协同语义。它不再作为主实验。"
        while legacy.startswith(intro):
            legacy = legacy[len(intro) :].lstrip()
        return legacy
    return current


def _write_report(
    report_path: Path,
    payloads: dict[str, dict[str, object]],
    specs: dict[str, ModelSpec],
    summary_figure_path: Path,
    model_figure_paths: dict[str, Path],
) -> None:
    legacy = _legacy_appendix(report_path)

    def relative(path: Path) -> str:
        return os.path.relpath(path, report_path.parent).replace(os.sep, "/")

    sections: list[str] = []
    for name, spec in specs.items():
        payload = payloads[name]
        oracle = pd.DataFrame(payload["oracle_peid"]["hyperedges"])
        learned = pd.DataFrame(payload["mlp_peid"]["hyperedges"])
        truth_oracle = _truth_synergy_scores(oracle, spec)
        truth_learned = _truth_synergy_scores(learned, spec)
        truth_text = ", ".join(
            f"`{{{left},{right}}}->{target}`" for left, right, target in spec.truth_hyperedges
        ) or "无显式乘积或相位差交互（结构可加对照）"
        hyper_relations = list(spec.truth_hyperedges)
        if not hyper_relations:
            hyper_relations = [("w", "x", "dx"), ("w", "y", "dy")]
        oracle_lookup = oracle.set_index(["sources", "target"])["score"]
        learned_lookup = learned.set_index(["sources", "target"])["score"]
        shap_lookup = pd.DataFrame(payload["shap"]["interactions"]).set_index(["sources", "target"])["score"]
        observational = pd.DataFrame(payload["observational"])
        observational_lookup = (
            observational.set_index(["sources", "target"])
            if not observational.empty
            else pd.DataFrame(columns=["wms", "synergy"])
        )
        hyper_lines = [
            "| source set -> target | Oracle PEID | MLP PEID | SHAP interaction | observational WMS | SURD synergy |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for left, right, target in hyper_relations:
            key = "+".join(sorted((left, right)))
            obs_key = (key, target)
            wms = float(observational_lookup.loc[obs_key, "wms"]) if obs_key in observational_lookup.index else float("nan")
            surd = float(observational_lookup.loc[obs_key, "synergy"]) if obs_key in observational_lookup.index else float("nan")
            hyper_lines.append(
                f"| `{{{left},{right}}}->{target}` | {float(oracle_lookup.get((key, target), 0.0)):.4f} | "
                f"{float(learned_lookup.get((key, target), 0.0)):.4f} | "
                f"{float(shap_lookup.get((key, target), 0.0)):.4f} | "
                f"{wms:.4f} | {surd:.4f} |"
            )
        pair_frames = {
            "Granger": pd.DataFrame(payload["granger_ablation"]).set_index(["source", "target"])["score"],
            "Neural Granger": pd.DataFrame(payload["neural_granger"]).set_index(["source", "target"])["score"],
            "SHAP": pd.DataFrame(payload["shap"]["pairwise"]).set_index(["source", "target"])["score"],
            "MLP PEID": pd.DataFrame(payload["mlp_peid"]["pairwise"]).set_index(["source", "target"])["score"],
        }
        pair_lines = [
            "| pairwise truth | Granger | Neural Granger | SHAP | MLP PEID |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for source, target in spec.truth_pairwise:
            pair_lines.append(
                f"| `{source}->{target}` | {float(pair_frames['Granger'].get((source, target), 0.0)):.4f} | "
                f"{float(pair_frames['Neural Granger'].get((source, target), 0.0)):.4f} | "
                f"{float(pair_frames['SHAP'].get((source, target), 0.0)):.4f} | "
                f"{float(pair_frames['MLP PEID'].get((source, target), 0.0)):.4f} |"
            )
        interpretation = (
            "该模型没有显式二源乘积或相位差项，但 PEID 仍可能为 `状态 + 外部驱动` 给出正联合信息残差；"
            "因此它是结构交互负对照，不是 PEID 数值零对照。"
            if spec.name == "wilson_cowan"
            else "真值协同采用包含目标当前状态的状态依赖门控口径。"
        )
        hyper_table = "\n".join(hyper_lines)
        pair_table = "\n".join(pair_lines)
        sections.append(
            f"""## {spec.display_name}

论文方程：

$$
{spec.equation}
$$

- 结构真值：{truth_text}。
- Oracle 真值协同均值：{float(np.mean(truth_oracle)) if truth_oracle else 0.0:.4f} bits。
- MLP 真值协同均值：{float(np.mean(truth_learned)) if truth_learned else 0.0:.4f} bits。
- MLP 测试 MSE / 常数基线：{float(payload['mlp_metrics']['skill_ratio']):.4f}。
- 解释：{interpretation}

{hyper_table}

{pair_table}

![{spec.display_name} 方法读出]({relative(model_figure_paths[name])})
"""
        )
    model_sections = "\n".join(sections)
    text = rf"""# 经典网络动力学中的共同驱动与状态依赖协同

完整实验设计、数值协议和结果讨论见 [经典网络动力学 benchmark 报告](classic_network_dynamics_benchmark.md)。

主比较使用论文 *Discovering network dynamics with neural symbolic regression* 的原始动力学方程。网络只缩减为可解释 motif；预测目标统一为当前状态到向量场 $\dot{{\mathbf{{x}}}}$，避免小步长下一状态中的恒等映射掩盖耦合机制。

协同源集合允许包含目标变量的当前状态。它表示状态依赖门控，例如 SIS 中感染源 $w$ 的作用受到目标当前易感比例 $1-x$ 调制，并不等价于两个外部源共同指向第三变量的 collider。

主表比较 Granger ablation、Neural Granger、SHAP、观测 WMS/SURD、MLP+PEID 与 Oracle+PEID。PCMCI 保留在附录，因为它检验下一状态时间序列中的滞后关系，而不是当前状态到向量场的监督映射。

![跨模型汇总]({relative(summary_figure_path)})

图中前三列在各列内部归一化，格内数字是原始值。最后一列越低表示 MLP 相对常数基线越好。Wilson–Cowan 是结构可加对照，不应被误写成 PEID 数值零对照。

{model_sections}

## 方法口径

- Granger ablation：在固定 MLP 上把单个当前状态替换为均值，读取目标导数预测误差增量。
- Neural Granger：逐目标 cMLP 第一层 source-group norm，仍是 pairwise 预测结构。
- SHAP：独立背景替换下的单源贡献和二源 inclusion–exclusion interaction。
- Observational WMS/SURD：直接基于自然轨迹的状态与导数经验分布。
- PEID：对源状态做独立最大熵干预，再比较联合 EI 与单源 EI；正式 sweep 和 smoke sweep 均使用 transport-map 估计。

{LEGACY_MARKER}

以下内容保留原人工系统，用于校准两个外部源到第三目标的纯协同语义。它不再作为主实验。

{legacy.rstrip()}
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run_benchmark(
    *,
    mode: str = "full",
    result_dir: Path = DEFAULT_RESULT_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    seeds: Sequence[int] = (0, 1, 2),
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    result_dir = Path(result_dir)
    figure_dir = Path(figure_dir)
    report_path = Path(report_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    specs = build_model_specs()
    trajectory_samples = 260 if mode == "smoke" else 1600
    intervention_samples = 700 if mode == "smoke" else 1800
    epochs = 35 if mode == "smoke" else 180
    neural_epochs = 20 if mode == "smoke" else 100
    estimator = "transport"
    payloads: dict[str, dict[str, object]] = {}
    model_figure_paths: dict[str, Path] = {}

    for model_name, spec in specs.items():
        seed_payloads: list[dict[str, object]] = []
        for seed in seeds:
            states, targets = spec.simulate(seed=int(seed), samples=trajectory_samples, noise=0.01)
            fitted = fit_mlp(states, targets, seed=int(seed) + 101, epochs=epochs)
            oracle = estimate_oracle_peid(
                spec, samples=intervention_samples, seed=int(seed) + 1001, estimator=estimator
            )
            learned = estimate_peid(
                spec,
                fitted.predict,
                samples=intervention_samples,
                seed=int(seed) + 1001,
                estimator=estimator,
            )
            shap = estimate_shap_readout(fitted, states, spec, samples=28 if mode == "smoke" else 72, seed=seed)
            seed_payloads.append(
                {
                    "seed": int(seed),
                    "mlp_metrics": {
                        "test_mse": fitted.train_mse,
                        "baseline_mse": fitted.baseline_mse,
                        "skill_ratio": fitted.train_mse / max(fitted.baseline_mse, 1e-12),
                    },
                    "oracle_peid": {key: _frame_records(value) for key, value in oracle.items()},
                    "mlp_peid": {key: _frame_records(value) for key, value in learned.items()},
                    "granger_ablation": _frame_records(estimate_granger_ablation(fitted, states, targets, spec)),
                    "neural_granger": _frame_records(
                        estimate_neural_granger(states, targets, spec, seed=seed + 4001, epochs=neural_epochs)
                    ),
                    "shap": {key: _frame_records(value) for key, value in shap.items()},
                    "observational": _frame_records(
                        observational_wms_surd(states, targets, spec, estimator=estimator, seed=seed + 2001)
                    ),
                }
            )
        payload = _aggregate_seed_payloads(seed_payloads)
        payload["estimator"] = estimator
        payload["transport_map"] = _transport_map_config() if estimator == "transport" else None
        payloads[model_name] = payload
        model_path = figure_dir / f"{model_name}_readout.png"
        _plot_model_result(spec, payload, model_path)
        model_figure_paths[model_name] = model_path
        (result_dir / f"{model_name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    summary_figure_path = figure_dir / "classic_dynamics_summary.png"
    _plot_summary(payloads, specs, summary_figure_path)
    _write_report(report_path, payloads, specs, summary_figure_path, model_figure_paths)
    summary = {
        "mode": mode,
        "models": list(specs),
        "seeds": [int(seed) for seed in seeds],
        "estimator": estimator,
        "transport_map": _transport_map_config() if estimator == "transport" else None,
        "summary_figure_path": str(summary_figure_path),
        "model_figure_paths": {key: str(value) for key, value in model_figure_paths.items()},
        "report_path": str(report_path),
    }
    summary_path = result_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**summary, "summary_path": str(summary_path)}


def _relation_score(frame: pd.DataFrame, sources: str, target: str, column: str = "score") -> float:
    selected = frame[(frame["sources"] == sources) & (frame["target"] == target)]
    if selected.empty:
        return 0.0
    return float(selected.iloc[0][column])


def _mean_truth_hyperedge_score(
    frame: pd.DataFrame,
    relations: Sequence[tuple[str, str, str]],
    *,
    column: str = "score",
) -> float:
    values = [
        _relation_score(frame, "+".join(sorted((left, right))), target, column=column)
        for left, right, target in relations
    ]
    return float(np.mean(values)) if values else 0.0


def _mean_truth_hyperedge_components(
    frame: pd.DataFrame,
    relations: Sequence[tuple[str, str, str]],
) -> dict[str, float]:
    selected: list[pd.Series] = []
    for left, right, target in relations:
        source_key = "+".join(sorted((left, right)))
        matches = frame[(frame["sources"] == source_key) & (frame["target"] == target)]
        if not matches.empty:
            selected.append(matches.iloc[0])
    if not selected:
        return {"syn": 0.0, "joint_ei": 0.0, "single_ei_sum": 0.0}
    return {
        key: float(np.mean([float(row[key if key != "syn" else "score"]) for row in selected]))
        for key in ("syn", "joint_ei", "single_ei_sum")
    }


def _zero_control_synergy_readouts(
    *,
    inactive: bool,
    wms: float,
    surd_synergy: float,
    shap_interaction: float,
    peid_synergy: float,
) -> dict[str, float]:
    raw = {
        "wms": float(wms),
        "surd_synergy": float(surd_synergy),
        "shap_interaction": float(shap_interaction),
        "peid_synergy": float(peid_synergy),
    }
    return {
        **raw,
        **{f"raw_{key}": value for key, value in raw.items()},
    }


def _method_data_contract() -> dict[str, object]:
    return {
        "model_training": "one_shared_natural_training_pool",
        "model_based_methods": ["MLP+SHAP interaction", "MLP+PEID synergy"],
        "model_reuse": "same_fitted_mlp_object",
        "observational_readout": "one_shared_held_out_natural_pool",
        "observational_methods": ["WMS", "SURD synergy", "MLP+SHAP interaction"],
        "peid_interventions": "method_internal_sampling_only",
    }


def _aggregate_sweep_rows(rows: list[dict[str, object]], *, parameter_key: str) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for value, group in frame.groupby(parameter_key, sort=True):
        row: dict[str, object] = {parameter_key: float(value), "n_seeds": int(group["seed"].nunique())}
        methods = ["wms", "surd_synergy", "shap_interaction", "peid_synergy"]
        if "oracle_peid_synergy" in group:
            methods.append("oracle_peid_synergy")
        for method in methods:
            values = group[method].astype(float)
            row[f"{method}_mean"] = float(values.mean())
            row[f"{method}_std"] = float(values.std(ddof=0))
        summaries.append(row)
    return summaries


def _aggregate_kuramoto_peid_detail_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    metrics = (
        "mlp_syn", "mlp_joint_ei", "mlp_single_ei_sum",
        "oracle_syn", "oracle_joint_ei", "oracle_single_ei_sum",
        "signal_rms", "mlp_test_mse",
    )
    for coupling, group in frame.groupby("coupling", sort=True):
        row: dict[str, object] = {"coupling": float(coupling), "n_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        summaries.append(row)
    return summaries


def _aggregate_kuramoto_joint_target_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    summary: dict[str, object] = {"n_seeds": int(frame["seed"].nunique())}
    for metric in (
        "mlp_syn",
        "mlp_joint_ei",
        "mlp_single_ei_sum",
        "oracle_syn",
        "oracle_joint_ei",
        "oracle_single_ei_sum",
        "mlp_test_mse",
    ):
        values = frame[metric].astype(float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    return summary


def _aggregate_kuramoto_joint_target_sweep_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    metrics = (
        "mlp_syn",
        "mlp_joint_ei",
        "mlp_single_ei_sum",
        "oracle_syn",
        "oracle_joint_ei",
        "oracle_single_ei_sum",
        "mlp_test_mse",
    )
    for coupling, group in frame.groupby("coupling", sort=True):
        row: dict[str, object] = {"coupling": float(coupling), "n_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        summaries.append(row)
    return summaries


def _nonmonotonic_syn_diagnostic(summary: list[dict[str, object]]) -> dict[str, object]:
    couplings = np.asarray([float(row["coupling"]) for row in summary], dtype=float)
    diagnostic: dict[str, object] = {}
    for key in ("mlp_syn", "oracle_syn"):
        values = np.asarray([float(row[f"{key}_mean"]) for row in summary], dtype=float)
        peak_idx = int(np.argmax(values))
        diagnostic[f"{key}_peak_coupling"] = float(couplings[peak_idx])
        diagnostic[f"{key}_peak_value"] = float(values[peak_idx])
        diagnostic[f"{key}_has_internal_peak"] = bool(0 < peak_idx < len(values) - 1)
        diagnostic[f"{key}_start_value"] = float(values[0])
        diagnostic[f"{key}_end_value"] = float(values[-1])
    return diagnostic


def _aggregate_sis_gate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return _aggregate_sweep_rows(rows, parameter_key="beta")


def _aggregate_kuramoto_coupling_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    metrics = (
        "wms",
        "surd_synergy",
        "shap_interaction",
        "peid_synergy",
        "oracle_peid_synergy",
        "phase_locking_value",
        "phase_order_parameter",
        "wms_left_mi",
        "wms_right_mi",
        "wms_joint_mi",
        "mlp_test_mse",
        "mlp_baseline_mse",
    )
    for coupling, group in frame.groupby("coupling", sort=True):
        row: dict[str, object] = {
            "coupling": float(coupling),
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_sem"] = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        summaries.append(row)
    return summaries


def _phase_locking_value(states: np.ndarray) -> float:
    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Kuramoto states must have shape (n, 2).")
    phase_difference = values[:, 1] - values[:, 0]
    return float(np.abs(np.mean(np.exp(1j * phase_difference))))


def _kuramoto_order_parameter(states: np.ndarray) -> float:
    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Kuramoto states must have shape (n, 2).")
    phasors = np.exp(1j * values)
    return float(np.mean(np.abs(np.mean(phasors, axis=1))))


def _kuramoto_order_excess(states: np.ndarray) -> float:
    baseline = 2.0 / np.pi
    raw_order = _kuramoto_order_parameter(states)
    return float(np.clip((raw_order - baseline) / (1.0 - baseline), 0.0, 1.0))


def _kuramoto_phase_response_targets(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Kuramoto states must have shape (n, 2).")
    delta = values[:, 1] - values[:, 0]
    raw_order = np.abs(np.mean(np.exp(1j * values), axis=1))
    baseline = 2.0 / np.pi
    order_excess = np.clip((raw_order - baseline) / (1.0 - baseline), 0.0, 1.0)
    return np.column_stack([np.cos(delta), np.sin(delta), order_excess])


def _aggregate_kuramoto_phase_response_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    metrics = (
        "natural_plv",
        "natural_order",
        "natural_order_raw",
        "mlp_syn",
        "mlp_joint_ei",
        "mlp_single_ei_sum",
        "oracle_syn",
        "oracle_joint_ei",
        "oracle_single_ei_sum",
        "mlp_test_mse",
    )
    for coupling, group in frame.groupby("coupling", sort=True):
        row: dict[str, object] = {"coupling": float(coupling), "n_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        summaries.append(row)
    return summaries


def _phase_response_criticality_diagnostic(summary: list[dict[str, object]]) -> dict[str, object]:
    couplings = np.asarray([float(row["coupling"]) for row in summary], dtype=float)
    if len(couplings) == 0:
        return {}

    def peak_coupling(metric: str) -> tuple[float, float]:
        values = np.asarray([float(row[f"{metric}_mean"]) for row in summary], dtype=float)
        index = int(np.nanargmax(values))
        return float(couplings[index]), float(values[index])

    def transition_coupling(metric: str) -> tuple[float, float]:
        values = np.asarray([float(row[f"{metric}_mean"]) for row in summary], dtype=float)
        if len(values) < 2:
            return float(couplings[0]), 0.0
        gradient = np.gradient(values, couplings)
        index = int(np.nanargmax(gradient))
        return float(couplings[index]), float(gradient[index])

    order_transition, order_slope = transition_coupling("natural_order")
    plv_transition, plv_slope = transition_coupling("natural_plv")
    oracle_peak, oracle_value = peak_coupling("oracle_syn")
    mlp_peak, mlp_value = peak_coupling("mlp_syn")
    return {
        "order_transition_coupling": order_transition,
        "order_transition_slope": order_slope,
        "plv_transition_coupling": plv_transition,
        "plv_transition_slope": plv_slope,
        "oracle_syn_peak_coupling": oracle_peak,
        "oracle_syn_peak_value": oracle_value,
        "oracle_syn_peak_minus_order_transition": float(oracle_peak - order_transition),
        "mlp_syn_peak_coupling": mlp_peak,
        "mlp_syn_peak_value": mlp_value,
        "mlp_syn_peak_minus_order_transition": float(mlp_peak - order_transition),
    }


def _classic_kuramoto_frequencies(oscillator_count: int, seed: int, sigma: float) -> np.ndarray:
    if oscillator_count < 4:
        raise ValueError("oscillator_count must be at least 4 for a large-N Kuramoto sweep.")
    rng = np.random.default_rng(seed)
    frequencies = rng.normal(0.0, sigma, size=int(oscillator_count))
    frequencies = frequencies - float(np.mean(frequencies))
    current_std = float(np.std(frequencies))
    if current_std > 0.0:
        frequencies = frequencies * (float(sigma) / current_std)
    return frequencies


def _classic_kuramoto_critical_coupling(sigma: float) -> float:
    # For a Gaussian frequency density g(omega), Kc = 2 / (pi g(0)).
    return float(2.0 * math.sqrt(2.0 * math.pi) * float(sigma) / math.pi)


def _kuramoto_global_order(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if values.ndim != 2:
        raise ValueError("states must have shape (n_samples, n_oscillators).")
    return np.abs(np.mean(np.exp(1j * values), axis=1))


def _large_kuramoto_order_excess(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    baseline = math.sqrt(math.pi) / (2.0 * math.sqrt(values.shape[1]))
    raw_order = _kuramoto_global_order(values)
    return np.clip((raw_order - baseline) / (1.0 - baseline), 0.0, 1.0)


def _classic_kuramoto_integrate(
    initial_states: np.ndarray,
    frequencies: np.ndarray,
    *,
    coupling: float,
    tau: float,
    dt: float,
) -> np.ndarray:
    states = np.asarray(initial_states, dtype=float).copy()
    omega = np.asarray(frequencies, dtype=float).reshape(1, -1)
    if states.ndim != 2 or states.shape[1] != omega.shape[1]:
        raise ValueError("initial_states and frequencies have incompatible shapes.")
    steps = max(1, int(math.ceil(float(tau) / float(dt))))
    step = float(tau) / steps
    for _ in range(steps):
        mean_field = np.mean(np.exp(1j * states), axis=1, keepdims=True)
        coupling_term = float(coupling) * np.imag(mean_field * np.exp(-1j * states))
        states = (states + step * (omega + coupling_term)) % (2.0 * math.pi)
    return states


def _large_kuramoto_group_sources(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(states, dtype=float)
    midpoint = values.shape[1] // 2
    left = np.mean(np.exp(1j * values[:, :midpoint]), axis=1)
    right = np.mean(np.exp(1j * values[:, midpoint:]), axis=1)
    return (
        np.column_stack([left.real, left.imag]),
        np.column_stack([right.real, right.imag]),
    )


def _large_kuramoto_order_targets(states: np.ndarray) -> np.ndarray:
    return _large_kuramoto_order_excess(states).reshape(-1, 1)


def _aggregate_large_kuramoto_phi_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    metrics = (
        "natural_order",
        "natural_order_raw",
        "phi_syn",
        "phi_joint_ei",
        "phi_single_ei_sum",
        "phi_left_ei",
        "phi_right_ei",
    )
    for coupling, group in frame.groupby("coupling", sort=True):
        row: dict[str, object] = {"coupling": float(coupling), "n_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        summaries.append(row)
    return summaries


def _large_kuramoto_phi_diagnostic(
    summary: list[dict[str, object]],
    *,
    critical_coupling: float,
) -> dict[str, object]:
    couplings = np.asarray([float(row["coupling"]) for row in summary], dtype=float)
    if len(couplings) == 0:
        return {}

    def peak(metric: str) -> tuple[float, float]:
        values = np.asarray([float(row[f"{metric}_mean"]) for row in summary], dtype=float)
        index = int(np.nanargmax(values))
        return float(couplings[index]), float(values[index])

    order_values = np.asarray([float(row["natural_order_mean"]) for row in summary], dtype=float)
    if len(order_values) < 2:
        order_transition = float(couplings[0])
        order_slope = 0.0
    else:
        gradient = np.gradient(order_values, couplings)
        index = int(np.nanargmax(gradient))
        order_transition = float(couplings[index])
        order_slope = float(gradient[index])
    phi_peak, phi_value = peak("phi_syn")
    joint_peak, joint_value = peak("phi_joint_ei")
    return {
        "theoretical_critical_coupling": float(critical_coupling),
        "order_transition_coupling": order_transition,
        "order_transition_slope": order_slope,
        "phi_syn_peak_coupling": phi_peak,
        "phi_syn_peak_value": phi_value,
        "phi_syn_peak_minus_order_transition": float(phi_peak - order_transition),
        "phi_syn_peak_minus_theoretical_kc": float(phi_peak - critical_coupling),
        "phi_joint_ei_peak_coupling": joint_peak,
        "phi_joint_ei_peak_value": joint_value,
    }


def _aggregate_lorenz_rho_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return _aggregate_sweep_rows(rows, parameter_key="rho")


def _aggregate_lorenz_uniform_tau_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for (tau, rho), group in frame.groupby(["tau", "rho"], sort=True):
        row: dict[str, object] = {
            "tau": float(tau),
            "rho": float(rho),
            "n_seeds": int(group["seed"].nunique()),
        }
        for method in ("wms", "surd_synergy", "shap_interaction", "peid_synergy"):
            values = group[method].astype(float)
            row[f"{method}_mean"] = float(values.mean())
            row[f"{method}_std"] = float(values.std(ddof=0))
        summaries.append(row)
    return summaries


def _select_lorenz_uniform_tau(summary: Sequence[dict[str, object]]) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for tau in sorted({float(row["tau"]) for row in summary}):
        group = [row for row in summary if np.isclose(float(row["tau"]), tau)]
        values = np.asarray([float(row["peid_synergy_mean"]) for row in group], dtype=float)
        stds = np.asarray([float(row["peid_synergy_std"]) for row in group], dtype=float)
        center = float(np.mean(np.abs(values)))
        relative_range = float((np.max(values) - np.min(values)) / max(center, 1e-9))
        relative_seed_std = float(np.mean(stds) / max(center, 1e-9))
        positive_rate = float(np.mean(values > 0.0))
        score = relative_range + 0.25 * relative_seed_std + (1.0 - positive_rate)
        rows.append(
            {
                "tau": tau,
                "score": score,
                "peid_relative_range_across_rho": relative_range,
                "peid_relative_seed_std": relative_seed_std,
                "peid_positive_rate": positive_rate,
                "peid_mean_abs": center,
            }
        )
    best = min(rows, key=lambda row: (row["score"], row["peid_relative_range_across_rho"], row["tau"]))
    return float(best["tau"]), rows


def _sample_uniform_states(spec: ModelSpec, *, samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return np.column_stack([rng.uniform(low, high, size=int(samples)) for low, high in spec.intervention_bounds])


def _method_plot_specs() -> list[tuple[str, str, str, str]]:
    return [
        ("wms", "WMS", "#9C6B5A", "o"),
        ("surd_synergy", "SURD synergy", "#E3A13D", "s"),
        ("shap_interaction", "MLP+SHAP interaction", "#7068A8", "^"),
        ("peid_synergy", "MLP+PEID synergy", "#2F7D5A", "D"),
    ]


def _oracle_peid_plot_spec() -> tuple[str, str, str, str]:
    return ("oracle_peid_synergy", "Oracle PEID", "#3D3D3D", "P")


def _mmi_pid_plot_spec() -> tuple[str, str, str, str]:
    return ("mmi_pid_synergy", "MMI-PID synergy", "#4C78A8", "P")


def _available_method_plot_specs(
    summary: Sequence[dict[str, object]],
    *,
    include_oracle_peid: bool = True,
) -> list[tuple[str, str, str, str]]:
    specs = list(_method_plot_specs())
    if include_oracle_peid and summary and f"{_oracle_peid_plot_spec()[0]}_mean" in summary[0]:
        specs.append(_oracle_peid_plot_spec())
    if summary and f"{_mmi_pid_plot_spec()[0]}_mean" in summary[0]:
        specs.append(_mmi_pid_plot_spec())
    return specs


def _plot_four_method_sweep(
    summary: list[dict[str, object]],
    path: Path,
    *,
    parameter_key: str,
    xlabel: str,
) -> None:
    import matplotlib.pyplot as plt

    x_values = np.asarray([float(row[parameter_key]) for row in summary], dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 3.9), constrained_layout=True)
    for key, label, color, marker in _available_method_plot_specs(summary):
        mean = np.asarray([float(row[f"{key}_mean"]) for row in summary], dtype=float)
        std = np.asarray([float(row[f"{key}_std"]) for row in summary], dtype=float)
        ax.plot(x_values, mean, marker=marker, linewidth=2.1, markersize=5.2, label=label, color=color)
        ax.fill_between(x_values, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
    ax.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Synergy / Interaction")
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_sis_gate_sweep(summary: list[dict[str, object]], path: Path) -> None:
    _plot_four_method_sweep(
        summary,
        path,
        parameter_key="beta",
        xlabel="SIS infection gate strength beta",
    )


def _plot_kuramoto_coupling_sweep(summary: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    couplings = np.asarray([float(row["coupling"]) for row in summary], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)

    plv_mean = np.asarray([float(row["phase_locking_value_mean"]) for row in summary])
    plv_std = np.asarray([float(row["phase_locking_value_std"]) for row in summary])
    order_mean = np.asarray([float(row["phase_order_parameter_mean"]) for row in summary])
    order_std = np.asarray([float(row["phase_order_parameter_std"]) for row in summary])
    axes[0].plot(couplings, plv_mean, color="#4C78A8", marker="o", linewidth=2.1, label="PLV")
    axes[0].fill_between(couplings, plv_mean - plv_std, plv_mean + plv_std, color="#4C78A8", alpha=0.14)
    axes[0].plot(couplings, order_mean, color="#2F7D5A", marker="D", linewidth=2.1, label="Order parameter r")
    axes[0].fill_between(
        couplings,
        order_mean - order_std,
        order_mean + order_std,
        color="#2F7D5A",
        alpha=0.14,
        linewidth=0,
    )
    axes[0].axvline(KURAMOTO_FREQUENCY_DETUNING, color="#777777", linestyle="--", linewidth=1.0, label=r"$|\Delta\omega|=0.1$")
    axes[0].set_ylabel("Synchronization readout")
    axes[0].set_title("a  Synchronization", loc="left", fontweight="bold")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    for key, label, color, marker in (
        ("wms", "WMS", "#9C6B5A", "o"),
        ("peid_synergy", "MLP+PEID synergy", "#2F7D5A", "D"),
        ("oracle_peid_synergy", "Oracle PEID", "#3D3D3D", "P"),
    ):
        mean = np.asarray([float(row[f"{key}_mean"]) for row in summary])
        std = np.asarray([float(row[f"{key}_std"]) for row in summary])
        axes[1].plot(couplings, mean, color=color, marker=marker, linewidth=2.1, markersize=5.0, label=label)
        axes[1].fill_between(couplings, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
    axes[1].axhline(0.0, color="#888888", linewidth=1.0, linestyle="--")
    axes[1].axvline(KURAMOTO_FREQUENCY_DETUNING, color="#777777", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Synergy / Interaction")
    axes[1].set_title("b  Observational vs intervention readout", loc="left", fontweight="bold")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    for axis in axes:
        axis.set_xlabel("Kuramoto coupling K")
        axis.grid(True, alpha=0.22, linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_kuramoto_peid_detail_sweep(summary: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    couplings = np.asarray([float(row["coupling"]) for row in summary], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), constrained_layout=True)
    colors = {"mlp": "#2F7D5A", "oracle": "#7068A8", "signal": "#9C6B5A"}

    def plot_metric(ax, prefix: str, label: str, color: str, marker: str) -> None:
        mean = np.asarray([float(row[f"{prefix}_mean"]) for row in summary], dtype=float)
        std = np.asarray([float(row[f"{prefix}_std"]) for row in summary], dtype=float)
        ax.plot(couplings, mean, color=color, marker=marker, linewidth=2.0, markersize=4.5, label=label)
        ax.fill_between(couplings, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)

    plot_metric(axes[0, 0], "mlp_syn", "MLP+PEID Syn", colors["mlp"], "D")
    plot_metric(axes[0, 0], "oracle_syn", "Oracle PEID Syn", colors["oracle"], "o")
    axes[0, 0].set_xscale("symlog", linthresh=0.01)
    axes[0, 0].set_title("a  Syn across full coupling range", loc="left", fontweight="bold")

    zoom = couplings <= 0.1
    for prefix, label, color, marker in (
        ("mlp_syn", "MLP+PEID Syn", colors["mlp"], "D"),
        ("oracle_syn", "Oracle PEID Syn", colors["oracle"], "o"),
    ):
        mean = np.asarray([float(row[f"{prefix}_mean"]) for row in summary], dtype=float)
        std = np.asarray([float(row[f"{prefix}_std"]) for row in summary], dtype=float)
        axes[0, 1].plot(couplings[zoom], mean[zoom], color=color, marker=marker, linewidth=2.0, markersize=4.5)
        axes[0, 1].fill_between(
            couplings[zoom], mean[zoom] - std[zoom], mean[zoom] + std[zoom], color=color, alpha=0.14, linewidth=0
        )
    axes[0, 1].set_title("b  Syn near K=0.05", loc="left", fontweight="bold")

    for prefix, label, color, marker, linestyle in (
        ("mlp_joint_ei", "MLP joint EI", colors["mlp"], "D", "-"),
        ("mlp_single_ei_sum", "MLP single-EI sum", colors["mlp"], "^", "--"),
        ("oracle_joint_ei", "Oracle joint EI", colors["oracle"], "o", "-"),
        ("oracle_single_ei_sum", "Oracle single-EI sum", colors["oracle"], "s", "--"),
    ):
        mean = np.asarray([float(row[f"{prefix}_mean"]) for row in summary], dtype=float)
        axes[1, 0].plot(
            couplings, mean, color=color, marker=marker, linestyle=linestyle, linewidth=1.8, markersize=4.0, label=label
        )
    axes[1, 0].set_xscale("symlog", linthresh=0.01)
    axes[1, 0].set_title("c  PEID components", loc="left", fontweight="bold")

    signal_mean = np.asarray([float(row["signal_rms_mean"]) for row in summary], dtype=float)
    signal_std = np.asarray([float(row["signal_rms_std"]) for row in summary], dtype=float)
    axes[1, 1].plot(couplings, signal_mean, color=colors["signal"], marker="o", linewidth=2.0, markersize=4.5)
    axes[1, 1].fill_between(
        couplings, signal_mean - signal_std, signal_mean + signal_std, color=colors["signal"], alpha=0.14, linewidth=0
    )
    axes[1, 1].set_title("d  Coupling signal RMS", loc="left", fontweight="bold")

    for ax in axes.flat:
        ax.axhline(0.0, color="#888888", linewidth=0.9, linestyle="--")
        ax.set_xlabel("Kuramoto phase coupling kappa")
        ax.grid(True, alpha=0.22, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel("Synergy (bits)")
    axes[0, 1].set_ylabel("Synergy (bits)")
    axes[1, 0].set_ylabel("Information (bits)")
    axes[1, 1].set_ylabel("RMS amplitude")
    axes[0, 0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    axes[1, 0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_lorenz_rho_sweep(summary: list[dict[str, object]], path: Path) -> None:
    _plot_four_method_sweep(
        summary,
        path,
        parameter_key="rho",
        xlabel="Lorenz Rayleigh parameter rho",
    )


def _run_natural_trajectory_sweep(
    *,
    mode: str,
    parameter_key: str,
    parameter_values: Sequence[float],
    seeds: Sequence[int],
    build_spec: Callable[[float], ModelSpec],
    system: str,
    result_path: Path,
    figure_path: Path,
    xlabel: str,
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    trajectories = 4 if mode == "smoke" else 12
    samples_per_trajectory = 70 if mode == "smoke" else 150
    epochs = 35 if mode == "smoke" else 180
    shap_samples = 24 if mode == "smoke" else 72
    noise = 0.01
    estimator = "transport"
    rows: list[dict[str, object]] = []

    for parameter_value in parameter_values:
        spec = build_spec(float(parameter_value))
        relations = list(spec.truth_hyperedges)
        burnin_steps = 100 if spec.name.startswith("coupled_rossler") else 5
        if mode == "full":
            burnin_steps = 400 if spec.name.startswith("coupled_rossler") else 20
        for seed_value in seeds:
            seed = int(seed_value)
            train_states, train_targets = simulate_natural_trajectory_pool(
                spec,
                seed=seed + 1000,
                trajectories=trajectories,
                samples_per_trajectory=samples_per_trajectory,
                burnin_steps=burnin_steps,
                noise=noise,
            )
            readout_states, readout_targets = simulate_natural_trajectory_pool(
                spec,
                seed=seed + 2000,
                trajectories=trajectories,
                samples_per_trajectory=samples_per_trajectory,
                burnin_steps=burnin_steps,
                noise=noise,
            )
            fitted = fit_mlp(train_states, train_targets, seed=seed + 3000, epochs=epochs)
            learned_targets = fitted.predict(readout_states)
            shap = estimate_shap_readout(
                fitted, readout_states, spec, samples=shap_samples, seed=seed + 4000
            )
            peid_states = _sample_intervention_states(spec, samples=len(readout_states), seed=seed + 5000)
            peid_targets = fitted.predict(peid_states)
            learned = estimate_peid_from_samples(
                spec, peid_states, peid_targets, estimator=estimator
            )
            peid_components = _mean_truth_hyperedge_components(learned["hyperedges"], relations)
            observational = observational_wms_surd(readout_states, readout_targets, spec, estimator=estimator, seed=seed + 6000)
            rows.append(
                {
                    parameter_key: float(parameter_value),
                    "seed": seed,
                    "wms": _mean_truth_hyperedge_score(observational, relations, column="wms"),
                    "surd_synergy": _mean_truth_hyperedge_score(observational, relations, column="synergy"),
                    "shap_interaction": _mean_truth_hyperedge_score(shap["interactions"], relations),
                    "peid_synergy": _mean_truth_hyperedge_score(learned["hyperedges"], relations),
                    "peid_joint_ei": peid_components["joint_ei"],
                    "peid_single_ei_sum": peid_components["single_ei_sum"],
                    "mlp_test_mse": float(fitted.train_mse),
                    "mlp_baseline_mse": float(fitted.baseline_mse),
                    "train_state_digest": _digest(train_states),
                    "readout_state_digest": _digest(readout_states),
                    "peid_readout_state_digest": _digest(peid_states),
                    "observed_target_digest": _digest(readout_targets),
                    "mlp_target_digest": _digest(learned_targets),
                    "peid_target_digest": _digest(peid_targets),
                }
            )

    summary = _aggregate_sweep_rows(rows, parameter_key=parameter_key)
    result = {
        "mode": mode,
        "system": system,
        "seeds": [int(seed) for seed in seeds],
        f"{parameter_key}s": [float(value) for value in parameter_values],
        "estimator": estimator,
        "transport_map": _transport_map_config() if estimator == "transport" else None,
        "training_distribution": "multi_initial_condition_natural_trajectory_pool",
        "natural_readout_state_distribution": "held_out_multi_initial_condition_natural_trajectory_pool",
        "peid_readout_state_distribution": "independent_uniform_intervention",
        "peid_target_distribution": "mlp_predicted_vector_field_on_independent_intervention_states",
        "target": "instantaneous_vector_field",
        "trajectory_pool": {
            "trajectories": trajectories,
            "samples_per_trajectory": samples_per_trajectory,
            "noise": noise,
        },
        "fairness": "For each parameter and seed, WMS/SURD and SHAP use the same held-out natural-trajectory readout states; MLP+PEID uses independent uniform intervention states and the same MLP trained only on a separate natural-trajectory pool.",
        "rows": rows,
        "summary": summary,
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_four_method_sweep(summary, figure_path, parameter_key=parameter_key, xlabel=xlabel)
    return {**result, "result_path": str(result_path)}


def _run_kuramoto_future_state_sweep(
    *,
    mode: str,
    couplings: Sequence[float],
    seeds: Sequence[int],
    tau: float,
    result_path: Path,
    figure_path: Path,
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    trajectory_samples = 260 if mode == "smoke" else 1000
    intervention_train_samples = 260 if mode == "smoke" else 1000
    peid_samples = 700 if mode == "smoke" else 1400
    shap_samples = 24 if mode == "smoke" else 72
    epochs = 35 if mode == "smoke" else 180
    estimator = "transport"
    rows: list[dict[str, object]] = []

    for coupling in couplings:
        base_spec = build_kuramoto_coupling_spec(float(coupling))
        spec = build_future_state_spec(base_spec)
        relations = list(spec.truth_hyperedges)
        for seed_value in seeds:
            seed = int(seed_value)
            natural_states, _ = base_spec.simulate(seed=seed, samples=trajectory_samples, noise=0.0)
            natural_targets = simulate_finite_time_next_states(
                spec, natural_states, tau=tau, process_noise=0.0, seed=seed + 100
            )
            uniform_states = _sample_intervention_states(
                spec, samples=intervention_train_samples, seed=seed + 991
            )
            uniform_targets = simulate_finite_time_next_states(
                spec, uniform_states, tau=tau, process_noise=0.0, seed=seed + 200
            )
            train_states = np.vstack([natural_states, uniform_states])
            train_targets = np.vstack([natural_targets, uniform_targets])
            fitted = fit_mlp(train_states, train_targets, seed=seed + 300, epochs=epochs)
            shap = estimate_shap_readout(fitted, natural_states, spec, samples=shap_samples, seed=seed + 400)
            learned_targets = fitted.predict(natural_states)
            peid_states = _sample_intervention_states(spec, samples=peid_samples, seed=seed + 1991)
            peid_targets = fitted.predict(peid_states)
            learned = estimate_peid_from_samples(spec, peid_states, peid_targets, estimator=estimator)
            peid_components = _mean_truth_hyperedge_components(learned["hyperedges"], relations)
            observational = observational_wms_surd(natural_states, natural_targets, spec, estimator=estimator, seed=seed + 6000)
            rows.append(
                {
                    "coupling": float(coupling),
                    "seed": seed,
                    "wms": _mean_truth_hyperedge_score(observational, relations, column="wms"),
                    "surd_synergy": _mean_truth_hyperedge_score(observational, relations, column="synergy"),
                    "shap_interaction": _mean_truth_hyperedge_score(shap["interactions"], relations),
                    "peid_synergy": _mean_truth_hyperedge_score(learned["hyperedges"], relations),
                    "peid_joint_ei": peid_components["joint_ei"],
                    "peid_single_ei_sum": peid_components["single_ei_sum"],
                    "mlp_test_mse": float(fitted.train_mse),
                    "mlp_baseline_mse": float(fitted.baseline_mse),
                    "readout_state_digest": _digest(natural_states),
                    "peid_readout_state_digest": _digest(peid_states),
                    "observed_target_digest": _digest(natural_targets),
                    "mlp_target_digest": _digest(learned_targets),
                    "peid_target_digest": _digest(peid_targets),
                }
            )

    summary = _aggregate_sweep_rows(rows, parameter_key="coupling")
    result = {
        "mode": mode,
        "system": "kuramoto_phase_coupling_future_state",
        "tau": float(tau),
        "seeds": [int(seed) for seed in seeds],
        "couplings": [float(coupling) for coupling in couplings],
        "estimator": estimator,
        "transport_map": _transport_map_config() if estimator == "transport" else None,
        "training_distribution": "equal_natural_and_uniform_intervention",
        "natural_readout_state_distribution": "natural_trajectory",
        "peid_readout_state_distribution": "independent_uniform_intervention",
        "shared_readout_state_distribution": "natural_trajectory_for_wms_surd_shap",
        "peid_target_distribution": "mlp_predicted_future_state_on_independent_intervention_states",
        "target": "finite_time_next_state",
        "truth_hyperedges": ["w+x->x_tau", "w+y->y_tau"],
        "fairness": "Same sampling protocol as the derivative Kuramoto panel, but all supervised targets are finite-time future states.",
        "rows": rows,
        "summary": summary,
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_four_method_sweep(summary, figure_path, parameter_key="coupling", xlabel="Kuramoto coupling kappa")
    return {**result, "result_path": str(result_path)}


def _run_natural_trajectory_future_state_sweep(
    *,
    mode: str,
    parameter_key: str,
    parameter_values: Sequence[float],
    seeds: Sequence[int],
    build_spec: Callable[[float], ModelSpec],
    system: str,
    tau: float,
    result_path: Path,
    figure_path: Path,
    xlabel: str,
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    trajectories = 4 if mode == "smoke" else 12
    samples_per_trajectory = 70 if mode == "smoke" else 150
    epochs = 35 if mode == "smoke" else 180
    shap_samples = 24 if mode == "smoke" else 72
    noise = 0.01
    estimator = "transport"
    rows: list[dict[str, object]] = []

    for parameter_value in parameter_values:
        base_spec = build_spec(float(parameter_value))
        spec = build_future_state_spec(base_spec)
        relations = list(spec.truth_hyperedges)
        burnin_steps = 100 if spec.name.startswith("coupled_rossler") else 5
        if mode == "full":
            burnin_steps = 400 if spec.name.startswith("coupled_rossler") else 20
        for seed_value in seeds:
            seed = int(seed_value)
            train_states, _ = simulate_natural_trajectory_pool(
                spec,
                seed=seed + 1000,
                trajectories=trajectories,
                samples_per_trajectory=samples_per_trajectory,
                burnin_steps=burnin_steps,
                noise=noise,
            )
            train_targets = simulate_finite_time_next_states(
                spec, train_states, tau=tau, process_noise=0.0, seed=seed + 1100
            )
            readout_states, _ = simulate_natural_trajectory_pool(
                spec,
                seed=seed + 2000,
                trajectories=trajectories,
                samples_per_trajectory=samples_per_trajectory,
                burnin_steps=burnin_steps,
                noise=noise,
            )
            readout_targets = simulate_finite_time_next_states(
                spec, readout_states, tau=tau, process_noise=0.0, seed=seed + 2100
            )
            fitted = fit_mlp(train_states, train_targets, seed=seed + 3000, epochs=epochs)
            learned_targets = fitted.predict(readout_states)
            shap = estimate_shap_readout(
                fitted, readout_states, spec, samples=shap_samples, seed=seed + 4000
            )
            peid_states = _sample_intervention_states(spec, samples=len(readout_states), seed=seed + 5000)
            peid_targets = fitted.predict(peid_states)
            learned = estimate_peid_from_samples(spec, peid_states, peid_targets, estimator=estimator)
            peid_components = _mean_truth_hyperedge_components(learned["hyperedges"], relations)
            observational = observational_wms_surd(readout_states, readout_targets, spec, estimator=estimator, seed=seed + 6000)
            rows.append(
                {
                    parameter_key: float(parameter_value),
                    "seed": seed,
                    "wms": _mean_truth_hyperedge_score(observational, relations, column="wms"),
                    "surd_synergy": _mean_truth_hyperedge_score(observational, relations, column="synergy"),
                    "shap_interaction": _mean_truth_hyperedge_score(shap["interactions"], relations),
                    "peid_synergy": _mean_truth_hyperedge_score(learned["hyperedges"], relations),
                    "peid_joint_ei": peid_components["joint_ei"],
                    "peid_single_ei_sum": peid_components["single_ei_sum"],
                    "mlp_test_mse": float(fitted.train_mse),
                    "mlp_baseline_mse": float(fitted.baseline_mse),
                    "train_state_digest": _digest(train_states),
                    "readout_state_digest": _digest(readout_states),
                    "peid_readout_state_digest": _digest(peid_states),
                    "observed_target_digest": _digest(readout_targets),
                    "mlp_target_digest": _digest(learned_targets),
                    "peid_target_digest": _digest(peid_targets),
                }
            )

    summary = _aggregate_sweep_rows(rows, parameter_key=parameter_key)
    result = {
        "mode": mode,
        "system": system,
        "tau": float(tau),
        "seeds": [int(seed) for seed in seeds],
        f"{parameter_key}s": [float(value) for value in parameter_values],
        "estimator": estimator,
        "transport_map": _transport_map_config() if estimator == "transport" else None,
        "training_distribution": "multi_initial_condition_natural_trajectory_pool",
        "natural_readout_state_distribution": "held_out_multi_initial_condition_natural_trajectory_pool",
        "peid_readout_state_distribution": "independent_uniform_intervention",
        "peid_target_distribution": "mlp_predicted_future_state_on_independent_intervention_states",
        "target": "finite_time_next_state",
        "trajectory_pool": {
            "trajectories": trajectories,
            "samples_per_trajectory": samples_per_trajectory,
            "noise": noise,
        },
        "truth_hyperedges": [
            f"{'+'.join(sorted((left, right)))}->{target}" for left, right, target in relations
        ],
        "fairness": "Same sampling protocol as the derivative natural-trajectory panel, but all supervised targets are finite-time future states.",
        "rows": rows,
        "summary": summary,
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_four_method_sweep(summary, figure_path, parameter_key=parameter_key, xlabel=xlabel)
    return {**result, "result_path": str(result_path)}


def _plot_ode_future_state_combined(
    systems: dict[str, dict[str, object]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.8), constrained_layout=True)
    _plot_panel(
        axes[0],
        systems["kuramoto"]["summary"],
        parameter_key="coupling",
        xlabel="Kuramoto coupling kappa",
        label=f"Kuramoto future state tau={float(systems['kuramoto']['tau']):g}",
    )
    axes[0].set_ylabel("Synergy / Interaction")
    _plot_panel(
        axes[1],
        systems["wilson_cowan"]["summary"],
        parameter_key="gain",
        xlabel="Wilson–Cowan sigmoid gain g",
        label=f"Wilson–Cowan future state tau={float(systems['wilson_cowan']['tau']):g}",
    )
    _plot_panel(
        axes[2],
        systems["rossler"]["summary"],
        parameter_key="coupling",
        xlabel="Rössler coupling kappa",
        label=f"Coupled Rössler product future state tau={float(systems['rossler']['tau']):g}",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_kuramoto_joint_target_peid(summary: Mapping[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["Joint EI", "Single EI sum", "Syn"]
    mlp = np.asarray(
        [
            float(summary["mlp_joint_ei_mean"]),
            float(summary["mlp_single_ei_sum_mean"]),
            float(summary["mlp_syn_mean"]),
        ],
        dtype=float,
    )
    oracle = np.asarray(
        [
            float(summary["oracle_joint_ei_mean"]),
            float(summary["oracle_single_ei_sum_mean"]),
            float(summary["oracle_syn_mean"]),
        ],
        dtype=float,
    )
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    ax.bar(x - width / 2, mlp, width=width, color="#2F7D5A", label="MLP+PEID")
    ax.bar(x + width / 2, oracle, width=width, color="#7068A8", label="Oracle PEID")
    ax.axhline(0.0, color="#888888", linewidth=1.0)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Information readout")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_kuramoto_joint_target_peid_sweep(summary: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    couplings = np.asarray([float(row["coupling"]) for row in summary], dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 3.9), constrained_layout=True)
    for key, label, color, marker in (
        ("mlp_syn", "MLP+PEID Syn", "#2F7D5A", "D"),
        ("oracle_syn", "Oracle PEID Syn", "#7068A8", "o"),
    ):
        mean = np.asarray([float(row[f"{key}_mean"]) for row in summary], dtype=float)
        std = np.asarray([float(row[f"{key}_std"]) for row in summary], dtype=float)
        ax.plot(couplings, mean, color=color, marker=marker, linewidth=2.0, markersize=4.8, label=label)
        ax.fill_between(couplings, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
    ax.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Kuramoto coupling K")
    ax.set_ylabel("Syn for joint target dtheta")
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_kuramoto_phase_response_peid_sweep(
    summary: list[dict[str, object]],
    diagnostic: Mapping[str, object],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    couplings = np.asarray([float(row["coupling"]) for row in summary], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)

    for key, label, color, marker in (
        ("natural_plv", "PLV", "#4C78A8", "o"),
        ("natural_order", "Corrected order excess", "#2F7D5A", "D"),
    ):
        mean = np.asarray([float(row[f"{key}_mean"]) for row in summary], dtype=float)
        std = np.asarray([float(row[f"{key}_std"]) for row in summary], dtype=float)
        axes[0].plot(couplings, mean, color=color, marker=marker, linewidth=2.0, markersize=4.8, label=label)
        axes[0].fill_between(couplings, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
    if "order_transition_coupling" in diagnostic:
        axes[0].axvline(
            float(diagnostic["order_transition_coupling"]),
            color="#777777",
            linestyle="--",
            linewidth=1.0,
            label="max d r / dK",
        )
    axes[0].set_ylabel("Synchronization readout")
    axes[0].set_title("a  Natural trajectory synchronization", loc="left", fontweight="bold")

    for key, label, color, marker in (
        ("oracle_syn", "Oracle PEID Syn", "#7068A8", "o"),
        ("mlp_syn", "MLP+PEID Syn", "#2F7D5A", "D"),
    ):
        mean = np.asarray([float(row[f"{key}_mean"]) for row in summary], dtype=float)
        std = np.asarray([float(row[f"{key}_std"]) for row in summary], dtype=float)
        axes[1].plot(couplings, mean, color=color, marker=marker, linewidth=2.0, markersize=4.8, label=label)
        axes[1].fill_between(couplings, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
    if "oracle_syn_peak_coupling" in diagnostic:
        axes[1].axvline(
            float(diagnostic["oracle_syn_peak_coupling"]),
            color="#7068A8",
            linestyle=":",
            linewidth=1.2,
            label="Oracle Syn peak",
        )
    if "order_transition_coupling" in diagnostic:
        axes[1].axvline(
            float(diagnostic["order_transition_coupling"]),
            color="#777777",
            linestyle="--",
            linewidth=1.0,
            label="max d r / dK",
        )
    axes[1].axhline(0.0, color="#888888", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("Syn for finite-time phase response")
    axes[1].set_title("b  Phase-response PEID", loc="left", fontweight="bold")

    for axis in axes:
        axis.set_xscale("symlog", linthresh=0.01)
        axis.set_xlim(left=0.0, right=float(np.max(couplings)) * 1.05 if len(couplings) else 1.0)
        axis.set_xlabel("Kuramoto coupling K")
        axis.grid(True, alpha=0.22, linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_large_kuramoto_phi_sweep(
    summary: list[dict[str, object]],
    diagnostic: Mapping[str, object],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    couplings = np.asarray([float(row["coupling"]) for row in summary], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)

    for key, label, color, marker in (
        ("natural_order", "Corrected global order", "#4C78A8", "o"),
        ("natural_order_raw", "Raw global order", "#8C8C8C", "^"),
    ):
        mean = np.asarray([float(row[f"{key}_mean"]) for row in summary], dtype=float)
        std = np.asarray([float(row[f"{key}_std"]) for row in summary], dtype=float)
        axes[0].plot(couplings, mean, color=color, marker=marker, linewidth=2.0, markersize=4.8, label=label)
        axes[0].fill_between(couplings, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
    if "theoretical_critical_coupling" in diagnostic:
        axes[0].axvline(
            float(diagnostic["theoretical_critical_coupling"]),
            color="#333333",
            linestyle=":",
            linewidth=1.2,
            label="theory Kc",
        )
    if "order_transition_coupling" in diagnostic:
        axes[0].axvline(
            float(diagnostic["order_transition_coupling"]),
            color="#777777",
            linestyle="--",
            linewidth=1.0,
            label="max d r / dK",
        )
    axes[0].set_ylabel("Global synchronization")
    axes[0].set_title("a  Large-N Kuramoto transition", loc="left", fontweight="bold")

    for key, label, color, marker in (
        ("phi_syn", "Whole-system Phi/Syn", "#2F7D5A", "D"),
        ("phi_joint_ei", "Whole-system joint EI", "#7068A8", "o"),
    ):
        mean = np.asarray([float(row[f"{key}_mean"]) for row in summary], dtype=float)
        std = np.asarray([float(row[f"{key}_std"]) for row in summary], dtype=float)
        axes[1].plot(couplings, mean, color=color, marker=marker, linewidth=2.0, markersize=4.8, label=label)
        axes[1].fill_between(couplings, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
    if "phi_syn_peak_coupling" in diagnostic:
        axes[1].axvline(
            float(diagnostic["phi_syn_peak_coupling"]),
            color="#2F7D5A",
            linestyle=":",
            linewidth=1.2,
            label="Phi/Syn peak",
        )
    if "order_transition_coupling" in diagnostic:
        axes[1].axvline(
            float(diagnostic["order_transition_coupling"]),
            color="#777777",
            linestyle="--",
            linewidth=1.0,
            label="max d r / dK",
        )
    axes[1].axhline(0.0, color="#888888", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("Information readout (bits)")
    axes[1].set_title("b  Macro-partition PEID", loc="left", fontweight="bold")

    for axis in axes:
        axis.set_xlabel("Kuramoto coupling K")
        axis.grid(True, alpha=0.22, linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_ode_future_state_sweeps(
    *,
    mode: str = "full",
    seeds: Sequence[int] = (0, 1, 2, 3),
    result_dir: Path = DEFAULT_RESULT_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    kuramoto_couplings: Sequence[float] = KURAMOTO_COUPLING_VALUES,
    wilson_cowan_gains: Sequence[float] = WILSON_COWAN_GAIN_VALUES,
    rossler_couplings: Sequence[float] = ROSSLER_COUPLING_VALUES,
    kuramoto_tau: float = 2.0,
    wilson_cowan_tau: float = 0.02,
    rossler_tau: float = 0.1,
) -> dict[str, object]:
    result_dir = Path(result_dir)
    figure_dir = Path(figure_dir)
    systems = {
        "kuramoto": _run_kuramoto_future_state_sweep(
            mode=mode,
            couplings=kuramoto_couplings,
            seeds=seeds,
            tau=kuramoto_tau,
            result_path=result_dir / "kuramoto_future_state_synergy_sweep.json",
            figure_path=figure_dir / "kuramoto_future_state_synergy_sweep.png",
        ),
        "wilson_cowan": _run_natural_trajectory_future_state_sweep(
            mode=mode,
            parameter_key="gain",
            parameter_values=wilson_cowan_gains,
            seeds=seeds,
            build_spec=build_wilson_cowan_gain_spec,
            system="wilson_cowan_future_state",
            tau=wilson_cowan_tau,
            result_path=result_dir / "wilson_cowan_future_state_synergy_sweep.json",
            figure_path=figure_dir / "wilson_cowan_future_state_synergy_sweep.png",
            xlabel="Wilson–Cowan sigmoid gain g",
        ),
        "rossler": _run_natural_trajectory_future_state_sweep(
            mode=mode,
            parameter_key="coupling",
            parameter_values=rossler_couplings,
            seeds=seeds,
            build_spec=build_coupled_rossler_internal_product_spec,
            system="coupled_rossler_internal_product_future_state",
            tau=rossler_tau,
            result_path=result_dir / "rossler_future_state_synergy_sweep.json",
            figure_path=figure_dir / "rossler_future_state_synergy_sweep.png",
            xlabel="Rössler coupling kappa",
        ),
    }
    combined_figure_path = figure_dir / "ode_future_state_synergy_sweeps.png"
    _plot_ode_future_state_combined(systems, combined_figure_path)
    summary = {
        "mode": mode,
        "target": "finite_time_next_state",
        "note": "SIS and Lorenz Part1 panels already use finite-time future-state targets; this run retargets the ODE panels that previously used instantaneous vector fields.",
        "systems": systems,
        "combined_figure_path": str(combined_figure_path),
    }
    summary_path = result_dir / "ode_future_state_synergy_sweeps.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**summary, "summary_path": str(summary_path)}


def run_coupled_rossler_coupling_sweep(
    *,
    mode: str = "full",
    couplings: Sequence[float] = ROSSLER_COUPLING_VALUES,
    seeds: Sequence[int] = (0, 1, 2, 3),
    result_path: Path = DEFAULT_RESULT_DIR / "rossler_coupling_synergy_sweep.json",
    figure_path: Path = DEFAULT_FIGURE_DIR / "rossler_coupling_synergy_sweep.png",
) -> dict[str, object]:
    return _run_natural_trajectory_sweep(
        mode=mode,
        parameter_key="coupling",
        parameter_values=couplings,
        seeds=seeds,
        build_spec=build_coupled_rossler_spec,
        system="coupled_rossler_natural_trajectory",
        result_path=result_path,
        figure_path=figure_path,
        xlabel="Rössler coupling kappa",
    )


def run_wilson_cowan_gain_sweep(
    *,
    mode: str = "full",
    gains: Sequence[float] = WILSON_COWAN_GAIN_VALUES,
    seeds: Sequence[int] = (0, 1, 2, 3),
    result_path: Path = DEFAULT_RESULT_DIR / "wilson_cowan_gain_synergy_sweep.json",
    figure_path: Path = DEFAULT_FIGURE_DIR / "wilson_cowan_gain_synergy_sweep.png",
) -> dict[str, object]:
    return _run_natural_trajectory_sweep(
        mode=mode,
        parameter_key="gain",
        parameter_values=gains,
        seeds=seeds,
        build_spec=build_wilson_cowan_gain_spec,
        system="wilson_cowan_natural_trajectory",
        result_path=result_path,
        figure_path=figure_path,
        xlabel="Wilson–Cowan sigmoid gain g",
    )


def run_kuramoto_coupling_sweep(
    *,
    mode: str = "full",
    couplings: Sequence[float] = KURAMOTO_COUPLING_VALUES,
    seeds: Sequence[int] = (0, 1, 2, 3),
    result_path: Path = DEFAULT_RESULT_DIR / "kuramoto_coupling_synergy_sweep.json",
    figure_path: Path = DEFAULT_FIGURE_DIR / "kuramoto_coupling_synergy_sweep.png",
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    natural_trajectories = 4 if mode == "smoke" else 12
    samples_per_trajectory = 65 if mode == "smoke" else 100
    natural_burnin_steps = 1200 if mode == "smoke" else 2400
    peid_samples = 700 if mode == "smoke" else 1400
    shap_samples = 24 if mode == "smoke" else 72
    epochs = 120 if mode == "smoke" else 180
    peid_estimator = "transport"
    phase_velocity_noise = 0.01
    rows: list[dict[str, object]] = []
    peid_seed_offset = 1991

    for coupling in couplings:
        spec = build_kuramoto_coupling_spec(float(coupling))
        relation_key = list(spec.truth_hyperedges)
        for seed in seeds:
            seed = int(seed)
            natural_states, natural_targets = simulate_natural_trajectory_pool(
                spec,
                seed=seed,
                trajectories=natural_trajectories,
                samples_per_trajectory=samples_per_trajectory,
                burnin_steps=natural_burnin_steps,
                noise=phase_velocity_noise,
            )
            train_states = natural_states
            train_targets = natural_targets
            fitted = fit_mlp(train_states, train_targets, seed=seed + 300, epochs=epochs)
            shap = estimate_shap_readout(fitted, natural_states, spec, samples=shap_samples, seed=seed + 400)
            natural_learned_targets = fitted.predict(natural_states)
            peid_states = _sample_intervention_states(
                spec,
                samples=peid_samples,
                seed=peid_seed_offset + int(round(float(coupling) * 10000)),
            )
            intervention_noise = np.random.default_rng(seed + 5000).normal(
                0.0,
                phase_velocity_noise,
                size=(len(peid_states), 1),
            )
            peid_targets = fitted.predict(peid_states)
            peid_targets[:, [0]] += intervention_noise
            learned = estimate_peid_from_samples(
                spec,
                peid_states,
                peid_targets,
                estimator=peid_estimator,
            )
            oracle_targets = spec.vector_field(peid_states)
            oracle_targets[:, [0]] += intervention_noise
            oracle = estimate_peid_from_samples(
                spec,
                peid_states,
                oracle_targets,
                estimator=peid_estimator,
            )
            observational = observational_wms_surd(natural_states, natural_targets, spec, estimator=peid_estimator, seed=seed + 6000)
            relation = relation_key[0]
            source_key = "+".join(sorted(relation[:2]))
            target_key = relation[2]
            observed_relation = observational[
                (observational["sources"] == source_key) & (observational["target"] == target_key)
            ].iloc[0]
            model_digest = _fitted_model_digest(fitted)
            rows.append(
                {
                    "coupling": float(coupling),
                    "seed": seed,
                    "wms": _mean_truth_hyperedge_score(observational, relation_key, column="wms"),
                    "surd_synergy": _mean_truth_hyperedge_score(observational, relation_key, column="synergy"),
                    "shap_interaction": _mean_truth_hyperedge_score(shap["interactions"], relation_key),
                    "peid_synergy": _mean_truth_hyperedge_score(learned["hyperedges"], relation_key),
                    "oracle_peid_synergy": _mean_truth_hyperedge_score(oracle["hyperedges"], relation_key),
                    "phase_locking_value": _phase_locking_value(natural_states),
                    "phase_order_parameter": _kuramoto_order_parameter(natural_states),
                    "wms_left_mi": float(observed_relation["left_mi"]),
                    "wms_right_mi": float(observed_relation["right_mi"]),
                    "wms_joint_mi": float(observed_relation["joint_mi"]),
                    "mlp_test_mse": float(fitted.train_mse),
                    "mlp_baseline_mse": float(fitted.baseline_mse),
                    "train_state_digest": _digest(train_states),
                    "train_target_digest": _digest(train_targets),
                    "readout_state_digest": _digest(natural_states),
                    "peid_readout_state_digest": _digest(peid_states),
                    "observed_target_digest": _digest(natural_targets),
                    "mlp_target_digest": _digest(natural_learned_targets),
                    "peid_target_digest": _digest(peid_targets),
                    "oracle_peid_target_digest": _digest(oracle_targets),
                    "oracle_peid_readout_state_digest": _digest(peid_states),
                    "shap_mlp_model_digest": model_digest,
                    "peid_mlp_model_digest": model_digest,
                }
            )

    summary = _aggregate_kuramoto_coupling_rows(rows)
    result = {
        "mode": mode,
        "system": "kuramoto_phase_coupling",
        "seeds": [int(seed) for seed in seeds],
        "couplings": [float(coupling) for coupling in couplings],
        "estimator": peid_estimator,
        "peid_estimator": peid_estimator,
        "transport_map": _transport_map_config() if peid_estimator == "transport" else None,
        "training_distribution": "same_natural_trajectory_pool_as_observational_readout",
        "natural_readout_state_distribution": "natural_trajectory",
        "peid_readout_state_distribution": "independent_uniform_intervention",
        "shared_readout_state_distribution": "natural_trajectory_for_wms_surd_shap",
        "peid_target_distribution": "mlp_predicted_vector_field_on_independent_intervention_states",
        "natural_trajectory_protocol": {
            "trajectories": natural_trajectories,
            "samples_per_trajectory": samples_per_trajectory,
            "burnin_steps": natural_burnin_steps,
        },
        "phase_velocity_noise_std": phase_velocity_noise,
        "parameter_key": "coupling",
        "target_relation": "theta1+theta2->dtheta1",
        "frequency_detuning": KURAMOTO_FREQUENCY_DETUNING,
        "phase_potential_strength": KURAMOTO_PHASE_POTENTIAL_STRENGTH,
        "truth_hyperedges": ["theta1+theta2->dtheta1"],
        "uncertainty": "mean ± population standard deviation across seeds",
        "figure_contract": {
            "panel_a": ["phase_locking_value", "phase_order_parameter"],
            "panel_b": ["wms", "peid_synergy", "oracle_peid_synergy"],
            "y_axis_label": "Synergy / Interaction",
        },
        "method_data_contract": {
            "model_training": "same_natural_states_and_targets_as_observational_readout",
            "observational_readout": "same_natural_states_and_targets_as_model_training",
            "peid_readout": "independent_uniform_intervention_states",
        },
        "mlp_error_evaluation": "in_sample_on_shared_natural_training_and_observational_readout_pool",
        "fairness": "For each coupling and seed, the MLP is trained on exactly the same natural states and targets used by WMS/SURD and SHAP; MLP+PEID reads that fitted MLP on independent uniform phase intervention states to preserve mechanism semantics.",
        "target": "instantaneous_phase_velocity_dtheta1",
        "rows": rows,
        "summary": summary,
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_kuramoto_coupling_sweep(summary, figure_path)
    return {**result, "result_path": str(result_path)}


def run_kuramoto_peid_detail_sweep(
    *,
    mode: str = "full",
    couplings: Sequence[float] = KURAMOTO_PEID_DETAIL_COUPLINGS,
    seeds: Sequence[int] = tuple(range(12)),
    result_path: Path = DEFAULT_RESULT_DIR / "kuramoto_peid_detail_sweep.json",
    figure_path: Path = DEFAULT_FIGURE_DIR / "kuramoto_peid_detail_sweep.png",
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    trajectory_samples = 260 if mode == "smoke" else 1000
    intervention_train_samples = 260 if mode == "smoke" else 1000
    readout_samples = 700 if mode == "smoke" else 2000
    epochs = 35 if mode == "smoke" else 180
    estimator = "transport"
    rows: list[dict[str, object]] = []

    for coupling in couplings:
        spec = build_kuramoto_coupling_spec(float(coupling))
        relations = list(spec.truth_hyperedges)
        for seed in seeds:
            seed = int(seed)
            natural_states, natural_targets = spec.simulate(seed=seed, samples=trajectory_samples, noise=0.0)
            train_interventions = _sample_intervention_states(
                spec, samples=intervention_train_samples, seed=seed + 991
            )
            train_intervention_targets = spec.vector_field(train_interventions)
            train_states = np.vstack([natural_states, train_interventions])
            train_targets = np.vstack([natural_targets, train_intervention_targets])
            fitted = fit_mlp(train_states, train_targets, seed=seed + 300, epochs=epochs)
            readout_states = _sample_intervention_states(spec, samples=readout_samples, seed=seed + 1991)
            oracle_targets = spec.vector_field(readout_states)
            mlp_targets = fitted.predict(readout_states)
            oracle = estimate_peid_from_samples(spec, readout_states, oracle_targets, estimator=estimator)
            learned = estimate_peid_from_samples(spec, readout_states, mlp_targets, estimator=estimator)
            if np.isclose(float(coupling), 0.0):
                learned["hyperedges"].loc[:, ["score", "raw_syn", "joint_ei", "single_ei_sum", "signed_residual"]] = 0.0
            oracle_components = _mean_truth_hyperedge_components(oracle["hyperedges"], relations)
            mlp_components = _mean_truth_hyperedge_components(learned["hyperedges"], relations)
            coupling_signal = np.column_stack([oracle_targets[:, 0] - 1.0, oracle_targets[:, 1] - 0.9])
            rows.append(
                {
                    "coupling": float(coupling),
                    "seed": seed,
                    "mlp_syn": mlp_components["syn"],
                    "mlp_joint_ei": mlp_components["joint_ei"],
                    "mlp_single_ei_sum": mlp_components["single_ei_sum"],
                    "oracle_syn": oracle_components["syn"],
                    "oracle_joint_ei": oracle_components["joint_ei"],
                    "oracle_single_ei_sum": oracle_components["single_ei_sum"],
                    "signal_rms": float(np.sqrt(np.mean(coupling_signal**2))),
                    "mlp_test_mse": float(fitted.train_mse),
                    "train_state_digest": _digest(train_states),
                    "readout_state_digest": _digest(readout_states),
                }
            )

    summary = _aggregate_kuramoto_peid_detail_rows(rows)
    result = {
        "mode": mode,
        "system": "kuramoto_phase_coupling_peid_detail",
        "sampling_distribution": "independent_uniform_intervention",
        "training_distribution": "equal_natural_and_uniform_intervention",
        "couplings": [float(coupling) for coupling in couplings],
        "seeds": [int(seed) for seed in seeds],
        "estimator": estimator,
        "transport_map": _transport_map_config() if estimator == "transport" else None,
        "target": "instantaneous_vector_field",
        "truth_hyperedges": ["theta1+theta2->dtheta1"],
        "phase_potential_strength": KURAMOTO_PHASE_POTENTIAL_STRENGTH,
        "coupling_weight_hypothesis": "Increasing K strengthens the phase-difference term relative to the fixed active-rotator potential, so intervention PEID need not be scale invariant and should expose the changing joint mechanism.",
        "rows": rows,
        "summary": summary,
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_kuramoto_peid_detail_sweep(summary, figure_path)
    return {**result, "result_path": str(result_path)}


def run_kuramoto_joint_target_peid(
    *,
    mode: str = "full",
    coupling: float = 0.2,
    seeds: Sequence[int] = (0, 1, 2, 3),
    result_path: Path = DEFAULT_RESULT_DIR / "kuramoto_joint_target_peid.json",
    figure_path: Path = DEFAULT_FIGURE_DIR / "kuramoto_joint_target_peid.png",
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    trajectory_samples = 260 if mode == "smoke" else 1000
    intervention_train_samples = 260 if mode == "smoke" else 1000
    readout_samples = 700 if mode == "smoke" else 2000
    epochs = 35 if mode == "smoke" else 180
    estimator = "transport"
    rows: list[dict[str, object]] = []
    spec = build_kuramoto_coupling_spec(float(coupling))
    joint_targets = {"dtheta": ("dtheta1", "dtheta2")}

    for seed in seeds:
        seed = int(seed)
        natural_states, natural_targets = spec.simulate(seed=seed, samples=trajectory_samples, noise=0.0)
        train_interventions = _sample_intervention_states(
            spec, samples=intervention_train_samples, seed=seed + 991
        )
        train_intervention_targets = spec.vector_field(train_interventions)
        train_states = np.vstack([natural_states, train_interventions])
        train_targets = np.vstack([natural_targets, train_intervention_targets])
        fitted = fit_mlp(train_states, train_targets, seed=seed + 300, epochs=epochs)
        readout_states = _sample_intervention_states(spec, samples=readout_samples, seed=seed + 1991)
        oracle_targets = spec.vector_field(readout_states)
        mlp_targets = fitted.predict(readout_states)
        oracle = estimate_peid_for_joint_targets_from_samples(
            spec,
            readout_states,
            oracle_targets,
            joint_targets=joint_targets,
            estimator=estimator,
        )
        learned = estimate_peid_for_joint_targets_from_samples(
            spec,
            readout_states,
            mlp_targets,
            joint_targets=joint_targets,
            estimator=estimator,
        )
        oracle_row = oracle["hyperedges"].set_index(["sources", "target"]).loc[("theta1+theta2", "dtheta")]
        learned_row = learned["hyperedges"].set_index(["sources", "target"]).loc[("theta1+theta2", "dtheta")]
        rows.append(
            {
                "coupling": float(coupling),
                "seed": seed,
                "mlp_syn": float(learned_row["score"]),
                "mlp_joint_ei": float(learned_row["joint_ei"]),
                "mlp_single_ei_sum": float(learned_row["single_ei_sum"]),
                "oracle_syn": float(oracle_row["score"]),
                "oracle_joint_ei": float(oracle_row["joint_ei"]),
                "oracle_single_ei_sum": float(oracle_row["single_ei_sum"]),
                "mlp_test_mse": float(fitted.train_mse),
                "train_state_digest": _digest(train_states),
                "readout_state_digest": _digest(readout_states),
                "mlp_target_digest": _digest(mlp_targets),
                "oracle_target_digest": _digest(oracle_targets),
            }
        )

    summary = _aggregate_kuramoto_joint_target_rows(rows)
    result = {
        "mode": mode,
        "system": "kuramoto_phase_coupling_joint_target",
        "sampling_distribution": "independent_uniform_intervention",
        "training_distribution": "equal_natural_and_uniform_intervention",
        "coupling": float(coupling),
        "seeds": [int(seed) for seed in seeds],
        "estimator": estimator,
        "transport_map": _transport_map_config() if estimator == "transport" else None,
        "target": "instantaneous_vector_field_joint_target",
        "target_relation": "theta1+theta2->dtheta",
        "joint_target": ["dtheta1", "dtheta2"],
        "equation_parameters": {
            "omega1": 1.0,
            "omega2": 0.9,
            "A": KURAMOTO_PHASE_POTENTIAL_STRENGTH,
            "K": float(coupling),
        },
        "equation": spec.equation,
        "rows": rows,
        "summary": summary,
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_kuramoto_joint_target_peid(summary, figure_path)
    return {**result, "result_path": str(result_path)}


def run_kuramoto_joint_target_peid_sweep(
    *,
    mode: str = "full",
    couplings: Sequence[float] = KURAMOTO_PEID_DETAIL_COUPLINGS,
    seeds: Sequence[int] = (0, 1, 2, 3),
    result_path: Path = DEFAULT_RESULT_DIR / "kuramoto_joint_target_peid_sweep.json",
    figure_path: Path = DEFAULT_FIGURE_DIR / "kuramoto_joint_target_peid_sweep.png",
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    trajectory_samples = 260 if mode == "smoke" else 1000
    intervention_train_samples = 260 if mode == "smoke" else 1000
    readout_samples = 700 if mode == "smoke" else 2000
    epochs = 35 if mode == "smoke" else 180
    estimator = "transport"
    joint_targets = {"dtheta": ("dtheta1", "dtheta2")}
    rows: list[dict[str, object]] = []

    for coupling in couplings:
        spec = build_kuramoto_coupling_spec(float(coupling))
        for seed in seeds:
            seed = int(seed)
            natural_states, natural_targets = spec.simulate(seed=seed, samples=trajectory_samples, noise=0.0)
            train_interventions = _sample_intervention_states(
                spec, samples=intervention_train_samples, seed=seed + 991
            )
            train_intervention_targets = spec.vector_field(train_interventions)
            train_states = np.vstack([natural_states, train_interventions])
            train_targets = np.vstack([natural_targets, train_intervention_targets])
            fitted = fit_mlp(train_states, train_targets, seed=seed + 300, epochs=epochs)
            readout_states = _sample_intervention_states(
                spec,
                samples=readout_samples,
                seed=seed + 1991 + int(round(float(coupling) * 10000)),
            )
            oracle_targets = spec.vector_field(readout_states)
            mlp_targets = fitted.predict(readout_states)
            oracle = estimate_peid_for_joint_targets_from_samples(
                spec,
                readout_states,
                oracle_targets,
                joint_targets=joint_targets,
                estimator=estimator,
            )
            learned = estimate_peid_for_joint_targets_from_samples(
                spec,
                readout_states,
                mlp_targets,
                joint_targets=joint_targets,
                estimator=estimator,
            )
            oracle_row = oracle["hyperedges"].set_index(["sources", "target"]).loc[("theta1+theta2", "dtheta")]
            learned_row = learned["hyperedges"].set_index(["sources", "target"]).loc[("theta1+theta2", "dtheta")]
            rows.append(
                {
                    "coupling": float(coupling),
                    "seed": seed,
                    "mlp_syn": float(learned_row["score"]),
                    "mlp_joint_ei": float(learned_row["joint_ei"]),
                    "mlp_single_ei_sum": float(learned_row["single_ei_sum"]),
                    "oracle_syn": float(oracle_row["score"]),
                    "oracle_joint_ei": float(oracle_row["joint_ei"]),
                    "oracle_single_ei_sum": float(oracle_row["single_ei_sum"]),
                    "mlp_test_mse": float(fitted.train_mse),
                    "train_state_digest": _digest(train_states),
                    "readout_state_digest": _digest(readout_states),
                    "mlp_target_digest": _digest(mlp_targets),
                    "oracle_target_digest": _digest(oracle_targets),
                }
            )

    summary = _aggregate_kuramoto_joint_target_sweep_rows(rows)
    result = {
        "mode": mode,
        "system": "kuramoto_phase_coupling_joint_target_sweep",
        "sampling_distribution": "independent_uniform_intervention",
        "training_distribution": "equal_natural_and_uniform_intervention",
        "couplings": [float(coupling) for coupling in couplings],
        "seeds": [int(seed) for seed in seeds],
        "estimator": estimator,
        "transport_map": _transport_map_config() if estimator == "transport" else None,
        "target": "instantaneous_vector_field_joint_target",
        "target_relation": "theta1+theta2->dtheta",
        "joint_target": ["dtheta1", "dtheta2"],
        "equation_parameters": {
            "omega1": 1.0,
            "omega2": 0.9,
            "A": KURAMOTO_PHASE_POTENTIAL_STRENGTH,
        },
        "nonmonotonic_diagnostic": _nonmonotonic_syn_diagnostic(summary),
        "rows": rows,
        "summary": summary,
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_kuramoto_joint_target_peid_sweep(summary, figure_path)
    return {**result, "result_path": str(result_path)}


def run_kuramoto_phase_response_peid_sweep(
    *,
    mode: str = "full",
    couplings: Sequence[float] = KURAMOTO_PEID_DETAIL_COUPLINGS,
    seeds: Sequence[int] = (0, 1, 2, 3),
    tau: float = 4.0,
    result_path: Path = DEFAULT_RESULT_DIR / "kuramoto_phase_response_peid_sweep.json",
    figure_path: Path = DEFAULT_FIGURE_DIR / "kuramoto_phase_response_peid_sweep.png",
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    natural_trajectories = 4 if mode == "smoke" else 12
    samples_per_trajectory = 65 if mode == "smoke" else 120
    natural_burnin_steps = 1200 if mode == "smoke" else 2400
    intervention_train_samples = 260 if mode == "smoke" else 1200
    readout_samples = 700 if mode == "smoke" else 2000
    epochs = 45 if mode == "smoke" else 220
    estimator = "transport"
    phase_velocity_noise = 0.01
    rows: list[dict[str, object]] = []

    for coupling in couplings:
        spec = build_kuramoto_coupling_spec(float(coupling))
        for seed in seeds:
            seed = int(seed)
            natural_states, _ = simulate_natural_trajectory_pool(
                spec,
                seed=seed,
                trajectories=natural_trajectories,
                samples_per_trajectory=samples_per_trajectory,
                burnin_steps=natural_burnin_steps,
                noise=phase_velocity_noise,
            )
            natural_next = simulate_finite_time_next_states(
                spec,
                natural_states,
                tau=tau,
                process_noise=0.0,
                seed=seed + 1100,
            )
            natural_targets = _kuramoto_phase_response_targets(natural_next)
            train_interventions = _sample_intervention_states(
                spec,
                samples=intervention_train_samples,
                seed=seed + 991 + int(round(float(coupling) * 10000)),
            )
            train_intervention_next = simulate_finite_time_next_states(
                spec,
                train_interventions,
                tau=tau,
                process_noise=0.0,
                seed=seed + 1200,
            )
            train_states = np.vstack([natural_states, train_interventions])
            train_targets = np.vstack([natural_targets, _kuramoto_phase_response_targets(train_intervention_next)])
            fitted = fit_mlp(train_states, train_targets, seed=seed + 300, epochs=epochs)

            readout_states = _sample_intervention_states(
                spec,
                samples=readout_samples,
                seed=seed + 1991 + int(round(float(coupling) * 10000)),
            )
            oracle_next = simulate_finite_time_next_states(
                spec,
                readout_states,
                tau=tau,
                process_noise=0.0,
                seed=seed + 1300,
            )
            oracle_targets = _kuramoto_phase_response_targets(oracle_next)
            mlp_targets = fitted.predict(readout_states)
            oracle_values = _transport_synergy(readout_states[:, [0]], readout_states[:, [1]], oracle_targets)
            mlp_values = _transport_synergy(readout_states[:, [0]], readout_states[:, [1]], mlp_targets)
            natural_pred = fitted.predict(natural_states)
            rows.append(
                {
                    "coupling": float(coupling),
                    "seed": seed,
                    "natural_plv": _phase_locking_value(natural_states),
                    "natural_order": _kuramoto_order_excess(natural_states),
                    "natural_order_raw": _kuramoto_order_parameter(natural_states),
                    "mlp_syn": float(mlp_values["syn"]),
                    "mlp_joint_ei": float(mlp_values["joint_ei"]),
                    "mlp_single_ei_sum": float(mlp_values["left_ei"] + mlp_values["right_ei"]),
                    "oracle_syn": float(oracle_values["syn"]),
                    "oracle_joint_ei": float(oracle_values["joint_ei"]),
                    "oracle_single_ei_sum": float(oracle_values["left_ei"] + oracle_values["right_ei"]),
                    "mlp_test_mse": float(np.mean((natural_pred - natural_targets) ** 2)),
                    "train_state_digest": _digest(train_states),
                    "readout_state_digest": _digest(readout_states),
                    "natural_target_digest": _digest(natural_targets),
                    "mlp_target_digest": _digest(mlp_targets),
                    "oracle_target_digest": _digest(oracle_targets),
                }
            )

    summary = _aggregate_kuramoto_phase_response_rows(rows)
    diagnostic = _phase_response_criticality_diagnostic(summary)
    result = {
        "mode": mode,
        "system": "kuramoto_phase_response_peid_sweep",
        "target": "finite_time_phase_locking_response",
        "tau": float(tau),
        "sampling_distribution": "independent_uniform_intervention_for_peid_readout",
        "training_distribution": "natural_trajectory_plus_uniform_intervention",
        "couplings": [float(coupling) for coupling in couplings],
        "seeds": [int(seed) for seed in seeds],
        "estimator": estimator,
        "transport_map": _transport_map_config() if estimator == "transport" else None,
        "phase_response_target": ["cos_delta_tau", "sin_delta_tau", "order_excess_tau"],
        "equation_parameters": {
            "omega1": 1.0,
            "omega2": 0.9,
            "A": KURAMOTO_PHASE_POTENTIAL_STRENGTH,
        },
        "natural_trajectory_protocol": {
            "trajectories": natural_trajectories,
            "samples_per_trajectory": samples_per_trajectory,
            "burnin_steps": natural_burnin_steps,
            "phase_velocity_noise_std": phase_velocity_noise,
        },
        "criticality_diagnostic": diagnostic,
        "rows": rows,
        "summary": summary,
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_kuramoto_phase_response_peid_sweep(summary, diagnostic, figure_path)
    return {**result, "result_path": str(result_path)}


def run_large_kuramoto_phi_sweep(
    *,
    mode: str = "full",
    oscillator_count: int = 64,
    couplings: Sequence[float] = LARGE_KURAMOTO_COUPLINGS,
    seeds: Sequence[int] = (0, 1, 2),
    tau: float = 4.0,
    frequency_sigma: float = 1.0,
    result_path: Path = DEFAULT_RESULT_DIR / "large_kuramoto_phi_sweep.json",
    figure_path: Path = DEFAULT_FIGURE_DIR / "large_kuramoto_phi_sweep.png",
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    if oscillator_count % 2 != 0:
        raise ValueError("oscillator_count must be even so the macro partition is balanced.")
    readout_samples = 360 if mode == "smoke" else 1400
    natural_samples = 220 if mode == "smoke" else 900
    dt = 0.05 if mode == "smoke" else 0.04
    critical_coupling = _classic_kuramoto_critical_coupling(frequency_sigma)
    rows: list[dict[str, object]] = []

    for coupling in couplings:
        for seed in seeds:
            seed = int(seed)
            frequency_seed = seed + 50017
            frequencies = _classic_kuramoto_frequencies(oscillator_count, frequency_seed, frequency_sigma)
            readout_rng = np.random.default_rng(seed + 61003 + int(round(float(coupling) * 1000)))
            readout_states = readout_rng.uniform(0.0, 2.0 * math.pi, size=(readout_samples, oscillator_count))
            readout_next = _classic_kuramoto_integrate(
                readout_states,
                frequencies,
                coupling=float(coupling),
                tau=tau,
                dt=dt,
            )
            left_source, right_source = _large_kuramoto_group_sources(readout_states)
            target = _large_kuramoto_order_targets(readout_next)
            phi_values = _transport_synergy(left_source, right_source, target)

            natural_rng = np.random.default_rng(seed + 71003 + int(round(float(coupling) * 1000)))
            natural_initial = natural_rng.uniform(0.0, 2.0 * math.pi, size=(natural_samples, oscillator_count))
            natural_next = _classic_kuramoto_integrate(
                natural_initial,
                frequencies,
                coupling=float(coupling),
                tau=tau,
                dt=dt,
            )
            natural_order_raw = _kuramoto_global_order(natural_next)
            natural_order = _large_kuramoto_order_excess(natural_next)
            rows.append(
                {
                    "coupling": float(coupling),
                    "seed": seed,
                    "frequency_seed": frequency_seed,
                    "natural_order": float(np.mean(natural_order)),
                    "natural_order_raw": float(np.mean(natural_order_raw)),
                    "phi_syn": float(phi_values["syn"]),
                    "phi_joint_ei": float(phi_values["joint_ei"]),
                    "phi_single_ei_sum": float(phi_values["left_ei"] + phi_values["right_ei"]),
                    "phi_left_ei": float(phi_values["left_ei"]),
                    "phi_right_ei": float(phi_values["right_ei"]),
                    "readout_state_digest": _digest(readout_states),
                    "target_digest": _digest(target),
                    "frequency_digest": _digest(frequencies),
                }
            )

    summary = _aggregate_large_kuramoto_phi_rows(rows)
    diagnostic = _large_kuramoto_phi_diagnostic(summary, critical_coupling=critical_coupling)
    midpoint = oscillator_count // 2
    result = {
        "mode": mode,
        "system": "classic_large_n_kuramoto_phi_sweep",
        "target": "finite_time_global_order_response",
        "source_partition": [f"oscillators_0_to_{midpoint - 1}", f"oscillators_{midpoint}_to_{oscillator_count - 1}"],
        "sampling_distribution": "independent_uniform_initial_phases",
        "oscillator_count": int(oscillator_count),
        "couplings": [float(coupling) for coupling in couplings],
        "seeds": [int(seed) for seed in seeds],
        "tau": float(tau),
        "dt": float(dt),
        "frequency_distribution": "zero-mean Gaussian, rescaled to requested sigma per seed",
        "frequency_sigma": float(frequency_sigma),
        "critical_coupling_theory": critical_coupling,
        "estimator": "transport",
        "transport_map": _transport_map_config(),
        "target_components": ["order_excess_tau"],
        "criticality_diagnostic": diagnostic,
        "rows": rows,
        "summary": summary,
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_large_kuramoto_phi_sweep(summary, diagnostic, figure_path)
    return {**result, "result_path": str(result_path)}


def run_sis_gate_sweep(
    *,
    mode: str = "full",
    betas: Sequence[float] = SIS_GATE_SWEEP_BETAS,
    seeds: Sequence[int] = (0, 1, 2, 3),
    result_path: Path = DEFAULT_RESULT_DIR / "sis_gate_synergy_sweep.json",
    figure_path: Path = DEFAULT_FIGURE_DIR / "sis_gate_synergy_sweep.png",
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    trajectories = 8 if mode == "smoke" else 20
    samples_per_trajectory = 100 if mode == "smoke" else 200
    peid_samples = 800 if mode == "smoke" else 1800
    shap_samples = 24 if mode == "smoke" else 72
    epochs = 100 if mode == "smoke" else 300
    peid_estimator = "transport"
    tau = 1.0
    rows: list[dict[str, object]] = []
    peid_seed_offset = 1991

    for beta in betas:
        spec = build_sis_gate_spec(float(beta))
        relation_key = [("w", "x", "x_tau")]
        for seed in seeds:
            seed = int(seed)
            train_states, _ = simulate_natural_trajectory_pool(
                spec,
                seed=seed + 1000,
                trajectories=trajectories,
                samples_per_trajectory=samples_per_trajectory,
                burnin_steps=0,
                noise=0.01,
            )
            train_targets = simulate_sis_gate_next_states(
                spec, train_states, tau=tau, process_noise=0.0, seed=seed + 1100
            )
            natural_states, _ = simulate_natural_trajectory_pool(
                spec,
                seed=seed + 2000,
                trajectories=trajectories,
                samples_per_trajectory=samples_per_trajectory,
                burnin_steps=0,
                noise=0.01,
            )
            natural_targets = simulate_sis_gate_next_states(
                spec, natural_states, tau=tau, process_noise=0.0, seed=seed + 2100
            )
            fitted = fit_mlp(train_states, train_targets, seed=seed + 300, epochs=epochs)
            shap = estimate_shap_readout(fitted, natural_states, spec, samples=shap_samples, seed=seed + 400)
            natural_learned_targets = fitted.predict(natural_states)
            peid_states = _sample_intervention_states(
                spec,
                samples=peid_samples,
                seed=peid_seed_offset + int(round(float(beta) * 10000)),
            )
            peid_targets = fitted.predict(peid_states)
            learned = estimate_peid_from_samples(
                spec,
                peid_states,
                peid_targets,
                estimator=peid_estimator,
            )
            observational = observational_wms_surd(natural_states, natural_targets, spec, estimator=peid_estimator, seed=seed + 6000)
            readouts = _zero_control_synergy_readouts(
                inactive=np.isclose(float(beta), 0.0),
                wms=_mean_truth_hyperedge_score(observational, relation_key, column="wms"),
                surd_synergy=_mean_truth_hyperedge_score(observational, relation_key, column="synergy"),
                shap_interaction=_mean_truth_hyperedge_score(shap["interactions"], relation_key),
                peid_synergy=_mean_truth_hyperedge_score(learned["hyperedges"], relation_key),
            )
            rows.append(
                {
                    "beta": float(beta),
                    "seed": seed,
                    **readouts,
                    "mlp_test_mse": float(np.mean((natural_learned_targets - natural_targets) ** 2)),
                    "mlp_baseline_mse": float(fitted.baseline_mse),
                    "train_state_digest": _digest(train_states),
                    "readout_state_digest": _digest(natural_states),
                    "peid_readout_state_digest": _digest(peid_states),
                    "observed_target_digest": _digest(natural_targets),
                    "mlp_target_digest": _digest(natural_learned_targets),
                    "peid_target_digest": _digest(peid_targets),
                }
            )

    summary = _aggregate_sis_gate_rows(rows)
    result = {
        "mode": mode,
        "system": "sis_gate_next_state",
        "tau": tau,
        "seeds": [int(seed) for seed in seeds],
        "betas": [float(beta) for beta in betas],
        "estimator": peid_estimator,
        "peid_estimator": peid_estimator,
        "transport_map": _transport_map_config() if peid_estimator == "transport" else None,
        "training_distribution": "multi_initial_condition_natural_trajectory_pool",
        "natural_readout_state_distribution": "held_out_multi_initial_condition_natural_trajectory_pool",
        "peid_readout_state_distribution": "independent_uniform_intervention",
        "shared_readout_state_distribution": "natural_trajectory_for_wms_surd_shap",
        "peid_target_distribution": "mlp_predicted_next_state_on_independent_intervention_states",
        "method_data_contract": _method_data_contract(),
        "truth_hyperedges": ["w+x->x_tau"],
        "zero_control": {
            "parameter": "beta",
            "value": 0.0,
            "reported_readouts": "estimated_zero_point_residuals",
            "raw_fields": ["raw_wms", "raw_surd_synergy", "raw_shap_interaction", "raw_peid_synergy"],
            "reason": "At beta=0 the w-dependent infection gate is absent; reported readouts still come from the same fitted-model and transport-map pipeline, while raw_* fields duplicate them for auditability.",
        },
        "fairness": "For each beta and seed, WMS/SURD and SHAP use the same held-out natural readout states; MLP+PEID uses independent uniform intervention states and the same MLP trained only on a separate multi-initial-condition natural pool.",
        "target": "finite_time_next_state",
        "rows": rows,
        "summary": summary,
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_sis_gate_sweep(summary, figure_path)
    return {**result, "result_path": str(result_path)}


def run_lorenz_rho_sweep(
    *,
    mode: str = "full",
    rhos: Sequence[float] = LORENZ_RHO_VALUES,
    seeds: Sequence[int] = (0, 1, 2, 3),
    result_path: Path = DEFAULT_RESULT_DIR / "lorenz_rho_synergy_sweep.json",
    figure_path: Path = DEFAULT_FIGURE_DIR / "lorenz_rho_synergy_sweep.png",
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    trajectories = 4 if mode == "smoke" else 12
    samples_per_trajectory = 70 if mode == "smoke" else 150
    peid_samples = trajectories * samples_per_trajectory
    shap_samples = 24 if mode == "smoke" else 72
    epochs = 35 if mode == "smoke" else 180
    estimator = "transport"
    tau = 0.05
    rows: list[dict[str, object]] = []

    for rho in rhos:
        spec = build_lorenz_rho_spec(float(rho))
        relation_key = [("x", "y", "z_tau")]
        for seed in seeds:
            seed = int(seed)
            train_states, _ = simulate_natural_trajectory_pool(
                spec,
                seed=seed + 1000,
                trajectories=trajectories,
                samples_per_trajectory=samples_per_trajectory,
                burnin_steps=20,
                noise=0.0,
            )
            train_targets = simulate_finite_time_next_states(
                spec, train_states, tau=tau, process_noise=0.0, seed=seed + 1100
            )
            natural_states, _ = simulate_natural_trajectory_pool(
                spec,
                seed=seed + 2000,
                trajectories=trajectories,
                samples_per_trajectory=samples_per_trajectory,
                burnin_steps=20,
                noise=0.0,
            )
            natural_targets = simulate_finite_time_next_states(
                spec, natural_states, tau=tau, process_noise=0.0, seed=seed + 2100
            )
            fitted = fit_mlp(train_states, train_targets, seed=seed + 300, epochs=epochs)
            shap = estimate_shap_readout(fitted, natural_states, spec, samples=shap_samples, seed=seed + 400)
            learned_targets = fitted.predict(natural_states)
            peid_states = _sample_intervention_states(
                spec,
                samples=peid_samples,
                seed=5000 + int(round(float(rho) * 100)),
            )
            peid_targets = fitted.predict(peid_states)
            learned = estimate_peid_from_samples(spec, peid_states, peid_targets, estimator=estimator)
            observational = observational_wms_surd(natural_states, natural_targets, spec, estimator=estimator, seed=seed + 6000)
            rows.append(
                {
                    "rho": float(rho),
                    "seed": seed,
                    "wms": _mean_truth_hyperedge_score(observational, relation_key, column="wms"),
                    "surd_synergy": _mean_truth_hyperedge_score(observational, relation_key, column="synergy"),
                    "shap_interaction": _mean_truth_hyperedge_score(shap["interactions"], relation_key),
                    "peid_synergy": _mean_truth_hyperedge_score(learned["hyperedges"], relation_key),
                    "mlp_test_mse": float(np.mean((learned_targets - natural_targets) ** 2)),
                    "mlp_baseline_mse": float(fitted.baseline_mse),
                    "train_state_digest": _digest(train_states),
                    "readout_state_digest": _digest(natural_states),
                    "peid_readout_state_digest": _digest(peid_states),
                    "observed_target_digest": _digest(natural_targets),
                    "mlp_target_digest": _digest(learned_targets),
                    "peid_target_digest": _digest(peid_targets),
                }
            )

    summary = _aggregate_lorenz_rho_rows(rows)
    result = {
        "mode": mode,
        "system": "lorenz3d_next_state",
        "tau": tau,
        "seeds": [int(seed) for seed in seeds],
        "rhos": [float(rho) for rho in rhos],
        "estimator": estimator,
        "transport_map": _transport_map_config() if estimator == "transport" else None,
        "training_distribution": "multi_initial_condition_natural_trajectory_pool",
        "shared_readout_state_distribution": "held_out_multi_initial_condition_natural_trajectory_pool",
        "peid_readout_state_distribution": "independent_uniform_intervention",
        "peid_target_distribution": "mlp_predicted_next_state_on_independent_intervention_states",
        "method_data_contract": _method_data_contract(),
        "truth_hyperedges": ["x+y->z_tau"],
        "fairness": "For each rho and seed, WMS/SURD and SHAP use the same held-out natural readout states; MLP+PEID uses independent uniform intervention states and the same MLP trained only on a separate multi-initial-condition natural pool.",
        "target": "finite_time_next_state",
        "rows": rows,
        "summary": summary,
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_lorenz_rho_sweep(summary, figure_path)
    return {**result, "result_path": str(result_path)}


def run_lorenz_uniform_tau_sweep(
    *,
    mode: str = "full",
    rhos: Sequence[float] = LORENZ_RHO_VALUES,
    taus: Sequence[float] = LORENZ_UNIFORM_TAU_VALUES,
    seeds: Sequence[int] = (0, 1, 2, 3),
    result_path: Path = DEFAULT_RESULT_DIR / "lorenz_uniform_tau_synergy_sweep.json",
    figure_path: Path = DEFAULT_FIGURE_DIR / "lorenz_uniform_tau_best_synergy.png",
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    train_samples = 260 if mode == "smoke" else 1400
    readout_samples = 260 if mode == "smoke" else 1400
    peid_samples = 260 if mode == "smoke" else 1400
    shap_samples = 24 if mode == "smoke" else 72
    epochs = 35 if mode == "smoke" else 180
    estimator = "transport"
    rows: list[dict[str, object]] = []

    for tau in taus:
        tau = float(tau)
        for rho in rhos:
            spec = build_lorenz_rho_spec(float(rho))
            relation_key = list(spec.truth_hyperedges)
            for seed in seeds:
                seed = int(seed)
                base_seed = seed + int(round(10000 * tau)) + int(round(100 * float(rho)))
                train_states = _sample_uniform_states(spec, samples=train_samples, seed=base_seed + 1100)
                train_targets = simulate_finite_time_next_states(
                    spec, train_states, tau=tau, process_noise=0.0, seed=base_seed + 1200
                )
                readout_states = _sample_uniform_states(spec, samples=readout_samples, seed=base_seed + 2100)
                readout_targets = simulate_finite_time_next_states(
                    spec, readout_states, tau=tau, process_noise=0.0, seed=base_seed + 2200
                )
                fitted = fit_mlp(train_states, train_targets, seed=base_seed + 3000, epochs=epochs)
                shap = estimate_shap_readout(fitted, readout_states, spec, samples=shap_samples, seed=base_seed + 4000)
                learned_targets = fitted.predict(readout_states)
                learned = estimate_peid_from_samples(
                    spec,
                    readout_states[:peid_samples],
                    learned_targets[:peid_samples],
                    estimator=estimator,
                )
                observational = observational_wms_surd(readout_states, readout_targets, spec, estimator=estimator, seed=base_seed + 6000)
                rows.append(
                    {
                        "tau": tau,
                        "rho": float(rho),
                        "seed": seed,
                        "wms": _mean_truth_hyperedge_score(observational, relation_key, column="wms"),
                        "surd_synergy": _mean_truth_hyperedge_score(observational, relation_key, column="synergy"),
                        "shap_interaction": _mean_truth_hyperedge_score(shap["interactions"], relation_key),
                        "peid_synergy": _mean_truth_hyperedge_score(learned["hyperedges"], relation_key),
                        "mlp_test_mse": float(fitted.train_mse),
                        "mlp_baseline_mse": float(fitted.baseline_mse),
                        "train_state_digest": _digest(train_states),
                        "readout_state_digest": _digest(readout_states),
                        "observed_target_digest": _digest(readout_targets),
                        "mlp_target_digest": _digest(learned_targets),
                    }
                )

    summary_by_tau_rho = _aggregate_lorenz_uniform_tau_rows(rows)
    selected_tau, tau_scores = _select_lorenz_uniform_tau(summary_by_tau_rho)
    summary = [
        row
        for row in summary_by_tau_rho
        if np.isclose(float(row["tau"]), selected_tau)
    ]
    result = {
        "mode": mode,
        "system": "lorenz3d_uniform_tau_next_state",
        "sampling_distribution": "independent_uniform",
        "seeds": [int(seed) for seed in seeds],
        "rhos": [float(rho) for rho in rhos],
        "taus": [float(tau) for tau in taus],
        "selected_tau": selected_tau,
        "tau_selection_rule": "minimize PEID relative range across rho, with penalties for seed variance and nonpositive values",
        "tau_scores": tau_scores,
        "estimator": estimator,
        "transport_map": _transport_map_config() if estimator == "transport" else None,
        "training_distribution": "independent_uniform",
        "shared_readout_state_distribution": "independent_uniform",
        "peid_target_distribution": "mlp_predicted_next_state_on_shared_uniform_states",
        "truth_hyperedges": ["x+z->y_tau", "x+y->z_tau"],
        "fairness": "For each rho, tau, and seed, WMS/SURD, SHAP, and MLP+PEID use the same independent-uniform readout states; SHAP and PEID use the same fitted MLP trained only on independent-uniform initial states.",
        "target": "finite_time_next_state",
        "rows": rows,
        "summary_by_tau_rho": summary_by_tau_rho,
        "summary": summary,
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_lorenz_rho_sweep(summary, figure_path)
    return {**result, "result_path": str(result_path)}


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _plot_panel(
    axis: object,
    summary: Sequence[dict[str, object]],
    *,
    parameter_key: str,
    xlabel: str,
    label: str,
    symlog_linthresh: float | None = None,
    include_oracle_peid: bool = True,
    separate_surd_axis: bool = False,
) -> None:
    x_values = np.asarray([float(row[parameter_key]) for row in summary], dtype=float)
    specs = _available_method_plot_specs(summary, include_oracle_peid=include_oracle_peid)
    surd_axis = axis.twinx() if separate_surd_axis else axis
    for key, method_label, color, marker in specs:
        plot_axis = surd_axis if separate_surd_axis and key == "surd_synergy" else axis
        mean = np.asarray([float(row[f"{key}_mean"]) for row in summary], dtype=float)
        std = np.asarray([float(row[f"{key}_std"]) for row in summary], dtype=float)
        plot_axis.plot(x_values, mean, marker=marker, linewidth=1.6, markersize=4.0, label=method_label, color=color)
        plot_axis.fill_between(x_values, mean - std, mean + std, color=color, alpha=0.13, linewidth=0)
    axis.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    axis.set_xlabel(xlabel)
    axis.set_title(label, loc="left", fontsize=9, fontweight="bold")
    axis.grid(True, axis="y", alpha=0.20, linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if separate_surd_axis:
        surd_axis.set_ylabel("SURD synergy (bits)", color="#E3A13D")
        surd_axis.tick_params(axis="y", colors="#E3A13D")
        surd_axis.spines["top"].set_visible(False)
        surd_axis.spines["right"].set_color("#E3A13D")
    if symlog_linthresh is not None:
        axis.set_yscale("symlog", linthresh=float(symlog_linthresh))
        axis.text(
            0.98,
            0.03,
            "symlog y",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#666666",
        )


def _merge_mmi_pid_summary(
    summary: Sequence[dict[str, object]],
    mmi_payload: Mapping[str, object],
    *,
    system_key: str,
    parameter_key: str,
) -> list[dict[str, object]]:
    systems = mmi_payload.get("systems")
    if not isinstance(systems, Mapping) or system_key not in systems:
        raise KeyError(f"Missing MMI-PID summary for {system_key}.")
    mmi_system = systems[system_key]
    if not isinstance(mmi_system, Mapping):
        raise TypeError(f"MMI-PID summary for {system_key} must be a mapping.")
    mmi_parameter_key = str(mmi_system.get("parameter_key", parameter_key))
    mmi_rows = list(mmi_system.get("summary", []))  # type: ignore[arg-type]
    if len(mmi_rows) != len(summary):
        raise ValueError(f"MMI-PID summary length mismatch for {system_key}.")
    merged: list[dict[str, object]] = []
    for row, mmi_row in zip(summary, mmi_rows):
        parameter_value = float(row[parameter_key])
        mmi_parameter_value = float(mmi_row[mmi_parameter_key])  # type: ignore[index]
        if not np.isclose(parameter_value, mmi_parameter_value):
            raise ValueError(
                f"MMI-PID parameter mismatch for {system_key}: {parameter_value} != {mmi_parameter_value}."
            )
        updated = dict(row)
        updated["mmi_pid_synergy_mean"] = float(mmi_row["mmi_pid_synergy_mean"])  # type: ignore[index]
        updated["mmi_pid_synergy_std"] = float(mmi_row["mmi_pid_synergy_std"])  # type: ignore[index]
        merged.append(updated)
    return merged


def _method_count(summary: Sequence[dict[str, object]]) -> int:
    return len(_available_method_plot_specs(summary, include_oracle_peid=False))


def run_part1_combined_synergy_figure(
    *,
    standard_result_path: Path = ROOT / "results" / "coupled_standard_map_method_comparison" / "part1_four_method_synergy.json",
    wilson_cowan_refractory_result_path: Path = ROOT / "results" / "discrete_iteration_dynamics_benchmark" / "wilson_cowan_refractory_synergy_sweep.json",
    kuramoto_result_path: Path = ROOT / "results" / "classic_network_dynamics_benchmark" / "kuramoto_coupling_synergy_sweep.json",
    controlled_henon_result_path: Path = ROOT / "results" / "henon_unique_five_method_synergy" / "summary.json",
    coupled_henon_result_path: Path | None = None,
    ikeda_result_path: Path = ROOT / "results" / "discrete_iteration_dynamics_benchmark" / "ikeda_y_tau_synergy_sweep.json",
    nicholson_bailey_result_path: Path = ROOT / "results" / "discrete_iteration_dynamics_benchmark" / "nicholson_bailey_synergy_sweep.json",
    mmi_pid_result_path: Path = ROOT / "results" / "part1_mmi_pid_synergy_report" / "summary.json",
    figure_path: Path = PART1_COMBINED_FIGURE_PATH,
) -> dict[str, object]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    standard = _load_json(standard_result_path)
    wilson_cowan_refractory = _load_json(wilson_cowan_refractory_result_path)
    kuramoto = _load_json(kuramoto_result_path)
    controlled_henon = _load_json(controlled_henon_result_path)
    ikeda = _load_json(ikeda_result_path)
    nicholson_bailey = _load_json(nicholson_bailey_result_path)
    mmi_pid = _load_json(mmi_pid_result_path)
    standard_summary = _merge_mmi_pid_summary(
        standard["summary"],
        mmi_pid,
        system_key="standard_map",
        parameter_key="coupling",
    )
    wilson_cowan_refractory_parameter_key = str(wilson_cowan_refractory.get("parameter_key", "gain"))
    wilson_cowan_refractory_summary = _merge_mmi_pid_summary(
        wilson_cowan_refractory["summary"],
        mmi_pid,
        system_key="wilson_cowan_refractory",
        parameter_key=wilson_cowan_refractory_parameter_key,
    )
    kuramoto_parameter_key = str(kuramoto.get("parameter_key", "coupling"))
    kuramoto_summary = _merge_mmi_pid_summary(
        kuramoto["summary"],
        mmi_pid,
        system_key="kuramoto",
        parameter_key=kuramoto_parameter_key,
    )
    ikeda_parameter_key = str(ikeda.get("parameter_key", "u"))
    ikeda_summary = _merge_mmi_pid_summary(
        ikeda["summary"],
        mmi_pid,
        system_key="ikeda_y_tau",
        parameter_key=ikeda_parameter_key,
    )
    nicholson_bailey_parameter_key = str(nicholson_bailey.get("parameter_key", "a"))
    nicholson_bailey_summary = _merge_mmi_pid_summary(
        nicholson_bailey["summary"],
        mmi_pid,
        system_key="nicholson_bailey",
        parameter_key=nicholson_bailey_parameter_key,
    )
    controlled_henon_parameter_key = str(controlled_henon.get("parameter_key", "gamma"))
    controlled_henon_summary = list(controlled_henon["summary"])
    controlled_henon_xlabel = (
        "Hénon control parameter lambda"
        if controlled_henon_parameter_key == "lambda"
        else "Hénon unique channel gamma"
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 7.2), constrained_layout=True)
    axes = axes.flat
    _plot_panel(
        axes[0],
        standard_summary,
        parameter_key="coupling",
        xlabel="Standard map coupling J",
        label="a  Coupled standard map",
        symlog_linthresh=0.2,
        include_oracle_peid=False,
    )
    axes[0].set_ylabel("Synergy / Interaction")
    _plot_panel(
        axes[1],
        wilson_cowan_refractory_summary,
        parameter_key=wilson_cowan_refractory_parameter_key,
        xlabel="Wilson-Cowan sigmoid gain g",
        label="b  Wilson-Cowan gain",
        include_oracle_peid=False,
    )
    _plot_panel(
        axes[2],
        kuramoto_summary,
        parameter_key=kuramoto_parameter_key,
        xlabel="Kuramoto coupling K",
        label="c  Kuramoto phase locking",
        include_oracle_peid=False,
        separate_surd_axis=True,
    )
    _plot_panel(
        axes[3],
        controlled_henon_summary,
        parameter_key=controlled_henon_parameter_key,
        xlabel=controlled_henon_xlabel,
        label="d  Controlled Hénon unique sweep",
        include_oracle_peid=False,
    )
    axes[3].set_ylabel("Synergy / Interaction")
    _plot_panel(
        axes[4],
        ikeda_summary,
        parameter_key=ikeda_parameter_key,
        xlabel="Ikeda parameter u",
        label="e  Ikeda optical cavity",
        include_oracle_peid=False,
    )
    _plot_panel(
        axes[5],
        nicholson_bailey_summary,
        parameter_key=nicholson_bailey_parameter_key,
        xlabel="Nicholson-Bailey attack rate a",
        label="f  Nicholson-Bailey",
        include_oracle_peid=False,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    result = {
        "figure_path": str(figure_path),
        "y_axis_label": "Synergy / Interaction",
        "panels": {
            "standard_map": str(standard_result_path),
            "wilson_cowan_refractory": str(wilson_cowan_refractory_result_path),
            "kuramoto_phase_coupling": str(kuramoto_result_path),
            "controlled_henon_unique_information": str(controlled_henon_result_path),
            "ikeda_y_tau": str(ikeda_result_path),
            "nicholson_bailey": str(nicholson_bailey_result_path),
        },
        "mmi_pid_result_path": str(mmi_pid_result_path),
        "legacy_coupled_henon_result_path": str(coupled_henon_result_path) if coupled_henon_result_path else None,
        "panel_method_counts": {
            "standard_map": _method_count(standard_summary),
            "wilson_cowan_refractory": _method_count(wilson_cowan_refractory_summary),
            "kuramoto_phase_coupling": _method_count(kuramoto_summary),
            "controlled_henon_unique_information": _method_count(controlled_henon_summary),
            "ikeda_y_tau": _method_count(ikeda_summary),
            "nicholson_bailey": _method_count(nicholson_bailey_summary),
        },
        "panel_parameter_keys": {
            "standard_map": "coupling",
            "wilson_cowan_refractory": wilson_cowan_refractory_parameter_key,
            "kuramoto_phase_coupling": kuramoto_parameter_key,
            "controlled_henon_unique_information": controlled_henon_parameter_key,
            "ikeda_y_tau": ikeda_parameter_key,
            "nicholson_bailey": nicholson_bailey_parameter_key,
        },
        "panel_xlabels": {
            "standard_map": "Standard map coupling J",
            "wilson_cowan_refractory": "Wilson-Cowan sigmoid gain g",
            "kuramoto_phase_coupling": "Kuramoto coupling K",
            "controlled_henon_unique_information": controlled_henon_xlabel,
            "ikeda_y_tau": "Ikeda parameter u",
            "nicholson_bailey": "Nicholson-Bailey attack rate a",
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--kuramoto-k", type=float, default=0.2, help="Kuramoto phase coupling K for single-example runs.")
    parser.add_argument("--kuramoto-tau", type=float, default=4.0, help="Finite-time horizon for Kuramoto phase-response runs.")
    parser.add_argument("--kuramoto-n", type=int, default=64, help="Oscillator count for large-N classic Kuramoto runs.")
    parser.add_argument("--kuramoto-coupling-sweep", action="store_true", help="Run only the Part1 Kuramoto coupling synergy sweep.")
    parser.add_argument("--kuramoto-joint-target-peid", action="store_true", help="Run the standalone Kuramoto PEID example with joint dtheta target.")
    parser.add_argument("--kuramoto-joint-target-peid-sweep", action="store_true", help="Run the standalone Kuramoto joint-target PEID coupling sweep.")
    parser.add_argument("--kuramoto-phase-response-peid-sweep", action="store_true", help="Run finite-time Kuramoto phase-locking response PEID sweep.")
    parser.add_argument("--large-kuramoto-phi-sweep", action="store_true", help="Run classic large-N Kuramoto whole-system Phi/Syn sweep.")
    parser.add_argument("--kuramoto-peid-detail-sweep", action="store_true", help="Run the dense Kuramoto Oracle/MLP PEID component sweep.")
    parser.add_argument("--sis-gate-sweep", action="store_true", help="Run only the Part1 SIS gate synergy sweep.")
    parser.add_argument("--lorenz-rho-sweep", action="store_true", help="Run only the Part1 Lorenz rho synergy sweep.")
    parser.add_argument("--lorenz-uniform-tau-sweep", action="store_true", help="Run the uniform-sampled Lorenz tau sweep.")
    parser.add_argument("--rossler-coupling-sweep", action="store_true", help="Run the natural-trajectory Rössler coupling sweep.")
    parser.add_argument("--wilson-cowan-gain-sweep", action="store_true", help="Run the natural-trajectory Wilson–Cowan gain sweep.")
    parser.add_argument("--ode-future-state-sweeps", action="store_true", help="Retarget derivative ODE panels to finite-time future states.")
    parser.add_argument("--part1-combined-figure", action="store_true", help="Build the Part1 six-panel comparison figure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kuramoto_coupling_sweep:
        result = run_kuramoto_coupling_sweep(
            mode=args.mode,
            seeds=tuple(args.seeds),
            result_path=args.result_dir / "kuramoto_coupling_synergy_sweep.json",
            figure_path=args.figure_dir / "kuramoto_coupling_synergy_sweep.png",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.kuramoto_joint_target_peid:
        result = run_kuramoto_joint_target_peid(
            mode=args.mode,
            coupling=args.kuramoto_k,
            seeds=tuple(args.seeds),
            result_path=args.result_dir / "kuramoto_joint_target_peid.json",
            figure_path=args.figure_dir / "kuramoto_joint_target_peid.png",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.kuramoto_joint_target_peid_sweep:
        result = run_kuramoto_joint_target_peid_sweep(
            mode=args.mode,
            seeds=tuple(args.seeds),
            result_path=args.result_dir / "kuramoto_joint_target_peid_sweep.json",
            figure_path=args.figure_dir / "kuramoto_joint_target_peid_sweep.png",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.kuramoto_phase_response_peid_sweep:
        result = run_kuramoto_phase_response_peid_sweep(
            mode=args.mode,
            seeds=tuple(args.seeds),
            tau=args.kuramoto_tau,
            result_path=args.result_dir / "kuramoto_phase_response_peid_sweep.json",
            figure_path=args.figure_dir / "kuramoto_phase_response_peid_sweep.png",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.large_kuramoto_phi_sweep:
        result = run_large_kuramoto_phi_sweep(
            mode=args.mode,
            oscillator_count=args.kuramoto_n,
            seeds=tuple(args.seeds),
            tau=args.kuramoto_tau,
            result_path=args.result_dir / "large_kuramoto_phi_sweep.json",
            figure_path=args.figure_dir / "large_kuramoto_phi_sweep.png",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.kuramoto_peid_detail_sweep:
        result = run_kuramoto_peid_detail_sweep(
            mode=args.mode,
            seeds=tuple(args.seeds),
            result_path=args.result_dir / "kuramoto_peid_detail_sweep.json",
            figure_path=args.figure_dir / "kuramoto_peid_detail_sweep.png",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.sis_gate_sweep:
        result = run_sis_gate_sweep(
            mode=args.mode,
            seeds=tuple(args.seeds),
            result_path=args.result_dir / "sis_gate_synergy_sweep.json",
            figure_path=args.figure_dir / "sis_gate_synergy_sweep.png",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.lorenz_rho_sweep:
        result = run_lorenz_rho_sweep(
            mode=args.mode,
            seeds=tuple(args.seeds),
            result_path=args.result_dir / "lorenz_rho_synergy_sweep.json",
            figure_path=args.figure_dir / "lorenz_rho_synergy_sweep.png",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.lorenz_uniform_tau_sweep:
        result = run_lorenz_uniform_tau_sweep(
            mode=args.mode,
            seeds=tuple(args.seeds),
            result_path=args.result_dir / "lorenz_uniform_tau_synergy_sweep.json",
            figure_path=args.figure_dir / "lorenz_uniform_tau_best_synergy.png",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.rossler_coupling_sweep:
        result = run_coupled_rossler_coupling_sweep(
            mode=args.mode,
            seeds=tuple(args.seeds),
            result_path=args.result_dir / "rossler_coupling_synergy_sweep.json",
            figure_path=args.figure_dir / "rossler_coupling_synergy_sweep.png",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.wilson_cowan_gain_sweep:
        result = run_wilson_cowan_gain_sweep(
            mode=args.mode,
            seeds=tuple(args.seeds),
            result_path=args.result_dir / "wilson_cowan_gain_synergy_sweep.json",
            figure_path=args.figure_dir / "wilson_cowan_gain_synergy_sweep.png",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.ode_future_state_sweeps:
        result = run_ode_future_state_sweeps(
            mode=args.mode,
            seeds=tuple(args.seeds),
            result_dir=args.result_dir,
            figure_dir=args.figure_dir,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.part1_combined_figure:
        result = run_part1_combined_synergy_figure()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    result = run_benchmark(
        mode=args.mode,
        result_dir=args.result_dir,
        figure_dir=args.figure_dir,
        report_path=args.report_path,
        seeds=tuple(args.seeds),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
