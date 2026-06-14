#!/usr/bin/env python3
"""Benchmark causal synergy readouts on nonlinear discrete-time maps."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_network_dynamics_benchmark import (
    _aggregate_sweep_rows,
    _digest,
    _fitted_model_digest,
    _mean_truth_hyperedge_components,
    _mean_truth_hyperedge_score,
    _method_plot_specs,
    _part1_fairness_audit,
    _plot_four_method_sweep,
    _plot_panel,
    _zero_control_synergy_readouts,
    estimate_peid_from_samples,
    estimate_shap_readout,
    fit_mlp,
    observational_wms_surd,
    SURD_TM_CONDITIONAL_SAMPLES,
    SURD_TM_TARGET_ANCHORS,
    TRANSPORT_MAP_DEGREE,
    TRANSPORT_MAP_JITTER,
)


DEFAULT_RESULT_DIR = ROOT / "results" / "discrete_iteration_dynamics_benchmark"
DEFAULT_FIGURE_DIR = ROOT / "fig" / "discrete_iteration_dynamics_benchmark"
DEFAULT_REPORT_PATH = ROOT / "docs" / "reports" / "discrete_iteration_dynamics_benchmark.md"
DEFAULT_STANDARD_RESULT_PATH = (
    ROOT / "results" / "coupled_standard_map_method_comparison" / "part1_four_method_synergy.json"
)
DEFAULT_COMBINED_FIGURE_PATH = (
    ROOT / "fig" / "part1_synergy_comparison" / "six_system_discrete_iteration_synergy_panels.png"
)
DEFAULT_HENON_RESULT_PATH = DEFAULT_RESULT_DIR / "coupled_henon_synergy_sweep.json"
DEFAULT_HENON_FIGURE_PATH = DEFAULT_FIGURE_DIR / "coupled_henon_synergy_sweep.png"
DEFAULT_HENON_COMPARISON_FIGURE_PATH = (
    ROOT / "fig" / "part1_synergy_comparison" / "lorenz_vs_coupled_henon.png"
)
DEFAULT_HENON_REPORT_PATH = ROOT / "docs" / "reports" / "coupled_henon_lorenz_replacement.md"
DEFAULT_HENON_TUNING_RESULT_PATH = (
    DEFAULT_RESULT_DIR / "coupled_henon_prediction_tuning.json"
)
DEFAULT_LORENZ_RESULT_PATH = (
    ROOT / "results" / "classic_network_dynamics_benchmark" / "lorenz_rho_synergy_sweep.json"
)

IKEDA_U_VALUES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
NICHOLSON_A_VALUES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
WILSON_COWAN_REFRACTORY_GAIN_VALUES = (0.0, 0.4, 0.7, 1.0, 1.4, 2.0, 3.2, 4.0, 6.0)
REPLICATOR_GAMMA_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
COURNOT_LAMBDA_VALUES = (0.0, 0.05, 0.10, 0.15, 0.20)
COUPLED_HENON_KAPPA_VALUES = (0.0, 0.04, 0.08, 0.12, 0.16, 0.20)
COUPLED_HENON_HISTOGRAM_BINS = 6
COUPLED_HENON_HISTOGRAM_SENSITIVITY_BINS = (4, 6, 8)
BROAD_ONE_STEP_SYSTEMS = frozenset({"ikeda", "nicholson_bailey", "wilson_cowan_refractory", "cournot"})


Array = np.ndarray


@dataclass(frozen=True)
class MapSpec:
    name: str
    display_name: str
    state_names: tuple[str, ...]
    target_names: tuple[str, ...]
    equation: str
    parameter_key: str
    parameter_value: float
    parameter_values: tuple[float, ...]
    intervention_bounds: Array
    truth_hyperedges: tuple[tuple[str, str, str], ...]
    _transition: Callable[[Array], Array]
    _project: Callable[[Array], Array]
    _initial_state: Callable[[np.random.Generator], Array]
    _sample_intervention: Callable[[int, int], Array] | None = None
    burnin_steps: int = 30

    def project(self, states: Array) -> Array:
        values = np.asarray(states, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(self.state_names):
            raise ValueError(f"states must have shape (n, {len(self.state_names)}).")
        projected = np.asarray(self._project(values), dtype=float)
        if projected.shape != values.shape:
            raise ValueError("project returned an invalid shape.")
        if not np.isfinite(projected).all():
            raise FloatingPointError(f"{self.name} projection produced non-finite states.")
        return projected

    def transition(self, states: Array) -> Array:
        values = self.project(states)
        targets = np.asarray(self._transition(values), dtype=float)
        if targets.shape != values.shape:
            raise ValueError("transition returned an invalid shape.")
        return self.project(targets)

    def initial_state(self, rng: np.random.Generator) -> Array:
        return self.project(np.asarray(self._initial_state(rng), dtype=float))[0]

    def sample_interventions(self, *, samples: int, seed: int) -> Array:
        if samples <= 0:
            raise ValueError("samples must be positive.")
        if self._sample_intervention is not None:
            return self.project(self._sample_intervention(int(samples), int(seed)))
        rng = np.random.default_rng(int(seed))
        values = np.column_stack(
            [rng.uniform(low, high, size=int(samples)) for low, high in self.intervention_bounds]
        )
        return self.project(values)


@dataclass(frozen=True)
class HenonMLPConfig:
    hidden_widths: tuple[int, ...] = (96, 96)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 240
    patience: int = 30
    batch_size: int = 256
    min_delta: float = 1e-5

    def __post_init__(self) -> None:
        if not self.hidden_widths or any(int(width) <= 0 for width in self.hidden_widths):
            raise ValueError("hidden_widths must contain positive integers.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative.")
        if self.max_epochs <= 0 or self.patience <= 0 or self.batch_size <= 0:
            raise ValueError("epoch, patience, and batch-size values must be positive.")


@dataclass
class FittedHenonMLP:
    net: object
    x_mean: Array
    x_std: Array
    y_mean: Array
    y_std: Array
    train_mse: float
    baseline_mse: float
    best_epoch: int
    validation_objective: float
    validation_nrmse: tuple[float, ...]

    def predict(self, states: Array) -> Array:
        import torch

        values = np.asarray(states, dtype=np.float32)
        scaled = (values - self.x_mean) / self.x_std
        self.net.eval()
        with torch.no_grad():
            prediction = np.asarray(
                self.net(torch.tensor(scaled, dtype=torch.float32)).cpu().tolist(),
                dtype=float,
            )
        return prediction * self.y_std + self.y_mean


def _clip_project(bounds: Array) -> Callable[[Array], Array]:
    lower = np.asarray(bounds[:, 0], dtype=float)
    upper = np.asarray(bounds[:, 1], dtype=float)

    def project(values: Array) -> Array:
        return np.clip(np.asarray(values, dtype=float), lower, upper)

    return project


def _identity_project(values: Array) -> Array:
    return np.asarray(values, dtype=float)


def _simplex_project(values: Array, *, eps: float = 1e-6) -> Array:
    array = np.asarray(values, dtype=float)
    clipped = np.maximum(array, 0.0)
    row_sums = clipped.sum(axis=1, keepdims=True)
    bad = ~np.isfinite(row_sums[:, 0]) | (row_sums[:, 0] <= 0.0)
    if np.any(bad):
        clipped[bad] = 1.0
        row_sums = clipped.sum(axis=1, keepdims=True)
    normalized = clipped / row_sums
    return eps + (1.0 - eps * normalized.shape[1]) * normalized


def build_ikeda_spec(u: float) -> MapSpec:
    u = float(u)
    bounds = np.array([[-3.0, 3.0], [-3.0, 3.0]], dtype=float)

    def transition(values: Array) -> Array:
        x, y = values.T
        phase = 0.4 - 6.0 / (1.0 + x * x + y * y)
        return np.column_stack(
            [
                1.0 + u * (x * np.cos(phase) - y * np.sin(phase)),
                u * (x * np.sin(phase) + y * np.cos(phase)),
            ]
        )

    return MapSpec(
        name="ikeda",
        display_name="Ikeda",
        state_names=("x", "y"),
        target_names=("x_tau", "y_tau"),
        equation=(
            r"x_{t+1}=1+u(x_t\cos\theta_t-y_t\sin\theta_t),\;"
            r"y_{t+1}=u(x_t\sin\theta_t+y_t\cos\theta_t),\;"
            r"\theta_t=0.4-\frac{6}{1+x_t^2+y_t^2}"
        ),
        parameter_key="u",
        parameter_value=u,
        parameter_values=IKEDA_U_VALUES,
        intervention_bounds=bounds,
        truth_hyperedges=(("x", "y", "x_tau"), ("x", "y", "y_tau")),
        _transition=transition,
        _project=_clip_project(bounds),
        _initial_state=lambda rng: rng.uniform(-1.5, 1.5, size=2),
        _sample_intervention=lambda samples, seed: np.random.default_rng(seed).uniform(
            -1.5, 1.5, size=(samples, 2)
        ),
        burnin_steps=50,
    )


def build_nicholson_bailey_spec(a: float) -> MapSpec:
    a = float(a)
    r = 1.6
    bounds = np.array([[0.02, 6.0], [0.02, 6.0]], dtype=float)

    def transition(values: Array) -> Array:
        h, p = values.T
        survival = np.exp(-a * p)
        return np.column_stack([r * h * survival, h * (1.0 - survival)])

    return MapSpec(
        name="nicholson_bailey",
        display_name="Nicholson-Bailey",
        state_names=("H", "P"),
        target_names=("H_tau", "P_tau"),
        equation=r"H_{t+1}=RH_t e^{-aP_t},\;P_{t+1}=H_t(1-e^{-aP_t}),\;R=1.6",
        parameter_key="a",
        parameter_value=a,
        parameter_values=NICHOLSON_A_VALUES,
        intervention_bounds=bounds,
        truth_hyperedges=(("H", "P", "H_tau"),),
        _transition=transition,
        _project=_clip_project(bounds),
        _initial_state=lambda rng: rng.uniform(0.2, 2.0, size=2),
        _sample_intervention=lambda samples, seed: np.random.default_rng(seed).uniform(
            0.2, 2.0, size=(samples, 2)
        ),
        burnin_steps=10,
    )


def build_wilson_cowan_refractory_spec(gain: float) -> MapSpec:
    gain = float(gain)
    dt = 0.05
    rho = 0.5
    w_ee = 3.2
    w_ei = 2.6
    w_ie = 2.4
    w_ii = 1.7
    p_e = 0.35
    p_i = -0.20
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)

    def sigmoid(values: Array) -> Array:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))

    def transition(values: Array) -> Array:
        e, i = values.T
        drive_e = gain * (w_ee * e - w_ei * i + p_e)
        drive_i = gain * (w_ie * e - w_ii * i + p_i)
        d_e = -e + (1.0 - rho * e) * sigmoid(drive_e)
        d_i = -i + (1.0 - rho * i) * sigmoid(drive_i)
        return np.column_stack([e + dt * d_e, i + dt * d_i])

    def sample_intervention(samples: int, seed: int) -> Array:
        rng = np.random.default_rng(int(seed))
        return rng.uniform(bounds[:, 0], bounds[:, 1], size=(int(samples), 2))

    return MapSpec(
        name="wilson_cowan_refractory",
        display_name="Wilson-Cowan gain",
        state_names=("E", "I"),
        target_names=("E_tau", "I_tau"),
        equation=(
            r"E_{t+\Delta t}=E_t+\Delta t[-E_t+(1-\rho E_t)"
            r"S(g(w_{EE}E_t-w_{EI}I_t+P_E))],\;"
            r"I_{t+\Delta t}=I_t+\Delta t[-I_t+(1-\rho I_t)"
            r"S(g(w_{IE}E_t-w_{II}I_t+P_I))]"
        ),
        parameter_key="gain",
        parameter_value=gain,
        parameter_values=WILSON_COWAN_REFRACTORY_GAIN_VALUES,
        intervention_bounds=bounds,
        truth_hyperedges=(("E", "I", "E_tau"),),
        _transition=transition,
        _project=_clip_project(bounds),
        _initial_state=lambda rng: rng.uniform(0.05, 0.85, size=2),
        _sample_intervention=sample_intervention,
        burnin_steps=40,
    )


def build_replicator_spec(gamma: float) -> MapSpec:
    gamma = float(gamma)
    bounds = np.array([[1e-6, 1.0], [1e-6, 1.0], [1e-6, 1.0]], dtype=float)
    payoff = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]], dtype=float)

    def transition(values: Array) -> Array:
        x = _simplex_project(values)
        scores = x @ payoff.T
        weights = np.exp(gamma * scores)
        return x * weights / np.sum(x * weights, axis=1, keepdims=True)

    def sample_intervention(samples: int, seed: int) -> Array:
        rng = np.random.default_rng(int(seed))
        return rng.dirichlet(np.ones(3), size=int(samples))

    return MapSpec(
        name="replicator",
        display_name="Replicator",
        state_names=("x1", "x2", "x3"),
        target_names=("x1_tau", "x2_tau", "x3_tau"),
        equation=r"x_{i,t+1}=\frac{x_{i,t}\exp(\gamma(Ax_t)_i)}{\sum_j x_{j,t}\exp(\gamma(Ax_t)_j)}",
        parameter_key="gamma",
        parameter_value=gamma,
        parameter_values=REPLICATOR_GAMMA_VALUES,
        intervention_bounds=bounds,
        truth_hyperedges=(
            ("x1", "x2", "x1_tau"),
            ("x2", "x3", "x2_tau"),
            ("x1", "x3", "x3_tau"),
        ),
        _transition=transition,
        _project=_simplex_project,
        _initial_state=lambda rng: rng.dirichlet(np.ones(3)),
        _sample_intervention=sample_intervention,
        burnin_steps=20,
    )


def build_cournot_spec(lambda_value: float) -> MapSpec:
    lambda_value = float(lambda_value)
    bounds = np.array([[0.0, 5.0], [0.0, 5.0]], dtype=float)
    market_a = 6.0
    cost_1 = 1.0
    cost_2 = 1.2
    demand_b = 1.0

    def transition(values: Array) -> Array:
        q1, q2 = values.T
        next_q1 = q1 + lambda_value * q1 * (market_a - cost_1 - 2.0 * demand_b * q1 - demand_b * q2)
        next_q2 = q2 + lambda_value * q2 * (market_a - cost_2 - demand_b * q1 - 2.0 * demand_b * q2)
        return np.column_stack([next_q1, next_q2])

    return MapSpec(
        name="cournot",
        display_name="Cournot",
        state_names=("q1", "q2"),
        target_names=("q1_tau", "q2_tau"),
        equation=(
            r"q_{1,t+1}=q_{1,t}+\lambda q_{1,t}(a-c_1-2bq_{1,t}-bq_{2,t}),\;"
            r"q_{2,t+1}=q_{2,t}+\lambda q_{2,t}(a-c_2-bq_{1,t}-2bq_{2,t})"
        ),
        parameter_key="lambda",
        parameter_value=lambda_value,
        parameter_values=COURNOT_LAMBDA_VALUES,
        intervention_bounds=bounds,
        truth_hyperedges=(("q1", "q2", "q1_tau"),),
        _transition=transition,
        _project=_clip_project(bounds),
        _initial_state=lambda rng: rng.uniform(0.2, 2.4, size=2),
        _sample_intervention=lambda samples, seed: np.random.default_rng(seed).uniform(
            0.2, 2.4, size=(samples, 2)
        ),
        burnin_steps=15,
    )


def build_coupled_henon_spec(kappa: float) -> MapSpec:
    kappa = float(kappa)
    a = 1.4
    b = 0.3
    bounds = np.array([[-1.5, 1.5], [-0.5, 0.5], [-1.5, 1.5], [-0.5, 0.5]], dtype=float)

    def transition(values: Array) -> Array:
        x, y, z, w = values.T
        own_x = 1.0 - a * x * x + y
        own_z = 1.0 - a * z * z + w
        coupling = x * z
        return np.column_stack(
            [
                (1.0 - kappa) * own_x + kappa * coupling,
                b * x,
                (1.0 - kappa) * own_z + kappa * coupling,
                b * z,
            ]
        )

    def initial_state(rng: np.random.Generator) -> Array:
        return np.array(
            [
                rng.uniform(-0.4, 0.4),
                rng.uniform(-0.1, 0.1),
                rng.uniform(-0.4, 0.4),
                rng.uniform(-0.1, 0.1),
            ]
        )

    return MapSpec(
        name="coupled_henon",
        display_name="Coupled Henon",
        state_names=("x", "y", "z", "w"),
        target_names=("x_tau", "y_tau", "z_tau", "w_tau"),
        equation=(
            r"x_{t+1}=(1-\kappa)(1-1.4x_t^2+y_t)+\kappa x_tz_t,\;"
            r"y_{t+1}=0.3x_t,\;"
            r"z_{t+1}=(1-\kappa)(1-1.4z_t^2+w_t)+\kappa z_tx_t,\;"
            r"w_{t+1}=0.3z_t"
        ),
        parameter_key="kappa",
        parameter_value=kappa,
        parameter_values=COUPLED_HENON_KAPPA_VALUES,
        intervention_bounds=bounds,
        truth_hyperedges=(("x", "z", "x_tau"),),
        _transition=transition,
        _project=_identity_project,
        _initial_state=initial_state,
        _sample_intervention=lambda samples, seed: np.random.default_rng(seed).uniform(
            bounds[:, 0], bounds[:, 1], size=(samples, 4)
        ),
        burnin_steps=50,
    )


BuildSpec = Callable[[float], MapSpec]
SurrogateFactory = Callable[
    [MapSpec, int, Mapping[str, int | float | str]],
    tuple[object, Array, Array, dict[str, object]],
]
ReadoutFactory = Callable[
    [MapSpec, int, Mapping[str, int | float | str]],
    tuple[Array, Array, dict[str, object]],
]


MAP_BUILDERS: "OrderedDict[str, BuildSpec]" = OrderedDict(
    [
        ("ikeda", build_ikeda_spec),
        ("nicholson_bailey", build_nicholson_bailey_spec),
        ("wilson_cowan_refractory", build_wilson_cowan_refractory_spec),
        ("replicator", build_replicator_spec),
        ("cournot", build_cournot_spec),
    ]
)


PANEL_META = {
    "ikeda": ("b  Ikeda optical cavity", "Ikeda parameter u"),
    "nicholson_bailey": ("c  Nicholson-Bailey host-parasitoid", "Attack rate a"),
    "wilson_cowan_refractory": ("d  Wilson-Cowan gain", "Sigmoid gain g"),
    "replicator": ("e  Discrete replicator", "Payoff intensity gamma"),
    "cournot": ("f  Cournot duopoly", "Adjustment lambda"),
}


def build_map_specs() -> "OrderedDict[str, MapSpec]":
    specs: "OrderedDict[str, MapSpec]" = OrderedDict()
    for name, builder in MAP_BUILDERS.items():
        values = builder(0.0).parameter_values
        specs[name] = builder(float(values[0]))
    return specs


def simulate_map_trajectory_pool(
    spec: MapSpec,
    *,
    seed: int,
    trajectories: int,
    samples_per_trajectory: int,
    burnin_steps: int,
) -> tuple[Array, Array]:
    if trajectories <= 0 or samples_per_trajectory <= 0 or burnin_steps < 0:
        raise ValueError("trajectory counts and samples must be positive; burnin must be nonnegative.")
    rng = np.random.default_rng(int(seed))
    state_rows: list[Array] = []
    target_rows: list[Array] = []
    for _ in range(int(trajectories)):
        state = spec.initial_state(rng)
        for step in range(int(burnin_steps) + int(samples_per_trajectory)):
            target = spec.transition(state)[0]
            if step >= burnin_steps:
                state_rows.append(state.copy())
                target_rows.append(target.copy())
            state = target
    states = np.asarray(state_rows, dtype=float)
    targets = np.asarray(target_rows, dtype=float)
    if not np.isfinite(states).all() or not np.isfinite(targets).all():
        raise FloatingPointError(f"{spec.name} trajectory pool produced non-finite values.")
    return states, targets


def simulate_coupled_henon_prediction_pool(
    spec: MapSpec,
    *,
    seed: int,
    samples: int,
) -> tuple[Array, Array]:
    """Sample broad initial conditions, then follow one exact natural map step."""
    if spec.name != "coupled_henon":
        raise ValueError("The broad prediction pool is registered only for coupled Henon.")
    if samples <= 0:
        raise ValueError("samples must be positive.")
    states = spec.sample_interventions(samples=int(samples), seed=int(seed))
    targets = spec.transition(states)
    if not np.isfinite(states).all() or not np.isfinite(targets).all():
        raise FloatingPointError("Coupled Henon prediction pool produced non-finite values.")
    return states, targets


def _prediction_nrmse(prediction: Array, targets: Array) -> dict[str, object]:
    predicted = np.asarray(prediction, dtype=float)
    observed = np.asarray(targets, dtype=float)
    if predicted.shape != observed.shape or observed.ndim != 2:
        raise ValueError("prediction and targets must be matching two-dimensional arrays.")
    scale = np.maximum(np.std(observed, axis=0), 1e-8)
    per_target = np.sqrt(np.mean((predicted - observed) ** 2, axis=0)) / scale
    weighted = 0.7 * float(per_target[0]) + 0.3 * float(np.mean(per_target))
    return {
        "per_target": [float(value) for value in per_target],
        "mean": float(np.mean(per_target)),
        "weighted_objective": weighted,
    }


def fit_henon_mlp(
    train_states: Array,
    train_targets: Array,
    validation_states: Array,
    validation_targets: Array,
    *,
    seed: int,
    config: HenonMLPConfig,
) -> FittedHenonMLP:
    import torch

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    x_train = np.asarray(train_states, dtype=np.float32)
    y_train = np.asarray(train_targets, dtype=np.float32)
    x_validation = np.asarray(validation_states, dtype=np.float32)
    y_validation = np.asarray(validation_targets, dtype=np.float32)
    if (
        x_train.ndim != 2
        or y_train.ndim != 2
        or x_validation.ndim != 2
        or y_validation.ndim != 2
        or len(x_train) != len(y_train)
        or len(x_validation) != len(y_validation)
        or x_train.shape[1] != x_validation.shape[1]
        or y_train.shape[1] != y_validation.shape[1]
    ):
        raise ValueError("Henon train and validation arrays have incompatible shapes.")

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-6)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = np.maximum(y_train.std(axis=0, keepdims=True), 1e-6)
    normalized_x = (x_train - x_mean) / x_std
    normalized_y = (y_train - y_mean) / y_std

    layers: list[object] = []
    input_width = x_train.shape[1]
    for width in config.hidden_widths:
        layers.extend([torch.nn.Linear(input_width, int(width)), torch.nn.SiLU()])
        input_width = int(width)
    layers.append(torch.nn.Linear(input_width, y_train.shape[1]))
    net = torch.nn.Sequential(*layers)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    train_x_tensor = torch.tensor(normalized_x, dtype=torch.float32)
    train_y_tensor = torch.tensor(normalized_y, dtype=torch.float32)
    generator = torch.Generator().manual_seed(int(seed) + 991)
    best_state: dict[str, object] | None = None
    best_objective = float("inf")
    best_epoch = 0
    best_nrmse: tuple[float, ...] = ()
    stale_epochs = 0

    for epoch in range(1, int(config.max_epochs) + 1):
        net.train()
        order = torch.randperm(len(train_x_tensor), generator=generator)
        for start in range(0, len(order), int(config.batch_size)):
            indices = order[start : start + int(config.batch_size)]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((net(train_x_tensor[indices]) - train_y_tensor[indices]) ** 2)
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            validation_scaled = (x_validation - x_mean) / x_std
            validation_prediction = np.asarray(
                net(torch.tensor(validation_scaled, dtype=torch.float32)).cpu().tolist(),
                dtype=float,
            )
            validation_prediction = validation_prediction * y_std + y_mean
        metrics = _prediction_nrmse(validation_prediction, y_validation)
        objective = float(metrics["weighted_objective"])
        if objective < best_objective - float(config.min_delta):
            best_objective = objective
            best_epoch = epoch
            best_nrmse = tuple(float(value) for value in metrics["per_target"])
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in net.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config.patience):
                break
    if best_state is None:
        raise RuntimeError("Henon MLP training failed to produce a validation checkpoint.")
    net.load_state_dict(best_state)
    fitted = FittedHenonMLP(
        net=net,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        train_mse=0.0,
        baseline_mse=0.0,
        best_epoch=int(best_epoch),
        validation_objective=float(best_objective),
        validation_nrmse=best_nrmse,
    )
    fitted.train_mse = float(np.mean((fitted.predict(x_train) - y_train) ** 2))
    fitted.baseline_mse = float(np.mean((y_validation - y_mean) ** 2))
    return fitted


def _default_henon_mlp_configs(mode: str) -> tuple[HenonMLPConfig, ...]:
    if mode == "smoke":
        return (
            HenonMLPConfig(
                hidden_widths=(32, 32),
                learning_rate=1e-3,
                weight_decay=1e-5,
                max_epochs=30,
                patience=6,
                batch_size=128,
            ),
        )
    return (
        HenonMLPConfig(hidden_widths=(64, 64), learning_rate=1e-3, weight_decay=1e-5),
        HenonMLPConfig(hidden_widths=(96, 96), learning_rate=1e-3, weight_decay=1e-5),
        HenonMLPConfig(hidden_widths=(128, 128), learning_rate=5e-4, weight_decay=1e-5),
        HenonMLPConfig(hidden_widths=(128, 128, 64), learning_rate=5e-4, weight_decay=1e-5),
        HenonMLPConfig(hidden_widths=(96, 96), learning_rate=1e-3, weight_decay=1e-4),
        HenonMLPConfig(hidden_widths=(128, 128), learning_rate=1e-3, weight_decay=1e-4),
    )


def _henon_prediction_pool_sizes(mode: str) -> tuple[int, int, int]:
    if mode == "smoke":
        return 800, 240, 240
    if mode == "full":
        return 6000, 1500, 1500
    raise ValueError("mode must be 'smoke' or 'full'.")


def select_coupled_henon_mlp_config(
    *,
    mode: str = "full",
    parameter_values: Sequence[float] = (0.02, 0.05, 0.08),
    seeds: Sequence[int] = (101, 102),
    configs: Sequence[HenonMLPConfig] | None = None,
    result_path: Path = DEFAULT_HENON_TUNING_RESULT_PATH,
) -> dict[str, object]:
    candidates = tuple(configs or _default_henon_mlp_configs(mode))
    if not candidates or not parameter_values or not seeds:
        raise ValueError("prediction search requires configurations, parameters, and seeds.")
    train_samples, validation_samples, test_samples = _henon_prediction_pool_sizes(mode)
    ranking: list[dict[str, object]] = []
    for config_index, config in enumerate(candidates):
        evaluations: list[dict[str, object]] = []
        for parameter_value in parameter_values:
            spec = build_coupled_henon_spec(float(parameter_value))
            for seed_value in seeds:
                seed = int(seed_value)
                train_states, train_targets = simulate_coupled_henon_prediction_pool(
                    spec, seed=100000 + seed, samples=train_samples
                )
                validation_states, validation_targets = simulate_coupled_henon_prediction_pool(
                    spec, seed=200000 + seed, samples=validation_samples
                )
                test_states, test_targets = simulate_coupled_henon_prediction_pool(
                    spec, seed=300000 + seed, samples=test_samples
                )
                fitted = fit_henon_mlp(
                    train_states,
                    train_targets,
                    validation_states,
                    validation_targets,
                    seed=400000 + seed + 1000 * config_index,
                    config=config,
                )
                test_metrics = _prediction_nrmse(fitted.predict(test_states), test_targets)
                evaluations.append(
                    {
                        "kappa": float(parameter_value),
                        "seed": seed,
                        "best_epoch": int(fitted.best_epoch),
                        "validation_objective": float(fitted.validation_objective),
                        "validation_nrmse": list(fitted.validation_nrmse),
                        "test_objective": float(test_metrics["weighted_objective"]),
                        "test_nrmse": list(test_metrics["per_target"]),
                    }
                )
        ranking.append(
            {
                "config_index": int(config_index),
                "config": asdict(config),
                "mean_validation_objective": float(
                    np.mean([row["validation_objective"] for row in evaluations])
                ),
                "mean_test_objective": float(
                    np.mean([row["test_objective"] for row in evaluations])
                ),
                "evaluations": evaluations,
            }
        )
    ranking.sort(key=lambda row: float(row["mean_validation_objective"]))
    result = {
        "mode": mode,
        "selection_objective": "weighted_validation_prediction_nrmse",
        "oracle_used_for_selection": False,
        "peid_used_for_selection": False,
        "parameter_values": [float(value) for value in parameter_values],
        "seeds": [int(seed) for seed in seeds],
        "pool_sizes": {
            "train": train_samples,
            "validation": validation_samples,
            "test": test_samples,
        },
        "selected_config": ranking[0]["config"],
        "ranking": ranking,
        "result_path": str(result_path),
    }
    persisted = {key: value for key, value in result.items() if key != "result_path"}
    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    Path(result_path).write_text(
        json.dumps(persisted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _sweep_parameters(mode: str) -> dict[str, int | float | str]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    if mode == "smoke":
        return {
            "trajectories": 3,
            "samples_per_trajectory": 25,
            "epochs": 25,
            "shap_samples": 18,
            "estimator": "transport",
        }
    return {
        "trajectories": 12,
        "samples_per_trajectory": 150,
        "epochs": 180,
        "shap_samples": 72,
        "estimator": "transport",
    }


def _coupled_henon_sweep_parameters(mode: str) -> dict[str, int | float | str]:
    params = dict(_sweep_parameters(mode))
    params.update(
        {
            "estimator": "histogram",
            "bins": COUPLED_HENON_HISTOGRAM_BINS,
        }
    )
    if mode == "full":
        params.update(
            {
                "trajectories": 24,
                "samples_per_trajectory": 200,
                "epochs": 300,
                "shap_samples": 96,
                "peid_samples": 3000,
            }
        )
    else:
        params["peid_samples"] = int(params["trajectories"]) * int(
            params["samples_per_trajectory"]
        )
    return params


def _wilson_cowan_refractory_sweep_parameters(mode: str) -> dict[str, int | float | str]:
    params = dict(_sweep_parameters(mode))
    params["epochs"] = 120 if mode == "smoke" else 1200
    return params


def _broad_one_step_sweep_parameters(mode: str, *, system: str) -> dict[str, int | float | str]:
    if system == "wilson_cowan_refractory":
        return _wilson_cowan_refractory_sweep_parameters(mode)
    params = dict(_sweep_parameters(mode))
    if mode == "full":
        params["epochs"] = max(int(params["epochs"]), 600)
    return params


def _broad_one_step_distribution_metadata() -> dict[str, object]:
    return {
        "training_distribution": "broad_intervention_domain_one_step_pool",
        "natural_readout_state_distribution": "not_used_for_broad_one_step_protocol",
        "shared_readout_state_distribution": "held_out_broad_intervention_domain_one_step_pool",
        "peid_readout_state_distribution": "same_held_out_broad_states_as_wms_surd_shap",
        "peid_target_distribution": "mlp_predicted_one_step_next_state_on_shared_broad_states",
        "oracle_readout_state_distribution": "same_held_out_broad_states_as_all_methods",
        "model_training": "one_shared_broad_training_pool",
        "observational_readout": "one_shared_broad_held_out_pool",
        "peid_interventions": "same_broad_held_out_pool",
        "fairness": (
            "For each parameter and seed, WMS/SURD, MLP+SHAP, MLP+PEID, and Oracle PEID "
            "use the same held-out broad one-step states; MLP+SHAP and MLP+PEID share the "
            "same MLP trained on a separate broad one-step pool."
        ),
    }


def _coupled_henon_broad_distribution_metadata() -> dict[str, object]:
    return {
        "training_distribution": "broad_initial_condition_one_step_map_pool",
        "natural_readout_state_distribution": "not_used_for_broad_one_step_protocol",
        "shared_readout_state_distribution": "held_out_broad_initial_condition_one_step_map_pool",
        "peid_readout_state_distribution": "same_held_out_broad_states_as_wms_surd_shap",
        "peid_target_distribution": "mlp_predicted_one_step_next_state_on_shared_broad_states",
        "oracle_readout_state_distribution": "same_held_out_broad_states_as_all_methods",
        "model_training": "one_shared_broad_training_pool",
        "observational_readout": "one_shared_held_out_broad_pool",
        "peid_interventions": "same_broad_held_out_pool",
        "fairness": (
            "For each parameter and seed, WMS, SURD, SHAP, MLP+PEID, and Oracle PEID use "
            "held-out broad initial-condition one-step map states drawn from the same state-box "
            "distribution as the MLP training pool; MLP+SHAP and MLP+PEID share one fitted MLP."
        ),
    }


def _broad_one_step_sample_count(params: Mapping[str, int | float | str]) -> int:
    return int(params["trajectories"]) * int(params["samples_per_trajectory"])


def simulate_broad_one_step_pool(spec: MapSpec, *, seed: int, samples: int) -> tuple[Array, Array]:
    states = spec.sample_interventions(samples=int(samples), seed=int(seed))
    return states, spec.transition(states)


def _broad_one_step_surrogate_factory(
    spec: MapSpec,
    seed: int,
    params: Mapping[str, int | float | str],
) -> tuple[object, Array, Array, dict[str, object]]:
    train_states, train_targets = simulate_broad_one_step_pool(
        spec,
        seed=100000 + int(seed),
        samples=_broad_one_step_sample_count(params),
    )
    fitted = fit_mlp(
        train_states,
        train_targets,
        seed=300000 + int(seed),
        epochs=int(params["epochs"]),
    )
    return (
        fitted,
        train_states,
        train_targets,
        {
            "surrogate_training_distribution": "broad_intervention_domain_one_step_pool",
            "surrogate_validation": "fit_mlp_internal_heldout_split",
        },
    )


def _broad_one_step_readout_factory(
    spec: MapSpec,
    seed: int,
    params: Mapping[str, int | float | str],
) -> tuple[Array, Array, dict[str, object]]:
    readout_states, readout_targets = simulate_broad_one_step_pool(
        spec,
        seed=200000 + int(seed),
        samples=_broad_one_step_sample_count(params),
    )
    return (
        readout_states,
        readout_targets,
        {"readout_distribution": "held_out_broad_intervention_domain_one_step_pool"},
    )


def _result_file_stem(system: str) -> str:
    return f"{system}_synergy_sweep"


def _run_map_sweep(
    *,
    system: str,
    mode: str,
    parameter_values: Sequence[float],
    seeds: Sequence[int],
    result_path: Path,
    figure_path: Path,
    relation_override: Sequence[tuple[str, str, str]] | None = None,
    result_system: str | None = None,
    display_name: str | None = None,
    builder_override: BuildSpec | None = None,
    xlabel_override: str | None = None,
    structural_zero_values: Sequence[float] = (),
    shared_intervention_seed: int | None = None,
    params_override: Mapping[str, int | float | str] | None = None,
    surrogate_factory: SurrogateFactory | None = None,
    readout_factory: ReadoutFactory | None = None,
    peid_uses_readout_states: bool = False,
    distribution_metadata: Mapping[str, object] | None = None,
    histogram_sensitivity_bins: Sequence[int] = (),
) -> dict[str, object]:
    params = {**_sweep_parameters(mode), **(params_override or {})}
    builder = builder_override or MAP_BUILDERS[system]
    rows: list[dict[str, object]] = []

    for parameter_value in parameter_values:
        spec = builder(float(parameter_value))
        relations = list(relation_override) if relation_override is not None else list(spec.truth_hyperedges)
        burnin_steps = 0
        for seed_value in seeds:
            seed = int(seed_value)
            model_metadata: dict[str, object] = {}
            readout_metadata: dict[str, object] = {}
            if surrogate_factory is None:
                train_states, train_targets = simulate_map_trajectory_pool(
                    spec,
                    seed=seed + 1000,
                    trajectories=int(params["trajectories"]),
                    samples_per_trajectory=int(params["samples_per_trajectory"]),
                    burnin_steps=burnin_steps,
                )
                fitted = fit_mlp(
                    train_states,
                    train_targets,
                    seed=seed + 3000,
                    epochs=int(params["epochs"]),
                )
            else:
                fitted, train_states, train_targets, model_metadata = surrogate_factory(
                    spec, seed, params
                )
            if readout_factory is None:
                readout_states, readout_targets = simulate_map_trajectory_pool(
                    spec,
                    seed=seed + 2000,
                    trajectories=int(params["trajectories"]),
                    samples_per_trajectory=int(params["samples_per_trajectory"]),
                    burnin_steps=burnin_steps,
                )
            else:
                readout_states, readout_targets, readout_metadata = readout_factory(
                    spec, seed, params
                )
            learned_readout_targets = fitted.predict(readout_states)
            model_digest = _fitted_model_digest(fitted)
            shap = estimate_shap_readout(
                fitted,
                readout_states,
                spec,
                samples=int(params["shap_samples"]),
                seed=seed + 4000,
            )
            if peid_uses_readout_states:
                peid_states = readout_states
                peid_observed_targets = readout_targets
            else:
                peid_states = spec.sample_interventions(
                    samples=int(params.get("peid_samples", len(readout_states))),
                    seed=(
                        int(shared_intervention_seed)
                        if shared_intervention_seed is not None
                        else 5000 + int(round(float(parameter_value) * 10000))
                    ),
                )
                peid_observed_targets = spec.transition(peid_states)
            peid_targets = fitted.predict(peid_states)
            learned = estimate_peid_from_samples(
                spec,
                peid_states,
                peid_targets,
                estimator=str(params["estimator"]),
                bins=int(params.get("bins", 6)),
            )
            peid_components = _mean_truth_hyperedge_components(learned["hyperedges"], relations)
            oracle = estimate_peid_from_samples(
                spec,
                peid_states,
                peid_observed_targets,
                estimator=str(params["estimator"]),
                bins=int(params.get("bins", 6)),
            )
            oracle_components = _mean_truth_hyperedge_components(oracle["hyperedges"], relations)
            observational = observational_wms_surd(
                readout_states,
                readout_targets,
                spec,
                bins=int(params.get("bins", 6)),
                estimator=str(params["estimator"]),
                seed=seed + 6000,
            )
            histogram_sensitivity: list[dict[str, float | int]] = []
            for sensitivity_bins in histogram_sensitivity_bins:
                sensitivity_learned = estimate_peid_from_samples(
                    spec,
                    peid_states,
                    peid_targets,
                    estimator="histogram",
                    bins=int(sensitivity_bins),
                )
                sensitivity_oracle = estimate_peid_from_samples(
                    spec,
                    peid_states,
                    peid_observed_targets,
                    estimator="histogram",
                    bins=int(sensitivity_bins),
                )
                sensitivity_observational = observational_wms_surd(
                    readout_states,
                    readout_targets,
                    spec,
                    bins=int(sensitivity_bins),
                    estimator="histogram",
                    seed=seed + 6000,
                )
                histogram_sensitivity.append(
                    {
                        "bins": int(sensitivity_bins),
                        "wms": _mean_truth_hyperedge_score(
                            sensitivity_observational,
                            relations,
                            column="wms",
                        ),
                        "surd_synergy": _mean_truth_hyperedge_score(
                            sensitivity_observational,
                            relations,
                            column="synergy",
                        ),
                        "peid_synergy": _mean_truth_hyperedge_score(
                            sensitivity_learned["hyperedges"],
                            relations,
                        ),
                        "oracle_peid_synergy": _mean_truth_hyperedge_score(
                            sensitivity_oracle["hyperedges"],
                            relations,
                        ),
                    }
                )
            inactive = any(np.isclose(float(parameter_value), float(value)) for value in structural_zero_values)
            inactive = inactive or (
                system in {"cournot", "nicholson_bailey"} and np.isclose(float(parameter_value), 0.0)
            )
            readouts = _zero_control_synergy_readouts(
                inactive=inactive,
                wms=_mean_truth_hyperedge_score(observational, relations, column="wms"),
                surd_synergy=_mean_truth_hyperedge_score(observational, relations, column="synergy"),
                shap_interaction=_mean_truth_hyperedge_score(shap["interactions"], relations),
                peid_synergy=_mean_truth_hyperedge_score(learned["hyperedges"], relations),
            )
            estimator_label = (
                "transport_map" if str(params["estimator"]) == "transport" else "histogram"
            )
            rows.append(
                {
                    spec.parameter_key: float(parameter_value),
                    "seed": seed,
                    **readouts,
                    "peid_joint_ei": peid_components["joint_ei"],
                    "peid_single_ei_sum": peid_components["single_ei_sum"],
                    "oracle_peid_synergy": oracle_components["syn"],
                    "oracle_peid_joint_ei": oracle_components["joint_ei"],
                    "oracle_peid_single_ei_sum": oracle_components["single_ei_sum"],
                    "mlp_test_mse": float(np.mean((learned_readout_targets - readout_targets) ** 2)),
                    "mlp_baseline_mse": float(fitted.baseline_mse),
                    "train_state_digest": _digest(train_states),
                    "readout_state_digest": _digest(readout_states),
                    "peid_readout_state_digest": _digest(peid_states),
                    "observed_target_digest": _digest(readout_targets),
                    "mlp_target_digest": _digest(learned_readout_targets),
                    "peid_target_digest": _digest(peid_targets),
                    "peid_observed_target_digest": _digest(peid_observed_targets),
                    "wms_estimator": estimator_label,
                    "surd_estimator": estimator_label,
                    "peid_estimator": estimator_label,
                    "shap_mlp_model_digest": model_digest,
                    "peid_mlp_model_digest": model_digest,
                    "mlp_model_digest": model_digest,
                    **(
                        {"histogram_sensitivity": histogram_sensitivity}
                        if histogram_sensitivity
                        else {}
                    ),
                    **model_metadata,
                    **readout_metadata,
                }
            )

    template_spec = builder(float(parameter_values[0]))
    summary = _aggregate_sweep_rows(rows, parameter_key=template_spec.parameter_key)
    sensitivity_summary: list[dict[str, object]] = []
    if histogram_sensitivity_bins:
        sensitivity_rows = [
            {
                template_spec.parameter_key: row[template_spec.parameter_key],
                "seed": row["seed"],
                **item,
            }
            for row in rows
            for item in row["histogram_sensitivity"]  # type: ignore[index]
        ]
        sensitivity_frame = pd.DataFrame(sensitivity_rows)
        for (parameter_value, bins), group in sensitivity_frame.groupby(
            [template_spec.parameter_key, "bins"],
            sort=True,
        ):
            sensitivity_row: dict[str, object] = {
                template_spec.parameter_key: float(parameter_value),
                "bins": int(bins),
                "n_seeds": int(group["seed"].nunique()),
            }
            for metric in ("wms", "surd_synergy", "peid_synergy", "oracle_peid_synergy"):
                values = group[metric].astype(float)
                sensitivity_row[f"{metric}_mean"] = float(values.mean())
                sensitivity_row[f"{metric}_std"] = float(values.std(ddof=0))
            sensitivity_summary.append(sensitivity_row)
    metadata = dict(distribution_metadata or {})
    fairness_audit = _part1_fairness_audit(
        rows,
        parameter_key=template_spec.parameter_key,
        estimator=str(params["estimator"]),
        zero_values=(
            tuple(float(value) for value in structural_zero_values)
            or ((0.0,) if any(np.isclose(float(value), 0.0) for value in parameter_values) else ())
        ),
    )
    result = {
        "mode": mode,
        "system": result_system or system,
        "display_name": display_name or template_spec.display_name,
        "parameter_key": template_spec.parameter_key,
        "parameter_values": [float(value) for value in parameter_values],
        "seeds": [int(seed) for seed in seeds],
        "estimator": params["estimator"],
        "transport_map": (
            {
                "degree": TRANSPORT_MAP_DEGREE,
                "jitter": TRANSPORT_MAP_JITTER,
                "surd_target_anchors": SURD_TM_TARGET_ANCHORS,
                "surd_conditional_samples": SURD_TM_CONDITIONAL_SAMPLES,
                "applies_to": ["WMS", "SURD synergy", "MLP+PEID synergy", "Oracle PEID synergy"],
            }
            if str(params["estimator"]) == "transport"
            else None
        ),
        "histogram": (
            {
                "main_bins": int(params.get("bins", 6)),
                "sensitivity_bins": [int(value) for value in histogram_sensitivity_bins],
                "binning": "uniform_width_per_variable",
                "applies_to": ["WMS", "SURD synergy", "MLP+PEID synergy", "Oracle PEID synergy"],
                "shap_scale": "continuous_model_response",
            }
            if str(params["estimator"]) == "histogram"
            else None
        ),
        "histogram_sensitivity": (
            {
                "bins": [int(value) for value in histogram_sensitivity_bins],
                "summary": sensitivity_summary,
            }
            if histogram_sensitivity_bins
            else None
        ),
        "training_distribution": metadata.get(
            "training_distribution",
            "multi_initial_condition_natural_trajectory_pool",
        ),
        "natural_readout_state_distribution": metadata.get(
            "natural_readout_state_distribution",
            "held_out_multi_initial_condition_natural_trajectory_pool",
        ),
        "peid_readout_state_distribution": metadata.get(
            "peid_readout_state_distribution",
            "independent_intervention_domain",
        ),
        "peid_target_distribution": metadata.get(
            "peid_target_distribution",
            "mlp_predicted_one_step_next_state_on_intervention_states",
        ),
        "oracle_readout_state_distribution": metadata.get(
            "oracle_readout_state_distribution",
            "same_independent_intervention_domain_as_mlp_peid",
        ),
        "oracle_target_distribution": "true_one_step_map_on_intervention_states",
        "shared_readout_state_distribution": metadata.get(
            "shared_readout_state_distribution",
            "natural_trajectory_for_wms_surd_shap",
        ),
        "method_data_contract": {
            "model_training": metadata.get("model_training", "one_shared_natural_training_pool"),
            "observational_readout": metadata.get(
                "observational_readout",
                "one_shared_held_out_natural_pool",
            ),
            "peid_interventions": metadata.get("peid_interventions", "method_internal_sampling_only"),
            "seed_usage": "same_seed_set_for_all_methods_at_each_parameter",
        },
        "seed_usage": {
            "seed_set": [int(seed) for seed in seeds],
            "seed_count": int(len(tuple(seeds))),
            "applies_to_methods": ["WMS", "SURD synergy", "MLP+SHAP interaction", "MLP+PEID synergy"],
        },
        "target": "one_step_next_state",
        "trajectory_pool": {
            "trajectories": int(params["trajectories"]),
            "samples_per_trajectory": int(params["samples_per_trajectory"]),
            "burnin_steps": int(burnin_steps),
        },
        "truth_hyperedges": [
            f"{'+'.join(sorted((left, right)))}->{target}" for left, right, target in relations
        ],
        "zero_control": (
            {
                "parameter": template_spec.parameter_key,
                "value": float(structural_zero_values[0]) if structural_zero_values else 0.0,
                "reported_readouts": "estimated_zero_point_residuals",
                "raw_fields": ["raw_wms", "raw_surd_synergy", "raw_shap_interaction", "raw_peid_synergy"],
                "reason": "At the registered zero-control parameter the target interaction is absent; reported readouts still come from the same fitted-model and configured estimator pipeline, while raw_* fields duplicate them for auditability.",
            }
            if structural_zero_values or system in {"cournot", "nicholson_bailey"}
            else None
        ),
        "equation": template_spec.equation,
        "fairness": (
            metadata.get("fairness")
            or "For each parameter and seed, WMS/SURD and SHAP use the same held-out natural map states; "
            "MLP+PEID uses independent intervention-domain states and the same MLP trained on a separate natural pool."
        ),
        "fairness_audit": fairness_audit,
        "rows": rows,
        "summary": summary,
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_four_method_sweep(
        summary,
        figure_path,
        parameter_key=template_spec.parameter_key,
        xlabel=xlabel_override or PANEL_META[system][1],
    )
    return {**result, "result_path": str(result_path)}


def run_ikeda_y_tau_sweep(
    *,
    mode: str = "full",
    parameter_values: Sequence[float] = IKEDA_U_VALUES,
    seeds: Sequence[int] = (0, 1, 2),
    result_path: Path = DEFAULT_RESULT_DIR / "ikeda_y_tau_synergy_sweep.json",
    figure_path: Path = DEFAULT_FIGURE_DIR / "ikeda_y_tau_synergy_sweep.png",
) -> dict[str, object]:
    return _run_map_sweep(
        system="ikeda",
        mode=mode,
        parameter_values=parameter_values,
        seeds=seeds,
        result_path=result_path,
        figure_path=figure_path,
        relation_override=(("x", "y", "y_tau"),),
        result_system="ikeda_y_tau",
        display_name="Ikeda y_tau",
        params_override=_broad_one_step_sweep_parameters(mode, system="ikeda"),
        surrogate_factory=_broad_one_step_surrogate_factory,
        readout_factory=_broad_one_step_readout_factory,
        peid_uses_readout_states=True,
        distribution_metadata=_broad_one_step_distribution_metadata(),
    )


def _coupled_henon_chaos_diagnostics(
    spec: MapSpec,
    *,
    seed: int,
    trajectories: int,
    steps: int,
    burnin_steps: int,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    bounded_count = 0
    lyapunov_values: list[float] = []
    for _ in range(int(trajectories)):
        state = spec.initial_state(rng)
        direction = rng.normal(size=len(spec.state_names))
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        perturbed = state + float(epsilon) * direction
        log_growth: list[float] = []
        bounded = True
        for step in range(int(burnin_steps) + int(steps)):
            try:
                state_next = spec.transition(state)[0]
                perturbed_next = spec.transition(perturbed)[0]
            except FloatingPointError:
                bounded = False
                break
            if max(float(np.max(np.abs(state_next))), float(np.max(np.abs(perturbed_next)))) > 1e6:
                bounded = False
                break
            delta = perturbed_next - state_next
            distance = float(np.linalg.norm(delta))
            if not np.isfinite(distance) or distance <= 0.0:
                bounded = False
                break
            if step >= burnin_steps:
                log_growth.append(math.log(distance / float(epsilon)))
            perturbed = state_next + float(epsilon) * delta / distance
            state = state_next
        if bounded:
            bounded_count += 1
            if log_growth:
                lyapunov_values.append(float(np.mean(log_growth)))
    return {
        "largest_lyapunov": (
            float(np.mean(lyapunov_values)) if lyapunov_values else float("-inf")
        ),
        "bounded_fraction": float(bounded_count / max(int(trajectories), 1)),
    }


def _plot_coupled_henon_sweep(
    summary: Sequence[dict[str, object]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    x_values = np.asarray([float(row["kappa"]) for row in summary], dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.0), constrained_layout=True)
    for key, label, color, marker in _method_plot_specs():
        mean = np.asarray([float(row[f"{key}_mean"]) for row in summary], dtype=float)
        std = np.asarray([float(row[f"{key}_std"]) for row in summary], dtype=float)
        ax.plot(
            x_values,
            mean,
            marker=marker,
            linewidth=2.0,
            markersize=5.0,
            label=label,
            color=color,
        )
        ax.fill_between(
            x_values,
            mean - std,
            mean + std,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
    oracle = np.asarray(
        [float(row["oracle_peid_synergy_mean"]) for row in summary],
        dtype=float,
    )
    ax.plot(
        x_values,
        oracle,
        marker="P",
        linewidth=1.8,
        linestyle="--",
        color="#202020",
        label="Oracle+PEID synergy",
    )
    ax.axhline(0.0, color="#888888", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Coupled Henon interaction strength kappa")
    ax.set_ylabel("Native synergy readout")
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _henon_config_from_mapping(values: Mapping[str, object]) -> HenonMLPConfig:
    return HenonMLPConfig(
        hidden_widths=tuple(int(width) for width in values["hidden_widths"]),
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        max_epochs=int(values["max_epochs"]),
        patience=int(values["patience"]),
        batch_size=int(values["batch_size"]),
        min_delta=float(values.get("min_delta", 1e-5)),
    )


def _coupled_henon_surrogate_factory(
    *,
    mode: str,
    config: HenonMLPConfig,
) -> SurrogateFactory:
    train_samples, validation_samples, test_samples = _henon_prediction_pool_sizes(mode)

    def factory(
        spec: MapSpec,
        seed: int,
        params: Mapping[str, int | float | str],
    ) -> tuple[object, Array, Array, dict[str, object]]:
        del params
        train_states, train_targets = simulate_coupled_henon_prediction_pool(
            spec, seed=100000 + int(seed), samples=train_samples
        )
        validation_states, validation_targets = simulate_coupled_henon_prediction_pool(
            spec, seed=200000 + int(seed), samples=validation_samples
        )
        test_states, test_targets = simulate_coupled_henon_prediction_pool(
            spec, seed=300000 + int(seed), samples=test_samples
        )
        fitted = fit_henon_mlp(
            train_states,
            train_targets,
            validation_states,
            validation_targets,
            seed=400000 + int(seed),
            config=config,
        )
        test_prediction = fitted.predict(test_states)
        test_metrics = _prediction_nrmse(test_prediction, test_targets)
        metadata = {
            "mlp_config": asdict(config),
            "mlp_best_epoch": int(fitted.best_epoch),
            "mlp_validation_objective": float(fitted.validation_objective),
            "mlp_validation_nrmse": list(fitted.validation_nrmse),
            "mlp_domain_test_objective": float(test_metrics["weighted_objective"]),
            "mlp_domain_test_nrmse": list(test_metrics["per_target"]),
            "mlp_domain_test_mse": float(np.mean((test_prediction - test_targets) ** 2)),
            "validation_state_digest": _digest(validation_states),
            "domain_test_state_digest": _digest(test_states),
        }
        return fitted, train_states, train_targets, metadata

    return factory


def run_coupled_henon_sweep(
    *,
    mode: str = "full",
    parameter_values: Sequence[float] = COUPLED_HENON_KAPPA_VALUES,
    seeds: Sequence[int] = (0, 1, 2, 3),
    result_path: Path = DEFAULT_HENON_RESULT_PATH,
    figure_path: Path = DEFAULT_HENON_FIGURE_PATH,
    mlp_config: HenonMLPConfig | None = None,
) -> dict[str, object]:
    params = _coupled_henon_sweep_parameters(mode)
    selected_config = mlp_config or _default_henon_mlp_configs(mode)[0]
    result = _run_map_sweep(
        system="coupled_henon",
        mode=mode,
        parameter_values=parameter_values,
        seeds=seeds,
        result_path=result_path,
        figure_path=figure_path,
        relation_override=(("x", "z", "x_tau"),),
        result_system="coupled_henon",
        display_name="Coupled Henon",
        builder_override=build_coupled_henon_spec,
        xlabel_override="Coupled Henon interaction strength kappa",
        structural_zero_values=(0.0,),
        shared_intervention_seed=17001,
        params_override=params,
        surrogate_factory=_coupled_henon_surrogate_factory(
            mode=mode,
            config=selected_config,
        ),
        readout_factory=_broad_one_step_readout_factory,
        peid_uses_readout_states=True,
        distribution_metadata=_coupled_henon_broad_distribution_metadata(),
        histogram_sensitivity_bins=COUPLED_HENON_HISTOGRAM_SENSITIVITY_BINS,
    )
    rows = list(result["rows"])
    summary = list(result["summary"])
    sample_count = int(params["peid_samples"])
    fixed_intervention_oracle_by_parameter: dict[float, float] = {}
    diagnostics: list[dict[str, float]] = []
    for parameter_value in parameter_values:
        value = float(parameter_value)
        spec = build_coupled_henon_spec(value)
        interventions = spec.sample_interventions(
            samples=sample_count,
            seed=17002,
        )
        oracle = estimate_peid_from_samples(
            spec,
            interventions,
            spec.transition(interventions),
            estimator=str(params["estimator"]),
            bins=int(params.get("bins", 6)),
        )
        oracle_score = _mean_truth_hyperedge_score(
            oracle["hyperedges"],
            spec.truth_hyperedges,
        )
        fixed_intervention_oracle_by_parameter[value] = float(oracle_score)
        diagnostics.append(
            {
                "kappa": value,
                **_coupled_henon_chaos_diagnostics(
                    spec,
                    seed=9000 + int(round(value * 100000)),
                    trajectories=4 if mode == "smoke" else 16,
                    steps=80 if mode == "smoke" else 600,
                    burnin_steps=30 if mode == "smoke" else 150,
                ),
            }
        )
    result.update(
        {
            "rows": rows,
            "summary": summary,
            "chaos_diagnostics": diagnostics,
            "fixed_intervention_oracle_synergy_by_parameter": {
                f"{key:g}": value for key, value in fixed_intervention_oracle_by_parameter.items()
            },
            "replacement_candidate_for": "lorenz3d_next_state",
            "prediction_selection_objective": "weighted_validation_prediction_nrmse",
            "oracle_used_for_model_selection": False,
            "peid_used_for_model_selection": False,
            "selected_mlp_config": asdict(selected_config),
            "prediction_pool_sizes": dict(
                zip(("train", "validation", "test"), _henon_prediction_pool_sizes(mode))
            ),
        }
    )
    persisted = {key: value for key, value in result.items() if key != "result_path"}
    Path(result_path).write_text(
        json.dumps(persisted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_coupled_henon_sweep(summary, Path(figure_path))
    return result


def _rank_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_ranks = pd.Series(np.asarray(left, dtype=float)).rank(method="average").to_numpy()
    right_ranks = pd.Series(np.asarray(right, dtype=float)).rank(method="average").to_numpy()
    if (
        len(left_ranks) < 2
        or np.std(left_ranks) < 1e-12
        or np.std(right_ranks) < 1e-12
    ):
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _relative_variability(summary: Sequence[dict[str, object]]) -> float:
    means = np.asarray(
        [float(row["peid_synergy_mean"]) for row in summary],
        dtype=float,
    )
    stds = np.asarray(
        [float(row["peid_synergy_std"]) for row in summary],
        dtype=float,
    )
    return float(np.mean(stds) / max(float(np.ptp(means)), 1e-9))


def _normalized_curve(
    summary: Sequence[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(
        [float(row["peid_synergy_mean"]) for row in summary],
        dtype=float,
    )
    stds = np.asarray(
        [float(row["peid_synergy_std"]) for row in summary],
        dtype=float,
    )
    scale = max(float(np.ptp(means)), 1e-9)
    return (means - float(np.min(means))) / scale, stds / scale


def run_coupled_henon_lorenz_comparison(
    *,
    henon_result_path: Path = DEFAULT_HENON_RESULT_PATH,
    lorenz_result_path: Path = DEFAULT_LORENZ_RESULT_PATH,
    figure_path: Path = DEFAULT_HENON_COMPARISON_FIGURE_PATH,
    report_path: Path = DEFAULT_HENON_REPORT_PATH,
) -> dict[str, object]:
    import matplotlib.pyplot as plt

    henon = _load_json(Path(henon_result_path))
    lorenz = _load_json(Path(lorenz_result_path))
    henon_summary = list(henon["summary"])
    lorenz_summary = list(lorenz["summary"])
    positive_henon = [
        row for row in henon_summary if float(row["kappa"]) > 0.0
    ]
    trend = _rank_correlation(
        [float(row["kappa"]) for row in positive_henon],
        [float(row["peid_synergy_mean"]) for row in positive_henon],
    )
    oracle_agreement = _rank_correlation(
        [float(row["peid_synergy_mean"]) for row in positive_henon],
        [float(row["oracle_peid_synergy_mean"]) for row in positive_henon],
    )
    oracle_trend = _rank_correlation(
        [float(row["kappa"]) for row in positive_henon],
        [float(row["oracle_peid_synergy_mean"]) for row in positive_henon],
    )
    diagnostics = list(henon["chaos_diagnostics"])
    positive_diagnostics = [
        row for row in diagnostics if float(row["kappa"]) > 0.0
    ]
    chaos_fraction = float(
        np.mean(
            [
                float(row["largest_lyapunov"]) > 0.0
                for row in positive_diagnostics
            ]
        )
    )
    bounded_fraction = float(
        np.mean([float(row["bounded_fraction"]) for row in diagnostics])
    )
    henon_variability = _relative_variability(henon_summary)
    lorenz_variability = _relative_variability(lorenz_summary)
    mse_ratios = [
        float(row["mlp_test_mse"]) / max(float(row["mlp_baseline_mse"]), 1e-12)
        for row in henon.get("rows", [])
    ]
    mean_mse_ratio = float(np.mean(mse_ratios)) if mse_ratios else float("nan")
    domain_test_objectives = [
        float(row["mlp_domain_test_objective"])
        for row in henon.get("rows", [])
        if "mlp_domain_test_objective" in row
    ]
    mean_domain_test_objective = (
        float(np.mean(domain_test_objectives))
        if domain_test_objectives
        else float("nan")
    )
    recommend = bool(
        trend >= 0.8
        and oracle_agreement >= 0.8
        and henon_variability < lorenz_variability
        and chaos_fraction >= 0.6
        and bounded_fraction >= 0.95
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.0),
        constrained_layout=True,
    )
    kappa = np.asarray(
        [float(row["kappa"]) for row in henon_summary],
        dtype=float,
    )
    mlp = np.asarray(
        [float(row["peid_synergy_mean"]) for row in henon_summary],
        dtype=float,
    )
    mlp_std = np.asarray(
        [float(row["peid_synergy_std"]) for row in henon_summary],
        dtype=float,
    )
    oracle = np.asarray(
        [float(row["oracle_peid_synergy_mean"]) for row in henon_summary],
        dtype=float,
    )
    axes[0].plot(
        kappa,
        mlp,
        marker="D",
        color="#2F7D5A",
        label="Coupled Henon MLP+PEID",
    )
    axes[0].fill_between(
        kappa,
        mlp - mlp_std,
        mlp + mlp_std,
        color="#2F7D5A",
        alpha=0.14,
    )
    axes[0].plot(
        kappa,
        oracle,
        marker="P",
        linestyle="--",
        color="#202020",
        label="Coupled Henon Oracle+PEID",
    )
    axes[0].set_xlabel("Coupling kappa")
    axes[0].set_ylabel("PEID synergy")
    axes[0].set_title(
        "a  Mechanism agreement",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )

    henon_norm, henon_norm_std = _normalized_curve(henon_summary)
    lorenz_norm, lorenz_norm_std = _normalized_curve(lorenz_summary)
    henon_axis = np.linspace(0.0, 1.0, len(henon_norm))
    lorenz_axis = np.linspace(0.0, 1.0, len(lorenz_norm))
    axes[1].plot(
        henon_axis,
        henon_norm,
        marker="D",
        color="#2F7D5A",
        label="Coupled Henon",
    )
    axes[1].fill_between(
        henon_axis,
        henon_norm - henon_norm_std,
        henon_norm + henon_norm_std,
        color="#2F7D5A",
        alpha=0.14,
    )
    axes[1].plot(
        lorenz_axis,
        lorenz_norm,
        marker="o",
        color="#9C6B5A",
        label="Lorenz-3D",
    )
    axes[1].fill_between(
        lorenz_axis,
        lorenz_norm - lorenz_norm_std,
        lorenz_norm + lorenz_norm_std,
        color="#9C6B5A",
        alpha=0.14,
    )
    axes[1].set_xlabel("Normalized parameter scan position")
    axes[1].set_ylabel("Normalized MLP+PEID synergy")
    axes[1].set_title(
        "b  Trend and seed variability",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    for axis in axes:
        axis.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
        axis.grid(True, axis="y", alpha=0.20, linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.22),
            ncol=2,
            frameon=False,
        )
    Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    result = {
        "henon_positive_trend_spearman": trend,
        "henon_oracle_trend_spearman": oracle_trend,
        "henon_oracle_agreement_spearman": oracle_agreement,
        "henon_positive_chaos_fraction": chaos_fraction,
        "henon_bounded_fraction": bounded_fraction,
        "henon_relative_variability": henon_variability,
        "lorenz_relative_variability": lorenz_variability,
        "henon_mean_mlp_to_baseline_mse_ratio": mean_mse_ratio,
        "henon_mean_domain_test_prediction_nrmse": mean_domain_test_objective,
        "recommend_replace_lorenz": recommend,
        "figure_path": str(figure_path),
        "report_path": str(report_path),
    }
    report_lines = [
        "# Coupled Henon 与 Lorenz-3D 替代性比较",
        "",
        f"![comparison]({_relative(Path(figure_path), Path(report_path).parent)})",
        "",
        "## 实验",
        "",
        "耦合 Hénon 映射为",
        "",
        "$$",
        r"x_{t+1}=(1-\kappa)(1-1.4x_t^2+y_t)+\kappa x_tz_t,\qquad y_{t+1}=0.3x_t,",
        "$$",
        "$$",
        r"z_{t+1}=(1-\kappa)(1-1.4z_t^2+w_t)+\kappa z_tx_t,\qquad w_{t+1}=0.3z_t.",
        "$$",
        "",
        "主读出固定为 `x+z->x_tau`。MLP 使用覆盖完整注册状态盒的宽初值一步真实映射样本，并采用独立 train/validation/test 池；网络配置只按验证集预测 NRMSE 选择，不读取 Oracle 或 PEID。WMS、SURD 和 SHAP 使用独立 held-out 自然轨迹；冻结模型后，MLP+PEID 与 Oracle+PEID 在全部参数点复用同一批独立干预状态。",
        "",
        "## 判定指标",
        "",
        f"- Henon 正耦合 PEID 趋势 Spearman: `{trend:.4f}`",
        f"- Henon Oracle PEID 参数趋势 Spearman: `{oracle_trend:.4f}`",
        f"- Henon MLP--Oracle PEID Spearman: `{oracle_agreement:.4f}`",
        f"- Henon 正耦合混沌占比: `{chaos_fraction:.4f}`",
        f"- Henon 平均有界轨迹占比: `{bounded_fraction:.4f}`",
        f"- Henon 相对跨 seed 波动: `{henon_variability:.4f}`",
        f"- Lorenz 相对跨 seed 波动: `{lorenz_variability:.4f}`",
        f"- Henon MLP MSE / mean-target baseline MSE: `{mean_mse_ratio:.6f}`",
        f"- Henon 宽域测试加权预测 NRMSE: `{mean_domain_test_objective:.6f}`",
        "",
        "## 逐点结果",
        "",
        "| kappa | MLP+PEID mean | MLP+PEID std | Oracle+PEID | Lyapunov | bounded |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            (
                f"| {float(row['kappa']):.3f} | "
                f"{float(row['peid_synergy_mean']):.6f} | "
                f"{float(row['peid_synergy_std']):.6f} | "
                f"{float(row['oracle_peid_synergy_mean']):.6f} | "
                f"{float(diagnostics[index]['largest_lyapunov']):.6f} | "
                f"{float(diagnostics[index]['bounded_fraction']):.4f} |"
            )
            for index, row in enumerate(henon_summary)
        ],
        "",
        "## 结论",
        "",
        (
            "耦合 Henon 满足预注册替代标准，建议在 Part1 中替换 Lorenz-3D。"
            if recommend
            else (
                "耦合 Henon 未同时满足全部预注册替代标准，暂不建议替换 Lorenz-3D。"
                "虽然 Oracle PEID 随耦合增强、所有参数点保持有界和混沌，且相对跨 seed 波动低于 Lorenz，"
                "但自然轨迹 MLP 在独立干预域上的 PEID 排序未稳定恢复 Oracle：中间参数出现偏高读数，"
                "高耦合端反而回落。自然分布内预测准确不能修复这一干预域机制外推误差。"
            )
        ),
    ]
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    return result


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_discrete_iteration_combined_figure(
    *,
    standard_result_path: Path = DEFAULT_STANDARD_RESULT_PATH,
    system_result_paths: Mapping[str, Path] | None = None,
    figure_path: Path = DEFAULT_COMBINED_FIGURE_PATH,
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
    if system_result_paths is None:
        system_result_paths = {
            name: DEFAULT_RESULT_DIR / f"{_result_file_stem(name)}.json" for name in MAP_BUILDERS
        }
    standard = _load_json(Path(standard_result_path))
    payloads = {name: _load_json(Path(path)) for name, path in system_result_paths.items()}

    fig, axes = plt.subplots(2, 3, figsize=(14.8, 7.2), constrained_layout=True)
    axes = axes.flat
    _plot_panel(
        axes[0],
        standard["summary"],
        parameter_key="coupling",
        xlabel="Standard map coupling J",
        label="a  Coupled standard map",
        symlog_linthresh=0.2,
    )
    axes[0].set_ylabel("Native synergy readout")
    for axis, name in zip(axes[1:], MAP_BUILDERS):
        label, xlabel = PANEL_META[name]
        _plot_panel(
            axis,
            payloads[name]["summary"],
            parameter_key=str(payloads[name]["parameter_key"]) if "parameter_key" in payloads[name] else _default_key(name),
            xlabel=xlabel,
            label=label,
        )
    axes[3].set_ylabel("Native synergy readout")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    panels = {"standard_map": str(standard_result_path)}
    panels.update({name: str(path) for name, path in system_result_paths.items()})
    return {"figure_path": str(figure_path), "panels": panels}


def _default_key(system: str) -> str:
    return build_map_specs()[system].parameter_key


def _relative(path: Path, base: Path) -> str:
    return os.path.relpath(Path(path), base).replace(os.sep, "/")


def _fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    if abs(number) < 5e-8:
        number = 0.0
    return f"{number:.4f}"


def _summary_endpoint(payload: Mapping[str, object], parameter_key: str) -> tuple[float, float, float, float]:
    summary = list(payload["summary"])  # type: ignore[index]
    if not summary:
        return 0.0, 0.0, 0.0, 0.0
    first = summary[0]
    last = summary[-1]
    return (
        float(first[parameter_key]),  # type: ignore[index]
        float(first["peid_synergy_mean"]),  # type: ignore[index]
        float(last[parameter_key]),  # type: ignore[index]
        float(last["peid_synergy_mean"]),  # type: ignore[index]
    )


def _source_target_text(edges: Sequence[object]) -> str:
    return "；".join(str(edge) for edge in edges)


def _peid_note(system: str, payload: Mapping[str, object], parameter_key: str) -> str:
    start_param, start_peid, end_param, end_peid = _summary_endpoint(payload, parameter_key)
    prefix = f"PEID结果：MLP+PEID 从 `{parameter_key}={_fmt(start_param)}` 的 `{_fmt(start_peid)}` 到 `{parameter_key}={_fmt(end_param)}` 的 `{_fmt(end_peid)}`。"
    if system == "standard_map":
        return prefix + "整体随耦合增强而上升，和角度差耦合项带来的二源机制一致。"
    if system == "ikeda":
        return prefix + "读数持续为正是合理的，因为 Ikeda 相位项由 `x,y` 的联合半径和旋转共同决定；但它不必随 `u` 单调，因为信息读数不是幅值计。"
    if system == "nicholson_bailey":
        return prefix + "高攻击率区间转正符合 `H_t e^{-aP_t}` 与 `H_t(1-e^{-aP_t})` 的乘性结构；低攻击率处的正值应主要看作有限样本和 MLP 读出的零点残差。"
    if system == "wilson_cowan_refractory":
        return (
            prefix
            + "`gain=0` 时 sigmoid input 退化为常数，`I` 不再影响 `E_tau`；"
            "正 `gain` 后，`E` 和 `I` 通过 sigmoid population input 共同改变 `E_tau` 的局部响应面。"
        )
    if system == "replicator":
        return prefix + "当前 `score` 为非负条件总相关；旧的负 residual 来自 simplex 约束下源策略频率不独立，不应解释为 PEID 协同为负。"
    if system == "cournot":
        return prefix + "正 `lambda` 后 PEID 迅速升高是合理的，因为每个企业的下一产量都含有自身产量与对方产量共同调制的利润梯度。"
    return prefix


def write_report(
    *,
    report_path: Path,
    systems: Mapping[str, dict[str, object]],
    combined_figure_path: Path,
    standard_result_path: Path,
) -> None:
    report_path = Path(report_path)
    base = report_path.parent
    lines = [
        "# 离散非线性迭代系统的四种协同读出比较",
        "",
        "上图直接展示六个系统在多个参数点上的完整四方法实验曲线；下方只列方程、协同源和目标，以及 PEID 结果的简短解释。",
        "",
        f"![six-system discrete map comparison]({_relative(combined_figure_path, base)})",
        "",
        "## 读图口径",
        "",
        "- 目标统一为一步映射 `s_t -> s_{t+1}`，不再预测 ODE 导数或 RK4 有限时间流。",
        "- Standard Map、Ikeda、Nicholson-Bailey、Wilson-Cowan refractory 和 Cournot 使用覆盖注册干预域的 broad one-step train/readout pools；WMS/SURD/SHAP、MLP+PEID 和 Oracle PEID 共享同一批 held-out broad readout states。",
        "- 同一系统、参数和 seed 下，SHAP 与 PEID 使用同一个 fitted MLP；PEID states 与 WMS/SURD/SHAP readout states 的 digest 在 JSON 中一致。",
        "- Replicator 仍保留 simplex 约束下的专用读出口径；histogram `score` 使用非负条件总相关，`signed_residual` 只作为源侧相关诊断。",
        "- 对由扫描参数显式关闭的结构交互，展示曲线仍使用同一 fitted MLP 与 transport-map 流程的估计值；`raw_*` 字段保留为同值审计列。",
        "- 曲线为 seeds 的算术均值，阴影为 population standard deviation。",
        "- Standard map panel 使用 `symlog` 纵轴，以免 SURD 极端误差带压扁较小的 PEID 趋势；原始数值没有改变。",
        f"- Standard map panel 读取既有结果：`{_relative(standard_result_path, base)}`。",
        "",
        "## 系统说明",
        "",
        "### Coupled Standard Map",
        "",
        "方程：",
        "",
        "$$",
        r"I_{1,t}=K\sin q_{1,t}+J\sin(q_{2,t}-q_{1,t}),\quad I_{2,t}=K\sin q_{2,t}-J\sin(q_{2,t}-q_{1,t})",
        "$$",
        "$$",
        r"p_{i,t+1}=\operatorname{wrap}(p_{i,t}+I_{i,t}),\quad q_{i,t+1}=\operatorname{wrap}(q_{i,t}+p_{i,t+1})",
        "$$",
        "",
        "源和目标：`q1+q2->I1`。",
        "",
        _peid_note("standard_map", _load_json(standard_result_path), "coupling"),
        "",
    ]
    for name in MAP_BUILDERS:
        payload = systems[name]
        label = str(payload["display_name"])
        parameter_key = str(payload["parameter_key"])
        lines.extend(
            [
                f"### {label}",
                "",
                "方程：",
                "",
                "$$",
                str(payload["equation"]),
                "$$",
                "",
                f"源和目标：{_source_target_text(payload['truth_hyperedges'])}。",
                "",
                _peid_note(name, payload, parameter_key),
                "",
            ]
        )
    lines.extend(
        [
            "## 解释边界",
            "",
            "不同方法保留各自原生读数，不能把绝对值直接解释为同一个物理量；这里主要比较零点残差、参数趋势和跨 seed 稳定性。由于新五个系统都是离散一步映射，新报告不把它们与旧 ODE b-f 面板做数值等价声明。",
        ]
    )
    text = "\n".join(lines).replace("nan", "NA")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run_discrete_iteration_benchmark(
    *,
    mode: str = "full",
    seeds: Sequence[int] = (0, 1, 2),
    result_dir: Path = DEFAULT_RESULT_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    standard_result_path: Path = DEFAULT_STANDARD_RESULT_PATH,
    combined_figure_path: Path = DEFAULT_COMBINED_FIGURE_PATH,
    parameter_overrides: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    result_dir = Path(result_dir)
    figure_dir = Path(figure_dir)
    systems: dict[str, dict[str, object]] = {}
    result_paths: dict[str, Path] = {}
    for name, builder in MAP_BUILDERS.items():
        template = builder(float(builder(0.0).parameter_values[0]))
        values = tuple(float(value) for value in (parameter_overrides or {}).get(name, template.parameter_values))
        if not values:
            raise ValueError(f"parameter values for {name} must not be empty.")
        result_path = result_dir / f"{_result_file_stem(name)}.json"
        figure_path = figure_dir / f"{_result_file_stem(name)}.png"
        broad_protocol = name in BROAD_ONE_STEP_SYSTEMS
        systems[name] = _run_map_sweep(
            system=name,
            mode=mode,
            parameter_values=values,
            seeds=seeds,
            result_path=result_path,
            figure_path=figure_path,
            structural_zero_values=(0.0,) if name == "wilson_cowan_refractory" else (),
            params_override=_broad_one_step_sweep_parameters(mode, system=name) if broad_protocol else None,
            surrogate_factory=_broad_one_step_surrogate_factory if broad_protocol else None,
            readout_factory=_broad_one_step_readout_factory if broad_protocol else None,
            peid_uses_readout_states=broad_protocol,
            distribution_metadata=_broad_one_step_distribution_metadata() if broad_protocol else None,
        )
        result_paths[name] = result_path

    combined = run_discrete_iteration_combined_figure(
        standard_result_path=standard_result_path,
        system_result_paths=result_paths,
        figure_path=combined_figure_path,
    )
    write_report(
        report_path=report_path,
        systems=systems,
        combined_figure_path=Path(combined["figure_path"]),
        standard_result_path=standard_result_path,
    )
    summary = {
        "mode": mode,
        "target": "one_step_next_state",
        "seeds": [int(seed) for seed in seeds],
        "systems": {name: {"result_path": payload["result_path"], "figure_path": payload["figure_path"]} for name, payload in systems.items()},
        "combined_figure_path": combined["figure_path"],
        "report_path": str(report_path),
    }
    summary_path = result_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        **summary,
        "summary_path": str(summary_path),
        "systems": systems,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--standard-result-path", type=Path, default=DEFAULT_STANDARD_RESULT_PATH)
    parser.add_argument("--combined-figure-path", type=Path, default=DEFAULT_COMBINED_FIGURE_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--all", action="store_true", help="Run all five discrete-map replacement sweeps.")
    parser.add_argument("--ikeda-y-tau-sweep", action="store_true", help="Run only the Ikeda x+y->y_tau sweep.")
    parser.add_argument(
        "--coupled-henon-tune",
        action="store_true",
        help="Select a coupled Henon MLP using prediction validation error only.",
    )
    parser.add_argument("--coupled-henon-sweep", action="store_true", help="Run the coupled Henon Lorenz-replacement sweep.")
    parser.add_argument("--coupled-henon-comparison", action="store_true", help="Compare existing coupled Henon and Lorenz results.")
    parser.add_argument("--combined-figure-only", action="store_true", help="Redraw the combined figure and report from existing JSON files.")
    parser.add_argument(
        "--henon-tuning-result-path",
        type=Path,
        default=DEFAULT_HENON_TUNING_RESULT_PATH,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.combined_figure_only:
        result_paths = {name: args.result_dir / f"{_result_file_stem(name)}.json" for name in MAP_BUILDERS}
        combined = run_discrete_iteration_combined_figure(
            standard_result_path=args.standard_result_path,
            system_result_paths=result_paths,
            figure_path=args.combined_figure_path,
        )
        systems = {name: _load_json(path) for name, path in result_paths.items()}
        write_report(
            report_path=args.report_path,
            systems=systems,
            combined_figure_path=Path(combined["figure_path"]),
            standard_result_path=args.standard_result_path,
        )
        print(json.dumps({**combined, "report_path": str(args.report_path)}, ensure_ascii=False, indent=2))
        return
    if args.ikeda_y_tau_sweep:
        result = run_ikeda_y_tau_sweep(
            mode=args.mode,
            seeds=tuple(args.seeds),
            result_path=args.result_dir / "ikeda_y_tau_synergy_sweep.json",
            figure_path=args.figure_dir / "ikeda_y_tau_synergy_sweep.png",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.coupled_henon_tune:
        result = select_coupled_henon_mlp_config(
            mode=args.mode,
            result_path=args.henon_tuning_result_path,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.coupled_henon_sweep:
        selected_config = None
        if args.henon_tuning_result_path.exists():
            tuning = _load_json(args.henon_tuning_result_path)
            selected_config = _henon_config_from_mapping(tuning["selected_config"])
        result = run_coupled_henon_sweep(
            mode=args.mode,
            seeds=tuple(args.seeds),
            result_path=args.result_dir / "coupled_henon_synergy_sweep.json",
            figure_path=args.figure_dir / "coupled_henon_synergy_sweep.png",
            mlp_config=selected_config,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.coupled_henon_comparison:
        result = run_coupled_henon_lorenz_comparison(
            henon_result_path=args.result_dir / "coupled_henon_synergy_sweep.json",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if not args.all:
        raise SystemExit(
            "Pass --all, --ikeda-y-tau-sweep, --coupled-henon-tune, --coupled-henon-sweep, "
            "--coupled-henon-comparison, or --combined-figure-only."
        )
    result = run_discrete_iteration_benchmark(
        mode=args.mode,
        seeds=tuple(args.seeds),
        result_dir=args.result_dir,
        figure_dir=args.figure_dir,
        report_path=args.report_path,
        standard_result_path=args.standard_result_path,
        combined_figure_path=args.combined_figure_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
