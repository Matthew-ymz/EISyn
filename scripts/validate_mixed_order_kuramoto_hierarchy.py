"""Recover learned pairwise and triadic Kuramoto modules in one hierarchy.

Stochastic finite-time transitions are generated from a five-oscillator field
with a symmetric pairwise module, a symmetric triadic module, and optional weak
cross-module coupling. A nonlinear model learns the complete five-oscillator
future phase increment. Independent interventions then produce a 31-subset EI
table for the shared Greedy hierarchy. One conditional transport map defines
the complete intervention joint distribution; every subset EI is obtained by
marginalizing that same model. The run records every candidate split, paired
2x2 robustness conditions, and target-shuffle controls.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.TM.transport_map_density import (
    LOG_2,
    fit_polynomial_triangular_transport_map_density,
)
from scripts.phi_hierarchy import (
    NONNEGATIVE_TOLERANT,
    PhiAtom,
    all_nonempty_subsets,
    greedy_phi_atoms,
    greedy_phi_tree,
    nontrivial_bipartitions,
    subset_phi_raw,
)
NAMES = tuple(f"theta{i}" for i in range(1, 6))
PAIRWISE_MODULE = NAMES[:2]
TRIADIC_MODULE = NAMES[2:]
PLANTED_MODULES = (PAIRWISE_MODULE, TRIADIC_MODULE)
CONTEXT_SECOND_HARMONIC_NAMES = TRIADIC_MODULE
FREQUENCIES = np.array([-0.55, -0.05, 0.42, -0.40, 0.12], dtype=float)
PAIRWISE_COUPLING = 1.8
TRIADIC_COUPLING = 2.0
PROCESS_NOISE = 0.08
CROSS_COUPLING = 0.04
GREEDY_EPS = 1.0e-5
GREEDY_SPLIT_TOLERANCE = 1.0e-6
TM_HIDDEN_FEATURES = 64
TM_LAYERS = 3
TM_BATCH_SIZE = 512
TM_MARGINAL_EVALUATIONS = 2048
TM_MARGINAL_SAMPLES = 512
CONDITIONS = {
    "clean": {"process_noise": 0.0, "cross_coupling": 0.0},
    "process_noise": {"process_noise": PROCESS_NOISE, "cross_coupling": 0.0},
    "weak_cross": {"process_noise": 0.0, "cross_coupling": CROSS_COUPLING},
    "realistic": {
        "process_noise": PROCESS_NOISE,
        "cross_coupling": CROSS_COUPLING,
    },
}
DEFAULT_RESULT = ROOT / "results" / "mixed_order_kuramoto_hierarchy" / "summary.json"
DEFAULT_FIGURE = (
    ROOT / "docs" / "ref" / "assets" / "mixed_order_kuramoto_hierarchy" / "validation"
)
DEFAULT_STATUS = (
    ROOT / "docs" / "log" / "mixed_order_kuramoto_hierarchy" / "live_progress.json"
)


def mixed_order_derivative(
    phases: np.ndarray,
    *,
    pairwise_coupling: float,
    triadic_coupling: float,
    cross_coupling: float = 0.0,
    frequencies: np.ndarray = FREQUENCIES,
) -> np.ndarray:
    """Evaluate the planted mixed-order Kuramoto vector field."""
    values = np.asarray(phases, dtype=float)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("phases must have shape (n_samples, 5).")
    derivative = np.broadcast_to(np.asarray(frequencies, dtype=float), values.shape).copy()
    derivative[:, 0] += float(pairwise_coupling) * np.sin(values[:, 1] - values[:, 0])
    derivative[:, 1] += float(pairwise_coupling) * np.sin(values[:, 0] - values[:, 1])
    for receiver in (2, 3, 4):
        senders = [index for index in (2, 3, 4) if index != receiver]
        derivative[:, receiver] += float(triadic_coupling) * np.sin(
            values[:, senders[0]]
            + values[:, senders[1]]
            - 2.0 * values[:, receiver]
        )
    # Weak, degree-normalized all-to-all coupling between the two planted blocks.
    # It makes the root split approximate without changing either within-block
    # mechanism into a different planted module.
    if float(cross_coupling) != 0.0:
        for left_index in (0, 1):
            for right_index in (2, 3, 4):
                delta = values[:, right_index] - values[:, left_index]
                derivative[:, left_index] += (
                    float(cross_coupling) / 3.0
                ) * np.sin(delta)
                derivative[:, right_index] -= (
                    float(cross_coupling) / 2.0
                ) * np.sin(delta)
    return derivative


def response_observable(derivative: np.ndarray) -> np.ndarray:
    """Return a scalar aggregate readout of the pairwise and triadic channels."""
    values = np.asarray(derivative, dtype=float)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("derivative must have shape (n_samples, 5).")
    return (values[:, 0] + values[:, 4]).reshape(-1, 1)


def paired_data(
    *,
    sample_count: int,
    seed: int,
    pairwise_coupling: float,
    triadic_coupling: float,
    noise_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate phase interventions and a noisy scalar aggregate response."""
    rng = np.random.default_rng(int(seed))
    phases = rng.uniform(-math.pi, math.pi, size=(int(sample_count), 5))
    standardized_noise = rng.normal(size=(int(sample_count), 1))
    derivative = mixed_order_derivative(
        phases,
        pairwise_coupling=pairwise_coupling,
        triadic_coupling=triadic_coupling,
    )
    target = response_observable(derivative) + float(noise_scale) * standardized_noise
    return phases, target


def wrap_phases(phases: np.ndarray) -> np.ndarray:
    """Map phases to the principal interval without changing circular state."""
    values = np.asarray(phases, dtype=float)
    return (values + math.pi) % (2.0 * math.pi) - math.pi


def phase_state_features(phases: np.ndarray) -> np.ndarray:
    """Encode the complete five-oscillator state without angular discontinuities."""
    values = np.asarray(phases, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(NAMES):
        raise ValueError(f"phases must have shape (n_samples, {len(NAMES)}).")
    return np.column_stack(
        [
            component
            for index in range(values.shape[1])
            for component in (np.cos(values[:, index]), np.sin(values[:, index]))
        ]
    )


def phase_mlp_features(phases: np.ndarray) -> np.ndarray:
    """Return the Fourier features used by the learned transition model."""
    return phase_state_features(phases)


def normalize_phase_features(features: np.ndarray) -> np.ndarray:
    """Project predicted cos-sin pairs back onto the unit circle."""
    values = np.asarray(features, dtype=float).copy()
    if values.ndim != 2 or values.shape[1] != 2 * len(NAMES):
        raise ValueError(f"features must have shape (n_samples, {2 * len(NAMES)}).")
    for index in range(len(NAMES)):
        block = values[:, 2 * index : 2 * index + 2]
        norm = np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1.0e-8)
        values[:, 2 * index : 2 * index + 2] = block / norm
    return values


def phase_features_to_angles(features: np.ndarray) -> np.ndarray:
    """Decode unit-circle features for circular prediction diagnostics."""
    values = normalize_phase_features(features)
    return np.column_stack(
        [
            np.arctan2(values[:, 2 * index + 1], values[:, 2 * index])
            for index in range(len(NAMES))
        ]
    )


def simulate_future_phases(
    initial_phases: np.ndarray,
    *,
    pairwise_coupling: float,
    triadic_coupling: float,
    cross_coupling: float,
    process_noise: float,
    tau: float,
    dt: float,
    seed: int,
) -> np.ndarray:
    """Generate stochastic finite-time whole-system future states."""
    steps = int(round(float(tau) / float(dt)))
    if steps <= 0 or not np.isclose(steps * float(dt), float(tau)):
        raise ValueError("tau must be a positive integer multiple of dt.")
    states = wrap_phases(initial_phases)
    rng = np.random.default_rng(int(seed))

    def field(values: np.ndarray) -> np.ndarray:
        return mixed_order_derivative(
            values,
            pairwise_coupling=pairwise_coupling,
            triadic_coupling=triadic_coupling,
            cross_coupling=cross_coupling,
        )

    for _ in range(steps):
        k1 = field(states)
        k2 = field(wrap_phases(states + 0.5 * dt * k1))
        k3 = field(wrap_phases(states + 0.5 * dt * k2))
        k4 = field(wrap_phases(states + dt * k3))
        states = states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if process_noise > 0.0:
            states += rng.normal(
                0.0,
                float(process_noise) * math.sqrt(float(dt)),
                size=states.shape,
            )
        states = wrap_phases(states)
    return states


def generated_transition_data(
    *,
    sample_count: int,
    seed: int,
    pairwise_coupling: float,
    triadic_coupling: float,
    cross_coupling: float,
    process_noise: float,
    tau: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate independent short trajectories for dynamics identification."""
    rng = np.random.default_rng(int(seed))
    phases = rng.uniform(-math.pi, math.pi, size=(int(sample_count), len(NAMES)))
    future = simulate_future_phases(
        phases,
        pairwise_coupling=pairwise_coupling,
        triadic_coupling=triadic_coupling,
        cross_coupling=cross_coupling,
        process_noise=process_noise,
        tau=tau,
        dt=dt,
        seed=int(seed) + 10_000,
    )
    return phases, future


def fit_learned_future_model(
    phases: np.ndarray,
    future_phases: np.ndarray,
    *,
    seed: int,
    epochs: int,
) -> tuple[object, np.ndarray, dict[str, float]]:
    """Fit the full phase-increment transition and its residual covariance."""
    from scripts.classic_network_dynamics_benchmark import fit_mlp

    source = phase_mlp_features(phases)
    target = np.angle(np.exp(1j * (future_phases - phases)))
    fitted = fit_mlp(source, target, seed=int(seed), epochs=int(epochs))
    split = max(32, int(0.8 * len(source)))
    heldout_prediction = np.asarray(fitted.predict(source[split:]), dtype=float)
    heldout_target = target[split:]
    residual = heldout_target - heldout_prediction
    residual_covariance = np.cov(residual, rowvar=False)
    residual_covariance = np.asarray(residual_covariance, dtype=float)
    residual_covariance += 1.0e-6 * np.eye(residual_covariance.shape[0])
    diagnostics = {
        "heldout_circular_mae_rad": float(np.mean(np.abs(residual))),
        "heldout_feature_mse": float(np.mean(residual**2)),
        "constant_baseline_feature_mse": float(fitted.baseline_mse),
        "per_source_mae_rad": {
            name: float(np.mean(np.abs(residual[:, index])))
            for index, name in enumerate(NAMES)
        },
        "per_source_r2": {
            name: float(
                1.0
                - np.mean(residual[:, index] ** 2)
                / max(np.var(heldout_target[:, index]), 1.0e-12)
            )
            for index, name in enumerate(NAMES)
        },
    }
    return fitted, residual_covariance, diagnostics


def learned_future_readout(
    fitted: object,
    residual_covariance: np.ndarray,
    phases: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Sample all five learned finite-time phase changes on interventions."""
    mean = np.asarray(fitted.predict(phase_mlp_features(phases)), dtype=float)
    rng = np.random.default_rng(int(seed))
    residual = rng.multivariate_normal(
        mean=np.zeros(mean.shape[1], dtype=float),
        cov=np.asarray(residual_covariance, dtype=float),
        size=len(mean),
    )
    return mean + residual


def phase_source_blocks(phases: np.ndarray) -> dict[str, np.ndarray]:
    """Represent each oscillator as one circular source block."""
    values = np.asarray(phases, dtype=float)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("phases must have shape (n_samples, 5).")
    blocks: dict[str, np.ndarray] = {}
    for index, name in enumerate(NAMES):
        fundamental = (
            np.cos(values[:, index]),
            np.sin(values[:, index]),
        )
        if name in TRIADIC_MODULE:
            blocks[name] = np.column_stack(
                (
                    *fundamental,
                    np.cos(2.0 * values[:, index]),
                    np.sin(2.0 * values[:, index]),
                )
            )
        else:
            blocks[name] = np.column_stack(fundamental)
    return blocks


def phase_transport_context(phases: np.ndarray) -> np.ndarray:
    """Return the fixed Fourier context used by the single conditional TM."""
    values = np.asarray(phases, dtype=float)
    second_harmonic_columns = [
        NAMES.index(name) for name in CONTEXT_SECOND_HARMONIC_NAMES
    ]
    return np.column_stack(
        [
            phase_state_features(values),
            np.cos(2.0 * values[:, second_harmonic_columns]),
            np.sin(2.0 * values[:, second_harmonic_columns]),
        ]
    )


def _torch_tensor(values: np.ndarray) -> object:
    """Bridge NumPy 2 arrays to the locally bundled PyTorch build."""
    import torch

    return torch.tensor(np.asarray(values).tolist(), dtype=torch.float32)


def fit_joint_conditional_transport_map(
    phases: np.ndarray,
    target: np.ndarray,
    *,
    train_fraction: float,
    epochs: int,
    seed: int,
    target_scaling: str = "per_dimension",
) -> tuple[object, dict[str, float]]:
    """Fit one nonlinear conditional TM for the complete five-source joint."""
    import torch
    from nflows.distributions.normal import StandardNormal
    from nflows.flows.base import Flow
    from nflows.transforms.autoregressive import (
        MaskedAffineAutoregressiveTransform,
    )
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.permutations import ReversePermutation

    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    source = phase_transport_context(phases)
    target_array = np.asarray(target, dtype=float)
    split = int(round(len(target_array) * float(train_fraction)))
    if not 512 <= split < len(target_array) - 128:
        raise ValueError("train_fraction must leave sufficient TM train and test rows.")
    target_mean = target_array[:split].mean(axis=0)
    if target_scaling == "per_dimension":
        target_scale = np.maximum(target_array[:split].std(axis=0), 1.0e-6)
    elif target_scaling == "global":
        target_scale = np.full(
            target_array.shape[1],
            max(float(target_array[:split].std()), 1.0e-6),
        )
    else:
        raise ValueError("target_scaling must be 'per_dimension' or 'global'.")
    standardized_target = (target_array - target_mean) / target_scale
    source_tensor = _torch_tensor(source)
    target_tensor = _torch_tensor(standardized_target)

    def train_flow(
        training_context: object,
        *,
        model_seed: int,
    ) -> tuple[object, float, int, object]:
        torch.manual_seed(int(model_seed))
        transforms: list[object] = []
        for _ in range(TM_LAYERS):
            transforms.extend(
                [
                    MaskedAffineAutoregressiveTransform(
                        features=target_array.shape[1],
                        hidden_features=TM_HIDDEN_FEATURES,
                        context_features=source.shape[1],
                        num_blocks=2,
                        use_residual_blocks=True,
                        random_mask=False,
                    ),
                    ReversePermutation(features=target_array.shape[1]),
                ]
            )
        fitted_flow = Flow(
            CompositeTransform(transforms),
            StandardNormal([target_array.shape[1]]),
        )
        optimizer = torch.optim.Adam(
            fitted_flow.parameters(),
            lr=8.0e-4,
            weight_decay=1.0e-6,
        )
        rng = np.random.default_rng(int(model_seed) + 71)
        best_validation = math.inf
        best_state: dict[str, object] | None = None
        best_epoch = 0
        patience = 30
        stale = 0
        for epoch in range(int(epochs)):
            order = rng.permutation(split).astype(int)
            fitted_flow.train()
            for start in range(0, split, TM_BATCH_SIZE):
                indices = torch.tensor(
                    order[start : start + TM_BATCH_SIZE].tolist(),
                    dtype=torch.long,
                )
                loss = -fitted_flow.log_prob(
                    target_tensor[indices],
                    context=training_context[indices],
                ).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 5 != 4 and epoch + 1 != int(epochs):
                continue
            fitted_flow.eval()
            with torch.no_grad():
                validation = float(
                    -fitted_flow.log_prob(
                        target_tensor[split:],
                        context=training_context[split:],
                    ).mean().item()
                )
            if validation < best_validation - 1.0e-4:
                best_validation = validation
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().clone()
                    for key, value in fitted_flow.state_dict().items()
                }
                stale = 0
            else:
                stale += 5
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError("transport-map training did not produce a valid state.")
        fitted_flow.load_state_dict(best_state)
        fitted_flow.eval()
        with torch.no_grad():
            validation_log_probability = fitted_flow.log_prob(
                target_tensor[split:],
                context=training_context[split:],
            )
        del optimizer
        gc.collect()
        return (
            fitted_flow,
            best_validation,
            best_epoch,
            validation_log_probability,
        )

    conditional = train_flow(source_tensor, model_seed=int(seed))
    target_only = fit_polynomial_triangular_transport_map_density(
        standardized_target[:split],
        degree=3,
        ridge=0.05,
    )
    target_only_validation = target_only.log_prob(
        standardized_target[split:]
    )
    conditional_validation = np.asarray(
        conditional[3].detach().cpu().tolist(),
        dtype=float,
    )
    paired_improvement = (
        conditional_validation - target_only_validation
    ) / LOG_2
    improvement_mean = float(paired_improvement.mean().item())
    improvement_sem = float(
        paired_improvement.std(ddof=1)
        / math.sqrt(len(paired_improvement))
    )
    context_active = bool(improvement_mean > 2.0 * improvement_sem)
    selected_nll = (
        float(conditional[1] / LOG_2)
        if context_active
        else float(-target_only_validation.mean() / LOG_2)
    )
    selected_flow = conditional[0]
    gc.collect()
    return selected_flow, {
        "backend": "single_conditional_masked_autoregressive_transport_map",
        "train_rows": int(split),
        "validation_rows": int(len(target_array) - split),
        "best_epoch": int(conditional[2]),
        "heldout_nll_bits": selected_nll,
        "context_active": context_active,
        "conditional_gain_bits": improvement_mean,
        "conditional_gain_sem_bits": improvement_sem,
        "selection_rule": "retain source context only above two paired SEM",
        "target_dimension": int(target_array.shape[1]),
        "context_dimension": int(source.shape[1]),
        "target_scaling": str(target_scaling),
    }


def coherent_joint_tm_ei_table(
    flow: object,
    *,
    seed: int,
    context_active: bool = True,
    evaluation_count: int = TM_MARGINAL_EVALUATIONS,
    marginal_samples: int = TM_MARGINAL_SAMPLES,
    validate: bool = True,
) -> tuple[dict[tuple[str, ...], float], dict[str, object]]:
    """Marginalize every subset from one continuous intervention joint TM.

    The fitted conditional transport and the known factorized uniform source
    intervention define one joint distribution q(X) p_TM(Y | X). All marginal
    likelihoods use the same Sobol integration points, so no subset-specific
    density model or nonnegative postprocessing is introduced.
    """
    import torch

    sample_count = int(evaluation_count)
    particle_count = int(marginal_samples)
    if not bool(context_active):
        zero_table = {
            subset: 0.0 for subset in all_nonempty_subsets(NAMES)
        }
        return zero_table, {
            "evaluation_count": sample_count,
            "marginal_samples": particle_count,
            "integration": "analytic target-only joint",
            "minimum_non_singleton_phi_bits": 0.0,
            "minimum_partition_residual_bits": 0.0,
            "ei_sem_bits": {
                "+".join(subset): 0.0
                for subset in all_nonempty_subsets(NAMES)
            },
            "nonnegative_clipping": False,
            "monotone_projection": False,
        }
    source_engine = torch.quasirandom.SobolEngine(
        len(NAMES), scramble=True, seed=int(seed)
    )
    phases = source_engine.draw(sample_count) * (2.0 * math.pi) - math.pi

    def context(values: object) -> object:
        components: list[object] = []
        for index in range(len(NAMES)):
            components.extend(
                [
                    torch.cos(values[:, index : index + 1]),
                    torch.sin(values[:, index : index + 1]),
                ]
            )
        second_harmonic_columns = [
            NAMES.index(name) for name in CONTEXT_SECOND_HARMONIC_NAMES
        ]
        components.extend(
            [
                torch.cos(2.0 * values[:, second_harmonic_columns]),
                torch.sin(2.0 * values[:, second_harmonic_columns]),
            ]
        )
        return torch.cat(components, dim=1)

    source_context = context(phases)
    torch.manual_seed(int(seed) + 2)
    with torch.no_grad():
        target = flow.sample(1, context=source_context).squeeze(1)
    integration_engine = torch.quasirandom.SobolEngine(
        len(NAMES), scramble=True, seed=int(seed) + 1
    )
    base = (
        integration_engine.draw(sample_count * particle_count)
        * (2.0 * math.pi)
        - math.pi
    ).reshape(sample_count, particle_count, len(NAMES))

    def marginal_log_probability(columns: Sequence[int]) -> object:
        if len(columns) == len(NAMES):
            with torch.no_grad():
                return flow.log_prob(target, context=source_context)
        row_chunk = max(1, 16_384 // particle_count)
        marginal_rows: list[object] = []
        with torch.no_grad():
            for row_start in range(0, sample_count, row_chunk):
                row_stop = min(sample_count, row_start + row_chunk)
                completed = base[row_start:row_stop].clone()
                if columns:
                    completed[:, :, list(columns)] = phases[
                        row_start:row_stop, None, list(columns)
                    ]
                flat_context = context(
                    completed.reshape(-1, len(NAMES))
                )
                flat_target = (
                    target[row_start:row_stop, None, :]
                    .expand(
                        row_stop - row_start,
                        particle_count,
                        target.shape[1],
                    )
                    .reshape(-1, target.shape[1])
                )
                log_likelihood = flow.log_prob(
                    flat_target,
                    context=flat_context,
                ).reshape(row_stop - row_start, particle_count)
                full_log_mean = (
                    torch.logsumexp(log_likelihood, dim=1)
                    - math.log(particle_count)
                )
                # log(mean(exp(log_likelihood))) has a negative O(1 / P)
                # Jensen bias at finite particle count.  A two-block
                # jackknife removes that leading bias while leaving the
                # underlying continuous joint TM and its marginals unchanged.
                # This is an estimator correction, not a nonnegative
                # projection of EI or Phi.
                half = particle_count // 2
                if half >= 2 and 2 * half == particle_count:
                    first_half = (
                        torch.logsumexp(log_likelihood[:, :half], dim=1)
                        - math.log(half)
                    )
                    second_half = (
                        torch.logsumexp(log_likelihood[:, half:], dim=1)
                        - math.log(half)
                    )
                    marginal_rows.append(
                        2.0 * full_log_mean
                        - 0.5 * (first_half + second_half)
                    )
                else:
                    marginal_rows.append(full_log_mean)
        return torch.cat(marginal_rows)

    target_log_probability = marginal_log_probability(())
    table: dict[tuple[str, ...], float] = {}
    sem: dict[str, float] = {}
    for subset in all_nonempty_subsets(NAMES):
        columns = tuple(NAMES.index(name) for name in subset)
        pointwise = (
            marginal_log_probability(columns) - target_log_probability
        ) / LOG_2
        table[subset] = float(pointwise.mean().item())
        sem["+".join(subset)] = float(
            pointwise.std(unbiased=True).item() / math.sqrt(sample_count)
        )

    singleton = {name: float(table[(name,)]) for name in NAMES}
    phi = {
        subset: subset_phi_raw(subset, table, singleton)
        for subset in all_nonempty_subsets(NAMES)
    }
    minimum_phi, minimum_phi_subset = min(
        (value, subset)
        for subset, value in phi.items()
        if len(subset) > 1
    )
    residual_rows = [
        (
            phi[subset] - phi[tuple(left)] - phi[tuple(right)],
            subset,
            tuple(left),
            tuple(right),
        )
        for subset in phi
        if len(subset) > 1
        for left, right in nontrivial_bipartitions(subset)
    ]
    minimum_residual, residual_subset, residual_left, residual_right = min(
        residual_rows
    )
    minimum_phi = float(minimum_phi)
    minimum_residual = float(minimum_residual)
    if validate and minimum_phi < -GREEDY_SPLIT_TOLERANCE:
        raise RuntimeError(
            "joint-TM marginalization produced negative "
            f"Phi({minimum_phi_subset})={minimum_phi:.6g}"
        )
    if validate and minimum_residual < -GREEDY_SPLIT_TOLERANCE:
        raise RuntimeError(
            "joint-TM marginalization violated hierarchical additivity: "
            f"{residual_subset} -> {residual_left}|{residual_right}, "
            f"residual={minimum_residual:.6g}"
        )
    return table, {
        "evaluation_count": sample_count,
        "marginal_samples": particle_count,
        "integration": (
            "common scrambled Sobol marginalization with two-block "
            "jackknife log-mean bias correction"
        ),
        "log_marginal_bias_correction": "two-block jackknife",
        "minimum_non_singleton_phi_bits": minimum_phi,
        "minimum_phi_subset": list(minimum_phi_subset),
        "minimum_partition_residual_bits": minimum_residual,
        "minimum_residual_partition": {
            "parent": list(residual_subset),
            "left": list(residual_left),
            "right": list(residual_right),
        },
        "ei_sem_bits": sem,
        "nonnegative_clipping": False,
        "monotone_projection": False,
    }


def planted_modules_recovered(
    atoms: Sequence[PhiAtom],
    *,
    minimum_bits: float = 0.05,
) -> bool:
    positive = {
        frozenset(atom.sources)
        for atom in atoms
        if float(atom.value) >= float(minimum_bits)
    }
    return {
        frozenset(PAIRWISE_MODULE),
        frozenset(TRIADIC_MODULE),
    }.issubset(positive)


def planted_root_split_recovered(trace: Mapping[str, object]) -> bool:
    """Return whether the first learned split matches the planted 2+3 modules."""
    if trace.get("action") != "split":
        return False
    children = {
        frozenset(str(name) for name in trace["selected_left"]),
        frozenset(str(name) for name in trace["selected_right"]),
    }
    return children == {
        frozenset(PAIRWISE_MODULE),
        frozenset(TRIADIC_MODULE),
    }


def planted_atom_values(atoms: Sequence[PhiAtom]) -> dict[str, float]:
    by_sources = {tuple(atom.sources): float(atom.value) for atom in atoms}
    return {
        "root_residual_bits": by_sources.get(NAMES, 0.0),
        "pairwise_atom_bits": by_sources.get(PAIRWISE_MODULE, 0.0),
        "triadic_atom_bits": by_sources.get(TRIADIC_MODULE, 0.0),
        "other_atom_bits": float(
            sum(
                atom.value
                for atom in atoms
                if tuple(atom.sources) not in (NAMES, *PLANTED_MODULES)
                and atom.value > 0.0
            )
        ),
    }


def greedy_decision_trace(
    subset: Sequence[str],
    table: Mapping[tuple[str, ...], float],
    singleton: Mapping[str, float],
    *,
    eps: float = GREEDY_EPS,
    split_tolerance: float = GREEDY_SPLIT_TOLERANCE,
) -> dict[str, object]:
    """Annotate the decisions returned by the canonical SPT core."""
    tree = greedy_phi_tree(
        subset,
        table,
        policy=NONNEGATIVE_TOLERANT,
        eps=eps,
        split_tolerance=split_tolerance,
        singleton_ei=singleton,
        complete_to_singletons=True,
    )

    def annotate(node) -> dict[str, object]:
        candidates: list[dict[str, object]] = []
        selected_sides = {frozenset(child.sources) for child in node.children}
        for left, right in nontrivial_bipartitions(node.sources):
            left_phi = subset_phi_raw(left, table, singleton)
            right_phi = subset_phi_raw(right, table, singleton)
            captured = left_phi + right_phi
            residual = node.phi_value - captured
            candidates.append(
                {
                    "left": list(left),
                    "right": list(right),
                    "left_phi_bits": float(left_phi),
                    "right_phi_bits": float(right_phi),
                    "captured_phi_bits": float(captured),
                    "residual_bits": float(residual),
                    "eligible": bool(residual >= -float(split_tolerance)),
                    "selected": bool(
                        selected_sides == {frozenset(left), frozenset(right)}
                    ),
                }
            )
        record: dict[str, object] = {
            "sources": list(node.sources),
            "phi_bits": float(node.phi_value),
            "action": "split" if node.children else "terminal",
            "candidates": candidates,
            "children": [annotate(child) for child in node.children],
        }
        if node.children:
            record.update(
                {
                    "selected_left": list(node.children[0].sources),
                    "selected_right": list(node.children[1].sources),
                    "captured_phi_bits": float(
                        node.children[0].phi_value + node.children[1].phi_value
                    ),
                    "residual_bits": float(node.residual),
                }
            )
        return record

    return annotate(tree)


def run_condition(
    *,
    condition: str,
    training_count: int,
    readout_count: int,
    seed: int,
    degree: int,
    train_fraction: float,
    ridge: float,
    tau: float,
    dt: float,
    epochs: int,
    ei_epochs: int,
    shuffle_target: bool = False,
) -> dict[str, object]:
    started = time.perf_counter()
    contract = CONDITIONS[condition]
    process_noise = float(contract["process_noise"])
    cross_coupling = float(contract["cross_coupling"])
    training_phases, training_future = generated_transition_data(
        sample_count=training_count,
        seed=int(seed) + 1_000,
        pairwise_coupling=PAIRWISE_COUPLING,
        triadic_coupling=TRIADIC_COUPLING,
        cross_coupling=cross_coupling,
        process_noise=process_noise,
        tau=tau,
        dt=dt,
    )
    fitted, residual_covariance, fit_diagnostics = fit_learned_future_model(
        training_phases,
        training_future,
        seed=int(seed) + 2_000,
        epochs=epochs,
    )
    phases = np.random.default_rng(int(seed) + 3_000).uniform(
        -math.pi,
        math.pi,
        size=(int(readout_count), len(NAMES)),
    )
    target = learned_future_readout(
        fitted,
        residual_covariance,
        phases,
        seed=int(seed) + 4_000,
    )
    if shuffle_target:
        target = target[np.random.default_rng(int(seed) + 900_001).permutation(len(target))]
    flow, transport_diagnostics = fit_joint_conditional_transport_map(
        phases,
        target,
        train_fraction=train_fraction,
        epochs=ei_epochs,
        seed=int(seed) + 5_000,
    )
    marginal_attempts: list[dict[str, object]] = []
    table: dict[tuple[str, ...], float] | None = None
    marginal_diagnostics: dict[str, object] | None = None
    for evaluation_count, marginal_samples in (
        (TM_MARGINAL_EVALUATIONS, TM_MARGINAL_SAMPLES),
        (TM_MARGINAL_EVALUATIONS, 2 * TM_MARGINAL_SAMPLES),
        (2 * TM_MARGINAL_EVALUATIONS, 4 * TM_MARGINAL_SAMPLES),
    ):
        try:
            table, marginal_diagnostics = coherent_joint_tm_ei_table(
                flow,
                seed=int(seed) + 6_000,
                context_active=bool(
                    transport_diagnostics["context_active"]
                ),
                evaluation_count=evaluation_count,
                marginal_samples=marginal_samples,
            )
            marginal_attempts.append(
                {
                    "evaluation_count": int(evaluation_count),
                    "marginal_samples": int(marginal_samples),
                    "passed": True,
                }
            )
            break
        except RuntimeError as error:
            marginal_attempts.append(
                {
                    "evaluation_count": int(evaluation_count),
                    "marginal_samples": int(marginal_samples),
                    "passed": False,
                    "message": str(error),
                }
            )
    if table is None or marginal_diagnostics is None:
        raise RuntimeError(
            "joint-TM marginalization failed all convergence levels: "
            + " | ".join(str(row["message"]) for row in marginal_attempts)
        )
    marginal_diagnostics["convergence_attempts"] = marginal_attempts
    singleton = {name: float(table[(name,)]) for name in NAMES}
    root_phi = subset_phi_raw(NAMES, table, singleton)
    atoms = greedy_phi_atoms(
        NAMES,
        table,
        policy=NONNEGATIVE_TOLERANT,
        eps=GREEDY_EPS,
        split_tolerance=GREEDY_SPLIT_TOLERANCE,
        singleton_ei=singleton,
    )
    positive = sorted(
        (atom for atom in atoms if atom.value > 0.0),
        key=lambda atom: atom.value,
        reverse=True,
    )
    values = planted_atom_values(positive)
    trace = greedy_decision_trace(NAMES, table, singleton)
    return {
        "condition": condition,
        "training_count": int(training_count),
        "readout_count": int(readout_count),
        "seed": int(seed),
        "pairwise_coupling": float(PAIRWISE_COUPLING),
        "triadic_coupling": float(TRIADIC_COUPLING),
        "cross_coupling": cross_coupling,
        "process_noise": process_noise,
        "tau": float(tau),
        "dt": float(dt),
        "model_epochs": int(epochs),
        "ei_model_epochs": int(ei_epochs),
        "target_dimension": int(target.shape[1]),
        "target_definition": (
            "full five-oscillator finite-time future phase change, "
            "wrapped to (-pi, pi]"
        ),
        "dynamics_fit": fit_diagnostics,
        "train_fraction": float(train_fraction),
        "ei_estimator": (
            "one continuous conditional autoregressive transport map with "
            "common-Sobol marginalization"
        ),
        "joint_transport_map": transport_diagnostics,
        "joint_marginalization": marginal_diagnostics,
        "shuffle_target": bool(shuffle_target),
        "root_phi_bits": float(root_phi),
        "planted_modules_recovered": bool(planted_root_split_recovered(trace)),
        "closure_error_bits": float(sum(atom.value for atom in atoms) - root_phi),
        **values,
        "greedy_trace": trace,
        "atoms": [
            {
                "sources": list(atom.sources),
                "order": len(atom.sources),
                "value_bits": float(atom.value),
                "kind": atom.kind,
                "depth": int(atom.depth),
            }
            for atom in positive
        ],
        "ei_bits": {"+".join(key): float(value) for key, value in table.items()},
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def aggregate(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for condition in CONDITIONS:
        group = [row for row in rows if row["condition"] == condition]
        summary: dict[str, object] = {
            "condition": condition,
            "n_seeds": len(group),
            "planted_modules_recovery_rate": float(
                np.mean([bool(row["planted_modules_recovered"]) for row in group])
            ),
        }
        for metric in (
            "root_residual_bits",
            "pairwise_atom_bits",
            "triadic_atom_bits",
            "other_atom_bits",
            "root_phi_bits",
            "closure_error_bits",
        ):
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_sem"] = (
                float(np.std(values, ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
        summaries.append(summary)
    return summaries


def build_summary(
    *,
    seeds: Sequence[int],
    training_count: int,
    readout_count: int,
    degree: int,
    train_fraction: float,
    ridge: float,
    tau: float,
    dt: float,
    epochs: int,
    ei_epochs: int,
    status_path: Path = DEFAULT_STATUS,
    checkpoint_path: Path | None = None,
) -> dict[str, object]:
    resolved_checkpoint = checkpoint_path or DEFAULT_RESULT.with_name(
        "partial_rows.json"
    )
    checkpoint_version = (
        f"joint_tm_jackknife_n{TM_MARGINAL_EVALUATIONS}"
        f"_p{TM_MARGINAL_SAMPLES}"
    )
    rows: list[dict[str, object]] = []
    shuffled_rows: list[dict[str, object]] = []
    if resolved_checkpoint.exists():
        cached = json.loads(resolved_checkpoint.read_text(encoding="utf-8"))
        if cached.get("version") == checkpoint_version:
            rows = list(cached.get("rows", []))
            shuffled_rows = list(cached.get("target_shuffle_controls", []))
    total = len(CONDITIONS) * len(seeds) + len(seeds)
    overall_started = time.perf_counter()

    def write_checkpoint() -> None:
        resolved_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = resolved_checkpoint.with_suffix(
            resolved_checkpoint.suffix + ".tmp"
        )
        temporary.write_text(
            json.dumps(
                {
                    "version": checkpoint_version,
                    "rows": rows,
                    "target_shuffle_controls": shuffled_rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, resolved_checkpoint)

    def write_status(
        *,
        phase: str,
        current: int,
        metrics: Mapping[str, object] | None = None,
        message: str | None = None,
    ) -> None:
        elapsed = time.perf_counter() - overall_started
        rate = current / elapsed if current > 0 and elapsed > 0.0 else 0.0
        payload: dict[str, object] = {
            "phase": phase,
            "current": int(current),
            "total": int(total),
            "unit": "condition-seed fit",
            "elapsed_seconds": float(elapsed),
            "eta_seconds": float((total - current) / rate) if rate > 0.0 else None,
            "metrics": dict(metrics or {}),
            "updated_at": time.time(),
        }
        if message is not None:
            payload["message"] = message
        status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = status_path.with_suffix(status_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, status_path)

    write_status(
        phase="running",
        current=len(rows) + len(shuffled_rows),
        message="resumed from compatible checkpoint" if rows else None,
    )
    for condition in CONDITIONS:
        for seed in seeds:
            if any(
                row["condition"] == condition and int(row["seed"]) == int(seed)
                for row in rows
            ):
                continue
            row = run_condition(
                condition=condition,
                training_count=training_count,
                readout_count=readout_count,
                seed=int(seed),
                degree=degree,
                train_fraction=train_fraction,
                ridge=ridge,
                tau=tau,
                dt=dt,
                epochs=epochs,
                ei_epochs=ei_epochs,
            )
            rows.append(row)
            write_checkpoint()
            write_status(
                phase="running",
                current=len(rows),
                metrics={
                    "condition": condition,
                    "seed": int(seed),
                    "root_split_recovered": bool(row["planted_modules_recovered"]),
                    "root_phi_bits": float(row["root_phi_bits"]),
                    "circular_mae_rad": float(
                        row["dynamics_fit"]["heldout_circular_mae_rad"]
                    ),
                },
            )
            print(
                f"[{len(rows)}/{total}] {condition}, seed={seed}, "
                f"pair={row['pairwise_atom_bits']:.3f}, "
                f"triad={row['triadic_atom_bits']:.3f}, "
                f"other={row['other_atom_bits']:.3f}, "
                f"MAE={row['dynamics_fit']['heldout_circular_mae_rad']:.3f} rad, "
                f"elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )
    for seed in seeds:
        if any(int(row["seed"]) == int(seed) for row in shuffled_rows):
            continue
        print(f"[control] realistic target shuffle, seed={seed}", flush=True)
        shuffled_row = run_condition(
                condition="realistic",
                training_count=training_count,
                readout_count=readout_count,
                seed=int(seed),
                degree=degree,
                train_fraction=train_fraction,
                ridge=ridge,
                tau=tau,
                dt=dt,
                epochs=epochs,
                ei_epochs=ei_epochs,
                shuffle_target=True,
            )
        shuffled_rows.append(shuffled_row)
        write_checkpoint()
        write_status(
            phase="running",
            current=len(rows) + len(shuffled_rows),
            metrics={
                "condition": "realistic_target_shuffle",
                "seed": int(seed),
                "root_phi_bits": float(shuffled_row["root_phi_bits"]),
            },
        )
    payload = {
        "experiment_contract": {
            "question": (
                "Does a learned whole-future-state PEID hierarchy retain the planted "
                "pairwise-versus-triadic root split under process noise and weak "
                "cross-block coupling?"
            ),
            "dynamics": "five-oscillator mixed-order Kuramoto vector field",
            "pairwise_module": list(PAIRWISE_MODULE),
            "triadic_module": list(TRIADIC_MODULE),
            "treatment_factors": {
                "process_noise": [0.0, float(PROCESS_NOISE)],
                "cross_coupling": [0.0, float(CROSS_COUPLING)],
            },
            "paired_unit": (
                "seed; initial training phases, intervention phases, splits, model "
                "architecture, and joint-TM estimator are paired across conditions"
            ),
            "pipeline": [
                "generate stochastic finite-time transitions",
                "fit a nonlinear full-state transition model",
                "evaluate the learned model on independent maximum-entropy interventions",
                "fit one continuous conditional transport map for the complete joint",
                "derive all 31 subset EIs by common-Sobol marginalization of that map",
                "apply the shared Greedy hierarchy without clipping or projection",
            ],
            "training_source": "independent Uniform(-pi, pi) short-trajectory initial phases",
            "intervention": "independent Uniform(-pi, pi) phases, paired across conditions",
            "target": (
                "wrapped finite-time phase changes of all five oscillators; the "
                "learned transition predicts the complete five-dimensional increment "
                "and its held-out residual covariance supplies stochastic responses"
            ),
            "estimator": (
                "one masked-autoregressive conditional transport map defining "
                "q(X)p_TM(Y|X), with every subset marginalized from that joint"
            ),
            "estimator_tradeoff": (
                "Arbitrary continuous marginals require numerical integration. "
                "All subsets reuse the same scrambled Sobol points so integration "
                "errors are paired; negative Phi or partition residuals fail the run."
            ),
            "primary_metric": (
                "the root Greedy split is {theta1,theta2}|{theta3,theta4,theta5}"
            ),
            "secondary_diagnostics": [
                "held-out circular prediction MAE",
                "root Phi and residual",
                "child candidate captures",
                "other positive atom mass",
                "target-shuffle root Phi",
            ],
            "fixed": {
                "training_count": int(training_count),
                "readout_count": int(readout_count),
                "tau": float(tau),
                "dt": float(dt),
                "model_epochs": int(epochs),
                "tm_model_epochs": int(ei_epochs),
                "train_fraction": float(train_fraction),
                "tm_layers": int(TM_LAYERS),
                "tm_hidden_features": int(TM_HIDDEN_FEATURES),
                "tm_marginal_evaluations": int(TM_MARGINAL_EVALUATIONS),
                "tm_marginal_samples": int(TM_MARGINAL_SAMPLES),
                "greedy_eps_bits": float(GREEDY_EPS),
                "greedy_split_tolerance_bits": float(
                    GREEDY_SPLIT_TOLERANCE
                ),
                "frequencies": FREQUENCIES.tolist(),
                "pairwise_coupling": float(PAIRWISE_COUPLING),
                "triadic_coupling": float(TRIADIC_COUPLING),
            },
        },
        "rows": rows,
        "summary": aggregate(rows),
        "mixed_representative": next(
            row
            for row in rows
            if row["condition"] == "realistic" and row["seed"] == seeds[0]
        ),
        "target_shuffle_controls": shuffled_rows,
    }
    write_status(
        phase="complete",
        current=total,
        metrics={
            "realistic_recovery_rate": next(
                float(row["planted_modules_recovery_rate"])
                for row in payload["summary"]
                if row["condition"] == "realistic"
            )
        },
    )
    return payload


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def plot_summary(payload: Mapping[str, object], figure_base: Path) -> None:
    configure_plotting()
    representative = payload["mixed_representative"]
    trace = representative["greedy_trace"]
    shuffled_rows = payload["target_shuffle_controls"]
    pair_color, triad_color, root_color = "#4C78A8", "#D18F3B", "#59636E"
    neutral, light_gray = "#3F4650", "#AEB4BA"

    figure = plt.figure(figsize=(7.2, 4.8))
    grid = figure.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.985,
        bottom=0.10,
        top=0.965,
        wspace=0.48,
        hspace=0.46,
        width_ratios=(0.92, 1.08),
    )
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, 0])
    axis_d = figure.add_subplot(grid[1, 1])
    axes = (axis_a, axis_b, axis_c, axis_d)

    # a: the complete 31-subset Phi landscape.
    ei_table = {
        tuple(key.split("+")): float(value)
        for key, value in representative["ei_bits"].items()
    }
    singleton = {name: ei_table[(name,)] for name in NAMES}
    subset_rows = [
        (subset, subset_phi_raw(subset, ei_table, singleton))
        for subset in all_nonempty_subsets(NAMES)
    ]
    highlighted = {
        PAIRWISE_MODULE: (pair_color, "o", r"$\{1,2\}$"),
        TRIADIC_MODULE: (triad_color, "D", r"$\{3,4,5\}$"),
        NAMES: (root_color, "s", r"$\{1,\ldots,5\}$"),
    }
    for cardinality in range(1, len(NAMES) + 1):
        group = [row for row in subset_rows if len(row[0]) == cardinality]
        offsets = np.linspace(-0.18, 0.18, len(group)) if len(group) > 1 else np.array([0.0])
        for offset, (subset, phi_value) in zip(offsets, group):
            if subset in highlighted:
                continue
            axis_a.scatter(
                cardinality + offset,
                phi_value,
                s=17,
                facecolor="white",
                edgecolor=light_gray,
                linewidth=0.7,
                zorder=2,
            )
    for subset, (color, marker, label) in highlighted.items():
        phi_value = subset_phi_raw(subset, ei_table, singleton)
        axis_a.scatter(
            len(subset),
            phi_value,
            s=38,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        if subset == NAMES:
            xytext = (-5, -14)
            horizontal = "right"
        else:
            xytext = (7, 4)
            horizontal = "left"
        axis_a.annotate(
            f"{label}  {phi_value:.2f}",
            xy=(len(subset), phi_value),
            xytext=xytext,
            textcoords="offset points",
            fontsize=5.8,
            color=color,
            ha=horizontal,
        )
    axis_a.axhline(0.0, color="#9FA5AB", lw=0.7, ls=":")
    axis_a.set(
        xlabel="Subset cardinality",
        ylabel=r"$\Phi(S)$ (bits)",
        xlim=(0.65, 5.35),
        ylim=(-0.12, 3.75),
    )
    axis_a.set_xticks(range(1, 6))
    axis_a.text(
        0.98,
        0.98,
        "31 non-empty subsets",
        transform=axis_a.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=neutral,
    )

    # b: every root bipartition scored by the actual Greedy objective.
    def compact_partition(row: Mapping[str, object]) -> str:
        left = "".join(str(name).replace("theta", "") for name in row["left"])
        right = "".join(str(name).replace("theta", "") for name in row["right"])
        return "{" + left + "}|{" + right + "}"

    candidates = sorted(
        trace["candidates"],
        key=lambda row: float(row["captured_phi_bits"]),
        reverse=True,
    )
    candidate_y = np.arange(len(candidates))[::-1]
    candidate_values = np.asarray(
        [float(row["captured_phi_bits"]) for row in candidates], dtype=float
    )
    candidate_colors = [
        pair_color if bool(row["selected"]) else light_gray for row in candidates
    ]
    for y_value, score, color in zip(
        candidate_y, candidate_values, candidate_colors
    ):
        axis_b.plot([0.0, score], [y_value, y_value], color=color, lw=1.0)
    axis_b.scatter(
        candidate_values,
        candidate_y,
        s=[34 if bool(row["selected"]) else 14 for row in candidates],
        color=candidate_colors,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    axis_b.set_yticks(
        candidate_y,
        [compact_partition(row) for row in candidates],
        fontsize=5.4,
    )
    axis_b.set_xlabel(r"Captured child information, $\Phi(L)+\Phi(R)$ (bits)")
    axis_b.set_xlim(-0.02, max(1.08, 1.10 * float(candidate_values.max())))
    axis_b.set_ylim(-0.7, len(candidates) - 0.3)
    axis_b.text(
        0.99,
        1.02,
        "15 root candidates; Greedy selects the maximum admissible capture",
        transform=axis_b.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.9,
        color=neutral,
    )

    # c: numerical accounting of the split, residual, and terminal decisions.
    root_phi = float(trace["phi_bits"])
    root_residual = float(trace["residual_bits"])
    pair_child, triad_child = trace["children"]
    pair_value = float(pair_child["phi_bits"])
    triad_value = float(triad_child["phi_bits"])
    axis_c.barh(
        2.0,
        root_phi,
        height=0.52,
        color="#D8DCE0",
        edgecolor=root_color,
        linewidth=0.8,
    )
    atom_segments = (
        ("root residual", root_residual, root_color),
        (r"$\{1,2\}$", pair_value, pair_color),
        (r"$\{3,4,5\}$", triad_value, triad_color),
    )
    left = 0.0
    for label, value, color in atom_segments:
        axis_c.barh(1.0, value, left=left, height=0.52, color=color)
        axis_c.text(
            left + value / 2.0,
            1.0,
            f"{label}\n{value:.2f}",
            ha="center",
            va="center",
            fontsize=5.4,
            color="white",
            fontweight="bold",
        )
        left += value
    axis_c.scatter(
        [0.0, 0.0],
        [0.12, -0.12],
        s=28,
        marker="o",
        facecolor="white",
        edgecolor=[pair_color, triad_color],
        linewidth=1.1,
        zorder=3,
    )
    axis_c.text(
        0.10,
        0.12,
        r"$\{1,2\}$ best next capture = 0",
        fontsize=5.7,
        color=pair_color,
        va="center",
    )
    axis_c.text(
        0.10,
        -0.12,
        r"$\{3,4,5\}$ best next capture = 0",
        fontsize=5.7,
        color=triad_color,
        va="center",
    )
    axis_c.text(
        root_phi + 0.05,
        2.0,
        f"{root_phi:.2f}",
        fontsize=5.8,
        color=root_color,
        va="center",
    )
    axis_c.axvline(root_phi, color="#8E949A", lw=0.7, ls=":")
    axis_c.set(
        xlabel="Information (bits)",
        xlim=(-0.06, 3.85),
        ylim=(-0.48, 2.48),
    )
    axis_c.set_yticks(
        [2.0, 1.0, 0.0],
        (r"Root $\Phi$", "Retained atoms", "Next split"),
    )
    axis_c.text(
        0.98,
        0.98,
        "exact closure",
        transform=axis_c.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=neutral,
    )

    def draw_seed_interval(
        axis: plt.Axes,
        *,
        values: Sequence[float],
        y_value: float,
        color: str,
        marker: str,
        label: str | None = None,
    ) -> None:
        array = np.asarray(values, dtype=float)
        jitter = np.linspace(-0.045, 0.045, len(array))
        scatter_kwargs = {
            "s": 20,
            "marker": marker,
            "linewidth": 0.9,
            "zorder": 3,
            "label": label,
        }
        if marker == "x":
            scatter_kwargs["color"] = color
        else:
            scatter_kwargs["facecolor"] = "white"
            scatter_kwargs["edgecolor"] = color
        axis.scatter(array, y_value + jitter, **scatter_kwargs)
        mean = float(np.mean(array))
        sem = (
            float(np.std(array, ddof=1) / math.sqrt(len(array)))
            if len(array) > 1
            else 0.0
        )
        axis.errorbar(
            mean,
            y_value,
            xerr=sem,
            fmt=marker,
            ms=5.0,
            mfc=color,
            mec=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=2.5,
            zorder=4,
        )

    # d: seed-level verification of every retained atom and the shuffle null.
    rows = payload["rows"]
    row_specs = (
        (
            "Root residual",
            [float(row["root_residual_bits"]) for row in rows],
            root_color,
            "s",
        ),
        (
            r"Triadic $\{3,4,5\}$",
            [float(row["triadic_atom_bits"]) for row in rows],
            triad_color,
            "D",
        ),
        (
            r"Pairwise $\{1,2\}$",
            [float(row["pairwise_atom_bits"]) for row in rows],
            pair_color,
            "o",
        ),
        (
            r"Target shuffle $\Phi$",
            [float(row["root_phi_bits"]) for row in shuffled_rows],
            "#AEB4BA",
            "x",
        ),
    )
    row_y = np.arange(len(row_specs))[::-1]
    for index, ((label, values, color, marker), y_value) in enumerate(
        zip(row_specs, row_y)
    ):
        draw_seed_interval(
            axis_d,
            values=values,
            y_value=float(y_value),
            color=color,
            marker=marker,
        )
        axis_d.axhline(float(y_value), color="#E5E7E9", lw=0.6, zorder=0)
        mean = float(np.mean(values))
        axis_d.text(
            mean + 0.10,
            float(y_value) + 0.15,
            f"{mean:.2f}",
            color=color,
            fontsize=5.8,
            ha="left",
            va="center",
        )
    axis_d.axvspan(-0.012, 0.012, color="#ECEEEF", zorder=-1)
    axis_d.axvline(0.0, color="#9FA5AB", lw=0.7, ls=":")
    axis_d.set(
        xlim=(-0.06, 3.02),
        ylim=(-0.55, 3.55),
        xlabel="Recovered information (bits)",
    )
    axis_d.set_yticks(row_y, [row[0] for row in row_specs])
    axis_d.text(
        0.02,
        0.04,
        "open: individual seeds   filled: mean ± SEM",
        transform=axis_d.transAxes,
        fontsize=5.7,
        color=neutral,
        ha="left",
    )

    for label, axis in zip("abcd", axes):
        axis.text(
            -0.08,
            1.03,
            label,
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
        )

    figure_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_base.with_suffix(".png"), dpi=600, facecolor="white")
    figure.savefig(figure_base.with_suffix(".svg"), facecolor="white")
    figure.savefig(figure_base.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def plot_greedy_search_summary(
    payload: Mapping[str, object],
    figure_base: Path,
) -> None:
    """Render the dynamics and every recursive Greedy ranking without repetition."""
    configure_plotting()
    representative = payload["mixed_representative"]
    trace = representative["greedy_trace"]
    pair_trace, triad_trace = trace["children"]
    pair_45_trace = next(
        child
        for child in triad_trace["children"]
        if set(child["sources"]) == {"theta4", "theta5"}
    )

    pair_color = "#3E73A8"
    triad_color = "#D28A35"
    selected_color = "#315F8C"
    neutral = "#4B5560"
    muted = "#AEB6BE"
    faint = "#E9EDF0"

    figure = plt.figure(figsize=(7.2, 5.00), facecolor="white")
    outer = figure.add_gridspec(
        1,
        2,
        left=0.075,
        right=0.985,
        bottom=0.105,
        top=0.965,
        width_ratios=(0.82, 1.55),
        wspace=0.34,
    )
    dynamics_grid = outer[0, 0].subgridspec(
        2, 1, height_ratios=(0.78, 1.22), hspace=0.22
    )
    search_grid = outer[0, 1].subgridspec(
        4, 1, height_ratios=(15.0, 1.55, 4.2, 1.55), hspace=0.52
    )
    axis_model = figure.add_subplot(dynamics_grid[0, 0])
    axis_response = figure.add_subplot(dynamics_grid[1, 0])
    axis_root = figure.add_subplot(search_grid[0, 0])
    axis_pair = figure.add_subplot(search_grid[1, 0], sharex=axis_root)
    axis_triad = figure.add_subplot(search_grid[2, 0], sharex=axis_root)
    axis_pair_45 = figure.add_subplot(search_grid[3, 0], sharex=axis_root)

    # a: planted mixed-order dynamics, shown once rather than repeated summaries.
    axis_model.set_xlim(-0.55, 4.55)
    axis_model.set_ylim(-0.72, 1.18)
    axis_model.axis("off")
    node_xy = {
        1: (0.0, 0.45),
        2: (1.05, 0.45),
        3: (2.55, 0.15),
        4: (3.55, 0.85),
        5: (4.10, -0.15),
    }
    axis_model.plot(
        [node_xy[1][0], node_xy[2][0]],
        [node_xy[1][1], node_xy[2][1]],
        color=pair_color,
        lw=3.0,
        solid_capstyle="round",
        zorder=1,
    )
    if float(representative.get("cross_coupling", 0.0)) > 0.0:
        axis_model.plot(
            [node_xy[2][0], node_xy[3][0]],
            [node_xy[2][1], node_xy[3][1]],
            color=muted,
            lw=1.1,
            ls=(0, (2.0, 2.0)),
            zorder=0,
        )
    triad_polygon = mpl.patches.Polygon(
        [node_xy[3], node_xy[4], node_xy[5]],
        closed=True,
        facecolor=mpl.colors.to_rgba(triad_color, 0.12),
        edgecolor=triad_color,
        linewidth=1.8,
        joinstyle="round",
        zorder=1,
    )
    axis_model.add_patch(triad_polygon)
    for index, (x_value, y_value) in node_xy.items():
        color = pair_color if index <= 2 else triad_color
        axis_model.scatter(
            x_value,
            y_value,
            s=155,
            facecolor="white",
            edgecolor=color,
            linewidth=1.5,
            zorder=3,
        )
        axis_model.text(
            x_value,
            y_value,
            rf"$\theta_{index}$",
            ha="center",
            va="center",
            fontsize=6.6,
            color=color,
            zorder=4,
        )
    axis_model.text(
        0.52,
        0.88,
        "pairwise edge",
        color=pair_color,
        ha="center",
        fontsize=6.0,
        fontweight="bold",
    )
    axis_model.text(
        3.48,
        1.03,
        "triadic hyperedge",
        color=triad_color,
        ha="center",
        fontsize=6.0,
        fontweight="bold",
    )
    axis_model.text(
        2.02,
        -0.58,
        r"$\mathbf{Y}=\Delta_\tau\boldsymbol{\theta}_{1:5}$",
        color=neutral,
        ha="center",
        fontsize=6.1,
    )

    # One representative noisy five-dimensional trajectory is easier to read
    # than isolated vector-field terms.  The channels are vertically offset
    # only for display; every trace is the bounded phase signal sin(theta_i).
    trajectory_dt = 0.01
    trajectory_time = np.arange(0.0, 6.0 + trajectory_dt, trajectory_dt)
    trajectory = np.empty((len(trajectory_time), len(NAMES)), dtype=float)
    trajectory[0] = np.array([-1.45, -0.10, 1.35, -0.85, 0.30])
    trajectory_rng = np.random.default_rng(20_260_726)
    for time_index in range(1, len(trajectory_time)):
        derivative = mixed_order_derivative(
            trajectory[time_index - 1 : time_index],
            pairwise_coupling=float(representative["pairwise_coupling"]),
            triadic_coupling=float(representative["triadic_coupling"]),
            cross_coupling=float(representative["cross_coupling"]),
        )[0]
        stochastic_increment = (
            float(representative["process_noise"])
            * math.sqrt(trajectory_dt)
            * trajectory_rng.normal(size=len(NAMES))
        )
        trajectory[time_index] = (
            trajectory[time_index - 1]
            + trajectory_dt * derivative
            + stochastic_increment
        )
    phase_signal = np.sin(trajectory)
    offsets = np.arange(len(NAMES) - 1, -1, -1, dtype=float) * 1.18
    channel_colors = [
        pair_color,
        mpl.colors.to_hex(mpl.colors.to_rgba(pair_color, 0.78)),
        triad_color,
        mpl.colors.to_hex(mpl.colors.to_rgba(triad_color, 0.82)),
        mpl.colors.to_hex(mpl.colors.to_rgba(triad_color, 0.68)),
    ]
    for index, offset in enumerate(offsets):
        axis_response.axhline(offset, color=faint, lw=0.55, zorder=0)
        axis_response.plot(
            trajectory_time,
            offset + 0.43 * phase_signal[:, index],
            color=channel_colors[index],
            lw=1.15,
            zorder=2,
        )
    axis_response.set(
        xlim=(trajectory_time[0], trajectory_time[-1]),
        ylim=(-0.56, offsets[0] + 0.56),
        xlabel="Time (s)",
    )
    axis_response.set_yticks(
        offsets,
        [rf"$\theta_{index}$" for index in range(1, len(NAMES) + 1)],
    )
    axis_response.tick_params(axis="y", length=0, pad=4)
    for tick_label, color in zip(
        axis_response.get_yticklabels(), channel_colors
    ):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")
    axis_response.text(
        0.0,
        1.03,
        r"Five-dimensional trajectory, $\sin\theta_i(t)$",
        transform=axis_response.transAxes,
        color=neutral,
        fontsize=6.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    def compact_partition(row: Mapping[str, object]) -> str:
        left = "".join(str(name).replace("theta", "") for name in row["left"])
        right = "".join(str(name).replace("theta", "") for name in row["right"])
        return "{" + left + "}|{" + right + "}"

    root_scores = [
        float(row["captured_phi_bits"]) for row in trace["candidates"]
    ]
    score_xmax = max(0.10, 1.12 * max(root_scores, default=0.0))
    all_scores = [
        float(row["captured_phi_bits"])
        for node_trace in (trace, pair_trace, triad_trace)
        for row in node_trace["candidates"]
    ]
    score_xmin = min(
        -0.025 * score_xmax,
        1.12 * min(all_scores, default=0.0),
    )

    def draw_ranking(
        axis: plt.Axes,
        node_trace: Mapping[str, object],
        *,
        heading: str,
        atom_color: str,
        show_xlabel: bool,
    ) -> None:
        candidates = sorted(
            node_trace["candidates"],
            key=lambda row: (
                -float(row["captured_phi_bits"]),
                compact_partition(row),
            ),
        )
        y_values = np.arange(len(candidates))[::-1]
        scores = np.asarray(
            [float(row["captured_phi_bits"]) for row in candidates], dtype=float
        )
        selected = np.asarray(
            [bool(row.get("selected", False)) for row in candidates], dtype=bool
        )
        colors = [
            selected_color
            if flag
            else atom_color
            if node_trace["action"] == "terminal"
            else muted
            for flag in selected
        ]
        axis.barh(
            y_values,
            scores,
            height=0.48,
            color=colors,
            edgecolor="none",
            zorder=2,
        )
        axis.scatter(
            scores,
            y_values,
            s=np.where(selected, 25.0, 10.0),
            facecolor=colors,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        if np.allclose(scores, 0.0):
            axis.scatter(
                np.zeros_like(y_values, dtype=float),
                y_values,
                s=18,
                facecolor="white",
                edgecolor=atom_color,
                linewidth=1.0,
                zorder=4,
            )
        axis.set_yticks(
            y_values,
            [compact_partition(row) for row in candidates],
            fontsize=5.2,
        )
        axis.set_ylim(-0.65, len(candidates) - 0.15)
        axis.axvline(0.0, color=faint, lw=0.7, zorder=0)
        axis.text(
            0.0,
            1.08,
            heading,
            transform=axis.transAxes,
            color=atom_color,
            fontsize=6.2,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        if node_trace["action"] != "terminal":
            selected_score = float(node_trace["captured_phi_bits"])
            axis.text(
                min(selected_score + 0.035 * score_xmax, 0.96 * score_xmax),
                float(y_values[np.flatnonzero(selected)[0]]),
                f"{selected_score:.3f}",
                color=selected_color,
                fontsize=5.6,
                fontweight="bold",
                ha="left",
                va="center",
            )
        retained_residual = (
            float(node_trace["residual_bits"])
            if node_trace["action"] != "terminal"
            else float(node_trace["phi_bits"])
        )
        axis.text(
            0.99,
            1.08,
            rf"root $\Phi={float(node_trace['phi_bits']):.2f}$; retained residual "
            rf"$={retained_residual:.2f}$ bits",
            transform=axis.transAxes,
            color=neutral,
            fontsize=5.7,
            ha="right",
            va="bottom",
        )
        axis.set_xlim(score_xmin, score_xmax)
        axis.tick_params(axis="x", labelbottom=show_xlabel)
        if show_xlabel:
            axis.set_xlabel(r"Greedy capture, $C(L,R)=\Phi(L)+\Phi(R)$ (bits)")
        else:
            axis.set_xlabel("")
        axis.spines["left"].set_visible(False)
        axis.tick_params(axis="y", length=0, pad=3)

    def draw_two_node_atom(
        axis: plt.Axes,
        node_trace: Mapping[str, object],
        *,
        heading: str,
        atom_color: str,
    ) -> None:
        sources = "".join(
            str(name).replace("theta", "") for name in node_trace["sources"]
        )
        xi_value = float(node_trace["phi_bits"])
        axis.set_axis_off()
        pill = mpl.patches.FancyBboxPatch(
            (0.015, 0.16),
            0.16,
            0.50,
            boxstyle="round,pad=0.018,rounding_size=0.055",
            transform=axis.transAxes,
            facecolor=mpl.colors.to_rgba(atom_color, 0.13),
            edgecolor=atom_color,
            linewidth=1.0,
        )
        axis.add_patch(pill)
        axis.text(
            0.095,
            0.41,
            rf"$\{{{sources}\}}$",
            transform=axis.transAxes,
            color=atom_color,
            fontsize=6.0,
            fontweight="bold",
            ha="center",
            va="center",
        )
        axis.text(
            0.205,
            0.41,
            r"$\longrightarrow$",
            transform=axis.transAxes,
            color=atom_color,
            fontsize=6.4,
            ha="center",
            va="center",
        )
        axis.text(
            0.255,
            0.41,
            rf"$\Xi_{{\{{{sources}\}}}}={xi_value:.3f}\ \mathrm{{bits}}$",
            transform=axis.transAxes,
            color=neutral,
            fontsize=6.1,
            fontweight="bold",
            ha="left",
            va="center",
        )
        axis.text(
            0.015,
            0.88,
            heading,
            transform=axis.transAxes,
            color=atom_color,
            fontsize=5.7,
            ha="left",
            va="bottom",
        )

    draw_ranking(
        axis_root,
        trace,
        heading=r"Root $\{1,2,3,4,5\}$ · 15 candidate cuts",
        atom_color=neutral,
        show_xlabel=False,
    )
    draw_two_node_atom(
        axis_pair,
        pair_trace,
        heading="terminal two-node atom",
        atom_color=pair_color,
    )
    draw_ranking(
        axis_triad,
        triad_trace,
        heading=r"Recurse on $\{3,4,5\}$",
        atom_color=triad_color,
        show_xlabel=True,
    )
    draw_two_node_atom(
        axis_pair_45,
        pair_45_trace,
        heading="terminal two-node atom",
        atom_color=triad_color,
    )
    axis_root.text(
        -0.14,
        1.10,
        "b",
        transform=axis_root.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )
    axis_model.text(
        -0.10,
        1.06,
        "a",
        transform=axis_model.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )

    figure_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        figure_base.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        figure_base.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        figure_base.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--training-count", type=int, default=None)
    parser.add_argument("--readout-count", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ei-epochs", type=int, default=None)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--figure-base", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_count = int(args.training_count or (3200 if args.mode == "smoke" else 4800))
    readout_count = int(args.readout_count or (3600 if args.mode == "smoke" else 6000))
    epochs = int(args.epochs or (220 if args.mode == "smoke" else 300))
    ei_epochs = int(args.ei_epochs or (120 if args.mode == "smoke" else 180))
    seeds = tuple(range(1 if args.mode == "smoke" else int(args.seeds)))
    try:
        payload = build_summary(
            seeds=seeds,
            training_count=training_count,
            readout_count=readout_count,
            degree=3,
            train_fraction=0.8,
            ridge=0.10,
            tau=0.20,
            dt=0.01,
            epochs=epochs,
            ei_epochs=ei_epochs,
            status_path=args.status,
        )
    except Exception as error:
        failed = {
            "phase": "failed",
            "message": str(error),
            "updated_at": time.time(),
        }
        if args.status.exists():
            try:
                failed = {
                    **json.loads(args.status.read_text(encoding="utf-8")),
                    **failed,
                }
            except (OSError, json.JSONDecodeError):
                pass
        args.status.parent.mkdir(parents=True, exist_ok=True)
        args.status.write_text(
            json.dumps(failed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_greedy_search_summary(payload, args.figure_base)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Saved result: {args.result}")
    print(f"Saved figure: {args.figure_base}.{{png,svg,pdf}}")


if __name__ == "__main__":
    main()
