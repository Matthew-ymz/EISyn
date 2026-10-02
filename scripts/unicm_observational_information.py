"""Information priors from observed histories and actual future states.

The covariance is an affine Gaussian density estimate, not a learned transition
model. SURD uses all coalitions and the authors' specific-information allocation.
For a scalar Gaussian target, specific information is ordered identically at
every target state, so its exact Gaussian integral can be allocated from MI.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from scripts.analyze_unicm_11mode_shapley import mode_feature_blocks, standardize
from scripts.run_unicm_information_prior_comparison import (
    coalition_operators, nonnegative, TOLERANCE_BITS,
)

LOG2 = float(np.log(2.0))
OBSERVATIONAL_METHODS = ("phi_r", "phi_wms", "phi_si", "causal_density", "surd")


@dataclass
class GaussianObservation:
    source_covariance: np.ndarray
    cross_covariance: np.ndarray
    target_covariance: np.ndarray
    blocks: tuple
    mode_count: int
    leads: int


def fit_observation(history: np.ndarray, target: np.ndarray, ridge: float = 1e-6) -> GaussianObservation:
    """Fit ONLY the supplied paired observations; all coordinates retain correlation."""
    history, target = np.asarray(history, float), np.asarray(target, float)
    if history.ndim != 3 or target.ndim != 3 or history.shape[:2] != target.shape[:2]:
        raise ValueError("Expected paired [sample, mode, month] history and actual targets.")
    if len(history) < 3 or ridge <= 0:
        raise ValueError("Need at least three observations and a positive covariance ridge.")
    n, modes, months = history.shape
    x = standardize(history.transpose(0, 2, 1).reshape(n, -1), "observed history")
    y = standardize(target.transpose(0, 2, 1).reshape(n, -1), "actual future")
    cxx = x.T @ x / (n - 1) + ridge * np.eye(x.shape[1])
    cxy = x.T @ y / (n - 1)
    cyy = y.T @ y / (n - 1) + ridge * np.eye(y.shape[1])
    cholesky(cxx, lower=True)
    return GaussianObservation(cxx, cxy, cyy,
                               mode_feature_blocks(history_length=months, mode_count=modes),
                               modes, target.shape[2])


def _whiten_cross(density: GaussianObservation, modes: tuple) -> np.ndarray:
    columns = np.asarray([c for m in modes for c in density.blocks[m]], dtype=int)
    lower = cholesky(density.source_covariance[np.ix_(columns, columns)], lower=True)
    return solve_triangular(lower, density.cross_covariance[columns], lower=True)


def _conditional_variance(density: GaussianObservation, modes: tuple) -> np.ndarray:
    if not modes:
        return np.diag(density.target_covariance).copy()
    white = _whiten_cross(density, modes)
    variance = np.diag(density.target_covariance) - np.sum(white * white, axis=0)
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError(f"Invalid conditional variance for observed subset {modes}.")
    return variance


def _ld2(matrix: np.ndarray) -> np.ndarray:
    sign, value = np.linalg.slogdet(matrix)
    if np.any(sign <= 0) or not np.isfinite(value).all():
        raise ValueError("Observed pair covariance is not positive definite.")
    return value


def _shape_prior(flat: np.ndarray, density: GaussianObservation) -> np.ndarray:
    return flat.reshape(density.leads, density.mode_count, density.mode_count).transpose(1, 0, 2)


def observed_pair_information(density: GaussianObservation, method: str) -> dict:
    """Evaluate specified two-part partitions; no system MIP is claimed."""
    n, leads = density.mode_count, density.leads
    total_variance = np.diag(density.target_covariance)
    singleton_variance = np.stack([_conditional_variance(density, (m,)) for m in range(n)])
    singleton_mi = 0.5 * np.log(total_variance[None, :] / singleton_variance) / LOG2
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    edges = np.empty((leads, len(pairs)))
    identity_error = 0.0
    for pair_index, (a, b) in enumerate(pairs):
        outputs = np.arange(leads)[:, None] * n + np.asarray([a, b])[None, :]
        marginal = density.target_covariance[outputs[:, :, None], outputs[:, None, :]]
        white = _whiten_cross(density, (a, b))[:, outputs]
        conditional = marginal - np.einsum("dli,dlj->lij", white, white)
        joint = 0.5 * (_ld2(marginal) - _ld2(conditional)) / LOG2
        own = singleton_mi[a, outputs[:, 0]] + singleton_mi[b, outputs[:, 1]]
        wms = joint - own
        if method == "phi_wms":
            edges[:, pair_index] = wms
        elif method == "phi_r":
            correction = np.min(singleton_mi[np.asarray([a, b])[:, None, None], outputs[None]], axis=(0, 2))
            edges[:, pair_index] = wms + correction
        elif method == "phi_si":
            si = 0.5 * (np.log(singleton_variance[a, outputs[:, 0]])
                        + np.log(singleton_variance[b, outputs[:, 1]]) - _ld2(conditional)) / LOG2
            tc = 0.5 * (np.log(total_variance[outputs]).sum(axis=1) - _ld2(marginal)) / LOG2
            identity_error = max(identity_error, float(np.max(np.abs(si - wms - tc))))
            edges[:, pair_index] = si
        else:
            raise ValueError(method)
    return {"edges": edges, "pairs": pairs, "si_identity_error_bit": identity_error}


def observed_cd_information(density: GaussianObservation) -> dict:
    n = density.mode_count
    full = _conditional_variance(density, tuple(range(n)))
    complement = np.stack([_conditional_variance(density, tuple(m for m in range(n) if m != source))
                           for source in range(n)])
    transfer = (0.5 * np.log(complement / full) / LOG2).reshape(n, density.leads, n)
    # Published CD excludes self-connections; this is not numerical clipping.
    transfer[np.arange(n), :, np.arange(n)] = 0.0
    return {"transfer": transfer}


def all_observed_coalitions(density: GaussianObservation) -> np.ndarray:
    n = density.mode_count
    total = np.diag(density.target_covariance)
    table = np.zeros((1 << n, len(total)))
    for mask in range(1, 1 << n):
        modes = tuple(m for m in range(n) if mask & (1 << m))
        conditional = _conditional_variance(density, modes)
        table[mask] = 0.5 * np.log(total / conditional) / LOG2
    for m in range(n):
        absent = np.asarray([mask for mask in range(1 << n) if not mask & (1 << m)])
        minimum = float((table[absent | (1 << m)] - table[absent]).min())
        if minimum < -TOLERANCE_BITS:
            raise ValueError(f"Observed Gaussian coalition MI violates monotonicity: {minimum:g} bit.")
    return table


def assign_surd_specific(values: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Authors' allocation for one target state, including every nonempty subset.

    Equal specific-information values use ascending order then mask as an explicit
    tie convention. Removing a higher-order value below the lower-order maximum
    is a SURD allocation rule, not clipping an estimated synergy.
    """
    values = np.asarray(values, float)
    if values.shape != (1 << n,) or not np.isfinite(values).all():
        raise ValueError("SURD needs one complete finite coalition table.")
    masks = np.arange(1, 1 << n)
    sizes = np.asarray([int(m).bit_count() for m in masks])
    adjusted = values[masks].copy()
    for order in range(2, n + 1):
        lower_max = adjusted[sizes == order - 1].max()
        suppressed = (sizes == order) & (adjusted < lower_max)
        adjusted[suppressed] = 0.0
    ranking = np.lexsort((masks, sizes, adjusted))
    increments = np.diff(adjusted[ranking], prepend=0.0)
    redundancy = np.zeros(1 << n)
    synergy = np.zeros(1 << n)
    remaining = (1 << n) - 1
    for index, increment in zip(ranking, increments):
        mask = int(masks[index])
        if sizes[index] == 1:
            redundancy[remaining] += increment
            remaining ^= mask
        else:
            synergy[mask] += increment
    return redundancy, synergy


def gaussian_surd_allocation(table: np.ndarray, n: int, audit: dict) -> dict:
    """Exact SURD allocation within a scalar-target Gaussian density model.

    I(X_S;Y=y)=(-log(1-rho_S²)+rho_S²(z²-1))/(2 log 2).
    Its derivative in rho_S² is nonnegative for every z. Coalition ordering and
    SURD suppression are state invariant, and E[z²]=1. Thus allocation commutes
    with the Gaussian integral. This shortcut does not apply to nonlinear TM.
    """
    redundancy = np.empty_like(table)
    synergy = np.empty_like(table)
    for target in range(table.shape[1]):
        redundancy[:, target], synergy[:, target] = assign_surd_specific(table[:, target], n)
    redundancy = nonnegative(redundancy, "surd_redundancy_unique_atoms", audit)
    synergy = nonnegative(synergy, "surd_synergy_atoms", audit)
    closure = float(np.max(np.abs(redundancy.sum(axis=0) + synergy.sum(axis=0) - table[-1])))
    if closure > 1e-9:
        raise ValueError(f"SURD information closure failed: {closure:g} bit.")
    membership = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(float)
    sizes = membership.sum(axis=1)
    sharing = np.zeros_like(membership)
    sharing[sizes > 0] = membership[sizes > 0] / sizes[sizes > 0, None]
    attribution = synergy.T @ sharing
    attribution_closure = float(np.max(np.abs(attribution.sum(axis=1) - synergy.sum(axis=0))))
    if attribution_closure > 1e-9:
        raise ValueError("SURD source allocation does not conserve synergy.")
    return {"redundancy_unique_atoms": redundancy, "synergy_atoms": synergy,
            "coalition_mi": table, "attribution_flat": attribution,
            "closure_error_bit": closure, "attribution_closure_error_bit": attribution_closure}


def nonnegative_ridge_scores(raw: np.ndarray, method: str) -> tuple[np.ndarray, dict]:
    """Translate a signed source vector uniformly when needed; retain raw scores.

    WMS (and finite-history PhiR) are not declared PEID Syn. A common offset
    preserves ranking and differences; it is an explicit calibration interface.
    """
    raw = np.asarray(raw, float)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite observational attribution.")
    offset = -np.minimum(raw.min(axis=-1, keepdims=True), 0.0)
    if method not in ("phi_wms", "phi_r") and np.any(offset > TOLERANCE_BITS):
        raise ValueError(f"Unexpected significant negative {method} attribution.")
    return raw + offset, {"rule": "c_plus = c_raw - min(0, min_source(c_raw)); common offset per target/lead",
                          "minimum_raw_bit": float(raw.min()), "negative_raw_count": int(np.count_nonzero(raw < 0)),
                          "shifted_target_lead_count": int(np.count_nonzero(offset)),
                          "maximum_common_offset_bit": float(offset.max())}


def observational_prior(history: np.ndarray, target: np.ndarray, method: str,
                        ridge: float = 1e-6) -> tuple[np.ndarray, dict, dict]:
    audit = {"tolerance_bit": TOLERANCE_BITS}
    clocks = {}
    start, cpu = time.perf_counter(), time.process_time()
    density = fit_observation(history, target, ridge)
    clocks["density_fit_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    if method in ("phi_r", "phi_wms", "phi_si"):
        native = observed_pair_information(density, method)
    elif method == "causal_density":
        native = observed_cd_information(density)
    elif method == "surd":
        native = {"coalition_mi": all_observed_coalitions(density)}
    else:
        raise ValueError(method)
    clocks["information_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    n = density.mode_count
    if method in ("phi_r", "phi_wms", "phi_si"):
        edges = native["edges"]
        if method == "phi_si":
            edges = native["edges"] = nonnegative(edges, "phi_si_edges", audit)
            if native["si_identity_error_bit"] > 1e-9:
                raise ValueError("Natural SI = WMS + output TC check failed.")
            audit["si_identity_error_bit"] = native["si_identity_error_bit"]
        source = np.zeros((density.leads, n))
        for pair, (a, b) in enumerate(native["pairs"]):
            source[:, a] += edges[:, pair] / 2
            source[:, b] += edges[:, pair] / 2
        raw = np.broadcast_to(source[None], (n, density.leads, n)).copy()
        audit["native_signed_edges"] = {"minimum_bit": float(edges.min()),
                                          "negative_count": int(np.count_nonzero(edges < 0))}
    elif method == "causal_density":
        transfer = native["transfer"] = nonnegative(native["transfer"], "conditional_transfer", audit)
        outgoing = transfer.sum(axis=2) / (n * (n - 1))
        raw = np.broadcast_to(outgoing.T[None], (n, density.leads, n)).copy()
    else:
        native = gaussian_surd_allocation(native["coalition_mi"], n, audit)
        audit["surd_information_closure_error_bit"] = native["closure_error_bit"]
        audit["surd_attribution_closure_error_bit"] = native["attribution_closure_error_bit"]
        raw = _shape_prior(native["attribution_flat"], density)
    prior, mapping = nonnegative_ridge_scores(raw, method)
    native["raw_attribution"] = raw
    audit["ridge_mapping"] = mapping
    clocks["attribution_seconds"] = time.perf_counter() - start
    clocks["core_total_seconds"] = sum(clocks.values())
    clocks["process_cpu_seconds"] = time.process_time() - cpu
    audit["timing"] = clocks
    return prior, native, audit


def intervention_prior(history: np.ndarray, predictions: list[np.ndarray], method: str,
                       ridge: float = 1e-6) -> tuple[np.ndarray, dict]:
    """Fresh all-order EI/Xi computation from existing predictions, with timing.

    Multiresponse least squares shares a factorization across all 264 targets;
    this is algebraically identical to fitting each lead independently.
    """
    if method not in ("xi_shapley", "ei_shapley"):
        raise ValueError(method)
    total_start, cpu = time.perf_counter(), time.process_time()
    n_samples, history_months, modes = history.shape
    x = standardize(history.reshape(n_samples, -1), "intervention history")
    blocks = mode_feature_blocks(history_length=history_months, mode_count=modes)
    membership, shapley = coalition_operators(modes)
    audits, values = [], []
    stages = {"density_fit_seconds": 0.0, "information_seconds": 0.0, "attribution_seconds": 0.0}
    for prediction in predictions:
        start = time.perf_counter()
        y = standardize(prediction.reshape(n_samples, -1), "frozen future")
        coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        if rank != x.shape[1]:
            raise ValueError("Intervention affine readout is rank deficient.")
        residual = y - x @ coefficients
        residual_variance = np.sum(residual * residual, axis=0) / (n_samples - 1) + ridge
        diagonal = np.stack([np.sum(coefficients[np.asarray(b)] ** 2, axis=0) for b in blocks])
        stages["density_fit_seconds"] += time.perf_counter() - start
        start = time.perf_counter()
        conditional = residual_variance[None] + (1 - membership) @ diagonal
        ei = 0.5 * (np.log(conditional[0])[None] - np.log(conditional)) / LOG2
        stages["information_seconds"] += time.perf_counter() - start
        start = time.perf_counter()
        audit = {}
        ei = nonnegative(ei, "coalition_ei", audit)
        game = ei
        if method == "xi_shapley":
            game = nonnegative(ei - membership @ ei[1 << np.arange(modes)], "subset_xi", audit)
        attribution = nonnegative((shapley @ game).T, method, audit)
        if np.max(np.abs(attribution.sum(axis=1) - game[-1])) > 1e-9:
            raise ValueError("Intervention Shapley closure failed.")
        values.append(attribution.reshape(prediction.shape[1], modes, modes).transpose(1, 0, 2))
        stages["attribution_seconds"] += time.perf_counter() - start
        audits.append(audit)
    stages["core_total_seconds"] = time.perf_counter() - total_start
    stages["process_cpu_seconds"] = time.process_time() - cpu
    return np.stack(values).mean(axis=0), {"timing": stages, "checkpoint_audits": audits}
