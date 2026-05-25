from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.special import digamma


__all__ = [
    "AffineTransportMapDensityEstimator",
    "QuadraticTriangularTransportMapDensityEstimator",
    "estimate_mutual_information_transport_map",
    "fit_affine_transport_map_density",
    "fit_quadratic_triangular_transport_map_density",
    "multivariate_gaussian_logpdf",
    "pairwise_effective_information_for_dynamics",
    "standard_gaussian_logpdf",
]


@dataclass(frozen=True)
class AffineTransportMapDensityEstimator:
    """Affine lower-triangular transport-map density estimator."""

    mean: np.ndarray
    covariance: np.ndarray
    sample_size: int
    backend: str = "affine_triangular_transport_map"

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        if mean.ndim != 1:
            raise ValueError("mean must be a one-dimensional array.")
        if covariance.shape != (mean.shape[0], mean.shape[0]):
            raise ValueError("covariance must have shape [dimension, dimension].")
        cholesky = np.linalg.cholesky(covariance)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "cholesky", cholesky)
        object.__setattr__(self, "log_det_cholesky", float(np.log(np.diag(cholesky)).sum()))
        object.__setattr__(self, "dimension", int(mean.shape[0]))

    def map_to_reference(self, samples: np.ndarray) -> np.ndarray:
        array = _coerce_samples(samples, expected_dim=self.dimension)
        centered = array - self.mean
        return np.linalg.solve(self.cholesky, centered.T).T

    def log_prob(self, samples: np.ndarray) -> np.ndarray:
        reference = self.map_to_reference(samples)
        quadratic = np.sum(reference**2, axis=1)
        return -0.5 * (self.dimension * np.log(2.0 * np.pi) + quadratic) - self.log_det_cholesky

    def pdf(self, samples: np.ndarray) -> np.ndarray:
        return np.exp(self.log_prob(samples))

    def marginal(self, indices: Sequence[int]) -> "AffineTransportMapDensityEstimator":
        subset = [int(index) for index in indices]
        if not subset:
            raise ValueError("indices must contain at least one dimension.")
        if min(subset) < 0 or max(subset) >= self.dimension:
            raise ValueError("indices contain a dimension outside the estimator support.")
        return AffineTransportMapDensityEstimator(
            mean=self.mean[subset],
            covariance=self.covariance[np.ix_(subset, subset)],
            sample_size=self.sample_size,
            backend=self.backend,
        )


@dataclass(frozen=True)
class QuadraticTriangularTransportMapDensityEstimator:
    """Two-dimensional quadratic triangular transport-map density estimator."""

    x1_mean: float
    x1_scale: float
    x2_coefficients: np.ndarray
    residual_scale: float
    sample_size: int
    backend: str = "quadratic_triangular_transport_map"

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.x2_coefficients, dtype=float)
        if coefficients.shape != (3,):
            raise ValueError("x2_coefficients must contain intercept, linear, and quadratic terms.")
        if float(self.x1_scale) <= 0.0:
            raise ValueError("x1_scale must be positive.")
        if float(self.residual_scale) <= 0.0:
            raise ValueError("residual_scale must be positive.")
        object.__setattr__(self, "x2_coefficients", coefficients)
        object.__setattr__(self, "x1_mean", float(self.x1_mean))
        object.__setattr__(self, "x1_scale", float(self.x1_scale))
        object.__setattr__(self, "residual_scale", float(self.residual_scale))
        object.__setattr__(self, "dimension", 2)
        object.__setattr__(
            self,
            "log_abs_det_jacobian",
            -float(np.log(self.x1_scale)) - float(np.log(self.residual_scale)),
        )

    def conditional_mean_x2(self, x1: np.ndarray) -> np.ndarray:
        values = np.asarray(x1, dtype=float)
        design = np.column_stack([np.ones(values.shape[0], dtype=float), values, values**2])
        return design @ self.x2_coefficients

    def map_to_reference(self, samples: np.ndarray) -> np.ndarray:
        array = _coerce_samples(samples, expected_dim=2)
        z1 = (array[:, 0] - self.x1_mean) / self.x1_scale
        predicted_x2 = self.conditional_mean_x2(array[:, 0])
        z2 = (array[:, 1] - predicted_x2) / self.residual_scale
        return np.column_stack([z1, z2])

    def log_prob(self, samples: np.ndarray) -> np.ndarray:
        reference = self.map_to_reference(samples)
        return standard_gaussian_logpdf(reference) + self.log_abs_det_jacobian

    def pdf(self, samples: np.ndarray) -> np.ndarray:
        return np.exp(self.log_prob(samples))


def fit_affine_transport_map_density(
    samples: np.ndarray,
    *,
    jitter: float = 1e-6,
) -> AffineTransportMapDensityEstimator:
    array = _coerce_samples(samples)
    if array.shape[0] < 2:
        raise ValueError("samples must contain at least two rows.")
    mean = array.mean(axis=0)
    covariance = np.cov(array, rowvar=False, bias=False)
    covariance = np.atleast_2d(covariance)
    covariance += float(jitter) * np.eye(covariance.shape[0], dtype=float)
    return AffineTransportMapDensityEstimator(
        mean=mean,
        covariance=covariance,
        sample_size=int(array.shape[0]),
    )


def fit_quadratic_triangular_transport_map_density(
    samples: np.ndarray,
    *,
    min_scale: float = 1e-8,
) -> QuadraticTriangularTransportMapDensityEstimator:
    array = _coerce_samples(samples, expected_dim=2)
    if array.shape[0] < 4:
        raise ValueError("samples must contain at least four rows.")
    x1 = array[:, 0]
    x2 = array[:, 1]
    x1_scale = max(float(np.std(x1, ddof=1)), float(min_scale))
    design = np.column_stack([np.ones(array.shape[0], dtype=float), x1, x1**2])
    coefficients, *_ = np.linalg.lstsq(design, x2, rcond=None)
    residuals = x2 - design @ coefficients
    residual_scale = max(float(np.std(residuals, ddof=1)), float(min_scale))
    return QuadraticTriangularTransportMapDensityEstimator(
        x1_mean=float(np.mean(x1)),
        x1_scale=x1_scale,
        x2_coefficients=coefficients,
        residual_scale=residual_scale,
        sample_size=int(array.shape[0]),
    )


def estimate_mutual_information_transport_map(
    x: np.ndarray,
    y: np.ndarray,
    *,
    jitter: float = 1e-6,
) -> dict[str, object]:
    """Estimate mutual information with an affine transport-map density model."""

    x_array = _coerce_samples(x)
    y_array = _coerce_samples(y)
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("x and y must have matching sample counts.")

    sample_size = int(x_array.shape[0])
    joint = np.concatenate([x_array, y_array], axis=1)
    joint_model = fit_affine_transport_map_density(joint, jitter=jitter)
    x_model = joint_model.marginal(list(range(x_array.shape[1])))
    y_model = joint_model.marginal(list(range(x_array.shape[1], joint.shape[1])))

    log_pxy = joint_model.log_prob(joint)
    log_px = x_model.log_prob(x_array)
    log_py = y_model.log_prob(y_array)
    pointwise_mi_raw = log_pxy - log_px - log_py
    bias_correction = 0.5 * (
        _gaussian_logdet_bias_correction(x_array.shape[1], sample_size)
        + _gaussian_logdet_bias_correction(y_array.shape[1], sample_size)
        - _gaussian_logdet_bias_correction(joint.shape[1], sample_size)
    )
    pointwise_mi = pointwise_mi_raw - bias_correction
    return {
        "backend": joint_model.backend,
        "mi_hat": float(pointwise_mi.mean()),
        "bias_correction": float(bias_correction),
        "pointwise_mi": pointwise_mi,
        "log_pxy": log_pxy,
        "log_px": log_px,
        "log_py": log_py,
    }


def pairwise_effective_information_for_dynamics(
    dynamics: Callable[[np.ndarray], np.ndarray],
    *,
    input_indices: Sequence[int] | int,
    output_indices: Sequence[int] | int,
    input_dim: int | None = None,
    center: np.ndarray | Sequence[float] | float | None = None,
    box_size: np.ndarray | Sequence[float] | float = 1.0,
    sample_count: int = 4096,
    seed: int = 0,
    jitter: float = 1e-6,
    clip_negative: bool = True,
) -> dict[str, object]:
    """Estimate pairwise EI for selected dynamics input-output coordinates.

    ``dynamics`` receives a sample matrix with shape ``[sample_count, input_dim]``.
    The returned ``pairwise_ei`` matrix has rows aligned with ``input_indices`` and
    columns aligned with ``output_indices``.
    """

    source_indices = _coerce_index_sequence(input_indices, name="input_indices")
    target_indices = _coerce_index_sequence(output_indices, name="output_indices")
    resolved_input_dim = _resolve_input_dim(
        input_dim=input_dim,
        center=center,
        input_indices=source_indices,
    )
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2.")

    center_array = _coerce_vector_parameter(
        0.0 if center is None else center,
        dimension=resolved_input_dim,
        name="center",
    )
    box_size_array = _coerce_vector_parameter(
        box_size,
        dimension=resolved_input_dim,
        name="box_size",
    )
    if bool(np.any(box_size_array <= 0.0)):
        raise ValueError("box_size must be positive in every input dimension.")

    rng = np.random.default_rng(seed)
    half_width = box_size_array / 2.0
    source_samples = rng.uniform(
        low=center_array - half_width,
        high=center_array + half_width,
        size=(int(sample_count), resolved_input_dim),
    )
    target_samples = _coerce_samples(dynamics(source_samples))
    if target_samples.shape[0] != source_samples.shape[0]:
        raise ValueError("dynamics output must preserve the input sample axis.")
    if max(target_indices) >= target_samples.shape[1]:
        raise ValueError("output_indices contain a dimension outside the dynamics output.")

    pairwise_ei = np.empty((len(source_indices), len(target_indices)), dtype=float)
    details: dict[tuple[int, int], dict[str, object]] = {}
    for row, source_index in enumerate(source_indices):
        if source_index >= resolved_input_dim:
            raise ValueError("input_indices contain a dimension outside the sampled input.")
        source_block = source_samples[:, [source_index]]
        for col, target_index in enumerate(target_indices):
            target_block = target_samples[:, [target_index]]
            summary = estimate_mutual_information_transport_map(source_block, target_block, jitter=jitter)
            value = float(summary["mi_hat"])
            if clip_negative:
                value = max(0.0, value)
            pairwise_ei[row, col] = value
            details[(source_index, target_index)] = summary

    return {
        "backend": "affine_triangular_transport_map",
        "input_indices": source_indices,
        "output_indices": target_indices,
        "pairwise_ei": pairwise_ei,
        "source_samples": source_samples,
        "target_samples": target_samples,
        "details": details,
    }


def standard_gaussian_logpdf(samples: np.ndarray) -> np.ndarray:
    array = _coerce_samples(samples)
    return -0.5 * (array.shape[1] * np.log(2.0 * np.pi) + np.sum(array**2, axis=1))


def multivariate_gaussian_logpdf(
    samples: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    array = _coerce_samples(samples)
    mean_array = np.asarray(mean, dtype=float)
    covariance_array = np.asarray(covariance, dtype=float)
    if mean_array.ndim != 1:
        raise ValueError("mean must be a one-dimensional array.")
    if array.shape[1] != mean_array.shape[0]:
        raise ValueError("samples and mean dimensions do not match.")
    if covariance_array.shape != (mean_array.shape[0], mean_array.shape[0]):
        raise ValueError("covariance must have shape [dimension, dimension].")
    cholesky = np.linalg.cholesky(covariance_array)
    centered = array - mean_array
    whitened = np.linalg.solve(cholesky, centered.T).T
    quadratic = np.sum(whitened**2, axis=1)
    log_det = 2.0 * float(np.log(np.diag(cholesky)).sum())
    return -0.5 * (mean_array.shape[0] * np.log(2.0 * np.pi) + log_det + quadratic)


def _coerce_samples(samples: np.ndarray, *, expected_dim: int | None = None) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("samples must be a one-dimensional or two-dimensional array.")
    if expected_dim is not None and array.shape[1] != expected_dim:
        raise ValueError(f"samples must have {expected_dim} columns.")
    return array


def _gaussian_logdet_bias_correction(dimension: int, sample_size: int) -> float:
    if sample_size <= dimension:
        return 0.0
    nu = sample_size - 1
    return float(
        sum(digamma((nu + 1 - index) / 2.0) for index in range(1, dimension + 1))
        + dimension * np.log(2.0)
        - dimension * np.log(nu)
    )


def _coerce_index_sequence(indices: Sequence[int] | int, *, name: str) -> list[int]:
    if isinstance(indices, (int, np.integer)):
        result = [int(indices)]
    else:
        result = [int(index) for index in indices]
    if not result:
        raise ValueError(f"{name} must contain at least one index.")
    if min(result) < 0:
        raise ValueError(f"{name} must contain nonnegative indices.")
    return result


def _resolve_input_dim(
    *,
    input_dim: int | None,
    center: np.ndarray | Sequence[float] | float | None,
    input_indices: Sequence[int],
) -> int:
    if input_dim is not None:
        resolved = int(input_dim)
    elif center is not None and np.asarray(center, dtype=float).ndim > 0:
        resolved = int(np.asarray(center, dtype=float).reshape(-1).shape[0])
    else:
        resolved = max(input_indices) + 1
    if resolved <= 0:
        raise ValueError("input_dim must be positive.")
    if max(input_indices) >= resolved:
        raise ValueError("input_indices contain a dimension outside the sampled input.")
    return resolved


def _coerce_vector_parameter(
    value: np.ndarray | Sequence[float] | float,
    *,
    dimension: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(dimension, float(array.item()), dtype=float)
    vector = array.reshape(-1)
    if vector.shape != (dimension,):
        raise ValueError(f"{name} must be scalar or have length input_dim.")
    return vector
