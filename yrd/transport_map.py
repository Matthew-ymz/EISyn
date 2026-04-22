from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import digamma


def clip_nonnegative_ei(value: float) -> float:
    """Clamp estimated EI-style quantities to the nonnegative domain."""

    return max(0.0, float(value))


def gaussian_logdet_bias_correction(dimension: int, sample_size: int) -> float:
    """Wishart-model bias term for logdet(sample covariance).

    Returns zero when the sample size is too small for the classical correction.
    """

    if sample_size <= dimension:
        return 0.0
    nu = sample_size - 1
    return float(
        sum(digamma((nu + 1 - index) / 2.0) for index in range(1, dimension + 1))
        + dimension * np.log(2.0)
        - dimension * np.log(nu)
    )


@dataclass(frozen=True)
class AffineTransportMapDensityEstimator:
    """Affine lower-triangular transport-map style density estimator."""

    mean: np.ndarray
    covariance: np.ndarray
    sample_size: int
    backend: str = "affine_triangular_transport_map"

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", np.asarray(self.mean, dtype=float))
        object.__setattr__(self, "covariance", np.asarray(self.covariance, dtype=float))
        cholesky = np.linalg.cholesky(self.covariance)
        object.__setattr__(self, "cholesky", cholesky)
        object.__setattr__(self, "log_det_cholesky", float(np.log(np.diag(cholesky)).sum()))
        object.__setattr__(self, "dimension", int(self.mean.shape[0]))

    def map_to_reference(self, samples: np.ndarray) -> np.ndarray:
        array = np.asarray(samples, dtype=float)
        centered = array - self.mean
        return np.linalg.solve(self.cholesky, centered.T).T

    def log_prob(self, samples: np.ndarray) -> np.ndarray:
        reference = self.map_to_reference(samples)
        quadratic = np.sum(reference**2, axis=1)
        return -0.5 * (self.dimension * np.log(2.0 * np.pi) + quadratic) - self.log_det_cholesky

    def marginal(self, indices: list[int] | tuple[int, ...]) -> "AffineTransportMapDensityEstimator":
        subset = list(indices)
        return AffineTransportMapDensityEstimator(
            mean=self.mean[subset],
            covariance=self.covariance[np.ix_(subset, subset)],
            sample_size=self.sample_size,
            backend=self.backend,
        )


def fit_affine_transport_map_density(samples: np.ndarray, *, jitter: float = 1e-6) -> AffineTransportMapDensityEstimator:
    array = np.asarray(samples, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError("samples must be a 2D array with at least two rows.")
    mean = array.mean(axis=0)
    covariance = np.cov(array, rowvar=False, bias=False)
    covariance = np.atleast_2d(covariance)
    covariance += float(jitter) * np.eye(covariance.shape[0], dtype=float)
    return AffineTransportMapDensityEstimator(
        mean=mean,
        covariance=covariance,
        sample_size=int(array.shape[0]),
    )


def estimate_mutual_information_transport_map(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    """Estimate mutual information from an affine transport-map density model."""

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.ndim != 2 or y_array.ndim != 2 or x_array.shape[0] != y_array.shape[0]:
        raise ValueError("x and y must be 2D arrays with matching sample counts.")

    sample_size = x_array.shape[0]
    joint = np.concatenate([x_array, y_array], axis=1)
    joint_model = fit_affine_transport_map_density(joint)
    x_model = joint_model.marginal(list(range(x_array.shape[1])))
    y_model = joint_model.marginal(list(range(x_array.shape[1], joint.shape[1])))

    log_pxy = joint_model.log_prob(joint)
    log_px = x_model.log_prob(x_array)
    log_py = y_model.log_prob(y_array)
    pointwise_mi_raw = log_pxy - log_px - log_py

    bias_correction = 0.5 * (
        gaussian_logdet_bias_correction(x_array.shape[1], sample_size)
        + gaussian_logdet_bias_correction(y_array.shape[1], sample_size)
        - gaussian_logdet_bias_correction(joint.shape[1], sample_size)
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


def lift_transport_source_features(source: np.ndarray) -> np.ndarray:
    """Lift 1D or 2D sources with low-order features for affine tm estimation."""

    array = np.asarray(source, dtype=float)
    if array.ndim != 2:
        raise ValueError("source must be a 2D array.")
    if array.shape[1] == 1:
        x = array[:, [0]]
        return np.concatenate([x, x**2, x**3], axis=1)
    if array.shape[1] == 2:
        x = array[:, [0]]
        y = array[:, [1]]
        return np.concatenate([x, y, x * y, x**2, y**2], axis=1)
    raise ValueError("Expected one-dimensional or two-dimensional source blocks.")


def summarize_two_source_synergy_transport_map(
    left_source: np.ndarray,
    right_source: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Estimate two-source synergy while preserving joint-minus-single semantics."""

    left_array = np.asarray(left_source, dtype=float)
    right_array = np.asarray(right_source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if left_array.ndim != 2 or right_array.ndim != 2 or target_array.ndim != 2:
        raise ValueError("left_source, right_source, and target must be 2D arrays.")
    if left_array.shape[0] != right_array.shape[0] or left_array.shape[0] != target_array.shape[0]:
        raise ValueError("left_source, right_source, and target must share the sample axis.")
    if left_array.shape[1] != 1 or right_array.shape[1] != 1:
        raise ValueError("left_source and right_source must each contain exactly one source dimension.")

    left_ei = float(
        estimate_mutual_information_transport_map(
            lift_transport_source_features(left_array),
            target_array,
        )["mi_hat"]
    )
    right_ei = float(
        estimate_mutual_information_transport_map(
            lift_transport_source_features(right_array),
            target_array,
        )["mi_hat"]
    )
    joint_source = np.concatenate([left_array, right_array], axis=1)
    joint_ei = float(
        estimate_mutual_information_transport_map(
            lift_transport_source_features(joint_source),
            target_array,
        )["mi_hat"]
    )
    left_ei = clip_nonnegative_ei(left_ei)
    right_ei = clip_nonnegative_ei(right_ei)
    joint_ei = clip_nonnegative_ei(joint_ei)
    return {
        "left_ei": left_ei,
        "right_ei": right_ei,
        "joint_ei": joint_ei,
        "syn": float(joint_ei - left_ei - right_ei),
    }


def _leading_principal_components(matrix: np.ndarray, max_components: int) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(matrix, dtype=float) - np.asarray(matrix, dtype=float).mean(axis=0, keepdims=True)
    if centered.shape[1] == 1:
        return centered, np.array([1.0], dtype=float)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    component_count = max(1, min(max_components, vh.shape[0]))
    components = vh[:component_count]
    projected = centered @ components.T
    energy = singular_values**2
    explained = energy[:component_count] / max(float(energy.sum()), 1e-12)
    return projected, explained


def _coerce_source_matrix(source_matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(source_matrix, dtype=float)
    if array.ndim < 2:
        raise ValueError("source_matrix must have at least a sample axis and one feature axis.")
    if array.ndim == 2:
        return array
    return array.reshape(array.shape[0], -1)


def project_source_groups_to_transport_features(
    source_matrix: np.ndarray,
    *,
    source_groups: dict[str, list[int] | tuple[int, ...]],
    max_components_per_group: int = 1,
) -> tuple[np.ndarray, dict[str, list[int]], dict[str, dict[str, object]]]:
    """Project each source group into a compact low-dimensional transport-map feature block."""

    array = _coerce_source_matrix(source_matrix)
    blocks: list[np.ndarray] = []
    projected_groups: dict[str, list[int]] = {}
    metadata: dict[str, dict[str, object]] = {}
    offset = 0
    for group_name, indices in source_groups.items():
        subset = array[:, list(indices)]
        projected, explained = _leading_principal_components(subset, max_components=max_components_per_group)
        blocks.append(projected)
        width = int(projected.shape[1])
        projected_groups[group_name] = list(range(offset, offset + width))
        metadata[group_name] = {
            "original_dim": int(subset.shape[1]),
            "projected_dim": width,
            "explained_variance_ratio": explained.tolist(),
        }
        offset += width
    return np.concatenate(blocks, axis=1), projected_groups, metadata


def summarize_transport_map_group_decomposition(
    source_matrix: np.ndarray,
    target_matrix: np.ndarray,
    *,
    source_groups: dict[str, list[int] | tuple[int, ...]],
    max_components_per_group: int = 1,
) -> dict[str, object]:
    """Estimate groupwise EI and synergy from a direct transport-map density model."""

    projected_source, projected_groups, feature_metadata = project_source_groups_to_transport_features(
        source_matrix,
        source_groups=source_groups,
        max_components_per_group=max_components_per_group,
    )
    overall = estimate_mutual_information_transport_map(projected_source, target_matrix)
    group_ei_tm: dict[str, float] = {}
    for group_name, indices in projected_groups.items():
        summary = estimate_mutual_information_transport_map(projected_source[:, indices], target_matrix)
        group_ei_tm[group_name] = clip_nonnegative_ei(summary["mi_hat"])
    ei_tm = clip_nonnegative_ei(overall["mi_hat"])
    syn_tm = float(ei_tm - sum(group_ei_tm.values()))
    return {
        "backend": str(overall["backend"]),
        "ei_tm": ei_tm,
        "syn_tm": syn_tm,
        "group_ei_tm": group_ei_tm,
        "projected_source_dim": int(projected_source.shape[1]),
        "projected_groups": projected_groups,
        "feature_metadata": feature_metadata,
        "bias_correction": float(overall["bias_correction"]),
    }
