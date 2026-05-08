from __future__ import annotations

# Consolidated YRD package module.

# The former yrd/*.py modules are merged here; submodule import paths are aliased at the end.


# --- Former yrd/config.py ---

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class YRDExperimentConfig:
    root_dir: Path = Path(".")
    dataset_path: Path = Path("data/dataset_yrd.nc")
    station_path: Path = Path("data/stations_yrd.csv")
    sample_mode: str = "windowed"
    history_hours: int = 24
    horizons: tuple[int, ...] = (1, 24)
    target_variables: tuple[str, str] = ("O3", "PM2.5")
    meteorology_variables: tuple[str, ...] = (
        "t2m",
        "d2m",
        "sp",
        "tp",
        "blh",
        "msdwswrf",
        "u100",
        "v100",
    )
    train_end: pd.Timestamp = pd.Timestamp("2021-12-31 23:00:00")
    val_end: pd.Timestamp = pd.Timestamp("2022-12-31 23:00:00")
    test_end: pd.Timestamp = pd.Timestamp("2023-12-31 23:00:00")
    model_name: str = "resmlp"
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    norm_type: str = "layernorm"
    activation: str = "silu"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 3
    max_epochs: int = 100
    early_stopping_patience: int = 10
    seed: int = 0
    box_size: float = math.sqrt(12.0)
    causal_graph_box_size_by_variable: dict[str, float] | None = None
    causal_graph_nonnegative_variables: tuple[str, ...] = ()
    smoke_station_count: int = 4
    smoke_samples_per_split: int = 48

    @property
    def input_variables(self) -> tuple[str, ...]:
        return self.target_variables + self.meteorology_variables

    @property
    def cache_dir(self) -> Path:
        return self.root_dir / "exp" / "cache" / "yrd_coupling"

    @property
    def results_dir(self) -> Path:
        return self.root_dir / "fig" / "yrd_shanghai" / "artifacts"


# --- Former yrd/transport_map.py ---

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


# --- Former yrd/analysis.py ---

import json
from pathlib import Path

import pandas as pd


def classify_pollution_events(
    frame: pd.DataFrame,
    *,
    o3_threshold: float,
    pm25_threshold: float,
) -> pd.Series:
    labels = []
    for _, row in frame.iterrows():
        if row["O3"] >= o3_threshold and row["PM2.5"] >= pm25_threshold:
            labels.append("compound")
        elif row["O3"] >= o3_threshold:
            labels.append("o3")
        elif row["PM2.5"] >= pm25_threshold:
            labels.append("pm25")
        else:
            labels.append("normal")
    return frame.assign(event=labels)["event"]


def write_markdown_summary(
    path: Path,
    *,
    title: str,
    bullets: list[str],
    intro: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if intro:
        lines.extend([intro, ""])
    for bullet in bullets:
        lines.append(f"- {bullet}")
    path.write_text("\n".join(lines) + "\n")


def metrics_bullets(metrics: dict[str, object]) -> list[str]:
    bullets: list[str] = []
    model_metrics = metrics.get("joint_test", {})
    baseline_metrics = metrics.get("baseline_test", {})
    for horizon, summary in sorted(model_metrics.items(), key=lambda item: int(item[0])):
        baseline = baseline_metrics.get(horizon, {})
        model_rmse = summary.get("rmse")
        base_rmse = baseline.get("rmse")
        if model_rmse is not None and base_rmse is not None:
            bullets.append(
                f"{horizon}h 预测上，联合 MLP 的 RMSE 为 {model_rmse:.4f}，持续性基线为 {base_rmse:.4f}。"
            )
    return bullets


def coupling_bullets(summary: dict[str, object]) -> list[str]:
    bullets = [
        f"整体 `EI^{{nis}}` 为 {summary['ei_nis']:.4f}，`Syn_p^{{nis}}` 为 {summary['syn_nis']:.4f}。",
    ]
    group_terms = summary.get("group_ei_nis", {})
    if group_terms:
        ranked = sorted(group_terms.items(), key=lambda item: item[1], reverse=True)
        top_name, top_value = ranked[0]
        bullets.append(f"源组中贡献最大的部分是 `{top_name}`，其对应的 `EI^{{nis}}` 为 {top_value:.4f}。")
    return bullets


def coupling_by_horizon_bullets(summary_by_horizon: dict[str, dict[str, object]]) -> list[str]:
    bullets: list[str] = []
    for horizon, summary in sorted(summary_by_horizon.items(), key=lambda item: int(item[0])):
        bullets.append(
            f"{horizon}h 目标上，整体 `EI^{{nis}}` 为 {summary['ei_nis']:.4f}，`Syn_p^{{nis}}` 为 {summary['syn_nis']:.4f}。"
        )
    if "1" in summary_by_horizon and "24" in summary_by_horizon:
        delta = summary_by_horizon["24"]["syn_nis"] - summary_by_horizon["1"]["syn_nis"]
        bullets.append(f"`24h - 1h` 的协同差值为 {delta:.4f}。")
    return bullets


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


# --- Former yrd/intervention_sampling.py ---

import numpy as np


def compute_training_input_center(x_train: np.ndarray) -> np.ndarray:
    array = np.asarray(x_train, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("x_train must have shape [samples, n_stations, n_features].")
    if array.shape[0] == 0:
        raise ValueError("x_train must contain at least one sample.")
    return array.mean(axis=0, dtype=np.float32)


def _broadcast_sampling_parameter(
    value: float | np.ndarray | list[float],
    *,
    shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return np.full(shape, float(array.item()), dtype=np.float32)
    try:
        return np.broadcast_to(array, shape).astype(np.float32, copy=True)
    except ValueError as exc:
        raise ValueError(f"{name} must be broadcastable to shape {shape}.") from exc


def sample_uniform_box_inputs(
    *,
    center: np.ndarray,
    box_size: float | np.ndarray | list[float],
    sample_count: int,
    seed: int,
    lower_bounds: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    center_array = np.asarray(center, dtype=np.float32)
    if center_array.ndim != 2:
        raise ValueError("center must have shape [n_stations, n_features].")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")

    box_size_array = _broadcast_sampling_parameter(
        box_size,
        shape=tuple(center_array.shape),
        name="box_size",
    )
    if bool(np.any(box_size_array <= 0.0)):
        raise ValueError("box_size must be positive in every dimension.")

    half_width = box_size_array / 2.0
    low = center_array - half_width
    high = center_array + half_width
    if lower_bounds is not None:
        lower_bound_array = _broadcast_sampling_parameter(
            lower_bounds,
            shape=tuple(center_array.shape),
            name="lower_bounds",
        )
        low = np.maximum(low, lower_bound_array)
        high = np.maximum(high, low)

    rng = np.random.default_rng(seed)
    samples = rng.uniform(
        low=low,
        high=high,
        size=(sample_count, *center_array.shape),
    )
    return samples.astype(np.float32)


def resolve_variable_box_size_by_feature(
    *,
    input_variables: tuple[str, ...],
    box_size_by_variable: dict[str, float],
    n_stations: int,
) -> np.ndarray:
    missing_variables = [name for name in input_variables if name not in box_size_by_variable]
    if missing_variables:
        missing_text = ", ".join(missing_variables)
        raise ValueError(f"Missing causal-graph box sizes for variables: {missing_text}.")
    widths = np.asarray([float(box_size_by_variable[name]) for name in input_variables], dtype=np.float32)
    if bool(np.any(widths <= 0.0)):
        raise ValueError("causal_graph_box_size_by_variable must be positive for every variable.")
    return np.broadcast_to(widths[None, :], (n_stations, len(input_variables))).astype(np.float32, copy=True)


def resolve_nonnegative_lower_bounds_by_feature(
    *,
    input_variables: tuple[str, ...],
    stats: dict[str, dict[str, float]],
    nonnegative_variables: tuple[str, ...],
    n_stations: int,
) -> np.ndarray | None:
    if not nonnegative_variables:
        return None
    lower_bounds = np.full((n_stations, len(input_variables)), -np.inf, dtype=np.float32)
    nonnegative_set = set(nonnegative_variables)
    for feature_index, variable_name in enumerate(input_variables):
        if variable_name not in nonnegative_set:
            continue
        variable_stats = stats[variable_name]
        std = float(variable_stats["std"])
        if std <= 0.0:
            std = 1.0
        lower_bounds[:, feature_index] = float((0.0 - float(variable_stats["mean"])) / std)
    return lower_bounds


def resolve_causal_graph_box_label(
    *,
    box_size: float | None = None,
    box_size_by_variable: dict[str, float] | None = None,
) -> str | float | None:
    if box_size_by_variable is not None:
        return "per-variable L_v (support-cover)"
    if box_size is None:
        return None
    return float(box_size)


def estimate_support_cover_box_profile(
    *,
    x_train: np.ndarray,
    input_variables: tuple[str, ...],
    gamma: float,
    stats: dict[str, dict[str, float]],
    nonnegative_variables: tuple[str, ...] = (),
) -> dict[str, object]:
    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")
    center = compute_training_input_center(x_train)
    if center.shape[1] != len(input_variables):
        raise ValueError("input_variables must match the trailing feature dimension of x_train.")

    array = np.asarray(x_train, dtype=np.float32)
    feature_min = array.min(axis=0).astype(np.float32)
    feature_max = array.max(axis=0).astype(np.float32)
    cover_radius_by_feature = np.maximum(center - feature_min, feature_max - center).astype(np.float32)
    cover_radius_vector = cover_radius_by_feature.max(axis=0).astype(np.float32)
    width_vector = np.maximum(
        2.0 * float(gamma) * cover_radius_vector,
        np.full_like(cover_radius_vector, np.finfo(np.float32).eps),
    )
    box_size_by_variable = {
        variable_name: float(width_vector[feature_index])
        for feature_index, variable_name in enumerate(input_variables)
    }
    box_size_by_feature = resolve_variable_box_size_by_feature(
        input_variables=input_variables,
        box_size_by_variable=box_size_by_variable,
        n_stations=center.shape[0],
    )
    lower_bounds = resolve_nonnegative_lower_bounds_by_feature(
        input_variables=input_variables,
        stats=stats,
        nonnegative_variables=nonnegative_variables,
        n_stations=center.shape[0],
    )
    support_low_by_feature = center - box_size_by_feature / 2.0
    support_high_by_feature = center + box_size_by_feature / 2.0
    if lower_bounds is not None:
        support_low_by_feature = np.maximum(support_low_by_feature, lower_bounds)
        support_high_by_feature = np.maximum(support_high_by_feature, support_low_by_feature)
    return {
        "box_mode": "per_variable",
        "gamma": float(gamma),
        "input_variables": list(input_variables),
        "center": center,
        "feature_min": feature_min,
        "feature_max": feature_max,
        "cover_radius_by_feature": cover_radius_by_feature,
        "support_low_by_feature": support_low_by_feature.astype(np.float32),
        "support_high_by_feature": support_high_by_feature.astype(np.float32),
        "box_size_by_feature": box_size_by_feature,
        "lower_bounds": lower_bounds,
        "center_by_variable": {
            variable_name: float(center[:, feature_index].mean())
            for feature_index, variable_name in enumerate(input_variables)
        },
        "train_min_by_variable": {
            variable_name: float(feature_min[:, feature_index].min())
            for feature_index, variable_name in enumerate(input_variables)
        },
        "train_max_by_variable": {
            variable_name: float(feature_max[:, feature_index].max())
            for feature_index, variable_name in enumerate(input_variables)
        },
        "cover_radius_by_variable": {
            variable_name: float(cover_radius_vector[feature_index])
            for feature_index, variable_name in enumerate(input_variables)
        },
        "box_size_by_variable": dict(box_size_by_variable),
        "lower_bound_by_variable": (
            {
                variable_name: float(lower_bounds[0, feature_index])
                for feature_index, variable_name in enumerate(input_variables)
                if np.isfinite(lower_bounds[0, feature_index])
            }
            if lower_bounds is not None
            else {}
        ),
        "nonnegative_variables": list(nonnegative_variables),
    }


def collapse_support_cover_box_profile_to_global_max(
    profile: dict[str, object],
    *,
    global_box_size_override: float | None = None,
) -> dict[str, object]:
    input_variables = list(profile.get("input_variables", []))
    if not input_variables:
        raise ValueError("profile must include at least one input variable.")

    original_box_size_by_variable = {
        str(name): float(value)
        for name, value in dict(profile.get("box_size_by_variable", {})).items()
    }
    missing_variables = [name for name in input_variables if name not in original_box_size_by_variable]
    if missing_variables:
        missing_text = ", ".join(missing_variables)
        raise ValueError(f"profile is missing box sizes for variables: {missing_text}.")

    global_box_size = max(original_box_size_by_variable[name] for name in input_variables)
    if global_box_size_override is not None:
        global_box_size = float(global_box_size_override)
        if global_box_size <= 0.0:
            raise ValueError("global_box_size_override must be positive.")
    center = np.asarray(profile["center"], dtype=np.float32)
    box_size_by_feature = np.full(center.shape, global_box_size, dtype=np.float32)
    lower_bounds = profile.get("lower_bounds")
    lower_bound_array = None
    if lower_bounds is not None:
        lower_bound_array = np.asarray(lower_bounds, dtype=np.float32).copy()

    support_low_by_feature = center - box_size_by_feature / 2.0
    support_high_by_feature = center + box_size_by_feature / 2.0
    if lower_bound_array is not None:
        support_low_by_feature = np.maximum(support_low_by_feature, lower_bound_array)
        support_high_by_feature = np.maximum(support_high_by_feature, support_low_by_feature)

    scalar_box_size_by_variable = {
        variable_name: float(global_box_size)
        for variable_name in input_variables
    }
    return {
        **profile,
        "box_mode": "global_max",
        "global_box_size": float(global_box_size),
        "global_box_size_override": (
            None if global_box_size_override is None else float(global_box_size_override)
        ),
        "original_box_size_by_variable": original_box_size_by_variable,
        "box_size_by_variable": scalar_box_size_by_variable,
        "box_size_by_feature": box_size_by_feature,
        "support_low_by_feature": support_low_by_feature.astype(np.float32),
        "support_high_by_feature": support_high_by_feature.astype(np.float32),
        "lower_bounds": lower_bound_array,
    }


_compute_training_input_center = compute_training_input_center
_sample_uniform_box_inputs = sample_uniform_box_inputs
_resolve_variable_box_size_by_feature = resolve_variable_box_size_by_feature
_resolve_nonnegative_lower_bounds_by_feature = resolve_nonnegative_lower_bounds_by_feature
_resolve_causal_graph_box_label = resolve_causal_graph_box_label


# --- Former yrd/groups.py ---

import math

import pandas as pd


def build_city_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    grouped = frame.groupby("city_en")["station_id"].apply(list)
    return grouped.to_dict()


def _haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_km = 6371.0
    lon1_r, lat1_r, lon2_r, lat2_r = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def build_nearest_neighbor_groups(frame: pd.DataFrame, *, k: int = 3) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for _, row in frame.iterrows():
        distances = []
        for _, other in frame.iterrows():
            if row["station_id"] == other["station_id"]:
                continue
            distance = _haversine(row["lon"], row["lat"], other["lon"], other["lat"])
            distances.append((distance, other["station_id"]))
        distances.sort(key=lambda item: item[0])
        groups[row["station_id"]] = [station_id for _, station_id in distances[:k]]
    return groups


# --- Former yrd/data.py ---

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr



def build_time_splits(cfg: YRDExperimentConfig) -> dict[str, pd.Timestamp]:
    return {
        "train_end": cfg.train_end,
        "val_end": cfg.val_end,
        "test_end": cfg.test_end,
    }


def load_station_metadata(cfg: YRDExperimentConfig) -> pd.DataFrame:
    path = cfg.root_dir / cfg.station_path
    frame = pd.read_csv(path)
    return frame.sort_values("station_id").reset_index(drop=True)


def select_station_metadata(
    metadata: pd.DataFrame,
    *,
    available_station_ids: list[str],
    city_en: str | None = None,
    station_limit: int | None = None,
) -> pd.DataFrame:
    selected = metadata[metadata["station_id"].isin(available_station_ids)].copy()
    if city_en is not None:
        selected = selected[selected["city_en"].str.lower() == city_en.lower()]
    selected = selected.sort_values("station_id").reset_index(drop=True)
    if station_limit is not None:
        selected = selected.head(station_limit).reset_index(drop=True)
    return selected


def load_dataset(
    cfg: YRDExperimentConfig,
    *,
    smoke: bool = False,
    city_en: str | None = None,
) -> tuple[xr.Dataset, pd.DataFrame]:
    metadata = load_station_metadata(cfg)
    ds = xr.open_dataset(cfg.root_dir / cfg.dataset_path)
    ds = ds[list(cfg.input_variables)].transpose("time", "station")

    station_limit = cfg.smoke_station_count if smoke else None
    metadata = select_station_metadata(
        metadata,
        available_station_ids=ds["station"].values.tolist(),
        city_en=city_en,
        station_limit=station_limit,
    )
    ds = ds.sel(station=metadata["station_id"].tolist())
    return ds, metadata


def standardize_dataset(
    ds: xr.Dataset,
    *,
    train_end: np.datetime64 | pd.Timestamp,
) -> tuple[xr.Dataset, dict[str, dict[str, float]]]:
    train = ds.sel(time=slice(None, train_end))
    stats: dict[str, dict[str, float]] = {}
    scaled = ds.copy()
    for name, da in ds.data_vars.items():
        mean = float(train[name].mean().item())
        std = float(train[name].std().item())
        if std == 0.0:
            std = 1.0
        scaled[name] = (da - mean) / std
        stats[name] = {"mean": mean, "std": std}
    return scaled, stats


def _target_names(metadata: pd.DataFrame, target_variables: tuple[str, ...]) -> list[str]:
    names: list[str] = []
    for _, row in metadata.iterrows():
        for variable in target_variables:
            names.append(f"{row['city_en']}__{row['station_id']}__{variable}")
    return names


def _future_split_name(
    future_time: pd.Timestamp,
    *,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    test_end: pd.Timestamp,
) -> str | None:
    if future_time <= train_end:
        return "train"
    if future_time <= val_end:
        return "val"
    if future_time <= test_end:
        return "test"
    return None


def build_windowed_samples(
    ds: xr.Dataset,
    metadata: pd.DataFrame,
    cfg: YRDExperimentConfig,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    scaled, stats = standardize_dataset(ds, train_end=cfg.train_end)
    feature_values = np.stack(
        [scaled[name].values.astype(np.float32) for name in cfg.input_variables],
        axis=-1,
    )
    target_values = np.stack(
        [scaled[name].values.astype(np.float32) for name in cfg.target_variables],
        axis=-1,
    )

    times = pd.to_datetime(ds["time"].values)
    max_horizon = max(cfg.horizons)
    n_time, n_stations, _ = feature_values.shape

    split_data: dict[str, dict[str, Any]] = {
        split: {"X": [], "times": [], "targets": {h: [] for h in cfg.horizons}}
        for split in ("train", "val", "test")
    }

    for end_index in range(cfg.history_hours - 1, n_time - max_horizon):
        x_window = feature_values[end_index - cfg.history_hours + 1 : end_index + 1]
        future_time = pd.Timestamp(times[end_index + max_horizon])
        split_name = _future_split_name(
            future_time,
            train_end=cfg.train_end,
            val_end=cfg.val_end,
            test_end=cfg.test_end,
        )
        if split_name is None:
            continue

        if smoke and len(split_data[split_name]["X"]) >= cfg.smoke_samples_per_split:
            continue

        split_data[split_name]["X"].append(x_window)
        split_data[split_name]["times"].append(future_time.isoformat())
        for horizon in cfg.horizons:
            target = target_values[end_index + horizon].reshape(-1)
            split_data[split_name]["targets"][horizon].append(target)

    target_names = _target_names(metadata, cfg.target_variables)
    target_dim = n_stations * len(cfg.target_variables)
    for split_name, payload in split_data.items():
        payload["X"] = _stack_or_empty(
            payload["X"],
            shape=(0, cfg.history_hours, n_stations, len(cfg.input_variables)),
        )
        payload["times"] = list(payload["times"])
        payload["targets"] = {
            horizon: _stack_or_empty(values, shape=(0, target_dim))
            for horizon, values in payload["targets"].items()
        }

    return {
        "splits": split_data,
        "stats": stats,
        "target_names": target_names,
        "station_ids": metadata["station_id"].tolist(),
        "city_names": metadata["city_en"].tolist(),
        "n_stations": n_stations,
        "n_features": len(cfg.input_variables),
    }


def _stack_or_empty(values: list[np.ndarray], *, shape: tuple[int, ...]) -> np.ndarray:
    if values:
        return np.stack(values, axis=0).astype(np.float32)
    return np.empty(shape, dtype=np.float32)


def build_one_step_samples(
    ds: xr.Dataset,
    metadata: pd.DataFrame,
    cfg: YRDExperimentConfig,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    scaled, stats = standardize_dataset(ds, train_end=cfg.train_end)
    feature_values = np.stack(
        [scaled[name].values.astype(np.float32) for name in cfg.input_variables],
        axis=-1,
    )
    target_values = np.stack(
        [scaled[name].values.astype(np.float32) for name in cfg.target_variables],
        axis=-1,
    )

    times = pd.to_datetime(ds["time"].values)
    n_time, n_stations, n_features = feature_values.shape
    max_horizon = max(cfg.horizons)

    split_data: dict[str, dict[str, Any]] = {
        split: {"X": [], "times": [], "targets": {h: [] for h in cfg.horizons}}
        for split in ("train", "val", "test")
    }

    for current_index in range(n_time - max_horizon):
        future_time = pd.Timestamp(times[current_index + max_horizon])
        split_name = _future_split_name(
            future_time,
            train_end=cfg.train_end,
            val_end=cfg.val_end,
            test_end=cfg.test_end,
        )
        if split_name is None:
            continue

        if smoke and len(split_data[split_name]["X"]) >= cfg.smoke_samples_per_split:
            continue

        split_data[split_name]["X"].append(feature_values[current_index])
        split_data[split_name]["times"].append(future_time.isoformat())
        for horizon in cfg.horizons:
            split_data[split_name]["targets"][horizon].append(target_values[current_index + horizon].reshape(-1))

    target_names = _target_names(metadata, cfg.target_variables)
    target_dim = n_stations * len(cfg.target_variables)
    for split_name, payload in split_data.items():
        payload["X"] = _stack_or_empty(payload["X"], shape=(0, n_stations, n_features))
        payload["times"] = list(payload["times"])
        payload["targets"] = {
            horizon: _stack_or_empty(values, shape=(0, target_dim))
            for horizon, values in payload["targets"].items()
        }

    return {
        "splits": split_data,
        "stats": stats,
        "target_names": target_names,
        "station_ids": metadata["station_id"].tolist(),
        "city_names": metadata["city_en"].tolist(),
        "n_stations": n_stations,
        "n_features": len(cfg.input_variables),
    }


def flatten_input_group_indices(
    cfg: YRDExperimentConfig,
    *,
    n_stations: int,
    station_index: int = 0,
) -> dict[str, list[int]]:
    n_features = len(cfg.input_variables)
    groups: dict[str, list[int]] = {
        "local_o3_history": [],
        "local_pm25_history": [],
        "local_meteorology_history": [],
        "cross_station_pollutants": [],
    }

    for hour_index in range(cfg.history_hours):
        for local_feature_name, group_name in (
            ("O3", "local_o3_history"),
            ("PM2.5", "local_pm25_history"),
        ):
            feature_index = cfg.input_variables.index(local_feature_name)
            flat_index = ((hour_index * n_stations) + station_index) * n_features + feature_index
            groups[group_name].append(flat_index)

        for met_name in cfg.meteorology_variables:
            feature_index = cfg.input_variables.index(met_name)
            flat_index = ((hour_index * n_stations) + station_index) * n_features + feature_index
            groups["local_meteorology_history"].append(flat_index)

        for other_station in range(n_stations):
            if other_station == station_index:
                continue
            for pollutant_name in cfg.target_variables:
                feature_index = cfg.input_variables.index(pollutant_name)
                flat_index = ((hour_index * n_stations) + other_station) * n_features + feature_index
                groups["cross_station_pollutants"].append(flat_index)

    return groups


# --- Former yrd/models.py ---

import torch


def _build_activation(name: str) -> torch.nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return torch.nn.ReLU()
    if normalized == "silu":
        return torch.nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def _build_norm(name: str, hidden_dim: int) -> torch.nn.Module:
    normalized = name.lower()
    if normalized == "layernorm":
        return torch.nn.LayerNorm(hidden_dim)
    if normalized == "rmsnorm":
        rmsnorm = getattr(torch.nn, "RMSNorm", None)
        if rmsnorm is None:
            return torch.nn.LayerNorm(hidden_dim)
        return rmsnorm(hidden_dim)
    raise ValueError(f"Unsupported norm type: {name}")


class ResidualMLPBlock(torch.nn.Module):
    def __init__(self, hidden_dim: int, *, dropout: float, norm_type: str, activation: str) -> None:
        super().__init__()
        self.norm = _build_norm(norm_type, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.activation = _build_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class PersistenceBaseline(torch.nn.Module):
    def __init__(self, target_dim: int, horizons: tuple[int, ...]) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.horizons = tuple(horizons)

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        if x.ndim == 4:
            latest_snapshot = x[:, -1]
        elif x.ndim == 3:
            latest_snapshot = x
        else:
            raise ValueError(f"PersistenceBaseline expects 3D or 4D input, got shape {tuple(x.shape)}.")
        batch_size, n_stations, n_features = latest_snapshot.shape
        if self.target_dim % n_stations == 0:
            target_features = self.target_dim // n_stations
            if target_features <= n_features:
                last = latest_snapshot[:, :, :target_features].reshape(batch_size, -1)
            else:
                last = latest_snapshot.reshape(batch_size, -1)[:, : self.target_dim]
        else:
            last = latest_snapshot.reshape(batch_size, -1)[:, : self.target_dim]
        return {horizon: last for horizon in self.horizons}


class SingleStationMLP(torch.nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        history_hours: int,
        target_dim: int,
        hidden_dim: int,
        horizons: tuple[int, ...],
    ) -> None:
        super().__init__()
        input_dim = n_features * history_hours
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.heads = torch.nn.ModuleDict(
            {str(horizon): torch.nn.Linear(hidden_dim, target_dim) for horizon in horizons}
        )

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        hidden = self.trunk(x.reshape(x.shape[0], -1))
        return {int(horizon): head(hidden) for horizon, head in self.heads.items()}


class JointStationMLP(torch.nn.Module):
    def __init__(
        self,
        *,
        n_stations: int,
        n_features: int,
        history_hours: int,
        target_dim: int,
        hidden_dim: int,
        horizons: tuple[int, ...],
        model_name: str = "baseline",
        num_layers: int = 2,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        input_dim = n_stations * n_features * history_hours
        self.model_name = model_name
        self.model_kwargs = {
            "n_stations": n_stations,
            "n_features": n_features,
            "history_hours": history_hours,
            "target_dim": target_dim,
            "hidden_dim": hidden_dim,
            "horizons": tuple(horizons),
            "model_name": model_name,
            "num_layers": num_layers,
            "dropout": dropout,
            "norm_type": norm_type,
            "activation": activation,
        }

        if model_name == "baseline":
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            )
        elif model_name == "resmlp":
            layers: list[torch.nn.Module] = [
                torch.nn.Linear(input_dim, hidden_dim),
                _build_activation(activation),
            ]
            block_count = max(1, num_layers)
            for _ in range(block_count):
                layers.append(
                    ResidualMLPBlock(
                        hidden_dim,
                        dropout=dropout,
                        norm_type=norm_type,
                        activation=activation,
                    )
                )
            layers.append(_build_norm(norm_type, hidden_dim))
            self.trunk = torch.nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        self.heads = torch.nn.ModuleDict(
            {str(horizon): torch.nn.Linear(hidden_dim, target_dim) for horizon in horizons}
        )

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        hidden = self.trunk(x.reshape(x.shape[0], -1))
        return {int(horizon): head(hidden) for horizon, head in self.heads.items()}


# --- Former yrd/coupling.py ---

import json
import math
from pathlib import Path
from typing import Callable
from itertools import combinations

import numpy as np
import torch



def build_target_index_map(target_names: list[str]) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    for index, name in enumerate(target_names):
        city, _, variable = name.split("__")
        key = f"{city.lower()}_{variable.lower()}"
        mapping.setdefault(key, []).append(index)
        mapping.setdefault(f"all_{variable.lower()}", []).append(index)
    return mapping


def select_evenly_spaced_indices(n_samples: int, sample_count: int) -> list[int]:
    if n_samples <= 0 or sample_count <= 0:
        return []
    if sample_count >= n_samples:
        return list(range(n_samples))
    return np.linspace(0, n_samples - 1, num=sample_count, dtype=int).tolist()


def _summary_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
    }


def summarize_coupling_summaries(summaries: list[dict[str, object]]) -> dict[str, object]:
    if not summaries:
        return {
            "sample_count": 0,
            "ei_nis": _summary_stats([0.0]),
            "syn_nis": _summary_stats([0.0]),
            "group_ei_nis": {},
        }

    group_names = sorted(
        {
            name
            for summary in summaries
            for name in dict(summary.get("group_ei_nis", {})).keys()
        }
    )
    return {
        "sample_count": len(summaries),
        "ei_nis": _summary_stats([float(summary["ei_nis"]) for summary in summaries]),
        "syn_nis": _summary_stats([float(summary["syn_nis"]) for summary in summaries]),
        "group_ei_nis": {
            name: _summary_stats(
                [float(dict(summary.get("group_ei_nis", {})).get(name, 0.0)) for summary in summaries]
            )
            for name in group_names
        },
    }


def jacobian_for_target_subset(
    model: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    target_indices: list[int],
) -> torch.Tensor:
    y = model(x)
    rows = []
    for index in target_indices:
        grad = torch.autograd.grad(y[index], x, retain_graph=True)[0]
        rows.append(grad)
    return torch.stack(rows, dim=0)


def estimate_residual_covariance(y_true: np.ndarray, y_pred: np.ndarray, *, atol: float = 1e-6) -> np.ndarray:
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    if residuals.ndim != 2:
        raise ValueError("Residuals must be 2D: [samples, target_dim].")
    covariance = np.cov(residuals, rowvar=False)
    if covariance.ndim == 0:
        covariance = np.array([[float(covariance)]], dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    covariance += np.eye(covariance.shape[0]) * atol
    return covariance


def _subset_ei_nis(
    jacobian: np.ndarray,
    sigma_eps: np.ndarray,
    subset: list[int],
    *,
    box_size: float,
    atol: float,
) -> float:
    if not subset:
        return 0.0

    subset = sorted(set(subset))
    all_sources = list(range(jacobian.shape[1]))
    complement = [index for index in all_sources if index not in subset]
    effective_noise = np.asarray(sigma_eps, dtype=float).copy()
    if complement:
        omitted_block = jacobian[:, complement]
        intervention_variance = (box_size**2) / 12.0
        effective_noise = effective_noise + intervention_variance * omitted_block @ omitted_block.T

    signal_block = jacobian[:, subset]
    gram = signal_block.T @ np.linalg.pinv(effective_noise, rcond=atol) @ signal_block
    gram = 0.5 * (gram + gram.T)
    eigvals = np.linalg.eigvalsh(gram)
    positive = eigvals[eigvals > atol]
    if positive.size == 0:
        return 0.0

    return float(
        len(subset) * math.log(box_size)
        - 0.5 * len(subset) * math.log(2.0 * math.pi * math.e)
        + 0.5 * np.log(positive).sum()
    )


def _prepare_target_view(
    jacobian: np.ndarray,
    sigma_eps: np.ndarray,
    target_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    if jacobian.shape[0] != len(target_indices):
        target_jacobian = jacobian[np.ix_(target_indices, list(range(jacobian.shape[1])))]
    else:
        target_jacobian = jacobian
    if sigma_eps.shape[0] != target_jacobian.shape[0]:
        target_sigma_eps = sigma_eps[np.ix_(target_indices, target_indices)]
    else:
        target_sigma_eps = sigma_eps
    return target_jacobian, target_sigma_eps


def _coerce_2d_array(array: np.ndarray) -> np.ndarray:
    matrix = np.asarray(array, dtype=float)
    if matrix.ndim == 1:
        return matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError("Expected a 1D or 2D array of empirical samples.")
    return matrix


def compute_group_ei_summary(
    *,
    method: str,
    source_groups: dict[str, list[int]],
    source_samples: np.ndarray | None = None,
    target_samples: np.ndarray | None = None,
    jacobian: np.ndarray | None = None,
    sigma_eps: np.ndarray | None = None,
    target_indices: list[int] | None = None,
    box_size: float = math.sqrt(12.0),
    atol: float = 1e-12,
) -> dict[str, object]:
    normalized_groups = {
        str(name): sorted(set(int(index) for index in indices))
        for name, indices in source_groups.items()
    }
    if method == "tm":
        if source_samples is None or target_samples is None:
            raise ValueError("tm summary requires source_samples and target_samples.")
        source_matrix = _coerce_2d_array(source_samples)
        target_matrix = _coerce_2d_array(target_samples)
        if source_matrix.shape[0] != target_matrix.shape[0]:
            raise ValueError("source_samples and target_samples must share the sample axis.")
        group_ei: dict[str, float] = {}
        overall_backend = "affine_triangular_transport_map"
        for group_name, indices in normalized_groups.items():
            summary = estimate_mutual_information_transport_map(source_matrix[:, indices], target_matrix)
            group_ei[group_name] = clip_nonnegative_ei(summary["mi_hat"])
            overall_backend = str(summary["backend"])
        all_indices = sorted({index for indices in normalized_groups.values() for index in indices})
        if not all_indices:
            overall_ei = 0.0
        else:
            overall = estimate_mutual_information_transport_map(source_matrix[:, all_indices], target_matrix)
            overall_ei = clip_nonnegative_ei(overall["mi_hat"])
            overall_backend = str(overall["backend"])
        return {
            "method": "tm",
            "backend": overall_backend,
            "ei": float(overall_ei),
            "group_ei": group_ei,
        }
    if method == "nis":
        if jacobian is None or sigma_eps is None or target_indices is None:
            raise ValueError("nis summary requires jacobian, sigma_eps, and target_indices.")
        summary = compute_subset_nis_summary(
            jacobian=np.asarray(jacobian, dtype=float),
            sigma_eps=np.asarray(sigma_eps, dtype=float),
            source_groups=normalized_groups,
            target_indices=list(target_indices),
            box_size=box_size,
            atol=atol,
        )
        return {
            "method": "nis",
            "backend": "nis_local_linear_gaussian",
            "ei": float(summary["ei_nis"]),
            "group_ei": {name: float(value) for name, value in summary["group_ei_nis"].items()},
        }
    raise ValueError(f"Unsupported coupling method: {method}")


def compute_group_synergy_summary(
    *,
    method: str,
    source_groups: dict[str, list[int]],
    source_samples: np.ndarray | None = None,
    target_samples: np.ndarray | None = None,
    jacobian: np.ndarray | None = None,
    sigma_eps: np.ndarray | None = None,
    target_indices: list[int] | None = None,
    box_size: float = math.sqrt(12.0),
    atol: float = 1e-12,
) -> dict[str, object]:
    if method == "nis":
        if jacobian is None or sigma_eps is None or target_indices is None:
            raise ValueError("nis summary requires jacobian, sigma_eps, and target_indices.")
        summary = compute_subset_nis_summary(
            jacobian=np.asarray(jacobian, dtype=float),
            sigma_eps=np.asarray(sigma_eps, dtype=float),
            source_groups=source_groups,
            target_indices=list(target_indices),
            box_size=box_size,
            atol=atol,
        )
        return {
            "method": "nis",
            "backend": "nis_local_linear_gaussian",
            "ei": float(summary["ei_nis"]),
            "syn": float(summary["syn_nis"]),
            "group_ei": {name: float(value) for name, value in summary["group_ei_nis"].items()},
        }
    base = compute_group_ei_summary(
        method=method,
        source_groups=source_groups,
        source_samples=source_samples,
        target_samples=target_samples,
        jacobian=jacobian,
        sigma_eps=sigma_eps,
        target_indices=target_indices,
        box_size=box_size,
        atol=atol,
    )
    return {
        **base,
        "syn": float(base["ei"] - sum(float(value) for value in dict(base["group_ei"]).values())),
    }


def compute_subset_nis_summary(
    *,
    jacobian: np.ndarray,
    sigma_eps: np.ndarray,
    source_groups: dict[str, list[int]],
    target_indices: list[int],
    box_size: float = math.sqrt(12.0),
    atol: float = 1e-12,
) -> dict[str, object]:
    target_jacobian, target_sigma_eps = _prepare_target_view(jacobian, sigma_eps, target_indices)

    group_eis = {
        name: clip_nonnegative_ei(
            _subset_ei_nis(target_jacobian, target_sigma_eps, indices, box_size=box_size, atol=atol)
        )
        for name, indices in source_groups.items()
    }
    whole_sources = sorted({index for indices in source_groups.values() for index in indices})
    ei_full = clip_nonnegative_ei(
        _subset_ei_nis(target_jacobian, target_sigma_eps, whole_sources, box_size=box_size, atol=atol)
    )
    syn_nis = float(ei_full - sum(group_eis.values()))
    return {
        "ei_nis": float(ei_full),
        "syn_nis": syn_nis,
        "target_dim": int(target_jacobian.shape[0]),
        "n_source_groups": len(source_groups),
        "group_ei_nis": {name: float(value) for name, value in group_eis.items()},
    }


def build_station_source_groups(
    *,
    history_hours: int,
    n_stations: int,
    n_features: int,
    station_ids: list[str] | None = None,
) -> dict[str, list[int]]:
    names = station_ids or [str(index) for index in range(n_stations)]
    if len(names) != n_stations:
        raise ValueError("station_ids must have length n_stations.")

    groups: dict[str, list[int]] = {}
    for station_index, station_id in enumerate(names):
        indices: list[int] = []
        for hour_index in range(history_hours):
            base = (hour_index * n_stations + station_index) * n_features
            indices.extend(range(base, base + n_features))
        groups[str(station_id)] = indices
    return groups


def build_one_step_station_source_groups(
    *,
    n_stations: int,
    n_features: int,
    station_ids: list[str] | None = None,
) -> dict[str, list[int]]:
    names = station_ids or [str(index) for index in range(n_stations)]
    if len(names) != n_stations:
        raise ValueError("station_ids must have length n_stations.")

    groups: dict[str, list[int]] = {}
    for station_index, station_id in enumerate(names):
        start = station_index * n_features
        groups[str(station_id)] = list(range(start, start + n_features))
    return groups


def build_one_step_station_pollutant_feature_groups(
    *,
    n_stations: int,
    n_features: int,
    pollutant_feature_indices: dict[str, int],
    station_ids: list[str] | None = None,
) -> dict[str, dict[str, list[int]]]:
    names = station_ids or [str(index) for index in range(n_stations)]
    if len(names) != n_stations:
        raise ValueError("station_ids must have length n_stations.")

    groups: dict[str, dict[str, list[int]]] = {}
    for station_index, station_id in enumerate(names):
        start = station_index * n_features
        station_groups: dict[str, list[int]] = {}
        for feature_name, feature_index in pollutant_feature_indices.items():
            if feature_index < 0 or feature_index >= n_features:
                raise ValueError("pollutant feature index is out of range.")
            station_groups[str(feature_name)] = [start + int(feature_index)]
        groups[str(station_id)] = station_groups
    return groups


def compute_station_level_nis_summary(
    *,
    jacobian: np.ndarray,
    sigma_eps: np.ndarray,
    station_source_groups: dict[str, list[int]],
    target_indices: list[int],
    box_size: float = math.sqrt(12.0),
    atol: float = 1e-12,
) -> dict[str, object]:
    base_summary = compute_subset_nis_summary(
        jacobian=jacobian,
        sigma_eps=sigma_eps,
        source_groups=station_source_groups,
        target_indices=target_indices,
        box_size=box_size,
        atol=atol,
    )
    target_jacobian, target_sigma_eps = _prepare_target_view(jacobian, sigma_eps, target_indices)
    pair_ei = {
        station_id: float(value)
        for station_id, value in base_summary["group_ei_nis"].items()
    }
    binary_synergy: dict[str, float] = {}
    for left_name, right_name in combinations(station_source_groups.keys(), 2):
        pair_indices = sorted(set(station_source_groups[left_name] + station_source_groups[right_name]))
        pair_value = _subset_ei_nis(
            target_jacobian,
            target_sigma_eps,
            pair_indices,
            box_size=box_size,
            atol=atol,
        )
        pair_value = clip_nonnegative_ei(pair_value)
        binary_synergy[f"{left_name}|{right_name}"] = float(pair_value - pair_ei[left_name] - pair_ei[right_name])

    return {
        "ei_nis": float(base_summary["ei_nis"]),
        "syn_nis": float(base_summary["syn_nis"]),
        "ei": float(base_summary["ei_nis"]),
        "syn": float(base_summary["syn_nis"]),
        "pairwise_station_ei_nis": pair_ei,
        "pairwise_station_ei": pair_ei,
        "binary_station_synergy_nis": binary_synergy,
        "binary_station_synergy": binary_synergy,
    }


def compute_station_level_ei_summary(
    *,
    method: str,
    station_source_groups: dict[str, list[int]],
    source_samples: np.ndarray | None = None,
    target_samples: np.ndarray | None = None,
    jacobian: np.ndarray | None = None,
    sigma_eps: np.ndarray | None = None,
    target_indices: list[int] | None = None,
    box_size: float = math.sqrt(12.0),
    atol: float = 1e-12,
) -> dict[str, object]:
    if method == "nis":
        summary = compute_station_level_nis_summary(
            jacobian=np.asarray(jacobian, dtype=float),
            sigma_eps=np.asarray(sigma_eps, dtype=float),
            station_source_groups=station_source_groups,
            target_indices=[] if target_indices is None else list(target_indices),
            box_size=box_size,
            atol=atol,
        )
        return {
            "method": "nis",
            "backend": "nis_local_linear_gaussian",
            **summary,
        }
    if method != "tm":
        raise ValueError(f"Unsupported coupling method: {method}")
    summary = compute_group_synergy_summary(
        method="tm",
        source_groups=station_source_groups,
        source_samples=source_samples,
        target_samples=target_samples,
    )
    return {
        "method": "tm",
        "backend": str(summary["backend"]),
        "ei": float(summary["ei"]),
        "syn": float(summary["syn"]),
        "pairwise_station_ei": {name: float(value) for name, value in summary["group_ei"].items()},
        "binary_station_synergy": {},
    }


def compute_station_pollutant_pair_synergy_summary(
    *,
    jacobian: np.ndarray | None = None,
    sigma_eps: np.ndarray | None = None,
    station_pollutant_feature_groups: dict[str, dict[str, list[int]]],
    target_indices: list[int] | None = None,
    method: str = "nis",
    source_samples: np.ndarray | None = None,
    target_samples: np.ndarray | None = None,
    left_feature: str = "O3",
    right_feature: str = "PM2.5",
    box_size: float = math.sqrt(12.0),
    atol: float = 1e-12,
) -> dict[str, object]:
    if method == "tm":
        if source_samples is None or target_samples is None:
            raise ValueError("tm pollutant-pair synergy requires source_samples and target_samples.")
        source_matrix = _coerce_2d_array(source_samples)
        target_matrix = _coerce_2d_array(target_samples)
        single_pollutant_ei: dict[str, dict[str, float]] = {}
        joint_station_pair_ei: dict[str, float] = {}
        station_pair_synergy: dict[str, float] = {}
        backend = "affine_triangular_transport_map"
        for station_id, feature_groups in station_pollutant_feature_groups.items():
            left_indices = list(feature_groups.get(left_feature, []))
            right_indices = list(feature_groups.get(right_feature, []))
            left_summary = estimate_mutual_information_transport_map(source_matrix[:, left_indices], target_matrix)
            right_summary = estimate_mutual_information_transport_map(source_matrix[:, right_indices], target_matrix)
            joint_indices = sorted(set(left_indices + right_indices))
            joint_summary = estimate_mutual_information_transport_map(source_matrix[:, joint_indices], target_matrix)
            backend = str(joint_summary["backend"])
            left_value = clip_nonnegative_ei(left_summary["mi_hat"])
            right_value = clip_nonnegative_ei(right_summary["mi_hat"])
            joint_value = clip_nonnegative_ei(joint_summary["mi_hat"])
            single_pollutant_ei[str(station_id)] = {
                str(left_feature): left_value,
                str(right_feature): right_value,
            }
            joint_station_pair_ei[str(station_id)] = joint_value
            station_pair_synergy[str(station_id)] = float(joint_value - left_value - right_value)
        return {
            "method": "tm",
            "backend": backend,
            "joint_station_pair_ei": joint_station_pair_ei,
            "single_pollutant_ei": single_pollutant_ei,
            "station_pair_synergy": station_pair_synergy,
            "joint_station_pair_ei_nis": joint_station_pair_ei,
            "single_pollutant_ei_nis": single_pollutant_ei,
            "station_pair_synergy_nis": station_pair_synergy,
        }

    if jacobian is None or sigma_eps is None or target_indices is None:
        raise ValueError("nis pollutant-pair synergy requires jacobian, sigma_eps, and target_indices.")
    target_jacobian, target_sigma_eps = _prepare_target_view(jacobian, sigma_eps, target_indices)

    single_pollutant_ei: dict[str, dict[str, float]] = {}
    joint_station_pair_ei: dict[str, float] = {}
    station_pair_synergy: dict[str, float] = {}
    for station_id, feature_groups in station_pollutant_feature_groups.items():
        left_indices = list(feature_groups.get(left_feature, []))
        right_indices = list(feature_groups.get(right_feature, []))
        left_value = _subset_ei_nis(
            target_jacobian,
            target_sigma_eps,
            left_indices,
            box_size=box_size,
            atol=atol,
        )
        left_value = clip_nonnegative_ei(left_value)
        right_value = _subset_ei_nis(
            target_jacobian,
            target_sigma_eps,
            right_indices,
            box_size=box_size,
            atol=atol,
        )
        right_value = clip_nonnegative_ei(right_value)
        joint_indices = sorted(set(left_indices + right_indices))
        joint_value = _subset_ei_nis(
            target_jacobian,
            target_sigma_eps,
            joint_indices,
            box_size=box_size,
            atol=atol,
        )
        joint_value = clip_nonnegative_ei(joint_value)
        single_pollutant_ei[str(station_id)] = {
            str(left_feature): float(left_value),
            str(right_feature): float(right_value),
        }
        joint_station_pair_ei[str(station_id)] = float(joint_value)
        station_pair_synergy[str(station_id)] = float(joint_value - left_value - right_value)

    return {
        "method": "nis",
        "backend": "nis_local_linear_gaussian",
        "joint_station_pair_ei": joint_station_pair_ei,
        "single_pollutant_ei": single_pollutant_ei,
        "station_pair_synergy": station_pair_synergy,
        "joint_station_pair_ei_nis": joint_station_pair_ei,
        "single_pollutant_ei_nis": single_pollutant_ei,
        "station_pair_synergy_nis": station_pair_synergy,
    }


def summarize_global_station_coupling(
    sample_summaries: list[dict[str, object]],
    *,
    station_ids: list[str],
) -> dict[str, object]:
    if not sample_summaries:
        return {
            "pairwise_edges": [],
            "binary_hyperedges": [],
            "per_target_station": {},
        }

    per_target_station: dict[str, dict[str, object]] = {}
    edge_rows: list[dict[str, object]] = []
    hyperedge_rows: list[dict[str, object]] = []

    for target_station_id in station_ids:
        target_rows = [row for row in sample_summaries if row.get("target_station_id") == target_station_id]
        if not target_rows:
            continue
        pair_names = sorted(
            {
                key
                for row in target_rows
                for key in dict(row.get("pairwise_station_ei", row.get("pairwise_station_ei_nis", {}))).keys()
            }
        )
        binary_names = sorted(
            {
                key
                for row in target_rows
                for key in dict(row.get("binary_station_synergy", row.get("binary_station_synergy_nis", {}))).keys()
            }
        )
        pair_summary = {
            name: _summary_stats(
                [
                    float(dict(row.get("pairwise_station_ei", row.get("pairwise_station_ei_nis", {}))).get(name, 0.0))
                    for row in target_rows
                ]
            )
            for name in pair_names
        }
        binary_summary = {
            name: _summary_stats(
                [
                    float(
                        dict(row.get("binary_station_synergy", row.get("binary_station_synergy_nis", {}))).get(
                            name,
                            0.0,
                        )
                    )
                    for row in target_rows
                ]
            )
            for name in binary_names
        }
        per_target_station[target_station_id] = {
            "sample_count": len(target_rows),
            "pairwise_station_ei_nis": pair_summary,
            "binary_station_synergy_nis": binary_summary,
        }

        for source_station_id, stats in pair_summary.items():
            edge_rows.append(
                {
                    "source_station_id": source_station_id,
                    "target_station_id": target_station_id,
                    "mean": float(stats["mean"]),
                    "std": float(stats["std"]),
                    "median": float(stats["median"]),
                }
            )
        for pair_name, stats in binary_summary.items():
            source_pair = tuple(pair_name.split("|"))
            hyperedge_rows.append(
                {
                    "source_station_ids": source_pair,
                    "target_station_id": target_station_id,
                    "mean": float(stats["mean"]),
                    "std": float(stats["std"]),
                    "median": float(stats["median"]),
                }
            )

    edge_rows.sort(key=lambda row: (row["target_station_id"], -abs(float(row["mean"])), row["source_station_id"]))
    hyperedge_rows.sort(
        key=lambda row: (row["target_station_id"], -abs(float(row["mean"])), row["source_station_ids"])
    )
    return {
        "pairwise_edges": edge_rows,
        "binary_hyperedges": hyperedge_rows,
        "per_target_station": per_target_station,
    }


def summarize_global_station_pollutant_synergy(
    sample_summaries: list[dict[str, object]],
    *,
    station_ids: list[str],
) -> dict[str, object]:
    if not sample_summaries:
        return {
            "conditional_synergy_edges": [],
            "conditional_synergy_ratio_edges": [],
            "per_target_station": {},
        }

    per_target_station: dict[str, dict[str, object]] = {}
    edge_rows: list[dict[str, object]] = []
    ratio_edge_rows: list[dict[str, object]] = []

    for target_station_id in station_ids:
        target_rows = [row for row in sample_summaries if row.get("target_station_id") == target_station_id]
        if not target_rows:
            continue

        synergy_summary = {
            station_id: _summary_stats(
                [
                    float(dict(row.get("station_pair_synergy", row.get("station_pair_synergy_nis", {}))).get(station_id, 0.0))
                    for row in target_rows
                ]
            )
            for station_id in station_ids
        }
        joint_summary = {
            station_id: _summary_stats(
                [
                    float(
                        dict(row.get("joint_station_pair_ei", row.get("joint_station_pair_ei_nis", {}))).get(
                            station_id,
                            0.0,
                        )
                    )
                    for row in target_rows
                ]
            )
            for station_id in station_ids
        }
        ratio_summary = {}
        for station_id in station_ids:
            samplewise_ratio_values = [
                (
                    float(dict(row.get("station_pair_synergy", row.get("station_pair_synergy_nis", {}))).get(station_id, 0.0))
                    / float(
                        dict(row.get("joint_station_pair_ei", row.get("joint_station_pair_ei_nis", {}))).get(
                            station_id,
                            0.0,
                        )
                    )
                )
                if abs(
                    float(
                        dict(row.get("joint_station_pair_ei", row.get("joint_station_pair_ei_nis", {}))).get(
                            station_id,
                            0.0,
                        )
                    )
                ) > 1e-12
                else 0.0
                for row in target_rows
            ]
            ratio_stats = _summary_stats(samplewise_ratio_values)
            synergy_mean = float(synergy_summary[station_id]["mean"])
            joint_mean = float(joint_summary[station_id]["mean"])
            ratio_stats["mean"] = synergy_mean / joint_mean if abs(joint_mean) > 1e-12 else 0.0
            ratio_summary[station_id] = ratio_stats
        single_summary = {
            station_id: {
                feature_name: _summary_stats(
                    [
                        float(
                            dict(
                                dict(row.get("single_pollutant_ei", row.get("single_pollutant_ei_nis", {}))).get(
                                    station_id,
                                    {},
                                )
                            ).get(feature_name, 0.0)
                        )
                        for row in target_rows
                    ]
                )
                for feature_name in ("O3", "PM2.5")
            }
            for station_id in station_ids
        }
        per_target_station[target_station_id] = {
            "sample_count": len(target_rows),
            "station_pair_synergy_nis": synergy_summary,
            "joint_station_pair_ei_nis": joint_summary,
            "conditional_synergy_ratio_nis": ratio_summary,
            "single_pollutant_ei_nis": single_summary,
        }

        for source_station_id, stats in synergy_summary.items():
            edge_rows.append(
                {
                    "source_station_id": source_station_id,
                    "target_station_id": target_station_id,
                    "mean": float(stats["mean"]),
                    "std": float(stats["std"]),
                    "median": float(stats["median"]),
                }
            )
        for source_station_id, stats in ratio_summary.items():
            ratio_edge_rows.append(
                {
                    "source_station_id": source_station_id,
                    "target_station_id": target_station_id,
                    "mean": float(stats["mean"]),
                    "std": float(stats["std"]),
                    "median": float(stats["median"]),
                }
            )

    edge_rows.sort(key=lambda row: (row["target_station_id"], -abs(float(row["mean"])), row["source_station_id"]))
    ratio_edge_rows.sort(
        key=lambda row: (row["target_station_id"], -abs(float(row["mean"])), row["source_station_id"])
    )
    return {
        "conditional_synergy_edges": edge_rows,
        "conditional_synergy_ratio_edges": ratio_edge_rows,
        "per_target_station": per_target_station,
    }


def summarize_global_station_single_pollutant_ei(
    sample_summaries: list[dict[str, object]],
    *,
    station_ids: list[str],
    feature_name: str,
) -> dict[str, object]:
    if not sample_summaries:
        return {
            "pairwise_edges": [],
            "per_target_station": {},
            "feature_name": str(feature_name),
        }

    per_target_station: dict[str, dict[str, object]] = {}
    edge_rows: list[dict[str, object]] = []

    for target_station_id in station_ids:
        target_rows = [row for row in sample_summaries if row.get("target_station_id") == target_station_id]
        if not target_rows:
            continue

        feature_summary = {
            station_id: _summary_stats(
                [
                    float(
                        dict(
                            dict(row.get("single_pollutant_ei", row.get("single_pollutant_ei_nis", {}))).get(
                                station_id,
                                {},
                            )
                        ).get(feature_name, 0.0)
                    )
                    for row in target_rows
                ]
            )
            for station_id in station_ids
        }
        per_target_station[target_station_id] = {
            "sample_count": len(target_rows),
            "feature_name": str(feature_name),
            "single_feature_ei_nis": feature_summary,
        }
        for source_station_id, stats in feature_summary.items():
            edge_rows.append(
                {
                    "source_station_id": source_station_id,
                    "target_station_id": target_station_id,
                    "feature_name": str(feature_name),
                    "mean": float(stats["mean"]),
                    "std": float(stats["std"]),
                    "median": float(stats["median"]),
                }
            )

    edge_rows.sort(key=lambda row: (row["target_station_id"], -abs(float(row["mean"])), row["source_station_id"]))
    return {
        "pairwise_edges": edge_rows,
        "per_target_station": per_target_station,
        "feature_name": str(feature_name),
    }


def save_coupling_summary(summary: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


# --- Former yrd/train.py ---

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch



def joint_model_kwargs(
    *,
    n_stations: int,
    n_features: int,
    history_hours: int,
    target_dim: int,
    hidden_dim: int,
    horizons: tuple[int, ...],
    model_name: str = "baseline",
    num_layers: int = 2,
    dropout: float = 0.0,
    norm_type: str = "layernorm",
    activation: str = "relu",
) -> dict[str, Any]:
    return {
        "n_stations": n_stations,
        "n_features": n_features,
        "history_hours": history_hours,
        "target_dim": target_dim,
        "hidden_dim": hidden_dim,
        "horizons": tuple(horizons),
        "model_name": model_name,
        "num_layers": num_layers,
        "dropout": dropout,
        "norm_type": norm_type,
        "activation": activation,
    }


def rebuild_joint_model_from_checkpoint(payload: dict[str, Any]) -> JointStationMLP:
    model = JointStationMLP(**payload["model_kwargs"])
    model.load_state_dict(payload["state_dict"])
    return model


def ensure_output_layout(root: Path) -> dict[str, Path]:
    cache_dir = root / "exp" / "cache" / "yrd_coupling"
    results_dir = root / "fig" / "yrd_shanghai" / "artifacts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return {"cache_dir": cache_dir, "results_dir": results_dir}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).to(dtype=torch.float32)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_true - y_pred
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    corr = float(np.corrcoef(y_true.reshape(-1), y_pred.reshape(-1))[0, 1]) if y_true.size > 1 else 1.0
    return {"rmse": rmse, "mae": mae, "corr": corr}


def _predict_numpy(model: torch.nn.Module, x: np.ndarray, horizons: tuple[int, ...]) -> dict[int, np.ndarray]:
    model.eval()
    with torch.no_grad():
        outputs = model(_to_tensor(x))
    return {horizon: tensor.cpu().numpy() for horizon, tensor in outputs.items() if horizon in horizons}


def train_joint_model_with_history(
    *,
    n_stations: int,
    n_features: int,
    history_hours: int,
    target_dim: int,
    hidden_dim: int,
    horizons: tuple[int, ...],
    learning_rate: float,
    weight_decay: float = 0.0,
    batch_size: int,
    epochs: int,
    max_epochs: int | None = None,
    early_stopping_patience: int | None = None,
    seed: int,
    x_train: np.ndarray,
    y_train: dict[int, np.ndarray],
    x_val: np.ndarray,
    y_val: dict[int, np.ndarray],
    model_name: str = "baseline",
    num_layers: int = 2,
    dropout: float = 0.0,
    norm_type: str = "layernorm",
    activation: str = "relu",
) -> dict[str, Any]:
    set_seed(seed)
    model_kwargs = joint_model_kwargs(
        n_stations=n_stations,
        n_features=n_features,
        history_hours=history_hours,
        target_dim=target_dim,
        hidden_dim=hidden_dim,
        horizons=horizons,
        model_name=model_name,
        num_layers=num_layers,
        dropout=dropout,
        norm_type=norm_type,
        activation=activation,
    )
    model = JointStationMLP(**model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_train_tensor = _to_tensor(x_train)
    y_train_tensor = {h: _to_tensor(values) for h, values in y_train.items()}
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    stopped_early = False
    epochs_without_improvement = 0
    effective_epochs = max_epochs if max_epochs is not None else epochs

    for epoch in range(effective_epochs):
        model.train()
        permutation = torch.randperm(x_train_tensor.shape[0])
        batch_losses: list[float] = []
        for start in range(0, x_train_tensor.shape[0], batch_size):
            batch_indices = permutation[start : start + batch_size]
            batch_x = x_train_tensor[batch_indices]
            predictions = model(batch_x)
            loss = sum(
                torch.nn.functional.mse_loss(predictions[horizon], y_train_tensor[horizon][batch_indices])
                for horizon in horizons
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        train_loss_history.append(float(np.mean(batch_losses)))

        val_predictions = _predict_numpy(model, x_val, horizons)
        val_loss = float(sum(np.mean((val_predictions[horizon] - y_val[horizon]) ** 2) for horizon in horizons))
        val_loss_history.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "model": model,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "best_state_dict": best_state,
        "model_kwargs": model_kwargs,
        "stopped_early": stopped_early,
        "early_stopping_patience": early_stopping_patience,
    }


def _train_joint_model(
    cfg: YRDExperimentConfig,
    *,
    x_train: np.ndarray,
    y_train: dict[int, np.ndarray],
    x_val: np.ndarray,
    y_val: dict[int, np.ndarray],
    target_dim: int,
    n_stations: int,
    n_features: int,
) -> JointStationMLP:
    result = train_joint_model_with_history(
        n_stations=n_stations,
        n_features=n_features,
        history_hours=cfg.history_hours,
        target_dim=target_dim,
        hidden_dim=cfg.hidden_dim,
        horizons=cfg.horizons,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        max_epochs=cfg.max_epochs,
        early_stopping_patience=cfg.early_stopping_patience,
        seed=cfg.seed,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        model_name=cfg.model_name,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        norm_type=cfg.norm_type,
        activation=cfg.activation,
    )
    return result["model"]


def run_smoke_pipeline(cfg: YRDExperimentConfig) -> dict[str, Any]:
    layout = ensure_output_layout(cfg.root_dir)
    ds, metadata = load_dataset(cfg, smoke=True)
    sample_bundle = build_windowed_samples(ds, metadata, cfg, smoke=True)
    splits = sample_bundle["splits"]

    x_train = splits["train"]["X"]
    x_val = splits["val"]["X"]
    x_test = splits["test"]["X"]
    y_train = splits["train"]["targets"]
    y_val = splits["val"]["targets"]
    y_test = splits["test"]["targets"]

    target_dim = y_train[cfg.horizons[0]].shape[1]
    baseline = PersistenceBaseline(target_dim=target_dim, horizons=cfg.horizons)
    model = _train_joint_model(
        cfg,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        target_dim=target_dim,
        n_stations=sample_bundle["n_stations"],
        n_features=sample_bundle["n_features"],
    )

    baseline_predictions = _predict_numpy(baseline, x_test, cfg.horizons)
    joint_predictions = _predict_numpy(model, x_test, cfg.horizons)

    metrics = {
        "baseline_test": {
            str(horizon): _compute_metrics(y_test[horizon], baseline_predictions[horizon])
            for horizon in cfg.horizons
        },
        "joint_test": {
            str(horizon): _compute_metrics(y_test[horizon], joint_predictions[horizon])
            for horizon in cfg.horizons
        },
    }

    source_groups = flatten_input_group_indices(
        cfg,
        n_stations=sample_bundle["n_stations"],
        station_index=0,
    )
    sample_x = _to_tensor(x_test[:1]).reshape(-1).detach().clone().requires_grad_(True)
    coupling_summary: dict[str, dict[str, Any]] = {}
    for horizon in cfg.horizons:
        horizon_model = (
            lambda tensor, target_horizon=horizon: model(
                tensor.reshape(1, cfg.history_hours, sample_bundle["n_stations"], sample_bundle["n_features"])
            )[target_horizon].reshape(-1)
        )
        jacobian = torch.autograd.functional.jacobian(horizon_model, sample_x).detach().cpu().numpy()
        sigma_eps = estimate_residual_covariance(y_test[horizon], joint_predictions[horizon])
        coupling_summary[str(horizon)] = compute_subset_nis_summary(
            jacobian=jacobian,
            sigma_eps=sigma_eps,
            source_groups=source_groups,
            target_indices=list(range(jacobian.shape[0])),
            box_size=cfg.box_size,
        )

    metrics_path = layout["cache_dir"] / "smoke_metrics.json"
    coupling_path = layout["cache_dir"] / "smoke_coupling_summary.json"
    prediction_path = layout["cache_dir"] / "smoke_predictions.npz"
    save_json(metrics_path, metrics)
    save_coupling_summary(coupling_summary, coupling_path)
    np.savez(
        prediction_path,
        y_test_1h=y_test[cfg.horizons[0]],
        pred_test_1h=joint_predictions[cfg.horizons[0]],
        y_test_24h=y_test[cfg.horizons[1]],
        pred_test_24h=joint_predictions[cfg.horizons[1]],
    )

    metrics_md_path = layout["results_dir"] / "smoke_metrics_summary.md"
    coupling_md_path = layout["results_dir"] / "smoke_coupling_summary.md"
    write_markdown_summary(
        metrics_md_path,
        title="YRD Smoke Metrics Summary",
        intro="这一文件解释了 smoke 级别多站点 MLP 预测实验的主要误差指标。",
        bullets=metrics_bullets(metrics),
    )
    write_markdown_summary(
        coupling_md_path,
        title="YRD Smoke Coupling Summary",
        intro="这一文件解释了基于局部 Jacobian 与残差协方差得到的连续耦合摘要。",
        bullets=coupling_by_horizon_bullets(coupling_summary),
    )

    return {
        "metrics": metrics,
        "coupling_summary": coupling_summary,
        "metrics_path": metrics_path,
        "coupling_path": coupling_path,
        "metrics_md_path": metrics_md_path,
        "coupling_md_path": coupling_md_path,
        "target_names": sample_bundle["target_names"],
    }


# --- Former yrd/plotting.py ---

import html
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_horizon_comparison_plot(frame: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ax.plot(frame["horizon"], frame["syn_nis"], marker="o", label="Syn_nis")
    ax.set_xlabel("Horizon (hours)")
    ax.set_ylabel("Coupling strength")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _edge_strength(row: pd.Series) -> float:
    for key in ("abs_mean", "mean", "value"):
        if key in row and pd.notna(row[key]):
            return abs(float(row[key]))
    return 0.0


def _normalize_source_pair(value: object) -> tuple[str, str]:
    if isinstance(value, str):
        if "|" in value:
            left, right = value.split("|", maxsplit=1)
            return left.strip(), right.strip()
        if "," in value:
            left, right = value.split(",", maxsplit=1)
            return left.strip(), right.strip()
        raise ValueError("source_station_ids must contain two station ids.")

    pair = tuple(value)  # type: ignore[arg-type]
    if len(pair) != 2:
        raise ValueError("source_station_ids must contain exactly two station ids.")
    return str(pair[0]), str(pair[1])


def _scale_positions(frame: pd.DataFrame, *, width: float, height: float) -> dict[str, tuple[float, float]]:
    required = {"station_id", "lon", "lat"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"station_positions is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("station_positions must be non-empty.")

    left_margin = 72.0
    right_margin = 72.0
    top_margin = 82.0
    bottom_margin = 68.0
    plot_width = width - left_margin - right_margin - 180.0
    plot_height = height - top_margin - bottom_margin

    lon_min = float(frame["lon"].min())
    lon_max = float(frame["lon"].max())
    lat_min = float(frame["lat"].min())
    lat_max = float(frame["lat"].max())
    lon_span = max(lon_max - lon_min, 1e-9)
    lat_span = max(lat_max - lat_min, 1e-9)

    positions: dict[str, tuple[float, float]] = {}
    for _, row in frame.iterrows():
        x = left_margin + (float(row["lon"]) - lon_min) / lon_span * plot_width
        y = top_margin + (lat_max - float(row["lat"])) / lat_span * plot_height
        positions[str(row["station_id"])] = (x, y)
    return positions


def _quadratic_path(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    curvature: float,
    stroke: str,
    stroke_width: float,
    opacity: float,
    marker_end: str = "",
    dasharray: str = "",
) -> str:
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    distance = max(math.hypot(dx, dy), 1e-9)
    px = -dy / distance
    py = dx / distance
    cx = 0.5 * (x1 + x2) + curvature * distance * px
    cy = 0.5 * (y1 + y2) + curvature * distance * py
    marker = f" marker-end='{marker_end}'" if marker_end else ""
    dash = f" stroke-dasharray='{dasharray}'" if dasharray else ""
    return (
        f"<path d='M {x1:.1f} {y1:.1f} Q {cx:.1f} {cy:.1f} {x2:.1f} {y2:.1f}' "
        f"fill='none' stroke='{stroke}' stroke-width='{stroke_width:.2f}' opacity='{opacity:.2f}'{marker}{dash}/>"
    )


def render_station_causal_graph_svg(
    *,
    station_positions: pd.DataFrame,
    pairwise_edges: pd.DataFrame,
    binary_hyperedges: pd.DataFrame,
    horizon_label: str,
) -> str:
    width = 1120.0
    height = 800.0
    node_radius = 8.0
    positions = _scale_positions(station_positions, width=width, height=height)

    pair_strengths = [
        _edge_strength(row)
        for _, row in pairwise_edges.iterrows()
    ] if not pairwise_edges.empty else [0.0]
    hyper_strengths = [
        _edge_strength(row)
        for _, row in binary_hyperedges.iterrows()
    ] if not binary_hyperedges.empty else [0.0]
    max_pair_strength = max(max(pair_strengths), 1e-9)
    max_hyper_strength = max(max(hyper_strengths), 1e-9)

    pieces = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{int(width)}' height='{int(height)}' viewBox='0 0 {int(width)} {int(height)}'>",
        "<defs><marker id='station-arrow' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto' markerUnits='userSpaceOnUse'>"
        "<path d='M 0 0 L 8 3 L 0 6 z' fill='#345995'/></marker></defs>",
        f"<rect x='0' y='0' width='{int(width)}' height='{int(height)}' fill='white'/>",
        "<rect x='48' y='60' width='860' height='660' rx='18' ry='18' fill='#fbfcfe' stroke='#d7dee6' stroke-width='1.0'/>",
        f"<text x='72' y='100' font-size='18' font-weight='700' fill='#111'>Shanghai station-level causal graph ({html.escape(horizon_label)})</text>",
        "<text x='72' y='122' font-size='11' fill='#5b6572'>Directed pairwise edges and binary hyperedge junctions</text>",
    ]

    for index, (_, row) in enumerate(pairwise_edges.iterrows()):
        source_id = str(row["source_station_id"])
        target_id = str(row["target_station_id"])
        if source_id not in positions or target_id not in positions:
            continue
        strength = _edge_strength(row)
        x0, y0 = positions[source_id]
        x1, y1 = positions[target_id]
        dx = x1 - x0
        dy = y1 - y0
        distance = max(math.hypot(dx, dy), 1e-9)
        ux = dx / distance
        uy = dy / distance
        start = (x0 + node_radius * ux, y0 + node_radius * uy)
        end = (x1 - node_radius * ux, y1 - node_radius * uy)
        curvature = 0.10 * (1 if index % 2 == 0 else -1)
        stroke_width = 1.2 + 2.6 * strength / max_pair_strength
        pieces.append(
            _quadratic_path(
                start,
                end,
                curvature=curvature,
                stroke="#345995",
                stroke_width=stroke_width,
                opacity=0.82,
                marker_end="url(#station-arrow)",
            )
        )

    for _, row in binary_hyperedges.iterrows():
        source_left, source_right = _normalize_source_pair(row["source_station_ids"])
        target_id = str(row["target_station_id"])
        if source_left not in positions or source_right not in positions or target_id not in positions:
            continue
        strength = _edge_strength(row)
        left_x, left_y = positions[source_left]
        right_x, right_y = positions[source_right]
        target_x, target_y = positions[target_id]
        midpoint_x = 0.5 * (left_x + right_x)
        midpoint_y = 0.5 * (left_y + right_y)
        junction_x = 0.58 * midpoint_x + 0.42 * target_x
        junction_y = 0.58 * midpoint_y + 0.42 * target_y
        stroke_width = 1.3 + 2.3 * strength / max_hyper_strength
        color = "#2F7D63"

        for x0, y0 in ((left_x, left_y), (right_x, right_y)):
            dx = junction_x - x0
            dy = junction_y - y0
            distance = max(math.hypot(dx, dy), 1e-9)
            ux = dx / distance
            uy = dy / distance
            pieces.append(
                f"<line x1='{x0 + node_radius * ux:.1f}' y1='{y0 + node_radius * uy:.1f}' "
                f"x2='{junction_x - 6.0 * ux:.1f}' y2='{junction_y - 6.0 * uy:.1f}' "
                f"stroke='{color}' stroke-width='{stroke_width:.2f}' stroke-dasharray='5,3' opacity='0.92'/>"
            )

        dx = target_x - junction_x
        dy = target_y - junction_y
        distance = max(math.hypot(dx, dy), 1e-9)
        ux = dx / distance
        uy = dy / distance
        pieces.append(
            f"<line x1='{junction_x + 6.0 * ux:.1f}' y1='{junction_y + 6.0 * uy:.1f}' "
            f"x2='{target_x - node_radius * ux:.1f}' y2='{target_y - node_radius * uy:.1f}' "
            f"stroke='{color}' stroke-width='{stroke_width + 0.15:.2f}' stroke-dasharray='5,3' opacity='0.95' marker-end='url(#station-arrow)'/>"
        )
        pieces.append(
            f"<circle cx='{junction_x:.1f}' cy='{junction_y:.1f}' r='5.6' fill='white' stroke='{color}' stroke-width='1.0' opacity='0.95'/>"
        )

    for _, row in station_positions.iterrows():
        station_id = str(row["station_id"])
        x, y = positions[station_id]
        pieces.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='8' fill='#D8DEE9' stroke='#51606f' stroke-width='1.0'/>"
        )
        pieces.append(
            f"<text x='{x + 10.0:.1f}' y='{y + 3.0:.1f}' font-size='10' fill='#233142'>{html.escape(station_id)}</text>"
        )

    legend_x = 960.0
    legend_y = 130.0
    pieces.append(
        f"<rect x='{legend_x - 18.0:.1f}' y='{legend_y - 38.0:.1f}' width='160' height='108' rx='10' ry='10' fill='#fbfcfe' stroke='#d7dee6' stroke-width='1.0'/>"
    )
    pieces.append(f"<text x='{legend_x:.1f}' y='{legend_y:.1f}' font-size='11.5' font-weight='700' fill='#333'>Legend</text>")
    pieces.append(
        f"<line x1='{legend_x:.1f}' y1='{legend_y + 20.0:.1f}' x2='{legend_x + 28.0:.1f}' y2='{legend_y + 20.0:.1f}' "
        "stroke='#345995' stroke-width='2.4' marker-end='url(#station-arrow)'/>"
    )
    pieces.append(
        f"<text x='{legend_x + 38.0:.1f}' y='{legend_y + 24.0:.1f}' font-size='10.5' fill='#333'>Pairwise edge</text>"
    )
    pieces.append(
        f"<line x1='{legend_x:.1f}' y1='{legend_y + 46.0:.1f}' x2='{legend_x + 28.0:.1f}' y2='{legend_y + 46.0:.1f}' "
        "stroke='#2F7D63' stroke-width='2.2' stroke-dasharray='5,3' marker-end='url(#station-arrow)'/>"
    )
    pieces.append(
        f"<circle cx='{legend_x + 14.0:.1f}' cy='{legend_y + 70.0:.1f}' r='5.2' fill='white' stroke='#2F7D63' stroke-width='1.0'/>"
    )
    pieces.append(
        f"<text x='{legend_x + 38.0:.1f}' y='{legend_y + 50.0:.1f}' font-size='10.5' fill='#333'>Binary hyperedge</text>"
    )
    pieces.append(
        f"<text x='{legend_x + 38.0:.1f}' y='{legend_y + 74.0:.1f}' font-size='10.5' fill='#333'>junction</text>"
    )
    pieces.append("</svg>")
    return "".join(pieces)


# --- Former yrd/shanghai_notebook.py ---

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D


PAIRWISE_EDGE_COLOR = "#345995"
HYPEREDGE_COLOR = "#2F7D63"
SYNERGY_POSITIVE_EDGE_COLOR = "#2F7D63"
SYNERGY_NEGATIVE_EDGE_COLOR = "#B04A5A"
SYNERGY_RATIO_EDGE_COLOR = "#C47C2E"


def configure_notebook_runtime() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def find_project_root(start: Path | None = None) -> Path:
    candidates = [start.resolve() if start is not None else Path.cwd().resolve()]
    candidates.extend(candidates[0].parents)
    fallback_candidate: Path | None = None
    for candidate in candidates:
        if not (candidate / "yrd").is_dir() or not (candidate / "data").is_dir():
            continue
        if fallback_candidate is None:
            fallback_candidate = candidate
        if (
            ((candidate / "data" / "dataset_yrd.nc").exists() and (candidate / "data" / "stations_yrd.csv").exists())
            or (
                (candidate / "data" / "dataset_bthsa.nc").exists()
                and (candidate / "data" / "stations_bthsa.csv").exists()
            )
        ):
            return candidate
    if fallback_candidate is not None:
        return fallback_candidate
    raise RuntimeError("Could not locate project root containing yrd/ and data/")


def build_default_shanghai_one_step_config(root: Path, *, test_mode: bool) -> YRDExperimentConfig:
    return replace(
        YRDExperimentConfig(root_dir=root),
        sample_mode="one_step",
        history_hours=1,
        horizons=(1,),
        model_name="resmlp",
        hidden_dim=16 if test_mode else 96,
        num_layers=2 if test_mode else 3,
        dropout=0.0 if test_mode else 0.05,
        norm_type="layernorm",
        activation="silu",
        learning_rate=5e-4,
        weight_decay=0.0 if test_mode else 1e-5,
        batch_size=8 if test_mode else 64,
        epochs=2 if test_mode else 12,
        max_epochs=4 if test_mode else 60,
        early_stopping_patience=2 if test_mode else 8,
        seed=0,
    )


def build_run_tag(*, test_mode: bool) -> str:
    return "shanghai_one_step_o3_station_graph_smoke" if test_mode else "shanghai_one_step_o3_station_graph"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False))


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def inverse_transform_targets(
    array: np.ndarray,
    target_names: list[str],
    stats: dict[str, dict[str, float]],
) -> np.ndarray:
    restored = np.asarray(array, dtype=np.float32).copy()
    for index, name in enumerate(target_names):
        variable = name.split("__")[-1]
        restored[:, index] = restored[:, index] * stats[variable]["std"] + stats[variable]["mean"]
    return restored


def metric_rows_for_scope(
    y_true_by_horizon: dict[int, np.ndarray],
    prediction_sets: dict[str, dict[int, np.ndarray]],
    *,
    target_names: list[str],
) -> list[dict[str, object]]:
    scope_indices = {
        "overall": list(range(len(target_names))),
        "O3": [index for index, name in enumerate(target_names) if name.endswith("__O3")],
        "PM2.5": [index for index, name in enumerate(target_names) if name.endswith("__PM2.5")],
    }
    rows: list[dict[str, object]] = []
    for model_name, predictions_by_horizon in prediction_sets.items():
        for horizon, y_true in y_true_by_horizon.items():
            for scope_name, indices in scope_indices.items():
                summary = _compute_metrics(y_true[:, indices], predictions_by_horizon[horizon][:, indices])
                rows.append(
                    {
                        "model": model_name,
                        "horizon": f"{horizon}h",
                        "scope": scope_name,
                        "rmse": summary["rmse"],
                        "mae": summary["mae"],
                        "corr": summary["corr"],
                    }
                )
    return rows


def station_metric_rows(
    y_true_by_horizon: dict[int, np.ndarray],
    prediction_sets: dict[str, dict[int, np.ndarray]],
    *,
    station_ids: list[str],
    target_width: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name, predictions_by_horizon in prediction_sets.items():
        for horizon, y_true in y_true_by_horizon.items():
            for station_index, station_id in enumerate(station_ids):
                start = station_index * target_width
                indices = list(range(start, start + target_width))
                summary = _compute_metrics(y_true[:, indices], predictions_by_horizon[horizon][:, indices])
                rows.append(
                    {
                        "model": model_name,
                        "horizon": f"{horizon}h",
                        "station_id": station_id,
                        "rmse": summary["rmse"],
                        "mae": summary["mae"],
                        "corr": summary["corr"],
                    }
                )
    return rows


def build_station_variable_index_map(target_names: list[str], variable: str) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    suffix = f"__{variable}"
    for index, name in enumerate(target_names):
        if not name.endswith(suffix):
            continue
        _, station_id, _ = name.split("__")
        mapping.setdefault(station_id, []).append(index)
    return mapping


def compute_training_input_center(x_train: np.ndarray) -> np.ndarray:
    return _compute_training_input_center(x_train)


def sample_uniform_box_inputs(
    *,
    center: np.ndarray,
    box_size: float | np.ndarray | list[float],
    sample_count: int,
    seed: int,
    lower_bounds: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    return _sample_uniform_box_inputs(
        center=center,
        box_size=box_size,
        sample_count=sample_count,
        seed=seed,
        lower_bounds=lower_bounds,
    )


def causal_graph_cache_is_valid(
    *,
    cache_meta: dict[str, object],
    requested_box_size: float | None = None,
    requested_box_size_by_variable: dict[str, float] | None = None,
    nonnegative_variables: tuple[str, ...] = (),
    coupling_sample_count: int,
    sampling_mode: str,
    sampling_seed: int,
) -> bool:
    cached_sample_count = int(cache_meta.get("coupling_sample_count", -1))
    cached_sampling_mode = str(cache_meta.get("sampling_mode", ""))
    cached_sampling_seed = int(cache_meta.get("sampling_seed", -1))
    if (
        cached_sample_count != coupling_sample_count
        or cached_sampling_mode != sampling_mode
        or cached_sampling_seed != sampling_seed
    ):
        return False
    if requested_box_size_by_variable is not None:
        cached_requested_box_sizes = cache_meta.get("requested_causal_graph_box_size_by_variable")
        cached_box_sizes = cache_meta.get("causal_graph_box_size_by_variable")
        cached_nonnegative_variables = tuple(cache_meta.get("causal_graph_nonnegative_variables", ()))
        return bool(
            isinstance(cached_requested_box_sizes, dict)
            and isinstance(cached_box_sizes, dict)
            and cached_requested_box_sizes == requested_box_size_by_variable
            and cached_box_sizes == requested_box_size_by_variable
            and cached_nonnegative_variables == tuple(nonnegative_variables)
        )

    cached_requested_box_size = float(cache_meta.get("requested_causal_graph_box_size", -1.0))
    cached_box_size = float(cache_meta.get("causal_graph_box_size", -1.0))
    return bool(cached_box_size > 0.0 and np.isclose(cached_requested_box_size, float(requested_box_size)))


def resolve_variable_box_size_by_feature(
    *,
    input_variables: tuple[str, ...],
    box_size_by_variable: dict[str, float],
    n_stations: int,
) -> np.ndarray:
    return _resolve_variable_box_size_by_feature(
        input_variables=input_variables,
        box_size_by_variable=box_size_by_variable,
        n_stations=n_stations,
    )


def resolve_nonnegative_lower_bounds_by_feature(
    *,
    input_variables: tuple[str, ...],
    stats: dict[str, dict[str, float]],
    nonnegative_variables: tuple[str, ...],
    n_stations: int,
) -> np.ndarray | None:
    return _resolve_nonnegative_lower_bounds_by_feature(
        input_variables=input_variables,
        stats=stats,
        nonnegative_variables=nonnegative_variables,
        n_stations=n_stations,
    )


def resolve_causal_graph_box_label(
    *,
    box_size: float | None = None,
    box_size_by_variable: dict[str, float] | None = None,
) -> str | float | None:
    return _resolve_causal_graph_box_label(
        box_size=box_size,
        box_size_by_variable=box_size_by_variable,
    )


def select_display_rows(
    frame: pd.DataFrame,
    *,
    per_target: int,
    source_col: str,
    target_col: str,
    ranking_col: str = "mean",
    positive_only: bool = False,
    include_self_loops: bool = True,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    filtered = frame.copy()
    if not include_self_loops:
        filtered = filtered[filtered[source_col] != filtered[target_col]].copy()
    if positive_only:
        filtered = filtered[filtered["mean"] > 0].copy()
    if filtered.empty:
        return filtered.reset_index(drop=True)
    filtered = filtered.sort_values([target_col, ranking_col], ascending=[True, False])
    return filtered.groupby(target_col, group_keys=False).head(per_target).reset_index(drop=True)


def select_display_hyperedge_rows(frame: pd.DataFrame, *, per_target: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    display_frame = frame.copy()
    display_frame["source_pair_label"] = display_frame["source_station_ids"].apply(
        lambda value: " + ".join(map(str, value))
    )
    display_frame["_abs_strength"] = display_frame["mean"].astype(float).abs()
    display_frame = display_frame.sort_values(["target_station_id", "_abs_strength"], ascending=[True, False])
    display_frame = display_frame.drop(columns="_abs_strength")
    return display_frame.groupby("target_station_id", group_keys=False).head(per_target).reset_index(drop=True)


def build_global_hyperedge_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ranked = frame.copy()
    ranked["source_pair_label"] = ranked["source_station_ids"].apply(lambda value: " + ".join(map(str, value)))
    ranked["hyperedge_label"] = ranked.apply(
        lambda row: f"{row['source_pair_label']} -> {row['target_station_id']}",
        axis=1,
    )
    ranked["_abs_strength"] = ranked["mean"].astype(float).abs()
    ranked = ranked.sort_values("_abs_strength", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    total_abs_mean = float(ranked["_abs_strength"].sum())
    ranked["cumulative_abs_mean"] = ranked["_abs_strength"].cumsum()
    ranked["cumulative_share"] = (
        ranked["cumulative_abs_mean"] / total_abs_mean if total_abs_mean > 0 else 0.0
    )
    ranked = ranked.drop(columns="_abs_strength")
    return ranked


def build_global_edge_ranking(frame: pd.DataFrame, *, sort_col: str = "mean") -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ranked = frame.copy()
    ranked["edge_label"] = ranked.apply(
        lambda row: f"{row['source_station_id']} -> {row['target_station_id']}",
        axis=1,
    )
    ranked["_abs_strength"] = ranked["mean"].astype(float).abs()
    if sort_col == "abs_mean":
        ranked = ranked.sort_values("_abs_strength", ascending=False).reset_index(drop=True)
    else:
        ranked = ranked.sort_values(sort_col, ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    total_abs_mean = float(ranked["_abs_strength"].sum())
    ranked["cumulative_abs_mean"] = ranked["_abs_strength"].cumsum()
    ranked["cumulative_share"] = (
        ranked["cumulative_abs_mean"] / total_abs_mean if total_abs_mean > 0 else 0.0
    )
    ranked = ranked.drop(columns="_abs_strength")
    return ranked


def summarize_hyperedge_cutoff(frame: pd.DataFrame, *, top_k_values: tuple[int, ...] = (5, 10, 15, 20)) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    if frame.empty:
        return pd.DataFrame(columns=["top_k", "edge_count", "cumulative_share", "cutoff_abs_mean"])
    for top_k in top_k_values:
        selected = frame.head(top_k)
        if selected.empty:
            continue
        rows.append(
            {
                "top_k": int(top_k),
                "edge_count": int(len(selected)),
                "cumulative_share": float(selected["cumulative_share"].iloc[-1]),
                "cutoff_abs_mean": float(abs(float(selected["mean"].iloc[-1]))),
            }
        )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _format_graph_title(base_title: str, *, box_size: float | str | None = None) -> str:
    if box_size is None:
        return base_title
    if isinstance(box_size, str):
        return f"{base_title}, {box_size}"
    return f"{base_title}, L = {float(box_size):.3f}"


def build_self_loop_node_strengths(
    frame: pd.DataFrame,
    *,
    station_ids: list[str],
    source_col: str = "source_station_id",
    target_col: str = "target_station_id",
    strength_col: str = "mean",
) -> dict[str, float]:
    strengths = {str(station_id): 0.0 for station_id in station_ids}
    if frame.empty:
        return strengths
    diagonal_rows = frame[frame[source_col] == frame[target_col]].copy()
    if diagonal_rows.empty:
        return strengths
    for _, row in diagonal_rows.iterrows():
        station_id = str(row[source_col])
        if station_id in strengths:
            strengths[station_id] = float(row[strength_col])
    return strengths


def draw_station_causal_graph(
    *,
    station_positions: pd.DataFrame,
    pairwise_edges: pd.DataFrame,
    horizon_label: str,
    out_path: Path,
    title: str | None = None,
    positive_color: str = PAIRWISE_EDGE_COLOR,
    negative_color: str | None = None,
    legend_label: str | None = None,
    strength_col: str = "abs_mean",
    box_size: float | None = None,
    alpha_min: float = 0.08,
    alpha_max: float = 0.82,
    arrow_mutation_scale: float = 18.0,
    arrow_shrink_target: float = 8.0,
    node_self_strengths: dict[str, float] | None = None,
    node_colorbar_label: str | None = None,
    show_title: bool = True,
    show_edge_legend: bool = True,
) -> None:
    position_map = {
        row["station_id"]: (float(row["lon"]), float(row["lat"]))
        for _, row in station_positions.iterrows()
    }
    fig, ax = plt.subplots(figsize=(10.8, 7.6), constrained_layout=True)
    node_colors = "#D8DEE9"
    node_norm: mcolors.Normalize | None = None
    if node_self_strengths is not None:
        node_values = np.asarray(
            [float(node_self_strengths.get(str(station_id), 0.0)) for station_id in station_positions["station_id"]],
            dtype=float,
        )
        max_abs_value = float(np.max(np.abs(node_values))) if len(node_values) else 0.0
        if max_abs_value > 0:
            node_norm = mcolors.TwoSlopeNorm(vmin=-max_abs_value, vcenter=0.0, vmax=max_abs_value)
            node_colors = matplotlib.colormaps["RdBu_r"](node_norm(node_values))
    ax.scatter(
        station_positions["lon"],
        station_positions["lat"],
        color=node_colors,
        s=110,
        edgecolors="#4C566A",
        linewidths=0.7,
        zorder=5,
    )
    for _, row in station_positions.iterrows():
        ax.text(
            row["lon"] + 0.004,
            row["lat"] + 0.002,
            row["station_id"],
            fontsize=7.4,
            color="#233142",
            zorder=6,
        )

    render_edges = pairwise_edges.copy()
    if strength_col in render_edges.columns:
        render_edges = render_edges[render_edges[strength_col] > 0].copy()

    if not render_edges.empty:
        max_edge = max(float(render_edges[strength_col].max()), 1e-6)
        for _, row in render_edges.iterrows():
            x0, y0 = position_map[row["source_station_id"]]
            x1, y1 = position_map[row["target_station_id"]]
            strength = float(row[strength_col])
            normalized_strength = max(0.0, min(1.0, strength / max_edge))
            width, alpha = style_pairwise_edge(normalized_strength, alpha_min=alpha_min, alpha_max=alpha_max)
            color = positive_color if negative_color is None or float(row["mean"]) >= 0 else negative_color
            rad = 0.12 if x0 <= x1 else -0.12
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "linewidth": width,
                    "alpha": alpha,
                    "connectionstyle": f"arc3,rad={rad}",
                    "shrinkA": 10,
                    "shrinkB": arrow_shrink_target,
                    "mutation_scale": arrow_mutation_scale,
                },
                zorder=2,
            )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.18, linewidth=0.6)
    if show_title:
        base_title = title or f"Shanghai O3 station-level causal graph ({horizon_label})"
        ax.set_title(_format_graph_title(base_title, box_size=box_size), fontsize=13)
    has_negative_display = negative_color is not None and not render_edges.empty and bool((render_edges["mean"] < 0).any())
    if negative_color is None or not has_negative_display:
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=positive_color,
                linewidth=2.4,
                label=legend_label or "O3 pairwise edge",
            )
        ]
    else:
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=positive_color,
                linewidth=2.4,
                label=f"{legend_label or 'Synergy edge'} (+)",
            ),
            Line2D(
                [0],
                [0],
                color=negative_color,
                linewidth=2.4,
                label=f"{legend_label or 'Synergy edge'} (-)",
            ),
        ]
    if show_edge_legend:
        ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    if node_norm is not None:
        colorbar = fig.colorbar(
            cm.ScalarMappable(norm=node_norm, cmap="RdBu_r"),
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        colorbar.set_label(node_colorbar_label or "Self-loop strength")
    _save_figure(fig, out_path)


def build_pairwise_edge_render_frame(
    pairwise_edges: pd.DataFrame,
    *,
    strength_col: str = "mean",
    alpha_min: float = 0.08,
    alpha_max: float = 0.82,
) -> pd.DataFrame:
    render_edges = pairwise_edges.copy()
    if strength_col in render_edges.columns:
        render_edges = render_edges[render_edges[strength_col] > 0].copy()
    if render_edges.empty:
        return render_edges

    max_edge = max(float(render_edges[strength_col].max()), 1e-6)
    render_edges["normalized_strength"] = render_edges[strength_col].astype(float) / max_edge
    render_edges["normalized_strength"] = render_edges["normalized_strength"].clip(lower=0.0, upper=1.0)
    styles = render_edges["normalized_strength"].apply(
        lambda value: style_pairwise_edge(value, alpha_min=alpha_min, alpha_max=alpha_max)
    )
    render_edges["linewidth"] = styles.str[0].astype(float)
    render_edges["alpha"] = styles.str[1].astype(float)
    return render_edges.reset_index(drop=True)


def select_top_fraction_edges(
    pairwise_edges: pd.DataFrame,
    *,
    strength_col: str = "mean",
    keep_fraction: float = 0.5,
) -> pd.DataFrame:
    if pairwise_edges.empty:
        return pairwise_edges.copy()

    keep_count = max(1, int(np.ceil(len(pairwise_edges) * float(keep_fraction))))
    return (
        pairwise_edges.sort_values(strength_col, ascending=False)
        .head(keep_count)
        .reset_index(drop=True)
    )


def style_pairwise_edge(
    normalized_strength: float,
    *,
    alpha_min: float = 0.08,
    alpha_max: float = 0.82,
) -> tuple[float, float]:
    contrast_strength = max(0.0, min(1.0, float(normalized_strength))) ** 1.9
    linewidth = 0.18 + 5.4 * contrast_strength
    alpha = alpha_min + (alpha_max - alpha_min) * contrast_strength
    return linewidth, alpha


def build_pairwise_degree_summary(
    pairwise_edges: pd.DataFrame,
    *,
    station_ids: list[str],
    strength_col: str = "mean",
) -> pd.DataFrame:
    edge_frame = build_pairwise_edge_render_frame(pairwise_edges, strength_col=strength_col)
    if edge_frame.empty:
        return pd.DataFrame(
            {
                "station_id": station_ids,
                "in_degree": np.zeros(len(station_ids), dtype=int),
                "out_degree": np.zeros(len(station_ids), dtype=int),
                "in_strength": np.zeros(len(station_ids), dtype=float),
                "out_strength": np.zeros(len(station_ids), dtype=float),
            }
        )

    in_degree = edge_frame.groupby("target_station_id").size().reindex(station_ids, fill_value=0)
    out_degree = edge_frame.groupby("source_station_id").size().reindex(station_ids, fill_value=0)
    in_strength = (
        edge_frame.groupby("target_station_id")[strength_col].sum().reindex(station_ids, fill_value=0.0)
    )
    out_strength = (
        edge_frame.groupby("source_station_id")[strength_col].sum().reindex(station_ids, fill_value=0.0)
    )
    return pd.DataFrame(
        {
            "station_id": station_ids,
            "in_degree": in_degree.astype(int).to_numpy(),
            "out_degree": out_degree.astype(int).to_numpy(),
            "in_strength": in_strength.astype(float).to_numpy(),
            "out_strength": out_strength.astype(float).to_numpy(),
        }
    )


def draw_pairwise_ei_degree_diagnostics(
    *,
    render_edges: pd.DataFrame,
    node_degree_summary: pd.DataFrame,
    out_path: Path,
    horizon_label: str,
    box_size: float | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), constrained_layout=True)
    bins = min(18, max(8, int(np.sqrt(max(len(render_edges), 1)))))

    axes[0, 0].hist(
        render_edges["mean"] if not render_edges.empty else [0.0],
        bins=bins,
        color=PAIRWISE_EDGE_COLOR,
        alpha=0.86,
        edgecolor="white",
    )
    axes[0, 0].set_title("All positive pairwise EI distribution", fontsize=12)
    axes[0, 0].set_xlabel("Mean pairwise EI (NIS)")
    axes[0, 0].set_ylabel("Edge count")
    axes[0, 0].grid(True, axis="y", alpha=0.18, linewidth=0.6)

    for ax, column, title in (
        (axes[0, 1], "in_degree", "In-degree distribution"),
        (axes[1, 0], "out_degree", "Out-degree distribution"),
    ):
        values = node_degree_summary[column].to_numpy(dtype=int)
        if values.size == 0:
            values = np.array([0], dtype=int)
        bins_edges = np.arange(values.min() - 0.5, values.max() + 1.5, 1.0)
        ax.hist(values, bins=bins_edges, color=PAIRWISE_EDGE_COLOR, alpha=0.86, edgecolor="white")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Node count")
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.6)

    axes[1, 1].hist(
        [node_degree_summary["in_strength"].to_numpy(dtype=float), node_degree_summary["out_strength"].to_numpy(dtype=float)],
        bins=min(14, max(6, len(node_degree_summary) // 2)),
        color=["#4C78A8", "#F28E2B"],
        alpha=0.72,
        edgecolor="white",
        label=["Weighted in-strength", "Weighted out-strength"],
    )
    axes[1, 1].set_title("Weighted degree-strength distribution", fontsize=12)
    axes[1, 1].set_xlabel("Sum of mean pairwise EI")
    axes[1, 1].set_ylabel("Node count")
    axes[1, 1].grid(True, axis="y", alpha=0.18, linewidth=0.6)
    axes[1, 1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.suptitle(
        _format_graph_title(f"Shanghai O3 pairwise-edge diagnostics ({horizon_label})", box_size=box_size),
        fontsize=13,
    )
    _save_figure(fig, out_path)


def build_causal_summary_records(
    *,
    raw_records: list[dict[str, object]],
    station_source_groups: dict[str, list[int]],
    station_pollutant_feature_groups: dict[str, dict[str, list[int]]],
    box_size: float,
    causal_graph_variable: str,
) -> list[dict[str, object]]:
    sample_records: list[dict[str, object]] = []
    for raw_record in raw_records:
        jacobian = np.asarray(raw_record["jacobian"], dtype=float)
        target_indices = list(raw_record["target_indices"])
        sigma_eps = np.asarray(raw_record["sigma_eps"], dtype=float)
        summary = compute_station_level_nis_summary(
            jacobian=jacobian,
            sigma_eps=sigma_eps,
            station_source_groups=station_source_groups,
            target_indices=target_indices,
            box_size=box_size,
        )
        pollutant_pair_summary = compute_station_pollutant_pair_synergy_summary(
            jacobian=jacobian,
            sigma_eps=sigma_eps,
            station_pollutant_feature_groups=station_pollutant_feature_groups,
            target_indices=target_indices,
            box_size=box_size,
        )
        sample_records.append(
            {
                "sample_id": int(raw_record["sample_id"]),
                "sampling_mode": str(raw_record["sampling_mode"]),
                "horizon": "1h",
                "target_station_id": raw_record["target_station_id"],
                "target_variable": causal_graph_variable,
                "ei_nis": float(summary["ei_nis"]),
                "pairwise_station_ei_nis": summary["pairwise_station_ei_nis"],
                "binary_station_synergy_nis": summary["binary_station_synergy_nis"],
                "joint_station_pair_ei_nis": pollutant_pair_summary["joint_station_pair_ei_nis"],
                "single_pollutant_ei_nis": pollutant_pair_summary["single_pollutant_ei_nis"],
                "station_pair_synergy_nis": pollutant_pair_summary["station_pair_synergy_nis"],
            }
        )
    return sample_records


def build_transport_map_global_causal_summary(
    *,
    source_samples: np.ndarray | list[object],
    predicted_next_o3: np.ndarray | list[object],
    station_ids: list[str],
    o3_feature_index: int,
    pm25_feature_index: int,
) -> dict[str, object]:
    source_array = np.asarray(source_samples, dtype=float)
    target_array = np.asarray(predicted_next_o3, dtype=float)
    if source_array.ndim != 3:
        raise ValueError("source_samples must have shape [samples, stations, features].")
    if target_array.ndim != 2:
        raise ValueError("predicted_next_o3 must have shape [samples, stations].")
    if source_array.shape[0] != target_array.shape[0] or source_array.shape[1] != len(station_ids):
        raise ValueError("source_samples and predicted_next_o3 must align with station_ids.")

    current_o3 = source_array[:, :, int(o3_feature_index)]
    current_o3_pm25 = np.empty((source_array.shape[0], len(station_ids) * 2), dtype=float)
    station_o3_groups: dict[str, list[int]] = {}
    station_pollutant_feature_groups: dict[str, dict[str, list[int]]] = {}
    for station_index, station_id in enumerate(station_ids):
        current_o3_pm25[:, 2 * station_index] = source_array[:, station_index, int(o3_feature_index)]
        current_o3_pm25[:, 2 * station_index + 1] = source_array[:, station_index, int(pm25_feature_index)]
        station_o3_groups[str(station_id)] = [station_index]
        station_pollutant_feature_groups[str(station_id)] = {
            "O3": [2 * station_index],
            "PM2.5": [2 * station_index + 1],
        }

    summary_records: list[dict[str, object]] = []
    for target_index, target_station_id in enumerate(station_ids):
        target_matrix = target_array[:, [target_index]]
        pairwise_summary = compute_station_level_ei_summary(
            method="tm",
            source_samples=current_o3,
            target_samples=target_matrix,
            station_source_groups=station_o3_groups,
        )
        pollutant_summary = compute_station_pollutant_pair_synergy_summary(
            method="tm",
            source_samples=current_o3_pm25,
            target_samples=target_matrix,
            station_pollutant_feature_groups=station_pollutant_feature_groups,
        )
        summary_records.append(
            {
                "target_station_id": str(target_station_id),
                "pairwise_station_ei": pairwise_summary["pairwise_station_ei"],
                "binary_station_synergy": pairwise_summary["binary_station_synergy"],
                "joint_station_pair_ei": pollutant_summary["joint_station_pair_ei"],
                "single_pollutant_ei": pollutant_summary["single_pollutant_ei"],
                "station_pair_synergy": pollutant_summary["station_pair_synergy"],
            }
        )

    station_coupling_summary = summarize_global_station_coupling(summary_records, station_ids=station_ids)
    pollutant_synergy_summary = summarize_global_station_pollutant_synergy(summary_records, station_ids=station_ids)
    return {
        "coupling_method": "tm",
        "1h": {
            **station_coupling_summary,
            "conditional_synergy_edges": pollutant_synergy_summary["conditional_synergy_edges"],
            "conditional_synergy_ratio_edges": pollutant_synergy_summary["conditional_synergy_ratio_edges"],
            "conditional_synergy_per_target_station": pollutant_synergy_summary["per_target_station"],
        },
    }


def resolve_causal_graph_box_size(
    *,
    raw_records: list[dict[str, object]],
    station_ids: list[str],
    station_source_groups: dict[str, list[int]],
    station_pollutant_feature_groups: dict[str, dict[str, list[int]]],
    requested_box_size: float,
    graph_edges_per_target: int,
    causal_graph_variable: str,
) -> tuple[float, list[dict[str, object]], dict[str, object]]:
    candidate_box_sizes = [float(requested_box_size) * (2.0**power) for power in range(10)]
    fallback_payload: tuple[float, list[dict[str, object]], dict[str, object]] | None = None

    for box_size in candidate_box_sizes:
        sample_records = build_causal_summary_records(
            raw_records=raw_records,
            station_source_groups=station_source_groups,
            station_pollutant_feature_groups=station_pollutant_feature_groups,
            box_size=box_size,
            causal_graph_variable=causal_graph_variable,
        )
        station_coupling_summary = summarize_global_station_coupling(sample_records, station_ids=station_ids)
        pollutant_synergy_summary = summarize_global_station_pollutant_synergy(sample_records, station_ids=station_ids)
        global_summary_by_horizon = {
            "1h": {
                **station_coupling_summary,
                "conditional_synergy_edges": pollutant_synergy_summary["conditional_synergy_edges"],
                "conditional_synergy_ratio_edges": pollutant_synergy_summary["conditional_synergy_ratio_edges"],
                "conditional_synergy_per_target_station": pollutant_synergy_summary["per_target_station"],
            }
        }
        pairwise_edges_df = pd.DataFrame(global_summary_by_horizon["1h"]["pairwise_edges"])
        non_self_pairwise = (
            pairwise_edges_df[pairwise_edges_df["source_station_id"] != pairwise_edges_df["target_station_id"]].copy()
            if not pairwise_edges_df.empty
            else pd.DataFrame()
        )
        payload = (box_size, sample_records, global_summary_by_horizon)
        if fallback_payload is None:
            fallback_payload = payload
        if (
            not non_self_pairwise.empty
            and len(non_self_pairwise) == len(station_ids) * max(len(station_ids) - 1, 0)
            and bool((non_self_pairwise["mean"] > 0).all())
        ):
            return payload

    assert fallback_payload is not None
    return fallback_payload


def draw_station_hyperedge_graph(
    *,
    station_positions: pd.DataFrame,
    binary_hyperedges: pd.DataFrame,
    horizon_label: str,
    out_path: Path,
    box_size: float | None = None,
    arrow_mutation_scale: float = 18.0,
    arrow_shrink_target: float = 8.0,
) -> None:
    position_map = {
        row["station_id"]: (float(row["lon"]), float(row["lat"]))
        for _, row in station_positions.iterrows()
    }
    fig, ax = plt.subplots(figsize=(12.4, 7.6), constrained_layout=True)
    ax.scatter(station_positions["lon"], station_positions["lat"], color="#D8DEE9", s=92, zorder=5)
    for _, row in station_positions.iterrows():
        ax.text(
            row["lon"] + 0.004,
            row["lat"] + 0.002,
            row["station_id"],
            fontsize=7.4,
            color="#233142",
            zorder=6,
        )

    if not binary_hyperedges.empty:
        max_edge = max(float(binary_hyperedges["mean"].abs().max()), 1e-6)
        for _, row in binary_hyperedges.iterrows():
            source_left, source_right = tuple(row["source_station_ids"])
            target_id = row["target_station_id"]
            left_x, left_y = position_map[source_left]
            right_x, right_y = position_map[source_right]
            target_x, target_y = position_map[target_id]
            midpoint_x = 0.5 * (left_x + right_x)
            midpoint_y = 0.5 * (left_y + right_y)
            junction_x = 0.58 * midpoint_x + 0.42 * target_x
            junction_y = 0.58 * midpoint_y + 0.42 * target_y
            width = 1.0 + 3.0 * abs(float(row["mean"])) / max_edge

            for source_x, source_y in ((left_x, left_y), (right_x, right_y)):
                ax.plot(
                    [source_x, junction_x],
                    [source_y, junction_y],
                    color=HYPEREDGE_COLOR,
                    linewidth=width,
                    linestyle=(0, (5, 3)),
                    alpha=0.8,
                    zorder=2,
                )
            ax.scatter(junction_x, junction_y, s=34, facecolors="white", edgecolors=HYPEREDGE_COLOR, zorder=4)
            ax.annotate(
                "",
                xy=(target_x, target_y),
                xytext=(junction_x, junction_y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": HYPEREDGE_COLOR,
                    "linewidth": width,
                    "alpha": 0.88,
                    "linestyle": (0, (5, 3)),
                    "shrinkA": 8,
                    "shrinkB": arrow_shrink_target,
                    "mutation_scale": arrow_mutation_scale,
                },
                zorder=3,
            )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.set_title(
        _format_graph_title(f"Shanghai O3 station-level binary hyperedges ({horizon_label})", box_size=box_size),
        fontsize=13,
    )
    legend_handles = [
        Line2D([0], [0], color=HYPEREDGE_COLOR, linewidth=2.4, linestyle=(0, (5, 3)), label="O3 binary hyperedge"),
        Line2D([0], [0], marker="o", color=HYPEREDGE_COLOR, markerfacecolor="white", linewidth=0, label="Hyperedge junction"),
    ]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_figure(fig, out_path)


def draw_global_hyperedge_topk_plot(
    *,
    ranked_hyperedges: pd.DataFrame,
    out_path: Path,
    top_k: int = 20,
    box_size: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12.6, 5.8), constrained_layout=True)
    plot_frame = ranked_hyperedges.head(top_k).copy()
    if plot_frame.empty:
        ax.text(0.5, 0.5, "No binary hyperedges available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        _save_figure(fig, out_path)
        return
    colors = [HYPEREDGE_COLOR if float(value) >= 0 else "#B04A5A" for value in plot_frame["mean"]]
    ax.bar(range(len(plot_frame)), plot_frame["mean"].abs(), color=colors, alpha=0.86)
    ax.set_xticks(range(len(plot_frame)))
    ax.set_xticklabels(plot_frame["hyperedge_label"], rotation=60, ha="right", fontsize=8.5)
    ax.set_xlabel("Global hyperedge rank")
    ax.set_ylabel("Absolute mean synergy")
    ax.set_title(
        _format_graph_title(f"Global top {len(plot_frame)} station-level binary hyperedges (1h)", box_size=box_size),
        fontsize=13,
    )
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.6)
    legend_handles = [
        Line2D([0], [0], color=HYPEREDGE_COLOR, linewidth=6, label="Positive synergy"),
        Line2D([0], [0], color="#B04A5A", linewidth=6, label="Negative synergy"),
    ]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_figure(fig, out_path)


def draw_global_hyperedge_cumulative_plot(
    *,
    ranked_hyperedges: pd.DataFrame,
    out_path: Path,
    box_size: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    if ranked_hyperedges.empty:
        ax.text(0.5, 0.5, "No binary hyperedges available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        _save_figure(fig, out_path)
        return
    ax.plot(
        ranked_hyperedges["rank"],
        ranked_hyperedges["cumulative_share"],
        color=HYPEREDGE_COLOR,
        linewidth=2.4,
        marker="o",
        markersize=4.6,
    )
    for share in (0.5, 0.75, 0.9):
        ax.axhline(share, color="#8793A0", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xlabel("Top-k hyperedges")
    ax.set_ylabel("Cumulative absolute-strength share")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(
        _format_graph_title("Cumulative coverage of global binary hyperedge strength (1h)", box_size=box_size),
        fontsize=13,
    )
    ax.grid(True, alpha=0.18, linewidth=0.6)
    _save_figure(fig, out_path)


def prepare_shanghai_one_step_bundle(
    *,
    cfg: YRDExperimentConfig,
    city_en: str,
    run_tag: str,
    use_smoke: bool,
    coupling_sample_count: int,
    graph_edges_per_target: int,
    causal_graph_edge_keep_fraction: float,
    causal_graph_arrow_mutation_scale: float,
    causal_graph_arrow_shrink_target: float,
    causal_graph_variable: str = "O3",
) -> dict[str, object]:
    if use_smoke:
        smoke_root = Path(tempfile.gettempdir()) / "eisyn_yrd_smoke"
        cache_dir = smoke_root / "exp" / "cache" / "yrd_coupling" / run_tag
        results_dir = smoke_root / "fig" / "yrd_shanghai" / "artifacts" / run_tag
    else:
        cache_dir = cfg.cache_dir / run_tag
        results_dir = cfg.results_dir / run_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        "config": cache_dir / "config.json",
        "checkpoint": cache_dir / "joint_model_checkpoint.pt",
        "loss_history": cache_dir / "loss_history.json",
        "predictions": cache_dir / "test_predictions.npz",
        "metrics": cache_dir / "metrics_summary.json",
        "global_coupling_samples": cache_dir / "o3_station_coupling_samples.json",
        "global_coupling_summary": cache_dir / "o3_station_causal_graph_summary.json",
        "run_manifest": results_dir / "run_manifest.json",
        "global_graph_1h": results_dir / "o3_station_causal_graph_1h.png",
        "global_graph_edge_weights_1h": results_dir / "o3_station_causal_graph_edge_weights_1h.csv",
        "global_graph_degree_summary_1h": results_dir / "o3_station_causal_graph_degree_summary_1h.csv",
        "global_graph_diagnostics_1h": results_dir / "o3_station_causal_graph_diagnostics_1h.png",
        "global_hypergraph_1h": results_dir / "o3_station_binary_hypergraph_1h.png",
        "global_hyperedge_topk_1h": results_dir / "o3_station_binary_hyperedge_topk_1h.png",
        "global_hyperedge_cumulative_1h": results_dir / "o3_station_binary_hyperedge_cumulative_1h.png",
        "conditional_synergy_graph_1h": results_dir / "o3_pm25_conditional_synergy_graph_1h.png",
        "conditional_synergy_ratio_graph_1h": results_dir / "o3_pm25_conditional_synergy_ratio_graph_1h.png",
        "conclusion": results_dir / "shanghai_one_step_conclusion.md",
    }

    ds, metadata = load_dataset(cfg, smoke=use_smoke, city_en=city_en)
    full_ds, _ = load_dataset(cfg, smoke=False, city_en=city_en)
    full_city_metadata = select_station_metadata(
        load_station_metadata(cfg),
        available_station_ids=full_ds["station"].values.tolist(),
        city_en=city_en,
    )
    sample_bundle = build_one_step_samples(ds, metadata, cfg, smoke=use_smoke)
    splits = sample_bundle["splits"]
    stats = sample_bundle["stats"]
    target_names = sample_bundle["target_names"]
    station_ids = sample_bundle["station_ids"]
    target_width = len(cfg.target_variables)
    station_target_indices = build_station_variable_index_map(target_names, causal_graph_variable)

    x_train = splits["train"]["X"]
    x_val = splits["val"]["X"]
    x_test = splits["test"]["X"]
    y_train_scaled = splits["train"]["targets"]
    y_val_scaled = splits["val"]["targets"]
    y_test_scaled = splits["test"]["targets"]
    target_dim = y_train_scaled[1].shape[1]
    effective_input_dim = sample_bundle["n_stations"] * sample_bundle["n_features"]
    full_effective_input_dim = len(full_city_metadata) * sample_bundle["n_features"]
    requested_box_size_by_variable = (
        {name: float(value) for name, value in cfg.causal_graph_box_size_by_variable.items()}
        if cfg.causal_graph_box_size_by_variable is not None
        else None
    )
    nonnegative_variables = tuple(str(name) for name in cfg.causal_graph_nonnegative_variables)
    box_label = resolve_causal_graph_box_label(
        box_size=None if requested_box_size_by_variable is not None else cfg.box_size,
        box_size_by_variable=requested_box_size_by_variable,
    )

    run_context = {
        "run_tag": run_tag,
        "city_en": city_en,
        "test_mode": use_smoke,
        "use_smoke": use_smoke,
        "sample_mode": cfg.sample_mode,
        "data_resolution_hours": 1,
        "history_hours": cfg.history_hours,
        "horizons": list(cfg.horizons),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "hidden_dim": cfg.hidden_dim,
        "model_name": cfg.model_name,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "norm_type": cfg.norm_type,
        "activation": cfg.activation,
        "station_count": len(station_ids),
        "input_shape": [sample_bundle["n_stations"], sample_bundle["n_features"]],
        "effective_input_dim": int(effective_input_dim),
        "full_effective_input_dim": int(full_effective_input_dim),
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "coupling_sample_count": coupling_sample_count,
        "graph_edges_per_target": graph_edges_per_target,
        "causal_graph_variable": causal_graph_variable,
        "causal_graph_sampling_mode": "uniform_box_centered_at_train_mean",
        "causal_graph_center_source": "train_input_mean",
        "causal_graph_sampling_seed": int(cfg.seed),
        "causal_graph_edge_keep_fraction": float(causal_graph_edge_keep_fraction),
        "causal_graph_arrow_mutation_scale": float(causal_graph_arrow_mutation_scale),
        "causal_graph_arrow_shrink_target": float(causal_graph_arrow_shrink_target),
    }
    if requested_box_size_by_variable is None:
        run_context["requested_causal_graph_box_size"] = float(cfg.box_size)
        run_context["causal_graph_box_size"] = float(cfg.box_size)
    else:
        run_context["requested_causal_graph_box_size_by_variable"] = requested_box_size_by_variable
        run_context["causal_graph_box_size_by_variable"] = dict(requested_box_size_by_variable)
        run_context["causal_graph_nonnegative_variables"] = list(nonnegative_variables)
        run_context["causal_graph_box_label"] = str(box_label)
    save_json(artifact_paths["config"], run_context)
    return {
        "cfg": cfg,
        "artifact_paths": artifact_paths,
        "sample_bundle": sample_bundle,
        "splits": splits,
        "stats": stats,
        "target_names": target_names,
        "station_ids": station_ids,
        "target_width": target_width,
        "station_target_indices": station_target_indices,
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train_scaled": y_train_scaled,
        "y_val_scaled": y_val_scaled,
        "y_test_scaled": y_test_scaled,
        "target_dim": target_dim,
        "effective_input_dim": int(effective_input_dim),
        "full_effective_input_dim": int(full_effective_input_dim),
        "run_context": run_context,
        "full_city_metadata": full_city_metadata,
        "coupling_sample_count": coupling_sample_count,
        "graph_edges_per_target": graph_edges_per_target,
        "causal_graph_edge_keep_fraction": float(causal_graph_edge_keep_fraction),
        "causal_graph_arrow_mutation_scale": float(causal_graph_arrow_mutation_scale),
        "causal_graph_arrow_shrink_target": float(causal_graph_arrow_shrink_target),
        "causal_graph_variable": causal_graph_variable,
    }


def run_or_load_one_step_predictions(
    bundle: dict[str, object],
    *,
    force_retrain: bool,
) -> dict[str, object]:
    cfg = bundle["cfg"]
    artifact_paths = bundle["artifact_paths"]
    target_dim = bundle["target_dim"]
    target_names = bundle["target_names"]
    stats = bundle["stats"]
    x_test = bundle["x_test"]
    y_test_scaled = bundle["y_test_scaled"]

    assert isinstance(cfg, YRDExperimentConfig)
    assert isinstance(artifact_paths, dict)
    assert isinstance(target_names, list)
    assert isinstance(stats, dict)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_test_scaled, dict)

    set_seed(cfg.seed)
    baseline_model = PersistenceBaseline(target_dim=int(target_dim), horizons=cfg.horizons)

    if (
        Path(artifact_paths["checkpoint"]).exists()
        and Path(artifact_paths["loss_history"]).exists()
        and not force_retrain
    ):
        checkpoint_payload = torch.load(artifact_paths["checkpoint"], map_location="cpu")
        if "model_kwargs" not in checkpoint_payload:
            raise RuntimeError(
                "Checkpoint is missing model_kwargs metadata. "
                "Set FORCE_RETRAIN = True to regenerate caches with the one-step format."
            )
        joint_model = rebuild_joint_model_from_checkpoint(checkpoint_payload)
        loss_history_payload = load_json(artifact_paths["loss_history"])
    else:
        training_result = train_joint_model_with_history(
            n_stations=bundle["sample_bundle"]["n_stations"],
            n_features=bundle["sample_bundle"]["n_features"],
            history_hours=cfg.history_hours,
            target_dim=int(target_dim),
            hidden_dim=cfg.hidden_dim,
            horizons=cfg.horizons,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            max_epochs=cfg.max_epochs,
            early_stopping_patience=cfg.early_stopping_patience,
            seed=cfg.seed,
            x_train=bundle["x_train"],
            y_train=bundle["y_train_scaled"],
            x_val=bundle["x_val"],
            y_val=bundle["y_val_scaled"],
            model_name=cfg.model_name,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            norm_type=cfg.norm_type,
            activation=cfg.activation,
        )
        joint_model = training_result["model"]
        checkpoint_payload = {
            "state_dict": joint_model.state_dict(),
            "best_epoch": training_result["best_epoch"],
            "best_val_loss": training_result["best_val_loss"],
            "train_loss_history": training_result["train_loss_history"],
            "val_loss_history": training_result["val_loss_history"],
            "model_kwargs": training_result["model_kwargs"],
            "run_context": bundle["run_context"],
        }
        torch.save(checkpoint_payload, artifact_paths["checkpoint"])
        loss_history_payload = {
            "best_epoch": training_result["best_epoch"],
            "best_val_loss": training_result["best_val_loss"],
            "train_loss_history": training_result["train_loss_history"],
            "val_loss_history": training_result["val_loss_history"],
        }
        save_json(artifact_paths["loss_history"], loss_history_payload)

    joint_model.eval()
    baseline_scaled_predictions = _predict_numpy(baseline_model, x_test, cfg.horizons)
    joint_scaled_predictions = _predict_numpy(joint_model, x_test, cfg.horizons)

    y_test_original = {1: inverse_transform_targets(y_test_scaled[1], target_names, stats)}
    baseline_original_predictions = {
        1: inverse_transform_targets(baseline_scaled_predictions[1], target_names, stats)
    }
    joint_original_predictions = {
        1: inverse_transform_targets(joint_scaled_predictions[1], target_names, stats)
    }

    np.savez(
        artifact_paths["predictions"],
        y_test_scaled_1h=y_test_scaled[1],
        baseline_scaled_1h=baseline_scaled_predictions[1],
        joint_scaled_1h=joint_scaled_predictions[1],
        y_test_original_1h=y_test_original[1],
        baseline_original_1h=baseline_original_predictions[1],
        joint_original_1h=joint_original_predictions[1],
    )

    run_manifest = {
        "cache_dir": str(Path(artifact_paths["config"]).parent),
        "results_dir": str(Path(artifact_paths["run_manifest"]).parent),
    }
    for key in ("config", "checkpoint", "loss_history", "predictions"):
        path = Path(artifact_paths[key])
        if path.exists():
            run_manifest[key] = str(path)
    save_json(artifact_paths["run_manifest"], run_manifest)
    return {
        "baseline_model": baseline_model,
        "joint_model": joint_model,
        "checkpoint_payload": checkpoint_payload,
        "loss_history_payload": loss_history_payload,
        "baseline_scaled_predictions": baseline_scaled_predictions,
        "joint_scaled_predictions": joint_scaled_predictions,
        "y_test_original": y_test_original,
        "baseline_original_predictions": baseline_original_predictions,
        "joint_original_predictions": joint_original_predictions,
        "run_manifest": run_manifest,
    }


def build_prediction_tables(
    bundle: dict[str, object],
    predictions: dict[str, object],
) -> dict[str, object]:
    cfg = bundle["cfg"]
    target_names = bundle["target_names"]
    station_ids = bundle["station_ids"]
    target_width = bundle["target_width"]

    assert isinstance(cfg, YRDExperimentConfig)
    assert isinstance(target_names, list)
    assert isinstance(station_ids, list)
    assert isinstance(target_width, int)

    prediction_sets = {
        "PersistenceBaseline": predictions["baseline_original_predictions"],
        "JointStationMLP": predictions["joint_original_predictions"],
    }
    metrics_frame = pd.DataFrame(
        metric_rows_for_scope(
            predictions["y_test_original"],
            prediction_sets,
            target_names=target_names,
        )
    )
    metrics_overall_df = metrics_frame[metrics_frame["scope"] == "overall"].reset_index(drop=True)
    metrics_pollutant_df = metrics_frame[metrics_frame["scope"].isin(cfg.target_variables)].reset_index(drop=True)
    station_metrics_df = pd.DataFrame(
        station_metric_rows(
            predictions["y_test_original"],
            prediction_sets,
            station_ids=station_ids,
            target_width=target_width,
        )
    )

    save_json(
        bundle["artifact_paths"]["metrics"],
        {
            "overall": metrics_overall_df.to_dict(orient="records"),
            "by_pollutant": metrics_pollutant_df.to_dict(orient="records"),
            "by_station": station_metrics_df.to_dict(orient="records"),
        },
    )
    return {
        "prediction_sets": prediction_sets,
        "metrics_frame": metrics_frame,
        "metrics_overall_df": metrics_overall_df,
        "metrics_pollutant_df": metrics_pollutant_df,
        "station_metrics_df": station_metrics_df,
    }


def compute_station_causal_graph_results(
    bundle: dict[str, object],
    predictions: dict[str, object],
    *,
    force_recompute: bool,
    coupling_method: str = "nis",
) -> dict[str, object]:
    cfg = bundle["cfg"]
    x_train = bundle["x_train"]
    station_ids = bundle["station_ids"]
    sample_bundle = bundle["sample_bundle"]
    causal_graph_variable = bundle["causal_graph_variable"]
    graph_edges_per_target = bundle["graph_edges_per_target"]
    causal_graph_edge_keep_fraction = float(bundle["causal_graph_edge_keep_fraction"])
    causal_graph_arrow_mutation_scale = float(bundle["causal_graph_arrow_mutation_scale"])
    causal_graph_arrow_shrink_target = float(bundle["causal_graph_arrow_shrink_target"])
    run_context = bundle["run_context"]
    requested_box_size_by_variable = run_context.get("requested_causal_graph_box_size_by_variable")
    if requested_box_size_by_variable is not None:
        requested_box_size_by_variable = {
            str(name): float(value)
            for name, value in dict(requested_box_size_by_variable).items()
        }
    requested_box_size = (
        float(run_context.get("requested_causal_graph_box_size", cfg.box_size))
        if requested_box_size_by_variable is None
        else None
    )
    nonnegative_variables = tuple(str(name) for name in run_context.get("causal_graph_nonnegative_variables", ()))
    sampling_mode = str(run_context.get("causal_graph_sampling_mode", "uniform_box_centered_at_train_mean"))
    sampling_seed = int(run_context.get("causal_graph_sampling_seed", cfg.seed))
    coupling_sample_count = int(bundle["coupling_sample_count"])
    coupling_method = str(coupling_method).lower()

    assert isinstance(cfg, YRDExperimentConfig)
    assert isinstance(x_train, np.ndarray)
    assert isinstance(station_ids, list)
    assert isinstance(sample_bundle, dict)
    assert isinstance(causal_graph_variable, str)
    assert isinstance(graph_edges_per_target, int)

    station_source_groups = build_one_step_station_source_groups(
        n_stations=sample_bundle["n_stations"],
        n_features=sample_bundle["n_features"],
        station_ids=station_ids,
    )
    station_pollutant_feature_groups = build_one_step_station_pollutant_feature_groups(
        n_stations=sample_bundle["n_stations"],
        n_features=sample_bundle["n_features"],
        pollutant_feature_indices={
            "O3": cfg.input_variables.index("O3"),
            "PM2.5": cfg.input_variables.index("PM2.5"),
        },
        station_ids=station_ids,
    )
    sigma_eps_by_horizon = {
        1: estimate_residual_covariance(
            bundle["y_test_scaled"][1],
            predictions["joint_scaled_predictions"][1],
        )
    }
    artifact_paths = bundle["artifact_paths"]

    sample_cache_path = Path(artifact_paths["global_coupling_samples"])
    summary_cache_path = Path(artifact_paths["global_coupling_summary"])
    has_cache = sample_cache_path.exists() and summary_cache_path.exists() and not force_recompute
    if has_cache:
        global_coupling_sample_records = load_json(artifact_paths["global_coupling_samples"])["records"]
        global_coupling_summary_by_horizon = load_json(artifact_paths["global_coupling_summary"])
        horizon_cache = dict(global_coupling_summary_by_horizon.get("1h", {}))
        has_binary_cache = "binary_hyperedges" in horizon_cache
        has_synergy_cache = "conditional_synergy_edges" in horizon_cache
        has_ratio_cache = "conditional_synergy_ratio_edges" in horizon_cache
        cache_meta = dict(global_coupling_summary_by_horizon.get("_meta", {}))
        if (
            not has_binary_cache
            or not has_synergy_cache
            or not has_ratio_cache
            or str(cache_meta.get("coupling_method", "nis")) != coupling_method
            or not causal_graph_cache_is_valid(
                cache_meta=cache_meta,
                requested_box_size=requested_box_size,
                requested_box_size_by_variable=requested_box_size_by_variable,
                nonnegative_variables=nonnegative_variables,
                coupling_sample_count=coupling_sample_count,
                sampling_mode=sampling_mode,
                sampling_seed=sampling_seed,
            )
        ):
            has_cache = False
        else:
            if requested_box_size_by_variable is None:
                run_context["causal_graph_box_size"] = float(cache_meta["causal_graph_box_size"])
            else:
                run_context["causal_graph_box_size_by_variable"] = dict(cache_meta["causal_graph_box_size_by_variable"])
                run_context["causal_graph_nonnegative_variables"] = list(cache_meta["causal_graph_nonnegative_variables"])
            cached_pairwise_edges = pd.DataFrame(global_coupling_summary_by_horizon["1h"]["pairwise_edges"])
            cached_non_self_pairwise = cached_pairwise_edges[
                cached_pairwise_edges["source_station_id"] != cached_pairwise_edges["target_station_id"]
            ].copy()
            if cached_non_self_pairwise.empty or not bool((cached_non_self_pairwise["mean"] > 0).all()):
                has_cache = False
        if has_cache:
            run_context["coupling_method"] = coupling_method
            save_json(artifact_paths["config"], run_context)
    if not has_cache:
        intervention_center = compute_training_input_center(x_train)
        sampling_box_size: float | np.ndarray
        sampling_lower_bounds: np.ndarray | None = None
        if requested_box_size_by_variable is None:
            sampling_box_size = float(requested_box_size)
        else:
            sampling_box_size = resolve_variable_box_size_by_feature(
                input_variables=cfg.input_variables,
                box_size_by_variable=requested_box_size_by_variable,
                n_stations=sample_bundle["n_stations"],
            )
            sampling_lower_bounds = resolve_nonnegative_lower_bounds_by_feature(
                input_variables=cfg.input_variables,
                stats=bundle["stats"],
                nonnegative_variables=nonnegative_variables,
                n_stations=sample_bundle["n_stations"],
            )
        synthetic_inputs = sample_uniform_box_inputs(
            center=intervention_center,
            box_size=sampling_box_size,
            sample_count=coupling_sample_count,
            seed=sampling_seed,
            lower_bounds=sampling_lower_bounds,
        )
        joint_model = predictions["joint_model"]
        joint_model.eval()
        if coupling_method == "nis":
            if requested_box_size_by_variable is not None:
                raise ValueError("Variable-specific causal-graph box sizes are currently supported only for TM coupling.")
            raw_coupling_records: list[dict[str, object]] = []
            for sample_id, synthetic_input in enumerate(synthetic_inputs):
                sample_x = torch.from_numpy(synthetic_input[None, ...]).to(dtype=torch.float32)
                flat_sample = sample_x.reshape(-1).detach().clone().requires_grad_(True)

                def horizon_model(tensor: torch.Tensor) -> torch.Tensor:
                    shaped = tensor.reshape(1, sample_bundle["n_stations"], sample_bundle["n_features"])
                    return joint_model(shaped)[1].reshape(-1)

                for target_station_id, target_indices in bundle["station_target_indices"].items():
                    jacobian = jacobian_for_target_subset(
                        horizon_model,
                        flat_sample,
                        target_indices=target_indices,
                    ).detach().cpu().numpy()
                    raw_coupling_records.append(
                        {
                            "sample_id": int(sample_id),
                            "sampling_mode": sampling_mode,
                            "target_station_id": target_station_id,
                            "target_indices": list(target_indices),
                            "jacobian": jacobian.tolist(),
                            "sigma_eps": sigma_eps_by_horizon[1].tolist(),
                        }
                    )

            selected_box_size, global_coupling_sample_records, global_coupling_summary_by_horizon = resolve_causal_graph_box_size(
                raw_records=raw_coupling_records,
                station_ids=station_ids,
                station_source_groups=station_source_groups,
                station_pollutant_feature_groups=station_pollutant_feature_groups,
                requested_box_size=float(requested_box_size),
                graph_edges_per_target=graph_edges_per_target,
                causal_graph_variable=causal_graph_variable,
            )
        elif coupling_method == "tm":
            with torch.no_grad():
                predicted_next = joint_model(torch.from_numpy(synthetic_inputs).to(dtype=torch.float32))[1].detach().cpu().numpy()
            predicted_next_o3 = np.stack(
                [
                    predicted_next[:, bundle["station_target_indices"][station_id][0]]
                    for station_id in station_ids
                ],
                axis=1,
            )
            selected_box_size = requested_box_size
            global_coupling_sample_records = [
                {
                    "sample_id": int(sample_id),
                    "sampling_mode": sampling_mode,
                    "source_sample": synthetic_inputs[sample_id].tolist(),
                    "predicted_next_o3": predicted_next_o3[sample_id].tolist(),
                }
                for sample_id in range(synthetic_inputs.shape[0])
            ]
            global_coupling_summary_by_horizon = build_transport_map_global_causal_summary(
                source_samples=synthetic_inputs,
                predicted_next_o3=predicted_next_o3,
                station_ids=station_ids,
                o3_feature_index=cfg.input_variables.index("O3"),
                pm25_feature_index=cfg.input_variables.index("PM2.5"),
            )
        else:
            raise ValueError(f"Unsupported coupling method: {coupling_method}")
        meta: dict[str, object] = {
            "coupling_method": coupling_method,
            "coupling_sample_count": coupling_sample_count,
            "sampling_mode": sampling_mode,
            "sampling_seed": sampling_seed,
        }
        if requested_box_size_by_variable is None:
            meta["requested_causal_graph_box_size"] = float(requested_box_size)
            meta["causal_graph_box_size"] = float(selected_box_size)
            run_context["causal_graph_box_size"] = float(selected_box_size)
        else:
            meta["requested_causal_graph_box_size_by_variable"] = dict(requested_box_size_by_variable)
            meta["causal_graph_box_size_by_variable"] = dict(requested_box_size_by_variable)
            meta["causal_graph_nonnegative_variables"] = list(nonnegative_variables)
            run_context["causal_graph_box_size_by_variable"] = dict(requested_box_size_by_variable)
            run_context["causal_graph_nonnegative_variables"] = list(nonnegative_variables)
        global_coupling_summary_by_horizon["_meta"] = meta
        run_context["coupling_method"] = coupling_method
        save_json(artifact_paths["config"], run_context)
        save_json(artifact_paths["global_coupling_samples"], {"records": global_coupling_sample_records})
        save_json(artifact_paths["global_coupling_summary"], global_coupling_summary_by_horizon)

    pairwise_edges_df = pd.DataFrame(global_coupling_summary_by_horizon["1h"]["pairwise_edges"])
    binary_hyperedges_df = pd.DataFrame(global_coupling_summary_by_horizon["1h"]["binary_hyperedges"])
    conditional_synergy_edges_df = pd.DataFrame(global_coupling_summary_by_horizon["1h"]["conditional_synergy_edges"])
    conditional_synergy_ratio_edges_df = pd.DataFrame(
        global_coupling_summary_by_horizon["1h"]["conditional_synergy_ratio_edges"]
    )
    pairwise_self_loop_strengths_1h = build_self_loop_node_strengths(pairwise_edges_df, station_ids=station_ids)
    conditional_synergy_self_loop_strengths_1h = build_self_loop_node_strengths(
        conditional_synergy_edges_df,
        station_ids=station_ids,
    )
    conditional_synergy_ratio_self_loop_strengths_1h = build_self_loop_node_strengths(
        conditional_synergy_ratio_edges_df,
        station_ids=station_ids,
    )
    pairwise_display_1h = (
        pairwise_edges_df[
            (pairwise_edges_df["source_station_id"] != pairwise_edges_df["target_station_id"])
            & (pairwise_edges_df["mean"] > 0)
        ]
        .sort_values(["target_station_id", "mean"], ascending=[True, False])
        .reset_index(drop=True)
        if not pairwise_edges_df.empty
        else pd.DataFrame()
    )
    binary_hyperedges_display_1h = (
        select_display_hyperedge_rows(binary_hyperedges_df, per_target=graph_edges_per_target)
        if not binary_hyperedges_df.empty
        else pd.DataFrame()
    )
    conditional_synergy_edges_display_1h = (
        select_display_rows(
            conditional_synergy_edges_df,
            per_target=graph_edges_per_target,
            source_col="source_station_id",
            target_col="target_station_id",
            ranking_col="mean",
            positive_only=True,
            include_self_loops=False,
        )
        if not conditional_synergy_edges_df.empty
        else pd.DataFrame()
    )
    conditional_synergy_ratio_edges_display_1h = (
        select_display_rows(
            conditional_synergy_ratio_edges_df,
            per_target=graph_edges_per_target,
            source_col="source_station_id",
            target_col="target_station_id",
            ranking_col="mean",
            positive_only=True,
            include_self_loops=False,
        )
        if not conditional_synergy_ratio_edges_df.empty
        else pd.DataFrame()
    )
    conditional_synergy_edges_global_ranked_1h = build_global_edge_ranking(
        conditional_synergy_edges_df,
        sort_col="mean",
    )
    conditional_synergy_ratio_edges_global_ranked_1h = build_global_edge_ranking(
        conditional_synergy_ratio_edges_df,
        sort_col="mean",
    )
    binary_hyperedges_global_ranked_1h = build_global_hyperedge_ranking(binary_hyperedges_df)
    binary_hyperedge_cutoff_summary_1h = summarize_hyperedge_cutoff(binary_hyperedges_global_ranked_1h)
    box_annotation = run_context.get("causal_graph_box_label")
    if box_annotation is None:
        box_annotation = run_context.get("causal_graph_box_size")
    pairwise_visual_display_1h = select_top_fraction_edges(
        pairwise_display_1h,
        strength_col="mean",
        keep_fraction=causal_graph_edge_keep_fraction,
    )
    pairwise_render_edges_1h = build_pairwise_edge_render_frame(pairwise_visual_display_1h, strength_col="mean")
    pairwise_degree_summary_1h = build_pairwise_degree_summary(
        pairwise_display_1h,
        station_ids=station_ids,
        strength_col="mean",
    )
    pairwise_diagnostics_edges_1h = build_pairwise_edge_render_frame(pairwise_display_1h, strength_col="mean")
    conditional_synergy_visual_display_1h = select_top_fraction_edges(
        conditional_synergy_edges_display_1h,
        strength_col="mean",
        keep_fraction=causal_graph_edge_keep_fraction,
    )
    conditional_synergy_ratio_visual_display_1h = select_top_fraction_edges(
        conditional_synergy_ratio_edges_display_1h,
        strength_col="mean",
        keep_fraction=causal_graph_edge_keep_fraction,
    )
    pairwise_render_edges_1h.to_csv(artifact_paths["global_graph_edge_weights_1h"], index=False)
    pairwise_degree_summary_1h.to_csv(artifact_paths["global_graph_degree_summary_1h"], index=False)
    draw_station_causal_graph(
        station_positions=bundle["full_city_metadata"][["station_id", "lon", "lat"]],
        pairwise_edges=pairwise_visual_display_1h,
        horizon_label="1h",
        out_path=artifact_paths["global_graph_1h"],
        strength_col="mean",
        box_size=box_annotation,
        arrow_mutation_scale=causal_graph_arrow_mutation_scale,
        arrow_shrink_target=causal_graph_arrow_shrink_target,
        node_self_strengths=pairwise_self_loop_strengths_1h,
        node_colorbar_label="Self EI^tm",
    )
    draw_pairwise_ei_degree_diagnostics(
        render_edges=pairwise_diagnostics_edges_1h,
        node_degree_summary=pairwise_degree_summary_1h,
        out_path=artifact_paths["global_graph_diagnostics_1h"],
        horizon_label="1h",
        box_size=box_annotation,
    )
    draw_station_hyperedge_graph(
        station_positions=bundle["full_city_metadata"][["station_id", "lon", "lat"]],
        binary_hyperedges=binary_hyperedges_display_1h,
        horizon_label="1h",
        out_path=artifact_paths["global_hypergraph_1h"],
        box_size=box_annotation,
        arrow_mutation_scale=causal_graph_arrow_mutation_scale,
        arrow_shrink_target=causal_graph_arrow_shrink_target,
    )
    draw_global_hyperedge_topk_plot(
        ranked_hyperedges=binary_hyperedges_global_ranked_1h,
        out_path=artifact_paths["global_hyperedge_topk_1h"],
        box_size=box_annotation,
    )
    draw_global_hyperedge_cumulative_plot(
        ranked_hyperedges=binary_hyperedges_global_ranked_1h,
        out_path=artifact_paths["global_hyperedge_cumulative_1h"],
        box_size=box_annotation,
    )
    draw_station_causal_graph(
        station_positions=bundle["full_city_metadata"][["station_id", "lon", "lat"]],
        pairwise_edges=conditional_synergy_visual_display_1h,
        horizon_label="1h",
        out_path=artifact_paths["conditional_synergy_graph_1h"],
        title="Shanghai O3 + PM2.5 conditional synergy graph (1h)",
        positive_color=SYNERGY_POSITIVE_EDGE_COLOR,
        legend_label="O3 + PM2.5 conditional synergy",
        strength_col="mean",
        box_size=box_annotation,
        arrow_mutation_scale=causal_graph_arrow_mutation_scale,
        arrow_shrink_target=causal_graph_arrow_shrink_target,
        node_self_strengths=conditional_synergy_self_loop_strengths_1h,
        node_colorbar_label="Self Syn^tm",
    )
    draw_station_causal_graph(
        station_positions=bundle["full_city_metadata"][["station_id", "lon", "lat"]],
        pairwise_edges=conditional_synergy_ratio_visual_display_1h,
        horizon_label="1h",
        out_path=artifact_paths["conditional_synergy_ratio_graph_1h"],
        title="Shanghai O3 + PM2.5 synergy-to-EI ratio graph (1h)",
        positive_color=SYNERGY_RATIO_EDGE_COLOR,
        legend_label="Syn / EI_joint",
        strength_col="mean",
        box_size=box_annotation,
        arrow_mutation_scale=causal_graph_arrow_mutation_scale,
        arrow_shrink_target=causal_graph_arrow_shrink_target,
        node_self_strengths=conditional_synergy_ratio_self_loop_strengths_1h,
        node_colorbar_label="Self Syn / EI_joint",
    )
    return {
        "binary_hyperedges_df": binary_hyperedges_df,
        "binary_hyperedges_display_1h": binary_hyperedges_display_1h,
        "binary_hyperedges_global_ranked_1h": binary_hyperedges_global_ranked_1h,
        "binary_hyperedge_cutoff_summary_1h": binary_hyperedge_cutoff_summary_1h,
        "conditional_synergy_edges_df": conditional_synergy_edges_df,
        "conditional_synergy_edges_display_1h": conditional_synergy_edges_display_1h,
        "conditional_synergy_edges_global_ranked_1h": conditional_synergy_edges_global_ranked_1h,
        "conditional_synergy_self_loop_strengths_1h": conditional_synergy_self_loop_strengths_1h,
        "conditional_synergy_ratio_edges_df": conditional_synergy_ratio_edges_df,
        "conditional_synergy_ratio_edges_display_1h": conditional_synergy_ratio_edges_display_1h,
        "conditional_synergy_ratio_edges_global_ranked_1h": conditional_synergy_ratio_edges_global_ranked_1h,
        "conditional_synergy_ratio_self_loop_strengths_1h": conditional_synergy_ratio_self_loop_strengths_1h,
        "pairwise_display_1h": pairwise_display_1h,
        "pairwise_self_loop_strengths_1h": pairwise_self_loop_strengths_1h,
        "pairwise_visual_display_1h": pairwise_visual_display_1h,
        "pairwise_render_edges_1h": pairwise_render_edges_1h,
        "pairwise_degree_summary_1h": pairwise_degree_summary_1h,
        "global_coupling_summary_by_horizon": global_coupling_summary_by_horizon,
    }


def metric_value(
    frame: pd.DataFrame,
    *,
    model: str,
    horizon: str,
    scope: str,
    metric: str,
) -> float:
    row = frame[
        frame["model"].eq(model)
        & frame["horizon"].eq(horizon)
        & frame["scope"].eq(scope)
    ].iloc[0]
    return float(row[metric])


def write_shanghai_one_step_conclusion(
    bundle: dict[str, object],
    *,
    metrics_frame: pd.DataFrame,
    conditional_synergy_edges_display_1h: pd.DataFrame,
) -> str:
    sample_bundle = bundle["sample_bundle"]
    station_ids = bundle["station_ids"]
    effective_input_dim = bundle["effective_input_dim"]
    full_effective_input_dim = bundle["full_effective_input_dim"]

    assert isinstance(sample_bundle, dict)
    assert isinstance(station_ids, list)
    assert isinstance(effective_input_dim, int)
    assert isinstance(full_effective_input_dim, int)

    joint_1h_rmse = metric_value(
        metrics_frame,
        model="JointStationMLP",
        horizon="1h",
        scope="overall",
        metric="rmse",
    )
    baseline_1h_rmse = metric_value(
        metrics_frame,
        model="PersistenceBaseline",
        horizon="1h",
        scope="overall",
        metric="rmse",
    )
    strongest_conditional_synergy_edge_1h = (
        conditional_synergy_edges_display_1h.sort_values("mean", ascending=False).iloc[0]
        if not conditional_synergy_edges_display_1h.empty
        else None
    )
    run_context = bundle["run_context"]
    if "causal_graph_box_size_by_variable" in run_context:
        box_description = "按变量设置的 train-q99 干预盒 L_v，并对非负变量施加原始量纲下的 0 下界裁剪"
    else:
        box_description = f"边长 L = {float(run_context['causal_graph_box_size']):.3f} 的输入盒"

    conclusion_lines = [
        f"本 notebook 使用原始 1h 分辨率数据，在上海单城 {len(station_ids)} 个站点上构造当前时刻联合快照输入。",
        (
            f"每个样本的输入形状为 {sample_bundle['n_stations']} x {sample_bundle['n_features']}；"
            f"当前运行的有效输入维度为 {effective_input_dim}，"
            f"full Shanghai 设置下对应 {full_effective_input_dim}（即 190 维一步输入）。"
        ),
        (
            f"当前实验只做一步 1h 预测；在整体 1h 预测上，"
            f"JointStationMLP 的 RMSE 为 {joint_1h_rmse:.3f}，"
            f"PersistenceBaseline 的 RMSE 为 {baseline_1h_rmse:.3f}。"
        ),
        (
            "因果图平均不是跨真实测试时刻取样，而是对训练集输入均值居中的均匀干预样本做 Monte Carlo 平均；"
            f"采样分布为{box_description}。"
        ),
        (
            "在条件协同图分析中，source side 表示每个源站点当前时刻的 O3 + PM2.5，"
            "target side 表示下一小时目标站点的 O3，其余输入维度保留为条件背景。"
        ),
    ]
    if strongest_conditional_synergy_edge_1h is not None:
        conclusion_lines.append(
            "在当前 1h O3 + PM2.5 条件协同图里，最强的显示边是 "
            f"{strongest_conditional_synergy_edge_1h['source_station_id']} -> "
            f"{strongest_conditional_synergy_edge_1h['target_station_id']}，"
            f"其平均协同强度为 {strongest_conditional_synergy_edge_1h['mean']:.3f}。"
        )
    final_conclusion_text = "\n".join(conclusion_lines)
    Path(bundle["artifact_paths"]["conclusion"]).write_text(final_conclusion_text)
    return final_conclusion_text


# --- Former yrd/air_search.py ---

import json
import math
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch


_YRD_CITIES = frozenset({"shanghai", "nanjing", "hangzhou"})
_BTHSA_CITIES = frozenset({"beijing"})
DEFAULT_TM_NONNEGATIVE_VARIABLES = (
    "O3",
    "PM2.5",
    "t2m",
    "d2m",
    "sp",
    "tp",
    "blh",
    "msdwswrf",
)
DEFAULT_COARSE_SAMPLE_COUNT = 16
DEFAULT_COARSE_SAMPLE_COUNT_SMOKE = 4
DEFAULT_NEGATIVE_RATIO_THRESHOLD = 0.10


@dataclass(frozen=True)
class CityScope:
    city_en: str
    dataset_path: Path
    station_path: Path


def _to_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def resolve_city_scope(city_en: str) -> CityScope:
    city_key = str(city_en).strip().lower()
    if city_key in _YRD_CITIES:
        return CityScope(
            city_en=city_key,
            dataset_path=Path("data/dataset_yrd.nc"),
            station_path=Path("data/stations_yrd.csv"),
        )
    if city_key in _BTHSA_CITIES:
        return CityScope(
            city_en=city_key,
            dataset_path=Path("data/dataset_bthsa.nc"),
            station_path=Path("data/stations_bthsa.csv"),
        )
    supported = sorted(_YRD_CITIES | _BTHSA_CITIES)
    raise ValueError(f"Unsupported city_en={city_en!r}. Supported values: {supported}")


def build_air_search_config(
    root_dir: Path,
    *,
    city_en: str,
    horizon: int,
    test_mode: bool,
) -> YRDExperimentConfig:
    scope = resolve_city_scope(city_en)
    horizon_int = int(horizon)
    if horizon_int <= 0:
        raise ValueError("horizon must be a positive integer number of hours.")
    return replace(
        YRDExperimentConfig(
            root_dir=Path(root_dir),
            dataset_path=scope.dataset_path,
            station_path=scope.station_path,
        ),
        sample_mode="one_step",
        history_hours=1,
        horizons=(horizon_int,),
        model_name="resmlp",
        hidden_dim=16 if test_mode else 96,
        num_layers=2 if test_mode else 3,
        dropout=0.0 if test_mode else 0.05,
        norm_type="layernorm",
        activation="silu",
        learning_rate=5e-4,
        weight_decay=0.0 if test_mode else 1e-5,
        batch_size=8 if test_mode else 64,
        epochs=2 if test_mode else 12,
        max_epochs=4 if test_mode else 60,
        early_stopping_patience=2 if test_mode else 8,
        seed=0,
    )


def build_air_search_artifact_paths(
    *,
    root_dir: Path,
    city_en: str,
    horizon: int,
    run_tag: str,
    use_smoke: bool,
) -> dict[str, Path]:
    scope = resolve_city_scope(city_en)
    horizon_int = int(horizon)
    if horizon_int <= 0:
        raise ValueError("horizon must be a positive integer number of hours.")
    horizon_label = f"{horizon_int}h"
    base_root = Path(tempfile.gettempdir()) / "eisyn" if use_smoke else Path(root_dir)
    cache_dir = (
        base_root / "exp" / "cache" / "yrd_coupling" / "air_search" / scope.city_en / horizon_label / run_tag
    )
    results_dir = base_root / "fig" / "yrd_air_search" / scope.city_en / horizon_label / run_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return {
        "cache_dir": cache_dir,
        "results_dir": results_dir,
        "config": cache_dir / "config.json",
        "checkpoint": cache_dir / "joint_model_checkpoint.pt",
        "loss_history": cache_dir / "loss_history.json",
        "predictions": cache_dir / "test_predictions.npz",
        "metrics": cache_dir / "metrics_summary.json",
        "coarse_summary": cache_dir / "coarse_summary.json",
        "refine_summary": cache_dir / "refine_summary.json",
        "leaderboard_row": cache_dir / "leaderboard_row.json",
        "run_manifest": results_dir / "run_manifest.json",
        "o3_pairwise_graph": results_dir / "o3_pairwise_graph.png",
        "pm25_to_o3_pairwise_graph": results_dir / "pm25_to_o3_pairwise_graph.png",
        "o3_pm25_synergy_graph": results_dir / "o3_pm25_synergy_graph.png",
        "report_manifest": results_dir / "report_manifest.json",
    }


def inverse_transform_targets(
    array: np.ndarray,
    target_names: list[str],
    stats: dict[str, dict[str, float]],
) -> np.ndarray:
    restored = np.asarray(array, dtype=np.float32).copy()
    for index, name in enumerate(target_names):
        variable = name.split("__")[-1]
        restored[:, index] = restored[:, index] * stats[variable]["std"] + stats[variable]["mean"]
    return restored


def _single_horizon(cfg: YRDExperimentConfig) -> int:
    if len(cfg.horizons) != 1:
        raise ValueError("Air search bundle helpers currently expect exactly one forecast horizon per run.")
    return int(cfg.horizons[0])


def resolve_data_root(root_dir: Path, *, dataset_path: Path, station_path: Path) -> Path:
    start = Path(root_dir).resolve()
    for candidate in (start, *start.parents):
        if (candidate / dataset_path).exists() and (candidate / station_path).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate data root containing {dataset_path} and {station_path} starting from {start}."
    )


def _resolve_target_dim(splits: dict[str, dict[str, object]], *, horizon: int) -> int:
    for split_name in ("train", "val", "test"):
        payload = splits[split_name]
        targets = payload["targets"]
        if not isinstance(targets, dict):
            continue
        values = targets.get(horizon)
        if isinstance(values, np.ndarray) and values.ndim == 2 and values.shape[1] > 0:
            return int(values.shape[1])
    raise ValueError(f"Could not resolve target_dim for horizon={horizon}.")


def prepare_air_search_bundle(
    *,
    cfg: YRDExperimentConfig,
    city_en: str,
    run_tag: str,
    use_smoke: bool,
) -> dict[str, object]:
    horizon = _single_horizon(cfg)
    artifact_paths = build_air_search_artifact_paths(
        root_dir=cfg.root_dir,
        city_en=city_en,
        horizon=horizon,
        run_tag=run_tag,
        use_smoke=use_smoke,
    )
    data_cfg = replace(
        cfg,
        root_dir=resolve_data_root(
            cfg.root_dir,
            dataset_path=cfg.dataset_path,
            station_path=cfg.station_path,
        ),
    )
    ds, metadata = load_dataset(data_cfg, smoke=use_smoke, city_en=city_en)
    sample_bundle = build_one_step_samples(ds, metadata, cfg, smoke=use_smoke)
    splits = sample_bundle["splits"]
    stats = sample_bundle["stats"]
    target_names = sample_bundle["target_names"]
    station_ids = sample_bundle["station_ids"]
    target_width = len(cfg.target_variables)
    target_dim = _resolve_target_dim(splits, horizon=horizon)

    x_train = splits["train"]["X"]
    x_val = splits["val"]["X"]
    x_test = splits["test"]["X"]
    y_train_scaled = splits["train"]["targets"]
    y_val_scaled = splits["val"]["targets"]
    y_test_scaled = splits["test"]["targets"]
    effective_input_dim = sample_bundle["n_stations"] * sample_bundle["n_features"]
    run_context = {
        "run_tag": run_tag,
        "city_en": str(city_en),
        "data_root": str(data_cfg.root_dir),
        "test_mode": bool(use_smoke),
        "use_smoke": bool(use_smoke),
        "sample_mode": cfg.sample_mode,
        "data_resolution_hours": 1,
        "history_hours": cfg.history_hours,
        "horizons": list(cfg.horizons),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "hidden_dim": cfg.hidden_dim,
        "model_name": cfg.model_name,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "norm_type": cfg.norm_type,
        "activation": cfg.activation,
        "station_count": len(station_ids),
        "input_shape": [sample_bundle["n_stations"], sample_bundle["n_features"]],
        "effective_input_dim": int(effective_input_dim),
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "test_samples": int(x_test.shape[0]),
    }
    save_json(artifact_paths["config"], _to_jsonable(run_context))
    return {
        "cfg": cfg,
        "artifact_paths": artifact_paths,
        "sample_bundle": sample_bundle,
        "city_metadata": metadata,
        "splits": splits,
        "stats": stats,
        "target_names": target_names,
        "station_ids": station_ids,
        "target_width": target_width,
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train_scaled": y_train_scaled,
        "y_val_scaled": y_val_scaled,
        "y_test_scaled": y_test_scaled,
        "target_dim": target_dim,
        "effective_input_dim": int(effective_input_dim),
        "run_context": run_context,
    }


def _build_metrics_payload(
    *,
    cfg: YRDExperimentConfig,
    y_test_original: dict[int, np.ndarray],
    baseline_original_predictions: dict[int, np.ndarray],
    joint_original_predictions: dict[int, np.ndarray],
) -> dict[str, object]:
    baseline_metrics: dict[str, dict[str, float]] = {}
    joint_metrics: dict[str, dict[str, float]] = {}
    for horizon in cfg.horizons:
        baseline_metrics[str(horizon)] = _compute_metrics(
            y_test_original[horizon],
            baseline_original_predictions[horizon],
        )
        joint_metrics[str(horizon)] = _compute_metrics(
            y_test_original[horizon],
            joint_original_predictions[horizon],
        )
    return {
        "baseline_test": baseline_metrics,
        "joint_test": joint_metrics,
    }


def _prediction_payload(
    *,
    cfg: YRDExperimentConfig,
    y_test_scaled: dict[int, np.ndarray],
    baseline_scaled_predictions: dict[int, np.ndarray],
    joint_scaled_predictions: dict[int, np.ndarray],
    y_test_original: dict[int, np.ndarray],
    baseline_original_predictions: dict[int, np.ndarray],
    joint_original_predictions: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for horizon in cfg.horizons:
        horizon_label = f"{int(horizon)}h"
        payload[f"y_test_scaled_{horizon_label}"] = y_test_scaled[horizon]
        payload[f"baseline_scaled_{horizon_label}"] = baseline_scaled_predictions[horizon]
        payload[f"joint_scaled_{horizon_label}"] = joint_scaled_predictions[horizon]
        payload[f"y_test_original_{horizon_label}"] = y_test_original[horizon]
        payload[f"baseline_original_{horizon_label}"] = baseline_original_predictions[horizon]
        payload[f"joint_original_{horizon_label}"] = joint_original_predictions[horizon]
    return payload


def run_or_load_air_search_predictions(
    bundle: dict[str, object],
    *,
    force_retrain: bool,
) -> dict[str, object]:
    cfg = bundle["cfg"]
    artifact_paths = bundle["artifact_paths"]
    target_dim = bundle["target_dim"]
    target_names = bundle["target_names"]
    stats = bundle["stats"]
    x_test = bundle["x_test"]
    y_test_scaled = bundle["y_test_scaled"]

    assert isinstance(cfg, YRDExperimentConfig)
    assert isinstance(artifact_paths, dict)
    assert isinstance(target_dim, int)
    assert isinstance(target_names, list)
    assert isinstance(stats, dict)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_test_scaled, dict)

    set_seed(cfg.seed)
    baseline_model = PersistenceBaseline(target_dim=target_dim, horizons=cfg.horizons)
    checkpoint_path = Path(artifact_paths["checkpoint"])
    loss_history_path = Path(artifact_paths["loss_history"])

    if checkpoint_path.exists() and loss_history_path.exists() and not force_retrain:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        if "model_kwargs" not in checkpoint_payload:
            raise RuntimeError(
                "Checkpoint is missing model_kwargs metadata. Set force_retrain=True to rebuild the cache."
            )
        joint_model = rebuild_joint_model_from_checkpoint(checkpoint_payload)
        loss_history_payload = load_json(loss_history_path)
    else:
        training_result = train_joint_model_with_history(
            n_stations=bundle["sample_bundle"]["n_stations"],
            n_features=bundle["sample_bundle"]["n_features"],
            history_hours=cfg.history_hours,
            target_dim=target_dim,
            hidden_dim=cfg.hidden_dim,
            horizons=cfg.horizons,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            max_epochs=cfg.max_epochs,
            early_stopping_patience=cfg.early_stopping_patience,
            seed=cfg.seed,
            x_train=bundle["x_train"],
            y_train=bundle["y_train_scaled"],
            x_val=bundle["x_val"],
            y_val=bundle["y_val_scaled"],
            model_name=cfg.model_name,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            norm_type=cfg.norm_type,
            activation=cfg.activation,
        )
        joint_model = training_result["model"]
        checkpoint_payload = {
            "state_dict": joint_model.state_dict(),
            "best_epoch": training_result["best_epoch"],
            "best_val_loss": training_result["best_val_loss"],
            "train_loss_history": training_result["train_loss_history"],
            "val_loss_history": training_result["val_loss_history"],
            "model_kwargs": training_result["model_kwargs"],
            "run_context": bundle["run_context"],
        }
        torch.save(checkpoint_payload, checkpoint_path)
        loss_history_payload = {
            "best_epoch": training_result["best_epoch"],
            "best_val_loss": training_result["best_val_loss"],
            "train_loss_history": training_result["train_loss_history"],
            "val_loss_history": training_result["val_loss_history"],
        }
        save_json(loss_history_path, _to_jsonable(loss_history_payload))

    joint_model.eval()
    baseline_scaled_predictions = _predict_numpy(baseline_model, x_test, cfg.horizons)
    joint_scaled_predictions = _predict_numpy(joint_model, x_test, cfg.horizons)
    y_test_original = {
        horizon: inverse_transform_targets(y_test_scaled[horizon], target_names, stats)
        for horizon in cfg.horizons
    }
    baseline_original_predictions = {
        horizon: inverse_transform_targets(baseline_scaled_predictions[horizon], target_names, stats)
        for horizon in cfg.horizons
    }
    joint_original_predictions = {
        horizon: inverse_transform_targets(joint_scaled_predictions[horizon], target_names, stats)
        for horizon in cfg.horizons
    }
    np.savez(
        artifact_paths["predictions"],
        **_prediction_payload(
            cfg=cfg,
            y_test_scaled=y_test_scaled,
            baseline_scaled_predictions=baseline_scaled_predictions,
            joint_scaled_predictions=joint_scaled_predictions,
            y_test_original=y_test_original,
            baseline_original_predictions=baseline_original_predictions,
            joint_original_predictions=joint_original_predictions,
        ),
    )
    metrics_payload = _build_metrics_payload(
        cfg=cfg,
        y_test_original=y_test_original,
        baseline_original_predictions=baseline_original_predictions,
        joint_original_predictions=joint_original_predictions,
    )
    save_json(Path(artifact_paths["metrics"]), _to_jsonable(metrics_payload))
    run_manifest = {
        "cache_dir": str(Path(artifact_paths["config"]).parent),
        "results_dir": str(Path(artifact_paths["run_manifest"]).parent),
        "config": str(artifact_paths["config"]),
        "checkpoint": str(checkpoint_path),
        "loss_history": str(loss_history_path),
        "predictions": str(artifact_paths["predictions"]),
        "metrics": str(artifact_paths["metrics"]),
    }
    save_json(Path(artifact_paths["run_manifest"]), run_manifest)
    return {
        "baseline_model": baseline_model,
        "joint_model": joint_model,
        "checkpoint_payload": checkpoint_payload,
        "loss_history_payload": loss_history_payload,
        "baseline_scaled_predictions": baseline_scaled_predictions,
        "joint_scaled_predictions": joint_scaled_predictions,
        "y_test_original": y_test_original,
        "baseline_original_predictions": baseline_original_predictions,
        "joint_original_predictions": joint_original_predictions,
        "metrics_payload": metrics_payload,
        "run_manifest": run_manifest,
    }


def summarize_air_search_station_pollutant_effects(
    *,
    sample_summaries: list[dict[str, object]],
    station_ids: list[str],
    pairwise_feature_name: str = "PM2.5",
) -> dict[str, object]:
    return {
        "conditional_synergy": summarize_global_station_pollutant_synergy(
            sample_summaries,
            station_ids=station_ids,
        ),
        "single_pollutant_pairwise": summarize_global_station_single_pollutant_ei(
            sample_summaries,
            station_ids=station_ids,
            feature_name=pairwise_feature_name,
        ),
    }


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def parse_float_csv(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


def ensure_air_tuning_state(root_dir: Path) -> dict[str, Path]:
    log_dir = Path(root_dir) / "docs" / "log" / "air_tuning"
    log_dir.mkdir(parents=True, exist_ok=True)
    return {
        "log_dir": log_dir,
        "run_history": log_dir / "run_history.jsonl",
        "coarse_leaderboard": log_dir / "coarse_leaderboard.json",
        "refine_results": log_dir / "refine_results.json",
        "report_manifest": log_dir / "report_manifest.json",
    }


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_jsonable(payload), ensure_ascii=False) + "\n")


def build_station_variable_index_map(target_names: list[str], variable: str) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    suffix = f"__{variable}"
    for index, name in enumerate(target_names):
        if not name.endswith(suffix):
            continue
        _, station_id, _ = name.split("__")
        mapping.setdefault(station_id, []).append(index)
    return mapping


def _subset_metric(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    suffix: str,
) -> dict[str, float]:
    indices = [index for index, name in enumerate(target_names) if name.endswith(f"__{suffix}")]
    if not indices:
        raise ValueError(f"Could not find targets ending with __{suffix}.")
    return _compute_metrics(y_true[:, indices], y_pred[:, indices])


def _nonself_edge_frame(edge_rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(edge_rows)
    if frame.empty or "source_station_id" not in frame.columns or "target_station_id" not in frame.columns:
        return frame
    filtered = frame[frame["source_station_id"] != frame["target_station_id"]].copy()
    return filtered if not filtered.empty else frame


def summarize_edge_distribution(edge_rows: list[dict[str, object]]) -> dict[str, float]:
    frame = _nonself_edge_frame(edge_rows)
    if frame.empty:
        return {"mean": 0.0, "positive_mean": 0.0, "negative_ratio": 0.0, "count": 0.0}
    values = frame["mean"].astype(float).to_numpy()
    positive_values = values[values > 0.0]
    return {
        "mean": float(values.mean()),
        "positive_mean": float(positive_values.mean()) if positive_values.size else 0.0,
        "negative_ratio": float(np.mean(values < 0.0)),
        "count": float(values.size),
    }


def build_coarse_row(
    *,
    city_en: str,
    horizon: int,
    o3_rmse: float,
    baseline_o3_rmse: float,
    syn_mean: float,
    syn_negative_ratio: float,
    pm25_to_o3_mean: float,
    pm25_negative_ratio: float,
) -> dict[str, object]:
    passes_accuracy_gate = float(o3_rmse) < float(baseline_o3_rmse)
    return {
        "city_en": str(city_en),
        "horizon": int(horizon),
        "o3_rmse": float(o3_rmse),
        "baseline_o3_rmse": float(baseline_o3_rmse),
        "passes_accuracy_gate": bool(passes_accuracy_gate),
        "primary_syn_mean": float(syn_mean),
        "primary_syn_abs_mean": abs(float(syn_mean)),
        "syn_negative_ratio": float(syn_negative_ratio),
        "pm25_to_o3_mean": float(pm25_to_o3_mean),
        "pm25_negative_ratio": float(pm25_negative_ratio),
    }


def choose_tm_gamma(
    rows: list[dict[str, object]],
    *,
    negative_ratio_threshold: float = DEFAULT_NEGATIVE_RATIO_THRESHOLD,
) -> dict[str, object]:
    if not rows:
        raise ValueError("rows must not be empty.")
    eligible = [
        row for row in rows
        if float(row.get("syn_negative_ratio", 1.0)) <= float(negative_ratio_threshold)
    ]
    candidates = eligible or rows
    return sorted(
        candidates,
        key=lambda row: (
            float(row.get("gamma", math.inf)),
            float(row.get("syn_negative_ratio", math.inf)),
            -abs(float(row.get("syn_mean", row.get("primary_syn_mean", 0.0)))),
        ),
    )[0]


def build_report_manifest(
    *,
    city_en: str,
    horizon: int,
    selected_refine_run: dict[str, object],
    graph_paths: dict[str, str],
) -> dict[str, object]:
    return {
        "city_en": str(city_en),
        "horizon": int(horizon),
        "selected_refine_run": _to_jsonable(selected_refine_run),
        "graphs": dict(graph_paths),
    }


def _horizon_label(horizon: int) -> str:
    return f"{int(horizon)}h"


def _coarse_run_tag(*, use_smoke: bool) -> str:
    return "coarse_smoke" if use_smoke else "coarse"


def _refine_run_tag(*, gamma: float, sample_count: int, seed: int, use_smoke: bool) -> str:
    gamma_label = str(f"{float(gamma):.2f}").replace(".", "p")
    prefix = "refine_smoke" if use_smoke else "refine"
    return f"{prefix}_tm_g{gamma_label}_m{int(sample_count)}_seed{int(seed)}"


def _sort_coarse_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            not bool(row["passes_accuracy_gate"]),
            -abs(float(row["primary_syn_mean"])),
            float(row["syn_negative_ratio"]),
            -float(row["pm25_to_o3_mean"]),
            str(row["city_en"]),
            int(row["horizon"]),
        ),
    )


def _station_groups(bundle: dict[str, object]) -> tuple[dict[str, list[int]], dict[str, dict[str, list[int]]]]:
    cfg = bundle["cfg"]
    sample_bundle = bundle["sample_bundle"]
    station_ids = bundle["station_ids"]
    assert isinstance(cfg, YRDExperimentConfig)
    assert isinstance(sample_bundle, dict)
    assert isinstance(station_ids, list)
    station_source_groups = build_one_step_station_source_groups(
        n_stations=sample_bundle["n_stations"],
        n_features=sample_bundle["n_features"],
        station_ids=station_ids,
    )
    station_pollutant_feature_groups = build_one_step_station_pollutant_feature_groups(
        n_stations=sample_bundle["n_stations"],
        n_features=sample_bundle["n_features"],
        pollutant_feature_indices={
            "O3": cfg.input_variables.index("O3"),
            "PM2.5": cfg.input_variables.index("PM2.5"),
        },
        station_ids=station_ids,
    )
    return station_source_groups, station_pollutant_feature_groups


def _o3_target_indices(bundle: dict[str, object], *, target_variable: str = "O3") -> dict[str, list[int]]:
    target_names = bundle["target_names"]
    assert isinstance(target_names, list)
    return build_station_variable_index_map(target_names, target_variable)


def compute_air_search_nis_summary(
    bundle: dict[str, object],
    predictions: dict[str, object],
    *,
    coupling_sample_count: int,
    sampling_seed: int,
    target_variable: str = "O3",
) -> dict[str, object]:
    cfg = bundle["cfg"]
    x_train = bundle["x_train"]
    station_ids = bundle["station_ids"]
    assert isinstance(cfg, YRDExperimentConfig)
    assert isinstance(x_train, np.ndarray)
    assert isinstance(station_ids, list)
    horizon = _single_horizon(cfg)
    center = compute_training_input_center(x_train)
    synthetic_inputs = sample_uniform_box_inputs(
        center=center,
        box_size=float(cfg.box_size),
        sample_count=int(coupling_sample_count),
        seed=int(sampling_seed),
    )
    station_source_groups, station_pollutant_feature_groups = _station_groups(bundle)
    target_indices_by_station = _o3_target_indices(bundle, target_variable=target_variable)
    sigma_eps = estimate_residual_covariance(
        bundle["y_test_scaled"][horizon],
        predictions["joint_scaled_predictions"][horizon],
    )
    joint_model = predictions["joint_model"]
    joint_model.eval()

    sample_summaries: list[dict[str, object]] = []
    for sample_id, synthetic_input in enumerate(synthetic_inputs):
        sample_x = torch.from_numpy(synthetic_input[None, ...]).to(dtype=torch.float32)
        flat_sample = sample_x.reshape(-1).detach().clone().requires_grad_(True)

        def horizon_model(tensor: torch.Tensor) -> torch.Tensor:
            shaped = tensor.reshape(1, bundle["sample_bundle"]["n_stations"], bundle["sample_bundle"]["n_features"])
            return joint_model(shaped)[horizon].reshape(-1)

        for target_station_id, target_indices in target_indices_by_station.items():
            jacobian = jacobian_for_target_subset(
                horizon_model,
                flat_sample,
                target_indices=target_indices,
            ).detach().cpu().numpy()
            sample_summaries.append(
                {
                    "sample_id": int(sample_id),
                    "target_station_id": target_station_id,
                    **compute_station_level_nis_summary(
                        jacobian=jacobian,
                        sigma_eps=sigma_eps,
                        station_source_groups=station_source_groups,
                        target_indices=target_indices,
                        box_size=float(cfg.box_size),
                    ),
                    **compute_station_pollutant_pair_synergy_summary(
                        jacobian=jacobian,
                        sigma_eps=sigma_eps,
                        station_pollutant_feature_groups=station_pollutant_feature_groups,
                        target_indices=target_indices,
                        method="nis",
                        box_size=float(cfg.box_size),
                    ),
                }
            )

    return {
        "method": "nis",
        "horizon": int(horizon),
        "coupling_sample_count": int(coupling_sample_count),
        "sampling_seed": int(sampling_seed),
        "sample_summaries": sample_summaries,
        "o3_pairwise": summarize_global_station_coupling(sample_summaries, station_ids=station_ids),
        **summarize_air_search_station_pollutant_effects(
            sample_summaries=sample_summaries,
            station_ids=station_ids,
            pairwise_feature_name="PM2.5",
        ),
    }


def compute_air_search_tm_summary(
    bundle: dict[str, object],
    predictions: dict[str, object],
    *,
    sample_count: int,
    sampling_seed: int,
    gamma: float,
    target_variable: str = "O3",
    nonnegative_variables: tuple[str, ...] = DEFAULT_TM_NONNEGATIVE_VARIABLES,
    box_mode: str = "per_variable",
    global_box_size_override: float | None = None,
) -> dict[str, object]:
    cfg = bundle["cfg"]
    station_ids = bundle["station_ids"]
    assert isinstance(cfg, YRDExperimentConfig)
    assert isinstance(station_ids, list)
    horizon = _single_horizon(cfg)
    profile = estimate_support_cover_box_profile(
        x_train=bundle["x_train"],
        input_variables=cfg.input_variables,
        gamma=float(gamma),
        stats=bundle["stats"],
        nonnegative_variables=tuple(nonnegative_variables),
    )
    if box_mode == "global_max":
        profile = collapse_support_cover_box_profile_to_global_max(
            profile,
            global_box_size_override=global_box_size_override,
        )
    elif box_mode != "per_variable":
        raise ValueError(f"Unsupported box_mode={box_mode!r}.")
    synthetic_inputs = sample_uniform_box_inputs(
        center=np.asarray(profile["center"], dtype=np.float32),
        box_size=np.asarray(profile["box_size_by_feature"], dtype=np.float32),
        sample_count=int(sample_count),
        seed=int(sampling_seed),
        lower_bounds=None if profile["lower_bounds"] is None else np.asarray(profile["lower_bounds"], dtype=np.float32),
    )
    flat_source_samples = synthetic_inputs.reshape(synthetic_inputs.shape[0], -1)
    joint_model = predictions["joint_model"]
    joint_model.eval()
    with torch.no_grad():
        predicted_next = joint_model(torch.from_numpy(synthetic_inputs).to(dtype=torch.float32))[horizon].detach().cpu().numpy()
    station_source_groups, station_pollutant_feature_groups = _station_groups(bundle)
    target_indices_by_station = _o3_target_indices(bundle, target_variable=target_variable)
    sample_summaries: list[dict[str, object]] = []
    for target_station_id, target_indices in target_indices_by_station.items():
        target_samples = predicted_next[:, target_indices]
        sample_summaries.append(
            {
                "target_station_id": target_station_id,
                **compute_station_level_ei_summary(
                    method="tm",
                    station_source_groups=station_source_groups,
                    source_samples=flat_source_samples,
                    target_samples=target_samples,
                ),
                **compute_station_pollutant_pair_synergy_summary(
                    method="tm",
                    station_pollutant_feature_groups=station_pollutant_feature_groups,
                    source_samples=flat_source_samples,
                    target_samples=target_samples,
                ),
            }
        )
    return {
        "method": "tm",
        "horizon": int(horizon),
        "gamma": float(gamma),
        "sample_count": int(sample_count),
        "sampling_seed": int(sampling_seed),
        "box_mode": str(profile.get("box_mode", box_mode)),
        "profile": profile,
        "sample_summaries": sample_summaries,
        "o3_pairwise": summarize_global_station_coupling(sample_summaries, station_ids=station_ids),
        **summarize_air_search_station_pollutant_effects(
            sample_summaries=sample_summaries,
            station_ids=station_ids,
            pairwise_feature_name="PM2.5",
        ),
    }


def _graph_edge_frame(summary: dict[str, object], key: str) -> pd.DataFrame:
    return pd.DataFrame(summary.get(key, []))


def export_air_search_graphs(
    *,
    bundle: dict[str, object],
    summary: dict[str, object],
) -> dict[str, str]:
    artifact_paths = bundle["artifact_paths"]
    station_positions = bundle["city_metadata"][["station_id", "lon", "lat"]]
    station_ids = bundle["station_ids"]
    horizon_label = _horizon_label(_single_horizon(bundle["cfg"]))

    o3_pairwise_df = _graph_edge_frame(summary["o3_pairwise"], "pairwise_edges")
    pm25_pairwise_df = _graph_edge_frame(summary["single_pollutant_pairwise"], "pairwise_edges")
    synergy_df = _graph_edge_frame(summary["conditional_synergy"], "conditional_synergy_edges")
    o3_display = o3_pairwise_df[
        (o3_pairwise_df["source_station_id"] != o3_pairwise_df["target_station_id"])
        & (o3_pairwise_df["mean"] > 0.0)
    ].copy() if not o3_pairwise_df.empty else pd.DataFrame()
    draw_station_causal_graph(
        station_positions=station_positions,
        pairwise_edges=o3_display,
        horizon_label=horizon_label,
        out_path=artifact_paths["o3_pairwise_graph"],
        title=f"{bundle['run_context']['city_en'].title()} O3 -> O3 pairwise graph ({horizon_label})",
        strength_col="mean",
        node_self_strengths=build_self_loop_node_strengths(o3_pairwise_df, station_ids=station_ids),
    )
    pm25_display = pm25_pairwise_df[
        (pm25_pairwise_df["source_station_id"] != pm25_pairwise_df["target_station_id"])
        & (pm25_pairwise_df["mean"] > 0.0)
    ].copy() if not pm25_pairwise_df.empty else pd.DataFrame()
    draw_station_causal_graph(
        station_positions=station_positions,
        pairwise_edges=pm25_display,
        horizon_label=horizon_label,
        out_path=artifact_paths["pm25_to_o3_pairwise_graph"],
        title=f"{bundle['run_context']['city_en'].title()} PM2.5 -> O3 pairwise graph ({horizon_label})",
        strength_col="mean",
        node_self_strengths=build_self_loop_node_strengths(pm25_pairwise_df, station_ids=station_ids),
        legend_label="PM2.5 -> O3 edge",
    )
    synergy_display = synergy_df[synergy_df["source_station_id"] != synergy_df["target_station_id"]].copy() if not synergy_df.empty else pd.DataFrame()
    if not synergy_display.empty:
        synergy_display["abs_mean"] = synergy_display["mean"].abs()
    draw_station_causal_graph(
        station_positions=station_positions,
        pairwise_edges=synergy_display,
        horizon_label=horizon_label,
        out_path=artifact_paths["o3_pm25_synergy_graph"],
        title=f"{bundle['run_context']['city_en'].title()} O3+PM2.5 -> O3 synergy graph ({horizon_label})",
        strength_col="abs_mean",
        positive_color="#2F7D63",
        negative_color="#B04A5A",
        legend_label="Synergy edge",
        node_self_strengths=build_self_loop_node_strengths(synergy_df, station_ids=station_ids),
        node_colorbar_label="Self Syn",
    )
    return {
        "o3_pairwise": str(artifact_paths["o3_pairwise_graph"]),
        "pm25_to_o3_pairwise": str(artifact_paths["pm25_to_o3_pairwise_graph"]),
        "o3_pm25_synergy": str(artifact_paths["o3_pm25_synergy_graph"]),
    }


def _coarse_result_row(
    *,
    city_en: str,
    horizon: int,
    bundle: dict[str, object],
    predictions: dict[str, object],
    coarse_summary: dict[str, object],
) -> dict[str, object]:
    o3_metrics = _subset_metric(
        y_true=predictions["y_test_original"][horizon],
        y_pred=predictions["joint_original_predictions"][horizon],
        target_names=bundle["target_names"],
        suffix="O3",
    )
    baseline_o3_metrics = _subset_metric(
        y_true=predictions["y_test_original"][horizon],
        y_pred=predictions["baseline_original_predictions"][horizon],
        target_names=bundle["target_names"],
        suffix="O3",
    )
    syn_stats = summarize_edge_distribution(coarse_summary["conditional_synergy"]["conditional_synergy_edges"])
    pm25_stats = summarize_edge_distribution(coarse_summary["single_pollutant_pairwise"]["pairwise_edges"])
    row = build_coarse_row(
        city_en=city_en,
        horizon=horizon,
        o3_rmse=o3_metrics["rmse"],
        baseline_o3_rmse=baseline_o3_metrics["rmse"],
        syn_mean=syn_stats["mean"],
        syn_negative_ratio=syn_stats["negative_ratio"],
        pm25_to_o3_mean=pm25_stats["mean"],
        pm25_negative_ratio=pm25_stats["negative_ratio"],
    )
    row.update(
        {
            "stage": "coarse",
            "run_tag": str(bundle["run_context"]["run_tag"]),
            "artifact_dir": str(bundle["artifact_paths"]["results_dir"]),
        }
    )
    return row


def run_coarse_stage(
    *,
    root_dir: Path,
    cities: list[str],
    horizons: list[int],
    smoke: bool,
    force_retrain: bool = False,
    force_recompute_coupling: bool = False,
    coupling_sample_count: int | None = None,
) -> dict[str, object]:
    state_paths = ensure_air_tuning_state(root_dir)
    effective_sample_count = (
        int(coupling_sample_count)
        if coupling_sample_count is not None
        else (DEFAULT_COARSE_SAMPLE_COUNT_SMOKE if smoke else DEFAULT_COARSE_SAMPLE_COUNT)
    )
    rows: list[dict[str, object]] = []
    for city_en in cities:
        for horizon in horizons:
            cfg = build_air_search_config(root_dir, city_en=city_en, horizon=horizon, test_mode=smoke)
            bundle = prepare_air_search_bundle(
                cfg=cfg,
                city_en=city_en,
                run_tag=_coarse_run_tag(use_smoke=smoke),
                use_smoke=smoke,
            )
            predictions = run_or_load_air_search_predictions(bundle, force_retrain=force_retrain)
            coarse_summary_path = Path(bundle["artifact_paths"]["coarse_summary"])
            if coarse_summary_path.exists() and not force_recompute_coupling:
                coarse_summary = load_json(coarse_summary_path)
            else:
                coarse_summary = compute_air_search_nis_summary(
                    bundle,
                    predictions,
                    coupling_sample_count=effective_sample_count,
                    sampling_seed=cfg.seed,
                )
                save_json(coarse_summary_path, _to_jsonable(coarse_summary))
            row = _coarse_result_row(
                city_en=city_en,
                horizon=horizon,
                bundle=bundle,
                predictions=predictions,
                coarse_summary=coarse_summary,
            )
            save_json(Path(bundle["artifact_paths"]["leaderboard_row"]), _to_jsonable(row))
            append_jsonl(Path(state_paths["run_history"]), row)
            rows.append(row)
    leaderboard = {"rows": _sort_coarse_rows(rows)}
    save_json(Path(state_paths["coarse_leaderboard"]), _to_jsonable(leaderboard))
    return leaderboard


def _load_leaderboard_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = load_json(path)
    rows = payload.get("rows", [])
    return list(rows) if isinstance(rows, list) else []


def _merge_unique_rows(
    existing_rows: list[dict[str, object]],
    new_rows: list[dict[str, object]],
    *,
    key_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    merged: dict[tuple[object, ...], dict[str, object]] = {}
    for row in existing_rows + new_rows:
        key = tuple(row.get(field) for field in key_fields)
        merged[key] = row
    return list(merged.values())


def _group_refine_rows_by_gamma(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[float, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(float(row["gamma"]), []).append(row)
    aggregated: list[dict[str, object]] = []
    for gamma, gamma_rows in grouped.items():
        aggregated.append(
            {
                "gamma": float(gamma),
                "syn_negative_ratio": float(np.mean([float(row["syn_negative_ratio"]) for row in gamma_rows])),
                "syn_mean": float(np.mean([float(row["syn_mean"]) for row in gamma_rows])),
            }
        )
    return sorted(aggregated, key=lambda row: float(row["gamma"]))


def run_refine_stage(
    *,
    root_dir: Path,
    cities: list[str],
    horizons: list[int],
    smoke: bool,
    top_k: int,
    force_retrain: bool = False,
    force_recompute_coupling: bool = False,
    tm_sample_counts: list[int] | None = None,
    tm_seeds: list[int] | None = None,
    tm_gammas: list[float] | None = None,
) -> dict[str, object]:
    state_paths = ensure_air_tuning_state(root_dir)
    existing_payload = (
        load_json(Path(state_paths["refine_results"]))
        if Path(state_paths["refine_results"]).exists()
        else {"rows": [], "reports": []}
    )
    shortlist = [
        row for row in _load_leaderboard_rows(Path(state_paths["coarse_leaderboard"]))
        if row.get("city_en") in cities and int(row.get("horizon", -1)) in horizons
    ]
    shortlist = _sort_coarse_rows(shortlist)[: max(1, int(top_k))]
    tm_sample_counts = tm_sample_counts or ([64] if smoke else [512])
    tm_seeds = tm_seeds or [0]
    tm_gammas = tm_gammas or [1.0, 1.1, 1.2]
    refine_rows: list[dict[str, object]] = []
    report_manifests: list[dict[str, object]] = []

    for coarse_row in shortlist:
        city_en = str(coarse_row["city_en"])
        horizon = int(coarse_row["horizon"])
        cfg = build_air_search_config(root_dir, city_en=city_en, horizon=horizon, test_mode=smoke)
        candidate_records: list[dict[str, object]] = []
        for gamma in tm_gammas:
            for sample_count in tm_sample_counts:
                for seed in tm_seeds:
                    bundle = prepare_air_search_bundle(
                        cfg=cfg,
                        city_en=city_en,
                        run_tag=_refine_run_tag(
                            gamma=float(gamma),
                            sample_count=int(sample_count),
                            seed=int(seed),
                            use_smoke=smoke,
                        ),
                        use_smoke=smoke,
                    )
                    predictions = run_or_load_air_search_predictions(bundle, force_retrain=force_retrain)
                    refine_summary_path = Path(bundle["artifact_paths"]["refine_summary"])
                    if refine_summary_path.exists() and not force_recompute_coupling:
                        refine_summary = load_json(refine_summary_path)
                    else:
                        refine_summary = compute_air_search_tm_summary(
                            bundle,
                            predictions,
                            sample_count=int(sample_count),
                            sampling_seed=int(seed),
                            gamma=float(gamma),
                        )
                        save_json(refine_summary_path, _to_jsonable(refine_summary))
                    syn_stats = summarize_edge_distribution(
                        refine_summary["conditional_synergy"]["conditional_synergy_edges"]
                    )
                    pm25_stats = summarize_edge_distribution(
                        refine_summary["single_pollutant_pairwise"]["pairwise_edges"]
                    )
                    record = {
                        "stage": "refine",
                        "city_en": city_en,
                        "horizon": int(horizon),
                        "gamma": float(gamma),
                        "sample_count": int(sample_count),
                        "seed": int(seed),
                        "syn_mean": float(syn_stats["mean"]),
                        "syn_negative_ratio": float(syn_stats["negative_ratio"]),
                        "pm25_to_o3_mean": float(pm25_stats["mean"]),
                        "pm25_negative_ratio": float(pm25_stats["negative_ratio"]),
                        "run_tag": str(bundle["run_context"]["run_tag"]),
                        "artifact_dir": str(bundle["artifact_paths"]["results_dir"]),
                        "summary": refine_summary,
                    }
                    append_jsonl(
                        Path(state_paths["run_history"]),
                        {key: value for key, value in record.items() if key != "summary"},
                    )
                    candidate_records.append(record)
                    refine_rows.append({key: value for key, value in record.items() if key != "summary"})

        gamma_rows = _group_refine_rows_by_gamma(candidate_records)
        winner = choose_tm_gamma(gamma_rows)
        chosen_candidates = [
            row for row in candidate_records
            if float(row["gamma"]) == float(winner["gamma"])
        ]
        selected = sorted(
            chosen_candidates,
            key=lambda row: (-int(row["sample_count"]), int(row["seed"])),
        )[0]
        graph_paths = export_air_search_graphs(
            bundle=prepare_air_search_bundle(
                cfg=cfg,
                city_en=city_en,
                run_tag=str(selected["run_tag"]),
                use_smoke=smoke,
            ),
            summary=selected["summary"],
        )
        manifest = build_report_manifest(
            city_en=city_en,
            horizon=horizon,
            selected_refine_run={key: value for key, value in selected.items() if key != "summary"},
            graph_paths=graph_paths,
        )
        manifest_path = build_air_search_artifact_paths(
            root_dir=root_dir,
            city_en=city_en,
            horizon=horizon,
            run_tag=str(selected["run_tag"]),
            use_smoke=smoke,
        )["report_manifest"]
        save_json(Path(manifest_path), _to_jsonable(manifest))
        report_manifests.append(manifest)

    payload = {
        "rows": _merge_unique_rows(
            list(existing_payload.get("rows", [])),
            refine_rows,
            key_fields=("city_en", "horizon", "gamma", "sample_count", "seed", "run_tag"),
        ),
        "reports": _merge_unique_rows(
            list(existing_payload.get("reports", [])),
            report_manifests,
            key_fields=("city_en", "horizon"),
        ),
    }
    save_json(Path(state_paths["refine_results"]), _to_jsonable(payload))
    save_json(Path(state_paths["report_manifest"]), _to_jsonable({"reports": payload["reports"]}))
    return payload


def run_report_stage(
    *,
    root_dir: Path,
) -> dict[str, object]:
    state_paths = ensure_air_tuning_state(root_dir)
    refine_payload = load_json(Path(state_paths["refine_results"])) if Path(state_paths["refine_results"]).exists() else {"rows": [], "reports": []}
    reports = list(refine_payload.get("reports", []))
    payload = {"reports": reports}
    save_json(Path(state_paths["report_manifest"]), _to_jsonable(payload))
    return payload


# --- Former yrd/air_search_notebook.py ---

import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt



def build_tm_run_tag(
    *,
    gamma: float,
    sample_count: int,
    seed: int,
    use_smoke: bool,
    box_mode: str = "per_variable",
    global_box_size_override: float | None = None,
) -> str:
    gamma_label = str(f"{float(gamma):.2f}").replace(".", "p")
    prefix = "refine_smoke" if use_smoke else "refine"
    run_tag = f"{prefix}_tm_g{gamma_label}_m{int(sample_count)}_seed{int(seed)}"
    if box_mode == "per_variable":
        return run_tag
    if box_mode == "global_max":
        if global_box_size_override is None:
            return f"{run_tag}_lmax"
        override_label = f"{float(global_box_size_override):.4f}".replace(".", "p")
        return f"{run_tag}_l{override_label}"
    raise ValueError(f"Unsupported box_mode={box_mode!r}.")


def _to_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False) + "\n")


def filter_station_edge_frame(
    frame: pd.DataFrame,
    *,
    top_k_edges: int,
    min_abs_strength: float,
    include_self_loops: bool,
    positive_only: bool,
    sort_by_abs: bool,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    filtered = frame.copy()
    if not include_self_loops:
        filtered = filtered[filtered["source_station_id"] != filtered["target_station_id"]].copy()
    filtered["abs_mean"] = filtered["mean"].astype(float).abs()
    filtered = filtered[filtered["abs_mean"] >= float(min_abs_strength)].copy()
    if positive_only:
        filtered = filtered[filtered["mean"].astype(float) > 0.0].copy()
    if filtered.empty:
        return filtered.reset_index(drop=True)
    sort_col = "abs_mean" if sort_by_abs else "mean"
    return (
        filtered.sort_values(sort_col, ascending=False)
        .head(max(1, int(top_k_edges)))
        .reset_index(drop=True)
    )


def build_support_cover_profile_table(profile: dict[str, object]) -> pd.DataFrame:
    variables = list(profile.get("input_variables", []))
    rows: list[dict[str, object]] = []
    lower_bound_by_variable = dict(profile.get("lower_bound_by_variable", {}))
    nonnegative_variables = set(profile.get("nonnegative_variables", []))
    for variable in variables:
        rows.append(
            {
                "variable": variable,
                "center": float(dict(profile.get("center_by_variable", {})).get(variable, 0.0)),
                "train_min": float(dict(profile.get("train_min_by_variable", {})).get(variable, 0.0)),
                "train_max": float(dict(profile.get("train_max_by_variable", {})).get(variable, 0.0)),
                "cover_radius": float(dict(profile.get("cover_radius_by_variable", {})).get(variable, 0.0)),
                "box_size_Lv": float(dict(profile.get("box_size_by_variable", {})).get(variable, 0.0)),
                "lower_bound": (
                    float(lower_bound_by_variable[variable])
                    if variable in lower_bound_by_variable
                    else None
                ),
                "nonnegative_clipped": bool(variable in nonnegative_variables),
            }
        )
    return pd.DataFrame(rows)


def _horizon_label(horizon: int) -> str:
    return f"{int(horizon)}h"


def _metric_summary_table(
    *,
    bundle: dict[str, object],
    predictions: dict[str, object],
    tm_summary: dict[str, object],
) -> pd.DataFrame:
    horizon = int(bundle["cfg"].horizons[0])
    metrics_frame = pd.DataFrame(
        metric_rows_for_scope(
            predictions["y_test_original"],
            {
                "joint_model": predictions["joint_original_predictions"],
                "persistence": predictions["baseline_original_predictions"],
            },
            target_names=bundle["target_names"],
        )
    )
    metrics_scope = metrics_frame[
        (metrics_frame["horizon"] == _horizon_label(horizon))
        & (metrics_frame["scope"].isin(["overall", "O3", "PM2.5"]))
    ].copy()
    syn_stats = summarize_edge_distribution(tm_summary["conditional_synergy"]["conditional_synergy_edges"])
    pm25_stats = summarize_edge_distribution(tm_summary["single_pollutant_pairwise"]["pairwise_edges"])
    extra_rows = pd.DataFrame(
        [
            {"metric": "Syn mean", "value": float(syn_stats["mean"])},
            {"metric": "Syn negative ratio", "value": float(syn_stats["negative_ratio"])},
            {"metric": "PM2.5 -> O3 mean", "value": float(pm25_stats["mean"])},
            {"metric": "PM2.5 -> O3 negative ratio", "value": float(pm25_stats["negative_ratio"])},
        ]
    )
    if metrics_scope.empty:
        return extra_rows
    metric_rows = []
    for _, row in metrics_scope.iterrows():
        metric_rows.append(
            {
                "metric": f"{row['model']} {row['scope']} RMSE",
                "value": float(row["rmse"]),
            }
        )
        metric_rows.append(
            {
                "metric": f"{row['model']} {row['scope']} Corr",
                "value": float(row["corr"]),
            }
        )
    return pd.concat([pd.DataFrame(metric_rows), extra_rows], ignore_index=True)


def _build_notebook_graph_paths(
    *,
    root_dir: Path,
    city_en: str,
    horizon: int,
    run_tag: str,
    use_smoke: bool,
    top_k_edges: int,
    min_abs_strength: float,
    show_negative_synergy_edges: bool,
) -> dict[str, Path]:
    base_paths = build_air_search_artifact_paths(
        root_dir=root_dir,
        city_en=city_en,
        horizon=horizon,
        run_tag=run_tag,
        use_smoke=use_smoke,
    )
    view_tag = (
        f"notebook_top{int(top_k_edges)}_"
        f"min{float(min_abs_strength):.4f}".replace(".", "p")
        + f"_neg{int(bool(show_negative_synergy_edges))}"
    )
    results_dir = Path(base_paths["results_dir"]) / view_tag
    results_dir.mkdir(parents=True, exist_ok=True)
    return {
        "results_dir": results_dir,
        "o3_pairwise": results_dir / "o3_pairwise_graph.png",
        "pm25_to_o3_pairwise": results_dir / "pm25_to_o3_pairwise_graph.png",
        "o3_pm25_synergy": results_dir / "o3_pm25_synergy_graph.png",
        "combined_panel": results_dir / "combined_tm_graph_panel_labeled.png",
    }


def _draw_combined_panel_edges(
    ax: plt.Axes,
    *,
    position_map: dict[str, tuple[float, float]],
    edges: pd.DataFrame,
    color: str,
    negative_color: str | None = None,
    strength_col: str = "mean",
) -> None:
    if edges.empty:
        return
    render_edges = edges.copy()
    if strength_col in render_edges.columns:
        render_edges = render_edges[render_edges[strength_col] > 0].copy()
    if render_edges.empty:
        return
    max_edge = max(float(render_edges[strength_col].max()), 1e-6)
    for _, row in render_edges.iterrows():
        x0, y0 = position_map[row["source_station_id"]]
        x1, y1 = position_map[row["target_station_id"]]
        edge_color = color if negative_color is None or float(row["mean"]) >= 0 else negative_color
        width, alpha = style_pairwise_edge(float(row[strength_col]) / max_edge, alpha_min=0.08, alpha_max=0.82)
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": edge_color,
                "linewidth": width,
                "alpha": alpha,
                "connectionstyle": f"arc3,rad={0.12 if x0 <= x1 else -0.12}",
                "shrinkA": 10,
                "shrinkB": 8,
                "mutation_scale": 16,
            },
            zorder=2,
        )


def save_combined_tm_graph_panel(
    results: dict[str, object],
    out_path: Path | str | None = None,
    *,
    panel_labels: tuple[str, str, str] | list[str] = ("(a)", "(b)", "(c)"),
    panel_label_fontsize: float = 15.0,
    panel_label_x: float = 0.5,
    panel_label_y: float = -0.17,
    panel_label_fontweight: str = "bold",
) -> Path:
    station_positions = results["station_positions_df"]
    station_ids = list(results["station_ids"])
    graph_paths = {key: Path(value) for key, value in results["graph_paths"].items()}
    output_path = Path(out_path) if out_path is not None else graph_paths["combined_panel"]
    position_map = {row["station_id"]: (float(row["lon"]), float(row["lat"])) for _, row in station_positions.iterrows()}

    panels = [
        ("O3 -> O3", results["o3_pairwise_display_df"], results["o3_pairwise_df"], "mean", "#345995", None),
        ("PM2.5 -> O3", results["pm25_to_o3_display_df"], results["pm25_to_o3_df"], "mean", "#345995", None),
        ("O3+PM2.5 -> O3", results["synergy_display_df"], results["synergy_df"], "abs_mean", "#2F7D63", "#B04A5A"),
    ]
    node_strengths = [build_self_loop_node_strengths(frame, station_ids=station_ids) for _, _, frame, _, _, _ in panels]
    node_values = np.asarray([float(strengths.get(station_id, 0.0)) for strengths in node_strengths for station_id in station_ids])
    max_abs_node = max(float(np.max(np.abs(node_values))) if len(node_values) else 0.0, 1e-9)
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs_node, vcenter=0.0, vmax=max_abs_node)

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.6), sharex=True, sharey=True, constrained_layout=True)
    panel_tags = list(panel_labels)
    for ax, tag, (label, edges, _, strength_col, color, negative_color), strengths in zip(axes, panel_tags, panels, node_strengths):
        node_colors = matplotlib.colormaps["RdBu_r"]([norm(float(strengths.get(station_id, 0.0))) for station_id in station_ids])
        ax.scatter(station_positions["lon"], station_positions["lat"], color=node_colors, s=74, edgecolors="#4C566A", linewidths=0.6, zorder=5)
        for _, row in station_positions.iterrows():
            ax.text(row["lon"] + 0.003, row["lat"] + 0.002, row["station_id"], fontsize=6.7, color="#233142", zorder=6)
        _draw_combined_panel_edges(ax, position_map=position_map, edges=edges, color=color, negative_color=negative_color, strength_col=strength_col)
        ax.text(
            float(panel_label_x),
            float(panel_label_y),
            tag,
            transform=ax.transAxes,
            fontsize=float(panel_label_fontsize),
            fontweight=panel_label_fontweight,
            va="top",
            ha="center",
            clip_on=False,
        )
        ax.set_title(label, fontsize=10.5)
        ax.set_xlabel("Longitude")
        ax.grid(True, alpha=0.16, linewidth=0.6)
    axes[0].set_ylabel("Latitude")
    colorbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=axes, fraction=0.025, pad=0.015)
    colorbar.set_label("Self-loop strength")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_graphs(
    *,
    bundle: dict[str, object],
    o3_pairwise_display_df: pd.DataFrame,
    pm25_to_o3_display_df: pd.DataFrame,
    synergy_display_df: pd.DataFrame,
    graph_paths: dict[str, Path],
) -> None:
    station_positions = bundle["city_metadata"][["station_id", "lon", "lat"]]
    station_ids = bundle["station_ids"]
    horizon_label = _horizon_label(int(bundle["cfg"].horizons[0]))

    o3_pairwise_df = pd.DataFrame(bundle["tm_summary"]["o3_pairwise"]["pairwise_edges"])
    pm25_pairwise_df = pd.DataFrame(bundle["tm_summary"]["single_pollutant_pairwise"]["pairwise_edges"])
    synergy_df = pd.DataFrame(bundle["tm_summary"]["conditional_synergy"]["conditional_synergy_edges"])

    draw_station_causal_graph(
        station_positions=station_positions,
        pairwise_edges=o3_pairwise_display_df,
        horizon_label=horizon_label,
        out_path=graph_paths["o3_pairwise"],
        title=f"{bundle['run_context']['city_en'].title()} O3 -> O3 pairwise graph ({horizon_label})",
        strength_col="mean",
        node_self_strengths=build_self_loop_node_strengths(o3_pairwise_df, station_ids=station_ids),
        show_title=False,
        show_edge_legend=False,
    )
    draw_station_causal_graph(
        station_positions=station_positions,
        pairwise_edges=pm25_to_o3_display_df,
        horizon_label=horizon_label,
        out_path=graph_paths["pm25_to_o3_pairwise"],
        title=f"{bundle['run_context']['city_en'].title()} PM2.5 -> O3 pairwise graph ({horizon_label})",
        strength_col="mean",
        node_self_strengths=build_self_loop_node_strengths(pm25_pairwise_df, station_ids=station_ids),
        legend_label="PM2.5 -> O3 edge",
        show_title=False,
        show_edge_legend=False,
    )
    synergy_render_df = synergy_display_df.copy()
    if not synergy_render_df.empty:
        synergy_render_df["abs_mean"] = synergy_render_df["mean"].astype(float).abs()
    draw_station_causal_graph(
        station_positions=station_positions,
        pairwise_edges=synergy_render_df,
        horizon_label=horizon_label,
        out_path=graph_paths["o3_pm25_synergy"],
        title=f"{bundle['run_context']['city_en'].title()} O3+PM2.5 -> O3 synergy graph ({horizon_label})",
        strength_col="abs_mean",
        positive_color="#2F7D63",
        negative_color="#B04A5A",
        legend_label="Synergy edge",
        node_self_strengths=build_self_loop_node_strengths(synergy_df, station_ids=station_ids),
        node_colorbar_label="Self Syn",
        show_title=False,
        show_edge_legend=False,
    )


def run_air_tm_notebook_case(
    *,
    root_dir: Path,
    city_en: str,
    horizon: int,
    tm_sample_count: int,
    sampling_seed: int,
    gamma: float,
    top_k_edges: int,
    min_abs_strength: float,
    show_negative_synergy_edges: bool,
    force_retrain: bool,
    force_recompute_tm: bool,
    use_smoke: bool,
    box_mode: str = "per_variable",
    global_box_size_override: float | None = None,
) -> dict[str, object]:
    resolved_root = find_project_root(Path(root_dir))
    cfg = build_air_search_config(
        resolved_root,
        city_en=city_en,
        horizon=int(horizon),
        test_mode=use_smoke,
    )
    run_tag = build_tm_run_tag(
        gamma=float(gamma),
        sample_count=int(tm_sample_count),
        seed=int(sampling_seed),
        use_smoke=use_smoke,
        box_mode=box_mode,
        global_box_size_override=global_box_size_override,
    )
    bundle = prepare_air_search_bundle(
        cfg=cfg,
        city_en=city_en,
        run_tag=run_tag,
        use_smoke=use_smoke,
    )
    predictions = run_or_load_air_search_predictions(bundle, force_retrain=force_retrain)
    refine_summary_path = Path(bundle["artifact_paths"]["refine_summary"])
    used_cached_results = refine_summary_path.exists() and not force_recompute_tm
    if used_cached_results:
        tm_summary = load_json(refine_summary_path)
    else:
        tm_summary = compute_air_search_tm_summary(
            bundle,
            predictions,
            sample_count=int(tm_sample_count),
            sampling_seed=int(sampling_seed),
            gamma=float(gamma),
            box_mode=box_mode,
            global_box_size_override=global_box_size_override,
        )
        _save_json(refine_summary_path, tm_summary)
    bundle["tm_summary"] = tm_summary

    o3_pairwise_df = pd.DataFrame(tm_summary["o3_pairwise"]["pairwise_edges"])
    pm25_to_o3_df = pd.DataFrame(tm_summary["single_pollutant_pairwise"]["pairwise_edges"])
    synergy_df = pd.DataFrame(tm_summary["conditional_synergy"]["conditional_synergy_edges"])

    o3_pairwise_display_df = filter_station_edge_frame(
        o3_pairwise_df,
        top_k_edges=int(top_k_edges),
        min_abs_strength=float(min_abs_strength),
        include_self_loops=False,
        positive_only=True,
        sort_by_abs=False,
    )
    pm25_to_o3_display_df = filter_station_edge_frame(
        pm25_to_o3_df,
        top_k_edges=int(top_k_edges),
        min_abs_strength=float(min_abs_strength),
        include_self_loops=False,
        positive_only=True,
        sort_by_abs=False,
    )
    synergy_display_df = filter_station_edge_frame(
        synergy_df,
        top_k_edges=int(top_k_edges),
        min_abs_strength=float(min_abs_strength),
        include_self_loops=False,
        positive_only=not bool(show_negative_synergy_edges),
        sort_by_abs=True,
    )
    o3_pairwise_ranked_df = build_global_edge_ranking(o3_pairwise_display_df, sort_col="mean")
    pm25_to_o3_ranked_df = build_global_edge_ranking(pm25_to_o3_display_df, sort_col="mean")
    synergy_ranked_df = build_global_edge_ranking(synergy_display_df, sort_col="abs_mean")
    profile_variable_df = build_support_cover_profile_table(tm_summary["profile"])
    summary_metrics_df = _metric_summary_table(
        bundle=bundle,
        predictions=predictions,
        tm_summary=tm_summary,
    )
    graph_paths = _build_notebook_graph_paths(
        root_dir=resolved_root,
        city_en=city_en,
        horizon=int(horizon),
        run_tag=run_tag,
        use_smoke=use_smoke,
        top_k_edges=int(top_k_edges),
        min_abs_strength=float(min_abs_strength),
        show_negative_synergy_edges=bool(show_negative_synergy_edges),
    )
    _render_graphs(
        bundle=bundle,
        o3_pairwise_display_df=o3_pairwise_display_df,
        pm25_to_o3_display_df=pm25_to_o3_display_df,
        synergy_display_df=synergy_display_df,
        graph_paths=graph_paths,
    )
    profile = tm_summary["profile"]
    if box_mode == "global_max":
        global_box_size = float(profile["global_box_size"])
        global_box_size_override_value = profile.get("global_box_size_override")
        box_description = f"scalar L=max_v L_v={global_box_size:.4f}"
    else:
        global_box_size = None
        global_box_size_override_value = None
        box_description = "support-cover L_v"
    final_conclusion_text = (
        f"{city_en.title()} {int(horizon)}h uses {box_description} with gamma={float(gamma):.2f}. "
        f"The notebook defaults to cached TM results, exposes O3 -> O3, PM2.5 -> O3, and "
        f"O3 + PM2.5 -> O3 synergy graphs, and lets you compare top-{int(top_k_edges)} edges "
        f"under different sample_count/seed settings."
    )
    return {
        "run_context": {
            "city_en": str(city_en),
            "horizon": int(horizon),
            "tm_sample_count": int(tm_sample_count),
            "sampling_seed": int(sampling_seed),
            "gamma": float(gamma),
            "box_mode": str(profile.get("box_mode", box_mode)),
            **({"global_box_size": global_box_size} if global_box_size is not None else {}),
            **(
                {"global_box_size_override": float(global_box_size_override_value)}
                if global_box_size_override_value is not None
                else {}
            ),
            "run_tag": run_tag,
            "used_cached_results": bool(used_cached_results),
            "results_dir": str(graph_paths["results_dir"]),
        },
        "summary_metrics_df": summary_metrics_df,
        "profile_variable_df": profile_variable_df,
        "o3_pairwise_df": o3_pairwise_df,
        "pm25_to_o3_df": pm25_to_o3_df,
        "synergy_df": synergy_df,
        "o3_pairwise_display_df": o3_pairwise_display_df,
        "pm25_to_o3_display_df": pm25_to_o3_display_df,
        "synergy_display_df": synergy_display_df,
        "o3_pairwise_ranked_df": o3_pairwise_ranked_df,
        "pm25_to_o3_ranked_df": pm25_to_o3_ranked_df,
        "synergy_ranked_df": synergy_ranked_df,
        "graph_paths": {key: str(value) for key, value in graph_paths.items() if key != "results_dir"},
        "station_positions_df": bundle["city_metadata"][["station_id", "lon", "lat"]].copy(),
        "station_ids": list(bundle["station_ids"]),
        "profile": tm_summary["profile"],
        "tm_summary": tm_summary,
        "final_conclusion_text": final_conclusion_text,
    }


# --- Compatibility aliases for former yrd submodules ---

import sys as _yrd_sys

_yrd_module = _yrd_sys.modules[__name__]

_YRD_SUBMODULE_ALIASES = ('config', 'transport_map', 'analysis', 'intervention_sampling', 'groups', 'data', 'models', 'coupling', 'train', 'plotting', 'shanghai_notebook', 'air_search', 'air_search_notebook')

for _yrd_submodule in _YRD_SUBMODULE_ALIASES:

    _yrd_sys.modules[f"{__name__}.{_yrd_submodule}"] = _yrd_module

    setattr(_yrd_module, _yrd_submodule, _yrd_module)

__all__ = [name for name in globals() if not name.startswith("_")]
