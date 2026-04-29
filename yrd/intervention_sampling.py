from __future__ import annotations

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
