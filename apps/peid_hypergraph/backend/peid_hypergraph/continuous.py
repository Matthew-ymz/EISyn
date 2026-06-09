from __future__ import annotations

import itertools

import numpy as np

from exp.TM.transport_map_density import estimate_mutual_information_transport_map


def _clip_ei(value: float) -> float:
    return float(max(float(value), 0.0))


def _lift_source_features(source: np.ndarray) -> np.ndarray:
    array = np.asarray(source, dtype=float)
    if array.ndim != 2:
        raise ValueError("source must be a 2D array.")
    if array.shape[1] == 1:
        return array
    columns = [array]
    for left, right in itertools.combinations(range(array.shape[1]), 2):
        columns.append((array[:, [left]] * array[:, [right]]))
    return np.concatenate(columns, axis=1)


def _estimate_ei(source: np.ndarray, target: np.ndarray) -> float:
    lifted = _lift_source_features(source)
    summary = estimate_mutual_information_transport_map(lifted, target, degree=2)
    return _clip_ei(float(summary["mi_hat"]))


def _sine_common_driver(payload: dict[str, object]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, dict[str, object]]:
    variables = ("w", "x", "y", "z")
    sample_count = int(payload.get("intervention_samples", 2048))
    if sample_count < 100:
        raise ValueError("intervention_samples must be at least 100.")
    seed = int(payload.get("seed", 42))
    rng = np.random.default_rng(seed)
    sources = rng.uniform(-2.0, 2.0, size=(sample_count, len(variables)))
    w = sources[:, 0]
    x = sources[:, 1]
    y = sources[:, 2]
    z = sources[:, 3]

    alpha = float(payload.get("alpha", 1.0))
    beta = float(payload.get("beta", 0.75))
    noise_std = float(payload.get("noise_std", 0.05))
    if noise_std < 0.0:
        raise ValueError("noise_std must be nonnegative.")
    noise = rng.normal(0.0, noise_std, size=(sample_count, len(variables)))

    next_state = np.column_stack(
        [
            0.78 * w + noise[:, 0],
            0.42 * x + 0.82 * beta * w + noise[:, 1],
            0.38 * y + 0.76 * beta * w + noise[:, 2],
            0.22 * z + alpha * np.sin(x * y) + noise[:, 3],
        ]
    )
    metadata = {
        "mode": "continuous_sine",
        "example": "sine_common_driver",
        "alpha": alpha,
        "beta": beta,
        "noise_std": noise_std,
        "intervention_samples": sample_count,
        "seed": seed,
        "max_source_order": int(payload.get("max_source_order", 2)),
    }
    return variables, sources, next_state, metadata


def compute_continuous_graph(payload: dict[str, object]) -> dict[str, object]:
    example = str(payload.get("example", "sine_common_driver"))
    if example != "sine_common_driver":
        raise ValueError("Only the sine_common_driver continuous example is implemented in the first version.")
    variables, source_samples, target_samples, metadata = _sine_common_driver(payload)
    max_source_order = int(metadata["max_source_order"])
    if max_source_order < 1 or max_source_order > len(variables):
        raise ValueError("max_source_order must be between 1 and the node count.")

    nodes = [{"id": name, "label": name} for name in variables]
    pairwise_edges: list[dict[str, object]] = []
    ei_lookup: dict[tuple[tuple[str, ...], str], float] = {}
    index_by_name = {name: idx for idx, name in enumerate(variables)}

    for source in variables:
        source_idx = index_by_name[source]
        for target in variables:
            target_idx = index_by_name[target]
            ei = _estimate_ei(source_samples[:, [source_idx]], target_samples[:, [target_idx]])
            ei_lookup[((source,), target)] = ei
            pairwise_edges.append({"source": source, "target": target, "ei": ei})

    hyperedges: list[dict[str, object]] = []
    signed_interactions: list[dict[str, object]] = []
    for order in range(2, max_source_order + 1):
        for source_set in itertools.combinations(variables, order):
            source_indices = [index_by_name[source] for source in source_set]
            for target in variables:
                target_idx = index_by_name[target]
                joint_ei = _estimate_ei(source_samples[:, source_indices], target_samples[:, [target_idx]])
                singleton_sum = sum(ei_lookup[((source,), target)] for source in source_set)
                if order == 2:
                    interaction = joint_ei - singleton_sum
                    interaction_type = "synergy"
                else:
                    interaction = joint_ei
                    for size in range(1, order):
                        sign = -1 if (order - size) % 2 else 1
                        for subset in itertools.combinations(source_set, size):
                            subset_indices = [index_by_name[source] for source in subset]
                            subset_ei = _estimate_ei(source_samples[:, subset_indices], target_samples[:, [target_idx]])
                            interaction += sign * subset_ei
                    interaction_type = "signed_interaction"
                row = {
                    "sources": list(source_set),
                    "target": target,
                    "source_order": order,
                    "joint_ei": joint_ei,
                    "single_ei_sum": singleton_sum,
                    "synergy": float(interaction),
                    "display_value": max(float(interaction), 0.0),
                    "interaction_type": interaction_type,
                }
                if interaction > 1.0e-9:
                    hyperedges.append(row)
                elif order >= 3:
                    signed_interactions.append(row)

    return {
        "nodes": nodes,
        "pairwise_edges": pairwise_edges,
        "hyperedges": hyperedges,
        "diagnostics": {
            "signed_interactions": signed_interactions,
            "estimator": "transport_map",
            "source_intervention": "maximum_entropy_independent",
            "state_family": "continuous",
        },
        "metadata": metadata,
    }
