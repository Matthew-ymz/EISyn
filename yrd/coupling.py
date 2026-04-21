from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable
from itertools import combinations

import numpy as np
import torch

from yrd.transport_map import estimate_mutual_information_transport_map


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
            group_ei[group_name] = float(summary["mi_hat"])
            overall_backend = str(summary["backend"])
        all_indices = sorted({index for indices in normalized_groups.values() for index in indices})
        if not all_indices:
            overall_ei = 0.0
        else:
            overall = estimate_mutual_information_transport_map(source_matrix[:, all_indices], target_matrix)
            overall_ei = float(overall["mi_hat"])
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
        name: _subset_ei_nis(target_jacobian, target_sigma_eps, indices, box_size=box_size, atol=atol)
        for name, indices in source_groups.items()
    }
    whole_sources = sorted({index for indices in source_groups.values() for index in indices})
    ei_full = _subset_ei_nis(target_jacobian, target_sigma_eps, whole_sources, box_size=box_size, atol=atol)
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
            left_value = float(left_summary["mi_hat"])
            right_value = float(right_summary["mi_hat"])
            joint_value = float(joint_summary["mi_hat"])
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
        right_value = _subset_ei_nis(
            target_jacobian,
            target_sigma_eps,
            right_indices,
            box_size=box_size,
            atol=atol,
        )
        joint_indices = sorted(set(left_indices + right_indices))
        joint_value = _subset_ei_nis(
            target_jacobian,
            target_sigma_eps,
            joint_indices,
            box_size=box_size,
            atol=atol,
        )
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


def save_coupling_summary(summary: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
