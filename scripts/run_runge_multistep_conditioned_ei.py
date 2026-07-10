#!/usr/bin/env python3
"""Runge multistep conditioned EI experiment with MLP closed-loop rollout."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_runge_gateway_mediator_map import (  # noqa: E402
    COASTLINE_URL,
    DEFAULT_COMPONENT_MAPS,
    LAND_URL,
    add_geographic_ticks,
    add_labels,
    component_center,
    draw_world,
    extract_lines,
    extract_polygons,
    load_geojson,
    local_to_paper,
)
from scripts.run_runge_pairwise_mlp_ei import (  # noqa: E402
    AveragedTransition,
    PairwiseMlpEiConfig,
    WeightedAveragedTransition,
    _frame_content_hash,
    _jsonable_config,
    _model_config_hash,
    build_lagged_dataset,
    build_scaled_ridge_transition,
    discrete_effective_information,
    load_component_scores,
    predict_mlp,
    regression_metrics,
    sample_max_entropy_features,
    select_linear_blend_weight,
    split_temporal_arrays,
    train_or_load_mlp,
)


NEW_RUNGE_BASE = ROOT / "results" / "runge_slp_daily_1948_2026_20260628" / "mlp_tm_ei_lag04" / "results" / "runge"
NEW_RUNGE_FIG_BASE = ROOT / "fig" / "runge_slp_daily_1948_2026_20260628" / "multistep_conditioned_ei"
DEFAULT_PAIRWISE_MANIFEST = NEW_RUNGE_BASE / "pairwise_mlp_tm_ei_path_effects" / "manifest.json"
DEFAULT_REFERENCE_GATEWAY = NEW_RUNGE_BASE / "pairwise_mlp_tm_ei_path_effects" / "gateway_scores.csv"
DEFAULT_REFERENCE_MEDIATOR = NEW_RUNGE_BASE / "pairwise_mlp_tm_ei_path_effects" / "mediator_scores.csv"
DEFAULT_RESULT_DIR = NEW_RUNGE_BASE / "multistep_conditioned_ei"
DEFAULT_ASSET_DIR = NEW_RUNGE_FIG_BASE
DEFAULT_REPORT = ROOT / "docs" / "reports" / "Runge_Multistep_EI_Path_Design.md"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.65,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def rollout_mlp_closed_loop(
    model: object,
    scalers: dict[str, np.ndarray],
    initial_features: np.ndarray,
    *,
    n_components: int,
    lag: int,
    horizons: int,
) -> np.ndarray:
    """Roll the one-step MLP forward without resampling interventions."""
    window = np.asarray(initial_features, dtype=float).copy()
    expected = int(n_components) * int(lag)
    if window.ndim != 2 or window.shape[1] != expected:
        raise ValueError(f"initial_features must have shape (samples, {expected}).")
    predictions = np.empty((window.shape[0], int(horizons), int(n_components)), dtype=float)
    for horizon in range(int(horizons)):
        next_state = np.asarray(predict_mlp(model, scalers, window), dtype=float)
        if next_state.shape != (window.shape[0], int(n_components)):
            raise ValueError("model prediction shape does not match n_components.")
        predictions[:, horizon, :] = next_state
        window = np.concatenate([window[:, int(n_components) :], next_state], axis=1)
    return predictions


def conditional_average_matrix(pairwise_ei: np.ndarray, joint_ei: np.ndarray) -> np.ndarray:
    """Average EI({i,r}->j)-EI(r->j) over r != i,j."""
    pairwise = np.asarray(pairwise_ei, dtype=float)
    joint = np.asarray(joint_ei, dtype=float)
    if pairwise.ndim != 2 or pairwise.shape[0] != pairwise.shape[1]:
        raise ValueError("pairwise_ei must be a square matrix.")
    n = pairwise.shape[0]
    if joint.shape != (n, n, n):
        raise ValueError("joint_ei must have shape (n, n, n).")
    conditioned = np.zeros((n, n), dtype=float)
    for source in range(n):
        for target in range(n):
            if source == target:
                continue
            terms = []
            for other in range(n):
                if other in (source, target):
                    continue
                value = float(joint[source, other, target]) - float(pairwise[other, target])
                if np.isfinite(value):
                    terms.append(value)
            conditioned[source, target] = float(np.mean(terms)) if terms else 0.0
    np.fill_diagonal(conditioned, 0.0)
    return conditioned


def cumulative_positive_matrix(horizon_signed_mats: np.ndarray, checkpoint_horizon: int) -> np.ndarray:
    mats = np.asarray(horizon_signed_mats, dtype=float)
    if mats.ndim != 3 or mats.shape[1] != mats.shape[2]:
        raise ValueError("horizon_signed_mats must have shape (horizon, n, n).")
    if int(checkpoint_horizon) < 1 or int(checkpoint_horizon) > mats.shape[0]:
        raise ValueError("checkpoint_horizon is outside computed horizons.")
    return np.maximum(mats[: int(checkpoint_horizon)], 0.0).sum(axis=0)


def compute_multistep_scores(
    total_matrix: np.ndarray,
    one_step_positive_matrix: np.ndarray,
    names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute ACE/ACS and unit-consistent AMCE from multistep conditioned EI."""
    total = np.nan_to_num(np.asarray(total_matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).copy()
    inbound = np.nan_to_num(np.asarray(one_step_positive_matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).copy()
    if total.ndim != 2 or total.shape[0] != total.shape[1]:
        raise ValueError("total_matrix must be square.")
    if inbound.shape != total.shape:
        raise ValueError("one_step_positive_matrix must match total_matrix.")
    n = total.shape[0]
    if len(names) != n:
        raise ValueError("names length must match matrix shape.")
    total = np.maximum(total, 0.0)
    inbound = np.maximum(inbound, 0.0)
    np.fill_diagonal(total, 0.0)
    np.fill_diagonal(inbound, 0.0)
    denom = max(1, n - 1)
    gateway_rows: list[dict[str, object]] = []
    mediator_rows: list[dict[str, object]] = []
    for idx, name in enumerate(names):
        gateway_rows.append(
            {
                "component": name,
                "component_index": idx,
                "paper_component": local_to_paper(idx),
                "ace": float(np.sum(total[idx, :]) / denom),
                "acs": float(np.sum(total[:, idx]) / denom),
                "total_out_strength": float(np.sum(total[idx, :])),
                "total_in_strength": float(np.sum(total[:, idx])),
            }
        )
        incoming = inbound[:, idx].copy()
        incoming[idx] = 0.0
        incoming_sum = float(np.sum(incoming))
        weights = incoming / incoming_sum if incoming_sum > 1.0e-12 else np.zeros_like(incoming)
        amce = 0.0
        product_amce = 0.0
        for target in range(n):
            if target == idx:
                continue
            source_weight = float(np.sum(weights) - weights[target])
            source_raw = float(np.sum(incoming) - incoming[target])
            amce += source_weight * float(total[idx, target])
            product_amce += source_raw * float(total[idx, target])
        mediator_rows.append(
            {
                "component": name,
                "component_index": idx,
                "paper_component": local_to_paper(idx),
                "amce": float(amce / max(1, n - 2)),
                "amce_product_diagnostic": float(product_amce / max(1, (n - 1) * (n - 2))),
                "incoming_conditioned_ei": incoming_sum,
                "mediated_fraction": 0.0,
            }
        )
    gateway = pd.DataFrame(gateway_rows)
    gateway["out_rank"] = gateway["ace"].rank(ascending=False, method="min").astype(int)
    gateway["in_rank"] = gateway["acs"].rank(ascending=False, method="min").astype(int)
    gateway = gateway.sort_values("ace", ascending=False).reset_index(drop=True)
    mediator = pd.DataFrame(mediator_rows)
    total_amce = float(mediator["amce"].sum())
    if total_amce > 1.0e-12:
        mediator["mediated_fraction"] = mediator["amce"] / total_amce
    mediator["amce_rank"] = mediator["amce"].rank(ascending=False, method="min").astype(int)
    mediator = mediator.sort_values("amce", ascending=False).reset_index(drop=True)
    return gateway, mediator


def top_indices(frame: pd.DataFrame, value_col: str, *, k: int = 5) -> list[int]:
    return [int(idx) for idx in frame.sort_values(value_col, ascending=False)["component_index"].head(int(k)).tolist()]


def checkpoint_top_sets_match(current: dict[str, Sequence[int]], reference: dict[str, set[int]]) -> bool:
    for metric in ("ace", "acs", "amce"):
        if set(int(v) for v in current[metric][:5]) != set(int(v) for v in reference[metric]):
            return False
    return True


def load_reference_top_sets(gateway_path: Path, mediator_path: Path, *, k: int = 5) -> dict[str, set[int]]:
    gateway = pd.read_csv(gateway_path)
    mediator = pd.read_csv(mediator_path)
    return {
        "ace": set(top_indices(gateway, "ace", k=k)),
        "acs": set(top_indices(gateway, "acs", k=k)),
        "amce": set(top_indices(mediator, "amce", k=k)),
    }


def rank_correlations(
    gateway: pd.DataFrame,
    mediator: pd.DataFrame,
    reference_gateway: pd.DataFrame,
    reference_mediator: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    current = {
        "ace": gateway.set_index("component_index")["ace"],
        "acs": gateway.set_index("component_index")["acs"],
        "amce": mediator.set_index("component_index")["amce"],
    }
    reference = {
        "ace": reference_gateway.set_index("component_index")["ace"],
        "acs": reference_gateway.set_index("component_index")["acs"],
        "amce": reference_mediator.set_index("component_index")["amce"],
    }
    out: dict[str, dict[str, float]] = {}
    for metric in ("ace", "acs", "amce"):
        frame = pd.concat([current[metric], reference[metric]], axis=1, keys=["current", "reference"]).dropna()
        out[metric] = {
            "spearman": float(frame["current"].corr(frame["reference"], method="spearman")),
            "kendall": float(frame["current"].corr(frame["reference"], method="kendall")),
        }
    return out


def overlap_summary(
    gateway: pd.DataFrame,
    mediator: pd.DataFrame,
    reference_gateway: pd.DataFrame,
    reference_mediator: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for metric, current_frame, ref_frame, value_col in (
        ("ace", gateway, reference_gateway, "ace"),
        ("acs", gateway, reference_gateway, "acs"),
        ("amce", mediator, reference_mediator, "amce"),
    ):
        summary[metric] = {}
        for k in (5, 10):
            current = set(top_indices(current_frame, value_col, k=k))
            reference = set(top_indices(ref_frame, value_col, k=k))
            summary[metric][f"top{k}_overlap"] = int(len(current & reference))
    return summary


def _discretize_for_ei(values: np.ndarray, bins: int) -> np.ndarray:
    from scripts.run_runge_pairwise_mlp_ei import _discretize

    arr = np.asarray(values)
    if arr.ndim == 1:
        return _discretize(arr, int(bins))
    return np.column_stack([_discretize(arr[:, col], int(bins)) for col in range(arr.shape[1])])


def estimate_mi(source_state: np.ndarray, target_state: np.ndarray, *, estimator: str, bins: int) -> tuple[float, float | None]:
    if estimator == "discrete":
        source_codes = _discretize_for_ei(source_state, int(bins))
        target_codes = _discretize_for_ei(target_state, int(bins))
        return max(0.0, float(discrete_effective_information(source_codes, target_codes))), None
    if estimator == "tm":
        from exp.TM.transport_map_density import estimate_mutual_information_transport_map

        summary = estimate_mutual_information_transport_map(np.asarray(source_state, dtype=float), np.asarray(target_state, dtype=float))
        return max(0.0, float(summary["mi_hat"])), float(summary.get("bias_correction", 0.0))
    raise ValueError("estimator must be 'tm' or 'discrete'.")


def source_state_matrix(features: np.ndarray, *, n_components: int, lag: int, source_mode: str) -> list[np.ndarray]:
    states: list[np.ndarray] = []
    for source in range(int(n_components)):
        cols = [lag_idx * int(n_components) + source for lag_idx in range(int(lag))]
        if source_mode == "latest":
            states.append(np.asarray(features[:, [cols[-1]]], dtype=float))
        elif source_mode == "history":
            states.append(np.asarray(features[:, cols], dtype=float))
        else:
            raise ValueError("source_mode must be 'latest' or 'history'.")
    return states


def estimate_horizon_conditioned_ei(
    source_states: Sequence[np.ndarray],
    horizon_targets: np.ndarray,
    *,
    estimator: str,
    bins: int,
    names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    n = len(source_states)
    if horizon_targets.shape[1] != n:
        raise ValueError("horizon_targets must contain one column per component.")
    pairwise = np.zeros((n, n), dtype=float)
    pairwise_bias = np.full((n, n), np.nan, dtype=float)
    for source in range(n):
        for target in range(n):
            ei, bias = estimate_mi(source_states[source], horizon_targets[:, [target]], estimator=estimator, bins=int(bins))
            pairwise[source, target] = ei
            if bias is not None:
                pairwise_bias[source, target] = bias

    joint = np.full((n, n, n), np.nan, dtype=float)
    rows: list[dict[str, object]] = []
    for first in range(n):
        for second in range(first + 1, n):
            joint_source = np.concatenate([source_states[first], source_states[second]], axis=1)
            for target in range(n):
                if target in (first, second):
                    continue
                ei, bias = estimate_mi(joint_source, horizon_targets[:, [target]], estimator=estimator, bins=int(bins))
                joint[first, second, target] = ei
                joint[second, first, target] = ei
                rows.append(
                    {
                        "source_i": names[first],
                        "source_r": names[second],
                        "target": names[target],
                        "source_i_index": first,
                        "source_r_index": second,
                        "target_index": target,
                        "joint_ei": float(ei),
                        "bias_correction": bias,
                    }
                )
    conditioned = conditional_average_matrix(pairwise, joint)
    edge_rows: list[dict[str, object]] = []
    for source in range(n):
        for target in range(n):
            edge_rows.append(
                {
                    "source": names[source],
                    "target": names[target],
                    "source_index": source,
                    "target_index": target,
                    "pairwise_ei": float(pairwise[source, target]),
                    "pairwise_bias_correction": None if np.isnan(pairwise_bias[source, target]) else float(pairwise_bias[source, target]),
                    "conditioned_ei_signed": float(conditioned[source, target]),
                    "conditioned_ei_positive": float(max(conditioned[source, target], 0.0)),
                }
            )
    return pairwise, joint, conditioned, pd.DataFrame(edge_rows)


def source_pairs(n_components: int) -> list[tuple[int, int]]:
    return [(first, second) for first in range(int(n_components)) for second in range(first + 1, int(n_components))]


def pairwise_edge_frame(
    pairwise: np.ndarray,
    pairwise_bias: np.ndarray,
    conditioned: np.ndarray,
    names: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source in range(len(names)):
        for target in range(len(names)):
            rows.append(
                {
                    "source": names[source],
                    "target": names[target],
                    "source_index": source,
                    "target_index": target,
                    "pairwise_ei": float(pairwise[source, target]),
                    "pairwise_bias_correction": None if np.isnan(pairwise_bias[source, target]) else float(pairwise_bias[source, target]),
                    "conditioned_ei_signed": float(conditioned[source, target]),
                    "conditioned_ei_positive": float(max(conditioned[source, target], 0.0)),
                }
            )
    return pd.DataFrame(rows)


def estimate_pairwise_for_horizon(
    source_states: Sequence[np.ndarray],
    horizon_targets: np.ndarray,
    *,
    estimator: str,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(source_states)
    pairwise = np.zeros((n, n), dtype=float)
    bias_matrix = np.full((n, n), np.nan, dtype=float)
    for source in range(n):
        for target in range(n):
            ei, bias = estimate_mi(source_states[source], horizon_targets[:, [target]], estimator=estimator, bins=int(bins))
            pairwise[source, target] = ei
            if bias is not None:
                bias_matrix[source, target] = bias
    return pairwise, bias_matrix


def estimate_joint_ei_chunks(
    source_states: Sequence[np.ndarray],
    horizon_targets: np.ndarray,
    *,
    estimator: str,
    bins: int,
    horizon_dir: Path,
    chunk_size: int,
    resume: bool,
) -> np.ndarray:
    n = len(source_states)
    pairs = source_pairs(n)
    joint_dir = horizon_dir / "joint_chunks"
    joint_dir.mkdir(parents=True, exist_ok=True)
    for chunk_start in range(0, len(pairs), max(1, int(chunk_size))):
        chunk_pairs = pairs[chunk_start : chunk_start + max(1, int(chunk_size))]
        chunk_path = joint_dir / f"chunk_{chunk_start // max(1, int(chunk_size)):04d}.npz"
        if chunk_path.exists() and bool(resume):
            print(f"[joint] reuse {chunk_path}", flush=True)
            continue
        print(f"[joint] computing {chunk_path} pairs={len(chunk_pairs)}", flush=True)
        values = np.full((len(chunk_pairs), n), np.nan, dtype=float)
        bias_values = np.full((len(chunk_pairs), n), np.nan, dtype=float)
        for pair_idx, (first, second) in enumerate(chunk_pairs):
            joint_source = np.concatenate([source_states[first], source_states[second]], axis=1)
            for target in range(n):
                if target in (first, second):
                    continue
                ei, bias = estimate_mi(joint_source, horizon_targets[:, [target]], estimator=estimator, bins=int(bins))
                values[pair_idx, target] = ei
                if bias is not None:
                    bias_values[pair_idx, target] = bias
        np.savez_compressed(
            chunk_path,
            first=np.asarray([pair[0] for pair in chunk_pairs], dtype=np.int16),
            second=np.asarray([pair[1] for pair in chunk_pairs], dtype=np.int16),
            values=values,
            bias=bias_values,
        )

    joint = np.full((n, n, n), np.nan, dtype=float)
    for chunk_path in sorted(joint_dir.glob("chunk_*.npz")):
        payload = np.load(chunk_path)
        first_indices = payload["first"].astype(int)
        second_indices = payload["second"].astype(int)
        values = payload["values"].astype(float)
        for row_idx, (first, second) in enumerate(zip(first_indices, second_indices, strict=True)):
            joint[first, second, :] = values[row_idx]
            joint[second, first, :] = values[row_idx]
    expected_chunks = math.ceil(len(pairs) / max(1, int(chunk_size)))
    actual_chunks = len(list(joint_dir.glob("chunk_*.npz")))
    if actual_chunks < expected_chunks:
        raise RuntimeError(f"joint EI chunks incomplete: {actual_chunks}/{expected_chunks}.")
    return joint


def estimate_horizon_conditioned_ei_cached(
    source_states: Sequence[np.ndarray],
    horizon_targets: np.ndarray,
    *,
    estimator: str,
    bins: int,
    names: Sequence[str],
    horizon_dir: Path,
    chunk_size: int,
    resume: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    horizon_dir.mkdir(parents=True, exist_ok=True)
    pairwise_path = horizon_dir / "pairwise_ei.npy"
    pairwise_bias_path = horizon_dir / "pairwise_ei_bias.npy"
    if pairwise_path.exists() and pairwise_bias_path.exists() and bool(resume):
        pairwise = np.load(pairwise_path)
        pairwise_bias = np.load(pairwise_bias_path)
    else:
        pairwise, pairwise_bias = estimate_pairwise_for_horizon(
            source_states,
            horizon_targets,
            estimator=estimator,
            bins=int(bins),
        )
        np.save(pairwise_path, pairwise)
        np.save(pairwise_bias_path, pairwise_bias)
        save_matrix_csv(pairwise, names, horizon_dir / "pairwise_ei_matrix.csv")
    joint_path = horizon_dir / "joint_ei.npy"
    if joint_path.exists() and bool(resume):
        joint = np.load(joint_path)
    else:
        joint = estimate_joint_ei_chunks(
            source_states,
            horizon_targets,
            estimator=estimator,
            bins=int(bins),
            horizon_dir=horizon_dir,
            chunk_size=int(chunk_size),
            resume=bool(resume),
        )
        np.save(joint_path, joint)
    conditioned = conditional_average_matrix(pairwise, joint)
    edges = pairwise_edge_frame(pairwise, pairwise_bias, conditioned, names)
    return pairwise, joint, conditioned, edges


def config_from_manifest(path: Path) -> PairwiseMlpEiConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = payload.get("config", {})
    fields = PairwiseMlpEiConfig.__dataclass_fields__
    kwargs = {key: value for key, value in config.items() if key in fields}
    for path_key in ("component_scores", "linear_coefficients", "output_dir"):
        if path_key in kwargs and kwargs[path_key] is not None:
            kwargs[path_key] = Path(kwargs[path_key])
    if "ensemble_ridge_alphas" in kwargs:
        kwargs["ensemble_ridge_alphas"] = tuple(float(v) for v in kwargs["ensemble_ridge_alphas"])
    kwargs["force_retrain"] = False
    return PairwiseMlpEiConfig(**kwargs)


def load_cached_pairwise_model(config: PairwiseMlpEiConfig) -> tuple[object, dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]], list[str], dict[str, object]]:
    frame = load_component_scores(config.component_scores)
    names = list(frame.columns)
    features, targets = build_lagged_dataset(frame, lag=int(config.lag), horizon=1)
    splits = split_temporal_arrays(features, targets, train_fraction=float(config.train_fraction), val_fraction=float(config.val_fraction))
    data_hash = _frame_content_hash(frame)
    ensemble_alphas = tuple(config.ensemble_ridge_alphas) if config.ensemble_ridge_alphas else (float(config.ridge_alpha),)
    models: list[object] = []
    model_paths: list[str] = []
    member_summaries: list[dict[str, object]] = []
    scalers: dict[str, np.ndarray] | None = None
    cache_reused = True
    model_dir = Path(config.output_dir) / "results" / "runge" / "pairwise_mlp_ei"
    for alpha in ensemble_alphas:
        member_config = replace(config, ridge_alpha=float(alpha), ensemble_ridge_alphas=(), force_retrain=False)
        member_hash = _model_config_hash(member_config, n_components=len(names), n_rows=len(frame), data_hash=data_hash)
        if len(ensemble_alphas) == 1:
            member_path = model_dir / "mlp_transition.pt"
        else:
            alpha_label = str(float(alpha)).replace("-", "m").replace(".", "p")
            member_path = model_dir / f"mlp_transition_alpha{alpha_label}_{member_hash}.pt"
        member_model, member_scalers, _, reused = train_or_load_mlp(splits, member_config, member_path, config_hash=member_hash)
        models.append(member_model)
        model_paths.append(str(member_path))
        scalers = member_scalers if scalers is None else scalers
        cache_reused = cache_reused and bool(reused)
        member_summaries.append(
            {
                "ridge_alpha": float(alpha),
                "model_cache": str(member_path),
                "model_config_hash": member_hash,
                "cache_reused": bool(reused),
                "training": getattr(member_model, "training_summary", {}),
            }
        )
    model: object = models[0] if len(models) == 1 else AveragedTransition(models)
    assert scalers is not None
    linear_blend: dict[str, object] = {"enabled": False}
    if int(config.linear_blend_grid_steps) > 1:
        ridge_transition = build_scaled_ridge_transition(splits, scalers, ridge_alpha=float(config.ridge_alpha))
        blend = select_linear_blend_weight(
            model,
            ridge_transition,
            scalers,
            splits,
            names,
            grid_steps=int(config.linear_blend_grid_steps),
        )
        model = WeightedAveragedTransition(
            [model, ridge_transition],
            [float(blend["mlp_weight"]), float(blend["ridge_weight"])],
            training_summary={
                "type": "validation_linear_blend",
                "base_model": getattr(model, "training_summary", {}),
                "ridge_model": getattr(ridge_transition, "training_summary", {}),
                **blend,
            },
        )
        linear_blend = {
            "enabled": True,
            "ridge_alpha": float(config.ridge_alpha),
            "grid_steps": int(config.linear_blend_grid_steps),
            **blend,
        }
    return model, scalers, splits, names, {
        "model_caches": model_paths,
        "model_cache_reused": bool(cache_reused),
        "ensemble_members": member_summaries,
        "linear_blend": linear_blend,
        "component_scores_hash": data_hash,
        "n_rows": int(len(frame)),
        "n_lagged_samples": int(len(features)),
    }


def save_matrix_csv(matrix: np.ndarray, names: Sequence[str], path: Path) -> None:
    pd.DataFrame(matrix, index=names, columns=names).to_csv(path)


def build_node_frame(component_maps: np.ndarray, gateway: pd.DataFrame, mediator: pd.DataFrame) -> pd.DataFrame:
    lat = np.linspace(-90.0, 90.0, component_maps.shape[0])
    lon = np.linspace(0.0, 360.0, component_maps.shape[1], endpoint=False)
    lon = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    maps = component_maps[:, order, :]
    lon = lon[order]
    merged = gateway.merge(mediator[["component_index", "amce", "mediated_fraction"]], on="component_index", how="left")
    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        local = int(row.component_index)
        center_lon, center_lat = component_center(maps[..., local], lat, lon)
        rows.append(
            {
                "local": local,
                "paper": local_to_paper(local),
                "lon": center_lon,
                "lat": center_lat,
                "ace": float(row.ace),
                "acs": float(row.acs),
                "amce": float(row.amce),
                "mediated_fraction": float(row.mediated_fraction),
            }
        )
    return pd.DataFrame(rows)


def _scatter_metric(ax: plt.Axes, nodes: pd.DataFrame, metric: str, norm: mpl.colors.Normalize, cmap_name: str) -> None:
    values = nodes[metric].to_numpy(dtype=float)
    vmax = max(float(np.nanmax(values)), 1.0e-12)
    sizes = 190.0 + 330.0 * np.clip(values / vmax, 0.0, 1.0)
    ax.scatter(
        np.radians(nodes["lon"].to_numpy()),
        np.radians(nodes["lat"].to_numpy()),
        s=sizes,
        c=values,
        cmap=mpl.colormaps[cmap_name],
        norm=norm,
        edgecolors="#2d2d2d",
        linewidths=0.32,
        alpha=0.96,
        zorder=4,
    )
    add_labels(ax, nodes)


def plot_three_panel_map(nodes: pd.DataFrame, output: Path, *, save_svg: bool = True, save_pdf: bool = True) -> Path:
    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    fig = plt.figure(figsize=(11.8, 3.9), constrained_layout=True)
    axes = [fig.add_subplot(1, 3, idx + 1, projection="mollweide") for idx in range(3)]
    for ax in axes:
        draw_world(ax, land, coastlines)
        add_geographic_ticks(ax)
    metrics = [
        ("ACE", "ace", "OrRd"),
        ("ACS", "acs", "YlGnBu"),
        ("AMCE", "amce", "Greens"),
    ]
    for label, ax, (title, metric, cmap) in zip(["a", "b", "c"], axes, metrics, strict=True):
        vmax = max(float(nodes[metric].max()), 1.0e-12)
        norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
        _scatter_metric(ax, nodes, metric, norm, cmap)
        ax.text(-0.07, 1.06, label, transform=ax.transAxes, fontsize=14, fontweight="bold")
        ax.text(0.5, 1.06, f"Multistep conditioned EI {title}", transform=ax.transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colormaps[cmap])
        cbar = fig.colorbar(sm, ax=ax, location="bottom", shrink=0.74, pad=0.075, aspect=24)
        cbar.set_label(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if save_svg:
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    if save_pdf:
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output


def markdown_top_table(gateway: pd.DataFrame, mediator: pd.DataFrame, k: int = 10) -> str:
    rows = []
    for rank in range(int(k)):
        ace = gateway.sort_values("ace", ascending=False).iloc[rank]
        acs = gateway.sort_values("acs", ascending=False).iloc[rank]
        amce = mediator.sort_values("amce", ascending=False).iloc[rank]
        rows.append(
            [
                rank + 1,
                f"{int(ace.paper_component)} ({float(ace.ace):.4g})",
                f"{int(acs.paper_component)} ({float(acs.acs):.4g})",
                f"{int(amce.paper_component)} ({float(amce.amce):.4g})",
            ]
        )
    lines = ["| Rank | ACE | ACS | AMCE |", "|---:|---:|---:|---:|"]
    lines.extend(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in rows)
    return "\n".join(lines)


def append_report_section(
    report_path: Path,
    *,
    figure_path: Path,
    final_horizon: int,
    stopped_early: bool,
    checkpoint_rows: list[dict[str, object]],
    gateway: pd.DataFrame,
    mediator: pd.DataFrame,
    correlations: dict[str, dict[str, float]],
    overlaps: dict[str, dict[str, int]],
) -> None:
    start = "<!-- multistep-conditioned-ei-results:start -->"
    end = "<!-- multistep-conditioned-ei-results:end -->"
    try:
        rel_figure = figure_path.relative_to(report_path.parent)
    except ValueError:
        rel_figure = Path(os.path.relpath(figure_path, report_path.parent))
    stop_text = "触发早停" if stopped_early else "未触发早停"
    checkpoint_table = ["| H | ACE top-5 overlap | ACS top-5 overlap | AMCE top-5 overlap | all matched |", "|---:|---:|---:|---:|:---:|"]
    for row in checkpoint_rows:
        overlap = row["overlap"]
        checkpoint_table.append(
            f"| {int(row['horizon'])} | {int(overlap['ace']['top5_overlap'])}/5 | {int(overlap['acs']['top5_overlap'])}/5 | {int(overlap['amce']['top5_overlap'])}/5 | {'yes' if row['matched'] else 'no'} |"
        )
    corr_lines = ["| Metric | Spearman | Kendall | top-5 overlap | top-10 overlap |", "|---|---:|---:|---:|---:|"]
    for metric in ("ace", "acs", "amce"):
        corr_lines.append(
            f"| {metric.upper()} | {correlations[metric]['spearman']:.3f} | {correlations[metric]['kendall']:.3f} | {overlaps[metric]['top5_overlap']}/5 | {overlaps[metric]['top10_overlap']}/10 |"
        )
    section = f"""
{start}

### 实验结果：MLP 自迭代多步条件 EI

本次重跑使用训练好的 Runge one-step MLP，不重训模型。初始历史窗口从训练集分位数范围内做 bounded maximum-entropy intervention；之后每个 horizon 都由 MLP 闭环自迭代生成，不在中间步重新采样。对每个 horizon 先估计 pairwise EI 与联合源 EI，再用
\\[
\\bar E^{{[h],(2)}}_{{ij}}=\\frac{{1}}{{n-2}}\\sum_{{r\\ne i,j}}\\left[EI(X_{{\\{{i,r\\}},t}}\\to \\widehat X_{{j,t+h}})-EI(X_{{r,t}}\\to \\widehat X_{{j,t+h}})\\right]
\\]
构造 signed 条件边权；地图主指标使用正部 \\(E^{{[h],(2)+}}_{{ij}}=\\max(\\bar E^{{[h],(2)}}_{{ij}},0)\\)，并累计到 \\(H={final_horizon}\\)。

Checkpoint 结果：{stop_text}，最终绘图 horizon 为 \\(H={final_horizon}\\)。

{chr(10).join(checkpoint_table)}

![Runge multistep conditioned EI map]({rel_figure.as_posix()})

Top 节点如下，括号内为对应指标值：

{markdown_top_table(gateway, mediator, k=min(10, len(gateway)))}

与当前 `Part2 earth` Runge MLP-TM-EI path-effect 排名相比：

{chr(10).join(corr_lines)}

视觉上，新的三项指标仍集中在原 Runge 地图中若干高影响节点附近，但数值单位保持为直接估计得到的多步 EI 累计 bit，而不是 EI 邻接矩阵连乘后的路径权重。

{end}
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    old = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if start in old and end in old:
        prefix = old.split(start)[0].rstrip()
        suffix = old.split(end, 1)[1].lstrip()
        new_text = f"{prefix}\n\n{section.strip()}\n\n{suffix}".rstrip() + "\n"
    else:
        new_text = old.rstrip() + "\n\n" + section.strip() + "\n"
    report_path.write_text(new_text, encoding="utf-8")


def parse_checkpoints(text: str, max_horizon: int) -> list[int]:
    normalized = text.strip().lower()
    if normalized in {"auto", "each", "every-h", "every_h"}:
        return list(range(1, int(max_horizon) + 1))
    if normalized in {"every5", "every-5", "legacy"}:
        return list(range(5, int(max_horizon) + 1, 5))
    checkpoints = sorted({int(part) for part in text.split(",") if part.strip()})
    return [h for h in checkpoints if 1 <= h <= int(max_horizon)]


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    result_dir = Path(args.result_dir).expanduser()
    result_dir.mkdir(parents=True, exist_ok=True)
    config = config_from_manifest(Path(args.pairwise_manifest).expanduser())
    config = replace(
        config,
        intervention_samples=int(args.intervention_samples),
        ei_estimator=str(args.estimator),
        source_mode=str(args.source_mode),
        force_retrain=False,
    )
    model, scalers, splits, names, model_info = load_cached_pairwise_model(config)
    if int(args.n_components) > 0 and int(args.n_components) != len(names):
        raise ValueError(
            "--n-components must match the cached model component count. "
            "For a small smoke run, first create a matching small pairwise manifest."
        )
    n_components = len(names)
    checkpoints = parse_checkpoints(str(args.checkpoints), int(args.max_horizon))
    if not checkpoints:
        raise ValueError("at least one checkpoint is required.")
    max_horizon = max(checkpoints)
    features = sample_max_entropy_features(
        splits["train"][0],
        n_components=n_components,
        lag=int(config.lag),
        samples=int(args.intervention_samples),
        low_q=float(config.quantile_low),
        high_q=float(config.quantile_high),
        seed=int(config.seed),
    )
    source_states = source_state_matrix(features, n_components=n_components, lag=int(config.lag), source_mode=str(args.source_mode))
    rollout_path = result_dir / "rollout_predictions.npy"
    if rollout_path.exists() and bool(args.resume):
        predictions = np.load(rollout_path)
    else:
        predictions = rollout_mlp_closed_loop(model, scalers, features, n_components=n_components, lag=int(config.lag), horizons=max_horizon)
        np.save(rollout_path, predictions)

    reference_gateway = pd.read_csv(Path(args.reference_gateway).expanduser())
    reference_mediator = pd.read_csv(Path(args.reference_mediator).expanduser())
    reference_top = load_reference_top_sets(Path(args.reference_gateway).expanduser(), Path(args.reference_mediator).expanduser(), k=5)
    checkpoint_set = set(int(value) for value in checkpoints)
    signed_list: list[np.ndarray] = []
    checkpoint_rows: list[dict[str, object]] = []
    final_horizon = max_horizon
    stopped_early = False
    final_gateway: pd.DataFrame | None = None
    final_mediator: pd.DataFrame | None = None
    one_step_positive: np.ndarray | None = None

    for horizon in range(1, max_horizon + 1):
        horizon_dir = result_dir / f"horizon_{horizon:03d}"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        signed_path = horizon_dir / "conditioned_ei_signed.npy"
        if signed_path.exists() and bool(args.resume):
            print(f"[horizon {horizon}] reuse conditioned EI", flush=True)
            signed = np.load(signed_path)
        else:
            print(f"[horizon {horizon}] estimating conditioned EI", flush=True)
            pairwise, joint, signed, edges = estimate_horizon_conditioned_ei_cached(
                source_states,
                predictions[:, horizon - 1, :],
                estimator=str(args.estimator),
                bins=int(args.bins),
                names=names,
                horizon_dir=horizon_dir,
                chunk_size=int(args.joint_chunk_size),
                resume=bool(args.resume),
            )
            np.save(signed_path, signed)
            np.save(horizon_dir / "conditioned_ei_positive.npy", np.maximum(signed, 0.0))
            save_matrix_csv(pairwise, names, horizon_dir / "pairwise_ei_matrix.csv")
            save_matrix_csv(signed, names, horizon_dir / "conditioned_ei_signed.csv")
            save_matrix_csv(np.maximum(signed, 0.0), names, horizon_dir / "conditioned_ei_positive.csv")
            edges.to_csv(horizon_dir / "conditioned_ei_edges.csv", index=False)

        signed_list.append(signed)
        if one_step_positive is None:
            one_step_positive = np.maximum(signed, 0.0)

        if horizon not in checkpoint_set:
            continue

        print(f"[checkpoint H={horizon}] scoring and comparing top-5", flush=True)
        signed_mats_so_far = np.stack(signed_list, axis=0)
        total = cumulative_positive_matrix(signed_mats_so_far, horizon)
        gateway, mediator = compute_multistep_scores(total, one_step_positive, names)
        checkpoint_dir = result_dir / f"checkpoint_H{horizon:03d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        gateway.to_csv(checkpoint_dir / "gateway_scores.csv", index=False)
        mediator.to_csv(checkpoint_dir / "mediator_scores.csv", index=False)
        save_matrix_csv(total, names, checkpoint_dir / "total_conditioned_ei_matrix.csv")
        overlaps = overlap_summary(gateway, mediator, reference_gateway, reference_mediator)
        current_top = {
            "ace": top_indices(gateway, "ace", k=5),
            "acs": top_indices(gateway, "acs", k=5),
            "amce": top_indices(mediator, "amce", k=5),
        }
        matched = checkpoint_top_sets_match(current_top, reference_top)
        checkpoint_rows.append(
            {
                "horizon": int(horizon),
                "top5": current_top,
                "overlap": overlaps,
                "matched": bool(matched),
            }
        )
        final_horizon = int(horizon)
        final_gateway = gateway
        final_mediator = mediator
        if matched:
            stopped_early = True
            break

    assert final_gateway is not None and final_mediator is not None
    signed_mats = np.stack(signed_list, axis=0)
    np.save(result_dir / "conditioned_ei_signed_by_horizon.npy", signed_mats)
    final_dir = result_dir / f"checkpoint_H{final_horizon:03d}"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_gateway.to_csv(result_dir / "gateway_scores.csv", index=False)
    final_mediator.to_csv(result_dir / "mediator_scores.csv", index=False)
    correlations = rank_correlations(final_gateway, final_mediator, reference_gateway, reference_mediator)
    overlaps = overlap_summary(final_gateway, final_mediator, reference_gateway, reference_mediator)
    component_maps = np.load(Path(args.component_maps).expanduser())["component_maps"]
    nodes = build_node_frame(component_maps, final_gateway, final_mediator)
    nodes.to_csv(result_dir / "map_nodes.csv", index=False)
    figure_path = Path(args.asset_dir).expanduser() / "runge_multistep_conditioned_ei_map.png"
    plot_three_panel_map(nodes, figure_path, save_svg=True, save_pdf=True)
    manifest = {
        "config": _jsonable_config(config),
        "n_components": int(n_components),
        "max_horizon_requested": int(args.max_horizon),
        "computed_horizons": int(max_horizon),
        "checkpoints": checkpoints,
        "final_horizon": int(final_horizon),
        "stopped_early": bool(stopped_early),
        "checkpoint_rows": checkpoint_rows,
        "rank_correlations": correlations,
        "overlap_summary": overlaps,
        "reference_gateway": str(Path(args.reference_gateway).expanduser()),
        "reference_mediator": str(Path(args.reference_mediator).expanduser()),
        "figure": str(figure_path),
        **model_info,
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(checkpoint_rows).to_json(result_dir / "checkpoint_summary.jsonl", orient="records", lines=True)
    append_report_section(
        Path(args.report).expanduser(),
        figure_path=figure_path,
        final_horizon=final_horizon,
        stopped_early=stopped_early,
        checkpoint_rows=checkpoint_rows,
        gateway=final_gateway,
        mediator=final_mediator,
        correlations=correlations,
        overlaps=overlaps,
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise-manifest", default=str(DEFAULT_PAIRWISE_MANIFEST))
    parser.add_argument("--reference-gateway", default=str(DEFAULT_REFERENCE_GATEWAY))
    parser.add_argument("--reference-mediator", default=str(DEFAULT_REFERENCE_MEDIATOR))
    parser.add_argument("--component-maps", default=str(DEFAULT_COMPONENT_MAPS))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--max-horizon", type=int, default=10)
    parser.add_argument("--checkpoints", default="auto")
    parser.add_argument("--intervention-samples", type=int, default=4096)
    parser.add_argument("--estimator", choices=["tm", "discrete"], default="tm")
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--source-mode", choices=["latest", "history"], default="latest")
    parser.add_argument("--n-components", type=int, default=0)
    parser.add_argument("--joint-chunk-size", type=int, default=60)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest = run_experiment(args)
    print(json.dumps({"result_dir": str(Path(args.result_dir).expanduser()), "final_horizon": manifest["final_horizon"], "stopped_early": manifest["stopped_early"]}, indent=2))


if __name__ == "__main__":
    main()
