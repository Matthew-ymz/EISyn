#!/usr/bin/env python3
"""Exhaustive, batched degree-3 TM scoring for Runge hyperedges."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import os
import resource
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from exp.TM.transport_map_density import (
    LOG_2,
    _polynomial_design,
    _polynomial_exponents,
    estimate_mutual_information_transport_map,
)

DEFAULT_PAIRWISE_MANIFEST = (
    ROOT
    / "results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/pairwise_mlp_tm_ei_path_effects/manifest.json"
)
DEFAULT_ROLLOUT = (
    ROOT
    / "results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/"
    "multistep_conditioned_ei_tm_forced_edges/rollout_predictions_H060_n4096.npy"
)
DEFAULT_OLD_RERANK_DIR = (
    ROOT
    / "results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/"
    "multistep_conditioned_ei_tm_targeted"
)
DEFAULT_RESULT_DIR = (
    ROOT
    / "results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/"
    "multistep_conditioned_ei_tm_exhaustive"
)
SCHEMA_VERSION = 1


def enumerate_cross_target_candidates(n_components: int) -> np.ndarray:
    """Enumerate canonical source pairs crossed with targets outside the pair."""

    rows = [
        (source_a, source_b, target)
        for source_a, source_b in itertools.combinations(range(int(n_components)), 2)
        for target in range(int(n_components))
        if target not in (source_a, source_b)
    ]
    return np.asarray(rows, dtype=np.int16 if int(n_components) < 32768 else np.int32)


def fingerprint_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def fingerprint_file(path: str | Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_json(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _conditional_fit_stats(
    responses: np.ndarray,
    predictors: np.ndarray,
    *,
    degree: int = 3,
    ridge: float = 1e-6,
    min_scale: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit final triangular-map dimensions in a batch and return mean log density contributions."""

    y = np.asarray(responses, dtype=float)
    x = np.asarray(predictors, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim != 2 or x.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError("responses and predictors must be matching 2D arrays.")
    mean = x.mean(axis=0) if x.shape[1] else np.empty(0, dtype=float)
    scale = x.std(axis=0, ddof=1) if x.shape[1] else np.empty(0, dtype=float)
    scale = np.where(scale > float(min_scale), scale, 1.0)
    exponents = _polynomial_exponents(x.shape[1], int(degree))
    design = _polynomial_design(x, exponents=exponents, mean=mean, scale=scale)
    gram = design.T @ design + float(ridge) * np.eye(design.shape[1], dtype=float)
    gram[0, 0] -= float(ridge)
    coefficients = np.linalg.solve(gram, design.T @ y)
    residuals = y - design @ coefficients
    empirical_scales = residuals.std(axis=0, ddof=1)
    if y.shape[1] > 1:
        for column in np.flatnonzero(empirical_scales <= float(min_scale)):
            coefficients[:, column] = np.linalg.solve(gram, design.T @ y[:, column])
            residuals[:, column] = y[:, column] - design @ coefficients[:, column]
        empirical_scales = residuals.std(axis=0, ddof=1)
    residual_scales = np.maximum(empirical_scales, float(min_scale))
    standardized = residuals / residual_scales[None, :]
    mean_log_prob = (
        -0.5 * (np.log(2.0 * np.pi) + np.mean(standardized**2, axis=0))
        - np.log(residual_scales)
    )
    return mean_log_prob, empirical_scales <= float(min_scale)


def _conditional_mean_log_prob(
    responses: np.ndarray,
    predictors: np.ndarray,
    *,
    degree: int = 3,
    ridge: float = 1e-6,
    min_scale: float = 1e-8,
) -> np.ndarray:
    return _conditional_fit_stats(
        responses,
        predictors,
        degree=degree,
        ridge=ridge,
        min_scale=min_scale,
    )[0]


@dataclass(frozen=True)
class SourceDensityCache:
    source_hash: str
    degree: int
    ridge: float
    min_scale: float
    marginal_log_prob: np.ndarray
    marginal_bound: np.ndarray
    pair_conditional_log_prob: np.ndarray
    pair_conditional_bound: np.ndarray


def prepare_source_cache(
    sources: np.ndarray,
    *,
    degree: int = 3,
    ridge: float = 1e-6,
    min_scale: float = 1e-8,
) -> SourceDensityCache:
    source_values = np.asarray(sources, dtype=float)
    if source_values.ndim != 2:
        raise ValueError("sources must be a [samples, components] array.")
    empty = np.empty((source_values.shape[0], 0), dtype=float)
    marginal, marginal_bound = _conditional_fit_stats(
        source_values,
        empty,
        degree=degree,
        ridge=ridge,
        min_scale=min_scale,
    )
    n_components = source_values.shape[1]
    pair_log_prob = np.full((n_components, n_components), np.nan, dtype=float)
    pair_bound = np.zeros((n_components, n_components), dtype=bool)
    for source_a in range(n_components - 1):
        source_b_indices = list(range(source_a + 1, n_components))
        values, bounds = _conditional_fit_stats(
            source_values[:, source_b_indices],
            source_values[:, [source_a]],
            degree=degree,
            ridge=ridge,
            min_scale=min_scale,
        )
        pair_log_prob[source_a, source_b_indices] = values
        pair_bound[source_a, source_b_indices] = bounds
    return SourceDensityCache(
        source_hash=fingerprint_array(source_values),
        degree=int(degree),
        ridge=float(ridge),
        min_scale=float(min_scale),
        marginal_log_prob=marginal,
        marginal_bound=marginal_bound,
        pair_conditional_log_prob=pair_log_prob,
        pair_conditional_bound=pair_bound,
    )


def score_target_degree3_tm(
    sources: np.ndarray,
    target: np.ndarray,
    *,
    target_index: int,
    degree: int = 3,
    ridge: float = 1e-6,
    min_scale: float = 1e-8,
    source_cache: SourceDensityCache | None = None,
) -> pd.DataFrame:
    """Score every canonical cross-target source pair for one target vector."""

    source_values = np.asarray(sources, dtype=float)
    target_values = np.asarray(target, dtype=float)
    if target_values.ndim == 1:
        target_values = target_values[:, None]
    if source_values.ndim != 2 or target_values.shape != (source_values.shape[0], 1):
        raise ValueError("sources must be [samples, components] and target must be one column.")
    n_components = source_values.shape[1]
    target_local = int(target_index)
    if not 0 <= target_local < n_components:
        raise ValueError("target_index is outside the source component range.")

    cache = source_cache or prepare_source_cache(
        source_values,
        degree=degree,
        ridge=ridge,
        min_scale=min_scale,
    )
    if (
        cache.source_hash != fingerprint_array(source_values)
        or cache.degree != int(degree)
        or cache.ridge != float(ridge)
        or cache.min_scale != float(min_scale)
    ):
        raise ValueError("source_cache does not match sources or estimator configuration.")
    source_marginal = cache.marginal_log_prob
    source_marginal_bound = cache.marginal_bound
    source_given_target, source_given_target_bound = _conditional_fit_stats(
        source_values,
        target_values,
        degree=degree,
        ridge=ridge,
        min_scale=min_scale,
    )
    raw_single = (source_given_target - source_marginal) / LOG_2
    clipped_single = np.maximum(raw_single, 0.0)

    rows: list[dict[str, float | int]] = []
    for source_a in range(n_components):
        source_b_indices = [index for index in range(source_a + 1, n_components) if index != target_local]
        if source_a == target_local or not source_b_indices:
            continue
        source_b_values = source_values[:, source_b_indices]
        source_b_given_a = cache.pair_conditional_log_prob[source_a, source_b_indices]
        source_b_given_a_bound = cache.pair_conditional_bound[source_a, source_b_indices]
        source_b_given_target_a, source_b_given_target_a_bound = _conditional_fit_stats(
            source_b_values,
            np.column_stack([target_values, source_values[:, source_a]]),
            degree=degree,
            ridge=ridge,
            min_scale=min_scale,
        )
        raw_joint = (
            source_given_target[source_a]
            + source_b_given_target_a
            - source_marginal[source_a]
            - source_b_given_a
        ) / LOG_2
        joint_ei = np.maximum(raw_joint, 0.0)
        for offset, source_b in enumerate(source_b_indices):
            if bool(
                source_marginal_bound[source_a]
                or source_marginal_bound[source_b]
                or source_given_target_bound[source_a]
                or source_given_target_bound[source_b]
                or source_b_given_a_bound[offset]
                or source_b_given_target_a_bound[offset]
            ):
                raw_a = float(
                    estimate_mutual_information_transport_map(
                        source_values[:, [source_a]], target_values, degree=degree
                    )["mi_hat"]
                )
                raw_b = float(
                    estimate_mutual_information_transport_map(
                        source_values[:, [source_b]], target_values, degree=degree
                    )["mi_hat"]
                )
                raw_pair = float(
                    estimate_mutual_information_transport_map(
                        source_values[:, [source_a, source_b]], target_values, degree=degree
                    )["mi_hat"]
                )
                ei_a = max(0.0, raw_a)
                ei_b = max(0.0, raw_b)
                pair_ei = max(0.0, raw_pair)
            else:
                raw_a = float(raw_single[source_a])
                raw_b = float(raw_single[source_b])
                raw_pair = float(raw_joint[offset])
                ei_a = float(clipped_single[source_a])
                ei_b = float(clipped_single[source_b])
                pair_ei = float(joint_ei[offset])
            rows.append(
                {
                    "source_a": int(source_a),
                    "source_b": int(source_b),
                    "target": target_local,
                    "raw_ei_a": raw_a,
                    "raw_ei_b": raw_b,
                    "raw_joint_ei": raw_pair,
                    "ei_a": ei_a,
                    "ei_b": ei_b,
                    "joint_ei": pair_ei,
                    "delta2_tm": float(pair_ei - ei_a - ei_b),
                }
            )
    return pd.DataFrame(rows)


def score_horizon_degree3_tm(
    sources: np.ndarray,
    targets: np.ndarray,
    *,
    degree: int = 3,
    ridge: float = 1e-6,
    min_scale: float = 1e-8,
) -> pd.DataFrame:
    source_values = np.asarray(sources, dtype=float)
    target_values = np.asarray(targets, dtype=float)
    if source_values.ndim != 2 or target_values.shape != source_values.shape:
        raise ValueError("sources and targets must be matching [samples, components] arrays.")
    source_cache = prepare_source_cache(
        source_values,
        degree=degree,
        ridge=ridge,
        min_scale=min_scale,
    )
    frames = [
        score_target_degree3_tm(
            source_values,
            target_values[:, [target]],
            target_index=target,
            degree=degree,
            ridge=ridge,
            min_scale=min_scale,
            source_cache=source_cache,
        )
        for target in range(source_values.shape[1])
    ]
    return pd.concat(frames, ignore_index=True).sort_values(
        ["source_a", "source_b", "target"], kind="mergesort", ignore_index=True
    )


CHUNK_COLUMNS = (
    "source_a",
    "source_b",
    "target",
    "raw_ei_a",
    "raw_ei_b",
    "raw_joint_ei",
    "ei_a",
    "ei_b",
    "joint_ei",
    "delta2_tm",
)
RANKING_COLUMNS = (*CHUNK_COLUMNS, "tm_rank")


def _write_npz_atomic(path: str | Path, payload: dict[str, np.ndarray]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=f".{output.stem}.", suffix=".npz", dir=output.parent, delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(temporary, **payload)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def write_chunk_atomic(path: str | Path, frame: pd.DataFrame, metadata: dict[str, object]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    missing = [column for column in CHUNK_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"chunk frame is missing columns: {missing}")
    payload = {column: frame[column].to_numpy() for column in CHUNK_COLUMNS}
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    return _write_npz_atomic(output, payload)


def load_valid_chunk(path: str | Path, *, expected_metadata: dict[str, object]) -> pd.DataFrame | None:
    chunk = Path(path)
    if not chunk.exists():
        return None
    try:
        with np.load(chunk, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            if metadata != expected_metadata:
                return None
            arrays = {column: np.asarray(payload[column]) for column in CHUNK_COLUMNS}
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        return None
    frame = pd.DataFrame(arrays)
    target = int(expected_metadata["target"])
    n_components = int(expected_metadata["n_components"])
    expected_count = (n_components - 1) * (n_components - 2) // 2
    if len(frame) != expected_count:
        return None
    numeric = frame.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric).all():
        return None
    if not bool(
        (frame["source_a"] < frame["source_b"]).all()
        and (frame["source_a"] != target).all()
        and (frame["source_b"] != target).all()
        and (frame["target"] == target).all()
    ):
        return None
    keys = frame[["source_a", "source_b", "target"]]
    if keys.duplicated().any():
        return None
    expected_keys = pd.DataFrame(
        [(a, b, target) for a, b in itertools.combinations(range(n_components), 2) if target not in (a, b)],
        columns=["source_a", "source_b", "target"],
    ).astype(keys.dtypes.to_dict())
    if not keys.reset_index(drop=True).equals(expected_keys):
        return None
    if fingerprint_array(keys.to_numpy(dtype=np.int16)) != expected_metadata.get("candidate_order_hash"):
        return None
    return frame


def write_ranking_atomic(path: str | Path, frame: pd.DataFrame, metadata: dict[str, object]) -> Path:
    ranked = frame.sort_values("delta2_tm", ascending=False, kind="mergesort", ignore_index=True).copy()
    ranked["tm_rank"] = np.arange(1, len(ranked) + 1)
    payload = {column: ranked[column].to_numpy() for column in RANKING_COLUMNS}
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    return _write_npz_atomic(path, payload)


def load_valid_ranking(path: str | Path, *, expected_metadata: dict[str, object]) -> pd.DataFrame | None:
    try:
        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            if metadata != expected_metadata:
                return None
            arrays = {column: np.asarray(payload[column]) for column in RANKING_COLUMNS}
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if len({len(values) for values in arrays.values()}) != 1:
        return None
    frame = pd.DataFrame(arrays)
    if len(frame) != int(expected_metadata["candidate_count"]):
        return None
    if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
        return None
    if frame["tm_rank"].tolist() != list(range(1, len(frame) + 1)):
        return None
    if not frame["delta2_tm"].is_monotonic_decreasing:
        return None
    keys = frame[["source_a", "source_b", "target"]].to_numpy(dtype=np.int16)
    if fingerprint_array(keys) != expected_metadata.get("candidate_order_hash"):
        return None
    return frame


def validate_h1_gate(
    path: str | Path,
    *,
    expected_input_fingerprint: str,
    expected_estimator_fingerprint: str,
    expected_candidate_count: int = 102660,
) -> bool:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    required_truthy = ("all_passed", "finite", "rank_diagnostics_complete")
    return bool(
        payload.get("input_fingerprint") == str(expected_input_fingerprint)
        and payload.get("estimator_fingerprint") == str(expected_estimator_fingerprint)
        and all(bool(payload.get(key)) for key in required_truthy)
        and int(payload.get("candidate_count", -1)) == int(expected_candidate_count)
        and float(payload.get("max_abs_error", float("inf"))) <= 1e-8
        and float(payload.get("runtime_seconds", 0.0)) > 0.0
        and float(payload.get("peak_memory_mb", 0.0)) > 0.0
    )


def parse_horizons(text: str) -> list[int]:
    values: list[int] = []
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            values.extend(range(int(left), int(right) + 1))
        else:
            values.append(int(item))
    if not values:
        raise ValueError("at least one horizon is required.")
    return sorted(dict.fromkeys(values))


def load_rollout_builder():
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from run_runge_forced_tm_edge_trends import build_rollout_inputs

    return build_rollout_inputs


def _controlled_inputs(args: argparse.Namespace, max_horizon: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    build_rollout_inputs = load_rollout_builder()

    rollout_path = Path(args.rollout_path).expanduser().resolve()
    namespace = argparse.Namespace(
        pairwise_manifest=str(Path(args.pairwise_manifest).expanduser()),
        intervention_samples=int(args.intervention_samples),
        source_mode=str(args.source_mode),
        result_dir=str(rollout_path.parent),
        resume=True,
    )
    source_states, predictions, names, model_info = build_rollout_inputs(namespace, int(max_horizon))
    exact_rollout = np.load(rollout_path, mmap_mode="r")
    if predictions.shape != exact_rollout.shape or fingerprint_array(predictions) != fingerprint_array(exact_rollout):
        raise RuntimeError("reconstructed rollout does not match the exact old top-1000 rollout cache.")
    sources = np.column_stack(source_states)
    source_hash = fingerprint_array(sources)
    result_root = Path(args.result_dir).expanduser().resolve()
    source_path = result_root / f"source_samples_n{sources.shape[0]}.npy"
    prior_manifest = result_root / "input_manifest.json"
    if source_path.exists():
        saved_sources = np.load(source_path, mmap_mode="r")
        if saved_sources.shape != sources.shape or fingerprint_array(saved_sources) != source_hash:
            raise RuntimeError("reconstructed source samples do not match the persisted exhaustive-run source artifact.")
    else:
        if prior_manifest.exists():
            prior_identity = json.loads(prior_manifest.read_text(encoding="utf-8"))
            if prior_identity.get("sources_hash") != source_hash:
                raise RuntimeError("reconstructed source samples do not match the prior exhaustive-run manifest.")
        result_root.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(prefix=f".{source_path.stem}.", suffix=".npy", dir=result_root, delete=False)
        temporary = Path(handle.name)
        handle.close()
        try:
            np.save(temporary, sources)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, source_path)
        finally:
            temporary.unlink(missing_ok=True)
    model_files = [Path(path) for path in model_info.get("model_caches", [])]
    model_hashes = {str(path): fingerprint_file(path) for path in model_files}
    identity = {
        "sources_hash": source_hash,
        "rollout_hash": fingerprint_array(exact_rollout),
        "rollout_path": str(rollout_path),
        "pairwise_manifest": str(Path(args.pairwise_manifest).expanduser().resolve()),
        "model_hashes": model_hashes,
        "model_info": model_info,
        "intervention_samples": int(args.intervention_samples),
        "source_mode": str(args.source_mode),
        "names": list(names),
    }
    identity["input_fingerprint"] = fingerprint_json(identity)
    return sources, np.asarray(exact_rollout), identity


def _chunk_metadata(
    identity: dict[str, object],
    *,
    horizon: int,
    target: int,
    n_components: int,
    sample_count: int,
    degree: int,
    ridge: float,
    min_scale: float,
) -> dict[str, object]:
    pairs = np.asarray(
        [(a, b, target) for a, b in itertools.combinations(range(n_components), 2) if target not in (a, b)],
        dtype=np.int16,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "input_fingerprint": identity["input_fingerprint"],
        "sources_hash": identity["sources_hash"],
        "rollout_hash": identity["rollout_hash"],
        "horizon": int(horizon),
        "target": int(target),
        "sample_count": int(sample_count),
        "n_components": int(n_components),
        "degree": int(degree),
        "ridge": float(ridge),
        "min_scale": float(min_scale),
        "source_mode": identity["source_mode"],
        "candidate_order_hash": fingerprint_array(pairs),
    }


def _legacy_validation(frame: pd.DataFrame, sources: np.ndarray, targets: np.ndarray, *, sample_count: int) -> float:
    if frame.empty:
        return float("inf")
    indices = np.linspace(0, len(frame) - 1, num=min(int(sample_count), len(frame)), dtype=int)
    errors: list[float] = []
    for row in frame.iloc[indices].itertuples(index=False):
        target = targets[:, [int(row.target)]]
        raw_a = float(estimate_mutual_information_transport_map(sources[:, [int(row.source_a)]], target, degree=3)["mi_hat"])
        raw_b = float(estimate_mutual_information_transport_map(sources[:, [int(row.source_b)]], target, degree=3)["mi_hat"])
        raw_joint = float(
            estimate_mutual_information_transport_map(
                sources[:, [int(row.source_a), int(row.source_b)]], target, degree=3
            )["mi_hat"]
        )
        expected = np.asarray(
            [raw_a, raw_b, raw_joint, max(0.0, raw_joint) - max(0.0, raw_a) - max(0.0, raw_b)]
        )
        actual = np.asarray([row.raw_ei_a, row.raw_ei_b, row.raw_joint_ei, row.delta2_tm])
        errors.append(float(np.max(np.abs(actual - expected))))
    return max(errors, default=0.0)


def _rank_diagnostics(frame: pd.DataFrame, old_path: Path) -> dict[str, object]:
    ranked = frame.sort_values("delta2_tm", ascending=False, kind="mergesort").reset_index(drop=True)
    ranked["exhaustive_tm_rank"] = np.arange(1, len(ranked) + 1)
    if not old_path.exists():
        return {"complete": False, "reason": f"missing {old_path}"}
    old = pd.read_csv(old_path)
    merged = old.merge(
        ranked[["source_a", "source_b", "target", "exhaustive_tm_rank", "delta2_tm"]],
        left_on=["source_a", "source_b", "target_index"],
        right_on=["source_a", "source_b", "target"],
        how="left",
        suffixes=("_old", "_exhaustive"),
    )
    old_keys = set(zip(old.source_a.astype(int), old.source_b.astype(int), old.target_index.astype(int)))
    exhaustive_top10 = ranked.head(10)
    top10_in_old = sum(
        (int(row.source_a), int(row.source_b), int(row.target)) in old_keys
        for row in exhaustive_top10.itertuples(index=False)
    )
    return {
        "complete": bool(merged["exhaustive_tm_rank"].notna().all()),
        "old_candidate_count": int(len(old)),
        "exhaustive_top10_in_old_shortlist": int(top10_in_old),
        "exhaustive_top10_missing_from_old_shortlist": int(10 - top10_in_old),
        "best_old_discrete_candidate_exhaustive_tm_rank": int(
            merged.sort_values("discrete_candidate_rank").iloc[0]["exhaustive_tm_rank"]
        ),
    }


def _peak_memory_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def run_exhaustive(args: argparse.Namespace) -> dict[str, object]:
    if int(args.degree) != 3:
        raise ValueError("this exhaustive runner is definitionally degree-3; --degree must equal 3.")
    horizons = parse_horizons(args.horizons)
    max_horizon = max(max(horizons), 60)
    sources, predictions, identity = _controlled_inputs(args, max_horizon)
    result_root = Path(args.result_dir).expanduser().resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "input_manifest.json").write_text(json.dumps(identity, indent=2, default=str), encoding="utf-8")
    estimator_config = {
        "schema_version": SCHEMA_VERSION,
        "degree": int(args.degree),
        "ridge": float(args.ridge),
        "min_scale": float(args.min_scale),
        "n_components": int(sources.shape[1]),
        "candidate_universe_hash": fingerprint_array(enumerate_cross_target_candidates(sources.shape[1])),
    }
    estimator_fingerprint = fingerprint_json(estimator_config)
    expected_candidate_count = len(enumerate_cross_target_candidates(sources.shape[1]))
    gate_path = result_root / "h1_gate.json"
    if 1 not in horizons and not validate_h1_gate(
        gate_path,
        expected_input_fingerprint=str(identity["input_fingerprint"]),
        expected_estimator_fingerprint=estimator_fingerprint,
        expected_candidate_count=expected_candidate_count,
    ):
        raise RuntimeError("H=1 fail-closed gate is missing or invalid; refusing later horizons.")
    cache = prepare_source_cache(sources, degree=args.degree, ridge=args.ridge, min_scale=args.min_scale)
    summaries: list[dict[str, object]] = []
    for horizon in horizons:
        if horizon != 1 and not validate_h1_gate(
            gate_path,
            expected_input_fingerprint=str(identity["input_fingerprint"]),
            expected_estimator_fingerprint=estimator_fingerprint,
            expected_candidate_count=expected_candidate_count,
        ):
            raise RuntimeError("H=1 gate did not pass; refusing later horizons.")
        started = time.perf_counter()
        horizon_dir = result_root / f"H{horizon:03d}"
        chunk_dir = horizon_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        targets = predictions[:, horizon - 1, :]
        reused_targets: list[int] = []

        def score_one(target: int) -> pd.DataFrame:
            metadata = _chunk_metadata(
                identity,
                horizon=horizon,
                target=target,
                n_components=sources.shape[1],
                sample_count=sources.shape[0],
                degree=args.degree,
                ridge=args.ridge,
                min_scale=args.min_scale,
            )
            path = chunk_dir / f"target_{target:03d}.npz"
            existing = load_valid_chunk(path, expected_metadata=metadata) if args.resume else None
            if existing is not None:
                reused_targets.append(target)
                return existing
            frame = score_target_degree3_tm(
                sources,
                targets[:, [target]],
                target_index=target,
                degree=args.degree,
                ridge=args.ridge,
                min_scale=args.min_scale,
                source_cache=cache,
            )
            write_chunk_atomic(path, frame, metadata)
            print(f"[H={horizon:03d}] target={target:02d} rows={len(frame)}", flush=True)
            return frame

        if int(args.workers) <= 1:
            frames = [score_one(target) for target in range(sources.shape[1])]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
                frames = list(executor.map(score_one, range(sources.shape[1])))
        frame = pd.concat(frames, ignore_index=True).sort_values(
            ["source_a", "source_b", "target"], kind="mergesort", ignore_index=True
        )
        expected_count = expected_candidate_count
        finite = bool(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all())
        if len(frame) != expected_count or not finite:
            raise RuntimeError(f"H={horizon} incomplete or non-finite: rows={len(frame)}, finite={finite}.")
        ranking_path = horizon_dir / "full_ranking.npz"
        ranked = frame.sort_values("delta2_tm", ascending=False, kind="mergesort", ignore_index=True)
        ranking_metadata = {
            "schema_version": SCHEMA_VERSION,
            "input_fingerprint": identity["input_fingerprint"],
            "estimator_fingerprint": estimator_fingerprint,
            "horizon": int(horizon),
            "candidate_count": int(len(ranked)),
            "candidate_order_hash": fingerprint_array(
                ranked[["source_a", "source_b", "target"]].to_numpy(dtype=np.int16)
            ),
        }
        write_ranking_atomic(ranking_path, ranked, ranking_metadata)
        ranked["tm_rank"] = np.arange(1, len(ranked) + 1)
        (horizon_dir / "top200.json").write_text(
            json.dumps(ranked.head(200).to_dict(orient="records"), indent=2), encoding="utf-8"
        )
        elapsed = time.perf_counter() - started
        max_error = _legacy_validation(ranked, sources, targets, sample_count=args.validation_samples)
        old_path = Path(args.old_rerank_dir) / f"H{horizon:03d}_discrete_top1000_tm_rerank.csv"
        diagnostics = _rank_diagnostics(ranked, old_path)
        previous_summary_path = horizon_dir / "summary.json"
        previous_summary = {}
        if len(reused_targets) == sources.shape[1] and previous_summary_path.exists():
            previous_summary = json.loads(previous_summary_path.read_text(encoding="utf-8"))
        summary = {
            "horizon": int(horizon),
            "input_fingerprint": identity["input_fingerprint"],
            "estimator_fingerprint": estimator_fingerprint,
            "ranking_metadata": ranking_metadata,
            "candidate_count": int(len(frame)),
            "finite": finite,
            "max_abs_error": float(max_error),
            "runtime_seconds": float(previous_summary.get("runtime_seconds", elapsed)),
            "peak_memory_mb": float(previous_summary.get("peak_memory_mb", _peak_memory_mb())),
            "rank_diagnostics": diagnostics,
        }
        (horizon_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)
        if horizon == 1:
            gate = {
                **summary,
                "rank_diagnostics_complete": bool(diagnostics.get("complete")),
                "all_passed": bool(
                    max_error <= 1e-8
                    and len(frame) == expected_candidate_count
                    and finite
                    and diagnostics.get("complete")
                    and elapsed > 0.0
                    and _peak_memory_mb() > 0.0
                ),
            }
            gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
            if not validate_h1_gate(
                gate_path,
                expected_input_fingerprint=str(identity["input_fingerprint"]),
                expected_estimator_fingerprint=estimator_fingerprint,
                expected_candidate_count=expected_candidate_count,
            ):
                raise RuntimeError("H=1 gate failed; refusing later horizons.")
    return {"input_fingerprint": identity["input_fingerprint"], "summaries": summaries}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise-manifest", default=str(DEFAULT_PAIRWISE_MANIFEST))
    parser.add_argument("--rollout-path", default=str(DEFAULT_ROLLOUT))
    parser.add_argument("--old-rerank-dir", default=str(DEFAULT_OLD_RERANK_DIR))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--horizons", default="1")
    parser.add_argument("--intervention-samples", type=int, default=4096)
    parser.add_argument("--source-mode", default="latest", choices=["latest", "history"])
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--min-scale", type=float, default=1e-8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--validation-samples", type=int, default=24)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    result = run_exhaustive(build_arg_parser().parse_args())
    print(json.dumps(result, indent=2), flush=True)


__all__: Sequence[str] = (
    "enumerate_cross_target_candidates",
    "fingerprint_array",
    "load_valid_chunk",
    "load_valid_ranking",
    "load_rollout_builder",
    "prepare_source_cache",
    "score_horizon_degree3_tm",
    "score_target_degree3_tm",
    "validate_h1_gate",
    "write_chunk_atomic",
    "write_ranking_atomic",
)


if __name__ == "__main__":
    main()
