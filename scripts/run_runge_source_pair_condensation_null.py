#!/usr/bin/env python3
"""Smoke-test source-pair condensation against matched random VAR(4) nulls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm.auto import tqdm

from scripts.run_runge_exhaustive_degree3_tm import (
    DEFAULT_PAIRWISE_MANIFEST,
    DEFAULT_RESULT_DIR as DEFAULT_HYBRID_RANKING_DIR,
    load_valid_ranking,
    prepare_source_cache,
    score_target_degree3_tm,
)
from scripts.run_runge_multistep_conditioned_ei import config_from_manifest
from scripts.run_runge_pairwise_mlp_ei import (
    build_lagged_dataset,
    fit_ridge_linear_map,
    load_component_scores,
    sample_max_entropy_features,
    split_temporal_arrays,
)


DEFAULT_OUTPUT_DIR = ROOT / "results/runge_source_pair_condensation_null_smoke"
DEFAULT_FIGURE_BASE = ROOT / "fig/runge_source_pair_condensation_null_smoke"
DEFAULT_HORIZONS = (1, 10, 20, 60)
DEFAULT_NULL_SEEDS = (42000, 42001, 42002, 42003, 42004)
DEFAULT_TOP_KS = (50, 100, 200, 500)
PRIMARY_TOP_K = 200
SCHEMA_VERSION = 1


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def companion_matrix(weight: np.ndarray, *, n_components: int, lag: int) -> np.ndarray:
    matrix = np.asarray(weight, dtype=float)
    expected = (int(n_components), int(n_components) * int(lag))
    if matrix.shape != expected:
        raise ValueError(f"weight must have shape {expected}.")
    # Raw feature blocks run oldest -> latest; companion blocks run latest -> oldest.
    blocks = [
        matrix[:, (int(lag) - 1 - index) * int(n_components) : (int(lag) - index) * int(n_components)]
        for index in range(int(lag))
    ]
    companion = np.zeros((int(lag) * int(n_components), int(lag) * int(n_components)), dtype=float)
    companion[: int(n_components), :] = np.concatenate(blocks, axis=1)
    if int(lag) > 1:
        companion[int(n_components) :, : -int(n_components)] = np.eye(
            (int(lag) - 1) * int(n_components), dtype=float
        )
    return companion


def spectral_radius(weight: np.ndarray, *, n_components: int, lag: int) -> float:
    values = np.linalg.eigvals(companion_matrix(weight, n_components=n_components, lag=lag))
    return float(np.max(np.abs(values)))


def calibrate_spectral_radius(
    weight: np.ndarray,
    *,
    n_components: int,
    lag: int,
    target_spectral_radius: float,
) -> tuple[np.ndarray, float, float]:
    matrix = np.asarray(weight, dtype=float)

    def objective(log_scale: float) -> float:
        radius = spectral_radius(
            math.exp(float(log_scale)) * matrix,
            n_components=n_components,
            lag=lag,
        )
        return float((math.log(max(radius, 1e-12)) - math.log(float(target_spectral_radius))) ** 2)

    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=(-3.0, 3.0),
        options={"xatol": 1e-8, "maxiter": 100},
    )
    scale = float(math.exp(float(result.x)))
    calibrated = scale * matrix
    achieved = spectral_radius(calibrated, n_components=n_components, lag=lag)
    if abs(achieved - float(target_spectral_radius)) > 1e-5:
        raise RuntimeError(
            "Spectral calibration failed: "
            f"target={target_spectral_radius}, achieved={achieved}, scale={scale}."
        )
    return calibrated, scale, achieved


def spectrum_descriptors(
    weight: np.ndarray,
    *,
    n_components: int,
    lag: int,
    horizon: int = 60,
) -> dict[str, float | int]:
    companion = companion_matrix(weight, n_components=n_components, lag=lag)
    eigenvalues, eigenvectors = np.linalg.eig(companion)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    magnitudes = np.abs(eigenvalues)
    radius = float(magnitudes[0])
    normalized = magnitudes / max(radius, 1e-12)
    spectral_mass = normalized ** (2 * int(horizon))
    effective_modes = float(
        spectral_mass.sum() ** 2 / max(float(np.sum(spectral_mass**2)), 1e-30)
    )
    observed_leading = np.abs(eigenvectors[: int(n_components), 0]) ** 2
    leading_support = float(
        observed_leading.sum() ** 2 / max(float(np.sum(observed_leading**2)), 1e-30)
    )
    transfer = np.linalg.matrix_power(companion, int(horizon))[: int(n_components), : int(n_components)]
    singular_values = np.linalg.svd(transfer, compute_uv=False)
    transfer_mass = singular_values**2
    transfer_effective_rank = float(
        transfer_mass.sum() ** 2 / max(float(np.sum(transfer_mass**2)), 1e-30)
    )
    commutator = companion.T @ companion - companion @ companion.T
    nonnormality = float(np.linalg.norm(commutator) / max(np.linalg.norm(companion) ** 2, 1e-30))
    return {
        "spectral_radius": radius,
        "spectral_gap_fraction": float(1.0 - magnitudes[1] / max(radius, 1e-12)),
        "slow_mode_count_90pct": int(np.sum(magnitudes >= 0.90 * radius)),
        "slow_mode_count_95pct": int(np.sum(magnitudes >= 0.95 * radius)),
        f"spectral_effective_modes_h{int(horizon)}": effective_modes,
        "complex_eigenvalue_fraction": float(np.mean(np.abs(np.imag(eigenvalues)) > 1e-8)),
        "leading_observed_mode_support": leading_support,
        f"latest_to_future_transfer_effective_rank_h{int(horizon)}": transfer_effective_rank,
        "companion_nonnormality": nonnormality,
    }


def lag_blocks(weight: np.ndarray, *, n_components: int, lag: int) -> list[np.ndarray]:
    matrix = np.asarray(weight, dtype=float)
    return [
        matrix[:, (int(lag) - 1 - index) * int(n_components) : (int(lag) - index) * int(n_components)]
        for index in range(int(lag))
    ]


def fixed_point(weight: np.ndarray, bias: np.ndarray, *, n_components: int, lag: int) -> np.ndarray:
    transition = np.eye(int(n_components), dtype=float) - sum(
        lag_blocks(weight, n_components=n_components, lag=lag)
    )
    return np.linalg.solve(transition, np.asarray(bias, dtype=float))


def fit_linear_backbone(pairwise_manifest: Path) -> dict[str, object]:
    manifest = json.loads(pairwise_manifest.read_text(encoding="utf-8"))
    config = config_from_manifest(pairwise_manifest)
    frame = load_component_scores(config.component_scores)
    features, targets = build_lagged_dataset(frame, lag=int(config.lag), horizon=1)
    splits = split_temporal_arrays(
        features,
        targets,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
    )
    x_train, y_train = splits["train"]
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std = np.where(x_std > 1e-8, x_std, 1.0)
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std = np.where(y_std > 1e-8, y_std, 1.0)
    x_scaled = (x_train - x_mean) / x_std
    y_scaled = (y_train - y_mean) / y_std

    alphas = [float(value) for value in config.ensemble_ridge_alphas]
    if not alphas:
        alphas = [float(config.ridge_alpha)]
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for alpha in alphas:
        weight, bias = fit_ridge_linear_map(x_scaled, y_scaled, alpha=alpha)
        weights.append(np.asarray(weight, dtype=float))
        biases.append(np.asarray(bias, dtype=float))

    blend = dict(manifest.get("linear_blend", {}))
    mlp_weight = float(blend.get("mlp_weight", 1.0))
    ridge_weight = float(blend.get("ridge_weight", 0.0))
    ridge_alpha = float(blend.get("ridge_alpha", config.ridge_alpha))
    if ridge_alpha not in alphas:
        extra_weight, extra_bias = fit_ridge_linear_map(x_scaled, y_scaled, alpha=ridge_alpha)
        ridge_component_weight = np.asarray(extra_weight, dtype=float)
        ridge_component_bias = np.asarray(extra_bias, dtype=float)
    else:
        index = alphas.index(ridge_alpha)
        ridge_component_weight = weights[index]
        ridge_component_bias = biases[index]

    scaled_weight = mlp_weight * np.mean(weights, axis=0) + ridge_weight * ridge_component_weight
    scaled_bias = mlp_weight * np.mean(biases, axis=0) + ridge_weight * ridge_component_bias
    raw_weight = y_std[:, None] * scaled_weight / x_std[None, :]
    raw_bias = y_mean + y_std * scaled_bias - raw_weight @ x_mean
    n_components = targets.shape[1]
    radius = spectral_radius(raw_weight, n_components=n_components, lag=int(config.lag))
    center = fixed_point(raw_weight, raw_bias, n_components=n_components, lag=int(config.lag))
    intervention_features = sample_max_entropy_features(
        x_train,
        n_components=n_components,
        lag=int(config.lag),
        samples=int(config.intervention_samples),
        low_q=float(config.quantile_low),
        high_q=float(config.quantile_high),
        seed=int(config.seed),
    )
    return {
        "config": config,
        "names": list(frame.columns),
        "features": intervention_features,
        "weight": raw_weight,
        "bias": raw_bias,
        "spectral_radius": radius,
        "fixed_point": center,
        "alphas": alphas,
        "blend": {
            "mlp_weight": mlp_weight,
            "ridge_weight": ridge_weight,
            "ridge_alpha": ridge_alpha,
        },
    }


def rewire_var_coefficients(
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    n_components: int,
    lag: int,
    seed: int,
    target_spectral_radius: float,
    retained_fixed_point: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    original = np.asarray(weight, dtype=float)
    rewired = original.copy()
    rng = np.random.default_rng(int(seed))
    for lag_index in range(int(lag)):
        start = lag_index * int(n_components)
        block = rewired[:, start : start + int(n_components)].copy()
        for target in range(int(n_components)):
            off_diagonal = np.concatenate(
                [np.arange(target, dtype=int), np.arange(target + 1, int(n_components), dtype=int)]
            )
            permutation = rng.permutation(off_diagonal)
            block[target, off_diagonal] = block[target, permutation]
        rewired[:, start : start + int(n_components)] = block

    unscaled_radius = spectral_radius(rewired, n_components=n_components, lag=lag)

    calibrated, scale, achieved_radius = calibrate_spectral_radius(
        rewired,
        n_components=n_components,
        lag=lag,
        target_spectral_radius=target_spectral_radius,
    )
    center = (
        np.asarray(retained_fixed_point, dtype=float)
        if retained_fixed_point is not None
        else fixed_point(original, bias, n_components=n_components, lag=lag)
    )
    calibrated_bias = center - sum(
        lag_blocks(calibrated, n_components=n_components, lag=lag)
    ) @ center
    return calibrated, calibrated_bias, {
        "seed": int(seed),
        "target_spectral_radius": float(target_spectral_radius),
        "unscaled_spectral_radius": float(unscaled_radius),
        "spectral_scale": scale,
        "achieved_spectral_radius": float(achieved_radius),
        "fixed_point_norm": float(np.linalg.norm(center)),
        "coefficient_frobenius_before": float(np.linalg.norm(original)),
        "coefficient_frobenius_after": float(np.linalg.norm(calibrated)),
    }


def rollout_linear_var(
    weight: np.ndarray,
    bias: np.ndarray,
    initial_features: np.ndarray,
    *,
    n_components: int,
    lag: int,
    horizons: int,
) -> np.ndarray:
    window = np.asarray(initial_features, dtype=float).copy()
    predictions = np.empty((len(window), int(horizons), int(n_components)), dtype=float)
    for horizon in range(int(horizons)):
        next_state = window @ np.asarray(weight, dtype=float).T + np.asarray(bias, dtype=float)
        if not bool(np.isfinite(next_state).all()):
            raise RuntimeError(f"Non-finite linear rollout at horizon {horizon + 1}.")
        predictions[:, horizon, :] = next_state
        window = np.concatenate([window[:, int(n_components) :], next_state], axis=1)
    return predictions


def concentration_metrics(
    ranking: pd.DataFrame,
    *,
    top_ks: Iterable[int],
    nonnegative_tolerance: float,
) -> dict[str, object]:
    ordered = ranking.sort_values("delta2_tm", ascending=False, kind="mergesort", ignore_index=True)
    maximum_k = max(int(value) for value in top_ks)
    consumed = ordered.head(maximum_k).copy()
    values = consumed["delta2_tm"].to_numpy(dtype=float)
    significant = values < -float(nonnegative_tolerance)
    if bool(significant.any()):
        raise RuntimeError(
            "Significant Syn nonnegativity violation in consumed top-K: "
            f"minimum={float(values.min())}, tolerance={nonnegative_tolerance}, count={int(significant.sum())}."
        )
    within = (values < 0.0) & ~significant
    consumed["weight"] = np.where(within, 0.0, values)
    by_k: dict[str, object] = {}
    for top_k in sorted({int(value) for value in top_ks}):
        top = consumed.head(top_k)
        pairs = (
            top.groupby(["source_a", "source_b"], as_index=False)
            .agg(weight=("weight", "sum"), target_count=("target", "nunique"))
            .query("weight > 0")
            .sort_values("weight", ascending=False, ignore_index=True)
        )
        total = float(pairs["weight"].sum())
        probabilities = pairs["weight"].to_numpy(dtype=float) / total if total > 0.0 else np.empty(0)
        effective = float(1.0 / np.sum(probabilities**2)) if len(probabilities) else 0.0
        by_k[str(top_k)] = {
            "distinct_pair_count": int(len(pairs)),
            "effective_pair_count": effective,
            "max_pair_share": float(probabilities.max()) if len(probabilities) else 0.0,
            "top5_pair_share": float(np.sort(probabilities)[-5:].sum()) if len(probabilities) else 0.0,
            "positive_synergy_mass_bits": total,
        }
    return {
        "top_k_metrics": by_k,
        "minimum_consumed_syn_bits": float(values.min()),
        "negative_within_tolerance_count": int(within.sum()),
        "significant_nonnegativity_violation_count": 0,
        "top500": consumed,
    }


class ProgressRecorder:
    def __init__(self, path: Path, *, total: int, current: int = 0) -> None:
        self.path = path
        self.total = int(total)
        self.current = int(current)
        self.started = time.monotonic()

    def write(self, *, phase: str, message: str, metrics: dict[str, object] | None = None) -> None:
        elapsed = time.monotonic() - self.started
        rate = self.current / elapsed if elapsed > 0.0 else 0.0
        atomic_write_json(
            self.path,
            {
                "phase": phase,
                "current": self.current,
                "total": self.total,
                "unit": "target-score",
                "elapsed_seconds": elapsed,
                "eta_seconds": (self.total - self.current) / rate if rate > 0.0 else None,
                "message": message,
                "metrics": metrics or {},
                "pid": os.getpid(),
                "updated_at": time.time(),
            },
        )

    def advance(self, *, message: str, metrics: dict[str, object] | None = None) -> None:
        self.current += 1
        self.write(phase="score", message=message, metrics=metrics)


def metric_path(output_dir: Path, condition: str, horizon: int) -> Path:
    return output_dir / "metrics" / f"{condition}_H{int(horizon):03d}.json"


def top500_path(output_dir: Path, condition: str, horizon: int) -> Path:
    return output_dir / "top500" / f"{condition}_H{int(horizon):03d}.npz"


def persist_condition_metric(
    output_dir: Path,
    *,
    condition: str,
    horizon: int,
    condition_type: str,
    metric_payload: dict[str, object],
    diagnostics: dict[str, object],
) -> dict[str, object]:
    top500 = metric_payload.pop("top500")
    assert isinstance(top500, pd.DataFrame)
    top_path = top500_path(output_dir, condition, horizon)
    atomic_savez(
        top_path,
        source_a=top500["source_a"].to_numpy(dtype=np.int16),
        source_b=top500["source_b"].to_numpy(dtype=np.int16),
        target=top500["target"].to_numpy(dtype=np.int16),
        delta2_tm=top500["delta2_tm"].to_numpy(dtype=np.float64),
        tm_rank=np.arange(1, len(top500) + 1, dtype=np.int16),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "condition": condition,
        "condition_type": condition_type,
        "horizon": int(horizon),
        "diagnostics": diagnostics,
        "top500": str(top_path.resolve()),
        **metric_payload,
    }
    atomic_write_json(metric_path(output_dir, condition, horizon), payload)
    return payload


def score_prediction_condition(
    *,
    condition: str,
    condition_type: str,
    predictions: np.ndarray,
    sources: np.ndarray,
    source_cache: object,
    horizons: list[int],
    degree: int,
    ridge: float,
    min_scale: float,
    top_ks: tuple[int, ...],
    nonnegative_tolerance: float,
    output_dir: Path,
    progress: ProgressRecorder,
    bar: tqdm,
    diagnostics: dict[str, object],
    resume: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_components = sources.shape[1]
    for horizon in horizons:
        saved = metric_path(output_dir, condition, horizon)
        saved_top = top500_path(output_dir, condition, horizon)
        if resume and saved.exists() and saved_top.exists():
            rows.append(json.loads(saved.read_text(encoding="utf-8")))
            progress.current += n_components
            bar.update(n_components)
            progress.write(
                phase="score",
                message=f"Reused {condition}, H={horizon}",
                metrics={"condition": condition, "horizon": horizon, "resume": True},
            )
            continue
        target_frames: list[pd.DataFrame] = []
        for target in range(n_components):
            target_frames.append(
                score_target_degree3_tm(
                    sources,
                    predictions[:, int(horizon) - 1, [target]],
                    target_index=target,
                    degree=int(degree),
                    ridge=float(ridge),
                    min_scale=float(min_scale),
                    source_cache=source_cache,
                )
            )
            progress.advance(
                message=f"Scoring {condition}, H={horizon}, target={target + 1}/{n_components}",
                metrics={"condition": condition, "horizon": horizon, "target": target + 1},
            )
            bar.update(1)
            bar.set_postfix(condition=condition, horizon=horizon, target=target + 1)
        ranking = pd.concat(target_frames, ignore_index=True)
        ranking = ranking.sort_values("delta2_tm", ascending=False, kind="mergesort", ignore_index=True)
        metric_payload = concentration_metrics(
            ranking,
            top_ks=top_ks,
            nonnegative_tolerance=nonnegative_tolerance,
        )
        rows.append(
            persist_condition_metric(
                output_dir,
                condition=condition,
                horizon=horizon,
                condition_type=condition_type,
                metric_payload=metric_payload,
                diagnostics=diagnostics,
            )
        )
    return rows


def load_hybrid_reference(
    *,
    ranking_dir: Path,
    horizons: list[int],
    top_ks: tuple[int, ...],
    nonnegative_tolerance: float,
    output_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        horizon_dir = ranking_dir / f"H{int(horizon):03d}"
        summary = json.loads((horizon_dir / "summary.json").read_text(encoding="utf-8"))
        ranking = load_valid_ranking(
            horizon_dir / "full_ranking.npz",
            expected_metadata=summary["ranking_metadata"],
        )
        if ranking is None:
            raise RuntimeError(f"Hybrid reference ranking failed integrity validation at H={horizon}.")
        metric_payload = concentration_metrics(
            ranking,
            top_ks=top_ks,
            nonnegative_tolerance=nonnegative_tolerance,
        )
        rows.append(
            persist_condition_metric(
                output_dir,
                condition="earth_hybrid",
                horizon=horizon,
                condition_type="published_hybrid_reference",
                metric_payload=metric_payload,
                diagnostics={
                    "input_fingerprint": summary["input_fingerprint"],
                    "estimator_fingerprint": summary["estimator_fingerprint"],
                    "candidate_count": summary["candidate_count"],
                },
            )
        )
    return rows


def summarize_rows(rows: list[dict[str, object]], *, primary_top_k: int) -> dict[str, object]:
    conditions = sorted({str(row["condition"]) for row in rows})
    by_condition: dict[str, object] = {}
    for condition in conditions:
        selected = sorted(
            [row for row in rows if row["condition"] == condition],
            key=lambda item: int(item["horizon"]),
        )
        curve = []
        for row in selected:
            metrics = row["top_k_metrics"][str(int(primary_top_k))]
            curve.append({"horizon": int(row["horizon"]), **metrics})
        first = curve[0]
        last = curve[-1]
        by_condition[condition] = {
            "condition_type": selected[0]["condition_type"],
            "curve": curve,
            "h60_over_h1_distinct_pair_count": float(last["distinct_pair_count"] / first["distinct_pair_count"]),
            "h60_over_h1_effective_pair_count": float(last["effective_pair_count"] / first["effective_pair_count"]),
            "h60_minus_h1_max_pair_share": float(last["max_pair_share"] - first["max_pair_share"]),
        }
    null_conditions = [name for name in conditions if name.startswith("null_")]
    null_effective_ratios = np.asarray(
        [by_condition[name]["h60_over_h1_effective_pair_count"] for name in null_conditions], dtype=float
    )
    null_distinct_ratios = np.asarray(
        [by_condition[name]["h60_over_h1_distinct_pair_count"] for name in null_conditions], dtype=float
    )
    earth_linear = by_condition["earth_linear"]
    hybrid = by_condition["earth_hybrid"]
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "smoke_only_no_significance",
        "primary_top_k": int(primary_top_k),
        "conditions": by_condition,
        "null_screen": {
            "n_nulls": len(null_conditions),
            "effective_ratio_range": [float(null_effective_ratios.min()), float(null_effective_ratios.max())],
            "distinct_ratio_range": [float(null_distinct_ratios.min()), float(null_distinct_ratios.max())],
            "earth_linear_effective_ratio_outside_null_range": bool(
                earth_linear["h60_over_h1_effective_pair_count"] < null_effective_ratios.min()
                or earth_linear["h60_over_h1_effective_pair_count"] > null_effective_ratios.max()
            ),
            "earth_linear_distinct_ratio_outside_null_range": bool(
                earth_linear["h60_over_h1_distinct_pair_count"] < null_distinct_ratios.min()
                or earth_linear["h60_over_h1_distinct_pair_count"] > null_distinct_ratios.max()
            ),
        },
        "linear_ablation_gate": {
            "hybrid_effective_ratio": hybrid["h60_over_h1_effective_pair_count"],
            "earth_linear_effective_ratio": earth_linear["h60_over_h1_effective_pair_count"],
            "absolute_ratio_difference": float(
                abs(
                    hybrid["h60_over_h1_effective_pair_count"]
                    - earth_linear["h60_over_h1_effective_pair_count"]
                )
            ),
        },
    }


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.16, 1.04, label, transform=axis.transAxes, fontsize=8.2, fontweight="bold", va="bottom")


def plot_summary(summary: dict[str, object], *, figure_base: Path) -> list[Path]:
    configure_matplotlib()
    conditions = summary["conditions"]
    null_names = sorted(name for name in conditions if name.startswith("null_"))
    horizons = np.asarray([row["horizon"] for row in conditions["earth_hybrid"]["curve"]], dtype=float)
    colors = {"earth_hybrid": "#D97732", "earth_linear": "#356A8A", "null": "#A7B0B8"}
    labels = {"earth_hybrid": "Published hybrid", "earth_linear": "Linear ablation"}

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7), layout="constrained")
    metric_specs = (
        ("distinct_pair_count", "Source pairs retained in top-200"),
        ("effective_pair_count", "Effective source-pair count"),
        ("max_pair_share", "Maximum source-pair mass share"),
    )
    handles: list[object] = []
    handle_labels: list[str] = []
    for axis, (metric, ylabel), letter in zip(axes.flat[:3], metric_specs, ("a", "b", "c")):
        null_curves = np.asarray(
            [[row[metric] for row in conditions[name]["curve"]] for name in null_names], dtype=float
        )
        for values in null_curves:
            axis.plot(horizons, values, color=colors["null"], linewidth=0.65, alpha=0.55, zorder=1)
        null_mean = null_curves.mean(axis=0)
        null_sd = null_curves.std(axis=0, ddof=1) if len(null_curves) > 1 else np.zeros_like(null_mean)
        axis.fill_between(horizons, null_mean - null_sd, null_mean + null_sd, color=colors["null"], alpha=0.22, linewidth=0)
        null_handle = axis.plot(horizons, null_mean, color="#737D86", linewidth=1.15, linestyle="--", zorder=2)[0]
        for name in ("earth_hybrid", "earth_linear"):
            values = np.asarray([row[metric] for row in conditions[name]["curve"]], dtype=float)
            handle = axis.plot(
                horizons,
                values,
                color=colors[name],
                linewidth=1.6,
                marker="o",
                markersize=3.0,
                zorder=3,
            )[0]
            if letter == "a":
                handles.append(handle)
                handle_labels.append(labels[name])
        if letter == "a":
            handles.append(null_handle)
            handle_labels.append("Rewired VAR null mean ± SD")
        axis.set_xlabel("Forecast horizon, H")
        axis.set_ylabel(ylabel)
        axis.set_xticks(horizons)
        axis.grid(axis="y", color="#E8EBEF", linewidth=0.55)
        axis.set_ylim(bottom=0.0)
        add_panel_label(axis, letter)

    axis = axes.flat[3]
    x = np.arange(2, dtype=float)
    null_distinct = np.asarray(
        [conditions[name]["h60_over_h1_distinct_pair_count"] for name in null_names], dtype=float
    )
    null_effective = np.asarray(
        [conditions[name]["h60_over_h1_effective_pair_count"] for name in null_names], dtype=float
    )
    for offset, values in zip(x, (null_distinct, null_effective)):
        jitter = np.linspace(-0.08, 0.08, len(values))
        axis.scatter(offset + jitter, values, s=14, color=colors["null"], edgecolor="white", linewidth=0.35, zorder=2)
        axis.errorbar(
            offset,
            float(values.mean()),
            yerr=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            fmt="_",
            markersize=8,
            color="#737D86",
            linewidth=1.0,
            capsize=2.0,
            zorder=3,
        )
    for name, marker in (("earth_hybrid", "D"), ("earth_linear", "o")):
        values = [
            conditions[name]["h60_over_h1_distinct_pair_count"],
            conditions[name]["h60_over_h1_effective_pair_count"],
        ]
        axis.scatter(x, values, s=26, marker=marker, color=colors[name], edgecolor="white", linewidth=0.45, zorder=4)
    axis.axhline(1.0, color="#8E9399", linestyle=":", linewidth=0.75)
    axis.set_xticks(x, ["Distinct pairs", "Effective pairs"])
    axis.set_ylabel("H=60 / H=1 ratio")
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", color="#E8EBEF", linewidth=0.55)
    add_panel_label(axis, "d")

    fig.legend(
        handles,
        handle_labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=6.0,
        handlelength=2.0,
    )
    fig.text(0.5, -0.015, "The null ensemble is a smoke-test range, not a significance interval.", ha="center", fontsize=5.8)
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in ((".png", {"dpi": 600}), (".svg", {}), (".pdf", {})):
        path = figure_base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def parse_ints(text: str) -> list[int]:
    return sorted({int(part.strip()) for part in str(text).split(",") if part.strip()})


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pairwise_manifest = Path(args.pairwise_manifest).expanduser().resolve()
    hybrid_ranking_dir = Path(args.hybrid_ranking_dir).expanduser().resolve()
    horizons = parse_ints(args.horizons)
    null_seeds = parse_ints(args.null_seeds)
    top_ks = tuple(parse_ints(args.top_ks))
    if PRIMARY_TOP_K not in top_ks:
        raise ValueError(f"top_ks must include primary top-K={PRIMARY_TOP_K}.")

    backbone = fit_linear_backbone(pairwise_manifest)
    config = backbone["config"]
    features = np.asarray(backbone["features"], dtype=float)
    n_components = len(backbone["names"])
    lag = int(config.lag)
    if int(args.intervention_samples) != len(features):
        features = sample_max_entropy_features(
            split_temporal_arrays(
                *build_lagged_dataset(load_component_scores(config.component_scores), lag=lag, horizon=1),
                train_fraction=float(config.train_fraction),
                val_fraction=float(config.val_fraction),
            )["train"][0],
            n_components=n_components,
            lag=lag,
            samples=int(args.intervention_samples),
            low_q=float(config.quantile_low),
            high_q=float(config.quantile_high),
            seed=int(config.seed),
        )
    source_columns = [(lag - 1) * n_components + source for source in range(n_components)]
    sources = features[:, source_columns]
    existing_sources = hybrid_ranking_dir / f"source_samples_n{len(sources)}.npy"
    if existing_sources.exists():
        reference_sources = np.load(existing_sources, mmap_mode="r")
        if reference_sources.shape != sources.shape or fingerprint_array(reference_sources) != fingerprint_array(sources):
            raise RuntimeError("Linear-ablation source samples do not match the published hybrid source samples.")

    input_manifest = {
        "schema_version": SCHEMA_VERSION,
        "pairwise_manifest": str(pairwise_manifest),
        "hybrid_ranking_dir": str(hybrid_ranking_dir),
        "n_components": n_components,
        "lag": lag,
        "horizons": horizons,
        "intervention_samples": len(sources),
        "source_fingerprint": fingerprint_array(sources),
        "tm_degree": int(args.degree),
        "tm_ridge": float(args.ridge),
        "tm_min_scale": float(args.min_scale),
        "nonnegative_tolerance_bits": float(args.nonnegative_tolerance),
        "top_ks": list(top_ks),
        "null_seeds": null_seeds,
        "linear_backbone": {
            "spectral_radius": float(backbone["spectral_radius"]),
            "fixed_point_norm": float(np.linalg.norm(backbone["fixed_point"])),
            "ensemble_ridge_alphas": backbone["alphas"],
            "final_blend": backbone["blend"],
            "weight_fingerprint": fingerprint_array(backbone["weight"]),
        },
    }
    atomic_write_json(output_dir / "input_manifest.json", input_manifest)

    source_cache = prepare_source_cache(
        sources,
        degree=int(args.degree),
        ridge=float(args.ridge),
        min_scale=float(args.min_scale),
    )
    total_target_scores = (1 + len(null_seeds)) * len(horizons) * n_components
    progress = ProgressRecorder(output_dir / "live_progress.json", total=total_target_scores)
    progress.write(phase="prepare", message="Inputs validated and source TM cache prepared")

    all_rows = load_hybrid_reference(
        ranking_dir=hybrid_ranking_dir,
        horizons=horizons,
        top_ks=top_ks,
        nonnegative_tolerance=float(args.nonnegative_tolerance),
        output_dir=output_dir,
    )
    max_horizon = max(horizons)
    with tqdm(total=total_target_scores, desc="condensation null smoke", unit="target", mininterval=1.0) as bar:
        linear_predictions = rollout_linear_var(
            backbone["weight"],
            backbone["bias"],
            features,
            n_components=n_components,
            lag=lag,
            horizons=max_horizon,
        )
        all_rows.extend(
            score_prediction_condition(
                condition="earth_linear",
                condition_type="linear_backbone_ablation",
                predictions=linear_predictions,
                sources=sources,
                source_cache=source_cache,
                horizons=horizons,
                degree=int(args.degree),
                ridge=float(args.ridge),
                min_scale=float(args.min_scale),
                top_ks=top_ks,
                nonnegative_tolerance=float(args.nonnegative_tolerance),
                output_dir=output_dir,
                progress=progress,
                bar=bar,
                diagnostics={
                    "spectral_radius": float(backbone["spectral_radius"]),
                    "rollout_variance_by_horizon": {
                        str(horizon): float(np.mean(np.var(linear_predictions[:, horizon - 1, :], axis=0)))
                        for horizon in horizons
                    },
                },
                resume=bool(args.resume),
            )
        )
        null_diagnostics: list[dict[str, object]] = []
        for null_index, seed in enumerate(null_seeds):
            null_weight, null_bias, diagnostics = rewire_var_coefficients(
                backbone["weight"],
                backbone["bias"],
                n_components=n_components,
                lag=lag,
                seed=seed,
                target_spectral_radius=float(backbone["spectral_radius"]),
                retained_fixed_point=np.asarray(backbone["fixed_point"]),
            )
            null_predictions = rollout_linear_var(
                null_weight,
                null_bias,
                features,
                n_components=n_components,
                lag=lag,
                horizons=max_horizon,
            )
            diagnostics["weight_fingerprint"] = fingerprint_array(null_weight)
            diagnostics["rollout_variance_by_horizon"] = {
                str(horizon): float(np.mean(np.var(null_predictions[:, horizon - 1, :], axis=0)))
                for horizon in horizons
            }
            null_diagnostics.append(diagnostics)
            all_rows.extend(
                score_prediction_condition(
                    condition=f"null_{null_index:02d}",
                    condition_type="spectrally_matched_rewired_var4",
                    predictions=null_predictions,
                    sources=sources,
                    source_cache=source_cache,
                    horizons=horizons,
                    degree=int(args.degree),
                    ridge=float(args.ridge),
                    min_scale=float(args.min_scale),
                    top_ks=top_ks,
                    nonnegative_tolerance=float(args.nonnegative_tolerance),
                    output_dir=output_dir,
                    progress=progress,
                    bar=bar,
                    diagnostics=diagnostics,
                    resume=bool(args.resume),
                )
            )
    atomic_write_json(output_dir / "null_diagnostics.json", null_diagnostics)
    summary = summarize_rows(all_rows, primary_top_k=PRIMARY_TOP_K)
    outputs = plot_summary(summary, figure_base=Path(args.figure_base).expanduser().resolve())
    summary["figure_outputs"] = [str(path) for path in outputs]
    summary["input_manifest"] = str((output_dir / "input_manifest.json").resolve())
    summary["null_diagnostics"] = str((output_dir / "null_diagnostics.json").resolve())
    atomic_write_json(output_dir / "summary.json", summary)
    progress.current = total_target_scores
    progress.write(
        phase="complete",
        message=f"{len(null_seeds)}-null condensation smoke test complete",
        metrics={
            "linear_outside_null_effective_range": summary["null_screen"][
                "earth_linear_effective_ratio_outside_null_range"
            ],
            "linear_outside_null_distinct_range": summary["null_screen"][
                "earth_linear_distinct_ratio_outside_null_range"
            ],
        },
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise-manifest", default=str(DEFAULT_PAIRWISE_MANIFEST))
    parser.add_argument("--hybrid-ranking-dir", default=str(DEFAULT_HYBRID_RANKING_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--figure-base", default=str(DEFAULT_FIGURE_BASE))
    parser.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    parser.add_argument("--null-seeds", default=",".join(map(str, DEFAULT_NULL_SEEDS)))
    parser.add_argument("--top-ks", default=",".join(map(str, DEFAULT_TOP_KS)))
    parser.add_argument("--intervention-samples", type=int, default=4096)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--min-scale", type=float, default=1e-8)
    parser.add_argument("--nonnegative-tolerance", type=float, default=1e-10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    parsed = build_arg_parser().parse_args()
    try:
        result = run(parsed)
    except Exception as error:
        output_dir = Path(parsed.output_dir).expanduser().resolve()
        prior: dict[str, object] = {}
        status_path = output_dir / "live_progress.json"
        if status_path.exists():
            try:
                prior = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                prior = {}
        atomic_write_json(
            status_path,
            {
                "phase": "failed",
                "current": prior.get("current", 0),
                "total": prior.get("total"),
                "unit": "target-score",
                "message": f"{type(error).__name__}: {error}",
                "metrics": {},
                "pid": os.getpid(),
                "updated_at": time.time(),
            },
        )
        raise
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
