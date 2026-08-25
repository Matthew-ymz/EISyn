#!/usr/bin/env python3
"""Run a no-confounder sine-frequency MLP+PEID experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import digamma

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_granger_peid_mlp import (  # noqa: E402
    DEFAULT_FIGURE_DIR,
    DEFAULT_RESULT_DIR,
    SimConfig,
    _intervention_features,
    make_lagged_dataset,
    train_mlp_transition_model,
)

DEFAULT_RESULT_PATH = DEFAULT_RESULT_DIR / "sine_frequency_mlp_peid_sweep.json"
DEFAULT_FIGURE_PATH = DEFAULT_FIGURE_DIR / "sine_frequency_mlp_peid_sweep.png"
DEFAULT_REPORT_PATH = ROOT / "docs" / "reports" / "Sine_Frequency_MLP_PEID.md"
DEFAULT_STATUS_PATH = ROOT / "docs" / "log" / "sine_frequency_mlp_peid_progress.json"
VARIABLES = ("x", "y", "z")
SYN_NONNEGATIVE_TOLERANCE_BITS = 1e-2


def simulate_no_confounder_sine_system(
    *,
    n_samples: int,
    seed: int,
    alpha: float,
    sine_frequency: float,
    noise: float,
) -> pd.DataFrame:
    """Generate x/y/z dynamics with only the sine hyperedge into z."""

    rng = np.random.default_rng(int(seed))
    data = np.zeros((int(n_samples), len(VARIABLES)), dtype=float)
    data[0, 0] = rng.normal(0.0, 0.4)
    data[0, 1] = rng.normal(0.0, 0.4)
    data[0, 2] = rng.normal(0.0, 0.2)
    for t in range(int(n_samples) - 1):
        data[t + 1, 0] = 0.42 * data[t, 0] + rng.normal(0.0, 0.55)
        data[t + 1, 1] = 0.38 * data[t, 1] + rng.normal(0.0, 0.55)
        data[t + 1, 2] = (
            0.22 * data[t, 2]
            + float(alpha) * np.sin(float(sine_frequency) * data[t, 0] * data[t, 1])
            + rng.normal(0.0, float(noise))
        )
    return pd.DataFrame(data, columns=VARIABLES)


def _standardize_feature_columns(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=float)
    center = array.mean(axis=0, keepdims=True)
    scale = array.std(axis=0, keepdims=True)
    return (array - center) / np.where(scale > 1e-10, scale, 1.0)


def _gaussian_logdet_bias_correction(dimension: int, sample_size: int) -> float:
    if sample_size <= dimension:
        return 0.0
    degrees = sample_size - 1
    return float(
        sum(digamma((degrees + 1 - index) / 2.0) for index in range(1, dimension + 1))
        + dimension * np.log(2.0)
        - dimension * np.log(degrees)
    )


def _estimate_mutual_information_affine_tm(x: np.ndarray, y: np.ndarray) -> float:
    """Fast covariance form of the affine triangular-TM MI estimator."""

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    joint = np.concatenate([x_array, y_array], axis=1)

    def logdet_covariance(values: np.ndarray) -> float:
        covariance = np.atleast_2d(np.cov(values, rowvar=False, bias=False))
        covariance += 1e-6 * np.eye(covariance.shape[0], dtype=float)
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise RuntimeError("affine TM covariance is not positive definite")
        return float(logdet)

    sample_size = len(joint)
    raw = 0.5 * (
        logdet_covariance(x_array)
        + logdet_covariance(y_array)
        - logdet_covariance(joint)
    )
    correction = 0.5 * (
        _gaussian_logdet_bias_correction(x_array.shape[1], sample_size)
        + _gaussian_logdet_bias_correction(y_array.shape[1], sample_size)
        - _gaussian_logdet_bias_correction(joint.shape[1], sample_size)
    )
    return float(raw - correction)


def _harmonic_transport_source_features(
    source: np.ndarray,
    *,
    max_harmonic: int,
    selected_harmonic: int | None = None,
) -> np.ndarray:
    """Use one fixed harmonic dictionary for every frequency condition."""

    array = np.asarray(source, dtype=float)
    if array.ndim != 2 or array.shape[1] not in (1, 2):
        raise ValueError("source must contain one or two scalar source columns")
    if int(max_harmonic) < 1:
        raise ValueError("max_harmonic must be positive")
    if array.shape[1] == 1:
        x = array[:, [0]]
        features = np.concatenate([x, x**2, x**3], axis=1)
        return _standardize_feature_columns(features)

    x = array[:, [0]]
    y = array[:, [1]]
    xy = x * y
    features = [x, y, x**2, y**2, xy]
    harmonics = (
        (int(selected_harmonic),)
        if selected_harmonic is not None
        else range(1, int(max_harmonic) + 1)
    )
    for harmonic in harmonics:
        features.extend([np.sin(harmonic * xy), np.cos(harmonic * xy)])
    return _standardize_feature_columns(np.concatenate(features, axis=1))


def _summarize_harmonic_tm_synergy(
    left_source: np.ndarray,
    right_source: np.ndarray,
    target: np.ndarray,
    *,
    max_harmonic: int,
) -> dict[str, float]:
    left = np.asarray(left_source, dtype=float)
    right = np.asarray(right_source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    sample_count = len(target_array)
    if sample_count < 64:
        raise ValueError("harmonic TM cross-fitting requires at least 64 intervention samples")
    xy = left[:, 0] * right[:, 0]

    def select_harmonic(indices: np.ndarray) -> int:
        target_values = target_array[indices, 0]
        scores = []
        for harmonic in range(1, int(max_harmonic) + 1):
            sine = np.sin(harmonic * xy[indices])
            cosine = np.cos(harmonic * xy[indices])
            correlations = []
            for feature in (sine, cosine):
                correlation = np.corrcoef(feature, target_values)[0, 1]
                correlations.append(abs(float(correlation)) if np.isfinite(correlation) else 0.0)
            scores.append(max(correlations))
        return int(np.argmax(scores) + 1)

    even = np.arange(0, sample_count, 2, dtype=int)
    odd = np.arange(1, sample_count, 2, dtype=int)
    fold_results: list[dict[str, float]] = []
    selected_harmonics: list[int] = []
    for selection_indices, evaluation_indices in ((even, odd), (odd, even)):
        selected = select_harmonic(selection_indices)
        selected_harmonics.append(selected)
        left_features = _harmonic_transport_source_features(
            left[evaluation_indices], max_harmonic=max_harmonic
        )
        right_features = _harmonic_transport_source_features(
            right[evaluation_indices], max_harmonic=max_harmonic
        )
        joint_features = _harmonic_transport_source_features(
            np.concatenate([left[evaluation_indices], right[evaluation_indices]], axis=1),
            max_harmonic=max_harmonic,
            selected_harmonic=selected,
        )
        evaluation_target = target_array[evaluation_indices]
        left_ei = max(
            0.0,
            _estimate_mutual_information_affine_tm(left_features, evaluation_target),
        )
        right_ei = max(
            0.0,
            _estimate_mutual_information_affine_tm(right_features, evaluation_target),
        )
        joint_ei = max(
            0.0,
            _estimate_mutual_information_affine_tm(joint_features, evaluation_target),
        )
        fold_results.append(
            {
                "left_ei": left_ei,
                "right_ei": right_ei,
                "joint_ei": joint_ei,
                "syn": float(joint_ei - left_ei - right_ei),
            }
        )
    averaged = {
        key: float(np.mean([fold[key] for fold in fold_results]))
        for key in ("left_ei", "right_ei", "joint_ei", "syn")
    }
    result = {
        "backend": "cross_fitted_selected_harmonic_affine_triangular_transport_map",
        "max_harmonic": int(max_harmonic),
        "selected_harmonic_fold_1": int(selected_harmonics[0]),
        "selected_harmonic_fold_2": int(selected_harmonics[1]),
        "selected_harmonic": float(np.mean(selected_harmonics)),
        **averaged,
    }
    return result


def _paired_intervention_sources(
    *,
    samples: int,
    seed: int,
    support: Mapping[str, tuple[float, float]],
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    rows = {
        name: rng.uniform(float(support[name][0]), float(support[name][1]), size=int(samples))
        for name in VARIABLES
    }
    return pd.DataFrame(rows)


def _known_dynamics_xy_z(
    *,
    alpha: float,
    sine_frequency: float,
    intervention: pd.DataFrame,
    max_harmonic: int,
) -> dict[str, float]:
    x = intervention[["x"]].to_numpy(dtype=float)
    y = intervention[["y"]].to_numpy(dtype=float)
    z_state = intervention["z"].to_numpy(dtype=float)
    target = (
        0.22 * z_state
        + float(alpha) * np.sin(float(sine_frequency) * x[:, 0] * y[:, 0])
    ).reshape(-1, 1)
    result = _summarize_harmonic_tm_synergy(
        x,
        y,
        target,
        max_harmonic=int(max_harmonic),
    )
    return {
        "sine_frequency": float(sine_frequency),
        **{
            key: float(result[key])
            for key in (
                "left_ei",
                "right_ei",
                "joint_ei",
                "syn",
                "selected_harmonic_fold_1",
                "selected_harmonic_fold_2",
                "selected_harmonic",
            )
        },
    }


def _slope(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    if len(frame) < 2:
        return float("nan")
    x = frame[x_col].to_numpy(dtype=float)
    y = frame[y_col].to_numpy(dtype=float)
    if len(np.unique(x)) < 2:
        return float("nan")
    return float(np.polyfit(x, y, deg=1)[0])


def _trend(summary: pd.DataFrame) -> dict[str, object]:
    alpha_slope_by_k = []
    for k, group in summary.groupby("k"):
        ordered = group.sort_values("alpha")
        alpha_slope_by_k.append(
            {
                "k": float(k),
                "mlp_peid_synergy_slope_per_alpha": _slope(ordered, "alpha", "mlp_peid_xy_synergy_mean"),
                "known_dynamics_peid_synergy_slope_per_alpha": _slope(
                    ordered, "alpha", "known_dynamics_peid_xy_synergy_mean"
                ),
                "z_intervention_r2_slope_per_alpha": _slope(ordered, "alpha", "z_intervention_r2_mean"),
            }
        )

    k_slope_by_alpha = []
    for alpha, group in summary.groupby("alpha"):
        ordered = group.sort_values("k")
        k_slope_by_alpha.append(
            {
                "alpha": float(alpha),
                "mlp_peid_synergy_slope_per_k": _slope(ordered, "k", "mlp_peid_xy_synergy_mean"),
                "known_dynamics_peid_synergy_slope_per_k": _slope(
                    ordered, "k", "known_dynamics_peid_xy_synergy_mean"
                ),
                "z_intervention_r2_slope_per_k": _slope(ordered, "k", "z_intervention_r2_mean"),
            }
        )

    return {
        "alpha_slope_by_k": alpha_slope_by_k,
        "k_slope_by_alpha": k_slope_by_alpha,
    }


def run_sine_frequency_mlp_peid_sweep(
    *,
    alpha_values: Sequence[float] = (0.25, 0.5, 1.0, 1.5, 2.0),
    k_values: Sequence[float] = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0),
    seeds: Sequence[int] = (0, 1, 2, 3),
    n_samples: int = 1100,
    noise: float = 0.05,
    mlp_epochs: int = 90,
    intervention_samples: int = 2048,
    bins: int = 4,
    intervention_support: Mapping[str, tuple[float, float]] | None = None,
    intervention_seed_offset: int = 17021,
    tm_max_harmonic: int = 10,
    status_path: Path | None = None,
) -> dict[str, object]:
    support = dict(
        intervention_support
        or {
            "x": (-1.8, 1.8),
            "y": (-1.8, 1.8),
            "z": (-1.25, 1.25),
        }
    )

    for name in VARIABLES:
        low, high = support[name]
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"invalid intervention support for {name!r}")
    if tuple(support["x"]) != tuple(support["y"]):
        raise ValueError("intervention support for x and y must match")
    if int(tm_max_harmonic) < max(float(value) for value in k_values):
        raise ValueError("tm_max_harmonic must cover the largest tested frequency")

    paired_interventions = {
        int(seed): _paired_intervention_sources(
            samples=int(intervention_samples),
            seed=int(intervention_seed_offset) + int(seed),
            support=support,
        )
        for seed in seeds
    }

    rows: list[dict[str, float]] = []
    total_runs = len(alpha_values) * len(k_values) * len(seeds)
    started = time.monotonic()

    def write_status(current: int, phase: str, message: str = "") -> None:
        if status_path is None:
            return
        elapsed = time.monotonic() - started
        rate = current / elapsed if current > 0 and elapsed > 0 else 0.0
        payload = {
            "phase": phase,
            "current": int(current),
            "total": int(total_runs),
            "unit": "condition",
            "elapsed_seconds": float(elapsed),
            "eta_seconds": float((total_runs - current) / rate) if rate > 0 else None,
            "message": message,
            "updated_at": time.time(),
        }
        path = Path(status_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)

    write_status(0, "running")
    for alpha in alpha_values:
        for k in k_values:
            for seed in seeds:
                series = simulate_no_confounder_sine_system(
                    n_samples=int(n_samples),
                    seed=int(seed),
                    alpha=float(alpha),
                    sine_frequency=float(k),
                    noise=float(noise),
                )
                config = SimConfig(
                    mechanism="common_driver_sine_synergy",
                    n_samples=int(n_samples),
                    noise=float(noise),
                    seed=int(seed),
                    synergy_strength=float(alpha),
                    common_driver_strength=0.0,
                    mlp_epochs=int(mlp_epochs),
                    intervention_samples=int(intervention_samples),
                    bins=int(bins),
                    variable_names=VARIABLES,
                )
                features, targets = make_lagged_dataset(series, lag=config.lag)
                model = train_mlp_transition_model(features, targets, config)
                samples = paired_interventions[int(seed)]
                predictions = model.predict(_intervention_features(samples, config))
                tm_peid = _summarize_harmonic_tm_synergy(
                    samples[["x"]].to_numpy(dtype=float),
                    samples[["y"]].to_numpy(dtype=float),
                    predictions[:, [VARIABLES.index("z")]],
                    max_harmonic=int(tm_max_harmonic),
                )
                known = _known_dynamics_xy_z(
                    alpha=float(alpha),
                    sine_frequency=float(k),
                    intervention=samples,
                    max_harmonic=int(tm_max_harmonic),
                )
                z_target = (
                    0.22 * samples["z"].to_numpy(dtype=float)
                    + float(alpha)
                    * np.sin(
                        float(k)
                        * samples["x"].to_numpy(dtype=float)
                        * samples["y"].to_numpy(dtype=float)
                    )
                )
                z_pred = predictions[:, VARIABLES.index("z")]
                z_mse = float(np.mean((z_target - z_pred) ** 2))
                z_baseline_mse = float(np.mean((z_target - float(np.mean(z_target))) ** 2))
                rows.append(
                    {
                        "run_id": f"alpha={float(alpha):g}|sine_frequency={float(k):g}|seed={int(seed)}",
                        "alpha": float(alpha),
                        "k": float(k),
                        "seed": float(seed),
                        "final_train_loss": float(model.loss_history[-1]) if model.loss_history else float("nan"),
                        "z_intervention_mse": z_mse,
                        "z_intervention_r2": float(1.0 - z_mse / (z_baseline_mse + 1e-12)),
                        "mlp_peid_unique_x": float(tm_peid["left_ei"]),
                        "mlp_peid_unique_y": float(tm_peid["right_ei"]),
                        "mlp_peid_xy_joint": float(tm_peid["joint_ei"]),
                        "mlp_peid_xy_synergy": float(tm_peid["syn"]),
                        "mlp_selected_harmonic": float(tm_peid["selected_harmonic"]),
                        "mlp_selected_harmonic_fold_1": float(tm_peid["selected_harmonic_fold_1"]),
                        "mlp_selected_harmonic_fold_2": float(tm_peid["selected_harmonic_fold_2"]),
                        "known_dynamics_peid_unique_x": float(known["left_ei"]),
                        "known_dynamics_peid_unique_y": float(known["right_ei"]),
                        "known_dynamics_peid_xy_joint": float(known["joint_ei"]),
                        "known_dynamics_peid_xy_synergy": float(known["syn"]),
                        "known_dynamics_selected_harmonic": float(known["selected_harmonic"]),
                    }
                )
                write_status(len(rows), "running", rows[-1]["run_id"])

    frame = pd.DataFrame(rows)
    for column in ("mlp_peid_xy_synergy", "known_dynamics_peid_xy_synergy"):
        violations = frame[column] < -float(SYN_NONNEGATIVE_TOLERANCE_BITS)
        if bool(violations.any()):
            write_status(len(rows), "failed", f"nonnegativity violation in {column}")
            raise RuntimeError(
                f"{column} violates Syn nonnegativity: minimum={frame[column].min():.8g} bits, "
                f"tolerance={SYN_NONNEGATIVE_TOLERANCE_BITS:.8g} bits, count={int(violations.sum())}"
            )
    aggregations = dict(
        final_train_loss_mean=("final_train_loss", "mean"),
        final_train_loss_std=("final_train_loss", "std"),
        z_intervention_r2_mean=("z_intervention_r2", "mean"),
        z_intervention_r2_std=("z_intervention_r2", "std"),
        mlp_peid_unique_x_mean=("mlp_peid_unique_x", "mean"),
        mlp_peid_unique_x_std=("mlp_peid_unique_x", "std"),
        mlp_peid_unique_y_mean=("mlp_peid_unique_y", "mean"),
        mlp_peid_unique_y_std=("mlp_peid_unique_y", "std"),
        mlp_peid_xy_joint_mean=("mlp_peid_xy_joint", "mean"),
        mlp_peid_xy_joint_std=("mlp_peid_xy_joint", "std"),
        mlp_peid_xy_synergy_mean=("mlp_peid_xy_synergy", "mean"),
        mlp_peid_xy_synergy_std=("mlp_peid_xy_synergy", "std"),
        known_dynamics_peid_unique_x_mean=("known_dynamics_peid_unique_x", "mean"),
        known_dynamics_peid_unique_x_std=("known_dynamics_peid_unique_x", "std"),
        known_dynamics_peid_unique_y_mean=("known_dynamics_peid_unique_y", "mean"),
        known_dynamics_peid_unique_y_std=("known_dynamics_peid_unique_y", "std"),
        known_dynamics_peid_xy_joint_mean=("known_dynamics_peid_xy_joint", "mean"),
        known_dynamics_peid_xy_joint_std=("known_dynamics_peid_xy_joint", "std"),
        known_dynamics_peid_xy_synergy_mean=("known_dynamics_peid_xy_synergy", "mean"),
        known_dynamics_peid_xy_synergy_std=("known_dynamics_peid_xy_synergy", "std"),
    )
    summary = (
        frame.groupby(["alpha", "k"], as_index=False)
        .agg(**aggregations)
        .sort_values(["alpha", "k"])
        .reset_index(drop=True)
    )
    summary_by_alpha = (
        summary.groupby("alpha", as_index=False)
        .agg(
            mlp_peid_xy_synergy_mean=("mlp_peid_xy_synergy_mean", "mean"),
            mlp_peid_xy_synergy_std=("mlp_peid_xy_synergy_mean", "std"),
            known_dynamics_peid_xy_synergy_mean=("known_dynamics_peid_xy_synergy_mean", "mean"),
            known_dynamics_peid_xy_synergy_std=("known_dynamics_peid_xy_synergy_mean", "std"),
            z_intervention_r2_mean=("z_intervention_r2_mean", "mean"),
            z_intervention_r2_std=("z_intervention_r2_mean", "std"),
        )
        .sort_values("alpha")
        .reset_index(drop=True)
    )
    summary_by_k = (
        summary.groupby("k", as_index=False)
        .agg(
            mlp_peid_xy_synergy_mean=("mlp_peid_xy_synergy_mean", "mean"),
            mlp_peid_xy_synergy_std=("mlp_peid_xy_synergy_mean", "std"),
            known_dynamics_peid_xy_synergy_mean=("known_dynamics_peid_xy_synergy_mean", "mean"),
            known_dynamics_peid_xy_synergy_std=("known_dynamics_peid_xy_synergy_mean", "std"),
            z_intervention_r2_mean=("z_intervention_r2_mean", "mean"),
            z_intervention_r2_std=("z_intervention_r2_mean", "std"),
        )
        .sort_values("k")
        .reset_index(drop=True)
    )
    result = {
        "config": {
            "alpha_values": [float(value) for value in alpha_values],
            "k_values": [float(value) for value in k_values],
            "seeds": [int(value) for value in seeds],
            "n_samples": int(n_samples),
            "noise": float(noise),
            "mlp_epochs": int(mlp_epochs),
            "intervention_samples": int(intervention_samples),
            "bins": int(bins),
            "variables": list(VARIABLES),
            "confounders": [],
            "intervention_support": {
                name: [float(bound) for bound in support[name]]
                for name in VARIABLES
            },
            "intervention_seed_offset": int(intervention_seed_offset),
            "paired_interventions_across_all_conditions": True,
            "paired_interventions_between_mlp_and_known_dynamics": True,
            "tm_backend": "cross_fitted_selected_harmonic_affine_triangular_transport_map",
            "tm_max_harmonic": int(tm_max_harmonic),
            "syn_nonnegative_tolerance_bits": float(SYN_NONNEGATIVE_TOLERANCE_BITS),
        },
        "units": {"mlp_peid": "bits", "known_dynamics_peid": "bits"},
        "nonnegativity_audit": {
            "tolerance_bits": float(SYN_NONNEGATIVE_TOLERANCE_BITS),
            "mlp_values_in_numerical_zero_band": int(
                ((frame["mlp_peid_xy_synergy"] < 0.0) & (frame["mlp_peid_xy_synergy"] >= -SYN_NONNEGATIVE_TOLERANCE_BITS)).sum()
            ),
            "known_dynamics_values_in_numerical_zero_band": int(
                ((frame["known_dynamics_peid_xy_synergy"] < 0.0) & (frame["known_dynamics_peid_xy_synergy"] >= -SYN_NONNEGATIVE_TOLERANCE_BITS)).sum()
            ),
            "significant_violation_count": 0,
        },
        "runs": rows,
        "summary": summary.to_dict("records"),
        "summary_by_alpha": summary_by_alpha.to_dict("records"),
        "summary_by_k": summary_by_k.to_dict("records"),
        "trend": _trend(summary),
    }
    write_status(total_runs, "complete")
    return result


def plot_sine_frequency_sweep(result: dict[str, object], figure_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(result["summary"]).sort_values(["alpha", "k"])
    runs = pd.DataFrame(result["runs"]).sort_values(["alpha", "k", "seed"])
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    ax_heat, ax_reference, ax_alpha, ax_fit = axes.ravel()
    learned_color = "#356D9A"
    known_color = "#4A4A4A"
    frequency_colors = mpl.colormaps["Blues"](np.linspace(0.35, 0.9, frame["k"].nunique()))

    true_levels = sorted(runs["k"].unique())
    selected_levels = list(range(1, int(result["config"]["tm_max_harmonic"]) + 1))
    recovery = np.zeros((len(true_levels), len(selected_levels)), dtype=float)
    for row_index, true_k in enumerate(true_levels):
        selected = np.rint(
            runs.loc[runs["k"].eq(true_k), "mlp_selected_harmonic"].to_numpy(dtype=float)
        ).astype(int)
        for harmonic in selected_levels:
            recovery[row_index, harmonic - 1] = float(np.mean(selected == harmonic))
    image = ax_heat.imshow(recovery, origin="lower", aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax_heat.set(
        xticks=np.arange(len(selected_levels)),
        xticklabels=[f"{value:g}" for value in selected_levels],
        yticks=np.arange(len(true_levels)),
        yticklabels=[f"{value:g}" for value in true_levels],
        xlabel="Selected harmonic",
        ylabel=r"True spatial frequency $k$",
    )
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.047, pad=0.03)
    colorbar.set_label("Selection frequency")

    grouped = runs.groupby("k", as_index=False).agg(
        learned_mean=("mlp_peid_xy_synergy", "mean"),
        learned_std=("mlp_peid_xy_synergy", "std"),
        known_mean=("known_dynamics_peid_xy_synergy", "mean"),
        known_std=("known_dynamics_peid_xy_synergy", "std"),
    )
    k_axis = grouped["k"].to_numpy(dtype=float)
    for mean_name, std_name, label, color, marker in (
        ("learned_mean", "learned_std", "Learned MLP", learned_color, "o"),
        ("known_mean", "known_std", "Known dynamics", known_color, "s"),
    ):
        mean = grouped[mean_name].to_numpy(dtype=float)
        std = grouped[std_name].fillna(0.0).to_numpy(dtype=float)
        ax_reference.plot(k_axis, mean, color=color, marker=marker, linewidth=1.6, markersize=4, label=label)
        ax_reference.fill_between(k_axis, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
    ax_reference.set(xlabel=r"Spatial frequency $k$", ylabel="Syn (bits)")
    ax_reference.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    for color, (k, group) in zip(frequency_colors, frame.groupby("k"), strict=True):
        ordered = group.sort_values("alpha")
        ax_alpha.errorbar(
            ordered["alpha"],
            ordered["mlp_peid_xy_synergy_mean"],
            yerr=ordered["mlp_peid_xy_synergy_std"].fillna(0.0),
            color=color,
            marker="o",
            linewidth=1.25,
            markersize=3.2,
            capsize=2,
            label=rf"$k={float(k):g}$",
        )
    ax_alpha.set(xlabel=r"Amplitude $\alpha$", ylabel="Learned Syn (bits)")
    ax_alpha.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1)

    for color, (alpha, group) in zip(
        mpl.colormaps["PuRd"](np.linspace(0.35, 0.85, frame["alpha"].nunique())),
        frame.groupby("alpha"),
        strict=True,
    ):
        ordered = group.sort_values("k")
        ax_fit.errorbar(
            ordered["k"],
            ordered["z_intervention_r2_mean"],
            yerr=ordered["z_intervention_r2_std"].fillna(0.0),
            color=color,
            marker="o",
            linewidth=1.2,
            markersize=3.2,
            capsize=2,
            label=rf"$\alpha={float(alpha):g}$",
        )
    ax_fit.axhline(0.0, color="#A0A0A0", linewidth=0.7, linestyle="--")
    ax_fit.set(xlabel=r"Spatial frequency $k$", ylabel=r"Fixed-support $R^2$")
    ax_fit.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    for label, axis in zip("abcd", axes.ravel(), strict=True):
        axis.text(-0.18, 1.06, label, transform=axis.transAxes, fontsize=8, fontweight="bold", va="top")
        axis.tick_params(width=0.7, length=3)

    fig.savefig(figure_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return figure_path


def _fmt(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.4g}"


def write_sine_frequency_report(
    result: dict[str, object],
    figure_path: Path,
    report_path: Path,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fig_rel = os.path.relpath(figure_path, start=report_path.parent)
    config = dict(result["config"])
    summary = pd.DataFrame(result["summary"])
    summary_by_k = pd.DataFrame(result["summary_by_k"])
    trend = dict(result.get("trend", {}))
    table_lines = [
        "| $k$ | Learned Syn | Known-dynamics Syn | Fixed-support $R^2$ | Learned Syn range across $\\alpha$ |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_by_k.to_dict("records"):
        group = summary.loc[summary["k"].eq(float(row["k"])), "mlp_peid_xy_synergy_mean"]
        table_lines.append(
            "| {k:.1f} | {learned} | {known} | {r2} | {span} |".format(
                k=float(row["k"]),
                learned=_fmt(row["mlp_peid_xy_synergy_mean"]),
                known=_fmt(row["known_dynamics_peid_xy_synergy_mean"]),
                r2=_fmt(row["z_intervention_r2_mean"]),
                span=_fmt(float(group.max() - group.min())),
            )
        )
    alpha_trend_lines = [
        "| fixed $k$ | Learned Syn slope / $\\alpha$ | Known-dynamics slope / $\\alpha$ | Fixed-support $R^2$ slope / $\\alpha$ |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in trend.get("alpha_slope_by_k", []):
        alpha_trend_lines.append(
            "| {k:.1f} | {mlp} | {known} | {r2} |".format(
                k=float(row["k"]),
                mlp=_fmt(row.get("mlp_peid_synergy_slope_per_alpha")),
                known=_fmt(row.get("known_dynamics_peid_synergy_slope_per_alpha")),
                r2=_fmt(row.get("z_intervention_r2_slope_per_alpha")),
            )
        )
    audit = dict(result["nonnegativity_audit"])
    runs = pd.DataFrame(result["runs"])
    known_recovery = float(np.mean(runs["known_dynamics_selected_harmonic"].eq(runs["k"])))
    learned_recovery = float(np.mean(runs["mlp_selected_harmonic"].eq(runs["k"])))
    k1_r2 = float(runs.loc[runs["k"].eq(1.0), "z_intervention_r2"].mean())
    high_k_r2 = runs.loc[runs["k"].gt(1.0)].groupby("k")["z_intervention_r2"].mean()
    text = rf"""# 固定干预支持下的 Sine 振幅—空间频率校准实验

本实验检验：当只改变振幅参数 $\alpha$ 或响应面的空间频率 $k$ 时，MLP+PEID 的 ${{x,y}}\rightarrow z$ 协同读数如何变化。这里的 $k$ 是 $x_ty_t$ 响应面上的空间振荡频率，不是时间采样频率。

$$
\begin{{aligned}}
x_{{t+1}} &= 0.42x_t + \eta^x_t,\\
y_{{t+1}} &= 0.38y_t + \eta^y_t,\\
z_{{t+1}} &= 0.22z_t + \alpha\sin(kx_ty_t) + \eta^z_t.
\end{{aligned}}
$$

## 受控比较协议

- treatment：$\alpha\in{config["alpha_values"]}$ 与 $k\in{config["k_values"]}$ 的全因子扫描；
- pairing：每个 seed 的同一批 `{config["intervention_samples"]}` 个干预状态复用于全部 $(\alpha,k)$ 条件；
- support：$x,y\in{config["intervention_support"]["x"]}$，$z\in{config["intervention_support"]["z"]}$，在全部条件中固定；
- readout：learned MLP 与 known dynamics 使用完全相同的干预状态和 TM 估计协议；
- estimator：先从统一的 $1,\ldots,{config["tm_max_harmonic"]}$ 阶固定谐波字典中自动选择响应最强的谐波，再在未参与选择的另一半样本上运行 affine triangular TM；交换两半样本后取平均。该 cross-fitting 协议不读取当前条件的真实 $k$；
- diagnostic：$R^2$ 在固定干预支持上针对无噪声条件均值计算，而不是训练集 $R^2$；
- nonnegativity：原生 Syn 单位中的容差为 `{audit["tolerance_bits"]}` bits；显著违规数为 `{audit["significant_violation_count"]}`。

训练协议固定为 `{config["n_samples"]}` 个轨迹样本、noise `{config["noise"]}`、`{config["mlp_epochs"]}` epochs 和 seeds `{config["seeds"]}`。系统没有共同驱动或隐藏变量。

![无 confounder sine frequency sweep]({fig_rel})

*图｜固定干预支持下的振幅—空间频率校准。a，learned MLP 从 1–10 阶候选字典中选择各谐波的频率；每个真实 $k$ 汇总 5 个 $\alpha$ 条件和 4 个 seeds。b，learned MLP 与 known-dynamics TM 的 Syn；点为跨 $\alpha$ 和 seed 的均值，阴影为相应标准差。c，各 $k$ 下 learned Syn 随 $\alpha$ 的变化；点和误差棒分别为 4 个 seeds 的均值和标准差。d，固定干预支持上的条件均值预测 $R^2$；点和误差棒分别为 4 个 seeds 的均值和标准差。*

## 结果判断

1. **Known-dynamics 基准能够识别空间频率。** 在不读取真实 $k$ 的 cross-fitted 选频协议下，真实谐波恢复率为 `{known_recovery:.1%}`。
2. **当前 learned MLP 没有复现高频识别。** 总体真实谐波恢复率为 `{learned_recovery:.1%}`，成功条件主要集中在 $k=1$。固定支持 $R^2$ 在 $k=1$ 时为 `{k1_r2:.3f}`，而 $k>1$ 时各条件均值仅为 `{float(high_k_r2.min()):.3f}`–`{float(high_k_r2.max()):.3f}`。
3. **振幅不变性没有复现。** Known-dynamics Syn 随 $\alpha$ 稳定增加，其各 $k$ 条件的斜率约为 `{min(float(row["known_dynamics_peid_synergy_slope_per_alpha"]) for row in trend["alpha_slope_by_k"]):.3f}`–`{max(float(row["known_dynamics_peid_synergy_slope_per_alpha"]) for row in trend["alpha_slope_by_k"]):.3f}` bits / unit $\alpha$。旧结果中近似水平的振幅曲线主要来自低阶 TM 特征饱和，不能解释为 PEID 对物理振幅严格不敏感。

## 汇总结果

{chr(10).join(table_lines)}

固定 $k$ 时沿振幅方向的线性斜率：

{chr(10).join(alpha_trend_lines)}

## 解释边界

“Known dynamics” 是在已知条件均值函数上运行相同 TM 估计器所得的机制基准，不是解析真值。Learned 与 known-dynamics 曲线的差异同时反映有限轨迹学习误差与有限样本 TM 误差。只有在固定支持 $R^2$ 保持良好时，才可把 Syn 随 $k$ 的变化主要解释为对响应面几何的敏感性；若二者同时下降，则应解释为 surrogate 分辨率边界。
"""
    report_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return report_path


def _parse_float_values(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("value list must contain at least one numeric value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--alpha-values", type=_parse_float_values, default=(0.25, 0.5, 1.0, 1.5, 2.0))
    parser.add_argument("--k-values", type=_parse_float_values, default=(1.0, 2.0, 4.0, 6.0, 8.0, 10.0))
    parser.add_argument("--seeds", type=lambda text: tuple(int(value) for value in text.split(",")), default=(0, 1, 2, 3))
    parser.add_argument("--n-samples", type=int, default=1100)
    parser.add_argument("--mlp-epochs", type=int, default=90)
    parser.add_argument("--intervention-samples", type=int, default=2048)
    parser.add_argument("--tm-max-harmonic", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_sine_frequency_mlp_peid_sweep(
        alpha_values=args.alpha_values,
        k_values=args.k_values,
        seeds=args.seeds,
        n_samples=args.n_samples,
        mlp_epochs=args.mlp_epochs,
        intervention_samples=args.intervention_samples,
        tm_max_harmonic=args.tm_max_harmonic,
        status_path=args.status_path,
    )
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    figure_path = plot_sine_frequency_sweep(result, args.figure_path)
    report_path = write_sine_frequency_report(result, figure_path, args.report_path)
    print(
        json.dumps(
            {
                "result_path": str(args.result_path),
                "figure_path": str(figure_path),
                "report_path": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
