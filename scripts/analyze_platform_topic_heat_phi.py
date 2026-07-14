#!/usr/bin/env python3
"""Tune one-step topic-heat predictors and estimate TM Phi with hierarchy.

The analysis uses the model-ready hourly series, but preserves the original
observation mask: interpolated values may be predictors, never fitting or
scoring targets.  Phi is evaluated from the fitted stochastic transition under
an independent bounded-uniform intervention over platform-history variables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from exp.TM.transport_map_density import estimate_mutual_information_transport_map


DEFAULT_DATA = ROOT / "data/platform_topic_heat_index_hourly_model_ready.csv"
DEFAULT_MASK = ROOT / "data/platform_topic_heat_index_hourly_imputation_mask.csv"
DEFAULT_OUTPUT = ROOT / "results/platform_topic_heat_phi"
DEFAULT_STATUS = ROOT / "docs/log/platform_topic_heat_phi_live_progress.json"
PLATFORMS = ("哔哩哔哩", "小红书", "快手", "抖音")
PLATFORM_PLOT_LABELS = {
    "哔哩哔哩": "Bilibili",
    "小红书": "Xiaohongshu",
    "快手": "Kuaishou",
    "抖音": "Douyin",
}
ALPHAS = tuple(10.0**power for power in range(-4, 4))
GAMMAS = (0.5, 1.0, 2.0, 4.0)
VALIDATION_FOLDS = ((120, 144), (144, 168), (168, 192))
TEST_START = 192
EPS = 1.0e-10


@dataclass(frozen=True)
class Config:
    order: int
    alpha: float
    feature_family: str
    gamma: float | None


@dataclass
class FittedTransition:
    config: Config
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    models: list[Ridge]
    residual_covariance: np.ndarray
    feature_columns: int


@dataclass(frozen=True)
class SignedAtom:
    sources: tuple[str, ...]
    value: float
    kind: str
    depth: int


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def load_series(data_path: Path, mask_path: Path) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    frame = pd.read_csv(data_path, parse_dates=["time_bin"]).set_index("time_bin").sort_index()
    mask = pd.read_csv(mask_path, parse_dates=["time_bin"]).set_index("time_bin").sort_index()
    if tuple(frame.columns) != PLATFORMS or tuple(mask.columns) != PLATFORMS:
        raise ValueError(f"Expected platform columns {PLATFORMS} in data and mask.")
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    if not frame.index.equals(expected) or not mask.index.equals(expected):
        raise ValueError("Model-ready data and mask must share a contiguous hourly index.")
    values = frame.to_numpy(dtype=float)
    observed = mask.eq("observed").to_numpy(dtype=bool)
    if not np.isfinite(values).all():
        raise ValueError("Model-ready series contains non-finite values.")
    return frame.index, values, observed


def make_supervised(values: np.ndarray, observed: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return lagged features, targets, target masks, and source masks."""
    if order < 1 or len(values) <= order:
        raise ValueError("order must leave at least one target row.")
    histories = [values[order - lag : len(values) - lag] for lag in range(1, order + 1)]
    history_mask = [observed[order - lag : len(values) - lag] for lag in range(1, order + 1)]
    x = np.concatenate(histories, axis=1)
    x_observed = np.concatenate(history_mask, axis=1)
    y = values[order:]
    y_observed = observed[order:]
    return x, y, y_observed, x_observed


def robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(values, dtype=float).mean(axis=0)
    scale = np.asarray(values, dtype=float).std(axis=0, ddof=1)
    return mean, np.where(scale > 1.0e-10, scale, 1.0)


def expanded_features(x_z: np.ndarray, config: Config) -> np.ndarray:
    if config.feature_family == "linear":
        return x_z
    if config.feature_family == "linear_sigmoid" and config.gamma is not None:
        return np.concatenate([x_z, sigmoid(float(config.gamma) * x_z)], axis=1)
    raise ValueError(f"Unsupported feature family: {config.feature_family}")


def fit_transition(
    x: np.ndarray,
    y: np.ndarray,
    y_observed: np.ndarray,
    train_rows: np.ndarray,
    config: Config,
) -> FittedTransition:
    if train_rows.size < 4:
        raise ValueError("Need at least four train rows.")
    x_mean, x_scale = robust_scale(x[train_rows])
    x_z = (x - x_mean) / x_scale
    design = expanded_features(x_z, config)
    n_targets = y.shape[1]
    y_mean = np.empty(n_targets)
    y_scale = np.empty(n_targets)
    models: list[Ridge] = []
    for target in range(n_targets):
        usable = train_rows[y_observed[train_rows, target]]
        if usable.size < max(8, design.shape[1] // 4):
            raise ValueError(f"Insufficient observed targets for {PLATFORMS[target]}.")
        mean, scale = robust_scale(y[usable][:, [target]])
        y_mean[target], y_scale[target] = float(mean[0]), float(scale[0])
        model = Ridge(alpha=float(config.alpha), fit_intercept=True)
        model.fit(design[usable], (y[usable, target] - y_mean[target]) / y_scale[target])
        models.append(model)

    prediction_z = np.column_stack([model.predict(design) for model in models])
    residual_z = (y - y_mean) / y_scale - prediction_z
    complete = y_observed[train_rows].all(axis=1)
    covariance_rows = train_rows[complete]
    if covariance_rows.size < 8:
        raise ValueError("Insufficient jointly observed rows for residual covariance.")
    residual_covariance = np.cov(residual_z[covariance_rows], rowvar=False, bias=False)
    residual_covariance = nearest_positive_definite(np.atleast_2d(residual_covariance), floor=1.0e-6)
    return FittedTransition(config, x_mean, x_scale, y_mean, y_scale, models, residual_covariance, design.shape[1])


def predict_transition(fit: FittedTransition, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_z = (np.asarray(x, dtype=float) - fit.x_mean) / fit.x_scale
    design = expanded_features(x_z, fit.config)
    prediction_z = np.column_stack([model.predict(design) for model in fit.models])
    prediction = prediction_z * fit.y_scale + fit.y_mean
    return prediction, prediction_z


def standardized_rmse(y: np.ndarray, prediction: np.ndarray, observed: np.ndarray, y_mean: np.ndarray, y_scale: np.ndarray) -> tuple[float, list[float]]:
    scores = []
    for target in range(y.shape[1]):
        use = observed[:, target]
        if not np.any(use):
            scores.append(float("nan"))
            continue
        error = ((y[use, target] - prediction[use, target]) / y_scale[target]) ** 2
        scores.append(float(np.sqrt(error.mean())))
    return float(np.nanmean(scores)), scores


def validation_rows(order: int, end: int) -> np.ndarray:
    return np.arange(order, end, dtype=int) - order


def evaluate_config(values: np.ndarray, observed: np.ndarray, config: Config) -> dict[str, Any]:
    x, y, y_observed, _ = make_supervised(values, observed, config.order)
    fold_rows: list[dict[str, Any]] = []
    for fold_index, (valid_start, valid_end) in enumerate(VALIDATION_FOLDS, start=1):
        train = validation_rows(config.order, valid_start)
        valid = np.arange(valid_start, valid_end, dtype=int) - config.order
        fit = fit_transition(x, y, y_observed, train, config)
        prediction, _ = predict_transition(fit, x[valid])
        score, platform_scores = standardized_rmse(y[valid], prediction, y_observed[valid], fit.y_mean, fit.y_scale)
        fold_rows.append({"fold": fold_index, "train_end_hour": valid_start - 1, "validation_start_hour": valid_start, "validation_end_hour": valid_end - 1, "standardized_rmse": score, **{f"rmse_{name}": value for name, value in zip(PLATFORMS, platform_scores)}})
    return {"config": config, "folds": fold_rows, "mean_validation_rmse": float(np.mean([row["standardized_rmse"] for row in fold_rows]))}


def candidate_configs() -> Iterable[Config]:
    for order in range(1, 7):
        for alpha in ALPHAS:
            yield Config(order, alpha, "linear", None)
        for alpha in ALPHAS:
            for gamma in GAMMAS:
                yield Config(order, alpha, "linear_sigmoid", gamma)


def tune(values: np.ndarray, observed: np.ndarray) -> tuple[Config, pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    all_results = []
    for config in candidate_configs():
        result = evaluate_config(values, observed, config)
        all_results.append(result)
        row = {**asdict(config), "mean_validation_standardized_rmse": result["mean_validation_rmse"]}
        for fold in result["folds"]:
            row[f"fold_{fold['fold']}_standardized_rmse"] = fold["standardized_rmse"]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(
        ["mean_validation_standardized_rmse", "order", "feature_family", "alpha", "gamma"],
        na_position="first",
    ).reset_index(drop=True)
    winner = table.iloc[0]
    config = Config(int(winner.order), float(winner.alpha), str(winner.feature_family), None if pd.isna(winner.gamma) else float(winner.gamma))
    return config, table, all_results


def evaluate_test(values: np.ndarray, observed: np.ndarray, config: Config, index: pd.DatetimeIndex) -> tuple[FittedTransition, pd.DataFrame, dict[str, Any]]:
    x, y, y_observed, _ = make_supervised(values, observed, config.order)
    development = validation_rows(config.order, TEST_START)
    test = np.arange(TEST_START, len(values), dtype=int) - config.order
    fit = fit_transition(x, y, y_observed, development, config)
    prediction, _ = predict_transition(fit, x[test])
    persistence = x[test, : len(PLATFORMS)]
    model_score, model_platform = standardized_rmse(y[test], prediction, y_observed[test], fit.y_mean, fit.y_scale)
    persistence_score, persistence_platform = standardized_rmse(y[test], persistence, y_observed[test], fit.y_mean, fit.y_scale)
    records = []
    for local, row in enumerate(test):
        record: dict[str, Any] = {"time_bin": index[row + config.order].isoformat()}
        for column, name in enumerate(PLATFORMS):
            record[f"observed_{name}"] = bool(y_observed[row, column])
            record[f"actual_{name}"] = float(y[row, column]) if y_observed[row, column] else np.nan
            record[f"prediction_{name}"] = float(prediction[local, column])
            record[f"persistence_{name}"] = float(persistence[local, column])
        records.append(record)
    metrics = {
        "selected_model_standardized_rmse": model_score,
        "persistence_standardized_rmse": persistence_score,
        "selected_model_by_platform": dict(zip(PLATFORMS, model_platform)),
        "persistence_by_platform": dict(zip(PLATFORMS, persistence_platform)),
    }
    return fit, pd.DataFrame(records), metrics


def nearest_positive_definite(matrix: np.ndarray, floor: float = 1.0e-8) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


def all_subsets(names: Sequence[str]) -> list[tuple[str, ...]]:
    return [combo for size in range(1, len(names) + 1) for combo in itertools.combinations(names, size)]


def subset_phi(subset: tuple[str, ...], ei: Mapping[tuple[str, ...], float]) -> float:
    return float(ei[subset] - sum(float(ei[(name,)]) for name in subset))


def bipartitions(names: tuple[str, ...]) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    if len(names) <= 1:
        return []
    first, rest = names[0], names[1:]
    partitions = []
    for mask in range(1 << len(rest)):
        left_set = {first, *(name for bit, name in enumerate(rest) if mask & (1 << bit))}
        if len(left_set) == len(names):
            continue
        left = tuple(name for name in names if name in left_set)
        right = tuple(name for name in names if name not in left_set)
        partitions.append((left, right))
    return partitions


def signed_greedy_atoms(names: tuple[str, ...], ei: Mapping[tuple[str, ...], float], depth: int = 0) -> list[SignedAtom]:
    block_phi = subset_phi(names, ei)
    if len(names) <= 1:
        return []
    candidates = []
    for left, right in bipartitions(names):
        captured = subset_phi(left, ei) + subset_phi(right, ei)
        residual = block_phi - captured
        candidates.append((captured, abs(residual), left, right, residual))
    captured, _, left, right, residual = max(candidates, key=lambda item: (item[0], -item[1]))
    if captured <= EPS:
        return [SignedAtom(names, block_phi, "terminal", depth)]
    atoms = []
    if abs(residual) > EPS:
        atoms.append(SignedAtom(names, residual, "split_residual", depth))
    atoms.extend(signed_greedy_atoms(left, ei, depth + 1))
    atoms.extend(signed_greedy_atoms(right, ei, depth + 1))
    return atoms


def interventional_samples(
    values: np.ndarray,
    observed: np.ndarray,
    fit: FittedTransition,
    *,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]], list[list[float]]]:
    x, _, _, x_observed = make_supervised(values, observed, fit.config.order)
    x_z = (x - fit.x_mean) / fit.x_scale
    lower, upper = np.empty(x.shape[1]), np.empty(x.shape[1])
    for column in range(x.shape[1]):
        usable = x_observed[:, column]
        support = x_z[usable, column] if np.any(usable) else x_z[:, column]
        lower[column], upper[column] = float(np.min(support)), float(np.max(support))
        if upper[column] - lower[column] < 1.0e-8:
            lower[column], upper[column] = lower[column] - 0.5, upper[column] + 0.5
    rng = np.random.default_rng(seed)
    source_z = rng.uniform(lower, upper, size=(int(sample_count), x.shape[1]))
    design = expanded_features(source_z, fit.config)
    mean_target = np.column_stack([model.predict(design) for model in fit.models])
    noise = rng.multivariate_normal(np.zeros(len(PLATFORMS)), fit.residual_covariance, size=int(sample_count))
    target_z = mean_target + noise
    blocks = {name: [lag * len(PLATFORMS) + platform for lag in range(fit.config.order)] for platform, name in enumerate(PLATFORMS)}
    bounds = [[float(lo), float(hi)] for lo, hi in zip(lower, upper)]
    return source_z, target_z, blocks, bounds


def compute_phi(
    values: np.ndarray,
    observed: np.ndarray,
    fit: FittedTransition,
    *,
    sample_count: int,
    seed: int,
    tm_degree: int,
) -> tuple[pd.DataFrame, list[SignedAtom], dict[str, Any]]:
    source, target, blocks, bounds = interventional_samples(values, observed, fit, sample_count=sample_count, seed=seed)
    ei: dict[tuple[str, ...], float] = {}
    for subset in all_subsets(PLATFORMS):
        columns = [column for name in subset for column in blocks[name]]
        estimate = estimate_mutual_information_transport_map(source[:, columns], target, degree=int(tm_degree))
        ei[subset] = float(estimate["mi_hat"])
    rows = []
    for subset in all_subsets(PLATFORMS):
        rows.append({"sources": " + ".join(subset), "source_count": len(subset), "ei_bits": ei[subset], "phi_bits": subset_phi(subset, ei)})
    atoms = signed_greedy_atoms(tuple(PLATFORMS), ei)
    full = tuple(PLATFORMS)
    audit = {
        "sample_count": int(sample_count),
        "seed": int(seed),
        "tm_degree": int(tm_degree),
        "tm_backend": f"polynomial_triangular_transport_map_degree_{int(tm_degree)}",
        "intervention": "independent bounded-uniform over every platform-history coordinate",
        "source_bounds_standardized": bounds,
        "source_blocks": blocks,
        "overall_phi_bits": subset_phi(full, ei),
        "atom_sum_bits": float(sum(atom.value for atom in atoms)),
    }
    if not math.isclose(audit["overall_phi_bits"], audit["atom_sum_bits"], abs_tol=1.0e-7):
        raise AssertionError("Signed hierarchy does not conserve full Phi.")
    return pd.DataFrame(rows), atoms, audit


def config_digest(config: Config, sample_count: int, tm_degree: int) -> str:
    payload = json.dumps({"config": asdict(config), "sample_count": sample_count, "tm_degree": tm_degree}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def read_null_cache(path: Path, digest: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    cached = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("config_digest") == digest:
            cached.append(row)
    return cached


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_progress_status(
    path: Path | None,
    *,
    started_at: float,
    current: int,
    total: int,
    latest_phi: float | None = None,
    phase: str = "null_permutations",
    message: str | None = None,
) -> None:
    """Atomically publish lightweight progress for detached experiment monitoring."""
    if path is None:
        return
    elapsed = time.monotonic() - started_at
    rate = current / elapsed if elapsed > 0.0 else 0.0
    payload: dict[str, Any] = {
        "phase": phase,
        "current": int(current),
        "total": int(total),
        "unit": "null replicate",
        "elapsed_seconds": elapsed,
        "eta_seconds": (total - current) / rate if rate > 0.0 else None,
        "metrics": {"latest_phi_bits": latest_phi} if latest_phi is not None else {},
        "updated_at": time.time(),
    }
    if message is not None:
        payload["message"] = message
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def null_distribution(
    values: np.ndarray,
    observed: np.ndarray,
    config: Config,
    *,
    replicates: int,
    sample_count: int,
    tm_degree: int,
    seed: int,
    cache_path: Path,
    status_path: Path | None = None,
) -> list[dict[str, Any]]:
    started_at = time.monotonic()
    digest = config_digest(config, sample_count, tm_degree)
    cached = {int(row["replicate"]): row for row in read_null_cache(cache_path, digest)}
    write_progress_status(
        status_path,
        started_at=started_at,
        current=len(cached),
        total=int(replicates),
        latest_phi=float(cached[max(cached)]["overall_phi_bits"]) if cached else None,
        message="Resuming cached null replicates." if cached else "Starting null replicates.",
    )
    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, len(values), size=(int(replicates), len(PLATFORMS)))
    results = []
    for replicate in range(int(replicates)):
        if replicate in cached:
            results.append(cached[replicate])
            continue
        try:
            shifted_values = np.column_stack([np.roll(values[:, column], int(shifts[replicate, column])) for column in range(len(PLATFORMS))])
            shifted_observed = np.column_stack([np.roll(observed[:, column], int(shifts[replicate, column])) for column in range(len(PLATFORMS))])
            x, y, y_observed, _ = make_supervised(shifted_values, shifted_observed, config.order)
            fit = fit_transition(x, y, y_observed, np.arange(len(x), dtype=int), config)
            _, atoms, audit = compute_phi(shifted_values, shifted_observed, fit, sample_count=sample_count, seed=seed + 10_000 + replicate, tm_degree=tm_degree)
        except Exception as error:
            write_progress_status(status_path, started_at=started_at, current=len(results), total=int(replicates), phase="failed", message=str(error))
            raise
        row = {
            "config_digest": digest,
            "replicate": replicate,
            "shifts": [int(value) for value in shifts[replicate]],
            "overall_phi_bits": audit["overall_phi_bits"],
            "atoms": [{"sources": list(atom.sources), "value": atom.value, "kind": atom.kind, "depth": atom.depth} for atom in atoms],
        }
        append_jsonl(cache_path, row)
        results.append(row)
        write_progress_status(status_path, started_at=started_at, current=len(results), total=int(replicates), latest_phi=row["overall_phi_bits"])
        print(f"null {replicate + 1}/{replicates}: Phi={row['overall_phi_bits']:.4f} bits", flush=True)
    ordered = sorted(results, key=lambda row: int(row["replicate"]))
    write_progress_status(status_path, started_at=started_at, current=len(ordered), total=int(replicates), latest_phi=float(ordered[-1]["overall_phi_bits"]) if ordered else None, phase="complete")
    return ordered


def configure_plots() -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8, "axes.spines.right": False, "axes.spines.top": False, "legend.frameon": False})


def save_figure(fig: plt.Figure, base: Path) -> None:
    for suffix, kwargs in ((".png", {"dpi": 300}), (".svg", {}), (".pdf", {})):
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def plot_forecast(table: pd.DataFrame, destination: Path) -> None:
    configure_plots()
    times = pd.to_datetime(table["time_bin"])
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.2), sharex=True, constrained_layout=True)
    handles = None
    for axis, name in zip(axes.flat, PLATFORMS):
        actual = table[f"actual_{name}"]
        axis.plot(times, actual, color="#1F4E79", marker="o", markersize=2.5, linewidth=1.2, label="Observed")
        axis.plot(times, table[f"prediction_{name}"], color="#D55E00", linewidth=1.1, label="Selected model")
        axis.set_title(PLATFORM_PLOT_LABELS[name], fontsize=9, fontweight="bold")
        axis.set_ylabel("Heat index")
        axis.tick_params(axis="x", rotation=25)
        handles = axis.get_legend_handles_labels()
    assert handles is not None
    fig.legend(handles[0], handles[1], loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False)
    save_figure(fig, destination)


def plot_null(observed_phi: float, null_rows: Sequence[Mapping[str, Any]], destination: Path) -> None:
    configure_plots()
    values = np.asarray([float(row["overall_phi_bits"]) for row in null_rows])
    fig, axis = plt.subplots(figsize=(5.4, 3.2), constrained_layout=True)
    axis.hist(values, bins=min(24, max(8, len(values) // 8)), color="#B8C7D9", edgecolor="white")
    axis.axvline(observed_phi, color="#D55E00", linewidth=1.5, label="Observed Phi")
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    axis.set(xlabel="Four-platform Phi (bits)", ylabel="Null replicates")
    save_figure(fig, destination)


def plot_atoms(atoms: Sequence[SignedAtom], destination: Path) -> None:
    configure_plots()
    labels = [" + ".join(PLATFORM_PLOT_LABELS[name] for name in atom.sources) for atom in atoms]
    values = np.asarray([atom.value for atom in atoms])
    colors = np.where(values >= 0.0, "#3D7E6A", "#B04A5A")
    fig, axis = plt.subplots(figsize=(max(6.0, 1.15 * len(atoms)), 3.4), constrained_layout=True)
    positions = np.arange(len(atoms))
    axis.bar(positions, values, color=colors, width=0.7)
    axis.axhline(0.0, color="#333333", linewidth=0.7)
    axis.set(xticks=positions, xticklabels=labels, ylabel="Signed hierarchy atom (bits)")
    axis.tick_params(axis="x", rotation=25)
    save_figure(fig, destination)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL, float_format="%.12g")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    index, values, observed = load_series(Path(args.data), Path(args.mask))
    config, selection, _ = tune(values, observed)
    write_csv(selection, output / "model_selection.csv")
    test_fit, test_predictions, test_metrics = evaluate_test(values, observed, config, index)
    write_csv(test_predictions, output / "test_predictions.csv")
    x, y, y_observed, _ = make_supervised(values, observed, config.order)
    final_fit = fit_transition(x, y, y_observed, np.arange(len(x), dtype=int), config)
    subset_table, atoms, phi_audit = compute_phi(values, observed, final_fit, sample_count=int(args.tm_samples), seed=int(args.seed), tm_degree=int(args.tm_degree))
    write_csv(subset_table, output / "subset_ei_phi.csv")
    atom_table = pd.DataFrame([{"sources": " + ".join(atom.sources), "value_bits": atom.value, "kind": atom.kind, "depth": atom.depth} for atom in atoms])
    write_csv(atom_table, output / "hierarchy_atoms.csv")
    null_rows = null_distribution(values, observed, config, replicates=int(args.null_replicates), sample_count=int(args.tm_samples), tm_degree=int(args.tm_degree), seed=int(args.seed), cache_path=output / "null_replicates.jsonl", status_path=args.status_path)
    observed_phi = float(phi_audit["overall_phi_bits"])
    null_values = np.asarray([float(row["overall_phi_bits"]) for row in null_rows])
    null_summary = {
        "replicates": int(len(null_values)),
        "mean_bits": float(null_values.mean()),
        "std_bits": float(null_values.std(ddof=1)) if len(null_values) > 1 else 0.0,
        "empirical_p_ge_observed": float((1 + np.sum(null_values >= observed_phi)) / (len(null_values) + 1)),
    }
    plot_forecast(test_predictions, output / "test_forecast")
    plot_null(observed_phi, null_rows, output / "phi_null_distribution")
    plot_atoms(atoms, output / "hierarchy_atoms")
    summary = {
        "data": str(Path(args.data)),
        "mask": str(Path(args.mask)),
        "time_range": {"start": index.min().isoformat(), "end": index.max().isoformat(), "n_hours": len(index)},
        "observation_counts": dict(zip(PLATFORMS, observed.sum(axis=0).astype(int).tolist())),
        "selection_protocol": {"development_end": TEST_START - 1, "test_hours": 24, "validation_folds": VALIDATION_FOLDS, "metric": "equal-weight standardized RMSE", "target_policy": "interpolated targets excluded; interpolated predictors allowed"},
        "selected_config": asdict(config),
        "test_metrics": test_metrics,
        "residual_covariance_standardized": final_fit.residual_covariance.tolist(),
        "phi": phi_audit,
        "null": {"method": "independent non-zero circular shifts by platform; selected hyperparameters fixed", **null_summary},
        "limitations": ["Zotero connector unavailable; TM/PEID method follows repository implementation.", "Small sample and 43 imputed Kuaishou values limit inferential power.", "Phi concerns the fitted transition under a bounded uniform intervention, not observed-platform absolute volume."],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--tm-samples", type=int, default=8192)
    parser.add_argument("--tm-degree", type=int, default=2)
    parser.add_argument("--null-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.tm_samples < 128 or args.null_replicates < 1 or args.tm_degree < 1:
        raise ValueError("tm-samples >=128, null-replicates >=1, and tm-degree >=1 are required.")
    summary = run(args)
    print(json.dumps({"selected_config": summary["selected_config"], "test_metrics": summary["test_metrics"], "phi_bits": summary["phi"]["overall_phi_bits"], "null_p": summary["null"]["empirical_p_ge_observed"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
