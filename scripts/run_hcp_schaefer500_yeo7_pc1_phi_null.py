#!/usr/bin/env python3
"""Compute history-source PhiEID for the fitted 7D Yeo7-PC1 dynamics and paired nulls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hcp_lausanne_phi_eid_pilot import circular_shift_null, gaussian_phi_from_linear_transition
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import DEFAULT_DATA, DEFAULT_LABELS, default_data_key, default_yeo7_labels, fit_yeo7_pc1, load_hcp_series, load_yeo7_groups


DEFAULT_OUTPUT_DIR = ROOT / "results" / "hcp_schaefer500_yeo7_pc1_phi_null"


def _state_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(values, dtype=float).mean(axis=0, keepdims=True)
    scale = np.asarray(values, dtype=float).std(axis=0, ddof=1, keepdims=True)
    return mean, np.where(scale > 1.0e-12, scale, 1.0)


def _history_samples(series: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(series, dtype=float)
    if values.ndim != 2 or order < 1 or len(values) <= order + 1:
        raise ValueError("series must have enough rows for the requested positive history order.")
    return np.concatenate([values[order - 1 - lag : -1 - lag] for lag in range(order)], axis=1), values[order:]


def fit_delta_history_phi(
    series: np.ndarray,
    *,
    alpha: float,
    order: int,
    development_end: int,
    covariance_ridge: float = 1.0e-6,
) -> dict[str, Any]:
    """Fit the selected delta Ridge and express it as a history-to-next-state transition."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 2 or development_end > len(values):
        raise ValueError("series must contain the requested development segment.")
    history, next_state = _history_samples(values, int(order))
    n_state = values.shape[1]
    train_rows = development_end - int(order)
    if train_rows < 3:
        raise ValueError("development segment is too short for the history order.")
    train_history = history[:train_rows]
    train_next = next_state[:train_rows]
    state_mean, state_scale = _state_scale(train_history[:, :n_state])
    train_x = np.concatenate(
        [(train_history[:, start : start + n_state] - state_mean) / state_scale for start in range(0, train_history.shape[1], n_state)], axis=1
    )
    delta = train_next - train_history[:, :n_state]
    delta_mean, delta_scale = _state_scale(delta)
    train_delta_z = (delta - delta_mean) / delta_scale
    model = Ridge(alpha=float(alpha), fit_intercept=True).fit(train_x, train_delta_z)
    residual_z = train_delta_z - model.predict(train_x)
    delta_to_state = np.diag((delta_scale / state_scale).reshape(-1))
    transition = np.zeros((n_state, train_x.shape[1]), dtype=float)
    transition[:, :n_state] = np.eye(n_state)
    transition += delta_to_state @ np.asarray(model.coef_, dtype=float)
    residual_covariance = np.atleast_2d(np.cov(residual_z, rowvar=False, bias=False))
    noise_covariance = delta_to_state @ residual_covariance @ delta_to_state.T
    phi = gaussian_phi_from_linear_transition(transition, noise_covariance, ridge=float(covariance_ridge))
    holdout_x = np.concatenate(
        [
            (history[train_rows:, start : start + n_state] - state_mean) / state_scale
            for start in range(0, history.shape[1], n_state)
        ],
        axis=1,
    )
    if len(holdout_x):
        holdout_current = history[train_rows:, :n_state]
        holdout_next = next_state[train_rows:]
        predicted_delta = model.predict(holdout_x) * delta_scale + delta_mean
        predicted_next = holdout_current + predicted_delta
        true_z = (holdout_next - state_mean) / state_scale
        predicted_z = (predicted_next - state_mean) / state_scale
        persistence_z = (holdout_current - state_mean) / state_scale
        holdout_rmse = float(np.sqrt(np.mean((true_z - predicted_z) ** 2)))
        persistence_rmse = float(np.sqrt(np.mean((true_z - persistence_z) ** 2)))
        heldout = {
            "rmse": holdout_rmse,
            "persistence_rmse": persistence_rmse,
            "skill_ratio": float(holdout_rmse / max(persistence_rmse, 1.0e-12)),
        }
    else:
        heldout = {"rmse": float("nan"), "persistence_rmse": float("nan"), "skill_ratio": float("nan")}
    return {
        "transition": transition,
        "noise_covariance": noise_covariance,
        "phi": phi,
        "n_source_variables": int(transition.shape[1]),
        "n_target_variables": int(transition.shape[0]),
        "heldout": heldout,
    }


def _subject_seed(seed: int, replicate: int) -> int:
    return int(np.random.SeedSequence([int(seed), int(replicate)]).generate_state(1)[0])


def summarize_null(observed: float, null_values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(null_values, dtype=float)
    return {
        "n_null": int(len(values)),
        "null_mean": float(values.mean()),
        "null_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "observed_minus_null_mean": float(observed - values.mean()),
        "observed_greater_fraction": float(np.mean(observed > values)),
        "empirical_p_ge_observed": float((1 + np.sum(values >= observed)) / (len(values) + 1)),
    }


def plot_null_distribution(observed: float, null_values: np.ndarray, destination: Path) -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8, "axes.spines.right": False, "axes.spines.top": False})
    fig, axis = plt.subplots(figsize=(4.2, 2.8), constrained_layout=True)
    axis.hist(null_values, bins=18, color="#B8C7D9", edgecolor="white")
    axis.axvline(observed, color="#D55E00", linewidth=1.4)
    axis.annotate("Observed", xy=(observed, axis.get_ylim()[1]), xytext=(4, -4), textcoords="offset points", color="#D55E00", ha="left", va="top", fontsize=8)
    axis.set(xlabel="History-source $\\Phi^{EID}$ (bits)", ylabel="Null replicates")
    for suffix, kwargs in ((".png", {"dpi": 300}), (".svg", {}), (".pdf", {})):
        fig.savefig(destination.with_suffix(suffix), bbox_inches="tight", **kwargs)


def run(
    data_path: Path,
    labels_path: Path,
    output_dir: Path,
    *,
    development_end: int = 900,
    order: int = 8,
    alpha: float = 10.0,
    null_replicates: int = 200,
    seed: int = 20260714,
    parcel_count: int = 500,
    data_key: str | None = None,
) -> dict[str, Any]:
    count = int(parcel_count)
    key = data_key or default_data_key(count)
    raw = load_hcp_series(data_path, parcel_count=count, data_key=key)
    groups = load_yeo7_groups(labels_path, expected_parcels=count)
    reduced = fit_yeo7_pc1(raw[:development_end], groups).transform(raw)
    observed_fit = fit_delta_history_phi(reduced, alpha=alpha, order=order, development_end=development_end)
    null_values = []
    for replicate in range(null_replicates):
        null_series = circular_shift_null(reduced[:development_end], seed=_subject_seed(seed, replicate))
        null_fit = fit_delta_history_phi(null_series, alpha=alpha, order=order, development_end=development_end)
        null_values.append(float(null_fit["phi"]["raw_phi"]))
    observed = float(observed_fit["phi"]["raw_phi"])
    comparison = summarize_null(observed, np.asarray(null_values, dtype=float))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "subject": data_path.parent.name,
        "config": {"parcel_count": count, "data_key": key, "labels": str(labels_path), "network_sizes": {name: len(indices) for name, indices in groups.items()}, "representation": "Yeo7 network PC1 fitted on the development segment only", "development_end": int(development_end), "model": "delta Ridge", "order": int(order), "alpha": float(alpha), "source_definition": "[x_t, ..., x_{t-order+1}] (all 7 network PC1 variables at each lag)", "target_definition": "x_{t+1} (seven network PC1 variables)", "phi_definition": "EI(history; next_state) - sum over individual history-source variables EI(source_j; next_state)", "null": "independent non-zero circular shift of each of seven PC1 time series; model refit with fixed order and alpha", "null_replicates": int(null_replicates), "test_segment_used_for_fitting": False},
        "observed": {"raw_phi": observed, "joint_ei": float(observed_fit["phi"]["joint_ei"]), "singleton_ei_sum": float(observed_fit["phi"]["singleton_ei_sum"]), "n_source_variables": observed_fit["n_source_variables"], "n_target_variables": observed_fit["n_target_variables"]},
        "null_raw_phi": null_values,
        "null_comparison": comparison,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    plot_null_distribution(observed, np.asarray(null_values, dtype=float), output_dir / "null_distribution")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--parcel-count", type=int, choices=(500, 1000), default=500)
    parser.add_argument("--data-key", default="", help="MAT variable name; defaults to Schaefer<parcel-count>.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--development-end", type=int, default=900)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--null-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args(argv)
    labels = args.labels or default_yeo7_labels(args.parcel_count)
    summary = run(args.data, labels, args.output_dir, development_end=args.development_end, order=args.order, alpha=args.alpha, null_replicates=args.null_replicates, seed=args.seed, parcel_count=args.parcel_count, data_key=args.data_key or None)
    print(json.dumps({"observed_raw_phi": summary["observed"]["raw_phi"], **summary["null_comparison"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
