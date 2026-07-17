#!/usr/bin/env python3
"""Run Liang information-flow readouts on the sine beta common-driver sweep.

The experiment deliberately uses the same raw observed variables as the other
readouts in Part 1: x, y, z, and w.  It does not append an engineered sin(x*y)
feature, so the Liang readout is tested as an observational pairwise/multivariate
time-series method rather than as an oracle with access to the nonlinear term.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.linalg import logm


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_granger_peid_mlp import (  # noqa: E402
    BETA_COMMON_DRIVER_SWEEP_VALUES,
    COMMON_DRIVER_MEMORY_COEFFICIENT,
    COMMON_DRIVER_SOURCE_COEFFICIENT,
    COMMON_DRIVER_TARGET_COEFFICIENT,
    SOURCE_MEMORY_COEFFICIENT,
    TARGET_MEMORY_COEFFICIENT,
    SimConfig,
    simulate_system,
)


RESULT_DIR = ROOT / "results" / "granger_peid_mlp_comparison"
FIGURE_DIR = ROOT / "fig" / "granger_peid_mlp_comparison"
DEFAULT_RESULT_PATH = RESULT_DIR / "sine_beta_liang_information_flow.json"
DEFAULT_FIGURE_STEM = FIGURE_DIR / "sine_beta_liang_information_flow"
VARIABLES = ("x", "y", "z", "w")
SOURCES_TO_Z = ("w", "x", "y")


@dataclass(frozen=True)
class LiangFlowResult:
    flow: pd.DataFrame
    normalized: pd.DataFrame
    coefficient: pd.DataFrame
    standard_error: pd.DataFrame
    ci_low: pd.DataFrame
    ci_high: pd.DataFrame
    significant_95: pd.DataFrame
    residual_noise_rate: pd.Series


def _cov_frame(values: np.ndarray, names: Iterable[str]) -> pd.DataFrame:
    names = tuple(names)
    return pd.DataFrame(np.cov(values, rowvar=False, bias=False), index=names, columns=names)


def estimate_liang_euler(
    series: pd.DataFrame,
    *,
    variables: tuple[str, ...] = VARIABLES,
    dt: float = 1.0,
) -> LiangFlowResult:
    """Estimate Liang information flow by the continuous-time linear MLE form.

    For target i and source j, the multivariate linear estimator is
    T_{j->i} = a_{ij} C_{ij} / C_{ii}, where a_{ij} is the coefficient of source
    j in the linear model for dX_i/dt.  This follows the standard Liang linear
    Gaussian information-flow form, applied here to the one-step finite
    difference used by the existing discrete simulation.
    """

    current = series.loc[:, variables].iloc[:-1].to_numpy(dtype=float)
    future = series.loc[:, variables].iloc[1:].to_numpy(dtype=float)
    derivative = (future - current) / float(dt)
    names = tuple(variables)
    n, p = current.shape

    centered_current = current - current.mean(axis=0, keepdims=True)
    centered_derivative = derivative - derivative.mean(axis=0, keepdims=True)
    covariance = _cov_frame(current, names)
    design = np.column_stack([np.ones(n), centered_current])
    xtx_inv = np.linalg.pinv(design.T @ design)

    coefficients = pd.DataFrame(0.0, index=names, columns=names)
    coefficient_se = pd.DataFrame(0.0, index=names, columns=names)
    flow = pd.DataFrame(0.0, index=names, columns=names)
    flow_se = pd.DataFrame(0.0, index=names, columns=names)
    residual_noise_rate = pd.Series(0.0, index=names)

    for target_idx, target in enumerate(names):
        y = centered_derivative[:, target_idx]
        beta_hat, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ beta_hat
        residual = y - fitted
        dof = max(1, n - design.shape[1])
        sigma2 = float(residual.T @ residual / dof)
        beta_cov = sigma2 * xtx_inv
        target_var = float(covariance.loc[target, target])
        if target_var <= 0.0:
            raise ValueError(f"Non-positive variance for target {target!r}.")
        residual_noise_rate.loc[target] = max(0.0, sigma2 / (2.0 * target_var))

        for source_idx, source in enumerate(names):
            coef = float(beta_hat[source_idx + 1])
            coef_se = float(np.sqrt(max(0.0, beta_cov[source_idx + 1, source_idx + 1])))
            covariance_ratio = float(covariance.loc[target, source] / target_var)
            coefficients.loc[target, source] = coef
            coefficient_se.loc[target, source] = coef_se
            flow.loc[target, source] = coef * covariance_ratio
            flow_se.loc[target, source] = abs(covariance_ratio) * coef_se

    ci_low = flow - 1.96 * flow_se
    ci_high = flow + 1.96 * flow_se
    significant = (ci_low > 0.0) | (ci_high < 0.0)
    normalized = pd.DataFrame(0.0, index=names, columns=names)
    for target in names:
        denom = float(residual_noise_rate.loc[target] + abs(coefficients.loc[target, target]))
        denom += float(np.abs(flow.loc[target, [name for name in names if name != target]]).sum())
        if denom <= 0.0:
            continue
        for source in names:
            if source != target:
                normalized.loc[target, source] = flow.loc[target, source] / denom

    return LiangFlowResult(
        flow=flow,
        normalized=normalized,
        coefficient=coefficients,
        standard_error=flow_se,
        ci_low=ci_low,
        ci_high=ci_high,
        significant_95=significant,
        residual_noise_rate=residual_noise_rate,
    )


def estimate_liang_matrix_log(
    series: pd.DataFrame,
    *,
    variables: tuple[str, ...] = VARIABLES,
    dt: float = 1.0,
) -> pd.DataFrame:
    """Finite-time Liang-style matrix-log estimator for a one-step map."""

    current = series.loc[:, variables].iloc[:-1].to_numpy(dtype=float)
    future = series.loc[:, variables].iloc[1:].to_numpy(dtype=float)
    names = tuple(variables)
    current_centered = current - current.mean(axis=0, keepdims=True)
    future_centered = future - future.mean(axis=0, keepdims=True)
    covariance = np.cov(current_centered, rowvar=False, bias=False)
    cross_covariance = current_centered.T @ future_centered / (len(current_centered) - 1)
    propagator = logm(cross_covariance.T @ np.linalg.pinv(covariance)) / float(dt)
    propagator = np.real_if_close(propagator, tol=1000).real

    flow = pd.DataFrame(0.0, index=names, columns=names)
    for target_idx, target in enumerate(names):
        target_var = float(covariance[target_idx, target_idx])
        if target_var <= 0.0:
            raise ValueError(f"Non-positive variance for target {target!r}.")
        for source_idx, source in enumerate(names):
            flow.loc[target, source] = (
                float(propagator[target_idx, source_idx])
                * float(covariance[target_idx, source_idx])
                / target_var
            )
    return flow


def _mean_std(rows: pd.DataFrame, value: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean = rows.pivot(index="beta", columns="source", values=f"{value}_mean")
    std = rows.pivot(index="beta", columns="source", values=f"{value}_std")
    return mean, std


def run_experiment(
    *,
    beta_values: tuple[float, ...] = BETA_COMMON_DRIVER_SWEEP_VALUES,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    n_samples: int = 1100,
    alpha: float = 1.0,
    noise: float = 0.05,
    show_progress: bool = False,
) -> dict[str, object]:
    run_rows: list[dict[str, object]] = []
    beta_seed_pairs = [
        (float(beta), int(seed)) for beta in beta_values for seed in seeds
    ]
    if show_progress:
        from tqdm.auto import tqdm

        beta_seed_pairs = tqdm(
            beta_seed_pairs, desc="Liang simple sine beta", unit="run", mininterval=1.0
        )
    for beta, seed in beta_seed_pairs:
        config = SimConfig(
            mechanism="common_driver_sine_synergy",
            n_samples=int(n_samples),
            noise=float(noise),
            seed=int(seed),
            synergy_strength=float(alpha),
            common_driver_strength=float(beta),
        )
        series, _ = simulate_system(config)
        euler = estimate_liang_euler(series)
        matrix_log = estimate_liang_matrix_log(series)
        xy_corr = float(series["x"].corr(series["y"]))
        wz_corr = float(series["w"].corr(series["z"]))
        for source in SOURCES_TO_Z:
            run_rows.append(
                {
                    "beta": float(beta),
                    "seed": int(seed),
                    "source": source,
                    "target": "z",
                    "liang_flow": float(euler.flow.loc["z", source]),
                    "liang_flow_se": float(euler.standard_error.loc["z", source]),
                    "liang_flow_ci_low": float(euler.ci_low.loc["z", source]),
                    "liang_flow_ci_high": float(euler.ci_high.loc["z", source]),
                    "liang_flow_significant_95": bool(
                        euler.significant_95.loc["z", source]
                    ),
                    "liang_tau": float(euler.normalized.loc["z", source]),
                    "liang_matrix_log_flow": float(matrix_log.loc["z", source]),
                    "xy_corr": xy_corr,
                    "wz_corr": wz_corr,
                }
            )

    run_frame = pd.DataFrame(run_rows)
    grouped = (
        run_frame.groupby(["beta", "source"], as_index=False)
        .agg(
            liang_flow_mean=("liang_flow", "mean"),
            liang_flow_std=("liang_flow", "std"),
            liang_tau_mean=("liang_tau", "mean"),
            liang_tau_std=("liang_tau", "std"),
            liang_matrix_log_flow_mean=("liang_matrix_log_flow", "mean"),
            liang_matrix_log_flow_std=("liang_matrix_log_flow", "std"),
            significant_rate=("liang_flow_significant_95", "mean"),
            xy_corr_mean=("xy_corr", "mean"),
            wz_corr_mean=("wz_corr", "mean"),
        )
        .sort_values(["beta", "source"])
    )
    trends: dict[str, dict[str, float]] = {}
    beta_array = np.asarray(tuple(beta_values), dtype=float)
    for source in SOURCES_TO_Z:
        source_summary = grouped[grouped["source"] == source].sort_values("beta")
        trends[source] = {}
        for metric in (
            "liang_flow_mean",
            "liang_tau_mean",
            "liang_matrix_log_flow_mean",
        ):
            values = source_summary[metric].to_numpy(dtype=float)
            slope, intercept = np.polyfit(beta_array, values, deg=1)
            trends[source][f"{metric}_slope"] = float(slope)
            trends[source][f"{metric}_intercept"] = float(intercept)

    return {
        "config": {
            "mechanism": "common_driver_sine_synergy",
            "beta_values": [float(value) for value in beta_values],
            "seeds": [int(seed) for seed in seeds],
            "n_samples": int(n_samples),
            "alpha": float(alpha),
            "noise": float(noise),
            "dynamics_coefficients": {
                "w_memory": float(COMMON_DRIVER_MEMORY_COEFFICIENT),
                "x_y_memory": float(SOURCE_MEMORY_COEFFICIENT),
                "w_to_x_y": float(COMMON_DRIVER_SOURCE_COEFFICIENT),
                "z_memory": float(TARGET_MEMORY_COEFFICIENT),
                "w_to_z": float(COMMON_DRIVER_TARGET_COEFFICIENT),
                "sin_xy": float(alpha),
            },
            "variables": list(VARIABLES),
            "target": "z",
            "sources_to_z": list(SOURCES_TO_Z),
            "engineered_sin_xy_feature": False,
            "dt": 1.0,
        },
        "runs": run_frame.to_dict(orient="records"),
        "summary": grouped.to_dict(orient="records"),
        "trends": trends,
    }


def save_json(result: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_result(result: dict[str, object], figure_stem: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    summary = pd.DataFrame(result["summary"])
    colors = {"w": "#4C78A8", "x": "#F58518", "y": "#54A24B"}
    labels = {"w": "w → z", "x": "x → z", "y": "y → z"}
    panels = [
        ("liang_flow", "Euler Liang IF", "Information flow"),
        ("liang_tau", "Normalized Euler Liang IF", "Normalized flow"),
        ("liang_matrix_log_flow", "Matrix-log Liang IF", "Information flow"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.45), constrained_layout=True)
    for axis, (metric, panel_label, ylabel) in zip(axes, panels, strict=True):
        for source in SOURCES_TO_Z:
            source_summary = summary[summary["source"] == source].sort_values("beta")
            x_values = source_summary["beta"].to_numpy(dtype=float)
            mean = source_summary[f"{metric}_mean"].to_numpy(dtype=float)
            std = source_summary[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
            axis.plot(
                x_values,
                mean,
                color=colors[source],
                linewidth=1.7,
                marker="o",
                markersize=2.8,
                label=labels[source],
            )
            axis.fill_between(
                x_values,
                mean - std,
                mean + std,
                color=colors[source],
                alpha=0.16,
                linewidth=0.0,
            )
        axis.axhline(0.0, color="#666666", linewidth=0.7, linestyle="--", alpha=0.7)
        axis.set_xlabel(r"Common-driver strength $\beta$")
        axis.set_ylabel(ylabel)
        axis.text(0.02, 0.96, panel_label, transform=axis.transAxes, va="top", ha="left")
        axis.tick_params(length=3, width=0.7)
        axis.set_xlim(-0.02, 1.02)

    axes[-1].legend(loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False)
    figure_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".png": {"dpi": 600},
        ".svg": {},
        ".pdf": {},
    }.items():
        fig.savefig(figure_stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--figure-stem", type=Path, default=DEFAULT_FIGURE_STEM)
    parser.add_argument("--n-samples", type=int, default=1100)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        n_samples=args.n_samples,
        alpha=args.alpha,
        noise=args.noise,
    )
    save_json(result, args.result_path)
    plot_result(result, args.figure_stem)
    print(f"Wrote {args.result_path}")
    print(f"Wrote {args.figure_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
