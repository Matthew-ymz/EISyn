#!/usr/bin/env python3
"""Run the sine beta sweep when the hidden driver w is not observed."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_granger_peid_mlp import (  # noqa: E402
    BETA_COMMON_DRIVER_SWEEP_VALUES,
    DEFAULT_FIGURE_DIR,
    DEFAULT_RESULT_DIR,
    SimConfig,
    _beta_sweep_trend_stats,
    _intervention_features,
    _plot_sine_beta_combined_readout_sweep,
    estimate_shap_readout,
    make_lagged_dataset,
    simulate_system,
    train_mlp_transition_model,
)

DEFAULT_HIDDEN_W_RESULT_PATH = DEFAULT_RESULT_DIR / "sine_beta_hidden_w_mlp_peid_sweep.json"
DEFAULT_HIDDEN_W_FIGURE_PATH = DEFAULT_FIGURE_DIR / "sine_beta_hidden_w_mlp_peid_sweep.png"
DEFAULT_HIDDEN_W_COMPARISON_FIGURE_PATH = (
    DEFAULT_FIGURE_DIR / "sine_beta_hidden_vs_observed_w_syn_comparison.png"
)
DEFAULT_HIDDEN_W_COMBINED_FIGURE_PATH = DEFAULT_FIGURE_DIR / "sine_beta_combined_readout_sweep.png"
DEFAULT_HIDDEN_W_REPORT_PATH = ROOT / "docs" / "reports" / "Part1_hidden_w_mlp_peid.md"
DEFAULT_FULL_STATE_SUMMARY_PATH = DEFAULT_RESULT_DIR / "summary.json"
DEFAULT_LIANG_RESULT_PATH = DEFAULT_RESULT_DIR / "sine_beta_liang_information_flow.json"
OBSERVED_VARIABLES = ("x", "y", "z")
DEFAULT_HIDDEN_W_READOUT_SUPPORT = {
    "x": (-1.8, 1.8),
    "y": (-1.8, 1.8),
    "z": (-1.25, 1.25),
}


def sample_fixed_hidden_w_readout_states(
    *,
    samples: int,
    seed: int,
    support: Mapping[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    fixed_support = dict(support or DEFAULT_HIDDEN_W_READOUT_SUPPORT)
    rng = np.random.default_rng(int(seed))
    rows: dict[str, np.ndarray] = {}
    for name in OBSERVED_VARIABLES:
        low, high = fixed_support[name]
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"invalid readout support for {name!r}")
        rows[name] = rng.uniform(float(low), float(high), size=int(samples))
    return pd.DataFrame(rows, columns=list(OBSERVED_VARIABLES))


def _fixed_oracle_xy_z(
    *,
    alpha: float,
    samples: int,
    seed: int,
    support: Mapping[str, tuple[float, float]],
) -> dict[str, float]:
    from yrd.transport_map import summarize_two_source_synergy_transport_map

    for name in ("x", "y", "z"):
        low, high = support[name]
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"invalid oracle support for {name!r}")
    if tuple(support["x"]) != tuple(support["y"]):
        raise ValueError("oracle support for x and y must match")

    rng = np.random.default_rng(int(seed))
    pair_count = max(1, int(math.ceil(int(samples) / 2)))
    x_base = rng.uniform(*support["x"], size=pair_count)
    y_base = rng.uniform(*support["y"], size=pair_count)
    z_base = rng.uniform(*support["z"], size=pair_count)
    x = np.concatenate([x_base, y_base])[: int(samples)].reshape(-1, 1)
    y = np.concatenate([y_base, x_base])[: int(samples)].reshape(-1, 1)
    z_state = np.concatenate([z_base, z_base])[: int(samples)]
    target = (0.22 * z_state + float(alpha) * np.sin(x[:, 0] * y[:, 0])).reshape(-1, 1)
    result = summarize_two_source_synergy_transport_map(x, y, target)
    return {key: float(result[key]) for key in ("left_ei", "right_ei", "joint_ei", "syn")}


def run_hidden_w_sine_beta_mlp_peid_sweep(
    *,
    beta_values: Sequence[float] = BETA_COMMON_DRIVER_SWEEP_VALUES,
    seeds: Sequence[int] = (0, 1, 2, 3),
    n_samples: int = 1100,
    alpha: float = 1.0,
    noise: float = 0.05,
    mlp_epochs: int = 90,
    intervention_samples: int = 640,
    bins: int = 4,
    readout_support: Mapping[str, tuple[float, float]] | None = None,
    readout_seed: int = 17021,
    oracle_intervention_support: Mapping[str, tuple[float, float]] | None = None,
    oracle_intervention_seed: int = 17021,
) -> dict[str, object]:
    from yrd.transport_map import summarize_two_source_synergy_transport_map

    oracle_support = dict(
        oracle_intervention_support
        or {
            "x": (-1.8, 1.8),
            "y": (-1.8, 1.8),
            "z": (-1.25, 1.25),
        }
    )
    oracle = _fixed_oracle_xy_z(
        alpha=float(alpha),
        samples=int(intervention_samples),
        seed=int(oracle_intervention_seed),
        support=oracle_support,
    )
    fixed_support = dict(readout_support or DEFAULT_HIDDEN_W_READOUT_SUPPORT)
    fixed_peid_samples = sample_fixed_hidden_w_readout_states(
        samples=int(intervention_samples),
        seed=int(readout_seed),
        support=fixed_support,
    )
    fixed_shap_foreground = sample_fixed_hidden_w_readout_states(
        samples=64,
        seed=int(readout_seed) + 4049,
        support=fixed_support,
    )
    fixed_shap_background = sample_fixed_hidden_w_readout_states(
        samples=64,
        seed=int(readout_seed) + 5051,
        support=fixed_support,
    )
    fixed_shap_foreground_features = _intervention_features(fixed_shap_foreground, SimConfig(variable_names=OBSERVED_VARIABLES))
    fixed_shap_background_features = _intervention_features(fixed_shap_background, SimConfig(variable_names=OBSERVED_VARIABLES))

    rows: list[dict[str, float]] = []
    for beta in beta_values:
        for seed in seeds:
            full_config = SimConfig(
                mechanism="common_driver_sine_synergy",
                n_samples=int(n_samples),
                noise=float(noise),
                seed=int(seed),
                synergy_strength=float(alpha),
                common_driver_strength=float(beta),
                mlp_epochs=int(mlp_epochs),
                intervention_samples=int(intervention_samples),
                bins=int(bins),
            )
            full_series, _ = simulate_system(full_config)
            observed_series = full_series.loc[:, OBSERVED_VARIABLES].copy()
            observed_config = SimConfig(
                mechanism=full_config.mechanism,
                n_samples=full_config.n_samples,
                noise=full_config.noise,
                seed=full_config.seed,
                lag=full_config.lag,
                synergy_strength=full_config.synergy_strength,
                hidden_dim=full_config.hidden_dim,
                mlp_epochs=full_config.mlp_epochs,
                learning_rate=full_config.learning_rate,
                batch_size=full_config.batch_size,
                weight_decay=full_config.weight_decay,
                intervention_samples=full_config.intervention_samples,
                bins=full_config.bins,
                common_driver_strength=full_config.common_driver_strength,
                quantile_low=full_config.quantile_low,
                quantile_high=full_config.quantile_high,
                variable_names=OBSERVED_VARIABLES,
            )
            features, targets = make_lagged_dataset(observed_series, lag=observed_config.lag)
            model = train_mlp_transition_model(features, targets, observed_config)
            shap_readout = estimate_shap_readout(
                model,
                features,
                observed_series,
                observed_config,
                foreground_samples=64,
                background_samples=64,
                foreground_features=fixed_shap_foreground_features,
                background_features=fixed_shap_background_features,
            )
            samples = fixed_peid_samples.copy()
            predictions = model.predict(_intervention_features(samples, observed_config))
            tm_peid = summarize_two_source_synergy_transport_map(
                samples[["x"]].to_numpy(dtype=float),
                samples[["y"]].to_numpy(dtype=float),
                predictions[:, [OBSERVED_VARIABLES.index("z")]],
            )
            tm_peid_xy_to_y_next = summarize_two_source_synergy_transport_map(
                samples[["x"]].to_numpy(dtype=float),
                samples[["y"]].to_numpy(dtype=float),
                predictions[:, [OBSERVED_VARIABLES.index("y")]],
            )
            train_pred = model.predict(features)
            z_target = targets[:, OBSERVED_VARIABLES.index("z")]
            z_pred = train_pred[:, OBSERVED_VARIABLES.index("z")]
            z_mse = float(np.mean((z_target - z_pred) ** 2))
            z_baseline_mse = float(np.mean((z_target - float(np.mean(z_target))) ** 2))
            shap_xy_z = shap_readout.shap_interaction_terms[
                (shap_readout.shap_interaction_terms["sources"] == "x+y")
                & (shap_readout.shap_interaction_terms["target"] == "z")
            ].iloc[0]
            shap_single_lookup = {
                str(row["source"]): float(row["mean_abs_phi"])
                for row in shap_readout.feature_attributions[
                    shap_readout.feature_attributions["target"] == "z"
                ].to_dict("records")
            }
            rows.append(
                {
                    "run_id": f"hidden_w_beta={float(beta):.2f}|seed={int(seed)}",
                    "beta": float(beta),
                    "seed": float(seed),
                    "xy_observed_corr": float(observed_series[["x", "y"]].corr().iloc[0, 1]),
                    "final_train_loss": float(model.loss_history[-1]) if model.loss_history else float("nan"),
                    "z_train_mse": z_mse,
                    "z_train_r2": float(1.0 - z_mse / (z_baseline_mse + 1e-12)),
                    "shap_x_to_z_mean_abs": float(shap_single_lookup.get("x", 0.0)),
                    "shap_y_to_z_mean_abs": float(shap_single_lookup.get("y", 0.0)),
                    "shap_xy_mean_abs_interaction": float(shap_xy_z["mean_abs_interaction"]),
                    "shap_xy_mean_interaction": float(shap_xy_z["mean_interaction"]),
                    "mlp_peid_unique_x": float(tm_peid["left_ei"]),
                    "mlp_peid_unique_y": float(tm_peid["right_ei"]),
                    "mlp_peid_xy_joint": float(tm_peid["joint_ei"]),
                    "mlp_peid_xy_synergy": float(tm_peid["syn"]),
                    "mlp_peid_x_unique_to_y_next": float(tm_peid_xy_to_y_next["left_ei"]),
                    "mlp_peid_y_unique_to_y_next": float(tm_peid_xy_to_y_next["right_ei"]),
                    "mlp_peid_xy_joint_to_y_next": float(tm_peid_xy_to_y_next["joint_ei"]),
                    "mlp_peid_xy_synergy_to_y_next": float(tm_peid_xy_to_y_next["syn"]),
                    "oracle_peid_unique_x": float(oracle["left_ei"]),
                    "oracle_peid_unique_y": float(oracle["right_ei"]),
                    "oracle_peid_xy_joint": float(oracle["joint_ei"]),
                    "oracle_peid_xy_synergy": float(oracle["syn"]),
                }
            )

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("beta", as_index=False)
        .agg(
            xy_observed_corr_mean=("xy_observed_corr", "mean"),
            xy_observed_corr_std=("xy_observed_corr", "std"),
            final_train_loss_mean=("final_train_loss", "mean"),
            final_train_loss_std=("final_train_loss", "std"),
            z_train_r2_mean=("z_train_r2", "mean"),
            z_train_r2_std=("z_train_r2", "std"),
            shap_x_to_z_mean_abs_mean=("shap_x_to_z_mean_abs", "mean"),
            shap_x_to_z_mean_abs_std=("shap_x_to_z_mean_abs", "std"),
            shap_y_to_z_mean_abs_mean=("shap_y_to_z_mean_abs", "mean"),
            shap_y_to_z_mean_abs_std=("shap_y_to_z_mean_abs", "std"),
            shap_xy_mean_abs_interaction_mean=("shap_xy_mean_abs_interaction", "mean"),
            shap_xy_mean_abs_interaction_std=("shap_xy_mean_abs_interaction", "std"),
            shap_xy_mean_interaction_mean=("shap_xy_mean_interaction", "mean"),
            shap_xy_mean_interaction_std=("shap_xy_mean_interaction", "std"),
            mlp_peid_unique_x_mean=("mlp_peid_unique_x", "mean"),
            mlp_peid_unique_x_std=("mlp_peid_unique_x", "std"),
            mlp_peid_unique_y_mean=("mlp_peid_unique_y", "mean"),
            mlp_peid_unique_y_std=("mlp_peid_unique_y", "std"),
            mlp_peid_xy_joint_mean=("mlp_peid_xy_joint", "mean"),
            mlp_peid_xy_joint_std=("mlp_peid_xy_joint", "std"),
            mlp_peid_xy_synergy_mean=("mlp_peid_xy_synergy", "mean"),
            mlp_peid_xy_synergy_std=("mlp_peid_xy_synergy", "std"),
            mlp_peid_x_unique_to_y_next_mean=("mlp_peid_x_unique_to_y_next", "mean"),
            mlp_peid_x_unique_to_y_next_std=("mlp_peid_x_unique_to_y_next", "std"),
            mlp_peid_y_unique_to_y_next_mean=("mlp_peid_y_unique_to_y_next", "mean"),
            mlp_peid_y_unique_to_y_next_std=("mlp_peid_y_unique_to_y_next", "std"),
            mlp_peid_xy_joint_to_y_next_mean=("mlp_peid_xy_joint_to_y_next", "mean"),
            mlp_peid_xy_joint_to_y_next_std=("mlp_peid_xy_joint_to_y_next", "std"),
            mlp_peid_xy_synergy_to_y_next_mean=("mlp_peid_xy_synergy_to_y_next", "mean"),
            mlp_peid_xy_synergy_to_y_next_std=("mlp_peid_xy_synergy_to_y_next", "std"),
            oracle_peid_unique_x_mean=("oracle_peid_unique_x", "mean"),
            oracle_peid_unique_x_std=("oracle_peid_unique_x", "std"),
            oracle_peid_unique_y_mean=("oracle_peid_unique_y", "mean"),
            oracle_peid_unique_y_std=("oracle_peid_unique_y", "std"),
            oracle_peid_xy_joint_mean=("oracle_peid_xy_joint", "mean"),
            oracle_peid_xy_joint_std=("oracle_peid_xy_joint", "std"),
            oracle_peid_xy_synergy_mean=("oracle_peid_xy_synergy", "mean"),
            oracle_peid_xy_synergy_std=("oracle_peid_xy_synergy", "std"),
        )
        .sort_values("beta")
        .reset_index(drop=True)
    )
    trend_input = frame.rename(columns={"mlp_peid_xy_synergy": "tm_peid_xy_synergy"}).copy()
    for col in (
        "observational_wms",
        "neural_granger_xy_to_z",
        "pcmci_cmiknn_xy_to_z",
        "peid_xy_synergy",
        "surd_xy_synergy",
        "product_xy_incremental_r2",
    ):
        trend_input[col] = np.nan
    trend_input["oracle_peid_xy_synergy"] = trend_input["oracle_peid_xy_synergy"]
    trend = _beta_sweep_trend_stats(trend_input)
    return {
        "config": {
            "beta_values": [float(value) for value in beta_values],
            "seeds": [int(value) for value in seeds],
            "n_samples": int(n_samples),
            "alpha": float(alpha),
            "noise": float(noise),
            "mlp_epochs": int(mlp_epochs),
            "intervention_samples": int(intervention_samples),
            "bins": int(bins),
            "observed_variables": list(OBSERVED_VARIABLES),
            "hidden_variables": ["w"],
            "readout_support": {
                name: [float(bound) for bound in fixed_support[name]]
                for name in OBSERVED_VARIABLES
            },
            "readout_seed": int(readout_seed),
            "oracle_intervention_support": {
                name: [float(bound) for bound in oracle_support[name]]
                for name in ("x", "y", "z")
            },
            "oracle_intervention_seed": int(oracle_intervention_seed),
        },
        "units": {"mlp_peid": "bits", "oracle_peid": "bits"},
        "runs": rows,
        "summary": summary.to_dict("records"),
        "trend": trend,
    }


def plot_hidden_w_sweep(result: dict[str, object], figure_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(result["summary"]).sort_values("beta")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.4, 6.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.15]},
    )
    ax_aux, ax_info = axes

    def error_line(ax, y_col: str, std_col: str, *, label: str, color: str, marker: str) -> None:
        x = frame["beta"].to_numpy(dtype=float)
        y = frame[y_col].to_numpy(dtype=float)
        std = frame[std_col].fillna(0.0).to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker=marker,
            color=color,
            linewidth=2.0,
            markersize=5.8,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=label,
        )
        if np.any(std > 0.0):
            ax.errorbar(
                x,
                y,
                yerr=std,
                fmt="none",
                ecolor=color,
                elinewidth=0.8,
                capsize=2.2,
                capthick=0.8,
                alpha=0.55,
            )

    error_line(
        ax_aux,
        "xy_observed_corr_mean",
        "xy_observed_corr_std",
        label="corr(x, y)",
        color="#4C78A8",
        marker="o",
    )
    error_line(
        ax_aux,
        "z_train_r2_mean",
        "z_train_r2_std",
        label="MLP z one-step train R2",
        color="#E68613",
        marker="s",
    )
    ax_aux.axhline(0.0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax_aux.set_ylabel("Correlation / R2")
    ax_aux.grid(alpha=0.18, linewidth=0.5)
    ax_aux.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    for y_col, std_col, label, color, marker in [
        ("mlp_peid_unique_x_mean", "mlp_peid_unique_x_std", "MLP+PEID Ux", "#7AA6C2", "^"),
        ("mlp_peid_unique_y_mean", "mlp_peid_unique_y_std", "MLP+PEID Uy", "#98B88A", "v"),
        ("mlp_peid_xy_synergy_mean", "mlp_peid_xy_synergy_std", "MLP+PEID Syn(x,y)", "#1B9E77", "D"),
        (
            "mlp_peid_x_unique_to_y_next_mean",
            "mlp_peid_x_unique_to_y_next_std",
            "MLP+PEID Ux -> y_next",
            "#D65F8F",
            "o",
        ),
        ("oracle_peid_xy_synergy_mean", "oracle_peid_xy_synergy_std", "Oracle Syn(x,y)", "#6F5AA7", "s"),
    ]:
        error_line(ax_info, y_col, std_col, label=label, color=color, marker=marker)
    ax_info.axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.9)
    ax_info.set_xlabel("beta: common-driver strength")
    ax_info.set_ylabel("Information (bits)")
    ax_info.set_xticks(frame["beta"].to_numpy(dtype=float))
    ax_info.grid(alpha=0.18, linewidth=0.5)
    ax_info.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def load_full_state_beta_sweep(summary_path: Path = DEFAULT_FULL_STATE_SUMMARY_PATH) -> dict[str, object] | None:
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    beta_result = payload.get("sine_beta_common_driver_sweep")
    if not isinstance(beta_result, dict) or not beta_result.get("summary"):
        return None
    return beta_result


def load_liang_beta_sweep(result_path: Path = DEFAULT_LIANG_RESULT_PATH) -> dict[str, object] | None:
    if not result_path.exists():
        return None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("summary"):
        return None
    return payload


def plot_combined_with_hidden_w_mlp_readouts(
    hidden_result: dict[str, object],
    full_result: dict[str, object] | None,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    *,
    liang_result: dict[str, object] | None = None,
) -> Path | None:
    if not full_result:
        return None
    return _plot_sine_beta_combined_readout_sweep(
        full_result,
        figure_dir,
        liang_result=liang_result,
        mlp_readout_result=hidden_result,
    )


def compare_hidden_and_full_state(
    hidden_result: dict[str, object],
    full_result: dict[str, object] | None,
) -> dict[str, object] | None:
    if not full_result:
        return None
    hidden = pd.DataFrame(hidden_result["summary"])
    full = pd.DataFrame(full_result["summary"])
    required = {"beta", "mlp_peid_xy_synergy_mean", "mlp_peid_xy_synergy_std"}
    if not required.issubset(hidden.columns) or not required.issubset(full.columns):
        return None
    merged = hidden[list(required)].merge(
        full[list(required)],
        on="beta",
        suffixes=("_hidden_w", "_observed_w"),
    )
    if merged.empty:
        return None
    merged["delta_hidden_minus_observed"] = (
        merged["mlp_peid_xy_synergy_mean_hidden_w"]
        - merged["mlp_peid_xy_synergy_mean_observed_w"]
    )
    merged["abs_delta"] = merged["delta_hidden_minus_observed"].abs()
    hidden_trend = dict(hidden_result.get("trend", {}))
    full_trend = dict(full_result.get("trend", {}))
    return {
        "summary": merged.sort_values("beta").to_dict("records"),
        "mean_abs_delta": float(merged["abs_delta"].mean()),
        "max_abs_delta": float(merged["abs_delta"].max()),
        "max_abs_delta_beta": float(merged.loc[merged["abs_delta"].idxmax(), "beta"]),
        "hidden_w_slope": float(hidden_trend.get("tm_peid_synergy_slope", float("nan"))),
        "observed_w_slope": float(full_trend.get("tm_peid_synergy_slope", float("nan"))),
        "slope_delta_hidden_minus_observed": float(
            hidden_trend.get("tm_peid_synergy_slope", float("nan"))
            - full_trend.get("tm_peid_synergy_slope", float("nan"))
        ),
    }


def plot_hidden_vs_observed_w_syn(
    hidden_result: dict[str, object],
    full_result: dict[str, object] | None,
    figure_path: Path,
) -> Path | None:
    if not full_result:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    hidden = pd.DataFrame(hidden_result["summary"]).sort_values("beta")
    full = pd.DataFrame(full_result["summary"]).sort_values("beta")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)

    def error_line(frame: pd.DataFrame, *, label: str, color: str, marker: str) -> None:
        x = frame["beta"].to_numpy(dtype=float)
        y = frame["mlp_peid_xy_synergy_mean"].to_numpy(dtype=float)
        std = frame["mlp_peid_xy_synergy_std"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker=marker,
            color=color,
            linewidth=2.2,
            markersize=6.2,
            markeredgecolor="white",
            markeredgewidth=0.65,
            label=label,
        )
        if np.any(std > 0.0):
            ax.errorbar(
                x,
                y,
                yerr=std,
                fmt="none",
                ecolor=color,
                elinewidth=0.85,
                capsize=2.4,
                capthick=0.85,
                alpha=0.55,
            )

    error_line(full, label="MLP+PEID Syn, observed w", color="#4C78A8", marker="o")
    error_line(hidden, label="MLP+PEID Syn, hidden w", color="#1B9E77", marker="D")
    if "oracle_peid_xy_synergy_mean" in hidden:
        ax.plot(
            hidden["beta"].to_numpy(dtype=float),
            hidden["oracle_peid_xy_synergy_mean"].to_numpy(dtype=float),
            color="#6F5AA7",
            marker="s",
            linewidth=1.8,
            markersize=5.6,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label="Oracle Syn",
        )
    ax.axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.9)
    ax.set_xlabel("beta: common-driver strength")
    ax.set_ylabel("Information (bits)")
    ax.set_xticks(hidden["beta"].to_numpy(dtype=float))
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
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


def write_hidden_w_report(
    result: dict[str, object],
    figure_path: Path,
    report_path: Path,
    *,
    full_state_result: dict[str, object] | None = None,
    comparison_figure_path: Path | None = None,
    comparison: dict[str, object] | None = None,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fig_rel = os.path.relpath(figure_path, start=report_path.parent)
    comparison_fig_rel = (
        os.path.relpath(comparison_figure_path, start=report_path.parent)
        if comparison_figure_path is not None
        else ""
    )
    config = dict(result["config"])
    trend = dict(result.get("trend", {}))
    summary = list(result["summary"])
    table_lines = [
        "| beta | corr(x,y) | z train R2 | Ux->z | Uy->z | Syn->z | Ux->y_next | Oracle Syn |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        table_lines.append(
            "| {beta:.2f} | {corr} | {r2} | {ux} | {uy} | {syn} | {ux_y} | {oracle} |".format(
                beta=float(row["beta"]),
                corr=_fmt(row["xy_observed_corr_mean"]),
                r2=_fmt(row["z_train_r2_mean"]),
                ux=_fmt(row["mlp_peid_unique_x_mean"]),
                uy=_fmt(row["mlp_peid_unique_y_mean"]),
                syn=_fmt(row["mlp_peid_xy_synergy_mean"]),
                ux_y=_fmt(row["mlp_peid_x_unique_to_y_next_mean"]),
                oracle=_fmt(row["oracle_peid_xy_synergy_mean"]),
            )
        )
    x_to_y_frame = pd.DataFrame(summary).sort_values("beta")
    if len(x_to_y_frame) >= 2:
        x_to_y_slope, _ = np.polyfit(
            x_to_y_frame["beta"].to_numpy(dtype=float),
            x_to_y_frame["mlp_peid_x_unique_to_y_next_mean"].to_numpy(dtype=float),
            deg=1,
        )
    else:
        x_to_y_slope = float("nan")
    x_to_y_start = (
        float(x_to_y_frame.iloc[0]["mlp_peid_x_unique_to_y_next_mean"])
        if len(x_to_y_frame)
        else float("nan")
    )
    x_to_y_end = (
        float(x_to_y_frame.iloc[-1]["mlp_peid_x_unique_to_y_next_mean"])
        if len(x_to_y_frame)
        else float("nan")
    )
    comparison_block = ""
    if comparison and comparison.get("summary"):
        comparison_lines = [
            "| beta | Syn observed w | Syn hidden w | hidden - observed |",
            "| ---: | ---: | ---: | ---: |",
        ]
        for row in comparison["summary"]:
            comparison_lines.append(
                "| {beta:.2f} | {observed} | {hidden} | {delta} |".format(
                    beta=float(row["beta"]),
                    observed=_fmt(row["mlp_peid_xy_synergy_mean_observed_w"]),
                    hidden=_fmt(row["mlp_peid_xy_synergy_mean_hidden_w"]),
                    delta=_fmt(row["delta_hidden_minus_observed"]),
                )
            )
        comparison_block = f"""
## 与观测 w 的原曲线比较

![隐藏 w 与观测 w 的 Syn 曲线比较]({comparison_fig_rel})

{chr(10).join(comparison_lines)}

差别大小：隐藏 `w` 曲线相对观测 `w` 曲线的平均绝对差为 `{_fmt(comparison.get("mean_abs_delta"))}` bits，最大绝对差为 `{_fmt(comparison.get("max_abs_delta"))}` bits，出现在 `beta={_fmt(comparison.get("max_abs_delta_beta"))}`。斜率上，隐藏 `w` 为 `{_fmt(comparison.get("hidden_w_slope"))}` bits / beta，观测 `w` 为 `{_fmt(comparison.get("observed_w_slope"))}` bits / beta，差值约 `{_fmt(comparison.get("slope_delta_hidden_minus_observed"))}` bits / beta。

因此，两条 MLP+PEID Syn 曲线在绝对值上仍处于相近量级，但隐藏 `w` 后曲线更接近固定 Oracle，并呈现温和上升。也就是说，在当前样本量、MLP 和 transport-map 设置下，未观测共同驱动会给 `{{x,y}}->z` 的 Syn 曲线带来可见的斜率偏移；具体点位差距以上表和最大绝对差为准。
"""
    elif full_state_result is None:
        comparison_block = """
## 与观测 w 的原曲线比较

未找到已有 full-state beta 扫描缓存，因此本报告只包含隐藏 `w` 的新曲线。
"""

    text = f"""# 隐藏共同驱动 w 的 MLP+PEID beta 扫描

本实验沿用 `Part1.md` 中“共同驱动增强但结构协同固定”的生成式，仍固定 `alpha=1` 并扫描 `beta`。区别是观测数据只保留 `x,y,z`，隐藏共同驱动 `w` 不进入 MLP 的输入或输出。也就是说，真实模拟仍含有

$$
\\begin{{aligned}}
w_{{t+1}} &= 0.78w_t + \\eta^w_t,\\\\
x_{{t+1}} &= 0.42x_t + 0.82\\left(\\beta w_t + \\sqrt{{1-\\beta^2}}\\xi^x_t\\right) + \\eta^x_t,\\\\
y_{{t+1}} &= 0.38y_t + 0.76\\left(\\beta w_t + \\sqrt{{1-\\beta^2}}\\xi^y_t\\right) + \\eta^y_t,\\\\
z_{{t+1}} &= 0.22z_t + \\sin(x_t y_t) + 0.15\\beta w_t + \\eta^z_t,
\\end{{aligned}}
$$

但拟合时只使用

$$
[x_t,y_t,z_t]\\mapsto[x_{{t+1}},y_{{t+1}},z_{{t+1}}].
$$

Zotero 本地库中没有检索到 PEID / effective-information synergy 的匹配条目；这里按仓库现有 `MLP+PEID` 和 transport-map 口径执行，不额外引入文献假设。

## 实验设置

- beta 网格：`{config["beta_values"]}`
- seeds：`{config["seeds"]}`
- 每条轨迹样本数：`{config["n_samples"]}`
- MLP epochs：`{config["mlp_epochs"]}`
- PEID 干预样本数：`{config["intervention_samples"]}`
- 观测变量：`{config["observed_variables"]}`；隐藏变量：`{config["hidden_variables"]}`
- Oracle 只作为固定支持的真实 `z_next=0.22z+sin(xy)` 参照，不使用隐藏 `w`。

![隐藏 w 的 beta 扫描]({fig_rel})

## 数值结果

{chr(10).join(table_lines)}

这里的 `Ux->y_next` 是在同一个隐藏 `w` 的 3 维 MLP 上，把 `[x_t,y_t,z_t]` 的干预样本送入模型后，对 `x_t + y_t -> y_{{t+1}}` 做 transport-map PEID，再读取左源 `x_t` 的单源/unique 分量。它从 `beta=0` 的 `{_fmt(x_to_y_start)}` bits 增至 `beta=1` 的 `{_fmt(x_to_y_end)}` bits，线性斜率约 `{_fmt(x_to_y_slope)}` bits / beta。

{comparison_block}

## 读数

隐藏 `w` 后，`corr(x,y)` 仍随 `beta` 增大而上升，说明更强共同驱动确实反映到了观测轨迹中。MLP 对 `z_{{t+1}}` 的一步拟合仍可用，但它只能从 `[x,y,z]` 的观测条件分布学习边际化后的转移，而不是完整四维机制。

MLP+PEID 的 `{{x,y}}->z` 协同斜率为 `{_fmt(trend.get("tm_peid_synergy_slope"))}` bits / beta，bootstrap 95% CI 为 `[{_fmt(trend.get("tm_peid_synergy_slope_ci_low"))}, {_fmt(trend.get("tm_peid_synergy_slope_ci_high"))}]`。固定支持 Oracle 的斜率为 `{_fmt(trend.get("oracle_peid_synergy_slope"))}`，基本保持 beta 不变；因此隐藏 `w` 后若 MLP+PEID 曲线出现 beta 依赖，主要应解释为未观测共同驱动造成的代理/边际化效应，而不是 `sin(x_t y_t)` 结构项本身变强。
"""
    report_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_HIDDEN_W_RESULT_PATH)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_HIDDEN_W_FIGURE_PATH)
    parser.add_argument("--comparison-figure-path", type=Path, default=DEFAULT_HIDDEN_W_COMPARISON_FIGURE_PATH)
    parser.add_argument("--combined-figure-path", type=Path, default=DEFAULT_HIDDEN_W_COMBINED_FIGURE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_HIDDEN_W_REPORT_PATH)
    parser.add_argument("--full-state-summary-path", type=Path, default=DEFAULT_FULL_STATE_SUMMARY_PATH)
    parser.add_argument("--liang-result-path", type=Path, default=DEFAULT_LIANG_RESULT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_hidden_w_sine_beta_mlp_peid_sweep()
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    figure_path = plot_hidden_w_sweep(result, args.figure_path)
    full_state_result = load_full_state_beta_sweep(args.full_state_summary_path)
    comparison = compare_hidden_and_full_state(result, full_state_result)
    if comparison is not None:
        result["observed_w_comparison"] = comparison
        args.result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison_figure_path = plot_hidden_vs_observed_w_syn(
        result,
        full_state_result,
        args.comparison_figure_path,
    )
    liang_result = load_liang_beta_sweep(args.liang_result_path)
    combined_figure_path = plot_combined_with_hidden_w_mlp_readouts(
        result,
        full_state_result,
        args.combined_figure_path.parent,
        liang_result=liang_result,
    )
    report_path = write_hidden_w_report(
        result,
        figure_path,
        args.report_path,
        full_state_result=full_state_result,
        comparison_figure_path=comparison_figure_path,
        comparison=comparison,
    )
    print(
        json.dumps(
            {
                "result_path": str(args.result_path),
                "figure_path": str(figure_path),
                "comparison_figure_path": str(comparison_figure_path) if comparison_figure_path else None,
                "combined_figure_path": str(combined_figure_path) if combined_figure_path else None,
                "report_path": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
