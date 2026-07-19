#!/usr/bin/env python3
"""Run the simple-coefficient 4D MLP+PEID sweep on fixed w/x/y/z support."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_granger_peid_mlp import (
    BETA_COMMON_DRIVER_SWEEP_VALUES,
    DEFAULT_FIGURE_DIR,
    DEFAULT_RESULT_DIR,
    SimConfig,
    _intervention_features,
    _plot_sine_beta_combined_readout_sweep,
    make_lagged_dataset,
    simulate_system,
    train_mlp_transition_model,
)


DEFAULT_FULL_RESULT = DEFAULT_RESULT_DIR / "sine_beta_simple_coefficients_full_state.json"
DEFAULT_LIANG_RESULT = DEFAULT_RESULT_DIR / "sine_beta_simple_coefficients_liang.json"
DEFAULT_RESULT_PATH = (
    DEFAULT_RESULT_DIR / "sine_beta_simple_coefficients_wxyz_mlp_fixed_support.json"
)
DEFAULT_FIGURE_STEM = "sine_beta_simple_coefficients_wxyz_mlp_fixed_support"
DEFAULT_COMPARISON_STEM = "sine_beta_wxyz_mlp_support_comparison"
DEFAULT_SUPPORTS: dict[str, tuple[float, float]] = {
    "x": (-1.8, 1.8),
    "y": (-1.8, 1.8),
    "z": (-1.25, 1.25),
    "w": (-1.0, 1.0),
}


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_fixed_intervention_states(
    *,
    variable_names: Sequence[str],
    supports: Mapping[str, tuple[float, float]],
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Draw paired readout states from beta-invariant per-variable supports."""

    if int(n_samples) < 16:
        raise ValueError("n_samples must be at least 16")
    missing = [name for name in variable_names if name not in supports]
    if missing:
        raise ValueError(f"Missing fixed supports for {missing}")
    rng = np.random.default_rng(int(seed) + 1009)
    columns: dict[str, np.ndarray] = {}
    for name in variable_names:
        low, high = (float(value) for value in supports[name])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"Invalid fixed support for {name!r}: {(low, high)}")
        columns[name] = rng.uniform(low, high, size=int(n_samples))
    return pd.DataFrame(columns, columns=list(variable_names))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - residual / total) if total > 1e-12 else float("nan")


def _tv_metrics(beta: np.ndarray, values: np.ndarray) -> dict[str, float]:
    order = np.argsort(beta)
    ordered_beta = np.asarray(beta, dtype=float)[order]
    ordered_values = np.asarray(values, dtype=float)[order]
    beta_span = float(ordered_beta[-1] - ordered_beta[0])
    if beta_span <= 0.0:
        return {"absolute_tv": float("nan"), "relative_tv": float("nan")}
    absolute_tv = float(np.sum(np.abs(np.diff(ordered_values))) / beta_span)
    mean_abs = float(np.mean(np.abs(ordered_values)))
    return {
        "absolute_tv": absolute_tv,
        "relative_tv": float(absolute_tv / mean_abs) if mean_abs > 1e-12 else float("inf"),
        "mean_abs": mean_abs,
        "minimum": float(np.min(ordered_values)),
        "maximum": float(np.max(ordered_values)),
    }


def _curve_metrics(
    summary: Sequence[Mapping[str, object]],
    column: str,
) -> dict[str, float]:
    rows = sorted(summary, key=lambda row: float(row["beta"]))
    beta = np.asarray([float(row["beta"]) for row in rows])
    values = np.asarray([float(row[column]) for row in rows])
    return _tv_metrics(beta, values)


def calculate_oracle_free_sensitivity(
    fixed_result: Mapping[str, object],
    baseline_result: Mapping[str, object],
) -> dict[str, object]:
    """Calculate absolute and self-normalized TV without using Oracle values."""

    fixed_summary = list(fixed_result["summary"])
    baseline_summary = list(baseline_result["summary"])
    return {
        "definition": {
            "absolute_tv": "sum(abs(delta curve)) / (beta_max - beta_min)",
            "relative_tv": "absolute_tv / mean(abs(curve))",
            "oracle_used": False,
        },
        "synergy": {
            "MLP+PEID (4D fixed support)": _curve_metrics(
                fixed_summary, "mlp_peid_xy_synergy_mean"
            ),
            "MMI-PID": _curve_metrics(baseline_summary, "mmi_pid_xy_synergy_mean"),
            "SURD": _curve_metrics(baseline_summary, "surd_xy_synergy_mean"),
            "Observational WMS": _curve_metrics(baseline_summary, "observational_wms_mean"),
        },
        "unique_x": {
            "MLP+PEID (4D fixed support)": _curve_metrics(
                fixed_summary, "mlp_peid_unique_x_mean"
            ),
            "MMI-PID": _curve_metrics(baseline_summary, "mmi_pid_unique_x_mean"),
            "SURD": _curve_metrics(baseline_summary, "surd_unique_x_mean"),
        },
        "unique_y": {
            "MLP+PEID (4D fixed support)": _curve_metrics(
                fixed_summary, "mlp_peid_unique_y_mean"
            ),
            "MMI-PID": _curve_metrics(baseline_summary, "mmi_pid_unique_y_mean"),
            "SURD": _curve_metrics(baseline_summary, "surd_unique_y_mean"),
        },
        "mlp_support_ablation": {
            "empirical_context_synergy": _curve_metrics(
                baseline_summary, "mlp_peid_xy_synergy_mean"
            ),
            "fixed_4d_support_synergy": _curve_metrics(
                fixed_summary, "mlp_peid_xy_synergy_mean"
            ),
        },
    }


def run_fixed_support_sweep(
    *,
    beta_values: Sequence[float] = BETA_COMMON_DRIVER_SWEEP_VALUES,
    seeds: Sequence[int] = (0, 1, 2, 3),
    supports: Mapping[str, tuple[float, float]] = DEFAULT_SUPPORTS,
    n_samples: int = 1100,
    noise: float = 0.05,
    mlp_epochs: int = 90,
    intervention_samples: int = 640,
    bins: int = 4,
    show_progress: bool = True,
) -> dict[str, object]:
    from yrd.transport_map import summarize_two_source_synergy_transport_map

    pairs: Sequence[tuple[float, int]] = [
        (float(beta), int(seed)) for beta in beta_values for seed in seeds
    ]
    if show_progress:
        from tqdm.auto import tqdm

        pairs = tqdm(pairs, desc="fixed-support 4D MLP+PEID", unit="run", mininterval=1.0)

    rows: list[dict[str, float | str]] = []
    for beta, seed in pairs:
        config = SimConfig(
            mechanism="common_driver_sine_synergy",
            n_samples=int(n_samples),
            noise=float(noise),
            seed=int(seed),
            synergy_strength=1.0,
            common_driver_strength=float(beta),
            mlp_epochs=int(mlp_epochs),
            intervention_samples=int(intervention_samples),
            bins=int(bins),
        )
        series, _ = simulate_system(config)
        features, targets = make_lagged_dataset(series, lag=config.lag)
        model = train_mlp_transition_model(features, targets, config)
        fixed_states = sample_fixed_intervention_states(
            variable_names=config.variable_names,
            supports=supports,
            n_samples=int(intervention_samples),
            seed=int(seed),
        )
        intervention_predictions = model.predict(_intervention_features(fixed_states, config))
        target_index = config.variable_names.index("z")
        peid = summarize_two_source_synergy_transport_map(
            fixed_states[["x"]].to_numpy(dtype=float),
            fixed_states[["y"]].to_numpy(dtype=float),
            intervention_predictions[:, [target_index]],
        )
        train_predictions = model.predict(features)[:, target_index]
        rows.append(
            {
                "run_id": f"beta={beta:.2f}|seed={seed}",
                "beta": beta,
                "seed": float(seed),
                "mlp_peid_unique_x": float(peid["left_ei"]),
                "mlp_peid_unique_y": float(peid["right_ei"]),
                "mlp_peid_xy_synergy": float(peid["syn"]),
                "mlp_peid_xy_joint": float(peid["joint_ei"]),
                "final_train_loss": float(model.loss_history[-1]),
                "z_train_r2": _r2_score(targets[:, target_index], train_predictions),
            }
        )

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("beta", as_index=False)
        .agg(
            mlp_peid_unique_x_mean=("mlp_peid_unique_x", "mean"),
            mlp_peid_unique_x_std=("mlp_peid_unique_x", "std"),
            mlp_peid_unique_y_mean=("mlp_peid_unique_y", "mean"),
            mlp_peid_unique_y_std=("mlp_peid_unique_y", "std"),
            mlp_peid_xy_synergy_mean=("mlp_peid_xy_synergy", "mean"),
            mlp_peid_xy_synergy_std=("mlp_peid_xy_synergy", "std"),
            mlp_peid_xy_joint_mean=("mlp_peid_xy_joint", "mean"),
            mlp_peid_xy_joint_std=("mlp_peid_xy_joint", "std"),
            final_train_loss_mean=("final_train_loss", "mean"),
            final_train_loss_std=("final_train_loss", "std"),
            z_train_r2_mean=("z_train_r2", "mean"),
            z_train_r2_std=("z_train_r2", "std"),
        )
        .sort_values("beta")
    )
    beta = summary["beta"].to_numpy(dtype=float)
    trend: dict[str, float] = {}
    for name in ("mlp_peid_unique_x", "mlp_peid_unique_y", "mlp_peid_xy_synergy"):
        values = summary[f"{name}_mean"].to_numpy(dtype=float)
        trend[f"{name}_slope"] = float(np.polyfit(beta, values, 1)[0]) if len(beta) > 1 else float("nan")

    return {
        "config": {
            "scientific_question": "What changes when only the 4D MLP intervention support is fixed across beta?",
            "beta_values": [float(value) for value in beta_values],
            "seeds": [int(value) for value in seeds],
            "n_samples": int(n_samples),
            "noise": float(noise),
            "mlp_epochs": int(mlp_epochs),
            "intervention_samples": int(intervention_samples),
            "intervention_support": {
                name: [float(bound) for bound in supports[name]]
                for name in ("w", "x", "y", "z")
            },
            "readout_sampling": "per-seed fixed uniform points reused across every beta",
            "observed_variables": ["w", "x", "y", "z"],
            "hidden_variables": [],
        },
        "runs": frame.to_dict("records"),
        "summary": summary.to_dict("records"),
        "trend": trend,
    }


def plot_support_comparison(
    fixed_result: Mapping[str, object],
    baseline_result: Mapping[str, object],
    figure_dir: Path,
    *,
    stem: str = DEFAULT_COMPARISON_STEM,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixed = pd.DataFrame(fixed_result["summary"]).sort_values("beta")
    baseline = pd.DataFrame(baseline_result["summary"]).sort_values("beta")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(6.3, 3.0), constrained_layout=True)
    specs = [
        (baseline, "Empirical $w,z$ context support", "#8C8C8C", "o"),
        (fixed, "Fixed $w,x,y,z$ intervention support", "#009E73", "s"),
    ]
    for frame, label, color, marker in specs:
        x = frame["beta"].to_numpy(dtype=float)
        y = frame["mlp_peid_xy_synergy_mean"].to_numpy(dtype=float)
        sd = frame["mlp_peid_xy_synergy_std"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            markevery=max(1, (len(x) - 1) // 5),
            linewidth=1.8,
            markersize=3.4,
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=label,
        )
        ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.14, linewidth=0)
    ax.set_xlabel(r"$\beta$: common-driver strength")
    ax.set_ylabel("MLP+PEID synergy (bits)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / f"{stem}.png"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    return png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--full-result", type=Path, default=DEFAULT_FULL_RESULT)
    parser.add_argument("--liang-result", type=Path, default=DEFAULT_LIANG_RESULT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = _read_json(args.full_result)
    liang = _read_json(args.liang_result)
    sweep_kwargs: dict[str, object] = {}
    if args.smoke:
        sweep_kwargs.update(
            beta_values=(0.0, 1.0),
            seeds=(0,),
            n_samples=320,
            mlp_epochs=5,
            intervention_samples=64,
        )
    result = run_fixed_support_sweep(**sweep_kwargs)
    result["sensitivity"] = calculate_oracle_free_sensitivity(result, baseline)
    _write_json(args.result_path, result)

    combined = _plot_sine_beta_combined_readout_sweep(
        baseline,
        args.figure_dir,
        liang_result=liang,
        mlp_readout_result=result,
        stem=DEFAULT_FIGURE_STEM + ("_smoke" if args.smoke else ""),
    )
    comparison = plot_support_comparison(
        result,
        baseline,
        args.figure_dir,
        stem=DEFAULT_COMPARISON_STEM + ("_smoke" if args.smoke else ""),
    )
    print(
        json.dumps(
            {
                "result": str(args.result_path),
                "combined_figure": str(combined),
                "comparison_figure": str(comparison),
                "fixed_support_synergy": result["sensitivity"]["synergy"][
                    "MLP+PEID (4D fixed support)"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
