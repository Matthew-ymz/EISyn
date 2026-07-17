#!/usr/bin/env python3
"""Compare paired 3D and 4D MLP forecasts on the simple sine-beta system."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_granger_peid_mlp import (  # noqa: E402
    BETA_COMMON_DRIVER_SWEEP_VALUES,
    DEFAULT_FIGURE_DIR,
    DEFAULT_RESULT_DIR,
    SimConfig,
    simulate_system,
    train_mlp_transition_model,
)

DEFAULT_RESULT_PATH = DEFAULT_RESULT_DIR / "sine_beta_3d_vs_4d_mlp_forecast.json"
DEFAULT_FIGURE_STEM = DEFAULT_FIGURE_DIR / "sine_beta_3d_vs_4d_mlp_forecast"
THREE_D_VARIABLES = ("x", "y", "z")
FOUR_D_VARIABLES = ("x", "y", "z", "w")


def _r2_rmse(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    target = np.asarray(target, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    mse = float(np.mean((target - prediction) ** 2))
    baseline_mse = float(np.mean((target - float(np.mean(target))) ** 2))
    return float(1.0 - mse / (baseline_mse + 1e-12)), float(np.sqrt(mse))


def _parameter_count(model: object) -> int:
    return int(sum(int(parameter.numel()) for parameter in model.net.parameters()))


def _linear_slope(frame: pd.DataFrame, column: str) -> float:
    slope, _ = np.polyfit(
        frame["beta"].to_numpy(dtype=float),
        frame[column].to_numpy(dtype=float),
        deg=1,
    )
    return float(slope)


def _bootstrap_mean_ci(
    values: np.ndarray, *, seed: int = 18201, n_boot: int = 5000
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(values, size=(int(n_boot), len(values)), replace=True)
    low, high = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def run_comparison(
    *,
    beta_values: Sequence[float] = BETA_COMMON_DRIVER_SWEEP_VALUES,
    seeds: Sequence[int] = (0, 1, 2, 3),
    n_samples: int = 1100,
    noise: float = 0.05,
    alpha: float = 1.0,
    mlp_epochs: int = 90,
    test_fraction: float = 0.20,
    show_progress: bool = False,
) -> dict[str, object]:
    if not 0.0 < float(test_fraction) < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5")

    pairs = [(float(beta), int(seed)) for beta in beta_values for seed in seeds]
    if show_progress:
        from tqdm.auto import tqdm

        pairs = tqdm(pairs, desc="3D vs 4D MLP", unit="pair", mininterval=1.0)

    rows: list[dict[str, object]] = []
    for beta, seed in pairs:
        simulation_config = SimConfig(
            mechanism="common_driver_sine_synergy",
            n_samples=int(n_samples),
            noise=float(noise),
            seed=int(seed),
            synergy_strength=float(alpha),
            common_driver_strength=float(beta),
            mlp_epochs=int(mlp_epochs),
        )
        series, _ = simulate_system(simulation_config)
        current = series.iloc[:-1].reset_index(drop=True)
        z_next = series[["z"]].iloc[1:].to_numpy(dtype=float)
        split = int(np.floor((1.0 - float(test_fraction)) * len(current)))
        if split < 32 or len(current) - split < 16:
            raise ValueError("train/test split is too small")

        predictions: dict[str, np.ndarray] = {}
        parameter_counts: dict[str, int] = {}
        for condition, variables in (
            ("3D", THREE_D_VARIABLES),
            ("4D", FOUR_D_VARIABLES),
        ):
            features = current.loc[:, variables].to_numpy(dtype=float)
            model_config = SimConfig(
                mechanism="common_driver_sine_synergy",
                n_samples=int(n_samples),
                noise=float(noise),
                seed=int(seed),
                synergy_strength=float(alpha),
                common_driver_strength=float(beta),
                mlp_epochs=int(mlp_epochs),
                variable_names=tuple(variables),
            )
            model = train_mlp_transition_model(
                features[:split], z_next[:split], model_config
            )
            predictions[condition] = model.predict(features[split:])[:, 0]
            parameter_counts[condition] = _parameter_count(model)

        target = z_next[split:, 0]
        r2_3d, rmse_3d = _r2_rmse(target, predictions["3D"])
        r2_4d, rmse_4d = _r2_rmse(target, predictions["4D"])
        rows.append(
            {
                "beta": beta,
                "seed": seed,
                "train_samples": split,
                "test_samples": len(current) - split,
                "r2_3d": r2_3d,
                "r2_4d": r2_4d,
                "delta_r2_3d_minus_4d": r2_3d - r2_4d,
                "rmse_3d": rmse_3d,
                "rmse_4d": rmse_4d,
                "delta_rmse_3d_minus_4d": rmse_3d - rmse_4d,
                "parameters_3d": parameter_counts["3D"],
                "parameters_4d": parameter_counts["4D"],
            }
        )

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("beta", as_index=False)
        .agg(
            r2_3d_mean=("r2_3d", "mean"),
            r2_3d_std=("r2_3d", "std"),
            r2_4d_mean=("r2_4d", "mean"),
            r2_4d_std=("r2_4d", "std"),
            delta_r2_mean=("delta_r2_3d_minus_4d", "mean"),
            delta_r2_std=("delta_r2_3d_minus_4d", "std"),
            rmse_3d_mean=("rmse_3d", "mean"),
            rmse_3d_std=("rmse_3d", "std"),
            rmse_4d_mean=("rmse_4d", "mean"),
            rmse_4d_std=("rmse_4d", "std"),
            delta_rmse_mean=("delta_rmse_3d_minus_4d", "mean"),
            delta_rmse_std=("delta_rmse_3d_minus_4d", "std"),
        )
        .sort_values("beta")
        .reset_index(drop=True)
    )
    delta_ci = _bootstrap_mean_ci(frame["delta_r2_3d_minus_4d"].to_numpy(dtype=float))
    per_seed_delta_slopes = {
        str(int(seed)): _linear_slope(group.sort_values("beta"), "delta_r2_3d_minus_4d")
        for seed, group in frame.groupby("seed")
    }
    statistics = {
        "r2_3d_beta_slope": _linear_slope(frame, "r2_3d"),
        "r2_4d_beta_slope": _linear_slope(frame, "r2_4d"),
        "mean_delta_r2_3d_minus_4d": float(frame["delta_r2_3d_minus_4d"].mean()),
        "std_delta_r2_3d_minus_4d": float(frame["delta_r2_3d_minus_4d"].std(ddof=1)),
        "mean_delta_r2_ci_low": delta_ci[0],
        "mean_delta_r2_ci_high": delta_ci[1],
        "three_d_better_count": int((frame["delta_r2_3d_minus_4d"] > 0.0).sum()),
        "four_d_better_count": int((frame["delta_r2_3d_minus_4d"] < 0.0).sum()),
        "n_pairs": int(len(frame)),
        "per_seed_delta_beta_slopes": per_seed_delta_slopes,
    }
    return {
        "contract": {
            "scientific_question": "What changes when only w_t is removed from the MLP input?",
            "treatment_levels": {
                "3D": list(THREE_D_VARIABLES),
                "4D": list(FOUR_D_VARIABLES),
            },
            "target": "z_next",
            "pairing_unit": "beta x seed trajectory and chronological split",
            "primary_metric": "held-out z_next R2",
            "secondary_metric": "held-out z_next RMSE",
            "limitation": "The 4D first layer has one additional input column and therefore 32 additional weights.",
        },
        "config": {
            "beta_values": [float(value) for value in beta_values],
            "seeds": [int(value) for value in seeds],
            "n_samples": int(n_samples),
            "noise": float(noise),
            "alpha": float(alpha),
            "mlp_epochs": int(mlp_epochs),
            "hidden_dim": 32,
            "batch_size": 256,
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "test_fraction": float(test_fraction),
            "split": "chronological 80/20",
        },
        "runs": rows,
        "summary": summary.to_dict("records"),
        "statistics": statistics,
    }


def plot_comparison(result: dict[str, object], figure_stem: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.frameon": False,
            "legend.fontsize": 6.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    frame = pd.DataFrame(result["summary"]).sort_values("beta")
    beta = frame["beta"].to_numpy(dtype=float)
    colors = {"3D": "#1B9E77", "4D": "#4C78A8"}
    fig, axes = plt.subplots(
        2, 1, figsize=(7.2, 4.8), sharex=True, constrained_layout=True
    )

    ax = axes[0]
    for label, mean_col, std_col, marker in (
        ("3D MLP: x, y, z", "r2_3d_mean", "r2_3d_std", "D"),
        ("4D MLP: x, y, z, w", "r2_4d_mean", "r2_4d_std", "o"),
    ):
        key = label[:2]
        mean = frame[mean_col].to_numpy(dtype=float)
        std = frame[std_col].fillna(0.0).to_numpy(dtype=float)
        ax.plot(
            beta,
            mean,
            color=colors[key],
            marker=marker,
            markevery=2,
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=1.8,
            label=label,
        )
        ax.fill_between(
            beta, mean - std, mean + std, color=colors[key], alpha=0.14, linewidth=0
        )
    ax.axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Held-out R² for z(t+1)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.text(-0.075, 1.02, "a", transform=ax.transAxes, fontsize=8, fontweight="bold")

    ax = axes[1]
    delta = frame["delta_r2_mean"].to_numpy(dtype=float)
    delta_std = frame["delta_r2_std"].fillna(0.0).to_numpy(dtype=float)
    ax.plot(
        beta,
        delta,
        color="#7C5AA6",
        marker="s",
        markevery=2,
        markersize=4.0,
        markeredgecolor="white",
        markeredgewidth=0.5,
        linewidth=1.8,
    )
    ax.fill_between(
        beta,
        delta - delta_std,
        delta + delta_std,
        color="#7C5AA6",
        alpha=0.16,
        linewidth=0,
    )
    ax.axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.9)
    ax.set_ylabel("Paired ΔR² (3D − 4D)")
    ax.set_xlabel("β: common-driver strength")
    ax.set_xlim(-0.01, 1.01)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.text(-0.075, 1.02, "b", transform=ax.transAxes, fontsize=8, fontweight="bold")

    figure_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = figure_stem.with_suffix(".png")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(figure_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(figure_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--figure-stem", type=Path, default=DEFAULT_FIGURE_STEM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = (
        {"beta_values": (0.0, 1.0), "seeds": (0,), "n_samples": 320, "mlp_epochs": 3}
        if args.smoke
        else {"show_progress": True}
    )
    result = run_comparison(**kwargs)
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    figure_path = plot_comparison(result, args.figure_stem)
    print(
        json.dumps(
            {
                "result_path": str(args.result_path),
                "figure_path": str(figure_path),
                "statistics": result["statistics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
