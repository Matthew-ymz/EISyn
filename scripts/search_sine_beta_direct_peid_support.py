#!/usr/bin/env python3
"""Sweep symmetric x/y intervention supports for direct MLP+PEID without ANOVA."""

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
    _sample_intervention_sources,
    make_lagged_dataset,
    simulate_system,
    train_mlp_transition_model,
)


DEFAULT_HALF_WIDTHS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.8, 2.0)
DEFAULT_CHECKPOINT = DEFAULT_RESULT_DIR / "sine_beta_direct_support_sweep_runs.jsonl"
DEFAULT_RESULT = DEFAULT_RESULT_DIR / "sine_beta_direct_support_sweep.json"
DEFAULT_BASE_RESULT = DEFAULT_RESULT_DIR / "sine_beta_one_decimal_all_methods.json"
DEFAULT_FIGURE_STEM = "sine_beta_direct_support_sweep"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - residual / total) if total > 1e-12 else float("nan")


def _absolute_tv(values: Sequence[float]) -> float:
    return float(np.abs(np.diff(np.asarray(values, dtype=float))).sum())


def _curve_metrics(frame: pd.DataFrame, column: str) -> dict[str, float]:
    summary = frame.groupby("beta", as_index=False)[column].mean().sort_values("beta")
    values = summary[column].to_numpy(dtype=float)
    mean_abs = float(np.mean(np.abs(values)))
    return {
        "absolute_tv": _absolute_tv(values),
        "mean_abs": mean_abs,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "range_over_mean": float((np.max(values) - np.min(values)) / mean_abs)
        if mean_abs > 1e-12
        else float("inf"),
    }


def _baseline_metrics(
    baseline_runs: Sequence[Mapping[str, object]],
    *,
    seeds: Sequence[int],
) -> dict[str, dict[str, float]]:
    frame = pd.DataFrame(baseline_runs)
    frame = frame[frame["seed"].astype(int).isin([int(seed) for seed in seeds])]
    return {
        "MMI-PID": _curve_metrics(frame, "mmi_pid_xy_synergy"),
        "SURD": _curve_metrics(frame, "surd_xy_synergy"),
        "Observational WMS": _curve_metrics(frame, "observational_wms"),
    }


def run_sweep(
    *,
    beta_values: Sequence[float],
    seeds: Sequence[int],
    half_widths: Sequence[float],
    n_samples: int,
    mlp_epochs: int,
    intervention_samples: int,
    checkpoint_path: Path,
    show_progress: bool,
) -> dict[str, object]:
    from yrd.transport_map import summarize_two_source_synergy_transport_map

    rows: list[dict[str, float | str]] = []
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    completed = {
        (round(float(row["beta"]), 8), int(float(row["seed"])), round(float(row["half_width"]), 8))
        for row in rows
    }
    pairs = [(float(beta), int(seed)) for beta in beta_values for seed in seeds]
    if show_progress:
        from tqdm.auto import tqdm

        pairs = tqdm(pairs, desc="direct PEID support sweep", unit="model", mininterval=1.0)

    for beta, seed in pairs:
        missing = [
            float(width)
            for width in half_widths
            if (round(beta, 8), seed, round(float(width), 8)) not in completed
        ]
        if not missing:
            continue
        config = SimConfig(
            mechanism="common_driver_sine_synergy",
            n_samples=int(n_samples),
            noise=0.1,
            seed=seed,
            synergy_strength=1.0,
            common_driver_strength=beta,
            mlp_epochs=int(mlp_epochs),
            intervention_samples=int(intervention_samples),
            bins=4,
        )
        series, _ = simulate_system(config)
        features, targets = make_lagged_dataset(series, lag=config.lag)
        model = train_mlp_transition_model(features, targets, config)
        target_index = config.variable_names.index("z")
        train_prediction = model.predict(features)[:, target_index]
        train_r2 = _r2_score(targets[:, target_index], train_prediction)

        context = _sample_intervention_sources(series, config)
        unit_rng = np.random.default_rng(seed + 1009)
        unit_x = unit_rng.uniform(-1.0, 1.0, size=int(intervention_samples))
        unit_y = unit_rng.uniform(-1.0, 1.0, size=int(intervention_samples))
        x_low, x_high = np.quantile(series["x"].to_numpy(dtype=float), [0.01, 0.99])
        y_low, y_high = np.quantile(series["y"].to_numpy(dtype=float), [0.01, 0.99])

        for width in missing:
            states = context.copy()
            states["x"] = width * unit_x
            states["y"] = width * unit_y
            prediction = model.predict(_intervention_features(states, config))[:, [target_index]]
            peid = summarize_two_source_synergy_transport_map(
                states[["x"]].to_numpy(dtype=float),
                states[["y"]].to_numpy(dtype=float),
                prediction,
            )
            extrapolated = (
                (states["x"].to_numpy(dtype=float) < x_low)
                | (states["x"].to_numpy(dtype=float) > x_high)
                | (states["y"].to_numpy(dtype=float) < y_low)
                | (states["y"].to_numpy(dtype=float) > y_high)
            )
            row: dict[str, float | str] = {
                "run_id": f"beta={beta:.2f}|seed={seed}|half_width={width:g}",
                "beta": beta,
                "seed": float(seed),
                "half_width": width,
                "mlp_peid_unique_x": float(peid["left_ei"]),
                "mlp_peid_unique_y": float(peid["right_ei"]),
                "mlp_peid_xy_synergy": float(peid["syn"]),
                "mlp_peid_xy_joint": float(peid["joint_ei"]),
                "z_train_r2": train_r2,
                "extrapolation_fraction": float(np.mean(extrapolated)),
            }
            rows.append(row)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    frame = pd.DataFrame(rows)
    requested = frame[
        frame["beta"].round(8).isin([round(float(value), 8) for value in beta_values])
        & frame["seed"].astype(int).isin([int(seed) for seed in seeds])
        & frame["half_width"].round(8).isin([round(float(value), 8) for value in half_widths])
    ].copy()
    summary = (
        requested.groupby(["half_width", "beta"], as_index=False)
        .agg(
            mlp_peid_unique_x_mean=("mlp_peid_unique_x", "mean"),
            mlp_peid_unique_x_std=("mlp_peid_unique_x", "std"),
            mlp_peid_unique_y_mean=("mlp_peid_unique_y", "mean"),
            mlp_peid_unique_y_std=("mlp_peid_unique_y", "std"),
            mlp_peid_xy_synergy_mean=("mlp_peid_xy_synergy", "mean"),
            mlp_peid_xy_synergy_std=("mlp_peid_xy_synergy", "std"),
            z_train_r2_mean=("z_train_r2", "mean"),
            extrapolation_fraction_mean=("extrapolation_fraction", "mean"),
        )
        .sort_values(["half_width", "beta"])
    )
    return {
        "config": {
            "function_anova": False,
            "beta_values": [float(value) for value in beta_values],
            "seeds": [int(seed) for seed in seeds],
            "half_widths": [float(value) for value in half_widths],
            "source_support": "x,y uniform on [-half_width, half_width]",
            "context_support": "w,z random samples from each beta-specific empirical support",
            "paired": "same trained model, w/z samples, and standardized x/y random draws across supports",
            "n_samples": int(n_samples),
            "mlp_epochs": int(mlp_epochs),
            "intervention_samples": int(intervention_samples),
        },
        "runs": requested.to_dict("records"),
        "summary": summary.to_dict("records"),
    }


def add_sensitivity(result: dict[str, object], baseline: Mapping[str, object]) -> None:
    frame = pd.DataFrame(result["runs"])
    sensitivity: dict[str, object] = {"selection_seeds": {}, "validation_seeds": {}}
    for split_name, seeds in (("selection_seeds", (0, 1)), ("validation_seeds", (2, 3))):
        split = frame[frame["seed"].astype(int).isin(seeds)]
        support_rows: dict[str, object] = {}
        for width, support_frame in split.groupby("half_width"):
            support_rows[f"{float(width):g}"] = {
                "synergy": _curve_metrics(support_frame, "mlp_peid_xy_synergy"),
                "unique_x": _curve_metrics(support_frame, "mlp_peid_unique_x"),
                "unique_y": _curve_metrics(support_frame, "mlp_peid_unique_y"),
                "mean_extrapolation_fraction": float(support_frame["extrapolation_fraction"].mean()),
                "mean_z_train_r2": float(support_frame["z_train_r2"].mean()),
            }
        sensitivity[split_name] = {
            "supports": support_rows,
            "baselines": _baseline_metrics(baseline["runs"], seeds=seeds),
        }
    selection = sensitivity["selection_seeds"]["supports"]
    selected = min(selection, key=lambda key: selection[key]["synergy"]["absolute_tv"])
    sensitivity["selection_rule"] = "minimum synergy absolute TV on seeds 0,1"
    sensitivity["selected_half_width"] = float(selected)
    broad_reference = selection["1.8"]["synergy"]["mean_abs"]
    signal_floor = 0.5 * float(broad_reference)
    eligible = {
        key: value
        for key, value in selection.items()
        if float(value["synergy"]["mean_abs"]) >= signal_floor
    }
    signal_preserving = min(
        eligible, key=lambda key: eligible[key]["synergy"]["absolute_tv"]
    )
    validation = sensitivity["validation_seeds"]
    validation_tv = validation["supports"][signal_preserving]["synergy"]["absolute_tv"]
    best_validation_baseline_tv = min(
        value["absolute_tv"] for value in validation["baselines"].values()
    )
    sensitivity["signal_preservation_audit"] = {
        "definition": "selection-seed mean synergy must be at least 50% of the a=1.8 reference",
        "mean_synergy_floor": signal_floor,
        "eligible_half_widths": [float(key) for key in eligible],
        "selected_half_width": float(signal_preserving),
        "validation_synergy_absolute_tv": float(validation_tv),
        "best_validation_baseline_absolute_tv": float(best_validation_baseline_tv),
        "passes_validation": bool(validation_tv < best_validation_baseline_tv),
        "adopted_for_main_comparison": False,
    }
    result["sensitivity"] = sensitivity


def plot_sweep(result: Mapping[str, object], output_stem: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    sensitivity = result["sensitivity"]
    widths = np.asarray(result["config"]["half_widths"], dtype=float)
    mpl.rcParams.update(
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
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    colors = {"selection_seeds": "#8F8F8F", "validation_seeds": "#009E73"}
    labels = {"selection_seeds": "selection seeds 0–1", "validation_seeds": "validation seeds 2–3"}
    for split_name in ("selection_seeds", "validation_seeds"):
        supports = sensitivity[split_name]["supports"]
        y = [supports[f"{width:g}"]["synergy"]["absolute_tv"] for width in widths]
        axes[0].plot(widths, y, marker="o", color=colors[split_name], label=labels[split_name])
    validation_baselines = sensitivity["validation_seeds"]["baselines"]
    for method, style in (("SURD", "--"), ("MMI-PID", ":")):
        axes[0].axhline(
            validation_baselines[method]["absolute_tv"],
            color="#6B7280",
            linestyle=style,
            linewidth=1.0,
            label=f"{method}, validation",
        )
    axes[0].set_ylabel("Synergy absolute TV (bits)")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)

    validation = sensitivity["validation_seeds"]["supports"]
    for component, color, marker in (("unique_x", "#4E79A7", "s"), ("unique_y", "#B07AA1", "^")):
        y = [validation[f"{width:g}"][component]["absolute_tv"] for width in widths]
        axes[1].plot(widths, y, color=color, marker=marker, label=component.replace("unique_", r"$U_") + "$")
    axes[1].set_ylabel("Unique-information absolute TV (bits)")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)

    extrapolation = [validation[f"{width:g}"]["mean_extrapolation_fraction"] for width in widths]
    axes[2].plot(widths, extrapolation, color="#E69F00", marker="o")
    axes[2].set_ylabel("Intervention extrapolation fraction")
    axes[2].set_ylim(bottom=0.0)

    for ax in axes:
        ax.set_xlabel(r"Support half-width $a$")
        ax.set_xticks(widths)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
        ax.set_axisbelow(True)
    DEFAULT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = DEFAULT_FIGURE_DIR / output_stem
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return output.with_suffix(".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        beta_values = (0.0, 1.0)
        seeds = (0,)
        half_widths = (0.75, 1.0, 1.8)
        n_samples, mlp_epochs, intervention_samples = 320, 5, 64
        suffix = "_smoke"
        checkpoint = args.checkpoint.with_name(args.checkpoint.stem + suffix + args.checkpoint.suffix)
        result_path = args.result.with_name(args.result.stem + suffix + args.result.suffix)
    else:
        beta_values = BETA_COMMON_DRIVER_SWEEP_VALUES
        seeds = (0, 1, 2, 3)
        half_widths = DEFAULT_HALF_WIDTHS
        n_samples, mlp_epochs, intervention_samples = 1100, 90, 640
        suffix = ""
        checkpoint = args.checkpoint
        result_path = args.result
    baseline = _read_json(DEFAULT_BASE_RESULT)
    result = run_sweep(
        beta_values=beta_values,
        seeds=seeds,
        half_widths=half_widths,
        n_samples=n_samples,
        mlp_epochs=mlp_epochs,
        intervention_samples=intervention_samples,
        checkpoint_path=checkpoint,
        show_progress=True,
    )
    if not args.smoke:
        add_sensitivity(result, baseline)
        figure = plot_sweep(result, DEFAULT_FIGURE_STEM)
    else:
        figure = None
    _write_json(result_path, result)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "checkpoint": str(checkpoint),
                "figure": str(figure) if figure else None,
                "selected_half_width": result.get("sensitivity", {}).get("selected_half_width"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
