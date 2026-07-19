#!/usr/bin/env python3
"""Audit MLP+PEID intervention sample size for the rounded sine DGP."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
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
from scripts.run_original_neighborhood_one_decimal import _plot_slope_audit, _slopes
from yrd.transport_map import summarize_two_source_synergy_transport_map


BASE_RESULT = DEFAULT_RESULT_DIR / "sine_beta_original_neighborhood_one_decimal.json"
DEFAULT_RESULT = DEFAULT_RESULT_DIR / "sine_beta_intervention_sample_robustness.json"
DEFAULT_CHECKPOINT = DEFAULT_RESULT_DIR / "sine_beta_intervention_samples_runs.jsonl"
DEFAULT_STATUS = ROOT / "docs" / "log" / "sine_beta_intervention_samples" / "live_progress.json"
MAIN_FIGURE_STEM = "sine_beta_original_neighborhood_one_decimal_all_methods"
ROBUSTNESS_FIGURE_STEM = "sine_beta_intervention_sample_robustness"
AUDIT_FIGURE_STEM = "sine_beta_original_vs_one_decimal_slope_audit"

FULL_SAMPLE_SIZES = (320, 640, 1280, 2560, 5120)
FULL_SEEDS = (0, 1, 2, 3)
ROUNDED_DYNAMICS = {
    "w_memory": 0.8,
    "x_memory": 0.4,
    "y_memory": 0.4,
    "w_to_x": 0.8,
    "w_to_y": 0.8,
    "z_memory": 0.2,
    "w_to_z": 0.1,
}


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_status(
    path: Path,
    *,
    phase: str,
    current: int,
    total: int,
    started: float,
    metrics: Mapping[str, object] | None = None,
    message: str | None = None,
) -> None:
    elapsed = max(0.0, time.monotonic() - started)
    rate = current / elapsed if current > 0 and elapsed > 0 else 0.0
    payload: dict[str, object] = {
        "phase": phase,
        "current": int(current),
        "total": int(total),
        "unit": "model",
        "elapsed_seconds": elapsed,
        "eta_seconds": (total - current) / rate if rate > 0 else None,
        "metrics": dict(metrics or {}),
        "updated_at": time.time(),
    }
    if message:
        payload["message"] = message
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _config(beta: float, seed: int, intervention_samples: int) -> SimConfig:
    return SimConfig(
        mechanism="common_driver_sine_synergy",
        n_samples=1100,
        noise=0.05,
        seed=int(seed),
        synergy_strength=1.0,
        common_driver_strength=float(beta),
        mlp_epochs=90,
        intervention_samples=int(intervention_samples),
        driver_memory_coefficient=ROUNDED_DYNAMICS["w_memory"],
        x_memory_coefficient=ROUNDED_DYNAMICS["x_memory"],
        y_memory_coefficient=ROUNDED_DYNAMICS["y_memory"],
        driver_to_x_coefficient=ROUNDED_DYNAMICS["w_to_x"],
        driver_to_y_coefficient=ROUNDED_DYNAMICS["w_to_y"],
        target_memory_coefficient=ROUNDED_DYNAMICS["z_memory"],
        driver_to_target_coefficient=ROUNDED_DYNAMICS["w_to_z"],
    )


def _empirical_bounds(values: np.ndarray, config: SimConfig) -> tuple[float, float]:
    low = float(np.quantile(values, config.quantile_low))
    high = float(np.quantile(values, config.quantile_high))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(values)), float(np.max(values))
    return low, high


def _nested_swap_paired_interventions(
    series: pd.DataFrame,
    config: SimConfig,
    *,
    maximum_samples: int,
    source_support: tuple[float, float] = (-1.8, 1.8),
) -> pd.DataFrame:
    """Return nested interventions with exact x/y swap pairing and paired context."""

    if maximum_samples % 2:
        raise ValueError("maximum_samples must be even for swap pairing.")
    low, high = (float(value) for value in source_support)
    pair_count = maximum_samples // 2
    rng = np.random.default_rng(int(config.seed) + 1009)
    x_base = rng.uniform(low, high, size=pair_count)
    y_base = rng.uniform(low, high, size=pair_count)
    rows: dict[str, np.ndarray] = {
        "x": np.column_stack([x_base, y_base]).reshape(-1),
        "y": np.column_stack([y_base, x_base]).reshape(-1),
    }
    for name in config.variable_names:
        if name in rows:
            continue
        values = series[name].to_numpy(dtype=float)
        context_low, context_high = _empirical_bounds(values, config)
        if context_high <= context_low:
            base = np.full(pair_count, context_low, dtype=float)
        else:
            base = rng.uniform(context_low, context_high, size=pair_count)
        rows[name] = np.repeat(base, 2)
    return pd.DataFrame(rows).loc[:, config.variable_names]


def _load_checkpoint(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_checkpoint(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _run_readouts(
    *,
    beta_values: Sequence[float],
    seeds: Sequence[int],
    sample_sizes: Sequence[int],
    checkpoint_path: Path,
    status_path: Path,
    smoke: bool,
) -> list[dict[str, object]]:
    from tqdm.auto import tqdm

    sample_sizes = tuple(sorted({int(value) for value in sample_sizes}))
    maximum_samples = max(sample_sizes)
    rows = _load_checkpoint(checkpoint_path)
    completed = {
        (round(float(row["beta"]), 8), int(row["seed"]), int(row["intervention_samples"]))
        for row in rows
    }
    pairs = [(float(beta), int(seed)) for beta in beta_values for seed in seeds]
    finished_models = sum(
        all((round(beta, 8), seed, count) in completed for count in sample_sizes)
        for beta, seed in pairs
    )
    started = time.monotonic()
    _write_status(
        status_path,
        phase="training",
        current=finished_models,
        total=len(pairs),
        started=started,
    )
    bar = tqdm(pairs, desc="intervention robustness", unit="model", mininterval=1.0)
    for model_index, (beta, seed) in enumerate(bar, start=1):
        missing = [
            count
            for count in sample_sizes
            if (round(beta, 8), seed, count) not in completed
        ]
        if not missing:
            continue
        config = _config(beta, seed, maximum_samples)
        if smoke:
            config = SimConfig(**{**config.__dict__, "n_samples": 320, "mlp_epochs": 5})
        series, _ = simulate_system(config)
        features, targets = make_lagged_dataset(series, lag=config.lag)
        model = train_mlp_transition_model(features, targets, config)
        interventions = _nested_swap_paired_interventions(
            series,
            config,
            maximum_samples=maximum_samples,
        )
        predictions = model.predict(_intervention_features(interventions, config))
        z_index = config.variable_names.index("z")
        new_rows: list[dict[str, object]] = []
        for count in missing:
            estimate = summarize_two_source_synergy_transport_map(
                interventions[["x"]].to_numpy(dtype=float)[:count],
                interventions[["y"]].to_numpy(dtype=float)[:count],
                predictions[:count, [z_index]],
            )
            new_rows.append(
                {
                    "beta": beta,
                    "seed": seed,
                    "intervention_samples": count,
                    "left_ei": float(estimate["left_ei"]),
                    "right_ei": float(estimate["right_ei"]),
                    "joint_ei": float(estimate["joint_ei"]),
                    "synergy": float(estimate["syn"]),
                    "final_train_loss": float(model.loss_history[-1]),
                    "sampling": "nested_swap_paired_uniform_xy_paired_empirical_context",
                    "function_anova": False,
                    "oracle_information_used": False,
                }
            )
        _append_checkpoint(checkpoint_path, new_rows)
        rows.extend(new_rows)
        completed.update(
            (round(beta, 8), seed, int(row["intervention_samples"]))
            for row in new_rows
        )
        finished_models = sum(
            all((round(pair_beta, 8), pair_seed, count) in completed for count in sample_sizes)
            for pair_beta, pair_seed in pairs
        )
        latest = new_rows[-1]
        bar.set_postfix(
            ux=f"{latest['left_ei']:.4f}",
            uy=f"{latest['right_ei']:.4f}",
            syn=f"{latest['synergy']:.3f}",
        )
        _write_status(
            status_path,
            phase="training",
            current=finished_models,
            total=len(pairs),
            started=started,
            metrics={
                "beta": beta,
                "seed": seed,
                "intervention_samples": maximum_samples,
                "ux": latest["left_ei"],
                "uy": latest["right_ei"],
                "synergy": latest["synergy"],
            },
        )
    filtered = [
        row
        for row in rows
        if round(float(row["beta"]), 8) in {round(float(beta), 8) for beta in beta_values}
        and int(row["seed"]) in {int(seed) for seed in seeds}
        and int(row["intervention_samples"]) in set(sample_sizes)
    ]
    _write_status(
        status_path,
        phase="analysis",
        current=len(pairs),
        total=len(pairs),
        started=started,
    )
    return filtered


def _robustness_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["intervention_samples", "beta"], as_index=False)
        .agg(
            ux_mean=("left_ei", "mean"),
            ux_std=("left_ei", "std"),
            uy_mean=("right_ei", "mean"),
            uy_std=("right_ei", "std"),
            synergy_mean=("synergy", "mean"),
            synergy_std=("synergy", "std"),
            joint_ei_mean=("joint_ei", "mean"),
            joint_ei_std=("joint_ei", "std"),
        )
        .sort_values(["intervention_samples", "beta"])
    )
    return summary.where(pd.notna(summary), None).to_dict("records")


def _metric_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    output: list[dict[str, object]] = []
    for sample_size, group in frame.groupby("intervention_samples"):
        beta_mean = group.groupby("beta", as_index=False).mean(numeric_only=True).sort_values("beta")
        beta = beta_mean["beta"].to_numpy(dtype=float)
        row: dict[str, object] = {"intervention_samples": int(sample_size)}
        for label, column in (("ux", "left_ei"), ("uy", "right_ei"), ("synergy", "synergy")):
            values = beta_mean[column].to_numpy(dtype=float)
            row[f"{label}_mean"] = float(np.mean(values))
            row[f"{label}_max"] = float(np.max(values))
            row[f"{label}_absolute_tv"] = float(np.abs(np.diff(values)).sum())
            row[f"{label}_slope"] = float(np.polyfit(beta, values, 1)[0])
        output.append(row)
    return sorted(output, key=lambda row: int(row["intervention_samples"]))


def _updated_full_result(
    base_full: Mapping[str, object],
    robustness_rows: Sequence[Mapping[str, object]],
    *,
    maximum_samples: int,
) -> dict[str, object]:
    selected = {
        (round(float(row["beta"]), 8), int(row["seed"])): row
        for row in robustness_rows
        if int(row["intervention_samples"]) == maximum_samples
    }
    updated_runs: list[dict[str, object]] = []
    for base_row in base_full["runs"]:
        key = (round(float(base_row["beta"]), 8), int(float(base_row["seed"])))
        estimate = selected[key]
        row = dict(base_row)
        row.update(
            {
                "tm_peid_xy_joint_ei": float(estimate["joint_ei"]),
                "tm_peid_xy_left_ei": float(estimate["left_ei"]),
                "tm_peid_xy_right_ei": float(estimate["right_ei"]),
                "tm_peid_xy_synergy": float(estimate["synergy"]),
                "mlp_peid_redundancy": 0.0,
                "mlp_peid_unique_x": float(estimate["left_ei"]),
                "mlp_peid_unique_y": float(estimate["right_ei"]),
                "mlp_peid_xy_synergy": float(estimate["synergy"]),
                "mlp_peid_xy_joint": float(estimate["joint_ei"]),
                "mlp_peid_intervention_samples": maximum_samples,
                "mlp_peid_intervention_sampling": str(estimate["sampling"]),
            }
        )
        updated_runs.append(row)

    updated_summary = copy.deepcopy(base_full["summary"])
    run_frame = pd.DataFrame(updated_runs)
    replacements = {
        "tm_peid_xy_joint_ei": "tm_peid_xy_joint_ei",
        "tm_peid_xy_left_ei": "tm_peid_xy_left_ei",
        "tm_peid_xy_right_ei": "tm_peid_xy_right_ei",
        "tm_peid_xy_synergy": "tm_peid_xy_synergy",
        "mlp_peid_redundancy": "mlp_peid_redundancy",
        "mlp_peid_unique_x": "mlp_peid_unique_x",
        "mlp_peid_unique_y": "mlp_peid_unique_y",
        "mlp_peid_xy_synergy": "mlp_peid_xy_synergy",
        "mlp_peid_xy_joint": "mlp_peid_xy_joint",
    }
    aggregate = run_frame.groupby("beta", as_index=True)
    for summary_row in updated_summary:
        beta = float(summary_row["beta"])
        beta_runs = aggregate.get_group(beta)
        for prefix, column in replacements.items():
            values = beta_runs[column].to_numpy(dtype=float)
            summary_row[f"{prefix}_mean"] = float(np.mean(values))
            summary_row[f"{prefix}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    config = copy.deepcopy(base_full.get("config", {}))
    config["intervention_samples"] = maximum_samples
    config["intervention_sampling"] = "nested_swap_paired_uniform_xy_paired_empirical_context"
    config["function_anova"] = False
    config["oracle_information_used_for_mlp_peid"] = False
    return {
        **copy.deepcopy(dict(base_full)),
        "config": config,
        "runs": updated_runs,
        "summary": updated_summary,
    }


def _plot_robustness(
    robustness_summary: Sequence[Mapping[str, object]],
    metric_summary: Sequence[Mapping[str, object]],
    *,
    stem: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    frame = pd.DataFrame(robustness_summary)
    metrics = pd.DataFrame(metric_summary).sort_values("intervention_samples")
    sizes = sorted(frame["intervention_samples"].astype(int).unique())
    colors = plt.cm.viridis(np.linspace(0.12, 0.9, len(sizes)))
    color_by_size = dict(zip(sizes, colors))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    for ax, column, ylabel, panel in [
        (axes[0, 0], "ux_mean", r"$U_x$ (bits)", "a"),
        (axes[0, 1], "uy_mean", r"$U_y$ (bits)", "b"),
        (axes[1, 0], "synergy_mean", "Synergy (bits)", "c"),
    ]:
        for size in sizes:
            subset = frame[frame["intervention_samples"].astype(int) == size].sort_values("beta")
            ax.plot(
                subset["beta"],
                subset[column],
                color=color_by_size[size],
                linewidth=1.5 if size == max(sizes) else 0.9,
                marker="o" if size == max(sizes) else None,
                markersize=2.5,
                markevery=4,
                label=f"n={size}",
                zorder=3 if size == max(sizes) else 2,
            )
        ax.set_xlabel(r"Common-driver strength $\beta$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.01, 1.01)
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.text(-0.12, 1.03, panel, transform=ax.transAxes, fontweight="bold", fontsize=8)
    axes[0, 0].axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.8)
    axes[0, 1].axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.8)

    ax = axes[1, 1]
    x = metrics["intervention_samples"].to_numpy(dtype=float)
    for column, label, color, marker, linestyle in [
        ("ux_mean", r"mean $U_x$", "#009E73", "^", "-"),
        ("uy_mean", r"mean $U_y$", "#7E57C2", "v", "-"),
        ("ux_absolute_tv", r"TV $U_x$", "#009E73", "^", "--"),
        ("uy_absolute_tv", r"TV $U_y$", "#7E57C2", "v", "--"),
    ]:
        ax.plot(x, metrics[column], label=label, color=color, marker=marker, linestyle=linestyle, linewidth=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [str(int(value)) for value in x])
    ax.set_xlabel("Intervention samples")
    ax.set_ylabel("Offset / absolute TV (bits)")
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.text(-0.12, 1.03, "d", transform=ax.transAxes, fontweight="bold", fontsize=8)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=len(sizes))
    DEFAULT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = DEFAULT_FIGURE_DIR / stem
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return output.with_suffix(".png")


def run(*, smoke: bool = False) -> dict[str, object]:
    base = _read_json(BASE_RESULT)
    beta_values = (0.0, 1.0) if smoke else BETA_COMMON_DRIVER_SWEEP_VALUES
    seeds = (0,) if smoke else FULL_SEEDS
    sample_sizes = (64, 128) if smoke else FULL_SAMPLE_SIZES
    suffix = "_smoke" if smoke else ""
    checkpoint = DEFAULT_CHECKPOINT.with_name(DEFAULT_CHECKPOINT.stem + suffix + DEFAULT_CHECKPOINT.suffix)
    status = DEFAULT_STATUS.with_name(DEFAULT_STATUS.stem + suffix + DEFAULT_STATUS.suffix)
    started = time.monotonic()
    try:
        rows = _run_readouts(
            beta_values=beta_values,
            seeds=seeds,
            sample_sizes=sample_sizes,
            checkpoint_path=checkpoint,
            status_path=status,
            smoke=smoke,
        )
        robustness_summary = _robustness_summary(rows)
        metric_summary = _metric_summary(rows)
        if smoke:
            payload = {
                "mode": "smoke",
                "function_anova": False,
                "oracle_information_used": False,
                "runs": rows,
                "summary": robustness_summary,
                "metrics": metric_summary,
            }
            _write_json(DEFAULT_RESULT.with_name(DEFAULT_RESULT.stem + suffix + ".json"), payload)
            _write_status(status, phase="complete", current=2, total=2, started=started)
            return payload

        maximum_samples = max(sample_sizes)
        updated_full = _updated_full_result(
            base["full_result"],
            rows,
            maximum_samples=maximum_samples,
        )
        main_figure = _plot_sine_beta_combined_readout_sweep(
            updated_full,
            DEFAULT_FIGURE_DIR,
            liang_result=base.get("liang_result"),
            stem=MAIN_FIGURE_STEM,
            include_oracle=False,
        )
        robustness_figure = _plot_robustness(
            robustness_summary,
            metric_summary,
            stem=ROBUSTNESS_FIGURE_STEM,
        )
        rounded_slopes = _slopes(updated_full["runs"])
        audit_figure = _plot_slope_audit(
            base["original_full_slopes"],
            rounded_slopes,
            stem=AUDIT_FIGURE_STEM,
        )
        payload = {
            "scientific_question": "What changes when only MLP+PEID intervention sample count changes?",
            "function_anova": False,
            "oracle_information_used": False,
            "sample_sizes": list(sample_sizes),
            "maximum_samples_for_main_figure": maximum_samples,
            "sampling": "nested_swap_paired_uniform_xy_paired_empirical_context",
            "controlled_variables": {
                "dynamics": ROUNDED_DYNAMICS,
                "beta_values": list(beta_values),
                "seeds": list(seeds),
                "n_samples": 1100,
                "mlp_epochs": 90,
                "source_support": [-1.8, 1.8],
            },
            "runs": rows,
            "robustness_summary": robustness_summary,
            "metric_summary": metric_summary,
            "updated_full_result": updated_full,
            "rounded_full_slopes": rounded_slopes,
            "main_figure": str(main_figure),
            "robustness_figure": str(robustness_figure),
            "audit_figure": str(audit_figure),
        }
        _write_json(DEFAULT_RESULT, payload)
        _write_status(
            status,
            phase="complete",
            current=len(beta_values) * len(seeds),
            total=len(beta_values) * len(seeds),
            started=started,
            metrics={"maximum_samples": maximum_samples},
        )
        return payload
    except Exception as error:
        _write_status(
            status,
            phase="failed",
            current=0,
            total=len(beta_values) * len(seeds),
            started=started,
            message=str(error),
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(smoke=args.smoke)
    print(
        json.dumps(
            {
                "mode": result.get("mode", "full"),
                "sample_sizes": result.get("sample_sizes"),
                "main_figure": result.get("main_figure"),
                "robustness_figure": result.get("robustness_figure"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
