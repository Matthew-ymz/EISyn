from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_unicm_all_mode_target_phi_eid import compute_phi_eid_for_target
from scripts.unicm_peid_syn_analysis import (
    MODE_NAMES,
    PREDICTION_LENGTH,
    create_ei_estimator,
    load_unicm_model,
    predict_modeformer_all_modes_from_history,
    resolve_checkpoint_paths,
    sample_full_history_mode_inputs,
)


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "unicm_start_month_sweep"
DEFAULT_FIG_ROOT = ROOT / "fig" / "unicm_start_month_sweep"


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class ProgressTracker:
    def __init__(self, path: Path, total_work: int) -> None:
        self.path = path
        self.total_work = int(total_work)
        self.completed_work = 0
        self.started = time.monotonic()

    def write(
        self,
        *,
        phase: str,
        message: str,
        metrics: dict[str, object] | None = None,
        completed_increment: int = 0,
        condition_current: int | None = None,
        condition_total: int | None = None,
    ) -> None:
        self.completed_work += int(completed_increment)
        elapsed = time.monotonic() - self.started
        rate = self.completed_work / elapsed if elapsed > 0 and self.completed_work > 0 else 0.0
        eta = (self.total_work - self.completed_work) / rate if rate > 0 else None
        payload: dict[str, object] = {
            "phase": phase,
            "message": message,
            "current": int(self.completed_work),
            "total": int(self.total_work),
            "unit": "sample-condition",
            "elapsed_seconds": float(elapsed),
            "eta_seconds": float(eta) if eta is not None else None,
            "pid": int(os.getpid()),
            "updated_at": time.time(),
            "metrics": metrics or {},
        }
        if condition_current is not None:
            payload["condition_current"] = int(condition_current)
        if condition_total is not None:
            payload["condition_total"] = int(condition_total)
        atomic_write_json(self.path, payload)


def cache_path(cache_dir: Path, *, seed: int, start_month: int, n_samples: int, device: str) -> Path:
    return cache_dir / (
        f"overall_pred_seed{seed}_samples{n_samples}_sampling20260619_bound4_"
        f"fullhist12_start{start_month}_{device}.npz"
    )


def cache_metadata(
    *, seed: int, start_month: int, n_samples: int, device: str, batch_size: int
) -> dict[str, object]:
    return {
        "seed": int(seed),
        "start_month": int(start_month),
        "n_samples": int(n_samples),
        "sampling_seed": 20260619,
        "intervention_bound": 4.0,
        "sampling_mode": "full_history_max_entropy",
        "history_shape": [12, len(MODE_NAMES)],
        "device": str(device),
        "inference_batch_size": int(batch_size),
    }


def save_prediction_cache(path: Path, targets: np.ndarray, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        all_mode_targets=np.asarray(targets, dtype=np.float32),
        metadata=json.dumps(metadata, sort_keys=True),
    )
    os.replace(temporary, path)


def load_prediction_cache(path: Path, expected: dict[str, object]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "all_mode_targets" not in payload or "metadata" not in payload:
            raise ValueError(f"Incomplete prediction cache: {path}")
        targets = payload["all_mode_targets"].astype(float)
        raw_metadata = payload["metadata"].item()
    metadata = json.loads(str(raw_metadata))
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Cache metadata mismatch for {key}: {metadata.get(key)!r} != {value!r}")
    expected_shape = (int(expected["n_samples"]), PREDICTION_LENGTH, len(MODE_NAMES))
    if tuple(targets.shape) != expected_shape:
        raise ValueError(f"Cache shape {targets.shape} != {expected_shape}: {path}")
    if not np.isfinite(targets).all():
        raise ValueError(f"Non-finite predictions in {path}")
    return targets


def condition_metric_path(metrics_dir: Path, *, seed: int, start_month: int) -> Path:
    return metrics_dir / f"start{start_month:02d}_seed{seed}.csv"


def compute_condition_rows(
    history_modes: np.ndarray,
    predictions: np.ndarray,
    *,
    seed: int,
    start_month: int,
    n_samples: int,
    batch_size: int,
    estimator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lead in range(1, PREDICTION_LENGTH + 1):
        target = predictions[:, lead - 1, :]
        values = compute_phi_eid_for_target(history_modes, target, estimator=estimator)
        calendar_month = (int(start_month) + int(lead) - 1) % 12
        rows.append(
            {
                "seed": int(seed),
                "start_month": int(start_month),
                "n_samples": int(n_samples),
                "inference_batch_size": int(batch_size),
                "start_month_name": MONTHS[int(start_month)],
                "lead": int(lead),
                "forecast_year": 1 + (int(lead) - 1) // 12,
                "target_calendar_month": int(calendar_month),
                "target_calendar_month_name": MONTHS[int(calendar_month)],
                "whole_ei": float(values["whole_ei"]),
                "singleton_ei_sum": float(values["singleton_ei_sum"]),
                "raw_phi_eid": float(values["raw_phi_eid"]),
                "phi_eid": float(values["phi_eid"]),
            }
        )
    return pd.DataFrame(rows)


def condition_metrics_are_valid(
    rows: pd.DataFrame,
    *,
    seed: int,
    start_month: int,
    n_samples: int,
    batch_size: int,
) -> bool:
    required = {
        "seed",
        "start_month",
        "lead",
        "n_samples",
        "inference_batch_size",
        "phi_eid",
    }
    if not required.issubset(rows.columns) or len(rows) != PREDICTION_LENGTH:
        return False
    expected = {
        "seed": int(seed),
        "start_month": int(start_month),
        "n_samples": int(n_samples),
        "inference_batch_size": int(batch_size),
    }
    if any(rows[key].nunique() != 1 or int(rows[key].iloc[0]) != value for key, value in expected.items()):
        return False
    if sorted(rows["lead"].astype(int).tolist()) != list(range(1, PREDICTION_LENGTH + 1)):
        return False
    return bool(np.isfinite(rows.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all())


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [base.with_suffix(".png"), base.with_suffix(".svg"), base.with_suffix(".pdf")]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], bbox_inches="tight")
    plt.close(fig)
    return outputs


def shared_color_limits(rows: pd.DataFrame) -> tuple[float, float]:
    means = rows.groupby(["start_month", "lead"], as_index=False)["phi_eid"].mean()["phi_eid"]
    vmin = float(means.min())
    vmax = float(means.max())
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return vmin, vmax


def plot_lead_coordinates(rows: pd.DataFrame, base: Path) -> list[Path]:
    configure_matplotlib()
    vmin, vmax = shared_color_limits(rows)
    heat = rows.pivot_table(index="start_month", columns="lead", values="phi_eid", aggfunc="mean").reindex(
        index=range(12), columns=range(1, 25)
    )
    by_start = rows.groupby(["start_month", "lead"], as_index=False)["phi_eid"].mean()
    lead_stats = rows.groupby("lead", as_index=False).agg(mean=("phi_eid", "mean"), std=("phi_eid", "std"))

    fig, (ax_heat, ax_line) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.25),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    image = ax_heat.imshow(heat.to_numpy(), aspect="auto", origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
    ax_heat.set_xlabel("Forecast lead (months)")
    ax_heat.set_ylabel("Start month")
    ax_heat.set_xticks([0, 3, 7, 11, 15, 19, 23], [1, 4, 8, 12, 16, 20, 24])
    ax_heat.set_yticks(range(12), MONTHS)
    ax_heat.text(-0.12, 1.04, "a", transform=ax_heat.transAxes, fontweight="bold", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax_heat, location="right", fraction=0.046, pad=0.03)
    colorbar.set_label("Integrated increment (bits)")

    cmap = mpl.colormaps["twilight_shifted"]
    for start_month in range(12):
        values = by_start[by_start["start_month"] == start_month]
        ax_line.plot(
            values["lead"],
            values["phi_eid"],
            color=cmap(start_month / 12.0),
            linewidth=0.75,
            alpha=0.65,
        )
    x = lead_stats["lead"].to_numpy(dtype=float)
    mean = lead_stats["mean"].to_numpy(dtype=float)
    std = lead_stats["std"].fillna(0.0).to_numpy(dtype=float)
    ax_line.fill_between(x, mean - std, mean + std, color="#4C78A8", alpha=0.16, linewidth=0)
    ax_line.plot(x, mean, color="#2F5F8F", linewidth=1.8, label="Mean ± SD")
    ax_line.axhline(0.0, color="#888888", linestyle=":", linewidth=0.7)
    ax_line.set_xlabel("Forecast lead (months)")
    ax_line.set_ylabel("Integrated increment (bits)")
    ax_line.set_xlim(1, 24)
    ax_line.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax_line.text(-0.16, 1.04, "b", transform=ax_line.transAxes, fontweight="bold", fontsize=8)
    return save_figure(fig, base)


def calendar_heat(rows: pd.DataFrame, forecast_year: int) -> pd.DataFrame:
    subset = rows[rows["forecast_year"] == int(forecast_year)]
    return subset.pivot_table(
        index="start_month", columns="target_calendar_month", values="phi_eid", aggfunc="mean"
    ).reindex(index=range(12), columns=range(12))


def plot_calendar_coordinates(rows: pd.DataFrame, base: Path) -> list[Path]:
    configure_matplotlib()
    vmin, vmax = shared_color_limits(rows)
    heat1 = calendar_heat(rows, 1)
    heat2 = calendar_heat(rows, 2)
    calendar_stats = rows.groupby("target_calendar_month", as_index=False).agg(
        mean=("phi_eid", "mean"), std=("phi_eid", "std")
    ).sort_values("target_calendar_month")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 3.15),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.05]},
    )
    for panel, (ax, heat, label, year) in enumerate(
        [(axes[0], heat1, "a", 1), (axes[1], heat2, "b", 2)]
    ):
        image = ax.imshow(heat.to_numpy(), aspect="auto", origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Target calendar month")
        ax.set_xticks(range(12), MONTHS, rotation=45, ha="right")
        ax.set_yticks(range(12), MONTHS if panel == 0 else [])
        if panel == 0:
            ax.set_ylabel("Start month")
        ax.text(0.03, 0.96, f"Year {year}", transform=ax.transAxes, va="top", color="white", fontsize=6.5)
        ax.text(-0.16, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=8)
    colorbar = fig.colorbar(image, ax=axes[:2], location="right", fraction=0.035, pad=0.025)
    colorbar.set_label("Integrated increment (bits)")

    x = calendar_stats["target_calendar_month"].to_numpy(dtype=float)
    mean = calendar_stats["mean"].to_numpy(dtype=float)
    std = calendar_stats["std"].fillna(0.0).to_numpy(dtype=float)
    axes[2].fill_between(x, mean - std, mean + std, color="#E3A857", alpha=0.18, linewidth=0)
    axes[2].plot(x, mean, color="#B97823", marker="o", markersize=2.5, linewidth=1.6, label="Mean ± SD")
    axes[2].axhline(0.0, color="#888888", linestyle=":", linewidth=0.7)
    axes[2].set_xlabel("Target calendar month")
    axes[2].set_ylabel("Integrated increment (bits)")
    axes[2].set_xticks(range(12), MONTHS, rotation=45, ha="right")
    axes[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    axes[2].text(-0.18, 1.04, "c", transform=axes[2].transAxes, fontweight="bold", fontsize=8)
    return save_figure(fig, base)


def normalized_concentration(values: Iterable[int], support_size: int) -> float:
    counts = pd.Series(list(values), dtype=int).value_counts().to_numpy(dtype=float)
    probabilities = counts / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(1.0 - entropy / np.log(float(support_size)))


def variance_decomposition(rows: pd.DataFrame) -> dict[str, float]:
    values = rows["phi_eid"].to_numpy(dtype=float)
    grand = float(values.mean())
    ss_total = float(np.sum((values - grand) ** 2))

    def effect_ss(column: str) -> float:
        means = rows.groupby(column)["phi_eid"].mean()
        counts = rows.groupby(column)["phi_eid"].size()
        return float(np.sum(counts.to_numpy(dtype=float) * (means.to_numpy(dtype=float) - grand) ** 2))

    ss_lead = effect_ss("lead")
    ss_calendar = effect_ss("target_calendar_month")
    ss_seed = effect_ss("seed")
    ss_residual = max(0.0, ss_total - ss_lead - ss_calendar - ss_seed)
    denominator = ss_total if ss_total > 0 else 1.0
    return {
        "grand_mean_bits": grand,
        "ss_total": ss_total,
        "ss_lead": ss_lead,
        "ss_calendar_month": ss_calendar,
        "ss_checkpoint": ss_seed,
        "ss_residual_and_interactions": ss_residual,
        "eta2_lead": ss_lead / denominator,
        "eta2_calendar_month": ss_calendar / denominator,
        "eta2_checkpoint": ss_seed / denominator,
        "eta2_residual_and_interactions": ss_residual / denominator,
    }


def summarize_phase(
    rows: pd.DataFrame,
    *,
    phase_name: str,
    n_samples: int,
    output_dir: Path,
    figure_dir: Path,
    nonnegative_tolerance: float,
) -> dict[str, object]:
    rows = rows.sort_values(["seed", "start_month", "lead"]).reset_index(drop=True)
    rows_path = output_dir / "start_month_phi_eid_rows.csv"
    rows.to_csv(rows_path, index=False)

    negative_within = rows[(rows["phi_eid"] < 0) & (rows["phi_eid"] >= -float(nonnegative_tolerance))]
    violations = rows[rows["phi_eid"] < -float(nonnegative_tolerance)]
    if not violations.empty:
        raise RuntimeError(
            "Significant Xi nonnegativity violation: "
            f"minimum={rows['phi_eid'].min():.8g} bits, tolerance={nonnegative_tolerance}, "
            f"count={len(violations)}"
        )

    lead_profile = rows.groupby("lead", as_index=False).agg(
        phi_eid_mean=("phi_eid", "mean"),
        phi_eid_std=("phi_eid", "std"),
        phi_eid_min=("phi_eid", "min"),
        phi_eid_max=("phi_eid", "max"),
    )
    lead_profile.to_csv(output_dir / "lead_profile.csv", index=False)
    calendar_profile = rows.groupby("target_calendar_month", as_index=False).agg(
        phi_eid_mean=("phi_eid", "mean"),
        phi_eid_std=("phi_eid", "std"),
        phi_eid_min=("phi_eid", "min"),
        phi_eid_max=("phi_eid", "max"),
    )
    calendar_profile["target_calendar_month_name"] = calendar_profile["target_calendar_month"].map(
        dict(enumerate(MONTHS))
    )
    calendar_profile.to_csv(output_dir / "calendar_profile.csv", index=False)

    peak_indices = rows.groupby(["seed", "start_month"])["phi_eid"].idxmax()
    peaks = rows.loc[peak_indices, ["seed", "start_month", "lead", "target_calendar_month", "phi_eid"]].copy()
    peaks = peaks.rename(columns={"lead": "peak_lead", "target_calendar_month": "peak_calendar_month", "phi_eid": "peak_phi_eid"})
    peaks["peak_calendar_month_name"] = peaks["peak_calendar_month"].map(dict(enumerate(MONTHS)))
    peaks.to_csv(output_dir / "peak_summary.csv", index=False)

    variance = variance_decomposition(rows)
    atomic_write_json(output_dir / "variance_decomposition.json", variance)
    lead_mode = int(peaks["peak_lead"].mode().iloc[0])
    calendar_mode = int(peaks["peak_calendar_month"].mode().iloc[0])
    lead_window_fraction = float(peaks["peak_lead"].between(7, 10).mean())
    calendar_window_fraction = float(peaks["peak_calendar_month"].isin([6, 7, 8, 9]).mean())
    lead_concentration = normalized_concentration(peaks["peak_lead"], 24)
    calendar_concentration = normalized_concentration(peaks["peak_calendar_month"], 12)

    if variance["eta2_lead"] > 1.5 * variance["eta2_calendar_month"]:
        classification = "forecast-lead dominant"
    elif variance["eta2_calendar_month"] > 1.5 * variance["eta2_lead"]:
        classification = "calendar-month dominant"
    else:
        classification = "mixed lead and calendar-month organization"

    figures = []
    figures.extend(plot_lead_coordinates(rows, figure_dir / f"unicm_start_month_{phase_name}_lead_coordinates"))
    figures.extend(plot_calendar_coordinates(rows, figure_dir / f"unicm_start_month_{phase_name}_calendar_coordinates"))

    summary: dict[str, object] = {
        "phase": phase_name,
        "n_samples": int(n_samples),
        "n_checkpoints": int(rows["seed"].nunique()),
        "n_start_months": int(rows["start_month"].nunique()),
        "n_leads": int(rows["lead"].nunique()),
        "minimum_phi_eid_bits": float(rows["phi_eid"].min()),
        "maximum_phi_eid_bits": float(rows["phi_eid"].max()),
        "nonnegative_tolerance_bits": float(nonnegative_tolerance),
        "negative_within_tolerance_count": int(len(negative_within)),
        "significant_nonnegativity_violation_count": 0,
        "modal_peak_lead": lead_mode,
        "modal_peak_calendar_month": calendar_mode,
        "modal_peak_calendar_month_name": MONTHS[calendar_mode],
        "peak_in_lead_7_10_fraction": lead_window_fraction,
        "peak_in_jul_oct_fraction": calendar_window_fraction,
        "peak_lead_concentration": lead_concentration,
        "peak_calendar_month_concentration": calendar_concentration,
        "classification": classification,
        "variance_decomposition": variance,
        "figures": [str(path) for path in figures],
        "rows": str(rows_path),
    }
    atomic_write_json(output_dir / "summary.json", summary)

    checkpoint_text = ", ".join(str(value) for value in sorted(rows["seed"].unique()))
    start_month_text = ", ".join(str(value) for value in sorted(rows["start_month"].unique()))
    lead_text = ", ".join(str(value) for value in sorted(rows["lead"].unique()))
    report = f"""# UniCM 12-start-month sweep: {phase_name}

## Stable finding

The descriptive classification is **{classification}**.

## Evidence

- Samples per condition: `{n_samples}`.
- Checkpoints: `{checkpoint_text}`; start months: `{start_month_text}`; leads: `{lead_text}`.
- Modal peak lead: `{lead_mode}`; fraction of checkpoint--start-month units peaking at leads 7--10: `{lead_window_fraction:.3f}`.
- Modal peak target month: `{MONTHS[calendar_mode]}`; fraction peaking in July--October: `{calendar_window_fraction:.3f}`.
- Variance fractions: lead `{variance['eta2_lead']:.3f}`, calendar month `{variance['eta2_calendar_month']:.3f}`, checkpoint `{variance['eta2_checkpoint']:.3f}`, residual/interactions `{variance['eta2_residual_and_interactions']:.3f}`.
- Signed nonnegativity audit: minimum `{rows['phi_eid'].min():.6g}` bits; tolerance `{nonnegative_tolerance}` bits; `{len(negative_within)}` values within tolerance; no violations below tolerance.

## Controls and limits

Only the calendar-month timestamps change. All histories, intervention support, checkpoints, leads, source/target definitions, and affine degree-1 TM estimation are paired. The experiment isolates UniCM's month embedding under a maximum-entropy input distribution; it is not an empirical rolling-initialization forecast evaluation.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return summary


def aggregate_condition_metrics(metrics_dir: Path, seeds: list[int], start_months: list[int]) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        for start_month in start_months:
            path = condition_metric_path(metrics_dir, seed=seed, start_month=start_month)
            if not path.exists():
                raise FileNotFoundError(f"Missing condition metrics: {path}")
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def run_phase(
    *,
    phase_name: str,
    n_samples: int,
    seeds: list[int],
    start_months: list[int],
    checkpoint_paths: dict[int, Path],
    output_root: Path,
    figure_root: Path,
    device: str,
    batch_size: int,
    tracker: ProgressTracker,
    nonnegative_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    import torch

    phase_dir = output_root / f"{phase_name}_n{n_samples}"
    cache_dir = phase_dir / "cache"
    metrics_dir = phase_dir / "condition_metrics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    history_modes = sample_full_history_mode_inputs(
        n_samples=int(n_samples), intervention_bound=4.0, seed=20260619
    )
    estimator, estimator_metadata = create_ei_estimator(
        "transport_map", tm_degree=1, tm_jitter=1e-6, clip_negative=False
    )
    phase_manifest = {
        "phase": phase_name,
        "n_samples": int(n_samples),
        "sampling_seed": 20260619,
        "intervention_bound": 4.0,
        "seeds": seeds,
        "start_months": start_months,
        "leads": list(range(1, 25)),
        "source_definition": "all 11 mode histories; 12 months per mode",
        "target_definition": "all 11 predicted modes at each lead",
        "estimator": estimator_metadata,
        "device": device,
        "batch_size": int(batch_size),
        "signed_outputs": True,
        "nonnegative_tolerance_bits": float(nonnegative_tolerance),
    }
    atomic_write_json(phase_dir / "manifest.json", phase_manifest)

    total_conditions = len(seeds) * len(start_months)
    condition_index = 0
    progress = tqdm(total=total_conditions, desc=f"{phase_name} n={n_samples}", unit="condition", mininterval=1.0)
    for seed in seeds:
        tracker.write(
            phase=f"{phase_name}:load_checkpoint",
            message=f"Loading UniCM checkpoint seed {seed}",
            condition_current=condition_index,
            condition_total=total_conditions,
        )
        model = load_unicm_model(checkpoint_paths[seed], device)
        for start_month in start_months:
            condition_index += 1
            expected_metadata = cache_metadata(
                seed=seed,
                start_month=start_month,
                n_samples=n_samples,
                device=device,
                batch_size=batch_size,
            )
            prediction_path = cache_path(
                cache_dir, seed=seed, start_month=start_month, n_samples=n_samples, device=device
            )
            metric_path = condition_metric_path(metrics_dir, seed=seed, start_month=start_month)
            tracker.write(
                phase=f"{phase_name}:predict",
                message=f"seed={seed}, start_month={start_month}, n={n_samples}",
                metrics={"seed": seed, "start_month": start_month, "n_samples": n_samples},
                condition_current=condition_index - 1,
                condition_total=total_conditions,
            )
            started = time.monotonic()
            cache_reused = False
            if prediction_path.exists():
                try:
                    predictions = load_prediction_cache(prediction_path, expected_metadata)
                    cache_reused = True
                except ValueError as exc:
                    tracker.write(
                        phase=f"{phase_name}:recompute_invalid_cache",
                        message=f"Recomputing invalid cache for seed={seed}, start_month={start_month}",
                        metrics={
                            "seed": seed,
                            "start_month": start_month,
                            "n_samples": n_samples,
                            "reason": str(exc),
                        },
                        condition_current=condition_index - 1,
                        condition_total=total_conditions,
                    )
            if not cache_reused:
                predictions = predict_modeformer_all_modes_from_history(
                    model,
                    history_modes,
                    device=device,
                    batch_size=int(batch_size),
                    start_month=int(start_month),
                )
                nonfinite_count = int(np.size(predictions) - np.isfinite(predictions).sum())
                if nonfinite_count:
                    raise RuntimeError(
                        "UniCM produced non-finite predictions before estimation: "
                        f"seed={seed}, start_month={start_month}, n_samples={n_samples}, "
                        f"batch_size={batch_size}, count={nonfinite_count}"
                    )
                save_prediction_cache(prediction_path, predictions, expected_metadata)

            metrics_reused = False
            if metric_path.exists():
                candidate_rows = pd.read_csv(metric_path)
                metrics_reused = condition_metrics_are_valid(
                    candidate_rows,
                    seed=seed,
                    start_month=start_month,
                    n_samples=n_samples,
                    batch_size=batch_size,
                )
                if metrics_reused:
                    condition_rows = candidate_rows
            if not metrics_reused:
                tracker.write(
                    phase=f"{phase_name}:estimate",
                    message=f"Estimating Xi for seed={seed}, start_month={start_month}",
                    metrics={"seed": seed, "start_month": start_month, "n_samples": n_samples},
                    condition_current=condition_index - 1,
                    condition_total=total_conditions,
                )
                condition_rows = compute_condition_rows(
                    history_modes,
                    predictions,
                    seed=seed,
                    start_month=start_month,
                    n_samples=n_samples,
                    batch_size=batch_size,
                    estimator=estimator,
                )
                condition_rows.to_csv(metric_path, index=False)
            elapsed = time.monotonic() - started
            tracker.write(
                phase=f"{phase_name}:running",
                message=f"Completed seed={seed}, start_month={start_month}",
                metrics={
                    "seed": seed,
                    "start_month": start_month,
                    "n_samples": n_samples,
                    "condition_seconds": elapsed,
                    "cache_reused": cache_reused,
                    "metrics_reused": metrics_reused,
                    "peak_lead": int(condition_rows.loc[condition_rows["phi_eid"].idxmax(), "lead"]),
                    "peak_phi_eid_bits": float(condition_rows["phi_eid"].max()),
                },
                completed_increment=int(n_samples),
                condition_current=condition_index,
                condition_total=total_conditions,
            )
            progress.update(1)
            progress.set_postfix(seed=seed, month=MONTHS[start_month], seconds=f"{elapsed:.1f}")
            del predictions
            if device == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
        del model
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    progress.close()

    rows = aggregate_condition_metrics(metrics_dir, seeds, start_months)
    tracker.write(
        phase=f"{phase_name}:summarize",
        message=f"Summarizing {phase_name} results",
        condition_current=total_conditions,
        condition_total=total_conditions,
    )
    summary = summarize_phase(
        rows,
        phase_name=phase_name,
        n_samples=n_samples,
        output_dir=phase_dir,
        figure_dir=figure_root,
        nonnegative_tolerance=nonnegative_tolerance,
    )
    tracker.write(
        phase=f"{phase_name}:complete",
        message=f"{phase_name} phase complete: {summary['classification']}",
        metrics={
            "classification": summary["classification"],
            "modal_peak_lead": summary["modal_peak_lead"],
            "modal_peak_calendar_month": summary["modal_peak_calendar_month_name"],
        },
        condition_current=total_conditions,
        condition_total=total_conditions,
    )
    return rows, summary


def compare_phases(
    preview_rows: pd.DataFrame,
    full_rows: pd.DataFrame,
    preview_summary: dict[str, object],
    full_summary: dict[str, object],
    output_root: Path,
) -> dict[str, object]:
    keys = ["seed", "start_month", "lead"]
    merged = preview_rows[keys + ["phi_eid"]].merge(
        full_rows[keys + ["phi_eid"]], on=keys, suffixes=("_preview", "_full"), validate="one_to_one"
    )
    pearson = float(merged["phi_eid_preview"].corr(merged["phi_eid_full"], method="pearson"))
    preview_rank = merged["phi_eid_preview"].rank(method="average")
    full_rank = merged["phi_eid_full"].rank(method="average")
    spearman = float(preview_rank.corr(full_rank, method="pearson"))

    def peak_table(rows: pd.DataFrame, name: str) -> pd.DataFrame:
        indices = rows.groupby(["seed", "start_month"])["phi_eid"].idxmax()
        return rows.loc[indices, ["seed", "start_month", "lead", "target_calendar_month"]].rename(
            columns={"lead": f"peak_lead_{name}", "target_calendar_month": f"peak_calendar_month_{name}"}
        )

    peaks = peak_table(preview_rows, "preview").merge(
        peak_table(full_rows, "full"), on=["seed", "start_month"], validate="one_to_one"
    )
    comparison = {
        "cellwise_pearson": pearson,
        "cellwise_spearman": spearman,
        "exact_peak_lead_agreement_fraction": float((peaks["peak_lead_preview"] == peaks["peak_lead_full"]).mean()),
        "peak_lead_within_one_month_fraction": float(
            (np.abs(peaks["peak_lead_preview"] - peaks["peak_lead_full"]) <= 1).mean()
        ),
        "exact_peak_calendar_month_agreement_fraction": float(
            (peaks["peak_calendar_month_preview"] == peaks["peak_calendar_month_full"]).mean()
        ),
        "preview_classification": preview_summary["classification"],
        "full_classification": full_summary["classification"],
    }
    atomic_write_json(output_root / "preview_vs_full_comparison.json", comparison)
    merged.to_csv(output_root / "preview_vs_full_cell_comparison.csv", index=False)
    peaks.to_csv(output_root / "preview_vs_full_peak_comparison.csv", index=False)
    return comparison


def build_final_report(
    preview_summary: dict[str, object],
    full_summary: dict[str, object],
    comparison: dict[str, object],
    output_root: Path,
) -> None:
    full_variance = dict(full_summary["variance_decomposition"])
    report = f"""# UniCM 12-start-month sweep

## Stable finding

The formal 8,192-sample experiment is classified as **{full_summary['classification']}**.

## Formal evidence

- Modal peak lead: `{full_summary['modal_peak_lead']}`; lead-7--10 peak fraction: `{full_summary['peak_in_lead_7_10_fraction']:.3f}`.
- Modal peak target month: `{full_summary['modal_peak_calendar_month_name']}`; July--October peak fraction: `{full_summary['peak_in_jul_oct_fraction']:.3f}`.
- Variance fractions: lead `{full_variance['eta2_lead']:.3f}`, calendar month `{full_variance['eta2_calendar_month']:.3f}`, checkpoint `{full_variance['eta2_checkpoint']:.3f}`, residual/interactions `{full_variance['eta2_residual_and_interactions']:.3f}`.
- Signed nonnegativity audit: minimum `{full_summary['minimum_phi_eid_bits']:.6g}` bits; tolerance `{full_summary['nonnegative_tolerance_bits']}` bits; no significant violations.

## Preview-to-formal validation

- Cell-wise Pearson/Spearman: `{comparison['cellwise_pearson']:.4f}` / `{comparison['cellwise_spearman']:.4f}`.
- Exact peak-lead agreement: `{comparison['exact_peak_lead_agreement_fraction']:.3f}`; agreement within one month: `{comparison['peak_lead_within_one_month_fraction']:.3f}`.
- Preview classification: `{comparison['preview_classification']}`.
- Formal classification: `{comparison['full_classification']}`.

## Interpretation boundary

Only timestamp month embeddings changed; intervention histories and all other factors were paired. The result therefore distinguishes lead-locked from calendar-locked organization inside the frozen Modeformer under maximum-entropy interventions. It does not establish observational seasonal forecast skill because empirical month-conditioned input distributions were not used.
"""
    (output_root / "report.md").write_text(report, encoding="utf-8")


def parse_int_list(raw: list[int]) -> list[int]:
    values = [int(value) for value in raw]
    if len(values) != len(set(values)):
        raise ValueError("Repeated values are not allowed.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paired UniCM start-month preview and formal sweeps.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIG_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "data" / "UniCM-checkpoint" / "src" / "experiments")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--start-months", nargs="+", type=int, default=list(range(12)))
    parser.add_argument("--preview-samples", type=int, default=1024)
    parser.add_argument("--full-samples", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--nonnegative-tolerance", type=float, default=0.002)
    parser.add_argument("--skip-full", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = parse_int_list(args.seeds)
    start_months = parse_int_list(args.start_months)
    if any(month < 0 or month > 11 for month in start_months):
        raise ValueError("start months must be in [0, 11].")
    if int(args.preview_samples) <= 0 or int(args.full_samples) <= 0:
        raise ValueError("sample counts must be positive.")

    import torch

    torch.set_num_threads(int(args.torch_threads))
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")

    output_root = Path(args.output_root)
    figure_root = Path(args.figure_root)
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = resolve_checkpoint_paths(Path(args.checkpoint_root), seeds)
    conditions = len(seeds) * len(start_months)
    total_work = conditions * int(args.preview_samples)
    if not args.skip_full:
        total_work += conditions * int(args.full_samples)
    tracker = ProgressTracker(output_root / "live_progress.json", total_work=total_work)
    tracker.write(phase="setup", message="Validating experiment contract and checkpoints")

    try:
        preview_rows, preview_summary = run_phase(
            phase_name="preview",
            n_samples=int(args.preview_samples),
            seeds=seeds,
            start_months=start_months,
            checkpoint_paths=checkpoint_paths,
            output_root=output_root,
            figure_root=figure_root,
            device=str(args.device),
            batch_size=int(args.batch_size),
            tracker=tracker,
            nonnegative_tolerance=float(args.nonnegative_tolerance),
        )
        if args.skip_full:
            tracker.write(
                phase="complete",
                message=f"Preview complete: {preview_summary['classification']}",
                metrics={"preview_classification": preview_summary["classification"]},
            )
            return 0

        full_rows, full_summary = run_phase(
            phase_name="full",
            n_samples=int(args.full_samples),
            seeds=seeds,
            start_months=start_months,
            checkpoint_paths=checkpoint_paths,
            output_root=output_root,
            figure_root=figure_root,
            device=str(args.device),
            batch_size=int(args.batch_size),
            tracker=tracker,
            nonnegative_tolerance=float(args.nonnegative_tolerance),
        )
        comparison = compare_phases(preview_rows, full_rows, preview_summary, full_summary, output_root)
        build_final_report(preview_summary, full_summary, comparison, output_root)
        tracker.write(
            phase="complete",
            message=f"Preview and formal sweep complete: {full_summary['classification']}",
            metrics={
                "preview_classification": preview_summary["classification"],
                "full_classification": full_summary["classification"],
                "cellwise_pearson": comparison["cellwise_pearson"],
                "cellwise_spearman": comparison["cellwise_spearman"],
            },
        )
        return 0
    except Exception as exc:
        tracker.write(
            phase="failed",
            message=str(exc),
            metrics={"traceback": traceback.format_exc(limit=12)},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
