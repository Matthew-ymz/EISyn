#!/usr/bin/env python3
"""Measure horizon-integrated source-mode condensation from cached TM rankings."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "results/runge_source_pair_condensation_null_smoke"
EXTENSION_DIR = BASE_DIR / "spectral_extension"
OUTPUT_DIR = BASE_DIR / "condensation_strength"
FIGURE_BASE = ROOT / "fig/runge_source_mode_condensation_strength"
HORIZONS = (1, 10, 20, 60)
TOP_KS = (50, 100, 200, 500)
PRIMARY_TOP_K = 200
N_SOURCES = 60
NONNEGATIVE_TOLERANCE_BITS = 1e-10
EARTH_CONDITIONS = ("earth_hybrid", "earth_linear")
RANDOM_NULLS = tuple(f"null_{index:02d}" for index in range(5))
DESIGNED_CONTROLS = (
    "shape_low_00",
    "shape_low_01",
    "shape_mid_00",
    "shape_mid_01",
    "shape_high_00",
    "shape_high_01",
)


def ranking_path(condition: str, horizon: int) -> Path:
    root = BASE_DIR if condition in (*EARTH_CONDITIONS, *RANDOM_NULLS) else EXTENSION_DIR
    return root / "top500" / f"{condition}_H{horizon:03d}.npz"


def source_participation(
    path: Path,
    *,
    top_k: int,
    n_sources: int = N_SOURCES,
    tolerance: float = NONNEGATIVE_TOLERANCE_BITS,
) -> tuple[np.ndarray, dict[str, float | int]]:
    with np.load(path) as ranking:
        source_a = np.asarray(ranking["source_a"][:top_k], dtype=int)
        source_b = np.asarray(ranking["source_b"][:top_k], dtype=int)
        weights = np.asarray(ranking["delta2_tm"][:top_k], dtype=float)
    violations = weights < -float(tolerance)
    if np.any(violations):
        raise RuntimeError(
            f"Significant Syn nonnegativity violation in {path}: "
            f"minimum={weights.min():.6g}, tolerance={tolerance:.6g}, count={violations.sum()}."
        )
    near_zero = (weights < 0.0) & ~violations
    audited = weights.copy()
    audited[near_zero] = 0.0
    total = float(audited.sum())
    if total <= 0.0:
        raise RuntimeError(f"Nonpositive top-{top_k} Syn mass in {path}.")
    participation = np.zeros(n_sources, dtype=float)
    np.add.at(participation, source_a, 0.5 * audited)
    np.add.at(participation, source_b, 0.5 * audited)
    participation /= total
    return participation, {
        "minimum_syn_bits": float(weights.min()),
        "negative_within_tolerance_count": int(near_zero.sum()),
        "significant_violation_count": int(violations.sum()),
    }


def instantaneous_metrics(participation: np.ndarray) -> dict[str, float]:
    positive = participation[participation > 0.0]
    effective = float(1.0 / np.sum(participation**2))
    entropy = float(-np.sum(positive * np.log(positive)))
    return {
        "effective_source_count": effective,
        "effective_source_concentration": float((N_SOURCES - effective) / (N_SOURCES - 1)),
        "normalized_source_entropy": float(entropy / math.log(N_SOURCES)),
        "entropy_concentration": float(1.0 - entropy / math.log(N_SOURCES)),
        "maximum_source_share": float(np.max(participation)),
        "top5_source_share": float(np.sort(participation)[-5:].sum()),
    }


def integrate_over_log_horizon(values: list[float]) -> float:
    x = np.log(np.asarray(HORIZONS, dtype=float))
    x = (x - x[0]) / (x[-1] - x[0])
    return float(np.trapz(np.asarray(values, dtype=float), x))


def condition_metrics(condition: str, top_k: int) -> dict[str, object]:
    by_horizon: dict[str, dict[str, float]] = {}
    audits: dict[str, dict[str, float | int]] = {}
    for horizon in HORIZONS:
        participation, audit = source_participation(ranking_path(condition, horizon), top_k=top_k)
        by_horizon[str(horizon)] = instantaneous_metrics(participation)
        audits[str(horizon)] = audit
    effective_counts = [by_horizon[str(h)]["effective_source_count"] for h in HORIZONS]
    effective_concentration = [by_horizon[str(h)]["effective_source_concentration"] for h in HORIZONS]
    sustained_strength = integrate_over_log_horizon(effective_concentration)
    dynamic_gain = integrate_over_log_horizon(
        [value - effective_concentration[0] for value in effective_concentration]
    )
    available_headroom = 1.0 - effective_concentration[0]
    return {
        "condition": condition,
        "top_k": top_k,
        "by_horizon": by_horizon,
        "strength": {
            "sustained_source_concentration_auc": sustained_strength,
            "baseline_adjusted_concentration_gain_auc": dynamic_gain,
            "headroom_normalized_concentration_gain_auc": float(dynamic_gain / available_headroom),
            "entropy_concentration_auc": integrate_over_log_horizon(
                [by_horizon[str(h)]["entropy_concentration"] for h in HORIZONS]
            ),
            "top5_source_share_auc": integrate_over_log_horizon(
                [by_horizon[str(h)]["top5_source_share"] for h in HORIZONS]
            ),
            "endpoint_log_effective_contraction": float(math.log(effective_counts[0] / effective_counts[-1])),
        },
        "nonnegative_audit": audits,
    }


def empirical_summary(observed: float, null_values: list[float]) -> dict[str, float | int]:
    null = np.asarray(null_values, dtype=float)
    sample_sd = float(np.std(null, ddof=1)) if len(null) > 1 else float("nan")
    exceedances = int(np.sum(null >= observed))
    return {
        "observed": float(observed),
        "null_n": int(len(null)),
        "null_mean": float(np.mean(null)),
        "null_sample_sd": sample_sd,
        "null_min": float(np.min(null)),
        "null_max": float(np.max(null)),
        "standardized_effect": float((observed - np.mean(null)) / sample_sd),
        "null_exceedance_count": exceedances,
        "one_sided_empirical_p_plus_one": float((1 + exceedances) / (len(null) + 1)),
    }


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def plot_summary(results: dict[str, dict[str, dict[str, object]]], output_base: Path) -> list[str]:
    configure_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), layout="constrained")
    colors = {"earth_hybrid": "#D97732", "earth_linear": "#356A8A"}

    axis = axes[0]
    null_curves = np.asarray(
        [[results[name][str(PRIMARY_TOP_K)]["by_horizon"][str(h)]["effective_source_count"] for h in HORIZONS] for name in RANDOM_NULLS]
    )
    axis.fill_between(HORIZONS, null_curves.min(axis=0), null_curves.max(axis=0), color="#CBD1D6", alpha=0.55, label="Random null range (n=5)")
    axis.plot(HORIZONS, null_curves.mean(axis=0), color="#8C969F", marker="o", linewidth=1.1, label="Random null mean")
    for condition, label in (("earth_hybrid", "SLP hybrid"), ("earth_linear", "SLP linear")):
        values = [results[condition][str(PRIMARY_TOP_K)]["by_horizon"][str(h)]["effective_source_count"] for h in HORIZONS]
        axis.plot(HORIZONS, values, color=colors[condition], marker="o", linewidth=1.5, label=label)
    axis.set_xscale("log")
    axis.set_xticks(HORIZONS, labels=[str(h) for h in HORIZONS])
    axis.set_xlabel("Forecast horizon")
    axis.set_ylabel("Effective source modes")
    axis.grid(color="#E8EBEF", linewidth=0.5)
    axis.text(-0.17, 1.04, "a", transform=axis.transAxes, fontsize=8.2, fontweight="bold")

    axis = axes[1]
    primary_key = "sustained_source_concentration_auc"
    dynamic_key = "headroom_normalized_concentration_gain_auc"
    null_strength = [results[name][str(PRIMARY_TOP_K)]["strength"][primary_key] for name in RANDOM_NULLS]
    designed_strength = [results[name][str(PRIMARY_TOP_K)]["strength"][primary_key] for name in DESIGNED_CONTROLS]
    axis.scatter(np.zeros(len(null_strength)), null_strength, color="#929CA5", s=20)
    axis.scatter(np.ones(len(designed_strength)), designed_strength, facecolor="white", edgecolor="#70838E", s=20)
    for index, condition in enumerate(EARTH_CONDITIONS, start=2):
        axis.scatter(index, results[condition][str(PRIMARY_TOP_K)]["strength"][primary_key], color=colors[condition], marker="*", s=55, zorder=4)
    axis.set_xticks([0, 1, 2, 3], labels=["Random\nnull", "Spectral\ncontrols", "SLP\nhybrid", "SLP\nlinear"])
    axis.set_ylabel("Sustained source concentration")
    axis.grid(axis="y", color="#E8EBEF", linewidth=0.5)
    axis.text(-0.17, 1.04, "b", transform=axis.transAxes, fontsize=8.2, fontweight="bold")

    axis = axes[2]
    null_dynamic = [results[name][str(PRIMARY_TOP_K)]["strength"][dynamic_key] for name in RANDOM_NULLS]
    designed_dynamic = [results[name][str(PRIMARY_TOP_K)]["strength"][dynamic_key] for name in DESIGNED_CONTROLS]
    axis.scatter(np.zeros(len(null_dynamic)), null_dynamic, color="#929CA5", s=20)
    axis.scatter(np.ones(len(designed_dynamic)), designed_dynamic, facecolor="white", edgecolor="#70838E", s=20)
    for index, condition in enumerate(EARTH_CONDITIONS, start=2):
        axis.scatter(index, results[condition][str(PRIMARY_TOP_K)]["strength"][dynamic_key], color=colors[condition], marker="*", s=55, zorder=4)
    axis.set_xticks([0, 1, 2, 3], labels=["Random\nnull", "Spectral\ncontrols", "SLP\nhybrid", "SLP\nlinear"])
    axis.set_ylabel("Baseline-adjusted concentration gain")
    axis.grid(axis="y", color="#E8EBEF", linewidth=0.5)
    axis.text(-0.17, 1.04, "c", transform=axis.transAxes, fontsize=8.2, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=5.8)
    outputs: list[str] = []
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 600}), (".svg", {}), (".pdf", {})):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conditions = (*EARTH_CONDITIONS, *RANDOM_NULLS, *DESIGNED_CONTROLS)
    results = {condition: {str(k): condition_metrics(condition, k) for k in TOP_KS} for condition in conditions}
    primary_key = "sustained_source_concentration_auc"
    dynamic_key = "headroom_normalized_concentration_gain_auc"
    inference = {
        condition: empirical_summary(
            float(results[condition][str(PRIMARY_TOP_K)]["strength"][primary_key]),
            [float(results[name][str(PRIMARY_TOP_K)]["strength"][primary_key]) for name in RANDOM_NULLS],
        )
        for condition in EARTH_CONDITIONS
    }
    top_k_robustness = {
        str(k): empirical_summary(
            float(results["earth_hybrid"][str(k)]["strength"][primary_key]),
            [float(results[name][str(k)]["strength"][primary_key]) for name in RANDOM_NULLS],
        )
        for k in TOP_KS
    }
    dynamic_inference = {
        condition: empirical_summary(
            float(results[condition][str(PRIMARY_TOP_K)]["strength"][dynamic_key]),
            [float(results[name][str(PRIMARY_TOP_K)]["strength"][dynamic_key]) for name in RANDOM_NULLS],
        )
        for condition in EARTH_CONDITIONS
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "primary_metric": {
            "name": primary_key,
            "top_k": PRIMARY_TOP_K,
            "direction": "higher means stronger concentration sustained across the forecast range",
            "range": [0.0, 1.0],
            "definition": "log-horizon area of (60 - inverse_Simpson_source_count) / 59",
        },
        "conditions": results,
        "random_null_inference": inference,
        "baseline_adjusted_dynamic_inference": dynamic_inference,
        "top_k_robustness_earth_hybrid": top_k_robustness,
        "designed_spectral_controls_are_not_exchangeable_null_draws": True,
        "figure_contract": {
            "core_conclusion": "Sustained source concentration separates descriptively, whereas baseline-adjusted horizon-induced gain does not.",
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib",
            "panels": {"a": "effective source trajectory", "b": "sustained concentration", "c": "baseline-adjusted gain"},
            "reviewer_risk": "n=5 null resolution and loss of separation at top-500",
        },
    }
    summary["figure_outputs"] = plot_summary(results, FIGURE_BASE)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    hybrid = inference["earth_hybrid"]
    linear = inference["earth_linear"]
    hybrid_dynamic = dynamic_inference["earth_hybrid"]
    report = f"""# Source-mode condensation strength

## Definition

For each horizon, each source pair's positive audited Syn mass is split equally
between its two endpoint modes. If the resulting 60-mode participation
distribution is $q$, its effective source count is $N_{{eff}}=1/\\sum_i q_i^2$.
The instantaneous concentration is $(60-N_{{eff}})/59$. The sustained
concentration score is its trapezoidal area over normalized log forecast horizon.
It lies in $[0,1]$, is invariant to source relabeling and a common rescaling of
Syn, and rewards focality that appears early and persists. A separate dynamic
score subtracts the H=1 concentration and normalizes by its remaining headroom.

## Result

- Primary top-K: 200, fixed by the original experiment contract.
- SLP hybrid sustained concentration: `{hybrid['observed']:.3f}`.
- SLP linear sustained concentration: `{linear['observed']:.3f}`.
- Five random matched-radius nulls: `{hybrid['null_min']:.3f}`--`{hybrid['null_max']:.3f}`, mean `{hybrid['null_mean']:.3f}`.
- Standardized hybrid-minus-null effect: `{hybrid['standardized_effect']:.2f}` null SD.
- The hybrid exceeds all five random nulls, but the plus-one empirical one-sided p-value is `{hybrid['one_sided_empirical_p_plus_one']:.3f}` because n=5.
- Baseline-adjusted dynamic gain for the hybrid is `{hybrid_dynamic['observed']:.3f}`,
  versus null range `{hybrid_dynamic['null_min']:.3f}`--`{hybrid_dynamic['null_max']:.3f}`;
  this does not separate.

## Robustness and boundary

The defensible interpretation is therefore unusually strong sustained source
focality across the forecast range, not unusually strong horizon-induced
condensation relative to H=1. The sustained score exceeds all random nulls for
top-50, top-100, and top-200, but not at top-500. The six spectrum-stratified controls are
reported descriptively but are not exchangeable random null draws because they
were selected by spectral diversity. A formal 0.05 tail claim requires at least
19 independent prespecified null draws; 99 are recommended for a stable empirical
tail estimate.
"""
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"inference": inference, "dynamic_inference": dynamic_inference, "top_k_robustness": top_k_robustness}, indent=2))


if __name__ == "__main__":
    main()
