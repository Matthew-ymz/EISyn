#!/usr/bin/env python3
"""Extend the Runge condensation null with spectrum-stratified VARs and a radius sweep."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from scripts.run_runge_exhaustive_degree3_tm import DEFAULT_PAIRWISE_MANIFEST, prepare_source_cache
from scripts.run_runge_source_pair_condensation_null import (
    DEFAULT_HYBRID_RANKING_DIR,
    ProgressRecorder,
    atomic_savez,
    atomic_write_json,
    calibrate_spectral_radius,
    companion_matrix,
    fingerprint_array,
    fit_linear_backbone,
    lag_blocks,
    parse_ints,
    rewire_var_coefficients,
    rollout_linear_var,
    score_prediction_condition,
    spectrum_descriptors,
)


DEFAULT_BASE_DIR = ROOT / "results/runge_source_pair_condensation_null_smoke"
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DIR / "spectral_extension"
DEFAULT_FIGURE_BASE = ROOT / "fig/runge_source_pair_condensation_spectral_extension"
DEFAULT_EIGEN_FIGURE_BASE = ROOT / "fig/runge_source_pair_condensation_eigenspectra"
DEFAULT_HORIZONS = (1, 10, 20, 60)
DEFAULT_CANDIDATE_SEEDS = tuple(range(42100, 42124))
DEFAULT_RADIUS_LEVELS = (0.80, 0.90, 0.98)
DEFAULT_TOP_KS = (50, 100, 200, 500)
PRIMARY_TOP_K = 200


def calibrated_bias_for_fixed_point(
    weight: np.ndarray,
    fixed_point: np.ndarray,
    *,
    n_components: int,
    lag: int,
) -> np.ndarray:
    return np.asarray(fixed_point, dtype=float) - sum(
        lag_blocks(weight, n_components=n_components, lag=lag)
    ) @ np.asarray(fixed_point, dtype=float)


def select_spectrum_strata(candidates: list[dict[str, object]], *, per_stratum: int = 2) -> list[dict[str, object]]:
    key = "spectral_effective_modes_h60"
    ordered = sorted(candidates, key=lambda row: float(row["spectrum"][key]))
    if len(ordered) < 3 * int(per_stratum):
        raise ValueError("Not enough candidates for low/mid/high spectrum strata.")
    low = ordered[: int(per_stratum)]
    high = ordered[-int(per_stratum) :]
    excluded = {int(row["seed"]) for row in [*low, *high]}
    median = float(np.median([float(row["spectrum"][key]) for row in ordered]))
    middle = sorted(
        [row for row in ordered if int(row["seed"]) not in excluded],
        key=lambda row: abs(float(row["spectrum"][key]) - median),
    )[: int(per_stratum)]
    selected: list[dict[str, object]] = []
    for stratum, rows in (("low", low), ("mid", middle), ("high", high)):
        for index, row in enumerate(rows):
            selected.append({**row, "stratum": stratum, "condition": f"shape_{stratum}_{index:02d}"})
    return selected


def load_metric_rows(directory: Path) -> list[dict[str, object]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((directory / "metrics").glob("*.json"))]


def metric_changes(rows: list[dict[str, object]], *, top_k: int) -> dict[str, dict[str, float | str]]:
    result: dict[str, dict[str, float | str]] = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        selected = sorted(
            [row for row in rows if row["condition"] == condition], key=lambda row: int(row["horizon"])
        )
        if len(selected) < 2:
            continue
        first = selected[0]["top_k_metrics"][str(int(top_k))]
        last = selected[-1]["top_k_metrics"][str(int(top_k))]
        result[condition] = {
            "condition_type": str(selected[0]["condition_type"]),
            "distinct_pair_ratio": float(last["distinct_pair_count"] / first["distinct_pair_count"]),
            "effective_pair_ratio": float(last["effective_pair_count"] / first["effective_pair_count"]),
            "max_pair_share_change": float(last["max_pair_share"] - first["max_pair_share"]),
            "h1_effective_pair_count": float(first["effective_pair_count"]),
            "h60_effective_pair_count": float(last["effective_pair_count"]),
        }
    return result


def correlation_payload(
    condition_names: list[str],
    descriptors: dict[str, dict[str, float | int]],
    changes: dict[str, dict[str, float | str]],
) -> dict[str, object]:
    x_keys = (
        "spectral_gap_fraction",
        "slow_mode_count_90pct",
        "spectral_effective_modes_h60",
        "leading_observed_mode_support",
        "latest_to_future_transfer_effective_rank_h60",
        "companion_nonnormality",
    )
    y_keys = ("distinct_pair_ratio", "effective_pair_ratio", "max_pair_share_change")
    output: dict[str, object] = {"n": len(condition_names), "conditions": condition_names, "spearman": {}}
    for x_key in x_keys:
        output["spearman"][x_key] = {}
        x = np.asarray([float(descriptors[name][x_key]) for name in condition_names], dtype=float)
        for y_key in y_keys:
            y = np.asarray([float(changes[name][y_key]) for name in condition_names], dtype=float)
            rho, p_value = spearmanr(x, y)
            output["spearman"][x_key][y_key] = {
                "rho": float(rho),
                "p_value_exploratory": float(p_value),
            }
    return output


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.17, 1.04, label, transform=axis.transAxes, fontsize=8.2, fontweight="bold", va="bottom")


def save_figure(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in ((".png", {"dpi": 600}), (".svg", {}), (".pdf", {})):
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_spectral_relationships(
    *,
    descriptors: dict[str, dict[str, float | int]],
    changes: dict[str, dict[str, float | str]],
    matched_names: list[str],
    selected_metadata: dict[str, dict[str, object]],
    radius_names: list[str],
    correlation: dict[str, object],
    figure_base: Path,
) -> list[Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7), layout="constrained")
    palette = {"existing": "#9AA4AD", "low": "#6F5B95", "mid": "#7B8E8B", "high": "#3D7D8C"}

    def style(name: str) -> tuple[str, str, float]:
        if name == "earth_linear":
            return "#D97732", "*", 54.0
        metadata = selected_metadata.get(name, {})
        stratum = str(metadata.get("stratum", "existing"))
        return palette.get(stratum, palette["existing"]), "o", 24.0

    panels = (
        (
            "spectral_effective_modes_h60",
            "effective_pair_ratio",
            "Spectral effective modes at H=60",
            "Effective-pair H=60 / H=1",
        ),
        (
            "latest_to_future_transfer_effective_rank_h60",
            "distinct_pair_ratio",
            "Latest-to-future transfer rank at H=60",
            "Distinct-pair H=60 / H=1",
        ),
        (
            "leading_observed_mode_support",
            "max_pair_share_change",
            "Leading-mode source support",
            "Change in maximum pair share",
        ),
    )
    for axis, (x_key, y_key, xlabel, ylabel), letter in zip(axes.flat[:3], panels, ("a", "b", "c")):
        for name in [*matched_names, "earth_linear"]:
            color, marker, size = style(name)
            axis.scatter(
                float(descriptors[name][x_key]),
                float(changes[name][y_key]),
                color=color,
                marker=marker,
                s=size,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
        rho = correlation["spearman"][x_key][y_key]["rho"]
        axis.text(0.04, 0.95, rf"$\rho_s={rho:.2f}$", transform=axis.transAxes, va="top", fontsize=6.0)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(color="#E8EBEF", linewidth=0.55)
        add_panel_label(axis, letter)

    axis = axes.flat[3]
    sweep_names = [*radius_names, "earth_linear"]
    sweep_names = sorted(sweep_names, key=lambda name: float(descriptors[name]["spectral_radius"]))
    radii = np.asarray([float(descriptors[name]["spectral_radius"]) for name in sweep_names])
    distinct = np.asarray([float(changes[name]["distinct_pair_ratio"]) for name in sweep_names])
    effective = np.asarray([float(changes[name]["effective_pair_ratio"]) for name in sweep_names])
    line_a = axis.plot(radii, distinct, color="#356A8A", marker="o", linewidth=1.4, label="Distinct pairs")[0]
    line_b = axis.plot(radii, effective, color="#D97732", marker="s", linewidth=1.4, label="Effective pairs")[0]
    axis.set_xlabel("Companion spectral radius")
    axis.set_ylabel("H=60 / H=1 ratio")
    axis.set_ylim(bottom=0.0)
    axis.grid(color="#E8EBEF", linewidth=0.55)
    add_panel_label(axis, "d")

    legend_handles = [
        plt.Line2D([], [], marker="*", linestyle="none", color="#D97732", markersize=8, label="Fitted SLP linear"),
        plt.Line2D([], [], marker="o", linestyle="none", color=palette["existing"], markersize=4, label="Initial matched null"),
        plt.Line2D([], [], marker="o", linestyle="none", color=palette["low"], markersize=4, label="Low spectral diversity"),
        plt.Line2D([], [], marker="o", linestyle="none", color=palette["mid"], markersize=4, label="Mid spectral diversity"),
        plt.Line2D([], [], marker="o", linestyle="none", color=palette["high"], markersize=4, label="High spectral diversity"),
        line_a,
        line_b,
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=5.8,
        handlelength=1.8,
    )
    fig.text(0.5, -0.015, "Matched-shape correlations are exploratory; n is reported in the source summary.", ha="center", fontsize=5.7)
    return save_figure(fig, figure_base)


def plot_eigenspectra(
    *,
    weights: dict[str, np.ndarray],
    descriptors: dict[str, dict[str, float | int]],
    representative_names: list[str],
    figure_base: Path,
) -> list[Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), layout="constrained", sharex=True, sharey=True)
    display = {
        "earth_linear": "Fitted SLP linear",
        "rho_080": r"Same organization, $\rho=0.80$",
        "rho_098": r"Same organization, $\rho=0.98$",
    }
    for axis, name, letter in zip(axes.flat, representative_names, ("a", "b", "c", "d", "e", "f")):
        values = np.linalg.eigvals(companion_matrix(weights[name], n_components=60, lag=4))
        magnitudes = np.abs(values)
        axis.scatter(
            np.real(values),
            np.imag(values),
            c=magnitudes,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=7,
            alpha=0.72,
            edgecolors="none",
        )
        theta = np.linspace(0.0, 2.0 * np.pi, 400)
        radius = float(descriptors[name]["spectral_radius"])
        axis.plot(radius * np.cos(theta), radius * np.sin(theta), color="#6F7780", linestyle=":", linewidth=0.7)
        axis.axhline(0.0, color="#D9DDE1", linewidth=0.45)
        axis.axvline(0.0, color="#D9DDE1", linewidth=0.45)
        title = display.get(name, name.replace("shape_", "Shape ").replace("_", " "))
        axis.set_title(title, fontsize=6.3, fontweight="bold")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-1.02, 1.02)
        axis.set_ylim(-1.02, 1.02)
        axis.set_xlabel(r"Re($\lambda$)")
        axis.set_ylabel(r"Im($\lambda$)")
        add_panel_label(axis, letter)
    return save_figure(fig, figure_base)


def build_report(summary: dict[str, object], output_dir: Path) -> None:
    corr = summary["matched_shape_correlation"]["spearman"]["spectral_effective_modes_h60"][
        "effective_pair_ratio"
    ]
    radius = summary["radius_sweep"]
    text = f"""# Runge source-pair condensation: spectral extension

## Main finding

Long-horizon condensation covaries moderately with the number of dynamically
surviving spectral modes. The effective rank of the actual latest-to-future
transfer is a stronger predictor. The smoke design therefore supports a spectral
mechanism, but not an eigenvalue-only explanation.

## Evidence

- Matched-radius rewired VARs: `n={summary['matched_shape_correlation']['n']}`.
- Spearman correlation between H=60 spectral effective modes and the H=60/H=1
  effective-pair ratio: `{corr['rho']:.3f}` (exploratory p=`{corr['p_value_exploratory']:.4g}`).
- Spearman correlation between H=60 latest-to-future transfer rank and the same
  ratio: `{summary['matched_shape_correlation']['spearman']['latest_to_future_transfer_effective_rank_h60']['effective_pair_ratio']['rho']:.3f}`
  (exploratory p=`{summary['matched_shape_correlation']['spearman']['latest_to_future_transfer_effective_rank_h60']['effective_pair_ratio']['p_value_exploratory']:.4g}`).
- Spectral-radius sweep levels: `{', '.join(f'{value:.3f}' for value in radius['radii'])}`.
- Effective-pair ratios across that sweep: `{', '.join(f'{value:.3f}' for value in radius['effective_pair_ratios'])}`.
- A larger spectral radius raises the H=60/H=1 ratio in this controlled sweep,
  meaning that slower overall decay preserves dispersion rather than causing
  stronger condensation.

## Boundary

The spectrum-stratified rewiring changes eigenvectors and non-normality together
with the non-leading eigenvalue distribution. The separate radius sweep holds
coefficient organization fixed and isolates the dominant decay scale, but does
not independently manipulate every eigenvalue. Therefore the result diagnoses
spectral control without proving that eigenvalues alone determine condensation.
"""
    (output_dir / "report.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    base_dir = Path(args.base_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = parse_ints(args.horizons)
    top_ks = tuple(parse_ints(args.top_ks))
    candidate_seeds = parse_ints(args.candidate_seeds)
    radius_levels = [float(value) for value in str(args.radius_levels).split(",") if value.strip()]
    total = (3 * int(args.per_stratum) + len(radius_levels)) * len(horizons) * 60
    progress = ProgressRecorder(output_dir / "live_progress.json", total=total)
    progress.write(phase="prepare", message="Loading the fitted linear backbone")

    backbone = fit_linear_backbone(Path(args.pairwise_manifest).expanduser().resolve())
    config = backbone["config"]
    features = np.asarray(backbone["features"], dtype=float)
    n_components = len(backbone["names"])
    lag = int(config.lag)
    source_columns = [(lag - 1) * n_components + source for source in range(n_components)]
    sources = features[:, source_columns]
    existing_sources = Path(args.hybrid_ranking_dir).expanduser().resolve() / f"source_samples_n{len(sources)}.npy"
    if not existing_sources.exists() or fingerprint_array(np.load(existing_sources, mmap_mode="r")) != fingerprint_array(sources):
        raise RuntimeError("Spectral extension source samples do not match the published hybrid source samples.")

    earth_spectrum = spectrum_descriptors(backbone["weight"], n_components=n_components, lag=lag, horizon=60)
    descriptors: dict[str, dict[str, float | int]] = {"earth_linear": earth_spectrum}
    weights: dict[str, np.ndarray] = {"earth_linear": np.asarray(backbone["weight"], dtype=float)}
    biases: dict[str, np.ndarray] = {"earth_linear": np.asarray(backbone["bias"], dtype=float)}

    initial_null_seeds = [42000, 42001, 42002, 42003, 42004]
    for index, seed in enumerate(initial_null_seeds):
        weight, bias, _ = rewire_var_coefficients(
            backbone["weight"],
            backbone["bias"],
            n_components=n_components,
            lag=lag,
            seed=seed,
            target_spectral_radius=float(backbone["spectral_radius"]),
            retained_fixed_point=np.asarray(backbone["fixed_point"]),
        )
        name = f"null_{index:02d}"
        weights[name], biases[name] = weight, bias
        descriptors[name] = spectrum_descriptors(weight, n_components=n_components, lag=lag, horizon=60)

    candidates: list[dict[str, object]] = []
    candidate_weights: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for candidate_index, seed in enumerate(candidate_seeds, start=1):
        weight, bias, calibration = rewire_var_coefficients(
            backbone["weight"],
            backbone["bias"],
            n_components=n_components,
            lag=lag,
            seed=seed,
            target_spectral_radius=float(backbone["spectral_radius"]),
            retained_fixed_point=np.asarray(backbone["fixed_point"]),
        )
        spectrum = spectrum_descriptors(weight, n_components=n_components, lag=lag, horizon=60)
        candidates.append({"seed": seed, "calibration": calibration, "spectrum": spectrum})
        candidate_weights[seed] = (weight, bias)
        progress.write(
            phase="candidate_generation",
            message=f"Generated spectral candidate {candidate_index}/{len(candidate_seeds)}",
            metrics={"candidate_seed": seed},
        )
    selected = select_spectrum_strata(candidates, per_stratum=int(args.per_stratum))
    selected_metadata: dict[str, dict[str, object]] = {}
    model_dir = output_dir / "models"
    for row in selected:
        name = str(row["condition"])
        seed = int(row["seed"])
        weight, bias = candidate_weights[seed]
        weights[name], biases[name] = weight, bias
        descriptors[name] = dict(row["spectrum"])
        selected_metadata[name] = row
        atomic_savez(model_dir / f"{name}.npz", weight=weight, bias=bias)

    radius_names: list[str] = []
    for radius in radius_levels:
        weight, scale, achieved = calibrate_spectral_radius(
            backbone["weight"],
            n_components=n_components,
            lag=lag,
            target_spectral_radius=radius,
        )
        bias = calibrated_bias_for_fixed_point(
            weight,
            np.asarray(backbone["fixed_point"]),
            n_components=n_components,
            lag=lag,
        )
        name = f"rho_{int(round(100 * radius)):03d}"
        radius_names.append(name)
        weights[name], biases[name] = weight, bias
        descriptors[name] = spectrum_descriptors(weight, n_components=n_components, lag=lag, horizon=60)
        descriptors[name]["coefficient_scale"] = scale
        descriptors[name]["achieved_spectral_radius"] = achieved
        atomic_savez(model_dir / f"{name}.npz", weight=weight, bias=bias)

    atomic_write_json(output_dir / "candidate_spectrum_bank.json", {"candidates": candidates, "selected": selected})
    atomic_write_json(output_dir / "spectrum_descriptors.json", descriptors)

    new_names = [str(row["condition"]) for row in selected] + radius_names
    source_cache = prepare_source_cache(
        sources,
        degree=int(args.degree),
        ridge=float(args.ridge),
        min_scale=float(args.min_scale),
    )
    total = len(new_names) * len(horizons) * n_components
    if progress.total != total:
        raise RuntimeError(f"Progress total mismatch: prepared {progress.total}, resolved {total}.")
    progress.write(phase="prepare", message="Spectrum strata selected and source TM cache prepared")
    rows: list[dict[str, object]] = []
    with tqdm(total=total, desc="spectral condensation extension", unit="target", mininterval=1.0) as bar:
        for name in new_names:
            predictions = rollout_linear_var(
                weights[name],
                biases[name],
                features,
                n_components=n_components,
                lag=lag,
                horizons=max(horizons),
            )
            rows.extend(
                score_prediction_condition(
                    condition=name,
                    condition_type=("spectrum_stratified_rewired_var4" if name.startswith("shape_") else "fixed_organization_radius_sweep"),
                    predictions=predictions,
                    sources=sources,
                    source_cache=source_cache,
                    horizons=horizons,
                    degree=int(args.degree),
                    ridge=float(args.ridge),
                    min_scale=float(args.min_scale),
                    top_ks=top_ks,
                    nonnegative_tolerance=float(args.nonnegative_tolerance),
                    output_dir=output_dir,
                    progress=progress,
                    bar=bar,
                    diagnostics={"spectrum": descriptors[name]},
                    resume=bool(args.resume),
                )
            )

    base_rows = load_metric_rows(base_dir)
    extension_rows = load_metric_rows(output_dir)
    all_changes = {**metric_changes(base_rows, top_k=PRIMARY_TOP_K), **metric_changes(extension_rows, top_k=PRIMARY_TOP_K)}
    matched_names = [f"null_{index:02d}" for index in range(5)] + [str(row["condition"]) for row in selected]
    correlation = correlation_payload(matched_names, descriptors, all_changes)
    radius_sweep_names = sorted([*radius_names, "earth_linear"], key=lambda name: float(descriptors[name]["spectral_radius"]))
    summary: dict[str, object] = {
        "classification": "exploratory_spectral_control",
        "matched_shape_correlation": correlation,
        "selected_spectrum_strata": selected_metadata,
        "condition_changes": all_changes,
        "spectrum_descriptors": descriptors,
        "radius_sweep": {
            "conditions": radius_sweep_names,
            "radii": [float(descriptors[name]["spectral_radius"]) for name in radius_sweep_names],
            "distinct_pair_ratios": [float(all_changes[name]["distinct_pair_ratio"]) for name in radius_sweep_names],
            "effective_pair_ratios": [float(all_changes[name]["effective_pair_ratio"]) for name in radius_sweep_names],
        },
    }
    relationship_outputs = plot_spectral_relationships(
        descriptors=descriptors,
        changes=all_changes,
        matched_names=matched_names,
        selected_metadata=selected_metadata,
        radius_names=radius_names,
        correlation=correlation,
        figure_base=Path(args.figure_base).expanduser().resolve(),
    )
    shape_names = [str(row["condition"]) for row in selected]
    sorted_shape = sorted(shape_names, key=lambda name: float(descriptors[name]["spectral_effective_modes_h60"]))
    representative = ["earth_linear", sorted_shape[0], sorted_shape[len(sorted_shape) // 2], sorted_shape[-1], "rho_080", "rho_098"]
    eigenspectrum_outputs = plot_eigenspectra(
        weights=weights,
        descriptors=descriptors,
        representative_names=representative,
        figure_base=Path(args.eigen_figure_base).expanduser().resolve(),
    )
    spectra_arrays: dict[str, np.ndarray] = {}
    for name, weight in weights.items():
        values = np.linalg.eigvals(companion_matrix(weight, n_components=n_components, lag=lag))
        spectra_arrays[f"{name}_real"] = np.real(values)
        spectra_arrays[f"{name}_imag"] = np.imag(values)
    atomic_savez(output_dir / "eigenvalues.npz", **spectra_arrays)
    summary["figure_outputs"] = [str(path) for path in [*relationship_outputs, *eigenspectrum_outputs]]
    atomic_write_json(output_dir / "summary.json", summary)
    build_report(summary, output_dir)
    progress.current = total
    progress.write(
        phase="complete",
        message="Spectrum-stratified nulls and radius sweep complete",
        metrics={
            "matched_nulls": len(matched_names),
            "spectral_effective_mode_spearman": correlation["spearman"]["spectral_effective_modes_h60"][
                "effective_pair_ratio"
            ]["rho"],
        },
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise-manifest", default=str(DEFAULT_PAIRWISE_MANIFEST))
    parser.add_argument("--hybrid-ranking-dir", default=str(DEFAULT_HYBRID_RANKING_DIR))
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--figure-base", default=str(DEFAULT_FIGURE_BASE))
    parser.add_argument("--eigen-figure-base", default=str(DEFAULT_EIGEN_FIGURE_BASE))
    parser.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    parser.add_argument("--candidate-seeds", default=",".join(map(str, DEFAULT_CANDIDATE_SEEDS)))
    parser.add_argument("--per-stratum", type=int, default=2)
    parser.add_argument("--radius-levels", default=",".join(map(str, DEFAULT_RADIUS_LEVELS)))
    parser.add_argument("--top-ks", default=",".join(map(str, DEFAULT_TOP_KS)))
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--min-scale", type=float, default=1e-8)
    parser.add_argument("--nonnegative-tolerance", type=float, default=1e-10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    parsed = build_arg_parser().parse_args()
    try:
        result = run(parsed)
    except Exception as error:
        output_dir = Path(parsed.output_dir).expanduser().resolve()
        status_path = output_dir / "live_progress.json"
        prior: dict[str, object] = {}
        if status_path.exists():
            try:
                prior = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                prior = {}
        atomic_write_json(
            status_path,
            {
                "phase": "failed",
                "current": prior.get("current", 0),
                "total": prior.get("total"),
                "unit": "target-score",
                "message": f"{type(error).__name__}: {error}",
                "metrics": {},
                "pid": os.getpid(),
                "updated_at": time.time(),
            },
        )
        raise
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
