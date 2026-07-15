#!/usr/bin/env python3
"""Compare pair-mean and full-system DMF PhiEID peak shapes under matched controls.

Figure contract: the hero curve asks whether averaging two-ROI PhiEID reveals a
critical-region peak; paired seed panels test peak alignment and sharpness
against the already completed full-system experiment.  Python/matplotlib is the
exclusive rendering backend, and seed variation (not inter-pair variation) is
used for inferential error bars.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAIR_INPUT = ROOT / "results" / "dmf_pairwise_phi_eid_confirmation" / "support030_070_tau400_n2048_seeds3_10.npz"
FULL_INPUT = ROOT / "results" / "dmf_phi_eid_peak_alignment" / "paired_support_horizon.npz"
FIGURE = ROOT / "fig" / "dmf_pairwise_phi_eid_confirmation"
PAIR_CURVE_FIGURE = ROOT / "fig" / "dmf_pairwise_phi_eid_mean_curve"
SUMMARY = ROOT / "results" / "dmf_pairwise_phi_eid_confirmation" / "summary.json"
KURAMOTO_KC = 1.5957691216057306


def sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.std(values, axis=axis, ddof=1) / np.sqrt(values.shape[axis])


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def fwhm(g: np.ndarray, values: np.ndarray, peak_index: int) -> float:
    baseline = 0.5 * (float(values[0]) + float(values[-1]))
    height = float(values[peak_index]) - baseline
    if peak_index == 0 or peak_index == len(values) - 1 or height <= 0.0:
        return float("nan")
    level = baseline + 0.5 * height
    left = peak_index
    while left > 0 and values[left] >= level:
        left -= 1
    right = peak_index
    while right < len(values) - 1 and values[right] >= level:
        right += 1
    if left == peak_index or right == peak_index:
        return float("nan")
    left_cross = np.interp(level, (values[left], values[left + 1]), (g[left], g[left + 1]))
    right_cross = np.interp(level, (values[right], values[right - 1]), (g[right], g[right - 1]))
    return float(right_cross - left_cross)


def peak_metrics(g: np.ndarray, curves: np.ndarray) -> dict[str, np.ndarray]:
    mask = (g >= 1.3 - 1e-10) & (g <= 2.0 + 1e-10)
    local_g = g[mask]
    local = curves[:, mask]
    index = np.argmax(local, axis=1)
    peak_value = local[np.arange(local.shape[0]), index]
    baseline = 0.5 * (local[:, 0] + local[:, -1])
    return {
        "peak_g": local_g[index],
        "peak_value_bits": peak_value,
        "distance_to_kuramoto_kc": np.abs(local_g[index] - KURAMOTO_KC),
        "prominence_bits": peak_value - baseline,
        "fwhm_g": np.asarray([fwhm(local_g, curve, int(item)) for curve, item in zip(local, index)]),
    }


def mean_sem(values: np.ndarray) -> dict[str, object]:
    valid = np.asarray(values, dtype=float)
    finite = valid[np.isfinite(valid)]
    return {"mean": float(np.mean(finite)), "sem": float(np.std(finite, ddof=1) / np.sqrt(finite.size)), "per_seed": valid.tolist()}


def load_data(pair_path: Path, full_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(pair_path) as pair_archive:
        if not np.all(pair_archive["completed"]):
            raise RuntimeError("Pairwise confirmation is incomplete.")
        g = np.asarray(pair_archive["G"], dtype=float)
        seeds = np.asarray(pair_archive["seeds"], dtype=int)
        pair_curves = np.asarray(pair_archive["pair_mean_phi_eid"], dtype=float)
    with np.load(full_path) as full_archive:
        full_g = np.asarray(full_archive["G"], dtype=float)
        full_seeds = np.asarray(full_archive["seeds"], dtype=int)
        supports = np.asarray(full_archive["supports"], dtype=float)
        horizons = np.asarray(full_archive["horizons"], dtype=int)
        support_index = int(np.flatnonzero(np.all(np.isclose(supports, (0.3, 0.7)), axis=1))[0])
        horizon_index = int(np.flatnonzero(horizons == 400)[0])
        full_curves = np.asarray(full_archive["phi_eid"], dtype=float)[support_index, :, :, horizon_index]
    if not (np.array_equal(g, full_g) and np.array_equal(seeds, full_seeds) and pair_curves.shape == full_curves.shape):
        raise RuntimeError("Pairwise and full-system experiments are not seed/G matched.")
    return g, seeds, pair_curves, full_curves


def paired_panel(axis: plt.Axes, pair_values: np.ndarray, full_values: np.ndarray, *, ylabel: str) -> None:
    positions = (0, 1)
    for first, second in zip(full_values, pair_values):
        axis.plot(positions, (first, second), color="0.78", lw=0.7, zorder=1)
    axis.scatter(np.zeros(len(full_values)), full_values, s=18, color="#6A3D9A", alpha=0.9, zorder=2, label="Full system")
    axis.scatter(np.ones(len(pair_values)), pair_values, s=18, color="#1B9E77", alpha=0.9, zorder=2, label="Pair mean")
    for x, values in zip(positions, (full_values, pair_values)):
        axis.errorbar(x, np.nanmean(values), yerr=np.nanstd(values, ddof=1) / np.sqrt(np.isfinite(values).sum()),
                      color="black", marker="_", ms=8, lw=0.8, capsize=2, zorder=3)
    axis.set_xticks(positions, ("Full\nsystem", "Pair\nmean"))
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="0.9", lw=0.6)


def plot(g: np.ndarray, pair_curves: np.ndarray, pair_metrics: dict[str, np.ndarray],
         full_metrics: dict[str, np.ndarray], output: Path) -> None:
    configure_matplotlib()
    fig = plt.figure(figsize=(8.3, 3.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(2.1, 0.9, 0.9))
    hero = fig.add_subplot(grid[0, 0])
    alignment = fig.add_subplot(grid[0, 1])
    width = fig.add_subplot(grid[0, 2])
    mean = pair_curves.mean(axis=0)
    hero.plot(g, mean, color="#1B9E77", lw=1.8, label="Mean pair PhiEID")
    hero.fill_between(g, mean - sem(pair_curves), mean + sem(pair_curves), color="#1B9E77", alpha=0.16, lw=0)
    hero.axvline(KURAMOTO_KC, color="0.35", ls=":", lw=1.0, label=r"Kuramoto $K_c$")
    hero.set_xlabel("Global coupling $G$")
    hero.set_ylabel(r"Mean pair $\Phi^{EID}$ (bits)")
    hero.grid(axis="y", color="0.9", lw=0.6)
    hero.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2, fontsize=6.8)
    hero.text(0.03, 0.05, "Mean ± seed SEM; 3403 ROI pairs per seed", transform=hero.transAxes, fontsize=6.1)
    paired_panel(alignment, pair_metrics["distance_to_kuramoto_kc"], full_metrics["distance_to_kuramoto_kc"], ylabel=r"$|G_{peak}-K_c|$")
    paired_panel(width, pair_metrics["fwhm_g"], full_metrics["fwhm_g"], ylabel="FWHM in $G$")
    for letter, axis in zip(("A", "B", "C"), (hero, alignment, width)):
        axis.text(-0.15, 1.05, letter, transform=axis.transAxes, fontsize=10, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_pair_curve(g: np.ndarray, pair_curves: np.ndarray, output: Path) -> None:
    """Standalone appendix panel: no critical-point or full-system comparison."""
    configure_matplotlib()
    mean = pair_curves.mean(axis=0)
    fig, axis = plt.subplots(figsize=(5.2, 3.25), constrained_layout=True)
    axis.plot(g, mean, color="#1B9E77", lw=1.8)
    axis.fill_between(g, mean - sem(pair_curves), mean + sem(pair_curves), color="#1B9E77", alpha=0.16, lw=0)
    axis.set_xlabel("Global coupling $G$")
    axis.set_ylabel(r"Mean pair $\Phi^{EID}$ (bits)")
    axis.grid(axis="y", color="0.9", lw=0.6)
    axis.text(0.03, 0.05, "Mean ± seed SEM; 3403 ROI pairs per seed; n=8 seeds", transform=axis.transAxes, fontsize=6.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paired all-ROI-pair PhiEID confirmation.")
    parser.add_argument("--pair-input", type=Path, default=PAIR_INPUT)
    parser.add_argument("--full-input", type=Path, default=FULL_INPUT)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    parser.add_argument("--pair-curve-figure", type=Path, default=PAIR_CURVE_FIGURE)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    g, seeds, pair_curves, full_curves = load_data(args.pair_input, args.full_input)
    pair_metrics = peak_metrics(g, pair_curves)
    full_metrics = peak_metrics(g, full_curves)
    plot(g, pair_curves, pair_metrics, full_metrics, args.figure)
    plot_pair_curve(g, pair_curves, args.pair_curve_figure)
    summary: dict[str, object] = {
        "experiment_contract": {
            "scientific_question": "Does changing only the source integration scale from all ROIs to every two-ROI mean sharpen or better align the PhiEID peak?",
            "treatment": "integration scale: all-ROI full system versus mean across all unordered two-ROI source/target systems",
            "paired_controls": "same seeds, G values, U(0.30,0.70) intervention support, tau=400, 2048 samples, direct dynamics, no clipping, calibrated J_FIC schedule, and Gaussian EI estimator",
            "pair_count": 3403,
            "seeds": seeds.tolist(),
            "G": g.tolist(),
        },
        "kuramoto_critical_coupling": KURAMOTO_KC,
        "peak_window": [1.3, 2.0],
        "metric_definition": "ROI=(sE,sI); pair PhiEID equals joint pair EI minus the sum of the two one-ROI EIs to the same future pair.",
        "pair_mean_metrics": {name: mean_sem(values) for name, values in pair_metrics.items()},
        "full_system_metrics": {name: mean_sem(values) for name, values in full_metrics.items()},
        "paired_deltas_pair_minus_full": {name: mean_sem(pair_metrics[name] - full_metrics[name]) for name in pair_metrics},
        "interpretation_boundary": "Seed-level pairing supports a comparison under this DMF intervention protocol. The all-pair average is not a substitute for a full PhiID target-side decomposition, and FWHM is only defined when a peak crosses half prominence within the fixed analysis window.",
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved figure: {args.figure}")
    print(f"Saved appendix curve: {args.pair_curve_figure}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
