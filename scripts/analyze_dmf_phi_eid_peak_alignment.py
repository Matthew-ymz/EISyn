#!/usr/bin/env python3
"""Summarize the paired DMF PhiEID peak-alignment confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_dmf_83_region_oracle_phi_eid import resolve_path


DEFAULT_INPUT = ROOT / "results" / "dmf_phi_eid_peak_alignment" / "paired_support_horizon.npz"
DEFAULT_FIGURE = ROOT / "fig" / "dmf_phi_eid_peak_alignment.png"
DEFAULT_SUMMARY = ROOT / "results" / "dmf_phi_eid_peak_alignment" / "summary.json"
KURAMOTO_KC = 1.5957691216057306
COLORS = ("#6A3D9A", "#1B9E77", "#E66101", "#377EB8")


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.std(array, axis=axis, ddof=1) / np.sqrt(array.shape[axis])


def configuration_names(supports: np.ndarray, horizons: np.ndarray) -> list[str]:
    return [
        rf"$U({low:.2f},{high:.2f})$, $\tau={int(horizon)}$"
        for low, high in supports for horizon in horizons
    ]


def full_width_half_prominence(g: np.ndarray, values: np.ndarray, peak_index: int) -> float:
    """FWHM above the mean end-point baseline, with linear crossings."""
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
    if left == peak_index or right == peak_index or values[left] == values[left + 1] or values[right] == values[right - 1]:
        return float("nan")
    left_cross = np.interp(level, [values[left], values[left + 1]], [g[left], g[left + 1]])
    right_cross = np.interp(level, [values[right], values[right - 1]], [g[right], g[right - 1]])
    return float(right_cross - left_cross)


def peak_metrics(g: np.ndarray, values: np.ndarray) -> dict[str, np.ndarray]:
    critical = (g >= 1.3 - 1e-9) & (g <= 2.0 + 1e-9)
    g_window = g[critical]
    curves = np.asarray(values, dtype=float)[:, critical]
    peak_position = np.argmax(curves, axis=1)
    peak_g = g_window[peak_position]
    peak_value = curves[np.arange(curves.shape[0]), peak_position]
    endpoint_baseline = 0.5 * (curves[:, 0] + curves[:, -1])
    prominence = peak_value - endpoint_baseline
    width = np.asarray([full_width_half_prominence(g_window, curve, int(index)) for curve, index in zip(curves, peak_position)])
    spacing = float(np.median(np.diff(g_window)))
    curvature = np.full(curves.shape[0], np.nan, dtype=float)
    valid = (peak_position > 0) & (peak_position < len(g_window) - 1)
    position = peak_position[valid]
    curvature[valid] = -(
        curves[np.flatnonzero(valid), position - 1]
        - 2.0 * curves[np.flatnonzero(valid), position]
        + curves[np.flatnonzero(valid), position + 1]
    ) / spacing**2
    return {
        "peak_g": peak_g,
        "peak_value_bits": peak_value,
        "prominence_bits": prominence,
        "fwhm_g": width,
        "local_negative_second_derivative_bits_per_g2": curvature,
        "distance_to_kuramoto_kc": np.abs(peak_g - KURAMOTO_KC),
    }


def load_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        if not np.all(np.asarray(archive["completed"], dtype=bool)):
            raise RuntimeError("The paired confirmation is incomplete; do not summarize partial curves as a result.")
        return {name: np.asarray(archive[name]) for name in archive.files}


def plot(payload: dict[str, np.ndarray], output: Path, metrics: list[dict[str, np.ndarray]]) -> None:
    configure_matplotlib()
    g = np.asarray(payload["G"], dtype=float)
    supports = np.asarray(payload["supports"], dtype=float)
    horizons = np.asarray(payload["horizons"], dtype=int)
    phi = np.asarray(payload["phi_eid"], dtype=float)
    whole = np.asarray(payload["whole_ei"], dtype=float)
    singleton = np.asarray(payload["singleton_ei_sum"], dtype=float)
    labels = configuration_names(supports, horizons)
    fig = plt.figure(figsize=(10.6, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.55, 0.95, 1.05))
    hero = fig.add_subplot(grid[:, 0])
    peak_axis = fig.add_subplot(grid[0, 1])
    width_axis = fig.add_subplot(grid[1, 1])
    mechanism = fig.add_subplot(grid[:, 2])

    index = 0
    for support_index in range(len(supports)):
        for horizon_index in range(len(horizons)):
            curves = phi[support_index, :, :, horizon_index]
            color = COLORS[index]
            mean = curves.mean(axis=0)
            hero.plot(g, mean, color=color, lw=1.8, label=labels[index])
            hero.fill_between(g, mean - sem(curves), mean + sem(curves), color=color, alpha=0.14, lw=0)
            index += 1
    hero.axvline(KURAMOTO_KC, color="0.3", ls=":", lw=1.0, label=r"Kuramoto $K_c$")
    hero.set_xlabel("Global coupling $G$")
    hero.set_ylabel(r"$\Phi^{EID}$ (bits)")
    hero.set_xlim(0.97, 3.03)
    hero.grid(axis="y", color="0.9", lw=0.6)
    hero.legend(loc="upper center", bbox_to_anchor=(0.5, 1.19), ncol=2, fontsize=6.4)

    positions = np.arange(len(metrics))
    for index, summary in enumerate(metrics):
        distances = np.asarray(summary["distance_to_kuramoto_kc"])
        peak_axis.scatter(np.full(len(distances), index), distances, color=COLORS[index], s=14, alpha=0.8, zorder=3)
        peak_axis.errorbar(index, distances.mean(), yerr=sem(distances), color="black", marker="_", capsize=2, lw=0.8, zorder=4)
        widths = np.asarray(summary["fwhm_g"])
        width_axis.scatter(np.full(np.isfinite(widths).sum(), index), widths[np.isfinite(widths)], color=COLORS[index], s=14, alpha=0.8, zorder=3)
        if np.isfinite(widths).any():
            width_axis.errorbar(index, np.nanmean(widths), yerr=np.nanstd(widths, ddof=1) / np.sqrt(np.isfinite(widths).sum()), color="black", marker="_", capsize=2, lw=0.8, zorder=4)
    for axis, ylabel in ((peak_axis, r"$|G_{peak}-K_c|$"), (width_axis, "FWHM in $G$")):
        axis.set_xticks(positions, [f"S{support + 1}\nT{horizon}" for support in range(len(supports)) for horizon in horizons])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="0.9", lw=0.6)

    # Mechanism audit uses the condition with the lowest mean peak-to-Kc distance;
    # selection is deterministic and reported in the JSON rather than hand-picked.
    best_index = int(np.argmin([np.mean(item["distance_to_kuramoto_kc"]) for item in metrics]))
    support_index, horizon_index = divmod(best_index, len(horizons))
    selected = (support_index, slice(None), slice(None), horizon_index)
    for values, color, label in (
        (whole[selected], "#0072B2", "Whole EI"),
        (singleton[selected], "#D55E00", "Singleton-EI sum"),
        (phi[selected], COLORS[best_index], r"$\Phi^{EID}$"),
    ):
        mean = values.mean(axis=0)
        mechanism.plot(g, mean, color=color, lw=1.7, label=label)
        mechanism.fill_between(g, mean - sem(values), mean + sem(values), color=color, alpha=0.12, lw=0)
    mechanism.axvline(KURAMOTO_KC, color="0.3", ls=":", lw=1.0)
    mechanism.set_xlabel("Global coupling $G$")
    mechanism.set_ylabel("Information (bits)")
    mechanism.grid(axis="y", color="0.9", lw=0.6)
    mechanism.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=1, fontsize=6.4)
    mechanism.text(0.03, 0.04, f"Mechanism audit: S{support_index + 1}, T{horizons[horizon_index]}", transform=mechanism.transAxes, fontsize=6.1)

    for letter, axis in zip(("A", "B", "C", "D"), (hero, peak_axis, width_axis, mechanism)):
        axis.text(-0.16, 1.05, letter, transform=axis.transAxes, fontsize=10, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_summary(payload: dict[str, np.ndarray], metrics: list[dict[str, np.ndarray]]) -> dict[str, object]:
    supports = np.asarray(payload["supports"], dtype=float)
    horizons = np.asarray(payload["horizons"], dtype=int)
    rows = []
    index = 0
    for support_index, (low, high) in enumerate(supports):
        for horizon_index, horizon in enumerate(horizons):
            record: dict[str, object] = {
                "support": [float(low), float(high)],
                "horizon_steps": int(horizon),
            }
            for name, values in metrics[index].items():
                finite = np.asarray(values, dtype=float)
                record[name] = {
                    "mean": float(np.nanmean(finite)),
                    "sem": float(np.nanstd(finite, ddof=1) / np.sqrt(np.isfinite(finite).sum())),
                    "per_seed": finite.tolist(),
                }
            rows.append(record)
            index += 1
    best = min(rows, key=lambda row: float(row["distance_to_kuramoto_kc"]["mean"]))
    return {
        "experiment_contract": {
            "treatments": "fixed intervention support and prediction horizon only",
            "paired_controls": "same seeds, G values, underlying uniform draws, dynamics, noise stream, Gaussian estimator, and no state clipping",
            "samples_per_condition": int(np.asarray(payload["sample_count"]).item()),
            "seeds": np.asarray(payload["seeds"], dtype=int).tolist(),
            "G": np.asarray(payload["G"], dtype=float).tolist(),
            "j_fic_schedule": "linear interpolation of the pre-calibrated 0.1-G schedule onto the 0.05-G analysis grid",
        },
        "kuramoto_critical_coupling": KURAMOTO_KC,
        "peak_window": [1.3, 2.0],
        "peak_metric_definition": "per-seed argmax in the dense G window; FWHM is defined above the mean of the two window endpoints",
        "configurations": rows,
        "best_mean_peak_alignment": best,
        "interpretation_boundary": "The best configuration is selected by mean peak-to-Kuramoto distance, not visual appearance. A sharper or better-aligned peak does not establish that it is universally optimal outside this DMF intervention protocol.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paired DMF PhiEID peak-alignment results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_payload(resolve_path(args.input))
    metric_rows = []
    for support_index in range(len(payload["supports"])):
        for horizon_index in range(len(payload["horizons"])):
            metric_rows.append(peak_metrics(payload["G"], payload["phi_eid"][support_index, :, :, horizon_index]))
    plot(payload, resolve_path(args.figure), metric_rows)
    summary = make_summary(payload, metric_rows)
    summary_path = resolve_path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved figure: {resolve_path(args.figure)}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
