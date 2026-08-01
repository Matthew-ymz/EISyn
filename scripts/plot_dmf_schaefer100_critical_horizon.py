#!/usr/bin/env python3
"""Plot Schaefer100 DMF critical-diagnostic and multi-horizon experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CRITICAL = (
    ROOT / "results" / "dmf_schaefer100" / "critical_diagnostics" / "full" / "results.npz"
)
DEFAULT_HORIZON = (
    ROOT / "results" / "dmf_schaefer100" / "multihorizon" / "full" / "results.npz"
)
DEFAULT_OUTPUT = ROOT / "fig" / "dmf_schaefer100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critical", type=Path, default=DEFAULT_CRITICAL)
    parser.add_argument("--horizon", type=Path, default=DEFAULT_HORIZON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def sd(values: np.ndarray, axis: int = 0) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape[axis] <= 1:
        return np.zeros_like(np.mean(array, axis=axis))
    return np.std(array, axis=axis, ddof=1)


def save(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.05,
        label,
        transform=axis.transAxes,
        fontsize=8.5,
        fontweight="bold",
    )


def curve_with_sd(
    axis: plt.Axes,
    x: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    label: str | None = None,
    marker: str | None = None,
) -> None:
    mean = np.mean(values, axis=0)
    error = sd(values, axis=0)
    axis.plot(
        x,
        mean,
        color=color,
        lw=1.45,
        marker=marker,
        ms=2.2,
        label=label,
        zorder=3,
    )
    axis.fill_between(x, mean - error, mean + error, color=color, alpha=0.22, lw=0)


def plot_critical(data: dict[str, np.ndarray], output_dir: Path) -> dict[str, object]:
    g = np.asarray(data["G"], dtype=float)
    phi = np.asarray(data["phi_eid"], dtype=float)
    susceptibility = np.asarray(data["rate_susceptibility"], dtype=float)
    metastability = np.asarray(data["metastability"], dtype=float)
    jacobian = np.asarray(data["jacobian_max_real"], dtype=float)
    peak_g = g[np.argmax(phi, axis=1)]
    peak_center = float(np.median(peak_g))
    peak_low, peak_high = float(np.min(peak_g)), float(np.max(peak_g))

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.5),
        sharex=True,
        constrained_layout=True,
    )
    panels = axes.ravel()
    curve_with_sd(panels[0], g, phi, color="#6A3D9A", marker="o")
    panels[0].set_ylabel(r"Full-system $\Xi$ (bits)")
    curve_with_sd(panels[1], g, susceptibility, color="#0072B2")
    panels[1].set_ylabel(r"Rate susceptibility (Hz$^2$)")
    curve_with_sd(panels[2], g, metastability, color="#1B9E77")
    panels[2].set_ylabel("Phase metastability")
    panels[3].plot(g, jacobian, color="#D55E00", lw=1.45)
    panels[3].axhline(0.0, color="0.35", lw=0.7, ls=":")
    panels[3].set_ylabel(r"Jacobian max Re$(\lambda)$ (s$^{-1}$)")

    for label, axis in zip("ABCD", panels):
        axis.axvspan(peak_low, peak_high, color="#6A3D9A", alpha=0.08, lw=0)
        axis.axvline(peak_center, color="#6A3D9A", lw=0.8, ls="--")
        axis.grid(True, color="0.90", lw=0.5)
        axis.set_xlabel("Global coupling $G$")
        panel_label(axis, label)

    save(figure, output_dir / "dmf_schaefer100_critical_diagnostics")
    return {
        "xi_peak_G_by_seed": peak_g.tolist(),
        "xi_peak_band": [peak_low, peak_high],
        "xi_peak_median_G": peak_center,
        "susceptibility_peak_G": float(
            g[np.argmax(np.mean(susceptibility, axis=0))]
        ),
        "metastability_peak_G": float(
            g[np.argmax(np.mean(metastability, axis=0))]
        ),
        "jacobian_closest_to_zero_G": float(g[np.argmin(np.abs(jacobian))]),
    }


def plot_horizon(data: dict[str, np.ndarray], output_dir: Path) -> dict[str, object]:
    g = np.asarray(data["G"], dtype=float)
    horizons = np.asarray(data["horizons"], dtype=int)
    phi = np.asarray(data["phi_eid"], dtype=float)
    mean_phi = np.mean(phi, axis=0)
    peak_indices = np.argmax(phi, axis=1)
    peak_g = g[peak_indices]
    peak_mean = np.mean(peak_g, axis=0)
    peak_sd = sd(peak_g, axis=0)

    figure = plt.figure(figsize=(5.2, 2.8), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=(0.88, 1.12))
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])

    for seed_values in peak_g:
        ax_a.plot(
            horizons,
            seed_values,
            color="0.72",
            lw=0.6,
            alpha=0.65,
            zorder=1,
        )
    ax_a.errorbar(
        horizons,
        peak_mean,
        yerr=peak_sd,
        color="#6A3D9A",
        marker="o",
        ms=3.0,
        lw=1.35,
        capsize=2,
        zorder=3,
    )
    ax_a.set_xlabel("Target horizon (steps)")
    ax_a.set_ylabel(r"Peak coupling $G_{\mathrm{peak}}$")
    ax_a.grid(True, color="0.90", lw=0.5)
    panel_label(ax_a, "A")

    selected = [
        int(np.argmin(np.abs(horizons - target)))
        for target in (50, 100, 300, 500)
    ]
    selected = list(dict.fromkeys(selected))
    colors = ("#56B4E9", "#009E73", "#6A3D9A", "#D55E00")
    for color, horizon_index in zip(colors, selected):
        curve_with_sd(
            ax_b,
            g,
            phi[:, :, horizon_index],
            color=color,
            label=rf"$\tau={horizons[horizon_index]}$",
        )
    ax_b.set_xlabel("Global coupling $G$")
    ax_b.set_ylabel(r"Full-system $\Xi$ (bits)")
    ax_b.grid(True, color="0.90", lw=0.5)
    ax_b.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        fontsize=6.2,
    )
    panel_label(ax_b, "B")

    save(figure, output_dir / "dmf_schaefer100_multihorizon_appendix")
    return {
        "horizons": horizons.tolist(),
        "peak_G_mean": peak_mean.tolist(),
        "peak_G_sd": peak_sd.tolist(),
        "peak_G_by_seed": peak_g.tolist(),
        "maximum_mean_Xi_by_horizon": np.max(mean_phi, axis=0).tolist(),
    }


def main() -> None:
    args = parse_args()
    configure()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    critical_path = args.critical if args.critical.is_absolute() else ROOT / args.critical
    horizon_path = args.horizon if args.horizon.is_absolute() else ROOT / args.horizon
    summary = {
        "critical": plot_critical(load(critical_path), output_dir),
        "horizon": plot_horizon(load(horizon_path), output_dir),
        "uncertainty": "cross-seed SD; n=8 seeds",
        "metastability_boundary": "band-limited neural-rate phase proxy, not BOLD metastability",
    }
    (output_dir / "dmf_schaefer100_new_experiments_figure_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
