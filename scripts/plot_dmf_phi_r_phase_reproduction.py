from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "exp" / "brain" / "result_lausanne_fig6" / "count_00_fig6b_mean_rate.npz"
DEFAULT_OUTPUT = ROOT / "fig" / "dmf_phi_r_phase_reproduction"


def load_sweep(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as archive:
        g_values = np.asarray(archive["G"], dtype=float)
        mean_rate = np.asarray(archive["mean_rate_hz"], dtype=float)
        phi_r = np.asarray(archive["phi_r"], dtype=float)
        critical_g = float(np.asarray(archive["critical_G"]).item())
    if not (g_values.ndim == mean_rate.ndim == phi_r.ndim == 1):
        raise ValueError("G, mean_rate_hz, and phi_r must be one-dimensional arrays.")
    if not (g_values.shape == mean_rate.shape == phi_r.shape):
        raise ValueError("G, mean_rate_hz, and phi_r must have the same shape.")
    return g_values, mean_rate, phi_r, critical_g


def plot(g_values: np.ndarray, mean_rate: np.ndarray, phi_r: np.ndarray, critical_g: float, output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    phi_peak_index = int(np.argmax(phi_r))
    phi_peak_g = float(g_values[phi_peak_index])
    phi_peak = float(phi_r[phi_peak_index])

    fig, axes = plt.subplots(2, 1, figsize=(5.6, 4.6), sharex=True, constrained_layout=True)
    critical_window = (1.7, 1.9)
    for axis in axes:
        axis.axvspan(*critical_window, color="0.92", zorder=0)
        axis.axvline(critical_g, color="0.40", ls=":", lw=1.0, zorder=1)
        axis.grid(True, color="0.86", lw=0.7, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].plot(g_values, mean_rate, color="0.28", lw=1.1, zorder=2)
    axes[0].scatter(g_values, mean_rate, color="black", s=20, zorder=3)
    axes[0].set_ylabel("Mean firing rate (Hz)")
    axes[0].annotate(
        rf"max $d\,\mathrm{{rate}}/dG$ at $G={critical_g:.1f}$",
        xy=(critical_g, np.interp(critical_g, g_values, mean_rate)),
        xytext=(10, -18),
        textcoords="offset points",
        fontsize=8,
        color="0.25",
    )

    phi_color = "#D55E00"
    axes[1].plot(g_values, phi_r, color=phi_color, lw=1.2, zorder=2)
    axes[1].scatter(g_values, phi_r, color=phi_color, s=22, zorder=3)
    axes[1].scatter([phi_peak_g], [phi_peak], color=phi_color, edgecolor="black", linewidth=0.5, s=30, zorder=4)
    axes[1].annotate(
        rf"$\Phi^R$ peak: $G={phi_peak_g:.1f}$",
        xy=(phi_peak_g, phi_peak),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8,
        color=phi_color,
    )
    axes[1].set_xlabel("Global coupling $G$")
    axes[1].set_ylabel(r"$\Phi^R$ (pair mean)")
    axes[1].set_ylim(bottom=0.0)

    for label, axis in zip(("A", "B"), axes):
        axis.text(-0.13, 1.03, label, transform=axis.transAxes, fontsize=12, fontweight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 600}), (".svg", {}), (".pdf", {})):
        fig.savefig(output.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the 83-ROI DMF PhiR phase-transition reproduction.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    g_values, mean_rate, phi_r, critical_g = load_sweep(args.source)
    plot(g_values, mean_rate, phi_r, critical_g, args.output)


if __name__ == "__main__":
    main()
