from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "dmf_83_region_oracle_phi_eid" / "dmf_83_region_oracle_phi_eid_curves.npz"
OUTPUT_BASE = ROOT / "fig" / "dmf_83_region_oracle_phi_eid_main_g11"
PLOT_MIN_G = 1.1
CRITICAL_LOW = 1.7
CRITICAL_HIGH = 1.9


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def safe_sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    count = np.sum(np.isfinite(values), axis=axis)
    std = np.nanstd(values, axis=axis, ddof=1 if np.nanmax(count) > 1 else 0)
    return std / np.sqrt(np.maximum(count, 1))


def plot_results(
    *,
    g_values: np.ndarray,
    mean_rate_hz: np.ndarray,
    phi: np.ndarray,
    joint_ei: np.ndarray,
    singleton_sum: np.ndarray,
    distribution_names: Sequence[str],
) -> None:
    configure_matplotlib()
    colors = {"uniform": "#D55E00"}
    plot_mask = g_values >= PLOT_MIN_G - 1.0e-9
    plot_g = g_values[plot_mask]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), constrained_layout=True)
    ax_rate, ax_phi, ax_ei = axes

    ax_rate.axvspan(CRITICAL_LOW, CRITICAL_HIGH, color="0.88", zorder=0)
    ax_rate.plot(plot_g, mean_rate_hz[plot_mask], color="0.25", lw=1.4)
    ax_rate.scatter(plot_g, mean_rate_hz[plot_mask], color="black", s=14)
    ax_rate.set_xlabel("Global coupling G")
    ax_rate.set_ylabel("Mean firing rate (Hz)")
    ax_rate.grid(True, color="0.88", lw=0.8)

    for d_index, distribution in enumerate(distribution_names):
        values = phi[d_index][:, plot_mask]
        mean = np.nanmean(values, axis=0)
        sem = safe_sem(values, axis=0)
        color = colors.get(distribution, "0.25")
        ax_phi.plot(plot_g, mean, color=color, lw=1.8)
        ax_phi.fill_between(plot_g, mean - sem, mean + sem, color=color, alpha=0.18, lw=0.0)
    ax_phi.axhline(0.0, color="0.55", lw=0.9, ls="--")
    ax_phi.axvspan(CRITICAL_LOW, CRITICAL_HIGH, color="0.90", zorder=0)
    ax_phi.set_xlabel("Global coupling G")
    ax_phi.set_ylabel(r"83-region $\Phi^{EID}$ (bits)")
    ax_phi.grid(True, color="0.88", lw=0.8)

    for values, color, label in (
        (joint_ei[:, plot_mask], "#4C78A8", "Whole EI"),
        (singleton_sum[:, plot_mask], "#D98C2F", "Sum of regional EI"),
    ):
        mean = np.nanmean(values, axis=0)
        sem = safe_sem(values, axis=0)
        ax_ei.plot(plot_g, mean, color=color, lw=1.8, label=label)
        ax_ei.fill_between(plot_g, mean - sem, mean + sem, color=color, alpha=0.16, lw=0.0)
    ax_ei.axvspan(CRITICAL_LOW, CRITICAL_HIGH, color="0.90", zorder=0)
    ax_ei.set_xlabel("Global coupling G")
    ax_ei.set_ylabel("Effective information (bits)")
    ax_ei.grid(True, color="0.88", lw=0.8)
    ax_ei.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    for label, axis in zip(("A", "B", "C"), axes):
        axis.text(-0.18, 1.05, label, transform=axis.transAxes, fontsize=12, fontweight="bold")

    OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_BASE.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(OUTPUT_BASE.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT_BASE.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cache = np.load(CACHE, allow_pickle=True)
    all_distributions = [str(item) for item in cache["distributions"]]
    uniform_index = all_distributions.index("uniform")
    plot_results(
        g_values=np.asarray(cache["G"], dtype=float),
        mean_rate_hz=np.asarray(cache["mean_rate_hz"], dtype=float),
        phi=np.asarray(cache["phi_eid"], dtype=float)[uniform_index : uniform_index + 1],
        joint_ei=np.asarray(cache["joint_ei"], dtype=float)[uniform_index],
        singleton_sum=np.asarray(cache["singleton_sum"], dtype=float)[uniform_index],
        distribution_names=["uniform"],
    )
    print(f"Saved {OUTPUT_BASE.with_suffix('.png')}")
    print(f"Saved {OUTPUT_BASE.with_suffix('.svg')}")
    print(f"Saved {OUTPUT_BASE.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
