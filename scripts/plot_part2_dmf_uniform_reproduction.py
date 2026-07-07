from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_CACHE = ROOT / "results" / "dmf_module_tm_phi_eid" / "module_tm_phi_eid_curves.npz"
REGION_CACHE = ROOT / "results" / "dmf_83_region_oracle_phi_eid" / "dmf_83_region_oracle_phi_eid_curves.npz"
MODULE_SUMMARY = ROOT / "results" / "dmf_module_tm_phi_eid" / "summary.json"
REGION_SUMMARY = ROOT / "results" / "dmf_83_region_oracle_phi_eid" / "summary.json"
OUTPUT_BASE = ROOT / "fig" / "part2_dmf_phi_comparison"
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


def sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    count = np.sum(np.isfinite(values), axis=axis)
    std = np.nanstd(values, axis=axis, ddof=1 if np.nanmax(count) > 1 else 0)
    return std / np.sqrt(np.maximum(count, 1))


def uniform_slice(cache: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    distributions = [str(item) for item in cache["distributions"]]
    if "uniform" not in distributions:
        raise ValueError("Cache does not contain a uniform intervention distribution.")
    index = distributions.index("uniform")
    return np.asarray(cache["G"], dtype=float), np.asarray(cache["phi_eid"], dtype=float)[index]


def peak_values(summary_path: Path) -> np.ndarray:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return np.asarray(
        [
            float(row["peak_g"])
            for row in payload["rows"]
            if str(row["distribution"]) == "uniform"
        ],
        dtype=float,
    )


def plot() -> None:
    configure_matplotlib()
    module_cache = np.load(MODULE_CACHE, allow_pickle=True)
    region_cache = np.load(REGION_CACHE, allow_pickle=True)
    g_values, module_phi = uniform_slice(module_cache)
    region_g, region_phi = uniform_slice(region_cache)
    if not np.allclose(g_values, region_g):
        raise ValueError("Module and 83-region caches use different G grids.")

    mean_rate = np.asarray(module_cache["mean_rate_hz"], dtype=float)
    module_mean = np.nanmean(module_phi, axis=0)
    module_sem = sem(module_phi, axis=0)
    region_mean = np.nanmean(region_phi, axis=0)
    region_sem = sem(region_phi, axis=0)
    module_peaks = peak_values(MODULE_SUMMARY)
    region_peaks = peak_values(REGION_SUMMARY)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.2, 5.8),
        constrained_layout=True,
    )
    ax_rate, ax_module, ax_region, ax_peak = axes.flat

    for axis in axes.flat:
        axis.axvspan(CRITICAL_LOW, CRITICAL_HIGH, color="0.90", zorder=0)
        axis.grid(True, color="0.88", lw=0.8)

    ax_rate.plot(g_values, mean_rate, color="0.25", lw=1.5)
    ax_rate.scatter(g_values, mean_rate, color="black", s=14, zorder=2)
    ax_rate.set_ylabel("Mean firing rate (Hz)")
    ax_rate.set_xlabel("Global coupling G")

    ax_module.plot(g_values, module_mean, color="#D55E00", lw=1.6, label="Uniform intervention")
    ax_module.fill_between(
        g_values,
        module_mean - module_sem,
        module_mean + module_sem,
        color="#D55E00",
        alpha=0.18,
        lw=0.0,
    )
    ax_module.scatter([g_values[0]], [module_mean[0]], facecolor="white", edgecolor="#D55E00", s=34, zorder=3)
    ax_module.annotate(
        "boundary audit",
        xy=(g_values[0], module_mean[0]),
        xytext=(1.18, module_mean[0] * 0.88),
        arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 0.8},
        color="0.35",
        fontsize=7,
    )
    ax_module.set_ylabel(r"Module $\Phi^{EID}_{TM}$ (bits)")
    ax_module.set_xlabel("Global coupling G")
    ax_module.set_ylim(bottom=0.0)
    ax_module.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax_region.plot(g_values, region_mean, color="#0072B2", lw=1.6, label="Uniform intervention")
    ax_region.fill_between(
        g_values,
        region_mean - region_sem,
        region_mean + region_sem,
        color="#0072B2",
        alpha=0.18,
        lw=0.0,
    )
    ax_region.axhline(0.0, color="0.55", lw=0.9, ls="--")
    ax_region.set_ylabel(r"83-region signed $\Phi^{EID}$ (bits)")
    ax_region.set_xlabel("Global coupling G")
    ax_region.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    module_y = np.zeros_like(module_peaks) + np.linspace(-0.07, 0.07, module_peaks.size)
    region_y = np.ones_like(region_peaks) + np.linspace(-0.07, 0.07, region_peaks.size)
    ax_peak.scatter(
        module_peaks,
        module_y,
        color="#D55E00",
        s=28,
        edgecolor="white",
        linewidth=0.4,
        label="Module TM",
    )
    ax_peak.scatter(
        region_peaks,
        region_y,
        color="#0072B2",
        s=28,
        edgecolor="white",
        linewidth=0.4,
        label="83-region whole-state",
    )
    ax_peak.set_yticks([0.0, 1.0])
    ax_peak.set_yticklabels(["Module TM", "83-region"])
    ax_peak.set_xlabel("Identified peak G")
    ax_peak.set_xlim(float(g_values.min()) - 0.05, float(g_values.max()) + 0.05)
    ax_peak.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    for label, axis in zip(("A", "B", "C", "D"), axes.flat):
        axis.text(-0.14, 1.05, label, transform=axis.transAxes, fontsize=12, fontweight="bold")

    OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_BASE.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(OUTPUT_BASE.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT_BASE.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_BASE.with_suffix('.png')}")
    print(f"Saved {OUTPUT_BASE.with_suffix('.svg')}")
    print(f"Saved {OUTPUT_BASE.with_suffix('.pdf')}")


if __name__ == "__main__":
    plot()
