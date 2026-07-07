from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "dmf_83_region_oracle_phi_eid" / "dmf_83_region_oracle_phi_eid_curves.npz"
OUTPUT_EI_BASE = ROOT / "fig" / "dmf_83_region_oracle_ei_components_g11"
OUTPUT_DETDEG_BASE = ROOT / "fig" / "dmf_83_region_oracle_determinism_degeneracy_g11"
PLOT_MIN_G = 1.1
CRITICAL_LOW = 1.7
CRITICAL_HIGH = 1.9
SOURCE_COUNT = 83


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


def save_all(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")


def plot_mean_sem(
    axis: plt.Axes,
    x: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    label: str,
    marker: str = "o",
) -> None:
    mean = np.nanmean(values, axis=0)
    error = sem(values, axis=0)
    axis.plot(x, mean, color=color, marker=marker, linewidth=1.8, markersize=4.0, label=label)
    axis.fill_between(x, mean - error, mean + error, color=color, alpha=0.16, linewidth=0.0)


def finish_axis(axis: plt.Axes) -> None:
    axis.axvspan(CRITICAL_LOW, CRITICAL_HIGH, color="0.90", zorder=0)
    axis.grid(True, color="0.88", lw=0.8)
    axis.set_xlabel("Global coupling G")
    axis.set_xlim(PLOT_MIN_G - 0.05, 3.05)


def require_fields(cache: np.lib.npyio.NpzFile, fields: Iterable[str]) -> None:
    missing = [field for field in fields if field not in cache.files]
    if missing:
        raise ValueError(
            "Cache is missing required field(s): "
            + ", ".join(missing)
            + ". Re-run scripts/validate_dmf_83_region_oracle_phi_eid.py first."
        )


def build_components(cache: np.lib.npyio.NpzFile, *, distribution: str) -> dict[str, np.ndarray | float]:
    require_fields(cache, ("joint_ei", "singleton_sum", "target_entropy"))
    distributions = [str(item) for item in cache["distributions"]]
    if distribution not in distributions:
        raise ValueError(f"Distribution {distribution!r} not found in cache: {distributions}")
    d_index = distributions.index(distribution)
    g_values = np.asarray(cache["G"], dtype=float)
    plot_mask = g_values >= PLOT_MIN_G - 1.0e-9
    joint_ei = np.asarray(cache["joint_ei"], dtype=float)[d_index][:, plot_mask]
    singleton_sum = np.asarray(cache["singleton_sum"], dtype=float)[d_index][:, plot_mask]
    target_entropy = np.asarray(cache["target_entropy"], dtype=float)[d_index][:, plot_mask]
    target_reference_entropy = float(np.nanmax(target_entropy))
    joint_degeneracy = target_reference_entropy - target_entropy
    joint_determinism = joint_ei + joint_degeneracy
    singleton_degeneracy_sum = float(SOURCE_COUNT) * joint_degeneracy
    singleton_determinism_sum = singleton_sum + singleton_degeneracy_sum

    return {
        "G": g_values[plot_mask],
        "joint_ei": joint_ei,
        "singleton_sum": singleton_sum,
        "target_reference_entropy": target_reference_entropy,
        "joint_determinism": joint_determinism,
        "joint_degeneracy": joint_degeneracy,
        "singleton_determinism_sum": singleton_determinism_sum,
        "singleton_degeneracy_sum": singleton_degeneracy_sum,
    }


def plot_ei_components(components: dict[str, np.ndarray | float], *, distribution: str) -> None:
    x = np.asarray(components["G"], dtype=float)
    fig, axis = plt.subplots(1, 1, figsize=(5.4, 3.5), constrained_layout=True)
    finish_axis(axis)
    plot_mean_sem(
        axis,
        x,
        np.asarray(components["joint_ei"], dtype=float),
        color="#4C78A8",
        label=r"$EI(all;Y)$",
        marker="o",
    )
    plot_mean_sem(
        axis,
        x,
        np.asarray(components["singleton_sum"], dtype=float),
        color="#D98C2F",
        label=r"$\sum_i EI(i;Y)$",
        marker="D",
    )
    axis.set_ylabel("Effective information (bits)")
    axis.set_title(f"83-region oracle EI components ({distribution})", loc="left", fontweight="bold")
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save_all(fig, OUTPUT_EI_BASE)
    plt.close(fig)


def plot_determinism_degeneracy(components: dict[str, np.ndarray | float], *, distribution: str) -> None:
    x = np.asarray(components["G"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.3), constrained_layout=True, sharex=True)
    panels = (
        (axes[0, 0], "joint_determinism", r"$Det(all;Y)$", "#4C78A8", "a  Whole EI determinism"),
        (axes[0, 1], "joint_degeneracy", r"$Deg(all;Y)$", "#4C78A8", "b  Whole EI degeneracy"),
        (axes[1, 0], "singleton_determinism_sum", r"$\sum_i Det(i;Y)$", "#D98C2F", "c  Singleton-sum determinism"),
        (axes[1, 1], "singleton_degeneracy_sum", r"$\sum_i Deg(i;Y)$", "#D98C2F", "d  Singleton-sum degeneracy"),
    )
    for axis, key, label, color, title in panels:
        finish_axis(axis)
        plot_mean_sem(axis, x, np.asarray(components[key], dtype=float), color=color, label=label)
        axis.set_ylabel("Information component (bits)")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(
        f"83-region oracle determinism / degeneracy ({distribution}, "
        + rf"$H_0={float(components['target_reference_entropy']):.3f}$ bits)",
        fontsize=9,
        fontweight="bold",
    )
    save_all(fig, OUTPUT_DETDEG_BASE)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DMF 83-region oracle EI components.")
    parser.add_argument("--distribution", default="uniform", choices=("gaussian", "uniform"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    cache = np.load(CACHE, allow_pickle=True)
    components = build_components(cache, distribution=str(args.distribution))
    plot_ei_components(components, distribution=str(args.distribution))
    plot_determinism_degeneracy(components, distribution=str(args.distribution))
    print(f"Saved {OUTPUT_EI_BASE.with_suffix('.png')}")
    print(f"Saved {OUTPUT_DETDEG_BASE.with_suffix('.png')}")


if __name__ == "__main__":
    main()
