from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_dmf_fixed_uniform_multihorizon import configure_matplotlib, safe_sem, save_figure


DMF_PATH = ROOT / "results" / "dmf_fullstate_uniform_support" / "confirm_c050_h020_tau300_n2048_no_clip_seeds3_10.npz"
MEAN_RATE_PATH = ROOT / "exp" / "brain" / "result_lausanne_fig6" / "count_00_fig6b_mean_rate.npz"
KURAMOTO_PATH = ROOT / "results" / "classic_network_dynamics_benchmark" / "large_kuramoto_oracle_nsource_whole_state_tau_sweep_n64.json"
OUTPUT_DIR = ROOT / "fig"
CRITICAL_COUPLING = 1.5957691216057306
CRITICAL_LOW = 1.6
CRITICAL_HIGH = 1.8


def load_dmf() -> dict[str, np.ndarray]:
    with np.load(DMF_PATH, allow_pickle=True) as archive:
        modes = [str(item) for item in archive["modes"]]
        direct_index = modes.index("direct")
        return {
            "G": np.asarray(archive["G"], dtype=float),
            "whole_ei": np.asarray(archive["whole_ei"], dtype=float)[direct_index],
            "singleton_ei_sum": np.asarray(archive["singleton_ei_sum"], dtype=float)[direct_index],
            "phi_eid": np.asarray(archive["phi_eid"], dtype=float)[direct_index],
            "target_entropy": np.asarray(archive["target_entropy"], dtype=float)[direct_index],
            "joint_conditional_entropy": np.asarray(archive["joint_conditional_entropy"], dtype=float)[direct_index],
            "singleton_conditional_entropy_sum": np.asarray(
                archive["singleton_conditional_entropy_sum"], dtype=float,
            )[direct_index],
        }


def load_mean_rate() -> dict[str, np.ndarray]:
    """Load the dense spontaneous-DMF firing-rate sweep used as order parameter."""
    with np.load(MEAN_RATE_PATH) as archive:
        rate_g = np.asarray(archive["G"], dtype=float)
        mean_rate = np.asarray(archive["mean_rate_hz"], dtype=float)
    if rate_g.shape != mean_rate.shape or rate_g.ndim != 1:
        raise ValueError("Mean-rate G and value arrays must be matching one-dimensional curves.")
    return {"G": rate_g, "mean_rate_hz": mean_rate}


def load_kuramoto() -> dict[str, np.ndarray]:
    payload = json.loads(KURAMOTO_PATH.read_text(encoding="utf-8"))
    rows = [row for row in payload["summary"] if float(row["tau"]) == 4.0 and 1.0 <= float(row["coupling"]) <= 3.0]
    return {
        "G": np.asarray([row["coupling"] for row in rows], dtype=float),
        "whole_ei": np.asarray([row["oracle_joint_ei_mean"] for row in rows], dtype=float),
        "whole_ei_sem": np.asarray([row["oracle_joint_ei_sem"] for row in rows], dtype=float),
        "singleton_ei_sum": np.asarray([row["oracle_singleton_ei_sum_mean"] for row in rows], dtype=float),
        "singleton_ei_sum_sem": np.asarray([row["oracle_singleton_ei_sum_sem"] for row in rows], dtype=float),
        "phi_eid": np.asarray([row["oracle_phi_mean"] for row in rows], dtype=float),
        "phi_eid_sem": np.asarray([row["oracle_phi_sem"] for row in rows], dtype=float),
    }


def add_critical_annotation(axis) -> None:
    axis.axvspan(CRITICAL_LOW, CRITICAL_HIGH, color="0.92", zorder=0)
    axis.axvline(CRITICAL_COUPLING, color="0.35", lw=1.0, ls=":")


def add_mean_sem(axis, g_values: np.ndarray, values: np.ndarray, *, color: str, label: str) -> None:
    mean = np.mean(values, axis=0)
    error = safe_sem(values)
    axis.plot(g_values, mean, color=color, lw=1.9, label=label)
    axis.fill_between(g_values, mean - error, mean + error, color=color, alpha=0.16, lw=0)


def format_axis(axis, ylabel: str, panel: str) -> None:
    add_critical_annotation(axis)
    axis.grid(True, color="0.88", lw=0.8)
    axis.set_xlabel("Global coupling")
    axis.set_ylabel(ylabel)
    axis.text(-0.18, 1.05, panel, transform=axis.transAxes, fontsize=12, fontweight="bold")


def plot_dmf_confirmation(dmf: dict[str, np.ndarray], mean_rate: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.0), constrained_layout=True)
    axes[0].plot(mean_rate["G"], mean_rate["mean_rate_hz"], color="0.35", lw=1.2, zorder=1)
    axes[0].scatter(mean_rate["G"], mean_rate["mean_rate_hz"], color="black", s=15, zorder=2)
    format_axis(axes[0], "Mean firing rate (Hz)", "A")

    add_mean_sem(axes[1], dmf["G"], dmf["whole_ei"], color="#0072B2", label="Whole EI")
    add_mean_sem(
        axes[1], dmf["G"], dmf["singleton_ei_sum"], color="#D55E00", label="Sum of regional EI",
    )
    format_axis(axes[1], "Effective information (bits)", "B")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.23), ncol=2, frameon=False)

    add_mean_sem(axes[2], dmf["G"], dmf["phi_eid"], color="#6A3D9A", label=r"$\Phi^{EID}$")
    axes[2].scatter(
        dmf["G"], np.mean(dmf["phi_eid"], axis=0), color="#6A3D9A", s=18, zorder=3,
    )
    axes[2].axhline(0.0, color="0.55", lw=0.8, ls="--")
    format_axis(axes[2], r"$\Phi^{EID}$ (bits)", "C")
    save_figure(fig, OUTPUT_DIR / "dmf_fullstate_maxent_critical_confirmation")


def baseline_normalize(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) / float(np.asarray(values, dtype=float)[0])


def peak_normalize(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) / float(np.max(np.asarray(values, dtype=float)))


def plot_shape_alignment(dmf: dict[str, np.ndarray], kuramoto: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), constrained_layout=True)
    metrics = (
        ("whole_ei", "Whole EI / value at G=1", "A", "baseline"),
        ("singleton_ei_sum", "Regional-EI sum / value at G=1", "B", "baseline"),
        ("phi_eid", r"$\Phi^{EID}$ / peak", "C", "peak"),
    )
    for axis, (metric, ylabel, panel, normalization) in zip(axes, metrics):
        if normalization == "baseline":
            dmf_mean = baseline_normalize(np.mean(dmf[metric], axis=0))
            kuramoto_mean = baseline_normalize(kuramoto[metric])
        else:
            dmf_mean = peak_normalize(np.mean(dmf[metric], axis=0))
            kuramoto_mean = peak_normalize(kuramoto[metric])
        axis.plot(dmf["G"], dmf_mean, color="#0072B2", lw=1.9, label="DMF")
        axis.plot(kuramoto["G"], kuramoto_mean, color="#CC79A7", lw=1.9, label="Kuramoto")
        format_axis(axis, ylabel, panel)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save_figure(fig, OUTPUT_DIR / "dmf_kuramoto_fullstate_shape_alignment")


def plot_determinism_degeneracy(dmf: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib()
    reference_entropy = float(np.max(dmf["target_entropy"]))
    source_count = 166
    components = (
        ("Whole determinism", reference_entropy - dmf["joint_conditional_entropy"], "#0072B2", "A"),
        ("Whole degeneracy", reference_entropy - dmf["target_entropy"], "#009E73", "B"),
        (
            "Sum of regional determinism",
            source_count * reference_entropy - dmf["singleton_conditional_entropy_sum"],
            "#D55E00", "C",
        ),
        ("Sum of regional degeneracy", source_count * (reference_entropy - dmf["target_entropy"]), "#CC79A7", "D"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 5.1), constrained_layout=True, sharex=True)
    for axis, (label, values, color, panel) in zip(axes.flat, components):
        add_mean_sem(axis, dmf["G"], values, color=color, label=label)
        format_axis(axis, f"{label} (bits)", panel)
    for axis in axes[-1]:
        axis.set_xlabel("Global coupling")
    save_figure(fig, OUTPUT_DIR / "dmf_fullstate_maxent_determinism_degeneracy")


def normalize_per_seed(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    lower = np.min(array, axis=1, keepdims=True)
    span = np.max(array, axis=1, keepdims=True) - lower
    if np.any(span <= 0.0):
        raise ValueError("Each component must vary over the coupling sweep to compare rates.")
    return (array - lower) / span


def plot_integrated_determinism_degeneracy_raw(dmf: dict[str, np.ndarray]) -> None:
    """Retain the original units while combining whole and regional components."""
    import matplotlib.pyplot as plt

    configure_matplotlib()
    reference_entropy = float(np.max(dmf["target_entropy"]))
    source_count = 166
    whole_components = (
        ("Whole determinism", reference_entropy - dmf["joint_conditional_entropy"], "#0072B2", "-"),
        ("Whole degeneracy", reference_entropy - dmf["target_entropy"], "#009E73", "-"),
    )
    regional_components = (
        (
            "Regional determinism sum",
            source_count * reference_entropy - dmf["singleton_conditional_entropy_sum"],
            "#D55E00", "--",
        ),
        (
            "Regional degeneracy sum",
            source_count * (reference_entropy - dmf["target_entropy"]),
            "#CC79A7", "--",
        ),
    )
    fig, axis = plt.subplots(figsize=(7.7, 3.2), constrained_layout=True)
    right_axis = axis.twinx()
    for label, values, color, linestyle in whole_components:
        mean, error = np.mean(values, axis=0), safe_sem(values)
        axis.plot(dmf["G"], mean, color=color, lw=1.9, ls=linestyle, label=label)
        axis.fill_between(dmf["G"], mean - error, mean + error, color=color, alpha=0.13, lw=0)
    for label, values, color, linestyle in regional_components:
        mean, error = np.mean(values, axis=0), safe_sem(values)
        right_axis.plot(dmf["G"], mean, color=color, lw=1.9, ls=linestyle, label=label)
        right_axis.fill_between(dmf["G"], mean - error, mean + error, color=color, alpha=0.13, lw=0)
    add_critical_annotation(axis)
    axis.grid(True, color="0.88", lw=0.8)
    axis.set_xlabel("Global coupling")
    axis.set_ylabel("Whole components (bits)")
    right_axis.set_ylabel("Regional sums (bits)")
    handles, labels = axis.get_legend_handles_labels()
    right_handles, right_labels = right_axis.get_legend_handles_labels()
    right_axis.legend(
        handles + right_handles, labels + right_labels,
        loc="center left", bbox_to_anchor=(1.21, 0.5), frameon=False,
    )
    save_figure(fig, OUTPUT_DIR / "dmf_fullstate_maxent_detdeg_integrated_raw")


def plot_integrated_determinism_degeneracy(dmf: dict[str, np.ndarray]) -> None:
    """Overlay all components after per-seed range normalization and show their rates."""
    import matplotlib.pyplot as plt

    configure_matplotlib()
    reference_entropy = float(np.max(dmf["target_entropy"]))
    source_count = 166
    components = (
        ("Whole determinism", reference_entropy - dmf["joint_conditional_entropy"], "#0072B2", "-"),
        ("Whole degeneracy", reference_entropy - dmf["target_entropy"], "#009E73", "-"),
        (
            "Regional determinism sum",
            source_count * reference_entropy - dmf["singleton_conditional_entropy_sum"],
            "#D55E00", "--",
        ),
        (
            "Regional degeneracy sum",
            source_count * (reference_entropy - dmf["target_entropy"]),
            "#CC79A7", "--",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.1), constrained_layout=True, sharex=True)
    for label, values, color, linestyle in components:
        normalized = normalize_per_seed(values)
        rate = np.gradient(normalized, dmf["G"], axis=1)
        for axis, plotted, legend_label in (
            (axes[0], normalized, None),
            (axes[1], rate, label),
        ):
            mean, error = np.mean(plotted, axis=0), safe_sem(plotted)
            axis.plot(dmf["G"], mean, color=color, lw=1.9, ls=linestyle, label=legend_label)
            axis.fill_between(dmf["G"], mean - error, mean + error, color=color, alpha=0.13, lw=0)
    for panel, axis, ylabel in (
        ("A", axes[0], "Relative level within each component's range"),
        ("B", axes[1], "Change rate (component range / coupling)"),
    ):
        add_critical_annotation(axis)
        axis.grid(True, color="0.88", lw=0.8)
        axis.set_xlabel("Global coupling")
        axis.set_ylabel(ylabel)
        axis.text(-0.18, 1.05, panel, transform=axis.transAxes, fontsize=12, fontweight="bold")
    axes[1].axhline(0.0, color="0.55", lw=0.8, ls="--")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save_figure(fig, OUTPUT_DIR / "dmf_fullstate_maxent_detdeg_integrated_rate")


def main() -> None:
    dmf = load_dmf()
    plot_dmf_confirmation(dmf, load_mean_rate())
    plot_shape_alignment(dmf, load_kuramoto())
    plot_determinism_degeneracy(dmf)
    plot_integrated_determinism_degeneracy_raw(dmf)
    plot_integrated_determinism_degeneracy(dmf)


if __name__ == "__main__":
    main()
