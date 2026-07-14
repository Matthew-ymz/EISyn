from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_dmf_fixed_uniform_multihorizon import configure_matplotlib, safe_sem, save_figure


OUTPUT_DIR = ROOT / "fig"
RESULTS_DIR = ROOT / "results" / "dmf_fullstate_uniform_support"
RAW_FULL_SUPPORT = ROOT / "results" / "dmf_diffusive_coupling_control" / "smoke_raw_sc_seed0.npz"


def direct_metrics(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        modes = [str(item) for item in archive["modes"]]
        index = modes.index("direct")
        metrics = {
            name: np.asarray(archive[name], dtype=float)[index]
            for name in (
                "whole_ei", "singleton_ei_sum", "phi_eid", "target_entropy",
                "joint_conditional_entropy", "singleton_conditional_entropy_sum",
            )
        }
        return {"G": np.asarray(archive["G"], dtype=float), **metrics}


def combine(*datasets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    reference = datasets[0]
    combined = {"G": reference["G"]}
    for name in reference:
        if name == "G":
            continue
        combined[name] = np.concatenate([dataset[name] for dataset in datasets], axis=0)
    return combined


def plot_support_sweep(supports: dict[float, dict[str, np.ndarray]]) -> None:
    configure_matplotlib()
    colors = {0.20: "#0072B2", 0.35: "#E69F00", 0.50: "#D55E00"}
    fig, axes = __import__("matplotlib.pyplot", fromlist=["plt"]).subplots(
        1, 3, figsize=(10.4, 3.0), constrained_layout=True,
    )
    metric_panels = (
        ("whole_ei", "Whole EI (bits)", "A"),
        ("singleton_ei_sum", "Sum of regional EI (bits)", "B"),
        ("phi_eid", r"$\Phi^{EID}$ (bits)", "C"),
    )
    for axis, (metric, ylabel, panel) in zip(axes, metric_panels):
        for half_width, data in supports.items():
            axis.plot(data["G"], np.mean(data[metric], axis=0), lw=1.9, color=colors[half_width],
                      label=rf"$h={half_width:.2f}$")
        axis.grid(True, color="0.88", lw=0.8)
        axis.set_xlabel("Global coupling G")
        axis.set_ylabel(ylabel)
        axis.text(-0.18, 1.05, panel, transform=axis.transAxes, fontsize=12, fontweight="bold")
    axes[-1].axhline(0.0, color="0.55", lw=0.8, ls="--")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
                    title="Uniform support")
    save_figure(fig, OUTPUT_DIR / "dmf_fullstate_uniform_support_sweep")


def plot_confirmation(data: dict[str, np.ndarray]) -> None:
    configure_matplotlib()
    plt = __import__("matplotlib.pyplot", fromlist=["plt"])
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0), constrained_layout=True)
    g_values = data["G"]
    for metric, color, label in (
        ("whole_ei", "#0072B2", "Whole EI"),
        ("singleton_ei_sum", "#D55E00", "Sum of regional EI"),
    ):
        mean, error = np.mean(data[metric], axis=0), safe_sem(data[metric])
        axes[0].plot(g_values, mean, color=color, lw=1.9, label=label)
        axes[0].fill_between(g_values, mean - error, mean + error, color=color, alpha=0.16, lw=0)
    mean, error = np.mean(data["phi_eid"], axis=0), safe_sem(data["phi_eid"])
    axes[1].plot(g_values, mean, color="#6A3D9A", lw=1.9, label=r"$\Phi^{EID}$")
    axes[1].fill_between(g_values, mean - error, mean + error, color="#6A3D9A", alpha=0.16, lw=0)
    axes[1].axhline(0.0, color="0.55", lw=0.8, ls="--")
    for panel, axis, ylabel in zip(("A", "B"), axes, ("Effective information (bits)", r"$\Phi^{EID}$ (bits)")):
        axis.grid(True, color="0.88", lw=0.8)
        axis.set_xlabel("Global coupling G")
        axis.set_ylabel(ylabel)
        axis.text(-0.18, 1.05, panel, transform=axis.transAxes, fontsize=12, fontweight="bold")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save_figure(fig, OUTPUT_DIR / "dmf_fullstate_uniform_h020_confirmation")


def plot_determinism_degeneracy(data: dict[str, np.ndarray]) -> None:
    configure_matplotlib()
    plt = __import__("matplotlib.pyplot", fromlist=["plt"])
    h0 = float(np.max(data["target_entropy"]))
    source_count = 166
    components = {
        "Whole determinism": h0 - data["joint_conditional_entropy"],
        "Whole degeneracy": h0 - data["target_entropy"],
        "Sum of regional determinism": source_count * h0 - data["singleton_conditional_entropy_sum"],
        "Sum of regional degeneracy": source_count * (h0 - data["target_entropy"]),
    }
    colors = ("#0072B2", "#009E73", "#D55E00", "#CC79A7")
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 5.1), constrained_layout=True, sharex=True)
    for panel, axis, (label, values), color in zip("ABCD", axes.flat, components.items(), colors):
        mean, error = np.mean(values, axis=0), safe_sem(values)
        axis.plot(data["G"], mean, color=color, lw=1.8)
        axis.fill_between(data["G"], mean - error, mean + error, color=color, alpha=0.16, lw=0)
        axis.grid(True, color="0.88", lw=0.8)
        axis.set_ylabel(label + " (bits)")
        axis.text(-0.18, 1.05, panel, transform=axis.transAxes, fontsize=12, fontweight="bold")
    for axis in axes[-1]:
        axis.set_xlabel("Global coupling G")
    save_figure(fig, OUTPUT_DIR / "dmf_fullstate_uniform_h020_determinism_degeneracy")


def main() -> None:
    h020 = combine(
        direct_metrics(RESULTS_DIR / "smoke_h020_seed0.npz"),
        direct_metrics(RESULTS_DIR / "confirm_h020_seeds1_2.npz"),
    )
    supports = {
        0.20: h020,
        0.35: direct_metrics(RESULTS_DIR / "smoke_h035_seed0.npz"),
        0.50: direct_metrics(RAW_FULL_SUPPORT),
    }
    plot_support_sweep(supports)
    plot_confirmation(h020)
    plot_determinism_degeneracy(h020)


if __name__ == "__main__":
    main()
