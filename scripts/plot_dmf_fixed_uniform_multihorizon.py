from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "results" / "dmf_fixed_uniform_multihorizon" / "confirmatory_chunks"
DEFAULT_RESULTS = ROOT / "results" / "dmf_fixed_uniform_multihorizon" / "tau100_confirmatory_aggregate.npz"
DEFAULT_MAIN_FIGURE = ROOT / "fig" / "dmf_fixed_uniform_tau100_confirmatory"
DEFAULT_ALIGNMENT_FIGURE = ROOT / "fig" / "dmf_kuramoto_ei_alignment_tau100"
DEFAULT_DETDEG_FIGURE = ROOT / "fig" / "dmf_fixed_uniform_determinism_degeneracy"
KURAMOTO_CACHE = ROOT / "results" / "classic_network_dynamics_benchmark" / "large_kuramoto_oracle_nsource_whole_state_phi_sweep_n64.json"
METRICS = (
    "whole_ei",
    "singleton_ei_sum",
    "phi_eid",
    "target_variance_retained",
    "target_spatial_sd",
    "target_mean_offdiag_correlation",
)
DECOMPOSITION_METRICS = (
    "target_entropy",
    "joint_conditional_entropy",
    "singleton_conditional_entropy_sum",
)
CRITICAL_LOW = 1.7
CRITICAL_HIGH = 1.9
DMF_REFERENCE_TRANSITION = 1.8


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
    ddof = 1 if np.all(count > 1) else 0
    return np.nanstd(values, axis=axis, ddof=ddof) / np.sqrt(np.maximum(count, 1))


def aggregate_confirmatory_chunks(
    chunks_dir: Path,
    *,
    seeds: Iterable[int],
    g_indices: Iterable[int],
    horizon: int,
) -> dict[str, np.ndarray]:
    requested_seeds = tuple(int(seed) for seed in seeds)
    requested_indices = tuple(int(index) for index in g_indices)
    if len(set(requested_indices)) != len(requested_indices):
        raise ValueError("Requested global-coupling indices must be unique.")
    index_position = {index: position for position, index in enumerate(requested_indices)}
    shape = (len(requested_seeds), len(requested_indices))
    combined = {name: np.full(shape, np.nan, dtype=float) for name in METRICS}
    combined["G"] = np.full(len(requested_indices), np.nan, dtype=float)
    combined["clip_fraction"] = np.full(shape, np.nan, dtype=float)

    candidates: list[tuple[int, Path]] = []
    for path in chunks_dir.glob("*_g*_seed*.npz"):
        with np.load(path) as archive:
            selected = np.asarray(archive["selected_g_indices"], dtype=int)
            if selected.size and set(selected).issubset(index_position):
                candidates.append((int(selected.size), path))
    available_decomposition_metrics = tuple(
        name
        for name in DECOMPOSITION_METRICS
        if candidates and all(name in np.load(path).files for _, path in candidates)
    )
    for name in available_decomposition_metrics:
        combined[name] = np.full(shape, np.nan, dtype=float)
    for _, path in sorted(candidates, key=lambda item: (item[0], item[1].name)):
        with np.load(path) as archive:
            selected = np.asarray(archive["selected_g_indices"], dtype=int)
            archive_seeds = np.asarray(archive["seeds"], dtype=int)
            horizons = np.asarray(archive["horizons"], dtype=int)
            matching_horizon = np.flatnonzero(horizons == int(horizon))
            if archive_seeds.size != 1 or matching_horizon.size != 1:
                raise ValueError(f"{path.name} must contain exactly one seed and horizon={horizon}.")
            seed = int(archive_seeds[0])
            if seed not in requested_seeds:
                continue
            seed_position = requested_seeds.index(seed)
            horizon_position = int(matching_horizon[0])
            for local_position, index in enumerate(selected):
                if int(index) not in index_position:
                    continue
                global_position = index_position[int(index)]
                if np.isfinite(combined["whole_ei"][seed_position, global_position]):
                    continue
                for name in METRICS + available_decomposition_metrics:
                    combined[name][seed_position, global_position] = float(
                        archive[name][0, local_position, horizon_position]
                    )
                combined["clip_fraction"][seed_position, global_position] = float(
                    archive["clip_fraction"][0, local_position, horizon_position]
                )
                g_value = float(archive["G"][local_position])
                if np.isfinite(combined["G"][global_position]) and not np.isclose(
                    combined["G"][global_position], g_value
                ):
                    raise ValueError(f"Inconsistent G value for index {index}.")
                combined["G"][global_position] = g_value

    missing = np.argwhere(~np.isfinite(combined["whole_ei"]))
    if missing.size:
        locations = ", ".join(
            f"seed={requested_seeds[seed_position]}, g_index={requested_indices[g_position]}"
            for seed_position, g_position in missing
        )
        raise ValueError(f"Confirmatory chunks are missing: {locations}")
    if not np.all(np.isfinite(combined["G"])):
        raise ValueError("Confirmatory chunks are missing one or more G values.")
    combined["seeds"] = np.asarray(requested_seeds, dtype=int)
    combined["selected_g_indices"] = np.asarray(requested_indices, dtype=int)
    combined["horizon"] = np.asarray(int(horizon), dtype=int)
    target_states = set()
    sample_counts = set()
    for _, path in candidates:
        with np.load(path) as archive:
            if "target_state" in archive.files:
                target_states.add(str(np.asarray(archive["target_state"]).item()))
            if "sample_count" in archive.files:
                sample_counts.add(int(np.asarray(archive["sample_count"]).item()))
    if len(target_states) > 1:
        raise ValueError(f"Confirmatory chunks disagree on target state: {sorted(target_states)}")
    if len(sample_counts) > 1:
        raise ValueError(f"Confirmatory chunks disagree on sample count: {sorted(sample_counts)}")
    combined["target_state"] = np.asarray(next(iter(target_states), "se"))
    combined["sample_count"] = np.asarray(next(iter(sample_counts), -1), dtype=int)
    source_counts = set()
    for _, path in candidates:
        with np.load(path) as archive:
            if "source_count" in archive.files:
                source_counts.add(int(np.asarray(archive["source_count"]).item()))
    if len(source_counts) > 1:
        raise ValueError(f"Confirmatory chunks disagree on source count: {sorted(source_counts)}")
    combined["source_count"] = np.asarray(next(iter(source_counts), -1), dtype=int)
    return combined


def determinism_degeneracy_components(combined: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    required = ("target_entropy", "joint_conditional_entropy", "singleton_conditional_entropy_sum")
    missing = [name for name in required if name not in combined]
    if missing:
        raise ValueError("Missing determinism/degeneracy fields: " + ", ".join(missing))
    target_reference_entropy = float(np.nanmax(np.asarray(combined["target_entropy"], dtype=float)))
    source_count = int(np.asarray(combined["source_count"]).item())
    if source_count < 1:
        raise ValueError("A positive source_count is required for determinism/degeneracy.")
    target_entropy = np.asarray(combined["target_entropy"], dtype=float)
    joint_conditional_entropy = np.asarray(combined["joint_conditional_entropy"], dtype=float)
    singleton_conditional_entropy_sum = np.asarray(combined["singleton_conditional_entropy_sum"], dtype=float)
    joint_degeneracy = target_reference_entropy - target_entropy
    return {
        "target_reference_entropy": target_reference_entropy,
        "joint_determinism": target_reference_entropy - joint_conditional_entropy,
        "joint_degeneracy": joint_degeneracy,
        "singleton_determinism_sum": float(source_count) * target_reference_entropy - singleton_conditional_entropy_sum,
        "singleton_degeneracy_sum": float(source_count) * joint_degeneracy,
    }


def save_figure(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def add_mean_sem(
    ax: plt.Axes, g_values: np.ndarray, values: np.ndarray, *, color: str, label: str,
) -> None:
    mean = np.nanmean(values, axis=0)
    sem = safe_sem(values)
    ax.plot(g_values, mean, color=color, lw=1.8, label=label)
    ax.fill_between(g_values, mean - sem, mean + sem, color=color, alpha=0.16, lw=0.0)


def plot_confirmatory_results(combined: dict[str, np.ndarray], output: Path) -> None:
    configure_matplotlib()
    g_values = combined["G"]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.0), constrained_layout=True)
    ax_ei, ax_phi, ax_contract = axes
    for axis in axes:
        axis.axvspan(CRITICAL_LOW, CRITICAL_HIGH, color="0.90", zorder=0)
        axis.grid(True, color="0.88", lw=0.8)
        axis.set_xlabel("Global coupling G")

    add_mean_sem(ax_ei, g_values, combined["whole_ei"], color="#4C78A8", label="Whole EI")
    add_mean_sem(
        ax_ei, g_values, combined["singleton_ei_sum"], color="#D55E00", label="Sum of regional EI",
    )
    ax_ei.set_ylabel("Effective information (bits)")
    ax_ei.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    add_mean_sem(ax_phi, g_values, combined["phi_eid"], color="#6A3D9A", label=r"$\Phi^{EID}$")
    ax_phi.axhline(0.0, color="0.55", lw=0.9, ls="--")
    ax_phi.set_ylabel(r"83-source $\Phi^{EID}$ (bits)")

    add_mean_sem(
        ax_contract,
        g_values,
        combined["target_variance_retained"],
        color="#009E73",
        label="Intervention variance retained",
    )
    ax_contract.set_ylabel("Target variance / source variance")
    ax_contract.set_ylim(bottom=0.0)

    for label, axis in zip(("A", "B", "C"), axes):
        axis.text(-0.18, 1.05, label, transform=axis.transAxes, fontsize=12, fontweight="bold")
    save_figure(fig, output)


def plot_determinism_degeneracy(combined: dict[str, np.ndarray], output: Path) -> None:
    components = determinism_degeneracy_components(combined)
    configure_matplotlib()
    g_values = np.asarray(combined["G"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 5.3), constrained_layout=True, sharex=True)
    panels = (
        (axes[0, 0], "joint_determinism", "Whole determinism", "#4C78A8", "A"),
        (axes[0, 1], "joint_degeneracy", "Whole degeneracy", "#009E73", "B"),
        (axes[1, 0], "singleton_determinism_sum", "Sum of regional determinism", "#D55E00", "C"),
        (axes[1, 1], "singleton_degeneracy_sum", "Sum of regional degeneracy", "#CC79A7", "D"),
    )
    for axis, key, ylabel, color, panel_label in panels:
        axis.axvspan(CRITICAL_LOW, CRITICAL_HIGH, color="0.90", zorder=0)
        values = np.asarray(components[key], dtype=float)
        mean = np.nanmean(values, axis=0)
        error = safe_sem(values)
        axis.plot(g_values, mean, color=color, lw=1.8)
        axis.fill_between(g_values, mean - error, mean + error, color=color, alpha=0.16, lw=0.0)
        axis.grid(True, color="0.88", lw=0.8)
        axis.set_ylabel(ylabel + " (bits)")
        axis.text(-0.18, 1.05, panel_label, transform=axis.transAxes, fontsize=12, fontweight="bold")
    for axis in axes[1]:
        axis.set_xlabel("Global coupling G")
    fig.suptitle(
        rf"Fixed-reference determinism / degeneracy ($H_0={float(components['target_reference_entropy']):.2f}$ bits)",
        fontsize=9,
    )
    save_figure(fig, output)


def load_kuramoto_summary(path: Path) -> dict[str, np.ndarray | float]:
    cache = json.loads(path.read_text(encoding="utf-8"))
    rows = cache["summary"]
    return {
        "coupling": np.asarray([row["coupling"] for row in rows], dtype=float),
        "whole_ei": np.asarray([row["oracle_joint_ei_mean"] for row in rows], dtype=float),
        "singleton_ei_sum": np.asarray([row["oracle_singleton_ei_sum_mean"] for row in rows], dtype=float),
        "phi": np.asarray([row["oracle_phi_mean"] for row in rows], dtype=float),
        "critical_coupling": float(cache["critical_coupling_theory"]),
    }


def baseline_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values / values[..., :1]


def max_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    maximum = np.nanmax(values, axis=-1, keepdims=True)
    return values / maximum


def plot_shape_alignment(combined: dict[str, np.ndarray], kuramoto: dict[str, np.ndarray | float], output: Path) -> None:
    configure_matplotlib()
    dmf_x = combined["G"] / DMF_REFERENCE_TRANSITION
    kuramoto_x = np.asarray(kuramoto["coupling"], dtype=float) / float(kuramoto["critical_coupling"])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0), constrained_layout=True)
    ax_ei, ax_phi = axes
    for axis in axes:
        axis.axvspan(CRITICAL_LOW / DMF_REFERENCE_TRANSITION, CRITICAL_HIGH / DMF_REFERENCE_TRANSITION, color="0.90", zorder=0)
        axis.axvline(1.0, color="0.45", lw=0.9, ls="--", zorder=1)
        axis.grid(True, color="0.88", lw=0.8)
        axis.set_xlabel("Coupling / reference transition")

    for values, color, label in (
        (combined["whole_ei"], "#4C78A8", "Whole EI"),
        (combined["singleton_ei_sum"], "#D55E00", "Sum of regional EI"),
    ):
        normalized = baseline_normalize(values)
        ax_ei.plot(dmf_x, np.nanmean(normalized, axis=0), color=color, lw=1.8, label=f"DMF {label}")
        ax_ei.fill_between(
            dmf_x,
            np.nanmean(normalized, axis=0) - safe_sem(normalized),
            np.nanmean(normalized, axis=0) + safe_sem(normalized),
            color=color,
            alpha=0.16,
            lw=0.0,
        )
    ax_ei.plot(kuramoto_x, baseline_normalize(np.asarray(kuramoto["whole_ei"])), color="#4C78A8", lw=1.3, ls="--", label="Kuramoto whole EI")
    ax_ei.plot(kuramoto_x, baseline_normalize(np.asarray(kuramoto["singleton_ei_sum"])), color="#D55E00", lw=1.3, ls="--", label="Kuramoto sum EI")
    ax_ei.set_ylabel("EI / low-coupling EI")
    ax_ei.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    dmf_phi = max_normalize(combined["phi_eid"])
    kuramoto_phi = max_normalize(np.asarray(kuramoto["phi"]))
    ax_phi.plot(dmf_x, np.nanmean(dmf_phi, axis=0), color="#6A3D9A", lw=1.8, label=r"DMF $\Phi^{EID}$")
    ax_phi.fill_between(
        dmf_x,
        np.nanmean(dmf_phi, axis=0) - safe_sem(dmf_phi),
        np.nanmean(dmf_phi, axis=0) + safe_sem(dmf_phi),
        color="#6A3D9A",
        alpha=0.16,
        lw=0.0,
    )
    ax_phi.plot(kuramoto_x, kuramoto_phi, color="#6A3D9A", lw=1.3, ls="--", label=r"Kuramoto $\Phi$")
    ax_phi.set_ylabel("Phi / peak Phi")
    ax_phi.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    for label, axis in zip(("A", "B"), axes):
        axis.text(-0.18, 1.05, label, transform=axis.transAxes, fontsize=12, fontweight="bold")
    save_figure(fig, output)


def build_summary(combined: dict[str, np.ndarray]) -> dict[str, object]:
    mean_phi = np.nanmean(combined["phi_eid"], axis=0)
    peak_position = int(np.nanargmax(mean_phi))
    whole_mean = np.nanmean(combined["whole_ei"], axis=0)
    singleton_mean = np.nanmean(combined["singleton_ei_sum"], axis=0)
    return {
        "source": "fixed independent U(0,1)^83, no clipping",
        "horizon_steps": int(combined["horizon"]),
        "horizon_seconds": int(combined["horizon"]) * 0.001,
        "target_state": str(np.asarray(combined["target_state"]).item()),
        "sample_count": int(combined["sample_count"]),
        "seeds": [int(seed) for seed in combined["seeds"]],
        "max_clip_fraction": float(np.nanmax(combined["clip_fraction"])),
        "phi_peak_g": float(combined["G"][peak_position]),
        "phi_peak_value": float(mean_phi[peak_position]),
        "whole_ei_low_to_high_change": float(whole_mean[-1] - whole_mean[0]),
        "singleton_ei_sum_low_to_high_change": float(singleton_mean[-1] - singleton_mean[0]),
        "whole_ei_low_to_high_ratio": float(whole_mean[-1] / whole_mean[0]),
        "singleton_ei_sum_low_to_high_ratio": float(singleton_mean[-1] / singleton_mean[0]),
        "target_variance_retained_low": float(np.nanmean(combined["target_variance_retained"], axis=0)[0]),
        "target_variance_retained_high": float(np.nanmean(combined["target_variance_retained"], axis=0)[-1]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and plot confirmatory fixed-uniform DMF EI results.")
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--main-figure", type=Path, default=DEFAULT_MAIN_FIGURE)
    parser.add_argument("--alignment-figure", type=Path, default=DEFAULT_ALIGNMENT_FIGURE)
    parser.add_argument("--detdeg-figure", type=Path, default=DEFAULT_DETDEG_FIGURE)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--g-indices", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--horizon", type=int, default=100)
    return parser.parse_args()


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def main() -> None:
    args = parse_args()
    combined = aggregate_confirmatory_chunks(
        args.chunks_dir, seeds=parse_int_list(args.seeds), g_indices=parse_int_list(args.g_indices), horizon=args.horizon,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **combined)
    summary = build_summary(combined)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_confirmatory_results(combined, args.main_figure)
    plot_shape_alignment(combined, load_kuramoto_summary(KURAMOTO_CACHE), args.alignment_figure)
    if all(name in combined for name in DECOMPOSITION_METRICS):
        plot_determinism_degeneracy(combined, args.detdeg_figure)
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
