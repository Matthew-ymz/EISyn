#!/usr/bin/env python3
"""Exploratory mean-pair PhiEID sweep under the fixed DMF intervention protocol.

For every unordered ROI pair (i, j), this calculates source-side regional
synergy as EI((ROI_i, ROI_j) -> future pair) minus the two one-ROI EIs.  Each
ROI is represented by its excitatory and inhibitory state coordinates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_dmf_diffusive_fullstate_control import dmf_step_batch_with_operator
from scripts.run_dmf_fixed_uniform_multihorizon import fixed_uniform_initial_state
from scripts.validate_dmf_83_region_oracle_phi_eid import (
    DEFAULT_SOURCE_RESULTS,
    gaussian_singleton_source_phi,
    load_dmf_module,
    resolve_path,
    standardize,
)


DEFAULT_OUTPUT = ROOT / "results" / "dmf_pairwise_phi_eid_smoke" / "seed3_n512_tau400.npz"
DEFAULT_FIGURE = ROOT / "fig" / "dmf_pairwise_phi_eid_smoke.png"
KURAMOTO_KC = 1.5957691216057306


def parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in str(raw).split(",") if item.strip())


def interpolate_j_fic(source_g: np.ndarray, source_j_fic: np.ndarray, requested_g: np.ndarray) -> np.ndarray:
    if requested_g.min() < source_g.min() or requested_g.max() > source_g.max():
        raise ValueError("Requested G values must lie within the calibrated J_FIC schedule.")
    return np.column_stack([
        np.interp(requested_g, source_g, source_j_fic[:, region])
        for region in range(source_j_fic.shape[1])
    ])


def pair_phi_eid(source_z: np.ndarray, target_z: np.ndarray, *, ridge: float) -> np.ndarray:
    """Regional two-source PhiEID for all unordered ROI pairs.

    The source and target arrays are ordered as [sE_0..sE_n, sI_0..sI_n].
    A singleton source in this hierarchy is therefore one whole ROI (sE, sI),
    rather than one coordinate within an ROI.
    """
    node_count = source_z.shape[1] // 2
    if source_z.shape[1] != target_z.shape[1] or 2 * node_count != source_z.shape[1]:
        raise ValueError("Expected matching full-state arrays with even feature count.")
    left, right = np.triu_indices(node_count, k=1)
    values = np.empty(left.size, dtype=float)
    for index, (i, j) in enumerate(zip(left, right)):
        coordinates = np.asarray((i, j, node_count + i, node_count + j), dtype=int)
        source_pair = source_z[:, coordinates]
        target_pair = target_z[:, coordinates]
        joint = float(gaussian_singleton_source_phi(source_pair, target_pair, ridge=ridge)["joint_ei"])
        first = float(gaussian_singleton_source_phi(source_pair[:, (0, 2)], target_pair, ridge=ridge)["joint_ei"])
        second = float(gaussian_singleton_source_phi(source_pair[:, (1, 3)], target_pair, ridge=ridge)["joint_ei"])
        values[index] = joint - first - second
    return values


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def plot(g: np.ndarray, pair_phi: np.ndarray, output: Path) -> None:
    configure_matplotlib()
    mean = pair_phi.mean(axis=1)
    sem = pair_phi.std(axis=1, ddof=1) / np.sqrt(pair_phi.shape[1])
    q10, q90 = np.quantile(pair_phi, (0.10, 0.90), axis=1)
    peak_index = int(np.argmax(mean))
    fig, axis = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    axis.fill_between(g, q10, q90, color="#1B9E77", alpha=0.14, lw=0, label="Pair 10–90% range")
    axis.errorbar(g, mean, yerr=sem, color="#1B9E77", marker="o", ms=3.2, lw=1.5, capsize=2,
                  label="Pair mean ± descriptive SEM")
    axis.axvline(KURAMOTO_KC, color="0.35", ls=":", lw=1.0, label=r"Kuramoto $K_c$")
    axis.scatter(g[peak_index], mean[peak_index], color="#1B9E77", edgecolor="black", linewidth=0.45, s=34, zorder=4)
    axis.annotate(rf"max at $G={g[peak_index]:.2f}$", xy=(g[peak_index], mean[peak_index]),
                  xytext=(8, 10), textcoords="offset points", fontsize=7.5)
    axis.set_xlabel("Global coupling $G$")
    axis.set_ylabel(r"Mean pair $\Phi^{EID}$ (bits)")
    axis.grid(axis="y", color="0.9", lw=0.6)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.23), ncol=2, fontsize=7)
    axis.text(0.02, 0.03, "Exploratory smoke: 1 seed, 512 interventions, 3403 ROI pairs", transform=axis.transAxes, fontsize=6.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an exploratory all-ROI-pair DMF PhiEID sweep.")
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--g-values", type=parse_float_list,
                        default=(1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0))
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--support-low", type=float, default=0.3)
    parser.add_argument("--support-high", type=float, default=0.7)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.support_low < args.support_high <= 1.0:
        raise ValueError("Support must satisfy 0 <= low < high <= 1.")
    g_values = np.asarray(args.g_values, dtype=float)
    dmf = load_dmf_module()
    with np.load(resolve_path(args.source_results)) as archive:
        source_g = np.asarray(archive["G"], dtype=float)
        connectivity = np.asarray(archive["connectivity"], dtype=float)
        source_j_fic = np.asarray(archive["j_fic"], dtype=float)
    j_fic = interpolate_j_fic(source_g, source_j_fic, g_values)
    parameters = dmf.DMFParameters(t_total=1.0, burn_in=0.0, dt=float(args.dt), sigma=float(args.sigma))
    node_count = connectivity.shape[0]
    pair_count = node_count * (node_count - 1) // 2
    all_pair_phi = np.empty((g_values.size, pair_count), dtype=float)

    for g_index, coupling_g in enumerate(g_values):
        key = int(args.seed) * 1_000_000 + int(round(float(coupling_g) * 100)) * 1_000
        source_rng = np.random.default_rng(key)
        source_se, source_si = fixed_uniform_initial_state(
            source_rng, sample_count=int(args.sample_count), dimension=node_count, source_state="se_si",
            se_low=float(args.support_low), se_high=float(args.support_high),
            si_low=float(args.support_low), si_high=float(args.support_high),
        )
        if source_si is None:  # pragma: no cover
            raise RuntimeError("The full-state intervention unexpectedly omitted sI.")
        source_z, _, _ = standardize(np.concatenate((source_se, source_si), axis=1))
        noise_rng = np.random.default_rng(key + 17)
        state_se, state_si = source_se, source_si
        for _ in range(int(args.horizon)):
            state_se, state_si = dmf_step_batch_with_operator(
                dmf, state_se, state_si, connectivity=connectivity, coupling_g=float(coupling_g),
                j_fic=j_fic[g_index], parameters=parameters, mode="direct", state_boundary="none", rng=noise_rng,
            )
        target_z, _, _ = standardize(np.concatenate((state_se, state_si), axis=1))
        all_pair_phi[g_index] = pair_phi_eid(source_z, target_z, ridge=float(args.ridge))
        print(f"seed={args.seed} G={coupling_g:.2f} mean_pair_phi={all_pair_phi[g_index].mean():.8f}", flush=True)

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, G=g_values, pair_phi_eid=all_pair_phi, seed=np.asarray(int(args.seed)),
        sample_count=np.asarray(int(args.sample_count)), horizon=np.asarray(int(args.horizon)),
        support=np.asarray((args.support_low, args.support_high)), source_state=np.asarray("se_si"),
        target_state=np.asarray("se_si"), pair_definition=np.asarray("ROI=(sE,sI); joint EI minus one-ROI EI sum"),
        pair_count=np.asarray(pair_count), j_fic_interpolation=np.asarray("linear"), state_boundary=np.asarray("none"),
    )
    plot(g_values, all_pair_phi, resolve_path(args.figure))
    peak = int(np.argmax(all_pair_phi.mean(axis=1)))
    print(f"Saved {output}")
    print(f"Mean-pair peak: G={g_values[peak]:.2f}, phi={all_pair_phi[peak].mean():.8f}")


if __name__ == "__main__":
    main()
