#!/usr/bin/env python3
"""Validate Schaefer100 DMF determinism/degeneracy against Kuramoto shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DMF = ROOT / "results" / "dmf_schaefer100" / "full" / "main_confirmation.npz"
DEFAULT_KURAMOTO = (
    ROOT
    / "results"
    / "classic_network_dynamics_benchmark"
    / "n64_detdeg"
    / "large_kuramoto_oracle_nsource_whole_state_phi_sweep.json"
)
DEFAULT_RAW_FIGURE = ROOT / "fig" / "dmf_schaefer100" / "dmf_schaefer100_detdeg_appendix_raw"
DEFAULT_SHAPE_FIGURE = ROOT / "fig" / "dmf_schaefer100" / "dmf_schaefer100_detdeg_kuramoto_shape"
DEFAULT_SUMMARY = ROOT / "results" / "dmf_schaefer100" / "full" / "detdeg_appendix_summary.json"

COMPONENTS = (
    ("whole_determinism", "Whole-source determinism", "#4C78A8"),
    ("whole_degeneracy", "Whole-source degeneracy", "#009E73"),
    ("singleton_determinism_sum", "Singleton-sum determinism", "#D55E00"),
    ("singleton_degeneracy_sum", "Singleton-sum degeneracy", "#CC79A7"),
)


def configure_matplotlib() -> None:
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


def sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape[axis] <= 1:
        return np.zeros_like(np.mean(array, axis=axis))
    return np.std(array, axis=axis, ddof=1) / np.sqrt(array.shape[axis])


def save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def load_dmf(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as archive:
        modes = [str(value) for value in archive["modes"]]
        mode = modes.index("direct")
        target_entropy = np.asarray(archive["target_entropy"], dtype=float)[mode]
        whole_ei = np.asarray(archive["whole_ei"], dtype=float)[mode]
        singleton_ei_sum = np.asarray(archive["singleton_ei_sum"], dtype=float)[mode]
        source_count = int(np.asarray(archive["source_count"]).item())
        reference_entropy = float(np.max(target_entropy))
        whole_degeneracy = reference_entropy - target_entropy
        singleton_degeneracy_sum = float(source_count) * whole_degeneracy
        components = {
            "whole_determinism": whole_ei + whole_degeneracy,
            "whole_degeneracy": whole_degeneracy,
            "singleton_determinism_sum": singleton_ei_sum + singleton_degeneracy_sum,
            "singleton_degeneracy_sum": singleton_degeneracy_sum,
        }
        return {
            "coupling": np.asarray(archive["G"], dtype=float),
            "seeds": np.asarray(archive["seeds"], dtype=int),
            "reference_entropy": reference_entropy,
            "source_count": source_count,
            "whole_ei": whole_ei,
            "singleton_ei_sum": singleton_ei_sum,
            "phi_eid": np.asarray(archive["phi_eid"], dtype=float)[mode],
            "components": components,
        }


def load_kuramoto(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload["rows"])
    couplings = np.asarray(sorted({float(row["coupling"]) for row in rows}), dtype=float)
    seeds = np.asarray(sorted({int(row["seed"]) for row in rows}), dtype=int)
    row_lookup = {(int(row["seed"]), float(row["coupling"])): row for row in rows}
    key_map = {
        "whole_determinism": "oracle_joint_determinism",
        "whole_degeneracy": "oracle_joint_degeneracy",
        "singleton_determinism_sum": "oracle_singleton_determinism_sum",
        "singleton_degeneracy_sum": "oracle_singleton_degeneracy_sum",
    }
    components: dict[str, np.ndarray] = {}
    for component, row_key in key_map.items():
        components[component] = np.asarray(
            [
                [float(row_lookup[(int(seed), float(coupling))][row_key]) for coupling in couplings]
                for seed in seeds
            ],
            dtype=float,
        )
    return {
        "coupling": couplings,
        "seeds": seeds,
        "critical_coupling": float(payload["critical_coupling_theory"]),
        "components": components,
        "estimator": str(payload["estimator"]),
    }


def range_normalize_per_seed(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    lower = np.min(array, axis=1, keepdims=True)
    span = np.max(array, axis=1, keepdims=True) - lower
    if np.any(span <= 0.0):
        raise ValueError("Every component must vary across the coupling sweep.")
    return (array - lower) / span


def plot_raw_dmf(
    dmf: dict[str, Any],
    output: Path,
    *,
    rate_transition: float,
) -> None:
    configure_matplotlib()
    coupling = np.asarray(dmf["coupling"], dtype=float)
    phi_mean = np.mean(np.asarray(dmf["phi_eid"], dtype=float), axis=0)
    phi_peak = float(coupling[int(np.argmax(phi_mean))])
    figure, axes = plt.subplots(2, 2, figsize=(8.5, 5.25), constrained_layout=True, sharex=True)
    for panel, axis, (key, label, color) in zip("ABCD", axes.flat, COMPONENTS):
        values = np.asarray(dmf["components"][key], dtype=float)
        mean = np.mean(values, axis=0)
        error = sem(values, axis=0)
        axis.plot(coupling, mean, color=color, lw=1.7)
        axis.fill_between(coupling, mean - error, mean + error, color=color, alpha=0.15, lw=0)
        axis.axvline(phi_peak, color="#6A3D9A", lw=0.9, ls="--", zorder=0)
        axis.axvline(rate_transition, color="0.25", lw=0.9, ls=":", zorder=0)
        axis.set_title(label, loc="left", fontsize=7.4)
        axis.set_ylabel("Component (bits)")
        axis.grid(True, color="0.90", lw=0.55)
        axis.text(-0.14, 1.06, panel, transform=axis.transAxes, fontsize=10, fontweight="bold")
    for axis in axes[-1]:
        axis.set_xlabel("Global coupling $G$")
    figure.legend(
        handles=(
            Line2D([0], [0], color="#6A3D9A", lw=0.9, ls="--", label=rf"$\Xi$ peak ($G={phi_peak:.1f}$)"),
            Line2D([0], [0], color="0.25", lw=0.9, ls=":", label=rf"Rate transition ($G={rate_transition:.1f}$)"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=2,
    )
    save_figure(figure, output)


def plot_shape_comparison(
    dmf: dict[str, Any],
    kuramoto: dict[str, Any],
    output: Path,
    *,
    dmf_transition: float,
) -> None:
    configure_matplotlib()
    dmf_x = np.asarray(dmf["coupling"], dtype=float) / float(dmf_transition)
    kuramoto_x = np.asarray(kuramoto["coupling"], dtype=float) / float(kuramoto["critical_coupling"])
    x_max = min(float(np.max(dmf_x)), float(np.max(kuramoto_x)))
    figure, axes = plt.subplots(2, 2, figsize=(8.5, 5.25), constrained_layout=True, sharex=True, sharey=True)
    for panel, axis, (key, label, _color) in zip("ABCD", axes.flat, COMPONENTS):
        for model, x, values, color, linestyle in (
            ("Schaefer100 DMF", dmf_x, dmf["components"][key], "#6A3D9A", "-"),
            ("Kuramoto $N=64$", kuramoto_x, kuramoto["components"][key], "#4C78A8", "--"),
        ):
            normalized = range_normalize_per_seed(np.asarray(values, dtype=float))
            mean = np.mean(normalized, axis=0)
            error = sem(normalized, axis=0)
            axis.plot(x, mean, color=color, ls=linestyle, lw=1.7, label=model)
            axis.fill_between(x, mean - error, mean + error, color=color, alpha=0.13, lw=0)
        axis.axvline(1.0, color="0.35", lw=0.8, ls=":", zorder=0)
        axis.set_xlim(0.0, x_max)
        axis.set_ylim(-0.05, 1.05)
        axis.set_title(label, loc="left", fontsize=7.4)
        axis.set_ylabel("Within-sweep normalized level")
        axis.grid(True, color="0.90", lw=0.55)
        axis.text(-0.14, 1.06, panel, transform=axis.transAxes, fontsize=10, fontweight="bold")
    for axis in axes[-1]:
        axis.set_xlabel("Coupling / transition reference")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.025), ncol=2)
    save_figure(figure, output)


def analyze_shapes(
    dmf: dict[str, Any],
    kuramoto: dict[str, Any],
    *,
    dmf_transition: float,
) -> dict[str, Any]:
    dmf_x = np.asarray(dmf["coupling"], dtype=float) / float(dmf_transition)
    kuramoto_x = np.asarray(kuramoto["coupling"], dtype=float) / float(kuramoto["critical_coupling"])
    common_x = np.linspace(0.0, min(float(np.max(dmf_x)), float(np.max(kuramoto_x))), 201)
    comparisons: dict[str, Any] = {}
    for key, _label, _color in COMPONENTS:
        dmf_normalized = range_normalize_per_seed(np.asarray(dmf["components"][key], dtype=float))
        kuramoto_normalized = range_normalize_per_seed(np.asarray(kuramoto["components"][key], dtype=float))
        dmf_mean = np.mean(dmf_normalized, axis=0)
        kuramoto_mean = np.mean(kuramoto_normalized, axis=0)
        dmf_interp = np.interp(common_x, dmf_x, dmf_mean)
        kuramoto_interp = np.interp(common_x, kuramoto_x, kuramoto_mean)
        pearson = stats.pearsonr(dmf_interp, kuramoto_interp)
        spearman = stats.spearmanr(dmf_interp, kuramoto_interp)

        dmf_start = int(np.argmin(np.abs(dmf_x - 1.0)))
        dmf_end = int(np.argmin(np.abs(dmf_x - 2.0)))
        kuramoto_start = int(np.argmin(np.abs(kuramoto_x - 1.0)))
        kuramoto_end = int(np.argmin(np.abs(kuramoto_x - 2.0)))
        dmf_raw = np.asarray(dmf["components"][key], dtype=float)
        kuramoto_raw = np.asarray(kuramoto["components"][key], dtype=float)
        dmf_delta = dmf_raw[:, dmf_end] - dmf_raw[:, dmf_start]
        kuramoto_delta = kuramoto_raw[:, kuramoto_end] - kuramoto_raw[:, kuramoto_start]
        comparisons[key] = {
            "pearson_shape_r": float(pearson.statistic),
            "spearman_shape_rho": float(spearman.statistic),
            "common_relative_coupling_range": [float(common_x[0]), float(common_x[-1])],
            "dmf_post_transition_delta_bits_mean": float(np.mean(dmf_delta)),
            "dmf_post_transition_positive_seed_count": int(np.sum(dmf_delta > 0.0)),
            "dmf_seed_count": int(len(dmf_delta)),
            "kuramoto_post_transition_delta_bits_mean": float(np.mean(kuramoto_delta)),
            "kuramoto_post_transition_positive_seed_count": int(np.sum(kuramoto_delta > 0.0)),
            "kuramoto_seed_count": int(len(kuramoto_delta)),
        }
    return comparisons


def write_summary(
    dmf: dict[str, Any],
    kuramoto: dict[str, Any],
    path: Path,
    *,
    dmf_transition: float,
    raw_figure: Path,
    shape_figure: Path,
) -> None:
    coupling = np.asarray(dmf["coupling"], dtype=float)
    phi_mean = np.mean(np.asarray(dmf["phi_eid"], dtype=float), axis=0)
    component_extrema: dict[str, Any] = {}
    for key, _label, _color in COMPONENTS:
        mean = np.mean(np.asarray(dmf["components"][key], dtype=float), axis=0)
        component_extrema[key] = {
            "minimum_G": float(coupling[int(np.argmin(mean))]),
            "minimum_bits": float(np.min(mean)),
            "maximum_G": float(coupling[int(np.argmax(mean))]),
            "maximum_bits": float(np.max(mean)),
        }
    payload = {
        "experiment_contract": {
            "question": "Do Schaefer100 DMF whole/singleton-sum determinism and degeneracy follow the same coupling-dependent shapes as the previous N=64 Kuramoto experiment?",
            "dmf_fixed": {
                "seeds": np.asarray(dmf["seeds"], dtype=int).tolist(),
                "G": coupling.tolist(),
                "source_count": int(dmf["source_count"]),
                "source_support": "independent U(0.30, 0.70)^200 full E/I state",
                "target": "full 200-dimensional E/I state at 300 integration steps",
                "estimator": "Gaussian log-determinant EI with ridge 1e-6",
            },
            "cross_model_boundary": "Shape-only exploratory comparison; equations, state spaces, estimators, source counts, horizons, and coupling units differ.",
        },
        "definition": {
            "reference_entropy": "H0 = maximum target entropy over each model sweep",
            "degeneracy": "Deg(S;Y) = H0 - H(Y)",
            "determinism": "Det(S;Y) = EI(S;Y) + Deg(S;Y), preserving EI = Det - Deg exactly",
        },
        "dmf": {
            "reference_entropy_bits": float(dmf["reference_entropy"]),
            "phi_peak_G": float(coupling[int(np.argmax(phi_mean))]),
            "rate_transition_G": float(dmf_transition),
            "component_extrema": component_extrema,
        },
        "kuramoto": {
            "critical_coupling": float(kuramoto["critical_coupling"]),
            "seeds": np.asarray(kuramoto["seeds"], dtype=int).tolist(),
            "estimator": str(kuramoto["estimator"]),
        },
        "shape_comparison": analyze_shapes(dmf, kuramoto, dmf_transition=dmf_transition),
        "conclusion": (
            "The Schaefer100 DMF decomposition does not reproduce the Kuramoto determinism/degeneracy shape. "
            "All four DMF components collapse from the rate-transition point to twice that coupling, whereas "
            "all four Kuramoto components increase over the analogous relative-coupling interval."
        ),
        "figures": {
            "raw_dmf": str(raw_figure),
            "shape_comparison": str(shape_figure),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dmf", type=Path, default=DEFAULT_DMF)
    parser.add_argument("--kuramoto", type=Path, default=DEFAULT_KURAMOTO)
    parser.add_argument("--raw-figure", type=Path, default=DEFAULT_RAW_FIGURE)
    parser.add_argument("--shape-figure", type=Path, default=DEFAULT_SHAPE_FIGURE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--dmf-transition", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dmf = load_dmf(args.dmf)
    kuramoto = load_kuramoto(args.kuramoto)
    plot_raw_dmf(dmf, args.raw_figure, rate_transition=float(args.dmf_transition))
    plot_shape_comparison(
        dmf,
        kuramoto,
        args.shape_figure,
        dmf_transition=float(args.dmf_transition),
    )
    write_summary(
        dmf,
        kuramoto,
        args.summary,
        dmf_transition=float(args.dmf_transition),
        raw_figure=args.raw_figure,
        shape_figure=args.shape_figure,
    )
    print(f"Saved raw figure: {args.raw_figure}")
    print(f"Saved shape figure: {args.shape_figure}")
    print(f"Saved summary: {args.summary}")


if __name__ == "__main__":
    main()
