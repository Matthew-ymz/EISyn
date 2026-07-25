"""Recover planted dynamical modules in a modular Kuramoto network.

Six heterogeneous oscillators are arranged as two densely coupled communities.
Independent uniform phase interventions are propagated through the known
Kuramoto vector field.  Degree-2 transport-map EI is then estimated for every
non-empty oscillator subset and decomposed with the shared greedy hierarchy.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.TM.transport_map_density import LOG_2, fit_polynomial_triangular_transport_map_density
from scripts.phi_hierarchy import (
    NONNEGATIVE_TOLERANT,
    PhiAtom,
    all_nonempty_subsets,
    greedy_phi_atoms,
    nontrivial_bipartitions,
    subset_phi_raw,
)
from yrd import clip_nonnegative_ei


NAMES = tuple(f"theta{i}" for i in range(1, 7))
PLANTED_MODULES = (NAMES[:3], NAMES[3:])
FREQUENCIES = np.array([-0.55, -0.05, 0.42, -0.40, 0.12, 0.51], dtype=float)
DEFAULT_RESULT = ROOT / "results" / "greedy_hierarchy_kuramoto" / "summary.json"
DEFAULT_FIGURE = ROOT / "docs" / "ref" / "assets" / "greedy_hierarchy_kuramoto" / "validation"


def coupling_matrix(*, within_coupling: float, cross_coupling: float) -> np.ndarray:
    """Return a symmetric two-community coupling matrix with normalized row budgets."""
    matrix = np.zeros((6, 6), dtype=float)
    matrix[:3, :3] = float(within_coupling) / 2.0
    matrix[3:, 3:] = float(within_coupling) / 2.0
    matrix[:3, 3:] = float(cross_coupling) / 3.0
    matrix[3:, :3] = float(cross_coupling) / 3.0
    np.fill_diagonal(matrix, 0.0)
    return matrix


def kuramoto_derivative(
    phases: np.ndarray,
    *,
    within_coupling: float,
    cross_coupling: float,
    frequencies: np.ndarray = FREQUENCIES,
) -> np.ndarray:
    """Evaluate the known modular Kuramoto vector field for batched phases."""
    values = np.asarray(phases, dtype=float)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("phases must have shape (n_samples, 6).")
    weights = coupling_matrix(within_coupling=within_coupling, cross_coupling=cross_coupling)
    phase_difference = values[:, None, :] - values[:, :, None]
    interaction = np.sum(weights[None, :, :] * np.sin(phase_difference), axis=2)
    return np.asarray(frequencies, dtype=float).reshape(1, 6) + interaction


def paired_data(
    *,
    sample_count: int,
    seed: int,
    within_coupling: float,
    cross_coupling: float,
    noise_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate paired intervention data; phases/noise are shared across coupling levels."""
    rng = np.random.default_rng(int(seed))
    phases = rng.uniform(-math.pi, math.pi, size=(int(sample_count), 6))
    standardized_noise = rng.normal(size=(int(sample_count), 6))
    target = kuramoto_derivative(
        phases,
        within_coupling=within_coupling,
        cross_coupling=cross_coupling,
    ) + float(noise_scale) * standardized_noise
    return phases, target


def phase_source_blocks(phases: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(phases, dtype=float)
    return {
        name: np.column_stack((np.cos(values[:, index]), np.sin(values[:, index])))
        for index, name in enumerate(NAMES)
    }


def transport_map_ei_table(
    phases: np.ndarray,
    target: np.ndarray,
    *,
    degree: int,
) -> dict[tuple[str, ...], float]:
    """Estimate EI for all 63 source subsets with a shared target density."""
    blocks = phase_source_blocks(phases)
    target_array = np.asarray(target, dtype=float)
    target_model = fit_polynomial_triangular_transport_map_density(target_array, degree=int(degree))
    target_log_prob = target_model.log_prob(target_array)
    table: dict[tuple[str, ...], float] = {}
    for subset in all_nonempty_subsets(NAMES):
        source = np.column_stack([blocks[name] for name in subset])
        joint = np.column_stack((target_array, source))
        joint_model = fit_polynomial_triangular_transport_map_density(joint, degree=int(degree))
        source_model = fit_polynomial_triangular_transport_map_density(source, degree=int(degree))
        pointwise = (
            joint_model.log_prob(joint)
            - source_model.log_prob(source)
            - target_log_prob
        ) / LOG_2
        table[subset] = float(clip_nonnegative_ei(float(np.mean(pointwise))))
    return table


def best_root_split(
    table: Mapping[tuple[str, ...], float],
    *,
    split_tolerance: float = 0.10,
) -> tuple[tuple[str, ...], tuple[str, ...], float, float]:
    """Return the same maximum-captured-Phi root split used by the hierarchy."""
    singleton = {name: float(table[(name,)]) for name in NAMES}
    root_phi = subset_phi_raw(NAMES, table, singleton)
    candidates = []
    for left, right in nontrivial_bipartitions(NAMES):
        left_phi = subset_phi_raw(left, table, singleton)
        right_phi = subset_phi_raw(right, table, singleton)
        residual = root_phi - left_phi - right_phi
        if residual >= -float(split_tolerance):
            candidates.append((left_phi + right_phi, -residual, left, right, residual))
    if not candidates:
        raise RuntimeError("No root split satisfies the hierarchy residual tolerance.")
    captured, _, left, right, residual = max(candidates)
    return left, right, float(residual), float(captured)


def is_planted_split(left: Sequence[str], right: Sequence[str]) -> bool:
    observed = {frozenset(left), frozenset(right)}
    truth = {frozenset(module) for module in PLANTED_MODULES}
    return observed == truth


def atom_mass_summary(atoms: Sequence[PhiAtom]) -> dict[str, float]:
    positive = [atom for atom in atoms if atom.value > 0.0]
    total = float(sum(atom.value for atom in positive))
    within = float(
        sum(
            atom.value
            for atom in positive
            if any(set(atom.sources).issubset(set(module)) for module in PLANTED_MODULES)
        )
    )
    cross = float(
        sum(
            atom.value
            for atom in positive
            if all(set(atom.sources) & set(module) for module in PLANTED_MODULES)
        )
    )
    return {
        "total_positive_atom_bits": total,
        "within_module_atom_bits": within,
        "cross_module_atom_bits": cross,
        "within_module_mass_fraction": within / total if total > 0.0 else 0.0,
        "cross_module_mass_fraction": cross / total if total > 0.0 else 0.0,
    }


def run_condition(
    *,
    sample_count: int,
    seed: int,
    within_coupling: float,
    cross_coupling: float,
    noise_scale: float,
    degree: int,
    shuffle_target: bool = False,
) -> dict[str, object]:
    started = time.perf_counter()
    phases, target = paired_data(
        sample_count=sample_count,
        seed=seed,
        within_coupling=within_coupling,
        cross_coupling=cross_coupling,
        noise_scale=noise_scale,
    )
    if shuffle_target:
        target = target[np.random.default_rng(int(seed) + 900_001).permutation(len(target))]
    table = transport_map_ei_table(phases, target, degree=degree)
    singleton = {name: float(table[(name,)]) for name in NAMES}
    root_phi = subset_phi_raw(NAMES, table, singleton)
    atoms = greedy_phi_atoms(
        NAMES,
        table,
        policy=NONNEGATIVE_TOLERANT,
        eps=1.0e-5,
        split_tolerance=0.10,
        singleton_ei=singleton,
    )
    left, right, root_residual, root_captured = best_root_split(table)
    mass = atom_mass_summary(atoms)
    ranked = sorted((atom for atom in atoms if atom.value > 0.0), key=lambda atom: atom.value, reverse=True)
    return {
        "sample_count": int(sample_count),
        "seed": int(seed),
        "within_coupling": float(within_coupling),
        "cross_coupling": float(cross_coupling),
        "noise_scale": float(noise_scale),
        "transport_degree": int(degree),
        "shuffle_target": bool(shuffle_target),
        "root_phi_bits": float(root_phi),
        "root_split": [list(left), list(right)],
        "root_split_exact": bool(is_planted_split(left, right)),
        "root_residual_bits": float(root_residual),
        "root_residual_fraction": float(root_residual / root_phi) if abs(root_phi) > 1.0e-12 else 0.0,
        "root_captured_bits": float(root_captured),
        "closure_error_bits": float(sum(atom.value for atom in atoms) - root_phi),
        **mass,
        "atoms": [
            {
                "sources": list(atom.sources),
                "value_bits": float(atom.value),
                "kind": atom.kind,
                "depth": int(atom.depth),
            }
            for atom in ranked
        ],
        "ei_bits": {"+".join(key): float(value) for key, value in table.items()},
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def aggregate(rows: Sequence[Mapping[str, object]]) -> list[dict[str, float]]:
    summaries: list[dict[str, float]] = []
    for cross_coupling in sorted({float(row["cross_coupling"]) for row in rows}):
        group = [row for row in rows if float(row["cross_coupling"]) == cross_coupling]
        summary: dict[str, float] = {
            "cross_coupling": cross_coupling,
            "n_seeds": int(len(group)),
            "root_split_exact_rate": float(np.mean([row["root_split_exact"] for row in group])),
        }
        for metric in (
            "root_phi_bits",
            "root_residual_bits",
            "root_residual_fraction",
            "within_module_mass_fraction",
            "cross_module_mass_fraction",
            "closure_error_bits",
        ):
            values = np.asarray([float(row[metric]) for row in group])
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_sem"] = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def build_summary(
    *,
    seeds: Sequence[int],
    cross_couplings: Sequence[float],
    sample_count: int,
    within_coupling: float,
    noise_scale: float,
    degree: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    total_conditions = len(cross_couplings) * len(seeds)
    for cross_coupling in cross_couplings:
        for seed in seeds:
            row = run_condition(
            sample_count=sample_count,
            seed=int(seed),
            within_coupling=within_coupling,
            cross_coupling=float(cross_coupling),
            noise_scale=noise_scale,
            degree=degree,
        )
            rows.append(row)
            print(
                f"[{len(rows)}/{total_conditions}] K_out={float(cross_coupling):.2f}, "
                f"seed={int(seed)}, root_exact={int(row['root_split_exact'])}, "
                f"elapsed={float(row['elapsed_seconds']):.1f}s",
                flush=True,
            )
    representative = next(
        row for row in rows if int(row["seed"]) == int(seeds[0]) and float(row["cross_coupling"]) == float(cross_couplings[0])
    )
    print("[control] fitting full target-shuffle EI table", flush=True)
    shuffled = run_condition(
        sample_count=sample_count,
        seed=int(seeds[0]),
        within_coupling=within_coupling,
        cross_coupling=float(cross_couplings[0]),
        noise_scale=noise_scale,
        degree=degree,
        shuffle_target=True,
    )
    return {
        "experiment_contract": {
            "question": "What changes in the recovered hierarchy when only cross-community Kuramoto coupling changes?",
            "dynamics": "six-oscillator modular Kuramoto vector field",
            "planted_modules": [list(module) for module in PLANTED_MODULES],
            "intervention": "independent Uniform(-pi, pi) phases",
            "target": "noisy instantaneous phase velocities from the known vector field",
            "estimator": f"degree-{degree} polynomial triangular transport-map EI",
            "primary_metric": "exact planted root bipartition recovery",
            "paired_unit": "seed; phases, frequencies, and standardized target noise shared across coupling levels",
            "fixed": {
                "within_coupling": float(within_coupling),
                "sample_count": int(sample_count),
                "noise_scale": float(noise_scale),
                "frequencies": FREQUENCIES.tolist(),
            },
        },
        "rows": rows,
        "summary": aggregate(rows),
        "representative": representative,
        "target_shuffle_control": shuffled,
    }


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def plot_summary(payload: Mapping[str, object], figure_base: Path) -> None:
    configure_plotting()
    summary = payload["summary"]
    representative = payload["representative"]
    shuffled = payload["target_shuffle_control"]
    figure, axes = plt.subplots(1, 4, figsize=(7.2, 2.15), constrained_layout=True)

    positions = np.array([[0.0, 0.7], [-0.65, -0.35], [0.65, -0.35], [2.2, 0.7], [1.55, -0.35], [2.85, -0.35]])
    within_color, cross_color = "#4477AA", "#CC6677"
    for module in ((0, 1, 2), (3, 4, 5)):
        for left, right in itertools.combinations(module, 2):
            axes[0].plot(*zip(positions[left], positions[right]), color=within_color, lw=2.0, zorder=1)
    for left in (0, 1, 2):
        for right in (3, 4, 5):
            axes[0].plot(*zip(positions[left], positions[right]), color=cross_color, lw=0.45, alpha=0.35, zorder=0)
    axes[0].scatter(positions[:, 0], positions[:, 1], s=120, color="white", edgecolor="0.15", lw=0.8, zorder=2)
    for index, (x_value, y_value) in enumerate(positions):
        axes[0].text(x_value, y_value, str(index + 1), ha="center", va="center", zorder=3)
    axes[0].text(0.0, -0.78, r"$K_{in}$", color=within_color, ha="center")
    axes[0].text(2.2, -0.78, r"$K_{in}$", color=within_color, ha="center")
    axes[0].text(1.1, 0.98, r"$K_{out}$", color=cross_color, ha="center")
    axes[0].set_xlim(-1.0, 3.2)
    axes[0].set_ylim(-1.0, 1.15)
    axes[0].axis("off")

    atoms = representative["atoms"][:8]
    labels = ["{" + ",".join(source.replace("theta", "") for source in row["sources"]) + "}" for row in atoms]
    values = np.asarray([float(row["value_bits"]) for row in atoms])
    colors = [
        within_color if any(set(row["sources"]).issubset(set(module)) for module in PLANTED_MODULES) else cross_color
        for row in atoms
    ]
    order = np.arange(len(atoms))[::-1]
    axes[1].barh(order, values, color=colors)
    axes[1].set_yticks(order, labels)
    axes[1].set_xlabel("Greedy atom (bits)")
    axes[1].text(0.98, 0.04, f"within mass = {representative['within_module_mass_fraction']:.1%}", transform=axes[1].transAxes, ha="right")

    couplings = np.asarray([row["cross_coupling"] for row in summary], dtype=float)
    for metric, label, color, marker in (
        ("within_module_mass_fraction", "Within planted modules", within_color, "o"),
        ("cross_module_mass_fraction", "Cross-module residual", cross_color, "s"),
    ):
        mean = np.asarray([row[f"{metric}_mean"] for row in summary], dtype=float)
        sem = np.asarray([row[f"{metric}_sem"] for row in summary], dtype=float)
        axes[2].errorbar(couplings, mean, yerr=sem, color=color, marker=marker, lw=1.2, capsize=2, label=label)
    axes[2].set(xlabel=r"Cross coupling $K_{out}$", ylabel="Fraction of positive atom mass", ylim=(-0.04, 1.04))
    axes[2].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1)

    recovery = np.asarray([row["root_split_exact_rate"] for row in summary], dtype=float)
    axes[3].plot(couplings, recovery, color=within_color, marker="o", lw=1.2)
    axes[3].set(xlabel=r"Cross coupling $K_{out}$", ylabel="Exact planted root split", ylim=(-0.04, 1.04))
    axes[3].text(
        0.04,
        0.06,
        f"shuffle: Phi {shuffled['root_phi_bits']:.3f} bits\nroot recovered: {int(shuffled['root_split_exact'])}",
        transform=axes[3].transAxes,
        va="bottom",
    )

    for label, axis in zip("abcd", axes):
        axis.text(-0.16, 1.08, label, transform=axis.transAxes, fontsize=8, fontweight="bold", va="top")

    figure_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(figure_base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(figure_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--figure-base", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_count = int(args.sample_count or (500 if args.mode == "smoke" else 1200))
    cross_couplings = (0.0, 0.25) if args.mode == "smoke" else (0.0, 0.25, 0.75, 1.50)
    payload = build_summary(
        seeds=tuple(range(int(args.seeds))),
        cross_couplings=cross_couplings,
        sample_count=sample_count,
        within_coupling=1.50,
        noise_scale=0.08,
        degree=int(args.degree),
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_summary(payload, args.figure_base)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Saved result: {args.result}")
    print(f"Saved figure: {args.figure_base}.{{png,svg,pdf}}")


if __name__ == "__main__":
    main()
