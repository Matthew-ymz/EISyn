#!/usr/bin/env python3
"""Analytic toy examples for spectral-gap-driven PEID source condensation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results/runge_synergy_source_condensation_toy"
FIGURE_BASE = ROOT / "fig/runge_synergy_source_condensation_toy"
DIMENSION = 12
LEADING_EIGENVALUE = 0.95
GAP_RATIOS = (0.25, 0.50, 0.75, 0.90, 0.97)
HORIZONS = np.arange(1, 81, dtype=int)
HEATMAP_HORIZON = 10
NOISE_STD = 0.2
LOCAL_SUPPORT = 3
CONDENSATION_THRESHOLD = 4.0


def orthonormal_basis_with_first(first: np.ndarray, *, seed: int = 17) -> np.ndarray:
    first = np.asarray(first, dtype=float)
    first = first / np.linalg.norm(first)
    rng = np.random.default_rng(seed)
    candidates = rng.normal(size=(len(first), len(first) - 1))
    candidates -= first[:, None] * (first @ candidates)[None, :]
    complement, _ = np.linalg.qr(candidates)
    basis = np.column_stack([first, complement])
    if not np.allclose(basis.T @ basis, np.eye(len(first)), atol=1e-10):
        raise RuntimeError("Failed to construct an orthonormal eigenbasis.")
    return basis


def spectrum_for_ratio(q: float, *, dimension: int = DIMENSION) -> np.ndarray:
    if not 0.0 < q < 1.0:
        raise ValueError("The gap ratio q must lie in (0, 1).")
    subdominant_profile = np.linspace(1.0, 0.35, dimension - 1)
    return np.r_[LEADING_EIGENVALUE, LEADING_EIGENVALUE * q * subdominant_profile]


def symmetric_system(first: np.ndarray, q: float, *, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    basis = orthonormal_basis_with_first(first, seed=seed)
    eigenvalues = spectrum_for_ratio(q, dimension=len(first))
    matrix = basis @ np.diag(eigenvalues) @ basis.T
    return matrix, eigenvalues


def no_gap_rotation_system(
    *, dimension: int = DIMENSION, radius: float = LEADING_EIGENVALUE, angle: float = 0.173
) -> np.ndarray:
    if dimension % 2:
        raise ValueError("The block-rotation example requires an even dimension.")
    matrix = np.zeros((dimension, dimension), dtype=float)
    rotation = radius * np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=float
    )
    for start in range(0, dimension, 2):
        matrix[start : start + 2, start : start + 2] = rotation
    return matrix


def participation_effective_rank(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mass = float(np.sum(values))
    if mass <= 0.0:
        raise ValueError("Participation weights must have positive mass.")
    probabilities = values / mass
    return float(1.0 / np.sum(probabilities**2))


def transfer_effective_rank(matrix: np.ndarray) -> float:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return participation_effective_rank(singular_values**2)


def transfer_source_distribution(matrix: np.ndarray) -> np.ndarray:
    column_energy = np.sum(np.asarray(matrix, dtype=float) ** 2, axis=0)
    return column_energy / np.sum(column_energy)


def gaussian_pair_synergy_matrix(
    matrix: np.ndarray, *, noise_variances: float | np.ndarray
) -> np.ndarray:
    """Exact PEID pair synergy for independent Gaussian sources and target noise."""
    matrix = np.asarray(matrix, dtype=float)
    dimension = matrix.shape[1]
    result = np.zeros((dimension, dimension), dtype=float)
    row_energy = np.sum(matrix**2, axis=1)
    target_noise = np.broadcast_to(np.asarray(noise_variances, dtype=float), matrix.shape[0])
    if np.any(target_noise <= 0.0):
        raise ValueError("Every target noise variance must be positive.")
    for source_a in range(dimension):
        a2 = matrix[:, source_a] ** 2
        for source_b in range(source_a + 1, dimension):
            b2 = matrix[:, source_b] ** 2
            residual_variance = target_noise + row_energy - a2 - b2
            if np.any(residual_variance <= 0.0):
                raise RuntimeError("The Gaussian residual variance must remain positive.")
            increment = (a2 * b2) / (
                residual_variance * (residual_variance + a2 + b2)
            )
            synergy_bits = float(np.sum(0.5 * np.log1p(increment) / np.log(2.0)))
            result[source_a, source_b] = synergy_bits
            result[source_b, source_a] = synergy_bits
    return result


def horizon_transition_and_noise(
    matrix: np.ndarray, horizon: int, *, noise_std: float = NOISE_STD
) -> tuple[np.ndarray, np.ndarray]:
    """Return A^h and covariance of accumulated process noise."""
    matrix = np.asarray(matrix, dtype=float)
    transition = np.eye(len(matrix), dtype=float)
    covariance = np.zeros_like(matrix)
    innovation_covariance = float(noise_std) ** 2 * np.eye(len(matrix), dtype=float)
    for _ in range(int(horizon)):
        transition = matrix @ transition
        covariance = matrix @ covariance @ matrix.T + innovation_covariance
    return transition, covariance


def normalized_pair_mass(synergy: np.ndarray) -> np.ndarray:
    upper = np.triu(np.asarray(synergy, dtype=float), k=1)
    total = float(np.sum(upper))
    if total <= 0.0:
        raise RuntimeError("The analytic pair-synergy mass vanished.")
    normalized = upper / total
    return normalized + normalized.T


def synergy_source_distribution(pair_mass: np.ndarray) -> np.ndarray:
    distribution = 0.5 * np.sum(np.asarray(pair_mass, dtype=float), axis=1)
    if not np.isclose(np.sum(distribution), 1.0, atol=1e-10):
        raise RuntimeError("Pair-to-source participation does not sum to one.")
    return distribution


def summarize_system(matrix: np.ndarray) -> dict[str, object]:
    transfer_rank: list[float] = []
    transfer_sources: list[float] = []
    synergy_sources: list[float] = []
    synergy_pairs: list[float] = []
    absolute_synergy: list[float] = []
    for horizon in HORIZONS:
        transition, noise_covariance = horizon_transition_and_noise(matrix, int(horizon))
        transfer_rank.append(transfer_effective_rank(transition))
        transfer_distribution = transfer_source_distribution(transition)
        transfer_sources.append(participation_effective_rank(transfer_distribution))
        synergy = gaussian_pair_synergy_matrix(
            transition, noise_variances=np.diag(noise_covariance)
        )
        absolute_synergy.append(float(np.sum(np.triu(synergy, k=1))))
        pair_mass = normalized_pair_mass(synergy)
        source_distribution = synergy_source_distribution(pair_mass)
        synergy_sources.append(participation_effective_rank(source_distribution))
        synergy_pairs.append(participation_effective_rank(pair_mass[np.triu_indices(len(matrix), k=1)]))
    return {
        "transfer_effective_rank": transfer_rank,
        "transfer_effective_sources": transfer_sources,
        "synergy_effective_sources": synergy_sources,
        "synergy_effective_pairs": synergy_pairs,
        "absolute_synergy_bits": absolute_synergy,
    }


def first_threshold_horizon(values: list[float], *, threshold: float) -> int | None:
    for horizon, value in zip(HORIZONS, values):
        if float(value) <= float(threshold):
            return int(horizon)
    return None


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.4,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.17, 1.04, label, transform=axis.transAxes, fontsize=8.2, fontweight="bold")


def plot_results(
    *,
    systems: dict[str, dict[str, object]],
    matrices: dict[str, np.ndarray],
    gap_table: list[dict[str, float | int]],
) -> list[str]:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.75), layout="constrained")
    colors = {
        "q025": "#B8CBD8",
        "q050": "#91B2C7",
        "q075": "#6796B2",
        "q090": "#3E7A9D",
        "q097": "#145F86",
        "no_gap": "#6F6F76",
    }
    labels = {
        "q025": r"$q=0.25$",
        "q050": r"$q=0.50$",
        "q075": r"$q=0.75$",
        "q090": r"$q=0.90$",
        "q097": r"$q=0.97$",
        "no_gap": "No gap",
    }
    ordered = ("q025", "q050", "q075", "q090", "q097", "no_gap")

    axis = axes[0, 0]
    handles = []
    for name in ordered:
        line = axis.plot(
            HORIZONS,
            systems[name]["transfer_effective_rank"],
            color=colors[name],
            linewidth=1.25,
            linestyle=("--" if name == "no_gap" else "-"),
            label=labels[name],
        )[0]
        handles.append(line)
    axis.set_xlabel("Forecast horizon")
    axis.set_ylabel("Effective transfer rank")
    axis.set_ylim(0.7, DIMENSION + 0.5)
    axis.grid(color="#E8EBEF", linewidth=0.5)
    add_panel_label(axis, "a")

    axis = axes[0, 1]
    for name in ordered:
        axis.plot(
            HORIZONS,
            systems[name]["synergy_effective_sources"],
            color=colors[name],
            linewidth=1.25,
            linestyle=("--" if name == "no_gap" else "-"),
        )
    axis.axhline(LOCAL_SUPPORT, color="#A9B0B6", linestyle=":", linewidth=0.8)
    axis.set_xlabel("Forecast horizon")
    axis.set_ylabel("Effective PEID source modes")
    axis.set_ylim(2.5, DIMENSION + 0.5)
    axis.grid(color="#E8EBEF", linewidth=0.5)
    add_panel_label(axis, "b")

    axis = axes[0, 2]
    timescale = np.asarray([float(row["gap_timescale"]) for row in gap_table])
    observed = np.asarray([float(row["horizon_to_four_sources"]) for row in gap_table])
    axis.plot(timescale, observed, color="#356A8A", marker="o", linewidth=1.2)
    for row in gap_table:
        axis.annotate(
            f"q={row['q']:.2f}",
            (float(row["gap_timescale"]), float(row["horizon_to_four_sources"])),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.2,
        )
    axis.set_xlabel(r"Gap timescale $1/\log(1/q)$")
    axis.set_ylabel("Horizon to ≤4 source modes")
    axis.grid(color="#E8EBEF", linewidth=0.5)
    add_panel_label(axis, "c")

    heatmap_names = ("q050", "q097", "dense_q050")
    heatmap_titles = (
        r"Localized, strong gap ($q=0.50$)",
        r"Localized, weak gap ($q=0.97$)",
        r"Dense, strong gap ($q=0.50$)",
    )
    heatmaps = []
    for name in heatmap_names:
        transition, noise_covariance = horizon_transition_and_noise(
            matrices[name], HEATMAP_HORIZON
        )
        heatmaps.append(
            normalized_pair_mass(
                gaussian_pair_synergy_matrix(
                    transition, noise_variances=np.diag(noise_covariance)
                )
            )
        )
    maximum = max(float(np.max(matrix)) for matrix in heatmaps)
    image = None
    for axis, matrix, title, letter in zip(axes[1], heatmaps, heatmap_titles, ("d", "e", "f")):
        masked = np.ma.masked_where(np.eye(DIMENSION, dtype=bool), matrix)
        image = axis.imshow(masked, cmap="magma", vmin=0.0, vmax=maximum, interpolation="nearest")
        axis.set_title(title, fontsize=6.1, fontweight="bold")
        axis.set_xlabel("Source mode")
        axis.set_ylabel("Source mode")
        axis.set_xticks([0, 2, 5, 8, 11], labels=["1", "3", "6", "9", "12"])
        axis.set_yticks([0, 2, 5, 8, 11], labels=["1", "3", "6", "9", "12"])
        pair_effective = participation_effective_rank(matrix[np.triu_indices(DIMENSION, k=1)])
        axis.text(
            0.04,
            0.07,
            rf"$N_{{\mathrm{{pair,eff}}}}={pair_effective:.1f}$",
            transform=axis.transAxes,
            color="white",
            fontsize=5.5,
            ha="left",
            va="bottom",
        )
        add_panel_label(axis, letter)
    if image is None:
        raise RuntimeError("Heatmap rendering did not initialize.")
    colorbar = fig.colorbar(image, ax=list(axes[1]), orientation="horizontal", fraction=0.06, pad=0.11)
    colorbar.set_label(f"Normalized pair-synergy mass at H={HEATMAP_HORIZON}")
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.74), fontsize=5.6)

    outputs: list[str] = []
    FIGURE_BASE.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 600}), (".svg", {}), (".pdf", {})):
        path = FIGURE_BASE.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    localized = np.r_[np.ones(LOCAL_SUPPORT) / math.sqrt(LOCAL_SUPPORT), np.zeros(DIMENSION - LOCAL_SUPPORT)]
    dense = np.ones(DIMENSION) / math.sqrt(DIMENSION)
    matrices: dict[str, np.ndarray] = {}
    spectra: dict[str, list[float]] = {}
    for q in GAP_RATIOS:
        name = f"q{int(round(100 * q)):03d}"
        matrices[name], eigenvalues = symmetric_system(localized, q)
        spectra[name] = eigenvalues.tolist()
    matrices["dense_q050"], dense_eigenvalues = symmetric_system(dense, 0.50)
    spectra["dense_q050"] = dense_eigenvalues.tolist()
    matrices["no_gap"] = no_gap_rotation_system()
    spectra["no_gap"] = np.linalg.eigvals(matrices["no_gap"]).tolist()

    systems = {name: summarize_system(matrix) for name, matrix in matrices.items()}
    gap_table: list[dict[str, float | int]] = []
    for q in GAP_RATIOS:
        name = f"q{int(round(100 * q)):03d}"
        threshold_horizon = first_threshold_horizon(
            systems[name]["synergy_effective_sources"], threshold=CONDENSATION_THRESHOLD
        )
        if threshold_horizon is None:
            raise RuntimeError(f"Condition {name} did not reach the condensation threshold.")
        gap_table.append(
            {
                "condition": name,
                "q": q,
                "relative_gap": 1.0 - q,
                "gap_timescale": float(1.0 / math.log(1.0 / q)),
                "horizon_to_four_sources": threshold_horizon,
                "h60_effective_sources": float(
                    systems[name]["synergy_effective_sources"][int(np.where(HORIZONS == 60)[0][0])]
                ),
            }
        )

    figure_outputs = plot_results(systems=systems, matrices=matrices, gap_table=gap_table)
    summary = {
        "schema_version": 1,
        "dimension": DIMENSION,
        "leading_eigenvalue": LEADING_EIGENVALUE,
        "gap_ratios": GAP_RATIOS,
        "horizons": HORIZONS.tolist(),
        "process_noise_std": NOISE_STD,
        "local_support": LOCAL_SUPPORT,
        "condensation_threshold_effective_sources": CONDENSATION_THRESHOLD,
        "gap_table": gap_table,
        "systems": systems,
        "spectra": {
            name: [
                {"real": float(np.real(value)), "imag": float(np.imag(value))}
                for value in values
            ]
            for name, values in spectra.items()
        },
        "matched_spectrum_check": {
            "localized_q050": sorted(np.linalg.eigvalsh(matrices["q050"]).tolist()),
            "dense_q050": sorted(np.linalg.eigvalsh(matrices["dense_q050"]).tolist()),
            "maximum_absolute_difference": float(
                np.max(
                    np.abs(
                        np.sort(np.linalg.eigvalsh(matrices["q050"]))
                        - np.sort(np.linalg.eigvalsh(matrices["dense_q050"]))
                    )
                )
            ),
        },
        "figure_outputs": figure_outputs,
        "figure_contract": {
            "core_conclusion": "The spectral gap controls the horizon scale of condensation, conditional on a localized dominant input direction.",
            "backend": "Python/matplotlib",
            "heatmap_horizon": HEATMAP_HORIZON,
            "reviewer_risks": [
                "normalized concentration can coexist with vanishing absolute synergy",
                "a dense dominant eigenvector blocks coordinate condensation despite rank-one propagation",
            ],
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"gap_table": gap_table, "matched_spectrum_check": summary["matched_spectrum_check"]}, indent=2))


if __name__ == "__main__":
    main()
