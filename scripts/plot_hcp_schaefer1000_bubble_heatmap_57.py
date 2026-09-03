#!/usr/bin/env python3
"""Plot frequency-by-strength bubbles for HCP greedy-hierarchy atoms."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "results/hcp_schaefer1000_task_evoked_xi_57/full/k1_p3_a1/arrays.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/hcp_schaefer1000_task_evoked_xi_57/final"
    / "hcp_schaefer1000_bubble_heatmap_57.png"
)

STATE_LABELS = (
    "REST",
    "Emotion",
    "Gambling",
    "Language",
    "Motor",
    "Relational",
    "Social",
    "WM",
)
NETWORK_SHORT = {
    "Vis": "V",
    "SomMot": "SM",
    "DorsAttn": "DAN",
    "SalVentAttn": "VAN",
    "Limbic": "Lim",
    "Cont": "FPN",
    "Default": "DMN",
}
SYN_NONNEGATIVE_TOLERANCE_BITS = 1.0e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=40)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def compact_atom(name: str) -> str:
    return "+".join(NETWORK_SHORT[item] for item in name.split("+"))


def validate_and_normalize(values: np.ndarray) -> tuple[np.ndarray, int]:
    if values.ndim != 3 or values.shape[:2] != (8, 57):
        raise ValueError(f"Expected atom_value shape (8, 57, p), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("atom_value contains non-finite values")
    violations = values < -SYN_NONNEGATIVE_TOLERANCE_BITS
    if np.any(violations):
        raise ValueError(
            "Significant Syn nonnegativity violation: "
            f"minimum={values.min():.6g} bits, "
            f"threshold={-SYN_NONNEGATIVE_TOLERANCE_BITS:.6g} bits, "
            f"count={int(violations.sum())}"
        )
    numerical = (values < 0.0) & ~violations
    normalized = values.copy()
    normalized[numerical] = 0.0
    return normalized, int(numerical.sum())


def bubble_area(frequency: np.ndarray) -> np.ndarray:
    # Scatter size is marker area in points squared; zero support stays invisible.
    return np.where(frequency > 0.0, 16.0 + 344.0 * frequency, 0.0)


def plot(input_path: Path, output_path: Path, top_n: int) -> dict[str, float | int]:
    archive = np.load(input_path)
    values, numerical_zero_count = validate_and_normalize(
        np.asarray(archive["atom_value"], dtype=float)
    )
    names = archive["atom_names"].astype(str)
    states = archive["states"].astype(str)
    if tuple(states) != (
        "REST",
        "EMOTION",
        "GAMBLING",
        "LANGUAGE",
        "MOTOR",
        "RELATIONAL",
        "SOCIAL",
        "WM",
    ):
        raise ValueError(f"Unexpected state order: {states.tolist()}")
    if len(names) != 120:
        raise ValueError(f"Expected 120 atoms, got {len(names)}")
    if not 1 <= top_n <= len(names):
        raise ValueError(f"top_n must lie in [1, {len(names)}]")

    present = values > 0.0
    support_count = present.sum(axis=1)
    frequency = support_count / values.shape[1]
    conditional_mean = np.divide(
        values.sum(axis=1),
        support_count,
        out=np.zeros((values.shape[0], values.shape[2]), dtype=float),
        where=support_count > 0,
    )
    zero_filled_mean = values.mean(axis=1)

    # Preserve the selection rule used by the existing panel B: mean atom share.
    mean_share = np.asarray(archive["atom_share"], dtype=float).mean(axis=(0, 1))
    selected = np.argsort(mean_share)[::-1][:top_n]

    configure_style()
    height = max(7.2, 0.285 * top_n + 2.0)
    figure = plt.figure(figsize=(10.6, height), layout="constrained")
    grid = figure.add_gridspec(1, 2, width_ratios=(9.1, 1.55), wspace=0.06)
    axis = figure.add_subplot(grid[0, 0])
    guide = figure.add_subplot(grid[0, 1])
    guide.axis("off")

    x = np.tile(np.arange(len(states)), top_n)
    y = np.repeat(np.arange(top_n), len(states))
    plotted_frequency = frequency[:, selected].T
    plotted_strength = conditional_mean[:, selected].T

    vmax = float(plotted_strength.max())
    norm = mpl.colors.PowerNorm(gamma=0.62, vmin=0.0, vmax=vmax)
    scatter = axis.scatter(
        x,
        y,
        s=bubble_area(plotted_frequency).ravel(),
        c=plotted_strength.ravel(),
        cmap="magma_r",
        norm=norm,
        edgecolors="#3A3A3A",
        linewidths=0.28,
        zorder=3,
    )

    for row in range(top_n):
        for column in range(len(states)):
            if support_count[column, selected[row]] < 5:
                continue
            strength = plotted_strength[row, column]
            axis.text(
                column,
                row,
                f"{strength:.2f}",
                ha="center",
                va="center",
                fontsize=4.6,
                color="white" if norm(strength) > 0.50 else "#202020",
                zorder=4,
            )

    axis.set(
        xlim=(-0.55, len(states) - 0.45),
        ylim=(top_n - 0.45, -0.55),
        xticks=np.arange(len(states)),
        xticklabels=STATE_LABELS,
        yticks=np.arange(top_n),
        yticklabels=[compact_atom(names[index]) for index in selected],
        xlabel="State",
        ylabel="Greedy hierarchy atom",
    )
    axis.xaxis.tick_top()
    axis.xaxis.set_label_position("top")
    axis.tick_params(axis="x", length=0, pad=6, labelrotation=28)
    axis.tick_params(axis="y", length=0, pad=3, labelsize=6.3)
    axis.set_xticks(np.arange(-0.5, len(states), 1.0), minor=True)
    axis.set_yticks(np.arange(-0.5, top_n, 1.0), minor=True)
    axis.grid(which="minor", color="#E6E7E9", linewidth=0.45, zorder=0)
    axis.tick_params(which="minor", bottom=False, left=False, top=False)
    axis.axvline(0.5, color="#8A8F96", linewidth=0.85, zorder=1)

    color_axis = guide.inset_axes([0.10, 0.54, 0.20, 0.34])
    colorbar = figure.colorbar(scatter, cax=color_axis)
    colorbar.set_label("Mean Syn when present (bits)", labelpad=7)
    colorbar.ax.tick_params(labelsize=6.3, length=2)

    legend_frequency = (0.10, 0.25, 0.50, 0.75, 1.00)
    handles = [
        plt.scatter(
            [],
            [],
            s=float(bubble_area(np.asarray([value]))[0]),
            facecolor="#B7A6C8",
            edgecolor="#3A3A3A",
            linewidth=0.35,
        )
        for value in legend_frequency
    ]
    guide.legend(
        handles,
        [f"{int(value * 100)}%" for value in legend_frequency],
        title="Occurrence\n(57 SPTs)",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.48),
        frameon=False,
        labelspacing=1.05,
        handletextpad=1.0,
        borderaxespad=0.0,
        fontsize=6.5,
        title_fontsize=7.0,
    )
    guide.text(
        0.0,
        0.08,
        "Color and text: mean Syn when present (bits)\n"
        "Text shown when support is at least 5/57\n"
        f"Top {top_n} by mean atom share\n"
        "Blank: absent from all 57 SPTs",
        transform=guide.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
        color="#454545",
        linespacing=1.45,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    max_frequency = frequency.max(axis=0)
    return {
        "candidate_atoms": int(len(names)),
        "observed_atoms": int(np.count_nonzero(max_frequency > 0.0)),
        "atoms_supported_by_at_least_5_of_57_in_any_state": int(
            np.count_nonzero(max_frequency >= 5 / 57)
        ),
        "displayed_atoms": int(top_n),
        "minimum_max_state_support_among_displayed": int(
            support_count[:, selected].max(axis=0).min()
        ),
        "syn_nonnegative_tolerance_bits": SYN_NONNEGATIVE_TOLERANCE_BITS,
        "numerical_zero_count": numerical_zero_count,
        "minimum_syn_bits": float(values.min()),
        "maximum_conditional_mean_bits": vmax,
        "maximum_zero_filled_mean_bits_retained_not_displayed": float(
            zero_filled_mean[:, selected].max()
        ),
    }


def main() -> int:
    args = parse_args()
    summary = plot(args.input, args.output, args.top_n)
    print(f"Saved {args.output}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
