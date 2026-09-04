#!/usr/bin/env python3
"""Plot coalition-order summaries of HCP minimum-bipartition synergy."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "results/hcp_min_bipartition_synergy_57/minimum_bipartition_synergy_57.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/hcp_min_bipartition_synergy_57/"
    "minimum_bipartition_synergy_order_heatmap_57.png"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.linewidth": 0.7,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
        }
    )


def load_order_summary(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(archive["subject_values_bits"], dtype=float)
        sizes = np.asarray(archive["coalition_sizes"], dtype=int)
        subjects = archive["subjects"].astype(str)
        tolerance = float(np.asarray(archive["syn_tolerance_bits"]).item())

    if values.ndim != 3 or values.shape[0] != len(STATE_LABELS):
        raise ValueError(f"Expected state-by-subject-by-coalition values, got {values.shape}")
    if values.shape[1] != len(subjects) or values.shape[2] != len(sizes):
        raise ValueError("Subject or coalition metadata does not match the value array")
    if not np.isfinite(values).all():
        raise ValueError("Synergy cache contains non-finite values")

    violation = values < -tolerance
    if np.any(violation):
        raise ValueError(
            "Minimum-bipartition Syn nonnegativity violation: "
            f"minimum={values.min():.12g}, threshold={-tolerance:.12g}, "
            f"affected_count={int(violation.sum())}"
        )

    orders = np.arange(2, 8)
    subject_order_means = np.stack(
        [values[:, :, sizes == order].mean(axis=2) for order in orders], axis=0
    )
    group_means = subject_order_means.mean(axis=2)
    return group_means, orders, subjects


def text_color(value: float, image: mpl.image.AxesImage) -> str:
    red, green, blue, _ = image.cmap(image.norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#111111" if luminance > 0.56 else "#FFFFFF"


def plot_heatmap(
    group_means: np.ndarray,
    orders: np.ndarray,
    subject_count: int,
    output_path: Path,
) -> None:
    configure_style()
    figure, axis = plt.subplots(figsize=(6.8, 3.05), layout="constrained")
    color_max = float(np.ceil(group_means.max() * 10.0) / 10.0)
    image = axis.imshow(
        group_means,
        cmap="viridis",
        vmin=0.0,
        vmax=color_max,
        aspect="auto",
        interpolation="nearest",
    )

    axis.set(
        xticks=np.arange(len(STATE_LABELS)),
        xticklabels=STATE_LABELS,
        yticks=np.arange(len(orders)),
        yticklabels=[str(order) for order in orders],
        xlabel="State",
        ylabel="Coalition order",
    )
    axis.xaxis.tick_top()
    axis.xaxis.set_label_position("top")
    axis.tick_params(axis="x", labelrotation=30, labelsize=7.2, pad=3)
    axis.tick_params(axis="y", labelsize=7.2, pad=3)

    # A restrained outline makes the REST column easy to trace without changing its scale.
    axis.add_patch(
        mpl.patches.Rectangle(
            (-0.5, -0.5),
            1.0,
            len(orders),
            fill=False,
            edgecolor="white",
            linewidth=1.25,
            clip_on=False,
        )
    )
    axis.set_xticks(np.arange(-0.5, len(STATE_LABELS), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(orders), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.45, alpha=0.55)
    axis.tick_params(which="minor", bottom=False, left=False)

    for row in range(group_means.shape[0]):
        for column in range(group_means.shape[1]):
            value = group_means[row, column]
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=6.4,
                fontweight="medium",
                color=text_color(value, image),
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.036, pad=0.025, aspect=22)
    colorbar.set_label("Mean minimum-bipartition Syn (bits)", fontsize=7.0)
    colorbar.ax.tick_params(labelsize=6.5, length=2)
    axis.text(
        1.0,
        -0.135,
        f"n = {subject_count}; subject-level means across same-order coalitions",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color="#4A4A4A",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    group_means, orders, subjects = load_order_summary(args.input)
    plot_heatmap(group_means, orders, len(subjects), args.output)
    print(f"subjects={len(subjects)} orders={orders.tolist()}")
    print(np.array2string(group_means, precision=6, suppress_small=False))
    print(args.output)


if __name__ == "__main__":
    main()
