#!/usr/bin/env python3
"""Plot the processed Schaefer-100 structural matrix used by the DMF model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "dmf_schaefer100" / "group_mean_native.npz"
DEFAULT_OUTPUT = ROOT / "fig" / "dmf_schaefer100_connectivity_heatmap.png"

NETWORK_NAMES = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal attention",
    "SalVentAttn": "Salience / ventral attention",
    "Limbic": "Limbic",
    "Cont": "Control",
    "Default": "Default mode",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def roi_group(label: str) -> tuple[str, str]:
    parts = str(label).split("_")
    if len(parts) < 4 or parts[1] not in {"LH", "RH"}:
        raise ValueError(f"Unexpected Schaefer-100 label: {label}")
    return parts[1], parts[2]


def contiguous_blocks(labels: np.ndarray) -> list[tuple[int, int, str]]:
    groups = [roi_group(str(label)) for label in labels]
    blocks: list[tuple[int, int, str]] = []
    start = 0
    for index in range(1, len(groups) + 1):
        if index == len(groups) or groups[index] != groups[start]:
            hemisphere, network = groups[start]
            blocks.append((start, index, f"{hemisphere} {NETWORK_NAMES[network]}"))
            start = index
    return blocks


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=True) as archive:
        connectivity = np.asarray(archive["connectivity"], dtype=float)
        labels = np.asarray(archive["labels"]).astype(str)

    if connectivity.shape != (100, 100):
        raise ValueError(f"Expected a 100 x 100 matrix, got {connectivity.shape}.")
    if not np.isfinite(connectivity).all() or np.any(connectivity < 0.0):
        raise ValueError("Connectivity must be finite and non-negative.")
    if not np.allclose(connectivity, connectivity.T, atol=1.0e-12):
        raise ValueError("Connectivity must be symmetric.")
    if not np.allclose(np.diag(connectivity), 0.0, atol=1.0e-12):
        raise ValueError("Connectivity diagonal must be zero.")

    positive = connectivity[connectivity > 0.0]
    vmin = max(1.0e-6, float(positive.min()))
    vmax = float(positive.max())
    blocks = contiguous_blocks(labels)

    configure()
    figure, axis = plt.subplots(figsize=(7.15, 6.3), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list(
        "scholarly_blue",
        ("#EEF3F5", "#C7DADD", "#8DB8BC", "#528D96", "#2F6375", "#17364F"),
    )
    cmap.set_under("#FFFFFF")
    cmap.set_bad("#FFFFFF")
    image = axis.imshow(
        connectivity,
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
        origin="upper",
        rasterized=True,
    )

    centers = [(start + stop - 1) / 2 for start, stop, _ in blocks]
    names = [name for _, _, name in blocks]
    axis.set_xticks(centers, names, rotation=55, ha="right", rotation_mode="anchor", fontsize=5.4)
    axis.set_yticks(centers, names, fontsize=5.4)
    axis.set_xlabel("Source ROI (Schaefer-100 order)")
    axis.set_ylabel("Target ROI (Schaefer-100 order)")
    axis.tick_params(length=0, pad=2)

    for _, stop, _ in blocks[:-1]:
        position = stop - 0.5
        linewidth = 1.15 if stop == 50 else 0.45
        color = "white" if stop == 50 else (1.0, 1.0, 1.0, 0.60)
        axis.axhline(position, color=color, linewidth=linewidth)
        axis.axvline(position, color=color, linewidth=linewidth)

    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
    colorbar.set_label("Mean structural weight (a.u.; log scale)")
    colorbar.ax.tick_params(labelsize=6, width=0.6)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
