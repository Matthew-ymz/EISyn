#!/usr/bin/env python3
"""Plot matched SLP loading maps behind the 60-PC to 67-PC source-pair change."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from plot_runge_gateway_mediator_map import (
    COASTLINE_URL,
    add_geographic_ticks,
    extract_lines,
    load_geojson,
    split_dateline,
)


ROOT = Path(__file__).resolve().parents[1]
OLD_MAPS = (
    ROOT
    / "results/runge_slp_daily_1948_2026_20260628/results/runge/2015_gateways/component_maps.npz"
)
NEW_MAPS = (
    ROOT
    / "results/runge_slp_daily_1948_2026_pc67_20260731/results/runge/2015_gateways/component_maps.npz"
)
DEFAULT_OUTPUT = ROOT / "fig/runge_slp_pc60_pc67_spatial_modes"


def draw_coastlines(ax: plt.Axes, coastlines: list[list[tuple[float, float]]]) -> None:
    for line in coastlines:
        for segment in split_dateline(line):
            ax.plot(
                np.radians([point[0] for point in segment]),
                np.radians([point[1] for point in segment]),
                color="#30343A",
                linewidth=0.38,
                alpha=0.82,
                zorder=3,
            )


def sparsify_longitude_labels(ax: plt.Axes) -> None:
    longitude_labels = {"120°W", "60°W", "0°", "60°E", "120°E"}
    for text in list(ax.texts):
        if text.get_text() in longitude_labels:
            text.remove()
    projection_lat = np.radians(-67.0)
    for longitude, label in ((-120.0, "120°W"), (0.0, "0°"), (120.0, "120°E")):
        display_x, _ = ax.transData.transform((np.radians(longitude), projection_lat))
        axes_x, _ = ax.transAxes.inverted().transform((display_x, 0.0))
        ax.text(
            axes_x,
            -0.052,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=5.4,
            color="#4a4a4a",
            clip_on=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-maps", type=Path, default=OLD_MAPS)
    parser.add_argument("--new-maps", type=Path, default=NEW_MAPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    old = np.load(args.old_maps, allow_pickle=False)["component_maps"]
    new = np.load(args.new_maps, allow_pickle=False)["component_maps"]
    panels = (
        (old[:, :, 0], "60 PCs: No.0"),
        (old[:, :, 1], "60 PCs: No.1"),
        (old[:, :, 8], "60 PCs: local No.8"),
        (new[:, :, 0], "67 PCs: No.0  ($r=0.972$)"),
        (new[:, :, 1], "67 PCs: No.1  ($r=0.923$)"),
        (new[:, :, 7], "67 PCs: local No.7  ($r=0.823$)"),
    )
    limit = float(np.percentile(np.abs(np.stack([panel[0] for panel in panels])), 99.0))
    lat = np.linspace(-90.0, 90.0, old.shape[0])
    lon = ((np.linspace(0.0, 360.0, old.shape[1], endpoint=False) + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    coastlines = extract_lines(load_geojson(COASTLINE_URL))

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.65,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.2, 3.25),
        subplot_kw={"projection": "mollweide"},
        constrained_layout=True,
    )
    image = None
    for label, ax, (values, title) in zip("abcdef", axes.flat, panels, strict=True):
        image = ax.pcolormesh(
            np.radians(lon[order]),
            np.radians(lat),
            values[:, order],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            shading="auto",
            rasterized=True,
            zorder=1,
        )
        draw_coastlines(ax, coastlines)
        add_geographic_ticks(ax)
        sparsify_longitude_labels(ax)
        ax.set_title(title, fontsize=7.2, pad=4)
        ax.text(
            -0.05,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=8.2,
        )
    assert image is not None
    colorbar = fig.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        fraction=0.055,
        pad=0.08,
        aspect=45,
    )
    colorbar.set_label("Rotated PCA loading")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
