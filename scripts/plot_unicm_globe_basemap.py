#!/usr/bin/env python3
"""Draw the 11 UniCM modes as geographic nodes, without inferred edges."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_unicm_mode_regions import MODES, region_center
from plot_runge_gateway_mediator_map import (
    LAND_URL, COASTLINE_URL, load_geojson, extract_polygons, extract_lines,
    draw_world, format_lat_label,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/reports/assets/unicm_globe_basemap.png"


def main():
    # Geographic overview only: equal node size/color carries no estimated metric.
    # Dipoles use the midpoint of their two region centers as a schematic anchor.
    land = extract_polygons(load_geojson(LAND_URL))
    coast = extract_lines(load_geojson(COASTLINE_URL))
    if not land or not coast:
        raise RuntimeError("Natural Earth geography could not be loaded.")
    fig = plt.figure(figsize=(10.3, 4.6), facecolor="white")
    ax = fig.add_axes([0.045, 0.08, 0.76, 0.87], projection="mollweide")
    draw_world(ax, land, coast)
    ax.set_xticks([])
    ticks = np.arange(-60, 61, 30)
    ax.set_yticks(np.radians(ticks))
    ax.set_yticklabels([format_lat_label(v) for v in ticks], fontsize=8)
    ax.grid(axis="y", color="#b8b8b8", linestyle="--", linewidth=0.45, alpha=0.7)
    ax.spines["geo"].set_linewidth(1.0)
    # WWV and nino4 are separated by only five longitude degrees at the equator.
    # Offsets are in display points and do not change the geographic anchor.
    offsets = {9: (-3, 13), 10: (-8, -17)}
    for mode in MODES:
        centers = np.array([region_center(r) for r in mode.regions])
        lon, lat = centers.mean(axis=0)
        xy = tuple(np.radians([lon, lat]))
        offset = offsets.get(mode.index, (0, 0))
        ax.annotate(
            str(mode.index), xy=xy, xytext=offset, textcoords="offset points",
            ha="center", va="center", fontsize=9, color="white", zorder=8,
            bbox=dict(boxstyle="circle,pad=0.38", fc="#3976a1", ec="#263e4c", lw=1),
        )
    labels = fig.add_axes([0.84, 0.14, 0.15, 0.72])
    labels.axis("off")
    for i, mode in enumerate(MODES):
        y = 1 - i / 10
        labels.text(0.03, y, str(mode.index), ha="center", va="center", fontsize=9,
                    color="white", bbox=dict(boxstyle="circle,pad=0.38",
                    fc="#3976a1", ec="#263e4c", lw=1))
        labels.text(0.22, y, mode.name, va="center", fontsize=10, color="#252525")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=320, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
