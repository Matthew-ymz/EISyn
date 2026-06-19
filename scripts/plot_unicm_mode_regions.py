#!/usr/bin/env python3
"""Plot UniCM climate-mode index regions on global longitude-latitude maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

from plot_runge_gateway_mediator_map import (
    COASTLINE_URL,
    LAND_URL,
    add_geographic_ticks,
    draw_world,
    extract_lines,
    extract_polygons,
    load_geojson,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "reports" / "assets" / "unicm_mode_geography.png"


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


@dataclass(frozen=True)
class Region:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    tag: str = ""


@dataclass(frozen=True)
class ModeRegion:
    index: int
    name: str
    source: str
    regions: tuple[Region, ...]
    color: str


MODES: tuple[ModeRegion, ...] = (
    ModeRegion(0, "nino", "SST", (Region(190, 240, -5, 5),), "#0F6B8C"),
    ModeRegion(1, "NPMM", "SST", (Region(200, 240, 10, 25),), "#D55E00"),
    ModeRegion(2, "SPMM", "SST", (Region(250, 270, -25, -15),), "#009E73"),
    ModeRegion(3, "IOB", "SST", (Region(40, 100, -20, 0),), "#7B3294"),
    ModeRegion(
        4,
        "IOD",
        "SST",
        (Region(50, 70, -10, 10, "W"), Region(90, 110, -10, 0, "E")),
        "#C23B22",
    ),
    ModeRegion(
        5,
        "SIOD",
        "SST",
        (Region(65, 80, -25, -10, "W"), Region(90, 120, -30, -10, "E")),
        "#4C78A8",
    ),
    ModeRegion(6, "TNA", "SST", (Region(305, 345, 5, 25),), "#A6761D"),
    ModeRegion(7, "nino12", "SST", (Region(270, 280, -10, 0),), "#E7298A"),
    ModeRegion(8, "nino3", "SST", (Region(210, 270, -5, 5),), "#66A61E"),
    ModeRegion(9, "nino4", "SST", (Region(200, 210, -5, 5),), "#1B9E77"),
    ModeRegion(10, "WWV", "t20d", (Region(120, 280, -5, 5),), "#5E3C99"),
)


def lon_to_180(lon: float) -> float:
    return ((float(lon) + 180.0) % 360.0) - 180.0


def split_lon_range(lon_min: float, lon_max: float) -> list[tuple[float, float]]:
    left = lon_to_180(lon_min)
    right = lon_to_180(lon_max)
    if float(lon_max) - float(lon_min) >= 360.0:
        return [(-180.0, 180.0)]
    if left <= right:
        return [(left, right)]
    return [(left, 180.0), (-180.0, right)]


def region_center(region: Region) -> tuple[float, float]:
    span_mid = float(region.lon_min) + 0.5 * (float(region.lon_max) - float(region.lon_min))
    return lon_to_180(span_mid), 0.5 * (float(region.lat_min) + float(region.lat_max))


def draw_region(
    ax: plt.Axes,
    region: Region,
    *,
    color: str,
    alpha: float,
    linewidth: float,
    label: str | None = None,
) -> None:
    for lon_left, lon_right in split_lon_range(region.lon_min, region.lon_max):
        lons = np.array([lon_left, lon_right, lon_right, lon_left, lon_left], dtype=float)
        lats = np.array([region.lat_min, region.lat_min, region.lat_max, region.lat_max, region.lat_min], dtype=float)
        ax.fill(
            np.radians(lons),
            np.radians(lats),
            facecolor=color,
            edgecolor=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=4,
        )
        ax.plot(np.radians(lons), np.radians(lats), color=color, linewidth=linewidth, alpha=0.95, zorder=5)

    if label:
        lon, lat = region_center(region)
        ax.text(
            np.radians(lon),
            np.radians(lat),
            label,
            ha="center",
            va="center",
            fontsize=6.2,
            fontweight="bold",
            color="#111111",
            zorder=6,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white", alpha=0.86)],
        )


def draw_mode(ax: plt.Axes, mode: ModeRegion, *, overview: bool) -> None:
    for region in mode.regions:
        if overview:
            label = f"{mode.index}{region.tag}" if region.tag else str(mode.index)
        else:
            label = f"{mode.name}{region.tag}" if region.tag else mode.name
        draw_region(
            ax,
            region,
            color=mode.color,
            alpha=0.18 if overview else 0.28,
            linewidth=0.72 if overview else 1.0,
            label=label,
        )


def setup_axis(
    ax: plt.Axes,
    land: Iterable[list[tuple[float, float]]],
    coastlines: Iterable[list[tuple[float, float]]],
    panel_label: str,
) -> None:
    draw_world(ax, land, coastlines)
    add_geographic_ticks(ax)
    ax.text(
        0.015,
        0.98,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color="#111111",
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        zorder=10,
    )


def plot_unicm_mode_regions(output: Path = DEFAULT_OUTPUT) -> Path:
    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))

    mode_cols = 3
    mode_rows = int(np.ceil(len(MODES) / mode_cols))
    fig = plt.figure(figsize=(8.2, 8.9), constrained_layout=True)
    grid = fig.add_gridspec(
        mode_rows + 1,
        mode_cols,
        height_ratios=[1.25] + [1.0] * mode_rows,
    )

    ax_overview = fig.add_subplot(grid[0, :], projection="mollweide")
    setup_axis(ax_overview, land, coastlines, "All UniCM mode index regions")
    for mode in MODES:
        draw_mode(ax_overview, mode, overview=True)

    for idx, mode in enumerate(MODES):
        row = 1 + idx // mode_cols
        col = idx % mode_cols
        ax = fig.add_subplot(grid[row, col], projection="mollweide")
        setup_axis(ax, land, coastlines, f"{mode.index}: {mode.name} ({mode.source})")
        draw_mode(ax, mode, overview=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    output = plot_unicm_mode_regions(DEFAULT_OUTPUT)
    print(output)


if __name__ == "__main__":
    main()
