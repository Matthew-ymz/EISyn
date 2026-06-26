#!/usr/bin/env python3
"""Draw a global map of Earth-system sensitive regions.

The map is intentionally approximate: shaded footprints mark broad sensitive
zones synthesized from assessment-level climate and Earth-system literature.
"""

from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "reports" / "assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

NATURAL_EARTH_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_110m_admin_0_countries.geojson"
)
CACHE_PATH = ASSET_DIR / "ne_110m_admin_0_countries.geojson"
OUT_BASE = ASSET_DIR / "global_earth_sensitive_regions"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.7,
    }
)


COLORS = {
    "cryosphere": "#4C78A8",
    "coast": "#F58518",
    "dryland": "#B279A2",
    "reef": "#54A24B",
    "forest": "#2F855A",
    "monsoon": "#E45756",
}


def load_countries() -> dict:
    if not CACHE_PATH.exists():
        with urllib.request.urlopen(NATURAL_EARTH_URL, timeout=20) as response:
            CACHE_PATH.write_bytes(response.read())
    return json.loads(CACHE_PATH.read_text())


def iter_polygons(geometry: dict):
    geom_type = geometry["type"]
    coords = geometry["coordinates"]
    if geom_type == "Polygon":
        yield coords
    elif geom_type == "MultiPolygon":
        yield from coords


def split_ring_at_dateline(ring: list[list[float]]) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = [[]]
    previous_lon = None
    for lon, lat, *_ in ring:
        if previous_lon is not None and abs(lon - previous_lon) > 180:
            segments.append([])
        segments[-1].append((lon, lat))
        previous_lon = lon
    return [segment for segment in segments if len(segment) > 1]


def draw_base_map(ax: plt.Axes) -> None:
    data = load_countries()
    ax.set_facecolor("#EAF3F7")
    for feature in data["features"]:
        geom = feature.get("geometry")
        if not geom:
            continue
        for polygon in iter_polygons(geom):
            exterior = polygon[0]
            for segment in split_ring_at_dateline(exterior):
                x, y = zip(*segment)
                ax.fill(x, y, color="#F4F1EA", ec="#B7B7B7", lw=0.28, zorder=1)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-62, 86)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-60, 91, 30))
    ax.grid(color="white", lw=0.6, alpha=0.85, zorder=0)
    ax.tick_params(length=0, colors="#5A5A5A", labelsize=7)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_ellipse(
    ax: plt.Axes,
    lon: float,
    lat: float,
    width: float,
    height: float,
    angle: float,
    color: str,
    alpha: float = 0.25,
) -> None:
    ax.add_patch(
        Ellipse(
            (lon, lat),
            width,
            height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            lw=1.0,
            alpha=alpha,
            zorder=3,
        )
    )


def add_rect(
    ax: plt.Axes,
    lon: float,
    lat: float,
    width: float,
    height: float,
    color: str,
    alpha: float = 0.18,
) -> None:
    ax.add_patch(
        Rectangle(
            (lon, lat),
            width,
            height,
            facecolor=color,
            edgecolor=color,
            lw=0.8,
            alpha=alpha,
            zorder=2,
        )
    )


def add_label(ax: plt.Axes, lon: float, lat: float, text: str) -> None:
    ax.text(
        lon,
        lat,
        text,
        ha="center",
        va="center",
        fontsize=6.7,
        weight="bold",
        color="#1F2933",
        bbox={
            "boxstyle": "circle,pad=0.18",
            "fc": "white",
            "ec": "#4B5563",
            "lw": 0.5,
            "alpha": 0.92,
        },
        zorder=6,
    )


def draw_sensitive_regions(ax: plt.Axes) -> None:
    c = COLORS

    # Cryosphere and high mountains.
    add_rect(ax, -180, 60, 360, 25, c["cryosphere"], 0.16)
    add_rect(ax, -180, -62, 360, 13, c["cryosphere"], 0.16)
    add_ellipse(ax, -42, 72, 34, 20, -18, c["cryosphere"], 0.32)
    add_ellipse(ax, -106, 68, 70, 18, 5, c["cryosphere"], 0.22)
    add_ellipse(ax, 100, 66, 120, 18, 0, c["cryosphere"], 0.22)
    add_ellipse(ax, 86, 32, 48, 14, 5, c["cryosphere"], 0.34)
    add_ellipse(ax, -70, -20, 18, 72, -8, c["cryosphere"], 0.27)
    add_ellipse(ax, 10, 46, 24, 9, 0, c["cryosphere"], 0.24)

    # Low-lying coasts, deltas, and small islands.
    for lon, lat, w, h, angle in [
        (90, 23, 15, 9, -20),
        (106, 13, 14, 9, -20),
        (31, 30, 13, 8, 0),
        (121, 31, 16, 7, 10),
        (113, 22, 14, 7, -10),
        (73, 3, 20, 8, 0),
        (-90, 29, 22, 7, 0),
    ]:
        add_ellipse(ax, lon, lat, w, h, angle, c["coast"], 0.34)

    # Coral reef systems.
    for lon, lat, w, h, angle in [
        (147, -18, 22, 10, -30),
        (124, 2, 34, 18, 5),
        (-76, 18, 30, 12, 0),
        (42, 20, 26, 8, -20),
        (60, -6, 34, 11, -10),
        (-150, -16, 44, 13, 0),
    ]:
        add_ellipse(ax, lon, lat, w, h, angle, c["reef"], 0.32)

    # Drylands and transition zones.
    for lon, lat, w, h, angle in [
        (15, 15, 70, 13, 0),
        (43, 27, 78, 24, 0),
        (70, 43, 70, 18, 0),
        (105, 43, 48, 14, 0),
        (-112, 35, 33, 17, 0),
        (132, -25, 48, 28, 0),
        (-68, -24, 22, 14, 0),
    ]:
        add_ellipse(ax, lon, lat, w, h, angle, c["dryland"], 0.27)

    # Tropical forests and peatlands.
    for lon, lat, w, h, angle in [
        (-62, -5, 52, 24, -10),
        (21, 0, 35, 18, 0),
        (108, 0, 34, 14, 0),
    ]:
        add_ellipse(ax, lon, lat, w, h, angle, c["forest"], 0.30)

    # Monsoon belts and monsoon margins.
    for lon, lat, w, h, angle in [
        (78, 21, 44, 25, 0),
        (112, 30, 43, 22, 0),
        (7, 10, 46, 18, 0),
    ]:
        add_ellipse(ax, lon, lat, w, h, angle, c["monsoon"], 0.20)

    label_positions = [
        (-142, 72, "1"),
        (-42, 72, "2"),
        (-97, -55, "3"),
        (86, 35, "4"),
        (-72, -17, "5"),
        (100, 22, "6"),
        (125, 0, "7"),
        (-62, -5, "8"),
        (20, 0, "9"),
        (15, 16, "10"),
        (69, 43, "11"),
        (105, 43, "12"),
        (90, 23, "13"),
        (121, 31, "14"),
        (147, -18, "15"),
    ]
    for lon, lat, text in label_positions:
        add_label(ax, lon, lat, text)


def add_side_notes(fig: plt.Figure, ax: plt.Axes) -> None:
    handles = [
        Patch(facecolor=COLORS["cryosphere"], alpha=0.45, label="Cryosphere / high mountains"),
        Patch(facecolor=COLORS["coast"], alpha=0.45, label="Low coasts, deltas, small islands"),
        Patch(facecolor=COLORS["reef"], alpha=0.45, label="Warm-water coral reefs"),
        Patch(facecolor=COLORS["dryland"], alpha=0.45, label="Drylands and transition zones"),
        Patch(facecolor=COLORS["forest"], alpha=0.45, label="Tropical forests and peatlands"),
        Patch(facecolor=COLORS["monsoon"], alpha=0.45, label="Monsoon belts and margins"),
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.015, 0.72),
        frameon=False,
        fontsize=7.2,
        handlelength=1.5,
    )

    notes = (
        "Numbered examples\n"
        "1 Arctic/permafrost belt\n"
        "2 Greenland Ice Sheet margin\n"
        "3 West Antarctic ice-sheet sector\n"
        "4 Tibetan Plateau / Himalaya\n"
        "5 Andes cryosphere\n"
        "6 South and East Asian monsoon\n"
        "7 Coral Triangle\n"
        "8 Amazon rainforest\n"
        "9 Congo Basin\n"
        "10 Sahel dryland transition\n"
        "11 Central Asian drylands\n"
        "12 Mongolia-North China margin\n"
        "13 Ganges-Brahmaputra delta\n"
        "14 Yangtze/Pearl delta coast\n"
        "15 Great Barrier Reef"
    )
    fig.text(
        0.765,
        0.43,
        notes,
        ha="left",
        va="top",
        fontsize=7.1,
        linespacing=1.23,
        color="#263238",
    )

    fig.text(
        0.076,
        0.07,
        "Shaded footprints are schematic, not jurisdictional boundaries; they mark broad regions with high sensitivity to warming, sea-level rise, hydrologic shifts, land degradation, or ecosystem tipping risk.",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#4B5563",
    )


def main() -> None:
    fig = plt.figure(figsize=(12.6, 6.4), constrained_layout=False)
    ax = fig.add_axes([0.055, 0.12, 0.68, 0.78])
    draw_base_map(ax)
    draw_sensitive_regions(ax)
    add_side_notes(fig, ax)

    fig.text(
        0.055,
        0.945,
        "Global Earth-System Sensitive Regions",
        ha="left",
        va="top",
        fontsize=13,
        weight="bold",
        color="#111827",
    )
    fig.text(
        0.055,
        0.905,
        "Broad hotspots where cryosphere, coastal, dryland, reef, forest, and monsoon systems respond strongly to environmental perturbations",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#374151",
    )

    for ext, kwargs in {
        "png": {"dpi": 600},
        "svg": {},
        "pdf": {},
    }.items():
        fig.savefig(OUT_BASE.with_suffix(f".{ext}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
