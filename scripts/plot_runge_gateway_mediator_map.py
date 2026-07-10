#!/usr/bin/env python3
"""Plot Runge component nodes on a global gateway/mediator map."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPONENT_MAPS = (
    ROOT / "results" / "runge_slp_daily_1948_2026_20260628" / "results" / "runge" / "2015_gateways" / "component_maps.npz"
)
DEFAULT_GATEWAY = ROOT / "results" / "runge" / "2015_gateways" / "gateway_scores.csv"
DEFAULT_MEDIATOR = ROOT / "results" / "runge" / "2015_gateways" / "mediator_scores.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "reports" / "assets" / "part2_runge_gateway_mediator_map.png"
COASTLINE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_coastline.geojson"
LAND_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
PAPER_TO_LOCAL = {18: 7, 26: 8, 48: 21, 7: 18, 8: 26, 21: 48}


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


def local_to_paper(local_index: int) -> int:
    return int(PAPER_TO_LOCAL.get(int(local_index), int(local_index)))


def load_geojson(url: str, timeout: float = 8.0) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(url, timeout=float(timeout)) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def extract_lines(data: dict[str, object] | None) -> list[list[tuple[float, float]]]:
    if not data:
        return []
    lines: list[list[tuple[float, float]]] = []
    for feature in data.get("features", []):  # type: ignore[union-attr]
        geometry = feature.get("geometry") or {}
        if geometry.get("type") == "LineString":
            lines.append([(float(x), float(y)) for x, y in geometry.get("coordinates", [])])
        elif geometry.get("type") == "MultiLineString":
            for segment in geometry.get("coordinates", []):
                lines.append([(float(x), float(y)) for x, y in segment])
    return lines


def extract_polygons(data: dict[str, object] | None) -> list[list[tuple[float, float]]]:
    if not data:
        return []
    polygons: list[list[tuple[float, float]]] = []
    for feature in data.get("features", []):  # type: ignore[union-attr]
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates", [])
        if geometry.get("type") == "Polygon":
            if coordinates:
                polygons.append([(float(x), float(y)) for x, y in coordinates[0]])
        elif geometry.get("type") == "MultiPolygon":
            for polygon in coordinates:
                if polygon:
                    polygons.append([(float(x), float(y)) for x, y in polygon[0]])
    return polygons


def split_dateline(points: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    if len(points) < 2:
        return [points]
    segments: list[list[tuple[float, float]]] = [[]]
    previous_lon: float | None = None
    for lon, lat in points:
        lon = ((float(lon) + 180.0) % 360.0) - 180.0
        if previous_lon is not None and abs(lon - previous_lon) > 180.0:
            segments.append([])
        segments[-1].append((lon, float(lat)))
        previous_lon = lon
    return [segment for segment in segments if len(segment) >= 2]


def draw_world(ax: plt.Axes, land: Iterable[list[tuple[float, float]]], coastlines: Iterable[list[tuple[float, float]]]) -> None:
    ax.set_facecolor("white")
    ax.grid(color="#b8b8b8", linestyle="--", linewidth=0.38, alpha=0.75)
    for polygon in land:
        for segment in split_dateline(polygon):
            lon = np.radians([point[0] for point in segment])
            lat = np.radians([point[1] for point in segment])
            ax.fill(lon, lat, color="#d9d9d9", alpha=0.82, linewidth=0.0, zorder=1)
    for line in coastlines:
        for segment in split_dateline(line):
            lon = np.radians([point[0] for point in segment])
            lat = np.radians([point[1] for point in segment])
            ax.plot(lon, lat, color="#6f6f6f", linewidth=0.32, alpha=0.72, zorder=2)


def format_lon_label(degrees: float) -> str:
    value = int(abs(round(float(degrees))))
    if value == 0:
        return "0°"
    suffix = "E" if degrees > 0 else "W"
    return f"{value}°{suffix}"


def format_lat_label(degrees: float) -> str:
    value = int(abs(round(float(degrees))))
    if value == 0:
        return "0°"
    suffix = "N" if degrees > 0 else "S"
    return f"{value}°{suffix}"


def add_geographic_ticks(ax: plt.Axes) -> None:
    lon_ticks = np.arange(-120, 121, 60)
    lat_ticks = np.arange(-60, 61, 30)
    ax.set_xticks(np.radians(lon_ticks))
    ax.set_yticks(np.radians(lat_ticks))
    ax.set_xticklabels([])
    ax.set_yticklabels([format_lat_label(deg) for deg in lat_ticks], fontsize=6, color="#4a4a4a")
    ax.tick_params(axis="both", which="major", pad=1.5, length=2.0, width=0.35, colors="#4a4a4a")
    ax.tick_params(axis="x", length=0.0)
    projection_lat = np.radians(-67.0)
    for deg in lon_ticks:
        display_x, _ = ax.transData.transform((np.radians(float(deg)), projection_lat))
        axes_x, _ = ax.transAxes.inverted().transform((display_x, 0.0))
        ax.plot(
            [axes_x, axes_x],
            [-0.014, -0.034],
            transform=ax.transAxes,
            color="#4a4a4a",
            linewidth=0.35,
            clip_on=False,
            zorder=10,
        )
        ax.text(
            axes_x,
            -0.052,
            format_lon_label(float(deg)),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6,
            color="#4a4a4a",
            clip_on=False,
        )


def sign_normalized_map(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    idx = np.unravel_index(np.nanargmax(np.abs(arr)), arr.shape)
    sign = 1.0 if arr[idx] >= 0.0 else -1.0
    return sign * arr


def component_center(values: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[float, float]:
    arr = sign_normalized_map(values)
    positive = np.maximum(arr, 0.0)
    threshold = max(float(np.nanpercentile(positive, 97.5)), 0.58 * float(np.nanmax(positive)))
    weights = np.where(positive >= threshold, positive, 0.0)
    if float(np.sum(weights)) <= 0.0:
        idx = np.unravel_index(np.nanargmax(arr), arr.shape)
        return float(lon[idx[1]]), float(lat[idx[0]])
    lon_rad = np.radians(lon)
    lon_weights = np.sum(weights, axis=0)
    x = float(np.sum(lon_weights * np.cos(lon_rad)))
    y = float(np.sum(lon_weights * np.sin(lon_rad)))
    center_lon = float(np.degrees(np.arctan2(y, x)))
    lat_weights = np.sum(weights, axis=1)
    center_lat = float(np.sum(lat_weights * lat) / np.sum(lat_weights))
    return center_lon, center_lat


def build_node_frame(component_maps: np.ndarray, gateway_path: Path, mediator_path: Path) -> pd.DataFrame:
    lat = np.linspace(-90.0, 90.0, component_maps.shape[0])
    lon = np.linspace(0.0, 360.0, component_maps.shape[1], endpoint=False)
    lon = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    maps = component_maps[:, order, :]
    lon = lon[order]

    gateway = pd.read_csv(gateway_path)
    mediator = pd.read_csv(mediator_path)
    mediator = mediator.rename(columns={"amce": "amce"})
    rows: list[dict[str, float | int]] = []
    for _, row in gateway.iterrows():
        local = int(row["component"])
        paper = int(row.get("paper_component", local_to_paper(local)))
        center_lon, center_lat = component_center(maps[..., local], lat, lon)
        med_row = mediator[mediator["component"].astype(int) == local]
        amce = float(med_row["amce"].iloc[0]) if len(med_row) else 0.0
        mediated_fraction = float(med_row["mediated_fraction"].iloc[0]) if len(med_row) and "mediated_fraction" in med_row else 0.0
        rows.append(
            {
                "local": local,
                "paper": paper,
                "lon": center_lon,
                "lat": center_lat,
                "ace": float(row["ace"]),
                "acs": float(row["acs"]),
                "amce": amce,
                "mediated_fraction": mediated_fraction,
            }
        )
    return pd.DataFrame(rows)


def draw_panel_a(ax: plt.Axes, nodes: pd.DataFrame, norm: mpl.colors.Normalize, cmap: mpl.colors.Colormap) -> None:
    lon = np.radians(nodes["lon"].to_numpy())
    lat = np.radians(nodes["lat"].to_numpy())
    ax.scatter(lon, lat, s=360, c=nodes["ace"], cmap=cmap, norm=norm, edgecolors="#3d1d0d", linewidths=0.32, alpha=0.96, zorder=4)
    ax.scatter(lon, lat, s=190, c=nodes["acs"], cmap=cmap, norm=norm, edgecolors="none", alpha=0.98, zorder=5)
    add_labels(ax, nodes)


def draw_panel_b(ax: plt.Axes, nodes: pd.DataFrame, norm: mpl.colors.Normalize, cmap: mpl.colors.Colormap) -> None:
    sizes = 185.0 + 360.0 * np.clip(nodes["mediated_fraction"].to_numpy(), 0.0, 1.0)
    ax.scatter(
        np.radians(nodes["lon"].to_numpy()),
        np.radians(nodes["lat"].to_numpy()),
        s=sizes,
        c=nodes["amce"],
        cmap=cmap,
        norm=norm,
        edgecolors="#173317",
        linewidths=0.32,
        alpha=0.96,
        zorder=4,
    )
    add_labels(ax, nodes)


def add_labels(ax: plt.Axes, nodes: pd.DataFrame) -> None:
    stroke = [pe.withStroke(linewidth=1.15, foreground="white")]
    coords = np.column_stack((np.radians(nodes["lon"].to_numpy()), np.radians(nodes["lat"].to_numpy())))
    offsets = label_offsets_points(ax, coords, nodes["paper"].to_numpy())
    for idx, row in enumerate(nodes.itertuples(index=False)):
        ax.annotate(
            str(int(row.paper)),
            xy=(np.radians(float(row.lon)), np.radians(float(row.lat))),
            xytext=(float(offsets[idx, 0]), float(offsets[idx, 1])),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=7.1,
            fontweight="bold",
            color="black",
            path_effects=stroke,
            clip_on=False,
            zorder=6,
        )


def label_offsets_points(ax: plt.Axes, coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Repel text labels in screen space while keeping bubbles fixed."""
    fig = ax.figure
    fig.canvas.draw()
    base = ax.transData.transform(coords)
    pos = base.copy()
    bbox = ax.get_window_extent()
    margin = 9.0
    pos[:, 0] = np.clip(pos[:, 0], bbox.x0 + margin, bbox.x1 - margin)
    pos[:, 1] = np.clip(pos[:, 1], bbox.y0 + margin, bbox.y1 - margin)

    widths = np.asarray([12.0 + 8.0 * len(str(int(label))) for label in labels], dtype=float)
    heights = np.full(len(labels), 14.0, dtype=float)
    for _ in range(90):
        moved = False
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                min_dx = 0.5 * (widths[i] + widths[j]) + 5.0
                min_dy = 0.5 * (heights[i] + heights[j]) + 3.0
                overlap_x = min_dx - abs(dx)
                overlap_y = min_dy - abs(dy)
                if overlap_x <= 0 or overlap_y <= 0:
                    continue
                if overlap_x < overlap_y:
                    sign = 1.0 if dx >= 0.0 else -1.0
                    shift = 0.5 * overlap_x + 0.2
                    pos[i, 0] -= sign * shift
                    pos[j, 0] += sign * shift
                else:
                    sign = 1.0 if dy >= 0.0 else -1.0
                    shift = 0.5 * overlap_y + 0.2
                    pos[i, 1] -= sign * shift
                    pos[j, 1] += sign * shift
                moved = True
        pos[:, 0] = np.clip(pos[:, 0], bbox.x0 + margin, bbox.x1 - margin)
        pos[:, 1] = np.clip(pos[:, 1], bbox.y0 + margin, bbox.y1 - margin)
        if not moved:
            break

    return (pos - base) * 72.0 / float(fig.dpi)


def plot_gateway_mediator_map(nodes: pd.DataFrame, output: Path, save_svg: bool = False) -> Path:
    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    fig = plt.figure(figsize=(10.4, 4.25), constrained_layout=True)
    axes = [fig.add_subplot(1, 2, 1, projection="mollweide"), fig.add_subplot(1, 2, 2, projection="mollweide")]
    for ax in axes:
        draw_world(ax, land, coastlines)
        add_geographic_ticks(ax)

    red_cmap = mpl.colormaps["OrRd"]
    green_cmap = mpl.colormaps["Greens"]
    red_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(0.06, float(nodes[["ace", "acs"]].to_numpy().max())))
    green_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(0.0015, float(nodes["amce"].max())))
    draw_panel_a(axes[0], nodes, red_norm, red_cmap)
    draw_panel_b(axes[1], nodes, green_norm, green_cmap)

    axes[0].text(-0.08, 1.06, "a", transform=axes[0].transAxes, fontsize=16, fontweight="bold")
    axes[1].text(-0.08, 1.06, "b", transform=axes[1].transAxes, fontsize=16, fontweight="bold")
    sm_red = mpl.cm.ScalarMappable(norm=red_norm, cmap=red_cmap)
    sm_green = mpl.cm.ScalarMappable(norm=green_norm, cmap=green_cmap)
    cbar_a = fig.colorbar(sm_red, ax=axes[0], location="bottom", shrink=0.53, pad=0.08, aspect=22)
    cbar_a.set_label("ACS (inner node) and ACE (outer ring)")
    cbar_b = fig.colorbar(sm_green, ax=axes[1], location="bottom", shrink=0.53, pad=0.08, aspect=22)
    cbar_b.set_label("AMCE")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if save_svg:
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-maps", default=str(DEFAULT_COMPONENT_MAPS))
    parser.add_argument("--gateway-scores", default=str(DEFAULT_GATEWAY))
    parser.add_argument("--mediator-scores", default=str(DEFAULT_MEDIATOR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--save-svg", action="store_true", help="Also export SVG; this is large because the map uses land polygons.")
    args = parser.parse_args()

    maps = np.load(Path(args.component_maps).expanduser())["component_maps"]
    nodes = build_node_frame(maps, Path(args.gateway_scores).expanduser(), Path(args.mediator_scores).expanduser())
    output = plot_gateway_mediator_map(nodes, Path(args.output).expanduser(), save_svg=bool(args.save_svg))
    print(output)


if __name__ == "__main__":
    main()
