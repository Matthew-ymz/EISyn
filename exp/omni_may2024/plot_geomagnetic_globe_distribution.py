#!/usr/bin/env python3
"""May 2024 China geomagnetic disturbance as continuous fields on a globe.

Sources accessed 2026-10-09:
  Wang et al. (2026), Space Weather, doi:10.1029/2025SW004610.
  https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025SW004610
  Author's copy (used to inspect/digitize the published figures):
  https://www.swl.ac.cn/~hli/papers/Wang%20et%20al._2026_Space%20Weather.pdf
  Coordinates: Table 1; definitions: section 2.1; peaks: section 3.1,
  Figure 5 (PDF page 7) and Figure 8 (PDF page 9).

The source removes April 2024 quiet-day baselines and filters 1-second records
to 1-minute resolution. Here the first quantity is -min(B_h) in the main phase,
and the second is max(abs(dB_h/dt)) in the main phase. Station-specific maxima
need not occur simultaneously; this is an event summary, not a snapshot.
Most values were read from raster markers in the PDF. They are rounded to
5 nT / 1 nT per minute, except explicitly reported values (720, 538, 449 nT,
and 60, 32 nT/min), which retain their stated source precision. Close-latitude
markers were identified by their labels, not by latitude sorting alone.
These rounded inputs are visualization support, not raw magnetic time series.

The continuous field is piecewise linear interpolation in longitude/latitude
(longitude scaled by cos(36 degrees)). It is masked outside the station convex
hull, mainland-China land polygons, and the visible hemisphere. There is no
extrapolation, smoothing, or global disturbance estimate. No station glyphs
are drawn. Grey land indicates no displayed interpolation, not zero impact.
The method supplies a visual summary; it does not estimate forecast errors,
ionospheric disturbance, GIC, causal effects, or statistical uncertainty.

Background geography: Natural Earth 1:50m country polygons, stored in the
adjacent geomagnetic_map_context.geojson with its download URL and date.
Orthographic projection is implemented directly on a unit sphere centered at
105 E, 35 N. Background-only illumination leaves metric colors unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Circle, Polygon
from scipy.interpolate import LinearNDInterpolator
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "geomagnetic_storm_globe_distribution.png"
GEOGRAPHY = HERE / "geomagnetic_map_context.geojson"
CENTER = (105.0, 35.0)
REGION = (117.0, 125.0, 48.0, 54.0)

# Code, geographic latitude, longitude, -min(B_h) [nT], peak |dB_h/dt| [nT/min].
# Figure-read inputs are approximate; the docstring records retained exact values.
STATIONS = [
    ("OMOHE", 53.49, 122.34, 720, 60),
    ("OYAKS", 50.48, 121.69, 538, 60),
    ("OMZLN", 49.57, 117.45, 590, 55),
    ("OHUTB", 44.36, 86.94, 490, 52),
    ("ONOAN", 44.09, 124.91, 550, 43),
    ("OCSSL", 40.30, 116.19, 550, 49),
    ("OJIYG", 39.81, 98.22, 540, 36),
    ("ODLZS", 39.58, 121.77, 515, 36),
    ("OKSHI", 39.51, 75.81, 505, 50),
    ("OHBLF", 39.50, 116.70, 455, 43),
    ("OQIMO", 38.14, 85.45, 450, 43),
    ("OGERM", 36.43, 94.87, 485, 47),
    ("OTANC", 34.70, 118.46, 449, 47),
    ("OGAER", 32.52, 80.11, 450, 40),
    ("OYICH", 30.92, 113.34, 470, 44),
    ("OPIXI", 30.91, 103.76, 480, 41),
    ("OHZXH", 30.25, 120.11, 475, 41),
    ("OLSCG", 29.63, 91.04, 470, 39),
    ("OLJGC", 26.97, 100.18, 465, 46),
    ("ONANA", 25.02, 118.51, 470, 36),
    ("OZHQI", 23.05, 113.20, 485, 36),
    ("OQIZH", 19.03, 109.85, 495, 32),
    ("OLEDO", 18.44, 108.97, 495, 32),
]


def font() -> FontProperties:
    for path in [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]:
        if Path(path).exists():
            fontManager.addfont(path)
            return FontProperties(fname=path)
    return FontProperties(family="sans-serif")


CJK = font()
TEXT = "#1E3346"
MUTED = "#596D7D"
FRAME = "#177E8B"


def label(ax, x, y, text, size=12, **kwargs):
    kwargs.setdefault("color", TEXT)
    return ax.text(x, y, text, fontproperties=CJK, fontsize=size, **kwargs)


def project(lon, lat):
    lon, lat = np.broadcast_arrays(np.asarray(lon), np.asarray(lat))
    phi = np.deg2rad(lat)
    delta = np.deg2rad(lon - CENTER[0])
    phi0 = np.deg2rad(CENTER[1])
    x = np.cos(phi) * np.sin(delta)
    y = np.cos(phi0) * np.sin(phi) - np.sin(phi0) * np.cos(phi) * np.cos(delta)
    z = np.sin(phi0) * np.sin(phi) + np.cos(phi0) * np.cos(phi) * np.cos(delta)
    return x, y, z


def geographic_background(geography):
    """Rasterize geometry before spherical reprojection; preserve polygon holes."""
    fig = plt.figure(figsize=(24, 12), dpi=100, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="black")
    ax.set(xlim=(-180, 180), ylim=(-90, 90))
    ax.axis("off")
    for feature in geography["features"]:
        geometry = feature["geometry"]
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        is_china = feature["properties"]["iso"] == "CHN"
        for polygon in polygons:
            # Red channel = land, green channel = mainland-China mask.
            ax.add_patch(Polygon(polygon[0], facecolor=(1, float(is_china), 0), edgecolor="none", antialiased=False))
            for hole in polygon[1:]:
                ax.add_patch(Polygon(hole, facecolor="black", edgecolor="none", antialiased=False))
    fig.canvas.draw()
    raster = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return raster


def globe_raster(geography, n=1000):
    xy = np.linspace(-1.025, 1.025, n)
    x, y = np.meshgrid(xy, xy)
    valid = x * x + y * y <= 1
    z = np.sqrt(np.maximum(0, 1 - x * x - y * y))
    phi0 = np.deg2rad(CENTER[1])
    lat = np.rad2deg(np.arcsin(np.clip(y * np.cos(phi0) + z * np.sin(phi0), -1, 1)))
    lon = CENTER[0] + np.rad2deg(np.arctan2(x, z * np.cos(phi0) - y * np.sin(phi0)))
    lon = (lon + 180) % 360 - 180

    raster = geographic_background(geography)
    ix = np.clip(((lon + 180) / 360 * raster.shape[1]).astype(int), 0, raster.shape[1] - 1)
    iy = np.clip(((90 - lat) / 180 * raster.shape[0]).astype(int), 0, raster.shape[0] - 1)
    land = raster[iy, ix, 0] > 128
    china = raster[iy, ix, 1] > 128

    # A gentle spherical shading is applied only to the geographic background.
    light = np.clip(0.85 + 0.15 * z - 0.03 * x + 0.03 * y, 0.82, 1.0)
    rgb = np.where(land[..., None], np.array([0.87, 0.89, 0.91]), np.array([0.94, 0.97, 0.99]))
    rgb = rgb * light[..., None]
    background = np.concatenate([rgb, valid[..., None].astype(float)], axis=2)

    coords = np.array([(s[2], s[1]) for s in STATIONS])
    scale = np.array([np.cos(np.deg2rad(36)), 1.0])
    fields = []
    for column in (3, 4):
        values = np.array([s[column] for s in STATIONS], dtype=float)
        interp = LinearNDInterpolator(coords * scale, values, fill_value=np.nan)
        field = interp(np.stack([lon, lat], axis=-1) * scale)
        field[~valid | ~china] = np.nan
        # Linear interpolation must stay within the source range; do not clip it.
        if np.nanmin(field) < values.min() - 1e-8 or np.nanmax(field) > values.max() + 1e-8:
            raise ValueError("Interpolation left the observed range")
        fields.append(field)
    return background, fields


def geography_lines(ax, geography):
    for feature in geography["features"]:
        geometry = feature["geometry"]
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        china = feature["properties"]["iso"] == "CHN"
        for polygon in polygons:
            coords = np.asarray(polygon[0])
            x, y, z = project(coords[:, 0], coords[:, 1])
            x, y = np.where(z > 0.012, x, np.nan), np.where(z > 0.012, y, np.nan)
            ax.plot(x, y, color="#607887" if china else "#879AA7", lw=0.65 if china else 0.38, alpha=0.8, zorder=4)


def graticule(ax):
    for lat in np.arange(-60, 91, 30):
        lon = np.linspace(-180, 180, 1801)
        x, y, z = project(lon, lat)
        ax.plot(np.where(z > 0.012, x, np.nan), np.where(z > 0.012, y, np.nan), color="#91A7B5", alpha=0.46, lw=0.46, zorder=3)
    for lon in np.arange(-180, 180, 30):
        lat = np.linspace(-90, 90, 721)
        x, y, z = project(lon, lat)
        ax.plot(np.where(z > 0.012, x, np.nan), np.where(z > 0.012, y, np.nan), color="#91A7B5", alpha=0.46, lw=0.46, zorder=3)


def regional_frame(ax):
    west, east, south, north = REGION
    lon = np.concatenate([np.linspace(west, east, 80), np.full(80, east), np.linspace(east, west, 80), np.full(80, west)])
    lat = np.concatenate([np.full(80, south), np.linspace(south, north, 80), np.full(80, north), np.linspace(north, south, 80)])
    x, y, _ = project(lon, lat)
    ax.plot(x, y, color="white", lw=3.2, zorder=6)
    ax.plot(x, y, color=FRAME, lw=1.5, zorder=7)
    # The text stays outside the globe and any colored field.
    anchor = project(east, north)
    ax.annotate("", xy=anchor[:2], xytext=(0.64, 0.93), arrowprops={"arrowstyle": "-", "lw": 0.85, "color": FRAME}, zorder=7)
    label(ax, 0.48, 1.0, "东北重点区", size=12, ha="left", va="bottom", color=FRAME)
    label(ax, 0.48, 0.95, "48–54°N · 117–125°E", size=9.2, ha="left", va="bottom", color=MUTED)


def main():
    geography = json.loads(GEOGRAPHY.read_text())
    background, fields = globe_raster(geography)
    cmap_b = LinearSegmentedColormap.from_list("disturbance", ["#F0EAF6", "#C4ABDE", "#9871BE", "#664294", "#3D175E"])
    cmap_d = LinearSegmentedColormap.from_list("rate", ["#FFF0D3", "#F9C56A", "#EC923F", "#CE5A24", "#92341F"])
    specs = [
        ("a", "主相磁场下降幅度", cmap_b, Normalize(449, 720), [449, 500, 550, 600, 650, 720], "磁场下降幅度（nT）"),
        ("b", "最大磁场变化速率", cmap_d, Normalize(32, 60), [32, 40, 50, 60], "最大磁场变化速率（nT/min）"),
    ]
    with plt.rc_context({"font.family": CJK.get_name(), "font.size": 10, "axes.linewidth": 0.6}):
        fig = plt.figure(figsize=(13.6, 8.1), dpi=240, facecolor="white")
        grid = fig.add_gridspec(2, 2, height_ratios=[1, 0.042], left=0.035, right=0.97, top=0.83, bottom=0.20, hspace=0.10, wspace=0.065)
        fig.text(0.048, 0.948, "2024年5月地磁暴的空间分布", fontproperties=CJK, fontsize=22, color=TEXT, va="top")
        fig.text(0.048, 0.896, "2024-05-10–12  ·  中国子午工程观测  ·  地球视图", fontproperties=CJK, fontsize=11, color=MUTED, va="top")
        for i, (letter, title, cmap, norm, ticks, unit) in enumerate(specs):
            ax = fig.add_subplot(grid[0, i])
            ax.set(xlim=(-1.10, 1.10), ylim=(-1.03, 1.12), aspect="equal")
            ax.axis("off")
            ax.imshow(background, extent=(-1.025, 1.025, -1.025, 1.025), origin="lower", interpolation="bilinear", zorder=1)
            color = cmap(norm(np.nan_to_num(fields[i], nan=norm.vmin)))
            color[..., 3] = np.isfinite(fields[i]).astype(float)
            ax.imshow(color, extent=(-1.025, 1.025, -1.025, 1.025), origin="lower", interpolation="nearest", zorder=2)
            graticule(ax)
            geography_lines(ax, geography)
            ax.add_patch(Circle((0, 0), 1, facecolor="none", edgecolor="#8099A9", lw=0.85, zorder=5))
            regional_frame(ax)
            ax.set_title(title, loc="left", fontproperties=CJK, fontsize=13, color=TEXT, pad=13)
            ax.text(-0.065, 1.075, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", color=TEXT, ha="left", va="top")
            cax = fig.add_subplot(grid[1, i])
            # Keep the guide outside the sphere, at its own reserved location.
            pos = cax.get_position()
            cax.set_position([pos.x0 + pos.width * 0.12, pos.y0, pos.width * 0.76, pos.height])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=ticks)
            cb.outline.set_visible(False)
            cb.ax.tick_params(length=3, labelsize=9, color=MUTED, labelcolor=MUTED)
            cb.set_label(unit, fontproperties=CJK, fontsize=11, color=TEXT, labelpad=8)
        fig.text(0.048, 0.090, "有色区域：23站主相峰值的线性空间插值；灰色陆地：无插值覆盖。各地峰值并非同一时刻。", fontproperties=CJK, fontsize=9.2, color=MUTED)
        fig.text(0.048, 0.059, "峰值据论文原图近似读取；未估计天气预报误差。  数据：Wang et al., Space Weather (2026), 10.1029/2025SW004610", fontproperties=CJK, fontsize=8.4, color=MUTED)
        fig.savefig(OUT, dpi=240, facecolor="white", bbox_inches="tight", pad_inches=0.16)
        plt.close(fig)
    print(f"figure: {OUT}")
    print("23 source locations; station glyphs: 0; convex-hull linear interpolation; no extrapolation")
    print("displayed finite pixels:", [int(np.isfinite(field).sum()) for field in fields])


if __name__ == "__main__":
    main()
