#!/usr/bin/env python3
"""Plot monthly Runge MLP-TM-EI gateway and mediator centers by variable."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULT_SUBDIR = ROOT / "results" / "runge_monthly_variable_comparison"
DEFAULT_OUTPUT = ROOT / "fig" / "runge_monthly_variable_comparison" / "gateway_mediator_centers.png"
LAND_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
COASTLINE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_coastline.geojson"

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
        if geometry.get("type") == "Polygon" and coordinates:
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
    ax.grid(color="#b8b8b8", linestyle="--", linewidth=0.35, alpha=0.72)
    for polygon in land:
        for segment in split_dateline(polygon):
            ax.fill(
                np.radians([point[0] for point in segment]),
                np.radians([point[1] for point in segment]),
                color="#e1e1dd",
                alpha=0.86,
                linewidth=0.0,
                zorder=1,
            )
    for line in coastlines:
        for segment in split_dateline(line):
            ax.plot(
                np.radians([point[0] for point in segment]),
                np.radians([point[1] for point in segment]),
                color="#6f6f6f",
                linewidth=0.30,
                alpha=0.78,
                zorder=2,
            )


def load_centers(result_root: Path, dataset: str, *, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    from scripts.run_runge_monthly_variable_comparison import component_centers_from_scores

    dataset_dir = result_root / dataset
    maps_payload = np.load(dataset_dir / "component_maps.npz")
    component_maps = maps_payload["component_maps"]
    lat = maps_payload["lat"]
    lon = maps_payload["lon"]
    pair_dir = dataset_dir / "mlp_tm_ei" / "results" / "runge" / "pairwise_mlp_tm_ei_path_effects"
    gateway = pd.read_csv(pair_dir / "gateway_scores.csv")
    mediator = pd.read_csv(pair_dir / "mediator_scores.csv")
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    label = manifest.get("dataset", {}).get("display_name") or dataset
    ace_centers = component_centers_from_scores(gateway, component_maps, lat, lon, metric="ace", top_k=top_k)
    acs_centers = component_centers_from_scores(gateway, component_maps, lat, lon, metric="acs", top_k=top_k)
    mediator_centers = component_centers_from_scores(mediator, component_maps, lat, lon, metric="amce", top_k=top_k)
    ace_centers["dataset"] = dataset
    acs_centers["dataset"] = dataset
    mediator_centers["dataset"] = dataset
    return ace_centers, acs_centers, mediator_centers, str(label)


def draw_centers(ax: plt.Axes, centers: pd.DataFrame, *, norm: mpl.colors.Normalize, cmap: mpl.colors.Colormap) -> None:
    if centers.empty:
        return
    scores = centers["score"].to_numpy(dtype=float)
    score_max = max(float(np.nanmax(scores)), 1.0e-12)
    sizes = 55.0 + 210.0 * np.sqrt(np.clip(scores / score_max, 0.0, 1.0))
    display_lon = np.clip(centers["lon"].to_numpy(dtype=float), -168.0, 168.0)
    ax.scatter(
        np.radians(display_lon),
        np.radians(centers["lat"].to_numpy(dtype=float)),
        s=sizes,
        c=scores,
        cmap=cmap,
        norm=norm,
        edgecolors="#1f2933",
        linewidths=0.38,
        alpha=0.95,
        zorder=5,
    )
    stroke = [pe.withStroke(linewidth=1.2, foreground="white")]
    for row in centers.itertuples(index=False):
        lon = float(np.clip(float(row.lon), -168.0, 168.0))
        lat = float(row.lat)
        dx = -5.0 if lon > 110.0 else 5.0
        ha = "right" if lon > 110.0 else "left"
        dy = -5.0 if lat > 50.0 else 5.0
        va = "top" if lat > 50.0 else "bottom"
        ax.annotate(
            str(row.label),
            xy=(np.radians(lon), np.radians(lat)),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=6.6,
            fontweight="bold",
            color="#111111",
            path_effects=stroke,
            zorder=6,
        )


def plot_monthly_variable_gateway_map(
    result_root: str | Path,
    output: str | Path,
    *,
    dataset_names: Sequence[str],
    top_k: int = 5,
    save_svg: bool = False,
) -> Path:
    root = Path(result_root)
    ace_frames: list[pd.DataFrame] = []
    acs_frames: list[pd.DataFrame] = []
    mediator_frames: list[pd.DataFrame] = []
    labels: list[str] = []
    for dataset in dataset_names:
        ace, acs, mediator, label = load_centers(root, dataset, top_k=int(top_k))
        ace_frames.append(ace)
        acs_frames.append(acs)
        mediator_frames.append(mediator)
        labels.append(label)

    ace_all = pd.concat(ace_frames, ignore_index=True) if ace_frames else pd.DataFrame({"score": []})
    acs_all = pd.concat(acs_frames, ignore_index=True) if acs_frames else pd.DataFrame({"score": []})
    mediator_all = pd.concat(mediator_frames, ignore_index=True) if mediator_frames else pd.DataFrame({"score": []})
    ace_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(1.0e-9, float(ace_all["score"].max())))
    acs_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(1.0e-9, float(acs_all["score"].max())))
    mediator_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(1.0e-9, float(mediator_all["score"].max())))
    ace_cmap = mpl.colormaps["OrRd"]
    acs_cmap = mpl.colormaps["PuBu"]
    mediator_cmap = mpl.colormaps["Greens"]
    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))

    n_rows = len(dataset_names)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(12.6, max(2.15 * n_rows, 3.2)),
        constrained_layout=True,
        subplot_kw={"projection": "mollweide"},
        squeeze=False,
    )
    for row_idx, label in enumerate(labels):
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            draw_world(ax, land, coastlines)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if col_idx == 0:
                ax.text(-0.08, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=8.2, fontweight="bold")
        draw_centers(axes[row_idx, 0], ace_frames[row_idx], norm=ace_norm, cmap=ace_cmap)
        draw_centers(axes[row_idx, 1], acs_frames[row_idx], norm=acs_norm, cmap=acs_cmap)
        draw_centers(axes[row_idx, 2], mediator_frames[row_idx], norm=mediator_norm, cmap=mediator_cmap)

    axes[0, 0].set_title(f"Top {top_k} gateways (ACE)", fontsize=9)
    axes[0, 1].set_title(f"Top {top_k} incoming effect (ACS)", fontsize=9)
    axes[0, 2].set_title(f"Top {top_k} mediators (AMCE)", fontsize=9)
    sm_ace = mpl.cm.ScalarMappable(norm=ace_norm, cmap=ace_cmap)
    sm_acs = mpl.cm.ScalarMappable(norm=acs_norm, cmap=acs_cmap)
    sm_mediator = mpl.cm.ScalarMappable(norm=mediator_norm, cmap=mediator_cmap)
    cbar_a = fig.colorbar(sm_ace, ax=axes[:, 0].ravel().tolist(), location="bottom", shrink=0.60, pad=0.025, aspect=26)
    cbar_a.set_label("ACE")
    cbar_a.locator = MaxNLocator(nbins=5)
    cbar_a.update_ticks()
    cbar_b = fig.colorbar(sm_acs, ax=axes[:, 1].ravel().tolist(), location="bottom", shrink=0.60, pad=0.025, aspect=26)
    cbar_b.set_label("ACS")
    cbar_b.locator = MaxNLocator(nbins=5)
    cbar_b.update_ticks()
    cbar_c = fig.colorbar(sm_mediator, ax=axes[:, 2].ravel().tolist(), location="bottom", shrink=0.60, pad=0.025, aspect=26)
    cbar_c.set_label("AMCE")
    cbar_c.locator = MaxNLocator(nbins=5)
    cbar_c.update_ticks()

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    if save_svg:
        fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=RESULT_SUBDIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", default="slp_monthly,t2m_monthly,air1000_monthly,sst_monthly")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save-svg", action="store_true")
    args = parser.parse_args()
    dataset_names = [part.strip() for part in args.datasets.split(",") if part.strip()]
    output = plot_monthly_variable_gateway_map(
        args.result_root,
        args.output,
        dataset_names=dataset_names,
        top_k=int(args.top_k),
        save_svg=bool(args.save_svg),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
