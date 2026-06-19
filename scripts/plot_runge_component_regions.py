#!/usr/bin/env python3
"""Plot selected Runge Varimax component regions on global maps."""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPONENT_MAPS = ROOT / "results" / "runge" / "2015_gateways" / "component_maps.npz"
DEFAULT_GATEWAY_SCORES = ROOT / "results" / "runge" / "2015_gateways" / "gateway_scores.csv"
DEFAULT_MEDIATOR_SCORES = ROOT / "results" / "runge" / "2015_gateways" / "mediator_scores.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "reports" / "assets" / "part2_runge_component_regions.png"
DEFAULT_TOP_PER_METRIC = 5
PAPER_TO_LOCAL = {18: 7, 26: 8, 48: 21, 7: 18, 8: 26, 21: 48}
COASTLINE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_coastline.geojson"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.55,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def parse_nodes(value: str | None) -> list[int]:
    if not value:
        return []
    nodes = []
    for part in value.split(","):
        part = part.strip().replace("No.", "").replace("no.", "")
        if part:
            nodes.append(int(part))
    return nodes


def _append_ranked_nodes(
    selected: list[int],
    rows: list[dict[str, float | int]],
    metric: str,
    top_per_metric: int,
) -> None:
    ranked = sorted(rows, key=lambda row: float(row[metric]), reverse=True)
    for row in ranked[:top_per_metric]:
        node = int(row["paper_component"])
        if node not in selected:
            selected.append(node)


def select_top_nodes(gateway_path: Path, mediator_path: Path, top_per_metric: int) -> list[int]:
    rows: list[dict[str, float | int]] = []
    with gateway_path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "paper_component": int(raw["paper_component"]),
                    "ace": float(raw["ace"]),
                    "acs": float(raw["acs"]),
                }
            )

    selected: list[int] = []
    for metric in ("ace", "acs"):
        _append_ranked_nodes(selected, rows, metric, top_per_metric)

    mediator_rows: list[dict[str, float | int]] = []
    with mediator_path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            mediator_rows.append(
                {
                    "paper_component": int(raw["paper_component"]),
                    "amce": float(raw["amce"]),
                }
            )
    _append_ranked_nodes(selected, mediator_rows, "amce", top_per_metric)
    return selected


def paper_to_local(node: int) -> int:
    return int(PAPER_TO_LOCAL.get(int(node), int(node)))


def load_coastlines(timeout: float = 8.0) -> list[list[tuple[float, float]]]:
    try:
        with urllib.request.urlopen(COASTLINE_URL, timeout=float(timeout)) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []

    lines: list[list[tuple[float, float]]] = []
    for feature in data.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") == "LineString":
            lines.append([(float(x), float(y)) for x, y in geometry.get("coordinates", [])])
        elif geometry.get("type") == "MultiLineString":
            for segment in geometry.get("coordinates", []):
                lines.append([(float(x), float(y)) for x, y in segment])
    return lines


def roll_to_180(component_maps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_lat, n_lon, _ = component_maps.shape
    lat = np.linspace(-90.0, 90.0, n_lat)
    lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon180)
    return lat, lon180[order], component_maps[:, order, :]


def sign_normalized_map(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    idx = np.unravel_index(np.nanargmax(np.abs(arr)), arr.shape)
    sign = 1.0 if arr[idx] >= 0.0 else -1.0
    return sign * arr


def draw_coastlines(ax: plt.Axes, coastlines: Iterable[list[tuple[float, float]]]) -> None:
    for line in coastlines:
        if len(line) < 2:
            continue
        xs = np.asarray([point[0] for point in line], dtype=float)
        ys = np.asarray([point[1] for point in line], dtype=float)
        jumps = np.where(np.abs(np.diff(xs)) > 180.0)[0] + 1
        for xseg, yseg in zip(np.split(xs, jumps), np.split(ys, jumps)):
            ax.plot(xseg, yseg, color="#4c4c4c", linewidth=0.32, alpha=0.7, zorder=3)


def plot_component_regions(component_maps: np.ndarray, nodes: list[int], output: Path) -> Path:
    lat, lon, maps = roll_to_180(component_maps)
    selected = [sign_normalized_map(maps[..., paper_to_local(node)]) for node in nodes]
    vlim = float(np.nanpercentile(np.abs(np.stack(selected, axis=-1)), 99.2))
    vlim = max(vlim, 1.0e-9)
    coastlines = load_coastlines()

    n_cols = 3 if len(nodes) <= 9 else 5
    n_rows = int(np.ceil(len(nodes) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.12 * n_cols + 0.7, 3.1 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    last_image = None
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    for ax in axes.ravel():
        ax.set_visible(False)

    for index, (node, values) in enumerate(zip(nodes, selected)):
        ax = axes.ravel()[index]
        ax.set_visible(True)
        last_image = ax.imshow(
            values,
            extent=(-180.0, 180.0, -90.0, 90.0),
            origin="lower",
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            interpolation="bilinear",
            aspect="auto",
            zorder=1,
        )
        draw_coastlines(ax, coastlines)
        threshold = 0.62 * float(np.nanmax(values))
        if np.isfinite(threshold) and threshold > 0:
            mask = np.ma.masked_where(values < threshold, values)
            ax.imshow(
                mask,
                extent=(-180.0, 180.0, -90.0, 90.0),
                origin="lower",
                cmap=mpl.colors.ListedColormap(["#111111"]),
                alpha=0.16,
                interpolation="nearest",
                aspect="auto",
                zorder=4,
            )
        peak = np.unravel_index(np.nanargmax(values), values.shape)
        peak_lon = float(lon[peak[1]])
        peak_lat = float(lat[peak[0]])
        ax.scatter([peak_lon], [peak_lat], s=13, color="#111111", edgecolor="white", linewidth=0.35, zorder=5)
        ax.text(
            0.02,
            0.95,
            f"No.{node}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            zorder=6,
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-80, 80)
        ax.set_xticks([-120, 0, 120])
        ax.set_yticks([-60, 0, 60])
        ax.tick_params(labelsize=5.5, length=1.5, width=0.45)
        ax.grid(color="#9d9d9d", linewidth=0.25, alpha=0.35, zorder=2)

    if last_image is not None:
        colorbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), location="right", shrink=0.82, pad=0.01)
        colorbar.set_label("sign-normalized rotated loading")
        colorbar.ax.tick_params(labelsize=6)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-maps", default=str(DEFAULT_COMPONENT_MAPS))
    parser.add_argument("--nodes", default=None, help="Comma-separated paper labels, e.g. No.0,No.33,No.59")
    parser.add_argument("--gateway-scores", default=str(DEFAULT_GATEWAY_SCORES))
    parser.add_argument("--mediator-scores", default=str(DEFAULT_MEDIATOR_SCORES))
    parser.add_argument("--top-per-metric", type=int, default=DEFAULT_TOP_PER_METRIC)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    data = np.load(Path(args.component_maps).expanduser())["component_maps"]
    nodes = parse_nodes(args.nodes)
    if not nodes:
        nodes = select_top_nodes(
            Path(args.gateway_scores).expanduser(),
            Path(args.mediator_scores).expanduser(),
            args.top_per_metric,
        )
    output = plot_component_regions(data, nodes, Path(args.output).expanduser())
    print(output)


if __name__ == "__main__":
    main()
