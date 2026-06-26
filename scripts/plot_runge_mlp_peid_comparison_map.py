#!/usr/bin/env python3
"""Plot pairwise MLP-TM-EI and PEID second-order Runge maps together."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_runge_gateway_mediator_map import (
    COASTLINE_URL,
    DEFAULT_COMPONENT_MAPS,
    LAND_URL,
    add_geographic_ticks,
    add_labels,
    component_center,
    draw_world,
    extract_lines,
    extract_polygons,
    load_geojson,
    local_to_paper,
)
from plot_runge_linear_mlp_peid_map import build_mlp_peid_node_frame


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRWISE_GATEWAY = ROOT / "results" / "runge" / "pairwise_mlp_tm_ei_path_effects" / "gateway_scores.csv"
DEFAULT_PAIRWISE_MEDIATOR = ROOT / "results" / "runge" / "pairwise_mlp_tm_ei_path_effects" / "mediator_scores.csv"
DEFAULT_PEID_GATEWAY = ROOT / "results" / "runge" / "peid_hypergraph" / "hyper_gateway_scores.csv"
DEFAULT_PEID_MEDIATOR = ROOT / "results" / "runge" / "peid_hypergraph" / "hyper_mediator_scores.csv"
DEFAULT_HYPEREDGES = ROOT / "results" / "runge" / "peid_hypergraph" / "peid_hyperedges.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "reports" / "assets" / "part2_runge_mlp_peid_comparison_map.png"


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


def build_pairwise_node_frame(component_maps: np.ndarray, gateway_path: Path, mediator_path: Path) -> pd.DataFrame:
    lat = np.linspace(-90.0, 90.0, component_maps.shape[0])
    lon = np.linspace(0.0, 360.0, component_maps.shape[1], endpoint=False)
    lon = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    maps = component_maps[:, order, :]
    lon = lon[order]

    gateway = pd.read_csv(gateway_path).copy()
    mediator = pd.read_csv(mediator_path).copy()
    frame = gateway.merge(mediator[["component_index", "amce", "mediated_fraction"]], on="component_index", how="left")

    rows: list[dict[str, float | int]] = []
    for row in frame.itertuples(index=False):
        local = int(row.component_index)
        center_lon, center_lat = component_center(maps[..., local], lat, lon)
        rows.append(
            {
                "local": local,
                "paper": local_to_paper(local),
                "lon": center_lon,
                "lat": center_lat,
                "ace": float(row.ace),
                "acs": float(row.acs),
                "amce": float(row.amce),
                "mediated_fraction": float(row.mediated_fraction),
            }
        )
    return pd.DataFrame(rows)


def draw_ace_acs(ax: plt.Axes, nodes: pd.DataFrame, norm: mpl.colors.Normalize) -> None:
    cmap = mpl.colormaps["OrRd"]
    lon = np.radians(nodes["lon"].to_numpy())
    lat = np.radians(nodes["lat"].to_numpy())
    ax.scatter(
        lon,
        lat,
        s=360,
        c=nodes["ace"],
        cmap=cmap,
        norm=norm,
        edgecolors="#3d1d0d",
        linewidths=0.32,
        alpha=0.96,
        zorder=4,
    )
    ax.scatter(
        lon,
        lat,
        s=190,
        c=nodes["acs"],
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        alpha=0.98,
        zorder=5,
    )
    add_labels(ax, nodes)


def draw_amce(ax: plt.Axes, nodes: pd.DataFrame, norm: mpl.colors.Normalize) -> None:
    cmap = mpl.colormaps["Greens"]
    values = nodes["amce"].to_numpy(dtype=float)
    sizes = 185.0 + 360.0 * np.clip(values / max(float(np.nanmax(values)), 1.0e-12), 0.0, 1.0)
    ax.scatter(
        np.radians(nodes["lon"].to_numpy()),
        np.radians(nodes["lat"].to_numpy()),
        s=sizes,
        c=values,
        cmap=cmap,
        norm=norm,
        edgecolors="#173317",
        linewidths=0.32,
        alpha=0.96,
        zorder=4,
    )
    add_labels(ax, nodes)


def add_colorbar(fig: plt.Figure, ax: plt.Axes, norm: mpl.colors.Normalize, cmap_name: str, label: str) -> None:
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colormaps[cmap_name])
    cbar = fig.colorbar(sm, ax=ax, location="bottom", shrink=0.68, pad=0.07, aspect=24)
    cbar.set_label(label)


def plot_map(pairwise_nodes: pd.DataFrame, peid_nodes: pd.DataFrame, output: Path, *, save_svg: bool) -> Path:
    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    fig = plt.figure(figsize=(10.6, 8.0), constrained_layout=True)
    axes = [
        fig.add_subplot(2, 2, 1, projection="mollweide"),
        fig.add_subplot(2, 2, 2, projection="mollweide"),
        fig.add_subplot(2, 2, 3, projection="mollweide"),
        fig.add_subplot(2, 2, 4, projection="mollweide"),
    ]
    for ax in axes:
        draw_world(ax, land, coastlines)
        add_geographic_ticks(ax)

    pair_ace_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(0.005, float(pairwise_nodes[["ace", "acs"]].to_numpy().max())))
    pair_amce_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(1.8e-5, float(pairwise_nodes["amce"].max())))
    peid_ace_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(0.018, float(peid_nodes[["ace", "acs"]].to_numpy().max())))
    peid_amce_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(0.00095, float(peid_nodes["amce"].max())))

    draw_ace_acs(axes[0], pairwise_nodes, pair_ace_norm)
    draw_amce(axes[1], pairwise_nodes, pair_amce_norm)
    draw_ace_acs(axes[2], peid_nodes, peid_ace_norm)
    draw_amce(axes[3], peid_nodes, peid_amce_norm)

    panel_labels = ["a", "b", "c", "d"]
    titles = [
        "Without second-order synergy",
        "Without second-order synergy",
        "With second-order PEID",
        "With second-order PEID",
    ]
    for label, title, ax in zip(panel_labels, titles, axes, strict=True):
        ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=16, fontweight="bold")
        ax.text(0.5, 1.06, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")

    add_colorbar(fig, axes[0], pair_ace_norm, "OrRd", "path ACS (inner node) and path ACE (outer ring)")
    add_colorbar(fig, axes[1], pair_amce_norm, "Greens", "path AMCE")
    add_colorbar(fig, axes[2], peid_ace_norm, "OrRd", "Hyper-ACS (inner node) and Hyper-ACE (outer ring)")
    add_colorbar(fig, axes[3], peid_amce_norm, "Greens", "Hyper-AMCE")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if save_svg:
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-maps", default=str(DEFAULT_COMPONENT_MAPS))
    parser.add_argument("--pairwise-gateway-scores", default=str(DEFAULT_PAIRWISE_GATEWAY))
    parser.add_argument("--pairwise-mediator-scores", default=str(DEFAULT_PAIRWISE_MEDIATOR))
    parser.add_argument("--peid-gateway-scores", default=str(DEFAULT_PEID_GATEWAY))
    parser.add_argument("--peid-mediator-scores", default=str(DEFAULT_PEID_MEDIATOR))
    parser.add_argument("--hyperedges", default=str(DEFAULT_HYPEREDGES))
    parser.add_argument("--significance-z", type=float, default=2.0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--save-svg", action="store_true")
    args = parser.parse_args()

    maps = np.load(Path(args.component_maps).expanduser())["component_maps"]
    pairwise_nodes = build_pairwise_node_frame(
        maps,
        Path(args.pairwise_gateway_scores).expanduser(),
        Path(args.pairwise_mediator_scores).expanduser(),
    )
    peid_nodes = build_mlp_peid_node_frame(
        maps,
        Path(args.peid_gateway_scores).expanduser(),
        Path(args.peid_mediator_scores).expanduser(),
        Path(args.hyperedges).expanduser(),
        significance_z=float(args.significance_z),
    )
    output = plot_map(pairwise_nodes, peid_nodes, Path(args.output).expanduser(), save_svg=bool(args.save_svg))
    print(output)


if __name__ == "__main__":
    main()
