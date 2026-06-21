#!/usr/bin/env python3
"""Plot linear Runge and MLP+PEID gateway/mediator maps together."""

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
    DEFAULT_GATEWAY,
    DEFAULT_MEDIATOR,
    LAND_URL,
    add_geographic_ticks,
    add_labels,
    build_node_frame as build_linear_node_frame,
    component_center,
    draw_world,
    extract_lines,
    extract_polygons,
    load_geojson,
    local_to_paper,
)
from plot_runge_peid_synergy_map import aggregate_hyper_acs


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MLP_GATEWAY = ROOT / "results" / "runge" / "peid_hypergraph" / "hyper_gateway_scores.csv"
DEFAULT_MLP_MEDIATOR = ROOT / "results" / "runge" / "peid_hypergraph" / "hyper_mediator_scores.csv"
DEFAULT_HYPEREDGES = ROOT / "results" / "runge" / "peid_hypergraph" / "peid_hyperedges.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "reports" / "assets" / "part2_runge_linear_mlp_peid_map.png"


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


def build_mlp_peid_node_frame(
    component_maps: np.ndarray,
    gateway_path: Path,
    mediator_path: Path,
    hyperedges_path: Path,
    *,
    significance_z: float,
) -> pd.DataFrame:
    lat = np.linspace(-90.0, 90.0, component_maps.shape[0])
    lon = np.linspace(0.0, 360.0, component_maps.shape[1], endpoint=False)
    lon = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    maps = component_maps[:, order, :]
    lon = lon[order]

    gateway = pd.read_csv(gateway_path).copy()
    mediator = pd.read_csv(mediator_path).copy()
    hyperedges = pd.read_csv(hyperedges_path)
    acs = aggregate_hyper_acs(
        hyperedges,
        n_components=component_maps.shape[2],
        significance_z=float(significance_z),
    )
    frame = gateway.merge(acs, on="component_index", how="left")
    frame = frame.merge(
        mediator[["component_index", "hyper_amce_total"]],
        on="component_index",
        how="left",
    )

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
                "ace": float(row.hyper_ace_total),
                "acs": float(row.hyper_acs_total),
                "amce": float(row.hyper_amce_total),
            }
        )
    return pd.DataFrame(rows)


def draw_acs_ace(ax: plt.Axes, nodes: pd.DataFrame, norm: mpl.colors.Normalize) -> None:
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
    fraction = nodes["amce"].to_numpy() / max(float(nodes["amce"].max()), 1.0e-12)
    sizes = 185.0 + 360.0 * np.clip(fraction, 0.0, 1.0)
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


def add_colorbar(fig: plt.Figure, ax: plt.Axes, norm: mpl.colors.Normalize, cmap_name: str, label: str) -> None:
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colormaps[cmap_name])
    cbar = fig.colorbar(sm, ax=ax, location="bottom", shrink=0.68, pad=0.07, aspect=24)
    cbar.set_label(label)


def plot_combined_map(
    linear_nodes: pd.DataFrame,
    mlp_peid_nodes: pd.DataFrame,
    output: Path,
    *,
    save_svg: bool,
) -> Path:
    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    fig = plt.figure(figsize=(10.6, 8.15), constrained_layout=True)
    axes = [
        fig.add_subplot(2, 2, 1, projection="mollweide"),
        fig.add_subplot(2, 2, 2, projection="mollweide"),
        fig.add_subplot(2, 2, 3, projection="mollweide"),
        fig.add_subplot(2, 2, 4, projection="mollweide"),
    ]
    for ax in axes:
        draw_world(ax, land, coastlines)
        add_geographic_ticks(ax)

    linear_ace_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(0.06, float(linear_nodes[["ace", "acs"]].to_numpy().max())))
    linear_amce_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(0.0015, float(linear_nodes["amce"].max())))
    mlp_ace_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(0.010, float(mlp_peid_nodes[["ace", "acs"]].to_numpy().max())))
    mlp_amce_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(0.0005, float(mlp_peid_nodes["amce"].max())))

    draw_acs_ace(axes[0], linear_nodes, linear_ace_norm)
    draw_amce(axes[1], linear_nodes, linear_amce_norm)
    draw_acs_ace(axes[2], mlp_peid_nodes, mlp_ace_norm)
    draw_amce(axes[3], mlp_peid_nodes, mlp_amce_norm)

    panel_labels = ["a", "b", "c", "d"]
    for label, ax in zip(panel_labels, axes, strict=True):
        ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=16, fontweight="bold")
    axes[0].text(0.5, 1.06, "Linear reproduction", transform=axes[0].transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")
    axes[1].text(0.5, 1.06, "Linear reproduction", transform=axes[1].transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")
    axes[2].text(0.5, 1.06, "MLP-TM-EI + PEID", transform=axes[2].transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")
    axes[3].text(0.5, 1.06, "MLP-TM-EI + PEID", transform=axes[3].transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")

    add_colorbar(fig, axes[0], linear_ace_norm, "OrRd", "ACS (inner node) and ACE (outer ring)")
    add_colorbar(fig, axes[1], linear_amce_norm, "Greens", "AMCE")
    add_colorbar(fig, axes[2], mlp_ace_norm, "OrRd", "Hyper-ACS (inner node) and hyper-ACE (outer ring)")
    add_colorbar(fig, axes[3], mlp_amce_norm, "Greens", "Hyper-AMCE")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if save_svg:
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-maps", default=str(DEFAULT_COMPONENT_MAPS))
    parser.add_argument("--linear-gateway-scores", default=str(DEFAULT_GATEWAY))
    parser.add_argument("--linear-mediator-scores", default=str(DEFAULT_MEDIATOR))
    parser.add_argument("--mlp-peid-gateway-scores", default=str(DEFAULT_MLP_GATEWAY))
    parser.add_argument("--mlp-peid-mediator-scores", default=str(DEFAULT_MLP_MEDIATOR))
    parser.add_argument("--hyperedges", default=str(DEFAULT_HYPEREDGES))
    parser.add_argument("--significance-z", type=float, default=2.0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--save-svg", action="store_true")
    args = parser.parse_args()

    maps = np.load(Path(args.component_maps).expanduser())["component_maps"]
    linear_nodes = build_linear_node_frame(
        maps,
        Path(args.linear_gateway_scores).expanduser(),
        Path(args.linear_mediator_scores).expanduser(),
    )
    mlp_peid_nodes = build_mlp_peid_node_frame(
        maps,
        Path(args.mlp_peid_gateway_scores).expanduser(),
        Path(args.mlp_peid_mediator_scores).expanduser(),
        Path(args.hyperedges).expanduser(),
        significance_z=float(args.significance_z),
    )
    output = plot_combined_map(
        linear_nodes,
        mlp_peid_nodes,
        Path(args.output).expanduser(),
        save_svg=bool(args.save_svg),
    )
    print(output)


if __name__ == "__main__":
    main()
