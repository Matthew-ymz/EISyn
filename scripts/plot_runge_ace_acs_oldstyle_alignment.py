#!/usr/bin/env python3
"""Plot old-vs-new Runge Hyper-ACE/Hyper-ACS alignment maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_runge_gateway_mediator_map import (
    COASTLINE_URL,
    LAND_URL,
    add_geographic_ticks,
    add_labels,
    draw_world,
    extract_lines,
    extract_polygons,
    load_geojson,
)
from plot_runge_peid_synergy_map import build_node_frame


ROOT = Path(__file__).resolve().parents[1]
OLD_COMPONENT_MAPS = ROOT / "results" / "runge" / "2015_gateways" / "component_maps.npz"
OLD_GATEWAY = ROOT / "results" / "runge" / "peid_hypergraph" / "hyper_gateway_scores.csv"
OLD_HYPEREDGES = ROOT / "results" / "runge" / "peid_hypergraph" / "peid_hyperedges.csv"
NEW_COMPONENT_MAPS = (
    ROOT / "results" / "runge_slp_daily_1948_2026_20260628" / "results" / "runge" / "2015_gateways" / "component_maps.npz"
)
NEW_BASE = ROOT / "results" / "runge_slp_daily_1948_2026_oldstyle_ace_acs" / "mlp_tm_ei_lag04"
NEW_GATEWAY = NEW_BASE / "results" / "runge" / "peid_hypergraph" / "hyper_gateway_scores.csv"
NEW_HYPEREDGES = NEW_BASE / "results" / "runge" / "peid_hypergraph" / "peid_hyperedges.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "reports" / "assets" / "runge_mlp_peid_ace_acs_old_vs_2026_oldstyle.png"
DEFAULT_MANIFEST = NEW_BASE / "results" / "runge" / "ace_acs_alignment" / "manifest.json"


def load_nodes(component_maps_path: Path, gateway_path: Path, hyperedges_path: Path, significance_z: float) -> pd.DataFrame:
    maps = np.load(component_maps_path)["component_maps"]
    hyperedges = pd.read_csv(hyperedges_path)
    nodes = build_node_frame(maps, gateway_path, hyperedges, significance_z=significance_z)
    return nodes.rename(columns={"hyper_ace_total": "ace", "hyper_acs_total": "acs"})


def robust_vmax_excluding_largest_ace(nodes: pd.DataFrame) -> float:
    """Use the largest non-outlier ACE/ACS value, clipping only the ACE maximum."""
    max_ace_index = nodes["ace"].astype(float).idxmax()
    values = pd.concat(
        [
            nodes.loc[nodes.index != max_ace_index, "ace"].astype(float),
            nodes["acs"].astype(float),
        ],
        ignore_index=True,
    )
    return float(values.max())


def draw_ace_acs_panel(
    ax: plt.Axes,
    nodes: pd.DataFrame,
    label_nodes: pd.DataFrame,
    norm: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap,
    land: list,
    coastlines: list,
) -> None:
    draw_world(ax, land, coastlines)
    add_geographic_ticks(ax)
    lon = np.radians(nodes["lon"].astype(float).to_numpy())
    lat = np.radians(nodes["lat"].astype(float).to_numpy())
    ax.scatter(
        lon,
        lat,
        s=368,
        c=nodes["ace"].astype(float).to_numpy(),
        cmap=cmap,
        norm=norm,
        edgecolors="#8a3f22",
        linewidths=0.55,
        alpha=0.96,
        zorder=4,
    )
    ax.scatter(
        lon,
        lat,
        s=188,
        c=nodes["acs"].astype(float).to_numpy(),
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        alpha=0.98,
        zorder=5,
    )
    add_labels(ax, label_nodes)


def select_label_nodes(nodes: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    if top_n is None:
        return nodes.copy()
    label_ids = set(nodes.nlargest(top_n, "ace")["local"].astype(int))
    label_ids |= set(nodes.nlargest(top_n, "acs")["local"].astype(int))
    return nodes[nodes["local"].astype(int).isin(label_ids)].copy()


def add_panel_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    norm: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap,
    *,
    label: str,
    extend: str = "neither",
) -> None:
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, location="bottom", shrink=0.66, pad=0.09, aspect=24, extend=extend)
    cbar.set_label(label)
    cbar.ax.tick_params(labelsize=7)


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, object] = {}
    if path.exists():
        existing = json.loads(path.read_text())
    existing.update(payload)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--significance-z", type=float, default=2.0)
    args = parser.parse_args()

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
        }
    )

    old_nodes = load_nodes(OLD_COMPONENT_MAPS, OLD_GATEWAY, OLD_HYPEREDGES, float(args.significance_z))
    new_nodes = load_nodes(NEW_COMPONENT_MAPS, NEW_GATEWAY, NEW_HYPEREDGES, float(args.significance_z))

    old_vmax = float(old_nodes[["ace", "acs"]].to_numpy().max())
    new_raw_vmax = float(new_nodes[["ace", "acs"]].to_numpy().max())
    new_robust_vmax = robust_vmax_excluding_largest_ace(new_nodes)
    old_norm = mpl.colors.Normalize(vmin=0.0, vmax=old_vmax)
    new_norm = mpl.colors.Normalize(vmin=0.0, vmax=new_robust_vmax, clip=False)
    old_labels = select_label_nodes(old_nodes)
    new_labels = select_label_nodes(new_nodes)
    cmap = mpl.colormaps["OrRd"]

    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))

    fig = plt.figure(figsize=(13.2, 5.1), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.06)
    axes = [fig.add_subplot(grid[0, 0], projection="mollweide"), fig.add_subplot(grid[0, 1], projection="mollweide")]

    draw_ace_acs_panel(axes[0], old_nodes, old_labels, old_norm, cmap, land, coastlines)
    draw_ace_acs_panel(axes[1], new_nodes, new_labels, new_norm, cmap, land, coastlines)
    axes[0].set_title("Old 1948-2011, old PEID settings", fontsize=9, fontweight="bold", pad=8)
    axes[1].set_title("New 1948-2026, old PEID settings", fontsize=9, fontweight="bold", pad=8)
    axes[0].text(-0.06, 1.03, "a", transform=axes[0].transAxes, fontsize=16, fontweight="bold")
    axes[1].text(-0.06, 1.03, "b", transform=axes[1].transAxes, fontsize=16, fontweight="bold")

    add_panel_colorbar(fig, axes[0], old_norm, cmap, label="Hyper-ACS (inner node) and Hyper-ACE (outer ring)")
    add_panel_colorbar(
        fig,
        axes[1],
        new_norm,
        cmap,
        label="Hyper-ACS (inner node) and Hyper-ACE (outer ring), clipped at non-outlier max",
        extend="max",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=500, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    write_manifest(
        args.manifest,
        {
            "figure_png": str(args.output.relative_to(ROOT)),
            "figure_svg": str(args.output.with_suffix(".svg").relative_to(ROOT)),
            "figure_pdf": str(args.output.with_suffix(".pdf").relative_to(ROOT)),
            "colorbar_mode": "separate_per_panel",
            "panel_vmax": {"a": old_vmax, "b_raw": new_raw_vmax, "b_color_cap": new_robust_vmax},
            "b_color_cap_mode": "largest ACE value clipped; vmax is max of all remaining ACE values and all ACS values",
            "label_rule": "all component nodes labeled",
        },
    )


if __name__ == "__main__":
    main()
