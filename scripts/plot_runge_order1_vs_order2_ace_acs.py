#!/usr/bin/env python3
"""Plot first-order-only vs second-order-aware Runge ACE/ACS maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_runge_ace_acs_oldstyle_alignment import (
    add_panel_colorbar,
    draw_ace_acs_panel,
    robust_vmax_excluding_largest_ace,
    select_label_nodes,
)
from plot_runge_gateway_mediator_map import (
    COASTLINE_URL,
    LAND_URL,
    component_center,
    extract_lines,
    extract_polygons,
    load_geojson,
    local_to_paper,
)
from plot_runge_peid_synergy_map import aggregate_hyper_acs


ROOT = Path(__file__).resolve().parents[1]
NEW_COMPONENT_MAPS = (
    ROOT / "results" / "runge_slp_daily_1948_2026_20260628" / "results" / "runge" / "2015_gateways" / "component_maps.npz"
)
NEW_BASE = ROOT / "results" / "runge_slp_daily_1948_2026_oldstyle_ace_acs" / "mlp_tm_ei_lag04"
NEW_GATEWAY = NEW_BASE / "results" / "runge" / "peid_hypergraph" / "hyper_gateway_scores.csv"
NEW_HYPEREDGES = NEW_BASE / "results" / "runge" / "peid_hypergraph" / "peid_hyperedges.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "reports" / "assets" / "runge_mlp_peid_order1_vs_order2_ace_acs_1948_2026.png"
DEFAULT_MANIFEST = NEW_BASE / "results" / "runge" / "ace_acs_alignment" / "order1_vs_order2_manifest.json"
DEFAULT_SUMMARY = NEW_BASE / "results" / "runge" / "ace_acs_alignment" / "order1_vs_order2_summary.csv"


def load_component_maps(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maps = np.load(path)["component_maps"]
    lat = np.linspace(-90.0, 90.0, maps.shape[0])
    lon = np.linspace(0.0, 360.0, maps.shape[1], endpoint=False)
    lon = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    return maps[:, order, :], lat, lon[order]


def build_nodes(
    component_maps_path: Path,
    gateway_path: Path,
    hyperedges_path: Path,
    *,
    significance_z: float,
    include_order2: bool,
) -> pd.DataFrame:
    maps, lat, lon = load_component_maps(component_maps_path)
    gateway = pd.read_csv(gateway_path)
    hyperedges = pd.read_csv(hyperedges_path)
    acs = aggregate_hyper_acs(hyperedges, n_components=maps.shape[2], significance_z=significance_z)
    gateway = gateway.merge(acs, on="component_index", how="left")
    rows: list[dict[str, float | int]] = []
    ace_col = "hyper_ace_total" if include_order2 else "hyper_ace_order1"
    acs_col = "hyper_acs_total" if include_order2 else "hyper_acs_order1"
    for row in gateway.itertuples(index=False):
        local = int(row.component_index)
        center_lon, center_lat = component_center(maps[..., local], lat, lon)
        rows.append(
            {
                "local": local,
                "paper": local_to_paper(local),
                "lon": center_lon,
                "lat": center_lat,
                "ace": float(getattr(row, ace_col)),
                "acs": float(getattr(row, acs_col)),
            }
        )
    return pd.DataFrame(rows)


def summarize_nodes(label: str, nodes: pd.DataFrame) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for metric in ["ace", "acs"]:
        for rank, row in enumerate(nodes.nlargest(5, metric).itertuples(index=False), start=1):
            rows.append(
                {
                    "panel": label,
                    "metric": metric.upper(),
                    "rank": rank,
                    "paper": int(row.paper),
                    "local": int(row.local),
                    "value": float(getattr(row, metric)),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
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

    order1_nodes = build_nodes(NEW_COMPONENT_MAPS, NEW_GATEWAY, NEW_HYPEREDGES, significance_z=args.significance_z, include_order2=False)
    total_nodes = build_nodes(NEW_COMPONENT_MAPS, NEW_GATEWAY, NEW_HYPEREDGES, significance_z=args.significance_z, include_order2=True)
    cap = robust_vmax_excluding_largest_ace(total_nodes)
    raw_vmax = float(total_nodes[["ace", "acs"]].to_numpy().max())
    norm = mpl.colors.Normalize(vmin=0.0, vmax=cap, clip=False)
    cmap = mpl.colormaps["OrRd"]

    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    labels = select_label_nodes(total_nodes)

    fig = plt.figure(figsize=(13.2, 5.1), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.06)
    axes = [fig.add_subplot(grid[0, 0], projection="mollweide"), fig.add_subplot(grid[0, 1], projection="mollweide")]
    draw_ace_acs_panel(axes[0], order1_nodes, labels, norm, cmap, land, coastlines)
    draw_ace_acs_panel(axes[1], total_nodes, labels, norm, cmap, land, coastlines)
    axes[0].set_title("First-order only", fontsize=9, fontweight="bold", pad=8)
    axes[1].set_title("First + significant second-order synergy", fontsize=9, fontweight="bold", pad=8)
    axes[0].text(-0.06, 1.03, "a", transform=axes[0].transAxes, fontsize=16, fontweight="bold")
    axes[1].text(-0.06, 1.03, "b", transform=axes[1].transAxes, fontsize=16, fontweight="bold")
    label = "Hyper-ACS (inner node) and Hyper-ACE (outer ring), common clipped scale"
    add_panel_colorbar(fig, axes[0], norm, cmap, label=label, extend="max")
    add_panel_colorbar(fig, axes[1], norm, cmap, label=label, extend="max")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=500, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(summarize_nodes("order1_only", order1_nodes) + summarize_nodes("order1_plus_order2", total_nodes))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "figure_png": str(args.output.relative_to(ROOT)),
                "figure_svg": str(args.output.with_suffix(".svg").relative_to(ROOT)),
                "figure_pdf": str(args.output.with_suffix(".pdf").relative_to(ROOT)),
                "summary_csv": str(args.summary.relative_to(ROOT)),
                "dataset": "runge_slp_daily_1948_2026_oldstyle_ace_acs/mlp_tm_ei_lag04",
                "panel_a": "hyper_ace_order1 and hyper_acs_order1 only",
                "panel_b": "hyper_ace_total and hyper_acs_total, order2 gated by abs(z)>=significance_z",
                "significance_z": float(args.significance_z),
                "colorbar_mode": "common robust clipped scale",
                "raw_vmax": raw_vmax,
                "color_cap": cap,
                "color_cap_mode": "largest total ACE value clipped; vmax is max of all remaining total ACE values and all total ACS values",
                "label_rule": "all component nodes labeled",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
