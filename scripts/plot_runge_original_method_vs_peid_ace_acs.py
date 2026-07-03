#!/usr/bin/env python3
"""Compare Runge 2015 ACE/ACS with MLP+PEID Hyper-ACE/Hyper-ACS on 1948-2026 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_runge_ace_acs_oldstyle_alignment import add_panel_colorbar, draw_ace_acs_panel, robust_vmax_excluding_largest_ace
from plot_runge_gateway_mediator_map import (
    COASTLINE_URL,
    LAND_URL,
    build_node_frame as build_original_node_frame,
    extract_lines,
    extract_polygons,
    load_geojson,
)
from plot_runge_order1_vs_order2_ace_acs import (
    NEW_BASE,
    NEW_COMPONENT_MAPS,
    NEW_HYPEREDGES,
    build_nodes as build_peid_nodes,
)


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_BASE = ROOT / "results" / "runge_slp_daily_1948_2026_20260628" / "results" / "runge" / "2015_gateways"
ORIGINAL_CORRECTED_BASE = (
    ROOT / "results" / "runge_slp_daily_1948_2026_20260628" / "results" / "runge" / "2015_gateways_pcstable_corrected"
)
ORIGINAL_COMPONENT_MAPS = ORIGINAL_BASE / "component_maps.npz"
ORIGINAL_GATEWAY = ORIGINAL_CORRECTED_BASE / "gateway_scores.csv"
ORIGINAL_MEDIATOR = ORIGINAL_CORRECTED_BASE / "mediator_scores.csv"
PEID_GATEWAY = NEW_BASE / "results" / "runge" / "peid_hypergraph" / "hyper_gateway_scores.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "reports" / "assets" / "runge_original_method_vs_mlp_peid_ace_acs_1948_2026.png"
DEFAULT_SUMMARY = NEW_BASE / "results" / "runge" / "ace_acs_alignment" / "original_method_vs_peid_summary.csv"
DEFAULT_MANIFEST = NEW_BASE / "results" / "runge" / "ace_acs_alignment" / "original_method_vs_peid_manifest.json"


def select_label_nodes(*frames: pd.DataFrame) -> pd.DataFrame:
    return frames[0].copy()


def summarize(panel: str, nodes: pd.DataFrame) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for metric in ["ace", "acs"]:
        for rank, row in enumerate(nodes.nlargest(10, metric).itertuples(index=False), start=1):
            rows.append(
                {
                    "panel": panel,
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
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
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

    maps = np.load(ORIGINAL_COMPONENT_MAPS)["component_maps"]
    original_nodes = build_original_node_frame(maps, ORIGINAL_GATEWAY, ORIGINAL_MEDIATOR)
    peid_nodes = build_peid_nodes(
        NEW_COMPONENT_MAPS,
        PEID_GATEWAY,
        NEW_HYPEREDGES,
        significance_z=float(args.significance_z),
        include_order2=True,
    )
    label_nodes = select_label_nodes(original_nodes, peid_nodes)

    original_vmax = float(original_nodes[["ace", "acs"]].to_numpy().max())
    peid_raw_vmax = float(peid_nodes[["ace", "acs"]].to_numpy().max())
    peid_cap = robust_vmax_excluding_largest_ace(peid_nodes)
    original_norm = mpl.colors.Normalize(vmin=0.0, vmax=original_vmax)
    peid_norm = mpl.colors.Normalize(vmin=0.0, vmax=peid_cap, clip=False)
    cmap = mpl.colormaps["OrRd"]

    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    fig = plt.figure(figsize=(13.2, 5.1), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.06)
    axes = [fig.add_subplot(grid[0, 0], projection="mollweide"), fig.add_subplot(grid[0, 1], projection="mollweide")]
    draw_ace_acs_panel(axes[0], original_nodes, label_nodes, original_norm, cmap, land, coastlines)
    draw_ace_acs_panel(axes[1], peid_nodes, label_nodes, peid_norm, cmap, land, coastlines)
    axes[0].set_title("Runge 2015 method (PC-stable), new 1948-2026 data", fontsize=9, fontweight="bold", pad=8)
    axes[1].set_title("MLP+PEID, new 1948-2026 data", fontsize=9, fontweight="bold", pad=8)
    axes[0].text(-0.06, 1.03, "a", transform=axes[0].transAxes, fontsize=16, fontweight="bold")
    axes[1].text(-0.06, 1.03, "b", transform=axes[1].transAxes, fontsize=16, fontweight="bold")
    add_panel_colorbar(fig, axes[0], original_norm, cmap, label="ACS (inner node) and ACE (outer ring)")
    add_panel_colorbar(
        fig,
        axes[1],
        peid_norm,
        cmap,
        label="Hyper-ACS (inner node) and Hyper-ACE (outer ring), clipped at non-outlier max",
        extend="max",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=500, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(summarize("runge_2015_method", original_nodes) + summarize("mlp_peid", peid_nodes))
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
                "original_method_inputs": {
                    "component_maps": str(ORIGINAL_COMPONENT_MAPS.relative_to(ROOT)),
                    "gateway_scores": str(ORIGINAL_GATEWAY.relative_to(ROOT)),
                    "mediator_scores": str(ORIGINAL_MEDIATOR.relative_to(ROOT)),
                },
                "peid_inputs": {
                    "component_maps": str(NEW_COMPONENT_MAPS.relative_to(ROOT)),
                    "gateway_scores": str(PEID_GATEWAY.relative_to(ROOT)),
                    "hyperedges": str(NEW_HYPEREDGES.relative_to(ROOT)),
                },
                "original_method": (
                    "Runge et al. 2015 ACE/ACS from max-absolute lagged causal effects in the linear SEM causal "
                    "network, using PC-stable parents for sparse regression"
                ),
                "peid_method": "MLP+PEID Hyper-ACE/Hyper-ACS with significant order-2 terms",
                "significance_z": float(args.significance_z),
                "panel_vmax": {
                    "runge_2015_method": original_vmax,
                    "mlp_peid_raw": peid_raw_vmax,
                    "mlp_peid_color_cap": peid_cap,
                },
                "label_rule": "all component nodes labeled",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
