#!/usr/bin/env python3
"""Plot normalized first-order and first/second-order composite ACE/ACS maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_runge_ace_acs_oldstyle_alignment import add_panel_colorbar, draw_ace_acs_panel, select_label_nodes
from plot_runge_gateway_mediator_map import COASTLINE_URL, LAND_URL, extract_lines, extract_polygons, load_geojson
from plot_runge_order1_vs_order2_ace_acs import (
    NEW_BASE,
    NEW_COMPONENT_MAPS,
    NEW_GATEWAY,
    NEW_HYPEREDGES,
    build_nodes,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "reports" / "assets" / "runge_mlp_peid_normalized_order1_order2_ace_acs_1948_2026.png"
DEFAULT_SUMMARY = NEW_BASE / "results" / "runge" / "ace_acs_alignment" / "normalized_order1_order2_summary.csv"
DEFAULT_MANIFEST = NEW_BASE / "results" / "runge" / "ace_acs_alignment" / "normalized_order1_order2_manifest.json"


def safe_max_normalize(values: pd.Series) -> pd.Series:
    vmax = float(values.max())
    if vmax <= 0.0:
        return values * 0.0
    return values.astype(float) / vmax


def build_normalized_nodes(order1_nodes: pd.DataFrame, total_nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    order2_nodes = total_nodes[["local", "paper", "lon", "lat"]].copy()
    order2_nodes["ace"] = total_nodes["ace"].astype(float) - order1_nodes["ace"].astype(float)
    order2_nodes["acs"] = total_nodes["acs"].astype(float) - order1_nodes["acs"].astype(float)

    order1_norm = order1_nodes.copy()
    composite = order1_nodes.copy()
    for metric in ["ace", "acs"]:
        o1 = safe_max_normalize(order1_nodes[metric])
        o2 = safe_max_normalize(order2_nodes[metric])
        order1_norm[metric] = o1
        composite[metric] = 0.5 * (o1 + o2)
    return order1_norm, order2_nodes, composite


def summarize(order1_norm: pd.DataFrame, order2_nodes: pd.DataFrame, composite: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    frames = {
        "order1_normalized": order1_norm,
        "order2_raw": order2_nodes,
        "normalized_average": composite,
    }
    for panel, nodes in frames.items():
        for metric in ["ace", "acs"]:
            for rank, row in enumerate(nodes.nlargest(8, metric).itertuples(index=False), start=1):
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
    return pd.DataFrame(rows)


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

    order1_nodes = build_nodes(NEW_COMPONENT_MAPS, NEW_GATEWAY, NEW_HYPEREDGES, significance_z=args.significance_z, include_order2=False)
    total_nodes = build_nodes(NEW_COMPONENT_MAPS, NEW_GATEWAY, NEW_HYPEREDGES, significance_z=args.significance_z, include_order2=True)
    order1_norm, order2_nodes, composite = build_normalized_nodes(order1_nodes, total_nodes)
    labels = select_label_nodes(composite)

    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = mpl.colormaps["OrRd"]

    fig = plt.figure(figsize=(13.2, 5.1), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.06)
    axes = [fig.add_subplot(grid[0, 0], projection="mollweide"), fig.add_subplot(grid[0, 1], projection="mollweide")]
    draw_ace_acs_panel(axes[0], order1_norm, labels, norm, cmap, land, coastlines)
    draw_ace_acs_panel(axes[1], composite, labels, norm, cmap, land, coastlines)
    axes[0].set_title("Normalized first-order only", fontsize=9, fontweight="bold", pad=8)
    axes[1].set_title("Mean of normalized first- and second-order terms", fontsize=9, fontweight="bold", pad=8)
    axes[0].text(-0.06, 1.03, "a", transform=axes[0].transAxes, fontsize=16, fontweight="bold")
    axes[1].text(-0.06, 1.03, "b", transform=axes[1].transAxes, fontsize=16, fontweight="bold")
    add_panel_colorbar(fig, axes[0], norm, cmap, label="Normalized score")
    add_panel_colorbar(fig, axes[1], norm, cmap, label="0.5 x normalized first-order + 0.5 x normalized second-order")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=500, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    summary = summarize(order1_norm, order2_nodes, composite)
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
                "definition": "normalized_average = 0.5 * order1 / max(order1) + 0.5 * order2 / max(order2), applied separately to ACE and ACS",
                "panel_a": "order1 / max(order1), separately for ACE and ACS",
                "panel_b": "mean of separately max-normalized order1 and order2 terms",
                "significance_z": float(args.significance_z),
                "order2_ace_max": float(order2_nodes["ace"].max()),
                "order2_acs_max": float(order2_nodes["acs"].max()),
                "order1_ace_max": float(order1_nodes["ace"].max()),
                "order1_acs_max": float(order1_nodes["acs"].max()),
                "label_rule": "all component nodes labeled",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
