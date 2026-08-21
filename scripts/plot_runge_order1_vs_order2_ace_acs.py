#!/usr/bin/env python3
"""Plot Runge ACE/ACS against the first- and second-order Ridge+PEID composite."""

from __future__ import annotations

import argparse
import ast
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
    build_node_frame as build_original_node_frame,
    component_center,
    extract_lines,
    extract_polygons,
    load_geojson,
    local_to_paper,
)

ROOT = Path(__file__).resolve().parents[1]
NEW_COMPONENT_MAPS = (
    ROOT / "results" / "runge_slp_daily_1948_2026_20260628" / "results" / "runge" / "2015_gateways" / "component_maps.npz"
)
ORIGINAL_BASE = ROOT / "results" / "runge_slp_daily_1948_2026_20260628" / "results" / "runge" / "2015_gateways"
ORIGINAL_CORRECTED_BASE = (
    ROOT / "results" / "runge_slp_daily_1948_2026_20260628" / "results" / "runge" / "2015_gateways_pcstable_corrected"
)
ORIGINAL_GATEWAY = ORIGINAL_CORRECTED_BASE / "gateway_scores.csv"
ORIGINAL_MEDIATOR = ORIGINAL_CORRECTED_BASE / "mediator_scores.csv"
NEW_BASE = ROOT / "results" / "runge_slp_daily_1948_2026_oldstyle_ace_acs" / "mlp_tm_ei_lag04"
NEW_GATEWAY = NEW_BASE / "results" / "runge" / "peid_hypergraph" / "hyper_gateway_scores.csv"
NEW_HYPEREDGES = NEW_BASE / "results" / "runge" / "peid_hypergraph" / "peid_hyperedges.csv"
DEFAULT_OUTPUT = ROOT / "fig" / "runge_node_ace_acs_comparison_1948_2026.png"
DEFAULT_MANIFEST = NEW_BASE / "results" / "runge" / "ace_acs_alignment" / "node_comparison_manifest.json"
DEFAULT_SUMMARY = NEW_BASE / "results" / "runge" / "ace_acs_alignment" / "node_comparison_summary.csv"


def parse_subset(value: object) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(v) for v in ast.literal_eval(str(value)))


def aggregate_hyper_ace_acs(
    hyperedges: pd.DataFrame,
    *,
    n_components: int,
    significance_z: float,
) -> pd.DataFrame:
    n = int(n_components)
    ace_order1 = np.zeros(n, dtype=float)
    acs_order1 = np.zeros(n, dtype=float)
    ace_order2_sum = np.zeros(n, dtype=float)
    acs_order2_sum = np.zeros(n, dtype=float)
    ace_order2_count = np.zeros(n, dtype=int)
    acs_order2_count = np.zeros(n, dtype=int)

    for row in hyperedges.itertuples(index=False):
        order = int(row.order)
        target = int(row.target_index)
        if target < 0 or target >= n:
            continue
        subset = parse_subset(row.subset)
        if order == 1:
            for src in subset:
                if 0 <= int(src) < n:
                    value = abs(float(row.delta_K))
                    ace_order1[int(src)] += value / float(order)
                    acs_order1[target] += value
        elif order == 2:
            z_value = getattr(row, "z", np.nan)
            if np.isnan(z_value) or abs(float(z_value)) < float(significance_z):
                continue
            value = abs(float(row.delta_K))
            for src in subset:
                if 0 <= int(src) < n:
                    ace_order2_sum[int(src)] += value / float(order)
                    ace_order2_count[int(src)] += 1
            acs_order2_sum[target] += value
            acs_order2_count[target] += 1

    order1_denom = max(1, n - 1)
    ace_order2 = np.divide(
        ace_order2_sum,
        ace_order2_count,
        out=np.zeros_like(ace_order2_sum),
        where=ace_order2_count > 0,
    )
    acs_order2 = np.divide(
        acs_order2_sum,
        acs_order2_count,
        out=np.zeros_like(acs_order2_sum),
        where=acs_order2_count > 0,
    )
    return pd.DataFrame(
        {
            "component_index": np.arange(n),
            "hyper_ace_order1": ace_order1 / order1_denom,
            "hyper_acs_order1": acs_order1 / order1_denom,
            "hyper_ace_order2": ace_order2,
            "hyper_acs_order2": acs_order2,
            "hyper_ace_order2_count": ace_order2_count,
            "hyper_acs_order2_count": acs_order2_count,
            "hyper_ace_total": ace_order1 / order1_denom + ace_order2,
            "hyper_acs_total": acs_order1 / order1_denom + acs_order2,
        }
    )


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
    hyperedges = pd.read_csv(hyperedges_path)
    scores = aggregate_hyper_ace_acs(hyperedges, n_components=maps.shape[2], significance_z=significance_z)
    rows: list[dict[str, float | int]] = []
    ace_col = "hyper_ace_total" if include_order2 else "hyper_ace_order1"
    acs_col = "hyper_acs_total" if include_order2 else "hyper_acs_order1"
    for row in scores.itertuples(index=False):
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


def build_runge_nodes(component_maps_path: Path, gateway_path: Path, mediator_path: Path) -> pd.DataFrame:
    maps = np.load(component_maps_path)["component_maps"]
    nodes = build_original_node_frame(maps, gateway_path, mediator_path)
    return nodes[["local", "paper", "lon", "lat", "ace", "acs"]].copy()


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


def relpath_for_manifest(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-maps", type=Path, default=NEW_COMPONENT_MAPS)
    parser.add_argument("--original-gateway", type=Path, default=ORIGINAL_GATEWAY)
    parser.add_argument("--original-mediator", type=Path, default=ORIGINAL_MEDIATOR)
    parser.add_argument("--gateway", type=Path, default=NEW_GATEWAY)
    parser.add_argument("--hyperedges", type=Path, default=NEW_HYPEREDGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--significance-z", type=float, default=2.0)
    parser.add_argument("--suptitle", default="")
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

    original_nodes = build_runge_nodes(args.component_maps, args.original_gateway, args.original_mediator)
    total_nodes = build_nodes(args.component_maps, args.gateway, args.hyperedges, significance_z=args.significance_z, include_order2=True)
    original_vmax = float(original_nodes[["ace", "acs"]].to_numpy().max())
    ridge_cap = robust_vmax_excluding_largest_ace(total_nodes)
    raw_vmax = float(total_nodes[["ace", "acs"]].to_numpy().max())
    original_norm = mpl.colors.Normalize(vmin=0.0, vmax=original_vmax)
    ridge_norm = mpl.colors.Normalize(vmin=0.0, vmax=ridge_cap, clip=False)
    cmap = mpl.colormaps["OrRd"]

    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    labels = select_label_nodes(original_nodes)

    fig = plt.figure(figsize=(11.8, 4.95), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.06)
    axes = [fig.add_subplot(grid[0, idx], projection="mollweide") for idx in range(2)]
    draw_ace_acs_panel(axes[0], original_nodes, labels, original_norm, cmap, land, coastlines)
    draw_ace_acs_panel(axes[1], total_nodes, labels, ridge_norm, cmap, land, coastlines)
    axes[0].set_title("Runge 2015 method (PC-stable)", fontsize=9, fontweight="bold", pad=8)
    axes[1].set_title("Ridge+PEID: first- and second-order composite", fontsize=9, fontweight="bold", pad=8)
    if args.suptitle:
        fig.suptitle(args.suptitle, fontsize=10, fontweight="bold")
    for letter, ax in zip(["a", "b"], axes):
        ax.text(-0.06, 1.03, letter, transform=ax.transAxes, fontsize=16, fontweight="bold")
    add_panel_colorbar(fig, axes[0], original_norm, cmap, label="ACS (inner node) and ACE (outer ring)")
    add_panel_colorbar(
        fig,
        axes[1],
        ridge_norm,
        cmap,
        label="Hyper-ACS (inner node) and Hyper-ACE (outer ring), clipped scale",
        extend="max",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=500, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(
        summarize_nodes("runge_2015_pcstable", original_nodes)
        + summarize_nodes("ridge_order1_plus_order2", total_nodes)
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "figure_png": relpath_for_manifest(args.output),
                "figure_svg": relpath_for_manifest(args.output.with_suffix(".svg")),
                "figure_pdf": relpath_for_manifest(args.output.with_suffix(".pdf")),
                "summary_csv": relpath_for_manifest(args.summary),
                "component_maps": relpath_for_manifest(args.component_maps),
                "original_gateway_scores": relpath_for_manifest(args.original_gateway),
                "original_mediator_scores": relpath_for_manifest(args.original_mediator),
                "hyperedges": relpath_for_manifest(args.hyperedges),
                "panel_a": "Runge 2015 PC-stable reproduction ACE/ACS on 1948-2026 data",
                "panel_b": "Ridge+PEID order1 averaged by n-1 plus significant order2 averaged by per-node significant hyperedge count",
                "definition": "order1 terms are divided by n-1; order2 ACE is mean |Syn|/|K| over significant outgoing hyperedges involving the source; order2 ACS is mean |Syn| over significant incoming hyperedges for the target; empty order2 sets contribute 0",
                "significance_z": float(args.significance_z),
                "colorbar_mode": "panel a uses the Runge ACE/ACS scale; panel b uses the Ridge+PEID robust clipped scale",
                "runge_vmax": original_vmax,
                "ridge_raw_vmax": raw_vmax,
                "ridge_color_cap": ridge_cap,
                "ridge_color_cap_mode": "largest total ACE value clipped; vmax is max of all remaining total ACE values and all total ACS values",
                "label_rule": "all component nodes labeled",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
