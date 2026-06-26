#!/usr/bin/env python3
"""Plot top integrated MLP-TM-EI/PEID source-pair synergies on world maps."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

from plot_runge_gateway_mediator_map import (
    COASTLINE_URL,
    DEFAULT_COMPONENT_MAPS,
    LAND_URL,
    add_geographic_ticks,
    component_center,
    draw_world,
    extract_lines,
    extract_polygons,
    load_geojson,
    local_to_paper,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HYPEREDGES = ROOT / "results" / "runge" / "peid_hypergraph" / "peid_hyperedges.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "reports" / "assets" / "part2_runge_mlp_top_pair_synergy_map.png"
DEFAULT_FIG_OUTPUT = ROOT / "fig" / "runge" / "peid_hypergraph" / "top_pair_spatial_synergy_map.png"


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


def parse_subset(value: object) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    parsed = ast.literal_eval(str(value))
    return tuple(int(v) for v in parsed)


def load_nodes(component_maps_path: Path) -> pd.DataFrame:
    component_maps = np.load(component_maps_path)["component_maps"]
    lat = np.linspace(-90.0, 90.0, component_maps.shape[0])
    lon = np.linspace(0.0, 360.0, component_maps.shape[1], endpoint=False)
    lon = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    maps = component_maps[:, order, :]
    lon = lon[order]
    rows: list[dict[str, float | int]] = []
    for local in range(component_maps.shape[2]):
        center_lon, center_lat = component_center(maps[..., local], lat, lon)
        rows.append({"local": local, "paper": local_to_paper(local), "lon": center_lon, "lat": center_lat})
    return pd.DataFrame(rows)


def load_significant_order2(hyperedges_path: Path, significance_z: float) -> pd.DataFrame:
    frame = pd.read_csv(hyperedges_path)
    frame = frame[(frame["order"].astype(int) == 2) & (frame["delta_K"].astype(float) > 0.0)].copy()
    frame["subset_tuple"] = frame["subset"].apply(parse_subset)
    frame = frame[np.abs(frame["z"].astype(float)) >= float(significance_z)].copy()
    if frame.empty:
        raise ValueError("No significant positive order-2 PEID hyperedges found.")
    return frame


def summarize_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby("subset_tuple", as_index=False)
        .agg(
            total_positive_delta2=("delta_K", "sum"),
            max_delta2=("delta_K", "max"),
            positive_target_count=("target_index", "nunique"),
        )
        .sort_values("total_positive_delta2", ascending=False)
        .reset_index(drop=True)
    )
    summary["source_a"] = summary["subset_tuple"].apply(lambda subset: int(subset[0]))
    summary["source_b"] = summary["subset_tuple"].apply(lambda subset: int(subset[1]))
    summary["source_a_paper"] = summary["source_a"].map(local_to_paper)
    summary["source_b_paper"] = summary["source_b"].map(local_to_paper)
    return summary


def select_top_pair_panels(frame: pd.DataFrame, *, top_pairs: int, top_targets: int) -> list[tuple[pd.Series, pd.DataFrame]]:
    summary = summarize_pairs(frame).head(int(top_pairs))
    panels: list[tuple[pd.Series, pd.DataFrame]] = []
    for _, pair in summary.iterrows():
        subset = tuple(pair["subset_tuple"])
        edges = (
            frame[frame["subset_tuple"] == subset]
            .sort_values("delta_K", ascending=False)
            .head(int(top_targets))
            .reset_index(drop=True)
        )
        panels.append((pair, edges))
    return panels


def to_axes_xy(ax: plt.Axes, lon: float, lat: float) -> np.ndarray:
    display = ax.transData.transform((np.radians(float(lon)), np.radians(float(lat))))
    return ax.transAxes.inverted().transform(display)


def draw_labels(ax: plt.Axes, nodes: pd.DataFrame, active: set[int]) -> None:
    active_nodes = nodes[nodes["local"].astype(int).isin(active)].copy()
    for row in active_nodes.itertuples(index=False):
        ax.text(
            np.radians(float(row.lon)),
            np.radians(float(row.lat)),
            str(int(row.paper)),
            ha="center",
            va="center",
            fontsize=7.2,
            weight="bold",
            color="white",
            path_effects=[pe.withStroke(linewidth=1.45, foreground="#1b1b1b")],
            zorder=7,
        )


def offset_hub(source_mid: np.ndarray, target_xy: np.ndarray, idx: int) -> np.ndarray:
    """Disperse hyperedge hub nodes near the source-pair midpoint."""
    direction = target_xy - source_mid
    length = float(np.linalg.norm(direction))
    if length < 1.0e-9:
        direction = np.array([1.0, 0.0])
        length = 1.0
    direction = direction / length
    perpendicular = np.array([-direction[1], direction[0]])
    side = -1.0 if idx % 2 else 1.0
    shell = 1 + (idx // 2) % 5
    radial = 0.011 * shell
    along = 0.010 * ((idx % 3) - 1)
    base = 0.58 * source_mid + 0.42 * target_xy
    hub = base + side * radial * perpendicular + along * direction
    return np.clip(hub, np.array([0.04, 0.08]), np.array([0.96, 0.92]))


def draw_panel(
    ax: plt.Axes,
    *,
    nodes: pd.DataFrame,
    pair: pd.Series,
    edges: pd.DataFrame,
    land: list[list[tuple[float, float]]],
    coastlines: list[list[tuple[float, float]]],
    panel_label: str,
) -> None:
    draw_world(ax, land, coastlines)
    add_geographic_ticks(ax)

    source_nodes = {int(pair["source_a"]), int(pair["source_b"])}
    target_nodes = {int(value) for value in edges["target_index"].astype(int).tolist()}
    active_nodes = source_nodes | target_nodes
    inactive = nodes[~nodes["local"].astype(int).isin(active_nodes)]
    active_targets = nodes[nodes["local"].astype(int).isin(target_nodes - source_nodes)]
    active_sources = nodes[nodes["local"].astype(int).isin(source_nodes)]

    ax.scatter(
        np.radians(inactive["lon"].to_numpy()),
        np.radians(inactive["lat"].to_numpy()),
        s=52,
        color="#9aa0a6",
        edgecolors="white",
        linewidths=0.22,
        alpha=0.30,
        zorder=3,
    )
    if not active_targets.empty:
        ax.scatter(
            np.radians(active_targets["lon"].to_numpy()),
            np.radians(active_targets["lat"].to_numpy()),
            s=210,
            color="#5bb8b1",
            edgecolors="#204b5a",
            linewidths=0.65,
            alpha=0.96,
            zorder=4,
        )
    ax.scatter(
        np.radians(active_sources["lon"].to_numpy()),
        np.radians(active_sources["lat"].to_numpy()),
        s=350,
        color="#1f78b4",
        edgecolors="#112f43",
        linewidths=0.85,
        alpha=0.98,
        zorder=5,
    )
    ax.scatter(
        np.radians(active_sources["lon"].to_numpy()),
        np.radians(active_sources["lat"].to_numpy()),
        s=186,
        color="#65c5c8",
        edgecolors="none",
        alpha=0.92,
        zorder=6,
    )

    lookup = nodes.set_index("local")
    source_xy = []
    for source in sorted(source_nodes):
        row = lookup.loc[source]
        source_xy.append(to_axes_xy(ax, float(row.lon), float(row.lat)))
    source_mid = 0.5 * (source_xy[0] + source_xy[1])
    edge_color = "#7c2d6c"

    values = edges["delta_K"].astype(float).to_numpy()
    vmin, vmax = float(values.min()), float(values.max())
    denom = max(vmax - vmin, 1.0e-12)
    for idx, row in enumerate(edges.itertuples(index=False)):
        target = int(row.target_index)
        target_node = lookup.loc[target]
        target_xy = to_axes_xy(ax, float(target_node.lon), float(target_node.lat))
        hub = offset_hub(source_mid, target_xy, idx)
        strength = (float(row.delta_K) - vmin) / denom
        linewidth = 0.55 + 2.35 * strength
        alpha = 0.20 + 0.48 * strength
        for start_xy in source_xy:
            ax.plot(
                [start_xy[0], hub[0]],
                [start_xy[1], hub[1]],
                transform=ax.transAxes,
                color=edge_color,
                linewidth=max(0.45, linewidth * 0.72),
                alpha=max(0.13, alpha * 0.48),
                solid_capstyle="round",
                zorder=3.35,
            )
        arrow = mpatches.FancyArrowPatch(
            posA=hub,
            posB=target_xy,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=5.1 + 3.0 * strength,
            linewidth=linewidth,
            color=edge_color,
            alpha=alpha,
            shrinkA=1.0,
            shrinkB=9.0 if target in source_nodes else 7.0,
            connectionstyle=f"arc3,rad={0.075 if idx % 2 == 0 else -0.075}",
            clip_on=True,
            zorder=3.8,
        )
        ax.add_patch(arrow)
        ax.scatter(
            [hub[0]],
            [hub[1]],
            transform=ax.transAxes,
            s=10 + 13 * strength,
            color=edge_color,
            edgecolors="white",
            linewidths=0.28,
            alpha=min(0.82, alpha + 0.22),
            zorder=6.3,
        )

    draw_labels(ax, nodes, active_nodes)
    note = (
        f"{panel_label}  No.{int(pair['source_a_paper'])} + No.{int(pair['source_b_paper'])}; "
        f"ΣΔ2+={float(pair['total_positive_delta2']):.4f}; $X_t\\to X_{{t+1}}$"
    )
    ax.text(
        0.02,
        0.035,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        zorder=9,
    )


def draw_figure(nodes: pd.DataFrame, panels: list[tuple[pd.Series, pd.DataFrame]], output: Path) -> Path:
    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    fig = plt.figure(figsize=(7.6, 3.85 * len(panels)), constrained_layout=True)
    for idx, (pair, edges) in enumerate(panels, start=1):
        ax = fig.add_subplot(len(panels), 1, idx, projection="mollweide")
        draw_panel(
            ax,
            nodes=nodes,
            pair=pair,
            edges=edges,
            land=land,
            coastlines=coastlines,
            panel_label=f"Rank {idx}",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-maps", type=Path, default=DEFAULT_COMPONENT_MAPS)
    parser.add_argument("--hyperedges", type=Path, default=DEFAULT_HYPEREDGES)
    parser.add_argument("--significance-z", type=float, default=2.0)
    parser.add_argument("--top-pairs", type=int, default=3)
    parser.add_argument("--top-targets", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fig-output", type=Path, default=DEFAULT_FIG_OUTPUT)
    args = parser.parse_args()

    nodes = load_nodes(args.component_maps)
    frame = load_significant_order2(args.hyperedges, args.significance_z)
    panels = select_top_pair_panels(frame, top_pairs=args.top_pairs, top_targets=args.top_targets)
    output = draw_figure(nodes, panels, args.output)
    if args.fig_output:
        args.fig_output.parent.mkdir(parents=True, exist_ok=True)
        args.fig_output.write_bytes(output.read_bytes())
    print(output)
    for rank, (pair, edges) in enumerate(panels, start=1):
        report = edges.assign(
            source_a_paper=int(pair["source_a_paper"]),
            source_b_paper=int(pair["source_b_paper"]),
            target_paper=edges["target_index"].astype(int).map(local_to_paper),
        )
        print(f"\nRank {rank}: No.{int(pair['source_a_paper'])} + No.{int(pair['source_b_paper'])}, "
              f"total={float(pair['total_positive_delta2']):.6f}")
        print(report[["target_paper", "delta_K", "z"]].to_string(index=False))


if __name__ == "__main__":
    main()
