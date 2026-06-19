#!/usr/bin/env python3
"""Plot PEID synergy-aware Runge component nodes and top hyperedges."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_runge_gateway_mediator_map import (
    DEFAULT_COMPONENT_MAPS,
    LAND_URL,
    COASTLINE_URL,
    add_geographic_ticks,
    add_labels,
    component_center,
    draw_world,
    extract_lines,
    extract_polygons,
    load_geojson,
    local_to_paper,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HYPER_GATEWAY = ROOT / "results" / "runge" / "peid_hypergraph" / "hyper_gateway_scores.csv"
DEFAULT_HYPEREDGES = ROOT / "results" / "runge" / "peid_hypergraph" / "peid_hyperedges.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "reports" / "assets" / "part2_runge_peid_synergy_map.png"
PAPER_TO_LOCAL = {18: 7, 26: 8, 48: 21, 7: 18, 8: 26, 21: 48}


def parse_node_list(value: str | None) -> list[int]:
    if not value:
        return []
    nodes: list[int] = []
    for part in value.split(","):
        cleaned = part.strip().replace("No.", "").replace("no.", "")
        if cleaned:
            nodes.append(int(cleaned))
    return nodes


def paper_to_local(paper_index: int) -> int:
    return int(PAPER_TO_LOCAL.get(int(paper_index), int(paper_index)))


def parse_subset(value: object) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    parsed = ast.literal_eval(str(value))
    return tuple(int(v) for v in parsed)


def aggregate_hyper_acs(hyperedges: pd.DataFrame, *, n_components: int, significance_z: float) -> pd.DataFrame:
    """Incoming PEID analogue of ACS, using the same order gating as hyper-ACE."""
    n = int(n_components)
    order1 = np.zeros(n, dtype=float)
    order2 = np.zeros(n, dtype=float)
    for row in hyperedges.itertuples(index=False):
        order = int(row.order)
        target = int(row.target_index)
        if target < 0 or target >= n:
            continue
        if order == 1:
            order1[target] += abs(float(row.delta_K))
        elif order == 2:
            z_value = getattr(row, "z", np.nan)
            if np.isnan(z_value) or abs(float(z_value)) < float(significance_z):
                continue
            order2[target] += abs(float(row.delta_K))
    denom = max(1, n - 1)
    return pd.DataFrame(
        {
            "component_index": np.arange(n),
            "hyper_acs_order1": order1 / denom,
            "hyper_acs_order2": order2 / denom,
            "hyper_acs_total": (order1 + order2) / denom,
        }
    )


def build_node_frame(
    component_maps: np.ndarray,
    hyper_gateway_path: Path,
    hyperedges: pd.DataFrame,
    *,
    significance_z: float,
) -> pd.DataFrame:
    lat = np.linspace(-90.0, 90.0, component_maps.shape[0])
    lon = np.linspace(0.0, 360.0, component_maps.shape[1], endpoint=False)
    lon = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    maps = component_maps[:, order, :]
    lon = lon[order]

    gateway = pd.read_csv(hyper_gateway_path)
    acs = aggregate_hyper_acs(hyperedges, n_components=component_maps.shape[2], significance_z=significance_z)
    gateway = gateway.merge(acs, on="component_index", how="left")
    rows: list[dict[str, float | int]] = []
    for row in gateway.itertuples(index=False):
        local = int(row.component_index)
        center_lon, center_lat = component_center(maps[..., local], lat, lon)
        rows.append(
            {
                "local": local,
                "paper": local_to_paper(local),
                "lon": center_lon,
                "lat": center_lat,
                "hyper_ace_total": float(row.hyper_ace_total),
                "hyper_acs_total": float(row.hyper_acs_total),
                "hyper_ace_order2": float(row.hyper_ace_order2),
                "hyper_acs_order2": float(row.hyper_acs_order2),
            }
        )
    return pd.DataFrame(rows)


def select_top_hyperedges(
    hyperedges: pd.DataFrame,
    *,
    top_n: int,
    significance_z: float,
    cross_target_only: bool,
    focal_paper_nodes: list[int] | None = None,
    focal_mode: str = "any",
) -> pd.DataFrame:
    frame = hyperedges[(hyperedges["order"].astype(int) == 2) & (hyperedges["delta_K"].astype(float) > 0.0)].copy()
    frame["subset_tuple"] = frame["subset"].apply(parse_subset)
    frame = frame[np.abs(frame["z"].astype(float)) >= float(significance_z)]
    if cross_target_only:
        frame = frame[~frame.apply(lambda row: int(row["target_index"]) in row["subset_tuple"], axis=1)]
    focal_local = {paper_to_local(node) for node in (focal_paper_nodes or [])}
    if focal_local:
        source_mask = frame["subset_tuple"].apply(lambda subset: bool(focal_local.intersection(subset)))
        target_mask = frame["target_index"].astype(int).isin(focal_local)
        if focal_mode == "source":
            frame = frame[source_mask]
        elif focal_mode == "target":
            frame = frame[target_mask]
        elif focal_mode == "any":
            frame = frame[source_mask | target_mask]
        else:
            raise ValueError(f"Unsupported focal_mode: {focal_mode}")
    return frame.sort_values("delta_K", ascending=False).head(int(top_n)).reset_index(drop=True)


def to_axes_xy(ax: plt.Axes, lon: float, lat: float) -> np.ndarray:
    display = ax.transData.transform((np.radians(float(lon)), np.radians(float(lat))))
    return ax.transAxes.inverted().transform(display)


def draw_top_hyperedges(ax: plt.Axes, nodes: pd.DataFrame, hyperedges: pd.DataFrame) -> None:
    if hyperedges.empty:
        return
    node_lookup = nodes.set_index("local")
    values = hyperedges["delta_K"].astype(float).to_numpy()
    vmin = float(values.min())
    vmax = float(values.max())
    denom = max(vmax - vmin, 1.0e-12)
    color = "#7c2d6c"
    for idx, row in enumerate(hyperedges.itertuples(index=False)):
        sources = tuple(int(v) for v in row.subset_tuple)
        target = int(row.target_index)
        if len(sources) != 2 or target not in node_lookup.index or any(src not in node_lookup.index for src in sources):
            continue
        src_xy = []
        for src in sources:
            node = node_lookup.loc[src]
            src_xy.append(to_axes_xy(ax, float(node.lon), float(node.lat)))
        target_node = node_lookup.loc[target]
        target_xy = to_axes_xy(ax, float(target_node.lon), float(target_node.lat))
        src_mid = 0.5 * (src_xy[0] + src_xy[1])
        hub_xy = 0.72 * src_mid + 0.28 * target_xy
        strength = (float(row.delta_K) - vmin) / denom
        linewidth = 0.8 + 1.5 * strength
        alpha = 0.28 + 0.28 * strength
        for start_xy in src_xy:
            ax.plot(
                [start_xy[0], hub_xy[0]],
                [start_xy[1], hub_xy[1]],
                transform=ax.transAxes,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                solid_capstyle="round",
                clip_on=True,
                zorder=3.1,
            )
        arrow = mpatches.FancyArrowPatch(
            posA=hub_xy,
            posB=target_xy,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=5.5 + 2.2 * strength,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            shrinkA=1.0,
            shrinkB=9.0,
            connectionstyle=f"arc3,rad={0.08 if idx % 2 == 0 else -0.08}",
            clip_on=True,
            zorder=3.2,
        )
        ax.add_patch(arrow)
        ax.scatter(
            [hub_xy[0]],
            [hub_xy[1]],
            transform=ax.transAxes,
            s=8.0 + 10.0 * strength,
            color=color,
            alpha=min(0.72, alpha + 0.18),
            edgecolors="white",
            linewidths=0.25,
            clip_on=True,
            zorder=3.3,
        )


def hyperedge_participants(hyperedges: pd.DataFrame) -> set[int]:
    participants: set[int] = set()
    for row in hyperedges.itertuples(index=False):
        participants.update(int(value) for value in row.subset_tuple)
        participants.add(int(row.target_index))
    return participants


def draw_nodes(
    ax: plt.Axes,
    nodes: pd.DataFrame,
    norm: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap,
    *,
    active_local_nodes: set[int] | None = None,
) -> None:
    active_local_nodes = active_local_nodes or set(nodes["local"].astype(int).tolist())
    active = nodes[nodes["local"].astype(int).isin(active_local_nodes)].copy()
    inactive = nodes[~nodes["local"].astype(int).isin(active_local_nodes)].copy()
    if not inactive.empty:
        ax.scatter(
            np.radians(inactive["lon"].to_numpy()),
            np.radians(inactive["lat"].to_numpy()),
            s=58,
            color="#9aa0a6",
            edgecolors="white",
            linewidths=0.22,
            alpha=0.34,
            zorder=3.8,
        )
    if active.empty:
        return
    lon = np.radians(active["lon"].to_numpy())
    lat = np.radians(active["lat"].to_numpy())
    ax.scatter(
        lon,
        lat,
        s=368,
        c=active["hyper_ace_total"],
        cmap=cmap,
        norm=norm,
        edgecolors="#12333d",
        linewidths=0.34,
        alpha=0.94,
        zorder=4,
    )
    ax.scatter(
        lon,
        lat,
        s=188,
        c=active["hyper_acs_total"],
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        alpha=0.98,
        zorder=5,
    )
    add_labels(ax, active)


def plot_peid_synergy_map(
    nodes: pd.DataFrame,
    hyperedges: pd.DataFrame,
    output: Path,
    *,
    focal_paper_nodes: list[int] | None = None,
    focal_mode: str = "any",
    top_hyperedges: int = 4,
    significance_z: float = 2.0,
    cross_target_only: bool = True,
    save_svg: bool = False,
) -> Path:
    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    focal_paper_nodes = focal_paper_nodes or []
    panel_nodes = focal_paper_nodes if len(focal_paper_nodes) > 1 else focal_paper_nodes[:1]
    if not panel_nodes:
        panel_nodes = [local_to_paper(int(value)) for value in sorted(hyperedge_participants(hyperedges))[:1]]
    fig = plt.figure(figsize=(6.9, 4.1 * len(panel_nodes) + 0.75), constrained_layout=True)
    cmap = mpl.colormaps["YlGnBu"]
    vmax = max(0.010, float(nodes[["hyper_ace_total", "hyper_acs_total"]].to_numpy().max()))
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
    axes: list[plt.Axes] = []
    for index, paper_node in enumerate(panel_nodes, start=1):
        ax = fig.add_subplot(len(panel_nodes), 1, index, projection="mollweide")
        axes.append(ax)
        draw_world(ax, land, coastlines)
        add_geographic_ticks(ax)
        panel_edges = select_top_hyperedges(
            hyperedges,
            top_n=int(top_hyperedges),
            significance_z=float(significance_z),
            cross_target_only=bool(cross_target_only),
            focal_paper_nodes=[int(paper_node)],
            focal_mode=focal_mode,
        )
        active_nodes = hyperedge_participants(panel_edges)
        active_nodes.add(paper_to_local(int(paper_node)))
        draw_top_hyperedges(ax, nodes, panel_edges)
        draw_nodes(ax, nodes, norm, cmap, active_local_nodes=active_nodes)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, location="bottom", shrink=0.55, pad=0.08, aspect=24)
    cbar.set_label("PEID hyper-ACS (inner node) and hyper-ACE (outer ring)")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if save_svg:
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-maps", default=str(DEFAULT_COMPONENT_MAPS))
    parser.add_argument("--hyper-gateway-scores", default=str(DEFAULT_HYPER_GATEWAY))
    parser.add_argument("--hyperedges", default=str(DEFAULT_HYPEREDGES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-hyperedges", type=int, default=4)
    parser.add_argument("--significance-z", type=float, default=2.0)
    parser.add_argument("--focal-paper-nodes", default="0,1", help="Comma-separated paper node labels to focus on.")
    parser.add_argument("--focal-mode", choices=["any", "source", "target"], default="any")
    parser.add_argument("--include-self-target", action="store_true")
    parser.add_argument("--save-svg", action="store_true")
    args = parser.parse_args()

    maps = np.load(Path(args.component_maps).expanduser())["component_maps"]
    hyperedges = pd.read_csv(Path(args.hyperedges).expanduser())
    nodes = build_node_frame(
        maps,
        Path(args.hyper_gateway_scores).expanduser(),
        hyperedges,
        significance_z=float(args.significance_z),
    )
    output = plot_peid_synergy_map(
        nodes,
        hyperedges,
        Path(args.output).expanduser(),
        focal_paper_nodes=parse_node_list(args.focal_paper_nodes),
        focal_mode=str(args.focal_mode),
        top_hyperedges=int(args.top_hyperedges),
        significance_z=float(args.significance_z),
        cross_target_only=not bool(args.include_self_target),
        save_svg=bool(args.save_svg),
    )
    print(output)


if __name__ == "__main__":
    main()
