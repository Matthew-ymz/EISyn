#!/usr/bin/env python3
"""Plot SLP and T2M PEID node maps with matched geographic styling."""

from __future__ import annotations

import argparse
import ast
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
    component_center,
    draw_world,
    extract_lines,
    extract_polygons,
    load_geojson,
    local_to_paper,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "reports" / "assets" / "part2_runge_slp_t2m_peid_comparison_map.png"


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


def _component_coordinates(payload: np.lib.npyio.NpzFile, component_maps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "lat" in payload:
        lat = np.asarray(payload["lat"], dtype=float)
    else:
        lat = np.linspace(-90.0, 90.0, component_maps.shape[0])
    if "lon" in payload:
        lon = np.asarray(payload["lon"], dtype=float)
    else:
        lon = np.linspace(0.0, 360.0, component_maps.shape[1], endpoint=False)
    lon = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    return lat, lon[order], component_maps[:, order, :]


def _parse_subset(value: object) -> tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(int(item) for item in value)
    text = str(value)
    if "+" in text:
        return tuple(int(part.strip().replace("component_", "")) - 1 for part in text.split("+"))
    if "," in text and not text.strip().startswith("[") and not text.strip().startswith("("):
        return tuple(int(part.strip()) for part in text.split(",") if part.strip())
    parsed = ast.literal_eval(text)
    if isinstance(parsed, int):
        return (int(parsed),)
    return tuple(int(item) for item in parsed)


def _order2_hyperedges(path: Path, *, significance_z: float | None) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()
    frame = frame[frame["order"].astype(int) == 2].copy()
    if "delta_K" in frame:
        frame = frame[frame["delta_K"].astype(float) > 0.0].copy()
    if significance_z is not None and "z" in frame:
        frame = frame[np.abs(frame["z"].astype(float)) >= float(significance_z)].copy()
    if "subset_tuple" not in frame:
        if "subset" in frame:
            frame["subset_tuple"] = frame["subset"].apply(_parse_subset)
        elif {"source_a_index", "source_b_index"}.issubset(frame.columns):
            frame["subset_tuple"] = frame.apply(lambda row: (int(row["source_a_index"]), int(row["source_b_index"])), axis=1)
        else:
            frame["subset_tuple"] = frame["subset_str"].apply(_parse_subset)
    return frame.reset_index(drop=True)


def build_dataset_nodes(
    *,
    component_maps_path: Path,
    pairwise_gateway_path: Path,
    pairwise_mediator_path: Path,
    hyperedges_path: Path,
    significance_z: float | None,
) -> pd.DataFrame:
    payload = np.load(component_maps_path)
    component_maps = np.asarray(payload["component_maps"], dtype=float)
    lat, lon, maps = _component_coordinates(payload, component_maps)
    n_components = component_maps.shape[2]
    gateway = pd.read_csv(pairwise_gateway_path).copy()
    mediator = pd.read_csv(pairwise_mediator_path).copy()
    hyperedges = _order2_hyperedges(hyperedges_path, significance_z=significance_z)

    if "component_index" not in gateway:
        gateway["component_index"] = gateway["component"].astype(int)
    if "component_index" not in mediator:
        mediator["component_index"] = mediator["component"].astype(int)

    ace = gateway.set_index("component_index")["ace"].reindex(range(n_components), fill_value=0.0).to_numpy(dtype=float)
    acs = gateway.set_index("component_index")["acs"].reindex(range(n_components), fill_value=0.0).to_numpy(dtype=float)
    source_share = np.zeros(n_components, dtype=float)
    target_share = np.zeros(n_components, dtype=float)
    for row in hyperedges.itertuples(index=False):
        delta = abs(float(row.delta_K))
        subset = tuple(int(value) for value in row.subset_tuple)
        if len(subset) == 0:
            continue
        for source in subset:
            if 0 <= source < n_components:
                source_share[source] += delta / float(len(subset))
        target = int(row.target_index)
        if 0 <= target < n_components:
            target_share[target] += delta
    denom = max(1, n_components - 1)
    hyper_ace = ace + source_share / denom
    hyper_acs = acs + target_share / denom
    hyper_amce = source_share / denom

    rows: list[dict[str, float | int]] = []
    for local in range(n_components):
        center_lon, center_lat = component_center(maps[..., local], lat, lon)
        rows.append(
            {
                "local": local,
                "paper": local_to_paper(local),
                "lon": center_lon,
                "lat": center_lat,
                "ace": float(hyper_ace[local]),
                "acs": float(hyper_acs[local]),
                "amce": float(hyper_amce[local]),
                "order2_source_share": float(source_share[local] / denom),
                "order2_target_share": float(target_share[local] / denom),
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


def plot_comparison(slp_nodes: pd.DataFrame, t2m_nodes: pd.DataFrame, output: Path, *, save_svg: bool) -> Path:
    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    fig = plt.figure(figsize=(10.8, 8.1), constrained_layout=True)
    axes = [
        fig.add_subplot(2, 2, 1, projection="mollweide"),
        fig.add_subplot(2, 2, 2, projection="mollweide"),
        fig.add_subplot(2, 2, 3, projection="mollweide"),
        fig.add_subplot(2, 2, 4, projection="mollweide"),
    ]
    for ax in axes:
        draw_world(ax, land, coastlines)
        add_geographic_ticks(ax)

    slp_ace_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(0.002, float(slp_nodes[["ace", "acs"]].to_numpy().max())))
    slp_amce_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(0.0002, float(slp_nodes["amce"].max())))
    t2m_ace_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(0.018, float(t2m_nodes[["ace", "acs"]].to_numpy().max())))
    t2m_amce_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(0.0010, float(t2m_nodes["amce"].max())))

    draw_ace_acs(axes[0], slp_nodes, slp_ace_norm)
    draw_amce(axes[1], slp_nodes, slp_amce_norm)
    draw_ace_acs(axes[2], t2m_nodes, t2m_ace_norm)
    draw_amce(axes[3], t2m_nodes, t2m_amce_norm)

    panel_labels = ["a", "b", "c", "d"]
    titles = ["SLP with second-order PEID", "SLP with second-order PEID", "T2M with second-order PEID", "T2M with second-order PEID"]
    for label, title, ax in zip(panel_labels, titles, axes, strict=True):
        ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=16, fontweight="bold")
        ax.text(0.5, 1.06, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")

    add_colorbar(fig, axes[0], slp_ace_norm, "OrRd", "SLP Hyper-ACS (inner node) and Hyper-ACE (outer ring)")
    add_colorbar(fig, axes[1], slp_amce_norm, "Greens", "SLP Hyper-AMCE")
    add_colorbar(fig, axes[2], t2m_ace_norm, "OrRd", "T2M Hyper-ACS (inner node) and Hyper-ACE (outer ring)")
    add_colorbar(fig, axes[3], t2m_amce_norm, "Greens", "T2M Hyper-AMCE")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if save_svg:
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slp-component-maps", default=str(ROOT / "results" / "runge" / "2015_gateways" / "component_maps.npz"))
    parser.add_argument("--slp-gateway", default=str(ROOT / "results" / "runge" / "pairwise_mlp_tm_ei_path_effects" / "gateway_scores.csv"))
    parser.add_argument("--slp-mediator", default=str(ROOT / "results" / "runge" / "pairwise_mlp_tm_ei_path_effects" / "mediator_scores.csv"))
    parser.add_argument("--slp-hyperedges", default=str(ROOT / "results" / "runge" / "peid_hypergraph" / "peid_hyperedges.csv"))
    parser.add_argument("--t2m-component-maps", default=str(ROOT / "results" / "runge_t2m_daily_weekmean" / "component_maps.npz"))
    parser.add_argument("--t2m-gateway", default=str(ROOT / "results" / "runge_t2m_daily_weekmean" / "pairwise_tm_ei_path_effects" / "gateway_scores.csv"))
    parser.add_argument("--t2m-mediator", default=str(ROOT / "results" / "runge_t2m_daily_weekmean" / "pairwise_tm_ei_path_effects" / "mediator_scores.csv"))
    parser.add_argument("--t2m-hyperedges", default=str(ROOT / "results" / "runge_t2m_daily_weekmean" / "peid_hypergraph" / "peid_hyperedges.csv"))
    parser.add_argument("--slp-significance-z", type=float, default=2.0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--save-svg", action="store_true")
    args = parser.parse_args()

    slp_nodes = build_dataset_nodes(
        component_maps_path=Path(args.slp_component_maps).expanduser(),
        pairwise_gateway_path=Path(args.slp_gateway).expanduser(),
        pairwise_mediator_path=Path(args.slp_mediator).expanduser(),
        hyperedges_path=Path(args.slp_hyperedges).expanduser(),
        significance_z=float(args.slp_significance_z),
    )
    t2m_nodes = build_dataset_nodes(
        component_maps_path=Path(args.t2m_component_maps).expanduser(),
        pairwise_gateway_path=Path(args.t2m_gateway).expanduser(),
        pairwise_mediator_path=Path(args.t2m_mediator).expanduser(),
        hyperedges_path=Path(args.t2m_hyperedges).expanduser(),
        significance_z=None,
    )
    output = plot_comparison(slp_nodes, t2m_nodes, Path(args.output).expanduser(), save_svg=bool(args.save_svg))
    print(output)


if __name__ == "__main__":
    main()
