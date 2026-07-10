#!/usr/bin/env python3
"""Plot top second-order multistep Runge ridge hyperedges on a world map."""

from __future__ import annotations

import argparse
import json
import time
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
DEFAULT_RESULT_DIR = (
    ROOT
    / "results"
    / "runge_slp_daily_1948_2026_20260628"
    / "mlp_tm_ei_lag04"
    / "results"
    / "runge"
    / "multistep_conditioned_ei"
)
DEFAULT_OUTPUT = ROOT / "fig" / "runge_slp_daily_1948_2026_20260628" / "multistep_conditioned_ei" / "node0_top_order2_hyperedges.png"
DEFAULT_ALL_OUTPUT = ROOT / "fig" / "runge_slp_daily_1948_2026_20260628" / "multistep_conditioned_ei" / "top10_order2_hyperedges_H010.png"


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


def infer_final_horizon(result_dir: Path) -> int:
    manifest = result_dir / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if "final_horizon" in payload:
            return int(payload["final_horizon"])
    horizons = sorted(result_dir.glob("horizon_*/joint_ei.npy"))
    if not horizons:
        raise FileNotFoundError(f"No horizon_*/joint_ei.npy files found in {result_dir}")
    return int(horizons[-1].parent.name.split("_")[-1])


def load_multistep_delta2(result_dir: Path, horizon: int, *, cumulative: bool) -> np.ndarray:
    deltas: list[np.ndarray] = []
    horizons = range(1, int(horizon) + 1) if cumulative else [int(horizon)]
    for h in horizons:
        horizon_dir = result_dir / f"horizon_{h:03d}"
        pairwise = np.load(horizon_dir / "pairwise_ei.npy")
        joint = np.load(horizon_dir / "joint_ei.npy")
        if joint.shape != (pairwise.shape[0], pairwise.shape[0], pairwise.shape[0]):
            raise ValueError(f"Unexpected joint_ei shape at horizon {h}: {joint.shape}")
        delta = joint - pairwise[:, None, :] - pairwise[None, :, :]
        for idx in range(delta.shape[0]):
            delta[idx, idx, :] = np.nan
        deltas.append(delta)
    return np.nansum(np.stack(deltas, axis=0), axis=0)


def select_top_hyperedges(
    delta2: np.ndarray,
    *,
    top_n: int,
    focal_local: int | None,
    cross_target_only: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n = delta2.shape[0]
    for source_a in range(n):
        for source_b in range(source_a + 1, n):
            for target in range(n):
                if focal_local is not None and int(focal_local) not in {source_a, source_b, target}:
                    continue
                if cross_target_only and int(target) in {source_a, source_b}:
                    continue
                value = float(delta2[source_a, source_b, target])
                if not np.isfinite(value) or value <= 0.0:
                    continue
                rows.append(
                    {
                        "source_a": source_a,
                        "source_b": source_b,
                        "target_index": target,
                        "delta2": value,
                        "source_a_paper": local_to_paper(source_a),
                        "source_b_paper": local_to_paper(source_b),
                        "target_paper": local_to_paper(target),
                    }
                )
    if not rows:
        scope = "all nodes" if focal_local is None else f"local node {focal_local}"
        raise ValueError(f"No positive order-2 hyperedges found for {scope}.")
    return pd.DataFrame(rows).sort_values("delta2", ascending=False).head(int(top_n)).reset_index(drop=True)


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
            zorder=8,
        )


def offset_hub(source_mid: np.ndarray, target_xy: np.ndarray, idx: int) -> np.ndarray:
    direction = target_xy - source_mid
    length = float(np.linalg.norm(direction))
    if length < 1.0e-9:
        direction = np.array([1.0, 0.0])
        length = 1.0
    direction = direction / length
    perpendicular = np.array([-direction[1], direction[0]])
    side = -1.0 if idx % 2 else 1.0
    shell = 1 + (idx // 2) % 5
    hub = 0.58 * source_mid + 0.42 * target_xy + side * 0.012 * shell * perpendicular
    return np.clip(hub, np.array([0.04, 0.08]), np.array([0.96, 0.92]))


def draw_hyperedges(ax: plt.Axes, nodes: pd.DataFrame, hyperedges: pd.DataFrame, focal_local: int | None) -> None:
    lookup = nodes.set_index("local")
    active: set[int] = set()
    if focal_local is not None:
        active.add(int(focal_local))
    for row in hyperedges.itertuples(index=False):
        active.update([int(row.source_a), int(row.source_b), int(row.target_index)])

    inactive_nodes = nodes[~nodes["local"].astype(int).isin(active)]
    ax.scatter(
        np.radians(inactive_nodes["lon"].to_numpy()),
        np.radians(inactive_nodes["lat"].to_numpy()),
        s=48,
        color="#9aa0a6",
        edgecolors="white",
        linewidths=0.22,
        alpha=0.28,
        zorder=3,
    )

    active_nodes = nodes[nodes["local"].astype(int).isin(active)]
    colors = [
        "#1f78b4" if focal_local is not None and int(v) == int(focal_local) else "#2a9d8f"
        for v in active_nodes["local"]
    ]
    sizes = [
        350 if focal_local is not None and int(v) == int(focal_local) else 210
        for v in active_nodes["local"]
    ]
    ax.scatter(
        np.radians(active_nodes["lon"].to_numpy()),
        np.radians(active_nodes["lat"].to_numpy()),
        s=sizes,
        color=colors,
        edgecolors="#18343c",
        linewidths=0.72,
        alpha=0.96,
        zorder=5,
    )

    values = hyperedges["delta2"].to_numpy(dtype=float)
    vmin, vmax = float(values.min()), float(values.max())
    denom = max(vmax - vmin, 1.0e-12)
    edge_color = "#7c2d6c"
    for idx, row in enumerate(hyperedges.itertuples(index=False)):
        sources = (int(row.source_a), int(row.source_b))
        target = int(row.target_index)
        source_xy = []
        for source in sources:
            node = lookup.loc[source]
            source_xy.append(to_axes_xy(ax, float(node.lon), float(node.lat)))
        target_node = lookup.loc[target]
        target_xy = to_axes_xy(ax, float(target_node.lon), float(target_node.lat))
        source_mid = 0.5 * (source_xy[0] + source_xy[1])
        hub = offset_hub(source_mid, target_xy, idx)
        strength = (float(row.delta2) - vmin) / denom
        linewidth = 0.55 + 2.35 * strength
        alpha = 0.22 + 0.50 * strength
        for start_xy in source_xy:
            ax.plot(
                [start_xy[0], hub[0]],
                [start_xy[1], hub[1]],
                transform=ax.transAxes,
                color=edge_color,
                linewidth=max(0.45, linewidth * 0.72),
                alpha=max(0.14, alpha * 0.55),
                solid_capstyle="round",
                zorder=4.2,
            )
        arrow = mpatches.FancyArrowPatch(
            posA=hub,
            posB=target_xy,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=5.5 + 3.0 * strength,
            linewidth=linewidth,
            color=edge_color,
            alpha=alpha,
            shrinkA=1.0,
            shrinkB=9.0,
            connectionstyle=f"arc3,rad={0.075 if idx % 2 == 0 else -0.075}",
            clip_on=True,
            zorder=4.4,
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
            alpha=min(0.84, alpha + 0.22),
            zorder=7,
        )
    draw_labels(ax, nodes, active)


def plot_figure(
    nodes: pd.DataFrame,
    hyperedges: pd.DataFrame,
    output: Path,
    *,
    focal_local: int | None,
    horizon: int,
    cumulative: bool,
) -> Path:
    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    fig = plt.figure(figsize=(7.7, 4.35), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="mollweide")
    draw_world(ax, land, coastlines)
    add_geographic_ticks(ax)
    draw_hyperedges(ax, nodes, hyperedges, focal_local)
    mode = f"H=1..{int(horizon)} cumulative" if cumulative else f"H={int(horizon)}"
    scope = "second-order hyperedges" if focal_local is None else f"second-order hyperedges incident on No.{local_to_paper(focal_local)}"
    ax.set_title(
        f"Top {len(hyperedges)} {scope} ({mode})",
        fontsize=8.4,
        fontweight="bold",
        pad=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=350, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--component-maps", type=Path, default=DEFAULT_COMPONENT_MAPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--focal-local", type=int, default=0)
    parser.add_argument("--all-hyperedges", action="store_true", help="Rank all source-pair -> target hyperedges instead of a focal-node subset.")
    parser.add_argument("--include-self-target", action="store_true", help="Allow target to be one of the two sources.")
    parser.add_argument("--horizon", type=int, default=0, help="0 means manifest final_horizon.")
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--single-horizon", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    result_dir = Path(args.result_dir).expanduser()
    horizon = int(args.horizon) if int(args.horizon) > 0 else infer_final_horizon(result_dir)
    cumulative = not bool(args.single_horizon)
    focal_local = None if bool(args.all_hyperedges) else int(args.focal_local)
    output = Path(args.output).expanduser()
    if bool(args.all_hyperedges) and output == DEFAULT_OUTPUT:
        output = DEFAULT_ALL_OUTPUT

    nodes = load_nodes(Path(args.component_maps).expanduser())
    delta2 = load_multistep_delta2(result_dir, horizon, cumulative=cumulative)
    hyperedges = select_top_hyperedges(
        delta2,
        top_n=int(args.top_n),
        focal_local=focal_local,
        cross_target_only=not bool(args.include_self_target),
    )
    output = plot_figure(nodes, hyperedges, output, focal_local=focal_local, horizon=horizon, cumulative=cumulative)

    table_path = output.with_suffix(".csv")
    hyperedges.to_csv(table_path, index=False)
    elapsed = time.perf_counter() - started
    print(output)
    print(table_path)
    print(hyperedges[["source_a_paper", "source_b_paper", "target_paper", "delta2"]].to_string(index=False))
    print(f"elapsed_seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()
