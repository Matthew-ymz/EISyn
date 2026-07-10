#!/usr/bin/env python3
"""Analyze TM-reranked Runge hyperedge trends across forecast horizons."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_runge_gateway_mediator_map import DEFAULT_COMPONENT_MAPS
from plot_runge_multistep_ridge_node0_hyperedges import load_nodes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TM_DIR = (
    ROOT
    / "results"
    / "runge_slp_daily_1948_2026_20260628"
    / "mlp_tm_ei_lag04"
    / "results"
    / "runge"
    / "multistep_conditioned_ei_tm_targeted"
)
DEFAULT_FIG_DIR = ROOT / "fig" / "runge_slp_daily_1948_2026_20260628" / "multistep_conditioned_ei_tm_targeted"
DEFAULT_HORIZONS = "1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60"
DEFAULT_EDGES = "0+6->32,0+1->28,0+1->50,0+1->46"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one horizon is required.")
    return sorted(dict.fromkeys(values))


def parse_edge_list(text: str) -> list[str]:
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one edge label is required.")
    return values


def paper_edge_label(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(
        lambda row: f"{int(row.source_a_paper)}+{int(row.source_b_paper)}->{int(row.target_paper)}",
        axis=1,
    )


def load_top10(tm_dir: Path, fig_dir: Path) -> pd.DataFrame:
    candidates = [
        fig_dir / "top10_order2_hyperedges_by_horizon_H001_H060_tm_reranked.csv",
        tm_dir / "top10_order2_hyperedges_by_horizon_H001_H060_tm_reranked.csv",
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            frame["edge_label_paper"] = paper_edge_label(frame)
            frame["pair_label"] = frame.apply(
                lambda row: f"{int(row.source_a_paper)}+{int(row.source_b_paper)}",
                axis=1,
            )
            return frame
    raise FileNotFoundError("TM-reranked cross-horizon top10 table was not found.")


def load_horizon_cache(tm_dir: Path, horizon: int) -> pd.DataFrame:
    path = tm_dir / f"H{int(horizon):03d}_discrete_top1000_tm_rerank.csv"
    frame = pd.read_csv(path)
    frame["horizon"] = int(horizon)
    frame["edge_label_paper"] = paper_edge_label(frame)
    return frame


def great_circle_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_km = 6371.0
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    value = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * radius_km * np.arcsin(np.sqrt(value)))


def add_distances(top10: pd.DataFrame, component_maps: Path) -> pd.DataFrame:
    nodes = load_nodes(component_maps).set_index("local")
    rows: list[dict[str, object]] = []
    for _, row in top10.iterrows():
        source_a = nodes.loc[int(row.source_a)]
        source_b = nodes.loc[int(row.source_b)]
        target = nodes.loc[int(row.target_index)]
        dist_a_target = great_circle_km(source_a.lon, source_a.lat, target.lon, target.lat)
        dist_b_target = great_circle_km(source_b.lon, source_b.lat, target.lon, target.lat)
        dist_sources = great_circle_km(source_a.lon, source_a.lat, source_b.lon, source_b.lat)
        item = row.to_dict()
        item.update(
            {
                "dist_source_a_target_km": dist_a_target,
                "dist_source_b_target_km": dist_b_target,
                "dist_source_pair_km": dist_sources,
                "dist_nearest_source_target_km": min(dist_a_target, dist_b_target),
                "dist_farthest_source_target_km": max(dist_a_target, dist_b_target),
                "dist_mean_source_target_km": 0.5 * (dist_a_target + dist_b_target),
                "dist_max_node_span_km": max(dist_a_target, dist_b_target, dist_sources),
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def trajectory_table(tm_dir: Path, horizons: list[int], edge_labels: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        frame = load_horizon_cache(tm_dir, horizon)
        for edge in edge_labels:
            hit = frame[frame["edge_label_paper"] == edge]
            if hit.empty:
                rows.append(
                    {
                        "horizon": horizon,
                        "edge_label_paper": edge,
                        "delta2_tm": np.nan,
                        "tm_rank_within_discrete_top1000": np.nan,
                        "in_discrete_top1000": False,
                    }
                )
            else:
                first = hit.sort_values("tm_rank_within_discrete_top1000").iloc[0]
                rows.append(
                    {
                        "horizon": horizon,
                        "edge_label_paper": edge,
                        "delta2_tm": float(first.delta2_tm),
                        "tm_rank_within_discrete_top1000": int(first.tm_rank_within_discrete_top1000),
                        "in_discrete_top1000": True,
                    }
                )
    return pd.DataFrame(rows)


def bin_label(horizon: int) -> str:
    if int(horizon) <= 5:
        return "H<=5"
    if int(horizon) <= 15:
        return "6<=H<=15"
    return "H>=20"


def summarize_distances(with_distances: pd.DataFrame) -> pd.DataFrame:
    frame = with_distances.copy()
    frame["horizon_bin"] = frame["horizon"].map(bin_label)
    grouped = (
        frame.groupby("horizon_bin", sort=False)
        .agg(
            n=("horizon", "size"),
            median_nearest_source_target_km=("dist_nearest_source_target_km", "median"),
            median_mean_source_target_km=("dist_mean_source_target_km", "median"),
            median_farthest_source_target_km=("dist_farthest_source_target_km", "median"),
            median_source_pair_km=("dist_source_pair_km", "median"),
            median_max_node_span_km=("dist_max_node_span_km", "median"),
            mean_delta2_tm=("delta2_tm", "mean"),
            max_delta2_tm=("delta2_tm", "max"),
        )
        .reset_index()
    )
    return grouped


def plot_trends(
    trajectories: pd.DataFrame,
    with_distances: pd.DataFrame,
    distance_summary: pd.DataFrame,
    horizons: list[int],
    output: Path,
) -> None:
    major_horizon_ticks = [h for h in [1, 5, 10, 20, 40, 60] if h in set(horizons)]
    fig = plt.figure(figsize=(7.3, 4.9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.18, 1.0])
    ax_traj = fig.add_subplot(grid[:, 0])
    ax_dist = fig.add_subplot(grid[0, 1])
    ax_bin = fig.add_subplot(grid[1, 1])

    colors = {
        "0+6->32": "#4c78a8",
        "0+1->28": "#f58518",
        "0+1->50": "#54a24b",
        "0+1->46": "#b279a2",
    }
    for edge, subset in trajectories.groupby("edge_label_paper", sort=False):
        subset = subset.sort_values("horizon")
        plotted = subset[subset["in_discrete_top1000"]]
        ax_traj.plot(
            plotted["horizon"],
            plotted["delta2_tm"],
            marker="o",
            linewidth=1.6,
            markersize=3.8,
            color=colors.get(str(edge), "#666666"),
            label=str(edge),
        )
    ax_traj.set_xlabel("Horizon H")
    ax_traj.set_ylabel(r"$\Delta_{2,\mathrm{TM}}$")
    ax_traj.set_xticks(major_horizon_ticks)
    ax_traj.grid(axis="y", color="#d8d8d8", linewidth=0.45)
    ax_traj.text(0.0, 1.02, "a", transform=ax_traj.transAxes, weight="bold", fontsize=8)
    ax_traj.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Hyperedge")

    by_h = (
        with_distances.groupby("horizon")
        .agg(
            median_mean_source_target_km=("dist_mean_source_target_km", "median"),
            median_farthest_source_target_km=("dist_farthest_source_target_km", "median"),
            median_source_pair_km=("dist_source_pair_km", "median"),
        )
        .reindex(horizons)
    )
    ax_dist.plot(
        horizons,
        by_h["median_mean_source_target_km"] / 1000.0,
        marker="o",
        linewidth=1.35,
        markersize=3.0,
        color="#4c78a8",
        label="mean source-target",
    )
    ax_dist.plot(
        horizons,
        by_h["median_farthest_source_target_km"] / 1000.0,
        marker="s",
        linewidth=1.35,
        markersize=3.0,
        color="#e45756",
        label="farthest source-target",
    )
    ax_dist.plot(
        horizons,
        by_h["median_source_pair_km"] / 1000.0,
        marker="^",
        linewidth=1.35,
        markersize=3.0,
        color="#54a24b",
        label="source-source",
    )
    ax_dist.set_xlabel("Horizon H")
    ax_dist.set_ylabel("Median distance (10$^3$ km)")
    ax_dist.set_xticks(major_horizon_ticks)
    ax_dist.grid(axis="y", color="#d8d8d8", linewidth=0.45)
    ax_dist.text(0.0, 1.04, "b", transform=ax_dist.transAxes, weight="bold", fontsize=8)
    ax_dist.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    bins = ["H<=5", "6<=H<=15", "H>=20"]
    summary = distance_summary.set_index("horizon_bin").reindex(bins)
    x = np.arange(len(bins))
    width = 0.24
    ax_bin.bar(
        x - width,
        summary["median_mean_source_target_km"] / 1000.0,
        width=width,
        color="#8ab6d6",
        label="mean source-target",
    )
    ax_bin.bar(
        x,
        summary["median_farthest_source_target_km"] / 1000.0,
        width=width,
        color="#f2a6a3",
        label="farthest source-target",
    )
    ax_bin.bar(
        x + width,
        summary["median_source_pair_km"] / 1000.0,
        width=width,
        color="#9dd29a",
        label="source-source",
    )
    ax_bin.set_xticks(x)
    ax_bin.set_xticklabels(bins)
    ax_bin.set_ylabel("Median distance (10$^3$ km)")
    ax_bin.grid(axis="y", color="#d8d8d8", linewidth=0.45)
    ax_bin.text(0.0, 1.04, "c", transform=ax_bin.transAxes, weight="bold", fontsize=8)
    ax_bin.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=450, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tm-dir", type=Path, default=DEFAULT_TM_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--component-maps", type=Path, default=DEFAULT_COMPONENT_MAPS)
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS)
    parser.add_argument("--edges", default=DEFAULT_EDGES)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FIG_DIR / "top10_order2_hyperedges_by_horizon_H001_H060_tm_trends.png",
    )
    args = parser.parse_args()

    horizons = parse_int_list(args.horizons)
    edge_labels = parse_edge_list(args.edges)
    tm_dir = Path(args.tm_dir).expanduser()
    fig_dir = Path(args.fig_dir).expanduser()
    top10 = load_top10(tm_dir, fig_dir)
    with_distances = add_distances(top10, Path(args.component_maps).expanduser())
    distance_summary = summarize_distances(with_distances)
    trajectories = trajectory_table(tm_dir, horizons, edge_labels)

    output = Path(args.output).expanduser()
    plot_trends(trajectories, with_distances, distance_summary, horizons, output)

    trend_csv = output.with_suffix(".csv")
    trajectories.to_csv(trend_csv, index=False)
    with_distances.to_csv(output.with_name(output.stem + "_top10_distances.csv"), index=False)
    distance_summary.to_csv(output.with_name(output.stem + "_distance_summary.csv"), index=False)

    normalized_top10 = fig_dir / "top10_order2_hyperedges_by_horizon_H001_H060_tm_reranked.csv"
    top10.to_csv(normalized_top10, index=False)
    result_top10 = tm_dir / "top10_order2_hyperedges_by_horizon_H001_H060_tm_reranked.csv"
    top10.to_csv(result_top10, index=False)
    print(output)


if __name__ == "__main__":
    main()
