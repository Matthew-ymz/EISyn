#!/usr/bin/env python3
"""Publication panels for confirmed NYC Taxi spatial synergy hyperedges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon

from plot_nyc_taxi_temporal_coupling_map import load_geojson, outer_rings, zone_id


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/nyc_taxi_mgstn_ei/spatial_hyperedges/spatial_hyperedge_full_summary.json"
OUTPUT = ROOT / "fig/nyc_taxi_spatial_hyperedge_panels"

TEAL = "#3E8882"
BLUE = "#4C78A8"
ORANGE = "#D18B47"
GRAY = "#87939A"
DARK = "#263238"
EDGE_COLORS = ["#3E8882", "#4C78A8", "#D18B47", "#8367A7", "#B85C70"]


def panel_label(ax, label: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def candidate_units(payload: dict, candidate_id: str) -> list[dict]:
    return [unit for unit in payload["units"] if unit["candidate_id"] == candidate_id]


def display_name(candidate: dict) -> str:
    substitutions = {
        "Upper East Side North": "UES North",
        "Upper East Side South": "UES South",
        "Upper West Side North": "UWS North",
        "Upper West Side South": "UWS South",
        "Financial District North": "FiDi North",
        "Financial District South": "FiDi South",
        "Greenwich Village North": "Greenwich Village N",
        "Greenwich Village South": "Greenwich Village S",
        "Sutton Place/Turtle Bay North": "Sutton Pl./Turtle Bay N",
        "TriBeCa/Civic Center": "TriBeCa/Civic Ctr.",
    }
    left = substitutions.get(candidate["source_a_name"], candidate["source_a_name"])
    right = substitutions.get(candidate["source_b_name"], candidate["source_b_name"])
    target = substitutions.get(candidate["target_name"], candidate["target_name"])
    return f"{left} + {right}\n→ {target}"


def ranked_candidates(payload: dict, maximum: int = 8) -> list[dict]:
    confirmed = [row for row in payload["candidates"] if row["confirmed"]]
    remaining = [row for row in payload["candidates"] if not row["confirmed"]]
    confirmed.sort(key=lambda row: row["paired_delta_mean_bits"], reverse=True)
    remaining.sort(key=lambda row: row["paired_delta_mean_bits"], reverse=True)
    return (confirmed + remaining)[:maximum]


def add_rank_panel(
    ax,
    payload: dict,
    *,
    label: str = "a",
    maximum: int = 8,
    compact_labels: bool = False,
) -> None:
    selected = ranked_candidates(payload, maximum=maximum)
    y = np.arange(len(selected))[::-1]
    rng = np.random.default_rng(31)
    plotted_values = []
    for position, (row, y_value) in enumerate(zip(selected, y)):
        units = candidate_units(payload, row["candidate_id"])
        values = np.asarray([unit["paired_delta_bits"] for unit in units], dtype=float)
        plotted_values.extend(values.tolist())
        color = TEAL if row["confirmed"] else GRAY
        jitter = rng.uniform(-0.10, 0.10, len(values))
        ax.scatter(values, np.full(len(values), y_value) + jitter, s=13,
                   facecolor="white", edgecolor=color, linewidth=0.55, alpha=0.72, zorder=2)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ax.errorbar(mean, y_value, xerr=sd, fmt="o", ms=5.2, color=color,
                    ecolor=color, elinewidth=1.15, capsize=2.2, zorder=3)
        if row["confirmed"]:
            ax.text(mean, y_value + 0.24, str(position + 1), color=color,
                    fontsize=6.8, ha="center", va="bottom", fontweight="bold")
    ax.axvline(0.0, color="#6D777D", lw=0.75, zorder=0)
    plotted_values = np.asarray(plotted_values)
    span = float(np.ptp(plotted_values))
    ax.set_xlim(float(plotted_values.min() - 0.08 * span), float(plotted_values.max() + 0.08 * span))
    labels = [f"H{index + 1}" for index in range(len(selected))] if compact_labels else [
        display_name(row) for row in selected
    ]
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", labelsize=7.2 if compact_labels else 6.6, pad=2)
    ax.set_xlabel("Synergy above shuffled null (bits)")
    ax.set_ylim(-0.65, len(selected) - 0.35)
    ax.text(0.98, 0.98, f"{payload['confirmed_count']} / {payload['candidate_count']} confirmed",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.2, color="#657078")
    panel_label(ax, label, -0.04, 1.03)


def add_screen_confirmation_panel(ax, payload: dict, *, label: str = "b") -> None:
    candidates = payload["candidates"]
    screen = 1000 * np.asarray([row["screen_interaction_rms_z"] for row in candidates])
    formal = np.asarray([row["paired_delta_mean_bits"] for row in candidates])
    ax.scatter(screen, formal, s=18, color="#9AA5AA", alpha=0.58,
               edgecolor="white", linewidth=0.3)
    selected = ranked_candidates(payload, maximum=8)
    selected_ids = {row["candidate_id"] for row in selected}
    mask = np.asarray([row["candidate_id"] in selected_ids for row in candidates])
    ax.scatter(screen[mask], formal[mask], s=26, color=TEAL,
               edgecolor="white", linewidth=0.45, zorder=3)
    ax.axhline(0.0, color="#6D777D", lw=0.75, zorder=0)
    ax.set_xlabel("Screen interaction (10^-3 z)")
    ax.set_ylabel("Synergy above shuffled null (bits)")
    y_pad = 0.12 * float(np.ptp(formal))
    ax.set_ylim(float(formal.min() - y_pad), float(formal.max() + y_pad))
    panel_label(ax, label, -0.13, 1.03)


def polygon_area_centroid(ring: np.ndarray) -> tuple[float, np.ndarray]:
    points = np.asarray(ring, dtype=float)
    if len(points) < 3:
        return 0.0, points.mean(axis=0)
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    cross = points[:-1, 0] * points[1:, 1] - points[1:, 0] * points[:-1, 1]
    signed_area = 0.5 * cross.sum()
    if abs(signed_area) < 1e-14:
        return 0.0, points[:-1].mean(axis=0)
    centroid = np.asarray([
        np.sum((points[:-1, 0] + points[1:, 0]) * cross),
        np.sum((points[:-1, 1] + points[1:, 1]) * cross),
    ]) / (6.0 * signed_area)
    return abs(float(signed_area)), centroid


def feature_centroid(feature: dict) -> np.ndarray:
    parts = [polygon_area_centroid(ring) for ring in outer_rings(feature)]
    weights = np.asarray([area for area, _ in parts])
    centers = np.asarray([center for _, center in parts])
    if weights.sum() <= 0:
        return centers.mean(axis=0)
    return np.average(centers, axis=0, weights=weights)


def add_map_background(ax, features: list[dict], selected_ids: set[int]) -> None:
    all_patches = [Polygon(ring, closed=True) for feature in features for ring in outer_rings(feature)]
    ax.add_collection(PatchCollection(
        all_patches, facecolor="#EDF0F1", edgecolor="white", linewidth=0.20, zorder=0
    ))
    selected_patches = [
        Polygon(ring, closed=True)
        for feature in features if zone_id(feature) in selected_ids
        for ring in outer_rings(feature)
    ]
    ax.add_collection(PatchCollection(
        selected_patches, facecolor="#F8F9F7", edgecolor="#647078", linewidth=0.38, zorder=1
    ))


def add_hyperedge_map(ax, payload: dict, *, label: str = "b", maximum: int = 5) -> None:
    geojson = load_geojson()
    features = geojson["features"]
    feature_by_id = {zone_id(feature): feature for feature in features}
    selected_ids = {
        int(value)
        for candidate in payload["candidates"]
        for value in (candidate["source_a_id"], candidate["source_b_id"], candidate["target_id"])
    }
    add_map_background(ax, features, selected_ids)
    points = np.vstack([
        ring for location_id in selected_ids for ring in outer_rings(feature_by_id[location_id])
    ])
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    ax.set_xlim(x_min - 0.055 * (x_max - x_min), x_max + 0.055 * (x_max - x_min))
    ax.set_ylim(y_min - 0.02 * (y_max - y_min), y_max + 0.02 * (y_max - y_min))
    ax.set_aspect("equal")
    ax.set_axis_off()
    panel_label(ax, label, -0.02, 1.02)

    confirmed = [row for row in payload["candidates"] if row["confirmed"]]
    confirmed.sort(key=lambda row: row["paired_delta_mean_bits"], reverse=True)
    confirmed = confirmed[:maximum]
    if not confirmed:
        ax.text(0.5, 0.5, "No candidate passed\nthe confirmation threshold",
                transform=ax.transAxes, ha="center", va="center", color="#657078", fontsize=9)
        return
    strengths = np.asarray([row["paired_delta_mean_bits"] for row in confirmed])
    span = float(np.ptp(strengths))
    widths = 1.2 + 2.2 * ((strengths - strengths.min()) / span if span > 0 else np.ones(len(strengths)))
    for index, (row, width, color) in enumerate(zip(confirmed, widths, EDGE_COLORS), start=1):
        source_a = feature_centroid(feature_by_id[int(row["source_a_id"])])
        source_b = feature_centroid(feature_by_id[int(row["source_b_id"])])
        target = feature_centroid(feature_by_id[int(row["target_id"])])
        junction = 0.5 * (source_a + source_b)
        ax.plot([source_a[0], junction[0]], [source_a[1], junction[1]],
                color=color, lw=width, alpha=0.78, solid_capstyle="round", zorder=3)
        ax.plot([source_b[0], junction[0]], [source_b[1], junction[1]],
                color=color, lw=width, alpha=0.78, solid_capstyle="round", zorder=3)
        ax.add_patch(FancyArrowPatch(
            junction, target, arrowstyle="-|>", mutation_scale=8 + width,
            color=color, lw=width, alpha=0.85, shrinkA=2, shrinkB=4, zorder=3,
        ))
        ax.scatter([source_a[0], source_b[0]], [source_a[1], source_b[1]], s=22,
                   facecolor="white", edgecolor=color, linewidth=1.0, zorder=4)
        ax.scatter(target[0], target[1], s=28, marker="s", facecolor=color,
                   edgecolor="white", linewidth=0.65, zorder=4)
        ax.text(target[0], target[1], str(index), ha="center", va="center",
                color="white", fontsize=5.8, fontweight="bold", zorder=5)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=DARK, markersize=5, label="Source outflow"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=DARK,
               markeredgecolor="white", markersize=5.5, label="Target inflow"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.055),
              ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    audit = payload["nonnegative_audit"]
    if audit["violation_count"]:
        raise RuntimeError(
            f"formal spatial Syn failed nonnegativity audit: min={audit['minimum_raw_syn_bits']}, "
            f"count={audit['violation_count']}"
        )
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(11.2, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.32, 0.68], wspace=0.08)
    add_rank_panel(fig.add_subplot(grid[0, 0]), payload, maximum=6)
    second = fig.add_subplot(grid[0, 1])
    if payload["confirmed_count"]:
        add_hyperedge_map(second, payload, maximum=5)
    else:
        add_screen_confirmation_panel(second, payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(args.output.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
