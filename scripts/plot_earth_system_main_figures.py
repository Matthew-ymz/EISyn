#!/usr/bin/env python3
"""Build the two manuscript-level Earth-system result figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_runge_exhaustive_tm_maps import (
    DEFAULT_COMPONENT_MAPS,
    DEFAULT_RESULT_DIR as RUNGE_RESULT_DIR,
    load_exhaustive_top10,
    load_nodes,
)
from scripts.plot_runge_gateway_mediator_map import (
    COASTLINE_URL,
    LAND_URL,
    draw_world,
    extract_lines,
    extract_polygons,
    load_geojson,
)
from scripts.plot_runge_source_pair_condensation import (
    ROBUSTNESS_K,
    build_metrics as build_source_pair_metrics,
    load_rankings,
)
FIG_DIR = ROOT / "fig"
RUNGE_TREND_CSV = (
    FIG_DIR
    / "runge_slp_daily_1948_2026_20260628"
    / "multistep_conditioned_ei_tm_targeted"
    / "forced_tm_edge_trends_H001_H060.csv"
)
UNICM_PHI_ROWS = (
    ROOT
    / "results"
    / "unicm_all_mode_target_phi_eid_cpu_bound4_n8192"
    / "all_mode_target_phi_eid_rows.csv"
)
UNICM_GREEDY_DIR = (
    ROOT / "results" / "unicm_phi_eid_greedy_decomposition_cpu_bound4_n8192"
)
UNICM_TARGET_XI_LEADS = (
    ROOT
    / "results"
    / "unicm_target_resolved_xi_tm_degree1_signed_n8192"
    / "target_resolved_xi_lead_summary.csv"
)
UNICM_CALIBRATION_SUMMARY = (
    ROOT
    / "results"
    / "unicm_synergy_regularized_forecast"
    / "summary.json"
)
HORIZONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60)
MODE_ORDER = (
    "nino",
    "nino12",
    "nino3",
    "nino4",
    "IOD",
    "IOB",
    "SIOD",
    "WWV",
    "NPMM",
    "SPMM",
    "TNA",
)
TARGET_MODE_ORDER = (
    "nino",
    "nino12",
    "nino3",
    "nino4",
    "WWV",
    "NPMM",
    "SPMM",
    "IOB",
    "IOD",
    "SIOD",
    "TNA",
)

BLUE = "#3F6F9F"
TEAL = "#2A9D8F"
ORANGE = "#D9822B"
VIOLET = "#8064A2"
INK = "#172033"
MID_GREY = "#8D96A5"
LIGHT_GREY = "#E9EDF2"
SOURCE_PAIR_COLORS = (
    ORANGE,
    "#426B8A",
    "#6F8EA5",
    "#8EA9B8",
    "#6D8F8A",
)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "axes.linewidth": 0.65,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color="#111111",
        clip_on=False,
    )


def save_figure(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        base.with_suffix(".png"),
        base.with_suffix(".svg"),
        base.with_suffix(".pdf"),
    ]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], bbox_inches="tight")
    plt.close(fig)
    return outputs


def align_and_expand_map_row(
    fig: plt.Figure,
    axes: list[plt.Axes],
    *,
    gap: float = 0.008,
    width_scale: float = 1.08,
) -> None:
    """Give the three map panels equal size and centre the middle panel."""
    fig.canvas.draw()
    positions = [ax.get_position() for ax in axes]
    current_span = positions[-1].x1 - positions[0].x0
    target_span = min(current_span * width_scale, 0.98)
    panel_width = (target_span - gap * (len(axes) - 1)) / len(axes)
    left = 0.5 - target_span / 2.0
    top = max(position.y1 for position in positions)
    height = panel_width * fig.get_figwidth() / (2.0 * fig.get_figheight())
    bottom = top - height

    for index, ax in enumerate(axes):
        ax.set_in_layout(False)
        ax.set_anchor("C")
        ax.set_position(
            [
                left + index * (panel_width + gap),
                bottom,
                panel_width,
                height,
            ]
        )


def add_latitude_ticks_only(ax: plt.Axes) -> None:
    latitudes = (-60, -30, 0, 30, 60)
    labels = ("60°S", "30°S", "0°", "30°N", "60°N")
    ax.set_xticks([])
    ax.set_yticks(np.radians(latitudes), labels)
    ax.tick_params(axis="y", labelsize=4.1, pad=1.0, length=1.6, width=0.35)


def _axes_xy(ax: plt.Axes, lon: float, lat: float) -> np.ndarray:
    display = ax.transData.transform((np.radians(lon), np.radians(lat)))
    return ax.transAxes.inverted().transform(display)


def draw_compact_runge_map(
    ax: plt.Axes,
    nodes: pd.DataFrame,
    frame: pd.DataFrame,
    land: list[list[tuple[float, float]]],
    coastlines: list[list[tuple[float, float]]],
    horizon: int,
) -> None:
    draw_world(ax, land, coastlines)
    add_latitude_ticks_only(ax)
    lookup = nodes.set_index("local")
    active = set(
        frame[["source_a_local", "source_b_local", "target_local"]]
        .to_numpy()
        .ravel()
        .astype(int)
    )
    sources = set(
        frame[["source_a_local", "source_b_local"]].to_numpy().ravel().astype(int)
    )
    targets = set(frame["target_local"].astype(int)) - sources
    inactive = nodes[~nodes["local"].isin(active)]
    ax.scatter(
        np.radians(inactive.lon),
        np.radians(inactive.lat),
        s=5,
        color="#A7ADB5",
        edgecolors="none",
        alpha=0.28,
        zorder=3,
    )
    for subset, color, size, edge in (
        (targets, TEAL, 28, "#153D42"),
        (sources, BLUE, 48, "#173852"),
    ):
        selected = nodes[nodes["local"].isin(subset)]
        ax.scatter(
            np.radians(selected.lon),
            np.radians(selected.lat),
            s=size,
            color=color,
            edgecolors=edge,
            linewidths=0.45,
            zorder=6,
        )
    values = frame["delta2_tm"].to_numpy(dtype=float)
    span = max(float(values.max() - values.min()), 1e-12)
    for index, row in enumerate(frame.itertuples(index=False)):
        src = [
            _axes_xy(ax, float(lookup.loc[node].lon), float(lookup.loc[node].lat))
            for node in (row.source_a_local, row.source_b_local)
        ]
        target = _axes_xy(
            ax,
            float(lookup.loc[row.target_local].lon),
            float(lookup.loc[row.target_local].lat),
        )
        midpoint = 0.5 * (src[0] + src[1])
        hub = 0.58 * midpoint + 0.42 * target
        direction = target - midpoint
        norm = max(float(np.linalg.norm(direction)), 1e-9)
        perpendicular = np.array([-direction[1], direction[0]]) / norm
        hub = np.clip(
            hub + (1 if index % 2 == 0 else -1) * 0.009 * perpendicular,
            (0.04, 0.08),
            (0.96, 0.92),
        )
        strength = (float(row.delta2_tm) - float(values.min())) / span
        width = 0.35 + 1.25 * strength
        alpha = 0.18 + 0.48 * strength
        for start in src:
            ax.plot(
                [start[0], hub[0]],
                [start[1], hub[1]],
                transform=ax.transAxes,
                color=VIOLET,
                linewidth=max(0.3, width * 0.68),
                alpha=max(0.14, alpha * 0.65),
                zorder=4,
            )
        ax.annotate(
            "",
            xy=target,
            xytext=hub,
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops={
                "arrowstyle": "-|>",
                "color": VIOLET,
                "linewidth": width,
                "alpha": alpha,
                "shrinkA": 0,
                "shrinkB": 3.5,
                "mutation_scale": 5.0,
            },
            zorder=5,
        )
        ax.scatter(
            [hub[0]],
            [hub[1]],
            transform=ax.transAxes,
            s=3.0 + 5.0 * strength,
            color=VIOLET,
            edgecolors="white",
            linewidths=0.2,
            alpha=min(0.86, alpha + 0.2),
            zorder=7,
        )
    for row in nodes[nodes["local"].isin(active)].itertuples(index=False):
        ax.text(
            np.radians(row.lon),
            np.radians(row.lat),
            str(int(row.paper)),
            ha="center",
            va="center",
            fontsize=3.8,
            fontweight="bold",
            color="white",
            path_effects=[pe.withStroke(linewidth=0.9, foreground="#111111")],
            zorder=8,
        )
    ax.text(
        0.5,
        1.04,
        rf"$H={horizon}$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.8,
        fontweight="bold",
    )


def load_runge_top10_matrix(
    result_dir: Path,
) -> tuple[dict[int, pd.DataFrame], list[str], np.ndarray]:
    frames = {
        horizon: load_exhaustive_top10(result_dir, horizon=horizon)
        for horizon in HORIZONS
    }
    recurrence: dict[str, int] = {}
    for frame in frames.values():
        for row in frame.itertuples(index=False):
            edge = f"{int(row.source_a_paper)}+{int(row.source_b_paper)}→{int(row.target_paper)}"
            recurrence[edge] = recurrence.get(edge, 0) + 1
    selected = [
        edge
        for edge, _ in sorted(recurrence.items(), key=lambda item: (-item[1], item[0]))[:9]
    ]
    matrix = np.full((len(selected), len(HORIZONS)), np.nan)
    lookup = {edge: idx for idx, edge in enumerate(selected)}
    for col, horizon in enumerate(HORIZONS):
        for row in frames[horizon].itertuples(index=False):
            edge = f"{int(row.source_a_paper)}+{int(row.source_b_paper)}→{int(row.target_paper)}"
            if edge in lookup:
                matrix[lookup[edge], col] = int(row.tm_rank)
    return frames, selected, matrix


def _great_circle_km(
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
) -> float:
    radius_km = 6371.0
    phi_a, phi_b = np.radians([lat_a, lat_b])
    delta_lon = np.radians(lon_b - lon_a)
    haversine = (
        np.sin((phi_b - phi_a) / 2.0) ** 2
        + np.cos(phi_a) * np.cos(phi_b) * np.sin(delta_lon / 2.0) ** 2
    )
    return float(2.0 * radius_km * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0))))


def build_pair01_geographic_coverage(
    rankings: dict[int, pd.DataFrame],
    nodes: pd.DataFrame,
    *,
    top_ks: tuple[int, ...] = ROBUSTNESS_K,
) -> pd.DataFrame:
    node_lookup = nodes.set_index("local")
    rows: list[dict[str, float | int]] = []
    for horizon, ranking in rankings.items():
        for top_k in top_ks:
            top = ranking.head(top_k)
            focal = top[(top["source_a"] == 0) & (top["source_b"] == 1)].copy()
            targets = focal["target"].astype(int).tolist()
            if not targets:
                rows.append(
                    {
                        "horizon": horizon,
                        "top_k": top_k,
                        "target_count": 0,
                        "max_target_span_km": 0.0,
                    }
                )
                continue

            coordinates = node_lookup.loc[targets, ["lat", "lon"]].to_numpy(dtype=float)
            distances = np.asarray(
                [
                    _great_circle_km(*coordinates[first], *coordinates[second])
                    for first in range(len(coordinates))
                    for second in range(first + 1, len(coordinates))
                ],
                dtype=float,
            )
            max_span = float(np.max(distances)) if distances.size else 0.0
            rows.append(
                {
                    "horizon": horizon,
                    "top_k": top_k,
                    "target_count": len(targets),
                    "max_target_span_km": max_span,
                }
            )
    return pd.DataFrame(rows)


def plot_runge_figure(
    output_base: Path,
    *,
    result_dir: Path = RUNGE_RESULT_DIR,
    component_maps: Path = DEFAULT_COMPONENT_MAPS,
    trend_csv: Path = RUNGE_TREND_CSV,
) -> list[Path]:
    nodes = load_nodes(component_maps)
    frames, _, _ = load_runge_top10_matrix(result_dir)
    rankings = load_rankings(result_dir, list(HORIZONS))
    effective, pair_weights, selected_pairs = build_source_pair_metrics(
        rankings,
        list(HORIZONS),
        top_k=200,
        robustness_k=ROBUSTNESS_K,
    )
    selected_pairs = selected_pairs[:5]
    coverage = build_pair01_geographic_coverage(
        rankings,
        nodes,
    )
    trends = pd.read_csv(trend_csv)
    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    positions = np.arange(len(HORIZONS), dtype=float)
    sparse_tick_horizons = (1, 5, 10, 20, 40, 60)
    sparse_tick_positions = [HORIZONS.index(horizon) for horizon in sparse_tick_horizons]

    fig = plt.figure(figsize=(7.2, 7.15), layout="constrained")
    grid = fig.add_gridspec(
        3,
        6,
        height_ratios=[0.68, 1.0, 1.0],
        hspace=0.16,
    )
    map_axes = [
        fig.add_subplot(grid[0, 0:2], projection="mollweide"),
        fig.add_subplot(grid[0, 2:4], projection="mollweide"),
        fig.add_subplot(grid[0, 4:6], projection="mollweide"),
    ]
    for label, horizon, ax in zip("abc", (1, 10, 60), map_axes, strict=True):
        draw_compact_runge_map(
            ax,
            nodes,
            frames[horizon],
            land,
            coastlines,
            horizon,
        )
        add_panel_label(ax, label, x=-0.04, y=1.05)

    ax_d = fig.add_subplot(grid[1, 0:2])
    cutoff_colors = {
        50: "#AAB4BF",
        100: "#8297AA",
        200: BLUE,
        500: "#294A65",
    }
    for top_k, frame in effective.groupby("top_k", sort=True):
        frame = frame.sort_values("horizon")
        primary = int(top_k) == 200
        ax_d.plot(
            positions,
            frame["valid_pair_count"],
            color=cutoff_colors[int(top_k)],
            linewidth=1.45 if primary else 0.9,
            marker="o" if primary else None,
            markersize=2.2,
            label=f"top-{int(top_k)}",
            zorder=3 if primary else 2,
        )
    primary_effective = effective[effective["top_k"] == 200].sort_values("horizon")
    ax_d.text(
        0.02,
        0.48,
        f"{primary_effective.iloc[0].valid_pair_count:.0f}",
        transform=ax_d.transAxes,
        color=BLUE,
        fontsize=5.5,
        fontweight="bold",
    )
    ax_d.text(
        0.98,
        0.04,
        f"{primary_effective.iloc[-1].valid_pair_count:.0f}",
        transform=ax_d.transAxes,
        ha="right",
        color=BLUE,
        fontsize=5.5,
        fontweight="bold",
    )
    ax_d.set_xticks(sparse_tick_positions, sparse_tick_horizons)
    ax_d.set_xlabel("Evaluated horizon, $H$")
    ax_d.set_ylabel("Source pairs retained in top-$K$")
    ax_d.grid(axis="y", color=LIGHT_GREY, linewidth=0.55)
    ax_d.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        fontsize=5.0,
        handlelength=1.3,
        columnspacing=0.8,
    )
    add_panel_label(ax_d, "d", x=-0.18, y=1.02)

    ax_e = fig.add_subplot(grid[1, 2:6])
    pair_pivot = (
        pair_weights.pivot(index="horizon", columns="pair", values="share")
        .reindex(HORIZONS)
        .fillna(0.0)
    )
    selected_arrays = [
        pair_pivot[pair].to_numpy(dtype=float)
        if pair in pair_pivot.columns
        else np.zeros(len(HORIZONS))
        for pair in selected_pairs
    ]
    selected_total = np.sum(selected_arrays, axis=0)
    area_values = [100.0 * values for values in selected_arrays]
    area_values.append(100.0 * np.maximum(0.0, 1.0 - selected_total))
    area_labels = selected_pairs + ["Other source pairs"]
    ax_e.stackplot(
        positions,
        area_values,
        labels=area_labels,
        colors=[*SOURCE_PAIR_COLORS[: len(selected_pairs)], "#E4E8EC"],
        linewidth=0.22,
        edgecolor="white",
    )
    ax_e.axvline(HORIZONS.index(20), color="#606870", linestyle=":", linewidth=0.7)
    ax_e.set_xlim(positions[0], positions[-1])
    ax_e.set_ylim(0, 100)
    ax_e.set_xticks(positions, HORIZONS)
    ax_e.set_yticks((0, 25, 50, 75, 100))
    ax_e.set_xlabel("Evaluated forecast horizon, $H$")
    ax_e.set_ylabel("Top-200 synergy-mass composition (%)")
    ax_e.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=5.0,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    add_panel_label(ax_e, "e", x=-0.085, y=1.02)

    ax_f = fig.add_subplot(grid[2, 0:2])
    primary_coverage = (
        coverage[coverage["top_k"] == 200]
        .set_index("horizon")
        .reindex(HORIZONS)
    )
    span_line = ax_f.plot(
        positions,
        primary_coverage["max_target_span_km"],
        color=TEAL,
        linewidth=1.45,
        marker="o",
        markersize=2.2,
        label="maximum target span",
    )[0]
    ax_f.set_xticks(sparse_tick_positions, sparse_tick_horizons)
    ax_f.set_xlabel("Evaluated horizon, $H$")
    ax_f.set_ylabel("Maximum target span (km)", color=TEAL)
    ax_f.tick_params(axis="y", colors=TEAL)
    ax_f.set_ylim(0, 21000.0)
    ax_f.set_yticks((0, 5000, 10000, 15000, 20000))
    ax_f.grid(axis="y", color=LIGHT_GREY, linewidth=0.55)
    ax_f_right = ax_f.twinx()
    target_line = ax_f_right.plot(
        positions,
        primary_coverage["target_count"],
        color=INK,
        linewidth=1.05,
        marker="s",
        markersize=2.0,
        label="distinct targets",
    )[0]
    ax_f_right.set_ylabel("Distinct targets", color=INK)
    ax_f_right.tick_params(axis="y", colors=INK)
    ax_f_right.spines["top"].set_visible(False)
    ax_f_right.spines["right"].set_linewidth(0.65)
    ax_f.legend(
        [span_line, target_line],
        ["maximum span", "distinct targets"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        fontsize=4.8,
        handlelength=1.3,
        columnspacing=0.7,
    )
    add_panel_label(ax_f, "f", x=-0.18, y=1.02)

    ax_g = fig.add_subplot(grid[2, 2:6])
    colors = {
        "0+6->32": BLUE,
        "0+1->28": ORANGE,
        "0+1->50": TEAL,
        "0+1->46": VIOLET,
    }
    for edge, frame in trends.groupby("edge_label_paper", sort=False):
        frame = frame.sort_values("horizon")
        color = colors.get(str(edge), MID_GREY)
        ax_g.plot(
            frame["horizon"],
            frame["delta2_tm"],
            color=color,
            linewidth=1.35,
            marker="o",
            markersize=2.5,
        )
        last = frame.iloc[-1]
        ax_g.text(
            float(last["horizon"]) + 1.2,
            float(last["delta2_tm"]),
            str(edge).replace("->", "→"),
            color=color,
            fontsize=5.5,
            va="center",
        )
    ax_g.axhline(0, color="#555555", linewidth=0.65)
    ax_g.set_xlim(0.5, 69)
    ax_g.set_ylim(-0.0004, 0.0215)
    ax_g.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    ax_g.set_xlabel("Forecast horizon, $H$")
    ax_g.set_ylabel(r"Synergistic increment, $\Delta_{2,\mathrm{TM}}$ (bits)")
    ax_g.grid(axis="y", color=LIGHT_GREY, linewidth=0.55)
    add_panel_label(ax_g, "g", x=-0.085, y=1.02)

    align_and_expand_map_row(fig, map_axes)
    outputs = save_figure(fig, output_base)
    coverage_records = (
        primary_coverage.reset_index()[
            [
                "horizon",
                "target_count",
                "max_target_span_km",
            ]
        ]
        .to_dict(orient="records")
    )
    summary = {
        "source_pair_top_k": 200,
        "maximum_target_span_definition": (
            "Largest great-circle distance between the component centres of all "
            "No.0 + No.1 targets retained in the global top-200 at each horizon."
        ),
        "pair_0_1_target_coverage": coverage_records,
        "outputs": [str(path) for path in outputs],
    }
    output_base.with_name(f"{output_base.name}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return outputs


def plot_unicm_figure(output_base: Path) -> list[Path]:
    phi_rows = pd.read_csv(UNICM_PHI_ROWS)
    phi = (
        phi_rows.groupby("lead", as_index=False)["phi_eid"]
        .agg(["mean", "std"])
        .reset_index()
    )
    order = pd.read_csv(UNICM_GREEDY_DIR / "unicm_phi_eid_greedy_order_summary.csv")
    totals = pd.read_csv(UNICM_GREEDY_DIR / "unicm_phi_eid_greedy_total_summary.csv")
    atoms = pd.read_csv(UNICM_GREEDY_DIR / "unicm_phi_eid_lead8_top_atoms.csv").head(8)
    target_xi = pd.read_csv(UNICM_TARGET_XI_LEADS)
    calibration = json.loads(UNICM_CALIBRATION_SUMMARY.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(7.2, 8.15), layout="constrained")
    grid = fig.add_gridspec(
        4,
        5,
        height_ratios=[0.78, 1.02, 1.02, 0.90],
        hspace=0.08,
    )

    ax_a = fig.add_subplot(grid[0, :])
    ax_a.fill_between(
        phi["lead"],
        phi["mean"] - phi["std"],
        phi["mean"] + phi["std"],
        color="#C9D8E8",
        alpha=0.7,
        linewidth=0,
    )
    ax_a.plot(
        phi["lead"],
        phi["mean"],
        color=BLUE,
        marker="o",
        markersize=2.7,
        linewidth=1.45,
    )
    peak = phi.loc[phi["mean"].idxmax()]
    ax_a.scatter([peak["lead"]], [peak["mean"]], s=30, color=ORANGE, zorder=5)
    ax_a.annotate(
        f"peak at lead {int(peak['lead'])}\n{peak['mean']:.3f} ± {peak['std']:.3f} bits",
        xy=(float(peak["lead"]), float(peak["mean"])),
        xytext=(11.0, 0.213),
        fontsize=5.7,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": ORANGE, "linewidth": 0.7},
        ha="left",
        va="center",
    )
    ax_a.axhline(0, color="#666666", linewidth=0.6, linestyle=":")
    ax_a.set_xlabel("Prediction lead (months)")
    ax_a.set_ylabel(r"Integrated increment, $\Xi$ (bits)")
    ax_a.yaxis.set_label_coords(-0.055, 0.50)
    ax_a.set_xlim(1, 24)
    ax_a.set_ylim(0, max(0.235, float((phi["mean"] + phi["std"]).max()) * 1.04))
    ax_a.grid(axis="y", color=LIGHT_GREY, linewidth=0.55)
    add_panel_label(ax_a, "a", x=-0.04, y=1.02)

    ax_b = fig.add_subplot(grid[1, 0:2])
    pivot = order.pivot(index="lead", columns="order", values="mean").fillna(0.0)
    grouped = pd.DataFrame(index=pivot.index)
    for degree in (2, 3, 4, 5):
        grouped[f"order {degree}"] = pivot.get(degree, 0.0)
    grouped["orders 6–11"] = pivot[[col for col in pivot.columns if int(col) >= 6]].sum(axis=1)
    stack_colors = ["#38598C", "#547AA5", "#7393B3", "#95AEC4", "#C7CDD4"]
    ax_b.stackplot(
        grouped.index,
        [grouped[col].to_numpy() for col in grouped.columns],
        labels=list(grouped.columns),
        colors=stack_colors,
        linewidth=0,
    )
    ax_b.plot(
        totals["lead"],
        totals["phi_atom_sum_mean"],
        color="#111111",
        linewidth=1.0,
        label="atom sum",
    )
    ax_b.axvline(8, color=ORANGE, linewidth=0.85, linestyle="--")
    ax_b.set_xlabel("Prediction lead (months)")
    ax_b.set_ylabel(r"Hierarchical $\xi_C$ mass (bits)")
    ax_b.set_xlim(1, 24)
    ax_b.set_ylim(0, 0.195)
    ax_b.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        fontsize=5.1,
        handlelength=1.3,
        columnspacing=0.9,
    )
    add_panel_label(ax_b, "b", x=-0.16, y=1.02)

    sub = grid[1, 2:5].subgridspec(1, 2, width_ratios=[0.74, 1.15], wspace=0.04)
    ax_bar = fig.add_subplot(sub[0, 0])
    ax_matrix = fig.add_subplot(sub[0, 1])
    y = np.arange(len(atoms))
    order_color = {2: "#38598C", 3: "#547AA5", 4: "#7393B3", 5: "#95AEC4"}
    colors = [order_color.get(int(value), "#B7C0C9") for value in atoms["order"]]
    ax_bar.barh(
        y,
        atoms["mean"],
        xerr=atoms["std"],
        color=colors,
        edgecolor="#26323F",
        linewidth=0.3,
        error_kw={"ecolor": "#56616F", "elinewidth": 0.65, "capsize": 1.5},
    )
    ax_bar.set_yticks(y, atoms["atom"])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel(r"Lead-8 $\xi_C$ (bits)")
    ax_bar.set_ylabel("Hierarchical atom")
    ax_bar.grid(axis="x", color=LIGHT_GREY, linewidth=0.5)
    ax_bar.set_xlim(0, max(0.055, float((atoms["mean"] + atoms["std"]).max()) * 1.08))
    ax_bar.ticklabel_format(
        axis="x",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    ax_bar.xaxis.get_offset_text().set_fontsize(4.8)
    for idx, row in atoms.iterrows():
        ax_bar.text(
            min(float(row["mean"]) + 0.001, ax_bar.get_xlim()[1] * 0.78),
            idx,
            f"{100 * float(row['fraction']):.1f}%",
            va="center",
            fontsize=4.8,
            color=INK,
        )
    add_panel_label(ax_bar, "c", x=-0.30, y=1.02)

    x = np.arange(len(MODE_ORDER))
    for yi, row in enumerate(atoms.itertuples(index=False)):
        members = set(str(row.sources).split("|"))
        selected = [idx for idx, mode in enumerate(MODE_ORDER) if mode in members]
        if selected:
            ax_matrix.plot(
                selected,
                [yi] * len(selected),
                color="#A6AFBA",
                linewidth=0.8,
                zorder=1,
            )
        ax_matrix.scatter(x, np.full_like(x, yi), s=7, color="#E3E7EB", zorder=2)
        ax_matrix.scatter(
            selected,
            [yi] * len(selected),
            s=20,
            color=INK,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )
    ax_matrix.set_xlim(-0.5, len(MODE_ORDER) - 0.5)
    ax_matrix.set_ylim(len(atoms) - 0.5, -0.5)
    ax_matrix.set_xticks(
        x,
        ["ENSO" if value == "nino" else value for value in MODE_ORDER],
        rotation=55,
        ha="right",
    )
    ax_matrix.set_yticks(y, atoms["atom"])
    ax_matrix.tick_params(length=0, labelsize=4.8)
    for spine in ax_matrix.spines.values():
        spine.set_visible(False)
    ax_matrix.text(
        0.0,
        1.02,
        "source-mode membership",
        transform=ax_matrix.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        fontweight="bold",
    )

    ax_d = fig.add_subplot(grid[2, :])
    target_heat = (
        target_xi.pivot(index="target", columns="lead", values="xi_mean")
        .reindex(TARGET_MODE_ORDER)
        .sort_index(axis=1)
    )
    target_values = target_heat.to_numpy(dtype=float)
    image = ax_d.imshow(
        target_values,
        aspect="auto",
        interpolation="nearest",
        cmap="YlOrRd",
        norm=mpl.colors.Normalize(vmin=0.0, vmax=float(np.nanmax(target_values))),
    )
    ax_d.set_yticks(
        np.arange(len(TARGET_MODE_ORDER)),
        ["ENSO" if mode == "nino" else mode for mode in TARGET_MODE_ORDER],
    )
    ax_d.set_xticks(
        np.arange(0, len(target_heat.columns), 2),
        [str(int(value)) for value in target_heat.columns[::2]],
    )
    ax_d.set_xlabel("Prediction lead (months)")
    ax_d.set_ylabel("Predicted target mode")
    ax_d.axvline(5.5, color="#313131", linewidth=0.55, linestyle=":")
    ax_d.axvline(9.5, color="#313131", linewidth=0.55, linestyle=":")
    ax_d.text(
        7.5,
        -0.82,
        "lead 7–10",
        ha="center",
        va="bottom",
        fontsize=5.4,
        color="#444444",
        clip_on=False,
    )
    for boundary in (4.5, 6.5, 9.5):
        ax_d.axhline(boundary, color="white", linewidth=0.75)
    colorbar = fig.colorbar(image, ax=ax_d, fraction=0.018, pad=0.015)
    colorbar.set_label(r"Target-resolved $\Xi_j$ (bits)")
    add_panel_label(ax_d, "d", x=-0.055, y=1.02)

    metrics = calibration["test_metrics"]
    method_keys = ("frozen", "univariate", "uniform", "syn_regularized")
    method_labels = ("Frozen", "Univariate", "Uniform ridge", "Syn prior")
    method_values = np.asarray(
        [float(metrics[key]["mean_cell_nrmse"]) for key in method_keys]
    )
    method_colors = (MID_GREY, "#91A7CF", BLUE, ORANGE)

    ax_e = fig.add_subplot(grid[3, 0:2])
    method_y = np.arange(len(method_keys))[::-1]
    ax_e.scatter(
        method_values,
        method_y,
        s=31,
        color=method_colors,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    for value, y_value in zip(method_values, method_y):
        ax_e.text(
            value + 0.0021,
            y_value,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=5.4,
            color=INK,
        )
    ax_e.set_yticks(method_y, method_labels)
    ax_e.set_xlabel("Test normalized RMSE (lower is better)")
    ax_e.set_xlim(0.91, 1.012)
    ax_e.set_xticks((0.92, 0.96, 1.00))
    ax_e.set_ylim(-0.55, 3.55)
    ax_e.grid(axis="x", color=LIGHT_GREY, linewidth=0.5)
    total_gain = float(metrics["frozen"]["mean_cell_nrmse"]) - float(
        metrics["syn_regularized"]["mean_cell_nrmse"]
    )
    syn_gain = float(metrics["uniform"]["mean_cell_nrmse"]) - float(
        metrics["syn_regularized"]["mean_cell_nrmse"]
    )
    ax_e.text(
        0.01,
        1.04,
        f"total gain {total_gain:.3f}; Syn-specific gain {syn_gain:.3f}",
        transform=ax_e.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.3,
        color=INK,
    )
    add_panel_label(ax_e, "e", x=-0.18, y=1.04)

    ax_f = fig.add_subplot(grid[3, 2:5])
    uniform_score = float(metrics["uniform"]["mean_cell_nrmse"])
    random_scores = np.asarray(
        calibration["shuffled_syn_control"]["scores"],
        dtype=float,
    )
    random_repeats = int(calibration["shuffled_syn_control"]["repeats"])
    random_p = float(
        calibration["shuffled_syn_control"]["fraction_null_at_least_as_good"]
    )
    null_gains = uniform_score - random_scores
    rng = np.random.default_rng(20260728)
    jitter = rng.uniform(-0.15, 0.15, size=len(null_gains))
    ax_f.axvline(0, color="#686F78", linewidth=0.7, linestyle="--", zorder=1)
    ax_f.scatter(
        null_gains,
        jitter,
        s=15,
        color="#B4BFCC",
        edgecolor="white",
        linewidth=0.25,
        alpha=0.62,
        zorder=2,
    )
    ax_f.scatter(
        [syn_gain],
        [0],
        marker="D",
        s=40,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    ax_f.text(
        0.02,
        0.92,
        f"{random_repeats} shuffled Syn priors",
        transform=ax_f.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color="#657080",
    )
    ax_f.text(
        0.98,
        0.92,
        rf"$P={random_p:.3f}$",
        transform=ax_f.transAxes,
        ha="right",
        va="top",
        fontsize=5.6,
        color=INK,
    )
    ax_f.set_xlabel("Normalized RMSE gain over uniform ridge")
    ax_f.set_yticks([])
    ax_f.set_xlim(
        min(-0.017, float(null_gains.min()) - 0.002),
        max(0.024, syn_gain + 0.003),
    )
    ax_f.set_ylim(-0.38, 0.38)
    ax_f.spines["left"].set_visible(False)
    ax_f.grid(axis="x", color=LIGHT_GREY, linewidth=0.5)
    add_panel_label(ax_f, "f", x=-0.08, y=1.04)
    return save_figure(fig, output_base)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIG_DIR,
        help="Directory for the publication figure bundle.",
    )
    args = parser.parse_args()
    configure_matplotlib()
    runge_outputs = plot_runge_figure(
        Path(args.output_dir) / "earth_slp_hyperedge_dynamics"
    )
    unicm_outputs = plot_unicm_figure(
        Path(args.output_dir) / "earth_unicm_hierarchical_ei"
    )
    print(
        json.dumps(
            {
                "runge": [str(path) for path in runge_outputs],
                "unicm": [str(path) for path in unicm_outputs],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
