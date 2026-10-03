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
from scripts.analyze_unicm_11mode_xi_hierarchy_tree import (
    _tree_from_record,
    render_trees,
)
FIG_DIR = ROOT / "fig"
RUNGE_TREND_CSV = (
    FIG_DIR
    / "runge_slp_daily_1948_2026_20260628"
    / "multistep_conditioned_ei_tm_targeted"
    / "forced_tm_edge_trends_H001_H060.csv"
)
UNICM_SHAPLEY_SUMMARY = (
    ROOT
    / "results"
    / "unicm_11mode_shapley_affine_n16384_independent"
    / "summary.json"
)
UNICM_CALIBRATION_SUMMARY = (
    ROOT
    / "results"
    / "unicm_synergy_regularized_forecast_normfit_1980_2003"
    / "summary.json"
)
UNICM_TARGET_XI_CALIBRATION_SUMMARY = (
    ROOT
    / "results"
    / "unicm_target_xi_shapley_prior_normfit_1980_2003_n16384"
    / "summary.json"
)
UNICM_INFORMATION_PRIOR_SUMMARY = (
    ROOT / "results" / "unicm_observational_prior_comparison_fit253_surd_allorders" / "summary.json"
)
# EI-Shapley remains in the complete comparison cache and appendix figure.
UNICM_SHAP_PRIOR_SUMMARY = ROOT / "results/unicm_frozen_shap_prior_fit253_pilot/summary.json"
UNICM_MAIN_PRIOR_METHODS = (
    "xi_shapley", "shap", "phi_r", "phi_wms", "phi_si", "causal_density", "surd",
)
UNICM_MAIN_METHOD_KEYS = (
    "frozen", "xi_shapley", "shap", "univariate", "uniform", *UNICM_MAIN_PRIOR_METHODS[2:],
)
# Estimation scope and data-source qualifiers are explained in earth.md appendix B.
UNICM_MAIN_METHOD_LABELS = {
    "frozen": "Frozen",
    "xi_shapley": r"$\Xi$-Shapley",
    "shap": "SHAP",
    "univariate": "Univariate",
    "uniform": "Uniform ridge",
    "phi_r": r"$\Phi^R$",
    "phi_wms": r"$\Phi^{\mathrm{WMS}}$",
    "phi_si": r"$\Phi_{\mathrm{SI}}$",
    "causal_density": "Causal density",
    "surd": "Gaussian SURD",
}
UNICM_SPT_UNIFORM_SUMMARY = (
    ROOT
    / "results"
    / "unicm_xi_hierarchy_uniform_n16384"
    / "summary.json"
)
UNICM_SPT_ORDER_MASS = ROOT / "results" / "unicm_spt_order_mass" / "order_mass.npz"
UNICM_HYPERGRAPH_MAP = ROOT / "docs" / "reports" / "assets" / "unicm_pair_hypergraph_leads.png"
UNICM_SPT_LEADS = (1, 8, 24)
UNICM_SPT_CHECKPOINT = 2
UNICM_NODE_MODE_KEY = (
    "Node–mode key:  0 ENSO  ·  1 NPMM  ·  2 SPMM  ·  3 IOB  ·  "
    "4 IOD  ·  5 SIOD  ·  6 TNA  ·  7 Niño 1+2  ·  8 Niño 3  ·  "
    "9 Niño 4  ·  10 WWV"
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
SOURCE_SHARE_MODE_ORDER = (
    "nino",
    "nino12",
    "nino3",
    "nino4",
    "NPMM",
    "SPMM",
    "IOB",
    "IOD",
    "SIOD",
    "TNA",
    "WWV",
)
NATS_PER_BIT = np.log(2.0)

BLUE = "#3F6F9F"
TEAL = "#2A9D8F"
ORANGE = "#D9822B"
VIOLET = "#8064A2"
ARCTIC_RED = "#B85C6F"
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


def load_unicm_checkpoint2_lead_trees() -> tuple[list[object], float]:
    payload = json.loads(UNICM_SPT_UNIFORM_SUMMARY.read_text(encoding="utf-8"))
    if not bool(payload["all_nine_conditions_valid_nonnegative_spt"]):
        raise ValueError("The fixed-parameter UniCM SPT audit did not pass all nine conditions.")
    trees = []
    for lead in UNICM_SPT_LEADS:
        row = next(
            record
            for record in payload["results"]
            if int(record["checkpoint"]) == UNICM_SPT_CHECKPOINT
            and int(record["lead"]) == lead
        )
        if row["status"] != "valid_nonnegative_spt":
            raise ValueError(
                f"Invalid fixed-parameter SPT for checkpoint {UNICM_SPT_CHECKPOINT}, lead {lead}."
            )
        trees.append(_tree_from_record(row["tree"]))
    return trees, float(payload["syn_tolerance_bits"])


def draw_unicm_hypergraph_maps(canvas) -> None:
    """Place the cached maps while rebuilding titles and panel labels consistently."""
    image = plt.imread(UNICM_HYPERGRAPH_MAP)
    crop_top = int(round(image.shape[0] * 0.05))
    crop_bottom = int(round(image.shape[0] * 0.76))
    panel_edges = np.linspace(0, image.shape[1], 4, dtype=int)
    axes = np.atleast_1d(canvas.subplots(1, 3))
    for axis, lead, left, right in zip(
        axes,
        UNICM_SPT_LEADS,
        panel_edges[:-1],
        panel_edges[1:],
        strict=True,
    ):
        axis.imshow(
            image[crop_top:crop_bottom, left:right],
            interpolation="lanczos",
            aspect="auto",
        )
        axis.axis("off")
        unit = "month" if lead == 1 else "months"
        axis.set_title(f"Lead = {lead} {unit}", fontsize=8.2, pad=2.5)
    for x, label in zip((0.002, 0.315, 0.628), "abc", strict=True):
        canvas.text(
            x,
            0.995,
            label,
            ha="left",
            va="top",
            fontsize=8.2,
            fontweight="bold",
            color="#111111",
        )
    canvas.text(
        0.5,
        0.005,
        UNICM_NODE_MODE_KEY,
        ha="center",
        va="bottom",
        fontsize=6.3,
        color=INK,
    )


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


def add_latitude_ticks_only(ax: plt.Axes, *, scale: float = 1.0) -> None:
    latitudes = (-60, -30, 0, 30, 60)
    labels = ("60°S", "30°S", "0°", "30°N", "60°N")
    ax.set_xticks([])
    ax.set_yticks(np.radians(latitudes), labels)
    ax.tick_params(
        axis="y",
        labelsize=4.1 * scale,
        pad=1.0,
        length=1.6 * scale,
        width=0.35 * scale,
    )


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
    *,
    scale: float = 1.0,
) -> None:
    draw_world(ax, land, coastlines)
    add_latitude_ticks_only(ax, scale=scale)
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
        s=5 * scale**2,
        color="#A7ADB5",
        edgecolors="none",
        alpha=0.28,
        zorder=3,
    )
    for subset, color, size, edge in (
        (targets, TEAL, 28 * scale**2, "#153D42"),
        (sources, BLUE, 48 * scale**2, "#173852"),
    ):
        selected = nodes[nodes["local"].isin(subset)]
        ax.scatter(
            np.radians(selected.lon),
            np.radians(selected.lat),
            s=size,
            color=color,
            edgecolors=edge,
            linewidths=0.45 * scale,
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
            s=(3.0 + 5.0 * strength) * scale**2,
            color=VIOLET,
            edgecolors="white",
            linewidths=0.2 * scale,
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
            fontsize=3.8 * scale,
            fontweight="bold",
            color="white",
            path_effects=[pe.withStroke(linewidth=0.9 * scale, foreground="#111111")],
            zorder=8,
        )
    ax.text(
        0.5,
        1.04,
        rf"$\ell={horizon}$ {'week' if int(horizon) == 1 else 'weeks'}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.8 * scale,
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
    focal_pair: tuple[int, int] = (0, 1),
) -> pd.DataFrame:
    node_lookup = nodes.set_index("local")
    rows: list[dict[str, float | int]] = []
    for horizon, ranking in rankings.items():
        for top_k in top_ks:
            top = ranking.head(top_k)
            focal = top[
                (top["source_a"] == int(focal_pair[0]))
                & (top["source_b"] == int(focal_pair[1]))
            ].copy()
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
    focal_pair: tuple[int, int] = (0, 1),
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
        focal_pair=focal_pair,
    )
    trends = pd.read_csv(trend_csv)
    arctic_edge = (0, 3, 37)
    arctic_label = "+".join(str(value) for value in arctic_edge[:2]) + f"->{arctic_edge[2]}"
    if arctic_label not in set(trends["edge_label_paper"].astype(str)):
        arctic_rows: list[dict[str, float | int | str]] = []
        for horizon in HORIZONS:
            ranking = rankings[horizon]
            match = ranking[
                (ranking["source_a"].astype(int) == arctic_edge[0])
                & (ranking["source_b"].astype(int) == arctic_edge[1])
                & (ranking["target"].astype(int) == arctic_edge[2])
            ]
            if len(match) != 1:
                raise RuntimeError(
                    f"Expected exactly one Arctic edge {arctic_label} at H={horizon}, "
                    f"received {len(match)}."
                )
            arctic_rows.append(
                {
                    "horizon": horizon,
                    "edge_label_paper": arctic_label,
                    "delta2_tm": float(match.iloc[0]["delta2_tm"]),
                }
            )
        trends = pd.concat([trends, pd.DataFrame(arctic_rows)], ignore_index=True)
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
        line_color = cutoff_colors[int(top_k)]
        ax_d.plot(
            positions,
            frame["valid_pair_count"],
            color=line_color,
            linewidth=1.45 if primary else 0.9,
            marker="o",
            markersize=2.2 if primary else 1.7,
            markeredgewidth=0,
            label=f"top-{int(top_k)}",
            zorder=3 if primary else 2,
        )
        for x, row, ha in (
            (positions[0], frame.iloc[0], "left"),
            (positions[-1], frame.iloc[-1], "right"),
        ):
            place_below = int(top_k) == 500 and ha == "left"
            ax_d.annotate(
                f"{row.valid_pair_count:.0f}",
                (x, float(row.valid_pair_count)),
                xytext=(3 if ha == "left" else -3, -5 if place_below else 3),
                textcoords="offset points",
                ha=ha,
                va="top" if place_below else "bottom",
                color=line_color,
                fontsize=5.5 if primary else 4.8,
                fontweight="bold" if primary else "normal",
                zorder=4,
            )
    ax_d.set_xticks(sparse_tick_positions, sparse_tick_horizons)
    ax_d.set_xlabel(r"Prediction lead, $\ell$ (weeks)")
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
    ax_e.set_xlabel(r"Prediction lead, $\ell$ (weeks)")
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
    ax_f.set_xlabel(r"Prediction lead, $\ell$ (weeks)")
    ax_f.set_ylabel("Maximum target span (km)", color=TEAL)
    ax_f.tick_params(axis="y", colors=TEAL)
    ax_f.set_ylim(0, 21000.0)
    ax_f.set_yticks((0, 5000, 10000, 15000, 20000))
    ax_f.grid(axis="y", color=LIGHT_GREY, linewidth=0.55)
    ax_f_right = ax_f.twinx()
    target_line = ax_f_right.plot(
        positions,
        primary_coverage["target_count"],
        color=VIOLET,
        linewidth=1.05,
        marker="s",
        markersize=2.0,
        label="distinct targets",
    )[0]
    ax_f_right.set_ylabel("Distinct targets", color=VIOLET)
    ax_f_right.tick_params(axis="y", colors=VIOLET)
    ax_f_right.spines["top"].set_visible(False)
    ax_f_right.spines["right"].set_linewidth(0.65)
    ax_f_right.spines["right"].set_color(VIOLET)
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

    trends = trends.copy()
    trends["delta2_tm"] = NATS_PER_BIT * trends["delta2_tm"].astype(float)
    ax_g = fig.add_subplot(grid[2, 2:6])
    colors = {
        "0+6->32": BLUE,
        "0+1->28": ORANGE,
        "0+1->50": TEAL,
        "0+1->46": VIOLET,
        arctic_label: ARCTIC_RED,
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
    ax_g.set_ylim(NATS_PER_BIT * -0.0004, NATS_PER_BIT * 0.0215)
    ax_g.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    ax_g.set_xlabel(r"Prediction lead, $\ell$ (weeks)")
    ax_g.set_ylabel(r"TM estimate of $Syn^{\mathrm{EID}}$ (nats)")
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
            f"No.{focal_pair[0]} + No.{focal_pair[1]} targets retained in the "
            "global top-200 at each horizon."
        ),
        "pair_0_1_target_coverage": coverage_records,
        "arctic_representative_edge": {
            "edge": arctic_label,
            "arctic_component": 3,
            "selection_rule": "Global top-ranked Arctic-related edge at H=1.",
            "curve": [
                {
                    "horizon": int(row.horizon),
                    "syn_eid_tm_bits": float(row.delta2_tm),
                }
                for row in trends[trends["edge_label_paper"] == arctic_label]
                .sort_values("horizon")
                .itertuples(index=False)
            ],
        },
        "outputs": [str(path) for path in outputs],
    }
    output_base.with_name(f"{output_base.name}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return outputs


def plot_unicm_rmse_display_comparison(output_base: Path) -> list[Path]:
    """Compare baseline-relative nRMSE with RMSE in cached model-index units.

    The latter removes evaluation target scaling, not the field normalization
    used upstream. It weights high-variance modes more heavily.
    """
    from scripts.run_unicm_information_prior_comparison import COLORS

    comparison = json.loads(UNICM_INFORMATION_PRIOR_SUMMARY.read_text())
    shap = json.loads(UNICM_SHAP_PRIOR_SUMMARY.read_text())
    entries = {**comparison["methods"], "shap": shap["methods"]["shap"]}
    labels = UNICM_MAIN_METHOD_LABELS
    keys = UNICM_MAIN_METHOD_KEYS
    nrmse = np.asarray([entries[k]["test_nrmse"] for k in keys])
    reductions = 100.0 * (1.0 - nrmse / entries["frozen"]["test_nrmse"])
    with np.load(UNICM_SHAP_PRIOR_SUMMARY.parent / "evaluation_arrays.npz") as cache:
        raw = np.asarray([
            np.sqrt(np.mean((cache[f"prediction_{k}"] - cache["test_target"]) ** 2, axis=0)).mean()
            for k in keys
        ])
        cached_nrmse = np.asarray([
            (np.sqrt(np.mean((cache[f"prediction_{k}"] - cache["test_target"]) ** 2, axis=0))
             / cache["target_scale"].reshape(11, 1)).mean()
            for k in keys
        ])
    if not np.allclose(cached_nrmse, nrmse, rtol=0, atol=1e-9):
        raise ValueError("Display comparison cache differs from the main comparison.")
    palette = {**COLORS, "shap": "#BE6B32"}
    y = np.arange(len(keys))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), sharey=True, constrained_layout=True)
    for ax, values, percent in zip(axes, (reductions, raw), (True, False), strict=True):
        span = float(np.ptp(values))
        ax.scatter(values, y, s=36, c=[palette[k] for k in keys], edgecolors="white", linewidths=0.5, zorder=3)
        for value, row in zip(values, y, strict=True):
            ax.text(value + 0.018 * span, row, f"{value:.2f}%" if percent else f"{value:.4f}",
                    fontsize=8, va="center", color=INK)
        ax.set_xlim(float(values.min()) - 0.12 * span, float(values.max()) + 0.26 * span)
        ax.set_ylim(-0.55, len(keys) - 0.45)
        ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.5)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax.set_xlabel("nRMSE reduction vs Frozen (%) · higher is better" if percent
                      else "RMSE in model-index units · lower is better", fontsize=8)
    axes[0].axvline(0, color="#686F78", linewidth=0.7, linestyle="--", zorder=1)
    axes[0].set_yticks(y, [labels[k] for k in keys], fontsize=8)
    add_panel_label(axes[0], "a", x=-0.48, y=1.04)
    add_panel_label(axes[1], "b", x=-0.06, y=1.04)
    output = output_base.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [output]


def plot_unicm_figure(
    output_base: Path, *, spt_order_cache: Path | None = None,
    spt_node_value_mode: str = "absolute",
    spt_preview_output: Path | None = None,
) -> list[Path]:
    trees, syn_tolerance = load_unicm_checkpoint2_lead_trees()
    shapley = json.loads(UNICM_SHAPLEY_SUMMARY.read_text(encoding="utf-8"))
    calibration = json.loads(UNICM_CALIBRATION_SUMMARY.read_text(encoding="utf-8"))
    xi_calibration = json.loads(
        UNICM_TARGET_XI_CALIBRATION_SUMMARY.read_text(encoding="utf-8")
    )
    from scripts.run_unicm_information_prior_comparison import COLORS
    methods = ("xi_shapley", "phi_r", "ei_shapley", "phi_wms", "phi_si", "causal_density", "surd")
    prior_comparison = json.loads(UNICM_INFORMATION_PRIOR_SUMMARY.read_text(encoding="utf-8"))
    if prior_comparison["parameter_mode"] != "fixed_xi":
        raise ValueError("Main panel j requires fixed Xi-selected parameters.")
    if prior_comparison["shared_hyperparameters"] != {"alpha": 30000.0, "gamma": 3.0}:
        raise ValueError("Main panel j must use the original Xi-selected alpha and gamma.")
    if prior_comparison["included_methods"] != list(methods):
        raise ValueError("Main panel j information-prior scope differs from the active comparison.")
    if prior_comparison["samples"] != {"fit": 253, "validation": 36, "test": 96}:
        raise ValueError("Main panel j chronological sample counts differ from the declared experiment.")
    for key in methods:
        expected_distribution = ("independent_uniform_intervention_history_and_frozen_output"
                                 if key in ("xi_shapley", "ei_shapley") else
                                 "fit_only_natural_history_and_actual_future")
        if prior_comparison["prior_distributions"][key] != expected_distribution:
            raise ValueError(f"Main panel j prior distribution mismatch: {key}.")
    if prior_comparison["prior_data_audit"]["uses_model_predictions_for_observational_prior"]:
        raise ValueError("Observational priors must use actual future targets.")
    reference_scores = {
        **{key: calibration["test_metrics"][key]["mean_cell_nrmse"]
           for key in ("frozen", "univariate")},
        "xi_shapley": xi_calibration["test_nrmse"]["target_xi_shapley_prior"],
    }
    for key, score in reference_scores.items():
        if not np.isclose(prior_comparison["methods"][key]["test_nrmse"], score, rtol=0, atol=1e-9):
            raise ValueError(f"Main panel j reference score mismatch: {key}.")
    shap_prior = json.loads(UNICM_SHAP_PRIOR_SUMMARY.read_text(encoding="utf-8"))
    if (shap_prior["parameter_mode"] != "fixed_xi"
            or shap_prior["shared_hyperparameters"] != prior_comparison["shared_hyperparameters"]
            or shap_prior["samples"] != prior_comparison["samples"]
            or not shap_prior["identity"]["fit_only_prior"]):
        raise ValueError("Main SHAP comparison must preserve fixed parameters and fit-only priors.")
    for key, entry in prior_comparison["methods"].items():
        if not np.isclose(shap_prior["methods"][key]["test_nrmse"], entry["test_nrmse"], rtol=0, atol=1e-9):
            raise ValueError(f"Main SHAP reference score mismatch: {key}.")
    fig = plt.figure(figsize=(12.8, 12.0), layout="constrained")
    grid = fig.add_gridspec(
        4,
        6,
        height_ratios=[1.65, 2.95, 2.25, 2.4],
        hspace=0.045,
    )

    map_canvas = fig.add_subfigure(grid[0, :])
    draw_unicm_hypergraph_maps(map_canvas)

    tree_canvas = fig.add_subfigure(grid[1, :])
    with mpl.rc_context():
        render_trees(
            trees,
            [UNICM_SPT_CHECKPOINT] * len(trees),
            output_base.with_suffix(".png"),
            lead=8,
            split_objective="raw_residual",
            dpi=600,
            syn_tolerance=syn_tolerance,
            canvas=tree_canvas,
            show_colorbar=True,
            compact_core_annotation=True,
            show_checkpoint=False,
            show_tree_metrics=False,
            show_root_info=spt_node_value_mode == "root_share",
            core_highlights=(False, True, False),
            node_label_fontsize=7.6,
            terminal_label_fontsize=8.3,
            core_label_fontsize=9.2,
            root_info_fontsize=7.8,
            compact_node_labels=True,
            display_scale=NATS_PER_BIT,
            display_unit="nats",
            node_value_mode=spt_node_value_mode,
        )
    for axis, label in zip(tree_canvas.axes[:3], "def", strict=True):
        add_panel_label(axis, label, x=-0.06, y=1.02)
    tree_canvas.text(
        0.998,
        0.995,
        "checkpoint 2  |  n = 16,384",
        ha="right",
        va="top",
        fontsize=7.0,
        color=MID_GREY,
    )

    middle_grid = grid[2, :].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.10)
    bottom_grid = grid[3, :].subgridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.10)
    order_grid = middle_grid[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.035], wspace=0.06)
    share_grid = middle_grid[0, 1].subgridspec(1, 2, width_ratios=[1.0, 0.035], wspace=0.06)
    ax_d = fig.add_subplot(order_grid[0, 0])
    ax_d_colorbar = fig.add_subplot(order_grid[0, 1])
    from scripts.compute_unicm_spt_order_mass import load_spt_order_mass
    selected_order_cache = spt_order_cache if spt_order_cache is not None else UNICM_SPT_ORDER_MASS
    order_values = NATS_PER_BIT * load_spt_order_mass(selected_order_cache)["mean_mass_bits"]
    image = ax_d.imshow(
        order_values, aspect="auto", interpolation="nearest", cmap="viridis",
        norm=mpl.colors.Normalize(vmin=0.0, vmax=float(order_values.max())),
    )
    ax_d.set_yticks(np.arange(10), [str(k) for k in range(2, 12)])
    heatmap_tick_indices = np.asarray((0, 3, 7, 11, 15, 19, 23))
    ax_d.set_xticks(heatmap_tick_indices, [str(k + 1) for k in heatmap_tick_indices])
    ax_d.set_xlabel(r"Prediction lead, $\ell$ (months)")
    ax_d.set_ylabel("SPT node order")
    colorbar = fig.colorbar(image, cax=ax_d_colorbar)
    colorbar.set_label("SPT order mass (nats)")
    ax_d.text(1.0, 1.02, "Tree-wise order sum | 3-checkpoint mean | n = 16,384", transform=ax_d.transAxes,
              ha="right", va="bottom", fontsize=5.4, color="#444444")
    add_panel_label(ax_d, "g", x=-0.13, y=1.02)

    ax_e = fig.add_subplot(share_grid[0, 0])
    ax_e_colorbar = fig.add_subplot(share_grid[0, 1])
    shapley_leads = np.asarray(
        [int(record["lead"]) for record in shapley["lead_summary"]]
    )
    source_share_values = np.asarray(
        [
            [
                float(record["shapley_percent_mean"][mode])
                for record in shapley["lead_summary"]
            ]
            for mode in SOURCE_SHARE_MODE_ORDER
        ]
    )
    source_image = ax_e.imshow(
        source_share_values,
        aspect="auto",
        interpolation="nearest",
        cmap="YlGnBu",
        extent=(0.5, 24.5, len(SOURCE_SHARE_MODE_ORDER) - 0.5, -0.5),
    )
    source_leaders = np.argmax(source_share_values, axis=0)
    ax_e.scatter(
        shapley_leads,
        source_leaders,
        marker="o",
        s=7,
        facecolor="white",
        edgecolor=INK,
        linewidth=0.35,
    )
    ax_e.set_xticks((1, 4, 8, 12, 16, 20, 24))
    ax_e.set_yticks(
        np.arange(len(SOURCE_SHARE_MODE_ORDER)),
        ["ENSO" if mode == "nino" else mode for mode in SOURCE_SHARE_MODE_ORDER],
    )
    ax_e.set_xlabel(r"Prediction lead, $\ell$ (months)")
    ax_e.set_ylabel("Source mode")
    source_colorbar = fig.colorbar(source_image, cax=ax_e_colorbar)
    source_colorbar.set_label("Mean Shapley share (%)")
    ax_e.text(
        1.0,
        1.02,
        "Independent-source affine TM | 3 checkpoints | n = 16,384",
        transform=ax_e.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.4,
        color="#444444",
    )
    add_panel_label(ax_e, "h", x=-0.13, y=1.02)

    metrics = calibration["test_metrics"]
    method_keys = UNICM_MAIN_METHOD_KEYS
    comparison_entries = {**prior_comparison["methods"], "shap": shap_prior["methods"]["shap"]}
    for key in ("uniform", *UNICM_MAIN_PRIOR_METHODS):
        entry = comparison_entries[key]
        if entry["alpha"] != 30000.0 or entry["gamma"] != 3.0 or len(entry["validation_scores"]) != 1:
            raise ValueError(f"Main panel j has searched or mismatched parameters: {key}.")
    current_labels = UNICM_MAIN_METHOD_LABELS
    method_labels = [current_labels[key] for key in method_keys]
    method_nrmse = np.asarray([comparison_entries[key]["test_nrmse"] for key in method_keys])
    # Re-express the same aggregate scores against Frozen; preserve ranking and
    # equal target/lead weighting rather than change the evaluation metric.
    method_values = 100.0 * (1.0 - method_nrmse / comparison_entries["frozen"]["test_nrmse"])
    xi_color = COLORS["xi_shapley"]
    method_palette = {**COLORS, "shap": "#BE6B32"}
    method_colors = [method_palette[key] for key in method_keys]

    ax_f = fig.add_subplot(bottom_grid[0, 1])
    method_y = np.arange(len(method_keys))[::-1]
    ax_f.scatter(
        method_values,
        method_y,
        s=31,
        color=method_colors,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    for value, y_value in zip(method_values, method_y):
        ax_f.text(
            value + 0.18,
            y_value,
            f"{value:.2f}%",
            va="center",
            ha="left",
            fontsize=6.0,
            color=INK,
        )
    ax_f.set_yticks(method_y, method_labels)
    ax_f.tick_params(axis="y", labelsize=6.5)
    ax_f.set_xlabel("Test nRMSE reduction vs Frozen (%) · higher is better")
    ax_f.axvline(0.0, color="#686F78", linewidth=0.7, linestyle="--", zorder=1)
    ax_f.text(1.0, 1.02, r"Shared $\Xi$ settings: $\alpha=30000$, $\gamma=3$", transform=ax_f.transAxes,
              ha="right", va="bottom", fontsize=5.8, color=MID_GREY)
    score_span = float(method_values.max() - method_values.min())
    score_pad = max(1.0, 0.15 * score_span)
    ax_f.set_xlim(
        float(method_values.min()) - score_pad,
        float(method_values.max()) + score_pad,
    )
    ax_f.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax_f.set_ylim(-0.55, len(method_keys) - 0.45)
    ax_f.grid(axis="x", color=LIGHT_GREY, linewidth=0.5)
    add_panel_label(ax_f, "j", x=-0.38, y=1.04)

    ax_g = fig.add_subplot(bottom_grid[0, 0])
    frozen_score = float(metrics["frozen"]["mean_cell_nrmse"])
    xi_score = float(xi_calibration["test_nrmse"]["target_xi_shapley_prior"])
    xi_reduction_pct = (frozen_score - xi_score) / frozen_score * 100.0
    random_scores = np.asarray(
        xi_calibration["shuffled_xi_control"]["scores"],
        dtype=float,
    )
    random_p = float(
        xi_calibration["shuffled_xi_control"]["fraction_null_at_least_as_good"]
    )
    null_reductions_pct = (frozen_score - random_scores) / frozen_score * 100.0
    visible_null = null_reductions_pct
    rng = np.random.default_rng(20260728)
    null_y = 1.0 + rng.uniform(-0.18, 0.18, size=len(visible_null))
    ax_g.axvline(0, color="#686F78", linewidth=0.7, linestyle="--", zorder=1)
    ax_g.axhline(1.0, color=LIGHT_GREY, linewidth=0.5, zorder=0)
    ax_g.axhline(0.0, color=LIGHT_GREY, linewidth=0.5, zorder=0)
    ax_g.scatter(
        visible_null,
        null_y,
        s=14,
        color="#B4BFCC",
        edgecolor="white",
        linewidth=0.25,
        alpha=0.62,
        zorder=2,
    )
    ax_g.scatter(
        [xi_reduction_pct],
        [0.0],
        s=32,
        marker="o",
        color=xi_color,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax_g.text(
        0.98,
        0.92,
        rf"$P={random_p:.3f}$",
        transform=ax_g.transAxes,
        ha="right",
        va="top",
        fontsize=5.6,
        color=INK,
    )
    ax_g.text(
        xi_reduction_pct - 0.10,
        0.13,
        f"{xi_reduction_pct:.2f}",
        ha="right",
        va="bottom",
        fontsize=5.5,
        color=xi_color,
    )
    ax_g.set_yticks(
        (1.0, 0.0),
        ("Shuffled Xi priors", "Xi-Shapley prior"),
    )
    ax_g.set_xlim(0.0, max(11.0, xi_reduction_pct + 0.5))
    ax_g.set_ylim(-0.42, 1.42)
    ax_g.grid(axis="x", color=LIGHT_GREY, linewidth=0.5)
    ax_g.set_xlabel("Test RMSE reduction vs frozen ensemble (%)")
    add_panel_label(ax_g, "i", x=-0.18, y=1.04)
    output = output_base.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    if spt_preview_output is not None:
        spt_preview_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(spt_preview_output, dpi=600,
                    bbox_inches=tree_canvas.bbox.transformed(fig.dpi_scale_trans.inverted()))
    plt.close(fig)
    return [output] if spt_preview_output is None else [output, spt_preview_output]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIG_DIR,
        help="Directory for the publication figure bundle.",
    )
    parser.add_argument("--runge-result-dir", type=Path, default=RUNGE_RESULT_DIR)
    parser.add_argument("--runge-component-maps", type=Path, default=DEFAULT_COMPONENT_MAPS)
    parser.add_argument("--runge-trend-csv", type=Path, default=RUNGE_TREND_CSV)
    parser.add_argument("--runge-focal-pair", default="0,1")
    parser.add_argument("--skip-unicm", action="store_true")
    parser.add_argument("--skip-runge", action="store_true")
    parser.add_argument("--spt-node-values", choices=("absolute", "root_share"),
                        default="absolute", help="SPT node values and color: local Syn or percent of root Xi.")
    args = parser.parse_args()
    configure_matplotlib()
    focal_pair = tuple(int(value) for value in str(args.runge_focal_pair).split(","))
    if len(focal_pair) != 2:
        raise ValueError("--runge-focal-pair must contain two comma-separated indices.")
    runge_outputs = [] if args.skip_runge else plot_runge_figure(
        Path(args.output_dir) / "earth_slp_hyperedge_dynamics",
        result_dir=args.runge_result_dir,
        component_maps=args.runge_component_maps,
        trend_csv=args.runge_trend_csv,
        focal_pair=(min(focal_pair), max(focal_pair)),
    )
    unicm_outputs = [] if args.skip_unicm else plot_unicm_figure(
        Path(args.output_dir) / "earth_unicm_hierarchy_overview",
        spt_node_value_mode=args.spt_node_values,
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
