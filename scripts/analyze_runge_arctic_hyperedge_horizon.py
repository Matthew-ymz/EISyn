#!/usr/bin/env python3
"""Analyze Arctic-related Runge SLP binary hyperedges across forecast horizons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_runge_exhaustive_tm_maps import DEFAULT_COMPONENT_MAPS, load_nodes
from scripts.plot_runge_gateway_mediator_map import (
    COASTLINE_URL,
    extract_lines,
    load_geojson,
    local_to_paper,
    sign_normalized_map,
    split_dateline,
)
from scripts.plot_runge_source_pair_condensation import load_rankings
from scripts.run_runge_exhaustive_degree3_tm import DEFAULT_RESULT_DIR


DEFAULT_OUTPUT = ROOT / "fig" / "earth_slp_arctic_hyperedge_horizon"
HORIZONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60)
TOP_K_VALUES = (50, 100, 200, 500)
PRIMARY_TOP_K = 200
ARCTIC_CIRCLE_LATITUDE = 66.5
SYN_TOLERANCE_BITS = 1e-10

BLUE = "#416F96"
TEAL = "#4C8C87"
ORANGE = "#D9822B"
VIOLET = "#7B6A9A"
INK = "#1D2733"
GRID = "#E6EAF0"
ROLE_COLORS = {"Arctic source": BLUE, "Arctic target": ORANGE}


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


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.12) -> None:
    ax.text(
        x,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color="#111111",
        clip_on=False,
    )


def identify_arctic_components(nodes: pd.DataFrame) -> pd.DataFrame:
    arctic = nodes[nodes["lat"] >= ARCTIC_CIRCLE_LATITUDE].copy()
    if arctic.empty:
        raise RuntimeError("No component centre lies north of the operational Arctic-circle threshold.")
    return arctic.sort_values("lat", ascending=False, ignore_index=True)


def arctic_masks(frame: pd.DataFrame, arctic_nodes: set[int]) -> tuple[pd.Series, pd.Series]:
    source = frame["source_a"].isin(arctic_nodes) | frame["source_b"].isin(arctic_nodes)
    target = frame["target"].isin(arctic_nodes)
    if bool((source & target).any()):
        raise RuntimeError("A candidate cannot use the same Arctic component as source and target.")
    return source, target


def audited_positive(values: pd.Series) -> tuple[pd.Series, int, float]:
    numeric = values.astype(float)
    minimum = float(numeric.min()) if len(numeric) else float("nan")
    violations = numeric < -SYN_TOLERANCE_BITS
    if bool(violations.any()):
        raise RuntimeError(
            "Selected synergy values violate nonnegativity: "
            f"minimum={minimum:.12g} bits, tolerance={SYN_TOLERANCE_BITS:.12g} bits, "
            f"count={int(violations.sum())}."
        )
    numerical_zero = (numeric < 0.0) & ~violations
    adjusted = numeric.mask(numerical_zero, 0.0)
    return adjusted, int(numerical_zero.sum()), minimum


def build_metrics(
    rankings: dict[int, pd.DataFrame],
    arctic_nodes: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    cutoff_rows: list[dict[str, float | int]] = []
    leader_rows: list[dict[str, float | int | str]] = []
    numerical_zero_count = 0
    selected_minimum = float("inf")

    for horizon in HORIZONS:
        ranking = rankings[horizon]
        audited_top, zero_count, minimum = audited_positive(
            ranking.head(max(TOP_K_VALUES))["delta2_tm"]
        )
        numerical_zero_count += zero_count
        selected_minimum = min(selected_minimum, minimum)
        source_all, target_all = arctic_masks(ranking, arctic_nodes)
        related_all = source_all | target_all
        related = ranking.loc[related_all]
        if related.empty:
            raise RuntimeError(f"No Arctic-related candidates are available at H={horizon}.")
        leader = related.iloc[0]
        leader_role = "Arctic source" if bool(source_all.loc[leader.name]) else "Arctic target"
        leader_rows.append(
            {
                "horizon": horizon,
                "tm_rank": int(leader["tm_rank"]),
                "delta2_tm": float(leader["delta2_tm"]),
                "role": leader_role,
                "source_a": int(leader["source_a"]),
                "source_b": int(leader["source_b"]),
                "target": int(leader["target"]),
                "label": (
                    f"No.{local_to_paper(int(leader['source_a']))} + "
                    f"No.{local_to_paper(int(leader['source_b']))} → "
                    f"No.{local_to_paper(int(leader['target']))}"
                ),
            }
        )

        for top_k in TOP_K_VALUES:
            top = ranking.head(top_k).copy()
            source, target = arctic_masks(top, arctic_nodes)
            positive = audited_top.iloc[:top_k]
            total_mass = float(positive.sum())
            if total_mass <= 0.0:
                raise RuntimeError(f"Top-{top_k} has no positive synergy mass at H={horizon}.")
            cutoff_rows.append(
                {
                    "horizon": horizon,
                    "top_k": top_k,
                    "related_count": int((source | target).sum()),
                    "source_count": int(source.sum()),
                    "target_count": int(target.sum()),
                    "related_share": float(positive[source | target].sum() / total_mass),
                    "source_share": float(positive[source].sum() / total_mass),
                    "target_share": float(positive[target].sum() / total_mass),
                }
            )

    audit = {
        "syn_nonnegative_tolerance_bits": SYN_TOLERANCE_BITS,
        "selected_minimum_delta2_tm_bits": selected_minimum,
        "numerical_zero_affected_count": numerical_zero_count,
        "significant_nonnegativity_violation_count": 0,
    }
    return pd.DataFrame(cutoff_rows), pd.DataFrame(leader_rows), audit


def draw_component_panel(
    ax: plt.Axes,
    component_maps: np.ndarray,
    nodes: pd.DataFrame,
    arctic_local: int,
) -> None:
    values = sign_normalized_map(component_maps[:, :, arctic_local])
    latitudes = np.linspace(-90.0, 90.0, values.shape[0])
    longitudes = ((np.linspace(0.0, 360.0, values.shape[1], endpoint=False) + 180.0) % 360.0) - 180.0
    order = np.argsort(longitudes)
    longitudes = longitudes[order]
    values = values[:, order]
    limit = float(np.nanpercentile(np.abs(values[latitudes >= 35.0]), 99.2))
    image = ax.pcolormesh(
        longitudes,
        latitudes,
        values,
        shading="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        rasterized=True,
    )
    for line in extract_lines(load_geojson(COASTLINE_URL)):
        for segment in split_dateline(line):
            ax.plot(
                [point[0] for point in segment],
                [point[1] for point in segment],
                color="#555E68",
                linewidth=0.32,
                alpha=0.72,
            )
    centre = nodes.set_index("local").loc[arctic_local]
    ax.axhline(
        ARCTIC_CIRCLE_LATITUDE,
        color=INK,
        linestyle="--",
        linewidth=0.7,
        label="Arctic Circle",
    )
    ax.scatter(
        [float(centre["lon"])],
        [float(centre["lat"])],
        s=26,
        color=INK,
        edgecolors="white",
        linewidths=0.6,
        zorder=4,
    )
    ax.text(
        float(centre["lon"]) + 7.0,
        float(centre["lat"]) + 1.0,
        f"No.{int(centre['paper'])}",
        color=INK,
        fontsize=6.0,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
    )
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(35.0, 90.0)
    ax.set_xticks((-120, 0, 120), ("120°W", "0°", "120°E"))
    ax.set_yticks((40, 60, 80), ("40°N", "60°N", "80°N"))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=5.3,
        handlelength=1.6,
    )
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.04, pad=0.025)
    colorbar.set_label("Signed loading", fontsize=5.8)
    colorbar.ax.tick_params(labelsize=5.0, width=0.35, length=2.0)


def plot_figure(
    component_maps_path: Path,
    nodes: pd.DataFrame,
    arctic: pd.DataFrame,
    metrics: pd.DataFrame,
    leaders: pd.DataFrame,
    output_base: Path,
) -> list[Path]:
    component_maps = np.load(component_maps_path, allow_pickle=False)["component_maps"]
    arctic_local = int(arctic.iloc[0]["local"])
    fig = plt.figure(figsize=(7.2, 5.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(0.94, 1.16), hspace=0.16, wspace=0.12)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    draw_component_panel(ax_a, component_maps, nodes, arctic_local)
    add_panel_label(ax_a, "a", x=-0.15)

    cutoff_colors = {50: "#A8B3BE", 100: "#7E96AA", 200: BLUE, 500: "#294A65"}
    for top_k, frame in metrics.groupby("top_k", sort=True):
        frame = frame.sort_values("horizon")
        primary = int(top_k) == PRIMARY_TOP_K
        ax_b.plot(
            frame["horizon"],
            100.0 * frame["related_share"],
            color=cutoff_colors[int(top_k)],
            linewidth=1.55 if primary else 0.9,
            marker="o" if primary else None,
            markersize=2.7,
            label=f"top-{int(top_k)}",
            zorder=3 if primary else 2,
        )
    ax_b.set_xlim(0.5, 61.0)
    ax_b.set_ylim(bottom=0.0)
    ax_b.set_xticks((1, 5, 10, 20, 40, 60))
    ax_b.set_xlabel("Forecast horizon, $H$")
    ax_b.set_ylabel("Arctic-related synergy-mass share (%)")
    ax_b.grid(axis="y", color=GRID, linewidth=0.55)
    ax_b.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        fontsize=5.1,
        handlelength=1.35,
        columnspacing=0.75,
    )
    add_panel_label(ax_b, "b")

    primary = metrics[metrics["top_k"] == PRIMARY_TOP_K].sort_values("horizon")
    for role, column in (("Arctic source", "source_share"), ("Arctic target", "target_share")):
        ax_c.plot(
            primary["horizon"],
            100.0 * primary[column],
            color=ROLE_COLORS[role],
            linewidth=1.45,
            marker="o" if role == "Arctic source" else "s",
            markersize=2.5,
            label=role,
        )
    ax_c.axvspan(50.0, 60.0, color="#EEF1F4", alpha=0.75, zorder=0)
    ax_c.annotate(
        "source share reaches zero",
        xy=(60.0, 100.0 * float(primary.iloc[-1]["source_share"])),
        xytext=(30.0, 5.7),
        fontsize=5.2,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": "#7D858D", "linewidth": 0.55},
    )
    ax_c.set_xlim(0.5, 61.0)
    ax_c.set_ylim(bottom=0.0)
    ax_c.set_xticks((1, 5, 10, 20, 40, 60))
    ax_c.set_xlabel("Forecast horizon, $H$")
    ax_c.set_ylabel("Role-resolved top-200 mass share (%)")
    ax_c.grid(axis="y", color=GRID, linewidth=0.55)
    ax_c.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        fontsize=5.2,
        handlelength=1.4,
        columnspacing=0.9,
    )
    add_panel_label(ax_c, "c", x=-0.15)

    strength = leaders["delta2_tm"].to_numpy(dtype=float)
    min_size, max_size = 18.0, 54.0
    strength_scale = (strength - strength.min()) / max(float(np.ptp(strength)), 1e-12)
    marker_sizes = min_size + (max_size - min_size) * strength_scale
    for role, frame in leaders.groupby("role", sort=False):
        indices = frame.index.to_numpy(dtype=int)
        ax_d.scatter(
            frame["horizon"],
            frame["tm_rank"],
            s=marker_sizes[indices],
            color=ROLE_COLORS[str(role)],
            edgecolors="white",
            linewidths=0.55,
            label=str(role),
            zorder=3,
        )
    ax_d.plot(leaders["horizon"], leaders["tm_rank"], color="#9098A1", linewidth=0.7, zorder=1)
    label_offsets = {
        1: (5, 8),
        7: (5, -13),
        15: (5, 8),
        60: (-5, 9),
    }
    for horizon, offset in label_offsets.items():
        row = leaders[leaders["horizon"] == horizon].iloc[0]
        ax_d.annotate(
            str(row["label"]).replace("No.", ""),
            (float(row["horizon"]), float(row["tm_rank"])),
            xytext=offset,
            textcoords="offset points",
            ha="right" if horizon == 60 else "left",
            va="bottom" if offset[1] >= 0 else "top",
            fontsize=4.9,
            color=ROLE_COLORS[str(row["role"])],
        )
    ax_d.set_xlim(0.5, 61.0)
    ax_d.set_ylim(72.0, 0.0)
    ax_d.set_xticks((1, 5, 10, 20, 40, 60))
    ax_d.set_yticks((1, 20, 40, 60))
    ax_d.set_xlabel("Forecast horizon, $H$")
    ax_d.set_ylabel("Best global rank (lower is stronger)")
    ax_d.grid(axis="y", color=GRID, linewidth=0.55)
    role_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.2,
            markerfacecolor=ROLE_COLORS[role],
            markeredgecolor="white",
            label=role,
        )
        for role in ("Arctic source", "Arctic target")
    ]
    ax_d.legend(
        handles=role_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        fontsize=5.2,
        columnspacing=0.9,
    )
    add_panel_label(ax_d, "d")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_base.with_suffix(".png"),
        output_base.with_suffix(".svg"),
        output_base.with_suffix(".pdf"),
    ]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--component-maps", type=Path, default=DEFAULT_COMPONENT_MAPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    configure_matplotlib()
    nodes = load_nodes(args.component_maps)
    arctic = identify_arctic_components(nodes)
    rankings = load_rankings(args.result_dir, list(HORIZONS))
    metrics, leaders, audit = build_metrics(rankings, set(arctic["local"].astype(int)))
    outputs = plot_figure(args.component_maps, nodes, arctic, metrics, leaders, args.output)

    primary = metrics[metrics["top_k"] == PRIMARY_TOP_K].set_index("horizon")
    summary = {
        "operational_arctic_definition": (
            f"Component centre latitude >= {ARCTIC_CIRCLE_LATITUDE:.1f} degrees north."
        ),
        "arctic_components": arctic.to_dict(orient="records"),
        "horizons": list(HORIZONS),
        "candidate_count_per_horizon": int(len(rankings[HORIZONS[0]])),
        "primary_top_k": PRIMARY_TOP_K,
        "top_k_metrics": metrics.to_dict(orient="records"),
        "strongest_arctic_hyperedge_by_horizon": leaders.to_dict(orient="records"),
        "headline": {
            "top200_related_share_h1_percent": 100.0 * float(primary.loc[1, "related_share"]),
            "top200_related_share_h15_percent": 100.0 * float(primary.loc[15, "related_share"]),
            "top200_related_share_h60_percent": 100.0 * float(primary.loc[60, "related_share"]),
            "top200_source_share_h60_percent": 100.0 * float(primary.loc[60, "source_share"]),
            "top200_target_share_h60_percent": 100.0 * float(primary.loc[60, "target_share"]),
        },
        "nonnegativity_audit": audit,
        "outputs": [str(path) for path in outputs],
    }
    args.output.with_name(f"{args.output.name}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["headline"], indent=2))


if __name__ == "__main__":
    main()
