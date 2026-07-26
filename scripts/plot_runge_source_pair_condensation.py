#!/usr/bin/env python3
"""Plot the scale-dependent condensation of Runge SLP source pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plot_runge_gateway_mediator_map import local_to_paper
from scripts.run_runge_exhaustive_degree3_tm import (
    DEFAULT_RESULT_DIR,
    load_valid_ranking,
    parse_horizons,
)


DEFAULT_OUTPUT = ROOT / "fig/earth_slp_source_pair_condensation"
DEFAULT_HORIZONS = "1-10,15,20,30,40,50,60"
DEFAULT_TOP_K = 200
ROBUSTNESS_K = (50, 100, 200, 500)

INK = "#1D2733"
BLUE = "#3D6F97"
ORANGE = "#D97732"
LIGHT_ORANGE = "#F4DDCC"
GRID = "#E8EBEF"
PAIR_COLORS = (
    ORANGE,
    "#426B8A",
    "#6F8EA5",
    "#8EA9B8",
    "#6D8F8A",
    "#A08BA6",
    "#B59A77",
    "#87919E",
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


def pair_label(source_a: int, source_b: int) -> str:
    return f"No.{local_to_paper(source_a)} + No.{local_to_paper(source_b)}"


def load_rankings(result_dir: Path, horizons: list[int]) -> dict[int, pd.DataFrame]:
    rankings: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        horizon_dir = result_dir / f"H{horizon:03d}"
        summary = json.loads((horizon_dir / "summary.json").read_text(encoding="utf-8"))
        ranking = load_valid_ranking(
            horizon_dir / "full_ranking.npz",
            expected_metadata=summary["ranking_metadata"],
        )
        if ranking is None:
            raise RuntimeError(f"H={horizon} full ranking failed integrity validation.")
        rankings[horizon] = ranking
    return rankings


def aggregate_pair_weights(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    top = frame.head(top_k).copy()
    top["weight"] = top["delta2_tm"].clip(lower=0.0)
    return (
        top.groupby(["source_a", "source_b"], as_index=False)
        .agg(weight=("weight", "sum"), target_count=("target", "nunique"))
        .sort_values("weight", ascending=False, ignore_index=True)
    )


def build_metrics(
    rankings: dict[int, pd.DataFrame],
    horizons: list[int],
    *,
    top_k: int,
    robustness_k: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    effective_rows: list[dict[str, float | int]] = []
    pair_rows: list[dict[str, float | int | str]] = []
    focal_pair = (0, 1)

    for horizon in horizons:
        ranking = rankings[horizon]
        for cutoff in robustness_k:
            pair_weights = aggregate_pair_weights(ranking, cutoff)
            effective_rows.append(
                {
                    "horizon": horizon,
                    "top_k": cutoff,
                    "valid_pair_count": int((pair_weights["weight"] > 0).sum()),
                }
            )

        pair_weights = aggregate_pair_weights(ranking, top_k)
        total_weight = float(pair_weights["weight"].sum())
        for row in pair_weights.itertuples(index=False):
            pair_rows.append(
                {
                    "horizon": horizon,
                    "source_a": int(row.source_a),
                    "source_b": int(row.source_b),
                    "pair": pair_label(int(row.source_a), int(row.source_b)),
                    "weight": float(row.weight),
                    "share": float(row.weight / total_weight) if total_weight > 0 else 0.0,
                    "target_count": int(row.target_count),
                }
            )

    effective = pd.DataFrame(effective_rows)
    pairs = pd.DataFrame(pair_rows)
    integrated = (
        pairs.groupby("pair", as_index=False)["share"]
        .sum()
        .sort_values("share", ascending=False, ignore_index=True)
    )
    focal_label = pair_label(*focal_pair)
    selected = [focal_label]
    selected.extend(
        pair
        for pair in integrated["pair"].tolist()
        if pair != focal_label
    )
    return effective, pairs, selected[:8]


def plot_figure(
    effective: pd.DataFrame,
    pairs: pd.DataFrame,
    selected_pairs: list[str],
    horizons: list[int],
    *,
    top_k: int,
    output_base: Path,
) -> list[Path]:
    positions = np.arange(len(horizons), dtype=float)
    position_lookup = {horizon: index for index, horizon in enumerate(horizons)}
    focal_label = pair_label(0, 1)

    fig = plt.figure(figsize=(7.2, 4.8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(0.88, 1.18), hspace=0.22)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    k_colors = {
        50: "#AAB4BF",
        100: "#8297AA",
        200: BLUE,
        500: "#294A65",
    }
    for cutoff, frame in effective.groupby("top_k", sort=True):
        frame = frame.sort_values("horizon")
        is_primary = int(cutoff) == top_k
        ax_a.plot(
            positions,
            frame["valid_pair_count"],
            color=k_colors.get(int(cutoff), "#777777"),
            linewidth=1.55 if is_primary else 0.95,
            marker="o" if is_primary else None,
            markersize=2.5,
            label=f"top-{int(cutoff)}",
            zorder=3 if is_primary else 2,
        )
    primary = effective[effective["top_k"] == top_k].sort_values("horizon")
    first = primary.iloc[0]
    last = primary.iloc[-1]
    ax_a.annotate(
        f"{first['valid_pair_count']:.0f}",
        (positions[0], float(first["valid_pair_count"])),
        xytext=(4, 3),
        textcoords="offset points",
        color=BLUE,
        fontsize=5.8,
        fontweight="bold",
    )
    ax_a.annotate(
        f"{last['valid_pair_count']:.0f}",
        (positions[-1], float(last["valid_pair_count"])),
        xytext=(-3, 4),
        textcoords="offset points",
        ha="right",
        color=BLUE,
        fontsize=5.8,
        fontweight="bold",
    )
    ax_a.set_ylabel("Source pairs retained in top-$K$")
    ax_a.set_xlabel("Evaluated forecast horizon, $H$")
    ax_a.set_xticks(positions, horizons, rotation=45)
    ax_a.grid(axis="y", color=GRID, linewidth=0.55)
    ax_a.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        fontsize=5.2,
        handlelength=1.4,
        columnspacing=0.8,
    )
    add_panel_label(ax_a, "a")

    focal = (
        pairs[pairs["pair"] == focal_label]
        .set_index("horizon")
        .reindex(horizons)
        .fillna({"share": 0.0, "target_count": 0})
    )
    share = 100.0 * focal["share"].to_numpy(dtype=float)
    target_count = focal["target_count"].to_numpy(dtype=float)
    ax_b.fill_between(positions, share, color=LIGHT_ORANGE, alpha=0.68, linewidth=0)
    share_line = ax_b.plot(
        positions,
        share,
        color=ORANGE,
        linewidth=1.5,
        marker="o",
        markersize=2.5,
        label="synergy mass",
    )[0]
    ax_b.set_ylabel(f"{focal_label} share in top-{top_k} (%)", color=ORANGE)
    ax_b.tick_params(axis="y", colors=ORANGE)
    ax_b.set_xlabel("Evaluated forecast horizon, $H$")
    ax_b.set_xticks(positions, horizons, rotation=45)
    ax_b.set_ylim(bottom=0)
    ax_b.grid(axis="y", color=GRID, linewidth=0.55)

    ax_b_right = ax_b.twinx()
    target_line = ax_b_right.plot(
        positions,
        target_count,
        color=INK,
        linewidth=1.15,
        marker="s",
        markersize=2.2,
        label="target fan-out",
    )[0]
    ax_b_right.set_ylabel("Distinct targets", color=INK)
    ax_b_right.tick_params(axis="y", colors=INK)
    ax_b_right.spines["top"].set_visible(False)
    ax_b_right.spines["right"].set_linewidth(0.65)
    ax_b.legend(
        [share_line, target_line],
        ["synergy mass", "target fan-out"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        fontsize=5.3,
        handlelength=1.6,
        columnspacing=1.0,
    )
    h20_index = position_lookup[20]
    ax_b.axvline(h20_index, color="#8E9399", linestyle=":", linewidth=0.7)
    ax_b.text(
        h20_index - 0.25,
        ax_b.get_ylim()[1] * 0.97,
        "top-10 fully\nuses No.0 + No.1",
        ha="right",
        va="top",
        fontsize=5.2,
        color="#626A73",
    )
    add_panel_label(ax_b, "b")

    pair_pivot = (
        pairs.pivot(index="horizon", columns="pair", values="share")
        .reindex(horizons)
        .fillna(0.0)
    )
    selected_arrays = [
        pair_pivot[pair].to_numpy(dtype=float)
        if pair in pair_pivot.columns
        else np.zeros(len(horizons))
        for pair in selected_pairs
    ]
    selected_sum = np.sum(selected_arrays, axis=0)
    other = np.maximum(0.0, 1.0 - selected_sum)
    area_arrays = [100.0 * values for values in selected_arrays] + [100.0 * other]
    area_labels = selected_pairs + ["Other source pairs"]
    area_colors = list(PAIR_COLORS[: len(selected_pairs)]) + ["#E4E8EC"]
    ax_c.stackplot(
        positions,
        area_arrays,
        labels=area_labels,
        colors=area_colors,
        linewidth=0.25,
        edgecolor="white",
    )
    ax_c.axvline(h20_index, color="#5E646B", linestyle=":", linewidth=0.75)
    ax_c.set_xlim(positions[0], positions[-1])
    ax_c.set_ylim(0, 100)
    ax_c.set_ylabel(f"Synergy-mass composition of top-{top_k} (%)")
    ax_c.set_xlabel("Evaluated forecast horizon, $H$")
    ax_c.set_xticks(positions, horizons)
    ax_c.set_yticks((0, 25, 50, 75, 100))
    ax_c.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=5,
        fontsize=5.2,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    add_panel_label(ax_c, "c", x=-0.055)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, kwargs in ((".png", {"dpi": 600}), (".svg", {}), (".pdf", {})):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    configure_matplotlib()
    horizons = parse_horizons(args.horizons)
    rankings = load_rankings(args.result_dir, horizons)
    effective, pairs, selected_pairs = build_metrics(
        rankings,
        horizons,
        top_k=args.top_k,
        robustness_k=ROBUSTNESS_K,
    )
    outputs = plot_figure(
        effective,
        pairs,
        selected_pairs,
        horizons,
        top_k=args.top_k,
        output_base=args.output,
    )

    focal_label = pair_label(0, 1)
    focal = (
        pairs[pairs["pair"] == focal_label]
        .set_index("horizon")
        .reindex(horizons)
        .fillna({"share": 0.0, "target_count": 0})
    )
    summary = {
        "claim": (
            "The leading SLP synergy architecture condenses from many transient "
            "source pairs into a No.0 + No.1-centred multi-target backbone."
        ),
        "horizons": horizons,
        "primary_top_k": args.top_k,
        "robustness_top_k": list(ROBUSTNESS_K),
        "selected_area_pairs": selected_pairs,
        "valid_pair_count_top200": {
            str(int(row.horizon)): int(row.valid_pair_count)
            for row in effective[effective["top_k"] == args.top_k].itertuples(index=False)
        },
        "focal_pair_share_top200": {
            str(horizon): float(focal.loc[horizon, "share"])
            for horizon in horizons
        },
        "focal_pair_target_count_top200": {
            str(horizon): int(focal.loc[horizon, "target_count"])
            for horizon in horizons
        },
        "outputs": [str(path) for path in outputs],
    }
    summary_path = args.output.with_name(f"{args.output.name}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": summary["outputs"], "summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
