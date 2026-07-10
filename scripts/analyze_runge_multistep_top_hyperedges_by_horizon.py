#!/usr/bin/env python3
"""Compare top second-order Runge ridge hyperedges across selected horizons."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_runge_gateway_mediator_map import local_to_paper
from plot_runge_multistep_ridge_node0_hyperedges import DEFAULT_RESULT_DIR, load_multistep_delta2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "fig" / "runge_slp_daily_1948_2026_20260628" / "multistep_conditioned_ei" / "top10_order2_hyperedges_by_horizon.png"


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


def parse_horizons(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one horizon is required.")
    return sorted(dict.fromkeys(values))


def top_hyperedges_for_horizon(result_dir: Path, horizon: int, top_n: int) -> pd.DataFrame:
    delta2 = load_multistep_delta2(result_dir, int(horizon), cumulative=False)
    rows: list[dict[str, object]] = []
    n = delta2.shape[0]
    for source_a in range(n):
        for source_b in range(source_a + 1, n):
            for target in range(n):
                if target in (source_a, source_b):
                    continue
                value = float(delta2[source_a, source_b, target])
                if not np.isfinite(value) or value <= 0.0:
                    continue
                source_a_paper = local_to_paper(source_a)
                source_b_paper = local_to_paper(source_b)
                target_paper = local_to_paper(target)
                edge_label = f"{source_a_paper}+{source_b_paper}->{target_paper}"
                pair_label = f"{source_a_paper}+{source_b_paper}"
                rows.append(
                    {
                        "horizon": int(horizon),
                        "source_a": source_a,
                        "source_b": source_b,
                        "target_index": target,
                        "source_a_paper": source_a_paper,
                        "source_b_paper": source_b_paper,
                        "target_paper": target_paper,
                        "pair_label": pair_label,
                        "edge_label": edge_label,
                        "delta2": value,
                    }
                )
    frame = pd.DataFrame(rows).sort_values("delta2", ascending=False).head(int(top_n)).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame


def classify_edge(row: pd.Series) -> str:
    sources = {int(row["source_a_paper"]), int(row["source_b_paper"])}
    if sources == {3, 18}:
        return "No.3+18"
    if 0 in sources:
        return "No.0 source"
    return "Other"


def select_heatmap_edges(top: pd.DataFrame, horizons: list[int], max_rows: int) -> list[str]:
    final_h = max(horizons)
    final_edges = top[top["horizon"] == final_h]["edge_label"].tolist()
    counts = top.groupby("edge_label").size().sort_values(ascending=False)
    recurring = [edge for edge in counts.index if counts.loc[edge] >= 2]
    ordered: list[str] = []
    for edge in recurring + final_edges:
        if edge not in ordered:
            ordered.append(edge)
    if len(ordered) < int(max_rows):
        strongest = top.groupby("edge_label")["delta2"].max().sort_values(ascending=False).index.tolist()
        for edge in strongest:
            if edge not in ordered:
                ordered.append(edge)
            if len(ordered) >= int(max_rows):
                break
    return ordered[: int(max_rows)]


def plot_summary(top: pd.DataFrame, horizons: list[int], output: Path) -> None:
    top = top.copy()
    top["family"] = top.apply(classify_edge, axis=1)
    families = ["No.0 source", "No.3+18", "Other"]
    family_colors = {"No.0 source": "#4c78a8", "No.3+18": "#f58518", "Other": "#b8b8b8"}
    counts = (
        top.groupby(["horizon", "family"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=horizons, columns=families, fill_value=0)
    )
    max_delta = top.groupby("horizon")["delta2"].max().reindex(horizons)

    heat_edges = select_heatmap_edges(top, horizons, max_rows=16)
    matrix = np.full((len(heat_edges), len(horizons)), np.nan, dtype=float)
    rank_text = [["" for _ in horizons] for _ in heat_edges]
    for row_idx, edge in enumerate(heat_edges):
        subset = top[top["edge_label"] == edge]
        for item in subset.itertuples(index=False):
            col_idx = horizons.index(int(item.horizon))
            matrix[row_idx, col_idx] = float(item.delta2)
            rank_text[row_idx][col_idx] = str(int(item.rank))

    fig = plt.figure(figsize=(8.2, 6.3), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[0.95, 2.3])
    ax_bar = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[1, 0])

    bottom = np.zeros(len(horizons), dtype=float)
    x = np.arange(len(horizons))
    for family in families:
        values = counts[family].to_numpy(dtype=float)
        ax_bar.bar(x, values, bottom=bottom, color=family_colors[family], width=0.68, label=family)
        bottom += values
    ax_line = ax_bar.twinx()
    ax_line.plot(x, max_delta.to_numpy(dtype=float), color="#333333", marker="o", linewidth=1.5, markersize=3.5, label="top delta2")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([str(h) for h in horizons])
    ax_bar.set_ylabel("Top-10 count")
    ax_bar.set_xlabel("Horizon H")
    ax_line.set_ylabel("max delta2")
    ax_bar.set_ylim(0, 10)
    ax_bar.legend(loc="center left", bbox_to_anchor=(1.08, 0.62), frameon=False)
    ax_line.legend(loc="center left", bbox_to_anchor=(1.08, 0.20), frameon=False)

    masked = np.ma.masked_invalid(matrix)
    cmap = mpl.colormaps["YlOrRd"].copy()
    cmap.set_bad("#f2f2f2")
    vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
    image = ax_heat.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    ax_heat.set_xticks(np.arange(len(horizons)))
    ax_heat.set_xticklabels([str(h) for h in horizons])
    ax_heat.set_yticks(np.arange(len(heat_edges)))
    ax_heat.set_yticklabels(heat_edges)
    ax_heat.set_xlabel("Horizon H")
    ax_heat.set_title("Selected top-10 hyperedges across horizons; cell text is rank")
    for row_idx in range(len(heat_edges)):
        for col_idx in range(len(horizons)):
            if rank_text[row_idx][col_idx]:
                ax_heat.text(col_idx, row_idx, rank_text[row_idx][col_idx], ha="center", va="center", fontsize=6.5)
    cbar = fig.colorbar(image, ax=ax_heat, location="right", shrink=0.82, pad=0.02)
    cbar.set_label("delta2")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=350, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_summary(top: pd.DataFrame, output: Path, horizons: list[int]) -> None:
    top = top.copy()
    top["family"] = top.apply(classify_edge, axis=1)
    family_counts = (
        top.groupby(["horizon", "family"])
        .size()
        .rename("top10_count")
        .reset_index()
        .sort_values(["horizon", "family"])
    )
    family_counts.to_csv(output.with_name(output.stem + "_family_counts.csv"), index=False)
    recurrence = (
        top.groupby("edge_label")
        .agg(
            appearances=("horizon", "count"),
            first_horizon=("horizon", "min"),
            last_horizon=("horizon", "max"),
            max_delta2=("delta2", "max"),
            best_rank=("rank", "min"),
        )
        .sort_values(["appearances", "max_delta2"], ascending=[False, False])
        .reset_index()
    )
    recurrence.to_csv(output.with_name(output.stem + "_recurrence.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--horizons", default="1,2,4,6,8,10")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    horizons = parse_horizons(str(args.horizons))
    result_dir = Path(args.result_dir).expanduser()
    top = pd.concat(
        [top_hyperedges_for_horizon(result_dir, horizon, int(args.top_n)) for horizon in horizons],
        ignore_index=True,
    )
    output = Path(args.output).expanduser()
    csv_path = output.with_suffix(".csv")
    top.to_csv(csv_path, index=False)
    write_summary(top, output, horizons)
    plot_summary(top, horizons, output)
    print(output)
    print(csv_path)
    print(top[["horizon", "rank", "edge_label", "delta2"]].to_string(index=False))


if __name__ == "__main__":
    main()
