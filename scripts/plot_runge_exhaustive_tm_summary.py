#!/usr/bin/env python3
"""Summarize and plot exhaustive degree-3 TM Runge hyperedge results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
    DEFAULT_OLD_RERANK_DIR,
    DEFAULT_RESULT_DIR,
    load_valid_ranking,
    parse_horizons,
)


DEFAULT_OUTPUT = ROOT / "fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive_summary"
DEFAULT_HORIZONS = "1-10,15,20,30,40,50,60"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def load_summary(result_dir: Path, old_dir: Path, horizons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    for horizon in horizons:
        summary = json.loads((result_dir / f"H{horizon:03d}" / "summary.json").read_text(encoding="utf-8"))
        ranking = load_valid_ranking(
            result_dir / f"H{horizon:03d}" / "full_ranking.npz",
            expected_metadata=summary["ranking_metadata"],
        )
        if ranking is None:
            raise RuntimeError(f"H={horizon} full ranking failed metadata or integrity validation.")
        old = pd.read_csv(old_dir / f"H{horizon:03d}_discrete_top1000_tm_rerank.csv")
        horizon_rows.append(
            {
                "horizon": horizon,
                "top10_covered": int(summary["rank_diagnostics"]["exhaustive_top10_in_old_shortlist"]),
                "old_discrete_top1_tm_rank": int(summary["rank_diagnostics"]["best_old_discrete_candidate_exhaustive_tm_rank"]),
                "max_delta2_exhaustive": float(ranking.iloc[0]["delta2_tm"]),
                "max_delta2_old_shortlist": float(old["delta2_tm"].max()),
                "runtime_seconds": float(summary["runtime_seconds"]),
                "max_abs_error": float(summary["max_abs_error"]),
            }
        )
        for rank in range(10):
            source_a = int(ranking.iloc[rank]["source_a"])
            source_b = int(ranking.iloc[rank]["source_b"])
            target = int(ranking.iloc[rank]["target"])
            top_rows.append(
                {
                    "horizon": horizon,
                    "rank": rank + 1,
                    "edge": f"{local_to_paper(source_a)}+{local_to_paper(source_b)}→{local_to_paper(target)}",
                    "delta2_tm": float(ranking.iloc[rank]["delta2_tm"]),
                }
            )
    return pd.DataFrame(horizon_rows), pd.DataFrame(top_rows)


def plot_summary(horizon_frame: pd.DataFrame, top_frame: pd.DataFrame, output_base: Path) -> list[Path]:
    horizons = horizon_frame["horizon"].astype(int).tolist()
    positions = np.arange(len(horizons))
    fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.86, 1.2])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    coverage = horizon_frame["top10_covered"].to_numpy(dtype=float)
    ax_a.bar(positions, coverage, color="#4C78A8", width=0.72)
    ax_a.axhline(10, color="#555555", linewidth=0.8, linestyle="--")
    ax_a.set_ylabel("Exhaustive TM top-10\nin discrete top-1000")
    ax_a.set_ylim(0, 10.8)
    ax_a.set_xticks(positions, horizons, rotation=45)
    ax_a.set_xlabel("Forecast horizon H")

    old_ranks = horizon_frame["old_discrete_top1_tm_rank"].to_numpy(dtype=float)
    ax_b.plot(positions, old_ranks, marker="o", color="#E45756", linewidth=1.25, markersize=3.2)
    ax_b.axhline(1000, color="#777777", linestyle="--", linewidth=0.8, label="Top-1000 cutoff")
    ax_b.set_yscale("log")
    ax_b.set_ylabel("TM rank of discrete top-1")
    ax_b.set_xticks(positions, horizons, rotation=45)
    ax_b.set_xlabel("Forecast horizon H")
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, 1.19), frameon=False, ncol=1)

    ax_c.plot(
        positions,
        horizon_frame["max_delta2_exhaustive"],
        marker="o",
        color="#4C78A8",
        linewidth=1.45,
        markersize=3.1,
        label="Exhaustive TM",
    )
    ax_c.plot(
        positions,
        horizon_frame["max_delta2_old_shortlist"],
        marker="s",
        color="#B8B8B8",
        linewidth=1.15,
        markersize=2.8,
        label="Discrete shortlist",
    )
    ax_c.set_ylabel(r"Maximum $\Delta_{2,\mathrm{TM}}$ (bits)")
    ax_c.set_xticks(positions, horizons, rotation=45)
    ax_c.set_xlabel("Forecast horizon H")
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False, ncol=2)

    recurrence = Counter(top_frame["edge"])
    selected = [edge for edge, _ in sorted(recurrence.items(), key=lambda item: (-item[1], item[0]))[:12]]
    matrix = np.full((len(selected), len(horizons)), np.nan, dtype=float)
    horizon_to_col = {horizon: index for index, horizon in enumerate(horizons)}
    edge_to_row = {edge: index for index, edge in enumerate(selected)}
    for row in top_frame.itertuples(index=False):
        if row.edge in edge_to_row:
            matrix[edge_to_row[row.edge], horizon_to_col[int(row.horizon)]] = int(row.rank)
    cmap = mpl.colormaps["viridis_r"].copy()
    cmap.set_bad("#F1F1F1")
    image = ax_d.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=1, vmax=10)
    ax_d.set_yticks(np.arange(len(selected)), selected)
    ax_d.set_xticks(positions, horizons, rotation=45)
    ax_d.set_xlabel("Forecast horizon H")
    ax_d.set_ylabel("Recurrent exhaustive TM top-10 hyperedges")
    colorbar = fig.colorbar(image, ax=ax_d, location="right", shrink=0.82, pad=0.03)
    colorbar.set_label("Within-horizon rank")

    for label, ax in zip("abcd", (ax_a, ax_b, ax_c, ax_d), strict=True):
        ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, kwargs in ((".png", {"dpi": 600}), (".svg", {}), (".pdf", {})):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--old-rerank-dir", type=Path, default=DEFAULT_OLD_RERANK_DIR)
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    horizon_frame, top_frame = load_summary(args.result_dir, args.old_rerank_dir, parse_horizons(args.horizons))
    outputs = plot_summary(horizon_frame, top_frame, args.output)
    payload = {
        "horizons": horizon_frame.to_dict(orient="records"),
        "top10_recurrence": (
            top_frame.groupby("edge", as_index=False)
            .agg(count=("horizon", "nunique"), max_delta2=("delta2_tm", "max"))
            .sort_values(["count", "max_delta2"], ascending=[False, False])
            .to_dict(orient="records")
        ),
        "outputs": [str(path) for path in outputs],
    }
    args.output.with_name(f"{args.output.name}_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": payload["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
