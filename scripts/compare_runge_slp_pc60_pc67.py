#!/usr/bin/env python3
"""Compare the controlled 60-PC and 67-PC exhaustive SLP experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_earth_system_main_figures import build_pair01_geographic_coverage
from scripts.plot_runge_exhaustive_tm_maps import load_nodes
from scripts.plot_runge_gateway_mediator_map import PAPER_TO_LOCAL, local_to_paper
from scripts.plot_runge_source_pair_condensation import (
    ROBUSTNESS_K,
    aggregate_pair_weights,
    build_metrics,
    load_rankings,
)

HORIZONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60)
OLD_ROOT = ROOT / "results/runge_slp_daily_1948_2026_20260628"
NEW_ROOT = ROOT / "results/runge_slp_daily_1948_2026_pc67_20260731"
OLD_EXHAUSTIVE = (
    OLD_ROOT
    / "mlp_tm_ei_lag04/results/runge/multistep_conditioned_ei_tm_exhaustive"
)
NEW_EXHAUSTIVE = (
    NEW_ROOT
    / "mlp_tm_ei_lag04/results/runge/multistep_conditioned_ei_tm_exhaustive"
)
OLD_MAPS = OLD_ROOT / "results/runge/2015_gateways/component_maps.npz"
NEW_MAPS = NEW_ROOT / "results/runge/2015_gateways/component_maps.npz"
OLD_TRENDS = (
    ROOT
    / "fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/"
    "forced_tm_edge_trends_H001_H060.csv"
)
NEW_TRENDS = (
    ROOT
    / "fig/runge_slp_daily_1948_2026_pc67_20260731/multistep_conditioned_ei_tm_targeted/"
    "forced_tm_edge_trends_H001_H060.csv"
)
DEFAULT_OUTPUT = ROOT / "fig/runge_slp_pc60_pc67_stability"


def parse_edge(text: str) -> tuple[int, int, int]:
    sources, target = str(text).split("->", 1)
    source_a, source_b = sources.split("+", 1)
    return int(source_a), int(source_b), int(target)


def paper_to_local(index: int) -> int:
    return int(PAPER_TO_LOCAL.get(int(index), int(index)))


def match_component_maps(old_path: Path, new_path: Path) -> tuple[pd.DataFrame, dict[int, int]]:
    old = np.load(old_path, allow_pickle=False)["component_maps"]
    new = np.load(new_path, allow_pickle=False)["component_maps"]
    if old.shape[:2] != new.shape[:2]:
        raise ValueError("The component-map grids do not match.")
    old_flat = old.reshape(-1, old.shape[-1]).astype(float)
    new_flat = new.reshape(-1, new.shape[-1]).astype(float)
    old_flat = (old_flat - old_flat.mean(axis=0)) / old_flat.std(axis=0, ddof=1)
    new_flat = (new_flat - new_flat.mean(axis=0)) / new_flat.std(axis=0, ddof=1)
    correlations = old_flat.T @ new_flat / (old_flat.shape[0] - 1)
    old_indices, new_indices = linear_sum_assignment(-np.abs(correlations))
    frame = pd.DataFrame(
        {
            "pc60": old_indices.astype(int),
            "pc67": new_indices.astype(int),
            "spatial_correlation": correlations[old_indices, new_indices],
        }
    ).sort_values("pc60", ignore_index=True)
    frame["absolute_spatial_correlation"] = frame["spatial_correlation"].abs()
    return frame, dict(zip(frame["pc60"].astype(int), frame["pc67"].astype(int), strict=True))


def curve_correlation(left: np.ndarray, right: np.ndarray, method: str) -> float | None:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) <= 1e-14 or np.std(y[valid]) <= 1e-14:
        return None
    result = pearsonr(x[valid], y[valid]) if method == "pearson" else spearmanr(x[valid], y[valid])
    return float(result.statistic)


def focal_pair_series(
    rankings: dict[int, pd.DataFrame],
    pair: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    shares, targets = [], []
    for horizon in HORIZONS:
        weights = aggregate_pair_weights(rankings[horizon], 200)
        total = float(weights["weight"].sum())
        focal = weights[
            (weights["source_a"] == pair[0]) & (weights["source_b"] == pair[1])
        ]
        shares.append(float(focal.iloc[0]["weight"] / total) if len(focal) and total > 0 else 0.0)
        targets.append(int(focal.iloc[0]["target_count"]) if len(focal) else 0)
    return np.asarray(shares), np.asarray(targets)


def mapped_top10_overlap(
    old_rankings: dict[int, pd.DataFrame],
    new_rankings: dict[int, pd.DataFrame],
    old_to_new: dict[int, int],
) -> dict[str, float]:
    reverse = {new: old for old, new in old_to_new.items()}
    overlaps = {}
    for horizon in HORIZONS:
        old_keys = {
            (int(row.source_a), int(row.source_b), int(row.target))
            for row in old_rankings[horizon].head(10).itertuples(index=False)
        }
        new_keys = set()
        for row in new_rankings[horizon].head(10).itertuples(index=False):
            values = (int(row.source_a), int(row.source_b), int(row.target))
            if all(value in reverse for value in values):
                a, b, target = (reverse[value] for value in values)
                new_keys.add((min(a, b), max(a, b), target))
        overlaps[str(horizon)] = len(old_keys & new_keys) / 10.0
    return overlaps


def dominant_pair_comparison(
    old_rankings: dict[int, pd.DataFrame],
    new_rankings: dict[int, pd.DataFrame],
    old_to_new: dict[int, int],
) -> dict[str, dict[str, object]]:
    reverse = {new: old for old, new in old_to_new.items()}
    result = {}
    for horizon in (10, 60):
        old_row = aggregate_pair_weights(old_rankings[horizon], 200).iloc[0]
        new_row = aggregate_pair_weights(new_rankings[horizon], 200).iloc[0]
        old_pair = (int(old_row.source_a), int(old_row.source_b))
        new_pair = (int(new_row.source_a), int(new_row.source_b))
        mapped_new = (
            tuple(sorted(reverse[value] for value in new_pair))
            if all(value in reverse for value in new_pair)
            else None
        )
        result[str(horizon)] = {
            "pc60": list(old_pair),
            "pc67": list(new_pair),
            "pc67_mapped_to_pc60": list(mapped_new) if mapped_new is not None else None,
            "same_after_matching": bool(mapped_new == old_pair),
        }
    return result


def forced_curve_metrics(
    old_path: Path,
    new_path: Path,
    old_to_new: dict[int, int],
) -> dict[str, dict[str, object]]:
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    result: dict[str, dict[str, object]] = {}
    for old_label, old_frame in old.groupby("edge_label_paper", sort=False):
        a_paper, b_paper, target_paper = parse_edge(str(old_label))
        a = old_to_new[paper_to_local(a_paper)]
        b = old_to_new[paper_to_local(b_paper)]
        target = old_to_new[paper_to_local(target_paper)]
        new_label = f"{local_to_paper(min(a, b))}+{local_to_paper(max(a, b))}->{local_to_paper(target)}"
        new_frame = new[new["edge_label_paper"] == new_label]
        merged = old_frame[["horizon", "delta2_tm"]].merge(
            new_frame[["horizon", "delta2_tm"]],
            on="horizon",
            suffixes=("_pc60", "_pc67"),
        )
        correlation = curve_correlation(
            merged["delta2_tm_pc60"].to_numpy(),
            merged["delta2_tm_pc67"].to_numpy(),
            "pearson",
        )
        result[str(old_label)] = {
            "mapped_pc67_edge": new_label,
            "pearson": correlation,
            "passes_0_90": bool(correlation is not None and correlation >= 0.90),
        }
    return result


def plot_comparison(
    output: Path,
    retained_old: np.ndarray,
    retained_new: np.ndarray,
    share_old: np.ndarray,
    share_new: np.ndarray,
    targets_old: np.ndarray,
    targets_new: np.ndarray,
    span_old: np.ndarray,
    span_new: np.ndarray,
) -> list[Path]:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    positions = np.arange(len(HORIZONS))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    panels = (
        (retained_old, retained_new, "Source pairs retained in top-200"),
        (100 * share_old, 100 * share_new, "Matched No.0 + No.1 synergy mass (%)"),
        (targets_old, targets_new, "Matched No.0 + No.1 distinct targets"),
        (span_old, span_new, "Maximum target span (km)"),
    )
    for label, ax, (old, new, ylabel) in zip("abcd", axes.flat, panels, strict=True):
        ax.plot(positions, old, color="#3F6F9F", marker="o", markersize=2.4, label="60 PCs")
        ax.plot(
            positions,
            new,
            color="#D9822B",
            linestyle="--",
            marker="s",
            markersize=2.2,
            label="67 PCs",
        )
        ax.set_xticks(positions, HORIZONS, rotation=45)
        ax.set_xlabel("Evaluated horizon, $H$")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E9EDF2", linewidth=0.55)
        ax.text(-0.13, 1.03, label, transform=ax.transAxes, fontweight="bold", fontsize=8.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [output.with_suffix(suffix) for suffix in (".png", ".svg", ".pdf")]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-exhaustive", type=Path, default=OLD_EXHAUSTIVE)
    parser.add_argument("--new-exhaustive", type=Path, default=NEW_EXHAUSTIVE)
    parser.add_argument("--old-maps", type=Path, default=OLD_MAPS)
    parser.add_argument("--new-maps", type=Path, default=NEW_MAPS)
    parser.add_argument("--old-trends", type=Path, default=OLD_TRENDS)
    parser.add_argument("--new-trends", type=Path, default=NEW_TRENDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    matches, old_to_new = match_component_maps(args.old_maps, args.new_maps)
    old_rankings = load_rankings(args.old_exhaustive, list(HORIZONS))
    new_rankings = load_rankings(args.new_exhaustive, list(HORIZONS))
    effective_old, _, _ = build_metrics(
        old_rankings, list(HORIZONS), top_k=200, robustness_k=ROBUSTNESS_K
    )
    effective_new, _, _ = build_metrics(
        new_rankings, list(HORIZONS), top_k=200, robustness_k=ROBUSTNESS_K
    )
    retained_old = (
        effective_old[effective_old["top_k"] == 200]
        .set_index("horizon")
        .loc[list(HORIZONS), "valid_pair_count"]
        .to_numpy(dtype=float)
    )
    retained_new = (
        effective_new[effective_new["top_k"] == 200]
        .set_index("horizon")
        .loc[list(HORIZONS), "valid_pair_count"]
        .to_numpy(dtype=float)
    )
    new_focal = tuple(sorted((old_to_new[0], old_to_new[1])))
    share_old, targets_old = focal_pair_series(old_rankings, (0, 1))
    share_new, targets_new = focal_pair_series(new_rankings, new_focal)
    coverage_old = build_pair01_geographic_coverage(
        old_rankings, load_nodes(args.old_maps), focal_pair=(0, 1)
    )
    coverage_new = build_pair01_geographic_coverage(
        new_rankings, load_nodes(args.new_maps), focal_pair=new_focal
    )
    span_old = (
        coverage_old[coverage_old["top_k"] == 200]
        .set_index("horizon")
        .loc[list(HORIZONS), "max_target_span_km"]
        .to_numpy()
    )
    span_new = (
        coverage_new[coverage_new["top_k"] == 200]
        .set_index("horizon")
        .loc[list(HORIZONS), "max_target_span_km"]
        .to_numpy()
    )
    mean_relative_deviation = float(
        np.mean(np.abs(retained_new - retained_old) / np.maximum(retained_old, 1.0))
    )
    metrics = {
        "component_matching": {
            "mean_absolute_spatial_correlation": float(matches["absolute_spatial_correlation"].mean()),
            "median_absolute_spatial_correlation": float(matches["absolute_spatial_correlation"].median()),
            "minimum_absolute_spatial_correlation": float(matches["absolute_spatial_correlation"].min()),
            "matches_below_0_90": int((matches["absolute_spatial_correlation"] < 0.90).sum()),
            "unmatched_pc67": sorted(set(range(67)) - set(old_to_new.values())),
            "matches": matches.to_dict(orient="records"),
        },
        "retained_pair_curve": {
            "spearman": curve_correlation(retained_old, retained_new, "spearman"),
            "mean_relative_deviation": mean_relative_deviation,
        },
        "focal_pair": {"pc60": [0, 1], "pc67": list(new_focal)},
        "focal_mass_curve": {
            "spearman": curve_correlation(share_old, share_new, "spearman"),
        },
        "focal_target_count_curve": {
            "spearman": curve_correlation(targets_old, targets_new, "spearman"),
        },
        "focal_maximum_span_curve": {
            "spearman": curve_correlation(span_old, span_new, "spearman"),
        },
        "top10_matched_overlap_fraction": mapped_top10_overlap(
            old_rankings, new_rankings, old_to_new
        ),
        "dominant_pair_top200": dominant_pair_comparison(
            old_rankings, new_rankings, old_to_new
        ),
        "forced_edges": forced_curve_metrics(args.old_trends, args.new_trends, old_to_new),
    }
    metrics["pre_registered_checks"] = {
        "retained_pair_curve_pass": bool(
            metrics["retained_pair_curve"]["spearman"] is not None
            and metrics["retained_pair_curve"]["spearman"] >= 0.90
            and mean_relative_deviation <= 0.15
        ),
        "focal_mass_curve_pass": bool(
            metrics["focal_mass_curve"]["spearman"] is not None
            and metrics["focal_mass_curve"]["spearman"] >= 0.90
        ),
        "focal_target_count_curve_pass": bool(
            metrics["focal_target_count_curve"]["spearman"] is not None
            and metrics["focal_target_count_curve"]["spearman"] >= 0.90
        ),
        "focal_maximum_span_curve_pass": bool(
            metrics["focal_maximum_span_curve"]["spearman"] is not None
            and metrics["focal_maximum_span_curve"]["spearman"] >= 0.90
        ),
        "all_forced_edges_pass": all(
            bool(item["passes_0_90"]) for item in metrics["forced_edges"].values()
        ),
        "dominant_pair_identity_pass": all(
            bool(item["same_after_matching"])
            for item in metrics["dominant_pair_top200"].values()
        ),
    }
    outputs = plot_comparison(
        args.output,
        retained_old,
        retained_new,
        share_old,
        share_new,
        targets_old,
        targets_new,
        span_old,
        span_new,
    )
    metrics["outputs"] = [str(path) for path in outputs]
    summary = args.output.with_name(f"{args.output.name}_summary.json")
    summary.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"summary": str(summary), "outputs": metrics["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
