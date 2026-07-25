#!/usr/bin/env python3
"""Paired Runge PEID estimator-order comparison on fixed intervention rollouts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.TM.transport_map_density import estimate_mutual_information_transport_map
from scripts.run_runge_exhaustive_degree3_tm import (
    fingerprint_array,
    score_horizon_degree3_tm,
)


BASE = (
    ROOT
    / "results"
    / "runge_slp_daily_1948_2026_20260628"
    / "mlp_tm_ei_lag04"
    / "results"
    / "runge"
)
DEFAULT_SOURCE = BASE / "multistep_conditioned_ei_tm_exhaustive" / "source_samples_n4096.npy"
DEFAULT_ROLLOUT = (
    BASE
    / "multistep_conditioned_ei_tm_forced_edges"
    / "rollout_predictions_H060_n4096.npy"
)
DEFAULT_DEGREE3 = BASE / "multistep_conditioned_ei_tm_exhaustive"
DEFAULT_OUTPUT = BASE / "multistep_conditioned_ei_estimator_comparison"
DEFAULT_FIGURE = ROOT / "fig" / "runge_tm_estimator_comparison"

RANKING_HORIZONS = (1, 10, 60)
CURVE_HORIZONS = tuple(range(1, 11)) + (15, 20, 30, 40, 50, 60)
DEGREES = (1, 2, 3, 4)
REPRESENTATIVE_EDGES = (
    (0, 6, 32, "Early peak"),
    (0, 1, 28, "Mid-range peak"),
    (0, 1, 50, "Long plateau"),
    (0, 1, 46, "Long-range growth"),
)
RANKING_COLUMNS = (
    "source_a",
    "source_b",
    "target",
    "raw_ei_a",
    "raw_ei_b",
    "raw_joint_ei",
    "ei_a",
    "ei_b",
    "joint_ei",
    "delta2_tm",
    "tm_rank",
)


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(temporary, **payload)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ranking_path(output_dir: Path, degree: int, horizon: int) -> Path:
    return output_dir / f"degree{degree}" / f"H{horizon:03d}" / "full_ranking.npz"


def _load_ranking(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {column: np.asarray(payload[column]) for column in RANKING_COLUMNS}
    return pd.DataFrame(arrays)


def _load_degree3_ranking(degree3_dir: Path, horizon: int) -> pd.DataFrame:
    return _load_ranking(degree3_dir / f"H{horizon:03d}" / "full_ranking.npz")


def _compute_ranking(
    sources: np.ndarray,
    targets: np.ndarray,
    *,
    degree: int,
    horizon: int,
) -> pd.DataFrame:
    frame = score_horizon_degree3_tm(sources, targets, degree=int(degree))
    frame = frame.sort_values("delta2_tm", ascending=False, kind="mergesort").reset_index(drop=True)
    frame["tm_rank"] = np.arange(1, len(frame) + 1)
    return frame


def _save_ranking(
    frame: pd.DataFrame,
    path: Path,
    *,
    degree: int,
    horizon: int,
    source_hash: str,
    rollout_hash: str,
) -> None:
    metadata = {
        "degree": int(degree),
        "horizon": int(horizon),
        "source_hash": source_hash,
        "rollout_hash": rollout_hash,
        "candidate_count": int(len(frame)),
    }
    payload = {column: frame[column].to_numpy() for column in RANKING_COLUMNS}
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    _atomic_npz(path, payload)


def _candidate_key(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[["source_a", "source_b", "target"]])


def _align_to_reference(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    keyed = frame.set_index(["source_a", "source_b", "target"])
    return keyed.loc[_candidate_key(reference)].reset_index()


def _overlap(left: pd.DataFrame, right: pd.DataFrame, n: int) -> float:
    left_keys = set(map(tuple, left.head(n)[["source_a", "source_b", "target"]].to_numpy()))
    right_keys = set(map(tuple, right.head(n)[["source_a", "source_b", "target"]].to_numpy()))
    return len(left_keys & right_keys) / float(n)


def _rank_summary(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    degree: int,
    horizon: int,
    runtime_seconds: float,
) -> dict[str, float | int]:
    aligned = _align_to_reference(frame, reference)
    reference_values = reference["delta2_tm"].to_numpy(dtype=float)
    values = aligned["delta2_tm"].to_numpy(dtype=float)
    rho = float(spearmanr(values, reference_values).statistic)
    return {
        "degree": int(degree),
        "horizon": int(horizon),
        "candidate_count": int(len(frame)),
        "top1_delta2_bits": float(frame.iloc[0]["delta2_tm"]),
        "top10_mean_delta2_bits": float(frame.head(10)["delta2_tm"].mean()),
        "top10_median_delta2_bits": float(frame.head(10)["delta2_tm"].median()),
        "positive_fraction": float((frame["delta2_tm"] > 0).mean()),
        "spearman_vs_degree3": rho,
        "top10_overlap_vs_degree3": _overlap(frame, reference, 10),
        "top50_overlap_vs_degree3": _overlap(frame, reference, 50),
        "median_abs_delta_vs_degree3_bits": float(np.median(np.abs(values - reference_values))),
        "runtime_seconds": float(runtime_seconds),
    }


def _edge_metrics(
    sources: np.ndarray,
    target: np.ndarray,
    *,
    source_a: int,
    source_b: int,
    degree: int,
) -> dict[str, float]:
    target_values = np.asarray(target, dtype=float).reshape(-1, 1)
    raw_a = float(
        estimate_mutual_information_transport_map(
            sources[:, [source_a]], target_values, degree=int(degree)
        )["mi_hat"]
    )
    raw_b = float(
        estimate_mutual_information_transport_map(
            sources[:, [source_b]], target_values, degree=int(degree)
        )["mi_hat"]
    )
    raw_joint = float(
        estimate_mutual_information_transport_map(
            sources[:, [source_a, source_b]], target_values, degree=int(degree)
        )["mi_hat"]
    )
    ei_a = max(0.0, raw_a)
    ei_b = max(0.0, raw_b)
    joint = max(0.0, raw_joint)
    return {
        "raw_ei_a": raw_a,
        "raw_ei_b": raw_b,
        "raw_joint_ei": raw_joint,
        "ei_a": ei_a,
        "ei_b": ei_b,
        "joint_ei": joint,
        "delta2_tm": joint - ei_a - ei_b,
    }


def _curve_rows(sources: np.ndarray, rollout: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for degree in DEGREES:
        for horizon in CURVE_HORIZONS:
            for source_a, source_b, target, pattern in REPRESENTATIVE_EDGES:
                metrics = _edge_metrics(
                    sources,
                    rollout[:, horizon - 1, target],
                    source_a=source_a,
                    source_b=source_b,
                    degree=degree,
                )
                rows.append(
                    {
                        "degree": degree,
                        "estimator": estimator_label(degree),
                        "horizon": horizon,
                        "source_a": source_a,
                        "source_b": source_b,
                        "target": target,
                        "edge": f"{source_a}+{source_b}→{target}",
                        "pattern": pattern,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def estimator_label(degree: int) -> str:
    return {
        1: "Gaussian / affine TM",
        2: "Quadratic TM",
        3: "Cubic TM",
        4: "Quartic TM",
    }[int(degree)]


def _top_rows(rankings: dict[tuple[int, int], pd.DataFrame], n: int = 20) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for (degree, horizon), frame in sorted(rankings.items()):
        current = frame.head(n).copy()
        current.insert(0, "degree", degree)
        current.insert(1, "estimator", estimator_label(degree))
        current.insert(2, "horizon", horizon)
        frames.append(current)
    return pd.concat(frames, ignore_index=True)


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.18, 1.08, label, transform=axis.transAxes, fontsize=9, fontweight="bold")


def plot_comparison(curves: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    _configure_plotting()
    colors = {
        1: "#6E7781",
        2: "#5B8DB8",
        3: "#D7793F",
        4: "#8C6BB1",
    }
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 0.92))
    curve_axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    rank_axis = fig.add_subplot(grid[0, 2])
    overlap_axis = fig.add_subplot(grid[1, 2])

    for axis, (_, edge_frame), label in zip(
        curve_axes,
        curves.groupby(["source_a", "source_b", "target"], sort=False),
        ("a", "b", "c", "d"),
    ):
        first = edge_frame.iloc[0]
        for degree in DEGREES:
            values = edge_frame[edge_frame["degree"] == degree].sort_values("horizon")
            axis.plot(
                values["horizon"],
                values["delta2_tm"],
                color=colors[degree],
                marker=markers[degree],
                markersize=2.8,
                linewidth=1.2,
                label=estimator_label(degree),
            )
        axis.axhline(0.0, color="#B8B8B8", linewidth=0.7, zorder=0)
        axis.set_title(f"{first['edge']}  |  {first['pattern']}", loc="left", fontsize=7)
        axis.set_xlabel("Prediction horizon, H")
        axis.set_ylabel(r"$\Delta_2$ (bits)")
        axis.set_xticks((1, 10, 20, 40, 60))
        _panel_label(axis, label)

    for degree in DEGREES:
        values = summary[summary["degree"] == degree].sort_values("horizon")
        rank_axis.plot(
            values["horizon"],
            values["spearman_vs_degree3"],
            color=colors[degree],
            marker=markers[degree],
            markersize=3.5,
            linewidth=1.2,
        )
        overlap_axis.plot(
            values["horizon"],
            values["top10_overlap_vs_degree3"],
            color=colors[degree],
            marker=markers[degree],
            markersize=3.5,
            linewidth=1.2,
        )
    rank_axis.set_xlabel("Prediction horizon, H")
    rank_axis.set_ylabel("Full-rank Spearman vs cubic TM")
    rank_axis.set_xticks(RANKING_HORIZONS)
    rank_axis.set_ylim(-0.05, 1.05)
    _panel_label(rank_axis, "e")
    overlap_axis.set_xlabel("Prediction horizon, H")
    overlap_axis.set_ylabel("Top-10 overlap vs cubic TM")
    overlap_axis.set_xticks(RANKING_HORIZONS)
    overlap_axis.set_ylim(-0.05, 1.05)
    _panel_label(overlap_axis, "f")

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[degree],
            marker=markers[degree],
            markersize=3.5,
            linewidth=1.2,
            label=estimator_label(degree),
        )
        for degree in DEGREES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=4,
        frameon=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_contract(output_dir: Path, source_hash: str, rollout_hash: str) -> None:
    contract = {
        "scientific_question": (
            "What changes when only the Runge continuous EI estimator order changes?"
        ),
        "treatment_factor": "transport-map polynomial degree / Gaussian affine limit",
        "treatment_levels": {
            "1": "Gaussian / affine TM",
            "2": "quadratic nonlinear TM",
            "3": "cubic nonlinear TM; manuscript baseline",
            "4": "quartic nonlinear TM",
        },
        "pairing_unit": "same source pair, target, prediction horizon, and intervention rollout",
        "primary_metrics": [
            "full-candidate Spearman rank correlation versus degree 3",
            "top-10 overlap versus degree 3",
            "paired delta2 magnitude",
        ],
        "ranking_horizons": list(RANKING_HORIZONS),
        "curve_horizons": list(CURVE_HORIZONS),
        "frozen": {
            "source_hash": source_hash,
            "rollout_hash": rollout_hash,
            "sample_count": 4096,
            "candidate_count_per_horizon": 102660,
            "source_mode": "latest",
            "ridge": 1e-6,
            "readout": "same frozen MLP ensemble rollout",
            "postprocessing": "clip each MI at zero, then delta2 = joint - source_a - source_b",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    source_path = Path(args.source).expanduser().resolve()
    rollout_path = Path(args.rollout).expanduser().resolve()
    degree3_dir = Path(args.degree3_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    figure_path = Path(args.figure).expanduser().resolve()
    sources = np.load(source_path)
    rollout = np.load(rollout_path, mmap_mode="r")
    if sources.shape != (4096, 60) or rollout.shape != (4096, 60, 60):
        raise ValueError(f"Unexpected controlled input shapes: {sources.shape}, {rollout.shape}")
    source_hash = fingerprint_array(sources)
    rollout_hash = fingerprint_array(rollout)
    _write_contract(output_dir, source_hash, rollout_hash)

    rankings: dict[tuple[int, int], pd.DataFrame] = {}
    summary_rows: list[dict[str, float | int]] = []
    for horizon in RANKING_HORIZONS:
        reference = _load_degree3_ranking(degree3_dir, horizon)
        rankings[(3, horizon)] = reference
        for degree in DEGREES:
            started = time.perf_counter()
            if degree == 3:
                frame = reference
                runtime = 0.0
            else:
                path = _ranking_path(output_dir, degree, horizon)
                if path.exists() and args.resume:
                    frame = _load_ranking(path)
                    runtime = 0.0
                else:
                    frame = _compute_ranking(
                        sources,
                        np.asarray(rollout[:, horizon - 1, :]),
                        degree=degree,
                        horizon=horizon,
                    )
                    runtime = time.perf_counter() - started
                    _save_ranking(
                        frame,
                        path,
                        degree=degree,
                        horizon=horizon,
                        source_hash=source_hash,
                        rollout_hash=rollout_hash,
                    )
                rankings[(degree, horizon)] = frame
            summary_rows.append(
                _rank_summary(
                    frame,
                    reference,
                    degree=degree,
                    horizon=horizon,
                    runtime_seconds=runtime,
                )
            )
            print(
                f"H={horizon:03d} degree={degree} "
                f"top1={frame.iloc[0]['delta2_tm']:.6f} "
                f"rho={summary_rows[-1]['spearman_vs_degree3']:.3f}",
                flush=True,
            )

    summary = pd.DataFrame(summary_rows).sort_values(["horizon", "degree"])
    curves = _curve_rows(sources, np.asarray(rollout))
    top = _top_rows(rankings)
    summary.to_csv(output_dir / "ranking_summary.csv", index=False)
    curves.to_csv(output_dir / "representative_edge_curves.csv", index=False)
    top.to_csv(output_dir / "top20_by_estimator.csv", index=False)
    plot_comparison(curves, summary, figure_path)

    manifest = {
        "source": str(source_path),
        "rollout": str(rollout_path),
        "source_hash": source_hash,
        "rollout_hash": rollout_hash,
        "degrees": list(DEGREES),
        "ranking_horizons": list(RANKING_HORIZONS),
        "curve_horizons": list(CURVE_HORIZONS),
        "outputs": {
            "ranking_summary": str(output_dir / "ranking_summary.csv"),
            "representative_edge_curves": str(output_dir / "representative_edge_curves.csv"),
            "top20": str(output_dir / "top20_by_estimator.csv"),
            "figure_png": str(figure_path.with_suffix(".png")),
            "figure_svg": str(figure_path.with_suffix(".svg")),
            "figure_pdf": str(figure_path.with_suffix(".pdf")),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--rollout", default=str(DEFAULT_ROLLOUT))
    parser.add_argument("--degree3-dir", default=str(DEFAULT_DEGREE3))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--figure", default=str(DEFAULT_FIGURE))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2), flush=True)


if __name__ == "__main__":
    main()
