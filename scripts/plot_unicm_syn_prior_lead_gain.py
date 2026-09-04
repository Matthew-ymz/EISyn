#!/usr/bin/env python3
"""Plot the lead-wise forecast gain from the UniCM Syn regularization prior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_unicm_synergy_guided_forecast import (
    cell_rmse,
    chronological_split,
    circular_block_indices,
    target_scaling,
)


DEFAULT_RESULTS = (
    ROOT / "results" / "unicm_synergy_regularized_forecast_extended_1980_2018"
)
DEFAULT_INPUT = ROOT / "data" / "ORAS5" / "modeformer_1980_2014"


def leadwise_gain(
    baseline: np.ndarray,
    treatment: np.ndarray,
    target: np.ndarray,
    target_scale: np.ndarray,
) -> np.ndarray:
    """Return target-averaged nRMSE reduction for every prediction lead."""

    return (
        cell_rmse(baseline, target, target_scale)
        - cell_rmse(treatment, target, target_scale)
    ).mean(axis=0)


def bootstrap_leadwise_gain(
    baseline: np.ndarray,
    treatment: np.ndarray,
    target: np.ndarray,
    target_scale: np.ndarray,
    *,
    replicates: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((replicates, target.shape[2]), dtype=np.float64)
    for replicate in range(replicates):
        sample = circular_block_indices(rng, len(target), block_length)
        bootstrap[replicate] = leadwise_gain(
            baseline[sample], treatment[sample], target[sample], target_scale
        )
    return bootstrap


def plot_diagnostic(
    leads: np.ndarray,
    pair_syn: np.ndarray,
    gain: np.ndarray,
    gain_low: np.ndarray,
    gain_high: np.ndarray,
    output: Path,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    ink = "#273142"
    teal = "#287D76"
    orange = "#D9822B"
    negative = "#9AA7B5"
    grid = "#E7EBEF"

    fig, (ax_syn, ax_gain) = plt.subplots(
        2,
        1,
        figsize=(5.6, 4.1),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (0.72, 1.28)},
    )

    for axis in (ax_syn, ax_gain):
        axis.axvspan(6.5, 10.5, color="#E9F3F1", alpha=0.9, lw=0, zorder=0)
        axis.grid(axis="y", color=grid, linewidth=0.55, zorder=0)

    ax_syn.plot(
        leads,
        pair_syn,
        color=teal,
        linewidth=1.45,
        marker="o",
        markersize=3.0,
        markeredgecolor="white",
        markeredgewidth=0.4,
        zorder=3,
    )
    syn_peak = int(np.argmax(pair_syn))
    ax_syn.scatter(
        leads[syn_peak],
        pair_syn[syn_peak],
        s=29,
        color=teal,
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )
    ax_syn.annotate(
        f"Pair-Syn maximum: lead {leads[syn_peak]}",
        xy=(leads[syn_peak], pair_syn[syn_peak]),
        xytext=(4.2, pair_syn[syn_peak] * 1.055),
        color=teal,
        fontsize=6.5,
        arrowprops={"arrowstyle": "-", "color": teal, "lw": 0.6},
        ha="left",
        va="center",
    )
    ax_syn.text(
        8.5,
        0.96,
        "lead 7–10",
        transform=ax_syn.get_xaxis_transform(),
        ha="center",
        va="top",
        color="#55736F",
        fontsize=6.2,
    )
    ax_syn.set_ylabel("Mean pair Syn\nper target (bits)")
    ax_syn.tick_params(axis="x", which="both", bottom=False)

    ax_gain.axhline(0, color="#69737E", linewidth=0.75, linestyle="--", zorder=1)
    ax_gain.fill_between(
        leads,
        gain_low,
        gain_high,
        color=orange,
        alpha=0.17,
        linewidth=0,
        zorder=1,
    )
    ax_gain.plot(leads, gain, color=orange, linewidth=1.35, zorder=2)
    point_colors = np.where(gain >= 0, orange, negative)
    ax_gain.scatter(
        leads,
        gain,
        s=18,
        c=point_colors,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    gain_peak = int(np.argmax(gain))
    ax_gain.scatter(
        leads[gain_peak],
        gain[gain_peak],
        s=42,
        color=orange,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    ax_gain.annotate(
        f"Maximum gain: lead {leads[gain_peak]}",
        xy=(leads[gain_peak], gain[gain_peak]),
        xytext=(4.0, gain[gain_peak] * 0.91),
        color=ink,
        fontsize=6.5,
        arrowprops={"arrowstyle": "-", "color": "#66717D", "lw": 0.6},
        ha="left",
        va="center",
    )
    ax_gain.set(
        xlabel="Prediction lead (months)",
        ylabel="Normalized RMSE gain\nover uniform ridge",
        xticks=(1, 4, 8, 12, 16, 20, 24),
        xlim=(0.5, 24.5),
    )
    ax_gain.text(
        0.99,
        0.97,
        "positive = Syn prior is better\nline: point estimate; band: 95% block-bootstrap CI",
        transform=ax_gain.transAxes,
        ha="right",
        va="top",
        color="#66717D",
        fontsize=6.1,
    )

    for label, axis in zip("ab", (ax_syn, ax_gain)):
        axis.text(
            -0.12,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=8.5,
            fontweight="bold",
            color=ink,
            ha="left",
            va="bottom",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    summary = json.loads((args.results_dir / "summary.json").read_text())
    diagnostics = summary["syn_nonnegativity"]
    if int(diagnostics["significant_negative_count"]) != 0:
        raise ValueError("Cached Syn input contains a significant nonnegativity violation.")

    with np.load(args.results_dir / "evaluation_arrays.npz") as archive:
        centrality = archive["centrality"].astype(np.float64)
        target = archive["test_target"].astype(np.float64)
        uniform = archive["prediction_uniform"].astype(np.float64)
        syn_prior = archive["prediction_syn_regularized"].astype(np.float64)
    if np.any(centrality < 0):
        raise ValueError("Cached post-tolerance Syn centrality must be nonnegative.")

    with np.load(args.input_dir / "model_inputs.npz") as archive:
        full_target = archive["targets"].astype(np.float64)
    with np.load(args.input_dir / "modeformer_predictions.npz") as archive:
        target_dates = archive["target_dates"].astype(str)

    split_info = summary["chronological_split"]
    split = chronological_split(target_dates, **split_info)
    _, target_scale = target_scaling(full_target, split.fit)
    target_scale = target_scale.reshape(1, target.shape[1], 1)

    gain = leadwise_gain(uniform, syn_prior, target, target_scale)
    bootstrap = bootstrap_leadwise_gain(
        uniform,
        syn_prior,
        target,
        target_scale,
        replicates=args.bootstrap,
        block_length=args.bootstrap_block,
        seed=args.seed,
    )
    gain_low, gain_high = np.percentile(bootstrap, [2.5, 97.5], axis=0)

    # Each source-pair contribution enters two incident source centralities.
    pair_syn = centrality.sum(axis=2).mean(axis=0) / 2.0
    leads = np.arange(1, gain.size + 1)
    plot_diagnostic(leads, pair_syn, gain, gain_low, gain_high, args.output)

    window = np.arange(6, 10)
    outside = np.r_[0:6, 10:24]
    window_contrast = bootstrap[:, window].mean(axis=1) - bootstrap[:, outside].mean(axis=1)
    result = {
        "peak_gain_lead": int(leads[np.argmax(gain)]),
        "peak_gain": float(np.max(gain)),
        "lead_8_gain": float(gain[7]),
        "lead_8_ci95": [float(gain_low[7]), float(gain_high[7])],
        "lead_7_10_mean_gain": float(gain[window].mean()),
        "lead_7_10_ci95": np.percentile(
            bootstrap[:, window].mean(axis=1), [2.5, 97.5]
        ).tolist(),
        "lead_7_10_minus_other_leads": float(
            gain[window].mean() - gain[outside].mean()
        ),
        "lead_7_10_minus_other_leads_ci95": np.percentile(
            window_contrast, [2.5, 97.5]
        ).tolist(),
        "peak_pair_syn_lead": int(leads[np.argmax(pair_syn)]),
        "syn_numerical_zero_tolerance_bit": float(
            diagnostics["zero_tolerance_bit"]
        ),
        "syn_numerical_zero_count": int(diagnostics["numerical_zero_count"]),
        "output": str(args.output),
    }
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS / "syn_prior_gain_by_lead.png",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--bootstrap-block", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
