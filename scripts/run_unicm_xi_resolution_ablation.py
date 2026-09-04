#!/usr/bin/env python3
"""Ablate target and lead resolution in the UniCM Xi-Shapley prior."""

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
    bootstrap_gain,
    chronological_split,
    history_summaries,
    mean_cell_nrmse,
    target_scaling,
)
from scripts.run_unicm_synergy_regularized_calibration import (
    prepare_designs,
    tune_prior,
)


DEFAULT_INPUT = ROOT / "data" / "ORAS5" / "modeformer_1980_2018"
DEFAULT_CALIBRATION = (
    ROOT / "results" / "unicm_synergy_regularized_forecast_extended_1980_2018"
)
DEFAULT_XI = ROOT / "results" / "unicm_target_xi_shapley_prior"
DEFAULT_OUTPUT = ROOT / "results" / "unicm_xi_resolution_ablation"


def broadcast_variants(centrality: np.ndarray) -> dict[str, np.ndarray]:
    target_count, lead_count, source_count = centrality.shape
    lead_only = np.broadcast_to(
        centrality.mean(axis=0, keepdims=True),
        (target_count, lead_count, source_count),
    ).copy()
    target_only = np.broadcast_to(
        centrality.mean(axis=1, keepdims=True),
        (target_count, lead_count, source_count),
    ).copy()
    global_prior = np.broadcast_to(
        centrality.mean(axis=(0, 1), keepdims=True),
        (target_count, lead_count, source_count),
    ).copy()
    return {
        "target_x_lead": centrality.copy(),
        "lead_only": lead_only,
        "target_only": target_only,
        "global": global_prior,
    }


def plot_results(results: dict[str, dict[str, object]], output: Path) -> None:
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
    keys = ("target_x_lead", "lead_only", "target_only", "global")
    labels = ("Target × lead", "Lead only", "Target only", "Global")
    colors = ("#287D76", "#4C78A8", "#D9822B", "#8B949E")
    y = np.arange(len(keys))[::-1]
    fig, ax = plt.subplots(figsize=(4.6, 2.45), constrained_layout=True)
    ax.axvline(0, color="#68727D", linewidth=0.75, linestyle="--", zorder=1)
    for row, key, color in zip(y, keys, colors, strict=True):
        gain = float(results[key]["gain_over_uniform"])
        low, high = (float(value) for value in results[key]["gain_ci95"])
        ax.errorbar(
            gain,
            row,
            xerr=np.asarray([[gain - low], [high - gain]]),
            fmt="o",
            markersize=5.5,
            color=color,
            ecolor=color,
            elinewidth=1.15,
            capsize=2.5,
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=3,
        )
        if np.isclose(gain, 0.0, atol=1e-12):
            gain_label = "0.000"
        elif abs(gain) < 0.001:
            gain_label = f"{gain:+.4f}"
        else:
            gain_label = f"{gain:+.3f}"
        ax.text(
            high + 0.0014,
            row,
            gain_label,
            va="center",
            ha="left",
            fontsize=6.2,
            color="#273142",
        )
    ax.set_yticks(y, labels)
    ax.set_xlabel("Normalized RMSE gain over uniform ridge")
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.grid(axis="x", color="#E7EBEF", linewidth=0.55)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    calibration = json.loads(
        (args.calibration_dir / "summary.json").read_text(encoding="utf-8")
    )
    with np.load(args.xi_dir / "evaluation_arrays.npz", allow_pickle=False) as archive:
        centrality = archive["centrality"].astype(np.float64)
    with np.load(args.input_dir / "model_inputs.npz", allow_pickle=False) as archive:
        history = archive["history"].astype(np.float64)
        target = archive["targets"].astype(np.float64)
    with np.load(
        args.input_dir / "modeformer_predictions.npz", allow_pickle=False
    ) as archive:
        ensemble = archive["ensemble_prediction"].astype(np.float64)
        target_dates = archive["target_dates"].astype(str)

    split = chronological_split(target_dates, **calibration["chronological_split"])
    _, target_scale = target_scaling(target, split.fit)
    target_scale = target_scale.reshape(1, target.shape[1], 1)
    additive_history, _ = history_summaries(history)
    designs = prepare_designs(ensemble, target, additive_history, split)
    alphas = [float(value) for value in args.alphas.split(",")]
    gammas = [float(value) for value in args.gammas.split(",")]

    _, _, _, _, uniform_prediction = tune_prior(
        designs,
        centrality,
        target[split.validation],
        target_scale,
        alphas=alphas,
        gammas=[0.0],
        floor_fraction=args.floor_fraction,
    )
    test_target = target[split.test]
    uniform_score = mean_cell_nrmse(uniform_prediction, test_target, target_scale)
    results: dict[str, dict[str, object]] = {}
    predictions: dict[str, np.ndarray] = {}
    for key, treatment in broadcast_variants(centrality).items():
        alpha, gamma, tuning, _, prediction = tune_prior(
            designs,
            treatment,
            target[split.validation],
            target_scale,
            alphas=alphas,
            gammas=gammas,
            floor_fraction=args.floor_fraction,
        )
        score = mean_cell_nrmse(prediction, test_target, target_scale)
        bootstrap = bootstrap_gain(
            uniform_prediction,
            prediction,
            test_target,
            target_scale,
            args.bootstrap,
            args.bootstrap_block,
            args.seed,
        )
        results[key] = {
            "alpha": alpha,
            "gamma": gamma,
            "validation_nrmse": float(min(tuning.values())),
            "test_nrmse": score,
            "gain_over_uniform": uniform_score - score,
            "gain_ci95": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            "bootstrap_positive_fraction": float(np.mean(bootstrap > 0)),
        }
        predictions[key] = prediction
        print(
            f"{key}: alpha={alpha:g}, gamma={gamma:g}, "
            f"test_nRMSE={score:.6f}, gain={uniform_score - score:+.6f}",
            flush=True,
        )

    full_over_lead = bootstrap_gain(
        predictions["lead_only"],
        predictions["target_x_lead"],
        test_target,
        target_scale,
        args.bootstrap,
        args.bootstrap_block,
        args.seed,
    )
    full_over_target = bootstrap_gain(
        predictions["target_only"],
        predictions["target_x_lead"],
        test_target,
        target_scale,
        args.bootstrap,
        args.bootstrap_block,
        args.seed,
    )
    report = {
        "status": "completed",
        "question": "Does Xi-Shapley prior improvement persist without target and/or lead resolution?",
        "treatment": "resolution of the same cached target-specific Xi-Shapley centrality tensor",
        "uniform_test_nrmse": uniform_score,
        "results": results,
        "target_x_lead_gain_over_lead_only": {
            "point": float(
                results["lead_only"]["test_nrmse"]
                - results["target_x_lead"]["test_nrmse"]
            ),
            "ci95": np.percentile(full_over_lead, [2.5, 97.5]).tolist(),
        },
        "target_x_lead_gain_over_target_only": {
            "point": float(
                results["target_only"]["test_nrmse"]
                - results["target_x_lead"]["test_nrmse"]
            ),
            "ci95": np.percentile(full_over_target, [2.5, 97.5]).tolist(),
        },
        "controls": {
            "same_cached_centrality_values_before_aggregation": True,
            "same_features_and_parameter_count": True,
            "same_split_and_metric": True,
            "each_treatment_retuned_on_validation": True,
            "test_used_for_selection": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output = args.output_dir / "xi_resolution_ablation.png"
    plot_results(results, output)
    print(json.dumps({"output": str(output)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--xi-dir", type=Path, default=DEFAULT_XI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alphas", default="100,300,1000,3000,10000,30000")
    parser.add_argument("--gammas", default="0,0.5,1,2,3")
    parser.add_argument("--floor-fraction", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--bootstrap-block", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260909)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
