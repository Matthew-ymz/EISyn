#!/usr/bin/env python3
"""Test a system-Xi Shapley regularization prior for UniCM calibration."""

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
    cell_rmse,
    chronological_split,
    circular_block_indices,
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
DEFAULT_SHAPLEY = ROOT / "results" / "unicm_11mode_shapley_affine" / "summary.json"


def load_system_xi_shapley(
    path: Path,
    mode_names: list[str],
    n_targets: int,
) -> tuple[np.ndarray, dict[str, object]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    audit = report["audit"]
    if int(audit["significant_nonnegativity_violation_count"]) != 0:
        raise ValueError("System-Xi Shapley cache has a nonnegativity violation.")
    if float(audit["maximum_absolute_closure_error_bits"]) > float(
        audit["closure_tolerance_bits"]
    ):
        raise ValueError("System-Xi Shapley cache fails its closure audit.")

    by_lead = {int(row["lead"]): row for row in report["lead_summary"]}
    expected = set(range(1, 25))
    if set(by_lead) != expected:
        raise ValueError("System-Xi Shapley cache must contain all 24 leads.")

    centrality = np.empty((n_targets, 24, len(mode_names)), dtype=np.float64)
    for lead in range(1, 25):
        values = np.asarray(
            [
                by_lead[lead]["shapley_bits_mean"][
                    "nino" if name == "ENSO" else name
                ]
                for name in mode_names
            ],
            dtype=np.float64,
        )
        if np.any(values < 0):
            raise ValueError("System-Xi Shapley centrality must be nonnegative.")
        centrality[:, lead - 1, :] = values
    return centrality, audit


def leadwise_gain(
    baseline: np.ndarray,
    treatment: np.ndarray,
    target: np.ndarray,
    target_scale: np.ndarray,
) -> np.ndarray:
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
    draws = np.empty((replicates, target.shape[2]), dtype=np.float64)
    for replicate in range(replicates):
        sample = circular_block_indices(rng, len(target), block_length)
        draws[replicate] = leadwise_gain(
            baseline[sample], treatment[sample], target[sample], target_scale
        )
    return draws


def plot_results(
    aggregate_gain: dict[str, float],
    aggregate_ci: dict[str, np.ndarray],
    lead_gain: dict[str, np.ndarray],
    lead_ci: dict[str, np.ndarray],
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
    labels = ["Target-pair Syn prior", r"System-$\Xi$ Shapley prior"]
    keys = ["pair_syn", "system_xi_shapley"]
    colors = {"pair_syn": "#D9822B", "system_xi_shapley": "#287D76"}
    ink = "#273142"
    grid_color = "#E7EBEF"

    fig, (ax_summary, ax_lead) = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (0.78, 1.62)},
    )

    y = np.arange(2)[::-1]
    ax_summary.axvline(0, color="#6D7782", lw=0.75, ls="--", zorder=1)
    for row, key in zip(y, keys):
        value = aggregate_gain[key]
        low, high = aggregate_ci[key]
        ax_summary.errorbar(
            value,
            row,
            xerr=np.asarray([[value - low], [high - value]]),
            fmt="o",
            markersize=5.3,
            color=colors[key],
            ecolor=colors[key],
            elinewidth=1.15,
            capsize=2.3,
            markeredgecolor="white",
            markeredgewidth=0.55,
            zorder=3,
        )
        ax_summary.text(
            high + 0.002,
            row,
            f"{value:+.3f}",
            ha="left",
            va="center",
            color=ink,
            fontsize=6.4,
        )
    ax_summary.set(
        xlabel="Normalized RMSE gain\nover uniform ridge",
        yticks=y,
        yticklabels=labels,
        ylim=(-0.6, 1.6),
    )
    ax_summary.grid(axis="x", color=grid_color, linewidth=0.55)

    leads = np.arange(1, 25)
    ax_lead.axhline(0, color="#6D7782", lw=0.75, ls="--", zorder=1)
    ax_lead.axvspan(6.5, 10.5, color="#E9F3F1", alpha=0.85, lw=0, zorder=0)
    for key, label in zip(keys, labels):
        low, high = lead_ci[key]
        ax_lead.fill_between(
            leads,
            low,
            high,
            color=colors[key],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax_lead.plot(
            leads,
            lead_gain[key],
            color=colors[key],
            linewidth=1.3,
            marker="o",
            markersize=2.7,
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=label,
            zorder=3,
        )
    ax_lead.text(
        8.5,
        0.96,
        "lead 7–10",
        transform=ax_lead.get_xaxis_transform(),
        ha="center",
        va="top",
        color="#55736F",
        fontsize=6.2,
    )
    ax_lead.set(
        xlabel="Prediction lead (months)",
        ylabel="Normalized RMSE gain over uniform ridge",
        xticks=(1, 4, 8, 12, 16, 20, 24),
        xlim=(0.5, 24.5),
    )
    ax_lead.grid(axis="y", color=grid_color, linewidth=0.55)
    ax_lead.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.4,
    )

    for label, axis in zip("ab", (ax_summary, ax_lead)):
        axis.text(
            -0.16,
            1.05,
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
    calibration = json.loads(
        (args.calibration_dir / "summary.json").read_text(encoding="utf-8")
    )
    with np.load(args.input_dir / "model_inputs.npz") as archive:
        history = archive["history"].astype(np.float64)
        target = archive["targets"].astype(np.float64)
        mode_names = archive["mode_names"].astype(str).tolist()
    with np.load(args.input_dir / "modeformer_predictions.npz") as archive:
        ensemble = archive["ensemble_prediction"].astype(np.float64)
        target_dates = archive["target_dates"].astype(str)

    split = chronological_split(target_dates, **calibration["chronological_split"])
    _, target_scale = target_scaling(target, split.fit)
    target_scale = target_scale.reshape(1, len(mode_names), 1)
    additive_history, _ = history_summaries(history)
    designs = prepare_designs(ensemble, target, additive_history, split)
    system_centrality, audit = load_system_xi_shapley(
        args.shapley_summary, mode_names, len(mode_names)
    )

    alphas = [float(value) for value in args.alphas.split(",")]
    gammas = [float(value) for value in args.gammas.split(",")]
    alpha, gamma, tuning, _, test_system = tune_prior(
        designs,
        system_centrality,
        target[split.validation],
        target_scale,
        alphas=alphas,
        gammas=gammas,
        floor_fraction=args.floor_fraction,
    )

    with np.load(args.calibration_dir / "evaluation_arrays.npz") as archive:
        cached_target = archive["test_target"].astype(np.float64)
        test_uniform = archive["prediction_uniform"].astype(np.float64)
        test_pair = archive["prediction_syn_regularized"].astype(np.float64)
    test_target = target[split.test]
    if not np.allclose(test_target, cached_target, rtol=0, atol=1e-6):
        raise ValueError("Cached and reconstructed test targets do not match.")

    predictions = {
        "pair_syn": test_pair,
        "system_xi_shapley": test_system,
    }
    aggregate_gain: dict[str, float] = {}
    aggregate_ci: dict[str, np.ndarray] = {}
    lead_gain: dict[str, np.ndarray] = {}
    lead_ci: dict[str, np.ndarray] = {}
    for index, (key, prediction) in enumerate(predictions.items()):
        aggregate_gain[key] = mean_cell_nrmse(
            test_uniform, test_target, target_scale
        ) - mean_cell_nrmse(prediction, test_target, target_scale)
        aggregate_bootstrap = bootstrap_gain(
            test_uniform,
            prediction,
            test_target,
            target_scale,
            args.bootstrap,
            args.bootstrap_block,
            args.seed + index,
        )
        aggregate_ci[key] = np.percentile(aggregate_bootstrap, [2.5, 97.5])
        lead_gain[key] = leadwise_gain(
            test_uniform, prediction, test_target, target_scale
        )
        lead_bootstrap = bootstrap_leadwise_gain(
            test_uniform,
            prediction,
            test_target,
            target_scale,
            replicates=args.bootstrap,
            block_length=args.bootstrap_block,
            seed=args.seed + 100 + index,
        )
        lead_ci[key] = np.percentile(lead_bootstrap, [2.5, 97.5], axis=0)

    plot_results(aggregate_gain, aggregate_ci, lead_gain, lead_ci, args.output)
    result = {
        "definition": "Exact Shapley allocation of system Xi across 11 source modes",
        "treatment_scope": "one joint-target source prior per lead, shared by all targets",
        "selected_hyperparameters": {"alpha": alpha, "gamma": gamma},
        "validation_nrmse": float(min(tuning.values())),
        "test_nrmse": {
            "uniform_ridge": mean_cell_nrmse(
                test_uniform, test_target, target_scale
            ),
            "target_pair_syn_prior": mean_cell_nrmse(
                test_pair, test_target, target_scale
            ),
            "system_xi_shapley_prior": mean_cell_nrmse(
                test_system, test_target, target_scale
            ),
        },
        "gain_over_uniform": {
            key: {
                "point": aggregate_gain[key],
                "ci95": aggregate_ci[key].tolist(),
            }
            for key in predictions
        },
        "system_xi_shapley_lead_8_gain": float(
            lead_gain["system_xi_shapley"][7]
        ),
        "system_xi_shapley_lead_7_10_mean_gain": float(
            lead_gain["system_xi_shapley"][6:10].mean()
        ),
        "shapley_audit": audit,
        "output": str(args.output),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--shapley-summary", type=Path, default=DEFAULT_SHAPLEY)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CALIBRATION / "system_xi_shapley_prior_comparison.png",
    )
    parser.add_argument("--alphas", default="100,300,1000,3000,10000,30000")
    parser.add_argument("--gammas", default="0,0.5,1,2,3")
    parser.add_argument("--floor-fraction", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--bootstrap-block", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
