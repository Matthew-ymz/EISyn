#!/usr/bin/env python3
"""Evaluate adjacent-lead coefficient smoothing for UniCM calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


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
    feature_penalty,
    predict_generalized_ridge,
    prepare_designs,
)


DEFAULT_INPUT = ROOT / "data" / "ORAS5" / "modeformer_1980_2018_normfit_1980_2003"
DEFAULT_CALIBRATION = ROOT / "results" / "unicm_synergy_regularized_forecast_normfit_1980_2003"
DEFAULT_XI = (
    ROOT
    / "results"
    / "unicm_target_xi_shapley_prior_normfit_1980_2003_n16384"
    / "target_xi_shapley_centrality.npz"
)
DEFAULT_OUTPUT = ROOT / "results" / "unicm_target_xi_lead_smoothing_pilot"


def lead_smoothing_matrix(n_leads: int, n_features: int) -> sparse.csr_matrix:
    difference = sparse.diags(
        diagonals=(-np.ones(n_leads - 1), np.ones(n_leads - 1)),
        offsets=(0, 1),
        shape=(n_leads - 1, n_leads),
        format="csr",
    )
    return sparse.kron(
        difference.T @ difference,
        sparse.eye(n_features, format="csr"),
        format="csr",
    )


def predict_lead_smoothed_ridge(
    designs,
    centrality: np.ndarray,
    *,
    alpha: float,
    gamma: float,
    floor_fraction: float,
    smoothing_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jointly fit the 24 lead-specific coefficient vectors for each target."""

    n_targets, n_leads, _ = centrality.shape
    first = next(iter(designs.values()))
    n_features = first.x_fit.shape[1]
    n_validation = first.x_validation.shape[0]
    n_test = first.x_test.shape[0]
    validation = np.empty((n_validation, n_targets, n_leads), dtype=np.float64)
    test = np.empty((n_test, n_targets, n_leads), dtype=np.float64)
    coefficients = np.empty((n_targets, n_leads, n_features), dtype=np.float64)
    smooth = lead_smoothing_matrix(n_leads, n_features)

    for target_index in range(n_targets):
        blocks = []
        rhs = []
        for lead in range(n_leads):
            design = designs[(target_index, lead)]
            penalty = feature_penalty(
                centrality[target_index, lead],
                gamma,
                floor_fraction=floor_fraction,
            )
            block = sparse.csr_matrix(design.gram) + sparse.diags(
                float(alpha) * penalty,
                format="csr",
            )
            blocks.append(block)
            rhs.append(design.rhs)
        system = sparse.block_diag(blocks, format="csr")
        if smoothing_lambda > 0:
            system = system + float(smoothing_lambda) * smooth
        solution = spsolve(system, np.concatenate(rhs)).reshape(n_leads, n_features)
        if not np.isfinite(solution).all():
            raise ValueError("Lead-smoothed ridge produced non-finite coefficients.")
        coefficients[target_index] = solution
        for lead in range(n_leads):
            design = designs[(target_index, lead)]
            validation[:, target_index, lead] = (
                design.x_validation @ solution[lead] + design.y_mean
            )
            test[:, target_index, lead] = design.x_test @ solution[lead] + design.y_mean
    return validation, test, coefficients


def tune_lambda(
    designs,
    centrality: np.ndarray,
    validation_target: np.ndarray,
    target_scale: np.ndarray,
    *,
    alpha: float,
    gamma: float,
    floor_fraction: float,
    lambdas: list[float],
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    records: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray]] = []
    scores: dict[str, float] = {}
    for value in lambdas:
        validation, test, coefficients = predict_lead_smoothed_ridge(
            designs,
            centrality,
            alpha=alpha,
            gamma=gamma,
            floor_fraction=floor_fraction,
            smoothing_lambda=value,
        )
        score = mean_cell_nrmse(validation, validation_target, target_scale)
        scores[f"lambda={value:g}"] = float(score)
        records.append((float(score), value, validation, test, coefficients))
    best = min(records, key=lambda item: (item[0], item[1]))
    return best[1], scores, best[2], best[3], best[4]


def bootstrap_lead_gain(
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
    values = np.empty((replicates, target.shape[2]), dtype=np.float64)
    for replicate in range(replicates):
        sample = circular_block_indices(rng, len(target), block_length)
        values[replicate] = (
            cell_rmse(baseline[sample], target[sample], target_scale)
            - cell_rmse(treatment[sample], target[sample], target_scale)
        ).mean(axis=0)
    return values


def plot_results(result: dict[str, object], output: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    colors = {"uniform": "#6F88A8", "xi": "#287D76"}
    neutral = "#7A828A"
    accent = "#C76D3A"
    grid = "#E5E9ED"
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    ax_curve, ax_score, ax_smooth, ax_prior = axes.ravel()

    for key, label in (("uniform", "Uniform"), ("xi", r"$\Xi$-Shapley")):
        scores = result["validation_scores"][key]
        xs = np.asarray([float(name.split("=")[1]) for name in scores])
        ys = np.asarray(list(scores.values()), dtype=float)
        ax_curve.plot(
            xs + 1.0,
            ys,
            marker="o",
            ms=3.2,
            lw=1.2,
            color=colors[key],
            label=label,
        )
        chosen = float(result["selected_lambda"][key])
        chosen_y = scores[f"lambda={chosen:g}"]
        ax_curve.scatter(chosen + 1.0, chosen_y, s=28, color=colors[key], edgecolor="white", zorder=4)
    ax_curve.set_xscale("log")
    ax_curve.set(
        xlabel=r"Lead-smoothing strength $\lambda$ (+1 for log scale)",
        ylabel="Validation mean nRMSE",
    )
    ax_curve.grid(axis="y", color=grid, lw=0.55)
    ax_curve.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2)

    method_order = ["independent_uniform", "independent_xi", "smoothed_uniform", "smoothed_xi"]
    labels = ["Independent\nuniform", "Independent\n$\\Xi$", "Smoothed\nuniform", "Smoothed\n$\\Xi$"]
    values = [float(result["test_nrmse"][key]) for key in method_order]
    point_colors = [neutral, colors["xi"], colors["uniform"], accent]
    x = np.arange(4)
    ax_score.scatter(x, values, s=35, color=point_colors, edgecolor="white", linewidth=0.5, zorder=3)
    for xpos, value in zip(x, values):
        ax_score.text(xpos, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=6.3)
    ax_score.set(ylabel="Test mean nRMSE", xticks=x, xticklabels=labels)
    ax_score.grid(axis="y", color=grid, lw=0.55)

    leads = np.arange(1, 25)
    comparisons = (
        (ax_smooth, "smoothed_xi_over_independent_xi", "Gain from lead smoothing", accent),
        (ax_prior, "smoothed_xi_over_smoothed_uniform", r"$\Xi$ gain under matched smoothing", colors["xi"]),
    )
    for axis, key, ylabel, color in comparisons:
        point = np.asarray(result["lead_gain"][key], dtype=float)
        interval = np.asarray(result["lead_gain_ci95"][key], dtype=float)
        axis.axhline(0, color="#6D7782", lw=0.75, ls="--", zorder=1)
        axis.fill_between(leads, interval[0], interval[1], color=color, alpha=0.14, linewidth=0)
        axis.plot(leads, point, color=color, lw=1.25, marker="o", ms=2.7, markeredgecolor="white", markeredgewidth=0.35)
        axis.set(
            xlabel="Prediction lead (months)",
            ylabel=ylabel,
            xticks=(1, 4, 8, 12, 16, 20, 24),
            xlim=(0.5, 24.5),
        )
        axis.grid(axis="y", color=grid, lw=0.55)

    for label, axis in zip("abcd", axes.ravel()):
        axis.text(-0.16, 1.06, label, transform=axis.transAxes, fontsize=8.5, fontweight="bold", va="bottom")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration = json.loads((args.calibration_dir / "summary.json").read_text(encoding="utf-8"))
    with np.load(args.input_dir / "model_inputs.npz", allow_pickle=False) as archive:
        history = archive["history"].astype(np.float64)
        target = archive["targets"].astype(np.float64)
        mode_names = archive["mode_names"].astype(str).tolist()
        metadata = json.loads(str(archive["metadata"]))
    with np.load(args.input_dir / "modeformer_predictions.npz", allow_pickle=False) as archive:
        ensemble = archive["ensemble_prediction"].astype(np.float64)
        target_dates = archive["target_dates"].astype(str)
    with np.load(args.xi_cache, allow_pickle=False) as archive:
        centrality = archive["centrality"].astype(np.float64)
        xi_config = json.loads(str(archive["config"]))
        xi_audit = json.loads(str(archive["audit"]))

    split = chronological_split(target_dates, **calibration["chronological_split"])
    _, target_scale = target_scaling(target, split.fit)
    target_scale = target_scale.reshape(1, len(mode_names), 1)
    additive_history, _ = history_summaries(history)
    designs = prepare_designs(ensemble, target, additive_history, split)
    lambdas = [float(value) for value in args.lambdas.split(",")]
    if args.smoke:
        lambdas = [0.0, 100.0, 10000.0]

    settings = {
        "uniform": {"alpha": args.uniform_alpha, "gamma": 0.0},
        "xi": {"alpha": args.xi_alpha, "gamma": args.xi_gamma},
    }
    selected_lambda = {}
    validation_scores = {}
    predictions = {}
    coefficients = {}
    for key, setting in settings.items():
        chosen, scores, _, test_prediction, coefficient = tune_lambda(
            designs,
            centrality,
            target[split.validation],
            target_scale,
            alpha=setting["alpha"],
            gamma=setting["gamma"],
            floor_fraction=args.floor_fraction,
            lambdas=lambdas,
        )
        selected_lambda[key] = chosen
        validation_scores[key] = scores
        predictions[f"smoothed_{key}"] = test_prediction
        coefficients[f"smoothed_{key}"] = coefficient

    _, independent_uniform = predict_generalized_ridge(
        designs,
        centrality,
        alpha=args.uniform_alpha,
        gamma=0.0,
        floor_fraction=args.floor_fraction,
    )
    _, independent_xi = predict_generalized_ridge(
        designs,
        centrality,
        alpha=args.xi_alpha,
        gamma=args.xi_gamma,
        floor_fraction=args.floor_fraction,
    )
    predictions["independent_uniform"] = independent_uniform
    predictions["independent_xi"] = independent_xi

    zero_uniform = predict_lead_smoothed_ridge(
        designs,
        centrality,
        alpha=args.uniform_alpha,
        gamma=0.0,
        floor_fraction=args.floor_fraction,
        smoothing_lambda=0.0,
    )[1]
    zero_xi = predict_lead_smoothed_ridge(
        designs,
        centrality,
        alpha=args.xi_alpha,
        gamma=args.xi_gamma,
        floor_fraction=args.floor_fraction,
        smoothing_lambda=0.0,
    )[1]
    equivalence = {
        "uniform_max_abs_prediction_difference": float(np.max(np.abs(zero_uniform - independent_uniform))),
        "xi_max_abs_prediction_difference": float(np.max(np.abs(zero_xi - independent_xi))),
    }
    if max(equivalence.values()) > 1e-9:
        raise ValueError(f"lambda=0 equivalence check failed: {equivalence}")

    test_target = target[split.test]
    test_nrmse = {
        key: mean_cell_nrmse(value, test_target, target_scale)
        for key, value in predictions.items()
    }
    comparisons = {
        "smoothed_xi_over_independent_xi": (independent_xi, predictions["smoothed_xi"]),
        "smoothed_xi_over_smoothed_uniform": (predictions["smoothed_uniform"], predictions["smoothed_xi"]),
        "smoothed_uniform_over_independent_uniform": (independent_uniform, predictions["smoothed_uniform"]),
    }
    aggregate_gain = {}
    aggregate_gain_ci95 = {}
    lead_gain = {}
    lead_gain_ci95 = {}
    bootstrap_arrays = {}
    for index, (key, (baseline, treatment)) in enumerate(comparisons.items()):
        aggregate_gain[key] = test_nrmse[
            "independent_xi" if key == "smoothed_xi_over_independent_xi" else (
                "smoothed_uniform" if key == "smoothed_xi_over_smoothed_uniform" else "independent_uniform"
            )
        ] - test_nrmse[
            "smoothed_xi" if key != "smoothed_uniform_over_independent_uniform" else "smoothed_uniform"
        ]
        boot = bootstrap_gain(
            baseline,
            treatment,
            test_target,
            target_scale,
            args.bootstrap,
            args.bootstrap_block,
            args.seed + index,
        )
        lead_boot = bootstrap_lead_gain(
            baseline,
            treatment,
            test_target,
            target_scale,
            replicates=args.bootstrap,
            block_length=args.bootstrap_block,
            seed=args.seed + 100 + index,
        )
        aggregate_gain_ci95[key] = np.percentile(boot, [2.5, 97.5]).tolist()
        lead_gain[key] = (
            cell_rmse(baseline, test_target, target_scale)
            - cell_rmse(treatment, test_target, target_scale)
        ).mean(axis=0).tolist()
        lead_gain_ci95[key] = np.percentile(lead_boot, [2.5, 97.5], axis=0).tolist()
        bootstrap_arrays[f"bootstrap_{key}"] = boot

    result = {
        "status": "smoke" if args.smoke else "completed",
        "question": "What changes when only adjacent-lead coefficient smoothing is added to otherwise fixed UniCM generalized-ridge calibrators?",
        "design": "2x2 factorial: uniform versus Xi-Shapley prior, each with independent versus adjacent-lead-smoothed coefficients",
        "figure_contract": {
            "conclusion": "Determine whether validation-selected adjacent-lead smoothing lowers held-out nRMSE and preserves an Xi-Shapley advantage over a smoothing-matched uniform control.",
            "evidence": "Validation curves, four held-out nRMSE values, and paired lead-resolved gains with 12-month block-bootstrap intervals.",
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib",
            "output": "one 600-dpi PNG",
            "review_risks": ["test-informed lambda selection", "unmatched smoothing budget", "legend overlap", "hidden lambda=0 mismatch"],
        },
        "input_preprocessing": {
            "period": metadata["period"],
            "normalization_fit_period": metadata["normalization_fit_period"],
        },
        "samples": {"fit": len(split.fit), "validation": len(split.validation), "test": len(split.test)},
        "chronological_split": calibration["chronological_split"],
        "xi_intervention_config": xi_config,
        "xi_audit": xi_audit,
        "fixed_hyperparameters": {
            "uniform_alpha": args.uniform_alpha,
            "xi_alpha": args.xi_alpha,
            "xi_gamma": args.xi_gamma,
            "floor_fraction": args.floor_fraction,
        },
        "lambda_grid": lambdas,
        "selected_lambda": selected_lambda,
        "validation_scores": validation_scores,
        "test_nrmse": test_nrmse,
        "aggregate_gain": aggregate_gain,
        "aggregate_gain_ci95": aggregate_gain_ci95,
        "lead_gain": lead_gain,
        "lead_gain_ci95": lead_gain_ci95,
        "lambda_zero_equivalence": equivalence,
        "controls": {
            "same_data_and_split": True,
            "same_features_and_parameter_count": True,
            "same_target_scale_and_metric": True,
            "same_xi_centrality_cache": True,
            "lambda_selected_on_validation_only": True,
            "test_used_for_selection": False,
        },
    }
    summary_path = args.output_dir / ("smoke_summary.json" if args.smoke else "summary.json")
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.smoke:
        np.savez_compressed(
            args.output_dir / "evaluation_arrays.npz",
            target=test_target.astype(np.float32),
            target_scale=target_scale.astype(np.float64),
            **{f"prediction_{key}": value.astype(np.float32) for key, value in predictions.items()},
            **{f"coefficient_{key}": value.astype(np.float64) for key, value in coefficients.items()},
            **bootstrap_arrays,
        )
        plot_results(result, args.output_dir / "lead_smoothing_comparison.png")
    print(json.dumps({
        "status": result["status"],
        "selected_lambda": selected_lambda,
        "test_nrmse": test_nrmse,
        "aggregate_gain": aggregate_gain,
        "aggregate_gain_ci95": aggregate_gain_ci95,
        "lambda_zero_equivalence": equivalence,
    }, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--xi-cache", type=Path, default=DEFAULT_XI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--uniform-alpha", type=float, default=1000.0)
    parser.add_argument("--xi-alpha", type=float, default=30000.0)
    parser.add_argument("--xi-gamma", type=float, default=3.0)
    parser.add_argument("--floor-fraction", type=float, default=0.05)
    parser.add_argument(
        "--lambdas",
        default="0,1,3,10,30,100,300,1000,3000,10000,30000,100000,300000,1000000",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--bootstrap-block", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--smoke", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
