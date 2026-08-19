#!/usr/bin/env python3
"""Use target-resolved Syn as a generalized-ridge prior for Modeformer calibration."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_unicm_synergy_guided_forecast import (
    bootstrap_gain,
    cell_rmse,
    chronological_split,
    history_summaries,
    mean_cell_acc,
    mean_cell_nrmse,
    method_metrics,
    standardize_fit,
    target_scaling,
)


DEFAULT_INPUT = ROOT / "data" / "ORAS5" / "modeformer_1980_2014"
DEFAULT_SYN = (
    ROOT
    / "results"
    / "unicm_target_pair_syn_tm_degree1_signed_n8192"
    / "target_pair_syn_summary.csv"
)
DEFAULT_OUTPUT = ROOT / "results" / "unicm_synergy_regularized_forecast"
DEFAULT_SYN_ZERO_TOLERANCE = 2e-3


@dataclass
class CellDesign:
    x_fit: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_fit: np.ndarray
    y_mean: float
    gram: np.ndarray
    rhs: np.ndarray


def normalize_name(name: object) -> str:
    return "ENSO" if str(name) == "nino" else str(name)


def load_syn_centrality(
    path: Path,
    mode_names: list[str],
    *,
    zero_tolerance: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0:
        raise ValueError("Syn zero tolerance must be finite and nonnegative.")
    frame = pd.read_csv(path)
    for column in ("target", "left_source", "right_source"):
        frame[column] = frame[column].map(normalize_name)
    value_column = "mean" if "mean" in frame.columns else "syn"
    values = frame[value_column].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Syn input contains non-finite values.")
    significant_negative = values < -zero_tolerance
    if np.any(significant_negative):
        raise ValueError(
            "Syn is nonnegative by definition, but the input contains "
            f"{int(np.count_nonzero(significant_negative))} estimates below "
            f"the declared numerical tolerance -{zero_tolerance:.6g} bit "
            f"(minimum {float(values.min()):.6g} bit)."
        )
    numerical_negative = (values < 0) & ~significant_negative
    adjusted_values = values.copy()
    adjusted_values[numerical_negative] = 0.0
    frame[value_column] = adjusted_values
    diagnostics: dict[str, float | int] = {
        "zero_tolerance_bit": float(zero_tolerance),
        "raw_negative_count": int(np.count_nonzero(values < 0)),
        "numerical_zero_count": int(np.count_nonzero(numerical_negative)),
        "significant_negative_count": 0,
        "minimum_raw_syn_bit": float(values.min()),
    }
    if diagnostics["numerical_zero_count"]:
        print(
            "Syn nonnegativity check: treated "
            f"{diagnostics['numerical_zero_count']} estimates in "
            f"[-{zero_tolerance:.6g}, 0) bit as numerical zero; "
            f"minimum={diagnostics['minimum_raw_syn_bit']:.6g} bit.",
            flush=True,
        )
    lookup = {name: index for index, name in enumerate(mode_names)}
    centrality = np.zeros((len(mode_names), 24, len(mode_names)), dtype=np.float64)
    for row in frame.itertuples():
        target = lookup[row.target]
        lead = int(row.lead) - 1
        value = float(getattr(row, value_column))
        centrality[target, lead, lookup[row.left_source]] += value
        centrality[target, lead, lookup[row.right_source]] += value
    if np.any(centrality.sum(axis=2) <= 0):
        raise ValueError("At least one target–lead cell has zero total Syn centrality.")
    return centrality, diagnostics


def prepare_designs(
    base: np.ndarray,
    target: np.ndarray,
    additive_history: np.ndarray,
    split,
) -> dict[tuple[int, int], CellDesign]:
    designs: dict[tuple[int, int], CellDesign] = {}
    for target_index in range(target.shape[1]):
        for lead in range(24):
            x = np.concatenate((base[:, :, lead], additive_history), axis=1)
            x_fit, x_validation, x_test = standardize_fit(
                x[split.fit], x[split.validation], x[split.test]
            )
            y_fit = target[split.fit, target_index, lead]
            designs[(target_index, lead)] = CellDesign(
                x_fit=x_fit,
                x_validation=x_validation,
                x_test=x_test,
                y_fit=y_fit - y_fit.mean(),
                y_mean=float(y_fit.mean()),
                gram=x_fit.T @ x_fit,
                rhs=x_fit.T @ (y_fit - y_fit.mean()),
            )
    return designs


def feature_penalty(
    centrality: np.ndarray,
    gamma: float,
    *,
    floor_fraction: float,
) -> np.ndarray:
    syn_centrality = np.asarray(centrality, dtype=np.float64)
    if np.any(syn_centrality < 0):
        raise ValueError(
            "Syn centrality must be nonnegative; negative values must be fixed upstream."
        )
    floor = max(float(syn_centrality.mean()) * float(floor_fraction), 1e-12)
    relative = (syn_centrality + floor) / (syn_centrality.mean() + floor)
    per_mode = relative ** (-float(gamma))
    # Feature order: 11 contemporaneous mode forecasts, followed by final,
    # mean, and trend summaries for all 11 histories.
    penalty = np.tile(per_mode, 4)
    return penalty / penalty.mean()


def predict_generalized_ridge(
    designs: dict[tuple[int, int], CellDesign],
    centrality: np.ndarray,
    *,
    alpha: float,
    gamma: float,
    floor_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_targets = centrality.shape[0]
    n_validation = next(iter(designs.values())).x_validation.shape[0]
    n_test = next(iter(designs.values())).x_test.shape[0]
    validation = np.empty((n_validation, n_targets, 24), dtype=np.float64)
    test = np.empty((n_test, n_targets, 24), dtype=np.float64)
    for lead in range(24):
        lead_designs = [
            designs[(target_index, lead)] for target_index in range(n_targets)
        ]
        gram = np.stack([design.gram for design in lead_designs]).copy()
        penalty = np.stack(
            [
                feature_penalty(
                    centrality[target_index, lead],
                    gamma,
                    floor_fraction=floor_fraction,
                )
                for target_index in range(n_targets)
            ]
        )
        diagonal = np.arange(gram.shape[-1])
        gram[:, diagonal, diagonal] += float(alpha) * penalty
        coefficient = np.linalg.solve(
            gram,
            np.stack([design.rhs for design in lead_designs]),
        )
        for target_index, design in enumerate(lead_designs):
            validation[:, target_index, lead] = (
                design.x_validation @ coefficient[target_index] + design.y_mean
            )
            test[:, target_index, lead] = (
                design.x_test @ coefficient[target_index] + design.y_mean
            )
    return validation, test


def tune_prior(
    designs: dict[tuple[int, int], CellDesign],
    centrality: np.ndarray,
    validation_target: np.ndarray,
    target_scale: np.ndarray,
    *,
    alphas: list[float],
    gammas: list[float],
    floor_fraction: float,
) -> tuple[float, float, dict[str, float], np.ndarray, np.ndarray]:
    scores: dict[str, float] = {}
    best: tuple[float, float, float, np.ndarray, np.ndarray] | None = None
    for alpha in alphas:
        for gamma in gammas:
            validation, test = predict_generalized_ridge(
                designs,
                centrality,
                alpha=alpha,
                gamma=gamma,
                floor_fraction=floor_fraction,
            )
            score = mean_cell_nrmse(validation, validation_target, target_scale)
            scores[f"alpha={alpha:g},gamma={gamma:g}"] = score
            if best is None or score < best[0]:
                best = (score, alpha, gamma, validation, test)
    assert best is not None
    return best[1], best[2], scores, best[3], best[4]


def shuffled_centrality(
    centrality: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = np.empty_like(centrality)
    for target_index in range(centrality.shape[0]):
        for lead in range(centrality.shape[1]):
            shuffled[target_index, lead] = centrality[
                target_index, lead, rng.permutation(centrality.shape[2])
            ]
    return shuffled


def plot_results(
    metrics: dict[str, dict[str, object]],
    mode_names: list[str],
    random_scores: np.ndarray,
    gain_cells: np.ndarray,
    output_dir: Path,
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
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    methods = ["frozen", "univariate", "uniform", "syn_regularized"]
    labels = ["Frozen", "Univariate", "Uniform ridge", "Syn-regularized"]
    colors = ["#7A828A", "#8DA0CB", "#4C78A8", "#D9822B"]
    values = [float(metrics[name]["mean_cell_nrmse"]) for name in methods]
    syn_score = float(metrics["syn_regularized"]["mean_cell_nrmse"])

    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(0.9, 1.25))
    ax_methods = fig.add_subplot(grid[0, 0])
    ax_targets = fig.add_subplot(grid[0, 1])
    ax_null = fig.add_subplot(grid[1, 0])
    ax_heatmap = fig.add_subplot(grid[1, 1])

    x = np.arange(len(methods))
    ax_methods.scatter(x, values, c=colors, s=30, zorder=3)
    ax_methods.set(
        ylabel="Test mean normalized RMSE",
        xticks=x,
        xticklabels=labels,
    )
    ax_methods.tick_params(axis="x", rotation=30)

    target_gain = np.asarray(metrics["uniform"]["target_nrmse"]) - np.asarray(
        metrics["syn_regularized"]["target_nrmse"]
    )
    order = np.argsort(target_gain)
    ax_targets.axvline(0, color="0.6", lw=0.8)
    ax_targets.scatter(
        target_gain[order],
        np.arange(len(mode_names)),
        color="#D9822B",
        s=23,
    )
    ax_targets.set(
        xlabel="Syn-prior gain over uniform ridge",
        yticks=np.arange(len(mode_names)),
        yticklabels=np.asarray(mode_names)[order],
    )

    null_gain = float(metrics["uniform"]["mean_cell_nrmse"]) - random_scores
    observed_gain = (
        float(metrics["uniform"]["mean_cell_nrmse"]) - syn_score
    )
    ax_null.hist(
        null_gain,
        bins=max(8, min(14, len(null_gain) // 2)),
        color="#B8C4D0",
        edgecolor="white",
        linewidth=0.4,
    )
    ax_null.axvline(0, color="0.35", lw=0.8, ls="--")
    ax_null.axvline(
        observed_gain,
        color="#D9822B",
        lw=1.7,
        label="Syn prior",
    )
    ax_null.set(
        xlabel="Gain over uniform ridge",
        ylabel="Shuffled Syn priors",
    )
    ax_null.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        frameon=False,
    )

    limit = max(float(np.max(np.abs(gain_cells))), 1e-4)
    image = ax_heatmap.imshow(
        gain_cells,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    ax_heatmap.set(
        xlabel="Prediction lead (months)",
        ylabel="Target mode",
        yticks=np.arange(len(mode_names)),
        yticklabels=mode_names,
    )
    ax_heatmap.set_xticks([0, 5, 11, 17, 23], [1, 6, 12, 18, 24])
    colorbar = fig.colorbar(image, ax=ax_heatmap, fraction=0.045, pad=0.03)
    colorbar.set_label("Normalized RMSE gain")

    for label, axis in zip("abcd", (ax_methods, ax_targets, ax_null, ax_heatmap)):
        axis.text(
            -0.16,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 400})):
        fig.savefig(
            output_dir / f"synergy_regularized_forecast.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    with np.load(args.input_dir / "model_inputs.npz", allow_pickle=False) as data:
        history = data["history"].astype(np.float64)
        target = data["targets"].astype(np.float64)
        mode_names = data["mode_names"].astype(str).tolist()
    with np.load(
        args.input_dir / "modeformer_predictions.npz", allow_pickle=False
    ) as data:
        predictions_by_seed = data["predictions_by_seed"].astype(np.float64)
        ensemble = data["ensemble_prediction"].astype(np.float64)
        target_dates = data["target_dates"].astype(str)

    split = chronological_split(
        target_dates,
        fit_end=args.fit_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    _, target_scale = target_scaling(target, split.fit)
    target_scale = target_scale.reshape(1, len(mode_names), 1)
    additive_history, _ = history_summaries(history)
    centrality, syn_nonnegativity = load_syn_centrality(
        args.target_pair_syn,
        mode_names,
        zero_tolerance=args.syn_zero_tolerance,
    )
    designs = prepare_designs(ensemble, target, additive_history, split)
    alphas = [float(value) for value in args.alphas.split(",")]
    gammas = [float(value) for value in args.gammas.split(",")]

    alpha, gamma, tuning, validation_syn, test_syn = tune_prior(
        designs,
        centrality,
        target[split.validation],
        target_scale,
        alphas=alphas,
        gammas=gammas,
        floor_fraction=args.floor_fraction,
    )
    print(
        f"Syn prior: alpha={alpha:g}, gamma={gamma:g}, "
        f"validation_nRMSE={tuning[f'alpha={alpha:g},gamma={gamma:g}']:.6f}",
        flush=True,
    )
    uniform_alpha, _, uniform_tuning, _, test_uniform = tune_prior(
        designs,
        centrality,
        target[split.validation],
        target_scale,
        alphas=alphas,
        gammas=[0.0],
        floor_fraction=args.floor_fraction,
    )
    print(
        f"Uniform prior: alpha={uniform_alpha:g}, "
        f"validation_nRMSE={uniform_tuning[f'alpha={uniform_alpha:g},gamma=0']:.6f}",
        flush=True,
    )

    # Univariate calibration is retained as a strong non-synergy baseline.
    empty = {
        lead: np.empty((len(history), 0), dtype=np.float64) for lead in range(24)
    }
    from scripts.run_unicm_synergy_guided_forecast import fit_strategy, select_alpha

    univariate_alphas = [
        float(value) for value in args.univariate_alphas.split(",")
    ]
    univariate_alpha, univariate_tuning = select_alpha(
        ensemble,
        target,
        additive_history,
        empty,
        split,
        target_scale,
        univariate_alphas,
        "univariate",
    )
    _, test_univariate = fit_strategy(
        ensemble,
        target,
        additive_history,
        empty,
        split,
        univariate_alpha,
        "univariate",
    )

    rng = np.random.default_rng(args.seed)
    random_scores = []
    random_hyperparameters = []
    from tqdm.auto import tqdm

    for repeat in tqdm(
        range(args.random_repeats),
        desc="shuffled-Syn null",
        unit="repeat",
        mininterval=1.0,
    ):
        null_centrality = shuffled_centrality(centrality, rng)
        null_alpha, null_gamma, _, _, null_test = tune_prior(
            designs,
            null_centrality,
            target[split.validation],
            target_scale,
            alphas=alphas,
            gammas=gammas,
            floor_fraction=args.floor_fraction,
        )
        random_scores.append(
            mean_cell_nrmse(null_test, target[split.test], target_scale)
        )
        random_hyperparameters.append(
            {"repeat": repeat, "alpha": null_alpha, "gamma": null_gamma}
        )
    random_scores_array = np.asarray(random_scores)

    test_target = target[split.test]
    predictions = {
        "frozen": ensemble[split.test],
        "univariate": test_univariate,
        "uniform": test_uniform,
        "syn_regularized": test_syn,
    }
    metrics = {
        name: method_metrics(prediction, test_target, target_scale)
        for name, prediction in predictions.items()
    }
    bootstrap_uniform = bootstrap_gain(
        test_uniform,
        test_syn,
        test_target,
        target_scale,
        args.bootstrap,
        args.bootstrap_block,
        args.seed + 100,
    )
    bootstrap_univariate = bootstrap_gain(
        test_univariate,
        test_syn,
        test_target,
        target_scale,
        args.bootstrap,
        args.bootstrap_block,
        args.seed + 101,
    )

    seed_gains = []
    for seed_index, seed_prediction in enumerate(predictions_by_seed):
        seed_design = prepare_designs(
            seed_prediction,
            target,
            additive_history,
            split,
        )
        _, seed_uniform = predict_generalized_ridge(
            seed_design,
            centrality,
            alpha=uniform_alpha,
            gamma=0.0,
            floor_fraction=args.floor_fraction,
        )
        _, seed_syn = predict_generalized_ridge(
            seed_design,
            centrality,
            alpha=alpha,
            gamma=gamma,
            floor_fraction=args.floor_fraction,
        )
        seed_gains.append(
            {
                "checkpoint_seed": seed_index + 1,
                "gain_over_uniform": mean_cell_nrmse(
                    seed_uniform, test_target, target_scale
                )
                - mean_cell_nrmse(seed_syn, test_target, target_scale),
            }
        )

    syn_score = float(metrics["syn_regularized"]["mean_cell_nrmse"])
    random_p = float(
        (1 + np.count_nonzero(random_scores_array <= syn_score))
        / (len(random_scores_array) + 1)
    )
    report = {
        "status": "completed",
        "question": "Can target-resolved source-pair Syn improve all-mode forecast calibration as a regularization prior?",
        "mode_names": mode_names,
        "samples": {
            "fit": len(split.fit),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "chronological_split": {
            "fit_end": args.fit_end,
            "validation_start": args.validation_start,
            "validation_end": args.validation_end,
            "test_start": args.test_start,
            "test_end": args.test_end,
        },
        "selected_hyperparameters": {
            "alpha": alpha,
            "gamma": gamma,
            "floor_fraction": args.floor_fraction,
            "uniform_alpha": uniform_alpha,
            "univariate_alpha": univariate_alpha,
        },
        "syn_nonnegativity": syn_nonnegativity,
        "validation_scores": tuning,
        "uniform_validation_scores": uniform_tuning,
        "univariate_validation_scores": univariate_tuning,
        "test_metrics": metrics,
        "gain_over_uniform": {
            "observed": float(
                metrics["uniform"]["mean_cell_nrmse"]
                - metrics["syn_regularized"]["mean_cell_nrmse"]
            ),
            "bootstrap_ci95": np.percentile(
                bootstrap_uniform, [2.5, 97.5]
            ).tolist(),
            "bootstrap_positive_fraction": float(np.mean(bootstrap_uniform > 0)),
        },
        "gain_over_univariate": {
            "observed": float(
                metrics["univariate"]["mean_cell_nrmse"]
                - metrics["syn_regularized"]["mean_cell_nrmse"]
            ),
            "bootstrap_ci95": np.percentile(
                bootstrap_univariate, [2.5, 97.5]
            ).tolist(),
            "bootstrap_positive_fraction": float(
                np.mean(bootstrap_univariate > 0)
            ),
        },
        "shuffled_syn_control": {
            "repeats": args.random_repeats,
            "scores": random_scores,
            "mean": float(random_scores_array.mean()),
            "std": float(random_scores_array.std(ddof=1)),
            "fraction_null_at_least_as_good": random_p,
            "hyperparameters": random_hyperparameters,
            "null_preserves": "each target-lead Syn-centrality value distribution",
            "null_destroys": "assignment of Syn centrality to source-mode labels",
        },
        "checkpoint_seed_gains": seed_gains,
        "controls": {
            "same_features": True,
            "same_parameter_count": True,
            "same_fit_validation_test": True,
            "each_null_retuned_on_validation": True,
            "test_used_for_selection": False,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    gain_cells = cell_rmse(test_uniform, test_target, target_scale) - cell_rmse(
        test_syn, test_target, target_scale
    )
    np.savez_compressed(
        args.output_dir / "evaluation_arrays.npz",
        centrality=centrality,
        random_scores=random_scores_array,
        bootstrap_uniform_gain=bootstrap_uniform,
        bootstrap_univariate_gain=bootstrap_univariate,
        gain_cells=gain_cells,
        test_target=test_target.astype(np.float32),
        **{
            f"prediction_{name}": value.astype(np.float32)
            for name, value in predictions.items()
        },
    )
    plot_results(
        metrics,
        mode_names,
        random_scores_array,
        gain_cells,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "test_nrmse": {
                    name: value["mean_cell_nrmse"]
                    for name, value in metrics.items()
                },
                "gain_over_uniform": report["gain_over_uniform"],
                "gain_over_univariate": report["gain_over_univariate"],
                "shuffled_syn_p": random_p,
                "checkpoint_seed_gains": seed_gains,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--target-pair-syn", type=Path, default=DEFAULT_SYN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--alphas",
        default="100,300,1000,3000,10000,30000",
    )
    parser.add_argument("--gammas", default="0,0.5,1,2,3")
    parser.add_argument(
        "--univariate-alphas",
        default="0.01,0.1,1,10,100,1000",
    )
    parser.add_argument("--floor-fraction", type=float, default=0.05)
    parser.add_argument(
        "--syn-zero-tolerance",
        type=float,
        default=DEFAULT_SYN_ZERO_TOLERANCE,
        help=(
            "Estimated Syn values in [-tolerance, 0) bit are numerical zero; "
            "values below -tolerance abort the experiment."
        ),
    )
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--fit-end", default="2001-12")
    parser.add_argument("--validation-start", default="2004-01")
    parser.add_argument("--validation-end", default="2006-12")
    parser.add_argument("--test-start", default="2009-01")
    parser.add_argument("--test-end", default="2012-12")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-block", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
