#!/usr/bin/env python3
"""Compare fixed-month and season-matched PEID Syn priors for UniCM calibration."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_unicm_target_pair_syn import (  # noqa: E402
    gaussian_mi_scalar_targets,
    source_columns,
)
from scripts.run_unicm_synergy_guided_forecast import (  # noqa: E402
    bootstrap_gain,
    cell_rmse,
    chronological_split,
    fit_strategy,
    history_summaries,
    method_metrics,
    select_alpha,
    standardize_fit,
    target_scaling,
)
from scripts.run_unicm_synergy_regularized_calibration import (  # noqa: E402
    feature_penalty,
)
from scripts.unicm_peid_syn_analysis import (  # noqa: E402
    MODE_NAMES,
    sample_full_history_mode_inputs,
)


DEFAULT_INPUT = ROOT / "data" / "ORAS5" / "modeformer_1980_2018_normfit_1980_2003"
DEFAULT_CACHE = ROOT / "results" / "unicm_start_month_sweep" / "full_n8192" / "cache"
DEFAULT_OUTPUT = ROOT / "results" / "unicm_season_matched_syn_calibration"
DEFAULT_SYN_CACHE = DEFAULT_OUTPUT / "monthly_target_syn_centrality.npz"
DEFAULT_SYN_ZERO_TOLERANCE = 2e-3


@dataclass
class WeightedDesign:
    x_fit: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_fit: np.ndarray
    y_mean: float


def canonical_mode_name(name: str) -> str:
    return "ENSO" if str(name) == "nino" else str(name)


def first_target_months(target_dates: np.ndarray) -> np.ndarray:
    """Return zero-based calendar months corresponding to cache ``start_month``."""

    first = target_dates[:, 0].astype("datetime64[M]")
    return (first.astype(np.int64) % 12).astype(np.int64)


def prediction_cache_path(
    cache_dir: Path,
    *,
    seed: int,
    month: int,
    n_samples: int,
    sampling_seed: int,
    intervention_bound: float,
) -> Path:
    pattern = (
        f"overall_pred_seed{seed}_samples{n_samples}_sampling{sampling_seed}_"
        f"bound{intervention_bound:g}_fullhist12_start{month}_*.npz"
    )
    matches = sorted(cache_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one cache matching {cache_dir / pattern}; found {len(matches)}."
        )
    return matches[0]


def compute_monthly_centrality(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, object]]:
    history = sample_full_history_mode_inputs(
        n_samples=args.n_samples,
        intervention_bound=args.intervention_bound,
        seed=args.sampling_seed,
    ).astype(np.float64)
    source = history.transpose(0, 2, 1).reshape(args.n_samples, -1)
    source_centered = source - source.mean(axis=0, keepdims=True)
    denominator = args.n_samples - 1
    source_covariance = source_centered.T @ source_centered / denominator
    monthly = np.zeros((12, len(MODE_NAMES), 24, len(MODE_NAMES)), dtype=np.float64)
    numerical_zero_count = 0
    minimum_raw_syn = np.inf
    prediction_devices: set[str] = set()

    for month in range(12):
        predictions = []
        for seed in args.checkpoint_seeds:
            path = prediction_cache_path(
                args.prediction_cache_dir,
                seed=seed,
                month=month,
                n_samples=args.n_samples,
                sampling_seed=args.sampling_seed,
                intervention_bound=args.intervention_bound,
            )
            with np.load(path, allow_pickle=False) as payload:
                values = np.asarray(payload["all_mode_targets"], dtype=np.float64)
                cache_metadata = json.loads(str(payload["metadata"]))
            prediction_devices.add(str(cache_metadata.get("device", "unknown")))
            if (
                args.required_prediction_device
                and cache_metadata.get("device") != args.required_prediction_device
            ):
                raise RuntimeError(
                    f"Prediction cache device is {cache_metadata.get('device')!r}, but "
                    f"{args.required_prediction_device!r} is required: {path}"
                )
            if values.shape != (args.n_samples, 24, len(MODE_NAMES)):
                raise ValueError(f"Unexpected prediction shape {values.shape}: {path}")
            predictions.append(values.reshape(args.n_samples, -1))
        target = np.concatenate(predictions, axis=1)
        target_centered = target - target.mean(axis=0, keepdims=True)
        cross_covariance = source_centered.T @ target_centered / denominator
        target_variance = np.sum(target_centered**2, axis=0) / denominator

        singleton = np.empty((len(MODE_NAMES), target.shape[1]), dtype=np.float64)
        for mode in range(len(MODE_NAMES)):
            columns = source_columns(mode)
            singleton[mode] = gaussian_mi_scalar_targets(
                source_covariance[np.ix_(columns, columns)],
                cross_covariance[columns],
                target_variance,
                jitter=args.jitter,
            )

        by_seed = np.zeros(
            (len(args.checkpoint_seeds), len(MODE_NAMES), 24, len(MODE_NAMES)),
            dtype=np.float64,
        )
        for left in range(len(MODE_NAMES)):
            for right in range(left + 1, len(MODE_NAMES)):
                columns = np.concatenate((source_columns(left), source_columns(right)))
                joint = gaussian_mi_scalar_targets(
                    source_covariance[np.ix_(columns, columns)],
                    cross_covariance[columns],
                    target_variance,
                    jitter=args.jitter,
                )
                raw_syn = joint - singleton[left] - singleton[right]
                minimum_raw_syn = min(minimum_raw_syn, float(raw_syn.min()))
                violations = raw_syn < -args.syn_zero_tolerance
                if np.any(violations):
                    raise RuntimeError(
                        "Significant Syn nonnegativity violation: "
                        f"minimum={float(raw_syn.min()):.8g} bit, "
                        f"tolerance={args.syn_zero_tolerance:g} bit, "
                        f"count={int(np.count_nonzero(violations))}."
                    )
                numerical = (raw_syn < 0) & ~violations
                numerical_zero_count += int(np.count_nonzero(numerical))
                syn = raw_syn.copy()
                syn[numerical] = 0.0
                shaped = syn.reshape(len(args.checkpoint_seeds), 24, len(MODE_NAMES))
                shaped = shaped.transpose(0, 2, 1)
                by_seed[:, :, :, left] += shaped
                by_seed[:, :, :, right] += shaped
        monthly[month] = by_seed.mean(axis=0)
        print(f"Computed target-specific Syn centrality for month {month + 1}/12", flush=True)

    if np.any(monthly.sum(axis=-1) <= 0):
        raise RuntimeError("At least one month-target-lead cell has zero Syn centrality.")
    diagnostics: dict[str, object] = {
        "estimator": "signed affine degree-1 TM / Gaussian logdet",
        "n_samples": args.n_samples,
        "sampling_seed": args.sampling_seed,
        "intervention_bound": args.intervention_bound,
        "checkpoint_seeds": list(args.checkpoint_seeds),
        "jitter": args.jitter,
        "syn_zero_tolerance_bit": args.syn_zero_tolerance,
        "numerical_zero_count": numerical_zero_count,
        "minimum_raw_syn_bit": minimum_raw_syn,
        "significant_negative_count": 0,
        "prediction_devices": sorted(prediction_devices),
    }
    args.syn_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.syn_cache,
        centrality=monthly,
        mode_names=np.asarray(MODE_NAMES, dtype="U"),
        metadata=json.dumps(diagnostics, sort_keys=True),
    )
    return monthly, diagnostics


def load_or_compute_monthly_centrality(
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    if not args.recompute_syn and args.syn_cache.exists():
        with np.load(args.syn_cache, allow_pickle=False) as payload:
            centrality = np.asarray(payload["centrality"], dtype=np.float64)
            mode_names = payload["mode_names"].astype(str).tolist()
            diagnostics = json.loads(str(payload["metadata"]))
        if centrality.shape != (12, len(MODE_NAMES), 24, len(MODE_NAMES)):
            raise ValueError(f"Unexpected monthly centrality shape {centrality.shape}.")
        if mode_names != list(MODE_NAMES):
            raise ValueError("Mode order in the monthly Syn cache does not match UniCM.")
        cached_devices = diagnostics.get("prediction_devices", [])
        if args.required_prediction_device and cached_devices != [args.required_prediction_device]:
            raise RuntimeError(
                "Monthly Syn cache has incompatible or missing prediction-device provenance: "
                f"{cached_devices!r}; required {[args.required_prediction_device]!r}. "
                "Recompute from matching caches."
            )
        return centrality, diagnostics
    return compute_monthly_centrality(args)


def feature_scales(centrality: np.ndarray, gamma: float, floor_fraction: float) -> np.ndarray:
    penalty = feature_penalty(centrality, gamma, floor_fraction=floor_fraction)
    return 1.0 / np.sqrt(penalty)


def prepare_weighted_designs(
    base: np.ndarray,
    target: np.ndarray,
    additive_history: np.ndarray,
    split,
    monthly_centrality: np.ndarray,
    sample_months: np.ndarray,
    *,
    gamma: float,
    floor_fraction: float,
    fixed_month: int | None,
    month_offset: int = 0,
) -> dict[tuple[int, int], WeightedDesign]:
    selected_months = (
        np.full(len(sample_months), int(fixed_month), dtype=np.int64)
        if fixed_month is not None
        else (sample_months + int(month_offset)) % 12
    )
    designs: dict[tuple[int, int], WeightedDesign] = {}
    for lead in range(24):
        raw = np.concatenate((base[:, :, lead], additive_history), axis=1)
        x_fit, x_validation, x_test = standardize_fit(
            raw[split.fit], raw[split.validation], raw[split.test]
        )
        for target_index in range(target.shape[1]):
            month_scale = np.stack(
                [
                    feature_scales(
                        monthly_centrality[month, target_index, lead],
                        gamma,
                        floor_fraction,
                    )
                    for month in range(12)
                ]
            )
            fit = x_fit * month_scale[selected_months[split.fit]]
            validation = x_validation * month_scale[selected_months[split.validation]]
            test = x_test * month_scale[selected_months[split.test]]
            y = target[split.fit, target_index, lead]
            designs[(target_index, lead)] = WeightedDesign(
                x_fit=fit,
                x_validation=validation,
                x_test=test,
                y_fit=y - y.mean(),
                y_mean=float(y.mean()),
            )
    return designs


def predict_weighted_ridge(
    designs: dict[tuple[int, int], WeightedDesign],
    *,
    alpha: float,
    n_targets: int,
) -> tuple[np.ndarray, np.ndarray]:
    first = next(iter(designs.values()))
    validation = np.empty((len(first.x_validation), n_targets, 24), dtype=np.float64)
    test = np.empty((len(first.x_test), n_targets, 24), dtype=np.float64)
    for target_index in range(n_targets):
        for lead in range(24):
            design = designs[(target_index, lead)]
            gram = design.x_fit.T @ design.x_fit
            gram.flat[:: gram.shape[0] + 1] += float(alpha)
            coefficient = np.linalg.solve(gram, design.x_fit.T @ design.y_fit)
            validation[:, target_index, lead] = design.x_validation @ coefficient + design.y_mean
            test[:, target_index, lead] = design.x_test @ coefficient + design.y_mean
    return validation, test


def tune_condition(
    base: np.ndarray,
    target: np.ndarray,
    additive_history: np.ndarray,
    split,
    target_scale: np.ndarray,
    monthly_centrality: np.ndarray,
    sample_months: np.ndarray,
    *,
    alphas: list[float],
    gammas: list[float],
    floor_fraction: float,
    fixed_month: int | None,
    month_offset: int = 0,
) -> tuple[float, float, dict[str, float], np.ndarray]:
    best: tuple[float, float, float, np.ndarray] | None = None
    scores: dict[str, float] = {}
    for gamma in gammas:
        designs = prepare_weighted_designs(
            base,
            target,
            additive_history,
            split,
            monthly_centrality,
            sample_months,
            gamma=gamma,
            floor_fraction=floor_fraction,
            fixed_month=fixed_month,
            month_offset=month_offset,
        )
        for alpha in alphas:
            validation, test = predict_weighted_ridge(
                designs, alpha=alpha, n_targets=target.shape[1]
            )
            score = float(
                np.mean(
                    cell_rmse(
                        validation,
                        target[split.validation],
                        target_scale,
                    )
                )
            )
            scores[f"alpha={alpha:g},gamma={gamma:g}"] = score
            if best is None or score < best[0]:
                best = (score, alpha, gamma, test)
    assert best is not None
    return best[1], best[2], scores, best[3]


def plot_result(
    metrics: dict[str, dict[str, object]],
    mode_names: list[str],
    circular_scores: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    names = ["frozen", "univariate", "uniform", "fixed_syn", "matched_syn"]
    labels = ["Frozen", "Univariate", "Uniform", "Fixed Syn", "Matched Syn"]
    colors = ["#8A9197", "#91A7C4", "#6F8FB3", "#D9A067", "#C75B4A"]
    values = [float(metrics[name]["mean_cell_nrmse"]) for name in names]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.45),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.9, 1.05, 1.0]},
    )
    axes[0].scatter(np.arange(len(names)), values, c=colors, s=28, zorder=3)
    axes[0].set_xticks(np.arange(len(names)), labels, rotation=35, ha="right")
    axes[0].set_ylabel("Test mean normalized RMSE")

    gain = np.asarray(metrics["fixed_syn"]["target_nrmse"]) - np.asarray(
        metrics["matched_syn"]["target_nrmse"]
    )
    order = np.argsort(gain)
    axes[1].axvline(0, color="0.65", lw=0.8)
    axes[1].scatter(gain[order], np.arange(len(mode_names)), color="#C75B4A", s=22)
    axes[1].set_yticks(np.arange(len(mode_names)), np.asarray(mode_names)[order])
    axes[1].set_xlabel("Matched-minus-fixed gain")

    fixed_score = float(metrics["fixed_syn"]["mean_cell_nrmse"])
    observed_gain = fixed_score - float(metrics["matched_syn"]["mean_cell_nrmse"])
    null_gain = fixed_score - circular_scores
    axes[2].axhline(0, color="0.65", lw=0.8)
    axes[2].scatter(np.arange(1, 12), null_gain, color="#93A8B8", s=20, label="Circular shifts")
    axes[2].axhline(observed_gain, color="#C75B4A", lw=1.6, label="Correct month")
    axes[2].set_xlabel("Calendar offset (months)")
    axes[2].set_ylabel("Gain over fixed Syn")
    axes[2].set_xticks([1, 3, 6, 9, 11])
    axes[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    for label, axis in zip("abc", axes):
        axis.text(-0.16, 1.05, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_contract(path: Path, args: argparse.Namespace) -> None:
    path.write_text(
        f"""# Season-matched Syn prior: experiment contract

## Scientific question

What changes when only the calendar-month assignment of the target-specific PEID Syn prior changes from a fixed month to each sample's actual first forecast month?

## Treatment and controls

- Treatment: fixed `start_month=0` Syn prior versus correctly matched monthly Syn prior.
- Capacity: one shared 44-coefficient ridge readout per target and lead in both conditions; no month-specific learned coefficients.
- Frozen: ORAS5 inputs and fit-only normalization, three released UniCM checkpoints, maximum-entropy intervention samples, affine degree-1 TM estimator, feature set, chronological split, hyperparameter grids, metrics, and bootstrap procedure.
- Fit/validation/test issue dates: through `{args.fit_end}` / `{args.validation_start}`--`{args.validation_end}` / `{args.test_start}`--`{args.test_end}`.
- Primary endpoint: paired test mean-cell nRMSE difference, fixed minus matched (positive favors matching).
- Null: circularly offset the month assignment by 1--11 months, retuning each offset on validation.
- Syn nonnegative tolerance: `{args.syn_zero_tolerance:g}` bit; values inside the tolerance are recorded as numerical zero, while larger violations abort.

The first target month maps directly to the model cache `start_month`: the 12-month history and lead-1 target have the same calendar-month index modulo 12.
""",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_contract(args.output_dir / "experiment_contract.md", args)
    monthly_centrality, syn_diagnostics = load_or_compute_monthly_centrality(args)

    with np.load(args.input_dir / "model_inputs.npz", allow_pickle=False) as payload:
        history = payload["history"].astype(np.float64)
        target = payload["targets"].astype(np.float64)
        mode_names = payload["mode_names"].astype(str).tolist()
        input_metadata = json.loads(str(payload["metadata"]))
    with np.load(args.input_dir / "modeformer_predictions.npz", allow_pickle=False) as payload:
        predictions_by_seed = payload["predictions_by_seed"].astype(np.float64)
        ensemble = payload["ensemble_prediction"].astype(np.float64)
        target_dates = payload["target_dates"].astype(str)
    if [canonical_mode_name(name) for name in mode_names] != [
        canonical_mode_name(name) for name in MODE_NAMES
    ]:
        raise ValueError("Observed mode order does not match the intervention cache.")

    split = chronological_split(
        target_dates,
        fit_end=args.fit_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    sample_months = first_target_months(target_dates)
    _, scale = target_scaling(target, split.fit)
    scale = scale.reshape(1, len(mode_names), 1)
    additive_history, _ = history_summaries(history)
    alphas = [float(value) for value in args.alphas.split(",")]
    gammas = [float(value) for value in args.gammas.split(",")]

    fixed_alpha, fixed_gamma, fixed_tuning, test_fixed = tune_condition(
        ensemble,
        target,
        additive_history,
        split,
        scale,
        monthly_centrality,
        sample_months,
        alphas=alphas,
        gammas=gammas,
        floor_fraction=args.floor_fraction,
        fixed_month=0,
    )
    matched_alpha, matched_gamma, matched_tuning, test_matched = tune_condition(
        ensemble,
        target,
        additive_history,
        split,
        scale,
        monthly_centrality,
        sample_months,
        alphas=alphas,
        gammas=gammas,
        floor_fraction=args.floor_fraction,
        fixed_month=None,
    )
    uniform_alpha, _, uniform_tuning, test_uniform = tune_condition(
        ensemble,
        target,
        additive_history,
        split,
        scale,
        monthly_centrality,
        sample_months,
        alphas=alphas,
        gammas=[0.0],
        floor_fraction=args.floor_fraction,
        fixed_month=0,
    )

    empty = {lead: np.empty((len(history), 0), dtype=np.float64) for lead in range(24)}
    univariate_alphas = [float(value) for value in args.univariate_alphas.split(",")]
    univariate_alpha, univariate_tuning = select_alpha(
        ensemble,
        target,
        additive_history,
        empty,
        split,
        scale,
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

    circular_scores = []
    circular_hyperparameters = []
    for offset in range(1, 12):
        alpha, gamma, _, prediction = tune_condition(
            ensemble,
            target,
            additive_history,
            split,
            scale,
            monthly_centrality,
            sample_months,
            alphas=alphas,
            gammas=gammas,
            floor_fraction=args.floor_fraction,
            fixed_month=None,
            month_offset=offset,
        )
        score = method_metrics(prediction, target[split.test], scale)["mean_cell_nrmse"]
        circular_scores.append(float(score))
        circular_hyperparameters.append({"offset": offset, "alpha": alpha, "gamma": gamma})
        print(f"Circular month offset {offset}/11: test nRMSE={float(score):.6f}", flush=True)
    circular_scores_array = np.asarray(circular_scores)

    predictions = {
        "frozen": ensemble[split.test],
        "univariate": test_univariate,
        "uniform": test_uniform,
        "fixed_syn": test_fixed,
        "matched_syn": test_matched,
    }
    test_target = target[split.test]
    metrics = {
        name: method_metrics(prediction, test_target, scale)
        for name, prediction in predictions.items()
    }
    bootstrap = bootstrap_gain(
        test_fixed,
        test_matched,
        test_target,
        scale,
        args.bootstrap,
        args.bootstrap_block,
        args.seed,
    )

    checkpoint_gains = []
    for seed_index, seed_prediction in enumerate(predictions_by_seed):
        fixed_designs = prepare_weighted_designs(
            seed_prediction,
            target,
            additive_history,
            split,
            monthly_centrality,
            sample_months,
            gamma=fixed_gamma,
            floor_fraction=args.floor_fraction,
            fixed_month=0,
        )
        _, seed_fixed = predict_weighted_ridge(
            fixed_designs, alpha=fixed_alpha, n_targets=len(mode_names)
        )
        matched_designs = prepare_weighted_designs(
            seed_prediction,
            target,
            additive_history,
            split,
            monthly_centrality,
            sample_months,
            gamma=matched_gamma,
            floor_fraction=args.floor_fraction,
            fixed_month=None,
        )
        _, seed_matched = predict_weighted_ridge(
            matched_designs, alpha=matched_alpha, n_targets=len(mode_names)
        )
        fixed_score = method_metrics(seed_fixed, test_target, scale)["mean_cell_nrmse"]
        matched_score = method_metrics(seed_matched, test_target, scale)["mean_cell_nrmse"]
        checkpoint_gains.append(
            {
                "checkpoint_seed": seed_index + 1,
                "fixed_nrmse": float(fixed_score),
                "matched_nrmse": float(matched_score),
                "gain_fixed_minus_matched": float(fixed_score - matched_score),
            }
        )

    observed_gain = float(
        metrics["fixed_syn"]["mean_cell_nrmse"]
        - metrics["matched_syn"]["mean_cell_nrmse"]
    )
    circular_p = float(
        (1 + np.count_nonzero(circular_scores_array <= metrics["matched_syn"]["mean_cell_nrmse"]))
        / 12
    )
    summary = {
        "status": "completed",
        "question": "Does matching the target-specific Syn prior to the actual forecast calendar month improve UniCM calibration?",
        "input_preprocessing": input_metadata,
        "samples": {"fit": len(split.fit), "validation": len(split.validation), "test": len(split.test)},
        "chronological_split": {
            "fit_end": args.fit_end,
            "validation_start": args.validation_start,
            "validation_end": args.validation_end,
            "test_start": args.test_start,
            "test_end": args.test_end,
        },
        "selected_hyperparameters": {
            "fixed_syn": {"alpha": fixed_alpha, "gamma": fixed_gamma},
            "matched_syn": {"alpha": matched_alpha, "gamma": matched_gamma},
            "uniform_alpha": uniform_alpha,
            "univariate_alpha": univariate_alpha,
            "floor_fraction": args.floor_fraction,
        },
        "syn_diagnostics": syn_diagnostics,
        "test_metrics": metrics,
        "gain_fixed_minus_matched": {
            "observed": observed_gain,
            "relative_percent": 100.0 * observed_gain / float(metrics["fixed_syn"]["mean_cell_nrmse"]),
            "bootstrap_ci95": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            "bootstrap_positive_fraction": float(np.mean(bootstrap > 0)),
        },
        "circular_month_null": {
            "scores": circular_scores,
            "hyperparameters": circular_hyperparameters,
            "fraction_offset_at_least_as_good_with_finite_correction": circular_p,
            "preserves": "monthly Syn surfaces, shared coefficient count, validation tuning, and month frequencies",
            "destroys": "correct alignment between sample calendar month and Syn prior",
        },
        "checkpoint_gains": checkpoint_gains,
        "validation_scores": {
            "fixed_syn": fixed_tuning,
            "matched_syn": matched_tuning,
            "uniform": uniform_tuning,
            "univariate": univariate_tuning,
        },
        "controls": {
            "same_features": True,
            "same_parameter_count": True,
            "same_split": True,
            "same_intervention_samples": True,
            "same_estimator": True,
            "test_used_for_selection": False,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "evaluation_arrays.npz",
        circular_scores=circular_scores_array,
        bootstrap_gain=bootstrap,
        gain_cells=cell_rmse(test_fixed, test_target, scale)
        - cell_rmse(test_matched, test_target, scale),
        test_target=test_target.astype(np.float32),
        **{f"prediction_{name}": value.astype(np.float32) for name, value in predictions.items()},
    )
    plot_result(
        metrics,
        mode_names,
        circular_scores_array,
        args.output_dir / "season_matched_syn_calibration.png",
    )
    report = f"""# UniCM season-matched Syn calibration

Matching the target-specific Syn prior to each sample's calendar month changed test mean-cell nRMSE from **{float(metrics['fixed_syn']['mean_cell_nrmse']):.5f}** to **{float(metrics['matched_syn']['mean_cell_nrmse']):.5f}**. The paired gain (fixed minus matched) is **{observed_gain:.5f}** ({100.0 * observed_gain / float(metrics['fixed_syn']['mean_cell_nrmse']):.2f}%), with a 12-month block-bootstrap 95% interval of [{np.percentile(bootstrap, 2.5):.5f}, {np.percentile(bootstrap, 97.5):.5f}].

The frozen, univariate, uniform-ridge, fixed-Syn, and matched-Syn nRMSE values are {float(metrics['frozen']['mean_cell_nrmse']):.5f}, {float(metrics['univariate']['mean_cell_nrmse']):.5f}, {float(metrics['uniform']['mean_cell_nrmse']):.5f}, {float(metrics['fixed_syn']['mean_cell_nrmse']):.5f}, and {float(metrics['matched_syn']['mean_cell_nrmse']):.5f}, respectively.

This is a controlled development-period result on 2009--2016 issue dates, not a new untouched final holdout. The Zotero PEID literature interface was unavailable in this task, so the implementation follows the repository's existing PEID/TM definitions.
"""
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"test_nrmse": {k: v["mean_cell_nrmse"] for k, v in metrics.items()}, "gain": summary["gain_fixed_minus_matched"], "checkpoint_gains": checkpoint_gains}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prediction-cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--required-prediction-device",
        default="cpu",
        help="Reject caches produced on a different device; use an empty value only for diagnostics.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--syn-cache", type=Path, default=DEFAULT_SYN_CACHE)
    parser.add_argument("--recompute-syn", action="store_true")
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--sampling-seed", type=int, default=20260619)
    parser.add_argument("--intervention-bound", type=float, default=4.0)
    parser.add_argument("--checkpoint-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--syn-zero-tolerance", type=float, default=DEFAULT_SYN_ZERO_TOLERANCE)
    parser.add_argument("--alphas", default="100,300,1000,3000,10000,30000")
    parser.add_argument("--gammas", default="0,0.5,1,2,3")
    parser.add_argument("--univariate-alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--floor-fraction", type=float, default=0.05)
    parser.add_argument("--fit-end", default="2001-12")
    parser.add_argument("--validation-start", default="2004-01")
    parser.add_argument("--validation-end", default="2006-12")
    parser.add_argument("--test-start", default="2009-01")
    parser.add_argument("--test-end", default="2016-12")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-block", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260920)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
