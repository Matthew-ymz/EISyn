#!/usr/bin/env python3
"""Test a target-specific all-order Xi-Shapley calibration prior for UniCM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_unicm_11mode_shapley import (
    coalition_ei_table,
    exact_shapley,
    fit_affine_readout,
    mode_feature_blocks,
    standardize,
)
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
    shuffled_centrality,
    tune_prior,
)
from scripts.unicm_peid_syn_analysis import (
    MODE_NAMES,
    overall_prediction_cache_path,
    sample_full_history_mode_inputs,
)


DEFAULT_INPUT = ROOT / "data" / "ORAS5" / "modeformer_1980_2018"
DEFAULT_CACHE = ROOT / "results" / "unicm_overall_ei_cpu_bound4_n8192" / "cache"
DEFAULT_CALIBRATION = (
    ROOT / "results" / "unicm_synergy_regularized_forecast_extended_1980_2018"
)
DEFAULT_OUTPUT = ROOT / "results" / "unicm_target_xi_shapley_prior"
XI_TOLERANCE_BITS = 1e-8
CLOSURE_TOLERANCE_BITS = 1e-10


def prediction_cache_path(cache_dir: Path, seed: int, args: argparse.Namespace) -> Path:
    cache_args = SimpleNamespace(
        n_samples=args.n_samples,
        sampling_seed=args.sampling_seed,
        intervention_bound=args.intervention_bound,
        start_month=args.start_month,
        device="cpu",
    )
    return overall_prediction_cache_path(cache_dir, seed=seed, args=cache_args)


def target_xi_shapley_centrality(
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Return checkpoint-mean [target, lead, source] exact Shapley values."""

    source = sample_full_history_mode_inputs(
        n_samples=args.n_samples,
        intervention_bound=args.intervention_bound,
        seed=args.sampling_seed,
    ).astype(np.float64)
    standardized_source = standardize(source.reshape(args.n_samples, -1), "source")
    blocks = mode_feature_blocks()
    by_seed = np.empty(
        (len(args.checkpoint_seeds), len(MODE_NAMES), 24, len(MODE_NAMES)),
        dtype=np.float64,
    )
    minimum_subset_xi = np.inf
    minimum_shapley = np.inf
    maximum_closure_error = 0.0
    subset_numerical_zero_count = 0
    shapley_numerical_zero_count = 0

    for seed_index, seed in enumerate(args.checkpoint_seeds):
        path = prediction_cache_path(args.cache_dir, seed, args)
        with np.load(path, allow_pickle=False) as archive:
            predictions = archive["all_mode_targets"].astype(np.float64)
        expected = (args.n_samples, 24, len(MODE_NAMES))
        if predictions.shape != expected:
            raise ValueError(f"Unexpected prediction shape {predictions.shape}; expected {expected}.")

        for lead in range(24):
            coefficients, residual_covariance = fit_affine_readout(
                standardized_source,
                predictions[:, lead, :],
                args.covariance_ridge,
            )
            for target_index in range(len(MODE_NAMES)):
                ei = coalition_ei_table(
                    coefficients[:, target_index : target_index + 1],
                    np.asarray([[residual_covariance[target_index, target_index]]]),
                    blocks,
                )
                singleton = np.asarray(
                    [ei[1 << source_index] for source_index in range(len(MODE_NAMES))],
                    dtype=np.float64,
                )
                xi = {
                    mask: ei[mask]
                    - sum(
                        singleton[source_index]
                        for source_index in range(len(MODE_NAMES))
                        if mask & (1 << source_index)
                    )
                    for mask in range(1 << len(MODE_NAMES))
                }
                tested = np.asarray(
                    [value for mask, value in xi.items() if mask.bit_count() >= 2],
                    dtype=np.float64,
                )
                minimum_subset_xi = min(minimum_subset_xi, float(tested.min()))
                significant = tested < -XI_TOLERANCE_BITS
                if np.any(significant):
                    raise ValueError(
                        "Target-specific Xi violates nonnegativity: "
                        f"seed={seed}, lead={lead + 1}, target={MODE_NAMES[target_index]}, "
                        f"minimum={float(tested.min()):.12g} bit, "
                        f"threshold=-{XI_TOLERANCE_BITS:.12g} bit, "
                        f"count={int(np.count_nonzero(significant))}."
                    )
                numerical_masks = [
                    mask
                    for mask, value in xi.items()
                    if mask.bit_count() >= 2 and -XI_TOLERANCE_BITS <= value < 0
                ]
                subset_numerical_zero_count += len(numerical_masks)
                for mask in numerical_masks:
                    xi[mask] = 0.0

                attribution = exact_shapley(xi, len(MODE_NAMES))
                total = float(xi[(1 << len(MODE_NAMES)) - 1])
                closure_error = float(attribution.sum() - total)
                maximum_closure_error = max(maximum_closure_error, abs(closure_error))
                if abs(closure_error) > CLOSURE_TOLERANCE_BITS:
                    raise ValueError(
                        "Target-specific Xi Shapley closure failed: "
                        f"seed={seed}, lead={lead + 1}, target={MODE_NAMES[target_index]}, "
                        f"error={closure_error:.12g} bit."
                    )
                minimum_shapley = min(minimum_shapley, float(attribution.min()))
                if np.any(attribution < -XI_TOLERANCE_BITS):
                    raise ValueError(
                        "Target-specific Xi Shapley centrality is significantly negative: "
                        f"seed={seed}, lead={lead + 1}, target={MODE_NAMES[target_index]}, "
                        f"minimum={float(attribution.min()):.12g} bit."
                    )
                numerical = (attribution < 0) & (attribution >= -XI_TOLERANCE_BITS)
                shapley_numerical_zero_count += int(np.count_nonzero(numerical))
                attribution[numerical] = 0.0
                if float(attribution.sum()) <= 0:
                    raise ValueError("A target-lead Shapley centrality vector has zero total mass.")
                by_seed[seed_index, target_index, lead] = attribution
        print(f"checkpoint {seed}: target-specific Xi-Shapley complete", flush=True)

    return by_seed.mean(axis=0), {
        "xi_tolerance_bit": XI_TOLERANCE_BITS,
        "closure_tolerance_bit": CLOSURE_TOLERANCE_BITS,
        "minimum_subset_xi_bit": float(minimum_subset_xi),
        "subset_numerical_zero_count": int(subset_numerical_zero_count),
        "minimum_shapley_bit": float(minimum_shapley),
        "shapley_numerical_zero_count": int(shapley_numerical_zero_count),
        "significant_nonnegativity_violation_count": 0,
        "maximum_absolute_closure_error_bit": float(maximum_closure_error),
    }


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
    values = np.empty((replicates, target.shape[2]), dtype=np.float64)
    for replicate in range(replicates):
        sample = circular_block_indices(rng, len(target), block_length)
        values[replicate] = leadwise_gain(
            baseline[sample], treatment[sample], target[sample], target_scale
        )
    return values


def plot_comparison(
    gains: dict[str, float],
    gain_ci: dict[str, np.ndarray],
    lead_gains: dict[str, np.ndarray],
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
    keys = ("pair_syn", "target_xi_shapley")
    labels = ("Target-pair Syn prior", r"Target-specific $\Xi$ Shapley prior")
    colors = {"pair_syn": "#D9822B", "target_xi_shapley": "#287D76"}
    ink = "#273142"
    grid = "#E7EBEF"

    fig, (ax_summary, ax_lead) = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (0.82, 1.58)},
    )
    y = np.arange(len(keys))[::-1]
    ax_summary.axvline(0, color="#6D7782", lw=0.75, ls="--", zorder=1)
    for row, key in zip(y, keys):
        value = gains[key]
        low, high = gain_ci[key]
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
            high + 0.0015,
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
    ax_summary.grid(axis="x", color=grid, linewidth=0.55)

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
            lead_gains[key],
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
    ax_lead.grid(axis="y", color=grid, linewidth=0.55)
    ax_lead.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.25,
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
    centrality, centrality_audit = target_xi_shapley_centrality(args)

    with np.load(args.input_dir / "model_inputs.npz", allow_pickle=False) as archive:
        history = archive["history"].astype(np.float64)
        target = archive["targets"].astype(np.float64)
        mode_names = archive["mode_names"].astype(str).tolist()
    with np.load(
        args.input_dir / "modeformer_predictions.npz", allow_pickle=False
    ) as archive:
        ensemble = archive["ensemble_prediction"].astype(np.float64)
        target_dates = archive["target_dates"].astype(str)
    expected_names = ["ENSO" if name == "nino" else name for name in MODE_NAMES]
    if mode_names != expected_names:
        raise ValueError("Calibration and intervention-cache mode orders do not match.")

    split = chronological_split(target_dates, **calibration["chronological_split"])
    _, target_scale = target_scaling(target, split.fit)
    target_scale = target_scale.reshape(1, len(mode_names), 1)
    additive_history, _ = history_summaries(history)
    designs = prepare_designs(ensemble, target, additive_history, split)
    alphas = [float(value) for value in args.alphas.split(",")]
    gammas = [float(value) for value in args.gammas.split(",")]
    alpha, gamma, tuning, _, test_xi = tune_prior(
        designs,
        centrality,
        target[split.validation],
        target_scale,
        alphas=alphas,
        gammas=gammas,
        floor_fraction=args.floor_fraction,
    )
    print(
        f"Target-specific Xi prior: alpha={alpha:g}, gamma={gamma:g}, "
        f"validation_nRMSE={min(tuning.values()):.6f}",
        flush=True,
    )

    with np.load(args.calibration_dir / "evaluation_arrays.npz") as archive:
        cached_target = archive["test_target"].astype(np.float64)
        test_uniform = archive["prediction_uniform"].astype(np.float64)
        test_pair = archive["prediction_syn_regularized"].astype(np.float64)
    test_target = target[split.test]
    if not np.allclose(test_target, cached_target, rtol=0, atol=1e-6):
        raise ValueError("Cached and reconstructed test targets do not match.")

    rng = np.random.default_rng(args.seed)
    random_scores: list[float] = []
    random_hyperparameters: list[dict[str, float | int]] = []
    from tqdm.auto import tqdm

    for repeat in tqdm(
        range(args.random_repeats),
        desc="shuffled-Xi null",
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
            mean_cell_nrmse(null_test, test_target, target_scale)
        )
        random_hyperparameters.append(
            {"repeat": repeat, "alpha": null_alpha, "gamma": null_gamma}
        )
    random_scores_array = np.asarray(random_scores, dtype=np.float64)
    xi_score = mean_cell_nrmse(test_xi, test_target, target_scale)
    random_p = float(
        (1 + np.count_nonzero(random_scores_array <= xi_score))
        / (len(random_scores_array) + 1)
    )

    predictions = {"pair_syn": test_pair, "target_xi_shapley": test_xi}
    gains: dict[str, float] = {}
    gain_ci: dict[str, np.ndarray] = {}
    lead_gains: dict[str, np.ndarray] = {}
    lead_ci: dict[str, np.ndarray] = {}
    for index, (key, prediction) in enumerate(predictions.items()):
        gains[key] = mean_cell_nrmse(
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
        gain_ci[key] = np.percentile(aggregate_bootstrap, [2.5, 97.5])
        lead_gains[key] = leadwise_gain(
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

    pair_to_xi_bootstrap = bootstrap_gain(
        test_pair,
        test_xi,
        test_target,
        target_scale,
        args.bootstrap,
        args.bootstrap_block,
        args.seed + 200,
    )
    xi_lead_bootstrap = bootstrap_leadwise_gain(
        test_uniform,
        test_xi,
        test_target,
        target_scale,
        replicates=args.bootstrap,
        block_length=args.bootstrap_block,
        seed=args.seed + 201,
    )

    output = args.output_dir / "target_xi_shapley_prior_comparison.png"
    plot_comparison(gains, gain_ci, lead_gains, lead_ci, output)
    result = {
        "status": "completed",
        "definition": "Target- and lead-specific exact Shapley allocation of Xi(S)=EI(S->Y_jl)-sum singleton EI",
        "centrality_audit": centrality_audit,
        "selected_hyperparameters": {"alpha": alpha, "gamma": gamma},
        "validation_nrmse": float(min(tuning.values())),
        "test_nrmse": {
            "uniform_ridge": mean_cell_nrmse(test_uniform, test_target, target_scale),
            "target_pair_syn_prior": mean_cell_nrmse(test_pair, test_target, target_scale),
            "target_xi_shapley_prior": xi_score,
        },
        "gain_over_uniform": {
            key: {"point": gains[key], "ci95": gain_ci[key].tolist()}
            for key in predictions
        },
        "gain_over_target_pair_syn": {
            "point": mean_cell_nrmse(test_pair, test_target, target_scale)
            - mean_cell_nrmse(test_xi, test_target, target_scale),
            "ci95": np.percentile(pair_to_xi_bootstrap, [2.5, 97.5]).tolist(),
        },
        "target_xi_peak_gain_lead": int(np.argmax(lead_gains["target_xi_shapley"]) + 1),
        "target_xi_lead_8_gain": float(lead_gains["target_xi_shapley"][7]),
        "target_xi_lead_8_gain_ci95": np.percentile(
            xi_lead_bootstrap[:, 7], [2.5, 97.5]
        ).tolist(),
        "target_xi_lead_7_10_mean_gain": float(
            lead_gains["target_xi_shapley"][6:10].mean()
        ),
        "target_xi_lead_7_10_mean_gain_ci95": np.percentile(
            xi_lead_bootstrap[:, 6:10].mean(axis=1), [2.5, 97.5]
        ).tolist(),
        "shuffled_xi_control": {
            "repeats": args.random_repeats,
            "scores": random_scores,
            "mean": float(random_scores_array.mean()),
            "std": float(random_scores_array.std(ddof=1)),
            "fraction_null_at_least_as_good": random_p,
            "hyperparameters": random_hyperparameters,
            "null_preserves": "each target-lead Xi-Shapley centrality value distribution",
            "null_destroys": "assignment of Xi-Shapley centrality to source-mode labels",
            "each_null_retuned_on_validation": True,
        },
        "output": str(output),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "evaluation_arrays.npz",
        centrality=centrality,
        random_scores=random_scores_array,
        prediction_target_xi_shapley=test_xi.astype(np.float32),
        bootstrap_pair_to_xi_gain=pair_to_xi_bootstrap,
        bootstrap_xi_lead_gain=xi_lead_bootstrap,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--sampling-seed", type=int, default=20260619)
    parser.add_argument("--intervention-bound", type=float, default=4.0)
    parser.add_argument("--start-month", type=int, default=0)
    parser.add_argument("--covariance-ridge", type=float, default=1e-6)
    parser.add_argument("--alphas", default="100,300,1000,3000,10000,30000")
    parser.add_argument("--gammas", default="0,0.5,1,2,3")
    parser.add_argument("--floor-fraction", type=float, default=0.05)
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--bootstrap-block", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
