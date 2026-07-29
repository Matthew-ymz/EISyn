#!/usr/bin/env python3
"""Test whether frozen Modeformer synergy scores improve forecast correction."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "ORAS5" / "modeformer_1980_2014"
DEFAULT_OUTPUT = ROOT / "results" / "unicm_synergy_guided_forecast"
DEFAULT_PAIR_SYN = (
    ROOT
    / "results"
    / "unicm_all_mode_target_pair_syn_tm_degree1_n8192"
    / "all_mode_target_pair_syn_rows.csv"
)
DEFAULT_TARGET_XI = (
    ROOT
    / "results"
    / "unicm_target_resolved_xi_tm_degree1_signed_n8192"
    / "target_resolved_xi_rows.csv"
)
DEFAULT_TARGET_PAIR_SYN = (
    ROOT
    / "results"
    / "unicm_target_pair_syn_tm_degree1_signed_n8192"
    / "target_pair_syn_summary.csv"
)


@dataclass(frozen=True)
class Split:
    fit: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    issue_dates: np.ndarray


def issue_months(target_dates: np.ndarray) -> np.ndarray:
    first = target_dates[:, 0].astype("datetime64[M]")
    return first - np.timedelta64(1, "M")


def chronological_split(target_dates: np.ndarray) -> Split:
    issue = issue_months(target_dates)
    fit = np.flatnonzero(issue <= np.datetime64("2001-12"))
    validation = np.flatnonzero(
        (issue >= np.datetime64("2004-01"))
        & (issue <= np.datetime64("2006-12"))
    )
    test = np.flatnonzero(
        (issue >= np.datetime64("2009-01"))
        & (issue <= np.datetime64("2012-12"))
    )
    if min(len(fit), len(validation), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition.")
    return Split(fit=fit, validation=validation, test=test, issue_dates=issue)


def history_summaries(history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return additive [N, 33] and pair-ready [N, 11, 4] summaries."""

    time = np.arange(history.shape[-1], dtype=np.float64)
    centered = time - time.mean()
    trend = np.einsum("nmt,t->nm", history, centered) / np.sum(centered**2)
    final = history[:, :, -1]
    mean = history.mean(axis=-1)
    additive = np.concatenate((final, mean, trend), axis=1)
    pair_ready = np.stack((final, mean, trend), axis=-1)
    return additive.astype(np.float64), pair_ready.astype(np.float64)


def all_pairs(n_modes: int) -> list[tuple[int, int]]:
    return [(left, right) for left in range(n_modes) for right in range(left + 1, n_modes)]


def pair_features(
    history: np.ndarray,
    pair_ready: np.ndarray,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for left, right in pairs:
        columns.extend(
            (
                pair_ready[:, left, 0] * pair_ready[:, right, 0],
                pair_ready[:, left, 1] * pair_ready[:, right, 1],
                pair_ready[:, left, 2] * pair_ready[:, right, 2],
                np.mean(history[:, left] * history[:, right], axis=-1),
            )
        )
    if not columns:
        return np.empty((history.shape[0], 0), dtype=np.float64)
    return np.stack(columns, axis=1).astype(np.float64)


def normalize_name(name: str) -> str:
    return "ENSO" if name == "nino" else str(name)


def load_pair_rankings(
    path: Path,
    mode_names: list[str],
) -> tuple[dict[int, list[tuple[int, int]]], np.ndarray]:
    frame = pd.read_csv(path)
    frame["left_source"] = frame["left_source"].map(normalize_name)
    frame["right_source"] = frame["right_source"].map(normalize_name)
    lookup = {name: index for index, name in enumerate(mode_names)}
    summary = (
        frame.groupby(["lead", "left_source", "right_source"], as_index=False)["syn"]
        .mean()
        .sort_values(["lead", "syn"], ascending=[True, False])
    )
    rankings: dict[int, list[tuple[int, int]]] = {}
    score = np.full((24, len(mode_names), len(mode_names)), np.nan, dtype=np.float64)
    for lead, group in summary.groupby("lead", sort=True):
        pairs: list[tuple[int, int]] = []
        for row in group.itertuples():
            left, right = lookup[row.left_source], lookup[row.right_source]
            pair = (min(left, right), max(left, right))
            pairs.append(pair)
            score[int(lead) - 1, pair[0], pair[1]] = float(row.syn)
            score[int(lead) - 1, pair[1], pair[0]] = float(row.syn)
        rankings[int(lead) - 1] = pairs
    if set(rankings) != set(range(24)):
        raise ValueError("Pair Syn table does not cover all 24 leads.")
    return rankings, score


def load_target_xi(path: Path, mode_names: list[str]) -> np.ndarray:
    frame = pd.read_csv(path)
    frame["display_target"] = frame["display_target"].map(normalize_name)
    summary = frame.groupby(["display_target", "lead"], as_index=False)["xi_target"].mean()
    lookup = {name: index for index, name in enumerate(mode_names)}
    xi = np.full((len(mode_names), 24), np.nan, dtype=np.float64)
    for row in summary.itertuples():
        xi[lookup[row.display_target], int(row.lead) - 1] = float(row.xi_target)
    if not np.isfinite(xi).all():
        raise ValueError("Target-resolved Xi table does not cover all target–lead cells.")
    return xi


def load_target_pair_rankings(
    path: Path,
    mode_names: list[str],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    frame = pd.read_csv(path)
    frame["target"] = frame["target"].map(normalize_name)
    frame["left_source"] = frame["left_source"].map(normalize_name)
    frame["right_source"] = frame["right_source"].map(normalize_name)
    score_column = "mean" if "mean" in frame.columns else "syn"
    lookup = {name: index for index, name in enumerate(mode_names)}
    rankings: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for (target, lead), group in frame.groupby(["target", "lead"], sort=True):
        ordered = group.sort_values(score_column, ascending=False)
        rankings[(lookup[target], int(lead) - 1)] = [
            tuple(
                sorted(
                    (
                        lookup[row.left_source],
                        lookup[row.right_source],
                    )
                )
            )
            for row in ordered.itertuples()
        ]
    expected = {
        (target, lead)
        for target in range(len(mode_names))
        for lead in range(24)
    }
    if set(rankings) != expected:
        raise ValueError("Target-pair Syn table does not cover all target–lead cells.")
    return rankings


def standardize_fit(
    x_fit: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    mean = x_fit.mean(axis=0, keepdims=True)
    scale = x_fit.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return tuple((array - mean) / scale for array in (x_fit, *others))


def target_scaling(targets: np.ndarray, fit_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = targets[fit_index].mean(axis=(0, 2), keepdims=True)
    scale = targets[fit_index].std(axis=(0, 2), keepdims=True)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean, scale


def cell_rmse(
    prediction: np.ndarray,
    target: np.ndarray,
    target_scale: np.ndarray,
) -> np.ndarray:
    normalized_error = (prediction - target) / target_scale
    return np.sqrt(np.mean(normalized_error**2, axis=0))


def mean_cell_nrmse(
    prediction: np.ndarray,
    target: np.ndarray,
    target_scale: np.ndarray,
) -> float:
    return float(np.mean(cell_rmse(prediction, target, target_scale)))


def mean_cell_acc(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction - prediction.mean(axis=0, keepdims=True)
    target = target - target.mean(axis=0, keepdims=True)
    numerator = np.sum(prediction * target, axis=0)
    denominator = np.sqrt(np.sum(prediction**2, axis=0) * np.sum(target**2, axis=0))
    correlation = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 1e-12,
    )
    return float(np.nanmean(correlation))


def build_features(
    base_forecast: np.ndarray,
    additive_history: np.ndarray,
    interaction_features: np.ndarray,
    lead: int,
    method: str,
) -> np.ndarray:
    if method == "univariate":
        # Multi-output Ridge cannot express a separate univariate model from one
        # shared column, so the caller handles this method target by target.
        raise ValueError("univariate features are target-specific")
    pieces = [base_forecast[:, :, lead], additive_history]
    if method != "additive":
        pieces.append(interaction_features)
    return np.concatenate(pieces, axis=1).astype(np.float64)


def ridge_multioutput(
    x: np.ndarray,
    y: np.ndarray,
    split: Split,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_fit, x_val, x_test = standardize_fit(
        x[split.fit], x[split.validation], x[split.test]
    )
    y_fit = y[split.fit]
    y_mean = y_fit.mean(axis=0, keepdims=True)
    y_centered = y_fit - y_mean
    if x_fit.shape[1] <= x_fit.shape[0]:
        gram = x_fit.T @ x_fit
        gram.flat[:: gram.shape[0] + 1] += alpha
        coefficient = np.linalg.solve(gram, x_fit.T @ y_centered)
    else:
        gram = x_fit @ x_fit.T
        gram.flat[:: gram.shape[0] + 1] += alpha
        coefficient = x_fit.T @ np.linalg.solve(gram, y_centered)
    return x_val @ coefficient + y_mean, x_test @ coefficient + y_mean


def ridge_univariate(
    base: np.ndarray,
    target: np.ndarray,
    split: Split,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_targets = target.shape[1]
    validation = np.empty((len(split.validation), n_targets, 24), dtype=np.float64)
    test = np.empty((len(split.test), n_targets, 24), dtype=np.float64)
    for lead in range(24):
        for target_index in range(n_targets):
            x = base[:, target_index, lead : lead + 1].astype(np.float64)
            val, tst = ridge_multioutput(
                x,
                target[:, target_index, lead : lead + 1].astype(np.float64),
                split,
                alpha,
            )
            validation[:, target_index, lead] = val[:, 0]
            test[:, target_index, lead] = tst[:, 0]
    return validation, test


def fit_strategy(
    base: np.ndarray,
    target: np.ndarray,
    additive_history: np.ndarray,
    interaction_by_lead: dict[int, np.ndarray],
    split: Split,
    alpha: float,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "univariate":
        return ridge_univariate(base, target, split, alpha)
    validation = np.empty((len(split.validation), target.shape[1], 24), dtype=np.float64)
    test = np.empty((len(split.test), target.shape[1], 24), dtype=np.float64)
    for lead in range(24):
        x = build_features(
            base,
            additive_history,
            interaction_by_lead.get(
                lead, np.empty((len(base), 0), dtype=np.float64)
            ),
            lead,
            method,
        )
        val, tst = ridge_multioutput(
            x,
            target[:, :, lead].astype(np.float64),
            split,
            alpha,
        )
        validation[:, :, lead] = val
        test[:, :, lead] = tst
    return validation, test


def fit_target_pair_strategy(
    base: np.ndarray,
    target: np.ndarray,
    additive_history: np.ndarray,
    interaction_by_cell: dict[tuple[int, int], np.ndarray],
    split: Split,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    validation = np.empty((len(split.validation), target.shape[1], 24), dtype=np.float64)
    test = np.empty((len(split.test), target.shape[1], 24), dtype=np.float64)
    for target_index in range(target.shape[1]):
        for lead in range(24):
            x = np.concatenate(
                (
                    base[:, :, lead],
                    additive_history,
                    interaction_by_cell[(target_index, lead)],
                ),
                axis=1,
            )
            val, tst = ridge_multioutput(
                x,
                target[:, target_index, lead : lead + 1],
                split,
                alpha,
            )
            validation[:, target_index, lead] = val[:, 0]
            test[:, target_index, lead] = tst[:, 0]
    return validation, test


def select_target_pair_alpha(
    base: np.ndarray,
    target: np.ndarray,
    additive_history: np.ndarray,
    interaction_by_cell: dict[tuple[int, int], np.ndarray],
    split: Split,
    target_scale: np.ndarray,
    alpha_grid: list[float],
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for alpha in alpha_grid:
        validation, _ = fit_target_pair_strategy(
            base,
            target,
            additive_history,
            interaction_by_cell,
            split,
            alpha,
        )
        scores[str(alpha)] = mean_cell_nrmse(
            validation,
            target[split.validation],
            target_scale,
        )
    best = min(alpha_grid, key=lambda value: scores[str(value)])
    return best, scores


def select_alpha(
    base: np.ndarray,
    target: np.ndarray,
    additive_history: np.ndarray,
    interaction_by_lead: dict[int, np.ndarray],
    split: Split,
    target_scale: np.ndarray,
    alpha_grid: list[float],
    method: str,
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    best_alpha, best_score = alpha_grid[0], math.inf
    for alpha in alpha_grid:
        validation, _ = fit_strategy(
            base,
            target,
            additive_history,
            interaction_by_lead,
            split,
            alpha,
            method,
        )
        score = mean_cell_nrmse(
            validation,
            target[split.validation],
            target_scale,
        )
        scores[str(alpha)] = score
        if score < best_score:
            best_alpha, best_score = alpha, score
    return best_alpha, scores


def circular_block_indices(
    rng: np.random.Generator,
    n_samples: int,
    block_length: int,
) -> np.ndarray:
    count = int(math.ceil(n_samples / block_length))
    starts = rng.integers(0, n_samples, size=count)
    blocks = [
        (start + np.arange(block_length, dtype=int)) % n_samples for start in starts
    ]
    return np.concatenate(blocks)[:n_samples]


def bootstrap_gain(
    baseline: np.ndarray,
    treatment: np.ndarray,
    target: np.ndarray,
    target_scale: np.ndarray,
    replicates: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gains = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = circular_block_indices(rng, len(target), block_length)
        gains[index] = mean_cell_nrmse(
            baseline[sample], target[sample], target_scale
        ) - mean_cell_nrmse(treatment[sample], target[sample], target_scale)
    return gains


def method_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    target_scale: np.ndarray,
) -> dict[str, object]:
    rmse = cell_rmse(prediction, target, target_scale)
    return {
        "mean_cell_nrmse": float(rmse.mean()),
        "mean_cell_acc": mean_cell_acc(prediction, target),
        "target_nrmse": rmse.mean(axis=1).tolist(),
        "lead_nrmse": rmse.mean(axis=0).tolist(),
    }


def make_random_rankings(
    pairs: list[tuple[int, int]],
    top_k: int,
    repeats: int,
    seed: int,
) -> list[dict[int, list[tuple[int, int]]]]:
    rng = np.random.default_rng(seed)
    rankings = []
    for _ in range(repeats):
        rankings.append(
            {
                lead: [pairs[index] for index in rng.permutation(len(pairs))[:top_k]]
                for lead in range(24)
            }
        )
    return rankings


def interaction_map(
    history: np.ndarray,
    pair_ready: np.ndarray,
    rankings: dict[int, list[tuple[int, int]]],
) -> dict[int, np.ndarray]:
    return {
        lead: pair_features(history, pair_ready, rankings[lead])
        for lead in range(24)
    }


def target_interaction_map(
    history: np.ndarray,
    pair_ready: np.ndarray,
    rankings: dict[tuple[int, int], list[tuple[int, int]]],
    top_k: int,
) -> dict[tuple[int, int], np.ndarray]:
    return {
        cell: pair_features(history, pair_ready, ranking[:top_k])
        for cell, ranking in rankings.items()
    }


def make_random_target_rankings(
    pairs: list[tuple[int, int]],
    n_targets: int,
    top_k: int,
    repeats: int,
    seed: int,
) -> list[dict[tuple[int, int], list[tuple[int, int]]]]:
    rng = np.random.default_rng(seed)
    outputs = []
    for _ in range(repeats):
        outputs.append(
            {
                (target, lead): [
                    pairs[index]
                    for index in rng.permutation(len(pairs))[:top_k]
                ]
                for target in range(n_targets)
                for lead in range(24)
            }
        )
    return outputs


def prediction_for_gate(
    additive: np.ndarray,
    interaction: np.ndarray,
    gate: np.ndarray,
) -> np.ndarray:
    return np.where(gate[None, :, :], interaction, additive)


def plot_results(
    *,
    metrics: dict[str, dict[str, object]],
    mode_names: list[str],
    xi: np.ndarray,
    syn_gain: np.ndarray,
    random_pair_scores: np.ndarray,
    random_gate_scores: np.ndarray,
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
    methods = [
        "frozen",
        "univariate",
        "additive",
        "target_syn_pair",
        "validation_additive_selector",
        "validation_target_selector",
    ]
    labels = [
        "Frozen",
        "Univariate",
        "Additive",
        "Target Syn",
        "Calibrated selector",
        "Selector + Syn",
    ]
    values = [float(metrics[name]["mean_cell_nrmse"]) for name in methods]
    additive_score = float(metrics["additive"]["mean_cell_nrmse"])
    syn_score = float(metrics["target_syn_pair"]["mean_cell_nrmse"])
    xi_score = float(metrics["xi_gated"]["mean_cell_nrmse"])

    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.2), height_ratios=(1.0, 1.0))
    ax_method = fig.add_subplot(grid[0, 0])
    ax_target = fig.add_subplot(grid[0, 1])
    ax_pair_null = fig.add_subplot(grid[1, 0])
    ax_gate = fig.add_subplot(grid[1, 1])

    colors = [
        "#7A828A",
        "#8DA0CB",
        "#4C78A8",
        "#D9822B",
        "#3A8F70",
        "#D9822B",
    ]
    positions = np.arange(len(methods))
    ax_method.scatter(positions, values, c=colors, s=28, zorder=3)
    ax_method.plot(positions, values, color="0.75", lw=0.8, zorder=1)
    ax_method.set(
        ylabel="Test mean normalized RMSE",
        xticks=positions,
        xticklabels=labels,
    )
    ax_method.tick_params(axis="x", rotation=35)

    target_gain = (
        np.asarray(metrics["additive"]["target_nrmse"])
        - np.asarray(metrics["target_syn_pair"]["target_nrmse"])
    )
    order = np.argsort(target_gain)
    ax_target.axvline(0, color="0.6", lw=0.8)
    ax_target.scatter(target_gain[order], np.arange(len(mode_names)), color="#D9822B", s=22)
    ax_target.set(
        xlabel="Target-Syn gain over additive (normalized RMSE)",
        yticks=np.arange(len(mode_names)),
        yticklabels=np.asarray(mode_names)[order],
    )

    random_pair_gain = additive_score - random_pair_scores
    ax_pair_null.hist(
        random_pair_gain,
        bins=14,
        color="#B8C4D0",
        edgecolor="white",
        linewidth=0.4,
    )
    ax_pair_null.axvline(0, color="0.35", lw=0.8, ls="--")
    ax_pair_null.axvline(
        additive_score - syn_score,
        color="#D9822B",
        lw=1.6,
        label="Target-specific Syn",
    )
    ax_pair_null.set(
        xlabel="Pair-feature gain over additive",
        ylabel="Random rankings",
    )
    ax_pair_null.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        frameon=False,
    )

    gate_count = int(np.count_nonzero(xi >= np.quantile(xi, 0.70)))
    random_gate_gain = additive_score - random_gate_scores
    ax_gate.hist(
        random_gate_gain,
        bins=14,
        color="#BFD8CE",
        edgecolor="white",
        linewidth=0.4,
    )
    ax_gate.axvline(0, color="0.35", lw=0.8, ls="--")
    ax_gate.axvline(
        additive_score - xi_score,
        color="#3A8F70",
        lw=1.6,
        label=rf"Top-$\Xi$ gate ({gate_count} cells)",
    )
    ax_gate.set(
        xlabel=r"Gate gain over additive",
        ylabel="Random gates",
    )
    ax_gate.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        frameon=False,
    )

    for label, axis in zip("abcd", (ax_method, ax_target, ax_pair_null, ax_gate)):
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
            output_dir / f"synergy_guided_forecast.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    with np.load(input_dir / "model_inputs.npz", allow_pickle=False) as data:
        history = data["history"].astype(np.float64)
        targets = data["targets"].astype(np.float64)
        mode_names = data["mode_names"].astype(str).tolist()
    with np.load(input_dir / "modeformer_predictions.npz", allow_pickle=False) as data:
        predictions_by_seed = data["predictions_by_seed"].astype(np.float64)
        ensemble = data["ensemble_prediction"].astype(np.float64)
        prediction_targets = data["targets"].astype(np.float64)
        target_dates = data["target_dates"].astype(str)
    if not np.allclose(targets, prediction_targets):
        raise ValueError("Prediction and preprocessing targets disagree.")

    split = chronological_split(target_dates)
    target_mean, target_scale = target_scaling(targets, split.fit)
    target_scale_cells = target_scale.reshape(1, len(mode_names), 1)
    additive_history, pair_ready = history_summaries(history)
    pairs = all_pairs(len(mode_names))
    syn_rankings, pair_syn = load_pair_rankings(args.pair_syn, mode_names)
    target_syn_rankings = load_target_pair_rankings(args.target_pair_syn, mode_names)
    xi = load_target_xi(args.target_xi, mode_names)
    syn_top = {
        lead: ranking[: args.top_pairs] for lead, ranking in syn_rankings.items()
    }
    empty_interactions = {
        lead: np.empty((len(history), 0), dtype=np.float64) for lead in range(24)
    }
    syn_interactions = interaction_map(history, pair_ready, syn_top)
    target_syn_interactions = target_interaction_map(
        history,
        pair_ready,
        target_syn_rankings,
        args.top_pairs,
    )
    all_interactions = {
        lead: pair_features(history, pair_ready, pairs) for lead in range(24)
    }
    random_target_rankings = make_random_target_rankings(
        pairs,
        len(mode_names),
        args.top_pairs,
        args.random_repeats,
        args.seed,
    )

    alpha_grid = [float(value) for value in args.alpha_grid.split(",")]
    tuning: dict[str, object] = {}
    selected_alpha: dict[str, float] = {}
    for method, interactions in (
        ("univariate", empty_interactions),
        ("additive", empty_interactions),
        ("syn_pair", syn_interactions),
        ("all_pair", all_interactions),
    ):
        alpha, scores = select_alpha(
            ensemble,
            targets,
            additive_history,
            interactions,
            split,
            target_scale_cells,
            alpha_grid,
            method if method != "syn_pair" and method != "all_pair" else method,
        )
        selected_alpha[method] = alpha
        tuning[method] = scores
        print(f"{method}: alpha={alpha} validation_nRMSE={scores[str(alpha)]:.5f}")

    target_syn_alpha, target_syn_scores = select_target_pair_alpha(
        ensemble,
        targets,
        additive_history,
        target_syn_interactions,
        split,
        target_scale_cells,
        alpha_grid,
    )
    selected_alpha["target_syn_pair"] = target_syn_alpha
    selected_alpha["random_pair"] = target_syn_alpha
    tuning["target_syn_pair"] = target_syn_scores
    print(
        f"target_syn_pair: alpha={target_syn_alpha} "
        f"validation_nRMSE={target_syn_scores[str(target_syn_alpha)]:.5f}"
    )

    predictions_validation: dict[str, np.ndarray] = {}
    predictions_test: dict[str, np.ndarray] = {
        "frozen": ensemble[split.test].copy()
    }
    for method, interactions in (
        ("univariate", empty_interactions),
        ("additive", empty_interactions),
        ("syn_pair", syn_interactions),
        ("all_pair", all_interactions),
    ):
        val, tst = fit_strategy(
            ensemble,
            targets,
            additive_history,
            interactions,
            split,
            selected_alpha[method],
            method,
        )
        predictions_validation[method] = val
        predictions_test[method] = tst

    target_val, target_test = fit_target_pair_strategy(
        ensemble,
        targets,
        additive_history,
        target_syn_interactions,
        split,
        target_syn_alpha,
    )
    predictions_validation["target_syn_pair"] = target_val
    predictions_test["target_syn_pair"] = target_test

    random_test_predictions = []
    for ranking in random_target_rankings:
        interactions = target_interaction_map(
            history,
            pair_ready,
            ranking,
            args.top_pairs,
        )
        _, prediction = fit_target_pair_strategy(
            ensemble,
            targets,
            additive_history,
            interactions,
            split,
            target_syn_alpha,
        )
        random_test_predictions.append(prediction)
    random_test_predictions_array = np.stack(random_test_predictions)

    gate_count = int(math.ceil(args.gate_fraction * xi.size))
    xi_gate = np.zeros(xi.size, dtype=bool)
    xi_gate[np.argsort(xi.ravel())[-gate_count:]] = True
    xi_gate = xi_gate.reshape(xi.shape)
    predictions_test["xi_gated"] = prediction_for_gate(
        predictions_test["additive"], predictions_test["target_syn_pair"], xi_gate
    )

    val_add_cell = cell_rmse(
        predictions_validation["additive"],
        targets[split.validation],
        target_scale_cells,
    )
    val_syn_cell = cell_rmse(
        predictions_validation["target_syn_pair"],
        targets[split.validation],
        target_scale_cells,
    )
    validation_gate = np.zeros(xi.size, dtype=bool)
    validation_gain = (val_add_cell - val_syn_cell).ravel()
    validation_gate[np.argsort(validation_gain)[-gate_count:]] = True
    validation_gate = validation_gate.reshape(xi.shape)
    predictions_test["validation_gated"] = prediction_for_gate(
        predictions_test["additive"],
        predictions_test["target_syn_pair"],
        validation_gate,
    )

    def target_selector(
        candidate_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        validation_scores = np.stack(
            [
                cell_rmse(
                    predictions_validation[name],
                    targets[split.validation],
                    target_scale_cells,
                ).mean(axis=1)
                for name in candidate_names
            ]
        )
        choice = np.argmin(validation_scores, axis=0)
        selected = np.empty_like(predictions_test[candidate_names[0]])
        for target_index, candidate_index in enumerate(choice):
            selected[:, target_index] = predictions_test[
                candidate_names[int(candidate_index)]
            ][:, target_index]
        return selected, choice

    predictions_test["validation_additive_selector"], additive_choice = target_selector(
        ["univariate", "additive"]
    )
    predictions_test["validation_syn_selector"], syn_choice = target_selector(
        ["univariate", "target_syn_pair"]
    )
    predictions_test["validation_target_selector"], target_choice = target_selector(
        ["univariate", "additive", "target_syn_pair"]
    )

    rng = np.random.default_rng(args.seed + 17)
    random_gate_predictions = []
    for _ in range(args.random_repeats):
        gate = np.zeros(xi.size, dtype=bool)
        gate[rng.choice(xi.size, size=gate_count, replace=False)] = True
        gate = gate.reshape(xi.shape)
        random_gate_predictions.append(
            prediction_for_gate(
                predictions_test["additive"],
                predictions_test["target_syn_pair"],
                gate,
            )
        )
    random_gate_predictions_array = np.stack(random_gate_predictions)

    test_target = targets[split.test]
    metrics = {
        name: method_metrics(prediction, test_target, target_scale_cells)
        for name, prediction in predictions_test.items()
    }
    random_pair_scores = np.asarray(
        [
            mean_cell_nrmse(prediction, test_target, target_scale_cells)
            for prediction in random_test_predictions_array
        ]
    )
    random_gate_scores = np.asarray(
        [
            mean_cell_nrmse(prediction, test_target, target_scale_cells)
            for prediction in random_gate_predictions_array
        ]
    )

    seed_gains = []
    for seed_index, seed_prediction in enumerate(predictions_by_seed):
        _, additive_seed = fit_strategy(
            seed_prediction,
            targets,
            additive_history,
            empty_interactions,
            split,
            selected_alpha["additive"],
            "additive",
        )
        _, syn_seed = fit_target_pair_strategy(
            seed_prediction,
            targets,
            additive_history,
            target_syn_interactions,
            split,
            selected_alpha["target_syn_pair"],
        )
        seed_gains.append(
            {
                "checkpoint_seed": seed_index + 1,
                "target_syn_pair_gain_over_additive": mean_cell_nrmse(
                    additive_seed, test_target, target_scale_cells
                )
                - mean_cell_nrmse(syn_seed, test_target, target_scale_cells),
            }
        )

    bootstrap = bootstrap_gain(
        predictions_test["additive"],
        predictions_test["target_syn_pair"],
        test_target,
        target_scale_cells,
        args.bootstrap,
        args.bootstrap_block,
        args.seed + 100,
    )
    bootstrap_selector_increment = bootstrap_gain(
        predictions_test["validation_additive_selector"],
        predictions_test["validation_target_selector"],
        test_target,
        target_scale_cells,
        args.bootstrap,
        args.bootstrap_block,
        args.seed + 101,
    )
    syn_score = float(metrics["target_syn_pair"]["mean_cell_nrmse"])
    syn_percentile = float(
        (1 + np.count_nonzero(random_pair_scores <= syn_score))
        / (len(random_pair_scores) + 1)
    )
    xi_score = float(metrics["xi_gated"]["mean_cell_nrmse"])
    xi_percentile = float(
        (1 + np.count_nonzero(random_gate_scores <= xi_score))
        / (len(random_gate_scores) + 1)
    )
    report = {
        "status": "completed",
        "question": "Can frozen maximum-entropy synergy scores improve all-mode ORAS5 forecast correction?",
        "mode_names": mode_names,
        "samples": {
            "fit": int(len(split.fit)),
            "validation": int(len(split.validation)),
            "test": int(len(split.test)),
        },
        "issue_periods": {
            "fit": [str(split.issue_dates[split.fit[0]]), str(split.issue_dates[split.fit[-1]])],
            "validation": [
                str(split.issue_dates[split.validation[0]]),
                str(split.issue_dates[split.validation[-1]]),
            ],
            "test": [
                str(split.issue_dates[split.test[0]]),
                str(split.issue_dates[split.test[-1]]),
            ],
        },
        "top_pairs_per_lead": args.top_pairs,
        "target_gate_fraction": args.gate_fraction,
        "selected_alpha": selected_alpha,
        "validation_scores": tuning,
        "test_metrics": metrics,
        "random_pair_control": {
            "repeats": args.random_repeats,
            "scores": random_pair_scores.tolist(),
            "mean": float(random_pair_scores.mean()),
            "std": float(random_pair_scores.std(ddof=1)),
            "fraction_random_at_least_as_good_as_syn": syn_percentile,
        },
        "random_gate_control": {
            "repeats": args.random_repeats,
            "scores": random_gate_scores.tolist(),
            "mean": float(random_gate_scores.mean()),
            "std": float(random_gate_scores.std(ddof=1)),
            "fraction_random_at_least_as_good_as_xi": xi_percentile,
        },
        "target_syn_pair_gain_over_additive": {
            "observed": float(
                metrics["additive"]["mean_cell_nrmse"]
                - metrics["target_syn_pair"]["mean_cell_nrmse"]
            ),
            "bootstrap_ci95": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            "bootstrap_positive_fraction": float(np.mean(bootstrap > 0)),
            "checkpoint_seed_gains": seed_gains,
        },
        "validation_target_selection": {
            "candidate_order": ["univariate", "additive", "target_syn_pair"],
            "selected_by_target": {
                mode_names[index]: ["univariate", "additive", "target_syn_pair"][
                    int(choice)
                ]
                for index, choice in enumerate(target_choice)
            },
            "additive_only_selector_by_target": {
                mode_names[index]: ["univariate", "additive"][int(choice)]
                for index, choice in enumerate(additive_choice)
            },
            "syn_only_selector_by_target": {
                mode_names[index]: ["univariate", "target_syn_pair"][int(choice)]
                for index, choice in enumerate(syn_choice)
            },
            "synergy_increment_over_additive_selector": float(
                metrics["validation_additive_selector"]["mean_cell_nrmse"]
                - metrics["validation_target_selector"]["mean_cell_nrmse"]
            ),
            "synergy_increment_bootstrap_ci95": np.percentile(
                bootstrap_selector_increment, [2.5, 97.5]
            ).tolist(),
            "synergy_increment_bootstrap_positive_fraction": float(
                np.mean(bootstrap_selector_increment > 0)
            ),
        },
        "controls": {
            "score_source": "maximum-entropy intervention; no ORAS forecast errors",
            "feature_count_matched": True,
            "same_fit_validation_test": True,
            "test_used_for_selection": False,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "evaluation_arrays.npz",
        xi=xi,
        pair_syn=pair_syn,
        xi_gate=xi_gate,
        validation_gate=validation_gate,
        bootstrap_syn_gain=bootstrap,
        bootstrap_selector_increment=bootstrap_selector_increment,
        random_pair_scores=random_pair_scores,
        random_gate_scores=random_gate_scores,
        test_target=test_target.astype(np.float32),
        **{
            f"prediction_{name}": prediction.astype(np.float32)
            for name, prediction in predictions_test.items()
        },
    )
    plot_results(
        metrics=metrics,
        mode_names=mode_names,
        xi=xi,
        syn_gain=np.asarray(metrics["additive"]["target_nrmse"])
        - np.asarray(metrics["target_syn_pair"]["target_nrmse"]),
        random_pair_scores=random_pair_scores,
        random_gate_scores=random_gate_scores,
        output_dir=output_dir,
    )
    print(json.dumps(report["target_syn_pair_gain_over_additive"], indent=2))
    print(
        json.dumps(
            {
                "test_nrmse": {
                    name: value["mean_cell_nrmse"] for name, value in metrics.items()
                },
                "random_pair_p": syn_percentile,
                "random_gate_p": xi_percentile,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pair-syn", type=Path, default=DEFAULT_PAIR_SYN)
    parser.add_argument(
        "--target-pair-syn",
        type=Path,
        default=DEFAULT_TARGET_PAIR_SYN,
    )
    parser.add_argument("--target-xi", type=Path, default=DEFAULT_TARGET_XI)
    parser.add_argument("--top-pairs", type=int, default=8)
    parser.add_argument("--gate-fraction", type=float, default=0.30)
    parser.add_argument("--random-repeats", type=int, default=40)
    parser.add_argument(
        "--alpha-grid",
        default="0.01,0.1,1,10,100,1000",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-block", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
