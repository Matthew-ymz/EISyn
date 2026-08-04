#!/usr/bin/env python3
"""Exact affine-TM Shapley attribution for the first five Runge SLP PCs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = (
    ROOT
    / "results"
    / "runge_slp_daily_1948_2026_20260628"
    / "mlp_tm_ei_lag04"
    / "results"
    / "runge"
)
DEFAULT_SOURCE = (
    DEFAULT_RESULT_ROOT
    / "multistep_conditioned_ei_tm_exhaustive"
    / "source_samples_n4096.npy"
)
DEFAULT_ROLLOUT = (
    DEFAULT_RESULT_ROOT
    / "multistep_conditioned_ei_tm_forced_edges"
    / "rollout_predictions_H060_n4096.npy"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_ROOT / "slp_pc05_shapley_affine"
DEFAULT_FIGURE_STEM = ROOT / "fig" / "earth_slp_pc05_shapley"

HORIZONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60)
PC_LABELS = tuple(f"No.{index}" for index in range(5))
SIX_BLOCK_LABELS = (*PC_LABELS, "Others")
COLORS = ("#355F8A", "#4C84A8", "#63A4A3", "#94B96B", "#D89A49", "#9AA2AC")
INK = "#24303D"
GRID = "#E6EAF0"
MAIN_COVARIANCE_RIDGE = 1e-6
RIDGE_SENSITIVITY = (1e-8, 1e-6, 1e-4)
SYN_NONNEGATIVE_TOLERANCE_BITS = 1e-8
CLOSURE_TOLERANCE_BITS = 1e-10


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.3,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def standardize(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite two-dimensional array.")
    scale = values.std(axis=0, ddof=1)
    if np.any(scale <= 0.0):
        raise ValueError(f"{name} contains a constant dimension.")
    return (values - values.mean(axis=0)) / scale


def stable_logdet(matrix: np.ndarray, name: str) -> float:
    sign, value = np.linalg.slogdet(matrix)
    if sign <= 0.0 or not np.isfinite(value):
        minimum = float(np.linalg.eigvalsh(matrix).min())
        raise RuntimeError(f"{name} is not positive definite; minimum eigenvalue={minimum:.6g}.")
    return float(value)


def fit_affine_intervention_model(
    standardized_source: np.ndarray,
    target: np.ndarray,
    covariance_ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Y = X B + epsilon under the known independent intervention covariance."""

    standardized_target = standardize(target, "rollout target")
    coefficients, _, rank, _ = np.linalg.lstsq(
        standardized_source,
        standardized_target,
        rcond=None,
    )
    if rank != standardized_source.shape[1]:
        raise RuntimeError(
            f"Affine readout is rank deficient: rank={rank}, dimensions={standardized_source.shape[1]}."
        )
    residual = standardized_target - standardized_source @ coefficients
    residual_covariance = np.cov(residual, rowvar=False)
    residual_covariance += covariance_ridge * np.eye(residual_covariance.shape[0])
    stable_logdet(residual_covariance, "residual covariance")
    return coefficients, residual_covariance


def coalition_ei_bits(
    coefficients: np.ndarray,
    residual_covariance: np.ndarray,
    source_dimensions: tuple[int, ...],
) -> float:
    """Affine-Gaussian EI for a source subset with all other sources marginalized."""

    all_dimensions = np.arange(coefficients.shape[0])
    selected = np.zeros(coefficients.shape[0], dtype=bool)
    selected[list(source_dimensions)] = True
    full_target_covariance = coefficients.T @ coefficients + residual_covariance
    conditional_coefficients = coefficients[all_dimensions[~selected]]
    conditional_covariance = conditional_coefficients.T @ conditional_coefficients + residual_covariance
    return (
        0.5
        * (
            stable_logdet(full_target_covariance, "full target covariance")
            - stable_logdet(conditional_covariance, "conditional target covariance")
        )
        / math.log(2.0)
    )


def exact_shapley(values: dict[int, float], player_count: int) -> np.ndarray:
    attribution = np.zeros(player_count, dtype=np.float64)
    normalization = math.factorial(player_count)
    for player in range(player_count):
        for coalition in range(1 << player_count):
            if coalition & (1 << player):
                continue
            size = coalition.bit_count()
            weight = (
                math.factorial(size)
                * math.factorial(player_count - size - 1)
                / normalization
            )
            attribution[player] += weight * (
                values[coalition | (1 << player)] - values[coalition]
            )
    return attribution


def audit_nonnegative(
    values: dict[int, float],
    player_count: int,
    horizon: int,
    game_name: str,
) -> tuple[dict[int, float], int, float]:
    audited = dict(values)
    tested = [
        value
        for mask, value in values.items()
        if mask.bit_count() >= 2
    ]
    minimum = float(min(tested))
    violating = [value for value in tested if value < -SYN_NONNEGATIVE_TOLERANCE_BITS]
    if violating:
        raise RuntimeError(
            f"{game_name} at H={horizon} violates Syn nonnegativity: "
            f"minimum={minimum:.12g} bits, tolerance={SYN_NONNEGATIVE_TOLERANCE_BITS:.12g} bits, "
            f"count={len(violating)}."
        )
    zero_count = 0
    for mask, value in tuple(audited.items()):
        if mask.bit_count() >= 2 and value < 0.0:
            audited[mask] = 0.0
            zero_count += 1
    if len(audited) != 1 << player_count:
        raise RuntimeError("Coalition table is incomplete.")
    return audited, zero_count, minimum


def evaluate_game(
    coefficients: np.ndarray,
    residual_covariance: np.ndarray,
    blocks: tuple[tuple[int, ...], ...],
    labels: tuple[str, ...],
    horizon: int,
    game_name: str,
) -> tuple[dict[str, object], dict[str, float | int]]:
    player_count = len(blocks)
    ei_values: dict[int, float] = {}
    interaction_values: dict[int, float] = {}
    dimensions_by_mask: dict[int, tuple[int, ...]] = {}

    for mask in range(1 << player_count):
        dimensions = tuple(
            itertools.chain.from_iterable(
                blocks[player]
                for player in range(player_count)
                if mask & (1 << player)
            )
        )
        dimensions_by_mask[mask] = dimensions
        ei_values[mask] = (
            0.0
            if mask == 0
            else coalition_ei_bits(coefficients, residual_covariance, dimensions)
        )

    for mask in range(1 << player_count):
        interaction_values[mask] = ei_values[mask] - sum(
            ei_values[1 << player]
            for player in range(player_count)
            if mask & (1 << player)
        )

    interaction_values, zero_count, minimum = audit_nonnegative(
        interaction_values,
        player_count,
        horizon,
        game_name,
    )
    attribution = exact_shapley(interaction_values, player_count)
    grand_mask = (1 << player_count) - 1
    total_interaction = interaction_values[grand_mask]
    closure_error = float(attribution.sum() - total_interaction)
    if abs(closure_error) > CLOSURE_TOLERANCE_BITS:
        raise RuntimeError(
            f"{game_name} Shapley closure fails at H={horizon}: {closure_error:.12g} bits."
        )
    if total_interaction <= 0.0:
        raise RuntimeError(f"{game_name} has no positive grand-coalition interaction at H={horizon}.")
    if float(attribution.min()) < -SYN_NONNEGATIVE_TOLERANCE_BITS:
        raise RuntimeError(
            f"{game_name} has a negative Shapley contribution at H={horizon}: "
            f"minimum={float(attribution.min()):.12g} bits."
        )

    percentages = 100.0 * attribution / total_interaction
    record = {
        "horizon": horizon,
        "grand_coalition_ei_bits": ei_values[grand_mask],
        "singleton_ei_bits": {
            labels[player]: ei_values[1 << player]
            for player in range(player_count)
        },
        "grand_coalition_interaction_bits": total_interaction,
        "shapley_bits": {
            labels[player]: float(attribution[player])
            for player in range(player_count)
        },
        "shapley_percent": {
            labels[player]: float(percentages[player])
            for player in range(player_count)
        },
        "closure_error_bits": closure_error,
    }
    audit = {
        "numerical_zero_affected_count": zero_count,
        "minimum_interaction_bits": minimum,
        "closure_error_bits": closure_error,
    }
    return record, audit


def evaluate_all(
    source: np.ndarray,
    rollout: np.ndarray,
    covariance_ridge: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float | int]]:
    standardized_source = standardize(source, "source samples")
    five_blocks = tuple((index,) for index in range(5))
    six_blocks = (*five_blocks, tuple(range(5, source.shape[1])))
    five_records: list[dict[str, object]] = []
    six_records: list[dict[str, object]] = []
    audits: list[dict[str, float | int]] = []

    for horizon in HORIZONS:
        coefficients, residual_covariance = fit_affine_intervention_model(
            standardized_source,
            rollout[:, horizon - 1, :],
            covariance_ridge,
        )
        five_record, five_audit = evaluate_game(
            coefficients,
            residual_covariance,
            five_blocks,
            PC_LABELS,
            horizon,
            "five-PC game",
        )
        six_record, six_audit = evaluate_game(
            coefficients,
            residual_covariance,
            six_blocks,
            SIX_BLOCK_LABELS,
            horizon,
            "five-PC-plus-Others game",
        )
        five_records.append(five_record)
        six_records.append(six_record)
        audits.extend((five_audit, six_audit))

    return five_records, six_records, {
        "syn_nonnegative_tolerance_bits": SYN_NONNEGATIVE_TOLERANCE_BITS,
        "closure_tolerance_bits": CLOSURE_TOLERANCE_BITS,
        "minimum_interaction_bits": float(min(audit["minimum_interaction_bits"] for audit in audits)),
        "numerical_zero_affected_count": int(
            sum(audit["numerical_zero_affected_count"] for audit in audits)
        ),
        "maximum_absolute_closure_error_bits": float(
            max(abs(audit["closure_error_bits"]) for audit in audits)
        ),
        "significant_nonnegativity_violation_count": 0,
    }


def record_matrix(
    records: list[dict[str, object]],
    labels: tuple[str, ...],
    field: str,
) -> np.ndarray:
    return np.asarray(
        [[record[field][label] for label in labels] for record in records],
        dtype=np.float64,
    )


def ridge_sensitivity(
    source: np.ndarray,
    rollout: np.ndarray,
) -> dict[str, object]:
    by_ridge: dict[str, dict[str, object]] = {}
    five_percentages: list[np.ndarray] = []
    six_percentages: list[np.ndarray] = []
    for ridge in RIDGE_SENSITIVITY:
        five, six, audit = evaluate_all(source, rollout, ridge)
        five_matrix = record_matrix(five, PC_LABELS, "shapley_percent")
        six_matrix = record_matrix(six, SIX_BLOCK_LABELS, "shapley_percent")
        five_percentages.append(five_matrix)
        six_percentages.append(six_matrix)
        by_ridge[f"{ridge:.0e}"] = {
            "five_pc_key_horizons": {
                str(horizon): five[HORIZONS.index(horizon)]["shapley_percent"]
                for horizon in (1, 10, 60)
            },
            "six_block_key_horizons": {
                str(horizon): six[HORIZONS.index(horizon)]["shapley_percent"]
                for horizon in (1, 10, 60)
            },
            "audit": audit,
        }
    five_stack = np.stack(five_percentages)
    six_stack = np.stack(six_percentages)
    return {
        "covariance_ridges": list(RIDGE_SENSITIVITY),
        "maximum_five_pc_percentage_point_range": float(
            (five_stack.max(axis=0) - five_stack.min(axis=0)).max()
        ),
        "maximum_six_block_percentage_point_range": float(
            (six_stack.max(axis=0) - six_stack.min(axis=0)).max()
        ),
        "by_ridge": by_ridge,
    }


def high_order_feasibility(sample_count: int) -> dict[str, object]:
    previous_coordinates = 64  # five-PC full coalition plus 60-dimensional target
    rows = []
    for degree in (1, 2, 3):
        feature_count = math.comb(previous_coordinates + degree, degree)
        if degree == 1:
            decision = "used_as_affine_logdet_equivalent"
        elif degree == 2:
            decision = "not_run_insufficient_rows_per_feature_for_robust_full_target_validation"
        else:
            decision = "not_identifiable_feature_count_exceeds_sample_count"
        rows.append(
            {
                "degree": degree,
                "maximum_triangular_stage_feature_count": feature_count,
                "samples_per_feature": sample_count / feature_count,
                "decision": decision,
            }
        )
    return {
        "scope": "five-PC grand coalition plus the fixed 60-dimensional future SLP target",
        "joint_dimension": 65,
        "sample_count": sample_count,
        "basis_count_definition": "number of monomials of total degree <= q in 64 preceding coordinates",
        "assessment": rows,
        "conclusion": (
            "Degree 2 has only 1.91 samples per final-stage basis term and degree 3 has more "
            "basis terms than samples. A full-target nonlinear-TM check was therefore not run; "
            "changing to a scalar target would answer a different attribution question."
        ),
    }


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.03,
        label,
        transform=axis.transAxes,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="#111111",
    )


def format_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_xticks((1, 10, 20, 30, 40, 50, 60))
    axis.set_xlim(1, 60)


def plot_figure(
    five_records: list[dict[str, object]],
    six_records: list[dict[str, object]],
    figure_stem: Path,
) -> None:
    configure_matplotlib()
    horizons = np.asarray(HORIZONS)
    five_bits = record_matrix(five_records, PC_LABELS, "shapley_bits")
    five_percent = record_matrix(five_records, PC_LABELS, "shapley_percent")
    six_percent = record_matrix(six_records, SIX_BLOCK_LABELS, "shapley_percent")
    five_total = np.asarray(
        [record["grand_coalition_interaction_bits"] for record in five_records]
    )
    six_total = np.asarray(
        [record["grand_coalition_interaction_bits"] for record in six_records]
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.25),
        constrained_layout=True,
        gridspec_kw={"hspace": 0.12, "wspace": 0.14},
    )
    figure.get_layout_engine().set(w_pad=0.05, h_pad=0.05, hspace=0.12, wspace=0.12)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    ax_a.stackplot(horizons, five_percent.T, colors=COLORS[:5], alpha=0.94)
    ax_a.set_ylim(0, 100)
    ax_a.set_ylabel("Shapley composition (%)")
    ax_a.set_title("Five-PC exact game", fontsize=7.6, pad=3)
    format_axis(ax_a)
    add_panel_label(ax_a, "a")

    for index, label in enumerate(PC_LABELS):
        ax_b.plot(
            horizons,
            five_bits[:, index],
            color=COLORS[index],
            linewidth=1.5,
            marker="o",
            markersize=2.5,
            label=label,
        )
    ax_b.set_ylabel("Shapley contribution (bits)")
    ax_b.set_title("Absolute attribution", fontsize=7.6, pad=3)
    format_axis(ax_b)
    add_panel_label(ax_b, "b")

    ax_c.stackplot(horizons, six_percent.T, colors=COLORS, alpha=0.94)
    ax_c.set_ylim(0, 100)
    ax_c.set_xlabel("Forecast horizon, H")
    ax_c.set_ylabel("Shapley composition (%)")
    ax_c.set_title("Five PCs plus Others block", fontsize=7.6, pad=3)
    format_axis(ax_c)
    add_panel_label(ax_c, "c")

    ax_d.plot(
        horizons,
        five_total,
        color=INK,
        linewidth=1.7,
        marker="o",
        markersize=2.7,
        label="No.0–4 interaction",
    )
    ax_d.plot(
        horizons,
        six_total,
        color="#7C8795",
        linewidth=1.7,
        marker="s",
        markersize=2.6,
        label="No.0–4 + Others interaction",
    )
    ax_d.set_yscale("log")
    ax_d.set_xlabel("Forecast horizon, H")
    ax_d.set_ylabel("Grand-coalition interaction (bits)")
    ax_d.set_title("Interaction mass across scales", fontsize=7.6, pad=3)
    format_axis(ax_d)
    ax_d.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        fontsize=5.8,
        handlelength=2.2,
    )
    add_panel_label(ax_d, "d")

    handles = [Patch(facecolor=COLORS[index], label=label) for index, label in enumerate(SIX_BLOCK_LABELS)]
    figure.legend(
        handles=handles,
        loc="outside upper center",
        ncol=6,
        fontsize=6.2,
        columnspacing=1.2,
        handlelength=1.3,
        bbox_to_anchor=(0.5, 1.015),
    )

    figure_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(
            figure_stem.with_suffix(f".{suffix}"),
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def key_findings(
    five_records: list[dict[str, object]],
    six_records: list[dict[str, object]],
) -> dict[str, object]:
    findings: dict[str, object] = {}
    for horizon in (1, 10, 60):
        index = HORIZONS.index(horizon)
        five = five_records[index]
        six = six_records[index]
        leader = max(five["shapley_percent"], key=five["shapley_percent"].get)
        findings[str(horizon)] = {
            "five_pc_interaction_bits": five["grand_coalition_interaction_bits"],
            "five_pc_leader": leader,
            "five_pc_leader_percent": five["shapley_percent"][leader],
            "five_pc_percentages": five["shapley_percent"],
            "six_block_interaction_bits": six["grand_coalition_interaction_bits"],
            "others_percent": six["shapley_percent"]["Others"],
        }
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--rollout", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-stem", type=Path, default=DEFAULT_FIGURE_STEM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = np.load(args.source)
    rollout = np.load(args.rollout)
    if source.shape != (4096, 60):
        raise ValueError(f"Expected source shape (4096, 60), received {source.shape}.")
    if rollout.shape != (4096, 60, 60):
        raise ValueError(f"Expected rollout shape (4096, 60, 60), received {rollout.shape}.")
    if not np.isfinite(source).all() or not np.isfinite(rollout).all():
        raise ValueError("Source or rollout cache contains non-finite values.")

    five_records, six_records, audit = evaluate_all(
        source,
        rollout,
        MAIN_COVARIANCE_RIDGE,
    )
    sensitivity = ridge_sensitivity(source, rollout)
    feasibility = high_order_feasibility(source.shape[0])
    plot_figure(five_records, six_records, args.figure_stem)

    payload = {
        "analysis": "Runge SLP first-five-PC exact Shapley attribution",
        "inputs": {
            "source_samples": str(args.source),
            "source_samples_sha256": sha256(args.source),
            "rollout_predictions": str(args.rollout),
            "rollout_predictions_sha256": sha256(args.rollout),
            "source_shape": list(source.shape),
            "rollout_shape": list(rollout.shape),
        },
        "method": {
            "horizons": list(HORIZONS),
            "target": "all 60 future SLP components at each horizon",
            "source_prior": "independent bounded-uniform maximum-entropy intervention coordinates",
            "affine_equivalent_source_covariance": "identity after per-coordinate standardization",
            "readout": "affine degree-1 TM / linear-Gaussian log-det equivalent",
            "main_covariance_ridge": MAIN_COVARIANCE_RIDGE,
            "five_pc_players": list(PC_LABELS),
            "six_block_players": list(SIX_BLOCK_LABELS),
            "five_pc_coalition_count": 32,
            "six_block_coalition_count": 64,
            "coalition_value": "v(S) = EI(S -> Y) - sum_i EI({i} -> Y)",
            "percentage_denominator": "v(N) within the corresponding game",
        },
        "five_pc_game": five_records,
        "six_block_game": six_records,
        "audit": audit,
        "ridge_sensitivity": sensitivity,
        "high_order_feasibility": feasibility,
        "key_findings": key_findings(five_records, six_records),
        "figure_files": [
            str(args.figure_stem.with_suffix(f".{suffix}"))
            for suffix in ("png", "svg", "pdf")
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "key_findings": payload["key_findings"]}, indent=2))


if __name__ == "__main__":
    main()
