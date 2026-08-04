#!/usr/bin/env python3
"""Exact 11-player Shapley attribution of UniCM integrated EI increments."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.unicm_peid_syn_analysis import (  # noqa: E402
    HISTORY_LENGTH,
    MODE_NAMES,
    overall_prediction_cache_path,
    sample_full_history_mode_inputs,
)


MODE_LABELS = tuple(MODE_NAMES)
CORE_LABELS = ("nino", "IOD", "nino12", "nino3", "nino4")
LEADS = tuple(range(1, 25))
SEEDS = (1, 2, 3)
N_SAMPLES = 8192
SAMPLING_SEED = 20260619
INTERVENTION_BOUND = 4.0
START_MONTH = 0
DEVICE = "cpu"
MAIN_COVARIANCE_RIDGE = 1e-6
RIDGE_SENSITIVITY = (1e-8, 1e-6, 1e-4)
SYN_NONNEGATIVE_TOLERANCE_BITS = 1e-8
CLOSURE_TOLERANCE_BITS = 1e-10
DEFAULT_CACHE_DIR = ROOT / "results" / "unicm_overall_ei_cpu_bound4_n8192" / "cache"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "unicm_11mode_shapley_affine"
DEFAULT_FIGURE_STEM = ROOT / "fig" / "earth_unicm_11mode_shapley"

COLORS = (
    "#4C78A8",
    "#72A0C1",
    "#59A14F",
    "#8DBB72",
    "#E28E2B",
    "#F2B45C",
    "#B279A2",
    "#D49AC2",
    "#E15759",
    "#F28E8B",
    "#79706E",
)
INK = "#26323E"
GRID = "#E6EAF0"


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
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite two-dimensional array.")
    scale = array.std(axis=0, ddof=1)
    if np.any(scale <= 0.0):
        raise ValueError(f"{name} contains a constant dimension.")
    return (array - array.mean(axis=0)) / scale


def stable_logdet(matrix: np.ndarray, name: str) -> float:
    sign, value = np.linalg.slogdet(np.asarray(matrix, dtype=np.float64))
    if sign <= 0.0 or not np.isfinite(value):
        minimum = float(np.linalg.eigvalsh(matrix).min())
        raise RuntimeError(f"{name} is not positive definite; minimum eigenvalue={minimum:.6g}.")
    return float(value)


def mode_feature_blocks(
    *,
    history_length: int = HISTORY_LENGTH,
    mode_count: int = len(MODE_LABELS),
) -> tuple[tuple[int, ...], ...]:
    """Return month-major flattened columns grouped by climate mode."""

    return tuple(
        tuple(month * int(mode_count) + mode for month in range(int(history_length)))
        for mode in range(int(mode_count))
    )


def fit_affine_readout(
    standardized_source: np.ndarray,
    target: np.ndarray,
    covariance_ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    standardized_target = standardize(target, "UniCM target")
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
    residual_covariance += float(covariance_ridge) * np.eye(residual_covariance.shape[0])
    stable_logdet(residual_covariance, "residual covariance")
    return coefficients, residual_covariance


def exact_shapley(values: dict[int, float], player_count: int) -> np.ndarray:
    expected = 1 << int(player_count)
    if set(values) != set(range(expected)):
        raise ValueError("Coalition table is incomplete.")
    attribution = np.zeros(int(player_count), dtype=np.float64)
    normalization = math.factorial(int(player_count))
    for player in range(int(player_count)):
        player_bit = 1 << player
        for coalition in range(expected):
            if coalition & player_bit:
                continue
            size = coalition.bit_count()
            weight = (
                math.factorial(size)
                * math.factorial(int(player_count) - size - 1)
                / normalization
            )
            attribution[player] += weight * (
                values[coalition | player_bit] - values[coalition]
            )
    return attribution


def coalition_ei_table(
    coefficients: np.ndarray,
    residual_covariance: np.ndarray,
    blocks: Sequence[Sequence[int]],
) -> dict[int, float]:
    """Evaluate all coalitions under the known independent intervention covariance."""

    group_grams = np.stack(
        [coefficients[np.asarray(block, dtype=int)].T @ coefficients[np.asarray(block, dtype=int)] for block in blocks]
    )
    full_target_covariance = residual_covariance + group_grams.sum(axis=0)
    full_logdet = stable_logdet(full_target_covariance, "full target covariance")
    player_count = len(blocks)
    full_mask = (1 << player_count) - 1
    table = {0: 0.0}
    for mask in range(1, 1 << player_count):
        omitted = [index for index in range(player_count) if not mask & (1 << index)]
        conditional = residual_covariance.copy()
        if omitted:
            conditional += group_grams[omitted].sum(axis=0)
        table[mask] = 0.5 * (
            full_logdet - stable_logdet(conditional, f"conditional covariance mask={mask}")
        ) / math.log(2.0)
    if table[full_mask] <= 0.0:
        raise RuntimeError("Grand-coalition EI must be positive.")
    return table


def evaluate_game(
    coefficients: np.ndarray,
    residual_covariance: np.ndarray,
    *,
    seed: int,
    lead: int,
) -> tuple[dict[str, object], dict[str, float | int]]:
    player_count = len(MODE_LABELS)
    ei_values = coalition_ei_table(coefficients, residual_covariance, mode_feature_blocks())
    singleton_values = np.asarray([ei_values[1 << player] for player in range(player_count)])
    interaction_values = {
        mask: ei_values[mask]
        - sum(singleton_values[player] for player in range(player_count) if mask & (1 << player))
        for mask in range(1 << player_count)
    }

    tested = [value for mask, value in interaction_values.items() if mask.bit_count() >= 2]
    minimum = float(min(tested))
    violations = [value for value in tested if value < -SYN_NONNEGATIVE_TOLERANCE_BITS]
    if violations:
        raise RuntimeError(
            f"UniCM seed={seed}, lead={lead} violates Syn nonnegativity: "
            f"minimum={minimum:.12g} bits, tolerance={SYN_NONNEGATIVE_TOLERANCE_BITS:.12g} bits, "
            f"count={len(violations)}."
        )
    numerical_zero_count = sum(
        1 for value in tested if -SYN_NONNEGATIVE_TOLERANCE_BITS <= value < 0.0
    )
    if numerical_zero_count:
        raise RuntimeError(
            "A tolerance-range negative interaction was encountered. The analysis does not silently project it to zero."
        )

    attribution = exact_shapley(interaction_values, player_count)
    full_mask = (1 << player_count) - 1
    total = float(interaction_values[full_mask])
    closure_error = float(attribution.sum() - total)
    if abs(closure_error) > CLOSURE_TOLERANCE_BITS:
        raise RuntimeError(
            f"Shapley closure fails for seed={seed}, lead={lead}: {closure_error:.12g} bits."
        )
    if total <= 0.0:
        raise RuntimeError(f"Grand-coalition interaction is not positive for seed={seed}, lead={lead}.")
    if float(attribution.min()) < -SYN_NONNEGATIVE_TOLERANCE_BITS:
        raise RuntimeError(
            f"Negative Shapley contribution for seed={seed}, lead={lead}: {float(attribution.min()):.12g} bits."
        )
    percentages = 100.0 * attribution / total
    return (
        {
            "seed": int(seed),
            "lead": int(lead),
            "grand_coalition_ei_bits": float(ei_values[full_mask]),
            "singleton_ei_sum_bits": float(singleton_values.sum()),
            "grand_coalition_interaction_bits": total,
            "shapley_bits": {
                label: float(attribution[index]) for index, label in enumerate(MODE_LABELS)
            },
            "shapley_percent": {
                label: float(percentages[index]) for index, label in enumerate(MODE_LABELS)
            },
            "closure_error_bits": closure_error,
            "percentage_sum": float(percentages.sum()),
        },
        {
            "minimum_interaction_bits": minimum,
            "numerical_zero_affected_count": numerical_zero_count,
            "closure_error_bits": closure_error,
            "minimum_shapley_bits": float(attribution.min()),
        },
    )


def cache_path(cache_dir: Path, seed: int) -> Path:
    args = argparse.Namespace(
        n_samples=N_SAMPLES,
        sampling_seed=SAMPLING_SEED,
        intervention_bound=INTERVENTION_BOUND,
        start_month=START_MONTH,
        device=DEVICE,
    )
    return overall_prediction_cache_path(cache_dir, seed=int(seed), args=args)


def evaluate_all(
    source: np.ndarray,
    cache_dir: Path,
    covariance_ridge: float,
) -> tuple[list[dict[str, object]], dict[str, float | int]]:
    standardized_source = standardize(source.reshape(source.shape[0], -1), "UniCM source")
    records: list[dict[str, object]] = []
    audits: list[dict[str, float | int]] = []
    for seed in SEEDS:
        path = cache_path(cache_dir, seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing UniCM prediction cache: {path}")
        with np.load(path, allow_pickle=False) as payload:
            predictions = np.asarray(payload["all_mode_targets"], dtype=np.float64)
        if predictions.shape != (N_SAMPLES, len(LEADS), len(MODE_LABELS)):
            raise ValueError(f"Unexpected prediction shape {predictions.shape} in {path}.")
        for lead in LEADS:
            coefficients, residual_covariance = fit_affine_readout(
                standardized_source,
                predictions[:, lead - 1, :],
                covariance_ridge,
            )
            record, audit = evaluate_game(
                coefficients,
                residual_covariance,
                seed=seed,
                lead=lead,
            )
            records.append(record)
            audits.append(audit)
    return records, {
        "syn_nonnegative_tolerance_bits": SYN_NONNEGATIVE_TOLERANCE_BITS,
        "closure_tolerance_bits": CLOSURE_TOLERANCE_BITS,
        "minimum_interaction_bits": float(min(audit["minimum_interaction_bits"] for audit in audits)),
        "numerical_zero_affected_count": int(
            sum(audit["numerical_zero_affected_count"] for audit in audits)
        ),
        "minimum_shapley_bits": float(min(audit["minimum_shapley_bits"] for audit in audits)),
        "maximum_absolute_closure_error_bits": float(
            max(abs(audit["closure_error_bits"]) for audit in audits)
        ),
        "significant_nonnegativity_violation_count": 0,
    }


def record_matrix(
    records: Sequence[dict[str, object]],
    field: str,
) -> np.ndarray:
    return np.asarray(
        [
            [record[field][label] for label in MODE_LABELS]
            for record in sorted(records, key=lambda row: (int(row["seed"]), int(row["lead"])))
        ],
        dtype=np.float64,
    ).reshape(len(SEEDS), len(LEADS), len(MODE_LABELS))


def aggregate_records(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    percent = record_matrix(records, "shapley_percent")
    bits = record_matrix(records, "shapley_bits")
    by_key = {(int(record["seed"]), int(record["lead"])): record for record in records}
    rows = []
    for lead_index, lead in enumerate(LEADS):
        totals = np.asarray(
            [by_key[(seed, lead)]["grand_coalition_interaction_bits"] for seed in SEEDS],
            dtype=float,
        )
        rows.append(
            {
                "lead": lead,
                "interaction_mean_bits": float(totals.mean()),
                "interaction_std_bits": float(totals.std(ddof=1)),
                "shapley_percent_mean": {
                    label: float(percent[:, lead_index, index].mean())
                    for index, label in enumerate(MODE_LABELS)
                },
                "shapley_percent_std": {
                    label: float(percent[:, lead_index, index].std(ddof=1))
                    for index, label in enumerate(MODE_LABELS)
                },
                "shapley_bits_mean": {
                    label: float(bits[:, lead_index, index].mean())
                    for index, label in enumerate(MODE_LABELS)
                },
                "shapley_bits_std": {
                    label: float(bits[:, lead_index, index].std(ddof=1))
                    for index, label in enumerate(MODE_LABELS)
                },
            }
        )
    return rows


def aggregate_matrix(aggregate: Sequence[dict[str, object]], field: str) -> np.ndarray:
    return np.asarray(
        [[record[field][label] for label in MODE_LABELS] for record in aggregate],
        dtype=np.float64,
    )


def ridge_sensitivity(
    source: np.ndarray,
    cache_dir: Path,
    *,
    main_records: Sequence[dict[str, object]],
    main_audit: dict[str, float | int],
) -> dict[str, object]:
    matrices = []
    by_ridge = {}
    for ridge in RIDGE_SENSITIVITY:
        if ridge == MAIN_COVARIANCE_RIDGE:
            records, audit = list(main_records), dict(main_audit)
        else:
            records, audit = evaluate_all(source, cache_dir, ridge)
        aggregate = aggregate_records(records)
        matrix = aggregate_matrix(aggregate, "shapley_percent_mean")
        matrices.append(matrix)
        by_ridge[f"{ridge:.0e}"] = {
            "key_leads": {
                str(lead): aggregate[LEADS.index(lead)]["shapley_percent_mean"]
                for lead in (1, 8, 12, 24)
            },
            "audit": audit,
        }
    stack = np.stack(matrices)
    ranges = stack.max(axis=0) - stack.min(axis=0)
    return {
        "covariance_ridges": list(RIDGE_SENSITIVITY),
        "maximum_percentage_point_range": float(ranges.max()),
        "maximum_key_lead_percentage_point_range": float(
            ranges[[LEADS.index(lead) for lead in (1, 8, 12, 24)]].max()
        ),
        "by_ridge": by_ridge,
    }


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.11,
        0.99,
        label,
        transform=axis.transAxes,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="#111111",
    )


def format_lead_axis(axis: plt.Axes) -> None:
    axis.set_xlim(1, 24)
    axis.set_xticks((1, 4, 8, 12, 16, 20, 24))
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)


def plot_figure(
    records: Sequence[dict[str, object]],
    aggregate: Sequence[dict[str, object]],
    figure_stem: Path,
) -> None:
    configure_matplotlib()
    leads = np.asarray(LEADS)
    percent_mean = aggregate_matrix(aggregate, "shapley_percent_mean")
    bits_mean = aggregate_matrix(aggregate, "shapley_bits_mean")
    by_key = {(int(record["seed"]), int(record["lead"])): record for record in records}
    totals = np.asarray(
        [
            [by_key[(seed, lead)]["grand_coalition_interaction_bits"] for lead in LEADS]
            for seed in SEEDS
        ],
        dtype=float,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.4, 5.35),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (0.95, 1.05), "hspace": 0.12, "wspace": 0.16},
    )
    figure.get_layout_engine().set(w_pad=0.05, h_pad=0.05, hspace=0.12, wspace=0.12)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    ax_a.stackplot(leads, percent_mean.T, colors=COLORS, alpha=0.94)
    ax_a.set_ylim(0, 100)
    ax_a.set_ylabel("Mean Shapley composition (%)")
    format_lead_axis(ax_a)
    add_panel_label(ax_a, "a")

    image = ax_b.imshow(
        percent_mean.T,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap="YlGnBu",
        extent=(0.5, 24.5, len(MODE_LABELS) - 0.5, -0.5),
    )
    leaders = np.argmax(percent_mean, axis=1)
    ax_b.scatter(leads, leaders, marker="o", s=7, facecolor="white", edgecolor=INK, linewidth=0.35)
    ax_b.set_xticks((1, 4, 8, 12, 16, 20, 24))
    ax_b.set_yticks(np.arange(len(MODE_LABELS)), MODE_LABELS)
    ax_b.set_xlabel("Forecast lead (months)")
    ax_b.set_ylabel("Source mode")
    colorbar = figure.colorbar(image, ax=ax_b, fraction=0.045, pad=0.025)
    colorbar.set_label("Mean share (%)")
    add_panel_label(ax_b, "b")

    for index, label in enumerate(MODE_LABELS):
        ax_c.plot(
            leads,
            bits_mean[:, index],
            color=COLORS[index],
            linewidth=1.25,
            marker="o",
            markersize=2.0,
            label=label,
        )
    ax_c.set_xlabel("Forecast lead (months)")
    ax_c.set_ylabel("Mean Shapley contribution (bits)")
    format_lead_axis(ax_c)
    add_panel_label(ax_c, "c")

    for seed_index, seed in enumerate(SEEDS):
        ax_d.plot(
            leads,
            totals[seed_index],
            color="#AAB2BB",
            linewidth=0.75,
            alpha=0.75,
        )
    total_mean = totals.mean(axis=0)
    total_std = totals.std(axis=0, ddof=1)
    ax_d.fill_between(
        leads,
        total_mean - total_std,
        total_mean + total_std,
        color="#4C78A8",
        alpha=0.16,
        linewidth=0,
    )
    ax_d.plot(leads, total_mean, color=INK, linewidth=1.7, marker="o", markersize=2.5)
    ax_d.set_xlabel("Forecast lead (months)")
    ax_d.set_ylabel(r"Grand-coalition interaction, $v(N)$ (bits)")
    format_lead_axis(ax_d)
    add_panel_label(ax_d, "d")

    handles = [Patch(facecolor=COLORS[index], label=label) for index, label in enumerate(MODE_LABELS)]
    figure.legend(
        handles=handles,
        loc="outside upper center",
        ncol=len(MODE_LABELS),
        fontsize=5.8,
        columnspacing=0.72,
        handlelength=1.0,
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
    records: Sequence[dict[str, object]],
    aggregate: Sequence[dict[str, object]],
) -> dict[str, object]:
    findings = {}
    for lead in (1, 6, 8, 12, 24):
        record = aggregate[LEADS.index(lead)]
        shares = record["shapley_percent_mean"]
        leader = max(shares, key=shares.get)
        seed_core_shares = np.asarray(
            [
                sum(seed_record["shapley_percent"][label] for label in CORE_LABELS)
                for seed_record in records
                if int(seed_record["lead"]) == lead
            ],
            dtype=float,
        )
        findings[str(lead)] = {
            "interaction_mean_bits": record["interaction_mean_bits"],
            "interaction_std_bits": record["interaction_std_bits"],
            "leader": leader,
            "leader_percent": shares[leader],
            "enso_iod_core_percent_mean": float(seed_core_shares.mean()),
            "enso_iod_core_percent_std": float(seed_core_shares.std(ddof=1)),
            "percentages": shares,
        }
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-stem", type=Path, default=DEFAULT_FIGURE_STEM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = sample_full_history_mode_inputs(
        n_samples=N_SAMPLES,
        intervention_bound=INTERVENTION_BOUND,
        seed=SAMPLING_SEED,
    )
    records, audit = evaluate_all(source, args.cache_dir, MAIN_COVARIANCE_RIDGE)
    aggregate = aggregate_records(records)
    sensitivity = ridge_sensitivity(
        source,
        args.cache_dir,
        main_records=records,
        main_audit=audit,
    )
    plot_figure(records, aggregate, args.figure_stem)

    input_files = [cache_path(args.cache_dir, seed) for seed in SEEDS]
    payload = {
        "analysis": "UniCM 11-mode exact Shapley attribution of integrated EI increment",
        "inputs": {
            "prediction_caches": [
                {
                    "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                    "sha256": sha256(path),
                }
                for path in input_files
            ],
            "source_shape": list(source.shape),
            "prediction_shape_per_checkpoint": [N_SAMPLES, len(LEADS), len(MODE_LABELS)],
        },
        "method": {
            "players": list(MODE_LABELS),
            "player_definition": "one climate mode's complete 12-month history",
            "target": "all 11 future UniCM modes jointly at each forecast lead",
            "seeds": list(SEEDS),
            "leads": list(LEADS),
            "n_samples": N_SAMPLES,
            "sampling_seed": SAMPLING_SEED,
            "intervention_bound": INTERVENTION_BOUND,
            "start_month": START_MONTH,
            "source_prior": "independent bounded-uniform maximum-entropy intervention coordinates",
            "affine_equivalent_source_covariance": "identity after per-coordinate standardization",
            "readout": "affine degree-1 TM / linear-Gaussian log-det equivalent",
            "main_covariance_ridge": MAIN_COVARIANCE_RIDGE,
            "coalition_count": 1 << len(MODE_LABELS),
            "coalition_value": "v(S) = EI(S -> Y) - sum_i EI({i} -> Y)",
            "percentage_denominator": "v(N) within each checkpoint and lead before checkpoint averaging",
        },
        "seed_lead_records": records,
        "lead_summary": aggregate,
        "audit": audit,
        "ridge_sensitivity": sensitivity,
        "key_findings": key_findings(records, aggregate),
        "figure_files": [
            str(args.figure_stem.with_suffix(f".{suffix}").relative_to(ROOT))
            for suffix in ("png", "svg", "pdf")
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "key_findings": payload["key_findings"]}, indent=2))


if __name__ == "__main__":
    main()
