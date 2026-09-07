#!/usr/bin/env python3
"""Validate source-order effects in PEID nulls and frozen MGSTN readouts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/source_order_bias_validation"
MGSTN = (
    ROOT
    / "results/nyc_taxi_mgstn_spt/"
    "affine_regression_v3_time_block_search_n4096_pc2_r1e-06_exact8"
)
INTERVENTION = (
    ROOT / "results/nyc_taxi_mgstn_spt/shared_intervention_n4096_pc2.npz"
)
STATES = ("weekday_peak", "weekend_midday", "rainy_high_demand")
TIME_NAMES = ("recent", "daily", "weekly")
SEEDS = (0, 1, 2)
PANEL_SEED = 20260906
PANEL_COUNT = 8
SOURCE_COUNT = 10
TOLERANCE_BITS = 1.0e-8
LOG2 = float(np.log(2.0))


def subset_orders(n: int) -> np.ndarray:
    return np.asarray([mask.bit_count() for mask in range(1 << n)], dtype=int)


def metrics_from_ei(ei: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return singleton-partition total Syn and minimum-bipartition Syn."""
    singleton = np.asarray([ei[1 << index] for index in range(n)])
    total = np.zeros_like(ei)
    minimum = np.zeros_like(ei)
    for mask in range(1, 1 << n):
        total[mask] = ei[mask] - sum(
            singleton[index] for index in range(n) if mask & (1 << index)
        )
        if mask.bit_count() < 2:
            continue
        anchor = mask & -mask
        left = (mask - 1) & mask
        best_children = -np.inf
        while left:
            if left & anchor:
                best_children = max(best_children, ei[left] + ei[mask ^ left])
            left = (left - 1) & mask
        minimum[mask] = ei[mask] - best_children
    return total, minimum


def order_means(values: np.ndarray, n: int) -> np.ndarray:
    orders = subset_orders(n)
    return np.asarray(
        [values[orders == order].mean() for order in range(2, n + 1)], dtype=float
    )


def parity_ei(n: int, interaction_order: int) -> np.ndarray:
    """Exact EI table for one parity bit plus irrelevant independent sources."""
    required = (1 << interaction_order) - 1
    return np.asarray(
        [1.0 if mask & required == required else 0.0 for mask in range(1 << n)],
        dtype=float,
    )


def convergent_additive_ei(n: int, noise_variance: float = 2.0) -> np.ndarray:
    """Exact Gaussian MI for Y=sum_i X_i+noise with independent unit X_i."""
    total_variance = n + noise_variance
    return np.asarray(
        [
            0.5 * np.log2(total_variance / (total_variance - mask.bit_count()))
            for mask in range(1 << n)
        ],
        dtype=float,
    )


def separable_additive_ei(n: int, singleton_bits: float = 0.5) -> np.ndarray:
    """Exact EI for independent coordinate-wise additive Gaussian channels."""
    return np.asarray(
        [singleton_bits * mask.bit_count() for mask in range(1 << n)], dtype=float
    )


def gaussian_ei_table(conditional: np.ndarray, atom_count: int) -> np.ndarray:
    """Evaluate coherent affine-TM EI for all subsets of two-dimensional atoms."""
    expected = 2 * atom_count
    if conditional.shape != (expected, expected):
        raise ValueError(f"Expected {(expected, expected)}, got {conditional.shape}")
    ei = np.zeros(1 << atom_count, dtype=float)
    for mask in range(1, 1 << atom_count):
        dims = [
            dimension
            for atom in range(atom_count)
            if mask & (1 << atom)
            for dimension in (2 * atom, 2 * atom + 1)
        ]
        sign, logdet = np.linalg.slogdet(conditional[np.ix_(dims, dims)])
        if sign <= 0:
            raise RuntimeError("MGSTN conditional covariance is not positive definite")
        ei[mask] = -0.5 * logdet / LOG2
    return ei


def block_diagonal_null(conditional: np.ndarray, atom_count: int) -> np.ndarray:
    null = np.zeros_like(conditional)
    for atom in range(atom_count):
        dims = slice(2 * atom, 2 * atom + 2)
        null[dims, dims] = conditional[dims, dims]
    return null


def mgstn_profiles() -> tuple[list[dict[str, object]], list[list[int]], dict[str, object]]:
    with np.load(INTERVENTION, allow_pickle=False) as archive:
        zone_ids = archive["zone_ids"].astype(int)
    rng = np.random.default_rng(PANEL_SEED)
    panels = [
        np.sort(rng.choice(len(zone_ids), SOURCE_COUNT, replace=False))
        for _ in range(PANEL_COUNT)
    ]
    rows: list[dict[str, object]] = []
    edge_deltas = {"total": [], "minimum": []}
    for seed in SEEDS:
        for state in STATES:
            cache = MGSTN / f"seed_{seed}/{state}/affine_cache.npz"
            with np.load(cache, allow_pickle=False) as archive:
                full_conditional = archive["conditional"].astype(float)
            for time_index, time_name in enumerate(TIME_NAMES):
                for panel_index, zones in enumerate(panels):
                    atoms = zones * len(TIME_NAMES) + time_index
                    dims = np.asarray(
                        [d for atom in atoms for d in (2 * atom, 2 * atom + 1)],
                        dtype=int,
                    )
                    conditional = full_conditional[np.ix_(dims, dims)]
                    ei = gaussian_ei_table(conditional, SOURCE_COUNT)
                    total, minimum = metrics_from_ei(ei, SOURCE_COUNT)
                    valid = subset_orders(SOURCE_COUNT) >= 2
                    for name, values in (("total", total), ("minimum", minimum)):
                        bad = values[valid] < -TOLERANCE_BITS
                        if bad.any():
                            raise RuntimeError(
                                f"MGSTN {name} Syn nonnegativity violation: "
                                f"minimum={values[valid].min():.12g}, "
                                f"threshold={-TOLERANCE_BITS}, affected_count={bad.sum()}"
                            )
                        edge_deltas[name].extend(
                            values[mask | (1 << added)] - values[mask]
                            for mask in range(1 << SOURCE_COUNT)
                            for added in range(SOURCE_COUNT)
                            if not mask & (1 << added)
                        )
                    null_ei = gaussian_ei_table(
                        block_diagonal_null(conditional, SOURCE_COUNT), SOURCE_COUNT
                    )
                    null_total, null_minimum = metrics_from_ei(null_ei, SOURCE_COUNT)
                    for order, total_mean, minimum_mean, null_total_mean, null_minimum_mean in zip(
                        range(2, SOURCE_COUNT + 1),
                        order_means(total, SOURCE_COUNT),
                        order_means(minimum, SOURCE_COUNT),
                        order_means(null_total, SOURCE_COUNT),
                        order_means(null_minimum, SOURCE_COUNT),
                    ):
                        rows.append(
                            {
                                "seed": seed,
                                "state": state,
                                "time_scale": time_name,
                                "panel": panel_index,
                                "order": order,
                                "total_syn_bits": float(total_mean),
                                "minimum_bipartition_syn_bits": float(minimum_mean),
                                "block_diagonal_total_syn_bits": float(null_total_mean),
                                "block_diagonal_minimum_syn_bits": float(null_minimum_mean),
                            }
                        )
    nested_audit = {}
    for name, values in edge_deltas.items():
        deltas = np.asarray(values)
        nested_audit[name] = {
            "edge_count": int(len(deltas)),
            "negative_violation_count": int(np.sum(deltas < -TOLERANCE_BITS)),
            "within_tolerance_count": int(np.sum(np.abs(deltas) <= TOLERANCE_BITS)),
            "minimum_delta_bits": float(deltas.min()),
        }
    return (
        rows,
        [[int(zone_ids[index]) for index in panel] for panel in panels],
        nested_audit,
    )


def adjacent_audit(rows: list[dict[str, object]], field: str) -> dict[str, object]:
    units: dict[tuple[object, ...], dict[int, float]] = {}
    for row in rows:
        key = (row["seed"], row["state"], row["time_scale"], row["panel"])
        units.setdefault(key, {})[int(row["order"])] = float(row[field])
    by_order = {}
    all_deltas = []
    for order in range(3, SOURCE_COUNT + 1):
        deltas = np.asarray(
            [values[order] - values[order - 1] for values in units.values()]
        )
        all_deltas.extend(deltas.tolist())
        by_order[str(order)] = {
            "mean_delta_bits": float(deltas.mean()),
            "sd_delta_bits": float(deltas.std(ddof=1)),
            "negative_violation_count": int(np.sum(deltas < -TOLERANCE_BITS)),
            "minimum_delta_bits": float(deltas.min()),
        }
    deltas = np.asarray(all_deltas)
    return {
        "paired_unit_count": len(units),
        "adjacent_delta_count": int(len(deltas)),
        "positive_count": int(np.sum(deltas > TOLERANCE_BITS)),
        "within_tolerance_count": int(np.sum(np.abs(deltas) <= TOLERANCE_BITS)),
        "negative_violation_count": int(np.sum(deltas < -TOLERANCE_BITS)),
        "minimum_adjacent_delta_bits": float(deltas.min()),
        "mean_adjacent_delta_bits": float(deltas.mean()),
        "by_destination_order": by_order,
    }


def summarize(
    rows: list[dict[str, object]],
    panels: list[list[int]],
    nested_audit: dict[str, object],
) -> dict[str, object]:
    null_specs = {
        "noise_only": np.zeros(1 << SOURCE_COUNT),
        "separable_additive_vector": separable_additive_ei(SOURCE_COUNT),
        "convergent_additive_scalar": convergent_additive_ei(SOURCE_COUNT),
        "pair_parity_embedded": parity_ei(SOURCE_COUNT, 2),
        "seven_way_parity_embedded": parity_ei(SOURCE_COUNT, 7),
        "ten_way_parity": parity_ei(SOURCE_COUNT, 10),
    }
    nulls = {}
    for name, ei in null_specs.items():
        total, minimum = metrics_from_ei(ei, SOURCE_COUNT)
        nulls[name] = {
            "total_syn_order_means_bits": order_means(total, SOURCE_COUNT).tolist(),
            "minimum_bipartition_order_means_bits": order_means(
                minimum, SOURCE_COUNT
            ).tolist(),
        }

    grouped = {}
    for order in range(2, SOURCE_COUNT + 1):
        block = [row for row in rows if int(row["order"]) == order]
        grouped[str(order)] = {}
        for field in (
            "total_syn_bits",
            "minimum_bipartition_syn_bits",
            "block_diagonal_total_syn_bits",
            "block_diagonal_minimum_syn_bits",
        ):
            values = np.asarray([float(row[field]) for row in block])
            grouped[str(order)][field] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
            }

    return {
        "status": "complete",
        "definition": {
            "total_syn": "EI(S;T) - sum_i EI(X_i;T)",
            "minimum_bipartition_syn": "EI(S;T) - max_{A|S\\A}[EI(A;T)+EI(S\\A;T)]",
            "nonnegative_tolerance_bits": TOLERANCE_BITS,
        },
        "null_models": nulls,
        "seven_way_nested_path": {
            "source_counts": [7, 8, 9, 10],
            "total_syn_bits": [1.0, 1.0, 1.0, 1.0],
            "minimum_bipartition_syn_bits": [1.0, 0.0, 0.0, 0.0],
        },
        "mgstn": {
            "estimator": "cached coherent generative affine TM (X~N(0,I))",
            "model_seeds": list(SEEDS),
            "states": list(STATES),
            "time_scales": list(TIME_NAMES),
            "panel_seed": PANEL_SEED,
            "zone_panels": panels,
            "paired_units": PANEL_COUNT * len(SEEDS) * len(STATES) * len(TIME_NAMES),
            "order_summary": grouped,
            "total_syn_adjacent_audit": adjacent_audit(rows, "total_syn_bits"),
            "minimum_bipartition_adjacent_audit": adjacent_audit(
                rows, "minimum_bipartition_syn_bits"
            ),
            "nested_coalition_edge_audit": nested_audit,
        },
    }


def plot(summary: dict[str, object]) -> Path:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(11.8, 2.55), constrained_layout=True)
    orders = np.arange(2, SOURCE_COUNT + 1)

    null_names = (
        "noise_only",
        "separable_additive_vector",
        "convergent_additive_scalar",
        "pair_parity_embedded",
        "seven_way_parity_embedded",
        "ten_way_parity",
    )
    labels = (
        "Noise only",
        "Separable additive",
        "Convergent additive",
        "2-way parity",
        "7-way parity",
        "10-way parity",
    )
    matrix = np.asarray(
        [
            summary["null_models"][name]["minimum_bipartition_order_means_bits"]
            for name in null_names
        ]
    )
    display = matrix.copy()
    for row in range(len(display)):
        maximum = display[row].max()
        if maximum > 0:
            display[row] /= maximum
    image = axes[0].imshow(display, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[0].set(
        xticks=np.arange(len(orders)),
        xticklabels=orders,
        yticks=np.arange(len(labels)),
        yticklabels=labels,
        xlabel="Source order",
    )
    axes[0].tick_params(length=0)
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.03)
    colorbar.set_label("Row-normalized irreducible Syn")

    path = summary["seven_way_nested_path"]
    source_counts = np.asarray(path["source_counts"])
    axes[1].plot(
        source_counts,
        path["total_syn_bits"],
        marker="o",
        color="#4C78A8",
        label="Total singleton-partition Syn",
    )
    axes[1].plot(
        source_counts,
        path["minimum_bipartition_syn_bits"],
        marker="s",
        color="#D97757",
        label="Minimum-bipartition Syn",
    )
    axes[1].set(
        xlabel="Nested system size",
        ylabel="Syn (bits)",
        xticks=source_counts,
        ylim=(-0.05, 1.08),
    )
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.29), ncol=1)

    grouped = summary["mgstn"]["order_summary"]
    styles = (
        ("total_syn_bits", "Total singleton-partition Syn", "#4C78A8", "o"),
        (
            "minimum_bipartition_syn_bits",
            "Minimum-bipartition Syn",
            "#D97757",
            "s",
        ),
    )
    for field, label, color, marker in styles:
        mean = np.asarray([grouped[str(order)][field]["mean"] for order in orders])
        sd = np.asarray([grouped[str(order)][field]["sd"] for order in orders])
        axes[2].plot(orders, mean, color=color, marker=marker, label=label)
        axes[2].fill_between(orders, mean - sd, mean + sd, color=color, alpha=0.16)
    axes[2].axhline(0, color="0.45", lw=0.7, ls="--")
    axes[2].set(
        xlabel="Source order",
        ylabel="Mean Syn (bits)",
        xticks=orders,
    )
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, 1.29), ncol=1)

    marginal = summary["mgstn"]["minimum_bipartition_adjacent_audit"][
        "by_destination_order"
    ]
    marginal_orders = np.arange(3, SOURCE_COUNT + 1)
    marginal_mean = np.asarray(
        [marginal[str(order)]["mean_delta_bits"] for order in marginal_orders]
    )
    marginal_sd = np.asarray(
        [marginal[str(order)]["sd_delta_bits"] for order in marginal_orders]
    )
    axes[3].plot(marginal_orders, marginal_mean, color="#D97757", marker="s")
    axes[3].fill_between(
        marginal_orders,
        marginal_mean - marginal_sd,
        marginal_mean + marginal_sd,
        color="#D97757",
        alpha=0.16,
    )
    axes[3].axhline(0, color="0.45", lw=0.7, ls="--")
    axes[3].set(
        xlabel="Destination source order",
        ylabel=r"Marginal irreducible Syn, $\Delta_k$ (bits)",
        xticks=marginal_orders,
    )

    for label, axis in zip("abcd", axes):
        axis.text(-0.16, 1.08, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    output = OUT / "source_order_bias_validation.png"
    fig.savefig(output, dpi=450, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, panels, nested_audit = mgstn_profiles()
    summary = summarize(rows, panels, nested_audit)
    summary_path = OUT / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    figure = plot(summary)
    print(json.dumps({"summary": str(summary_path), "figure": str(figure)}, indent=2))


if __name__ == "__main__":
    main()
