"""Sweep pairwise-edge asymmetry in the learned six-oscillator comparison.

Only the relative weights of the three edges inside {theta1, theta2, theta3}
change. Their squared-weight sum is fixed, so the pairwise module retains the
same aggregate RMS coupling scale at every asymmetry level. All MLP, readout,
transport-map, intervention, noise, and Greedy settings are paired and fixed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import validate_mixed_order_kuramoto_hierarchy as learned_base
from scripts.validate_equal_size_mixed_order_kuramoto import (
    CROSS_COUPLING,
    FREQUENCIES,
    NAMES,
    NOISE_SCALE,
    PAIRWISE_COUPLING,
    TRIADIC_COUPLING,
)
from scripts.validate_equal_size_mixed_order_kuramoto_learned import (
    configure_learned_base,
    six_oscillator_derivative,
)
from scripts.validate_equal_size_mixed_order_kuramoto_mlp import (
    atomic_write_json,
    run_seed,
)


RHO_LEVELS = (1.0, 0.5, 0.25)
RESULT_DIR = ROOT / "results" / "pairwise_asymmetry_kuramoto_mlp"
DEFAULT_RESULT = RESULT_DIR / "summary.json"
DEFAULT_SMOKE_RESULT = RESULT_DIR / "smoke.json"
DEFAULT_CHECKPOINT = RESULT_DIR / "partial_rows.json"
DEFAULT_STATUS = (
    ROOT
    / "docs"
    / "log"
    / "pairwise_asymmetry_kuramoto_mlp"
    / "live_progress.json"
)


def normalized_edge_weights(
    rho: float,
    *,
    pairwise_coupling: float = PAIRWISE_COUPLING,
) -> tuple[float, float, float]:
    """Return (w12, w13, w23) with fixed sum of squared edge weights."""
    ratio = float(rho)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("rho must lie in (0, 1].")
    symmetric_edge = float(pairwise_coupling) / 2.0
    scale = symmetric_edge * math.sqrt(3.0 / (1.0 + 2.0 * ratio**2))
    return scale, scale * ratio, scale * ratio


def make_asymmetric_derivative(rho: float):
    """Bind a fixed-RMS asymmetric pairwise triangle to the six-node field."""
    edge_weights = normalized_edge_weights(float(rho))
    edge_pairs = ((0, 1), (0, 2), (1, 2))

    def derivative(
        phases: np.ndarray,
        *,
        pairwise_coupling: float,
        triadic_coupling: float,
        cross_coupling: float = 0.0,
        frequencies: np.ndarray = FREQUENCIES,
    ) -> np.ndarray:
        values = np.asarray(phases, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(NAMES):
            raise ValueError(
                f"phases must have shape (n_samples, {len(NAMES)})."
            )
        output = six_oscillator_derivative(
            values,
            pairwise_coupling=0.0,
            triadic_coupling=float(triadic_coupling),
            cross_coupling=float(cross_coupling),
            frequencies=np.asarray(frequencies, dtype=float),
        )
        relative_scale = float(pairwise_coupling) / PAIRWISE_COUPLING
        for (left, right), base_weight in zip(edge_pairs, edge_weights):
            weight = relative_scale * base_weight
            contribution = weight * np.sin(
                values[:, right] - values[:, left]
            )
            output[:, left] += contribution
            output[:, right] -= contribution
        return output

    return derivative


def enrich_row(
    row: Mapping[str, object],
    *,
    rho: float,
) -> dict[str, object]:
    output = dict(row)
    weights = normalized_edge_weights(float(rho))
    output["rho"] = float(rho)
    output["asymmetry"] = float(1.0 - rho)
    output["pairwise_edge_weights"] = {
        "theta1-theta2": weights[0],
        "theta1-theta3": weights[1],
        "theta2-theta3": weights[2],
    }
    output["pairwise_edge_squared_sum"] = float(sum(w * w for w in weights))
    output["pair_atom_fraction_123"] = float(
        row["modules"]["pairwise"]["pair_atom_fraction"]
    )
    output["triple_residual_fraction_123"] = float(
        row["modules"]["pairwise"]["triple_residual_fraction"]
    )
    output["triple_residual_fraction_456"] = float(
        row["modules"]["triadic"]["triple_residual_fraction"]
    )
    output["triple_fraction_contrast_456_minus_123"] = float(
        output["triple_residual_fraction_456"]
        - output["triple_residual_fraction_123"]
    )
    return output


def _mean_sem(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sem": float(
            array.std(ddof=1) / math.sqrt(len(array))
            if len(array) > 1
            else 0.0
        ),
    }


def aggregate(
    rows: Sequence[Mapping[str, object]],
    *,
    rho_levels: Sequence[float],
) -> dict[str, object]:
    by_rho: dict[str, object] = {}
    for rho in rho_levels:
        selected = [
            row for row in rows if np.isclose(float(row["rho"]), float(rho))
        ]
        by_rho[f"{float(rho):g}"] = {
            "n_seeds": len(selected),
            "edge_weights": selected[0]["pairwise_edge_weights"],
            "pair_atom_fraction_123": _mean_sem(
                [float(row["pair_atom_fraction_123"]) for row in selected]
            ),
            "triple_residual_fraction_123": _mean_sem(
                [
                    float(row["triple_residual_fraction_123"])
                    for row in selected
                ]
            ),
            "triple_residual_fraction_456": _mean_sem(
                [
                    float(row["triple_residual_fraction_456"])
                    for row in selected
                ]
            ),
            "triple_fraction_contrast_456_minus_123": _mean_sem(
                [
                    float(row["triple_fraction_contrast_456_minus_123"])
                    for row in selected
                ]
            ),
            "heldout_circular_mae_rad": _mean_sem(
                [
                    float(row["dynamics_fit"]["heldout_circular_mae_rad"])
                    for row in selected
                ]
            ),
        }
    baseline = {
        int(row["seed"]): float(row["pair_atom_fraction_123"])
        for row in rows
        if np.isclose(float(row["rho"]), 1.0)
    }
    paired_changes: dict[str, object] = {}
    for rho in rho_levels:
        values = [
            float(row["pair_atom_fraction_123"]) - baseline[int(row["seed"])]
            for row in rows
            if np.isclose(float(row["rho"]), float(rho))
        ]
        paired_changes[f"{float(rho):g}"] = {
            "values": values,
            **_mean_sem(values),
            "positive_count": int(np.sum(np.asarray(values) > 0.0)),
            "n_seeds": len(values),
        }
    ordered_pair_means = [
        float(by_rho[f"{float(rho):g}"]["pair_atom_fraction_123"]["mean"])
        for rho in rho_levels
    ]
    monotone = bool(
        all(
            later >= earlier
            for earlier, later in zip(
                ordered_pair_means,
                ordered_pair_means[1:],
            )
        )
    )
    return {
        "by_rho": by_rho,
        "paired_pair_fraction_change_from_symmetric": paired_changes,
        "pair_fraction_monotone_as_rho_decreases": monotone,
    }


def build_summary(
    *,
    rho_levels: Sequence[float],
    seeds: Sequence[int],
    training_count: int,
    readout_count: int,
    model_epochs: int,
    degree: int,
    checkpoint_path: Path,
    status_path: Path,
) -> dict[str, object]:
    version = (
        f"pair_asymmetry_v1_rho{'-'.join(f'{rho:g}' for rho in rho_levels)}_"
        f"n{training_count}_r{readout_count}_e{model_epochs}_d{degree}"
    )
    rows: list[dict[str, object]] = []
    if checkpoint_path.exists():
        cached = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if cached.get("version") == version:
            rows = list(cached.get("rows", []))
    total = len(rho_levels) * len(seeds)
    started = time.perf_counter()
    for seed in seeds:
        for rho in rho_levels:
            if any(
                int(row["seed"]) == int(seed)
                and np.isclose(float(row["rho"]), float(rho))
                for row in rows
            ):
                continue
            learned_base.mixed_order_derivative = make_asymmetric_derivative(
                float(rho)
            )
            atomic_write_json(
                status_path,
                {
                    "phase": "running",
                    "current": len(rows),
                    "total": total,
                    "unit": "condition-seed",
                    "metrics": {
                        "seed": int(seed),
                        "rho": float(rho),
                        "stage": "MLP + module TM",
                    },
                    "updated_at": time.time(),
                },
            )
            row = run_seed(
                training_count=int(training_count),
                readout_count=int(readout_count),
                model_epochs=int(model_epochs),
                degree=int(degree),
                seed=int(seed),
            )
            enriched = enrich_row(row, rho=float(rho))
            rows.append(enriched)
            atomic_write_json(
                checkpoint_path,
                {"version": version, "rows": rows},
            )
            print(
                f"[{len(rows)}/{total}] seed={seed}, rho={rho:g}, "
                f"pair={enriched['pair_atom_fraction_123']:.3f}, "
                f"triad={enriched['triple_residual_fraction_456']:.3f}, "
                f"contrast={enriched['triple_fraction_contrast_456_minus_123']:.3f}, "
                f"MAE={row['dynamics_fit']['heldout_circular_mae_rad']:.3f}, "
                f"elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )
    summary = aggregate(rows, rho_levels=rho_levels)
    payload = {
        "experiment_contract": {
            "scientific_question": (
                "What changes when only the concentration of fixed-RMS "
                "pairwise triangle edge weights changes?"
            ),
            "treatment_factor": "rho in normalized weights c(rho)*(1,rho,rho)",
            "treatment_levels": list(rho_levels),
            "pairing_unit": "seed with shared initial phases, process noise, MLP seed, and intervention/readout samples",
            "primary_metric": "pair_atom_fraction_123",
            "secondary_metrics": [
                "triple_residual_fraction_123",
                "triple_residual_fraction_456",
                "triple_fraction_contrast_456_minus_123",
                "heldout_circular_mae_rad",
            ],
            "minimum_interpretable_effect": (
                "+0.10 pair-atom fraction at rho=0.5 relative to rho=1"
            ),
            "frozen": {
                "pairwise_edge_squared_sum": float(
                    3.0 * (PAIRWISE_COUPLING / 2.0) ** 2
                ),
                "triadic_coupling": TRIADIC_COUPLING,
                "cross_coupling": CROSS_COUPLING,
                "process_noise": NOISE_SCALE,
                "frequencies": FREQUENCIES.tolist(),
                "training_count": int(training_count),
                "readout_count": int(readout_count),
                "model_epochs": int(model_epochs),
                "tm_degree": int(degree),
                "tau": 0.20,
                "dt": 0.01,
                "intervention_support": "independent Uniform(-pi, pi)",
                "mlp_features": (
                    "symmetric first- and second-harmonic features for all nodes"
                ),
                "nonnegative_tolerance_bits": 0.10,
                "postprocessing": "no EI clipping and no monotone projection",
            },
        },
        "rows": rows,
        "summary": summary,
    }
    atomic_write_json(
        status_path,
        {
            "phase": "complete",
            "current": total,
            "total": total,
            "unit": "condition-seed",
            "elapsed_seconds": float(time.perf_counter() - started),
            "metrics": {
                "monotone": bool(
                    summary["pair_fraction_monotone_as_rho_decreases"]
                ),
                "rho_0.5_pair_fraction": float(
                    summary["by_rho"]["0.5"]["pair_atom_fraction_123"]["mean"]
                ),
            },
            "updated_at": time.time(),
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--training-count", type=int, default=None)
    parser.add_argument("--readout-count", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    args = parser.parse_args()
    smoke = args.mode == "smoke"
    configure_learned_base(smoke=smoke)
    training_count = int(args.training_count or (3200 if smoke else 4800))
    readout_count = int(args.readout_count or (5000 if smoke else 6000))
    seeds = tuple(range(1 if smoke else int(args.seeds)))
    result_path = args.result or (
        DEFAULT_SMOKE_RESULT if smoke else DEFAULT_RESULT
    )
    checkpoint_path = args.checkpoint or (
        RESULT_DIR / "smoke_partial_rows.json"
        if smoke
        else DEFAULT_CHECKPOINT
    )
    payload = build_summary(
        rho_levels=RHO_LEVELS,
        seeds=seeds,
        training_count=training_count,
        readout_count=readout_count,
        model_epochs=int(args.epochs),
        degree=int(args.degree),
        checkpoint_path=checkpoint_path,
        status_path=args.status,
    )
    atomic_write_json(result_path, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Saved result: {result_path}")
    print(f"Saved status: {args.status}")


if __name__ == "__main__":
    main()
