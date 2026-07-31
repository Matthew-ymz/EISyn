"""Learn and decompose an equal-size mixed-order six-oscillator system.

The known six-oscillator mixed-order Kuramoto field is used only to generate
finite-time stochastic transitions. An MLP learns the complete six-dimensional
future phase increment. Independent maximum-entropy phase interventions are
then propagated through the learned model, and one conditional transport map
defines the joint distribution used for all 63 subset EIs and the shared
Greedy hierarchy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import validate_mixed_order_kuramoto_hierarchy as learned_base
from scripts.phi_hierarchy import (
    NONNEGATIVE_TOLERANT,
    greedy_phi_atoms,
)
from scripts.validate_equal_size_mixed_order_kuramoto import (
    CROSS_COUPLING,
    FREQUENCIES,
    NAMES,
    NOISE_SCALE,
    PAIRWISE_COUPLING,
    PAIRWISE_MODULE,
    TRIADIC_COUPLING,
    TRIADIC_MODULE,
)


PLANTED_MODULES = (PAIRWISE_MODULE, TRIADIC_MODULE)
NONNEGATIVE_TOLERANCE_BITS = 1.0e-6
RESULT_DIR = ROOT / "results" / "equal_size_mixed_order_kuramoto_learned"
DEFAULT_RESULT = RESULT_DIR / "summary.json"
DEFAULT_CHECKPOINT = RESULT_DIR / "partial_rows.json"
DEFAULT_STATUS = (
    ROOT
    / "docs"
    / "log"
    / "equal_size_mixed_order_kuramoto_learned"
    / "live_progress.json"
)


def six_oscillator_derivative(
    phases: np.ndarray,
    *,
    pairwise_coupling: float,
    triadic_coupling: float,
    cross_coupling: float = 0.0,
    frequencies: np.ndarray = FREQUENCIES,
) -> np.ndarray:
    """Evaluate the controlled pairwise-triangle plus triadic-hyperedge field."""
    values = np.asarray(phases, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(NAMES):
        raise ValueError(f"phases must have shape (n_samples, {len(NAMES)}).")
    derivative = np.broadcast_to(
        np.asarray(frequencies, dtype=float),
        values.shape,
    ).copy()
    for receiver in range(3):
        senders = [index for index in range(3) if index != receiver]
        derivative[:, receiver] += (float(pairwise_coupling) / 2.0) * sum(
            np.sin(values[:, sender] - values[:, receiver])
            for sender in senders
        )
    for receiver in range(3, 6):
        senders = [index for index in range(3, 6) if index != receiver]
        derivative[:, receiver] += float(triadic_coupling) * np.sin(
            values[:, senders[0]]
            + values[:, senders[1]]
            - 2.0 * values[:, receiver]
        )
    if float(cross_coupling) != 0.0:
        for left in range(3):
            for right in range(3, 6):
                delta = values[:, right] - values[:, left]
                contribution = (float(cross_coupling) / 3.0) * np.sin(delta)
                derivative[:, left] += contribution
                derivative[:, right] -= contribution
    return derivative


def six_oscillator_transport_context(phases: np.ndarray) -> np.ndarray:
    """Use identical first- and second-harmonic context for all six sources."""
    values = np.asarray(phases, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(NAMES):
        raise ValueError(f"phases must have shape (n_samples, {len(NAMES)}).")
    return np.column_stack(
        (
            learned_base.phase_state_features(values),
            np.cos(2.0 * values),
            np.sin(2.0 * values),
        )
    )


def six_oscillator_mlp_features(phases: np.ndarray) -> np.ndarray:
    """Use the same first- and second-harmonic features for every oscillator."""
    return six_oscillator_transport_context(phases)


def configure_learned_base(*, smoke: bool) -> None:
    """Bind the audited five-node machinery to the controlled six-node field."""
    learned_base.NAMES = NAMES
    learned_base.PAIRWISE_MODULE = PAIRWISE_MODULE
    learned_base.TRIADIC_MODULE = TRIADIC_MODULE
    learned_base.PLANTED_MODULES = PLANTED_MODULES
    learned_base.CONTEXT_SECOND_HARMONIC_NAMES = NAMES
    learned_base.FREQUENCIES = FREQUENCIES
    learned_base.PAIRWISE_COUPLING = PAIRWISE_COUPLING
    learned_base.TRIADIC_COUPLING = TRIADIC_COUPLING
    learned_base.PROCESS_NOISE = NOISE_SCALE
    learned_base.CROSS_COUPLING = CROSS_COUPLING
    learned_base.GREEDY_SPLIT_TOLERANCE = NONNEGATIVE_TOLERANCE_BITS
    learned_base.CONDITIONS = {
        "realistic": {
            "process_noise": NOISE_SCALE,
            "cross_coupling": CROSS_COUPLING,
        }
    }
    learned_base.mixed_order_derivative = six_oscillator_derivative
    learned_base.phase_mlp_features = six_oscillator_mlp_features
    learned_base.phase_transport_context = six_oscillator_transport_context
    learned_base.TM_MARGINAL_EVALUATIONS = 1024 if smoke else 2048
    learned_base.TM_MARGINAL_SAMPLES = 256 if smoke else 512


def module_atom_summary(
    row: Mapping[str, object],
    module: Sequence[str],
) -> dict[str, object]:
    """Decompose one planted block from the learned joint EI table."""
    table = {
        tuple(key.split("+")): float(value)
        for key, value in row["ei_bits"].items()
    }
    ordered = tuple(module)
    singleton = {name: float(table[(name,)]) for name in NAMES}
    atoms = greedy_phi_atoms(
        ordered,
        table,
        policy=NONNEGATIVE_TOLERANT,
        eps=learned_base.GREEDY_EPS,
        split_tolerance=NONNEGATIVE_TOLERANCE_BITS,
        singleton_ei=singleton,
    )
    positive = [atom for atom in atoms if atom.value > 0.0]
    triple = float(
        sum(atom.value for atom in positive if tuple(atom.sources) == ordered)
    )
    pair = float(sum(atom.value for atom in positive if len(atom.sources) == 2))
    total = float(sum(atom.value for atom in positive))
    return {
        "sources": list(ordered),
        "pair_atom_bits": pair,
        "triple_residual_bits": triple,
        "positive_atom_bits": total,
        "pair_atom_fraction": pair / total if total > 0.0 else 0.0,
        "triple_residual_fraction": triple / total if total > 0.0 else 0.0,
        "atoms": [
            {
                "sources": list(atom.sources),
                "value_bits": float(atom.value),
                "kind": atom.kind,
                "depth": int(atom.depth),
            }
            for atom in sorted(
                positive,
                key=lambda atom: atom.value,
                reverse=True,
            )
        ],
    }


def enrich_row(row: dict[str, object]) -> dict[str, object]:
    """Attach six-node-specific definitions and comparable module summaries."""
    row["target_dimension"] = len(NAMES)
    row["target_definition"] = (
        "full six-oscillator finite-time future phase change, wrapped to (-pi, pi]"
    )
    row["pairwise_module_summary"] = module_atom_summary(
        row,
        PAIRWISE_MODULE,
    )
    row["triadic_module_summary"] = module_atom_summary(
        row,
        TRIADIC_MODULE,
    )
    diagnostics = row["joint_marginalization"]
    row["nonnegative_audit"] = {
        "tolerance_bits": NONNEGATIVE_TOLERANCE_BITS,
        "minimum_non_singleton_phi_bits": float(
            diagnostics["minimum_non_singleton_phi_bits"]
        ),
        "minimum_partition_residual_bits": float(
            diagnostics["minimum_partition_residual_bits"]
        ),
        "near_zero_negative_phi_count": int(
            -NONNEGATIVE_TOLERANCE_BITS
            <= float(diagnostics["minimum_non_singleton_phi_bits"])
            < 0.0
        ),
        "near_zero_negative_residual_count": int(
            -NONNEGATIVE_TOLERANCE_BITS
            <= float(diagnostics["minimum_partition_residual_bits"])
            < 0.0
        ),
    }
    return row


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write small progress and checkpoint files atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate learned six-node diagnostics and paired module contrasts."""
    summary: dict[str, object] = {
        "n_seeds": len(rows),
        "root_split_recovery_rate": float(
            np.mean([bool(row["planted_modules_recovered"]) for row in rows])
        ),
    }
    for metric, getter in (
        (
            "heldout_circular_mae_rad",
            lambda row: row["dynamics_fit"]["heldout_circular_mae_rad"],
        ),
        ("root_phi_bits", lambda row: row["root_phi_bits"]),
        ("root_residual_bits", lambda row: row["root_residual_bits"]),
    ):
        values = np.asarray([float(getter(row)) for row in rows], dtype=float)
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_sem"] = (
            float(np.std(values, ddof=1) / math.sqrt(len(values)))
            if len(values) > 1
            else 0.0
        )
    for mechanism, field in (
        ("pairwise", "pairwise_module_summary"),
        ("triadic", "triadic_module_summary"),
    ):
        mechanism_summary: dict[str, float] = {}
        for metric in (
            "pair_atom_bits",
            "triple_residual_bits",
            "positive_atom_bits",
            "pair_atom_fraction",
            "triple_residual_fraction",
        ):
            values = np.asarray(
                [float(row[field][metric]) for row in rows],
                dtype=float,
            )
            mechanism_summary[f"{metric}_mean"] = float(np.mean(values))
            mechanism_summary[f"{metric}_sem"] = (
                float(np.std(values, ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
        summary[mechanism] = mechanism_summary
    deltas = np.asarray(
        [
            float(row["triadic_module_summary"]["triple_residual_fraction"])
            - float(row["pairwise_module_summary"]["triple_residual_fraction"])
            for row in rows
        ],
        dtype=float,
    )
    summary["paired_delta_triple_fraction"] = {
        "values": deltas.tolist(),
        "mean": float(np.mean(deltas)),
        "sem": (
            float(np.std(deltas, ddof=1) / math.sqrt(len(deltas)))
            if len(deltas) > 1
            else 0.0
        ),
        "positive_count": int(np.sum(deltas > 0.0)),
        "n_seeds": int(len(deltas)),
    }
    return summary


def build_summary(
    *,
    seeds: Sequence[int],
    training_count: int,
    readout_count: int,
    model_epochs: int,
    tm_epochs: int,
    status_path: Path,
    checkpoint_path: Path,
    include_shuffle: bool,
) -> dict[str, object]:
    """Run the learned six-node experiment with resumable seed checkpoints."""
    rows: list[dict[str, object]] = []
    shuffle_rows: list[dict[str, object]] = []
    checkpoint_version = (
        f"six_node_equal_modules_v2_fourier_mlp_"
        f"n{learned_base.TM_MARGINAL_EVALUATIONS}_"
        f"p{learned_base.TM_MARGINAL_SAMPLES}"
    )
    if checkpoint_path.exists():
        cached = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if cached.get("version") == checkpoint_version:
            rows = list(cached.get("rows", []))
            shuffle_rows = list(cached.get("target_shuffle_controls", []))
    total = len(seeds) + (1 if include_shuffle else 0)
    started = time.perf_counter()

    def checkpoint() -> None:
        atomic_write_json(
            checkpoint_path,
            {
                "version": checkpoint_version,
                "rows": rows,
                "target_shuffle_controls": shuffle_rows,
            },
        )

    def status(
        phase: str,
        *,
        metrics: Mapping[str, object] | None = None,
        message: str | None = None,
    ) -> None:
        completed = len(rows) + len(shuffle_rows)
        elapsed = time.perf_counter() - started
        rate = completed / elapsed if completed > 0 and elapsed > 0.0 else 0.0
        payload: dict[str, object] = {
            "phase": phase,
            "current": completed,
            "total": total,
            "unit": "seed fit",
            "elapsed_seconds": float(elapsed),
            "eta_seconds": (
                float((total - completed) / rate) if rate > 0.0 else None
            ),
            "metrics": dict(metrics or {}),
            "updated_at": time.time(),
        }
        if message is not None:
            payload["message"] = message
        atomic_write_json(status_path, payload)

    status(
        "running",
        message="resumed from compatible checkpoint" if rows else None,
    )
    for seed in seeds:
        if any(int(row["seed"]) == int(seed) for row in rows):
            continue
        status(
            "running",
            metrics={"seed": int(seed), "stage": "MLP + joint TM"},
        )
        row = learned_base.run_condition(
            condition="realistic",
            training_count=training_count,
            readout_count=readout_count,
            seed=int(seed),
            degree=3,
            train_fraction=0.8,
            ridge=0.10,
            tau=0.20,
            dt=0.01,
            epochs=model_epochs,
            ei_epochs=tm_epochs,
        )
        rows.append(enrich_row(row))
        checkpoint()
        status(
            "running",
            metrics={
                "seed": int(seed),
                "root_split_recovered": bool(
                    row["planted_modules_recovered"]
                ),
                "mae_rad": float(
                    row["dynamics_fit"]["heldout_circular_mae_rad"]
                ),
                "context_active": bool(
                    row["joint_transport_map"]["context_active"]
                ),
            },
        )
        print(
            f"[{len(rows)}/{total}] seed={seed}, "
            f"root={int(row['planted_modules_recovered'])}, "
            f"MAE={row['dynamics_fit']['heldout_circular_mae_rad']:.3f} rad, "
            f"pair triple={row['pairwise_module_summary']['triple_residual_fraction']:.3f}, "
            f"triad triple={row['triadic_module_summary']['triple_residual_fraction']:.3f}, "
            f"elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )
    if include_shuffle and not shuffle_rows:
        seed = int(seeds[0])
        status(
            "running",
            metrics={"seed": seed, "stage": "target shuffle"},
        )
        shuffled = learned_base.run_condition(
            condition="realistic",
            training_count=training_count,
            readout_count=readout_count,
            seed=seed,
            degree=3,
            train_fraction=0.8,
            ridge=0.10,
            tau=0.20,
            dt=0.01,
            epochs=model_epochs,
            ei_epochs=tm_epochs,
            shuffle_target=True,
        )
        shuffle_rows.append(enrich_row(shuffled))
        checkpoint()
    payload = {
        "experiment_contract": {
            "question": (
                "Does an MLP learned from finite-time six-oscillator transitions "
                "retain the pairwise-versus-triadic hierarchy?"
            ),
            "dynamics": (
                "six-oscillator mixed-order Kuramoto field with equal-size "
                "pairwise and triadic modules"
            ),
            "pairwise_module": list(PAIRWISE_MODULE),
            "triadic_module": list(TRIADIC_MODULE),
            "pipeline": [
                "generate stochastic finite-time transitions from the known field",
                (
                    "fit an MLP with symmetric first- and second-harmonic inputs "
                    "to the full six-dimensional wrapped phase increment"
                ),
                "evaluate the MLP on independent Uniform(-pi, pi) interventions",
                "fit one conditional autoregressive transport map",
                "derive all 63 subset EIs by common-Sobol marginalization",
                "apply the shared Greedy hierarchy",
            ],
            "fixed": {
                "training_count": int(training_count),
                "readout_count": int(readout_count),
                "tau": 0.20,
                "dt": 0.01,
                "process_noise": NOISE_SCALE,
                "cross_coupling": CROSS_COUPLING,
                "frequencies": FREQUENCIES.tolist(),
                "pairwise_total_coupling": PAIRWISE_COUPLING,
                "triadic_rms_matched_coupling": TRIADIC_COUPLING,
                "model_epochs": int(model_epochs),
                "tm_epochs": int(tm_epochs),
                "tm_marginal_evaluations": int(
                    learned_base.TM_MARGINAL_EVALUATIONS
                ),
                "tm_marginal_samples": int(
                    learned_base.TM_MARGINAL_SAMPLES
                ),
                "nonnegative_tolerance_bits": NONNEGATIVE_TOLERANCE_BITS,
                "mlp_input": (
                    "first- and second-harmonic sine/cosine features for all "
                    "six oscillators"
                ),
            },
            "primary_metrics": [
                "exact root split {1,2,3}|{4,5,6}",
                "within-module triple-residual fraction",
            ],
        },
        "rows": rows,
        "summary": aggregate(rows),
        "target_shuffle_controls": shuffle_rows,
    }
    atomic_write_json(
        status_path,
        {
            "phase": "complete",
            "current": total,
            "total": total,
            "unit": "seed fit",
            "elapsed_seconds": float(time.perf_counter() - started),
            "eta_seconds": 0.0,
            "metrics": {
                "root_split_recovery_rate": float(
                    payload["summary"]["root_split_recovery_rate"]
                ),
                "mae_rad": float(
                    payload["summary"]["heldout_circular_mae_rad_mean"]
                ),
            },
            "updated_at": time.time(),
        },
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--training-count", type=int, default=None)
    parser.add_argument("--readout-count", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--tm-epochs", type=int, default=None)
    parser.add_argument("--include-shuffle", action="store_true")
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoke = args.mode == "smoke"
    configure_learned_base(smoke=smoke)
    training_count = int(
        args.training_count or (3200 if smoke else 4800)
    )
    readout_count = int(
        args.readout_count or (3600 if smoke else 6000)
    )
    model_epochs = int(args.epochs or 800)
    tm_epochs = int(args.tm_epochs or (120 if smoke else 180))
    seeds = tuple(range(1 if smoke else int(args.seeds)))
    result_path = args.result or (
        RESULT_DIR / "smoke.json" if smoke else DEFAULT_RESULT
    )
    checkpoint_path = args.checkpoint or (
        RESULT_DIR / "smoke_partial_rows.json"
        if smoke
        else DEFAULT_CHECKPOINT
    )
    try:
        payload = build_summary(
            seeds=seeds,
            training_count=training_count,
            readout_count=readout_count,
            model_epochs=model_epochs,
            tm_epochs=tm_epochs,
            status_path=args.status,
            checkpoint_path=checkpoint_path,
            include_shuffle=bool(args.include_shuffle),
        )
    except Exception as error:
        previous: dict[str, object] = {}
        if args.status.exists():
            try:
                previous = json.loads(args.status.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
        atomic_write_json(
            args.status,
            {
                **previous,
                "phase": "failed",
                "message": str(error),
                "updated_at": time.time(),
            },
        )
        raise
    atomic_write_json(result_path, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Saved result: {result_path}")
    print(f"Saved status: {args.status}")


if __name__ == "__main__":
    main()
