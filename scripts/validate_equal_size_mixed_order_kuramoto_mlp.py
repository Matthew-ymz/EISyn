"""Learn the six-oscillator mechanism and decompose module-local EI.

The known mixed-order Kuramoto field generates stochastic finite-time
transitions. A symmetric Fourier-feature MLP learns the complete six-output
phase increment. The learned channel is then evaluated under independent
uniform phase interventions and analyzed with the same module-local
polynomial triangular transport-map estimator as the oracle comparison.
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

from exp.TM.transport_map_density import (
    LOG_2,
    fit_polynomial_triangular_transport_map_density,
)
from scripts import validate_mixed_order_kuramoto_hierarchy as learned_base
from scripts.phi_hierarchy import NONNEGATIVE_TOLERANT, greedy_phi_atoms
from scripts.validate_equal_size_mixed_order_kuramoto import (
    CROSS_COUPLING,
    FREQUENCIES,
    MODULES,
    NAMES,
    NOISE_SCALE,
    PAIRWISE_COUPLING,
    TRIADIC_COUPLING,
    audit_phi_table,
    atom_summary,
    source_blocks,
)
from scripts.validate_equal_size_mixed_order_kuramoto_learned import (
    configure_learned_base,
)


NONNEGATIVE_TOLERANCE_BITS = 0.10
RESULT_DIR = ROOT / "results" / "equal_size_mixed_order_kuramoto_mlp"
DEFAULT_RESULT = RESULT_DIR / "summary.json"
DEFAULT_CHECKPOINT = RESULT_DIR / "partial_rows.json"
DEFAULT_STATUS = (
    ROOT
    / "docs"
    / "log"
    / "equal_size_mixed_order_kuramoto_mlp"
    / "live_progress.json"
)


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def estimate_module_table(
    phases: np.ndarray,
    target: np.ndarray,
    *,
    module: Sequence[str],
    target_log_probability: np.ndarray,
    degree: int,
) -> tuple[dict[tuple[str, ...], float], dict[str, object]]:
    """Estimate seven module EIs without clipping or monotone projection."""
    blocks = source_blocks(phases)
    table: dict[tuple[str, ...], float] = {}
    near_zero_negative: list[tuple[str, ...]] = []
    from scripts.phi_hierarchy import all_nonempty_subsets

    for subset in all_nonempty_subsets(module):
        source = np.column_stack([blocks[name] for name in subset])
        joint = np.column_stack((target, source))
        joint_model = fit_polynomial_triangular_transport_map_density(
            joint,
            degree=int(degree),
        )
        source_model = fit_polynomial_triangular_transport_map_density(
            source,
            degree=int(degree),
        )
        pointwise = (
            joint_model.log_prob(joint)
            - source_model.log_prob(source)
            - target_log_probability
        ) / LOG_2
        estimate = float(np.mean(pointwise))
        if estimate < -NONNEGATIVE_TOLERANCE_BITS:
            raise RuntimeError(
                f"EI{tuple(subset)}={estimate:.6g} bits is below "
                f"-{NONNEGATIVE_TOLERANCE_BITS:.6g} bits."
            )
        if estimate < 0.0:
            near_zero_negative.append(tuple(subset))
        table[tuple(subset)] = estimate
    return table, {
        "tolerance_bits": NONNEGATIVE_TOLERANCE_BITS,
        "near_zero_negative_ei_count": len(near_zero_negative),
        "near_zero_negative_ei_subsets": [
            list(subset) for subset in near_zero_negative
        ],
        "nonnegative_clipping": False,
        "monotone_projection": False,
    }


def run_seed(
    *,
    training_count: int,
    readout_count: int,
    model_epochs: int,
    degree: int,
    seed: int,
) -> dict[str, object]:
    started = time.perf_counter()
    phases, future = learned_base.generated_transition_data(
        sample_count=int(training_count),
        seed=int(seed),
        pairwise_coupling=PAIRWISE_COUPLING,
        triadic_coupling=TRIADIC_COUPLING,
        cross_coupling=CROSS_COUPLING,
        process_noise=NOISE_SCALE,
        tau=0.20,
        dt=0.01,
    )
    fitted, residual_covariance, fit_diagnostics = (
        learned_base.fit_learned_future_model(
            phases,
            future,
            seed=int(seed),
            epochs=int(model_epochs),
        )
    )
    rng = np.random.default_rng(int(seed) + 30_000)
    readout_phases = rng.uniform(
        -math.pi,
        math.pi,
        size=(int(readout_count), len(NAMES)),
    )
    readout_target = learned_base.learned_future_readout(
        fitted,
        residual_covariance,
        readout_phases,
        seed=int(seed) + 40_000,
    )
    target_model = fit_polynomial_triangular_transport_map_density(
        readout_target,
        degree=int(degree),
    )
    target_log_probability = target_model.log_prob(readout_target)
    modules: dict[str, object] = {}
    for mechanism, module in MODULES.items():
        table, ei_audit = estimate_module_table(
            readout_phases,
            readout_target,
            module=module,
            target_log_probability=target_log_probability,
            degree=degree,
        )
        phi_audit = audit_phi_table(module, table)
        atoms = greedy_phi_atoms(
            module,
            table,
            policy=NONNEGATIVE_TOLERANT,
            eps=1.0e-5,
            split_tolerance=NONNEGATIVE_TOLERANCE_BITS,
            singleton_ei={name: float(table[(name,)]) for name in module},
        )
        modules[mechanism] = {
            "sources": list(module),
            **atom_summary(module, atoms),
            "ei_audit": ei_audit,
            "phi_audit": phi_audit,
            "ei_bits": {
                "+".join(subset): float(value)
                for subset, value in table.items()
            },
            "atoms": [
                {
                    "sources": list(atom.sources),
                    "value_bits": float(atom.value),
                    "kind": atom.kind,
                    "depth": int(atom.depth),
                }
                for atom in sorted(
                    atoms,
                    key=lambda item: item.value,
                    reverse=True,
                )
                if atom.value > 0.0
            ],
        }
    return {
        "seed": int(seed),
        "dynamics_fit": fit_diagnostics,
        "modules": modules,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {"n_seeds": len(rows)}
    mae = np.asarray(
        [
            float(row["dynamics_fit"]["heldout_circular_mae_rad"])
            for row in rows
        ]
    )
    output["heldout_circular_mae_rad_mean"] = float(mae.mean())
    output["heldout_circular_mae_rad_sem"] = float(
        mae.std(ddof=1) / math.sqrt(len(mae)) if len(mae) > 1 else 0.0
    )
    for mechanism in MODULES:
        metrics: dict[str, float] = {}
        for metric in (
            "pair_atom_bits",
            "triple_residual_bits",
            "positive_atom_bits",
            "pair_atom_fraction",
            "triple_residual_fraction",
        ):
            values = np.asarray(
                [
                    float(row["modules"][mechanism][metric])
                    for row in rows
                ]
            )
            metrics[f"{metric}_mean"] = float(values.mean())
            metrics[f"{metric}_sem"] = float(
                values.std(ddof=1) / math.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )
        output[mechanism] = metrics
    delta = np.asarray(
        [
            float(row["modules"]["triadic"]["triple_residual_fraction"])
            - float(row["modules"]["pairwise"]["triple_residual_fraction"])
            for row in rows
        ]
    )
    output["paired_delta_triple_fraction"] = {
        "values": delta.tolist(),
        "mean": float(delta.mean()),
        "sem": float(
            delta.std(ddof=1) / math.sqrt(len(delta))
            if len(delta) > 1
            else 0.0
        ),
        "positive_count": int(np.sum(delta > 0.0)),
        "n_seeds": len(delta),
    }
    return output


def build_summary(
    *,
    seeds: Sequence[int],
    training_count: int,
    readout_count: int,
    model_epochs: int,
    degree: int,
    checkpoint_path: Path,
    status_path: Path,
) -> dict[str, object]:
    version = (
        f"module_local_tm_v1_n{training_count}_r{readout_count}_"
        f"e{model_epochs}_d{degree}"
    )
    rows: list[dict[str, object]] = []
    if checkpoint_path.exists():
        cached = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if cached.get("version") == version:
            rows = list(cached.get("rows", []))
    started = time.perf_counter()
    for seed in seeds:
        if any(int(row["seed"]) == int(seed) for row in rows):
            continue
        atomic_write_json(
            status_path,
            {
                "phase": "running",
                "current": len(rows),
                "total": len(seeds),
                "unit": "seed",
                "metrics": {"seed": int(seed), "stage": "MLP + module TM"},
                "updated_at": time.time(),
            },
        )
        row = run_seed(
            training_count=training_count,
            readout_count=readout_count,
            model_epochs=model_epochs,
            degree=degree,
            seed=int(seed),
        )
        rows.append(row)
        atomic_write_json(
            checkpoint_path,
            {"version": version, "rows": rows},
        )
        print(
            f"[{len(rows)}/{len(seeds)}] seed={seed}, "
            f"MAE={row['dynamics_fit']['heldout_circular_mae_rad']:.3f}, "
            f"pair={row['modules']['pairwise']['triple_residual_fraction']:.3f}, "
            f"triad={row['modules']['triadic']['triple_residual_fraction']:.3f}, "
            f"elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )
    payload = {
        "experiment_contract": {
            "data_generator": (
                "six-oscillator pairwise-triangle plus triadic-hyperedge "
                "finite-time stochastic Kuramoto channel"
            ),
            "mlp_target": (
                "complete six-dimensional wrapped phase increment over tau=0.20"
            ),
            "mlp_input": (
                "identical first- and second-harmonic sine/cosine features "
                "for all six oscillators"
            ),
            "intervention": "independent Uniform(-pi, pi) phases",
            "ei_estimator": (
                "module-local degree-3 polynomial triangular transport maps, "
                "matching the oracle comparison"
            ),
            "decomposition": "shared nonnegative-tolerant Greedy Phi hierarchy",
            "fixed": {
                "training_count": int(training_count),
                "readout_count": int(readout_count),
                "model_epochs": int(model_epochs),
                "tau": 0.20,
                "dt": 0.01,
                "process_noise": NOISE_SCALE,
                "cross_coupling": CROSS_COUPLING,
                "frequencies": FREQUENCIES.tolist(),
                "pairwise_total_coupling": PAIRWISE_COUPLING,
                "triadic_rms_matched_coupling": TRIADIC_COUPLING,
                "tm_degree": int(degree),
                "nonnegative_tolerance_bits": NONNEGATIVE_TOLERANCE_BITS,
            },
        },
        "rows": rows,
        "summary": aggregate(rows),
        "diagnostic_note": (
            "A separate six-dimensional conditional-flow smoke test underfit "
            "the triadic dependence; it is not used for this oracle-matched "
            "module comparison."
        ),
    }
    atomic_write_json(
        status_path,
        {
            "phase": "complete",
            "current": len(rows),
            "total": len(seeds),
            "unit": "seed",
            "elapsed_seconds": float(time.perf_counter() - started),
            "metrics": {
                "mae_rad": payload["summary"][
                    "heldout_circular_mae_rad_mean"
                ],
                "delta_triple_fraction": payload["summary"][
                    "paired_delta_triple_fraction"
                ]["mean"],
            },
            "updated_at": time.time(),
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--training-count", type=int, default=4800)
    parser.add_argument("--readout-count", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    args = parser.parse_args()
    configure_learned_base(smoke=False)
    payload = build_summary(
        seeds=tuple(range(int(args.seeds))),
        training_count=int(args.training_count),
        readout_count=int(args.readout_count),
        model_epochs=int(args.epochs),
        degree=int(args.degree),
        checkpoint_path=args.checkpoint,
        status_path=args.status,
    )
    atomic_write_json(args.result, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Saved result: {args.result}")
    print(f"Saved status: {args.status}")


if __name__ == "__main__":
    main()
