"""Compare equal-size pairwise and triadic Kuramoto modules.

The six-oscillator oracle field contains a three-node pairwise module and a
three-node irreducible triadic module. A weak nonzero pairwise coupling joins
the modules. Paired maximum-entropy phase interventions and one shared target
density are used to compare the Greedy Phi atom composition of the two blocks.
"""

from __future__ import annotations

import argparse
import itertools
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

from exp.TM.transport_map_density import (
    LOG_2,
    fit_polynomial_triangular_transport_map_density,
)
from scripts.phi_hierarchy import (
    NONNEGATIVE_TOLERANT,
    PhiAtom,
    all_nonempty_subsets,
    greedy_phi_atoms,
    nontrivial_bipartitions,
    subset_phi_raw,
)


NAMES = tuple(f"theta{i}" for i in range(1, 7))
PAIRWISE_MODULE = NAMES[:3]
TRIADIC_MODULE = NAMES[3:]
MODULES = {
    "pairwise": PAIRWISE_MODULE,
    "triadic": TRIADIC_MODULE,
}
FREQUENCIES = np.array([-0.40, 0.00, 0.40, -0.40, 0.00, 0.40], dtype=float)
PAIRWISE_COUPLING = 1.50
# Equalize the RMS local-coupling response under independent uniform phases:
# Var[Kp/2 (sin a + sin b)] = Var[Kt sin c].
TRIADIC_COUPLING = PAIRWISE_COUPLING / math.sqrt(2.0)
CROSS_COUPLING = 0.04
NOISE_SCALE = 0.08
NONNEGATIVE_TOLERANCE_BITS = 0.10
DEFAULT_RESULT = (
    ROOT / "results" / "equal_size_mixed_order_kuramoto" / "summary.json"
)


def mixed_order_derivative(phases: np.ndarray) -> np.ndarray:
    """Evaluate the controlled six-oscillator mixed-order vector field."""
    values = np.asarray(phases, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(NAMES):
        raise ValueError(f"phases must have shape (n_samples, {len(NAMES)}).")
    derivative = np.broadcast_to(FREQUENCIES, values.shape).copy()

    for receiver in range(3):
        senders = [index for index in range(3) if index != receiver]
        derivative[:, receiver] += (PAIRWISE_COUPLING / 2.0) * sum(
            np.sin(values[:, sender] - values[:, receiver])
            for sender in senders
        )

    for receiver in range(3, 6):
        senders = [index for index in range(3, 6) if index != receiver]
        derivative[:, receiver] += TRIADIC_COUPLING * np.sin(
            values[:, senders[0]]
            + values[:, senders[1]]
            - 2.0 * values[:, receiver]
        )

    for left in range(3):
        for right in range(3, 6):
            delta = values[:, right] - values[:, left]
            contribution = (CROSS_COUPLING / 3.0) * np.sin(delta)
            derivative[:, left] += contribution
            derivative[:, right] -= contribution
    return derivative


def paired_data(*, sample_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate paired interventions and noisy full-system velocities."""
    rng = np.random.default_rng(int(seed))
    phases = rng.uniform(-math.pi, math.pi, size=(int(sample_count), len(NAMES)))
    target = mixed_order_derivative(phases)
    target += NOISE_SCALE * rng.normal(size=target.shape)
    return phases, target


def source_blocks(phases: np.ndarray) -> dict[str, np.ndarray]:
    """Use the same two-harmonic representation for every oscillator."""
    values = np.asarray(phases, dtype=float)
    return {
        name: np.column_stack(
            (
                np.cos(values[:, index]),
                np.sin(values[:, index]),
                np.cos(2.0 * values[:, index]),
                np.sin(2.0 * values[:, index]),
            )
        )
        for index, name in enumerate(NAMES)
    }


def estimate_module_ei_table(
    phases: np.ndarray,
    target: np.ndarray,
    *,
    module: Sequence[str],
    degree: int,
) -> dict[tuple[str, ...], float]:
    """Estimate the seven within-module EI values against one full target."""
    blocks = source_blocks(phases)
    target_array = np.asarray(target, dtype=float)
    target_model = fit_polynomial_triangular_transport_map_density(
        target_array,
        degree=int(degree),
    )
    target_log_prob = target_model.log_prob(target_array)
    table: dict[tuple[str, ...], float] = {}
    for subset in all_nonempty_subsets(module):
        source = np.column_stack([blocks[name] for name in subset])
        joint = np.column_stack((target_array, source))
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
            - target_log_prob
        ) / LOG_2
        estimate = float(np.mean(pointwise))
        if estimate < -NONNEGATIVE_TOLERANCE_BITS:
            raise RuntimeError(
                f"EI{tuple(subset)}={estimate:.6g} bits is below "
                f"-{NONNEGATIVE_TOLERANCE_BITS:.6g} bits."
            )
        table[tuple(subset)] = 0.0 if estimate < 0.0 else estimate
    return table


def audit_phi_table(
    module: Sequence[str],
    table: Mapping[tuple[str, ...], float],
) -> dict[str, object]:
    """Fail on significant nonnegativity violations and record near-zero cases."""
    ordered = tuple(module)
    singleton = {name: float(table[(name,)]) for name in ordered}
    phi = {
        subset: subset_phi_raw(subset, table, singleton)
        for subset in all_nonempty_subsets(ordered)
    }
    residuals = [
        (
            float(phi[subset] - phi[left] - phi[right]),
            subset,
            left,
            right,
        )
        for subset in phi
        if len(subset) > 1
        for left, right in nontrivial_bipartitions(subset)
    ]
    minimum_phi, minimum_phi_subset = min(
        (value, subset) for subset, value in phi.items() if len(subset) > 1
    )
    minimum_residual, parent, left, right = min(residuals)
    if minimum_phi < -NONNEGATIVE_TOLERANCE_BITS:
        affected = sum(
            value < -NONNEGATIVE_TOLERANCE_BITS
            for subset, value in phi.items()
            if len(subset) > 1
        )
        raise RuntimeError(
            f"Significant Phi nonnegativity violation: min={minimum_phi:.6g} bits, "
            f"threshold=-{NONNEGATIVE_TOLERANCE_BITS:.6g}, affected={affected}, "
            f"subset={minimum_phi_subset}."
        )
    if minimum_residual < -NONNEGATIVE_TOLERANCE_BITS:
        affected = sum(
            value < -NONNEGATIVE_TOLERANCE_BITS
            for value, *_ in residuals
        )
        raise RuntimeError(
            "Significant hierarchical nonnegativity violation: "
            f"min={minimum_residual:.6g} bits, "
            f"threshold=-{NONNEGATIVE_TOLERANCE_BITS:.6g}, affected={affected}, "
            f"partition={parent}->{left}|{right}."
        )
    return {
        "tolerance_bits": NONNEGATIVE_TOLERANCE_BITS,
        "minimum_phi_bits": float(minimum_phi),
        "minimum_phi_subset": list(minimum_phi_subset),
        "near_zero_negative_phi_count": int(
            sum(
                -NONNEGATIVE_TOLERANCE_BITS <= value < 0.0
                for subset, value in phi.items()
                if len(subset) > 1
            )
        ),
        "minimum_partition_residual_bits": float(minimum_residual),
        "minimum_residual_partition": {
            "parent": list(parent),
            "left": list(left),
            "right": list(right),
        },
        "near_zero_negative_residual_count": int(
            sum(
                -NONNEGATIVE_TOLERANCE_BITS <= value < 0.0
                for value, *_ in residuals
            )
        ),
    }


def atom_summary(
    module: Sequence[str],
    atoms: Sequence[PhiAtom],
) -> dict[str, float]:
    """Summarize pair-versus-triple mass in a three-source Greedy tree."""
    ordered = tuple(module)
    triple = float(
        sum(atom.value for atom in atoms if tuple(atom.sources) == ordered)
    )
    pair = float(sum(atom.value for atom in atoms if len(atom.sources) == 2))
    total = float(sum(atom.value for atom in atoms if atom.value > 0.0))
    return {
        "pair_atom_bits": pair,
        "triple_residual_bits": triple,
        "positive_atom_bits": total,
        "pair_atom_fraction": pair / total if total > 0.0 else 0.0,
        "triple_residual_fraction": triple / total if total > 0.0 else 0.0,
    }


def run_seed(*, sample_count: int, seed: int, degree: int) -> dict[str, object]:
    """Run the paired mechanism comparison for one random seed."""
    started = time.perf_counter()
    phases, target = paired_data(sample_count=sample_count, seed=seed)
    modules: dict[str, object] = {}
    for mechanism, module in MODULES.items():
        table = estimate_module_ei_table(
            phases,
            target,
            module=module,
            degree=degree,
        )
        audit = audit_phi_table(module, table)
        singleton = {name: float(table[(name,)]) for name in module}
        atoms = greedy_phi_atoms(
            module,
            table,
            policy=NONNEGATIVE_TOLERANT,
            eps=1.0e-5,
            split_tolerance=NONNEGATIVE_TOLERANCE_BITS,
            singleton_ei=singleton,
        )
        modules[mechanism] = {
            "sources": list(module),
            **atom_summary(module, atoms),
            "nonnegative_audit": audit,
            "atoms": [
                {
                    "sources": list(atom.sources),
                    "value_bits": float(atom.value),
                    "kind": atom.kind,
                    "depth": int(atom.depth),
                }
                for atom in sorted(
                    atoms,
                    key=lambda atom: atom.value,
                    reverse=True,
                )
                if atom.value > 0.0
            ],
            "ei_bits": {
                "+".join(subset): float(value)
                for subset, value in table.items()
            },
        }
    return {
        "seed": int(seed),
        "modules": modules,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate paired mechanism contrasts over seeds."""
    output: dict[str, object] = {}
    for mechanism in MODULES:
        module_rows = [row["modules"][mechanism] for row in rows]
        metrics: dict[str, float] = {}
        for metric in (
            "pair_atom_bits",
            "triple_residual_bits",
            "positive_atom_bits",
            "pair_atom_fraction",
            "triple_residual_fraction",
        ):
            values = np.asarray(
                [float(row[metric]) for row in module_rows],
                dtype=float,
            )
            metrics[f"{metric}_mean"] = float(np.mean(values))
            metrics[f"{metric}_sem"] = (
                float(np.std(values, ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
        output[mechanism] = metrics
    paired_delta = np.asarray(
        [
            float(row["modules"]["triadic"]["triple_residual_fraction"])
            - float(row["modules"]["pairwise"]["triple_residual_fraction"])
            for row in rows
        ]
    )
    output["paired_delta_triple_fraction"] = {
        "values": paired_delta.tolist(),
        "mean": float(np.mean(paired_delta)),
        "sem": (
            float(np.std(paired_delta, ddof=1) / math.sqrt(len(paired_delta)))
            if len(paired_delta) > 1
            else 0.0
        ),
        "positive_count": int(np.sum(paired_delta > 0.0)),
        "n_seeds": int(len(paired_delta)),
    }
    return output


def build_summary(
    *,
    seeds: Sequence[int],
    sample_count: int,
    degree: int,
) -> dict[str, object]:
    """Run and package the complete controlled comparison."""
    rows = []
    for index, seed in enumerate(seeds, start=1):
        row = run_seed(sample_count=sample_count, seed=int(seed), degree=degree)
        rows.append(row)
        print(
            f"[{index}/{len(seeds)}] seed={seed}, "
            f"triple fraction pairwise="
            f"{row['modules']['pairwise']['triple_residual_fraction']:.3f}, "
            f"triadic={row['modules']['triadic']['triple_residual_fraction']:.3f}, "
            f"elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )
    return {
        "experiment_contract": {
            "question": (
                "What changes in the Greedy hierarchy when only local "
                "Kuramoto interaction order changes?"
            ),
            "treatment": {
                "pairwise": list(PAIRWISE_MODULE),
                "triadic": list(TRIADIC_MODULE),
            },
            "fixed": {
                "module_size": 3,
                "frequencies_per_module": FREQUENCIES[:3].tolist(),
                "pairwise_total_coupling": PAIRWISE_COUPLING,
                "triadic_coupling_rms_matched": TRIADIC_COUPLING,
                "cross_coupling": CROSS_COUPLING,
                "noise_scale": NOISE_SCALE,
                "sample_count": int(sample_count),
                "transport_degree": int(degree),
                "source_features": "first and second circular harmonics",
                "target": "full six-dimensional instantaneous phase velocity",
                "intervention": "independent Uniform(-pi, pi) phases",
            },
            "primary_metric": "triple-residual fraction of positive Greedy atom mass",
            "paired_unit": "seed; phases, target noise, target density, support, and estimator",
            "nonnegative_tolerance_bits": NONNEGATIVE_TOLERANCE_BITS,
        },
        "rows": rows,
        "summary": aggregate(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_count = int(
        args.sample_count or (400 if args.mode == "smoke" else 1200)
    )
    payload = build_summary(
        seeds=tuple(range(int(args.seeds))),
        sample_count=sample_count,
        degree=int(args.degree),
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Saved result: {args.result}")


if __name__ == "__main__":
    main()
