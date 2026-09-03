#!/usr/bin/env python3
"""Compute the earlier panel-d module-local polynomial-TM atoms across K_out."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from exp.TM.transport_map_density import fit_polynomial_triangular_transport_map_density
from scripts import run_mixed_order_kuramoto_kout_main as experiment
from scripts.phi_hierarchy import NONNEGATIVE_TOLERANT, greedy_phi_atoms
from scripts.validate_equal_size_mixed_order_kuramoto import audit_phi_table, atom_summary
from scripts.validate_equal_size_mixed_order_kuramoto_mlp import estimate_module_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-out", type=float, nargs="+", default=(0.0, 0.04, 5.0))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training-count", type=int, default=4800)
    parser.add_argument("--readout-count", type=int, default=4000)
    parser.add_argument("--model-epochs", type=int, default=800)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mixed_order_kuramoto_kout_main/module_polynomial_metrics.json"),
    )
    args = parser.parse_args()
    experiment._configure_coherent_tm()
    from scripts.classic_network_dynamics_benchmark import fit_mlp

    rows: list[dict[str, object]] = []
    for k_out in args.k_out:
        phases, future = experiment.coherent_tm.generated_transition_data(
            sample_count=args.training_count,
            seed=args.seed + 1_000,
            pairwise_coupling=experiment.WITHIN_COUPLING,
            triadic_coupling=experiment.WITHIN_COUPLING / math.sqrt(2.0),
            cross_coupling=float(k_out),
            process_noise=experiment.DYNAMICS_NOISE_SCALE,
            tau=experiment.PREDICTION_HORIZON,
            dt=experiment.INTEGRATION_STEP,
        )
        target_increment = np.angle(np.exp(1j * (future - phases)))
        features = experiment._mlp_phase_features(phases)
        fitted = fit_mlp(features, target_increment, seed=args.seed + 2_000, epochs=args.model_epochs)
        split = max(32, int(0.8 * len(features)))
        residual = target_increment[split:] - np.asarray(fitted.predict(features[split:]), dtype=float)
        covariance = np.asarray(np.cov(residual, rowvar=False), dtype=float)
        covariance += 1.0e-6 * np.eye(len(experiment.NAMES))
        rng = np.random.default_rng(args.seed + 3_000)
        readout_phases = rng.uniform(
            -math.pi,
            math.pi,
            size=(args.readout_count, len(experiment.NAMES)),
        )
        mean = np.asarray(
            fitted.predict(experiment._mlp_phase_features(readout_phases)),
            dtype=float,
        )
        target = mean + np.random.default_rng(args.seed + 4_000).multivariate_normal(
            np.zeros(len(experiment.NAMES)),
            covariance,
            size=args.readout_count,
        )
        target_model = fit_polynomial_triangular_transport_map_density(target, degree=args.degree)
        target_log_probability = target_model.log_prob(target)
        modules: dict[str, object] = {}
        for label, module in (
            ("pairwise", experiment.PAIRWISE_MODULE),
            ("triadic", experiment.TRIADIC_MODULE),
        ):
            table, ei_audit = estimate_module_table(
                readout_phases,
                target,
                module=module,
                target_log_probability=target_log_probability,
                degree=args.degree,
            )
            phi_audit = audit_phi_table(module, table)
            atoms = greedy_phi_atoms(
                module,
                table,
                policy=NONNEGATIVE_TOLERANT,
                split_tolerance=experiment.SYN_TOLERANCE_BITS,
                singleton_ei={name: float(table[(name,)]) for name in module},
            )
            modules[label] = {
                **atom_summary(module, atoms),
                "ei_audit": ei_audit,
                "phi_audit": phi_audit,
                "atoms": [
                    {"sources": list(atom.sources), "value_bits": float(atom.value)}
                    for atom in atoms
                    if atom.value > 0.0
                ],
            }
        row = {"k_out": float(k_out), "modules": modules}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    payload = {
        "contract": {
            "target": "complete six-dimensional wrapped future phase increment",
            "training_count": args.training_count,
            "readout_count": args.readout_count,
            "model_epochs": args.model_epochs,
            "degree": args.degree,
            "tau": experiment.PREDICTION_HORIZON,
            "process_noise": experiment.DYNAMICS_NOISE_SCALE,
            "only_treatment": "K_out",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
