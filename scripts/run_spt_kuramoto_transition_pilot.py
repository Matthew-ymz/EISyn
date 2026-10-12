#!/usr/bin/env python3
"""Paired learned-component / full-channel K_out pilot, with resumable audits."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_mixed_order_kuramoto_kout_main as main
from scripts.spt import flatten_nodes

GRID = (0, .04, .1, .2, .4, .6, .8, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5)
TRACKS = ("learned_components", "full_channel")
OUT = ROOT / "results/spt_kuramoto_transition_pilot"
STATUS = ROOT / "docs/log/spt_kuramoto_transition_pilot/status.json"


def save(path, payload):
    main._write_status(Path(path), payload)


def digest(array):
    values = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str((values.shape, str(values.dtype))).encode())
    h.update(values.tobytes())
    return h.hexdigest()


def table_rows(table):
    return [{"sources": list(s), "ei_bits": float(v)} for s, v in table.items()]


def partition_audit(table):
    singles = {name: table[(name,)] for name in main.NAMES}
    xi = {s: main.subset_phi_raw(s, table, singles) for s in table}
    candidates = []
    for s, value in xi.items():
        for left, right in main.nontrivial_bipartitions(s):
            score = xi[left] + xi[right]
            candidates.append({"sources": list(s), "left": list(left),
                               "right": list(right), "score_bits": float(score),
                               "syn_bits": float(value - score)})
    syn = np.asarray([r["syn_bits"] for r in candidates])
    block = np.asarray([v for s, v in xi.items() if len(s) > 1])
    tol = main.SYN_TOLERANCE_BITS
    audit = {"tolerance_bits": tol, "candidate_count": len(syn),
             "minimum_partition_syn_bits": float(syn.min()),
             "minimum_non_singleton_xi_bits": float(block.min()),
             "tolerance_negative_partition_count": int(np.sum((syn < 0) & (syn >= -tol))),
             "tolerance_negative_xi_count": int(np.sum((block < 0) & (block >= -tol))),
             "violating_partition_count": int(np.sum(syn < -tol)),
             "violating_xi_count": int(np.sum(block < -tol)),
             "raw_values_retained": True, "nonnegative_clipping": False}
    return candidates, audit


def prepare_channel(k_out, seed, args):
    """Generate and fit once; both tracks receive these exact same arrays."""
    from scripts.classic_network_dynamics_benchmark import fit_mlp
    tm = main.coherent_tm
    phases_train, future = tm.generated_transition_data(
        sample_count=args.training_count, seed=seed + 1000,
        pairwise_coupling=main.WITHIN_COUPLING,
        triadic_coupling=main.WITHIN_COUPLING / math.sqrt(2),
        cross_coupling=k_out, process_noise=main.DYNAMICS_NOISE_SCALE,
        tau=main.PREDICTION_HORIZON, dt=main.INTEGRATION_STEP)
    increment = np.angle(np.exp(1j * (future - phases_train)))
    features = main._mlp_phase_features(phases_train)
    fitted = fit_mlp(features, increment, seed=seed + 2000, epochs=args.model_epochs)
    split = max(32, int(.8 * len(features)))
    predicted = np.asarray(fitted.predict(features[split:]), dtype=float)
    residual = increment[split:] - predicted
    covariance = np.asarray(np.cov(residual, rowvar=False), dtype=float)
    covariance += 1e-6 * np.eye(6)
    components, effects = main.learned_dependency_components(fitted, seed=seed + 2500)
    phases = np.random.default_rng(seed + 3000).uniform(-math.pi, math.pi,
                                                       (args.readout_count, 6))
    target = np.asarray(fitted.predict(main._mlp_phase_features(phases)), dtype=float)
    target += np.random.default_rng(seed + 4000).multivariate_normal(
        np.zeros(6), covariance, size=len(phases))
    diagnostics = {
        "heldout_mae": float(np.abs(residual).mean()),
        "heldout_mse": float(np.mean(residual ** 2)),
        "constant_baseline_mse": float(fitted.baseline_mse),
        "per_source_r2": {name: float(1 - np.mean(residual[:, i] ** 2) /
                            max(np.var(increment[split:, i]), 1e-12))
                          for i, name in enumerate(main.NAMES)},
        "dependency_threshold_output_sd": .25,
        "dependency_effect_matrix": effects.tolist(),
        "learned_dependency_components": [[main.NAMES[i] for i in c] for c in components],
        "maximum_cross_module_effect": float(max(effects[:3, 3:].max(), effects[3:, :3].max())),
        "residual_covariance": covariance.tolist(),
    }
    hashes = {"training_phases": digest(phases_train), "training_increment": digest(increment),
              "readout_phases": digest(phases), "readout_target_increment": digest(target)}
    del fitted
    gc.collect()
    return phases, target, components, diagnostics, hashes


def estimate_track(phases, target, components, k_out, seed, track, args, dynamics, hashes):
    started = time.monotonic()
    record = {"k_out": float(k_out), "seed": seed, "track": track,
              "shared_channel_hashes": hashes, "dynamics_diagnostics": dynamics,
              "component_fit_diagnostics": [], "component_marginal_diagnostics": [],
              "estimator_components": [[main.NAMES[i] for i in c] for c in components]}
    tables = []
    try:
        for ci, component in enumerate(components):
            names = tuple(main.NAMES[i] for i in component)
            main.coherent_tm.phase_transport_context = (
                lambda values, names=names: main._numpy_context_for_names(values, names))
            flow, fit = main.coherent_tm.fit_joint_conditional_transport_map(
                phases[:, component], target[:, component], train_fraction=.8,
                epochs=args.tm_epochs, seed=seed + 5000 + ci,
                target_scaling="per_dimension")
            record["component_fit_diagnostics"].append({"sources": list(names), **fit})
            try:
                local, marginal = main.coherent_ei_table_without_resampling(
                    flow, source_names=names, seed=seed + 6000 + ci,
                    evaluation_count=args.evaluation_count, marginal_samples=args.marginal_samples,
                    context_active=bool(fit["context_active"]))
            except main.JointTMNonnegativityError as error:
                record["failed_component_ei_table"] = table_rows(error.table)
                record["component_marginal_diagnostics"].append(
                    {"sources": list(names), **error.diagnostics})
                raise
            finally:
                del flow
            record["component_marginal_diagnostics"].append({"sources": list(names), **marginal})
            tables.append((names, local))
        table = main.combine_independent_component_ei_tables(tables)
        record["ei_table"] = table_rows(table)
        candidates, audit = partition_audit(table)
        record["partition_candidates"] = candidates
        record["nonnegative_audit"] = audit
        if audit["violating_partition_count"] or audit["violating_xi_count"]:
            raise RuntimeError(f"Combined EI nonnegativity violation: {audit}")
        tree = main.free_spt(table)
        nodes = flatten_nodes(tree)
        record["nodes"] = [{"sources": list(n.sources), "xi_bits": n.xi_value,
                            "syn_bits": n.syn_value, "depth": n.depth,
                            "children": [list(c.sources) for c in n.children]} for n in nodes]
        root_candidates = sorted((c for c in candidates if len(c["sources"]) == 6),
                                 key=lambda c: (-c["score_bits"], c["left"], c["right"]))
        planted = next(c for c in root_candidates if c["left"] == list(main.NAMES[:3]))
        others = [c for c in root_candidates if c is not planted]
        root_split = [list(c.sources) for c in tree.children]
        planted_root = {frozenset(c.sources) for c in tree.children} == {
            frozenset(main.NAMES[:3]), frozenset(main.NAMES[3:])}
        record.update({
            "status": "valid", "root_split": root_split,
            "planted_root_split": planted_root,
            "root_xi_bits": tree.xi_value, "root_syn_bits": tree.syn_value,
            "root_syn_fraction": tree.syn_value / tree.xi_value if tree.xi_value else None,
            "root_best_second_gap_bits": root_candidates[0]["score_bits"] - root_candidates[1]["score_bits"],
            "planted_vs_best_other_margin_bits": planted["score_bits"] - max(c["score_bits"] for c in others),
            "sackin_index": sum(n.depth for n in nodes if not n.children),
            "maximum_leaf_depth": max(n.depth for n in nodes if not n.children),
            "nontrivial_clades": [list(n.sources) for n in nodes if 1 < len(n.sources) < 6],
            "closure_error_bits": sum(n.syn_value for n in nodes) - tree.xi_value,
            "selected_tolerance_negative_count": sum(n.syn_value < 0 for n in nodes),
            "zero_point_control": ("pass" if abs(tree.syn_value) <= main.SYN_TOLERANCE_BITS
                                   and planted_root else "fail") if k_out == 0 else "not_applicable",
        })
    except Exception as error:
        record.update({"status": "failed", "error_type": type(error).__name__,
                       "error": str(error), "traceback": traceback.format_exc()})
        print(f"FAILED seed={seed} K_out={k_out:g} {track}: {error}", flush=True)
    record["elapsed_seconds"] = time.monotonic() - started
    return record


def contract(args):
    import torch
    import nflows  # Fail before launching the sweep when the TM dependency is absent.
    code = [Path(__file__), Path(main.__file__),
            ROOT / "scripts/validate_mixed_order_kuramoto_hierarchy.py",
            ROOT / "scripts/classic_network_dynamics_benchmark.py", ROOT / "scripts/spt.py"]
    return {
        "grid": args.k_out, "seeds": args.seeds, "tracks": list(TRACKS),
        "training_count": args.training_count, "readout_count": args.readout_count,
        "model_epochs": args.model_epochs, "tm_epochs": args.tm_epochs,
        "evaluation_count": args.evaluation_count, "marginal_samples": args.marginal_samples,
        "source": "independent Uniform(-pi,pi)^6 maximum-entropy phase interventions",
        "target": "fixed complete six-dimensional wrapped increment, tau=0.20, dt=0.01",
        "couplings": {"within": 1.5, "triadic": 1.5 / math.sqrt(2), "cross_edge": "K_out/3"},
        "process_noise": .08, "residual_covariance_ridge": 1e-6,
        "component_threshold_output_sd": .25, "syn_tolerance_bits": main.SYN_TOLERANCE_BITS,
        "nonnegative_clipping": False, "split_objective": "raw residual, exact all bipartitions",
        "pairing": "one dynamics fit and exact shared training/readout arrays per (K_out, seed); SHA256 verified",
        "ei_sampling": "native local Sobol/modeled-target sampling; dimensions differ when factorized",
        "reuse": "when learned components contain all six nodes, full track reuses the identical joint TM estimates",
        "manuscript": {"parent": "P6UJCVG8", "main_attachment": "DXGC7JEA", "supplement": "MWIWKSVG",
                       "version": "latest available attachments at task audit; no explicit draft date/version",
                       "relevant_locations": "main Eq1-3, Methods Eq5-12; SI S1.2, S5 AlgorithmS1, S7"},
        "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                    "nflows": importlib.metadata.version("nflows")},
    }


def run(args):
    main._configure_coherent_tm()
    cfg = contract(args)
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    contract_path = args.output / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text())["fingerprint"] != fingerprint:
        raise RuntimeError("Existing pilot contract differs; choose a separate output directory.")
    save(contract_path, {"fingerprint": fingerprint, "contract": cfg})
    started = time.monotonic()
    records = []
    count = len(args.k_out) * len(args.seeds) * 2
    for k_out, seed in itertools.product(args.k_out, args.seeds):
        if time.monotonic() - started >= args.max_minutes * 60:
            print("Compute budget reached; no further conditions launched.", flush=True)
            break
        prefix = f"kout_{k_out:g}".replace(".", "p") + f"_seed_{seed}"
        paths = {track: args.output / f"{prefix}_{track}.json" for track in TRACKS}
        if all(p.exists() for p in paths.values()):
            records.extend(json.loads(paths[t].read_text()) for t in TRACKS)
            continue
        save(args.status, {"phase": "running", "completed": len(records), "total": count,
                           "k_out": k_out, "seed": seed, "pid": os.getpid(), "updated_at": time.time()})
        print(f"PAIR K_out={k_out:g} seed={seed}: shared dynamics fit", flush=True)
        pair_start = time.monotonic()
        phases, target, components, dynamics, hashes = prepare_channel(k_out, seed, args)
        native = None
        for track in TRACKS:
            if paths[track].exists():
                record = json.loads(paths[track].read_text())
                if record["shared_channel_hashes"] != hashes:
                    raise RuntimeError("Resumed channel hashes differ from existing paired record.")
            elif track == "full_channel" and components == [tuple(range(6))] and native is not None:
                record = copy.deepcopy(native)
                record.update({"track": track, "reused_from": "learned_components", "elapsed_seconds": 0.0})
            else:
                print(f"ESTIMATE K_out={k_out:g} seed={seed}: {track}", flush=True)
                record = estimate_track(phases, target,
                    components if track == "learned_components" else [tuple(range(6))],
                    k_out, seed, track, args, dynamics, hashes)
            record["contract_fingerprint"] = fingerprint
            save(paths[track], record)
            records.append(record)
            if track == "learned_components":
                native = record
            print(f"DONE K_out={k_out:g} seed={seed}: {track} {record['status']} "
                  f"root={record.get('root_split')} margin={record.get('planted_vs_best_other_margin_bits')}", flush=True)
        if records[-1]["shared_channel_hashes"] != records[-2]["shared_channel_hashes"]:
            raise RuntimeError("Paired readout hashes differ.")
        print(f"PAIR elapsed={time.monotonic() - pair_start:.1f}s", flush=True)
        gc.collect()
    payload = {"contract": cfg, "fingerprint": fingerprint, "conditions": records,
               "elapsed_seconds": time.monotonic() - started,
               "coverage": {"completed": len(records), "planned": count,
                            "failed": sum(r["status"] == "failed" for r in records)}}
    save(args.output / "summary.json", payload)
    save(args.status, {"phase": "complete" if len(records) == count else "budget_stopped",
                       **payload["coverage"], "pid": os.getpid(), "updated_at": time.time(),
                       "result": str(args.output / "summary.json")})
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-out", type=float, nargs="+", default=list(GRID))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--training-count", type=int, default=4800)
    parser.add_argument("--readout-count", type=int, default=4000)
    parser.add_argument("--model-epochs", type=int, default=800)
    parser.add_argument("--tm-epochs", type=int, default=100)
    parser.add_argument("--evaluation-count", type=int, default=512)
    parser.add_argument("--marginal-samples", type=int, default=256)
    parser.add_argument("--max-minutes", type=float, default=90)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--status", type=Path, default=STATUS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        save(args.status, {"phase": "failed", "error": str(error), "pid": os.getpid(), "updated_at": time.time()})
        raise
