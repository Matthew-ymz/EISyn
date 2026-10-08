#!/usr/bin/env python3
"""Paired G=0..1.70, dG=0.01 DMF trees using the original figure's search.

Resumes per-condition NPZ/JSON caches. No extra seeds or finer grid are launched.
Use --preflight to reproduce the existing G=1.3 simulation and tree first.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_dmf_critical_phi_hierarchy_topology import conditional_source_covariance
from scripts.analyze_dmf_schaefer100_xi_hierarchy_tree import _tree_metrics, flatten_nodes, pairwise_syn_affinity
from scripts.analyze_runge_slp_pc60_xi_hierarchy import _node_record
from scripts.compare_runge_slp_pc60_xi_horizons import _node_from_record
from scripts.plot_dmf_schaefer100_tree_examples import CachedScalarLogdetXiOracle, audit_tree, sha256
from scripts.run_dmf_diffusive_fullstate_control import rollout
from scripts.run_dmf_paired_spt_pilot import paired_sources, noise_seed
from scripts.spt import SPTConfig, build_spt, spectral_candidate_selector
from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module, standardize

OUTPUT = ROOT / "results/dmf_schaefer100/xi_hierarchy_tree/G_resolution_seed04"
SOURCE = ROOT / "results/dmf_schaefer100/source/group_mean_native_mean_rate.npz"
OLD_COV = ROOT / "results/dmf_schaefer100/unconstrained_spt_wide/shards/seeds3_4/covariance"
OLD_TREE = ROOT / "results/dmf_schaefer100/xi_hierarchy_tree/examples/wide/seed04_G1.30.json"
LABELS = ROOT / "results/dmf_schaefer100/full/critical_yeo7.npz"
TOL = 1e-8  # bits; this absorbs numerical roundoff, not statistical background
SEED = 4
CODE_PATHS = (
    "exp/brain/dmf_fig6.py", "scripts/run_dmf_diffusive_fullstate_control.py",
    "scripts/run_dmf_paired_spt_pilot.py", "scripts/run_dmf_fixed_uniform_multihorizon.py",
    "scripts/validate_dmf_83_region_oracle_phi_eid.py",
    "scripts/analyze_dmf_critical_phi_hierarchy_topology.py", "scripts/spt.py",
    "scripts/analyze_runge_slp_pc60_xi_hierarchy.py",
    "scripts/analyze_dmf_schaefer100_xi_hierarchy_tree.py",
    "scripts/plot_dmf_schaefer100_tree_examples.py",
    "scripts/run_dmf_spt_G_resolution.py",
)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    temporary.replace(path)


def fingerprints() -> dict:
    return {name: sha256(ROOT / name) for name in CODE_PATHS}


def context():
    with np.load(OLD_COV / "seed04_G1.30.npz") as a:
        simulation = json.loads(str(a["config_json"].item()))
        expected_digest = str(a["input_sha256"].item())
    if simulation["source_sha256"] != sha256(SOURCE):
        raise ValueError("The structural connectome / JFIC source has changed")
    required = {"sample_count": 2048, "support": [.3, .7], "horizon": 300, "dt": .001,
                "sigma": .01, "ridge": 1e-6, "boundary": "none", "JFIC_reference_G": 1.,
                "source_seed_offset": 31000, "noise_seed_offset": 31017,
                "paired_inputs_across_G": True, "paired_noise_across_G": True}
    if any(simulation.get(key) != value for key, value in required.items()):
        raise ValueError("Cached simulation differs from the approved protocol")
    with np.load(SOURCE) as a:
        sc = a["connectivity"].copy()
        jf = a["j_fic"].copy()
        index = np.flatnonzero(np.isclose(a["G"], 1.))
    if sc.shape != (100, 100) or len(index) != 1 or not np.array_equal(jf, np.broadcast_to(jf[index[0]], jf.shape)):
        raise ValueError("Require the original 100-ROI SC and frozen G=1 JFIC")
    se, si = paired_sources(SEED, 2048, 100)
    source = np.concatenate((se, si), axis=1)
    digest = hashlib.sha256(source.tobytes()).hexdigest()
    if digest != expected_digest:
        raise ValueError("Actual paired input draws differ from the legacy cache")
    with np.load(LABELS, allow_pickle=True) as a:
        labels = a["region_labels"].tolist()
        names = a["network_names"].tolist()
        membership = a["network_membership"].tolist()
    return simulation, digest, sc, jf[index[0]], se, si, standardize(source)[0], labels, names, membership


def simulate(g, ctx):
    simulation, digest, sc, jf, se, si, source, *_ = ctx
    started = time.monotonic()
    dmf = load_dmf_module()
    parameters = dmf.DMFParameters(t_total=1, burn_in=0, dt=.001, sigma=.01)
    # The same generator and exactly two (2048,100) noise draws per step at every G.
    te, ti = rollout(dmf, se, si, connectivity=sc, coupling_g=g, j_fic=jf, parameters=parameters,
                     mode="direct", state_boundary="none", horizon=300,
                     rng=np.random.default_rng(noise_seed(SEED)))
    target = np.concatenate((te, ti), axis=1)
    if not np.isfinite(target).all() or np.any(target.std(axis=0, ddof=1) <= 0):
        raise RuntimeError(f"Nonfinite or degenerate target: G={g:.3f}")
    _, conditional, _ = conditional_source_covariance(source, standardize(target)[0], ridge=1e-6)
    if not np.isfinite(conditional).all() or not np.allclose(conditional, conditional.T, atol=1e-12, rtol=0):
        raise RuntimeError(f"Invalid conditional covariance: G={g:.3f}")
    return conditional, {
        "elapsed_seconds": time.monotonic() - started,
        "target_min": float(target.min()), "target_max": float(target.max()),
        "target_minimum_sd": float(target.std(axis=0, ddof=1).min()),
        "target_outside_unit_interval_count": int(((target < 0) | (target > 1)).sum()),
    }


def covariance(g, ctx, output, code):
    simulation, digest, *_ = ctx
    local = output / "covariance" / f"seed04_G{g:.3f}.npz"
    old = OLD_COV / f"seed04_G{g:.2f}.npz"
    for path in (local, old):
        if not path.exists():
            continue
        with np.load(path) as a:
            if (json.loads(str(a["config_json"].item())) != simulation
                    or str(a["input_sha256"].item()) != digest
                    or int(a["seed"].item()) != SEED or float(a["G"].item()) != g):
                if path == local:
                    raise ValueError(f"Existing local cache provenance mismatch: {path}")
                continue
            if path == local and json.loads(str(a["code_sha256_json"].item())) != code:
                raise ValueError(f"Code changed since covariance was cached: {path}")
            return a["conditional_covariance"].copy(), path, True
    value, metadata = simulate(g, ctx)
    np.savez_compressed(local, conditional_covariance=value, seed=SEED, G=g,
                        config_json=json.dumps(simulation), input_sha256=digest,
                        code_sha256_json=json.dumps(code), **metadata)
    return value, local, False


def construct(conditional):
    started = time.monotonic()
    oracle = CachedScalarLogdetXiOracle(conditional, [(i, i + 100) for i in range(100)])
    affinity, pair_zero = pairwise_syn_affinity(oracle, 100, tolerance=TOL)
    generate = spectral_candidate_selector(affinity, exact_max_size=8)
    gaps = []

    def selector(sources):
        kind, candidates = generate(sources)
        candidates = list(candidates)
        scored = [(float(oracle.xi(left) + oracle.xi(right)), left, right) for left, right in candidates]
        # Observational audit only: return exactly the original candidates and kind.
        ranked = sorted(scored, key=lambda x: (-x[0], float(oracle.xi(sources) - x[0]), x[1], x[2]))
        gaps.append({"indices": list(sources), "candidate_count": len(ranked),
                     "best_children": [list(ranked[0][1]), list(ranked[0][2])],
                     "best_child_xi_sum_bits": ranked[0][0],
                     "second_child_xi_sum_bits": ranked[1][0] if len(ranked) > 1 else None,
                     "best_second_gap_bits": ranked[0][0] - ranked[1][0] if len(ranked) > 1 else None})
        return kind, candidates

    result = build_spt(tuple(range(100)), oracle,
                       config=SPTConfig(syn_tolerance=TOL, complete_to_singletons=True),
                       candidate_selector=selector)
    tree = result.root
    validation = audit_tree(tree, oracle.xi(range(100)))
    nodes = flatten_nodes(tree)
    internal = [n for n in nodes if n.children]
    leaves = [n for n in nodes if not n.children]
    selected = {n.indices: n for n in internal}
    if any(tuple(tuple(x) for x in gap["best_children"]) != tuple(c.indices for c in selected[tuple(gap["indices"])].children)
           for gap in gaps):
        raise RuntimeError("Candidate-gap audit rank differs from the original SPT selection")
    return {
        "tree": _node_record(tree), "validation": validation, "search_audit": asdict(result.audit),
        "pair_tolerance_zero_count": pair_zero,
        "scalar_logdet_cache_legacy_maximum_error_bits": oracle.legacy_maximum_error_bits,
        "candidate_gaps": gaps, "tree_metrics": _tree_metrics(tree),
        "overall_xi_nats": tree.xi_bits * math.log(2),
        "within_roi_xi_nats": sum(n.xi_bits for n in leaves) * math.log(2),
        "cross_roi_xi_nats": sum(n.syn_bits for n in internal) * math.log(2),
        "maximum_local_syn_nats": max(n.syn_bits for n in internal) * math.log(2),
        "maximum_local_syn_share_percent": 100 * max(n.syn_bits for n in internal) / tree.xi_bits,
        "elapsed_seconds": time.monotonic() - started,
    }


def preflight(ctx, output, code):
    value, metadata = simulate(1.3, ctx)
    with np.load(OLD_COV / "seed04_G1.30.npz") as a:
        maximum = float(np.max(np.abs(value - a["conditional_covariance"])))
    if maximum > 1e-10:
        raise RuntimeError(f"Anchor simulation no longer reproduces legacy covariance: max={maximum}")
    current = construct(value)
    old = json.loads(OLD_TREE.read_text())
    first = {n.indices: n for n in flatten_nodes(_node_from_record(current["tree"]))}
    second = {n.indices: n for n in flatten_nodes(_node_from_record(old["tree"]))}
    if set(first) != set(second):
        raise RuntimeError("Anchor tree differs from the original search on paired input")
    syn_error = max(abs(first[k].syn_bits - second[k].syn_bits) for k in first)
    if syn_error > TOL:
        raise RuntimeError(f"Anchor tree Syn differs from the legacy tree: max={syn_error} bits")
    proof = {"status": "passed", "code_sha256": code, "simulation": ctx[0], "input_sha256": ctx[1],
             "covariance_maximum_difference": maximum, "all_199_nodes_identical_members": True,
             "maximum_syn_difference_bits": syn_error, "simulation_timing": metadata,
             "tree_timing_seconds": current["elapsed_seconds"], "validation": current["validation"],
             "legacy_anchor_covariance": str(OLD_COV / "seed04_G1.30.npz"),
             "legacy_anchor_tree": str(OLD_TREE)}
    write_json(output / "preflight.json", proof)
    print("PREFLIGHT PASSED " + json.dumps(proof, separators=(",", ":")), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for name in ("covariance", "trees"):
        (output / name).mkdir(exist_ok=True)
    # Restrict BLAS oversubscription; preflight explicitly checks the resulting numerics.
    with threadpool_limits(limits=1):
        ctx = context()
        code = fingerprints()
        if args.preflight:
            preflight(ctx, output, code)
            return
        proof = json.loads((output / "preflight.json").read_text())
        if proof["status"] != "passed" or proof["code_sha256"] != code or proof["input_sha256"] != ctx[1]:
            raise ValueError("Run a matching --preflight before the full scan")
        gs = np.arange(171) / 100.
        config = {"version": 1, "simulation": ctx[0], "seed": SEED, "input_sha256": ctx[1],
                  "search": {"exact_max_size": 8, "strategy": "original spectral candidates",
                             "objective": "maximum child Xi sum / raw residual", "tolerance_bits": TOL},
                  "code_sha256": code, "scalar_xi_within_roi_leaves": True,
                  "estimator": "existing conditional Gaussian moment model; diagonal source covariance; no nonlinear TM",
                  "blas_threads": 1}
        write_json(output / "contract.json", {"config": config, "G": gs.tolist(), "n_conditions": 171,
                   "user_approval": "2026-10-08: 执行推荐方案", "threadpool_info": threadpool_info()})
        started = time.monotonic()
        records = []
        for g in gs:
            if fingerprints() != code:
                raise RuntimeError("A scientific source file changed during the scan; do not mix caches")
            begin = time.monotonic()
            conditional, path, cov_reused = covariance(float(g), ctx, output, code)
            expected_xi = CachedScalarLogdetXiOracle(conditional, [(i, i + 100) for i in range(100)]).xi(range(100))
            provenance = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                          "reused": cov_reused, "input_sha256": ctx[1]}
            treepath = output / "trees" / f"seed04_G{g:.3f}.json"
            if treepath.exists():
                record = json.loads(treepath.read_text())
                if record["config"] != config or record["covariance"]["sha256"] != provenance["sha256"] or record["G"] != float(g):
                    raise ValueError(f"Tree cache provenance mismatch: {treepath}")
                audit_tree(_node_from_record(record["tree"]), expected_xi)
                tree_reused = True
            else:
                record = construct(conditional)
                record.update(config=config, covariance=provenance, G=float(g), seed=SEED)
                write_json(treepath, record)
                tree_reused = False
            records.append({"G": float(g), "seed": SEED, "tree_path": str(treepath.relative_to(ROOT)),
                            "overall_xi_nats": record["overall_xi_nats"], "cross_roi_xi_nats": record["cross_roi_xi_nats"],
                            "within_roi_xi_nats": record["within_roi_xi_nats"], "tree_metrics": record["tree_metrics"],
                            "validation": record["validation"], "elapsed_seconds": record["elapsed_seconds"],
                            "maximum_local_syn_share_percent": record["maximum_local_syn_share_percent"]})
            print(f"done {len(records)}/171 G={g:.3f} Xi={record['overall_xi_nats']:.6f} nats "
                  f"depth={record['tree_metrics']['maximum_depth']} cov={'reused' if cov_reused else 'new'} "
                  f"tree={'reused' if tree_reused else 'new'} {time.monotonic()-begin:.1f}s", flush=True)
        write_json(output / "summary.json", {"status": "complete", "config": config, "G": gs.tolist(),
                   "labels": ctx[7], "network_names": ctx[8], "network_membership": ctx[9],
                   "records": records, "elapsed_seconds": time.monotonic() - started,
                   "preflight": proof})
        print(f"COMPLETE: 171 conditions, {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
