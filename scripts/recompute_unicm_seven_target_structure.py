#!/usr/bin/env python3
"""Recompute candidate UniCM structures with eleven sources and seven targets.

The joint7 game reads a seven-dimensional future at each lead.  scalar7 instead
produces seven separate games, never their sum.  Both retain the eleven complete
12-month source histories and use the existing independent uniform interventions.
No frozen-model inference or training is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_unicm_11mode_shapley import exact_shapley
from scripts.compute_unicm_order_syn import (
    ESTIMATOR_VERSION,
    fit_independent_source_affine_channel,
    independent_source_ei_table_from_channel,
)
from scripts.spt import (
    NONNEGATIVE_TOLERANT,
    RAW_RESIDUAL,
    SPTConfig,
    build_spt_from_ei_table,
    flatten_nodes,
)
from scripts.unicm_peid_syn_analysis import MODE_NAMES, sample_full_history_mode_inputs

SOURCES = tuple(MODE_NAMES)
TARGETS = SOURCES[:7]
TARGET_DISPLAY = ("ENSO", *TARGETS[1:])
PAIRS = tuple(itertools.combinations(range(len(SOURCES)), 2))
TOLERANCE_BITS = 1e-8
CLOSURE_BITS = 1e-10


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_prediction_cache(path: Path, checkpoint: int, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    """Check archived generation settings before consuming frozen predictions."""
    expected = {
        "checkpoint": int(checkpoint), "device": "cpu",
        "n_samples": int(args.n_samples), "sampling_seed": int(args.sampling_seed),
        "intervention_bound": float(args.intervention_bound),
        "start_month": int(args.start_month),
    }
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive:
            raise ValueError(f"Prediction cache has no generation metadata: {path}")
        metadata = json.loads(str(archive["metadata"].item()))
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid prediction cache generation metadata: {path}")
        for key, value in expected.items():
            if key not in metadata or metadata[key] != value:
                raise ValueError(f"Prediction cache metadata mismatch: {path}; "
                                 f"{key}: expected={value!r}, actual={metadata.get(key)!r}")
        predictions = archive["all_mode_targets"].astype(float)
    if predictions.shape != (args.n_samples, 24, len(SOURCES)) or not np.isfinite(predictions).all():
        raise ValueError(f"Invalid intervention prediction cache {path}")
    return predictions, metadata


def audit_nonnegative(values: np.ndarray, context: str) -> dict:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite {context}")
    significant = array < -TOLERANCE_BITS
    audit = {
        "minimum_bits": float(array.min()),
        "threshold_bits": -TOLERANCE_BITS,
        "affected_count": int(significant.sum()),
        "tolerance_negative_count": int(((array < 0) & ~significant).sum()),
        "count": int(array.size),
        "projection": "none; tolerance-range negatives retained in raw arrays",
    }
    if significant.any():
        raise ValueError(f"{context} nonnegativity violation: minimum={array.min():.12g} bit, "
                         f"threshold={-TOLERANCE_BITS:.12g} bit, "
                         f"affected_count={significant.sum()}")
    return audit


def mask_table(table: dict) -> np.ndarray:
    values = np.zeros(1 << len(SOURCES), dtype=float)
    for names, value in table.items():
        mask = sum(1 << SOURCES.index(name) for name in names)
        values[mask] = value
    return values


def tree_record(node) -> dict:
    return {
        "sources": list(node.sources), "order": node.order,
        "phi_value_bits": float(node.xi_value),
        "residual_syn_bits": float(node.syn_value),
        "action": node.action, "atom_kind": node.atom_kind, "depth": node.depth,
        "children": [tree_record(child) for child in node.children],
    }


def analyze_table(table: dict, *, checkpoint: int, lead: int,
                  target: str) -> tuple[np.ndarray, np.ndarray, dict, dict, np.ndarray, np.ndarray]:
    ei = mask_table(table)
    membership = ((np.arange(len(ei))[:, None] >> np.arange(len(SOURCES))) & 1)
    xi = ei - membership @ ei[1 << np.arange(len(SOURCES))]
    xi_audit = audit_nonnegative(xi[membership.sum(axis=1) >= 2], "coalition Xi")
    attribution = exact_shapley(dict(enumerate(xi)), len(SOURCES))
    shapley_audit = audit_nonnegative(attribution, "Xi Shapley attribution")
    closure = float(attribution.sum() - xi[-1])
    if abs(closure) > CLOSURE_BITS or xi[-1] <= 0:
        raise ValueError(f"Invalid Shapley closure/root Xi: {closure=}, root={xi[-1]}")
    result = build_spt_from_ei_table(
        SOURCES, table,
        config=SPTConfig(policy=NONNEGATIVE_TOLERANT, split_objective=RAW_RESIDUAL,
                         syn_tolerance=TOLERANCE_BITS, eps=TOLERANCE_BITS,
                         complete_to_singletons=True),
    )
    nodes = [node for node in flatten_nodes(result.root) if node.children]
    selected = np.asarray([node.syn_value for node in nodes])
    selected_audit = audit_nonnegative(selected, "selected SPT Syn")
    if abs(result.closure_error) > CLOSURE_BITS:
        raise ValueError(f"SPT closure error {result.closure_error:.12g} bit")
    # Preserve raw signed values within tolerance. No silent clipping or
    # accumulation of tolerance corrections can break the tree identity.
    mass = np.zeros(len(SOURCES) - 1)
    counts = np.zeros(len(SOURCES) - 1, dtype=int)
    for node in nodes:
        mass[node.order - 2] += node.syn_value
        counts[node.order - 2] += 1
    if abs(mass.sum() - xi[-1]) > CLOSURE_BITS:
        raise ValueError("SPT order masses do not close to the coalition Xi")
    record = {
        "seed": checkpoint, "checkpoint": checkpoint, "lead": lead, "target": target,
        "grand_coalition_ei_bits": float(ei[-1]),
        "singleton_ei_sum_bits": float(ei[1 << np.arange(len(SOURCES))].sum()),
        "grand_coalition_interaction_bits": float(xi[-1]),
        "shapley_bits": dict(zip(SOURCES, attribution.tolist())),
        "shapley_percent": dict(zip(SOURCES, (100 * attribution / xi[-1]).tolist())),
        "closure_error_bits": closure,
        "percentage_sum": float(100 * attribution.sum() / xi[-1]),
    }
    tree = {
        "checkpoint": checkpoint, "lead": lead, "target": target,
        "status": "valid_nonnegative_spt", "tree": tree_record(result.root),
        "xi_bits": float(result.root.xi_value),
        "order_mass_bits": dict(zip(map(str, range(2, 12)), mass.tolist())),
        "node_counts": dict(zip(map(str, range(2, 12)), counts.tolist())),
        "closure_error_bits": result.closure_error,
        "audits": {"xi": xi_audit, "shapley": shapley_audit,
                   "selected_syn": selected_audit, "split_candidates": asdict(result.audit)},
    }
    return ei, xi, record, tree, mass, counts


def lead_summary(records: list[dict], leads: list[int], targets: list[str]) -> list[dict]:
    rows = []
    for target in targets:
        for lead in leads:
            selected = [row for row in records if row["target"] == target and row["lead"] == lead]
            totals = np.asarray([row["grand_coalition_interaction_bits"] for row in selected])
            row = {"lead": lead, "target": target,
                   "interaction_mean_bits": float(totals.mean()),
                   "interaction_std_bits": float(totals.std(ddof=1)) if len(totals) > 1 else None}
            for field in ("shapley_bits", "shapley_percent"):
                matrix = np.asarray([[item[field][name] for name in SOURCES] for item in selected])
                row[field + "_mean"] = dict(zip(SOURCES, matrix.mean(axis=0).tolist()))
                row[field + "_std"] = (dict(zip(SOURCES, matrix.std(axis=0, ddof=1).tolist()))
                                         if len(matrix) > 1 else dict.fromkeys(SOURCES, None))
            rows.append(row)
    return rows


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    histories = sample_full_history_mode_inputs(n_samples=args.n_samples,
        intervention_bound=args.intervention_bound, seed=args.sampling_seed).astype(float)
    paths = [args.cache_dir / (f"checkpoint{seed}_samples{args.n_samples}_sampling"
              f"{args.sampling_seed}_bound{args.intervention_bound:g}_fullhist12_start"
              f"{args.start_month}_cpu.npz") for seed in args.checkpoints]
    metadata = {
        "estimator": ESTIMATOR_VERSION, "approximation": "affine degree-1 TM / Gaussian readout",
        "source_names": list(SOURCES), "source_display_names": ["ENSO", *SOURCES[1:]],
        "target_names": list(TARGETS), "target_display_names": list(TARGET_DISPLAY),
        "history_months": 12, "n_samples": args.n_samples, "sampling_seed": args.sampling_seed,
        "intervention_bound": args.intervention_bound, "start_month": args.start_month,
        "source_prior": f"independent uniform; covariance=({args.intervention_bound**2 / 3:g}) I_132",
        "covariance_ridge": args.covariance_ridge,
        "regularization": "once on residual covariance; ridge=jitter*max(trace(residual)/target_dim,1)",
        "syn_tolerance_bits": TOLERANCE_BITS, "closure_tolerance_bits": CLOSURE_BITS,
        "tolerance_negative_rule": "retained raw and counted; no projection",
        "checkpoints": args.checkpoints, "leads": args.leads,
        "manuscript": {"parent_key": "P6UJCVG8", "title": "Emergent hierarchical organization of causal interactions in complex systems",
            "main_attachment_key": "DXGC7JEA", "appendix_attachment_key": "MWIWKSVG",
            "version": "No explicit manuscript version/date found; current attached draft checked 2026-10-23",
            "relevant_sections": "S1.2 continuous TM; S1.3 Eqs S16-S18; S5 SPT; S12.1 Earth experiment",
            "intentional_revision": "11 sources retained; joint/scalar future target restricted to seven primary climate modes"},
        "prediction_caches": {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): sha256(path) for path in paths},
        "prediction_cache_metadata": {},
        "fit_scope": "refit affine readout from intervention histories to selected seven-dimensional output at every checkpoint/lead",
        "frozen_inference_or_training_performed": False,
    }
    configurations = (("joint7", "scalar7") if args.target_mode == "both" else
                      (() if args.target_mode == "pairs" else (args.target_mode,)))
    collected = {key: {"records": [], "trees": [], "ei": [], "xi": [], "mass": [], "counts": []}
                 for key in configurations}
    scalar_pairs = np.empty((len(args.checkpoints), len(args.leads), len(PAIRS), len(TARGETS)))
    pair_audits = []
    for checkpoint_index, (checkpoint, path) in enumerate(zip(args.checkpoints, paths, strict=True)):
        predictions, cache_metadata = verified_prediction_cache(path, checkpoint, args)
        metadata["prediction_cache_metadata"][str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)] = cache_metadata
        for lead_index, lead in enumerate(args.leads):
            weights, residual, fit_audit = fit_independent_source_affine_channel(histories, predictions[:, lead - 1, :7])
            scalar_tables = []
            # Scalar residual regularization is done separately for each target,
            # matching a scalar affine-TM fit rather than slicing an already
            # regularized joint covariance matrix.
            for target_index in range(len(TARGETS)):
                table, audit = independent_source_ei_table_from_channel(
                    weights[:, target_index:target_index + 1], residual[target_index:target_index + 1, target_index:target_index + 1],
                    intervention_bound=args.intervention_bound, jitter=args.covariance_ridge)
                scalar_tables.append((table, audit))
                scalar_pairs[checkpoint_index, lead_index, :, target_index] = [
                    table[(SOURCES[a], SOURCES[b])] - table[(SOURCES[a],)] - table[(SOURCES[b],)]
                    for a, b in PAIRS]
            pair_audit = audit_nonnegative(scalar_pairs[checkpoint_index, lead_index], "scalar-target pair Syn")
            pair_audits.append({"checkpoint": checkpoint, "lead": lead, **pair_audit})
            for configuration in configurations:
                data = collected[configuration]
                if configuration == "joint7":
                    table, audit = independent_source_ei_table_from_channel(weights, residual,
                        intervention_bound=args.intervention_bound, jitter=args.covariance_ridge)
                    games = [("joint7", table, audit)]
                else:
                    games = [(name, table, audit) for name, (table, audit) in zip(TARGETS, scalar_tables, strict=True)]
                lead_arrays = {key: [] for key in ("ei", "xi", "mass", "counts")}
                for name, table, audit in games:
                    ei, xi, record, tree, mass, counts = analyze_table(table, checkpoint=checkpoint, lead=lead, target=name)
                    tree["estimator_audit"] = {**fit_audit, **audit}
                    data["records"].append(record)
                    data["trees"].append(tree)
                    for key, array in zip(lead_arrays, (ei, xi, mass, counts), strict=True):
                        lead_arrays[key].append(array)
                for key in lead_arrays:
                    data[key].append(lead_arrays[key][0] if configuration == "joint7" else lead_arrays[key])
            print(f"checkpoint={checkpoint} lead={lead}: seven-target structure complete", flush=True)
    shape = (len(args.checkpoints), len(args.leads))
    for configuration, data in collected.items():
        output = args.output_dir / configuration
        output.mkdir(parents=True, exist_ok=True)
        info = {**metadata, "target_configuration": configuration,
            "target_definition": ("joint seven-dimensional future at each lead" if configuration == "joint7" else "seven separate scalar future games; not summed"),
            "source_coalition_count": 1 << len(SOURCES), "spt_split_objective": RAW_RESIDUAL,
            "spt_search": "all unordered bipartitions at every node; recurse to singletons",
            "aggregation": "order sums/attribution normalized within each tree/game, then equal checkpoint mean"}
        arrays = {}
        for key in ("ei", "xi", "mass", "counts"):
            array = np.asarray(data[key])
            arrays[key] = array.reshape(shape + array.shape[1:])
        np.savez_compressed(output / "ei_tables.npz", ei_bits=arrays["ei"], xi_bits=arrays["xi"],
            masks=np.arange(1 << len(SOURCES)), checkpoints=args.checkpoints, leads=args.leads,
            source_names=SOURCES, target_names=TARGETS, metadata=json.dumps(info))
        xi = arrays["xi"][..., -1]
        mass = arrays["mass"]
        shares = mass / xi[..., None]
        mean_mass = mass.mean(axis=0).T if configuration == "joint7" else mass.mean(axis=0).transpose(1, 2, 0)
        mean_share = shares.mean(axis=0).T if configuration == "joint7" else shares.mean(axis=0).transpose(1, 2, 0)
        np.savez_compressed(output / "order_mass.npz", mass_bits=mass, share=shares,
            node_counts=arrays["counts"], xi_bits=xi, mean_mass_bits=mean_mass,
            mean_share=mean_share, checkpoints=args.checkpoints, leads=args.leads,
            orders=np.arange(2, 12), metadata=json.dumps(info))
        if configuration == 'joint7':
            # Every source pair reads the SAME seven-dimensional future. A
            # geographic target-specific arrow would describe a different game.
            pair_masks = np.asarray([(1 << a) | (1 << b) for a, b in PAIRS])
            joint_pair = arrays['xi'][..., pair_masks]
            pair_audit = audit_nonnegative(joint_pair, 'joint-seven-target pair Syn')
            np.savez_compressed(output / 'pair_syn.npz', values_bits=joint_pair,
                pairs=PAIRS, leads=args.leads, checkpoints=args.checkpoints,
                target_names=TARGETS, metadata=json.dumps({**info, 'pair_audit': pair_audit}))
            panels = []
            for lead in (1,8,24):
                if lead not in args.leads:
                    continue
                li = args.leads.index(lead)
                ranked = [dict(lead=lead, a=a, b=b, target='joint7',
                    mean_bits=float(joint_pair[:,li,pi].mean()),
                    std_bits=float(joint_pair[:,li,pi].std(ddof=1)) if len(args.checkpoints)>1 else None)
                    for pi,(a,b) in enumerate(PAIRS)]
                panels.append(sorted(ranked,key=lambda row:-row['mean_bits'])[:10])
            write_json(output / 'displayed_edges.json', dict(
                selection='top 10 of 55 source pairs; same seven-dimensional joint future target',
                eligible_source_pairs_per_lead=55, panels=panels, target_names=list(TARGETS),
                audit=pair_audit))
        write_json(output / "shapley_summary.json", {"status": "complete", "method": info,
            "records": data["records"], "lead_summary": lead_summary(data["records"], args.leads,
                ["joint7"] if configuration == "joint7" else list(TARGETS))})
        write_json(output / "hierarchy_trees.json", {"status": "complete", "method": info,
            "syn_tolerance_bits": TOLERANCE_BITS, "all_conditions_valid_nonnegative_spt": True,
            "results": data["trees"]})
    pair_metadata = {**metadata, "target_configuration": "scalar7",
        "target_definition": "each of seven scalar future modes",
        "audit": audit_nonnegative(scalar_pairs, "all pair Syn"),
        "selection": "top 10 per lead; target distinct from each of two sources; rank by equal-checkpoint mean"}
    np.savez_compressed(args.output_dir / "scalar_pair_syn_all_leads.npz", values_bits=scalar_pairs,
        pairs=PAIRS, leads=args.leads, checkpoints=args.checkpoints, target_names=TARGETS,
        metadata=json.dumps(pair_metadata))
    displayed_leads = [lead for lead in (1, 8, 24) if lead in args.leads]
    selected = []
    mean_pairs = scalar_pairs.mean(axis=0)
    for lead in displayed_leads:
        lead_index = args.leads.index(lead)
        rows = [{"lead": lead, "a": a, "b": b, "t": t,
                 "mean_bits": float(mean_pairs[lead_index, pair_index, t]),
                 "std_bits": (float(scalar_pairs[:, lead_index, pair_index, t].std(ddof=1))
                              if len(args.checkpoints) > 1 else None)}
                for pair_index, (a, b) in enumerate(PAIRS) for t in range(len(TARGETS)) if t not in (a, b)]
        selected.append(sorted(rows, key=lambda row: -row["mean_bits"])[:10])
    np.savez_compressed(args.output_dir / "pair_syn.npz",
        values_bits=scalar_pairs[:, [args.leads.index(lead) for lead in displayed_leads]],
        pairs=PAIRS, leads=displayed_leads, checkpoints=args.checkpoints, target_names=TARGETS,
        metadata=json.dumps(pair_metadata))
    write_json(args.output_dir / "displayed_edges.json", {"selection": pair_metadata["selection"],
        "eligible_triples_per_lead": 315, "target_names": list(TARGETS), "panels": selected})
    write_json(args.output_dir / "summary.json", {"status": "complete", "method": metadata,
        "target_configurations": list(configurations), "pair_audit": pair_metadata["audit"],
        "pair_fit_audits": pair_audits, "elapsed_seconds": time.perf_counter() - started})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "results/unicm_xi_hierarchy_uniform_n16384/cache")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/unicm_seven_target_candidate/structure")
    p.add_argument("--target-mode", choices=("joint7", "scalar7", "both", "pairs"), default="joint7")
    p.add_argument("--n-samples", type=int, default=16384)
    p.add_argument("--sampling-seed", type=int, default=20260901)
    p.add_argument("--intervention-bound", type=float, default=4.)
    p.add_argument("--start-month", type=int, default=0)
    p.add_argument("--covariance-ridge", type=float, default=1e-6)
    p.add_argument("--checkpoints", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--leads", nargs="+", type=int, default=list(range(1, 25)))
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    if not arguments.leads or len(set(arguments.leads)) != len(arguments.leads) or any(not 1 <= lead <= 24 for lead in arguments.leads):
        raise ValueError("Leads must be unique values in 1..24")
    if len(set(arguments.checkpoints)) != len(arguments.checkpoints):
        raise ValueError("Checkpoint seeds must be unique")
    run(arguments)
