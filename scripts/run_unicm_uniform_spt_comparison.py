#!/usr/bin/env python3
"""Run one fixed UniCM SPT configuration for 3 checkpoints x 3 leads."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_unicm_11mode_xi_hierarchy_tree import (  # noqa: E402
    _node_record,
    _tree_metrics,
)
from scripts.plot_unicm_all_mode_target_pair_syn import extract_all_mode_target  # noqa: E402
from scripts.plot_unicm_phi_eid_greedy_decomposition import (  # noqa: E402
    compute_subset_ei_table_from_covariance,
    precompute_source_logdets,
)
from scripts.plot_unicm_spt_lead_comparison import (  # noqa: E402
    INK,
    SYN_COLOR,
    _draw_panel,
)
from scripts.spt import (  # noqa: E402
    NONNEGATIVE_TOLERANT,
    RAW_RESIDUAL,
    SIGNED,
    SPTConfig,
    SPTNonnegativityError,
    build_spt_from_ei_table,
    flatten_nodes,
    nontrivial_bipartitions,
)
from scripts.unicm_peid_syn_analysis import (  # noqa: E402
    HISTORY_LENGTH,
    MODE_NAMES,
    PREDICTION_LENGTH,
    load_unicm_model,
    make_full_history_mode_tensor,
    resolve_checkpoint_paths,
    sample_full_history_mode_inputs,
)


OUTPUT_DIR = ROOT / "results/unicm_xi_hierarchy_uniform_n16384"
FIGURE = ROOT / "fig/earth_unicm_spt_leads01_08_24_checkpoints.png"
CHECKPOINT_ROOT = ROOT / "data/UniCM-checkpoint/src/experiments"
LEADS = (1, 8, 24)
CHECKPOINTS = (1, 2, 3)


def _prediction_cache_path(args: argparse.Namespace, checkpoint: int) -> Path:
    return Path(args.output_dir) / "cache" / (
        f"checkpoint{checkpoint}_samples{args.n_samples}_sampling{args.sampling_seed}"
        f"_bound{args.intervention_bound:g}_fullhist12_start0_{args.device}.npz"
    )


def _predict_all_leads(model, histories: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    import torch

    rows: list[np.ndarray] = []
    started = time.monotonic()
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(histories), args.batch_size), start=1):
            end = min(len(histories), start + args.batch_size)
            predictor = torch.tensor(
                make_full_history_mode_tensor(histories[start:end]),
                device=args.device,
                dtype=torch.float32,
            )
            months = torch.arange(
                HISTORY_LENGTH + PREDICTION_LENGTH,
                device=args.device,
                dtype=torch.int64,
            ) % 12
            timestamps = months.unsqueeze(0).repeat(end - start, 1)
            output, _, _ = model.forward_sep(
                predictor,
                timestamps,
                model.encoder_mode,
                model.decoder_mode,
                model.linear_output_mode,
                model.predictor_emb_mode,
                model.predictand_emb_mode,
                1,
                [1, 1],
                train=False,
            )
            rows.append(output.squeeze(-1).squeeze(2).detach().cpu().numpy().astype(np.float32))
            if batch_index == 1 or end == len(histories) or batch_index % 4 == 0:
                print(
                    f"  predicted {end}/{len(histories)} samples in "
                    f"{time.monotonic() - started:.1f}s",
                    flush=True,
                )
    predictions = np.concatenate(rows, axis=0)
    expected = (args.n_samples, PREDICTION_LENGTH, len(MODE_NAMES))
    if predictions.shape != expected or not np.isfinite(predictions).all():
        raise ValueError(f"Malformed predictions: shape={predictions.shape}, expected={expected}")
    return predictions


def _load_or_predict(
    checkpoint: int,
    checkpoint_path: Path,
    histories: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    cache_path = _prediction_cache_path(args, checkpoint)
    if cache_path.exists():
        with np.load(cache_path) as payload:
            predictions = np.asarray(payload["all_mode_targets"], dtype=np.float32)
        expected = (args.n_samples, PREDICTION_LENGTH, len(MODE_NAMES))
        if predictions.shape != expected or not np.isfinite(predictions).all():
            raise ValueError(f"Invalid cache {cache_path}: shape={predictions.shape}")
        print(f"checkpoint {checkpoint}: reused {cache_path}", flush=True)
        return predictions

    print(f"checkpoint {checkpoint}: loading frozen model", flush=True)
    model = load_unicm_model(checkpoint_path, args.device)
    predictions = _predict_all_leads(model, histories, args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        all_mode_targets=predictions,
        metadata=json.dumps(
            {
                "checkpoint": checkpoint,
                "n_samples": args.n_samples,
                "sampling_seed": args.sampling_seed,
                "intervention_bound": args.intervention_bound,
                "start_month": 0,
                "device": args.device,
            },
            sort_keys=True,
        ),
    )
    print(f"checkpoint {checkpoint}: saved {cache_path}", flush=True)
    return predictions


def _tree_signature(tree) -> str:
    def topology(node):
        return {
            "sources": list(node.sources),
            "children": [topology(child) for child in node.children],
        }

    encoded = json.dumps(topology(tree), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _audit_tree_candidates(tree, ei_table, singleton_ei, tolerance: float) -> dict[str, object]:
    values: list[float] = []

    def xi(sources) -> float:
        key = tuple(name for name in MODE_NAMES if name in set(sources))
        if len(key) <= 1:
            return 0.0
        return float(ei_table[key] - sum(singleton_ei[name] for name in key))

    for node in flatten_nodes(tree):
        if not node.children:
            continue
        for left, right in nontrivial_bipartitions(node.sources):
            values.append(float(xi(node.sources) - xi(left) - xi(right)))
    array = np.asarray(values, dtype=float)
    return {
        "candidate_count": int(array.size),
        "minimum_candidate_syn_bits": float(array.min()),
        "tolerance_negative_candidate_count": int(np.count_nonzero((array < 0.0) & (array >= -tolerance))),
        "significant_violation_count": int(np.count_nonzero(array < -tolerance)),
        "syn_tolerance_bits": float(tolerance),
    }


def _evaluate_condition(
    histories_flat: np.ndarray,
    subset_columns,
    source_logdets,
    target: np.ndarray,
    args: argparse.Namespace,
) -> tuple[object, dict[str, object]]:
    ei_table = compute_subset_ei_table_from_covariance(
        histories_flat,
        target,
        subset_columns,
        source_logdets,
        jitter=args.jitter,
    )
    mode_names = tuple(MODE_NAMES)
    singleton_ei = {name: float(ei_table[(name,)]) for name in mode_names}
    signed = build_spt_from_ei_table(
        mode_names,
        ei_table,
        singleton_ei=singleton_ei,
        config=SPTConfig(
            policy=SIGNED,
            split_objective=RAW_RESIDUAL,
            syn_tolerance=args.syn_tolerance,
            eps=args.eps,
            complete_to_singletons=True,
        ),
    )
    candidate_audit = _audit_tree_candidates(
        signed.root,
        ei_table,
        singleton_ei,
        args.syn_tolerance,
    )
    canonical_status = "valid_nonnegative_spt"
    canonical_signature = None
    algorithm_match = False
    try:
        canonical = build_spt_from_ei_table(
            mode_names,
            ei_table,
            singleton_ei=singleton_ei,
            config=SPTConfig(
                policy=NONNEGATIVE_TOLERANT,
                split_objective=RAW_RESIDUAL,
                syn_tolerance=args.syn_tolerance,
                eps=args.eps,
                complete_to_singletons=True,
            ),
        )
        canonical_signature = _tree_signature(canonical.root)
        algorithm_match = canonical_signature == _tree_signature(signed.root)
        if not algorithm_match:
            canonical_status = "algorithm_output_mismatch"
    except SPTNonnegativityError as exc:
        canonical_status = f"invalid_nonnegative_spt: {exc}"

    valid = (
        int(candidate_audit["significant_violation_count"]) == 0
        and canonical_status == "valid_nonnegative_spt"
        and algorithm_match
    )
    diagnostics = {
        "status": "valid_nonnegative_spt" if valid else "invalid_nonnegative_spt",
        "candidate_audit": candidate_audit,
        "canonical_builder_status": canonical_status,
        "signed_diagnostic_topology_sha256": _tree_signature(signed.root),
        "canonical_topology_sha256": canonical_signature,
        "same_topology_from_canonical_and_signed_paths": algorithm_match,
        "canonical_algorithm": {
            "builder": "scripts.spt.build_spt_from_ei_table",
            "policy": NONNEGATIVE_TOLERANT,
            "split_objective": RAW_RESIDUAL,
            "exact_bipartitions": True,
            "complete_to_singletons": True,
            "eps_bits": args.eps,
            "syn_tolerance_bits": args.syn_tolerance,
        },
    }
    return signed.root, diagnostics


def _render(trees: dict[tuple[int, int], object], validity, output: Path, tolerance: float) -> None:
    positives = [
        float(node.syn_value)
        for tree in trees.values()
        for node in flatten_nodes(tree)
        if node.children and node.syn_value >= 0.0
    ]
    norm = Normalize(vmin=0.0, vmax=max(positives) if positives else 1.0)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("syn", ["#F1F7F5", SYN_COLOR])
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(3, 3, figsize=(14.2, 12.8), layout="constrained")
    for row, lead in enumerate(LEADS):
        for column, checkpoint in enumerate(CHECKPOINTS):
            _draw_panel(
                axes[row, column],
                trees[(lead, checkpoint)],
                lead=lead,
                seed=checkpoint,
                norm=norm,
                cmap=cmap,
                invalid=not validity[(lead, checkpoint)],
            )
    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        location="right",
        shrink=0.63,
        aspect=34,
        pad=0.012,
    )
    colorbar.set_label("Local hierarchy Syn (bits)", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=5.7, width=0.5, length=2.2)
    figure.text(
        0.5,
        -0.008,
        f"Same n=16,384 interventions, source histories, joint target definition, estimator and canonical SPT; "
        f"only checkpoint and forecast lead change. Nonnegative tolerance: {tolerance:g} bits.",
        ha="center",
        va="top",
        fontsize=6.3,
        color=INK,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    histories = sample_full_history_mode_inputs(
        n_samples=args.n_samples,
        intervention_bound=args.intervention_bound,
        seed=args.sampling_seed,
    )
    history_flat, subset_columns, source_logdets = precompute_source_logdets(
        histories,
        jitter=args.jitter,
    )
    checkpoint_paths = resolve_checkpoint_paths(Path(args.checkpoint_root), CHECKPOINTS)
    predictions = {
        checkpoint: _load_or_predict(checkpoint, checkpoint_paths[checkpoint], histories, args)
        for checkpoint in CHECKPOINTS
    }
    trees = {}
    validity = {}
    rows = []
    for lead in LEADS:
        for checkpoint in CHECKPOINTS:
            condition_started = time.monotonic()
            tree, diagnostics = _evaluate_condition(
                history_flat,
                subset_columns,
                source_logdets,
                extract_all_mode_target(predictions[checkpoint], lead=lead),
                args,
            )
            trees[(lead, checkpoint)] = tree
            validity[(lead, checkpoint)] = diagnostics["status"] == "valid_nonnegative_spt"
            row = {
                "lead": lead,
                "checkpoint": checkpoint,
                "xi_bits": float(tree.xi_value),
                "metrics": _tree_metrics(tree, args.syn_tolerance),
                **diagnostics,
                "tree": _node_record(tree),
            }
            rows.append(row)
            print(
                f"lead={lead} checkpoint={checkpoint} status={row['status']} "
                f"Xi={row['xi_bits']:.6f} min_candidate="
                f"{row['candidate_audit']['minimum_candidate_syn_bits']:.6f} "
                f"violations={row['candidate_audit']['significant_violation_count']} "
                f"analysis={time.monotonic() - condition_started:.1f}s",
                flush=True,
            )
            partial = {
                "status": "running",
                "completed_conditions": len(rows),
                "total_conditions": 9,
                "results": rows,
            }
            (output_dir / "partial.json").write_text(
                json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    _render(trees, validity, Path(args.figure), args.syn_tolerance)
    all_valid = all(validity.values())
    payload = {
        "experiment": "UniCM fixed-parameter 3-checkpoint x 3-lead exact SPT comparison",
        "status": "complete_valid" if all_valid else "complete_with_nonnegativity_failures",
        "all_nine_conditions_valid_nonnegative_spt": all_valid,
        "checkpoints": list(CHECKPOINTS),
        "leads": list(LEADS),
        "n_samples": args.n_samples,
        "sampling_seed": args.sampling_seed,
        "intervention_bound": args.intervention_bound,
        "start_month": 0,
        "estimator": "affine degree-1 TM / Gaussian log-det equivalent",
        "jitter": args.jitter,
        "syn_tolerance_bits": args.syn_tolerance,
        "eps_bits": args.eps,
        "algorithm_source": "scripts.spt.build_spt_from_ei_table",
        "figure": str(args.figure),
        "elapsed_seconds": time.monotonic() - started,
        "results": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "partial.json").write_text(
        json.dumps(
            {
                "status": payload["status"],
                "completed_conditions": len(rows),
                "total_conditions": len(LEADS) * len(CHECKPOINTS),
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(Path(args.figure), flush=True)
    if not all_valid:
        failed = [
            f"lead {row['lead']} checkpoint {row['checkpoint']}"
            for row in rows
            if row["status"] != "valid_nonnegative_spt"
        ]
        raise RuntimeError("Nonnegative SPT validation failed for: " + ", ".join(failed))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    parser.add_argument("--n-samples", type=int, default=16384)
    parser.add_argument("--sampling-seed", type=int, default=20260901)
    parser.add_argument("--intervention-bound", type=float, default=4.0)
    parser.add_argument("--jitter", type=float, default=1.0e-6)
    parser.add_argument("--syn-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--eps", type=float, default=1.0e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
