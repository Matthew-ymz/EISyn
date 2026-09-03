#!/usr/bin/env python3
"""Compare the UniCM lead-8 Xi hierarchy with a joint 24-month target hierarchy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_unicm_11mode_xi_hierarchy_tree import (  # noqa: E402
    CORE_MODES,
    _expand_pair_leaves,
    _node_record,
    _tree_metrics,
    _validate_syn,
    render_trees,
)
from scripts.phi_hierarchy import (  # noqa: E402
    NONNEGATIVE_TOLERANT,
    RAW_RESIDUAL,
    greedy_phi_tree,
)
from scripts.plot_unicm_phi_eid_greedy_decomposition import (  # noqa: E402
    compute_subset_ei_table_from_covariance,
    precompute_source_logdets,
)
from scripts.unicm_peid_syn_analysis import (  # noqa: E402
    MODE_NAMES,
    load_full_history_prediction_cache,
    overall_prediction_cache_path,
    sample_full_history_mode_inputs,
)


DEFAULT_CACHE_DIR = ROOT / "results/unicm_overall_ei_cpu_bound4_n8192/cache"
DEFAULT_OUTPUT_DIR = ROOT / "results/unicm_xi_hierarchy_joint24"
DEFAULT_FIGURE = ROOT / "fig/earth_unicm_11mode_xi_hierarchy_joint24.png"
DEFAULT_LEAD8_SUMMARY = ROOT / "results/unicm_xi_hierarchy_tree/summary.json"


def _cache_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        n_samples=int(args.cache_samples),
        sampling_seed=int(args.sampling_seed),
        intervention_bound=float(args.intervention_bound),
        start_month=int(args.start_month),
        device="cpu",
    )


def _joint_24_target(predictions: np.ndarray, sample_limit: int) -> np.ndarray:
    array = np.asarray(predictions[:sample_limit], dtype=float)
    if array.ndim != 3 or tuple(array.shape[1:]) != (24, len(MODE_NAMES)):
        raise ValueError(f"Expected predictions shaped (n, 24, 11), got {array.shape}.")
    return array.reshape(array.shape[0], -1)


def _condition_number(samples: np.ndarray, jitter: float) -> float:
    covariance = np.atleast_2d(np.cov(np.asarray(samples, dtype=float), rowvar=False, bias=False))
    scale = float(np.trace(covariance) / covariance.shape[0])
    covariance = covariance + float(jitter) * max(scale, 1.0) * np.eye(covariance.shape[0])
    return float(np.linalg.cond(covariance))


def _core_record(tree) -> dict[str, object]:
    stack = [tree]
    while stack:
        node = stack.pop()
        if frozenset(node.sources) == CORE_MODES:
            return {
                "present": True,
                "phi_bits": float(node.phi_value),
                "share_of_total_xi": float(node.phi_value / tree.phi_value),
                "local_syn_bits": float(node.residual),
                "depth": int(node.depth),
            }
        stack.extend(node.children)
    return {
        "present": False,
        "phi_bits": 0.0,
        "share_of_total_xi": 0.0,
        "local_syn_bits": 0.0,
        "depth": None,
    }


def _render_joint(trees, seeds, output: Path, dpi: int, tolerance: float) -> None:
    figure = plt.figure(figsize=(18.0, 8.8), constrained_layout=True)
    render_trees(
        trees,
        seeds,
        output,
        lead=24,
        split_objective=RAW_RESIDUAL,
        dpi=dpi,
        syn_tolerance=tolerance,
        canvas=figure,
        show_colorbar=True,
    )
    figure.text(
        0.5,
        -0.015,
        "Joint target: all 11 UniCM modes across leads 1--24 (264 dimensions)  |  "
        "Sources: the same 11 full 12-month histories  |  Nodes show local Syn.",
        ha="center",
        va="top",
        fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    if not 3 <= int(args.sample_limit) <= int(args.cache_samples):
        raise ValueError("sample-limit must be between 3 and cache-samples.")
    mode_names = tuple(MODE_NAMES)
    histories = sample_full_history_mode_inputs(
        n_samples=int(args.cache_samples),
        intervention_bound=float(args.intervention_bound),
        seed=int(args.sampling_seed),
    )[: int(args.sample_limit)]
    history_flat, subset_columns, source_logdets = precompute_source_logdets(
        histories,
        jitter=float(args.jitter),
    )
    cache_args = _cache_args(args)
    trees = []
    checkpoint_rows = []
    for seed in args.seeds:
        cache_path = overall_prediction_cache_path(Path(args.cache_dir), seed=int(seed), args=cache_args)
        predictions = load_full_history_prediction_cache(cache_path, n_samples=int(args.cache_samples))
        target = _joint_24_target(predictions, int(args.sample_limit))
        ei_table = compute_subset_ei_table_from_covariance(
            history_flat,
            target,
            subset_columns,
            source_logdets,
            jitter=float(args.jitter),
        )
        singleton_ei = {name: float(ei_table[(name,)]) for name in mode_names}
        tree = greedy_phi_tree(
            mode_names,
            ei_table,
            policy=NONNEGATIVE_TOLERANT,
            eps=float(args.eps),
            split_tolerance=float(args.split_tolerance),
            singleton_ei=singleton_ei,
            split_objective=RAW_RESIDUAL,
        )
        tree = _expand_pair_leaves(tree)
        validation = _validate_syn([tree], float(args.split_tolerance))
        metrics = _tree_metrics(tree, float(args.split_tolerance))
        if abs(float(metrics["closure_error_bits"])) > 1.0e-8:
            raise RuntimeError(f"Checkpoint {seed} hierarchy closure failed.")
        checkpoint_rows.append(
            {
                "seed": int(seed),
                "xi_bits": float(tree.phi_value),
                "whole_ei_bits": float(ei_table[mode_names]),
                "singleton_ei_sum_bits": float(sum(singleton_ei.values())),
                "target_covariance_condition_number": _condition_number(target, float(args.jitter)),
                "metrics": metrics,
                "syn_validation": validation,
                "enso_iod_five_mode_core": _core_record(tree),
                "tree": _node_record(tree),
            }
        )
        trees.append(tree)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = Path(args.figure)
    _render_joint(trees, args.seeds, figure_path, int(args.dpi), float(args.split_tolerance))
    lead8 = json.loads(Path(args.lead8_summary).read_text(encoding="utf-8"))
    lead8_by_seed = {int(row["seed"]): row for row in lead8["checkpoints"]}
    comparisons = []
    for row in checkpoint_rows:
        baseline = lead8_by_seed[int(row["seed"])]
        comparisons.append(
            {
                "seed": int(row["seed"]),
                "joint24_minus_lead8_xi_bits": float(row["xi_bits"] - baseline["xi_bits"]),
                "lead8_spine_fraction": float(baseline["metrics"]["dominant_spine_fraction"]),
                "joint24_spine_fraction": float(row["metrics"]["dominant_spine_fraction"]),
                "lead8_imbalance": float(baseline["metrics"]["normalized_colless_imbalance"]),
                "joint24_imbalance": float(row["metrics"]["normalized_colless_imbalance"]),
                "joint24_core_present": bool(row["enso_iod_five_mode_core"]["present"]),
            }
        )
    payload = {
        "experiment": "UniCM exact 11-mode Xi hierarchy for the joint 24-month trajectory target",
        "comparison_question": "What changes when only the target changes from lead 8 to the joint lead-1--24 trajectory?",
        "source_definition": "each of 11 sources is one mode's full 12-month history",
        "target_definition": "all 11 future UniCM modes over leads 1--24, flattened to 264 dimensions",
        "seeds": [int(seed) for seed in args.seeds],
        "sample_limit": int(args.sample_limit),
        "cache_samples": int(args.cache_samples),
        "sampling_seed": int(args.sampling_seed),
        "intervention_bound": float(args.intervention_bound),
        "estimator": "affine degree-1 TM / Gaussian log-det equivalent",
        "jitter": float(args.jitter),
        "eps_bits": float(args.eps),
        "split_tolerance_bits": float(args.split_tolerance),
        "figure": str(figure_path),
        "checkpoints": checkpoint_rows,
        "paired_lead8_comparison": comparisons,
        "limitation": "Target dimensionality and temporal scope change together by definition; temporal redundancy is part of the joint target.",
    }
    summary_path = output_dir / ("summary.json" if int(args.sample_limit) == int(args.cache_samples) else "smoke_summary.json")
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--lead8-summary", type=Path, default=DEFAULT_LEAD8_SUMMARY)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--cache-samples", type=int, default=8192)
    parser.add_argument("--sample-limit", type=int, default=8192)
    parser.add_argument("--sampling-seed", type=int, default=20260619)
    parser.add_argument("--intervention-bound", type=float, default=4.0)
    parser.add_argument("--start-month", type=int, default=0)
    parser.add_argument("--jitter", type=float, default=1.0e-6)
    parser.add_argument("--eps", type=float, default=1.0e-5)
    parser.add_argument("--split-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(
        "[done] "
        + ", ".join(
            f"seed={row['seed']} Xi={row['xi_bits']:.6f} "
            f"spine={row['metrics']['dominant_spine_fraction']:.3f}"
            for row in payload["checkpoints"]
        )
    )


if __name__ == "__main__":
    main()
