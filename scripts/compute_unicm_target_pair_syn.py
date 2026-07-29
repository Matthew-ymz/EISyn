#!/usr/bin/env python3
"""Compute target-resolved source-pair Syn from existing UniCM prediction caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.unicm_peid_syn_analysis import (  # noqa: E402
    MODE_NAMES,
    load_full_history_prediction_cache,
    overall_prediction_cache_path,
    sample_full_history_mode_inputs,
)


DEFAULT_CACHE = ROOT / "results" / "unicm_overall_ei_cpu_bound4_n8192" / "cache"
DEFAULT_OUTPUT = ROOT / "results" / "unicm_target_pair_syn_tm_degree1_signed_n8192"


def source_columns(mode_index: int, history_length: int = 12) -> np.ndarray:
    start = int(mode_index) * int(history_length)
    return np.arange(start, start + int(history_length))


def gaussian_mi_scalar_targets(
    source_covariance: np.ndarray,
    source_target_covariance: np.ndarray,
    target_variance: np.ndarray,
    *,
    jitter: float,
) -> np.ndarray:
    regularized_source = source_covariance.copy()
    regularized_source.flat[:: regularized_source.shape[0] + 1] += float(jitter)
    regularized_target = target_variance + float(jitter)
    explained = np.sum(
        source_target_covariance
        * np.linalg.solve(regularized_source, source_target_covariance),
        axis=0,
    )
    conditional = np.maximum(regularized_target - explained, float(jitter))
    return 0.5 * np.log2(regularized_target / conditional)


def run(args: argparse.Namespace) -> int:
    mode_names = list(MODE_NAMES)
    history = sample_full_history_mode_inputs(
        n_samples=args.n_samples,
        intervention_bound=args.intervention_bound,
        seed=args.sampling_seed,
    ).astype(np.float64)
    # [sample, mode, history] -> source columns grouped by mode.
    source = history.transpose(0, 2, 1).reshape(args.n_samples, -1)
    cache_args = SimpleNamespace(
        n_samples=args.n_samples,
        sampling_seed=args.sampling_seed,
        intervention_bound=args.intervention_bound,
        start_month=args.start_month,
        device=args.device,
    )
    predictions = []
    output_keys: list[tuple[int, int, int]] = []
    for seed in args.seeds:
        values = load_full_history_prediction_cache(
            overall_prediction_cache_path(args.cache_dir, seed=seed, args=cache_args),
            n_samples=args.n_samples,
        ).astype(np.float64)
        for lead in range(24):
            for target_index in range(len(mode_names)):
                predictions.append(values[:, lead, target_index])
                output_keys.append((seed, lead + 1, target_index))
    target = np.stack(predictions, axis=1)

    source_centered = source - source.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    denominator = args.n_samples - 1
    source_covariance = source_centered.T @ source_centered / denominator
    cross_covariance = source_centered.T @ target_centered / denominator
    target_variance = np.sum(target_centered**2, axis=0) / denominator

    singleton_mi = np.empty((len(mode_names), target.shape[1]), dtype=np.float64)
    for mode_index in range(len(mode_names)):
        columns = source_columns(mode_index)
        singleton_mi[mode_index] = gaussian_mi_scalar_targets(
            source_covariance[np.ix_(columns, columns)],
            cross_covariance[columns],
            target_variance,
            jitter=args.jitter,
        )

    rows: list[dict[str, object]] = []
    for left in range(len(mode_names)):
        for right in range(left + 1, len(mode_names)):
            columns = np.concatenate((source_columns(left), source_columns(right)))
            joint = gaussian_mi_scalar_targets(
                source_covariance[np.ix_(columns, columns)],
                cross_covariance[columns],
                target_variance,
                jitter=args.jitter,
            )
            syn = joint - singleton_mi[left] - singleton_mi[right]
            for output_index, (seed, lead, target_index) in enumerate(output_keys):
                rows.append(
                    {
                        "seed": seed,
                        "lead": lead,
                        "target": mode_names[target_index],
                        "target_index": target_index,
                        "pair": f"{mode_names[left]}|{mode_names[right]}",
                        "left_source": mode_names[left],
                        "right_source": mode_names[right],
                        "left_ei": singleton_mi[left, output_index],
                        "right_ei": singleton_mi[right, output_index],
                        "joint_ei": joint[output_index],
                        "syn": syn[output_index],
                    }
                )

    frame = pd.DataFrame(rows).sort_values(
        ["target_index", "lead", "pair", "seed"]
    )
    summary = (
        frame.groupby(
            ["target", "target_index", "lead", "pair", "left_source", "right_source"],
            as_index=False,
        )["syn"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["target_index", "lead", "mean"], ascending=[True, True, False])
    )
    summary["rank"] = (
        summary.groupby(["target_index", "lead"])["mean"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "target_pair_syn_rows.csv", index=False)
    summary.to_csv(args.output_dir / "target_pair_syn_summary.csv", index=False)
    manifest = {
        "target_definition": "each scalar predicted Modeformer mode",
        "source_definition": "each pair of 12-month mode histories",
        "n_samples": args.n_samples,
        "intervention_bound": args.intervention_bound,
        "sampling_seed": args.sampling_seed,
        "start_month": args.start_month,
        "checkpoint_seeds": args.seeds,
        "estimator": "signed affine degree-1 TM / Gaussian logdet",
        "jitter": args.jitter,
        "rows": len(frame),
        "targets": len(mode_names),
        "pairs": 55,
        "leads": 24,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--intervention-bound", type=float, default=4.0)
    parser.add_argument("--sampling-seed", type=int, default=20260619)
    parser.add_argument("--start-month", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--jitter", type=float, default=1e-6)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
