#!/usr/bin/env python3
"""Compute selected Runge multistep conditioned EI horizons without filling every H."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_runge_multistep_conditioned_ei import (  # noqa: E402
    DEFAULT_PAIRWISE_MANIFEST,
    DEFAULT_RESULT_DIR,
    conditional_average_matrix,
    config_from_manifest,
    estimate_mi,
    estimate_pairwise_for_horizon,
    load_cached_pairwise_model,
    pairwise_edge_frame,
    rollout_mlp_closed_loop,
    sample_max_entropy_features,
    save_matrix_csv,
    source_state_matrix,
)


def parse_horizons(text: str) -> list[int]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not values:
        raise ValueError("At least one horizon is required.")
    if values[0] < 1:
        raise ValueError("Horizons must be positive.")
    return values


def ensure_rollout(
    result_dir: Path,
    *,
    model: object,
    scalers: dict[str, np.ndarray],
    features: np.ndarray,
    n_components: int,
    lag: int,
    max_horizon: int,
    resume: bool,
) -> np.ndarray:
    rollout_path = result_dir / "rollout_predictions.npy"
    if rollout_path.exists() and bool(resume):
        existing = np.load(rollout_path)
        if existing.ndim == 3 and existing.shape[1] >= int(max_horizon):
            return existing
    predictions = rollout_mlp_closed_loop(
        model,
        scalers,
        features,
        n_components=int(n_components),
        lag=int(lag),
        horizons=int(max_horizon),
    )
    np.save(rollout_path, predictions)
    return predictions


def source_pairs(n_components: int) -> list[tuple[int, int]]:
    return [(first, second) for first in range(int(n_components)) for second in range(first + 1, int(n_components))]


def _compute_joint_chunk(
    *,
    chunk_path: str,
    chunk_pairs: list[tuple[int, int]],
    source_states: list[np.ndarray],
    horizon_targets: np.ndarray,
    estimator: str,
    bins: int,
) -> str:
    n = len(source_states)
    values = np.full((len(chunk_pairs), n), np.nan, dtype=float)
    bias_values = np.full((len(chunk_pairs), n), np.nan, dtype=float)
    for pair_idx, (first, second) in enumerate(chunk_pairs):
        joint_source = np.concatenate([source_states[first], source_states[second]], axis=1)
        for target in range(n):
            if target in (first, second):
                continue
            ei, bias = estimate_mi(joint_source, horizon_targets[:, [target]], estimator=estimator, bins=int(bins))
            values[pair_idx, target] = ei
            if bias is not None:
                bias_values[pair_idx, target] = bias
    path = Path(chunk_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        first=np.asarray([pair[0] for pair in chunk_pairs], dtype=np.int16),
        second=np.asarray([pair[1] for pair in chunk_pairs], dtype=np.int16),
        values=values,
        bias=bias_values,
    )
    return str(path)


def estimate_horizon_conditioned_ei_parallel(
    source_states: list[np.ndarray],
    horizon_targets: np.ndarray,
    *,
    estimator: str,
    bins: int,
    names: list[str],
    horizon_dir: Path,
    chunk_size: int,
    resume: bool,
    jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon_dir.mkdir(parents=True, exist_ok=True)
    pairwise_path = horizon_dir / "pairwise_ei.npy"
    pairwise_bias_path = horizon_dir / "pairwise_ei_bias.npy"
    if pairwise_path.exists() and pairwise_bias_path.exists() and bool(resume):
        pairwise = np.load(pairwise_path)
        pairwise_bias = np.load(pairwise_bias_path)
    else:
        pairwise, pairwise_bias = estimate_pairwise_for_horizon(
            source_states,
            horizon_targets,
            estimator=estimator,
            bins=int(bins),
        )
        np.save(pairwise_path, pairwise)
        np.save(pairwise_bias_path, pairwise_bias)
        save_matrix_csv(pairwise, names, horizon_dir / "pairwise_ei_matrix.csv")

    joint_path = horizon_dir / "joint_ei.npy"
    pairs = source_pairs(len(source_states))
    chunk_size = max(1, int(chunk_size))
    joint_dir = horizon_dir / "joint_chunks"
    joint_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[Path, list[tuple[int, int]]]] = []
    for chunk_start in range(0, len(pairs), chunk_size):
        chunk_pairs = pairs[chunk_start : chunk_start + chunk_size]
        chunk_path = joint_dir / f"chunk_{chunk_start // chunk_size:04d}.npz"
        if chunk_path.exists() and bool(resume):
            print(f"[joint] reuse {chunk_path}", flush=True)
            continue
        tasks.append((chunk_path, chunk_pairs))

    if tasks:
        if int(jobs) <= 1:
            for chunk_path, chunk_pairs in tasks:
                print(f"[joint] computing {chunk_path} pairs={len(chunk_pairs)}", flush=True)
                _compute_joint_chunk(
                    chunk_path=str(chunk_path),
                    chunk_pairs=chunk_pairs,
                    source_states=source_states,
                    horizon_targets=horizon_targets,
                    estimator=estimator,
                    bins=int(bins),
                )
        else:
            print(f"[joint] computing {len(tasks)} chunks with jobs={int(jobs)}", flush=True)
            with ProcessPoolExecutor(max_workers=int(jobs)) as pool:
                futures = [
                    pool.submit(
                        _compute_joint_chunk,
                        chunk_path=str(chunk_path),
                        chunk_pairs=chunk_pairs,
                        source_states=source_states,
                        horizon_targets=horizon_targets,
                        estimator=estimator,
                        bins=int(bins),
                    )
                    for chunk_path, chunk_pairs in tasks
                ]
                for future in as_completed(futures):
                    print(f"[joint] done {future.result()}", flush=True)

    expected_chunks = math.ceil(len(pairs) / chunk_size)
    chunk_paths = sorted(joint_dir.glob("chunk_*.npz"))
    if len(chunk_paths) < expected_chunks:
        raise RuntimeError(f"joint EI chunks incomplete: {len(chunk_paths)}/{expected_chunks}.")
    joint = np.full((len(source_states), len(source_states), len(source_states)), np.nan, dtype=float)
    for chunk_path in chunk_paths:
        payload = np.load(chunk_path)
        first_indices = payload["first"].astype(int)
        second_indices = payload["second"].astype(int)
        values = payload["values"].astype(float)
        for row_idx, (first, second) in enumerate(zip(first_indices, second_indices, strict=True)):
            joint[first, second, :] = values[row_idx]
            joint[second, first, :] = values[row_idx]
    np.save(joint_path, joint)
    conditioned = conditional_average_matrix(pairwise, joint)
    return pairwise, pairwise_bias, joint, conditioned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise-manifest", type=Path, default=DEFAULT_PAIRWISE_MANIFEST)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--horizons", default="15,20,30,40,50,60")
    parser.add_argument("--intervention-samples", type=int, default=4096)
    parser.add_argument("--estimator", choices=["tm", "discrete"], default="tm")
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--source-mode", choices=["latest", "history"], default="latest")
    parser.add_argument("--joint-chunk-size", type=int, default=60)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    horizons = parse_horizons(str(args.horizons))
    result_dir = Path(args.result_dir).expanduser()
    result_dir.mkdir(parents=True, exist_ok=True)
    config = config_from_manifest(Path(args.pairwise_manifest).expanduser())
    config = replace(
        config,
        intervention_samples=int(args.intervention_samples),
        ei_estimator=str(args.estimator),
        source_mode=str(args.source_mode),
        force_retrain=False,
    )
    model, scalers, splits, names, model_info = load_cached_pairwise_model(config)
    n_components = len(names)
    features = sample_max_entropy_features(
        splits["train"][0],
        n_components=n_components,
        lag=int(config.lag),
        samples=int(args.intervention_samples),
        low_q=float(config.quantile_low),
        high_q=float(config.quantile_high),
        seed=int(config.seed),
    )
    source_states = source_state_matrix(
        features,
        n_components=n_components,
        lag=int(config.lag),
        source_mode=str(args.source_mode),
    )
    predictions = ensure_rollout(
        result_dir,
        model=model,
        scalers=scalers,
        features=features,
        n_components=n_components,
        lag=int(config.lag),
        max_horizon=max(horizons),
        resume=bool(args.resume),
    )

    rows: list[dict[str, object]] = []
    for horizon in horizons:
        started = time.perf_counter()
        horizon_dir = result_dir / f"horizon_{horizon:03d}"
        print(f"[horizon {horizon}] start", flush=True)
        pairwise, pairwise_bias, joint, signed = estimate_horizon_conditioned_ei_parallel(
            source_states,
            predictions[:, int(horizon) - 1, :],
            estimator=str(args.estimator),
            bins=int(args.bins),
            names=names,
            horizon_dir=horizon_dir,
            chunk_size=int(args.joint_chunk_size),
            resume=bool(args.resume),
            jobs=int(args.jobs),
        )
        edges = pairwise_edge_frame(pairwise, pairwise_bias, signed, names)
        np.save(horizon_dir / "conditioned_ei_signed.npy", signed)
        np.save(horizon_dir / "conditioned_ei_positive.npy", np.maximum(signed, 0.0))
        save_matrix_csv(pairwise, names, horizon_dir / "pairwise_ei_matrix.csv")
        save_matrix_csv(signed, names, horizon_dir / "conditioned_ei_signed.csv")
        save_matrix_csv(np.maximum(signed, 0.0), names, horizon_dir / "conditioned_ei_positive.csv")
        edges.to_csv(horizon_dir / "conditioned_ei_edges.csv", index=False)
        elapsed = time.perf_counter() - started
        row = {
            "horizon": int(horizon),
            "elapsed_seconds": float(elapsed),
            "horizon_dir": str(horizon_dir),
            "joint_shape": list(joint.shape),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary_path = result_dir / "selected_horizon_summary.json"
    payload = {
        "horizons": horizons,
        "config": {
            "intervention_samples": int(args.intervention_samples),
            "estimator": str(args.estimator),
            "source_mode": str(args.source_mode),
            "joint_chunk_size": int(args.joint_chunk_size),
            "jobs": int(args.jobs),
        },
        "runs": rows,
        **model_info,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
