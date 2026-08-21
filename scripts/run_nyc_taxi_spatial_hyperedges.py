#!/usr/bin/env python3
"""Discover and confirm directed spatial synergy hyperedges in NYC Taxi MGSTN.

The discovery screen ranks finite two-source interactions for every target zone.
Candidates are then frozen and re-estimated on independent interventions with a
quadratic triangular transport map (TM).  Sources are independently sampled,
so the PEID synergy is the conditional mutual information I(A; B | C).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TRAIN_SCRIPT = ROOT / "scripts/train_nyc_taxi_mgstn.py"
EI_SCRIPT = ROOT / "scripts/compute_nyc_taxi_mgstn_ei.py"
TM_SCRIPT = ROOT / "scripts/compute_nyc_taxi_mgstn_quadratic_tm_full.py"
RUN_DIR = ROOT / "results/nyc_taxi_mgstn_ei/spatial_hyperedges"
STATUS = ROOT / "docs/log/nyc_taxi_spatial_hyperedges/live_progress.json"

NODES = 66
FLOW_IN = 0
FLOW_OUT = 1
LOG_2 = float(np.log(2.0))
NONNEGATIVE_TOLERANCE_BITS = 0.05
DEFAULT_STATES = ["weekday_peak", "weekend_midday", "rainy_high_demand"]
DEFAULT_SEEDS = [0, 1, 2]
SCREEN_STATE = "weekday_peak"
SCREEN_SAMPLES = 256
FORMAL_SAMPLES = 4096
PCA_COMPONENTS = 2
TOP_K_PER_TARGET = 3


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_status(*, phase: str, current: int, total: int, started: float, message: str = "", metrics=None) -> None:
    elapsed = time.monotonic() - started
    rate = current / elapsed if elapsed > 0 and current > 0 else 0.0
    atomic_json(
        STATUS,
        {
            "phase": phase,
            "current": int(current),
            "total": int(total),
            "unit": "pair-state-stage",
            "elapsed_seconds": float(elapsed),
            "eta_seconds": float((total - current) / rate) if rate > 0 else None,
            "message": message,
            "metrics": metrics or {},
            "updated_at": time.time(),
        },
    )


def source_outflow(raw_full: np.ndarray) -> np.ndarray:
    """Select 19 outflow lags from [recent(2x7), daily(2x5), weekly(2x7)]."""
    columns = np.r_[np.arange(7, 14), np.arange(19, 24), np.arange(31, 38)]
    return np.asarray(raw_full[..., columns], dtype=np.float32)


def hurdle(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return np.concatenate([(array > 0).astype(np.float32), np.log1p(array)], axis=-1)


def prepare_source_features(pools_raw: dict[str, np.ndarray], *, components: int) -> np.ndarray:
    path = RUN_DIR / f"source_hurdle_pca_pc{components}.npz"
    if path.exists():
        saved = np.load(path, allow_pickle=False)
        return saved["scores"]
    outflow = source_outflow(pools_raw["full"])
    scores = np.empty((NODES, len(outflow), components), dtype=np.float32)
    metadata = []
    for zone in tqdm(range(NODES), desc="source PCA", unit="zone"):
        features = hurdle(outflow[:, zone])
        mean = features.mean(axis=0)
        centered = features - mean
        model = PCA(n_components=components, svd_solver="full")
        scores[zone] = model.fit_transform(centered).astype(np.float32)
        metadata.append(
            {
                "zone_index": zone,
                "input_dim": int(features.shape[1]),
                "components": int(components),
                "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
            }
        )
    atomic_npz(path, scores=scores, metadata=np.asarray(json.dumps(metadata)))
    return scores


def make_center_batch(center: dict[str, torch.Tensor], width: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.unsqueeze(0).expand(width, *value.shape).clone().to(device)
        for key, value in center.items()
    }


def replace_source_outflow(
    batch: dict[str, torch.Tensor],
    zone: int,
    indices: np.ndarray,
    pools_z: dict[str, np.ndarray],
    device: torch.device,
) -> None:
    for branch in ("recent", "daily", "weekly"):
        values = torch.from_numpy(pools_z[branch][indices, zone, FLOW_OUT]).to(device)
        batch[branch][:, zone, FLOW_OUT] = values


def screen_pair(
    model,
    center: dict[str, torch.Tensor],
    source_a: int,
    source_b: int,
    pools_z: dict[str, np.ndarray],
    *,
    samples: int,
    pair_seed: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    rng = np.random.default_rng(pair_seed)
    count = len(pools_z["full"])
    a0 = rng.integers(0, count, samples, dtype=np.int32)
    a1 = rng.integers(0, count, samples, dtype=np.int32)
    b0 = rng.integers(0, count, samples, dtype=np.int32)
    b1 = rng.integers(0, count, samples, dtype=np.int32)
    combinations_ = ((a0, b0), (a1, b0), (a0, b1), (a1, b1))
    all_a = np.concatenate([item[0] for item in combinations_])
    all_b = np.concatenate([item[1] for item in combinations_])
    outputs = []
    with torch.inference_mode():
        chunks = []
        for start in range(0, len(all_a), batch_size):
            stop = min(len(all_a), start + batch_size)
            batch = make_center_batch(center, stop - start, device)
            replace_source_outflow(batch, source_a, all_a[start:stop], pools_z, device)
            replace_source_outflow(batch, source_b, all_b[start:stop], pools_z, device)
            chunks.append(model(batch)[:, :, FLOW_IN].detach().cpu().numpy())
        combined = np.concatenate(chunks)
        outputs = np.split(combined, 4)
    interaction = outputs[3] - outputs[1] - outputs[2] + outputs[0]
    return np.sqrt(np.mean(interaction.astype(np.float64) ** 2, axis=0))


def run_screen(
    training,
    ei,
    data: dict,
    prepared: dict,
    split: dict,
    pools_z: dict[str, np.ndarray],
    *,
    samples: int,
    device: torch.device,
    batch_size: int,
    pair_limit: int | None,
    started: float,
) -> Path:
    pairs = np.asarray(list(combinations(range(NODES), 2)), dtype=np.int16)
    if pair_limit is not None:
        pairs = pairs[:pair_limit]
    cache = RUN_DIR / f"screen_seed0_{SCREEN_STATE}_n{samples}.npz"
    scores = np.full((len(pairs), NODES), np.nan, dtype=np.float32)
    completed = np.zeros(len(pairs), dtype=bool)
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        if np.array_equal(saved["pairs"], pairs):
            scores = saved["scores"]
            completed = saved["completed"].astype(bool)
    model, _ = ei.load_model(training, data, prepared["attributes_z"].shape[1], 0, device)
    center_item = ei.select_centers(training, data, prepared, split, [SCREEN_STATE])[0]
    center = center_item["batch"]
    total = len(pairs)
    bar = tqdm(total=total, initial=int(completed.sum()), desc="spatial screen", unit="pair")
    for index, (source_a, source_b) in enumerate(pairs):
        if completed[index]:
            continue
        scores[index] = screen_pair(
            model,
            center,
            int(source_a),
            int(source_b),
            pools_z,
            samples=samples,
            pair_seed=91_000 + index,
            device=device,
            batch_size=batch_size,
        )
        scores[index, [source_a, source_b]] = np.nan
        completed[index] = True
        bar.update(1)
        if completed.sum() % 10 == 0 or completed.all():
            atomic_npz(cache, pairs=pairs, scores=scores, completed=completed)
            write_status(
                phase="screen",
                current=int(completed.sum()),
                total=total,
                started=started,
                metrics={"pair": [int(source_a), int(source_b)], "screen_samples": samples},
            )
    bar.close()
    return cache


def freeze_candidates(screen_path: Path, data: dict, *, top_k: int) -> Path:
    output = RUN_DIR / f"frozen_candidates_{screen_path.stem}_top{top_k}.json"
    if output.exists():
        return output
    saved = np.load(screen_path, allow_pickle=False)
    if not bool(saved["completed"].all()):
        raise RuntimeError("cannot freeze candidates from an incomplete screen")
    pairs = saved["pairs"].astype(int)
    scores = saved["scores"].astype(float)
    zone_ids = data["zone_ids"].astype(int)
    zone_names = data["zone_names"].astype(str)
    candidates = []
    for target in range(NODES):
        valid = np.flatnonzero(np.isfinite(scores[:, target]))
        order = valid[np.argsort(scores[valid, target])[::-1][:top_k]]
        for rank, pair_index in enumerate(order, start=1):
            source_a, source_b = pairs[pair_index]
            candidates.append(
                {
                    "candidate_id": f"{zone_ids[source_a]}_{zone_ids[source_b]}_to_{zone_ids[target]}",
                    "source_a_index": int(source_a),
                    "source_b_index": int(source_b),
                    "target_index": int(target),
                    "source_a_id": int(zone_ids[source_a]),
                    "source_b_id": int(zone_ids[source_b]),
                    "target_id": int(zone_ids[target]),
                    "source_a_name": str(zone_names[source_a]),
                    "source_b_name": str(zone_names[source_b]),
                    "target_name": str(zone_names[target]),
                    "screen_rank": rank,
                    "screen_interaction_rms_z": float(scores[pair_index, target]),
                }
            )
    atomic_json(
        output,
        {
            "status": "frozen",
            "screen": str(screen_path.relative_to(ROOT)),
            "screen_model_seed": 0,
            "screen_state": SCREEN_STATE,
            "screen_samples": int(screen_path.stem.rsplit("_n", 1)[1]),
            "top_k_per_target": top_k,
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
    )
    return output


def intervention_path(pair: tuple[int, int], samples: int) -> Path:
    return RUN_DIR / f"formal_interventions/n{samples}/pair_{pair[0]}_{pair[1]}.npz"


def prepare_formal_intervention(
    pair: tuple[int, int],
    source_scores: np.ndarray,
    *,
    samples: int,
) -> Path:
    path = intervention_path(pair, samples)
    if path.exists():
        return path
    seed = 710_000 + pair[0] * 1000 + pair[1]
    rng = np.random.default_rng(seed)
    count = source_scores.shape[1]
    left_indices = rng.integers(0, count, samples, dtype=np.int32)
    right_indices = rng.integers(0, count, samples, dtype=np.int32)
    order = rng.permutation(samples)
    fit_n = int(0.70 * samples)
    fit_indices = np.sort(order[:fit_n]).astype(np.int32)
    evaluate_indices = np.sort(order[fit_n:]).astype(np.int32)
    atomic_npz(
        path,
        left_indices=left_indices,
        right_indices=right_indices,
        left_features=source_scores[pair[0], left_indices],
        right_features=source_scores[pair[1], right_indices],
        fit_indices=fit_indices,
        evaluate_indices=evaluate_indices,
    )
    return path


def predict_pair_targets(
    model,
    center: dict[str, torch.Tensor],
    pair: tuple[int, int],
    target_indices: list[int],
    intervention: dict[str, np.ndarray],
    pools_z: dict[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    samples = len(intervention["left_indices"])
    output = []
    with torch.inference_mode():
        for start in range(0, samples, batch_size):
            stop = min(samples, start + batch_size)
            batch = make_center_batch(center, stop - start, device)
            replace_source_outflow(batch, pair[0], intervention["left_indices"][start:stop], pools_z, device)
            replace_source_outflow(batch, pair[1], intervention["right_indices"][start:stop], pools_z, device)
            output.append(model(batch)[:, target_indices, FLOW_IN].detach().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def conditional_log_prob(model, samples: np.ndarray, start: int) -> np.ndarray:
    reference = model.map_to_reference(samples)
    selected = reference[:, start:]
    scales = model.residual_scales[start:]
    return -0.5 * np.sum(np.log(2.0 * np.pi) + selected**2, axis=1) - float(np.log(scales).sum())


def directional_cmi_pointwise(
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    fit_indices: np.ndarray,
    evaluate_indices: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    from exp.TM.transport_map_density import fit_polynomial_triangular_transport_map_density

    target_2d = np.asarray(target, dtype=np.float64).reshape(-1, 1)
    full_fit = np.column_stack([target_2d[fit_indices], left[fit_indices], right[fit_indices]])
    reduced_fit = np.column_stack([target_2d[fit_indices], right[fit_indices]])
    full_eval = np.column_stack([target_2d[evaluate_indices], left[evaluate_indices], right[evaluate_indices]])
    reduced_eval = np.column_stack([target_2d[evaluate_indices], right[evaluate_indices]])
    full_model = fit_polynomial_triangular_transport_map_density(full_fit, degree=2, ridge=ridge)
    reduced_model = fit_polynomial_triangular_transport_map_density(reduced_fit, degree=2, ridge=ridge)
    return (
        conditional_log_prob(full_model, full_eval, start=1 + left.shape[1])
        - conditional_log_prob(reduced_model, reduced_eval, start=1)
    ) / LOG_2


def estimate_symmetric_cmi(
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    fit_indices: np.ndarray,
    evaluate_indices: np.ndarray,
    *,
    ridge: float,
) -> dict[str, float]:
    forward = directional_cmi_pointwise(
        left, right, target, fit_indices, evaluate_indices, ridge=ridge
    )
    reverse = directional_cmi_pointwise(
        right, left, target, fit_indices, evaluate_indices, ridge=ridge
    )
    pointwise = 0.5 * (forward + reverse)
    return {
        "syn_bits_raw": float(pointwise.mean()),
        "syn_bits_se": float(pointwise.std(ddof=1) / np.sqrt(len(pointwise))),
        "order_gap_bits": float(abs(forward.mean() - reverse.mean())),
        "evaluation_samples": int(len(pointwise)),
    }


def response_path(samples: int, seed: int, state: str, pair: tuple[int, int]) -> Path:
    return RUN_DIR / f"formal_responses/n{samples}/seed_{seed}/{state}/pair_{pair[0]}_{pair[1]}.npz"


def unit_path(samples: int, seed: int, state: str, candidate_id: str) -> Path:
    return RUN_DIR / f"formal_units/n{samples}/seed_{seed}/{state}/{candidate_id}.json"


def aggregate(candidate_path: Path, unit_paths: list[Path], *, samples: int, ridge: float) -> dict:
    frozen = json.loads(candidate_path.read_text(encoding="utf-8"))
    units = [json.loads(path.read_text(encoding="utf-8")) for path in unit_paths]
    raw = np.asarray([unit["observed"]["syn_bits_raw"] for unit in units], dtype=float)
    violations = raw < -NONNEGATIVE_TOLERANCE_BITS
    numerical = (raw < 0) & ~violations
    by_candidate = defaultdict(list)
    for unit in units:
        by_candidate[unit["candidate_id"]].append(unit)
    summaries = []
    for candidate in frozen["candidates"]:
        rows = by_candidate[candidate["candidate_id"]]
        if not rows:
            continue
        observed = np.asarray([row["observed"]["syn_bits_raw"] for row in rows])
        null = np.asarray([row["null"]["syn_bits_raw"] for row in rows])
        delta = observed - null
        seed_means = []
        for seed in sorted({row["model_seed"] for row in rows}):
            seed_rows = [row for row in rows if row["model_seed"] == seed]
            seed_means.append(float(np.mean([
                row["observed"]["syn_bits_raw"] - row["null"]["syn_bits_raw"] for row in seed_rows
            ])))
        state_means = {}
        for state in sorted({row["state"] for row in rows}):
            state_rows = [row for row in rows if row["state"] == state]
            state_means[state] = float(np.mean([
                row["observed"]["syn_bits_raw"] - row["null"]["syn_bits_raw"] for row in state_rows
            ]))
        summaries.append(
            {
                **candidate,
                "observed_mean_bits": float(observed.mean()),
                "null_mean_bits": float(null.mean()),
                "paired_delta_mean_bits": float(delta.mean()),
                "paired_delta_sd_bits": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                "positive_seed_count": int(np.sum(np.asarray(seed_means) > 0)),
                "seed_mean_deltas_bits": seed_means,
                "state_mean_deltas_bits": state_means,
                "confirmed": bool(delta.mean() >= 0.05 and np.sum(np.asarray(seed_means) > 0) >= 2),
            }
        )
    summaries.sort(key=lambda row: row["paired_delta_mean_bits"], reverse=True)
    audit = {
        "status": "invalid_nonnegative_violation" if bool(violations.any()) else "complete",
        "tolerance_bits": NONNEGATIVE_TOLERANCE_BITS,
        "minimum_raw_syn_bits": float(raw.min()),
        "numerical_zero_count": int(numerical.sum()),
        "violation_count": int(violations.sum()),
    }
    return {
        "status": audit["status"],
        "method": "independent finite-amplitude source interventions; hurdle PCA2; symmetric quadratic TM CMI",
        "samples_per_candidate_state_seed": samples,
        "ridge": ridge,
        "nonnegative_audit": audit,
        "candidate_count": len(summaries),
        "confirmed_count": int(sum(row["confirmed"] for row in summaries)),
        "candidates": summaries,
        "units": units,
    }


def run_confirm(
    training,
    ei,
    data: dict,
    prepared: dict,
    split: dict,
    pools_z: dict[str, np.ndarray],
    source_scores: np.ndarray,
    candidate_path: Path,
    *,
    samples: int,
    states: list[str],
    seeds: list[int],
    ridge: float,
    device: torch.device,
    batch_size: int,
    target_limit: int | None,
    started: float,
) -> Path:
    frozen = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidates = frozen["candidates"]
    if target_limit is not None:
        allowed = sorted({row["target_index"] for row in candidates})[:target_limit]
        candidates = [row for row in candidates if row["target_index"] in allowed]
    pair_targets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in candidates:
        pair = (row["source_a_index"], row["source_b_index"])
        pair_targets[pair].append(row["target_index"])
    for pair in pair_targets:
        pair_targets[pair] = sorted(set(pair_targets[pair]))
        prepare_formal_intervention(pair, source_scores, samples=samples)

    total = len(candidates) * len(states) * len(seeds)
    current = 0
    unit_paths: list[Path] = []
    bar = tqdm(total=total, desc="formal spatial TM", unit="candidate-state-seed")
    for model_seed in seeds:
        model, _ = ei.load_model(training, data, prepared["attributes_z"].shape[1], model_seed, device)
        noise_covariance, noise_meta = ei.residual_covariance(training, model, prepared, split, device)
        centers = ei.select_centers(training, data, prepared, split, states)
        centers_by_state = {item["name"]: item["batch"] for item in centers}
        for state in states:
            center = centers_by_state[state]
            for pair, target_indices in pair_targets.items():
                intervention_saved = np.load(intervention_path(pair, samples), allow_pickle=False)
                intervention = {key: intervention_saved[key] for key in intervention_saved.files}
                response_cache = response_path(samples, model_seed, state, pair)
                if response_cache.exists():
                    response_saved = np.load(response_cache, allow_pickle=False)
                    if not np.array_equal(response_saved["target_indices"], np.asarray(target_indices)):
                        raise RuntimeError(f"target list mismatch in {response_cache}")
                    response = response_saved["response"]
                else:
                    response = predict_pair_targets(
                        model,
                        center,
                        pair,
                        target_indices,
                        intervention,
                        pools_z,
                        device=device,
                        batch_size=batch_size,
                    )
                    atomic_npz(
                        response_cache,
                        target_indices=np.asarray(target_indices, dtype=np.int16),
                        response=response,
                    )
                rows_for_pair = [
                    row for row in candidates
                    if (row["source_a_index"], row["source_b_index"]) == pair
                ]
                for candidate in rows_for_pair:
                    path = unit_path(samples, model_seed, state, candidate["candidate_id"])
                    unit_paths.append(path)
                    if path.exists():
                        current += 1
                        bar.update(1)
                        continue
                    target_position = target_indices.index(candidate["target_index"])
                    target = response[:, target_position].astype(np.float64)
                    target_output_index = 2 * candidate["target_index"] + FLOW_IN
                    noise_sd = float(np.sqrt(noise_covariance[target_output_index, target_output_index]))
                    rng = np.random.default_rng(
                        2_000_000 + model_seed * 100_000 + candidate["target_index"] * 1000
                        + pair[0] * 10 + pair[1]
                    )
                    target = target + rng.normal(0.0, noise_sd, size=samples)
                    permutation = rng.permutation(samples)
                    left = intervention["left_features"].astype(np.float64)
                    right = intervention["right_features"].astype(np.float64)
                    fit_indices = intervention["fit_indices"].astype(int)
                    evaluate_indices = intervention["evaluate_indices"].astype(int)
                    observed = estimate_symmetric_cmi(
                        left, right, target, fit_indices, evaluate_indices, ridge=ridge
                    )
                    null = estimate_symmetric_cmi(
                        left, right, target[permutation], fit_indices, evaluate_indices, ridge=ridge
                    )
                    record = {
                        **candidate,
                        "state": state,
                        "model_seed": model_seed,
                        "samples": samples,
                        "noise_sd_z": noise_sd,
                        "noise_metadata": noise_meta,
                        "observed": observed,
                        "null": null,
                        "paired_delta_bits": observed["syn_bits_raw"] - null["syn_bits_raw"],
                    }
                    atomic_json(path, record)
                    current += 1
                    bar.update(1)
                    write_status(
                        phase="formal_tm",
                        current=current,
                        total=total,
                        started=started,
                        metrics={
                            "seed": model_seed,
                            "state": state,
                            "candidate": candidate["candidate_id"],
                            "observed_syn_bits": observed["syn_bits_raw"],
                            "null_syn_bits": null["syn_bits_raw"],
                        },
                    )
    bar.close()
    result = aggregate(candidate_path, unit_paths, samples=samples, ridge=ridge)
    suffix = "smoke" if target_limit is not None else "full"
    summary_path = RUN_DIR / f"spatial_hyperedge_{suffix}_summary.json"
    atomic_json(summary_path, result)
    audit = result["nonnegative_audit"]
    if audit["violation_count"]:
        raise RuntimeError(
            "formal spatial TM Syn nonnegativity violation: "
            f"min={audit['minimum_raw_syn_bits']:.6g}, "
            f"threshold={-NONNEGATIVE_TOLERANCE_BITS}, count={audit['violation_count']}"
        )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "screen", "confirm"], default="all")
    parser.add_argument("--screen-samples", type=int, default=SCREEN_SAMPLES)
    parser.add_argument("--formal-samples", type=int, default=FORMAL_SAMPLES)
    parser.add_argument("--top-k", type=int, default=TOP_K_PER_TARGET)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--states", nargs="+", default=DEFAULT_STATES)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    pair_limit = 12 if args.smoke else None
    target_limit = 2 if args.smoke else None
    if args.smoke:
        args.screen_samples = min(args.screen_samples, 64)
        args.formal_samples = min(args.formal_samples, 1024)
        args.states = [args.states[0]]
        args.model_seeds = [args.model_seeds[0]]

    started = time.monotonic()
    training = load_module(TRAIN_SCRIPT, "nyc_taxi_training_spatial_hyperedges")
    ei = load_module(EI_SCRIPT, "nyc_taxi_ei_spatial_hyperedges")
    tm = load_module(TM_SCRIPT, "nyc_taxi_tm_spatial_hyperedges")
    data, prepared, split = training.load_data(False)
    training_times = np.asarray(split["indices"]["train"], dtype=int)
    pools_z = tm.history_pools(prepared["flow_z"].astype(np.float32), training_times)
    pools_raw = tm.history_pools(data["flow"].astype(np.float32), training_times)
    device = ei.choose_device(args.device)
    try:
        screen_path = RUN_DIR / f"screen_seed0_{SCREEN_STATE}_n{args.screen_samples}.npz"
        if args.stage in {"all", "screen"}:
            screen_path = run_screen(
                training,
                ei,
                data,
                prepared,
                split,
                pools_z,
                samples=args.screen_samples,
                device=device,
                batch_size=args.batch_size,
                pair_limit=pair_limit,
                started=started,
            )
        candidate_path = freeze_candidates(screen_path, data, top_k=args.top_k)
        if args.stage == "screen":
            write_status(
                phase="screen_complete", current=1, total=1, started=started,
                metrics={"candidates": str(candidate_path)},
            )
            return
        source_scores = prepare_source_features(pools_raw, components=PCA_COMPONENTS)
        summary_path = run_confirm(
            training,
            ei,
            data,
            prepared,
            split,
            pools_z,
            source_scores,
            candidate_path,
            samples=args.formal_samples,
            states=args.states,
            seeds=args.model_seeds,
            ridge=args.ridge,
            device=device,
            batch_size=args.batch_size,
            target_limit=target_limit,
            started=started,
        )
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        write_status(
            phase="complete", current=1, total=1, started=started,
            metrics={
                "summary": str(summary_path),
                "confirmed_count": result["confirmed_count"],
                "audit": result["nonnegative_audit"],
            },
        )
        print(json.dumps({
            "summary": str(summary_path),
            "confirmed_count": result["confirmed_count"],
            "audit": result["nonnegative_audit"],
        }, ensure_ascii=False, indent=2))
    except Exception as error:
        write_status(phase="failed", current=0, total=1, started=started, message=str(error))
        raise


if __name__ == "__main__":
    main()
