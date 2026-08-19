#!/usr/bin/env python3
"""Full-zone finite-amplitude quadratic-TM audit for NYC Taxi MGSTN.

Two target-wise, two-block PEID quantities are estimated for every Taxi Zone:

Temporal Syn_i = I(recent_i; macro_i | next-hour flow_i)
Spatial Syn_i  = I(history_i; history_-i | next-hour flow_i)

Each source pair is independently bootstrapped from empirical training windows,
so the conditional mutual information equals the two-block PEID Syn. The target
is produced by the frozen MGSTN plus validation-residual noise. No Jacobian is
used. Baseline standardized-history PCA and a hurdle representation separating
activity from positive magnitude share samples, PCA width, TM degree, and split.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
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
RUN_DIR = ROOT / "results/nyc_taxi_mgstn_ei/finite_quadratic_tm"
STATUS = ROOT / "docs/log/nyc_taxi_finite_quadratic_tm/live_progress.json"
LOG_2 = float(np.log(2.0))
NONNEGATIVE_TOLERANCE_BITS = 0.05
DEFAULT_STATES = ["weekday_peak", "weekend_midday", "rainy_high_demand"]
DEFAULT_SEEDS = [0, 1, 2]
SPARSE_ZONE_IDS = {120, 127, 128, 153, 194, 202}
INTERVENTION_VERSION = "v2_centered_hurdle"


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


def write_status(*, phase: str, current: int, total: int, started: float, metrics=None, message=None) -> None:
    elapsed = time.monotonic() - started
    rate = current / elapsed if elapsed > 0 and current > 0 else 0.0
    payload = {
        "phase": phase,
        "current": int(current),
        "total": int(total),
        "unit": "zone-stage",
        "elapsed_seconds": elapsed,
        "eta_seconds": (total - current) / rate if rate > 0 else None,
        "metrics": metrics or {},
        "message": message,
        "updated_at": time.time(),
    }
    atomic_json(STATUS, payload)


def history_pools(flow: np.ndarray, times: np.ndarray) -> dict[str, np.ndarray]:
    recent_indices = times[:, None] - np.arange(7, 0, -1)[None, :]
    daily_indices = times[:, None] - 24 * np.arange(5, 0, -1)[None, :]
    weekly_indices = times[:, None] - 168 * np.arange(7, 0, -1)[None, :]
    recent = flow[recent_indices].transpose(0, 2, 3, 1)
    daily = flow[daily_indices].transpose(0, 2, 3, 1)
    weekly = flow[weekly_indices].transpose(0, 2, 3, 1)
    full = np.concatenate(
        [
            recent.reshape(len(times), recent.shape[1], -1),
            daily.reshape(len(times), daily.shape[1], -1),
            weekly.reshape(len(times), weekly.shape[1], -1),
        ],
        axis=2,
    )
    return {"recent": recent, "daily": daily, "weekly": weekly, "full": full}


def hurdle_features(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    return np.concatenate([(values > 0).astype(np.float32), np.log1p(values)], axis=1)


def fit_pca_features(
    values: np.ndarray,
    fit_indices: np.ndarray,
    *,
    components: int,
    seed: int,
    standardize: bool = True,
) -> tuple[np.ndarray, dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    mean = array[fit_indices].mean(axis=0)
    scale = array[fit_indices].std(axis=0) if standardize else np.ones(array.shape[1], dtype=np.float32)
    scale[scale < 1e-6] = 1.0
    standardized = (array - mean) / scale
    pca = PCA(n_components=int(components), svd_solver="randomized", random_state=int(seed))
    pca.fit(standardized[fit_indices])
    projected = pca.transform(standardized).astype(np.float32)
    metadata = {
        "input_dim": int(array.shape[1]),
        "components": int(components),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
        "column_standardization": bool(standardize),
    }
    return projected, metadata, mean.astype(np.float32), scale.astype(np.float32), pca.components_.astype(np.float32)


def conditional_log_prob(model, samples: np.ndarray, start: int) -> np.ndarray:
    reference = model.map_to_reference(samples)
    selected = reference[:, start:]
    scales = model.residual_scales[start:]
    return -0.5 * np.sum(np.log(2.0 * np.pi) + selected**2, axis=1) - float(np.log(scales).sum())


def estimate_quadratic_cmi(
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    fit_indices: np.ndarray,
    evaluate_indices: np.ndarray,
    *,
    ridge: float,
) -> dict[str, float]:
    from exp.TM.transport_map_density import fit_polynomial_triangular_transport_map_density

    full_fit = np.column_stack([target[fit_indices], left[fit_indices], right[fit_indices]])
    reduced_fit = np.column_stack([target[fit_indices], right[fit_indices]])
    full_eval = np.column_stack([target[evaluate_indices], left[evaluate_indices], right[evaluate_indices]])
    reduced_eval = np.column_stack([target[evaluate_indices], right[evaluate_indices]])
    full_model = fit_polynomial_triangular_transport_map_density(full_fit, degree=2, ridge=float(ridge))
    reduced_model = fit_polynomial_triangular_transport_map_density(reduced_fit, degree=2, ridge=float(ridge))
    pointwise = (
        conditional_log_prob(full_model, full_eval, start=2 + left.shape[1])
        - conditional_log_prob(reduced_model, reduced_eval, start=2)
    ) / LOG_2
    return {
        "syn_bits_raw": float(pointwise.mean()),
        "syn_bits_se": float(pointwise.std(ddof=1) / np.sqrt(len(pointwise))),
        "evaluation_samples": int(len(pointwise)),
    }


def intervention_cache_path(samples: int, components: int, zone_id: int) -> Path:
    return RUN_DIR / f"interventions/{INTERVENTION_VERSION}_n{samples}_pc{components}/zone_{zone_id}.npz"


def prepare_global_intervention(
    pools_z: dict[str, np.ndarray],
    pools_raw: dict[str, np.ndarray],
    *,
    samples: int,
    components: int,
    seed: int,
) -> dict[str, np.ndarray | dict]:
    cache = RUN_DIR / f"interventions/{INTERVENTION_VERSION}_n{samples}_pc{components}/global_external.npz"
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        return {key: saved[key] for key in saved.files}
    rng = np.random.default_rng(seed)
    external_indices = rng.integers(0, len(pools_z["full"]), size=samples, dtype=np.int32)
    order = rng.permutation(samples)
    fit_n = int(0.70 * samples)
    fit_indices = np.sort(order[:fit_n]).astype(np.int32)
    evaluate_indices = np.sort(order[fit_n:]).astype(np.int32)
    baseline = pools_z["full"][external_indices].reshape(samples, -1)
    hurdle = hurdle_features(pools_raw["full"][external_indices].reshape(samples, -1))
    base_scores, base_meta, base_mean, base_scale, base_basis = fit_pca_features(
        baseline, fit_indices, components=components, seed=seed
    )
    hurdle_scores, hurdle_meta, hurdle_mean, hurdle_scale, hurdle_basis = fit_pca_features(
        hurdle, fit_indices, components=components, seed=seed + 1, standardize=False
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        external_indices=external_indices,
        fit_indices=fit_indices,
        evaluate_indices=evaluate_indices,
        baseline_scores=base_scores,
        baseline_mean=base_mean,
        baseline_scale=base_scale,
        baseline_basis=base_basis,
        hurdle_scores=hurdle_scores,
        hurdle_mean=hurdle_mean,
        hurdle_scale=hurdle_scale,
        hurdle_basis=hurdle_basis,
        metadata=np.asarray(json.dumps({"baseline": base_meta, "hurdle": hurdle_meta})),
    )
    return {
        "external_indices": external_indices,
        "fit_indices": fit_indices,
        "evaluate_indices": evaluate_indices,
        "baseline_scores": base_scores,
        "baseline_mean": base_mean,
        "baseline_scale": base_scale,
        "baseline_basis": base_basis,
        "hurdle_scores": hurdle_scores,
        "hurdle_mean": hurdle_mean,
        "hurdle_scale": hurdle_scale,
        "hurdle_basis": hurdle_basis,
        "metadata": np.asarray(json.dumps({"baseline": base_meta, "hurdle": hurdle_meta})),
    }


def prepare_zone_intervention(
    zone: int,
    zone_id: int,
    pools_z: dict[str, np.ndarray],
    pools_raw: dict[str, np.ndarray],
    global_intervention: dict,
    *,
    samples: int,
    components: int,
    seed: int,
) -> Path:
    cache = intervention_cache_path(samples, components, zone_id)
    if cache.exists():
        return cache
    rng = np.random.default_rng(seed + zone_id)
    pool_size = len(pools_z["full"])
    recent_indices = rng.integers(0, pool_size, size=samples, dtype=np.int32)
    macro_indices = rng.integers(0, pool_size, size=samples, dtype=np.int32)
    own_indices = rng.integers(0, pool_size, size=samples, dtype=np.int32)
    fit_indices = np.asarray(global_intervention["fit_indices"], dtype=int)

    temporal_recent_z = pools_z["recent"][recent_indices, zone].reshape(samples, -1)
    temporal_macro_z = np.concatenate(
        [
            pools_z["daily"][macro_indices, zone].reshape(samples, -1),
            pools_z["weekly"][macro_indices, zone].reshape(samples, -1),
        ],
        axis=1,
    )
    temporal_recent_raw = pools_raw["recent"][recent_indices, zone].reshape(samples, -1)
    temporal_macro_raw = np.concatenate(
        [
            pools_raw["daily"][macro_indices, zone].reshape(samples, -1),
            pools_raw["weekly"][macro_indices, zone].reshape(samples, -1),
        ],
        axis=1,
    )
    own_z = pools_z["full"][own_indices, zone]
    own_raw = pools_raw["full"][own_indices, zone]

    features = {}
    metadata = {}
    value_blocks = {
        "temporal_recent_baseline": temporal_recent_z,
        "temporal_macro_baseline": temporal_macro_z,
        "spatial_own_baseline": own_z,
        "temporal_recent_hurdle": hurdle_features(temporal_recent_raw),
        "temporal_macro_hurdle": hurdle_features(temporal_macro_raw),
        "spatial_own_hurdle": hurdle_features(own_raw),
    }
    for name, values in value_blocks.items():
        projected, meta, *_ = fit_pca_features(
            values,
            fit_indices,
            components=components,
            seed=seed + zone_id + len(features),
            standardize=not name.endswith("hurdle"),
        )
        features[name] = projected
        metadata[name] = meta

    # Reuse a common citywide external PCA, then remove this zone's linear contribution.
    external_indices = np.asarray(global_intervention["external_indices"], dtype=int)
    full_zone_z = pools_z["full"][external_indices, zone]
    start = zone * 38
    stop = start + 38
    standardized_zone = (
        full_zone_z - global_intervention["baseline_mean"][start:stop]
    ) / global_intervention["baseline_scale"][start:stop]
    features["spatial_external_baseline"] = (
        global_intervention["baseline_scores"]
        - standardized_zone @ global_intervention["baseline_basis"][:, start:stop].T
    ).astype(np.float32)

    full_zone_hurdle = hurdle_features(full_zone_z * 0 + pools_raw["full"][external_indices, zone])
    hurdle_start = zone * 38
    hurdle_activity = slice(hurdle_start, hurdle_start + 38)
    hurdle_magnitude = slice(2508 + hurdle_start, 2508 + hurdle_start + 38)
    hurdle_columns = np.r_[np.arange(hurdle_activity.start, hurdle_activity.stop),
                           np.arange(hurdle_magnitude.start, hurdle_magnitude.stop)]
    standardized_hurdle = (
        full_zone_hurdle - global_intervention["hurdle_mean"][hurdle_columns]
    ) / global_intervention["hurdle_scale"][hurdle_columns]
    features["spatial_external_hurdle"] = (
        global_intervention["hurdle_scores"]
        - standardized_hurdle @ global_intervention["hurdle_basis"][:, hurdle_columns].T
    ).astype(np.float32)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        recent_indices=recent_indices,
        macro_indices=macro_indices,
        own_indices=own_indices,
        external_indices=external_indices.astype(np.int32),
        fit_indices=np.asarray(global_intervention["fit_indices"], dtype=np.int32),
        evaluate_indices=np.asarray(global_intervention["evaluate_indices"], dtype=np.int32),
        metadata=np.asarray(json.dumps(metadata)),
        **features,
    )
    return cache


def predict_intervention(
    model,
    center: dict[str, torch.Tensor],
    zone: int,
    indices: dict[str, np.ndarray],
    pools_z: dict[str, np.ndarray],
    *,
    kind: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    samples = len(indices["recent_indices"])
    output = []
    with torch.inference_mode():
        for start in range(0, samples, batch_size):
            stop = min(samples, start + batch_size)
            width = stop - start
            batch = {}
            if kind == "temporal":
                for key, value in center.items():
                    batch[key] = value.unsqueeze(0).expand(width, *value.shape).clone().to(device)
                r = indices["recent_indices"][start:stop]
                m = indices["macro_indices"][start:stop]
                batch["recent"][:, zone] = torch.from_numpy(pools_z["recent"][r, zone]).to(device)
                batch["daily"][:, zone] = torch.from_numpy(pools_z["daily"][m, zone]).to(device)
                batch["weekly"][:, zone] = torch.from_numpy(pools_z["weekly"][m, zone]).to(device)
            elif kind == "spatial":
                external = indices["external_indices"][start:stop]
                own = indices["own_indices"][start:stop]
                for branch in ("recent", "daily", "weekly"):
                    values = pools_z[branch][external].copy()
                    values[:, zone] = pools_z[branch][own, zone]
                    batch[branch] = torch.from_numpy(values).to(device)
                for key, value in center.items():
                    if key in {"recent", "daily", "weekly"}:
                        continue
                    batch[key] = value.unsqueeze(0).expand(width, *value.shape).clone().to(device)
            else:
                raise ValueError(kind)
            output.append(model(batch)[:, zone].cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def ridge_tag(ridge: float) -> str:
    return f"{ridge:.0e}".replace("-", "m").replace("+", "p")


def unit_path(samples: int, components: int, ridge: float, seed: int, state: str, zone_id: int) -> Path:
    return RUN_DIR / (
        f"units/{INTERVENTION_VERSION}_n{samples}_pc{components}_r{ridge_tag(ridge)}/"
        f"seed_{seed}/{state}/zone_{zone_id}.json"
    )


def response_path(samples: int, seed: int, state: str, zone_id: int) -> Path:
    return RUN_DIR / f"responses/n{samples}/seed_{seed}/{state}/zone_{zone_id}.npz"


def aggregate_results(paths: list[Path], *, samples: int, components: int, ridge: float) -> dict:
    units = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    keys = ["baseline_temporal", "baseline_spatial", "hurdle_temporal", "hurdle_spatial"]
    audits = {}
    for method in ("baseline", "hurdle"):
        method_keys = [f"{method}_temporal", f"{method}_spatial"]
        raw_values = np.asarray([unit[key]["syn_bits_raw"] for unit in units for key in method_keys])
        violations = raw_values < -NONNEGATIVE_TOLERANCE_BITS
        numerical = (raw_values < 0) & ~violations
        audits[method] = {
            "status": "invalid_nonnegative_violation" if np.any(violations) else "complete",
            "tolerance_bits": NONNEGATIVE_TOLERANCE_BITS,
            "minimum_raw_syn_bits": float(raw_values.min()),
            "numerical_zero_count": int(numerical.sum()),
            "violation_count": int(violations.sum()),
        }
        if not np.any(violations):
            for unit in units:
                for key in method_keys:
                    raw = unit[key]["syn_bits_raw"]
                    unit[key]["syn_bits_interpreted"] = (
                        0.0 if -NONNEGATIVE_TOLERANCE_BITS <= raw < 0 else raw
                    )

    states = sorted({unit["state"] for unit in units})
    seeds = sorted({unit["model_seed"] for unit in units})
    summaries = []
    for method in ("baseline", "hurdle"):
        if audits[method]["violation_count"]:
            continue
        for state in states:
            seed_rows = []
            for seed in seeds:
                selected = [unit for unit in units if unit["state"] == state and unit["model_seed"] == seed]
                temporal = float(sum(unit[f"{method}_temporal"]["syn_bits_interpreted"] for unit in selected))
                spatial = float(sum(unit[f"{method}_spatial"]["syn_bits_interpreted"] for unit in selected))
                sparse_temporal = float(sum(
                    unit[f"{method}_temporal"]["syn_bits_interpreted"]
                    for unit in selected if unit["zone_id"] in SPARSE_ZONE_IDS
                ))
                seed_rows.append({
                    "seed": seed,
                    "temporal_syn_bits": temporal,
                    "spatial_syn_bits": spatial,
                    "temporal_share": temporal / (temporal + spatial) if temporal + spatial > 0 else None,
                    "sparse_temporal_share": sparse_temporal / temporal if temporal > 0 else None,
                })
            summaries.append({
                "method": method,
                "state": state,
                "seed_rows": seed_rows,
                "temporal_syn_bits_mean": float(np.mean([row["temporal_syn_bits"] for row in seed_rows])),
                "spatial_syn_bits_mean": float(np.mean([row["spatial_syn_bits"] for row in seed_rows])),
                "temporal_share_mean": float(np.mean([
                    row["temporal_share"] for row in seed_rows if row["temporal_share"] is not None
                ])) if any(row["temporal_share"] is not None for row in seed_rows) else None,
                "temporal_share_sd": float(np.std(
                    [row["temporal_share"] for row in seed_rows if row["temporal_share"] is not None],
                    ddof=1,
                )) if sum(row["temporal_share"] is not None for row in seed_rows) > 1 else 0.0,
                "sparse_temporal_share_mean": float(np.mean([
                    row["sparse_temporal_share"]
                    for row in seed_rows if row["sparse_temporal_share"] is not None
                ])) if any(row["sparse_temporal_share"] is not None for row in seed_rows) else None,
            })
    return {
        "status": (
            "invalid_hurdle_nonnegative_violation"
            if audits["hurdle"]["violation_count"]
            else "complete_hurdle_baseline_invalid"
            if audits["baseline"]["violation_count"]
            else "complete"
        ),
        "formal_method": "hurdle" if not audits["hurdle"]["violation_count"] else None,
        "definition": {
            "temporal": "sum_i I(recent_i; macro_i | next-hour flow_i)",
            "spatial": "sum_i I(history_i; history_-i | next-hour flow_i)",
            "note": "target-wise two-block finite PEID; not the former 66-block all-output affine partition",
        },
        "estimator": "degree-2 polynomial autoregressive triangular TM, held-out evaluation",
        "representations": {
            "baseline": "standardized history with train-fit PCA",
            "hurdle": "separate activity indicators and log1p positive magnitudes with train-fit PCA",
        },
        "samples": samples,
        "components_per_block": components,
        "ridge": ridge,
        "nonnegative_audit": audits,
        "summary": summaries,
        "units": units,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--states", nargs="+", default=DEFAULT_STATES)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--zone-ids", type=int, nargs="+")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.samples = min(args.samples, 4096)
        args.states = [args.states[0]]
        args.model_seeds = [args.model_seeds[0]]

    started = time.monotonic()
    training = load_module(TRAIN_SCRIPT, "nyc_taxi_training_quadratic_tm")
    ei = load_module(EI_SCRIPT, "nyc_taxi_ei_quadratic_tm")
    data, prepared, split = training.load_data(False)
    zone_ids_all = data["zone_ids"].astype(int)
    zone_names = data["zone_names"].astype(str)
    requested_ids = args.zone_ids or zone_ids_all.tolist()
    if args.smoke and args.zone_ids is None:
        requested_ids = [120, 79]
    zones = [int(np.where(zone_ids_all == location_id)[0][0]) for location_id in requested_ids]
    training_times = np.asarray(split["indices"]["train"], dtype=int)
    pools_z = history_pools(prepared["flow_z"].astype(np.float32), training_times)
    pools_raw = history_pools(data["flow"].astype(np.float32), training_times)
    total = len(zones) + len(zones) * len(args.states) * len(args.model_seeds)
    current = 0
    write_status(phase="interventions", current=current, total=total, started=started)

    try:
        global_intervention = prepare_global_intervention(
            pools_z, pools_raw, samples=args.samples, components=args.components, seed=41
        )
        bar = tqdm(total=total, desc="finite quadratic TM", unit="zone-stage", mininterval=1.0)
        for zone, location_id in zip(zones, requested_ids):
            prepare_zone_intervention(
                zone, location_id, pools_z, pools_raw, global_intervention,
                samples=args.samples, components=args.components, seed=73,
            )
            current += 1
            bar.update(1)
            write_status(
                phase="interventions", current=current, total=total, started=started,
                metrics={"zone_id": location_id, "zone_name": str(zone_names[zone])},
            )

        device = ei.choose_device(args.device)
        unit_paths = []
        for model_seed in args.model_seeds:
            model, _ = ei.load_model(training, data, prepared["attributes_z"].shape[1], model_seed, device)
            noise_covariance, noise_meta = ei.residual_covariance(training, model, prepared, split, device)
            centers = ei.select_centers(training, data, prepared, split, args.states)
            center_by_name = {
                item["name"]: {key: value for key, value in item["batch"].items() if key != "target"}
                for item in centers
            }
            for state in args.states:
                center = center_by_name[state]
                for zone, location_id in zip(zones, requested_ids):
                    path = unit_path(args.samples, args.components, args.ridge, model_seed, state, location_id)
                    unit_paths.append(path)
                    if path.exists():
                        current += 1
                        bar.update(1)
                        continue
                    saved = np.load(intervention_cache_path(args.samples, args.components, location_id), allow_pickle=False)
                    indices = {key: saved[key] for key in (
                        "recent_indices", "macro_indices", "own_indices", "external_indices"
                    )}
                    response_cache = response_path(args.samples, model_seed, state, location_id)
                    if response_cache.exists():
                        response = np.load(response_cache, allow_pickle=False)
                        target_temporal = response["target_temporal"]
                        target_spatial = response["target_spatial"]
                    else:
                        target_temporal = predict_intervention(
                            model, center, zone, indices, pools_z, kind="temporal",
                            device=device, batch_size=args.batch_size,
                        )
                        target_spatial = predict_intervention(
                            model, center, zone, indices, pools_z, kind="spatial",
                            device=device, batch_size=args.batch_size,
                        )
                        rows = np.arange(2 * zone, 2 * zone + 2)
                        covariance = noise_covariance[np.ix_(rows, rows)]
                        noise = np.random.default_rng(
                            100_000 + model_seed * 1000 + location_id
                        ).multivariate_normal(np.zeros(2), covariance, size=args.samples)
                        target_temporal += noise
                        target_spatial += noise
                        response_cache.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            response_cache,
                            target_temporal=target_temporal.astype(np.float32),
                            target_spatial=target_spatial.astype(np.float32),
                        )
                    fit_indices = saved["fit_indices"].astype(int)
                    evaluate_indices = saved["evaluate_indices"].astype(int)
                    record = {
                        "zone_id": int(location_id),
                        "zone_name": str(zone_names[zone]),
                        "state": state,
                        "model_seed": int(model_seed),
                        "samples": int(args.samples),
                        "noise_metadata": noise_meta,
                    }
                    for method in ("baseline", "hurdle"):
                        record[f"{method}_temporal"] = estimate_quadratic_cmi(
                            saved[f"temporal_recent_{method}"], saved[f"temporal_macro_{method}"],
                            target_temporal, fit_indices, evaluate_indices, ridge=args.ridge,
                        )
                        record[f"{method}_spatial"] = estimate_quadratic_cmi(
                            saved[f"spatial_own_{method}"], saved[f"spatial_external_{method}"],
                            target_spatial, fit_indices, evaluate_indices, ridge=args.ridge,
                        )
                    atomic_json(path, record)
                    current += 1
                    bar.update(1)
                    bar.set_postfix(seed=model_seed, state=state, zone=location_id)
                    write_status(
                        phase="finite_tm", current=current, total=total, started=started,
                        metrics={"seed": model_seed, "state": state, "zone_id": location_id},
                    )
        bar.close()
        result = aggregate_results(
            unit_paths, samples=args.samples, components=args.components, ridge=args.ridge
        )
        suffix = "smoke" if args.smoke else "full"
        summary_path = RUN_DIR / f"quadratic_tm_{suffix}_summary.json"
        atomic_json(summary_path, result)
        if result["nonnegative_audit"]["hurdle"]["violation_count"]:
            audit = result["nonnegative_audit"]["hurdle"]
            raise RuntimeError(
                "formal hurdle quadratic TM Syn nonnegativity violation: "
                f"min={audit['minimum_raw_syn_bits']:.6g}, "
                f"threshold={-NONNEGATIVE_TOLERANCE_BITS}, count={audit['violation_count']}"
            )
        write_status(
            phase="complete", current=total, total=total, started=started,
            metrics={"summary": str(summary_path), "audit": result["nonnegative_audit"]},
        )
        print(json.dumps({"status": result["status"], "summary": result["summary"]}, ensure_ascii=False, indent=2))
    except Exception as error:
        write_status(
            phase="failed", current=current, total=total, started=started,
            message=str(error),
        )
        raise


if __name__ == "__main__":
    main()
