#!/usr/bin/env python3
"""Finite-amplitude polynomial-TM audit of MGSTN temporal synergy.

The pilot avoids infinitesimal Jacobians. Recent and macro-history blocks are
sampled independently from empirical training windows, passed through the
frozen MGSTN, and reduced separately to two PCA scores. Since the intervention
enforces R independent of M, PEID Syn equals I(R; M | Y). We estimate that CMI
directly with degree-1/2/3 polynomial triangular transport maps.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TRAIN_SCRIPT = ROOT / "scripts/train_nyc_taxi_mgstn.py"
EI_SCRIPT = ROOT / "scripts/compute_nyc_taxi_mgstn_ei.py"
DECOMPOSITION = ROOT / "results/nyc_taxi_mgstn_ei/temporal_coupling.npz"
OUTPUT = ROOT / "results/nyc_taxi_mgstn_ei/finite_polynomial_tm_pilot.json"
TOLERANCE_BITS = 0.05
LOG_2 = float(np.log(2.0))
ZONE_IDS = [120, 153, 128, 202, 79, 237, 234, 239, 125]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def empirical_windows(flow_z: np.ndarray, times: np.ndarray, zone: int) -> tuple[np.ndarray, np.ndarray]:
    recent = np.stack([flow_z[t - 7 : t, zone].T.reshape(-1) for t in times])
    daily = np.stack([flow_z[t - 24 * np.arange(5, 0, -1), zone].T.reshape(-1) for t in times])
    weekly = np.stack([flow_z[t - 168 * np.arange(7, 0, -1), zone].T.reshape(-1) for t in times])
    return recent.astype(np.float32), np.concatenate([daily, weekly], axis=1).astype(np.float32)


def project_block(train: np.ndarray, evaluate: np.ndarray, components: int = 2) -> tuple[np.ndarray, np.ndarray, list[float]]:
    mean = train.mean(axis=0, keepdims=True)
    centered = train - mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    width = min(int(components), vh.shape[0])
    basis = vh[:width].T
    energy = singular_values**2
    explained = energy[:width] / max(float(energy.sum()), 1e-12)
    return centered @ basis, (evaluate - mean) @ basis, explained.tolist()


def conditional_log_prob(model, samples: np.ndarray, start: int) -> np.ndarray:
    reference = model.map_to_reference(samples)
    selected = reference[:, start:]
    scales = model.residual_scales[start:]
    return -0.5 * np.sum(np.log(2.0 * np.pi) + selected**2, axis=1) - float(np.log(scales).sum())


def estimate_cmi_polynomial_tm(
    recent: np.ndarray,
    macro: np.ndarray,
    target: np.ndarray,
    *,
    degree: int,
    split_seed: int,
    components: int,
) -> dict[str, object]:
    from exp.TM.transport_map_density import fit_polynomial_triangular_transport_map_density

    rng = np.random.default_rng(split_seed)
    order = rng.permutation(len(target))
    fit_n = int(0.70 * len(order))
    fit, evaluate = order[:fit_n], order[fit_n:]
    recent_fit, recent_eval, recent_explained = project_block(
        recent[fit], recent[evaluate], components=components
    )
    macro_fit, macro_eval, macro_explained = project_block(
        macro[fit], macro[evaluate], components=components
    )

    full_fit = np.column_stack([target[fit], recent_fit, macro_fit])
    reduced_fit = np.column_stack([target[fit], macro_fit])
    full_eval = np.column_stack([target[evaluate], recent_eval, macro_eval])
    reduced_eval = np.column_stack([target[evaluate], macro_eval])
    full_model = fit_polynomial_triangular_transport_map_density(full_fit, degree=int(degree))
    reduced_model = fit_polynomial_triangular_transport_map_density(reduced_fit, degree=int(degree))

    # log p(M | Y,R) - log p(M | Y) = pointwise I(R;M | Y).
    pointwise = (
        conditional_log_prob(full_model, full_eval, start=2 + recent_fit.shape[1])
        - conditional_log_prob(reduced_model, reduced_eval, start=2)
    ) / LOG_2
    return {
        "degree": int(degree),
        "syn_bits_raw": float(pointwise.mean()),
        "syn_bits_se": float(pointwise.std(ddof=1) / np.sqrt(len(pointwise))),
        "evaluation_samples": int(len(pointwise)),
        "components_per_group": int(components),
        "recent_pca_explained": recent_explained,
        "macro_pca_explained": macro_explained,
    }


def model_response(
    model,
    center: dict[str, torch.Tensor],
    zone: int,
    recent: np.ndarray,
    macro: np.ndarray,
    noise_covariance: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    noise_seed: int,
) -> np.ndarray:
    count = len(recent)
    predictions = []
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            width = stop - start
            batch = {}
            for key, value in center.items():
                expanded = value.unsqueeze(0).expand(width, *value.shape).clone()
                batch[key] = expanded.to(device)
            batch["recent"][:, zone] = torch.from_numpy(recent[start:stop].reshape(width, 2, 7)).to(device)
            batch["daily"][:, zone] = torch.from_numpy(macro[start:stop, :10].reshape(width, 2, 5)).to(device)
            batch["weekly"][:, zone] = torch.from_numpy(macro[start:stop, 10:].reshape(width, 2, 7)).to(device)
            predictions.append(model(batch)[:, zone].cpu().numpy())
    prediction = np.concatenate(predictions).astype(np.float64)
    rows = np.arange(2 * zone, 2 * zone + 2)
    covariance = noise_covariance[np.ix_(rows, rows)]
    noise = np.random.default_rng(noise_seed).multivariate_normal(np.zeros(2), covariance, size=count)
    return prediction + noise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--state", default="weekday_peak")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--zone-ids", type=int, nargs="+", default=ZONE_IDS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.samples < 1000:
        raise ValueError("at least 1000 intervention samples are required")

    training = load_module(TRAIN_SCRIPT, "nyc_taxi_mgstn_training_finite_tm")
    ei = load_module(EI_SCRIPT, "nyc_taxi_mgstn_ei_finite_tm")
    data, prepared, split = training.load_data(False)
    device = ei.choose_device(args.device)
    model, _ = ei.load_model(training, data, prepared["attributes_z"].shape[1], args.model_seed, device)
    centers = ei.select_centers(training, data, prepared, split, [args.state])
    center = {key: value for key, value in centers[0]["batch"].items() if key != "target"}
    noise_covariance, noise_meta = ei.residual_covariance(training, model, prepared, split, device)
    training_times = np.asarray(split["indices"]["train"], dtype=int)
    temporal = np.load(DECOMPOSITION)
    affine_mask = (
        (temporal["center_names"].astype(str) == args.state)
        & (temporal["seeds"].astype(int) == args.model_seed)
    )
    if int(affine_mask.sum()) != 1:
        raise RuntimeError("unable to identify affine comparator")

    zone_ids = data["zone_ids"].astype(int)
    zone_names = data["zone_names"].astype(str)
    records = []
    for location_id in args.zone_ids:
        zone = int(np.where(zone_ids == location_id)[0][0])
        recent_pool, macro_pool = empirical_windows(prepared["flow_z"], training_times, zone)
        for replicate in range(args.replicates):
            rng = np.random.default_rng(10_000 + 101 * replicate + location_id)
            recent = recent_pool[rng.integers(0, len(recent_pool), size=args.samples)]
            macro = macro_pool[rng.integers(0, len(macro_pool), size=args.samples)]
            target = model_response(
                model,
                center,
                zone,
                recent,
                macro,
                noise_covariance,
                device=device,
                batch_size=args.batch_size,
                noise_seed=20_000 + 101 * replicate + location_id,
            )
            for degree in (1, 2, 3):
                estimate = estimate_cmi_polynomial_tm(
                    recent,
                    macro,
                    target,
                    degree=degree,
                    split_seed=30_000 + 101 * replicate + location_id,
                    components=args.components,
                )
                records.append({
                    "zone_id": int(location_id),
                    "zone_name": str(zone_names[zone]),
                    "replicate": int(replicate),
                    "local_affine_all_target_syn_bits": float(
                        temporal["hierarchical_coupling"][affine_mask, zone, 1][0]
                    ),
                    **estimate,
                })
        print(f"completed {location_id} {zone_names[zone]}", flush=True)

    raw = np.asarray([record["syn_bits_raw"] for record in records])
    below = raw < -TOLERANCE_BITS
    audit = {
        "tolerance_bits": TOLERANCE_BITS,
        "minimum_raw_syn_bits": float(raw.min()),
        "numerical_zero_count": int(np.sum((raw < 0) & ~below)),
        "violation_count": int(below.sum()),
    }
    summary = []
    for location_id in args.zone_ids:
        for degree in (1, 2, 3):
            selected = [
                record for record in records
                if record["zone_id"] == location_id and record["degree"] == degree
            ]
            values = np.asarray([record["syn_bits_raw"] for record in selected])
            summary.append({
                "zone_id": int(location_id),
                "zone_name": selected[0]["zone_name"],
                "degree": int(degree),
                "syn_bits_mean": float(values.mean()),
                "syn_bits_sd": float(values.std(ddof=1)),
                "local_affine_all_target_syn_bits": selected[0]["local_affine_all_target_syn_bits"],
            })
    payload = {
        "status": "invalid_nonnegative_violation" if audit["violation_count"] else "complete",
        "definition": "Syn = I(recent; macro | own-zone output), because empirical block interventions are independent",
        "intervention": "independent empirical training-window bootstrap for recent and joint daily-weekly macro blocks",
        "target": "own-zone standardized inflow/outflow plus checkpoint validation residual noise",
        "estimator": "held-out polynomial autoregressive triangular TM after separate 2-PC projections",
        "state": args.state,
        "model_seed": args.model_seed,
        "samples": args.samples,
        "replicates": args.replicates,
        "components_per_group": args.components,
        "noise_metadata": noise_meta,
        "nonnegative_audit": audit,
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if audit["violation_count"]:
        raise RuntimeError(
            "finite polynomial TM Syn nonnegativity violation: "
            f"min={audit['minimum_raw_syn_bits']:.6g}, threshold={-TOLERANCE_BITS}, "
            f"count={audit['violation_count']}"
        )
    print(json.dumps({"status": payload["status"], "audit": audit, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
