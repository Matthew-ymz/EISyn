#!/usr/bin/env python3
"""Affine-TM PEID of the trained NYC Taxi MGSTN.

The nonlinear predictor is linearized at several fixed traffic contexts.  Each
local Jacobian, together with the checkpoint-specific validation residual
covariance, defines one Gaussian transition under independent unit-Gaussian
interventions on the standardized flow-history coordinates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_nyc_taxi_mgstn.py"
DATA_PATH = ROOT / "data" / "nyc_taxi_mgstn_2023" / "nyc_taxi_mgstn_hourly.npz"
CHECKPOINT_DIR = ROOT / "results" / "nyc_taxi_mgstn"
OUTPUT_DIR = ROOT / "results" / "nyc_taxi_mgstn_ei"
LOG_DIR = ROOT / "docs" / "log" / "nyc_taxi_mgstn"
PROGRESS = LOG_DIR / "ei_live_progress.json"
NONNEGATIVE_TOLERANCE_BITS = 1e-8
JITTER_RELATIVE = 1e-8
LN2 = float(np.log(2.0))
NODES = 66
TARGET_DIM = 132
BRANCH_LENGTHS = {"recent": 7, "daily": 5, "weekly": 7}


def load_training_module():
    spec = importlib.util.spec_from_file_location("nyc_taxi_mgstn_training", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def checkpoint_path(seed: int) -> Path:
    return CHECKPOINT_DIR / f"h96_tl2_gl2_seed_{seed}.pt"


def load_model(module, data: dict, attribute_dim: int, seed: int, device: torch.device):
    path = checkpoint_path(seed)
    state = torch.load(path, map_location=device, weights_only=False)
    config = module.Config(**state["config"])
    model = module.MGSTN(data, config, attribute_dim).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, config


def residual_covariance(module, model, prepared: dict, split_info: dict, device: torch.device) -> tuple[np.ndarray, dict]:
    dataset = module.TaxiWindows(
        prepared["flow_z"], prepared["attributes_z"], prepared["targets"], split_info["indices"]["valid"]
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    residuals = []
    with torch.no_grad():
        for batch in loader:
            moved = module.move(batch, device)
            prediction = model(moved)
            residuals.append((moved["target"] - prediction).cpu().numpy())
    residual = np.concatenate(residuals).reshape(-1, TARGET_DIM).astype(np.float64)
    estimator = LedoitWolf().fit(residual)
    covariance = np.asarray(estimator.covariance_, dtype=np.float64)
    return covariance, {
        "validation_samples": int(len(residual)),
        "ledoit_wolf_shrinkage": float(estimator.shrinkage_),
        "residual_rmse_z": float(np.sqrt(np.mean(residual**2))),
    }


def select_centers(module, data: dict, prepared: dict, split_info: dict, names: list[str]):
    valid_indices = np.asarray(split_info["indices"]["valid"], dtype=int)
    dataset = module.TaxiWindows(
        prepared["flow_z"], prepared["attributes_z"], prepared["targets"], valid_indices
    )
    times = pd.date_range("2023-01-01", "2024-01-01", freq="h", inclusive="left")
    flow = np.asarray(data["flow"], dtype=float)
    weather = np.asarray(data["weather"], dtype=float)
    choices: dict[str, int | None] = {"mean": None}

    def highest(mask: np.ndarray) -> int:
        candidates = valid_indices[mask[valid_indices]]
        if len(candidates) == 0:
            raise RuntimeError("representative-center mask selected no validation timestamp")
        totals = flow[candidates].sum(axis=(1, 2))
        return int(candidates[int(np.argmax(totals))])

    weekday_peak_mask = (times.dayofweek < 5) & np.isin(times.hour, [8, 9])
    weekend_mask = (times.dayofweek >= 5) & np.isin(times.hour, [12, 13, 14])
    rainy_mask = weather[:, 1] > 0.0
    choices["weekday_peak"] = highest(np.asarray(weekday_peak_mask))
    choices["weekend_midday"] = highest(np.asarray(weekend_mask))
    choices["rainy_high_demand"] = highest(np.asarray(rainy_mask))

    centers = []
    for name in names:
        if name not in choices:
            raise ValueError(f"unknown center {name!r}")
        timestamp_index = choices[name]
        if timestamp_index is None:
            center = {
                "recent": torch.zeros(NODES, 2, 7),
                "daily": torch.zeros(NODES, 2, 5),
                "weekly": torch.zeros(NODES, 2, 7),
                "recent_attr": torch.zeros(7, prepared["attributes_z"].shape[1]),
                "daily_attr": torch.zeros(5, prepared["attributes_z"].shape[1]),
                "weekly_attr": torch.zeros(7, prepared["attributes_z"].shape[1]),
                "target_attr": torch.zeros(prepared["attributes_z"].shape[1]),
            }
            timestamp = "standardized_mean"
        else:
            local_index = int(np.where(valid_indices == timestamp_index)[0][0])
            sample = dataset[local_index]
            center = {key: value.clone() for key, value in sample.items() if key != "target"}
            timestamp = str(times[timestamp_index])
        centers.append({"name": name, "timestamp_index": timestamp_index, "timestamp": timestamp, "batch": center})
    return centers


def flatten_flow(center: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([center[name].reshape(-1) for name in BRANCH_LENGTHS])


def source_columns() -> tuple[list[np.ndarray], dict[str, list[np.ndarray]]]:
    offsets = {}
    cursor = 0
    for branch, length in BRANCH_LENGTHS.items():
        offsets[branch] = cursor
        cursor += NODES * 2 * length
    region_columns = []
    granular_columns = {branch: [] for branch in BRANCH_LENGTHS}
    for node in range(NODES):
        pieces = []
        for branch, length in BRANCH_LENGTHS.items():
            start = offsets[branch] + node * 2 * length
            columns = np.arange(start, start + 2 * length, dtype=int)
            granular_columns[branch].append(columns)
            pieces.append(columns)
        region_columns.append(np.concatenate(pieces))
    if cursor != 2508:
        raise RuntimeError(f"unexpected MGSTN source dimension {cursor}")
    return region_columns, granular_columns


def model_function(model, center: dict[str, torch.Tensor], device: torch.device):
    lengths = BRANCH_LENGTHS
    attribute_batch = {
        key: value.unsqueeze(0).to(device)
        for key, value in center.items()
        if key.endswith("attr")
    }

    def function(flat: torch.Tensor) -> torch.Tensor:
        batch = dict(attribute_batch)
        cursor = 0
        for branch, length in lengths.items():
            width = NODES * 2 * length
            batch[branch] = flat[cursor : cursor + width].reshape(1, NODES, 2, length)
            cursor += width
        return model(batch).reshape(-1)

    return function


def jacobian_reverse_rows(function, flat: torch.Tensor, *, progress_callback=None) -> np.ndarray:
    source = flat.detach().clone().requires_grad_(True)
    output = function(source)
    rows = []
    for index in range(output.numel()):
        gradient = torch.autograd.grad(output[index], source, retain_graph=index + 1 < output.numel())[0]
        rows.append(gradient.detach().cpu())
        if progress_callback is not None and (index % 8 == 0 or index + 1 == output.numel()):
            progress_callback(index + 1, output.numel())
    return torch.stack(rows).numpy().astype(np.float64)


def stabilized_logdet(matrix: np.ndarray) -> tuple[float, float]:
    symmetric = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T)
    scale = max(float(np.trace(symmetric) / symmetric.shape[0]), 1e-12)
    jitter = JITTER_RELATIVE * scale
    sign, value = np.linalg.slogdet(symmetric + jitter * np.eye(symmetric.shape[0]))
    if sign <= 0:
        minimum = float(np.linalg.eigvalsh(symmetric).min())
        raise RuntimeError(f"covariance is not positive definite after fixed jitter; min_eigenvalue={minimum}")
    return float(value), float(jitter)


def gaussian_ei(total_cov: np.ndarray, conditional_cov: np.ndarray) -> tuple[float, float]:
    total_logdet, total_jitter = stabilized_logdet(total_cov)
    conditional_logdet, conditional_jitter = stabilized_logdet(conditional_cov)
    value = 0.5 * (total_logdet - conditional_logdet) / LN2
    return float(value), max(total_jitter, conditional_jitter)


def assert_nonnegative(values: np.ndarray, label: str) -> dict:
    array = np.asarray(values, dtype=float)
    affected = int(np.sum((array < 0.0) & (array >= -NONNEGATIVE_TOLERANCE_BITS)))
    violations = array < -NONNEGATIVE_TOLERANCE_BITS
    if bool(np.any(violations)):
        raise RuntimeError(
            f"{label} nonnegativity violation: min={array.min():.12g} bits, "
            f"threshold={-NONNEGATIVE_TOLERANCE_BITS:.12g}, affected={int(np.sum(violations))}"
        )
    return {
        "label": label,
        "tolerance_bits": NONNEGATIVE_TOLERANCE_BITS,
        "numerical_zero_count": affected,
        "minimum_bits": float(array.min()),
    }


def decompose_jacobian(jacobian: np.ndarray, noise_cov: np.ndarray, region_cols, granular_cols, pairs):
    total_cov = noise_cov + jacobian @ jacobian.T
    joint_ei, max_jitter = gaussian_ei(total_cov, noise_cov)
    region_ei = np.empty(NODES, dtype=float)
    granular_ei = np.empty((NODES, 3), dtype=float)
    branch_names = list(BRANCH_LENGTHS)
    for node, columns in enumerate(region_cols):
        contribution = jacobian[:, columns] @ jacobian[:, columns].T
        region_ei[node], jitter = gaussian_ei(total_cov, total_cov - contribution)
        max_jitter = max(max_jitter, jitter)
        for branch_index, branch in enumerate(branch_names):
            part_cols = granular_cols[branch][node]
            part = jacobian[:, part_cols] @ jacobian[:, part_cols].T
            granular_ei[node, branch_index], jitter = gaussian_ei(total_cov, total_cov - part)
            max_jitter = max(max_jitter, jitter)

    within_region = region_ei - granular_ei.sum(axis=1)
    cross_region = float(joint_ei - region_ei.sum())
    fine_xi = float(joint_ei - granular_ei.sum())

    target_joint = np.empty(NODES, dtype=float)
    target_cross = np.empty(NODES, dtype=float)
    target_region_ei = np.empty((NODES, NODES), dtype=float)
    pair_syn = np.empty((NODES, len(pairs)), dtype=np.float32)
    for target in range(NODES):
        rows = np.arange(2 * target, 2 * target + 2)
        local_j = jacobian[rows]
        local_noise = noise_cov[np.ix_(rows, rows)]
        local_total = local_noise + local_j @ local_j.T
        target_joint[target], jitter = gaussian_ei(local_total, local_noise)
        max_jitter = max(max_jitter, jitter)
        contributions = np.empty((NODES, 2, 2), dtype=float)
        for source, columns in enumerate(region_cols):
            block = local_j[:, columns]
            contributions[source] = block @ block.T
            target_region_ei[target, source], jitter = gaussian_ei(
                local_total, local_total - contributions[source]
            )
            max_jitter = max(max_jitter, jitter)
        target_cross[target] = target_joint[target] - target_region_ei[target].sum()
        for pair_index, (left, right) in enumerate(pairs):
            joint_pair, jitter = gaussian_ei(
                local_total, local_total - contributions[left] - contributions[right]
            )
            max_jitter = max(max_jitter, jitter)
            pair_syn[target, pair_index] = joint_pair - target_region_ei[target, left] - target_region_ei[target, right]

    audits = [
        assert_nonnegative(np.asarray([cross_region]), "system cross-region Xi"),
        assert_nonnegative(within_region, "within-region Xi"),
        assert_nonnegative(np.asarray([fine_xi]), "system fine Xi"),
        assert_nonnegative(target_cross, "target-wise cross-region Xi"),
        assert_nonnegative(pair_syn, "target-wise pair Syn"),
    ]
    return {
        "joint_ei": joint_ei,
        "cross_region_xi": cross_region,
        "within_region_xi": float(within_region.sum()),
        "fine_xi": fine_xi,
        "decomposition_identity_error": float(fine_xi - cross_region - within_region.sum()),
        "region_ei": region_ei,
        "granular_ei": granular_ei,
        "within_region_by_region": within_region,
        "target_joint_ei": target_joint,
        "target_cross_region_xi": target_cross,
        "target_region_ei": target_region_ei,
        "target_pair_syn": pair_syn,
        "maximum_covariance_jitter": max_jitter,
        "nonnegative_audits": audits,
    }


def summarize(units: list[dict], data: dict, pairs: list[tuple[int, int]], center_metadata: list[dict], residual_meta: list[dict]):
    ddof = 1 if len(units) > 1 else 0
    scalar_names = ["joint_ei", "cross_region_xi", "within_region_xi", "fine_xi"]
    scalar_summary = {}
    for name in scalar_names:
        values = np.asarray([unit[name] for unit in units], dtype=float)
        scalar_summary[name] = {
            "mean_bits": float(values.mean()),
            "sd_bits": float(values.std(ddof=ddof)),
            "min_bits": float(values.min()),
            "max_bits": float(values.max()),
        }
    scalar_summary["cross_region_fraction_of_fine_xi"] = {
        "mean": float(np.mean([unit["cross_region_xi"] / unit["fine_xi"] for unit in units])),
        "sd": float(np.std([unit["cross_region_xi"] / unit["fine_xi"] for unit in units], ddof=ddof)),
    }
    zone_ids = np.asarray(data["zone_ids"], dtype=int)
    zone_names = np.asarray(data["zone_names"]).astype(str)
    target_values = np.stack([unit["target_cross_region_xi"] for unit in units])
    target_means = target_values.mean(axis=0)
    target_sds = target_values.std(axis=0, ddof=ddof)
    target_order = np.argsort(target_means)[::-1]
    top_targets = [
        {
            "rank": rank + 1,
            "zone_id": int(zone_ids[index]),
            "zone_name": str(zone_names[index]),
            "cross_region_xi_mean_bits": float(target_means[index]),
            "cross_region_xi_sd_bits": float(target_sds[index]),
        }
        for rank, index in enumerate(target_order[:15])
    ]

    pair_values = np.stack([unit["target_pair_syn"] for unit in units])
    pair_means = pair_values.mean(axis=0)
    pair_sds = pair_values.std(axis=0, ddof=ddof)
    flattened_order = np.argsort(pair_means.ravel())[::-1]
    top_pairs = []
    for flat_index in flattened_order[:30]:
        target, pair_index = np.unravel_index(flat_index, pair_means.shape)
        left, right = pairs[pair_index]
        top_pairs.append(
            {
                "target_zone_id": int(zone_ids[target]),
                "target_zone_name": str(zone_names[target]),
                "source_left_zone_id": int(zone_ids[left]),
                "source_left_zone_name": str(zone_names[left]),
                "source_right_zone_id": int(zone_ids[right]),
                "source_right_zone_name": str(zone_names[right]),
                "pair_syn_mean_bits": float(pair_means[target, pair_index]),
                "pair_syn_sd_bits": float(pair_sds[target, pair_index]),
            }
        )

    region_ei_values = np.stack([unit["region_ei"] for unit in units])
    source_means = region_ei_values.mean(axis=0)
    source_sds = region_ei_values.std(axis=0, ddof=ddof)
    source_order = np.argsort(source_means)[::-1]
    top_sources = [
        {
            "rank": rank + 1,
            "zone_id": int(zone_ids[index]),
            "zone_name": str(zone_names[index]),
            "region_ei_mean_bits": float(source_means[index]),
            "region_ei_sd_bits": float(source_sds[index]),
        }
        for rank, index in enumerate(source_order[:15])
    ]
    audits = [audit for unit in units for audit in unit["nonnegative_audits"]]
    return {
        "contract": {
            "estimator": "local affine triangular transport map / Gaussian log-determinant",
            "intervention": "independent unit-Gaussian perturbation on 2508 standardized flow-history coordinates",
            "source_partition": "66 region histories; refined into recent/daily/weekly within each region",
            "target": "132D next-hour standardized inflow/outflow",
            "background_attributes": "fixed at each analysis center",
            "nonnegative_tolerance_bits": NONNEGATIVE_TOLERANCE_BITS,
            "covariance_jitter_relative": JITTER_RELATIVE,
            "zotero_peid_item_key": "MYATYWAJ",
            "evidence_level": "全文 26/26 pages",
            "limitation": "local affine readout; does not capture far-from-center nonlinear curvature or XOR-like effects",
        },
        "units": len(units),
        "seeds": sorted({int(unit["seed"]) for unit in units}),
        "centers": center_metadata,
        "residual_covariance": residual_meta,
        "system": scalar_summary,
        "top_target_receivers": top_targets,
        "top_region_sources": top_sources,
        "top_targetwise_pairs": top_pairs,
        "nonnegative_audit": {
            "minimum_bits": float(min(audit["minimum_bits"] for audit in audits)),
            "numerical_zero_count": int(sum(audit["numerical_zero_count"] for audit in audits)),
            "violations": 0,
            "tolerance_bits": NONNEGATIVE_TOLERANCE_BITS,
        },
        "decomposition_identity_audit": {
            "identity": "fine Xi = cross-region Xi + sum(within-region Xi)",
            "maximum_absolute_error_bits": float(
                max(abs(unit["decomposition_identity_error"]) for unit in units)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--centers",
        nargs="+",
        default=["mean", "weekday_peak", "weekend_midday", "rainy_high_demand"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = args.seeds[:1]
        args.centers = args.centers[:1]
    started = time.monotonic()
    device = choose_device(args.device)
    module = load_training_module()
    data, prepared, split_info = module.load_data(False)
    centers = select_centers(module, data, prepared, split_info, list(args.centers))
    region_cols, granular_cols = source_columns()
    pairs = list(combinations(range(NODES), 2))
    total_jacobians = len(args.seeds) * len(centers)
    completed_jacobians = 0
    units = []
    jacobians = []
    residual_covariances = []
    residual_metadata = []
    center_metadata = [
        {key: value for key, value in center.items() if key != "batch"} for center in centers
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        model, config = load_model(module, data, prepared["attributes_z"].shape[1], seed, device)
        noise_cov, residual_meta = residual_covariance(module, model, prepared, split_info, device)
        residual_covariances.append(noise_cov)
        residual_metadata.append({"seed": int(seed), **residual_meta})
        for center in centers:
            function = model_function(model, center["batch"], device)
            flat = flatten_flow(center["batch"]).to(device)

            def progress_callback(row, rows):
                atomic_json(
                    PROGRESS,
                    {
                        "phase": "jacobian",
                        "current": completed_jacobians * rows + row,
                        "total": total_jacobians * rows,
                        "unit": "output-gradient",
                        "elapsed_seconds": time.monotonic() - started,
                        "metrics": {"seed": seed, "center": center["name"], "row": row},
                        "updated_at": time.time(),
                    },
                )

            jacobian = jacobian_reverse_rows(function, flat, progress_callback=progress_callback)
            jacobians.append(jacobian.astype(np.float32))
            result = decompose_jacobian(jacobian, noise_cov, region_cols, granular_cols, pairs)
            result.update({"seed": int(seed), "center": center["name"], "timestamp": center["timestamp"]})
            units.append(result)
            completed_jacobians += 1
            print(
                f"seed={seed} center={center['name']} joint={result['joint_ei']:.6f} "
                f"cross={result['cross_region_xi']:.6f} within={result['within_region_xi']:.6f}",
                flush=True,
            )

    summary = summarize(units, data, pairs, center_metadata, residual_metadata)
    arrays = {
        "jacobians": np.stack(jacobians),
        "noise_covariances": np.stack(residual_covariances),
        "seeds": np.asarray([unit["seed"] for unit in units], dtype=int),
        "center_names": np.asarray([unit["center"] for unit in units]),
        "joint_ei": np.asarray([unit["joint_ei"] for unit in units]),
        "cross_region_xi": np.asarray([unit["cross_region_xi"] for unit in units]),
        "within_region_xi": np.asarray([unit["within_region_xi"] for unit in units]),
        "fine_xi": np.asarray([unit["fine_xi"] for unit in units]),
        "region_ei": np.stack([unit["region_ei"] for unit in units]),
        "granular_ei": np.stack([unit["granular_ei"] for unit in units]),
        "within_region_by_region": np.stack([unit["within_region_by_region"] for unit in units]),
        "target_joint_ei": np.stack([unit["target_joint_ei"] for unit in units]),
        "target_cross_region_xi": np.stack([unit["target_cross_region_xi"] for unit in units]),
        "target_region_ei": np.stack([unit["target_region_ei"] for unit in units]),
        "target_pair_syn": np.stack([unit["target_pair_syn"] for unit in units]),
        "pair_indices": np.asarray(pairs, dtype=int),
        "zone_ids": np.asarray(data["zone_ids"], dtype=int),
        "zone_names": np.asarray(data["zone_names"]),
    }
    stem = "smoke" if args.smoke else "full"
    np.savez_compressed(OUTPUT_DIR / f"{stem}_decomposition.npz", **arrays)
    atomic_json(OUTPUT_DIR / f"{stem}_summary.json", summary)
    atomic_json(
        PROGRESS,
        {
            "phase": "complete",
            "current": total_jacobians,
            "total": total_jacobians,
            "unit": "jacobian",
            "elapsed_seconds": time.monotonic() - started,
            "metrics": {
                "joint_ei_mean": summary["system"]["joint_ei"]["mean_bits"],
                "cross_region_xi_mean": summary["system"]["cross_region_xi"]["mean_bits"],
                "within_region_xi_mean": summary["system"]["within_region_xi"]["mean_bits"],
            },
            "updated_at": time.time(),
        },
    )
    print(json.dumps(summary["system"], indent=2), flush=True)


if __name__ == "__main__":
    main()
