#!/usr/bin/env python3
"""Refine MGSTN PEID into interpretable temporal-scale couplings.

The hierarchy is fixed before inspection:
1. daily + weekly -> macro-rhythm coupling;
2. recent + (daily, weekly) -> micro-macro coupling.

Both terms are two-block PEID Syn estimates under the same full intervention.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/nyc_taxi_mgstn_ei/full_decomposition.npz"
OUTPUT_DIR = ROOT / "results/nyc_taxi_mgstn_ei"
TOLERANCE_BITS = 1e-8
JITTER_RELATIVE = 1e-8
LN2 = np.log(2.0)


def load_ei_module():
    path = ROOT / "scripts/compute_nyc_taxi_mgstn_ei.py"
    spec = importlib.util.spec_from_file_location("nyc_taxi_mgstn_ei", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def subset_ei(block: np.ndarray, solved_block: np.ndarray, indices: np.ndarray) -> float:
    local = np.eye(len(indices)) - block[:, indices].T @ solved_block[:, indices]
    local = 0.5 * (local + local.T)
    sign, logdet = np.linalg.slogdet(local)
    if sign <= 0:
        raise RuntimeError("conditional determinant is non-positive in temporal coupling readout")
    return float(-0.5 * logdet / LN2)


def audit_nonnegative(name: str, values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    violations = values < -TOLERANCE_BITS
    if np.any(violations):
        raise RuntimeError(
            f"{name}: nonnegativity violation; min={values.min():.12g}, "
            f"threshold={-TOLERANCE_BITS:.12g}, count={violations.sum()}"
        )
    return {
        "name": name,
        "minimum_bits": float(values.min()),
        "numerical_zero_count": int(np.sum((values < 0) & (values >= -TOLERANCE_BITS))),
        "violation_count": 0,
        "tolerance_bits": TOLERANCE_BITS,
    }


def main():
    saved = np.load(INPUT)
    module = load_ei_module()
    _, granular_columns = module.source_columns()
    branches = list(module.BRANCH_LENGTHS)
    if branches != ["recent", "daily", "weekly"]:
        raise RuntimeError(f"unexpected branch order: {branches}")

    units = len(saved["seeds"])
    regions = len(saved["zone_ids"])
    hierarchical = np.empty((units, regions, 2), dtype=np.float64)
    pairwise = np.empty((units, regions, 3), dtype=np.float64)
    total_temporal = np.empty((units, regions), dtype=np.float64)

    for unit, seed in enumerate(saved["seeds"]):
        jacobian = saved["jacobians"][unit].astype(np.float64)
        total_cov = saved["noise_covariances"][int(seed)] + jacobian @ jacobian.T
        scale = max(float(np.trace(total_cov) / total_cov.shape[0]), 1e-12)
        factor = cho_factor(
            total_cov + JITTER_RELATIVE * scale * np.eye(total_cov.shape[0]),
            lower=True,
            check_finite=False,
        )
        for region in range(regions):
            groups = [granular_columns[name][region] for name in branches]
            columns = np.concatenate(groups)
            block = jacobian[:, columns]
            solved = cho_solve(factor, block, check_finite=False)
            lengths = [len(group) for group in groups]
            stops = np.cumsum([0, *lengths])
            index = [np.arange(stops[i], stops[i + 1]) for i in range(3)]
            single = np.asarray([subset_ei(block, solved, idx) for idx in index])
            rd = subset_ei(block, solved, np.r_[index[0], index[1]]) - single[0] - single[1]
            rw = subset_ei(block, solved, np.r_[index[0], index[2]]) - single[0] - single[2]
            dw_ei = subset_ei(block, solved, np.r_[index[1], index[2]])
            dw = dw_ei - single[1] - single[2]
            all_ei = subset_ei(block, solved, np.arange(len(columns)))
            micro_macro = all_ei - single[0] - dw_ei
            pairwise[unit, region] = (rd, rw, dw)
            hierarchical[unit, region] = (dw, micro_macro)
            total_temporal[unit, region] = all_ei - single.sum()

    identity_error = total_temporal - hierarchical.sum(axis=2)
    audits = [
        audit_nonnegative("daily-weekly macro-rhythm Syn", hierarchical[:, :, 0]),
        audit_nonnegative("recent versus macro-rhythm Syn", hierarchical[:, :, 1]),
        audit_nonnegative("recent-daily pair Syn", pairwise[:, :, 0]),
        audit_nonnegative("recent-weekly pair Syn", pairwise[:, :, 1]),
        audit_nonnegative("daily-weekly pair Syn", pairwise[:, :, 2]),
        audit_nonnegative("total within-region temporal Syn", total_temporal),
    ]
    if float(np.max(np.abs(identity_error))) > 1e-10:
        raise RuntimeError(f"hierarchical identity failure: {np.max(np.abs(identity_error))}")

    center_names = saved["center_names"].astype(str)
    state_summary = {}
    for center in np.unique(center_names):
        mask = center_names == center
        state_values = hierarchical[mask].sum(axis=1)
        shares = state_values / state_values.sum(axis=1, keepdims=True)
        state_summary[center] = {
            "daily_weekly_share_mean": float(shares[:, 0].mean()),
            "daily_weekly_share_sd": float(shares[:, 0].std(ddof=1)),
            "recent_macro_share_mean": float(shares[:, 1].mean()),
            "recent_macro_share_sd": float(shares[:, 1].std(ddof=1)),
        }

    real = center_names != "mean"
    observed = hierarchical[real].sum(axis=1)
    observed_shares = observed / observed.sum(axis=1, keepdims=True)
    summary = {
        "definition": {
            "hierarchy": [
                "daily + weekly -> macro-rhythm Syn",
                "recent + macro-rhythm -> micro-macro Syn",
            ],
            "identity": "within-region temporal Syn = macro-rhythm Syn + micro-macro Syn",
            "intervention": "same independent unit-Gaussian 2508D flow intervention as full MGSTN decomposition",
            "estimator": "local affine triangular TM with fixed relative covariance jitter",
        },
        "observed_states": {
            "units": int(real.sum()),
            "daily_weekly_share_mean": float(observed_shares[:, 0].mean()),
            "daily_weekly_share_sd": float(observed_shares[:, 0].std(ddof=1)),
            "recent_macro_share_mean": float(observed_shares[:, 1].mean()),
            "recent_macro_share_sd": float(observed_shares[:, 1].std(ddof=1)),
        },
        "by_state": state_summary,
        "nonnegative_audit": audits,
        "maximum_identity_error_bits": float(np.max(np.abs(identity_error))),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "temporal_coupling.npz",
        hierarchical_coupling=hierarchical,
        pairwise_coupling=pairwise,
        total_temporal_syn=total_temporal,
        identity_error=identity_error,
        seeds=saved["seeds"],
        center_names=saved["center_names"],
        zone_ids=saved["zone_ids"],
        zone_names=saved["zone_names"],
        hierarchical_labels=np.asarray(["Daily–weekly macro rhythm", "Recent–macro rhythm"]),
        pairwise_labels=np.asarray(["Recent–daily", "Recent–weekly", "Daily–weekly"]),
    )
    (OUTPUT_DIR / "temporal_coupling_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
