#!/usr/bin/env python3
"""Compute a grouped Gaussian-TM Xi decomposition for NYC Taxi Ridge.

The intervention distribution is N(0, I) over the standardized lag variables.
Each spatial source is one taxi zone containing all seven lags.  Calendar
features are conditioned background variables and are excluded from Xi.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nyc_taxi_synergy_model_screen import DATA_CACHE, LAGS, make_design, metrics


OUTPUT = ROOT / "results" / "nyc_taxi_global_ridge_xi.json"
ALPHA = 100.0
COVARIANCE_RIDGE = 1.0e-6
XI_TOLERANCE_BITS = 1.0e-10


def safe_logdet_psd(matrix: np.ndarray, floor: float = 1.0e-12) -> float:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    return float(np.log(np.maximum(eigenvalues, floor)).sum())


def affine_tm_ei_bits(
    transition: np.ndarray,
    noise_covariance: np.ndarray,
    source_indices: np.ndarray,
) -> float:
    """I(X_S;Y) under one shared full intervention X~N(0,I), in bits.

    Sources outside S remain active and are marginalized.  Hence they enter
    Cov(Y|X_S); deleting them would define a different intervention system and
    would invalidate the PEID nonnegativity identity.
    """
    trans = np.asarray(transition, dtype=float)
    selected_set = set(np.asarray(source_indices, dtype=int).tolist())
    complement = np.array(
        [index for index in range(trans.shape[1]) if index not in selected_set],
        dtype=int,
    )
    noise = np.asarray(noise_covariance, dtype=float)
    target_covariance = noise + trans @ trans.T
    conditional_target_covariance = noise.copy()
    if complement.size:
        omitted = trans[:, complement]
        conditional_target_covariance = conditional_target_covariance + omitted @ omitted.T
    return float(
        0.5
        * (
            safe_logdet_psd(target_covariance)
            - safe_logdet_psd(conditional_target_covariance)
        )
        / math.log(2.0)
    )


def scalar_target_ei_bits(
    transition_row: np.ndarray,
    noise_variance: float,
    source_indices: np.ndarray,
) -> float:
    row = np.asarray(transition_row, dtype=float)
    selected_set = set(np.asarray(source_indices, dtype=int).tolist())
    complement = np.array(
        [index for index in range(row.size) if index not in selected_set], dtype=int
    )
    full_variance = float(noise_variance + row @ row)
    conditional_variance = float(noise_variance)
    if complement.size:
        conditional_variance += float(row[complement] @ row[complement])
    return float(0.5 * math.log2(full_variance / conditional_variance))


def signed_diagnostics(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum_bits": float(array.min(initial=0.0)),
        "negative_within_tolerance_count": int(
            np.count_nonzero((array < 0.0) & (array >= -XI_TOLERANCE_BITS))
        ),
        "below_negative_tolerance_count": int(np.count_nonzero(array < -XI_TOLERANCE_BITS)),
    }


def records(zone_ids: np.ndarray, values: np.ndarray, key: str) -> list[dict[str, float | int]]:
    return [
        {"zone_id": int(zone_id), key: float(value)}
        for zone_id, value in zip(zone_ids, np.asarray(values, dtype=float), strict=True)
    ]


def main() -> None:
    cached = np.load(DATA_CACHE, allow_pickle=False)
    counts = cached["counts"]
    zone_ids = cached["zone_ids"]
    design = make_design(counts, zone_ids, smoke=False)

    model = Ridge(alpha=ALPHA)
    model.fit(design.x[design.train_mask], design.y[design.train_mask])
    prediction = model.predict(design.x)

    n_zones = len(zone_ids)
    n_lag_sources = len(LAGS) * n_zones
    transition = np.asarray(model.coef_[:, :n_lag_sources], dtype=float)
    train_residual = design.y[design.train_mask] - prediction[design.train_mask]
    empirical_noise_covariance = np.cov(train_residual, rowvar=False, bias=False)
    noise_covariance = (
        0.5 * (empirical_noise_covariance + empirical_noise_covariance.T)
        + COVARIANCE_RIDGE * np.eye(n_zones)
    )

    all_indices = np.arange(n_lag_sources)
    scalar_ei = np.array(
        [affine_tm_ei_bits(transition, noise_covariance, np.array([index])) for index in all_indices]
    )
    region_indices = [
        np.array([lag_index * n_zones + zone for lag_index in range(len(LAGS))])
        for zone in range(n_zones)
    ]
    region_ei = np.array(
        [affine_tm_ei_bits(transition, noise_covariance, indices) for indices in region_indices]
    )
    joint_ei = affine_tm_ei_bits(transition, noise_covariance, all_indices)

    scalar_sum = float(scalar_ei.sum())
    region_sum = float(region_ei.sum())
    within_region_xi = np.array(
        [region_ei[zone] - scalar_ei[indices].sum() for zone, indices in enumerate(region_indices)]
    )
    within_region_xi_sum = float(within_region_xi.sum())
    cross_region_xi = float(joint_ei - region_sum)
    system_xi = float(joint_ei - scalar_sum)
    identity_error = float(system_xi - (cross_region_xi + within_region_xi_sum))

    full_target_covariance = noise_covariance + transition @ transition.T
    posterior_source_covariance = np.eye(n_lag_sources) - (
        transition.T @ np.linalg.solve(full_target_covariance, transition)
    )
    posterior_source_covariance = 0.5 * (
        posterior_source_covariance + posterior_source_covariance.T
    )
    posterior_logdet = safe_logdet_psd(posterior_source_covariance)
    posterior_scalar_logdet_sum = float(
        np.log(np.maximum(np.diag(posterior_source_covariance), 1.0e-12)).sum()
    )
    posterior_region_logdet_sum = float(
        sum(
            safe_logdet_psd(posterior_source_covariance[np.ix_(indices, indices)])
            for indices in region_indices
        )
    )
    system_xi_from_conditional_tc = float(
        0.5 * (posterior_scalar_logdet_sum - posterior_logdet) / math.log(2.0)
    )
    cross_region_xi_from_conditional_tc = float(
        0.5 * (posterior_region_logdet_sum - posterior_logdet) / math.log(2.0)
    )
    within_region_xi_from_conditional_tc = float(
        0.5
        * (posterior_scalar_logdet_sum - posterior_region_logdet_sum)
        / math.log(2.0)
    )

    lag_ei_sums = {
        str(lag): float(scalar_ei[lag_index * n_zones : (lag_index + 1) * n_zones].sum())
        for lag_index, lag in enumerate(LAGS)
    }

    target_joint_ei = np.empty(n_zones)
    target_region_sum = np.empty(n_zones)
    target_scalar_sum = np.empty(n_zones)
    for target in range(n_zones):
        row = transition[target]
        noise_variance = float(noise_covariance[target, target])
        target_joint_ei[target] = scalar_target_ei_bits(row, noise_variance, all_indices)
        target_region_sum[target] = sum(
            scalar_target_ei_bits(row, noise_variance, indices) for indices in region_indices
        )
        target_scalar_sum[target] = sum(
            scalar_target_ei_bits(row, noise_variance, np.array([index])) for index in all_indices
        )
    target_cross_region_xi = target_joint_ei - target_region_sum
    target_within_region_xi = target_region_sum - target_scalar_sum
    target_system_xi = target_joint_ei - target_scalar_sum

    nonnegative_arrays = {
        "system_xi": np.array([system_xi]),
        "cross_region_xi": np.array([cross_region_xi]),
        "within_region_xi": within_region_xi,
        "target_cross_region_xi": target_cross_region_xi,
        "target_within_region_xi": target_within_region_xi,
        "target_system_xi": target_system_xi,
    }
    for label, values in nonnegative_arrays.items():
        violations = np.asarray(values, dtype=float) < -XI_TOLERANCE_BITS
        if np.any(violations):
            raise ValueError(
                f"{label} nonnegativity violation: minimum={float(np.min(values)):.12g} bits, "
                f"threshold={-XI_TOLERANCE_BITS:.1e}, count={int(np.count_nonzero(violations))}."
            )

    covariance_ridge_sensitivity = []
    symmetric_empirical_noise = 0.5 * (
        empirical_noise_covariance + empirical_noise_covariance.T
    )
    for covariance_ridge in (1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4):
        sensitivity_noise = symmetric_empirical_noise + covariance_ridge * np.eye(n_zones)
        sensitivity_joint = affine_tm_ei_bits(transition, sensitivity_noise, all_indices)
        sensitivity_region_sum = sum(
            affine_tm_ei_bits(transition, sensitivity_noise, indices)
            for indices in region_indices
        )
        sensitivity_scalar_sum = sum(
            affine_tm_ei_bits(transition, sensitivity_noise, np.array([index]))
            for index in all_indices
        )
        covariance_ridge_sensitivity.append(
            {
                "covariance_ridge": covariance_ridge,
                "joint_ei_bits": sensitivity_joint,
                "cross_region_xi_bits": float(sensitivity_joint - sensitivity_region_sum),
                "system_xi_bits": float(sensitivity_joint - sensitivity_scalar_sum),
            }
        )

    output = {
        "definition": {
            "system_xi": "EI(all 483 scalar lag sources; all 69 targets) - sum scalar-lag EI",
            "cross_region_xi": "EI(all 69 region-history groups; all targets) - sum region-history EI",
            "within_region_xi_sum": "sum_z [EI(all 7 lags of zone z; all targets) - sum_lag EI(zone z, lag)]",
            "identity": "system_xi = cross_region_xi + within_region_xi_sum",
            "estimator": "affine Gaussian transport-map/log-determinant with complement sources marginalized under one shared full intervention",
            "intervention": "independent standard normal over standardized lag coordinates",
            "source_group": "one Manhattan taxi zone with lags [1,2,3,6,12,48,336] half-hours",
            "target": "joint standardized log1p pickup demand in all 69 Manhattan zones at t",
            "calendar_treatment": "four calendar covariates held as background; excluded from source groups",
            "units": "bits",
        },
        "model": {
            "name": "Global Ridge",
            "alpha": ALPHA,
            "reason": "affine Xi-compatible surrogate for numerically best Interaction Ridge",
            "interaction_ridge_test_rmse": 0.537921,
            "global_ridge_test_metrics_recomputed": metrics(prediction, design, design.test_mask),
            "relative_rmse_gap_percent": float((0.538034 / 0.537921 - 1.0) * 100.0),
        },
        "dimensions": {
            "regions": n_zones,
            "lags_per_region": len(LAGS),
            "scalar_lag_sources": n_lag_sources,
            "targets": n_zones,
            "train_samples": int(design.train_mask.sum()),
        },
        "decomposition_bits": {
            "joint_ei": joint_ei,
            "scalar_singleton_ei_sum": scalar_sum,
            "region_group_ei_sum": region_sum,
            "system_xi": system_xi,
            "cross_region_xi": cross_region_xi,
            "within_region_xi_sum": within_region_xi_sum,
            "identity_error": identity_error,
            "cross_region_fraction_of_system_xi": float(cross_region_xi / system_xi),
            "within_region_fraction_of_system_xi": float(within_region_xi_sum / system_xi),
            "system_xi_fraction_of_joint_ei": float(system_xi / joint_ei),
        },
        "independent_conditional_tc_verification": {
            "system_xi_bits": system_xi_from_conditional_tc,
            "cross_region_xi_bits": cross_region_xi_from_conditional_tc,
            "within_region_xi_sum_bits": within_region_xi_from_conditional_tc,
            "system_abs_error_bits": abs(system_xi_from_conditional_tc - system_xi),
            "cross_region_abs_error_bits": abs(
                cross_region_xi_from_conditional_tc - cross_region_xi
            ),
            "within_region_abs_error_bits": abs(
                within_region_xi_from_conditional_tc - within_region_xi_sum
            ),
        },
        "lag_scalar_ei_sums_bits": lag_ei_sums,
        "region_ei": records(zone_ids, region_ei, "ei_bits"),
        "within_region_xi": records(zone_ids, within_region_xi, "xi_bits"),
        "target_resolved": [
            {
                "target_zone_id": int(zone_ids[target]),
                "joint_ei_bits": float(target_joint_ei[target]),
                "cross_region_xi_bits": float(target_cross_region_xi[target]),
                "within_region_xi_bits": float(target_within_region_xi[target]),
                "system_xi_bits": float(target_system_xi[target]),
            }
            for target in range(n_zones)
        ],
        "nonnegative_tolerance": {
            "bits": XI_TOLERANCE_BITS,
            "policy": "No clipping. Values in [-tolerance,0) are numerical zero; computation fails explicitly for values below -tolerance.",
            "within_region_xi": signed_diagnostics(within_region_xi),
            "target_cross_region_xi": signed_diagnostics(target_cross_region_xi),
            "target_within_region_xi": signed_diagnostics(target_within_region_xi),
            "target_system_xi": signed_diagnostics(target_system_xi),
        },
        "covariance_ridge_sensitivity": covariance_ridge_sensitivity,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), **output["decomposition_bits"]}, indent=2))


if __name__ == "__main__":
    main()
