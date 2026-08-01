#!/usr/bin/env python3
"""Run paired Schaefer100 DMF scans for three structural-connectome nulls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import networkx as nx
import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_dmf_critical_phi_yeo7_hierarchy import (
    load_region_labels,
    yeo7_membership,
)
from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module


BASE = ROOT / "results" / "dmf_schaefer100" / "structural_nulls"
EMPIRICAL_SOURCE = (
    ROOT / "results" / "dmf_schaefer100" / "source" / "group_mean_native_mean_rate.npz"
)
EMPIRICAL_MAIN = ROOT / "results" / "dmf_schaefer100" / "full" / "main_confirmation.npz"
EMPIRICAL_YEO = ROOT / "results" / "dmf_schaefer100" / "full" / "critical_yeo7.npz"
LABELS = ROOT / "results" / "dmf_schaefer100" / "schaefer100_labels.txt"
STATUS = ROOT / "docs" / "log" / "dmf_schaefer100_structural_nulls_progress.json"
LOG = ROOT / "docs" / "log" / "dmf_schaefer100_structural_nulls.log"
PYTHON = Path("/opt/anaconda3/envs/py311/bin/python")
SYN_TOLERANCE_BITS = 1.0e-8
NULL_SEEDS = {
    "weight_shuffle": 20_260_731,
    "degree_strength": 20_260_732,
    "yeo_block": 20_260_733,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=BASE)
    parser.add_argument("--status", type=Path, default=STATUS)
    parser.add_argument("--log", type=Path, default=LOG)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def spectral_radius(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvalsh(np.asarray(matrix, dtype=float)))))


def upper_values(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(matrix.shape[0], 1)
    return np.asarray(matrix, dtype=float)[indices]


def symmetric_from_upper(values: np.ndarray, dimension: int) -> np.ndarray:
    matrix = np.zeros((dimension, dimension), dtype=float)
    indices = np.triu_indices(dimension, 1)
    matrix[indices] = np.asarray(values, dtype=float)
    return matrix + matrix.T


def weight_shuffle_null(empirical: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = upper_values(empirical).copy()
    rng.shuffle(values)
    return symmetric_from_upper(values, empirical.shape[0])


def balance_to_strengths(
    base: np.ndarray,
    target_strength: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    positive = np.asarray(base, dtype=float)
    target = np.asarray(target_strength, dtype=float)
    scale = max(float(np.mean(target)), 1.0e-12)

    def residual(log_scale: np.ndarray) -> np.ndarray:
        factors = np.exp(log_scale)
        current = (positive * factors[:, None] * factors[None, :]).sum(axis=1)
        return (current - target) / scale

    fit = least_squares(
        residual,
        np.zeros(len(target), dtype=float),
        bounds=(-20.0, 20.0),
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=10_000,
    )
    factors = np.exp(fit.x)
    matrix = positive * factors[:, None] * factors[None, :]
    np.fill_diagonal(matrix, 0.0)
    absolute_error = np.abs(matrix.sum(axis=1) - target)
    relative_error = absolute_error / np.maximum(target, 1.0e-12)
    return matrix, {
        "optimizer_success": bool(fit.success),
        "optimizer_cost": float(fit.cost),
        "maximum_absolute_strength_error": float(np.max(absolute_error)),
        "maximum_relative_strength_error": float(np.max(relative_error)),
    }


def degree_strength_null(empirical: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    dimension = empirical.shape[0]
    positive_mask = np.asarray(empirical, dtype=float) > 0.0
    complement = nx.Graph()
    complement.add_nodes_from(range(dimension))
    missing_i, missing_j = np.where(np.triu(~positive_mask, 1))
    complement.add_edges_from(zip(missing_i.tolist(), missing_j.tolist()))
    edge_count = complement.number_of_edges()
    if edge_count < 2:
        raise RuntimeError("The complement graph has too few edges to rewire.")
    requested_swaps = 20 * edge_count
    nx.double_edge_swap(
        complement,
        nswap=requested_swaps,
        max_tries=400 * requested_swaps,
        seed=seed,
    )
    rewired_missing = np.zeros_like(positive_mask, dtype=bool)
    for left, right in complement.edges():
        rewired_missing[left, right] = True
        rewired_missing[right, left] = True
    rewired_support = ~rewired_missing
    np.fill_diagonal(rewired_support, False)
    observed_positive_weights = upper_values(empirical)
    observed_positive_weights = observed_positive_weights[observed_positive_weights > 0.0].copy()
    rng.shuffle(observed_positive_weights)
    upper_support = np.triu(rewired_support, 1)
    positions = np.where(upper_support)
    if len(positions[0]) != len(observed_positive_weights):
        raise RuntimeError("Degree-preserving rewiring changed the positive-edge count.")
    base = np.zeros_like(empirical, dtype=float)
    base[positions] = observed_positive_weights
    base = base + base.T
    balanced, audit = balance_to_strengths(base, empirical.sum(axis=1))
    original_missing = set(zip(missing_i.tolist(), missing_j.tolist()))
    new_missing = set((min(i, j), max(i, j)) for i, j in complement.edges())
    audit.update(
        {
            "complement_edge_count": int(edge_count),
            "requested_double_edge_swaps": int(requested_swaps),
            "missing_edge_overlap_fraction": float(
                len(original_missing & new_missing) / max(len(original_missing), 1)
            ),
        }
    )
    return balanced, audit


def yeo_block_null(
    empirical: np.ndarray,
    membership: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = np.zeros_like(empirical, dtype=float)
    block_count = int(np.max(membership)) + 1
    for left_block in range(block_count):
        for right_block in range(left_block, block_count):
            positions: list[tuple[int, int]] = []
            for left in range(empirical.shape[0]):
                for right in range(left + 1, empirical.shape[0]):
                    blocks = (int(membership[left]), int(membership[right]))
                    if tuple(sorted(blocks)) == (left_block, right_block):
                        positions.append((left, right))
            values = np.asarray([empirical[left, right] for left, right in positions])
            rng.shuffle(values)
            for (left, right), value in zip(positions, values):
                matrix[left, right] = matrix[right, left] = float(value)
    return matrix


def block_multiset_equal(
    empirical: np.ndarray,
    null: np.ndarray,
    membership: np.ndarray,
) -> bool:
    block_count = int(np.max(membership)) + 1
    for left_block in range(block_count):
        for right_block in range(left_block, block_count):
            empirical_values: list[float] = []
            null_values: list[float] = []
            for left in range(empirical.shape[0]):
                for right in range(left + 1, empirical.shape[0]):
                    if tuple(sorted((int(membership[left]), int(membership[right])))) != (
                        left_block,
                        right_block,
                    ):
                        continue
                    empirical_values.append(float(empirical[left, right]))
                    null_values.append(float(null[left, right]))
            if not np.array_equal(np.sort(empirical_values), np.sort(null_values)):
                return False
    return True


def matrix_audit(
    empirical: np.ndarray,
    matrix: np.ndarray,
    membership: np.ndarray,
    kind: str,
    extra: dict[str, float] | None = None,
) -> dict[str, object]:
    empirical_degree = np.sum(empirical > 0.0, axis=1)
    degree = np.sum(matrix > 0.0, axis=1)
    empirical_strength = empirical.sum(axis=1)
    strength = matrix.sum(axis=1)
    values = upper_values(matrix)
    audit: dict[str, object] = {
        "kind": kind,
        "finite": bool(np.isfinite(matrix).all()),
        "nonnegative": bool(np.all(matrix >= 0.0)),
        "symmetric": bool(np.allclose(matrix, matrix.T, atol=1.0e-12)),
        "zero_diagonal": bool(np.allclose(np.diag(matrix), 0.0, atol=1.0e-12)),
        "density": float(np.mean(values > 0.0)),
        "total_upper_weight": float(np.sum(values)),
        "spectral_radius": spectral_radius(matrix),
        "global_weight_multiset_equal": bool(
            np.array_equal(np.sort(upper_values(empirical)), np.sort(values))
        ),
        "degree_sequence_equal": bool(
            np.array_equal(np.sort(empirical_degree), np.sort(degree))
        ),
        "nodewise_degree_equal": bool(np.array_equal(empirical_degree, degree)),
        "maximum_absolute_strength_error": float(
            np.max(np.abs(strength - empirical_strength))
        ),
        "maximum_relative_strength_error": float(
            np.max(
                np.abs(strength - empirical_strength)
                / np.maximum(empirical_strength, 1.0e-12)
            )
        ),
        "yeo_block_weight_multisets_equal": bool(
            block_multiset_equal(empirical, matrix, membership)
        ),
    }
    if extra:
        audit.update(extra)
    required = (
        audit["finite"]
        and audit["nonnegative"]
        and audit["symmetric"]
        and audit["zero_diagonal"]
    )
    if kind == "weight_shuffle":
        required = required and audit["global_weight_multiset_equal"]
    elif kind == "degree_strength":
        required = (
            required
            and audit["nodewise_degree_equal"]
            and float(audit["maximum_relative_strength_error"]) < 1.0e-8
        )
    elif kind == "yeo_block":
        required = required and audit["yeo_block_weight_multisets_equal"]
    audit["passed"] = bool(required)
    if not required:
        raise RuntimeError(f"Structural audit failed for {kind}: {audit}")
    return audit


def make_nulls(empirical: np.ndarray, membership: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    degree_matrix, degree_extra = degree_strength_null(
        empirical, NULL_SEEDS["degree_strength"]
    )
    matrices = {
        "weight_shuffle": weight_shuffle_null(empirical, NULL_SEEDS["weight_shuffle"]),
        "degree_strength": degree_matrix,
        "yeo_block": yeo_block_null(empirical, membership, NULL_SEEDS["yeo_block"]),
    }
    audits = {
        kind: matrix_audit(
            empirical,
            matrix,
            membership,
            kind,
            degree_extra if kind == "degree_strength" else None,
        )
        for kind, matrix in matrices.items()
    }
    return matrices, audits


def write_status(
    path: Path,
    *,
    phase: str,
    current: int,
    total: int,
    started: float,
    metrics: dict[str, object] | None = None,
    message: str | None = None,
) -> None:
    elapsed = time.monotonic() - started
    rate = current / elapsed if current > 0 and elapsed > 0 else 0.0
    payload: dict[str, object] = {
        "phase": phase,
        "current": current,
        "total": total,
        "unit": "condition",
        "elapsed_seconds": elapsed,
        "eta_seconds": (total - current) / rate if rate > 0 else None,
        "metrics": metrics or {},
        "updated_at": time.time(),
    }
    if message:
        payload["message"] = message
    atomic_json(path, payload)


def run_subprocess(
    command: list[str],
    *,
    log_path: Path,
    status_path: Path,
    started: float,
    phase: str,
    offset: int,
    total: int,
) -> int:
    completed = 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {phase}\n")
        log_handle.write(" ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
            if re.search(r"\bseed=\d+\s+G=", line):
                completed += 1
                write_status(
                    status_path,
                    phase=phase,
                    current=offset + completed,
                    total=total,
                    started=started,
                    metrics={"last_output": line.strip()},
                )
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return completed


def syn_audit(values: np.ndarray, label: str) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    minimum = float(np.min(array))
    violation_count = int(np.sum(array < -SYN_TOLERANCE_BITS))
    near_zero_count = int(np.sum((array < 0.0) & (array >= -SYN_TOLERANCE_BITS)))
    if violation_count:
        raise RuntimeError(
            f"{label} has {violation_count} Syn values below -{SYN_TOLERANCE_BITS:g}; "
            f"minimum={minimum:.6g} bits."
        )
    return {
        "tolerance_bits": SYN_TOLERANCE_BITS,
        "minimum_bits": minimum,
        "near_zero_count": near_zero_count,
        "violation_count": violation_count,
    }


def peak_window(g: np.ndarray, phi: np.ndarray) -> np.ndarray:
    peak = int(np.argmax(np.mean(phi, axis=0)))
    start = min(max(peak - 1, 0), len(g) - 3)
    return np.asarray(g[start : start + 3], dtype=float)


def condition_summary(
    name: str,
    source_path: Path,
    main_path: Path,
    yeo_path: Path,
) -> dict[str, object]:
    with np.load(source_path) as source:
        connectivity = np.asarray(source["connectivity"], dtype=float)
        jfic_converged = bool(np.all(source["j_fic_calibration_converged"]))
        jfic_error = float(np.nanmax(source["j_fic_calibration_max_abs_error_hz"]))
    with np.load(main_path, allow_pickle=True) as main:
        g = np.asarray(main["G"], dtype=float)
        modes = [str(value) for value in main["modes"]]
        phi = np.asarray(main["phi_eid"], dtype=float)[modes.index("direct")]
        clip_fraction = np.asarray(main["clip_fraction"], dtype=float)
        seeds = np.asarray(main["seeds"], dtype=int)
    with np.load(yeo_path, allow_pickle=True) as yeo:
        fine = np.asarray(yeo["fine_phi"], dtype=float)
        cross = np.asarray(yeo["cross_roi"], dtype=float)
        between = np.asarray(yeo["between_groups"], dtype=float)
        hierarchy_error = float(np.asarray(yeo["hierarchy_max_abs_error"]).item())
        shapley_error = float(np.max(np.abs(yeo["between_group_shapley_sum_error"])))
    peak_index = np.argmax(phi, axis=1)
    peak_g_by_seed = g[peak_index]
    peak_phi_by_seed = phi[np.arange(len(seeds)), peak_index]
    cross_fraction_by_seed = np.mean(cross / fine, axis=1)
    between_fraction_by_seed = np.mean(between / fine, axis=1)
    return {
        "name": name,
        "G": g.tolist(),
        "seeds": seeds.tolist(),
        "phi_mean": np.mean(phi, axis=0).tolist(),
        "phi_sd": np.std(phi, axis=0, ddof=1).tolist() if len(seeds) > 1 else np.zeros_like(g).tolist(),
        "peak_G_by_seed": peak_g_by_seed.tolist(),
        "peak_G_mean": float(np.mean(peak_g_by_seed)),
        "peak_G_mean_curve": float(g[np.argmax(np.mean(phi, axis=0))]),
        "peak_phi_by_seed_bits": peak_phi_by_seed.tolist(),
        "peak_phi_mean_bits": float(np.mean(peak_phi_by_seed)),
        "effective_peak_coupling_mean": float(np.mean(peak_g_by_seed) * spectral_radius(connectivity)),
        "cross_roi_fraction_by_seed": cross_fraction_by_seed.tolist(),
        "cross_roi_fraction_mean": float(np.mean(cross_fraction_by_seed)),
        "between_network_fraction_by_seed": between_fraction_by_seed.tolist(),
        "between_network_fraction_mean": float(np.mean(between_fraction_by_seed)),
        "spectral_radius": spectral_radius(connectivity),
        "maximum_state_boundary_fraction": float(np.max(clip_fraction)),
        "jfic_converged": jfic_converged,
        "jfic_max_abs_error_hz": jfic_error,
        "hierarchy_max_abs_error_bits": hierarchy_error,
        "shapley_max_abs_error_bits": shapley_error,
        "syn_nonnegative_audit": syn_audit(phi, f"{name} main scan"),
        "hierarchy_syn_nonnegative_audit": syn_audit(fine, f"{name} hierarchy"),
    }


def paired_deltas(empirical: dict[str, object], null: dict[str, object]) -> dict[str, object]:
    fields = {
        "peak_phi_bits": "peak_phi_by_seed_bits",
        "peak_G": "peak_G_by_seed",
        "cross_roi_fraction": "cross_roi_fraction_by_seed",
        "between_network_fraction": "between_network_fraction_by_seed",
    }
    result: dict[str, object] = {}
    for output_name, field in fields.items():
        observed = np.asarray(empirical[field], dtype=float)
        comparison = np.asarray(null[field], dtype=float)
        delta = observed - comparison
        result[output_name] = {
            "empirical_minus_null_by_seed": delta.tolist(),
            "mean": float(np.mean(delta)),
            "sd": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
            "empirical_greater_count": int(np.sum(delta > 0.0)),
            "seed_count": int(len(delta)),
        }
    result["effective_peak_coupling"] = float(
        empirical["effective_peak_coupling_mean"]
    ) - float(null["effective_peak_coupling_mean"])
    return result


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    status_path = args.status if args.status.is_absolute() else ROOT / args.status
    log_path = args.log if args.log.is_absolute() else ROOT / args.log
    run_dir = output_dir / args.mode
    run_dir.mkdir(parents=True, exist_ok=True)
    with np.load(EMPIRICAL_SOURCE) as source:
        empirical = np.asarray(source["connectivity"], dtype=float)
    labels = load_region_labels(LABELS, empirical.shape[0])
    membership, _, network_names = yeo7_membership(labels)
    matrices, audits = make_nulls(empirical, membership)
    np.savez_compressed(
        run_dir / "null_connectomes.npz",
        empirical=empirical,
        network_membership=membership,
        network_names=np.asarray(network_names, dtype=object),
        **matrices,
    )
    atomic_json(run_dir / "structural_audit.json", audits)

    if args.mode == "full":
        g_values = np.round(np.arange(0.0, 3.0001, 0.1), 1)
        seeds = tuple(range(3, 11))
        sample_count = 2048
        horizon = 300
    else:
        g_values = np.asarray((1.2, 1.3, 1.4), dtype=float)
        seeds = (3,)
        sample_count = 256
        horizon = 30
    conditions_per_null = 1 + len(seeds) * len(g_values) + len(seeds) * 3
    total = len(matrices) * conditions_per_null
    completed = 0
    started = time.monotonic()
    write_status(
        status_path,
        phase="structural_audit_complete",
        current=completed,
        total=total,
        started=started,
        metrics={"mode": args.mode},
    )
    dmf = load_dmf_module()
    summaries: dict[str, dict[str, object]] = {}
    try:
        for kind, matrix in matrices.items():
            condition_dir = run_dir / kind
            condition_dir.mkdir(parents=True, exist_ok=True)
            source_path = condition_dir / "source.npz"
            main_path = condition_dir / "main_confirmation.npz"
            yeo_path = condition_dir / "critical_yeo7.npz"
            if args.force or not source_path.exists():
                write_status(
                    status_path,
                    phase=f"{kind}:jfic_calibration",
                    current=completed,
                    total=total,
                    started=started,
                )
                dmf.reproduce_fig6b_mean_rate_transition(
                    connectivity=matrix,
                    g_values=g_values,
                    j_fic_reference_g=1.0,
                    seed=3,
                    continuation=True,
                    compute_phi=False,
                    fic_parameters=dmf.FICParameters(max_iterations=30),
                    expected_regions=100,
                    max_regions=100,
                    results_path=source_path,
                )
            completed += 1
            with np.load(source_path) as source:
                if not np.all(source["j_fic_calibration_converged"]):
                    raise RuntimeError(f"JFIC did not converge for {kind}.")
            if args.force or not main_path.exists():
                g_indices = ",".join(str(index) for index in range(len(g_values)))
                command = [
                    str(PYTHON),
                    "-u",
                    "scripts/run_dmf_diffusive_fullstate_control.py",
                    "--source-results",
                    str(source_path),
                    "--output",
                    str(main_path),
                    "--seeds",
                    ",".join(str(seed) for seed in seeds),
                    "--g-indices",
                    g_indices,
                    "--modes",
                    "direct",
                    "--source-state",
                    "se_si",
                    "--se-low",
                    "0.3",
                    "--se-high",
                    "0.7",
                    "--si-low",
                    "0.3",
                    "--si-high",
                    "0.7",
                    "--sample-count",
                    str(sample_count),
                    "--horizon",
                    str(horizon),
                    "--ridge",
                    "1e-6",
                    "--dt",
                    "0.001",
                    "--sigma",
                    "0.01",
                    "--state-boundary",
                    "none",
                ]
                counted = run_subprocess(
                    command,
                    log_path=log_path,
                    status_path=status_path,
                    started=started,
                    phase=f"{kind}:main_scan",
                    offset=completed,
                    total=total,
                )
                completed += counted
            else:
                completed += len(seeds) * len(g_values)
            with np.load(main_path, allow_pickle=True) as main_archive:
                modes = [str(value) for value in main_archive["modes"]]
                phi = np.asarray(main_archive["phi_eid"], dtype=float)[modes.index("direct")]
                main_g = np.asarray(main_archive["G"], dtype=float)
            syn_audit(phi, f"{kind} main scan")
            critical_g = peak_window(main_g, phi)
            if args.force or not yeo_path.exists():
                command = [
                    str(PYTHON),
                    "-u",
                    "scripts/analyze_dmf_critical_phi_yeo7_hierarchy.py",
                    "--main-confirmation",
                    str(main_path),
                    "--source-results",
                    str(source_path),
                    "--connectivity-labels",
                    str(LABELS),
                    "--critical-g",
                    ",".join(f"{value:.1f}" for value in critical_g),
                    "--output",
                    str(yeo_path),
                    "--figure",
                    str(condition_dir / "yeo7.png"),
                    "--summary",
                    str(condition_dir / "yeo7_summary.json"),
                ]
                counted = run_subprocess(
                    command,
                    log_path=log_path,
                    status_path=status_path,
                    started=started,
                    phase=f"{kind}:yeo7_hierarchy",
                    offset=completed,
                    total=total,
                )
                completed += counted
            else:
                completed += len(seeds) * 3
            summaries[kind] = condition_summary(kind, source_path, main_path, yeo_path)

        empirical_summary = condition_summary(
            "empirical", EMPIRICAL_SOURCE, EMPIRICAL_MAIN, EMPIRICAL_YEO
        )
        summary = {
            "experiment": "Schaefer100 DMF structural-connectome null pilot",
            "mode": args.mode,
            "null_realizations_per_family": 1,
            "estimator": "Gaussian conditional covariance, ridge=1e-6",
            "syn_nonnegative_tolerance_bits": SYN_TOLERANCE_BITS,
            "empirical": empirical_summary,
            "nulls": summaries,
            "paired_deltas": (
                {
                    kind: paired_deltas(empirical_summary, item)
                    for kind, item in summaries.items()
                }
                if args.mode == "full"
                else {}
            ),
            "structural_audits": audits,
            "limitation": (
                "One graph per null family estimates exploratory effect direction, not a null distribution."
            ),
        }
        atomic_json(run_dir / "summary.json", summary)
        write_status(
            status_path,
            phase="complete",
            current=total,
            total=total,
            started=started,
            metrics={"summary": str((run_dir / "summary.json").relative_to(ROOT))},
        )
    except Exception as error:
        write_status(
            status_path,
            phase="failed",
            current=completed,
            total=total,
            started=started,
            message=str(error),
        )
        raise


if __name__ == "__main__":
    main()
