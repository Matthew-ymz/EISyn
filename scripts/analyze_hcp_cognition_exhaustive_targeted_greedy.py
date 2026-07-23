#!/usr/bin/env python3
"""Exhaustive targeted-greedy search for HCP cognition associations."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, t as student_t
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_cognition_targeted_greedy_followup import (
    AUDIT_TOLERANCE,
    EPS,
    atom_key,
    configure_style,
    json_ready,
    load_cached_records,
    load_subjects_and_cognition,
    save_figure,
    short_atom,
    value_for_atom,
)
from scripts.analyze_hcp_schaefer500_yeo7_network_attribution import (
    DEFAULT_REST_ROOT,
    DEFAULT_TASK_ROOT,
    NETWORK_ORDER,
    discover_inputs,
)
from scripts.analyze_hcp_task_evoked_pc2_xi_hierarchy import (
    STATES,
    network_module_indices,
)
from scripts.phi_hierarchy import (
    greedy_phi_atoms,
    nontrivial_bipartitions,
    subset_phi_raw,
)
from scripts.run_hcp_schaefer500_yeo7_module_phi_decomposition import module_ei_table
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import (
    default_yeo7_labels,
    load_yeo7_groups,
)
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi
from scripts.tune_hcp_task_evoked_xi_hierarchy import prepare_projection


CONFIG_ID = "k1_p3_a1"
K = 1
ORDER = 3
ALPHA = 1.0
SPLIT_TOLERANCE = 1.0e-4
PERMUTATION_SEED = 2026072301
DEFAULT_PERMUTATIONS = 20_000
DEFAULT_COGNITION = (
    ROOT / "results/hcp_single_group_sem_full_1206/selected_29_sem_results.csv"
)
DEFAULT_ARRAYS = (
    ROOT / "results/hcp_schaefer500_task_evoked_xi_tuning/full/k1_p3_a1/arrays.npz"
)
DEFAULT_CACHED_RECORDS = (
    ROOT / "results/hcp_schaefer500_task_evoked_xi_tuning/full/records.jsonl"
)
DEFAULT_OUTPUT = ROOT / "results/hcp_cognition_exhaustive_targeted_greedy"
SCORES = ("cry_score", "mem_score", "spd_score")
SCORE_LABELS = {
    "cry_score": "Crystallized cognition",
    "mem_score": "Memory",
    "spd_score": "Processing speed",
}
METRIC_LABELS = {
    "targeted_first_residual": "Targeted first-step residual",
    "fixed_block_synergy": "Fixed-coalition total synergy",
    "forced_root_bridge_residual": "Forced root bridge residual",
}
METRIC_COLORS = {
    "targeted_first_residual": "#B65F3C",
    "fixed_block_synergy": "#6F83B5",
    "forced_root_bridge_residual": "#71A6A1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rest-root", type=Path, default=DEFAULT_REST_ROOT)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--labels", type=Path, default=default_yeo7_labels(500))
    parser.add_argument("--cognition", type=Path, default=DEFAULT_COGNITION)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument("--cached-records", type=Path, default=DEFAULT_CACHED_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def all_coalitions() -> tuple[tuple[str, ...], ...]:
    names = tuple(NETWORK_ORDER)
    return tuple(
        coalition
        for size in range(2, len(names) + 1)
        for coalition in itertools.combinations(names, size)
    )


def first_greedy_residual(
    subset: Sequence[str],
    table: Mapping[tuple[str, ...], float],
    singleton: Mapping[str, float],
) -> float:
    """Return the atom at the root of a greedy run started from ``subset``."""
    ordered = tuple(subset)
    block_phi = subset_phi_raw(ordered, table, singleton)
    if len(ordered) <= 1 or block_phi <= 1.0e-5:
        return 0.0
    candidates: list[tuple[float, float, tuple[str, ...], tuple[str, ...]]] = []
    for left, right in nontrivial_bipartitions(ordered):
        left_phi = subset_phi_raw(left, table, singleton)
        right_phi = subset_phi_raw(right, table, singleton)
        residual = block_phi - left_phi - right_phi
        if residual < -SPLIT_TOLERANCE:
            continue
        candidates.append((left_phi + right_phi, residual, left, right))
    if not candidates:
        return float(block_phi)
    captured, residual, _, _ = candidates[0]
    for candidate in candidates[1:]:
        candidate_captured, candidate_residual, candidate_left, candidate_right = candidate
        if candidate_captured > captured or (
            np.isclose(candidate_captured, captured) and candidate_residual < residual
        ):
            captured, residual, _, _ = (
                candidate_captured,
                candidate_residual,
                candidate_left,
                candidate_right,
            )
    if captured <= 1.0e-5:
        return float(block_phi)
    return float(residual) if residual > 1.0e-5 else 0.0


def compute_model_metrics(
    table: Mapping[tuple[str, ...], float], coalitions: Sequence[tuple[str, ...]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    singleton = {name: float(table[(name,)]) for name in NETWORK_ORDER}
    full = tuple(NETWORK_ORDER)
    full_phi = subset_phi_raw(full, table, singleton)
    free_atoms = greedy_phi_atoms(full, table, singleton_ei=singleton)
    targeted = np.zeros(len(coalitions), dtype=float)
    block = np.zeros(len(coalitions), dtype=float)
    bridge = np.full(len(coalitions), np.nan, dtype=float)
    free = np.zeros(len(coalitions), dtype=float)
    for index, coalition in enumerate(coalitions):
        targeted[index] = first_greedy_residual(coalition, table, singleton)
        block[index] = subset_phi_raw(coalition, table, singleton)
        free[index] = value_for_atom(free_atoms, coalition)[0]
        if coalition != full:
            complement = tuple(name for name in full if name not in coalition)
            bridge[index] = full_phi - block[index] - subset_phi_raw(
                complement, table, singleton
            )
        standalone = value_for_atom(
            greedy_phi_atoms(coalition, table, singleton_ei=singleton), coalition
        )[0]
        if not np.isclose(targeted[index], standalone, atol=AUDIT_TOLERANCE, rtol=0.0):
            raise AssertionError(
                f"First-step audit failed for {coalition}: {targeted[index]} vs {standalone}"
            )
    return targeted, block, bridge, free, float(full_phi)


def compute_metrics(args: argparse.Namespace) -> dict[str, np.ndarray]:
    subjects, _ = load_subjects_and_cognition(args.arrays, args.cognition, int(args.max_subjects))
    archive = np.load(args.arrays)
    archive_subjects = archive["subjects"].astype(str).tolist()
    state_names = archive["states"].astype(str).tolist()
    coalitions = all_coalitions()
    archive_coalitions = archive["atom_names"].astype(str).tolist()
    if [atom_key(item) for item in coalitions] != archive_coalitions:
        raise ValueError("Coalition ordering does not match the existing arrays")
    discovered = discover_inputs(args.rest_root, args.task_root)
    groups = load_yeo7_groups(args.labels, expected_parcels=500)
    n_states, n_subjects, n_coalitions = len(STATES), len(subjects), len(coalitions)
    targeted = np.zeros((n_states, n_subjects, n_coalitions), dtype=float)
    block = np.zeros_like(targeted)
    bridge = np.full_like(targeted, np.nan)
    free = np.zeros_like(targeted)
    full_phi = np.zeros((n_states, n_subjects), dtype=float)
    jobs = [
        (state_index, subject_index, state, subject)
        for state_index, state in enumerate(STATES)
        for subject_index, subject in enumerate(subjects)
    ]
    for state_index, subject_index, state, subject in tqdm(
        jobs, desc="Exhaustive targeted greedy", unit="model"
    ):
        if subject not in discovered:
            raise KeyError(f"Missing imaging input for {subject}")
        projections, _, development_end = prepare_projection(
            Path(discovered[subject][state]), groups, state=state, max_components=2
        )
        fitted = fit_delta_history_phi(
            projections[K], alpha=ALPHA, order=ORDER, development_end=development_end
        )
        indices = network_module_indices(NETWORK_ORDER, n_components=K, order=ORDER)
        table = module_ei_table(
            fitted["transition"], fitted["noise_covariance"], indices, ridge=1.0e-6
        )
        values = compute_model_metrics(table, coalitions)
        targeted[state_index, subject_index] = values[0]
        block[state_index, subject_index] = values[1]
        bridge[state_index, subject_index] = values[2]
        free[state_index, subject_index] = values[3]
        full_phi[state_index, subject_index] = values[4]
    archive_state_indices = [state_names.index(state) for state in STATES]
    archive_subject_indices = [archive_subjects.index(subject) for subject in subjects]
    cached_free = (
        np.asarray(archive["atom_share"], dtype=float)
        * np.asarray(archive["cross_xi"], dtype=float)[:, :, None]
    )[np.ix_(archive_state_indices, archive_subject_indices, np.arange(n_coalitions))]
    maximum_free_difference = float(np.max(np.abs(free - cached_free)))
    if maximum_free_difference > AUDIT_TOLERANCE:
        raise AssertionError(
            f"Full free-greedy audit failed: {maximum_free_difference:.3e} bits"
        )
    return {
        "states": np.asarray(STATES),
        "subjects": np.asarray(subjects),
        "coalitions": np.asarray([atom_key(item) for item in coalitions]),
        "coalition_sizes": np.asarray([len(item) for item in coalitions], dtype=int),
        "targeted_first_residual": targeted,
        "fixed_block_synergy": block,
        "forced_root_bridge_residual": bridge,
        "rerun_free_greedy": free,
        "full_cross_synergy": full_phi,
        "max_abs_rerun_free_minus_cached_bits": np.asarray(maximum_free_difference),
    }


def save_metrics(metrics: Mapping[str, np.ndarray], path: Path) -> None:
    np.savez_compressed(path, **metrics)


def load_metrics(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path)
    return {key: np.asarray(archive[key]) for key in archive.files}


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    finite = np.isfinite(values)
    adjusted = np.full_like(values, np.nan)
    finite_values = values[finite]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    corrected = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(corrected)
    restored[order] = np.minimum(corrected, 1.0)
    adjusted[finite] = restored
    return adjusted


def association_matrix(values: np.ndarray, score: np.ndarray) -> dict[str, np.ndarray]:
    matrix = np.asarray(values, dtype=float)
    ranks_y = rankdata(matrix, axis=1)
    ranks_x = rankdata(np.asarray(score, dtype=float))
    centered_y = ranks_y - ranks_y.mean(axis=1, keepdims=True)
    centered_x = ranks_x - ranks_x.mean()
    norm_y = np.linalg.norm(centered_y, axis=1)
    norm_x = float(np.linalg.norm(centered_x))
    valid = norm_y > 0.0
    rho = np.full(len(matrix), np.nan)
    rho[valid] = (centered_y[valid] @ centered_x) / (norm_y[valid] * norm_x)
    dof = len(score) - 2
    transformed = rho * np.sqrt(dof / np.maximum(1.0 - rho * rho, 1.0e-15))
    p_raw = 2.0 * student_t.sf(np.abs(transformed), df=dof)
    standardized_y = np.zeros_like(centered_y)
    standardized_y[valid] = centered_y[valid] / norm_y[valid, None]
    standardized_x = centered_x / norm_x
    return {
        "rho": rho,
        "p_raw": p_raw,
        "rank_standardized_features": standardized_y,
        "rank_standardized_score": standardized_x,
        "valid": valid,
    }


def permutation_statistics(
    standardized_features: np.ndarray,
    standardized_score: np.ndarray,
    observed_rho: np.ndarray,
    *,
    repeats: int,
    seed: int,
    chunk_size: int = 1_000,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    observed_abs = np.abs(observed_rho)
    point_counts = np.zeros(len(observed_rho), dtype=np.int64)
    max_counts = np.zeros(len(observed_rho), dtype=np.int64)
    valid = np.isfinite(observed_rho)
    for start in tqdm(
        range(0, repeats, chunk_size), desc="Label permutations", unit="chunk", leave=False
    ):
        count = min(chunk_size, repeats - start)
        permutations = np.asarray(
            [rng.permutation(len(standardized_score)) for _ in range(count)], dtype=int
        )
        permuted_scores = standardized_score[permutations]
        permuted_rho = standardized_features @ permuted_scores.T
        absolute = np.abs(permuted_rho)
        point_counts[valid] += np.sum(
            absolute[valid] >= observed_abs[valid, None] - 1.0e-15, axis=1
        )
        maximum = np.nanmax(absolute[valid], axis=0)
        max_counts[valid] += np.sum(
            maximum[None, :] >= observed_abs[valid, None] - 1.0e-15, axis=1
        )
    point = np.full(len(observed_rho), np.nan)
    max_t = np.full(len(observed_rho), np.nan)
    point[valid] = (point_counts[valid] + 1.0) / (repeats + 1.0)
    max_t[valid] = (max_counts[valid] + 1.0) / (repeats + 1.0)
    return point, max_t


def flatten_features(metrics: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    states = metrics["states"].astype(str).tolist()
    coalitions = metrics["coalitions"].astype(str).tolist()
    sizes = metrics["coalition_sizes"].astype(int).tolist()
    matrices = []
    metadata: list[dict[str, Any]] = []
    for metric in METRIC_LABELS:
        values = np.asarray(metrics[metric], dtype=float)
        for state_index, state in enumerate(states):
            for coalition_index, coalition in enumerate(coalitions):
                vector = values[state_index, :, coalition_index]
                if not np.isfinite(vector).all():
                    continue
                matrices.append(vector)
                metadata.append(
                    {
                        "metric": metric,
                        "state": state,
                        "coalition": coalition,
                        "coalition_size": int(sizes[coalition_index]),
                        "state_index": state_index,
                        "coalition_index": coalition_index,
                    }
                )
    return np.asarray(matrices, dtype=float), metadata


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan")
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def leave_one_out(x: np.ndarray, y: np.ndarray, rho: float) -> dict[str, float]:
    values = []
    for index in range(len(x)):
        keep = np.arange(len(x)) != index
        value, _ = safe_spearman(x[keep], y[keep])
        if np.isfinite(value):
            values.append(value)
    array = np.asarray(values, dtype=float)
    return {
        "minimum_rho": float(array.min()),
        "median_rho": float(np.median(array)),
        "maximum_rho": float(array.max()),
        "same_direction_fraction": float(np.mean(np.sign(array) == np.sign(rho))),
    }


def deterministic_split(subjects: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(
        range(len(subjects)),
        key=lambda index: hashlib.sha256(
            f"hcp-cognition-discovery-20260723|{subjects[index]}".encode()
        ).hexdigest(),
    )
    discovery = np.zeros(len(subjects), dtype=bool)
    discovery[ordered[:15]] = True
    return discovery, ~discovery


def summarize_score(
    score_name: str,
    score: np.ndarray,
    feature_values: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    subjects: Sequence[str],
    *,
    permutations: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    association = association_matrix(feature_values, score)
    point_perm, max_t = permutation_statistics(
        association["rank_standardized_features"],
        association["rank_standardized_score"],
        association["rho"],
        repeats=permutations,
        seed=PERMUTATION_SEED + SCORES.index(score_name),
    )
    q_values = bh_adjust(association["p_raw"])
    rows = []
    for index, item in enumerate(metadata):
        values = feature_values[index]
        rows.append(
            {
                "score": score_name,
                "score_label": SCORE_LABELS[score_name],
                **dict(item),
                "rho": float(association["rho"][index]),
                "p_raw_two_sided": float(association["p_raw"][index]),
                "p_permutation_pointwise": float(point_perm[index]),
                "q_bh_search_space": float(q_values[index]),
                "p_fwer_max_abs_rho": float(max_t[index]),
                "positive_subjects": int(np.count_nonzero(values > EPS)),
                "negative_subjects": int(np.count_nonzero(values < -EPS)),
                "near_zero_subjects": int(np.count_nonzero(np.abs(values) <= EPS)),
                "mean_bits": float(values.mean()),
            }
        )
    ranked = sorted(
        [row for row in rows if np.isfinite(row["rho"])],
        key=lambda row: (
            float(row["p_permutation_pointwise"]),
            float(row["p_raw_two_sided"]),
            -abs(float(row["rho"])),
        ),
    )
    selected = ranked[0]
    selected_index = rows.index(selected)
    selected_values = feature_values[selected_index]
    selected["leave_one_out"] = leave_one_out(score, selected_values, float(selected["rho"]))
    discovery, confirmation = deterministic_split(subjects)
    discovery_rho = []
    discovery_p = []
    for values in feature_values:
        rho, p_value = safe_spearman(score[discovery], values[discovery])
        discovery_rho.append(rho)
        discovery_p.append(p_value)
    discovery_order = sorted(
        [index for index in range(len(rows)) if np.isfinite(discovery_rho[index])],
        key=lambda index: (discovery_p[index], -abs(discovery_rho[index])),
    )
    discovery_index = discovery_order[0]
    confirmation_rho, confirmation_p = safe_spearman(
        score[confirmation], feature_values[discovery_index, confirmation]
    )
    summary = {
        "score": score_name,
        "score_label": SCORE_LABELS[score_name],
        "feature_count": len(rows),
        "raw_p_below_0_05": int(sum(row["p_raw_two_sided"] < 0.05 for row in rows)),
        "pointwise_permutation_p_below_0_05": int(
            sum(row["p_permutation_pointwise"] < 0.05 for row in rows)
        ),
        "bh_q_below_0_05": int(sum(row["q_bh_search_space"] < 0.05 for row in rows)),
        "max_t_p_below_0_05": int(sum(row["p_fwer_max_abs_rho"] < 0.05 for row in rows)),
        "selected_full_sample": selected,
        "top_10": ranked[:10],
        "split_half_diagnostic": {
            "discovery_n": int(discovery.sum()),
            "confirmation_n": int(confirmation.sum()),
            "discovery_selected": rows[discovery_index],
            "discovery_rho": float(discovery_rho[discovery_index]),
            "discovery_p_raw": float(discovery_p[discovery_index]),
            "confirmation_rho": confirmation_rho,
            "confirmation_p_raw": confirmation_p,
            "same_direction": bool(
                np.isfinite(confirmation_rho)
                and np.sign(confirmation_rho) == np.sign(discovery_rho[discovery_index])
            ),
        },
    }
    return rows, summary


def selected_feature_values(
    metrics: Mapping[str, np.ndarray], candidate: Mapping[str, Any]
) -> np.ndarray:
    states = metrics["states"].astype(str).tolist()
    coalitions = metrics["coalitions"].astype(str).tolist()
    return np.asarray(metrics[candidate["metric"]], dtype=float)[
        states.index(str(candidate["state"])), :, coalitions.index(str(candidate["coalition"]))
    ]


def plot_selected_scatter(
    summaries: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, np.ndarray],
    cognition: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(10.6, 3.25), constrained_layout=True)
    subject_ids = [subject.removeprefix("sub-") for subject in metrics["subjects"].astype(str)]
    for axis, score_name, label in zip(axes, SCORES, "abc", strict=True):
        candidate = summaries[score_name]["selected_full_sample"]
        x = cognition.loc[subject_ids, score_name].to_numpy(dtype=float)
        y = selected_feature_values(metrics, candidate)
        color = METRIC_COLORS[candidate["metric"]]
        axis.scatter(x, y, s=31, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(float(x.min()), float(x.max()), 200)
        axis.plot(line_x, slope * line_x + intercept, color="#4B5563", lw=0.95, ls="--")
        loo = candidate["leave_one_out"]
        axis.text(
            0.03,
            0.97,
            f"rho={candidate['rho']:.3f}, raw p={candidate['p_raw_two_sided']:.3g}\n"
            f"perm p={candidate['p_permutation_pointwise']:.3g}, q={candidate['q_bh_search_space']:.3g}\n"
            f"LOO rho=[{loo['minimum_rho']:.3f}, {loo['maximum_rho']:.3f}]",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.1,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
            zorder=5,
        )
        axis.set(
            xlabel=SCORE_LABELS[score_name],
            ylabel="Synergy readout (bits)" if axis is axes[0] else "",
            title=f"{candidate['state']} | {short_atom(candidate['coalition'].split('+'))}\n"
            f"{METRIC_LABELS[candidate['metric']]}",
        )
        axis.text(-0.15, 1.08, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    save_figure(figure, output / "exhaustive_top_candidates_scatter")


def plot_search_landscape(
    rows_by_score: Mapping[str, Sequence[Mapping[str, Any]]],
    summaries: Mapping[str, Mapping[str, Any]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), constrained_layout=True)
    for axis, score_name, label in zip(axes, SCORES, "abc", strict=True):
        rows = rows_by_score[score_name]
        for metric in METRIC_LABELS:
            selected = [row for row in rows if row["metric"] == metric]
            axis.scatter(
                [row["rho"] for row in selected],
                [-math.log10(max(row["p_raw_two_sided"], 1.0e-300)) for row in selected],
                s=8,
                color=METRIC_COLORS[metric],
                alpha=0.38,
                linewidth=0,
                label=METRIC_LABELS[metric] if axis is axes[0] else None,
            )
        top = summaries[score_name]["selected_full_sample"]
        axis.scatter(
            [top["rho"]],
            [-math.log10(top["p_raw_two_sided"])],
            marker="*",
            s=85,
            color="#111111",
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
        axis.axhline(-math.log10(0.05), color="#666666", lw=0.8, ls="--")
        axis.set(
            xlabel="Spearman rho",
            ylabel=r"$-\log_{10}$(raw p)" if axis is axes[0] else "",
            title=SCORE_LABELS[score_name],
            xlim=(-0.75, 0.75),
        )
        axis.text(-0.15, 1.07, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3)
    save_figure(figure, output / "exhaustive_search_landscape")


def fmt(value: Any) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.4g}"


def write_report(summary: Mapping[str, Any], output: Path) -> None:
    lines = [
        "# Exhaustive targeted-greedy cognition search",
        "",
        "This search freezes the 29 subjects, full seven-network target, task-evoked PCA, `k=1`, `p=3`, `alpha=1`, chronological split, and affine/Gaussian EI estimator. It scans all 120 source coalitions in REST and seven tasks using three hierarchy readouts.",
        "",
        "## Implementation audit",
        "",
        f"Recomputed free-greedy values reproduce all existing state-subject-coalition cells with maximum absolute difference `{summary['audit']['max_abs_rerun_free_minus_cached_bits']:.3e}` bits.",
        "",
        "## Selected full-sample exploratory candidates",
        "",
        "| Cognition | State | Coalition | Readout | rho | Raw p | Pointwise permutation p | BH q | maxT p | LOO rho range |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for score_name in SCORES:
        candidate = summary["scores"][score_name]["selected_full_sample"]
        loo = candidate["leave_one_out"]
        lines.append(
            f"| {SCORE_LABELS[score_name]} | {candidate['state']} | {candidate['coalition']} | "
            f"{METRIC_LABELS[candidate['metric']]} | {candidate['rho']:+.3f} | "
            f"{candidate['p_raw_two_sided']:.4g} | {candidate['p_permutation_pointwise']:.4g} | "
            f"{candidate['q_bh_search_space']:.4g} | {candidate['p_fwer_max_abs_rho']:.4g} | "
            f"[{loo['minimum_rho']:+.3f}, {loo['maximum_rho']:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Search-space evidence",
            "",
            "| Cognition | Features | Raw p<0.05 | Pointwise permutation p<0.05 | BH q<0.05 | maxT p<0.05 | Split-half confirmation |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for score_name in SCORES:
        item = summary["scores"][score_name]
        split = item["split_half_diagnostic"]
        lines.append(
            f"| {SCORE_LABELS[score_name]} | {item['feature_count']} | {item['raw_p_below_0_05']} | "
            f"{item['pointwise_permutation_p_below_0_05']} | {item['bh_q_below_0_05']} | "
            f"{item['max_t_p_below_0_05']} | rho={fmt(split['confirmation_rho'])}, "
            f"p={fmt(split['confirmation_p_raw'])}, same direction={split['same_direction']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Raw p<0.05 and pointwise permutation p<0.05 identify exploratory leads, not search-space-corrected discoveries. The same 29-subject cohort is used for search and effect estimation. BH q and maxT p quantify the full search burden; unrestricted permutations also do not model possible HCP family exchangeability. Independent subjects or a prespecified replication are required for confirmation.",
            "",
            "The local PEID full text supports fixed source subsets with an unchanged target, while noting that continuous-variable synergy is not theoretically guaranteed to be nonnegative (Zotero key: `MYATYWAJ`, full text).",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / (
        "metrics.npz" if not args.max_subjects else f"smoke_metrics_{args.max_subjects}.npz"
    )
    if cache_path.is_file() and not args.recompute:
        metrics = load_metrics(cache_path)
    else:
        metrics = compute_metrics(args)
        save_metrics(metrics, cache_path)
    if args.max_subjects:
        result = {
            "smoke_subjects": int(args.max_subjects),
            "max_abs_rerun_free_minus_cached_bits": float(
                metrics["max_abs_rerun_free_minus_cached_bits"]
            ),
        }
        print(json.dumps(result, indent=2))
        return result
    subjects, cognition = load_subjects_and_cognition(args.arrays, args.cognition, 0)
    if metrics["subjects"].astype(str).tolist() != subjects:
        raise ValueError("Metric-cache subject order does not match cognition order")
    feature_values, metadata = flatten_features(metrics)
    if len(metadata) != 2_872:
        raise AssertionError(f"Expected 2,872 searchable features, found {len(metadata)}")
    subject_ids = [subject.removeprefix("sub-") for subject in subjects]
    rows_by_score: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for score_name in SCORES:
        values = cognition.loc[subject_ids, score_name].to_numpy(dtype=float)
        rows, score_summary = summarize_score(
            score_name,
            values,
            feature_values,
            metadata,
            subjects,
            permutations=int(args.permutations),
        )
        rows_by_score[score_name] = rows
        summaries[score_name] = score_summary
    with (args.output_dir / "all_associations.jsonl").open("w", encoding="utf-8") as handle:
        for score_name in SCORES:
            for row in rows_by_score[score_name]:
                handle.write(json.dumps(json_ready(row), ensure_ascii=False, allow_nan=False) + "\n")
    summary = {
        "experiment": "Exhaustive targeted-greedy HCP cognition search",
        "config": {
            "subjects": 29,
            "states": metrics["states"].astype(str).tolist(),
            "coalitions": 120,
            "searchable_features_per_score": len(metadata),
            "metrics": list(METRIC_LABELS),
            "permutations": int(args.permutations),
            "permutation_seed": PERMUTATION_SEED,
            "target": "full seven-network next state",
            "k": K,
            "order": ORDER,
            "alpha": ALPHA,
        },
        "audit": {
            "max_abs_rerun_free_minus_cached_bits": float(
                metrics["max_abs_rerun_free_minus_cached_bits"]
            ),
            "tolerance_bits": AUDIT_TOLERANCE,
            "passed": bool(
                float(metrics["max_abs_rerun_free_minus_cached_bits"])
                <= AUDIT_TOLERANCE
            ),
        },
        "scores": summaries,
        "interpretation_boundary": (
            "Same-cohort exhaustive exploration; raw and pointwise permutation p values do not correct the search. "
            "BH q/maxT and independent replication determine confirmatory evidence."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_selected_scatter(summaries, metrics, cognition, args.output_dir)
    plot_search_landscape(rows_by_score, summaries, args.output_dir)
    write_report(summary, args.output_dir)
    print(json.dumps(json_ready({"audit": summary["audit"], "scores": summaries}), indent=2, ensure_ascii=False, allow_nan=False))
    return summary


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
