#!/usr/bin/env python3
"""Replicate triplet PED and O-information across 57 paired HCP subjects."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma, ndtri
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import load_yeo7_groups
from scripts.tune_hcp_task_evoked_xi_hierarchy import prepare_projection


REST_ROOT = ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_57_brain"
TASK_ROOT = ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_57_brain"
LABELS = (
    ROOT
    / "data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30"
    / "_atlas_labels/Schaefer2018_1000Parcels_7Networks_order.txt"
)
DEFAULT_OUTPUT = ROOT / "results/hcp_57_ped_oinfo_replication"

NETWORKS = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
SHORT = {
    "Vis": "V",
    "SomMot": "SM",
    "DorsAttn": "DAN",
    "SalVentAttn": "VAN",
    "Limbic": "Lim",
    "Cont": "FPN",
    "Default": "DMN",
}
STATES = ("REST", "EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")
STATE_LABELS = ("REST", "Emotion", "Gambling", "Language", "Motor", "Relational", "Social", "WM")
COMMON_LENGTH = 176
BOOTSTRAPS = 5_000
SEED = 20260915
PED_TOLERANCE_BITS = 1.0e-10

# Exact n=3 antichain ordering and Moebius inverse shipped with the authors'
# SxPID implementation. Embedding this 18-node lattice avoids its 55 MB n=2..5
# lookup file while preserving the authors' calculation exactly.
PED_ANTICHAINS = (
    ((1, 2, 3),),
    ((2, 3),),
    ((1, 3),),
    ((1, 3), (2, 3)),
    ((1, 2),),
    ((1, 2), (2, 3)),
    ((1, 2), (1, 3)),
    ((1, 2), (1, 3), (2, 3)),
    ((3,),),
    ((3,), (1, 2)),
    ((2,),),
    ((2,), (1, 3)),
    ((2,), (3,)),
    ((1,),),
    ((1,), (2, 3)),
    ((1,), (3,)),
    ((1,), (2,)),
    ((1,), (2,), (3,)),
)
PED_MOEBIUS = np.asarray(
    [
        [1, -1, -1, 1, -1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, -1, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, -1, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, -1, 1, 0, -1, 1, 1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=float,
)
PED_RED_INDEX = PED_ANTICHAINS.index(((1,), (2,), (3,)))
PED_SYN_INDICES = tuple(
    PED_ANTICHAINS.index(key)
    for key in (
        ((3,), (1, 2)),
        ((2,), (1, 3)),
        ((1,), (2, 3)),
        ((1, 2), (1, 3), (2, 3)),
        ((1, 3), (2, 3)),
        ((1, 2), (2, 3)),
        ((1, 2), (1, 3)),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true", help="Run only the first subject.")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def triplets() -> tuple[tuple[str, str, str], ...]:
    return tuple(itertools.combinations(NETWORKS, 3))


def triplet_names(values: Sequence[Sequence[str]]) -> np.ndarray:
    return np.asarray(["+".join(value) for value in values])


def compact_name(name: str) -> str:
    return "+".join(SHORT[item] for item in name.split("+"))


def subject_ids() -> np.ndarray:
    rest = {path.name for path in REST_ROOT.iterdir() if path.name.startswith("sub-")}
    task = {path.name for path in TASK_ROOT.iterdir() if path.name.startswith("sub-")}
    subjects = np.asarray(sorted(rest & task))
    if subjects.shape != (57,):
        raise ValueError(f"Expected 57 paired subjects, found {len(subjects)}")
    return subjects


def rest_path(subject: str) -> Path:
    matches = [path for path in (REST_ROOT / subject).iterdir() if "REST1_LR" in path.name and path.suffix == ".mat"]
    if len(matches) != 1:
        raise ValueError(f"Expected one REST1_LR file for {subject}, found {len(matches)}")
    return matches[0]


def state_path(subject: str, state: str) -> Path:
    if state == "REST":
        return rest_path(subject)
    path = TASK_ROOT / subject / f"{state}_LR.mat"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_projection(subject: str, state: str, groups: Mapping[str, Sequence[int]]) -> np.ndarray:
    kwargs: dict[str, Any] = {"state": state, "max_components": 1, "expected_parcels": 1000}
    if state == "REST":
        kwargs["rest_data_key"] = "Schaefer1000"
    else:
        kwargs.update(task_retained_key="Schaefer1000_taskRetained", task_regressed_key="Schaefer1000_taskRegressed")
    projections, _, _ = prepare_projection(state_path(subject, state), groups, **kwargs)
    values = np.asarray(projections[1], dtype=float)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) < COMMON_LENGTH or not np.isfinite(values).all():
        raise ValueError(f"Invalid projection for {subject}/{state}: {values.shape}")
    return values


def binarize_like_authors(values: np.ndarray) -> np.ndarray:
    """Author-equivalent per-channel z-score followed by sign binarization."""
    values = np.asarray(values, dtype=float)
    standard_deviation = values.std(axis=0)
    if np.any(standard_deviation == 0.0):
        raise ValueError("PED cannot z-score a constant channel")
    z_values = (values - values.mean(axis=0)) / standard_deviation
    return (z_values > 0.0).astype(np.int16)


def ped_atoms_from_binary(binary_triplet: np.ndarray) -> np.ndarray:
    """Exact informative Hsx/SxPID atoms for one binary triplet."""
    data = np.asarray(binary_triplet, dtype=np.int16)
    if data.ndim != 2 or data.shape[1] != 3 or not np.all((data == 0) | (data == 1)):
        raise ValueError("PED input must be an n-by-3 binary array")
    states, counts = np.unique(data, axis=0, return_counts=True)
    probabilities = counts.astype(float) / float(len(data))
    i_cap_plus = np.zeros(len(PED_ANTICHAINS), dtype=float)
    for realization, realization_probability in zip(states, probabilities, strict=True):
        for index, antichain in enumerate(PED_ANTICHAINS):
            union = np.zeros(len(states), dtype=bool)
            for source_set in antichain:
                columns = np.asarray(source_set, dtype=int) - 1
                union |= np.all(states[:, columns] == realization[columns], axis=1)
            i_cap_plus[index] += realization_probability * -np.log2(probabilities[union].sum())
    return PED_MOEBIUS @ i_cap_plus


def triplet_ped(values: np.ndarray, combinations: Sequence[tuple[str, str, str]]) -> tuple[np.ndarray, np.ndarray, float, int]:
    binary = binarize_like_authors(values)
    network_index = {name: position for position, name in enumerate(NETWORKS)}
    red = np.empty(len(combinations), dtype=float)
    syn = np.empty(len(combinations), dtype=float)
    minimum_atom = np.inf
    tolerance_negative_count = 0
    for row, names in enumerate(combinations):
        atoms = ped_atoms_from_binary(binary[:, [network_index[name] for name in names]])
        minimum_atom = min(minimum_atom, float(atoms.min()))
        tolerance_negative_count += int(np.sum((atoms < 0.0) & (atoms >= -PED_TOLERANCE_BITS)))
        violations = atoms < -PED_TOLERANCE_BITS
        if np.any(violations):
            raise ValueError(
                "PED nonnegativity violation: "
                f"minimum={atoms.min():.12g}, threshold={-PED_TOLERANCE_BITS:.12g}, affected_count={int(violations.sum())}"
            )
        red[row] = atoms[PED_RED_INDEX]
        syn[row] = atoms[np.asarray(PED_SYN_INDICES)].sum()
    return red, syn, minimum_atom, tolerance_negative_count


def copnorm_demean(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    ranks = np.argsort(np.argsort(x, axis=1), axis=1).astype(float) + 1.0
    gaussian = ndtri(ranks / float(x.shape[1] + 1))
    gaussian -= gaussian.mean(axis=1, keepdims=True)
    return gaussian


def entropy_gc_preprocessed(values: np.ndarray, indices: Sequence[int]) -> float:
    x = np.asarray(values[np.asarray(indices, dtype=int)], dtype=float)
    n_features, n_samples = x.shape
    covariance = x @ x.T / float(n_samples - 1)
    cholesky = np.linalg.cholesky(covariance)
    entropy_nats = np.log(np.diag(cholesky)).sum() + 0.5 * n_features * (np.log(2.0 * np.pi) + 1.0)
    psi_terms = 0.5 * digamma((n_samples - np.arange(1, n_features + 1, dtype=float)) / 2.0)
    dterm = 0.5 * (np.log(2.0) - np.log(n_samples - 1.0))
    return float((entropy_nats - n_features * dterm - psi_terms.sum()) / np.log(2.0))


def triplet_oinfo(values: np.ndarray, combinations: Sequence[tuple[str, str, str]]) -> np.ndarray:
    x = copnorm_demean(np.asarray(values, dtype=float).T)
    network_index = {name: position for position, name in enumerate(NETWORKS)}
    singleton_h = {i: entropy_gc_preprocessed(x, (i,)) for i in range(7)}
    pair_h: dict[tuple[int, int], float] = {}
    output = np.empty(len(combinations), dtype=float)
    for row, names in enumerate(combinations):
        ids = tuple(network_index[name] for name in names)
        pairs = tuple(tuple(sorted(pair)) for pair in itertools.combinations(ids, 2))
        for pair in pairs:
            if pair not in pair_h:
                pair_h[pair] = entropy_gc_preprocessed(x, pair)
        output[row] = sum(singleton_h[item] for item in ids) - sum(pair_h[pair] for pair in pairs) + entropy_gc_preprocessed(x, ids)
    return output


def compute_window(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    combinations = triplets()
    red, syn, minimum, near_zero = triplet_ped(values, combinations)
    return red, syn, triplet_oinfo(values, combinations), minimum, near_zero


def compute_subject(subject: str) -> tuple[str, dict[str, np.ndarray], np.ndarray, float, int]:
    groups = load_yeo7_groups(LABELS, expected_parcels=1000)
    keys = ("ped_red_full", "ped_syn_full", "oinfo_full", "ped_red_common", "ped_syn_common", "oinfo_common", "ped_red_first", "ped_syn_first", "oinfo_first", "ped_red_second", "ped_syn_second", "oinfo_second")
    output = {key: np.empty((len(STATES), len(triplets())), dtype=float) for key in keys}
    lengths = np.empty(len(STATES), dtype=int)
    minimum_atom = np.inf
    tolerance_negative_count = 0
    for state_index, state in enumerate(STATES):
        projection = load_projection(subject, state, groups)
        midpoint = len(projection) // 2
        windows = {
            "full": projection,
            "common": projection[:COMMON_LENGTH],
            "first": projection[:midpoint],
            "second": projection[midpoint:],
        }
        for label, window in windows.items():
            red, syn, oinfo, minimum, near_zero = compute_window(window)
            output[f"ped_red_{label}"][state_index] = red
            output[f"ped_syn_{label}"][state_index] = syn
            output[f"oinfo_{label}"][state_index] = oinfo
            minimum_atom = min(minimum_atom, minimum)
            tolerance_negative_count += near_zero
        lengths[state_index] = len(projection)
    return subject, output, lengths, minimum_atom, tolerance_negative_count


def compute_all(subjects: np.ndarray, *, smoke: bool, workers: int) -> tuple[dict[str, np.ndarray], np.ndarray, float, int]:
    selected = subjects[:1] if smoke else subjects
    keys = ("ped_red_full", "ped_syn_full", "oinfo_full", "ped_red_common", "ped_syn_common", "oinfo_common", "ped_red_first", "ped_syn_first", "oinfo_first", "ped_red_second", "ped_syn_second", "oinfo_second")
    output = {key: np.empty((len(STATES), len(selected), len(triplets())), dtype=float) for key in keys}
    lengths = np.empty((len(STATES), len(selected)), dtype=int)
    lookup = {str(subject): index for index, subject in enumerate(selected)}
    minimum_atom = np.inf
    tolerance_negative_count = 0
    if smoke or workers == 1:
        iterator = map(compute_subject, selected.astype(str).tolist())
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(workers)))
        iterator = executor.map(compute_subject, selected.astype(str).tolist())
    try:
        for completed, result in enumerate(iterator, start=1):
            subject, subject_output, subject_lengths, subject_minimum, subject_near_zero = result
            index = lookup[subject]
            for key in keys:
                output[key][:, index] = subject_output[key]
            lengths[:, index] = subject_lengths
            minimum_atom = min(minimum_atom, subject_minimum)
            tolerance_negative_count += subject_near_zero
            print(f"PED/O-information {completed}/{len(selected)}: {subject}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return output, lengths, minimum_atom, tolerance_negative_count


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def paired_state_tests(subject_metric: np.ndarray) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    p_values = []
    rest = subject_metric[0]
    for index, state in enumerate(STATES[1:], start=1):
        delta = subject_metric[index] - rest
        test = wilcoxon(subject_metric[index], rest, alternative="two-sided", zero_method="wilcox")
        p_values.append(float(test.pvalue))
        rows.append({"state": state, "task_minus_rest_mean_bits": float(delta.mean()), "task_minus_rest_median_bits": float(np.median(delta)), "task_above_rest_fraction": float(np.mean(delta > 0.0)), "p_value": float(test.pvalue)})
    for row, q_value in zip(rows, bh_adjust(p_values), strict=True):
        row["q_bh_7"] = float(q_value)
    return rows


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    samples = rng.integers(0, values.shape[1], size=(BOOTSTRAPS, values.shape[1]))
    boot = np.take(values, samples, axis=1).mean(axis=2)
    return np.percentile(boot, [2.5, 97.5], axis=1).T


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(spearmanr(np.ravel(left), np.ravel(right)).statistic)


def winner_stability(values: np.ndarray, names: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for state_index, state in enumerate(STATES):
        counts = np.zeros(values.shape[2], dtype=int)
        for omitted in range(values.shape[1]):
            counts[int(np.argmax(np.delete(values[state_index], omitted, axis=0).mean(axis=0)))] += 1
        order = np.argsort(counts)[::-1]
        output[state] = [{"triplet": str(names[index]), "leave_one_out_wins": int(counts[index])} for index in order[:3] if counts[index] > 0]
    return output


def top_rows(values: np.ndarray, names: np.ndarray, *, largest: bool = True) -> dict[str, list[dict[str, Any]]]:
    rows = {}
    for state_index, state in enumerate(STATES):
        mean = values[state_index].mean(axis=0)
        order = np.argsort(mean)
        if largest:
            order = order[::-1]
        rows[state] = [{"rank": rank, "triplet": str(names[index]), "mean_bits": float(mean[index]), "subject_positive_fraction": float(np.mean(values[state_index, :, index] > 0.0))} for rank, index in enumerate(order[:5], start=1)]
    return rows


def metric_summary(data: Mapping[str, np.ndarray], names: np.ndarray, lengths: np.ndarray, minimum_atom: float, tolerance_negative_count: int) -> dict[str, Any]:
    red = data["ped_red_full"]
    syn = data["ped_syn_full"]
    oinfo = data["oinfo_full"]
    rng = np.random.default_rng(SEED)
    aggregates = {}
    for label, values in (("ped_redundancy", red.mean(axis=2)), ("ped_synergy", syn.mean(axis=2)), ("oinfo_signed", oinfo.mean(axis=2))):
        ci = bootstrap_mean_ci(values, rng)
        aggregates[label] = {
            "state_mean_bits": {state: float(values[i].mean()) for i, state in enumerate(STATES)},
            "state_bootstrap_95_ci_bits": {state: [float(ci[i, 0]), float(ci[i, 1])] for i, state in enumerate(STATES)},
            "paired_task_vs_rest": paired_state_tests(values),
        }
    signed_group_mean = oinfo.mean(axis=1)
    ped_balance = red.mean(axis=1) - syn.mean(axis=1)
    robustness = {}
    for metric in ("ped_red", "ped_syn", "oinfo"):
        full = data[f"{metric}_full"]
        common = data[f"{metric}_common"]
        first = data[f"{metric}_first"]
        second = data[f"{metric}_second"]
        robustness[metric] = {
            "full_vs_common_all_values_spearman": safe_spearman(full, common),
            "split_half_all_values_spearman": safe_spearman(first, second),
            "state_group_mean_full_vs_common_spearman": {state: safe_spearman(full[i].mean(axis=0), common[i].mean(axis=0)) for i, state in enumerate(STATES)},
            "state_group_mean_split_half_spearman": {state: safe_spearman(first[i].mean(axis=0), second[i].mean(axis=0)) for i, state in enumerate(STATES)},
        }
    return {
        "experiment": "57-subject HCP PED and O-information triplet replication",
        "states": list(STATES),
        "n_subjects": int(red.shape[1]),
        "n_triplets": int(red.shape[2]),
        "sequence_length_by_state": {state: sorted(set(int(value) for value in lengths[index])) for index, state in enumerate(STATES)},
        "estimators": {
            "ped": "author-matched mean/z-score sign binarization plus discrete informative Hsx/SxPID; Red and seven Syn atoms in native bits; norm=False",
            "oinfo": "HOI-compatible rank Gaussian-copula entropy with finite-sample bias correction, in bits",
            "primary": "full available sequence",
            "common_length_sensitivity": f"first {COMMON_LENGTH} samples, independently centered/estimated",
            "split_half_sensitivity": "each half independently centered/estimated",
        },
        "ped_nonnegativity": {"tolerance_bits": PED_TOLERANCE_BITS, "minimum_partial_atom_bits": float(minimum_atom), "tolerance_negative_atom_count": int(tolerance_negative_count), "significant_violation_count": 0},
        "aggregate_metrics": aggregates,
        "top_triplets": {"ped_redundancy": top_rows(red, names), "ped_synergy": top_rows(syn, names), "oinfo_most_redundant": top_rows(oinfo, names), "oinfo_most_synergistic": top_rows(oinfo, names, largest=False)},
        "oinfo_sign": {state: {"minimum_group_mean_bits": float(signed_group_mean[i].min()), "maximum_group_mean_bits": float(signed_group_mean[i].max()), "synergy_dominated_group_triplet_count": int(np.sum(signed_group_mean[i] < 0.0))} for i, state in enumerate(STATES)},
        "ped_balance": {
            state: {
                "redundancy_dominated_group_triplet_count": int(np.sum(ped_balance[i] > 0.0)),
                "synergy_dominated_group_triplet_count": int(np.sum(ped_balance[i] < 0.0)),
                "most_redundancy_dominated_triplet": str(names[int(np.argmax(ped_balance[i]))]),
                "most_redundancy_dominated_balance_bits": float(np.max(ped_balance[i])),
                "most_synergy_dominated_triplet": str(names[int(np.argmin(ped_balance[i]))]),
                "most_synergy_dominated_balance_bits": float(np.min(ped_balance[i])),
            }
            for i, state in enumerate(STATES)
        },
        "robustness": robustness,
        "cross_metric": {
            "ped_red_vs_oinfo_group_triplet_spearman": {state: safe_spearman(red[i].mean(axis=0), oinfo[i].mean(axis=0)) for i, state in enumerate(STATES)},
            "ped_balance_red_minus_syn_vs_oinfo_group_triplet_spearman": {state: safe_spearman((red[i] - syn[i]).mean(axis=0), oinfo[i].mean(axis=0)) for i, state in enumerate(STATES)},
        },
        "leave_one_subject_out_winners": {"ped_redundancy": winner_stability(red, names), "ped_synergy": winner_stability(syn, names), "oinfo_most_redundant": winner_stability(oinfo, names)},
    }


def configure_style() -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 7.0, "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})


def plot_state_overview(data: Mapping[str, np.ndarray], output: Path) -> None:
    configure_style()
    metrics = ((data["ped_red_full"].mean(axis=2), "PED redundancy (bits)", "#B35C27"), (data["ped_syn_full"].mean(axis=2), "PED synergy (bits)", "#25766C"), (data["oinfo_full"].mean(axis=2), "Signed O-information (bits)", "#406CA8"))
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.3), constrained_layout=True)
    for panel, (axis, (values, ylabel, color)) in enumerate(zip(axes, metrics, strict=True)):
        positions = np.arange(len(STATES))
        box = axis.boxplot(values.T, positions=positions, widths=0.58, patch_artist=True, showfliers=False)
        for patch in box["boxes"]:
            patch.set(facecolor=color, alpha=0.24, edgecolor=color, linewidth=0.8)
        for element in ("whiskers", "caps", "medians"):
            for artist in box[element]:
                artist.set(color=color, linewidth=0.8)
        axis.scatter(positions, values.mean(axis=1), s=17, color=color, edgecolor="white", linewidth=0.4, zorder=3)
        if panel == 2:
            axis.axhline(0.0, color="#777777", linewidth=0.7, linestyle="--")
        axis.set_xticks(positions, STATE_LABELS, rotation=42, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(chr(ord("a") + panel), loc="left", fontweight="bold")
        axis.axvline(0.5, color="#B8B8B8", linewidth=0.7)
    figure.savefig(output, dpi=400, bbox_inches="tight")
    plt.close(figure)


def plot_top_heatmaps(data: Mapping[str, np.ndarray], names: np.ndarray, output: Path) -> None:
    configure_style()
    matrices = ((data["ped_red_full"].mean(axis=1), "PED Red", "Oranges"), (data["ped_syn_full"].mean(axis=1), "PED Syn", "viridis"), (data["oinfo_full"].mean(axis=1), "O-information", "coolwarm"))
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 5.0), constrained_layout=True)
    for panel, (axis, (matrix, label, cmap)) in enumerate(zip(axes, matrices, strict=True)):
        score = np.abs(matrix).mean(axis=0) if panel == 2 else matrix.mean(axis=0)
        top = np.argsort(score)[::-1][:12]
        if panel == 2:
            bound = float(np.abs(matrix[:, top]).max())
            image = axis.imshow(matrix[:, top].T, aspect="auto", cmap=cmap, vmin=-bound, vmax=bound)
        else:
            image = axis.imshow(matrix[:, top].T, aspect="auto", cmap=cmap, vmin=0.0)
        axis.set_xticks(np.arange(len(STATES)), STATE_LABELS, rotation=42, ha="right")
        axis.set_yticks(np.arange(len(top)), [compact_name(str(names[index])) for index in top])
        axis.set_title(f"{chr(ord('a') + panel)}  {label}", loc="left", fontweight="bold")
        axis.set_xlabel("State")
        bar = figure.colorbar(image, ax=axis, location="top", shrink=0.74, pad=0.03)
        bar.set_label("bits")
    figure.savefig(output, dpi=400, bbox_inches="tight")
    plt.close(figure)


def plot_robustness(data: Mapping[str, np.ndarray], output: Path) -> None:
    configure_style()
    figure, axes = plt.subplots(2, 3, figsize=(10.8, 6.3), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.8, len(STATES)))
    for column, metric in enumerate(("ped_red", "ped_syn", "oinfo")):
        for row, (left_label, right_label, xlabel, ylabel) in enumerate((("full", "common", "Full sequence", f"First {COMMON_LENGTH}"), ("first", "second", "First half", "Second half"))):
            axis = axes[row, column]
            left, right = data[f"{metric}_{left_label}"], data[f"{metric}_{right_label}"]
            for index in range(len(STATES)):
                axis.scatter(left[index].mean(axis=0), right[index].mean(axis=0), s=11, alpha=0.7, color=colors[index], label=STATE_LABELS[index], edgecolor="none")
            minimum = float(min(left.mean(axis=1).min(), right.mean(axis=1).min()))
            maximum = float(max(left.mean(axis=1).max(), right.mean(axis=1).max()))
            padding = max(0.01, 0.06 * (maximum - minimum))
            lower, upper = minimum - padding, maximum + padding
            axis.plot([lower, upper], [lower, upper], color="#555555", linewidth=0.8, linestyle="--")
            axis.set(xlim=(lower, upper), ylim=(lower, upper), xlabel=f"{xlabel} (bits)", ylabel=f"{ylabel} (bits)")
            axis.set_title(f"{chr(ord('a') + row * 3 + column)}  {metric.replace('_', ' ').upper()}", loc="left", fontweight="bold")
    axes[1, 2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="State")
    figure.savefig(output, dpi=400, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    subjects = subject_ids()
    data, lengths, minimum_atom, tolerance_negative_count = compute_all(subjects, smoke=args.smoke, workers=args.workers)
    if args.smoke:
        print(json.dumps({"subject": str(subjects[0]), "lengths": lengths[:, 0].tolist(), "ped_red_range_bits": [float(data["ped_red_full"].min()), float(data["ped_red_full"].max())], "ped_syn_range_bits": [float(data["ped_syn_full"].min()), float(data["ped_syn_full"].max())], "oinfo_range_bits": [float(data["oinfo_full"].min()), float(data["oinfo_full"].max())], "minimum_partial_atom_bits": minimum_atom}, indent=2))
        return
    names = triplet_names(triplets())
    summary = metric_summary(data, names, lengths, minimum_atom, tolerance_negative_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "metrics.npz", states=np.asarray(STATES), subjects=subjects, triplets=names, sequence_lengths=lengths, common_length=np.asarray(COMMON_LENGTH), ped_tolerance_bits=np.asarray(PED_TOLERANCE_BITS), **{f"{key}_bits": value for key, value in data.items()})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_state_overview(data, args.output_dir / "state_overview.png")
    plot_top_heatmaps(data, names, args.output_dir / "top_triplet_heatmaps.png")
    plot_robustness(data, args.output_dir / "robustness.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
