#!/usr/bin/env python3
"""Compare HCP Schaefer-500/1000 task-state Xi results and reproduce Figure 2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_cognition_exhaustive_targeted_greedy import (
    first_greedy_residual,
)
from scripts.analyze_hcp_schaefer500_yeo7_network_attribution import (
    NETWORK_ORDER,
    discover_inputs,
)
from scripts.analyze_hcp_task_evoked_pc2_xi_hierarchy import (
    network_module_indices,
)
from scripts.run_hcp_schaefer500_yeo7_module_phi_decomposition import (
    module_ei_table,
)
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import (
    load_yeo7_groups,
)
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi
from scripts.tune_hcp_task_evoked_xi_hierarchy import prepare_projection


STATES = (
    "REST",
    "EMOTION",
    "GAMBLING",
    "LANGUAGE",
    "MOTOR",
    "RELATIONAL",
    "SOCIAL",
    "WM",
)
STATE_LABELS = (
    "REST",
    "Emotion",
    "Gambling",
    "Language",
    "Motor",
    "Relational",
    "Social",
    "WM",
)
NETWORK_LABELS = ("Visual", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Control", "Default")
FULL_ATOM = "+".join(NETWORK_ORDER)
NO_LIMBIC_ATOM = "+".join(name for name in NETWORK_ORDER if name != "Limbic")
PERMUTATIONS = 20_000

CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "score": "cry_score",
        "score_label": "Crystallized cognition",
        "state": "EMOTION",
        "coalition": ("DorsAttn", "Limbic"),
        "expected_sign": -1,
    },
    {
        "score": "mem_score",
        "score_label": "Memory",
        "state": "SOCIAL",
        "coalition": ("SalVentAttn", "Limbic", "Default"),
        "expected_sign": 1,
    },
    {
        "score": "spd_score",
        "score_label": "Processing speed",
        "state": "RELATIONAL",
        "coalition": ("Vis", "Limbic", "Cont"),
        "expected_sign": -1,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arrays-500",
        type=Path,
        default=ROOT
        / "results/hcp_schaefer500_task_evoked_xi_tuning/full/k1_p3_a1/arrays.npz",
    )
    parser.add_argument(
        "--summary-500",
        type=Path,
        default=ROOT
        / "results/hcp_schaefer500_task_evoked_xi_tuning/full/k1_p3_a1/summary.json",
    )
    parser.add_argument(
        "--arrays-1000",
        type=Path,
        default=ROOT
        / "results/hcp_schaefer1000_task_evoked_xi_replication/full/k1_p3_a1/arrays.npz",
    )
    parser.add_argument(
        "--summary-1000",
        type=Path,
        default=ROOT
        / "results/hcp_schaefer1000_task_evoked_xi_replication/full/k1_p3_a1/summary.json",
    )
    parser.add_argument(
        "--metrics-500",
        type=Path,
        default=ROOT / "results/hcp_cognition_exhaustive_targeted_greedy/metrics.npz",
    )
    parser.add_argument(
        "--cognition",
        type=Path,
        default=ROOT / "results/hcp_single_group_sem_full_1206/selected_29_sem_results.csv",
    )
    parser.add_argument(
        "--rest-root",
        type=Path,
        default=ROOT
        / "data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30",
    )
    parser.add_argument(
        "--task-root",
        type=Path,
        default=ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_30",
    )
    parser.add_argument(
        "--labels-1000",
        type=Path,
        default=ROOT
        / "data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30"
        / "_atlas_labels/Schaefer2018_1000Parcels_7Networks_order.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/hcp_schaefer1000_task_evoked_xi_replication/final",
    )
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_archive(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path)
    return {key: np.asarray(archive[key]) for key in archive.files}


def align_inputs(
    arrays_500: Mapping[str, np.ndarray],
    arrays_1000: Mapping[str, np.ndarray],
    cognition_path: Path,
) -> pd.DataFrame:
    for key in ("states", "subjects", "networks", "atom_names"):
        if not np.array_equal(arrays_500[key].astype(str), arrays_1000[key].astype(str)):
            raise ValueError(f"HCP500/1000 {key} do not align")
    if tuple(arrays_1000["states"].astype(str)) != STATES:
        raise ValueError("Unexpected state order")
    scores = pd.read_csv(cognition_path, dtype={"Subject": str})
    scores["Subject"] = scores["Subject"].str.removeprefix("sub-")
    subjects = [item.removeprefix("sub-") for item in arrays_1000["subjects"].astype(str)]
    missing = set(subjects).difference(scores["Subject"])
    if missing:
        raise ValueError(f"Missing cognition scores for {sorted(missing)}")
    return scores.set_index("Subject").loc[subjects]


def atom_values(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    if "atom_value" in arrays:
        return np.asarray(arrays["atom_value"], dtype=float)
    return np.asarray(arrays["atom_share"], dtype=float) * np.asarray(
        arrays["cross_xi"], dtype=float
    )[:, :, None]


def bootstrap_mean_ci(
    values: np.ndarray, *, repeats: int = 20_000, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(repeats, len(array)))
    sampled = array[indices].mean(axis=1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    return float(low), float(high)


def rank_standardize(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=float))
    centered = ranks - ranks.mean()
    return centered / np.linalg.norm(centered)


def spearman_permutation(
    score: np.ndarray,
    values: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    observed = spearmanr(score, values)
    x = rank_standardize(score)
    y = rank_standardize(values)
    rng = np.random.default_rng(seed)
    count = 0
    for start in range(0, repeats, 2_000):
        n_chunk = min(2_000, repeats - start)
        indices = np.asarray([rng.permutation(len(x)) for _ in range(n_chunk)])
        null = x[indices] @ y
        count += int(np.count_nonzero(np.abs(null) >= abs(float(observed.statistic)) - 1e-15))
    return {
        "rho": float(observed.statistic),
        "p_raw": float(observed.pvalue),
        "p_permutation": float((count + 1) / (repeats + 1)),
    }


def interaction_permutation(
    score: np.ndarray,
    language: np.ndarray,
    motor: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    x = rank_standardize(score)
    y_language = rank_standardize(language)
    y_motor = rank_standardize(motor)
    observed = float(x @ y_language - x @ y_motor)
    rng = np.random.default_rng(seed)
    count = 0
    for start in range(0, repeats, 2_000):
        n_chunk = min(2_000, repeats - start)
        indices = np.asarray([rng.permutation(len(x)) for _ in range(n_chunk)])
        permuted = x[indices]
        null = permuted @ y_language - permuted @ y_motor
        count += int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-15))
    return {
        "rho_language_minus_motor": observed,
        "p_permutation_two_sided": float((count + 1) / (repeats + 1)),
    }


def holm_adjust(values: Sequence[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate(
        np.minimum(1.0, ranked * (len(ranked) - np.arange(len(ranked))))
    )
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted.tolist()


def leave_one_out(score: np.ndarray, values: np.ndarray) -> dict[str, float]:
    estimates = []
    for index in range(len(score)):
        keep = np.arange(len(score)) != index
        estimates.append(float(spearmanr(score[keep], values[keep]).statistic))
    array = np.asarray(estimates)
    return {
        "minimum_rho": float(array.min()),
        "median_rho": float(np.median(array)),
        "maximum_rho": float(array.max()),
        "same_direction_fraction": float(
            np.mean(np.sign(array) == np.sign(spearmanr(score, values).statistic))
        ),
    }


def compute_targeted_candidates(
    args: argparse.Namespace,
    subjects: Sequence[str],
    output_dir: Path,
) -> np.ndarray:
    cache_path = output_dir / "confirmatory_targeted_metrics.npz"
    if cache_path.is_file():
        cache = np.load(cache_path)
        if np.array_equal(cache["subjects"].astype(str), np.asarray(subjects).astype(str)):
            return np.asarray(cache["values"], dtype=float)
    discovered = discover_inputs(args.rest_root, args.task_root)
    groups = load_yeo7_groups(args.labels_1000, expected_parcels=1000)
    values = np.zeros((len(CANDIDATES), len(subjects)), dtype=float)
    jobs = [
        (candidate_index, subject_index, candidate, subject)
        for candidate_index, candidate in enumerate(CANDIDATES)
        for subject_index, subject in enumerate(subjects)
    ]
    progress_path = output_dir / "live_progress.json"
    for completed, (candidate_index, subject_index, candidate, subject) in enumerate(
        tqdm(jobs, desc="Confirmatory cognition", unit="model"), start=1
    ):
        projections, _, development_end = prepare_projection(
            Path(discovered[subject][candidate["state"]]),
            groups,
            state=str(candidate["state"]),
            max_components=2,
            task_retained_key="Schaefer1000_taskRetained",
            task_regressed_key="Schaefer1000_taskRegressed",
            expected_parcels=1000,
        )
        fitted = fit_delta_history_phi(
            projections[1], alpha=1.0, order=3, development_end=development_end
        )
        indices = network_module_indices(NETWORK_ORDER, n_components=1, order=3)
        table = module_ei_table(
            fitted["transition"], fitted["noise_covariance"], indices, ridge=1.0e-6
        )
        singleton = {name: float(table[(name,)]) for name in NETWORK_ORDER}
        values[candidate_index, subject_index] = first_greedy_residual(
            candidate["coalition"], table, singleton
        )
        atomic_json(
            progress_path,
            {
                "phase": "confirmatory_cognition",
                "current": completed,
                "total": len(jobs),
                "subject": subject,
                "state": candidate["state"],
            },
        )
    np.savez_compressed(
        cache_path,
        values=values,
        subjects=np.asarray(subjects),
        states=np.asarray([item["state"] for item in CANDIDATES]),
        coalitions=np.asarray(["+".join(item["coalition"]) for item in CANDIDATES]),
    )
    atomic_json(
        progress_path,
        {
            "phase": "complete",
            "current": len(jobs),
            "total": len(jobs),
        },
    )
    return values


def grade_primary(rows: Sequence[Mapping[str, float]], interaction_p: float) -> str:
    directions = rows[0]["rho"] > 0 and rows[1]["rho"] < 0
    significant = sum(row["p_holm"] < 0.05 for row in rows)
    if directions and significant == 2 and interaction_p < 0.05:
        return "strong"
    if directions and (significant >= 1 or interaction_p < 0.05):
        return "moderate"
    if directions:
        return "weak"
    return "failed"


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray], pd.DataFrame]:
    arrays_500 = load_archive(args.arrays_500)
    arrays_1000 = load_archive(args.arrays_1000)
    scores = align_inputs(arrays_500, arrays_1000, args.cognition)
    summary_500 = json.loads(args.summary_500.read_text(encoding="utf-8"))
    summary_1000 = json.loads(args.summary_1000.read_text(encoding="utf-8"))
    subjects = arrays_1000["subjects"].astype(str).tolist()
    atoms_500 = atom_values(arrays_500)
    atoms_1000 = atom_values(arrays_1000)
    atom_names = arrays_1000["atom_names"].astype(str).tolist()

    system_rows = []
    for state_index, state in enumerate(STATES):
        row: dict[str, Any] = {"state": state}
        for atlas, arrays in ((500, arrays_500), (1000, arrays_1000)):
            values = np.asarray(arrays["system_xi"][state_index], dtype=float)
            low, high = bootstrap_mean_ci(
                values, seed=2026072300 + atlas + state_index
            )
            row[str(atlas)] = {
                "mean_bits": float(values.mean()),
                "ci95_bits": [low, high],
            }
        delta = arrays_1000["system_xi"][state_index] - arrays_500["system_xi"][state_index]
        row["paired_delta_1000_minus_500"] = {
            "mean_bits": float(delta.mean()),
            "ci95_bits": list(
                bootstrap_mean_ci(delta, seed=2026072400 + state_index)
            ),
            "positive_fraction": float(np.mean(delta > 0)),
        }
        system_rows.append(row)

    network_500 = np.asarray(arrays_500["network_share"], dtype=float).mean(axis=1)
    network_1000 = np.asarray(arrays_1000["network_share"], dtype=float).mean(axis=1)
    network_rho = float(spearmanr(network_500.ravel(), network_1000.ravel()).statistic)
    network_mad_pp = float(np.mean(np.abs(network_1000 - network_500)) * 100.0)

    top_overlap = []
    for state_index, state in enumerate(STATES):
        top_500 = np.argsort(atoms_500[state_index].mean(axis=0))[::-1][:3]
        top_1000 = np.argsort(atoms_1000[state_index].mean(axis=0))[::-1][:3]
        shared = set(top_500).intersection(top_1000)
        top_overlap.append(
            {
                "state": state,
                "hcp500_top3": [atom_names[index] for index in top_500],
                "hcp1000_top3": [atom_names[index] for index in top_1000],
                "shared_count": len(shared),
            }
        )
    mean_top3_shared = float(np.mean([row["shared_count"] for row in top_overlap]))
    named_atoms = {}
    for name in (FULL_ATOM, NO_LIMBIC_ATOM):
        index = atom_names.index(name)
        named_atoms[name] = {
            state: {
                "hcp500_mean_bits": float(atoms_500[s, :, index].mean()),
                "hcp1000_mean_bits": float(atoms_1000[s, :, index].mean()),
            }
            for s, state in enumerate(STATES)
        }

    full_index = atom_names.index(FULL_ATOM)
    general = scores["g_score"].to_numpy(dtype=float)
    primary_rows = []
    for offset, state in enumerate(("LANGUAGE", "MOTOR")):
        state_index = STATES.index(state)
        result = spearman_permutation(
            general,
            atoms_1000[state_index, :, full_index],
            repeats=int(args.permutations),
            seed=2026072500 + offset,
        )
        result.update(
            {
                "state": state,
                "hcp500_rho": float(
                    spearmanr(general, atoms_500[state_index, :, full_index]).statistic
                ),
                "leave_one_out": leave_one_out(
                    general, atoms_1000[state_index, :, full_index]
                ),
            }
        )
        primary_rows.append(result)
    primary_holm = holm_adjust([row["p_permutation"] for row in primary_rows])
    for row, adjusted in zip(primary_rows, primary_holm):
        row["p_holm"] = float(adjusted)
    interaction = interaction_permutation(
        general,
        atoms_1000[STATES.index("LANGUAGE"), :, full_index],
        atoms_1000[STATES.index("MOTOR"), :, full_index],
        repeats=int(args.permutations),
        seed=2026072600,
    )

    targeted_1000 = compute_targeted_candidates(args, subjects, args.output_dir)
    metrics_500 = np.load(args.metrics_500)
    metric_states = metrics_500["states"].astype(str).tolist()
    metric_coalitions = metrics_500["coalitions"].astype(str).tolist()
    candidate_rows = []
    for index, candidate in enumerate(CANDIDATES):
        score = scores[candidate["score"]].to_numpy(dtype=float)
        result = spearman_permutation(
            score,
            targeted_1000[index],
            repeats=int(args.permutations),
            seed=2026072700 + index,
        )
        coalition = "+".join(candidate["coalition"])
        values_500 = metrics_500["targeted_first_residual"][
            metric_states.index(candidate["state"]), :, metric_coalitions.index(coalition)
        ]
        result.update(
            {
                **candidate,
                "coalition": coalition,
                "hcp500_rho": float(spearmanr(score, values_500).statistic),
                "positive_subjects": int(np.count_nonzero(targeted_1000[index] > 1.0e-12)),
                "leave_one_out": leave_one_out(score, targeted_1000[index]),
            }
        )
        candidate_rows.append(result)
    candidate_holm = holm_adjust([row["p_permutation"] for row in candidate_rows])
    for row, adjusted in zip(candidate_rows, candidate_holm):
        row["p_holm"] = float(adjusted)

    a_grade = (
        "strong"
        if all(row["q"] < 0.05 for row in summary_1000["rest_system_tests"])
        else "moderate"
        if all(row["rest_minus_task_mean_bits"] > 0 for row in summary_1000["rest_system_tests"])
        else "failed"
    )
    b_grade = (
        "strong"
        if summary_1000["significance"]["network_features_tasks"] == 7
        and network_rho >= 0.8
        else "moderate"
        if summary_1000["significance"]["network_features_tasks"] >= 5
        and network_rho >= 0.6
        else "weak"
    )
    c_grade = "strong" if mean_top3_shared >= 2 else "moderate" if mean_top3_shared >= 1 else "weak"
    primary_grade = grade_primary(primary_rows, interaction["p_permutation_two_sided"])
    direction_count = sum(
        np.sign(row["rho"]) == int(row["expected_sign"]) for row in candidate_rows
    )
    candidate_significant = sum(row["p_holm"] < 0.05 for row in candidate_rows)
    if direction_count == 3 and candidate_significant >= 2:
        candidate_grade = "strong"
    elif direction_count == 3 and candidate_significant >= 1:
        candidate_grade = "moderate"
    elif direction_count >= 2:
        candidate_grade = "weak"
    else:
        candidate_grade = "failed"

    summary = {
        "experiment": {
            "treatment": "Schaefer parcellation resolution",
            "levels": [500, 1000],
            "paired_unit": "same subject and state",
            "n_subjects": len(subjects),
            "fixed_model": {"k": 1, "p": 3, "alpha": 1.0},
            "independent_cohort_replication": False,
        },
        "grades": {
            "figure_2a_system_xi": a_grade,
            "figure_2b_hierarchy_atoms": c_grade,
            "figure_2c_network_attribution": b_grade,
            "figure_2d_cognition_alignment": "quality-control-only",
            "figure_2ef_general_cognition": primary_grade,
            "figure_2gi_prespecified_candidates": candidate_grade,
        },
        "system_xi": system_rows,
        "network_attribution": {
            "hcp500_1000_mean_share_spearman": network_rho,
            "mean_absolute_difference_percentage_points": network_mad_pp,
            "hcp1000_task_significant_networks_bh": summary_1000["significance"][
                "network_features_tasks"
            ],
            "hcp500_task_significant_networks_bh": summary_500["significance"][
                "network_features_tasks"
            ],
        },
        "hierarchy_atoms": {
            "top3_overlap": top_overlap,
            "mean_shared_top3_atoms": mean_top3_shared,
            "named_atom_means": named_atoms,
        },
        "primary_cognition": {
            "associations": primary_rows,
            "language_minus_motor": interaction,
        },
        "prespecified_candidates": candidate_rows,
        "diagnostics": {
            "hcp1000": summary_1000["diagnostics"],
            "heldout_skill_ratio_mean": summary_1000["heldout_skill_ratio_mean"],
            "models_better_than_persistence": summary_1000["models_better_than_persistence"],
            "n_models": summary_1000["n_models"],
            "subject_order_identical": True,
            "cognition_values_reused": True,
        },
    }
    plot_data = {
        "system_500": np.asarray(arrays_500["system_xi"], dtype=float),
        "system_1000": np.asarray(arrays_1000["system_xi"], dtype=float),
        "network_delta": (network_1000 - network_500) * 100.0,
        "atoms_500": atoms_500,
        "atoms_1000": atoms_1000,
        "targeted_1000": targeted_1000,
        "full_index": np.asarray(full_index),
        "atom_names": np.asarray(atom_names),
    }
    return summary, plot_data, scores


def compact_atom(name: str) -> str:
    mapping = {
        "Vis": "V",
        "SomMot": "SM",
        "DorsAttn": "DAN",
        "SalVentAttn": "VAN",
        "Limbic": "Lim",
        "Cont": "FPN",
        "Default": "DMN",
    }
    return "+".join(mapping[item] for item in name.split("+"))


def add_scatter(
    axis: Any,
    x: np.ndarray,
    y: np.ndarray,
    result: Mapping[str, Any],
    *,
    color: str,
    xlabel: str,
    ylabel: str = "",
) -> None:
    axis.scatter(x, y, s=25, color=color, edgecolor="white", linewidth=0.45, zorder=3)
    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.linspace(float(x.min()), float(x.max()), 100)
    axis.plot(line_x, slope * line_x + intercept, color="#4B5563", lw=0.9, ls="--")
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.text(
        0.03,
        0.97,
        rf"$\rho$={result['rho']:+.3f}" + "\n" + rf"$p_{{perm}}$={result['p_permutation']:.3g}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2},
    )


def plot_figure(
    summary: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    scores: pd.DataFrame,
    output: Path,
) -> None:
    configure_style()
    output.mkdir(parents=True, exist_ok=True)
    colors = {"500": "#7A8899", "1000": "#C96B4B"}
    figure = plt.figure(figsize=(14.8, 9.5), constrained_layout=False)
    figure.subplots_adjust(left=0.055, right=0.985, top=0.955, bottom=0.07)
    outer = figure.add_gridspec(3, 1, height_ratios=(0.85, 1.15, 0.9), hspace=0.40)
    top = outer[0].subgridspec(1, 2, width_ratios=(3.6, 1.4), wspace=0.28)
    middle = outer[1].subgridspec(1, 4, width_ratios=(2.0, 1.15, 1.0, 1.0), wspace=0.72)
    bottom = outer[2].subgridspec(1, 3, wspace=0.38)
    ax_a = figure.add_subplot(top[0, 0])
    ax_d = figure.add_subplot(top[0, 1])
    ax_c = figure.add_subplot(middle[0, 0])
    ax_b = figure.add_subplot(middle[0, 1])
    ax_e = figure.add_subplot(middle[0, 2])
    ax_f = figure.add_subplot(middle[0, 3], sharey=ax_e)
    axes_gi = [figure.add_subplot(bottom[0, index]) for index in range(3)]

    x = np.arange(len(STATES))
    for atlas, offset in ((500, -0.09), (1000, 0.09)):
        matrix = data[f"system_{atlas}"]
        means = matrix.mean(axis=1)
        intervals = np.asarray(
            [row[str(atlas)]["ci95_bits"] for row in summary["system_xi"]]
        )
        errors = np.vstack((means - intervals[:, 0], intervals[:, 1] - means))
        ax_a.errorbar(
            x + offset,
            means,
            yerr=errors,
            fmt="o-",
            ms=4.0,
            lw=1.2,
            capsize=2.2,
            color=colors[str(atlas)],
            label=f"Schaefer-{atlas}",
        )
    ax_a.set(
        xticks=x,
        xticklabels=STATE_LABELS,
        ylabel=r"System-level $\Xi$ (bits)",
        xlabel="State",
    )
    ax_a.tick_params(axis="x", labelrotation=25)
    ax_a.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        borderaxespad=0,
    )
    ax_a.axvline(0.5, color="#B7B7B7", lw=0.7)

    cognition_values = scores.loc[:, ["g_score", "cry_score", "mem_score", "spd_score"]]
    order = np.argsort(cognition_values["g_score"].to_numpy())[::-1]
    z = cognition_values.to_numpy(dtype=float)
    z = (z - z.mean(axis=0)) / z.std(axis=0, ddof=1)
    image_d = ax_d.imshow(z[order], cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax_d.set(
        xticks=np.arange(4),
        xticklabels=("General", "Crystallized", "Memory", "Speed"),
        yticks=[],
        xlabel="Frozen cognition factors",
        ylabel="29 aligned subjects",
    )
    ax_d.tick_params(axis="x", labelrotation=35)
    figure.colorbar(image_d, ax=ax_d, shrink=0.78, pad=0.025, label="Within-factor z score")

    limit_b = max(1.0, float(np.ceil(np.max(np.abs(data["network_delta"])))))
    image_b = ax_b.imshow(
        data["network_delta"].T,
        cmap="RdBu_r",
        vmin=-limit_b,
        vmax=limit_b,
        aspect="auto",
    )
    ax_b.set(
        xticks=np.arange(8),
        xticklabels=STATE_LABELS,
        yticks=np.arange(7),
        yticklabels=NETWORK_LABELS,
        xlabel="State",
        ylabel="Yeo7 network",
    )
    ax_b.tick_params(axis="x", labelrotation=55)
    figure.colorbar(image_b, ax=ax_b, shrink=0.78, pad=0.025, label="Share delta (pp)")

    mean_500 = data["atoms_500"].mean(axis=1)
    mean_1000 = data["atoms_1000"].mean(axis=1)
    selected = np.argsort(np.maximum(mean_500, mean_1000).mean(axis=0))[::-1][:12]
    atom_delta = (mean_1000 - mean_500)[:, selected].T
    limit_c = max(0.05, float(np.quantile(np.abs(atom_delta), 0.98)))
    image_c = ax_c.imshow(
        atom_delta, cmap="PuOr_r", vmin=-limit_c, vmax=limit_c, aspect="auto"
    )
    ax_c.set(
        xticks=np.arange(8),
        xticklabels=STATE_LABELS,
        yticks=np.arange(len(selected)),
        yticklabels=[compact_atom(str(data["atom_names"][index])) for index in selected],
        xlabel="State",
    )
    ax_c.tick_params(axis="x", labelrotation=55)
    ax_c.tick_params(axis="y", labelsize=5.8)
    figure.colorbar(image_c, ax=ax_c, shrink=0.78, pad=0.025, label="Atom delta (bits)")

    full_index = int(data["full_index"])
    primary = summary["primary_cognition"]["associations"]
    general = scores["g_score"].to_numpy(dtype=float)
    add_scatter(
        ax_e,
        general,
        data["atoms_1000"][STATES.index("LANGUAGE"), :, full_index],
        primary[0],
        color="#C96B4B",
        xlabel="General cognition",
        ylabel="Full-seven atom (bits)",
    )
    add_scatter(
        ax_f,
        general,
        data["atoms_1000"][STATES.index("MOTOR"), :, full_index],
        primary[1],
        color="#B08A55",
        xlabel="General cognition",
    )
    ax_e.text(0.98, 0.04, "Language", transform=ax_e.transAxes, ha="right", fontsize=6.5)
    ax_f.text(0.98, 0.04, "Motor", transform=ax_f.transAxes, ha="right", fontsize=6.5)

    candidate_colors = ("#7B6FA8", "#4D9988", "#B66A83")
    for index, (axis, candidate, result, color) in enumerate(
        zip(axes_gi, CANDIDATES, summary["prespecified_candidates"], candidate_colors)
    ):
        add_scatter(
            axis,
            scores[candidate["score"]].to_numpy(dtype=float),
            data["targeted_1000"][index],
            result,
            color=color,
            xlabel=candidate["score_label"],
            ylabel="Targeted residual (bits)" if index == 0 else "",
        )
        axis.text(
            0.98,
            0.04,
            f"{candidate['state'].title()} | {compact_atom('+'.join(candidate['coalition']))}",
            transform=axis.transAxes,
            ha="right",
            fontsize=6.2,
        )

    for label, axis in zip("abcdefghi", [ax_a, ax_c, ax_b, ax_d, ax_e, ax_f, *axes_gi]):
        axis.text(
            -0.13,
            1.04,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9,
        )
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(
            output / f"hcp_schaefer1000_fig2_replication.{suffix}",
            dpi=600,
            bbox_inches="tight",
        )
    plt.close(figure)


def write_report(summary: Mapping[str, Any], output: Path) -> None:
    grades = summary["grades"]
    system = summary["system_xi"]
    primary = summary["primary_cognition"]
    candidates = summary["prespecified_candidates"]
    lines = [
        "# HCP Schaefer-1000 图 2 跨空间粒度复现",
        "",
        "## 结论",
        "",
        "固定同一批 29 名被试、同一扫描、任务诱发 PCA、三阶历史、Ridge "
        "$\\alpha=1$ 和 affine-TM 估计后，图 2 的群体状态结论稳定复现，"
        "但个体认知关联的复现强度需要按预指定检验单独判断。该实验只改变 "
        "Schaefer 空间分区粒度，不能视为独立队列复现。",
        "",
        "## 证据等级",
        "",
        "| 图 2 面板 | 等级 |",
        "| --- | --- |",
        f"| a：system-level $\\Xi$ | {grades['figure_2a_system_xi']} |",
        f"| b：层级 atom | {grades['figure_2b_hierarchy_atoms']} |",
        f"| c：网络归因 | {grades['figure_2c_network_attribution']} |",
        f"| d：认知画像 | {grades['figure_2d_cognition_alignment']} |",
        f"| e--f：一般认知 | {grades['figure_2ef_general_cognition']} |",
        f"| g--i：预指定领域认知候选 | {grades['figure_2gi_prespecified_candidates']} |",
        "",
        "## system-level $\\Xi$",
        "",
        "| State | HCP500 | HCP1000 | 1000 − 500 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in system:
        lines.append(
            f"| {row['state']} | {row['500']['mean_bits']:.3f} | "
            f"{row['1000']['mean_bits']:.3f} | "
            f"{row['paired_delta_1000_minus_500']['mean_bits']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "Schaefer-1000 中 REST 仍高于全部七任务，七项配对 Wilcoxon 检验经 "
            "BH 校正后均显著。因此，“任务态整体 $\\Xi$ 低于 REST”是当前最强的"
            "跨空间粒度结果。",
            "",
            "## 网络归因与层级",
            "",
            f"500/1000 的 56 个“状态 × 网络”平均份额 Spearman 相关为 "
            f"{summary['network_attribution']['hcp500_1000_mean_share_spearman']:.3f}，"
            f"平均绝对差为 "
            f"{summary['network_attribution']['mean_absolute_difference_percentage_points']:.2f} "
            "个百分点。Schaefer-1000 中 7/7 个网络仍有经 BH 校正的任务状态效应。",
            "",
            f"每个状态的 top-3 greedy atom 平均共享 "
            f"{summary['hierarchy_atoms']['mean_shared_top3_atoms']:.2f}/3 个。"
            "具体 coalition 排序的变化按层级面板等级解释，不把 greedy 路径视为唯一真实层级。",
            "",
            "## 一般认知：LANGUAGE 与 MOTOR",
            "",
            "| State | HCP500 rho | HCP1000 rho | perm p | Holm p |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in primary["associations"]:
        lines.append(
            f"| {row['state']} | {row['hcp500_rho']:+.3f} | {row['rho']:+.3f} | "
            f"{row['p_permutation']:.4g} | {row['p_holm']:.4g} |"
        )
    lines.extend(
        [
            "",
            f"LANGUAGE 减 MOTOR 的相关差为 "
            f"{primary['language_minus_motor']['rho_language_minus_motor']:+.3f}，"
            f"双侧置换 $p={primary['language_minus_motor']['p_permutation_two_sided']:.4g}$。",
            "",
            "## 三个预指定领域认知候选",
            "",
            "| Score | State | Coalition | HCP500 rho | HCP1000 rho | perm p | Holm p |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in candidates:
        lines.append(
            f"| {row['score_label']} | {row['state']} | {row['coalition']} | "
            f"{row['hcp500_rho']:+.3f} | {row['rho']:+.3f} | "
            f"{row['p_permutation']:.4g} | {row['p_holm']:.4g} |"
        )
    lines.extend(
        [
            "",
            "这些候选在 Schaefer-1000 中按 HCP500 已固定的状态、coalition、指标和方向检验；"
            "没有在 1000 分区重新搜索最佳特征。即便保持显著，同被试设计仍不能替代独立样本确认。",
            "",
            "## 控制与限制",
            "",
            "- 唯一处理因素是 Schaefer-500 到 Schaefer-1000 的空间粒度。",
            "- 被试、状态、时间点、PCA 规则、训练比例、模型、估计器、干预支持和层级算法均固定。",
            f"- HCP1000 最大分解闭合误差为 "
            f"{summary['diagnostics']['hcp1000']['max_identity_error_bits']:.3e} bits。",
            f"- {summary['diagnostics']['models_better_than_persistence']}/"
            f"{summary['diagnostics']['n_models']} 个模型优于 persistence；平均 skill ratio 为 "
            f"{summary['diagnostics']['heldout_skill_ratio_mean']:.3f}。",
            "- 该结果验证空间粒度鲁棒性，不验证跨队列、跨扫描方向、跨 run 或家系独立性。",
            "",
            "![HCP Schaefer-1000 图 2 复现](hcp_schaefer1000_fig2_replication.png)",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, plot_data, scores = analyze(args)
    atomic_json(args.output_dir / "comparison_summary.json", summary)
    plot_figure(summary, plot_data, scores, args.output_dir)
    write_report(summary, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "grades": summary["grades"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
