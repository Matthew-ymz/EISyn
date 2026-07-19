#!/usr/bin/env python3
"""Run k=2 Yeo7 PCA Xi attribution and hierarchy comparisons across HCP states."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.decomposition import PCA
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_schaefer500_yeo7_network_attribution import (
    DEFAULT_REST_ROOT,
    DEFAULT_TASK_ROOT,
    NETWORK_LABELS,
    NETWORK_ORDER,
    discover_inputs,
    exact_shapley,
    load_rest_series,
)
from scripts.analyze_hcp_task_evoked_pc1_phi_attribution import load_task_pair
from scripts.phi_hierarchy import greedy_phi_atoms
from scripts.run_hcp_lausanne_phi_eid_pilot import ei_for_source_indices
from scripts.run_hcp_schaefer500_all_tasks_phi import DISPLAY_NAMES, development_end_for_length
from scripts.run_hcp_schaefer500_yeo7_module_phi_decomposition import module_ei_table
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import default_yeo7_labels, load_yeo7_groups
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi


STATES = ("REST", "EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")
TASKS = STATES[1:]
N_COMPONENTS = 2
ORDER = 5
ALPHA = 10.0
DEFAULT_OUTPUT = ROOT / "results" / "hcp_schaefer500_task_evoked_pc2_xi_hierarchy"
CONFIG_ID = "hcp_task_evoked_pc2_xi_hierarchy_v1_p5_alpha10"


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_cache(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("config_id") == CONFIG_ID:
            rows[(str(row["subject"]), str(row["state"]))] = row
    return rows


def network_module_indices(
    network_names: Sequence[str], *, n_components: int, order: int
) -> dict[str, list[int]]:
    names = tuple(network_names)
    state_dimension = len(names) * int(n_components)
    return {
        name: [
            lag * state_dimension + network_index * int(n_components) + component
            for lag in range(int(order))
            for component in range(int(n_components))
        ]
        for network_index, name in enumerate(names)
    }


def fit_project_network_pca(
    fitting_signal: np.ndarray,
    projection_signal: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    *,
    development_end: int,
    n_components: int,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    projected = []
    explained: dict[str, list[float]] = {}
    for network, indices in groups.items():
        selected = np.asarray(indices, dtype=int)
        model = PCA(n_components=int(n_components), svd_solver="full").fit(
            np.asarray(fitting_signal[:development_end, selected], dtype=float)
        )
        projected.append(model.transform(np.asarray(projection_signal[:, selected], dtype=float)))
        explained[network] = [float(value) for value in model.explained_variance_ratio_]
    return np.concatenate(projected, axis=1), explained


def decompose_transition(
    transition: np.ndarray,
    noise_covariance: np.ndarray,
    network_names: Sequence[str],
    *,
    n_components: int,
    order: int,
) -> dict[str, Any]:
    names = tuple(network_names)
    indices = network_module_indices(names, n_components=n_components, order=order)
    table = module_ei_table(transition, noise_covariance, indices, ridge=1.0e-6)
    module_ei = {name: float(table[(name,)]) for name in names}
    scalar_ei = {
        name: float(
            sum(
                ei_for_source_indices(transition, noise_covariance, [index], ridge=1.0e-6)
                for index in indices[name]
            )
        )
        for name in names
    }
    within_xi = {name: float(module_ei[name] - scalar_ei[name]) for name in names}

    def cross_value(subset: tuple[str, ...]) -> float:
        if not subset:
            return 0.0
        ordered = tuple(name for name in names if name in subset)
        return float(table[ordered] - sum(module_ei[name] for name in ordered))

    cross_shapley = exact_shapley(names, cross_value)
    network_attribution = {
        name: float(within_xi[name] + cross_shapley[name]) for name in names
    }
    full = tuple(names)
    joint_ei = float(table[full])
    scalar_ei_sum = float(sum(scalar_ei.values()))
    system_xi = float(joint_ei - scalar_ei_sum)
    cross_xi = float(joint_ei - sum(module_ei.values()))
    atoms = greedy_phi_atoms(full, table, singleton_ei=module_ei)
    atom_rows = [
        {
            "sources": list(atom.sources),
            "value": float(atom.value),
            "kind": str(atom.kind),
            "depth": int(atom.depth),
        }
        for atom in atoms
        if float(atom.value) > 1.0e-10 and len(atom.sources) >= 2
    ]
    return {
        "joint_ei": joint_ei,
        "system_xi": system_xi,
        "cross_network_xi": cross_xi,
        "within_network_xi_sum": float(sum(within_xi.values())),
        "module_ei": module_ei,
        "scalar_ei_sum": scalar_ei_sum,
        "scalar_ei_by_network": scalar_ei,
        "within_network_xi": within_xi,
        "cross_network_shapley": cross_shapley,
        "network_attribution": network_attribution,
        "atoms": atom_rows,
        "identity_errors": {
            "within_plus_cross_minus_system": float(sum(within_xi.values()) + cross_xi - system_xi),
            "network_attribution_minus_system": float(sum(network_attribution.values()) - system_xi),
            "cross_shapley_minus_cross": float(sum(cross_shapley.values()) - cross_xi),
            "atom_sum_minus_cross": float(sum(row["value"] for row in atom_rows) - cross_xi),
        },
    }


def analyze_state(
    path: Path,
    groups: Mapping[str, Sequence[int]],
    *,
    subject: str,
    state: str,
) -> dict[str, Any]:
    if state == "REST":
        projection_signal = load_rest_series(path)
        fitting_signal = projection_signal
    else:
        retained, regressed = load_task_pair(path)
        projection_signal = retained
        fitting_signal = retained - regressed
    development_end = development_end_for_length(len(projection_signal))
    reduced, explained = fit_project_network_pca(
        fitting_signal,
        projection_signal,
        groups,
        development_end=development_end,
        n_components=N_COMPONENTS,
    )
    fitted = fit_delta_history_phi(
        reduced,
        alpha=ALPHA,
        order=ORDER,
        development_end=development_end,
    )
    decomposition = decompose_transition(
        fitted["transition"],
        fitted["noise_covariance"],
        tuple(groups),
        n_components=N_COMPONENTS,
        order=ORDER,
    )
    development = reduced[:development_end]
    scale = development.std(axis=0, ddof=1)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return {
        "config_id": CONFIG_ID,
        "subject": subject,
        "state": state,
        "n_timepoints": int(len(projection_signal)),
        "development_end": int(development_end),
        "pca_explained_variance_ratio": explained,
        "quality_diagnostics": {
            "max_abs_development_pc_zscore": float(
                np.max(np.abs((development - development.mean(axis=0)) / scale))
            ),
            "noise_covariance_condition": float(np.linalg.cond(fitted["noise_covariance"])),
        },
        **decomposition,
    }


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, repeats: int = 10_000) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(int(repeats), len(array)))
    return [float(value) for value in np.quantile(array[indices].mean(axis=1), [0.025, 0.975])]


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    values = np.where(np.isfinite(values), values, 1.0)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def friedman_by_feature(values: np.ndarray) -> list[dict[str, float | int]]:
    results = []
    for feature in range(values.shape[2]):
        selected = np.asarray(values[:, :, feature], dtype=float)
        if np.allclose(selected, selected[0, 0]):
            statistic, p_value = 0.0, 1.0
        else:
            test = friedmanchisquare(*[selected[state] for state in range(selected.shape[0])])
            statistic, p_value = float(test.statistic), float(test.pvalue)
        results.append({"feature": feature, "statistic": statistic, "p": p_value})
    adjusted = bh_adjust([float(row["p"]) for row in results])
    for row, q_value in zip(results, adjusted):
        row["q"] = float(q_value)
    return results


def pairwise_distribution_tests(
    values: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    results = []
    for left in range(values.shape[0]):
        for right in range(left + 1, values.shape[0]):
            difference = np.asarray(values[left] - values[right], dtype=float)
            observed = float(0.5 * np.abs(difference.mean(axis=0)).sum())
            signs = rng.choice((-1.0, 1.0), size=(int(repeats), difference.shape[0]))
            permuted_means = signs @ difference / difference.shape[0]
            null_statistics = 0.5 * np.abs(permuted_means).sum(axis=1)
            p_value = float((1 + np.sum(null_statistics >= observed)) / (int(repeats) + 1))
            results.append(
                {
                    "left_index": left,
                    "right_index": right,
                    "left": STATES[left],
                    "right": STATES[right],
                    "total_variation": observed,
                    "p": p_value,
                }
            )
    adjusted = bh_adjust([float(row["p"]) for row in results])
    for row, q_value in zip(results, adjusted):
        row["q"] = float(q_value)
    return results


def paired_rest_network_tests(network_share: np.ndarray) -> list[dict[str, Any]]:
    results = []
    for task_index, task in enumerate(TASKS, start=1):
        for network_index, network in enumerate(NETWORK_ORDER):
            difference = network_share[task_index, :, network_index] - network_share[0, :, network_index]
            if np.allclose(difference, 0.0):
                p_value = 1.0
            else:
                p_value = float(wilcoxon(difference, zero_method="wilcox", alternative="two-sided").pvalue)
            results.append(
                {
                    "task": task,
                    "network": network,
                    "mean_difference_percentage_points": float(100.0 * difference.mean()),
                    "p": p_value,
                }
            )
    adjusted = bh_adjust([float(row["p"]) for row in results])
    for row, q_value in zip(results, adjusted):
        row["q"] = float(q_value)
    return results


def paired_rest_system_tests(system_xi: np.ndarray) -> list[dict[str, Any]]:
    results = []
    for task_index, task in enumerate(TASKS, start=1):
        difference = system_xi[0] - system_xi[task_index]
        if np.allclose(difference, 0.0):
            p_value = 1.0
        else:
            p_value = float(
                wilcoxon(difference, zero_method="wilcox", alternative="two-sided").pvalue
            )
        results.append(
            {
                "task": task,
                "rest_minus_task_mean_bits": float(difference.mean()),
                "subjects_rest_greater": int(np.sum(difference > 0.0)),
                "p": p_value,
            }
        )
    adjusted = bh_adjust([float(row["p"]) for row in results])
    for row, q_value in zip(results, adjusted):
        row["q"] = float(q_value)
    return results


def all_atom_subsets() -> tuple[tuple[str, ...], ...]:
    return tuple(
        subset
        for size in range(2, len(NETWORK_ORDER) + 1)
        for subset in itertools.combinations(NETWORK_ORDER, size)
    )


def build_summary_and_arrays(
    records: Mapping[tuple[str, str], Mapping[str, Any]],
    subjects: Sequence[str],
    *,
    permutation_repeats: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    n_states, n_subjects, n_networks = len(STATES), len(subjects), len(NETWORK_ORDER)
    system_xi = np.empty((n_states, n_subjects), dtype=float)
    cross_xi = np.empty_like(system_xi)
    within_xi_sum = np.empty_like(system_xi)
    network_attribution = np.empty((n_states, n_subjects, n_networks), dtype=float)
    within_network_xi = np.empty_like(network_attribution)
    cross_shapley = np.empty_like(network_attribution)
    atom_subsets = all_atom_subsets()
    atom_lookup = {sources: index for index, sources in enumerate(atom_subsets)}
    atom_value = np.zeros((n_states, n_subjects, len(atom_subsets)), dtype=float)
    pca_cumulative = np.empty((n_states, n_subjects, n_networks), dtype=float)

    for state_index, state in enumerate(STATES):
        for subject_index, subject in enumerate(subjects):
            row = records[(subject, state)]
            system_xi[state_index, subject_index] = float(row["system_xi"])
            cross_xi[state_index, subject_index] = float(row["cross_network_xi"])
            within_xi_sum[state_index, subject_index] = float(row["within_network_xi_sum"])
            network_attribution[state_index, subject_index] = [
                float(row["network_attribution"][network]) for network in NETWORK_ORDER
            ]
            within_network_xi[state_index, subject_index] = [
                float(row["within_network_xi"][network]) for network in NETWORK_ORDER
            ]
            cross_shapley[state_index, subject_index] = [
                float(row["cross_network_shapley"][network]) for network in NETWORK_ORDER
            ]
            pca_cumulative[state_index, subject_index] = [
                float(sum(row["pca_explained_variance_ratio"][network]))
                for network in NETWORK_ORDER
            ]
            for atom in row["atoms"]:
                sources = tuple(atom["sources"])
                atom_value[state_index, subject_index, atom_lookup[sources]] = float(atom["value"])

    if np.any(system_xi <= 1.0e-10) or np.any(cross_xi <= 1.0e-10):
        raise ValueError("Network and atom shares require positive system and cross-network Xi.")
    network_share = network_attribution / system_xi[:, :, None]
    atom_share = atom_value / cross_xi[:, :, None]
    maximum_network_closure = float(np.max(np.abs(network_share.sum(axis=2) - 1.0)))
    maximum_atom_closure = float(np.max(np.abs(atom_share.sum(axis=2) - 1.0)))

    network_all = friedman_by_feature(network_share)
    network_task = friedman_by_feature(network_share[1:])
    atom_all = friedman_by_feature(atom_share)
    atom_task = friedman_by_feature(atom_share[1:])
    network_pairwise = pairwise_distribution_tests(
        network_share, repeats=permutation_repeats, seed=2026074101
    )
    atom_pairwise = pairwise_distribution_tests(
        atom_share, repeats=permutation_repeats, seed=2026074102
    )
    rest_system_tests = paired_rest_system_tests(system_xi)
    rest_network_tests = paired_rest_network_tests(network_share)

    state_summary = {}
    for state_index, state in enumerate(STATES):
        state_summary[state] = {
            "system_xi_mean_bits": float(system_xi[state_index].mean()),
            "system_xi_bootstrap_95_ci": bootstrap_mean_ci(
                system_xi[state_index], seed=2026074200 + state_index
            ),
            "cross_network_xi_mean_bits": float(cross_xi[state_index].mean()),
            "within_network_xi_sum_mean_bits": float(within_xi_sum[state_index].mean()),
            "cross_network_fraction_mean": float(
                np.mean(cross_xi[state_index] / system_xi[state_index])
            ),
            "network_share_percent": {
                network: float(100.0 * network_share[state_index, :, network_index].mean())
                for network_index, network in enumerate(NETWORK_ORDER)
            },
            "pca_cumulative_explained_variance_mean": {
                network: float(pca_cumulative[state_index, :, network_index].mean())
                for network_index, network in enumerate(NETWORK_ORDER)
            },
        }

    atom_means = atom_share.mean(axis=1)
    mean_across_states = atom_means.mean(axis=0)
    leading_indices = np.argsort(mean_across_states)[::-1]
    leading_atoms = [
        {
            "sources": list(atom_subsets[index]),
            "mean_cross_xi_share": float(mean_across_states[index]),
            "state_share_percent": {
                state: float(100.0 * atom_means[state_index, index])
                for state_index, state in enumerate(STATES)
            },
            "all_state_friedman_p": float(atom_all[index]["p"]),
            "all_state_friedman_q": float(atom_all[index]["q"]),
            "task_only_friedman_p": float(atom_task[index]["p"]),
            "task_only_friedman_q": float(atom_task[index]["q"]),
        }
        for index in leading_indices
        if mean_across_states[index] > 1.0e-10
    ]

    maximum_identity_error = max(
        abs(float(value))
        for row in records.values()
        for value in row["identity_errors"].values()
    )
    quality_flags = [
        {
            "subject": subject,
            "state": state,
            **records[(subject, state)]["quality_diagnostics"],
        }
        for subject in subjects
        for state in STATES
        if float(records[(subject, state)]["quality_diagnostics"]["max_abs_development_pc_zscore"]) > 10.0
        or float(records[(subject, state)]["quality_diagnostics"]["noise_covariance_condition"]) > 500.0
    ]

    summary = {
        "config": {
            "config_id": CONFIG_ID,
            "states": list(STATES),
            "subjects": list(subjects),
            "n_subjects": len(subjects),
            "n_components_per_network": N_COMPONENTS,
            "state_dimension": N_COMPONENTS * len(NETWORK_ORDER),
            "history_order": ORDER,
            "source_dimension": N_COMPONENTS * len(NETWORK_ORDER) * ORDER,
            "target_dimension": N_COMPONENTS * len(NETWORK_ORDER),
            "alpha": ALPHA,
            "estimator": "Gaussian log-det affine continuous-EI approximation",
            "permutation_repeats": int(permutation_repeats),
        },
        "diagnostics": {
            "maximum_identity_error_bits": maximum_identity_error,
            "maximum_network_share_closure_error": maximum_network_closure,
            "maximum_atom_share_closure_error": maximum_atom_closure,
            "quality_flags": quality_flags,
            "n_quality_flags": len(quality_flags),
        },
        "state_summary": state_summary,
        "statistics": {
            "system_rest_vs_task": rest_system_tests,
            "network_all_state_friedman": [
                {"network": NETWORK_ORDER[index], **row} for index, row in enumerate(network_all)
            ],
            "network_task_only_friedman": [
                {"network": NETWORK_ORDER[index], **row} for index, row in enumerate(network_task)
            ],
            "network_rest_vs_task": rest_network_tests,
            "network_distribution_pairwise": network_pairwise,
            "atom_all_state_friedman": [
                {"sources": list(atom_subsets[index]), **row} for index, row in enumerate(atom_all)
            ],
            "atom_task_only_friedman": [
                {"sources": list(atom_subsets[index]), **row} for index, row in enumerate(atom_task)
            ],
            "atom_distribution_pairwise": atom_pairwise,
        },
        "leading_atoms": leading_atoms,
    }
    arrays = {
        "system_xi": system_xi,
        "cross_network_xi": cross_xi,
        "within_network_xi_sum": within_xi_sum,
        "network_attribution": network_attribution,
        "network_share": network_share,
        "within_network_xi": within_network_xi,
        "cross_network_shapley": cross_shapley,
        "atom_value": atom_value,
        "atom_share": atom_share,
        "pca_cumulative_explained_variance": pca_cumulative,
    }
    return summary, arrays


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
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def pairwise_matrices(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    distance = np.zeros((len(STATES), len(STATES)), dtype=float)
    significant = np.zeros_like(distance, dtype=bool)
    for row in rows:
        left, right = int(row["left_index"]), int(row["right_index"])
        distance[left, right] = distance[right, left] = 100.0 * float(row["total_variation"])
        significant[left, right] = significant[right, left] = float(row["q"]) < 0.05
    return distance, significant


def short_atom_label(sources: Sequence[str]) -> str:
    abbreviations = {
        "Vis": "Vis",
        "SomMot": "Som",
        "DorsAttn": "DAN",
        "SalVentAttn": "SVAN",
        "Limbic": "Lim",
        "Cont": "Cont",
        "Default": "Def",
    }
    return "+".join(abbreviations[source] for source in sources)


def draw_pairwise_matrix(
    fig: Any,
    axis: Any,
    distance: np.ndarray,
    significant: np.ndarray,
    *,
    colorbar_label: str,
) -> None:
    upper = float(np.ceil(np.max(distance)))
    image = axis.imshow(distance, cmap="YlOrRd", vmin=0.0, vmax=max(upper, 1.0))
    labels = [DISPLAY_NAMES[state] for state in STATES]
    axis.set(
        xticks=np.arange(len(STATES)),
        xticklabels=labels,
        yticks=np.arange(len(STATES)),
        yticklabels=labels,
    )
    axis.tick_params(axis="x", labelrotation=40, length=0)
    axis.tick_params(axis="y", length=0)
    for row in range(len(STATES)):
        for column in range(len(STATES)):
            if row == column:
                text = "–"
            else:
                text = f"{distance[row, column]:.1f}{'*' if significant[row, column] else ''}"
            normalized = distance[row, column] / max(upper, 1.0)
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=4.6,
                color="white" if normalized > 0.58 else "black",
            )
    fig.colorbar(image, ax=axis, shrink=0.80, pad=0.02).set_label(colorbar_label)


def plot_main(summary: Mapping[str, Any], arrays: Mapping[str, np.ndarray], output: Path) -> None:
    configure_style()
    network_mean = arrays["network_share"].mean(axis=1).T * 100.0
    network_distance, network_significant = pairwise_matrices(
        summary["statistics"]["network_distribution_pairwise"]
    )
    atom_distance, atom_significant = pairwise_matrices(
        summary["statistics"]["atom_distribution_pairwise"]
    )
    atom_subsets = all_atom_subsets()
    atom_mean = arrays["atom_share"].mean(axis=1)
    ranking = np.argsort(atom_mean.mean(axis=0))[::-1]
    selected = [index for index in ranking if atom_mean[:, index].mean() > 1.0e-10][:12]
    atom_panel = atom_mean[:, selected].T * 100.0

    fig = plt.figure(figsize=(12.2, 8.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1.0), height_ratios=(1.0, 1.2))
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(2)]

    lower, upper = float(np.floor(network_mean.min())), float(np.ceil(network_mean.max()))
    image = axes[0].imshow(network_mean, cmap="YlGnBu", vmin=lower, vmax=upper, aspect="auto")
    axes[0].set(
        xticks=np.arange(len(STATES)),
        xticklabels=[DISPLAY_NAMES[state] for state in STATES],
        yticks=np.arange(len(NETWORK_ORDER)),
        yticklabels=[NETWORK_LABELS[network] for network in NETWORK_ORDER],
        xlabel="State",
        ylabel="Yeo7 network",
    )
    axes[0].tick_params(axis="x", labelrotation=35, length=0)
    axes[0].tick_params(axis="y", length=0)
    axes[0].axvline(0.5, color="#333333", linewidth=0.9)
    for row in range(network_mean.shape[0]):
        for column in range(network_mean.shape[1]):
            value = network_mean[row, column]
            normalized = (value - lower) / max(upper - lower, 1.0e-12)
            axes[0].text(
                column, row, f"{value:.1f}", ha="center", va="center", fontsize=5.0,
                color="white" if normalized > 0.60 else "black",
            )
    fig.colorbar(image, ax=axes[0], shrink=0.80, pad=0.02).set_label(r"Share of system-level $\Xi$ (%)")

    draw_pairwise_matrix(
        fig,
        axes[1],
        network_distance,
        network_significant,
        colorbar_label="Network-attribution TV distance (%)",
    )

    atom_upper = float(np.ceil(np.quantile(atom_panel, 0.995)))
    image = axes[2].imshow(atom_panel, cmap="magma_r", vmin=0.0, vmax=max(atom_upper, 1.0), aspect="auto")
    axes[2].set(
        xticks=np.arange(len(STATES)),
        xticklabels=[DISPLAY_NAMES[state] for state in STATES],
        yticks=np.arange(len(selected)),
        yticklabels=[short_atom_label(atom_subsets[index]) for index in selected],
        xlabel="State",
        ylabel="Greedy hierarchy atom",
    )
    axes[2].tick_params(axis="x", labelrotation=35, length=0)
    axes[2].tick_params(axis="y", length=0)
    axes[2].axvline(0.5, color="#F0F0F0", linewidth=0.9)
    for row in range(atom_panel.shape[0]):
        for column in range(atom_panel.shape[1]):
            value = atom_panel[row, column]
            axes[2].text(
                column, row, f"{value:.1f}", ha="center", va="center", fontsize=4.4,
                color="white" if value > 0.38 * max(atom_upper, 1.0) else "black",
            )
    fig.colorbar(image, ax=axes[2], shrink=0.80, pad=0.02).set_label(r"Share of cross-network $\Xi$ (%)")

    draw_pairwise_matrix(
        fig,
        axes[3],
        atom_distance,
        atom_significant,
        colorbar_label="Hierarchy-atom TV distance (%)",
    )
    for label, axis in zip("abcd", axes):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_budget(summary: Mapping[str, Any], arrays: Mapping[str, np.ndarray], output: Path) -> None:
    configure_style()
    means = arrays["system_xi"].mean(axis=1)
    intervals = np.asarray(
        [summary["state_summary"][state]["system_xi_bootstrap_95_ci"] for state in STATES]
    )
    cross_fraction = np.mean(arrays["cross_network_xi"] / arrays["system_xi"], axis=1)
    within_fraction = 1.0 - cross_fraction
    x = np.arange(len(STATES))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)
    axes[0].errorbar(
        x,
        means,
        yerr=np.vstack([means - intervals[:, 0], intervals[:, 1] - means]),
        fmt="o",
        color="#4C78A8",
        ecolor="#9CB7CF",
        capsize=2,
        markersize=4,
    )
    axes[0].set(
        xticks=x,
        xticklabels=[DISPLAY_NAMES[state] for state in STATES],
        ylabel=r"System-level $\Xi$ (bits)",
        xlabel="State",
    )
    axes[0].tick_params(axis="x", labelrotation=35)
    axes[0].axvline(0.5, color="#777777", linewidth=0.8)

    axes[1].bar(x, within_fraction, color="#A9C5D1", label="Within-network")
    axes[1].bar(x, cross_fraction, bottom=within_fraction, color="#D98B5F", label="Cross-network")
    axes[1].set(
        xticks=x,
        xticklabels=[DISPLAY_NAMES[state] for state in STATES],
        ylabel=r"Fraction of system-level $\Xi$",
        xlabel="State",
        ylim=(0.0, 1.0),
    )
    axes[1].tick_params(axis="x", labelrotation=35)
    axes[1].axvline(0.5, color="#777777", linewidth=0.8)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    for label, axis in zip("ab", axes):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def significant_pair_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return int(sum(float(row["q"]) < 0.05 for row in rows))


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    stats = summary["statistics"]
    network_pairwise = stats["network_distribution_pairwise"]
    atom_pairwise = stats["atom_distribution_pairwise"]
    significant_network_omnibus = [
        row for row in stats["network_all_state_friedman"] if float(row["q"]) < 0.05
    ]
    significant_network_task = [
        row for row in stats["network_task_only_friedman"] if float(row["q"]) < 0.05
    ]
    significant_atoms_all = [
        row for row in stats["atom_all_state_friedman"] if float(row["q"]) < 0.05
    ]
    significant_atoms_task = [
        row for row in stats["atom_task_only_friedman"] if float(row["q"]) < 0.05
    ]
    significant_rest_network = [
        row for row in stats["network_rest_vs_task"] if float(row["q"]) < 0.05
    ]

    lines = [
        "# HCP Schaefer-500：每网络两个主成分的 Xi 层级分解",
        "",
        "每个 Yeo7 网络保留前两个主成分。任务态的 PCA 载荷由训练前缀中的 taskRetained-taskRegressed 拟合，再投影完整 taskRetained；REST 由自身训练前缀拟合并投影。随后固定使用五阶 Delta-Ridge（alpha=10）、Gaussian log-det EI、七个不可拆网络模块、精确 Shapley 归因和 nonnegative-tolerant greedy 层级分解。",
        "",
        "历史字段 Phi 在本报告统一记为联合有效信息增量 Xi。70个最细源是七网络、两个PC与五个滞后的笛卡尔积；每个网络模块包含10个源坐标。",
        "",
        "## 系统级结果",
        "",
        "| State | Xi (bits) | 95% CI | Within-network Xi | Cross-network Xi | Cross fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for state in STATES:
        item = summary["state_summary"][state]
        ci = item["system_xi_bootstrap_95_ci"]
        lines.append(
            f"| {DISPLAY_NAMES[state]} | {item['system_xi_mean_bits']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{item['within_network_xi_sum_mean_bits']:.4f} | {item['cross_network_xi_mean_bits']:.4f} | "
            f"{100*item['cross_network_fraction_mean']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "REST 与各任务的系统级 Xi 采用被试内 Wilcoxon 检验，并在七个比较间做 BH 校正。正差值表示 REST 更高。",
            "",
            "| Task | REST - task (bits) | REST higher subjects | BH q |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in stats["system_rest_vs_task"]:
        lines.append(
            f"| {DISPLAY_NAMES[row['task']]} | {row['rest_minus_task_mean_bits']:+.4f} | "
            f"{row['subjects_rest_greater']}/{summary['config']['n_subjects']} | {row['q']:.3g} |"
        )
    lines.extend(
        [
            "",
            "## 网络自身归因差异",
            "",
            f"八状态 Friedman 检验经七网络 BH 校正后显著的网络为 {len(significant_network_omnibus)}/7；仅七任务检验显著的网络为 {len(significant_network_task)}/7。REST–任务的49个网络对比中，BH q<0.05 的对比为 {len(significant_rest_network)}/49。",
            "",
            f"七网络归因分布的28个状态对中，配对 sign-flip 检验经 BH 校正后显著 {significant_pair_count(network_pairwise)}/28。星号已标在主图 b。",
            "",
            "| Network | All-state q | Task-only q | REST share | Lowest task share | Highest task share |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    all_lookup = {row["network"]: row for row in stats["network_all_state_friedman"]}
    task_lookup = {row["network"]: row for row in stats["network_task_only_friedman"]}
    for network in NETWORK_ORDER:
        task_shares = {
            state: summary["state_summary"][state]["network_share_percent"][network]
            for state in TASKS
        }
        low = min(task_shares, key=task_shares.get)
        high = max(task_shares, key=task_shares.get)
        lines.append(
            f"| {NETWORK_LABELS[network]} | {all_lookup[network]['q']:.3g} | {task_lookup[network]['q']:.3g} | "
            f"{summary['state_summary']['REST']['network_share_percent'][network]:.2f}% | "
            f"{DISPLAY_NAMES[low]} {task_shares[low]:.2f}% | {DISPLAY_NAMES[high]} {task_shares[high]:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 脑区组合与层级原子",
            "",
            f"120个可能的多网络子集以未入选时记0的 greedy atom share 进行检验。八状态 Friedman + BH 后显著 {len(significant_atoms_all)}/120；仅七任务显著 {len(significant_atoms_task)}/120。",
            "",
            f"层级原子分布的28个状态对中，配对 sign-flip + BH 后显著 {significant_pair_count(atom_pairwise)}/28。星号已标在主图 d。",
            "",
            "| Leading atom | Mean cross-Xi share | All-state q | Task-only q |",
            "|---|---:|---:|---:|",
        ]
    )
    for atom in summary["leading_atoms"][:15]:
        lines.append(
            f"| {short_atom_label(atom['sources'])} | {100*atom['mean_cross_xi_share']:.2f}% | "
            f"{atom['all_state_friedman_q']:.3g} | {atom['task_only_friedman_q']:.3g} |"
        )
    diagnostics = summary["diagnostics"]
    lines.extend(
        [
            "",
            "## 诊断与限制",
            "",
            f"最大分解恒等式误差为 {diagnostics['maximum_identity_error_bits']:.3e} bits；网络份额和原子份额最大闭合误差分别为 {diagnostics['maximum_network_share_closure_error']:.3e} 和 {diagnostics['maximum_atom_share_closure_error']:.3e}；质量标记模型 {diagnostics['n_quality_flags']}/{len(STATES)*summary['config']['n_subjects']}。",
            "",
            "星号表示整个七网络归因分布或整个层级原子分布的配对多变量差异经28个状态对 BH 校正后 q<0.05。单个 greedy atom 的检验属于路径依赖的探索性结果；不能将未显著解释为组合不存在。",
            "",
            "REST 与任务使用不同的 PCA 拟合信号，因此 REST–任务差异包含状态与表征口径两部分；任务之间使用相同规则。Gaussian log-det 结果只描述当前线性高斯代理动力学。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    discovered = discover_inputs(args.rest_root, args.task_root)
    subjects = sorted(discovered)
    if args.max_subjects is not None:
        subjects = subjects[: int(args.max_subjects)]
    groups = load_yeo7_groups(args.labels, expected_parcels=500)
    cache_path = args.output_dir / "records.jsonl"
    records = load_cache(cache_path)
    jobs = [
        (subject, state, Path(discovered[subject][state]))
        for subject in subjects
        for state in STATES
        if (subject, state) not in records
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(jobs, desc="PC2 Xi hierarchy", unit="model", mininterval=1.0)
    for completed, (subject, state, path) in enumerate(progress, start=1):
        row = analyze_state(path, groups, subject=subject, state=state)
        records[(subject, state)] = row
        append_jsonl(cache_path, row)
        done = len(subjects) * len(STATES) - len(jobs) + completed
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        atomic_json(
            args.output_dir / "live_progress.json",
            {
                "phase": "compute",
                "current": done,
                "total": len(subjects) * len(STATES),
                "unit": "model",
                "elapsed_seconds": elapsed,
                "eta_seconds": (len(subjects) * len(STATES) - done) / rate if rate > 0 else None,
                "metrics": {"subject": subject, "state": state, "system_xi": row["system_xi"]},
                "updated_at": time.time(),
            },
        )
        progress.set_postfix(state=state, xi=f"{row['system_xi']:.2f}")

    selected = {(subject, state): records[(subject, state)] for subject in subjects for state in STATES}
    summary, arrays = build_summary_and_arrays(
        selected, subjects, permutation_repeats=int(args.permutation_repeats)
    )
    summary["config"]["computed_rows"] = len(jobs)
    summary["config"]["reused_rows"] = len(subjects) * len(STATES) - len(jobs)
    atomic_json(args.output_dir / "summary.json", summary)
    np.savez_compressed(
        args.output_dir / "arrays.npz",
        states=np.asarray(STATES),
        subjects=np.asarray(subjects),
        networks=np.asarray(NETWORK_ORDER),
        atom_names=np.asarray(["+".join(sources) for sources in all_atom_subsets()]),
        **arrays,
    )
    plot_main(summary, arrays, args.output_dir / "pc2_xi_hierarchy_state_differences")
    plot_budget(summary, arrays, args.output_dir / "pc2_xi_system_budget")
    write_report(summary, args.output_dir / "report.md")
    atomic_json(
        args.output_dir / "live_progress.json",
        {
            "phase": "complete",
            "current": len(subjects) * len(STATES),
            "total": len(subjects) * len(STATES),
            "unit": "model",
            "elapsed_seconds": time.monotonic() - started,
            "metrics": {
                "significant_network_pairs": significant_pair_count(
                    summary["statistics"]["network_distribution_pairwise"]
                ),
                "significant_atom_pairs": significant_pair_count(
                    summary["statistics"]["atom_distribution_pairwise"]
                ),
            },
            "updated_at": time.time(),
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rest-root", type=Path, default=DEFAULT_REST_ROOT)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--labels", type=Path, default=default_yeo7_labels(500))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--permutation-repeats", type=int, default=5000)
    args = parser.parse_args(argv)
    result = run(args)
    print(
        json.dumps(
            {
                "n_subjects": result["config"]["n_subjects"],
                "diagnostics": result["diagnostics"],
                "significant_network_pairs": significant_pair_count(
                    result["statistics"]["network_distribution_pairwise"]
                ),
                "significant_atom_pairs": significant_pair_count(
                    result["statistics"]["atom_distribution_pairwise"]
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
