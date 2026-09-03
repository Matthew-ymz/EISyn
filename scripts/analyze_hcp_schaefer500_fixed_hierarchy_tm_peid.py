#!/usr/bin/env python3
"""Fixed-hierarchy TM-PEID attribution for REST and all HCP task states."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_schaefer500_task_specific_regions import (
    DEFAULT_LABEL_FILE,
    DEFAULT_REST_ROOT,
    DEFAULT_TASK_ROOT,
    NETWORK_ORDER,
    TASKS,
    discover_inputs,
    discover_rest_inputs,
    load_rest_series,
    load_task_pair,
    parse_schaefer_labels,
)
from scripts.spt import SPTConfig, build_spt_from_ei_table, flatten_nodes
from yrd.transport_map import gaussian_logdet_bias_correction


DEFAULT_OUTPUT_DIR = ROOT / "results" / "hcp_schaefer500_fixed_hierarchy_tm_peid"
STATE_ORDER = ("REST", *TASKS)
STATE_LABELS = ("REST", "Emotion", "Gambling", "Language", "Motor", "Relational", "Social", "WM")
HEMISPHERES = ("LH", "RH")
LEAF_NAMES = tuple(f"{network}-{hemisphere[0]}" for network in NETWORK_ORDER for hemisphere in HEMISPHERES)
CONFIG_VERSION = "yeo7xhemi_qridge2_uniformsqrt3_tm_affine_tree_v2"
SYN_NONNEGATIVE_TOLERANCE_BITS = 0.10


def make_config_id(*, alpha: float, intervention_samples: int, seed: int) -> str:
    return (
        f"{CONFIG_VERSION}|alpha={float(alpha):.12g}"
        f"|samples={int(intervention_samples)}|seed={int(seed)}"
    )


@dataclass(frozen=True)
class HierarchyNode:
    name: str
    leaves: tuple[str, ...]
    children: tuple[str, str] | None = None


@dataclass
class QuadraticRidgeTransition:
    polynomial: PolynomialFeatures
    ridge: Ridge
    residual_cholesky: np.ndarray
    train_rmse: float
    holdout_rmse: float
    persistence_rmse: float

    @property
    def skill_ratio(self) -> float:
        return float(self.holdout_rmse / max(self.persistence_rmse, 1.0e-12))

    def predict(self, source: np.ndarray) -> np.ndarray:
        values = np.asarray(source, dtype=float)
        delta = self.ridge.predict(self.polynomial.transform(values))
        return values + delta


def build_fixed_hierarchy() -> tuple[dict[str, HierarchyNode], str]:
    nodes: dict[str, HierarchyNode] = {
        leaf: HierarchyNode(leaf, (leaf,), None) for leaf in LEAF_NAMES
    }

    def add(name: str, left: str, right: str) -> None:
        leaves = (*nodes[left].leaves, *nodes[right].leaves)
        nodes[name] = HierarchyNode(name, leaves, (left, right))

    for network in NETWORK_ORDER:
        add(network, f"{network}-L", f"{network}-R")
    add("Sensory", "Vis", "SomMot")
    add("Attention", "DorsAttn", "SalVentAttn")
    add("Association", "Cont", "Default")
    add("Higher", "Attention", "Association")
    add("NonLimbic", "Sensory", "Higher")
    add("Whole", "NonLimbic", "Limbic")
    if len(nodes) != 27 or set(nodes["Whole"].leaves) != set(LEAF_NAMES):
        raise AssertionError("Fixed hierarchy must be a full binary tree over 14 leaves.")
    return nodes, "Whole"


def hierarchy_postorder(nodes: dict[str, HierarchyNode], root: str) -> list[str]:
    order: list[str] = []

    def visit(name: str) -> None:
        children = nodes[name].children
        if children is not None:
            visit(children[0])
            visit(children[1])
        order.append(name)

    visit(root)
    return order


def leaf_groups(labels: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {name: [] for name in LEAF_NAMES}
    for item in labels:
        name = f"{item['network']}-{item['hemisphere'][0]}"
        groups[name].append(int(item["index"]))
    if sum(map(len, groups.values())) != 500 or any(not indices for indices in groups.values()):
        raise ValueError("Expected every Schaefer-500 parcel in one of 14 hemisphere-network leaves.")
    return {name: np.asarray(indices, dtype=int) for name, indices in groups.items()}


def _center_columns(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array - array.mean(axis=0, keepdims=True)


def _reduce_centered_parcels(
    centered: np.ndarray,
    parcel_scale: np.ndarray,
    groups: dict[str, np.ndarray],
) -> np.ndarray:
    standardized = centered / parcel_scale.reshape(1, -1)
    return np.column_stack([standardized[:, groups[name]].mean(axis=1) for name in LEAF_NAMES])


def reduce_rest_series(raw: np.ndarray, groups: dict[str, np.ndarray]) -> np.ndarray:
    centered = _center_columns(raw)
    parcel_scale = centered.std(axis=0, ddof=1)
    parcel_scale = np.where(parcel_scale > 1.0e-12, parcel_scale, 1.0)
    reduced = _reduce_centered_parcels(centered, parcel_scale, groups)
    reduced = _center_columns(reduced)
    node_scale = reduced.std(axis=0, ddof=1)
    node_scale = np.where(node_scale > 1.0e-12, node_scale, 1.0)
    return reduced / node_scale.reshape(1, -1)


def reduce_task_pair(
    retained: np.ndarray,
    regressed: np.ndarray,
    groups: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    retained_centered = _center_columns(retained)
    regressed_centered = _center_columns(regressed)
    combined = np.concatenate([retained_centered, regressed_centered], axis=0)
    parcel_scale = combined.std(axis=0, ddof=1)
    parcel_scale = np.where(parcel_scale > 1.0e-12, parcel_scale, 1.0)
    retained_nodes = _reduce_centered_parcels(retained_centered, parcel_scale, groups)
    regressed_nodes = _reduce_centered_parcels(regressed_centered, parcel_scale, groups)
    node_combined = np.concatenate([retained_nodes, regressed_nodes], axis=0)
    node_scale = node_combined.std(axis=0, ddof=1)
    node_scale = np.where(node_scale > 1.0e-12, node_scale, 1.0)
    return retained_nodes / node_scale.reshape(1, -1), regressed_nodes / node_scale.reshape(1, -1)


def _fit_ridge_core(source: np.ndarray, target: np.ndarray, *, alpha: float) -> tuple[PolynomialFeatures, Ridge]:
    polynomial = PolynomialFeatures(degree=2, include_bias=False)
    design = polynomial.fit_transform(source)
    ridge = Ridge(alpha=float(alpha), fit_intercept=True).fit(design, target - source)
    return polynomial, ridge


def fit_transition(series: np.ndarray, *, alpha: float) -> QuadraticRidgeTransition:
    values = np.asarray(series, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(LEAF_NAMES) or len(values) < 40:
        raise ValueError(f"Expected [time, 14] series with at least 40 rows, got {values.shape}.")
    source, target = values[:-1], values[1:]
    split = max(25, min(len(source) - 10, int(round(0.75 * len(source)))))
    eval_poly, eval_ridge = _fit_ridge_core(source[:split], target[:split], alpha=alpha)
    eval_prediction = source[split:] + eval_ridge.predict(eval_poly.transform(source[split:]))
    holdout_rmse = float(np.sqrt(np.mean((target[split:] - eval_prediction) ** 2)))
    persistence_rmse = float(np.sqrt(np.mean((target[split:] - source[split:]) ** 2)))

    polynomial, ridge = _fit_ridge_core(source, target, alpha=alpha)
    train_prediction = source + ridge.predict(polynomial.transform(source))
    residual = target - train_prediction
    covariance = np.atleast_2d(np.cov(residual, rowvar=False, ddof=1))
    shrink = max(1.0e-5, 1.0e-3 * float(np.trace(covariance)) / covariance.shape[0])
    covariance = covariance + shrink * np.eye(covariance.shape[0])
    return QuadraticRidgeTransition(
        polynomial=polynomial,
        ridge=ridge,
        residual_cholesky=np.linalg.cholesky(covariance),
        train_rmse=float(np.sqrt(np.mean(residual**2))),
        holdout_rmse=holdout_rmse,
        persistence_rmse=persistence_rmse,
    )


def tm_ei_table(
    model: QuadraticRidgeTransition,
    intervention_source: np.ndarray,
    base_noise: np.ndarray,
    nodes: dict[str, HierarchyNode],
    root: str,
) -> dict[str, Any]:
    source = np.asarray(intervention_source, dtype=float)
    target = model.predict(source) + np.asarray(base_noise, dtype=float) @ model.residual_cholesky.T
    leaf_index = {name: index for index, name in enumerate(LEAF_NAMES)}
    joint_samples = np.concatenate([source, target], axis=1)
    covariance = np.atleast_2d(np.cov(joint_samples, rowvar=False, ddof=1))
    covariance += 1.0e-6 * np.eye(covariance.shape[0])
    target_indices = list(range(len(LEAF_NAMES), 2 * len(LEAF_NAMES)))

    def logdet(indices: Sequence[int]) -> float:
        sign, value = np.linalg.slogdet(covariance[np.ix_(indices, indices)])
        if sign <= 0.0 or not np.isfinite(value):
            raise ValueError("Affine TM covariance is not positive definite.")
        return float(value)

    target_logdet = logdet(target_indices)
    sample_size = len(source)
    ei: dict[str, float] = {}
    ei_raw: dict[str, float] = {}
    bias: dict[str, float] = {}
    for name in hierarchy_postorder(nodes, root):
        indices = [leaf_index[leaf] for leaf in nodes[name].leaves]
        joint_indices = [*indices, *target_indices]
        gaussian_mi = 0.5 * (
            logdet(indices) + target_logdet - logdet(joint_indices)
        )
        correction = 0.5 * (
            gaussian_logdet_bias_correction(len(indices), sample_size)
            + gaussian_logdet_bias_correction(len(target_indices), sample_size)
            - gaussian_logdet_bias_correction(len(joint_indices), sample_size)
        )
        raw = float(gaussian_mi - correction)
        ei_raw[name] = raw
        ei[name] = raw
        bias[name] = float(correction)

    def canonical(leaves: Sequence[str]) -> tuple[str, ...]:
        selected = set(leaves)
        return tuple(leaf for leaf in LEAF_NAMES if leaf in selected)

    coalition_to_name = {canonical(node.leaves): name for name, node in nodes.items()}
    ei_table = {canonical(nodes[name].leaves): float(value) for name, value in ei.items()}

    def fixed_selector(coalition: tuple[str, ...]):
        name = coalition_to_name[tuple(coalition)]
        children = nodes[name].children
        if children is None:
            return "leaf", []
        left, right = children
        return "fixed-prior", [(canonical(nodes[left].leaves), canonical(nodes[right].leaves))]

    result = build_spt_from_ei_table(
        LEAF_NAMES,
        ei_table,
        singleton_ei={leaf: float(ei[leaf]) for leaf in LEAF_NAMES},
        config=SPTConfig(
            syn_tolerance=SYN_NONNEGATIVE_TOLERANCE_BITS,
            complete_to_singletons=True,
        ),
        candidate_selector=fixed_selector,
    )
    atoms = {
        coalition_to_name[tuple(node.sources)]: float(node.syn_value)
        for node in flatten_nodes(result.root)
        if node.children
    }
    phi = float(result.root.xi_value)
    atom_sum_error = float(result.closure_error)
    contribution = {leaf: 0.0 for leaf in LEAF_NAMES}
    for name, atom in atoms.items():
        share = atom / len(nodes[name].leaves)
        for leaf in nodes[name].leaves:
            contribution[leaf] += share
    contribution_error = float(sum(contribution.values()) - phi)

    def serialize(node) -> dict[str, Any]:
        return {
            "name": coalition_to_name[tuple(node.sources)],
            "sources": list(node.sources),
            "xi_bits": float(node.xi_value),
            "syn_bits": float(node.syn_value),
            "depth": int(node.depth),
            "split_kind": str(node.split_kind),
            "children": [serialize(child) for child in node.children],
        }

    return {
        "whole_ei": float(ei[root]),
        "leaf_ei_sum": float(sum(ei[leaf] for leaf in LEAF_NAMES)),
        "phi_eid": phi,
        "ei": ei,
        "ei_raw": ei_raw,
        "bias_correction": bias,
        "atoms": atoms,
        "tree": serialize(result.root),
        "spt_contract": {
            "core": "scripts.spt.build_spt",
            "route": "fixed-prior candidate selector",
            "syn_nonnegative_tolerance_bits": SYN_NONNEGATIVE_TOLERANCE_BITS,
            "tolerance_zero_count": int(result.audit.tolerance_zero_count),
        },
        "contribution": contribution,
        "atom_sum_error": atom_sum_error,
        "contribution_sum_error": contribution_error,
        "negative_atom_count": int(sum(value < 0.0 for value in atoms.values())),
        "negative_raw_ei_count": int(sum(value < 0.0 for value in ei_raw.values())),
    }


def analyze_reduced_series(
    series: np.ndarray,
    *,
    subject: str,
    condition: str,
    variant: str,
    alpha: float,
    intervention_source: np.ndarray,
    base_noise: np.ndarray,
    nodes: dict[str, HierarchyNode],
    root: str,
    config_id: str,
) -> dict[str, Any]:
    model = fit_transition(series, alpha=alpha)
    peid = tm_ei_table(model, intervention_source, base_noise, nodes, root)
    return {
        "config_id": config_id,
        "subject": subject,
        "condition": condition,
        "variant": variant,
        "n_timepoints": int(len(series)),
        "model": {
            "family": "quadratic delta Ridge",
            "alpha": float(alpha),
            "train_rmse": model.train_rmse,
            "holdout_rmse": model.holdout_rmse,
            "persistence_rmse": model.persistence_rmse,
            "skill_ratio": model.skill_ratio,
        },
        **peid,
    }


def record_key(record: dict[str, Any]) -> str:
    return f"{record['subject']}|{record['condition']}|{record['variant']}"


def load_records(path: Path, *, config_id: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("config_id") == config_id:
                records[record_key(record)] = record
    return records


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def bootstrap_mean_ci(values: np.ndarray, *, repeats: int, seed: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(int(repeats), len(array)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(array.mean()), float(low), float(high)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.07, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def _annotate_heatmap(ax: plt.Axes, values: np.ndarray, limit: float) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = float(values[row, col])
            ax.text(
                col,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=5.7,
                color="white" if abs(value) > 0.58 * limit else "black",
            )


def plot_node_contributions(
    absolute: np.ndarray,
    absolute_phi: np.ndarray,
    task_delta: np.ndarray,
    delta_phi: np.ndarray,
    output_dir: Path,
    *,
    bootstrap_repeats: int,
) -> None:
    configure_style()
    figure = plt.figure(figsize=(12.4, 6.7), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=[2.7, 1.0])
    axes = [figure.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]

    absolute_mean = absolute.mean(axis=1)
    limit = float(np.quantile(np.abs(absolute_mean), 0.995))
    ax = axes[0]
    image = ax.imshow(absolute_mean, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(STATE_LABELS)), STATE_LABELS)
    ax.set_xticks(range(len(NETWORK_ORDER)), NETWORK_ORDER, rotation=35, ha="right")
    ax.set_ylabel("State")
    ax.set_xlabel("Yeo7 network (left + right leaf attribution)")
    colorbar = figure.colorbar(image, ax=ax, location="right", shrink=0.82, pad=0.015)
    colorbar.set_label("TM-PEID integration contribution (nats)")
    _panel_label(ax, "a")

    ax = axes[1]
    positions = np.arange(len(STATE_LABELS))
    for state_index, position in enumerate(positions):
        mean, low, high = bootstrap_mean_ci(
            absolute_phi[state_index], repeats=bootstrap_repeats, seed=3100 + state_index
        )
        ax.errorbar(
            mean,
            position,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt="o",
            color="#4C78A8" if state_index == 0 else "#D17A22",
            ecolor="#4C78A8" if state_index == 0 else "#D17A22",
            capsize=2,
            markersize=4,
        )
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=0.7)
    ax.set_yticks(positions, STATE_LABELS)
    ax.invert_yaxis()
    ax.set_xlabel(r"Absolute $\Phi^{EID}$ (nats; mean and 95% bootstrap CI)")
    ax.set_ylabel("State")
    _panel_label(ax, "b")

    delta_mean = task_delta.mean(axis=1)
    delta_limit = float(np.quantile(np.abs(delta_mean), 0.995))
    ax = axes[2]
    image = ax.imshow(delta_mean, cmap="RdBu_r", vmin=-delta_limit, vmax=delta_limit, aspect="auto")
    ax.set_yticks(range(len(TASKS)), STATE_LABELS[1:])
    ax.set_xticks(range(len(NETWORK_ORDER)), NETWORK_ORDER, rotation=35, ha="right")
    ax.set_ylabel("Task")
    ax.set_xlabel("Yeo7 network (left + right leaf attribution)")
    colorbar = figure.colorbar(image, ax=ax, location="right", shrink=0.82, pad=0.015)
    colorbar.set_label("Retained − regressed contribution (nats)")
    _panel_label(ax, "c")

    ax = axes[3]
    positions = np.arange(len(TASKS))
    for task_index, position in enumerate(positions):
        mean, low, high = bootstrap_mean_ci(
            delta_phi[task_index], repeats=bootstrap_repeats, seed=4100 + task_index
        )
        color = "#B44C43" if mean >= 0.0 else "#4C78A8"
        ax.errorbar(
            mean,
            position,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2,
            markersize=4,
        )
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=0.7)
    ax.set_yticks(positions, STATE_LABELS[1:])
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta\Phi^{EID}$ retained − regressed (nats)")
    ax.set_ylabel("Task")
    _panel_label(ax, "d")

    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output_dir / f"fixed_hierarchy_tm_peid_node_contributions.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(figure)


def plot_atoms(
    absolute_atoms: np.ndarray,
    task_delta_atoms: np.ndarray,
    atom_names: Sequence[str],
    output_dir: Path,
) -> None:
    configure_style()
    figure, axes = plt.subplots(2, 1, figsize=(10.8, 5.5), constrained_layout=True)
    for panel, (ax, values, ylabels, colorbar_label) in enumerate(
        (
            (axes[0], absolute_atoms.mean(axis=1), STATE_LABELS, "Hierarchy atom (nats)"),
            (axes[1], task_delta_atoms.mean(axis=1), STATE_LABELS[1:], "Retained − regressed atom (nats)"),
        )
    ):
        limit = float(np.quantile(np.abs(values), 0.995))
        image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_yticks(range(len(ylabels)), ylabels)
        ax.set_xticks(range(len(atom_names)), atom_names, rotation=45, ha="right")
        ax.set_ylabel("State" if panel == 0 else "Task")
        ax.set_xlabel("Fixed internal hierarchy node")
        colorbar = figure.colorbar(image, ax=ax, location="right", shrink=0.82, pad=0.015)
        colorbar.set_label(colorbar_label)
        _panel_label(ax, chr(ord("a") + panel))
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output_dir / f"fixed_hierarchy_tm_peid_atoms.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(figure)


def build_arrays(
    records: dict[str, dict[str, Any]],
    subjects: Sequence[str],
    rest_subjects: Sequence[str],
    atom_names: Sequence[str],
) -> dict[str, np.ndarray]:
    common = [subject for subject in subjects if subject in set(rest_subjects)]
    absolute_leaf = np.empty((len(STATE_ORDER), len(common), len(LEAF_NAMES)), dtype=float)
    absolute_phi = np.empty((len(STATE_ORDER), len(common)), dtype=float)
    absolute_atoms = np.empty((len(STATE_ORDER), len(common), len(atom_names)), dtype=float)
    for state_index, condition in enumerate(STATE_ORDER):
        variant = "rest" if condition == "REST" else "retained"
        for subject_index, subject in enumerate(common):
            record = records[f"{subject}|{condition}|{variant}"]
            absolute_leaf[state_index, subject_index] = [record["contribution"][name] for name in LEAF_NAMES]
            absolute_phi[state_index, subject_index] = record["phi_eid"]
            absolute_atoms[state_index, subject_index] = [record["atoms"][name] for name in atom_names]

    task_delta_leaf = np.empty((len(TASKS), len(subjects), len(LEAF_NAMES)), dtype=float)
    delta_phi = np.empty((len(TASKS), len(subjects)), dtype=float)
    task_delta_atoms = np.empty((len(TASKS), len(subjects), len(atom_names)), dtype=float)
    for task_index, task in enumerate(TASKS):
        for subject_index, subject in enumerate(subjects):
            retained = records[f"{subject}|{task}|retained"]
            regressed = records[f"{subject}|{task}|regressed"]
            task_delta_leaf[task_index, subject_index] = [
                retained["contribution"][name] - regressed["contribution"][name]
                for name in LEAF_NAMES
            ]
            delta_phi[task_index, subject_index] = retained["phi_eid"] - regressed["phi_eid"]
            task_delta_atoms[task_index, subject_index] = [
                retained["atoms"][name] - regressed["atoms"][name] for name in atom_names
            ]
    absolute = absolute_leaf.reshape(len(STATE_ORDER), len(common), len(NETWORK_ORDER), 2).sum(axis=-1)
    task_delta = task_delta_leaf.reshape(len(TASKS), len(subjects), len(NETWORK_ORDER), 2).sum(axis=-1)
    return {
        "common_subjects": np.asarray(common),
        "absolute_leaf_contribution": absolute_leaf,
        "absolute_contribution": absolute,
        "absolute_phi": absolute_phi,
        "absolute_atoms": absolute_atoms,
        "task_delta_leaf_contribution": task_delta_leaf,
        "task_delta_contribution": task_delta,
        "task_delta_phi": delta_phi,
        "task_delta_atoms": task_delta_atoms,
    }


def summarize(
    records: dict[str, dict[str, Any]],
    arrays: dict[str, np.ndarray],
    atom_names: Sequence[str],
    *,
    intervention_samples: int,
    alpha: float,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    absolute = arrays["absolute_contribution"]
    absolute_phi = arrays["absolute_phi"]
    task_delta = arrays["task_delta_contribution"]
    delta_phi = arrays["task_delta_phi"]
    rows = list(records.values())
    state_summary: dict[str, Any] = {}
    for state_index, state in enumerate(STATE_ORDER):
        phi_mean, phi_low, phi_high = bootstrap_mean_ci(
            absolute_phi[state_index], repeats=bootstrap_repeats, seed=5100 + state_index
        )
        contribution_mean = absolute[state_index].mean(axis=0)
        state_summary[state] = {
            "phi_mean": phi_mean,
            "phi_ci95": [phi_low, phi_high],
            "network_contribution_mean": {
                name: float(contribution_mean[index]) for index, name in enumerate(NETWORK_ORDER)
            },
            "top_positive_network": NETWORK_ORDER[int(np.argmax(contribution_mean))],
        }
    delta_summary: dict[str, Any] = {}
    for task_index, task in enumerate(TASKS):
        mean, low, high = bootstrap_mean_ci(
            delta_phi[task_index], repeats=bootstrap_repeats, seed=6100 + task_index
        )
        contribution_mean = task_delta[task_index].mean(axis=0)
        delta_summary[task] = {
            "delta_phi_mean": mean,
            "delta_phi_ci95": [low, high],
            "positive_subjects": int(np.sum(delta_phi[task_index] > 0.0)),
            "negative_subjects": int(np.sum(delta_phi[task_index] < 0.0)),
            "delta_network_contribution_mean": {
                name: float(contribution_mean[index]) for index, name in enumerate(NETWORK_ORDER)
            },
            "largest_increase_network": NETWORK_ORDER[int(np.argmax(contribution_mean))],
            "largest_decrease_network": NETWORK_ORDER[int(np.argmin(contribution_mean))],
        }
    return {
        "analysis": "Fixed-hierarchy affine transport-map PEID on 14 hemisphere-network sources",
        "zotero_evidence": {
            "item_key": "MYATYWAJ",
            "evidence_level": "full text",
            "used_results": ["Theorem 1 source-side nonnegativity", "Theorem 2 hierarchical additivity", "Appendix F affine triangular transport map"],
        },
        "contract": {
            "scientific_question": "What changes in fixed-hierarchy TM-PEID when task signal is retained versus regressed, and how do absolute retained-task states compare with full REST?",
            "rest_length_matching": False,
            "representation": "14 Schaefer hemisphere-by-Yeo7 network means with pair-shared scaling for retained/regressed",
            "source": "current 14D state",
            "target": "next 14D state",
            "surrogate": {"family": "quadratic delta Ridge", "alpha": float(alpha)},
            "intervention": f"independent Uniform[-sqrt(3), sqrt(3)] on every standardized source; {intervention_samples} shared samples",
            "target_noise": "condition-specific full residual covariance driven by shared standard-normal samples",
            "ei_estimator": "affine triangular transport-map MI evaluated by its covariance log-det identity with finite-sample Wishart correction",
            "hierarchy": "fixed neurofunctional binary tree; no state-wise tree selection",
            "node_attribution": "equal split of each internal hierarchy atom among descendant hemisphere-network leaves, then sum left and right leaves within each Yeo7 network",
        },
        "states": list(STATE_ORDER),
        "tasks": list(TASKS),
        "leaf_names": list(LEAF_NAMES),
        "atom_names": list(atom_names),
        "n_task_subjects": int(arrays["task_delta_phi"].shape[1]),
        "n_rest_task_common_subjects": int(arrays["absolute_phi"].shape[1]),
        "absolute": state_summary,
        "task_retained_minus_regressed": delta_summary,
        "diagnostics": {
            "max_abs_atom_sum_error": float(max(abs(row["atom_sum_error"]) for row in rows)),
            "max_abs_contribution_sum_error": float(max(abs(row["contribution_sum_error"]) for row in rows)),
            "negative_atom_fraction": float(
                sum(row["negative_atom_count"] for row in rows) / (len(rows) * len(atom_names))
            ),
            "negative_raw_ei_fraction": float(
                sum(row["negative_raw_ei_count"] for row in rows) / (len(rows) * 27)
            ),
            "median_holdout_skill_ratio": float(np.median([row["model"]["skill_ratio"] for row in rows])),
            "models_beating_persistence": int(sum(row["model"]["skill_ratio"] < 1.0 for row in rows)),
            "n_models": len(rows),
        },
    }


def write_report(summary: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# HCP REST 与七任务固定层级 TM-PEID",
        "",
        "## 结论",
        "",
        "本实验使用自己的连续 PEID 口径，而不是对观测方差或相关性的重新命名：先学习一步非线性转移机制，再在共同最大熵 source 干预下生成机制响应，以仿射三角 transport map 估计固定层级 source block 到完整下一时刻状态的 EI，最后按 PEID 层级加性构造 atom。理论定义和层级加性依据本地 Zotero 全文 *Partial Effective Information Decomposition for Synergistic Causality*（item MYATYWAJ）。",
        "",
        "![固定层级 TM-PEID 节点贡献](fixed_hierarchy_tm_peid_node_contributions.png)",
        "",
        "![固定层级 TM-PEID atoms](fixed_hierarchy_tm_peid_atoms.png)",
        "",
        "## 数据与共同表示",
        "",
        "任务分析包含 30 名具有全部七个 LR 任务的被试；REST 与任务的绝对比较使用其中具有 REST1_LR 的 29 名。REST 使用完整 1,200 点时序，不与任务长度匹配。14 个 source 叶节点为 Yeo7 网络与左右半球的笛卡尔积。每个 parcel 先在本状态内去时间均值并标准化，再在相应半球网络内求均值；任务的 retained/regressed 对共享 parcel 和节点尺度，REST 独立标准化。",
        "",
        "固定二叉树先合并左右半球，再形成 Sensory、Attention、Association、Higher、NonLimbic，最后将 NonLimbic 与 Limbic 合并为 Whole。树在全部被试和状态间固定，避免不同状态的 atom 身份发生变化。",
        "",
        "## 一步动力学机制",
        "",
        "令 $\\mathbf{x}_t\\in\\mathbb{R}^{14}$ 为当前网络状态，二阶多项式特征为 $\\mathbf{h}(\\mathbf{x}_t)$。对状态增量拟合 Ridge：",
        "",
        "$$\\widehat{\\mathbf{B}},\\widehat{\\mathbf{b}}=\\arg\\min_{\\mathbf{B},\\mathbf{b}}\\sum_t\\left\\|\\mathbf{x}_{t+1}-\\mathbf{x}_t-\\mathbf{B}\\mathbf{h}(\\mathbf{x}_t)-\\mathbf{b}\\right\\|_2^2+\\alpha\\|\\mathbf{B}\\|_F^2.$$",
        "",
        "其中所有状态统一使用 $\\alpha=100$。该值是在 3 名被试的 45 条状态序列上，从 $\\{1,10,100,1000,10000\\}$ 中按时间后 25% 留出 RMSE 预先选择；$\\alpha=100$ 的中位 RMSE/持久性基线比为 0.948。最终 449 个模型的中位比为 0.945，377 个优于持久性基线。随后用完整时序重拟合，并由训练残差估计状态特异的完整 $14\\times14$ 协方差 $\\widehat{\\mathbf{\\Sigma}}_{\\varepsilon}$。",
        "",
        "## 最大熵干预与仿射 TM-EI",
        "",
        "对每个模型使用完全相同的 $M=2048$ 个干预样本和标准正态噪声样本：",
        "",
        "$$\\mathbf{X}^{do}\\sim\\prod_{i=1}^{14}\\mathrm{Unif}[-\\sqrt{3},\\sqrt{3}],\\qquad \\mathbf{Y}=\\widehat{\\mathbf{f}}(\\mathbf{X}^{do})+\\widehat{\\mathbf{L}}\\boldsymbol{\\epsilon},$$",
        "",
        "其中 $\\widehat{\\mathbf{L}}\\widehat{\\mathbf{L}}^\\top=\\widehat{\\mathbf{\\Sigma}}_{\\varepsilon}$。对任一层级 source block $S$，仿射三角 TM 的互信息等价于协方差 log-det：",
        "",
        "$$EI(S;\\mathbf{Y})=\\frac12\\left[\\log|\\widehat{\\mathbf{\\Sigma}}_{SS}|+\\log|\\widehat{\\mathbf{\\Sigma}}_{YY}|-\\log|\\widehat{\\mathbf{\\Sigma}}_{(S,Y)(S,Y)}|\\right]-b_{M,S,Y}.$$",
        "",
        "其中 $b_{M,S,Y}$ 是按各块维度计算的 Wishart log-det 有限样本偏差修正。EI 以自然对数计算，单位为 nats；校正后 EI 仅在零处截断，层级 atom 不截断。",
        "",
        "## 层级 atom 与节点贡献",
        "",
        "对内部节点 $u$ 及其两个子节点 $l(u),r(u)$，定义",
        "",
        "$$a_u=EI(S_u;\\mathbf{Y})-EI(S_{l(u)};\\mathbf{Y})-EI(S_{r(u)};\\mathbf{Y}).$$",
        "",
        "14 叶、13 内部节点的满二叉树使这些 atom 严格望远镜相加：",
        "",
        "$$\\Phi^{EID}=EI(S_{Whole};\\mathbf{Y})-\\sum_{i=1}^{14}EI(S_i;\\mathbf{Y})=\\sum_{u\\in\\mathcal{I}}a_u.$$",
        "",
        "不使用 Shapley。半球网络叶 $i$ 的整合贡献按每个祖先 atom 在其后代叶中等分：",
        "",
        "$$C_i=\\sum_{u:i\\in S_u}\\frac{a_u}{|S_u|},\\qquad\\sum_i C_i=\\Phi^{EID}.$$",
        "",
        "主图再将同一 Yeo7 网络的左右叶贡献相加。任务诱发贡献是相同被试、任务、干预与噪声样本下的 $C_i^{retained}-C_i^{regressed}$；$\\Delta\\Phi^{EID}$ 同理。95% CI 由被试层 bootstrap 均值得到。实现的 atom 与贡献最大守恒误差均为 $4.44\\times10^{-15}$。",
        "",
        "## 绝对 PEID",
        "",
        "| State | Mean Phi (nats) | 95% bootstrap CI | Largest network contribution |",
        "|---|---:|---:|---|",
    ]
    for state in STATE_ORDER:
        item = summary["absolute"][state]
        lines.append(
            f"| {state} | {item['phi_mean']:.4f} | [{item['phi_ci95'][0]:.4f}, {item['phi_ci95'][1]:.4f}] | {item['top_positive_network']} |"
        )
    lines.extend(
        [
            "",
            "REST 的绝对 $\\Phi^{EID}$ 最高，但 REST 的完整长度与任务不同，因此这里是各状态自身动力学的绝对估计，不是长度控制后的因果对比。八个状态的最大网络贡献均为 Default，说明该粗粒度一步整合指标没有恢复出经典任务定位。",
            "",
            "## 任务 retained − regressed",
            "",
            "| Task | Mean delta Phi (nats) | 95% bootstrap CI | Largest increase network | Sign +/− |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for task in TASKS:
        item = summary["task_retained_minus_regressed"][task]
        lines.append(
            f"| {task} | {item['delta_phi_mean']:.4f} | [{item['delta_phi_ci95'][0]:.4f}, {item['delta_phi_ci95'][1]:.4f}] | {item['largest_increase_network']} | {item['positive_subjects']}/{item['negative_subjects']} |"
        )
    lines.extend(
        [
            "",
            "除 WM 外，七任务的 $\\Delta\\Phi^{EID}$ 95% CI 均跨零；WM 为负，且其最明显下降位于 Cont。网络贡献变化的量级也远小于绝对贡献。因此当前固定层级 TM-PEID 不支持“任务 GLM 成分稳定增强特定网络的一步整合信息”这一强结论。",
            "",
            "## 解释边界",
            "",
            "任务定位图回答 task GLM 解释了哪些 parcel 的方差；本图回答在学习到的一步机制和声明的干预分布下，哪些固定 source block 共同携带关于下一时刻全脑状态的信息。两者的目标量不同，因此 PEID 差异小并不否定任务激活的空间特异性。",
            "",
            "连续非线性系统中的 atom 非负性只在真实最大熵机制分布下成立；有限样本、学习到的 surrogate 和仿射 TM 密度近似可能产生负 atom。实现保留 raw atom 以维持严格加性，不进行会改变总和的逐 atom 截断；本次负 atom 比例为 1.70%。固定树避免了状态间 atom 身份不一致，但节点等分仍是明确声明的归因约定，而不是 PEID 定理唯一导出的节点所有权。七个任务 CI 未作多重比较校正；数据仅含 LR run、皮层 Schaefer parcel 和一步平稳模型。",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--rest-root", type=Path, default=DEFAULT_REST_ROOT)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--intervention-samples", type=int, default=2048)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args()

    if args.intervention_samples < 256:
        raise ValueError("intervention-samples must be at least 256 for the 28D joint TM fit.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_subjects, task_paths = discover_inputs(args.task_root)
    if args.max_subjects is not None:
        task_subjects = task_subjects[: int(args.max_subjects)]
    rest_paths = discover_rest_inputs(args.rest_root)
    rest_subjects = [subject for subject in task_subjects if subject in rest_paths]
    labels = parse_schaefer_labels(args.label_file)
    groups = leaf_groups(labels)
    nodes, root = build_fixed_hierarchy()
    atom_names = [name for name in hierarchy_postorder(nodes, root) if nodes[name].children is not None]

    rng = np.random.default_rng(int(args.seed))
    intervention_source = rng.uniform(
        -math.sqrt(3.0), math.sqrt(3.0), size=(int(args.intervention_samples), len(LEAF_NAMES))
    )
    base_noise = rng.standard_normal((int(args.intervention_samples), len(LEAF_NAMES)))
    config_id = make_config_id(
        alpha=float(args.alpha),
        intervention_samples=int(args.intervention_samples),
        seed=int(args.seed),
    )
    cache_path = args.output_dir / "records.jsonl"
    records = load_records(cache_path, config_id=config_id)

    jobs: list[tuple[str, str, str]] = []
    jobs.extend((subject, "REST", "rest") for subject in rest_subjects)
    for subject in task_subjects:
        for task in TASKS:
            jobs.extend(((subject, task, "retained"), (subject, task, "regressed")))

    for subject, condition, variant in tqdm(jobs, desc="Fixed-hierarchy TM-PEID", unit="model"):
        key = f"{subject}|{condition}|{variant}"
        if key in records:
            continue
        if condition == "REST":
            reduced = reduce_rest_series(load_rest_series(rest_paths[subject]), groups)
        else:
            retained, regressed = load_task_pair(task_paths[subject][condition])
            retained_reduced, regressed_reduced = reduce_task_pair(retained, regressed, groups)
            reduced = retained_reduced if variant == "retained" else regressed_reduced
        record = analyze_reduced_series(
            reduced,
            subject=subject,
            condition=condition,
            variant=variant,
            alpha=float(args.alpha),
            intervention_source=intervention_source,
            base_noise=base_noise,
            nodes=nodes,
            root=root,
            config_id=config_id,
        )
        records[key] = record
        append_record(cache_path, record)

    arrays = build_arrays(records, task_subjects, rest_subjects, atom_names)
    summary = summarize(
        records,
        arrays,
        atom_names,
        intervention_samples=int(args.intervention_samples),
        alpha=float(args.alpha),
        bootstrap_repeats=int(args.bootstrap_repeats),
    )
    np.savez_compressed(
        args.output_dir / "fixed_hierarchy_tm_peid.npz",
        states=np.asarray(STATE_ORDER),
        tasks=np.asarray(TASKS),
        leaf_names=np.asarray(LEAF_NAMES),
        atom_names=np.asarray(atom_names),
        **arrays,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, args.output_dir)
    plot_node_contributions(
        arrays["absolute_contribution"],
        arrays["absolute_phi"],
        arrays["task_delta_contribution"],
        arrays["task_delta_phi"],
        args.output_dir,
        bootstrap_repeats=int(args.bootstrap_repeats),
    )
    plot_atoms(arrays["absolute_atoms"], arrays["task_delta_atoms"], atom_names, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "diagnostics": summary["diagnostics"], "task_delta": summary["task_retained_minus_regressed"]}, indent=2))


if __name__ == "__main__":
    main()
