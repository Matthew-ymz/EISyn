#!/usr/bin/env python3
"""Scan K_out for one mixed-order Kuramoto model and render the main SPT figure."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.TM.transport_map_density import LOG_2
from scripts.phi_hierarchy import (
    NONNEGATIVE_TOLERANT,
    PhiTreeNode,
    flatten_phi_tree,
    greedy_phi_tree,
    nontrivial_bipartitions,
    subset_phi_raw,
)
from scripts.run_mixed_order_kuramoto_spt_example import (
    NOISE_SCALE,
    PAIRWISE_ASYMMETRY,
    SYN_TOLERANCE_BITS,
    WITHIN_COUPLING,
)
from scripts.synergy_hierarchy_tree_plot import plot_synergy_hierarchy_tree
from scripts import validate_mixed_order_kuramoto_hierarchy as coherent_tm


NAMES = tuple(f"theta{i}" for i in range(1, 7))
PAIRWISE_MODULE = NAMES[:3]
TRIADIC_MODULE = NAMES[3:]
FREQUENCIES = np.array([-0.4, 0.0, 0.4, -0.4, 0.0, 0.4], dtype=float)
DEFAULT_K_OUT = (0.0, 0.04, 5.0)
DYNAMICS_NOISE_SCALE = 0.08
READOUT_NOISE_FLOOR = 0.00
EXPERIMENT_PAIRWISE_ASYMMETRY = 0.25
PREDICTION_HORIZON = 0.20
INTEGRATION_STEP = 0.01
DEFAULT_RESULT = ROOT / "results/mixed_order_kuramoto_kout_main/summary.json"
DEFAULT_FIGURE = ROOT / "docs/reports/assets/kuramoto_hierarchy/kuramoto_mixed_order_kout_complete_spt.png"
DEFAULT_STATUS = ROOT / "docs/log/mixed_order_kuramoto_kout_main/live_progress.json"


def _configure_coherent_tm() -> None:
    """Reuse the learned-dynamics and conditional-TM implementation with six sources."""
    coherent_tm.NAMES = NAMES
    coherent_tm.TRIADIC_MODULE = TRIADIC_MODULE
    coherent_tm.PAIRWISE_MODULE = PAIRWISE_MODULE
    coherent_tm.CONTEXT_SECOND_HARMONIC_NAMES = NAMES
    coherent_tm.FREQUENCIES = FREQUENCIES
    coherent_tm.mixed_order_derivative = _mixed_order_derivative_adapter
    coherent_tm.phase_mlp_features = _mlp_phase_features
    coherent_tm.phase_transport_context = _numpy_phase_context


def _mlp_phase_features(values: np.ndarray) -> np.ndarray:
    """Use the earlier experiment's symmetric first/second-harmonic library."""
    phases = np.asarray(values, dtype=float)
    return np.column_stack(
        [
            component
            for index in range(phases.shape[1])
            for component in (
                np.cos(phases[:, index]),
                np.sin(phases[:, index]),
                np.cos(2.0 * phases[:, index]),
                np.sin(2.0 * phases[:, index]),
            )
        ]
    )


def pairwise_ring_weights(
    asymmetry: float = EXPERIMENT_PAIRWISE_ASYMMETRY,
) -> dict[tuple[int, int], float]:
    """Match the old panel-d fixed-RMS pairwise triangle on nodes 1--3."""
    scale = (WITHIN_COUPLING / 2.0) * math.sqrt(
        3.0 / (1.0 + 2.0 * float(asymmetry) ** 2)
    )
    return {
        (0, 1): scale,
        (0, 2): scale * float(asymmetry),
        (1, 2): scale * float(asymmetry),
    }


NETWORK_POSITIONS = {
    "theta1": (-1.12, 0.72),
    "theta2": (-1.25, -0.62),
    "theta3": (-0.28, -0.05),
    "theta4": (0.28, -0.05),
    "theta5": (1.25, -0.62),
    "theta6": (1.12, 0.72),
}


def render_network(
    output: Path,
    *,
    pairwise_asymmetry: float,
    cross_coupling: float,
    cross_topology: str = "all_to_all",
) -> Path:
    """Render the old panel-d module orientation for the three-regime figure."""
    if cross_topology != "all_to_all":
        raise ValueError("Only all_to_all cross coupling is used here.")
    figure, axis = plt.subplots(figsize=(5.0, 3.8), constrained_layout=True)
    graph = nx.Graph()
    graph.add_nodes_from(NAMES)
    if cross_coupling > 0.0:
        cross_edges = [(left, right) for left in PAIRWISE_MODULE for right in TRIADIC_MODULE]
        graph.add_edges_from(cross_edges)
        nx.draw_networkx_edges(
            graph,
            NETWORK_POSITIONS,
            edgelist=cross_edges,
            edge_color="#B8BDC5",
            width=0.65 + 2.4 * float(cross_coupling) / WITHIN_COUPLING,
            alpha=0.50,
            ax=axis,
        )
    triangle = np.asarray([NETWORK_POSITIONS[name] for name in TRIADIC_MODULE])
    axis.add_patch(
        Polygon(
            triangle,
            closed=True,
            facecolor="#D9903D",
            edgecolor="#B86722",
            alpha=0.28,
            linewidth=3.2,
        )
    )
    weights = pairwise_ring_weights(pairwise_asymmetry)
    for (left_index, right_index), weight in weights.items():
        left, right = NAMES[left_index], NAMES[right_index]
        axis.plot(
            [NETWORK_POSITIONS[left][0], NETWORK_POSITIONS[right][0]],
            [NETWORK_POSITIONS[left][1], NETWORK_POSITIONS[right][1]],
            color="#4477A8",
            linewidth=1.4 + 3.8 * weight / max(weights.values()),
            alpha=0.85,
            solid_capstyle="round",
            zorder=2,
        )
    nx.draw_networkx_nodes(
        graph,
        NETWORK_POSITIONS,
        nodelist=list(PAIRWISE_MODULE),
        node_color="#DCEAF3",
        edgecolors="#355F7A",
        linewidths=1.5,
        node_size=720,
        ax=axis,
    )
    nx.draw_networkx_nodes(
        graph,
        NETWORK_POSITIONS,
        nodelist=list(TRIADIC_MODULE),
        node_color="#F4DDC5",
        edgecolors="#A95E24",
        linewidths=1.5,
        node_size=720,
        ax=axis,
    )
    nx.draw_networkx_labels(
        graph,
        NETWORK_POSITIONS,
        labels={name: rf"$\theta_{{{index}}}$" for index, name in enumerate(NAMES, start=1)},
        font_size=10,
        font_color="#24313C",
        ax=axis,
    )
    axis.text(-0.82, -0.91, "pairwise triangle", ha="center", color="#355F7A", fontsize=9)
    axis.text(0.82, -0.91, "triadic hyperedge", ha="center", color="#A95E24", fontsize=9)
    axis.set(xlim=(-1.62, 1.62), ylim=(-1.05, 1.02))
    axis.set_aspect("equal")
    axis.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def _numpy_context_for_names(
    values: np.ndarray,
    source_names: tuple[str, ...],
) -> np.ndarray:
    """Add matching pairwise/triadic Fourier products for one source block."""
    phases = np.asarray(values, dtype=float)
    columns = [
            component
            for index in range(phases.shape[1])
            for component in (
                np.cos(phases[:, index]),
                np.sin(phases[:, index]),
                np.cos(2.0 * phases[:, index]),
                np.sin(2.0 * phases[:, index]),
            )
        ]
    index = {name: position for position, name in enumerate(source_names)}
    pair_names = tuple(name for name in PAIRWISE_MODULE if name in index)
    for left_name, right_name in itertools.combinations(pair_names, 2):
        left, right = index[left_name], index[right_name]
        columns.append(np.sin(phases[:, right] - phases[:, left]))
    triad_names = tuple(name for name in TRIADIC_MODULE if name in index)
    if len(triad_names) == 3:
        for receiver_name in triad_names:
            senders = [name for name in triad_names if name != receiver_name]
            columns.append(
                np.sin(
                    phases[:, index[senders[0]]]
                    + phases[:, index[senders[1]]]
                    - 2.0 * phases[:, index[receiver_name]]
                )
            )
    for left_name in pair_names:
        for right_name in triad_names:
            columns.append(
                np.sin(phases[:, index[right_name]] - phases[:, index[left_name]])
            )
    return np.column_stack(columns)


def _numpy_phase_context(values: np.ndarray) -> np.ndarray:
    return _numpy_context_for_names(values, NAMES)


def _mixed_order_derivative_adapter(
    phases: np.ndarray,
    *,
    pairwise_coupling: float,
    triadic_coupling: float,
    cross_coupling: float,
    frequencies: np.ndarray = FREQUENCIES,
) -> np.ndarray:
    """Evaluate the planted six-oscillator vector field for the shared simulator."""
    values = np.asarray(phases, dtype=float)
    derivative = np.broadcast_to(np.asarray(frequencies, dtype=float), values.shape).copy()
    for receiver in range(3, 6):
        senders = [index for index in range(3, 6) if index != receiver]
        derivative[:, receiver] += float(triadic_coupling) * np.sin(
            values[:, senders[0]] + values[:, senders[1]] - 2.0 * values[:, receiver]
        )
    pairwise_scale = float(pairwise_coupling) / float(WITHIN_COUPLING)
    for (left, right), weight in pairwise_ring_weights(
        EXPERIMENT_PAIRWISE_ASYMMETRY
    ).items():
        delta = values[:, right] - values[:, left]
        contribution = pairwise_scale * weight * np.sin(delta)
        derivative[:, left] += contribution
        derivative[:, right] -= contribution
    for left in range(3):
        for right in range(3, 6):
            delta = values[:, right] - values[:, left]
            contribution = (float(cross_coupling) / 3.0) * np.sin(delta)
            derivative[:, left] += contribution
            derivative[:, right] -= contribution
    return derivative


def learned_dependency_components(
    fitted: object,
    *,
    seed: int,
    threshold: float = 0.25,
    sample_count: int = 4096,
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Discover oscillator components from source-permutation effects on MLP outputs."""
    rng = np.random.default_rng(int(seed))
    phases = rng.uniform(-math.pi, math.pi, size=(int(sample_count), len(NAMES)))
    baseline = np.asarray(fitted.predict(_mlp_phase_features(phases)), dtype=float)
    output_scale = np.maximum(baseline.std(axis=0), 1.0e-12)
    effects = np.zeros((len(NAMES), len(NAMES)), dtype=float)
    for source in range(len(NAMES)):
        permuted = phases.copy()
        order = rng.permutation(len(phases))
        permuted[:, source] = phases[order, source]
        changed = np.asarray(fitted.predict(_mlp_phase_features(permuted)), dtype=float)
        effects[source] = np.sqrt(
            np.mean(((changed - baseline) / output_scale) ** 2, axis=0)
        )

    neighbors = {index: set() for index in range(len(NAMES))}
    for source in range(len(NAMES)):
        for target in range(len(NAMES)):
            if source != target and effects[source, target] >= float(threshold):
                neighbors[source].add(target)
                neighbors[target].add(source)
    unseen = set(range(len(NAMES)))
    components: list[tuple[int, ...]] = []
    while unseen:
        start = min(unseen)
        stack = [start]
        reached: set[int] = set()
        while stack:
            node = stack.pop()
            if node in reached:
                continue
            reached.add(node)
            stack.extend(sorted(neighbors[node] - reached))
        unseen -= reached
        components.append(tuple(sorted(reached)))
    components.sort(key=lambda component: component[0])
    return components, effects


def future_state_features(phases: np.ndarray) -> np.ndarray:
    """Return the complete six-node future state in an unwrapped phase chart."""
    values = np.asarray(phases, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(NAMES):
        raise ValueError(f"phases must have shape (n_samples, {len(NAMES)}).")
    return values.copy()


def combine_independent_component_ei_tables(
    component_tables: list[tuple[tuple[str, ...], dict[tuple[str, ...], float]]],
) -> dict[tuple[str, ...], float]:
    """Combine EI additively when the learned channel factorizes by components."""
    combined: dict[tuple[str, ...], float] = {}
    for size in range(1, len(NAMES) + 1):
        for subset in itertools.combinations(NAMES, size):
            selected = set(subset)
            value = 0.0
            for component_names, table in component_tables:
                local = tuple(name for name in component_names if name in selected)
                if local:
                    value += float(table[local])
            combined[subset] = value
    return combined


def interaction_components(k_out: float) -> list[tuple[int, ...]]:
    """Find exact channel factors from the nonzero interaction support."""
    neighbors = {index: set() for index in range(len(NAMES))}
    supports = [tuple(range(3, 6))]
    supports.extend(tuple(edge) for edge in pairwise_ring_weights(EXPERIMENT_PAIRWISE_ASYMMETRY))
    if not np.isclose(float(k_out), 0.0):
        supports.extend((left, right) for left in range(3) for right in range(3, 6))
    for support in supports:
        for left, right in itertools.combinations(support, 2):
            neighbors[left].add(right)
            neighbors[right].add(left)
    unseen = set(neighbors)
    components: list[tuple[int, ...]] = []
    while unseen:
        start = min(unseen)
        reached: set[int] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in reached:
                continue
            reached.add(node)
            stack.extend(sorted(neighbors[node] - reached))
        unseen -= reached
        components.append(tuple(sorted(reached)))
    return components


def _phase_context(values: object, source_names: tuple[str, ...]) -> object:
    import torch

    components: list[object] = []
    for index in range(values.shape[1]):
        components.extend(
            [
                torch.cos(values[:, index : index + 1]),
                torch.sin(values[:, index : index + 1]),
                torch.cos(2.0 * values[:, index : index + 1]),
                torch.sin(2.0 * values[:, index : index + 1]),
            ]
        )
    name_index = {name: index for index, name in enumerate(source_names)}
    pair_names = tuple(name for name in PAIRWISE_MODULE if name in name_index)
    for left_name, right_name in itertools.combinations(pair_names, 2):
        left, right = name_index[left_name], name_index[right_name]
        components.append(
            torch.sin(values[:, right : right + 1] - values[:, left : left + 1])
        )
    triad_names = tuple(name for name in TRIADIC_MODULE if name in name_index)
    if len(triad_names) == 3:
        for receiver_name in triad_names:
            senders = [name for name in triad_names if name != receiver_name]
            receiver = name_index[receiver_name]
            first, second = name_index[senders[0]], name_index[senders[1]]
            components.append(
                torch.sin(
                    values[:, first : first + 1]
                    + values[:, second : second + 1]
                    - 2.0 * values[:, receiver : receiver + 1]
                )
            )
    for left_name in pair_names:
        for right_name in triad_names:
            left, right = name_index[left_name], name_index[right_name]
            components.append(
                torch.sin(values[:, right : right + 1] - values[:, left : left + 1])
            )
    return torch.cat(components, dim=1)


def coherent_ei_table_without_resampling(
    flow: object,
    *,
    source_names: tuple[str, ...] = NAMES,
    seed: int,
    evaluation_count: int,
    marginal_samples: int,
    context_active: bool,
) -> tuple[dict[tuple[str, ...], float], dict[str, object]]:
    """Marginalize all 63 EIs from one joint TM without resampling correction."""
    import torch

    if not context_active:
        table = {
            subset: 0.0
            for size in range(1, len(source_names) + 1)
            for subset in __import__("itertools").combinations(source_names, size)
        }
        return table, {
            "minimum_non_singleton_xi_bits": 0.0,
            "minimum_partition_syn_bits": 0.0,
            "negative_non_singleton_xi_count": 0,
            "negative_partition_syn_count": 0,
            "violating_non_singleton_xi_count": 0,
            "violating_partition_syn_count": 0,
            "evaluation_count": int(evaluation_count),
            "marginal_samples": int(marginal_samples),
        }

    source_engine = torch.quasirandom.SobolEngine(
        len(source_names), scramble=True, seed=int(seed)
    )
    phases = source_engine.draw(int(evaluation_count)) * (2.0 * math.pi) - math.pi
    source_context = _phase_context(phases, source_names)
    torch.manual_seed(int(seed) + 2)
    with torch.no_grad():
        target = flow.sample(1, context=source_context).squeeze(1)
    integration_engine = torch.quasirandom.SobolEngine(
        len(source_names), scramble=True, seed=int(seed) + 1
    )
    base = (
        integration_engine.draw(int(evaluation_count) * int(marginal_samples))
        * (2.0 * math.pi)
        - math.pi
    ).reshape(int(evaluation_count), int(marginal_samples), len(source_names))

    def marginal_log_probability(columns: tuple[int, ...]) -> object:
        if len(columns) == len(source_names):
            with torch.no_grad():
                return flow.log_prob(target, context=source_context)
        row_chunk = max(1, 16_384 // int(marginal_samples))
        rows: list[object] = []
        with torch.no_grad():
            for start in range(0, int(evaluation_count), row_chunk):
                stop = min(int(evaluation_count), start + row_chunk)
                completed = base[start:stop].clone()
                if columns:
                    completed[:, :, list(columns)] = phases[
                        start:stop, None, list(columns)
                    ]
                flat_context = _phase_context(
                    completed.reshape(-1, len(source_names)),
                    source_names,
                )
                flat_target = (
                    target[start:stop, None, :]
                    .expand(stop - start, int(marginal_samples), target.shape[1])
                    .reshape(-1, target.shape[1])
                )
                log_likelihood = flow.log_prob(
                    flat_target, context=flat_context
                ).reshape(stop - start, int(marginal_samples))
                rows.append(
                    torch.logsumexp(log_likelihood, dim=1)
                    - math.log(int(marginal_samples))
                )
        return torch.cat(rows)

    target_log_probability = marginal_log_probability(())
    table: dict[tuple[str, ...], float] = {}
    for size in range(1, len(source_names) + 1):
        for subset in __import__("itertools").combinations(source_names, size):
            columns = tuple(source_names.index(name) for name in subset)
            pointwise = (
                marginal_log_probability(columns) - target_log_probability
            ) / LOG_2
            table[tuple(subset)] = float(pointwise.mean().item())

    singleton = {name: float(table[(name,)]) for name in source_names}
    xi = {
        subset: subset_phi_raw(subset, table, singleton)
        for subset in table
    }
    non_singleton = [value for subset, value in xi.items() if len(subset) > 1]
    partition_syn = [
        xi[subset] - xi[tuple(left)] - xi[tuple(right)]
        for subset in xi
        if len(subset) > 1
        for left, right in nontrivial_bipartitions(subset)
    ]
    minimum_xi = float(min(non_singleton))
    minimum_syn = float(min(partition_syn))
    negative_xi_count = int(sum(value < 0.0 for value in non_singleton))
    negative_syn_count = int(sum(value < 0.0 for value in partition_syn))
    violating_xi_count = int(
        sum(value < -SYN_TOLERANCE_BITS for value in non_singleton)
    )
    violating_syn_count = int(
        sum(value < -SYN_TOLERANCE_BITS for value in partition_syn)
    )
    if minimum_xi < -SYN_TOLERANCE_BITS or minimum_syn < -SYN_TOLERANCE_BITS:
        raise RuntimeError(
            "Coherent joint-TM nonnegativity violation: "
            f"minimum Xi={minimum_xi:.6g} bits, minimum Syn={minimum_syn:.6g} bits, "
            f"threshold=-{SYN_TOLERANCE_BITS:.6g} bits."
        )
    return table, {
        "minimum_non_singleton_xi_bits": minimum_xi,
        "minimum_partition_syn_bits": minimum_syn,
        "negative_non_singleton_xi_count": negative_xi_count,
        "negative_partition_syn_count": negative_syn_count,
        "violating_non_singleton_xi_count": violating_xi_count,
        "violating_partition_syn_count": violating_syn_count,
        "evaluation_count": int(evaluation_count),
        "marginal_samples": int(marginal_samples),
        "resampling_bias_correction": "none",
        "nonnegative_clipping": False,
    }


def free_spt(table: dict[tuple[str, ...], float]):
    """Build a complete SPT with every nontrivial bipartition eligible."""
    return greedy_phi_tree(
        NAMES,
        table,
        policy=NONNEGATIVE_TOLERANT,
        split_tolerance=SYN_TOLERANCE_BITS,
        complete_to_singletons=True,
    )


def _write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _condition_record(condition: dict[str, object]) -> dict[str, object]:
    tree = condition["tree"]
    return {
        "k_out": float(condition["k_out"]),
        "root_xi_bits": float(tree.phi_value),
        "root_syn_bits": float(tree.residual),
        "root_split": condition["root_split"],
        "planted_root_split": bool(condition["planted_root_split"]),
        "minimum_atom_bits": float(condition["minimum_atom_bits"]),
        "closure_error_bits": float(condition["closure_error_bits"]),
        "dynamics_diagnostics": condition["dynamics_diagnostics"],
        "fit_diagnostics": condition["fit_diagnostics"],
        "marginal_diagnostics": condition["marginal_diagnostics"],
        "elapsed_seconds": float(condition["elapsed_seconds"]),
        "atoms": [
            {
                "sources": list(atom.sources),
                "value_bits": float(atom.value),
                "depth": int(atom.depth),
            }
            for atom in flatten_phi_tree(tree)
        ],
    }


def tree_from_record(record: dict[str, object]) -> PhiTreeNode:
    """Reconstruct a rendered free-split tree from a saved condition record."""
    atom_values = {
        tuple(atom["sources"]): float(atom["value_bits"])
        for atom in record["atoms"]
    }
    source_rank = {name: index for index, name in enumerate(NAMES)}

    def visit(sources: tuple[str, ...], depth: int) -> PhiTreeNode:
        if len(sources) == 1:
            return PhiTreeNode(
                sources=sources,
                xi_value=0.0,
                syn_value=0.0,
                depth=depth,
                split_kind="leaf",
                children=(),
            )
        source_set = set(sources)
        proper = [
            coalition
            for coalition in atom_values
            if set(coalition) < source_set
        ]
        direct = [
            coalition
            for coalition in proper
            if not any(set(coalition) < set(other) < source_set for other in proper)
        ]
        covered = set().union(*(set(coalition) for coalition in direct)) if direct else set()
        child_sources = [*direct, *((name,) for name in sources if name not in covered)]
        child_sources.sort(key=lambda coalition: min(source_rank[name] for name in coalition))
        children = tuple(visit(tuple(coalition), depth + 1) for coalition in child_sources)
        syn = float(atom_values.get(sources, 0.0))
        xi = float(syn + sum(child.xi_value for child in children))
        return PhiTreeNode(
            sources=sources,
            xi_value=xi,
            syn_value=syn,
            depth=depth,
            split_kind="free_exact" if children else "leaf",
            children=children,
        )

    tree = visit(NAMES, 0)
    if not np.isclose(tree.xi_value, float(record["root_xi_bits"]), atol=1.0e-6):
        raise ValueError(
            f"Cached tree closure mismatch: reconstructed={tree.xi_value}, "
            f"saved={record['root_xi_bits']}."
        )
    return tree


def combine_condition_results(paths: list[Path], *, result: Path, figure: Path) -> dict[str, object]:
    records: list[dict[str, object]] = []
    contracts: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "conditions" in payload:
            if not payload["conditions"]:
                raise ValueError(f"No conditions found in {path}.")
            records.extend(dict(condition) for condition in payload["conditions"])
            contracts.append(dict(payload["experiment_contract"]))
        else:
            records.append(dict(payload))
    fixed_keys = (
        "training_count",
        "readout_count",
        "seed",
        "noise_scale",
        "readout_noise_floor",
        "model_epochs",
        "tm_epochs",
        "evaluation_count",
        "marginal_samples",
        "triadic_coupling",
        "pairwise_asymmetry",
        "estimator",
        "split_search",
        "resampling_bias_correction",
    )
    reference = contracts[0]
    for contract in contracts[1:]:
        differing = [key for key in fixed_keys if contract.get(key) != reference.get(key)]
        if differing:
            raise ValueError(f"Controlled-comparison mismatch in {differing}.")
    records.sort(key=lambda row: float(row["k_out"]))
    conditions = [{"k_out": row["k_out"], "tree": tree_from_record(row)} for row in records]
    render_main_figure(conditions, figure)
    combined = {
        "experiment_contract": {**reference, "k_out_values": [float(row["k_out"]) for row in records]},
        "conditions": records,
        "figure": str(figure),
    }
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    return combined


def _crop_white(image: np.ndarray, pad: int = 16) -> np.ndarray:
    occupied = np.any(np.asarray(image)[..., :3] < 0.985, axis=2)
    rows, columns = np.where(occupied)
    return image[
        max(0, rows.min() - pad):min(image.shape[0], rows.max() + pad + 1),
        max(0, columns.min() - pad):min(image.shape[1], columns.max() + pad + 1),
    ]


def render_main_figure(
    conditions: list[dict[str, object]],
    output: Path,
) -> Path:
    work = output.parent / ".mixed_order_kout_work"
    work.mkdir(parents=True, exist_ok=True)
    scale_max = max(
        (
            abs(float(atom.value))
            for condition in conditions
            for atom in flatten_phi_tree(condition["tree"])
        ),
        default=1.0,
    )
    panels: list[tuple[Path, Path]] = []
    for index, condition in enumerate(conditions):
        k_out = float(condition["k_out"])
        network_path = render_network(
            work / f"network_{index}.png",
            pairwise_asymmetry=EXPERIMENT_PAIRWISE_ASYMMETRY,
            cross_coupling=k_out,
            cross_topology="all_to_all",
        )
        tree_path = plot_synergy_hierarchy_tree(
            condition["tree"],
            work / f"tree_{index}.png",
            source_labels={name: rf"$\theta_{{{i}}}$" for i, name in enumerate(NAMES, start=1)},
            decimals=2,
            syn_scale_max=scale_max,
            dpi=450,
        )
        panels.append((network_path, tree_path))

    figure, axes = plt.subplots(
        len(conditions),
        2,
        figsize=(12.8, 3.55 * len(conditions)),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.84, 1.46]},
    )
    axes = np.atleast_2d(axes)
    axes[0, 0].set_title("Mixed-order coupling network", fontsize=13, pad=8)
    axes[0, 1].set_title("Complete SPT hierarchy", fontsize=13, pad=8)
    for row, (condition, paths) in enumerate(zip(conditions, panels, strict=True)):
        for axis, path in zip(axes[row], paths, strict=True):
            axis.imshow(_crop_white(plt.imread(path)))
            axis.axis("off")
        axes[row, 0].text(
            -0.03,
            0.50,
            rf"$K_{{\mathrm{{out}}}}={float(condition['k_out']):g}$",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=11,
            color="#24313C",
            rotation=90,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    for pair in panels:
        for path in pair:
            path.unlink(missing_ok=True)
    work.rmdir()
    return output


def run(
    *,
    training_count: int,
    readout_count: int,
    seed: int,
    model_epochs: int,
    tm_epochs: int,
    evaluation_count: int,
    marginal_samples: int,
    triadic_scale: float,
    k_out_values: tuple[float, ...],
    result: Path,
    figure: Path,
    status: Path,
) -> dict[str, object]:
    _configure_coherent_tm()
    started = time.monotonic()
    conditions: list[dict[str, object]] = []
    total = len(k_out_values)
    for index, k_out in enumerate(k_out_values):
        condition_started = time.monotonic()
        _write_status(
            status,
            {
                "phase": "running",
                "current": index,
                "total": total,
                "unit": "condition",
                "current_k_out": float(k_out),
                "elapsed_seconds": time.monotonic() - started,
                "updated_at": time.time(),
            },
        )
        from scripts.classic_network_dynamics_benchmark import fit_mlp

        print(f"[{index + 1}/{total}] generating finite-time transitions at K_out={k_out:g}", flush=True)
        training_phases, training_future = coherent_tm.generated_transition_data(
            sample_count=int(training_count),
            seed=int(seed) + 1_000,
            pairwise_coupling=WITHIN_COUPLING,
            triadic_coupling=(WITHIN_COUPLING / math.sqrt(2.0)) * float(triadic_scale),
            cross_coupling=float(k_out),
            process_noise=DYNAMICS_NOISE_SCALE,
            tau=PREDICTION_HORIZON,
            dt=INTEGRATION_STEP,
        )
        training_increment = np.angle(
            np.exp(1j * (training_future - training_phases))
        )
        print(f"[{index + 1}/{total}] fitting MLP dynamics at K_out={k_out:g}", flush=True)
        training_features = _mlp_phase_features(training_phases)
        fitted = fit_mlp(
            training_features,
            training_increment,
            seed=int(seed) + 2_000,
            epochs=int(model_epochs),
        )
        fit_split = max(32, int(0.8 * len(training_features)))
        heldout_target = training_increment[fit_split:]
        heldout_prediction = np.asarray(
            fitted.predict(training_features[fit_split:]), dtype=float
        )
        heldout_residual = heldout_target - heldout_prediction
        residual_covariance = np.asarray(
            np.cov(heldout_residual, rowvar=False),
            dtype=float,
        )
        residual_covariance += 1.0e-6 * np.eye(len(NAMES))
        per_source_r2 = {
            name: float(
                1.0
                - np.mean(heldout_residual[:, source] ** 2)
                / max(np.var(heldout_target[:, source]), 1.0e-12)
            )
            for source, name in enumerate(NAMES)
        }
        learned_components, dependency_effects = learned_dependency_components(
            fitted,
            seed=int(seed) + 2_500,
        )
        dynamics_diagnostics = {
            "heldout_mae": float(np.mean(np.abs(heldout_residual))),
            "heldout_mse": float(np.mean(heldout_residual**2)),
            "constant_baseline_mse": float(fitted.baseline_mse),
            "per_source_r2": per_source_r2,
            "dependency_threshold_output_sd": 0.25,
            "dependency_effect_matrix": dependency_effects.tolist(),
            "learned_dependency_components": [
                [NAMES[source] for source in component]
                for component in learned_components
            ],
            "residual_noise_model": "the earlier experiment's full held-out MLP residual covariance",
            "readout_noise_floor": READOUT_NOISE_FLOOR,
        }
        phases = np.random.default_rng(int(seed) + 3_000).uniform(
            -math.pi,
            math.pi,
            size=(int(readout_count), len(NAMES)),
        )
        readout_rng = np.random.default_rng(int(seed) + 4_000)
        predicted_increment = np.asarray(
            fitted.predict(_mlp_phase_features(phases)), dtype=float
        )
        sampled_increment = predicted_increment + readout_rng.multivariate_normal(
            mean=np.zeros(len(NAMES), dtype=float),
            cov=residual_covariance,
            size=len(phases),
        )
        target = sampled_increment
        print(
            f"[{index + 1}/{total}] fitting {len(learned_components)} probability TM component(s) "
            f"at K_out={k_out:g}",
            flush=True,
        )
        component_tables: list[
            tuple[tuple[str, ...], dict[tuple[str, ...], float]]
        ] = []
        component_fit_diagnostics: list[dict[str, object]] = []
        component_marginal_diagnostics: list[dict[str, object]] = []
        for component_index, component in enumerate(learned_components):
            component_names = tuple(NAMES[source] for source in component)
            coherent_tm.phase_transport_context = (
                lambda values, names=component_names: _numpy_context_for_names(values, names)
            )
            target_positions = tuple(
                source for source in component
            )
            flow, local_fit = coherent_tm.fit_joint_conditional_transport_map(
                phases[:, component],
                target[:, target_positions],
                train_fraction=0.8,
                epochs=int(tm_epochs),
                seed=int(seed) + 5_000 + component_index,
                target_scaling="per_dimension",
            )
            local_table, local_marginal = coherent_ei_table_without_resampling(
                flow,
                source_names=component_names,
                seed=int(seed) + 6_000 + component_index,
                evaluation_count=int(evaluation_count),
                marginal_samples=int(marginal_samples),
                context_active=bool(local_fit["context_active"]),
            )
            component_tables.append((component_names, local_table))
            component_fit_diagnostics.append(
                {
                    "sources": list(component_names),
                    "target_feature_indices": list(target_positions),
                    **local_fit,
                }
            )
            component_marginal_diagnostics.append(
                {"sources": list(component_names), **local_marginal}
            )
            del flow
        table = combine_independent_component_ei_tables(component_tables)
        fit_diagnostics = {
            "factorization": "data-driven MLP permutation-effect components at one fixed 0.25-output-SD threshold; SPT split search remains unconstrained",
            "components": component_fit_diagnostics,
            "context_active": any(
                bool(row["context_active"]) for row in component_fit_diagnostics
            ),
        }
        marginal_diagnostics = {
            "components": component_marginal_diagnostics,
            "minimum_non_singleton_xi_bits": float(
                min(
                    row["minimum_non_singleton_xi_bits"]
                    for row in component_marginal_diagnostics
                )
            ),
            "minimum_partition_syn_bits": float(
                min(
                    row["minimum_partition_syn_bits"]
                    for row in component_marginal_diagnostics
                )
            ),
            "negative_non_singleton_xi_count": int(
                sum(
                    row["negative_non_singleton_xi_count"]
                    for row in component_marginal_diagnostics
                )
            ),
            "negative_partition_syn_count": int(
                sum(
                    row["negative_partition_syn_count"]
                    for row in component_marginal_diagnostics
                )
            ),
            "violating_non_singleton_xi_count": int(
                sum(
                    row["violating_non_singleton_xi_count"]
                    for row in component_marginal_diagnostics
                )
            ),
            "violating_partition_syn_count": int(
                sum(
                    row["violating_partition_syn_count"]
                    for row in component_marginal_diagnostics
                )
            ),
            "nonnegative_clipping": False,
            "resampling_bias_correction": "none",
        }
        tree = free_spt(table)
        if np.isclose(float(k_out), 0.0) and abs(float(tree.residual)) > SYN_TOLERANCE_BITS:
            raise RuntimeError(
                "Disconnected-module EI zero-point violation at K_out=0: "
                f"root Syn={tree.residual:.6g} bits, "
                f"tolerance={SYN_TOLERANCE_BITS:.6g} bits."
            )
        atoms = flatten_phi_tree(tree)
        root_split = [list(child.sources) for child in tree.children]
        planted = {
            frozenset(TRIADIC_MODULE),
            frozenset(PAIRWISE_MODULE),
        }
        conditions.append(
            {
                "k_out": float(k_out),
                "tree": tree,
                "root_split": root_split,
                "planted_root_split": {
                    frozenset(child.sources) for child in tree.children
                } == planted,
                "minimum_atom_bits": float(min((atom.value for atom in atoms), default=0.0)),
                "closure_error_bits": float(sum(atom.value for atom in atoms) - tree.phi_value),
                "dynamics_diagnostics": dynamics_diagnostics,
                "fit_diagnostics": fit_diagnostics,
                "marginal_diagnostics": marginal_diagnostics,
                "elapsed_seconds": time.monotonic() - condition_started,
            }
        )
        condition_record = _condition_record(conditions[-1])
        k_out_slug = f"{float(k_out):g}".replace(".", "p")
        cache_name = f"{result.stem}_kout_{k_out_slug}.json"
        cache_path = result.parent / cache_name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(condition_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        del fitted
        gc.collect()
        elapsed = time.monotonic() - started
        rate = (index + 1) / elapsed
        _write_status(
            status,
            {
                "phase": "running" if index + 1 < total else "rendering",
                "current": index + 1,
                "total": total,
                "unit": "condition",
                "current_k_out": float(k_out),
                "elapsed_seconds": elapsed,
                "eta_seconds": (total - index - 1) / rate if rate > 0 else None,
                "metrics": {
                    "root_split": root_split,
                    "planted_root_split": bool(conditions[-1]["planted_root_split"]),
                    "root_syn_bits": float(tree.residual),
                    "dynamics_mae": float(dynamics_diagnostics["heldout_mae"]),
                },
                "completed_conditions": [
                    {
                        "k_out": float(row["k_out"]),
                        "root_split": row["root_split"],
                        "planted_root_split": bool(row["planted_root_split"]),
                    }
                    for row in conditions
                ],
                "updated_at": time.time(),
            },
        )

    render_main_figure(conditions, figure)
    payload = {
        "experiment_contract": {
            "question": "What hierarchy is selected when only K_out changes?",
            "training_count": int(training_count),
            "readout_count": int(readout_count),
            "seed": int(seed),
            "noise_scale": DYNAMICS_NOISE_SCALE,
            "readout_noise_floor": READOUT_NOISE_FLOOR,
            "prediction_horizon": PREDICTION_HORIZON,
            "integration_step": INTEGRATION_STEP,
            "model_epochs": int(model_epochs),
            "tm_epochs": int(tm_epochs),
            "evaluation_count": int(evaluation_count),
            "marginal_samples": int(marginal_samples),
            "k_out_values": list(map(float, k_out_values)),
            "triadic_scale": float(triadic_scale),
            "triadic_coupling": float((1.5 / math.sqrt(2.0)) * triadic_scale),
            "pairwise_asymmetry": EXPERIMENT_PAIRWISE_ASYMMETRY,
            "pairwise_edge_weights": {
                f"{NAMES[left]}-{NAMES[right]}": float(weight)
                for (left, right), weight in pairwise_ring_weights(
                    EXPERIMENT_PAIRWISE_ASYMMETRY
                ).items()
            },
            "cross_edge_weight": "K_out / 3",
            "mlp_source_features": "the same symmetric first- and second-harmonic circular features used in the earlier panel-d experiment",
            "tm_source_context": "the old symmetric circular features plus explicit pairwise, triadic, and candidate cross-edge Fourier terms, fixed across K_out",
            "target": "the same complete six-dimensional wrapped future phase increment over tau for every source subset; target never changes during SPT splitting",
            "pipeline": [
                "generate paired noisy finite-time Kuramoto transitions",
                "fit one full-state MLP transition model per K_out with fixed architecture and budget",
                "evaluate the MLP under independent maximum-entropy phase interventions",
                "reconstruct the complete six-node future state and fit one conditional transport map per learned dependency component",
                "derive all 63 subset EIs from component-consistent marginals and build the free SPT",
            ],
            "estimator": "MLP dynamics followed by one conditional neural transport map per learned dependency component; all 63 subset EIs use the same component rule and fixed estimator budget",
            "dependency_component_rule": "connect a source pair when its MLP permutation effect exceeds 0.25 output standard deviations; take connected components",
            "split_search": "all nontrivial bipartitions at every node",
            "resampling_bias_correction": "none",
            "syn_nonnegative_tolerance_bits": SYN_TOLERANCE_BITS,
        },
        "conditions": [_condition_record(condition) for condition in conditions],
        "figure": str(figure),
    }
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_status(
        status,
        {
            "phase": "complete",
            "current": total,
            "total": total,
            "unit": "condition",
            "elapsed_seconds": time.monotonic() - started,
            "result": str(result),
            "figure": str(figure),
            "updated_at": time.time(),
        },
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--training-count", type=int, default=None)
    parser.add_argument("--readout-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-epochs", type=int, default=None)
    parser.add_argument("--tm-epochs", type=int, default=None)
    parser.add_argument("--evaluation-count", type=int, default=None)
    parser.add_argument("--marginal-samples", type=int, default=None)
    parser.add_argument("--triadic-scale", type=float, default=1.0)
    parser.add_argument("--k-out", type=float, nargs="+", default=DEFAULT_K_OUT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--combine-results", type=Path, nargs="+")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.combine_results:
        combined = combine_condition_results(
            list(args.combine_results), result=args.result, figure=args.figure
        )
        print(
            json.dumps(
                [
                    {
                        key: condition[key]
                        for key in (
                            "k_out",
                            "root_xi_bits",
                            "root_syn_bits",
                            "root_split",
                            "planted_root_split",
                        )
                    }
                    for condition in combined["conditions"]
                ],
                indent=2,
            )
        )
        raise SystemExit(0)
    smoke = args.mode == "smoke"
    try:
        summary = run(
            training_count=int(args.training_count or (2400 if smoke else 4800)),
            readout_count=int(args.readout_count or (2400 if smoke else 4000)),
            seed=args.seed,
            model_epochs=int(args.model_epochs or (160 if smoke else 800)),
            tm_epochs=int(args.tm_epochs or (60 if smoke else 100)),
            evaluation_count=int(args.evaluation_count or (256 if smoke else 512)),
            marginal_samples=int(args.marginal_samples or (128 if smoke else 256)),
            triadic_scale=float(args.triadic_scale),
            k_out_values=tuple(args.k_out),
            result=args.result,
            figure=args.figure,
            status=args.status,
        )
    except Exception as error:
        _write_status(
            args.status,
            {
                "phase": "failed",
                "message": str(error),
                "updated_at": time.time(),
            },
        )
        raise
    print(
        json.dumps(
            [
                {
                    key: condition[key]
                    for key in (
                        "k_out",
                        "root_xi_bits",
                        "root_syn_bits",
                        "root_split",
                        "planted_root_split",
                        "minimum_atom_bits",
                    )
                }
                for condition in summary["conditions"]
            ],
            indent=2,
        )
    )
