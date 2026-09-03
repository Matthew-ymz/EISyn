#!/usr/bin/env python3
"""Build a complete SPT for triadic-hyperedge versus pairwise-ring modules."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.TM.transport_map_density import LOG_2, fit_polynomial_triangular_transport_map_density
from scripts.phi_hierarchy import PhiTreeNode, all_nonempty_subsets, flatten_phi_tree, greedy_phi_tree
from scripts.synergy_hierarchy_tree_plot import plot_synergy_hierarchy_tree


NAMES = tuple(f"theta{i}" for i in range(1, 7))
TRIADIC_MODULE = NAMES[:3]
PAIRWISE_MODULE = NAMES[3:]
FREQUENCIES = np.array([-0.4, 0.0, 0.4, -0.4, 0.0, 0.4], dtype=float)
WITHIN_COUPLING = 1.5
TRIADIC_COUPLING = WITHIN_COUPLING / math.sqrt(2.0)
PAIRWISE_ASYMMETRY = 0.10
CROSS_COUPLING = 0.0
NOISE_SCALE = 0.08
SYN_TOLERANCE_BITS = 0.10
DEFAULT_RESULT = ROOT / "results/mixed_order_kuramoto_spt_example/summary.json"
DEFAULT_FIGURE = ROOT / "docs/reports/assets/kuramoto_hierarchy/mixed_order_complete_spt.png"


def pairwise_ring_weights(
    asymmetry: float = PAIRWISE_ASYMMETRY,
) -> dict[tuple[int, int], float]:
    scale = (WITHIN_COUPLING / 2.0) * math.sqrt(
        3.0 / (1.0 + 2.0 * float(asymmetry) ** 2)
    )
    return {
        (3, 4): scale,
        (3, 5): scale * float(asymmetry),
        (4, 5): scale * float(asymmetry),
    }


def mixed_order_derivative(
    phases: np.ndarray,
    *,
    pairwise_asymmetry: float = PAIRWISE_ASYMMETRY,
    triadic_scale: float = 1.0,
    cross_coupling: float = CROSS_COUPLING,
) -> np.ndarray:
    values = np.asarray(phases, dtype=float)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("phases must have shape (n_samples, 6).")
    derivative = np.broadcast_to(FREQUENCIES, values.shape).copy()
    for receiver in range(3):
        senders = [index for index in range(3) if index != receiver]
        derivative[:, receiver] += (TRIADIC_COUPLING * float(triadic_scale)) * np.sin(
            values[:, senders[0]] + values[:, senders[1]] - 2.0 * values[:, receiver]
        )
    for (left, right), weight in pairwise_ring_weights(pairwise_asymmetry).items():
        delta = values[:, right] - values[:, left]
        derivative[:, left] += weight * np.sin(delta)
        derivative[:, right] -= weight * np.sin(delta)
    for left in range(3):
        for right in range(3, 6):
            delta = values[:, right] - values[:, left]
            contribution = (float(cross_coupling) / 3.0) * np.sin(delta)
            derivative[:, left] += contribution
            derivative[:, right] -= contribution
    return derivative


def paired_data(
    *,
    sample_count: int,
    seed: int,
    pairwise_asymmetry: float = PAIRWISE_ASYMMETRY,
    triadic_scale: float = 1.0,
    cross_coupling: float = CROSS_COUPLING,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    phases = rng.uniform(-math.pi, math.pi, size=(int(sample_count), 6))
    target = mixed_order_derivative(
        phases,
        pairwise_asymmetry=pairwise_asymmetry,
        triadic_scale=triadic_scale,
        cross_coupling=cross_coupling,
    ) + NOISE_SCALE * rng.normal(size=phases.shape)
    return phases, target


def source_blocks(phases: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(phases, dtype=float)
    return {
        name: np.column_stack(
            (
                np.cos(values[:, index]),
                np.sin(values[:, index]),
                np.cos(2.0 * values[:, index]),
                np.sin(2.0 * values[:, index]),
            )
        )
        for index, name in enumerate(NAMES)
    }


def estimate_ei_table(
    phases: np.ndarray,
    target: np.ndarray,
    *,
    module: tuple[str, ...],
    degree: int,
) -> dict[tuple[str, ...], float]:
    blocks = source_blocks(phases)
    target_model = fit_polynomial_triangular_transport_map_density(target, degree=int(degree))
    target_log_prob = target_model.log_prob(target)
    table: dict[tuple[str, ...], float] = {}
    for subset in all_nonempty_subsets(module):
        source = np.column_stack([blocks[name] for name in subset])
        joint = np.column_stack((target, source))
        joint_model = fit_polynomial_triangular_transport_map_density(joint, degree=int(degree))
        source_model = fit_polynomial_triangular_transport_map_density(source, degree=int(degree))
        pointwise = (
            joint_model.log_prob(joint)
            - source_model.log_prob(source)
            - target_log_prob
        ) / LOG_2
        table[subset] = float(np.mean(pointwise))
    return table


def build_module_tree(module: tuple[str, ...], table: dict[tuple[str, ...], float]):
    return greedy_phi_tree(
        module,
        table,
        split_tolerance=SYN_TOLERANCE_BITS,
        complete_to_singletons=True,
        depth=1,
    )


def build_tree(tables: dict[str, dict[tuple[str, ...], float]]):
    triadic = build_module_tree(TRIADIC_MODULE, tables["triadic"])
    pairwise = build_module_tree(PAIRWISE_MODULE, tables["pairwise"])
    tree = PhiTreeNode(
        sources=NAMES,
        xi_value=float(triadic.phi_value + pairwise.phi_value),
        syn_value=0.0,
        depth=0,
        split_kind="exact_disconnected_modules",
        children=(triadic, pairwise),
    )
    atoms = flatten_phi_tree(tree)
    below = [atom for atom in atoms if atom.value < -SYN_TOLERANCE_BITS]
    if below:
        minimum = min(atom.value for atom in below)
        raise RuntimeError(
            f"Syn nonnegativity violation: min={minimum:.6g} bits, "
            f"threshold=-{SYN_TOLERANCE_BITS:.6g}, affected={len(below)}."
        )
    return tree


POSITIONS = {
    "theta1": (-1.12, 0.72), "theta2": (-1.25, -0.62), "theta3": (-0.28, -0.05),
    "theta4": (0.28, -0.05), "theta5": (1.25, -0.62), "theta6": (1.12, 0.72),
}


def render_network(
    output: Path,
    *,
    pairwise_asymmetry: float,
    cross_coupling: float = CROSS_COUPLING,
    cross_topology: str = "all_to_all",
) -> Path:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
    figure, axis = plt.subplots(figsize=(5.0, 3.8), constrained_layout=True)
    graph = nx.Graph()
    graph.add_nodes_from(NAMES)
    if cross_coupling > 0.0:
        if cross_topology == "all_to_all":
            cross_edges = [
                (left, right)
                for left in TRIADIC_MODULE
                for right in PAIRWISE_MODULE
            ]
        elif cross_topology == "matching":
            cross_edges = list(zip(TRIADIC_MODULE, PAIRWISE_MODULE, strict=True))
        else:
            raise ValueError("cross_topology must be 'all_to_all' or 'matching'.")
        graph.add_edges_from(cross_edges)
        nx.draw_networkx_edges(
            graph,
            POSITIONS,
            edge_color="#B8BDC5",
            width=0.65 + 2.4 * float(cross_coupling) / WITHIN_COUPLING,
            alpha=0.50,
            ax=axis,
        )
    triangle = np.asarray([POSITIONS[name] for name in TRIADIC_MODULE])
    axis.add_patch(
        Polygon(triangle, closed=True, facecolor="#D9903D", edgecolor="#B86722", alpha=0.28, linewidth=3.2)
    )
    weights = pairwise_ring_weights(pairwise_asymmetry)
    for (left_index, right_index), weight in weights.items():
        left, right = NAMES[left_index], NAMES[right_index]
        axis.plot(
            [POSITIONS[left][0], POSITIONS[right][0]],
            [POSITIONS[left][1], POSITIONS[right][1]],
            color="#4477A8",
            linewidth=1.4 + 3.8 * weight / max(weights.values()),
            alpha=0.85,
            solid_capstyle="round",
            zorder=2,
        )
    nx.draw_networkx_nodes(
        graph, POSITIONS, nodelist=list(TRIADIC_MODULE), node_color="#F4DDC5",
        edgecolors="#A95E24", linewidths=1.5, node_size=720, ax=axis,
    )
    nx.draw_networkx_nodes(
        graph, POSITIONS, nodelist=list(PAIRWISE_MODULE), node_color="#DCEAF3",
        edgecolors="#355F7A", linewidths=1.5, node_size=720, ax=axis,
    )
    labels = {name: rf"$\theta_{{{index}}}$" for index, name in enumerate(NAMES, start=1)}
    nx.draw_networkx_labels(graph, POSITIONS, labels=labels, font_size=10, font_color="#24313C", ax=axis)
    axis.text(-0.82, -0.91, "triadic hyperedge", ha="center", color="#A95E24", fontsize=9)
    axis.text(0.82, -0.91, "pairwise ring", ha="center", color="#355F7A", fontsize=9)
    axis.set(xlim=(-1.62, 1.62), ylim=(-1.05, 1.02))
    axis.set_aspect("equal")
    axis.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def crop_white(image: np.ndarray, pad: int = 20) -> np.ndarray:
    occupied = np.any(np.asarray(image)[..., :3] < 0.985, axis=2)
    rows, columns = np.where(occupied)
    return image[
        max(0, rows.min() - pad):min(image.shape[0], rows.max() + pad + 1),
        max(0, columns.min() - pad):min(image.shape[1], columns.max() + pad + 1),
    ]


def render_figure(tree, output: Path, *, pairwise_asymmetry: float) -> Path:
    work = output.parent / ".mixed_order_spt_work"
    network_path = render_network(
        work / "network.png",
        pairwise_asymmetry=pairwise_asymmetry,
    )
    tree_path = plot_synergy_hierarchy_tree(
        tree,
        work / "tree.png",
        source_labels={name: rf"$\theta_{{{index}}}$" for index, name in enumerate(NAMES, start=1)},
        decimals=2,
        dpi=600,
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True, gridspec_kw={"width_ratios": [0.9, 1.4]})
    for axis, path in zip(axes, (network_path, tree_path), strict=True):
        axis.imshow(crop_white(plt.imread(path)))
        axis.axis("off")
    axes[0].set_title("Mixed-order coupling", fontsize=13, color="#24313C", pad=10)
    axes[1].set_title("Complete SPT hierarchy", fontsize=13, color="#24313C", pad=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    for path in (network_path, tree_path):
        path.unlink(missing_ok=True)
    work.rmdir()
    return output


def run(
    *,
    sample_count: int,
    seed: int,
    degree: int,
    pairwise_asymmetry: float,
    triadic_scale: float,
    result: Path,
    figure: Path,
) -> dict[str, object]:
    phases, target = paired_data(
        sample_count=sample_count,
        seed=seed,
        pairwise_asymmetry=pairwise_asymmetry,
        triadic_scale=triadic_scale,
    )
    tables: dict[str, dict[tuple[str, ...], float]] = {}
    for mechanism, module, columns in (
        ("triadic", TRIADIC_MODULE, slice(0, 3)),
        ("pairwise", PAIRWISE_MODULE, slice(3, 6)),
    ):
        tables[mechanism] = estimate_ei_table(
            phases,
            target[:, columns],
            module=module,
            degree=degree,
        )
    tree = build_tree(tables)
    atoms = flatten_phi_tree(tree)
    root_split = [list(child.sources) for child in tree.children]
    expected = {frozenset(TRIADIC_MODULE), frozenset(PAIRWISE_MODULE)}
    observed = {frozenset(child.sources) for child in tree.children}
    payload = {
        "experiment_contract": {
            "question": "Can one complete SPT distinguish a triadic hyperedge module from an asymmetric pairwise ring after an exact disconnected root split?",
            "treatment": "within-module interaction order",
            "triadic_module": list(TRIADIC_MODULE),
            "pairwise_module": list(PAIRWISE_MODULE),
            "fixed": {
                "sample_count": sample_count, "seed": seed, "noise_scale": NOISE_SCALE,
                "cross_coupling": CROSS_COUPLING, "transport_degree": degree,
                "source_features": "identical first and second circular harmonics",
                "target": "same-dimensional local three-velocity response for each module",
                "pairwise_asymmetry": float(pairwise_asymmetry),
                "pairwise_edge_weights": {
                    f"{NAMES[left]}-{NAMES[right]}": float(weight)
                    for (left, right), weight in pairwise_ring_weights(pairwise_asymmetry).items()
                },
                "triadic_scale": float(triadic_scale),
                "triadic_coupling": float(TRIADIC_COUPLING * triadic_scale),
                "syn_nonnegative_tolerance_bits": SYN_TOLERANCE_BITS,
            },
            "root_rule": "exact product-factorization identity at K_out=0; no clipping",
            "bias_correction": "none; full-sample TM estimates",
        },
        "root_phi_bits": float(tree.phi_value),
        "root_syn_bits": float(tree.residual),
        "root_split": root_split,
        "root_split_exact": observed == expected,
        "atoms": [
            {"sources": list(atom.sources), "value_bits": float(atom.value), "depth": atom.depth}
            for atom in atoms
        ],
        "minimum_atom_bits": float(min(atom.value for atom in atoms)),
        "below_negative_tolerance_count": int(sum(atom.value < -SYN_TOLERANCE_BITS for atom in atoms)),
        "closure_error_bits": float(sum(atom.value for atom in atoms) - tree.phi_value),
        "ei_bits": {
            mechanism: {"+".join(key): float(value) for key, value in table.items()}
            for mechanism, table in tables.items()
        },
        "figure": str(figure),
    }
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    render_figure(tree, figure, pairwise_asymmetry=pairwise_asymmetry)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--pairwise-asymmetry", type=float, default=PAIRWISE_ASYMMETRY)
    parser.add_argument("--triadic-scale", type=float, default=1.0)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run(
        sample_count=args.sample_count, seed=args.seed, degree=args.degree,
        pairwise_asymmetry=args.pairwise_asymmetry,
        triadic_scale=args.triadic_scale,
        result=args.result, figure=args.figure,
    )
    print(json.dumps({key: summary[key] for key in ("root_phi_bits", "root_syn_bits", "root_split", "root_split_exact", "minimum_atom_bits")}, indent=2))
