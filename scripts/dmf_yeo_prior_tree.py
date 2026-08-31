"""Yeo-first ROI tree using the existing fixed-target conditional-Xi estimator.

Contract: change only the admissible top-level split, from free binary search
to the prescribed seven Yeo blocks. Reuse the representative seed/G covariance,
E/I ROI blocks, raw-residual objective, and exact-search threshold. Network
subtrees use the same spectral candidate search as the unconstrained tree.
This is a constrained description, not evidence for discovering Yeo networks.
"""

from __future__ import annotations

import math
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from scripts.analyze_dmf_schaefer100_xi_hierarchy_tree import (
    ConditionalBlockXiOracle,
    SYN_NONNEGATIVE_TOLERANCE_BITS,
    _blend_with_white,
    _leaf_order,
    _short_roi_label,
)
from scripts.analyze_runge_slp_pc60_xi_hierarchy import (
    ScalableHierarchyNode,
    _audit_nonnegative,
    build_scalable_hierarchy,
    flatten_nodes,
)


def build_yeo_prior_tree(
    conditional: np.ndarray,
    membership: np.ndarray,
    *,
    exact_max_size: int = 8,
) -> tuple[ScalableHierarchyNode, dict]:
    membership = np.asarray(membership)
    if membership.ndim != 1 or not np.issubdtype(membership.dtype, np.integer):
        raise ValueError("Yeo membership must be a one-dimensional integer array")
    if set(membership.tolist()) != set(range(7)):
        raise ValueError("Exactly seven nonempty networks numbered 0..6 are required")
    roi_count = len(membership)
    if np.shape(conditional) != (2 * roi_count, 2 * roi_count):
        raise ValueError("Expected E-then-I conditional covariance for every ROI")
    if exact_max_size < 2:
        raise ValueError("The exact-search threshold must be at least two")
    tolerance = SYN_NONNEGATIVE_TOLERANCE_BITS
    oracle = ConditionalBlockXiOracle(
        conditional, [(i, i + roi_count) for i in range(roi_count)]
    )
    groups = [tuple(np.flatnonzero(membership == i).tolist()) for i in range(7)]
    affinity = np.zeros((roi_count, roi_count))
    pair_zero_count = 0
    for group in groups:
        for position, left in enumerate(group):
            for right in group[position + 1:]:
                value = oracle.xi((left, right))
                numerical_zero = _audit_nonnegative(
                    value, context=f"Yeo within-network pair {left}, {right}",
                    tolerance=tolerance,
                )
                pair_zero_count += int(numerical_zero)
                affinity[left, right] = affinity[right, left] = 0.0 if numerical_zero else value
    audit = {"candidate_count": 0, "tolerance_zero_count": 0}
    children = tuple(
        build_scalable_hierarchy(
            group, oracle, affinity, exact_max_size=exact_max_size,
            tolerance=tolerance, audit=audit, depth=1,
        )
        for group in groups
    )
    total = oracle.xi(range(roi_count))
    between = total - sum(child.xi_bits for child in children)
    root_zero_count = int(_audit_nonnegative(
        between, context="Yeo prior seven-way root", tolerance=tolerance,
    ))
    root = ScalableHierarchyNode(
        tuple(range(roi_count)), total, between, 0, "yeo7-prior", children,
    )
    closure = sum(node.syn_bits for node in flatten_nodes(root) if node.children) - total
    subtree_errors = [
        sum(n.syn_bits for n in flatten_nodes(child) if n.children) - child.xi_bits
        for child in children
    ]
    if max(abs(closure), max(abs(e) for e in subtree_errors)) > tolerance:
        raise RuntimeError("Yeo-prior tree or network-subtree closure failed")
    atoms = [n.syn_bits for n in flatten_nodes(root) if n.children]
    for value in atoms:
        _audit_nonnegative(value, context="Yeo-prior tree atom", tolerance=tolerance)
    return root, {
        "constraint": "fixed Yeo-7 root; data-driven binary search within each network",
        "syn_nonnegative_tolerance_bits": tolerance,
        "pair_tolerance_zero_count": pair_zero_count,
        "split_tolerance_zero_count": audit["tolerance_zero_count"],
        "root_tolerance_zero_count": root_zero_count,
        "tree_atom_tolerance_zero_count": sum(v < 0 for v in atoms),
        "significant_nonnegativity_violation_count": 0,
        "minimum_syn_bits": min(atoms),
        "candidate_split_count": audit["candidate_count"],
        "coalition_evaluation_count": oracle.evaluations,
        "cross_roi_xi_bits": total,
        "between_network_xi_bits": between,
        "within_network_xi_bits": [c.xi_bits for c in children],
        "closure_error_bits": closure,
        "subtree_closure_errors_bits": subtree_errors,
    }


def render_yeo_prior_tree(
    root: ScalableHierarchyNode,
    *,
    axis: plt.Axes,
    labels: Sequence[str],
    membership: np.ndarray,
    network_names: Sequence[str],
    network_colors: Sequence[str],
    network_order: Sequence[int],
    seed: int,
    coupling_g: float,
) -> None:
    if len(root.children) != 7 or sorted(network_order) != list(range(7)):
        raise ValueError("Expected a seven-way Yeo root and a permutation of seven networks")
    children = [root.children[i] for i in network_order]
    leaf_x, spans = {}, {}
    cursor = 0.0
    for network, child in zip(network_order, children):
        order = _leaf_order(child)
        if any(int(membership[i]) != network for i in order):
            raise ValueError("A prior subtree contains a ROI from another network")
        for offset, roi in enumerate(order):
            leaf_x[roi] = cursor + offset
        spans[network] = (cursor - 0.5, cursor + len(order) - 0.5)
        cursor += len(order) + 2.5
    positions = {}

    def position(node):
        if not node.children:
            result = (leaf_x[node.indices[0]], 0.0)
        else:
            points = [position(child) for child in node.children]
            result = (
                sum(p[0] * c.size for p, c in zip(points, node.children)) / node.size,
                math.log2(node.size) / math.log2(root.size),
            )
        positions[node.indices] = result
        return result

    for child in children:
        position(child)
    roots = [positions[child.indices] for child in children]
    axis.plot([roots[0][0], roots[-1][0]], [1, 1], color="#697887", lw=1.1)
    axis.text(
        (roots[0][0] + roots[-1][0]) / 2, 1,
        rf"Yeo-7 prior split  |  Between-network $\Xi={root.syn_bits:.3f}$ bits",
        ha="center", va="center", fontsize=6.5, color="#34414D",
        bbox=dict(facecolor="white", edgecolor="#9EA9B2", boxstyle="round,pad=0.35"),
        zorder=5,
    )
    short_names = {
        "Visual": "Visual", "Somatomotor": "SomMot", "Dorsal attention": "DorsAttn",
        "Salience / ventral attention": "Sal/Vent", "Limbic": "Limbic",
        "Frontoparietal control": "Control", "Default mode": "Default",
    }
    max_syn = max((n.syn_bits for c in children for n in flatten_nodes(c) if n.children), default=0.0)
    for network, child in zip(network_order, children):
        color = network_colors[network]
        lo, hi = spans[network]
        axis.add_patch(Rectangle((lo, -0.035), hi-lo, 0.94, facecolor=color, alpha=0.035, lw=0))
        cx, cy = positions[child.indices]
        axis.plot([cx, cx], [cy, 1], color=color, lw=1.05)
        axis.text(
            (lo+hi)/2, 0.82,
            f"{short_names.get(network_names[network], network_names[network])}/{child.size}\n"
            + rf"$\Xi_{{in}}$ {child.xi_bits:.3f}",
            ha="center", va="center", fontsize=5.5, color=color,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.8), zorder=5,
        )
        nodes = [n for n in flatten_nodes(child) if n.children]
        selected = {n.indices for n in sorted(nodes, key=lambda n: n.syn_bits, reverse=True)[:2]}
        label_points = []
        for node in nodes:
            x, y = positions[node.indices]
            points = [positions[c.indices] for c in node.children]
            axis.plot([points[0][0], points[1][0]], [y, y], color=color, alpha=0.60, lw=0.7)
            for sx, sy in points:
                axis.plot([sx, sx], [sy, y], color=color, alpha=0.60, lw=0.7)
            value = 0.0 if -SYN_NONNEGATIVE_TOLERANCE_BITS <= node.syn_bits < 0 else node.syn_bits
            strength = value/max_syn if max_syn > 0 else 0
            axis.scatter([x], [y], s=8+20*strength, color=_blend_with_white(color, 0.4),
                         edgecolor=color, linewidth=0.7, zorder=3)
            if node.indices in selected and all(abs(x-px)>3 or abs(y-py)>0.065 for px,py in label_points):
                axis.text(x, y, f"{value:.3f}", ha="center", va="center", fontsize=4.6,
                          bbox=dict(facecolor="white", edgecolor=color, boxstyle="round,pad=0.2"), zorder=4)
                label_points.append((x,y))
        for roi in _leaf_order(child):
            x = leaf_x[roi]
            axis.scatter([x], [0], s=12, facecolor="white", edgecolor=color, linewidth=1, zorder=4)
            axis.add_patch(Rectangle((x-0.48, -0.033), 0.96, 0.017, facecolor=color,
                                     edgecolor="white", lw=0.25))
            axis.text(x, -0.045, _short_roi_label(labels[roi]), rotation=90, ha="right", va="top",
                      fontsize=3.7, color="#55616D")
    axis.text(0.5, 1.015, rf"$G={coupling_g:g}$, seed {seed}  |  Cross-ROI $\Xi={root.xi_bits:.2f}$ bits",
              transform=axis.transAxes, ha="center", va="bottom", fontsize=7, color="#34414D")
    axis.text(0, -0.225, "Network labels: within-network Xi; node labels: local Syn (bits). Height: log ROI count.",
              fontsize=5.5, color="#55616D", ha="left")
    axis.set_xlim(-1, cursor-2.5)
    axis.set_ylim(-0.26, 1.10)
    axis.axis("off")
