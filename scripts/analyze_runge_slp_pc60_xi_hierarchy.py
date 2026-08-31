#!/usr/bin/env python3
"""Approximate and render a scalable Xi hierarchy for 60 Runge SLP PCs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_runge_slp_pc05_shapley import (
    coalition_ei_bits,
    fit_affine_intervention_model,
    standardize,
)
from scripts.phi_hierarchy import (
    ALL_ORDER_CROSS_DENSITY,
    RAW_RESIDUAL,
    split_objective_value,
)


RESULT_ROOT = (
    ROOT
    / "results"
    / "runge_slp_daily_1948_2026_20260628"
    / "mlp_tm_ei_lag04"
    / "results"
    / "runge"
)
DEFAULT_SOURCE = RESULT_ROOT / "multistep_conditioned_ei_tm_exhaustive/source_samples_n4096.npy"
DEFAULT_ROLLOUT = RESULT_ROOT / "multistep_conditioned_ei_tm_forced_edges/rollout_predictions_H060_n4096.npy"
DEFAULT_OUTPUT_DIR = ROOT / "results/runge/slp_pc60_xi_hierarchy"
DEFAULT_FIGURE = ROOT / "fig/earth_slp_pc60_xi_hierarchy.png"

SYN_NONNEGATIVE_TOLERANCE_BITS = 1.0e-8
SPLIT_COLOR = "#267A70"
EDGE_COLOR = "#B2BAC3"
INK = "#24313C"


@dataclass(frozen=True)
class ScalableHierarchyNode:
    indices: tuple[int, ...]
    xi_bits: float
    syn_bits: float
    depth: int
    search_kind: str
    children: tuple["ScalableHierarchyNode", ...] = ()

    @property
    def size(self) -> int:
        return len(self.indices)


class AffineXiOracle:
    """Cached affine-TM coalition values for a fixed 60-dimensional target."""

    def __init__(self, coefficients: np.ndarray, residual_covariance: np.ndarray):
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.residual_covariance = np.asarray(residual_covariance, dtype=np.float64)
        self.singletons = np.asarray(
            [
                coalition_ei_bits(self.coefficients, self.residual_covariance, (index,))
                for index in range(self.coefficients.shape[0])
            ],
            dtype=np.float64,
        )
        self._cache: dict[tuple[int, ...], float] = {
            (index,): 0.0 for index in range(self.coefficients.shape[0])
        }
        self.evaluations = self.coefficients.shape[0]

    def xi(self, indices: Iterable[int]) -> float:
        key = tuple(sorted(int(index) for index in indices))
        if not key:
            return 0.0
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        joint = coalition_ei_bits(self.coefficients, self.residual_covariance, key)
        value = float(joint - self.singletons[list(key)].sum())
        self._cache[key] = value
        self.evaluations += 1
        return value


def _audit_nonnegative(value: float, *, context: str, tolerance: float) -> bool:
    """Return whether a tiny negative was absorbed; fail on a real violation."""
    if value < -float(tolerance):
        raise RuntimeError(
            f"Syn nonnegativity violation in {context}: value={value:.12g} bits, "
            f"tolerance={tolerance:.12g} bits, affected_count=1."
        )
    return value < 0.0


def pairwise_syn_affinity(
    oracle: AffineXiOracle,
    node_count: int,
    *,
    tolerance: float,
) -> tuple[np.ndarray, int]:
    affinity = np.zeros((node_count, node_count), dtype=np.float64)
    tolerance_zero_count = 0
    for left in range(node_count):
        for right in range(left + 1, node_count):
            value = oracle.xi((left, right))
            if _audit_nonnegative(
                value,
                context=f"pair ({left}, {right})",
                tolerance=tolerance,
            ):
                tolerance_zero_count += 1
                display_value = 0.0
            else:
                display_value = value
            affinity[left, right] = affinity[right, left] = display_value
    return affinity, tolerance_zero_count


def _canonical_split(
    left: Iterable[int],
    right: Iterable[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    left_key = tuple(sorted(int(index) for index in left))
    right_key = tuple(sorted(int(index) for index in right))
    return (left_key, right_key) if left_key < right_key else (right_key, left_key)


def _exact_candidates(indices: tuple[int, ...]) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
    first, rest = indices[0], indices[1:]
    candidates: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    full = set(indices)
    for mask in range(1 << len(rest)):
        left = {first}
        left.update(rest[position] for position in range(len(rest)) if mask & (1 << position))
        if len(left) == len(indices):
            continue
        candidates.add(_canonical_split(left, full - left))
    return candidates


def _spectral_orders(indices: tuple[int, ...], affinity: np.ndarray) -> list[tuple[int, ...]]:
    local = affinity[np.ix_(indices, indices)]
    orders: list[tuple[int, ...]] = []
    if len(indices) >= 3 and float(local.max()) > 0.0:
        degree = local.sum(axis=1)
        inverse_root = np.zeros_like(degree)
        positive = degree > 0.0
        inverse_root[positive] = 1.0 / np.sqrt(degree[positive])
        laplacian = np.eye(len(indices)) - inverse_root[:, None] * local * inverse_root[None, :]
        _, vectors = np.linalg.eigh(laplacian)
        for column in range(1, min(5, vectors.shape[1])):
            order = tuple(indices[position] for position in np.argsort(vectors[:, column], kind="stable"))
            orders.append(order)

        maximum = float(local.max())
        distances = 1.0 - local / maximum
        np.fill_diagonal(distances, 0.0)
        condensed = squareform(distances, checks=False)
        hierarchy = linkage(condensed, method="average", optimal_ordering=True)
        orders.append(tuple(indices[position] for position in leaves_list(hierarchy)))
    orders.append(indices)
    return orders


def _scalable_candidates(
    indices: tuple[int, ...],
    affinity: np.ndarray,
) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
    candidates: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    full = set(indices)
    for order in _spectral_orders(indices, affinity):
        for cut in range(1, len(order)):
            candidates.add(_canonical_split(order[:cut], order[cut:]))
    for index in indices:
        candidates.add(_canonical_split((index,), full - {index}))
    return candidates


def build_scalable_hierarchy(
    indices: tuple[int, ...],
    oracle: AffineXiOracle,
    affinity: np.ndarray,
    *,
    exact_max_size: int,
    tolerance: float,
    split_objective: str = RAW_RESIDUAL,
    depth: int = 0,
    audit: dict[str, int] | None = None,
) -> ScalableHierarchyNode:
    audit = {"candidate_count": 0, "tolerance_zero_count": 0} if audit is None else audit
    block_xi = oracle.xi(indices)
    if len(indices) == 1:
        return ScalableHierarchyNode(indices, block_xi, 0.0, depth, "leaf")

    if len(indices) <= int(exact_max_size):
        candidates = _exact_candidates(indices)
        search_kind = "exact"
    else:
        candidates = _scalable_candidates(indices, affinity)
        search_kind = "spectral-candidate"
    audit["candidate_count"] += len(candidates)

    if split_objective not in (RAW_RESIDUAL, ALL_ORDER_CROSS_DENSITY):
        raise ValueError(f"Unsupported split objective: {split_objective!r}")
    best: tuple[float, float, float, tuple[int, ...], tuple[int, ...]] | None = None
    for left, right in sorted(candidates):
        captured = oracle.xi(left) + oracle.xi(right)
        residual = float(block_xi - captured)
        if _audit_nonnegative(
            residual,
            context=f"split {left} | {right}",
            tolerance=tolerance,
        ):
            audit["tolerance_zero_count"] += 1
        objective_value = split_objective_value(
            residual, len(left), len(right), objective=split_objective
        )
        candidate = (objective_value, captured, residual, left, right)
        if best is None:
            best = candidate
        elif split_objective == ALL_ORDER_CROSS_DENSITY:
            if (candidate[0], candidate[2], candidate[3], candidate[4]) < (
                best[0], best[2], best[3], best[4]
            ):
                best = candidate
        elif candidate[1] > best[1] or (
            np.isclose(candidate[1], best[1]) and candidate[2] < best[2]
        ):
            best = candidate
    if best is None:
        raise RuntimeError(f"No candidate split was generated for coalition {indices}.")

    _, _, residual, left, right = best
    children = (
        build_scalable_hierarchy(
            left,
            oracle,
            affinity,
            exact_max_size=exact_max_size,
            tolerance=tolerance,
            split_objective=split_objective,
            depth=depth + 1,
            audit=audit,
        ),
        build_scalable_hierarchy(
            right,
            oracle,
            affinity,
            exact_max_size=exact_max_size,
            tolerance=tolerance,
            split_objective=split_objective,
            depth=depth + 1,
            audit=audit,
        ),
    )
    return ScalableHierarchyNode(indices, block_xi, residual, depth, search_kind, children)


def flatten_nodes(root: ScalableHierarchyNode) -> list[ScalableHierarchyNode]:
    nodes = [root]
    for child in root.children:
        nodes.extend(flatten_nodes(child))
    return nodes


def _blend_with_white(color: str, strength: float) -> str:
    base = to_rgb(color)
    amount = min(1.0, max(0.0, float(strength)))
    return to_hex(tuple(1.0 - amount * (1.0 - channel) for channel in base))


def _leaf_order(root: ScalableHierarchyNode) -> list[int]:
    if not root.children:
        return [root.indices[0]]
    return [index for child in root.children for index in _leaf_order(child)]


def _compact_positions(
    root: ScalableHierarchyNode,
) -> tuple[dict[tuple[int, ...], tuple[float, float]], list[int]]:
    """Place every leaf on the baseline and spread a chain into a triangle."""
    leaf_order = _leaf_order(root)
    leaf_x = {index: float(position) for position, index in enumerate(leaf_order)}
    positions: dict[tuple[int, ...], tuple[float, float]] = {}

    def position(node: ScalableHierarchyNode) -> tuple[float, float]:
        if node.indices in positions:
            return positions[node.indices]
        if not node.children:
            point = (leaf_x[node.indices[0]], 0.0)
        else:
            children = [position(child) for child in node.children]
            x_value = sum(
                point[0] * child.size
                for point, child in zip(children, node.children, strict=True)
            ) / node.size
            point = (x_value, math.log2(node.size) / math.log2(root.size))
        positions[node.indices] = point
        return point

    position(root)
    return positions, leaf_order


def render_compact_tree(
    root: ScalableHierarchyNode,
    output_path: Path,
    *,
    tolerance: float,
    syn_scale_max: float | None = None,
    dpi: int = 600,
) -> Path:
    """Render all 60 leaves and all splits, while labeling only structural nodes."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    all_nodes = flatten_nodes(root)
    internal = [node for node in all_nodes if node.children]
    positions, leaf_order = _compact_positions(root)
    leaf_x = {index: float(position) for position, index in enumerate(leaf_order)}
    max_syn = (
        max(float(node.syn_bits) for node in internal)
        if syn_scale_max is None
        else float(syn_scale_max)
    )
    if max_syn <= 0.0:
        max_syn = 1.0

    figure, axis = plt.subplots(figsize=(15.2, 8.2), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")

    for node in internal:
        parent_x, parent_y = positions[node.indices]
        child_points = [positions[child.indices] for child in node.children]
        axis.plot(
            [child_points[0][0], child_points[1][0]],
            [parent_y, parent_y],
            color=EDGE_COLOR,
            linewidth=0.75,
            solid_capstyle="round",
            zorder=1,
        )
        for child_x, child_y in child_points:
            axis.plot(
                [child_x, child_x],
                [child_y, parent_y],
                color=EDGE_COLOR,
                linewidth=0.75,
                solid_capstyle="round",
                zorder=1,
            )

    ranked = sorted(
        (node for node in internal if node is not root and node.size >= 4),
        key=lambda node: (node.depth <= 2, node.syn_bits),
        reverse=True,
    )
    labeled: set[tuple[int, ...]] = set()
    label_points: list[tuple[float, float]] = []
    for node in ranked:
        x_value, y_value = positions[node.indices]
        if all(
            abs(x_value - old_x) >= 2.4 or abs(y_value - old_y) >= 0.075
            for old_x, old_y in label_points
        ):
            labeled.add(node.indices)
            label_points.append((x_value, y_value))
        if len(labeled) >= 12:
            break

    tolerance_zero_count = 0
    for node in internal:
        x_value, y_value = positions[node.indices]
        raw_syn = float(node.syn_bits)
        if raw_syn < 0.0:
            if raw_syn < -float(tolerance):
                raise RuntimeError(
                    f"Cannot render significant negative Syn {raw_syn:.12g} bits at {node.indices}."
                )
            tolerance_zero_count += 1
            display_syn = 0.0
        else:
            display_syn = raw_syn
        relative = min(1.0, display_syn / max_syn)
        face = _blend_with_white(SPLIT_COLOR, 0.12 + 0.60 * relative)
        edge = _blend_with_white(SPLIT_COLOR, 0.55 + 0.45 * relative)
        if node.indices in labeled:
            axis.text(
                x_value,
                y_value,
                f"n={node.size}\nSyn {display_syn:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color=INK,
                linespacing=1.18,
                bbox={
                    "boxstyle": "round,pad=0.27",
                    "facecolor": face,
                    "edgecolor": edge,
                    "linewidth": 0.8 + 1.7 * relative,
                },
                zorder=4,
            )
        else:
            axis.scatter(
                [x_value],
                [y_value],
                s=7.0 + 23.0 * relative,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.6 + relative,
                zorder=3,
            )

    for index in leaf_order:
        x_value = leaf_x[index]
        axis.scatter(
            [x_value], [0.0], s=9.0, facecolor="#F4F6F8",
            edgecolor="#7B8794", linewidth=0.65, zorder=4, clip_on=False,
        )
        axis.text(
            x_value,
            -0.031,
            str(index),
            ha="center",
            va="top",
            fontsize=4.8,
            color="#55616D",
            rotation=90,
        )

    root_x, _ = positions[root.indices]
    axis.text(
        root_x,
        1.115,
        rf"$\Xi$ = {root.xi_bits:.2f} bits",
        ha="center",
        va="center",
        fontsize=9.3,
        color=INK,
        zorder=5,
    )
    axis.set_xlim(-1.0, root.size)
    axis.set_ylim(-0.12, 1.16)
    axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    if tolerance_zero_count:
        print(f"[render] tolerance-zero Syn nodes: {tolerance_zero_count}")
    return output_path


def _dominant_spine(
    root: ScalableHierarchyNode,
) -> tuple[list[ScalableHierarchyNode], list[ScalableHierarchyNode | None]]:
    spine: list[ScalableHierarchyNode] = []
    side_branches: list[ScalableHierarchyNode | None] = []
    current = root
    while True:
        spine.append(current)
        if not current.children:
            side_branches.append(None)
            break
        dominant = max(current.children, key=lambda child: (child.size, -min(child.indices)))
        side = next(child for child in current.children if child is not dominant)
        side_branches.append(side)
        current = dominant
    return spine, side_branches


def render_backbone_tree(
    root: ScalableHierarchyNode,
    output_path: Path,
    *,
    tolerance: float,
    syn_scale_max: float | None = None,
    dpi: int = 600,
) -> Path:
    """Render an imbalanced hierarchy as a horizontal coalition backbone."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    spine, side_branches = _dominant_spine(root)
    internal = [node for node in flatten_nodes(root) if node.children]
    max_syn = (
        max(float(node.syn_bits) for node in internal)
        if syn_scale_max is None
        else float(syn_scale_max)
    )
    if max_syn <= 0.0:
        max_syn = 1.0

    figure, axis = plt.subplots(figsize=(14.2, 5.0), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    baseline = 0.50

    if len(spine) > 1:
        axis.plot(
            [0.0, float(len(spine) - 1)],
            [baseline, baseline],
            color=EDGE_COLOR,
            linewidth=1.15,
            solid_capstyle="round",
            zorder=1,
        )

    label_positions = set(range(0, max(1, len(spine) - 1), 8))
    label_positions.add(0)
    tolerance_zero_count = 0

    for position, (node, side) in enumerate(zip(spine, side_branches, strict=True)):
        x_value = float(position)
        if node.children:
            raw_syn = float(node.syn_bits)
            if raw_syn < 0.0:
                if raw_syn < -float(tolerance):
                    raise RuntimeError(
                        f"Cannot render significant negative Syn {raw_syn:.12g} bits at {node.indices}."
                    )
                tolerance_zero_count += 1
                display_syn = 0.0
            else:
                display_syn = raw_syn
            relative = min(1.0, display_syn / max_syn)
            face = _blend_with_white(SPLIT_COLOR, 0.12 + 0.60 * relative)
            edge = _blend_with_white(SPLIT_COLOR, 0.55 + 0.45 * relative)
            axis.scatter(
                [x_value],
                [baseline],
                s=18.0 + 34.0 * relative,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.9 + 1.5 * relative,
                zorder=3,
            )
            if position in label_positions:
                label_above = position >= len(spine) - 3 or (position // 8) % 2 == 0
                label_y = 0.605 if label_above else 0.395
                axis.text(
                    x_value,
                    label_y,
                    f"n={node.size}  Syn {display_syn:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color=INK,
                    bbox={
                        "boxstyle": "round,pad=0.27",
                        "facecolor": face,
                        "edgecolor": edge,
                        "linewidth": 0.85 + 1.25 * relative,
                    },
                    zorder=5,
                )

        if side is None:
            continue
        branch_above = position % 2 == 0
        branch_y = 0.76 if branch_above else 0.24
        axis.plot(
            [x_value, x_value],
            [baseline, branch_y],
            color=EDGE_COLOR,
            linewidth=0.8,
            solid_capstyle="round",
            zorder=1,
        )
        if side.size == 1:
            side_text = str(side.indices[0])
            facecolor = "#F4F6F8"
            edgecolor = "#7B8794"
            linewidth = 0.75
        else:
            members = ",".join(str(index) for index in side.indices)
            raw_side_syn = float(side.syn_bits)
            if raw_side_syn < 0.0:
                if raw_side_syn < -float(tolerance):
                    raise RuntimeError(
                        f"Cannot render significant negative Syn {raw_side_syn:.12g} bits at {side.indices}."
                    )
                tolerance_zero_count += 1
                side_syn = 0.0
            else:
                side_syn = raw_side_syn
            relative_side = min(1.0, side_syn / max_syn)
            side_text = (
                f"{{{members}}}\nSyn {side_syn:.2f}"
                if side.size <= 5
                else f"n={side.size}\nSyn {side_syn:.2f}"
            )
            facecolor = _blend_with_white(SPLIT_COLOR, 0.12 + 0.60 * relative_side)
            edgecolor = _blend_with_white(SPLIT_COLOR, 0.55 + 0.45 * relative_side)
            linewidth = 0.9 + 1.5 * relative_side
        axis.text(
            x_value,
            branch_y,
            side_text,
            ha="center",
            va="center",
            fontsize=5.2,
            color=INK,
            linespacing=1.15,
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "linewidth": linewidth,
            },
            zorder=4,
        )

    terminal = spine[-1]
    if not terminal.children:
        axis.text(
            float(len(spine) - 1),
            baseline,
            str(terminal.indices[0]),
            ha="center",
            va="center",
            fontsize=5.2,
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": "#F4F6F8",
                "edgecolor": "#7B8794",
                "linewidth": 0.75,
            },
            zorder=4,
        )

    axis.text(
        0.0,
        0.96,
        rf"$\Xi$ = {root.xi_bits:.2f} bits",
        ha="left",
        va="center",
        fontsize=9.3,
        color=INK,
        zorder=5,
    )
    axis.set_xlim(-1.0, float(len(spine)))
    axis.set_ylim(0.08, 1.02)
    axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    if tolerance_zero_count:
        print(f"[render] tolerance-zero Syn nodes: {tolerance_zero_count}")
    return output_path


def _node_record(node: ScalableHierarchyNode) -> dict[str, object]:
    return {
        "indices": list(node.indices),
        "size": node.size,
        "xi_bits": float(node.xi_bits),
        "syn_bits_raw": float(node.syn_bits),
        "depth": int(node.depth),
        "search_kind": node.search_kind,
        "children": [_node_record(child) for child in node.children],
    }


def run_analysis(
    source_path: Path,
    rollout_path: Path,
    output_dir: Path,
    figure_path: Path,
    *,
    horizon: int,
    exact_max_size: int,
    covariance_ridge: float,
    split_objective: str,
    dpi: int,
) -> dict[str, object]:
    source = np.load(source_path)
    rollout = np.load(rollout_path, mmap_mode="r")
    if source.shape != (4096, 60) or rollout.shape != (4096, 60, 60):
        raise ValueError(f"Unexpected source/rollout shapes: {source.shape}, {rollout.shape}")
    if not 1 <= int(horizon) <= 60:
        raise ValueError("horizon must be between 1 and 60")

    standardized_source = standardize(source, "source samples")
    coefficients, residual_covariance = fit_affine_intervention_model(
        standardized_source,
        np.asarray(rollout[:, int(horizon) - 1, :]),
        float(covariance_ridge),
    )
    oracle = AffineXiOracle(coefficients, residual_covariance)
    affinity, pair_tolerance_zero_count = pairwise_syn_affinity(
        oracle,
        60,
        tolerance=SYN_NONNEGATIVE_TOLERANCE_BITS,
    )
    audit = {"candidate_count": 0, "tolerance_zero_count": 0}
    tree = build_scalable_hierarchy(
        tuple(range(60)),
        oracle,
        affinity,
        exact_max_size=int(exact_max_size),
        tolerance=SYN_NONNEGATIVE_TOLERANCE_BITS,
        split_objective=split_objective,
        audit=audit,
    )
    internal = [node for node in flatten_nodes(tree) if node.children]
    closure_error = float(sum(node.syn_bits for node in internal) - tree.xi_bits)
    if abs(closure_error) > 1.0e-8:
        raise RuntimeError(f"Hierarchy closure failed: error={closure_error:.12g} bits")

    spine, _ = _dominant_spine(tree)
    layout = "triangular-dendrogram"
    render_compact_tree(
        tree,
        figure_path,
        tolerance=SYN_NONNEGATIVE_TOLERANCE_BITS,
        dpi=dpi,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "experiment": "Runge SLP 60-PC scalable Xi hierarchy",
        "status": "approximate candidate-search hierarchy; not exhaustive over 2^59 root bipartitions",
        "horizon": int(horizon),
        "source_shape": list(source.shape),
        "target_shape": [int(rollout.shape[0]), int(rollout.shape[2])],
        "estimator": "affine degree-1 TM / linear-Gaussian log-det equivalent",
        "covariance_ridge": float(covariance_ridge),
        "syn_nonnegative_tolerance_bits": SYN_NONNEGATIVE_TOLERANCE_BITS,
        "pair_tolerance_zero_count": int(pair_tolerance_zero_count),
        "split_tolerance_zero_count": int(audit["tolerance_zero_count"]),
        "significant_nonnegativity_violation_count": 0,
        "candidate_split_count": int(audit["candidate_count"]),
        "coalition_evaluation_count": int(oracle.evaluations),
        "exact_search_max_coalition_size": int(exact_max_size),
        "split_objective": str(split_objective),
        "split_objective_denominator": (
            "(2^|A| - 1)(2^|B| - 1)" if split_objective == ALL_ORDER_CROSS_DENSITY else "1"
        ),
        "reported_node_value": "raw unnormalized Syn residual in bits",
        "xi_bits": float(tree.xi_bits),
        "atom_count": len(internal),
        "closure_error_bits": closure_error,
        "figure": str(figure_path),
        "layout": layout,
        "dominant_spine_split_count": len(spine) - 1,
        "tree": _node_record(tree),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--rollout", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--exact-max-size", type=int, default=10)
    parser.add_argument("--covariance-ridge", type=float, default=1.0e-6)
    parser.add_argument(
        "--split-objective",
        choices=(RAW_RESIDUAL, ALL_ORDER_CROSS_DENSITY),
        default=RAW_RESIDUAL,
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_analysis(
        args.source,
        args.rollout,
        args.output_dir,
        args.figure,
        horizon=args.horizon,
        exact_max_size=args.exact_max_size,
        covariance_ridge=args.covariance_ridge,
        split_objective=args.split_objective,
        dpi=args.dpi,
    )
    print(
        f"[done] Xi={payload['xi_bits']:.6f} bits; "
        f"candidates={payload['candidate_split_count']}; "
        f"coalitions={payload['coalition_evaluation_count']}"
    )


if __name__ == "__main__":
    main()
