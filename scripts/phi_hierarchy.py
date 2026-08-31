"""Shared Phi-from-EI-table and greedy hierarchy utilities.

Experiments retain responsibility for fitting their proxy transition, selecting
an intervention distribution, and estimating the EI table.  This module only
implements the common deterministic step from that table to raw Phi and its
greedy hierarchical atoms.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np


NONNEGATIVE_TOLERANT = "nonnegative_tolerant"
SIGNED = "signed"
RAW_RESIDUAL = "raw_residual"
ALL_ORDER_CROSS_DENSITY = "all_order_cross_density"
HierarchyPolicy = Literal["nonnegative_tolerant", "signed"]
SplitObjective = Literal["raw_residual", "all_order_cross_density"]


@dataclass(frozen=True)
class PhiAtom:
    """A terminal or split-residual contribution in a greedy Phi hierarchy."""

    sources: tuple[str, ...]
    value: float
    kind: str
    depth: int


@dataclass(frozen=True)
class PhiTreeNode:
    """One coalition in the explicit greedy Phi hierarchy."""

    sources: tuple[str, ...]
    phi_value: float
    residual: float
    action: str
    atom_kind: str | None
    depth: int
    children: tuple["PhiTreeNode", ...] = ()

    @property
    def order(self) -> int:
        return len(self.sources)


def all_nonempty_subsets(names: Sequence[str]) -> list[tuple[str, ...]]:
    """Return non-empty subsets in deterministic increasing-cardinality order."""
    ordered = tuple(str(name) for name in names)
    return [combo for size in range(1, len(ordered) + 1) for combo in itertools.combinations(ordered, size)]


def subset_phi_raw(
    subset: Sequence[str],
    ei_table: Mapping[tuple[str, ...], float],
    singleton_ei: Mapping[str, float] | None = None,
) -> float:
    """Compute raw Phi(S) = EI(S) - sum_i EI({i})."""
    ordered = tuple(str(name) for name in subset)
    if singleton_ei is None:
        singleton_ei = {name: float(ei_table[(name,)]) for name in ordered}
    return float(ei_table[ordered] - sum(float(singleton_ei[name]) for name in ordered))


def nontrivial_bipartitions(subset: Sequence[str]) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Enumerate each unordered non-trivial bipartition exactly once."""
    ordered = tuple(str(name) for name in subset)
    if len(ordered) <= 1:
        return []
    first, rest = ordered[0], ordered[1:]
    full = set(ordered)
    splits: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for mask in range(1 << len(rest)):
        left = {first}
        for index, name in enumerate(rest):
            if mask & (1 << index):
                left.add(name)
        if len(left) == len(ordered):
            continue
        right = full - left
        splits.append((tuple(name for name in ordered if name in left), tuple(name for name in ordered if name in right)))
    return splits


def cross_coalition_count(left_size: int, right_size: int) -> int:
    """Count nonempty coalitions that draw at least one node from each side."""
    left = int(left_size)
    right = int(right_size)
    if left <= 0 or right <= 0:
        raise ValueError("Both split sides must be nonempty.")
    return ((1 << left) - 1) * ((1 << right) - 1)


def split_objective_value(
    residual: float,
    left_size: int,
    right_size: int,
    *,
    objective: SplitObjective,
) -> float:
    """Return the search-only split score; raw residual remains the reported atom."""
    if objective == RAW_RESIDUAL:
        return float(residual)
    if objective == ALL_ORDER_CROSS_DENSITY:
        return float(residual) / float(cross_coalition_count(left_size, right_size))
    raise ValueError(f"Unsupported split objective: {objective!r}")


def greedy_phi_tree(
    subset: Sequence[str],
    ei_table: Mapping[tuple[str, ...], float],
    *,
    policy: HierarchyPolicy = NONNEGATIVE_TOLERANT,
    eps: float = 1.0e-5,
    split_tolerance: float = 1.0e-4,
    depth: int = 0,
    singleton_ei: Mapping[str, float] | None = None,
    split_objective: SplitObjective = RAW_RESIDUAL,
) -> PhiTreeNode:
    """Return the full greedy bipartition tree, including zero-atom branches.

    ``nonnegative_tolerant`` reproduces the Earth/Brain rule: reject a split
    with residual below ``-split_tolerance``. ``signed`` reproduces the
    platform rule and uses absolute residual magnitude as the split tie-breaker.

    ``phi_value`` is the complete synergy budget for a coalition. ``residual``
    is the local hierarchy atom left at the selected split. The raw residual is
    retained even when it falls inside the declared numerical tolerance; no
    clipping or projection is applied.
    """
    if policy not in (NONNEGATIVE_TOLERANT, SIGNED):
        raise ValueError(f"Unsupported hierarchy policy: {policy!r}")
    if split_objective not in (RAW_RESIDUAL, ALL_ORDER_CROSS_DENSITY):
        raise ValueError(f"Unsupported split objective: {split_objective!r}")
    if policy == SIGNED and split_objective != RAW_RESIDUAL:
        raise ValueError("All-order residual normalization is defined only for the nonnegative policy.")

    ordered = tuple(str(name) for name in subset)
    if singleton_ei is None:
        singleton_ei = {name: float(ei_table[(name,)]) for name in ordered}
    block_phi = subset_phi_raw(ordered, ei_table, singleton_ei)
    if len(ordered) <= 1 or (policy == NONNEGATIVE_TOLERANT and block_phi <= float(eps)):
        return PhiTreeNode(ordered, block_phi, 0.0, "leaf", None, int(depth))

    candidates: list[tuple[float, float, float, tuple[str, ...], tuple[str, ...]]] = []
    for left, right in nontrivial_bipartitions(ordered):
        left_phi = subset_phi_raw(left, ei_table, singleton_ei)
        right_phi = subset_phi_raw(right, ei_table, singleton_ei)
        residual = block_phi - left_phi - right_phi
        if policy == NONNEGATIVE_TOLERANT and residual < -float(split_tolerance):
            continue
        captured = left_phi + right_phi
        objective_value = split_objective_value(
            residual, len(left), len(right), objective=split_objective
        )
        candidates.append((objective_value, captured, residual, left, right))

    if policy == SIGNED:
        if not candidates:
            return PhiTreeNode(ordered, block_phi, block_phi, "terminal", "terminal", int(depth))
        _, captured, residual, left, right = max(
            candidates, key=lambda item: (item[1], -abs(item[2]))
        )
        if captured <= float(eps):
            return PhiTreeNode(ordered, block_phi, block_phi, "terminal", "terminal", int(depth))
    else:
        if not candidates:
            return PhiTreeNode(ordered, block_phi, block_phi, "terminal", "terminal", int(depth))
        if split_objective == ALL_ORDER_CROSS_DENSITY:
            _, captured, residual, left, right = min(
                candidates,
                key=lambda item: (item[0], item[2], item[3], item[4]),
            )
        else:
            _, captured, residual, left, right = candidates[0]
            for candidate in candidates[1:]:
                _, candidate_captured, candidate_residual, candidate_left, candidate_right = candidate
                if candidate_captured > captured or (
                    np.isclose(candidate_captured, captured) and candidate_residual < residual
                ):
                    captured, residual, left, right = candidate_captured, candidate_residual, candidate_left, candidate_right
        if captured <= float(eps):
            return PhiTreeNode(ordered, block_phi, block_phi, "terminal", "terminal", int(depth))

    atom_kind = None
    if (policy == SIGNED and abs(residual) > float(eps)) or (
        policy == NONNEGATIVE_TOLERANT and residual > float(eps)
    ):
        atom_kind = "split_residual"
    children = (
        greedy_phi_tree(
            left,
            ei_table,
            policy=policy,
            eps=eps,
            split_tolerance=split_tolerance,
            depth=depth + 1,
            singleton_ei=singleton_ei,
            split_objective=split_objective,
        ),
        greedy_phi_tree(
            right,
            ei_table,
            policy=policy,
            eps=eps,
            split_tolerance=split_tolerance,
            depth=depth + 1,
            singleton_ei=singleton_ei,
            split_objective=split_objective,
        ),
    )
    return PhiTreeNode(
        ordered,
        block_phi,
        float(residual),
        "split",
        atom_kind,
        int(depth),
        children,
    )


def flatten_phi_tree(tree: PhiTreeNode) -> list[PhiAtom]:
    """Return hierarchy atoms in the historical root-left-right order."""
    atoms: list[PhiAtom] = []
    if tree.atom_kind is not None:
        atoms.append(PhiAtom(tree.sources, float(tree.residual), tree.atom_kind, int(tree.depth)))
    for child in tree.children:
        atoms.extend(flatten_phi_tree(child))
    return atoms


def greedy_phi_atoms(
    subset: Sequence[str],
    ei_table: Mapping[tuple[str, ...], float],
    *,
    policy: HierarchyPolicy = NONNEGATIVE_TOLERANT,
    eps: float = 1.0e-5,
    split_tolerance: float = 1.0e-4,
    depth: int = 0,
    singleton_ei: Mapping[str, float] | None = None,
    split_objective: SplitObjective = RAW_RESIDUAL,
) -> list[PhiAtom]:
    """Decompose raw Phi into the historical flat atom representation."""
    return flatten_phi_tree(
        greedy_phi_tree(
            subset,
            ei_table,
            policy=policy,
            eps=eps,
            split_tolerance=split_tolerance,
            depth=depth,
            singleton_ei=singleton_ei,
            split_objective=split_objective,
        )
    )
