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
HierarchyPolicy = Literal["nonnegative_tolerant", "signed"]


@dataclass(frozen=True)
class PhiAtom:
    """A terminal or split-residual contribution in a greedy Phi hierarchy."""

    sources: tuple[str, ...]
    value: float
    kind: str
    depth: int


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


def greedy_phi_atoms(
    subset: Sequence[str],
    ei_table: Mapping[tuple[str, ...], float],
    *,
    policy: HierarchyPolicy = NONNEGATIVE_TOLERANT,
    eps: float = 1.0e-5,
    split_tolerance: float = 1.0e-4,
    depth: int = 0,
    singleton_ei: Mapping[str, float] | None = None,
) -> list[PhiAtom]:
    """Decompose raw Phi by greedy bipartition.

    ``nonnegative_tolerant`` reproduces the Earth/Brain rule: reject a split
    with residual below ``-split_tolerance`` and retain positive atoms only.
    ``signed`` reproduces the platform rule: retain signed residuals and use
    absolute residual magnitude as the split tie-breaker.
    """
    if policy not in (NONNEGATIVE_TOLERANT, SIGNED):
        raise ValueError(f"Unsupported hierarchy policy: {policy!r}")

    ordered = tuple(str(name) for name in subset)
    if singleton_ei is None:
        singleton_ei = {name: float(ei_table[(name,)]) for name in ordered}
    block_phi = subset_phi_raw(ordered, ei_table, singleton_ei)
    if len(ordered) <= 1 or (policy == NONNEGATIVE_TOLERANT and block_phi <= float(eps)):
        return []

    candidates: list[tuple[float, float, tuple[str, ...], tuple[str, ...]]] = []
    for left, right in nontrivial_bipartitions(ordered):
        left_phi = subset_phi_raw(left, ei_table, singleton_ei)
        right_phi = subset_phi_raw(right, ei_table, singleton_ei)
        residual = block_phi - left_phi - right_phi
        if policy == NONNEGATIVE_TOLERANT and residual < -float(split_tolerance):
            continue
        captured = left_phi + right_phi
        candidates.append((captured, residual, left, right))

    if policy == SIGNED:
        if not candidates:
            return [PhiAtom(ordered, block_phi, "terminal", int(depth))]
        captured, residual, left, right = max(candidates, key=lambda item: (item[0], -abs(item[1])))
        if captured <= float(eps):
            return [PhiAtom(ordered, block_phi, "terminal", int(depth))]
    else:
        if not candidates:
            return [PhiAtom(ordered, block_phi, "terminal", int(depth))]
        captured, residual, left, right = candidates[0]
        for candidate in candidates[1:]:
            candidate_captured, candidate_residual, candidate_left, candidate_right = candidate
            if candidate_captured > captured or (
                np.isclose(candidate_captured, captured) and candidate_residual < residual
            ):
                captured, residual, left, right = candidate_captured, candidate_residual, candidate_left, candidate_right
        if captured <= float(eps):
            return [PhiAtom(ordered, block_phi, "terminal", int(depth))]

    atoms: list[PhiAtom] = []
    if (policy == SIGNED and abs(residual) > float(eps)) or (policy == NONNEGATIVE_TOLERANT and residual > float(eps)):
        atoms.append(PhiAtom(ordered, float(residual), "split_residual", int(depth)))
    atoms.extend(
        greedy_phi_atoms(
            left,
            ei_table,
            policy=policy,
            eps=eps,
            split_tolerance=split_tolerance,
            depth=depth + 1,
            singleton_ei=singleton_ei,
        )
    )
    atoms.extend(
        greedy_phi_atoms(
            right,
            ei_table,
            policy=policy,
            eps=eps,
            split_tolerance=split_tolerance,
            depth=depth + 1,
            singleton_ei=singleton_ei,
        )
    )
    return atoms
