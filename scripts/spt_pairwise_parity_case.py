"""Exact six-bit parity example: distinct modules, identical pairwise information.

Uniform independent source interventions; one-step complete six-bit future target.
Each module's parity is flipped with probability 0.1 and copied to its three
future coordinates. Enumerate the finite joint distribution, without a learned
estimator. Use the repository's canonical SPT and audit every partition.

Run: .venv/bin/python scripts/spt_pairwise_parity_case.py
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.spt import (  # noqa: E402
    SPTConfig,
    build_spt_from_ei_table,
    flatten_atoms,
    nontrivial_bipartitions,
)

SOURCES = tuple(range(1, 7))
NOISE = Fraction(1, 10)
TOLERANCE_BITS = 1e-10


def joint_rows(groups):
    rows = []
    for source in itertools.product((0, 1), repeat=6):
        parity = [sum(source[i - 1] for i in group) % 2 for group in groups]
        for flips in itertools.product((0, 1), repeat=2):
            probability = Fraction(1, 64)
            target = [0] * 6
            for group, value, flip in zip(groups, parity, flips):
                probability *= NOISE if flip else 1 - NOISE
                for i in group:
                    target[i - 1] = value ^ flip
            rows.append((source, tuple(target), probability))
    if sum(row[2] for row in rows) != 1:
        raise ArithmeticError("Joint probabilities do not sum to one")
    return rows


def mutual_information(rows, source_subset, target_subset=SOURCES):
    joint = defaultdict(Fraction)
    past = defaultdict(Fraction)
    future = defaultdict(Fraction)
    for x, y, probability in rows:
        a = tuple(x[i - 1] for i in source_subset)
        b = tuple(y[i - 1] for i in target_subset)
        joint[a, b] += probability
        past[a] += probability
        future[b] += probability
    # Exact rational probability ratios keep independent marginals exactly zero.
    return math.fsum(
        float(p) * math.log2(float(p / (past[a] * future[b])))
        for (a, b), p in joint.items()
    )


def analyze(groups):
    rows = joint_rows(groups)
    ei = {
        subset: mutual_information(rows, subset)
        for size in range(1, 7)
        for subset in itertools.combinations(SOURCES, size)
    }
    xi = {s: value - math.fsum(ei[(i,)] for i in s) for s, value in ei.items()}
    residuals = [
        xi[s] - xi[left] - xi[right]
        for s in xi if len(s) > 1
        for left, right in nontrivial_bipartitions(s)
    ]
    significant = sum(value < -TOLERANCE_BITS for value in residuals)
    if significant:
        raise ArithmeticError(
            f"Syn violation: minimum={min(residuals)} bits, "
            f"threshold={-TOLERANCE_BITS} bits, count={significant}"
        )
    tree = build_spt_from_ei_table(
        SOURCES, ei, config=SPTConfig(syn_tolerance=TOLERANCE_BITS)
    )
    pair_phi_r = []
    for pair in itertools.combinations(SOURCES, 2):
        own = math.fsum(mutual_information(rows, (i,), (i,)) for i in pair)
        redundancy = min(
            mutual_information(rows, (i,), (j,)) for i in pair for j in pair
        )
        pair_phi_r.append(mutual_information(rows, pair, pair) - own + redundancy)
    root_split = tuple(tuple(child.sources) for child in tree.root.children)
    if {frozenset(s) for s in root_split} != {frozenset(s) for s in groups}:
        raise ArithmeticError(f"SPT did not recover planted modules: {groups}")
    if abs(tree.closure_error) > TOLERANCE_BITS:
        raise ArithmeticError(f"Tree closure error: {tree.closure_error} bits")
    root_scores = sorted(
        (xi[left] + xi[right] for left, right in nontrivial_bipartitions(SOURCES)),
        reverse=True,
    )
    return dict(
        planted_groups=groups, root_split=root_split,
        xi_bits=xi[SOURCES], xi_nats=xi[SOURCES] * math.log(2),
        root_syn_bits=tree.root.syn_value,
        root_score_gap_bits=root_scores[0] - root_scores[1],
        positive_nodes=[dict(sources=a.sources, value_bits=a.value)
                        for a in flatten_atoms(tree.root) if a.value > TOLERANCE_BITS],
        mean_pair_phi_r_bits=math.fsum(pair_phi_r) / len(pair_phi_r),
        maximum_abs_pair_phi_r_bits=max(map(abs, pair_phi_r)),
        maximum_abs_pair_peid_bits=max(abs(xi[p]) for p in xi if len(p) == 2),
        audit=dict(tolerance_bits=TOLERANCE_BITS,
                   minimum_syn_bits=min(residuals),
                   tolerance_negative_count=sum(-TOLERANCE_BITS <= v < 0 for v in residuals),
                   significant_violation_count=significant,
                   partition_count=len(residuals),
                   closure_error_bits=tree.closure_error,
                   clipping=False),
    )


def main():
    # All ten unordered equal-sized module partitions, without choosing a winner.
    groups = [
        (left, tuple(i for i in SOURCES if i not in left))
        for left in itertools.combinations(SOURCES, 3) if 1 in left
    ]
    records = [analyze(g) for g in groups]
    if any(r['maximum_abs_pair_phi_r_bits'] != 0.0 or
           r['maximum_abs_pair_peid_bits'] != 0.0 for r in records):
        raise ArithmeticError("Pair matrices are not exactly zero")
    print(json.dumps(dict(
        protocol=dict(source='Uniform({0,1}^6)', target='Complete future six-bit state',
                      horizon=1, module_noise_probability=float(NOISE),
                      estimator='Exact finite joint probabilities; no TM required for discrete states'),
        case_count=len(records), recovered_root_count=len(records),
        matrix_only_root_identification_bound=1 / len(records), records=records,
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
