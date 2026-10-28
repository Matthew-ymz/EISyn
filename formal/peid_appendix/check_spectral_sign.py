#!/usr/bin/env python3
"""Reproduce the S4.2 tie counterexample without EI estimation or simulation.

The affinity matrix, its full eigenbasis, degrees and eigenvalues are checked
using exact rational arithmetic. SciPy supplies the auxiliary linkage ordering.
The corresponding finite candidate-pool difference is checked separately in Lean.
"""
from fractions import Fraction

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def main():
    n = 16
    hadamard = [[(-1) ** ((i & j).bit_count()) for j in range(n)] for i in range(n)]
    eigenvalues = [150] + list(range(-3, -18, -1))
    affinity = [[Fraction(sum(hadamard[i][k] * eigenvalues[k] * hadamard[j][k]
                              for k in range(n)), n)
                 for j in range(n)] for i in range(n)]
    for i in range(n):
        assert affinity[i][i] == 0
        assert sum(affinity[i]) == 150
        for j in range(n):
            assert affinity[i][j] == affinity[j][i]
            if i != j:
                assert affinity[i][j] > 0
            assert sum(hadamard[k][i] * hadamard[k][j] for k in range(n)) == (n if i == j else 0)
        for k in range(n):
            assert sum(affinity[i][j] * hadamard[j][k] for j in range(n)) == eigenvalues[k] * hadamard[i][k]
    # Thus L = I - W/150 has distinct eigenvalues 0, 153/150, ..., 167/150.
    # Its first four nonconstant eigendirections are precisely columns 1--4.
    w = np.array(affinity, dtype=float)
    distances = 1 - w / w.max()
    np.fill_diagonal(distances, 0)
    hierarchy = leaves_list(linkage(squareform(distances), method="average", optimal_ordering=True)).tolist()
    orders = [sorted(range(n), key=lambda i: hadamard[i][k]) for k in range(1, 5)]
    orders += [hierarchy, list(range(n))]
    flipped = [sorted(range(n), key=lambda i: -hadamard[i][1])] + orders[1:]

    def candidates(orderings):
        return {min(sum(2 ** i for i in order[:k]), (2 ** n - 1) - sum(2 ** i for i in order[:k]))
                for order in orderings for k in range(1, n)}

    before, after = candidates(orders), candidates(flipped)
    assert 5 not in before and 5 in after
    print("Exact connected affinity/eigenbasis checks passed; stable sorting changes the full candidate pool.")
    print("Candidate counts:", len(before), "->", len(after))
    print("Unordered split {0,2}|complement:", 5 in before, "->", 5 in after)
    print("Average-linkage ordering:", hierarchy)


if __name__ == "__main__":
    main()
