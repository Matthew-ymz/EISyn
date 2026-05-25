"""
network.py
==========
Network builders for complex network revival experiments.
Pure numpy (no scipy required).

Translated from MATLAB BuildSF.m, BuildRR.m, BuildNetwork.m, GCC.m
(Sanhedrai et al., Nature Physics 2022).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_er(N: int, k_avg: float, rng=None) -> np.ndarray:
    """Erdős-Rényi undirected graph, average degree k_avg."""
    if rng is None:
        rng = np.random.default_rng()
    p = k_avg / (N - 1)
    p = min(p, 1.0)
    upper = (rng.random((N, N)) < p)
    upper = np.triu(upper, k=1)
    A = (upper + upper.T).astype(float)
    np.fill_diagonal(A, 0)
    return A


def build_sf(N: int, gamma: float = 2.5, k0: float = 2.0, rng=None) -> np.ndarray:
    """
    Scale-free network via configuration model.
    P(k) ~ k^{-gamma}, k >= k0.
    Translated from BuildSF.m.
    """
    if rng is None:
        rng = np.random.default_rng()

    u = rng.random(N)
    ki = np.round(k0 * u ** (1.0 / (1 - gamma))).astype(int)
    ki = np.maximum(ki, 1)

    stubs = np.repeat(np.arange(N), ki)
    stubs = rng.permutation(stubs)
    if len(stubs) % 2 == 1:
        stubs = stubs[:-1]

    src = stubs[0::2]
    dst = stubs[1::2]

    A = np.zeros((N, N), dtype=float)
    for s, d in zip(src, dst):
        if s != d:
            A[s, d] = 1.0
            A[d, s] = 1.0
    return A


def build_rr(N: int, k: int, rng=None) -> np.ndarray:
    """Random-regular graph, degree k for every node."""
    if rng is None:
        rng = np.random.default_rng()

    stubs = np.tile(np.arange(N), k)
    stubs = rng.permutation(stubs)
    if len(stubs) % 2 == 1:
        stubs = stubs[:-1]

    src = stubs[0::2]
    dst = stubs[1::2]

    A = np.zeros((N, N), dtype=float)
    for s, d in zip(src, dst):
        if s != d:
            A[s, d] = 1.0
            A[d, s] = 1.0
    deg = A.sum(axis=1)
    mask = deg > 0
    A = A[mask][:, mask]
    return A


def build_two_community(
    N1: int, N2: int,
    k_intra: float, k_inter: float,
    rng=None,
):
    """
    Two-community modular network (for brain experiments, Fig 5).
    Returns (A, comm1_mask).
    """
    if rng is None:
        rng = np.random.default_rng()
    N = N1 + N2

    def er_block(nr, nc, k_avg, symmetric=False):
        p = k_avg / max(nc - 1, 1)
        p = min(p, 1.0)
        mat = (rng.random((nr, nc)) < p).astype(float)
        if symmetric:
            mat = np.triu(mat, 1)
            mat = mat + mat.T
        return mat

    A11 = er_block(N1, N1, k_intra, symmetric=True)
    A22 = er_block(N2, N2, k_intra, symmetric=True)
    A12 = er_block(N1, N2, k_inter)
    A21 = A12.T

    A = np.block([[A11, A12], [A21, A22]])
    np.fill_diagonal(A, 0)
    comm1 = np.zeros(N, dtype=bool)
    comm1[:N1] = True
    return A, comm1


# ---------------------------------------------------------------------------
# GCC extraction (BFS, pure numpy)
# ---------------------------------------------------------------------------

def largest_connected_component(A: np.ndarray):
    """
    Return (A_gcc, node_indices) for the largest connected component.
    BFS implementation. Translated from MATLAB GCC.m / onlyGCC.
    """
    N = A.shape[0]
    visited = np.zeros(N, dtype=bool)
    components = []

    for start in range(N):
        if visited[start]:
            continue
        # BFS
        comp = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            comp.append(node)
            neighbors = np.where(A[node] > 0)[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        components.append(comp)

    # Largest component
    largest = max(components, key=len)
    idx = np.array(sorted(largest))
    A_gcc = A[np.ix_(idx, idx)]
    return A_gcc, idx


# ---------------------------------------------------------------------------
# Network properties
# ---------------------------------------------------------------------------

def network_params(A: np.ndarray) -> dict:
    """
    κ = <k²>/<k> - 1   (excess degree / heterogeneity parameter)
    """
    deg = A.sum(axis=1)
    k1 = deg.mean()
    k2 = (deg ** 2).mean()
    kappa = k2 / k1 - 1 if k1 > 0 else 0.0
    return dict(k_avg=float(k1), k2_avg=float(k2),
                kappa=float(kappa), N=A.shape[0])


# ---------------------------------------------------------------------------
# Unified builder
# ---------------------------------------------------------------------------

def build_network(N: int, kind: str, params, rng=None, gcc=True):
    """
    kind: 'ER', 'SF', 'RR'
    params: scalar k (ER/RR) or (gamma, k0) tuple (SF)
    Returns (A, meta_dict)
    """
    if rng is None:
        rng = np.random.default_rng()

    if kind == "ER":
        A = build_er(N, float(params), rng=rng)
    elif kind == "SF":
        gamma, k0 = params
        A = build_sf(N, gamma=float(gamma), k0=float(k0), rng=rng)
    elif kind == "RR":
        A = build_rr(N, int(params), rng=rng)
    else:
        raise ValueError(f"Unknown network kind '{kind}'")

    if gcc:
        A, _ = largest_connected_component(A)

    meta = network_params(A)
    return A, meta
