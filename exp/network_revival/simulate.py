"""
simulate.py
===========
ODE solver and reigniting simulation for network revival experiments.

Translated from MATLAB SolveOdes.m (Sanhedrai et al., Nature Physics 2022).

Uses RK4 (pure numpy) so scipy is NOT required.
"""

import numpy as np
from typing import Optional

# scipy.sparse optional — fall back gracefully
try:
    import scipy.sparse as sp
    _HAS_SPARSE = True
except ImportError:
    sp = None
    _HAS_SPARSE = False


# ---------------------------------------------------------------------------
# RK4 integrator (scipy-free)
# ---------------------------------------------------------------------------

def _rk4_step(f, t, x, dt):
    k1 = f(t, x)
    k2 = f(t + dt / 2, x + dt / 2 * k1)
    k3 = f(t + dt / 2, x + dt / 2 * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _integrate(f, x0, t_start, t_end, dt=0.05, tol_ss=1e-3,
               return_traj=False):
    """
    Integrate dx/dt = f(t, x) from t_start to t_end using adaptive RK4.
    Stops early when converged (|dx/dt| < tol_ss).
    """
    x = x0.copy()
    t = t_start
    t_list = [t]
    x_list = [x.copy()]

    while t < t_end:
        dt_use = min(dt, t_end - t)
        x = _rk4_step(f, t, x, dt_use)
        t += dt_use
        if return_traj:
            t_list.append(t)
            x_list.append(x.copy())
        # convergence check
        if np.max(np.abs(f(t, x))) < tol_ss:
            break

    if return_traj:
        return x, np.array(t_list), np.array(x_list)
    return x, None, None


# ---------------------------------------------------------------------------
# Core ODE solver
# ---------------------------------------------------------------------------

def solve_odes(
    x0: np.ndarray,
    A,                           # dense or sparse adjacency
    model: dict,
    *,
    mode: str = "IC",
    fixed_mask: Optional[np.ndarray] = None,
    free_init: float = 1e-3,
    release: bool = True,
    T_force: float = 50.0,
    T_free: float = 50.0,
    tol_ss: float = 1e-3,
    dt: float = 0.05,
    return_traj: bool = False,
) -> dict:
    """
    Integrate the network ODE to steady state.

    dx_i/dt = M0(x_i) + M1(x_i) * sum_j [A_ij * M2(x_j)]

    Parameters
    ----------
    x0         : Initial state (N,). In BC mode forced nodes are overridden.
    A          : Weighted adjacency (dense ndarray or sparse).
    model      : Dict with M0, M1, M2 callables.
    mode       : 'IC' free evolution; 'BC' force fixed_mask nodes at Delta.
    fixed_mask : Boolean mask of forced nodes (BC mode).
    free_init  : Initial value of free nodes (BC mode).
    release    : If True, release forcing after T_force.
    T_force    : Duration of forcing phase.
    T_free     : Duration after release.
    tol_ss     : Convergence tolerance on |dx/dt|.
    dt         : RK4 time step.
    return_traj: Include full trajectory in output.
    """
    M0 = model["M0"]
    M1 = model["M1"]
    M2 = model["M2"]

    # Convert to dense (sparse not required)
    use_sparse = _HAS_SPARSE and sp is not None and sp.issparse(A)
    if use_sparse:
        A_op = A
        A_sum = np.array(A.sum(axis=1)).flatten()
    else:
        A_op = np.asarray(A, dtype=float)
        A_sum = A_op.sum(axis=1)

    N = A_op.shape[0]
    total_w = A_sum.sum()
    factor_xeff = A_sum / total_w if total_w > 0 else np.ones(N) / N

    x0 = np.asarray(x0, dtype=float).copy()

    if mode == "BC":
        if fixed_mask is None:
            fixed_mask = np.zeros(N, dtype=bool)
        free_mask = ~fixed_mask
        x0[free_mask] = free_init
    else:
        free_mask = np.ones(N, dtype=bool)
        fixed_mask = np.zeros(N, dtype=bool)

    def _matvec(x):
        if use_sparse:
            return A_op.dot(M2(x))
        return A_op @ M2(x)

    def rhs_bc(t, x):
        dx = M0(x) + M1(x) * _matvec(x)
        dx[fixed_mask] = 0.0
        return dx

    def rhs_free(t, x):
        return M0(x) + M1(x) * _matvec(x)

    t_all = []
    x_all = []

    # Phase 1
    rhs1 = rhs_bc if mode == "BC" else rhs_free
    T1 = T_force if mode == "BC" else T_force + T_free

    x_cur, t_tr, x_tr = _integrate(rhs1, x0, 0.0, T1, dt=dt,
                                    tol_ss=tol_ss, return_traj=return_traj)
    if return_traj and t_tr is not None:
        t_all.append(t_tr)
        x_all.append(x_tr)

    # Phase 2: release
    if mode == "BC" and release:
        x_cur, t_tr, x_tr = _integrate(rhs_free, x_cur, T_force, T_force + T_free,
                                        dt=dt, tol_ss=tol_ss, return_traj=return_traj)
        if return_traj and t_tr is not None:
            t_all.append(t_tr)
            x_all.append(x_tr)

    result = dict(
        x_ss=x_cur,
        x_mean=float(x_cur.mean()),
        x_eff=float(factor_xeff @ x_cur),
    )
    if return_traj and t_all:
        result["t"] = np.concatenate(t_all)
        result["x_traj"] = np.concatenate(x_all, axis=0)
    return result


# ---------------------------------------------------------------------------
# Reigniting (Step II)
# ---------------------------------------------------------------------------

def reignite(
    A,
    model: dict,
    source_nodes,
    Delta: float,
    n_trials: int = 10,
    free_init: float = 0.0,
    x_th: Optional[float] = None,
    T_force: float = 50.0,
    T_free: float = 50.0,
    rng=None,
) -> dict:
    """
    Reignite a collapsed network by forcing source_nodes to Delta.
    Returns success fraction η and per-trial steady states.
    """
    if rng is None:
        rng = np.random.default_rng()
    if x_th is None:
        x_th = model.get("x_th", 0.1)

    N = A.shape[0]
    if isinstance(source_nodes, (int, np.integer)):
        source_nodes = [int(source_nodes)]
    source_nodes = list(source_nodes)

    fixed_mask = np.zeros(N, dtype=bool)
    fixed_mask[source_nodes] = True

    x0 = np.full(N, free_init, dtype=float)
    x0[source_nodes] = Delta

    success_list = []
    x_ss_all = []

    for _ in range(n_trials):
        res = solve_odes(x0, A, model, mode="BC",
                         fixed_mask=fixed_mask, free_init=free_init,
                         release=True, T_force=T_force, T_free=T_free)
        x_ss = res["x_ss"]
        x_ss_all.append(x_ss.copy())
        free_mean = x_ss[~fixed_mask].mean() if (~fixed_mask).any() else x_ss.mean()
        success_list.append(bool(free_mean > x_th))

    eta = float(np.mean(success_list))
    return dict(eta=eta, x_ss_all=x_ss_all, success=success_list,
                source_nodes=source_nodes, Delta=Delta)


# ---------------------------------------------------------------------------
# Weight-collapse scan (Fig 3b: hysteresis)
# ---------------------------------------------------------------------------

def weight_collapse_scan(
    A_binary,
    model: dict,
    w_base: float = 1.0,
    n_steps: int = 30,
    free_init_active: float = 5.0,
    rng=None,
) -> dict:
    """
    Gradually reduce link weights (q from 0→1) tracking mean activity.
    Returns q_vec and x_forward (collapse direction).
    """
    if rng is None:
        rng = np.random.default_rng()

    N = A_binary.shape[0]
    use_sparse = _HAS_SPARSE and sp is not None and sp.issparse(A_binary)

    q_vec = np.linspace(0, 1, n_steps + 1)
    x_fwd = np.zeros(len(q_vec))

    x0 = rng.random(N) * free_init_active

    for i, q in enumerate(q_vec):
        w = (1 - q) * w_base
        A_w = w * A_binary
        res = solve_odes(x0, A_w, model, mode="IC",
                         T_force=60.0, T_free=0.0, tol_ss=1e-3)
        x0 = res["x_ss"]
        x_fwd[i] = res["x_mean"]

    return dict(q=q_vec, x_forward=x_fwd)
