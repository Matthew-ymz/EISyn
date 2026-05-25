"""
theory.py
=========
Mean-field recoverability theory for network revival.

Translated from MATLAB:
  - find_dc_by_wk_theory_intersection.m  → find_critical_delta()
  - is_rcoverable_theory.m               → is_recoverable()

The 1st-order mean-field recurrence (Eq. 5-6 in paper):
    x_s(0) = Δ
    F(x_s(l)) = M2(x_s(l-1))

F(x) = R(x)/ω - κ · M2(R^{-1}(ω·M2(x) + ω·κ·M2(x̄₀)))
R(x) = -M0(x)/M1(x)

Phase classification (matches MATLAB + paper logic):
  INACTIVE  (fps=[], f_mf<=0 everywhere) → Dc = inf   (can't revive)
  ACTIVE    (fps=[], f_mf>0 somewhere)   → Dc = 0     (always active)
  BISTABLE  → analyse F_mon vs M2 intersections:
      1 intersection  → structurally UNRECOVERABLE → Dc = inf
      3 intersections → RECOVERABLE → Dc = crossings[1]

Uses pure numpy (no scipy). n_pts=5000 for x_low accuracy.
"""

import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Bisection (replaces scipy.brentq)
# ---------------------------------------------------------------------------

def _bisect(f, a, b, tol=1e-8, max_iter=100):
    fa, fb = f(a), f(b)
    if not (np.isfinite(fa) and np.isfinite(fb)):
        return None
    if fa * fb > 0:
        return None
    for _ in range(max_iter):
        mid = (a + b) / 2.0
        fm = f(mid)
        if not np.isfinite(fm) or abs(b - a) < tol:
            return mid
        if fa * fm <= 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return (a + b) / 2.0


# ---------------------------------------------------------------------------
# Scalar evaluation helper
# ---------------------------------------------------------------------------

def _s(fn, x):
    return float(fn(np.array([float(x)]))[0])


# ---------------------------------------------------------------------------
# Mean-field phase detection
# ---------------------------------------------------------------------------

def _mf_phase(model: dict, omega: float, kappa: float, n_pts: int = 5000):
    """
    Classify the mean-field phase:
      'inactive' : only x=0 stable (f_mf ≤ 0 for all x>0)
      'bistable' : two stable fixed points x_low>0 and x_high>x_low
      'active'   : only x_high stable (x=0 unstable)

    Returns (phase_str, fps_array).
    fps_array is sorted array of non-zero fixed points found.
    """
    M0 = model["M0"]
    M1 = model["M1"]
    M2 = model["M2"]
    x_lo, x_hi = model.get("x_range", (0.0, 100.0))
    x_hi_scan = min(x_hi, 30.0)

    x_arr = np.linspace(x_lo + 1e-5, x_hi_scan, n_pts)
    f_mf = M0(x_arr) + omega * (kappa + 1) * M1(x_arr) * M2(x_arr)

    # Max of f_mf for x > 0 determines whether non-zero fps exist at all
    f_max = float(np.nanmax(f_mf))

    if f_max <= 0:
        # f_mf ≤ 0 everywhere → x=0 is the only attractor → INACTIVE
        return "inactive", np.array([])

    # Find sign-change intervals → non-zero fixed points
    def _f_mf_scalar(x):
        return (_s(M0, x) + omega * (kappa + 1) * _s(M1, x) * _s(M2, x))

    fps = []
    for i in range(len(x_arr) - 1):
        if np.isfinite(f_mf[i]) and np.isfinite(f_mf[i + 1]) and f_mf[i] * f_mf[i + 1] < 0:
            r = _bisect(_f_mf_scalar, x_arr[i], x_arr[i + 1], tol=1e-9)
            if r is not None:
                fps.append(r)
    fps = np.array(sorted(set(np.round(fps, 8))))

    if len(fps) == 0:
        # f_mf > 0 somewhere but no sign change found → active (x=0 unstable)
        return "active", np.array([])
    elif len(fps) == 1:
        # Only one non-zero fp: either x=0 is stable (bistable) or unstable (active)
        # Check stability of x=0: f_mf'(0) ≷ 0
        # For MM with h≥2: M2'(0)=0, so f_mf'(0) = M0'(0) < 0 → x=0 stable → bistable
        # For SIS: M1(0)=1, M2'(0)=1, f_mf'(0) = -β + ω(κ+1) → depends on params
        # Simple check: f_mf at small x
        f_small = _f_mf_scalar(1e-6)
        if f_small < 0:
            # x=0 stable, one active fp → bistable (hysteresis)
            return "bistable", fps
        else:
            # x=0 unstable, one active fp → active
            return "active", fps
    else:
        # Multiple fps: typically [x_low, x_unstable, x_high] or just [x_low, x_high]
        # x_low and x_high are stable; the middle one (if 3 fps) is unstable
        return "bistable", fps


def find_mean_field_fixed_points(
    model: dict,
    omega: float,
    kappa: float,
    n_pts: int = 5000,
) -> np.ndarray:
    """
    Returns sorted array of non-zero MF fixed points.
    For inactive phase: returns []. For active: may return [x_high].
    """
    _, fps = _mf_phase(model, omega, kappa, n_pts=n_pts)
    return fps


# ---------------------------------------------------------------------------
# Critical Delta Δ_c
# ---------------------------------------------------------------------------

def find_critical_delta(
    model: dict,
    omega: float,
    kappa: float,
    n_pts: int = 5000,
) -> float:
    """
    Find critical reigniting amplitude Δ_c.

    Phase rules:
      INACTIVE  → Dc = inf   (no revival possible even with large Δ)
      ACTIVE    → Dc = 0     (system already active)
      BISTABLE  → analyse F_mon vs M2:
          1 crossing → unrecoverable → inf
          ≥3 crossings → Dc = crossings[1]

    Returns np.inf or 0.0 or a finite positive value.
    """
    M0 = model["M0"]
    M1 = model["M1"]
    M2 = model["M2"]
    Rinv = model.get("Rinv", None)

    # Step 1: classify phase
    phase, fps = _mf_phase(model, omega, kappa, n_pts=n_pts)

    if phase == "inactive":
        return np.inf   # system stuck at x=0, single-node forcing can't propagate
    if phase == "active":
        return 0.0      # system already spontaneously active

    # BISTABLE: fps contains at least [x_low, ...] (stable collapsed state)
    x_low = float(fps[0])

    # Build x grid for F computation — scan up to 3×x_high (or 20)
    x_hi = float(fps[-1]) if len(fps) > 1 else 15.0
    x_hi_scan = min(max(x_hi * 3.0, 10.0), 30.0)
    x_arr = np.linspace(1e-5, x_hi_scan, n_pts)

    def R(x):
        m1 = _s(M1, x)
        return -_s(M0, x) / m1 if abs(m1) > 1e-15 else np.inf

    # Build Rinv numerically if not in model
    if Rinv is not None:
        def _Rinv(y):
            return float(Rinv(np.array([float(y)]))[0])
    else:
        R_vals = np.array([R(xi) for xi in x_arr])
        valid_r = np.isfinite(R_vals)
        _xr = x_arr[valid_r]; _Rv = R_vals[valid_r]
        sidx = np.argsort(_Rv)
        _Rs = _Rv[sidx]; _xs = _xr[sidx]
        def _Rinv(y):
            return float(np.interp(y, _Rs, _xs))

    m2_xlow = _s(M2, x_low)

    # Step 2: build F(x) — 1st-order approximation
    def F(x):
        try:
            rx = R(x)
            if not np.isfinite(rx):
                return np.nan
            inner = omega * _s(M2, x) + omega * kappa * m2_xlow
            if inner < 0:
                return np.nan
            rinv_val = _Rinv(inner)
            return rx / omega - kappa * _s(M2, rinv_val)
        except Exception:
            return np.nan

    F_arr = np.array([F(xi) for xi in x_arr])
    valid = np.isfinite(F_arr)
    if not valid.any():
        return np.inf

    x_v = x_arr[valid]
    F_v = F_arr[valid]

    # Step 3: monotone envelope of F
    F_mono = np.maximum.accumulate(F_v)
    M2_v = np.array([_s(M2, xi) for xi in x_v])

    # Step 4: find crossings of F_mono(x) - M2(x) = 0
    diff = F_mono - M2_v
    crossings = []
    for i in range(len(x_v) - 1):
        if np.isfinite(diff[i]) and np.isfinite(diff[i + 1]) and diff[i] * diff[i + 1] < 0:
            def _fc(x, i=i):
                fm = float(np.interp(x, x_v, F_mono))
                return fm - _s(M2, x)
            r = _bisect(_fc, x_v[i], x_v[i + 1])
            if r is not None:
                crossings.append(r)
    crossings = sorted(set(np.round(crossings, 6)))

    # Step 5: classify
    # MATLAB rule: len(fp)>1 → Dc=fp[1]; else → inf
    if len(crossings) > 1:
        return float(crossings[1])
    else:
        return np.inf   # structurally unrecoverable


# ---------------------------------------------------------------------------
# Recoverability check
# ---------------------------------------------------------------------------

def is_recoverable(model: dict, omega: float, kappa: float, Delta: float) -> bool:
    """Return True if Δ ≥ Δ_c(ω, κ)."""
    dc = find_critical_delta(model, omega, kappa)
    if np.isinf(dc):
        return False
    return bool(Delta >= dc)


# ---------------------------------------------------------------------------
# Critical omega — scan then bisect
# ---------------------------------------------------------------------------

def find_critical_omega(
    model: dict,
    kappa: float,
    Delta: float,
    omega_lo: float = 1e-2,
    omega_hi: float = 1e2,
    n_scan: int = 30,
) -> float:
    """
    Find ω_c: smallest ω where Δ ≥ Δ_c(ω, κ).

    Strategy: coarse scan first to find the transition interval,
    then bisection. Handles non-monotone Dc(ω) correctly.
    """
    # Coarse scan to find the first omega where recoverable
    omega_scan = np.logspace(np.log10(omega_lo), np.log10(omega_hi), n_scan)
    first_recov_idx = None
    for i, w in enumerate(omega_scan):
        if is_recoverable(model, w, kappa, Delta):
            first_recov_idx = i
            break

    if first_recov_idx is None:
        return float(omega_hi)   # never recoverable in range
    if first_recov_idx == 0:
        return float(omega_lo)   # recoverable even at omega_lo

    # Bisect between omega_scan[first_recov_idx-1] (unrecov) and omega_scan[first_recov_idx] (recov)
    w_lo = omega_scan[first_recov_idx - 1]
    w_hi = omega_scan[first_recov_idx]

    def f(w):
        return -1.0 if is_recoverable(model, w, kappa, Delta) else 1.0

    r = _bisect(f, w_lo, w_hi, tol=1e-4)
    return float(r) if r is not None else float(w_hi)


# ---------------------------------------------------------------------------
# Phase diagram (grid scan)
# ---------------------------------------------------------------------------

def phase_diagram_theory(
    model: dict,
    kappa_vec: np.ndarray,
    omega_vec: np.ndarray,
    Delta: float,
) -> np.ndarray:
    """
    Theoretical recoverability on (κ, ω) grid.
    Returns float matrix [len(omega_vec), len(kappa_vec)]: 1=recoverable, 0=not.
    """
    result = np.zeros((len(omega_vec), len(kappa_vec)), dtype=float)
    for j, kappa in enumerate(kappa_vec):
        for i, omega in enumerate(omega_vec):
            result[i, j] = float(is_recoverable(model, omega, kappa, Delta))
    return result
