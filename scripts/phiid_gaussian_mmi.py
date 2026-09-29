"""Bivariate Gaussian-MMI PhiID redundancy→redundancy and synergy→synergy.

The 16-atom Möbius inversion follows the official HOI AtomsPhiID implementation
(brainets/hoi, commit 4db2fbd701d40d33fe2375f3df5a4a4dc94c6b83). This small
analytic version needs only NumPy and returns the two atoms used for fingerprinting.
Unlike PEID Syn, Gaussian-MMI PhiID atoms may be signed; no clipping is applied.
"""

from __future__ import annotations

import numpy as np


_LOG2 = np.log(2.0)


def pair_atoms_from_standardized(
    first: np.ndarray, second: np.ndarray, *, tau: int = 1
) -> tuple[float, float]:
    """Return (rtr, sts) in bits for two already z-scored equal-length series."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if (first.ndim != 1 or second.shape != first.shape or len(first) <= tau + 3
            or tau < 1 or not np.isfinite(first).all() or not np.isfinite(second).all()):
        raise ValueError("PhiID inputs must be finite, equal-length 1D series")
    four = np.column_stack((first[:-tau], second[:-tau], first[tau:], second[tau:]))
    covariance = four.T @ four / (len(four) - 1)

    logdets: dict[tuple[int, ...], float] = {}

    def logdet(indices: tuple[int, ...]) -> float:
        if indices not in logdets:
            sign, value = np.linalg.slogdet(covariance[np.ix_(indices, indices)])
            if sign <= 0 or not np.isfinite(value):
                raise ValueError("PhiID Gaussian covariance is not positive definite")
            logdets[indices] = float(value)
        return logdets[indices]

    def mi(left: tuple[int, ...], right: tuple[int, ...]) -> float:
        joint = tuple(sorted((*left, *right)))
        return (logdet(left) + logdet(right) - logdet(joint)) / (2 * _LOG2)

    ixta = mi((0,), (2,))
    ixtb = mi((0,), (3,))
    iyta = mi((1,), (2,))
    iytb = mi((1,), (3,))
    ixtab = mi((0,), (2, 3))
    iytab = mi((1,), (2, 3))
    ixytab = mi((0, 1), (2, 3))
    ixyta = mi((0, 1), (2,))
    ixytb = mi((0, 1), (3,))

    rtr = min(ixta, ixtb, iyta, iytb)
    rxyta = min(ixta, iyta)
    rxytb = min(ixtb, iytb)
    rxytab = min(ixtab, iytab)
    rabtx = min(ixta, ixtb)
    rabty = min(iyta, iytb)
    rabtxy = min(ixyta, ixytb)

    # Last row of the official 16-atom product-lattice Möbius inverse.
    sts = (
        rtr - rxyta - rxytb + rxytab - rabtx - rabty + rabtxy
        + ixta + ixtb + iyta + iytb - ixyta - ixytb - ixtab - iytab + ixytab
    )
    return float(rtr), float(sts)


def phiid_pair_features(series: np.ndarray, *, tau: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Compute ordered upper-triangle rtr/sts maps from a [time, 7] PC1 series."""
    series = np.asarray(series, dtype=np.float64)
    if series.ndim != 2 or series.shape[1] != 7 or not np.isfinite(series).all():
        raise ValueError("Expected a finite [time, 7] network series")
    scales = series.std(axis=0, ddof=0)
    if np.any(scales <= 1e-12):
        raise ValueError("A network PC1 series is nearly constant")
    standardized = (series - series.mean(axis=0)) / scales
    red = np.empty(21, dtype=np.float64)
    syn = np.empty(21, dtype=np.float64)
    for edge, (left, right) in enumerate(zip(*np.triu_indices(7, 1))):
        red[edge], syn[edge] = pair_atoms_from_standardized(
            standardized[:, left], standardized[:, right], tau=tau
        )
    if not np.isfinite(red).all() or not np.isfinite(syn).all():
        raise ValueError("PhiID atom features are non-finite")
    return red, syn
