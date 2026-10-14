"""Experimental continuous-G conditional affine transport model.

Mean and residual triangular regressions use local predictors plus G-scaled
cross-ROI predictors with fixed positive penalties on their unscaled parameters.
The residual modified Cholesky factor gives L=(I-T)^(-1) diag(sqrt(v)), so
L=L_local+G*L_cross with bounded L_cross near zero. The penalty is on T_cross,
not directly on L_cross: this is a practical member of the proposed family,
not its joint maximum-likelihood optimizer. No Syn subtraction or projection.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve, solve_triangular

from exp.TM.transport_map_density import AffineTransportMapDensityEstimator
from scripts.dmf_response_benchmark import dmf_source_components

VERSION = 'dmf-continuous-scaled-ridge-modified-cholesky-v1'


@dataclass
class Moments:
    xx: np.ndarray
    xy: np.ndarray
    yy: np.ndarray
    prior: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    count: int


def sample_moments(x, y, halfwidth=.2, ridge=1e-6):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (x.ndim != 2 or y.shape != x.shape or x.shape[1] % 2
            or len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Finite paired E/I matrices with >=3 observations required')
    if halfwidth <= 0 or ridge <= 0 or not np.isfinite([halfwidth, ridge]).all():
        raise ValueError('Finite positive halfwidth and ridge required')
    xm, ym = x.mean(0), y.mean(0)
    xs, ys = x.std(0, ddof=1), y.std(0, ddof=1)
    if np.any(xs <= 0) or np.any(ys <= 0):
        raise ValueError('Every coordinate must have positive variance')
    xz, yz = (x-xm)/xs, (y-ym)/ys
    scale = len(x)-1
    return Moments(xz.T@xz/scale, xz.T@yz/scale, yz.T@yz/scale,
        np.diag(halfwidth**2/(3*xs**2)+ridge), xm, xs, ym, ys, len(x))


def _scaled_regression(xx, xy, local, cross, coupling, penalty):
    """Quadratic loss + penalty ||unscaled cross parameters||^2.

    Solve in G-scaled coordinates, including G=0. Local coefficients are
    unpenalized. Returns coefficients in the original predictor coordinates.
    """
    local, cross = np.asarray(local, int), np.asarray(cross, int)
    xy = np.asarray(xy, float)
    if xy.ndim == 1:
        xy = xy[:, None]
    coefficient = np.zeros_like(xy)
    if local.size:
        ll = xx[np.ix_(local, local)]
        local_rhs = solve(ll, xy[local], assume_a='pos')
    if cross.size:
        cc, rhs = xx[np.ix_(cross, cross)].copy(), xy[cross].copy()
        if local.size:
            lc = xx[np.ix_(local, cross)]
            ll_lc = solve(ll, lc, assume_a='pos')
            cc -= lc.T @ ll_lc
            rhs -= lc.T @ local_rhs
        scaled = coupling * coupling * .5*(cc+cc.T) + penalty*np.eye(len(cross))
        parameter = solve(scaled, coupling*rhs, assume_a='pos')
        coefficient[cross] = coupling*parameter
    if local.size:
        rhs = xy[local]
        if cross.size:
            rhs = rhs - xx[np.ix_(local, cross)] @ coefficient[cross]
        coefficient[local] = solve(ll, rhs, assume_a='pos')
    return coefficient


@dataclass
class ContinuousAffine:
    covariance: np.ndarray
    coefficient: np.ndarray
    noise: np.ndarray
    noise_factor: np.ndarray
    moments: Moments
    coupling: float
    mean_penalty: float
    noise_penalty: float
    diagnostics: dict

    def scores(self, x, y):
        """Conditional predictive NLL in original target units and standardized MSE."""
        m = self.moments
        if not np.isfinite(m.y_mean).all() or not np.isfinite(m.y_scale).all():
            raise ValueError('Native target mean/scale unavailable in recovered information-only moments')
        error = (np.asarray(y)-m.y_mean)/m.y_scale - (np.asarray(x)-m.x_mean)/m.x_scale @ self.coefficient
        whitened = solve_triangular(np.linalg.cholesky(self.noise), error.T, lower=True)
        logdet = np.linalg.slogdet(self.noise)[1] + 2*np.log(m.y_scale).sum()
        nll = .5*(len(m.y_scale)*np.log(2*np.pi)+logdet+np.square(whitened).sum(0))
        return dict(nll_per_transition=float(nll.mean()), standardized_mse=float(np.square(error).mean()))


def fit_continuous_affine(moments, connectivity, coupling, mean_penalty, noise_penalty, ridge=1e-6):
    """Same fitting formula and SC reachability mask at every G, including zero."""
    m = moments
    if (not np.isfinite([coupling, mean_penalty, noise_penalty, ridge]).all()
            or coupling < 0 or min(mean_penalty, noise_penalty, ridge) <= 0):
        raise ValueError('Nonnegative G and strictly positive finite penalties/ridge required')
    d = len(m.xx); n = d//2
    if np.shape(connectivity) != (n, n):
        raise ValueError('SC dimension differs from E/I data')
    # Fixed reachability partition, independent of G. All indirect paths survive.
    components = dmf_source_components(connectivity, 1.)
    coordinate_component = np.empty(d, int)
    for k, block in enumerate(components):
        coordinate_component[list(block)] = k
    roi = np.arange(d) % n
    local_mask = roi[:, None] == roi[None, :]
    coefficient = np.zeros((d, d))
    for i in range(n):
        local = np.array([i, i+n]); same = coordinate_component == coordinate_component[i]
        cross = np.flatnonzero(same & (roi != i))
        coefficient[:, local] = _scaled_regression(m.xx, m.xy[:, local], local, cross,
                                                   coupling, mean_penalty)
    residual = m.yy - coefficient.T@m.xy - m.xy.T@coefficient + coefficient.T@m.xx@coefficient
    residual = .5*(residual+residual.T)
    if np.any(np.diag(residual) <= 0):
        raise ArithmeticError('Nonpositive residual variance')
    order = np.column_stack((np.arange(n), np.arange(n)+n)).ravel()
    q = residual[np.ix_(order, order)]
    sd = np.sqrt(np.diag(q)); correlation = q / sd[:, None] / sd[None, :]
    triangular = np.zeros((d, d)); innovation = np.empty(d)
    for j in range(d):
        predecessors = np.arange(j)
        local = predecessors[roi[order[:j]] == roi[order[j]]]
        cross = predecessors[(roi[order[:j]] != roi[order[j]]) &
            (coordinate_component[order[:j]] == coordinate_component[order[j]])]
        beta = _scaled_regression(correlation, correlation[:, j], local, cross,
                                  coupling, noise_penalty)[:, 0]
        variance = 1 - 2*beta@correlation[:, j] + beta@correlation@beta
        if variance <= 0 or not np.isfinite(variance):
            raise ArithmeticError(f'Invalid innovation variance at coordinate {j}: {variance}')
        triangular[j] = beta * sd[j] / sd
        innovation[j] = variance * sd[j]**2
    factor = solve_triangular(np.eye(d)-triangular, np.diag(np.sqrt(innovation)), lower=True)
    noise_ordered = factor@factor.T + ridge*np.eye(d)
    # Cholesky in ROI-adjacent order, then permute back to repository [E; I].
    inverse = np.argsort(order)
    noise = noise_ordered[np.ix_(inverse, inverse)]
    factor_native = factor[np.ix_(inverse, inverse)]
    cross = m.prior @ coefficient
    joint = np.block([[m.prior, cross], [cross.T, coefficient.T@cross+noise]])
    covariance = AffineTransportMapDensityEstimator(np.zeros(2*d), joint, m.count).covariance
    cross_factor = factor_native.copy(); cross_factor[local_mask] = 0
    coefficient_cross = coefficient.copy(); coefficient_cross[local_mask] = 0
    diagnostics = dict(fixed_sc_components=len(components),
        mean_cross_norm=float(np.linalg.norm(coefficient_cross)),
        noise_cross_factor_norm=float(np.linalg.norm(cross_factor)),
        mean_unscaled_cross_norm=float(np.linalg.norm(coefficient_cross)/coupling) if coupling else 0.,
        noise_unscaled_cross_norm=float(np.linalg.norm(cross_factor)/coupling) if coupling else 0.,
        minimum_innovation_variance=float(innovation.min()),
        factor_covariance_error=float(np.max(np.abs(factor_native@factor_native.T+ridge*np.eye(d)-noise))))
    return ContinuousAffine(covariance, coefficient, noise, factor_native, m,
        coupling, mean_penalty, noise_penalty, diagnostics)
