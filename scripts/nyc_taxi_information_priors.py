"""Three-scale Taxi priors under the 2026-10-11 calibration protocol.

All information queries are marginals of one coherent affine triangular TM
density. Information is in bits. Empirical block interventions and the Gaussian
readout approximation are distinct from exact maximum-entropy EI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from exp.TM.transport_map_density import (
    AffineTransportMapDensityEstimator, fit_affine_transport_map_density,
)
from scripts.unicm_observational_information import gaussian_surd_allocation

VERSION = "taxi-three-scale-affine-tm-v1"
TOLERANCE_BITS = 1e-8
RIDGE = 1e-6
BRANCHES = ("recent", "daily", "weekly")
LAGS = (np.arange(7, 0, -1), 24 * np.arange(5, 0, -1), 168 * np.arange(7, 0, -1))
BLOCKS = ((0, 1), (2, 3), (4, 5))
NONNEGATIVE_PRIORS = {"xi_shapley", "ei_shapley", "pair_only", "surd", "correlation", "grouped_shap", "uniform"}


def nonnegative(values, name: str, audit: dict) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    if not raw.size or not np.isfinite(raw).all():
        raise ValueError(f"{name}: empty or nonfinite information")
    bad, tiny = raw < -TOLERANCE_BITS, (raw < 0) & (raw >= -TOLERANCE_BITS)
    record = audit.setdefault(name, dict(minimum_bit=float(raw.min()), tolerance_bit=TOLERANCE_BITS,
                                         numerical_zero_count=0, significant_negative_count=0))
    record["minimum_bit"] = min(record["minimum_bit"], float(raw.min()))
    record["numerical_zero_count"] += int(tiny.sum())
    record["significant_negative_count"] += int(bad.sum())
    if tiny.any():
        record.setdefault("numerical_zero_raw_values_bit", []).extend(raw[tiny].tolist())
    if bad.any():
        raise ValueError(f"{name}: minimum={raw.min():.12g} bit, threshold=-{TOLERANCE_BITS:g} bit, affected_count={bad.sum()}")
    result = raw.copy()
    result[tiny] = 0.0  # Declared numerical-zero rule; raw arrays remain in native output.
    return result


def positive_covariance(cov: np.ndarray, name: str, audit: dict, ridge: float = RIDGE) -> float:
    if not np.isfinite(cov).all() or not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError(f"{name}: nonfinite/asymmetric covariance")
    eigen = np.linalg.eigvalsh(cov)
    entry = dict(min_eigenvalue=float(eigen[0]), condition_number=float(np.linalg.cond(cov)),
                 maximum_diagonal=float(np.diag(cov).max()), ridge=ridge)
    audit.setdefault("covariances", {})[name] = entry
    if eigen[0] <= 0:
        raise ValueError(f"{name}: covariance not positive definite: {entry}")
    return float(2 * np.log(np.diag(np.linalg.cholesky(cov))).sum())


def shapley(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != 8 or not np.isfinite(values).all() or np.max(np.abs(values[0])) > TOLERANCE_BITS:
        raise ValueError("Expected complete three-player game with zero empty value")
    out = np.zeros((3,) + values.shape[1:])
    for m in range(3):
        for s in range(8):
            if not s & (1 << m):
                weight = 1 / (3 * math.comb(2, s.bit_count()))
                out[m] += weight * (values[s | (1 << m)] - values[s])
    if np.max(np.abs(out.sum(axis=0) - values[7])) > TOLERANCE_BITS:
        raise ValueError("Shapley efficiency failed")
    return out


def source_columns(mask: int) -> list[int]:
    return [j for m, block in enumerate(BLOCKS) if mask & (1 << m) for j in block]


class GaussianOracle:
    def __init__(self, density: AffineTransportMapDensityEstimator, audit: dict, name: str):
        self.density, self.cov, self.audit, self.name = density, density.covariance, audit, name
        positive_covariance(self.cov, name, audit)

    def logdet(self, indices: list[int]) -> float:
        if not indices:
            return 0.0
        cov = self.cov[np.ix_(indices, indices)]
        return float(2 * np.log(np.diag(np.linalg.cholesky(cov))).sum())

    def mi(self, source: list[int], target: list[int]) -> float:
        if not source:
            return 0.0
        raw = (self.logdet(source) + self.logdet(target) - self.logdet(source + target)) / (2 * np.log(2))
        return float(nonnegative([raw], self.name + ":mi", self.audit)[0])

    def omega(self, blocks: list[list[int]]) -> float:
        if len(blocks) < 3:
            return 0.0
        joined = [j for b in blocks for j in b]
        # Gaussian entropy constants cancel exactly in O-information.
        value = (len(blocks) - 2) * self.logdet(joined)
        value += sum(self.logdet(b) for b in blocks)
        value -= sum(self.logdet([j for k, b in enumerate(blocks) if k != m for j in b])
                     for m in range(len(blocks)))
        return value / (2 * np.log(2))


@dataclass
class SourceRepresentation:
    means: tuple[np.ndarray, ...]  # [zone,channel,2*history]
    bases: tuple[np.ndarray, ...]
    scales: np.ndarray  # [zone,scale,channel]
    explained: np.ndarray
    constant: np.ndarray

    @staticmethod
    def encode(raw: np.ndarray) -> np.ndarray:
        if np.any(raw < 0) or not np.isfinite(raw).all():
            raise ValueError("Hurdle encoding requires finite nonnegative counts")
        return np.concatenate([(raw > 0).astype(float), np.log1p(raw)], axis=-1)

    @classmethod
    def fit(cls, flow: np.ndarray, pool: np.ndarray):
        means, bases = [], []
        shape = (flow.shape[1], 3, 2)
        scales, explained, constant = np.ones(shape), np.zeros(shape), np.zeros(shape, dtype=bool)
        for m, lag in enumerate(LAGS):
            raw = flow[pool[:, None] - lag].transpose(0, 2, 3, 1)
            encoded = cls.encode(raw)
            mean = encoded.mean(axis=0)
            basis = np.zeros_like(mean)
            for zone in range(flow.shape[1]):
                for c in range(2):
                    x = encoded[:, zone, c] - mean[zone, c]
                    energy, vectors = np.linalg.eigh(x.T @ x / (len(x) - 1))
                    if energy[-1] <= 1e-14:
                        constant[zone, m, c] = True
                        continue
                    v = vectors[:, -1]
                    v *= 1 if v[np.argmax(np.abs(v))] >= 0 else -1
                    basis[zone, c] = v
                    scales[zone, m, c] = np.std(x @ v, ddof=0)
                    explained[zone, m, c] = energy[-1] / energy.sum()
            means.append(mean)
            bases.append(basis)
        return cls(tuple(means), tuple(bases), scales, explained, constant)

    def transform(self, flow: np.ndarray, times: np.ndarray) -> np.ndarray:
        scores = []
        for m, lag in enumerate(LAGS):
            raw = flow[times[:, None] - lag].transpose(0, 2, 3, 1)
            encoded = self.encode(raw) - self.means[m]
            scores.append(np.einsum("tzcl,zcl->tzc", encoded, self.bases[m]) / self.scales[:, m])
        return np.stack(scores, axis=2)  # [time,zone,scale,channel]


def fit_intervention(source: np.ndarray, response: np.ndarray, pool: np.ndarray,
                     fit: np.ndarray, evaluate: np.ndarray, audit: dict, name: str):
    """Fit a coherent independent-block affine response in standardized coordinates."""
    x, y, p = source.reshape(len(source), 6), response, pool.reshape(len(pool), 6)
    xmean, ymean = x[fit].mean(axis=0), y[fit].mean(axis=0)
    xc, yc = x[fit] - xmean, y[fit] - ymean
    coefficients = np.linalg.solve(xc.T @ xc / len(fit) + RIDGE * np.eye(6), xc.T @ yc / len(fit))
    intercept = ymean - xmean @ coefficients
    residual = yc - xc @ coefficients
    noise = np.cov(residual, rowvar=False) + RIDGE * np.eye(2)
    sigma = np.zeros((6, 6))
    for block in BLOCKS:
        sigma[np.ix_(block, block)] = np.cov(p[:, block], rowvar=False) + RIDGE * np.eye(2)
    cross = sigma @ coefficients
    cov = np.block([[sigma, cross], [cross.T, coefficients.T @ cross + noise]])
    positive_covariance(cov, name + ":joint_before_transport", audit)
    density = AffineTransportMapDensityEstimator(np.r_[np.zeros(6), intercept], cov, len(fit))
    oracle = GaussianOracle(density, audit, name)
    table = np.array([oracle.mi(source_columns(s), [6, 7]) for s in range(8)])
    singleton = table[[1, 2, 4]]
    membership = ((np.arange(8)[:, None] >> np.arange(3)) & 1)
    xi_raw = table - membership @ singleton
    xi = nonnegative(xi_raw, name + ":xi", audit)
    xi_score_raw, ei_score_raw = shapley(xi_raw), shapley(table)
    xi_score = nonnegative(xi_score_raw, name + ":xi_shapley", audit)
    ei_score = nonnegative(ei_score_raw, name + ":ei_shapley", audit)
    identity = float(np.max(np.abs(ei_score_raw - xi_score_raw - singleton)))
    if identity > TOLERANCE_BITS:
        raise ValueError("EI-Shapley minus Xi-Shapley identity failed")
    pair = np.zeros(3)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        syn = nonnegative([xi_raw[(1 << a) | (1 << b)]], name + ":pair_syn", audit)[0]
        pair[[a, b]] += syn / 2
    predicted = x[evaluate] @ coefficients + intercept
    holdout = np.c_[x[evaluate], y[evaluate]]
    reference = density.map_to_reference(holdout)
    empirical = np.cov(x, rowvar=False)
    cross_block = empirical.copy()
    for b in BLOCKS:
        cross_block[np.ix_(b, b)] = 0
    diagnostics = dict(heldout_rmse_z=float(np.sqrt(np.mean((y[evaluate] - predicted) ** 2))),
                       heldout_log_density=float(density.log_prob(holdout).mean()),
                       reference_mean=reference.mean(axis=0).tolist(),
                       reference_variance=reference.var(axis=0).tolist(),
                       empirical_cross_block_covariance_max=float(np.abs(cross_block).max()),
                       coherent_cross_block_covariance_max=0.0,
                       shapley_identity_error_bit=identity)
    native = dict(ei_bits=table, xi_bits_raw=xi_raw, xi_shapley_raw=xi_score_raw,
                  ei_shapley_raw=ei_score_raw, coefficients=coefficients,
                  joint_covariance=cov, residual_covariance=noise)
    return dict(xi_shapley=xi_score, ei_shapley=ei_score, pair_only=pair), native, diagnostics


def observational_priors(source: np.ndarray, target: np.ndarray, audit: dict, name: str):
    x = source.reshape(len(source), 6)
    density = fit_affine_transport_map_density(np.c_[x, target], jitter=RIDGE)
    oracle = GaussianOracle(density, audit, name)
    wms, phi, ogame, scalar = np.zeros(8), np.zeros(8), np.zeros(8), np.zeros((8, 2))
    for s in range(1, 8):
        columns = source_columns(s)
        inflow, outflow = columns[::2], columns[1::2]
        wms[s] = oracle.mi(columns, [6, 7]) - oracle.mi(inflow, [6]) - oracle.mi(outflow, [7])
        phi[s] = wms[s] + min(oracle.mi(a, [b]) for a in (inflow, outflow) for b in (6, 7))
        blocks = [list(BLOCKS[m]) for m in range(3) if s & (1 << m)]
        ogame[s] = -(oracle.omega(blocks + [[6, 7]]) - oracle.omega(blocks))
        scalar[s] = [oracle.mi(columns, [b]) for b in (6, 7)]
    nonnegative(phi, name + ":phi_r_native", audit)
    surd = gaussian_surd_allocation(scalar, 3, audit)
    raw_cov = np.cov(np.c_[x, target], rowvar=False)
    scale = np.sqrt(np.diag(raw_cov))
    denominator = scale[:6, None] * scale[None, 6:]
    correlation = np.divide(raw_cov[:6, 6:], denominator, out=np.zeros((6, 2)), where=denominator > 0)
    scores = dict(phi_r=shapley(phi), phi_wms=shapley(wms), o_shapley=shapley(ogame),
                  surd=surd["attribution_flat"].mean(axis=0),
                  correlation=np.abs(correlation).reshape(3, 2, 2).mean(axis=(1, 2)))
    native = dict(phi_r_game=phi, phi_wms_game=wms, o_game=ogame,
                  scalar_mi_bits=scalar, surd_synergy_atoms=surd["synergy_atoms"],
                  surd_redundancy_unique_atoms=surd["redundancy_unique_atoms"],
                  observational_joint_covariance=density.covariance)
    audit.setdefault("closures", {})[name] = dict(surd_information=surd["closure_error_bit"],
                                                 surd_attribution=surd["attribution_closure_error_bit"])
    return scores, native


def penalty_weights(raw: np.ndarray, method: str, gamma: float, audit: dict | None = None):
    record = {} if audit is None else audit
    raw = np.asarray(raw, dtype=float)
    if raw.shape[-1] != 3 or not np.isfinite(raw).all():
        raise ValueError("Expected finite three-scale prior")
    if method in NONNEGATIVE_PRIORS:
        if method in ("correlation", "grouped_shap", "uniform"):
            if (raw < 0).any():
                raise ValueError(f"{method}: negative dimensionless score")
            scores = raw.copy()
        else:
            scores = nonnegative(raw, method + ":prior", record)
        offset = np.zeros(raw.shape[:-1] + (1,))
    else:
        offset = -np.minimum(0, raw.min(axis=-1, keepdims=True))
        scores = raw + offset
    total = scores.sum(axis=-1, keepdims=True)
    normalized = np.divide(scores, total, out=np.full_like(scores, 1 / 3), where=total > 0)
    penalties = (normalized + 0.05 / 3) ** (-gamma)
    penalties /= penalties.mean(axis=-1, keepdims=True)
    return penalties, dict(negative_raw_count=int((raw < 0).sum()), offsets=offset.squeeze(-1).tolist(),
                           penalty_std=float(penalties.std()), gamma=float(gamma))
