"""Shared, unclipped DMF response trials and auditable affine-TM scores.

The intervention prior is analytically factorized. A full-system affine
transition is fitted before marginalizing background ROIs. All MI queries are
marginals of this ONE fitted joint density; no eigenvalue floors or MI clipping.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
import math
import numpy as np
from scipy.linalg import solve_triangular

from exp.TM.transport_map_density import AffineTransportMapDensityEstimator

TOL = 1e-8
METHODS = ('xi', 'pair_phi_r', 'weak_phi_r', 'phi_r_lu', 'wms', 'o_increment', 'whole_mi')
LABELS = ('ROI-block Xi', 'Pairwise Phi-R', 'Weakest split Phi-R', 'Phi-R-LU scalar',
          'Dynamic WMS', 'Target-augmented O', 'Whole MI', 'Return SC', 'Internal SC',
          'Rest FC', 'Initial response', 'Random')


def logdet(matrix):
    chol = np.linalg.cholesky(matrix)
    return 2 * np.log(np.diag(chol)).sum()


def fit_affine_joint(x, y, halfwidth, ridge=1e-6):
    """Affine TM with a known independent uniform prior's Gaussian moments.

    Ridge is added once to the prior and once to the residual covariance, in
    standardized units. It is not added again to subqueries or Schur complements.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    xs = x.std(axis=0, ddof=1); ys = y.std(axis=0, ddof=1)
    if np.any(xs <= 0) or np.any(ys <= 0) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid affine TM inputs')
    xz = (x - x.mean(axis=0)) / xs; yz = (y - y.mean(axis=0)) / ys
    coefficient = np.linalg.lstsq(xz, yz, rcond=None)[0]
    residual = yz - xz @ coefficient
    prior = np.diag((halfwidth ** 2 / 3) / xs ** 2 + ridge)
    noise = np.cov(residual, rowvar=False) + ridge * np.eye(y.shape[1])
    cross = prior @ coefficient
    future = coefficient.T @ cross + noise
    cov = np.block([[prior, cross], [cross.T, future]])
    # Construct the actual repository affine triangular TM and verify positivity.
    tm = AffineTransportMapDensityEstimator(np.zeros(len(cov)), cov, len(x))
    return tm.covariance


class MIOracle:
    def __init__(self, covariance):
        self.cov = covariance
        self.dim = covariance.shape[0] // 2
        self.n = self.dim // 2
        self.cache = {}; self.det_cache = {}
        self.requests = 0

    def dims(self, rois, future=False):
        offset = self.dim if future else 0
        return tuple(sorted(d + offset for i in rois for d in (i, self.n + i)))

    def determinant(self, dims):
        key = tuple(dims)
        if key not in self.det_cache:
            self.det_cache[key] = float(logdet(self.cov[np.ix_(key, key)])) if key else 0.
        return self.det_cache[key]

    def mi(self, past, future):
        key = (tuple(sorted(past)), tuple(sorted(future)))
        self.requests += 1
        if key not in self.cache:
            a = self.dims(key[0]); b = self.dims(key[1], True)
            self.cache[key] = (self.determinant(a) + self.determinant(b)
                               - self.determinant(a + b)) / (2 * math.log(2))
            if self.cache[key] < -TOL:
                raise ArithmeticError(f'MI below tolerance: {self.cache[key]} bits')
        return self.cache[key]

    def conditional_xi(self, a):
        dx = self.dims(a); dy = self.dims(a, True)
        xx = self.cov[np.ix_(dx, dx)]; xy = self.cov[np.ix_(dx, dy)]
        yy = self.cov[np.ix_(dy, dy)]
        conditional = xx - xy @ np.linalg.solve(yy, xy.T)
        blocks = []
        for i in a:
            inds = [dx.index(i), dx.index(self.n + i)]
            blocks.append(logdet(conditional[np.ix_(inds, inds)]))
        return float((sum(blocks) - logdet(conditional)) / (2 * math.log(2)))


def check_syn(values, tolerance=TOL):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ArithmeticError('Nonfinite Syn')
    bad = values < -tolerance
    if bad.any():
        raise ArithmeticError(f'Syn violation: min={values.min():.12g} bits; '
                              f'threshold={-tolerance} bits; affected={bad.sum()}')
    return int(((values < 0) & ~bad).sum())


@lru_cache(None)
def subsets(k):
    return tuple(tuple(i for i in range(k) if mask & (1 << i)) for mask in range(1, 1 << k))


def phi_lu(oracle, a):
    if not a:
        return 0.
    matrix = np.array([[oracle.mi((i,), (j,)) for j in a] for i in a])
    union = math.fsum((1 if len(s) % 2 else -1) * float(matrix[np.ix_(s, s)].min())
                      for s in subsets(len(a)))
    return oracle.mi(a, a) - union


def phi_bipartition(oracle, u, v):
    a = tuple(sorted(u + v))
    return (oracle.mi(a, a) - oracle.mi(u, u) - oracle.mi(v, v)
            + min(oracle.mi(u, u), oracle.mi(u, v), oracle.mi(v, u), oracle.mi(v, v)))


def score(oracle, a, method):
    a = tuple(sorted(a)); k = len(a)
    if not k:
        return 0.
    if method == 'xi':
        return oracle.mi(a, a) - sum(oracle.mi((i,), a) for i in a)
    if method == 'xi_fast':
        return oracle.conditional_xi(a)
    if method == 'pair_phi_r':
        return float(np.mean([phi_bipartition(oracle, (i,), (j,)) for i, j in combinations(a, 2)])) if k > 1 else 0.
    if method == 'weak_phi_r':
        if k == 1:
            return 0.
        values = []
        for tail in range((1 << (k - 1)) - 1):
            u = (a[0],) + tuple(a[i + 1] for i in range(k - 1) if tail & (1 << i))
            v = tuple(i for i in a if i not in u)
            values.append(phi_bipartition(oracle, u, v))
        return min(values)
    if method == 'phi_r_lu':
        return phi_lu(oracle, a)
    if method == 'wms':
        return oracle.mi(a, a) - sum(oracle.mi((i,), (i,)) for i in a)
    if method == 'o_increment':
        return ((k - 1) * oracle.mi(a, a)
                - sum(oracle.mi(tuple(j for j in a if j != i), a) for i in a))
    if method == 'whole_mi':
        return oracle.mi(a, a)
    if method == 'phi_full':
        return full_phi_lu(oracle, a)[0]
    raise ValueError(method)


@lru_cache(None)
def lattice(k):
    """Antichains and topologically sorted zeta; never construct product zeta."""
    if k > 4:
        raise ValueError('Full Phi-ID reference is restricted to k<=4')
    chains = []
    def visit(start, chosen):
        if chosen:
            chains.append(tuple(chosen))
        for mask in range(start, 1 << k):
            if all((mask & old) not in (mask, old) for old in chosen):
                visit(mask + 1, chosen + [mask])
    visit(1, [])
    relation = np.array([[all(any((left & right) == left for left in alpha)
                                   for right in beta) for alpha in chains] for beta in chains], float)
    order = np.argsort(relation.sum(axis=1), kind='stable')
    chains = tuple(chains[i] for i in order); zeta = relation[np.ix_(order, order)]
    if np.any(np.triu(zeta, 1)):
        raise ArithmeticError('Invalid topological order')
    return chains, zeta


def full_phi_lu(oracle, a):
    k = len(a); chains, zeta = lattice(k)
    all_subsets = subsets(k)
    mi = np.array([[oracle.mi(tuple(a[i] for i in u), tuple(a[i] for i in v))
                    for v in all_subsets] for u in all_subsets])
    r = np.empty((len(chains), len(chains)))
    for i, alpha in enumerate(chains):
        for j, beta in enumerate(chains):
            r[i, j] = mi[np.ix_([m - 1 for m in alpha], [m - 1 for m in beta])].min()
    left = solve_triangular(zeta, r, lower=True)
    atoms = solve_triangular(zeta, left.T, lower=True).T
    local_union = np.zeros_like(atoms, dtype=bool)
    for i in range(k):
        down = np.array([any(m == (1 << i) for m in alpha) for alpha in chains])
        local_union |= down[:, None] & down[None, :]
    whole = oracle.mi(a, a)
    return whole - atoms[local_union].sum(), float(atoms.sum() - whole), len(chains) ** 2


def weaken_return(sc, a, budget):
    inside = np.zeros(len(sc), bool); inside[list(a)] = True
    removed = sc * (inside[:, None] & ~inside[None, :])
    strength = removed.sum()
    alpha = budget / strength
    if not (0 <= alpha <= 1):
        raise ValueError('Invalid removal budget')
    cut = alpha * removed
    return sc - cut, cut, float(alpha), float(strength)


def step(dmf, se, si, matrices, jf, g, p, noise_e, noise_i, current=0.):
    ie = (p.w_e * p.i0 + p.w_plus * p.j_nmda * se
          + g * p.j_nmda * np.matmul(se, matrices.swapaxes(-1, -2)) - jf * si + current)
    ii = p.w_i * p.i0 + p.j_nmda * se - si
    re = dmf.transfer_function(ie, p.gain_e, p.threshold_e, p.shape_e)
    ri = dmf.transfer_function(ii, p.gain_i, p.threshold_i, p.shape_i)
    se = se + p.dt * (-se / p.tau_e + (1 - se) * p.gamma_e * re) + noise_e
    si = si + p.dt * (-si / p.tau_i + ri) + noise_i
    return se, si, re, ri


def relax(dmf, sc, jf, g, p, seed, count=1, seconds=2., sigma=None, initial=None):
    rng = np.random.default_rng(seed)
    se = np.full((count, len(sc)), p.init_se) if initial is None else np.broadcast_to(initial[0], (count, len(sc))).copy()
    si = np.full_like(se, p.init_si) if initial is None else np.broadcast_to(initial[1], se.shape).copy()
    traces = []; diagnostics = Diagnostics()
    scale = (p.sigma if sigma is None else sigma) * math.sqrt(p.dt)
    for _ in range(round(seconds / p.dt)):
        se, si, re, ri = step(dmf, se, si, sc, jf, g, p,
                              scale * rng.standard_normal(se.shape), scale * rng.standard_normal(si.shape))
        diagnostics.add(se, si, re, ri); traces.append(re.copy())
    return se, si, np.asarray(traces), diagnostics.record()


class Diagnostics:
    def __init__(self):
        self.state_min = np.inf; self.state_max = -np.inf
        self.rate_min = np.inf; self.rate_max = -np.inf
        self.outside = 0; self.states = 0; self.abnormal_rates = 0

    def add(self, se, si, re, ri):
        if not all(np.isfinite(x).all() for x in (se, si, re, ri)):
            raise ArithmeticError('Nonfinite DMF state/rate')
        self.state_min = min(self.state_min, se.min(), si.min())
        self.state_max = max(self.state_max, se.max(), si.max())
        self.rate_min = min(self.rate_min, re.min(), ri.min())
        self.rate_max = max(self.rate_max, re.max(), ri.max())
        self.outside += int(((se < 0) | (se > 1)).sum() + ((si < 0) | (si > 1)).sum())
        self.states += se.size + si.size
        self.abnormal_rates += int(((re < 0) | (re > 500)).sum() + ((ri < 0) | (ri > 500)).sum())
        if self.abnormal_rates:
            raise ArithmeticError(f'Abnormal DMF rate: max={self.rate_max} Hz')

    def record(self):
        return dict(state_min=float(self.state_min), state_max=float(self.state_max),
                    rate_min_hz=float(self.rate_min), rate_max_hz=float(self.rate_max),
                    outside_state_count=self.outside, state_count=self.states,
                    abnormal_rate_count=self.abnormal_rates)


def responses(dmf, sc, jf, g, p, initial, site, candidates, budget, amplitude, seed):
    """C x pulse/sham x repetition x ROI, shared exact noise for all conditions."""
    matrices = [sc]; cuts = [np.zeros_like(sc)]; alphas = []; strengths = []
    for a in candidates:
        matrix, cut, alpha, strength = weaken_return(sc, a, budget)
        matrices.append(matrix); cuts.append(cut); alphas.append(alpha); strengths.append(strength)
    matrices = np.asarray(matrices); cuts = np.asarray(cuts)
    reference_e, reference_i = initial
    count, n = reference_e.shape; conditions = len(matrices)
    se = np.tile(reference_e, (conditions, 2, 1, 1)).reshape(conditions, 2 * count, n)
    si = np.tile(reference_i, (conditions, 2, 1, 1)).reshape(se.shape)
    compensation = g * p.j_nmda * np.einsum('cij,rj->cri', cuts, reference_e)
    compensation = np.tile(compensation[:, None], (1, 2, 1, 1)).reshape(se.shape)
    rng = np.random.default_rng(seed); rates = []; diagnostics = Diagnostics()
    for t in range(301):
        current = np.zeros_like(se)
        if t < 10:
            current[:, :count, site] = amplitude
        else:
            current += compensation
        active = matrices if t >= 10 else np.broadcast_to(sc, matrices.shape)
        ne = p.sigma * math.sqrt(p.dt) * rng.standard_normal((count, n))
        ni = p.sigma * math.sqrt(p.dt) * rng.standard_normal((count, n))
        se, si, re, ri = step(dmf, se, si, active, jf, g, p,
                              np.tile(ne, (2, 1)), np.tile(ni, (2, 1)), current)
        diagnostics.add(se, si, re, ri)
        rates.append(re.reshape(conditions, 2, count, n))
    rates = np.asarray(rates).transpose(1, 2, 3, 0, 4)
    induced = rates[:, 0] - rates[:, 1]
    rms = np.sqrt(np.mean(induced ** 2, axis=-1))
    late = np.trapz(rms[:, :, 50:301], dx=p.dt, axis=-1)
    early = np.trapz(rms[:, :, 10:51], dx=p.dt, axis=-1)
    loss = late[0, None, :] - late[1:]
    return dict(rates=rates, rms=rms, late=late, early=early, loss=loss,
                alphas=np.asarray(alphas), strengths=np.asarray(strengths),
                removed_weight=cuts.sum(axis=(1, 2)), diagnostics=diagnostics.record())


def simulate_sources(dmf, x, sc, jf, g, p, seed):
    se, si = np.split(x.copy(), 2, axis=1)
    rng = np.random.default_rng(seed); diagnostic = Diagnostics()
    for _ in range(300):
        ne = p.sigma * math.sqrt(p.dt) * rng.standard_normal(se.shape)
        ni = p.sigma * math.sqrt(p.dt) * rng.standard_normal(si.shape)
        se, si, re, ri = step(dmf, se, si, sc, jf, g, p, ne, ni)
        diagnostic.add(se, si, re, ri)
    return np.concatenate([se, si], axis=1), diagnostic.record()
