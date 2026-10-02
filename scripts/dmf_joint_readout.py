"""Common-target Xi games (nats) and nested conditional maximum-entropy decoders.

Gaussian queries are marginals of ONE affine TM with a diagonal source prior.
The actual DMF samples retain the uniform box prior; Gaussian scores are an
explicit approximation, not estimates of a nonlinear high-order PID atom.
"""
from __future__ import annotations

from itertools import combinations
import math
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

TOL_NATS = 1e-8


def audit_nonnegative(values, tolerance=TOL_NATS):
    a = np.asarray(values, float)
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError('Tolerance must be finite and nonnegative')
    if not np.isfinite(a).all() or not a.size:
        raise ArithmeticError('Empty or nonfinite Xi/Syn audit')
    record = dict(minimum_nats=float(a.min()), tolerance_nats=float(tolerance),
                  tolerance_negative_count=int(((a < 0) & (a >= -tolerance)).sum()),
                  violation_count=int((a < -tolerance).sum()), count=int(a.size))
    if record['violation_count']:
        raise ArithmeticError(f"Xi/Syn violation: min={a.min():.12g} nats; "
                              f"threshold={-tolerance} nats; affected={record['violation_count']}")
    return record


def ld(a):
    return float(2 * np.log(np.diag(np.linalg.cholesky(a))).sum()) if a.size else 0.


class CommonTargetGame:
    """Fixed B=V; scalar-fine Xi v and ROI-block cross residual u."""
    def __init__(self, covariance):
        cov = np.asarray(covariance, float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or len(cov) % 4:
            raise ValueError('Expected [past E,I; future E,I] covariance')
        if not np.isfinite(cov).all() or not np.allclose(cov, cov.T, atol=1e-12):
            raise ValueError('Invalid covariance')
        self.d = len(cov) // 2
        self.n = self.d // 2
        prior = cov[:self.d, :self.d]
        if not np.allclose(prior, np.diag(np.diag(prior)), atol=1e-12, rtol=0):
            raise ValueError('Common factorized intervention prior required')
        ld(cov)
        cross = cov[:self.d, self.d:]
        self.conditional = prior - cross @ np.linalg.solve(cov[self.d:, self.d:], cross.T)
        self.conditional = .5 * (self.conditional + self.conditional.T)
        ld(self.conditional)
        self.log_prior = np.log(np.diag(prior))
        self.scalar_ei = .5 * (self.log_prior - np.log(np.diag(self.conditional)))
        self.cache = {(): 0.}
        self.roi_ei = np.array([self.ei((i,)) for i in range(self.n)])
        self.local_xi = self.roi_ei - self.scalar_ei[:self.n] - self.scalar_ei[self.n:]
        self.audited = list(self.scalar_ei) + list(self.local_xi)

    def dims(self, s):
        s = tuple(sorted(set(map(int, s))))
        if any(i < 0 or i >= self.n for i in s):
            raise ValueError('ROI index out of range')
        return s, (*s, *(i + self.n for i in s))

    def ei(self, s):
        s, dims = self.dims(s)
        if s not in self.cache:
            self.cache[s] = .5 * (self.log_prior[list(dims)].sum()
                                  - ld(self.conditional[np.ix_(dims, dims)]))
        return self.cache[s]

    def v(self, s):
        s, dims = self.dims(s)
        value = self.ei(s) - self.scalar_ei[list(dims)].sum()
        self.audited.append(float(value))
        return float(value)

    def u(self, s):
        s, _ = self.dims(s)
        value = self.ei(s) - self.roi_ei[list(s)].sum()
        self.audited.append(float(value))
        return float(value)

    def totals(self):
        s = tuple(range(self.n))
        whole, fine, cross = self.ei(s), self.v(s), self.u(s)
        local = float(self.local_xi.sum())
        error = float(abs(fine - local - cross))
        if error > TOL_NATS:
            raise ArithmeticError(f'Global Xi budget does not close: {error} nats')
        return dict(whole_ei_nats=whole, xi_nats=fine, roi_local_xi_nats=local,
                    cross_roi_nats=cross, closure_error_nats=error)


def exact_spt(game, members):
    """Manuscript Eq. (10), exhaustive unordered splits; ROI leaves retained.

    This small-set reference deliberately rejects large sets. Global approximate
    search needs a separately declared budget and must not be called exact.
    """
    members = tuple(sorted(members))
    if len(members) > 10:
        raise ValueError('Exact SPT reference restricted to <=10 ROI players')
    nodes = []
    def visit(s):
        if len(s) == 1:
            return
        splits = []
        for mask in range((1 << (len(s) - 1)) - 1):
            left = (s[0],) + tuple(s[i + 1] for i in range(len(s)-1) if mask & (1 << i))
            right = tuple(i for i in s if i not in left)
            splits.append((game.u(left) + game.u(right), left, right))
        _, left, right = max(splits, key=lambda t: (t[0], tuple(-i for i in t[1])))
        syn = game.u(s) - game.u(left) - game.u(right)
        audit_nonnegative([syn])
        nodes.append(dict(members=list(s), left=list(left), right=list(right),
                          roi_order=len(s), syn_nats=syn, split_count=len(splits)))
        visit(left); visit(right)
    if members:
        visit(members)
    mass = {k: math.fsum(n['syn_nats'] for n in nodes if n['roi_order'] == k)
            for k in range(2, len(members)+1)}
    error = abs(math.fsum(mass.values()) - game.u(members))
    if error > TOL_NATS:
        raise ArithmeticError(f'SPT closure failure: {error} nats')
    audit = audit_nonnegative([n['syn_nats'] for n in nodes]) if nodes else None
    return dict(nodes=nodes, mass_by_roi_order=mass, closure_error_nats=error, syn_audit=audit)


def exact_shapley(game, members):
    members = tuple(members); n = len(members)
    if n > 10:
        raise ValueError('Exact Shapley reference restricted to <=10 ROI players')
    value = np.zeros(n)
    for pos, i in enumerate(members):
        others = tuple(j for j in members if j != i)
        for k in range(n):
            weight = 1 / (n * math.comb(n-1, k))
            for s in combinations(others, k):
                value[pos] += weight * (game.u((*s, i)) - game.u(s))
    error = abs(value.sum() - game.u(members))
    if error > TOL_NATS:
        raise ArithmeticError(f'Shapley closure failure: {error} nats')
    audit = audit_nonnegative(value) if n else None
    return dict(members=list(members), cross_shapley_nats=value.tolist(),
                closure_error_nats=float(error), nonnegative_audit=audit)


def label_codes(source, reference, members):
    bits = (source[:, list(members)] > reference[list(members)]).astype(int)
    return bits @ (1 << np.arange(len(members))), bits


def label_statistics(k, order):
    """Walsh basis: factorized (1), conditional pairwise (2), joint (k).

    All energies use the same linear future features, penalty, optimizer and
    stopping criterion. At k=2, pairwise and joint are algebraically identical.
    """
    bits = ((np.arange(1 << k)[:, None] >> np.arange(k)) & 1)
    spins = 2 * bits - 1
    groups = [s for r in range(1, min(k, order)+1) for s in combinations(range(k), r)]
    return np.column_stack([spins[:, s].prod(1) for s in groups]).astype(float)


def fit_decoder(features, labels, k, order, penalty, maxiter=300):
    stats = label_statistics(k, order)
    target = stats[labels]
    shape = (features.shape[1], stats.shape[1])
    def objective(flat):
        coefficient = flat.reshape(shape)
        logits = features @ coefficient @ stats.T
        norm = logsumexp(logits, axis=1)
        prob = np.exp(logits - norm[:, None])
        loss = (norm - (logits[np.arange(len(labels)), labels])).mean()
        # Same shrinkage for every orthogonal label interaction, including bias.
        loss += .5 * penalty * np.sum(coefficient**2)
        gradient = features.T @ (prob @ stats - target) / len(labels) + penalty * coefficient
        return float(loss), gradient.ravel()
    start = time.perf_counter()
    fit = minimize(objective, np.zeros(np.prod(shape)), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-5))
    if not fit.success:
        raise ArithmeticError(f'Decoder did not converge: {fit.message}')
    return dict(coefficient=fit.x.reshape(shape), stats=stats, iterations=int(fit.nit),
                seconds=time.perf_counter()-start, parameters=int(np.prod(shape)),
                gradient_max=float(np.abs(fit.jac).max()))


def posterior(model, features):
    logits = features @ model['coefficient'] @ model['stats'].T
    return np.exp(logits - logsumexp(logits, axis=1)[:, None])


def decoder_metrics(prob, labels, k):
    codes = np.arange(1 << k)
    bits = ((codes[:, None] >> np.arange(k)) & 1)
    prediction = prob.argmax(1)
    predicted_bits = (prob @ bits >= .5).astype(int)
    true_bits = bits[labels]
    return dict(mode_accuracy=float((prediction == labels).mean()),
                bit_accuracy=float((predicted_bits == true_bits).mean()),
                nll_nats=float(-np.log(prob[np.arange(len(labels)), labels]).mean()),
                correct=prediction == labels)


def paired_effect(a, b):
    difference = np.asarray(a, float) - np.asarray(b, float)
    return dict(difference=float(difference.mean()),
                trajectory_sem=float(difference.std(ddof=1)/np.sqrt(len(difference))),
                discordant_count=int(np.count_nonzero(difference)), trajectories=len(difference))
