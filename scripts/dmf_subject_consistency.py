"""Common-density individual-SC DMF comparisons; every information value is nats."""
from __future__ import annotations

from dataclasses import asdict
from itertools import combinations
import math
import numpy as np
from scipy.linalg import cholesky
from scipy.stats import spearmanr

from scripts.dmf_joint_readout import CommonTargetGame, audit_nonnegative
from scripts.spt import SPTConfig, build_spt, canonical_split, spectral_candidate_selector

TOL = 1e-8


def network_values(game, groups):
    """Exact seven-block Shapley of EI(union)-sum EI(network), fixed full target."""
    n = len(groups)
    network_ei = np.array([game.ei(g) for g in groups])
    within = np.array([game.v(g) for g in groups])
    values = np.zeros(1 << n)
    for mask in range(1, 1 << n):
        selected = [i for i in range(n) if mask & (1 << i)]
        members = [r for i in selected for r in groups[i]]
        values[mask] = game.ei(members) - network_ei[selected].sum()
    shapley = np.zeros(n)
    marginals = []
    for i in range(n):
        for mask in range(1 << n):
            if mask & (1 << i):
                continue
            k = mask.bit_count()
            delta = values[mask | (1 << i)] - values[mask]
            shapley[i] += delta / (n * math.comb(n - 1, k))
            marginals.append(delta)
    error = abs(shapley.sum() - values[-1])
    budget_error = abs(within.sum() + values[-1] - game.v(range(game.n)))
    if max(error, budget_error) > TOL:
        raise ArithmeticError(f'Network closure failure: {error}, {budget_error} nats')
    audit = audit_nonnegative([*values, *marginals, *within, *shapley])
    return within, shapley, float(values[-1]), dict(
        shapley_closure_error_nats=float(error), budget_closure_error_nats=float(budget_error),
        nonnegative_audit=audit)


def pair_values(game):
    affinity = np.zeros((game.n, game.n))
    for i, j in combinations(range(game.n), 2):
        affinity[i, j] = affinity[j, i] = game.u((i, j))
    return affinity, audit_nonnegative(affinity[np.triu_indices(game.n, 1)])


def roi_shapley(game, pairs=256, seed=20261002):
    """Antithetic permutations; draws shared across individuals and conditions.

    Each prefix increment equals the scalar-fine Xi marginal. The sum closes to
    total Xi on every permutation. Subtract local E/I Xi to get cross-ROI Shapley.
    MC SE uses independent forward/reverse pairs, not 2*pairs observations.
    """
    n = game.n
    c = game.conditional
    scalar = np.log(np.diag(c))[:n] + np.log(np.diag(c))[n:]
    rng = np.random.default_rng(seed)
    samples = np.empty((pairs, n))
    minimum, count, error = np.inf, 0, 0.
    total = game.v(range(n))
    for draw in range(pairs):
        p = rng.permutation(n)
        terms = []
        for order in (p, p[::-1]):
            dims = np.column_stack((order, order + n)).ravel()
            factor = cholesky(c[np.ix_(dims, dims)], lower=True, check_finite=False)
            increment = 2 * np.log(np.diag(factor)).reshape(n, 2).sum(1)
            value = np.empty(n)
            value[order] = .5 * (scalar[order] - increment)
            a = audit_nonnegative(value)
            minimum = min(minimum, a['minimum_nats'])
            count += a['tolerance_negative_count']
            error = max(error, float(abs(value.sum() - total)))
            terms.append(value)
        samples[draw] = (terms[0] + terms[1]) / 2
    if error > TOL:
        raise ArithmeticError(f'ROI Shapley closure failure: {error} nats')
    mean = samples.mean(0)
    se = samples.std(0, ddof=1) / np.sqrt(pairs)
    cross = mean - game.local_xi
    cross_audit = audit_nonnegative(cross)
    first, second = samples[:pairs//2].mean(0), samples[pairs//2:].mean(0)
    return mean, se, cross, dict(
        antithetic_pairs=pairs, rng_seed=seed, tolerance_nats=TOL,
        minimum_marginal_nats=float(minimum), tolerance_negative_count=count,
        maximum_closure_error_nats=error, maximum_mc_se_nats=float(se.max()),
        split_half_spearman=float(spearmanr(first, second).statistic),
        split_half_top10_overlap=len(set(np.argsort(first)[-10:]) & set(np.argsort(second)[-10:])),
        cross_audit=cross_audit, convergence='fixed pilot budget; MC error reported, not certified converged')


class ResidualOracle:
    def __init__(self, game, affinity=None):
        self.game, self.affinity = game, affinity
        self.cache = {}

    def xi(self, s):
        s = tuple(sorted(s))
        if len(s) <= 1:
            return 0.
        if s not in self.cache:
            self.cache[s] = (self.game.u(s) if self.affinity is None
                             else float(self.affinity[np.ix_(s, s)].sum() / 2))
        return self.cache[s]


def tree_record(game, affinity, *, pairwise=False, extra=256):
    """Identical candidate rule for both objectives, exact <=10, no Yeo prior."""
    oracle = ResidualOracle(game, affinity if pairwise else None)
    base = spectral_candidate_selector(affinity, exact_max_size=10)
    gaps = []

    def select(s):
        kind, candidates = base(s)
        candidates = set(candidates)
        if len(s) > 10:
            candidates.update(canonical_split((i,), set(s)-{i}, s) for i in s)
            rng = np.random.default_rng(np.random.SeedSequence([731, *s]))
            for _ in range(extra):
                k = int(rng.integers(1, len(s)//2 + 1))
                left = tuple(rng.choice(s, size=k, replace=False).tolist())
                candidates.add(canonical_split(left, set(s)-set(left), s))
            kind = 'spectral-plus-singletons-plus-random'
        scores = sorted(oracle.xi(s)-oracle.xi(l)-oracle.xi(r) for l, r in candidates)
        audit_nonnegative(scores)
        gaps.append(dict(members=list(s), candidate_count=len(scores),
                         best_syn_nats=float(scores[0]),
                         runner_up_gap_nats=float(scores[1]-scores[0]) if len(scores)>1 else None))
        return kind, sorted(candidates)

    result = build_spt(tuple(range(game.n)), oracle,
                       config=SPTConfig(syn_tolerance=TOL), candidate_selector=select)
    if abs(result.closure_error) > TOL:
        raise ArithmeticError(f'SPT closure failure: {result.closure_error} nats')

    def pack(node):
        return dict(members=list(node.sources), value_nats=node.xi_value,
                    syn_nats_raw=node.syn_value, search=node.split_kind,
                    children=[pack(c) for c in node.children])

    node = result.root
    while node.size > 10 and node.children:
        node = sorted(node.children, key=lambda c: (-c.xi_value, c.sources))[0]
    core = list(node.sources) if node.size >= 2 else []
    return dict(tree=pack(result.root), core10=core, core_size=len(core),
                closure_error_nats=result.closure_error, audit=asdict(result.audit),
                gaps=gaps, objective='pairwise-additive proxy' if pairwise else 'cross-ROI residual u')


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b)/len(a | b) if a and b else None


def transition_interval(g, rates):
    """Coarse maximum rate slope; endpoint maxima remain unlocated."""
    slope = np.diff(np.asarray(rates)) / np.diff(g)
    i = int(np.argmax(slope))
    located = bool(0 < i < len(slope)-1 and slope[i] > 0)
    return dict(located=located, interval=[float(g[i]), float(g[i+1])],
                midpoint=float((g[i]+g[i+1])/2) if located else None,
                maximum_slope_hz_per_g=float(slope[i]),
                reason='interior maximum slope' if located else 'boundary or no positive rate rise')
