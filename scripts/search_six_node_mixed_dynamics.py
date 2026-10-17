"""Budgeted six-site mixed-order dynamics screen; exact discrete probabilities.

No module constraints are supplied to MIC or SPT. See the experiment contract in
docs/log/six_node_mixed_order_search/plan.md for selection and confirmation rules.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist, squareform
from sklearn.cluster import KMeans, SpectralClustering
from scipy.special import expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.compare_kuramoto_short_trajectory_clustering import (
    annotate, canonical_groups, hierarchy_result, groups_from_labels,
    standardize, distance_correlation_matrix, digest,
)
from scripts.spt import (SPTConfig, all_nonempty_subsets, nontrivial_bipartitions,
                         build_spt_from_ei_table)

OUT = ROOT / "results/six_node_mixed_dynamics_search"
LOG = ROOT / "docs/log/six_node_mixed_order_search"
BITS = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(int)
SPINS = 2 * BITS - 1
X = np.repeat(SPINS, 64, axis=0)
Y = np.tile(SPINS, (64, 1))
SUBSETS = all_nonempty_subsets(tuple(range(1, 7)))
WEIGHTS = {(0, 1): math.sqrt(1.5), (0, 2): math.sqrt(1.5)/4,
           (1, 2): math.sqrt(1.5)/4}
TOL = 1e-10


def check_budget():
    budget = json.loads((LOG / "budget.json").read_text())
    if time.time() >= budget["deadline_epoch"]:
        raise TimeoutError("Shared authorized deadline reached")
    return budget


def entropy(prob):
    positive = np.asarray(prob)[np.asarray(prob) > 0]
    return float(-np.sum(positive * np.log2(positive)))


class DiscreteInformation:
    """Marginal entropy of arbitrary per-node discrete observations, in bits."""
    def __init__(self, blocks, probabilities, cardinality):
        self.blocks = np.asarray(blocks, dtype=int)
        self.prob = np.asarray(probabilities, dtype=float)
        self.cardinality = cardinality
        self.cache = {(): 0.0}

    def h(self, group):
        key = tuple(sorted(group))
        if key not in self.cache:
            indices = np.array(key) - 1
            codes = self.blocks[:, indices] @ self.cardinality ** np.arange(len(key))
            self.cache[key] = entropy(np.bincount(codes, weights=self.prob,
                                                   minlength=self.cardinality**len(key)))
        return self.cache[key]

    def mi(self, left, right):
        return self.h(left) + self.h(right) - self.h(tuple(left) + tuple(right))


def mic(info, normalization="joint_entropy", order=tuple(range(1, 7))):
    active = [{"sources": [i], "children": []} for i in order]
    trace = []
    while len(active) > 1:
        candidates = []
        for a, b in itertools.combinations(active, 2):
            value = info.mi(a["sources"], b["sources"])
            denominator = (info.h(a["sources"] + b["sources"])
                           if normalization == "joint_entropy"
                           else len(a["sources"]) + len(b["sources"]))
            score = value / denominator if denominator else 0.
            candidates.append((score, tuple(a["sources"]), tuple(b["sources"]), value, a, b))
        score, _, _, value, a, b = min(candidates, key=lambda r: (-r[0], r[1], r[2]))
        trace.append(dict(left=a["sources"], right=b["sources"], mi_bits=value,
                          similarity=score))
        active = [node for node in active if node is not a and node is not b]
        active.append(dict(sources=sorted(a["sources"] + b["sources"]),
                           children=[a, b], merge_mi_bits=value, merge_similarity=score))
        active.sort(key=lambda node: tuple(node["sources"]))
    tree = active[0]
    return annotate(dict(kind="hierarchy", tree=tree, merge_trace=trace,
                         groups=canonical_groups(c["sources"] for c in tree["children"])), .04)


def observations(view):
    if view == "transition":
        return ((X + 1)//2 + 2*((Y + 1)//2)).astype(int), 4
    if view == "increment":
        return ((Y-X)//2 + 1).astype(int), 3
    if view == "future":
        return ((Y+1)//2).astype(int), 2
    raise ValueError(view)


def spin_kernel(beta=1., triad=1., laziness=0., cross=.04, family="heatbath", tau=1.):
    """Parallel heat bath for a mixed-order Ising Hamiltonian.

    H(x)=-sum_P w_ij x_i x_j-J3*x4*x5*x6-cross/3*sum_cross x_i*x_j.
    Independent lazy updates. Parallel updates need not preserve Gibbs equilibrium.
    """
    if family == "correlated_flip":
        energy = sum(w*SPINS[:, i]*SPINS[:, j] for (i, j), w in WEIGHTS.items())
        energy += triad*np.prod(SPINS[:, 3:], axis=1)
        energy += cross/3*SPINS[:, :3].sum(axis=1)*SPINS[:, 3:].sum(axis=1)
        logp = beta*energy
        pz = np.exp(logp-logsumexp(logp))
        z = X*Y
        codes = ((z+1)//2) @ (2**np.arange(6))
        return pz[codes].reshape(64, 64)
    fields = np.zeros((64, 6))
    for (i, j), w in WEIGHTS.items():
        fields[:, i] += w*SPINS[:, j]
        fields[:, j] += w*SPINS[:, i]
    for i in range(3, 6):
        fields[:, i] += triad*np.prod(SPINS[:, [j for j in range(3, 6) if j != i]], axis=1)
    fields[:, :3] += cross/3*SPINS[:, 3:].sum(axis=1)[:, None]
    fields[:, 3:] += cross/3*SPINS[:, :3].sum(axis=1)[:, None]
    if family == "glauber_ctmc":
        rates = expit(-2*beta*SPINS*fields)
        generator = np.zeros((64, 64))
        for code in range(64):
            for i in range(6):
                generator[code, code ^ (1 << i)] = rates[code, i]
        generator[np.diag_indices(64)] = -rates.sum(axis=1)
        kernel = expm(tau*generator)
        if kernel.min() < 0 or np.max(np.abs(kernel.sum(axis=1)-1)) > 1e-12:
            raise ArithmeticError("Invalid CTMC exponential; no projection")
        return kernel
    pplus = (1-laziness)*expit(2*beta*fields) + laziness*BITS
    kernel = np.prod(np.where(BITS[None, :, :] == 1, pplus[:, None, :],
                               1-pplus[:, None, :]), axis=2)
    if np.max(np.abs(kernel.sum(axis=1)-1)) > 1e-12 or kernel.min() < 0:
        raise ArithmeticError("Invalid Markov kernel")
    return kernel


def exact_ei(kernel):
    """Full fixed future target; independent uniform six-bit source intervention."""
    prob = kernel.reshape(-1)/64
    hy = entropy(kernel.mean(axis=0))
    source_bits = ((X+1)//2).astype(int)
    future_code = ((Y+1)//2).astype(int) @ (2**np.arange(6))
    table = {}
    for subset in SUBSETS:
        ids = np.array(subset)-1
        code_a = source_bits[:, ids] @ (2**np.arange(len(subset)))
        joint = np.bincount(64*code_a+future_code, weights=prob,
                            minlength=64*2**len(subset))
        table[subset] = len(subset)+hy-entropy(joint)
    return table


def spt(kernel):
    table = exact_ei(kernel)
    raw = []
    for a in SUBSETS:
        for l, r in nontrivial_bipartitions(a):
            raw.append(table[a]-table[l]-table[r])
    raw = np.asarray(raw)
    audit = dict(tolerance_bits=TOL, candidate_count=len(raw),
                 minimum_syn_bits=float(raw.min()),
                 small_negative_count=int(np.sum((raw < 0) & (raw >= -TOL))),
                 violation_count=int(np.sum(raw < -TOL)), clipping=False)
    if audit["violation_count"]:
        raise ArithmeticError(f"Syn min {raw.min()} bits below {-TOL}; count {audit['violation_count']}")
    built = build_spt_from_ei_table(tuple(range(1, 7)), table,
                                    config=SPTConfig(syn_tolerance=TOL, eps=TOL))
    def convert(node):
        return dict(sources=list(node.sources), children=[convert(c) for c in node.children],
                    xi_bits=node.xi_value, syn_bits=node.syn_value)
    tree = convert(built.root)
    return annotate(dict(kind="hierarchy", tree=tree,
                         groups=canonical_groups(c["sources"] for c in tree["children"]),
                         ei_bits={"".join(map(str, key)): val for key, val in table.items()},
                         syn_audit=audit, closure_error_bits=built.closure_error), .04)


def screen():
    check_budget()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for beta, triad, lazy in itertools.product((.5, 1., 2., 4.), (.5, 1., 2.), (0., .5, .9, .98)):
        check_budget()
        config = dict(family="heatbath", beta=beta, triad=triad, laziness=lazy, cross=.04)
        kernel = spin_kernel(**config)
        views = {}
        for view in ("transition", "increment", "future"):
            blocks, card = observations(view)
            info = DiscreteInformation(blocks, kernel.reshape(-1)/64, card)
            views[view] = {norm: mic(info, norm) for norm in ("joint_entropy", "dimension")}
        rows.append(dict(config=config, views=views))
    payload = dict(stage="population_MIC_screen_no_SPT", conditions=rows,
                   candidate_count=len(rows), timestamp=time.time())
    (OUT/"spin_screen.json").write_text(json.dumps(payload, indent=2))
    for view in ("transition", "increment", "future"):
        fails = [r for r in rows if not r["views"][view]["joint_entropy"]["exact_original_two_groups"]]
        print(view, "MIC failures", len(fails), "/", len(rows))
        for row in fails[:5]:
            print(row["config"], row["views"][view]["joint_entropy"]["groups"])


def lv_data(config, seed=0, count=4800):
    """Positive stochastic Ricker discretization of genuine higher-order LV.

    log(N_i'/N_i)=dt*(1-self*N_i-pair*sum w_ij*N_j
        -triad*N_j*N_k-cross/3*sum_other N_j)+noise*epsilon_i.
    The abundance drift N_i*g_i contains the three-distinct-site product only
    in module 456; no centered variables or hidden internal pair terms.
    """
    source = np.random.default_rng(1000+seed).uniform(.25, 1.75, (count, 6))
    growth = 1-config["self"]*source
    for (i, j), w in WEIGHTS.items():
        growth[:, i] -= w*source[:, j]
        growth[:, j] -= w*source[:, i]
    for i in range(3, 6):
        growth[:, i] -= math.sqrt(2*sum(w*w for w in WEIGHTS.values()))*np.prod(
            source[:, [j for j in range(3, 6) if j != i]], axis=1)
    growth[:, :3] -= .04/3*source[:, 3:].sum(axis=1)[:, None]
    growth[:, 3:] -= .04/3*source[:, :3].sum(axis=1)[:, None]
    eps = np.random.default_rng(11000+seed).normal(size=source.shape)
    future = source*np.exp(config["dt"]*growth+config["noise"]*eps)
    if not np.isfinite(future).all() or np.any(future <= 0):
        raise FloatingPointError("Invalid abundance; no clipping")
    return source, future


def screen_lv():
    from scripts.compare_kuramoto_short_trajectory_clustering import analyze_view
    from threadpoolctl import threadpool_limits
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for damping, dt, noise in itertools.product((.5, 2.), (.25, .5), (.04, .15, .4)):
        check_budget()
        config = dict(family="Ricker_HO_LV", self=damping, dt=dt, noise=noise)
        source, future = lv_data(config)
        views = {}
        with threadpool_limits(limits=1):
            for view, blocks in (("increment", (future-source)[:, :, None]),
                                 ("transition", np.stack([source, future], axis=2))):
                views[view] = analyze_view(blocks, .04, seed=0)
        rows.append(dict(config=config, views=views,
                         source_sha256=digest(source), future_sha256=digest(future)))
        (OUT/"lv_screen.json").write_text(json.dumps(dict(stage="selection_seed_0_no_SPT",
                                                         conditions=rows), indent=2))
        print(config, {v: {k: r["original_partition_ari"] for k, r in a["methods"].items()}
                       for v, a in views.items()}, flush=True)


def screen_ctmc():
    from threadpoolctl import threadpool_limits
    rows = []
    for beta, triad, tau in itertools.product((.5, 1., 2.), (.5, 1., 2.),
                                             (.01, .05, .1, .2, .5, 1., 2.)):
        check_budget()
        config = dict(family="glauber_ctmc", beta=beta, triad=triad, tau=tau, cross=.04)
        with threadpool_limits(limits=1):
            kernel = spin_kernel(**config)
        views = {}
        for view in ("transition", "increment", "future"):
            blocks, card = observations(view)
            info = DiscreteInformation(blocks, kernel.ravel()/64, card)
            views[view] = {norm: mic(info, norm) for norm in ("joint_entropy", "dimension")}
        moment = float(kernel.mean(axis=0) @ np.prod(SPINS[:, 3:], axis=1))
        rows.append(dict(config=config, views=views, future_triad_moment=moment))
    (OUT/"ctmc_screen.json").write_text(json.dumps(dict(stage="population_MIC_screen_no_SPT",
                                                        conditions=rows), indent=2))
    fails = [r for r in rows if not r["views"]["transition"]["joint_entropy"]["exact_original_two_groups"]]
    print("CTMC MIC transition failures", len(fails), "/", len(rows))
    for row in sorted(fails, key=lambda r:-abs(r["future_triad_moment"]))[:12]:
        print(row["config"], "moment", row["future_triad_moment"],
              row["views"]["transition"]["joint_entropy"]["groups"], flush=True)


TERMS = [a for a in SUBSETS if len(a) <= 3]
MONOMIALS = np.array([np.prod(SPINS[:, np.array(a)-1], axis=1) for a in TERMS])
MEMBERSHIP = np.array([[i in a for i in range(1, 7)] for a in TERMS], dtype=float)
FLIPPED = np.arange(64)[:, None] ^ (1 << np.arange(6))


def generic_generator(theta):
    """All 6 fields, all 15 pairs and all 20 triplets; no topology restriction."""
    delta_half = np.einsum("a,ax,ai->xi", theta, MONOMIALS, MEMBERSHIP)
    rates = expit(-2*delta_half)
    generator = np.zeros((64, 64))
    generator[np.arange(64)[:, None], FLIPPED] = rates
    generator[np.diag_indices(64)] = -rates.sum(axis=1)
    return generator, rates


def fit_objective(theta, counts, tau, penalty):
    generator, rates = generic_generator(theta)
    kernel = expm(tau*generator)
    if kernel.min() <= 0:
        raise ArithmeticError("Nonpositive fitted transition probability")
    value = -float(np.sum(counts*np.log(kernel))) + .5*penalty*float(theta@theta)
    # Adjoint Frechet derivative; evaluate all 41 coefficient gradients at once.
    adjoint = tau*expm_frechet((tau*generator).T, -counts/kernel, compute_expm=False)
    edge_adjoint = adjoint[np.arange(64)[:, None], FLIPPED] - adjoint.diagonal()[:, None]
    derivative = -2*rates*(1-rates)*edge_adjoint
    gradient = np.einsum("xi,ax,ai->a", derivative, MONOMIALS, MEMBERSHIP) + penalty*theta
    return value, gradient


def draw_joint(kernel, seed, count=4800, source_prob=None):
    rng = np.random.default_rng(1000+seed)
    initial = (rng.integers(0, 64, count) if source_prob is None
               else rng.choice(64, count, p=source_prob))
    uniforms = rng.random(count)
    future = (uniforms[:, None] > np.cumsum(kernel[initial], axis=1)).sum(axis=1)
    counts = np.bincount(64*initial+future, minlength=4096).reshape(64, 64)
    return initial, future, counts


def fit_ctmc(counts, tau=2., penalty=10., initial=None):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        fit = minimize(fit_objective, np.zeros(len(TERMS)) if initial is None else initial,
                       args=(counts, tau, penalty), jac=True, method="L-BFGS-B",
                       bounds=[(-4., 4.)]*len(TERMS),
                       options=dict(maxiter=150, ftol=1e-10, gtol=1e-5))
        kernel = expm(tau*generic_generator(fit.x)[0])
    if not fit.success:
        raise RuntimeError(f"CTMC fit failed: {fit.message}")
    return kernel, dict(success=bool(fit.success), message=str(fit.message),
                       iterations=int(fit.nit), objective=float(fit.fun), penalty=penalty,
                       coefficients={"".join(map(str, a)): float(v) for a, v in zip(TERMS, fit.x)},
                       maximum_gradient=float(np.abs(fit.jac).max()))


def discrete_methods(prob, view):
    """All algorithms share one joint distribution, observed as the same node blocks.

    Weighted profiles give exactly the same Euclidean geometry as repeating
    empirical observations. dCor is summed over the small categorical supports;
    no subsampling or continuous jitter is used.
    """
    from threadpoolctl import threadpool_limits
    codes, card = observations(view)
    info = DiscreteInformation(codes, prob, card)
    if view == "transition":
        raw = np.stack([X, Y], axis=2).astype(float)
        values = np.stack([2*(np.arange(4) % 2)-1, 2*(np.arange(4)//2)-1], axis=1)
    elif view == "increment":
        raw = (Y-X)[:, :, None].astype(float)
        values = np.arange(-2, 3, 2)[:, None]
    else:
        raw = Y[:, :, None].astype(float)
        values = np.array([[-1.], [1.]])
    means = np.einsum("n,nid->id", prob, raw)
    std = np.sqrt(np.einsum("n,nid->id", prob, (raw-means)**2))
    blocks = (raw-means)/std
    profiles = (blocks*np.sqrt(prob)[:, None, None]).transpose(1, 0, 2).reshape(6, -1)
    correlation = profiles@profiles.T/raw.shape[2]
    np.fill_diagonal(correlation, 1.)
    centered = []
    variance = []
    for i in range(6):
        pi = np.bincount(codes[:, i], weights=prob, minlength=card)
        di = cdist((values-means[i])/std[i], (values-means[i])/std[i])
        average = di@pi
        ai = di-average[:, None]-average[None, :]+pi@average
        centered.append(ai)
        variance.append(float(np.sum(np.outer(pi, pi)*ai**2)))
    dcor = np.eye(6)
    mi = np.zeros((6, 6))
    dcor_roundoff_count = 0
    for i, j in itertools.combinations(range(6), 2):
        pij = np.bincount(card*codes[:, i]+codes[:, j], weights=prob,
                          minlength=card*card).reshape(card, card)
        covariance = float(np.einsum("ab,cd,ac,bd", pij, pij, centered[i], centered[j]))
        squared = covariance/math.sqrt(variance[i]*variance[j])
        if squared < -1e-12 or squared > 1+1e-12:
            raise ArithmeticError("dCor numerical range violation")
        dcor_roundoff_count += int(squared < 0 or squared > 1)
        dcor[i, j] = dcor[j, i] = math.sqrt(min(1., max(0., squared)))
        mi[i, j] = mi[j, i] = info.mi([i+1], [j+1])
    with threadpool_limits(limits=1):
        km = KMeans(n_clusters=2, n_init=50, random_state=0).fit_predict(profiles)
        sp = SpectralClustering(n_clusters=2, affinity="precomputed", assign_labels="cluster_qr",
                                 random_state=0).fit_predict(np.abs(correlation))
        methods = dict(kmeans=dict(kind="partition", groups=groups_from_labels(km)),
                       spectral=dict(kind="partition", groups=groups_from_labels(sp)),
                       ward=hierarchy_result(linkage(profiles, method="ward", optimal_ordering=True)))
    for name, distance in (("correlation_hac", 1-np.abs(correlation)),
                            ("dcor_hac", 1-dcor),
                            ("mi_hac", 1-mi/max(float(np.abs(mi).max()), 1e-12))):
        np.fill_diagonal(distance, 0.)
        if distance.min() < -1e-12:
            raise ArithmeticError("Negative clustering distance")
        methods[name] = hierarchy_result(linkage(squareform(distance, checks=False),
                                                 method="average", optimal_ordering=True))
    methods = {name: annotate(result, .04) for name, result in methods.items()}
    methods["mic"] = mic(info)
    methods["mic_dimension"] = mic(info, "dimension")
    return dict(methods=methods, matrices=dict(correlation=correlation.tolist(),
                 distance_correlation=dcor.tolist(), pair_mi_bits=mi.tolist()),
                 dcor_roundoff_count=dcor_roundoff_count)


def confirm_ctmc():
    check_budget()
    selected = json.loads((OUT/"selected_ctmc_population.json").read_text())
    config = selected["config"]
    kernel = spin_kernel(**config)
    population = {view: discrete_methods(kernel.ravel()/64, view)
                  for view in ("transition", "increment", "future")}
    rows = []
    for seed in range(11):
        check_budget()
        initial, future, counts = draw_joint(kernel, seed)
        learned, fit = fit_ctmc(counts, tau=config["tau"], penalty=10.)
        empirical = {view: discrete_methods(counts.ravel()/counts.sum(), view)
                     for view in ("transition", "increment", "future")}
        model = {view: discrete_methods(learned.ravel()/64, view)
                 for view in ("transition", "increment", "future")}
        fitted_spt = spt(learned)
        # Independent predictive check is diagnostic only; no model selection.
        _, _, test_counts = draw_joint(kernel, 20000+seed)
        heldout_nll = float(-np.sum(test_counts*np.log(learned))/test_counts.sum())
        oracle_nll = float(-np.sum(test_counts*np.log(kernel))/test_counts.sum())
        rows.append(dict(seed=seed, role="selection" if seed == 0 else "confirmation",
                         source_sha256=digest(initial), future_sha256=digest(future),
                         empirical=empirical, model=model, fitted_spt=fitted_spt,
                         fit=fit, heldout_nll_nats=heldout_nll, oracle_heldout_nll_nats=oracle_nll))
        payload = dict(config=config, sample_count=4800, confirmation_seeds=list(range(1, 11)),
                       source_intervention="independent uniform +/-1; q(X)=1/64",
                       fit_model="generic heat-bath CTMC, all 6 fields, 15 pairs and 20 triplets",
                       fit_penalty=10., fit_penalty_selection="fixed before confirmation; no tuning",
                       population_spt=selected["spt"], population=population, conditions=rows)
        (OUT/"ctmc_confirmation.json").write_text(json.dumps(payload, indent=2))
        print("seed", seed, "SPT", fitted_spt["groups"],
              "raw MIC", empirical["transition"]["methods"]["mic"]["groups"],
              "model MIC", model["transition"]["methods"]["mic"]["groups"],
              "heldout excess NLL", heldout_nll-oracle_nll, flush=True)


def true_theta(config):
    theta = np.zeros(len(TERMS))
    for (i, j), weight in WEIGHTS.items():
        theta[TERMS.index((i+1, j+1))] = config["beta"]*weight
    theta[TERMS.index((4, 5, 6))] = config["beta"]*config["triad"]
    for i in range(1, 4):
        for j in range(4, 7):
            theta[TERMS.index((i, j))] = config["beta"]*config["cross"]/3
    return theta


def stationary_confirm():
    """Frozen-parameter sensitivity: short trajectories from natural equilibrium."""
    selected = json.loads((OUT/"selected_ctmc_population.json").read_text())
    config = selected["config"]
    kernel = spin_kernel(**config)
    theta = true_theta(config)
    logp = theta@MONOMIALS
    stationary = np.exp(logp-logsumexp(logp))
    balance = np.max(np.abs(stationary[:, None]*kernel-stationary[None, :]*kernel.T))
    assert balance < 1e-12
    observed_population = {view: discrete_methods((stationary[:, None]*kernel).ravel(), view)
                           for view in ("transition", "increment", "future")}
    rows = []
    for seed in range(1001, 1011):
        check_budget()
        initial, future, counts = draw_joint(kernel, seed, source_prob=stationary)
        learned, fit = fit_ctmc(counts, tau=config["tau"], penalty=10.)
        empirical = {view: discrete_methods(counts.ravel()/counts.sum(), view)
                     for view in ("transition", "increment", "future")}
        # Share the same reconstructed interventional joint with all baselines.
        model = {view: discrete_methods(learned.ravel()/64, view)
                 for view in ("transition", "increment", "future")}
        fitted_spt = spt(learned)
        _, _, test_counts = draw_joint(kernel, 20000+seed, source_prob=stationary)
        rows.append(dict(seed=seed, source_sha256=digest(initial), future_sha256=digest(future),
                         empirical=empirical, model=model, fitted_spt=fitted_spt, fit=fit,
                         heldout_nll_nats=float(-np.sum(test_counts*np.log(learned))/test_counts.sum()),
                         oracle_heldout_nll_nats=float(-np.sum(test_counts*np.log(kernel))/test_counts.sum())))
        payload = dict(config=config, sample_count=4800, confirmation_seeds=list(range(1001, 1011)),
                       source_sampling="independent initial states from the exact stationary Gibbs distribution",
                       intervention="SPT and reconstructed baselines: independent uniform source q=1/64",
                       true_stationary_probabilities=stationary.tolist(), detailed_balance_error=float(balance),
                       observed_population=observed_population, conditions=rows)
        (OUT/"ctmc_stationary_confirmation.json").write_text(json.dumps(payload, indent=2))
        print("stationary seed", seed, "SPT", fitted_spt["groups"],
              "raw MIC", empirical["transition"]["methods"]["mic"]["groups"],
              "intervention MIC", model["transition"]["methods"]["mic"]["groups"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["screen", "lv_screen", "ctmc_screen", "confirm", "stationary_confirm"])
    args = parser.parse_args()
    if args.stage == "screen":
        screen()
    elif args.stage == "lv_screen":
        screen_lv()
    elif args.stage == "ctmc_screen":
        screen_ctmc()
    elif args.stage == "confirm":
        confirm_ctmc()
    elif args.stage == "stationary_confirm":
        stationary_confirm()
