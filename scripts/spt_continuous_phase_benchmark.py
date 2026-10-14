"""Oracle continuous phase benchmark with pair-matched counterfactual hierarchies.

Nine source blocks of nine circular coordinates, iid uniform intervention,
complete future state, horizon one.  The known positive conditional density
admits an exact triangular CDF transport.  Periodic quadrature evaluates its
one-dimensional marginals instead of fitting an unnecessary density model.
Algorithms receive only information queries/matrices, never edge membership.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import i0e, i1e, ndtri
from scipy.stats import vonmises

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt import (SPTConfig, all_nonempty_subsets, build_spt_from_ei_table,
                         flatten_nodes, nontrivial_bipartitions)
from scripts.spt_continuous_hierarchy_pilot import audit_table
from scripts.spt_heterogeneous_hierarchy_case import SOURCES, quality
from scripts.spt_multilevel_parity_case import graph_tree

TAU = 1.0
TWO_PI = 2*np.pi
TOLERANCE_NATS = 1e-8
GAIN = 1.0
KAPPA = 2.0
LOCAL_KAPPA = np.array([1.2, 1.5, 1.8, 1.3, 2.0, 1.6, 1.4, 1.9, 1.1])
OMEGA = np.array([.13, -.21, .08, .34, -.17, .27, -.09, .19, -.31])
HARMONICS = np.array([1, 2, 1, 2, 2, 1, 2, 1, 1])
METHODS = ("spt", "block_phi_r", "pair_phi_r_hac", "pair_xi_hac", "future_mi_hac")


def log_i0(kappa):
    return np.log(i0e(kappa)) + np.abs(kappa)


def noise_entropy(kappa):
    return np.log(TWO_PI)+log_i0(kappa)-kappa*i1e(kappa)/i0e(kappa)


@lru_cache(maxsize=None)
def channel_integral(grid=512, gain=GAIN, kappa=KAPPA):
    angles = TWO_PI*np.arange(grid)/grid
    means = gain*np.sin(angles)
    density = np.exp(kappa*np.cos(angles[:, None]-means[None, :])-log_i0(kappa))/TWO_PI
    marginal = density.mean(axis=1)
    dz = TWO_PI/grid
    entropy = float(-dz*np.sum(marginal*np.log(marginal)))
    h_noise = float(noise_entropy(kappa))
    direct_mi = float(dz*np.mean(np.sum(density*(np.log(density)-np.log(marginal[:, None])), axis=0)))
    return {"grid": grid, "gain": gain, "kappa": kappa,
            "weight_nats": entropy-h_noise, "direct_mi_nats": direct_mi,
            "future_pair_mi_nats": float(np.log(TWO_PI)-entropy),
            "entropy_output_nats": entropy, "entropy_noise_nats": h_noise,
            "normalization_error": float(abs(dz*marginal.sum()-1)),
            "angles": angles, "marginal": marginal}


def counterfactuals():
    a_bottom = ((1,2,3), (4,5), (6,7), (8,9))
    architectures = {
        "A": [(*a_bottom, (1,2,3,4,5), (6,7,8,9), SOURCES),
              (*a_bottom, (1,2,3,6,7), (4,5,8,9), SOURCES),
              (*a_bottom, (1,2,3,8,9), (4,5,6,7), SOURCES)],
        "B": [],
        "C": [((1,2,3), (4,5,6), (7,8,9), (1,2,3,4,5,6), SOURCES),
              ((1,4,7), (2,5,8), (3,6,9), (1,2,4,5,7,8), SOURCES),
              ((1,5,9), (2,6,7), (3,4,8), (1,2,5,6,7,9), SOURCES)],
    }
    # All 3! x 2! cascade orders; fixed bottom dyads and highest branches.
    for left_order in itertools.permutations((3,4,5)):
        for right_order in itertools.permutations((8,9)):
            left = [tuple(sorted((1,2,*left_order[:k]))) for k in (0,1,2,3)]
            right = [tuple(sorted((6,7,*right_order[:k]))) for k in (0,1,2)]
            architectures["B"].append(tuple((*left, *right, SOURCES)))
    return architectures


class PhaseKernel:
    def __init__(self, supports, kappa=KAPPA):
        self.supports = tuple(tuple(e) for e in supports)
        if len(self.supports) > 8 or any(len(e)<2 for e in self.supports):
            raise ValueError("Expected at most eight non-singleton channels")
        self.kappa = kappa
        self.shifts = np.arange(len(self.supports))*.17

    def edge_mean(self, source, slot, edge):
        idx = np.asarray(edge)-1
        return GAIN*np.sin(np.sum(source[:, idx, slot+1]*HARMONICS[idx], axis=1)+self.shifts[slot])

    def step(self, source, rng):
        source = np.asarray(source)
        if source.ndim != 3 or source.shape[1:] != (9,9):
            raise ValueError("Expected full state with shape (samples,9,9)")
        target = rng.uniform(0, TWO_PI, source.shape)
        target[:, :, 0] = (source[:, :, 0]*HARMONICS+OMEGA*TAU
                            + rng.vonmises(0, LOCAL_KAPPA, (len(source),9))) % TWO_PI
        for slot, edge in enumerate(self.supports):
            idx = np.asarray(edge)-1
            total = self.edge_mean(source, slot, edge)+rng.vonmises(0, self.kappa, len(source))
            others = np.sum(target[:, idx[:-1], slot+1]-OMEGA[idx[:-1]]*TAU, axis=1)
            target[:, idx[-1], slot+1] = (total-others+OMEGA[idx[-1]]*TAU) % TWO_PI
        return target

    def log_conditional(self, source, target):
        value = np.full(len(source), -81*np.log(TWO_PI))
        local = target[:, :, 0]-source[:, :, 0]*HARMONICS-OMEGA*TAU
        value += np.sum(LOCAL_KAPPA*np.cos(local)-log_i0(LOCAL_KAPPA), axis=1)
        for slot, edge in enumerate(self.supports):
            idx = np.asarray(edge)-1
            phase = np.sum(target[:, idx, slot+1]-OMEGA[idx]*TAU, axis=1)
            value += self.kappa*np.cos(phase-self.edge_mean(source, slot, edge))-log_i0(self.kappa)
        return value

    def conditional_tm(self, source, target):
        # Ordering: nine local phases, then each slot's node phases.  The last
        # member of a slot has a von Mises conditional given earlier members.
        u = target/TWO_PI
        u = u.copy()
        local_loc = source[:, :, 0]*HARMONICS+OMEGA*TAU
        u[:, :, 0] = (vonmises.cdf(target[:, :, 0], LOCAL_KAPPA, loc=local_loc)
                       - vonmises.cdf(0, LOCAL_KAPPA, loc=local_loc))
        for slot, edge in enumerate(self.supports):
            idx = np.asarray(edge)-1
            others = np.sum(target[:, idx[:-1], slot+1]-OMEGA[idx[:-1]]*TAU, axis=1)
            loc = self.edge_mean(source, slot, edge)-others+OMEGA[idx[-1]]*TAU
            last = target[:, idx[-1], slot+1]
            u[:, idx[-1], slot+1] = (vonmises.cdf(last, self.kappa, loc=loc)
                                      - vonmises.cdf(0, self.kappa, loc=loc))
        if np.any((u <= 0) | (u >= 1)):
            raise ArithmeticError("TM evaluation reached a support boundary")
        return ndtri(u).transpose(0,2,1).reshape(len(source),81)


class InformationOracle:
    def __init__(self, kernel, grid=512):
        self._kernel = kernel
        self._channel = channel_integral(grid, GAIN, kernel.kappa)
        self._local = np.log(TWO_PI)-noise_entropy(LOCAL_KAPPA)
        self.query_count = 0

    def info(self, source, target=SOURCES):
        self.query_count += 1
        a, b = set(source), set(target)
        return math.fsum([*(float(self._local[i-1]) for i in a & b),
                          *(self._channel["weight_nats"] for e in self._kernel.supports
                            if set(e) <= a and set(e) <= b)])

    def future_mi(self, a, b):
        return math.fsum(self._channel["future_pair_mi_nats"] for e in self._kernel.supports
                         if set(e) == {a,b})


def block_tree(info, sources, *, normalize=False):
    clades, candidates = [], []
    def visit(a):
        if len(a)==1:
            return ()
        values = []
        for l, r in nontrivial_bipartitions(a):
            redundancy = min(info(u,v) for u in (l,r) for v in (l,r))
            value = info(a,a)-info(l,l)-info(r,r)+redundancy
            candidates.append(value)
            objective = value/min(len(l),len(r)) if normalize else value
            values.append((objective,l,r))
        _, l, r = min(values)
        for child in (l,r):
            if len(child)>1:
                clades.append(child)
            visit(child)
        return l,r
    root = visit(sources)
    return root, clades, candidates


def evaluate(info, future_mi, sources=SOURCES):
    # Source names and these two query callables are the only algorithm inputs.
    counts = {}
    original_info = info
    stage = "shared_ei_table"
    def info(a, b=None):
        counts[stage] = counts.get(stage, 0)+1
        return original_info(a,sources if b is None else b)
    ei = {s: info(s) for s in all_nonempty_subsets(sources)}
    audit = audit_table(ei, TOLERANCE_NATS)
    if audit["significant_violation_count"]:
        raise ArithmeticError(f"Syn violation: minimum={audit['minimum_syn_nats']} nats; "
                              f"threshold={-TOLERANCE_NATS}; affected_count={audit['significant_violation_count']}")
    tree = build_spt_from_ei_table(sources, ei, config=SPTConfig(syn_tolerance=TOLERANCE_NATS))
    nodes = flatten_nodes(tree.root)
    outputs = {"spt": {"root_split": tuple(c.sources for c in tree.root.children),
        "clades": [n.sources for n in nodes if 1<len(n.sources)<len(sources)], "status": "resolved"}}
    stage = "block_phi_r"
    root, clades, values = block_tree(info, sources)
    if min(values)<-TOLERANCE_NATS:
        raise ArithmeticError(f"Block PhiR violation: minimum={min(values)}; threshold={-TOLERANCE_NATS}; "
                              f"affected_count={sum(v < -TOLERANCE_NATS for v in values)}")
    outputs["block_phi_r"] = {"root_split": root, "clades": clades, "status": "resolved"}
    matrices = {m: np.zeros((len(sources),len(sources))) for m in METHODS[2:]}
    positions = {source:j for j,source in enumerate(sources)}
    stage = "pair_phi_r_hac"
    for i,j in itertools.combinations(sources,2):
        ii,jj = (i,), (j,)
        redundancy = min(info(u,v) for u in (ii,jj) for v in (ii,jj))
        matrices["pair_phi_r_hac"][positions[i],positions[j]] = info((i,j),(i,j))-info(ii,ii)-info(jj,jj)+redundancy
        matrices["pair_xi_hac"][positions[i],positions[j]] = ei[tuple(x for x in sources if x in (i,j))]-ei[ii]-ei[jj]
        matrices["future_mi_hac"][positions[i],positions[j]] = future_mi(i,j)
    for method, matrix in matrices.items():
        matrix += matrix.T
        if matrix.max() <= TOLERANCE_NATS:
            outputs[method] = {"status": "no_structure_information", "root_split": (), "clades": ()}
        else:
            root, clades = graph_tree(matrix)
            root = tuple(tuple(sources[i-1] for i in child) for child in root)
            clades = [tuple(sources[i-1] for i in child) for child in clades]
            outputs[method] = {"status": "tie_sensitive", "root_split": root, "clades": clades}
    audit.update({"closure_error_nats": tree.closure_error,
                  "block_phi_minimum_nats": min(values),
                  "block_phi_tolerance_negative_count": sum(-TOLERANCE_NATS <= x < 0 for x in values)})
    if abs(tree.closure_error)>TOLERANCE_NATS:
        raise ArithmeticError(f"Closure failed: {tree.closure_error}")
    return {"methods": outputs, "matrices_nats": {k:v.tolist() for k,v in matrices.items()},
            "information_query_counts": {**counts, "future_mi_hac":math.comb(len(sources),2),
                "pair_xi_hac": f"{math.comb(len(sources),2)} pairs + {len(sources)} singleton values reused from shared table"},
            "audit": audit, "xi_nats": tree.root.xi_value, "root_syn_nats": tree.root.syn_value,
            "nodes": [{"sources": n.sources, "children": [c.sources for c in n.children],
                       "depth": n.depth, "xi_nats": n.xi_value, "syn_nats": n.syn_value} for n in nodes]}


def score(output, supports):
    primitive = tuple(e for e in supports if not any(set(g)<set(e) for g in supports))
    upper = {frozenset(e) for e in supports if len(e)<9 and e not in primitive}
    if output["status"] == "no_structure_information":
        return {**output, "clade_f1": 0., "clade_precision": 0., "clade_recall": 0.,
                "upper_support_recall": 0., "correct_root": False}
    result = quality(output["root_split"], output["clades"], supports, primitive)
    found = {frozenset(c) for c in output["clades"]}
    return {**result, "status": output["status"],
            "upper_support_recall": len(found & upper)/len(upper)}


def validate_kernel(sample_count=65536, seed=20261013):
    checks = []
    integrals = [channel_integral(n) for n in (128,256,512,1024)]
    weights = [i["weight_nats"] for i in integrals]
    if max(weights)-min(weights)>TOLERANCE_NATS:
        raise ArithmeticError("Periodic quadrature did not converge")
    if max(abs(i["direct_mi_nats"]-i["weight_nats"]) for i in integrals)>TOLERANCE_NATS:
        raise ArithmeticError("Direct density MI and entropy MI disagree")
    # Independently integrate actual m-body kernels, including a missing readout.
    grid = TWO_PI*np.arange(64)/64
    for m in (2,3):
        mesh = np.meshgrid(*([grid]*m), indexing="ij")
        kernel = np.exp(KAPPA*np.cos(sum(mesh)-.71)-log_i0(KAPPA))/TWO_PI**m
        norm_error = abs(float(kernel.mean()*TWO_PI**m)-1)
        marginal_error = float(np.max(abs(kernel.mean(axis=-1)*TWO_PI-1/TWO_PI**(m-1))))
        if norm_error>TOLERANCE_NATS or marginal_error>TOLERANCE_NATS:
            raise ArithmeticError("Kernel normalization/pair-blind integration failed")
        checks.append({"readout_order": m, "normalization_error": norm_error,
                       "missing_readout_uniform_density_max_error": marginal_error})
    # Integrate a missing original source phase, rather than using Eq.(7).
    z = TWO_PI*np.arange(128)/128
    phi = TWO_PI*np.arange(256)/256
    base = np.exp(KAPPA*np.cos(z[:,None]-GAIN*np.sin(phi)[None,:])-log_i0(KAPPA)).mean(1)/TWO_PI
    for harmonic in (1,2):
        missing = np.exp(KAPPA*np.cos(z[:,None]-GAIN*np.sin(harmonic*phi+.83)[None,:])
                         -log_i0(KAPPA)).mean(1)/TWO_PI
        error = float(np.max(abs(missing-base)))
        if error>TOLERANCE_NATS:
            raise ArithmeticError("Missing original source phase marginal depends on retained phases")
        checks.append({"missing_source_harmonic":harmonic,"marginal_density_max_error":error})
    rng = np.random.default_rng(seed)
    source = rng.uniform(0,TWO_PI,(sample_count,9,9))
    for case, variants in counterfactuals().items():
        kernel = PhaseKernel(variants[0])
        target = kernel.step(source,rng)
        integral = channel_integral(1024)
        local_samples = (LOCAL_KAPPA*np.cos(target[:,:,0]-source[:,:,0]*HARMONICS-OMEGA*TAU)
                         -log_i0(LOCAL_KAPPA))
        edge_samples, sufficient_error = [], 0.
        log_marginal = np.full(sample_count,-81*np.log(TWO_PI))
        for slot,e in enumerate(kernel.supports):
            idx = np.asarray(e)-1
            total = np.sum(target[:,idx,slot+1]-OMEGA[idx]*TAU,axis=1) % TWO_PI
            pz = np.interp(total, np.r_[integral["angles"],TWO_PI],
                           np.r_[integral["marginal"],integral["marginal"][0]])
            contribution = (KAPPA*np.cos(total-kernel.edge_mean(source,slot,e))-log_i0(KAPPA)
                            - np.log(TWO_PI*pz))
            edge_samples.append(contribution)
            log_marginal += np.log(TWO_PI*pz)
        edges = np.stack(edge_samples,-1)
        full_ratio = kernel.log_conditional(source,target)-log_marginal
        reduced_ratio = local_samples.sum(1)+edges.sum(1)
        sufficient_error = float(np.max(abs(full_ratio-reduced_ratio)))
        if sufficient_error>1e-10:
            raise ArithmeticError("Complete target and sufficient readout density ratios disagree")
        oracle = InformationOracle(kernel)
        maximum_error, maximum_standardized = 0., 0.
        for subset in all_nonempty_subsets(SOURCES):
            selected = [j for j,e in enumerate(kernel.supports) if set(e)<=set(subset)]
            rows = local_samples[:,np.asarray(subset)-1].sum(1)+edges[:,selected].sum(1)
            error = abs(float(rows.mean())-oracle.info(subset))
            sem = rows.std(ddof=1)/np.sqrt(sample_count)
            maximum_error = max(maximum_error,error)
            maximum_standardized = max(maximum_standardized,error/sem if sem else 0.)
        if maximum_standardized>6:
            raise ArithmeticError("Simulation-derived EI inconsistent with oracle")
        # Exact conditional CDF transport checked on genuine kernel draws.
        tm = kernel.conditional_tm(source[:2048],target[:2048])
        if np.max(abs(tm.mean(0)))>.15 or np.max(abs(tm.var(0)-1))>.2:
            raise ArithmeticError("Conditional TM reference distribution check failed")
        # A finite-difference positive triangular diagonal independently checks
        # the implemented CDF map against the 81-dimensional kernel density.
        small_x, small_y = source[:4], target[:4]
        t = kernel.conditional_tm(small_x,small_y)
        diagonal = np.empty_like(t)
        h = 1e-6
        for channel in range(9):
            for node in range(9):
                plus, minus = small_y.copy(), small_y.copy()
                plus[:,node,channel] += h
                minus[:,node,channel] -= h
                coordinate = channel*9+node
                diagonal[:,coordinate] = (kernel.conditional_tm(small_x,plus)[:,coordinate]
                                           -kernel.conditional_tm(small_x,minus)[:,coordinate])/(2*h)
        if np.any(diagonal<=0):
            raise ArithmeticError("Conditional TM has a nonpositive diagonal")
        tm_log_density = -.5*np.sum(t*t+np.log(TWO_PI),axis=1)+np.log(diagonal).sum(1)
        tm_density_error = float(np.max(abs(tm_log_density-kernel.log_conditional(small_x,small_y))))
        if tm_density_error>1e-4:
            raise ArithmeticError("Conditional TM change-of-variables density disagrees with kernel")
        checks.append({"case":case,"sample_count":sample_count,"iid_seed":seed,
                       "complete_target_sufficient_ratio_max_error":sufficient_error,
                       "all_511_subset_mc_ei_max_error_nats":maximum_error,
                       "all_511_subset_mc_max_error_over_iid_sem":maximum_standardized,
                       "conditional_tm_density_max_error_nats":tm_density_error,
                       "conditional_tm_minimum_diagonal":float(diagonal.min()),
                       "conditional_tm_reference_mean_max_abs":float(np.max(abs(tm.mean(0)))),
                       "conditional_tm_reference_variance_max_error":float(np.max(abs(tm.var(0)-1)))})
    return {"grid_weights": [{k:v for k,v in item.items() if k not in ("angles","marginal")}
                             for item in integrals], "checks":checks}


def run(permutations=20, grid=512, validation_samples=65536):
    start = time.monotonic()
    validation = validate_kernel(validation_samples)
    rng = np.random.default_rng(20261013)
    # Identity plus nineteen random bijections; permutations measure label/tie
    # sensitivity, never independent physical systems.
    schedules = [SOURCES, *(tuple(map(int,rng.permutation(SOURCES))) for _ in range(permutations-1))]
    result = {"protocol": {"nodes":9,"coordinates_per_node":9,"source_coordinates":81,
        "target_coordinates":81,"intervention":"iid Uniform(0,2*pi)^81","horizon_steps":1,
        "gain":GAIN,"kappa_by_order":KAPPA,"local_kappa":LOCAL_KAPPA.tolist(),
        "omega":OMEGA.tolist(),"harmonics":HARMONICS.tolist(),"grid":grid,
        "syn_tolerance_nats":TOLERANCE_NATS,"clipping":False,
        "estimator":"exact circular conditional CDF triangular TM density with periodic marginal quadrature",
        "oracle_limit":"known model; no finite-data learning advantage or single-phase Kuramoto claim",
        "algorithm_inputs":"info(source,target) and future_mi(i,j); edge membership/active slots/variant IDs excluded",
        "resource_rule":"same exact joint and grid; query access differs by method; query counts recorded",
        "methods":METHODS,"spt_objective":"maximum child Xi, raw residual, exact splits, complete to singletons",
        "block_phi_r":"recursive minimum unnormalized MMI PhiR, target changes to current block",
        "scoring":"existing nonroot support F1; proper minimal-support refinements ignored; upper recall excludes minimal/root supports",
        "permutations":schedules,"replication":"common label/tie permutations, no physical confidence intervals",
        "zero_affinity":"abstain; zero recovery scores denote no reported support, never an invented measured tree",
        "null":"all interaction kappa=0, local memory unchanged; Xi=0; arbitrary binary refinements are not evidence",
        "manuscript":{"parent":"P6UJCVG8","main":"DXGC7JEA","supplement":"MWIWKSVG",
            "checked":"2026-10-13","version":"no explicit manuscript date/version; current sole main/SI pair",
            "locations":"Methods pp15-17 Eqs(4)-(12); SI S1.2 pp3-4, S3.2 p7, S5 pp8-9",
            "difference":"new circular block-state benchmark and known nonlinear TM, not SI affine-feature reproduction"}},
        "validation":validation,"records":[]}
    for case, variants in counterfactuals().items():
        for variant, supports in enumerate(variants):
            oracle = InformationOracle(PhaseKernel(supports),grid)
            for permutation_id, permutation in enumerate(schedules):
                to_original = dict(zip(SOURCES,permutation))
                to_label = {v:k for k,v in to_original.items()}
                def info(a,b=SOURCES):
                    return oracle.info(tuple(to_original[i] for i in a), tuple(to_original[i] for i in b))
                def future(a,b):
                    return oracle.future_mi(to_original[a],to_original[b])
                queries_before = oracle.query_count
                record = evaluate(info,future)
                # Scoring is performed after all algorithms finish.
                labeled = tuple(tuple(sorted(to_label[i] for i in e)) for e in supports)
                record.update({"case":case,"variant":variant,"permutation_id":permutation_id,
                    "hyperedges":labeled,"query_count":oracle.query_count-queries_before})
                record["methods"] = {m:score(v,labeled) for m,v in record["methods"].items()}
                result["records"].append(record)
            print(f"{case} variant {variant}: {permutations} labelings completed",flush=True)
    # Every family/permutation must have equal matrices in its counterfactuals.
    maximum_difference = {m:0. for m in METHODS[2:]}
    for case in counterfactuals():
        for p in range(permutations):
            matched = [r for r in result["records"] if r["case"]==case and r["permutation_id"]==p]
            for method in maximum_difference:
                base = np.asarray(matched[0]["matrices_nats"][method])
                difference = max(float(np.max(abs(np.asarray(r["matrices_nats"][method])-base))) for r in matched)
                maximum_difference[method] = max(maximum_difference[method],difference)
                if difference>TOLERANCE_NATS:
                    raise ArithmeticError("Counterfactual pair matrices differ")
    result["counterfactual_pair_matrix_max_difference_nats"] = maximum_difference
    null_oracle = InformationOracle(PhaseKernel(counterfactuals()["A"][0],kappa=0.),grid)
    null = evaluate(null_oracle.info,null_oracle.future_mi)
    if abs(null["xi_nats"])>TOLERANCE_NATS:
        raise ArithmeticError("No-interaction null has nonzero Xi")
    null["interpretation"] = "all methods have no between-node evidence; SPT/Block binary refinements arbitrary"
    result["null_control"] = null
    result["elapsed_seconds"] = time.monotonic()-start
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--permutations",type=int,default=20)
    parser.add_argument("--grid",type=int,default=512)
    parser.add_argument("--validation-samples",type=int,default=65536)
    args = parser.parse_args()
    if args.permutations<1:
        parser.error("At least one labeling is required")
    result = run(args.permutations,args.grid,args.validation_samples)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({"records":len(result["records"]),"seconds":result["elapsed_seconds"],
                      "pair_matrix_difference":result["counterfactual_pair_matrix_max_difference_nats"]}))


if __name__ == "__main__":
    main()
