"""Small oracle-TM pilot for genuinely continuous versions of A/B/C.

Nine scalar nodes, nonlinear stochastic coupled maps, horizon one.  The input
intervention is iid Uniform(-1, 1); the target is the complete nine-dimensional
future state, not a drift/increment or a thresholded readout.  All marginal
densities derive from one exact conditional triangular transport map.  Source
and noise evaluation draws are paired across cases; missing sources are
integrated using a second common scrambled Sobol rule.  This is an oracle
benchmark, not an estimated-dynamics/data-only validation of the manuscript's
affine feature-space estimator.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt import (SPTConfig, all_nonempty_subsets, build_spt_from_ei_table,
                         flatten_nodes, nontrivial_bipartitions)
from scripts.spt_heterogeneous_hierarchy_case import MODELS, SOURCES, quality
from scripts.spt_multilevel_parity_case import graph_tree

FULL = (1 << 9) - 1
# Declared before calculating results, in native continuous-information units.
SYN_TOLERANCE_NATS = 0.03
MEMORY = np.array([.16, .25, .19, .31, .22, .14, .28, .18, .24])
LOCAL = np.array([.09, .06, .12, .08, .10, .13, .07, .11, .05])
NOISE = np.array([.70, .78, .86, .74, .91, .81, .72, .88, .76])
RESPONSE = np.array([1.00, .88, 1.12, .93, 1.08, .84, 1.16, .96, 1.04])
PHASE = np.array([.00, .23, -.31, .47, -.18, .36, -.42, .14, -.27])
GAIN = {2: 1.6, 3: 1.6, 4: 1.3, 5: 1.3, 6: 1.0, 9: .35}


def mask(subset):
    return sum(1 << (int(i) - 1) for i in subset)


def columns(bits):
    return [i for i in range(9) if bits >> i & 1]


class ContinuousMap:
    def __init__(self, case):
        self.case = case
        self.supports = MODELS[case]["supports"]
        self.coefficients = np.zeros((len(self.supports), 9))
        for j, edge in enumerate(self.supports):
            idx = np.asarray(edge) - 1
            # Equal total squared response gain at a given interaction order;
            # the number of edges is a permitted architecture difference.
            self.coefficients[j, idx] = (GAIN[len(edge)] * RESPONSE[idx]
                                         / np.linalg.norm(RESPONSE[idx]))

    @staticmethod
    def local(values):
        return MEMORY * values + LOCAL * np.tanh(1.5 * values)

    @staticmethod
    def features(values):
        # Each component has mean 0 and variance 1 under the intervention.
        return np.sqrt(2.) * np.sin(np.pi * values + PHASE)

    def mean(self, values):
        features = self.features(values)
        products = np.stack([np.prod(features[..., np.asarray(e) - 1], axis=-1)
                             for e in self.supports], axis=-1)
        return self.local(values) + products @ self.coefficients

    def step(self, values, gaussian_noise):
        return self.mean(values) + NOISE * gaussian_noise

    def conditional_tm(self, source, target):
        """Exact positive-diagonal conditional Gaussian transport to N(0,I)."""
        return (target - self.mean(source)) / NOISE

    def log_prob(self, source, target):
        z = self.conditional_tm(source, target)
        return -.5 * np.sum(z*z + np.log(2*np.pi*NOISE**2), axis=-1)


class MarginalOracle:
    def __init__(self, model, count, particles, seed):
        for n in (count, particles):
            if n < 16 or n & (n-1):
                raise ValueError("Evaluation and particle counts must be powers of two >=16")
        self.model = model
        self.count, self.particles = count, particles
        draws = qmc.Sobol(18, scramble=True, seed=seed).random_base2(int(math.log2(count)))
        self.x = 2*draws[:, :9] - 1
        self.z = ndtri(draws[:, 9:])
        self.y = model.step(self.x, self.z)
        base = qmc.Sobol(9, scramble=True, seed=seed+101).random_base2(int(math.log2(particles)))
        self.base = 2*base - 1
        self.eval_features = model.features(self.x)
        self.base_features = model.features(self.base)
        self.local_eval = model.local(self.x)
        self.local_base = model.local(self.base)
        self.cache = {}
        self.integration_changes = {}

    def marginal(self, source_mask, target_mask=FULL):
        key = (source_mask, target_mask)
        if key in self.cache:
            return self.cache[key]
        idx = columns(target_mask)
        if source_mask == FULL:
            z = self.z[:, idx]
            result = -.5*np.sum(z*z + np.log(2*np.pi*NOISE[idx]**2), axis=1)
            self.integration_changes[key] = 0.
        else:
            kept = columns(source_mask)
            left_products, right_products = [], []
            for edge in self.model.supports:
                retained = [i-1 for i in edge if source_mask >> (i-1) & 1]
                missing = [i-1 for i in edge if not source_mask >> (i-1) & 1]
                left_products.append(np.prod(self.eval_features[:, retained], axis=1))
                right_products.append(np.prod(self.base_features[:, missing], axis=1))
            left_products = np.stack(left_products, axis=-1)
            right_products = np.stack(right_products, axis=-1)
            results, changes = [], []
            half = self.particles//2
            row_chunk = max(1, 131072//self.particles)
            coeff = self.model.coefficients[:, idx]
            constant = np.sum(np.log(2*np.pi*NOISE[idx]**2))
            for start in range(0, self.count, row_chunk):
                stop = min(self.count, start+row_chunk)
                products = left_products[start:stop, None, :] * right_products[None, :, :]
                means = products @ coeff
                for j, i in enumerate(idx):
                    means[..., j] += (self.local_eval[start:stop, i, None] if i in kept
                                      else self.local_base[None, :, i])
                residual = (self.y[start:stop, None, idx] - means)/NOISE[idx]
                ll = -.5*(np.sum(residual*residual, axis=-1) + constant)
                full = logsumexp(ll, axis=1) - np.log(self.particles)
                first = logsumexp(ll[:, :half], axis=1) - np.log(half)
                second = logsumexp(ll[:, half:], axis=1) - np.log(half)
                # Explicit first-order log-mixture jackknife, no EI/Syn projection.
                corrected = 2*full - .5*(first+second)
                results.append(corrected)
                changes.append(corrected-full)
            result = np.concatenate(results)
            self.integration_changes[key] = float(np.mean(np.concatenate(changes)))
        self.cache[key] = result
        return result

    def information_samples(self, source_mask, target_mask=FULL):
        return self.marginal(source_mask, target_mask) - self.marginal(0, target_mask)

    def info(self, source, target=SOURCES):
        return float(self.information_samples(mask(source), mask(target)).mean())


def audit_table(ei, tolerance=SYN_TOLERANCE_NATS):
    values = [(ei[s]-ei[l]-ei[r], s, l, r) for s in ei if len(s)>1
              for l, r in nontrivial_bipartitions(s)]
    smallest = min(values, key=lambda row: row[0])
    significant = sum(v < -tolerance for v, *_ in values)
    return {"tolerance_nats": tolerance, "partition_count": len(values),
            "minimum_syn_nats": smallest[0],
            "minimum_syn_split": [smallest[1], smallest[2], smallest[3]],
            "tolerance_negative_count": sum(-tolerance <= v < 0 for v, *_ in values),
            "significant_violation_count": significant,
            "clipping": False, "monotone_projection": False}


def score(model, root, clades):
    primitive = tuple(e for e in model.supports if not any(set(g)<set(e) for g in model.supports))
    return quality(root, clades, model.supports, primitive)


def analyze(case, count, particles, seed):
    start = time.monotonic()
    model = ContinuousMap(case)
    oracle = MarginalOracle(model, count, particles, seed)
    subsets = all_nonempty_subsets(SOURCES)
    information_samples = {s: oracle.information_samples(mask(s)) for s in subsets}
    ei = {s: float(v.mean()) for s, v in information_samples.items()}
    audit = audit_table(ei)
    record = {"case": case, "case_label": MODELS[case]["label"],
              "hyperedges": model.supports, "evaluation_count": count,
              "integration_particles": particles, "seed": seed,
              "ei_nats": {','.join(map(str,s)): v for s,v in ei.items()},
              "audit": audit, "status": "valid"}
    # A failing numerical oracle cannot be used to produce a hierarchy.
    if audit["significant_violation_count"]:
        record["status"] = "invalid_syn_estimate"
        record["error"] = (f"Syn nonnegativity failed: minimum={audit['minimum_syn_nats']:.8g} "
                           f"nats; threshold={-SYN_TOLERANCE_NATS}; "
                           f"count={audit['significant_violation_count']}; SPT not constructed")
    else:
        tree = build_spt_from_ei_table(SOURCES, ei,
                                     config=SPTConfig(syn_tolerance=SYN_TOLERANCE_NATS))
        nodes = flatten_nodes(tree.root)
        record["spt"] = score(model, tuple(c.sources for c in tree.root.children),
                              [n.sources for n in nodes if 1<len(n.sources)<9])
        record["nodes"] = [{"sources": n.sources, "children": [c.sources for c in n.children],
                            "depth": n.depth, "xi_nats": n.xi_value,
                            "syn_nats": n.syn_value} for n in nodes]
        record["xi_nats"] = tree.root.xi_value
        record["root_syn_nats"] = tree.root.syn_value
        audit["closure_error_nats"] = tree.closure_error
        splits = sorted([(ei[l]+ei[r], l, r) for l,r in nontrivial_bipartitions(SOURCES)],
                        reverse=True, key=lambda row: row[0])
        best, second = splits[:2]
        difference = (information_samples[best[1]] + information_samples[best[2]]
                      - information_samples[second[1]] - information_samples[second[2]])
        record["root_score_gap_nats"] = float(difference.mean())
        # Diagnostic only: QMC rows are not independent experimental replicates.
        record["root_score_gap_row_sem_nats"] = float(difference.std(ddof=1)/np.sqrt(count))
        record["root_split_first_half"] = max(nontrivial_bipartitions(SOURCES),
            key=lambda p: np.mean(information_samples[p[0]][:count//2]+information_samples[p[1]][:count//2]))
        record["root_split_second_half"] = max(nontrivial_bipartitions(SOURCES),
            key=lambda p: np.mean(information_samples[p[0]][count//2:]+information_samples[p[1]][count//2:]))

    pair_xi, phi_r, future = (np.zeros((9,9)) for _ in range(3))
    self_mi = {i: oracle.info((i,), (i,)) for i in SOURCES}
    for a,b in itertools.combinations(SOURCES,2):
        pair_xi[a-1,b-1] = pair_xi[b-1,a-1] = ei[(a,b)]-ei[(a,)]-ei[(b,)]
        redundancy = min(self_mi[a], self_mi[b], oracle.info((a,),(b,)), oracle.info((b,),(a,)))
        phi = oracle.info((a,b),(a,b)) - self_mi[a]-self_mi[b] + redundancy
        phi_r[a-1,b-1] = phi_r[b-1,a-1] = phi
        rows = (oracle.marginal(0,mask((a,b))) - oracle.marginal(0,mask((a,)))
                - oracle.marginal(0,mask((b,))))
        future[a-1,b-1] = future[b-1,a-1] = float(rows.mean())
    for name, matrix in (("pair_xi_hac", pair_xi), ("pair_phi_r_hac", phi_r),
                         ("future_mi_hac", future)):
        if name == "pair_xi_hac" and record["status"] != "valid":
            record[name] = {"status": "invalid_shared_ei", "error": record["error"]}
        elif matrix.max()>0:
            root, clades = graph_tree(matrix)
            record[name] = score(model, root, clades)
        else:
            record[name] = {"status": "no_positive_affinity"}
        record[name+"_matrix_nats"] = matrix.tolist()
    record["mean_pair_phi_r_nats"] = float(phi_r[np.triu_indices(9,1)].mean())
    record["maximum_abs_jackknife_correction_nats"] = max(map(abs, oracle.integration_changes.values()))
    record["full_target_jackknife_correction_nats"] = oracle.integration_changes[(0,FULL)]
    record["elapsed_seconds"] = time.monotonic()-start
    return record


def protocol(count, particles, seed):
    return {"model": "continuous_nonlinear_stochastic_coupled_map", "units": "nats",
            "equation": "x_i'=a_i*x_i+b_i*tanh(1.5*x_i)+sum_e c_ei*prod_{j in e}sqrt(2)*sin(pi*x_j+alpha_j)+sigma_i*epsilon_i",
            "source": "nine scalar current node states", "target": "complete nine scalar future node states",
            "intervention": "independent Uniform(-1,1)^9", "horizon_steps": 1,
            "memory": MEMORY.tolist(), "local_nonlinearity": LOCAL.tolist(),
            "noise_sd": NOISE.tolist(), "node_response": RESPONSE.tolist(),
            "node_phase": PHASE.tolist(), "hyperedge_gain_by_order": GAIN,
            "estimator": "exact nonlinear conditional triangular TM with Sobol source marginalization",
            "density_map": "t_x=normal_icdf((x+1)/2); t_y=(y-f(x))/sigma; diagonal derivative 1/sigma>0",
            "density_correction": "two-half first-order jackknife of log mixture likelihood",
            "evaluation_count": count, "integration_particles": particles, "seed": seed,
            "replication": "one paired scramble; exploratory pilot; no seed-level confidence intervals",
            "syn_tolerance_nats": SYN_TOLERANCE_NATS,
            "stopping": "three fixed architectures, no parameter search; significant Syn violation stops that SPT",
            "scoring": "mechanistic nonroot support F1; proper refinements inside minimal supports unscored equally for every method",
            "baselines": "raw pair Xi and MMI pair Phi_R plus future MI, all average linkage; no affinity clipping",
            "baseline_validity": "local-target Phi_R/future-MI scores are provisional until integration convergence; pair-Xi hierarchy is suppressed if the full-target Syn audit fails",
            "scope_limit": "not an ODE, not Kuramoto; oracle knowledge of the nonlinear mechanism; no recursive block Phi_R pilot",
            "manuscript": {"parent": "P6UJCVG8", "title": "Emergent hierarchical organization of causal interactions in complex systems",
                "main": "DXGC7JEA", "supplement": "MWIWKSVG", "checked": "2026-10-13",
                "version": "no explicit draft date/version in available attachments",
                "locations": "Methods Eqs (4)-(12), pp15-17; SI S1.2 Eqs(S7)-(S15), pp3-4; S5 pp8-9; S7 pp11-12",
                "estimator_difference": "exact known conditional TM replaces the SI affine-feature approximation; densities of missing sources still require numerical integration"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=int, default=1024)
    parser.add_argument("--particles", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cases", default="ABC")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"protocol": protocol(args.evaluations,args.particles,args.seed), "records": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        record = analyze(case,args.evaluations,args.particles,args.seed)
        result["records"].append(record)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
        print(json.dumps({"case": case,"status":record["status"],"seconds":record["elapsed_seconds"],
              "audit":record["audit"],"spt":record.get("spt"),
              "phi_r":record.get("pair_phi_r_hac"),"future":record.get("future_mi_hac")}),flush=True)
    if any(r["status"] != "valid" for r in result["records"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
