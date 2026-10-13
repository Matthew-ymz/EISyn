"""Fixed-query RQMC precision diagnostic; never constructs a tree from partial EI.

Both uncorrected and two-half jackknife densities are reported.  Independent
scrambles are the numerical repetition unit; all N/P comparisons reuse nested
prefixes and every query has the same complete-future-state denominator.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_continuous_hierarchy_pilot import (
    ContinuousMap, MarginalOracle, NOISE, FULL, mask, columns, SYN_TOLERANCE_NATS,
)
from scripts.spt_heterogeneous_hierarchy_case import SOURCES

SETTINGS = ((1024, 1024), (1024, 4096), (1024, 16384), (4096, 16384))


def residual_audit(values):
    minimum = min(values.values())
    affected = sum(v < -SYN_TOLERANCE_NATS for v in values.values())
    audit = {"minimum_nats": minimum, "significant_violation_count": affected,
             "tolerance_negative_count": sum(-SYN_TOLERANCE_NATS <= v < 0 for v in values.values()),
             "threshold_nats": -SYN_TOLERANCE_NATS,
             "status": "invalid_syn_estimate" if affected else "partial_queries_within_tolerance"}
    if affected:
        audit["error"] = (f"Syn nonnegativity failed: minimum={minimum:.8g} nats; "
                          f"threshold={-SYN_TOLERANCE_NATS}; affected_count={affected}; no tree constructed")
    return audit


def density_prefixes(oracle, bits, settings):
    result = {p: {kind: np.empty(oracle.count) for kind in ("raw", "jackknife", "ess")}
              for p in sorted({p for _, p in settings})}
    if bits == FULL:
        ll = oracle.model.log_prob(oracle.x, oracle.y)
        for p, item in result.items():
            item["raw"][:] = item["jackknife"][:] = ll
            item["ess"][:] = p
        return result
    kept = columns(bits)
    left, right = [], []
    for edge in oracle.model.supports:
        retained = [i-1 for i in edge if bits >> (i-1) & 1]
        missing = [i-1 for i in edge if not bits >> (i-1) & 1]
        left.append(np.prod(oracle.eval_features[:, retained], axis=1))
        right.append(np.prod(oracle.base_features[:, missing], axis=1))
    left, right = np.stack(left, -1), np.stack(right, -1)
    chunk = max(1, 131072 // oracle.particles)
    constant = np.sum(np.log(2*np.pi*NOISE**2))
    for start in range(0, oracle.count, chunk):
        stop = min(oracle.count, start+chunk)
        means = (left[start:stop, None, :] * right[None, :, :]) @ oracle.model.coefficients
        for i in range(9):
            means[..., i] += (oracle.local_eval[start:stop, i, None] if i in kept
                              else oracle.local_base[None, :, i])
        residual = (oracle.y[start:stop, None, :] - means) / NOISE
        ll = -.5 * (np.sum(residual*residual, axis=-1) + constant)
        for p, item in result.items():
            logs = ll[:, :p]
            norm = logsumexp(logs, axis=1)
            raw = norm - np.log(p)
            halves = .5*(logsumexp(logs[:, :p//2], axis=1)
                         + logsumexp(logs[:, p//2:], axis=1)) - np.log(p//2)
            item["raw"][start:stop] = raw
            item["jackknife"][start:stop] = 2*raw-halves
            item["ess"][start:stop] = np.exp(2*norm-logsumexp(2*logs, axis=1))
    return result


def run_case(case, seed, cached, settings=SETTINGS):
    start = time.monotonic()
    worst = tuple(tuple(x) for x in cached["audit"]["minimum_syn_split"])
    reference = ((1,2,3,4,5), (6,7,8,9)) if case != "C" else ((1,2,3,4,5,6), (7,8,9))
    queries = {"previous_worst": worst, "reference_root": (SOURCES, *reference)}
    bits = sorted({0, *(mask(s) for split in queries.values() for s in split)})
    oracle = MarginalOracle(ContinuousMap(case), max(n for n, _ in settings),
                            max(p for _, p in settings), seed)
    densities = {b: density_prefixes(oracle, b, settings) for b in bits}
    records = []
    for n, p in settings:
        ei = {kind: {b: float(np.mean(densities[b][p][kind][:n]
                                          - densities[0][p][kind][:n])) for b in bits}
              for kind in ("raw", "jackknife")}
        residuals = {kind: {name: ei[kind][mask(a)]-ei[kind][mask(l)]-ei[kind][mask(r)]
                           for name, (a, l, r) in queries.items()} for kind in ei}
        records.append({"case": case, "seed": seed, "evaluations": n, "particles": p,
                        "ei_nats": {kind: {str(b): v for b, v in vals.items()} for kind, vals in ei.items()},
                        "residual_nats": residuals,
                        "query_audit": {kind: residual_audit(vals) for kind, vals in residuals.items()},
                        "density_diagnostics": {str(b): {
                            "mean_log_density_nats": float(np.mean(densities[b][p]["raw"][:n])),
                            "jackknife_change_nats": float(np.mean(densities[b][p]["jackknife"][:n]
                                                                      - densities[b][p]["raw"][:n])),
                            "ess_median": float(np.median(densities[b][p]["ess"][:n])),
                            "ess_p10": float(np.quantile(densities[b][p]["ess"][:n], .1)),
                            "ess_below_10_fraction": float(np.mean(densities[b][p]["ess"][:n] < 10))}
                            for b in bits}})
    return {"case": case, "seed": seed, "queries": queries, "source_masks": bits,
            "elapsed_seconds": time.monotonic()-start, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=ROOT/"results/spt_continuous_abc_pilot/summary.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--cases", default="ABC")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cached = {r["case"]: r for r in json.loads(args.pilot.read_text())["records"]}
    settings = ((64, 64), (64, 256), (256, 256)) if args.smoke else SETTINGS
    result = {"protocol": {"settings": settings, "seeds": args.seeds,
        "syn_tolerance_nats": SYN_TOLERANCE_NATS, "units": "nats",
        "intervention": "iid Uniform(-1,1)^9", "target": "complete nine-dimensional future state",
        "horizon_steps": 1, "estimator": "exact conditional Gaussian triangular TM + uniform RQMC marginal integration",
        "repetition_unit": "independent paired Sobol scramble; not physical systems or evaluation rows",
        "prefix_pairing": "all queries/cases share evaluation and integration draws for a seed; N/P use nested prefixes",
        "reported_estimators": ["raw", "two_half_jackknife"], "clipping": False,
        "scope": "previous worst split and fixed reference root only; partial audit cannot certify full EI table or tree"},
        "runs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for case in args.cases:
            run = run_case(case, seed, cached[case], settings)
            result["runs"].append(run)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
            print(json.dumps({"case": case, "seed": seed, "seconds": run["elapsed_seconds"],
                              "highest_precision_residuals": run["records"][-1]["residual_nats"]}), flush=True)
    failures = [a for run in result["runs"] for record in run["records"]
                for a in record["query_audit"].values() if a["significant_violation_count"]]
    if failures:
        # Finish the independent diagnostics and preserve every failed setting;
        # none can be consumed as a validated EI table or hierarchy.
        print(f"Recorded {len(failures)} invalid estimator/settings: "
              + min(failures,key=lambda a:a["minimum_nats"])["error"],file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
