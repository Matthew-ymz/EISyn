#!/usr/bin/env python3
"""Bounded paired control-screening pilot for two-node pulse ignition.

Experiment contract
-------------------
Question: under a fixed simulation/query budget, does PEID/EI screening
improve selection of a two-source low-to-high basin pulse?  For each of 20
fixed instance seeds, all 28 source pairs share the same independent 5x5
discrete amplitude intervention table (25 labels per pair).  The ranking
methods are Syn (conditional-MI PEID), joint EI, observed success rate,
minimum observed successful energy (failed pairs rank at infinity), and a
fixed-seed random permutation.  Each method selects its top three pairs.
Selected pairs are evaluated on the same 21x21 grid; the all-pair 21x21
minimum is an evaluation-only exhaustive oracle and is not part of the
method budget.  A coarse-grid successful point is also admitted to the final
candidate set.

The two paired cost conditions use identical dynamics, interventions, and
labels. Syn, joint EI, success rate, and random rankings are shared; the
minimum-success-energy ranking is intentionally cost-based in each condition:
homogeneous r_i=1 and one fixed log-uniform
heterogeneous vector r_i in [0.25,4] per seed.  Pulse energy is exactly
t_force * (r_i*u_i**2 + r_j*u_j**2).  Continuous optima are not claimed.
The basin label is force_end > theta.  Syn is the discrete-frequency
quantity I(U_i;U_j|Y), in bits; tolerance is 1e-10 bits. Values in
[-tolerance,0) are recorded as numerical zero, while values below
-tolerance fail explicitly.  No clipping is used.

The default run prints a compact JSON summary to stdout and writes one PNG.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp.network_revival.joint_required_ignition import (  # noqa: E402
    JointRequiredIgnitionConfig,
    _hill,
    _integrate_constant_input,
    _peid_row,
    build_threshold_latch_instance,
)

SYN_TOL = 1e-10
METHODS = ("syn", "joint_ei", "success_rate", "min_success_energy", "random")


def response(instance, pair, amplitudes, cfg):
    left, right = pair
    di, dj = np.meshgrid(amplitudes, amplitudes, indexing="ij")
    forcing = (instance["weights"][left] * _hill(di.ravel(), instance)
               + instance["weights"][right] * _hill(dj.ravel(), instance))
    force_end, _, _ = _integrate_constant_input(
        np.zeros_like(forcing), forcing, theta=cfg.theta, kappa=cfg.kappa,
        duration=cfg.t_force, dt=cfg.dt)
    labels = (force_end > cfg.theta).astype(int)
    return di.ravel(), dj.ravel(), np.asarray(force_end), labels


def energy(di, dj, pair, rates, t_force):
    return float(t_force) * (float(rates[pair[0]]) * di**2 + float(rates[pair[1]]) * dj**2)


def rank_pairs(instance, pairs, coarse, rates, rng):
    rows = []
    tie = {pairs[int(index)]: rank for rank, index in enumerate(rng.permutation(len(pairs)))}
    for pair in pairs:
        di, dj, _, labels = coarse[pair]
        row = _peid_row(
            np.arange(5, dtype=int).repeat(5), np.tile(np.arange(5, dtype=int), 5), labels
        )
        raw_syn = float(row["synergy"])
        successes = labels.astype(bool)
        observed_energy = energy(di[successes], dj[successes], pair, rates, 8.0)
        rows.append({"pair": pair, "syn": raw_syn, "raw_syn": raw_syn, "joint_ei": row["joint_ei"],
                     "success_rate": float(np.mean(labels)),
                     "min_success_energy": float(np.min(observed_energy)) if np.any(successes) else np.inf,
                     "tie": tie[pair]})
    raw = np.asarray([row["raw_syn"] for row in rows])
    if not np.all(np.isfinite(raw)):
        raise RuntimeError("Non-finite Syn estimates in the discovery table.")
    violations = raw < -SYN_TOL
    if np.any(violations):
        raise RuntimeError(
            f"Syn nonnegativity violation: minimum={raw.min():.17g} bits, "
            f"threshold={-SYN_TOL:.17g} bits, affected_count={int(violations.sum())}"
        )
    for row in rows:
        if row["raw_syn"] < 0:
            row["syn"] = 0.0  # Declared tolerance; raw value retained for diagnostics.
    return rows


def select(rows, method):
    # A common random tie permutation is the final key for every method.
    keys = {"syn": lambda r: (-r["syn"], r["tie"]),
            "joint_ei": lambda r: (-r["joint_ei"], r["tie"]),
            "success_rate": lambda r: (-r["success_rate"], r["tie"]),
            "min_success_energy": lambda r: (r["min_success_energy"], r["tie"]),
            "random": lambda r: (r["tie"],)}
    return [r["pair"] for r in sorted(rows, key=keys[method])[:3]]


def evaluate_pair(pair, instance, fine, coarse, rates, cfg):
    di, dj, _, labels = fine[pair]
    costs = energy(di, dj, pair, rates, cfg.t_force)
    candidates = [(float(c), int(l)) for c, l in zip(costs, labels, strict=True)]
    # Explicitly admit all successful coarse-grid solutions to the candidate set.
    cdi, cdj, _, clabels = coarse[pair]
    c_costs = energy(cdi, cdj, pair, rates, cfg.t_force)
    candidates.extend((float(c), int(l)) for c, l in zip(c_costs, clabels, strict=True))
    successful = [c for c, label in candidates if label]
    return float(min(successful)) if successful else np.inf


def run(seed_count=20, seed_start=20260615, dt_scale=1.0):
    cfg = JointRequiredIgnitionConfig(
        source_count=8, amplitude_max=4.0, amplitude_levels=5,
        t_force=8.0, release_time=0.0, dt=0.04 * dt_scale,
    )
    pairs = list(combinations(range(cfg.source_count), 2))
    records = []
    syn_min = np.inf
    syn_negative_count = 0
    closest_threshold_gap = np.inf
    near_threshold_count = 0
    label_checks = []
    for offset in range(seed_count):
        seed = int(seed_start + offset)
        instance = build_threshold_latch_instance(cfg, seed)
        coarse = {p: response(instance, p, np.linspace(0, 4, 5), cfg) for p in pairs}
        fine = {p: response(instance, p, np.linspace(0, 4, 21), cfg) for p in pairs}
        all_force = np.concatenate([fine[p][2] for p in pairs])
        gaps = np.abs(all_force - cfg.theta)
        closest_threshold_gap = min(closest_threshold_gap, float(np.min(gaps)))
        near_threshold_count += int(np.sum(gaps <= 1e-6))
        rng = np.random.default_rng(seed + 9173)
        homogeneous = np.ones(cfg.source_count)
        heterogeneous = np.exp(rng.uniform(np.log(0.25), np.log(4.0), cfg.source_count))
        # Ranking is shared by the two cost conditions except where cost is the ranking score.
        rows_h = rank_pairs(instance, pairs, coarse, homogeneous, np.random.default_rng(seed + 441))
        rows_x = rank_pairs(instance, pairs, coarse, heterogeneous, np.random.default_rng(seed + 441))
        for row in rows_h:
            syn_min = min(syn_min, row["raw_syn"])
            syn_negative_count += int(row["raw_syn"] < 0)
        # Exact paired dt diagnostic on all 28 pairs for representative seeds.
        if offset < min(3, seed_count):
            cfg_half = JointRequiredIgnitionConfig(source_count=8, amplitude_max=4.0,
                amplitude_levels=5, t_force=8.0, release_time=0.0, dt=0.02)
            coarse_match = fine_match = True
            for pair in pairs:
                coarse_half = response(instance, pair, np.linspace(0, 4, 5), cfg_half)[3]
                fine_half = response(instance, pair, np.linspace(0, 4, 21), cfg_half)[3]
                coarse_match &= bool(np.array_equal(coarse[pair][3], coarse_half))
                fine_match &= bool(np.array_equal(fine[pair][3], fine_half))
            label_checks.append({"seed": seed, "n_pairs": len(pairs),
                                 "coarse_points_checked": len(pairs) * 25,
                                 "fine_points_checked": len(pairs) * 441,
                                 "coarse_match": coarse_match, "fine_match": fine_match})
        oracle = {}
        for rates_name, rates in (("homogeneous", homogeneous), ("heterogeneous", heterogeneous)):
            oracle_cost = min(evaluate_pair(p, instance, fine, coarse, rates, cfg) for p in pairs)
            global_coarse_best = min(evaluate_pair(p, instance, {p: coarse[p]}, coarse, rates, cfg) for p in pairs)
            ranked = rows_h if rates_name == "homogeneous" else rows_x
            for method in METHODS:
                selected = select(ranked, method)
                costs = [evaluate_pair(p, instance, fine, coarse, rates, cfg) for p in selected]
                best = min(global_coarse_best, min(costs))
                records.append({"seed": seed, "condition": rates_name, "method": method,
                                "best_cost": best, "oracle_cost": oracle_cost,
                                "regret": best - oracle_cost if np.isfinite(best) and np.isfinite(oracle_cost) else np.inf,
                                "relative_regret": (best - oracle_cost) / oracle_cost if np.isfinite(best) and np.isfinite(oracle_cost) and oracle_cost > 0 else np.inf,
                                "failed": not np.isfinite(best), "queries": 28 * 25 + 3 * (441 - 25),
                                "oracle_queries": 28 * 441})
    return records, {"raw_min_syn_bits": syn_min, "raw_negative_syn_count": syn_negative_count,
                     "syn_tolerance_bits": SYN_TOL, "dt_half_label_checks": label_checks,
                     "closest_fine_force_end_gap_to_theta": closest_threshold_gap,
                     "fine_points_within_1e-6_of_theta": near_threshold_count}


def summarize(records, diagnostics):
    out = {"n_seeds": len({r["seed"] for r in records}), "n_pairs": 28,
           "methods": list(METHODS), "diagnostics": diagnostics, "conditions": {}}
    for condition in ("homogeneous", "heterogeneous"):
        out["conditions"][condition] = {}
        for method in METHODS:
            rs = [r for r in records if r["condition"] == condition and r["method"] == method]
            finite = np.asarray([r["regret"] for r in rs if np.isfinite(r["regret"])])
            relative = np.asarray([r["relative_regret"] for r in rs if np.isfinite(r["relative_regret"])])
            out["conditions"][condition][method] = {
                "failure_rate": float(np.mean([r["failed"] for r in rs])),
                "finite_regret_mean": float(np.mean(finite)) if finite.size else None,
                "finite_regret_sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                "relative_regret_mean": float(np.mean(relative)) if relative.size else None,
                "relative_regret_sd": float(np.std(relative, ddof=1)) if relative.size > 1 else 0.0,
                "finite_regret_median": float(np.median(finite)) if finite.size else None,
                "finite_regret_count": int(finite.size),
                "paired_regret_differences_vs_syn": [
                    float(r["regret"] - s["regret"]) for r, s in zip(
                        [x for x in rs], [x for x in records if x["condition"] == condition and x["method"] == "syn"], strict=True)
                    if np.isfinite(r["regret"]) and np.isfinite(s["regret"])
                ],
                "queries_per_seed": 28 * 25 + 3 * (441 - 25),
                "oracle_queries_per_seed": 28 * 441,
            }
    return out


def plot_summary(summary, path):
    methods = list(summary["methods"])
    labels = ["Syn", "joint EI", "success rate", "min success energy", "random"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    colors = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#AA3377"]
    for ax, condition in zip(axes, ("homogeneous", "heterogeneous")):
        vals = [summary["conditions"][condition][m]["finite_regret_mean"] for m in methods]
        sds = [summary["conditions"][condition][m]["finite_regret_sd"] for m in methods]
        x = np.arange(len(methods))
        for xx, yy, sd, cc in zip(x, vals, sds, colors, strict=True):
            ax.errorbar([xx], [0 if yy is None else yy], yerr=[sd], fmt="none",
                        ecolor=cc, elinewidth=2, capsize=3)
            ax.scatter([xx], [0 if yy is None else yy], color=cc, s=28, zorder=3)
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylabel("Absolute finite regret")
        ax.set_title(condition)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run two seeds")
    ap.add_argument("--output", type=Path, default=ROOT / "fig" / "ei_control_screening_pilot.png")
    args = ap.parse_args()
    records, diagnostics = run(seed_count=2 if args.smoke else 20)
    summary = summarize(records, diagnostics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_summary(summary, args.output)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
