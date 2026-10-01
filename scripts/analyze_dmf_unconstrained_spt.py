#!/usr/bin/env python3
"""Cache-only DMF SPT preflight; independent G trees and explicit search audit."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt import (
    SPTConfig, SPTNode, SPTNonnegativityError, build_spt, canonical_split,
    flatten_nodes, nontrivial_bipartitions, pairwise_syn_affinity,
    spectral_candidate_selector,
)

TOL = 1e-8  # bits, numerical tolerance only


class ROIXiOracle:
    """Block conditional TC, equivalent to scalar Xi minus within-ROI E/I Xi.

    The supplied covariance conditions on the same complete future target.
    Cholesky evaluates positive-definite principal minors without eigenvalue floors.
    """
    def __init__(self, conditional: np.ndarray):
        self.conditional = np.asarray(conditional, dtype=float)
        if (self.conditional.ndim != 2 or self.conditional.shape[0] != self.conditional.shape[1]
                or self.conditional.shape[0] % 2 or not np.isfinite(self.conditional).all()
                or not np.allclose(self.conditional, self.conditional.T, atol=1e-12, rtol=0)):
            raise ValueError("Expected a finite symmetric E/I covariance with even dimension")
        self.count = self.conditional.shape[0] // 2
        cholesky(self.conditional, lower=True, check_finite=False)
        self.block_logdet = np.array([self.logdet((i, i + self.count)) for i in range(self.count)])
        self.cache = {(): 0.0, **{(i,): 0.0 for i in range(self.count)}}
        self.tolerance_zero_count = 0
        self.minimum_xi = 0.0

    def logdet(self, indices):
        local = self.conditional[np.ix_(indices, indices)]
        factor = cholesky(local, lower=True, check_finite=False)
        return float(2 * np.log(np.diag(factor)).sum())

    def xi(self, indices):
        key = tuple(sorted(map(int, indices)))
        if len(set(key)) != len(key) or any(i < 0 or i >= self.count for i in key):
            raise ValueError("Invalid ROI coalition")
        if key not in self.cache:
            source = key + tuple(i + self.count for i in key)
            value = float((self.block_logdet[list(key)].sum() - self.logdet(source)) / (2 * math.log(2)))
            self.minimum_xi = min(self.minimum_xi, value)
            if value < -TOL:
                raise SPTNonnegativityError(
                    f"Coalition Xi violation: minimum={value}, threshold={-TOL}, affected_count=1")
            self.tolerance_zero_count += int(value < 0)
            self.cache[key] = value  # retain raw numerical value
        return self.cache[key]


def node_record(node):
    return {"indices": list(node.sources), "xi_bits": node.xi_value,
            "syn_bits_raw": node.syn_value, "search_kind": node.split_kind,
            "children": [node_record(c) for c in node.children]}


def from_record(record, depth=0):
    return SPTNode(tuple(record["indices"]), record["xi_bits"], record["syn_bits_raw"],
                   depth, record["search_kind"], tuple(from_record(c, depth + 1) for c in record["children"]))


def retained_core(root, limit):
    node = root
    gaps = []
    while node.size > limit and node.children:
        ordered = sorted(node.children, key=lambda c: (-c.xi_value, c.sources))
        gaps.append(float(ordered[0].xi_value - ordered[1].xi_value))
        node = ordered[0]
    if node.size < 2:
        return {"members": [], "size": 0, "xi_bits": None, "minimum_path_gap_bits": min(gaps, default=None)}
    return {"members": list(node.sources), "size": node.size, "xi_bits": node.xi_value,
            "minimum_path_gap_bits": min(gaps, default=None)}


def weakest_split(oracle, members):
    if len(members) < 2:
        return None
    if len(members) > 10:
        raise ValueError("Exact core split restricted to at most 10 ROI")
    parent = oracle.xi(members)
    values = [parent - oracle.xi(l) - oracle.xi(r) for l, r in nontrivial_bipartitions(members)]
    minimum = min(values)
    count = sum(v < -TOL for v in values)
    if count:
        raise SPTNonnegativityError(f"Core split violation: minimum={minimum}, threshold={-TOL}, affected_count={count}")
    return {"bits_raw": minimum, "candidate_count": len(values),
            "tolerance_zero_count": sum(-TOL <= v < 0 for v in values)}


def clades(root):
    return {node.sources for node in flatten_nodes(root) if 1 < node.size < root.size}


def rf_distance(a, b):
    if set(a.sources) != set(b.sources):
        raise ValueError("Trees must share their leaves")
    ca, cb = clades(a), clades(b)
    return len(ca ^ cb) / max(1, len(ca) + len(cb))


def mass_distance(a, b):
    if a.xi_value <= TOL or b.xi_value <= TOL:
        return None  # no normalized interpretation near numerical zero
    def masses(root):
        nodes = [n for n in flatten_nodes(root) if n.children]
        # Only tolerance-scale negatives become zero, explicitly counted in build audit.
        total = sum(0.0 if -TOL <= n.syn_value < 0 else n.syn_value for n in nodes)
        return {n.sources: (0.0 if -TOL <= n.syn_value < 0 else n.syn_value) / total for n in nodes}
    ma, mb = masses(a), masses(b)
    return 0.5 * sum(abs(ma.get(k, 0) - mb.get(k, 0)) for k in ma.keys() | mb.keys())


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else None


def make_selector(affinity, oracle, extra, gaps):
    base = spectral_candidate_selector(affinity, exact_max_size=10)
    def select(sources):
        kind, candidates = base(sources)
        candidates = set(candidates)
        if extra and len(sources) > 10:
            # Nested augmentation: all original candidates remain eligible.
            candidates.update(canonical_split((i,), set(sources) - {i}, sources) for i in sources)
            rng = np.random.default_rng(np.random.SeedSequence([731, *sources]))
            for _ in range(extra):
                size = int(rng.integers(1, len(sources) // 2 + 1))
                left = tuple(rng.choice(sources, size=size, replace=False).tolist())
                candidates.add(canonical_split(left, set(sources) - set(left), sources))
            kind = "spectral-plus-random"
        scores = sorted(oracle.xi(sources) - oracle.xi(l) - oracle.xi(r) for l, r in candidates)
        gaps.append({"parent": list(sources), "candidate_count": len(scores),
                     "best_syn_bits": scores[0], "runner_up_gap_bits": scores[1] - scores[0] if len(scores) > 1 else None})
        return kind, sorted(candidates)
    return select


def summarize(records, gs, seeds):
    trees = {(r["seed"], r["G"], r["search"]): from_record(r["tree"]) for r in records}
    index = {(r["seed"], r["G"], r["search"]): r for r in records}
    within, across, sensitivity = [], [], []
    for g in gs:
        for i, seed in enumerate(seeds):
            for other in seeds[i + 1:]:
                a, b = trees[seed, g, "baseline"], trees[other, g, "baseline"]
                within.append({"G": g, "seeds": [seed, other], "rf": rf_distance(a,b), "tv": mass_distance(a,b),
                               "core_jaccard": jaccard(index[seed,g,"baseline"]["cores"]["10"]["members"], index[other,g,"baseline"]["cores"]["10"]["members"])})
        for seed in seeds:
            a, b = trees[seed,g,"baseline"], trees[seed,g,"expanded"]
            sensitivity.append({"G":g, "seed":seed, "rf":rf_distance(a,b), "tv":mass_distance(a,b),
                                "core_jaccard":jaccard(index[seed,g,"baseline"]["cores"]["10"]["members"],index[seed,g,"expanded"]["cores"]["10"]["members"])})
    for i, g in enumerate(gs):
        for h in gs[i + 1:]:
            for seed in seeds:
                a, b = trees[seed,g,"baseline"], trees[seed,h,"baseline"]
                across.append({"G_pair":[g,h], "seed":seed, "delta_G":h-g, "rf":rf_distance(a,b), "tv":mass_distance(a,b),
                               "core_jaccard":jaccard(index[seed,g,"baseline"]["cores"]["10"]["members"],index[seed,h,"baseline"]["cores"]["10"]["members"])})
    return {"within_G":within, "across_G":across, "search_sensitivity":sensitivity}


def plot(records, summary, fixed, gs, seeds, output):
    plt.rcParams.update({"font.family":"sans-serif", "font.size":8, "axes.spines.top":False,
                         "axes.spines.right":False, "legend.frameon":False})
    baseline = {(r["seed"],r["G"]):r for r in records if r["search"] == "baseline"}
    fig = plt.figure(figsize=(14.2,7.4), constrained_layout=True)
    grid = fig.add_gridspec(2,4, width_ratios=(1.1,1.1,1,1))
    ax = fig.add_subplot(grid[0,0])
    for seed in seeds:
        ax.plot(gs, [baseline[seed,g]["cross_roi_xi_bits"] for g in gs],color="#cbd5df",lw=0.8)
    vals=np.array([[baseline[s,g]["cross_roi_xi_bits"] for g in gs] for s in seeds])
    ax.errorbar(gs, vals.mean(0), yerr=vals.std(0,ddof=1)/np.sqrt(len(seeds)),color="#436882",marker="o",capsize=3)
    ax.set(xlabel="$G$",ylabel="Cross-ROI $\\Xi$ (bits)")
    ax.text(0,1.04,"a",transform=ax.transAxes,fontweight="bold")
    ax=fig.add_subplot(grid[0,1])
    groups=[summary["within_G"], [r for r in summary["across_G"] if np.isclose(r["delta_G"],0.1)],
            [r for r in summary["across_G"] if np.isclose(r["delta_G"],0.2)],summary["search_sensitivity"]]
    labels=["Same G\nnew seed","$\\Delta G=0.1$\nsame seed","$\\Delta G=0.2$\nsame seed","Expanded\nsearch"]
    for i, rows in enumerate(groups):
        values=np.array([r["rf"] for r in rows])
        ax.scatter(i+np.linspace(-0.12,0.12,len(values)),values,s=8,color="#b4c4d0",alpha=0.5)
        ax.scatter([i],[values.mean()],s=35,color="#436882",marker="D")
    ax.set(xticks=range(4),xticklabels=labels,ylabel="Normalized rooted RF",ylim=(0,1.04))
    ax.tick_params(axis="x",labelsize=7)
    ax.text(0,1.04,"b",transform=ax.transAxes,fontweight="bold")
    ax=fig.add_subplot(grid[0,2])
    for i,rows in enumerate(groups):
        values=np.array([r["tv"] for r in rows if r["tv"] is not None])
        ax.scatter(i+np.linspace(-0.12,0.12,len(values)),values,s=8,color="#b4c4d0",alpha=0.5)
        ax.scatter([i],[values.mean()],s=35,color="#436882",marker="D")
    ax.set(xticks=range(4),xticklabels=labels,ylabel="Node-mass total variation",ylim=(0,1.04))
    ax.tick_params(axis="x",labelsize=7)
    ax.text(0,1.04,"c",transform=ax.transAxes,fontweight="bold")
    ax=fig.add_subplot(grid[0,3])
    for i,rows in enumerate(groups):
        values=np.array([r["core_jaccard"] for r in rows if r["core_jaccard"] is not None])
        ax.scatter(i+np.linspace(-0.12,0.12,len(values)),values,s=8,color="#b4c4d0",alpha=0.5)
        ax.scatter([i],[values.mean()],s=35,color="#436882",marker="D")
    ax.set(xticks=range(4),xticklabels=labels,ylabel="$C_{10}$ Jaccard similarity",ylim=(-0.04,1.04))
    ax.tick_params(axis="x",labelsize=7)
    ax.text(0,1.04,"d",transform=ax.transAxes,fontweight="bold")
    for panel,search in enumerate(("baseline","expanded")):
        ax=fig.add_subplot(grid[1,panel])
        frequency=np.zeros((100,len(gs)))
        for r in records:
            if r["search"]==search:
                frequency[r["cores"]["10"]["members"],gs.index(r["G"])]+=1/len(seeds)
        # Display every ROI, in fixed atlas order, including never-selected regions.
        im=ax.imshow(frequency,aspect="auto",origin="lower",vmin=0,vmax=1,cmap="Blues",interpolation="nearest")
        ax.set(xticks=range(len(gs)),xticklabels=[f"{g:g}" for g in gs],xlabel="$G$",ylabel="ROI index (fixed atlas order)")
        fig.colorbar(im,ax=ax,label="$C_{10}$ inclusion frequency",fraction=0.06,pad=0.025)
        ax.text(0,1.04,"ef"[panel]+"  "+search.capitalize()+" search",transform=ax.transAxes,fontweight="bold")
    colors=["#436882","#aa725a","#68947b"]
    for panel,key,ylabel in [(2,"xi_bits","Fixed-coalition $\\Xi$ (bits)"),(3,"weakest_bits","Exact weakest split (bits)")]:
        ax=fig.add_subplot(grid[1,panel])
        for ref,color in zip(gs,colors):
            rows=[r for r in fixed if r["reference_G"]==ref and r["seed"]!=4]
            if not rows:
                continue
            vals=np.array([[next(r[key] for r in rows if r["seed"]==s and r["G"]==g) for g in gs] for s in seeds if s!=4])
            ax.errorbar(gs,vals.mean(0),yerr=vals.std(0,ddof=1)/np.sqrt(vals.shape[0]),color=color,marker="o",capsize=2,label=f"Selected at G={ref:g}")
        ax.set(xlabel="$G$",ylabel=ylabel)
        ax.text(0,1.04,"gh"[panel-2],transform=ax.transAxes,fontweight="bold")
        if panel==3:
            ax.legend(loc="center left",bbox_to_anchor=(1.02,0.5),fontsize=7)
    sampling = "shared inputs and noise" if records[0]["config"]["paired_inputs_across_G"] else "G-specific inputs"
    fig.suptitle(f"Cache preflight: 8 seeds; 2,048 samples; {sampling}; exact search at ≤10 ROI\nCurves: mean ± SEM; fixed cores selected by baseline search, seed 4; evaluated on the other 7 seeds",fontsize=9)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=300,bbox_inches="tight")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=ROOT/"results/dmf_schaefer100/full/critical_yeo7.npz")
    p.add_argument("--output-dir",type=Path,default=ROOT/"results/dmf_schaefer100/unconstrained_spt_preflight")
    p.add_argument("--figure",type=Path,default=ROOT/"fig/brain_dmf_unconstrained_spt_preflight.png")
    p.add_argument("--extra-candidates",type=int,default=256)
    p.add_argument("--max-conditions",type=int,default=None,help="Bounded smoke test; omit for all 24 cached conditions")
    args=p.parse_args()
    if args.extra_candidates<=0:
        p.error("Extra candidates must be positive")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    fingerprint=hashlib.sha256(args.input.read_bytes()).hexdigest()
    with np.load(args.input,allow_pickle=True) as a:
        gs=list(map(float,a["G"])); seeds=list(map(int,a["seeds"]))
        cov=a["conditional_covariance"].copy(); expected=a["cross_roi"].copy()
        labels=list(map(str,a["region_labels"]))
        paired=all(bool(a[k].item()) if k in a.files else False
                   for k in ("paired_inputs_across_G","paired_noise_across_G"))
    config={"input_sha256":fingerprint,"extra_candidates":args.extra_candidates,"tolerance_bits":TOL,
            "exact_max_size":10,"estimator":"existing Gaussian conditional covariance","paired_inputs_across_G":paired}
    records=[]; count=0
    for si,seed in enumerate(seeds):
        for gi,g in enumerate(gs):
            if args.max_conditions is not None and count>=args.max_conditions:
                break
            for search,extra in [("baseline",0),("expanded",args.extra_candidates)]:
                path=args.output_dir/f"seed{seed:02d}_G{g:.2f}_{search}.json"
                if path.exists():
                    record=json.loads(path.read_text())
                    if record.get("config") == config:
                        records.append(record)
                        print(f"reuse seed={seed} G={g} {search}",flush=True)
                        continue
                begin=time.perf_counter(); oracle=ROIXiOracle(cov[si,gi])
                affinity,pair_zeros=pairwise_syn_affinity(oracle,oracle.count,tolerance=TOL)
                gaps=[]
                result=build_spt(tuple(range(oracle.count)),oracle,config=SPTConfig(syn_tolerance=TOL),
                                 candidate_selector=make_selector(affinity,oracle,extra,gaps))
                if abs(result.closure_error)>TOL or abs(result.root.xi_value-expected[si,gi])>TOL:
                    raise RuntimeError("Closure or existing covariance Xi agreement failed")
                cores={str(limit):retained_core(result.root,limit) for limit in (5,10,20)}
                cores["10"]["weakest_split"]=weakest_split(oracle,cores["10"]["members"])
                record={"config":config,"seed":seed,"G":g,"search":search,"elapsed_seconds":time.perf_counter()-begin,
                        "cross_roi_xi_bits":result.root.xi_value,"closure_error_bits":result.closure_error,
                        "audit":asdict(result.audit),"pair_tolerance_zero_count":pair_zeros,
                        "xi_tolerance_zero_count":oracle.tolerance_zero_count,"coalition_count":len(oracle.cache),
                        "cores":cores,"candidate_gaps":gaps,"tree":node_record(result.root)}
                path.write_text(json.dumps(record,separators=(",",":"),allow_nan=False))
                records.append(record)
                print(f"done seed={seed} G={g} {search}: {record['elapsed_seconds']:.2f}s, C10={cores['10']['members']}",flush=True)
            count+=1
    if args.max_conditions is not None:
        print(f"smoke complete: {count} conditions; {time.perf_counter()-started:.2f}s",flush=True)
        return
    summary=summarize(records,gs,seeds)
    fixed=[]
    for gref in gs:
        ref=next(r for r in records if r["seed"]==4 and r["G"]==gref and r["search"]=="baseline")
        members=ref["cores"]["10"]["members"]
        if not members:
            continue
        for si,seed in enumerate(seeds):
            for gi,g in enumerate(gs):
                oracle=ROIXiOracle(cov[si,gi]); weak=weakest_split(oracle,members)
                fixed.append({"reference_G":gref,"reference_seed":4,"members":members,"seed":seed,"G":g,
                              "xi_bits":oracle.xi(members),"weakest_bits":weak["bits_raw"],"audit":weak})
    payload={"config":config,"status":"cache preflight only; formal common-input sweep not run",
             "G":gs,"seeds":seeds,"labels":labels,"records":records,"comparisons":summary,
             "fixed_reference_cores":fixed,"elapsed_seconds":time.perf_counter()-started,
             "limitations":([] if paired else ["Inputs differ across G in existing caches"]) + ["Only G=1.2,1.3,1.4; no G=0 background",
                            "Reference cores at these three G are provisional, not the planned 0.8,1.3,2.0 validation",
                            "Approximate spectral search with nested singleton/random augmentation; not global optimum",
                            "Numerical zero threshold is not a statistical background criterion"]}
    (args.output_dir/"summary.json").write_text(json.dumps(payload,separators=(",",":"),allow_nan=False))
    plot(records,summary,fixed,gs,seeds,args.figure)
    print(f"complete: {time.perf_counter()-started:.2f}s; {args.figure}",flush=True)


if __name__=="__main__":
    main()
