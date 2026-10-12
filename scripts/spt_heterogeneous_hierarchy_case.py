"""Exact heterogeneous nine-node Markov hierarchy: branched, deep, modular.

Each node has two fresh driver bits, one recurrent memory bit, eight readout
fields. Common independent-uniform intervention covers all 99 source bits;
target is the complete future 99-bit state, horizon=1. Node gates compress two
driver bits into independent activations with probabilities 1/4, 1/2 or 3/4.
These induced activation probabilities are NOT replaced by a uniform law.
Nested support parities have independent order-dependent dyadic flip noise.
Primitive readouts are copied; upper readouts are either copied or XOR shared.
Memory updates have heterogeneous nonlinear gates and node-specific flip noise.
Given its own future readouts a node can undo its memory-driving shift, giving
a BSC. Thus EI(S->T)=I(U_S;Z_visible(T))+sum_{i in S&T}[1-h2(rho_i)].
The memory terms cancel in Xi, while biased nonlinear activations change Xi.
Unused current readouts, fresh future drivers and independent masks integrate
out losslessly. All probabilities in the reduced activation/readout channel
are dyadic integer counts (denominator <=2^50); only entropy arithmetic floats.

All 511 subset EIs and 9,330 residuals are audited at 1e-10 bits, raw no clipping.
SPT has no planted split constraints. Ground truth is the declared laminar
support family, with internal refinements of minimal supports unscored for all
methods. They need not be EI ties in this heterogeneous model. This is a
development comparison; no universal ranking or finite-sample claims.

Zotero rechecked 2026-10-12: P6UJCVG8; DXGC7JEA Methods (5)-(12), pp15-17;
MWIWKSVG S5 pp8-9. No explicit draft version or date in current attachments.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt import SPTConfig, build_spt_from_ei_table, flatten_nodes, nontrivial_bipartitions
from scripts.spt_multilevel_parity_case import graph_tree

SOURCES = tuple(range(1, 10))
TOLERANCE = 1e-10
GATES = ("AND", "XOR", "OR", "NAND", "COPY", "AND", "OR", "XOR", "NOR")
ACTIVATION_NUMERATORS = (1, 2, 3, 3, 2, 1, 3, 2, 1)  # denominator 4
MEMORY_NOISE_NUMERATORS = (3, 4, 6, 5, 7, 4, 8, 6, 9)  # denominator 32
MEMORY_GATES = ("XOR", "AND", "OR", "MAJ", "XOR", "OR", "AND", "MAJ", "XOR")
MODELS = {
    "A": {"label": "Mixed branches",
          "supports": ((1,2,3),(4,5),(6,7),(8,9),(1,2,3,4,5),(6,7,8,9),SOURCES)},
    "B": {"label": "Deep cascade",
          "supports": ((1,2),(1,2,3),(1,2,3,4),(1,2,3,4,5),(6,7),(6,7,8),(6,7,8,9),SOURCES)},
    "C": {"label": "Triadic modules",
          "supports": ((1,2,3),(4,5,6),(7,8,9),(1,2,3,4,5,6),SOURCES)},
}
NOISE_BY_ORDER = {2:4, 3:3, 4:2, 5:2, 6:2, 9:1}  # denominator 16


def mask(sources):
    return sum(1 << (i-1) for i in sources)


def bits(count):
    return ((np.arange(1 << count)[:, None] >> np.arange(count)) & 1).astype(np.int64)


def entropy(counts):
    positive = counts[counts > 0].astype(float)
    if not len(positive):
        return 0.
    p = positive / positive.sum()
    return float(-np.sum(p*np.log2(p)))


def binary_entropy(p):
    return -p*math.log2(p)-(1-p)*math.log2(1-p)


def activate(drivers):
    """Boolean gates on each node's two driver fields, preserving batch axes."""
    values=[]
    for i,gate in enumerate(GATES):
        a,b=drivers[...,i,0],drivers[...,i,1]
        value={"AND":a&b,"OR":a|b,"XOR":a^b,"NAND":1-(a&b),
               "NOR":1-(a|b),"COPY":a}[gate]
        values.append(value)
    return np.stack(values,axis=-1)


def memory_drive(readouts,supports):
    values=[]
    for i,gate in enumerate(MEMORY_GATES,start=1):
        incident=[j for j,h in enumerate(supports) if i in h]
        a,b,c=(readouts[...,i-1,incident[j]] for j in (0,-1,len(incident)//2))
        value={"XOR":a^b,"AND":a&b,"OR":a|b,"MAJ":(a+b+c>=2).astype(np.int64)}[gate]
        values.append(value)
    return np.stack(values,axis=-1)


def step(state,case,mode,rng):
    """Actual node update for simulations; source states have shape (...,9,11)."""
    state=np.asarray(state,dtype=np.int64)
    if state.shape[-2:]!=(9,11) or np.any((state!=0)&(state!=1)) or mode not in ("copied","secret_shared"):
        raise ValueError("Expected full 9x11 state and valid readout mode")
    supports=MODELS[case]["supports"]
    primitive=tuple(h for h in supports if not any(set(g)<set(h) for g in supports))
    u=activate(state[...,:,:2])
    readouts=np.zeros((*state.shape[:-2],9,8),dtype=np.int64)
    for j,h in enumerate(supports):
        z=np.bitwise_xor.reduce(u[...,np.asarray(h)-1],axis=-1)
        z=z^(rng.random(z.shape)<NOISE_BY_ORDER[len(h)]/16).astype(np.int64)
        if mode=="copied" or h in primitive:
            readouts[...,np.asarray(h)-1,j]=z[...,None]
        else:
            shares=rng.integers(0,2,size=(*z.shape,len(h)-1),dtype=np.int64)
            readouts[...,np.asarray(h[:-1])-1,j]=shares
            readouts[...,h[-1]-1,j]=z^np.bitwise_xor.reduce(shares,axis=-1)
    future=np.zeros_like(state)
    future[...,:,:2]=rng.integers(0,2,size=(*state.shape[:-2],9,2),dtype=np.int64)
    future[...,3:]=readouts
    flips=(rng.random(state[...,2].shape)<np.asarray(MEMORY_NOISE_NUMERATORS)/32).astype(np.int64)
    future[...,2]=state[...,2]^memory_drive(readouts,supports)^flips
    return future


class Channel:
    def __init__(self, supports):
        self.supports = tuple(map(tuple, supports))
        self.primitive = tuple(h for h in self.supports
                               if not any(set(g) < set(h) for g in self.supports))
        self.activations = bits(9)
        self.targets = bits(len(supports))
        q = np.asarray(ACTIVATION_NUMERATORS)
        self.source_weights = np.prod(np.where(self.activations, q, 4-q), axis=1)
        p = np.asarray([NOISE_BY_ORDER[len(h)] for h in supports])
        self.noise_weights = np.prod(np.where(self.targets, p, 16-p), axis=1)
        z = np.stack([self.activations[:,np.asarray(h)-1].sum(axis=1)%2 for h in supports],axis=1)
        self.codes = (z*(1 << np.arange(len(supports)))).sum(axis=1)
        # For a fixed input, Z's distribution is a permutation of the flip law.
        self.counts = self.source_weights[:,None]*self.noise_weights[
            self.codes[:,None] ^ np.arange(len(self.targets))[None,:]]
        self.denominator = 4**9 * 16**len(supports)
        assert int(self.counts.sum()) == self.denominator <= 2**50
        self.memory_information = tuple(1-binary_entropy(k/32) for k in MEMORY_NOISE_NUMERATORS)
        self.query_count = 0

    @lru_cache(None)
    def source_projection(self, source_mask):
        selected = [i for i in range(9) if source_mask >> i & 1]
        return (self.activations[:,selected]*(1 << np.arange(len(selected)))).sum(axis=1)

    @lru_cache(None)
    def target_counts(self, target_mask):
        selected = [i for i in range(len(self.supports)) if target_mask >> i & 1]
        k = 1 << len(selected)
        projected = (self.targets[:,selected]*(1 << np.arange(len(selected)))).sum(axis=1)
        indices = np.arange(512)[:,None]*k + projected[None,:]
        # Integer masses <=2^50 are exactly representable as doubles in bincount.
        return np.bincount(indices.ravel(),weights=self.counts.ravel(),minlength=512*k).reshape(512,k)

    @lru_cache(None)
    def information(self, source_mask, target_mask):
        if not source_mask or not target_mask:
            return 0.
        self.query_count += 1
        counts = self.target_counts(target_mask)
        k = counts.shape[1]
        projection = self.source_projection(source_mask)
        grouped = np.bincount((projection[:,None]*k+np.arange(k)).ravel(),weights=counts.ravel(),
                             minlength=(1 << source_mask.bit_count())*k).reshape(-1,k)
        return entropy(grouped.sum(axis=1))+entropy(grouped.sum(axis=0))-entropy(grouped)

    def visible(self, target, mode):
        selected = set(target)
        return sum(1 << j for j,h in enumerate(self.supports)
                   if (bool(selected.intersection(h)) if mode == "copied" or h in self.primitive
                       else set(h) <= selected))

    def info(self, source, target, mode):
        return self.information(mask(source),self.visible(target,mode)) + math.fsum(
            self.memory_information[i-1] for i in set(source)&set(target))


def quality(root, clades, supports, primitive):
    truth = {frozenset(h) for h in supports if len(h)<9}
    found = {frozenset(h) for h in clades if not any(set(h)<set(g) for g in primitive)}
    common = len(found&truth)
    precision = common/len(found) if found else 0.
    recall = common/len(truth)
    parents = tuple(h for h in supports if len(h)<9 and not any(set(h)<set(g)<set(SOURCES) for g in supports))
    return {"clades": clades, "root_split": root,
            "correct_root": {frozenset(c) for c in root} == {frozenset(c) for c in parents},
            "clade_precision": precision, "clade_recall": recall,
            "clade_f1": 2*precision*recall/(precision+recall) if precision+recall else 0.}


def analyze(case, mode, channel):
    supports, primitive = channel.supports, channel.primitive
    ei = {s:channel.info(s,SOURCES,mode) for size in range(1,10) for s in itertools.combinations(SOURCES,size)}
    residuals = [ei[s]-ei[left]-ei[right] for s in ei if len(s)>1 for left,right in nontrivial_bipartitions(s)]
    violation = sum(r < -TOLERANCE for r in residuals)
    if violation:
        raise ArithmeticError(f"Syn violation: minimum={min(residuals)} bits, threshold={-TOLERANCE}, count={violation}")
    tree = build_spt_from_ei_table(SOURCES,ei,config=SPTConfig(syn_tolerance=TOLERANCE))
    nodes = flatten_nodes(tree.root)
    spt = quality(tuple(c.sources for c in tree.root.children),
                  [n.sources for n in nodes if 1<len(n.sources)<9],supports,primitive)
    if abs(tree.closure_error)>TOLERANCE:
        raise ArithmeticError(f"Closure failure: {tree.closure_error} bits")
    phi = np.zeros((9,9))
    peid = np.zeros((9,9))
    future = np.zeros((9,9))
    future_target_counts = channel.counts.sum(axis=0)
    def target_entropy(indices):
        selected = [i for i in range(len(supports)) if indices >> i & 1]
        projection=(channel.targets[:,selected]*(1 << np.arange(len(selected)))).sum(axis=1)
        return entropy(np.bincount(projection,weights=future_target_counts,minlength=1 << len(selected)))
    for a,b in itertools.combinations(SOURCES,2):
        ia,ib=(a,),(b,)
        redundancy=min(channel.info(u,v,mode) for u in (ia,ib) for v in (ia,ib))
        value=channel.info((a,b),(a,b),mode)-channel.info(ia,ia,mode)-channel.info(ib,ib,mode)+redundancy
        phi[a-1,b-1]=phi[b-1,a-1]=value
        peid[a-1,b-1]=peid[b-1,a-1]=ei[(a,b)]-ei[ia]-ei[ib]
        # In shared mode every shared channel has >=3 shares, so a pair sees
        # independent masks; copied primitive channels are its only readouts.
        va,vb=channel.visible(ia,mode),channel.visible(ib,mode)
        future[a-1,b-1]=future[b-1,a-1]=target_entropy(va)+target_entropy(vb)-target_entropy(va|vb)
    if np.min(phi)<-TOLERANCE:
        raise ArithmeticError(f"Pair PhiR negative: {phi.min()}")
    # Small entropy-cancellation roundoff is retained, with its count recorded.
    # The graph routine receives raw affinities; no max(0,.) projection is used.
    phi_root,phi_clades=graph_tree(phi.tolist())
    future_root,future_clades=graph_tree(future.tolist())
    block_clades,block_scores=[],[]
    def block_tree(source):
        if len(source)==1:
            return ()
        scored=[]
        for left,right in nontrivial_bipartitions(source):
            redundancy=min(channel.info(u,v,mode) for u in (left,right) for v in (left,right))
            value=channel.info(source,source,mode)-channel.info(left,left,mode)-channel.info(right,right,mode)+redundancy
            scored.append((value,left,right))
            block_scores.append(value)
        _,left,right=min(scored)
        for child in (left,right):
            if len(child)>1:
                block_clades.append(child)
            block_tree(child)
        return (left,right)
    block_root=block_tree(SOURCES)
    if min(block_scores)<-TOLERANCE:
        raise ArithmeticError(f"Block PhiR negative: {min(block_scores)}")
    truth_parents=tuple(h for h in supports if len(h)<9 and not any(set(h)<set(g)<set(SOURCES) for g in supports))
    root_scores=sorted((ei[l]+ei[r] for l,r in nontrivial_bipartitions(SOURCES)),reverse=True)
    return {"model_family":"heterogeneous", "case":case,"case_label":MODELS[case]["label"],
        "readout_mode":mode,"planted_bottom":primitive,"planted_parents":truth_parents,
        "planted_clades":tuple(h for h in supports if len(h)<9),"hyperedges":supports,
        "node_gates":GATES,"node_activation_probabilities":[p/4 for p in ACTIVATION_NUMERATORS],
        "node_memory_noise":[p/32 for p in MEMORY_NOISE_NUMERATORS],
        "node_memory_gates":MEMORY_GATES,
        "hyperedge_noise":[NOISE_BY_ORDER[len(h)]/16 for h in supports],
        "xi_bits":tree.root.xi_value,"root_syn_bits":tree.root.syn_value,
        "root_score_gap_bits":root_scores[0]-root_scores[1],
        **{k:v for k,v in spt.items() if k not in ('clades','root_split')},
        "pair_phi_r":phi.tolist(),"pair_peid":peid.tolist(),"pair_future_mi":future.tolist(),
        "phi_r_hac":quality(phi_root,phi_clades,supports,primitive),
        "future_mi_hac":quality(future_root,future_clades,supports,primitive),
        "block_phi_r":quality(block_root,block_clades,supports,primitive),
        "nodes":[{"sources":n.sources,"children":[c.sources for c in n.children],"depth":n.depth,
                  "xi_bits":n.xi_value,"syn_bits":n.syn_value} for n in nodes],
        "audit":{"tolerance_bits":TOLERANCE,"partition_count":len(residuals),
            "minimum_syn_bits":min(residuals),"tolerance_negative_count":sum(-TOLERANCE<=r<0 for r in residuals),
            "significant_violation_count":violation,"closure_error_bits":tree.closure_error,"clipping":False,
            "pair_phi_r_tolerance_negative_count":int(np.sum((phi<0)&(phi>=-TOLERANCE))),
            "probability_denominator":channel.denominator,"mi_queries":channel.query_count}}


def run():
    start=time.monotonic()
    channels={case:Channel(model["supports"]) for case,model in MODELS.items()}
    records=[analyze(case,mode,channels[case]) for mode in ("copied","secret_shared") for case in MODELS]
    return {"protocol":{"nodes":9,"bits_per_node":11,"source_bits":99,"target_bits":99,
        "intervention":"independent uniform full node states; activations retain their induced nonuniform law",
        "horizon":1,"gates":GATES,"activation_probabilities":[p/4 for p in ACTIVATION_NUMERATORS],
        "memory_flip_probabilities":[p/32 for p in MEMORY_NOISE_NUMERATORS],
        "memory_update":"m_next=m XOR Gamma(local_future_readout_fields) XOR node-specific noise",
        "memory_gates":MEMORY_GATES,
        "memory_gate_inputs":"first, last, middle incident hyperedge fields; first/last for two-input gates",
        "fresh_drivers":"two independent uniform bits per node each step; memory does not feed drivers/readout parity",
        "padded_readouts":"8 fields per node; inactive fields=0; current readout fields marginalized",
        "shared_policy":"minimal hyperedges copied; other hyperedges n-of-n XOR shared with independent masks",
        "estimator":"exact dyadic activation/readout joint plus analytic local-memory MI",
        "scoring":"mechanistic nonroot support F1; proper refinements inside minimal supports unscored for every method",
        "factor_layout":"3 different laminar architectures x 2 readout modes; heterogeneous node profile fixed",
        "scope":"synthetic development comparison, no architecture-isolated effect or finite-sample claim"},
        "elapsed_seconds":time.monotonic()-start,"records":records}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure",type=Path)
    args=parser.parse_args()
    result=run()
    if args.figure:
        from scripts.plot_spt_heterogeneous_hierarchy import plot
        plot(result["records"],args.figure)
    print(json.dumps(result,indent=2))
