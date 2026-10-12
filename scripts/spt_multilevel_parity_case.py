"""Exact three-level Markov example whose pair matrices hide the upper hierarchy.

Eight nodes each have a four-bit state (fresh driver and three readout fields).
Independent uniform interventions cover all 32 source bits. Future drivers refresh
independently. Compare copied ancestor readouts with parent/root XOR secret shares.
All EI queries retain the complete 32-bit future target. Ignored input bits,
independent future drivers/sharing masks and duplicate readouts integrate out exactly;
the remaining exact finite joint has four latent parities and seven noisy bits.
"""
from __future__ import annotations

from fractions import Fraction
import argparse
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_pairwise_parity_case import mutual_information
from scripts.spt import SPTConfig, build_spt_from_ei_table, flatten_nodes, nontrivial_bipartitions

SOURCES = tuple(range(1, 9))
BOTTOM = ((1, 2), (3, 4), (5, 6), (7, 8))
MATCHINGS = (((1, 2), (3, 4)), ((1, 3), (2, 4)), ((1, 4), (2, 3)))
NOISE = (Fraction(1, 4),) * 4 + (Fraction(1, 10),) * 3
TOLERANCE = 1e-10  # bits; raw tolerance-scale negatives are retained and counted.


def joint_rows(matching):
    rows = []
    for z in itertools.product((0, 1), repeat=4):
        noiseless = (*z, *(sum(z[i - 1] for i in parent) % 2 for parent in matching), sum(z) % 2)
        for flips in itertools.product((0, 1), repeat=7):
            p = Fraction(1, 16)
            for flip, noise in zip(flips, NOISE):
                p *= noise if flip else 1 - noise
            rows.append((z, tuple(v ^ e for v, e in zip(noiseless, flips)), p))
    assert sum(p for _, _, p in rows) == 1
    return rows


def graph_tree(matrix):
    import numpy as np
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform

    weights = np.asarray(matrix, dtype=float)
    distance = 1 - weights / weights.max()
    np.fill_diagonal(distance, 0.)
    root = to_tree(linkage(squareform(distance, checks=True), method="average"))
    clades = []

    def visit(node):
        if node.is_leaf():
            return (node.id + 1,)
        left, right = visit(node.left), visit(node.right)
        combined = tuple(sorted((*left, *right)))
        if len(combined) < len(matrix):
            clades.append(combined)
        return combined

    left, right = visit(root.left), visit(root.right)
    return (left, right), clades


def quality(root, clades, parents, bottom=BOTTOM):
    truth = {frozenset(c) for c in (*bottom, *parents)}
    # Proper refinements of a primitive parity block are unidentifiable. Ignore
    # them for every method, without looking at the estimated tree or its score.
    found = {frozenset(c) for c in clades
             if not any(set(c) < set(block) for block in bottom)}
    p, r = len(found & truth) / len(found), len(found & truth) / len(truth)
    return {"correct_root": {frozenset(c) for c in root} == {frozenset(c) for c in parents},
            "clade_precision": p, "clade_recall": r, "clade_f1": 2*p*r/(p+r) if p+r else 0.,
            "root_split": root, "clades": clades}


def analyze(matching, mode, *, bottom=BOTTOM, sources=SOURCES):
    """Four primitive parity blocks; their node counts may be heterogeneous."""
    if mode not in ("copied", "secret_shared"):
        raise ValueError(mode)
    if len(bottom) != 4 or set().union(*map(set, bottom)) != set(sources) or sum(map(len, bottom)) != len(sources):
        raise ValueError("Four disjoint primitive blocks must cover the sources")
    rows = joint_rows(matching)
    target = tuple(range(1, 8))
    # Selected original nodes reveal exactly the complete primitive parities.
    latent_ei = {s: mutual_information(rows, s, target)
                 for size in range(5) for s in itertools.combinations(range(1, 5), size)}
    ei = {}
    for size in range(1, len(sources) + 1):
        for subset in itertools.combinations(sources, size):
            revealed = tuple(i + 1 for i, block in enumerate(bottom) if set(block) <= set(subset))
            ei[subset] = latent_ei[revealed]
    assert all(ei[(i,)] == 0 for i in sources)
    syn = [ei[s] - ei[left] - ei[right] for s in ei if len(s) > 1
           for left, right in nontrivial_bipartitions(s)]
    count = sum(v < -TOLERANCE for v in syn)
    if count:
        raise ArithmeticError(f"Syn violation: minimum={min(syn)} bits, threshold={-TOLERANCE} bits, count={count}")
    result = build_spt_from_ei_table(sources, ei, config=SPTConfig(syn_tolerance=TOLERANCE))
    nodes = flatten_nodes(result.root)
    parents = tuple(tuple(sorted(i for j in group for i in bottom[j - 1])) for group in matching)
    truth = {frozenset(c) for c in (*bottom, *parents)}
    found = {frozenset(n.sources) for n in nodes if 1 < len(n.sources) < len(sources)
             and not any(set(n.sources) < set(block) for block in bottom)}
    root_correct = {frozenset(c.sources) for c in result.root.children} == {frozenset(c) for c in parents}
    gaps = []
    for node in nodes:
        if any(set(node.sources) <= set(block) for block in bottom):
            continue
        scores = sorted((ei[left] + ei[right] for left, right in nontrivial_bipartitions(node.sources)), reverse=True)
        gaps.append({"sources": node.sources, "best_second_gap_bits": scores[0] - scores[1]})
    if not root_correct or found != truth or min(g["best_second_gap_bits"] for g in gaps) <= TOLERANCE:
        raise ArithmeticError(f"Planted hierarchy not uniquely recovered: {matching}, gaps={gaps}")
    if abs(result.closure_error) > TOLERANCE:
        raise ArithmeticError(f"Closure error: {result.closure_error} bits")
    phi = [[0.] * len(sources) for _ in sources]
    peid = [[0.] * len(sources) for _ in sources]
    for j, pair in enumerate(bottom, start=1):
        if len(pair) != 2:
            continue  # A pair source cannot reveal a primitive triad parity.
        # Single-node source information is zero; MMI redundancy is therefore zero.
        # Across incomplete bottom pairs, independent missing partners mask both sources.
        # Secret-share ancestor fields in two-node targets are uniform nuisance.
        # Copied ancestors are jointly independent of this one latent parity.
        local = mutual_information(rows, (j,), (j,))
        for a, b in itertools.permutations(pair, 2):
            phi[a - 1][b - 1] = local
            peid[a - 1][b - 1] = ei[pair]
    def revealed(sources):
        return tuple(j + 1 for j, pair in enumerate(bottom) if set(pair) <= set(sources))

    def target_bits(target_sources):
        selected = set(target_sources)
        bottom_bits = tuple(j + 1 for j, pair in enumerate(bottom) if selected.intersection(pair))
        parent_bits = tuple(g + 5 for g, parent in enumerate(parents)
            if (selected.intersection(parent) if mode == "copied" else set(parent) <= selected))
        root_bits = (7,) if (bool(selected) if mode == "copied" else len(selected) == len(sources)) else ()
        return (*bottom_bits, *parent_bits, *root_bits)

    cache = {(s, target): v for s, v in latent_ei.items()}

    def info(source, target_sources):
        key = (revealed(source), target_bits(target_sources))
        if key not in cache:
            cache[key] = mutual_information(rows, *key) if key[0] and key[1] else 0.
        return cache[key]

    block_clades = []
    block_candidates = []

    def block_tree(sources):
        if len(sources) == 1:
            return ()
        scored = []
        for left, right in nontrivial_bipartitions(sources):
            redundancy = min(info(u, v) for u in (left, right) for v in (left, right))
            value = info(sources, sources) - info(left, left) - info(right, right) + redundancy
            block_candidates.append(value)
            scored.append((value, left, right))
        _, left, right = min(scored)
        for child in (left, right):
            if len(child) > 1:
                block_clades.append(child)
            block_tree(child)
        return (left, right)

    block_root = block_tree(sources)
    block_clades = [c for c in block_clades if len(c) < len(sources)]
    if min(block_candidates) < -TOLERANCE:
        raise ArithmeticError(f"Block PhiR violation: minimum={min(block_candidates)} bits, "
            f"threshold={-TOLERANCE} bits, count={sum(v < -TOLERANCE for v in block_candidates)}")
    future_mi = [[0.] * len(sources) for _ in sources]
    from collections import defaultdict
    future_probabilities = defaultdict(Fraction)
    for _, y, p in rows:
        future_probabilities[y] += p
    future_rows = [(y, y, p) for y, p in future_probabilities.items()]
    for a, b in itertools.combinations(sources, 2):
        if mode == "secret_shared":
            value = 1. if any(a in block and b in block for block in bottom) else 0.
        else:
            value = mutual_information(future_rows, target_bits((a,)), target_bits((b,)))
        future_mi[a-1][b-1] = future_mi[b-1][a-1] = value
    phi_root, phi_clades = graph_tree(phi)
    future_root, future_clades = graph_tree(future_mi)
    return {"readout_mode": mode, "matching": matching, "planted_parents": parents, "correct_root": root_correct,
            "clade_precision": len(found & truth) / len(found), "clade_recall": len(found & truth) / len(truth),
            "clade_f1": 1., "xi_bits": ei[sources], "root_syn_bits": result.root.syn_value,
            "root_score_gap_bits": gaps[0]["best_second_gap_bits"], "node_score_gaps": gaps,
            "mean_pair_phi_r_bits": math.fsum(phi[a][b] for a in range(len(sources)) for b in range(a + 1, len(sources))) / math.comb(len(sources), 2),
            "pair_phi_r": phi, "pair_peid": peid, "pair_future_mi": future_mi,
            "phi_r_hac": quality(phi_root, phi_clades, parents, bottom),
            "future_mi_hac": quality(future_root, future_clades, parents, bottom),
            "block_phi_r": {**quality(block_root, block_clades, parents, bottom),
                            "distinct_latent_mi_queries": len(cache),
                            "minimum_candidate_phi_r_bits": min(block_candidates)},
            "nodes": [{"sources": n.sources, "depth": n.depth, "xi_bits": n.xi_value,
                       "syn_bits": n.syn_value, "children": [c.sources for c in n.children]} for n in nodes],
            "audit": {"tolerance_bits": TOLERANCE, "partition_count": len(syn), "minimum_syn_bits": min(syn),
                      "tolerance_negative_count": sum(-TOLERANCE <= v < 0 for v in syn),
                      "significant_violation_count": count, "closure_error_bits": result.closure_error,
                      "clipping": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    records = [analyze(m, mode) for mode in ("copied", "secret_shared") for m in MATCHINGS]
    differences = {kind: max(abs(r[kind][i][j] - records[0][kind][i][j])
                            for r in records for i in range(8) for j in range(8))
                   for kind in ("pair_phi_r", "pair_peid")}
    if max(differences.values()) > TOLERANCE:
        raise ArithmeticError(f"Pair matrix invariance failed: {differences}")
    secret = [r for r in records if r["readout_mode"] == "secret_shared"]
    future_difference = max(abs(r["pair_future_mi"][i][j] - secret[0]["pair_future_mi"][i][j])
                            for r in secret for i in range(8) for j in range(8))
    if future_difference > TOLERANCE:
        raise ArithmeticError(f"Secret-shared future pair matrix differs: {future_difference}")
    if args.figure:
        plot_case(secret[0], args.figure)
    print(json.dumps({"protocol": {"nodes": 8, "bits_per_node": 4,
          "source": "independent Uniform({0,1}^4) for all eight node states",
          "target": "complete future 32-bit state; independent future driver bits included",
          "horizon": 1, "bottom_noise": .25, "parent_noise": .1, "root_noise": .1,
          "estimator": "exact rational probabilities with analytically justified latent reduction",
          "fresh_drivers": "independent Bernoulli(1/2) every time step; readouts do not drive the next update"},
          "readout_modes": ["copied", "secret_shared"],
          "sharing": "for each parent draw three independent uniform shares and fix the fourth XOR; for root draw seven and fix eighth; all 13 masks independent",
          "hac_ties": "SciPy average linkage default, fixed node labels 1..8; tie outcomes are not evidence of upper hierarchy",
          "block_phi_r_algorithm": "experimental recursive minimum unnormalized MMI PhiR adaptation; local future target changes with node",
          "case_count": 6, "recovered_root_count": sum(r["correct_root"] for r in records),
          "matrix_only_root_identification_bound": 1 / 3, "pair_matrix_max_difference_bits": differences,
          "secret_shared_future_pair_max_difference_bits": future_difference,
          "records": records}, indent=2))


def plot_case(record, path):
    import matplotlib.pyplot as plt
    import numpy as np

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        fig, (tree_ax, matrix_ax) = plt.subplots(1, 2, figsize=(12, 4.9),
            gridspec_kw={"width_ratios": [1.8, 1]}, constrained_layout=True)
        lookup = {tuple(n["sources"]): n for n in record["nodes"]}
        positions = {}
        count = [0]

        def layout(sources):
            node = lookup[tuple(sources)]
            if node["children"]:
                x = np.mean([layout(c) for c in node["children"]])
            else:
                x = count[0]
                count[0] += 1
            positions[tuple(sources)] = (x, -node["depth"])
            return x

        layout(SOURCES)
        for node in record["nodes"]:
            x, y = positions[tuple(node["sources"])]
            for child in node["children"]:
                cx, cy = positions[tuple(child)]
                tree_ax.plot([x, cx], [y, cy], color="#A9B5BE", linewidth=1.1, zorder=0)
            label = ",".join(map(str, node["sources"]))
            if node["children"]:
                label += f"\nSyn {100 * node['syn_bits'] / record['xi_bits']:.1f}%"
            tree_ax.text(x, y, label, ha="center", va="center", fontsize=8.5,
                bbox={"boxstyle": "round,pad=.35", "facecolor": "#DDEFE8" if node["children"] else "#F4F6F8",
                      "edgecolor": "#478E80" if node["children"] else "#A0ACB6"})
        tree_ax.set(xlim=(-.65, 7.65), ylim=(-3.55, .55))
        tree_ax.axis("off")
        tree_ax.set_title("a  SPT recovers the planted three-level hierarchy", loc="left", weight="bold", pad=20)
        matrix = np.asarray(record["pair_phi_r"])
        im = matrix_ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max(), interpolation="nearest")
        matrix_ax.set_anchor("N")
        matrix_ax.set(xticks=range(8), yticks=range(8), xticklabels=SOURCES, yticklabels=SOURCES,
                      xlabel="Node", ylabel="Node")
        for i, j in zip(*np.nonzero(matrix)):
            matrix_ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=7.5, color="white")
        matrix_ax.set_title(r"b  Pairwise $\Phi_R$ preserves only the bottom pairs", loc="left", weight="bold", pad=20)
        colorbar = fig.colorbar(im, ax=matrix_ax, fraction=.046, pad=.05)
        colorbar.set_label(r"$\Phi_R$ (bits)")
        colorbar.set_ticks([0, .1, matrix.max()], labels=["0", "0.100", f"{matrix.max():.3f}"])
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
