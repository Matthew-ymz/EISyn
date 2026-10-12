"""Composite figures for the exact hierarchy benchmarks (no Kuramoto refit).

Evidence: all 3 parent pairings x both readout modes from
spt_multilevel_parity_case.py or spt_nested_hypergraph_case.py.
Optional --input reuses either native JSON output;
without it, the same small exact finite model is recomputed. All conditions and
default HAC ties are retained. Representative trees use fixed pairing A.

Panels: planted supports drawn as a node/hyperedge network; identical pair PhiR
matrix and mean; eight actual trees; per-case clade F1. Upper supports are regions,
not clique-expanded pair edges. The eight-node plot rearranges node positions
by parent membership. The nine-node plot fixes positions across all cases and
uses nested 3/4/5-node polygons. F1 excludes the root and leaves; the nine-node
metric also ignores unidentifiable proper refinements of primitive triads.
Syn tolerances come from the native
records; raw values are not clipped. Only the matrix diagonal is display-masked.

Manuscript freshly rediscovered 2026-10-12: Zotero P6UJCVG8, main DXGC7JEA,
SI MWIWKSVG; Methods (5)-(12), SI S5/Algorithm S1. No explicit draft version;
attachment metadata does not resolve the version ambiguity. Both finite searches
is exact, and the figure does not claim scalable approximate search recovery.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Rectangle, Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_multilevel_parity_case import BOTTOM, MATCHINGS, SOURCES, analyze

OUT = ROOT / "docs/reports/assets/kuramoto_hierarchy/spt_hierarchy_comparison.png"
METHODS = ("spt", "phi_r_hac", "future_mi_hac", "block_phi_r")
NAMES = {"spt": "SPT", "phi_r_hac": r"Pairwise $\Phi_R$ + HAC",
         "future_mi_hac": "Future MI + HAC", "block_phi_r": r"Block-$\Phi_R$*"}
BLUE, GREEN, RED = "#286DA0", "#217B68", "#B74738"
INK, GRAY = "#24323C", "#87939C"


def load(input_path):
    if input_path:
        records = json.loads(input_path.read_text())["records"]
    else:
        records = [analyze(m, mode) for mode in ("copied", "secret_shared") for m in MATCHINGS]
    ordered = []
    for mode in ("copied", "secret_shared"):
        for matching in MATCHINGS:
            found = [r for r in records if r["readout_mode"] == mode
                     and tuple(map(tuple, r["matching"])) == matching]
            if len(found) != 1:
                raise ValueError(f"Expected one record for {mode}, {matching}")
            r = found[0]
            audit = r["audit"]
            tolerance = audit["tolerance_bits"]
            if audit["significant_violation_count"] or audit["minimum_syn_bits"] < -tolerance:
                raise ArithmeticError(f"Syn audit failed: minimum={audit['minimum_syn_bits']} bits, "
                                      f"threshold={-tolerance} bits, count={audit['significant_violation_count']}")
            if audit["clipping"] or abs(audit["closure_error_bits"]) > tolerance:
                raise ArithmeticError("Clipped or unclosed hierarchy input")
            ordered.append(r)
    if len(records) != 6:
        raise ValueError("Figure requires exactly the six native development cases")
    for key in ("pair_phi_r", "pair_peid"):
        matrices = np.array([r[key] for r in ordered])
        if np.max(np.abs(matrices - matrices[0])) != 0:
            raise ValueError(f"Claim of identical {key} matrices is unsupported")
    return ordered


def method_result(record, method):
    if method == "spt":
        return {"clades": [n["sources"] for n in record["nodes"] if 1 < len(n["sources"]) < len(record["pair_phi_r"])],
                "root_split": next(n["children"] for n in record["nodes"] if len(n["sources"]) == len(record["pair_phi_r"])),
                "correct_root": record["correct_root"], "clade_f1": record["clade_f1"]}
    return record[method]


def recall(record, method, truth):
    estimated = {frozenset(s) for s in method_result(record, method)["clades"]}
    return len(estimated & {frozenset(s) for s in truth}) / len(truth)


def header(fig, axes, letter, title):
    positions = [ax.get_subplotspec().get_position(fig) for ax in axes]
    x, y = min(p.x0 for p in positions), max(p.y1 for p in positions)
    fig.text(x, y + .017, f"{letter}  {title}", ha="left", va="bottom",
             fontsize=12.4, weight="bold", color=INK)


def ground_truth_network(ax, record):
    """Draw the exact 2-, 4- and 8-node parity supports; no added graph edges."""
    if len(record["pair_phi_r"]) == 9:
        return nested_hyperedge_network(ax, record)
    positions = {}
    parent_colors = (GREEN, "#C47E32")
    ax.add_patch(FancyBboxPatch((-1.98, -1.28), 3.96, 2.56,
        boxstyle="round,pad=.08,rounding_size=.18", fc="#F7F8F9", ec=GRAY,
        lw=1.0, linestyle=(0, (4, 3)), zorder=0))
    for parent_index, parent in enumerate(record["planted_parents"]):
        center = -1.18 if parent_index == 0 else 1.18
        color = parent_colors[parent_index]
        ax.add_patch(FancyBboxPatch((center - .67, -1.08), 1.34, 2.16,
            boxstyle="round,pad=.02,rounding_size=.16", fc=color, ec=color,
            alpha=.16, lw=1.2, zorder=1))
        pairs = [pair for pair in BOTTOM if set(pair) <= set(parent)]
        if len(pairs) != 2:
            raise ValueError(f"Parent does not contain exactly two planted pairs: {parent}")
        for pair_index, pair in enumerate(pairs):
            y = .80 if pair_index == 0 else -.80
            for i, x in zip(pair, (center - .43, center + .43)):
                positions[i] = (x, y)
            a, b = (positions[i] for i in pair)
            ax.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=2.5, zorder=2)
        for i in parent:
            x, y = positions[i]
            ax.add_patch(Circle((x, y), .15, fc="white", ec=color, lw=1.3, zorder=3))
            ax.text(x, y, str(i), ha="center", va="center", fontsize=11.2, color=INK, zorder=4)
    if set(positions) != set(SOURCES):
        raise ValueError("Network omitted a source node")
    ax.set(xlim=(-2.22, 2.22), ylim=(-1.50, 1.50), aspect="equal")
    ax.axis("off")


def nested_hyperedge_network(ax, record):
    """Fixed node coordinates: a primitive triad nested in a 5-node hyperedge.

    Convex padded regions enclose exactly their supports. They denote one
    multivariate parity channel, not a clique of physical pairwise edges.
    Position, node number and primitive-block membership are identical in A-C;
    only parent hyperedges change. No arbitrary node reordering hides differences.
    """
    from scipy.spatial import ConvexHull

    positions = {1: (-.27, .13), 2: (.27, .13), 3: (0., -.29)}
    for pair, degrees in zip(record["planted_bottom"][1:], (90, -30, 210)):
        theta = np.deg2rad(degrees)
        center = 1.34 * np.array([np.cos(theta), np.sin(theta)])
        tangent = .24 * np.array([-np.sin(theta), np.cos(theta)])
        positions[pair[0]], positions[pair[1]] = center - tangent, center + tangent

    def region(support, color, pad, opacity, linewidth, zorder):
        # Minkowski dilation: round corners without enclosing unlisted nodes.
        directions = np.linspace(0, 2*np.pi, 48, endpoint=False)
        offsets = pad * np.c_[np.cos(directions), np.sin(directions)]
        cloud = np.concatenate([np.asarray(positions[i]) + offsets for i in support])
        boundary = cloud[ConvexHull(cloud).vertices]
        from matplotlib.path import Path as MplPath
        inside = {i for i, point in positions.items() if MplPath(boundary).contains_point(point)}
        if inside != set(support):
            raise ValueError(f"Hyperedge region encloses {inside}, expected {support}")
        ax.add_patch(Polygon(boundary, fc=color, ec="none", alpha=opacity, zorder=zorder))
        ax.add_patch(Polygon(boundary, fill=False, ec=color, lw=linewidth, zorder=zorder+.1))

    ax.add_patch(Circle((0, 0), 1.57, fc="#FBFCFD", ec=GRAY, lw=.9,
                       linestyle=(0, (4, 3)), zorder=0))
    for parent in sorted(record["planted_parents"], key=len, reverse=True):
        region(parent, "#BE7727" if len(parent) == 5 else GREEN, .16, .15, 1.45, 1)
    region(record["planted_bottom"][0], BLUE, .135, .21, 1.75, 2)
    for pair in record["planted_bottom"][1:]:
        a, b = (positions[i] for i in pair)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=2.4, zorder=3)
    for i, (x, y) in positions.items():
        ax.add_patch(Circle((x, y), .11, fc="white", ec=BLUE, lw=1.15, zorder=4))
        ax.text(x, y, str(i), ha="center", va="center", fontsize=11.1, color=INK, zorder=5)
    ax.set(xlim=(-1.64, 1.64), ylim=(-1.64, 1.64), aspect="equal")
    ax.axis("off")


def tree(ax, record, method="spt", truth_only=False):
    """Reconstruct exactly the laminar tree from native clades and singleton leaves."""
    sources = tuple(range(1, len(record["pair_phi_r"]) + 1))
    bottom = tuple(map(tuple, record.get("planted_bottom", BOTTOM)))
    result = method_result(record, method)
    clades = {tuple(sorted(c)) for c in result["clades"]}
    all_sets = clades | {sources} | {(i,) for i in sources}
    children = {}
    for s in all_sets:
        if len(s) == 1:
            children[s] = ()
            continue
        proper = [c for c in all_sets if set(c) < set(s)]
        maximal = [c for c in proper if not any(set(c) < set(d) for d in proper)]
        if len(maximal) != 2 or set().union(*map(set, maximal)) != set(s):
            raise ValueError(f"Incomplete binary tree at {s}: {maximal}")
        children[s] = tuple(sorted(maximal))
    positions, count = {}, [0]

    def visit(s, depth=0):
        if children[s]:
            x = np.mean([visit(c, depth + 1) for c in children[s]])
        else:
            x = count[0]
            count[0] += 1
        positions[s] = (x, -depth)
        return x

    visit(sources)
    truth = {tuple(sorted(c)) for c in record.get("planted_clades", (*bottom, *record["planted_parents"]))}
    def unresolved(c):
        return any(set(c) < set(block) for block in bottom)
    for s in sorted(all_sets, key=lambda s: -len(s)):
        x, y = positions[s]
        for c in children[s]:
            cx, cy = positions[c]
            wrong = len(c) > 1 and c not in truth and not unresolved(c)
            ax.plot([x, cx], [y, cy], color=RED if wrong else "#ABB6BF",
                    linewidth=1.15, linestyle="--" if wrong else "-", zorder=0)
        if s == sources:
            edge, face, label = "#78858F", "#EFF2F4", f"1–{len(sources)}"
        elif len(s) == 1:
            edge, face, label = "#B7C0C6", "white", str(s[0])
        elif unresolved(s):
            edge, face, label = "#B7C0C6", "white", "".join(map(str, s))
        else:
            edge, face = ((BLUE, "#EAF3F9") if len(s) == 2 else (GREEN, "#E5F1EB")) if s in truth else (RED, "#FCEFEA")
            label = "".join(map(str, s))
        ax.text(x, y, label, ha="center", va="center", fontsize=10.6 if truth_only else 10.3,
                color=INK, bbox={"boxstyle": "round,pad=.23", "fc": face, "ec": edge, "lw": 1.0})
    depth = min(y for _, y in positions.values())
    ax.set(xlim=(-.40, len(sources) - .60), ylim=(depth - .22, .26))
    ax.axis("off")


def plot(records, path):
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .8, "xtick.color": INK, "ytick.color": INK,
                         "savefig.facecolor": "white"}):
        nested = len(records[0]["pair_phi_r"]) == 9
        count = len(records[0]["pair_phi_r"])
        fig = plt.figure(figsize=(18, 8.9 if nested else 8.3), facecolor="white")
        # Two bands; scores beside the two tree rows. No empty bottom panels.
        gs = fig.add_gridspec(2, 1, height_ratios=[1.15 if nested else .90, 2.12], left=.050,
                             right=.985, top=.937, bottom=.057, hspace=.23)
        top = gs[0].subgridspec(1, 3, width_ratios=[2.1, .82, .98], wspace=.21)
        truth_grid = top[0].subgridspec(1, 3, wspace=.09)
        truths = [fig.add_subplot(truth_grid[i]) for i in range(3)]
        for i, ax in enumerate(truths):
            ground_truth_network(ax, records[i])
            ax.text(.5, 1.02, "ABC"[i], transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=11.2, weight="bold")
        header(fig, truths, "a", "Ground truth")
        if not nested:
            truths[-1].legend(handles=[Line2D([], [], color=BLUE, lw=2.5, label="2-node"),
                                  Patch(fc="#DCEBE4", ec=GREEN, label="4-node"),
                                  Line2D([], [], color=GRAY, linestyle="--", label="8-node")],
            loc="lower right", bbox_to_anchor=(1, 1.18), ncol=3, frameon=False,
            fontsize=9.3, handlelength=1.2, handletextpad=.4, columnspacing=.8,
            borderaxespad=0)

        if nested:
            fig.legend(handles=[Patch(fc="#DCEAF5", ec=BLUE, label="3-node"),
                Patch(fc="#D8EBE3", ec=GREEN, label="4-node"),
                Patch(fc="#F4E3CE", ec="#BE7727", label="5-node"),
                Line2D([], [], color=GRAY, linestyle="--", label="9-node")],
                loc="upper center", bbox_to_anchor=(.277, .998), ncol=4, frameon=False,
                fontsize=9.1, handlelength=1.1, handletextpad=.35, columnspacing=.6,
                borderaxespad=0)

        matrix_ax = fig.add_subplot(top[1])
        data = np.array(records[0]["pair_phi_r"], dtype=float)
        matrix = np.ma.array(data, mask=np.eye(count, dtype=bool))
        cmap = LinearSegmentedColormap.from_list("affinity", ["#FAFCFE", "#296EAA"])
        cmap.set_bad("#E5E9EC")
        im = matrix_ax.imshow(matrix, cmap=cmap, vmin=0, vmax=data.max(),
                              interpolation="nearest", aspect="equal")
        colorbar = fig.colorbar(im, ax=matrix_ax, fraction=.045, pad=.05,
                               ticks=[0, data.max()])
        colorbar.ax.set_yticklabels(["0", f"{data.max():.3f}"])
        colorbar.ax.tick_params(labelsize=9.0, length=2)
        for pair in records[0].get("planted_bottom", BOTTOM):
            if len(pair) != 2:
                continue
            first, second = pair
            matrix_ax.add_patch(Rectangle((first - 1.5, first - 1.5), 2, 2,
                                         fill=False, ec=BLUE, lw=1.0))
        matrix_ax.set_xticks(range(count), range(1, count + 1))
        matrix_ax.set_yticks(range(count), range(1, count + 1))
        matrix_ax.tick_params(labelsize=9.1, length=2)
        header(fig, [matrix_ax], "b", r"Pairwise $\Phi_R$ (bits)")

        mean_ax = fig.add_subplot(top[2])
        means = np.array([r["mean_pair_phi_r_bits"] for r in records]).reshape(2, 3)
        mean_cmap = LinearSegmentedColormap.from_list("mean_values", ["#FAFCFE", "#D7E7F2"])
        mean_ax.imshow(means, cmap=mean_cmap, vmin=0, vmax=.035, aspect="auto",
                       interpolation="nearest")
        for i, j in itertools.product(range(2), range(3)):
            mean_ax.text(j, i, f"{means[i,j]:.5f}", ha="center", va="center",
                         fontsize=10.8, color=INK)
        mean_ax.set_xticks(range(3), list("ABC"))
        mean_ax.set_yticks(range(2), ["Copied", "Shared"])
        mean_ax.tick_params(length=0, labelsize=10.1)
        mean_ax.set_xticks([.5, 1.5], minor=True)
        mean_ax.set_yticks([.5], minor=True)
        mean_ax.grid(which="minor", color="white", linewidth=2)
        mean_ax.tick_params(which="minor", length=0)
        for spine in mean_ax.spines.values():
            spine.set_visible(False)
        header(fig, [mean_ax], "c", r"Mean $\Phi_R$ (bits)")

        bottom = gs[1].subgridspec(1, 2, width_ratios=[3.85, 1.12], wspace=.17)
        tree_grid = bottom[0].subgridspec(2, 4, hspace=.17, wspace=.14)
        tree_rows = []
        for row, (record, label) in enumerate(((records[0], "Copied"), (records[3], "Shared"))):
            axes = [fig.add_subplot(tree_grid[row, i]) for i in range(4)]
            for ax, method in zip(axes, METHODS):
                tree(ax, record, method)
                if row == 0:
                    ax.set_title(NAMES[method], fontsize=12.2, weight="bold", pad=6,
                                 color=GREEN if method == "spt" else INK)
            axes[0].text(-.065, .5, label, transform=axes[0].transAxes,
                         ha="right", va="center", rotation=90, fontsize=11.3,
                         color="#57636D")
            tree_rows.append(axes)
        p = tree_rows[0][0].get_position()
        fig.text(p.x0 - .018, p.y1 + .015, "d", fontsize=12.4, weight="bold", color=INK)
        fig.legend(handles=[Patch(fc="#EAF3F9", ec=BLUE, label="True 2-node" if nested else "True pair"),
                            Patch(fc="#E5F1EB", ec=GREEN, label="True 3/4/5-node" if nested else "True upper clade"),
                            Patch(fc="#FCEFEA", ec=RED, label="Wrong clade")],
                   loc="lower center", bbox_to_anchor=(.40, .005), ncol=3,
                   frameon=False, fontsize=10.0, handlelength=1.2, columnspacing=1.1)
        if nested:
            tree_rows[-1][0].text(.50, -.15, "Gray: unresolved triad refinements", transform=tree_rows[-1][0].transAxes,
                ha="center", va="top", color=GRAY, fontsize=8.1)

        score_ax = fig.add_subplot(bottom[1])
        # Transposed: six case rows, four method columns, beside the tree examples.
        score = np.array([[method_result(r, m)["clade_f1"] for m in METHODS] for r in records])
        f1_cmap = LinearSegmentedColormap.from_list("recovery", ["#FCF5F0", "#C5DECF", "#1D7967"])
        score_ax.imshow(score, cmap=f1_cmap, vmin=0, vmax=1,
                        aspect="auto", interpolation="nearest")
        for i, j in itertools.product(range(6), range(4)):
            score_ax.text(j, i, f"{score[i,j]:.2f}", ha="center", va="center", fontsize=11.3,
                          color="white" if score[i,j] > .82 else INK,
                          weight="bold" if j == 0 else "normal")
        score_ax.set_xticks(range(4), ["SPT", "Pair $\Phi_R$\n+ HAC", "Future MI\n+ HAC", "Block $\Phi_R$*"])
        score_ax.set_yticks(range(6), [f"{mode} {case}" for mode in ("Copied", "Shared") for case in "ABC"])
        score_ax.tick_params(axis="x", top=True, labeltop=True, bottom=False,
                             labelbottom=False, length=0, labelsize=9.2, pad=7)
        score_ax.tick_params(axis="y", length=0, labelsize=10.1)
        score_ax.axhline(2.5, color="white", linewidth=3)
        for spine in score_ax.spines.values():
            spine.set_visible(False)
        p = score_ax.get_position()
        fig.text(p.x0 - .012, p.y1 + .026, "e", fontsize=12.4, weight="bold", color=INK)
        score_ax.set_xlabel("Clade F1" + ("*" if nested else ""), labelpad=9, fontsize=11.0)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches=None, facecolor="white")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, help="Native JSON output from either exact hierarchy case script")
    p.add_argument("--output", type=Path, default=OUT)
    args = p.parse_args()
    records = load(args.input)
    plot(records, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
