#!/usr/bin/env python3
"""Show ranked SPT source coalitions as cortical membership maps plus local Syn.

This is a cache-only visualization: no simulation, EI estimation, or SPT search.
Default input is the unconstrained, ROI-block SPT displayed in the G=1.3,
seed-3 tree example. All internal nodes are eligible; nested/overlapping nodes
are retained. Ranking uses local syn_bits_raw, never subtree Xi or Shapley values.
Each ROI contains its E/I source coordinates; the fixed future target is the
complete 200-dimensional state at 300 ms. Cached EI uses a Gaussian conditional
covariance approximation to the common independent U(0.3,0.7) intervention.
Large-node splits were optimized within spectral candidates, not exhaustively.

Method reference checked 2026-10-06: Zotero P6UJCVG8, main DXGC7JEA,
Methods pp.15-17, Eqs.(5)-(12); supplement MWIWKSVG, S5 and S12.2.1.
Neither attachment identifies a manuscript revision/date; these are the only
available main/supplement attachments, so exact version remains unspecified.
Local Syn is the residual between the selected children, not a pure-order PID
atom. ROI count is a block count, not the number of scalar microscopic sources.
Anatomical mapping matches exact Schaefer parcel names. The original structural
connectome row order remains inferred, as in the existing cortical plots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.brain_surface_plot import SurfaceMesh, _draw_surface, _face_colors, parcel_values_to_vertices

DEFAULT_TREE = ROOT / "results/dmf_schaefer100/xi_hierarchy_tree/examples/wide/seed03_G1.30.json"
DEFAULT_SURFACE = ROOT / "results/dmf_schaefer100/schaefer100_fsaverage5_surface.npz"
DEFAULT_LABELS = ROOT / "results/dmf_schaefer100/schaefer100_labels.txt"
DEFAULT_OUTPUT = ROOT / "fig/dmf_schaefer100/dmf_spt_top_coalitions_surface.png"
NATS_PER_BIT = np.log(2.0)
MEMBER_COLOR = "#238B87"
NONMEMBER_COLOR = "#DCDCDC"


def ranked_nodes(record: dict, roi_count: int, top: int) -> tuple[list[dict], dict]:
    """Validate cached residuals before selecting all-node descending ranks."""
    tolerance = float(record["config"]["tolerance_bits"])
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Cached Syn tolerance must be finite and nonnegative (bits).")
    nodes, leaves = [], []

    def visit(node: dict) -> None:
        indices = tuple(int(i) for i in node["indices"])
        if len(indices) != node["size"] or len(set(indices)) != len(indices):
            raise ValueError("SPT node has inconsistent ROI membership/size.")
        if any(i < 0 or i >= roi_count for i in indices):
            raise ValueError("SPT index is outside the anatomical label order.")
        children = node["children"]
        if not children:
            if len(indices) != 1:
                raise ValueError("Expected one ROI block per SPT leaf.")
            leaves.append(node)
            return
        if len(children) != 2:
            raise ValueError("SPT must be binary.")
        left, right = (set(c["indices"]) for c in children)
        if left & right or left | right != set(indices):
            raise ValueError("Children do not form a disjoint partition of the parent.")
        syn = float(node["syn_bits_raw"])
        expected = float(node["xi_bits"]) - sum(float(c["xi_bits"]) for c in children)
        if not np.isfinite(syn) or abs(syn - expected) > tolerance:
            raise ArithmeticError("Cached local Syn does not match parent-minus-children Xi.")
        nodes.append(node)
        for child in children:
            visit(child)

    if set(record["tree"]["indices"]) != set(range(roi_count)):
        raise ValueError("SPT root and Schaefer labels do not cover the same ROI set.")
    visit(record["tree"])
    values = np.array([float(n["syn_bits_raw"]) for n in nodes])
    minimum = float(values.min())
    minor = int(np.count_nonzero((values < 0) & (values >= -tolerance)))
    significant = int(np.count_nonzero(values < -tolerance))
    if significant:
        raise ArithmeticError(
            f"Syn nonnegativity violation: minimum={minimum:.12g} bits, "
            f"threshold={-tolerance:.12g} bits, affected={significant}."
        )
    # Explicitly treat only tolerated negative estimation error as numerical zero.
    # Raw values are retained and used for ranking; no undocumented projection.
    for node in nodes:
        raw = float(node["syn_bits_raw"])
        node["display_syn_nats"] = (0.0 if -tolerance <= raw < 0 else raw) * NATS_PER_BIT
    closure = float(record["tree"]["xi_bits"]) - (
        sum(float(n["syn_bits_raw"]) for n in nodes)
        + sum(float(n["xi_bits"]) for n in leaves)
    )
    if abs(closure) > tolerance:
        raise ArithmeticError(f"ROI-block SPT closure error: {closure:.12g} bits.")
    ranked = sorted(nodes, key=lambda n: (-float(n["syn_bits_raw"]), tuple(n["indices"])))
    if not 1 <= top <= len(ranked):
        raise ValueError(f"--top must be in [1, {len(ranked)}].")
    audit = dict(
        internal_nodes=len(nodes), roi_leaves=len(leaves),
        syn_tolerance_bits=tolerance, syn_tolerance_nats=tolerance * NATS_PER_BIT,
        minimum_syn_bits=minimum, tolerated_negative_count=minor,
        significant_negative_count=significant, block_closure_error_bits=closure,
    )
    return ranked[:top], audit


def surface_membership(asset: dict, labels: list[str], indices: list[int]):
    """Exact-name mapping; nonmembers and the medial wall remain gray."""
    members = set(indices)
    values = {name: (1.0 if i in members else np.nan) for i, name in enumerate(labels)}
    mapped = set()
    hemispheres = []
    for side in ("left", "right"):
        names = list(map(str, asset[f"{side}_label_names"]))
        lookup = {i: values[name] for i, name in enumerate(names) if name in values}
        mapped.update(names[i] for i in lookup)
        mesh = SurfaceMesh(asset[f"{side}_coordinates"], asset[f"{side}_faces"], asset[f"{side}_sulc"])
        data = parcel_values_to_vertices(asset[f"{side}_vertex_labels"], lookup, background_labels=(0,))
        face_labels = asset[f"{side}_vertex_labels"][mesh.faces]
        # Light seams identify adjacent Schaefer parcels without introducing an
        # extra quantitative color scale. Mesh-face rendering preserves occlusion.
        boundaries = (np.ptp(face_labels, axis=1) > 0) & np.all(face_labels > 0, axis=1)
        hemispheres.append((mesh, data, boundaries))
    if mapped != set(labels):
        raise ValueError(f"Exact anatomical mapping is incomplete: {set(labels) - mapped}")
    return hemispheres


def plot(record: dict, selected: list[dict], labels: list[str], asset: dict, output: Path, provenance: dict) -> None:
    nrows = len(selected)
    # Fixed physical slots align brain maps, row labels, and a single shared bar axis.
    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": .7,
             "savefig.facecolor": "white", "text.color": "#242424"}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(12.2, 1.55 + 1.34 * nrows), facecolor="white")
        top, bottom = .835, .145
        row_height = (top - bottom) / nrows
        brain_left, brain_right = .115, .787
        width = (brain_right - brain_left) / 4
        context = rf"$G={record['coupling_g']:g}$ · seed {record['seed']} · Schaefer-100"
        fig.text(.027, .961, "Highest-synergy cortical coalitions", fontsize=14, weight="bold", va="center")
        fig.text(.027, .918, context + " · top 5 SPT nodes" if nrows == 5 else context + f" · top {nrows} SPT nodes",
                 fontsize=10.5, color=".35", va="center")
        fig.legend([Patch(facecolor=MEMBER_COLOR), Patch(facecolor=NONMEMBER_COLOR)],
                   ["Coalition members", "Other regions / medial wall"],
                   loc="center right", bbox_to_anchor=(.972, .945), frameon=False,
                   fontsize=9.5, ncols=1, handlelength=1.25, handleheight=1.0, labelspacing=.6)
        for col, label in enumerate(("Left lateral", "Right lateral", "Left medial", "Right medial")):
            fig.text(brain_left + (col + .5) * width, .865, label, ha="center", fontsize=10, color=".3")
        fig.text(.035, .865, "Rank / size", fontsize=10, color=".3")
        fig.text(.89, .865, "Local Syn", ha="center", fontsize=10, color=".3")

        cmap = ListedColormap([MEMBER_COLOR])
        for row, node in enumerate(selected):
            y = top - (row + 1) * row_height
            center = y + row_height / 2
            left, right = surface_membership(asset, labels, node["indices"])
            views = ((left, 180.), (right, 0.), (left, 0.), (right, 180.))
            # 3D axes enforce a square viewport. Extend the invisible viewport
            # beyond each row while keeping the actual cortical silhouette inside
            # its row; increasing camera zoom alone would crop anterior/posterior.
            viewport_height = 1.40 * row_height
            for col, ((mesh, values, boundaries), azim) in enumerate(views):
                ax = fig.add_axes([brain_left + col * width, center - viewport_height / 2,
                                   width, viewport_height], projection="3d")
                _draw_surface(ax, mesh, values, cmap=cmap, norm=Normalize(0, 1), elev=0,
                              azim=azim, background_color=NONMEMBER_COLOR, zoom=1.50)
                face_colors = _face_colors(mesh, values, cmap=cmap, norm=Normalize(0, 1),
                                           background_color=NONMEMBER_COLOR)
                member_seams = boundaries & np.all(np.isfinite(values[mesh.faces]), axis=1)
                face_colors[member_seams, :3] = .80 * face_colors[member_seams, :3] + .20
                ax.collections[-1].set_facecolor(face_colors)
            fig.text(.035, center + .016, f"{row + 1:02d}", fontsize=15, weight="bold", va="center")
            fig.text(.035, center - .020, f"{node['size']} ROIs", fontsize=10.5, color=".3", va="center")
            if row < nrows - 1:
                fig.add_artist(Line2D([.027, .98], [y, y], transform=fig.transFigure,
                                      color="#E8E8E8", linewidth=.6, zorder=0))

        bar = fig.add_axes([.824, bottom, .145, top - bottom])
        values = np.array([n["display_syn_nats"] for n in selected])
        bar.barh(np.arange(nrows), values, height=.30, color=MEMBER_COLOR, zorder=3)
        bar.set_ylim(nrows - .5, -.5)
        step = .1 if float(values.max()) > .2 else .05
        # Leave enough shared-scale space for four-decimal direct labels.
        limit = np.ceil((float(values.max()) + .13) / step) * step
        bar.set_xlim(0, limit)
        bar.set_xticks(np.arange(0, limit + step * .2, step))
        bar.set_xlabel("Syn (nats)", fontsize=10, labelpad=6)
        bar.set_yticks([])
        bar.tick_params(axis="x", labelsize=9, length=3, width=.65, color=".4")
        bar.grid(axis="x", color="#E6E6E6", linewidth=.6, zorder=0)
        bar.set_axisbelow(True)
        for side in ("top", "right", "left"):
            bar.spines[side].set_visible(False)
        bar.spines["bottom"].set_color(".45")
        for row, value in enumerate(values):
            bar.text(value + .013, row, f"{value:.4f}", va="center", fontsize=9.5)
        fig.text(.027, .037, "Rows ranked by local Syn; color marks membership, not contribution magnitude.",
                 fontsize=9, color=".35")
        fig.text(.027, .013, "Syn is the residual between each node's selected children; overlapping coalitions are retained.",
                 fontsize=8.5, color=".4")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches=None,
                    metadata={"Title": "Ranked SPT cortical coalitions", "Description": json.dumps(provenance)})
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--surface", type=Path, default=DEFAULT_SURFACE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    record = json.loads(args.tree.read_text())
    labels = args.labels.read_text().splitlines()
    if len(labels) != 100 or len(set(labels)) != 100:
        raise ValueError("Expected 100 unique Schaefer-100 anatomical labels.")
    selected, audit = ranked_nodes(record, len(labels), args.top)
    with np.load(args.surface, allow_pickle=False) as archive:
        asset = {key: archive[key] for key in archive.files}
    provenance = dict(
        tree=str(args.tree), tree_sha256=hashlib.sha256(args.tree.read_bytes()).hexdigest(),
        surface=str(args.surface), labels=str(args.labels), seed=record["seed"], G=record["coupling_g"],
        ranking="all internal SPT nodes, descending raw local Syn, index-tuple tie break",
        metric="selected-child residual Syn, bits converted to nats by ln(2)",
        anatomical_mapping="exact parcel-name match; inherited inferred SC row order",
        manuscript=dict(parent="P6UJCVG8", main="DXGC7JEA", supplement="MWIWKSVG",
                        checked="2026-10-06", revision="unspecified"),
        audit=audit,
        selected=[dict(rank=i+1, roi_count=n["size"], indices=n["indices"],
                       parcels=[labels[j] for j in n["indices"]], syn_nats=n["display_syn_nats"],
                       child_sizes=[c["size"] for c in n["children"]]) for i,n in enumerate(selected)],
    )
    plot(record, selected, labels, asset, args.output, provenance)
    print(json.dumps(dict(output=str(args.output), audit=audit,
                          ranks=[{k:v for k,v in row.items() if k not in ("indices","parcels")} for row in provenance["selected"]]), indent=2))


if __name__ == "__main__":
    main()
