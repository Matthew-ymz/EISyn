#!/usr/bin/env python3
"""Render exact lead-8 Xi hierarchy trees for the three frozen UniCM checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex, to_rgb
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phi_hierarchy import (
    ALL_ORDER_CROSS_DENSITY,
    NONNEGATIVE_TOLERANT,
    RAW_RESIDUAL,
    PhiTreeNode,
    greedy_phi_tree,
)
from scripts.plot_unicm_all_mode_target_pair_syn import extract_all_mode_target
from scripts.plot_unicm_phi_eid_greedy_decomposition import (
    compute_subset_ei_table_from_covariance,
    precompute_source_logdets,
)
from scripts.unicm_peid_syn_analysis import (
    MODE_NAMES,
    load_full_history_prediction_cache,
    overall_prediction_cache_path,
    sample_full_history_mode_inputs,
)


DEFAULT_CACHE_DIR = ROOT / "results/unicm_overall_ei_cpu_bound4_n8192/cache"
DEFAULT_OUTPUT_DIR = ROOT / "results/unicm_xi_hierarchy_tree"
DEFAULT_FIGURE = ROOT / "fig/earth_unicm_11mode_xi_hierarchy_lead08.png"
EDGE_COLOR = "#B7C0CA"
INK = "#25313C"
SYN_COLOR = "#267A70"
CORE_MODES = frozenset(("nino", "IOD", "nino12", "nino3", "nino4"))
MODE_COLORS = {
    "nino": "#3267A8",
    "nino12": "#5289C7",
    "nino3": "#6FA5D8",
    "nino4": "#8ABBDD",
    "WWV": "#56A6A6",
    "NPMM": "#3F8F83",
    "SPMM": "#75B79E",
    "IOB": "#D98245",
    "IOD": "#C75B35",
    "SIOD": "#E6A15C",
    "TNA": "#8A69A6",
}


def _cache_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        n_samples=int(args.n_samples),
        sampling_seed=int(args.sampling_seed),
        intervention_bound=float(args.intervention_bound),
        start_month=int(args.start_month),
        device="cpu",
    )


def _flatten(tree: PhiTreeNode) -> list[PhiTreeNode]:
    nodes = [tree]
    for child in tree.children:
        nodes.extend(_flatten(child))
    return nodes


def _terminal_order(tree: PhiTreeNode) -> list[PhiTreeNode]:
    if not tree.children:
        return [tree]
    return [leaf for child in tree.children for leaf in _terminal_order(child)]


def _expand_pair_leaves(tree: PhiTreeNode) -> PhiTreeNode:
    """Expose the unique singleton split of cached pair atoms without changing Syn.

    Singleton Xi is zero by construction. Larger stopped coalitions require EI
    data to choose their internal topology and must never be split arbitrarily.
    """
    if tree.children:
        return replace(tree, children=tuple(_expand_pair_leaves(c) for c in tree.children))
    if tree.order == 1:
        return tree
    if tree.order != 2 or not np.isclose(tree.phi_value, tree.residual, atol=1e-12, rtol=0):
        raise ValueError(f"Cannot expand cached coalition {tree.sources}: internal EI data required")
    return replace(tree, split_kind="expanded_pair", children=tuple(
        PhiTreeNode((name,), 0.0, 0.0, tree.depth + 1, "leaf")
        for name in tree.sources
    ))


def _tree_from_record(record: dict[str, object]) -> PhiTreeNode:
    return PhiTreeNode(
        tuple(record["sources"]), float(record["phi_value_bits"]),
        float(record["residual_syn_bits"]), int(record["depth"]),
        str(record["action"]),
        tuple(_tree_from_record(child) for child in record["children"]),
    )


def _validate_syn(trees: Sequence[PhiTreeNode], tolerance: float) -> dict[str, float | int]:
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Syn tolerance must be finite and nonnegative, in bits")
    values = np.asarray([node.residual for tree in trees for node in _flatten(tree)])
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite Syn in hierarchy")
    invalid = int(np.count_nonzero(values < -tolerance))
    if invalid:
        raise ValueError(f"Syn nonnegativity violation: minimum={values.min():.12g} bits, "
                         f"threshold={-tolerance:.12g} bits, count={invalid}")
    return {"tolerance_bits": tolerance, "minimum_syn_bits": float(values.min()),
            "display_zero_count": int(np.count_nonzero(values < 0)),
            "significant_violation_count": invalid}


def _blend_with_white(color: str, strength: float) -> str:
    base = to_rgb(color)
    amount = min(1.0, max(0.0, float(strength)))
    return to_hex(tuple(1.0 - amount * (1.0 - channel) for channel in base))


def _tree_metrics(tree: PhiTreeNode, tolerance: float = 1e-4) -> dict[str, float | int]:
    nodes = _flatten(tree)
    internal = [node for node in nodes if node.children]
    terminals = [node for node in nodes if not node.children]
    current = tree
    spine = 0
    while current.children:
        spine += 1
        current = max(current.children, key=lambda child: (child.order, child.sources))
    colless = sum(abs(node.children[0].order - node.children[1].order) for node in internal)
    maximum_colless = (tree.order - 1) * (tree.order - 2) / 2.0
    atom_nodes = [node for node in nodes if node.atom_kind is not None]
    closure_error = float(sum(node.residual for node in atom_nodes) - tree.phi_value)
    tolerance_negative = sum(-tolerance <= node.residual < 0.0 for node in atom_nodes)
    return {
        "internal_node_count": len(internal),
        "terminal_node_count": len(terminals),
        "maximum_depth": max(node.depth for node in terminals),
        "dominant_spine_split_count": spine,
        "dominant_spine_fraction": float(spine / len(internal)) if internal else 0.0,
        "normalized_colless_imbalance": float(colless / maximum_colless) if maximum_colless else 0.0,
        "closure_error_bits": closure_error,
        "tolerance_negative_atom_count": int(tolerance_negative),
    }


def _node_record(tree: PhiTreeNode) -> dict[str, object]:
    return {
        "sources": list(tree.sources),
        "order": tree.order,
        "phi_value_bits": float(tree.phi_value),
        "residual_syn_bits": float(tree.residual),
        "action": tree.action,
        "atom_kind": tree.atom_kind,
        "depth": tree.depth,
        "children": [_node_record(child) for child in tree.children],
    }


def _coalition_label(node: PhiTreeNode) -> str:
    if node.order <= 4:
        names = ["ENSO" if name == "nino" else name for name in node.sources]
        members = " +\n".join(" + ".join(names[i:i + 2]) for i in range(0, len(names), 2))
    elif frozenset(node.sources) == CORE_MODES:
        members = "5-mode core"
    else:
        members = f"{node.order} modes"
    return f"{members}\nSyn {node.residual:.3f}"


def _positions(tree: PhiTreeNode) -> tuple[dict[int, tuple[float, float]], list[PhiTreeNode]]:
    terminals = _terminal_order(tree)
    leaf_x = {}
    next_x = 0.0
    for node in terminals:
        leaf_x[id(node)] = next_x
        next_x += 1.5 if node.sources[0] in CORE_MODES else 1.0
    positions: dict[int, tuple[float, float]] = {}
    maximum_depth = max(node.depth for node in terminals)

    def position(node: PhiTreeNode) -> tuple[float, float]:
        if id(node) in positions:
            return positions[id(node)]
        if not node.children:
            point = (leaf_x[id(node)], 0.0)
        else:
            children = [position(child) for child in node.children]
            x_value = sum(point[0] * child.order for point, child in zip(children, node.children, strict=True)) / node.order
            point = (x_value, float(maximum_depth - node.depth))
        positions[id(node)] = point
        return point

    position(tree)
    return positions, terminals


def render_trees(
    trees: Sequence[PhiTreeNode],
    seeds: Sequence[int],
    output: Path,
    *,
    lead: int,
    split_objective: str,
    dpi: int,
    syn_tolerance: float = 1e-4,
    canvas=None,
    show_colorbar: bool = True,
    compact_core_annotation: bool = False,
    show_checkpoint: bool = True,
    show_tree_metrics: bool = True,
    core_highlights: Sequence[bool] | None = None,
) -> None:
    trees = [_expand_pair_leaves(tree) for tree in trees]
    _validate_syn(trees, syn_tolerance)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    all_internal = [node for tree in trees for node in _flatten(tree) if node.children]
    maximum_syn = max(float(node.residual) for node in all_internal)
    norm = Normalize(vmin=0.0, vmax=maximum_syn if maximum_syn > 0 else 1.0)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("syn", ["#F1F7F5", SYN_COLOR])
    if canvas is None:
        figure, axes = plt.subplots(1, len(trees), figsize=(18.0, 8.6), constrained_layout=True)
    else:
        figure, axes = canvas, canvas.subplots(1, len(trees))
    axes_array = np.atleast_1d(axes)
    if core_highlights is None:
        core_highlights = [True] * len(trees)
    if len(core_highlights) != len(trees):
        raise ValueError("core_highlights must match the number of trees")
    maximum_depth = max(node.depth for tree in trees for node in _terminal_order(tree))

    for axis, tree, seed, highlight_core in zip(
        axes_array, trees, seeds, core_highlights, strict=True
    ):
        positions, terminals = _positions(tree)
        internal = [node for node in _flatten(tree) if node.children]
        core = (
            next((node for node in internal if frozenset(node.sources) == CORE_MODES), None)
            if highlight_core
            else None
        )
        if core is not None:
            core_x = [positions[id(node)][0] for node in _terminal_order(core)]
            left, right = min(core_x) - 0.65, max(core_x) + 0.65
            top = positions[id(core)][1] + 0.42
            axis.add_patch(FancyBboxPatch(
                (left, -1.05), right - left, top + 1.05,
                boxstyle="round,pad=0.05,rounding_size=0.16",
                facecolor="#EDF5F2", edgecolor="#8BB9AA", linewidth=1.1, zorder=0,
            ))
            axis.plot([left, left, right, right], [-1.21, -1.35, -1.35, -1.21],
                      color=SYN_COLOR, linewidth=1.5)
            share = core.phi_value / tree.phi_value
            axis.text((left + right) / 2, -1.58, "ENSO–IOD synergy core",
                      ha="center", va="top", fontsize=9, weight="bold", color=SYN_COLOR)
            if not compact_core_annotation:
                axis.text((left + right) / 2, -1.98,
                          rf"$\Xi_{{core}}$ = {core.phi_value:.3f} bits  |  {share:.1%} of total $\Xi$",
                          ha="center", va="top", fontsize=8, color=INK)
        for node in internal:
            px, py = positions[id(node)]
            child_points = [positions[id(child)] for child in node.children]
            in_core = core is not None and set(node.sources).issubset(CORE_MODES)
            edge_color = SYN_COLOR if in_core else EDGE_COLOR
            edge_width = 1.8 if in_core else 0.85
            axis.plot(
                [child_points[0][0], child_points[1][0]], [py, py],
                color=edge_color, linewidth=edge_width, zorder=1,
            )
            for cx, cy in child_points:
                axis.plot([cx, cx], [cy, py], color=edge_color, linewidth=edge_width, zorder=1)

        for node in internal:
            x_value, y_value = positions[id(node)]
            syn = float(node.residual)
            # Only validated numerical negatives are displayed as zero; raw values stay intact.
            relative = float(norm(0.0 if syn < 0 else syn))
            axis.text(
                x_value, y_value, _coalition_label(node),
                ha="center", va="center", fontsize=6.5,
                color="white" if relative > 0.65 else INK, linespacing=1.15,
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "facecolor": cmap(relative),
                    "edgecolor": _blend_with_white(SYN_COLOR, 0.58 + 0.42 * relative),
                    "linewidth": 0.9 + 1.5 * relative,
                },
                zorder=4,
            )

        for terminal in terminals:
            label = "ENSO" if terminal.sources[0] == "nino" else terminal.sources[0]
            color = MODE_COLORS.get(terminal.sources[0], "#7B8794")
            axis.text(
                positions[id(terminal)][0], -0.22, label, rotation=58,
                ha="right", va="top", fontsize=7.5, color=INK,
                bbox={"boxstyle": "round,pad=0.22", "facecolor": _blend_with_white(color, 0.18),
                      "edgecolor": color, "linewidth": 0.9},
                clip_on=False,
            )

        checkpoint_line = f"checkpoint {seed}\n" if show_checkpoint else ""
        info = checkpoint_line + rf"$\Xi$ = {tree.phi_value:.3f} bits"
        if show_tree_metrics:
            metrics = _tree_metrics(tree)
            info += (
                "\n"
                f"spine {metrics['dominant_spine_fraction']:.0%}  |  "
                f"imbalance {metrics['normalized_colless_imbalance']:.2f}"
            )
        axis.text(
            0.02, 0.98,
            info,
            transform=axis.transAxes, ha="left", va="top", fontsize=7.4, color=INK, linespacing=1.35,
        )
        axis.set_xlim(-1.0, max(point[0] for point in positions.values()) + 0.7)
        axis.set_ylim(-2.8, maximum_depth + 1.5)
        axis.axis("off")

    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    if show_colorbar:
        colorbar = figure.colorbar(scalar, ax=list(axes_array), location="right", shrink=0.55, pad=0.02)
        colorbar.set_label("Local hierarchy Syn (bits)")
    if canvas is not None:
        return
    objective_label = (
        "all-order-normalized split selection; nodes show raw Syn"
        if split_objective == ALL_ORDER_CROSS_DENSITY
        else "raw-residual split selection"
    )
    figure.text(0.5, -0.02,
                f"Lead {lead}  |  {objective_label}  |  ENSO = nino  |  "
                "Shading marks core membership; node fill shows local Syn.",
                ha="center", va="top", fontsize=8, color=INK)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    if args.summary is not None:
        payload = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        trees = [_expand_pair_leaves(_tree_from_record(row["tree"])) for row in payload["checkpoints"]]
        tolerance = float(payload["split_tolerance_bits"])
        payload["syn_display_validation"] = _validate_syn(trees, tolerance)
        for row, tree in zip(payload["checkpoints"], trees, strict=True):
            row["tree"] = _node_record(tree)
            row["metrics"] = _tree_metrics(tree, tolerance)
            if abs(float(row["metrics"]["closure_error_bits"])) > 1e-8:
                raise ValueError("Expanded cached hierarchy does not close")
        payload["leaf_display"] = "singleton modes; terminal pair atoms retain their Syn and expose their unique binary split"
        payload["figure"] = str(args.figure)
        render_trees(trees, [row["seed"] for row in payload["checkpoints"]], Path(args.figure),
                     lead=int(payload["lead"]), split_objective=payload.get("split_objective", RAW_RESIDUAL),
                     dpi=int(args.dpi), syn_tolerance=tolerance)
        Path(args.summary).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    mode_names = tuple(MODE_NAMES)
    history_modes = sample_full_history_mode_inputs(
        n_samples=int(args.n_samples),
        intervention_bound=float(args.intervention_bound),
        seed=int(args.sampling_seed),
    )
    history_flat, subset_columns, source_logdets = precompute_source_logdets(
        history_modes, jitter=float(args.jitter),
    )
    cache_args = _cache_args(args)
    trees: list[PhiTreeNode] = []
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        cache_path = overall_prediction_cache_path(Path(args.cache_dir), seed=int(seed), args=cache_args)
        predictions = load_full_history_prediction_cache(cache_path, n_samples=int(args.n_samples))
        target = extract_all_mode_target(predictions, lead=int(args.lead))
        ei_table = compute_subset_ei_table_from_covariance(
            history_flat, target, subset_columns, source_logdets, jitter=float(args.jitter),
        )
        singleton_ei = {name: float(ei_table[(name,)]) for name in mode_names}
        tree = greedy_phi_tree(
            mode_names,
            ei_table,
            policy=NONNEGATIVE_TOLERANT,
            eps=float(args.eps),
            split_tolerance=float(args.split_tolerance),
            singleton_ei=singleton_ei,
            split_objective=str(args.split_objective),
        )
        tree = _expand_pair_leaves(tree)
        _validate_syn([tree], float(args.split_tolerance))
        metrics = _tree_metrics(tree, float(args.split_tolerance))
        if abs(float(metrics["closure_error_bits"])) > 1.0e-8:
            raise RuntimeError(
                f"Checkpoint {seed} hierarchy closure failed: {metrics['closure_error_bits']:.12g} bits"
            )
        rows.append(
            {
                "seed": int(seed),
                "xi_bits": float(tree.phi_value),
                "metrics": metrics,
                "tree": _node_record(tree),
            }
        )
        trees.append(tree)

    render_trees(
        trees,
        args.seeds,
        Path(args.figure),
        lead=int(args.lead),
        split_objective=str(args.split_objective),
        dpi=int(args.dpi),
        syn_tolerance=float(args.split_tolerance),
    )
    payload: dict[str, object] = {
        "experiment": "UniCM exact 11-mode Xi hierarchy at the mid-lead peak",
        "lead": int(args.lead),
        "seeds": [int(seed) for seed in args.seeds],
        "source_definition": "each of 11 sources is one mode's full 12-month history",
        "target_definition": "all 11 future UniCM modes jointly",
        "n_samples": int(args.n_samples),
        "intervention_bound": float(args.intervention_bound),
        "estimator": "affine degree-1 TM / Gaussian log-det equivalent",
        "hierarchy": "exact bipartition enumeration with the nonnegative-tolerant greedy policy",
        "split_objective": str(args.split_objective),
        "split_objective_denominator": (
            "(2^|A| - 1)(2^|B| - 1)"
            if args.split_objective == ALL_ORDER_CROSS_DENSITY
            else "1"
        ),
        "reported_node_value": "raw unnormalized Syn residual in bits",
        "eps_bits": float(args.eps),
        "split_tolerance_bits": float(args.split_tolerance),
        "figure": str(args.figure),
        "checkpoints": rows,
        "syn_display_validation": _validate_syn(trees, float(args.split_tolerance)),
        "leaf_display": "singleton modes; terminal pair atoms retain their Syn and expose their unique binary split",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, help="Redraw and refresh an existing tree summary without recomputing EI")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--lead", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--sampling-seed", type=int, default=20260619)
    parser.add_argument("--intervention-bound", type=float, default=4.0)
    parser.add_argument("--start-month", type=int, default=0)
    parser.add_argument("--jitter", type=float, default=1.0e-6)
    parser.add_argument("--eps", type=float, default=1.0e-5)
    parser.add_argument("--split-tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--split-objective",
        choices=(RAW_RESIDUAL, ALL_ORDER_CROSS_DENSITY),
        default=RAW_RESIDUAL,
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    payload = run_analysis(parse_args())
    print(
        "[done] "
        + ", ".join(
            f"seed={row['seed']} Xi={row['xi_bits']:.6f} spine={row['metrics']['dominant_spine_fraction']:.3f}"
            for row in payload["checkpoints"]
        )
    )


if __name__ == "__main__":
    main()
