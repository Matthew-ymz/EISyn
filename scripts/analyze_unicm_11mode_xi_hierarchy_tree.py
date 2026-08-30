#!/usr/bin/env python3
"""Render exact lead-8 Xi hierarchy trees for the three frozen UniCM checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex, to_rgb


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phi_hierarchy import NONNEGATIVE_TOLERANT, PhiTreeNode, greedy_phi_tree
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


def _blend_with_white(color: str, strength: float) -> str:
    base = to_rgb(color)
    amount = min(1.0, max(0.0, float(strength)))
    return to_hex(tuple(1.0 - amount * (1.0 - channel) for channel in base))


def _tree_metrics(tree: PhiTreeNode) -> dict[str, float | int]:
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
    tolerance_negative = sum(-1.0e-4 <= node.residual < 0.0 for node in atom_nodes)
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
    if node.order <= 5:
        members = " + ".join(node.sources)
    else:
        members = f"{node.order} modes"
    return f"{members}\nSyn {node.residual:.3f}"


def _positions(tree: PhiTreeNode) -> tuple[dict[int, tuple[float, float]], list[PhiTreeNode]]:
    terminals = _terminal_order(tree)
    leaf_x = {id(node): float(index) for index, node in enumerate(terminals)}
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
    dpi: int,
) -> None:
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
    figure, axes = plt.subplots(1, len(trees), figsize=(15.2, 7.4), constrained_layout=True)
    axes_array = np.atleast_1d(axes)

    for axis, tree, seed in zip(axes_array, trees, seeds, strict=True):
        positions, terminals = _positions(tree)
        internal = [node for node in _flatten(tree) if node.children]
        for node in internal:
            px, py = positions[id(node)]
            child_points = [positions[id(child)] for child in node.children]
            axis.plot(
                [child_points[0][0], child_points[1][0]], [py, py],
                color=EDGE_COLOR, linewidth=0.9, zorder=1,
            )
            for cx, cy in child_points:
                axis.plot([cx, cx], [cy, py], color=EDGE_COLOR, linewidth=0.9, zorder=1)

        for node in internal:
            x_value, y_value = positions[id(node)]
            syn = float(node.residual)
            relative = max(0.0, min(1.0, syn / maximum_syn)) if maximum_syn > 0.0 else 0.0
            axis.text(
                x_value, y_value, _coalition_label(node),
                ha="center", va="center", fontsize=5.6, color=INK, linespacing=1.15,
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "facecolor": _blend_with_white(SYN_COLOR, 0.12 + 0.62 * relative),
                    "edgecolor": _blend_with_white(SYN_COLOR, 0.58 + 0.42 * relative),
                    "linewidth": 0.9 + 1.5 * relative,
                },
                zorder=4,
            )

        for index, terminal in enumerate(terminals):
            label = " + ".join(terminal.sources)
            color = MODE_COLORS.get(terminal.sources[0], "#7B8794")
            axis.text(
                float(index), -0.42, label, rotation=58,
                ha="right", va="top", fontsize=6.3, color=INK,
                bbox={"boxstyle": "round,pad=0.22", "facecolor": _blend_with_white(color, 0.18),
                      "edgecolor": color, "linewidth": 0.9},
                clip_on=False,
            )

        metrics = _tree_metrics(tree)
        axis.text(
            0.02, 0.98,
            f"checkpoint {seed}\n"
            rf"$\Xi$ = {tree.phi_value:.3f} bits" "\n"
            f"spine {metrics['dominant_spine_fraction']:.0%}  |  imbalance {metrics['normalized_colless_imbalance']:.2f}",
            transform=axis.transAxes, ha="left", va="top", fontsize=7.4, color=INK, linespacing=1.35,
        )
        axis.set_xlim(-0.8, len(terminals) - 0.2)
        axis.set_ylim(-2.3, max(point[1] for point in positions.values()) + 1.25)
        axis.axis("off")

    scalar = ScalarMappable(norm=Normalize(vmin=0.0, vmax=maximum_syn), cmap=mpl.colors.LinearSegmentedColormap.from_list("syn", ["#F1F7F5", SYN_COLOR]))
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=list(axes_array), location="right", shrink=0.55, pad=0.02)
    colorbar.set_label("Local hierarchy Syn (bits)")
    figure.text(
        0.5, 1.01,
        f"Lead {lead}: exact 11-mode hierarchy; shared intervention support and Syn scale",
        ha="center", va="bottom", fontsize=9.2, color=INK,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
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
        )
        metrics = _tree_metrics(tree)
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

    render_trees(trees, args.seeds, Path(args.figure), lead=int(args.lead), dpi=int(args.dpi))
    payload: dict[str, object] = {
        "experiment": "UniCM exact 11-mode Xi hierarchy at the mid-lead peak",
        "lead": int(args.lead),
        "seeds": [int(seed) for seed in args.seeds],
        "source_definition": "each of 11 sources is one mode's full 12-month history",
        "target_definition": "all 11 future UniCM modes jointly",
        "n_samples": int(args.n_samples),
        "intervention_bound": float(args.intervention_bound),
        "estimator": "affine degree-1 TM / Gaussian log-det equivalent",
        "hierarchy": "exact bipartition enumeration with the existing nonnegative-tolerant greedy policy",
        "eps_bits": float(args.eps),
        "split_tolerance_bits": float(args.split_tolerance),
        "figure": str(args.figure),
        "checkpoints": rows,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
