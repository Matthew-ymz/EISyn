"""Minimal, reusable rendering for an explicit PEID synergy hierarchy tree."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb

from scripts.phi_hierarchy import PhiTreeNode


SYN_COLOR = "#267A70"
LEAF_EDGE_COLOR = "#7B8794"
LEAF_FACE_COLOR = "#F4F6F8"
EDGE_COLOR = "#AEB7C0"


def _blend_with_white(color: str, strength: float) -> str:
    base = to_rgb(color)
    amount = min(1.0, max(0.0, float(strength)))
    return to_hex(tuple(1.0 - amount * (1.0 - channel) for channel in base))


def _leaf_count(node: PhiTreeNode) -> int:
    if not node.children:
        return 1
    return sum(_leaf_count(child) for child in node.children)


def _max_depth(node: PhiTreeNode) -> int:
    if not node.children:
        return int(node.depth)
    return max(_max_depth(child) for child in node.children)


def _max_abs_syn(node: PhiTreeNode) -> float:
    own = abs(float(node.residual)) if node.atom_kind is not None else 0.0
    return max([own, *(_max_abs_syn(child) for child in node.children)])


def _layout(node: PhiTreeNode) -> dict[tuple[str, ...], tuple[float, float]]:
    positions: dict[tuple[str, ...], tuple[float, float]] = {}
    next_leaf = 0.0

    def visit(current: PhiTreeNode) -> float:
        nonlocal next_leaf
        if not current.children:
            x = next_leaf
            next_leaf += 1.0
        else:
            child_x = [visit(child) for child in current.children]
            x = sum(child_x) / len(child_x)
        positions[current.sources] = (x, -float(current.depth))
        return x

    visit(node)
    return positions


def _source_label(sources: tuple[str, ...], labels: Mapping[str, str]) -> str:
    rendered = [str(labels.get(source, source)) for source in sources]
    if len(rendered) == 1:
        return rendered[0]
    return "{" + ", ".join(rendered) + "}"


def _node_label(
    node: PhiTreeNode,
    *,
    labels: Mapping[str, str],
    decimals: int,
) -> str:
    source_text = _source_label(node.sources, labels)
    if node.atom_kind is None and not node.children:
        return source_text
    atom_text = f"Syn {node.residual:.{decimals}f}"
    return f"{source_text}\n{atom_text}"


def plot_synergy_hierarchy_tree(
    tree: PhiTreeNode,
    output_path: str | Path,
    *,
    source_labels: Mapping[str, str] | None = None,
    decimals: int = 2,
    show_root_total: bool = True,
    syn_scale_max: float | None = None,
    dpi: int = 600,
) -> Path:
    """Render only the synergy hierarchy tree and save one opaque PNG."""
    output = Path(output_path)
    if output.suffix.lower() != ".png":
        raise ValueError("The default reusable renderer writes one .png figure")
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = dict(source_labels or {})

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.8,
        }
    )

    leaves = _leaf_count(tree)
    depth = _max_depth(tree) - int(tree.depth)
    width = max(4.8, 1.45 * leaves)
    height = max(3.2, 1.35 * (depth + 1))
    figure, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    positions = _layout(tree)
    scale_max = _max_abs_syn(tree) if syn_scale_max is None else abs(float(syn_scale_max))
    if scale_max <= 0.0:
        scale_max = 1.0

    def draw_edges(node: PhiTreeNode) -> None:
        x0, y0 = positions[node.sources]
        for child in node.children:
            x1, y1 = positions[child.sources]
            axis.plot(
                [x0, x1],
                [y0, y1],
                color=EDGE_COLOR,
                linewidth=1.15,
                solid_capstyle="round",
                zorder=1,
            )
            draw_edges(child)

    draw_edges(tree)

    def draw_nodes(node: PhiTreeNode) -> None:
        x, y = positions[node.sources]
        relative_syn = min(1.0, abs(float(node.residual)) / scale_max) if node.atom_kind is not None else 0.0
        if node.atom_kind is not None:
            edgecolor = _blend_with_white(SYN_COLOR, 0.58 + 0.42 * relative_syn)
            facecolor = _blend_with_white(SYN_COLOR, 0.10 + 0.52 * relative_syn)
            linewidth = 1.35 + 2.25 * relative_syn
        else:
            edgecolor = LEAF_EDGE_COLOR
            facecolor = LEAF_FACE_COLOR
            linewidth = 0.9
        axis.text(
            x,
            y,
            _node_label(
                node,
                labels=labels,
                decimals=decimals,
            ),
            ha="center",
            va="center",
            color="#24313C",
            fontsize=8.2,
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.48",
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "linewidth": linewidth,
            },
            zorder=3,
        )
        for child in node.children:
            draw_nodes(child)

    draw_nodes(tree)
    if show_root_total:
        root_x, _ = positions[tree.sources]
        axis.text(
            root_x,
            0.58,
            f"$\\Xi$ = {tree.phi_value:.{decimals}f} bits",
            ha="center",
            va="center",
            color="#24313C",
            fontsize=9.0,
            zorder=4,
        )
    axis.set_xlim(-0.65, max(0.0, float(leaves - 1)) + 0.65)
    axis.set_ylim(-float(depth) - 0.55, 0.88 if show_root_total else 0.55)
    axis.set_aspect("auto")
    axis.axis("off")
    figure.savefig(output, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output
