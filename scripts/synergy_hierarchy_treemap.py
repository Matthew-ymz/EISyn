"""Area-preserving squarified view of cached SPT atoms.

Each node partitions its rectangle into its own Syn and its child subtree
masses. Singleton leaves have no Syn area; their names remain in coalition
labels. Frames mark positive-mass subtrees. No padding or minimum tile area
is added, so tile area / canvas area equals Syn / displayed total.

Layout: Bruls, Huizing & van Wijk (2000), doi:10.2312/VisSym/VisSym00/033-042.
Figure review used Scientific Agent Skills, Kassis et al. (2026),
doi:10.48550/arXiv.2609.00065 (current arXiv revision: 2 Sep 2026).
"""

from __future__ import annotations

import math
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scripts.phi_hierarchy import PhiTreeNode
from scripts.spt import flatten_nodes


Rect = tuple[float, float, float, float]
INK = "#24313C"
SYN_COLOR = "#267A70"


def _text_color(face) -> str:
    """Choose the higher-contrast ink using linearized sRGB luminance."""
    def luminance(rgb):
        linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4
                  for v in rgb[:3]]
        return sum(v * weight for v, weight in zip(linear, (.2126, .7152, .0722)))

    background = luminance(face)
    ink_color = "#000000"
    ink = luminance(mpl.colors.to_rgb(ink_color))
    white_contrast = 1.05 / (background + .05)
    ink_contrast = (background + .05) / (ink + .05)
    return "white" if white_contrast >= ink_contrast else ink_color


def squarified_rectangles(weights: list[float], bounds: Rect) -> list[Rect]:
    """Pack positive weights; stable descending sort, returned in input order."""
    if not weights:
        return []
    x, y, width, height = bounds
    if (not all(math.isfinite(v) and v > 0 for v in weights)
            or not all(math.isfinite(v) for v in bounds) or width <= 0 or height <= 0):
        raise ValueError("Squarified layout requires finite positive weights and dimensions")
    total = math.fsum(weights)
    remaining = sorted(
        ((i, w / total * width * height) for i, w in enumerate(weights)),
        key=lambda item: -item[1],
    )
    result: list[Rect | None] = [None] * len(weights)

    def worst(row, side):
        areas = [area for _, area in row]
        area = math.fsum(areas)
        return max(side * side * max(areas) / (area * area),
                   area * area / (side * side * min(areas)))

    while remaining:
        side = min(width, height)
        row = [remaining.pop(0)]
        while remaining and worst(row + [remaining[0]], side) <= worst(row, side):
            row.append(remaining.pop(0))
        area = math.fsum(value for _, value in row)
        if width >= height:
            strip = width if not remaining else area / height
            offset = y
            for index, value in row:
                extent = value / strip
                result[index] = (x, offset, strip, extent)
                offset += extent
            x += strip
            width -= strip
        else:
            strip = height if not remaining else area / width
            offset = x
            for index, value in row:
                extent = value / strip
                result[index] = (offset, y, extent, strip)
                offset += extent
            y += strip
            height -= strip
    return [rect for rect in result if rect is not None]


def synergy_treemap_layout(tree: PhiTreeNode, *, syn_tolerance: float = 1e-10,
                           bounds: Rect = (0, 0, 6, 4)):
    """Return atom tiles, subtree frames, and explicit numerical audit.

    Raw nodes remain untouched. Only Syn in [-syn_tolerance, 0) is displayed
    as zero. Closure is checked at every node before using areas.
    """
    if not math.isfinite(syn_tolerance) or syn_tolerance < 0:
        raise ValueError("syn_tolerance must be finite and nonnegative (bits)")
    nodes = flatten_nodes(tree)
    atoms = [node for node in nodes if node.atom_kind is not None]
    values = [float(node.residual) for node in atoms]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Syn values must be finite")
    violations = sum(value < -syn_tolerance for value in values)
    if violations:
        raise ValueError("Significant Syn nonnegativity violation: "
                         f"minimum={min(values):.12g} bits, "
                         f"threshold={-syn_tolerance:.12g} bits, affected_count={violations}")
    displayed = {id(node): (0.0 if node.residual < 0 else float(node.residual))
                 for node in atoms}
    masses: dict[int, float] = {}

    def mass(node):
        raw = (float(node.residual) if node.atom_kind is not None else 0.0)
        expected = math.fsum([raw, *(float(c.phi_value) for c in node.children)])
        if not math.isfinite(node.phi_value) or not math.isclose(
                expected, node.phi_value, rel_tol=1e-9, abs_tol=syn_tolerance):
            raise ValueError(f"SPT closure failed for {node.sources}: "
                             f"Xi={node.phi_value}, atom+children={expected}")
        masses[id(node)] = math.fsum([displayed.get(id(node), 0.0),
                                     *(mass(child) for child in node.children)])
        return masses[id(node)]

    total = mass(tree)
    tiles, frames = [], []

    def visit(node, rect):
        entries = []
        if displayed.get(id(node), 0.0) > 0:
            entries.append((node, True, displayed[id(node)]))
        entries.extend((child, False, masses[id(child)]) for child in node.children
                       if masses[id(child)] > 0)
        rectangles = squarified_rectangles([entry[2] for entry in entries], rect)
        for (entry, own_atom, value), child_rect in zip(entries, rectangles, strict=True):
            if own_atom:
                tiles.append((entry, child_rect, value))
            else:
                visit(entry, child_rect)
                frames.append((entry, child_rect))

    if total > 0:
        visit(tree, bounds)
    audit = {"syn_tolerance_bits": syn_tolerance,
             "tolerance_zero_count": sum(-syn_tolerance <= value < 0 for value in values),
             "zero_area_atom_count": sum(value == 0 for value in displayed.values()),
             "minimum_syn_bits": min(values, default=0.0),
             "display_total_bits": total,
             "display_minus_raw_xi_bits": total - float(tree.phi_value)}
    return tiles, frames, audit


def draw_synergy_treemap(axis: plt.Axes, tree: PhiTreeNode, *,
                        source_labels: Mapping[str, str] | None = None,
                        decimals: int = 3, syn_scale_max: float | None = None,
                        syn_tolerance: float = 1e-10) -> dict:
    """Draw exact atom areas with fitted labels and an outside atom ledger.

    The ledger retains every atom, including zero-area atoms and tiles too
    small for labels. Its indentation records depth, including zero-mass
    splits that cannot have visible frames. Returns the numerical audit.
    """
    labels = dict(source_labels or {})
    tiles, frames, audit = synergy_treemap_layout(tree, syn_tolerance=syn_tolerance)
    atoms = [n for n in flatten_nodes(tree) if n.atom_kind is not None]
    ids = {id(node): i for i, node in enumerate(atoms, 1)}
    maximum = max((value for _, _, value in tiles), default=0.0)
    scale = maximum if syn_scale_max is None else float(syn_scale_max)
    if not math.isfinite(scale) or scale < maximum - syn_tolerance:
        raise ValueError("syn_scale_max must be finite and cover every displayed Syn")
    norm = mpl.colors.Normalize(0, scale if scale > 0 else 1)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("syn_teal", ["#EAF3F1", SYN_COLOR])
    axis.set(xlim=(-0.02, 6.02), ylim=(-0.02, 4.02))
    axis.set_aspect("equal")
    axis.axis("off")
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    total = audit["display_total_bits"]
    for node, (x, y, w, h), value in tiles:
        face = cmap(norm(value))
        patch = Rectangle((x, y), w, h, facecolor=face, edgecolor="white", linewidth=1.5)
        axis.add_patch(patch)
        coalition = ", ".join(str(labels.get(s, s)) for s in node.sources)
        text = axis.text(x + w / 2, y + h / 2,
                         f"{coalition}\nSyn {value:.{decimals}f}\n{value / total:.1%}",
                         ha="center", va="center", color=_text_color(face),
                         fontsize=10, linespacing=1.45)
        box = patch.get_window_extent(renderer)
        candidates = [text.get_text(), f"#{ids[id(node)]}\n{value:.{decimals}f}",
                      f"#{ids[id(node)]}"]
        fitted = False
        for candidate in candidates:
            text.set_text(candidate)
            for size in (10, 9, 8, 7):
                text.set_fontsize(size)
                extent = text.get_window_extent(renderer)
                if extent.width < box.width - 12 and extent.height < box.height - 12:
                    fitted = True
                    break
            if fitted:
                break
        if not fitted:
            text.remove()
        else:
            text.set_clip_path(patch)
    # Frames use the original bounds: no inset that could change area ratios.
    for node, (x, y, w, h) in frames:
        axis.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=INK,
                                 linewidth=max(.5, 1.6 - .2 * (node.depth - tree.depth))))
    axis.add_patch(Rectangle((0, 0), 6, 4, fill=False, edgecolor=INK, linewidth=1.5))
    if total == 0:
        axis.text(3, 2, "No positive Syn area", ha="center", va="center", color=INK)
    axis.text(1.04, 1, "SPT atoms · Syn (bits)", transform=axis.transAxes,
              va="top", fontsize=9, weight="bold", color=INK)
    spacing = min(.12, .78 / max(1, len(atoms)))
    for index, node in enumerate(atoms, 1):
        members = ", ".join(str(labels.get(s, s)) for s in node.sources)
        depth = node.depth - tree.depth
        axis.text(1.04 + .016 * depth, .93 - spacing * (index - 1),
                  f"#{index}  {{{members}}}\n     {node.residual:.{decimals}f}",
                  transform=axis.transAxes, va="top", fontsize=8, color=INK,
                  linespacing=1.25)
    axis.text(1.04, -.02, "Indent = depth\nFrames = subtrees\nZero Syn = no area",
              transform=axis.transAxes, va="top", fontsize=7.5, color="#566573")
    return audit


def plot_synergy_treemap(tree, output_path, *, source_labels=None, decimals=3,
                         show_root_total=True, syn_scale_max=None,
                         syn_tolerance=1e-10, dpi=600):
    """Save an opaque PNG; provenance and numerical audit in PNG metadata."""
    import json
    from pathlib import Path

    output = Path(output_path)
    if output.suffix.lower() != ".png":
        raise ValueError("The reusable renderer writes one .png figure")
    with mpl.rc_context({"font.family": "sans-serif", "font.size": 8}):
        fig, axis = plt.subplots(figsize=(9.2, 5.2))
        fig.subplots_adjust(left=.025, right=.64, top=.88, bottom=.14)
        try:
            audit = draw_synergy_treemap(
                axis, tree, source_labels=source_labels, decimals=decimals,
                syn_scale_max=syn_scale_max, syn_tolerance=syn_tolerance)
            if show_root_total:
                axis.text(0, 1.08, rf"$\Xi$ = {tree.phi_value:.{decimals}f} bits",
                          transform=axis.transAxes, color=INK, fontsize=12, weight="bold")
            axis.text(0, -.085, "Area = Syn / total Syn  |  Fill = Syn (bits)",
                      transform=axis.transAxes, fontsize=8, color=INK)
            axis.text(0, -.15,
                      f"Numerical tolerance: {syn_tolerance:g} bits; "
                      f"{audit['tolerance_zero_count']} negatives displayed as zero",
                      transform=axis.transAxes, fontsize=7, color="#566573")
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white", metadata={
                "Description": "Nested SPT treemap; area encodes each local Syn share. "
                               "The outside ledger lists raw atoms in depth-first order.",
                "SPT numerical audit": json.dumps(audit, sort_keys=True),
                "Visualization workflow": "Scientific Agent Skills; doi:10.48550/arXiv.2609.00065"})
        finally:
            plt.close(fig)
    return output
