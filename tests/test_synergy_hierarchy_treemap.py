"""Check scientific encodings, not merely whether a renderer writes a file."""

import itertools
import json

import numpy as np
import matplotlib as mpl
import pytest
from PIL import Image

from scripts.spt import SPTNode
from scripts.synergy_hierarchy_tree_plot import plot_synergy_hierarchy_tree
from scripts.synergy_hierarchy_treemap import (
    squarified_rectangles, synergy_treemap_layout,
    _text_color,
)


def example_tree(root_syn=1.0):
    a = SPTNode(("a",), 0, 0, 2, "singleton")
    b = SPTNode(("b",), 0, 0, 2, "singleton")
    ab = SPTNode(("a", "b"), 3, 3, 1, "exact", (a, b))
    c = SPTNode(("c",), 0, 0, 1, "singleton")
    return SPTNode(("a", "b", "c"), 3 + root_syn, root_syn, 0, "exact", (ab, c))


def test_packing_preserves_areas_and_never_overlaps():
    weights = [17, 1, 3, 8, 2, .001]
    rects = squarified_rectangles(weights, (2, 3, 6, 4))
    for weight, (x, y, w, h) in zip(weights, rects):
        assert w * h == pytest.approx(24 * weight / sum(weights))
        assert 2 <= x <= x + w <= 8 + 1e-12
        assert 3 <= y <= y + h <= 7 + 1e-12
    for (x, y, w, h), (u, v, p, q) in itertools.combinations(rects, 2):
        overlap = max(0, min(x + w, u + p) - max(x, u)) * max(0, min(y + h, v + q) - max(y, v))
        assert overlap < 1e-12


def test_nested_layout_encodes_local_atoms_without_double_counting():
    tree = example_tree()
    tiles, frames, audit = synergy_treemap_layout(tree)
    assert audit["display_total_bits"] == 4
    assert len(tiles) == 2
    assert {n.sources for n, _, _ in tiles} == {tree.sources, ("a", "b")}
    assert sum(rect[2] * rect[3] for _, rect, _ in tiles) == pytest.approx(24)
    for _, (_, _, w, h), value in tiles:
        assert w * h / 24 == pytest.approx(value / tree.phi_value)
    frame = next(rect for n, rect in frames if n.sources == ("a", "b"))
    child = next(rect for n, rect, _ in tiles if n.sources == ("a", "b"))
    assert frame == child


def test_numerical_zero_is_explicit_and_raw_tree_is_preserved(tmp_path):
    tree = example_tree(-1e-11)
    _, _, audit = synergy_treemap_layout(tree, syn_tolerance=1e-10)
    assert tree.residual == -1e-11
    assert audit["tolerance_zero_count"] == 1
    assert audit["display_minus_raw_xi_bits"] == pytest.approx(1e-11, abs=1e-15)
    output = plot_synergy_hierarchy_tree(tree, tmp_path / "tree.png", dpi=80)
    with Image.open(output) as image:
        saved_audit = json.loads(image.info["SPT numerical audit"])
        assert saved_audit["tolerance_zero_count"] == 1
        assert image.getpixel((0, 0)) == (255, 255, 255, 255)


def test_significant_negative_fails_with_minimum_threshold_and_count():
    with pytest.raises(ValueError, match=r"minimum=-0.1 bits.*threshold=-1e-10 bits, affected_count=1"):
        synergy_treemap_layout(example_tree(-.1))


def test_all_zero_and_terminal_coalition_are_supported():
    zero = SPTNode(("a", "b"), 0, 0, 0, "terminal")
    tiles, frames, audit = synergy_treemap_layout(zero)
    assert not tiles and not frames
    assert audit["zero_area_atom_count"] == 1
    terminal = SPTNode(("a", "b"), 2, 2, 0, "terminal")
    tiles, _, audit = synergy_treemap_layout(terminal)
    assert len(tiles) == 1 and tiles[0][1] == (0, 0, 6, 4)


def test_inconsistent_subtree_total_fails():
    broken = SPTNode(("a", "b"), 10, 2, 0, "terminal")
    with pytest.raises(ValueError, match="closure failed"):
        synergy_treemap_layout(broken)


@pytest.mark.parametrize("tolerance", [-1, np.nan, np.inf])
def test_invalid_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match="syn_tolerance"):
        synergy_treemap_layout(example_tree(), syn_tolerance=tolerance)


def test_legacy_layout_remains_available(tmp_path):
    output = plot_synergy_hierarchy_tree(example_tree(), tmp_path / "old.png",
                                         layout="node_link", dpi=80)
    assert output.stat().st_size > 1000


def test_tile_text_contrast_covers_the_entire_palette():
    def luminance(rgb):
        rgb = np.asarray(rgb[:3])
        linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
        return float(linear @ np.asarray([.2126, .7152, .0722]))

    cmap = mpl.colors.LinearSegmentedColormap.from_list("syn", ["#EAF3F1", "#267A70"])
    for value in np.linspace(0, 1, 256):
        face = cmap(value)
        ink = mpl.colors.to_rgb(_text_color(face))
        lower, upper = sorted([luminance(face), luminance(ink)])
        assert (upper + .05) / (lower + .05) >= 4.5
