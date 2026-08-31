from __future__ import annotations

import json
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_greedy_hierarchy_kuramoto import (
    FREQUENCIES,
    PLANTED_MODULES,
    coupling_matrix,
    is_planted_split,
    kuramoto_derivative,
)
from scripts.phi_hierarchy import (
    ALL_ORDER_CROSS_DENSITY,
    cross_coalition_count,
    flatten_phi_tree,
    greedy_phi_atoms,
    greedy_phi_tree,
)
from scripts.plot_kuramoto_xi_hierarchy_trees import _ei_table, render_all
from scripts.plot_kuramoto_hierarchy_report_assets import render_network
from scripts.synergy_hierarchy_tree_plot import plot_synergy_hierarchy_tree


def representative_tree():
    summary = json.loads(
        (ROOT / "results/greedy_hierarchy_kuramoto/summary.json").read_text(encoding="utf-8")
    )
    row = summary["representative"]
    sources = tuple(source for block in row["root_split"] for source in block)
    return greedy_phi_tree(sources, _ei_table(row)), row


def test_coupling_matrix_has_two_normalized_planted_communities() -> None:
    matrix = coupling_matrix(within_coupling=1.5, cross_coupling=0.3)

    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert np.allclose(matrix[:3, :3][~np.eye(3, dtype=bool)], 0.75)
    assert np.allclose(matrix[:3, 3:], 0.1)


def test_zero_cross_coupling_blocks_cross_module_influence() -> None:
    phases = np.array([[0.1, -0.4, 0.8, -1.0, 0.3, 1.2]])
    changed = phases.copy()
    changed[:, 3:] += np.array([0.7, -0.2, 1.1])

    baseline = kuramoto_derivative(phases, within_coupling=1.5, cross_coupling=0.0)
    perturbed = kuramoto_derivative(changed, within_coupling=1.5, cross_coupling=0.0)

    assert np.allclose(baseline[:, :3], perturbed[:, :3])


def test_vector_field_is_equivariant_to_global_phase_shift() -> None:
    phases = np.array([[0.1, -0.4, 0.8, -1.0, 0.3, 1.2]])

    baseline = kuramoto_derivative(phases, within_coupling=1.5, cross_coupling=0.4)
    shifted = kuramoto_derivative(phases + 1.37, within_coupling=1.5, cross_coupling=0.4)

    assert np.allclose(baseline, shifted)
    assert np.allclose(baseline.mean(axis=1), FREQUENCIES.mean())


def test_planted_split_is_order_invariant() -> None:
    assert is_planted_split(PLANTED_MODULES[0], PLANTED_MODULES[1])
    assert is_planted_split(PLANTED_MODULES[1], PLANTED_MODULES[0])
    assert not is_planted_split(("theta1",), tuple(name for name in sum(PLANTED_MODULES, ()) if name != "theta1"))


def test_explicit_tree_preserves_atoms_and_kuramoto_root_split() -> None:
    tree, row = representative_tree()

    assert [child.sources for child in tree.children] == [tuple(block) for block in row["root_split"]]
    assert np.isclose(tree.phi_value, row["root_phi_bits"])
    assert np.isclose(tree.residual, row["root_residual_bits"])
    assert flatten_phi_tree(tree) == greedy_phi_atoms(tree.sources, _ei_table(row))
    assert np.isclose(sum(atom.value for atom in flatten_phi_tree(tree)), tree.phi_value)


def test_all_order_cross_count_and_search_only_normalization() -> None:
    assert cross_coalition_count(1, 3) == 7
    assert cross_coalition_count(2, 2) == 9

    names = ("a", "b", "c", "d")
    table = {subset: 0.0 for size in range(1, 5) for subset in itertools.combinations(names, size)}
    table[("a", "b")] = 4.4
    table[("c", "d")] = 4.4
    table[("b", "c", "d")] = 9.0
    table[names] = 10.0

    raw = greedy_phi_tree(names, table)
    normalized = greedy_phi_tree(names, table, split_objective=ALL_ORDER_CROSS_DENSITY)

    assert {child.sources for child in raw.children} == {("a",), ("b", "c", "d")}
    assert {child.sources for child in normalized.children} == {("a", "b"), ("c", "d")}
    assert np.isclose(raw.residual, 1.0)
    assert np.isclose(normalized.residual, 1.2)


def test_minimal_tree_renderer_writes_one_png(tmp_path: Path) -> None:
    tree, _ = representative_tree()
    output = tmp_path / "tree.png"

    returned = plot_synergy_hierarchy_tree(tree, output, dpi=100)

    assert returned == output
    assert output.is_file()
    assert output.stat().st_size > 1_000


def test_all_kuramoto_structure_trees_render_separately(tmp_path: Path) -> None:
    outputs = render_all(
        ROOT / "results/greedy_hierarchy_kuramoto/summary.json",
        tmp_path,
        seed=0,
        dpi=100,
    )

    assert [path.name for path in outputs] == [
        "kuramoto_xi_tree_kout_0p00.png",
        "kuramoto_xi_tree_kout_0p25.png",
        "kuramoto_xi_tree_kout_0p75.png",
        "kuramoto_xi_tree_kout_1p50.png",
    ]
    assert all(path.stat().st_size > 1_000 for path in outputs)


def test_kuramoto_network_renderer_writes_one_png(tmp_path: Path) -> None:
    output = tmp_path / "network.png"

    returned = render_network(
        within_coupling=1.5,
        cross_coupling=0.75,
        output_path=output,
        dpi=100,
    )

    assert returned == output
    assert output.is_file()
    assert output.stat().st_size > 1_000
