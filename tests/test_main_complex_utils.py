import unittest

import numpy as np

from utils import (
    discrete_subset_synergy,
    enumerate_binary_states,
    render_coarse_graining_comparison_svg,
    render_topology_mechanism_svg,
    render_static_causal_graph_svg,
)


def build_demo_tpm() -> np.ndarray:
    states = enumerate_binary_states(5)
    index_by_state = {tuple(state.tolist()): idx for idx, state in enumerate(states)}
    tpm = np.zeros((len(states), len(states)), dtype=float)
    for row_index, state in enumerate(states):
        a, b, c, d, e = (int(value) for value in state)
        future = (
            int(b and c),
            int(a and c),
            int(a and b),
            int(a and b),
            int(a and b),
        )
        tpm[row_index, index_by_state[future]] = 1.0
    return tpm


class MainComplexUtilityTests(unittest.TestCase):
    def test_render_topology_mechanism_svg_can_hide_summary_box(self) -> None:
        adjacency = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)
        node_specs = [
            {"beta": 0.0, "gamma": 0.0},
            {"beta": 0.6, "gamma": 0.8, "coop_pairs": [(0, 1)]},
        ]

        svg = render_topology_mechanism_svg(
            "Demo",
            "two-node",
            adjacency,
            node_specs,
            show_summary_box=False,
        )

        self.assertIn("Demo: two-node", svg)
        self.assertNotIn("Mechanism summary", svg)
        self.assertNotIn("beta=0.60", svg)
        self.assertNotIn("gamma=0.80", svg)

    def test_render_topology_mechanism_svg_can_hide_mechanism_hint(self) -> None:
        adjacency = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)
        node_specs = [
            {"beta": 0.0, "gamma": 0.0},
            {"beta": 0.6, "gamma": 0.8, "coop_pairs": [(0, 1)]},
        ]

        svg = render_topology_mechanism_svg(
            "Demo",
            "two-node",
            adjacency,
            node_specs,
            show_summary_box=False,
            show_mechanism_hint=False,
        )

        self.assertNotIn("fill=parity strength, border=cooperation strength", svg)

    def test_render_static_causal_graph_svg_uses_single_slice_layout(self) -> None:
        svg = render_static_causal_graph_svg(
            "Boolean core",
            node_labels=["A", "B", "C", "D", "E"],
            directed_edges=[],
            hyperedges=[
                {"sources": (1, 2), "target": 0, "label": "AND"},
                {"sources": (0, 1), "target": 3, "label": "AND"},
            ],
            subtitle="single-slice causal topology",
            width=520,
            height=320,
        )

        self.assertIn("Boolean core", svg)
        self.assertIn("single-slice causal topology", svg)
        self.assertIn(">A</text>", svg)
        self.assertIn(">E</text>", svg)
        self.assertIn(">AND</text>", svg)
        self.assertNotIn("sources at t", svg)
        self.assertNotIn("targets at t+1", svg)

    def test_discrete_subset_synergy_ranks_whole_above_core_in_demo_network(self) -> None:
        tpm = build_demo_tpm()

        core = discrete_subset_synergy(
            tpm,
            n_nodes=5,
            subset_indices=(0, 1, 2),
        )
        whole = discrete_subset_synergy(
            tpm,
            n_nodes=5,
            subset_indices=(0, 1, 2, 3, 4),
        )

        self.assertGreater(core["synergy"], 0.0)
        self.assertAlmostEqual(core["synergy"], whole["synergy"], places=8)
        self.assertAlmostEqual(core["whole_ei"], 2.0, places=8)
        self.assertAlmostEqual(whole["whole_ei"], 2.0, places=8)

    def test_render_coarse_graining_comparison_svg_uses_wider_safe_layout(self) -> None:
        svg = render_coarse_graining_comparison_svg(
            "Demo",
            micro_pairwise=np.zeros((2, 2), dtype=float),
            micro_labels=["long_micro_left", "long_micro_right"],
            macro_labels=["Alpha", "Beta"],
            groups=[(0,), (1,)],
            macro_pairwise=np.zeros((2, 2), dtype=float),
        )

        self.assertIn("width='1180'", svg)
        self.assertIn("height='820'", svg)
        self.assertIn("x='1030.0'", svg)
        self.assertIn("long_micro_right", svg)


if __name__ == "__main__":
    unittest.main()
