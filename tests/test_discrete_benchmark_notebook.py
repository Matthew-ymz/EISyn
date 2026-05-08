import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from utils import BENCHMARK_NETWORKS, render_topology_mechanism_svg


def load_notebook_namespace_until(notebook_path: Path, stop_when: str) -> dict[str, object]:
    notebook = json.loads(notebook_path.read_text())
    namespace: dict[str, object] = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        exec(compile(source, f"{notebook_path.name}_cell_{index}", "exec"), namespace, namespace)
        if stop_when in source:
            break
    return namespace


def execute_notebook(notebook_path: Path) -> dict[str, object]:
    notebook = json.loads(notebook_path.read_text())
    namespace: dict[str, object] = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        exec(compile(source, f"{notebook_path.name}_cell_{index}", "exec"), namespace, namespace)
    return namespace


class DiscreteBenchmarkNotebookTests(unittest.TestCase):
    def test_e_topology_is_dense_two_community_weak_bridge(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "exp" / "discrete_benchmark.ipynb"
        namespace = execute_notebook(notebook_path)

        adjacency_e = namespace["BENCHMARK_NETWORKS"]["E"]["adjacency"]

        community_a = {0, 1, 2, 3}
        community_b = {4, 5, 6, 7}
        edges_e = {
            (int(src), int(dst))
            for src, dst in zip(*np.nonzero(adjacency_e), strict=True)
        }
        cross_edges_e = {
            (src, dst)
            for src, dst in edges_e
            if (src in community_a and dst in community_b)
            or (src in community_b and dst in community_a)
        }
        intra_edges_a = {
            (src, dst)
            for src, dst in edges_e
            if src in community_a and dst in community_a
        }
        intra_edges_b = {
            (src, dst)
            for src, dst in edges_e
            if src in community_b and dst in community_b
        }

        self.assertEqual(namespace["BENCHMARK_NETWORKS"]["E"]["label"], "dense two-community weak bridge")
        self.assertEqual(int(np.count_nonzero(adjacency_e)), 12)
        self.assertEqual(len(cross_edges_e), 2)
        self.assertGreaterEqual(len(intra_edges_a), 5)
        self.assertGreaterEqual(len(intra_edges_b), 5)

    def test_e_synergy_stays_below_f_in_replacement_design(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "exp" / "discrete_benchmark.ipynb"
        namespace = execute_notebook(notebook_path)

        results = namespace["benchmark_results"]

        self.assertLess(results["E"]["phi_eid"], results["F"]["phi_eid"])
        self.assertLess(results["E"]["ei"], results["F"]["ei"])

    def test_e_topology_adds_requested_parity_groups(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "exp" / "discrete_benchmark.ipynb"
        namespace = execute_notebook(notebook_path)

        node_specs = namespace["BENCHMARK_NETWORKS"]["E"]["node_specs"]

        self.assertEqual(sorted(node_specs[0]["parity_sources"]), [1, 3])
        self.assertEqual(sorted(node_specs[3]["parity_sources"]), [1, 2])
        self.assertEqual(sorted(node_specs[5]["parity_sources"]), [4, 7])
        self.assertEqual(sorted(node_specs[7]["parity_sources"]), [5, 6])

    def test_coop_nodes_now_use_all_incoming_sources(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "exp" / "discrete_benchmark.ipynb"
        namespace = execute_notebook(notebook_path)

        network_b = namespace["BENCHMARK_NETWORKS"]["B"]
        incoming_to_node_2 = sorted(int(src) for src in np.flatnonzero(network_b["adjacency"][:, 2] > 0))
        incoming_to_node_4 = sorted(int(src) for src in np.flatnonzero(network_b["adjacency"][:, 4] > 0))

        self.assertEqual(sorted(network_b["node_specs"][2]["coop_sources"]), incoming_to_node_2)
        self.assertEqual(sorted(network_b["node_specs"][4]["coop_sources"]), incoming_to_node_4)

    def test_d_topology_uses_non_adjacent_sources_for_each_target(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "exp" / "discrete_benchmark.ipynb"
        namespace = execute_notebook(notebook_path)

        network_d = namespace["BENCHMARK_NETWORKS"]["D"]
        adjacency_d = network_d["adjacency"]
        n_nodes = adjacency_d.shape[0]

        for target in range(n_nodes):
            incoming = sorted(int(src) for src in np.flatnonzero(adjacency_d[:, target] > 0))
            self.assertEqual(len(incoming), 2)
            self.assertEqual(sorted(network_d["node_specs"][target]["coop_sources"]), incoming)

            circular_distances = {
                min((target - src) % n_nodes, (src - target) % n_nodes)
                for src in incoming
            }
            self.assertEqual(circular_distances, {2})

    def test_notebook_is_slim_and_focuses_on_topology_plus_ei_phi_outputs(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "exp" / "discrete_benchmark.ipynb"
        notebook = json.loads(notebook_path.read_text())

        markdown_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        code_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertLessEqual(len(notebook["cells"]), 8)
        self.assertIn("from utils import", code_text)
        self.assertIn("render_benchmark_topology_overview_svg", code_text)
        self.assertIn("render_metric_bar_chart_svg", code_text)
        self.assertNotIn("极小网络补充实验", markdown_text)
        self.assertNotIn("mini_results", code_text)
        self.assertNotIn("rho_high", code_text)
        self.assertNotIn("Two-node", markdown_text)
        self.assertNotIn("build_benchmark_summary_table_html", code_text)

    def test_outputs_use_lowercase_panel_labels_and_metric_bar_charts(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "exp" / "discrete_benchmark.ipynb"
        namespace = execute_notebook(notebook_path)

        topology_svg = namespace["topology_overview_svg"]
        ei_svg = namespace["ei_bar_chart_svg"]
        phi_svg = namespace["phi_bar_chart_svg"]
        summary_svg = namespace["ei_phi_summary_svg"]

        for letter in "abcdef":
            self.assertIn(f">{letter}</text>", topology_svg)
            self.assertIn(f">{letter}</text>", ei_svg)
            self.assertIn(f">{letter}</text>", phi_svg)
            self.assertIn(f">{letter}</text>", summary_svg)
        self.assertNotIn(">A</text>", topology_svg)
        self.assertNotIn("<table", ei_svg)
        self.assertNotIn("<table", phi_svg)
        self.assertNotIn("$\\Phi^{\\mathrm{EID}}$", phi_svg)
        self.assertIn(">EID</tspan>", phi_svg)
        self.assertIn(">Φ</tspan>", phi_svg)
        self.assertIn("baseline-shift='super'", phi_svg)
        self.assertIn(">Total EI</text>", summary_svg)
        self.assertIn(">EID</tspan>", summary_svg)
        self.assertIn(">Φ</tspan>", summary_svg)
        self.assertIn("baseline-shift='super'", summary_svg)

    def test_topology_label_style_is_exposed_in_notebook(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "exp" / "discrete_benchmark.ipynb"
        namespace = execute_notebook(notebook_path)

        label_style = namespace["topology_label_style"]
        topology_svg = namespace["topology_overview_svg"]

        self.assertEqual(label_style["font_size"], 25)
        self.assertEqual(label_style["dx"], 1)
        self.assertEqual(label_style["dy"], -1)
        self.assertTrue(label_style["show"])
        self.assertIn("font-size='25'", topology_svg)
        self.assertIn("x='21'", topology_svg)
        self.assertIn("y='19'", topology_svg)

    def test_topology_layout_and_legend_are_exposed_in_notebook(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "exp" / "discrete_benchmark.ipynb"
        namespace = execute_notebook(notebook_path)

        layout_style = namespace["topology_layout_style"]
        legend_style = namespace["topology_legend_style"]
        connection_style = namespace["topology_connection_style"]
        connection_filter_style = namespace["topology_connection_filter_style"]
        topology_svg = namespace["topology_overview_svg"]

        self.assertEqual(layout_style["panel_width"], 260)
        self.assertEqual(layout_style["panel_height"], 260)
        self.assertEqual(layout_style["horizontal_gap"], 18)
        self.assertEqual(layout_style["vertical_gap"], 22)
        self.assertEqual(legend_style["x"], 850)
        self.assertEqual(legend_style["y"], 20)
        self.assertIn("copy", connection_style)
        self.assertIn("cooperation", connection_style)
        self.assertIn("parity", connection_style)
        self.assertTrue(connection_filter_style["drop_shared_parity_sources"])
        self.assertIn("Mechanism legend", topology_svg)
        self.assertIn("copy", topology_svg)
        self.assertIn("cooperation", topology_svg)
        self.assertIn("parity", topology_svg)
        self.assertIn("id='copy_arrowhead'", topology_svg)
        self.assertIn("id='cooperation_arrowhead'", topology_svg)
        self.assertIn("id='parity_arrowhead'", topology_svg)
        self.assertNotIn(">coop</text>", topology_svg)
        self.assertEqual(topology_svg.count(">parity</text>"), 1)

    def test_f_topology_is_parity_only(self) -> None:
        self.assertEqual(int((BENCHMARK_NETWORKS["F"]["adjacency"] > 0).sum()), 0)
        for spec in BENCHMARK_NETWORKS["F"]["node_specs"]:
            self.assertEqual(spec["alpha"], 0.0)
            self.assertEqual(spec["beta"], 0.0)
            self.assertEqual(len(spec["coop_sources"]), 0)
            self.assertEqual(len(spec["parity_sources"]), 2)

        svg = render_topology_mechanism_svg(
            "F",
            str(BENCHMARK_NETWORKS["F"]["label"]),
            BENCHMARK_NETWORKS["F"]["adjacency"],
            BENCHMARK_NETWORKS["F"]["node_specs"],
            show_summary_box=False,
            show_panel_title=False,
            show_mechanism_hint=False,
            show_mechanism_text_labels=False,
            connection_style={
                "copy": {
                    "stroke": "#93a1af",
                    "marker_width": 8.5,
                    "marker_height": 7.0,
                    "marker_ref_x": 7.2,
                    "marker_ref_y": 3.5,
                },
                "cooperation": {
                    "stroke": "#e17c05",
                    "dash": "6,4",
                    "marker_width": 8.0,
                    "marker_height": 7.0,
                    "marker_ref_x": 7.0,
                    "marker_ref_y": 3.5,
                },
                "parity": {
                    "stroke": "#7b4ab5",
                    "dash": "1.5,4.5",
                    "marker_shape": "diamond",
                    "marker_width": 8.0,
                    "marker_height": 8.0,
                    "marker_ref_x": 7.0,
                    "marker_ref_y": 4.0,
                },
            },
            connection_filter_style={"drop_shared_parity_sources": True},
        )

        self.assertEqual(svg.count("marker-end='url(#copy_arrowhead)'"), 0)
        self.assertEqual(svg.count("marker-end='url(#cooperation_arrowhead)'"), 0)
        self.assertEqual(svg.count("marker-end='url(#parity_arrowhead)'"), 16)


if __name__ == "__main__":
    unittest.main()
