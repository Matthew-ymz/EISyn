import json
import re
import unittest
from pathlib import Path


def execute_notebook(notebook_path: Path) -> dict[str, object]:
    notebook = json.loads(notebook_path.read_text())
    namespace: dict[str, object] = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        exec(compile(source, f"{notebook_path.name}_cell_{index}", "exec"), namespace, namespace)
    return namespace


class BooleanMotifCausalGraphsNotebookTests(unittest.TestCase):
    COMBINED_EDGE_COLOR = "#0f6b6f"

    def test_notebook_is_trimmed_to_two_figures_with_exposed_arrow_controls(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "boolean_motif_causal_graphs.ipynb"
        notebook = json.loads(notebook_path.read_text())
        code_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertLessEqual(len(notebook["cells"]), 6)
        self.assertNotIn("motif 小型恢复基准", json.dumps(notebook, ensure_ascii=False))
        for name in (
            "PAIRWISE_WIDTH_BASE",
            "PAIRWISE_WIDTH_SCALE",
            "HYPEREDGE_STROKE_WIDTH",
            "HYPEREDGE_TARGET_STROKE_WIDTH",
            "COMBINED_HYPEREDGE_MAX_TARGET_DIM",
            "ARROW_MARKER_WIDTH",
            "ARROW_MARKER_HEIGHT",
            "ARROW_MARKER_REF_X",
            "ARROW_MARKER_REF_Y",
            "PAIRWISE_CURVATURE_SCALE",
            "PAIRWISE_START_OFFSET",
            "PAIRWISE_END_OFFSET",
            "HYPEREDGE_SOURCE_OFFSET",
            "HYPEREDGE_TARGET_OFFSET",
            "HYPEREDGE_JUNCTION_GAP",
            "HYPEREDGE_JUNCTION_VERTICAL_GAP",
            "ARROW_JUNCTION_X_FRAC",
            "ARROW_JUNCTION_Y_OFFSET",
        ):
            self.assertIn(name, code_text)
        self.assertIn("箭头头宽度", code_text)
        self.assertIn("普通边基础线宽", code_text)
        self.assertIn("普通边弯曲强度", code_text)
        self.assertIn("超边汇合点横向位置", code_text)
        self.assertIn("超边汇合点纵向偏移", code_text)
        self.assertIn("超边圆圈纵向间距", code_text)
        self.assertIn("HYPEREDGE_COLOR", code_text)

        namespace = execute_notebook(notebook_path)
        manifest = namespace["manifest"]

        self.assertEqual(
            set(manifest["files"]),
            {
                "ground_truth",
                "combined_ei_graph",
            },
        )
        self.assertEqual(manifest["files"]["combined_ei_graph"], "combined_ei_graph.svg")
        self.assertEqual(
            manifest["render_params"]["hyperedge_junction_vertical_gap"],
            namespace["HYPEREDGE_JUNCTION_VERTICAL_GAP"],
        )
        self.assertEqual(
            manifest["main_example"]["combined_edge_color"],
            self.COMBINED_EDGE_COLOR,
        )
        self.assertEqual(namespace["COMBINED_HYPEREDGE_MAX_TARGET_DIM"], 1)
        self.assertEqual(
            manifest["main_example"]["combined_hyperedge_max_target_dim"],
            namespace["COMBINED_HYPEREDGE_MAX_TARGET_DIM"],
        )
        self.assertEqual(len(manifest["main_example"]["multi_target_hyperedge_rows"]), 1)
        self.assertEqual(
            {row["目标节点组"] for row in manifest["main_example"]["multi_target_hyperedge_rows"]},
            {"x1, x2"},
        )
        self.assertEqual(len(manifest["main_example"]["pairwise_weight_rows"]), 3)
        self.assertEqual(len(manifest["main_example"]["hyperedge_weight_rows"]), 1)
        self.assertEqual(len(namespace["combined_hyperedges"]), 1)
        self.assertTrue(all("target" in row for row in namespace["combined_hyperedges"]))
        self.assertTrue(all("targets" not in row for row in namespace["combined_hyperedges"]))

        ground_truth_svg = (
            project_root / "fig" / "boolean_motif_causal_graphs" / "ground_truth_mechanism.svg"
        ).read_text()
        combined_svg = (
            project_root / "fig" / "boolean_motif_causal_graphs" / "combined_ei_graph.svg"
        ).read_text()

        self.assertNotIn("真实机制图", ground_truth_svg)
        self.assertNotIn("Ground-truth mechanism", ground_truth_svg)
        self.assertNotIn("合并后 EI 图", combined_svg)
        self.assertNotIn(">Legend</text>", ground_truth_svg)
        for mechanism_color in ("#2c7bb6", "#e17c05", "#7b4ab5"):
            self.assertIn(mechanism_color, ground_truth_svg)
        self.assertIn(
            "stroke='#2c7bb6' stroke-width='2.0' marker-end='url(#cg-arrow)'/>",
            ground_truth_svg,
        )
        self.assertIn(self.COMBINED_EDGE_COLOR, combined_svg)
        for old_color in ("#2c7bb6", "#7b4ab5", "#8c6d1f"):
            self.assertNotIn(old_color, combined_svg)
        hyperedge_cys = [
            float(match.group(1))
            for match in re.finditer(
                rf"circle cx='223.2' cy='([0-9.]+)' r='[0-9.]+' fill='white' stroke='{self.COMBINED_EDGE_COLOR}'",
                combined_svg,
            )
        ]
        if len(hyperedge_cys) < 2:
            hyperedge_cys = [
                float(match.group(1))
                for match in re.finditer(
                    r"circle cx='223.2' cy='([0-9.]+)' r='[0-9.]+' fill='white' stroke='#[0-9a-f]+'",
                    combined_svg,
                )
            ]
        self.assertEqual(len(hyperedge_cys), 1)

    def test_main_example_uses_updated_and_and_copy_structure(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "boolean_motif_causal_graphs.ipynb"

        namespace = execute_notebook(notebook_path)
        build_main_example = namespace["build_main_example"]
        tpm, directed_edges, hyperedges = build_main_example()

        self.assertEqual(directed_edges, [(1, 0, "COPY")])
        self.assertEqual(
            hyperedges,
            [
                {"sources": (1, 2), "target": 1, "label": "AND", "value": 1.0},
                {"sources": (0, 1), "target": 2, "label": "XOR", "value": 1.0},
            ],
        )

        enumerate_binary_states = namespace["enumerate_binary_states"]
        states = enumerate_binary_states(namespace["N_NODES"])
        state_to_index = {tuple(state.tolist()): i for i, state in enumerate(states)}
        self.assertEqual(int(tpm[state_to_index[(0, 1, 1)], state_to_index[(1, 1, 1)]]), 1)
        self.assertEqual(int(tpm[state_to_index[(1, 1, 0)], state_to_index[(1, 0, 0)]]), 1)

    def test_pairwise_ei_matches_updated_copy_dynamics(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "boolean_motif_causal_graphs.ipynb"

        namespace = execute_notebook(notebook_path)
        pairwise_ei = namespace["main_summary"]["pairwise_ei"]

        self.assertEqual(float(pairwise_ei[1, 0]), 1.0)
        self.assertEqual(float(pairwise_ei[2, 0]), 0.0)

    def test_render_params_change_pairwise_widths_and_arrow_geometry(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "boolean_motif_causal_graphs.ipynb"

        namespace = execute_notebook(notebook_path)
        render_causal_graph_svg = namespace["render_causal_graph_svg"]
        pairwise = namespace["main_summary"]["pairwise_ei"]
        hyperedges = namespace["combined_hyperedges"]
        node_labels = namespace["NODE_LABELS"]

        svg_thin = render_causal_graph_svg(
            "",
            pairwise_matrix=pairwise,
            hyperedges=hyperedges,
            node_labels=node_labels,
            subtitle="",
            edge_threshold=namespace["EDGE_THRESHOLD"],
            hyperedge_threshold=namespace["HYPEREDGE_THRESHOLD"],
            show_edge_values=False,
            show_text_labels=False,
            pairwise_width_base=0.1,
            pairwise_width_scale=0.2,
            arrow_marker_width=8.0,
            arrow_marker_height=6.0,
            arrow_marker_ref_x=7.0,
            arrow_marker_ref_y=3.0,
            pairwise_curvature_scale=namespace["PAIRWISE_CURVATURE_SCALE"],
            pairwise_start_offset=namespace["PAIRWISE_START_OFFSET"],
            pairwise_end_offset=namespace["PAIRWISE_END_OFFSET"],
            hyperedge_source_offset=namespace["HYPEREDGE_SOURCE_OFFSET"],
            hyperedge_target_offset=namespace["HYPEREDGE_TARGET_OFFSET"],
            hyperedge_junction_gap=namespace["HYPEREDGE_JUNCTION_GAP"],
            hyperedge_junction_vertical_gap=namespace["HYPEREDGE_JUNCTION_VERTICAL_GAP"],
            hyperedge_junction_x_frac=namespace["ARROW_JUNCTION_X_FRAC"],
            hyperedge_junction_y_offset=namespace["ARROW_JUNCTION_Y_OFFSET"],
        )
        svg_thick = render_causal_graph_svg(
            "",
            pairwise_matrix=pairwise,
            hyperedges=hyperedges,
            node_labels=node_labels,
            subtitle="",
            edge_threshold=namespace["EDGE_THRESHOLD"],
            hyperedge_threshold=namespace["HYPEREDGE_THRESHOLD"],
            show_edge_values=False,
            show_text_labels=False,
            pairwise_width_base=4.0,
            pairwise_width_scale=6.0,
            arrow_marker_width=500.0,
            arrow_marker_height=700.0,
            arrow_marker_ref_x=7.0,
            arrow_marker_ref_y=3.0,
            pairwise_curvature_scale=namespace["PAIRWISE_CURVATURE_SCALE"],
            pairwise_start_offset=namespace["PAIRWISE_START_OFFSET"],
            pairwise_end_offset=namespace["PAIRWISE_END_OFFSET"],
            hyperedge_source_offset=namespace["HYPEREDGE_SOURCE_OFFSET"],
            hyperedge_target_offset=namespace["HYPEREDGE_TARGET_OFFSET"],
            hyperedge_junction_gap=namespace["HYPEREDGE_JUNCTION_GAP"],
            hyperedge_junction_vertical_gap=namespace["HYPEREDGE_JUNCTION_VERTICAL_GAP"],
            hyperedge_junction_x_frac=namespace["ARROW_JUNCTION_X_FRAC"],
            hyperedge_junction_y_offset=namespace["ARROW_JUNCTION_Y_OFFSET"],
        )

        thin_widths = re.findall(r"stroke-width='([0-9.]+)'", svg_thin)
        thick_widths = re.findall(r"stroke-width='([0-9.]+)'", svg_thick)
        self.assertNotEqual(thin_widths[:3], thick_widths[:3])
        self.assertIn("M 0 0 L 500.0 3.0 L 0 700.0 z", svg_thick)


if __name__ == "__main__":
    unittest.main()
