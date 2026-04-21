import json
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


class RQ3ManualCaseNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.notebook_path = cls.repo_root / "exp" / "rq3_manual_case.ipynb"
        cls.notebook = json.loads(cls.notebook_path.read_text())
        cls.namespace = execute_notebook(cls.notebook_path)

    def test_notebook_keeps_a_thin_story_and_nearby_interpretation(self) -> None:
        markdown_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        code_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertIn("## 微观拓扑图解读", markdown_text)
        self.assertIn("## 最优粗粒化图解读", markdown_text)
        self.assertIn("## 代表性非最优粗粒化图解读", markdown_text)
        self.assertIn("关键指标", markdown_text)
        self.assertIn("run_manual_case_analysis", code_text)
        self.assertNotIn("def build_manual_micro_tpm", code_text)
        self.assertNotIn("def render_manual_micro_topology_svg", code_text)

    def test_notebook_recovers_manual_case_metrics(self) -> None:
        summary = self.namespace["MANUAL_CASE_RESULTS"]

        self.assertEqual(summary["n_micro_nodes"], 6)
        self.assertEqual(summary["n_macro_nodes"], 3)
        self.assertEqual(summary["candidate_count"], 90)
        self.assertEqual(summary["optimal_groups"], ((0, 1), (2, 3), (4, 5)))
        self.assertAlmostEqual(summary["optimal_ei"], 3.0, places=12)
        self.assertAlmostEqual(summary["optimal_syn"], 0.0, places=12)
        self.assertEqual(summary["nonoptimal_groups"], ((0, 1), (2, 4), (3, 5)))
        self.assertAlmostEqual(summary["nonoptimal_ei"], 1.6582251715410803, places=12)
        self.assertAlmostEqual(summary["nonoptimal_syn"], 0.39268642241731855, places=12)
        self.assertAlmostEqual(summary["nonoptimal_max_hyperedge"], 0.14982588955666737, places=12)
        self.assertGreater(summary["optimal_ei"], summary["or_partition_mean_ei"])
        self.assertGreater(summary["nonoptimal_syn"], summary["or_partition_mean_syn"])

    def test_notebook_exposes_figures_tables_and_cache_artifacts(self) -> None:
        summary = self.namespace["MANUAL_CASE_RESULTS"]
        topology_svg = self.namespace["topology_svg"]
        optimal_svg = self.namespace["optimal_comparison_svg"]
        nonoptimal_svg = self.namespace["nonoptimal_comparison_svg"]
        metrics_html = self.namespace["metrics_html"]
        cache_path = Path(self.namespace["manual_case_cache_path"])

        self.assertNotIn("Micro topology", topology_svg)
        self.assertNotIn(
            "context-gated pairs encode a macro cycle with only pairwise edges",
            topology_svg,
        )
        self.assertIn("AND/OR", topology_svg)
        self.assertIn("copy", topology_svg)
        self.assertNotIn("micro to macro comparison", optimal_svg)
        self.assertIn("A(t)", optimal_svg)
        self.assertIn("a1(t)", optimal_svg)
        self.assertIn("B(t+1)", optimal_svg)
        self.assertNotIn("micro to macro comparison", nonoptimal_svg)
        self.assertIn("关键指标", metrics_html)
        self.assertIn("3.000", metrics_html)
        self.assertIn("1.658", metrics_html)
        self.assertTrue(cache_path.exists())

        cache_payload = json.loads(cache_path.read_text())
        self.assertEqual(cache_payload["summary"]["optimal_groups"], [[0, 1], [2, 3], [4, 5]])
        self.assertEqual(cache_payload["summary"]["nonoptimal_groups"], [[0, 1], [2, 4], [3, 5]])

        fig_dir = self.repo_root / "fig" / "rq3_boolean_causal_emergence"
        self.assertTrue((fig_dir / "micro_topology_overview.png").exists())
        self.assertTrue((fig_dir / "micro_macro_coarse_graining_comparison.png").exists())
        self.assertTrue((fig_dir / "non_optimal_macro_comparison.png").exists())

        results_dir = self.repo_root / "results" / "rq3_manual_case"
        self.assertTrue((results_dir / "manual_case_summary.json").exists())
        self.assertTrue((results_dir / "metrics_table.html").exists())
        self.assertEqual(summary["figure_dir"], str(fig_dir))

    def test_topology_svg_keeps_the_original_manual_canvas_shape(self) -> None:
        topology_svg = self.namespace["topology_svg"]

        self.assertIn("viewBox='0 0 560 340'", topology_svg)
        self.assertIn("width='560'", topology_svg)
        self.assertIn("height='340'", topology_svg)
        self.assertIn(">a1</text>", topology_svg)
        self.assertIn(">a2</text>", topology_svg)
        self.assertIn(">c2</text>", topology_svg)
        self.assertNotIn("Mechanisms", topology_svg)
        self.assertIn("font-size='12.5' fill='#333'>copy</text>", topology_svg)
        self.assertIn("font-size='12.5' fill='#333'>AND/OR</text>", topology_svg)

    def test_micro_mechanism_hyperedges_follow_the_manual_dynamics(self) -> None:
        micro_mechanism_hyperedges = self.namespace["MICRO_MECHANISM_HYPEREDGES"]

        expected = {
            0: (2, 3, 4, 5),
            1: (2, 3, 4, 5),
            2: (0, 1, 4, 5),
            3: (0, 1, 4, 5),
            4: (0, 1, 2, 3),
            5: (0, 1, 2, 3),
        }
        self.assertEqual(len(micro_mechanism_hyperedges), 6)
        for row in micro_mechanism_hyperedges:
            self.assertEqual(tuple(row["sources"]), expected[row["target"]])

    def test_notebook_also_runs_when_cwd_is_exp_directory(self) -> None:
        notebook = json.loads(self.notebook_path.read_text())
        setup_source = "".join(notebook["cells"][1]["source"])
        import subprocess
        import sys

        completed = subprocess.run(
            [sys.executable, "-c", setup_source],
            cwd=self.notebook_path.parent,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
