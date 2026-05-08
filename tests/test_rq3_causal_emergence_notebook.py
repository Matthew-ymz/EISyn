import csv
import json
import unittest
from pathlib import Path

import numpy as np

import utils as support


def execute_notebook(notebook_path: Path) -> dict[str, object]:
    notebook = json.loads(notebook_path.read_text())
    namespace: dict[str, object] = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        exec(compile(source, f"{notebook_path.name}_cell_{index}", "exec"), namespace, namespace)
    return namespace


class RQ3CausalEmergenceNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.notebook_path = cls.repo_root / "exp" / "rq3_boolean_causal_emergence.ipynb"
        cls.notebook = json.loads(cls.notebook_path.read_text())
        cls.namespace = execute_notebook(cls.notebook_path)

    def test_notebook_keeps_a_single_example_plus_family_story(self) -> None:
        markdown_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )

        self.assertIn("## Hoel Figure 2 toy example", markdown_text)
        self.assertIn("## Hoel Figure 2 micro-mechanism family", markdown_text)
        self.assertIn("## Family-level decomposition statistics", markdown_text)
        self.assertIn("micro mechanism", markdown_text)
        self.assertNotIn("shared-noise", markdown_text)
        self.assertIn("这里 `EI(Z->X+)` 与 `EI(S_M)` 恰好取到同一个数值", markdown_text)
        self.assertIn("箱图", markdown_text)
        self.assertNotIn("3+1", markdown_text)
        self.assertNotIn("代表性系统", markdown_text)

    def test_notebook_builds_the_hoel_family_grid(self) -> None:
        summary = self.namespace["RQ3_RESULTS"]

        self.assertEqual(summary["candidate_count_per_system"], 147)
        self.assertEqual(summary["pair_partition_count"], 3)
        self.assertEqual(summary["mapping_count_by_block_size"], {2: 7})
        self.assertEqual(summary["n_family_systems"], 21)
        self.assertEqual(summary["n_family_rho_points"], 21)
        self.assertEqual(summary["n_family_q_off_slices"], 21)
        self.assertEqual(len(summary["family_q_off_values"]), 21)
        self.assertAlmostEqual(float(summary["family_q_off_values"][0]), 0.0, places=12)
        self.assertAlmostEqual(float(summary["family_q_off_values"][-1]), 1.0, places=12)

    def test_support_module_exposes_the_fixed_hoel_toy_example(self) -> None:
        builder = getattr(support, "build_hoel_fig2_toy_system_row", None)

        self.assertIsNotNone(builder)
        if builder is None:
            return

        hoel_row = builder()
        self.assertEqual(hoel_row["name"], "hoel_fig2_toy_example")
        self.assertEqual(int(hoel_row["candidate_count"]), 147)
        self.assertAlmostEqual(float(hoel_row["micro_ei"]), 1.1485793804973707, places=12)
        self.assertAlmostEqual(float(hoel_row["best_candidate"]["macro_ei"]), 1.5518285258976832, places=12)
        self.assertEqual(
            tuple(tuple(int(index) for index in block) for block in hoel_row["planted_groups"]),
            ((0, 1), (2, 3)),
        )
        self.assertEqual(
            tuple(tuple(int(bit) for bit in mapping) for mapping in hoel_row["planted_mappings"]),
            ((0, 0, 0, 1), (0, 0, 0, 1)),
        )

    def test_notebook_exposes_single_example_and_family_figures(self) -> None:
        hoel_example_svg = self.namespace["hoel_example_scatter_svg"]
        family_distribution_svg = self.namespace["family_distribution_svg"]
        summary_html = self.namespace["summary_html"]
        hoel_example_html = self.namespace["hoel_example_html"]
        family_stats_html = self.namespace["family_stats_html"]

        self.assertIn("Hoel Figure 2 micro-mechanism family", summary_html)
        self.assertIn("Hoel Figure 2 toy example", hoel_example_html)
        self.assertIn("mean rho", family_stats_html)
        self.assertNotIn("title", hoel_example_svg.lower())
        self.assertNotIn("legend", hoel_example_svg.lower())
        self.assertIn(">EI(Z-&gt;X+)</text>", hoel_example_svg)
        self.assertIn("neg. loss", hoel_example_svg)
        self.assertNotIn("title", family_distribution_svg.lower())
        self.assertNotIn("legend", family_distribution_svg.lower())
        self.assertIn("neg. loss", family_distribution_svg)
        self.assertIn("neg. syn", family_distribution_svg)
        self.assertIn("Syn macro", family_distribution_svg)
        self.assertIn("Spearman rho", family_distribution_svg)

    def test_family_grid_tracks_per_system_rho_distributions(self) -> None:
        family_rows = self.namespace["FAMILY_SYSTEM_ROWS"]
        family_rho_rows = self.namespace["FAMILY_RHO_ROWS"]
        family_term_stats = self.namespace["FAMILY_TERM_STATS"]

        self.assertEqual(len(family_rows), 21)
        self.assertEqual(len(family_rho_rows), 21)
        self.assertEqual(
            sorted({round(float(row["q_off"]), 6) for row in family_rows}),
            [round(index / 20.0, 6) for index in range(21)],
        )
        for q_off in [index / 20.0 for index in range(21)]:
            slice_rows = [
                row
                for row in family_rows
                if abs(float(row["q_off"]) - q_off) < 1e-12
            ]
            self.assertEqual(len(slice_rows), 1)
        self.assertEqual(
            sorted({round(float(row["q_off"]), 6) for row in family_rho_rows}),
            [round(index / 20.0, 6) for index in range(21)],
        )
        self.assertTrue(all(int(row["candidate_count"]) == 147 for row in family_rho_rows))
        self.assertTrue(all(-1.0 <= float(row["neg_syn_micro"]) <= 1.0 for row in family_rho_rows))
        self.assertTrue(all(-1.0 <= float(row["neg_loss_sum"]) <= 1.0 for row in family_rho_rows))
        self.assertTrue(all(-1.0 <= float(row["syn_macro"]) <= 1.0 for row in family_rho_rows))
        self.assertEqual(len({row["system_id"] for row in family_rho_rows}), 21)
        self.assertGreater(
            float(family_term_stats["neg_loss_sum"]["mean_rho"]),
            float(family_term_stats["syn_macro"]["mean_rho"]),
        )
        self.assertGreater(
            float(family_term_stats["syn_macro"]["mean_rho"]),
            float(family_term_stats["neg_syn_micro"]["mean_rho"]),
        )
        self.assertGreater(float(family_term_stats["neg_loss_sum"]["std"]), 0.1)
        self.assertGreater(float(family_term_stats["syn_macro"]["std"]), 0.05)
        self.assertLess(float(family_term_stats["neg_syn_micro"]["max"]), 0.05)
        self.assertLess(float(family_term_stats["neg_syn_micro"]["min"]), -0.05)

    def test_notebook_writes_cache_artifacts_for_reuse(self) -> None:
        cache_file = (
            self.repo_root
            / "exp"
            / "cache"
            / "rq3_boolean_causal_emergence"
            / "hoel_micro_mechanism_family_summary.json"
        )
        family_csv_file = (
            self.repo_root
            / "exp"
            / "cache"
            / "rq3_boolean_causal_emergence"
            / "family_system_metrics.csv"
        )
        family_rho_csv_file = (
            self.repo_root
            / "exp"
            / "cache"
            / "rq3_boolean_causal_emergence"
            / "family_system_rho_metrics.csv"
        )
        hoel_candidate_csv_file = (
            self.repo_root
            / "exp"
            / "cache"
            / "rq3_boolean_causal_emergence"
            / "hoel_candidate_metric_rows.csv"
        )

        self.assertTrue(cache_file.exists())
        self.assertTrue(family_csv_file.exists())
        self.assertTrue(family_rho_csv_file.exists())
        self.assertTrue(hoel_candidate_csv_file.exists())

        summary_payload = json.loads(cache_file.read_text())
        self.assertEqual(summary_payload["summary"]["n_family_systems"], 21)
        self.assertEqual(summary_payload["summary"]["n_family_rho_points"], 21)
        self.assertEqual(len(summary_payload["family_rows"]), 21)
        self.assertEqual(len(summary_payload["family_rho_rows"]), 21)

        with family_csv_file.open(newline="", encoding="utf-8") as handle:
            family_rows = list(csv.DictReader(handle))
        with family_rho_csv_file.open(newline="", encoding="utf-8") as handle:
            family_rho_rows = list(csv.DictReader(handle))
        with hoel_candidate_csv_file.open(newline="", encoding="utf-8") as handle:
            candidate_rows = list(csv.DictReader(handle))

        self.assertEqual(len(family_rows), 21)
        self.assertEqual(len(family_rho_rows), 21)
        self.assertEqual(len(candidate_rows), 147)


if __name__ == "__main__":
    unittest.main()
