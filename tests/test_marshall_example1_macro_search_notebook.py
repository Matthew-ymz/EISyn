import json
import os
import subprocess
import sys
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


class MarshallExample1MacroSearchNotebookTests(unittest.TestCase):
    def test_notebook_confirms_paper_mapping_over_all_pair_partitions(self) -> None:
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "exp"
            / "marshall_example1_macro_search.ipynb"
        )

        namespace = execute_notebook(notebook_path)

        summary = namespace["MARSHALL_NOTEBOOK_RESULTS"]

        self.assertEqual(summary["n_fixed_candidates"], 196)
        self.assertEqual(summary["n_all_candidates"], 588)
        self.assertEqual(summary["overall_best_groups"], ((0, 1), (2, 3)))
        self.assertEqual(summary["overall_best_alpha_mapping"], (0, 0, 0, 1))
        self.assertEqual(summary["overall_best_beta_mapping"], (0, 0, 0, 1))
        self.assertAlmostEqual(summary["overall_best_ei"], 1.2735731818301856, places=12)
        self.assertEqual(summary["best_tie_count"], 4)
        self.assertGreater(
            summary["best_partition_eis"]["{A,B} | {C,D}"],
            summary["best_partition_eis"]["{A,C} | {B,D}"],
        )
        self.assertGreater(
            summary["best_partition_eis"]["{A,B} | {C,D}"],
            summary["best_partition_eis"]["{A,D} | {B,C}"],
        )

        self.assertIn("All computed macro EI values", namespace["distribution_svg"])
        self.assertIn("{A,B} | {C,D}", namespace["distribution_svg"])
        self.assertIn("{A,C} | {B,D}", namespace["distribution_svg"])
        self.assertIn("{A,D} | {B,C}", namespace["distribution_svg"])
        self.assertEqual(len(namespace["all_results"]), 588)
        self.assertEqual(namespace["paper_result"]["groups"], ((0, 1), (2, 3)))

    def test_notebook_also_runs_when_cwd_is_exp_directory(self) -> None:
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "exp"
            / "marshall_example1_macro_search.ipynb"
        )
        notebook = json.loads(notebook_path.read_text())
        setup_source = "".join(notebook["cells"][1]["source"])
        completed = subprocess.run(
            [sys.executable, "-c", setup_source],
            cwd=notebook_path.parent,
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
