import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch


def execute_notebook(notebook_path: Path, cwd: Path | None = None) -> dict[str, object]:
    notebook = json.loads(notebook_path.read_text())
    namespace: dict[str, object] = {}
    with patch.dict(os.environ, {"YRD_NOTEBOOK_TEST_MODE": "1"}, clear=False):
        with patch("pathlib.Path.cwd", return_value=(cwd or Path.cwd()).resolve()):
            for index, cell in enumerate(notebook["cells"]):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                exec(compile(source, f"{notebook_path.name}_cell_{index}", "exec"), namespace, namespace)
    return namespace


class YRDSingleStationNotebookTests(unittest.TestCase):
    def test_notebook_runs_and_exposes_cache_first_full_experiment_artifacts(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "yrd_single_station_smoke_conclusion.ipynb"

        namespace = execute_notebook(notebook_path, cwd=project_root / "exp")

        self.assertIn("run_manifest", namespace)
        self.assertIn("metrics_overall_df", namespace)
        self.assertIn("coupling_overview_df", namespace)
        self.assertIn("final_conclusion_text", namespace)
        self.assertIn("cache_dir", namespace)
        self.assertIn("24h", namespace["final_conclusion_text"])


if __name__ == "__main__":
    unittest.main()
