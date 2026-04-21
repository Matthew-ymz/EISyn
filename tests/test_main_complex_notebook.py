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


class MainComplexNotebookTests(unittest.TestCase):
    def test_main_complex_notebook_ranks_phi_and_synergy_differently(self) -> None:
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "exp"
            / "main_complex.ipynb"
        )

        namespace = execute_notebook(notebook_path)

        self.assertEqual(tuple(namespace["phi_complexes"][0]["subset_indices"]), (0, 1, 2))
        self.assertEqual(tuple(namespace["phi_ranking"][0]["subset_indices"]), (0, 1, 2))
        self.assertEqual(tuple(namespace["synergy_ranking"][0]["subset_indices"]), (0, 1, 2, 3, 4))
        self.assertEqual(tuple(namespace["synergy_complexes"][0]["subset_indices"]), (0, 1, 2, 3, 4))


if __name__ == "__main__":
    unittest.main()
