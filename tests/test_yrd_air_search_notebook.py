import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from yrd.air_search_notebook import (
    build_support_cover_profile_table,
    filter_station_edge_frame,
    run_air_tm_notebook_case,
)


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


class AirSearchNotebookHelperTests(unittest.TestCase):
    def test_filter_station_edge_frame_keeps_top_k_by_absolute_strength(self) -> None:
        frame = pd.DataFrame(
            {
                "source_station_id": ["A", "B", "C", "D"],
                "target_station_id": ["X", "X", "X", "X"],
                "mean": [0.10, -0.70, 0.40, 0.90],
            }
        )

        filtered = filter_station_edge_frame(
            frame,
            top_k_edges=2,
            min_abs_strength=0.0,
            include_self_loops=False,
            positive_only=False,
            sort_by_abs=True,
        )

        self.assertEqual(filtered["mean"].tolist(), [0.90, -0.70])

    def test_build_support_cover_profile_table_exposes_variable_level_l_summary(self) -> None:
        table = build_support_cover_profile_table(
            {
                "input_variables": ["O3", "PM2.5"],
                "center_by_variable": {"O3": 0.1, "PM2.5": -0.2},
                "train_min_by_variable": {"O3": -1.0, "PM2.5": -1.5},
                "train_max_by_variable": {"O3": 1.2, "PM2.5": 1.6},
                "cover_radius_by_variable": {"O3": 1.1, "PM2.5": 1.8},
                "box_size_by_variable": {"O3": 2.2, "PM2.5": 3.6},
                "lower_bound_by_variable": {"O3": -0.5},
                "nonnegative_variables": ["O3"],
            }
        )

        self.assertListEqual(
            table.columns.tolist(),
            [
                "variable",
                "center",
                "train_min",
                "train_max",
                "cover_radius",
                "box_size_Lv",
                "lower_bound",
                "nonnegative_clipped",
            ],
        )
        self.assertEqual(table.loc[0, "variable"], "O3")
        self.assertTrue(bool(table.loc[0, "nonnegative_clipped"]))
        self.assertFalse(bool(table.loc[1, "nonnegative_clipped"]))

    def test_run_air_tm_notebook_case_smoke_returns_three_graph_views(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        results = run_air_tm_notebook_case(
            root_dir=project_root,
            city_en="hangzhou",
            horizon=3,
            tm_sample_count=32,
            sampling_seed=0,
            gamma=1.0,
            top_k_edges=8,
            min_abs_strength=0.0,
            show_negative_synergy_edges=True,
            force_retrain=False,
            force_recompute_tm=True,
            use_smoke=True,
        )

        self.assertEqual(results["run_context"]["city_en"], "hangzhou")
        self.assertEqual(results["run_context"]["horizon"], 3)
        self.assertEqual(results["run_context"]["tm_sample_count"], 32)
        self.assertFalse(results["summary_metrics_df"].empty)
        self.assertFalse(results["profile_variable_df"].empty)
        self.assertIn("o3_pairwise", results["graph_paths"])
        self.assertIn("pm25_to_o3_pairwise", results["graph_paths"])
        self.assertIn("o3_pm25_synergy", results["graph_paths"])

    def test_run_air_tm_notebook_case_global_max_box_mode_uses_scalar_l(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        results = run_air_tm_notebook_case(
            root_dir=project_root,
            city_en="hangzhou",
            horizon=3,
            tm_sample_count=32,
            sampling_seed=0,
            gamma=1.0,
            top_k_edges=8,
            min_abs_strength=0.0,
            show_negative_synergy_edges=True,
            force_retrain=False,
            force_recompute_tm=True,
            use_smoke=True,
            box_mode="global_max",
        )

        self.assertEqual(results["run_context"]["box_mode"], "global_max")
        self.assertIn("global_box_size", results["run_context"])
        self.assertEqual(results["run_context"]["run_tag"].endswith("_lmax"), True)
        self.assertEqual(
            set(results["profile"]["box_size_by_variable"].values()),
            {results["run_context"]["global_box_size"]},
        )
        self.assertEqual(
            max(results["profile"]["original_box_size_by_variable"].values()),
            results["run_context"]["global_box_size"],
        )

    def test_run_air_tm_notebook_case_global_max_override_uses_manual_l(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        results = run_air_tm_notebook_case(
            root_dir=project_root,
            city_en="hangzhou",
            horizon=3,
            tm_sample_count=32,
            sampling_seed=0,
            gamma=1.0,
            top_k_edges=8,
            min_abs_strength=0.0,
            show_negative_synergy_edges=True,
            force_retrain=False,
            force_recompute_tm=True,
            use_smoke=True,
            box_mode="global_max",
            global_box_size_override=9.5,
        )

        self.assertEqual(results["run_context"]["box_mode"], "global_max")
        self.assertEqual(results["run_context"]["global_box_size"], 9.5)
        self.assertEqual(results["run_context"]["run_tag"].endswith("_l9p5000"), True)
        self.assertEqual(set(results["profile"]["box_size_by_variable"].values()), {9.5})
        self.assertEqual(results["profile"]["global_box_size"], 9.5)


class AirSearchNotebookTests(unittest.TestCase):
    def test_hangzhou_tm_notebook_keeps_code_cells_short(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook = json.loads((project_root / "exp" / "yrd_hangzhou_tm_graph.ipynb").read_text())

        code_cell_lengths = []
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            non_empty_lines = [line for line in source.splitlines() if line.strip()]
            code_cell_lengths.append(len(non_empty_lines))

        self.assertTrue(code_cell_lengths)
        self.assertLessEqual(max(code_cell_lengths), 35)

    def test_hangzhou_tm_notebook_uses_html_image_display_instead_of_pyplot_show(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook = json.loads((project_root / "exp" / "yrd_hangzhou_tm_graph.ipynb").read_text())
        all_code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertIn("from IPython.display import HTML, display", all_code)
        self.assertIn("display(HTML(", all_code)
        self.assertIn("width:min(1200px,100%)", all_code)
        self.assertIn("display:block;max-width:1280px", all_code)
        self.assertNotIn("plt.show()", all_code)

    def test_hangzhou_tm_notebook_runs_and_exposes_thin_story_variables(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "yrd_hangzhou_tm_graph.ipynb"

        namespace = execute_notebook(notebook_path, cwd=project_root / "exp")

        self.assertIn("EXPERIMENT_CONFIG", namespace)
        self.assertIn("GRAPH_CONFIG", namespace)
        self.assertIn("results", namespace)
        self.assertIn("summary_metrics_df", namespace)
        self.assertIn("profile_variable_df", namespace)
        self.assertIn("o3_pairwise_display_df", namespace)
        self.assertIn("pm25_to_o3_display_df", namespace)
        self.assertIn("synergy_display_df", namespace)
        self.assertIn("graph_paths", namespace)
        self.assertIn("final_conclusion_text", namespace)
        self.assertIn("GLOBAL_MAX_EXPERIMENT_CONFIG", namespace)
        self.assertIn("global_max_results", namespace)
        self.assertIn("global_max_profile_variable_df", namespace)
        self.assertIn("global_max_graph_paths", namespace)
        self.assertIn("global_max_final_conclusion_text", namespace)
        self.assertIn("global_max_l", namespace)

        self.assertEqual(namespace["EXPERIMENT_CONFIG"]["city_en"], "hangzhou")
        self.assertTrue(namespace["EXPERIMENT_CONFIG"]["use_cached_results"])
        self.assertIn("support-cover", namespace["final_conclusion_text"])
        self.assertIn("PM2.5 -> O3", namespace["final_conclusion_text"])
        self.assertEqual(namespace["GLOBAL_MAX_EXPERIMENT_CONFIG"]["box_mode"], "global_max")
        self.assertIn("global_box_size_override", namespace["GLOBAL_MAX_EXPERIMENT_CONFIG"])
        self.assertIn("scalar L=max_v L_v", namespace["global_max_final_conclusion_text"])

        graph_paths = namespace["graph_paths"]
        self.assertTrue(Path(graph_paths["o3_pairwise"]).exists())
        self.assertTrue(Path(graph_paths["pm25_to_o3_pairwise"]).exists())
        self.assertTrue(Path(graph_paths["o3_pm25_synergy"]).exists())
        global_max_graph_paths = namespace["global_max_graph_paths"]
        self.assertTrue(Path(global_max_graph_paths["o3_pairwise"]).exists())
        self.assertTrue(Path(global_max_graph_paths["pm25_to_o3_pairwise"]).exists())
        self.assertTrue(Path(global_max_graph_paths["o3_pm25_synergy"]).exists())


if __name__ == "__main__":
    unittest.main()
