import json
import math
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from yrd.shanghai_notebook import (
    build_self_loop_node_strengths,
    build_global_edge_ranking,
    build_transport_map_global_causal_summary,
    build_causal_summary_records,
    build_pairwise_edge_render_frame,
    causal_graph_cache_is_valid,
    select_display_rows,
    select_top_fraction_edges,
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


class YRDShanghaiNotebookTests(unittest.TestCase):
    def test_build_transport_map_global_causal_summary_returns_tm_edges(self) -> None:
        source_samples = [
            [[0.0, 0.0], [0.2, 0.1]],
            [[0.5, 0.1], [0.3, 0.2]],
            [[1.0, 0.2], [0.4, 0.3]],
            [[1.5, 0.3], [0.5, 0.4]],
            [[2.0, 0.4], [0.6, 0.5]],
            [[2.5, 0.5], [0.7, 0.6]],
        ]
        predicted_next_o3 = [
            [0.8 * row[0][0] - 0.2 * row[0][1] + 0.1 * row[1][1], 0.5 * row[1][0] + 0.2 * row[0][0]]
            for row in source_samples
        ]

        summary = build_transport_map_global_causal_summary(
            source_samples=source_samples,
            predicted_next_o3=predicted_next_o3,
            station_ids=["A", "B"],
            o3_feature_index=0,
            pm25_feature_index=1,
        )

        self.assertEqual(summary["coupling_method"], "tm")
        self.assertIn("1h", summary)
        self.assertTrue(summary["1h"]["pairwise_edges"])
        self.assertTrue(summary["1h"]["conditional_synergy_edges"])
        self.assertTrue(summary["1h"]["conditional_synergy_ratio_edges"])
        edge = next(
            row for row in summary["1h"]["pairwise_edges"]
            if row["source_station_id"] == "A" and row["target_station_id"] == "A"
        )
        self.assertTrue(math.isfinite(edge["mean"]))

    def test_build_causal_summary_records_keep_synthetic_sample_identity_without_empirical_time(self) -> None:
        summary_records = build_causal_summary_records(
            raw_records=[
                {
                    "sample_id": 3,
                    "sampling_mode": "uniform_box_centered_at_train_mean",
                    "target_station_id": "A",
                    "target_indices": [0],
                    "jacobian": [[1.0, 0.2, 0.1, 0.3]],
                    "sigma_eps": [[0.5]],
                }
            ],
            station_source_groups={"A": [0, 1], "B": [2, 3]},
            station_pollutant_feature_groups={
                "A": {"O3": [0], "PM2.5": [1]},
                "B": {"O3": [2], "PM2.5": [3]},
            },
            box_size=1.0,
            causal_graph_variable="O3",
        )

        self.assertEqual(summary_records[0]["sample_id"], 3)
        self.assertEqual(summary_records[0]["sampling_mode"], "uniform_box_centered_at_train_mean")
        self.assertNotIn("sample_index", summary_records[0])
        self.assertNotIn("time", summary_records[0])

    def test_causal_graph_cache_requires_uniform_box_sampling_metadata(self) -> None:
        self.assertFalse(
            causal_graph_cache_is_valid(
                cache_meta={
                    "requested_causal_graph_box_size_by_variable": {"O3": 6.43},
                    "causal_graph_box_size_by_variable": {"O3": 6.43},
                    "causal_graph_nonnegative_variables": ["O3"],
                },
                requested_box_size_by_variable={"O3": 6.43},
                nonnegative_variables=("O3",),
                coupling_sample_count=4,
                sampling_mode="uniform_box_centered_at_train_mean",
                sampling_seed=0,
            )
        )

    def test_select_top_fraction_edges_keeps_top_half_by_weight(self) -> None:
        pairwise_edges = pd.DataFrame(
            {
                "source_station_id": ["A"] * 5,
                "target_station_id": ["B", "C", "D", "E", "F"],
                "mean": [0.1, 0.7, 0.4, 0.9, 0.2],
            }
        )

        selected_edges = select_top_fraction_edges(pairwise_edges, strength_col="mean", keep_fraction=0.5)

        self.assertEqual(len(selected_edges), 3)
        self.assertListEqual(selected_edges["mean"].tolist(), [0.9, 0.7, 0.4])

    def test_pairwise_edge_render_frame_emphasizes_strong_edges(self) -> None:
        pairwise_edges = pd.DataFrame(
            {
                "source_station_id": ["A", "A"],
                "target_station_id": ["B", "C"],
                "mean": [0.2, 1.0],
            }
        )

        render_edges = build_pairwise_edge_render_frame(pairwise_edges, strength_col="mean")
        weak_edge = render_edges.loc[render_edges["target_station_id"] == "B"].iloc[0]
        strong_edge = render_edges.loc[render_edges["target_station_id"] == "C"].iloc[0]

        self.assertLess(weak_edge["linewidth"], 0.7)
        self.assertGreater(strong_edge["linewidth"] - weak_edge["linewidth"], 4.5)
        self.assertLess(weak_edge["alpha"], 0.2)
        self.assertGreater(strong_edge["alpha"] - weak_edge["alpha"], 0.7)

    def test_select_display_rows_excludes_self_loops_when_requested(self) -> None:
        edges = pd.DataFrame(
            {
                "source_station_id": ["A", "B", "A"],
                "target_station_id": ["A", "A", "B"],
                "mean": [0.6, 0.4, -0.1],
                "std": [0.0, 0.0, 0.0],
                "median": [0.6, 0.4, -0.1],
            }
        )

        selected = select_display_rows(
            edges,
            per_target=2,
            source_col="source_station_id",
            target_col="target_station_id",
            ranking_col="mean",
            positive_only=True,
            include_self_loops=False,
        )

        self.assertFalse((selected["source_station_id"] == selected["target_station_id"]).any())
        self.assertFalse((selected["mean"] < 0).any())

    def test_build_global_edge_ranking_derives_absolute_strength_from_mean(self) -> None:
        edges = pd.DataFrame(
            {
                "source_station_id": ["A", "B"],
                "target_station_id": ["A", "A"],
                "mean": [-0.6, 0.2],
                "std": [0.0, 0.0],
                "median": [-0.6, 0.2],
            }
        )

        ranked = build_global_edge_ranking(edges, sort_col="abs_mean")

        self.assertListEqual(ranked["edge_label"].tolist(), ["A -> A", "B -> A"])
        self.assertAlmostEqual(ranked["cumulative_abs_mean"].iloc[-1], 0.8)

    def test_build_self_loop_node_strengths_uses_diagonal_edge_means(self) -> None:
        edges = pd.DataFrame(
            {
                "source_station_id": ["A", "A", "B", "C"],
                "target_station_id": ["A", "B", "B", "A"],
                "mean": [0.7, 0.2, -0.3, 0.1],
            }
        )

        strengths = build_self_loop_node_strengths(edges, station_ids=["A", "B", "C"])

        self.assertEqual(strengths["A"], 0.7)
        self.assertEqual(strengths["B"], -0.3)
        self.assertEqual(strengths["C"], 0.0)

    def test_notebook_keeps_code_cells_compact_for_reader_scanability(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook = json.loads((project_root / "exp" / "yrd_shanghai.ipynb").read_text())

        code_cell_lengths = []
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            non_empty_lines = [line for line in source.splitlines() if line.strip()]
            code_cell_lengths.append(len(non_empty_lines))

        self.assertTrue(code_cell_lengths)
        self.assertLessEqual(
            max(code_cell_lengths),
            35,
            "Shanghai notebook should keep code cells short and move bulky logic into helpers.",
        )

    def test_notebook_runs_as_one_step_hourly_experiment(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "yrd_shanghai.ipynb"

        namespace = execute_notebook(notebook_path, cwd=project_root / "exp")

        self.assertIn("CAUSAL_GRAPH_L", namespace)
        self.assertIn("CAUSAL_GRAPH_EDGE_KEEP_FRACTION", namespace)
        self.assertIn("CAUSAL_GRAPH_ARROW_MUTATION_SCALE", namespace)
        self.assertIn("CAUSAL_GRAPH_ARROW_SHRINK_TARGET", namespace)
        self.assertIn("run_context", namespace)
        self.assertIn("metrics_overall_df", namespace)
        self.assertIn("pairwise_display_1h", namespace)
        self.assertIn("conditional_synergy_edges_display_1h", namespace)
        self.assertIn("conditional_synergy_edges_global_ranked_1h", namespace)
        self.assertIn("conditional_synergy_ratio_edges_display_1h", namespace)
        self.assertIn("conditional_synergy_ratio_edges_global_ranked_1h", namespace)
        self.assertIn("final_conclusion_text", namespace)

        run_context = namespace["run_context"]
        self.assertEqual(run_context["sample_mode"], "one_step")
        self.assertEqual(run_context["horizons"], [1])
        self.assertEqual(
            run_context["effective_input_dim"],
            run_context["input_shape"][0] * run_context["input_shape"][1],
        )
        self.assertEqual(run_context["full_effective_input_dim"], 190)
        self.assertEqual(run_context["data_resolution_hours"], 1)
        self.assertIn("causal_graph_box_size", run_context)
        self.assertGreater(run_context["causal_graph_box_size"], 0.0)
        self.assertEqual(run_context["causal_graph_sampling_mode"], "uniform_box_centered_at_train_mean")
        self.assertEqual(run_context["causal_graph_center_source"], "train_input_mean")
        self.assertIn("causal_graph_sampling_seed", run_context)
        self.assertEqual(run_context["causal_graph_edge_keep_fraction"], namespace["CAUSAL_GRAPH_EDGE_KEEP_FRACTION"])
        self.assertEqual(
            run_context["causal_graph_arrow_mutation_scale"],
            namespace["CAUSAL_GRAPH_ARROW_MUTATION_SCALE"],
        )
        self.assertEqual(
            run_context["causal_graph_arrow_shrink_target"],
            namespace["CAUSAL_GRAPH_ARROW_SHRINK_TARGET"],
        )

        final_conclusion_text = namespace["final_conclusion_text"]
        self.assertIn("190", final_conclusion_text)
        self.assertIn("1h", final_conclusion_text)
        self.assertIn("O3 + PM2.5", final_conclusion_text)
        self.assertNotIn("24h", final_conclusion_text)

        self.assertNotIn("global_coupling_samples_df", namespace)
        pairwise_display_1h = namespace["pairwise_display_1h"]
        self.assertTrue((pairwise_display_1h["mean"] > 0).all())
        self.assertFalse(
            (pairwise_display_1h["source_station_id"] == pairwise_display_1h["target_station_id"]).any()
        )
        self.assertEqual(
            len(pairwise_display_1h),
            run_context["station_count"] * (run_context["station_count"] - 1),
        )
        pairwise_render_edges_1h = namespace["pairwise_render_edges_1h"]
        expected_render_count = max(
            1,
            int(math.ceil(len(pairwise_display_1h) * float(namespace["CAUSAL_GRAPH_EDGE_KEEP_FRACTION"]))),
        )
        self.assertEqual(
            len(pairwise_render_edges_1h),
            expected_render_count,
        )
        pairwise_degree_summary_1h = namespace["pairwise_degree_summary_1h"]
        self.assertEqual(int(pairwise_degree_summary_1h["in_degree"].sum()), len(pairwise_display_1h))
        self.assertEqual(int(pairwise_degree_summary_1h["out_degree"].sum()), len(pairwise_display_1h))

        artifact_paths = namespace["artifact_paths"]
        self.assertTrue(Path(artifact_paths["global_graph_1h"]).exists())
        self.assertTrue(Path(artifact_paths["conditional_synergy_graph_1h"]).exists())
        self.assertTrue(Path(artifact_paths["conditional_synergy_ratio_graph_1h"]).exists())

    def test_tm_notebook_runs_and_exports_tm_graphs(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "yrd_shanghai_tm_graph.ipynb"

        namespace = execute_notebook(notebook_path, cwd=project_root / "exp")

        self.assertEqual(namespace["CAUSAL_GRAPH_METHOD"], "tm")
        self.assertIn("CAUSAL_GRAPH_L_BY_VARIABLE_STD_Q99", namespace)
        self.assertIn("CAUSAL_GRAPH_NONNEGATIVE_VARIABLES", namespace)
        self.assertIn("pairwise_display_1h", namespace)
        self.assertIn("pairwise_edges_1h", namespace)
        self.assertIn("conditional_synergy_edges_display_1h", namespace)
        self.assertIn("conditional_synergy_edges_df", namespace)
        self.assertIn("negative_pairwise_edge_count_1h", namespace)
        self.assertIn("negative_conditional_synergy_edge_count_1h", namespace)
        self.assertIn("negative_pairwise_edge_distribution_1h", namespace)
        self.assertIn("negative_conditional_synergy_edge_distribution_1h", namespace)
        self.assertIn("negative_edge_summary_1h", namespace)
        self.assertIn("final_conclusion_text", namespace)
        self.assertIn("transport map", namespace["final_conclusion_text"])
        self.assertIn("负值边", namespace["final_conclusion_text"])
        self.assertIn("绝对值最大", namespace["final_conclusion_text"])
        self.assertEqual(
            namespace["negative_pairwise_edge_count_1h"],
            sum(float(row["mean"]) < 0 for row in namespace["pairwise_edges_1h"]),
        )
        self.assertEqual(
            namespace["negative_conditional_synergy_edge_count_1h"],
            int((namespace["conditional_synergy_edges_df"]["mean"] < 0).sum()),
        )
        self.assertEqual(
            namespace["negative_pairwise_edge_distribution_1h"]["count"],
            namespace["negative_pairwise_edge_count_1h"],
        )
        self.assertEqual(
            namespace["negative_conditional_synergy_edge_distribution_1h"]["count"],
            namespace["negative_conditional_synergy_edge_count_1h"],
        )
        self.assertIn("pairwise", namespace["negative_edge_summary_1h"])
        self.assertIn("conditional_synergy", namespace["negative_edge_summary_1h"])
        self.assertEqual(
            namespace["run_context"]["causal_graph_box_size_by_variable"],
            namespace["CAUSAL_GRAPH_L_BY_VARIABLE_STD_Q99"],
        )
        self.assertEqual(
            namespace["run_context"]["causal_graph_nonnegative_variables"],
            list(namespace["CAUSAL_GRAPH_NONNEGATIVE_VARIABLES"]),
        )
        self.assertIn("per-variable", namespace["run_context"]["causal_graph_box_label"])

        artifact_paths = namespace["artifact_paths"]
        self.assertTrue(Path(artifact_paths["global_graph_1h"]).exists())
        self.assertTrue(Path(artifact_paths["conditional_synergy_graph_1h"]).exists())

    def test_tm_notebook_exposes_sample_count_and_documents_independent_interventions(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook = json.loads((project_root / "exp" / "yrd_shanghai_tm_graph.ipynb").read_text())

        parameter_cell = "".join(notebook["cells"][2]["source"])
        math_cell = "".join(notebook["cells"][5]["source"])

        self.assertIn("CAUSAL_GRAPH_COUPLING_SAMPLE_COUNT =", parameter_cell)
        self.assertIn("CAUSAL_GRAPH_L_BY_VARIABLE_STD_Q99 = {", parameter_cell)
        self.assertIn("CAUSAL_GRAPH_NONNEGATIVE_VARIABLES =", parameter_cell)
        self.assertIn(
            "EFFECTIVE_CAUSAL_GRAPH_COUPLING_SAMPLE_COUNT = 4 if NOTEBOOK_TEST_MODE else CAUSAL_GRAPH_COUPLING_SAMPLE_COUNT",
            parameter_cell,
        )
        self.assertIn(
            "coupling_sample_count=EFFECTIVE_CAUSAL_GRAPH_COUPLING_SAMPLE_COUNT",
            parameter_cell,
        )
        self.assertIn("causal_graph_box_size_by_variable=CAUSAL_GRAPH_L_BY_VARIABLE_STD_Q99", parameter_cell)
        self.assertIn("causal_graph_nonnegative_variables=CAUSAL_GRAPH_NONNEGATIVE_VARIABLES", parameter_cell)
        self.assertIn("源端独立干预", math_cell)
        self.assertIn("有限样本估计", math_cell)


if __name__ == "__main__":
    unittest.main()
