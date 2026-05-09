import json
import importlib.util
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "exp" / "company_ce" / "peid_causal_hypergraph.ipynb"
MODULE_PATH = REPO_ROOT / "scripts" / "company_ce" / "peid_causal_hypergraph.py"
SPEC = importlib.util.spec_from_file_location("peid_causal_hypergraph", MODULE_PATH)
peid_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = peid_module
SPEC.loader.exec_module(peid_module)
plot_outputs = peid_module.plot_outputs
build_native_edges_for_review = peid_module.build_native_edges_for_review
compute_pairwise_edges = peid_module.compute_pairwise_edges
compute_synergy_hyperedges = peid_module.compute_synergy_hyperedges


def test_peid_notebook_explains_all_adjustable_parameters_in_chinese():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    required = [
        "SMOKE",
        "VARIABLES",
        "BINS",
        "MAX_SOURCE_ORDER",
        "ALPHA",
        "MIN_SOURCE_COUNT",
        "MIN_TOTAL_COUNT",
        "NULL_REPS",
        "TOP_K",
        "RANDOM_SEED",
        "YEAR_START",
        "YEAR_END",
        "WINSOR_LOWER",
        "WINSOR_UPPER",
        "RUN_ANALYSIS",
        "REUSE_CACHE_WHEN_NOT_RUNNING",
        "INPUT_PATH",
        "OUTPUT_DIR",
        "FIGURE_DIR",
        "RUN_SENSITIVITY",
        "SENSITIVITY_BINS_VALUES",
        "SENSITIVITY_MIN_SOURCE_COUNT_VALUES",
        "SENSITIVITY_ALPHA_VALUES",
        "SENSITIVITY_WINSOR_PAIRS",
        "SENSITIVITY_YEAR_WINDOWS",
    ]
    for name in required:
        assert f"`{name}`" in text
    assert "因果问题与识别边界" in text
    assert "潜在混杂" in text
    assert "不是干预效应估计" in text
    assert "因果证据汇总" in text
    assert "因果图结果分析" in text
    assert "结论" in text
    assert "原生结果" in text
    assert "全量边" in text
    assert "synergy_raw" in text
    assert "Top-K 仅为摘要视图" in text


def test_plot_outputs_exports_publication_pairwise_and_synergy_graphs(tmp_path):
    output_dir = tmp_path / "csv"
    figure_dir = tmp_path / "fig"
    output_dir.mkdir()

    pairwise = pd.DataFrame(
        [
            {"source": "at", "target": "revt", "ei": 0.08, "p_value": 0.02, "null_mean": 0.01},
            {"source": "at", "target": "at", "ei": 0.07, "p_value": 0.02, "null_mean": 0.02},
            {"source": "emp", "target": "revt", "ei": 0.06, "p_value": 0.04, "null_mean": 0.01},
            {"source": "revt", "target": "emp", "ei": 0.04, "p_value": 0.05, "null_mean": 0.01},
        ]
    )
    synergy = pd.DataFrame(
        [
            {
                "sources": "dltt+lt",
                "target": "dltt",
                "source_order": 2,
                "joint_ei": 0.08,
                "single_ei_sum": 0.075,
                "synergy_raw": 0.005,
                "synergy": 0.005,
                "p_value": 0.02,
                "null_mean": 0.0001,
            },
            {
                "sources": "at+ch",
                "target": "ch",
                "source_order": 2,
                "joint_ei": 0.04,
                "single_ei_sum": 0.036,
                "synergy_raw": 0.004,
                "synergy": 0.004,
                "p_value": 0.03,
                "null_mean": 0.0001,
            },
        ]
    )
    stability = pd.DataFrame(
        [
            {"source": "at", "target": "revt", "ei": 0.09, "period": "early"},
            {"source": "at", "target": "revt", "ei": 0.07, "period": "late"},
        ]
    )
    pairwise.to_csv(output_dir / "peid_pairwise_edges.csv", index=False)
    synergy.to_csv(output_dir / "peid_synergy_hyperedges.csv", index=False)
    stability.to_csv(output_dir / "peid_period_stability.csv", index=False)

    plot_outputs(output_dir, figure_dir, top_k=4)

    for stem in [
        "peid_top_pairwise_graph",
        "peid_top_synergy_hypergraph",
        "peid_causal_evidence_summary",
        "peid_synergy_raw_heatmap",
        "peid_synergy_raw_ranked",
    ]:
        assert (figure_dir / f"{stem}.png").exists()
        assert (figure_dir / f"{stem}.pdf").exists()


def test_native_review_table_keeps_all_pairwise_and_synergy_edges():
    pairwise = pd.DataFrame(
        [
            {"source": "a", "target": "a", "ei": 0.2, "p_value": 0.10, "null_mean": 0.01, "source_support": 2},
            {"source": "a", "target": "b", "ei": 0.1, "p_value": 0.20, "null_mean": 0.02, "source_support": 2},
        ]
    )
    synergy = pd.DataFrame(
        [
            {
                "sources": "a+b",
                "target": "a",
                "source_order": 2,
                "joint_ei": 0.25,
                "single_ei_sum": 0.15,
                "synergy_raw": 0.10,
                "synergy": 0.10,
                "p_value": 1.0,
                "null_mean": 0.0,
                "source_support": 4,
            }
        ]
    )

    review = build_native_edges_for_review(pairwise, synergy)

    assert len(review) == len(pairwise) + len(synergy)
    assert set(review["edge_type"]) == {"pairwise", "synergy"}
    synergy_edge = review[review["edge_type"] == "synergy"].iloc[0]
    assert synergy_edge["value"] == 0.10
    assert synergy_edge["value_col"] == "synergy_raw"
    assert synergy_edge["rank"] == 1


def test_pairwise_edges_are_full_grid_and_synergy_raw_is_not_truncated():
    states = pd.DataFrame(
        {
            "gvkey": ["1", "2", "3", "4"],
            "mid_year": [2000, 2000, 2000, 2000],
            "a_src": [1, 1, 2, 2],
            "b_src": [1, 2, 1, 2],
            "a_tgt": [1, 1, 2, 2],
            "b_tgt": [1, 1, 2, 2],
        }
    )

    pairwise = compute_pairwise_edges(states, ["a", "b"], bins=2, alpha=0.5, min_source_count=1)
    synergy = compute_synergy_hyperedges(states, ["a", "b"], bins=2, alpha=0.5, min_source_count=1, max_source_order=2)

    assert len(pairwise) == 4
    assert len(synergy) == 2
    assert (synergy["synergy"] >= 0).all()
    assert (synergy["synergy_raw"] >= 0).all()
    assert (synergy["synergy"] == synergy["synergy_raw"]).all()
