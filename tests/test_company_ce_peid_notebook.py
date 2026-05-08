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
    ]
    for name in required:
        assert f"`{name}`" in text
    assert "因果问题与识别边界" in text
    assert "潜在混杂" in text
    assert "不是干预效应估计" in text
    assert "因果证据汇总" in text
    assert "因果图结果分析" in text
    assert "结论" in text


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

    for stem in ["peid_top_pairwise_graph", "peid_top_synergy_hypergraph", "peid_causal_evidence_summary"]:
        assert (figure_dir / f"{stem}.png").exists()
        assert (figure_dir / f"{stem}.pdf").exists()
