import importlib.util
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "company_ce" / "peid_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("peid_sensitivity", MODULE_PATH)
peid_sensitivity = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = peid_sensitivity
SPEC.loader.exec_module(peid_sensitivity)


def test_sensitivity_summary_reports_rank_overlap_and_synergy_stability():
    baseline_pairwise = pd.DataFrame(
        [
            {"source": "a", "target": "x", "ei": 0.30, "p_value": 0.01},
            {"source": "b", "target": "x", "ei": 0.20, "p_value": 0.20},
            {"source": "c", "target": "x", "ei": 0.10, "p_value": 0.30},
        ]
    )
    variant_pairwise = pd.DataFrame(
        [
            {"source": "a", "target": "x", "ei": 0.25, "p_value": 0.02},
            {"source": "b", "target": "x", "ei": 0.21, "p_value": 0.04},
            {"source": "c", "target": "x", "ei": 0.05, "p_value": 0.50},
        ]
    )
    baseline_synergy = pd.DataFrame(
        [
            {"sources": "a+b", "target": "x", "source_order": 2, "synergy_raw": 0.10, "p_value": 1.0},
            {"sources": "a+c", "target": "x", "source_order": 2, "synergy_raw": 0.05, "p_value": 0.03},
        ]
    )
    variant_synergy = pd.DataFrame(
        [
            {"sources": "a+b", "target": "x", "source_order": 2, "synergy_raw": 0.05, "p_value": 0.40},
            {"sources": "a+c", "target": "x", "source_order": 2, "synergy_raw": 0.02, "p_value": 0.80},
        ]
    )

    row = peid_sensitivity.summarize_run_against_baseline(
        run_id="bins_4",
        varied_parameter="bins",
        varied_value="4",
        baseline_pairwise=baseline_pairwise,
        variant_pairwise=variant_pairwise,
        baseline_synergy=baseline_synergy,
        variant_synergy=variant_synergy,
        top_m=2,
        null_reps=100,
    )

    assert row["pairwise_edge_count"] == 3
    assert row["pairwise_spearman"] > 0.9
    assert row["pairwise_top_m_overlap"] == 1.0
    assert row["synergy_edge_count"] == 2
    assert row["synergy_sign_agreement"] == 1.0
    assert row["synergy_sign_flip_count"] == 0
    assert row["p_value_stability_is_formal"] is True
