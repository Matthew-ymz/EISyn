from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_unicm_shap_surrogate import (
    MODE_ORDER,
    aggregate_feature_interactions_to_pairs,
    aggregate_feature_shap_to_modes,
    build_peid_source_ei_lead_summary,
    compare_pair_interactions_with_peid,
    compare_shap_modes_with_peid,
    exact_group_shapley_interactions,
    feature_names,
    load_peid_summaries,
    summarize_pair_interaction_leads,
    summarize_pair_interactions,
)


def test_feature_names_cover_full_history_modes() -> None:
    names = feature_names()

    assert len(names) == 12 * len(MODE_ORDER)
    assert names[0] == "t-12:nino"
    assert names[-1] == "t-1:WWV"


def test_aggregate_feature_shap_to_modes_ranks_strong_mode_first() -> None:
    shap_values = np.zeros((3, 12 * len(MODE_ORDER)), dtype=float)
    nino_index = MODE_ORDER.index("nino")
    iod_index = MODE_ORDER.index("IOD")
    for month in range(12):
        shap_values[:, month * len(MODE_ORDER) + nino_index] = [1.0, -2.0, 3.0]
        shap_values[:, month * len(MODE_ORDER) + iod_index] = [0.2, -0.1, 0.3]

    rows = aggregate_feature_shap_to_modes(
        shap_values,
        target="nino",
        seed=1,
        lead=1,
        surrogate_r2=0.99,
    )

    assert len(rows) == len(MODE_ORDER)
    assert rows.iloc[0]["mode"] == "nino"
    assert rows.iloc[0]["rank_shap"] == 1
    assert rows.iloc[0]["mean_abs_shap"] > rows[rows["mode"] == "IOD"].iloc[0]["mean_abs_shap"]


def test_aggregate_feature_interactions_to_pairs_returns_55_off_diagonal_pairs() -> None:
    interaction_values = np.zeros((2, 12 * len(MODE_ORDER), 12 * len(MODE_ORDER)), dtype=float)
    nino_index = MODE_ORDER.index("nino")
    iod_index = MODE_ORDER.index("IOD")
    siod_index = MODE_ORDER.index("SIOD")
    for left_month in range(12):
        for right_month in range(12):
            left = left_month * len(MODE_ORDER) + iod_index
            right = right_month * len(MODE_ORDER) + siod_index
            interaction_values[:, left, right] = 2.0
            interaction_values[:, right, left] = 2.0
            weak = right_month * len(MODE_ORDER) + nino_index
            interaction_values[:, left, weak] = 0.1
            interaction_values[:, weak, left] = 0.1

    rows = aggregate_feature_interactions_to_pairs(
        interaction_values,
        target="IOD",
        seed=1,
        lead=1,
        surrogate_r2=0.98,
    )

    assert len(rows[~rows["is_diagonal"]]) == 55
    top_pairs = summarize_pair_interactions(rows, top_k=3)
    assert top_pairs.iloc[0]["pair"] == "IOD|SIOD"
    assert top_pairs.iloc[0]["rank_shap_interaction"] == 1


def test_exact_group_shapley_interactions_recovers_synthetic_pair() -> None:
    group_count = 4
    samples = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [2.0, -1.0, 0.0, 0.0],
            [-1.0, 2.0, 0.0, 0.0],
        ]
    )
    baseline = np.zeros(group_count, dtype=float)

    def predict(values: np.ndarray) -> np.ndarray:
        return values[:, 0] * values[:, 1] + 0.2 * values[:, 2]

    matrix = exact_group_shapley_interactions(
        samples,
        baseline,
        predict,
        group_names=[f"g{index}" for index in range(group_count)],
    )

    assert matrix.shape == (samples.shape[0], group_count, group_count)
    assert np.mean(np.abs(matrix[:, 0, 1])) > 10 * np.mean(np.abs(matrix[:, 0, 2]))
    assert np.allclose(matrix[:, 0, 1], matrix[:, 1, 0])


def test_peid_comparison_preserves_shap_only_targets_and_warnings(tmp_path: Path) -> None:
    nino_peid = tmp_path / "nino.csv"
    pd.DataFrame(
        [
            {
                "pair": "nino|IOD",
                "left_source": "nino",
                "right_source": "IOD",
                "target": "nino",
                "mean_left_ei": 0.5,
                "mean_right_ei": 0.1,
                "mean_joint_ei": 0.7,
                "mean_syn": 0.1,
                "rank_within_target": 1,
            }
        ]
    ).to_csv(nino_peid, index=False)

    peid, warnings = load_peid_summaries([nino_peid], targets=["nino", "IOD"])

    assert "No PEID pair summary rows found for target=IOD" in warnings

    mode_summary = pd.DataFrame(
        [
            {"target": "nino", "mode": "nino", "mean_abs_shap": 0.8, "rank_shap": 1},
            {"target": "IOD", "mode": "IOD", "mean_abs_shap": 0.9, "rank_shap": 1},
        ]
    )
    mode_comparison = compare_shap_modes_with_peid(mode_summary, peid)
    iod_row = mode_comparison[mode_comparison["target"] == "IOD"].iloc[0]
    assert bool(iod_row["peid_available"]) is False

    pair_summary = pd.DataFrame(
        [
            {"target": "nino", "pair": "nino|IOD", "mean_abs_interaction": 0.3, "rank_shap_interaction": 1},
            {"target": "IOD", "pair": "IOD|SIOD", "mean_abs_interaction": 0.4, "rank_shap_interaction": 1},
        ]
    )
    pair_comparison = compare_pair_interactions_with_peid(pair_summary, peid)
    iod_pair = pair_comparison[pair_comparison["target"] == "IOD"].iloc[0]
    assert bool(iod_pair["peid_available"]) is False


def test_build_peid_source_ei_lead_summary_deduplicates_pair_rows() -> None:
    rows = pd.DataFrame(
        [
            {
                "target": "nino",
                "seed": 1,
                "lead": 1,
                "left_source": "nino",
                "right_source": "IOD",
                "left_ei": 2.0,
                "right_ei": 0.5,
            },
            {
                "target": "nino",
                "seed": 1,
                "lead": 1,
                "left_source": "nino",
                "right_source": "NPMM",
                "left_ei": 2.0,
                "right_ei": 0.2,
            },
            {
                "target": "nino",
                "seed": 2,
                "lead": 1,
                "left_source": "nino",
                "right_source": "IOD",
                "left_ei": 4.0,
                "right_ei": 0.7,
            },
        ]
    )

    summary = build_peid_source_ei_lead_summary(rows)
    nino = summary[(summary["target"] == "nino") & (summary["mode"] == "nino")].iloc[0]

    assert nino["lead"] == 1
    assert nino["mean_peid_source_ei"] == 3.0
    assert np.isclose(nino["std_peid_source_ei"], np.sqrt(2.0))


def test_summarize_pair_interaction_leads_uses_seed_variation() -> None:
    rows = pd.DataFrame(
        [
            {"target": "IOD", "pair": "IOD|SIOD", "left_source": "IOD", "right_source": "SIOD", "lead": 1, "seed": 1, "mean_abs_interaction": 0.2, "is_diagonal": False},
            {"target": "IOD", "pair": "IOD|SIOD", "left_source": "IOD", "right_source": "SIOD", "lead": 1, "seed": 2, "mean_abs_interaction": 0.4, "is_diagonal": False},
            {"target": "IOD", "pair": "IOD|IOD", "left_source": "IOD", "right_source": "IOD", "lead": 1, "seed": 1, "mean_abs_interaction": 9.0, "is_diagonal": True},
        ]
    )

    summary = summarize_pair_interaction_leads(rows)

    assert len(summary) == 1
    assert summary.iloc[0]["pair"] == "IOD|SIOD"
    assert np.isclose(summary.iloc[0]["mean_abs_interaction"], 0.3)
    assert np.isclose(summary.iloc[0]["std_abs_interaction"], np.sqrt(0.02))
