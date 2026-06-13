from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_coupled_standard_map_methods import (
    METHOD_NAMES,
    _plot_part1_four_method_synergy,
    build_periodic_source_groups,
    comparison_ground_truth,
    run_part1_four_method_comparison,
    run_natural_peid_experiment,
    run_experiment,
)
from scripts.coupled_standard_map_peid import (
    StandardMapConfig,
    build_trajectory_dataset,
    periodic_features,
)


def test_full_dataset_uses_exact_10_3_3_trajectory_split() -> None:
    dataset = build_trajectory_dataset(
        StandardMapConfig(),
        trajectory_count=16,
        steps_per_trajectory=12,
        seed=7,
    )

    assert len(set(dataset.train.trajectory_ids.tolist())) == 10
    assert len(set(dataset.validation.trajectory_ids.tolist())) == 3
    assert len(set(dataset.test.trajectory_ids.tolist())) == 3
    assert set(dataset.train.trajectory_ids).isdisjoint(dataset.validation.trajectory_ids)
    assert set(dataset.train.trajectory_ids).isdisjoint(dataset.test.trajectory_ids)
    assert set(dataset.validation.trajectory_ids).isdisjoint(dataset.test.trajectory_ids)


def test_periodic_source_groups_cover_each_state_without_overlap() -> None:
    groups = build_periodic_source_groups()
    states = np.asarray([[0.2, -0.4, 1.1, -2.3]], dtype=float)
    encoded = periodic_features(states)

    assert list(groups) == ["q1", "p1", "q2", "p2"]
    assert [groups[name] for name in groups] == [(0, 4), (1, 5), (2, 6), (3, 7)]
    assert sorted(index for group in groups.values() for index in group) == list(range(8))
    np.testing.assert_allclose(periodic_features(states + 2.0 * np.pi), encoded)


def test_ground_truth_is_zero_for_cross_and_interaction_at_zero_coupling() -> None:
    truth = comparison_ground_truth(StandardMapConfig(k=1.5, coupling=0.0, noise_std=0.05))

    assert truth["own"] == 1.5**2 / 2.0
    assert truth["cross"] == 0.0
    assert truth["interaction"] == 0.0
    assert truth["momentum"] == 0.0


def test_part1_four_method_plot_is_created(tmp_path: Path) -> None:
    rows = [
        {
            "coupling": coupling,
            "wms_mean": coupling,
            "wms_std": 0.01,
            "surd_synergy_mean": 0.2,
            "surd_synergy_std": 0.02,
            "shap_interaction_mean": coupling / 2,
            "shap_interaction_std": 0.01,
            "peid_synergy_mean": coupling**2,
            "peid_synergy_std": 0.01,
        }
        for coupling in (0.0, 0.5, 1.0)
    ]
    path = tmp_path / "four_method.png"

    _plot_part1_four_method_synergy(rows, path)

    assert path.exists()


def test_part1_four_method_protocol_records_shared_training_contract(tmp_path: Path) -> None:
    result = run_part1_four_method_comparison(
        cached_arrays_path=tmp_path / "does-not-exist.npz",
        result_path=tmp_path / "part1.json",
        figure_path=tmp_path / "part1.png",
        couplings=(0.0,),
        seeds=(0, 1, 2),
        trajectory_count=6,
        steps_per_trajectory=10,
        epochs=1,
        hidden_width=4,
        intervention_samples=30,
    )

    contract = result["protocol"]["method_data_contract"]
    assert result["protocol"]["training_distribution"] == "broad_intervention_domain_one_step_pool"
    assert result["protocol"]["shared_readout_state_distribution"] == "held_out_broad_intervention_domain_one_step_pool"
    assert result["protocol"]["peid_state_distribution"] == "same_held_out_broad_states_as_wms_surd_shap"
    assert result["protocol"]["oracle_peid_state_distribution"] == "same_held_out_broad_states_as_all_methods"
    assert contract["model_training"] == "one_shared_broad_training_pool"
    assert contract["observational_readout"] == "one_shared_broad_held_out_pool"
    assert contract["model_reuse"] == "same_fitted_mlp_for_shap_and_peid"
    assert contract["peid_interventions"] == "same_broad_held_out_pool"
    assert contract["seed_usage"] == "same_seed_set_for_all_methods_at_each_coupling"
    assert result["protocol"]["seed_usage"]["seed_set"] == [0, 1, 2]
    assert result["summary"][0]["n_seeds"] == 3
    for row in result["runs"]:
        assert row["readout_state_digest"] == row["peid_state_digest"]
        assert row["readout_state_digest"] == row["oracle_peid_state_digest"]
        assert row["shap_mlp_model_digest"] == row["peid_mlp_model_digest"]
        assert {
            "train_state_digest",
            "validation_state_digest",
            "readout_state_digest",
            "readout_target_digest",
        } <= set(row)


def test_smoke_run_emits_all_methods_and_matched_interventions(tmp_path: Path) -> None:
    result = run_experiment(
        mode="smoke",
        couplings=(0.0, 0.6),
        seeds=(0,),
        result_dir=tmp_path / "results",
        figure_dir=tmp_path / "fig",
        report_path=tmp_path / "report.md",
    )

    assert set(result["methods"]) == set(METHOD_NAMES)
    assert len(result["runs"]) == 2
    assert Path(result["summary_path"]).exists()
    assert Path(result["arrays_path"]).exists()
    assert Path(result["figure_path"]).exists()
    assert Path(result["ground_truth_figure_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert "j0_false_positive" in result["diagnostics"]
    with np.load(result["arrays_path"]) as arrays:
        assert arrays["peid_synergy_by_target"].shape == (2, 2)
        assert arrays["shap_interaction_by_target"].shape == (2, 2)
    assert result["protocol"]["peid_state_distribution"] == "natural_test_trajectory"
    assert result["protocol"]["peid_target_distribution"] == "mlp_predicted_impulses"
    for row in result["runs"]:
        assert row["peid"]["state_distribution"] == "natural_test_trajectory"
        assert row["peid"]["target_distribution"] == "mlp_predicted_impulses"
        assert "oracle_peid" not in row
        assert set(row["method_status"]) == set(METHOD_NAMES)
        assert all(row["method_status"].values())
    saved = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    assert saved["protocol"]["mode"] == "smoke"
    report_text = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "I_{1,t}" in report_text
    assert "J^2/2" in report_text
    assert "ground_truth_curve" in report_text
    assert "$$" in report_text
    assert "\\[" not in report_text
    assert "Same-rotor angle EI" in report_text
    assert "Other-rotor angle EI" in report_text
    assert "(K^2+J^2)/2" in report_text


def test_natural_peid_smoke_run_writes_mlp_control(tmp_path: Path) -> None:
    result = run_natural_peid_experiment(
        couplings=(0.0, 0.6),
        seeds=(0,),
        result_path=tmp_path / "natural_peid.json",
        trajectory_count=6,
        steps_per_trajectory=90,
        bins=4,
        permutation_count=1,
    )

    assert result["protocol"]["state_distribution"] == "natural_test_trajectory"
    assert result["trends"].keys() >= {"natural_peid_synergy"}
    assert len(result["summary"]) == 2
    assert Path(result["result_path"]).exists()
    saved = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert saved["protocol"]["target_distribution"] == "mlp_predicted_impulses"
    assert saved["runs"][0]["natural_peid"]["target_distribution"] == "mlp_predicted_impulses"
