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
    build_periodic_source_groups,
    comparison_ground_truth,
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
    assert Path(result["report_path"]).exists()
    assert "j0_false_positive" in result["diagnostics"]
    with np.load(result["arrays_path"]) as arrays:
        assert arrays["peid_synergy_by_target"].shape == (2, 2)
        assert arrays["shap_interaction_by_target"].shape == (2, 2)
    for row in result["runs"]:
        assert row["oracle_intervention_digest"] == row["mlp_intervention_digest"]
        assert set(row["method_status"]) == set(METHOD_NAMES)
        assert all(row["method_status"].values())
    saved = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    assert saved["protocol"]["mode"] == "smoke"
