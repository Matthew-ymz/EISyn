from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.part1_mmi_pid_synergy_report import compute_mmi_pid_atoms, run_mmi_pid_six_system_report


def test_mmi_pid_atoms_obey_identity() -> None:
    atoms = compute_mmi_pid_atoms(left_mi=0.2, right_mi=0.5, joint_mi=0.9)

    assert atoms["redundancy"] == 0.2
    assert atoms["unique_left"] == 0.0
    assert atoms["unique_right"] == 0.3
    assert atoms["synergy"] == 0.4
    assert np.isclose(
        atoms["joint_mi"],
        atoms["redundancy"] + atoms["unique_left"] + atoms["unique_right"] + atoms["synergy"],
    )


def test_mmi_pid_smoke_report_emits_six_system_contract(tmp_path: Path) -> None:
    result = run_mmi_pid_six_system_report(
        mode="smoke",
        seeds=(0,),
        result_path=tmp_path / "summary.json",
        report_path=tmp_path / "report.md",
        figure_path=tmp_path / "mmi_pid.png",
        parameter_overrides={
            "standard_map": (0.0,),
            "wilson_cowan_refractory": (0.0,),
            "kuramoto": (0.0,),
            "coupled_henon": (0.0,),
            "ikeda_y_tau": (0.0,),
            "nicholson_bailey": (0.0,),
        },
        sample_overrides={
            "training_samples": 36,
            "validation_samples": 24,
            "readout_samples": 24,
            "peid_samples": 24,
            "epochs": 1,
            "hidden_width": 4,
        },
    )

    expected = {
        "standard_map",
        "wilson_cowan_refractory",
        "kuramoto",
        "coupled_henon",
        "ikeda_y_tau",
        "nicholson_bailey",
    }
    assert set(result["systems"]) == expected
    assert Path(result["result_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert Path(result["figure_path"]).exists()

    saved = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert set(saved["systems"]) == expected
    for system, payload in saved["systems"].items():
        assert payload["summary"][0]["n_seeds"] == 1
        assert payload["protocol"]["mmi_pid_definition"] == "MMI"
        assert payload["protocol"]["target_distribution"] == "observed_readout_targets"
        assert payload["audit"]["estimator_matches_part1"]
        assert payload["audit"]["no_mlp_used_for_mmi_pid"]
        row = payload["rows"][0]
        assert {
            "I_left",
            "I_right",
            "I_joint",
            "redundancy",
            "unique_left",
            "unique_right",
            "mmi_pid_synergy",
            "estimator",
            "readout_state_digest",
            "mmi_pid_state_digest",
        } <= set(row)
        assert "mlp_model_digest" not in row
        assert "oracle_mmi_pid_synergy" not in row
        assert row["mmi_pid_target_digest"] == row["observed_target_digest"]
        assert np.isclose(
            row["I_joint"],
            row["redundancy"] + row["unique_left"] + row["unique_right"] + row["mmi_pid_synergy"],
            atol=1e-8,
        ), system

    report_text = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "# MMI-PID Six-System Synergy Report" in report_text
    assert "Coupled standard map" in report_text
    assert "Nicholson-Bailey" in report_text
