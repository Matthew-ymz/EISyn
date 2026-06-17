from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.henon_mmi_pid_vs_mlp_peid import (
    decompose_mi_triplet,
    run_henon_mmi_pid_vs_mlp_peid,
    run_henon_unique_five_method_sweep,
    run_henon_unique_sweep_mmi_vs_mlp_peid,
)


def test_mmi_pid_exceeds_peid_residual_by_weaker_single_source_information() -> None:
    atoms = decompose_mi_triplet(left_mi=0.8, right_mi=0.4, joint_mi=1.0)

    assert np.isclose(atoms["mmi_pid_synergy"], 0.2)
    assert np.isclose(atoms["peid_residual"], -0.2)
    assert np.isclose(atoms["mmi_minus_peid"], 0.4)
    assert np.isclose(atoms["weaker_single_source_mi"], 0.4)


def test_henon_smoke_contrast_writes_report_and_separates_methods(tmp_path: Path) -> None:
    result = run_henon_mmi_pid_vs_mlp_peid(
        mode="smoke",
        seeds=(0,),
        result_path=tmp_path / "summary.json",
        report_path=tmp_path / "report.md",
        figure_path=tmp_path / "contrast.png",
        samples=420,
        epochs=20,
        bins=8,
    )

    assert Path(result["result_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert Path(result["figure_path"]).exists()

    saved = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert saved["system"] == "classic_henon_map"
    assert saved["relation"] == "x+y->x_tau"
    assert saved["summary"]["observed_mmi_pid_synergy_mean"] > saved["summary"]["mlp_peid_residual_mean"]
    assert saved["summary"]["observed_mmi_minus_oracle_peid_mean"] > 0.1
    assert saved["summary"]["observed_weaker_single_source_mi_mean"] > 0.1

    text = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "Hénon" in text
    assert "MMI-PID" in text
    assert "MLP+PEID" in text


def test_henon_unique_sweep_can_raise_mmi_while_lowering_peid(tmp_path: Path) -> None:
    result = run_henon_unique_sweep_mmi_vs_mlp_peid(
        mode="smoke",
        lambdas=(0.0, 0.5, 1.0),
        gamma_range=(0.3, 2.0),
        kappa_range=(0.5, 0.1),
        seeds=(0,),
        result_path=tmp_path / "sweep.json",
        report_path=tmp_path / "sweep.md",
        figure_path=tmp_path / "sweep.png",
        samples=700,
        epochs=25,
        bins=8,
    )

    assert Path(result["result_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert Path(result["figure_path"]).exists()
    assert result["parameter_key"] == "lambda"

    saved = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert len(saved["parameter_summary"]) == 3
    assert saved["parameter_summary"][0]["gamma"] < saved["parameter_summary"][-1]["gamma"]
    assert saved["parameter_summary"][0]["kappa"] > saved["parameter_summary"][-1]["kappa"]
    assert saved["diagnostics"]["observed_mmi_pid_synergy_dynamic_range"] > 0.05
    assert (
        saved["parameter_summary"][-1]["observed_mmi_pid_synergy_mean"]
        > saved["parameter_summary"][0]["observed_mmi_pid_synergy_mean"]
    )
    assert (
        saved["parameter_summary"][-1]["oracle_peid_residual_mean"]
        < saved["parameter_summary"][0]["oracle_peid_residual_mean"]
    )

    for row in saved["rows"]:
        assert np.isclose(
            row["observed_mmi_pid_synergy"] - row["oracle_peid_residual"],
            row["observed_weaker_single_source_mi"],
        )


def test_henon_unique_five_method_sweep_emits_only_five_plotted_summary_methods(tmp_path: Path) -> None:
    result = run_henon_unique_five_method_sweep(
        mode="smoke",
        lambdas=(0.0, 1.0),
        gamma_range=(0.3, 2.0),
        kappa_range=(0.5, 0.1),
        seeds=(0,),
        result_path=tmp_path / "five_method.json",
        figure_path=tmp_path / "five_method.png",
        samples=360,
        epochs=12,
        bins=6,
    )

    assert Path(result["result_path"]).exists()
    assert Path(result["figure_path"]).exists()
    assert result["system"] == "controlled_henon_unique_information_five_method"
    assert result["parameter_key"] == "lambda"

    saved = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert len(saved["summary"]) == 2
    assert saved["summary"][0]["gamma"] < saved["summary"][-1]["gamma"]
    assert saved["summary"][0]["kappa"] > saved["summary"][-1]["kappa"]
    plotted_methods = ("wms", "surd_synergy", "shap_interaction", "peid_synergy", "mmi_pid_synergy")
    forbidden_summary_prefixes = (
        "observed_left_mi",
        "observed_right_mi",
        "observed_joint_mi",
        "observed_weaker_single_source_mi",
    )
    for item in saved["summary"]:
        for method in plotted_methods:
            assert f"{method}_mean" in item
            assert f"{method}_std" in item
        assert sum(key.endswith("_mean") for key in item) == len(plotted_methods)
        assert not any(key.startswith(forbidden_summary_prefixes) for key in item)

    assert "observed_weaker_single_source_mi" in saved["rows"][0]
