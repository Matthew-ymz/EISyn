from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_network_dynamics_benchmark import (
    BENCHMARK_MODEL_NAMES,
    ModelSpec,
    build_model_specs,
    estimate_oracle_peid,
    run_benchmark,
)


def test_model_specs_expose_distinct_source_and_derivative_target_names() -> None:
    specs = build_model_specs()

    assert tuple(specs) == BENCHMARK_MODEL_NAMES
    for spec in specs.values():
        assert isinstance(spec, ModelSpec)
        assert len(spec.state_names) == len(spec.target_names)
        assert all(name.startswith("d") for name in spec.target_names)
        assert spec.intervention_bounds.shape == (len(spec.state_names), 2)


def test_vector_fields_match_the_paper_equations_at_hand_computed_states() -> None:
    specs = build_model_specs()

    kuramoto = specs["kuramoto"]
    state = np.array([[0.2, -0.4, 0.7]])
    expected = np.array(
        [[
            1.0 + 0.2 * np.sin(0.7 - 0.2),
            1.1 + 0.2 * np.sin(0.7 - (-0.4)),
            0.9,
        ]]
    )
    assert np.allclose(kuramoto.vector_field(state), expected)

    rossler = specs["coupled_rossler"]
    state = np.array([[1.0, 2.0, 0.5, -0.5, 0.25, 1.5]])
    x0, y0, z0, x1, y1, z1 = state[0]
    expected = np.array(
        [[
            -y0 - z0 + 0.5 * np.sin(x1 - x0),
            x0 + 0.165 * y0,
            2.0 + z0 * (x0 - 5.5),
            -y1 - z1 + 0.5 * np.sin(x0 - x1),
            x1 + 0.165 * y1,
            2.0 + z1 * (x1 - 5.5),
        ]]
    )
    assert np.allclose(rossler.vector_field(state), expected)

    sis = specs["sis"]
    state = np.array([[0.2, 0.4, 0.6]])
    w, x, y = state[0]
    expected = np.array(
        [[
            -0.8 * w + w * (1.0 - w),
            -1.0 * x + w * (1.0 - x),
            -1.2 * y + w * (1.0 - y),
        ]]
    )
    assert np.allclose(sis.vector_field(state), expected)

    wilson = specs["wilson_cowan"]
    state = np.array([[0.2, 0.4, 0.6]])
    sigmoid = lambda value: 1.0 / (1.0 + np.exp(-5.1 * (value - 1.0)))
    expected = np.array(
        [[
            -0.2 + sigmoid(0.2),
            -0.4 + sigmoid(0.2),
            -0.6 + sigmoid(0.2),
        ]]
    )
    assert np.allclose(wilson.vector_field(state), expected)


def test_simulation_is_finite_and_sis_respects_physical_bounds() -> None:
    specs = build_model_specs()
    for spec in specs.values():
        states, increments = spec.simulate(seed=3, samples=180, noise=0.0)
        assert states.shape == increments.shape == (180, len(spec.state_names))
        assert np.isfinite(states).all()
        assert np.isfinite(increments).all()

    sis_states, _ = specs["sis"].simulate(seed=5, samples=300, noise=0.01)
    assert np.min(sis_states) >= 0.0
    assert np.max(sis_states) <= 1.0


def test_oracle_peid_recovers_declared_nonlinear_state_dependencies() -> None:
    specs = build_model_specs()
    for model_name in ("kuramoto", "coupled_rossler", "sis"):
        spec = specs[model_name]
        graph = estimate_oracle_peid(spec, samples=1400, seed=11, estimator="histogram")
        for source_a, source_b, target in spec.truth_hyperedges:
            target_rows = graph["hyperedges"][graph["hyperedges"]["target"] == target]
            expected_key = "+".join(sorted((source_a, source_b)))
            expected_score = float(
                target_rows.loc[target_rows["sources"] == expected_key, "score"].iloc[0]
            )
            assert expected_score > 0.02
            if target.startswith("dz"):
                assert expected_score == float(target_rows["score"].max())


def test_wilson_cowan_has_zero_structural_interaction_but_nonzero_joint_peid() -> None:
    """Additive equations can still have joint-EI excess under PEID."""
    specs = build_model_specs()
    wilson = specs["wilson_cowan"]
    graph = estimate_oracle_peid(wilson, samples=1400, seed=11, estimator="histogram")
    wy = graph["hyperedges"].set_index(["sources", "target"])
    assert float(wy.loc[("w+y", "dy"), "score"]) > 0.2

    baseline = np.array([[0.2, 0.4, 0.6]])
    both = baseline.copy()
    left = baseline.copy()
    right = baseline.copy()
    both[0, [0, 2]] = [1.4, 1.3]
    left[0, 0] = 1.4
    right[0, 2] = 1.3
    interaction = (
        wilson.vector_field(both)[0, 2]
        - wilson.vector_field(left)[0, 2]
        - wilson.vector_field(right)[0, 2]
        + wilson.vector_field(baseline)[0, 2]
    )
    assert abs(float(interaction)) < 1e-12

    pairwise = graph["pairwise"].set_index(["source", "target"])
    assert float(pairwise.loc[("w", "dx"), "score"]) > 0.02
    assert float(pairwise.loc[("w", "dy"), "score"]) > 0.02


def test_smoke_benchmark_writes_json_png_and_report(tmp_path: Path) -> None:
    result = run_benchmark(
        mode="smoke",
        result_dir=tmp_path / "results",
        figure_dir=tmp_path / "figures",
        report_path=tmp_path / "report.md",
        seeds=(0,),
    )

    summary_path = Path(result["summary_path"])
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(summary["models"]) == set(BENCHMARK_MODEL_NAMES)
    assert Path(result["summary_figure_path"]).suffix == ".png"
    assert Path(result["summary_figure_path"]).exists()
    assert all(Path(path).exists() for path in result["model_figure_paths"].values())
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "经典网络动力学" in report
    assert "附录：原共同驱动 sine 基准" in report
    assert "nan" not in report.lower()
