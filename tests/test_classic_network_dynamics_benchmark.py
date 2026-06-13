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
    build_coupled_rossler_spec,
    build_kuramoto_coupling_spec,
    build_lorenz_rho_spec,
    build_sis_gate_spec,
    build_wilson_cowan_gain_spec,
    build_future_state_spec,
    build_model_specs,
    estimate_peid_from_samples,
    estimate_oracle_peid,
    run_benchmark,
    run_coupled_rossler_coupling_sweep,
    run_kuramoto_coupling_sweep,
    run_kuramoto_peid_detail_sweep,
    run_lorenz_rho_sweep,
    run_lorenz_uniform_tau_sweep,
    run_ode_future_state_sweeps,
    run_part1_combined_synergy_figure,
    _plot_panel,
    _zero_control_synergy_readouts,
    run_sis_gate_sweep,
    run_wilson_cowan_gain_sweep,
    simulate_natural_trajectory_pool,
    simulate_finite_time_next_states,
    simulate_sis_gate_next_states,
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


def test_histogram_peid_treats_near_constant_targets_as_degenerate() -> None:
    spec = build_kuramoto_coupling_spec(0.0)
    rng = np.random.default_rng(7)
    states = rng.uniform(-np.pi, np.pi, size=(700, 3))
    targets = np.column_stack(
        [
            np.full(len(states), 1.0) + rng.normal(0.0, 1e-8, len(states)),
            np.full(len(states), 1.1) + rng.normal(0.0, 1e-8, len(states)),
            np.full(len(states), 0.9) + rng.normal(0.0, 1e-8, len(states)),
        ]
    )

    graph = estimate_peid_from_samples(spec, states, targets, estimator="histogram")

    assert float(graph["pairwise"]["score"].abs().max()) == 0.0
    assert float(graph["hyperedges"]["score"].abs().max()) == 0.0


def test_sis_gate_parameter_controls_state_dependent_infection() -> None:
    state = np.array([[0.4, 0.3, 0.5]])
    inactive = build_sis_gate_spec(0.0)
    active = build_sis_gate_spec(1.0)

    inactive_field = inactive.vector_field(state)[0]
    active_field = active.vector_field(state)[0]
    assert np.isclose(inactive_field[1], -0.3)
    assert np.isclose(active_field[1], -0.3 + 0.4 * (1.0 - 0.3))
    assert active_field[1] > inactive_field[1]

    next_states = simulate_sis_gate_next_states(active, state, tau=1.0, process_noise=0.0, seed=7)
    assert next_states.shape == state.shape
    assert np.min(next_states) >= 0.0
    assert np.max(next_states) <= 1.0


def test_kuramoto_coupling_parameter_controls_phase_gate() -> None:
    state = np.array([[0.2, -0.4, 0.7]])
    inactive = build_kuramoto_coupling_spec(0.0)
    active = build_kuramoto_coupling_spec(0.2)

    inactive_field = inactive.vector_field(state)[0]
    active_field = active.vector_field(state)[0]
    assert np.allclose(inactive_field, np.array([1.0, 1.1, 0.9]))
    assert np.isclose(active_field[0], 1.0 + 0.2 * np.sin(0.7 - 0.2))
    assert np.isclose(active_field[1], 1.1 + 0.2 * np.sin(0.7 - (-0.4)))
    assert active_field[0] != inactive_field[0]
    assert active_field[1] != inactive_field[1]


def test_parameterized_rossler_and_wilson_cowan_specs_control_their_sweep_terms() -> None:
    rossler_state = np.array([[1.0, 2.0, 0.5, -0.5, 0.25, 1.5]])
    uncoupled = build_coupled_rossler_spec(0.0)
    coupled = build_coupled_rossler_spec(0.75)
    uncoupled_field = uncoupled.vector_field(rossler_state)[0]
    coupled_field = coupled.vector_field(rossler_state)[0]
    expected_delta = 0.75 * np.sin(rossler_state[0, 3] - rossler_state[0, 0])
    assert np.isclose(coupled_field[0] - uncoupled_field[0], expected_delta)
    assert np.isclose(coupled_field[3] - uncoupled_field[3], -expected_delta)

    wilson_state = np.array([[0.2, 0.4, 0.6]])
    low_gain = build_wilson_cowan_gain_spec(1.0)
    high_gain = build_wilson_cowan_gain_spec(7.5)
    low_field = low_gain.vector_field(wilson_state)[0]
    high_field = high_gain.vector_field(wilson_state)[0]
    assert high_field[1] < low_field[1]
    assert high_field[2] < low_field[2]


def test_future_state_spec_retargets_derivative_relations_to_state_targets() -> None:
    kuramoto = build_future_state_spec(build_kuramoto_coupling_spec(0.2))
    assert kuramoto.target_names == ("x_tau", "y_tau", "w_tau")
    assert kuramoto.truth_hyperedges == (("w", "x", "x_tau"), ("w", "y", "y_tau"))
    assert kuramoto.truth_pairwise == (("w", "x_tau"), ("w", "y_tau"))

    rossler = build_future_state_spec(build_coupled_rossler_spec(0.5))
    assert rossler.target_names == ("x0_tau", "y0_tau", "z0_tau", "x1_tau", "y1_tau", "z1_tau")
    assert rossler.truth_hyperedges == (("x0", "x1", "x0_tau"), ("x0", "x1", "x1_tau"))


def test_kuramoto_finite_time_next_states_wrap_to_phase_domain() -> None:
    spec = build_kuramoto_coupling_spec(0.2)
    states = np.array([[3.13, 3.12, 3.11]])
    next_states = simulate_finite_time_next_states(spec, states, tau=0.2, process_noise=0.0, seed=7)
    assert np.min(next_states) >= -np.pi
    assert np.max(next_states) <= np.pi


def test_natural_trajectory_pool_is_reproducible_and_separates_seeded_pools() -> None:
    spec = build_wilson_cowan_gain_spec(5.1)
    states_a, targets_a = simulate_natural_trajectory_pool(
        spec,
        seed=11,
        trajectories=4,
        samples_per_trajectory=25,
        burnin_steps=5,
        noise=0.01,
    )
    states_b, targets_b = simulate_natural_trajectory_pool(
        spec,
        seed=11,
        trajectories=4,
        samples_per_trajectory=25,
        burnin_steps=5,
        noise=0.01,
    )
    readout_states, _ = simulate_natural_trajectory_pool(
        spec,
        seed=12,
        trajectories=4,
        samples_per_trajectory=25,
        burnin_steps=5,
        noise=0.01,
    )

    assert states_a.shape == targets_a.shape == (100, 3)
    assert np.allclose(states_a, states_b)
    assert np.allclose(targets_a, targets_b)
    assert not np.allclose(states_a, readout_states)
    assert np.ptp(states_a[:, 0]) > 0.05


def test_kuramoto_coupling_sweep_smoke_writes_json_and_png(tmp_path: Path) -> None:
    result = run_kuramoto_coupling_sweep(
        mode="smoke",
        couplings=(0.0, 0.2),
        seeds=(0,),
        result_path=tmp_path / "kuramoto_coupling_synergy_sweep.json",
        figure_path=tmp_path / "kuramoto_coupling_synergy_sweep.png",
    )

    result_path = Path(result["result_path"])
    figure_path = Path(result["figure_path"])
    assert result_path.exists()
    assert figure_path.exists()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["system"] == "kuramoto_phase_coupling"
    assert payload["training_distribution"] == "equal_natural_and_uniform_intervention"
    assert payload["shared_readout_state_distribution"] == "natural_trajectory_for_wms_surd_shap"
    assert payload["peid_readout_state_distribution"] == "independent_uniform_intervention"
    assert [row["coupling"] for row in payload["summary"]] == [0.0, 0.2]
    assert "readout_state_digest" in payload["rows"][0]
    assert "peid_readout_state_digest" in payload["rows"][0]
    assert abs(float(payload["summary"][0]["peid_synergy_mean"])) < 0.35
    assert float(payload["summary"][1]["peid_synergy_mean"]) > float(payload["summary"][0]["peid_synergy_mean"])


def test_kuramoto_peid_detail_sweep_exposes_scale_invariant_oracle_components(tmp_path: Path) -> None:
    result = run_kuramoto_peid_detail_sweep(
        mode="smoke",
        couplings=(0.0, 0.01, 0.1),
        seeds=(0,),
        result_path=tmp_path / "kuramoto_peid_detail_sweep.json",
        figure_path=tmp_path / "kuramoto_peid_detail_sweep.png",
    )

    payload = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert Path(result["figure_path"]).exists()
    assert payload["system"] == "kuramoto_phase_coupling_peid_detail"
    assert payload["sampling_distribution"] == "independent_uniform_intervention"
    assert [row["coupling"] for row in payload["summary"]] == [0.0, 0.01, 0.1]
    assert {
        "mlp_syn_mean",
        "mlp_joint_ei_mean",
        "mlp_single_ei_sum_mean",
        "oracle_syn_mean",
        "oracle_joint_ei_mean",
        "oracle_single_ei_sum_mean",
        "signal_rms_mean",
    } <= set(payload["summary"][0])
    positive = payload["summary"][1:]
    assert payload["summary"][0]["mlp_syn_mean"] == 0.0
    assert np.isclose(positive[0]["oracle_syn_mean"], positive[1]["oracle_syn_mean"])
    assert np.isclose(positive[0]["oracle_joint_ei_mean"], positive[1]["oracle_joint_ei_mean"])
    assert positive[1]["signal_rms_mean"] > positive[0]["signal_rms_mean"]


def test_lorenz_rho_spec_matches_classic_equations() -> None:
    spec = build_lorenz_rho_spec(28.0)
    state = np.array([[1.0, 2.0, 3.0]])
    expected = np.array(
        [[
            10.0 * (2.0 - 1.0),
            1.0 * (28.0 - 3.0) - 2.0,
            1.0 * 2.0 - (8.0 / 3.0) * 3.0,
        ]]
    )
    assert np.allclose(spec.vector_field(state), expected)

    next_states = simulate_finite_time_next_states(spec, state, tau=0.05, process_noise=0.0, seed=7)
    assert next_states.shape == state.shape
    assert np.isfinite(next_states).all()


def test_sis_gate_sweep_smoke_writes_json_and_png(tmp_path: Path) -> None:
    result = run_sis_gate_sweep(
        mode="smoke",
        betas=(0.0, 1.0),
        seeds=(0,),
        result_path=tmp_path / "sis_gate_synergy_sweep.json",
        figure_path=tmp_path / "sis_gate_synergy_sweep.png",
    )

    result_path = Path(result["result_path"])
    figure_path = Path(result["figure_path"])
    assert result_path.exists()
    assert figure_path.exists()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["training_distribution"] == "multi_initial_condition_natural_trajectory_pool"
    assert payload["shared_readout_state_distribution"] == "natural_trajectory_for_wms_surd_shap"
    assert payload["peid_readout_state_distribution"] == "independent_uniform_intervention"
    assert payload["method_data_contract"]["model_training"] == "one_shared_natural_training_pool"
    assert payload["method_data_contract"]["observational_readout"] == "one_shared_held_out_natural_pool"
    assert payload["method_data_contract"]["peid_interventions"] == "method_internal_sampling_only"
    assert [row["beta"] for row in payload["summary"]] == [0.0, 1.0]
    assert {"wms_mean", "surd_synergy_mean", "shap_interaction_mean", "peid_synergy_mean"} <= set(
        payload["summary"][0]
    )
    assert "readout_state_digest" in payload["rows"][0]
    assert "train_state_digest" in payload["rows"][0]
    assert "peid_target_digest" in payload["rows"][0]
    assert "peid_readout_state_digest" in payload["rows"][0]
    for method in ("wms", "surd_synergy", "shap_interaction", "peid_synergy"):
        assert f"raw_{method}" in payload["rows"][0]
        assert float(payload["rows"][0][method]) == float(payload["rows"][0][f"raw_{method}"])
    assert float(payload["summary"][1]["peid_synergy_mean"]) > float(payload["summary"][0]["peid_synergy_mean"])


def test_lorenz_rho_sweep_smoke_writes_json_and_png(tmp_path: Path) -> None:
    result = run_lorenz_rho_sweep(
        mode="smoke",
        rhos=(10.0, 28.0),
        seeds=(0,),
        result_path=tmp_path / "lorenz_rho_synergy_sweep.json",
        figure_path=tmp_path / "lorenz_rho_synergy_sweep.png",
    )

    result_path = Path(result["result_path"])
    figure_path = Path(result["figure_path"])
    assert result_path.exists()
    assert figure_path.exists()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["system"] == "lorenz3d_next_state"
    assert payload["shared_readout_state_distribution"] == "held_out_multi_initial_condition_natural_trajectory_pool"
    assert payload["peid_readout_state_distribution"] == "independent_uniform_intervention"
    assert payload["method_data_contract"]["model_training"] == "one_shared_natural_training_pool"
    assert payload["method_data_contract"]["observational_readout"] == "one_shared_held_out_natural_pool"
    assert payload["method_data_contract"]["peid_interventions"] == "method_internal_sampling_only"
    assert payload["truth_hyperedges"] == ["x+y->z_tau"]
    assert [row["rho"] for row in payload["summary"]] == [10.0, 28.0]
    assert "train_state_digest" in payload["rows"][0]
    assert "readout_state_digest" in payload["rows"][0]
    assert "peid_target_digest" in payload["rows"][0]


def test_lorenz_uniform_tau_sweep_smoke_selects_tau_and_writes_outputs(tmp_path: Path) -> None:
    result = run_lorenz_uniform_tau_sweep(
        mode="smoke",
        rhos=(10.0, 28.0),
        taus=(0.01, 0.05),
        seeds=(0,),
        result_path=tmp_path / "lorenz_uniform_tau_synergy_sweep.json",
        figure_path=tmp_path / "lorenz_uniform_tau_best_synergy.png",
    )

    result_path = Path(result["result_path"])
    figure_path = Path(result["figure_path"])
    assert result_path.exists()
    assert figure_path.exists()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["system"] == "lorenz3d_uniform_tau_next_state"
    assert payload["sampling_distribution"] == "independent_uniform"
    assert payload["selected_tau"] in [0.01, 0.05]
    assert {row["tau"] for row in payload["summary"]} == {payload["selected_tau"]}
    assert [row["rho"] for row in payload["summary"]] == [10.0, 28.0]
    assert "train_state_digest" in payload["rows"][0]
    assert "readout_state_digest" in payload["rows"][0]


def test_natural_trajectory_model_sweeps_write_auditable_json_and_png(tmp_path: Path) -> None:
    rossler = run_coupled_rossler_coupling_sweep(
        mode="smoke",
        couplings=(0.0, 0.25),
        seeds=(0,),
        result_path=tmp_path / "rossler.json",
        figure_path=tmp_path / "rossler.png",
    )
    wilson = run_wilson_cowan_gain_sweep(
        mode="smoke",
        gains=(1.0, 5.1),
        seeds=(0,),
        result_path=tmp_path / "wilson.json",
        figure_path=tmp_path / "wilson.png",
    )

    for result, parameter, values in (
        (rossler, "coupling", [0.0, 0.25]),
        (wilson, "gain", [1.0, 5.1]),
    ):
        payload = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
        assert Path(result["figure_path"]).exists()
        assert payload["training_distribution"] == "multi_initial_condition_natural_trajectory_pool"
        assert payload["natural_readout_state_distribution"] == "held_out_multi_initial_condition_natural_trajectory_pool"
        assert payload["peid_readout_state_distribution"] == "independent_uniform_intervention"
        assert payload["target"] == "instantaneous_vector_field"
        assert [row[parameter] for row in payload["summary"]] == values
        assert {"wms_mean", "surd_synergy_mean", "shap_interaction_mean", "peid_synergy_mean"} <= set(
            payload["summary"][0]
        )
        assert payload["rows"][0]["train_state_digest"] != payload["rows"][0]["readout_state_digest"]
        assert payload["rows"][0]["peid_readout_state_digest"] != payload["rows"][0]["readout_state_digest"]
        assert {"peid_joint_ei", "peid_single_ei_sum"} <= set(payload["rows"][0])


def test_ode_future_state_sweeps_write_retargeted_results(tmp_path: Path) -> None:
    result = run_ode_future_state_sweeps(
        mode="smoke",
        seeds=(0,),
        result_dir=tmp_path,
        figure_dir=tmp_path,
        kuramoto_couplings=(0.0, 0.05),
        wilson_cowan_gains=(1.0, 2.0),
        rossler_couplings=(0.0, 0.1),
    )

    assert set(result["systems"]) == {"kuramoto", "wilson_cowan", "rossler"}
    assert Path(result["combined_figure_path"]).exists()
    for payload in result["systems"].values():
        assert payload["target"] == "finite_time_next_state"
        assert payload["truth_hyperedges"]
        assert Path(payload["result_path"]).exists()
        assert Path(payload["figure_path"]).exists()
    assert result["systems"]["kuramoto"]["tau"] == 2.0
    assert result["systems"]["wilson_cowan"]["tau"] == 0.02
    assert result["systems"]["rossler"]["tau"] == 0.1
    assert result["systems"]["rossler"]["truth_hyperedges"] == [
        "x0+z0->z0_tau",
        "x1+z1->z1_tau",
    ]


def test_wilson_cowan_natural_redundancy_does_not_replace_intervention_peid() -> None:
    spec = build_wilson_cowan_gain_spec(5.1)
    natural_states, _ = simulate_natural_trajectory_pool(
        spec,
        seed=2000,
        trajectories=12,
        samples_per_trajectory=150,
        burnin_steps=20,
        noise=0.01,
    )
    rng = np.random.default_rng(99)
    intervention_states = np.column_stack(
        [rng.uniform(low, high, size=len(natural_states)) for low, high in spec.intervention_bounds]
    )
    natural = estimate_peid_from_samples(
        spec, natural_states, spec.vector_field(natural_states), estimator="histogram"
    )["hyperedges"].set_index(["sources", "target"])
    intervention = estimate_peid_from_samples(
        spec, intervention_states, spec.vector_field(intervention_states), estimator="histogram"
    )["hyperedges"].set_index(["sources", "target"])

    assert float(natural.loc[("w+x", "dx"), "score"]) >= 0.0
    assert float(natural.loc[("w+x", "dx"), "signed_residual"]) < 0.0
    assert float(natural.loc[("w+x", "dx"), "source_tc"]) > float(intervention.loc[("w+x", "dx"), "source_tc"])
    assert float(intervention.loc[("w+x", "dx"), "score"]) > 0.0


def test_peid_hyperedge_score_keeps_signed_transport_syn(monkeypatch) -> None:
    spec = build_wilson_cowan_gain_spec(2.0)
    states = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ],
        dtype=float,
    )
    targets = spec.vector_field(states)

    def fake_transport_synergy(left, right, target):
        return {"left_ei": 0.2, "right_ei": 0.3, "joint_ei": 0.4, "syn": -0.1}

    monkeypatch.setattr(
        "scripts.classic_network_dynamics_benchmark._transport_synergy",
        fake_transport_synergy,
    )

    hyperedges = estimate_peid_from_samples(
        spec,
        states,
        targets,
        estimator="transport",
    )["hyperedges"]

    assert float(hyperedges["score"].min()) == -0.1
    assert float(hyperedges["raw_syn"].min()) == -0.1


def test_zero_control_readouts_report_raw_estimates() -> None:
    readouts = _zero_control_synergy_readouts(
        inactive=True,
        wms=0.1,
        surd_synergy=0.2,
        shap_interaction=0.3,
        peid_synergy=0.4,
    )

    assert readouts["wms"] == 0.1
    assert readouts["surd_synergy"] == 0.2
    assert readouts["shap_interaction"] == 0.3
    assert readouts["peid_synergy"] == 0.4
    assert readouts["raw_peid_synergy"] == 0.4


def test_plot_panel_can_use_symlog_to_keep_small_curves_visible() -> None:
    import matplotlib.pyplot as plt

    summary = [
        {
            "coupling": coupling,
            "wms_mean": coupling,
            "wms_std": 0.01,
            "surd_synergy_mean": surd,
            "surd_synergy_std": surd_std,
            "shap_interaction_mean": coupling / 2,
            "shap_interaction_std": 0.01,
            "peid_synergy_mean": coupling / 10,
            "peid_synergy_std": 0.005,
        }
        for coupling, surd, surd_std in ((0.0, 0.1, 0.1), (0.2, 3.7, 5.3), (1.0, 0.6, 0.1))
    ]
    fig, axis = plt.subplots()

    _plot_panel(
        axis,
        summary,
        parameter_key="coupling",
        xlabel="Coupling",
        label="Standard map",
        symlog_linthresh=0.2,
    )

    assert axis.get_yscale() == "symlog"
    assert any(text.get_text() == "symlog y" for text in axis.texts)
    plt.close(fig)


def test_plot_panel_includes_oracle_peid_when_summary_has_oracle_fields() -> None:
    import matplotlib.pyplot as plt

    summary = [
        {
            "coupling": coupling,
            "wms_mean": coupling,
            "wms_std": 0.0,
            "surd_synergy_mean": coupling,
            "surd_synergy_std": 0.0,
            "shap_interaction_mean": coupling,
            "shap_interaction_std": 0.0,
            "peid_synergy_mean": coupling,
            "peid_synergy_std": 0.0,
            "oracle_peid_synergy_mean": coupling + 0.1,
            "oracle_peid_synergy_std": 0.0,
        }
        for coupling in (0.0, 1.0)
    ]
    fig, axis = plt.subplots()

    _plot_panel(axis, summary, parameter_key="coupling", xlabel="Coupling", label="Panel")

    _, labels = axis.get_legend_handles_labels()
    assert "Oracle PEID" in labels
    plt.close(fig)


def test_plot_panel_can_hide_oracle_peid_when_requested() -> None:
    import matplotlib.pyplot as plt

    summary = [
        {
            "coupling": coupling,
            "wms_mean": coupling,
            "wms_std": 0.0,
            "surd_synergy_mean": coupling,
            "surd_synergy_std": 0.0,
            "shap_interaction_mean": coupling,
            "shap_interaction_std": 0.0,
            "peid_synergy_mean": coupling,
            "peid_synergy_std": 0.0,
            "oracle_peid_synergy_mean": coupling + 0.1,
            "oracle_peid_synergy_std": 0.0,
        }
        for coupling in (0.0, 1.0)
    ]
    fig, axis = plt.subplots()

    _plot_panel(
        axis,
        summary,
        parameter_key="coupling",
        xlabel="Coupling",
        label="Panel",
        include_oracle_peid=False,
    )

    _, labels = axis.get_legend_handles_labels()
    assert "Oracle PEID" not in labels
    assert labels == ["WMS", "SURD synergy", "MLP+SHAP interaction", "MLP+PEID synergy"]
    plt.close(fig)


def test_part1_combined_figure_uses_rulkov_henon_cournot_ikeda_and_nicholson_bailey(tmp_path: Path) -> None:
    standard = {
        "summary": [
            {
                "coupling": 0.0,
                "wms_mean": 0.0,
                "wms_std": 0.0,
                "surd_synergy_mean": 0.1,
                "surd_synergy_std": 0.0,
                "shap_interaction_mean": 0.0,
                "shap_interaction_std": 0.0,
                "peid_synergy_mean": 0.0,
                "peid_synergy_std": 0.0,
            },
            {
                "coupling": 1.0,
                "wms_mean": 0.2,
                "wms_std": 0.0,
                "surd_synergy_mean": 0.1,
                "surd_synergy_std": 0.0,
                "shap_interaction_mean": 0.3,
                "shap_interaction_std": 0.0,
                "peid_synergy_mean": 0.4,
                "peid_synergy_std": 0.0,
            },
        ]
    }
    rulkov = {
        "parameter_key": "alpha",
        "summary": [{**row, "alpha": row.pop("coupling")} for row in json.loads(json.dumps(standard["summary"]))],
    }
    henon = {"summary": [{**row, "kappa": row.pop("coupling")} for row in json.loads(json.dumps(standard["summary"]))]}
    cournot = {
        "parameter_key": "lambda",
        "summary": [{**row, "lambda": row.pop("coupling")} for row in json.loads(json.dumps(standard["summary"]))],
    }
    ikeda = {
        "parameter_key": "u",
        "summary": [{**row, "u": row.pop("coupling")} for row in json.loads(json.dumps(standard["summary"]))],
    }
    nicholson_bailey = {
        "parameter_key": "a",
        "summary": [{**row, "a": row.pop("coupling")} for row in json.loads(json.dumps(standard["summary"]))],
    }
    standard_path = tmp_path / "standard.json"
    rulkov_path = tmp_path / "rulkov.json"
    henon_path = tmp_path / "henon.json"
    cournot_path = tmp_path / "cournot.json"
    ikeda_path = tmp_path / "ikeda_y_tau.json"
    nicholson_bailey_path = tmp_path / "nicholson_bailey.json"
    standard_path.write_text(json.dumps(standard), encoding="utf-8")
    rulkov_path.write_text(json.dumps(rulkov), encoding="utf-8")
    henon_path.write_text(json.dumps(henon), encoding="utf-8")
    cournot_path.write_text(json.dumps(cournot), encoding="utf-8")
    ikeda_path.write_text(json.dumps(ikeda), encoding="utf-8")
    nicholson_bailey_path.write_text(json.dumps(nicholson_bailey), encoding="utf-8")

    result = run_part1_combined_synergy_figure(
        standard_result_path=standard_path,
        rulkov_result_path=rulkov_path,
        henon_result_path=henon_path,
        cournot_result_path=cournot_path,
        ikeda_result_path=ikeda_path,
        nicholson_bailey_result_path=nicholson_bailey_path,
        figure_path=tmp_path / "combined.png",
    )

    assert Path(result["figure_path"]).exists()
    assert result["panels"] == {
        "standard_map": str(standard_path),
        "rulkov": str(rulkov_path),
        "coupled_henon": str(henon_path),
        "cournot": str(cournot_path),
        "ikeda_y_tau": str(ikeda_path),
        "nicholson_bailey": str(nicholson_bailey_path),
    }


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
