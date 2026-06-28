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
    KURAMOTO_COUPLING_VALUES,
    LARGE_KURAMOTO_LEARNED_NSOURCE_COUPLINGS,
    LARGE_KURAMOTO_ORACLE_WHOLE_STATE_COUPLINGS,
    ModelSpec,
    build_coupled_rossler_spec,
    build_kuramoto_coupling_spec,
    build_lorenz_rho_spec,
    build_sis_gate_spec,
    build_wilson_cowan_gain_spec,
    build_future_state_spec,
    build_model_specs,
    estimate_peid_from_samples,
    estimate_peid_for_joint_targets_from_samples,
    estimate_oracle_peid,
    run_benchmark,
    run_coupled_rossler_coupling_sweep,
    run_kuramoto_coupling_sweep,
    run_kuramoto_joint_target_peid,
    run_kuramoto_joint_target_peid_sweep,
    run_large_kuramoto_phi_sweep,
    run_large_kuramoto_oracle_nsource_phi_susceptibility_sweep,
    run_large_kuramoto_learned_nsource_phi_sweep,
    run_large_kuramoto_oracle_nsource_whole_state_phi_sweep,
    run_kuramoto_phase_response_peid_sweep,
    run_kuramoto_peid_detail_sweep,
    run_lorenz_rho_sweep,
    run_lorenz_uniform_tau_sweep,
    run_ode_future_state_sweeps,
    run_part1_combined_synergy_figure,
    _plot_panel,
    _kuramoto_order_parameter,
    _kuramoto_order_excess,
    _nsource_peid_from_ei,
    _null_corrected_phi_from_values,
    _transport_nsource_phi,
    _smooth_curve_gaussian_kernel,
    _zero_control_synergy_readouts,
    run_sis_gate_sweep,
    run_wilson_cowan_gain_sweep,
    simulate_natural_trajectory_pool,
    simulate_finite_time_next_states,
    simulate_sis_gate_next_states,
    _aggregate_large_kuramoto_learned_nsource_rows,
)
from scripts.discrete_iteration_dynamics_benchmark import (
    _broad_one_step_distribution_metadata,
    _broad_one_step_readout_factory,
    _broad_one_step_surrogate_factory,
    _broad_one_step_sweep_parameters,
    _run_map_sweep,
    build_wilson_cowan_refractory_spec,
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


def test_discrete_wilson_cowan_refractory_sweeps_sigmoid_gain() -> None:
    zero_gain = build_wilson_cowan_refractory_spec(0.0)
    active_gain = build_wilson_cowan_refractory_spec(4.0)

    assert zero_gain.parameter_key == "gain"
    assert zero_gain.parameter_values == (0.0, 0.4, 0.7, 1.0, 1.4, 2.0, 3.2, 4.0, 6.0)

    baseline = np.array([[0.30, 0.40]])
    left = np.array([[0.36, 0.40]])
    right = np.array([[0.30, 0.46]])
    both = np.array([[0.36, 0.46]])
    zero_mixed_difference = (
        zero_gain.transition(both)[0, 0]
        - zero_gain.transition(left)[0, 0]
        - zero_gain.transition(right)[0, 0]
        + zero_gain.transition(baseline)[0, 0]
    )
    active_mixed_difference = (
        active_gain.transition(both)[0, 0]
        - active_gain.transition(left)[0, 0]
        - active_gain.transition(right)[0, 0]
        + active_gain.transition(baseline)[0, 0]
    )

    assert abs(float(zero_mixed_difference)) < 1e-12
    assert abs(float(active_mixed_difference)) > 1e-5


def test_histogram_peid_treats_near_constant_targets_as_degenerate() -> None:
    spec = build_kuramoto_coupling_spec(0.0)
    rng = np.random.default_rng(7)
    states = rng.uniform(-np.pi, np.pi, size=(700, 2))
    targets = np.column_stack(
        [
            np.full(len(states), 1.0) + rng.normal(0.0, 1e-8, len(states)),
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
    state = np.array([[0.2, 0.7]])
    inactive = build_kuramoto_coupling_spec(0.0)
    active = build_kuramoto_coupling_spec(0.2)

    inactive_field = inactive.vector_field(state)[0]
    active_field = active.vector_field(state)[0]
    assert inactive.state_names == ("theta1", "theta2")
    assert inactive.target_names == ("dtheta1", "dtheta2")
    assert inactive.truth_hyperedges == (("theta1", "theta2", "dtheta1"),)
    assert inactive.truth_pairwise == (("theta1", "dtheta1"), ("theta2", "dtheta1"))
    assert np.isclose(inactive_field[0], 1.0 + 0.2 * np.sin(0.2))
    assert np.isclose(inactive_field[1], 0.9 + 0.2 * np.sin(0.7))
    assert np.isclose(active_field[0], 1.0 + 0.2 * np.sin(0.2) + 0.2 * np.sin(0.7 - 0.2))
    assert np.isclose(active_field[1], inactive_field[1])
    assert active_field[0] != inactive_field[0]
    assert active_field[1] == inactive_field[1]


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
    assert kuramoto.target_names == ("theta1_tau", "theta2_tau")
    assert kuramoto.truth_hyperedges == (("theta1", "theta2", "theta1_tau"),)
    assert kuramoto.truth_pairwise == (("theta1", "theta1_tau"), ("theta2", "theta1_tau"))

    rossler = build_future_state_spec(build_coupled_rossler_spec(0.5))
    assert rossler.target_names == ("x0_tau", "y0_tau", "z0_tau", "x1_tau", "y1_tau", "z1_tau")
    assert rossler.truth_hyperedges == (("x0", "x1", "x0_tau"), ("x0", "x1", "x1_tau"))


def test_kuramoto_finite_time_next_states_wrap_to_phase_domain() -> None:
    spec = build_kuramoto_coupling_spec(0.2)
    states = np.array([[3.13, 3.11]])
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
    assert payload["training_distribution"] == "same_natural_trajectory_pool_as_observational_readout"
    assert payload["shared_readout_state_distribution"] == "natural_trajectory_for_wms_surd_shap"
    assert payload["peid_readout_state_distribution"] == "independent_uniform_intervention"
    assert payload["mlp_error_evaluation"] == "in_sample_on_shared_natural_training_and_observational_readout_pool"
    assert payload["target_relation"] == "theta1+theta2->dtheta1"
    assert payload["frequency_detuning"] == 0.1
    assert payload["phase_potential_strength"] == 0.2
    assert payload["figure_contract"] == {
        "panel_a": ["phase_locking_value", "phase_order_parameter"],
        "panel_b": ["wms", "peid_synergy", "oracle_peid_synergy"],
        "y_axis_label": "Synergy / Interaction",
    }
    assert [row["coupling"] for row in payload["summary"]] == [0.0, 0.2]
    assert "readout_state_digest" in payload["rows"][0]
    assert payload["rows"][0]["train_state_digest"] == payload["rows"][0]["readout_state_digest"]
    assert payload["rows"][0]["train_target_digest"] == payload["rows"][0]["observed_target_digest"]
    assert "peid_readout_state_digest" in payload["rows"][0]
    assert "phase_locking_value" in payload["rows"][0]
    assert "phase_order_parameter" in payload["rows"][0]
    assert "phase_order_parameter_mean" in payload["summary"][0]
    assert "oracle_peid_synergy" in payload["rows"][0]
    assert "wms_joint_mi" in payload["rows"][0]
    assert "wms_left_mi" in payload["rows"][0]
    assert "wms_right_mi" in payload["rows"][0]
    assert payload["rows"][0]["peid_readout_state_digest"] == payload["rows"][0]["oracle_peid_readout_state_digest"]
    assert payload["rows"][0]["shap_mlp_model_digest"] == payload["rows"][0]["peid_mlp_model_digest"]
    assert abs(float(payload["summary"][0]["peid_synergy_mean"])) < 0.35
    assert float(payload["summary"][1]["peid_synergy_mean"]) > float(payload["summary"][0]["peid_synergy_mean"])


def test_default_kuramoto_couplings_extend_into_synchronized_phase() -> None:
    assert KURAMOTO_COUPLING_VALUES[0] == 0.0
    assert max(KURAMOTO_COUPLING_VALUES) >= 2.0
    assert 0.5 in KURAMOTO_COUPLING_VALUES
    assert 1.0 in KURAMOTO_COUPLING_VALUES


def test_kuramoto_order_parameter_tracks_phase_coherence() -> None:
    synchronized = np.array([[0.1, 0.1], [1.0, 1.0], [-2.0, -2.0]])
    antiphase = np.array([[0.0, np.pi], [0.5, 0.5 + np.pi], [-1.0, -1.0 + np.pi]])
    random_phases = np.random.default_rng(7).uniform(-np.pi, np.pi, size=(4000, 2))

    assert np.isclose(_kuramoto_order_parameter(synchronized), 1.0)
    assert np.isclose(_kuramoto_order_parameter(antiphase), 0.0)
    assert _kuramoto_order_excess(random_phases) < 0.04
    assert np.isclose(_kuramoto_order_excess(synchronized), 1.0)


def test_kuramoto_peid_detail_sweep_exposes_active_rotator_coupling_components(tmp_path: Path) -> None:
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
    assert payload["truth_hyperedges"] == ["theta1+theta2->dtheta1"]
    assert payload["phase_potential_strength"] == 0.2
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
    assert positive[1]["oracle_syn_mean"] > positive[0]["oracle_syn_mean"]
    for row in positive:
        assert np.isclose(
            row["oracle_joint_ei_mean"] - row["oracle_single_ei_sum_mean"],
            row["oracle_syn_mean"],
        )
    assert positive[1]["signal_rms_mean"] > positive[0]["signal_rms_mean"]


def test_joint_target_peid_treats_kuramoto_two_phase_velocities_as_one_target() -> None:
    spec = build_kuramoto_coupling_spec(0.2)
    rng = np.random.default_rng(7)
    states = rng.uniform(-np.pi, np.pi, size=(700, 2))
    targets = spec.vector_field(states)

    graph = estimate_peid_for_joint_targets_from_samples(
        spec,
        states,
        targets,
        joint_targets={"dtheta": ("dtheta1", "dtheta2")},
        estimator="transport",
    )
    hyperedges = graph["hyperedges"].set_index(["sources", "target"])
    pairwise = graph["pairwise"].set_index(["source", "target"])

    assert list(graph["target_names"]) == ["dtheta"]
    assert ("theta1+theta2", "dtheta") in hyperedges.index
    assert ("theta1", "dtheta") in pairwise.index
    assert float(hyperedges.loc[("theta1+theta2", "dtheta"), "joint_ei"]) > 0.0
    assert np.isclose(
        float(hyperedges.loc[("theta1+theta2", "dtheta"), "score"]),
        float(hyperedges.loc[("theta1+theta2", "dtheta"), "signed_residual"]),
    )


def test_kuramoto_joint_target_peid_smoke_writes_single_example_payload(tmp_path: Path) -> None:
    result = run_kuramoto_joint_target_peid(
        mode="smoke",
        coupling=0.2,
        seeds=(0,),
        result_path=tmp_path / "kuramoto_joint_target_peid.json",
        figure_path=tmp_path / "kuramoto_joint_target_peid.png",
    )

    payload = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert Path(result["figure_path"]).exists()
    assert payload["system"] == "kuramoto_phase_coupling_joint_target"
    assert payload["coupling"] == 0.2
    assert payload["target_relation"] == "theta1+theta2->dtheta"
    assert payload["joint_target"] == ["dtheta1", "dtheta2"]
    assert payload["equation_parameters"] == {"omega1": 1.0, "omega2": 0.9, "A": 0.2, "K": 0.2}
    assert payload["summary"]["oracle_joint_ei_mean"] > 0.0
    assert "mlp_syn_mean" in payload["summary"]
    assert "oracle_syn_mean" in payload["summary"]


def test_kuramoto_joint_target_peid_sweep_tracks_syn_across_couplings(tmp_path: Path) -> None:
    result = run_kuramoto_joint_target_peid_sweep(
        mode="smoke",
        couplings=(0.0, 0.2, 1.0),
        seeds=(0,),
        result_path=tmp_path / "kuramoto_joint_target_peid_sweep.json",
        figure_path=tmp_path / "kuramoto_joint_target_peid_sweep.png",
    )

    payload = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert Path(result["figure_path"]).exists()
    assert payload["system"] == "kuramoto_phase_coupling_joint_target_sweep"
    assert payload["target_relation"] == "theta1+theta2->dtheta"
    assert [row["coupling"] for row in payload["summary"]] == [0.0, 0.2, 1.0]
    assert "oracle_syn_peak_coupling" in payload["nonmonotonic_diagnostic"]
    assert "mlp_syn_peak_coupling" in payload["nonmonotonic_diagnostic"]
    assert {"mlp_syn_mean", "oracle_syn_mean"} <= set(payload["summary"][0])


def test_kuramoto_phase_response_peid_sweep_reports_transition_and_syn_peaks(tmp_path: Path) -> None:
    result = run_kuramoto_phase_response_peid_sweep(
        mode="smoke",
        couplings=(0.0, 0.1, 0.3),
        seeds=(0,),
        tau=1.0,
        result_path=tmp_path / "kuramoto_phase_response_peid_sweep.json",
        figure_path=tmp_path / "kuramoto_phase_response_peid_sweep.png",
    )

    payload = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert Path(result["figure_path"]).exists()
    assert payload["system"] == "kuramoto_phase_response_peid_sweep"
    assert payload["target"] == "finite_time_phase_locking_response"
    assert payload["phase_response_target"] == ["cos_delta_tau", "sin_delta_tau", "order_excess_tau"]
    assert [row["coupling"] for row in payload["summary"]] == [0.0, 0.1, 0.3]
    assert "order_transition_coupling" in payload["criticality_diagnostic"]
    assert "oracle_syn_peak_coupling" in payload["criticality_diagnostic"]
    assert {"natural_plv_mean", "natural_order_mean", "natural_order_raw_mean", "oracle_syn_mean"} <= set(
        payload["summary"][0]
    )


def test_large_kuramoto_phi_sweep_reports_classic_transition_and_phi_peak(tmp_path: Path) -> None:
    result = run_large_kuramoto_phi_sweep(
        mode="smoke",
        oscillator_count=32,
        couplings=(0.0, 0.8, 1.6, 2.4),
        seeds=(0,),
        tau=3.0,
        result_path=tmp_path / "large_kuramoto_phi_sweep.json",
        figure_path=tmp_path / "large_kuramoto_phi_sweep.png",
    )

    payload = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert Path(result["figure_path"]).exists()
    assert payload["system"] == "classic_large_n_kuramoto_phi_sweep"
    assert payload["oscillator_count"] == 32
    assert payload["target"] == "finite_time_global_order_response"
    assert payload["source_partition"] == ["oscillators_0_to_15", "oscillators_16_to_31"]
    assert [row["coupling"] for row in payload["summary"]] == [0.0, 0.8, 1.6, 2.4]
    assert payload["critical_coupling_theory"] > 0.0
    assert "order_transition_coupling" in payload["criticality_diagnostic"]
    assert "phi_syn_peak_coupling" in payload["criticality_diagnostic"]
    assert {"natural_order_mean", "phi_syn_mean", "phi_joint_ei_mean"} <= set(payload["summary"][0])


def test_large_kuramoto_learned_nsource_phi_sweep_uses_all_oscillators_as_sources(tmp_path: Path) -> None:
    result = run_large_kuramoto_learned_nsource_phi_sweep(
        mode="smoke",
        oscillator_count=4,
        couplings=(0.0, 2.6),
        seeds=(0,),
        tau=2.0,
        nsource_null_shuffles=2,
        result_path=tmp_path / "large_kuramoto_learned_nsource_phi_sweep.json",
        figure_path=tmp_path / "large_kuramoto_learned_nsource_phi_sweep.png",
    )

    payload = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert Path(result["figure_path"]).exists()
    assert payload["system"] == "classic_large_n_kuramoto_learned_nsource_phi_sweep"
    assert payload["source_partition"] == "singleton_oscillators"
    assert payload["source_count"] == 4
    assert payload["target"] == "learned_finite_time_global_order_response"
    assert payload["target_components"] == ["delta_order_excess_tau"]
    assert payload["nsource_transport_map_degree"] == 1
    assert payload["nsource_null_shuffles"] == 2
    assert payload["phi_null_model"] == "target_shuffle"
    assert payload["phi_definition"] == "EI(all oscillator sources; target) - sum_i EI(oscillator_i; target)"
    assert "null_corrected_phi_definition" in payload
    assert [row["coupling"] for row in payload["summary"]] == [0.0, 2.6]
    assert {
        "learned_phi_mean",
        "learned_observed_phi_mean",
        "learned_null_phi_mean",
        "learned_joint_ei_mean",
        "learned_singleton_ei_sum_mean",
    } <= set(payload["summary"][0])
    assert payload["summary"][0]["learned_phi_mean"] <= payload["summary"][0]["learned_observed_phi_mean"]
    assert "learned_phi_peak_coupling" in payload["criticality_diagnostic"]


def test_large_kuramoto_learned_nsource_defaults_use_dense_transition_grid() -> None:
    transition_grid = [k for k in LARGE_KURAMOTO_LEARNED_NSOURCE_COUPLINGS if 1.2 <= k <= 2.2]

    assert len(transition_grid) >= 7
    assert 1.6 in LARGE_KURAMOTO_LEARNED_NSOURCE_COUPLINGS
    assert max(np.diff(transition_grid)) <= 0.2000001


def test_large_kuramoto_oracle_whole_state_grid_resolves_low_coupling_rise() -> None:
    low_grid = [k for k in LARGE_KURAMOTO_ORACLE_WHOLE_STATE_COUPLINGS if 0.0 <= k <= 1.0]

    assert low_grid == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def test_large_kuramoto_learned_nsource_summary_reports_sem() -> None:
    rows = []
    for seed, phi in enumerate((0.0, 0.1, 0.2)):
        rows.append(
            {
                "coupling": 1.6,
                "seed": seed,
                "natural_order": 0.5 + 0.01 * seed,
                "natural_order_raw": 0.6 + 0.01 * seed,
                "learned_phi": phi,
                "learned_raw_phi": phi,
                "learned_observed_phi": phi + 0.01,
                "learned_observed_raw_phi": phi + 0.01,
                "learned_null_phi": 0.01,
                "learned_null_phi_std": 0.0,
                "learned_null_corrected_phi": phi,
                "learned_joint_ei": phi + 0.2,
                "learned_singleton_ei_sum": 0.2,
                "oracle_phi": phi / 2.0,
                "oracle_raw_phi": phi / 2.0,
                "oracle_observed_phi": phi / 2.0 + 0.01,
                "oracle_observed_raw_phi": phi / 2.0 + 0.01,
                "oracle_null_phi": 0.01,
                "oracle_null_phi_std": 0.0,
                "oracle_null_corrected_phi": phi / 2.0,
                "oracle_joint_ei": phi / 2.0 + 0.2,
                "oracle_singleton_ei_sum": 0.2,
                "mlp_test_mse": 0.1,
                "mlp_baseline_mse": 0.2,
            }
        )

    summary = _aggregate_large_kuramoto_learned_nsource_rows(rows)

    assert summary[0]["learned_phi_sem"] < summary[0]["learned_phi_std"]
    assert summary[0]["natural_order_sem"] < summary[0]["natural_order_std"]


def test_nsource_phi_uses_effective_information_joint_minus_singleton_sum() -> None:
    decomposition = _nsource_peid_from_ei(np.array([0.40, 0.30, 0.20]), joint_ei=0.55)

    assert np.isclose(decomposition["singleton_ei_sum"], 0.90)
    assert np.isclose(decomposition["phi"], 0.55 - 0.90)
    assert np.isclose(decomposition["raw_phi"], 0.55 - 0.90)


def test_target_shuffle_null_correction_keeps_signed_residual_without_clipping() -> None:
    correction = _null_corrected_phi_from_values(observed_phi=0.02, null_values=np.array([0.03, 0.05]))

    assert correction["null_phi"] == 0.04
    assert correction["null_corrected_phi"] == -0.02


def test_affine_nsource_phi_uses_stable_gaussian_block_ctc_for_separable_identity() -> None:
    rng = np.random.default_rng(0)
    source_blocks = [rng.normal(size=(300, 2)) for _ in range(4)]
    target = np.column_stack(source_blocks)

    result = _transport_nsource_phi(source_blocks, target, degree=1)

    assert result["phi"] >= -1.0e-9
    assert result["phi"] < 0.05
    assert np.isclose(result["phi"], result["joint_ei"] - result["singleton_ei_sum"])
    assert result["phi_estimator"] == "gaussian_block_conditional_total_correlation"


def test_large_kuramoto_oracle_nsource_phi_susceptibility_sweep_reports_corrected_oracle_peak(
    tmp_path: Path,
) -> None:
    result = run_large_kuramoto_oracle_nsource_phi_susceptibility_sweep(
        mode="smoke",
        oscillator_count=4,
        couplings=(0.0, 1.6, 3.2),
        seeds=(0,),
        tau=2.0,
        nsource_null_shuffles=1,
        result_path=tmp_path / "large_kuramoto_oracle_nsource_phi_susceptibility_sweep.json",
        figure_path=tmp_path / "large_kuramoto_oracle_nsource_phi_susceptibility_sweep.png",
    )

    payload = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert Path(result["figure_path"]).exists()
    assert payload["system"] == "classic_large_n_kuramoto_oracle_nsource_phi_susceptibility_sweep"
    assert payload["estimator"] == "oracle_transport_map"
    assert payload["source_partition"] == "singleton_oscillators"
    assert payload["target_components"] == ["d_order_excess_tau_dK"]
    assert payload["phi_null_model"] == "target_shuffle"
    assert payload["phi_definition"] == "EI(all oscillator sources; target) - sum_i EI(oscillator_i; target)"
    assert "null_corrected_phi_definition" in payload
    assert {"oracle_phi_mean", "oracle_observed_phi_mean", "oracle_null_phi_mean", "oracle_phi_sem"} <= set(
        payload["summary"][0]
    )
    assert "oracle_phi_peak_coupling" in payload["criticality_diagnostic"]


def test_large_kuramoto_oracle_nsource_whole_state_phi_sweep_matches_dmf_style_target(
    tmp_path: Path,
) -> None:
    result = run_large_kuramoto_oracle_nsource_whole_state_phi_sweep(
        mode="smoke",
        oscillator_count=4,
        couplings=(0.0, 1.6),
        seeds=(0,),
        tau=2.0,
        nsource_null_shuffles=1,
        result_path=tmp_path / "large_kuramoto_oracle_nsource_whole_state_phi_sweep.json",
        figure_path=tmp_path / "large_kuramoto_oracle_nsource_whole_state_phi_sweep.png",
    )

    payload = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert Path(result["figure_path"]).exists()
    assert payload["system"] == "classic_large_n_kuramoto_oracle_nsource_whole_state_phi_sweep"
    assert payload["estimator"] == "oracle_transport_map"
    assert payload["source_partition"] == "singleton_oscillators"
    assert payload["source_feature"] == "per-oscillator cos(theta), sin(theta)"
    assert payload["target"] == "oracle_finite_time_whole_system_phase_state"
    assert payload["target_components"] == ["cos(theta_tau)_all", "sin(theta_tau)_all"]
    assert payload["target_dimension"] == 8
    assert payload["phi_definition"] == "EI(all oscillator sources; target) - sum_i EI(oscillator_i; target)"
    assert {"oracle_phi_mean", "oracle_joint_ei_mean", "oracle_singleton_ei_sum_mean"} <= set(payload["summary"][0])
    assert "oracle_phi_peak_coupling" in payload["criticality_diagnostic"]


def test_gaussian_kernel_smoothing_spreads_local_phi_peak_without_changing_grid() -> None:
    couplings = np.array([1.2, 1.4, 1.6])
    phi = np.array([0.0, 0.03, 0.0])

    smoothed = _smooth_curve_gaussian_kernel(couplings, phi, bandwidth=0.25)

    assert smoothed.shape == phi.shape
    assert 0.0 < smoothed[0] < smoothed[1] < phi[1]
    assert 0.0 < smoothed[2] < smoothed[1]


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


def test_zero_control_readouts_report_fitted_estimates_and_keep_raw_values() -> None:
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
    assert readouts["raw_wms"] == 0.1
    assert readouts["raw_surd_synergy"] == 0.2
    assert readouts["raw_shap_interaction"] == 0.3
    assert readouts["raw_peid_synergy"] == 0.4


def test_part1_discrete_zero_and_active_points_emit_passing_fairness_audit(tmp_path: Path) -> None:
    payload = _run_map_sweep(
        system="wilson_cowan_refractory",
        mode="smoke",
        parameter_values=(0.0, 0.4),
        seeds=(0,),
        result_path=tmp_path / "result.json",
        figure_path=tmp_path / "figure.png",
        structural_zero_values=(0.0,),
        params_override=_broad_one_step_sweep_parameters(
            "smoke", system="wilson_cowan_refractory"
        ),
        surrogate_factory=_broad_one_step_surrogate_factory,
        readout_factory=_broad_one_step_readout_factory,
        peid_uses_readout_states=True,
        distribution_metadata=_broad_one_step_distribution_metadata(),
    )

    assert payload["fairness_audit"]["passed"]
    assert payload["fairness_audit"]["zero_parameter_uses_same_pipeline"]
    assert payload["fairness_audit"]["parameter_matched_train_states"]
    assert payload["fairness_audit"]["parameter_matched_readout_states"]
    zero, active = payload["rows"]
    assert zero["train_state_digest"] == active["train_state_digest"]
    assert zero["readout_state_digest"] == active["readout_state_digest"]
    assert zero["readout_state_digest"] == zero["peid_readout_state_digest"]
    assert zero["shap_mlp_model_digest"] == zero["peid_mlp_model_digest"]
    assert zero["wms_estimator"] == active["wms_estimator"] == "transport_map"
    assert zero["surd_estimator"] == active["surd_estimator"] == "transport_map"
    assert zero["peid_estimator"] == active["peid_estimator"] == "transport_map"


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


def test_plot_panel_can_put_surd_on_a_separate_right_axis() -> None:
    import matplotlib.pyplot as plt

    summary = [
        {
            "coupling": coupling,
            "wms_mean": coupling,
            "wms_std": 0.0,
            "surd_synergy_mean": 10.0 * coupling,
            "surd_synergy_std": 0.0,
            "shap_interaction_mean": coupling,
            "shap_interaction_std": 0.0,
            "peid_synergy_mean": coupling,
            "peid_synergy_std": 0.0,
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
        separate_surd_axis=True,
    )

    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "SURD synergy (bits)"
    assert [line.get_label() for line in axis.lines if not line.get_label().startswith("_")] == [
        "WMS",
        "MLP+SHAP interaction",
        "MLP+PEID synergy",
    ]
    assert [line.get_label() for line in fig.axes[1].lines] == ["SURD synergy"]
    plt.close(fig)


def test_part1_combined_figure_uses_refractory_wilson_cowan_kuramoto_henon_ikeda_and_nicholson_bailey(tmp_path: Path) -> None:
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
    wilson_cowan_refractory = {
        "parameter_key": "gain",
        "summary": [{**row, "gain": row.pop("coupling")} for row in json.loads(json.dumps(standard["summary"]))],
    }
    kuramoto = {
        "parameter_key": "coupling",
        "summary": json.loads(json.dumps(standard["summary"])),
    }
    controlled_henon = {
        "system": "controlled_henon_unique_information_five_method",
        "parameter_key": "lambda",
        "summary": [
            {
                **row,
                "lambda": row.pop("coupling"),
                "gamma": 0.3,
                "kappa": 0.5,
                "mmi_pid_synergy_mean": 0.5,
                "mmi_pid_synergy_std": 0.0,
            }
            for row in json.loads(json.dumps(standard["summary"]))
        ],
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
    wilson_cowan_refractory_path = tmp_path / "wilson_cowan_refractory.json"
    kuramoto_path = tmp_path / "kuramoto.json"
    controlled_henon_path = tmp_path / "controlled_henon.json"
    ikeda_path = tmp_path / "ikeda_y_tau.json"
    nicholson_bailey_path = tmp_path / "nicholson_bailey.json"
    standard_path.write_text(json.dumps(standard), encoding="utf-8")
    wilson_cowan_refractory_path.write_text(json.dumps(wilson_cowan_refractory), encoding="utf-8")
    kuramoto_path.write_text(json.dumps(kuramoto), encoding="utf-8")
    controlled_henon_path.write_text(json.dumps(controlled_henon), encoding="utf-8")
    ikeda_path.write_text(json.dumps(ikeda), encoding="utf-8")
    nicholson_bailey_path.write_text(json.dumps(nicholson_bailey), encoding="utf-8")
    mmi_summary = {
        "systems": {
            system_key: {
                "parameter_key": parameter_key,
                "summary": [
                    {
                        parameter_key: value,
                        "mmi_pid_synergy_mean": 0.05 + value,
                        "mmi_pid_synergy_std": 0.0,
                    }
                    for value in (0.0, 1.0)
                ],
            }
            for system_key, parameter_key in {
                "standard_map": "coupling",
                "wilson_cowan_refractory": "gain",
                "kuramoto": "coupling",
                "ikeda_y_tau": "u",
                "nicholson_bailey": "a",
            }.items()
        }
    }
    mmi_summary_path = tmp_path / "mmi_summary.json"
    mmi_summary_path.write_text(json.dumps(mmi_summary), encoding="utf-8")

    result = run_part1_combined_synergy_figure(
        standard_result_path=standard_path,
        wilson_cowan_refractory_result_path=wilson_cowan_refractory_path,
        kuramoto_result_path=kuramoto_path,
        controlled_henon_result_path=controlled_henon_path,
        ikeda_result_path=ikeda_path,
        nicholson_bailey_result_path=nicholson_bailey_path,
        mmi_pid_result_path=mmi_summary_path,
        figure_path=tmp_path / "combined.png",
    )

    assert Path(result["figure_path"]).exists()
    assert result["panels"] == {
        "standard_map": str(standard_path),
        "wilson_cowan_refractory": str(wilson_cowan_refractory_path),
        "kuramoto_phase_coupling": str(kuramoto_path),
        "controlled_henon_unique_information": str(controlled_henon_path),
        "ikeda_y_tau": str(ikeda_path),
        "nicholson_bailey": str(nicholson_bailey_path),
    }
    assert result["y_axis_label"] == "Synergy / Interaction"
    assert result["panel_method_counts"] == {
        "standard_map": 5,
        "wilson_cowan_refractory": 5,
        "kuramoto_phase_coupling": 5,
        "controlled_henon_unique_information": 5,
        "ikeda_y_tau": 5,
        "nicholson_bailey": 5,
    }
    assert result["panel_parameter_keys"]["controlled_henon_unique_information"] == "lambda"
    assert result["panel_xlabels"]["controlled_henon_unique_information"] == "Hénon control parameter lambda"


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
