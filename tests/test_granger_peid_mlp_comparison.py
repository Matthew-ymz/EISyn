from __future__ import annotations

import json
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.compare_granger_peid_mlp as comparison
from scripts.compare_granger_peid_mlp import (
    SimConfig,
    _observational_wms,
    _plot_sine_alpha_neural_granger_sweep,
    _plot_sine_alpha_sweep,
    _plot_sine_beta_combined_readout_sweep,
    _plot_sine_beta_single_source_sweep,
    _plot_sine_beta_synergy_sweep,
    _proxy_y_readout_values,
    _sample_sine_beta_peid_intervention_sources,
    estimate_granger_graph,
    estimate_peid_graph,
    make_lagged_dataset,
    run_comparison_grid,
    run_lagged_proxy_common_driver_experiment,
    run_lag_sensitivity_lagged_proxy_experiment,
    run_neural_granger_readout,
    run_neural_granger_lagged_proxy_experiment,
    run_sine_alpha_sweep,
    run_sine_beta_common_driver_sweep,
    simulate_system,
    train_mlp_transition_model,
)


class _ConstantPredictionModel:
    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=float)


def test_proxy_y_readout_uses_single_standard_shap_baseline() -> None:
    edge_rows = [
        {
            "mechanism": "common_driver_sine_synergy",
            "run_id": "run-0",
            "edge_type": "conditional_shap",
            "source": "x",
            "target": "y",
            "mean_abs_phi": 0.7,
        },
        {
            "mechanism": "common_driver_sine_synergy",
            "run_id": "run-0",
            "edge_type": "interventional_shap",
            "source": "x",
            "target": "y",
            "mean_abs_phi": 0.2,
        },
        {
            "mechanism": "common_driver_sine_synergy",
            "run_id": "run-0",
            "edge_type": "peid_pairwise",
            "source": "x",
            "target": "y",
            "ei": 0.1,
        },
        {
            "mechanism": "common_driver_sine_synergy",
            "run_id": "run-0",
            "edge_type": "granger_pairwise",
            "source": "x",
            "target": "y",
            "score": 0.05,
        },
    ]

    rows = _proxy_y_readout_values(edge_rows)
    methods = {str(row["method"]) for row in rows}
    shap_x = next(row for row in rows if row["method"] == "SHAP" and row["source"] == "x")

    assert "SHAP" in methods
    assert "conditional SHAP" not in methods
    assert "interventional SHAP" not in methods
    assert float(shap_x["value"]) == 0.2


def test_peid_synergy_keeps_signed_joint_minus_single_value(monkeypatch) -> None:
    config = SimConfig(intervention_samples=32, bins=4)
    series = pd.DataFrame(
        {
            "x": np.linspace(-1.0, 1.0, 64),
            "y": np.linspace(1.0, -1.0, 64),
            "z": np.sin(np.linspace(0.0, 2.0, 64)),
            "w": np.cos(np.linspace(0.0, 2.0, 64)),
        }
    )
    single_values = {
        ("x", "z"): 0.6,
        ("y", "z"): 0.5,
    }
    joint_values = {
        ("x+y", "z"): 0.8,
    }
    names = config.variable_names
    call_index = {"single": 0, "joint": 0}

    def fake_effective_information(source_states, target_states):
        source_array = np.asarray(source_states)
        if source_array.ndim == 1:
            idx = call_index["single"]
            call_index["single"] += 1
            source = names[idx // len(names)]
            target = names[idx % len(names)]
            return single_values.get((source, target), 0.0)
        idx = call_index["joint"]
        call_index["joint"] += 1
        source_a, source_b = combinations_index[idx // len(names)]
        target = names[idx % len(names)]
        return joint_values.get((f"{source_a}+{source_b}", target), 0.0)

    combinations_index = [
        ("x", "y"),
        ("x", "z"),
        ("x", "w"),
        ("y", "z"),
        ("y", "w"),
        ("z", "w"),
    ]
    monkeypatch.setattr(comparison, "_effective_information_from_states", fake_effective_information)

    peid = estimate_peid_graph(_ConstantPredictionModel(), series, config)
    edge = peid.synergy_edges[
        (peid.synergy_edges["sources"] == "x+y") & (peid.synergy_edges["target"] == "z")
    ].iloc[0]

    assert abs(float(edge["synergy"]) + 0.3) < 1e-12


def test_observational_wms_keeps_signed_whole_minus_sum(monkeypatch) -> None:
    def fail_if_discretized(*args, **kwargs):
        raise AssertionError("observational WMS must use the degree-3 transport-map estimator")

    def fake_transport_wms(left, right, target):
        assert np.asarray(left).shape == (8, 1)
        assert np.asarray(right).shape == (8, 1)
        assert np.asarray(target).shape == (8, 1)
        return {
            "backend": "polynomial_triangular_transport_map_degree_3",
            "left_ei": 0.6,
            "right_ei": 0.5,
            "joint_ei": 0.8,
            "syn": -0.3,
        }

    monkeypatch.setattr(comparison, "_discretize_vector", fail_if_discretized)
    monkeypatch.setattr(
        "yrd.transport_map.summarize_two_source_synergy_transport_map",
        fake_transport_wms,
    )
    result = _observational_wms(
        np.arange(8, dtype=float),
        np.arange(8, dtype=float) + 1.0,
        np.arange(8, dtype=float) + 2.0,
    )

    assert abs(result["x_mi"] - 0.6) < 1e-12
    assert abs(result["y_mi"] - 0.5) < 1e-12
    assert abs(result["joint_mi"] - 0.8) < 1e-12
    assert abs(result["wms"] + 0.3) < 1e-12
    assert result["estimator"] == "polynomial_triangular_transport_map_degree_3"


def test_observational_wms_decreases_with_full_common_driver() -> None:
    values = {}
    for beta in (0.0, 1.0):
        series, _ = simulate_system(
            SimConfig(
                mechanism="common_driver_sine_synergy",
                n_samples=1100,
                noise=0.05,
                seed=0,
                synergy_strength=1.0,
                common_driver_strength=beta,
                bins=4,
            )
        )
        values[beta] = _observational_wms(
            series["x"].to_numpy(dtype=float)[:-1],
            series["y"].to_numpy(dtype=float)[:-1],
            series["z"].to_numpy(dtype=float)[1:],
            bins=4,
        )["wms"]

    assert values[0.0] > 0.0
    assert values[1.0] < 0.0


def test_common_driver_sine_synergy_adds_beta_scaled_driver_to_target() -> None:
    beta = 1.0
    series, truth = simulate_system(
        SimConfig(
            mechanism="common_driver_sine_synergy",
            n_samples=120,
            noise=0.0,
            seed=7,
            synergy_strength=1.0,
            common_driver_strength=beta,
        )
    )

    expected_target = (
        0.22 * series["z"].to_numpy(dtype=float)[:-1]
        + np.sin(
            series["x"].to_numpy(dtype=float)[:-1]
            * series["y"].to_numpy(dtype=float)[:-1]
        )
        + 0.15 * beta * series["w"].to_numpy(dtype=float)[:-1]
    )

    np.testing.assert_allclose(
        series["z"].to_numpy(dtype=float)[1:],
        expected_target,
        atol=1e-12,
    )
    assert ("w", "z") in truth["pairwise_edges"]


def test_beta_peid_intervention_uses_fixed_source_support_only() -> None:
    config = SimConfig(
        mechanism="common_driver_sine_synergy",
        n_samples=240,
        seed=3,
        intervention_samples=96,
    )
    series, _ = simulate_system(config)
    baseline = comparison._sample_intervention_sources(series, config)

    samples = _sample_sine_beta_peid_intervention_sources(
        series,
        config,
        source_support=(-1.8, 1.8),
    )

    assert samples["x"].between(-1.8, 1.8).all()
    assert samples["y"].between(-1.8, 1.8).all()
    assert np.isclose(float(samples["x"].min()), -1.8, atol=0.15)
    assert np.isclose(float(samples["x"].max()), 1.8, atol=0.15)
    assert np.allclose(samples["z"], baseline["z"])
    assert np.allclose(samples["w"], baseline["w"])


def _trained_model(config: SimConfig):
    series, truth = simulate_system(config)
    features, targets = make_lagged_dataset(series, lag=config.lag)
    model = train_mlp_transition_model(features, targets, config)
    return series, truth, features, targets, model


def test_xor_mechanism_peid_synergy_exceeds_single_source_ei() -> None:
    config = SimConfig(
        mechanism="xor_synergy",
        n_samples=1200,
        noise=0.0,
        seed=3,
        mlp_epochs=220,
        intervention_samples=768,
        bins=2,
    )
    series, truth, features, targets, model = _trained_model(config)

    peid = estimate_peid_graph(model, series, config)
    edge = peid.synergy_edges[
        (peid.synergy_edges["sources"] == "x+y") & (peid.synergy_edges["target"] == "z")
    ].iloc[0]
    xy_pairwise = peid.pairwise_edges[
        (peid.pairwise_edges["source"].isin(["x", "y"])) & (peid.pairwise_edges["target"] == "z")
    ]

    assert ("x", "y", "z") in truth["hyperedges"]
    assert float(edge["synergy"]) > 0.25
    assert float(edge["synergy"]) > float(xy_pairwise["ei"].max())


def test_linear_additive_mechanism_granger_and_peid_recover_pairwise_edges() -> None:
    config = SimConfig(
        mechanism="linear_additive",
        n_samples=1000,
        noise=0.03,
        seed=5,
        mlp_epochs=120,
        intervention_samples=512,
    )
    series, truth, features, targets, model = _trained_model(config)

    granger = estimate_granger_graph(model, features, targets, config)
    peid = estimate_peid_graph(model, series, config)

    granger_edges = granger.sort_values("score", ascending=False).head(4)
    peid_edges = peid.pairwise_edges.sort_values("ei", ascending=False).head(4)

    assert ("x", "z") in truth["pairwise_edges"]
    assert ("y", "z") in truth["pairwise_edges"]
    assert {("x", "z"), ("y", "z")}.issubset(set(zip(granger_edges["source"], granger_edges["target"])))
    assert {("x", "z"), ("y", "z")}.issubset(set(zip(peid_edges["source"], peid_edges["target"])))


def test_multiplicative_gate_joint_peid_exceeds_best_individual() -> None:
    config = SimConfig(
        mechanism="multiplicative_gate",
        n_samples=1200,
        noise=0.02,
        synergy_strength=1.5,
        seed=11,
        mlp_epochs=180,
        intervention_samples=768,
    )
    series, truth, features, targets, model = _trained_model(config)

    peid = estimate_peid_graph(model, series, config)
    edge = peid.synergy_edges[
        (peid.synergy_edges["sources"] == "x+y") & (peid.synergy_edges["target"] == "z")
    ].iloc[0]

    assert ("x", "y", "z") in truth["hyperedges"]
    assert float(edge["joint_ei"]) > float(edge["best_single_ei"])
    assert float(edge["synergy"]) > 0.05


def test_product_memory_synergy_is_second_order_dynamic_mechanism() -> None:
    config = SimConfig(
        mechanism="product_memory_synergy",
        n_samples=1600,
        noise=0.02,
        synergy_strength=1.8,
        seed=17,
        mlp_epochs=220,
        intervention_samples=1024,
        bins=5,
    )
    series, truth, features, targets, model = _trained_model(config)

    peid = estimate_peid_graph(model, series, config)
    edge = peid.synergy_edges[
        (peid.synergy_edges["sources"] == "x+y") & (peid.synergy_edges["target"] == "z")
    ].iloc[0]
    xy_pairwise = peid.pairwise_edges[
        (peid.pairwise_edges["source"].isin(["x", "y"])) & (peid.pairwise_edges["target"] == "z")
    ]

    assert ("x", "y", "z") in truth["hyperedges"]
    assert float(edge["joint_ei"]) > 1.0
    assert float(edge["synergy"]) > 0.12
    assert float(edge["synergy"]) > float(xy_pairwise["ei"].max())


def test_common_driver_sine_synergy_separates_driver_and_hyperedge() -> None:
    config = SimConfig(
        mechanism="common_driver_sine_synergy",
        n_samples=1600,
        noise=0.02,
        synergy_strength=1.2,
        seed=23,
        mlp_epochs=220,
        intervention_samples=1024,
        bins=5,
    )
    series, truth, features, targets, model = _trained_model(config)

    granger = estimate_granger_graph(model, features, targets, config)
    peid = estimate_peid_graph(model, series, config)
    edge = peid.synergy_edges[
        (peid.synergy_edges["sources"] == "x+y") & (peid.synergy_edges["target"] == "z")
    ].iloc[0]

    granger_w_to_x = granger[(granger["source"] == "w") & (granger["target"] == "x")].iloc[0]
    granger_w_to_y = granger[(granger["source"] == "w") & (granger["target"] == "y")].iloc[0]
    granger_w_to_z = granger[(granger["source"] == "w") & (granger["target"] == "z")].iloc[0]
    peid_w_to_z = peid.pairwise_edges[
        (peid.pairwise_edges["source"] == "w") & (peid.pairwise_edges["target"] == "z")
    ].iloc[0]

    assert {("w", "x"), ("w", "y"), ("w", "z")} == set(truth["pairwise_edges"])
    assert ("x", "y", "z") in truth["hyperedges"]
    assert float(granger_w_to_x["score"]) > 0.2
    assert float(granger_w_to_y["score"]) > 0.2
    assert float(granger_w_to_z["score"]) > 0.01
    assert float(peid_w_to_z["ei"]) > 0.01
    assert float(edge["joint_ei"]) > 1.0
    assert float(edge["synergy"]) > 0.5
    assert float(edge["synergy"]) > float(edge["best_single_ei"])


def test_lagged_proxy_common_driver_granger_false_positive_but_peid_keeps_proxy_small() -> None:
    result = run_lagged_proxy_common_driver_experiment(
        n_samples=5000,
        noise=0.05,
        seed=0,
        bins=8,
        intervention_samples=4096,
    )

    assert result["pairwise_linear_x_to_y_score"] > 0.2
    assert result["pairwise_linear_x_to_y_r2"] > 0.95
    assert result["pairwise_granger_x_to_y_score"] < 0.05
    assert result["pairwise_granger_edges"]["w->y"] > result["pairwise_granger_edges"]["x->y"] * 20.0
    assert abs(result["causal_state_coef_x_proxy"]) < 0.02
    assert result["causal_state_coef_w_driver"] > 0.65
    assert result["peid_ei_w_to_y"] > 2.0
    assert result["peid_mlp_y_train_mse"] < 0.01
    assert result["peid_ei_x_to_y"] < 0.2
    assert result["peid_proxy_to_driver_ratio"] < 0.08


def test_underlagged_mlp_makes_granger_and_peid_follow_proxy() -> None:
    result = run_lag_sensitivity_lagged_proxy_experiment(
        n_samples=5000,
        noise=0.05,
        seed=0,
        bins=8,
        intervention_samples=4096,
    )
    lag1 = result["by_lag"][1]
    lag2 = result["by_lag"][2]

    assert lag1["granger_edges"]["x->y"] > lag1["granger_edges"]["w->y"] * 20.0
    assert lag1["peid_ei_edges"]["x->y"] > lag1["peid_ei_edges"]["w->y"] * 5.0
    assert lag2["granger_edges"]["w->y"] > lag2["granger_edges"]["x->y"] * 20.0
    assert lag2["peid_ei_edges"]["w->y"] > lag2["peid_ei_edges"]["x->y"] * 10.0


def test_neural_granger_lagged_proxy_keeps_proxy_under_collinearity() -> None:
    result = run_neural_granger_lagged_proxy_experiment(
        n_samples=3000,
        noise=0.05,
        seed=0,
        model_seed=1,
        epochs=250,
    )
    rows = result["rows"]

    lag1_y_top = next(row for row in rows if row["max_lag"] == 1 and row["target"] == "y" and row["rank"] == 1)
    lag2_y_top = next(row for row in rows if row["max_lag"] == 2 and row["target"] == "y" and row["rank"] == 1)
    lag2_y_w = next(row for row in rows if row["max_lag"] == 2 and row["target"] == "y" and row["source"] == "w")
    lag2_y_x = next(row for row in rows if row["max_lag"] == 2 and row["target"] == "y" and row["source"] == "x")

    assert lag1_y_top["source"] == "x"
    assert lag2_y_top["source"] == "w"
    assert lag2_y_w["rank"] == 1
    assert lag2_y_w["strongest_lag"] == 2
    assert lag2_y_w["group_norm"] > 0.2
    assert lag2_y_x["rank"] == 2
    assert lag2_y_x["strongest_lag"] == 1
    assert lag2_y_x["group_norm"] > 0.1


def test_four_variable_neural_granger_readout_returns_sine_edges() -> None:
    config = SimConfig(
        mechanism="common_driver_sine_synergy",
        n_samples=320,
        noise=0.05,
        seed=0,
        synergy_strength=1.0,
        mlp_epochs=2,
    )
    series, _ = simulate_system(config)
    features, targets = make_lagged_dataset(series, lag=config.lag)

    readout = run_neural_granger_readout(
        features,
        targets,
        variable_names=config.variable_names,
        max_lag=config.lag,
        epochs=2,
        hidden_dim=8,
        group_lasso=0.01,
        seed=0,
    )
    rows = readout["rows"]
    edge_keys = {(row["source"], row["target"]) for row in rows}

    assert ("x", "z") in edge_keys
    assert ("y", "z") in edge_keys
    assert ("w", "z") in edge_keys
    assert ("w", "x") in edge_keys
    assert ("w", "y") in edge_keys
    assert all("group_norm" in row for row in rows)


def test_smoke_grid_writes_summary_edges_and_png(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    figure_dir = tmp_path / "fig"

    output = run_comparison_grid(
        mode="smoke",
        mechanisms=("common_driver_sine_synergy",),
        seeds=(0,),
        noise_values=(0.05,),
        sample_values=(700,),
        synergy_values=(1.0,),
        result_dir=result_dir,
        figure_dir=figure_dir,
    )

    summary_path = Path(output["summary_path"])
    edge_path = Path(output["edge_table_path"])
    figure_path = Path(output["figure_path"])

    assert summary_path.exists()
    assert edge_path.exists()
    assert figure_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["runs"]
    assert any(run["mechanism"] == "common_driver_sine_synergy" for run in summary["runs"])
    assert Path(summary["graph_figure_path"]).exists()
    assert Path(summary["graph_figure_path"]).name == "representative_causal_graphs.png"
    assert Path(summary["report_figure_path"]).exists()
    assert Path(summary["lagged_proxy_figure_path"]).exists()
    assert Path(summary["lagged_proxy_figure_path"]).name == "lagged_proxy_causal_graph.png"
    assert "sine_beta_common_driver_sweep" in summary
    assert Path(summary["report_markdown_path"]).exists()
    report_text = Path(summary["report_markdown_path"]).read_text(encoding="utf-8")
    assert "统一动力系统：共同驱动 + sine 协同" in report_text
    assert "Granger/ablation `w -> x`" in report_text
    assert "Neural Granger `w -> x`" in report_text
    assert "PEID synergy `{x, y} -> z`" in report_text
    assert "transport-map PEID" not in report_text
    assert "滞后共同驱动造成的 Granger 伪边" not in report_text
    assert "Neural Granger 在同一主例上的表现" not in report_text
    assert "二阶协同动力系统：乘积记忆机制" not in report_text
    assert edge_path.read_text(encoding="utf-8").strip()


def test_product_memory_synergy_report_section_is_generated(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    figure_dir = tmp_path / "fig"

    output = run_comparison_grid(
        mode="smoke",
        mechanisms=("product_memory_synergy",),
        seeds=(0,),
        noise_values=(0.05,),
        sample_values=(700,),
        synergy_values=(1.0,),
        result_dir=result_dir,
        figure_dir=figure_dir,
    )

    summary = json.loads(Path(output["summary_path"]).read_text(encoding="utf-8"))
    report_text = Path(summary["report_markdown_path"]).read_text(encoding="utf-8")

    assert any(run["mechanism"] == "product_memory_synergy" for run in summary["runs"])
    assert "统一动力系统：共同驱动 + sine 协同" in report_text
    assert "二阶协同动力系统：乘积记忆机制" not in report_text


def test_common_driver_sine_synergy_report_section_is_generated(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    figure_dir = tmp_path / "fig"

    output = run_comparison_grid(
        mode="smoke",
        mechanisms=("common_driver_sine_synergy",),
        seeds=(0,),
        noise_values=(0.05,),
        sample_values=(700,),
        synergy_values=(1.0,),
        result_dir=result_dir,
        figure_dir=figure_dir,
    )

    summary = json.loads(Path(output["summary_path"]).read_text(encoding="utf-8"))
    report_text = Path(summary["report_markdown_path"]).read_text(encoding="utf-8")

    assert any(run["mechanism"] == "common_driver_sine_synergy" for run in summary["runs"])
    assert "统一动力系统：共同驱动 + sine 协同" in report_text
    assert "PEID joint EI `{x, y} -> z`" in report_text
    assert "xor_synergy" not in report_text


def test_alpha_sweep_reports_transport_map_peid_for_sine_synergy() -> None:
    rows = run_sine_alpha_sweep(
        alpha_values=(0.0, 0.2),
        n_samples=240,
        noise=0.05,
        seed=0,
        mlp_epochs=2,
        intervention_samples=96,
        bins=4,
    )

    assert rows
    for row in rows:
        assert "granger_x_to_z" in row
        assert "granger_y_to_z" in row
        assert "granger_w_to_z" in row
        assert "neural_granger_x_to_z" in row
        assert "neural_granger_y_to_z" in row
        assert "neural_granger_w_to_z" in row
        assert "tm_peid_xy_joint_ei" in row
        assert "tm_peid_xy_synergy" in row
        assert "tm_peid_x_to_z" in row
        assert "tm_peid_y_to_z" in row


def test_alpha_sweep_aggregates_multiple_seeds() -> None:
    rows = run_sine_alpha_sweep(
        alpha_values=(0.0,),
        n_samples=180,
        noise=0.05,
        seeds=(0, 1),
        mlp_epochs=1,
        intervention_samples=64,
        bins=4,
        neural_granger_epochs=1,
    )

    assert len(rows) == 1
    assert rows[0]["n_seeds"] == 2
    assert "shap_xy_mean_abs_interaction_std" in rows[0]
    assert "tm_peid_xy_synergy_std" in rows[0]


def test_alpha_sweep_plot_combines_shap_without_product_r2(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.axes

    plotted_labels: list[str] = []
    original_plot = matplotlib.axes.Axes.plot

    def capture_plot(self, *args, **kwargs):
        label = kwargs.get("label")
        if label:
            plotted_labels.append(str(label))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", capture_plot)
    path = _plot_sine_alpha_sweep(
        [
            {
                "alpha": 0.0,
                "shap_x_to_z_mean_abs": 0.0,
                "shap_y_to_z_mean_abs": 0.0,
                "shap_w_to_z_mean_abs": 0.0,
                "shap_xy_mean_abs_interaction": 0.0,
                "product_xy_incremental_r2": 0.0,
                "granger_x_to_z": 0.0,
                "granger_y_to_z": 0.0,
                "granger_w_to_z": 0.0,
                "tm_peid_xy_joint_ei": 0.0,
                "tm_peid_xy_synergy": 0.0,
                "tm_peid_x_to_z": 0.0,
                "tm_peid_y_to_z": 0.0,
                "tm_peid_w_to_z": 0.0,
            },
            {
                "alpha": 1.0,
                "shap_x_to_z_mean_abs": 0.1,
                "shap_y_to_z_mean_abs": 0.1,
                "shap_w_to_z_mean_abs": 0.02,
                "shap_xy_mean_abs_interaction": 0.3,
                "product_xy_incremental_r2": 0.8,
                "granger_x_to_z": 0.2,
                "granger_y_to_z": 0.2,
                "granger_w_to_z": 0.03,
                "tm_peid_xy_joint_ei": 0.9,
                "tm_peid_xy_synergy": 0.8,
                "tm_peid_x_to_z": 0.02,
                "tm_peid_y_to_z": 0.02,
                "tm_peid_w_to_z": 0.01,
            },
        ],
        tmp_path,
    )

    assert path is not None and path.exists()
    assert "SHAP interaction (x,y)->z" in plotted_labels
    assert "Granger x->z" in plotted_labels
    assert "Granger y->z" in plotted_labels
    assert "Neural Granger x->z" not in plotted_labels
    assert "Neural Granger y->z" not in plotted_labels
    assert "Neural Granger w->z" not in plotted_labels
    assert "product probe incremental R2" not in plotted_labels


def test_alpha_neural_granger_sweep_plot_is_standalone(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.axes

    plotted_labels: list[str] = []
    original_plot = matplotlib.axes.Axes.plot

    def capture_plot(self, *args, **kwargs):
        label = kwargs.get("label")
        if label:
            plotted_labels.append(str(label))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", capture_plot)
    path = _plot_sine_alpha_neural_granger_sweep(
        [
            {
                "alpha": 0.0,
                "neural_granger_x_to_z": 0.0,
                "neural_granger_y_to_z": 0.0,
                "neural_granger_w_to_z": 0.0,
            },
            {
                "alpha": 1.0,
                "neural_granger_x_to_z": 1.0,
                "neural_granger_y_to_z": 0.8,
                "neural_granger_w_to_z": 0.1,
            },
        ],
        tmp_path,
    )

    assert plotted_labels == [
        "Neural Granger x->z",
        "Neural Granger y->z",
        "Neural Granger w->z",
    ]
    assert path is not None and path.exists()
    assert path.name == "sine_alpha_neural_granger_sweep.png"


def test_beta_sweep_reports_transport_map_peid_when_enabled(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    figure_dir = tmp_path / "fig"

    output = run_comparison_grid(
        mode="smoke",
        mechanisms=("common_driver_sine_synergy",),
        seeds=(0,),
        noise_values=(0.05,),
        sample_values=(700,),
        synergy_values=(1.0,),
        result_dir=result_dir,
        figure_dir=figure_dir,
        include_diagnostic_sweeps=True,
    )

    summary = json.loads(Path(output["summary_path"]).read_text(encoding="utf-8"))
    beta_sweep = summary["sine_beta_common_driver_sweep"]
    assert beta_sweep["summary"]
    run = beta_sweep["runs"][0]
    expected_run_fields = {
        "run_id",
        "shap_x_to_z_mean_abs",
        "shap_y_to_z_mean_abs",
        "shap_xy_mean_abs_interaction",
        "observational_x_to_z_mi",
        "observational_y_to_z_mi",
        "observational_xy_to_z_joint_mi",
        "observational_wms",
        "surd_redundancy",
        "surd_unique_x",
        "surd_unique_y",
        "surd_xy_synergy",
        "surd_xy_joint",
        "mlp_peid_redundancy",
        "mlp_peid_unique_x",
        "mlp_peid_unique_y",
        "mlp_peid_xy_synergy",
        "mlp_peid_xy_joint",
        "oracle_peid_redundancy",
        "oracle_peid_unique_x",
        "oracle_peid_unique_y",
        "oracle_peid_xy_synergy",
        "oracle_peid_xy_joint",
    }
    assert expected_run_fields <= set(run)
    assert np.isclose(
        run["surd_redundancy"] + run["surd_unique_x"] + run["surd_unique_y"] + run["surd_xy_synergy"],
        run["surd_xy_joint"],
    )
    assert np.isclose(
        run["mlp_peid_unique_x"] + run["mlp_peid_unique_y"] + run["mlp_peid_xy_synergy"],
        run["mlp_peid_xy_joint"],
    )
    assert run["mlp_peid_redundancy"] == 0.0
    assert run["oracle_peid_redundancy"] == 0.0
    assert "surd_xy_synergy_mean" in beta_sweep["summary"][0]
    assert "observational_wms_mean" in beta_sweep["summary"][0]
    assert "observational_wms_slope" in beta_sweep["trend"]
    assert beta_sweep["config"]["common_driver_target_coefficient"] == 0.15
    assert "tm_peid_synergy_slope" in beta_sweep["trend"]
    assert "surd_synergy_slope" in beta_sweep["trend"]
    assert "oracle_peid_synergy_slope" in beta_sweep["trend"]
    assert "neural_granger_xy_to_z_mean" in beta_sweep["summary"][0]
    assert "neural_granger_xy_to_z_slope" in beta_sweep["trend"]
    assert Path(summary["beta_sweep_figure_path"]).name == "sine_beta_single_source_readout_sweep.png"
    assert Path(summary["beta_sweep_figure_path"]).exists()
    assert Path(summary["beta_synergy_figure_path"]).name == "sine_beta_synergy_readout_sweep.png"
    assert Path(summary["beta_synergy_figure_path"]).exists()
    assert Path(summary["beta_combined_figure_path"]).name == "sine_beta_combined_readout_sweep.png"
    assert Path(summary["beta_combined_figure_path"]).exists()
    assert Path(summary["beta_validation_figure_path"]).exists()

    report_text = Path(summary["report_markdown_path"]).read_text(encoding="utf-8")
    assert "Observational SURD" in report_text
    assert "MLP+SHAP" in report_text
    assert "MLP+PEID" in report_text
    assert "Neural Granger" in report_text
    assert "observational WMS" in report_text


def test_beta_sweep_reports_neural_granger_fields() -> None:
    result = run_sine_beta_common_driver_sweep(
        beta_values=(0.0, 0.5),
        seeds=(0,),
        n_samples=240,
        mlp_epochs=2,
        intervention_samples=96,
        bins=4,
        neural_granger_epochs=2,
    )

    assert result["runs"]
    run = result["runs"][0]
    summary = result["summary"][0]
    trend = result["trend"]
    assert "neural_granger_x_to_z" in run
    assert "neural_granger_y_to_z" in run
    assert "neural_granger_w_to_z" in run
    assert "neural_granger_xy_to_z_mean" in summary
    assert "neural_granger_xy_to_z_slope" in trend
    assert "observational_wms" in run
    assert run["wms_estimator"] == "polynomial_triangular_transport_map_degree_3"
    assert result["config"]["peid_source_support"] == [-1.8, 1.8]
    assert "observational_wms_mean" in summary
    assert "observational_wms_slope" in trend


def test_beta_sweep_default_grid_has_twenty_one_evenly_spaced_values() -> None:
    beta_values = inspect.signature(run_sine_beta_common_driver_sweep).parameters["beta_values"].default

    assert len(beta_values) == 21
    np.testing.assert_allclose(beta_values, np.linspace(0.0, 1.0, 21))


def test_beta_sweep_oracle_uses_one_fixed_intervention_protocol() -> None:
    result = run_sine_beta_common_driver_sweep(
        beta_values=(0.0, 1.0),
        seeds=(0, 1),
        n_samples=180,
        mlp_epochs=1,
        intervention_samples=640,
        bins=4,
        neural_granger_epochs=1,
        pcmci_cmiknn_sig_samples=2,
    )

    oracle_values = {
        (
            row["oracle_peid_unique_x"],
            row["oracle_peid_unique_y"],
            row["oracle_peid_xy_synergy"],
            row["oracle_peid_xy_joint"],
        )
        for row in result["runs"]
    }
    assert len(oracle_values) == 1
    assert all(row["oracle_peid_xy_synergy_std"] == 0.0 for row in result["summary"])
    assert result["config"]["oracle_intervention_support"] == {
        "x": [-1.8, 1.8],
        "y": [-1.8, 1.8],
        "z": [-1.25, 1.25],
    }
    assert result["config"]["oracle_intervention_seed"] == 17021
    assert np.isclose(
        result["summary"][0]["oracle_peid_unique_x_mean"],
        result["summary"][0]["oracle_peid_unique_y_mean"],
    )
    assert 0.50 < result["summary"][0]["oracle_peid_xy_synergy_mean"] < 0.65


def test_beta_sweep_plots_single_source_and_synergy_ground_truth(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.axes

    plotted_labels: list[str] = []
    plotted_colors: dict[str, str] = {}
    plotted_alphas: dict[str, float] = {}
    bar_calls: list[object] = []
    errorbar_calls: list[object] = []
    fill_between_calls: list[object] = []
    horizontal_lines: list[float] = []
    original_plot = matplotlib.axes.Axes.plot
    original_bar = matplotlib.axes.Axes.bar
    original_errorbar = matplotlib.axes.Axes.errorbar
    original_fill_between = matplotlib.axes.Axes.fill_between
    original_axhline = matplotlib.axes.Axes.axhline

    def capture_plot(self, *args, **kwargs):
        label = kwargs.get("label")
        if label:
            plotted_labels.append(str(label))
            plotted_colors[str(label)] = str(kwargs.get("color"))
            plotted_alphas[str(label)] = float(kwargs.get("alpha", 1.0))
        return original_plot(self, *args, **kwargs)

    def capture_bar(self, *args, **kwargs):
        bar_calls.append(args)
        return original_bar(self, *args, **kwargs)

    def capture_errorbar(self, *args, **kwargs):
        errorbar_calls.append(kwargs.get("yerr"))
        return original_errorbar(self, *args, **kwargs)

    def capture_fill_between(self, *args, **kwargs):
        fill_between_calls.append(args)
        return original_fill_between(self, *args, **kwargs)

    def capture_axhline(self, y=0, *args, **kwargs):
        horizontal_lines.append(float(y))
        return original_axhline(self, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", capture_plot)
    monkeypatch.setattr(matplotlib.axes.Axes, "bar", capture_bar)
    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", capture_errorbar)
    monkeypatch.setattr(matplotlib.axes.Axes, "fill_between", capture_fill_between)
    monkeypatch.setattr(matplotlib.axes.Axes, "axhline", capture_axhline)
    beta_result = {
        "summary": [
            {
                    "beta": 0.0,
                    "xy_observed_corr_mean": 0.0,
                    "xy_observed_corr_std": 0.0,
                    "observational_x_to_z_mi_mean": 0.1,
                    "observational_x_to_z_mi_std": 0.0,
                    "observational_y_to_z_mi_mean": 0.1,
                    "observational_y_to_z_mi_std": 0.0,
                    "observational_wms_mean": 0.2,
                    "observational_wms_std": 0.0,
                    "shap_x_to_z_mean_abs_mean": 0.1,
                    "shap_x_to_z_mean_abs_std": 0.0,
                    "shap_y_to_z_mean_abs_mean": 0.1,
                    "shap_y_to_z_mean_abs_std": 0.0,
                    "shap_xy_mean_abs_interaction_mean": 0.2,
                    "shap_xy_mean_abs_interaction_std": 0.0,
                    "surd_unique_x_mean": 0.01,
                    "surd_unique_x_std": 0.0,
                    "surd_unique_y_mean": 0.01,
                    "surd_unique_y_std": 0.0,
                    "surd_xy_synergy_mean": 0.2,
                    "surd_xy_synergy_std": 0.0,
                    "mlp_peid_unique_x_mean": 0.01,
                    "mlp_peid_unique_x_std": 0.0,
                    "mlp_peid_unique_y_mean": 0.01,
                    "mlp_peid_unique_y_std": 0.0,
                    "mlp_peid_xy_synergy_mean": 0.5,
                    "mlp_peid_xy_synergy_std": 0.0,
                    "oracle_peid_unique_x_mean": 0.01,
                    "oracle_peid_unique_x_std": 0.0,
                    "oracle_peid_unique_y_mean": 0.01,
                    "oracle_peid_unique_y_std": 0.0,
                    "oracle_peid_xy_synergy_mean": 0.5,
                    "oracle_peid_xy_synergy_std": 0.0,
                    "pcmci_cmiknn_x_to_z_mean": 0.2,
                    "pcmci_cmiknn_x_to_z_std": 0.0,
                    "pcmci_cmiknn_y_to_z_mean": 0.2,
                    "pcmci_cmiknn_y_to_z_std": 0.0,
                    "pcmci_cmiknn_w_to_z_mean": 0.0,
                    "pcmci_cmiknn_w_to_z_std": 0.0,
                    "neural_granger_x_to_z_mean": 1.0,
                    "neural_granger_x_to_z_std": 0.0,
                    "neural_granger_y_to_z_mean": 1.1,
                    "neural_granger_y_to_z_std": 0.0,
                    "neural_granger_w_to_z_mean": 0.0,
                    "neural_granger_w_to_z_std": 0.0,
                    "neural_granger_xy_to_z_mean": 2.1,
                    "neural_granger_xy_to_z_std": 0.0,
                    "product_xy_incremental_r2_mean": 0.9,
                    "product_xy_incremental_r2_std": 0.0,
            },
            {
                    "beta": 1.0,
                    "xy_observed_corr_mean": 0.9,
                    "xy_observed_corr_std": 0.0,
                    "observational_x_to_z_mi_mean": 0.2,
                    "observational_x_to_z_mi_std": 0.0,
                    "observational_y_to_z_mi_mean": 0.2,
                    "observational_y_to_z_mi_std": 0.0,
                    "observational_wms_mean": -0.1,
                    "observational_wms_std": 0.0,
                    "shap_x_to_z_mean_abs_mean": 0.2,
                    "shap_x_to_z_mean_abs_std": 0.0,
                    "shap_y_to_z_mean_abs_mean": 0.2,
                    "shap_y_to_z_mean_abs_std": 0.0,
                    "shap_xy_mean_abs_interaction_mean": 0.4,
                    "shap_xy_mean_abs_interaction_std": 0.0,
                    "surd_unique_x_mean": 0.01,
                    "surd_unique_x_std": 0.0,
                    "surd_unique_y_mean": 0.01,
                    "surd_unique_y_std": 0.0,
                    "surd_xy_synergy_mean": 0.1,
                    "surd_xy_synergy_std": 0.0,
                    "mlp_peid_unique_x_mean": 0.02,
                    "mlp_peid_unique_x_std": 0.0,
                    "mlp_peid_unique_y_mean": 0.02,
                    "mlp_peid_unique_y_std": 0.0,
                    "mlp_peid_xy_synergy_mean": 0.5,
                    "mlp_peid_xy_synergy_std": 0.0,
                    "oracle_peid_unique_x_mean": 0.02,
                    "oracle_peid_unique_x_std": 0.0,
                    "oracle_peid_unique_y_mean": 0.02,
                    "oracle_peid_unique_y_std": 0.0,
                    "oracle_peid_xy_synergy_mean": 0.6,
                    "oracle_peid_xy_synergy_std": 0.0,
                    "pcmci_cmiknn_x_to_z_mean": 0.1,
                    "pcmci_cmiknn_x_to_z_std": 0.0,
                    "pcmci_cmiknn_y_to_z_mean": 0.1,
                    "pcmci_cmiknn_y_to_z_std": 0.0,
                    "pcmci_cmiknn_w_to_z_mean": 0.0,
                    "pcmci_cmiknn_w_to_z_std": 0.0,
                    "neural_granger_x_to_z_mean": 1.2,
                    "neural_granger_x_to_z_std": 0.0,
                    "neural_granger_y_to_z_mean": 1.3,
                    "neural_granger_y_to_z_std": 0.0,
                    "neural_granger_w_to_z_mean": 0.0,
                    "neural_granger_w_to_z_std": 0.0,
                    "neural_granger_xy_to_z_mean": 2.5,
                    "neural_granger_xy_to_z_std": 0.0,
                    "product_xy_incremental_r2_mean": 0.8,
                    "product_xy_incremental_r2_std": 0.0,
            },
        ]
    }
    single_path = _plot_sine_beta_single_source_sweep(beta_result, tmp_path)
    synergy_path = _plot_sine_beta_synergy_sweep(beta_result, tmp_path)
    combined_path = _plot_sine_beta_combined_readout_sweep(beta_result, tmp_path)

    assert single_path is not None and single_path.exists()
    assert single_path.name == "sine_beta_single_source_readout_sweep.png"
    assert synergy_path is not None and synergy_path.exists()
    assert synergy_path.name == "sine_beta_synergy_readout_sweep.png"
    assert combined_path is not None and combined_path.exists()
    assert combined_path.name == "sine_beta_combined_readout_sweep.png"
    for stem in (
        "sine_beta_single_source_readout_sweep",
        "sine_beta_synergy_readout_sweep",
        "sine_beta_combined_readout_sweep",
    ):
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}.pdf").exists()
        assert (tmp_path / f"{stem}.svg").exists()
    assert "observational WMS" in plotted_labels
    assert "observed corr(x,y)" not in plotted_labels
    assert r"Oracle+PEID $U_x$" in plotted_labels
    assert r"Oracle+PEID $S_{xy}$" in plotted_labels
    assert not any(label.startswith("GT ") for label in plotted_labels)
    assert plotted_colors[r"MLP+PEID $U_x$"] == "#009E73"
    assert plotted_colors[r"Oracle+PEID $U_x$"] == "#7E57C2"
    assert plotted_colors["SHAP x->z"] == "#E68613"
    assert plotted_colors[r"MLP+PEID $S_{xy}$"] == "#009E73"
    assert plotted_colors[r"Oracle+PEID $S_{xy}$"] == "#7E57C2"
    assert plotted_colors["observational WMS"] == "#4C78A8"
    assert plotted_colors[r"SURD $S_{xy}$"] == "#8C8C8C"
    assert plotted_colors["SHAP interaction x:y->z"] == "#E68613"
    assert plotted_alphas[r"MLP+PEID $U_x$"] == 1.0
    assert plotted_alphas[r"Oracle+PEID $U_x$"] == 1.0
    assert plotted_alphas[r"MLP+PEID $S_{xy}$"] == 1.0
    assert plotted_alphas[r"Oracle+PEID $S_{xy}$"] == 1.0
    assert plotted_alphas["SHAP x->z"] < 1.0
    assert plotted_alphas["NG x->z"] < 1.0
    assert plotted_alphas["observational WMS"] < 1.0
    assert plotted_alphas[r"SURD $S_{xy}$"] < 1.0
    assert plotted_alphas["SHAP interaction x:y->z"] < 1.0
    assert "PCMCI w->z" not in plotted_labels
    assert "NG x->z" in plotted_labels
    assert "NG y->z" in plotted_labels
    assert "NG w->z" not in plotted_labels
    assert "NG x+y->z sum" not in plotted_labels
    assert r"Product probe $R^2$" not in plotted_labels
    assert 0.0 in horizontal_lines
    assert not bar_calls
    assert not errorbar_calls
    assert not fill_between_calls
