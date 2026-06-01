from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_granger_peid_mlp import (
    SimConfig,
    estimate_granger_graph,
    estimate_peid_graph,
    make_lagged_dataset,
    run_comparison_grid,
    run_lagged_proxy_common_driver_experiment,
    run_lag_sensitivity_lagged_proxy_experiment,
    run_neural_granger_lagged_proxy_experiment,
    simulate_system,
    train_mlp_transition_model,
)


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

    assert {("w", "x"), ("w", "y")} == set(truth["pairwise_edges"])
    assert ("x", "y", "z") in truth["hyperedges"]
    assert float(granger_w_to_x["score"]) > 0.2
    assert float(granger_w_to_y["score"]) > 0.2
    assert float(granger_w_to_z["score"]) < 0.05
    assert float(peid_w_to_z["ei"]) < 0.08
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
    assert Path(summary["report_markdown_path"]).exists()
    report_text = Path(summary["report_markdown_path"]).read_text(encoding="utf-8")
    assert "统一动力系统：共同驱动 + sine 协同" in report_text
    assert "Granger/ablation `w -> x`" in report_text
    assert "PEID synergy `{x, y} -> z`" in report_text
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
