from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_granger_peid_mlp import (
    SimConfig,
    estimate_granger_graph,
    estimate_peid_graph,
    make_lagged_dataset,
    run_comparison_grid,
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


def test_smoke_grid_writes_summary_edges_and_png(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    figure_dir = tmp_path / "fig"

    output = run_comparison_grid(
        mode="smoke",
        mechanisms=("linear_additive", "xor_synergy"),
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
    assert any(run["mechanism"] == "xor_synergy" for run in summary["runs"])
    assert Path(summary["graph_figure_path"]).exists()
    assert Path(summary["graph_figure_path"]).name == "representative_causal_graphs.png"
    assert Path(summary["report_figure_path"]).exists()
    assert Path(summary["report_markdown_path"]).exists()
    report_text = Path(summary["report_markdown_path"]).read_text(encoding="utf-8")
    assert "MLP 学习下 Granger 与 PEID 因果图对照实验" in report_text
    assert "Ground truth 因果图" in report_text
    assert "MLP 学习情况" in report_text
    assert "time lag / Granger 识别的因果图" in report_text
    assert "PEID 识别的因果图" in report_text
    assert edge_path.read_text(encoding="utf-8").strip()
