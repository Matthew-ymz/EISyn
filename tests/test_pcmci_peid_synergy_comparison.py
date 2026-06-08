from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_pcmci_peid_synergy import (
    BenchmarkConfig,
    adjust_fdr_bh,
    estimate_tm_peid_graph,
    fit_transition_mlp,
    load_cached_json,
    oracle_transition,
    run_benchmark,
    run_pcmci_variant,
    save_cached_json,
    score_typed_graph,
    select_significant_relations,
    simulate_known_synergy_system,
)


def _small_config(**overrides: object) -> BenchmarkConfig:
    config = BenchmarkConfig(
        n_samples=900,
        burn_in=150,
        seed=7,
        mlp_hidden_dims=(48, 48),
        mlp_epochs=120,
        mlp_patience=18,
        intervention_samples=1200,
        intervention_batches=2,
        peid_permutations=12,
        pcmci_cmiknn_sig_samples=12,
    )
    return replace(config, **overrides)


def test_simulator_exports_exact_cross_variable_typed_truth() -> None:
    _, truth = simulate_known_synergy_system(_small_config(n_samples=300))

    assert set(truth["pairwise"]) == {("c", "x"), ("c", "y"), ("x", "p")}
    assert set(truth["hyperedges"]) == {("x", "y", "s")}
    assert set(truth["self_edges"]) == {
        ("c", "c"),
        ("x", "x"),
        ("y", "y"),
        ("p", "p"),
        ("s", "s"),
    }


def test_fdr_and_typed_scoring_distinguish_pairwise_from_hyperedges() -> None:
    q_values = adjust_fdr_bh([0.001, 0.02, 0.8])
    assert np.allclose(q_values, [0.003, 0.03, 0.8])

    truth = {
        "pairwise": [("c", "x"), ("c", "y"), ("x", "p")],
        "hyperedges": [("x", "y", "s")],
        "self_edges": [],
    }
    predicted = [
        {"relation_type": "pairwise", "sources": "c", "target": "x"},
        {"relation_type": "pairwise", "sources": "c", "target": "y"},
        {"relation_type": "pairwise", "sources": "x", "target": "p"},
        {"relation_type": "pairwise", "sources": "x", "target": "s"},
        {"relation_type": "pairwise", "sources": "y", "target": "s"},
    ]

    scores = score_typed_graph(predicted, truth)

    assert scores["pairwise_recall"] == 1.0
    assert scores["hyperedge_recall"] == 0.0
    assert scores["typed_f1"] < 1.0


def test_oracle_peid_separates_additive_and_synergy_targets() -> None:
    config = _small_config(intervention_samples=2400, intervention_batches=3, peid_permutations=0)
    series, _ = simulate_known_synergy_system(config)

    graph = estimate_tm_peid_graph(
        oracle_transition,
        series,
        config,
        method="Oracle + PEID",
        permutations=0,
    )
    xy = graph["hyperedges"].set_index(["sources", "target"])

    syn_to_s = float(xy.loc[("x+y", "s"), "score"])
    syn_to_p = float(xy.loc[("x+y", "p"), "score"])
    assert syn_to_s > 0.5
    assert syn_to_s > 10.0 * max(abs(syn_to_p), 1.0e-6)


def test_pcmci_parcorr_recovers_pairwise_edges_and_has_no_hyperedges() -> None:
    config = _small_config(n_samples=1200)
    series, _ = simulate_known_synergy_system(config)

    graph = run_pcmci_variant(series, config, variant="parcorr")
    significant = graph[graph["q_value"] <= config.q_threshold]
    predicted = set(zip(significant["sources"], significant["target"]))

    assert {("c", "x"), ("c", "y"), ("x", "p")}.issubset(predicted)
    assert set(graph["relation_type"]) == {"pairwise"}


def test_pcmci_cmiknn_projects_true_hyperedge_to_two_pairwise_dependencies() -> None:
    config = _small_config(n_samples=700, seed=0, pcmci_cmiknn_sig_samples=20)
    series, _ = simulate_known_synergy_system(config)

    graph = run_pcmci_variant(series, config, variant="cmiknn")
    raw_significant = graph[graph["p_value"] <= 0.05]
    predicted = set(zip(raw_significant["sources"], raw_significant["target"]))

    assert {("x", "s"), ("y", "s")}.issubset(predicted)
    assert set(graph["relation_type"]) == {"pairwise"}


def test_mlp_beats_persistence_and_peid_ranks_true_hyperedge_first() -> None:
    config = _small_config(n_samples=1400, intervention_samples=2200, intervention_batches=3)
    series, _ = simulate_known_synergy_system(config)

    fitted = fit_transition_mlp(series, config)
    graph = estimate_tm_peid_graph(
        fitted.predict,
        series.iloc[: fitted.train_end + 1],
        config,
        method="MLP + PEID",
        permutations=0,
    )

    assert fitted.metrics["p_test_rmse"] < fitted.metrics["p_persistence_rmse"]
    assert fitted.metrics["s_test_rmse"] < fitted.metrics["s_persistence_rmse"]
    cross_target = graph["hyperedges"][
        graph["hyperedges"].apply(lambda row: str(row["target"]) not in str(row["sources"]).split("+"), axis=1)
    ]
    top = cross_target.sort_values("score", ascending=False).iloc[0]
    assert (str(top["sources"]), str(top["target"])) == ("x+y", "s")


def test_json_cache_round_trip_uses_config_hash(tmp_path: Path) -> None:
    config = _small_config(n_samples=321)
    path = tmp_path / "cache.json"
    payload = {"value": 7, "nested": {"ok": True}}

    save_cached_json(path, payload, config)
    loaded = load_cached_json(path, config)

    assert loaded == payload
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["config_hash"] == config.config_hash()
    assert load_cached_json(path, replace(config, n_samples=322)) is None


def test_selection_excludes_self_and_target_containing_hyperedges() -> None:
    relations = [
        {
            "method": "MLP + PEID",
            "relation_type": "pairwise",
            "sources": "x",
            "target": "p",
            "score": 0.8,
            "q_value": 0.01,
        },
        {
            "method": "MLP + PEID",
            "relation_type": "pairwise",
            "sources": "x",
            "target": "x",
            "score": 0.9,
            "q_value": 0.01,
        },
        {
            "method": "MLP + PEID",
            "relation_type": "hyperedge",
            "sources": "x+y",
            "target": "s",
            "score": 0.7,
            "q_value": 0.01,
        },
        {
            "method": "MLP + PEID",
            "relation_type": "hyperedge",
            "sources": "x+p",
            "target": "p",
            "score": 0.6,
            "q_value": 0.01,
        },
    ]

    selected = select_significant_relations(relations, _small_config())
    keys = {(row["relation_type"], row["sources"], row["target"]) for row in selected}

    assert keys == {("pairwise", "x", "p"), ("hyperedge", "x+y", "s")}


def test_smoke_benchmark_writes_summary_and_png_figures(tmp_path: Path) -> None:
    config = _small_config(
        n_samples=360,
        burn_in=80,
        mlp_epochs=30,
        mlp_patience=6,
        intervention_samples=320,
        intervention_batches=1,
        peid_permutations=3,
        pcmci_cmiknn_sig_samples=3,
    )

    output = run_benchmark(
        mode="smoke",
        base_config=config,
        primary_seeds=(0,),
        sample_sizes=(),
        noise_values=(),
        result_dir=tmp_path / "results",
        figure_dir=tmp_path / "figures",
        force=True,
    )

    summary = json.loads(Path(output["summary_path"]).read_text(encoding="utf-8"))
    assert {row["method"] for row in summary["primary_metrics"]} == {
        "PCMCI-ParCorr",
        "PCMCI-CMIknn",
        "MLP + PEID",
        "Oracle + PEID",
    }
    for path in output["figure_paths"].values():
        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000
