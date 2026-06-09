from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "apps" / "peid_hypergraph" / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from peid_hypergraph.compute import compute_boolean_peid, compute_continuous_peid, filter_visible_graph


def _edge_value(result: dict[str, object], source: str, target: str) -> float:
    for edge in result["pairwise_edges"]:
        if edge["source"] == source and edge["target"] == target:
            return float(edge["ei"])
    raise AssertionError(f"missing pairwise edge {source}->{target}")


def _hyperedge(result: dict[str, object], sources: tuple[str, ...], target: str) -> dict[str, object]:
    expected = set(sources)
    for edge in result["hyperedges"]:
        if set(edge["sources"]) == expected and edge["target"] == target:
            return edge
    raise AssertionError(f"missing hyperedge {sources}->{target}")


def test_boolean_xor_has_synergy_without_singleton_ei() -> None:
    result = compute_boolean_peid(
        {
            "variables": ["x0", "x1", "x2"],
            "update_rules": {
                "x0": {"type": "copy", "inputs": ["x0"]},
                "x1": {"type": "copy", "inputs": ["x1"]},
                "x2": {"type": "xor", "inputs": ["x0", "x1"]},
            },
            "noise": 0.0,
            "max_source_order": 2,
        }
    )

    assert _edge_value(result, "x0", "x2") < 1.0e-9
    assert _edge_value(result, "x1", "x2") < 1.0e-9
    xy_to_x2 = _hyperedge(result, ("x0", "x1"), "x2")
    assert float(xy_to_x2["joint_ei"]) == 1.0
    assert float(xy_to_x2["synergy"]) == 1.0


def test_boolean_copy_has_pairwise_edge_without_false_hyperedge() -> None:
    result = compute_boolean_peid(
        {
            "variables": ["x0", "x1", "x2"],
            "update_rules": {
                "x0": {"type": "copy", "inputs": ["x0"]},
                "x1": {"type": "copy", "inputs": ["x0"]},
                "x2": {"type": "copy", "inputs": ["x2"]},
            },
            "noise": 0.0,
            "max_source_order": 2,
        }
    )

    assert _edge_value(result, "x0", "x1") == 1.0
    assert all(
        not (set(edge["sources"]) == {"x0", "x2"} and edge["target"] == "x1")
        for edge in result["hyperedges"]
    )


def test_boolean_and_has_singleton_ei_and_positive_synergy() -> None:
    result = compute_boolean_peid(
        {
            "variables": ["x0", "x1", "x2"],
            "update_rules": {
                "x0": {"type": "copy", "inputs": ["x0"]},
                "x1": {"type": "copy", "inputs": ["x1"]},
                "x2": {"type": "and", "inputs": ["x0", "x1"]},
            },
            "noise": 0.0,
            "max_source_order": 2,
        }
    )

    assert _edge_value(result, "x0", "x2") > 0.30
    assert _edge_value(result, "x1", "x2") > 0.30
    xy_to_x2 = _hyperedge(result, ("x0", "x1"), "x2")
    assert float(xy_to_x2["joint_ei"]) > 0.80
    assert float(xy_to_x2["synergy"]) > 0.18


def test_continuous_sine_default_recovers_common_driver_and_synergy() -> None:
    result = compute_continuous_peid(
        {
            "example": "sine_common_driver",
            "alpha": 1.0,
            "beta": 0.75,
            "noise_std": 0.03,
            "intervention_samples": 900,
            "seed": 11,
            "max_source_order": 2,
        }
    )

    assert [node["id"] for node in result["nodes"]] == ["w", "x", "y", "z"]
    assert _edge_value(result, "w", "x") > 0.5
    assert _edge_value(result, "w", "y") > 0.5
    assert _edge_value(result, "w", "z") < 0.15
    xy_to_z = _hyperedge(result, ("x", "y"), "z")
    assert float(xy_to_z["synergy"]) > 0.1
    assert result["diagnostics"]["estimator"] == "transport_map"


def test_display_filter_changes_visible_edges_without_mutating_full_result() -> None:
    result = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "pairwise_edges": [
            {"source": "a", "target": "b", "ei": 0.9},
            {"source": "b", "target": "a", "ei": 0.1},
        ],
        "hyperedges": [
            {"sources": ["a", "b"], "target": "a", "display_value": 0.7},
            {"sources": ["a", "b"], "target": "b", "display_value": 0.2},
        ],
    }

    visible = filter_visible_graph(result, pairwise_top_percent=50, hyperedge_top_k=1)

    assert len(visible["pairwise_edges"]) == 1
    assert visible["pairwise_edges"][0]["ei"] == 0.9
    assert len(visible["hyperedges"]) == 1
    assert visible["hyperedges"][0]["display_value"] == 0.7
    assert len(result["pairwise_edges"]) == 2
    assert len(result["hyperedges"]) == 2


def test_api_examples_and_compute_routes() -> None:
    from fastapi.testclient import TestClient

    from peid_hypergraph.api import create_app

    client = TestClient(create_app())

    examples = client.get("/api/examples")
    assert examples.status_code == 200
    example_ids = {item["id"] for item in examples.json()["examples"]}
    assert {"boolean_xor", "continuous_sine_common_driver"}.issubset(example_ids)

    boolean_response = client.post(
        "/api/compute/boolean",
        json={
            "variables": ["x0", "x1", "x2"],
            "update_rules": {
                "x0": {"type": "copy", "inputs": ["x0"]},
                "x1": {"type": "copy", "inputs": ["x1"]},
                "x2": {"type": "xor", "inputs": ["x0", "x1"]},
            },
            "noise": 0.0,
            "max_source_order": 2,
        },
    )
    assert boolean_response.status_code == 200
    assert boolean_response.json()["hyperedges"][0]["sources"] == ["x0", "x1"]

    continuous_response = client.post(
        "/api/compute/continuous",
        json={
            "example": "sine_common_driver",
            "intervention_samples": 300,
            "seed": 3,
            "max_source_order": 2,
        },
    )
    assert continuous_response.status_code == 200
    assert continuous_response.json()["diagnostics"]["estimator"] == "transport_map"
