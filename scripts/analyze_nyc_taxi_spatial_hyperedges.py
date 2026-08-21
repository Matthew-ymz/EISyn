#!/usr/bin/env python3
"""Network and sparsity diagnostics for confirmed NYC Taxi hyperedges."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

from plot_nyc_taxi_spatial_hyperedges import feature_centroid
from plot_nyc_taxi_temporal_coupling_map import load_geojson, zone_id


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/nyc_taxi_mgstn_ei/spatial_hyperedges/spatial_hyperedge_full_summary.json"
DATA = ROOT / "data/nyc_taxi_mgstn_2023/nyc_taxi_mgstn_hourly.npz"
OUTPUT = ROOT / "results/nyc_taxi_mgstn_ei/spatial_hyperedges/spatial_hyperedge_network_analysis.json"


def haversine_km(left: np.ndarray, right: np.ndarray) -> float:
    lon1, lat1 = np.radians(left)
    lon2, lat2 = np.radians(right)
    dlon, dlat = lon2 - lon1, lat2 - lat1
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(6371.0088 * 2 * np.arcsin(np.sqrt(value)))


def ranked(mapping: dict[int, float], names: dict[int, str], maximum: int = 8) -> list[dict]:
    return [
        {"zone_id": int(key), "zone_name": names[int(key)], "value": float(value)}
        for key, value in sorted(mapping.items(), key=lambda item: item[1], reverse=True)[:maximum]
    ]


def main() -> None:
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    if payload["nonnegative_audit"]["violation_count"]:
        raise RuntimeError("cannot analyze a spatial Syn result that failed nonnegativity")
    saved = np.load(DATA, allow_pickle=False)
    zone_ids = saved["zone_ids"].astype(int)
    zone_names = saved["zone_names"].astype(str)
    names = dict(zip(zone_ids.tolist(), zone_names.tolist()))
    flow = saved["flow"].astype(float)
    zone_index = {int(location_id): index for index, location_id in enumerate(zone_ids)}
    activity = {}
    for location_id, index in zone_index.items():
        activity[location_id] = {
            "mean_inflow": float(flow[:, index, 0].mean()),
            "mean_outflow": float(flow[:, index, 1].mean()),
            "zero_inflow_fraction": float(np.mean(flow[:, index, 0] == 0)),
            "zero_outflow_fraction": float(np.mean(flow[:, index, 1] == 0)),
        }

    features = load_geojson()["features"]
    centroids = {
        zone_id(feature): feature_centroid(feature)
        for feature in features if zone_id(feature) in zone_index
    }
    candidates = payload["candidates"]
    confirmed = [row for row in candidates if row["confirmed"]]
    for row in candidates:
        source_a = int(row["source_a_id"])
        source_b = int(row["source_b_id"])
        target = int(row["target_id"])
        row["source_pair_distance_km"] = haversine_km(centroids[source_a], centroids[source_b])
        row["mean_source_target_distance_km"] = float(np.mean([
            haversine_km(centroids[source_a], centroids[target]),
            haversine_km(centroids[source_b], centroids[target]),
        ]))
        row["minimum_source_mean_outflow"] = min(
            activity[source_a]["mean_outflow"], activity[source_b]["mean_outflow"]
        )
        row["maximum_source_zero_fraction"] = max(
            activity[source_a]["zero_outflow_fraction"], activity[source_b]["zero_outflow_fraction"]
        )

    deltas = np.asarray([row["paired_delta_mean_bits"] for row in candidates])
    screens = np.asarray([row["screen_interaction_rms_z"] for row in candidates])
    minimum_activity = np.asarray([row["minimum_source_mean_outflow"] for row in candidates])
    maximum_zero = np.asarray([row["maximum_source_zero_fraction"] for row in candidates])
    screen_corr = spearmanr(screens, deltas)
    activity_corr = spearmanr(minimum_activity, deltas)
    zero_corr = spearmanr(maximum_zero, deltas)

    graph = nx.DiGraph()
    source_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    source_strength: Counter[int] = Counter()
    target_strength: Counter[int] = Counter()
    for row in confirmed:
        strength = float(row["paired_delta_mean_bits"])
        target = int(row["target_id"])
        target_counts[target] += 1
        target_strength[target] += strength
        for source_key in ("source_a_id", "source_b_id"):
            source = int(row[source_key])
            source_counts[source] += 1
            source_strength[source] += strength
            previous = graph.get_edge_data(source, target, {}).get("weight", 0.0)
            graph.add_edge(source, target, weight=previous + 0.5 * strength)
    communities = []
    if graph.number_of_edges():
        undirected = graph.to_undirected()
        communities = [
            [names[int(node)] for node in sorted(group)]
            for group in nx.community.greedy_modularity_communities(undirected, weight="weight")
        ]

    state_means = {
        state: float(np.mean([row["state_mean_deltas_bits"][state] for row in candidates]))
        for state in sorted(candidates[0]["state_mean_deltas_bits"]) if candidates
    }
    result = {
        "status": "complete",
        "candidate_count": len(candidates),
        "confirmed_count": len(confirmed),
        "correlations": {
            "screen_vs_formal_delta_spearman_r": float(screen_corr.statistic),
            "screen_vs_formal_delta_p": float(screen_corr.pvalue),
            "minimum_source_activity_vs_delta_spearman_r": float(activity_corr.statistic),
            "minimum_source_activity_vs_delta_p": float(activity_corr.pvalue),
            "maximum_source_zero_fraction_vs_delta_spearman_r": float(zero_corr.statistic),
            "maximum_source_zero_fraction_vs_delta_p": float(zero_corr.pvalue),
        },
        "all_candidate_state_mean_deltas_bits": state_means,
        "network": {
            "nodes": graph.number_of_nodes(),
            "projected_directed_edges": graph.number_of_edges(),
            "weakly_connected_components": nx.number_weakly_connected_components(graph) if graph else 0,
            "top_source_participation": ranked(dict(source_counts), names),
            "top_target_participation": ranked(dict(target_counts), names),
            "top_source_strength": ranked(dict(source_strength), names),
            "top_target_strength": ranked(dict(target_strength), names),
            "communities": communities,
        },
        "activity_by_zone": {str(key): value for key, value in activity.items()},
        "confirmed_hyperedges": confirmed,
        "top_exploratory_hyperedges": sorted(
            candidates, key=lambda row: row["paired_delta_mean_bits"], reverse=True
        )[:10],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("confirmed_count", "correlations", "network")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
