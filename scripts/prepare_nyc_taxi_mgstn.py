#!/usr/bin/env python3
"""Build the paper-compatible hourly NYC Taxi dataset for MGSTN.

The cache contains both inclusive trip-end/trip-start flows and cross-zone-only
flows so the paper's undocumented flow convention can be audited against its
published HA baseline before model training.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "nyc_taxi_mgstn_2023"
MONTH_DIR = DATA_DIR / "monthly"
OUTPUT = DATA_DIR / "nyc_taxi_mgstn_hourly.npz"
PROGRESS = ROOT / "docs" / "log" / "nyc_taxi_mgstn" / "live_progress.json"
YEAR = 2023
INACTIVE_MANHATTAN_ZONES = {103, 104, 105}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def fetch(url: str, timeout: int = 300) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "EISyn-MGSTN-reproduction/1.0"})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:  # network retries are intentionally broad
            last_error = error
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {url}") from last_error


def load_zones() -> tuple[np.ndarray, list[str], np.ndarray]:
    lookup_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
    lookup = pd.read_csv(io.BytesIO(fetch(lookup_url)))
    selected = lookup[
        lookup["Borough"].eq("Manhattan")
        & ~lookup["LocationID"].isin(INACTIVE_MANHATTAN_ZONES)
    ].sort_values("LocationID")
    zone_ids = selected["LocationID"].to_numpy(dtype=np.int32)
    if len(zone_ids) != 66:
        raise RuntimeError(f"expected 66 active Manhattan zones, found {len(zone_ids)}")

    geojson_url = "https://data.cityofnewyork.us/api/v3/views/8meu-9t5y/query.geojson"
    geo = json.loads(fetch(geojson_url).decode("utf-8"))
    centroids: dict[int, tuple[float, float]] = {}
    for feature in geo["features"]:
        props = feature["properties"]
        location_id = int(props.get("locationid", props.get("LocationID", -1)))
        if location_id not in set(zone_ids.tolist()):
            continue
        geometry = feature["geometry"]
        coordinates = geometry["coordinates"]
        rings = coordinates if geometry["type"] == "Polygon" else [poly for poly in coordinates]
        points: list[tuple[float, float]] = []
        for polygon in rings:
            outer = polygon[0]
            points.extend((float(p[0]), float(p[1])) for p in outer)
        centroids[location_id] = (
            float(np.mean([p[1] for p in points])),
            float(np.mean([p[0] for p in points])),
        )
    if len(centroids) != 66:
        missing = sorted(set(zone_ids.tolist()) - set(centroids))
        raise RuntimeError(f"missing taxi-zone geometry for {missing}")
    centroid_array = np.asarray([centroids[int(zone)] for zone in zone_ids], dtype=np.float32)
    return zone_ids, selected["Zone"].astype(str).tolist(), centroid_array


def aggregate_month(month: int, zone_ids: np.ndarray) -> dict[str, np.ndarray | int]:
    cache = MONTH_DIR / f"month_{month:02d}.npz"
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        if "internal_cross_zone" in saved.files:
            return {key: saved[key] for key in saved.files}

    url = (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/"
        f"yellow_tripdata_{YEAR}-{month:02d}.parquet"
    )
    payload = fetch(url)
    table = pq.read_table(
        io.BytesIO(payload),
        columns=["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"],
    )
    frame = table.to_pandas()
    start = pd.Timestamp(f"{YEAR}-01-01")
    stop = pd.Timestamp(f"{YEAR + 1}-01-01")
    zone_to_col = {int(zone): i for i, zone in enumerate(zone_ids)}
    shape = (8760, len(zone_ids), 2)
    inclusive = np.zeros(shape, dtype=np.int32)
    cross_zone = np.zeros(shape, dtype=np.int32)
    internal_cross_zone = np.zeros(shape, dtype=np.int32)

    pickup_time = pd.to_datetime(frame["tpep_pickup_datetime"], errors="coerce")
    dropoff_time = pd.to_datetime(frame["tpep_dropoff_datetime"], errors="coerce")
    pickup_zone = pd.to_numeric(frame["PULocationID"], errors="coerce")
    dropoff_zone = pd.to_numeric(frame["DOLocationID"], errors="coerce")
    different = pickup_zone.ne(dropoff_zone).to_numpy()
    both_selected = (pickup_zone.isin(zone_ids) & dropoff_zone.isin(zone_ids)).to_numpy()

    def add_events(times: pd.Series, zones: pd.Series, channel: int, target: np.ndarray, mask_extra=None):
        valid = times.ge(start) & times.lt(stop) & zones.isin(zone_ids)
        if mask_extra is not None:
            valid &= mask_extra
        bins = ((times.loc[valid] - start).dt.total_seconds() // 3600).astype(np.int64).to_numpy()
        cols = zones.loc[valid].astype(int).map(zone_to_col).to_numpy(dtype=np.int64)
        np.add.at(target, (bins, cols, np.full(len(bins), channel, dtype=np.int64)), 1)
        return int(valid.sum())

    # Paper convention: channel 0 = inflow at dropoff time; channel 1 = outflow at pickup time.
    inclusive_in = add_events(dropoff_time, dropoff_zone, 0, inclusive)
    inclusive_out = add_events(pickup_time, pickup_zone, 1, inclusive)
    cross_in = add_events(dropoff_time, dropoff_zone, 0, cross_zone, different)
    cross_out = add_events(pickup_time, pickup_zone, 1, cross_zone, different)
    internal_mask = different & both_selected
    internal_in = add_events(dropoff_time, dropoff_zone, 0, internal_cross_zone, internal_mask)
    internal_out = add_events(pickup_time, pickup_zone, 1, internal_cross_zone, internal_mask)
    MONTH_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        inclusive=inclusive,
        cross_zone=cross_zone,
        internal_cross_zone=internal_cross_zone,
        raw_rows=np.asarray(len(frame)),
        inclusive_in=np.asarray(inclusive_in),
        inclusive_out=np.asarray(inclusive_out),
        cross_in=np.asarray(cross_in),
        cross_out=np.asarray(cross_out),
        internal_in=np.asarray(internal_in),
        internal_out=np.asarray(internal_out),
    )
    return {
        "inclusive": inclusive,
        "cross_zone": cross_zone,
        "internal_cross_zone": internal_cross_zone,
        "raw_rows": len(frame),
        "inclusive_in": inclusive_in,
        "inclusive_out": inclusive_out,
        "cross_in": cross_in,
        "cross_out": cross_out,
        "internal_in": internal_in,
        "internal_out": internal_out,
    }


def weather_features() -> tuple[np.ndarray, list[str]]:
    cache = DATA_DIR / "weather_hourly.npz"
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        return saved["features"], saved["names"].astype(str).tolist()
    params = {
        "latitude": 40.7769,
        "longitude": -73.8740,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "hourly": "temperature_2m,precipitation,weather_code",
        "timezone": "America/New_York",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    response = json.loads(fetch(url).decode("utf-8"))["hourly"]
    frame = pd.DataFrame(response)
    timestamp = pd.to_datetime(frame["time"])
    target = pd.date_range("2023-01-01", "2024-01-01", freq="h", inclusive="left")
    frame.index = timestamp
    frame = frame.drop(columns=["time"])
    frame = frame.reindex(target).interpolate(limit_direction="both")
    code = frame["weather_code"].fillna(0).to_numpy(dtype=int)
    flags = np.column_stack(
        [
            code == 0,
            np.isin(code, [1, 2, 3]),
            np.isin(code, [45, 48]),
            np.isin(code, [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]),
            np.isin(code, [71, 73, 75, 77, 85, 86]),
            np.isin(code, [95, 96, 99]),
        ]
    ).astype(np.float32)
    names = ["temperature_2m", "precipitation", "clear", "cloud", "fog", "rain", "snow", "storm"]
    features = np.column_stack(
        [
            frame["temperature_2m"].to_numpy(dtype=np.float32),
            frame["precipitation"].to_numpy(dtype=np.float32),
            flags,
        ]
    ).astype(np.float32)
    np.savez_compressed(cache, features=features, names=np.asarray(names))
    return features, names


def date_features() -> tuple[np.ndarray, list[str]]:
    times = pd.date_range("2023-01-01", "2024-01-01", freq="h", inclusive="left")
    federal_holidays = {
        "2023-01-02", "2023-01-16", "2023-02-20", "2023-05-29", "2023-06-19",
        "2023-07-04", "2023-09-04", "2023-10-09", "2023-11-10", "2023-11-23", "2023-12-25",
    }
    holiday = np.isin(times.strftime("%Y-%m-%d"), sorted(federal_holidays))
    workday = ((times.dayofweek < 5) & ~holiday).astype(np.float32)
    hour = np.eye(24, dtype=np.float32)[times.hour]
    weekday = np.eye(7, dtype=np.float32)[times.dayofweek]
    names = ["workday"] + [f"hour_{i}" for i in range(24)] + [f"weekday_{i}" for i in range(7)]
    return np.column_stack([workday, hour, weekday]).astype(np.float32), names


def haversine_matrix(centroids: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(centroids[:, 0])[:, None]
    lon = np.deg2rad(centroids[:, 1])[:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    value = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    return (2 * 6371.0 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))).astype(np.float32)


def behavior_semantic_types(flow: np.ndarray, n_types: int = 9) -> np.ndarray:
    """Deterministic fallback for the unavailable Foursquare check-in archive.

    It retains the paper's typed semantic graph, but types summarize train-period
    land-use rhythm (hour x weekday x in/out) rather than proprietary POI counts.
    """
    train_end = int(0.70 * len(flow))
    times = pd.date_range("2023-01-01", periods=train_end, freq="h")
    profiles = []
    for zone in range(flow.shape[1]):
        pieces = []
        for channel in range(2):
            values = flow[:train_end, zone, channel]
            hourly = np.asarray([values[times.hour == h].mean() for h in range(24)])
            weekday = np.asarray([values[times.dayofweek == d].mean() for d in range(7)])
            pieces.extend([hourly / (hourly.mean() + 1e-6), weekday / (weekday.mean() + 1e-6)])
        profiles.append(np.concatenate(pieces))
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=n_types, n_init=20, random_state=0).fit_predict(np.asarray(profiles)).astype(np.int64)


def semantic_adjacency(flow: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    train = flow[: int(0.70 * len(flow))].transpose(1, 0, 2).reshape(flow.shape[1], -1)
    corr = np.nan_to_num(np.corrcoef(train), nan=0.0)
    n = len(corr)
    adjacency = np.eye(n, dtype=np.float32)
    # Controlled-consensus approximation: within each node's strongest 5 links,
    # retain correlations above lambda times the two nodes' mean positive similarity.
    positive = np.maximum(corr, 0.0)
    means = (positive.sum(axis=1) - 1.0) / max(n - 1, 1)
    for i in range(n):
        candidates = np.argsort(corr[i])[::-1]
        retained = [j for j in candidates if j != i][:5]
        for j in retained:
            if corr[i, j] > threshold * min(means[i], means[j]):
                adjacency[i, j] = adjacency[j, i] = 1.0
    return adjacency


def historical_average_metrics(flow: np.ndarray) -> dict[str, dict[str, float]]:
    start = 7 * 168
    indices = np.arange(start, len(flow))
    n_train = int(0.70 * len(indices))
    n_valid = int(0.20 * len(indices))
    test = indices[n_train + n_valid :]
    train = indices[:n_train]
    prediction = np.broadcast_to(flow[train].mean(axis=0), flow[test].shape)
    result = {}
    for channel, name in enumerate(["inflow", "outflow"]):
        error = prediction[:, :, channel] - flow[test, :, channel]
        result[name] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.force:
        print(f"cache already exists: {OUTPUT}")
        return

    started = time.monotonic()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zone_ids, zone_names, centroids = load_zones()
    inclusive = np.zeros((8760, 66, 2), dtype=np.int32)
    cross_zone = np.zeros_like(inclusive)
    internal_cross_zone = np.zeros_like(inclusive)
    raw_rows = 0
    monthly_audit = []
    for month in range(1, 13):
        result = aggregate_month(month, zone_ids)
        inclusive += result["inclusive"]
        cross_zone += result["cross_zone"]
        internal_cross_zone += result["internal_cross_zone"]
        raw_rows += int(result["raw_rows"])
        monthly_audit.append({key: int(result[key]) for key in ["raw_rows", "inclusive_in", "inclusive_out", "cross_in", "cross_out", "internal_in", "internal_out"]})
        atomic_json(
            PROGRESS,
            {
                "phase": "aggregate_taxi",
                "current": month,
                "total": 12,
                "unit": "month",
                "elapsed_seconds": time.monotonic() - started,
                "metrics": {"raw_rows": raw_rows},
                "updated_at": time.time(),
            },
        )
        print(f"month {month:02d}/12 complete; raw rows={raw_rows:,}", flush=True)

    weather, weather_names = weather_features()
    dates, date_names = date_features()
    distance = haversine_matrix(centroids)
    normalized_distance = distance / max(float(distance.max()), 1e-6)
    distance_adjacency = ((normalized_distance <= 0.2) & (normalized_distance > 0)).astype(np.float32)
    np.fill_diagonal(distance_adjacency, 1.0)

    # Compare both plausible definitions with the paper's published HA row.
    ha_inclusive = historical_average_metrics(inclusive)
    ha_cross = historical_average_metrics(cross_zone)
    ha_internal = historical_average_metrics(internal_cross_zone)
    paper_ha = {"inflow": {"mae": 31.133, "rmse": 49.703}, "outflow": {"mae": 32.719, "rmse": 56.265}}
    def distance_to_paper(metrics):
        return sum(abs(metrics[c][m] / paper_ha[c][m] - 1.0) for c in paper_ha for m in paper_ha[c])
    candidates = {
        "inclusive": (inclusive, ha_inclusive),
        "cross_zone": (cross_zone, ha_cross),
        "internal_cross_zone": (internal_cross_zone, ha_internal),
    }
    flow_definition = min(candidates, key=lambda name: distance_to_paper(candidates[name][1]))
    selected_flow = candidates[flow_definition][0]
    semantic_types = behavior_semantic_types(selected_flow)
    semantic_adj = semantic_adjacency(selected_flow)

    metadata = {
        "year": YEAR,
        "zones": 66,
        "timestamps": 8760,
        "channels": ["inflow_dropoff", "outflow_pickup"],
        "excluded_zones": sorted(INACTIVE_MANHATTAN_ZONES),
        "raw_rows": raw_rows,
        "flow_definition_selected_by_paper_HA_match": flow_definition,
        "ha_inclusive": ha_inclusive,
        "ha_cross_zone": ha_cross,
        "ha_internal_cross_zone": ha_internal,
        "paper_ha": paper_ha,
        "semantic_type_source": "train-period mobility-rhythm clustering; Foursquare archive unavailable",
        "weather_source": "Open-Meteo archive at KLGA coordinates",
        "monthly_audit": monthly_audit,
    }
    np.savez_compressed(
        OUTPUT,
        flow=selected_flow,
        flow_inclusive=inclusive,
        flow_cross_zone=cross_zone,
        flow_internal_cross_zone=internal_cross_zone,
        zone_ids=zone_ids,
        zone_names=np.asarray(zone_names),
        centroids=centroids,
        distance_adjacency=distance_adjacency,
        semantic_adjacency=semantic_adj,
        semantic_types=semantic_types,
        weather=weather,
        weather_names=np.asarray(weather_names),
        date_features=dates,
        date_names=np.asarray(date_names),
        metadata=np.asarray(json.dumps(metadata)),
    )
    atomic_json(
        PROGRESS,
        {
            "phase": "complete",
            "current": 12,
            "total": 12,
            "unit": "month",
            "elapsed_seconds": time.monotonic() - started,
            "metrics": {"raw_rows": raw_rows, "flow_definition": flow_definition},
            "updated_at": time.time(),
        },
    )
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
