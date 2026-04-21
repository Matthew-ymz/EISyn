from __future__ import annotations

import math

import pandas as pd


def build_city_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    grouped = frame.groupby("city_en")["station_id"].apply(list)
    return grouped.to_dict()


def _haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_km = 6371.0
    lon1_r, lat1_r, lon2_r, lat2_r = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def build_nearest_neighbor_groups(frame: pd.DataFrame, *, k: int = 3) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for _, row in frame.iterrows():
        distances = []
        for _, other in frame.iterrows():
            if row["station_id"] == other["station_id"]:
                continue
            distance = _haversine(row["lon"], row["lat"], other["lon"], other["lat"])
            distances.append((distance, other["station_id"]))
        distances.sort(key=lambda item: item[0])
        groups[row["station_id"]] = [station_id for _, station_id in distances[:k]]
    return groups
