#!/usr/bin/env python3
"""Map finite quadratic-TM recent–macro synergy across Manhattan Taxi Zones."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/nyc_taxi_mgstn_ei/finite_quadratic_tm/quadratic_tm_full_summary.json"
GEOJSON = ROOT / "data/nyc_taxi_mgstn_2023/taxi_zones.geojson"
OUTPUT = ROOT / "fig/nyc_taxi_temporal_coupling_map"
GEOJSON_URL = "https://data.cityofnewyork.us/api/v3/views/8meu-9t5y/query.geojson"

STATES = ["weekday_peak", "weekend_midday", "rainy_high_demand"]
STATE_LABELS = ["Weekday peak", "Weekend midday", "Rainy high-demand"]
PANEL_LABELS = ["a", "b", "c"]
NONNEGATIVE_TOLERANCE_BITS = 0.05


def load_geojson() -> dict:
    if GEOJSON.exists():
        return json.loads(GEOJSON.read_text(encoding="utf-8"))
    request = urllib.request.Request(GEOJSON_URL, headers={"User-Agent": "EISyn-map/1.0"})
    last_error = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            geojson = json.loads(payload)
            if len(geojson.get("features", [])) < 250:
                raise RuntimeError("incomplete NYC Taxi Zone GeoJSON")
            GEOJSON.parent.mkdir(parents=True, exist_ok=True)
            GEOJSON.write_bytes(payload)
            return geojson
        except Exception as error:
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to download Taxi Zone GeoJSON: {last_error}")


def outer_rings(feature: dict) -> list[np.ndarray]:
    geometry = feature["geometry"]
    coordinates = geometry["coordinates"]
    polygons = [coordinates] if geometry["type"] == "Polygon" else coordinates
    return [np.asarray(polygon[0], dtype=float) for polygon in polygons if polygon and polygon[0]]


def zone_id(feature: dict) -> int:
    properties = feature["properties"]
    return int(properties.get("locationid", properties.get("LocationID", -1)))


def main() -> None:
    saved = json.loads(INPUT.read_text(encoding="utf-8"))
    if saved["nonnegative_audit"]["hurdle"]["violation_count"]:
        raise RuntimeError("formal hurdle quadratic TM failed its declared nonnegative audit")
    geojson = load_geojson()
    selected_ids = np.asarray(sorted({int(unit["zone_id"]) for unit in saved["units"]}), dtype=int)
    if len(selected_ids) != 66:
        raise RuntimeError(f"expected 66 Taxi Zones, found {len(selected_ids)}")
    values_by_state: list[np.ndarray] = []
    for state in STATES:
        rows = [unit for unit in saved["units"] if unit["state"] == state]
        seeds = sorted({int(unit["model_seed"]) for unit in rows})
        if len(rows) != 66 * 3 or len(seeds) != 3:
            raise RuntimeError(f"expected 198 zone-seed units for {state}, found {len(rows)}")
        by_key = {(int(unit["model_seed"]), int(unit["zone_id"])): unit for unit in rows}
        state_values = []
        for location_id in selected_ids:
            estimates = []
            for seed in seeds:
                raw = float(by_key[seed, int(location_id)]["hurdle_temporal"]["syn_bits_raw"])
                estimates.append(0.0 if -NONNEGATIVE_TOLERANCE_BITS <= raw < 0 else raw)
            state_values.append(float(np.mean(estimates)))
        values_by_state.append(np.asarray(state_values))
    values_by_state = np.asarray(values_by_state)

    features = geojson["features"]
    feature_by_id = {zone_id(feature): feature for feature in features}
    missing = sorted(set(selected_ids.tolist()) - set(feature_by_id))
    if missing:
        raise RuntimeError(f"missing Taxi Zone polygons: {missing}")

    selected_rings = [ring for location_id in selected_ids for ring in outer_rings(feature_by_id[location_id])]
    points = np.vstack(selected_rings)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_pad = 0.055 * (x_max - x_min)
    y_pad = 0.025 * (y_max - y_min)

    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    cmap = mpl.colormaps["magma"]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=float(values_by_state.max()))
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 7.0), constrained_layout=True)

    for panel_index, (ax, state_label, values) in enumerate(zip(axes, STATE_LABELS, values_by_state)):
        background_patches = []
        for feature in features:
            background_patches.extend(Polygon(ring, closed=True) for ring in outer_rings(feature))
        background = PatchCollection(
            background_patches, facecolor="#E7EBED", edgecolor="white", linewidth=0.25, zorder=0
        )
        ax.add_collection(background)

        foreground_patches = []
        foreground_values = []
        value_lookup = dict(zip(selected_ids.tolist(), values.tolist()))
        for location_id in selected_ids:
            rings = outer_rings(feature_by_id[int(location_id)])
            foreground_patches.extend(Polygon(ring, closed=True) for ring in rings)
            foreground_values.extend([value_lookup[int(location_id)]] * len(rings))
        foreground = PatchCollection(
            foreground_patches,
            cmap=cmap,
            norm=norm,
            edgecolor="#20282C",
            linewidth=0.48,
            zorder=1,
        )
        foreground.set_array(np.asarray(foreground_values))
        ax.add_collection(foreground)

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.text(0.5, 1.01, state_label, transform=ax.transAxes, ha="center", va="bottom", fontsize=11)
        ax.text(-0.02, 1.01, PANEL_LABELS[panel_index], transform=ax.transAxes,
                ha="right", va="bottom", fontsize=13, fontweight="bold")

    scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar_mappable, ax=axes, fraction=0.028, pad=0.018)
    colorbar.set_label("Finite recent–macro synergy (bits)")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(OUTPUT.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
