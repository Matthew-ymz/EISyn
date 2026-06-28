#!/usr/bin/env python3
"""Audit classic climate-mode coverage in Runge component rankings."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class RegionBox:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


@dataclass(frozen=True)
class ModeRegion:
    name: str
    boxes: tuple[RegionBox, ...]


DEFAULT_MODE_REGIONS: tuple[ModeRegion, ...] = (
    ModeRegion("ENSO", (RegionBox(-170.0, -80.0, -10.0, 10.0),)),
    ModeRegion("IOD", (RegionBox(40.0, 110.0, -12.0, 12.0),)),
    ModeRegion("Maritime_Continent", (RegionBox(90.0, 150.0, -15.0, 15.0),)),
    ModeRegion("Arctic", (RegionBox(-180.0, 180.0, 60.0, 90.0),)),
    ModeRegion("Tibetan_Plateau", (RegionBox(70.0, 105.0, 25.0, 40.0),)),
    ModeRegion("SPMM", (RegionBox(-120.0, -80.0, -30.0, -10.0),)),
    ModeRegion("NPMM", (RegionBox(-170.0, -110.0, 5.0, 30.0),)),
    ModeRegion("Tropical_Atlantic", (RegionBox(-60.0, -15.0, -5.0, 25.0),)),
    ModeRegion("South_Indian_Ocean_Dipole", (RegionBox(50.0, 100.0, -45.0, -15.0),)),
    ModeRegion("North_Atlantic_NAO", (RegionBox(-80.0, 40.0, 35.0, 75.0),)),
)


def _normalise_lon(lon: np.ndarray) -> np.ndarray:
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def _box_mask(lat: np.ndarray, lon: np.ndarray, box: RegionBox) -> np.ndarray:
    lat_mask = (lat >= float(box.lat_min)) & (lat <= float(box.lat_max))
    lon_min = float(box.lon_min)
    lon_max = float(box.lon_max)
    if lon_min <= -180.0 and lon_max >= 180.0:
        lon_mask = np.ones_like(lon, dtype=bool)
    elif lon_min <= lon_max:
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
    else:
        lon_mask = (lon >= lon_min) | (lon <= lon_max)
    return lat_mask[:, None] & lon_mask[None, :]


def _region_mask(lat: np.ndarray, lon: np.ndarray, region: ModeRegion) -> np.ndarray:
    mask = np.zeros((len(lat), len(lon)), dtype=bool)
    for box in region.boxes:
        mask |= _box_mask(lat, lon, box)
    return mask


def _peak_location(values: np.ndarray, lat: np.ndarray, lon: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    masked = np.where(mask, np.abs(values), np.nan)
    if np.all(np.isnan(masked)):
        return float("nan"), float("nan"), 0.0
    peak_flat = int(np.nanargmax(masked))
    lat_idx, lon_idx = np.unravel_index(peak_flat, masked.shape)
    return float(lon[lon_idx]), float(lat[lat_idx]), float(masked[lat_idx, lon_idx])


def find_mode_components(
    component_maps: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    regions: Iterable[ModeRegion] = DEFAULT_MODE_REGIONS,
    *,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Find the components whose loadings overlap each classic mode region."""

    maps = np.asarray(component_maps, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = _normalise_lon(np.asarray(lon, dtype=float))
    if maps.ndim != 3:
        raise ValueError("component_maps must have shape (lat, lon, component)")
    if maps.shape[:2] != (len(lat_arr), len(lon_arr)):
        raise ValueError("lat/lon dimensions do not match component_maps")

    rows: list[dict[str, Any]] = []
    for region in regions:
        mask = _region_mask(lat_arr, lon_arr, region)
        candidates: list[dict[str, Any]] = []
        for component_index in range(maps.shape[2]):
            values = maps[..., component_index]
            abs_values = np.abs(values)
            total_abs = float(np.nansum(abs_values))
            regional_abs = float(np.nansum(abs_values[mask]))
            overlap = regional_abs / total_abs if total_abs > 0.0 else 0.0
            center_lon, center_lat, peak_abs = _peak_location(values, lat_arr, lon_arr, mask)
            candidates.append(
                {
                    "mode": region.name,
                    "candidate_rank": 0,
                    "component": f"component_{component_index + 1:02d}",
                    "component_index": int(component_index),
                    "overlap_abs": overlap,
                    "regional_abs": regional_abs,
                    "total_abs": total_abs,
                    "peak_abs_in_region": peak_abs,
                    "center_lon": center_lon,
                    "center_lat": center_lat,
                }
            )
        candidates.sort(
            key=lambda item: (
                float(item["overlap_abs"]),
                float(item["peak_abs_in_region"]),
                float(item["regional_abs"]),
            ),
            reverse=True,
        )
        for rank, item in enumerate(candidates[: int(top_n)], start=1):
            item["candidate_rank"] = rank
            rows.append(item)
    return rows


def _coerce_component_index(frame: pd.DataFrame) -> pd.DataFrame:
    if "component_index" in frame.columns:
        out = frame.copy()
        out["component_index"] = out["component_index"].astype(int)
        return out
    if "component" not in frame.columns:
        raise ValueError("ranking frame must include component_index or component")
    out = frame.copy()

    def parse_component(value: Any) -> int:
        if isinstance(value, str) and value.startswith("component_"):
            return int(value.split("_", 1)[1]) - 1
        return int(value)

    out["component_index"] = out["component"].map(parse_component).astype(int)
    return out


def _metric_lookup(frame: pd.DataFrame | None, metric: str) -> tuple[dict[int, float], dict[int, int]]:
    if frame is None or metric not in frame.columns:
        return {}, {}
    indexed = _coerce_component_index(frame)
    values = indexed.set_index("component_index")[metric].astype(float)
    ranks = values.rank(ascending=False, method="min").astype(int)
    return values.to_dict(), ranks.to_dict()


def _get_value(values: dict[int, float], component_index: int) -> float | None:
    value = values.get(int(component_index))
    return None if value is None or not np.isfinite(value) else float(value)


def _get_rank(ranks: dict[int, int], component_index: int) -> int | None:
    value = ranks.get(int(component_index))
    return None if value is None else int(value)


def _rank_uplift(pair_rank: int | None, hyper_rank: int | None) -> int | None:
    if pair_rank is None or hyper_rank is None:
        return None
    return int(pair_rank) - int(hyper_rank)


def audit_classic_modes(
    component_maps: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    regions: Iterable[ModeRegion] = DEFAULT_MODE_REGIONS,
    *,
    pairwise_gateway: pd.DataFrame,
    pairwise_mediator: pd.DataFrame,
    hyper_gateway: pd.DataFrame | None = None,
    hyper_mediator: pd.DataFrame | None = None,
    top_n: int = 3,
    top_k: int = 10,
) -> dict[str, Any]:
    mode_rows = find_mode_components(component_maps, lat, lon, regions, top_n=top_n)

    ace_values, ace_ranks = _metric_lookup(pairwise_gateway, "ace")
    acs_values, acs_ranks = _metric_lookup(pairwise_gateway, "acs")
    amce_values, amce_ranks = _metric_lookup(pairwise_mediator, "amce")
    hyper_ace_values, hyper_ace_ranks = _metric_lookup(hyper_gateway, "hyper_ace_total")
    hyper_ace_order2, _ = _metric_lookup(hyper_gateway, "hyper_ace_order2")
    hyper_amce_values, hyper_amce_ranks = _metric_lookup(hyper_mediator, "hyper_amce_total")
    mediator_order2, _ = _metric_lookup(hyper_mediator, "mediator_synergy_order2")

    enriched: list[dict[str, Any]] = []
    for row in mode_rows:
        component_index = int(row["component_index"])
        pair_ace_rank = _get_rank(ace_ranks, component_index)
        pair_acs_rank = _get_rank(acs_ranks, component_index)
        pair_amce_rank = _get_rank(amce_ranks, component_index)
        hyper_ace_rank = _get_rank(hyper_ace_ranks, component_index)
        hyper_amce_rank = _get_rank(hyper_amce_ranks, component_index)
        out = dict(row)
        out.update(
            {
                "pairwise_ace": _get_value(ace_values, component_index),
                "pairwise_ace_rank": pair_ace_rank,
                "pairwise_acs": _get_value(acs_values, component_index),
                "pairwise_acs_rank": pair_acs_rank,
                "pairwise_amce": _get_value(amce_values, component_index),
                "pairwise_amce_rank": pair_amce_rank,
                "hyper_ace_total": _get_value(hyper_ace_values, component_index),
                "hyper_ace_rank": hyper_ace_rank,
                "hyper_ace_order2": _get_value(hyper_ace_order2, component_index),
                "hyper_amce_total": _get_value(hyper_amce_values, component_index),
                "hyper_amce_rank": hyper_amce_rank,
                "mediator_synergy_order2": _get_value(mediator_order2, component_index),
                "hyper_ace_rank_uplift": _rank_uplift(pair_ace_rank, hyper_ace_rank),
                "hyper_amce_rank_uplift": _rank_uplift(pair_amce_rank, hyper_amce_rank),
            }
        )
        enriched.append(out)

    primary_rows = [row for row in enriched if int(row["candidate_rank"]) == 1]

    def in_top(row: dict[str, Any], keys: tuple[str, ...]) -> bool:
        ranks = [row.get(key) for key in keys if row.get(key) is not None]
        return bool(ranks) and min(int(rank) for rank in ranks) <= int(top_k)

    summary = {
        "n_modes": len({row["mode"] for row in enriched}),
        "n_mode_components": len(enriched),
        "top_n": int(top_n),
        "top_k": int(top_k),
        "hyper_available": hyper_gateway is not None and hyper_mediator is not None,
        "pairwise_topk_any_primary_modes": sum(
            in_top(row, ("pairwise_ace_rank", "pairwise_acs_rank", "pairwise_amce_rank")) for row in primary_rows
        ),
        "pairwise_topk_gateway_primary_modes": sum(
            in_top(row, ("pairwise_ace_rank", "pairwise_acs_rank")) for row in primary_rows
        ),
        "pairwise_topk_mediator_primary_modes": sum(in_top(row, ("pairwise_amce_rank",)) for row in primary_rows),
        "hyper_topk_any_primary_modes": sum(
            in_top(row, ("hyper_ace_rank", "hyper_amce_rank")) for row in primary_rows
        ),
        "positive_hyper_ace_uplift_primary_modes": sum(
            (row.get("hyper_ace_rank_uplift") is not None and int(row["hyper_ace_rank_uplift"]) > 0) for row in primary_rows
        ),
        "positive_hyper_amce_uplift_primary_modes": sum(
            (row.get("hyper_amce_rank_uplift") is not None and int(row["hyper_amce_rank_uplift"]) > 0) for row in primary_rows
        ),
    }
    return {"summary": summary, "modes": enriched}


def load_component_maps(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(path)
    maps = np.asarray(payload["component_maps"], dtype=float)
    lat = np.asarray(payload["lat"], dtype=float) if "lat" in payload.files else np.linspace(-90.0, 90.0, maps.shape[0])
    lon = (
        np.asarray(payload["lon"], dtype=float)
        if "lon" in payload.files
        else np.linspace(0.0, 360.0, maps.shape[1], endpoint=False)
    )
    return maps, lat, lon


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    headers = [
        "mode",
        "component",
        "overlap_abs",
        "pairwise_ace_rank",
        "pairwise_acs_rank",
        "pairwise_amce_rank",
        "hyper_ace_rank",
        "hyper_amce_rank",
        "hyper_ace_rank_uplift",
        "hyper_amce_rank_uplift",
        "hyper_ace_order2",
        "mediator_synergy_order2",
    ]
    lines = [
        "# Runge classic mode audit",
        "",
        "## Summary",
        "",
    ]
    for key, value in report["summary"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Mode Candidates", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
    for row in report["modes"]:
        values = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-maps", type=Path, required=True)
    parser.add_argument("--pairwise-gateway", type=Path, required=True)
    parser.add_argument("--pairwise-mediator", type=Path, required=True)
    parser.add_argument("--hyper-gateway", type=Path)
    parser.add_argument("--hyper-mediator", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    maps, lat, lon = load_component_maps(args.component_maps.expanduser())
    pairwise_gateway = pd.read_csv(args.pairwise_gateway.expanduser())
    pairwise_mediator = pd.read_csv(args.pairwise_mediator.expanduser())
    hyper_gateway = pd.read_csv(args.hyper_gateway.expanduser()) if args.hyper_gateway else None
    hyper_mediator = pd.read_csv(args.hyper_mediator.expanduser()) if args.hyper_mediator else None
    report = audit_classic_modes(
        maps,
        lat,
        lon,
        pairwise_gateway=pairwise_gateway,
        pairwise_mediator=pairwise_mediator,
        hyper_gateway=hyper_gateway,
        hyper_mediator=hyper_mediator,
        top_n=args.top_n,
        top_k=args.top_k,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=True, indent=2, default=_json_default) + "\n", encoding="utf-8")
    if args.output_markdown:
        write_markdown(report, args.output_markdown)
    if not args.output_json and not args.output_markdown:
        print(json.dumps(report, ensure_ascii=True, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
