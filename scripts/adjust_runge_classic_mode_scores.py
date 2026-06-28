#!/usr/bin/env python3
"""Build classic-mode adjusted Runge PEID rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from scripts.audit_runge_classic_modes import DEFAULT_MODE_REGIONS, find_mode_components, load_component_maps
except ModuleNotFoundError:
    from audit_runge_classic_modes import DEFAULT_MODE_REGIONS, find_mode_components, load_component_maps


DEFAULT_PRIORITY_MODES: dict[str, float] = {
    "ENSO": 1.0,
    "IOD": 1.0,
    "Arctic": 1.0,
    "Tibetan_Plateau": 1.0,
    "SPMM": 1.0,
    "Maritime_Continent": 0.6,
    "NPMM": 0.6,
    "Tropical_Atlantic": 0.5,
    "South_Indian_Ocean_Dipole": 0.5,
    "North_Atlantic_NAO": 0.5,
}


def build_classic_weights(
    mode_rows: Iterable[dict[str, Any]],
    *,
    priority_modes: dict[str, float] = DEFAULT_PRIORITY_MODES,
    min_overlap: float = 0.02,
) -> dict[int, float]:
    weights: dict[int, float] = {}
    for row in mode_rows:
        mode = str(row["mode"])
        if mode not in priority_modes:
            continue
        if float(row.get("overlap_abs", 0.0)) < float(min_overlap):
            continue
        component_index = int(row["component_index"])
        candidate_rank = max(1, int(row.get("candidate_rank", 1)))
        mode_weight = float(priority_modes[mode])
        rank_weight = 1.0 / float(candidate_rank)
        weights[component_index] = max(weights.get(component_index, 0.0), mode_weight * rank_weight)
    return weights


def apply_classic_adjustment(
    scores: pd.DataFrame,
    *,
    score_column: str,
    rank_column: str,
    weights: dict[int, float],
    bonus_scale: float,
) -> pd.DataFrame:
    if "component_index" not in scores.columns:
        raise ValueError("scores must contain component_index")
    if score_column not in scores.columns:
        raise ValueError(f"scores must contain {score_column}")
    out = scores.copy()
    raw_column = f"{score_column}_raw"
    bonus_column = f"classic_bonus_{score_column}"
    out[raw_column] = out[score_column].astype(float)
    out["classic_mode_weight"] = out["component_index"].map(lambda idx: float(weights.get(int(idx), 0.0)))
    out[bonus_column] = out["classic_mode_weight"] * float(bonus_scale)
    out[score_column] = out[raw_column] + out[bonus_column]
    out[rank_column] = out[score_column].rank(ascending=False, method="min").astype(int)
    return out.sort_values(score_column, ascending=False).reset_index(drop=True)


def _score_scale(frame: pd.DataFrame, score_column: str, fraction: float) -> float:
    values = frame[score_column].astype(float).to_numpy()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.quantile(finite, 0.90) * float(fraction))


def _write_summary(
    output_path: Path,
    *,
    weights: dict[int, float],
    gateway_scale: float,
    mediator_scale: float,
    gateway: pd.DataFrame,
    mediator: pd.DataFrame,
) -> None:
    lines = [
        "# Runge classic-adjusted PEID scores",
        "",
        "## Adjustment",
        "",
        f"- gateway_bonus_scale: `{gateway_scale:.8g}`",
        f"- mediator_bonus_scale: `{mediator_scale:.8g}`",
        f"- weighted_components: `{len(weights)}`",
        "",
        "## Top Hyper-ACE",
        "",
        gateway[["component", "component_index", "hyper_ace_total", "hyper_ace_total_raw", "classic_bonus_hyper_ace_total", "hyper_ace_rank"]]
        .head(15)
        .to_markdown(index=False),
        "",
        "## Top Hyper-AMCE",
        "",
        mediator[
            [
                "component",
                "component_index",
                "hyper_amce_total",
                "hyper_amce_total_raw",
                "classic_bonus_hyper_amce_total",
                "hyper_amce_rank",
            ]
        ]
        .head(15)
        .to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-maps", type=Path, required=True)
    parser.add_argument("--hyper-gateway", type=Path, required=True)
    parser.add_argument("--hyper-mediator", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--min-overlap", type=float, default=0.02)
    parser.add_argument("--gateway-bonus-fraction", type=float, default=0.45)
    parser.add_argument("--mediator-bonus-fraction", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    maps, lat, lon = load_component_maps(args.component_maps.expanduser())
    mode_rows = find_mode_components(maps, lat, lon, DEFAULT_MODE_REGIONS, top_n=int(args.top_n))
    weights = build_classic_weights(mode_rows, min_overlap=float(args.min_overlap))
    gateway_raw = pd.read_csv(args.hyper_gateway.expanduser())
    mediator_raw = pd.read_csv(args.hyper_mediator.expanduser())
    gateway_scale = _score_scale(gateway_raw, "hyper_ace_total", float(args.gateway_bonus_fraction))
    mediator_scale = _score_scale(mediator_raw, "hyper_amce_total", float(args.mediator_bonus_fraction))
    gateway = apply_classic_adjustment(
        gateway_raw,
        score_column="hyper_ace_total",
        rank_column="hyper_ace_rank",
        weights=weights,
        bonus_scale=gateway_scale,
    )
    mediator = apply_classic_adjustment(
        mediator_raw,
        score_column="hyper_amce_total",
        rank_column="hyper_amce_rank",
        weights=weights,
        bonus_scale=mediator_scale,
    )
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mode_rows).to_json(output_dir / "classic_mode_components.json", orient="records", indent=2)
    gateway.to_csv(output_dir / "hyper_gateway_scores.csv", index=False)
    mediator.to_csv(output_dir / "hyper_mediator_scores.csv", index=False)
    manifest = {
        "method": "classic_mode_salience_adjusted_peid",
        "component_maps": str(args.component_maps.expanduser().resolve()),
        "hyper_gateway_raw": str(args.hyper_gateway.expanduser().resolve()),
        "hyper_mediator_raw": str(args.hyper_mediator.expanduser().resolve()),
        "priority_modes": DEFAULT_PRIORITY_MODES,
        "top_n": int(args.top_n),
        "min_overlap": float(args.min_overlap),
        "gateway_bonus_fraction": float(args.gateway_bonus_fraction),
        "mediator_bonus_fraction": float(args.mediator_bonus_fraction),
        "gateway_bonus_scale": gateway_scale,
        "mediator_bonus_scale": mediator_scale,
        "classic_weights": {str(key): value for key, value in sorted(weights.items())},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    _write_summary(
        output_dir / "summary.md",
        weights=weights,
        gateway_scale=gateway_scale,
        mediator_scale=mediator_scale,
        gateway=gateway,
        mediator=mediator,
    )


if __name__ == "__main__":
    main()
