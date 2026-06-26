#!/usr/bin/env python3
"""Compare monthly SLP lag choices against the weekly Runge SLP reference."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_runge_monthly_variable_comparison import component_centers_from_scores  # noqa: E402


def load_component_maps(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(path)
    maps = np.asarray(payload["component_maps"], dtype=float)
    if "lat" in payload.files and "lon" in payload.files:
        lat = np.asarray(payload["lat"], dtype=float)
        lon = np.asarray(payload["lon"], dtype=float)
    else:
        # Legacy weekly Runge artifacts predate lat/lon persistence. The shape is
        # the standard NCEP 2.5 degree global grid: 90..-90, 0..357.5.
        lat = np.linspace(90.0, -90.0, maps.shape[0], dtype=float)
        lon = np.arange(maps.shape[1], dtype=float) * 2.5
    return maps, lat, lon


def top_centers(result_dir: Path, maps_path: Path, metric: str, top_k: int) -> pd.DataFrame:
    maps, lat, lon = load_component_maps(maps_path)
    if metric in {"ace", "acs"}:
        scores = pd.read_csv(result_dir / "gateway_scores.csv")
    elif metric == "amce":
        scores = pd.read_csv(result_dir / "mediator_scores.csv")
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return component_centers_from_scores(scores, maps, lat, lon, metric=metric, top_k=top_k)


def great_circle_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lon = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lon / 2.0) ** 2
    return float(radius * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a))))


def nearest_distance_summary(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, float]:
    distances = []
    for _, row in candidate.iterrows():
        d = [
            great_circle_km(float(row["lon"]), float(row["lat"]), float(ref["lon"]), float(ref["lat"]))
            for _, ref in reference.iterrows()
        ]
        distances.append(min(d))
    return {
        "mean_nearest_km": float(np.mean(distances)),
        "median_nearest_km": float(np.median(distances)),
        "max_nearest_km": float(np.max(distances)),
    }


def read_overall_metrics(result_dir: Path) -> dict[str, float]:
    metrics = pd.read_csv(result_dir / "mlp_metrics.csv")
    row = metrics[(metrics["split"] == "test") & (metrics["component"] == "overall")].iloc[0]
    manifest = json.loads((result_dir / "manifest.json").read_text())
    blend = manifest.get("linear_blend", {})
    return {
        "test_rmse": float(row["rmse"]),
        "test_mae": float(row["mae"]),
        "test_corr": float(row["corr"]),
        "mlp_weight": float(blend.get("mlp_weight", np.nan)) if blend.get("enabled", False) else np.nan,
    }


def analyze(top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly = {
        "name": "weekly_reference",
        "result_dir": Path("results/runge/pairwise_mlp_tm_ei_path_effects"),
        "maps_path": Path("results/runge/2015_gateways/component_maps.npz"),
    }
    candidates = [
        {
            "name": "monthly_weekly_basis_lag1",
            "result_dir": Path(
                "results/runge_monthly_slp_weekly_basis/lag_01/slp_monthly_weekly_basis/mlp_tm_ei/results/runge/pairwise_mlp_tm_ei_path_effects"
            ),
            "maps_path": Path("results/runge_monthly_slp_weekly_basis/lag_01/slp_monthly_weekly_basis/component_maps.npz"),
        },
        {
            "name": "monthly_weekly_basis_lag1_globalq",
            "result_dir": Path(
                "results/runge_monthly_slp_weekly_basis/lag_01_globalq/slp_monthly_weekly_basis/mlp_tm_ei/results/runge/pairwise_mlp_tm_ei_path_effects"
            ),
            "maps_path": Path("results/runge_monthly_slp_weekly_basis/lag_01/slp_monthly_weekly_basis/component_maps.npz"),
        },
        {
            "name": "monthly_weekly_basis_lag1_dense",
            "result_dir": Path(
                "results/runge_monthly_slp_weekly_basis/lag_01_dense/slp_monthly_weekly_basis/mlp_tm_ei/results/runge/pairwise_mlp_tm_ei_path_effects"
            ),
            "maps_path": Path("results/runge_monthly_slp_weekly_basis/lag_01/slp_monthly_weekly_basis/component_maps.npz"),
        },
        {
            "name": "monthly_weekly_basis_lag1_bidir",
            "result_dir": Path(
                "results/runge_monthly_slp_weekly_basis/lag_01_bidir/slp_monthly_weekly_basis/mlp_tm_ei/results/runge/pairwise_mlp_tm_ei_path_effects"
            ),
            "maps_path": Path("results/runge_monthly_slp_weekly_basis/lag_01/slp_monthly_weekly_basis/component_maps.npz"),
        },
        {
            "name": "monthly_lag1",
            "result_dir": Path(
                "results/runge_monthly_slp_lag_sensitivity/lag_01/slp_monthly/mlp_tm_ei/results/runge/pairwise_mlp_tm_ei_path_effects"
            ),
            "maps_path": Path("results/runge_monthly_slp_lag_sensitivity/lag_01/slp_monthly/component_maps.npz"),
        },
        {
            "name": "monthly_lag2",
            "result_dir": Path(
                "results/runge_monthly_slp_lag_sensitivity/lag_02/slp_monthly/mlp_tm_ei/results/runge/pairwise_mlp_tm_ei_path_effects"
            ),
            "maps_path": Path("results/runge_monthly_slp_lag_sensitivity/lag_02/slp_monthly/component_maps.npz"),
        },
        {
            "name": "monthly_lag3",
            "result_dir": Path(
                "results/runge_monthly_slp_lag_sensitivity/lag_03/slp_monthly/mlp_tm_ei/results/runge/pairwise_mlp_tm_ei_path_effects"
            ),
            "maps_path": Path("results/runge_monthly_slp_lag_sensitivity/lag_03/slp_monthly/component_maps.npz"),
        },
        {
            "name": "monthly_lag4",
            "result_dir": Path(
                "results/runge_monthly_variable_comparison/slp_monthly/mlp_tm_ei/results/runge/pairwise_mlp_tm_ei_path_effects"
            ),
            "maps_path": Path("results/runge_monthly_variable_comparison/slp_monthly/component_maps.npz"),
        },
    ]

    rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    for metric in ["ace", "acs", "amce"]:
        ref_centers = top_centers(weekly["result_dir"], weekly["maps_path"], metric, top_k)
        for _, row in ref_centers.iterrows():
            top_rows.append({"run": weekly["name"], "metric": metric, **row.to_dict()})
        for candidate in candidates:
            centers = top_centers(candidate["result_dir"], candidate["maps_path"], metric, top_k)
            for _, row in centers.iterrows():
                top_rows.append({"run": candidate["name"], "metric": metric, **row.to_dict()})
            summary = nearest_distance_summary(ref_centers, centers)
            rows.append({"run": candidate["name"], "metric": metric, **summary})

    metric_rows = []
    metric_rows.append({"run": weekly["name"], **read_overall_metrics(weekly["result_dir"])})
    for candidate in candidates:
        metric_rows.append({"run": candidate["name"], **read_overall_metrics(candidate["result_dir"])})

    alignment = pd.DataFrame(rows)
    prediction = pd.DataFrame(metric_rows)
    merged = alignment.merge(prediction, on="run", how="left")
    top_table = pd.DataFrame(top_rows)
    return merged, top_table


def write_report(alignment: pd.DataFrame, top_table: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Monthly SLP lag alignment against weekly Runge reference",
        "",
        "Distances compare each monthly top-5 center with the nearest weekly top-5 center for the same metric.",
        "Lower distance means closer geographic alignment. Weekly coordinates are restored from the legacy NCEP 2.5 degree grid.",
        "",
        "## Alignment and prediction skill",
        "",
        "| run | metric | mean nearest km | median nearest km | test RMSE | test corr | MLP weight |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in alignment.sort_values(["metric", "run"]).iterrows():
        lines.append(
            f"| {row['run']} | {row['metric'].upper()} | {row['mean_nearest_km']:.0f} | "
            f"{row['median_nearest_km']:.0f} | {row['test_rmse']:.3f} | {row['test_corr']:.3f} | {row['mlp_weight']:.2f} |"
        )
    lines.extend(["", "## Top centers", ""])
    for run in [
        "weekly_reference",
        "monthly_weekly_basis_lag1",
        "monthly_weekly_basis_lag1_globalq",
        "monthly_weekly_basis_lag1_dense",
        "monthly_weekly_basis_lag1_bidir",
        "monthly_lag1",
        "monthly_lag2",
        "monthly_lag3",
        "monthly_lag4",
    ]:
        lines.extend([f"### {run}", "", "| metric | label | lon | lat | score |", "|---|---|---:|---:|---:|"])
        subset = top_table[top_table["run"] == run]
        for _, row in subset.iterrows():
            lines.append(
                f"| {str(row['metric']).upper()} | {row['label']} | {row['lon']:.1f} | {row['lat']:.1f} | {row['score']:.6g} |"
            )
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/runge_monthly_slp_lag_sensitivity/alignment_summary.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alignment, top_table = analyze(top_k=int(args.top_k))
    write_report(alignment, top_table, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
