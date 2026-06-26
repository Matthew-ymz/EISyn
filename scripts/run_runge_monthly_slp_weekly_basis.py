#!/usr/bin/env python3
"""Run monthly SLP MLP-TM-EI after projecting onto weekly Runge SLP maps."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_runge_monthly_variable_comparison import (  # noqa: E402
    MonthlyDatasetConfig,
    detrend_valid_cells,
    load_monthly_field,
    standardize_monthly_anomalies,
)


def project_to_weekly_basis(
    *,
    monthly_slp: Path,
    weekly_maps: Path,
    output_dir: Path,
    n_components: int,
) -> tuple[Path, Path]:
    config = MonthlyDatasetConfig(
        name="slp_monthly_weekly_basis",
        path=monthly_slp,
        variable="slp",
        display_name="Monthly SLP projected to weekly SLP basis",
    )
    raw = load_monthly_field(config)
    standardized = standardize_monthly_anomalies(raw)
    detrended = detrend_valid_cells(standardized)
    values = np.asarray(detrended.values, dtype=np.float64)
    n_time, n_lat, n_lon = values.shape

    weekly_payload = np.load(weekly_maps)
    maps = np.asarray(weekly_payload["component_maps"], dtype=np.float64)
    if maps.shape[:2] != (n_lat, n_lon):
        raise ValueError(f"weekly maps grid {maps.shape[:2]} does not match monthly field {(n_lat, n_lon)}")
    maps = maps[..., : int(n_components)]
    lat = np.asarray(detrended["lat"].values, dtype=float)
    if lat[0] < lat[-1]:
        # Legacy weekly maps follow the original NCEP order, 90..-90. The
        # monthly loader sorts latitude ascending, so flip the basis maps before
        # both projection and persistence.
        maps = maps[::-1, :, :]
    valid = np.all(np.isfinite(values.reshape(n_time, n_lat * n_lon)), axis=0)
    flat_values = values.reshape(n_time, n_lat * n_lon)[:, valid]
    flat_maps = maps.reshape(n_lat * n_lon, maps.shape[-1])[valid, :]

    # Treat stored Varimax loadings as spatial filters. Normalize each filter so
    # scores are comparable before standardization.
    norms = np.sqrt(np.sum(flat_maps**2, axis=0, keepdims=True))
    norms = np.where(norms > 1.0e-12, norms, 1.0)
    filters = flat_maps / norms
    scores = flat_values @ filters
    scores = (scores - scores.mean(axis=0, keepdims=True)) / np.maximum(scores.std(axis=0, ddof=1, keepdims=True), 1.0e-12)

    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [f"component_{idx + 1:02d}" for idx in range(scores.shape[1])]
    score_frame = pd.DataFrame(scores, index=pd.to_datetime(detrended["time"].values), columns=columns)
    score_frame.index.name = "time"
    scores_path = output_dir / "component_monthly_scores.csv"
    maps_path = output_dir / "component_maps.npz"
    manifest_path = output_dir / "manifest.json"
    score_frame.to_csv(scores_path)
    np.savez_compressed(
        maps_path,
        component_maps=maps,
        explained_variance_ratio=np.full(scores.shape[1], np.nan, dtype=float),
        lat=lat,
        lon=np.asarray(detrended["lon"].values, dtype=float),
        valid_mask=valid.reshape(n_lat, n_lon),
    )
    manifest = {
        "dataset": config.name,
        "input_path": str(monthly_slp),
        "weekly_basis_maps": str(weekly_maps),
        "preprocessing": {
            "monthly_deseasonalized": True,
            "monthly_standardized": True,
            "linear_detrended": True,
            "projected_to_weekly_slp_varimax_basis": True,
        },
        "n_monthly_samples": int(score_frame.shape[0]),
        "time_start": str(score_frame.index.min().date()),
        "time_end": str(score_frame.index.max().date()),
        "n_components": int(scores.shape[1]),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return scores_path, maps_path


def run_pairwise(args: argparse.Namespace, scores_path: Path, output_dir: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/run_runge_pairwise_mlp_ei.py",
        "--component-scores",
        str(scores_path),
        "--output-dir",
        str(output_dir),
        "--ei-estimator",
        "tm",
        "--gateway-mode",
        "path_effect",
        "--source-mode",
        "latest",
        "--lag",
        str(args.lag),
        "--horizon",
        str(args.horizon),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--batch-size",
        str(args.batch_size),
        "--weight-decay",
        str(args.weight_decay),
        "--ridge-alpha",
        str(args.ridge_alpha),
        "--ensemble-ridge-alphas",
        str(args.ensemble_ridge_alphas),
        "--linear-blend-grid-steps",
        str(args.linear_blend_grid_steps),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--scheduler-patience",
        str(args.scheduler_patience),
        "--gradient-clip-norm",
        str(args.gradient_clip_norm),
        "--intervention-samples",
        str(args.intervention_samples),
        "--seed",
        str(args.seed),
    ]
    if args.force_retrain:
        cmd.append("--force-retrain")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-slp", type=Path, default=Path("data/ncep_reanalysis_slp/monthly/slp.mon.mean.nc"))
    parser.add_argument("--weekly-maps", type=Path, default=Path("results/runge/2015_gateways/component_maps.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/runge_monthly_slp_weekly_basis/lag_01"))
    parser.add_argument("--n-components", type=int, default=60)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--ridge-alpha", type=float, default=1000.0)
    parser.add_argument("--ensemble-ridge-alphas", default="10,100,1000,3000")
    parser.add_argument("--linear-blend-grid-steps", type=int, default=101)
    parser.add_argument("--early-stopping-patience", type=int, default=80)
    parser.add_argument("--scheduler-patience", type=int, default=20)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--intervention-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.output_dir / "slp_monthly_weekly_basis"
    scores_path = dataset_dir / "component_monthly_scores.csv"
    if args.force_preprocess or not scores_path.exists():
        scores_path, _ = project_to_weekly_basis(
            monthly_slp=args.monthly_slp,
            weekly_maps=args.weekly_maps,
            output_dir=dataset_dir,
            n_components=int(args.n_components),
        )
    run_pairwise(args, scores_path, dataset_dir / "mlp_tm_ei")
    print(json.dumps({"scores": str(scores_path), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
