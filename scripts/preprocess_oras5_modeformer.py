#!/usr/bin/env python3
"""Build checkpoint-compatible UniCM Modeformer inputs from monthly ORAS5 fields."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
from netCDF4 import Dataset


MODE_NAMES = (
    "ENSO",
    "NPMM",
    "SPMM",
    "IOB",
    "IOD",
    "SIOD",
    "TNA",
    "nino12",
    "nino3",
    "nino4",
    "WWV",
)

# Half-open boxes reproduce the array slices in the released LoadData.py.
# Longitudes use 0..360 degrees east.
SST_REGIONS = {
    "ENSO": ((-5, 5, 190, 240),),
    "NPMM": ((10, 25, 200, 240),),
    "SPMM": ((-25, -15, 250, 270),),
    "IOB": ((-20, 20, 40, 100),),
    "IOD": ((-10, 10, 50, 70), (-10, 0, 90, 110)),
    "SIOD": ((-25, -10, 65, 80), (-30, -10, 90, 120)),
    "TNA": ((5, 25, 305, 345),),
    "nino12": ((-10, 0, 270, 280),),
    "nino3": ((-5, 5, 210, 270),),
    # This narrow box intentionally follows the released checkpoint code.
    "nino4": ((-5, 5, 200, 210),),
}
WWV_REGION = (-5, 5, 120, 280)


def load_monthly_field(root: Path, variable: str, start_year: int, end_year: int):
    files = sorted((root / variable).glob(f"{variable}_ORAS5_1m_*_r1x1.nc"))
    files = [
        path
        for path in files
        if start_year <= int(path.name.split("_1m_")[1][:4]) <= end_year
    ]
    expected = (end_year - start_year + 1) * 12
    if len(files) != expected:
        raise RuntimeError(f"{variable}: expected {expected} files, found {len(files)}")

    values = []
    dates = []
    lat = lon = None
    for path in files:
        stamp = path.name.split("_1m_")[1][:6]
        dates.append(f"{stamp[:4]}-{stamp[4:]}")
        with Dataset(path) as dataset:
            if lat is None:
                lat = np.asarray(dataset.variables["lat"][:], dtype=np.float32)
                lon = np.asarray(dataset.variables["lon"][:], dtype=np.float32)
            field = dataset.variables[variable][0]
            if np.ma.isMaskedArray(field):
                field = field.filled(0.0)
            values.append(np.nan_to_num(np.asarray(field, dtype=np.float32)))

    return np.stack(values), lat, lon, np.asarray(dates)


def code_style_anomalies(field: np.ndarray):
    if field.shape[0] % 12:
        raise RuntimeError("The monthly series must contain complete calendar years")
    climatology = field.reshape(-1, 12, *field.shape[1:]).mean(axis=0)
    anomalies = field - climatology[np.arange(field.shape[0]) % 12]
    valid = np.any(anomalies != 0, axis=0)
    scaler = float(np.nanstd(np.where(valid[None], anomalies, np.nan)))
    if not np.isfinite(scaler) or scaler <= 0:
        raise RuntimeError(f"Invalid anomaly scaler: {scaler}")
    return anomalies, anomalies / scaler, climatology, scaler


def region_mean(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    box: tuple[float, float, float, float],
) -> np.ndarray:
    lat0, lat1, lon0, lon1 = box
    lat_mask = (lat >= lat0) & (lat < lat1)
    lon_mask = (lon >= lon0) & (lon < lon1)
    if not lat_mask.any() or not lon_mask.any():
        raise RuntimeError(f"Empty region: {box}")
    return field[:, lat_mask][:, :, lon_mask].mean(axis=(1, 2))


def extract_modes(
    sst: np.ndarray,
    depth20: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    output = []
    for name in MODE_NAMES[:-1]:
        boxes = SST_REGIONS[name]
        value = region_mean(sst, lat, lon, boxes[0])
        if len(boxes) == 2:
            value = value - region_mean(sst, lat, lon, boxes[1])
        output.append(value)
    output.append(region_mean(depth20, lat, lon, WWV_REGION))
    return np.stack(output, axis=1).astype(np.float32)


def build_windows(series: np.ndarray, month_ids: np.ndarray):
    window = 36
    count = series.shape[0] - window + 1
    mode_windows = np.stack(
        [series[start : start + window].T for start in range(count)]
    )
    timestamp_windows = np.stack(
        [month_ids[start : start + window] for start in range(count)]
    ).astype(np.int64)
    return mode_windows, timestamp_windows


def verify_checkpoints(
    checkpoint_root: Path,
    windows: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    import torch

    source = checkpoint_root / "src"
    sys.path.insert(0, str(source))
    # pynvml is imported by the training module but is not needed for inference.
    sys.modules.setdefault("pynvml", ModuleType("pynvml"))
    from models import UniCM

    parameters = SimpleNamespace(
        d_size=256,
        device=torch.device("cpu"),
        input_channal=5,
        patch_size=[2, 2],
        emb_spatial_size=216,
        nheads=4,
        dim_feedforward=512,
        dropout=0.2,
        num_encoder_layers=4,
        num_decoder_layers=4,
        val_relative=[None] * 10,
        t20d_mode=1,
        mode_interaction="1",
        his_len=12,
        pred_len=24,
        autoregressive=0,
    )
    paths = sorted(
        glob.glob(str(source / "experiments" / "*" / "model_save" / "model_best.pkl"))
    )
    if len(paths) != 3:
        raise RuntimeError(f"Expected three checkpoints, found {len(paths)}")

    reports = []
    checkpoint_predictions = []
    for path in paths:
        model = UniCM(parameters)
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=True)
        model.eval()
        batches = []
        with torch.no_grad():
            for start in range(0, windows.shape[0], 64):
                x = torch.from_numpy(windows[start : start + 64]).float()
                ts = torch.from_numpy(timestamps[start : start + 64]).long()
                x_internal = x.permute(0, 2, 1).unsqueeze(-1).unsqueeze(2)
                pred, _, _ = model.forward_sep(
                    x_internal,
                    ts,
                    model.encoder_mode,
                    model.decoder_mode,
                    model.linear_output_mode,
                    model.predictor_emb_mode,
                    model.predictand_emb_mode,
                    1,
                    [1, 1],
                    train=False,
                )
                batches.append(
                    pred.squeeze(-1).squeeze(2).permute(0, 2, 1).cpu().numpy()
                )
        prediction = np.concatenate(batches).astype(np.float32)
        checkpoint_predictions.append(prediction)
        reports.append(
            {
                "checkpoint": str(Path(path).resolve()),
                "strict_load": True,
                "input_shape": list(windows.shape),
                "output_shape": list(prediction.shape),
                "output_finite": bool(np.isfinite(prediction).all()),
                "output_std": float(prediction.std()),
            }
        )
    return reports, np.stack(checkpoint_predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/ORAS5/icdc_r1x1_opa0")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ORAS5/modeformer_1980_2014/model_inputs.npz"),
    )
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("data/UniCM-checkpoint")
    )
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2014)
    args = parser.parse_args()

    sst, lat, lon, dates = load_monthly_field(
        args.raw_root, "sosstsst", args.start_year, args.end_year
    )
    depth20, lat2, lon2, dates2 = load_monthly_field(
        args.raw_root, "so20chgt", args.start_year, args.end_year
    )
    if not (
        np.array_equal(lat, lat2)
        and np.array_equal(lon, lon2)
        and np.array_equal(dates, dates2)
    ):
        raise RuntimeError("SST and 20 C depth coordinates are not aligned")

    sst_anom, sst_norm, sst_clim, sst_std = code_style_anomalies(sst)
    depth_anom, depth_norm, depth_clim, depth_std = code_style_anomalies(depth20)
    modes_normalized = extract_modes(sst_norm, depth_norm, lat, lon)
    modes_physical = extract_modes(sst_anom, depth_anom, lat, lon)
    month_ids = np.asarray([int(date[-2:]) - 1 for date in dates], dtype=np.int64)
    windows, timestamp_windows = build_windows(modes_normalized, month_ids)

    if not np.isfinite(windows).all():
        raise RuntimeError("Preprocessed Modeformer inputs contain NaN or Inf")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": "ORAS5 ICDC r1x1 opa0",
        "period": f"{args.start_year}-01/{args.end_year}-12",
        "mode_order": list(MODE_NAMES),
        "history_months": 12,
        "forecast_months": 24,
        "running_mean_months": 1,
        "normalization": (
            "calendar-month gridpoint anomalies divided by one field-wide "
            "anomaly standard deviation, matching released LoadData.py"
        ),
        "nino4_note": "Uses 200E-210E to match the released checkpoint code.",
        "sst_std": sst_std,
        "so20chgt_std": depth_std,
    }
    np.savez_compressed(
        args.output,
        mode_windows=windows,
        history=windows[:, :, :12],
        targets=windows[:, :, 12:],
        timestamps=timestamp_windows,
        mode_series=modes_normalized,
        mode_series_physical=modes_physical,
        mode_names=np.asarray(MODE_NAMES),
        dates=dates,
        latitude=lat,
        longitude=lon,
        sst_monthly_climatology=sst_clim,
        so20chgt_monthly_climatology=depth_clim,
        metadata=np.asarray(json.dumps(metadata)),
    )

    reports, predictions = verify_checkpoints(
        args.checkpoint_root, windows, timestamp_windows
    )
    prediction_path = args.output.with_name("modeformer_predictions.npz")
    np.savez_compressed(
        prediction_path,
        predictions_by_seed=predictions,
        ensemble_prediction=predictions.mean(axis=0),
        targets=windows[:, :, 12:],
        timestamps=timestamp_windows[:, 12:],
        mode_names=np.asarray(MODE_NAMES),
        target_dates=np.stack(
            [dates[start + 12 : start + 36] for start in range(windows.shape[0])]
        ),
    )
    report_path = args.output.with_name("compatibility_report.json")
    report = {
        "data_file": str(args.output.resolve()),
        "prediction_file": str(prediction_path.resolve()),
        "series_shape": list(modes_normalized.shape),
        "window_shape": list(windows.shape),
        "history_shape": list(windows[:, :, :12].shape),
        "target_shape": list(windows[:, :, 12:].shape),
        "timestamp_shape": list(timestamp_windows.shape),
        "metadata": metadata,
        "checkpoints": reports,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
