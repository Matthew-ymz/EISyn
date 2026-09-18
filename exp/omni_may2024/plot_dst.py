#!/usr/bin/env python3
"""Download and validate the provisional hourly Dst index for May 2024."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests


OUTPUT_DIR = Path(__file__).resolve().parent
DST_PATH = OUTPUT_DIR / "dst_index_may2024.csv"
DST_URL = (
    "https://wdc.kugi.kyoto-u.ac.jp/hapi/data"
    "?id=hour_dst_provisional"
    "&time.min=2024-05-01T00:00:00Z"
    "&time.max=2024-06-01T00:00:00Z"
    "&format=csv"
)


def download_dst() -> pd.DataFrame:
    response = requests.get(DST_URL, timeout=60)
    response.raise_for_status()
    data = pd.read_csv(
        StringIO(response.text),
        header=None,
        names=["datetime_utc", "dst_nT", "status"],
        parse_dates=["datetime_utc"],
    )
    validate_dst(data)
    return data


def validate_dst(data: pd.DataFrame) -> None:
    if len(data) != 31 * 24:
        raise RuntimeError(f"Expected 744 hourly Dst values, received {len(data)}")
    if data["datetime_utc"].isna().any() or not data["datetime_utc"].is_monotonic_increasing:
        raise RuntimeError("Dst timestamps are invalid or not increasing")
    if data["datetime_utc"].duplicated().any():
        raise RuntimeError("Dst data contain duplicate timestamps")
    if data["dst_nT"].isna().any():
        raise RuntimeError("Dst data contain missing values")

    minimum = data.loc[data["dst_nT"].idxmin()]
    expected_time = pd.Timestamp("2024-05-11T02:29:30Z")
    if minimum["dst_nT"] != -406 or minimum["datetime_utc"] != expected_time:
        raise RuntimeError(
            "Unexpected May 2024 Dst minimum: "
            f"{minimum['dst_nT']} nT at {minimum['datetime_utc']}"
        )


def load_dst(path: Path = DST_PATH) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["datetime_utc"])
    validate_dst(data)
    return data


def main() -> None:
    data = download_dst()
    data.to_csv(DST_PATH, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    minimum = data.loc[data["dst_nT"].idxmin()]
    print(f"saved: {DST_PATH}")
    print(f"records: {len(data)}")
    print(f"minimum: {minimum['dst_nT']} nT at {minimum['datetime_utc']}")


if __name__ == "__main__":
    main()
