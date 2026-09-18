#!/usr/bin/env python3
"""Download NASA OMNIWeb minute data for May 2024 and reproduce the curves."""

from __future__ import annotations

import html
import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


OMNI_ENDPOINT = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
OUTPUT_DIR = Path(__file__).resolve().parent
DATA_PATH = OUTPUT_DIR / "omni_min_202405_imf_speed.txt"
FIGURE_PATH = OUTPUT_DIR / "omni_min_202405_imf_speed.png"


def download_listing() -> str:
    """Return the official OMNIWeb plain-text listing embedded in its response."""
    form_data = [
        ("activity", "retrieve"),
        ("res", "min"),
        ("spacecraft", "omni_min"),
        ("start_date", "20240501"),
        ("end_date", "20240531"),
        ("vars", "13"),  # IMF Magnitude Avg (Scalar), nT
        ("vars", "21"),  # Flow Speed, km/sec
    ]
    response = requests.post(OMNI_ENDPOINT, data=form_data, timeout=120)
    response.raise_for_status()
    match = re.search(r"<pre>(.*?)</pre>", response.text, flags=re.I | re.S)
    if match is None:
        raise RuntimeError("OMNIWeb response did not contain the expected data listing")
    return html.unescape(match.group(1)).strip() + "\n"


def parse_listing(listing: str) -> pd.DataFrame:
    rows: list[list[float]] = []
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) != 6 or not parts[0].isdigit():
            continue
        rows.append([float(value) for value in parts])

    data = pd.DataFrame(
        rows,
        columns=["year", "day_of_year", "hour", "minute", "imf_magnitude", "speed"],
    )
    if len(data) != 31 * 24 * 60:
        raise RuntimeError(f"Expected 44,640 minute records, received {len(data):,}")

    data["time"] = (
        pd.to_datetime(data["year"].astype(int).astype(str), format="%Y")
        + pd.to_timedelta(data["day_of_year"] - 1, unit="D")
        + pd.to_timedelta(data["hour"], unit="h")
        + pd.to_timedelta(data["minute"], unit="m")
    )
    data.loc[data["imf_magnitude"] == 9999.99, "imf_magnitude"] = np.nan
    data.loc[data["speed"] == 99999.9, "speed"] = np.nan
    return data


def break_long_gaps(data: pd.DataFrame, max_gap_minutes: int = 30) -> pd.DataFrame:
    """Insert NaNs before valid samples separated by a long observation gap."""
    plotted = data.copy()
    for column in ("imf_magnitude", "speed"):
        valid_times = plotted.loc[plotted[column].notna(), "time"]
        long_gap_starts = valid_times.index[valid_times.diff().gt(pd.Timedelta(minutes=max_gap_minutes))]
        plotted.loc[long_gap_starts, column] = np.nan
    return plotted


def annotate_peak(ax: plt.Axes, data: pd.DataFrame, column: str, color: str) -> None:
    peak_index = data[column].idxmax()
    peak_time = data.loc[peak_index, "time"]
    peak_value = data.loc[peak_index, column]
    decimals = 2 if column == "imf_magnitude" else 1
    label_offset = (16, -27) if column == "imf_magnitude" else (18, 8)
    ax.scatter([peak_time], [peak_value], s=18, color=color, zorder=4)
    ax.annotate(
        f"Peak {peak_value:.{decimals}f}\n{peak_time:%d %b %H:%M} UTC",
        xy=(peak_time, peak_value),
        xytext=label_offset,
        textcoords="offset points",
        fontsize=8,
        color=color,
        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8},
    )


def plot_curves(data: pd.DataFrame) -> None:
    plotted = break_long_gaps(data)
    blue = "#2F6FB0"
    red = "#C75555"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.0, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.10},
        constrained_layout=True,
    )

    axes[0].plot(plotted["time"], plotted["imf_magnitude"], color=blue, lw=0.70)
    axes[1].plot(plotted["time"], plotted["speed"], color=red, lw=0.70)

    for ax in axes:
        ax.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.65)
        ax.margins(x=0)

    axes[0].set_ylabel("IMF magnitude\n(nT)")
    axes[1].set_ylabel("Flow speed\n(km s$^{-1}$)")
    axes[1].set_xlabel("Date in May 2024 (UTC)")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(250, 1100)

    annotate_peak(axes[0], data, "imf_magnitude", blue)
    annotate_peak(axes[1], data, "speed", red)

    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=5))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[1].set_xlim(pd.Timestamp("2024-05-01"), pd.Timestamp("2024-06-01"))

    fig.savefig(FIGURE_PATH, dpi=400, bbox_inches="tight", facecolor="white")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    listing = download_listing()
    DATA_PATH.write_text(listing, encoding="utf-8")
    data = parse_listing(listing)
    plot_curves(data)

    for column, unit in (("imf_magnitude", "nT"), ("speed", "km/s")):
        idx = data[column].idxmax()
        print(f"{column} peak: {data.loc[idx, column]:.2f} {unit} at {data.loc[idx, 'time']} UTC")
    print(f"valid records: {data['speed'].notna().sum():,}/{len(data):,}")
    print(f"data: {DATA_PATH}")
    print(f"figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
