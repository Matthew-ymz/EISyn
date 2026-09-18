#!/usr/bin/env python3
"""Plot May 2024 IMF magnitude, solar-wind speed, and geomagnetic Dst together."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from plot_dst import DST_PATH, load_dst
from reproduce_omni_curves import DATA_PATH, break_long_gaps, parse_listing


OUTPUT_DIR = Path(__file__).resolve().parent
FIGURE_PATH = OUTPUT_DIR / "imf_speed_dst_may2024.png"


def main() -> None:
    solar_wind = parse_listing(DATA_PATH.read_text(encoding="utf-8"))
    plotted_wind = break_long_gaps(solar_wind)
    dst = load_dst(DST_PATH)

    shock_time = pd.Timestamp("2024-05-10T16:36:00Z").tz_localize(None)
    storm_end = pd.Timestamp("2024-05-13T23:59:00Z").tz_localize(None)
    dst_time = dst["datetime_utc"].dt.tz_convert(None)

    imf_peak = solar_wind.loc[solar_wind["imf_magnitude"].idxmax()]
    speed_peak = solar_wind.loc[solar_wind["speed"].idxmax()]
    dst_minimum = dst.loc[dst["dst_nT"].idxmin()]
    dst_minimum_time = dst_minimum["datetime_utc"].tz_convert(None)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.2, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.15], "hspace": 0.10},
        constrained_layout=True,
    )

    colors = {"imf": "#2F6FB0", "speed": "#C75555", "dst": "#222222"}
    axes[0].plot(plotted_wind["time"], plotted_wind["imf_magnitude"], color=colors["imf"], lw=0.65)
    axes[1].plot(plotted_wind["time"], plotted_wind["speed"], color=colors["speed"], lw=0.65)
    axes[2].plot(dst_time, dst["dst_nT"], color=colors["dst"], lw=1.0)

    for ax in axes:
        ax.axvspan(shock_time, storm_end, color="#E8B04A", alpha=0.12, lw=0)
        ax.axvline(shock_time, color="#555555", lw=0.8, ls="--")
        ax.grid(axis="y", color="#D9D9D9", lw=0.5, alpha=0.7)
        ax.margins(x=0)

    axes[0].set_ylabel("IMF magnitude\n(nT)")
    axes[1].set_ylabel("Flow speed\n(km s$^{-1}$)")
    axes[2].set_ylabel("Dst (nT)")
    axes[2].set_xlabel("Date in May 2024 (UTC)")
    axes[0].set_ylim(0, 76)
    axes[1].set_ylim(250, 1100)
    axes[2].set_ylim(-450, 100)
    axes[2].axhline(0, color="#777777", lw=0.6)

    axes[0].annotate(
        "Interplanetary shock\n10 May 16:36 UTC",
        xy=(shock_time, 0.98),
        xycoords=("data", "axes fraction"),
        xytext=(-7, -4),
        textcoords="offset points",
        ha="right",
        va="top",
        color="#444444",
    )
    axes[0].scatter(imf_peak["time"], imf_peak["imf_magnitude"], s=16, color=colors["imf"], zorder=4)
    axes[0].annotate(
        f"IMF peak {imf_peak['imf_magnitude']:.2f} nT",
        xy=(imf_peak["time"], imf_peak["imf_magnitude"]),
        xytext=(24, -24),
        textcoords="offset points",
        color=colors["imf"],
        arrowprops={"arrowstyle": "-", "color": colors["imf"], "lw": 0.7},
    )
    axes[1].scatter(speed_peak["time"], speed_peak["speed"], s=16, color=colors["speed"], zorder=4)
    axes[1].annotate(
        f"Speed peak {speed_peak['speed']:.1f} km s$^{{-1}}$",
        xy=(speed_peak["time"], speed_peak["speed"]),
        xytext=(24, 7),
        textcoords="offset points",
        color=colors["speed"],
        arrowprops={"arrowstyle": "-", "color": colors["speed"], "lw": 0.7},
    )
    axes[2].scatter(dst_minimum_time, dst_minimum["dst_nT"], s=18, color=colors["dst"], zorder=4)
    axes[2].annotate(
        f"Dst minimum {dst_minimum['dst_nT']:.0f} nT\n11 May 02 UT",
        xy=(dst_minimum_time, dst_minimum["dst_nT"]),
        xytext=(28, 14),
        textcoords="offset points",
        color=colors["dst"],
        arrowprops={"arrowstyle": "-", "color": colors["dst"], "lw": 0.7},
    )

    axes[2].xaxis.set_major_locator(mdates.DayLocator(interval=5))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[2].set_xlim(pd.Timestamp("2024-05-01"), pd.Timestamp("2024-06-01"))
    fig.savefig(FIGURE_PATH, dpi=400, bbox_inches="tight", facecolor="white")

    lag = dst_minimum_time - shock_time
    print(f"figure: {FIGURE_PATH}")
    print(f"shock-to-Dst-minimum lag: {lag}")


if __name__ == "__main__":
    main()
