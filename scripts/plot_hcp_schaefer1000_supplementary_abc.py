#!/usr/bin/env python3
"""Render only the frozen full-cohort HCP state/order/attribution panels.

Reuses the 2026-10-22 caches without fitting models or recomputing statistics.
Panel b sums local SPT contributions by source-network count; it excludes the
within-network leaf terms. Panel c normalizes each participant before averaging.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/hcp_schaefer1000_full_replication_20261022"
OUTPUT = ROOT / "fig/hcp_schaefer1000_full_cohort_abc.png"
STATES = ("REST", "EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")
SHORT = ("REST", "EMO", "GAMB", "LANG", "MOTOR", "REL", "SOC", "WM")
NETWORKS = ("Vis", "SM", "DAN", "VAN", "Lim", "Cont", "DMN")
LN2 = float(np.log(2))
TOLERANCE_BITS = 1e-4  # Same native tolerance as the frozen experiment; no clipping.


def heatmap(ax, cax, values, rows, bar_label, *, vmax, fmt):
    im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xticks(range(8), SHORT, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    cb = ax.figure.colorbar(im, cax=cax)
    cb.set_label(bar_label)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            rgba = im.cmap(im.norm(values[i, j]))
            luminance = np.dot(rgba[:3], [.2126, .7152, .0722])
            ax.text(j, i, format(values[i, j], fmt), ha="center", va="center",
                    fontsize=13, color="black" if luminance > .55 else "white")


def main():
    manifest_bytes = (SOURCE / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    summary = json.loads((SOURCE / "main_statistics.json").read_text())
    assert summary["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    records = {}
    for sid in manifest["inputs"]:
        for state, row in json.loads((SOURCE / "subjects" / f"{sid}.json").read_text())["records"].items():
            records[sid, state] = row
    data = []
    for state in STATES:
        cohort = manifest["cohorts"][state]
        rows = [records[sid, state] for sid in cohort]
        for row in rows:
            for audit in row["numerical_audit"].values():
                if audit["tolerance_bits"] != TOLERANCE_BITS or audit["minimum_bits"] < -TOLERANCE_BITS:
                    raise ValueError(f"Syn validity mismatch: {state}/{row['subject']}: {audit}")
                if audit["significant_negative_count"] or audit["near_zero_negative_count"]:
                    raise ValueError(f"Unexpected negative estimates: {state}/{row['subject']}: {audit}")
        values = np.array([row["system_xi_bits"] for row in rows]) * LN2
        if len(values) != summary["descriptive"][state]["n"]:
            raise ValueError(f"Cohort size mismatch: {state}")
        np.testing.assert_allclose(values.mean(), summary["descriptive"][state]["mean_system_xi_nats"], atol=1e-12)
        data.append(values)
    mass = np.array([summary["descriptive"][s]["order_mass_mean_nats"] for s in STATES]).T
    percent = np.array([summary["descriptive"][s]["network_percent_mean"] for s in STATES]).T
    np.testing.assert_allclose(percent.sum(0), 100, atol=1e-10)
    if max(r["p_adjusted"] for r in summary["primary"]["A"]) >= 1e-4:
        raise ValueError("Annotation does not match the frozen BH-7 results")

    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
             "font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 14,
             "ytick.labelsize": 14, "axes.spines.top": False,
             "axes.spines.right": False, "axes.linewidth": .8}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(13.2, 8.4), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=.10, wspace=.13)
        ax = fig.add_subplot(grid[0, :])
        boxes = ax.boxplot(data, patch_artist=True, showfliers=False, widths=.52,
                           medianprops={"color": "#303030", "linewidth": 1},
                           whiskerprops={"linewidth": .8}, capprops={"linewidth": .8})
        rng = np.random.default_rng(20261022)  # Horizontal display jitter only.
        for i, (v, box) in enumerate(zip(data, boxes["boxes"]), 1):
            color = "#4C78A8" if i == 1 else "#D07A3A"
            box.set(facecolor=color, alpha=.18, edgecolor=color)
            ax.scatter(i + rng.uniform(-.17, .17, len(v)), v, s=3, alpha=.16,
                       color=color, linewidths=0, rasterized=True)
            ax.scatter(i, v.mean(), marker="D", s=25, color="white", edgecolor="#303030", zorder=5)
        ticks = [f"REST\nN={len(data[0]):,}"] + [
            f"{SHORT[i]}\nN={len(data[i]):,}\npaired={len(manifest['pairs'][state]):,}"
            for i, state in enumerate(STATES[1:], 1)]
        ax.set_xticks(range(1, 9), ticks)
        ax.set_ylim(0, max(map(np.max, data)) * 1.08)
        ax.set_ylabel(r"System $\Xi$ (nats)")
        ax.text(1, 1.035, "All available runs · diamonds: means · paired REST contrasts: BH-7, all q<0.0001",
                transform=ax.transAxes, ha="right", fontsize=12)
        lower = grid[1, :].subgridspec(1, 2, wspace=.10)
        left = lower[0].subgridspec(1, 2, width_ratios=[1, .035], wspace=.025)
        right = lower[1].subgridspec(1, 2, width_ratios=[1, .035], wspace=.025)
        axb = fig.add_subplot(left[0])
        heatmap(axb, fig.add_subplot(left[1]), mass, range(2, 8), "SPT contribution (nats)", vmax=mass.max(), fmt=".2f")
        axb.set_ylabel("Source networks per SPT node")
        axc = fig.add_subplot(right[0])
        heatmap(axc, fig.add_subplot(right[1]), percent, NETWORKS, r"Share of system $\Xi$ (%)", vmax=20, fmt=".1f")
        for panel, letter in zip([ax, axb, axc], "abc"):
            panel.text(-.055, 1.055, letter, transform=panel.transAxes,
                       fontweight="bold", fontsize=20, va="bottom", clip_on=False)
        OUTPUT.parent.mkdir(exist_ok=True)
        fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
