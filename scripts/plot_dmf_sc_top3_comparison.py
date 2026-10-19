#!/usr/bin/env python3
"""Compare frozen top-three SC rankings with exact-member SPT recurrence.

No simulation, ranking refit, or new inference. The common bar readout is the
number of participants meeting the existing above-background recurrence rule.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/dmf_schaefer100/sc_topology_comparison"
DEST = ROOT / "fig/dmf_schaefer100/sc_topology_comparison/sc_spt_top3.png"
METHODS = [
    ("spt", "SPT recurrence"),
    ("internal_mean", "Mean internal weight"),
    ("clique_intensity", "Weighted clique intensity"),
    ("modularity_contribution", "Modularity contribution"),
]
TEAL, GRAY = "#167E78", "#9AA8B3"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_summary():
    summary = json.loads((BASE / "summary.json").read_text())
    verification = json.loads((BASE / "verification.json").read_text())
    contract = json.loads((BASE / "contract.json").read_text())
    if summary["status"] != "complete" or summary["subject_count"] != 93:
        raise ValueError("The completed 93-participant comparison is required")
    if verification["status"] != "passed":
        raise ValueError("Original native-data verification did not pass")
    for name, expected in verification["verified_cache_sha256"].items():
        if sha(BASE / name) != expected:
            raise ValueError(f"Verified cache changed: {name}")
    for name, expected in contract["provenance"].items():
        if name.endswith("_sha256"):
            continue
        if sha(ROOT / name) != expected:
            raise ValueError(f"Original analysis source changed: {name}")
    return summary


def main():
    summary = verified_summary()
    style = {
        "font.family": "DejaVu Sans", "font.size": 11.5,
        "axes.labelsize": 11.5, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(4, 2, figsize=(10.8, 8.8),
                                 layout="constrained", sharex=True)
        for col, order in enumerate((3, 4)):
            record = summary["orders"][str(order)]
            spt_sets = {tuple(r["members_one_based"])
                        for r in record["spt_top20"][:3]}
            for row, (method, title) in enumerate(METHODS):
                ax = axes[row, col]
                selected = (record["spt_top20"] if method == "spt"
                            else record["sc_top20"][method])[:3]
                counts = np.array([r["above_background"]["critical_ge4"]
                                   for r in selected])
                if np.any((counts < 0) | (counts > 93)):
                    raise ValueError("Invalid participant recurrence count")
                labels = [f"{i+1}   " + "{" + ", ".join(
                    str(x) for x in r["members_one_based"]) + "}"
                          for i, r in enumerate(selected)]
                colors = [TEAL if tuple(r["members_one_based"]) in spt_sets
                          else GRAY for r in selected]
                y = np.arange(3)
                ax.barh(y, counts, height=.62, color=colors, zorder=2)
                for yi, value, color in zip(y, counts, colors):
                    if value == 0:
                        ax.plot(0, yi, "|", color=color, markersize=11,
                                markeredgewidth=2, zorder=3, clip_on=False)
                    ax.text(value + 1.3, yi, str(value), va="center",
                            fontsize=11, color="#24313B")
                ax.set_yticks(y, labels, fontsize=11)
                ax.tick_params(axis="y", length=0, pad=6)
                ax.invert_yaxis()
                ax.set_xlim(0, 65)
                ax.set_xticks([0, 20, 40, 60])
                ax.grid(axis="x", color="#E6EAED", lw=.65, zorder=0)
                ax.set_axisbelow(True)
                panel = "abcdefgh"[2 * row + col]
                prefix = f"{order}-ROI coalitions\n" if row == 0 else ""
                ax.set_title(prefix + title, loc="left", pad=10)
                ax.text(-.035, 1.03, panel, transform=ax.transAxes,
                        va="bottom", ha="right", fontweight="bold",
                        fontsize=12)
                if row == 3:
                    ax.set_xlabel("Participants with stable SPT recurrence")
        fig.legend(handles=[
            Patch(facecolor=TEAL, label="Among the SPT top three of the same size"),
            Patch(facecolor=GRAY, label="Other structurally selected coalition"),
        ], loc="outside upper center", ncol=1, frameon=False, fontsize=11)
        DEST.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(DEST, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(json.dumps({"figure": str(DEST), "source": str(BASE / "summary.json"),
                      "source_sha256": sha(BASE / "summary.json"),
                      "participant_count": 93, "new_simulations": 0}))


if __name__ == "__main__":
    main()
