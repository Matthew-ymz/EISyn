"""Render the saved data-only TPM pilot; never fit or select a model."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.compose_spt_three_examples import draw_tree, validate_syn, GREEN, RED, GRAY

DATA = ROOT / "results/six_node_data_driven_tpm"
FIGURE = ROOT / "docs/reports/assets/kuramoto_hierarchy/data_driven_tpm_pilot.png"
ORDER = ("empirical", "smoothed", "mlp", "ctmc_prior", "oracle")
LABELS = ("Empirical TPM", "Smoothed TPM", "MLP + TPM", "CTMC (known family)", "True channel")
COLORS = ("#4477A8", "#B56828")


def load_results():
    records = [json.loads((DATA/f"n{count}_seed0.json").read_text()) for count in (4800, 48000)]
    for record in records:
        if record["seed"] != 0 or record["mechanism_prior_in_data_driven_fits"]:
            raise ValueError("This figure expects the data-only seed-0 pilot")
        for method in record["methods"].values():
            result = method["spt"]
            validate_syn(result["tree"], "syn_bits", record["syn_tolerance_bits"],
                         result["syn_audit"], method["root_xi_bits"])
    return records


def make_figure(records):
    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
             "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
             "axes.labelcolor": "#26333E", "text.color": "#26333E"}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(13.6, 7.3), constrained_layout=True)
        outer = fig.add_gridspec(3, 1, height_ratios=(1, 1.2, .12))
        upper = outer[0].subgridspec(1, 3, width_ratios=(1.18, 1, 1.15), wspace=.12)
        ax = fig.add_subplot(upper[0])
        values = np.full((len(ORDER), 2), np.nan)
        for j, record in enumerate(records):
            for i, name in enumerate(ORDER):
                if name in record["methods"]:
                    values[i, j] = record["methods"][name]["module_f1"]
        ax.imshow(np.ma.masked_invalid(values), cmap="GnBu", vmin=0, vmax=1, aspect="auto")
        for i in range(len(ORDER)):
            for j in range(2):
                value = values[i, j]
                ax.text(j, i, "not refit" if np.isnan(value) else f"{value:.2f}",
                        ha="center", va="center", color="white" if value > .7 else "#26333E",
                        fontsize=9)
        ax.set_yticks(range(len(ORDER)), LABELS)
        ax.set_xticks([0, 1], ["4,800 pairs", "48,000 pairs"])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title("a  Module F1 (higher is better)", loc="left", weight="bold", fontsize=10)
        for panel, key, xlabel in [(1, "mean_absolute_subset_ei_error_bits", "Mean absolute EI error (bits)"),
                                   (2, "heldout_nll_nats", "Test excess NLL (nats / pair)")]:
            ax = fig.add_subplot(upper[panel])
            for j, record in enumerate(records):
                xs, ys = [], []
                for i, name in enumerate(ORDER):
                    if name not in record["methods"]:
                        continue
                    method = record["methods"][name]
                    y = i + (j-.5)*.22
                    value = method[key]
                    if value is None:
                        ax.text(.95, y, r"$\infty$", transform=ax.get_yaxis_transform(),
                                ha="right", va="center", color=COLORS[j], fontsize=12)
                    else:
                        if key == "heldout_nll_nats":
                            value -= record["methods"]["oracle"][key]
                        xs.append(value)
                        ys.append(y)
                ax.scatter(xs, ys, s=38, marker="o" if j == 0 else "s", color=COLORS[j],
                           edgecolor="white", linewidth=.4,
                           label=f"{record['sample_count']:,} pairs", zorder=3)
            ax.set(yticks=range(5), yticklabels=[""]*5, ylim=(4.6, -.6), xlabel=xlabel)
            ax.set_xlim(left=-.004 if panel == 1 else -.006)
            ax.grid(axis="x", color="#E4E8EB", linewidth=.6, zorder=0)
            ax.tick_params(axis="y", length=0)
            title = "b  All 63 source subsets" if panel == 1 else "c  Independent endpoint prediction"
            ax.set_title(title, loc="left", weight="bold", fontsize=10)
            if panel == 2:
                ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), frameon=False, fontsize=9)

        lower = outer[1].subgridspec(1, 4, wspace=.06)
        cases = [(records[0], "mlp", "d  MLP · 4,800 pairs"),
                 (records[1], "empirical", "e  Empirical TPM · 48,000 pairs"),
                 (records[1], "smoothed", "f  Smoothed TPM · 48,000 pairs"),
                 (records[1], "mlp", "g  MLP · 48,000 pairs")]
        for index, (record, name, title) in enumerate(cases):
            ax = fig.add_subplot(lower[index])
            draw_tree(ax, record["methods"][name]["spt"]["tree"], [[1, 2, 3], [4, 5, 6]],
                      field="syn_bits")
            ax.set_title(title, loc="left", weight="bold", fontsize=9.3, pad=11)
        guide = fig.add_subplot(outer[2])
        guide.axis("off")
        guide.legend(handles=[Patch(facecolor="white", edgecolor=GREEN, label="Recovered module"),
                              Patch(facecolor="white", edgecolor=RED, label="Unsupported merge"),
                              Patch(facecolor="white", edgecolor=GRAY, label="Unscored refinement / root"),
                              Line2D([], [], color="none", label="SPT fill and %: local Syn / tree Xi")],
                     ncol=4, frameon=False, loc="center", fontsize=8.5, handlelength=1.1,
                     columnspacing=1.8)
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    make_figure(load_results())
    print(FIGURE)
