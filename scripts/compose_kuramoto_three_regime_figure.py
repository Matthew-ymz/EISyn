#!/usr/bin/env python3
"""Compose the selected three-regime Kuramoto hierarchy and module-atom figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from scripts.run_mixed_order_kuramoto_kout_main import (
    NAMES,
    _crop_white,
    render_network,
    tree_from_record,
)
from scripts.synergy_hierarchy_tree_plot import plot_synergy_hierarchy_tree


ROOT = Path(__file__).resolve().parents[1]
GLOBAL_FILES = (
    ROOT / "results/mixed_order_kuramoto_kout_main/learned_component025_regime_screen_kout_0.json",
    ROOT / "results/mixed_order_kuramoto_kout_main/learned_component025_regime_screen_kout_0p04.json",
    ROOT / "results/mixed_order_kuramoto_kout_main/strong_regime_screen_kout_5.json",
)
GLOBAL_CONTRACT = ROOT / "results/mixed_order_kuramoto_kout_main/learned_component025_regime_screen.json"
MODULE_FILE = ROOT / "results/mixed_order_kuramoto_kout_main/module_polynomial_metrics.json"
OUTPUT = ROOT / "docs/reports/assets/kuramoto_hierarchy/kuramoto_mixed_order_kout_complete_spt.png"
SUMMARY = ROOT / "results/mixed_order_kuramoto_kout_main/summary.json"


def main() -> None:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in GLOBAL_FILES]
    module_payload = json.loads(MODULE_FILE.read_text(encoding="utf-8"))
    module_by_k = {float(row["k_out"]): row["modules"] for row in module_payload["rows"]}
    trees = [tree_from_record(record) for record in records]
    work = OUTPUT.parent / ".three_regime_work"
    work.mkdir(parents=True, exist_ok=True)
    scale_max = max(
        float(atom["value_bits"])
        for record in records
        for atom in record["atoms"]
    )
    panels: list[tuple[Path, Path]] = []
    for index, (record, tree) in enumerate(zip(records, trees, strict=True)):
        network_path = render_network(
            work / f"network_{index}.png",
            pairwise_asymmetry=0.25,
            cross_coupling=float(record["k_out"]),
        )
        tree_path = plot_synergy_hierarchy_tree(
            tree,
            work / f"tree_{index}.png",
            source_labels={name: rf"$\theta_{{{i}}}$" for i, name in enumerate(NAMES, start=1)},
            decimals=2,
            syn_scale_max=scale_max,
            dpi=450,
        )
        panels.append((network_path, tree_path))

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(15.2, 10.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.82, 1.42, 0.78]},
    )
    axes[0, 0].set_title("Mixed-order network", fontsize=12, pad=8)
    axes[0, 1].set_title("Complete SPT hierarchy", fontsize=12, pad=8)
    axes[0, 2].set_title("Module-local atoms", fontsize=12, pad=8)
    pair_color = "#4477AA"
    triple_color = "#D9922E"
    for row, (record, paths) in enumerate(zip(records, panels, strict=True)):
        for axis, path in zip(axes[row, :2], paths, strict=True):
            axis.imshow(_crop_white(plt.imread(path)))
            axis.axis("off")
        axes[row, 0].text(
            -0.03,
            0.5,
            rf"$K_{{\mathrm{{out}}}}={float(record['k_out']):g}$",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=11,
            rotation=90,
            color="#24313C",
        )
        modules = module_by_k[float(record["k_out"])]
        pair_values = np.asarray(
            [modules["pairwise"]["pair_atom_bits"], modules["triadic"]["pair_atom_bits"]]
        )
        triple_values = np.asarray(
            [
                modules["pairwise"]["triple_residual_bits"],
                modules["triadic"]["triple_residual_bits"],
            ]
        )
        axis = axes[row, 2]
        y = np.asarray([1.0, 0.0])
        axis.barh(y, pair_values, color=pair_color, height=0.56, edgecolor="white")
        axis.barh(
            y,
            triple_values,
            left=pair_values,
            color=triple_color,
            height=0.56,
            edgecolor="white",
        )
        for y_value, pair_value, triple_value in zip(y, pair_values, triple_values, strict=True):
            if pair_value >= 0.18:
                axis.text(pair_value / 2.0, y_value, f"{pair_value:.2f}", ha="center", va="center", color="white", fontsize=8)
            else:
                axis.text(pair_value + 0.03, y_value + 0.27, f"{pair_value:.2f}", ha="left", va="bottom", color=pair_color, fontsize=8)
            axis.text(
                pair_value + triple_value / 2.0,
                y_value,
                f"{triple_value:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
        axis.set_yticks(y, (r"pairwise $\{1,2,3\}$", r"triadic $\{4,5,6\}$"))
        axis.set_xlim(0.0, 3.05)
        axis.set_xlabel("bits")
        axis.grid(axis="x", color="0.92", linewidth=0.6, zorder=0)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    figure.legend(
        handles=(
            Patch(facecolor=pair_color, label="pair atom"),
            Patch(facecolor=triple_color, label="triple residual"),
        ),
        loc="outside upper right",
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    for pair in panels:
        for path in pair:
            path.unlink(missing_ok=True)
    work.rmdir()

    contract_payload = json.loads(GLOBAL_CONTRACT.read_text(encoding="utf-8"))
    contract = dict(contract_payload["experiment_contract"])
    contract["k_out_values"] = [float(record["k_out"]) for record in records]
    contract["tm_source_context"] = (
        "the old symmetric circular features plus explicit pairwise, triadic, and "
        "candidate cross-edge Fourier terms, fixed across K_out"
    )
    contract["pipeline"] = [
        "generate paired noisy finite-time Kuramoto transitions",
        "fit one full-state MLP transition model per K_out with fixed architecture and budget",
        "evaluate the MLP under independent maximum-entropy phase interventions",
        "reconstruct the complete six-node future state and fit one conditional transport map per learned dependency component",
        "derive all 63 subset EIs from component-consistent marginals and build the free SPT",
    ]
    contract["estimator"] = (
        "MLP dynamics followed by one conditional neural transport map per learned "
        "dependency component; all 63 subset EIs use the same component rule and fixed estimator budget"
    )
    contract["dependency_component_rule"] = (
        "connect a source pair when its MLP permutation effect exceeds 0.25 output "
        "standard deviations; take connected components"
    )
    contract["module_local_estimator"] = module_payload["contract"]
    for record in records:
        record["module_local_polynomial_tm"] = module_by_k[float(record["k_out"])]
    SUMMARY.write_text(
        json.dumps(
            {
                "experiment_contract": contract,
                "conditions": records,
                "figure": str(OUTPUT),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
