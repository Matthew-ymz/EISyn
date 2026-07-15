#!/usr/bin/env python3
"""Decompose critical-window DMF PhiEID with a Yeo-7 cortical partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_dmf_critical_phi_hierarchy_topology import (
    CONNECTIVITY_LABELS,
    CRITICAL_G,
    MAIN_CONFIRMATION,
    conditional_source_covariance,
    hierarchy_metrics,
    mean_sem,
    parse_float_list,
)
from scripts.plot_dmf_phi_eid_greedy_decomposition import load_region_labels
from scripts.run_dmf_diffusive_fullstate_control import rollout
from scripts.run_dmf_fixed_uniform_multihorizon import fixed_uniform_initial_state
from scripts.validate_dmf_83_region_oracle_phi_eid import (
    DEFAULT_SOURCE_RESULTS,
    load_dmf_module,
    resolve_path,
    standardize,
)


DEFAULT_OUTPUT = ROOT / "results" / "dmf_phi_eid_yeo7_hierarchy" / "critical_yeo7_hierarchy.npz"
DEFAULT_FIGURE = ROOT / "fig" / "dmf_phi_eid_yeo7_hierarchy.png"
DEFAULT_SUMMARY = ROOT / "results" / "dmf_phi_eid_yeo7_hierarchy" / "summary.json"
NETWORK_ORDER = (
    "Visual",
    "Somatomotor",
    "Dorsal attention",
    "Salience / ventral attention",
    "Limbic",
    "Frontoparietal control",
    "Default mode",
    "Non-cortical",
)
NETWORK_COLORS = (
    "#6A3D9A",
    "#1F78B4",
    "#33A02C",
    "#E31A1C",
    "#B15928",
    "#FF7F00",
    "#66A61E",
    "#8C8C8C",
)
COMPONENT_COLORS = ("#66CCEE", "#4477AA", "#AA3377")


# Majority-overlap assignment between FreeSurfer fsaverage5 aparc (Desikan--Killiany)
# and Yeo2011 7-network annotations. Subcortical and brain-stem labels are assigned
# below to a separate non-cortical group because Yeo-7 is cortical.
LEFT_DK_TO_YEO7 = {
    "cuneus": "Visual",
    "lateraloccipital": "Visual",
    "lingual": "Visual",
    "pericalcarine": "Visual",
    "bankssts": "Default mode",
    "entorhinal": "Limbic",
    "fusiform": "Visual",
    "inferiortemporal": "Limbic",
    "middletemporal": "Default mode",
    "parahippocampal": "Default mode",
    "superiortemporal": "Somatomotor",
    "temporalpole": "Limbic",
    "transversetemporal": "Somatomotor",
    "inferiorparietal": "Default mode",
    "postcentral": "Somatomotor",
    "precuneus": "Default mode",
    "superiorparietal": "Dorsal attention",
    "supramarginal": "Salience / ventral attention",
    "caudalanteriorcingulate": "Salience / ventral attention",
    "isthmuscingulate": "Default mode",
    "posteriorcingulate": "Salience / ventral attention",
    "rostralanteriorcingulate": "Default mode",
    "paracentral": "Somatomotor",
    "caudalmiddlefrontal": "Default mode",
    "frontalpole": "Limbic",
    "parsopercularis": "Salience / ventral attention",
    "parstriangularis": "Default mode",
    "precentral": "Somatomotor",
    "rostralmiddlefrontal": "Frontoparietal control",
    "superiorfrontal": "Default mode",
    "lateralorbitofrontal": "Limbic",
    "medialorbitofrontal": "Limbic",
    "parsorbitalis": "Default mode",
    "insula": "Salience / ventral attention",
}
RIGHT_DK_TO_YEO7 = {
    **LEFT_DK_TO_YEO7,
    "bankssts": "Somatomotor",
    "parahippocampal": "Visual",
    "caudalmiddlefrontal": "Frontoparietal control",
    "parstriangularis": "Salience / ventral attention",
}


def yeo7_membership(region_labels: list[str]) -> tuple[np.ndarray, list[list[int]]]:
    membership = np.empty(len(region_labels), dtype=int)
    groups = [[] for _ in NETWORK_ORDER]
    name_to_index = {name: index for index, name in enumerate(NETWORK_ORDER)}
    for roi, label in enumerate(region_labels):
        if label.startswith("ctx-lh-"):
            network = LEFT_DK_TO_YEO7[label.removeprefix("ctx-lh-")]
        elif label.startswith("ctx-rh-"):
            network = RIGHT_DK_TO_YEO7[label.removeprefix("ctx-rh-")]
        else:
            network = "Non-cortical"
        membership[roi] = name_to_index[network]
        groups[name_to_index[network]].append(roi)
    if any(not group for group in groups):
        raise ValueError("Every Yeo7 + non-cortical group must contain at least one ROI.")
    return membership, groups


def compute_payload(args: argparse.Namespace) -> dict[str, np.ndarray]:
    dmf = load_dmf_module()
    with np.load(resolve_path(args.main_confirmation)) as main:
        main_g = np.asarray(main["G"], dtype=float)
        seeds = np.asarray(main["seeds"], dtype=int)
        modes = [str(value) for value in np.asarray(main["modes"])]
        main_phi = np.asarray(main["phi_eid"], dtype=float)[modes.index("direct")]
        sample_count = int(np.asarray(main["sample_count"]).item())
        horizon = int(np.asarray(main["horizon"]).item())
        se_low = float(np.asarray(main["se_intervention_low"]).item())
        se_high = float(np.asarray(main["se_intervention_high"]).item())
        si_low = float(np.asarray(main["si_intervention_low"]).item())
        si_high = float(np.asarray(main["si_intervention_high"]).item())

    requested_g = np.asarray(args.critical_g, dtype=float)
    main_positions = np.asarray([int(np.flatnonzero(np.isclose(main_g, value))[0]) for value in requested_g])
    expected_phi = main_phi[:, main_positions]
    with np.load(resolve_path(args.source_results)) as source_archive:
        all_g = np.asarray(source_archive["G"], dtype=float)
        connectivity = np.asarray(source_archive["connectivity"], dtype=float)
        source_positions = np.asarray([int(np.flatnonzero(np.isclose(all_g, value))[0]) for value in requested_g])
        j_fic = np.asarray(source_archive["j_fic"], dtype=float)[source_positions]

    labels = load_region_labels(resolve_path(args.connectivity_labels), connectivity.shape[0])
    membership, groups = yeo7_membership(labels)
    roi_blocks = [(index, index + connectivity.shape[0]) for index in range(connectivity.shape[0])]
    parameters = dmf.DMFParameters(t_total=1.0, burn_in=0.0, dt=float(args.dt), sigma=float(args.sigma))
    shape = (len(seeds), len(requested_g))
    conditional_covariance = np.empty(shape + (2 * len(labels), 2 * len(labels)), dtype=float)
    fine_phi = np.empty(shape, dtype=float)
    within_roi = np.empty(shape, dtype=float)
    within_group = np.empty(shape, dtype=float)
    between_group = np.empty(shape, dtype=float)
    within_group_by_network = np.empty(shape + (len(groups),), dtype=float)

    for seed_index, seed in enumerate(seeds):
        for g_index, (coupling_g, source_position) in enumerate(zip(requested_g, source_positions)):
            source_rng = np.random.default_rng(int(seed) * 100_000 + int(source_position) * 1_000)
            source_se, source_si = fixed_uniform_initial_state(
                source_rng,
                sample_count=sample_count,
                dimension=len(labels),
                source_state="se_si",
                se_low=se_low,
                se_high=se_high,
                si_low=si_low,
                si_high=si_high,
            )
            if source_si is None:
                raise RuntimeError("The full-state protocol requires inhibitory interventions.")
            noise_rng = np.random.default_rng(int(seed) * 100_000 + int(source_position) * 1_000 + 17)
            target_se, target_si = rollout(
                dmf,
                source_se,
                source_si,
                connectivity=connectivity,
                coupling_g=float(coupling_g),
                j_fic=np.asarray(j_fic[g_index], dtype=float),
                parameters=parameters,
                mode="direct",
                state_boundary="none",
                horizon=horizon,
                rng=noise_rng,
            )
            source_z, _, _ = standardize(np.concatenate((source_se, source_si), axis=1))
            target_z, _, _ = standardize(np.concatenate((target_se, target_si), axis=1))
            _, conditional, metrics = conditional_source_covariance(source_z, target_z, ridge=float(args.ridge))
            hierarchy = hierarchy_metrics(conditional, roi_blocks, groups)
            conditional_covariance[seed_index, g_index] = conditional
            fine_phi[seed_index, g_index] = float(hierarchy["fine_phi"])
            within_roi[seed_index, g_index] = float(hierarchy["within_roi_total"])
            within_group[seed_index, g_index] = float(hierarchy["within_module_total"])
            between_group[seed_index, g_index] = float(hierarchy["between_module"])
            within_group_by_network[seed_index, g_index] = np.asarray(hierarchy["within_module"], dtype=float)
            print(f"seed={seed} G={coupling_g:.1f} Phi={fine_phi[seed_index, g_index]:.4f}", flush=True)
            if abs(float(metrics["raw_phi"]) - expected_phi[seed_index, g_index]) > float(args.validation_tolerance):
                raise RuntimeError("Recomputed PhiEID does not match the main confirmation.")

    hierarchy_error = np.max(np.abs(fine_phi - within_roi - within_group - between_group))
    if hierarchy_error > 1.0e-7:
        raise RuntimeError(f"Hierarchy additivity error is {hierarchy_error:.6g} bits.")
    return {
        "G": requested_g,
        "seeds": seeds,
        "region_labels": np.asarray(labels, dtype=object),
        "network_names": np.asarray(NETWORK_ORDER, dtype=object),
        "network_membership": membership,
        "network_sizes": np.asarray([len(group) for group in groups], dtype=int),
        "conditional_covariance": conditional_covariance,
        "expected_phi": expected_phi,
        "fine_phi": fine_phi,
        "within_roi": within_roi,
        "cross_roi": fine_phi - within_roi,
        "within_group_cross_roi": within_group,
        "between_groups": between_group,
        "within_group_by_network": within_group_by_network,
        "hierarchy_max_abs_error": np.asarray(float(hierarchy_error)),
    }


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "legend.frameon": False,
        }
    )


def plot_payload(payload: dict[str, np.ndarray], output: Path) -> None:
    configure_matplotlib()
    g_values = np.asarray(payload["G"], dtype=float)
    components = (
        ("Within ROI", np.asarray(payload["within_roi"]), COMPONENT_COLORS[0]),
        ("Cross-ROI, within functional group", np.asarray(payload["within_group_cross_roi"]), COMPONENT_COLORS[1]),
        ("Between functional groups", np.asarray(payload["between_groups"]), COMPONENT_COLORS[2]),
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.25), constrained_layout=True, gridspec_kw={"width_ratios": (1.25, 0.78, 1.28)})

    bottom = np.zeros_like(g_values)
    for label, values, color in components:
        mean, _ = mean_sem(values, axis=0)
        axes[0].bar(g_values, mean, width=0.075, bottom=bottom, color=color, label=label)
        bottom += mean
    total_mean, total_sem = mean_sem(np.asarray(payload["fine_phi"]), axis=0)
    axes[0].errorbar(g_values, total_mean, yerr=total_sem, color="black", marker="o", ms=3, lw=1, capsize=2)
    axes[0].set_xlabel("Global coupling $G$")
    axes[0].set_ylabel(r"$\Phi^{EID}$ components (bits)")
    axes[0].set_xticks(g_values)
    axes[0].grid(True, axis="y", color="0.90", lw=0.6)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=1)

    fine = np.asarray(payload["fine_phi"])
    roi_fraction = 100.0 * np.asarray(payload["within_roi"]) / fine
    cross_fraction = 100.0 - roi_fraction
    means = [float(roi_fraction.mean()), float(cross_fraction.mean())]
    sems = [float(roi_fraction.std(ddof=1) / np.sqrt(roi_fraction.size)), float(cross_fraction.std(ddof=1) / np.sqrt(cross_fraction.size))]
    axes[1].bar([0, 1], means, yerr=sems, color=(COMPONENT_COLORS[0], "#775DA6"), capsize=2, width=0.68)
    axes[1].set_xticks([0, 1], ["Within\nROI", "Cross\nROI"])
    axes[1].set_ylabel(r"Fraction of $\Phi^{EID}$ (%)")
    axes[1].set_ylim(0.0, 80.0)
    axes[1].grid(True, axis="y", color="0.90", lw=0.6)
    for index, value in enumerate(means):
        axes[1].text(index, value + 2.0, f"{value:.1f}%", ha="center", va="bottom")

    network_values = np.asarray(payload["within_group_by_network"]).mean(axis=(0, 1))
    order = np.argsort(network_values)
    y = np.arange(len(order))
    axes[2].barh(y, network_values[order], color=[NETWORK_COLORS[index] for index in order], height=0.68)
    axes[2].set_yticks(y, [str(payload["network_names"][index]) + f" (n={int(payload['network_sizes'][index])})" for index in order])
    axes[2].set_xlabel(r"Within-group cross-ROI $\Phi^{EID}$ (bits)")
    axes[2].grid(True, axis="x", color="0.90", lw=0.6)

    for panel, axis in zip(("A", "B", "C"), axes):
        axis.text(-0.16, 1.05, panel, transform=axis.transAxes, fontsize=11, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=450, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_summary(payload: dict[str, np.ndarray], path: Path) -> None:
    fine = np.asarray(payload["fine_phi"], dtype=float)
    within_roi = np.asarray(payload["within_roi"], dtype=float)
    cross_roi = np.asarray(payload["cross_roi"], dtype=float)
    within_group = np.asarray(payload["within_group_cross_roi"], dtype=float)
    between = np.asarray(payload["between_groups"], dtype=float)
    by_network = np.asarray(payload["within_group_by_network"], dtype=float).mean(axis=(0, 1))
    summary = {
        "experiment_contract": {
            "treatment": "mesoscale grouping changed from structural Louvain to cortical Yeo-7 plus a non-cortical group",
            "fixed": "G, seeds, intervention samples, dynamics, horizon, target, and Gaussian conditional-total-correlation estimator",
            "G": np.asarray(payload["G"], dtype=float).tolist(),
            "seeds": np.asarray(payload["seeds"], dtype=int).tolist(),
            "conditions": int(fine.size),
        },
        "mapping": {
            "method": "maximum fsaverage5 surface-vertex overlap between Desikan-Killiany aparc and Yeo2011 7-network annotations",
            "yeo7_annotation_source": "ThomasYeoLab/CBIG Yeo2011 fsaverage5 split-label annotations",
            "desikan_annotation_source": "FreeSurfer fsaverage5 aparc annotations",
            "cortical_roi_count": int(np.sum(np.asarray(payload["network_membership"]) < 7)),
            "noncortical_roi_count": int(np.sum(np.asarray(payload["network_membership"]) == 7)),
            "network_sizes": {str(name): int(size) for name, size in zip(payload["network_names"], payload["network_sizes"])},
            "region_assignments": {
                str(region): str(payload["network_names"][int(network)])
                for region, network in zip(payload["region_labels"], payload["network_membership"])
            },
        },
        "critical_window_mean": {
            "fine_phi_bits": float(fine.mean()),
            "within_roi_bits": float(within_roi.mean()),
            "within_roi_fraction": float((within_roi / fine).mean()),
            "cross_roi_bits": float(cross_roi.mean()),
            "cross_roi_fraction": float((cross_roi / fine).mean()),
            "within_functional_group_cross_roi_bits": float(within_group.mean()),
            "within_functional_group_fraction": float((within_group / fine).mean()),
            "between_functional_groups_bits": float(between.mean()),
            "between_functional_groups_fraction": float((between / fine).mean()),
        },
        "within_group_cross_roi_bits": {
            str(name): float(value) for name, value in zip(payload["network_names"], by_network)
        },
        "validation": {
            "hierarchy_max_abs_error_bits": float(payload["hierarchy_max_abs_error"]),
            "all_24_conditions_cross_roi_exceeds_within_roi": bool(np.all(cross_roi > within_roi)),
            "all_24_conditions_between_groups_exceeds_within_group_cross_roi": bool(np.all(between > within_group)),
        },
        "interpretation_boundary": (
            "Yeo-7 is cortical. The 15 subcortical/brain-stem ROIs are retained as one explicit non-cortical group. "
            "Majority-overlap assignments compress spatially heterogeneous Desikan parcels to one network label."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decompose critical DMF PhiEID using Yeo-7 functional groups.")
    parser.add_argument("--main-confirmation", type=Path, default=MAIN_CONFIRMATION)
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--connectivity-labels", type=Path, default=CONNECTIVITY_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--critical-g", type=parse_float_list, default=CRITICAL_G)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--validation-tolerance", type=float, default=1.0e-7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = compute_payload(args)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    plot_payload(payload, resolve_path(args.figure))
    write_summary(payload, resolve_path(args.summary))
    print(f"Saved results: {output}")
    print(f"Saved figure: {resolve_path(args.figure)}")
    print(f"Saved summary: {resolve_path(args.summary)}")


if __name__ == "__main__":
    main()
