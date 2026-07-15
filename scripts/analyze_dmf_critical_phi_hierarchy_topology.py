from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_dmf_phi_eid_greedy_decomposition import load_region_labels
from scripts.run_dmf_diffusive_fullstate_control import rollout
from scripts.run_dmf_fixed_uniform_multihorizon import fixed_uniform_initial_state
from scripts.validate_dmf_83_region_oracle_phi_eid import (
    DEFAULT_SOURCE_RESULTS,
    gaussian_singleton_source_phi,
    load_dmf_module,
    resolve_path,
    safe_logdet_psd,
    standardize,
)


MAIN_CONFIRMATION = (
    ROOT
    / "results"
    / "dmf_fullstate_uniform_support"
    / "confirm_c050_h020_tau300_n2048_no_clip_seeds3_10.npz"
)
CONNECTIVITY_LABELS = ROOT / "exp" / "brain" / "result_lausanne_fig6" / "count_00_connectivity.csv"
DEFAULT_OUTPUT = ROOT / "results" / "dmf_phi_eid_hierarchical_topology" / "critical_hierarchy.npz"
DEFAULT_FIGURE = ROOT / "fig" / "dmf_phi_eid_hierarchical_topology.png"
DEFAULT_CROSS_FIGURE = ROOT / "fig" / "dmf_cross_roi_coupling_vs_structural_strength.png"
DEFAULT_COMPARISON_FIGURE = ROOT / "fig" / "dmf_local_cross_roi_coupling_vs_strength.png"
DEFAULT_SUMMARY = ROOT / "results" / "dmf_phi_eid_hierarchical_topology" / "summary.json"
CRITICAL_G = (1.6, 1.7, 1.8)
COMMUNITY_COLORS = (
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
    "#AA3377", "#BBBBBB", "#EE8866", "#44AA99", "#999933",
)


def parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())


def conditional_source_covariance(
    source: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    source_array = np.asarray(source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    empirical_source_cov = np.cov(source_array, rowvar=False, bias=False)
    source_cov = np.diag(np.diag(np.atleast_2d(empirical_source_cov)))
    source_cov += float(ridge) * np.eye(source_cov.shape[0])

    coefficient, *_ = np.linalg.lstsq(source_array, target_array, rcond=None)
    transition = coefficient.T
    residual = target_array - source_array @ coefficient
    noise_cov = np.cov(residual, rowvar=False, bias=False)
    noise_cov = np.atleast_2d(noise_cov) + float(ridge) * np.eye(target_array.shape[1])
    target_cov = transition @ source_cov @ transition.T + noise_cov
    target_cov = 0.5 * (target_cov + target_cov.T) + float(ridge) * np.eye(target_array.shape[1])
    source_target_cov = source_cov @ transition.T
    conditional = source_cov - source_target_cov @ np.linalg.pinv(target_cov) @ source_target_cov.T
    conditional = 0.5 * (conditional + conditional.T) + float(ridge) * np.eye(source_cov.shape[0])
    metrics = gaussian_singleton_source_phi(source_array, target_array, ridge=float(ridge))
    return source_cov, conditional, metrics


def logdet_minor(matrix: np.ndarray, indices: Sequence[int]) -> float:
    selected = np.asarray(tuple(map(int, indices)), dtype=int)
    return safe_logdet_psd(np.asarray(matrix, dtype=float)[np.ix_(selected, selected)])


def conditional_total_correlation(
    conditional: np.ndarray,
    blocks: Sequence[Sequence[int]],
) -> float:
    return float(
        0.5
        * (
            sum(logdet_minor(conditional, block) for block in blocks)
            - safe_logdet_psd(conditional)
        )
        / math.log(2.0)
    )


def block_cross_leverage(
    conditional: np.ndarray,
    blocks: Sequence[Sequence[int]],
) -> np.ndarray:
    """Leave-one-block-out reduction in between-block conditional total correlation."""
    inverse = np.linalg.pinv(np.asarray(conditional, dtype=float))
    values = np.empty(len(blocks), dtype=float)
    for index, block in enumerate(blocks):
        block_indices = tuple(map(int, block))
        value = 0.5 * (
            logdet_minor(conditional, block_indices)
            + logdet_minor(inverse, block_indices)
        ) / math.log(2.0)
        values[index] = max(0.0, float(value))
    return values


def hierarchy_metrics(
    conditional: np.ndarray,
    roi_blocks: Sequence[Sequence[int]],
    module_roi_indices: Sequence[Sequence[int]],
) -> dict[str, np.ndarray | float]:
    singleton_blocks = [(index,) for index in range(conditional.shape[0])]
    fine_phi = conditional_total_correlation(conditional, singleton_blocks)
    within_roi = np.asarray(
        [conditional_total_correlation(conditional[np.ix_(block, block)], [(0,), (1,)]) for block in roi_blocks],
        dtype=float,
    )
    between_roi = conditional_total_correlation(conditional, roi_blocks)

    module_source_blocks = [
        tuple(source for roi in module for source in roi_blocks[int(roi)])
        for module in module_roi_indices
    ]
    within_module = np.empty(len(module_roi_indices), dtype=float)
    for module_index, roi_indices in enumerate(module_roi_indices):
        source_indices = module_source_blocks[module_index]
        local = conditional[np.ix_(source_indices, source_indices)]
        local_blocks = []
        offset = 0
        for _ in roi_indices:
            local_blocks.append((offset, offset + 1))
            offset += 2
        within_module[module_index] = conditional_total_correlation(local, local_blocks)
    between_module = conditional_total_correlation(conditional, module_source_blocks)

    roi_cross = block_cross_leverage(conditional, roi_blocks)
    module_cross = block_cross_leverage(conditional, module_source_blocks)
    return {
        "fine_phi": float(fine_phi),
        "within_roi": within_roi,
        "within_roi_total": float(within_roi.sum()),
        "between_roi": float(between_roi),
        "within_module": within_module,
        "within_module_total": float(within_module.sum()),
        "between_module": float(between_module),
        "roi_cross_leverage": roi_cross,
        "roi_involvement": within_roi + roi_cross,
        "module_cross_leverage": module_cross,
        "module_involvement": within_module + module_cross,
    }


def structural_communities(connectivity: np.ndarray, *, seed: int) -> tuple[np.ndarray, list[list[int]]]:
    symmetric = 0.5 * (np.asarray(connectivity, dtype=float) + np.asarray(connectivity, dtype=float).T)
    np.fill_diagonal(symmetric, 0.0)
    graph = nx.from_numpy_array(symmetric)
    communities = nx.community.louvain_communities(graph, weight="weight", seed=int(seed), resolution=1.0)
    ordered = sorted((sorted(map(int, community)) for community in communities), key=lambda x: (-len(x), x[0]))
    membership = np.empty(symmetric.shape[0], dtype=int)
    for module_index, nodes in enumerate(ordered):
        membership[nodes] = module_index
    return membership, ordered


def topology_metrics(connectivity: np.ndarray, membership: np.ndarray) -> dict[str, np.ndarray]:
    matrix = 0.5 * (np.asarray(connectivity, dtype=float) + np.asarray(connectivity, dtype=float).T)
    np.fill_diagonal(matrix, 0.0)
    strength = matrix.sum(axis=1)
    modules = np.unique(membership)
    module_strength = np.column_stack([matrix[:, membership == module].sum(axis=1) for module in modules])
    participation = np.zeros(matrix.shape[0], dtype=float)
    valid = strength > 0.0
    participation[valid] = 1.0 - np.sum(
        (module_strength[valid] / strength[valid, None]) ** 2,
        axis=1,
    )
    within_strength = module_strength[np.arange(matrix.shape[0]), membership]
    within_z = np.zeros(matrix.shape[0], dtype=float)
    for module in modules:
        mask = membership == module
        scale = within_strength[mask].std(ddof=1) if int(mask.sum()) > 1 else 0.0
        if scale > 0.0:
            within_z[mask] = (within_strength[mask] - within_strength[mask].mean()) / scale
    return {
        "strength": strength,
        "participation": participation,
        "within_strength": within_strength,
        "within_z": within_z,
        "module_strength": module_strength,
    }


def spearman_summary(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    result = stats.spearmanr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return {"rho": float(result.statistic), "p": float(result.pvalue)}


def compute_payload(args: argparse.Namespace) -> dict[str, np.ndarray]:
    dmf = load_dmf_module()
    with np.load(resolve_path(args.main_confirmation), allow_pickle=True) as main:
        main_g = np.asarray(main["G"], dtype=float)
        seeds = np.asarray(main["seeds"], dtype=int)
        modes = [str(item) for item in main["modes"]]
        main_phi = np.asarray(main["phi_eid"], dtype=float)[modes.index("direct")]
        sample_count = int(np.asarray(main["sample_count"]).item())
        horizon = int(np.asarray(main["horizon"]).item())
        se_low = float(np.asarray(main["se_intervention_low"]).item())
        se_high = float(np.asarray(main["se_intervention_high"]).item())
        si_low = float(np.asarray(main["si_intervention_low"]).item())
        si_high = float(np.asarray(main["si_intervention_high"]).item())

    requested_g = np.asarray(args.critical_g, dtype=float)
    main_positions = np.asarray([int(np.flatnonzero(np.isclose(main_g, value))[0]) for value in requested_g], dtype=int)
    expected_phi = main_phi[:, main_positions]

    with np.load(resolve_path(args.source_results)) as source_archive:
        all_g = np.asarray(source_archive["G"], dtype=float)
        connectivity = np.asarray(source_archive["connectivity"], dtype=float)
        source_positions = np.asarray(
            [int(np.flatnonzero(np.isclose(all_g, value))[0]) for value in requested_g], dtype=int,
        )
        j_fic = np.asarray(source_archive["j_fic"], dtype=float)[source_positions]

    labels = load_region_labels(resolve_path(args.connectivity_labels), connectivity.shape[0])
    membership, communities = structural_communities(connectivity, seed=int(args.community_seed))
    roi_blocks = [(index, index + connectivity.shape[0]) for index in range(connectivity.shape[0])]
    parameters = dmf.DMFParameters(t_total=1.0, burn_in=0.0, dt=float(args.dt), sigma=float(args.sigma))

    n_seed, n_g, n_roi, n_module = len(seeds), len(requested_g), len(labels), len(communities)
    scalar_names = ("fine_phi", "within_roi_total", "between_roi", "within_module_total", "between_module")
    scalars = {name: np.empty((n_seed, n_g), dtype=float) for name in scalar_names}
    roi_names = ("within_roi", "roi_cross_leverage", "roi_involvement")
    roi_values = {name: np.empty((n_seed, n_g, n_roi), dtype=float) for name in roi_names}
    module_names = ("within_module", "module_cross_leverage", "module_involvement")
    module_values = {name: np.empty((n_seed, n_g, n_module), dtype=float) for name in module_names}
    estimator_phi = np.empty((n_seed, n_g), dtype=float)

    for seed_index, seed in enumerate(seeds):
        for g_index, (coupling_g, source_position) in enumerate(zip(requested_g, source_positions)):
            source_rng = np.random.default_rng(int(seed) * 100_000 + int(source_position) * 1_000)
            source_se, source_si_optional = fixed_uniform_initial_state(
                source_rng,
                sample_count=sample_count,
                dimension=n_roi,
                source_state="se_si",
                se_low=se_low,
                se_high=se_high,
                si_low=si_low,
                si_high=si_high,
            )
            if source_si_optional is None:
                raise RuntimeError("The full-state protocol requires inhibitory interventions.")
            noise_rng = np.random.default_rng(int(seed) * 100_000 + int(source_position) * 1_000 + 17)
            target_se, target_si = rollout(
                dmf,
                source_se,
                source_si_optional,
                connectivity=connectivity,
                coupling_g=float(coupling_g),
                j_fic=np.asarray(j_fic[g_index], dtype=float),
                parameters=parameters,
                mode="direct",
                state_boundary="none",
                horizon=horizon,
                rng=noise_rng,
            )
            source_z, _, _ = standardize(np.concatenate((source_se, source_si_optional), axis=1))
            target_z, _, _ = standardize(np.concatenate((target_se, target_si), axis=1))
            _, conditional, metrics = conditional_source_covariance(source_z, target_z, ridge=float(args.ridge))
            hierarchy = hierarchy_metrics(conditional, roi_blocks, communities)
            estimator_phi[seed_index, g_index] = float(metrics["raw_phi"])
            for name in scalar_names:
                scalars[name][seed_index, g_index] = float(hierarchy[name])
            for name in roi_names:
                roi_values[name][seed_index, g_index] = np.asarray(hierarchy[name], dtype=float)
            for name in module_names:
                module_values[name][seed_index, g_index] = np.asarray(hierarchy[name], dtype=float)
            print(
                f"seed={seed} G={coupling_g:.1f} Phi={scalars['fine_phi'][seed_index, g_index]:.4f}",
                flush=True,
            )

    max_abs_error = float(np.max(np.abs(estimator_phi - expected_phi)))
    if max_abs_error > float(args.validation_tolerance):
        raise RuntimeError(
            f"Recomputed Phi differs from the main confirmation by {max_abs_error:.6g} bits; "
            f"tolerance is {args.validation_tolerance:.6g}."
        )
    hierarchy_error = np.max(
        np.abs(scalars["fine_phi"] - (scalars["within_roi_total"] + scalars["within_module_total"] + scalars["between_module"]))
    )
    if hierarchy_error > 1.0e-7:
        raise RuntimeError(f"Hierarchy additivity error is {hierarchy_error:.6g} bits.")

    topology = topology_metrics(connectivity, membership)
    return {
        "G": requested_g,
        "seeds": seeds,
        "region_labels": np.asarray(labels, dtype=object),
        "connectivity": connectivity,
        "community_membership": membership,
        "community_sizes": np.asarray([len(nodes) for nodes in communities], dtype=int),
        "expected_phi": expected_phi,
        "estimator_phi": estimator_phi,
        "validation_max_abs_error": np.asarray(max_abs_error),
        "hierarchy_max_abs_error": np.asarray(float(hierarchy_error)),
        **scalars,
        **roi_values,
        **module_values,
        **topology,
    }


def mean_sem(values: np.ndarray, axis: int | tuple[int, ...] = 0) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    count = np.prod([array.shape[index] for index in ((axis,) if isinstance(axis, int) else axis)])
    return np.mean(array, axis=axis), np.std(array, axis=axis, ddof=1) / np.sqrt(float(count))


def short_label(label: str) -> str:
    replacements = {"ctx-lh-": "L-", "ctx-rh-": "R-", "Left-": "L-", "Right-": "R-"}
    result = str(label)
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result.replace("-Proper", "")


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


def plot_payload(payload: dict[str, np.ndarray], output: Path, *, top_n: int = 15) -> None:
    configure_matplotlib()
    g_values = np.asarray(payload["G"], dtype=float)
    labels = np.asarray(payload["region_labels"], dtype=object)
    membership = np.asarray(payload["community_membership"], dtype=int)
    connectivity = np.asarray(payload["connectivity"], dtype=float)
    involvement = np.asarray(payload["roi_involvement"], dtype=float)
    score_mean = involvement.mean(axis=(0, 1))
    local_ei_coupling = np.asarray(payload["within_roi"], dtype=float).mean(axis=(0, 1))
    score_sem = involvement.reshape(-1, involvement.shape[-1]).std(axis=0, ddof=1) / np.sqrt(involvement.shape[0] * involvement.shape[1])
    strength = np.asarray(payload["strength"], dtype=float)
    participation = np.asarray(payload["participation"], dtype=float)
    module_involvement = np.asarray(payload["module_involvement"], dtype=float).mean(axis=(0, 1))
    within_module = np.asarray(payload["within_module"], dtype=float).mean(axis=(0, 1))
    module_cross = np.asarray(payload["module_cross_leverage"], dtype=float).mean(axis=(0, 1))

    figure = plt.figure(figsize=(11.8, 7.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1.0, 1.15, 1.25), height_ratios=(1.0, 1.0))
    ax_hierarchy = figure.add_subplot(grid[0, 0])
    ax_roi = figure.add_subplot(grid[0, 1])
    ax_matrix = figure.add_subplot(grid[:, 2])
    ax_module = figure.add_subplot(grid[1, 0])
    ax_topology = figure.add_subplot(grid[1, 1])

    components = (
        ("Within ROI", np.asarray(payload["within_roi_total"]), "#66CCEE"),
        ("Within structural module", np.asarray(payload["within_module_total"]), "#4477AA"),
        ("Between structural modules", np.asarray(payload["between_module"]), "#AA3377"),
    )
    bottom = np.zeros_like(g_values)
    for label, values, color in components:
        mean, sem = mean_sem(values, axis=0)
        ax_hierarchy.bar(g_values, mean, width=0.075, bottom=bottom, color=color, label=label)
        bottom += mean
    fine_mean, fine_sem = mean_sem(np.asarray(payload["fine_phi"]), axis=0)
    ax_hierarchy.errorbar(g_values, fine_mean, yerr=fine_sem, color="black", marker="o", ms=3, lw=1, capsize=2)
    ax_hierarchy.set_xlabel("Global coupling $G$")
    ax_hierarchy.set_ylabel(r"$\Phi^{EID}$ components (bits)")
    ax_hierarchy.set_xticks(g_values)
    ax_hierarchy.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=1)
    ax_hierarchy.grid(True, axis="y", color="0.90", lw=0.6)

    top_indices = np.argsort(score_mean)[-int(top_n):]
    y_positions = np.arange(top_indices.size)
    bar_colors = [COMMUNITY_COLORS[membership[index] % len(COMMUNITY_COLORS)] for index in top_indices]
    ax_roi.barh(y_positions, score_mean[top_indices], xerr=score_sem[top_indices], color=bar_colors, height=0.68)
    ax_roi.set_yticks(y_positions, [short_label(labels[index]) for index in top_indices])
    ax_roi.set_xlabel("ROI involvement (bits; mean ± SEM)")
    ax_roi.grid(True, axis="x", color="0.90", lw=0.6)

    matrix = 0.5 * (connectivity + connectivity.T)
    order = np.asarray(sorted(range(len(labels)), key=lambda index: (membership[index], -score_mean[index])), dtype=int)
    ordered = matrix[np.ix_(order, order)]
    positive = ordered[ordered > 0.0]
    vmax = float(np.quantile(positive, 0.985)) if positive.size else 1.0
    image = ax_matrix.imshow(ordered, cmap="magma_r", vmin=0.0, vmax=vmax, interpolation="nearest", aspect="equal")
    ordered_membership = membership[order]
    boundaries = np.flatnonzero(np.diff(ordered_membership)) + 0.5
    for boundary in boundaries:
        ax_matrix.axhline(boundary, color="white", lw=0.55)
        ax_matrix.axvline(boundary, color="white", lw=0.55)
    centers = []
    module_ids = []
    for module in np.unique(ordered_membership):
        locations = np.flatnonzero(ordered_membership == module)
        centers.append(float(locations.mean()))
        module_ids.append(int(module))
    matrix_module_labels = [
        f"M{module + 1}" if int(payload["community_sizes"][module]) > 1 else ""
        for module in module_ids
    ]
    ax_matrix.set_xticks(centers, matrix_module_labels, rotation=90)
    ax_matrix.set_yticks(centers, matrix_module_labels)
    ax_matrix.set_xlabel("ROI ordered by structural module and contribution")
    ax_matrix.set_ylabel("ROI ordered by structural module and contribution")
    figure.colorbar(image, ax=ax_matrix, fraction=0.035, pad=0.02, label="Structural weight")

    module_order = np.argsort(module_involvement)
    module_y = np.arange(module_order.size)
    ax_module.barh(module_y, within_module[module_order], color="#4477AA", label="Within-module")
    ax_module.barh(
        module_y,
        module_cross[module_order],
        left=within_module[module_order],
        color="#AA3377",
        label="Cross-module leverage",
    )
    ax_module.set_yticks(module_y, [f"M{index + 1} (n={int(payload['community_sizes'][index])})" for index in module_order])
    ax_module.set_xlabel("Module involvement (bits)")
    ax_module.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2)
    ax_module.grid(True, axis="x", color="0.90", lw=0.6)

    # Community membership is not part of this association test.  A single neutral
    # encoding keeps attention on the ROI-level strength--local E/I coupling relationship.
    ax_topology.scatter(
        strength,
        local_ei_coupling,
        s=23,
        color="#4C78A8",
        alpha=0.85,
    )
    correlation = spearman_summary(strength, local_ei_coupling)
    ax_topology.text(
        0.03,
        0.97,
        rf"Spearman $\rho={correlation['rho']:.2f}$",
        transform=ax_topology.transAxes,
        va="top",
    )
    ax_topology.text(
        0.50,
        1.10,
        r"$\phi^{\mathrm{EID}}_{r,E/I}=I(E_{r,t};I_{r,t}\mid\mathbf{X}_{t+\tau})$",
        transform=ax_topology.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.0,
    )
    ax_topology.set_xlabel("Weighted structural strength")
    ax_topology.set_ylabel(r"Local $\phi^{\mathrm{EID}}_{r,E/I}$ (bits)")
    ax_topology.grid(True, color="0.90", lw=0.6)

    for panel, axis in zip(("A", "B", "C", "D", "E"), (ax_hierarchy, ax_roi, ax_matrix, ax_module, ax_topology)):
        axis.text(-0.14, 1.05, panel, transform=axis.transAxes, fontsize=11, fontweight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=450, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_local_ei_vs_strength(payload: dict[str, np.ndarray], output: Path) -> None:
    """Plot the local E/I conditional coupling against structural strength."""
    configure_matplotlib()
    strength = np.asarray(payload["strength"], dtype=float)
    local_ei_coupling = np.asarray(payload["within_roi"], dtype=float).mean(axis=(0, 1))
    correlation = spearman_summary(strength, local_ei_coupling)

    figure, axis = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    axis.scatter(strength, local_ei_coupling, s=27, color="#4C78A8", alpha=0.85)
    axis.text(
        0.0,
        1.03,
        rf"Spearman $\rho={correlation['rho']:.2f}$, $n={strength.size}$",
        transform=axis.transAxes,
        va="bottom",
        clip_on=False,
    )
    axis.set_xlabel("Weighted structural strength")
    axis.set_ylabel("Local E/I conditional coupling (bits)")
    axis.grid(True, color="0.90", lw=0.6)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=450, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_cross_roi_vs_strength(payload: dict[str, np.ndarray], output: Path) -> None:
    """Plot each ROI's coupling to all other ROIs against structural strength."""
    configure_matplotlib()
    strength = np.asarray(payload["strength"], dtype=float)
    cross_roi_coupling = np.asarray(payload["roi_cross_leverage"], dtype=float).mean(axis=(0, 1))
    correlation = spearman_summary(strength, cross_roi_coupling)

    figure, axis = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    axis.scatter(strength, cross_roi_coupling, s=27, color="#4C78A8", alpha=0.85)
    axis.text(
        0.0,
        1.03,
        rf"Spearman $\rho={correlation['rho']:.2f}$, $n={strength.size}$",
        transform=axis.transAxes,
        va="bottom",
        clip_on=False,
    )
    axis.set_xlabel("Weighted structural strength")
    axis.set_ylabel("Cross-ROI conditional coupling (bits)")
    axis.grid(True, color="0.90", lw=0.6)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=450, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_local_cross_roi_comparison(payload: dict[str, np.ndarray], output: Path) -> None:
    """Compare local E/I and cross-ROI coupling against structural strength."""
    configure_matplotlib()
    strength = np.asarray(payload["strength"], dtype=float)
    local_ei_coupling = np.asarray(payload["within_roi"], dtype=float).mean(axis=(0, 1))
    cross_roi_coupling = np.asarray(payload["roi_cross_leverage"], dtype=float).mean(axis=(0, 1))
    panels = (
        ("a", local_ei_coupling, "Local E/I conditional coupling (bits)"),
        ("b", cross_roi_coupling, "Cross-ROI conditional coupling (bits)"),
    )

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)
    for axis, (panel, values, y_label) in zip(axes, panels):
        correlation = spearman_summary(strength, values)
        axis.scatter(strength, values, s=27, color="#4C78A8", alpha=0.85)
        axis.text(
            0.0,
            1.03,
            rf"Spearman $\rho={correlation['rho']:.2f}$, $n={strength.size}$",
            transform=axis.transAxes,
            va="bottom",
            clip_on=False,
        )
        axis.text(-0.16, 1.05, panel, transform=axis.transAxes, fontsize=11, fontweight="bold")
        axis.set_xlabel("Weighted structural strength")
        axis.set_ylabel(y_label)
        axis.grid(True, color="0.90", lw=0.6)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=450, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_summary(payload: dict[str, np.ndarray], path: Path) -> None:
    labels = np.asarray(payload["region_labels"], dtype=object)
    membership = np.asarray(payload["community_membership"], dtype=int)
    score = np.asarray(payload["roi_involvement"], dtype=float).mean(axis=(0, 1))
    score_conditions = np.asarray(payload["roi_involvement"], dtype=float).reshape(-1, len(labels))
    top15_frequency = np.zeros(len(labels), dtype=int)
    for condition in score_conditions:
        top15_frequency[np.argsort(condition)[-15:]] += 1
    rank_agreements = []
    for left in range(score_conditions.shape[0] - 1):
        for right in range(left + 1, score_conditions.shape[0]):
            rank_agreements.append(float(stats.spearmanr(score_conditions[left], score_conditions[right]).statistic))
    strength = np.asarray(payload["strength"], dtype=float)
    participation = np.asarray(payload["participation"], dtype=float)
    local_ei_coupling = np.asarray(payload["within_roi"], dtype=float).mean(axis=(0, 1))
    cross_roi_coupling = np.asarray(payload["roi_cross_leverage"], dtype=float).mean(axis=(0, 1))
    connectivity = 0.5 * (
        np.asarray(payload["connectivity"], dtype=float)
        + np.asarray(payload["connectivity"], dtype=float).T
    )
    np.fill_diagonal(connectivity, 0.0)
    top = np.argsort(score)[::-1][:15]
    module_score = np.asarray(payload["module_involvement"], dtype=float).mean(axis=(0, 1))
    summary = {
        "experiment_contract": {
            "G": np.asarray(payload["G"], dtype=float).tolist(),
            "seeds": np.asarray(payload["seeds"], dtype=int).tolist(),
            "sample_count": 2048,
            "intervention": "independent U(0.30,0.70)^166",
            "horizon_steps": 300,
            "state_boundary": "none",
            "estimator": "Gaussian conditional total correlation",
        },
        "validation": {
            "main_phi_max_abs_error_bits": float(payload["validation_max_abs_error"]),
            "hierarchy_max_abs_error_bits": float(payload["hierarchy_max_abs_error"]),
        },
        "hierarchy_mean_bits": {
            "fine_phi": float(np.mean(payload["fine_phi"])),
            "within_roi": float(np.mean(payload["within_roi_total"])),
            "within_structural_module": float(np.mean(payload["within_module_total"])),
            "between_structural_modules": float(np.mean(payload["between_module"])),
        },
        "top_regions": [
            {
                "rank": rank + 1,
                "region": str(labels[index]),
                "module": f"M{int(membership[index]) + 1}",
                "involvement_bits": float(score[index]),
                "top15_conditions": int(top15_frequency[index]),
                "condition_count": int(score_conditions.shape[0]),
                "strength": float(strength[index]),
                "participation": float(participation[index]),
            }
            for rank, index in enumerate(top)
        ],
        "module_ranking": [
            {"module": f"M{int(index) + 1}", "involvement_bits": float(module_score[index])}
            for index in np.argsort(module_score)[::-1]
        ],
        "module_topology": [
            {
                "module": f"M{int(module) + 1}",
                "size": int(np.sum(membership == module)),
                "involvement_bits": float(module_score[module]),
                "mean_strength": float(np.mean(strength[membership == module])),
                "mean_participation": float(np.mean(participation[membership == module])),
                "internal_weight_fraction": float(
                    connectivity[np.ix_(membership == module, membership == module)].sum()
                    / max(connectivity[membership == module].sum(), 1.0e-12)
                ),
                "top_regions": [
                    str(labels[index])
                    for index in sorted(
                        np.flatnonzero(membership == module),
                        key=lambda index: -score[index],
                    )[:5]
                ],
            }
            for module in range(int(membership.max()) + 1)
        ],
        "topology_correlations": {
            "roi_involvement": {
                "strength": spearman_summary(strength, score),
                "participation": spearman_summary(participation, score),
                "within_z": spearman_summary(np.asarray(payload["within_z"], dtype=float), score),
            },
            "local_ei_coupling": {
                "strength": spearman_summary(strength, local_ei_coupling),
                "participation": spearman_summary(participation, local_ei_coupling),
                "within_z": spearman_summary(np.asarray(payload["within_z"], dtype=float), local_ei_coupling),
            },
            "cross_roi_coupling": {
                "strength": spearman_summary(strength, cross_roi_coupling),
                "participation": spearman_summary(participation, cross_roi_coupling),
                "within_z": spearman_summary(np.asarray(payload["within_z"], dtype=float), cross_roi_coupling),
            },
        },
        "stability": {
            "median_pairwise_rank_spearman": float(np.median(rank_agreements)),
            "min_pairwise_rank_spearman": float(np.min(rank_agreements)),
            "condition_count": int(score_conditions.shape[0]),
        },
        "interpretation_boundary": (
            "ROI and module involvement are leave-one-block-out conditional-total-correlation reductions. "
            "They are nonnegative sensitivity scores but overlap and are not mutually exclusive atoms."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decompose critical DMF PhiEID and relate ROI involvement to SC topology.")
    parser.add_argument("--main-confirmation", type=Path, default=MAIN_CONFIRMATION)
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--connectivity-labels", type=Path, default=CONNECTIVITY_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--cross-figure", type=Path, default=DEFAULT_CROSS_FIGURE)
    parser.add_argument("--comparison-figure", type=Path, default=DEFAULT_COMPARISON_FIGURE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--critical-g", type=parse_float_list, default=CRITICAL_G)
    parser.add_argument("--community-seed", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--validation-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = compute_payload(args)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    plot_local_ei_vs_strength(payload, resolve_path(args.figure))
    plot_cross_roi_vs_strength(payload, resolve_path(args.cross_figure))
    plot_local_cross_roi_comparison(payload, resolve_path(args.comparison_figure))
    write_summary(payload, resolve_path(args.summary))
    print(f"Saved results: {output}")
    print(f"Saved figure: {resolve_path(args.figure)}")
    print(f"Saved cross-ROI figure: {resolve_path(args.cross_figure)}")
    print(f"Saved comparison figure: {resolve_path(args.comparison_figure)}")
    print(f"Saved summary: {resolve_path(args.summary)}")


if __name__ == "__main__":
    main()
