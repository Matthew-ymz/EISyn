#!/usr/bin/env python3
"""Plot lead 1, 8, and 24 UniCM SPTs for all three checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_unicm_11mode_xi_hierarchy_tree import (  # noqa: E402
    CORE_MODES,
    EDGE_COLOR,
    INK,
    MODE_COLORS,
    SYN_COLOR,
    _blend_with_white,
    _node_record,
    _positions,
    _terminal_order,
    _tree_from_record,
    _tree_metrics,
)
from scripts.phi_hierarchy import RAW_RESIDUAL, SIGNED, greedy_phi_tree  # noqa: E402
from scripts.plot_unicm_all_mode_target_pair_syn import extract_all_mode_target  # noqa: E402
from scripts.plot_unicm_phi_eid_greedy_decomposition import (  # noqa: E402
    compute_subset_ei_table_from_covariance,
    precompute_source_logdets,
)
from scripts.spt import flatten_nodes, nontrivial_bipartitions  # noqa: E402
from scripts.unicm_peid_syn_analysis import (  # noqa: E402
    MODE_NAMES,
    load_full_history_prediction_cache,
    overall_prediction_cache_path,
    sample_full_history_mode_inputs,
)


RESULT_ROOT = ROOT / "results/unicm_xi_hierarchy_lead_comparison"
OUTPUT = ROOT / "fig/earth_unicm_spt_leads01_08_24_checkpoints.png"
CACHE_DIR = ROOT / "results/unicm_overall_ei_cpu_bound4_n8192/cache"
LEADS = (1, 8, 24)
SEEDS = (1, 2, 3)
N_SAMPLES = 8192
SAMPLING_SEED = 20260619
INTERVENTION_BOUND = 4.0
JITTER = 1.0e-6
SYN_TOLERANCE = 1.0e-4
NEGATIVE_COLOR = "#B5483F"


def _load_valid_trees() -> dict[tuple[int, int], object]:
    paths = {
        8: RESULT_ROOT / "lead08/summary.json",
        24: RESULT_ROOT / "lead24/summary.json",
    }
    trees: dict[tuple[int, int], object] = {}
    for lead, path in paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["checkpoints"]:
            trees[(lead, int(row["seed"]))] = _tree_from_record(row["tree"])
    for seed in (2, 3):
        payload = json.loads(
            (RESULT_ROOT / f"lead01_seed{seed}_strict/summary.json").read_text(encoding="utf-8")
        )
        row = payload["checkpoints"][0]
        trees[(1, seed)] = _tree_from_record(row["tree"])
    return trees


def _lead1_seed1_signed_diagnostic() -> tuple[object, dict[str, object]]:
    histories = sample_full_history_mode_inputs(
        n_samples=N_SAMPLES,
        intervention_bound=INTERVENTION_BOUND,
        seed=SAMPLING_SEED,
    )
    history_flat, subset_columns, source_logdets = precompute_source_logdets(
        histories,
        jitter=JITTER,
    )
    cache_args = SimpleNamespace(
        n_samples=N_SAMPLES,
        sampling_seed=SAMPLING_SEED,
        intervention_bound=INTERVENTION_BOUND,
        start_month=0,
        device="cpu",
    )
    predictions = load_full_history_prediction_cache(
        overall_prediction_cache_path(CACHE_DIR, seed=1, args=cache_args),
        n_samples=N_SAMPLES,
    )
    target = extract_all_mode_target(predictions, lead=1)
    ei_table = compute_subset_ei_table_from_covariance(
        history_flat,
        target,
        subset_columns,
        source_logdets,
        jitter=JITTER,
    )
    mode_names = tuple(MODE_NAMES)
    singleton_ei = {name: float(ei_table[(name,)]) for name in mode_names}
    tree = greedy_phi_tree(
        mode_names,
        ei_table,
        policy=SIGNED,
        eps=1.0e-5,
        split_tolerance=SYN_TOLERANCE,
        singleton_ei=singleton_ei,
        split_objective=RAW_RESIDUAL,
    )
    candidate_values = []
    for node in flatten_nodes(tree):
        if not node.children:
            continue
        for left, right in nontrivial_bipartitions(node.sources):
            left_xi = float(ei_table[left] - sum(singleton_ei[name] for name in left))
            right_xi = float(ei_table[right] - sum(singleton_ei[name] for name in right))
            candidate_values.append(float(node.xi_value - left_xi - right_xi))
    values = np.asarray(candidate_values, dtype=float)
    audit = {
        "status": "invalid_nonnegative_spt_signed_diagnostic_only",
        "syn_tolerance_bits": SYN_TOLERANCE,
        "minimum_candidate_syn_bits": float(values.min()),
        "negative_candidate_count": int(np.count_nonzero(values < 0.0)),
        "significant_violation_count": int(np.count_nonzero(values < -SYN_TOLERANCE)),
        "candidate_count": int(values.size),
        "selected_root_syn_bits": float(tree.syn_value),
    }
    return tree, audit


def _coalition_label(node) -> str:
    names = ["ENSO" if name == "nino" else name for name in node.sources]
    if node.order <= 4:
        members = "+".join(names)
        if len(members) > 19:
            members = f"{node.order} modes"
    elif frozenset(node.sources) == CORE_MODES:
        members = "5-mode core"
    else:
        members = f"{node.order} modes"
    return f"{members}\n{node.syn_value:.3f}"


def _draw_panel(axis, tree, *, lead: int, seed: int, norm: Normalize, cmap, invalid: bool) -> None:
    positions, terminals = _positions(tree)
    internal = [node for node in flatten_nodes(tree) if node.children]
    core = next((node for node in internal if frozenset(node.sources) == CORE_MODES), None)
    if invalid:
        axis.set_facecolor("#FFF6F4")
    if core is not None:
        core_x = [positions[id(node)][0] for node in _terminal_order(core)]
        left, right = min(core_x) - 0.45, max(core_x) + 0.45
        top = positions[id(core)][1] + 0.30
        axis.add_patch(
            FancyBboxPatch(
                (left, -0.72),
                right - left,
                top + 0.72,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                facecolor="#EDF5F2",
                edgecolor="#8BB9AA",
                linewidth=0.75,
                zorder=0,
            )
        )
    for node in internal:
        _, parent_y = positions[id(node)]
        child_points = [positions[id(child)] for child in node.children]
        in_core = core is not None and set(node.sources).issubset(CORE_MODES)
        edge = SYN_COLOR if in_core else EDGE_COLOR
        width = 1.25 if in_core else 0.60
        axis.plot(
            [child_points[0][0], child_points[1][0]],
            [parent_y, parent_y],
            color=edge,
            linewidth=width,
            zorder=1,
        )
        for child_x, child_y in child_points:
            axis.plot([child_x, child_x], [child_y, parent_y], color=edge, linewidth=width, zorder=1)
    for node in internal:
        x_value, y_value = positions[id(node)]
        value = float(node.syn_value)
        if value < 0.0:
            face, edge, text_color = "#F5D8D4", NEGATIVE_COLOR, "#762820"
        else:
            relative = float(norm(value))
            face = cmap(relative)
            edge = _blend_with_white(SYN_COLOR, 0.55 + 0.45 * relative)
            text_color = "white" if relative > 0.68 else INK
        axis.text(
            x_value,
            y_value,
            _coalition_label(node),
            ha="center",
            va="center",
            fontsize=4.55,
            color=text_color,
            linespacing=1.02,
            bbox={
                "boxstyle": "round,pad=0.19",
                "facecolor": face,
                "edgecolor": edge,
                "linewidth": 0.7,
            },
            zorder=4,
        )
    for terminal in terminals:
        label = "ENSO" if terminal.sources[0] == "nino" else terminal.sources[0]
        color = MODE_COLORS.get(terminal.sources[0], "#7B8794")
        axis.text(
            positions[id(terminal)][0],
            -0.14,
            label,
            rotation=58,
            ha="right",
            va="top",
            fontsize=5.0,
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": _blend_with_white(color, 0.16),
                "edgecolor": color,
                "linewidth": 0.55,
            },
            clip_on=False,
        )
    metrics = _tree_metrics(tree, SYN_TOLERANCE)
    status = "INVALID signed diagnostic" if invalid else "valid nonnegative SPT"
    status_color = NEGATIVE_COLOR if invalid else "#5C6873"
    axis.text(
        0.01,
        0.985,
        rf"$\Xi$={tree.xi_value:.3f} bits" + "\n" + status,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.6,
        color=status_color,
        linespacing=1.25,
    )
    if lead == LEADS[0]:
        axis.text(
            0.5,
            1.035,
            f"checkpoint {seed}",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold",
            color=INK,
        )
    if seed == SEEDS[0]:
        axis.text(
            -0.10,
            0.5,
            f"Lead {lead}",
            transform=axis.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color=INK,
        )
    axis.set_xlim(-0.8, max(point[0] for point in positions.values()) + 0.55)
    axis.set_ylim(-1.15, max(point[1] for point in positions.values()) + 1.0)
    axis.axis("off")


def main() -> None:
    trees = _load_valid_trees()
    diagnostic_tree, audit = _lead1_seed1_signed_diagnostic()
    trees[(1, 1)] = diagnostic_tree
    all_positive = [
        float(node.syn_value)
        for tree in trees.values()
        for node in flatten_nodes(tree)
        if node.children and node.syn_value >= 0.0
    ]
    norm = Normalize(vmin=0.0, vmax=max(all_positive))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("syn", ["#F1F7F5", SYN_COLOR])
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(14.2, 12.8),
        layout="constrained",
        facecolor="white",
    )
    for row, lead in enumerate(LEADS):
        for column, seed in enumerate(SEEDS):
            _draw_panel(
                axes[row, column],
                trees[(lead, seed)],
                lead=lead,
                seed=seed,
                norm=norm,
                cmap=cmap,
                invalid=(lead, seed) == (1, 1),
            )
    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        location="right",
        shrink=0.63,
        aspect=34,
        pad=0.012,
    )
    colorbar.set_label("Local hierarchy Syn (bits)", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=5.7, width=0.5, length=2.2)
    figure.text(
        0.5,
        -0.008,
        "Same 12-month sources, interventions, estimator and 11-mode target definition; only forecast lead changes. "
        "Red denotes a significant nonnegativity violation, not physical negative Syn.",
        ha="center",
        va="top",
        fontsize=6.3,
        color=INK,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    rows = []
    for lead in LEADS:
        for seed in SEEDS:
            tree = trees[(lead, seed)]
            rows.append(
                {
                    "lead": lead,
                    "seed": seed,
                    "status": (
                        "invalid_nonnegative_spt_signed_diagnostic_only"
                        if (lead, seed) == (1, 1)
                        else "valid_nonnegative_spt"
                    ),
                    "xi_bits": float(tree.xi_value),
                    "metrics": _tree_metrics(tree, SYN_TOLERANCE),
                    "tree": _node_record(tree),
                }
            )
    payload = {
        "experiment": "UniCM lead-resolved exact SPT comparison",
        "treatment_levels": list(LEADS),
        "checkpoints": list(SEEDS),
        "source_definition": "each of 11 sources is one mode's full 12-month history",
        "target_definition": "all 11 future modes jointly at the selected lead",
        "n_samples": N_SAMPLES,
        "intervention_bound": INTERVENTION_BOUND,
        "sampling_seed": SAMPLING_SEED,
        "estimator": "affine degree-1 TM / Gaussian log-det equivalent",
        "syn_tolerance_bits": SYN_TOLERANCE,
        "lead1_checkpoint1_audit": audit,
        "figure": str(OUTPUT),
        "results": rows,
    }
    (RESULT_ROOT / "comparison_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
