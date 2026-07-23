#!/usr/bin/env python3
"""Targeted-greedy HCP cognition follow-up with exact free-path audits."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_task_evoked_pc2_xi_hierarchy import network_module_indices
from scripts.tune_hcp_task_evoked_xi_hierarchy import prepare_projection
from scripts.phi_hierarchy import greedy_phi_atoms, subset_phi_raw
from scripts.run_hcp_schaefer500_yeo7_module_phi_decomposition import module_ei_table
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import (
    default_yeo7_labels,
    load_yeo7_groups,
)
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi
from scripts.analyze_hcp_schaefer500_yeo7_network_attribution import (
    DEFAULT_REST_ROOT,
    DEFAULT_TASK_ROOT,
    NETWORK_ORDER,
    discover_inputs,
)


CONFIG_ID = "k1_p3_a1"
K = 1
ORDER = 3
ALPHA = 1.0
EPS = 1.0e-12
AUDIT_TOLERANCE = 1.0e-10
DEFAULT_COGNITION = (
    ROOT / "results/hcp_single_group_sem_full_1206/selected_29_sem_results.csv"
)
DEFAULT_ARRAYS = (
    ROOT / "results/hcp_schaefer500_task_evoked_xi_tuning/full/k1_p3_a1/arrays.npz"
)
DEFAULT_CACHED_RECORDS = (
    ROOT / "results/hcp_schaefer500_task_evoked_xi_tuning/full/records.jsonl"
)
DEFAULT_OUTPUT = ROOT / "results/hcp_cognition_targeted_greedy_followup"

CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "id": "crystallized_gambling",
        "score": "cry_score",
        "label": "Crystallized cognition",
        "state": "GAMBLING",
        "sources": ("Vis", "DorsAttn", "SalVentAttn", "Cont", "Default"),
    },
    {
        "id": "memory_relational",
        "score": "mem_score",
        "label": "Memory",
        "state": "RELATIONAL",
        "sources": ("Vis", "DorsAttn", "SalVentAttn", "Cont", "Default"),
    },
    {
        "id": "speed_motor",
        "score": "spd_score",
        "label": "Processing speed",
        "state": "MOTOR",
        "sources": (
            "SomMot",
            "DorsAttn",
            "SalVentAttn",
            "Limbic",
            "Cont",
            "Default",
        ),
    },
)

SHORT = {
    "Vis": "Vis",
    "SomMot": "Som",
    "DorsAttn": "DAN",
    "SalVentAttn": "SVAN",
    "Limbic": "Lim",
    "Cont": "Cont",
    "Default": "Def",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rest-root", type=Path, default=DEFAULT_REST_ROOT)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--labels", type=Path, default=default_yeo7_labels(500))
    parser.add_argument("--cognition", type=Path, default=DEFAULT_COGNITION)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument("--cached-records", type=Path, default=DEFAULT_CACHED_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-subjects", type=int, default=0)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def atom_key(sources: Sequence[str]) -> str:
    return "+".join(str(source) for source in sources)


def short_atom(sources: Sequence[str]) -> str:
    return "+".join(SHORT[str(source)] for source in sources)


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if len(x_array) < 3 or np.allclose(x_array, x_array[0]) or np.allclose(y_array, y_array[0]):
        return float("nan"), float("nan")
    result = spearmanr(x_array, y_array)
    return float(result.statistic), float(result.pvalue)


def load_subjects_and_cognition(
    arrays_path: Path, cognition_path: Path, max_subjects: int
) -> tuple[list[str], pd.DataFrame]:
    archive = np.load(arrays_path)
    subjects = [str(value) for value in archive["subjects"]]
    cognition = pd.read_csv(cognition_path, dtype={"Subject": str})
    cognition["Subject"] = cognition["Subject"].str.removeprefix("sub-")
    cognition = cognition.set_index("Subject")
    normalized = [subject.removeprefix("sub-") for subject in subjects]
    if set(normalized) != set(cognition.index):
        raise ValueError("Cognition and Xi subject sets do not match exactly")
    cognition = cognition.loc[normalized]
    if max_subjects:
        subjects = subjects[: int(max_subjects)]
        cognition = cognition.iloc[: int(max_subjects)]
    return subjects, cognition


def load_cached_records(path: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("config_id") == CONFIG_ID:
            rows[(str(row["subject"]), str(row["state"]))] = row
    return rows


def value_for_atom(atoms: Sequence[Any], sources: Sequence[str]) -> tuple[float, str | None, int | None]:
    target = tuple(sources)
    for atom in atoms:
        atom_sources = tuple(atom.sources) if hasattr(atom, "sources") else tuple(atom["sources"])
        if atom_sources == target:
            value = float(atom.value) if hasattr(atom, "value") else float(atom["value"])
            kind = str(atom.kind) if hasattr(atom, "kind") else str(atom["kind"])
            depth = int(atom.depth) if hasattr(atom, "depth") else int(atom["depth"])
            return value, kind, depth
    return 0.0, None, None


def candidate_metrics(
    table: Mapping[tuple[str, ...], float],
    singleton: Mapping[str, float],
    sources: tuple[str, ...],
) -> dict[str, Any]:
    full = tuple(NETWORK_ORDER)
    complement = tuple(source for source in full if source not in sources)
    free_atoms = greedy_phi_atoms(full, table, singleton_ei=singleton)
    targeted_atoms = greedy_phi_atoms(sources, table, singleton_ei=singleton)
    free_value, free_kind, free_depth = value_for_atom(free_atoms, sources)
    targeted_value, targeted_kind, targeted_depth = value_for_atom(targeted_atoms, sources)
    block_value = subset_phi_raw(sources, table, singleton)
    full_value = subset_phi_raw(full, table, singleton)
    complement_value = subset_phi_raw(complement, table, singleton) if complement else 0.0
    bridge_value = full_value - block_value - complement_value
    same_size = list(itertools.combinations(tuple(NETWORK_ORDER), len(sources)))
    same_size_metrics: dict[str, dict[str, float]] = {}
    for coalition in same_size:
        coalition_atoms = greedy_phi_atoms(coalition, table, singleton_ei=singleton)
        first_value, _, _ = value_for_atom(coalition_atoms, coalition)
        same_size_metrics[atom_key(coalition)] = {
            "targeted_first_residual_bits": float(first_value),
            "fixed_block_synergy_bits": float(subset_phi_raw(coalition, table, singleton)),
        }
    return {
        "free_candidate_atom_bits": free_value,
        "free_candidate_kind": free_kind,
        "free_candidate_depth": free_depth,
        "targeted_first_residual_bits": targeted_value,
        "targeted_first_kind": targeted_kind,
        "targeted_first_depth": targeted_depth,
        "fixed_block_synergy_bits": float(block_value),
        "forced_root_bridge_residual_bits": float(bridge_value),
        "full_cross_synergy_bits": float(full_value),
        "same_size_metrics": same_size_metrics,
    }


def compute_row(
    candidate: Mapping[str, Any],
    subject: str,
    path: Path,
    groups: Mapping[str, Sequence[int]],
    cached: Mapping[str, Any],
) -> dict[str, Any]:
    projections, _, development_end = prepare_projection(
        path, groups, state=str(candidate["state"]), max_components=2
    )
    reduced = projections[K]
    fitted = fit_delta_history_phi(
        reduced, alpha=ALPHA, order=ORDER, development_end=development_end
    )
    indices = network_module_indices(NETWORK_ORDER, n_components=K, order=ORDER)
    table = module_ei_table(
        fitted["transition"], fitted["noise_covariance"], indices, ridge=1.0e-6
    )
    singleton = {name: float(table[(name,)]) for name in NETWORK_ORDER}
    metrics = candidate_metrics(table, singleton, tuple(candidate["sources"]))
    cached_value, cached_kind, cached_depth = value_for_atom(
        cached["atoms"], tuple(candidate["sources"])
    )
    return {
        "config_id": CONFIG_ID,
        "candidate_id": str(candidate["id"]),
        "score": str(candidate["score"]),
        "score_label": str(candidate["label"]),
        "state": str(candidate["state"]),
        "sources": list(candidate["sources"]),
        "subject": subject,
        "development_end": int(development_end),
        "cached_free_candidate_atom_bits": float(cached_value),
        "cached_free_candidate_kind": cached_kind,
        "cached_free_candidate_depth": cached_depth,
        **metrics,
    }


def leave_one_out_summary(x: np.ndarray, y: np.ndarray, full_rho: float) -> dict[str, float]:
    values = []
    for index in range(len(x)):
        keep = np.arange(len(x)) != index
        rho, _ = safe_spearman(x[keep], y[keep])
        if np.isfinite(rho):
            values.append(rho)
    array = np.asarray(values, dtype=float)
    return {
        "minimum_rho": float(array.min()),
        "median_rho": float(np.median(array)),
        "maximum_rho": float(array.max()),
        "same_direction_fraction": float(np.mean(np.sign(array) == np.sign(full_rho))),
    }


def metric_summary(
    x: np.ndarray,
    y: np.ndarray,
    prior_selected: np.ndarray,
) -> dict[str, Any]:
    rho, p_value = safe_spearman(x, y)
    zero_mask = ~prior_selected
    zero_rho, zero_p = safe_spearman(x[zero_mask], y[zero_mask])
    return {
        "rho": rho,
        "p_raw_two_sided": p_value,
        "nonzero_subjects": int(np.count_nonzero(np.abs(y) > EPS)),
        "positive_subjects": int(np.count_nonzero(y > EPS)),
        "negative_subjects": int(np.count_nonzero(y < -EPS)),
        "prior_zero_subgroup": {
            "n": int(np.count_nonzero(zero_mask)),
            "rho": zero_rho,
            "p_raw_two_sided": zero_p,
        },
        "leave_one_out": leave_one_out_summary(x, y, rho),
    }


def summarize_same_size(
    rows: Sequence[Mapping[str, Any]], x: np.ndarray, selected_key: str
) -> dict[str, Any]:
    keys = list(rows[0]["same_size_metrics"])
    output: dict[str, Any] = {}
    for metric in ("targeted_first_residual_bits", "fixed_block_synergy_bits"):
        ranked = []
        for key in keys:
            values = np.asarray([row["same_size_metrics"][key][metric] for row in rows], dtype=float)
            rho, p_value = safe_spearman(x, values)
            ranked.append({"coalition": key, "rho": rho, "p_raw_two_sided": p_value})
        ranked.sort(key=lambda item: -abs(float(item["rho"])) if np.isfinite(item["rho"]) else np.inf)
        selected_rank = next(index for index, item in enumerate(ranked, start=1) if item["coalition"] == selected_key)
        output[metric] = {
            "selected_rank_by_absolute_rho": selected_rank,
            "coalition_count": len(ranked),
            "ranked": ranked,
        }
    return output


def build_summary(
    rows: list[dict[str, Any]], cognition: pd.DataFrame
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    all_cached_differences = []
    selected_targeted_differences = []
    for candidate in CANDIDATES:
        selected_rows = [row for row in rows if row["candidate_id"] == candidate["id"]]
        selected_rows.sort(key=lambda row: row["subject"])
        subject_ids = [row["subject"].removeprefix("sub-") for row in selected_rows]
        x = cognition.loc[subject_ids, candidate["score"]].to_numpy(dtype=float)
        cached_values = np.asarray(
            [row["cached_free_candidate_atom_bits"] for row in selected_rows], dtype=float
        )
        rerun_values = np.asarray(
            [row["free_candidate_atom_bits"] for row in selected_rows], dtype=float
        )
        targeted_values = np.asarray(
            [row["targeted_first_residual_bits"] for row in selected_rows], dtype=float
        )
        block_values = np.asarray(
            [row["fixed_block_synergy_bits"] for row in selected_rows], dtype=float
        )
        bridge_values = np.asarray(
            [row["forced_root_bridge_residual_bits"] for row in selected_rows], dtype=float
        )
        prior_selected = cached_values > EPS
        cached_difference = np.abs(rerun_values - cached_values)
        targeted_difference = np.abs(targeted_values[prior_selected] - rerun_values[prior_selected])
        all_cached_differences.extend(cached_difference.tolist())
        selected_targeted_differences.extend(targeted_difference.tolist())
        candidates[str(candidate["id"])] = {
            "score": candidate["score"],
            "score_label": candidate["label"],
            "state": candidate["state"],
            "sources": list(candidate["sources"]),
            "n_subjects": len(selected_rows),
            "prior_free_selected_subjects": int(np.count_nonzero(prior_selected)),
            "metrics": {
                "free_candidate_atom": metric_summary(x, rerun_values, prior_selected),
                "targeted_first_residual": metric_summary(x, targeted_values, prior_selected),
                "fixed_block_synergy": metric_summary(x, block_values, prior_selected),
                "forced_root_bridge_residual": metric_summary(x, bridge_values, prior_selected),
            },
            "same_cardinality_specificity": summarize_same_size(
                selected_rows, x, atom_key(candidate["sources"])
            ),
            "audit": {
                "max_abs_rerun_free_minus_cached_bits": float(cached_difference.max()),
                "max_abs_targeted_minus_rerun_free_on_prior_selected_bits": float(
                    targeted_difference.max() if len(targeted_difference) else 0.0
                ),
                "prior_selected_kinds": sorted(
                    {str(row["cached_free_candidate_kind"]) for row in selected_rows if row["cached_free_candidate_kind"]}
                ),
            },
        }
    maximum_cached = float(max(all_cached_differences))
    maximum_targeted = float(max(selected_targeted_differences))
    if maximum_cached > AUDIT_TOLERANCE or maximum_targeted > AUDIT_TOLERANCE:
        raise AssertionError(
            f"Exact audit failed: cached={maximum_cached:.3g}, targeted={maximum_targeted:.3g}"
        )
    return {
        "experiment": "HCP cognition targeted-greedy follow-up",
        "config": {
            "subjects": 29,
            "k": K,
            "order": ORDER,
            "alpha": ALPHA,
            "target": "full seven-network next state",
            "estimator": "existing affine/Gaussian log-determinant EI estimator",
            "selection": "three prespecified candidates from the prior exploratory raw-p screen",
        },
        "audit": {
            "tolerance_bits": AUDIT_TOLERANCE,
            "max_abs_rerun_free_minus_cached_bits": maximum_cached,
            "max_abs_targeted_minus_rerun_free_on_prior_selected_bits": maximum_targeted,
            "passed": True,
        },
        "candidates": candidates,
        "interpretation_boundary": (
            "Same-cohort targeted exploratory extension; raw p values are descriptive and the prior-zero subgroup "
            "is not an independent holdout."
        ),
    }


def save_figure(figure: plt.Figure, stem: Path) -> None:
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=400, bbox_inches="tight")
    plt.close(figure)


def add_linear_guide(axis: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    if len(x) >= 2 and not np.allclose(x, x[0]) and not np.allclose(y, y[0]):
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(float(x.min()), float(x.max()), 200)
        axis.plot(x_line, slope * x_line + intercept, color="#5B6573", lw=0.9, ls="--", zorder=1)


def plot_correlations(
    rows: list[dict[str, Any]], cognition: pd.DataFrame, summary: Mapping[str, Any], output: Path
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(10.8, 6.0), constrained_layout=True)
    for column, candidate in enumerate(CANDIDATES):
        selected_rows = sorted(
            [row for row in rows if row["candidate_id"] == candidate["id"]],
            key=lambda row: row["subject"],
        )
        subject_ids = [row["subject"].removeprefix("sub-") for row in selected_rows]
        x = cognition.loc[subject_ids, candidate["score"]].to_numpy(dtype=float)
        prior = np.asarray([row["cached_free_candidate_atom_bits"] > EPS for row in selected_rows])
        for row_index, (metric, ylabel) in enumerate(
            (
                ("targeted_first_residual_bits", "Targeted first-step residual (bits)"),
                ("fixed_block_synergy_bits", "Fixed-coalition total synergy (bits)"),
            )
        ):
            axis = axes[row_index, column]
            y = np.asarray([row[metric] for row in selected_rows], dtype=float)
            axis.scatter(
                x[~prior], y[~prior], s=25, color="#71A6A1", alpha=0.82, edgecolor="white", linewidth=0.4,
                label="not on prior free path" if column == 0 and row_index == 0 else None,
                zorder=3,
            )
            axis.scatter(
                x[prior], y[prior], s=32, color="#B65F3C", edgecolor="#333333", linewidth=0.55,
                label="on prior free path" if column == 0 and row_index == 0 else None,
                zorder=4,
            )
            add_linear_guide(axis, x, y)
            metric_key = "targeted_first_residual" if row_index == 0 else "fixed_block_synergy"
            stats = summary["candidates"][candidate["id"]]["metrics"][metric_key]
            axis.text(
                0.03, 0.97,
                f"rho={stats['rho']:.3f}, raw p={stats['p_raw_two_sided']:.3g}\n"
                f"nonzero={stats['nonzero_subjects']}/29",
                transform=axis.transAxes, ha="left", va="top", fontsize=6.2,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
                zorder=6,
            )
            axis.set(
                xlabel=candidate["label"] if row_index == 1 else "",
                ylabel=ylabel if column == 0 else "",
            )
            if row_index == 0:
                axis.set_title(
                    f"{candidate['state']} | {short_atom(candidate['sources'])}", fontsize=7.2
                )
    axes[0, 0].legend(loc="upper center", bbox_to_anchor=(1.65, 1.30), ncol=2, frameon=False)
    for label, axis in zip("abcdef", axes.ravel(), strict=True):
        axis.text(-0.16, 1.07, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    save_figure(figure, output / "targeted_greedy_cognition_scatter")


def plot_audit(rows: list[dict[str, Any]], summary: Mapping[str, Any], output: Path) -> None:
    selected = [row for row in rows if row["cached_free_candidate_atom_bits"] > EPS]
    cached = np.asarray([row["cached_free_candidate_atom_bits"] for row in selected], dtype=float)
    rerun = np.asarray([row["free_candidate_atom_bits"] for row in selected], dtype=float)
    targeted = np.asarray([row["targeted_first_residual_bits"] for row in selected], dtype=float)
    colors = {
        "crystallized_gambling": "#B65F3C",
        "memory_relational": "#6F83B5",
        "speed_motor": "#71A6A1",
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), constrained_layout=True)
    comparisons = (
        (cached, rerun, "Cached free-greedy atom (bits)", "Rerun free-greedy atom (bits)"),
        (rerun, targeted, "Rerun free-greedy atom (bits)", "Targeted-first residual (bits)"),
    )
    for axis, (x, y, xlabel, ylabel) in zip(axes, comparisons, strict=True):
        lower = float(min(x.min(), y.min())) - 0.04
        upper = float(max(x.max(), y.max())) + 0.04
        axis.plot([lower, upper], [lower, upper], color="#4B5563", lw=0.9, ls="--")
        for candidate in CANDIDATES:
            mask = np.asarray([row["candidate_id"] == candidate["id"] for row in selected])
            axis.scatter(
                x[mask], y[mask], s=30, color=colors[candidate["id"]], edgecolor="white", linewidth=0.45,
                label=candidate["label"] if axis is axes[0] else None, zorder=3,
            )
        axis.set(xlabel=xlabel, ylabel=ylabel, xlim=(lower, upper), ylim=(lower, upper), aspect="equal")
    axes[0].text(
        0.03, 0.97,
        f"max |delta|={summary['audit']['max_abs_rerun_free_minus_cached_bits']:.2e} bits",
        transform=axes[0].transAxes, ha="left", va="top", fontsize=6.2,
    )
    axes[1].text(
        0.03, 0.97,
        f"max |delta|={summary['audit']['max_abs_targeted_minus_rerun_free_on_prior_selected_bits']:.2e} bits",
        transform=axes[1].transAxes, ha="left", va="top", fontsize=6.2,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
    )
    for label, axis in zip("ab", axes, strict=True):
        axis.text(-0.16, 1.07, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    save_figure(figure, output / "targeted_greedy_exact_overlap_audit")


def fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.4g}"


def json_ready(value: Any) -> Any:
    """Convert non-finite statistical placeholders to strict-JSON null values."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def write_report(summary: Mapping[str, Any], output: Path) -> None:
    lines = [
        "# HCP cognition targeted-greedy follow-up",
        "",
        "The full seven-network next-state target, 29 subjects, task-evoked PCA, dynamics, chronological split, and affine/Gaussian EI estimator are frozen. Only the hierarchy readout changes.",
        "",
        "## Exact implementation audit",
        "",
        f"- Rerun free greedy versus cached free greedy: maximum absolute difference `{summary['audit']['max_abs_rerun_free_minus_cached_bits']:.3e}` bits.",
        f"- Targeted-first residual versus rerun free-path atom on previously selected subjects: maximum absolute difference `{summary['audit']['max_abs_targeted_minus_rerun_free_on_prior_selected_bits']:.3e}` bits.",
        f"- Required tolerance: `{summary['audit']['tolerance_bits']:.1e}` bits; audit passed: `{summary['audit']['passed']}`.",
        "",
        "All previously selected candidate atoms were split residuals. Fixed-coalition total synergy is therefore not expected to equal the prior atom value; the exact equality applies to the targeted-first residual.",
        "",
        "## Correlation results",
        "",
        "| Cognition | State | Prior selected | Targeted-first rho; raw p; nonzero | Fixed-block rho; raw p; nonzero | Prior-zero subgroup targeted rho; p | Same-size targeted rank |",
        "|---|---|---:|---|---|---|---:|",
    ]
    for candidate in CANDIDATES:
        item = summary["candidates"][candidate["id"]]
        targeted = item["metrics"]["targeted_first_residual"]
        block = item["metrics"]["fixed_block_synergy"]
        subgroup = targeted["prior_zero_subgroup"]
        specificity = item["same_cardinality_specificity"]["targeted_first_residual_bits"]
        lines.append(
            f"| {candidate['label']} | {candidate['state']} | {item['prior_free_selected_subjects']}/29 | "
            f"{fmt(targeted['rho'])}; {fmt(targeted['p_raw_two_sided'])}; {targeted['nonzero_subjects']}/29 | "
            f"{fmt(block['rho'])}; {fmt(block['p_raw_two_sided'])}; {block['nonzero_subjects']}/29 | "
            f"{fmt(subgroup['rho'])}; {fmt(subgroup['p_raw_two_sided'])} (n={subgroup['n']}) | "
            f"{specificity['selected_rank_by_absolute_rho']}/{specificity['coalition_count']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These candidates were selected in the same 29-subject cohort. Raw p values prioritize targeted exploratory hypotheses; the prior-zero subgroup is not an independent holdout. The existing affine/Gaussian EI estimator is deliberately retained for a one-factor comparison. The local PEID full text permits fixed source and target subsets, but notes that continuous-variable synergy is not theoretically guaranteed to be nonnegative (Zotero key: `MYATYWAJ`, full text).",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects, cognition = load_subjects_and_cognition(
        args.arrays, args.cognition, int(args.max_subjects)
    )
    discovered = discover_inputs(args.rest_root, args.task_root)
    groups = load_yeo7_groups(args.labels, expected_parcels=500)
    cached = load_cached_records(args.cached_records)
    jobs = [(candidate, subject) for candidate in CANDIDATES for subject in subjects]
    rows: list[dict[str, Any]] = []
    for candidate, subject in tqdm(jobs, desc="Targeted greedy", unit="model"):
        if subject not in discovered:
            raise KeyError(f"Missing imaging input for {subject}")
        key = (subject, str(candidate["state"]))
        if key not in cached:
            raise KeyError(f"Missing cached free-greedy record: {key}")
        rows.append(
            compute_row(
                candidate,
                subject,
                Path(discovered[subject][str(candidate["state"])]),
                groups,
                cached[key],
            )
        )
    records_path = args.output_dir / (
        "records.jsonl" if not args.max_subjects else f"smoke_records_{len(subjects)}.jsonl"
    )
    with records_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if len(subjects) != 29:
        maximum = max(
            abs(float(row["free_candidate_atom_bits"]) - float(row["cached_free_candidate_atom_bits"]))
            for row in rows
        )
        if maximum > AUDIT_TOLERANCE:
            raise AssertionError(f"Smoke audit failed: {maximum:.3g} bits")
        return {"smoke_subjects": len(subjects), "max_abs_cached_difference_bits": maximum}
    summary = build_summary(rows, cognition)
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_correlations(rows, cognition, summary, args.output_dir)
    plot_audit(rows, summary, args.output_dir)
    write_report(summary, args.output_dir)
    return summary


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps(result.get("audit", result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
