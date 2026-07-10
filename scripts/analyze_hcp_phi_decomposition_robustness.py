from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from run_hcp_lausanne_phi_eid_pilot import (
    MODULE_COLORS,
    MODULE_ORDER,
    circular_shift_null,
    fit_ridge_transition,
    greedy_phi_atoms,
    load_cached_roi_timeseries,
    make_lagged_samples,
    module_ei_table,
    module_indices_from_labels,
    ordered_roi_labels,
    roi_leave_one_out_burden,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "results" / "hcp_lausanne_phi_eid_pilot" / "summary.json"
DEFAULT_OUTPUT = ROOT / "results" / "hcp_lausanne_phi_eid_pilot" / "robustness_summary.json"
DEFAULT_FIGURE = ROOT / "fig" / "hcp_lausanne_phi_eid_robustness"
DEFAULT_REPORT = ROOT / "docs" / "log" / "hcp_lausanne_phi_eid_robustness.md"


def atom_map(row: Mapping[str, object]) -> dict[str, float]:
    return {str(atom["label"]): float(atom["value"]) for atom in row.get("module_atoms", [])}


def roi_map(row: Mapping[str, object]) -> dict[str, float]:
    return {str(item["roi"]): float(item["burden"]) for item in row.get("roi_burden", [])}


def atom_value_map_to_module_participation(atom_values: Mapping[str, float]) -> dict[str, float]:
    totals = {name: 0.0 for name in MODULE_ORDER}
    for label, value in atom_values.items():
        for source in str(label).split("+"):
            if source in totals:
                totals[source] += float(value)
    return totals


def atom_module_participation(row: Mapping[str, object]) -> dict[str, float]:
    totals = {name: 0.0 for name in MODULE_ORDER}
    for atom in row.get("module_atoms", []):
        value = float(atom["value"])
        for source in atom.get("sources", []):
            name = str(source)
            if name in totals:
                totals[name] += value
    return totals


def atom_order_totals(row: Mapping[str, object]) -> dict[int, float]:
    totals = {order: 0.0 for order in range(1, len(MODULE_ORDER) + 1)}
    for atom in row.get("module_atoms", []):
        order = int(atom.get("order", len(atom.get("sources", []))))
        totals[order] = totals.get(order, 0.0) + float(atom["value"])
    return totals


def top_mean_keys(maps: Sequence[Mapping[str, float]], *, limit: int) -> list[str]:
    totals: dict[str, float] = {}
    for values in maps:
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return [key for key, _ in sorted(totals.items(), key=lambda item: item[1], reverse=True)[:limit]]


def empirical_upper_p(observed: float, null_values: Sequence[float]) -> float:
    null_array = np.asarray(null_values, dtype=float)
    return float((1 + np.sum(null_array >= float(observed))) / (1 + null_array.size))


def bh_fdr(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        return []
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.empty_like(ranked)
    running = 1.0
    n = float(values.size)
    for idx in range(values.size - 1, -1, -1):
        running = min(running, ranked[idx] * n / float(idx + 1))
        adjusted[idx] = running
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return [float(x) for x in out]


def pearson_or_nan(left: Sequence[float], right: Sequence[float]) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if x.size < 2 or y.size < 2 or np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def nanmean_or_nan(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def topk_overlap(left: Sequence[str], right: Sequence[str], *, k: int) -> float:
    left_set = set(list(left)[:k])
    right_set = set(list(right)[:k])
    if not left_set and not right_set:
        return float("nan")
    return float(len(left_set & right_set) / max(1, min(k, len(left_set | right_set))))


def infer_lr_rl_run_pair(rows: Sequence[Mapping[str, object]]) -> tuple[str, str] | None:
    runs = sorted({str(row["run"]) for row in rows})
    for left in runs:
        if not left.endswith("_LR"):
            continue
        right = f"{left[:-3]}_RL"
        if right in runs:
            return left, right
    return None


def row_null_decomposition(
    *,
    series: np.ndarray,
    labels: Sequence[str],
    null_reps: int,
    seed: int,
    ridge_alpha: float,
    ridge: float,
    target_atoms: Sequence[str],
    target_rois: Sequence[str],
) -> dict[str, object]:
    modules = module_indices_from_labels(labels)
    module_names = tuple(name for name in MODULE_ORDER if name in modules)
    atom_values = {label: [] for label in target_atoms}
    module_participation = {name: [] for name in MODULE_ORDER}
    roi_values = {label: [] for label in target_rois}
    whole_phi = []
    for rep in range(int(null_reps)):
        shifted = circular_shift_null(series, seed=int(seed) + rep + 1000)
        source, target = make_lagged_samples(shifted, tau=1)
        fit = fit_ridge_transition(source, target, alpha=float(ridge_alpha), ridge=float(ridge))
        whole_phi.append(float(fit["phi"]["raw_phi"]))
        transition = np.asarray(fit["transition"], dtype=float)
        noise = np.asarray(fit["noise_covariance"], dtype=float)

        table = module_ei_table(transition, noise, modules, ridge=float(ridge))
        singleton = {name: float(table[(name,)]) for name in module_names}
        atoms = greedy_phi_atoms(module_names, table, singleton_ei=singleton)
        current_atoms = {"+".join(atom.sources): float(atom.value) for atom in atoms}
        for label in target_atoms:
            atom_values[label].append(float(current_atoms.get(label, 0.0)))
        current_participation = atom_value_map_to_module_participation(current_atoms)
        for name in MODULE_ORDER:
            module_participation[name].append(float(current_participation.get(name, 0.0)))

        if target_rois:
            current_rois = roi_leave_one_out_burden(transition, noise, labels, ridge=float(ridge))
            current_roi_map = {str(row["roi"]): float(row["burden"]) for row in current_rois}
            for label in target_rois:
                roi_values[label].append(float(current_roi_map.get(label, 0.0)))
    return {"whole_phi": whole_phi, "atoms": atom_values, "module_participation": module_participation, "rois": roi_values}


def load_series_for_row(row: Mapping[str, object], labels: Sequence[str], roi_cache_dir: Path | None) -> np.ndarray:
    if roi_cache_dir is not None:
        series, _ = load_cached_roi_timeseries(
            roi_cache_dir,
            subject=str(row["subject"]),
            run=str(row["run"]),
            expected_labels=labels,
        )
        return series
    raise ValueError("--roi-cache-dir is required for robustness null decomposition.")


def summarize_group_null(
    observed_maps: Sequence[Mapping[str, float]],
    null_maps_by_row: Sequence[Mapping[str, Sequence[float]]],
    targets: Sequence[str],
    *,
    null_reps: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target in targets:
        observed_values = np.asarray([float(values.get(target, 0.0)) for values in observed_maps], dtype=float)
        null_group = []
        for rep in range(int(null_reps)):
            null_group.append(
                float(np.mean([float(row_values.get(target, [0.0] * null_reps)[rep]) for row_values in null_maps_by_row]))
            )
        observed_mean = float(np.mean(observed_values))
        null_array = np.asarray(null_group, dtype=float)
        rows.append(
            {
                "label": str(target),
                "observed_mean": observed_mean,
                "observed_sd": float(np.std(observed_values, ddof=1)) if observed_values.size > 1 else 0.0,
                "null_mean": float(np.mean(null_array)),
                "null_sd": float(np.std(null_array, ddof=1)) if null_array.size > 1 else 0.0,
                "difference": float(observed_mean - np.mean(null_array)),
                "empirical_p": empirical_upper_p(observed_mean, null_array),
            }
        )
    adjusted = bh_fdr([row["empirical_p"] for row in rows])
    for row, q_value in zip(rows, adjusted):
        row["fdr_q"] = q_value
    return sorted(rows, key=lambda row: float(row["difference"]), reverse=True)


def run_pair_reliability(
    rows: Sequence[Mapping[str, object]],
    *,
    run_pair: tuple[str, str] | None = None,
) -> dict[str, object]:
    by_subject: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in rows:
        by_subject.setdefault(str(row["subject"]), {})[str(row["run"])] = row
    if run_pair is None:
        run_pair = infer_lr_rl_run_pair(rows)
    if run_pair is None:
        return {"paired_subjects": 0}
    left_run, right_run = run_pair
    paired = [
        (subject, run_rows[left_run], run_rows[right_run])
        for subject, run_rows in sorted(by_subject.items())
        if left_run in run_rows and right_run in run_rows
    ]
    if not paired:
        return {"paired_subjects": 0, "left_run": left_run, "right_run": right_run}

    lr_diff = []
    rl_diff = []
    atom_corr = []
    atom_overlap = []
    atom_participation_corr = []
    atom_order_corr = []
    roi_corr = []
    roi_overlap = []
    for _, left, right in paired:
        left_null = float(np.mean([n["raw_phi"] for n in left["null"]]))
        right_null = float(np.mean([n["raw_phi"] for n in right["null"]]))
        lr_diff.append(float(left["ridge_phi"]["raw_phi"]) - left_null)
        rl_diff.append(float(right["ridge_phi"]["raw_phi"]) - right_null)

        left_atoms = atom_map(left)
        right_atoms = atom_map(right)
        atom_labels = sorted(set(left_atoms) | set(right_atoms))
        atom_corr.append(pearson_or_nan([left_atoms.get(k, 0.0) for k in atom_labels], [right_atoms.get(k, 0.0) for k in atom_labels]))
        atom_overlap.append(
            topk_overlap(
                [k for k, _ in sorted(left_atoms.items(), key=lambda item: item[1], reverse=True)],
                [k for k, _ in sorted(right_atoms.items(), key=lambda item: item[1], reverse=True)],
                k=5,
            )
        )
        left_participation = atom_module_participation(left)
        right_participation = atom_module_participation(right)
        atom_participation_corr.append(
            pearson_or_nan(
                [left_participation.get(name, 0.0) for name in MODULE_ORDER],
                [right_participation.get(name, 0.0) for name in MODULE_ORDER],
            )
        )
        left_order = atom_order_totals(left)
        right_order = atom_order_totals(right)
        order_keys = list(range(1, len(MODULE_ORDER) + 1))
        atom_order_corr.append(
            pearson_or_nan(
                [left_order.get(order, 0.0) for order in order_keys],
                [right_order.get(order, 0.0) for order in order_keys],
            )
        )

        left_rois = roi_map(left)
        right_rois = roi_map(right)
        roi_labels = sorted(set(left_rois) | set(right_rois))
        roi_corr.append(pearson_or_nan([left_rois.get(k, 0.0) for k in roi_labels], [right_rois.get(k, 0.0) for k in roi_labels]))
        roi_overlap.append(
            topk_overlap(
                [k for k, _ in sorted(left_rois.items(), key=lambda item: item[1], reverse=True)],
                [k for k, _ in sorted(right_rois.items(), key=lambda item: item[1], reverse=True)],
                k=15,
            )
        )
    return {
        "paired_subjects": len(paired),
        "left_run": left_run,
        "right_run": right_run,
        "phi_diff_lr": lr_diff,
        "phi_diff_rl": rl_diff,
        "phi_diff_pearson": pearson_or_nan(lr_diff, rl_diff),
        "mean_atom_vector_pearson": nanmean_or_nan(atom_corr),
        "mean_top5_atom_overlap": nanmean_or_nan(atom_overlap),
        "mean_atom_module_participation_pearson": nanmean_or_nan(atom_participation_corr),
        "mean_atom_order_pearson": nanmean_or_nan(atom_order_corr),
        "mean_roi_vector_pearson": nanmean_or_nan(roi_corr),
        "mean_top15_roi_overlap": nanmean_or_nan(roi_overlap),
        "per_subject": [
            {
                "subject": subject,
                "lr_observed_minus_null": lr,
                "rl_observed_minus_null": rl,
                "atom_vector_pearson": ac,
                "top5_atom_overlap": ao,
                "atom_module_participation_pearson": apc,
                "atom_order_pearson": aoc,
                "roi_vector_pearson": rc,
                "top15_roi_overlap": ro,
            }
            for (subject, _, _), lr, rl, ac, ao, apc, aoc, rc, ro in zip(
                paired,
                lr_diff,
                rl_diff,
                atom_corr,
                atom_overlap,
                atom_participation_corr,
                atom_order_corr,
                roi_corr,
                roi_overlap,
            )
        ],
    }


def plot_robustness(summary: Mapping[str, object], figure_base: Path) -> None:
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8})
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    rows = summary["rows"]
    whole = summary["whole_phi"]
    atoms = summary["atom_tests"][:8]
    rois = summary["roi_tests"][:10]
    reliability = summary["run_reliability"]

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), constrained_layout=True)
    ax = axes[0, 0]
    labels = [f"{row['subject']} {row['run'].replace('REST1_', '')}" for row in rows]
    x = np.arange(len(rows))
    observed = np.asarray([row["observed"] for row in whole], dtype=float)
    null_mean = np.asarray([row["null_mean"] for row in whole], dtype=float)
    ax.plot(x, null_mean, "o", color="#7F7F7F", label="Null mean")
    ax.plot(x, observed, "o", color="#2F7D5A", label="Observed")
    for idx in x:
        ax.plot([idx, idx], [null_mean[idx], observed[idx]], color="#B8B8B8", linewidth=0.8)
    if len(labels) > 12:
        ax.set_xticks([])
        ax.set_xlabel(f"{len(labels)} subject-runs")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("Raw PhiEID (bits)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax = axes[0, 1]
    atom_labels = [row["label"] for row in atoms]
    atom_obs = [row["observed_mean"] for row in atoms]
    atom_null = [row["null_mean"] for row in atoms]
    y = np.arange(len(atom_labels))
    ax.barh(y + 0.18, atom_obs, height=0.35, color="#2F7D5A", label="Observed")
    ax.barh(y - 0.18, atom_null, height=0.35, color="#9E9E9E", label="Null")
    ax.set_yticks(y)
    ax.set_yticklabels(atom_labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Mean module atom")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax = axes[1, 0]
    roi_labels = [row["label"] for row in rois]
    roi_obs = [row["observed_mean"] for row in rois]
    roi_null = [row["null_mean"] for row in rois]
    roi_modules = [str(row.get("module", "")) for row in rois]
    colors = [MODULE_COLORS.get(module, "#2F7D5A") for module in roi_modules]
    y = np.arange(len(roi_labels))
    ax.barh(y + 0.18, roi_obs, height=0.35, color=colors)
    ax.barh(y - 0.18, roi_null, height=0.35, color="#9E9E9E")
    ax.set_yticks(y)
    ax.set_yticklabels(roi_labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Mean ROI burden")
    present_modules = [module for module in MODULE_ORDER if module in set(roi_modules)]
    legend_handles = [Patch(facecolor="#9E9E9E", edgecolor="none", label="Null mean")]
    legend_handles.extend(
        Patch(facecolor=MODULE_COLORS[module], edgecolor="none", label=f"Observed {module}")
        for module in present_modules
    )
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax = axes[1, 1]
    if int(reliability.get("paired_subjects", 0)) > 0:
        ax.scatter(reliability["phi_diff_lr"], reliability["phi_diff_rl"], color="#2F7D5A")
        all_values = np.asarray(reliability["phi_diff_lr"] + reliability["phi_diff_rl"], dtype=float)
        lo = float(np.min(all_values))
        hi = float(np.max(all_values))
        ax.plot([lo, hi], [lo, hi], color="#7F7F7F", linewidth=0.8)
        ax.set_xlabel("LR observed - null")
        ax.set_ylabel("RL observed - null")
        ax.text(
            0.02,
            0.98,
            f"r = {float(reliability['phi_diff_pearson']):.2f}\nROI r = {float(reliability['mean_roi_vector_pearson']):.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
        )
    else:
        ax.text(0.5, 0.5, "No paired LR/RL runs", ha="center", va="center")
        ax.set_axis_off()

    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(str(figure_base) + suffix, dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, summary: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    whole = summary["whole_phi"]
    atoms = summary["atom_tests"]
    participation = summary.get("module_participation_tests", [])
    rois = summary["roi_tests"]
    reliability = summary["run_reliability"]

    whole_lines = "\n".join(
        f"| {row['subject']} | {row['run']} | {row['observed']:.6f} | {row['null_mean']:.6f} | {row['difference']:.6f} | {row['empirical_p']:.6f} |"
        for row in whole
    )
    atom_lines = "\n".join(
        f"| {row['label']} | {row['observed_mean']:.6f} | {row['null_mean']:.6f} | {row['difference']:.6f} | {row['empirical_p']:.6f} | {row['fdr_q']:.6f} |"
        for row in atoms[:12]
    )
    participation_lines = "\n".join(
        f"| {row['label']} | {row['observed_mean']:.6f} | {row['null_mean']:.6f} | {row['difference']:.6f} | {row['empirical_p']:.6f} | {row['fdr_q']:.6f} |"
        for row in participation
    )
    roi_lines = "\n".join(
        f"| {row['label']} | {row.get('module', '')} | {row['observed_mean']:.6f} | {row['null_mean']:.6f} | {row['difference']:.6f} | {row['empirical_p']:.6f} | {row['fdr_q']:.6f} |"
        for row in rois[:15]
    )
    path.write_text(
        "\n".join(
            [
                "# HCP Lausanne-83 PhiEID Robustness",
                "",
                f"- Rows analyzed: `{len(summary['rows'])}`",
                f"- Null repetitions per row: `{summary['null_reps']}`",
                f"- Paired LR/RL subjects: `{reliability.get('paired_subjects', 0)}`",
                "",
                "## Whole-state PhiEID vs null",
                "",
                "| Subject | Run | Observed | Null mean | Difference | Empirical p |",
                "|---|---|---:|---:|---:|---:|",
                whole_lines,
                "",
                "## Module atom robustness",
                "",
                "| Module atom | Observed mean | Null mean | Difference | Empirical p | FDR q |",
                "|---|---:|---:|---:|---:|---:|",
                atom_lines,
                "",
                "## Module participation robustness",
                "",
                "| Module | Observed mean | Null mean | Difference | Empirical p | FDR q |",
                "|---|---:|---:|---:|---:|---:|",
                participation_lines,
                "",
                "## ROI burden robustness",
                "",
                "| ROI | Module | Observed mean | Null mean | Difference | Empirical p | FDR q |",
                "|---|---|---:|---:|---:|---:|---:|",
                roi_lines,
                "",
                "## LR/RL stability",
                "",
                f"- Phi observed-minus-null Pearson r: `{float(reliability.get('phi_diff_pearson', float('nan'))):.6f}`",
                f"- Mean module atom vector Pearson r: `{float(reliability.get('mean_atom_vector_pearson', float('nan'))):.6f}`",
                f"- Mean top-5 module atom overlap: `{float(reliability.get('mean_top5_atom_overlap', float('nan'))):.6f}`",
                f"- Mean module participation Pearson r: `{float(reliability.get('mean_atom_module_participation_pearson', float('nan'))):.6f}`",
                f"- Mean atom-order distribution Pearson r: `{float(reliability.get('mean_atom_order_pearson', float('nan'))):.6f}`",
                f"- Mean ROI burden vector Pearson r: `{float(reliability.get('mean_roi_vector_pearson', float('nan'))):.6f}`",
                f"- Mean top-15 ROI overlap: `{float(reliability.get('mean_top15_roi_overlap', float('nan'))):.6f}`",
                "",
                "## Interpretation rule",
                "",
                "Treat module atoms or ROI burdens as robust only when they are above their null distribution after FDR correction and are stable across LR/RL runs.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    summary_path = Path(args.summary).expanduser().resolve()
    source = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = source["rows"]
    labels = [str(label) for label in source["labels"]]
    null_reps = int(args.null_reps or source.get("config", {}).get("null_reps", 0))
    if null_reps <= 0:
        raise ValueError("null_reps must be positive.")
    ridge_alpha = float(source.get("config", {}).get("ridge_alpha", 1.0))
    ridge = float(source.get("config", {}).get("ridge", 1.0e-6))

    observed_atom_maps = [atom_map(row) for row in rows]
    observed_participation_maps = [atom_module_participation(row) for row in rows]
    observed_roi_maps = [roi_map(row) for row in rows]
    target_atoms = top_mean_keys(observed_atom_maps, limit=int(args.top_atoms))
    target_rois = top_mean_keys(observed_roi_maps, limit=int(args.top_rois))

    null_atom_maps = []
    null_participation_maps = []
    null_roi_maps = []
    whole_rows = []
    roi_cache_dir = None if args.roi_cache_dir is None else Path(args.roi_cache_dir).expanduser().resolve()
    for row_index, row in enumerate(rows):
        series = load_series_for_row(row, labels, roi_cache_dir)
        row_seed = int(args.seed) + row_index * 100
        null = row_null_decomposition(
            series=series,
            labels=labels,
            null_reps=null_reps,
            seed=row_seed,
            ridge_alpha=ridge_alpha,
            ridge=ridge,
            target_atoms=target_atoms,
            target_rois=target_rois,
        )
        null_atom_maps.append(null["atoms"])
        null_participation_maps.append(null["module_participation"])
        null_roi_maps.append(null["rois"])
        observed = float(row["ridge_phi"]["raw_phi"])
        null_phi = np.asarray(null["whole_phi"], dtype=float)
        whole_rows.append(
            {
                "subject": str(row["subject"]),
                "run": str(row["run"]),
                "observed": observed,
                "null_mean": float(np.mean(null_phi)),
                "null_sd": float(np.std(null_phi, ddof=1)) if null_phi.size > 1 else 0.0,
                "difference": float(observed - np.mean(null_phi)),
                "empirical_p": empirical_upper_p(observed, null_phi),
            }
        )

    atom_tests = summarize_group_null(observed_atom_maps, null_atom_maps, target_atoms, null_reps=null_reps)
    participation_tests = summarize_group_null(
        observed_participation_maps,
        null_participation_maps,
        MODULE_ORDER,
        null_reps=null_reps,
    )
    roi_tests = summarize_group_null(observed_roi_maps, null_roi_maps, target_rois, null_reps=null_reps)
    roi_modules = {
        str(label): next((str(item["module"]) for row in rows for item in row["roi_burden"] if str(item["roi"]) == str(label)), "")
        for label in target_rois
    }
    for row in roi_tests:
        row["module"] = roi_modules.get(str(row["label"]), "")
    result = {
        "summary_source": str(summary_path),
        "null_reps": null_reps,
        "rows": [{"subject": str(row["subject"]), "run": str(row["run"])} for row in rows],
        "whole_phi": whole_rows,
        "atom_tests": atom_tests,
        "module_participation_tests": participation_tests,
        "roi_tests": roi_tests,
        "run_reliability": run_pair_reliability(rows),
        "config": {
            "seed": int(args.seed),
            "top_atoms": int(args.top_atoms),
            "top_rois": int(args.top_rois),
            "ridge_alpha": ridge_alpha,
            "ridge": ridge,
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    plot_robustness(result, Path(args.figure_base).expanduser().resolve())
    write_report(Path(args.report).expanduser().resolve(), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze HCP Lausanne-83 PhiEID decomposition robustness.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--roi-cache-dir", default=str(DEFAULT_SUMMARY.parent / "roi_timeseries"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--figure-base", default=str(DEFAULT_FIGURE))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--null-reps", type=int, default=None)
    parser.add_argument("--top-atoms", type=int, default=12)
    parser.add_argument("--top-rois", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260707)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (RuntimeError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
