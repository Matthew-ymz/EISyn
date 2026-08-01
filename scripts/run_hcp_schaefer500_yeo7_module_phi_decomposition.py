#!/usr/bin/env python3
"""Greedily decompose Yeo7 history-source PhiEID without splitting within-network lags."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hcp_lausanne_phi_eid_pilot import circular_shift_null, ei_for_source_indices, greedy_phi_atoms, subset_phi_raw
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import DEFAULT_DATA, DEFAULT_LABELS, default_data_key, default_yeo7_labels, fit_yeo7_pc1, load_hcp_series, load_yeo7_groups
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import _subject_seed, fit_delta_history_phi
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null_all import DEFAULT_DATA_ROOT


DEFAULT_OUTPUT_DIR = ROOT / "results" / "hcp_schaefer500_yeo7_module_phi_decomposition"


def discover_subjects(data_root: Path) -> tuple[str, ...]:
    """Return all subjects with a Schaefer-500 MATLAB input, sorted by ID."""
    return tuple(path.parent.name for path in sorted(Path(data_root).glob("sub-*/*.mat")))


def network_history_indices(network_names: Sequence[str], *, order: int) -> dict[str, list[int]]:
    names = tuple(str(name) for name in network_names)
    return {name: [lag * len(names) + index for lag in range(int(order))] for index, name in enumerate(names)}


def module_ei_table(
    transition: np.ndarray,
    noise_covariance: np.ndarray,
    module_indices: Mapping[str, Sequence[int]],
    *,
    ridge: float = 1.0e-6,
) -> dict[tuple[str, ...], float]:
    names = tuple(module_indices)
    table: dict[tuple[str, ...], float] = {}
    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(names, size):
            indices = sorted(index for name in subset for index in module_indices[name])
            table[subset] = max(0.0, ei_for_source_indices(transition, noise_covariance, indices, ridge=ridge))
    return table


def decompose_modules(transition: np.ndarray, noise_covariance: np.ndarray, network_names: Sequence[str], *, order: int) -> tuple[dict[tuple[str, ...], float], list[Any]]:
    indices = network_history_indices(network_names, order=order)
    table = module_ei_table(transition, noise_covariance, indices)
    full = tuple(network_names)
    singleton = {name: table[(name,)] for name in full}
    return table, greedy_phi_atoms(full, table, singleton_ei=singleton)


def _block_phi(sources: Sequence[str], table: Mapping[tuple[str, ...], float]) -> float:
    ordered = tuple(sources)
    return subset_phi_raw(ordered, table, {name: table[(name,)] for name in ordered})


def analyze_subject(
    raw_series: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    *,
    subject: str,
    development_end: int,
    order: int,
    alpha: float,
    null_replicates: int,
    seed: int,
    top_k: int,
) -> dict[str, Any]:
    network_names = tuple(groups)
    reduced = fit_yeo7_pc1(np.asarray(raw_series, dtype=float)[:development_end], groups).transform(raw_series)
    observed_fit = fit_delta_history_phi(reduced, alpha=alpha, order=order, development_end=development_end)
    observed_table, atoms = decompose_modules(observed_fit["transition"], observed_fit["noise_covariance"], network_names, order=order)
    ranked_atoms = sorted((atom for atom in atoms if len(atom.sources) >= 2 and atom.value > 1e-9), key=lambda atom: atom.value, reverse=True)
    selected_atoms = ranked_atoms[:top_k]
    null_values: dict[tuple[str, ...], list[float]] = {tuple(atom.sources): [] for atom in selected_atoms}
    null_top_atoms = []
    for replicate in range(null_replicates):
        shifted = circular_shift_null(
            reduced[:development_end], seed=_subject_seed(seed + int(subject.removeprefix("sub-")), replicate)
        )
        null_fit = fit_delta_history_phi(shifted, alpha=alpha, order=order, development_end=development_end)
        null_table, null_atoms = decompose_modules(null_fit["transition"], null_fit["noise_covariance"], network_names, order=order)
        ranked_null_atoms = sorted((atom for atom in null_atoms if len(atom.sources) >= 2 and atom.value > 1e-9), key=lambda atom: atom.value, reverse=True)
        null_top_atoms.append([
            {"sources": list(atom.sources), "value": float(atom.value), "kind": atom.kind, "depth": int(atom.depth)}
            for atom in ranked_null_atoms[:top_k]
        ])
        for sources in null_values:
            null_values[sources].append(_block_phi(sources, null_table))
    top_atoms = []
    for atom in selected_atoms:
        sources = tuple(atom.sources)
        values = np.asarray(null_values[sources], dtype=float)
        block_phi = _block_phi(sources, observed_table)
        top_atoms.append({
            "sources": list(sources),
            "value": float(atom.value),
            "kind": atom.kind,
            "depth": int(atom.depth),
            "block_phi": float(block_phi),
            "null_block_phi_mean": float(values.mean()),
            "empirical_p": float((1 + np.sum(values >= block_phi)) / (len(values) + 1)),
        })
    full = tuple(network_names)
    return {
        "subject": subject,
        "module_full_phi": float(_block_phi(full, observed_table)),
        "atom_sum": float(sum(atom.value for atom in atoms)),
        "atoms": [{"sources": list(atom.sources), "value": float(atom.value), "kind": atom.kind, "depth": int(atom.depth)} for atom in atoms],
        "top_atoms": top_atoms,
        "null_top_atoms": null_top_atoms,
    }


def summarize_cores(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for atom in row["top_atoms"]:
            grouped[tuple(atom["sources"])].append(atom)
    summary = []
    for sources, atoms in grouped.items():
        values = np.asarray([float(atom["value"]) for atom in atoms])
        p_values = np.asarray([float(atom["empirical_p"]) for atom in atoms])
        summary.append({
            "sources": list(sources),
            "top_frequency": int(len(atoms)),
            "mean_atom_value_when_top": float(values.mean()),
            "median_atom_value_when_top": float(np.median(values)),
            "subjects_empirical_p_lt_0_05": int(np.sum(p_values < 0.05)),
            "mean_empirical_p_when_top": float(p_values.mean()),
        })
    return sorted(summary, key=lambda row: (-int(row["top_frequency"]), -float(row["mean_atom_value_when_top"]), row["sources"]))


def null_rank_frequency(rows: Sequence[Mapping[str, Any]], sources: Sequence[str], *, observed_frequency: int) -> dict[str, Any]:
    """Compare an observed core's cross-subject top-k frequency with matched null cohorts."""
    target = tuple(sources)
    replicate_count = len(rows[0]["null_top_atoms"])
    if any(len(row["null_top_atoms"]) != replicate_count for row in rows):
        raise ValueError("Every subject must provide the same number of null replicates.")
    frequency_by_replicate = [
        sum(any(tuple(atom["sources"]) == target for atom in row["null_top_atoms"][replicate]) for row in rows)
        for replicate in range(replicate_count)
    ]
    values = np.asarray(frequency_by_replicate, dtype=int)
    return {
        "sources": list(target),
        "observed_frequency": int(observed_frequency),
        "frequency_by_replicate": frequency_by_replicate,
        "null_frequency_mean": float(values.mean()),
        "null_frequency_max": int(values.max()),
        "empirical_p": float((1 + np.sum(values >= observed_frequency)) / (len(values) + 1)),
    }


def format_core_label(sources: Sequence[str]) -> str:
    """Abbreviate and wrap a module core label for a dense heatmap axis."""
    abbreviations = {"Vis": "Vis", "SomMot": "Som", "DorsAttn": "DAN", "SalVentAttn": "SVAN", "Limbic": "Lim", "Cont": "Cont", "Default": "Def"}
    missing = [name for name in abbreviations if name not in sources]
    if not missing:
        return "All 7"
    if len(missing) <= 2:
        return "Missing\n" + " + ".join(abbreviations[name] for name in missing)
    names = [abbreviations[name] for name in sources]
    return " + ".join(names[:3]) + ("\n" + " + ".join(names[3:]) if len(names) > 3 else "")


def summarize_null_cores(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for atoms in row["null_top_atoms"]:
            for atom in atoms:
                grouped[tuple(atom["sources"])].append(atom)
    summary = []
    for sources, atoms in grouped.items():
        values = np.asarray([float(atom["value"]) for atom in atoms])
        summary.append({"sources": list(sources), "top_frequency": int(len(atoms)), "mean_atom_value_when_top": float(values.mean())})
    return sorted(summary, key=lambda row: (-int(row["top_frequency"]), -float(row["mean_atom_value_when_top"]), row["sources"]))


def plot_core_consistency(rows: Sequence[Mapping[str, Any]], core_summary: Sequence[Mapping[str, Any]], destination: Path) -> None:
    selected = list(core_summary[:8])
    labels = [format_core_label(core["sources"]) for core in selected]
    matrix = np.full((len(rows), len(selected)), np.nan)
    for row_index, row in enumerate(rows):
        atom_by_sources = {tuple(atom["sources"]): atom for atom in row["top_atoms"]}
        for core_index, core in enumerate(selected):
            atom = atom_by_sources.get(tuple(core["sources"]))
            if atom is not None:
                matrix[row_index, core_index] = float(atom["value"])
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8, "axes.spines.right": False, "axes.spines.top": False})
    fig, axis = plt.subplots(figsize=(max(9.2, 1.15 * len(selected)), max(8.0, 0.28 * len(rows))), constrained_layout=True)
    image = axis.imshow(np.ma.masked_invalid(matrix), cmap="YlGnBu", aspect="auto")
    axis.set(xticks=np.arange(len(labels)), xticklabels=labels, yticks=np.arange(len(rows)), yticklabels=[row["subject"].removeprefix("sub-") for row in rows], ylabel="Subject")
    axis.tick_params(axis="x", labelsize=7, pad=5)
    axis.tick_params(axis="y", labelsize=7)
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Greedy atom contribution (bits)")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            if np.isfinite(matrix[row_index, col_index]):
                axis.text(col_index, row_index, f"{matrix[row_index, col_index]:.2f}", ha="center", va="center", fontsize=6, color="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 300}), (".svg", {}), (".pdf", {})):
        fig.savefig(destination.with_suffix(suffix), bbox_inches="tight", **kwargs)


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    subject_count = len(summary["rows"])
    null_replicates = int(summary["config"]["null_replicates"])
    order = int(summary["config"].get("order", 8))
    null_model_count = subject_count * null_replicates
    lines = [
        f"# {subject_count} 名被试 Yeo7 模块级历史 PhiEID greedy 分解",
        "",
        f"每个网络在 {order} 个滞后上的 PC1 值保持为不可拆模块原子。每被试仅报告 greedy 原子中贡献最高的三个核；null p 检验的是对应固定模块集合的 block synergy。",
        "",
        "## 跨被试 top 协同核",
        "",
        "| 核 | 位于 top-3 的被试数 | top 时原子贡献均值（bits） | top 时 p<0.05 的被试数 |",
        "|---|---:|---:|---:|",
    ]
    for core in summary["core_summary"]:
        lines.append(f"| {' + '.join(core['sources'])} | {core['top_frequency']} / {len(summary['rows'])} | {core['mean_atom_value_when_top']:.6f} | {core['subjects_empirical_p_lt_0_05']} / {core['top_frequency']} |")
    lines.extend(["", "## Null greedy 排名对照", "", f"每个 null 均使用与 observed 相同的 greedy top-3 流程。每个 replicate 将 {subject_count} 名被试的同编号 null 组成一个配对 cohort。", "", "| Observed 核 | observed top-3 频率 | null 频率均值；最大值 | empirical p（null 频率 ≥ observed） |", "|---|---:|---:|---:|"])
    for comparison in summary["null_rank_comparison"]:
        lines.append(f"| {' + '.join(comparison['sources'])} | {comparison['observed_frequency']} / {len(summary['rows'])} | {comparison['null_frequency_mean']:.3f}; {comparison['null_frequency_max']} | {comparison['empirical_p']:.6f} |")
    lines.extend(["", "## Null 中最常见的 top-3 核", "", f"| 核 | {null_model_count} 个 null 被试模型中进入 top-3 | top 时原子贡献均值（bits） |", "|---|---:|---:|"])
    for core in summary["null_core_summary"][:10]:
        lines.append(f"| {' + '.join(core['sources'])} | {core['top_frequency']} / {len(summary['rows']) * summary['config']['null_replicates']} | {core['mean_atom_value_when_top']:.6f} |")
    lines.extend(["", "## 每被试 top-3", ""])
    for row in summary["rows"]:
        lines.append(f"### {row['subject']}")
        for atom in row["top_atoms"]:
            lines.append(f"- {' + '.join(atom['sources'])}: atom={atom['value']:.6f} bits; block Phi={atom['block_phi']:.6f}; p={atom['empirical_p']:.6f}.")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    data_root: Path,
    labels: Path,
    output_dir: Path,
    *,
    subjects: Sequence[str] | None = None,
    development_end: int = 900,
    order: int = 8,
    alpha: float = 10.0,
    null_replicates: int = 20,
    seed: int = 20260714,
    top_k: int = 3,
    parcel_count: int = 500,
    data_key: str | None = None,
) -> dict[str, Any]:
    count = int(parcel_count)
    key = data_key or default_data_key(count)
    groups = load_yeo7_groups(labels, expected_parcels=count)
    subjects = tuple(subjects) if subjects is not None else discover_subjects(data_root)
    if not subjects:
        raise FileNotFoundError(f"No HCP subject MAT files found below {data_root}.")
    checkpoint_path = Path(output_dir) / "checkpoint.json"
    rows: list[dict[str, Any]] = []
    if checkpoint_path.is_file():
        rows = list(json.loads(checkpoint_path.read_text(encoding="utf-8")).get("rows", []))
    completed = {str(row["subject"]) for row in rows}
    for subject in subjects:
        if subject in completed:
            continue
        paths = sorted((Path(data_root) / subject).glob("*.mat"))
        if len(paths) != 1:
            raise FileNotFoundError(f"Expected exactly one MAT file for {subject}, found {len(paths)}.")
        raw = load_hcp_series(paths[0], parcel_count=count, data_key=key)
        print(f"[{len(rows) + 1}/{len(subjects)}] {subject}", flush=True)
        rows.append(analyze_subject(raw, groups, subject=subject, development_end=development_end, order=order, alpha=alpha, null_replicates=null_replicates, seed=seed, top_k=top_k))
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    core_summary = summarize_cores(rows)
    null_rank_comparison = [null_rank_frequency(rows, core["sources"], observed_frequency=int(core["top_frequency"])) for core in core_summary]
    null_core_summary = summarize_null_cores(rows)
    summary = {"config": {"subjects": list(subjects), "parcel_count": count, "data_key": key, "labels": str(labels), "network_sizes": {name: len(indices) for name, indices in groups.items()}, "representation": "Yeo7 PC1 fitted separately on each subject development segment", "module_atoms": f"all {order} lagged PC1 values for one network are inseparable", "development_end": int(development_end), "model": "delta Ridge", "order": int(order), "alpha": float(alpha), "null_replicates": int(null_replicates), "seed": int(seed), "top_k_per_subject": int(top_k), "core_significance": "null p applies to the same fixed module subset block synergy, not post-selection greedy atom contribution", "rank_null_test": "complete greedy top-k rerun for every null; matched cohorts pair same replicate index across all analyzed subjects"}, "rows": rows, "core_summary": core_summary, "null_rank_comparison": null_rank_comparison, "null_core_summary": null_core_summary}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(summary, output_dir / "report.md")
    plot_core_consistency(rows, core_summary, output_dir / "top_core_consistency")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--parcel-count", type=int, choices=(500, 1000), default=500)
    parser.add_argument("--data-key", default="", help="MAT variable name; defaults to Schaefer<parcel-count>.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subjects", default="", help="Comma-separated subject IDs; defaults to all available subjects.")
    parser.add_argument("--development-end", type=int, default=900)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--null-replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    subjects = tuple(value.strip() for value in args.subjects.split(",") if value.strip()) or None
    labels = args.labels or default_yeo7_labels(args.parcel_count)
    summary = run(args.data_root, labels, args.output_dir, subjects=subjects, development_end=args.development_end, order=args.order, alpha=args.alpha, null_replicates=args.null_replicates, seed=args.seed, top_k=args.top_k, parcel_count=args.parcel_count, data_key=args.data_key or None)
    print(json.dumps(summary["core_summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
