#!/usr/bin/env python3
"""Run paired 20-null history-source PhiEID comparisons for all HCP Yeo7-PC1 subjects."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hcp_lausanne_phi_eid_pilot import circular_shift_null
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import DEFAULT_DATA, DEFAULT_LABELS, default_data_key, default_yeo7_labels, fit_yeo7_pc1, load_hcp_series, load_yeo7_groups
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import _subject_seed, fit_delta_history_phi, summarize_null


DEFAULT_DATA_ROOT = DEFAULT_DATA.parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "hcp_schaefer500_yeo7_pc1_phi_null_all"


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
) -> dict[str, Any]:
    reduced = fit_yeo7_pc1(np.asarray(raw_series, dtype=float)[:development_end], groups).transform(raw_series)
    observed_fit = fit_delta_history_phi(reduced, alpha=alpha, order=order, development_end=development_end)
    null_values = []
    for replicate in range(null_replicates):
        shifted = circular_shift_null(reduced[:development_end], seed=_subject_seed(seed + int(subject.removeprefix("sub-")), replicate))
        null_values.append(float(fit_delta_history_phi(shifted, alpha=alpha, order=order, development_end=development_end)["phi"]["raw_phi"]))
    observed = float(observed_fit["phi"]["raw_phi"])
    return {
        "subject": subject,
        "observed_raw_phi": observed,
        "joint_ei": float(observed_fit["phi"]["joint_ei"]),
        "singleton_ei_sum": float(observed_fit["phi"]["singleton_ei_sum"]),
        "null_raw_phi": null_values,
        "null_comparison": summarize_null(observed, np.asarray(null_values, dtype=float)),
    }


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    observed = np.asarray([row["observed_raw_phi"] for row in rows], dtype=float)
    deltas = np.asarray([row["null_comparison"]["observed_minus_null_mean"] for row in rows], dtype=float)
    p_values = np.asarray([row["null_comparison"]["empirical_p_ge_observed"] for row in rows], dtype=float)
    return {
        "n_subjects": int(len(rows)),
        "observed_raw_phi_mean": float(observed.mean()),
        "observed_raw_phi_median": float(np.median(observed)),
        "observed_minus_null_mean_mean": float(deltas.mean()),
        "observed_minus_null_mean_median": float(np.median(deltas)),
        "subjects_observed_above_null_mean": int(np.sum(deltas > 0.0)),
        "subjects_empirical_p_lt_0_05": int(np.sum(p_values < 0.05)),
    }


def plot_subject_differences(rows: Sequence[Mapping[str, Any]], destination: Path) -> None:
    ordered = sorted(rows, key=lambda row: float(row["null_comparison"]["observed_minus_null_mean"]))
    subjects = [str(row["subject"]).removeprefix("sub-") for row in ordered]
    deltas = np.asarray([row["null_comparison"]["observed_minus_null_mean"] for row in ordered], dtype=float)
    significant = np.asarray([row["null_comparison"]["empirical_p_ge_observed"] < 0.05 for row in ordered])
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7, "axes.spines.right": False, "axes.spines.top": False})
    height = max(5.0, 0.22 * len(ordered) + 1.2)
    fig, axis = plt.subplots(figsize=(5.2, height), constrained_layout=True)
    positions = np.arange(len(ordered))
    colors = np.where(significant, "#228B63", "#C65B4B")
    axis.hlines(positions, 0.0, deltas, color=colors, linewidth=0.9)
    axis.scatter(deltas, positions, color=colors, s=18, zorder=3)
    axis.axvline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    axis.set(yticks=positions, yticklabels=subjects, xlabel="Observed − mean null $\\Phi^{EID}$ (bits)", ylabel="Subject")
    axis.text(0.02, 0.98, "green: empirical $p<0.05$", transform=axis.transAxes, va="top", fontsize=7)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 300}), (".svg", {}), (".pdf", {})):
        fig.savefig(destination.with_suffix(suffix), bbox_inches="tight", **kwargs)


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    aggregate = summary["aggregate"]
    config = summary["config"]
    lines = [
        f"# {aggregate['n_subjects']} 被试 Yeo7-PC1 history-source PhiEID 与 {config['null_replicates']}-null 比较",
        "",
        f"所有被试均固定使用训练段内拟合的 Yeo7-PC1、前 {config['development_end']} 点、Δ-Ridge `p={config['order']}, alpha={config['alpha']:g}`。"
        "每个 null 对七条 PC1 独立 circular shift 后重拟合相同模型。",
        "",
        "| 被试 | observed Phi（bits） | null mean | observed − null | empirical p |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        comparison = row["null_comparison"]
        lines.append(f"| {row['subject']} | {row['observed_raw_phi']:.6f} | {comparison['null_mean']:.6f} | {comparison['observed_minus_null_mean']:.6f} | {comparison['empirical_p_ge_observed']:.6f} |")
    lines.extend([
        "",
        f"Observed 高于 null mean：{aggregate['subjects_observed_above_null_mean']}/{aggregate['n_subjects']}。",
        f"经验 p<0.05：{aggregate['subjects_empirical_p_lt_0_05']}/{aggregate['n_subjects']}；{config['null_replicates']} 个 null 的最小可得经验 p 为 1/{config['null_replicates'] + 1}={1 / (config['null_replicates'] + 1):.6f}。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    data_root: Path,
    labels: Path,
    output_dir: Path,
    *,
    development_end: int = 900,
    order: int = 8,
    alpha: float = 10.0,
    null_replicates: int = 20,
    seed: int = 20260714,
    parcel_count: int = 500,
    data_key: str | None = None,
    subjects: Sequence[str] | None = None,
) -> dict[str, Any]:
    count = int(parcel_count)
    key = data_key or default_data_key(count)
    groups = load_yeo7_groups(labels, expected_parcels=count)
    files = sorted(Path(data_root).glob("sub-*/*.mat"))
    if subjects is not None:
        requested = {value if value.startswith("sub-") else f"sub-{value}" for value in subjects}
        files = [path for path in files if path.parent.name in requested]
    if not files:
        raise FileNotFoundError(f"No HCP subject MAT files found below {data_root}.")
    checkpoint_path = Path(output_dir) / "checkpoint.json"
    rows: list[dict[str, Any]] = []
    if checkpoint_path.is_file():
        rows = list(json.loads(checkpoint_path.read_text(encoding="utf-8")).get("rows", []))
    completed = {str(row["subject"]) for row in rows}
    for index, path in enumerate(files, start=1):
        if path.parent.name in completed:
            continue
        raw = load_hcp_series(path, parcel_count=count, data_key=key)
        print(f"[{index}/{len(files)}] {path.parent.name}", flush=True)
        rows.append(analyze_subject(raw, groups, subject=path.parent.name, development_end=development_end, order=order, alpha=alpha, null_replicates=null_replicates, seed=seed))
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    summary = {"config": {"parcel_count": count, "data_key": key, "labels": str(labels), "network_sizes": {name: len(indices) for name, indices in groups.items()}, "representation": "Yeo7 PC1 fitted separately on each subject development segment", "development_end": int(development_end), "model": "delta Ridge", "order": int(order), "alpha": float(alpha), "source_dimension": int(order * 7), "target_dimension": 7, "null": "independent non-zero circular shift of each of seven PC1 time series, with model refitting", "null_replicates": int(null_replicates), "seed": int(seed), "empirical_significance_threshold": 0.05}, "rows": rows, "aggregate": aggregate_rows(rows)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(summary, output_dir / "report.md")
    plot_subject_differences(rows, output_dir / "observed_minus_null")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--parcel-count", type=int, choices=(500, 1000), default=500)
    parser.add_argument("--data-key", default="", help="MAT variable name; defaults to Schaefer<parcel-count>.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--development-end", type=int, default=900)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--null-replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--subjects", default="", help="Comma-separated subject IDs; defaults to all available subjects.")
    args = parser.parse_args(argv)
    labels = args.labels or default_yeo7_labels(args.parcel_count)
    subjects = tuple(value.strip() for value in args.subjects.split(",") if value.strip()) or None
    summary = run(args.data_root, labels, args.output_dir, development_end=args.development_end, order=args.order, alpha=args.alpha, null_replicates=args.null_replicates, seed=args.seed, parcel_count=args.parcel_count, data_key=args.data_key or None, subjects=subjects)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
