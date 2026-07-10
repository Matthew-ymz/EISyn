from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from run_hcp_lausanne_phi_eid_pilot import MODULE_COLORS, MODULE_ORDER


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REST = ROOT / "results" / "hcp_lausanne_phi_eid_pilot" / "robustness_summary.json"
DEFAULT_WM = ROOT / "results" / "hcp_lausanne_phi_eid_wm" / "robustness_summary.json"
DEFAULT_OUTPUT = ROOT / "results" / "hcp_lausanne_phi_eid_rest_vs_wm_comparison.json"
DEFAULT_FIGURE = ROOT / "fig" / "hcp_lausanne_phi_eid_rest_vs_wm_atoms"
DEFAULT_REPORT = ROOT / "docs" / "reports" / "HCP_Lausanne83_REST_vs_WM_PhiEID_comparison.md"


def mean_or_nan(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan")
    return float(np.mean(array))


def whole_summary(summary: Mapping[str, object]) -> dict[str, float]:
    rows = list(summary.get("whole_phi", []))
    return {
        "observed_mean": mean_or_nan([float(row["observed"]) for row in rows]),
        "null_mean": mean_or_nan([float(row["null_mean"]) for row in rows]),
        "difference_mean": mean_or_nan([float(row["difference"]) for row in rows]),
        "median_empirical_p": float(np.median([float(row["empirical_p"]) for row in rows])) if rows else float("nan"),
        "paired_subjects": int(summary.get("run_reliability", {}).get("paired_subjects", 0)),
        "lr_rl_phi_diff_pearson": float(summary.get("run_reliability", {}).get("phi_diff_pearson", float("nan"))),
        "mean_top5_atom_overlap": float(summary.get("run_reliability", {}).get("mean_top5_atom_overlap", float("nan"))),
        "mean_module_participation_pearson": float(
            summary.get("run_reliability", {}).get("mean_atom_module_participation_pearson", float("nan"))
        ),
    }


def ranked_tests(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(row["label"]): {**dict(row), "rank": rank}
        for rank, row in enumerate(rows, start=1)
    }


def compare_atom_tests(
    rest_atoms: Sequence[Mapping[str, object]],
    wm_atoms: Sequence[Mapping[str, object]],
    *,
    top_k: int = 10,
) -> dict[str, object]:
    rest_ranked = ranked_tests(rest_atoms)
    wm_ranked = ranked_tests(wm_atoms)
    rest_top = [str(row["label"]) for row in rest_atoms[:top_k]]
    wm_top = [str(row["label"]) for row in wm_atoms[:top_k]]
    rest_set = set(rest_top)
    wm_set = set(wm_top)
    union = rest_set | wm_set
    overlap = float(len(rest_set & wm_set) / max(1, len(union)))
    labels = sorted(
        set(rest_ranked) | set(wm_ranked),
        key=lambda label: max(
            float(rest_ranked.get(label, {}).get("difference", 0.0)),
            float(wm_ranked.get(label, {}).get("difference", 0.0)),
        ),
        reverse=True,
    )
    rows = []
    for label in labels:
        rest = rest_ranked.get(label)
        wm = wm_ranked.get(label)
        rest_rank = None if rest is None else int(rest["rank"])
        wm_rank = None if wm is None else int(wm["rank"])
        rows.append(
            {
                "label": label,
                "rest_rank": rest_rank,
                "wm_rank": wm_rank,
                "rank_shift_wm_minus_rest": None if rest_rank is None or wm_rank is None else int(wm_rank - rest_rank),
                "rest_observed_mean": 0.0 if rest is None else float(rest["observed_mean"]),
                "wm_observed_mean": 0.0 if wm is None else float(wm["observed_mean"]),
                "rest_difference": 0.0 if rest is None else float(rest["difference"]),
                "wm_difference": 0.0 if wm is None else float(wm["difference"]),
                "rest_fdr_q": None if rest is None else float(rest.get("fdr_q", float("nan"))),
                "wm_fdr_q": None if wm is None else float(wm.get("fdr_q", float("nan"))),
            }
        )
    return {"top_k": int(top_k), "top_k_overlap": overlap, "rows": rows}


def compare_module_participation(
    rest_rows: Sequence[Mapping[str, object]],
    wm_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rest_by_label = {str(row["label"]): row for row in rest_rows}
    wm_by_label = {str(row["label"]): row for row in wm_rows}
    rows = []
    for label in MODULE_ORDER:
        rest = rest_by_label.get(label, {})
        wm = wm_by_label.get(label, {})
        rows.append(
            {
                "label": label,
                "rest_observed_mean": float(rest.get("observed_mean", 0.0)),
                "wm_observed_mean": float(wm.get("observed_mean", 0.0)),
                "rest_difference": float(rest.get("difference", 0.0)),
                "wm_difference": float(wm.get("difference", 0.0)),
                "wm_minus_rest_difference": float(wm.get("difference", 0.0)) - float(rest.get("difference", 0.0)),
                "rest_fdr_q": None if not rest else float(rest.get("fdr_q", float("nan"))),
                "wm_fdr_q": None if not wm else float(wm.get("fdr_q", float("nan"))),
            }
        )
    return rows


def atom_order_distribution(atom_rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in atom_rows:
        order = str(len(str(row["label"]).split("+")))
        totals[order] = totals.get(order, 0.0) + float(row.get("observed_mean", 0.0))
    return totals


def build_comparison(rest_summary: Mapping[str, object], wm_summary: Mapping[str, object]) -> dict[str, object]:
    atom_comparison = compare_atom_tests(
        list(rest_summary.get("atom_tests", [])),
        list(wm_summary.get("atom_tests", [])),
        top_k=10,
    )
    return {
        "rest": whole_summary(rest_summary),
        "wm": whole_summary(wm_summary),
        "whole_delta": {
            "wm_minus_rest_observed_mean": whole_summary(wm_summary)["observed_mean"] - whole_summary(rest_summary)["observed_mean"],
            "wm_minus_rest_difference_mean": whole_summary(wm_summary)["difference_mean"] - whole_summary(rest_summary)["difference_mean"],
        },
        "atom_comparison": atom_comparison,
        "module_participation_comparison": compare_module_participation(
            list(rest_summary.get("module_participation_tests", [])),
            list(wm_summary.get("module_participation_tests", [])),
        ),
        "atom_order_distribution": {
            "rest": atom_order_distribution(list(rest_summary.get("atom_tests", []))),
            "wm": atom_order_distribution(list(wm_summary.get("atom_tests", []))),
        },
        "config": {
            "rest_null_reps": int(rest_summary.get("null_reps", 0)),
            "wm_null_reps": int(wm_summary.get("null_reps", 0)),
        },
    }


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def plot_comparison(comparison: Mapping[str, object], figure_base: Path) -> None:
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8})
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    atom_rows = list(comparison["atom_comparison"]["rows"])[:10]
    participation_rows = list(comparison["module_participation_comparison"])
    order_rest = comparison["atom_order_distribution"]["rest"]
    order_wm = comparison["atom_order_distribution"]["wm"]

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.8), constrained_layout=True)
    ax = axes[0, 0]
    whole_labels = ["Observed", "Observed - null"]
    rest_values = [comparison["rest"]["observed_mean"], comparison["rest"]["difference_mean"]]
    wm_values = [comparison["wm"]["observed_mean"], comparison["wm"]["difference_mean"]]
    x = np.arange(len(whole_labels))
    ax.bar(x - 0.18, rest_values, width=0.35, color="#7F7F7F", label="REST1")
    ax.bar(x + 0.18, wm_values, width=0.35, color="#2F7D5A", label="WM")
    ax.set_xticks(x)
    ax.set_xticklabels(whole_labels)
    ax.set_ylabel("Raw PhiEID (bits)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax = axes[0, 1]
    labels = [str(row["label"]) for row in atom_rows]
    y = np.arange(len(labels))
    ax.barh(y + 0.18, [float(row["rest_difference"]) for row in atom_rows], height=0.35, color="#7F7F7F", label="REST1")
    ax.barh(y - 0.18, [float(row["wm_difference"]) for row in atom_rows], height=0.35, color="#2F7D5A", label="WM")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Observed - null")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax = axes[1, 0]
    labels = [str(row["label"]) for row in participation_rows]
    y = np.arange(len(labels))
    colors = [MODULE_COLORS.get(label, "#2F7D5A") for label in labels]
    ax.barh(y, [float(row["wm_minus_rest_difference"]) for row in participation_rows], color=colors)
    ax.axvline(0.0, color="#7F7F7F", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("WM - REST participation difference")

    ax = axes[1, 1]
    orders = sorted(set(order_rest) | set(order_wm), key=lambda item: int(item))
    x = np.arange(len(orders))
    ax.bar(x - 0.18, [float(order_rest.get(order, 0.0)) for order in orders], width=0.35, color="#7F7F7F", label="REST1")
    ax.bar(x + 0.18, [float(order_wm.get(order, 0.0)) for order in orders], width=0.35, color="#2F7D5A", label="WM")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{order}-module" for order in orders], rotation=30, ha="right")
    ax.set_ylabel("Mean atom PhiEID")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(str(figure_base) + suffix, dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_comparison_report(path: Path, comparison: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atom_rows = list(comparison["atom_comparison"]["rows"])[:12]
    participation_rows = list(comparison["module_participation_comparison"])
    atom_lines = "\n".join(
        "| {label} | {rest_rank} | {wm_rank} | {rest_diff} | {wm_diff} | {rest_q} | {wm_q} |".format(
            label=row["label"],
            rest_rank="" if row["rest_rank"] is None else row["rest_rank"],
            wm_rank="" if row["wm_rank"] is None else row["wm_rank"],
            rest_diff=fmt(row["rest_difference"]),
            wm_diff=fmt(row["wm_difference"]),
            rest_q=fmt(row["rest_fdr_q"]),
            wm_q=fmt(row["wm_fdr_q"]),
        )
        for row in atom_rows
    )
    participation_lines = "\n".join(
        f"| {row['label']} | {fmt(row['rest_difference'])} | {fmt(row['wm_difference'])} | {fmt(row['wm_minus_rest_difference'])} | {fmt(row['rest_fdr_q'])} | {fmt(row['wm_fdr_q'])} |"
        for row in participation_rows
    )
    order_lines = "\n".join(
        f"| {order} | {fmt(comparison['atom_order_distribution']['rest'].get(order, 0.0))} | {fmt(comparison['atom_order_distribution']['wm'].get(order, 0.0))} |"
        for order in sorted(
            set(comparison["atom_order_distribution"]["rest"]) | set(comparison["atom_order_distribution"]["wm"]),
            key=lambda item: int(item),
        )
    )
    path.write_text(
        "\n".join(
            [
                "# REST vs Working Memory PhiEID Comparison",
                "",
                "This report compares HCP REST1 and Working Memory task PhiEID using the same Lausanne-83 ROI pipeline, Ridge one-step transition model, circular-shift null, and greedy module atom decomposition.",
                "",
                "## Whole-state PhiEID",
                "",
                "| Condition | Observed mean | Null mean | Difference mean | Median empirical p | LR/RL r | Top-5 atom overlap |",
                "|---|---:|---:|---:|---:|---:|---:|",
                f"| REST1 | {fmt(comparison['rest']['observed_mean'])} | {fmt(comparison['rest']['null_mean'])} | {fmt(comparison['rest']['difference_mean'])} | {fmt(comparison['rest']['median_empirical_p'])} | {fmt(comparison['rest']['lr_rl_phi_diff_pearson'])} | {fmt(comparison['rest']['mean_top5_atom_overlap'])} |",
                f"| Working Memory | {fmt(comparison['wm']['observed_mean'])} | {fmt(comparison['wm']['null_mean'])} | {fmt(comparison['wm']['difference_mean'])} | {fmt(comparison['wm']['median_empirical_p'])} | {fmt(comparison['wm']['lr_rl_phi_diff_pearson'])} | {fmt(comparison['wm']['mean_top5_atom_overlap'])} |",
                "",
                "## Module atom distribution",
                "",
                f"Top-10 atom Jaccard overlap between REST1 and Working Memory is `{fmt(comparison['atom_comparison']['top_k_overlap'])}`.",
                "",
                "| Module atom | REST rank | WM rank | REST difference | WM difference | REST FDR q | WM FDR q |",
                "|---|---:|---:|---:|---:|---:|---:|",
                atom_lines,
                "",
                "## Module participation",
                "",
                "| Module | REST difference | WM difference | WM - REST | REST FDR q | WM FDR q |",
                "|---|---:|---:|---:|---:|---:|",
                participation_lines,
                "",
                "## Atom order distribution",
                "",
                "| Atom order | REST observed mean | WM observed mean |",
                "|---:|---:|---:|",
                order_lines,
                "",
                "## Interpretation rule",
                "",
                "Treat differences as strongest when the corresponding atom or module is above null after FDR correction and has meaningful LR/RL stability in both conditions. Exact greedy atom labels remain less stable than module participation, so the safest contrast is the network-family shift rather than a single exact atom.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    rest_summary = json.loads(Path(args.rest).expanduser().resolve().read_text(encoding="utf-8"))
    wm_summary = json.loads(Path(args.wm).expanduser().resolve().read_text(encoding="utf-8"))
    comparison = build_comparison(rest_summary, wm_summary)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    plot_comparison(comparison, Path(args.figure_base).expanduser().resolve())
    write_comparison_report(Path(args.report).expanduser().resolve(), comparison)
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare HCP REST1 and Working Memory PhiEID summaries.")
    parser.add_argument("--rest", default=str(DEFAULT_REST))
    parser.add_argument("--wm", default=str(DEFAULT_WM))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--figure-base", default=str(DEFAULT_FIGURE))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
