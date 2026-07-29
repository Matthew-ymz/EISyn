#!/usr/bin/env python3
"""Same-task behavior associations with continuous fixed-coalition synergy.

Every tested brain measure is defined for all 29 HCP subjects. The primary
family uses the absolute fixed-coalition synergy in the matching task state.
A prespecified sensitivity family uses task-minus-REST synergy. Inference uses
shared label permutations, task-wise/global BH-FDR, and task-wise/global max-T.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "results" / "hcp_cognition_exhaustive_targeted_greedy" / "metrics.npz"
BEHAVIOR_PATH = ROOT / "Data" / "unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
OUTPUT_DIR = ROOT / "results" / "hcp_task_score_fixed_synergy"

PERMUTATIONS = 50_000
SEED = 20260728
ALPHA = 0.05

TASKS: tuple[dict[str, str], ...] = (
    {"state": "EMOTION", "label": "Emotion", "field": "Emotion_Task_Acc",
     "score_label": "Emotion accuracy (%)"},
    {"state": "LANGUAGE", "label": "Language", "field": "Language_Task_Acc",
     "score_label": "Language accuracy (%)"},
    {"state": "RELATIONAL", "label": "Relational", "field": "Relational_Task_Acc",
     "score_label": "Relational accuracy (%)"},
    {"state": "SOCIAL", "label": "Social", "field": "derived_social_balanced_accuracy",
     "score_label": "Social balanced accuracy (%)"},
    {"state": "WM", "label": "Working memory", "field": "WM_Task_Acc",
     "score_label": "Working memory accuracy (%)"},
)

SHORT = {
    "Vis": "Vis", "SomMot": "Som", "DorsAttn": "DAN",
    "SalVentAttn": "SVAN", "Limbic": "Lim", "Cont": "Cont",
    "Default": "Def",
}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def short_name(name: str) -> str:
    return "+".join(SHORT[item] for item in name.split("+"))


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=float), method="average")
    centered = ranks - ranks.mean()
    norm = np.sqrt(np.sum(centered**2))
    if norm <= 1e-12:
        raise ValueError("Constant variable")
    return centered / norm


def bh_adjust(values: list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def load_scores(subjects: list[str]) -> dict[str, np.ndarray]:
    with BEHAVIOR_PATH.open(newline="", encoding="utf-8-sig") as handle:
        table = {str(row["Subject"]): row for row in csv.DictReader(handle)}
    scores: dict[str, np.ndarray] = {}
    for task in TASKS:
        if task["state"] == "SOCIAL":
            values = np.asarray([
                0.5 * (
                    float(table[subject]["Social_Task_Random_Perc_Random"])
                    + float(table[subject]["Social_Task_TOM_Perc_TOM"])
                )
                for subject in subjects
            ])
        else:
            values = np.asarray([
                float(table[subject][task["field"]]) for subject in subjects
            ])
        if len(values) != 29 or not np.isfinite(values).all():
            raise ValueError(f"Invalid behavior vector for {task['state']}")
        scores[task["state"]] = values
    return scores


def leave_one_out(x: np.ndarray, y: np.ndarray, rho: float) -> dict[str, float]:
    estimates = np.asarray([
        spearmanr(np.delete(x, index), np.delete(y, index)).statistic
        for index in range(len(x))
    ], dtype=float)
    return {
        "minimum": float(np.min(estimates)),
        "median": float(np.median(estimates)),
        "maximum": float(np.max(estimates)),
        "same_direction_fraction": float(np.mean(np.sign(estimates) == np.sign(rho))),
    }


def analyze() -> tuple[list[dict[str, Any]], dict[str, np.ndarray], np.lib.npyio.NpzFile]:
    archive = np.load(CACHE_PATH)
    states = archive["states"].astype(str).tolist()
    subjects = [str(item).removeprefix("sub-") for item in archive["subjects"]]
    coalitions = archive["coalitions"].astype(str).tolist()
    synergy = np.asarray(archive["fixed_block_synergy"], dtype=float)
    if synergy.shape != (8, 29, 120):
        raise ValueError(f"Unexpected synergy shape: {synergy.shape}")
    if len(subjects) != 29 or len(set(subjects)) != 29:
        raise ValueError("Expected exactly 29 unique subjects")
    if not np.isfinite(synergy).all() or np.any(synergy <= 0):
        raise ValueError("Fixed-coalition synergy must be finite and positive for all subjects")
    scores = load_scores(subjects)

    rng = np.random.default_rng(SEED)
    permutations = np.vstack([rng.permutation(29) for _ in range(PERMUTATIONS)])
    rows: list[dict[str, Any]] = []
    family_global_null = {
        "task_absolute": np.zeros(PERMUTATIONS, dtype=np.float32),
        "task_minus_rest": np.zeros(PERMUTATIONS, dtype=np.float32),
    }

    for task in TASKS:
        state = task["state"]
        x = scores[state]
        x_rank = normalized_ranks(x)
        task_index = states.index(state)
        rest_index = states.index("REST")
        matrices = {
            "task_absolute": synergy[task_index],
            "task_minus_rest": synergy[task_index] - synergy[rest_index],
        }
        for family, values in matrices.items():
            y_rank = np.column_stack([
                normalized_ranks(values[:, column]) for column in range(values.shape[1])
            ])
            observed = np.asarray(x_rank @ y_rank, dtype=float)
            null = np.asarray(x_rank[permutations] @ y_rank, dtype=np.float32)
            task_null_max = np.max(np.abs(null), axis=1)
            family_global_null[family] = np.maximum(
                family_global_null[family], task_null_max
            )
            start = len(rows)
            for column, coalition in enumerate(coalitions):
                rho = float(observed[column])
                p_perm = float(
                    (1 + np.count_nonzero(np.abs(null[:, column]) >= abs(rho)))
                    / (PERMUTATIONS + 1)
                )
                rows.append({
                    "state": state,
                    "task_label": task["label"],
                    "score_field": task["field"],
                    "score_label": task["score_label"],
                    "readout": family,
                    "coalition": coalition,
                    "coalition_short": short_name(coalition),
                    "coalition_size": int(coalition.count("+") + 1),
                    "n_subjects": 29,
                    "all_subjects_finite": True,
                    "rho": rho,
                    "p_asymptotic": float(spearmanr(x, values[:, column]).pvalue),
                    "p_permutation": p_perm,
                    "task_q": None,
                    "global_q_within_readout": None,
                    "p_task_max_t": float(
                        (1 + np.count_nonzero(task_null_max >= abs(rho)))
                        / (PERMUTATIONS + 1)
                    ),
                    "p_global_max_t_within_readout": None,
                    "leave_one_out": leave_one_out(x, values[:, column], rho),
                })
            selected = rows[start:]
            task_q = bh_adjust([row["p_permutation"] for row in selected])
            for row, q in zip(selected, task_q, strict=True):
                row["task_q"] = float(q)

    for family in family_global_null:
        selected = [row for row in rows if row["readout"] == family]
        global_q = bh_adjust([row["p_permutation"] for row in selected])
        global_null = family_global_null[family]
        for row, q in zip(selected, global_q, strict=True):
            row["global_q_within_readout"] = float(q)
            row["p_global_max_t_within_readout"] = float(
                (1 + np.count_nonzero(global_null >= abs(row["rho"])))
                / (PERMUTATIONS + 1)
            )
    return rows, scores, archive


def strongest(rows: list[dict[str, Any]], state: str, readout: str) -> dict[str, Any]:
    selected = [
        row for row in rows if row["state"] == state and row["readout"] == readout
    ]
    return max(selected, key=lambda row: abs(row["rho"]))


def split_half(
    archive: np.lib.npyio.NpzFile,
    scores: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Selection-aware diagnostic: select in 15 subjects, evaluate in 14."""
    states = archive["states"].astype(str).tolist()
    coalitions = archive["coalitions"].astype(str).tolist()
    synergy = np.asarray(archive["fixed_block_synergy"], dtype=float)
    rng = np.random.default_rng(SEED + 1)
    discovery = np.sort(rng.choice(29, size=15, replace=False))
    confirmation = np.setdiff1d(np.arange(29), discovery)
    output: dict[str, Any] = {
        "discovery_indices": discovery.tolist(),
        "confirmation_indices": confirmation.tolist(),
        "tasks": {},
    }
    for task in TASKS:
        state = task["state"]
        x = scores[state]
        values = synergy[states.index(state)]
        discovery_rho = np.asarray([
            spearmanr(x[discovery], values[discovery, column]).statistic
            for column in range(120)
        ])
        column = int(np.argmax(np.abs(discovery_rho)))
        confirmation_test = spearmanr(x[confirmation], values[confirmation, column])
        output["tasks"][state] = {
            "selected_coalition": coalitions[column],
            "discovery_n": 15,
            "discovery_rho": float(discovery_rho[column]),
            "confirmation_n": 14,
            "confirmation_rho": float(confirmation_test.statistic),
            "confirmation_p_asymptotic": float(confirmation_test.pvalue),
            "same_direction": bool(
                np.sign(discovery_rho[column]) == np.sign(confirmation_test.statistic)
            ),
        }
    return output


def build_summary(
    rows: list[dict[str, Any]],
    split: dict[str, Any],
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task in TASKS:
        state = task["state"]
        task_result: dict[str, Any] = {}
        for readout in ("task_absolute", "task_minus_rest"):
            selected = [
                row for row in rows
                if row["state"] == state and row["readout"] == readout
            ]
            task_result[readout] = {
                "strongest": strongest(rows, state, readout),
                "raw_p_lt_0_05": sum(row["p_permutation"] < ALPHA for row in selected),
                "task_fdr_q_lt_0_05": sum(row["task_q"] < ALPHA for row in selected),
                "task_max_t_p_lt_0_05": sum(
                    row["p_task_max_t"] < ALPHA for row in selected
                ),
                "global_fdr_q_lt_0_05": sum(
                    row["global_q_within_readout"] < ALPHA for row in selected
                ),
                "global_max_t_p_lt_0_05": sum(
                    row["p_global_max_t_within_readout"] < ALPHA for row in selected
                ),
            }
        task_result["split_half_absolute"] = split["tasks"][state]
        tasks[state] = task_result
    return {
        "experiment": "Same-task behavior vs continuous fixed-coalition synergy",
        "config": {
            "subjects": 29,
            "coalitions": 120,
            "task_score_pairs": 5,
            "tests_per_readout": 600,
            "permutations": PERMUTATIONS,
            "seed": SEED,
            "primary_readout": "task_absolute",
            "sensitivity_readout": "task_minus_rest",
            "correlation": "two-sided Spearman",
            "multiplicity": (
                "task-wise and global-within-readout BH-FDR plus permutation max-T"
            ),
        },
        "all_29_audit": {
            "every_test_n_29": all(row["n_subjects"] == 29 for row in rows),
            "every_value_finite": all(row["all_subjects_finite"] for row in rows),
            "total_associations": len(rows),
        },
        "tasks": tasks,
        "split_half": split,
        "interpretation": (
            "Corrected p/q values support confirmatory inference. Raw p<0.05 "
            "associations are exploratory. The split-half result evaluates whether "
            "a data-selected coalition preserves its direction in held-out subjects."
        ),
    }


def plot_scatter(
    rows: list[dict[str, Any]],
    scores: dict[str, np.ndarray],
    archive: np.lib.npyio.NpzFile,
) -> None:
    style()
    states = archive["states"].astype(str).tolist()
    coalitions = archive["coalitions"].astype(str).tolist()
    synergy = np.asarray(archive["fixed_block_synergy"], dtype=float)
    figure, axes = plt.subplots(2, 3, figsize=(7.4, 4.7), constrained_layout=True)
    axes_flat = axes.ravel()
    colors = ["#4477AA", "#228833", "#CC6677", "#AA3377", "#EE7733"]
    for axis, task, color in zip(axes_flat, TASKS, colors, strict=False):
        row = strongest(rows, task["state"], "task_absolute")
        column = coalitions.index(row["coalition"])
        x = scores[task["state"]]
        y = synergy[states.index(task["state"]), :, column]
        axis.scatter(x, y, s=20, color=color, alpha=0.82, edgecolor="white", linewidth=0.35)
        slope, intercept = np.polyfit(x, y, deg=1)
        grid = np.linspace(x.min(), x.max(), 100)
        axis.plot(grid, slope * grid + intercept, color="#59636E", lw=1, ls="--")
        y_range = float(np.ptp(y))
        axis.set_ylim(float(y.min() - 0.08 * y_range), float(y.max() + 0.28 * y_range))
        axis.set_title(f"{task['label']} · {row['coalition_short']}", loc="left", weight="bold")
        axis.set_xlabel(task["score_label"])
        axis.set_ylabel("Fixed-coalition synergy (bits)")
        axis.text(
            0.02, 0.98,
            (
                rf"$\rho$={row['rho']:+.3f} · perm. $p$={row['p_permutation']:.4f}"
                "\n"
                rf"task max-$T$ $p$={row['p_task_max_t']:.3f} · $n$=29"
            ),
            transform=axis.transAxes, ha="left", va="top", fontsize=6.2, color="#45515C",
        )
    axes_flat[-1].axis("off")
    figure.suptitle(
        "Task-matched performance and strongest continuous coalition synergy",
        x=0.01, ha="left", fontsize=9, weight="bold",
    )
    for extension in ("png", "svg", "pdf"):
        figure.savefig(
            OUTPUT_DIR / f"top_absolute_scatter.{extension}",
            dpi=300, bbox_inches="tight", facecolor="white",
        )
    plt.close(figure)


def plot_overview(rows: list[dict[str, Any]]) -> None:
    style()
    readouts = ("task_absolute", "task_minus_rest")
    matrix = np.asarray([
        [strongest(rows, task["state"], readout)["rho"] for readout in readouts]
        for task in TASKS
    ])
    figure, (heatmap, counts) = plt.subplots(
        1, 2, figsize=(7.2, 3.45), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.3, 2.1]},
    )
    image = heatmap.imshow(matrix, cmap="RdBu_r", vmin=-0.55, vmax=0.55, aspect="auto")
    heatmap.set_xticks([0, 1], ["Task state", "Task − REST"])
    heatmap.xaxis.tick_top()
    heatmap.set_yticks(range(5), [task["label"] for task in TASKS])
    heatmap.tick_params(length=0)
    for i, task in enumerate(TASKS):
        for j, readout in enumerate(readouts):
            row = strongest(rows, task["state"], readout)
            heatmap.text(j, i - 0.12, f"{row['rho']:+.2f}", ha="center", va="center",
                         color="white" if abs(row["rho"]) > 0.34 else "#263238",
                         fontsize=7, weight="bold")
            heatmap.text(j, i + 0.18, row["coalition_short"], ha="center", va="center",
                         color="white" if abs(row["rho"]) > 0.34 else "#263238",
                         fontsize=4.8)
    bar = figure.colorbar(image, ax=heatmap, orientation="horizontal", fraction=0.07, pad=0.08)
    bar.set_label(r"Strongest Spearman $\rho$ among 120 coalitions")

    x = np.arange(5)
    width = 0.24
    raw = np.asarray([
        sum(
            row["state"] == task["state"]
            and row["readout"] == "task_absolute"
            and row["p_permutation"] < ALPHA
            for row in rows
        )
        for task in TASKS
    ])
    fdr = np.asarray([
        sum(
            row["state"] == task["state"]
            and row["readout"] == "task_absolute"
            and row["global_q_within_readout"] < ALPHA
            for row in rows
        )
        for task in TASKS
    ])
    maxt = np.asarray([
        sum(
            row["state"] == task["state"]
            and row["readout"] == "task_absolute"
            and row["p_global_max_t_within_readout"] < ALPHA
            for row in rows
        )
        for task in TASKS
    ])
    counts.bar(x - width, raw, width, color="#D9A066", label="Raw permutation $p<0.05$")
    counts.bar(x, fdr, width, color="#4C78A8", label="Global FDR $q<0.05$")
    counts.bar(x + width, maxt, width, color="#2A9D8F", label="Global max-$T$ $p<0.05$")
    counts.set_xticks(x, [task["label"].replace("Working memory", "WM") for task in TASKS],
                      rotation=20, ha="right")
    counts.set_ylabel("Number of absolute-state coalitions")
    counts.set_title("Multiplicity-controlled evidence", loc="left", weight="bold")
    counts.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    for extension in ("png", "svg", "pdf"):
        figure.savefig(
            OUTPUT_DIR / f"fixed_synergy_overview.{extension}",
            dpi=300, bbox_inches="tight", facecolor="white",
        )
    plt.close(figure)


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Same-task performance–synergy experiment",
        "",
        "All reported correlations use the same 29 subjects. The primary analysis "
        "tests 120 continuously defined fixed coalitions in each matching task state "
        "(600 tests). Task-minus-REST is a separate sensitivity family.",
        "",
        "## Primary task-state results",
        "",
        "| Task | Strongest coalition | rho | permutation p | task FDR q | "
        "task max-T p | global FDR q | global max-T p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        row = summary["tasks"][task["state"]]["task_absolute"]["strongest"]
        lines.append(
            f"| {task['label']} | {row['coalition_short']} | {row['rho']:+.3f} | "
            f"{row['p_permutation']:.5f} | {row['task_q']:.4f} | "
            f"{row['p_task_max_t']:.4f} | {row['global_q_within_readout']:.4f} | "
            f"{row['p_global_max_t_within_readout']:.4f} |"
        )
    lines += [
        "",
        "## Task-minus-REST sensitivity",
        "",
        "| Task | Strongest coalition | rho | permutation p | task FDR q | "
        "task max-T p | global FDR q | global max-T p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        row = summary["tasks"][task["state"]]["task_minus_rest"]["strongest"]
        lines.append(
            f"| {task['label']} | {row['coalition_short']} | {row['rho']:+.3f} | "
            f"{row['p_permutation']:.5f} | {row['task_q']:.4f} | "
            f"{row['p_task_max_t']:.4f} | {row['global_q_within_readout']:.4f} | "
            f"{row['p_global_max_t_within_readout']:.4f} |"
        )
    lines += [
        "",
        "## Selection-aware split-half diagnostic",
        "",
        "| Task | Coalition selected in n=15 | discovery rho | held-out rho (n=14) | "
        "held-out p | same direction |",
        "|---|---|---:|---:|---:|---|",
    ]
    for task in TASKS:
        item = summary["tasks"][task["state"]]["split_half_absolute"]
        lines.append(
            f"| {task['label']} | {short_name(item['selected_coalition'])} | "
            f"{item['discovery_rho']:+.3f} | {item['confirmation_rho']:+.3f} | "
            f"{item['confirmation_p_asymptotic']:.4f} | "
            f"{'yes' if item['same_direction'] else 'no'} |"
        )
    lines += [
        "",
        "## Interpretation rule",
        "",
        "Only multiplicity-corrected results are treated as confirmatory. Raw "
        "permutation p-values are retained as exploratory evidence and are not "
        "relabelled as significant after searching 120 coalitions.",
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, scores, archive = analyze()
    split = split_half(archive, scores)
    summary = build_summary(rows, split)
    with (OUTPUT_DIR / "associations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(summary)
    plot_scatter(rows, scores, archive)
    plot_overview(rows)
    print(json.dumps({
        "output_dir": str(OUTPUT_DIR),
        "associations": len(rows),
        "all_29": summary["all_29_audit"],
        "primary": {
            state: summary["tasks"][state]["task_absolute"]
            for state in [task["state"] for task in TASKS]
        },
    }, indent=2))


if __name__ == "__main__":
    main()
