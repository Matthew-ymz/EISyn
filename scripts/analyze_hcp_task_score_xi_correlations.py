#!/usr/bin/env python3
"""Task-matched correlations between five HCP scores and Xi decompositions.

The experiment tests five prespecified state-score pairs.  Within each pair it
screens system Xi, cross-network Xi, seven absolute network contributions, and
120 absolute hierarchy-atom contributions.  Inference uses a shared subject
permutation across tasks, task-wise and global max-T control, and global
Benjamini-Hochberg FDR.  Sparse atoms with fewer than five positive subjects are
reported but excluded from the primary multiplicity family.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
ARRAYS_PATH = (
    ROOT
    / "results"
    / "hcp_schaefer500_task_evoked_xi_tuning"
    / "full"
    / "k1_p3_a1"
    / "arrays.npz"
)
BEHAVIOR_PATH = ROOT / "Data" / "unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
OUTPUT_DIR = ROOT / "results" / "hcp_task_score_xi_correlations"

PERMUTATIONS = 50_000
SEED = 20260728
MIN_ATOM_SUPPORT = 5
ALPHA = 0.05
ZERO_TOLERANCE = 1.0e-12

TASK_SPECS: tuple[dict[str, str], ...] = (
    {
        "state": "EMOTION",
        "label": "Emotion",
        "score_label": "Emotion accuracy (%)",
        "score_field": "Emotion_Task_Acc",
    },
    {
        "state": "LANGUAGE",
        "label": "Language",
        "score_label": "Language accuracy (%)",
        "score_field": "Language_Task_Acc",
    },
    {
        "state": "RELATIONAL",
        "label": "Relational",
        "score_label": "Relational accuracy (%)",
        "score_field": "Relational_Task_Acc",
    },
    {
        "state": "SOCIAL",
        "label": "Social",
        "score_label": "Social balanced accuracy (%)",
        "score_field": "derived_social_balanced_accuracy",
    },
    {
        "state": "WM",
        "label": "Working memory",
        "score_label": "Working memory accuracy (%)",
        "score_field": "WM_Task_Acc",
    },
)

NETWORK_SHORT = {
    "Vis": "Vis",
    "SomMot": "Som",
    "DorsAttn": "DAN",
    "SalVentAttn": "SVAN",
    "Limbic": "Lim",
    "Cont": "Cont",
    "Default": "Def",
}
FAMILY_LABELS = {
    "system": r"System $\Xi$",
    "cross": r"Cross-network $\Xi$",
    "network": "Best network",
    "atom": "Best hierarchy atom",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def short_combination(name: str) -> str:
    return "+".join(NETWORK_SHORT[item] for item in name.split("+"))


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        return values
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def normalized_ranks(values: np.ndarray) -> np.ndarray | None:
    ranks = rankdata(np.asarray(values, dtype=float), method="average")
    centered = ranks - ranks.mean()
    norm = float(np.sqrt(np.sum(centered**2)))
    if norm <= ZERO_TOLERANCE:
        return None
    return centered / norm


def load_behavior(subjects: list[str]) -> dict[str, np.ndarray]:
    subject_set = set(subjects)
    with BEHAVIOR_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = {
            str(row["Subject"]): row
            for row in csv.DictReader(handle)
            if str(row["Subject"]) in subject_set
        }
    missing = [subject for subject in subjects if subject not in rows]
    if missing:
        raise ValueError(f"Missing behavioral rows: {missing}")

    output: dict[str, np.ndarray] = {}
    for spec in TASK_SPECS:
        if spec["state"] == "SOCIAL":
            values = np.asarray(
                [
                    0.5
                    * (
                        float(rows[subject]["Social_Task_Random_Perc_Random"])
                        + float(rows[subject]["Social_Task_TOM_Perc_TOM"])
                    )
                    for subject in subjects
                ],
                dtype=float,
            )
        else:
            values = np.asarray(
                [float(rows[subject][spec["score_field"]]) for subject in subjects],
                dtype=float,
            )
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite task scores for {spec['state']}")
        output[spec["state"]] = values
    return output


def build_metrics(
    archive: np.lib.npyio.NpzFile, state_index: int
) -> tuple[list[dict[str, Any]], np.ndarray]:
    system = np.asarray(archive["system_xi"][state_index], dtype=float)
    cross = np.asarray(archive["cross_xi"][state_index], dtype=float)
    network_names = archive["networks"].astype(str).tolist()
    atom_names = archive["atom_names"].astype(str).tolist()
    network_absolute = (
        np.asarray(archive["network_share"][state_index], dtype=float)
        * system[:, None]
    )
    atom_absolute = (
        np.asarray(archive["atom_share"][state_index], dtype=float)
        * cross[:, None]
    )

    metadata: list[dict[str, Any]] = [
        {
            "family": "system",
            "name": "system_xi",
            "label": r"System $\Xi$",
            "short_label": r"System $\Xi$",
        },
        {
            "family": "cross",
            "name": "cross_network_xi",
            "label": r"Cross-network $\Xi$",
            "short_label": r"Cross $\Xi$",
        },
    ]
    columns = [system, cross]
    for index, name in enumerate(network_names):
        metadata.append(
            {
                "family": "network",
                "name": name,
                "label": f"{name} absolute contribution",
                "short_label": f"{NETWORK_SHORT[name]} network",
            }
        )
        columns.append(network_absolute[:, index])
    for index, name in enumerate(atom_names):
        metadata.append(
            {
                "family": "atom",
                "name": name,
                "label": f"{name} hierarchy-atom contribution",
                "short_label": short_combination(name),
            }
        )
        columns.append(atom_absolute[:, index])
    values = np.column_stack(columns)
    if values.shape != (len(system), 129):
        raise AssertionError(f"Expected 129 metrics, got {values.shape}")
    return metadata, values


def leave_one_out(x: np.ndarray, y: np.ndarray, rho: float) -> dict[str, float]:
    estimates = []
    for index in range(len(x)):
        keep = np.arange(len(x)) != index
        if normalized_ranks(x[keep]) is None or normalized_ranks(y[keep]) is None:
            continue
        value = float(spearmanr(x[keep], y[keep]).statistic)
        if np.isfinite(value):
            estimates.append(value)
    array = np.asarray(estimates, dtype=float)
    return {
        "rho_min": float(array.min()),
        "rho_median": float(np.median(array)),
        "rho_max": float(array.max()),
        "same_direction_fraction": float(np.mean(np.sign(array) == np.sign(rho))),
    }


def analyze() -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    archive = np.load(ARRAYS_PATH)
    states = archive["states"].astype(str).tolist()
    subjects = [str(value).removeprefix("sub-") for value in archive["subjects"]]
    if len(subjects) != 29 or len(set(subjects)) != 29:
        raise ValueError("Expected 29 unique imaging subjects")
    scores = load_behavior(subjects)

    rng = np.random.default_rng(SEED)
    permutation_indices = np.vstack(
        [rng.permutation(len(subjects)) for _ in range(PERMUTATIONS)]
    )
    global_null_max = np.zeros(PERMUTATIONS, dtype=np.float32)
    task_null_max: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    row_ranges: dict[str, tuple[int, int]] = {}

    for spec in TASK_SPECS:
        state = spec["state"]
        state_index = states.index(state)
        metadata, metric_values = build_metrics(archive, state_index)
        x = scores[state]
        x_rank = normalized_ranks(x)
        if x_rank is None:
            raise ValueError(f"Constant task score for {state}")
        y_rank_columns = []
        valid_indices = []
        for metric_index in range(metric_values.shape[1]):
            normalized = normalized_ranks(metric_values[:, metric_index])
            if normalized is not None:
                valid_indices.append(metric_index)
                y_rank_columns.append(normalized)
        y_rank = np.column_stack(y_rank_columns)
        observed_valid = np.asarray(x_rank @ y_rank, dtype=float)
        null_valid = np.asarray(
            x_rank[permutation_indices] @ y_rank, dtype=np.float32
        )
        valid_lookup = {metric_index: i for i, metric_index in enumerate(valid_indices)}

        start = len(rows)
        eligible_valid_positions = []
        for metric_index, meta in enumerate(metadata):
            y = metric_values[:, metric_index]
            nonzero = int(np.count_nonzero(np.abs(y) > ZERO_TOLERANCE))
            unique = int(np.unique(y).size)
            valid = metric_index in valid_lookup
            eligible = bool(
                valid
                and unique >= 3
                and (meta["family"] != "atom" or nonzero >= MIN_ATOM_SUPPORT)
            )
            if valid:
                valid_position = valid_lookup[metric_index]
                rho = float(observed_valid[valid_position])
                raw_asymptotic = float(spearmanr(x, y).pvalue)
                p_perm = float(
                    (1 + np.count_nonzero(np.abs(null_valid[:, valid_position]) >= abs(rho)))
                    / (PERMUTATIONS + 1)
                )
                if eligible:
                    eligible_valid_positions.append(valid_position)
            else:
                rho = raw_asymptotic = p_perm = float("nan")
            rows.append(
                {
                    "state": state,
                    "task_label": spec["label"],
                    "score_field": spec["score_field"],
                    "score_label": spec["score_label"],
                    "metric_index": metric_index,
                    **meta,
                    "n_subjects": len(subjects),
                    "nonzero_subjects": nonzero,
                    "unique_values": unique,
                    "eligible_primary_family": eligible,
                    "rho": rho,
                    "p_raw_asymptotic": raw_asymptotic,
                    "p_permutation": p_perm,
                    "task_q": None,
                    "global_q": None,
                    "p_task_max_t": None,
                    "p_global_max_t": None,
                    "leave_one_out": leave_one_out(x, y, rho) if valid else None,
                }
            )
        stop = len(rows)
        row_ranges[state] = (start, stop)
        eligible_null = np.abs(null_valid[:, eligible_valid_positions])
        current_task_max = np.max(eligible_null, axis=1)
        task_null_max[state] = current_task_max
        global_null_max = np.maximum(global_null_max, current_task_max)

        eligible_rows = [
            row for row in rows[start:stop] if row["eligible_primary_family"]
        ]
        task_q = bh_adjust([float(row["p_permutation"]) for row in eligible_rows])
        for row, q_value in zip(eligible_rows, task_q, strict=True):
            row["task_q"] = float(q_value)
            row["p_task_max_t"] = float(
                (1 + np.count_nonzero(current_task_max >= abs(float(row["rho"]))))
                / (PERMUTATIONS + 1)
            )

    eligible_rows = [row for row in rows if row["eligible_primary_family"]]
    global_q = bh_adjust([float(row["p_permutation"]) for row in eligible_rows])
    for row, q_value in zip(eligible_rows, global_q, strict=True):
        row["global_q"] = float(q_value)
        row["p_global_max_t"] = float(
            (1 + np.count_nonzero(global_null_max >= abs(float(row["rho"]))))
            / (PERMUTATIONS + 1)
        )
    return rows, scores, {
        "subjects": np.asarray(subjects),
        "global_null_max": global_null_max,
        **{f"{state}_null_max": values for state, values in task_null_max.items()},
    }


def strongest(
    rows: Iterable[dict[str, Any]], *, family: str | None = None
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row["eligible_primary_family"]
        and (family is None or row["family"] == family)
    ]
    return max(candidates, key=lambda row: abs(float(row["rho"])))


def task_rows(rows: list[dict[str, Any]], state: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["state"] == state]


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for spec in TASK_SPECS:
        selected = task_rows(rows, spec["state"])
        top = strongest(selected)
        by_family = {
            family: strongest(selected, family=family)
            for family in ("system", "cross", "network", "atom")
        }
        tasks[spec["state"]] = {
            "score_label": spec["score_label"],
            "n_tested": len(selected),
            "n_primary_eligible": int(
                sum(bool(row["eligible_primary_family"]) for row in selected)
            ),
            "raw_p_lt_0_05": int(
                sum(
                    bool(row["eligible_primary_family"])
                    and float(row["p_permutation"]) < ALPHA
                    for row in selected
                )
            ),
            "task_fdr_q_lt_0_05": int(
                sum(
                    row["task_q"] is not None and float(row["task_q"]) < ALPHA
                    for row in selected
                )
            ),
            "global_fdr_q_lt_0_05": int(
                sum(
                    row["global_q"] is not None and float(row["global_q"]) < ALPHA
                    for row in selected
                )
            ),
            "global_max_t_p_lt_0_05": int(
                sum(
                    row["p_global_max_t"] is not None
                    and float(row["p_global_max_t"]) < ALPHA
                    for row in selected
                )
            ),
            "strongest_overall": top,
            "strongest_by_family": by_family,
        }
    return {
        "experiment": "Task-matched HCP score–Xi correlations",
        "config": {
            "subjects": 29,
            "task_score_pairs": [spec["state"] for spec in TASK_SPECS],
            "metrics_per_task": 129,
            "total_metrics": len(rows),
            "permutations": PERMUTATIONS,
            "permutation_seed": SEED,
            "shared_subject_permutation_across_tasks": True,
            "minimum_atom_support": MIN_ATOM_SUPPORT,
            "primary_inference": (
                "absolute contributions; global permutation max-T and global "
                "Benjamini-Hochberg FDR across primary-eligible metrics"
            ),
            "brain_model": "existing k=1, p=3, alpha=1 affine/Gaussian Xi cache",
        },
        "counts": {
            "primary_eligible": int(
                sum(bool(row["eligible_primary_family"]) for row in rows)
            ),
            "raw_permutation_p_lt_0_05": int(
                sum(
                    bool(row["eligible_primary_family"])
                    and float(row["p_permutation"]) < ALPHA
                    for row in rows
                )
            ),
            "global_fdr_q_lt_0_05": int(
                sum(
                    row["global_q"] is not None and float(row["global_q"]) < ALPHA
                    for row in rows
                )
            ),
            "global_max_t_p_lt_0_05": int(
                sum(
                    row["p_global_max_t"] is not None
                    and float(row["p_global_max_t"]) < ALPHA
                    for row in rows
                )
            ),
        },
        "tasks": tasks,
        "interpretation_boundary": (
            "All network and hierarchy combinations are searched in the same "
            "29-subject cohort. Corrected results support inference; raw-p hits "
            "are exploratory and sparse atoms are excluded from the primary family."
        ),
    }


def plot_overview(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    configure_style()
    families = ("system", "cross", "network", "atom")
    matrix = np.empty((len(TASK_SPECS), len(families)), dtype=float)
    labels: list[list[str]] = []
    for row_index, spec in enumerate(TASK_SPECS):
        family_rows = summary["tasks"][spec["state"]]["strongest_by_family"]
        labels.append([])
        for column_index, family in enumerate(families):
            item = family_rows[family]
            matrix[row_index, column_index] = float(item["rho"])
            labels[-1].append(str(item["short_label"]))

    figure, (heatmap_ax, count_ax) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.7),
        gridspec_kw={"width_ratios": (2.25, 1.2)},
        constrained_layout=True,
    )
    image = heatmap_ax.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-0.7,
        vmax=0.7,
        aspect="auto",
        interpolation="nearest",
    )
    heatmap_ax.set_xticks(
        np.arange(len(families)), [FAMILY_LABELS[family] for family in families]
    )
    heatmap_ax.xaxis.tick_top()
    heatmap_ax.tick_params(axis="x", length=0, pad=6)
    heatmap_ax.set_yticks(
        np.arange(len(TASK_SPECS)), [spec["label"] for spec in TASK_SPECS]
    )
    heatmap_ax.tick_params(axis="y", length=0)
    heatmap_ax.set_xticks(np.arange(-0.5, len(families), 1), minor=True)
    heatmap_ax.set_yticks(np.arange(-0.5, len(TASK_SPECS), 1), minor=True)
    heatmap_ax.grid(which="minor", color="white", linewidth=1.0)
    heatmap_ax.tick_params(which="minor", bottom=False, left=False)
    for spine in heatmap_ax.spines.values():
        spine.set_visible(False)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            item = summary["tasks"][TASK_SPECS[row_index]["state"]][
                "strongest_by_family"
            ][families[column_index]]
            rho = matrix[row_index, column_index]
            marker = (
                "†"
                if item["p_global_max_t"] is not None
                and float(item["p_global_max_t"]) < ALPHA
                else (
                    "*"
                    if item["global_q"] is not None
                    and float(item["global_q"]) < ALPHA
                    else ""
                )
            )
            color = "white" if abs(rho) > 0.42 else "#263238"
            heatmap_ax.text(
                column_index,
                row_index - 0.10,
                f"{rho:+.2f}{marker}",
                ha="center",
                va="center",
                fontsize=6.4,
                color=color,
                weight="bold",
            )
            heatmap_ax.text(
                column_index,
                row_index + 0.18,
                labels[row_index][column_index],
                ha="center",
                va="center",
                fontsize=4.8,
                color=color,
            )
    colorbar = figure.colorbar(
        image, ax=heatmap_ax, orientation="horizontal", fraction=0.07, pad=0.09
    )
    colorbar.set_label(r"Spearman $\rho$ (strongest within family)")
    colorbar.outline.set_linewidth(0.5)

    y = np.arange(len(TASK_SPECS))
    raw_counts = np.asarray(
        [summary["tasks"][spec["state"]]["raw_p_lt_0_05"] for spec in TASK_SPECS]
    )
    fdr_counts = np.asarray(
        [
            summary["tasks"][spec["state"]]["global_fdr_q_lt_0_05"]
            for spec in TASK_SPECS
        ]
    )
    max_t_counts = np.asarray(
        [
            summary["tasks"][spec["state"]]["global_max_t_p_lt_0_05"]
            for spec in TASK_SPECS
        ]
    )
    count_matrix = np.column_stack([raw_counts, fdr_counts, max_t_counts])
    count_ax.imshow(
        count_matrix,
        cmap="YlOrBr",
        vmin=0,
        vmax=max(2, int(count_matrix.max())),
        aspect="auto",
        interpolation="nearest",
    )
    count_ax.set_xticks(
        np.arange(3),
        [
            "Raw perm.\n$p<0.05$",
            "Global FDR\n$q<0.05$",
            "Global max-$T$\n$p<0.05$",
        ],
    )
    count_ax.xaxis.tick_top()
    count_ax.tick_params(axis="x", length=0, pad=6)
    count_ax.set_yticks(y, [spec["label"] for spec in TASK_SPECS])
    count_ax.tick_params(axis="y", length=0)
    count_ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    count_ax.set_yticks(np.arange(-0.5, len(TASK_SPECS), 1), minor=True)
    count_ax.grid(which="minor", color="white", linewidth=1.0)
    count_ax.tick_params(which="minor", bottom=False, left=False)
    for spine in count_ax.spines.values():
        spine.set_visible(False)
    for row_index in range(count_matrix.shape[0]):
        for column_index in range(count_matrix.shape[1]):
            count_ax.text(
                column_index,
                row_index,
                str(int(count_matrix[row_index, column_index])),
                ha="center",
                va="center",
                fontsize=7,
                weight="bold",
                color="#263238",
            )
    heatmap_ax.text(
        -0.13, 1.10, "a", transform=heatmap_ax.transAxes, weight="bold", fontsize=9
    )
    count_ax.text(
        -0.23, 1.10, "b", transform=count_ax.transAxes, weight="bold", fontsize=9
    )
    save_figure(figure, OUTPUT_DIR / "task_score_xi_correlation_overview")


def add_linear_guide(axis: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    if len(x) >= 2 and not np.allclose(x, x[0]) and not np.allclose(y, y[0]):
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(float(x.min()), float(x.max()), 200)
        axis.plot(
            x_line,
            slope * x_line + intercept,
            color="#657487",
            linewidth=0.9,
            linestyle="--",
            zorder=1,
        )


def metric_values_for_row(
    archive: np.lib.npyio.NpzFile, row: dict[str, Any]
) -> np.ndarray:
    state_index = archive["states"].astype(str).tolist().index(str(row["state"]))
    _, values = build_metrics(archive, state_index)
    return np.asarray(values[:, int(row["metric_index"])], dtype=float)


def plot_top_scatter(
    rows: list[dict[str, Any]], scores: dict[str, np.ndarray], summary: dict[str, Any]
) -> None:
    configure_style()
    archive = np.load(ARRAYS_PATH)
    figure, axes = plt.subplots(2, 3, figsize=(8.8, 5.6), constrained_layout=True)
    for axis, spec in zip(axes.ravel()[:5], TASK_SPECS, strict=True):
        item = summary["tasks"][spec["state"]]["strongest_overall"]
        x = scores[spec["state"]]
        y = metric_values_for_row(archive, item)
        positive = np.abs(y) > ZERO_TOLERANCE
        axis.scatter(
            x[~positive],
            y[~positive],
            s=23,
            color="#C7CDD4",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        axis.scatter(
            x[positive],
            y[positive],
            s=27,
            color="#C98F76",
            alpha=0.88,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        add_linear_guide(axis, x, y)
        global_q = float(item["global_q"])
        global_max = float(item["p_global_max_t"])
        axis.text(
            0.03,
            0.97,
            rf"$\rho$={float(item['rho']):+.3f}"
            + "\n"
            + rf"perm. $p$={float(item['p_permutation']):.3g}"
            + "\n"
            + rf"global $q$={global_q:.3g}; max-$T$ $p$={global_max:.3g}"
            + "\n"
            + rf"positive support={int(item['nonzero_subjects'])}/29"
            + "\n"
            + rf"LOO $\rho$=[{float(item['leave_one_out']['rho_min']):+.2f}, "
            + rf"{float(item['leave_one_out']['rho_max']):+.2f}]",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=5.7,
            color="#3D4852",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.87,
                "pad": 1.2,
            },
        )
        axis.set_title(f"{spec['label']} · {item['short_label']}")
        axis.set_xlabel(spec["score_label"])
        axis.set_ylabel("Integrated effective information (bits)")
    summary_axis = axes.ravel()[5]
    summary_axis.axis("off")
    counts = summary["counts"]
    summary_axis.text(
        0.04,
        0.96,
        "Multiplicity summary",
        transform=summary_axis.transAxes,
        va="top",
        fontsize=8,
        weight="bold",
    )
    summary_axis.text(
        0.04,
        0.83,
        f"Tested: {summary['config']['total_metrics']}\n"
        f"Primary eligible: {counts['primary_eligible']}\n"
        f"Raw permutation p<0.05: {counts['raw_permutation_p_lt_0_05']}\n"
        f"Global FDR q<0.05: {counts['global_fdr_q_lt_0_05']}\n"
        f"Global max-T p<0.05: {counts['global_max_t_p_lt_0_05']}\n\n"
        "Top panels show the strongest eligible\n"
        "association per matched task. Selection and\n"
        "testing use the same cohort; corrected values\n"
        "govern inference.",
        transform=summary_axis.transAxes,
        va="top",
        linespacing=1.45,
        fontsize=6.4,
        color="#3D4852",
    )
    for label, axis in zip("abcdef", axes.ravel(), strict=True):
        axis.text(
            -0.17,
            1.08,
            label,
            transform=axis.transAxes,
            weight="bold",
            fontsize=9,
        )
    save_figure(figure, OUTPUT_DIR / "task_score_xi_top_scatter")


def save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "all_associations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    contract = f"""# HCP task-score–Xi correlation experiment contract

## Scientific question

Within the same task and the same 29 subjects, does task performance covary with
system-level or regional/network integrated effective information?

## Frozen design

- Subjects: the same 29 REST-task common subjects.
- State-score pairs: EMOTION, LANGUAGE, RELATIONAL, SOCIAL, and WM only.
- Brain cache: existing k=1, history p=3, alpha=1 affine/Gaussian Xi analysis.
- Metrics per task: system Xi, cross-network Xi, 7 absolute network
  contributions, and 120 absolute hierarchy-atom contributions.
- Statistic: two-sided Spearman correlation.
- Randomization: {PERMUTATIONS:,} shared subject-label permutations, seed {SEED}.
- Multiplicity: task-wise and global permutation max-T; task-wise and global
  Benjamini-Hochberg FDR.
- Primary support rule: hierarchy atoms require at least
  {MIN_ATOM_SUPPORT}/29 positive subjects.
- Social score: mean of Random-condition correct responses and TOM-condition
  correct responses.

## Interpretation boundary

The network combinations are searched and tested in the same cohort. Global
FDR/max-T results support inference; raw-p hits and best-within-family rankings
are exploratory. The existing estimator is retained to avoid changing the brain
metric while changing the behavioral score.
"""
    (OUTPUT_DIR / "experiment_contract.md").write_text(contract, encoding="utf-8")

    lines = [
        "# Task-matched HCP performance–Xi correlations",
        "",
        f"- Tested associations: {summary['config']['total_metrics']}.",
        f"- Primary eligible associations: {summary['counts']['primary_eligible']}.",
        f"- Raw permutation p<0.05: {summary['counts']['raw_permutation_p_lt_0_05']}.",
        f"- Global FDR q<0.05: {summary['counts']['global_fdr_q_lt_0_05']}.",
        f"- Global max-T p<0.05: {summary['counts']['global_max_t_p_lt_0_05']}.",
        "",
        "## Strongest eligible association per task",
        "",
        "| Task | Brain metric | rho | Perm. p | Global q | Global max-T p | LOO rho range |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for spec in TASK_SPECS:
        item = summary["tasks"][spec["state"]]["strongest_overall"]
        loo = item["leave_one_out"]
        lines.append(
            f"| {spec['label']} | {item['short_label']} | {item['rho']:+.3f} | "
            f"{item['p_permutation']:.3g} | {item['global_q']:.3g} | "
            f"{item['p_global_max_t']:.3g} | "
            f"[{loo['rho_min']:+.3f}, {loo['rho_max']:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            summary["interpretation_boundary"],
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows, scores, _ = analyze()
    summary = build_summary(rows)
    write_outputs(rows, summary)
    plot_overview(rows, summary)
    plot_top_scatter(rows, scores, summary)
    print(json.dumps(summary["counts"], indent=2))
    for spec in TASK_SPECS:
        item = summary["tasks"][spec["state"]]["strongest_overall"]
        print(
            spec["state"],
            item["short_label"],
            f"rho={item['rho']:+.3f}",
            f"p={item['p_permutation']:.3g}",
            f"q={item['global_q']:.3g}",
            f"maxT={item['p_global_max_t']:.3g}",
        )


if __name__ == "__main__":
    main()
