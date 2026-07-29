#!/usr/bin/env python3
"""Cross-atlas validation of five task-score synergy candidates.

Candidate coalitions are selected once from the Schaefer-500 exhaustive
analysis, then frozen and recomputed with Schaefer-1000 task projections.
The validation family therefore contains exactly five prespecified tests.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_schaefer500_yeo7_network_attribution import (
    NETWORK_ORDER,
    discover_inputs,
)
from scripts.analyze_hcp_task_evoked_pc2_xi_hierarchy import network_module_indices
from scripts.phi_hierarchy import subset_phi_raw
from scripts.run_hcp_schaefer500_yeo7_module_phi_decomposition import module_ei_table
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import (
    load_yeo7_groups,
)
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi
from scripts.tune_hcp_task_evoked_xi_hierarchy import prepare_projection


SOURCE_DIR = ROOT / "results" / "hcp_task_score_fixed_synergy"
SOURCE_SUMMARY = SOURCE_DIR / "summary.json"
SOURCE_METRICS = ROOT / "results" / "hcp_cognition_exhaustive_targeted_greedy" / "metrics.npz"
ARRAYS_1000 = (
    ROOT
    / "results"
    / "hcp_schaefer1000_task_evoked_xi_replication"
    / "full"
    / "k1_p3_a1"
    / "arrays.npz"
)
REST_ROOT = ROOT / "Data" / "hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30"
TASK_ROOT = ROOT / "Data" / "hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_30"
LABELS_1000 = REST_ROOT / "_atlas_labels" / "Schaefer2018_1000Parcels_7Networks_order.txt"
BEHAVIOR_PATH = ROOT / "Data" / "unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
OUTPUT_DIR = ROOT / "results" / "hcp_task_score_synergy_schaefer1000_validation"
CACHE_PATH = OUTPUT_DIR / "fixed_candidates_schaefer1000.npz"

TASKS = (
    ("EMOTION", "Emotion", "Emotion_Task_Acc", "Emotion accuracy (%)"),
    ("LANGUAGE", "Language", "Language_Task_Acc", "Language accuracy (%)"),
    ("RELATIONAL", "Relational", "Relational_Task_Acc", "Relational accuracy (%)"),
    ("SOCIAL", "Social", "derived_social_balanced_accuracy", "Social balanced accuracy (%)"),
    ("WM", "Working memory", "WM_Task_Acc", "Working memory accuracy (%)"),
)
SHORT = {
    "Vis": "Vis", "SomMot": "Som", "DorsAttn": "DAN",
    "SalVentAttn": "SVAN", "Limbic": "Lim", "Cont": "Cont",
    "Default": "Def",
}
PERMUTATIONS = 50_000
SEED = 20260729


def short_name(value: str) -> str:
    return "+".join(SHORT[item] for item in value.split("+"))


def holm_adjust(values: list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate(
        np.minimum(1.0, ranked * (len(p) - np.arange(len(p))))
    )
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, method="average")
    centered = ranks - ranks.mean()
    return centered / np.sqrt(np.sum(centered**2))


def load_candidates() -> list[dict[str, Any]]:
    summary = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    candidates = []
    for state, label, field, score_label in TASKS:
        source = summary["tasks"][state]["task_absolute"]["strongest"]
        candidates.append({
            "state": state,
            "label": label,
            "field": field,
            "score_label": score_label,
            "coalition": source["coalition"],
            "rho_500": float(source["rho"]),
            "p_500": float(source["p_permutation"]),
        })
    return candidates


def load_scores(subjects: list[str]) -> dict[str, np.ndarray]:
    with BEHAVIOR_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = {str(row["Subject"]): row for row in csv.DictReader(handle)}
    output = {}
    for state, _, field, _ in TASKS:
        if state == "SOCIAL":
            values = np.asarray([
                0.5 * (
                    float(rows[subject]["Social_Task_Random_Perc_Random"])
                    + float(rows[subject]["Social_Task_TOM_Perc_TOM"])
                )
                for subject in subjects
            ])
        else:
            values = np.asarray([float(rows[subject][field]) for subject in subjects])
        output[state] = values
    return output


def compute_candidates(
    candidates: list[dict[str, Any]], subjects: list[str]
) -> np.ndarray:
    expected_states = np.asarray([item["state"] for item in candidates])
    expected_coalitions = np.asarray([item["coalition"] for item in candidates])
    if CACHE_PATH.is_file():
        cache = np.load(CACHE_PATH)
        if (
            np.array_equal(cache["subjects"].astype(str), np.asarray(subjects))
            and np.array_equal(cache["states"].astype(str), expected_states)
            and np.array_equal(cache["coalitions"].astype(str), expected_coalitions)
        ):
            values = np.asarray(cache["values"], dtype=float)
            if values.shape == (5, 29) and np.isfinite(values).all():
                return values

    discovered = discover_inputs(REST_ROOT, TASK_ROOT)
    groups = load_yeo7_groups(LABELS_1000, expected_parcels=1000)
    indices = network_module_indices(NETWORK_ORDER, n_components=1, order=3)
    values = np.zeros((5, 29), dtype=float)
    jobs = [
        (candidate_index, subject_index)
        for candidate_index in range(5)
        for subject_index in range(29)
    ]
    for candidate_index, subject_index in tqdm(
        jobs, desc="Schaefer-1000 fixed candidates", unit="model"
    ):
        candidate = candidates[candidate_index]
        subject = subjects[subject_index]
        projections, _, development_end = prepare_projection(
            Path(discovered[subject][candidate["state"]]),
            groups,
            state=candidate["state"],
            max_components=2,
            task_retained_key="Schaefer1000_taskRetained",
            task_regressed_key="Schaefer1000_taskRegressed",
            expected_parcels=1000,
        )
        fitted = fit_delta_history_phi(
            projections[1], alpha=1.0, order=3, development_end=development_end
        )
        table = module_ei_table(
            fitted["transition"], fitted["noise_covariance"], indices, ridge=1.0e-6
        )
        singleton = {name: float(table[(name,)]) for name in NETWORK_ORDER}
        coalition = tuple(candidate["coalition"].split("+"))
        values[candidate_index, subject_index] = subset_phi_raw(
            coalition, table, singleton
        )
    np.savez_compressed(
        CACHE_PATH,
        values=values,
        subjects=np.asarray(subjects),
        states=expected_states,
        coalitions=expected_coalitions,
    )
    return values


def permutation_test(
    scores: dict[str, np.ndarray],
    candidates: list[dict[str, Any]],
    values: np.ndarray,
    subjects: list[str],
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    permutation_indices = np.vstack([rng.permutation(29) for _ in range(PERMUTATIONS)])
    source = np.load(SOURCE_METRICS)
    source_subjects = source["subjects"].astype(str).tolist()
    if source_subjects != subjects:
        raise ValueError("Schaefer-500/1000 subject order mismatch")
    source_states = source["states"].astype(str).tolist()
    source_coalitions = source["coalitions"].astype(str).tolist()
    results = []
    for index, candidate in enumerate(candidates):
        x = scores[candidate["state"]]
        y = values[index]
        x_rank = normalized_ranks(x)
        y_rank = normalized_ranks(y)
        rho = float(x_rank @ y_rank)
        null = x_rank[permutation_indices] @ y_rank
        p = float(
            (1 + np.count_nonzero(np.abs(null) >= abs(rho))) / (PERMUTATIONS + 1)
        )
        loo = np.asarray([
            spearmanr(np.delete(x, item), np.delete(y, item)).statistic
            for item in range(29)
        ])
        source_values = source["fixed_block_synergy"][
            source_states.index(candidate["state"]),
            :,
            source_coalitions.index(candidate["coalition"]),
        ]
        atlas_rho = float(spearmanr(source_values, y).statistic)
        results.append({
            **candidate,
            "n_subjects": 29,
            "rho_1000": rho,
            "p_1000_permutation": p,
            "p_1000_asymptotic": float(spearmanr(x, y).pvalue),
            "same_effect_direction_as_500": bool(
                np.sign(rho) == np.sign(candidate["rho_500"])
            ),
            "cross_atlas_metric_rho": atlas_rho,
            "leave_one_out_minimum": float(loo.min()),
            "leave_one_out_maximum": float(loo.max()),
            "leave_one_out_same_direction_fraction": float(
                np.mean(np.sign(loo) == np.sign(rho))
            ),
            "holm_p_across_five": None,
        })
    holm = holm_adjust([item["p_1000_permutation"] for item in results])
    for item, adjusted in zip(results, holm, strict=True):
        item["holm_p_across_five"] = float(adjusted)
    return results


def configure_style() -> None:
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


def plot_results(results: list[dict[str, Any]]) -> None:
    configure_style()
    figure, axis = plt.subplots(figsize=(6.8, 3.45), constrained_layout=True)
    y = np.arange(5)
    rho_500 = np.asarray([item["rho_500"] for item in results])
    rho_1000 = np.asarray([item["rho_1000"] for item in results])
    axis.axvline(0, color="#AAB2B9", lw=0.8)
    axis.scatter(rho_500, y - 0.12, s=34, marker="o", color="#4C78A8",
                 label="Schaefer-500 discovery", zorder=3)
    axis.scatter(rho_1000, y + 0.12, s=38, marker="D", color="#E07A5F",
                 label="Schaefer-1000 fixed validation", zorder=3)
    for row, item in enumerate(results):
        axis.plot([rho_500[row], rho_1000[row]], [row - 0.12, row + 0.12],
                  color="#C8CDD2", lw=0.8, zorder=1)
        axis.text(
            1.02, row,
            f"p={item['p_1000_permutation']:.3f}; Holm={item['holm_p_across_five']:.3f}",
            transform=axis.get_yaxis_transform(), ha="left", va="center",
            fontsize=6.2, color="#49545E",
        )
    axis.set_yticks(y, [
        f"{item['label']} · {short_name(item['coalition'])}" for item in results
    ])
    axis.invert_yaxis()
    axis.set_xlim(-0.65, 0.65)
    axis.set_xlabel(r"Spearman $\rho$ with matching task performance")
    axis.set_title("Cross-atlas validation of five frozen coalition hypotheses",
                   loc="left", weight="bold")
    axis.legend(loc="center left", bbox_to_anchor=(1.48, 0.5))
    for extension in ("png", "svg", "pdf"):
        figure.savefig(
            OUTPUT_DIR / f"cross_atlas_validation.{extension}",
            dpi=300, bbox_inches="tight", facecolor="white",
        )
    plt.close(figure)


def write_report(results: list[dict[str, Any]]) -> None:
    lines = [
        "# Schaefer-1000 cross-atlas validation",
        "",
        "Five coalitions were frozen from the Schaefer-500 task-matched search and "
        "recomputed without coalition search at Schaefer-1000 resolution. All tests "
        "use the same 29 subjects; Holm correction covers the five validation tests.",
        "",
        "| Task | Frozen coalition | rho (500) | rho (1000) | perm. p (1000) | "
        "Holm p | same direction | metric agreement |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['label']} | {short_name(item['coalition'])} | "
            f"{item['rho_500']:+.3f} | {item['rho_1000']:+.3f} | "
            f"{item['p_1000_permutation']:.5f} | {item['holm_p_across_five']:.4f} | "
            f"{'yes' if item['same_effect_direction_as_500'] else 'no'} | "
            f"{item['cross_atlas_metric_rho']:+.3f} |"
        )
    lines += [
        "",
        "A Schaefer-500 association is considered cross-atlas replicated only when "
        "the Schaefer-1000 effect has the same direction and Holm-adjusted p < 0.05.",
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates()
    arrays = np.load(ARRAYS_1000)
    subjects = [str(value) for value in arrays["subjects"].astype(str)]
    if len(subjects) != 29 or len(set(subjects)) != 29:
        raise ValueError("Expected 29 unique Schaefer-1000 subjects")
    values = compute_candidates(candidates, subjects)
    if values.shape != (5, 29) or not np.isfinite(values).all():
        raise ValueError("Incomplete Schaefer-1000 candidate matrix")
    scores = load_scores([subject.removeprefix("sub-") for subject in subjects])
    results = permutation_test(scores, candidates, values, subjects)
    payload = {
        "experiment": "Frozen Schaefer-500 candidates validated at Schaefer-1000",
        "subjects": 29,
        "tests": 5,
        "permutations": PERMUTATIONS,
        "holm_family": 5,
        "replication_rule": "same direction and Holm p < 0.05",
        "results": results,
        "replicated_tasks": [
            item["state"] for item in results
            if item["same_effect_direction_as_500"]
            and item["holm_p_across_five"] < 0.05
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(results)
    plot_results(results)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
