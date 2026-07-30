#!/usr/bin/env python3
"""Select behavior-diverse HCP supplement candidates without using brain data."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR = ROOT / "Data/unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
CURRENT_METRICS = (
    ROOT / "results/hcp_cognition_exhaustive_targeted_greedy/metrics.npz"
)
REST_ROOT = (
    ROOT
    / "Data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30"
)
TASK_ROOT = (
    ROOT / "Data/hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_30"
)
OUTPUT = ROOT / "results/hcp_behavior_diversity_supplement_candidates"

SCORE_FIELDS = (
    ("emotion_accuracy", "Emotion_Task_Acc", "Emotion accuracy"),
    ("language_accuracy", "Language_Task_Acc", "Language accuracy"),
    ("relational_accuracy", "Relational_Task_Acc", "Relational accuracy"),
    ("social_tom_accuracy", "Social_Task_TOM_Perc_TOM", "Social TOM accuracy"),
    (
        "social_random_accuracy",
        "Social_Task_Random_Perc_Random",
        "Social Random accuracy",
    ),
    ("wm_accuracy", "WM_Task_Acc", "Working-memory accuracy"),
)
DERIVED_SCORE = (
    "social_balanced_accuracy",
    "Social balanced accuracy",
)
SCAN_COMPLETENESS_FIELDS = (
    "fMRI_WM_PctCompl",
    "fMRI_Gamb_PctCompl",
    "fMRI_Mot_PctCompl",
    "fMRI_Lang_PctCompl",
    "fMRI_Soc_PctCompl",
    "fMRI_Rel_PctCompl",
    "fMRI_Emo_PctCompl",
)

# Social conditions receive modest extra weight because their current ceiling
# is the main design failure; the combined Social score is derived and is not
# independently weighted.
SCORE_WEIGHTS = np.asarray([1.0, 1.0, 1.0, 2.0, 1.5, 1.0])
RARITY_WEIGHT = 0.78
PROFILE_DISTANCE_WEIGHT = 0.22

# Nested age quotas for each block of ten new participants. They approximately
# match the eligible candidate pool while avoiding age-driven score enrichment.
AGE_QUOTAS = (
    {"22-25": 2, "26-30": 4, "31-35": 4, "36+": 0},
    {"22-25": 2, "26-30": 5, "31-35": 3, "36+": 0},
    {"22-25": 3, "26-30": 4, "31-35": 3, "36+": 0},
)


def is_number(value: str) -> bool:
    try:
        return bool(value.strip()) and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def scan_eligible(row: dict[str, str]) -> bool:
    return bool(
        row["3T_Full_MR_Compl"] == "true"
        and row["3T_Full_Task_fMRI"] == "true"
        and row["3T_RS-fMRI_Count"] == "4"
        and row["3T_RS-fMRI_PctCompl"] == "100.0"
        and all(row[field] == "100.0" for field in SCAN_COMPLETENESS_FIELDS)
        and row["QC_Issue"] == ""
    )


def score_complete(row: dict[str, str]) -> bool:
    return all(is_number(row[field]) for _, field, _ in SCORE_FIELDS)


def passes_behavior_validity_screen(row: dict[str, str]) -> bool:
    """Exclude only gross Social nonperformance, not ordinary low scorers."""
    tom = float(row["Social_Task_TOM_Perc_TOM"])
    random = float(row["Social_Task_Random_Perc_Random"])
    return 0.5 * (tom + random) >= 40.0


def minmax(values: np.ndarray) -> np.ndarray:
    lower = float(np.min(values))
    upper = float(np.max(values))
    if upper - lower <= 1.0e-12:
        return np.zeros_like(values)
    return (values - lower) / (upper - lower)


def build_categories(percentiles: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, list[int]]:
    categories = np.zeros(values.shape, dtype=int)
    category_counts: list[int] = []
    for column in range(values.shape[1]):
        if column in (3, 4):
            levels = sorted(set(values[:, column].tolist()))
            lookup = {level: index for index, level in enumerate(levels)}
            categories[:, column] = [
                lookup[value] for value in values[:, column]
            ]
            category_counts.append(len(levels))
        else:
            categories[:, column] = np.minimum(
                (percentiles[:, column] * 10).astype(int), 9
            )
            category_counts.append(10)
    return categories, category_counts


def select_candidates(
    current_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
) -> tuple[list[int], list[float], np.ndarray]:
    all_rows = current_rows + candidate_rows
    values = np.asarray([
        [float(row[field]) for _, field, _ in SCORE_FIELDS]
        for row in all_rows
    ])
    percentiles = np.column_stack([
        rankdata(values[:, column], method="average") / (len(values) + 1)
        for column in range(values.shape[1])
    ])
    categories, category_counts = build_categories(percentiles, values)
    selected = list(range(len(current_rows)))
    remaining = list(range(len(current_rows), len(all_rows)))
    selected_new: list[int] = []
    selection_scores: list[float] = []
    sex_counts = Counter(row["Gender"] for row in current_rows)

    for step in range(30):
        block = step // 10
        block_start = len(current_rows) + 10 * block
        block_age_counts = Counter(
            all_rows[index]["Age"] for index in selected[block_start:]
        )
        allowed_ages = {
            age for age, quota in AGE_QUOTAS[block].items()
            if block_age_counts[age] < quota
        }
        imbalance = {
            gender: abs(
                sex_counts["F"] + int(gender == "F")
                - sex_counts["M"] - int(gender == "M")
            )
            for gender in ("F", "M")
        }
        minimum_imbalance = min(imbalance.values())
        allowed_genders = {
            gender for gender, value in imbalance.items()
            if value == minimum_imbalance
        }
        eligible = [
            index for index in remaining
            if all_rows[index]["Age"] in allowed_ages
            and all_rows[index]["Gender"] in allowed_genders
        ]
        if not eligible:
            raise RuntimeError(
                f"No eligible candidate at step {step + 1}; "
                f"ages={allowed_ages}, genders={allowed_genders}"
            )

        rarity = np.zeros(len(eligible), dtype=float)
        for column in range(values.shape[1]):
            counts = np.bincount(
                categories[selected, column],
                minlength=category_counts[column],
            )
            candidate_counts = counts[categories[eligible, column]]
            gain = np.sqrt(candidate_counts + 1) - np.sqrt(candidate_counts)
            rarity += SCORE_WEIGHTS[column] * minmax(gain)
        rarity /= float(SCORE_WEIGHTS.sum())

        weighted_percentiles = percentiles * np.sqrt(SCORE_WEIGHTS)
        candidate_profiles = weighted_percentiles[eligible]
        selected_profiles = weighted_percentiles[selected]
        distances = np.sqrt(
            np.sum(
                (
                    candidate_profiles[:, None, :]
                    - selected_profiles[None, :, :]
                )
                ** 2,
                axis=2,
            )
        )
        nearest_profile_distance = minmax(np.min(distances, axis=1))
        objective = (
            RARITY_WEIGHT * rarity
            + PROFILE_DISTANCE_WEIGHT * nearest_profile_distance
        )
        best_position = int(np.argmax(objective))
        best_index = eligible[best_position]
        selected.append(best_index)
        selected_new.append(best_index)
        remaining.remove(best_index)
        selection_scores.append(float(objective[best_position]))
        sex_counts[all_rows[best_index]["Gender"]] += 1

    return selected_new, selection_scores, values


def mode_summary(values: np.ndarray) -> tuple[float, int]:
    counts = Counter(values.tolist())
    maximum = max(counts.values())
    modes = sorted(value for value, count in counts.items() if count == maximum)
    return float(modes[0]), int(maximum)


def distribution_summary(
    values: np.ndarray,
    cohort_indices: list[int],
    metric_key: str,
    metric_label: str,
) -> dict[str, Any]:
    selected = np.asarray(values[cohort_indices], dtype=float)
    mode, mode_count = mode_summary(selected)
    return {
        "metric": metric_key,
        "metric_label": metric_label,
        "n": int(len(selected)),
        "minimum": float(np.min(selected)),
        "p25": float(np.quantile(selected, 0.25)),
        "median": float(np.median(selected)),
        "p75": float(np.quantile(selected, 0.75)),
        "maximum": float(np.max(selected)),
        "iqr": float(np.quantile(selected, 0.75) - np.quantile(selected, 0.25)),
        "sd": float(np.std(selected, ddof=1)),
        "unique_values": int(len(np.unique(selected))),
        "mode": mode,
        "mode_count": mode_count,
        "mode_share": float(mode_count / len(selected)),
        "ceiling_100_count": int(np.count_nonzero(selected == 100.0)),
        "ceiling_100_share": float(np.mean(selected == 100.0)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with BEHAVIOR.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_subject = {str(row["Subject"]): row for row in rows}
    archive = np.load(CURRENT_METRICS)
    current_subjects = [
        str(value).removeprefix("sub-")
        for value in archive["subjects"].astype(str)
    ]
    if len(current_subjects) != 29 or len(set(current_subjects)) != 29:
        raise ValueError("Expected 29 unique current subjects")
    current_rows = [by_subject[subject] for subject in current_subjects]
    if not all(score_complete(row) for row in current_rows):
        raise ValueError("Current subjects do not have complete selection scores")

    current_set = set(current_subjects)
    candidate_rows = sorted(
        [
            row for row in rows
            if str(row["Subject"]) not in current_set
            and score_complete(row)
            and passes_behavior_validity_screen(row)
            and scan_eligible(row)
        ],
        key=lambda row: int(row["Subject"]),
    )
    if len(candidate_rows) != 734:
        raise ValueError(f"Unexpected eligible candidate count: {len(candidate_rows)}")

    selected_indices, selection_scores, all_values = select_candidates(
        current_rows, candidate_rows
    )
    all_rows = current_rows + candidate_rows
    selected_subjects = [str(all_rows[index]["Subject"]) for index in selected_indices]
    if len(selected_subjects) != 30 or len(set(selected_subjects)) != 30:
        raise ValueError("Selection did not return 30 unique new subjects")
    if current_set.intersection(selected_subjects):
        raise ValueError("A current subject was selected as a supplement")

    local_rest = {
        path.name.removeprefix("sub-")
        for path in REST_ROOT.glob("sub-*") if path.is_dir()
    }
    local_task = {
        path.name.removeprefix("sub-")
        for path in TASK_ROOT.glob("sub-*") if path.is_dir()
    }
    candidate_records = []
    for rank, (index, objective) in enumerate(
        zip(selected_indices, selection_scores, strict=True), start=1
    ):
        row = all_rows[index]
        subject = str(row["Subject"])
        tom = float(row["Social_Task_TOM_Perc_TOM"])
        random = float(row["Social_Task_Random_Perc_Random"])
        candidate_records.append({
            "priority_rank": rank,
            "supplement_tier": (
                "first_10" if rank <= 10 else (
                    "next_10" if rank <= 20 else "final_10"
                )
            ),
            "subject": subject,
            "gender": row["Gender"],
            "age_bin": row["Age"],
            "selection_objective": objective,
            "emotion_accuracy": float(row["Emotion_Task_Acc"]),
            "language_accuracy": float(row["Language_Task_Acc"]),
            "relational_accuracy": float(row["Relational_Task_Acc"]),
            "social_balanced_accuracy": 0.5 * (tom + random),
            "social_tom_accuracy": tom,
            "social_random_accuracy": random,
            "wm_accuracy": float(row["WM_Task_Acc"]),
            "hcp_scan_metadata_eligible": True,
            "qc_issue_blank": True,
            "local_rest_roi_available": subject in local_rest,
            "local_all_task_roi_available": subject in local_task,
            "requires_download_and_roi_extraction": not (
                subject in local_rest and subject in local_task
            ),
        })
    if any(
        not record["requires_download_and_roi_extraction"]
        for record in candidate_records
    ):
        raise ValueError("Unexpected candidate already has complete local ROI data")

    current_records = []
    for row in current_rows:
        subject = str(row["Subject"])
        tom = float(row["Social_Task_TOM_Perc_TOM"])
        random = float(row["Social_Task_Random_Perc_Random"])
        current_records.append({
            "subject": subject,
            "gender": row["Gender"],
            "age_bin": row["Age"],
            "emotion_accuracy": float(row["Emotion_Task_Acc"]),
            "language_accuracy": float(row["Language_Task_Acc"]),
            "relational_accuracy": float(row["Relational_Task_Acc"]),
            "social_balanced_accuracy": 0.5 * (tom + random),
            "social_tom_accuracy": tom,
            "social_random_accuracy": random,
            "wm_accuracy": float(row["WM_Task_Acc"]),
            "local_rest_roi_available": subject in local_rest,
            "local_all_task_roi_available": subject in local_task,
        })

    metric_arrays = {
        key: all_values[:, column]
        for column, (key, _, _) in enumerate(SCORE_FIELDS)
    }
    metric_arrays[DERIVED_SCORE[0]] = 0.5 * (
        metric_arrays["social_tom_accuracy"]
        + metric_arrays["social_random_accuracy"]
    )
    metric_labels = {
        key: label for key, _, label in SCORE_FIELDS
    }
    metric_labels[DERIVED_SCORE[0]] = DERIVED_SCORE[1]

    cohorts = {
        "current_29": list(range(29)),
        "current_plus_10": list(range(29)) + selected_indices[:10],
        "current_plus_20": list(range(29)) + selected_indices[:20],
        "current_plus_30": list(range(29)) + selected_indices[:30],
    }
    distributions = []
    for cohort, indices in cohorts.items():
        for metric, values in metric_arrays.items():
            record = distribution_summary(
                values, indices, metric, metric_labels[metric]
            )
            record["cohort"] = cohort
            distributions.append(record)

    sex_age = {}
    for cohort, indices in cohorts.items():
        cohort_rows = [all_rows[index] for index in indices]
        sex_age[cohort] = {
            "gender": dict(sorted(Counter(
                row["Gender"] for row in cohort_rows
            ).items())),
            "age_bin": dict(sorted(Counter(
                row["Age"] for row in cohort_rows
            ).items())),
        }

    payload = {
        "experiment": "Behavior-diversity supplement candidate selection",
        "source_rows": len(rows),
        "current_subjects": current_subjects,
        "current_subject_records": current_records,
        "eligible_candidate_pool": len(candidate_rows),
        "selection_uses_brain_metrics": False,
        "candidate_filter": {
            "complete_scores": [field for _, field, _ in SCORE_FIELDS],
            "minimum_social_balanced_accuracy": 40.0,
            "3T_Full_MR_Compl": "true",
            "3T_Full_Task_fMRI": "true",
            "3T_RS-fMRI_Count": "4",
            "3T_RS-fMRI_PctCompl": "100.0",
            "all_seven_task_fMRI_PctCompl": "100.0",
            "QC_Issue": "blank",
        },
        "selection_method": {
            "description": (
                "Greedy nested behavioral spectrum enrichment using underrepresented "
                "score strata and nearest-profile distance in empirical-percentile "
                "space; no neural value enters selection."
            ),
            "score_weights": {
                SCORE_FIELDS[index][0]: float(SCORE_WEIGHTS[index])
                for index in range(len(SCORE_FIELDS))
            },
            "rarity_weight": RARITY_WEIGHT,
            "profile_distance_weight": PROFILE_DISTANCE_WEIGHT,
            "age_quotas_per_block_of_10": AGE_QUOTAS,
            "gender_rule": (
                "At each step choose the gender producing the smallest combined "
                "current-plus-new imbalance."
            ),
        },
        "candidates": candidate_records,
        "distribution_comparison": distributions,
        "demographic_comparison": sex_age,
        "limitations": [
            (
                "The unrestricted table does not contain HCP family identifiers; "
                "family overlap must be checked with restricted data before download."
            ),
            (
                "Scan-completion metadata and blank QC_Issue do not guarantee that "
                "the exact ROI time-series products required by this repository are "
                "already available; all selected subjects require download and extraction."
            ),
            (
                "A minimal Social balanced-accuracy floor of 40% excludes gross "
                "nonperformance while retaining condition-specific low scorers; "
                "event-level behavioral QC remains necessary."
            ),
            (
                "Outcome-enriched sampling improves behavioral spread but changes the "
                "target population. Confirmatory brain-behavior inference should model "
                "the sampling design, use weights, or reserve a separate validation set."
            ),
        ],
    }
    (OUTPUT / "selection_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lookup = {
        (record["cohort"], record["metric"]): record
        for record in distributions
    }
    lines = [
        "# HCP behavior-diversity supplement candidates",
        "",
        f"- Eligible behavior-plus-scan-metadata pool: {len(candidate_rows)}.",
        "- Selection used behavioral scores only; no brain metric was inspected.",
        "- Recommended first increment: 10 subjects; the 20- and 30-subject lists are nested.",
        "- Every selected subject still requires imaging download and ROI extraction.",
        "",
        "## Priority list",
        "",
        "| Rank | Subject | Tier | Sex | Age | Emotion | Language | Relational | Social balanced | TOM | Random | WM |",
        "|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in candidate_records:
        lines.append(
            f"| {record['priority_rank']} | {record['subject']} | "
            f"{record['supplement_tier']} | {record['gender']} | "
            f"{record['age_bin']} | {record['emotion_accuracy']:.3f} | "
            f"{record['language_accuracy']:.3f} | "
            f"{record['relational_accuracy']:.3f} | "
            f"{record['social_balanced_accuracy']:.3f} | "
            f"{record['social_tom_accuracy']:.3f} | "
            f"{record['social_random_accuracy']:.3f} | "
            f"{record['wm_accuracy']:.3f} |"
        )
    lines += [
        "",
        "## Key distribution changes",
        "",
        "| Metric | Current 29 unique / mode share | +10 | +20 | +30 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in (
        "emotion_accuracy",
        "language_accuracy",
        "relational_accuracy",
        "social_balanced_accuracy",
        "social_tom_accuracy",
        "social_random_accuracy",
        "wm_accuracy",
    ):
        cells = []
        for cohort in cohorts:
            record = lookup[(cohort, metric)]
            cells.append(
                f"{record['unique_values']} / {100 * record['mode_share']:.1f}%"
            )
        lines.append(
            f"| {metric_labels[metric]} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        *[f"- {text}" for text in payload["limitations"]],
    ]
    (OUTPUT / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "eligible_pool": len(candidate_rows),
        "first_10": selected_subjects[:10],
        "next_10": selected_subjects[10:20],
        "final_10": selected_subjects[20:30],
        "tom_current": lookup[("current_29", "social_tom_accuracy")],
        "tom_plus_10": lookup[("current_plus_10", "social_tom_accuracy")],
        "tom_plus_30": lookup[("current_plus_30", "social_tom_accuracy")],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
