#!/usr/bin/env python3
"""Render Schaefer-1000 results with the exact HCP500 Figure 2 layout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_hcp_task_evoked_xi_tuning import (
    configure_style,
    plot_main_combined,
)


RESULTS_ROOT = ROOT / "results/hcp_schaefer1000_task_evoked_xi_replication"
FINAL_ROOT = RESULTS_ROOT / "final"
COGNITION = ROOT / "results/hcp_single_group_sem_full_1206/selected_29_sem_results.csv"
COMPATIBILITY_ROOT = FINAL_ROOT / "figure2_compatibility_input"


def prepare_compatibility_input() -> None:
    comparison = json.loads(
        (FINAL_ROOT / "comparison_summary.json").read_text(encoding="utf-8")
    )
    targeted = np.load(FINAL_ROOT / "confirmatory_targeted_metrics.npz")
    main = np.load(RESULTS_ROOT / "full/k1_p3_a1/arrays.npz")
    states = main["states"].astype(str)
    subjects = main["subjects"].astype(str)
    coalitions = targeted["coalitions"].astype(str)
    candidate_values = np.asarray(targeted["values"], dtype=float)
    metrics = np.full((len(states), len(subjects), len(coalitions)), np.nan)

    score_names = ("cry_score", "mem_score", "spd_score")
    summary_scores: dict[str, dict[str, object]] = {}
    for index, (score_name, candidate) in enumerate(
        zip(score_names, comparison["prespecified_candidates"], strict=True)
    ):
        state = str(candidate["state"])
        coalition = str(candidate["coalition"])
        if coalition != coalitions[index]:
            raise ValueError(
                f"Candidate coalition mismatch: {coalition} != {coalitions[index]}"
            )
        metrics[list(states).index(state), :, index] = candidate_values[index]
        summary_scores[score_name] = {
            "selected_full_sample": {
                "metric": "targeted_first_residual",
                "state": state,
                "coalition": coalition,
                "rho": float(candidate["rho"]),
                "p_raw_two_sided": float(candidate["p_raw"]),
                "p_permutation_pointwise": float(candidate["p_permutation"]),
            }
        }

    COMPATIBILITY_ROOT.mkdir(parents=True, exist_ok=True)
    (COMPATIBILITY_ROOT / "summary.json").write_text(
        json.dumps({"scores": summary_scores}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(
        COMPATIBILITY_ROOT / "metrics.npz",
        states=states,
        subjects=subjects,
        coalitions=coalitions,
        targeted_first_residual=metrics,
    )


def main() -> int:
    prepare_compatibility_input()
    configure_style()
    plot_main_combined(
        RESULTS_ROOT,
        FINAL_ROOT,
        COGNITION,
        COMPATIBILITY_ROOT,
    )
    print(
        json.dumps(
            {
                "png": str(FINAL_ROOT / "task_evoked_xi_main_combined.png"),
                "svg": str(FINAL_ROOT / "task_evoked_xi_main_combined.svg"),
                "pdf": str(FINAL_ROOT / "task_evoked_xi_main_combined.pdf"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
