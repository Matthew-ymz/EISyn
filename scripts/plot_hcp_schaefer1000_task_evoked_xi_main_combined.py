#!/usr/bin/env python3
"""Render the Schaefer-1000 main figure with direct Language behavior panels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
LANGUAGE_RESULTS = (
    ROOT / "results/hcp_language_story_math_candidates_schaefer1000_replication"
)
LANGUAGE_BEHAVIOR = (
    ROOT / "Data/unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
)
WM_BACK_RESULTS = ROOT / "results/hcp_wm_back_condition_correlations"
WM_FIXED_METRICS = (
    ROOT
    / "results/hcp_task_score_synergy_schaefer1000_validation"
    / "fixed_candidates_schaefer1000.npz"
)


def main() -> int:
    configure_style()
    plot_main_combined(
        RESULTS_ROOT,
        FINAL_ROOT,
        COGNITION,
        None,
        language_story_math_root=LANGUAGE_RESULTS,
        language_behavior_scores=LANGUAGE_BEHAVIOR,
        wm_back_condition_root=WM_BACK_RESULTS,
        wm_fixed_candidate_metrics=WM_FIXED_METRICS,
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
