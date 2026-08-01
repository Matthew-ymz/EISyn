#!/usr/bin/env python3
"""Run the frozen Schaefer-1000, 57-subject validation suite with status updates."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "hcp_schaefer1000_57_validation_suite"
TASK = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_57_brain"
REST = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_57_brain"
LABELS = (
    ROOT
    / "data"
    / "hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30"
    / "_atlas_labels"
    / "Schaefer2018_1000Parcels_7Networks_order.txt"
)


def write_status(stage: str, state: str, *, index: int, total: int, detail: str = "") -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "state": state,
        "stage_index": index,
        "stage_total": total,
        "detail": detail,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "live_progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def checkpoint_detail(stage_dir: Path) -> str:
    for name in ("checkpoint.json", "prediction_checkpoint.json"):
        path = stage_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = len(payload.get("rows", []))
            points = len(payload.get("completed_grid_points", []))
            return f"checkpoint rows={rows}, grid_points={points}"
        except (OSError, json.JSONDecodeError):
            return "checkpoint is being updated"
    return "running"


def main() -> int:
    py = sys.executable
    common_rest = ["--data-root", str(REST), "--labels", str(LABELS), "--parcel-count", "1000", "--data-key", "Schaefer1000"]
    common_task = ["--task-root", str(TASK), "--rest-root", str(REST), "--labels", str(LABELS), "--parcel-count", "1000", "--task-data-key", "Schaefer1000_taskRetained", "--rest-data-key", "Schaefer1000"]
    stages = [
        ("rest_null", OUT / "null", [py, str(ROOT / "scripts/run_hcp_schaefer500_yeo7_pc1_phi_null_all.py"), *common_rest, "--output-dir", str(OUT / "null"), "--development-end", "900", "--order", "5", "--alpha", "1", "--null-replicates", "20", "--seed", "20260714"], OUT / "null/summary.json"),
        ("module_null", OUT / "module", [py, str(ROOT / "scripts/run_hcp_schaefer500_yeo7_module_phi_decomposition.py"), *common_rest, "--output-dir", str(OUT / "module"), "--development-end", "900", "--order", "5", "--alpha", "1", "--null-replicates", "20", "--seed", "20260714", "--top-k", "3"], OUT / "module/summary.json"),
        ("robustness", OUT / "robustness", [py, str(ROOT / "scripts/run_hcp_schaefer500_phi_hyperparameter_robustness.py"), *common_task, "--output-dir", str(OUT / "robustness"), "--orders", "1,2,3,5,8", "--alphas", "0.1,1,10,100,1000", "--permutation-replicates", "200000"], OUT / "robustness/summary.json"),
        ("prediction", OUT / "prediction", [py, str(ROOT / "scripts/run_hcp_schaefer500_phi_prediction_error_grid.py"), *common_task, "--output-dir", str(OUT / "prediction"), "--phi-summary", str(OUT / "robustness/summary.json"), "--orders", "1,2,3,5,8", "--alphas", "0.1,1,10,100,1000"], OUT / "prediction/prediction_error_summary.json"),
        ("tevf", OUT / "tevf", [py, str(ROOT / "scripts/analyze_hcp_schaefer500_task_specific_regions.py"), "--task-root", str(TASK), "--rest-root", str(REST), "--label-file", str(LABELS), "--output-dir", str(OUT / "tevf"), "--parcel-count", "1000", "--permutations", "2000", "--seed", "20260718"], OUT / "tevf/summary.json"),
    ]
    for index, (name, stage_dir, command, expected) in enumerate(stages, start=1):
        if expected.is_file():
            write_status(name, "complete", index=index, total=len(stages), detail="reused completed output")
            continue
        stage_dir.mkdir(parents=True, exist_ok=True)
        write_status(name, "running", index=index, total=len(stages))
        with (stage_dir / "run.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
            while process.poll() is None:
                write_status(name, "running", index=index, total=len(stages), detail=checkpoint_detail(stage_dir))
                time.sleep(5)
        if process.returncode:
            write_status(name, "failed", index=index, total=len(stages), detail=f"exit code {process.returncode}")
            return int(process.returncode)
        if not expected.is_file():
            write_status(name, "failed", index=index, total=len(stages), detail=f"missing {expected.name}")
            return 2
        write_status(name, "complete", index=index, total=len(stages))
    write_status("all", "complete", index=len(stages), total=len(stages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
