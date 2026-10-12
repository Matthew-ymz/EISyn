#!/usr/bin/env python3
"""Refine observed pilot boundaries under the original total compute budget."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analyze_spt_kuramoto_transition_pilot import load_records, refinement_plan
from scripts.run_spt_kuramoto_transition_pilot import OUT, STATUS, save


def run():
    coarse = json.loads((OUT / "summary.json").read_text())
    if coarse["coverage"]["completed"] != 96:
        raise RuntimeError("Finish the approved coarse-grid pilot before refinement.")
    used = coarse["elapsed_seconds"]
    directories = [OUT]
    rounds = []
    stop_reason = "no_pending_boundary"
    for index in range(1, 13):
        records = load_records(directories)
        plan = refinement_plan(records)
        points = plan["first_event_midpoints"] or plan["k_out_priority_order"]
        if not points:
            break
        # Reserve two minutes so a started pair can finish without exceeding 90 minutes.
        remaining = 88 - used / 60
        if remaining <= 0:
            stop_reason = "total_compute_budget"
            break
        directory = OUT / f"refinement_round_{index:02d}"
        save(OUT / f"refinement_plan_{index:02d}.json", {**plan, "launched_k_out": points,
             "remaining_compute_minutes": remaining,
             "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        print(f"REFINEMENT round={index} K_out={points} remaining={remaining:.1f} minutes", flush=True)
        command = [sys.executable, str(ROOT / "scripts/run_spt_kuramoto_transition_pilot.py"),
                   "--output", str(directory), "--status", str(STATUS),
                   "--max-minutes", str(remaining), "--k-out", *map(str, points)]
        subprocess.run(command, cwd=ROOT, check=True)
        payload = json.loads((directory / "summary.json").read_text())
        used += payload["elapsed_seconds"]
        directories.append(directory)
        rounds.append({"directory": str(directory), "grid": points,
                       "coverage": payload["coverage"], "elapsed_seconds": payload["elapsed_seconds"]})
        if payload["coverage"]["completed"] != payload["coverage"]["planned"]:
            stop_reason = "total_compute_budget"
            break
    else:
        stop_reason = "round_limit"
    records = load_records(directories)
    final = {"coarse_records": 96, "total_records": len(records), "rounds": rounds,
             "total_compute_seconds": used, "stop_reason": stop_reason,
             "remaining_refinement": refinement_plan(records),
             "directories": list(map(str, directories))}
    save(OUT / "refinement_summary.json", final)
    save(STATUS, {"phase": "complete", "scope": "coarse plus budgeted refinement",
                  "completed": len(records), "compute_seconds": used, "stop_reason": stop_reason,
                  "result": str(OUT / "refinement_summary.json")})
    print(json.dumps({k: v for k, v in final.items() if k not in ("rounds", "remaining_refinement")}, indent=2), flush=True)


if __name__ == "__main__":
    run()
