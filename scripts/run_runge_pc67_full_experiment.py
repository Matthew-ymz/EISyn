#!/usr/bin/env python3
"""Run the controlled 67-PC SLP experiment with resumable progress reporting."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results/runge_slp_daily_1948_2026_pc67_20260731"
MLP_ROOT = RUN_ROOT / "mlp_tm_ei_lag04"
RUNGE_ROOT = MLP_ROOT / "results/runge"
PAIRWISE = RUNGE_ROOT / "pairwise_mlp_tm_ei_path_effects"
FORCED = RUNGE_ROOT / "multistep_conditioned_ei_tm_forced_edges"
EXHAUSTIVE = RUNGE_ROOT / "multistep_conditioned_ei_tm_exhaustive"
FIG_ROOT = ROOT / "fig/runge_slp_daily_1948_2026_pc67_20260731"
STATUS = ROOT / "docs/log/runge_slp_pc67_live_progress.json"
LOG = ROOT / "docs/log/runge_slp_pc67_full.log"
HORIZONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60)


def write_status(
    *,
    phase: str,
    current: int,
    total: int,
    unit: str,
    started: float,
    message: str = "",
) -> None:
    elapsed = time.monotonic() - started
    rate = current / elapsed if current > 0 and elapsed > 0 else 0.0
    payload = {
        "phase": phase,
        "current": int(current),
        "total": int(total),
        "unit": unit,
        "elapsed_seconds": elapsed,
        "eta_seconds": (total - current) / rate if rate > 0 else None,
        "metrics": {},
        "message": message,
        "updated_at": time.time(),
    }
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, STATUS)


def run_monitored(
    command: list[str],
    *,
    phase: str,
    total: int,
    unit: str,
    count_completed,
) -> None:
    started = time.monotonic()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(f"\n$ {' '.join(command)}\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            current = min(int(count_completed()), int(total))
            write_status(
                phase=phase,
                current=current,
                total=total,
                unit=unit,
                started=started,
            )
            time.sleep(2.0)
        current = min(int(count_completed()), int(total))
        write_status(
            phase=phase,
            current=current,
            total=total,
            unit=unit,
            started=started,
        )
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)


def main() -> int:
    python = sys.executable
    try:
        run_monitored(
            [
                python,
                "scripts/run_runge_pairwise_mlp_ei.py",
                "--component-scores",
                str(RUN_ROOT / "results/runge/2015_gateways/component_weekly_scores.csv"),
                "--output-dir",
                str(MLP_ROOT),
                "--lag",
                "4",
                "--hidden-dim",
                "128",
                "--num-layers",
                "1",
                "--dropout",
                "0.5",
                "--epochs",
                "120",
                "--learning-rate",
                "1e-3",
                "--weight-decay",
                "1e-3",
                "--ridge-alpha",
                "1000",
                "--ensemble-ridge-alphas",
                "10,100,1000,3000",
                "--linear-blend-grid-steps",
                "101",
                "--ei-estimator",
                "tm",
                "--gateway-mode",
                "path_effect",
                "--early-stopping-patience",
                "80",
                "--scheduler-patience",
                "20",
                "--gradient-clip-norm",
                "1.0",
            ],
            phase="mlp",
            total=4,
            unit="ensemble model",
            count_completed=lambda: len(list((RUNGE_ROOT / "pairwise_mlp_ei").glob("*.pt"))),
        )
        run_monitored(
            [
                python,
                "scripts/run_runge_forced_tm_edge_trends.py",
                "--pairwise-manifest",
                str(PAIRWISE / "manifest.json"),
                "--result-dir",
                str(FORCED),
                "--output",
                str(FIG_ROOT / "multistep_conditioned_ei_tm_targeted/forced_tm_edge_trends_H001_H060.png"),
                "--horizons",
                "1-10,15,20,30,40,50,60",
                "--edges",
                "0+6->30,0+1->56,0+1->57,0+1->50",
            ],
            phase="forced_edges",
            total=64,
            unit="edge-horizon",
            count_completed=lambda: len(list(FORCED.glob("H*.json"))),
        )
        run_monitored(
            [
                python,
                "scripts/run_runge_exhaustive_degree3_tm.py",
                "--pairwise-manifest",
                str(PAIRWISE / "manifest.json"),
                "--rollout-path",
                str(FORCED / "rollout_predictions_H060_n4096.npy"),
                "--old-rerank-dir",
                str(
                    ROOT
                    / "results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/"
                    "multistep_conditioned_ei_tm_targeted"
                ),
                "--result-dir",
                str(EXHAUSTIVE),
                "--horizons",
                "1-10,15,20,30,40,50,60",
                "--workers",
                "1",
                "--validation-samples",
                "24",
            ],
            phase="exhaustive_tm",
            total=len(HORIZONS) * 67,
            unit="target-horizon",
            count_completed=lambda: len(list(EXHAUSTIVE.glob("H*/chunks/target_*.npz"))),
        )
        run_monitored(
            [
                python,
                "scripts/plot_runge_source_pair_condensation.py",
                "--result-dir",
                str(EXHAUSTIVE),
                "--output",
                str(FIG_ROOT / "earth_slp_source_pair_condensation"),
            ],
            phase="figures",
            total=2,
            unit="figure",
            count_completed=lambda: int((FIG_ROOT / "earth_slp_source_pair_condensation.png").exists())
            + int((FIG_ROOT / "earth_slp_hyperedge_dynamics.png").exists()),
        )
        run_monitored(
            [
                python,
                "scripts/plot_earth_system_main_figures.py",
                "--output-dir",
                str(FIG_ROOT),
                "--runge-result-dir",
                str(EXHAUSTIVE),
                "--runge-component-maps",
                str(RUN_ROOT / "results/runge/2015_gateways/component_maps.npz"),
                "--runge-trend-csv",
                str(FIG_ROOT / "multistep_conditioned_ei_tm_targeted/forced_tm_edge_trends_H001_H060.csv"),
                "--runge-focal-pair",
                "0,1",
                "--skip-unicm",
            ],
            phase="figures",
            total=2,
            unit="figure",
            count_completed=lambda: int((FIG_ROOT / "earth_slp_source_pair_condensation.png").exists())
            + int((FIG_ROOT / "earth_slp_hyperedge_dynamics.png").exists()),
        )
        write_status(
            phase="complete",
            current=1,
            total=1,
            unit="experiment",
            started=time.monotonic(),
            message=str(FIG_ROOT / "earth_slp_hyperedge_dynamics.png"),
        )
        return 0
    except Exception as error:
        write_status(
            phase="failed",
            current=0,
            total=1,
            unit="experiment",
            started=time.monotonic(),
            message=str(error),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
