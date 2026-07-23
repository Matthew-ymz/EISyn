#!/usr/bin/env python3
"""Run the Schaefer100 DMF smoke or full reproduction with atomic progress."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/opt/anaconda3/envs/py311/bin/python")
BASE = ROOT / "results" / "dmf_schaefer100"
SOURCE = BASE / "source" / "group_mean_native_mean_rate.npz"
LABELS = BASE / "schaefer100_labels.txt"
STATUS = ROOT / "docs" / "log" / "dmf_schaefer100_progress.json"
LOG = ROOT / "docs" / "log" / "dmf_schaefer100_pipeline.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--source-results", type=Path, default=SOURCE)
    parser.add_argument("--status", type=Path, default=STATUS)
    parser.add_argument("--log", type=Path, default=LOG)
    parser.add_argument("--start-at", choices=("main", "wms", "topology", "yeo7", "plot"), default="main")
    return parser.parse_args()


def atomic_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def critical_window(source: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(source) as archive:
        g_values = np.asarray(archive["G"], dtype=float)
        critical = float(np.asarray(archive["critical_G"]).item())
    center = int(np.argmin(np.abs(g_values - critical)))
    center = min(max(center, 1), len(g_values) - 2)
    indices = np.arange(center - 1, center + 2, dtype=int)
    return indices, g_values[indices]


def run_phase(
    *,
    name: str,
    command: list[str],
    total: int,
    status_path: Path,
    log_path: Path,
    started: float,
) -> None:
    current = 0
    atomic_status(
        status_path,
        {
            "phase": name,
            "current": current,
            "total": total,
            "unit": "condition",
            "elapsed_seconds": time.monotonic() - started,
            "eta_seconds": None,
            "metrics": {},
            "updated_at": time.time(),
        },
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {name}\n")
        log_handle.write(" ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        phase_started = time.monotonic()
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
            if re.search(r"\bseed=\d+\s+G=", line):
                current = min(total, current + 1)
                elapsed = time.monotonic() - phase_started
                rate = current / elapsed if elapsed > 0 else 0.0
                atomic_status(
                    status_path,
                    {
                        "phase": name,
                        "current": current,
                        "total": total,
                        "unit": "condition",
                        "elapsed_seconds": time.monotonic() - started,
                        "eta_seconds": (total - current) / rate if rate > 0 else None,
                        "metrics": {"last_output": line.strip()},
                        "updated_at": time.time(),
                    },
                )
        return_code = process.wait()
    if return_code != 0:
        atomic_status(
            status_path,
            {
                "phase": "failed",
                "current": current,
                "total": total,
                "unit": "condition",
                "elapsed_seconds": time.monotonic() - started,
                "eta_seconds": None,
                "metrics": {"failed_phase": name},
                "message": f"Phase {name} exited with code {return_code}.",
                "updated_at": time.time(),
            },
        )
        raise subprocess.CalledProcessError(return_code, command)
    atomic_status(
        status_path,
        {
            "phase": name,
            "current": total,
            "total": total,
            "unit": "condition",
            "elapsed_seconds": time.monotonic() - started,
            "eta_seconds": 0.0,
            "metrics": {"status": "phase complete"},
            "updated_at": time.time(),
        },
    )


def main() -> None:
    args = parse_args()
    source = args.source_results if args.source_results.is_absolute() else ROOT / args.source_results
    if not source.exists():
        raise FileNotFoundError(f"Missing prepared source results: {source}")
    status_path = args.status if args.status.is_absolute() else ROOT / args.status
    log_path = args.log if args.log.is_absolute() else ROOT / args.log
    indices, critical_g = critical_window(source)
    suffix = "smoke" if args.mode == "smoke" else "full"
    run_dir = BASE / suffix
    figure_dir = ROOT / "fig" / "dmf_schaefer100"
    run_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke":
        seeds = "3"
        g_indices = ",".join(map(str, indices))
        sample_count = 256
        horizon = 30
    else:
        seeds = "3,4,5,6,7,8,9,10"
        with np.load(source) as archive:
            g_indices = ",".join(map(str, range(len(archive["G"]))))
        sample_count = 2048
        horizon = 300
    seed_count = len(seeds.split(","))
    g_count = len(g_indices.split(","))
    main_cache = run_dir / "main_confirmation.npz"
    wms_cache = run_dir / "observational_wms.npz"
    topology_cache = run_dir / "critical_topology.npz"
    yeo_cache = run_dir / "critical_yeo7.npz"
    critical_arg = ",".join(f"{value:.1f}" for value in critical_g)
    phases = ["main", "wms", "topology", "yeo7", "plot"]
    start_index = phases.index(args.start_at)
    started = time.monotonic()

    commands: list[tuple[str, list[str], int]] = [
        (
            "main",
            [
                str(PYTHON), "-u", "scripts/run_dmf_diffusive_fullstate_control.py",
                "--source-results", str(source), "--output", str(main_cache),
                "--seeds", seeds, "--g-indices", g_indices, "--modes", "direct",
                "--source-state", "se_si", "--se-low", "0.3", "--se-high", "0.7",
                "--si-low", "0.3", "--si-high", "0.7", "--sample-count", str(sample_count),
                "--horizon", str(horizon), "--ridge", "1e-6", "--dt", "0.001",
                "--sigma", "0.01", "--state-boundary", "none",
            ],
            seed_count * g_count,
        ),
        (
            "wms",
            [
                str(PYTHON), "-u", "scripts/run_dmf_aligned_observational_wms.py",
                "--source", str(source), "--reference", str(main_cache), "--output", str(wms_cache),
                "--status", str(run_dir / "wms_progress.json"), "--reuse-cache", str(wms_cache),
                "--no-resume", *( ["--dense-g"] if args.mode == "full" else [] ),
            ],
            seed_count * (3 if args.mode == "smoke" else g_count),
        ),
        (
            "topology",
            [
                str(PYTHON), "-u", "scripts/analyze_dmf_critical_phi_hierarchy_topology.py",
                "--main-confirmation", str(main_cache), "--source-results", str(source),
                "--connectivity-labels", str(LABELS), "--critical-g", critical_arg,
                "--output", str(topology_cache), "--figure", str(figure_dir / f"topology_{suffix}.png"),
                "--cross-figure", str(figure_dir / f"cross_strength_{suffix}.png"),
                "--comparison-figure", str(figure_dir / f"local_cross_{suffix}.png"),
                "--summary", str(run_dir / "topology_summary.json"),
            ],
            seed_count * 3,
        ),
        (
            "yeo7",
            [
                str(PYTHON), "-u", "scripts/analyze_dmf_critical_phi_yeo7_hierarchy.py",
                "--main-confirmation", str(main_cache), "--source-results", str(source),
                "--connectivity-labels", str(LABELS), "--critical-g", critical_arg,
                "--output", str(yeo_cache), "--figure", str(figure_dir / f"yeo7_{suffix}.png"),
                "--summary", str(run_dir / "yeo7_summary.json"),
            ],
            seed_count * 3,
        ),
        (
            "plot",
            [
                str(PYTHON), "-u", "scripts/plot_dmf_schaefer100_summary.py",
                "--source", str(source), "--main", str(main_cache), "--wms", str(wms_cache),
                "--topology", str(topology_cache), "--yeo7", str(yeo_cache),
                "--prep", str(BASE / "group_mean_native.npz"),
                "--output", str(figure_dir / f"dmf_schaefer100_summary_{suffix}"),
                *( ["--comparison-output", str(figure_dir / "dmf_83_vs_100_comparison")] if args.mode == "full" else [] ),
            ],
            1,
        ),
    ]
    try:
        for phase_index, (name, command, total) in enumerate(commands):
            if phase_index < start_index:
                continue
            run_phase(
                name=name,
                command=command,
                total=total,
                status_path=status_path,
                log_path=log_path,
                started=started,
            )
    except Exception:
        raise
    atomic_status(
        status_path,
        {
            "phase": "complete",
            "current": 1,
            "total": 1,
            "unit": "pipeline",
            "elapsed_seconds": time.monotonic() - started,
            "eta_seconds": 0.0,
            "metrics": {
                "mode": args.mode,
                "critical_window_G": critical_g.tolist(),
                "run_dir": str(run_dir.relative_to(ROOT)),
            },
            "updated_at": time.time(),
        },
    )


if __name__ == "__main__":
    main()
