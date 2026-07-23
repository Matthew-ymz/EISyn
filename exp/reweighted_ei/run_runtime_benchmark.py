from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp.reweighted_ei import (
    ExperimentConfig,
    aggregate_runtime_benchmark,
    benchmark_method_runtimes,
    plot_runtime_benchmark,
)
from exp.reweighted_ei.reweighted_ei_experiment import (
    observational_mi_quadrature,
    oracle_ei_quadrature,
)


RESULT_DIR = PROJECT_ROOT / "exp" / "reweighted_ei" / "results"
OUTPUT_PATH = RESULT_DIR / "runtime_benchmark.json"
STATUS_PATH = RESULT_DIR / "runtime_benchmark_progress.json"
SAMPLE_SIZES = (1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000)
METHOD_COUNT = 5


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    seeds = tuple(range(10))
    previous_records = pd.DataFrame()
    previous_truth_rows: list[dict[str, object]] = []
    if OUTPUT_PATH.exists():
        previous_payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        previous_records = pd.DataFrame(previous_payload.get("records", []))
        previous_truth_rows = list(previous_payload.get("truth_quadrature", []))

    valid_previous = previous_records[
        previous_records.get("n_samples", pd.Series(dtype=int)).isin(SAMPLE_SIZES)
        & previous_records.get("seed", pd.Series(dtype=int)).isin(seeds)
    ].copy() if not previous_records.empty else previous_records
    total_rows = (
        valid_previous[valid_previous["stage"] == "total"]
        if not valid_previous.empty
        else pd.DataFrame()
    )
    completed_pairs = {
        (int(sample_size), int(seed))
        for (sample_size, seed), block in total_rows.groupby(["n_samples", "seed"])
        if block["method"].nunique() == METHOD_COUNT
    }
    total = len(SAMPLE_SIZES) * len(seeds)
    completed = len(completed_pairs)
    experiment_started = time.monotonic()

    def write_status(*, phase: str, sample_size: int | None, seed: int | None) -> None:
        elapsed = time.monotonic() - experiment_started
        new_work = max(completed - len(completed_pairs), 0)
        rate = new_work / elapsed if elapsed > 0.0 else 0.0
        eta = (total - completed) / rate if rate > 0.0 else None
        _atomic_json(
            STATUS_PATH,
            {
                "phase": phase,
                "current": completed,
                "total": total,
                "unit": "seed",
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "sample_size": sample_size,
                "seed": seed,
                "updated_at": time.time(),
            },
        )

    frames = [valid_previous]
    progress = tqdm(
        total=total,
        initial=completed,
        desc="runtime benchmark",
        unit="seed",
        mininterval=1.0,
    )
    write_status(phase="running", sample_size=None, seed=None)
    for sample_size in SAMPLE_SIZES:
        config = ExperimentConfig(
            n_samples=sample_size,
            rho=0.5,
            seeds=seeds,
            intervention_samples=sample_size,
            mlp_epochs=120,
            tm_degree=5,
            knn_k=20,
            oracle_nodes=96,
            oracle_y_points=3_000,
        )
        for seed in seeds:
            if (sample_size, seed) in completed_pairs:
                continue
            frames.append(benchmark_method_runtimes(config, seeds=(seed,)))
            completed += 1
            progress.update(1)
            progress.set_postfix(n=sample_size, seed=seed)
            write_status(phase="running", sample_size=sample_size, seed=seed)
    progress.close()
    records = pd.concat(frames, ignore_index=True)

    truth_rows = previous_truth_rows
    completed_truth = {
        (str(row["method"]), int(row["repeat"]))
        for row in truth_rows
    }
    for repeat in range(3):
        truth_started = time.perf_counter()
        if ("ei_truth_quadrature", repeat) not in completed_truth:
            ei_value = oracle_ei_quadrature(noise_sd=0.3, nodes=96, y_points=3_000)
            truth_rows.append(
                {
                    "method": "ei_truth_quadrature",
                    "method_label": "EI truth quadrature",
                    "repeat": repeat,
                    "seconds": time.perf_counter() - truth_started,
                    "information_bits": ei_value,
                }
            )
        truth_started = time.perf_counter()
        if ("ordinary_mi_truth_quadrature", repeat) not in completed_truth:
            mi_value = observational_mi_quadrature(
                noise_sd=0.3,
                rho=0.5,
                nodes=96,
                y_points=3_000,
            )
            truth_rows.append(
                {
                    "method": "ordinary_mi_truth_quadrature",
                    "method_label": "Ordinary MI truth quadrature",
                    "repeat": repeat,
                    "seconds": time.perf_counter() - truth_started,
                    "information_bits": mi_value,
                }
            )

    summary = aggregate_runtime_benchmark(records)
    payload = {
        "schema_version": 1,
        "benchmark_contract": {
            "seeds": list(seeds),
            "sample_sizes": list(SAMPLE_SIZES),
            "rho": 0.5,
            "mlp_epochs": 120,
            "tm_degree": 5,
            "knn_k": 20,
            "timing_scope": "warm-process method time; common data generation excluded",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
        },
        "records": records.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "truth_quadrature": truth_rows,
    }
    _atomic_json(OUTPUT_PATH, payload)
    figure = plot_runtime_benchmark(records, output_dir=RESULT_DIR)
    plt.close(figure)
    write_status(phase="complete", sample_size=None, seed=None)
    print(f"wrote {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
