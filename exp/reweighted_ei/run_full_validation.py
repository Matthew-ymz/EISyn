from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp.reweighted_ei.reweighted_ei_experiment import (
    aggregate_full_validation,
    build_full_interpretation,
    evaluate_between_relation,
    full_grid_configs,
    full_grid_sweeps,
    plot_full_validation,
    run_experiment,
    summarize_full_agreement,
)


EXPERIMENT_DIR = PROJECT_ROOT / "exp" / "reweighted_ei"
CACHE_DIR = EXPERIMENT_DIR / "cache" / "full_validation"
RESULT_DIR = EXPERIMENT_DIR / "results"
STATUS_PATH = RESULT_DIR / "full_validation_progress.json"
RECORDS_PATH = RESULT_DIR / "full_validation_records.json"
SUMMARY_PATH = RESULT_DIR / "full_validation_summary.json"


def _atomic_json(path: Path, payload: object, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _cache_path(index: int, n_samples: int, rho: float) -> Path:
    return CACHE_DIR / f"grid_{index:02d}_n{n_samples}_rho{rho:.1f}.json"


def _completed_seed_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    if payload.get("schema_version") != 3:
        return 0
    return len(payload.get("completed_seeds", []))


def main() -> None:
    configs = list(full_grid_configs())
    sweeps = list(full_grid_sweeps())
    if len(configs) != len(sweeps):
        raise RuntimeError("full-grid configs and sweep labels must have matching lengths.")
    cache_by_condition: dict[tuple[int, float], Path] = {}
    cache_paths = []
    for index, config in enumerate(configs):
        key = (int(config.n_samples), float(config.rho))
        cache_paths.append(
            cache_by_condition.setdefault(
                key,
                _cache_path(index, config.n_samples, config.rho),
            )
        )
    total = sum(len(config.seeds) for config in configs)
    completed = sum(_completed_seed_count(path) for path in cache_paths)
    started = time.monotonic()
    frames: list[pd.DataFrame] = []

    def write_status(*, phase: str, grid_id: int | None, seed: int | None, message: str = "") -> None:
        elapsed = time.monotonic() - started
        new_work = max(completed - initial_completed, 0)
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
                "grid_id": grid_id,
                "seed": seed,
                "message": message,
                "updated_at": time.time(),
            },
        )

    initial_completed = completed
    write_status(phase="initializing", grid_id=None, seed=None)
    progress = tqdm(total=total, initial=completed, desc="full EI validation", unit="seed", mininterval=1.0)
    try:
        for index, (config, sweep, cache_path) in enumerate(zip(configs, sweeps, cache_paths)):
            def on_seed(seed: int, _local_completed: int, _local_total: int) -> None:
                nonlocal completed
                completed += 1
                progress.update(1)
                progress.set_postfix(grid=index, n=config.n_samples, rho=f"{config.rho:.1f}")
                write_status(phase="running", grid_id=index, seed=seed)

            frame = run_experiment(
                config,
                cache_path=cache_path,
                progress_callback=on_seed,
            )
            frame["grid_id"] = index
            frame["sweep"] = sweep
            frames.append(frame)
            write_status(phase="running", grid_id=index, seed=None, message="configuration complete")

        records = pd.concat(frames, ignore_index=True)
        summary = aggregate_full_validation(records)
        agreement = summarize_full_agreement(records)
        between_frames = []
        for grid_id, block in records.groupby("grid_id", sort=True):
            relation = evaluate_between_relation(block)
            relation["grid_id"] = int(grid_id)
            relation["sweep"] = str(block["sweep"].iloc[0])
            relation["n_samples"] = int(block["n_samples"].iloc[0])
            relation["rho"] = float(block["rho"].iloc[0])
            between_frames.append(relation)
        between = pd.concat(between_frames, ignore_index=True)

        _atomic_json(
            RECORDS_PATH,
            {"schema_version": 1, "records": records.to_dict(orient="records")},
        )
        _atomic_json(
            SUMMARY_PATH,
            {
                "schema_version": 1,
                "summary": summary.to_dict(orient="records"),
                "agreement": agreement.to_dict(orient="records"),
                "between": between.to_dict(orient="records"),
                "interpretation_markdown": build_full_interpretation(records),
            },
            indent=2,
        )
        figure = plot_full_validation(records, output_dir=RESULT_DIR)
        plt.close(figure)
        write_status(phase="complete", grid_id=None, seed=None, message="all outputs written")
    except Exception as error:
        write_status(phase="failed", grid_id=None, seed=None, message=str(error))
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()
