#!/usr/bin/env python3
"""Test Lead-1 checkpoint-1 Syn convergence with independent interventions."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_unicm_phi_eid_greedy_decomposition import (  # noqa: E402
    _regularized_logdet,
    compute_subset_ei_table_from_covariance,
    precompute_source_logdets,
)
from scripts.spt import nontrivial_bipartitions  # noqa: E402
from scripts.unicm_peid_syn_analysis import (  # noqa: E402
    MODE_NAMES,
    PREDICTION_LENGTH,
    HISTORY_LENGTH,
    load_unicm_model,
    make_full_history_mode_tensor,
    resolve_checkpoint_paths,
    sample_full_history_mode_inputs,
)


OUTPUT_DIR = ROOT / "results/unicm_lead1_nonnegativity_convergence"
FIGURE = ROOT / "fig/earth_unicm_lead1_nonnegativity_convergence.png"
CHECKPOINT_ROOT = ROOT / "data/UniCM-checkpoint/src/experiments"
FIXED_LEFT = ("nino", "NPMM", "SPMM", "IOD", "SIOD", "nino3", "nino4", "WWV")
FIXED_RIGHT = ("IOB", "TNA", "nino12")


def _cache_path(cache_dir: Path, sampling_seed: int, max_samples: int) -> Path:
    return cache_dir / f"checkpoint1_lead1_seed{sampling_seed}_n{max_samples}_bound4.npz"


def _predict_lead1(
    model,
    histories: np.ndarray,
    *,
    device: str,
    batch_size: int,
    progress_callback=None,
) -> np.ndarray:
    import torch

    rows = []
    count = int(histories.shape[0])
    started = time.monotonic()
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, count, int(batch_size))):
            end = min(count, start + int(batch_size))
            predictor = torch.tensor(
                make_full_history_mode_tensor(histories[start:end]),
                device=device,
                dtype=torch.float32,
            )
            months = torch.arange(
                HISTORY_LENGTH + PREDICTION_LENGTH,
                device=device,
                dtype=torch.int64,
            ) % 12
            timestamps = months.unsqueeze(0).repeat(end - start, 1)
            out, _, _ = model.forward_sep(
                predictor,
                timestamps,
                model.encoder_mode,
                model.decoder_mode,
                model.linear_output_mode,
                model.predictor_emb_mode,
                model.predictand_emb_mode,
                1,
                [1, 1],
                train=False,
            )
            all_modes = np.asarray(
                out.squeeze(-1).squeeze(2).detach().cpu().tolist(),
                dtype=np.float32,
            )
            rows.append(np.asarray(all_modes[:, 0, :], dtype=np.float32))
            if progress_callback is not None:
                progress_callback(end, count)
            if batch_index == 0 or end == count or (batch_index + 1) % 8 == 0:
                elapsed = time.monotonic() - started
                print(f"  predicted {end}/{count} samples in {elapsed:.1f}s", flush=True)
    return np.concatenate(rows, axis=0)


def _write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _mode_columns(names: tuple[str, ...]) -> list[int]:
    indices = {name: index for index, name in enumerate(MODE_NAMES)}
    return [
        month * len(MODE_NAMES) + indices[name]
        for month in range(HISTORY_LENGTH)
        for name in names
    ]


def _mi_cmi(histories: np.ndarray, target: np.ndarray, *, jitter: float) -> tuple[float, float]:
    flat = histories.reshape(histories.shape[0], -1).astype(float)
    left = flat[:, _mode_columns(FIXED_LEFT)]
    right = flat[:, _mode_columns(FIXED_RIGHT)]
    arrays = {
        "A": left,
        "B": right,
        "Y": target,
        "AB": np.concatenate([left, right], axis=1),
        "AY": np.concatenate([left, target], axis=1),
        "BY": np.concatenate([right, target], axis=1),
        "ABY": np.concatenate([left, right, target], axis=1),
    }
    logdet = {
        name: _regularized_logdet(np.cov(array, rowvar=False, bias=False), jitter=jitter)
        for name, array in arrays.items()
    }
    source_mi = 0.5 * (logdet["A"] + logdet["B"] - logdet["AB"]) / np.log(2.0)
    conditional_mi = 0.5 * (
        logdet["AY"] + logdet["BY"] - logdet["Y"] - logdet["ABY"]
    ) / np.log(2.0)
    return float(source_mi), float(conditional_mi)


def _evaluate_level(
    histories: np.ndarray,
    target: np.ndarray,
    *,
    jitter: float,
    tolerance: float,
) -> dict[str, float | int]:
    mode_names = tuple(MODE_NAMES)
    history_flat, subset_columns, source_logdets = precompute_source_logdets(
        histories,
        jitter=jitter,
    )
    ei_table = compute_subset_ei_table_from_covariance(
        history_flat,
        target,
        subset_columns,
        source_logdets,
        jitter=jitter,
    )
    singleton = {name: float(ei_table[(name,)]) for name in mode_names}

    def xi(subset: tuple[str, ...]) -> float:
        return float(ei_table[subset] - sum(singleton[name] for name in subset))

    total_xi = xi(mode_names)
    fixed_syn = float(total_xi - xi(FIXED_LEFT) - xi(FIXED_RIGHT))
    residuals = np.asarray(
        [total_xi - xi(left) - xi(right) for left, right in nontrivial_bipartitions(mode_names)],
        dtype=float,
    )
    source_mi, conditional_mi = _mi_cmi(histories, target, jitter=jitter)
    identity_error = float(fixed_syn - (conditional_mi - source_mi))
    if abs(identity_error) > 1.0e-8:
        raise RuntimeError(f"MI identity closure failed: {identity_error:.12g} bits")
    return {
        "total_xi_bits": total_xi,
        "fixed_split_syn_bits": fixed_syn,
        "minimum_root_syn_bits": float(residuals.min()),
        "negative_root_candidate_count": int(np.count_nonzero(residuals < 0.0)),
        "significant_negative_root_candidate_count": int(np.count_nonzero(residuals < -tolerance)),
        "root_candidate_count": int(residuals.size),
        "source_mi_bits": source_mi,
        "conditional_mi_bits": conditional_mi,
        "identity_error_bits": identity_error,
    }


def _summarize(rows: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    metrics = (
        "fixed_split_syn_bits",
        "minimum_root_syn_bits",
        "significant_negative_root_candidate_count",
        "source_mi_bits",
        "conditional_mi_bits",
        "total_xi_bits",
    )
    summaries = []
    for sample_size in sorted({int(row["sample_size"]) for row in rows}):
        current = [row for row in rows if int(row["sample_size"]) == sample_size]
        summary: dict[str, float | int] = {
            "sample_size": sample_size,
            "replicate_count": len(current),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in current], dtype=float)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary[f"{metric}_min"] = float(values.min())
            summary[f"{metric}_max"] = float(values.max())
        summary["fixed_split_positive_count"] = int(
            sum(float(row["fixed_split_syn_bits"]) > 0.0 for row in current)
        )
        summary["root_audit_pass_count"] = int(
            sum(int(row["significant_negative_root_candidate_count"]) == 0 for row in current)
        )
        summaries.append(summary)
    return summaries


def _plot(rows: list[dict[str, float | int]], summaries: list[dict[str, float | int]], output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), layout="constrained")
    levels = np.asarray([int(item["sample_size"]) for item in summaries], dtype=float)
    colors = {int(seed): color for seed, color in zip(
        sorted({int(row["sampling_seed"]) for row in rows}),
        mpl.colormaps["tab10"](np.linspace(0.05, 0.75, len({int(row["sampling_seed"]) for row in rows}))),
        strict=True,
    )}
    for metric, axis, panel, ylabel in (
        ("fixed_split_syn_bits", axes[0], "a", "Fixed-split Syn (bits)"),
        ("minimum_root_syn_bits", axes[1], "b", "Minimum root-candidate Syn (bits)"),
    ):
        for seed, color in colors.items():
            current = sorted(
                (row for row in rows if int(row["sampling_seed"]) == seed),
                key=lambda row: int(row["sample_size"]),
            )
            axis.plot(
                [int(row["sample_size"]) for row in current],
                [float(row[metric]) for row in current],
                color=color,
                marker="o",
                markersize=2.7,
                linewidth=0.75,
                alpha=0.55,
            )
        mean = np.asarray([float(item[f"{metric}_mean"]) for item in summaries])
        sd = np.asarray([float(item[f"{metric}_sd"]) for item in summaries])
        axis.plot(levels, mean, color="#17212B", marker="o", markersize=3.4, linewidth=1.45, label="mean")
        axis.fill_between(levels, mean - sd, mean + sd, color="#8EA3B5", alpha=0.25, linewidth=0)
        axis.axhline(0.0, color="#394955", linewidth=0.75)
        axis.axhline(-1.0e-4, color="#B5483F", linewidth=0.75, linestyle=(0, (3, 2)))
        axis.set_xscale("log", base=2)
        axis.set_xticks(levels, [f"{int(value / 1024)}k" for value in levels])
        axis.set_xlabel("Independent intervention samples")
        axis.set_ylabel(ylabel)
        axis.text(0.0, 1.02, panel, transform=axis.transAxes, fontweight="bold", fontsize=8, va="bottom")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=1)
    count_mean = np.asarray([
        float(item["significant_negative_root_candidate_count_mean"]) for item in summaries
    ])
    count_sd = np.asarray([
        float(item["significant_negative_root_candidate_count_sd"]) for item in summaries
    ])
    axes[2].errorbar(
        levels,
        count_mean,
        yerr=count_sd,
        color="#B5483F",
        marker="o",
        markersize=3.4,
        linewidth=1.2,
        capsize=2.0,
    )
    axes[2].axhline(0.0, color="#394955", linewidth=0.75)
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(levels, [f"{int(value / 1024)}k" for value in levels])
    axes[2].set_xlabel("Independent intervention samples")
    axes[2].set_ylabel("Root candidates below -1e-4 bits")
    axes[2].text(0.0, 1.02, "c", transform=axes[2].transAxes, fontweight="bold", fontsize=8, va="bottom")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    levels = sorted(set(int(value) for value in args.sample_levels))
    if levels[-1] > int(args.max_samples):
        raise ValueError("sample levels cannot exceed max-samples")
    output_dir = Path(args.output_dir)
    status_path = Path(args.status_file)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    batch_count = math.ceil(int(args.max_samples) / int(args.batch_size))
    total_work = len(args.sampling_seeds) * (batch_count + len(levels))
    completed_work = 0
    run_started = time.monotonic()

    def update_status(
        phase: str,
        *,
        seed: int | None = None,
        sample_size: int | None = None,
        metrics: dict[str, object] | None = None,
        status: str = "running",
    ) -> None:
        elapsed = time.monotonic() - run_started
        eta = elapsed * (total_work - completed_work) / completed_work if completed_work else None
        _write_status(
            status_path,
            {
                "status": status,
                "phase": phase,
                "current": completed_work,
                "total": total_work,
                "unit": "work units",
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "sampling_seed": seed,
                "sample_size": sample_size,
                "metrics": metrics or {},
                "pid": os.getpid(),
                "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )

    update_status("loading checkpoint")
    checkpoint = resolve_checkpoint_paths(Path(args.checkpoint_root), [1])[1]
    print(f"loading checkpoint {checkpoint}", flush=True)
    model = load_unicm_model(checkpoint, args.device)
    rows: list[dict[str, float | int]] = []
    for sampling_seed in args.sampling_seeds:
        sampling_seed = int(sampling_seed)
        print(f"sampling seed {sampling_seed}", flush=True)
        histories = sample_full_history_mode_inputs(
            n_samples=int(args.max_samples),
            intervention_bound=float(args.intervention_bound),
            seed=sampling_seed,
        )
        cache_path = _cache_path(cache_dir, sampling_seed, int(args.max_samples))
        if cache_path.exists():
            with np.load(cache_path) as payload:
                lead1 = np.asarray(payload["lead1_targets"], dtype=float)
            if tuple(lead1.shape) != (int(args.max_samples), len(MODE_NAMES)) or not np.isfinite(lead1).all():
                raise ValueError(f"Invalid prediction cache: {cache_path}")
            print(f"  reused {cache_path}", flush=True)
            completed_work += batch_count
            update_status("prediction cache reused", seed=sampling_seed)
        else:
            previous_batches = 0

            def prediction_progress(predicted: int, total: int) -> None:
                nonlocal completed_work, previous_batches
                current_batches = math.ceil(predicted / int(args.batch_size))
                completed_work += current_batches - previous_batches
                previous_batches = current_batches
                update_status(
                    "predicting Lead 1",
                    seed=sampling_seed,
                    sample_size=predicted,
                    metrics={"predicted_samples": predicted, "prediction_total": total},
                )

            lead1 = _predict_lead1(
                model,
                histories,
                device=str(args.device),
                batch_size=int(args.batch_size),
                progress_callback=prediction_progress,
            )
            if tuple(lead1.shape) != (int(args.max_samples), len(MODE_NAMES)) or not np.isfinite(lead1).all():
                raise ValueError(
                    f"Nonfinite or malformed Lead-1 predictions for sampling seed {sampling_seed}: "
                    f"shape={lead1.shape}, finite={bool(np.isfinite(lead1).all())}"
                )
            np.savez_compressed(
                cache_path,
                lead1_targets=lead1.astype(np.float32),
                metadata=json.dumps(
                    {
                        "checkpoint": 1,
                        "lead": 1,
                        "sampling_seed": sampling_seed,
                        "max_samples": int(args.max_samples),
                        "intervention_bound": float(args.intervention_bound),
                    },
                    sort_keys=True,
                ),
            )
        for sample_size in levels:
            started = time.monotonic()
            result = _evaluate_level(
                histories[:sample_size],
                lead1[:sample_size],
                jitter=float(args.jitter),
                tolerance=float(args.syn_tolerance),
            )
            row = {
                "sampling_seed": sampling_seed,
                "sample_size": sample_size,
                **result,
            }
            rows.append(row)
            completed_work += 1
            update_status(
                "evaluating Syn convergence",
                seed=sampling_seed,
                sample_size=sample_size,
                metrics={
                    "fixed_split_syn_bits": result["fixed_split_syn_bits"],
                    "minimum_root_syn_bits": result["minimum_root_syn_bits"],
                    "significant_negative_root_candidate_count": result[
                        "significant_negative_root_candidate_count"
                    ],
                },
            )
            print(
                f"  n={sample_size} fixed={result['fixed_split_syn_bits']:+.6f} "
                f"root_min={result['minimum_root_syn_bits']:+.6f} "
                f"violations={result['significant_negative_root_candidate_count']} "
                f"analysis={time.monotonic() - started:.1f}s",
                flush=True,
            )
        partial = {
            "rows": rows,
            "completed_sampling_seeds": sorted({int(row["sampling_seed"]) for row in rows}),
        }
        (output_dir / "partial.json").write_text(
            json.dumps(partial, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    summaries = _summarize(rows)
    _plot(rows, summaries, Path(args.figure))
    payload = {
        "experiment": "UniCM Lead-1 checkpoint-1 independent-intervention nonnegativity convergence",
        "checkpoint": 1,
        "lead": 1,
        "fixed_left": list(FIXED_LEFT),
        "fixed_right": list(FIXED_RIGHT),
        "sampling_seeds": [int(seed) for seed in args.sampling_seeds],
        "sample_levels": levels,
        "max_samples": int(args.max_samples),
        "intervention_bound": float(args.intervention_bound),
        "estimator": "affine degree-1 TM / Gaussian log-det equivalent",
        "jitter": float(args.jitter),
        "syn_tolerance_bits": float(args.syn_tolerance),
        "figure": str(args.figure),
        "rows": rows,
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    completed_work = total_work
    update_status(
        "complete",
        metrics={"summary": str(output_dir / "summary.json"), "figure": str(args.figure)},
        status="complete",
    )
    print(Path(args.figure), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    parser.add_argument("--sampling-seeds", nargs="+", type=int, default=[20260901, 20260902])
    parser.add_argument("--sample-levels", nargs="+", type=int, default=[2048, 4096, 8192, 16384])
    parser.add_argument("--max-samples", type=int, default=16384)
    parser.add_argument("--intervention-bound", type=float, default=4.0)
    parser.add_argument("--jitter", type=float, default=1.0e-6)
    parser.add_argument("--syn-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--status-file", type=Path, default=OUTPUT_DIR / "live_progress.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        _write_status(
            Path(args.status_file),
            {
                "status": "failed",
                "phase": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "pid": os.getpid(),
                "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    main()
