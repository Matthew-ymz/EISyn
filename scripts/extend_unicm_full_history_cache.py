from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.unicm_peid_syn_analysis import (  # noqa: E402
    HISTORY_LENGTH,
    MODE_NAMES,
    PREDICTION_LENGTH,
    load_unicm_model,
    overall_prediction_cache_path,
    predict_modeformer_all_modes_from_history,
    resolve_checkpoint_paths,
    sample_full_history_mode_inputs,
)


def sample_extension_history(
    *,
    base_n_samples: int,
    additional_n_samples: int,
    intervention_bound: float,
    sampling_seed: int,
) -> np.ndarray:
    if int(base_n_samples) < 1:
        raise ValueError("base_n_samples must be positive.")
    if int(additional_n_samples) < 1:
        raise ValueError("additional_n_samples must be positive.")
    total_n_samples = int(base_n_samples) + int(additional_n_samples)
    full_history = sample_full_history_mode_inputs(
        n_samples=total_n_samples,
        intervention_bound=float(intervention_bound),
        seed=int(sampling_seed),
    )
    return full_history[int(base_n_samples) :]


def _decode_metadata(raw_metadata: np.ndarray) -> dict[str, object]:
    try:
        value = raw_metadata.item()
        metadata = json.loads(str(value))
    except (AttributeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Prediction cache metadata is not valid JSON.") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Prediction cache metadata must be a JSON object.")
    return metadata


def _validate_metadata(metadata: dict[str, object], expected: dict[str, object], *, cache_path: Path) -> None:
    for key, expected_value in expected.items():
        if key not in metadata:
            raise ValueError(f"Prediction cache metadata is missing {key}: {cache_path}")
        if metadata[key] != expected_value:
            raise ValueError(
                f"Prediction cache metadata {key}={metadata[key]!r}, expected {expected_value!r}: {cache_path}"
            )


def load_validated_base_cache(
    cache_path: Path,
    *,
    seed: int,
    base_n_samples: int,
    sampling_seed: int,
    intervention_bound: float,
    start_month: int,
    device: str,
) -> tuple[np.ndarray, dict[str, object]]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Base prediction cache does not exist: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as payload:
        if "all_mode_targets" not in payload or "metadata" not in payload:
            raise ValueError(f"Base prediction cache must contain all_mode_targets and metadata: {cache_path}")
        targets = np.asarray(payload["all_mode_targets"], dtype=np.float32)
        metadata = _decode_metadata(payload["metadata"])

    expected_shape = (int(base_n_samples), PREDICTION_LENGTH, len(MODE_NAMES))
    if tuple(targets.shape) != expected_shape:
        raise ValueError(f"Base prediction cache has shape {tuple(targets.shape)}, expected {expected_shape}: {cache_path}")
    if not np.isfinite(targets).all():
        raise ValueError(f"Base prediction cache contains non-finite values: {cache_path}")

    expected_metadata = {
        "seed": int(seed),
        "n_samples": int(base_n_samples),
        "sampling_seed": int(sampling_seed),
        "intervention_bound": float(intervention_bound),
        "sampling_mode": "full_history_max_entropy",
        "history_shape": [HISTORY_LENGTH, len(MODE_NAMES)],
        "start_month": int(start_month),
        "device": str(device),
    }
    _validate_metadata(metadata, expected_metadata, cache_path=cache_path)
    return targets, metadata


def concatenate_prediction_cache(base_targets: np.ndarray, extension_targets: np.ndarray) -> np.ndarray:
    base = np.asarray(base_targets, dtype=np.float32)
    extension = np.asarray(extension_targets, dtype=np.float32)
    expected_tail = (PREDICTION_LENGTH, len(MODE_NAMES))
    for label, array in (("base_targets", base), ("extension_targets", extension)):
        if array.ndim != 3 or tuple(array.shape[1:]) != expected_tail:
            raise ValueError(f"{label} must have shape (n_samples, {PREDICTION_LENGTH}, {len(MODE_NAMES)}).")
        if not np.isfinite(array).all():
            raise ValueError(f"{label} contains non-finite values.")
    return np.concatenate([base, extension], axis=0)


def build_extended_cache_metadata(
    *,
    seed: int,
    n_samples: int,
    extended_from_n_samples: int,
    sampling_seed: int,
    intervention_bound: float,
    start_month: int,
    device: str,
) -> dict[str, object]:
    return {
        "seed": int(seed),
        "n_samples": int(n_samples),
        "sampling_seed": int(sampling_seed),
        "intervention_bound": float(intervention_bound),
        "sampling_mode": "full_history_max_entropy",
        "history_shape": [HISTORY_LENGTH, len(MODE_NAMES)],
        "start_month": int(start_month),
        "device": str(device),
        "extended_from_n_samples": int(extended_from_n_samples),
        "extension_strategy": "same_rng_continuation",
    }


def _cache_args(args: argparse.Namespace, *, n_samples: int) -> SimpleNamespace:
    return SimpleNamespace(
        n_samples=int(n_samples),
        sampling_seed=int(args.sampling_seed),
        intervention_bound=float(args.intervention_bound),
        start_month=int(args.start_month),
        device=str(args.device),
    )


def _write_cache(cache_path: Path, targets: np.ndarray, metadata: dict[str, object]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp.npz")
    np.savez(
        temporary_path,
        all_mode_targets=np.asarray(targets, dtype=np.float32),
        metadata=json.dumps(metadata, sort_keys=True),
    )
    temporary_path.replace(cache_path)


def run_extension(args: argparse.Namespace) -> dict[str, object]:
    import torch

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))

    seeds = [int(seed) for seed in args.seeds]
    base_n_samples = int(args.base_n_samples)
    total_n_samples = base_n_samples + int(args.additional_n_samples)
    base_cache_dir = Path(args.base_cache_dir)
    output_cache_dir = Path(args.output_cache_dir)
    output_cache_dir.mkdir(parents=True, exist_ok=True)

    base_cache_args = _cache_args(args, n_samples=base_n_samples)
    extended_cache_args = _cache_args(args, n_samples=total_n_samples)
    extension_history = sample_extension_history(
        base_n_samples=base_n_samples,
        additional_n_samples=int(args.additional_n_samples),
        intervention_bound=float(args.intervention_bound),
        sampling_seed=int(args.sampling_seed),
    )
    checkpoint_paths = resolve_checkpoint_paths(Path(args.checkpoint_root), seeds)

    cache_outputs: list[dict[str, object]] = []
    for seed in seeds:
        base_cache_path = overall_prediction_cache_path(base_cache_dir, seed=seed, args=base_cache_args)
        base_targets, _ = load_validated_base_cache(
            base_cache_path,
            seed=seed,
            base_n_samples=base_n_samples,
            sampling_seed=int(args.sampling_seed),
            intervention_bound=float(args.intervention_bound),
            start_month=int(args.start_month),
            device=str(args.device),
        )
        output_cache_path = overall_prediction_cache_path(output_cache_dir, seed=seed, args=extended_cache_args)
        metadata = build_extended_cache_metadata(
            seed=seed,
            n_samples=total_n_samples,
            extended_from_n_samples=base_n_samples,
            sampling_seed=int(args.sampling_seed),
            intervention_bound=float(args.intervention_bound),
            start_month=int(args.start_month),
            device=str(args.device),
        )

        if output_cache_path.exists():
            extended_targets, existing_metadata = load_validated_base_cache(
                output_cache_path,
                seed=seed,
                base_n_samples=total_n_samples,
                sampling_seed=int(args.sampling_seed),
                intervention_bound=float(args.intervention_bound),
                start_month=int(args.start_month),
                device=str(args.device),
            )
            _validate_metadata(
                existing_metadata,
                {
                    "extended_from_n_samples": base_n_samples,
                    "extension_strategy": "same_rng_continuation",
                },
                cache_path=output_cache_path,
            )
            if not np.array_equal(extended_targets[:base_n_samples], base_targets):
                raise ValueError(f"Extended cache prefix does not match the base cache: {output_cache_path}")
            status = "reused"
        else:
            print(f"[seed {seed}] forwarding {len(extension_history)} new samples", file=sys.stderr, flush=True)
            model = load_unicm_model(checkpoint_paths[seed], str(args.device))
            extension_targets = predict_modeformer_all_modes_from_history(
                model,
                extension_history,
                device=str(args.device),
                batch_size=int(args.batch_size),
                start_month=int(args.start_month),
            )
            extended_targets = concatenate_prediction_cache(base_targets, extension_targets)
            _write_cache(output_cache_path, extended_targets, metadata)
            del model
            status = "extended"

        cache_outputs.append(
            {
                "seed": seed,
                "status": status,
                "base_cache": str(base_cache_path),
                "output_cache": str(output_cache_path),
                "shape": list(extended_targets.shape),
                "prefix_exact": bool(np.array_equal(extended_targets[:base_n_samples], base_targets)),
            }
        )

    summary = {
        "base_n_samples": base_n_samples,
        "additional_n_samples": int(args.additional_n_samples),
        "n_samples": total_n_samples,
        "sampling_seed": int(args.sampling_seed),
        "intervention_bound": float(args.intervention_bound),
        "start_month": int(args.start_month),
        "device": str(args.device),
        "caches": cache_outputs,
    }
    summary_path = output_cache_dir.parent / "cache_extension_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extend UniCM full-history prediction caches using one RNG continuation.")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "data" / "UniCM-checkpoint" / "src" / "experiments",
    )
    parser.add_argument(
        "--base-cache-dir",
        type=Path,
        default=ROOT / "results" / "unicm_overall_ei_cpu_bound4_n4096" / "cache",
    )
    parser.add_argument(
        "--output-cache-dir",
        type=Path,
        default=ROOT / "results" / "unicm_overall_ei_cpu_bound4_n8192" / "cache",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--base-n-samples", type=int, default=4096)
    parser.add_argument("--additional-n-samples", type=int, default=4096)
    parser.add_argument("--sampling-seed", type=int, default=20260619)
    parser.add_argument("--intervention-bound", type=float, default=4.0)
    parser.add_argument("--start-month", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    outputs = run_extension(args)
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
