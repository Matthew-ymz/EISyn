from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _parse_int_csv(value: str) -> list[int]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        raise ValueError("Expected at least one integer in CSV argument.")
    return [int(item) for item in parts]


def _parse_str_csv(value: str) -> list[str]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return parts


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _jsonl_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _nonself_rows(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in edge_rows
        if str(row["source_station_id"]) != str(row["target_station_id"])
    ]


def _edge_values(rows: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in _nonself_rows(rows):
        key = f"{row['source_station_id']}->{row['target_station_id']}"
        values[key] = float(row["mean"])
    return values


def _edge_stats(rows: list[dict[str, Any]], *, eps_list: list[float]) -> dict[str, Any]:
    nonself = _nonself_rows(rows)
    values = np.asarray([float(row["mean"]) for row in nonself], dtype=float)
    if values.size == 0:
        return {
            "edge_count": 0,
            "neg_count": 0,
            "neg_ratio": 0.0,
            "neg_mass_mean": 0.0,
            "neg_abs_p90": 0.0,
            "neg_abs_max": 0.0,
            "pos_count": 0,
            "pos_p90": 0.0,
            "tolerance_neg_count": {str(eps): 0 for eps in eps_list},
            "tolerance_neg_ratio": {str(eps): 0.0 for eps in eps_list},
        }

    negatives = values[values < 0.0]
    positives = values[values > 0.0]
    neg_mass = np.maximum(-values, 0.0)
    tol_count = {str(eps): int((values < -float(eps)).sum()) for eps in eps_list}
    return {
        "edge_count": int(values.size),
        "neg_count": int(negatives.size),
        "neg_ratio": float((values < 0.0).mean()),
        "neg_mass_mean": float(neg_mass.mean()),
        "neg_abs_p90": float(np.quantile(np.abs(negatives), 0.9)) if negatives.size else 0.0,
        "neg_abs_max": float(np.abs(negatives).max()) if negatives.size else 0.0,
        "pos_count": int(positives.size),
        "pos_p90": float(np.quantile(positives, 0.9)) if positives.size else 0.0,
        "tolerance_neg_count": tol_count,
        "tolerance_neg_ratio": {
            key: float(value / values.size) for key, value in tol_count.items()
        },
    }


def _aggregate_sign_stability(
    edge_values_by_seed: dict[str, list[float]],
    *,
    eps_list: list[float],
) -> dict[str, Any]:
    if not edge_values_by_seed:
        return {
            "edge_count": 0,
            "always_positive_count": 0,
            "always_negative_count": 0,
            "mixed_sign_count": 0,
            "robust_positive_count": {str(eps): 0 for eps in eps_list},
        }

    series = np.asarray(list(edge_values_by_seed.values()), dtype=float)
    always_positive = int((series > 0.0).all(axis=1).sum())
    always_negative = int((series < 0.0).all(axis=1).sum())
    mixed_sign = int(series.shape[0] - always_positive - always_negative)
    robust_positive = {
        str(eps): int((series > float(eps)).all(axis=1).sum()) for eps in eps_list
    }
    return {
        "edge_count": int(series.shape[0]),
        "always_positive_count": always_positive,
        "always_negative_count": always_negative,
        "mixed_sign_count": mixed_sign,
        "robust_positive_count": robust_positive,
    }


def _metric_mean_std(records: list[dict[str, Any]], key_path: list[str]) -> tuple[float, float]:
    values = []
    for row in records:
        value: Any = row
        for key in key_path:
            value = value[key]
        values.append(float(value))
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std())


def run_scan(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    repo_root = Path(args.repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from yrd.shanghai_notebook import (
        build_default_shanghai_one_step_config,
        build_prediction_tables,
        build_transport_map_global_causal_summary,
        compute_training_input_center,
        prepare_shanghai_one_step_bundle,
        resolve_nonnegative_lower_bounds_by_feature,
        resolve_variable_box_size_by_feature,
        run_or_load_one_step_predictions,
        sample_uniform_box_inputs,
    )

    output_root = (repo_root / args.output_dir).resolve()
    batch_tag = args.batch_tag or _now_tag()
    batch_dir = output_root / f"batch_{batch_tag}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    edge_dir = batch_dir / "edge_values"
    edge_dir.mkdir(parents=True, exist_ok=True)

    sample_counts = _parse_int_csv(args.sample_counts)
    seeds = _parse_int_csv(args.seeds)
    eps_list = [float(value) for value in args.epsilons.split(",") if value.strip()]
    nonnegative_variables = tuple(_parse_str_csv(args.nonnegative_variables))
    box_size_profile_by_variable: dict[str, float] | None = None
    profile_name = str(args.box_size_profile_name).strip()
    if args.box_size_profile_json:
        profile_payload = json.loads(Path(args.box_size_profile_json).read_text())
        box_size_profile_by_variable = {str(k): float(v) for k, v in dict(profile_payload).items()}
    elif args.box_size_profiles_json:
        profile_table = json.loads(Path(args.box_size_profiles_json).read_text())
        if not profile_name:
            raise ValueError("--box-size-profile-name is required when --box-size-profiles-json is provided.")
        if profile_name not in profile_table:
            available = ", ".join(sorted(profile_table.keys()))
            raise ValueError(f"Unknown profile '{profile_name}'. Available: {available}")
        profile_payload = dict(profile_table[profile_name])
        box_size_profile_by_variable = {str(k): float(v) for k, v in dict(profile_payload).items()}

    cfg = replace(
        build_default_shanghai_one_step_config(repo_root, test_mode=False),
        box_size=float(args.box_size),
        causal_graph_box_size_by_variable=box_size_profile_by_variable,
        causal_graph_nonnegative_variables=nonnegative_variables,
    )
    bundle = prepare_shanghai_one_step_bundle(
        cfg=cfg,
        city_en=args.city_en,
        run_tag=args.run_tag,
        use_smoke=False,
        coupling_sample_count=max(sample_counts),
        graph_edges_per_target=int(args.graph_edges_per_target),
        causal_graph_edge_keep_fraction=float(args.causal_graph_edge_keep_fraction),
        causal_graph_arrow_mutation_scale=float(args.causal_graph_arrow_mutation_scale),
        causal_graph_arrow_shrink_target=float(args.causal_graph_arrow_shrink_target),
        causal_graph_variable=args.causal_graph_variable,
    )
    predictions = run_or_load_one_step_predictions(bundle, force_retrain=bool(args.force_retrain))
    metrics_bundle = build_prediction_tables(bundle, predictions)

    model = predictions["joint_model"]
    model.eval()
    center = compute_training_input_center(bundle["x_train"])
    station_ids = list(bundle["station_ids"])
    station_target_indices = dict(bundle["station_target_indices"])
    o3_feature_index = int(cfg.input_variables.index("O3"))
    pm25_feature_index = int(cfg.input_variables.index("PM2.5"))
    if cfg.causal_graph_box_size_by_variable is None:
        sampling_box_size: float | np.ndarray = float(args.box_size)
        sampling_lower_bounds: np.ndarray | None = None
    else:
        sampling_box_size = resolve_variable_box_size_by_feature(
            input_variables=cfg.input_variables,
            box_size_by_variable=cfg.causal_graph_box_size_by_variable,
            n_stations=bundle["sample_bundle"]["n_stations"],
        )
        sampling_lower_bounds = resolve_nonnegative_lower_bounds_by_feature(
            input_variables=cfg.input_variables,
            stats=bundle["stats"],
            nonnegative_variables=tuple(cfg.causal_graph_nonnegative_variables),
            n_stations=bundle["sample_bundle"]["n_stations"],
        )

    run_records: list[dict[str, Any]] = []
    edge_values_grouped: dict[int, dict[str, dict[str, list[float]]]] = {
        count: {"pairwise": {}, "conditional_synergy": {}} for count in sample_counts
    }

    for sample_count in sample_counts:
        for seed in seeds:
            started = time.perf_counter()
            synthetic_inputs = sample_uniform_box_inputs(
                center=center,
                box_size=sampling_box_size,
                sample_count=int(sample_count),
                seed=int(seed),
                lower_bounds=sampling_lower_bounds,
            )
            with torch.no_grad():
                predicted_next = (
                    model(torch.from_numpy(synthetic_inputs).to(dtype=torch.float32))[1]
                    .detach()
                    .cpu()
                    .numpy()
                )
            predicted_next_o3 = np.stack(
                [predicted_next[:, station_target_indices[station_id][0]] for station_id in station_ids],
                axis=1,
            )
            summary = build_transport_map_global_causal_summary(
                source_samples=synthetic_inputs,
                predicted_next_o3=predicted_next_o3,
                station_ids=station_ids,
                o3_feature_index=o3_feature_index,
                pm25_feature_index=pm25_feature_index,
            )["1h"]

            pair_rows = list(summary["pairwise_edges"])
            synergy_rows = list(summary["conditional_synergy_edges"])
            pair_stats = _edge_stats(pair_rows, eps_list=eps_list)
            synergy_stats = _edge_stats(synergy_rows, eps_list=eps_list)
            pair_values = _edge_values(pair_rows)
            synergy_values = _edge_values(synergy_rows)

            for edge_key, value in pair_values.items():
                edge_values_grouped[sample_count]["pairwise"].setdefault(edge_key, []).append(value)
            for edge_key, value in synergy_values.items():
                edge_values_grouped[sample_count]["conditional_synergy"].setdefault(edge_key, []).append(value)

            elapsed = float(time.perf_counter() - started)
            run_name = f"tm_stability_M{sample_count}_seed{seed}"
            edge_path = edge_dir / f"{run_name}.json"
            _json_dump(
                edge_path,
                {
                    "run_name": run_name,
                    "sample_count": int(sample_count),
                    "seed": int(seed),
                    "pairwise_nonself": pair_values,
                    "conditional_synergy_nonself": synergy_values,
                },
            )
            run_records.append(
                {
                    "run_name": run_name,
                    "sample_count": int(sample_count),
                    "seed": int(seed),
                    "box_size": float(args.box_size),
                    "box_size_profile_name": profile_name if profile_name else None,
                    "box_size_profile_by_variable": (
                        {k: float(v) for k, v in cfg.causal_graph_box_size_by_variable.items()}
                        if cfg.causal_graph_box_size_by_variable is not None
                        else None
                    ),
                    "nonnegative_variables": list(cfg.causal_graph_nonnegative_variables),
                    "status": "completed",
                    "elapsed_seconds": elapsed,
                    "pairwise": pair_stats,
                    "conditional_synergy": synergy_stats,
                    "edge_values_path": str(edge_path.relative_to(repo_root)),
                }
            )

    aggregate_by_sample_count: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    for sample_count in sample_counts:
        rows = [row for row in run_records if int(row["sample_count"]) == int(sample_count)]
        pair_neg_ratio_mean, pair_neg_ratio_std = _metric_mean_std(rows, ["pairwise", "neg_ratio"])
        pair_neg_mass_mean, pair_neg_mass_std = _metric_mean_std(rows, ["pairwise", "neg_mass_mean"])
        synergy_neg_ratio_mean, synergy_neg_ratio_std = _metric_mean_std(
            rows,
            ["conditional_synergy", "neg_ratio"],
        )
        synergy_neg_mass_mean, synergy_neg_mass_std = _metric_mean_std(
            rows,
            ["conditional_synergy", "neg_mass_mean"],
        )
        elapsed_mean, elapsed_std = _metric_mean_std(rows, ["elapsed_seconds"])
        pair_sign = _aggregate_sign_stability(
            edge_values_grouped[sample_count]["pairwise"],
            eps_list=eps_list,
        )
        synergy_sign = _aggregate_sign_stability(
            edge_values_grouped[sample_count]["conditional_synergy"],
            eps_list=eps_list,
        )
        aggregate = {
            "sample_count": int(sample_count),
            "run_count": len(rows),
            "pairwise_neg_ratio_mean": pair_neg_ratio_mean,
            "pairwise_neg_ratio_std": pair_neg_ratio_std,
            "pairwise_neg_mass_mean": pair_neg_mass_mean,
            "pairwise_neg_mass_std": pair_neg_mass_std,
            "conditional_synergy_neg_ratio_mean": synergy_neg_ratio_mean,
            "conditional_synergy_neg_ratio_std": synergy_neg_ratio_std,
            "conditional_synergy_neg_mass_mean": synergy_neg_mass_mean,
            "conditional_synergy_neg_mass_std": synergy_neg_mass_std,
            "elapsed_seconds_mean": elapsed_mean,
            "elapsed_seconds_std": elapsed_std,
            "pairwise_sign_stability": pair_sign,
            "conditional_synergy_sign_stability": synergy_sign,
        }
        aggregate_by_sample_count[str(sample_count)] = aggregate
        table_rows.append(aggregate)

    sorted_rows = sorted(
        table_rows,
        key=lambda row: (
            row["conditional_synergy_neg_ratio_mean"],
            row["pairwise_neg_ratio_mean"],
            row["elapsed_seconds_mean"],
        ),
    )
    recommendation = sorted_rows[0] if sorted_rows else None

    overall_metrics = metrics_bundle["metrics_overall_df"].to_dict(orient="records")
    baseline_metrics = {
        row["model"]: row
        for row in overall_metrics
        if str(row.get("horizon")) == "1h" and str(row.get("scope")) == "overall"
    }

    payload = {
        "batch_tag": batch_tag,
        "repo_root": str(repo_root),
        "run_tag": args.run_tag,
        "city_en": args.city_en,
        "fixed_model": {
            "checkpoint_path": str(Path(bundle["artifact_paths"]["checkpoint"]).resolve()),
            "force_retrain": bool(args.force_retrain),
            "box_size": float(args.box_size),
            "box_size_profile_name": profile_name if profile_name else None,
            "box_size_profile_by_variable": (
                {k: float(v) for k, v in cfg.causal_graph_box_size_by_variable.items()}
                if cfg.causal_graph_box_size_by_variable is not None
                else None
            ),
            "nonnegative_variables": list(cfg.causal_graph_nonnegative_variables),
        },
        "search_space": {
            "sample_counts": sample_counts,
            "seeds": seeds,
            "epsilons": eps_list,
        },
        "baseline_forecast_metrics_1h_overall": baseline_metrics,
        "run_records": run_records,
        "aggregate_by_sample_count": aggregate_by_sample_count,
        "recommended_sample_count": recommendation["sample_count"] if recommendation is not None else None,
        "recommendation_row": recommendation,
        "artifacts": {
            "run_records_jsonl": str((batch_dir / "run_records.jsonl").relative_to(repo_root)),
            "run_records_json": str((batch_dir / "run_records.json").relative_to(repo_root)),
            "aggregate_json": str((batch_dir / "aggregate_summary.json").relative_to(repo_root)),
            "aggregate_csv": str((batch_dir / "aggregate_summary.csv").relative_to(repo_root)),
        },
    }

    run_records_path = batch_dir / "run_records.jsonl"
    _jsonl_dump(run_records_path, run_records)
    _json_dump(batch_dir / "run_records.json", {"run_records": run_records})
    _json_dump(batch_dir / "aggregate_summary.json", payload)

    csv_rows = [
        {
            "sample_count": row["sample_count"],
            "run_count": row["run_count"],
            "pairwise_neg_ratio_mean": row["pairwise_neg_ratio_mean"],
            "pairwise_neg_ratio_std": row["pairwise_neg_ratio_std"],
            "pairwise_neg_mass_mean": row["pairwise_neg_mass_mean"],
            "pairwise_neg_mass_std": row["pairwise_neg_mass_std"],
            "conditional_synergy_neg_ratio_mean": row["conditional_synergy_neg_ratio_mean"],
            "conditional_synergy_neg_ratio_std": row["conditional_synergy_neg_ratio_std"],
            "conditional_synergy_neg_mass_mean": row["conditional_synergy_neg_mass_mean"],
            "conditional_synergy_neg_mass_std": row["conditional_synergy_neg_mass_std"],
            "elapsed_seconds_mean": row["elapsed_seconds_mean"],
            "elapsed_seconds_std": row["elapsed_seconds_std"],
            "pairwise_always_positive_count": row["pairwise_sign_stability"]["always_positive_count"],
            "pairwise_mixed_sign_count": row["pairwise_sign_stability"]["mixed_sign_count"],
            "synergy_always_positive_count": row["conditional_synergy_sign_stability"]["always_positive_count"],
            "synergy_mixed_sign_count": row["conditional_synergy_sign_stability"]["mixed_sign_count"],
        }
        for row in sorted_rows
    ]
    if csv_rows:
        header = list(csv_rows[0].keys())
        csv_lines = [",".join(header)]
        for row in csv_rows:
            csv_lines.append(",".join(str(row[key]) for key in header))
        (batch_dir / "aggregate_summary.csv").write_text("\n".join(csv_lines) + "\n")
    else:
        (batch_dir / "aggregate_summary.csv").write_text("")

    _json_dump(
        batch_dir / "batch_manifest.json",
        {
            "created_at": datetime.now().isoformat(),
            "command": " ".join(os.sys.argv),
            "batch_tag": batch_tag,
            "artifacts": payload["artifacts"],
        },
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed-model TM stability scan over sample_count x seed grid.",
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--city-en", default="shanghai")
    parser.add_argument(
        "--run-tag",
        default="shanghai_one_step_o3_station_graph_tm_causal_graph",
    )
    parser.add_argument("--box-size", type=float, default=20.0)
    parser.add_argument("--sample-counts", default="2048,4096,8192")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--epsilons", default="0,1e-5,5e-5,1e-4")
    parser.add_argument("--box-size-profile-json", default="")
    parser.add_argument("--box-size-profiles-json", default="")
    parser.add_argument("--box-size-profile-name", default="")
    parser.add_argument(
        "--nonnegative-variables",
        default="O3,PM2.5,t2m,d2m,sp,tp,blh,msdwswrf",
    )
    parser.add_argument(
        "--output-dir",
        default="exp/cache/yrd_coupling/tm_stability_phase1",
    )
    parser.add_argument("--batch-tag", default="")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--graph-edges-per-target", type=int, default=2)
    parser.add_argument("--causal-graph-edge-keep-fraction", type=float, default=1.0)
    parser.add_argument("--causal-graph-arrow-mutation-scale", type=float, default=24.0)
    parser.add_argument("--causal-graph-arrow-shrink-target", type=float, default=6.0)
    parser.add_argument("--causal-graph-variable", default="O3")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_scan(args)
    recommendation = summary.get("recommendation_row")
    if recommendation is None:
        print("No runs were executed.")
        return 1
    print(
        "Completed TM stability scan:",
        f"recommended_M={recommendation['sample_count']}",
        f"synergy_neg_ratio_mean={recommendation['conditional_synergy_neg_ratio_mean']:.6f}",
        f"pair_neg_ratio_mean={recommendation['pairwise_neg_ratio_mean']:.6f}",
    )
    print("Summary artifact:", summary["artifacts"]["aggregate_json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
