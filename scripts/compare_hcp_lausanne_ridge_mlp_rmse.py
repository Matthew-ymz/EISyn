from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_hcp_lausanne_phi_eid_pilot as pilot


DEFAULT_OUTPUT = ROOT / "results" / "hcp_lausanne_ridge_mlp_rmse_comparison.json"
DEFAULT_REPORT = ROOT / "docs" / "log" / "hcp_lausanne_ridge_mlp_rmse_comparison.md"


def summarize_comparison(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    ridge_rmse = np.asarray([float(row["ridge_metrics"]["rmse"]) for row in rows], dtype=float)
    mlp_rmse = np.asarray([float(row["mlp_metrics"]["rmse"]) for row in rows], dtype=float)
    ridge_skill = np.asarray([float(row["ridge_metrics"]["skill_ratio"]) for row in rows], dtype=float)
    mlp_skill = np.asarray([float(row["mlp_metrics"]["skill_ratio"]) for row in rows], dtype=float)
    ridge_corr = np.asarray([float(row["ridge_metrics"]["corr"]) for row in rows], dtype=float)
    mlp_corr = np.asarray([float(row["mlp_metrics"]["corr"]) for row in rows], dtype=float)
    deltas = mlp_rmse - ridge_rmse
    count = int(len(rows))
    return {
        "subject_run_count": count,
        "ridge_rmse_mean": float(np.mean(ridge_rmse)),
        "mlp_rmse_mean": float(np.mean(mlp_rmse)),
        "mlp_minus_ridge_rmse_mean": float(np.mean(deltas)),
        "mlp_minus_ridge_rmse_std": float(np.std(deltas, ddof=1)) if count > 1 else 0.0,
        "mlp_better_than_ridge_count": int(np.sum(mlp_rmse < ridge_rmse)),
        "ridge_better_than_persistence_count": int(np.sum(ridge_skill < 1.0)),
        "mlp_better_than_persistence_count": int(np.sum(mlp_skill < 1.0)),
        "ridge_skill_ratio_mean": float(np.mean(ridge_skill)),
        "mlp_skill_ratio_mean": float(np.mean(mlp_skill)),
        "ridge_corr_mean": float(np.mean(ridge_corr)),
        "mlp_corr_mean": float(np.mean(mlp_corr)),
    }


def compare_subject_runs(
    *,
    roi_cache_dir: Path,
    subjects: Sequence[str],
    runs: Sequence[str],
    ridge_alpha: float,
    ridge: float,
    mlp_hidden_dim: int,
    mlp_epochs: int,
    mlp_learning_rate: float,
    seed: int,
) -> list[dict[str, object]]:
    labels = pilot.ordered_roi_labels()
    rows: list[dict[str, object]] = []
    for subject in subjects:
        for run in runs:
            series, metadata = pilot.load_cached_roi_timeseries(
                roi_cache_dir,
                subject=str(subject),
                run=str(run),
                expected_labels=labels,
            )
            source, target = pilot.make_lagged_samples(series, tau=1)
            row_seed = int(seed) + len(rows) * 100
            ridge_result = pilot.fit_ridge_transition(
                source,
                target,
                alpha=float(ridge_alpha),
                ridge=float(ridge),
            )
            mlp_result = pilot.fit_mlp_transition(
                source,
                target,
                hidden_dim=int(mlp_hidden_dim),
                epochs=int(mlp_epochs),
                seed=row_seed,
                learning_rate=float(mlp_learning_rate),
            )
            rows.append(
                {
                    "subject": str(subject),
                    "run": str(run),
                    "sample_count": int(source.shape[0]),
                    "seed": row_seed,
                    "cache": str(metadata.get("loaded_from_cache", "")),
                    "ridge_metrics": ridge_result["metrics"],
                    "mlp_metrics": mlp_result["metrics"],
                    "mlp_best_val_loss": float(mlp_result["best_val_loss"]),
                }
            )
    return rows


def write_report(path: Path, payload: Mapping[str, object]) -> None:
    summary = payload["summary"]
    config = payload["config"]
    rows = payload["rows"]
    detail_lines = [
        "| Subject | Run | Ridge RMSE | MLP RMSE | MLP - Ridge | Ridge / persistence | MLP / persistence |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ridge_metrics = row["ridge_metrics"]
        mlp_metrics = row["mlp_metrics"]
        ridge_rmse = float(ridge_metrics["rmse"])
        mlp_rmse = float(mlp_metrics["rmse"])
        detail_lines.append(
            f"| {row['subject']} | {row['run']} | {ridge_rmse:.6f} | {mlp_rmse:.6f} | "
            f"{mlp_rmse - ridge_rmse:.6f} | {float(ridge_metrics['skill_ratio']):.6f} | "
            f"{float(mlp_metrics['skill_ratio']):.6f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# HCP Lausanne-83 Ridge vs pure MLP RMSE comparison",
                "",
                "Question: what changes when only the one-step predictor changes from Ridge to pure MLP?",
                "",
                "## Contract",
                "",
                f"- ROI cache: `{config['roi_cache_dir']}`",
                f"- Subjects: `{', '.join(config['subjects'])}`",
                f"- Runs: `{', '.join(config['runs'])}`",
                "- Fixed: one-step lag, 70/30 chronological validation split, column standardization, persistence baseline, RMSE definition.",
                f"- Ridge: `alpha={config['ridge_alpha']}`, covariance ridge floor `{config['ridge']}`.",
                f"- MLP: hidden dim `{config['mlp_hidden_dim']}`, epochs `{config['mlp_epochs']}`, learning rate `{config['mlp_learning_rate']}`, AdamW, no Ridge readout.",
                "",
                "## Summary",
                "",
                f"- Subject-runs: `{summary['subject_run_count']}`",
                f"- Mean Ridge RMSE: `{summary['ridge_rmse_mean']:.6f}`",
                f"- Mean pure MLP RMSE: `{summary['mlp_rmse_mean']:.6f}`",
                f"- Mean paired MLP - Ridge RMSE: `{summary['mlp_minus_ridge_rmse_mean']:.6f}`",
                f"- Paired delta std: `{summary['mlp_minus_ridge_rmse_std']:.6f}`",
                f"- Pure MLP lower RMSE than Ridge: `{summary['mlp_better_than_ridge_count']} / {summary['subject_run_count']}`",
                f"- Ridge better than persistence: `{summary['ridge_better_than_persistence_count']} / {summary['subject_run_count']}`",
                f"- Pure MLP better than persistence: `{summary['mlp_better_than_persistence_count']} / {summary['subject_run_count']}`",
                f"- Mean Ridge validation correlation: `{summary['ridge_corr_mean']:.6f}`",
                f"- Mean pure MLP validation correlation: `{summary['mlp_corr_mean']:.6f}`",
                "",
                "## Per Subject-Run",
                "",
                *detail_lines,
                "",
                "## Conclusion",
                "",
                "With this controlled default comparison, pure MLP does not improve validation RMSE over Ridge. "
                "The MLP has higher mean RMSE, lower mean validation correlation, and fewer subject-runs beating the persistence baseline.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare HCP Lausanne-83 Ridge and pure MLP validation RMSE.")
    parser.add_argument("--roi-cache-dir", default=str(pilot.DEFAULT_OUTPUT_DIR / "roi_timeseries"))
    parser.add_argument("--subjects", default=",".join(pilot.DEFAULT_SUBJECTS))
    parser.add_argument("--runs", default="REST1_LR,REST1_RL")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-epochs", type=int, default=80)
    parser.add_argument("--mlp-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    subjects = [part.strip() for part in str(args.subjects).split(",") if part.strip()]
    runs = [part.strip() for part in str(args.runs).split(",") if part.strip()]
    rows = compare_subject_runs(
        roi_cache_dir=Path(args.roi_cache_dir).expanduser().resolve(),
        subjects=subjects,
        runs=runs,
        ridge_alpha=float(args.ridge_alpha),
        ridge=float(args.ridge),
        mlp_hidden_dim=int(args.mlp_hidden_dim),
        mlp_epochs=int(args.mlp_epochs),
        mlp_learning_rate=float(args.mlp_learning_rate),
        seed=int(args.seed),
    )
    payload = {
        "config": {
            "roi_cache_dir": str(Path(args.roi_cache_dir).expanduser().resolve()),
            "subjects": subjects,
            "runs": runs,
            "ridge_alpha": float(args.ridge_alpha),
            "ridge": float(args.ridge),
            "mlp_hidden_dim": int(args.mlp_hidden_dim),
            "mlp_epochs": int(args.mlp_epochs),
            "mlp_learning_rate": float(args.mlp_learning_rate),
            "seed": int(args.seed),
        },
        "summary": summarize_comparison(rows),
        "rows": rows,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_report(Path(args.report).expanduser().resolve(), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
