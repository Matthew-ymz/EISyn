#!/usr/bin/env python3
"""Run the continuous TM follow-up experiment for the O3 H1 PEID manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "results" / "o3_h1_peid"
MANIFEST_PATH = RESULTS_DIR / "figure_manifest.json"
TM_RESULTS_DIR = RESULTS_DIR / "tm_continuous_peid"
DOCS_LOG_DIR = ROOT / "docs" / "log"


@dataclass(frozen=True)
class ContinuousTmPeidConfig:
    sample_count: int = 8192
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    quantile_low: float = 0.05
    quantile_high: float = 0.95
    target_noise_std: float = 0.25
    jitter: float = 1e-6


def _source_to_feature() -> dict[str, str]:
    from scripts.validate_o3_h1_peid import SOURCE_TO_FEATURE

    return dict(SOURCE_TO_FEATURE)


def _feature_columns() -> list[str]:
    from scripts.validate_o3_h1_peid import FEATURE_COLUMNS

    return list(FEATURE_COLUMNS)


def _fit_station_mean_poly_model(frame: pd.DataFrame):
    from scripts.validate_o3_h1_peid import build_nox_voc_poly_regressor

    train = frame[frame["split"] == "train"]
    model = build_nox_voc_poly_regressor()
    model.fit(train[_feature_columns()], train["O3_peak"])
    return model


def _standardize(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    scale = values.std(axis=0, ddof=1)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (values - values.mean(axis=0)) / scale


def _sample_feature_values(series: pd.Series, sample_count: int, rng: np.random.Generator) -> np.ndarray:
    clean = series.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError(f"Cannot sample from empty feature column {series.name!r}.")
    low = float(clean.quantile(0.05))
    high = float(clean.quantile(0.95))
    if low == high:
        return np.full(sample_count, low, dtype=float)
    if low > 0.0 and high > 0.0 and clean.max() / max(clean.min(), 1e-12) > 5.0:
        return np.exp(rng.uniform(np.log(low), np.log(high), size=sample_count))
    return rng.uniform(low, high, size=sample_count)


def build_intervention_samples(
    frame: pd.DataFrame,
    *,
    source_set: Sequence[str],
    sample_count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample source variables from empirical station-mean support without discretization."""

    mapping = _source_to_feature()
    rows: dict[str, np.ndarray] = {}
    for source in source_set:
        if source not in mapping:
            raise ValueError(f"Unknown source label {source!r}.")
        rows[source] = _sample_feature_values(frame[mapping[source]], sample_count, rng)
    return pd.DataFrame(rows)


def build_model_input_frame(
    base_frame: pd.DataFrame,
    source_samples: pd.DataFrame,
) -> pd.DataFrame:
    """Create model inputs with sampled source columns and median controls."""

    mapping = _source_to_feature()
    feature_cols = _feature_columns()
    medians = base_frame[feature_cols].median(numeric_only=True)
    model_frame = pd.DataFrame(
        np.tile(medians.to_numpy(dtype=float), (len(source_samples), 1)),
        columns=feature_cols,
    )
    for source in source_samples.columns:
        model_frame[mapping[source]] = source_samples[source].to_numpy(dtype=float)
    return model_frame


def estimate_continuous_source_set_information(
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    *,
    source_names: Sequence[str],
    jitter: float = 1e-6,
) -> dict[str, object]:
    """Estimate joint and individual continuous information with TM density MI."""

    from exp.TM.transport_map_density import estimate_mutual_information_transport_map

    sources = _standardize(np.asarray(source_samples, dtype=float))
    target = _standardize(np.asarray(target_samples, dtype=float).reshape(-1, 1))
    if sources.shape[1] != len(source_names):
        raise ValueError("source_names length must match source_samples columns.")
    joint = estimate_mutual_information_transport_map(sources, target, jitter=jitter)
    individual: dict[str, float] = {}
    for idx, source_name in enumerate(source_names):
        summary = estimate_mutual_information_transport_map(sources[:, [idx]], target, jitter=jitter)
        individual[str(source_name)] = max(0.0, float(summary["mi_hat"]))
    joint_mi = max(0.0, float(joint["mi_hat"]))
    individual_sum = float(sum(individual.values()))
    best_individual = max(individual.values()) if individual else 0.0
    return {
        "joint_mi_nats": joint_mi,
        "individual_mi_nats": individual,
        "individual_mi_sum_nats": individual_sum,
        "best_individual_mi_nats": float(best_individual),
        "gain_over_best_individual_nats": float(joint_mi - best_individual),
        "net_synergy_nats": float(joint_mi - individual_sum),
    }


def run_seed_batch(
    frame: pd.DataFrame,
    model,
    source_sets: Sequence[Sequence[str]],
    *,
    config: ContinuousTmPeidConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in config.seeds:
        rng = np.random.default_rng(seed)
        for source_set in source_sets:
            source_samples = build_intervention_samples(
                frame,
                source_set=source_set,
                sample_count=config.sample_count,
                rng=rng,
            )
            model_inputs = build_model_input_frame(frame, source_samples)
            target = model.predict(model_inputs[_feature_columns()])
            if config.target_noise_std > 0.0:
                target = target + rng.normal(scale=config.target_noise_std, size=len(target))
            info = estimate_continuous_source_set_information(
                source_samples.to_numpy(dtype=float),
                target,
                source_names=source_set,
                jitter=config.jitter,
            )
            rows.append(
                {
                    "seed": int(seed),
                    "sources": "+".join(source_set),
                    "source_order": int(len(source_set)),
                    "sample_count": int(config.sample_count),
                    "target_noise_std": float(config.target_noise_std),
                    "joint_mi_nats": info["joint_mi_nats"],
                    "individual_mi_sum_nats": info["individual_mi_sum_nats"],
                    "best_individual_mi_nats": info["best_individual_mi_nats"],
                    "gain_over_best_individual_nats": info["gain_over_best_individual_nats"],
                    "net_synergy_nats": info["net_synergy_nats"],
                    "individual_mi_json": json.dumps(info["individual_mi_nats"], sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def summarize_seed_runs(runs: pd.DataFrame) -> pd.DataFrame:
    if "source_order" not in runs.columns:
        runs = runs.copy()
        runs["source_order"] = runs["sources"].astype(str).str.split("+", regex=False).str.len()
    if "seed" not in runs.columns:
        runs = runs.copy()
        runs["seed"] = np.arange(len(runs), dtype=int)
    if "sample_count" not in runs.columns:
        runs = runs.copy()
        runs["sample_count"] = np.nan
    grouped = runs.groupby(["sources", "source_order"], as_index=False)
    summary = grouped.agg(
        n_seeds=("seed", "nunique"),
        sample_count=("sample_count", "first"),
        joint_mi_mean_nats=("joint_mi_nats", "mean"),
        joint_mi_sd_nats=("joint_mi_nats", "std"),
        gain_over_best_individual_mean_nats=("gain_over_best_individual_nats", "mean"),
        gain_over_best_individual_sd_nats=("gain_over_best_individual_nats", "std"),
        net_synergy_mean_nats=("net_synergy_nats", "mean"),
        net_synergy_sd_nats=("net_synergy_nats", "std"),
    )
    return summary.sort_values(
        ["gain_over_best_individual_mean_nats", "joint_mi_mean_nats"],
        ascending=[False, False],
    ).reset_index(drop=True)


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
        }
    )
    view = summary.sort_values("gain_over_best_individual_mean_nats", ascending=True)
    y = np.arange(len(view))
    height = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.barh(y - height / 2, view["joint_mi_mean_nats"], height=height, color="#4C78A8", label="Joint MI")
    ax.barh(
        y + height / 2,
        view["gain_over_best_individual_mean_nats"],
        height=height,
        color="#D17B0F",
        label="Gain over best individual",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(view["sources"])
    ax.set_xlabel("Information estimate (nats)")
    ax.set_ylabel("Source set")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(out_dir / "tm_continuous_peid_summary.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / "tm_continuous_peid_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def _source_sets_from_manifest(manifest: dict[str, object]) -> list[list[str]]:
    plan = manifest["tm_followup_plan"]
    assert isinstance(plan, dict)
    source_sets = [list(plan["primary_source_set"])]
    source_sets.extend([list(item) for item in plan["comparison_source_sets"]])
    unique: list[list[str]] = []
    seen: set[str] = set()
    for source_set in source_sets:
        key = "+".join(source_set)
        if key not in seen:
            seen.add(key)
            unique.append(source_set)
    return unique


def _write_notes(summary: pd.DataFrame, manifest: dict[str, object], out_dir: Path) -> None:
    top = summary.iloc[0]
    primary = "+".join(manifest["tm_followup_plan"]["primary_source_set"])
    primary_rows = summary[summary["sources"] == primary]
    lines = [
        "# Continuous TM PEID Follow-up",
        "",
        "This follow-up implements the `tm_followup_plan` from `figure_manifest.json` without discretizing sources.",
        "The source model is the station-mean `poly_nox_voc` response surface; failed RF/MLP controls remain excluded.",
        "",
        "## Summary",
        "",
        f"- Top source set by gain over best individual: {top['sources']} ({top['gain_over_best_individual_mean_nats']:.4f} nats).",
    ]
    if not primary_rows.empty:
        row = primary_rows.iloc[0]
        lines.append(
            f"- Primary NOx+VOC source set: joint MI={row['joint_mi_mean_nats']:.4f} nats, "
            f"gain={row['gain_over_best_individual_mean_nats']:.4f} nats, "
            f"net synergy={row['net_synergy_mean_nats']:.4f} nats."
        )
    lines.extend(
        [
            "- Estimator: affine triangular transport-map density MI on standardized continuous samples.",
            "- Sampling: empirical station-mean 5-95% support with five seed replicates.",
            "",
            "## Caveat",
            "",
            "The station-mean polynomial source model uses NOx and VOC by construction; meteorological comparison sets are negative controls for this follow-up.",
        ]
    )
    (out_dir / "tm_continuous_peid_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_run_history(summary: pd.DataFrame, config: ContinuousTmPeidConfig, out_dir: Path) -> None:
    DOCS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    top = summary.iloc[0]
    payload = {
        "stage": "o3_h1_tm_continuous_peid_followup",
        "command": "python3 scripts/run_o3_h1_tm_peid.py",
        "sample_count": config.sample_count,
        "seeds": list(config.seeds),
        "top_sources": top["sources"],
        "top_gain_over_best_individual_mean_nats": float(top["gain_over_best_individual_mean_nats"]),
        "artifacts_dir": str(out_dir),
    }
    with (DOCS_LOG_DIR / "run_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    leaderboard = DOCS_LOG_DIR / "leaderboard.md"
    with leaderboard.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n"
            "## O3 H1 Continuous TM PEID Follow-up\n\n"
            f"- top_sources: `{top['sources']}`\n"
            f"- gain_over_best_individual_mean_nats: `{top['gain_over_best_individual_mean_nats']:.6f}`\n"
            f"- artifacts: `{out_dir}`\n"
        )


def run_followup(
    *,
    manifest_path: Path = MANIFEST_PATH,
    config: ContinuousTmPeidConfig = ContinuousTmPeidConfig(),
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    frame = pd.read_csv(artifacts["station_mean_feature_table"])
    model = _fit_station_mean_poly_model(frame)
    source_sets = _source_sets_from_manifest(manifest)

    TM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = run_seed_batch(frame, model, source_sets, config=config)
    summary = summarize_seed_runs(runs)
    runs_path = TM_RESULTS_DIR / "tm_continuous_peid_seed_runs.csv"
    summary_path = TM_RESULTS_DIR / "tm_continuous_peid_summary.csv"
    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, TM_RESULTS_DIR)
    _write_notes(summary, manifest, TM_RESULTS_DIR)

    followup_manifest = {
        "config": asdict(config),
        "source_sets": source_sets,
        "source_model": manifest["tm_followup_plan"]["source_model"],
        "target": manifest["tm_followup_plan"]["target"],
        "artifacts": {
            "seed_runs": str(runs_path),
            "summary": str(summary_path),
            "summary_png": str(TM_RESULTS_DIR / "tm_continuous_peid_summary.png"),
            "summary_pdf": str(TM_RESULTS_DIR / "tm_continuous_peid_summary.pdf"),
            "notes": str(TM_RESULTS_DIR / "tm_continuous_peid_notes.md"),
        },
    }
    followup_manifest_path = TM_RESULTS_DIR / "tm_continuous_peid_manifest.json"
    followup_manifest["artifacts"]["manifest"] = str(followup_manifest_path)
    followup_manifest_path.write_text(json.dumps(followup_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest["tm_followup_results"] = followup_manifest
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _append_run_history(summary, config, TM_RESULTS_DIR)
    return followup_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run continuous TM PEID follow-up for O3 H1.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--sample-count", type=int, default=ContinuousTmPeidConfig.sample_count)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in ContinuousTmPeidConfig.seeds))
    parser.add_argument("--target-noise-std", type=float, default=ContinuousTmPeidConfig.target_noise_std)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    config = ContinuousTmPeidConfig(
        sample_count=args.sample_count,
        seeds=seeds,
        target_noise_std=args.target_noise_std,
    )
    result = run_followup(manifest_path=args.manifest, config=config)
    print(json.dumps(result["artifacts"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
