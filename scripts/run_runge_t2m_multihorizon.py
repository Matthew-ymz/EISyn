#!/usr/bin/env python3
"""Run Runge-style monthly 2m-air-temperature components across horizons."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from scripts.reproduce_runge2015_gateways import (
    compute_sem_effects,
    dependency_versions,
    detrend_time_axis,
    discover_causal_edges,
    ensure_causal_backend_available,
    fit_projected_varimax_components,
    save_causal_network_figure,
    save_component_map_figure,
    save_ranking_figure,
)


DEFAULT_INPUT = Path("data/ncep_reanalysis_runge_validation/air.2m.mon.mean.nc")
RESULT_SUBDIR = Path("results/runge_t2m_monthly")
FIG_SUBDIR = Path("fig/runge_t2m_monthly")

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


@dataclass(frozen=True)
class T2MRungeConfig:
    input_netcdf: Path = DEFAULT_INPUT
    variable: str = "air"
    output_dir: Path = Path(".")
    start_date: str | None = None
    end_date: str | None = None
    n_components: int = 60
    max_lag: int = 4
    pc_alpha: float = 0.001
    link_density: float = 0.2
    seed: int = 42
    causal_backend: str = "tigramite"


@dataclass(frozen=True)
class PreprocessArtifacts:
    result_dir: Path
    fig_dir: Path
    component_scores: Path
    linear_coefficients: Path
    component_maps: Path
    manifest: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    root_candidate = _repo_root() / candidate
    return root_candidate if root_candidate.exists() else candidate.resolve()


def parse_horizons(text: str) -> list[int]:
    horizons: set[int] = set()
    for raw_part in str(text).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                raise ValueError("horizon ranges must be ascending.")
            horizons.update(range(start, end + 1))
        else:
            horizons.add(int(part))
    if not horizons or min(horizons) < 1:
        raise ValueError("horizons must contain positive month intervals.")
    return sorted(horizons)


def standardize_monthly_anomalies(field: xr.DataArray) -> xr.DataArray:
    month = field["time"].dt.month
    counts = field.groupby(month).count("time")
    if int(counts.max()) <= 1:
        anomaly = field - field.mean("time")
        scale = field.std("time").where(lambda value: value > 0.0, 1.0)
        return (anomaly / scale).fillna(0.0)
    climatology = field.groupby(month).mean("time")
    anomaly = field.groupby(month) - climatology
    scale = anomaly.groupby(month).std("time")
    scale = scale.where(np.isfinite(scale) & (scale > 0.0), 1.0)
    return (anomaly.groupby(month) / scale).fillna(0.0)


def load_monthly_field(config: T2MRungeConfig) -> xr.DataArray:
    path = _resolve_path(config.input_netcdf)
    with xr.open_dataset(path) as ds:
        if config.variable not in ds:
            raise ValueError(f"{path} does not contain variable {config.variable!r}.")
        field = ds[config.variable].load()
    if "time" not in field.dims or "lat" not in field.dims or "lon" not in field.dims:
        raise ValueError("input variable must have time, lat, and lon dimensions.")
    field = field.sortby("time")
    if config.start_date or config.end_date:
        field = field.sel(time=slice(config.start_date, config.end_date))
    if int(field.sizes["time"]) < max(12, int(config.n_components)):
        raise ValueError("monthly field has too few time steps for the requested components.")
    return field


def _write_markdown_table(frame: pd.DataFrame, columns: Sequence[str], *, rows: int = 10) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.loc[:, [column for column in columns if column in frame.columns]].head(rows)
    return view.to_markdown(index=False, floatfmt=".6g")


def _write_preprocess_summary(path: Path, manifest: dict[str, object], gateway: pd.DataFrame, mediator: pd.DataFrame) -> None:
    lines = [
        "# 2m Air Temperature Monthly Runge Preprocessing",
        "",
        "This run uses monthly near-surface 2m air temperature, not the original daily SLP-to-weekly Runge pipeline.",
        "",
        "## Configuration",
        "",
        f"- Variable: `{manifest['variable']}`",
        f"- Frequency: `{manifest['frequency']}`",
        f"- Time range: `{manifest['time_start']}` to `{manifest['time_end']}`",
        f"- Components: `{manifest['n_components']}`",
        f"- Monthly samples: `{manifest['n_monthly_samples']}`",
        f"- Causal backend: `{manifest['config']['causal_backend']}`",
        "",
        "## Top Linear Gateways",
        "",
        _write_markdown_table(gateway, ["component", "ace", "acs", "direct_out_strength", "direct_in_strength"]),
        "",
        "## Top Linear Mediators",
        "",
        _write_markdown_table(mediator, ["component", "amce", "mediated_fraction"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_preprocessing(config: T2MRungeConfig) -> PreprocessArtifacts:
    ensure_causal_backend_available(config.causal_backend)  # type: ignore[arg-type]
    result_dir = Path(config.output_dir) / RESULT_SUBDIR
    fig_dir = Path(config.output_dir) / FIG_SUBDIR
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    field = load_monthly_field(config)
    standardized = detrend_time_axis(standardize_monthly_anomalies(field))
    scores, component_maps, explained = fit_projected_varimax_components(
        standardized,
        standardized,
        n_components=int(config.n_components),
        seed=int(config.seed),
    )
    scores.index.name = "time"

    edges = discover_causal_edges(
        scores,
        max_lag=int(config.max_lag),
        pc_alpha=float(config.pc_alpha),
        link_density=float(config.link_density),
        backend=config.causal_backend,  # type: ignore[arg-type]
    )
    effects = compute_sem_effects(edges, n_components=int(config.n_components), max_lag=int(config.max_lag))

    component_scores_path = result_dir / "component_monthly_scores.csv"
    linear_path = result_dir / "linear_coefficient_matrix.csv"
    component_maps_path = result_dir / "component_maps.npz"
    scores.to_csv(component_scores_path, index_label="time")
    np.savez_compressed(
        component_maps_path,
        component_maps=component_maps,
        explained_variance_ratio=explained,
        lat=np.asarray(field["lat"].values),
        lon=np.asarray(field["lon"].values),
    )
    pd.DataFrame(effects.linear_coefficient_matrix, index=scores.columns, columns=scores.columns).to_csv(linear_path)
    effects.direct_effects.to_csv(result_dir / "direct_effects.csv", index=False)
    effects.total_effects.to_csv(result_dir / "total_effects.csv", index=False)
    effects.path_effects.to_csv(result_dir / "mediated_path_effects.csv", index=False)
    effects.gateway_scores.to_csv(result_dir / "gateway_scores.csv", index=False)
    effects.mediator_scores.to_csv(result_dir / "mediator_scores.csv", index=False)

    save_component_map_figure(component_maps, fig_dir / "component_maps.png")
    save_causal_network_figure(edges, fig_dir / "linear_causal_network.png", n_components=int(config.n_components))
    save_ranking_figure(effects.gateway_scores, fig_dir / "linear_gateway_ranking.png", title="2m temperature linear gateway ranking")
    save_ranking_figure(effects.mediator_scores, fig_dir / "linear_mediator_ranking.png", title="2m temperature linear mediator ranking")

    manifest = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "input_netcdf": str(_resolve_path(config.input_netcdf)),
        "variable": config.variable,
        "frequency": "monthly",
        "time_start": str(pd.to_datetime(field["time"].values[0]).date()),
        "time_end": str(pd.to_datetime(field["time"].values[-1]).date()),
        "n_monthly_samples": int(scores.shape[0]),
        "n_components": int(config.n_components),
        "n_edges": int(len(edges)),
        "dependency_versions": dependency_versions(),
        "top_gateways": effects.gateway_scores.head(10).to_dict("records"),
        "top_mediators": effects.mediator_scores.head(10).to_dict("records"),
    }
    manifest_path = result_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_preprocess_summary(result_dir / "summary.md", manifest, effects.gateway_scores, effects.mediator_scores)
    return PreprocessArtifacts(
        result_dir=result_dir,
        fig_dir=fig_dir,
        component_scores=component_scores_path,
        linear_coefficients=linear_path,
        component_maps=component_maps_path,
        manifest=manifest_path,
    )


def _read_first_row(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def collect_horizon_summary(base_result_dir: str | Path, horizons: Sequence[int]) -> pd.DataFrame:
    root = Path(base_result_dir)
    rows: list[dict[str, object]] = []
    for horizon in sorted(int(value) for value in horizons):
        horizon_root = root / f"horizon_{horizon:02d}"
        pair_dir = horizon_root / "results" / "runge" / "pairwise_mlp_tm_ei_path_effects"
        peid_dir = horizon_root / "results" / "runge" / "peid_hypergraph"
        pair_manifest = json.loads((pair_dir / "manifest.json").read_text(encoding="utf-8")) if (pair_dir / "manifest.json").exists() else {}
        metrics = pd.read_csv(pair_dir / "mlp_metrics.csv") if (pair_dir / "mlp_metrics.csv").exists() else pd.DataFrame()
        test_overall = metrics[(metrics.get("split") == "test") & (metrics.get("component") == "overall")] if not metrics.empty else pd.DataFrame()
        metric_row = test_overall.iloc[0].to_dict() if not test_overall.empty else {}
        gateway = _read_first_row(pair_dir / "gateway_scores.csv")
        mediator = _read_first_row(pair_dir / "mediator_scores.csv")
        peid = pd.read_csv(peid_dir / "peid_hyperedges.csv") if (peid_dir / "peid_hyperedges.csv").exists() else pd.DataFrame()
        if not peid.empty and "delta_K" in peid.columns:
            candidates = peid[peid.get("order", 0) == 2].copy()
            if candidates.empty:
                candidates = peid.copy()
            candidates["_abs_delta"] = candidates["delta_K"].abs()
            peid_row = candidates.sort_values("_abs_delta", ascending=False).iloc[0].to_dict()
        else:
            peid_row = {}
        pairwise_matrix = pd.read_csv(pair_dir / "pairwise_ei_matrix.csv", index_col=0) if (pair_dir / "pairwise_ei_matrix.csv").exists() else pd.DataFrame()
        mean_pairwise_ei = float(np.nanmean(pairwise_matrix.to_numpy(dtype=float))) if not pairwise_matrix.empty else np.nan
        rows.append(
            {
                "horizon_months": horizon,
                "n_supervised_samples": int(pair_manifest.get("n_supervised_samples", pair_manifest.get("n_lagged_samples", -1))),
                "test_rmse": float(metric_row.get("rmse", np.nan)),
                "test_mae": float(metric_row.get("mae", np.nan)),
                "test_corr": float(metric_row.get("corr", np.nan)),
                "mean_pairwise_ei": mean_pairwise_ei,
                "top_gateway_component": gateway.get("component"),
                "top_gateway_ace": float(gateway.get("ace", np.nan)),
                "top_gateway_acs": float(gateway.get("acs", np.nan)),
                "top_mediator_component": mediator.get("component"),
                "top_mediator_amce": float(mediator.get("amce", np.nan)),
                "top_peid_subset": peid_row.get("subset_str"),
                "top_peid_target": peid_row.get("target"),
                "top_peid_delta": float(peid_row.get("delta_K", np.nan)),
                "top_peid_z": float(peid_row.get("z", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def save_horizon_profile(summary: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    x = summary["horizon_months"].to_numpy(dtype=float)
    axes[0].plot(x, summary["test_rmse"], marker="o", label="RMSE")
    axes[0].plot(x, summary["test_mae"], marker="s", label="MAE")
    axes[0].set_xlabel("Horizon (months)")
    axes[0].set_ylabel("Prediction error")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    axes[1].plot(x, summary["test_corr"], marker="o", color="#4C78A8", label="Test corr")
    axes[1].plot(x, summary["mean_pairwise_ei"], marker="s", color="#D97732", label="Mean pairwise EI")
    axes[1].set_xlabel("Horizon (months)")
    axes[1].set_ylabel("Score")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    axes[2].plot(x, summary["top_gateway_ace"], marker="o", color="#1B9E77", label="Top ACE")
    axes[2].plot(x, summary["top_mediator_amce"], marker="^", color="#7570B3", label="Top AMCE")
    axes[2].plot(x, summary["top_peid_delta"], marker="D", color="#E45756", label="Top PEID Delta")
    axes[2].set_xlabel("Horizon (months)")
    axes[2].set_ylabel("Information/path score")
    axes[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def _append_run_history(repo_root: Path, row: dict[str, object]) -> None:
    log_dir = repo_root / "docs" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "run_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_live_status(repo_root: Path, text: str) -> None:
    log_dir = repo_root / "docs" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "live_status.md").write_text(text, encoding="utf-8")


def _run_logged(command: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        subprocess.run(command, cwd=cwd, check=True, stdout=handle, stderr=subprocess.STDOUT, text=True)


def run_downstream_horizon(
    *,
    horizon: int,
    artifacts: PreprocessArtifacts,
    output_dir: Path,
    pairwise_args: Sequence[str],
    peid_args: Sequence[str],
    skip_peid: bool,
) -> Path:
    repo = _repo_root()
    result_root = Path(output_dir) / RESULT_SUBDIR
    horizon_root = result_root / f"horizon_{int(horizon):02d}"
    log_dir = repo / "docs" / "log" / "logs"
    pairwise_command = [
        sys.executable,
        str(repo / "scripts" / "run_runge_pairwise_mlp_ei.py"),
        "--component-scores",
        str(artifacts.component_scores),
        "--linear-coefficients",
        str(artifacts.linear_coefficients),
        "--output-dir",
        str(horizon_root),
        "--horizon",
        str(int(horizon)),
        *pairwise_args,
    ]
    _write_live_status(
        repo,
        "\n".join(
            [
                "# 当前状态",
                f"正在运行 2m 气温 monthly horizon={horizon} 的 pairwise MLP-TM-EI。",
                "",
                "# 监控文件",
                f"- `{log_dir / f'runge_t2m_h{horizon:02d}_pairwise.log'}`",
            ]
        ),
    )
    _run_logged(pairwise_command, cwd=repo, log_path=log_dir / f"runge_t2m_h{horizon:02d}_pairwise.log")
    _append_run_history(repo, {"experiment": "runge_t2m_monthly", "horizon": int(horizon), "stage": "pairwise", "status": "finished"})

    if not skip_peid:
        pair_dir = horizon_root / "results" / "runge" / "pairwise_mlp_tm_ei_path_effects"
        peid_command = [
            sys.executable,
            str(repo / "scripts" / "run_runge_peid_hypergraph.py"),
            "--component-scores",
            str(artifacts.component_scores),
            "--output-dir",
            str(horizon_root),
            "--horizon",
            str(int(horizon)),
            "--pairwise-matrix-path",
            str(pair_dir / "pairwise_ei_matrix.csv"),
            "--pairwise-gateway-path",
            str(pair_dir / "gateway_scores.csv"),
            "--pairwise-mediator-path",
            str(pair_dir / "mediator_scores.csv"),
            *peid_args,
        ]
        _write_live_status(
            repo,
            "\n".join(
                [
                    "# 当前状态",
                    f"正在运行 2m 气温 monthly horizon={horizon} 的 PEID hypergraph。",
                    "",
                    "# 监控文件",
                    f"- `{log_dir / f'runge_t2m_h{horizon:02d}_peid.log'}`",
                ]
            ),
        )
        _run_logged(peid_command, cwd=repo, log_path=log_dir / f"runge_t2m_h{horizon:02d}_peid.log")
        _append_run_history(repo, {"experiment": "runge_t2m_monthly", "horizon": int(horizon), "stage": "peid", "status": "finished"})
    return horizon_root


def write_final_report(summary: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    if summary.empty:
        text = "# 2m 气温 Runge 多 Horizon 结果\n\n尚未生成 horizon 结果。\n"
    else:
        best_corr = summary.sort_values("test_corr", ascending=False).iloc[0]
        best_peid = summary.reindex(summary["top_peid_delta"].abs().sort_values(ascending=False).index).iloc[0]
        text = "\n".join(
            [
                "# 2m 气温 Runge 多 Horizon 结果",
                "",
                "本实验使用 NCEP/NCAR Reanalysis 1 monthly near-surface 2m air temperature。它是月尺度遥相关扫描，不是原 Runge SLP 日资料到周尺度的复现。",
                "",
                "## Horizon 汇总",
                "",
                summary.to_markdown(index=False, floatfmt=".6g"),
                "",
                "## 主要读数",
                "",
                f"- 最高测试相关出现在 `{int(best_corr['horizon_months'])}` 个月，test corr = `{best_corr['test_corr']:.4g}`。",
                f"- 绝对值最大的二阶 PEID 增量出现在 `{int(best_peid['horizon_months'])}` 个月，subset `{best_peid['top_peid_subset']}` -> `{best_peid['top_peid_target']}`，Delta = `{best_peid['top_peid_delta']:.4g}`。",
                "",
            ]
        )
    output.write_text(text, encoding="utf-8")
    return output


def default_pairwise_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--lag",
        str(args.lag),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--batch-size",
        str(args.batch_size),
        "--weight-decay",
        str(args.weight_decay),
        "--ridge-alpha",
        str(args.ridge_alpha),
        "--ensemble-ridge-alphas",
        args.ensemble_ridge_alphas,
        "--linear-blend-grid-steps",
        str(args.linear_blend_grid_steps),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--scheduler-patience",
        str(args.scheduler_patience),
        "--gradient-clip-norm",
        str(args.gradient_clip_norm),
        "--intervention-samples",
        str(args.intervention_samples),
        "--ei-estimator",
        "tm",
        "--gateway-mode",
        "path_effect",
        "--source-mode",
        args.source_mode,
        "--seed",
        str(args.seed),
    ]
    if args.force_retrain:
        values.append("--force-retrain")
    return values


def default_peid_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--lag",
        str(args.lag),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--batch-size",
        str(args.batch_size),
        "--weight-decay",
        str(args.weight_decay),
        "--ridge-alpha",
        str(args.ridge_alpha),
        "--ensemble-ridge-alphas",
        args.ensemble_ridge_alphas,
        "--linear-blend-grid-steps",
        str(args.linear_blend_grid_steps),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--scheduler-patience",
        str(args.scheduler_patience),
        "--gradient-clip-norm",
        str(args.gradient_clip_norm),
        "--intervention-samples",
        str(args.intervention_samples),
        "--order-max",
        str(args.order_max),
        "--candidate-top-sources",
        str(args.candidate_top_sources),
        "--candidate-target-topk",
        str(args.candidate_target_topk),
        "--null-reps",
        str(args.null_reps),
        "--source-mode",
        args.source_mode,
        "--seed",
        str(args.seed),
    ]
    if args.force_retrain:
        values.append("--force-retrain")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-netcdf", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--variable", default="air")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--n-components", type=int, default=60)
    parser.add_argument("--max-lag", type=int, default=4)
    parser.add_argument("--pc-alpha", type=float, default=0.001)
    parser.add_argument("--link-density", type=float, default=0.2)
    parser.add_argument("--causal-backend", choices=["tigramite", "regression"], default="tigramite")
    parser.add_argument("--horizons", default="1-12")
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--ridge-alpha", type=float, default=1000.0)
    parser.add_argument("--ensemble-ridge-alphas", default="10,100,1000,3000")
    parser.add_argument("--linear-blend-grid-steps", type=int, default=101)
    parser.add_argument("--early-stopping-patience", type=int, default=80)
    parser.add_argument("--scheduler-patience", type=int, default=20)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--intervention-samples", type=int, default=4096)
    parser.add_argument("--source-mode", choices=["latest", "history"], default="latest")
    parser.add_argument("--order-max", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--candidate-top-sources", type=int, default=14)
    parser.add_argument("--candidate-target-topk", type=int, default=10)
    parser.add_argument("--null-reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-downstream", action="store_true")
    parser.add_argument("--skip-peid", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    horizons = parse_horizons(args.horizons)
    config = T2MRungeConfig(
        input_netcdf=args.input_netcdf,
        variable=args.variable,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        n_components=args.n_components,
        max_lag=args.max_lag,
        pc_alpha=args.pc_alpha,
        link_density=args.link_density,
        seed=args.seed,
        causal_backend=args.causal_backend,
    )
    artifacts = run_preprocessing(config)
    if not args.skip_downstream:
        pairwise_args = default_pairwise_args(args)
        peid_args = default_peid_args(args)
        for horizon in horizons:
            run_downstream_horizon(
                horizon=horizon,
                artifacts=artifacts,
                output_dir=args.output_dir,
                pairwise_args=pairwise_args,
                peid_args=peid_args,
                skip_peid=bool(args.skip_peid),
            )

    summary = collect_horizon_summary(Path(args.output_dir) / RESULT_SUBDIR, horizons)
    summary_path = Path(args.output_dir) / RESULT_SUBDIR / "horizon_summary.csv"
    summary.to_csv(summary_path, index=False)
    fig_path = save_horizon_profile(summary, Path(args.output_dir) / FIG_SUBDIR / "horizon_profile.png") if not summary.empty else None
    report_path = write_final_report(summary, Path(args.output_dir) / RESULT_SUBDIR / "multihorizon_report.md")
    _write_live_status(
        _repo_root(),
        "\n".join(
            [
                "# 当前状态",
                "2m 气温 monthly Runge 多 horizon 管线完成当前命令。",
                "",
                "# 最近结果",
                f"- summary: `{summary_path}`",
                f"- report: `{report_path}`",
                f"- figure: `{fig_path}`" if fig_path else "- figure: 未生成",
            ]
        ),
    )
    print(json.dumps({"summary": str(summary_path), "report": str(report_path), "horizons": horizons}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
