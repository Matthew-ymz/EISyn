#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from peid_causal_hypergraph import DEFAULT_INPUT, DEFAULT_PROJECT_ROOT, DEFAULT_VARIABLES, PeidConfig, plot_outputs, run_peid_analysis


DEFAULT_OUTPUT_ROOT = DEFAULT_PROJECT_ROOT / "results" / "company_ce" / "csv" / "peid_sensitivity"
DEFAULT_FIGURE_DIR = DEFAULT_PROJECT_ROOT / "fig" / "company_ce" / "peid_sensitivity"
DEFAULT_WINSOR_PAIRS = ((0.005, 0.995), (0.01, 0.99), (0.025, 0.975))
DEFAULT_YEAR_WINDOWS = (
    ("full", None, None),
    ("early", None, 2000),
    ("late", 2001, None),
)


@dataclass(frozen=True)
class SensitivityConfig:
    base_config: PeidConfig = field(default_factory=lambda: PeidConfig(bins=5))
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    figure_dir: str = str(DEFAULT_FIGURE_DIR)
    bins_values: tuple[int, ...] = (3, 4, 5, 6)
    min_source_count_values: tuple[int, ...] = (10, 20, 50)
    alpha_values: tuple[float, ...] = (0.1, 0.5, 1.0)
    winsor_pairs: tuple[tuple[float, float], ...] = DEFAULT_WINSOR_PAIRS
    year_windows: tuple[tuple[str, int | None, int | None], ...] = DEFAULT_YEAR_WINDOWS
    top_m: int = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-factor PEID sensitivity scans.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--variables", nargs="+", default=list(DEFAULT_VARIABLES))
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--max-source-order", type=int, default=2, choices=(2, 3))
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--min-source-count", type=int, default=20)
    parser.add_argument("--min-total-count", type=int, default=100)
    parser.add_argument("--null-reps", type=int, default=20)
    parser.add_argument("--top-m", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--skip-run-figures", action="store_true")
    return parser.parse_args()


def edge_key(frame: pd.DataFrame, key_cols: list[str]) -> pd.Series:
    return frame[key_cols].astype(str).agg("||".join, axis=1)


def top_overlap(baseline: pd.DataFrame, variant: pd.DataFrame, key_cols: list[str], value_col: str, top_m: int) -> float:
    if baseline.empty or variant.empty or top_m <= 0:
        return np.nan
    base_top = set(edge_key(baseline.sort_values(value_col, ascending=False).head(top_m), key_cols))
    var_top = set(edge_key(variant.sort_values(value_col, ascending=False).head(top_m), key_cols))
    if not base_top:
        return np.nan
    return float(len(base_top & var_top) / len(base_top))


def merge_edge_values(
    baseline: pd.DataFrame,
    variant: pd.DataFrame,
    key_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    keep_cols = [*key_cols, value_col, "p_value"]
    base = baseline.reindex(columns=keep_cols).rename(columns={value_col: "baseline_value", "p_value": "baseline_p_value"})
    var = variant.reindex(columns=keep_cols).rename(columns={value_col: "variant_value", "p_value": "variant_p_value"})
    merged = base.merge(var, on=key_cols, how="inner")
    merged["absolute_delta"] = (merged["variant_value"] - merged["baseline_value"]).abs()
    denom = merged["baseline_value"].abs().replace(0, np.nan)
    merged["relative_delta"] = merged["absolute_delta"] / denom
    return merged


def p_value_keep_rate(merged: pd.DataFrame, threshold: float = 0.05) -> float:
    if merged.empty or "baseline_p_value" not in merged or "variant_p_value" not in merged:
        return np.nan
    baseline_significant = merged["baseline_p_value"] <= threshold
    if not bool(baseline_significant.any()):
        return np.nan
    return float((merged.loc[baseline_significant, "variant_p_value"] <= threshold).mean())


def summarize_run_against_baseline(
    *,
    run_id: str,
    varied_parameter: str,
    varied_value: str,
    baseline_pairwise: pd.DataFrame,
    variant_pairwise: pd.DataFrame,
    baseline_synergy: pd.DataFrame,
    variant_synergy: pd.DataFrame,
    top_m: int,
    null_reps: int,
) -> dict[str, object]:
    pair_merged = merge_edge_values(baseline_pairwise, variant_pairwise, ["source", "target"], "ei")
    syn_merged = merge_edge_values(baseline_synergy, variant_synergy, ["sources", "target", "source_order"], "synergy_raw")

    pair_spearman = (
        float(pair_merged["baseline_value"].corr(pair_merged["variant_value"], method="spearman"))
        if len(pair_merged) >= 2
        else np.nan
    )
    syn_spearman = (
        float(syn_merged["baseline_value"].corr(syn_merged["variant_value"], method="spearman"))
        if len(syn_merged) >= 2
        else np.nan
    )

    baseline_sign = np.sign(syn_merged["baseline_value"].to_numpy(dtype=float)) if not syn_merged.empty else np.array([])
    variant_sign = np.sign(syn_merged["variant_value"].to_numpy(dtype=float)) if not syn_merged.empty else np.array([])
    nonzero_mask = (baseline_sign != 0) | (variant_sign != 0)
    sign_agreement = float((baseline_sign[nonzero_mask] == variant_sign[nonzero_mask]).mean()) if nonzero_mask.any() else np.nan
    sign_flip_count = int((baseline_sign[nonzero_mask] != variant_sign[nonzero_mask]).sum()) if nonzero_mask.any() else 0

    formal_p_values = bool(null_reps >= 100)
    return {
        "run_id": run_id,
        "varied_parameter": varied_parameter,
        "varied_value": varied_value,
        "null_reps": int(null_reps),
        "p_value_stability_is_formal": formal_p_values,
        "pairwise_edge_count": int(len(pair_merged)),
        "pairwise_spearman": pair_spearman,
        "pairwise_top_m_overlap": top_overlap(baseline_pairwise, variant_pairwise, ["source", "target"], "ei", top_m),
        "pairwise_abs_delta_mean": float(pair_merged["absolute_delta"].mean()) if not pair_merged.empty else np.nan,
        "pairwise_relative_delta_median": float(pair_merged["relative_delta"].median()) if not pair_merged.empty else np.nan,
        "pairwise_p_value_keep_rate": p_value_keep_rate(pair_merged) if formal_p_values else np.nan,
        "synergy_edge_count": int(len(syn_merged)),
        "synergy_spearman": syn_spearman,
        "synergy_top_m_overlap": top_overlap(
            baseline_synergy.assign(value_abs=baseline_synergy["synergy_raw"].abs()),
            variant_synergy.assign(value_abs=variant_synergy["synergy_raw"].abs()),
            ["sources", "target", "source_order"],
            "value_abs",
            top_m,
        ),
        "synergy_sign_agreement": sign_agreement,
        "synergy_sign_flip_count": sign_flip_count,
        "synergy_abs_delta_mean": float(syn_merged["absolute_delta"].mean()) if not syn_merged.empty else np.nan,
        "synergy_p_value_keep_rate": p_value_keep_rate(syn_merged) if formal_p_values else np.nan,
    }


def sensitivity_runs(config: SensitivityConfig) -> list[tuple[str, str, str, PeidConfig]]:
    base = config.base_config
    runs: list[tuple[str, str, str, PeidConfig]] = [("baseline", "baseline", "baseline", base)]
    for value in config.bins_values:
        if value != base.bins:
            runs.append((f"bins_{value}", "bins", str(value), replace(base, bins=value)))
    for value in config.min_source_count_values:
        if value != base.min_source_count:
            runs.append((f"min_source_count_{value}", "min_source_count", str(value), replace(base, min_source_count=value)))
    for value in config.alpha_values:
        if value != base.alpha:
            runs.append((f"alpha_{value:g}", "alpha", f"{value:g}", replace(base, alpha=value)))
    for lower, upper in config.winsor_pairs:
        if lower != base.winsor_lower or upper != base.winsor_upper:
            run_id = f"winsor_{lower:g}_{upper:g}".replace(".", "p")
            runs.append((run_id, "winsor", f"{lower:g},{upper:g}", replace(base, winsor_lower=lower, winsor_upper=upper)))
    for label, year_start, year_end in config.year_windows:
        if year_start != base.year_start or year_end != base.year_end:
            runs.append((f"years_{label}", "year_window", label, replace(base, year_start=year_start, year_end=year_end)))
    return runs


def read_run_edges(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(output_dir / "peid_pairwise_edges.csv"),
        pd.read_csv(output_dir / "peid_synergy_hyperedges.csv"),
    )


def plot_sensitivity_summary(summary: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({"pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})

    bins = summary[summary["varied_parameter"] == "bins"].copy()
    if bins.empty:
        return
    bins["bin_value"] = bins["varied_value"].astype(int)
    bins = bins.sort_values("bin_value")

    for metric, ylabel, stem in (
        ("pairwise_spearman", "Spearman rank correlation", "peid_bins_sensitivity_pairwise"),
        ("synergy_spearman", "Synergy_raw rank correlation", "peid_bins_sensitivity_synergy"),
    ):
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        ax.plot(bins["bin_value"], bins[metric], marker="o", color="#4C78A8")
        ax.set_xlabel("BINS")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)
        fig.savefig(figure_dir / f"{stem}.png", dpi=260, bbox_inches="tight")
        fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)


def run_sensitivity_analysis(config: SensitivityConfig, *, reuse_cache: bool = False, skip_run_figures: bool = False) -> dict[str, str]:
    output_root = Path(config.output_root)
    figure_dir = Path(config.figure_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    pairwise_rows: list[pd.DataFrame] = []
    synergy_rows: list[pd.DataFrame] = []
    edge_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for run_id, varied_parameter, varied_value, run_config in sensitivity_runs(config):
        run_output = output_root / run_id
        run_figure_dir = figure_dir / run_id
        run_config = replace(run_config, output_dir=str(run_output), figure_dir=str(run_figure_dir))
        result = run_peid_analysis(run_config, skip_figures=skip_run_figures, reuse_cache=reuse_cache)
        pairwise, synergy = read_run_edges(run_output)
        edge_cache[run_id] = (pairwise, synergy)
        pairwise_rows.append(pairwise.assign(run_id=run_id, varied_parameter=varied_parameter, varied_value=varied_value))
        synergy_rows.append(synergy.assign(run_id=run_id, varied_parameter=varied_parameter, varied_value=varied_value))
        run_rows.append(
            {
                "run_id": run_id,
                "varied_parameter": varied_parameter,
                "varied_value": varied_value,
                "output_dir": str(run_output),
                "figure_dir": str(run_figure_dir),
                "reused_cache": result["reused_cache"],
                "cache_reason": result.get("cache_reason", ""),
                **asdict(run_config),
            }
        )

    baseline_pairwise, baseline_synergy = edge_cache["baseline"]
    summary_rows = []
    for run_id, varied_parameter, varied_value, run_config in sensitivity_runs(config):
        if run_id == "baseline":
            continue
        pairwise, synergy = edge_cache[run_id]
        summary_rows.append(
            summarize_run_against_baseline(
                run_id=run_id,
                varied_parameter=varied_parameter,
                varied_value=varied_value,
                baseline_pairwise=baseline_pairwise,
                variant_pairwise=pairwise,
                baseline_synergy=baseline_synergy,
                variant_synergy=synergy,
                top_m=config.top_m,
                null_reps=run_config.null_reps,
            )
        )

    runs = pd.DataFrame(run_rows)
    pairwise_all = pd.concat(pairwise_rows, ignore_index=True, sort=False)
    synergy_all = pd.concat(synergy_rows, ignore_index=True, sort=False)
    summary = pd.DataFrame(summary_rows)

    runs.to_csv(output_root / "peid_sensitivity_runs.csv", index=False)
    pairwise_all.to_csv(output_root / "peid_sensitivity_pairwise_edges.csv", index=False)
    synergy_all.to_csv(output_root / "peid_sensitivity_synergy_edges.csv", index=False)
    summary.to_csv(output_root / "peid_sensitivity_summary.csv", index=False)
    with (output_root / "peid_sensitivity_config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2, ensure_ascii=False)
    plot_sensitivity_summary(summary, figure_dir)
    return {"output_root": str(output_root), "figure_dir": str(figure_dir), "runs": str(len(runs))}


def build_config_from_args(args: argparse.Namespace) -> SensitivityConfig:
    variables: Iterable[str] = args.variables
    bins_values = (3, 5) if args.smoke else (3, 4, 5, 6)
    min_source_count_values = (10, 20) if args.smoke else (10, 20, 50)
    alpha_values = (0.5,) if args.smoke else (0.1, 0.5, 1.0)
    winsor_pairs = ((0.01, 0.99),) if args.smoke else DEFAULT_WINSOR_PAIRS
    year_windows = (("full", None, None),) if args.smoke else DEFAULT_YEAR_WINDOWS
    base_config = PeidConfig(
        input_path=str(args.input),
        variables=tuple(variables),
        bins=args.bins,
        max_source_order=args.max_source_order,
        alpha=args.alpha,
        min_source_count=10 if args.smoke else args.min_source_count,
        min_total_count=args.min_total_count,
        null_reps=2 if args.smoke else args.null_reps,
        top_k=min(args.top_m, 5) if args.smoke else args.top_m,
        random_seed=args.random_seed,
        winsor_lower=0.01,
        winsor_upper=0.99,
    )
    if args.smoke:
        base_config = replace(base_config, variables=tuple(list(args.variables)[:3]))
    return SensitivityConfig(
        base_config=base_config,
        output_root=str(args.output_root),
        figure_dir=str(args.figure_dir),
        bins_values=bins_values,
        min_source_count_values=min_source_count_values,
        alpha_values=alpha_values,
        winsor_pairs=winsor_pairs,
        year_windows=year_windows,
        top_m=min(args.top_m, 5) if args.smoke else args.top_m,
    )


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)
    result = run_sensitivity_analysis(config, reuse_cache=args.reuse_cache, skip_run_figures=args.skip_run_figures)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
