#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_VARIABLES = ("at", "revt", "emp", "dltt")


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    default_input = repo_root / "data" / "inf_compustat_anual_US_filter_feas.csv"
    default_output = repo_root / "results" / "company_ce" / "csv" / "qtpm"

    parser = argparse.ArgumentParser(
        description="Build unconditional quantile transition probability matrices from Compustat annual data."
    )
    parser.add_argument("--input", type=Path, default=default_input, help="Input CSV path.")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Directory for CSV outputs.")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=list(DEFAULT_VARIABLES),
        help="Variables to analyze. Default: at revt emp dltt",
    )
    parser.add_argument("--quantiles", type=int, default=10, help="Number of quantile bins for the QTPM.")
    return parser.parse_args()


def prepare_panel(df: pd.DataFrame, variable: str) -> tuple[pd.DataFrame, dict[str, float]]:
    panel = df[["gvkey", "fyear", variable]].copy()
    panel = panel.dropna(subset=["gvkey", "fyear", variable])
    panel = panel[np.isfinite(panel[variable])]

    stats = {
        "n_raw_rows": float(len(panel)),
        "n_positive_rows": float((panel[variable] > 0).sum()),
    }

    panel = panel[panel[variable] > 0].copy()
    panel["gvkey"] = panel["gvkey"].astype(str)
    panel["fyear"] = panel["fyear"].astype(int)
    panel = panel.sort_values(["gvkey", "fyear"]).reset_index(drop=True)
    panel["log_value"] = np.log(panel[variable])
    panel["prev_fyear"] = panel.groupby("gvkey")["fyear"].shift(1)
    panel["prev_log_value"] = panel.groupby("gvkey")["log_value"].shift(1)
    panel["is_consecutive"] = (panel["fyear"] - panel["prev_fyear"]) == 1
    panel["growth"] = np.where(panel["is_consecutive"], panel["log_value"] - panel["prev_log_value"], np.nan)

    growth = panel.loc[panel["growth"].notna(), ["gvkey", "fyear", "growth"]].copy()
    growth = growth.rename(columns={"fyear": "mid_year", "growth": "growth_t"})
    growth["next_mid_year"] = growth.groupby("gvkey")["mid_year"].shift(-1)
    growth["growth_t1"] = growth.groupby("gvkey")["growth_t"].shift(-1)
    growth["has_consecutive_growth"] = (growth["next_mid_year"] - growth["mid_year"]) == 1
    pairs = growth.loc[growth["has_consecutive_growth"]].copy()
    pairs["start_year"] = pairs["mid_year"] - 1
    pairs["end_year"] = pairs["mid_year"] + 1
    pairs = pairs[["gvkey", "start_year", "mid_year", "end_year", "growth_t", "growth_t1"]].reset_index(drop=True)

    stats.update(
        {
            "n_growth_rows": float(len(growth.loc[growth["growth_t"].notna()])),
            "n_growth_pairs": float(len(pairs)),
            "n_firms_with_pairs": float(pairs["gvkey"].nunique()),
        }
    )
    return pairs, stats


def assign_quantile_bins(series: pd.Series, quantiles: int) -> tuple[pd.Series, pd.DataFrame]:
    labels = list(range(1, quantiles + 1))
    bins, edges = pd.qcut(series, q=quantiles, labels=labels, retbins=True, duplicates="drop")

    intervals = pd.DataFrame(
        {
            "bin": labels[: len(edges) - 1],
            "lower": edges[:-1],
            "upper": edges[1:],
        }
    )
    intervals.loc[0, "lower"] = -np.inf
    intervals.loc[len(intervals) - 1, "upper"] = np.inf
    return bins.astype(int), intervals


def build_qtpm_tables(pairs: pd.DataFrame, variable: str, quantiles: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_bin, x_intervals = assign_quantile_bins(pairs["growth_t"], quantiles=quantiles)
    y_bin, y_intervals = assign_quantile_bins(pairs["growth_t1"], quantiles=quantiles)

    labeled = pairs.copy()
    labeled["variable"] = variable
    labeled["x_bin"] = x_bin
    labeled["y_bin"] = y_bin

    x_categories = sorted(labeled["x_bin"].unique())
    y_categories = sorted(labeled["y_bin"].unique())
    counts = pd.crosstab(labeled["x_bin"], labeled["y_bin"], dropna=False).reindex(
        index=x_categories, columns=y_categories, fill_value=0
    )
    row_totals = counts.sum(axis=1)
    probabilities = counts.div(row_totals.replace(0, np.nan), axis=0).fillna(0.0)

    qtpm_long = (
        probabilities.stack()
        .rename("probability")
        .reset_index()
        .rename(columns={"x_bin": "from_bin", "y_bin": "to_bin"})
    )
    qtpm_long["count"] = [int(counts.loc[row.from_bin, row.to_bin]) for row in qtpm_long.itertuples()]
    qtpm_long["row_total"] = [int(row_totals.loc[row.from_bin]) for row in qtpm_long.itertuples()]
    qtpm_long["variable"] = variable

    quantiles_long = pd.concat(
        [
            x_intervals.assign(variable=variable, axis="growth_t"),
            y_intervals.assign(variable=variable, axis="growth_t1"),
        ],
        ignore_index=True,
    )[["variable", "axis", "bin", "lower", "upper"]]

    return labeled, qtpm_long, quantiles_long


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    usecols = ["gvkey", "fyear", *args.variables]
    df = pd.read_csv(args.input, usecols=usecols)

    all_pairs = []
    all_qtpm = []
    all_quantiles = []
    summary_rows = []

    for variable in args.variables:
        pairs, stats = prepare_panel(df=df, variable=variable)
        if pairs.empty:
            continue

        labeled_pairs, qtpm_long, quantiles_long = build_qtpm_tables(
            pairs=pairs,
            variable=variable,
            quantiles=args.quantiles,
        )

        summary_rows.append(
            {
                "variable": variable,
                "n_raw_rows": int(stats["n_raw_rows"]),
                "n_positive_rows": int(stats["n_positive_rows"]),
                "n_growth_rows": int(stats["n_growth_rows"]),
                "n_growth_pairs": int(stats["n_growth_pairs"]),
                "n_firms_with_pairs": int(stats["n_firms_with_pairs"]),
                "growth_t_mean": labeled_pairs["growth_t"].mean(),
                "growth_t_std": labeled_pairs["growth_t"].std(),
                "growth_t1_mean": labeled_pairs["growth_t1"].mean(),
                "growth_t1_std": labeled_pairs["growth_t1"].std(),
            }
        )

        matrix = qtpm_long.pivot(index="from_bin", columns="to_bin", values="probability").sort_index().sort_index(axis=1)
        counts = qtpm_long.pivot(index="from_bin", columns="to_bin", values="count").sort_index().sort_index(axis=1)
        matrix.to_csv(output_dir / f"qtpm_probability_matrix_{variable}.csv")
        counts.to_csv(output_dir / f"qtpm_count_matrix_{variable}.csv")

        all_pairs.append(labeled_pairs)
        all_qtpm.append(qtpm_long)
        all_quantiles.append(quantiles_long)

    if not all_pairs:
        raise SystemExit("No valid growth pairs were generated. Check input data and variable definitions.")

    pd.concat(all_pairs, ignore_index=True).to_csv(output_dir / "qtpm_growth_pairs.csv", index=False)
    pd.concat(all_qtpm, ignore_index=True).to_csv(output_dir / "qtpm_transitions.csv", index=False)
    pd.concat(all_quantiles, ignore_index=True).to_csv(output_dir / "qtpm_quantile_edges.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "qtpm_summary.csv", index=False)

    print(f"Wrote QTPM outputs to {output_dir}")


if __name__ == "__main__":
    main()
