#!/usr/bin/env python3
"""Create a regular, fully observed hourly topic-heat series for modelling.

The source heat index is not altered.  Missing values are filled only in the
model-ready copy and every imputed cell is explicitly flagged in the companion
mask and audit files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/platform_topic_heat_index_hourly_wide.csv"
OUTPUT = ROOT / "data/platform_topic_heat_index_hourly_model_ready.csv"
MASK_OUTPUT = ROOT / "data/platform_topic_heat_index_hourly_imputation_mask.csv"
AUDIT_OUTPUT = ROOT / "data/platform_topic_heat_index_hourly_imputation_audit.json"


def contiguous_missing_blocks(mask: pd.Series) -> list[dict[str, object]]:
    """Return the start, end, and length of each consecutive missing block."""
    blocks: list[dict[str, object]] = []
    groups = (mask != mask.shift()).cumsum()
    for _, block in mask[mask].groupby(groups[mask]):
        blocks.append(
            {
                "start": block.index.min().isoformat(),
                "end": block.index.max().isoformat(),
                "hours": int(block.size),
            }
        )
    return blocks


def main() -> None:
    raw = pd.read_csv(INPUT, parse_dates=["time_bin"])
    raw = raw.set_index("time_bin").sort_index()
    if raw.index.has_duplicates:
        raise ValueError("time_bin contains duplicate timestamps")

    expected_index = pd.date_range(raw.index.min(), raw.index.max(), freq="h")
    series = raw.reindex(expected_index)
    series.index.name = "time_bin"
    columns = list(series.columns)
    original_missing = series.isna()

    # Interior gaps are linearly interpolated in time.  This retains continuity
    # without inventing a discontinuity at the edge of an observed gap.  The one
    # endpoint gap in this dataset is filled with that platform's same-hour
    # median, rather than carrying the preceding value forward.
    filled = series.interpolate(method="time", limit_area="inside")
    methods = pd.DataFrame("observed", index=series.index, columns=columns)
    for column in columns:
        interior = original_missing[column] & filled[column].notna()
        methods.loc[interior, column] = "linear_time_interpolation"

        edge_missing = filled[column].isna()
        if edge_missing.any():
            hour_median = series[column].groupby(series.index.hour).median()
            fallback = pd.Series(
                [hour_median.loc[timestamp.hour] for timestamp in filled.index],
                index=filled.index,
            )
            filled.loc[edge_missing, column] = fallback.loc[edge_missing]
            methods.loc[edge_missing, column] = "platform_hour_of_day_median"

    if filled.isna().any().any():
        raise ValueError("imputation left missing values")
    if (filled < 0).any().any():
        raise ValueError("heat index must be non-negative")

    filled.reset_index().to_csv(OUTPUT, index=False, float_format="%.17g")
    methods.reset_index().to_csv(MASK_OUTPUT, index=False)

    audit = {
        "source": str(INPUT.relative_to(ROOT)),
        "model_ready_output": str(OUTPUT.relative_to(ROOT)),
        "time_frequency": "1h",
        "time_range": {"start": series.index.min().isoformat(), "end": series.index.max().isoformat()},
        "n_time_points": int(len(series)),
        "imputation_policy": {
            "interior_gaps": "linear interpolation in time",
            "endpoint_gaps": "platform-specific median at the same hour of day",
            "original_observations": "kept unchanged",
        },
        "platforms": {},
    }
    for column in columns:
        method_counts = methods[column].value_counts().to_dict()
        audit["platforms"][column] = {
            "observed_points": int((methods[column] == "observed").sum()),
            "imputed_points": int(original_missing[column].sum()),
            "method_counts": {key: int(value) for key, value in method_counts.items()},
            "missing_blocks": contiguous_missing_blocks(original_missing[column]),
        }
    AUDIT_OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
