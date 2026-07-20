from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


YEARS = list(range(2017, 2023))


PROFILE_COLUMNS = [
    "refYear",
    "reporterCode",
    "reporterISO",
    "reporterDesc",
    "flowCode",
    "flowDesc",
    "partnerCode",
    "cmdCode",
    "cmdDesc",
    "aggrLevel",
    "isLeaf",
    "primaryValue",
    "netWgt",
    "isAggregate",
]


def profile_csv(input_path: Path, output_path: Path) -> None:
    rows = 0
    missing = Counter()
    years = Counter()
    flows = Counter()
    levels = Counter()
    reporters: set[str] = set()
    products: set[str] = set()
    partner_codes: set[int] = set()
    world_rows = 0
    aggregate_rows = 0
    leaf_rows = 0
    primary_sum_world = 0.0

    for chunk in pd.read_csv(
        input_path,
        usecols=PROFILE_COLUMNS,
        chunksize=200_000,
        low_memory=False,
        encoding="utf-8-sig",
    ):
        rows += len(chunk)
        missing.update({c: int(chunk[c].isna().sum()) for c in PROFILE_COLUMNS})
        years.update(chunk["refYear"].dropna().astype(int).tolist())
        flows.update(chunk["flowDesc"].fillna("Missing").astype(str).tolist())
        levels.update(chunk["aggrLevel"].dropna().astype(int).tolist())
        reporters.update(chunk["reporterISO"].dropna().astype(str).unique())
        products.update(chunk["cmdCode"].dropna().astype(str).str.zfill(6).unique())
        partner_codes.update(chunk["partnerCode"].dropna().astype(int).unique())
        world_mask = chunk["partnerCode"].eq(0)
        world_rows += int(world_mask.sum())
        aggregate_rows += int(chunk["isAggregate"].fillna(False).astype(bool).sum())
        leaf_rows += int(chunk["isLeaf"].fillna(False).astype(bool).sum())
        primary_sum_world += float(
            pd.to_numeric(chunk.loc[world_mask, "primaryValue"], errors="coerce").sum()
        )

    payload = {
        "rows": rows,
        "years": dict(sorted(years.items())),
        "flows": dict(flows),
        "aggregation_levels": dict(sorted(levels.items())),
        "unique_reporter_iso": len(reporters),
        "unique_products": len(products),
        "unique_partner_codes": len(partner_codes),
        "world_rows": world_rows,
        "aggregate_rows": aggregate_rows,
        "leaf_rows": leaf_rows,
        "missing": dict(missing),
        "world_primary_value_sum": primary_sum_world,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def aggregate_world_series(input_path: Path) -> pd.DataFrame:
    usecols = [
        "refYear",
        "reporterISO",
        "reporterDesc",
        "partnerCode",
        "cmdCode",
        "cmdDesc",
        "primaryValue",
    ]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        input_path,
        usecols=usecols,
        chunksize=200_000,
        low_memory=False,
        encoding="utf-8-sig",
    ):
        world = chunk.loc[chunk["partnerCode"].eq(0)].copy()
        world["cmdCode"] = world["cmdCode"].astype(str).str.zfill(6)
        world["primaryValue"] = pd.to_numeric(world["primaryValue"], errors="coerce")
        grouped = (
            world.groupby(
                ["reporterISO", "reporterDesc", "cmdCode", "cmdDesc", "refYear"],
                observed=True,
                as_index=False,
            )["primaryValue"]
            .sum()
        )
        pieces.append(grouped)
    return (
        pd.concat(pieces, ignore_index=True)
        .groupby(
            ["reporterISO", "reporterDesc", "cmdCode", "cmdDesc", "refYear"],
            observed=True,
            as_index=False,
        )["primaryValue"]
        .sum()
    )


def select_representatives(series: pd.DataFrame, n_clusters: int = 6) -> tuple[pd.DataFrame, dict]:
    from sklearn.cluster import KMeans

    index_cols = ["reporterISO", "reporterDesc", "cmdCode", "cmdDesc"]
    pivot = series.pivot_table(
        index=index_cols,
        columns="refYear",
        values="primaryValue",
        aggfunc="sum",
    ).reindex(columns=YEARS)
    complete = pivot.dropna().copy()
    nonnegative = complete[(complete >= 0).all(axis=1)].copy()
    material_cutoff = float(nonnegative.sum(axis=1).quantile(0.50))
    candidates = nonnegative[nonnegative.sum(axis=1) >= material_cutoff].copy()

    values = np.log1p(candidates.to_numpy(dtype=float))
    centered = values - values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    scale[scale < 1e-8] = 1.0
    shapes = centered / scale

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=30)
    labels = model.fit_predict(shapes)
    distances = model.transform(shapes)

    chosen: list[int] = []
    used_reporters: set[str] = set()
    cluster_sizes: dict[str, int] = {}
    for cluster in range(n_clusters):
        members = np.flatnonzero(labels == cluster)
        cluster_sizes[str(cluster + 1)] = int(len(members))
        ranked = members[np.argsort(distances[members, cluster])]
        pick = int(ranked[0])
        for candidate_idx in ranked:
            reporter = str(candidates.index[int(candidate_idx)][0])
            if reporter not in used_reporters:
                pick = int(candidate_idx)
                break
        chosen.append(pick)
        used_reporters.add(str(candidates.index[pick][0]))

    selected = candidates.iloc[chosen].copy()
    selected.insert(0, "cluster", np.arange(1, n_clusters + 1))
    selected = selected.reset_index()
    selected.columns.name = None
    for idx, row in selected.iterrows():
        first = float(row[YEARS[0]])
        last = float(row[YEARS[-1]])
        selected.loc[idx, "cagr"] = (last / first) ** (1 / 5) - 1 if first > 0 else np.nan
        selected.loc[idx, "cv"] = float(
            np.std([row[y] for y in YEARS]) / np.mean([row[y] for y in YEARS])
        )

    meta = {
        "world_reporter_product_year_rows": int(len(series)),
        "complete_series": int(len(complete)),
        "nonnegative_complete_series": int(len(nonnegative)),
        "candidate_series_above_median_total": int(len(candidates)),
        "material_total_cutoff_usd": material_cutoff,
        "cluster_sizes": cluster_sizes,
    }
    return selected, meta


def compact_product(text: str, max_chars: int = 82) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def value_scale(max_value: float) -> tuple[float, str]:
    if max_value >= 1e9:
        return 1e9, "USD bn"
    if max_value >= 1e6:
        return 1e6, "USD mn"
    if max_value >= 1e3:
        return 1e3, "USD k"
    return 1.0, "USD"


def plot_representatives(selected: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )
    colors = ["#4C78A8", "#E09F3E", "#7A9E7E", "#9C6ADE", "#D17C78", "#5B8E7D"]
    fig, axes = plt.subplots(2, 3, figsize=(8.27, 6.15), constrained_layout=True)

    for ax, (_, row), color in zip(axes.flat, selected.iterrows(), colors):
        raw = np.array([float(row[y]) for y in YEARS])
        scale, unit = value_scale(float(raw.max()))
        values = raw / scale
        ax.plot(
            YEARS,
            values,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
        ax.fill_between(YEARS, 0, values, color=color, alpha=0.09, zorder=1)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks(YEARS)
        ax.tick_params(axis="x", rotation=0, labelsize=6.5)
        ax.set_ylabel(unit)
        ax.set_ylim(bottom=0)
        cagr = float(row["cagr"])
        cagr_text = "n/a" if not np.isfinite(cagr) else f"{cagr:+.1%} p.a."
        product_lines = "\n".join(textwrap.wrap(compact_product(row["cmdDesc"]), width=42))
        ax.set_title(
            f"{row['reporterDesc']} · HS {row['cmdCode']} · CAGR {cagr_text}\n"
            f"{product_lines}",
            loc="left",
            fontsize=6.9,
            fontweight="bold",
            pad=5,
        )

    fig.supxlabel("Year", fontsize=8)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in {
        "png": {"dpi": 400},
        "svg": {},
        "pdf": {},
    }.items():
        fig.savefig(output_dir / f"representative_trade_time_series.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def analyze_and_plot(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "profile.json"
    profile_csv(input_path, profile_path)
    world_series = aggregate_world_series(input_path)
    selected, selection_meta = select_representatives(world_series)
    plot_representatives(selected, output_dir)

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["selection"] = selection_meta
    profile["representatives"] = [
        {
            "cluster": int(row["cluster"]),
            "reporterISO": row["reporterISO"],
            "reporterDesc": row["reporterDesc"],
            "cmdCode": str(row["cmdCode"]),
            "cmdDesc": row["cmdDesc"],
            "values": {str(y): float(row[y]) for y in YEARS},
            "cagr": float(row["cagr"]) if np.isfinite(row["cagr"]) else None,
            "cv": float(row["cv"]),
        }
        for _, row in selected.iterrows()
    ]
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.plot:
        analyze_and_plot(args.input, args.output)
    else:
        profile_csv(args.input, args.output)


if __name__ == "__main__":
    main()
