#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_VARIABLES = ("at", "revt", "emp", "dltt", "lt", "ch")
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = DEFAULT_PROJECT_ROOT / "data" / "inf_compustat_anual_US_filter_feas.csv"
DEFAULT_CSV_DIR = DEFAULT_PROJECT_ROOT / "results" / "company_ce" / "csv" / "peid"
DEFAULT_FIGURE_DIR = DEFAULT_PROJECT_ROOT / "fig" / "company_ce" / "peid"


@dataclass(frozen=True)
class PeidConfig:
    input_path: str = str(DEFAULT_INPUT)
    output_dir: str = str(DEFAULT_CSV_DIR)
    figure_dir: str = str(DEFAULT_FIGURE_DIR)
    variables: tuple[str, ...] = DEFAULT_VARIABLES
    bins: int = 3
    max_source_order: int = 2
    alpha: float = 0.5
    min_source_count: int = 20
    min_total_count: int = 100
    null_reps: int = 20
    top_k: int = 12
    random_seed: int = 42
    year_start: int | None = None
    year_end: int | None = None
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate single-scale PEID causal edges and synergy hyperedges from discrete corporate dynamics."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--variables", nargs="+", default=list(DEFAULT_VARIABLES))
    parser.add_argument("--bins", type=int, default=3)
    parser.add_argument("--max-source-order", type=int, default=2, choices=(2, 3))
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--min-source-count", type=int, default=20)
    parser.add_argument("--min-total-count", type=int, default=100)
    parser.add_argument("--null-reps", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--year-start", type=int, default=None)
    parser.add_argument("--year-end", type=int, default=None)
    parser.add_argument("--winsor-lower", type=float, default=0.01)
    parser.add_argument("--winsor-upper", type=float, default=0.99)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--reuse-cache", action="store_true")
    return parser.parse_args()


def entropy_bits(probabilities: np.ndarray) -> float:
    probs = probabilities[probabilities > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


def variable_transform(series: pd.Series) -> str:
    nonmissing = series.dropna()
    if nonmissing.empty:
        return "missing"
    positive_share = float((nonmissing > 0).mean())
    return "log_growth" if positive_share >= 0.95 else "signed_relative_change"


def winsorize(series: pd.Series, lower: float, upper: float) -> pd.Series:
    finite = series[np.isfinite(series)]
    if finite.empty:
        return series
    lo, hi = finite.quantile([lower, upper]).to_numpy()
    return series.clip(lower=lo, upper=hi)


def build_growth_panel(
    input_path: Path,
    variables: Iterable[str],
    year_start: int | None = None,
    year_end: int | None = None,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = tuple(variables)
    usecols = ["gvkey", "fyear", *variables]
    df = pd.read_csv(input_path, usecols=usecols)
    df["gvkey"] = df["gvkey"].astype(str)
    df["fyear"] = df["fyear"].astype(int)
    if year_start is not None:
        df = df[df["fyear"] >= year_start]
    if year_end is not None:
        df = df[df["fyear"] <= year_end]

    audit_rows: list[dict[str, object]] = []
    growth_frames: list[pd.DataFrame] = []

    for variable in variables:
        panel = df[["gvkey", "fyear", variable]].dropna(subset=["gvkey", "fyear", variable]).copy()
        panel = panel[np.isfinite(panel[variable])]
        transform = variable_transform(panel[variable])
        n_raw = int(len(panel))
        n_positive = int((panel[variable] > 0).sum()) if n_raw else 0

        if transform == "log_growth":
            panel = panel[panel[variable] > 0].copy()
            panel["level"] = np.log(panel[variable])
            panel["growth"] = panel.groupby("gvkey")["level"].diff()
        elif transform == "signed_relative_change":
            panel = panel.sort_values(["gvkey", "fyear"]).copy()
            prev = panel.groupby("gvkey")[variable].shift(1)
            panel["growth"] = (panel[variable] - prev) / (prev.abs() + 1.0)
            panel["growth"] = winsorize(panel["growth"], winsor_lower, winsor_upper)
        else:
            panel["growth"] = np.nan

        panel = panel.sort_values(["gvkey", "fyear"])
        panel["prev_fyear"] = panel.groupby("gvkey")["fyear"].shift(1)
        panel["is_consecutive"] = (panel["fyear"] - panel["prev_fyear"]) == 1
        panel.loc[~panel["is_consecutive"], "growth"] = np.nan

        growth = panel.loc[panel["growth"].notna(), ["gvkey", "fyear", "growth"]].rename(
            columns={"fyear": "mid_year", "growth": variable}
        )
        growth_frames.append(growth)
        audit_rows.append(
            {
                "variable": variable,
                "transform": transform,
                "n_level_rows": n_raw,
                "n_positive_rows": n_positive,
                "n_growth_rows": int(len(growth)),
                "n_firms": int(growth["gvkey"].nunique()),
                "growth_mean": float(growth[variable].mean()) if len(growth) else np.nan,
                "growth_std": float(growth[variable].std()) if len(growth) else np.nan,
            }
        )

    wide = growth_frames[0]
    for frame in growth_frames[1:]:
        wide = wide.merge(frame, on=["gvkey", "mid_year"], how="inner")
    wide = wide.sort_values(["gvkey", "mid_year"]).reset_index(drop=True)

    target = wide.copy()
    target["mid_year"] = target["mid_year"] - 1
    pairs = wide.merge(target, on=["gvkey", "mid_year"], suffixes=("_src", "_tgt"), how="inner")
    pairs = pairs.sort_values(["gvkey", "mid_year"]).reset_index(drop=True)

    audit = pd.DataFrame(audit_rows)
    audit["n_complete_growth_rows"] = len(wide)
    audit["n_complete_transition_pairs"] = len(pairs)
    audit["n_complete_transition_firms"] = pairs["gvkey"].nunique() if len(pairs) else 0
    return pairs, audit


def assign_discrete_states(
    pairs: pd.DataFrame,
    variables: Iterable[str],
    bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = tuple(variables)
    states = pairs[["gvkey", "mid_year"]].copy()
    edge_rows: list[dict[str, object]] = []

    for variable in variables:
        source_col = f"{variable}_src"
        target_col = f"{variable}_tgt"
        values = pd.concat([pairs[source_col], pairs[target_col]], ignore_index=True)
        _, edges = pd.qcut(values, q=bins, labels=False, retbins=True, duplicates="drop")
        if len(edges) <= 2:
            raise ValueError(f"Variable {variable} has too few distinct values for {bins} bins.")
        edges[0] = -np.inf
        edges[-1] = np.inf
        labels = list(range(1, len(edges)))
        states[f"{variable}_src"] = pd.cut(pairs[source_col], bins=edges, labels=labels, include_lowest=True).astype(int)
        states[f"{variable}_tgt"] = pd.cut(pairs[target_col], bins=edges, labels=labels, include_lowest=True).astype(int)
        for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:]), start=1):
            edge_rows.append({"variable": variable, "bin": bin_index, "lower": lower, "upper": upper})

    return states, pd.DataFrame(edge_rows)


def effective_information(
    states: pd.DataFrame,
    source_cols: list[str],
    target_col: str,
    target_bins: int,
    alpha: float,
    min_source_count: int,
) -> dict[str, float]:
    grouped = states.groupby(source_cols + [target_col], observed=True).size().rename("count").reset_index()
    row_counts = grouped.groupby(source_cols, observed=True)["count"].sum().rename("row_total").reset_index()
    row_counts = row_counts[row_counts["row_total"] >= min_source_count]
    if len(row_counts) < 2:
        return {
            "ei": 0.0,
            "source_support": int(len(row_counts)),
            "total_count": int(row_counts["row_total"].sum()) if len(row_counts) else 0,
            "row_sum_error": 0.0,
        }

    filtered = grouped.merge(row_counts[source_cols], on=source_cols, how="inner")
    source_keys = [tuple(row[col] for col in source_cols) for _, row in row_counts.iterrows()]
    key_to_idx = {key: idx for idx, key in enumerate(source_keys)}
    counts = np.zeros((len(source_keys), target_bins), dtype=float)
    for row in filtered.itertuples(index=False):
        source_key = tuple(getattr(row, col) for col in source_cols)
        target_state = int(getattr(row, target_col))
        counts[key_to_idx[source_key], target_state - 1] += float(getattr(row, "count"))

    smoothed = counts + alpha
    probs = smoothed / smoothed.sum(axis=1, keepdims=True)
    row_sum_error = float(np.abs(probs.sum(axis=1) - 1.0).max())
    target_probs = probs.mean(axis=0)
    ei = entropy_bits(target_probs) - float(np.apply_along_axis(entropy_bits, 1, probs).mean())
    return {
        "ei": max(0.0, float(ei)),
        "source_support": int(len(source_keys)),
        "total_count": int(row_counts["row_total"].sum()),
        "row_sum_error": row_sum_error,
    }


def compute_pairwise_edges(states: pd.DataFrame, variables: Iterable[str], bins: int, alpha: float, min_source_count: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source, target in itertools.product(variables, variables):
        result = effective_information(
            states=states,
            source_cols=[f"{source}_src"],
            target_col=f"{target}_tgt",
            target_bins=bins,
            alpha=alpha,
            min_source_count=min_source_count,
        )
        rows.append({"source": source, "target": target, **result})
    return pd.DataFrame(rows).sort_values("ei", ascending=False).reset_index(drop=True)


def compute_synergy_hyperedges(
    states: pd.DataFrame,
    variables: Iterable[str],
    bins: int,
    alpha: float,
    min_source_count: int,
    max_source_order: int,
) -> pd.DataFrame:
    variables = tuple(variables)
    rows: list[dict[str, object]] = []
    single_cache: dict[tuple[str, str], float] = {}
    for source, target in itertools.product(variables, variables):
        single_cache[(source, target)] = effective_information(
            states,
            [f"{source}_src"],
            f"{target}_tgt",
            bins,
            alpha,
            min_source_count,
        )["ei"]

    for order in range(2, max_source_order + 1):
        for source_set in itertools.combinations(variables, order):
            source_cols = [f"{source}_src" for source in source_set]
            for target in variables:
                joint = effective_information(states, source_cols, f"{target}_tgt", bins, alpha, min_source_count)
                single_sum = float(sum(single_cache[(source, target)] for source in source_set))
                synergy_raw = float(joint["ei"] - single_sum)
                rows.append(
                    {
                        "sources": "+".join(source_set),
                        "target": target,
                        "source_order": order,
                        "joint_ei": joint["ei"],
                        "single_ei_sum": single_sum,
                        "synergy_raw": synergy_raw,
                        "synergy": max(0.0, synergy_raw),
                        "source_support": joint["source_support"],
                        "total_count": joint["total_count"],
                        "row_sum_error": joint["row_sum_error"],
                    }
                )
    return pd.DataFrame(rows).sort_values("synergy", ascending=False).reset_index(drop=True)


def null_target_shuffle(states: pd.DataFrame, variables: Iterable[str], rng: np.random.Generator) -> pd.DataFrame:
    shuffled = states.copy()
    for variable in variables:
        col = f"{variable}_tgt"
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
    return shuffled


def null_firm_time_shuffle(states: pd.DataFrame, variables: Iterable[str], rng: np.random.Generator) -> pd.DataFrame:
    shuffled = states.copy()
    for variable in variables:
        col = f"{variable}_tgt"
        pieces = []
        for _, group in shuffled.groupby("gvkey", sort=False):
            values = group[col].to_numpy().copy()
            rng.shuffle(values)
            pieces.append(pd.Series(values, index=group.index))
        shuffled[col] = pd.concat(pieces).sort_index()
    return shuffled


def empirical_nulls(
    states: pd.DataFrame,
    variables: Iterable[str],
    bins: int,
    alpha: float,
    min_source_count: int,
    max_source_order: int,
    null_reps: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    pair_rows: list[dict[str, object]] = []
    syn_rows: list[dict[str, object]] = []
    for null_type in ("target_shuffle", "firm_time_shuffle"):
        for rep in range(null_reps):
            null_states = (
                null_target_shuffle(states, variables, rng)
                if null_type == "target_shuffle"
                else null_firm_time_shuffle(states, variables, rng)
            )
            pairwise = compute_pairwise_edges(null_states, variables, bins, alpha, min_source_count)
            pairwise["null_type"] = null_type
            pairwise["rep"] = rep
            pair_rows.extend(pairwise[["source", "target", "ei", "null_type", "rep"]].to_dict("records"))
            synergy = compute_synergy_hyperedges(
                null_states,
                variables,
                bins,
                alpha,
                min_source_count,
                max_source_order,
            )
            synergy["null_type"] = null_type
            synergy["rep"] = rep
            syn_rows.extend(synergy[["sources", "target", "source_order", "synergy", "null_type", "rep"]].to_dict("records"))
    return pd.DataFrame(pair_rows), pd.DataFrame(syn_rows)


def attach_null_stats(
    observed: pd.DataFrame,
    nulls: pd.DataFrame,
    keys: list[str],
    value_col: str,
) -> pd.DataFrame:
    if nulls.empty:
        observed = observed.copy()
        observed["null_mean"] = np.nan
        observed["null_std"] = np.nan
        observed["z_score"] = np.nan
        observed["p_value"] = np.nan
        return observed

    rows: list[dict[str, object]] = []
    for key_values, obs_group in observed.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        mask = np.ones(len(nulls), dtype=bool)
        for key, value in zip(keys, key_values):
            mask &= nulls[key].to_numpy() == value
        null_values = nulls.loc[mask, value_col].to_numpy()
        obs = obs_group.iloc[0].to_dict()
        if null_values.size:
            obs["null_mean"] = float(null_values.mean())
            obs["null_std"] = float(null_values.std(ddof=1)) if null_values.size > 1 else 0.0
            denom = obs["null_std"] if obs["null_std"] > 0 else np.nan
            obs["z_score"] = float((obs[value_col] - obs["null_mean"]) / denom) if denom == denom else np.nan
            obs["p_value"] = float((1 + (null_values >= obs[value_col]).sum()) / (null_values.size + 1))
        else:
            obs["null_mean"] = np.nan
            obs["null_std"] = np.nan
            obs["z_score"] = np.nan
            obs["p_value"] = np.nan
        rows.append(obs)
    return pd.DataFrame(rows)


def compute_period_stability(
    states: pd.DataFrame,
    variables: Iterable[str],
    bins: int,
    alpha: float,
    min_source_count: int,
    top_pairwise: pd.DataFrame,
) -> pd.DataFrame:
    if states["mid_year"].nunique() < 4 or top_pairwise.empty:
        return pd.DataFrame()
    median_year = int(states["mid_year"].median())
    rows: list[pd.DataFrame] = []
    for period, subset in (
        ("early", states[states["mid_year"] <= median_year]),
        ("late", states[states["mid_year"] > median_year]),
    ):
        if len(subset) < 100:
            continue
        edges = compute_pairwise_edges(subset, variables, bins, alpha, min_source_count)
        edges["period"] = period
        rows.append(edges)
    if not rows:
        return pd.DataFrame()
    stability = pd.concat(rows, ignore_index=True)
    top_keys = set(zip(top_pairwise["source"], top_pairwise["target"]))
    stability = stability[[(source, target) in top_keys for source, target in zip(stability["source"], stability["target"])]]
    return stability.reset_index(drop=True)


def plot_outputs(output_dir: Path, figure_dir: Path, top_k: int) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    def configure_publication_style() -> font_manager.FontProperties | None:
        preferred_fonts = [
            "Arial Unicode MS",
            "PingFang SC",
            "Heiti SC",
            "Songti SC",
            "STHeiti",
            "Noto Sans CJK SC",
            "SimHei",
        ]
        installed = {font.name: font.fname for font in font_manager.fontManager.ttflist}
        chinese_font = next((name for name in preferred_fonts if name in installed), None)
        sans_fonts = [chinese_font] if chinese_font else []
        sans_fonts.extend(["Arial", "Helvetica", "DejaVu Sans", "sans-serif"])
        mpl.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": sans_fonts,
                "axes.spines.right": False,
                "axes.spines.top": False,
                "axes.linewidth": 0.8,
                "pdf.fonttype": 42,
                "svg.fonttype": "none",
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.unicode_minus": False,
            }
        )
        return font_manager.FontProperties(fname=installed[chinese_font]) if chinese_font else None

    def save_figure(fig, stem: str, dpi: int = 320) -> None:
        fig.savefig(figure_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")

    def variable_label(variable: str) -> str:
        labels = {
            "at": "at\n总资产",
            "revt": "revt\n营业收入",
            "emp": "emp\n员工数",
            "dltt": "dltt\n长期债务",
            "lt": "lt\n总负债",
            "ch": "ch\n现金",
        }
        return labels.get(variable, variable)

    def draw_box(ax, xy: tuple[float, float], width: float, height: float, label: str, face: str, font_prop) -> None:
        patch = FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=0.8,
            edgecolor="#293241",
            facecolor=face,
            zorder=3,
        )
        ax.add_patch(patch)
        ax.text(
            xy[0] + width / 2,
            xy[1] + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="#1f2933",
            fontproperties=font_prop,
            linespacing=1.25,
            zorder=4,
        )

    def draw_arrow(
        ax,
        start,
        end,
        width: float,
        color: str,
        rad: float,
        font_prop,
    ) -> None:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=width,
            color=color,
            alpha=0.78,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=4,
            shrinkB=4,
            zorder=2,
        )
        ax.add_patch(arrow)

    font_prop = configure_publication_style()

    figure_dir.mkdir(parents=True, exist_ok=True)
    pairwise = pd.read_csv(output_dir / "peid_pairwise_edges.csv")
    synergy = pd.read_csv(output_dir / "peid_synergy_hyperedges.csv")
    variables = sorted(set(pairwise["source"]).union(pairwise["target"]))

    matrix = pairwise.pivot(index="source", columns="target", values="ei").reindex(index=variables, columns=variables)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(), cmap="viridis")
    ax.set_xticks(range(len(variables)), variables, rotation=45, ha="right")
    ax.set_yticks(range(len(variables)), variables)
    ax.set_xlabel("Target variable at t+1")
    ax.set_ylabel("Source variable at t")
    fig.colorbar(image, ax=ax, label="Pairwise EI (bits)")
    save_figure(fig, "peid_pairwise_ei_heatmap", dpi=260)
    plt.close(fig)

    top_pairwise = pairwise.sort_values(["p_value", "ei"], ascending=[True, False]).head(top_k)
    fig, ax = plt.subplots(figsize=(9.4, max(5.4, 0.58 * len(variables) + 2.3)), constrained_layout=True)
    if not top_pairwise.empty:
        order_hint = {variable: idx for idx, variable in enumerate(DEFAULT_VARIABLES)}
        graph_vars = sorted(
            set(top_pairwise["source"]).union(top_pairwise["target"]),
            key=lambda value: (order_hint.get(value, 999), value),
        )
        y_positions = {variable: len(graph_vars) - idx - 1 for idx, variable in enumerate(graph_vars)}
        box_w, box_h = 0.18, 0.54
        left_x, right_x = 0.08, 0.74
        for variable in graph_vars:
            y = y_positions[variable]
            draw_box(ax, (left_x, y - box_h / 2), box_w, box_h, variable_label(variable), "#e8f1f7", font_prop)
            draw_box(ax, (right_x, y - box_h / 2), box_w, box_h, variable_label(variable), "#fff2dc", font_prop)

        weights = top_pairwise["ei"].to_numpy(dtype=float)
        max_weight = float(weights.max()) if weights.size else 1.0
        edge_color = "#4C78A8"
        for idx, row in enumerate(top_pairwise.itertuples()):
            y0 = y_positions[row.source]
            y1 = y_positions[row.target]
            rad = 0.08 * np.sign(y1 - y0) if y1 != y0 else 0.0
            if idx % 3 == 1:
                rad *= 1.45
            elif idx % 3 == 2:
                rad *= 0.65
            width = 0.8 + 4.0 * float(row.ei) / max_weight
            draw_arrow(
                ax,
                (left_x + box_w, y0),
                (right_x, y1),
                width,
                edge_color,
                rad,
                font_prop,
            )
        ax.text(0.17, len(graph_vars) - 0.15, "来源变量 t", ha="center", va="bottom", fontsize=10, fontweight="bold", fontproperties=font_prop)
        ax.text(0.83, len(graph_vars) - 0.15, "目标变量 t+1", ha="center", va="bottom", fontsize=10, fontweight="bold", fontproperties=font_prop)
        ax.text(0.50, -0.78, "线宽表示成对 EI 强度；边按经验零分布 p 值和 EI 排序；具体数值见证据汇总图。", ha="center", va="top", fontsize=7, color="#4b5563", fontproperties=font_prop)
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.0, len(graph_vars) - 0.02)
    else:
        ax.text(0.5, 0.5, "没有可用的成对因果边", ha="center", va="center", fontproperties=font_prop)
    ax.set_axis_off()
    save_figure(fig, "peid_top_pairwise_graph")
    plt.close(fig)

    top_synergy = synergy.sort_values(["p_value", "synergy"], ascending=[True, False]).head(top_k)
    fig, ax = plt.subplots(figsize=(10.4, max(5.2, 0.48 * len(top_synergy) + 2.2)), constrained_layout=True)
    if not top_synergy.empty:
        source_nodes = list(dict.fromkeys("{" + str(sources).replace("+", ", ") + "}" for sources in top_synergy["sources"]))
        target_nodes = list(dict.fromkeys(top_synergy["target"].tolist()))
        source_y = {node: len(source_nodes) - idx - 1 for idx, node in enumerate(source_nodes)}
        target_y = {
            node: (len(source_nodes) - 1) * (1 - idx / max(1, len(target_nodes) - 1)) if len(target_nodes) > 1 else (len(source_nodes) - 1) / 2
            for idx, node in enumerate(target_nodes)
        }
        box_w, box_h = 0.28, 0.38
        target_w, target_h = 0.18, 0.46
        left_x, right_x = 0.06, 0.77
        for node in source_nodes:
            draw_box(ax, (left_x, source_y[node] - box_h / 2), box_w, box_h, node, "#e8f1f7", font_prop)
        for node in target_nodes:
            draw_box(ax, (right_x, target_y[node] - target_h / 2), target_w, target_h, variable_label(node), "#fff2dc", font_prop)

        weights = top_synergy["synergy"].to_numpy(dtype=float)
        max_weight = float(weights.max()) if weights.size else 1.0
        for idx, row in enumerate(top_synergy.itertuples()):
            source_node = "{" + str(row.sources).replace("+", ", ") + "}"
            y0 = source_y[source_node]
            y1 = target_y[row.target]
            rad = 0.06 * np.sign(y1 - y0) if y1 != y0 else 0.0
            width = 0.8 + 4.5 * float(row.synergy) / max_weight
            draw_arrow(
                ax,
                (left_x + box_w, y0),
                (right_x, y1),
                width,
                "#D95F02",
                rad,
                font_prop,
            )
        ax.text(left_x + box_w / 2, len(source_nodes) - 0.12, "联合来源集合 t", ha="center", va="bottom", fontsize=10, fontweight="bold", fontproperties=font_prop)
        ax.text(right_x + target_w / 2, len(source_nodes) - 0.12, "目标变量 t+1", ha="center", va="bottom", fontsize=10, fontweight="bold", fontproperties=font_prop)
        ax.text(0.50, -0.78, "线宽表示截断正协同强度；协同衡量联合来源超过单变量 EI 加总的额外信息；具体数值见证据汇总图。", ha="center", va="top", fontsize=7, color="#4b5563", fontproperties=font_prop)
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.0, len(source_nodes) - 0.02)
    else:
        ax.text(0.5, 0.5, "没有可用的协同超边", ha="center", va="center", fontproperties=font_prop)
    ax.set_axis_off()
    save_figure(fig, "peid_top_synergy_hypergraph")
    plt.close(fig)

    all_combined = pd.concat(
        [
            pairwise.assign(edge_type="pairwise", label=pairwise["source"] + " -> " + pairwise["target"], value=pairwise["ei"]),
            synergy.assign(edge_type="synergy", label="{" + synergy["sources"].str.replace("+", ", ", regex=False) + "} -> " + synergy["target"], value=synergy["synergy"]),
        ],
        ignore_index=True,
        sort=False,
    )
    combined = all_combined.sort_values(["p_value", "value"], ascending=[True, False]).head(top_k)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(combined) + 2)), constrained_layout=True)
    y = np.arange(len(combined))
    ax.barh(y - 0.18, combined["value"], height=0.35, label="Observed", color="#4c78a8")
    ax.barh(y + 0.18, combined["null_mean"].fillna(0), height=0.35, label="Null mean", color="#cccccc")
    ax.set_yticks(y, combined["label"])
    ax.invert_yaxis()
    ax.set_xlabel("EI or synergy (bits)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save_figure(fig, "peid_null_comparison", dpi=260)
    plt.close(fig)

    summary = all_combined.copy()
    summary["excess_over_null"] = summary["value"] - summary["null_mean"].fillna(0.0)
    summary = summary.sort_values(["edge_type", "excess_over_null"], ascending=[True, False])
    pair_summary = summary[summary["edge_type"] == "pairwise"].head(min(top_k, 10))
    syn_summary = summary[summary["edge_type"] == "synergy"].head(min(top_k, 10))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(4.8, 0.36 * max(len(pair_summary), len(syn_summary)) + 2.2)), constrained_layout=True)
    for ax, frame, title, color, xlabel in (
        (axes[0], pair_summary, "成对跨期边", "#4C78A8", "EI - null mean (bits)"),
        (axes[1], syn_summary, "协同超边", "#D95F02", "synergy - null mean (bits)"),
    ):
        if frame.empty:
            ax.text(0.5, 0.5, "无可显示边", ha="center", va="center", fontproperties=font_prop)
            ax.set_axis_off()
            continue
        frame = frame.iloc[::-1]
        y = np.arange(len(frame))
        ax.barh(y, frame["excess_over_null"], color=color, alpha=0.86)
        ax.axvline(0, color="#4b5563", linewidth=0.8)
        labels = [
            f"{label}  (p={p_value:.3f})" if pd.notna(p_value) else str(label)
            for label, p_value in zip(frame["label"], frame["p_value"])
        ]
        ax.set_yticks(y, labels, fontsize=6.5)
        ax.set_title(title, fontsize=9, fontweight="bold", fontproperties=font_prop)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.6)
        ax.spines["left"].set_visible(False)
    fig.suptitle("因果证据汇总：观测信息量相对零分布的超额", fontsize=11, fontweight="bold", fontproperties=font_prop)
    save_figure(fig, "peid_causal_evidence_summary", dpi=300)
    plt.close(fig)

    stability_path = output_dir / "peid_period_stability.csv"
    if stability_path.exists() and stability_path.stat().st_size > 0:
        stability = pd.read_csv(stability_path)
        if not stability.empty:
            stability["edge"] = stability["source"] + " -> " + stability["target"]
            stable_matrix = stability.pivot(index="edge", columns="period", values="ei").fillna(0)
            fig, ax = plt.subplots(figsize=(6, max(5, 0.3 * len(stable_matrix) + 1)), constrained_layout=True)
            image = ax.imshow(stable_matrix.to_numpy(), cmap="magma")
            ax.set_xticks(range(len(stable_matrix.columns)), stable_matrix.columns)
            ax.set_yticks(range(len(stable_matrix.index)), stable_matrix.index)
            fig.colorbar(image, ax=ax, label="Pairwise EI (bits)")
            save_figure(fig, "peid_period_stability_heatmap", dpi=260)
            plt.close(fig)


def run_peid_analysis(config: PeidConfig, skip_figures: bool = False, reuse_cache: bool = False) -> dict[str, str]:
    output_dir = Path(config.output_dir)
    figure_dir = Path(config.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    expected = output_dir / "peid_pairwise_edges.csv"
    if reuse_cache and expected.exists():
        if not skip_figures:
            plot_outputs(output_dir, figure_dir, config.top_k)
        return {"output_dir": str(output_dir), "figure_dir": str(figure_dir), "reused_cache": "true"}

    pairs, audit = build_growth_panel(
        Path(config.input_path),
        config.variables,
        config.year_start,
        config.year_end,
        config.winsor_lower,
        config.winsor_upper,
    )
    if len(pairs) < config.min_total_count:
        raise ValueError(f"Only {len(pairs)} complete transition pairs were built; need at least {config.min_total_count}.")

    states, edges = assign_discrete_states(pairs, config.variables, config.bins)
    pairwise = compute_pairwise_edges(states, config.variables, config.bins, config.alpha, config.min_source_count)
    synergy = compute_synergy_hyperedges(
        states,
        config.variables,
        config.bins,
        config.alpha,
        config.min_source_count,
        config.max_source_order,
    )
    null_pairwise, null_synergy = empirical_nulls(
        states,
        config.variables,
        config.bins,
        config.alpha,
        config.min_source_count,
        config.max_source_order,
        config.null_reps,
        config.random_seed,
    )
    pairwise = attach_null_stats(pairwise, null_pairwise, ["source", "target"], "ei").sort_values(
        ["p_value", "ei"], ascending=[True, False]
    )
    synergy = attach_null_stats(synergy, null_synergy, ["sources", "target", "source_order"], "synergy").sort_values(
        ["p_value", "synergy"], ascending=[True, False]
    )
    top_pairwise = pairwise.head(config.top_k)
    stability = compute_period_stability(states, config.variables, config.bins, config.alpha, config.min_source_count, top_pairwise)

    audit.to_csv(output_dir / "peid_variable_audit.csv", index=False)
    edges.to_csv(output_dir / "peid_discretization_edges.csv", index=False)
    states.to_csv(output_dir / "peid_discrete_transition_states.csv", index=False)
    pairwise.to_csv(output_dir / "peid_pairwise_edges.csv", index=False)
    synergy.to_csv(output_dir / "peid_synergy_hyperedges.csv", index=False)
    null_pairwise.to_csv(output_dir / "peid_pairwise_null_samples.csv", index=False)
    null_synergy.to_csv(output_dir / "peid_synergy_null_samples.csv", index=False)
    stability.to_csv(output_dir / "peid_period_stability.csv", index=False)

    top_edges = pd.concat(
        [
            pairwise.head(config.top_k).assign(edge_type="pairwise", label=pairwise["source"] + " -> " + pairwise["target"]),
            synergy.head(config.top_k).assign(
                edge_type="synergy",
                label="{" + synergy["sources"].str.replace("+", ", ", regex=False) + "} -> " + synergy["target"],
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    top_edges.to_csv(output_dir / "peid_top_edges_for_figures.csv", index=False)
    with (output_dir / "peid_run_config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2, ensure_ascii=False)

    if not skip_figures:
        plot_outputs(output_dir, figure_dir, config.top_k)
    return {"output_dir": str(output_dir), "figure_dir": str(figure_dir), "reused_cache": "false"}


def main() -> None:
    args = parse_args()
    config = PeidConfig(
        input_path=str(args.input),
        output_dir=str(args.output_dir),
        figure_dir=str(args.figure_dir),
        variables=tuple(args.variables),
        bins=args.bins,
        max_source_order=args.max_source_order,
        alpha=args.alpha,
        min_source_count=args.min_source_count,
        min_total_count=args.min_total_count,
        null_reps=args.null_reps,
        top_k=args.top_k,
        random_seed=args.random_seed,
        year_start=args.year_start,
        year_end=args.year_end,
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
    )
    result = run_peid_analysis(config, skip_figures=args.skip_figures, reuse_cache=args.reuse_cache)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
