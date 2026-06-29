from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "reports" / "assets"

CURRENT_SAMPLE_COUNT = 8192
BASELINE_SAMPLE_COUNT = 4096
CURRENT_OVERALL_DIR = ROOT / "results" / f"unicm_overall_ei_cpu_bound4_n{CURRENT_SAMPLE_COUNT}"
CURRENT_SYN_DIR = ROOT / "results" / f"unicm_full_history_mode_pair_syn_cpu_bound4_n{CURRENT_SAMPLE_COUNT}"
BASELINE_SYN_DIR = ROOT / "results" / f"unicm_full_history_mode_pair_syn_cpu_bound4_n{BASELINE_SAMPLE_COUNT}"

OVERALL_ROWS = CURRENT_OVERALL_DIR / "overall_ei_rows.jsonl"
OVERALL_LEAD = CURRENT_OVERALL_DIR / "overall_ei_seed_lead_summary.csv"
OVERALL_ROBUST = CURRENT_OVERALL_DIR / "overall_ei_seed_robustness_summary.csv"
SYN_ROWS = CURRENT_SYN_DIR / "full_history_mode_pair_syn_rows.jsonl"
SYN_SUMMARY = CURRENT_SYN_DIR / "full_history_mode_pair_syn_summary.csv"
SYN_LEAD = CURRENT_SYN_DIR / "full_history_mode_pair_syn_lead_summary.csv"
BASELINE_SYN_SUMMARY = BASELINE_SYN_DIR / "full_history_mode_pair_syn_summary.csv"
BASELINE_SYN_LEAD = BASELINE_SYN_DIR / "full_history_mode_pair_syn_lead_summary.csv"
REPORT_PATH = ROOT / "docs" / "reports" / "log" / "unicm_modeformer_ei_syn_summary.md"

TARGETS = ["nino"]
SYN_TOP_K = 12
REQUIRED_PAIRS = ["NPMM|TNA", "nino|NPMM", "nino|TNA", "NPMM|nino3", "TNA|nino3"]
CONVERGENCE_PAIRS = ["nino|SPMM", "nino|nino3", "nino|NPMM", "nino|TNA", "NPMM|TNA"]
HIGHLIGHT_SOURCES = {"NPMM", "TNA", "SPMM", "WWV", "nino3", "nino4"}
DISPLAY_LABELS = {"nino": "ENSO"}

PAIR_LABELS = {
    "NPMM|TNA": "NPMM + TNA",
    "nino|NPMM": "ENSO + NPMM",
    "nino|TNA": "ENSO + TNA",
    "NPMM|nino3": "NPMM + nino3",
    "TNA|nino3": "TNA + nino3",
}

T_CRIT_DF2_95 = 4.302652729911275


def display_label(name: object) -> str:
    text = str(name)
    return DISPLAY_LABELS.get(text, text)


def display_pair(pair: object) -> str:
    return " + ".join(display_label(part) for part in str(pair).split("|"))


def target_asset_slug(target: str) -> str:
    if str(target) == "nino":
        return "enso"
    safe = "".join(char.lower() if char.isalnum() else "_" for char in str(target))
    return "_".join(part for part in safe.split("_") if part)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, base_name: str) -> tuple[Path, Path]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    png_path = ASSET_DIR / f"{base_name}.png"
    svg_path = ASSET_DIR / f"{base_name}.svg"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_paths = [OVERALL_ROWS, OVERALL_LEAD, OVERALL_ROBUST, SYN_ROWS, SYN_SUMMARY, SYN_LEAD]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required result table(s): " + ", ".join(str(path) for path in missing))

    overall_rows = pd.read_json(OVERALL_ROWS, lines=True)
    overall_lead = pd.read_csv(OVERALL_LEAD)
    overall_robust = pd.read_csv(OVERALL_ROBUST)
    syn_rows = pd.read_json(SYN_ROWS, lines=True)
    syn_summary = pd.read_csv(SYN_SUMMARY)
    syn_lead = pd.read_csv(SYN_LEAD)

    for frame_name, frame in {
        "overall_rows": overall_rows,
        "overall_lead": overall_lead,
        "overall_robust": overall_robust,
        "syn_rows": syn_rows,
        "syn_summary": syn_summary,
        "syn_lead": syn_lead,
    }.items():
        missing_targets = sorted(set(TARGETS) - set(frame["target"].astype(str)))
        if missing_targets:
            raise AssertionError(f"{frame_name} is missing target(s): {missing_targets}")

    for pair in REQUIRED_PAIRS:
        if pair not in set(syn_summary["pair"].astype(str)):
            raise AssertionError(f"Required pair is missing from Syn summary: {pair}")
        if pair not in set(syn_lead["pair"].astype(str)):
            raise AssertionError(f"Required pair is missing from Syn lead summary: {pair}")

    return overall_rows, overall_lead, overall_robust, syn_rows, syn_summary, syn_lead


def read_baseline_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    required_paths = [BASELINE_SYN_SUMMARY, BASELINE_SYN_LEAD]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing baseline result table(s): " + ", ".join(str(path) for path in missing))
    return pd.read_csv(BASELINE_SYN_SUMMARY), pd.read_csv(BASELINE_SYN_LEAD)


def compute_source_ei(syn_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in syn_summary.iterrows():
        rows.append({"target": row["target"], "source": row["left_source"], "ei": row["mean_left_ei"]})
        rows.append({"target": row["target"], "source": row["right_source"], "ei": row["mean_right_ei"]})
    source_ei = pd.DataFrame(rows)
    return (
        source_ei[source_ei["target"].isin(TARGETS)]
        .groupby(["target", "source"], as_index=False)["ei"]
        .mean()
        .sort_values(["target", "ei"], ascending=[True, False])
    )


def compute_source_ei_leads(syn_rows: pd.DataFrame, atol: float = 1e-10) -> pd.DataFrame:
    key_columns = ["target", "source", "seed", "lead"]
    left = syn_rows[["target", "left_source", "seed", "lead", "left_ei"]].rename(
        columns={"left_source": "source", "left_ei": "ei"}
    )
    right = syn_rows[["target", "right_source", "seed", "lead", "right_ei"]].rename(
        columns={"right_source": "source", "right_ei": "ei"}
    )
    long_rows = pd.concat([left, right], ignore_index=True)
    consistency = long_rows.groupby(key_columns, as_index=False)["ei"].agg(["min", "max"]).reset_index()
    inconsistent = consistency[(consistency["max"] - consistency["min"]).abs() > float(atol)]
    if not inconsistent.empty:
        first = inconsistent.iloc[0]
        raise ValueError(
            "Inconsistent single-source EI estimates for "
            f"target={first['target']}, source={first['source']}, seed={first['seed']}, lead={first['lead']}"
        )
    return (
        long_rows.groupby(key_columns, as_index=False)["ei"]
        .mean()
        .sort_values(key_columns)
        .reset_index(drop=True)
    )


def select_source_ei_curves(
    source_ei_leads: pd.DataFrame,
    target: str,
    top_k: int = 5,
    required_sources: tuple[str, ...] = ("NPMM", "TNA"),
) -> tuple[str, list[str]]:
    target_rows = source_ei_leads[source_ei_leads["target"].astype(str) == str(target)]
    available = set(target_rows["source"].astype(str))
    if target not in available:
        raise ValueError(f"Missing self source EI for target={target}")
    ranked = (
        target_rows[target_rows["source"].astype(str) != str(target)]
        .groupby("source", as_index=False)["ei"]
        .mean()
        .sort_values(["ei", "source"], ascending=[False, True])
    )
    selected = ranked.head(int(top_k))["source"].astype(str).tolist()
    for source in required_sources:
        if source in available and source not in selected:
            selected.append(source)
    return str(target), selected


def plot_overall_ei(overall_rows: pd.DataFrame, overall_lead: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(4.8 * len(TARGETS), 2.8), constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    colors = {1: "#4C78A8", 2: "#F58518", 3: "#54A24B"}
    for axis, target in zip(axes, TARGETS):
        target_rows = overall_rows[overall_rows["target"].astype(str) == target]
        for seed, seed_rows in target_rows.groupby("seed"):
            seed_rows = seed_rows.sort_values("lead")
            axis.plot(
                seed_rows["lead"],
                seed_rows["overall_ei"],
                marker="o",
                markersize=2.0,
                linewidth=1.0,
                color=colors.get(int(seed), "#777777"),
                alpha=0.55,
                label=f"Seed {int(seed)}",
            )
        mean_rows = overall_lead[overall_lead["target"].astype(str) == target].sort_values("lead")
        axis.plot(mean_rows["lead"], mean_rows["mean"], color="#111111", linewidth=1.8, label="Seed mean")
        axis.fill_between(
            mean_rows["lead"].to_numpy(dtype=float),
            (mean_rows["mean"] - mean_rows["std"]).to_numpy(dtype=float),
            (mean_rows["mean"] + mean_rows["std"]).to_numpy(dtype=float),
            color="#111111",
            alpha=0.12,
            linewidth=0,
        )
        axis.set_title(display_label(target), fontsize=7.5)
        axis.set_xlabel("Lead (months)")
        axis.set_ylabel("Overall EI (bits)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    return save_figure(fig, "unicm_enso_overall_ei_seed_overlay")[0]


def plot_source_ei(source_ei_leads: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.4, 3.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.9, 1.7]},
    )
    target = TARGETS[0]
    self_source, nonself_sources = select_source_ei_curves(source_ei_leads, target)
    source_summary = (
        source_ei_leads[source_ei_leads["target"].astype(str) == target]
        .groupby(["source", "lead"], as_index=False)["ei"]
        .agg(["mean", "std"])
        .reset_index()
    )

    self_axis, nonself_axis = axes
    self_rows = source_summary[source_summary["source"].astype(str) == self_source].sort_values("lead")
    self_x = self_rows["lead"].to_numpy(dtype=float)
    self_mean = self_rows["mean"].to_numpy(dtype=float)
    self_std = self_rows["std"].fillna(0).to_numpy(dtype=float)
    self_axis.plot(self_x, self_mean, color="#111111", marker="o", markersize=2.2, linewidth=1.7)
    self_axis.fill_between(
        self_x,
        self_mean - self_std,
        self_mean + self_std,
        color="#111111",
        alpha=0.12,
        linewidth=0,
    )
    self_axis.set_title("ENSO self", fontsize=7.5)
    self_axis.set_xlabel("Lead (months)")
    self_axis.set_ylabel("Single-source EI (bits)")

    palette = {
        "nino3": "#4C78A8",
        "nino12": "#72B7B2",
        "IOD": "#79706E",
        "SPMM": "#54A24B",
        "nino4": "#F58518",
        "NPMM": "#E45756",
        "TNA": "#B279A2",
    }
    fallback_colors = plt.get_cmap("tab10").colors
    for index, source in enumerate(nonself_sources):
        rows = source_summary[source_summary["source"].astype(str) == source].sort_values("lead")
        x = rows["lead"].to_numpy(dtype=float)
        mean = rows["mean"].to_numpy(dtype=float)
        std = rows["std"].fillna(0).to_numpy(dtype=float)
        color = palette.get(source, fallback_colors[index % len(fallback_colors)])
        nonself_axis.plot(
            x,
            mean,
            color=color,
            marker="o",
            markersize=2.0,
            linewidth=1.25,
            label=display_label(source),
        )
        nonself_axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.10, linewidth=0)
    nonself_axis.set_title("Non-self sources", fontsize=7.5)
    nonself_axis.set_xlabel("Lead (months)")
    nonself_axis.set_ylabel("Single-source EI (bits)")
    nonself_axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return save_figure(fig, "unicm_enso_source_ei_rankings")[0]


def select_syn_pairs(
    syn_summary: pd.DataFrame,
    target: str,
    top_k: int = SYN_TOP_K,
    required_pairs: list[str] | None = None,
) -> list[str]:
    target_rows = syn_summary[syn_summary["target"].astype(str) == target]
    top_pairs = target_rows.sort_values(["mean_syn", "pair"], ascending=[False, True]).head(int(top_k))["pair"].tolist()
    pairs: list[str] = []
    required = REQUIRED_PAIRS if required_pairs is None else required_pairs
    for pair in top_pairs + required:
        if pair in set(target_rows["pair"].astype(str)) and pair not in pairs:
            pairs.append(pair)
    return pairs


def plot_syn_leads(
    syn_summary: pd.DataFrame,
    syn_lead: pd.DataFrame,
    *,
    targets: list[str] | None = None,
    base_name: str | None = None,
    top_k: int = SYN_TOP_K,
) -> Path:
    target_names = TARGETS if targets is None else [str(target) for target in targets]
    if not target_names:
        raise ValueError("targets must contain at least one target.")
    if base_name is None:
        slug = target_asset_slug(target_names[0]) if len(target_names) == 1 else "multi_target"
        base_name = f"unicm_{slug}_mode_pair_syn_leads"
    fig, axes = plt.subplots(1, len(target_names), figsize=(6.4 * len(target_names), 3.8), constrained_layout=True, sharey=False)
    axes = np.atleast_1d(axes)
    palette = {
        "nino|SPMM": "#4C78A8",
        "nino|nino3": "#F58518",
        "nino|NPMM": "#E45756",
        "nino|TNA": "#B279A2",
        "NPMM|nino3": "#72B7B2",
        "TNA|nino3": "#54A24B",
        "NPMM|TNA": "#333333",
        "IOD|nino3": "#79706E",
        "nino12|nino3": "#9D755D",
        "SPMM|nino3": "#BAB0AC",
        "TNA|nino4": "#FF9DA6",
        "nino|nino12": "#A0CBE8",
        "nino|nino4": "#FFBE7D",
        "nino3|nino4": "#8CD17D",
        "SPMM|nino12": "#D4A6C8",
        "IOD|SIOD": "#B6992D",
        "nino|WWV": "#86BCB6",
        "nino|IOD": "#C85200",
        "NPMM|nino12": "#6B6ECF",
    }
    fallback_colors = plt.get_cmap("tab20").colors
    for axis, target in zip(axes, target_names):
        for color_index, pair in enumerate(select_syn_pairs(syn_summary, target, top_k=top_k)):
            subset = syn_lead[
                (syn_lead["target"].astype(str) == target) & (syn_lead["pair"].astype(str) == pair)
            ].sort_values("lead")
            if subset.empty:
                continue
            summary_row = syn_summary[
                (syn_summary["target"].astype(str) == target) & (syn_summary["pair"].astype(str) == pair)
            ].iloc[0]
            style = "--" if pair == "NPMM|TNA" else "-"
            width = 1.6 if pair in REQUIRED_PAIRS else 1.1
            color = palette.get(pair, fallback_colors[color_index % len(fallback_colors)])
            alpha = 1.0 if pair in REQUIRED_PAIRS or int(summary_row["rank_within_target"]) <= 5 else 0.78
            x = subset["lead"].to_numpy(dtype=float)
            mean = subset["mean"].to_numpy(dtype=float)
            axis.plot(
                x,
                mean,
                marker="o",
                markersize=2.0,
                linewidth=width,
                linestyle=style,
                color=color,
                alpha=alpha,
                label=PAIR_LABELS.get(pair, display_pair(pair)),
            )
            axis.axhline(
                float(summary_row["mean_syn"]),
                color=color,
                linewidth=0.55,
                linestyle=":",
                alpha=0.35,
            )
        axis.axhline(0, color="#888888", linewidth=0.7, linestyle=":")
        axis.set_title(display_label(target), fontsize=7.5)
        axis.set_xlabel("Lead (months)")
        axis.set_ylabel("Seed mean Syn (bits)")
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return save_figure(fig, base_name)[0]


def fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def fmt_p(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _two_sided_t_pvalue_df2(t_stat: float) -> float:
    t_abs = abs(float(t_stat))
    if not np.isfinite(t_abs):
        return float("nan")
    return float(1.0 - t_abs / np.sqrt(t_abs * t_abs + 2.0))


def _bh_qvalues(pvalues: pd.Series) -> pd.Series:
    values = pvalues.astype(float)
    finite = values[np.isfinite(values)]
    qvalues = pd.Series(np.nan, index=values.index, dtype=float)
    if finite.empty:
        return qvalues
    ordered = finite.sort_values()
    m = float(len(ordered))
    raw = ordered.to_numpy(dtype=float) * m / np.arange(1, len(ordered) + 1, dtype=float)
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    qvalues.loc[ordered.index] = adjusted
    return qvalues


def compute_syn_seed_stats(syn_summary: pd.DataFrame, syn_rows: pd.DataFrame) -> pd.DataFrame:
    seed_means = (
        syn_rows[syn_rows["target"].astype(str).isin(TARGETS)]
        .groupby(["target", "pair", "seed"], as_index=False)["syn"]
        .mean()
        .rename(columns={"syn": "seed_mean_syn"})
    )
    per_pair = (
        seed_means.groupby(["target", "pair"], as_index=False)
        .agg(
            seed_mean_syn=("seed_mean_syn", "mean"),
            syn_seed_sd=("seed_mean_syn", "std"),
            n_seeds=("seed", "nunique"),
            positive_seed_count=("seed_mean_syn", lambda values: int((values > 0).sum())),
        )
    )
    rank_rows = seed_means.copy()
    rank_rows["seed_rank"] = rank_rows.groupby(["target", "seed"])["seed_mean_syn"].rank(
        method="min", ascending=False
    )
    rank_summary = (
        rank_rows.groupby(["target", "pair"], as_index=False)
        .agg(
            seed_rank_min=("seed_rank", "min"),
            seed_rank_max=("seed_rank", "max"),
            seed_rank_mean=("seed_rank", "mean"),
        )
    )
    stats = syn_summary.merge(per_pair, on=["target", "pair"], how="left").merge(
        rank_summary, on=["target", "pair"], how="left"
    )
    stats["syn_seed_sd"] = stats["syn_seed_sd"].fillna(0.0)
    stats["syn_seed_sem"] = stats["syn_seed_sd"] / np.sqrt(stats["n_seeds"].astype(float))
    stats["syn_ci_low"] = stats["mean_syn"] - T_CRIT_DF2_95 * stats["syn_seed_sem"]
    stats["syn_ci_high"] = stats["mean_syn"] + T_CRIT_DF2_95 * stats["syn_seed_sem"]
    stats["t_vs_zero"] = np.where(stats["syn_seed_sem"] > 0, stats["mean_syn"] / stats["syn_seed_sem"], np.nan)
    stats["p_vs_zero"] = stats["t_vs_zero"].map(_two_sided_t_pvalue_df2)
    zero_sem = stats["syn_seed_sem"] == 0
    stats.loc[zero_sem & (stats["mean_syn"].abs() > 0), "p_vs_zero"] = 0.0
    stats.loc[zero_sem & (stats["mean_syn"].abs() == 0), "p_vs_zero"] = 1.0
    stats["q_bh_vs_zero"] = stats.groupby("target", group_keys=False)["p_vs_zero"].apply(_bh_qvalues)
    return stats


def compute_sampling_convergence(
    baseline_summary: pd.DataFrame,
    baseline_lead: pd.DataFrame,
    current_summary: pd.DataFrame,
    current_lead: pd.DataFrame,
    current_rows: pd.DataFrame,
    *,
    pairs: list[str] = CONVERGENCE_PAIRS,
) -> pd.DataFrame:
    def summary_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return frame[["target", "pair", "mean_syn", "rank_within_target"]].rename(
            columns={"mean_syn": f"{prefix}_mean_syn", "rank_within_target": f"{prefix}_rank"}
        )

    def lead_seed_sd(frame: pd.DataFrame, column: str) -> pd.DataFrame:
        return (
            frame.groupby(["target", "pair"], as_index=False)["std"]
            .mean()
            .rename(columns={"std": column})
        )

    bootstrap = (
        current_rows.groupby(["target", "pair"], as_index=False)["syn_bootstrap_std"]
        .mean()
        .rename(columns={"syn_bootstrap_std": "current_mean_bootstrap_sd"})
    )
    comparison = (
        summary_columns(baseline_summary, "baseline")
        .merge(summary_columns(current_summary, "current"), on=["target", "pair"], validate="one_to_one")
        .merge(
            lead_seed_sd(baseline_lead, "baseline_mean_lead_seed_sd"),
            on=["target", "pair"],
            validate="one_to_one",
        )
        .merge(
            lead_seed_sd(current_lead, "current_mean_lead_seed_sd"),
            on=["target", "pair"],
            validate="one_to_one",
        )
        .merge(bootstrap, on=["target", "pair"], validate="one_to_one")
    )
    comparison = comparison[
        (comparison["target"].astype(str) == "nino") & comparison["pair"].astype(str).isin(pairs)
    ].copy()
    missing_pairs = [pair for pair in pairs if pair not in set(comparison["pair"].astype(str))]
    if missing_pairs:
        raise AssertionError("Missing convergence pair(s): " + ", ".join(missing_pairs))
    comparison["mean_syn_delta"] = comparison["current_mean_syn"] - comparison["baseline_mean_syn"]
    comparison["seed_sd_ratio"] = np.where(
        comparison["baseline_mean_lead_seed_sd"] > 0,
        comparison["current_mean_lead_seed_sd"] / comparison["baseline_mean_lead_seed_sd"],
        np.nan,
    )
    order = {pair: index for index, pair in enumerate(pairs)}
    comparison["_order"] = comparison["pair"].map(order)
    return comparison.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def markdown_sampling_convergence(comparison: pd.DataFrame) -> str:
    lines = [
        "图 3 不再绘制误差棒；下表中的 `checkpoint SD` 仍定义为 3 个 checkpoint 的 lead-wise 标准差。`bootstrap SD` 单独估计固定 checkpoint 下最大熵采样的 Monte Carlo 波动，不与 checkpoint 间差异混用。",
        "",
        f"| Source pair | mean Syn {BASELINE_SAMPLE_COUNT} | mean Syn {CURRENT_SAMPLE_COUNT} | rank {BASELINE_SAMPLE_COUNT}→{CURRENT_SAMPLE_COUNT} | mean checkpoint SD {BASELINE_SAMPLE_COUNT} | mean checkpoint SD {CURRENT_SAMPLE_COUNT} | SD ratio | mean bootstrap SD {CURRENT_SAMPLE_COUNT} |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples():
        lines.append(
            f"| {display_pair(row.pair)} | {fmt(row.baseline_mean_syn)} | "
            f"{fmt(row.current_mean_syn)} | {int(row.baseline_rank)}→{int(row.current_rank)} | "
            f"{fmt(row.baseline_mean_lead_seed_sd)} | {fmt(row.current_mean_lead_seed_sd)} | "
            f"{fmt(row.seed_sd_ratio, 3)} | {fmt(row.current_mean_bootstrap_sd)} |"
        )
    reduced = int((comparison["seed_sd_ratio"] < 1.0).sum())
    lines.extend(
        [
            "",
            f"固定比较的 5 个 pair 中，`{reduced}/5` 个 pair 的平均 lead-wise checkpoint SD 在 {CURRENT_SAMPLE_COUNT} 样本下下降。"
            "该比例只说明采样收敛减少了部分估计噪声；剩余 checkpoint SD 仍包含真实的 checkpoint 机制差异。",
        ]
    )
    return "\n".join(lines)


def markdown_overall_table(overall_robust: pd.DataFrame) -> str:
    lines = [
        "| Target | mean EI 1..24 | mean EI 6..18 | Pearson min | Spearman min | top-3 overlap min |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        row = overall_robust[overall_robust["target"].astype(str) == target].iloc[0]
        lines.append(
            f"| {display_label(target)} | {fmt(row['mean_overall_ei_full'])} | {fmt(row['mean_overall_ei_climate'])} | "
            f"{fmt(row['pearson_min'], 3)} | {fmt(row['spearman_min'], 3)} | {int(row['top3_overlap_min'])} |"
        )
    return "\n".join(lines)


def markdown_source_table(source_ei: pd.DataFrame) -> str:
    lines = [
        "| Target | self EI | strongest non-self sources | NPMM EI | TNA EI |",
        "|---|---:|---|---:|---:|",
    ]
    for target in TARGETS:
        subset = source_ei[source_ei["target"] == target].sort_values("ei", ascending=False)
        self_ei = subset[subset["source"] == target]["ei"].iloc[0]
        non_self = subset[subset["source"] != target].head(4)
        source_text = "; ".join(f"{row.source} {fmt(row.ei)}" for row in non_self.itertuples())
        npmm = subset[subset["source"] == "NPMM"]["ei"].iloc[0]
        tna = subset[subset["source"] == "TNA"]["ei"].iloc[0]
        lines.append(f"| {display_label(target)} | {fmt(self_ei)} | {source_text} | {fmt(npmm)} | {fmt(tna)} |")
    return "\n".join(lines)


def markdown_syn_table_guide() -> str:
    return "\n".join(
        [
            "这张表的读法是：先用 `mean Syn 1..24` 和 `rank` 找候选 pair，再用区间与 seed 稳定性判断这个候选能不能信。",
            "",
            "- `Target`：被解释的目标变量。这里是 ENSO，也就是看 source pair 对未来 ENSO 的影响。",
            "- `Source pair`：两个输入源的组合，例如 `ENSO + SPMM`。它回答这两个源一起看时，是否比各自单独看多提供了额外信息。",
            "- `rank`：按 `mean Syn 1..24` 从大到小排序。rank 越靠前，平均 Syn 越大；但 rank 只看均值，不等于显著或稳定。",
            "- `mean Syn 1..24`：1 到 24 个月 lead 上的平均 Syn。越大表示二源组合的额外协同增益越强；接近 0 表示几乎没有协同；负值表示联合读数低于两个单源读数之和。",
            "- `Syn seed SD`：3 个 checkpoint seed 的 pair-level Syn 标准差。越大表示不同模型 seed 之间越不稳定。",
            "- `95% CI`：基于 3 个 seed 的 Syn 均值 95% 置信区间。区间跨过 0 时，不能稳妥地说该 pair 一定有正协同。这里 `n=3` 很小，所以 CI 只作为 sanity check。",
            "- `seed rank range`：每个 seed 单独排序后，该 pair 的 rank 范围。范围越窄，排名越稳定；范围很宽说明 rank 对 checkpoint seed 敏感。",
            "- `joint EI 1..24`：两个 source 合起来对 target 的平均 EI，也就是联合输入总共携带多少目标信息。",
            "- `left EI 1..24`：source pair 左边那个源单独对 target 的平均 EI。",
            "- `right EI 1..24`：source pair 右边那个源单独对 target 的平均 EI。",
            "",
            "Syn 的计算关系是 `Syn = joint EI - left EI - right EI`。因此，Syn 是几个 EI 项相减后的剩余量；即使 EI 本身较大，Syn 也可能很小，并且会更容易受 checkpoint seed 差异影响。",
        ]
    )


def markdown_syn_table(syn_stats: pd.DataFrame) -> str:
    rows: list[pd.Series] = []
    for target in TARGETS:
        target_rows = syn_stats[syn_stats["target"].astype(str) == target]
        keep = set(select_syn_pairs(syn_stats, target))
        rows.extend([row for _, row in target_rows[target_rows["pair"].isin(keep)].iterrows()])
    table = pd.DataFrame(rows).sort_values(["target", "rank_within_target"])
    lines = [
        "| Target | Source pair | rank | mean Syn 1..24 | Syn seed SD | 95% CI | seed rank range | joint EI 1..24 | left EI 1..24 | right EI 1..24 |",
        "|---|---|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for row in table.itertuples():
        ci = f"[{fmt(row.syn_ci_low)}, {fmt(row.syn_ci_high)}]"
        seed_rank_range = f"{int(row.seed_rank_min)}-{int(row.seed_rank_max)}"
        lines.append(
            f"| {display_label(row.target)} | {display_label(row.left_source)} + {display_label(row.right_source)} | {int(row.rank_within_target)} | "
            f"{fmt(row.mean_syn)} | {fmt(row.syn_seed_sd)} | {ci} | {seed_rank_range} | "
            f"{fmt(row.mean_joint_ei)} | {fmt(row.mean_left_ei)} | {fmt(row.mean_right_ei)} |"
        )
    return "\n".join(lines)


def build_syn_reliability_notes(syn_stats: pd.DataFrame) -> str:
    target_rows = syn_stats[syn_stats["target"].astype(str) == "nino"].copy()
    selected = target_rows[target_rows["pair"].isin(select_syn_pairs(syn_stats, "nino"))].sort_values(
        "rank_within_target"
    )
    top5 = selected.head(5)
    positive_top5 = int((top5["positive_seed_count"] == top5["n_seeds"]).sum())
    uncorrected = selected[selected["p_vs_zero"] < 0.05]["pair"].astype(str).tolist()
    corrected = selected[selected["q_bh_vs_zero"] < 0.05]["pair"].astype(str).tolist()
    top_pair = target_rows.sort_values("rank_within_target").iloc[0]
    spmm_pair = target_rows[target_rows["pair"].astype(str) == "nino|SPMM"].iloc[0]
    npmm_pair = target_rows[target_rows["pair"].astype(str) == "nino|NPMM"].iloc[0]
    tna_pair = target_rows[target_rows["pair"].astype(str) == "nino|TNA"].iloc[0]
    npmm_tna = target_rows[target_rows["pair"].astype(str) == "NPMM|TNA"].iloc[0]
    uncorrected_text = ", ".join(display_pair(pair) for pair in uncorrected) if uncorrected else "none"
    corrected_text = ", ".join(display_pair(pair) for pair in corrected) if corrected else "none"
    return "\n".join(
        [
            f"- Top-5 pair 中 `{positive_top5}/5` 个在 3 个 checkpoint seed 上均为正；这支持 rank 的方向性，但样本数只有 `n=3`。",
            f"- 未校正 one-sample t test（跨 3 个 seed 的 pair-mean Syn vs 0）达到 `p<0.05` 的 selected pair: `{uncorrected_text}`；Benjamini-Hochberg 校正后达到 `q<0.05` 的 selected pair: `{corrected_text}`。",
            f"- `{display_pair(top_pair.pair)}` 是当前 rank 1，平均 Syn 为 `{fmt(top_pair.mean_syn)}`，"
            f"95% CI 为 `[{fmt(top_pair.syn_ci_low)}, {fmt(top_pair.syn_ci_high)}]`，"
            f"seed rank range 为 `{int(top_pair.seed_rank_min)}-{int(top_pair.seed_rank_max)}`，"
            f"正值 seed 数为 `{int(top_pair.positive_seed_count)}/{int(top_pair.n_seeds)}`。",
            f"- `ENSO + SPMM` 当前为 rank `{int(spmm_pair.rank_within_target)}`，平均 Syn 为 `{fmt(spmm_pair.mean_syn)}`，"
            f"95% CI 为 `[{fmt(spmm_pair.syn_ci_low)}, {fmt(spmm_pair.syn_ci_high)}]`；它仍是重要候选，但不再适合作为唯一主导机制来表述。",
            f"- `ENSO + NPMM` 与 `ENSO + TNA` 分别为 rank `{int(npmm_pair.rank_within_target)}` 和 rank `{int(tna_pair.rank_within_target)}`；"
            f"二者方向均为正，但平均 Syn 较 top pair 明显更小，更适合解释为 ENSO 背景态上的弱到中等调制信号。",
            f"- `NPMM + TNA` 直接二源 Syn 的均值为 `{fmt(npmm_tna.mean_syn)}`，95% CI 为 `[{fmt(npmm_tna.syn_ci_low)}, {fmt(npmm_tna.syn_ci_high)}]`，且 seed rank range 为 `{int(npmm_tna.seed_rank_min)}-{int(npmm_tna.seed_rank_max)}`；这不足以支持强直接协同。",
        ]
    )


def pair_value(syn_summary: pd.DataFrame, target: str, pair: str, column: str = "mean_syn") -> float:
    row = syn_summary[(syn_summary["target"].astype(str) == target) & (syn_summary["pair"].astype(str) == pair)]
    if row.empty:
        raise AssertionError(f"Missing {pair} for {target}")
    return float(row.iloc[0][column])


def build_report(
    overall_robust: pd.DataFrame,
    source_ei: pd.DataFrame,
    syn_summary: pd.DataFrame,
    syn_stats: pd.DataFrame,
    convergence: pd.DataFrame,
    figure_paths: dict[str, Path],
) -> str:
    nino_npmm = pair_value(syn_summary, "nino", "nino|NPMM")
    nino_tna = pair_value(syn_summary, "nino", "nino|TNA")
    nino_npmm_tna = pair_value(syn_summary, "nino", "NPMM|TNA")
    nino_npmm_rank = int(pair_value(syn_summary, "nino", "nino|NPMM", "rank_within_target"))
    nino_tna_rank = int(pair_value(syn_summary, "nino", "nino|TNA", "rank_within_target"))
    top_pair_text = "、".join(
        display_pair(pair)
        for pair in syn_summary[syn_summary["target"].astype(str) == "nino"]
        .sort_values("rank_within_target")
        .head(5)["pair"]
    )
    markdown = dedent(
        f"""
        # UniCM ENSO target EI/Syn 证据报告

        ## 结论

        原文命题认为，在强厄尔尼诺事件发生前，NPMM 和 TNA 与 ENSO 的相互作用增强。这里仅保留 ENSO target：以 UniCM Modeformer 的 full-history maximum-entropy 机制读数来看，{CURRENT_SAMPLE_COUNT} 样本后的主信号更集中在 ENSO 自身历史与区域 ENSO 指数/太平洋模态的组合上。NPMM/TNA 仍有可见读数，但不再是最强协同候选；`NPMM + TNA` 两个外部 mode 单独构成强二源协同驱动的说法缺乏支持。

        本轮使用 `{CURRENT_SAMPLE_COUNT}` 个最大熵干预样本和 `200` 次 bootstrap；checkpoint seeds、干预范围与 `{BASELINE_SAMPLE_COUNT}`-sample baseline 保持一致。当前 top-5 为 `{top_pair_text}`。具体到远程模态，`ENSO + NPMM` 的平均 Syn 为 `{fmt(nino_npmm)}` bits（rank `{nino_npmm_rank}`），`ENSO + TNA` 为 `{fmt(nino_tna)}` bits（rank `{nino_tna_rank}`）；`NPMM + TNA` 直接到 ENSO 的平均 Syn 为 `{fmt(nino_npmm_tna)}` bits。由此得到的新 insight 是：NPMM/TNA 对 ENSO 的贡献不宜表述为两个远程 mode 本身的强协同，而应降级为弱到中等的背景态调制信号；当前最稳妥的主结论是 ENSO 自身历史与赤道太平洋区域结构共同控制了主要 Syn 增益。

        ## EI 证据：ENSO target 主要携带短中期记忆

        {markdown_overall_table(overall_robust)}

        ![ENSO overall EI](assets/{figure_paths['overall'].name})

        *图 1. ENSO target 的 full-history overall EI lead 曲线（最大熵样本 `n={CURRENT_SAMPLE_COUNT}`）。彩色细线为 checkpoint seed，黑线为 seed mean，阴影为 seed standard deviation。*

        这个趋势说明，UniCM learned mechanism 对 ENSO target 的有效信息主要集中在短中期。ENSO 在 lead 1 到 6 个月的 EI 明显高于后期，符合 ENSO 预测中短期记忆强、长期不确定性上升的物理直觉。

        ## 单源 EI：NPMM 是更稳定的远程信号，TNA 更像弱调制项

        {markdown_source_table(source_ei)}

        ![ENSO source EI lead curves](assets/{figure_paths['source'].name})

        *图 2. ENSO target 的单源 EI lead 曲线。左图单独显示 ENSO self source；右图显示按 lead 平均 EI 选出的非自身 Top-5，并保留 NPMM/TNA。实线和浅色带分别为 checkpoint seed mean 和 standard deviation。*

        单源 EI 显示，ENSO 自身区域指数仍是主要信息来源。排除自身后，NPMM 仍处于前列，说明北太平洋经向模态在 UniCM 中携带 ENSO 输出的远程信息。TNA 的单源 EI 较小，说明它不是最强的独立 ENSO source；它更可能通过与 ENSO 背景态或其他太平洋 mode 的组合产生可见影响。

        ## Syn 证据：主要增益来自 ENSO 自身历史与区域 ENSO 结构

        {markdown_syn_table_guide()}

        {markdown_syn_table(syn_stats)}

        ![ENSO mode-pair Syn leads](assets/{figure_paths['syn'].name})

        *图 3. ENSO target 的 mode-pair Syn lead 曲线（最大熵样本 `n={CURRENT_SAMPLE_COUNT}`）。实线为每个 lead 的 seed mean；同色浅虚线为该 pair 在 lead 1..24 上的平均 Syn，对应上表 `mean Syn 1..24`。黑色虚线为 `NPMM + TNA` 直接二源 Syn。为突出 lead 结构，本图不绘制 checkpoint seed standard deviation。*

        对 ENSO target，当前 top-5 是 `{top_pair_text}`。这说明 {CURRENT_SAMPLE_COUNT} 样本下主导的不是单个远程模态，而是 ENSO 自身历史与赤道太平洋区域结构、南太平洋模态及印度洋背景态共同形成的协同增益。NPMM 仍保留一定调制信号，TNA 则更弱；`NPMM + TNA` 直接二源 Syn 接近零，不能作为强协同驱动的主证据。

        用地球科学的话说，图 3 的意思很朴素：模型不是只看“ENSO 现在有多强”，还在看“暖异常更偏东、偏中太平洋，还是和其他海盆背景态一起出现”。前 1 到 7 个月，`ENSO + nino3` 和 `ENSO + nino4` 的 Syn 明显更高，说明 ENSO 的短期未来演变对赤道太平洋东西向 SST 结构很敏感；同样强度的 ENSO，如果空间型态不同，后面几个月的增长、衰减和位相演变也可能不同。这个解释和 ENSO diversity 文献是一致的：Trenberth and Stepaniak [1] 早就指出，单一 ENSO 指数不足以描述事件演变，需要额外刻画中东太平洋 SST 梯度；Capotondi et al. [2] 也把事件间差异总结为 ENSO 的振幅、空间型态、生命周期和触发机制差异。Ren and Jin [3] 进一步用 Niño3/Niño4 组合区分两类 ENSO，Kao and Yu [4] 与 Ashok et al. [5] 则分别从 EP/CP ENSO 和 ENSO Modoki 的角度说明，中太平洋型和东太平洋型事件不能简单当作同一种 ENSO 强度的线性放大。因此，这里的 `nino3` 和 `nino4` 更适合被解释为 ENSO 内部空间型态的调制因子，而不是 ENSO 之外的独立强迫源。曲线在 9 到 12 个月后整体贴近零，说明这种额外协同信息主要集中在短中期；到更长 lead，模型已经很难从这些二源组合里读出稳定的增量。

        ## Rank 与趋势可信度

        {build_syn_reliability_notes(syn_stats)}

        因为只有 3 个 checkpoint seed，显著性检验的自由度只有 2；这里的 p 值和 CI 只能作为 sanity check，不能替代更多 checkpoint 或更高样本的复核。rank 可信度更适合用三件事一起判断：跨 seed 是否同号、seed rank range 是否窄、lead 曲线是否在关键窗口保持同方向。

        ## {BASELINE_SAMPLE_COUNT}→{CURRENT_SAMPLE_COUNT} 样本收敛

        {markdown_sampling_convergence(convergence)}

        ## 解释边界

        本报告只分析 frozen UniCM Modeformer learned mechanism，不是 reanalysis 事件复现实验，也不是 1983 或 1997 个例归因。这里的 EI/Syn 使用 Gaussian log-det 估计，适合作为 full-history 机制筛查；若要把结论推进到最终 PEID 或事件级归因，需要进一步做非线性 transport-map PEID、高样本复核，或按厄尔尼诺事件窗口构造条件化干预。

        ## 参考文献

        [1] Trenberth, K. E., & Stepaniak, D. P. (2001). Indices of El Niño Evolution. *Journal of Climate*, 14(8), 1697-1701. https://doi.org/10.1175/1520-0442(2001)014%3C1697:LIOENO%3E2.0.CO;2

        [2] Capotondi, A., Wittenberg, A. T., Newman, M., Di Lorenzo, E., Yu, J.-Y., Braconnot, P., Cole, J., Dewitte, B., Giese, B., Guilyardi, E., Jin, F.-F., Karnauskas, K., Kirtman, B., Lee, T., Schneider, N., Xue, Y., & Yeh, S.-W. (2015). Understanding ENSO Diversity. *Bulletin of the American Meteorological Society*, 96(6), 921-938. https://doi.org/10.1175/BAMS-D-13-00117.1

        [3] Ren, H.-L., & Jin, F.-F. (2011). Niño indices for two types of ENSO. *Geophysical Research Letters*, 38, L04704. https://doi.org/10.1029/2010GL046031

        [4] Kao, H.-Y., & Yu, J.-Y. (2009). Contrasting Eastern-Pacific and Central-Pacific Types of ENSO. *Journal of Climate*, 22(3), 615-632. https://doi.org/10.1175/2008JCLI2309.1

        [5] Ashok, K., Behera, S. K., Rao, S. A., Weng, H., & Yamagata, T. (2007). El Niño Modoki and its possible teleconnection. *Journal of Geophysical Research: Oceans*, 112, C11007. https://doi.org/10.1029/2006JC003798
        """
    ).strip()
    markdown = "\n".join(line[8:] if line.startswith("        ") else line for line in markdown.splitlines())
    return markdown + "\n"


def write_report(path: Path, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def main() -> int:
    configure_matplotlib()
    overall_rows, overall_lead, overall_robust, syn_rows, syn_summary, syn_lead = read_inputs()
    baseline_summary, baseline_lead = read_baseline_inputs()
    source_ei = compute_source_ei(syn_summary)
    source_ei_leads = compute_source_ei_leads(syn_rows)
    syn_stats = compute_syn_seed_stats(syn_summary, syn_rows)
    convergence = compute_sampling_convergence(
        baseline_summary,
        baseline_lead,
        syn_summary,
        syn_lead,
        syn_rows,
    )

    figures = {
        "overall": plot_overall_ei(overall_rows, overall_lead),
        "source": plot_source_ei(source_ei_leads),
        "syn": plot_syn_leads(syn_summary, syn_lead),
    }

    for path in figures.values():
        print(f"wrote {path.relative_to(ROOT)}")
    write_report(REPORT_PATH, build_report(overall_robust, source_ei, syn_summary, syn_stats, convergence, figures))
    print(f"wrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
