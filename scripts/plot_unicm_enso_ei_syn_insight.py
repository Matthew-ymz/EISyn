from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "reports" / "assets"

OVERALL_ROWS = ROOT / "results" / "unicm_overall_ei_cpu_bound4_n1024" / "overall_ei_rows.jsonl"
OVERALL_LEAD = ROOT / "results" / "unicm_overall_ei_cpu_bound4_n1024" / "overall_ei_seed_lead_summary.csv"
OVERALL_ROBUST = ROOT / "results" / "unicm_overall_ei_cpu_bound4_n1024" / "overall_ei_seed_robustness_summary.csv"
SYN_SUMMARY = ROOT / "results" / "unicm_full_history_mode_pair_syn_cpu_bound4_n1024" / "full_history_mode_pair_syn_summary.csv"
SYN_LEAD = ROOT / "results" / "unicm_full_history_mode_pair_syn_cpu_bound4_n1024" / "full_history_mode_pair_syn_lead_summary.csv"

TARGETS = ["nino", "nino3"]
REQUIRED_PAIRS = ["NPMM|TNA", "nino|NPMM", "nino|TNA", "NPMM|nino3", "TNA|nino3"]
HIGHLIGHT_SOURCES = {"NPMM", "TNA", "SPMM", "WWV", "nino3", "nino4"}

PAIR_LABELS = {
    "NPMM|TNA": "NPMM + TNA",
    "nino|NPMM": "nino + NPMM",
    "nino|TNA": "nino + TNA",
    "NPMM|nino3": "NPMM + nino3",
    "TNA|nino3": "TNA + nino3",
}


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


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_paths = [OVERALL_ROWS, OVERALL_LEAD, OVERALL_ROBUST, SYN_SUMMARY, SYN_LEAD]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required result table(s): " + ", ".join(str(path) for path in missing))

    overall_rows = pd.read_json(OVERALL_ROWS, lines=True)
    overall_lead = pd.read_csv(OVERALL_LEAD)
    overall_robust = pd.read_csv(OVERALL_ROBUST)
    syn_summary = pd.read_csv(SYN_SUMMARY)
    syn_lead = pd.read_csv(SYN_LEAD)

    for frame_name, frame in {
        "overall_rows": overall_rows,
        "overall_lead": overall_lead,
        "overall_robust": overall_robust,
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

    return overall_rows, overall_lead, overall_robust, syn_summary, syn_lead


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


def plot_overall_ei(overall_rows: pd.DataFrame, overall_lead: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.8), constrained_layout=True, sharey=True)
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
        axis.set_title(target, fontsize=7.5)
        axis.set_xlabel("Lead (months)")
        axis.set_ylabel("Overall EI (bits)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    return save_figure(fig, "unicm_enso_overall_ei_seed_overlay")[0]


def plot_source_ei(source_ei: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1), constrained_layout=True, sharex=True)
    for axis, target in zip(axes, TARGETS):
        subset = source_ei[(source_ei["target"] == target) & (source_ei["source"] != target)].sort_values("ei")
        colors = [
            "#E45756" if source in {"NPMM", "TNA"} else "#4C78A8" if source in HIGHLIGHT_SOURCES else "#A0A0A0"
            for source in subset["source"]
        ]
        axis.barh(subset["source"], subset["ei"], color=colors, edgecolor="none")
        axis.set_title(target, fontsize=7.5)
        axis.set_xlabel("Single-source EI (bits)")
        axis.set_ylabel("")
    return save_figure(fig, "unicm_enso_source_ei_rankings")[0]


def select_syn_pairs(syn_summary: pd.DataFrame, target: str) -> list[str]:
    target_rows = syn_summary[syn_summary["target"].astype(str) == target]
    top_pairs = target_rows.sort_values(["mean_syn", "pair"], ascending=[False, True]).head(5)["pair"].tolist()
    pairs: list[str] = []
    for pair in top_pairs + REQUIRED_PAIRS:
        if pair in set(target_rows["pair"].astype(str)) and pair not in pairs:
            pairs.append(pair)
    return pairs


def plot_syn_leads(syn_summary: pd.DataFrame, syn_lead: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), constrained_layout=True, sharey=False)
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
    }
    for axis, target in zip(axes, TARGETS):
        for pair in select_syn_pairs(syn_summary, target):
            subset = syn_lead[
                (syn_lead["target"].astype(str) == target) & (syn_lead["pair"].astype(str) == pair)
            ].sort_values("lead")
            style = "--" if pair == "NPMM|TNA" else "-"
            width = 1.6 if pair in REQUIRED_PAIRS else 1.1
            axis.plot(
                subset["lead"],
                subset["mean"],
                marker="o",
                markersize=2.0,
                linewidth=width,
                linestyle=style,
                color=palette.get(pair, "#A0A0A0"),
                label=PAIR_LABELS.get(pair, pair.replace("|", " + ")),
            )
        axis.axhline(0, color="#888888", linewidth=0.7, linestyle=":")
        axis.set_title(target, fontsize=7.5)
        axis.set_xlabel("Lead (months)")
        axis.set_ylabel("Seed mean Syn (bits)")
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return save_figure(fig, "unicm_enso_mode_pair_syn_leads")[0]


def fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def markdown_overall_table(overall_robust: pd.DataFrame) -> str:
    lines = [
        "| Target | mean EI 1..24 | mean EI 6..18 | Pearson min | Spearman min | top-3 overlap min |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        row = overall_robust[overall_robust["target"].astype(str) == target].iloc[0]
        lines.append(
            f"| {target} | {fmt(row['mean_overall_ei_full'])} | {fmt(row['mean_overall_ei_climate'])} | "
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
        lines.append(f"| {target} | {fmt(self_ei)} | {source_text} | {fmt(npmm)} | {fmt(tna)} |")
    return "\n".join(lines)


def markdown_syn_table(syn_summary: pd.DataFrame) -> str:
    rows: list[pd.Series] = []
    for target in TARGETS:
        target_rows = syn_summary[syn_summary["target"].astype(str) == target]
        keep = set(target_rows.sort_values(["mean_syn", "pair"], ascending=[False, True]).head(5)["pair"])
        keep.update(REQUIRED_PAIRS)
        rows.extend([row for _, row in target_rows[target_rows["pair"].isin(keep)].iterrows()])
    table = pd.DataFrame(rows).sort_values(["target", "rank_within_target"])
    lines = [
        "| Target | Source pair | rank | mean Syn | joint EI | left EI | right EI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples():
        lines.append(
            f"| {row.target} | {row.left_source} + {row.right_source} | {int(row.rank_within_target)} | "
            f"{fmt(row.mean_syn)} | {fmt(row.mean_joint_ei)} | {fmt(row.mean_left_ei)} | {fmt(row.mean_right_ei)} |"
        )
    return "\n".join(lines)


def pair_value(syn_summary: pd.DataFrame, target: str, pair: str, column: str = "mean_syn") -> float:
    row = syn_summary[(syn_summary["target"].astype(str) == target) & (syn_summary["pair"].astype(str) == pair)]
    if row.empty:
        raise AssertionError(f"Missing {pair} for {target}")
    return float(row.iloc[0][column])


def build_report(
    overall_robust: pd.DataFrame,
    source_ei: pd.DataFrame,
    syn_summary: pd.DataFrame,
    figure_paths: dict[str, Path],
) -> str:
    nino_npmm = pair_value(syn_summary, "nino", "nino|NPMM")
    nino_tna = pair_value(syn_summary, "nino", "nino|TNA")
    nino_npmm_tna = pair_value(syn_summary, "nino", "NPMM|TNA")
    nino3_npmm = pair_value(syn_summary, "nino3", "NPMM|nino3")
    nino3_tna = pair_value(syn_summary, "nino3", "TNA|nino3")
    nino3_npmm_tna = pair_value(syn_summary, "nino3", "NPMM|TNA")

    markdown = dedent(
        f"""
        # UniCM ENSO target EI/Syn 证据报告

        ## 结论

        原文命题认为，在强厄尔尼诺事件发生前，NPMM 和 TNA 与 ENSO 的相互作用增强。以 UniCM Modeformer 的 full-history maximum-entropy 机制读数来看，这个命题得到的是有条件的支持：NPMM/TNA 的信号确实出现在 ENSO target 的 EI/Syn 结构中，但更像对 ENSO 自身状态或赤道太平洋 mode 的远程调制，而不是 `NPMM + TNA` 两个外部 mode 单独构成强二源协同驱动。

        具体地，`nino` target 中 `nino + NPMM` 的平均 Syn 为 `{fmt(nino_npmm)}` bits，`nino + TNA` 为 `{fmt(nino_tna)}` bits，分别排在该 target 的前列；但 `NPMM + TNA` 直接到 `nino` 的平均 Syn 只有 `{fmt(nino_npmm_tna)}` bits。`nino3` target 也类似：`NPMM + nino3` 为 `{fmt(nino3_npmm)}` bits，`TNA + nino3` 为 `{fmt(nino3_tna)}` bits，而 `NPMM + TNA` 直接到 `nino3` 为 `{fmt(nino3_npmm_tna)}` bits。由此得到的新 insight 是：NPMM/TNA 对 ENSO 的贡献不宜表述为两个远程 mode 本身的强协同，而应表述为它们在 ENSO 已有背景态、赤道太平洋热状态和区域 ENSO 指数共同存在时提供增益。

        ## EI 证据：ENSO target 主要携带短中期记忆

        {markdown_overall_table(overall_robust)}

        ![ENSO overall EI](assets/{figure_paths['overall'].name})

        *图 1. `nino` 与 `nino3` target 的 full-history overall EI lead 曲线。彩色细线为 checkpoint seed，黑线为 seed mean，阴影为 seed standard deviation。两个 ENSO target 都在短 lead 具有较高 EI，随后快速衰减并在中长 lead 进入较低平台。*

        这个趋势说明，UniCM learned mechanism 对 ENSO target 的有效信息主要集中在短中期。`nino` 和 `nino3` 在 lead 1 到 6 个月的 EI 明显高于后期，符合 ENSO 预测中短期记忆强、长期不确定性上升的物理直觉。与此同时，`nino` 和 `nino3` 的 Pearson 相关较高但 Spearman 排序不足，说明不同 checkpoint 对整体衰减形状一致，但对具体 lead 优先级的排序仍有不确定性。

        ## 单源 EI：NPMM 是更稳定的远程信号，TNA 更像弱调制项

        {markdown_source_table(source_ei)}

        ![ENSO source EI rankings](assets/{figure_paths['source'].name})

        *图 2. ENSO target 的非自身 source EI 排名。红色突出 NPMM 和 TNA；蓝色突出与 ENSO 或太平洋热状态相关的 mode。*

        单源 EI 显示，ENSO 自身区域指数仍是主要信息来源：`nino` 对自身历史的 EI 约为 `0.476540` bits，`nino3` 对自身历史的 EI 约为 `0.477284` bits。排除自身后，NPMM 在两个 ENSO target 上都处于前列，说明北太平洋经向模态在 UniCM 中确实携带 ENSO 输出的远程信息。TNA 的单源 EI 较小，说明它不是最强的独立 ENSO source；它更可能通过与 ENSO 背景态或其他太平洋 mode 的组合产生可见影响。

        ## Syn 证据：增强主要发生在 ENSO 背景态参与时

        {markdown_syn_table(syn_summary)}

        ![ENSO mode-pair Syn leads](assets/{figure_paths['syn'].name})

        *图 3. ENSO target 的 mode-pair Syn lead 曲线。实线包含 top pairs 和计划指定的候选远程调制 pair；黑色虚线为 `NPMM + TNA` 直接二源 Syn。*

        对 `nino` target，最高 Syn pair 包含 `nino + SPMM`、`nino + nino3`、`nino + NPMM` 和 `nino + TNA`。这说明 NPMM/TNA 的增强更依赖 ENSO 自身历史参与：它们不是替代赤道太平洋记忆，而是在已有 ENSO 背景态上增加额外解释量。对 `nino3` target，`NPMM + nino3` 也进入前列，进一步说明 NPMM 对 eastern Pacific ENSO 区域存在候选调制作用。相比之下，`NPMM + TNA` 直接二源 Syn 在两个 ENSO target 上都接近零或为负，不能作为强协同驱动的主证据。

        ## 解释边界

        本报告只分析 frozen UniCM Modeformer learned mechanism，不是 reanalysis 事件复现实验，也不是 1983 或 1997 个例归因。这里的 EI/Syn 使用 Gaussian log-det 估计，适合作为 full-history 机制筛查；若要把结论推进到最终 PEID 或事件级归因，需要进一步做非线性 transport-map PEID、高样本复核，或按厄尔尼诺事件窗口构造条件化干预。
        """
    ).strip()
    markdown = "\n".join(line[8:] if line.startswith("        ") else line for line in markdown.splitlines())
    return markdown + "\n"


def main() -> int:
    configure_matplotlib()
    overall_rows, overall_lead, overall_robust, syn_summary, syn_lead = read_inputs()
    source_ei = compute_source_ei(syn_summary)

    figures = {
        "overall": plot_overall_ei(overall_rows, overall_lead),
        "source": plot_source_ei(source_ei),
        "syn": plot_syn_leads(syn_summary, syn_lead),
    }

    for path in figures.values():
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
