from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.unicm_peid_syn_analysis import (
    HISTORY_LENGTH,
    MODE_NAMES,
    enumerate_full_history_mode_subsets,
    estimate_full_history_subset_ei,
    load_full_history_prediction_cache,
    mobius_ei_interaction,
    overall_prediction_cache_path,
    sample_full_history_mode_inputs,
    write_jsonl,
)


DEFAULT_CACHE_DIR = ROOT / "results" / "unicm_overall_ei_cpu_bound4_n1024" / "cache"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "unicm_full_history_mode_triple_syn_cpu_bound4_n1024"
DEFAULT_REPORT_PATH = ROOT / "docs" / "reports" / "Part2.md"
DEFAULT_ASSET_BASE = ROOT / "docs" / "reports" / "assets" / "unicm_enso_mode_triple_interaction_leads"
SECTION_TITLE = "## 三源 interaction: 高阶增量仍依赖 ENSO 背景态"
BOUNDARY_TITLE = "## 解释边界"


def _cache_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        n_samples=int(args.n_samples),
        sampling_seed=int(args.sampling_seed),
        intervention_bound=float(args.intervention_bound),
        start_month=int(args.start_month),
        device=str(args.device),
    )


def _load_and_validate_cache(cache_path: Path, *, seed: int, args: argparse.Namespace) -> np.ndarray:
    if not cache_path.exists():
        raise FileNotFoundError(f"Prediction cache is missing and forward is disabled: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as payload:
        if "metadata" not in payload:
            raise ValueError(f"Prediction cache metadata is missing; forward is disabled: {cache_path}")
        try:
            metadata = json.loads(str(payload["metadata"].item()))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Prediction cache metadata is invalid: {cache_path}") from exc

    expected = {
        "seed": int(seed),
        "n_samples": int(args.n_samples),
        "sampling_seed": int(args.sampling_seed),
        "intervention_bound": float(args.intervention_bound),
        "sampling_mode": "full_history_max_entropy",
        "history_shape": [HISTORY_LENGTH, len(MODE_NAMES)],
        "start_month": int(args.start_month),
        "device": str(args.device),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        details = ", ".join(f"{key}={actual!r} (expected {wanted!r})" for key, (actual, wanted) in mismatches.items())
        raise ValueError(f"Prediction cache metadata mismatch in {cache_path}: {details}")
    return load_full_history_prediction_cache(cache_path, n_samples=int(args.n_samples))


def _build_subset_lattice(
    history_modes: np.ndarray,
    source_modes: Sequence[str],
    target: np.ndarray,
) -> dict[tuple[str, ...], float]:
    lookup: dict[tuple[str, ...], float] = {}
    for order in (1, 2, 3):
        for subset in enumerate_full_history_mode_subsets(source_modes, order=order):
            lookup[subset] = estimate_full_history_subset_ei(history_modes, subset, target)
    return lookup


def summarize_triple_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    triple_cols = ["triple", "source_a", "source_b", "source_c", "target"]
    summary = (
        rows.groupby(triple_cols, as_index=False)
        .agg(
            mean_delta3=("delta3", "mean"),
            std_delta3=("delta3", "std"),
            min_delta3=("delta3", "min"),
            max_delta3=("delta3", "max"),
            mean_triple_ei=("triple_ei", "mean"),
            mean_pair_ab_ei=("pair_ab_ei", "mean"),
            mean_pair_ac_ei=("pair_ac_ei", "mean"),
            mean_pair_bc_ei=("pair_bc_ei", "mean"),
            mean_source_a_ei=("source_a_ei", "mean"),
            mean_source_b_ei=("source_b_ei", "mean"),
            mean_source_c_ei=("source_c_ei", "mean"),
        )
        .sort_values(["target", "mean_delta3", "triple"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    summary["std_delta3"] = summary["std_delta3"].fillna(0.0)
    summary["rank_within_target"] = summary.groupby("target")["mean_delta3"].rank(
        ascending=False, method="first"
    ).astype(int)

    seed_summary = (
        rows.groupby(["seed", *triple_cols], as_index=False)["delta3"]
        .mean()
        .rename(columns={"delta3": "mean_delta3"})
        .sort_values(["seed", "target", "mean_delta3", "triple"], ascending=[True, True, False, True])
    )
    seed_summary["rank_within_seed"] = seed_summary.groupby(["seed", "target"])["mean_delta3"].rank(
        ascending=False, method="first"
    ).astype(int)
    seed_stats = (
        seed_summary.groupby(triple_cols, as_index=False)
        .agg(
            seed_mean_delta3=("mean_delta3", "mean"),
            seed_std_delta3=("mean_delta3", "std"),
            positive_seed_count=("mean_delta3", lambda values: int((values > 0.0).sum())),
            seed_rank_min=("rank_within_seed", "min"),
            seed_rank_max=("rank_within_seed", "max"),
            n_seeds=("seed", "nunique"),
        )
    )
    seed_stats["seed_std_delta3"] = seed_stats["seed_std_delta3"].fillna(0.0)
    summary = summary.merge(seed_stats, on=triple_cols, how="left", validate="one_to_one")
    summary = summary.sort_values(["target", "mean_delta3", "triple"], ascending=[True, False, True]).reset_index(
        drop=True
    )
    summary["rank_within_target"] = summary.groupby("target").cumcount() + 1

    lead_summary = (
        rows.groupby([*triple_cols, "lead"], as_index=False)["delta3"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_delta3", "std": "std_delta3"})
    )
    lead_summary["std_delta3"] = lead_summary["std_delta3"].fillna(0.0)
    return summary, lead_summary, seed_summary


def plot_top_triple_interactions(
    summary: pd.DataFrame,
    lead_summary: pd.DataFrame,
    output_base: Path,
    *,
    top_k: int,
    asset_base: Path | None,
) -> list[Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

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
    top = summary.sort_values(["mean_delta3", "triple"], ascending=[False, True]).head(int(top_k))
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#72B7B2"]
    fig, ax = plt.subplots(figsize=(7.4, 3.7), constrained_layout=True)
    for color, row in zip(palette, top.itertuples(index=False)):
        curve = lead_summary[lead_summary["triple"].astype(str) == str(row.triple)].sort_values("lead")
        x = curve["lead"].to_numpy(dtype=float)
        mean = curve["mean_delta3"].to_numpy(dtype=float)
        label = str(row.triple).replace("|", " + ")
        ax.plot(
            x,
            mean,
            color=color,
            marker="o",
            markersize=2.2,
            linewidth=1.2,
            label=label,
        )
        ax.axhline(float(row.mean_delta3), color=color, linewidth=0.7, linestyle=":", alpha=0.55)
    ax.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lead (months)")
    ax.set_ylabel(r"Third-order EI interaction $\Delta_3$ (bits)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    bases = [Path(output_base)]
    if asset_base is not None:
        bases.append(Path(asset_base))
    paths: list[Path] = []
    for base in bases:
        base.parent.mkdir(parents=True, exist_ok=True)
        png_path = base.with_suffix(".png")
        svg_path = base.with_suffix(".svg")
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        paths.extend([png_path, svg_path])
    plt.close(fig)
    return paths


def _fmt(value: object) -> str:
    number = float(value)
    return "nan" if not np.isfinite(number) else f"{number:.6f}"


def build_report_section(summary: pd.DataFrame, *, rows_count: int, top_k: int) -> str:
    top = summary.sort_values(["mean_delta3", "triple"], ascending=[False, True]).head(10)
    bottom = summary.sort_values(["mean_delta3", "triple"], ascending=[True, True]).head(5)
    lines = [
        SECTION_TITLE,
        "",
        "本轮固定目标为未来 `nino`，复用 1024 个 full-history 最大熵干预样本和三个 checkpoint 的预测缓存，",
        "没有加载 UniCM 模型，也没有执行新 forward。11 个 source mode 的全部 165 个无序三元组均被保留。",
        "",
        "对三元组 $K=\\{i,j,k\\}$，精确三阶 EI interaction 使用布尔子集格 Möbius 反演：",
        "",
        "```math",
        "\\Delta_{ijk}=EI_{ijk}-EI_{ij}-EI_{ik}-EI_{jk}+EI_i+EI_j+EI_k.",
        "```",
        "",
        "三阶 interaction 是有符号量；只有正值作为三源协同候选，负值不截断。完整排名按 seeds `1,2,3` 与 leads `1..24` 的平均 $\\Delta_3$ 降序。",
        "",
        "| Rank | Source triple | mean Δ3 | seed SD | seed rank range | + seeds | joint EI |",
        "|---:|---|---:|---:|---|---:|---:|",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"| {int(row.rank_within_target)} | {str(row.triple).replace('|', ' + ')} | {_fmt(row.mean_delta3)} | "
            f"{_fmt(row.seed_std_delta3)} | {int(row.seed_rank_min)}-{int(row.seed_rank_max)} | "
            f"{int(row.positive_seed_count)}/{int(row.n_seeds)} | {_fmt(row.mean_triple_ei)} |"
        )
    top_row = top.iloc[0]
    nino_top_count = int(top["triple"].astype(str).str.split("|").map(lambda names: "nino" in names).sum())
    lines.extend(
        [
            "",
            f"排名第一的是 `{str(top_row['triple']).replace('|', ' + ')}`，平均 $\\Delta_3$ 为 `{_fmt(top_row['mean_delta3'])}` bits，"
            f"三个 seed 均为正，seed 内排名范围为 `{int(top_row['seed_rank_min'])}-{int(top_row['seed_rank_max'])}`。"
            f"Top-10 中 `{nino_top_count}/10` 个三元组包含 `nino` 自身历史，说明当前最强三阶候选仍主要表现为外部 mode 在 ENSO 已有背景态上的高阶增量，而不是三个纯外部 mode 的独立驱动。",
            "",
            "![Top UniCM third-order EI interactions](assets/unicm_enso_mode_triple_interaction_leads.png)",
            "",
            "*图 4. 平均三阶 interaction 排名前五的 lead 曲线。点线为三个 checkpoint seed 的均值；同色虚线为该三元组在全部 lead 和 seed 上的平均值。*",
            "",
            "### 最负的三阶 interaction",
            "",
            "| Source triple | mean Δ3 | seed rank range |",
            "|---|---:|---|",
        ]
    )
    for row in bottom.itertuples(index=False):
        lines.append(
            f"| {str(row.triple).replace('|', ' + ')} | {_fmt(row.mean_delta3)} | "
            f"{int(row.seed_rank_min)}-{int(row.seed_rank_max)} |"
        )
    lines.extend(
        [
            "",
            f"原始结果共 `{int(rows_count)}` 行，即 `165 triples × 3 seeds × 24 leads`。机器可读输出位于 "
            "`results/unicm_full_history_mode_triple_syn_cpu_bound4_n1024/`。",
            "",
            "这里的 Gaussian log-det $\\Delta_3$ 是 frozen Modeformer 机制的高阶筛查读数，不等同于 transport-map PEID 的最终非线性分解。",
        ]
    )
    return "\n".join(lines)


def _insert_or_replace_report_section(markdown: str, section: str) -> str:
    start = markdown.find(SECTION_TITLE)
    if start >= 0:
        boundary = markdown.find(BOUNDARY_TITLE, start)
        if boundary < 0:
            raise ValueError(f"Could not find {BOUNDARY_TITLE!r} after existing three-source section.")
        return markdown[:start].rstrip() + "\n\n" + section.rstrip() + "\n\n" + markdown[boundary:].lstrip()
    boundary = markdown.find(BOUNDARY_TITLE)
    if boundary < 0:
        raise ValueError(f"Could not find report insertion boundary {BOUNDARY_TITLE!r}.")
    return markdown[:boundary].rstrip() + "\n\n" + section.rstrip() + "\n\n" + markdown[boundary:].lstrip()


def run_three_source_syn(args: argparse.Namespace) -> dict[str, object]:
    source_modes = [str(name) for name in args.source_modes]
    triples = enumerate_full_history_mode_subsets(source_modes, order=3)
    leads = sorted({int(lead) for lead in args.leads})
    if not leads or min(leads) < 1 or max(leads) > 24:
        raise ValueError("leads must stay within [1, 24].")
    if str(args.target) not in MODE_NAMES:
        raise ValueError(f"Unknown target mode: {args.target}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_modes = sample_full_history_mode_inputs(
        n_samples=int(args.n_samples),
        intervention_bound=float(args.intervention_bound),
        seed=int(args.sampling_seed),
    )
    target_index = MODE_NAMES[str(args.target)]
    rows: list[dict[str, object]] = []
    for seed in [int(value) for value in args.seeds]:
        cache_path = overall_prediction_cache_path(Path(args.overall_cache_dir), seed=seed, args=_cache_args(args))
        print(f"[seed {seed}] loading prediction cache {cache_path}", file=sys.stderr, flush=True)
        all_mode_targets = _load_and_validate_cache(cache_path, seed=seed, args=args)
        for lead in leads:
            target = all_mode_targets[:, int(lead) - 1, target_index]
            ei_lookup = _build_subset_lattice(history_modes, source_modes, target)
            for source_a, source_b, source_c in triples:
                triple = (source_a, source_b, source_c)
                pair_ab = (source_a, source_b)
                pair_ac = (source_a, source_c)
                pair_bc = (source_b, source_c)
                rows.append(
                    {
                        "seed": seed,
                        "triple": "|".join(triple),
                        "source_a": source_a,
                        "source_b": source_b,
                        "source_c": source_c,
                        "target": str(args.target),
                        "lead": int(lead),
                        "n_samples": int(args.n_samples),
                        "intervention_bound": float(args.intervention_bound),
                        "sampling_seed": int(args.sampling_seed),
                        "sampling_mode": "full_history_max_entropy",
                        "backend": "gaussian_logdet_full_history_mobius_order3",
                        "triple_ei": float(ei_lookup[triple]),
                        "pair_ab_ei": float(ei_lookup[pair_ab]),
                        "pair_ac_ei": float(ei_lookup[pair_ac]),
                        "pair_bc_ei": float(ei_lookup[pair_bc]),
                        "source_a_ei": float(ei_lookup[(source_a,)]),
                        "source_b_ei": float(ei_lookup[(source_b,)]),
                        "source_c_ei": float(ei_lookup[(source_c,)]),
                        "delta3": mobius_ei_interaction(triple, ei_lookup),
                    }
                )

    rows_path = output_dir / "full_history_mode_triple_interaction_rows.jsonl"
    write_jsonl(rows, rows_path)
    frame = pd.DataFrame(rows)
    summary, lead_summary, seed_summary = summarize_triple_rows(frame)
    summary_path = output_dir / "full_history_mode_triple_interaction_summary.csv"
    lead_summary_path = output_dir / "full_history_mode_triple_interaction_lead_summary.csv"
    seed_summary_path = output_dir / "full_history_mode_triple_interaction_seed_summary.csv"
    top_path = output_dir / "full_history_mode_triple_interaction_top.csv"
    bottom_path = output_dir / "full_history_mode_triple_interaction_bottom.csv"
    summary.to_csv(summary_path, index=False)
    lead_summary.to_csv(lead_summary_path, index=False)
    seed_summary.to_csv(seed_summary_path, index=False)
    summary.head(int(args.top_k)).to_csv(top_path, index=False)
    summary.sort_values(["mean_delta3", "triple"], ascending=[True, True]).head(5).to_csv(bottom_path, index=False)

    asset_base = DEFAULT_ASSET_BASE if bool(args.report) else None
    figure_paths = plot_top_triple_interactions(
        summary,
        lead_summary,
        output_dir / "fig" / "full_history_mode_triple_interaction_top",
        top_k=min(5, int(args.top_k)),
        asset_base=asset_base,
    )
    if bool(args.report):
        report_path = Path(args.report_path)
        markdown = report_path.read_text(encoding="utf-8")
        section = build_report_section(summary, rows_count=len(frame), top_k=int(args.top_k))
        report_path.write_text(_insert_or_replace_report_section(markdown, section), encoding="utf-8")

    return {
        "rows": str(rows_path),
        "summary": str(summary_path),
        "lead_summary": str(lead_summary_path),
        "seed_summary": str(seed_summary_path),
        "top": str(top_path),
        "bottom": str(bottom_path),
        "figures": [str(path) for path in figure_paths],
        "report": str(args.report_path) if bool(args.report) else None,
        "n_rows": int(len(frame)),
        "n_triples": int(len(summary)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank cache-only UniCM third-order Möbius EI interactions for nino.")
    parser.add_argument("--overall-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--target", default="nino")
    parser.add_argument("--source-modes", nargs="+", default=list(MODE_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--leads", nargs="+", type=int, default=list(range(1, 25)))
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--sampling-seed", type=int, default=20260619)
    parser.add_argument("--intervention-bound", type=float, default=4.0)
    parser.add_argument("--start-month", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-report", action="store_false", dest="report")
    parser.set_defaults(report=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    outputs = run_three_source_syn(args)
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
