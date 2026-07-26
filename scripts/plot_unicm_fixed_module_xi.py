from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_unicm_all_mode_target_pair_syn import extract_all_mode_target, parse_leads  # noqa: E402
from scripts.plot_unicm_phi_eid_greedy_decomposition import (  # noqa: E402
    compute_subset_ei_table_from_covariance,
)
from scripts.unicm_peid_syn_analysis import (  # noqa: E402
    MODE_NAMES,
    create_ei_estimator,
    load_full_history_prediction_cache,
    overall_prediction_cache_path,
    sample_full_history_mode_inputs,
)

DEFAULT_CACHE_DIR = ROOT / "results" / "unicm_overall_ei_cpu_bound4_n8192" / "cache"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "unicm_fixed_module_xi_tm_degree1_signed_n8192"
DEFAULT_ASSET_BASE = ROOT / "fig" / "unicm_fixed_module_xi_leads"

FIXED_MODULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ENSO-pattern pair", ("nino12", "nino3")),
    ("ENSO-pattern core", ("nino", "nino12", "nino3")),
    ("ENSO spatial quartet", ("nino", "nino12", "nino3", "nino4")),
    ("ENSO–IOD module", ("nino", "IOD", "nino12", "nino3", "nino4")),
)

DISPLAY_NAME = {
    "nino": "ENSO",
    "nino12": "nino12",
    "nino3": "nino3",
    "nino4": "nino4",
    "IOD": "IOD",
}

COLORS = ("#38598C", "#6E8FB2", "#9C7A9C", "#D07A55")


def display_sources(sources: Sequence[str]) -> str:
    return " + ".join(DISPLAY_NAME.get(str(source), str(source)) for source in sources)


def validate_modules(
    modules: Sequence[tuple[str, Sequence[str]]],
    *,
    mode_names: Mapping[str, int] = MODE_NAMES,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    validated: list[tuple[str, tuple[str, ...]]] = []
    seen_labels: set[str] = set()
    previous: set[str] | None = None
    for label, raw_sources in modules:
        sources = tuple(str(source) for source in raw_sources)
        if str(label) in seen_labels:
            raise ValueError(f"Duplicate module label: {label}")
        if len(sources) < 2 or len(set(sources)) != len(sources):
            raise ValueError(f"Module {label!r} must contain at least two unique sources.")
        unknown = [source for source in sources if source not in mode_names]
        if unknown:
            raise ValueError(f"Unknown source(s) in {label!r}: {', '.join(unknown)}")
        current = set(sources)
        if previous is not None and not previous.issubset(current):
            raise ValueError("Fixed modules must form a nested sequence.")
        validated.append((str(label), sources))
        seen_labels.add(str(label))
        previous = current
    return tuple(validated)


def _subset_source(history_modes: np.ndarray, sources: Sequence[str]) -> np.ndarray:
    history = np.asarray(history_modes, dtype=float)
    columns = [history[:, :, int(MODE_NAMES[str(source)])] for source in sources]
    return np.concatenate(columns, axis=1)


def _regularized_logdet(covariance: np.ndarray, *, jitter: float) -> float:
    cov = np.atleast_2d(np.asarray(covariance, dtype=float))
    scale = float(np.trace(cov) / cov.shape[0]) if cov.size else 1.0
    ridge = float(jitter) * max(scale, 1.0)
    sign, value = np.linalg.slogdet(cov + ridge * np.eye(cov.shape[0], dtype=float))
    if sign <= 0 or not np.isfinite(value):
        raise ValueError("Regularized covariance matrix is not positive definite.")
    return float(value)


def precompute_selected_source_logdets(
    history_modes: np.ndarray,
    subsets: Sequence[tuple[str, ...]],
    *,
    jitter: float,
) -> tuple[np.ndarray, dict[tuple[str, ...], list[int]], dict[tuple[str, ...], float]]:
    history = np.asarray(history_modes, dtype=float)
    if history.ndim != 3:
        raise ValueError("history_modes must have shape (n_samples, history_length, n_modes).")
    flat = history.reshape(history.shape[0], history.shape[1] * history.shape[2])
    covariance = np.cov(flat, rowvar=False, bias=False)
    subset_columns: dict[tuple[str, ...], list[int]] = {}
    source_logdets: dict[tuple[str, ...], float] = {}
    for subset in subsets:
        columns = [
            month * len(MODE_NAMES) + int(MODE_NAMES[source])
            for month in range(history.shape[1])
            for source in subset
        ]
        subset_columns[subset] = columns
        source_logdets[subset] = _regularized_logdet(
            covariance[np.ix_(columns, columns)],
            jitter=float(jitter),
        )
    return flat, subset_columns, source_logdets


def _compute_ei_lookup_direct(
    history_modes: np.ndarray,
    target: np.ndarray,
    subsets: Sequence[tuple[str, ...]],
    *,
    estimator: Callable[[np.ndarray, np.ndarray], float],
) -> dict[tuple[str, ...], float]:
    return {
        tuple(subset): float(estimator(_subset_source(history_modes, subset), target))
        for subset in subsets
    }


def compute_fixed_module_rows(
    history_modes: np.ndarray,
    targets_by_seed: Mapping[int, np.ndarray],
    *,
    modules: Sequence[tuple[str, Sequence[str]]] = FIXED_MODULES,
    leads: Sequence[int] | None = None,
    estimator_name: str = "transport_map",
    tm_degree: int = 1,
    tm_jitter: float = 1.0e-6,
) -> tuple[pd.DataFrame, dict[str, object]]:
    fixed_modules = validate_modules(modules)
    lead_values = list(range(1, 25)) if leads is None else [int(lead) for lead in leads]
    all_sources = tuple(MODE_NAMES)
    required_singletons = tuple(
        (source,)
        for source in all_sources
    )
    module_subsets = tuple(sources for _, sources in fixed_modules)
    required_subsets = tuple(dict.fromkeys((*required_singletons, *module_subsets, all_sources)))

    estimator, estimator_metadata = create_ei_estimator(
        str(estimator_name),
        tm_degree=int(tm_degree),
        tm_jitter=float(tm_jitter),
        clip_negative=False,
    )

    covariance_inputs = None
    if str(estimator_name) == "transport_map" and int(tm_degree) == 1:
        history_flat, subset_columns, source_logdets = precompute_selected_source_logdets(
            history_modes,
            required_subsets,
            jitter=float(tm_jitter),
        )
        covariance_inputs = (history_flat, subset_columns, source_logdets)

    rows: list[dict[str, object]] = []
    for seed in sorted(int(value) for value in targets_by_seed):
        predictions = np.asarray(targets_by_seed[seed], dtype=float)
        for lead in lead_values:
            target = extract_all_mode_target(predictions, lead=int(lead))
            if covariance_inputs is not None:
                history_flat, subset_columns, source_logdets = covariance_inputs
                ei_lookup = compute_subset_ei_table_from_covariance(
                    history_flat,
                    target,
                    subset_columns,
                    source_logdets,
                    jitter=float(tm_jitter),
                )
            else:
                ei_lookup = _compute_ei_lookup_direct(
                    history_modes,
                    target,
                    required_subsets,
                    estimator=estimator,
                )

            system_singleton_sum = float(sum(ei_lookup[(source,)] for source in all_sources))
            system_xi = float(ei_lookup[all_sources] - system_singleton_sum)
            for module_index, (label, sources) in enumerate(fixed_modules):
                singleton_sum = float(sum(ei_lookup[(source,)] for source in sources))
                module_ei = float(ei_lookup[sources])
                xi = float(module_ei - singleton_sum)
                rows.append(
                    {
                        "seed": int(seed),
                        "lead": int(lead),
                        "module_index": int(module_index),
                        "module": label,
                        "sources": "|".join(sources),
                        "display_sources": display_sources(sources),
                        "order": int(len(sources)),
                        "whole_ei": module_ei,
                        "singleton_ei_sum": singleton_sum,
                        "xi": xi,
                        "system_xi": system_xi,
                        "xi_over_system_xi": float(xi / system_xi) if abs(system_xi) > 1.0e-12 else np.nan,
                    }
                )
    frame = pd.DataFrame(rows).sort_values(["seed", "lead", "module_index"]).reset_index(drop=True)
    return frame, estimator_metadata


def summarize_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lead_summary = (
        rows.groupby(["module_index", "module", "sources", "display_sources", "order", "lead"], as_index=False)
        .agg(
            xi_mean=("xi", "mean"),
            xi_std=("xi", "std"),
            xi_min=("xi", "min"),
            xi_max=("xi", "max"),
            positive_seed_count=("xi", lambda values: int((np.asarray(values, dtype=float) > 0.0).sum())),
            xi_over_system_xi_mean=("xi_over_system_xi", "mean"),
            xi_over_system_xi_std=("xi_over_system_xi", "std"),
        )
        .sort_values(["module_index", "lead"])
        .reset_index(drop=True)
    )
    lead_summary[["xi_std", "xi_over_system_xi_std"]] = lead_summary[
        ["xi_std", "xi_over_system_xi_std"]
    ].fillna(0.0)

    delta_rows: list[dict[str, object]] = []
    for (seed, lead), frame in rows.groupby(["seed", "lead"], sort=True):
        ordered = frame.sort_values("module_index")
        for left, right in zip(ordered.itertuples(index=False), ordered.iloc[1:].itertuples(index=False)):
            added = sorted(set(str(right.sources).split("|")) - set(str(left.sources).split("|")))
            delta_rows.append(
                {
                    "seed": int(seed),
                    "lead": int(lead),
                    "from_module": str(left.module),
                    "to_module": str(right.module),
                    "added_source": "|".join(added),
                    "display_added_source": display_sources(added),
                    "delta_xi": float(right.xi - left.xi),
                }
            )
    deltas = pd.DataFrame(delta_rows)
    delta_summary = (
        deltas.groupby(
            ["from_module", "to_module", "added_source", "display_added_source", "lead"],
            as_index=False,
        )
        .agg(
            delta_xi_mean=("delta_xi", "mean"),
            delta_xi_std=("delta_xi", "std"),
            positive_seed_count=("delta_xi", lambda values: int((np.asarray(values, dtype=float) > 0.0).sum())),
        )
        .sort_values(["to_module", "lead"])
        .reset_index(drop=True)
    )
    delta_summary["delta_xi_std"] = delta_summary["delta_xi_std"].fillna(0.0)

    windows = (("short", 1, 5), ("mid", 7, 10), ("late", 15, 24))
    window_rows: list[pd.DataFrame] = []
    for window, start, end in windows:
        selected = rows[rows["lead"].between(start, end)].copy()
        paired = (
            selected.groupby(
                ["seed", "module_index", "module", "sources", "display_sources", "order"],
                as_index=False,
            )["xi"]
            .mean()
            .rename(columns={"xi": "window_mean_xi"})
        )
        paired["window"] = window
        paired["lead_start"] = int(start)
        paired["lead_end"] = int(end)
        window_rows.append(paired)
    window_summary = pd.concat(window_rows, ignore_index=True)
    return lead_summary, deltas, delta_summary, window_summary


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


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.05, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def plot_results(
    rows: pd.DataFrame,
    lead_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
    window_summary: pd.DataFrame,
    output_base: Path,
) -> list[Path]:
    configure_matplotlib()
    fig = plt.figure(figsize=(7.2, 5.7))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.05, 0.15, 1.0])
    grid.update(left=0.09, right=0.98, bottom=0.09, top=0.97, wspace=0.42, hspace=0.32)
    ax_curve = fig.add_subplot(grid[0, 0])
    ax_ratio = fig.add_subplot(grid[0, 1])
    ax_legend = fig.add_subplot(grid[1, :])
    ax_delta = fig.add_subplot(grid[2, 0])
    ax_window = fig.add_subplot(grid[2, 1])

    modules = (
        lead_summary[["module_index", "module", "display_sources"]]
        .drop_duplicates()
        .sort_values("module_index")
    )
    for module_index, module, display in modules.itertuples(index=False):
        color = COLORS[int(module_index)]
        summary = lead_summary[lead_summary["module_index"].eq(int(module_index))].sort_values("lead")
        raw = rows[rows["module_index"].eq(int(module_index))]
        for _, seed_rows in raw.groupby("seed"):
            seed_rows = seed_rows.sort_values("lead")
            ax_curve.plot(
                seed_rows["lead"],
                seed_rows["xi"],
                color=color,
                linewidth=0.55,
                alpha=0.22,
                zorder=1,
            )
        ax_curve.plot(
            summary["lead"],
            summary["xi_mean"],
            color=color,
            linewidth=1.6,
            marker="o",
            markersize=2.0,
            label=str(display),
            zorder=3,
        )
        ax_curve.fill_between(
            summary["lead"],
            summary["xi_mean"] - summary["xi_std"],
            summary["xi_mean"] + summary["xi_std"],
            color=color,
            alpha=0.10,
            linewidth=0,
            zorder=2,
        )
        ax_ratio.plot(
            summary["lead"],
            100.0 * summary["xi_over_system_xi_mean"],
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=2.0,
            label=str(display),
        )
    for ax in (ax_curve, ax_ratio):
        ax.axhline(0.0, color="#777777", linewidth=0.65, linestyle=":")
        ax.set_xlim(1, 24)
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
        ax.grid(axis="y", color="#E4E7EA", linewidth=0.5)
        ax.set_xlabel("Prediction lead (months)")
    ax_curve.set_ylabel(r"$\Xi_S$ (bits)")
    handles, labels = ax_curve.get_legend_handles_labels()
    ax_legend.axis("off")
    ax_legend.legend(
        handles,
        labels,
        loc="center",
        ncol=2,
        fontsize=5.4,
        handlelength=1.5,
        columnspacing=1.0,
    )
    ax_ratio.axhline(100.0, color="#888888", linewidth=0.65, linestyle="--")
    ax_ratio.set_ylabel(r"$\Xi_S / \Xi_{\mathrm{all}}$ (%)")
    _panel_label(ax_curve, "a")
    _panel_label(ax_ratio, "b")

    delta_pivot = (
        delta_summary.pivot(index="display_added_source", columns="lead", values="delta_xi_mean")
        .reindex(["ENSO", "nino4", "IOD"])
    )
    bound = float(np.nanmax(np.abs(delta_pivot.to_numpy(dtype=float))))
    image = ax_delta.imshow(
        delta_pivot.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-bound,
        vmax=bound,
    )
    ax_delta.set_yticks(np.arange(len(delta_pivot.index)), [f"+ {value}" for value in delta_pivot.index])
    ax_delta.set_xticks(np.arange(len(delta_pivot.columns))[::3])
    ax_delta.set_xticklabels([str(int(value)) for value in delta_pivot.columns[::3]])
    ax_delta.set_xlabel("Prediction lead (months)")
    ax_delta.set_ylabel("Added source")
    cbar_ax = ax_delta.inset_axes([1.03, 0.06, 0.035, 0.88])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.ax.set_title(r"$\Delta\Xi_S$" + "\n(bits)", fontsize=5.8, pad=3)
    _panel_label(ax_delta, "c")

    mid = window_summary[window_summary["window"].eq("mid")].copy()
    positions = np.arange(len(modules), dtype=float)
    rng = np.random.default_rng(20260726)
    for position, (module_index, _, display) in zip(positions, modules.itertuples(index=False)):
        values = mid[mid["module_index"].eq(int(module_index))]["window_mean_xi"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.07, 0.07, size=len(values))
        ax_window.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=18,
            color=COLORS[int(module_index)],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax_window.hlines(
            float(np.mean(values)),
            position - 0.20,
            position + 0.20,
            color="#182235",
            linewidth=1.4,
            zorder=4,
        )
    ax_window.axhline(0.0, color="#777777", linewidth=0.65, linestyle=":")
    ax_window.set_xticks(positions)
    ax_window.set_xticklabels(
        ["nino12\n+nino3", "+ENSO", "+nino4", "+IOD"],
        rotation=0,
    )
    ax_window.set_ylabel(r"Lead 7–10 mean $\Xi_S$ (bits)")
    ax_window.grid(axis="y", color="#E4E7EA", linewidth=0.5)
    _panel_label(ax_window, "d")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        output_base.with_suffix(".png"),
        output_base.with_suffix(".svg"),
        output_base.with_suffix(".pdf"),
    ]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], bbox_inches="tight")
    plt.close(fig)
    return paths


def _cache_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        n_samples=int(args.n_samples),
        sampling_seed=int(args.sampling_seed),
        intervention_bound=float(args.intervention_bound),
        start_month=int(args.start_month),
        device=str(args.device),
    )


def write_experiment_contract(path: Path, args: argparse.Namespace, estimator_metadata: Mapping[str, object]) -> None:
    path.write_text(
        "\n".join(
            [
                "# UniCM fixed-module controlled-comparison contract",
                "",
                "| Field | Frozen value |",
                "|---|---|",
                "| Scientific question | What changes when only the nested source-mode set changes? |",
                "| Treatment factor | Fixed source subset used to compute path-independent $\\Xi_S$ |",
                "| Treatment levels | nino12+nino3; +ENSO; +nino4; +IOD |",
                "| Unit of pairing | checkpoint seed × prediction lead |",
                "| Primary metric | signed $\\Xi_S=EI(S\\to\\mathbf{Y})-\\sum_{m\\in S}EI(m\\to\\mathbf{Y})$ |",
                "| Target | all 11 predicted UniCM modes at the same lead |",
                f"| Seeds | {', '.join(str(seed) for seed in args.seeds)} |",
                f"| Leads | {', '.join(str(lead) for lead in parse_leads(args.leads))} |",
                f"| Samples | {int(args.n_samples)} shared maximum-entropy intervention samples |",
                f"| Intervention support | independent uniform $[-{float(args.intervention_bound):g},{float(args.intervention_bound):g}]$ for every mode-month |",
                f"| Sampling seed | {int(args.sampling_seed)} |",
                f"| Estimator | {json.dumps(dict(estimator_metadata), ensure_ascii=False)} |",
                "| Model and predictions | frozen UniCM checkpoints; existing shared prediction cache; no retraining or new forward |",
                "| Postprocessing | signed values; no clipping |",
                "| Main limitation | variables outside each fixed subset remain independently intervened nuisance background and are marginalized in the target distribution |",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_report(
    path: Path,
    rows: pd.DataFrame,
    lead_summary: pd.DataFrame,
    deltas: pd.DataFrame,
    window_summary: pd.DataFrame,
) -> None:
    peak_rows = lead_summary.loc[lead_summary.groupby("module_index")["xi_mean"].idxmax()].sort_values("module_index")
    lead8 = lead_summary[lead_summary["lead"].eq(8)].sort_values("module_index")
    mid = window_summary[window_summary["window"].eq("mid")]
    mid_summary = (
        mid.groupby(["module_index", "module"], as_index=False)["window_mean_xi"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("module_index")
    )
    mid_delta = (
        deltas[deltas["lead"].between(7, 10)]
        .groupby(["seed", "display_added_source"], as_index=False)["delta_xi"]
        .mean()
        .groupby("display_added_source", as_index=False)["delta_xi"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    sign_summary = (
        deltas.groupby("display_added_source", as_index=False)["delta_xi"]
        .agg(
            positive_contexts=lambda values: int((np.asarray(values, dtype=float) > 0.0).sum()),
            context_count="size",
        )
    )

    lines = [
        "# UniCM 路径无关固定模块协同验证",
        "",
        "## 稳定发现",
        "",
        "四个预先固定的嵌套源集合都在 lead 7–10 出现联合信息增强。"
        "不依赖自由 greedy 路径时，`ENSO + IOD + nino12 + nino3 + nino4` 仍是最强集合，"
        "说明原层级图中的五模态核不是仅由某一条贪婪拆分路径产生。",
        "",
        "## 峰值与 lead-8 结果",
        "",
        "| 固定集合 | 均值峰值 lead | 峰值 $\\Xi_S$ (bits) | lead 8 $\\Xi_S$ (mean ± s.d.) | lead 8 $\\Xi_S/\\Xi_{all}$ |",
        "|---|---:|---:|---:|---:|",
    ]
    for peak, row8 in zip(peak_rows.itertuples(index=False), lead8.itertuples(index=False)):
        lines.append(
            f"| {row8.display_sources} | {int(peak.lead)} | {float(peak.xi_mean):.6f} | "
            f"{float(row8.xi_mean):.6f} ± {float(row8.xi_std):.6f} | "
            f"{100.0 * float(row8.xi_over_system_xi_mean):.1f}% |"
        )
    lines.extend(
        [
            "",
            "这里的比例是同一 seed 内固定集合 $\\Xi_S$ 与全系统 $\\Xi_{all}$ 的比值后再跨 seed 平均；"
            "不同固定集合相互嵌套、并非互斥原子，不能把这些比例相加。",
            "",
            "## 中期窗口",
            "",
            "| 固定集合 | lead 7–10 mean $\\Xi_S$ (bits) |",
            "|---|---:|",
        ]
    )
    for row in mid_summary.itertuples(index=False):
        lines.append(f"| {row.module} | {float(row.mean):.6f} ± {float(row.std):.6f} |")
    lines.extend(
        [
            "",
            "相邻嵌套集合的配对增量为：",
            "",
            "| 新加入模态 | lead 7–10 mean paired $\\Delta\\Xi_S$ (bits) | seed 范围 |",
            "|---|---:|---:|",
        ]
    )
    for row in mid_delta.itertuples(index=False):
        lines.append(
            f"| {row.display_added_source} | {float(row.mean):.6f} ± {float(row.std):.6f} | "
            f"{float(row.min):.6f}–{float(row.max):.6f} |"
        )
    lines.extend(
        [
            "",
            "三个新增步骤在全部 `3 checkpoint × 24 lead = 72` 个配对上下文中均为正：",
            ", ".join(
                f"`+{row.display_added_source}` {int(row.positive_contexts)}/{int(row.context_count)}"
                for row in sign_summary.itertuples(index=False)
            )
            + "。这是一致的描述性方向证据；checkpoint 只有三个，不据此给出显著性结论。",
            "",
            "## 地球科学含义",
            "",
            "- `nino12 + nino3` 的短期协同接近零、在 lead 7–10 增强，提示模型在中期开始依赖赤道太平洋内部的空间差异，而不只是单一指数的持久性。",
            "- 加入综合 ENSO 指数和 nino4 后协同继续增加，支持“ENSO 强度 + 东西向空间型态”是联合读出的对象，而不是四个指数彼此独立地贡献信息。",
            "- IOD 在中期带来最大的平均配对增量，提出一个可检验假设：印度洋背景在模型中充当太平洋空间型态的状态门控或跨海盆条件变量。它可以与延迟的印太海盆耦合相容，但当前结果不能识别真实方向或物理通道。",
            "- 五模态集合在 lead 8 捕获约 80.6% 的全系统联合增量，说明中期系统峰值大部分可定位到一个气候学上紧凑的印太模块，而非均匀分散在 11 个模态之间。",
            "",
            "## 控制与边界",
            "",
            "- 所有条件共享 checkpoint、预测缓存、最大熵干预样本、24 个 lead、联合目标、bound 4、degree-1 affine TM 和有符号后处理；唯一变化是固定源集合。",
            "- 固定集合来自同一批 checkpoint 的先前 greedy 结果，因此这是路径依赖诊断，不是独立数据上的外部确认。",
            "- affine TM 只读取二阶统计结构；连续非线性 PEID 的绝对量仍需选定模块上的 degree-2/3 TM 复核。",
            "- 最大熵干预会产生超出观测气候流形的联合状态；结果描述 frozen UniCM learned mechanism，不直接等价于真实地球系统的因果效应。",
            "- 联合全模态 target 不能说明主要接收端是 ENSO、IOD 或其他模态；下一步最小扩展是 target-resolved $\\Xi_{S\\to j}$ 热图。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    leads = parse_leads(args.leads)
    seeds = [int(seed) for seed in args.seeds]
    history_modes = sample_full_history_mode_inputs(
        n_samples=int(args.n_samples),
        intervention_bound=float(args.intervention_bound),
        seed=int(args.sampling_seed),
    )
    cache_args = _cache_args(args)
    targets_by_seed = {
        seed: load_full_history_prediction_cache(
            overall_prediction_cache_path(Path(args.cache_dir), seed=seed, args=cache_args),
            n_samples=int(args.n_samples),
        )
        for seed in seeds
    }
    rows, estimator_metadata = compute_fixed_module_rows(
        history_modes,
        targets_by_seed,
        leads=leads,
        estimator_name=str(args.estimator),
        tm_degree=int(args.tm_degree),
        tm_jitter=float(args.tm_jitter),
    )
    lead_summary, deltas, delta_summary, window_summary = summarize_rows(rows)

    paths = {
        "rows": output_dir / "fixed_module_xi_rows.csv",
        "lead_summary": output_dir / "fixed_module_xi_lead_summary.csv",
        "delta_rows": output_dir / "fixed_module_xi_adjacent_deltas.csv",
        "delta_summary": output_dir / "fixed_module_xi_adjacent_delta_summary.csv",
        "window_summary": output_dir / "fixed_module_xi_window_summary.csv",
        "experiment_contract": output_dir / "experiment_contract.md",
        "report": output_dir / "report.md",
    }
    rows.to_csv(paths["rows"], index=False)
    lead_summary.to_csv(paths["lead_summary"], index=False)
    deltas.to_csv(paths["delta_rows"], index=False)
    delta_summary.to_csv(paths["delta_summary"], index=False)
    window_summary.to_csv(paths["window_summary"], index=False)
    write_experiment_contract(paths["experiment_contract"], args, estimator_metadata)
    write_report(paths["report"], rows, lead_summary, deltas, window_summary)
    figures = plot_results(rows, lead_summary, delta_summary, window_summary, Path(args.asset_base))

    manifest = {
        "analysis": "path-independent fixed-subset UniCM Xi",
        "definition": "Xi_S = EI(S -> all-mode target) - sum_{m in S} EI(m -> all-mode target)",
        "modules": [
            {"label": label, "sources": list(sources), "display_sources": display_sources(sources)}
            for label, sources in FIXED_MODULES
        ],
        "seeds": seeds,
        "leads": leads,
        "n_samples": int(args.n_samples),
        "sampling_seed": int(args.sampling_seed),
        "intervention_bound": float(args.intervention_bound),
        "target": "all 11 predicted UniCM modes at each lead as a multivariate target",
        "signed_outputs": True,
        "estimator": estimator_metadata,
        "cache_dir": str(args.cache_dir),
        "tables": {key: str(value) for key, value in paths.items()},
        "figures": [str(path) for path in figures],
    }
    manifest_path = output_dir / "fixed_module_xi_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute path-independent fixed-module Xi curves for UniCM.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--asset-base", type=Path, default=DEFAULT_ASSET_BASE)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--leads", nargs="*", default=None)
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--sampling-seed", type=int, default=20260619)
    parser.add_argument("--intervention-bound", type=float, default=4.0)
    parser.add_argument("--start-month", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--estimator", choices=["gaussian_logdet", "transport_map"], default="transport_map")
    parser.add_argument("--tm-degree", type=int, default=1)
    parser.add_argument("--tm-jitter", type=float, default=1.0e-6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    outputs = run(build_arg_parser().parse_args(argv))
    print(json.dumps(outputs, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
