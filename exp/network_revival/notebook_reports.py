from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .effective_information import (
    EIIgnitionThresholdConfig,
    PairIgnitionCostConfig,
    StateSpacePairSynergyConfig,
    StateSpaceEIConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def fmt_param(value: float) -> str:
    return str(value).replace(".", "p")


def state_space_run_id(config: StateSpaceEIConfig) -> str:
    return (
        f"fig5l_state_wout{fmt_param(config.wout)}_win{fmt_param(config.win)}_"
        f"tau{fmt_param(config.tau)}_n{config.sample_count}_seed{config.seed}"
    )


def pair_synergy_run_id(
    state_space_config: StateSpaceEIConfig,
    *,
    pair_count: int = 200,
    pair_seed: int = 42,
) -> str:
    return f"{state_space_run_id(state_space_config)}_pairs{int(pair_count)}_seed{int(pair_seed)}"


def make_state_space_config(
    *,
    sample_count: int = 10000,
    state_low: float = 0.0,
    state_high: float = 30.0,
    tau: float = 20.0,
    dt: float = 0.08,
    seed: int = 42,
    batch_size: int = 512,
    target_noise_fraction: float = 0.01,
    win: float = 20.0,
    wout: float = 5.0,
    show_progress: bool = True,
    output_root: str | Path | None = None,
) -> StateSpaceEIConfig:
    draft = StateSpaceEIConfig(
        sample_count=sample_count,
        state_low=state_low,
        state_high=state_high,
        tau=tau,
        dt=dt,
        seed=seed,
        batch_size=batch_size,
        target_noise_fraction=target_noise_fraction,
        show_progress=show_progress,
        win=win,
        wout=wout,
    )
    root = Path(output_root) if output_root is not None else REPO_ROOT / "results" / "network_revival_state_space_ei"
    return StateSpaceEIConfig(
        sample_count=sample_count,
        state_low=state_low,
        state_high=state_high,
        tau=tau,
        dt=dt,
        seed=seed,
        batch_size=batch_size,
        target_noise_fraction=target_noise_fraction,
        show_progress=show_progress,
        win=win,
        wout=wout,
        output_dir=root / state_space_run_id(draft),
    )


def make_ignition_threshold_config(
    state_space_config: StateSpaceEIConfig,
    *,
    per_stratum: int = 30,
    delta_low: float = 0.0,
    delta_high: float = 30.0,
    binary_steps: int = 10,
    success_threshold: float = 5.0,
    t_force: float = 12.0,
    dt: float = 0.08,
    tol_ss: float = 2e-3,
    show_progress: bool = True,
    output_root: str | Path | None = None,
) -> EIIgnitionThresholdConfig:
    run_id = state_space_run_id(state_space_config)
    root = (
        Path(output_root)
        if output_root is not None
        else REPO_ROOT / "results" / "network_revival_ei_ignition_threshold"
    )
    return EIIgnitionThresholdConfig(
        state_space_run_id=run_id,
        per_stratum=per_stratum,
        delta_low=delta_low,
        delta_high=delta_high,
        binary_steps=binary_steps,
        success_threshold=success_threshold,
        t_force=t_force,
        dt=dt,
        tol_ss=tol_ss,
        win=state_space_config.win,
        wout=state_space_config.wout,
        show_progress=show_progress,
        output_dir=root / run_id,
    )


def make_pair_synergy_config(
    state_space_config: StateSpaceEIConfig,
    *,
    pair_count: int = 200,
    pair_seed: int = 42,
    target_noise_fraction: float | None = None,
    seed: int | None = None,
    show_progress: bool = True,
    output_root: str | Path | None = None,
) -> StateSpacePairSynergyConfig:
    run_id = pair_synergy_run_id(state_space_config, pair_count=pair_count, pair_seed=pair_seed)
    root = (
        Path(output_root)
        if output_root is not None
        else REPO_ROOT / "results" / "network_revival_state_space_pair_synergy"
    )
    return StateSpacePairSynergyConfig(
        state_space_run_id=state_space_run_id(state_space_config),
        pair_count=pair_count,
        pair_seed=pair_seed,
        target_noise_fraction=state_space_config.target_noise_fraction
        if target_noise_fraction is None
        else target_noise_fraction,
        seed=state_space_config.seed if seed is None else seed,
        win=state_space_config.win,
        wout=state_space_config.wout,
        show_progress=show_progress,
        output_dir=root / run_id,
        state_space_output_dir=state_space_config.output_dir,
    )


def make_pair_ignition_cost_config(
    pair_synergy_config: StateSpacePairSynergyConfig,
    *,
    cost_low: float = 0.0,
    cost_high: float = 60.0,
    single_delta_low: float = 0.0,
    single_delta_high: float = 30.0,
    binary_steps: int = 10,
    success_threshold: float = 5.0,
    t_force: float = 12.0,
    dt: float = 0.08,
    tol_ss: float = 2e-3,
    show_progress: bool = True,
    output_root: str | Path | None = None,
) -> PairIgnitionCostConfig:
    run_id = f"{pair_synergy_config.state_space_run_id}_pairs{pair_synergy_config.pair_count}_seed{pair_synergy_config.pair_seed}"
    root = (
        Path(output_root)
        if output_root is not None
        else REPO_ROOT / "results" / "network_revival_pair_ignition_cost"
    )
    return PairIgnitionCostConfig(
        pair_synergy_run_id=run_id,
        cost_low=cost_low,
        cost_high=cost_high,
        single_delta_low=single_delta_low,
        single_delta_high=single_delta_high,
        binary_steps=binary_steps,
        success_threshold=success_threshold,
        t_force=t_force,
        dt=dt,
        tol_ss=tol_ss,
        win=pair_synergy_config.win,
        wout=pair_synergy_config.wout,
        show_progress=show_progress,
        output_dir=root / run_id,
    )


def summarize_state_space_result(result: dict[str, object], *, top_k: int = 30) -> dict[str, object]:
    summary = pd.DataFrame(result["node_summary"])
    final_mean = np.asarray(result["final_mean_activity"], dtype=float)
    ranked = summary.sort_values("ei_final_state", ascending=False).reset_index(drop=True)
    hist_counts, _ = np.histogram(final_mean, bins=40)
    return {
        "summary": summary,
        "ranked": ranked.head(int(top_k)).copy(),
        "top_nodes": ranked.head(3).copy(),
        "median_ei": float(summary["ei_final_state"].median()),
        "final_mean_min": float(np.min(final_mean)),
        "final_mean_median": float(np.median(final_mean)),
        "final_mean_max": float(np.max(final_mean)),
        "nonempty_hist_bins": int(np.count_nonzero(hist_counts)),
    }


def plot_state_space_report(
    result: dict[str, object],
    config: StateSpaceEIConfig,
    figure_dir: str | Path,
    *,
    top_k: int = 30,
) -> dict[str, Path]:
    out_dir = Path(figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = summarize_state_space_result(result, top_k=top_k)
    rank_df = stats["ranked"]
    final_mean_activity = np.asarray(result["final_mean_activity"], dtype=float)
    community_colors = {"M1": "#4C78A8", "M2": "#D1495B"}
    legend_handles = [Patch(facecolor=color, label=community) for community, color in community_colors.items()]
    paths = {
        "rank_top": out_dir / "fig5l_state_space_node_ei_rankings.png",
        "rank_tail": out_dir / "fig5l_state_space_node_ei_rankings_tail.png",
        "final_mean_hist": out_dir / "fig5l_state_space_final_mean_distribution.png",
    }

    x_pos = np.arange(len(rank_df))
    fig, ax = plt.subplots(figsize=(9.2, 4.2), constrained_layout=True)
    ax.bar(x_pos, rank_df["ei_final_state"].to_numpy(dtype=float), color=_community_colors(rank_df, community_colors))
    ax.set_ylabel(rf"$I(X_i(0); \mathbf{{x}}(\tau={config.tau:g}))$")
    ax.set_xlabel("Initial-state source node")
    ax.set_xticks(x_pos, [str(node) for node in rank_df["node"]], rotation=45, ha="right")
    ax.tick_params(direction="in")
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_png_pdf(fig, paths["rank_top"])
    plt.close(fig)

    tail_df = rank_df.iloc[2:].copy()
    fig, ax = plt.subplots(figsize=(9.2, 3.8), constrained_layout=True)
    tail_x = np.arange(len(tail_df))
    ax.bar(tail_x, tail_df["ei_final_state"].to_numpy(dtype=float), color=_community_colors(tail_df, community_colors))
    ax.set_ylabel(rf"$I(X_i(0); \mathbf{{x}}(\tau={config.tau:g}))$")
    ax.set_xlabel(f"Initial-state source node, ranks 3-{top_k}")
    ax.set_xticks(tail_x, [str(node) for node in tail_df["node"]], rotation=45, ha="right")
    ax.tick_params(direction="in")
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_png_pdf(fig, paths["rank_tail"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    ax.hist(final_mean_activity, bins=40, color="#6A994E", edgecolor="white", linewidth=0.4)
    ax.set_xlabel(rf"$\bar{{x}}(\tau={config.tau:g})$")
    ax.set_ylabel("Sample count")
    ax.tick_params(direction="in")
    _save_png_pdf(fig, paths["final_mean_hist"])
    plt.close(fig)
    return paths


def summarize_ignition_threshold_result(result: dict[str, object], config: EIIgnitionThresholdConfig) -> dict[str, object]:
    threshold = pd.DataFrame(result["threshold_rows"]).sort_values("ei_rank").reset_index(drop=True)
    threshold["is_censored"] = threshold["threshold_status"].ne("finite")
    threshold["effective_delta"] = threshold["critical_delta"].where(
        ~threshold["is_censored"], config.delta_high
    )
    finite = threshold.loc[~threshold["is_censored"]].copy()
    if len(finite) >= 3 and finite["critical_delta"].nunique() > 1 and finite["ei_final_state"].nunique() > 1:
        try:
            from scipy.stats import spearmanr

            corr = spearmanr(finite["ei_final_state"], finite["critical_delta"])
            spearman_r = float(corr.statistic)
            spearman_p = float(corr.pvalue)
        except Exception:
            spearman_r = float(finite["ei_final_state"].corr(finite["critical_delta"], method="spearman"))
            spearman_p = float("nan")
    else:
        spearman_r = float("nan")
        spearman_p = float("nan")
    stratum_order = ("top", "middle", "bottom")
    return {
        "threshold": threshold,
        "finite_count": int(len(finite)),
        "total_count": int(len(threshold)),
        "censored_rate": float(threshold["is_censored"].mean()) if len(threshold) else float("nan"),
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "stratum_medians": threshold.groupby("ei_stratum")["effective_delta"].median().reindex(stratum_order),
    }


def plot_ignition_threshold_report(
    result: dict[str, object],
    config: EIIgnitionThresholdConfig,
    figure_dir: str | Path,
) -> dict[str, Path]:
    out_dir = Path(figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = summarize_ignition_threshold_result(result, config)
    threshold = stats["threshold"]
    paths = {
        "ei_vs_delta": out_dir / "ei_vs_critical_delta.png",
        "stratum": out_dir / "threshold_by_ei_stratum.png",
        "rank_curve": out_dir / "ei_rank_threshold_curve.png",
    }

    finite = threshold.loc[~threshold["is_censored"]]
    censored = threshold.loc[threshold["is_censored"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.scatter(finite["ei_final_state"], finite["critical_delta"], s=34, color="#4C78A8", label="Finite threshold")
    if not censored.empty:
        ax.scatter(
            censored["ei_final_state"],
            censored["effective_delta"],
            s=46,
            marker="^",
            color="#D1495B",
            label="Censored above max Delta",
        )
    ax.set_xlabel("State-space EI")
    ax.set_ylabel("Critical ignition Delta")
    ax.grid(color="0.9", linewidth=0.8)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_png_pdf(fig, paths["ei_vs_delta"])
    plt.close(fig)

    stratum_order = ("top", "middle", "bottom")
    fig, ax = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    box_data = [
        threshold.loc[threshold["ei_stratum"].eq(label), "effective_delta"].to_numpy(dtype=float)
        for label in stratum_order
    ]
    ax.boxplot(box_data, labels=[label.capitalize() for label in stratum_order], showfliers=False)
    for xpos, label in enumerate(stratum_order, start=1):
        group = threshold.loc[threshold["ei_stratum"].eq(label)]
        jitter = np.linspace(-0.08, 0.08, len(group)) if len(group) else []
        colors = np.where(group["is_censored"], "#D1495B", "#4C78A8")
        ax.scatter(np.full(len(group), xpos) + jitter, group["effective_delta"], s=22, color=colors, alpha=0.85)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="#4C78A8", label="Finite threshold"),
        Line2D([0], [0], marker="o", linestyle="none", color="#D1495B", label="Censored above max Delta"),
    ]
    ax.set_xlabel("EI stratum")
    ax.set_ylabel("Critical ignition Delta")
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_png_pdf(fig, paths["stratum"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    for label, color in [("top", "#4C78A8"), ("middle", "#6A994E"), ("bottom", "#D1495B")]:
        group = threshold.loc[threshold["ei_stratum"].eq(label)].sort_values("ei_rank")
        finite_group = group.loc[~group["is_censored"]]
        censored_group = group.loc[group["is_censored"]]
        ax.plot(
            finite_group["ei_rank"],
            finite_group["critical_delta"],
            marker="o",
            ms=4,
            lw=1.2,
            color=color,
            label=label.capitalize(),
        )
        if not censored_group.empty:
            ax.scatter(
                censored_group["ei_rank"],
                censored_group["effective_delta"],
                marker="^",
                s=46,
                color=color,
                edgecolor="black",
                linewidth=0.4,
            )
    ax.set_xlabel("EI rank")
    ax.set_ylabel("Critical ignition Delta")
    ax.grid(color="0.9", linewidth=0.8)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_png_pdf(fig, paths["rank_curve"])
    plt.close(fig)
    return paths


def summarize_pair_synergy_result(result: dict[str, object], *, top_k: int = 30) -> dict[str, object]:
    pairs = pd.DataFrame(result["pair_rows"]).sort_values("rank_synergy").reset_index(drop=True)
    return {
        "pairs": pairs,
        "ranked": pairs.head(int(top_k)).copy(),
        "top_pairs": pairs.head(5).copy(),
        "median_synergy": float(pairs["synergy"].median()) if len(pairs) else float("nan"),
        "positive_rate": float((pairs["synergy"] > 0.0).mean()) if len(pairs) else float("nan"),
    }


def plot_pair_synergy_report(
    result: dict[str, object],
    figure_dir: str | Path,
    *,
    top_k: int = 30,
) -> dict[str, Path]:
    out_dir = Path(figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = summarize_pair_synergy_result(result, top_k=top_k)
    ranked = stats["ranked"]
    paths = {"rank_top": out_dir / "state_space_pair_synergy_rankings.png"}

    x_pos = np.arange(len(ranked))
    color_map = {"M1-M1": "#4C78A8", "M1-M2": "#6A994E", "M2-M1": "#6A994E", "M2-M2": "#D1495B"}
    colors = _pair_colors(ranked, color_map)
    handles = [
        Patch(facecolor="#4C78A8", label="M1-M1"),
        Patch(facecolor="#6A994E", label="Cross-module"),
        Patch(facecolor="#D1495B", label="M2-M2"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.2), constrained_layout=True)
    ax.bar(x_pos, ranked["synergy"].to_numpy(dtype=float), color=colors)
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xlabel("State-space source pair")
    ax.set_ylabel(r"$Syn(i,j \to \mathbf{x}(\tau))$")
    labels = [f"{int(row.pair_i)}-{int(row.pair_j)}" for row in ranked.itertuples()]
    ax.set_xticks(x_pos, labels, rotation=45, ha="right")
    ax.tick_params(direction="in")
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_png_pdf(fig, paths["rank_top"])
    plt.close(fig)
    return paths


def summarize_pair_ignition_cost_result(
    result: dict[str, object],
    config: PairIgnitionCostConfig,
) -> dict[str, object]:
    costs = pd.DataFrame(result["cost_rows"]).sort_values("rank_synergy").reset_index(drop=True)
    if costs.empty:
        return {
            "costs": costs,
            "finite_count": 0,
            "total_count": 0,
            "censored_rate": float("nan"),
            "spearman_synergy_cost": float("nan"),
            "spearman_synergy_saving": float("nan"),
            "median_cost_saving": float("nan"),
        }
    costs["is_censored"] = costs["threshold_status"].ne("finite")
    costs["effective_total_cost"] = costs["critical_total_cost"].where(
        ~costs["is_censored"], config.cost_high
    )
    finite = costs.loc[~costs["is_censored"]].copy()
    spearman_cost = _safe_spearman(finite["synergy"], finite["critical_total_cost"])
    spearman_saving = _safe_spearman(costs["synergy"], costs["cost_saving"])
    return {
        "costs": costs,
        "finite_count": int(len(finite)),
        "total_count": int(len(costs)),
        "censored_rate": float(costs["is_censored"].mean()),
        "spearman_synergy_cost": spearman_cost,
        "spearman_synergy_saving": spearman_saving,
        "median_cost_saving": float(costs["cost_saving"].median()),
    }


def plot_pair_ignition_cost_report(
    result: dict[str, object],
    config: PairIgnitionCostConfig,
    figure_dir: str | Path,
) -> dict[str, Path]:
    out_dir = Path(figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = summarize_pair_ignition_cost_result(result, config)
    costs = stats["costs"]
    paths = {
        "synergy_vs_cost": out_dir / "pair_synergy_vs_total_cost.png",
        "synergy_vs_saving": out_dir / "pair_synergy_vs_cost_saving_ratio.png",
    }

    finite = costs.loc[~costs["is_censored"]]
    censored = costs.loc[costs["is_censored"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.scatter(finite["synergy"], finite["critical_total_cost"], s=34, color="#4C78A8", label="Finite threshold")
    if not censored.empty:
        ax.scatter(
            censored["synergy"],
            censored["effective_total_cost"],
            s=46,
            marker="^",
            color="#D1495B",
            label="Censored above max cost",
        )
    ax.set_xlabel("State-space pair synergy")
    ax.set_ylabel("Critical total co-ignition cost")
    ax.grid(color="0.9", linewidth=0.8)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_png_pdf(fig, paths["synergy_vs_cost"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    colors = np.where(costs["cost_saving"] >= 0.0, "#6A994E", "#D1495B")
    ax.scatter(costs["synergy"], costs["cost_saving_ratio"], s=34, color=colors, alpha=0.88)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="#6A994E", label="Pair cheaper"),
        Line2D([0], [0], marker="o", linestyle="none", color="#D1495B", label="Pair not cheaper"),
    ]
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xlabel("State-space pair synergy")
    ax.set_ylabel("Cost saving ratio")
    ax.grid(color="0.9", linewidth=0.8)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_png_pdf(fig, paths["synergy_vs_saving"])
    plt.close(fig)
    return paths


def image_paths_with_pdf(paths: dict[str, Path]) -> dict[str, tuple[Path, Path]]:
    return {key: (path, path.with_suffix(".pdf")) for key, path in paths.items()}


def _community_colors(frame: pd.DataFrame, mapping: dict[str, str]) -> list[str]:
    return [mapping[str(value)] for value in frame["community"]]


def _pair_colors(frame: pd.DataFrame, mapping: dict[str, str]) -> list[str]:
    colors = []
    for row in frame.itertuples():
        key = f"{row.community_i}-{row.community_j}"
        if key in ("M2-M1", "M1-M2"):
            key = "M1-M2"
        colors.append(mapping[key])
    return colors


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_series = pd.Series(left, dtype=float)
    right_series = pd.Series(right, dtype=float)
    valid = left_series.notna() & right_series.notna()
    left_valid = left_series.loc[valid]
    right_valid = right_series.loc[valid]
    if len(left_valid) < 3 or left_valid.nunique() <= 1 or right_valid.nunique() <= 1:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(left_valid, right_valid).statistic)
    except Exception:
        return float(left_valid.corr(right_valid, method="spearman"))


def _save_png_pdf(fig, png_path: Path) -> None:
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
