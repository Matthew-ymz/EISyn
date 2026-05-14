"""Known nonlinear dynamics experiment helpers for ``exp/tm_nonlinear.ipynb``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from yrd.transport_map import summarize_two_source_synergy_transport_map


DEFAULT_FIG_DIR = Path("fig/transport_map_mutual_information")
DEFAULT_ALPHA_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_L_VALUES = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
DEFAULT_L_SWEEP_ALPHA_VALUES = (0.0, 0.5, 1.0)
DEFAULT_NOISE_SWEEP_ALPHA_VALUES = (0.0, 0.5, 1.0)
DEFAULT_NOISE_STD_VALUES = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class TmNonlinearConfig:
    seed: int = 11
    n_samples: int = 4000
    repeats: int = 4
    input_box_width: float = 2.0
    q1_noise_std: float = 0.05
    high_noise_std: float = 0.6
    fig_dir: Path = DEFAULT_FIG_DIR

    @property
    def resolved_fig_dir(self) -> Path:
        return Path(self.fig_dir)


@dataclass(frozen=True)
class ShapConfig:
    n_samples: int = 2000
    repeats: int = 2
    shap_sample_size: int = 500
    test_size: float = 0.25
    n_estimators: int = 120
    min_samples_leaf: int = 5


def simulate_alpha_case_intervention(
    *,
    alpha: float,
    n_samples: int,
    L: float,
    q1_noise_std: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    half = float(L) / 2.0
    q2_t = rng.uniform(-half, half, size=n_samples)
    q3_t = rng.uniform(-half, half, size=n_samples)
    q1_next = alpha * np.sin(q2_t * q3_t) + (1.0 - alpha) * q2_t
    q1_next = q1_next + float(q1_noise_std) * rng.normal(size=n_samples)
    return pd.DataFrame({"alpha": float(alpha), "q2_t": q2_t, "q3_t": q3_t, "q1_next": q1_next})


def estimate_tm_alpha_metrics(df: pd.DataFrame) -> dict[str, float]:
    summary = summarize_two_source_synergy_transport_map(
        df[["q2_t"]].to_numpy(),
        df[["q3_t"]].to_numpy(),
        df[["q1_next"]].to_numpy(),
    )
    return {
        "tm_ei": float(summary["joint_ei"]),
        "tm_syn": float(summary["syn"]),
        "tm_single_q2": float(summary["left_ei"]),
        "tm_single_q3": float(summary["right_ei"]),
    }


def run_alpha_sweep_tm(
    *,
    alpha_values: tuple[float, ...] = DEFAULT_ALPHA_VALUES,
    n_samples: int,
    repeats: int,
    L: float,
    q1_noise_std: float,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for alpha_index, alpha in enumerate(alpha_values):
        for repeat in range(repeats):
            run_seed = seed + 1000 * alpha_index + repeat
            df = simulate_alpha_case_intervention(
                alpha=float(alpha),
                n_samples=n_samples,
                L=L,
                q1_noise_std=q1_noise_std,
                seed=run_seed,
            )
            rows.append(
                {
                    "alpha": float(alpha),
                    "repeat": int(repeat),
                    "L": float(L),
                    "q1_noise_std": float(q1_noise_std),
                    **estimate_tm_alpha_metrics(df),
                }
            )
    return pd.DataFrame(rows)


def summarize_tm_runs(results_df: pd.DataFrame, *, group_columns: list[str]) -> pd.DataFrame:
    summary = (
        results_df.groupby(group_columns, as_index=False)
        .agg(
            tm_ei_mean=("tm_ei", "mean"),
            tm_ei_std=("tm_ei", "std"),
            tm_syn_mean=("tm_syn", "mean"),
            tm_syn_std=("tm_syn", "std"),
            tm_single_q2_mean=("tm_single_q2", "mean"),
            tm_single_q2_std=("tm_single_q2", "std"),
            tm_single_q3_mean=("tm_single_q3", "mean"),
            tm_single_q3_std=("tm_single_q3", "std"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    for metric in ("tm_syn", "tm_single_q2", "tm_single_q3"):
        summary[f"{metric}_ratio"] = np.where(
            np.abs(summary["tm_ei_mean"]) > 1e-12,
            summary[f"{metric}_mean"] / summary["tm_ei_mean"],
            np.nan,
        )
    return summary


def run_l_sweep_tm(
    *,
    l_values: tuple[float, ...] = DEFAULT_L_VALUES,
    alpha_values: tuple[float, ...] = DEFAULT_L_SWEEP_ALPHA_VALUES,
    config: TmNonlinearConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for l_index, L in enumerate(l_values):
        for alpha_index, alpha in enumerate(alpha_values):
            for repeat in range(config.repeats):
                run_seed = config.seed + 20000 + 10000 * l_index + 1000 * alpha_index + repeat
                df = simulate_alpha_case_intervention(
                    alpha=float(alpha),
                    n_samples=config.n_samples,
                    L=float(L),
                    q1_noise_std=config.q1_noise_std,
                    seed=run_seed,
                )
                rows.append(
                    {
                        "L": float(L),
                        "alpha": float(alpha),
                        "repeat": int(repeat),
                        "q1_noise_std": float(config.q1_noise_std),
                        **estimate_tm_alpha_metrics(df),
                    }
                )
    return pd.DataFrame(rows)


def run_fixed_alpha_noise_sweep_tm(
    *,
    alpha_values: tuple[float, ...] = DEFAULT_NOISE_SWEEP_ALPHA_VALUES,
    noise_std_values: tuple[float, ...] = DEFAULT_NOISE_STD_VALUES,
    config: TmNonlinearConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for alpha_index, alpha in enumerate(alpha_values):
        for noise_index, q1_noise_std in enumerate(noise_std_values):
            for repeat in range(config.repeats):
                run_seed = config.seed + 80000 + 10000 * alpha_index + 1000 * noise_index + repeat
                df = simulate_alpha_case_intervention(
                    alpha=float(alpha),
                    n_samples=config.n_samples,
                    L=config.input_box_width,
                    q1_noise_std=float(q1_noise_std),
                    seed=run_seed,
                )
                rows.append(
                    {
                        "alpha": float(alpha),
                        "q1_noise_std": float(q1_noise_std),
                        "repeat": int(repeat),
                        "L": float(config.input_box_width),
                        **estimate_tm_alpha_metrics(df),
                    }
                )
    return pd.DataFrame(rows)


def load_or_run_csvs(
    *,
    runs_path: Path,
    summary_path: Path,
    runner: Any,
    summarizer: Any,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not force and runs_path.exists() and summary_path.exists():
        return pd.read_csv(runs_path), pd.read_csv(summary_path)
    runs = runner()
    summary = summarizer(runs)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    return runs, summary


def _component_specs() -> list[tuple[str, str, str, str, str]]:
    return [
        ("tm_ei_mean", "tm_ei_std", r"$EI^{\mathrm{tm}}(Q_2,Q_3 \to Q_1)$", "#2563eb", "--"),
        ("tm_syn_mean", "tm_syn_std", r"$Syn^{\mathrm{tm}}$", "#d97706", "-"),
        ("tm_single_q2_mean", "tm_single_q2_std", r"$EI^{\mathrm{tm}}(Q_2 \to Q_1)$", "#15803d", "-"),
        ("tm_single_q3_mean", "tm_single_q3_std", r"$EI^{\mathrm{tm}}(Q_3 \to Q_1)$", "#b91c1c", "-"),
    ]


def plot_alpha_sweep_tm(summary_df: pd.DataFrame, *, output_dir: Path) -> tuple[Any, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    component_specs = _component_specs()
    line_width = 2.4
    marker_size = 5.8

    decomp_fig, decomp_ax = plt.subplots(figsize=(9.8, 5.2), constrained_layout=True)
    for mean_col, _, label, color, linestyle in component_specs:
        decomp_ax.plot(
            summary_df["alpha"],
            summary_df[mean_col],
            marker="o",
            markersize=marker_size,
            linewidth=line_width,
            linestyle=linestyle,
            color=color,
            label=label,
        )
    decomp_ax.set_xlabel("alpha")
    decomp_ax.set_ylabel("nats")
    decomp_ax.set_xticks(summary_df["alpha"])
    decomp_ax.grid(alpha=0.25)
    decomp_ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    share_fig, share_ax = plt.subplots(figsize=(9.8, 5.2), constrained_layout=True)
    ratio_specs = [
        ("tm_syn_ratio", "Syn / EI", "#d97706"),
        ("tm_single_q2_ratio", r"$EI(Q_2 \to Q_1) / EI$", "#15803d"),
        ("tm_single_q3_ratio", r"$EI(Q_3 \to Q_1) / EI$", "#b91c1c"),
    ]
    for ratio_col, label, color in ratio_specs:
        share_ax.plot(summary_df["alpha"], summary_df[ratio_col], marker="o", markersize=marker_size, linewidth=line_width, color=color, label=label)
    share_ax.set_xlabel("alpha")
    share_ax.set_ylabel("share")
    share_ax.set_xticks(summary_df["alpha"])
    share_ax.set_ylim(-0.05, 1.05)
    share_ax.grid(alpha=0.25)
    share_ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    decomp_fig.savefig(output_dir / "tm_alpha_ei_decomposition.png", dpi=220, bbox_inches="tight")
    decomp_fig.savefig(output_dir / "tm_alpha_ei_decomposition.pdf", bbox_inches="tight")
    share_fig.savefig(output_dir / "tm_alpha_share_ratio.png", dpi=220, bbox_inches="tight")
    share_fig.savefig(output_dir / "tm_alpha_share_ratio.pdf", bbox_inches="tight")
    return decomp_fig, share_fig


def plot_alpha_high_noise_decomposition(summary_df: pd.DataFrame, *, output_dir: Path) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.8, 5.2), constrained_layout=True)
    for mean_col, _, label, color, linestyle in _component_specs():
        ax.plot(summary_df["alpha"], summary_df[mean_col], marker="o", markersize=5.8, linewidth=2.4, linestyle=linestyle, color=color, label=label)
    ax.set_xlabel("alpha")
    ax.set_ylabel("nats")
    ax.set_xticks(summary_df["alpha"])
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(output_dir / "tm_alpha_high_noise_ei_decomposition.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "tm_alpha_high_noise_ei_decomposition.pdf", bbox_inches="tight")
    return fig


def plot_l_sweep_tm(summary_df: pd.DataFrame, *, output_dir: Path) -> tuple[Any, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_values = list(summary_df["alpha"].drop_duplicates())
    l_values = list(summary_df["L"].drop_duplicates())
    reliable_ratio_df = summary_df.copy()
    reliable_ratio_df["tm_syn_ratio_reliable"] = reliable_ratio_df["tm_syn_ratio"].where(reliable_ratio_df["tm_ei_mean"] >= 0.05)
    syn_ratio_grid = reliable_ratio_df.pivot(index="alpha", columns="L", values="tm_syn_ratio_reliable").loc[alpha_values, l_values]
    syn_grid = summary_df.pivot(index="alpha", columns="L", values="tm_syn_mean").loc[alpha_values, l_values]

    heatmap_fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    for ax, grid, colorbar_label, cmap in [
        (axes[0], syn_ratio_grid, "Syn / EI", "viridis"),
        (axes[1], syn_grid, r"$Syn^{\mathrm{tm}}$ (nats)", "magma"),
    ]:
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad("#e5e7eb")
        image = ax.imshow(np.ma.masked_invalid(grid.to_numpy(dtype=float)), aspect="auto", origin="lower", cmap=cmap_obj)
        ax.set_xlabel("L")
        ax.set_ylabel("alpha")
        ax.set_xticks(np.arange(len(l_values)), labels=[f"{value:g}" for value in l_values])
        ax.set_yticks(np.arange(len(alpha_values)), labels=[f"{value:g}" for value in alpha_values])
        for row_index, alpha in enumerate(alpha_values):
            for col_index, L in enumerate(l_values):
                value = grid.loc[alpha, L]
                label = "NA" if pd.isna(value) else f"{value:.2f}"
                ax.text(col_index, row_index, label, ha="center", va="center", fontsize=8, color="#111827" if pd.isna(value) else "white")
        heatmap_fig.colorbar(image, ax=ax, shrink=0.86, label=colorbar_label)

    line_fig, line_axes = plt.subplots(1, 2, figsize=(13.2, 4.9), constrained_layout=True)
    colors = {0.0: "#2563eb", 0.5: "#d97706", 1.0: "#15803d"}
    for alpha in alpha_values:
        part = summary_df.loc[summary_df["alpha"] == alpha].sort_values("L")
        color = colors.get(float(alpha), "#4b5563")
        reliable = part.copy()
        reliable["tm_syn_ratio_reliable"] = reliable["tm_syn_ratio"].where(reliable["tm_ei_mean"] >= 0.05)
        line_axes[0].plot(reliable["L"], reliable["tm_syn_ratio_reliable"], marker="o", linewidth=2.2, color=color, label=f"alpha = {alpha:g}")
        ratio_std = reliable["tm_syn_std"].to_numpy(dtype=float) / np.maximum(reliable["tm_ei_mean"].to_numpy(dtype=float), 1e-12)
        ratio_mean = reliable["tm_syn_ratio_reliable"].to_numpy(dtype=float)
        line_axes[0].fill_between(reliable["L"].to_numpy(dtype=float), np.maximum(0.0, ratio_mean - ratio_std), ratio_mean + ratio_std, color=color, alpha=0.12, linewidth=0)
    line_axes[0].set_xlabel("L")
    line_axes[0].set_ylabel("Syn / EI")
    line_axes[0].set_ylim(-0.05, 1.05)
    line_axes[0].grid(alpha=0.25)
    line_axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    alpha_one = summary_df.loc[np.isclose(summary_df["alpha"], 1.0)].sort_values("L")
    x_values = alpha_one["L"].to_numpy(dtype=float)
    for mean_col, std_col, label, color, linestyle in _component_specs():
        mean_values = alpha_one[mean_col].to_numpy(dtype=float)
        std_values = alpha_one[std_col].fillna(0.0).to_numpy(dtype=float)
        line_axes[1].plot(x_values, mean_values, marker="o", linewidth=2.2, linestyle=linestyle, color=color, label=label)
        line_axes[1].fill_between(x_values, mean_values - std_values, mean_values + std_values, color=color, alpha=0.10, linewidth=0)
    line_axes[1].set_xlabel("L")
    line_axes[1].set_ylabel("nats")
    line_axes[1].grid(alpha=0.25)
    line_axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    heatmap_fig.savefig(output_dir / "tm_l_sensitivity_heatmap.png", dpi=220, bbox_inches="tight")
    heatmap_fig.savefig(output_dir / "tm_l_sensitivity_heatmap.pdf", bbox_inches="tight")
    line_fig.savefig(output_dir / "tm_l_sensitivity_lines.png", dpi=220, bbox_inches="tight")
    line_fig.savefig(output_dir / "tm_l_sensitivity_lines.pdf", bbox_inches="tight")
    return heatmap_fig, line_fig


def plot_fixed_alpha_noise_sweep_tm(summary_df: pd.DataFrame, *, output_dir: Path) -> tuple[Any, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_values = list(summary_df["alpha"].drop_duplicates())
    component_specs = _component_specs()

    decomp_fig, decomp_axes = plt.subplots(1, len(alpha_values), figsize=(15.6, 4.8), sharey=True, constrained_layout=True)
    if len(alpha_values) == 1:
        decomp_axes = [decomp_axes]
    for ax, alpha in zip(decomp_axes, alpha_values):
        part = summary_df.loc[np.isclose(summary_df["alpha"], alpha)].sort_values("q1_noise_std")
        x_values = part["q1_noise_std"].to_numpy(dtype=float)
        for mean_col, std_col, label, color, linestyle in component_specs:
            mean_values = part[mean_col].to_numpy(dtype=float)
            std_values = part[std_col].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x_values, mean_values, marker="o", markersize=5.0, linewidth=2.1, linestyle=linestyle, color=color, label=label)
            ax.fill_between(x_values, mean_values - std_values, mean_values + std_values, color=color, alpha=0.10, linewidth=0)
        ax.set_title(f"alpha = {alpha:g}")
        ax.set_xlabel("noise std")
        ax.grid(alpha=0.25)
    decomp_axes[0].set_ylabel("nats")
    decomp_fig.legend(handles=[Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2.1, label=label) for _, _, label, color, linestyle in component_specs], loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    share_fig, share_axes = plt.subplots(1, len(alpha_values), figsize=(15.6, 4.8), sharey=True, constrained_layout=True)
    if len(alpha_values) == 1:
        share_axes = [share_axes]
    ratio_specs = [
        ("tm_syn_ratio", "Syn / EI", "#d97706"),
        ("tm_single_q2_ratio", r"$EI(Q_2 \to Q_1) / EI$", "#15803d"),
        ("tm_single_q3_ratio", r"$EI(Q_3 \to Q_1) / EI$", "#b91c1c"),
    ]
    for ax, alpha in zip(share_axes, alpha_values):
        part = summary_df.loc[np.isclose(summary_df["alpha"], alpha)].sort_values("q1_noise_std")
        x_values = part["q1_noise_std"].to_numpy(dtype=float)
        for ratio_col, label, color in ratio_specs:
            ax.plot(x_values, part[ratio_col], marker="o", markersize=5.0, linewidth=2.1, color=color, label=label)
        ax.set_title(f"alpha = {alpha:g}")
        ax.set_xlabel("noise std")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
    share_axes[0].set_ylabel("share")
    share_fig.legend(handles=[Line2D([0], [0], color=color, linewidth=2.1, label=label) for _, label, color in ratio_specs], loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    decomp_fig.savefig(output_dir / "tm_fixed_alpha_noise_sweep_decomposition.png", dpi=220, bbox_inches="tight")
    decomp_fig.savefig(output_dir / "tm_fixed_alpha_noise_sweep_decomposition.pdf", bbox_inches="tight")
    share_fig.savefig(output_dir / "tm_fixed_alpha_noise_sweep_share_ratio.png", dpi=220, bbox_inches="tight")
    share_fig.savefig(output_dir / "tm_fixed_alpha_noise_sweep_share_ratio.pdf", bbox_inches="tight")
    return decomp_fig, share_fig


def _require_shap() -> Any:
    try:
        import shap  # type: ignore
    except ImportError as exc:
        raise RuntimeError("This experiment requires SHAP. Install it with `python -m pip install shap`.") from exc
    return shap


def compute_shap_alpha_comparison(
    *,
    alpha_values: tuple[float, ...] = DEFAULT_ALPHA_VALUES,
    config: TmNonlinearConfig,
    shap_config: ShapConfig = ShapConfig(),
) -> pd.DataFrame:
    shap = _require_shap()
    rows: list[dict[str, float | int]] = []
    for alpha_index, alpha in enumerate(alpha_values):
        for repeat in range(shap_config.repeats):
            run_seed = config.seed + 120000 + 1000 * alpha_index + repeat
            df = simulate_alpha_case_intervention(
                alpha=float(alpha),
                n_samples=shap_config.n_samples,
                L=config.input_box_width,
                q1_noise_std=config.q1_noise_std,
                seed=run_seed,
            )
            x = df[["q2_t", "q3_t"]].to_numpy(dtype=float)
            y = df["q1_next"].to_numpy(dtype=float)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=shap_config.test_size, random_state=run_seed)
            model = RandomForestRegressor(
                n_estimators=shap_config.n_estimators,
                min_samples_leaf=shap_config.min_samples_leaf,
                random_state=run_seed,
                n_jobs=1,
            )
            model.fit(x_train, y_train)
            sample_count = min(int(shap_config.shap_sample_size), x_test.shape[0])
            sample = x_test[:sample_count]
            explainer = shap.TreeExplainer(model)
            shap_values = np.asarray(explainer.shap_values(sample), dtype=float)
            interactions = np.asarray(explainer.shap_interaction_values(sample), dtype=float)
            mean_abs_q2 = float(np.mean(np.abs(shap_values[:, 0])))
            mean_abs_q3 = float(np.mean(np.abs(shap_values[:, 1])))
            pair_interaction = float(np.mean(np.abs(interactions[:, 0, 1])))
            additive_total = mean_abs_q2 + mean_abs_q3
            rows.append(
                {
                    "alpha": float(alpha),
                    "repeat": int(repeat),
                    "L": float(config.input_box_width),
                    "q1_noise_std": float(config.q1_noise_std),
                    "shap_mean_abs_q2": mean_abs_q2,
                    "shap_mean_abs_q3": mean_abs_q3,
                    "shap_pair_interaction": pair_interaction,
                    "shap_q2_share": mean_abs_q2 / additive_total if additive_total > 1e-12 else np.nan,
                    "shap_q3_share": mean_abs_q3 / additive_total if additive_total > 1e-12 else np.nan,
                    "shap_interaction_share": pair_interaction / (additive_total + pair_interaction) if additive_total + pair_interaction > 1e-12 else np.nan,
                    "model_r2": float(r2_score(y_test, model.predict(x_test))),
                }
            )
    return pd.DataFrame(rows)


def summarize_shap_runs(runs: pd.DataFrame) -> pd.DataFrame:
    return (
        runs.groupby(["alpha", "L", "q1_noise_std"], as_index=False)
        .agg(
            shap_mean_abs_q2_mean=("shap_mean_abs_q2", "mean"),
            shap_mean_abs_q2_std=("shap_mean_abs_q2", "std"),
            shap_mean_abs_q3_mean=("shap_mean_abs_q3", "mean"),
            shap_mean_abs_q3_std=("shap_mean_abs_q3", "std"),
            shap_pair_interaction_mean=("shap_pair_interaction", "mean"),
            shap_pair_interaction_std=("shap_pair_interaction", "std"),
            shap_interaction_share_mean=("shap_interaction_share", "mean"),
            shap_interaction_share_std=("shap_interaction_share", "std"),
            model_r2_mean=("model_r2", "mean"),
            model_r2_std=("model_r2", "std"),
        )
        .sort_values("alpha")
        .reset_index(drop=True)
    )


def build_shap_peid_comparison(tm_summary: pd.DataFrame, shap_summary: pd.DataFrame) -> pd.DataFrame:
    cols = ["alpha", "L", "q1_noise_std", "tm_ei_mean", "tm_syn_mean", "tm_syn_ratio", "tm_single_q2_ratio", "tm_single_q3_ratio"]
    comparison = tm_summary[cols].merge(shap_summary, on=["alpha", "L", "q1_noise_std"], how="inner")
    return comparison.sort_values("alpha").reset_index(drop=True)


def plot_shap_peid_comparison(comparison: pd.DataFrame, *, output_dir: Path) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, wspace=0.08, hspace=0.04)
    alpha = comparison["alpha"].to_numpy(dtype=float)
    styles = {
        "peid_signal": {"color": "#d9822b", "marker": "o", "linewidth": 1.6, "markersize": 3.2, "linestyle": "-"},
        "shap_signal": {"color": "#3b6fb6", "marker": "o", "linewidth": 1.6, "markersize": 3.2, "linestyle": "--"},
        "q2_peid": {"color": "#2f7d59", "marker": "o", "linewidth": 1.4, "markersize": 3.0},
        "q3_peid": {"color": "#a33a3a", "marker": "o", "linewidth": 1.4, "markersize": 3.0},
        "q2_shap": {"color": "#2f7d59", "marker": "o", "linewidth": 1.4, "markersize": 3.0, "linestyle": "--"},
        "q3_shap": {"color": "#a33a3a", "marker": "o", "linewidth": 1.4, "markersize": 3.0, "linestyle": "--"},
    }

    handles = []
    handles.append(
        axes[0].plot(alpha, comparison["tm_syn_ratio"], label="PEID synergy / EI", **styles["peid_signal"])[0]
    )
    handles.append(
        axes[0].plot(alpha, comparison["shap_interaction_share_mean"], label="SHAP interaction share", **styles["shap_signal"])[0]
    )
    axes[0].set_ylabel("interaction share")

    handles.append(
        axes[1].plot(alpha, comparison["tm_single_q2_ratio"], label="PEID Q2 source share", **styles["q2_peid"])[0]
    )
    handles.append(
        axes[1].plot(alpha, comparison["tm_single_q3_ratio"], label="PEID Q3 source share", **styles["q3_peid"])[0]
    )
    handles.append(
        axes[1].plot(alpha, comparison["shap_mean_abs_q2_mean"], label="SHAP mean |Q2|", **styles["q2_shap"])[0]
    )
    handles.append(
        axes[1].plot(alpha, comparison["shap_mean_abs_q3_mean"], label="SHAP mean |Q3|", **styles["q3_shap"])[0]
    )
    axes[1].set_ylabel("source attribution")

    for panel_label, ax in zip(("a", "b"), axes):
        ax.text(-0.16, 1.04, panel_label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
        ax.set_xlabel(r"$\alpha$")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.grid(color="#d9dee7", alpha=0.45, linewidth=0.45)
        ax.tick_params(length=2.6, width=0.7, pad=2)

    labels = [handle.get_label() for handle in handles]
    legend_handles = [
        Line2D([0], [0], color=handle.get_color(), linestyle=handle.get_linestyle(), linewidth=handle.get_linewidth())
        for handle in handles
    ]
    fig.legend(
        legend_handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=False,
        handlelength=1.9,
        columnspacing=1.3,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    fig.savefig(output_dir / "tm_alpha_shap_peid_comparison.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "tm_alpha_shap_peid_comparison.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "tm_alpha_shap_peid_comparison.svg", bbox_inches="tight")
    fig.savefig(output_dir / "tm_alpha_shap_peid_comparison.tiff", dpi=600, bbox_inches="tight")
    return fig


def write_manifest(output_dir: Path, *, name: str, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
