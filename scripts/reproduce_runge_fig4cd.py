#!/usr/bin/env python3
"""Reproduce Runge et al. Fig. 4c,d from local causal-effect summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULT_DIR = Path("results/runge/2015_gateways")
DEFAULT_OUTPUT = Path("fig/runge/2015_gateways/fig4cd_reproduction.png")
HIGHLIGHT_COMPONENTS = {0, 1, 2, 18}
IMPORTANT_LABEL_COMPONENTS = HIGHLIGHT_COMPONENTS | {26, 48}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def build_fig4cd_frame(
    gateway: pd.DataFrame,
    mediator: pd.DataFrame,
    total_effects: pd.DataFrame,
    *,
    effect_threshold: float = 0.05,
) -> pd.DataFrame:
    """Join ACE/ACS/AMCE summaries and compute Fig. 4c,d node-size fractions."""

    required_gateway = {"component", "paper_component", "ace", "acs"}
    required_mediator = {"component", "paper_component", "amce", "mediated_fraction"}
    required_total = {"source", "target", "total_effect"}
    missing_gateway = required_gateway - set(gateway.columns)
    missing_mediator = required_mediator - set(mediator.columns)
    missing_total = required_total - set(total_effects.columns)
    if missing_gateway:
        raise ValueError(f"gateway table missing columns: {sorted(missing_gateway)}")
    if missing_mediator:
        raise ValueError(f"mediator table missing columns: {sorted(missing_mediator)}")
    if missing_total:
        raise ValueError(f"total-effect table missing columns: {sorted(missing_total)}")

    gateway_cols = gateway.loc[:, ["component", "paper_component", "ace", "acs"]].copy()
    mediator_cols = mediator.loc[:, ["component", "amce", "mediated_fraction"]].copy()
    frame = gateway_cols.merge(mediator_cols, on="component", how="left", validate="one_to_one")

    n_components = int(frame["component"].nunique())
    denom = max(1, n_components - 1)
    max_effect = (
        total_effects.assign(abs_effect=lambda df: df["total_effect"].abs())
        .groupby(["source", "target"], as_index=False)["abs_effect"]
        .max()
    )
    nout = (
        max_effect.loc[
            (max_effect["source"] != max_effect["target"])
            & (max_effect["abs_effect"] > float(effect_threshold))
        ]
        .groupby("source")["target"]
        .nunique()
        .reindex(frame["component"], fill_value=0)
        .to_numpy(dtype=float)
        / denom
    )

    frame["nout_fraction"] = nout
    frame["mediated_fraction"] = frame["mediated_fraction"].fillna(0.0)
    frame["amce"] = frame["amce"].fillna(0.0)
    frame["is_highlight"] = frame["paper_component"].astype(int).isin(HIGHLIGHT_COMPONENTS)
    return frame.sort_values("paper_component").reset_index(drop=True)


def _r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
    return corr * corr


def _density_line(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or np.allclose(values, values[0]):
        return np.zeros_like(grid)
    try:
        from scipy.stats import gaussian_kde

        density = gaussian_kde(values)(grid)
    except Exception:
        hist, edges = np.histogram(values, bins=min(12, max(3, len(values) // 3)), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        density = np.interp(grid, centers, hist, left=0.0, right=0.0)
    max_density = float(np.nanmax(density)) if np.any(np.isfinite(density)) else 0.0
    if max_density <= 0.0:
        return np.zeros_like(grid)
    return density / max_density


def _point_sizes(fraction: pd.Series, *, base: float = 16.0, scale: float = 520.0) -> np.ndarray:
    values = np.clip(fraction.to_numpy(dtype=float), 0.0, 1.0)
    return base + scale * np.sqrt(values)


def _draw_panel(
    fig: plt.Figure,
    spec,
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    size_column: str,
    xlabel: str,
    ylabel: str,
    panel_label: str,
    size_label: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    label_offsets: dict[int, tuple[float, float]] | None = None,
) -> None:
    inner = spec.subgridspec(
        2,
        2,
        width_ratios=(0.23, 1.0),
        height_ratios=(1.0, 0.22),
        wspace=0.03,
        hspace=0.03,
    )
    ax_y = fig.add_subplot(inner[0, 0])
    ax = fig.add_subplot(inner[0, 1])
    ax_x = fig.add_subplot(inner[1, 1], sharex=ax)
    fig.add_subplot(inner[1, 0]).axis("off")

    normal = frame.loc[~frame["is_highlight"]]
    highlight = frame.loc[frame["is_highlight"]]
    ax.scatter(
        normal[x_column],
        normal[y_column],
        s=_point_sizes(normal[size_column], base=8.0, scale=260.0),
        facecolor="white",
        edgecolor="black",
        linewidth=0.45,
        alpha=0.78,
        zorder=2,
    )
    ax.scatter(
        highlight[x_column],
        highlight[y_column],
        s=_point_sizes(highlight[size_column], base=35.0, scale=720.0),
        facecolor="#ff5a5f",
        edgecolor="#a00000",
        linewidth=0.65,
        alpha=0.82,
        zorder=3,
    )

    for row in frame.itertuples():
        x = float(getattr(row, x_column))
        y = float(getattr(row, y_column))
        paper_component = int(row.paper_component)
        label = str(paper_component)
        is_important = paper_component in IMPORTANT_LABEL_COMPONENTS
        size = 6.6 if is_important else 5.0
        weight = "bold" if is_important else "normal"
        dx, dy = (label_offsets or {}).get(paper_component, (0.0, 0.0))
        ax.text(
            x + dx * (xlim[1] - xlim[0]),
            y + dy * (ylim[1] - ylim[0]),
            label,
            ha="center",
            va="center",
            fontsize=size,
            weight=weight,
            color="black",
            zorder=4,
        )

    x_grid = np.linspace(xlim[0], xlim[1], 220)
    y_grid = np.linspace(ylim[0], ylim[1], 220)
    x_density = _density_line(frame[x_column].to_numpy(dtype=float), x_grid)
    y_density = _density_line(frame[y_column].to_numpy(dtype=float), y_grid)
    ax_x.plot(x_grid, x_density, color="#777777", lw=1.8)
    ax_y.plot(y_density, y_grid, color="#777777", lw=1.8)

    for density_ax in (ax_x, ax_y):
        density_ax.set_axis_off()
    ax_x.set_ylim(0.0, 1.08)
    ax_y.set_xlim(1.08, 0.0)
    ax_y.set_ylim(*ylim)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(width=0.65, length=2.5, pad=1.5)
    ax.text(-0.34, 1.04, panel_label, transform=ax.transAxes, fontsize=9, weight="bold")
    ax.text(0.03, -0.30, rf"$R^2={_r2(frame[x_column], frame[y_column]):.2f}$", transform=ax.transAxes)

    ref = float(frame[size_column].max())
    ax.scatter(
        [xlim[0] + 0.12 * (xlim[1] - xlim[0])],
        [ylim[1] - 0.10 * (ylim[1] - ylim[0])],
        s=_point_sizes(pd.Series([ref]), base=35.0, scale=720.0),
        facecolor="white",
        edgecolor="#555555",
        linewidth=0.55,
        zorder=1,
    )
    ax.text(
        xlim[0] + 0.12 * (xlim[1] - xlim[0]),
        ylim[1] - 0.10 * (ylim[1] - ylim[0]),
        f"{size_label}\n{ref * 100:.0f}%",
        ha="center",
        va="center",
        fontsize=5.5,
    )


def plot_fig4cd(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Draw a two-panel reproduction of Runge et al. Fig. 4c,d."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5.8, 2.7), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.0), wspace=0.20)

    c_xlim = (0.0, max(0.06, float(frame["acs"].max()) * 1.28))
    y_lim = (0.0, max(0.070, float(frame["ace"].max()) * 1.22))
    d_xlim = (0.0, max(0.0024, float(frame["amce"].max()) * 1.30))

    _draw_panel(
        fig,
        outer[0, 0],
        frame,
        x_column="acs",
        y_column="ace",
        size_column="nout_fraction",
        xlabel="ACS",
        ylabel="ACE",
        panel_label="c",
        size_label=r"$N^{out}$",
        xlim=c_xlim,
        ylim=y_lim,
        label_offsets={
            0: (0.045, 0.020),
            1: (0.010, -0.055),
            2: (0.040, 0.030),
            18: (-0.045, -0.010),
            26: (0.030, -0.015),
            48: (0.012, 0.030),
        },
    )
    _draw_panel(
        fig,
        outer[0, 1],
        frame,
        x_column="amce",
        y_column="ace",
        size_column="mediated_fraction",
        xlabel="AMCE",
        ylabel="ACE",
        panel_label="d",
        size_label=r"$|C_k|/c_{max}$",
        xlim=d_xlim,
        ylim=y_lim,
        label_offsets={
            0: (0.045, 0.020),
            1: (-0.035, -0.040),
            2: (-0.055, 0.030),
            18: (0.030, -0.030),
            26: (0.030, -0.020),
            48: (0.030, -0.050),
        },
    )

    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output


def load_fig4cd_frame(result_dir: str | Path = DEFAULT_RESULT_DIR) -> pd.DataFrame:
    result = Path(result_dir)
    return build_fig4cd_frame(
        pd.read_csv(result / "gateway_scores.csv"),
        pd.read_csv(result / "mediator_scores.csv"),
        pd.read_csv(result / "total_effects.csv"),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frame = load_fig4cd_frame(args.result_dir)
    output = plot_fig4cd(frame, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
