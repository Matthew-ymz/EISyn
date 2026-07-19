#!/usr/bin/env python3
"""Plot the lead-8 UniCM all-mode Xi greedy atom distribution."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "results" / "unicm_phi_eid_greedy_decomposition_cpu_bound4_n8192"
DEFAULT_ASSET_BASE = ROOT / "fig" / "unicm_phi_eid_lead8_distribution"
DEFAULT_LEAD = 8
MODE_ORDER = ("nino", "nino12", "nino3", "nino4", "IOD", "IOB", "SIOD", "WWV", "NPMM", "SPMM", "TNA")


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


def load_lead_tables(result_dir: Path, lead: int, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    module_lead = pd.read_csv(result_dir / "unicm_phi_eid_greedy_module_lead_summary.csv")
    order_summary = pd.read_csv(result_dir / "unicm_phi_eid_greedy_order_summary.csv")
    total_summary = pd.read_csv(result_dir / "unicm_phi_eid_greedy_total_summary.csv")

    lead_modules = (
        module_lead[module_lead["lead"].astype(int) == int(lead)]
        .copy()
        .sort_values(["mean", "sources"], ascending=[False, True])
        .reset_index(drop=True)
    )
    lead_modules = lead_modules.head(int(top_k)).copy()
    lead_modules["atom"] = [f"A{i}" for i in range(1, len(lead_modules) + 1)]
    lead_modules["fraction"] = lead_modules["mean"] / float(total_summary.loc[total_summary["lead"].astype(int) == int(lead), "phi_atom_sum_mean"].iloc[0])
    lead_modules["pretty_sources"] = lead_modules["sources"].str.replace("|", " + ", regex=False)

    lead_order = order_summary[order_summary["lead"].astype(int) == int(lead)].copy().sort_values("order")
    total = total_summary[total_summary["lead"].astype(int) == int(lead)].iloc[0]
    return lead_modules, lead_order, total


def order_color_map(orders: Sequence[int]) -> dict[int, tuple[float, float, float, float]]:
    unique = sorted(set(int(order) for order in orders))
    cmap = plt.get_cmap("viridis")
    values = np.linspace(0.15, 0.86, max(1, len(unique)))
    return {order: cmap(value) for order, value in zip(unique, values)}


def draw_module_bars(ax: plt.Axes, lead_modules: pd.DataFrame, total_phi: float, colors: dict[int, object]) -> None:
    y = np.arange(len(lead_modules))
    bar_colors = [colors[int(order)] for order in lead_modules["order"]]
    label_anchor = lead_modules["mean"].astype(float) + lead_modules["std"].fillna(0.0).astype(float)
    x_max = max(0.06, float(label_anchor.max()) * 1.22)
    ax.barh(
        y,
        lead_modules["mean"],
        xerr=lead_modules["std"],
        color=bar_colors,
        edgecolor="#222222",
        linewidth=0.35,
        error_kw={"ecolor": "#4b5563", "elinewidth": 0.8, "capsize": 2.0},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(lead_modules["atom"])
    ax.invert_yaxis()
    ax.set_xlabel(r"Greedy $\xi_C$ atom (bits)")
    ax.set_ylabel("Lead-8 atom")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.7)
    ax.set_xlim(0.0, x_max)
    for idx, row in enumerate(lead_modules.itertuples(index=False)):
        label = f"{row.mean:.3f}  ({100.0 * row.mean / total_phi:.1f}%)"
        x_label = min(x_max * 0.84, float(row.mean) + float(row.std) + 0.0014)
        ax.text(
            x_label,
            idx,
            label,
            va="center",
            ha="left",
            fontsize=6.8,
            color="#111111",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.4},
        )
    ax.text(0.0, 1.04, r"a  Largest hierarchical $\xi_C$ atoms", transform=ax.transAxes, fontsize=8.5, fontweight="bold")


def draw_membership_matrix(ax: plt.Axes, lead_modules: pd.DataFrame, mode_order: Sequence[str]) -> None:
    y = np.arange(len(lead_modules))
    x = np.arange(len(mode_order))
    ax.set_xlim(-0.5, len(mode_order) - 0.5)
    ax.set_ylim(len(lead_modules) - 0.5, -0.5)
    for yi, row in enumerate(lead_modules.itertuples(index=False)):
        members = set(str(row.sources).split("|"))
        xs = [idx for idx, name in enumerate(mode_order) if name in members]
        if xs:
            ax.plot(xs, [yi] * len(xs), color="#9ca3af", linewidth=1.0, zorder=1)
        ax.scatter(x, np.full_like(x, yi), s=12, color="#e5e7eb", zorder=2)
        ax.scatter(xs, [yi] * len(xs), s=33, color="#111827", edgecolor="#ffffff", linewidth=0.35, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(mode_order, rotation=45, ha="right")
    ax.set_yticks(y)
    ax.set_yticklabels(lead_modules["atom"])
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("b  Source-mode membership", loc="left", fontsize=8.5, fontweight="bold", pad=8)


def draw_order_distribution(ax: plt.Axes, lead_order: pd.DataFrame, total_phi: float, colors: dict[int, object]) -> None:
    orders = lead_order["order"].astype(int).to_numpy()
    values = lead_order["mean"].astype(float).to_numpy()
    stds = lead_order["std"].astype(float).to_numpy()
    bar_colors = [colors[int(order)] for order in orders]
    ax.bar(orders, values, yerr=stds, color=bar_colors, edgecolor="#222222", linewidth=0.35, error_kw={"ecolor": "#4b5563", "elinewidth": 0.75, "capsize": 2.0})
    ax.set_xlabel("Atom order")
    ax.set_ylabel("Bits")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    ax.set_xticks(orders)
    ax.set_ylim(0.0, max(0.05, float((values + stds).max()) * 1.18))
    for order, value in zip(orders, values):
        ax.text(order, value + 0.0013, f"{100.0 * value / total_phi:.0f}%", ha="center", va="bottom", fontsize=6.5)
    ax.set_title("c  Distribution by order", loc="left", fontsize=8.5, fontweight="bold", pad=8)


def draw_top_module_callout(ax: plt.Axes, lead_modules: pd.DataFrame, total: pd.Series) -> None:
    top = lead_modules.iloc[0]
    lines = [
        rf"Lead 8 total $\Xi$: {float(total['phi_eid_mean']):.3f} +/- {float(total['phi_eid_std']):.3f} bits",
        f"Top atom: {str(top['pretty_sources'])}",
        f"Top atom mass: {float(top['mean']):.3f} bits ({100.0 * float(top['fraction']):.1f}% of total)",
        rf"Top {len(lead_modules)} atoms cover {100.0 * float(lead_modules['mean'].sum()) / float(total['phi_atom_sum_mean']):.1f}% of lead-8 $\Xi$.",
    ]
    ax.axis("off")
    y = 0.92
    ax.text(0.0, y, "d  Lead-8 summary", transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top")
    for idx, line in enumerate(lines):
        ax.text(0.0, y - 0.17 * (idx + 1), line, transform=ax.transAxes, fontsize=7.2, va="top", wrap=True)


def plot_lead_distribution(result_dir: Path, output_base: Path, *, lead: int, top_k: int) -> list[Path]:
    configure_matplotlib()
    lead_modules, lead_order, total = load_lead_tables(result_dir, lead, top_k)
    total_phi = float(total["phi_atom_sum_mean"])
    colors = order_color_map([*lead_modules["order"].astype(int).tolist(), *lead_order["order"].astype(int).tolist()])

    fig = plt.figure(figsize=(10.8, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.45, 0.75])
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_matrix = fig.add_subplot(gs[0, 1])
    ax_order = fig.add_subplot(gs[1, 0])
    ax_callout = fig.add_subplot(gs[1, 1])

    draw_module_bars(ax_bar, lead_modules, total_phi, colors)
    draw_membership_matrix(ax_matrix, lead_modules, MODE_ORDER)
    draw_order_distribution(ax_order, lead_order, total_phi, colors)
    draw_top_module_callout(ax_callout, lead_modules, total)

    handles = [
        mpl.patches.Patch(facecolor=colors[order], edgecolor="#222222", linewidth=0.35, label=f"order {order}")
        for order in sorted(colors)
        if order in set(lead_modules["order"].astype(int)) or order in set(lead_order["order"].astype(int))
    ]
    fig.legend(handles=handles, loc="outside right center", frameon=False, title="Atom order")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(ext) for ext in (".png", ".svg", ".pdf")]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], bbox_inches="tight")
    plt.close(fig)

    summary_path = result_dir / f"unicm_phi_eid_lead{int(lead)}_top_atoms.csv"
    lead_modules.to_csv(summary_path, index=False)
    return [*paths, summary_path]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--asset-base", type=Path, default=DEFAULT_ASSET_BASE)
    parser.add_argument("--lead", type=int, default=DEFAULT_LEAD)
    parser.add_argument("--top-k", type=int, default=12)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    for path in plot_lead_distribution(Path(args.result_dir), Path(args.asset_base), lead=int(args.lead), top_k=int(args.top_k)):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
