#!/usr/bin/env python3
"""Compose the cached DMF evidence into one manuscript plate; no experiments.

Curves and trees are redrawn from the original cached values. Cortex images are
unaltered crops of the existing 300-dpi surface plate: only blank margins and old
labels are removed. Every cortex uses the same pixel crop size and display scale.
The three state columns pair a seed-3 tree with a three-seed cortical mean; they
are not estimates with identical aggregation. The output PDF retains vector text,
curves, trees and color scales, with the existing cortex rendering as raster.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plot_dmf_subject_fixed_window import BASE, read_inputs
from scripts.dmf_curve_shape import METHOD_COLORS, METHOD_LABELS
from scripts.plot_dmf_schaefer100_tree_examples import NETWORK_COLORS, audit_tree
from scripts.analyze_dmf_schaefer100_xi_hierarchy_tree import (
    render_tree, _tree_metrics, _blend_with_white, SPLIT_COLOR,
)
from scripts.compare_runge_slp_pc60_xi_horizons import _node_from_record

OUTPUT = ROOT / "fig/dmf_schaefer100/dmf_brain_main_composite"
ROI = ROOT / "results/dmf_schaefer100/roi_shapley_three_G_G1000"
SURFACE = ROOT / "fig/dmf_schaefer100/roi_shapley_three_G_G1000/roi_share_three_G.png"
TREE_PATHS = [
    ROOT / "results/dmf_schaefer100/xi_hierarchy_tree/examples/wide/seed03_G0.00.json",
    ROOT / "results/dmf_schaefer100/xi_hierarchy_tree/examples/wide/seed03_G1.30.json",
    ROOT / "results/dmf_schaefer100/xi_hierarchy_tree/examples/wide_G1000/seed03_G10.00.json",
]
# Physical dimensions, in millimetres, keep final type sizes meaningful.
WIDTH, HEIGHT = 190., 211.


def axes_mm(fig, x, top, width, height):
    return fig.add_axes([x / WIDTH, 1 - (top + height) / HEIGHT,
                         width / WIDTH, height / HEIGHT])


def text_mm(fig, x, top, text, **kwargs):
    return fig.text(x / WIDTH, 1 - top / HEIGHT, text,
                    va="center", **kwargs)


def letter(fig, x, top, value):
    text_mm(fig, x, top, value, fontsize=9.4, weight="bold")


def cortex_crops():
    """Remove white padding in known source slots, retaining all brain pixels.

    Original surface_plate uses x=.145..99 in four slots and y=.18..93 in
    three slots. The common padded pixel extent preserves inter-panel scale.
    No recoloring, rotation, contrast adjustment, segmentation or stretching.
    """
    original = np.asarray(Image.open(SURFACE).convert("RGB"))
    h, w = original.shape[:2]
    slots = []
    for row in range(3):
        for col in range(4):
            x0, x1 = [round(w * (.145 + j * (.99 - .145) / 4))
                      for j in (col, col + 1)]
            y0, y1 = [round(h * (.07 + j * .25)) for j in (row, row + 1)]
            tile = original[y0:y1, x0:x1]
            # Exact white-background extent; even faint antialias pixels survive.
            # This detects the extent only; displayed RGB is untouched.
            ys, xs = np.nonzero(np.any(tile != 255, axis=2))
            if len(xs) == 0:
                raise ValueError(f"Empty cortex slot: {row}, {col}")
            slots.append((tile, (xs.min(), ys.min(), xs.max()+1, ys.max()+1)))
    crop_w = max(b[2]-b[0] for _, b in slots) + 20
    crop_h = max(b[3]-b[1] for _, b in slots) + 20
    images = []
    for tile, (x0, y0, x1, y1) in slots:
        content = tile[y0:y1, x0:x1]
        canvas = np.full((crop_h, crop_w, 3), 255, dtype=np.uint8)
        x = (crop_w-content.shape[1])//2
        y = (crop_h-content.shape[0])//2
        canvas[y:y+content.shape[0], x:x+content.shape[1]] = content
        images.append(canvas)
    return images


def load_inputs():
    fixed = json.loads((BASE / "fixed_window_view.json").read_text())
    contract, g, values, rho, scenario, audit = read_inputs(fixed)
    print("Frozen 93-person curves and hit counts verified:", audit)
    with np.load(ROI / "roi_shapley.npz") as a:
        labels, membership = a["region_labels"].copy(), a["network_membership"].copy()
        assert np.array_equal(a["G"], [0, 1.3, 10])
        assert np.array_equal(a["seeds"], [3, 4, 5])
    summary = json.loads((ROI / "summary.json").read_text())
    for name, record in summary["nonnegative_audits"].items():
        print("Existing ROI nonnegative audit:", name, record)
    with np.load(ROI / "roi_tables.npz") as a:
        shares = a["mean_share_percent"].copy()
        raw = a["roi_shapley_nats"].copy()
        tolerance = 1e-8  # native nats, as declared by the source experiment
        count = int(np.count_nonzero(raw < -tolerance))
        if count:
            raise ArithmeticError(f"ROI nonnegativity violation: min={raw.min()}, "
                                  f"threshold={-tolerance} nats, affected_count={count}")
        if not np.allclose(shares.sum(1), 100, rtol=0, atol=1e-8):
            raise ValueError("ROI percentage budget does not close")
        if not np.allclose([shares.min(), shares.max()],
                           [summary["plot"]["vmin"], summary["plot"]["vmax"]]):
            raise ValueError("Surface figure scale differs from current cache")
    trees = []
    for p in TREE_PATHS:
        record = json.loads(p.read_text())
        tree = _node_from_record(record["tree"])
        print("Cached tree audit:", p.name, audit_tree(tree, tree.xi_bits))
        trees.append(tree)
    return g, values, rho, scenario, labels, membership, summary, trees


def draw(dpi):
    g, values, rho, scenario, labels, membership, summary, trees = load_inputs()
    images = cortex_crops()
    palette = LinearSegmentedColormap.from_list(
        "native_sc", plt.cm.viridis(np.linspace(.04, .90, 256)))
    norm = Normalize(float(rho.min()), float(rho.max()))
    colors = palette(norm(rho))
    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
             "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
             "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
             "axes.spines.top": False, "axes.spines.right": False,
             "axes.linewidth": .65, "xtick.major.width": .6, "ytick.major.width": .6,
             "xtick.major.size": 2.8, "ytick.major.size": 2.8,
             "pdf.fonttype": 42, "savefig.facecolor": "white"}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), facecolor="white")
        handles = [
            Line2D([], [], color=palette(.5), lw=.7, label="Individuals (n = 93; 3-seed means)"),
            Line2D([], [], color="#202020", ls=(0, (5, 2.4)), lw=1.35,
                   marker="o", ms=1.7, label="Mean SC (separate simulation)"),
        ]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.035, .995),
                   ncol=2, frameon=False, fontsize=6.8, handlelength=2.4,
                   columnspacing=1.2, borderaxespad=0.)
        specifications = [
            ("a", "rate", 13, 14, "Independent order parameter"),
            ("b", "xi", 80, 14, r"Integrated EI $\Xi$"),
            ("c", "phi_r", 13, 50, r"BOLD-like pairwise $\Phi^R$"),
            ("d", "wms", 80, 50, "Signed source WMS"),
        ]
        for panel, name, x, top, title in specifications:
            ax = axes_mm(fig, x, top, 52, 26)
            for i in range(93):
                ax.plot(g, values[name][i], color=colors[i], alpha=.50, lw=.42)
            ax.plot(g, values[name][-1], color="#202020", ls=(0, (5, 2.4)),
                    lw=1.35, marker="o", ms=1.6, markevery=5, zorder=4)
            ax.set(xlim=(0, 4), xticks=np.arange(5), title=title,
                   ylabel="Mean E rate (Hz)" if name == "rate" else "Information (nats)")
            if name in ("rate", "xi"):
                ax.set_ylim(bottom=0)
            ax.margins(y=.06)
            if top == 14:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("$G$", labelpad=1.5)
            ax.set_title(title, pad=3.5)
            letter(fig, x-10.3, top-3.3, panel)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=palette),
                          cax=axes_mm(fig, 25, 84.5, 96, 1.6), orientation="horizontal")
        cb.set_label("Native SC spectral radius (same individual colors)", fontsize=6.6, labelpad=2)
        cb.ax.tick_params(labelsize=6.3, length=2, pad=1.5)
        cb.outline.set_linewidth(.5)

        ax = axes_mm(fig, 148, 14, 37.5, 62)
        for position, name in zip([2, 1, 0], ("xi", "phi_r", "wms")):
            row = scenario["cohorts"]["all93"]["methods"][name]
            rate = 100*row["hit_rate"]
            ax.barh(position, rate, height=.43, color=METHOD_COLORS[name], linewidth=0)
            ax.text(rate+2., position, f"{rate:.1f}%\n{row['hit_count']}/93",
                    ha="left", va="center", fontsize=6.4, linespacing=1.2)
        ax.axvline(50, color=".65", ls=":", lw=.65, zorder=0)
        ax.set(xlim=(0, 100), ylim=(-.55, 2.55), xticks=[0, 50, 100],
               yticks=[2, 1, 0], yticklabels=[METHOD_LABELS[n] for n in ("xi", "phi_r", "wms")],
               title="Transition-window hits\nAll 93 participants",
               xlabel="Hits within own window (%)")
        ax.set_title(ax.get_title(), fontsize=7.5, pad=3.5)
        ax.set_xlabel(ax.get_xlabel(), fontsize=6.6, labelpad=2)
        letter(fig, 138, 10.7, "e")
        text_mm(fig, 166.5, 86.5, "Window width: 0.5 $G$\nPost hoc; $\Delta G=0.1$",
                fontsize=6.4, color=".35", ha="center", linespacing=1.4)

        # Shared state columns couple hierarchy and anatomy without extra empty cells.
        starts = [4, 67, 130]
        descriptions = ["No long-range coupling", "Peak", "Extreme coupling"]
        couplings = [0, 1.3, 10]
        for i, (x, coupling, description, tree) in enumerate(zip(starts, couplings, descriptions, trees)):
            text_mm(fig, x+28.5, 99, rf"$G={coupling:g}$  ·  {description}",
                    ha="center", fontsize=7.7)
            ax = axes_mm(fig, x, 105, 57, 31)
            render_tree(tree, None, labels=labels, network_membership=membership,
                        network_names=[], seed=3, coupling_g=coupling, dpi=dpi,
                        axis=ax, network_colors=NETWORK_COLORS, information_unit="nats",
                        node_value_mode="root_share", syn_color_max=3, show_colorbar=False,
                        show_roi_labels=False, show_network_strip_label=False, node_label_limit=0)
            # Existing renderer styles are tuned for a standalone 15-inch tree.
            # Scale marks/strokes for this 57-mm panel; every node is retained.
            for line in ax.lines:
                line.set_linewidth(.32)
            for collection in ax.collections:
                collection.set_sizes(collection.get_sizes()*.20)
                collection.set_linewidths(collection.get_linewidths()*.55)
            for artist in list(ax.texts):
                artist.remove()
            # Macroscopic structure only: seed information stays in the caption.
            # Reuse the removed title band to give every tree more height.
            ax.set_ylim(-.07, 1.06)
            letter(fig, x, 105, "fgh"[i])
            metrics = _tree_metrics(tree)
            text_mm(fig, x+28.5, 138.5,
                    f"Depth {metrics['maximum_depth']} · Spine {metrics['dominant_spine_fraction']:.1%}"
                    f" · Colless {metrics['normalized_colless_imbalance']:.3f}",
                    ha="center", fontsize=5.6, color=".30")

        network_labels = ["VIS", "SM", "DA", "SA/VA", "LIM", "FPN", "DMN"]
        fig.legend(handles=[Patch(facecolor=c, edgecolor="none", label=n)
                            for c, n in zip(NETWORK_COLORS, network_labels)],
                   loc="center left", bbox_to_anchor=(4/WIDTH, 1-144.5/HEIGHT),
                   ncol=7, frameon=False, fontsize=6.4, handlelength=.8,
                   handletextpad=.35, columnspacing=.9, borderaxespad=0.)
        tree_cmap = LinearSegmentedColormap.from_list("local_syn",
                    [_blend_with_white(SPLIT_COLOR, .12), _blend_with_white(SPLIT_COLOR, .74)])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 3), cmap=tree_cmap),
                          cax=axes_mm(fig, 155, 143.5, 30, 1.4), orientation="horizontal",
                          ticks=[0, 1, 2, 3])
        cb.ax.tick_params(labelsize=5.4, length=1.5, pad=1.)
        cb.outline.set_linewidth(.4)
        text_mm(fig, 129, 144.4, r"Local Syn / $\Xi$ (%)", fontsize=5.9, ha="center")

        text_mm(fig, 4, 152, "ROI attribution · mean across seeds 3, 4, 5",
                fontsize=6.6, weight="bold")
        text_mm(fig, 185, 152, "Left / right hemispheres; lateral above, medial below",
                fontsize=5.9, ha="right", color=".35")
        for i, x in enumerate(starts):
            row = summary["state_totals"][i]
            letter(fig, x, 157, "ijk"[i])
            text_mm(fig, x+30, 157,
                    rf"$\Xi={row['xi_mean_nats']:.2f}\pm{row['xi_seed_sd_nats']:.2f}$ nats",
                    fontsize=6.8, ha="center")
            for col in range(4):
                ax = axes_mm(fig, x + (col % 2)*29, 161 + (col//2)*18.2, 28, 17.2)
                ax.imshow(images[4*i+col], interpolation="none")
                ax.axis("off")
        cb = fig.colorbar(plt.cm.ScalarMappable(
            norm=Normalize(summary["plot"]["vmin"], summary["plot"]["vmax"]), cmap="viridis"),
            cax=axes_mm(fig, 48, 199.0, 94, 1.6), orientation="horizontal")
        cb.set_ticks(np.linspace(summary["plot"]["vmin"], summary["plot"]["vmax"], 5))
        cb.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        cb.ax.tick_params(labelsize=6.0, length=2, pad=1.2)
        cb.outline.set_linewidth(.5)
        cb.set_label(r"ROI attribution to overall $\Xi$ (%)", fontsize=6.7, labelpad=2)
        text_mm(fig, 95, 209.0, "Shared cortical scale · each state sums to 100% · ± seed SD",
                ha="center", fontsize=5.9, color=".35")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".pdf"):
            path = OUTPUT.with_suffix(suffix)
            fig.savefig(path, dpi=dpi, bbox_inches=None)
            print("Saved:", path)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=450)
    draw(parser.parse_args().dpi)
