#!/usr/bin/env python3
"""Compose the cached DMF evidence into one manuscript plate; no experiments.

Curves and trees are redrawn from the original cached values. Cortex images are
unaltered crops of the existing 300-dpi surface plate: only blank margins and old
labels are removed. Every cortex uses the same pixel crop size and display scale.
The three state columns pair a seed-4 tree with a three-seed cortical mean; they
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
    render_tree, _blend_with_white, SPLIT_COLOR,
)
from scripts.plot_dmf_spt_collapsed_tree import render_collapsed_tree, layout_audit
from scripts.plot_dmf_spt_peel_groups import load_comparison, PAIRED_SUMMARY, YEO_METADATA
from scripts.plot_dmf_schaefer100_summary import network_xi_values, sd
from scripts.run_dmf_subject_consistency import digest

OUTPUT = ROOT / "fig/dmf_schaefer100/dmf_brain_main_composite"
ROI = ROOT / "results/dmf_schaefer100/roi_shapley_three_G_G1000"
PEAK_ROI = ROOT / "results/dmf_schaefer100/roi_shapley/full_xi_results.npz"
SURFACE = ROOT / "fig/dmf_schaefer100/roi_shapley_three_G_G1000/roi_share_three_G.png"
TREE_SEED = 4
TREE_COUPLINGS = [0., 1.3, 10.]
# Physical dimensions, in millimetres, keep final type sizes meaningful.
WIDTH, HEIGHT = 190., 269.
PAIRWISE = ROOT / "results/dmf_schaefer100/natural_pairwise_phi_r"


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


def read_main_comparison():
    """Use the selected natural pairwise mean, preserving Xi/native WMS.

    All 94 SCs include the independently simulated mean-SC reference. Its curve
    is displayed separately and never enters the 93-participant hit count.
    """
    fixed = json.loads((BASE / "fixed_window_view.json").read_text())
    contract, g, values, rho, scenario, audit = read_inputs(fixed)
    c = json.loads((PAIRWISE / "contract.json").read_text())
    s = json.loads((PAIRWISE / "summary.json").read_text())
    sha = digest(PAIRWISE / "contract.json")
    if (s['contract_sha256'] != sha or c['pair_count'] != 19900
            or c['subject_ids'] != contract['subject_ids']
            or c['dense_curves_sha256'] != digest(BASE / "analysis_curves.npz")
            or c['dense_windows_sha256'] != digest(BASE / "window_summary.json")
            or c['query_script_sha256'] != digest(ROOT / "scripts/compare_dmf_state_aligned_phi_r.py")
            or c['runner_sha256'] != digest(ROOT / "scripts/compare_dmf_natural_pairwise_phi_r.py")
            or s['curves_sha256'] != digest(PAIRWISE / "curves.npz")):
        raise ValueError('Selected pairwise PhiR provenance differs')
    with np.load(PAIRWISE / "curves.npz") as a:
        raw = a['phi_r_nats'].copy()
        if (str(a['contract_sha256']) != sha or raw.shape != (94, 3, 41)
                or a['subject_ids'].tolist() != c['subject_ids']
                or not np.isfinite(raw).all() or not np.array_equal(a['G'], g)
                or a['seeds'].tolist() != [3, 4, 5]):
            raise ValueError('Invalid selected pairwise PhiR curves')
    if s['audit']['significant_violation_count'] or raw.min() < -1e-8:
        raise ArithmeticError('Selected pairwise PhiR nonnegativity violation')
    values['phi_r'] = raw.mean(1)
    scenario = json.loads(json.dumps(scenario))
    for row, aligned in zip(scenario['subject_rows'], s['subject_rows']):
        if row['subject'] != aligned['subject'] or row['window'] != aligned['window']:
            raise ValueError('Selected pairwise PhiR subject/window changed')
        row['methods']['phi_r'] = aligned['natural_pairwise']
    for name, cohort in scenario['cohorts'].items():
        cohort['methods']['phi_r'] = s['cohorts'][name]['methods']['natural_pairwise']
        # Original comparisons refer to the historical BOLD-like PhiR. This
        # figure-only selection does not recompute inferential statistics.
        cohort.pop('comparisons', None)
        cohort.pop('both_comparisons_significant', None)
    if scenario['cohorts']['all93']['methods']['phi_r']['hit_count'] != 29:
        raise ValueError('Selected pairwise PhiR hit count differs from 29/93')
    scenario['original_Xi_and_WMS_unchanged'] = True
    scenario['phi_r_protocol'] = 'natural-pairwise-mean'
    audit.update(phi_r_protocol=scenario['phi_r_protocol'],phi_r_pair_count=19900,
                 phi_r_individual_count=93,phi_r_mean_SC_reference_available=True,
                 pairwise_nonnegative_audit=s['audit'],pairwise_curves_sha256=digest(PAIRWISE/'curves.npz'))
    return contract, g, values, rho, scenario, audit


def read_network_comparison():
    """Reproduce the supplied Yeo panels from their existing 24-condition cache.

    Shares use the same full-system denominator in each seed/G condition.
    Average the three G values within seed, then show eight-seed mean +/- SD.
    The between-network component is the cached exact seven-group Shapley.
    """
    with np.load(YEO_METADATA, allow_pickle=True) as a:
        if (a['seeds'].tolist() != list(range(3, 11))
                or not np.array_equal(a['G'], [1.2, 1.3, 1.4])):
            raise ValueError('Network panels require the original eight-seed peak window')
        sizes = a['network_sizes'].copy()
        membership = a['network_membership'].copy()
        names = a['network_names'].tolist()
        if not np.array_equal(sizes, np.bincount(membership, minlength=7)):
            raise ValueError('Network sizes differ from ROI membership')
        within = network_xi_values(a['conditional_covariance'], membership)
        between = a['between_group_shapley'].copy()
        total = a['fine_phi'].copy()
    if within.shape != (8, 3, 7) or between.shape != within.shape or total.shape != (8, 3):
        raise ValueError('Invalid network contribution shapes')
    components = np.concatenate([within.ravel(), between.ravel()])
    tolerance = 1e-8  # native bits, same rule as the original network panels
    count = int(np.count_nonzero(components < -tolerance))
    if not np.isfinite(components).all() or not np.isfinite(total).all() or np.any(total <= 0):
        raise ValueError('Network shares require finite components and positive root Xi')
    if count:
        raise ArithmeticError(f'Network nonnegativity violation: minimum={components.min()} bits, '
                              f'threshold={-tolerance} bits, affected_count={count}')
    closure = float(np.abs(within.sum(-1) + between.sum(-1) - total).max())
    if closure > tolerance:
        raise ArithmeticError(f'Network contribution closure failed: {closure} bits')
    by_seed = [100 * values / total[..., None] for values in [within, between]]
    by_seed = [values.mean(1) for values in by_seed]
    means, errors = [values.mean(0) for values in by_seed], [sd(values, axis=0) for values in by_seed]
    if abs(sum(values.sum() for values in means) - 100) > 1e-8:
        raise ArithmeticError('Network percentage budget does not close')
    audit = dict(source_sha256=digest(YEO_METADATA), tolerance_bits=tolerance,
                 minimum_bits=float(components.min()), significant_violation_count=count,
                 tolerance_negative_count=int(np.count_nonzero(components < 0)),
                 closure_max_bits=closure, totals_percent=[float(v.sum()) for v in means],
                 mean_percent=[v.tolist() for v in means], seed_sd_percent=[v.tolist() for v in errors],
                 normalization='per-condition full Xi; then G mean within seed; then mean +/- seed SD')
    print('Existing network panel audit:', audit)
    return names, sizes, means, errors


def read_roi_network_attribution():
    """Sum cached ordinary ROI Shapley shares by post-hoc Yeo membership.

    This is a separate full-Xi ROI game, not the sum of the two network games.
    No permutation sampling or Shapley estimation is performed here.
    """
    with np.load(YEO_METADATA, allow_pickle=True) as reference, np.load(PEAK_ROI, allow_pickle=True) as a:
        meta = json.loads(str(a['summary_json']))
        if (meta['version'] != 'paired-permutation-full-xi-v2' or not meta['converged']
                or meta['source_sha256'] != digest(YEO_METADATA)):
            raise ValueError('ROI attribution provenance/convergence differs')
        for key in ['G', 'seeds', 'region_labels', 'network_membership', 'network_names']:
            if not np.array_equal(a[key], reference[key]):
                raise ValueError(f'ROI attribution differs from network data: {key}')
        raw = a['roi_shapley_bits'].copy()
        total = reference['fine_phi'].copy()
        membership = a['network_membership'].copy()
        if not np.array_equal(a['full_xi'], total):
            raise ValueError('ROI attribution root Xi differs from the network cache')
    if raw.shape != (8, 3, 100) or not np.isfinite(raw).all():
        raise ValueError('Invalid cached ROI attribution')
    tolerance = 1e-8  # native bits; raw values retained without clipping
    count = int(np.count_nonzero(raw < -tolerance))
    if count or meta['significant_nonnegativity_violation_count']:
        raise ArithmeticError(f'ROI nonnegativity violation: min={raw.min()} bits, '
                              f'threshold={-tolerance} bits, affected_count={count}')
    closure = float(np.abs(raw.sum(-1) - total).max())
    if closure > tolerance:
        raise ArithmeticError(f'ROI attribution budget failed: {closure} bits')
    grouped = np.stack([raw[..., membership == k].sum(-1) for k in range(7)], axis=-1)
    by_seed = (100 * grouped / total[..., None]).mean(1)
    mean, error = by_seed.mean(0), sd(by_seed, axis=0)
    if abs(mean.sum() - 100) > 1e-8:
        raise ArithmeticError('Grouped ROI attribution percentage budget failed')
    print('Existing ROI-to-network attribution audit:', dict(
        source_sha256=digest(PEAK_ROI), tolerance_bits=tolerance, minimum_bits=float(raw.min()),
        tolerance_negative_count=int(np.count_nonzero(raw < 0)), significant_violation_count=count,
        original_marginal_audit_min_bits=meta['minimum_marginal_bits'],
        original_marginal_tolerance_negative_count=meta['tolerance_negative_marginal_count'],
        closure_max_bits=closure, mean_percent=mean.tolist(), seed_sd_percent=error.tolist(),
        total_percent=float(mean.sum()), aggregation='ordinary ROI Shapley summed by Yeo group'))
    return mean, error


def load_inputs():
    contract, g, values, rho, scenario, audit = read_main_comparison()
    print("Selected 93-person curves and hit counts verified:", audit)
    with np.load(ROI / "roi_shapley.npz") as a:
        labels, membership = a["region_labels"].copy(), a["network_membership"].copy()
        assert np.array_equal(a["G"], [0, 1.3, 10])
        assert np.array_equal(a["seeds"], [3, 4, 5])
    with np.load(YEO_METADATA, allow_pickle=True) as a:
        network_names = a["network_names"].tolist()
        if (not np.array_equal(a["region_labels"], labels)
                or not np.array_equal(a["network_membership"], membership)):
            raise ValueError("Tree and cortical ROI metadata differ")
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
    conditions = load_comparison(PAIRED_SUMMARY, TREE_SEED, TREE_COUPLINGS)
    for tree, source in conditions:
        print("Cached tree audit:", source, audit_tree(tree, tree.xi_bits))
    return g, values, rho, scenario, labels, membership, network_names, summary, conditions


def draw(dpi):
    g, values, rho, scenario, labels, membership, network_names, summary, conditions = load_inputs()
    images = cortex_crops()
    network_names, network_sizes, network_means, network_errors = read_network_comparison()
    roi_network_mean, roi_network_error = read_roi_network_attribution()
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
            ("b", "xi", 80, 14, r"$\Xi$"),
            ("c", "phi_r", 13, 50, r"Pairwise $\Phi^R$"),
            ("d", "wms", 80, 50, "Signed source WMS"),
        ]
        for panel, name, x, top, title in specifications:
            ax = axes_mm(fig, x, top, 52, 26)
            for i in range(93):
                ax.plot(g, values[name][i], color=colors[i], alpha=.50, lw=.42)
            if len(values[name]) == 94:
                ax.plot(g, values[name][-1], color="#202020", ls=(0, (5, 2.4)),
                        lw=1.35, marker="o", ms=1.6, markevery=5, zorder=4)
            ax.set(xlim=(0, 4), xticks=np.arange(5), title=title,
                   ylabel="Mean E rate (Hz)" if name == "rate" else "Information (nats)")
            if name in ('rate', 'xi', 'phi_r'):
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
                    ha="left", va="center", fontsize=6.4, linespacing=1.2,
                    bbox=dict(facecolor="white", edgecolor="none", pad=.6))
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
        for i, (x, coupling, description, condition) in enumerate(zip(starts, couplings, descriptions, conditions)):
            tree, source = condition
            text_mm(fig, x+28.5, 99, rf"$G={coupling:g}$  ·  {description}",
                    ha="center", fontsize=7.7)
            ax = axes_mm(fig, x, 105, 57, 31)
            condense = coupling == 1.3
            if condense:
                tree_audit = render_collapsed_tree(
                    tree, axis=ax, labels=labels, network_membership=membership,
                    network_names=network_names, seed=TREE_SEED, coupling_g=coupling,
                    show_heading=False, show_colorbar=False, show_roi_labels=False,
                    node_label_limit=0)
            else:
                render_tree(tree, None, labels=labels, network_membership=membership,
                            network_names=network_names, seed=TREE_SEED, coupling_g=coupling, dpi=dpi,
                            axis=ax, network_colors=NETWORK_COLORS, information_unit="nats",
                            node_value_mode="root_share", syn_color_max=3, show_colorbar=False,
                            show_roi_labels=False, show_network_strip_label=False, node_label_limit=0)
                tree_audit = layout_audit(tree, collapse=False)
            # Existing renderer styles are tuned for a standalone 15-inch tree.
            # Scale marks/strokes for this 57-mm panel; every node is retained.
            for line in ax.lines:
                line.set_linewidth(.32)
            for collection in ax.collections:
                if collection.get_zorder() == 3:
                    # Local Syn percentage is encoded by color only; uniform
                    # node size/stroke avoids singling out particular nodes.
                    collection.set_sizes(np.full_like(collection.get_sizes(), 3.))
                    collection.set_linewidths(.40)
                else:
                    collection.set_sizes(collection.get_sizes()*.20)
                    collection.set_linewidths(collection.get_linewidths()*.55)
            # Remove the standalone title/statistics band. All tree nodes are
            # rendered as markers, without numerical labels or badges.
            for artist in list(ax.texts):
                artist.remove()
            ax.set_ylim(-.07, 1.06)
            tree_audit['display_label_count'] = 0
            tree_audit['percentage_encoding'] = 'color only; uniform node size and stroke'
            letter(fig, x, 105, "fgh"[i])
            print("Displayed tree audit:", source, {k:v for k,v in tree_audit.items()
                  if k not in {"nodes", "roi_order", "original_tree_metrics"}})
            original_depth = tree_audit['original_tree_metrics']['maximum_depth']
            depth = str(tree_audit['display_maximum_depth'])
            if condense:
                depth += f" (from {original_depth})"
            text_mm(fig, x+28.5, 138.5,
                    f"Multi-ROI forks {tree_audit['multi_roi_fork_count']} · Display depth {depth}",
                    ha="center", fontsize=5.6, color=".30")
        text_mm(fig, 95, 148.1,
                "Seed 4 · dashed forks / diamonds: condensed singleton exits; color = peak local Syn",
                fontsize=5.6, ha="center", color=".35")

        network_labels = ["VIS", "SM", "DA", "SA/VA", "LIM", "FPN", "DMN"]
        fig.legend(handles=[Patch(facecolor=c, edgecolor="none", label=n)
                            for c, n in zip(NETWORK_COLORS, network_labels)],
                   loc="center left", bbox_to_anchor=(4/WIDTH, 1-(144.5)/HEIGHT),
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
        # Bottom triptych: two fixed-network games and the independent ROI
        # game aggregated by network. Same colors/order and percent axis.
        order = np.argsort(network_means[0])[::-1]
        ypos = np.arange(7)[::-1]
        short = ['VIS', 'SM', 'DA', 'SA/VA', 'LIM', 'FPN', 'DMN']
        rows = [f'{short[i]} / {network_sizes[i]}' for i in order]
        bar_colors = [NETWORK_COLORS[i] for i in order]
        titles = [r'Within-network $\Xi$', 'Between-network Shapley', 'ROI Shapley by network']
        means = [*network_means, roi_network_mean]
        errors = [*network_errors, roi_network_error]
        for i, (x, mean, error, title) in enumerate(zip([19, 82, 145], means, errors, titles)):
            ax = axes_mm(fig, x, 225, 40, 33)
            ax.barh(ypos, mean[order], xerr=error[order], height=.72, color=bar_colors,
                    linewidth=0, capsize=1.3,
                    error_kw=dict(ecolor='#202020', elinewidth=.65, capthick=.65), zorder=2)
            ax.set(xlim=(0, 25), ylim=(-.65, 6.65), xticks=[0, 10, 20],
                   yticks=ypos, yticklabels=rows, xlabel=r'Overall $\Xi$ share (%)')
            ax.tick_params(axis='y', labelsize=6.1, pad=1.2, length=0)
            ax.tick_params(axis='x', labelsize=6.1)
            ax.set_xlabel(ax.get_xlabel(), fontsize=6.3, labelpad=2)
            ax.grid(axis='x', color='.91', lw=.5, zorder=0)
            letter(fig, starts[i], 217.5, 'lmn'[i])
            text_mm(fig, starts[i]+28.5, 219.2, title, fontsize=6.8, ha='center')
            text_mm(fig, x+40, 267, f'Total: {mean.sum():.2f}%',
                    ha='right', fontsize=5.9, color='.35')

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
