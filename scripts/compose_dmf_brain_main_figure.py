#!/usr/bin/env python3
"""Compose the cached DMF evidence into one manuscript plate; no experiments.

Curves use the versioned continuous-G readout; historical inputs remain
protected. G=0 and G=10 show one-level root/ROI displays; G=1.3 retains modules.
Cortex images are unaltered crops of the corrected 300-dpi surface plate, with
one shared pixel size and color scale. Seed-4 organization panels and three-seed
cortical means are not identically aggregated. The output PDF retains vector text,
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
from scripts.plot_dmf_schaefer100_summary import sd
from scripts.dmf_joint_readout import CommonTargetGame, audit_nonnegative
from scripts.dmf_subject_consistency import network_values
from scripts.run_dmf_subject_consistency import digest
from scripts.dmf_main_independent_components import (
    BASE as CORRECTED_BASE, ROI as CORRECTED_ROI, FIGURE as CORRECTED_FIGURE,
    load_manifest,
)
from scripts.dmf_dense_statistics import subject_landmarks
from scripts.analyze_dmf_subject_dense_windows import expanded_subject
from scripts.run_dmf_continuous_subject_curves import (
    BASE as CONTINUOUS_CURVES, load_curve_manifest,
)

OUTPUT = ROOT / "fig/dmf_schaefer100/dmf_brain_main_composite"
ROI = CORRECTED_ROI
SURFACE = CORRECTED_FIGURE / "roi_share_three_G.png"
TREE_SEED = 4
TREE_COUPLINGS = [0., 1.3, 10.]
# Physical dimensions, in millimetres, keep final type sizes meaningful.
WIDTH, HEIGHT = 190., 255.
PAIRWISE = ROOT / "results/dmf_schaefer100/natural_pairwise_phi_r"


def axes_mm(fig, x, top, width, height):
    return fig.add_axes([x / WIDTH, 1 - (top + height) / HEIGHT,
                         width / WIDTH, height / HEIGHT])


def text_mm(fig, x, top, text, **kwargs):
    return fig.text(x / WIDTH, 1 - top / HEIGHT, text,
                    va="center", **kwargs)


def letter(fig, x, top, value):
    text_mm(fig, x, top, value, fontsize=9.4, weight="bold")


def render_roi_star(axis, membership, *, order=None, fontsize=6.2, condensed=False):
    """One displayed root and every ROI leaf; geometry carries no Syn value."""
    membership = np.asarray(membership, int)
    order = np.arange(100) if order is None else np.asarray(order, int)
    if membership.shape != (100,) or not np.array_equal(np.sort(order), np.arange(100)):
        raise ValueError('One-level display must retain every ROI exactly once')
    x = np.arange(100, dtype=float)
    colors = [NETWORK_COLORS[k] for k in membership[order]]
    axis.set(xlim=(-1.5, 100.5), ylim=(-.07, 1.15)); axis.axis('off')
    style = (0, (2.5, 2)) if condensed else '-'
    axis.plot([49.5, 49.5], [.94, .65], color='#B3BAC0', lw=.45, zorder=1)
    axis.plot([0, 99], [.65, .65], color='#B3BAC0', lw=.45, ls=style, zorder=1)
    for roi in x:
        axis.plot([roi, roi], [.65, .015], color='#B3BAC0', lw=.32, zorder=1)
    axis.scatter([49.5], [.94], s=7, marker='D' if condensed else 'o',
                 facecolors='white', edgecolors='#89959B', linewidths=.5, zorder=3)
    axis.scatter(x, np.full(100, .015), s=2.3, color=colors, edgecolors='none', zorder=3)
    axis.text(49.5, 1.07, 'Root → 100 ROI', ha='center', va='center',
              fontsize=fontsize, color='.30')
    return dict(display_root_count=1, display_leaf_count=100, display_maximum_depth=1,
        scalar_leaf_count=0, roi_order=order.tolist(),
        colors='Yeo membership only; no Syn magnitude encoding',
        geometry='schematic; no optimization or quantitative edge length')


def render_g0_trivial_tree(axis, record, membership, *, fontsize=6.2):
    """Collapse zero-Syn joins; keep local E/I budgets inside ROI leaves."""
    local = np.asarray(record['local_roi_xi_nats'], float)
    membership = np.asarray(membership, int)
    audit = audit_nonnegative(local, 1e-8)
    if (local.shape != (100,) or membership.shape != (100,)
            or abs(local.sum()-record['full_xi_nats']) > 1e-8
            or abs(record['cross_roi_nats']) > 1e-8):
        raise ArithmeticError('Trivial G=0 tree does not close to the local E/I budget')
    display = render_roi_star(axis, membership, fontsize=fontsize)
    return dict(**display, roi_block_count=100, collapsed_cross_roi_forks='all zero-Syn joins',
        full_xi_nats=record['full_xi_nats'], cross_roi_nats_raw=record['cross_roi_nats'],
        local_ei_audit=audit, hidden_local_coordinates='E/I retained within every ROI block')


def render_g10_one_level_tree(axis, tree, membership, *, fontsize=6.2):
    """Condense the singleton chain, including its terminal pair, for display."""
    original = layout_audit(tree, collapse=False)
    if original['multi_roi_fork_count'] != 0 or original['selected_internal_node_count'] != 99:
        raise ValueError('One-level G=10 display requires a complete singleton chain')
    # A one-level display has no subtree adjacency to preserve. Use the same
    # native ROI order as G=0, rather than the arbitrary binary traversal.
    display = render_roi_star(axis, membership, order=sorted(tree.indices),
                              fontsize=fontsize, condensed=True)
    return dict(**display, represented_binary_forks=99,
        original_tree_metrics=original['original_tree_metrics'],
        original_syn_audit=audit_tree(tree, tree.xi_bits),
        interpretation='display contraction only; native ROI order shared with G=0; positive binary Syn retained in source tree')


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
    """Combine continuous-G Xi with the selected natural pairwise mean/native WMS.

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
    correction = load_manifest()  # retained lower-panel component readout
    continuous = load_curve_manifest()
    with np.load(CONTINUOUS_CURVES / 'analysis_curves.npz') as a:
        if (a['subject_ids'].tolist() != contract['subject_ids'] or not np.array_equal(a['G'], g)
                or str(a['curve_identity_sha256']) != continuous['identity_sha256']):
            raise ValueError('Continuous dense curve identity differs')
        for name in ['xi', 'whole_ei', 'partial_ei_sum']:
            values[name] = a[name].mean(1)
    corrected_rows = []
    for i, old in enumerate(scenario['subject_rows']):
        landmarks = subject_landmarks(g, values['rate'][i],
            {n:values[n][i] for n in ['xi', 'phi_r', 'wms']})
        row = expanded_subject(dict(subject=old['subject'], **landmarks), 2)
        if row['window'] != old['window']:
            raise ValueError('The independent transition window changed')
        for name in ['phi_r', 'wms']:
            if row['methods'][name]['hit'] != old['methods'][name]['hit']:
                raise ValueError('An unchanged natural comparator changed hit status')
        corrected_rows.append(row)
    scenario['subject_rows'] = corrected_rows
    memberships = dict(all93=set(contract['subject_ids'][:-1]),
        primary85=set(contract['holdout_subject_ids']),
        development8=set(contract['development_subject_ids']))
    for name, cohort in scenario['cohorts'].items():
        rows = [r for r in corrected_rows if r['subject'] in memberships[name]]
        cohort['methods'] = {n:dict(hit_count=sum(r['methods'][n]['hit'] for r in rows),
            denominator=len(rows), hit_rate=sum(r['methods'][n]['hit'] for r in rows)/len(rows),
            flat_count=sum(r['methods'][n]['flat'] for r in rows),
            boundary_extremum_count=sum(not r['methods'][n]['interior'] and not r['methods'][n]['flat'] for r in rows))
            for n in ['xi', 'phi_r', 'wms']}
    scenario['density_correction'] = continuous['identity_sha256']
    scenario['curve_estimator'] = 'continuous-G affine TM; penalties 0.1 / 10 fixed from pilot'
    scenario['lower_panel_estimator'] = 'previous independent-components / legacy tree readouts'
    scenario['original_WMS_unchanged'] = True
    scenario['phi_r_protocol'] = 'natural-pairwise-mean'
    audit.update(phi_r_protocol=scenario['phi_r_protocol'],phi_r_pair_count=19900,
                 phi_r_individual_count=93,phi_r_mean_SC_reference_available=True,
                 pairwise_nonnegative_audit=s['audit'],pairwise_curves_sha256=digest(PAIRWISE/'curves.npz'))
    audit.pop('all_93_landmarks_match_frozen', None)
    audit.update(density_correction=continuous['identity_sha256'],
        protected_source_landmarks_verified=True,
        continuous_curve_condition_count=continuous['condition_count'],
        continuous_mean_penalty=continuous['identity']['mean_penalty'],
        continuous_noise_penalty=continuous['identity']['noise_penalty'],
        continuous_nonnegative_audit=continuous['audit'],
        all_93_Xi_landmarks_match_frozen=False,
        independent_rate_windows_unchanged=True,
        EI_closure_max_nats=continuous['audit']['EI_closure_max_nats'],
        xi_minimum_nats=float(values['xi'].min()),
        xi_negative_within_tolerance_count=int(np.count_nonzero(values['xi'] < 0)),
        lower_panel_component_identity=correction['identity_sha256'],
        corrected_Xi_hits=scenario['cohorts']['all93']['methods']['xi']['hit_count'])
    return contract, g, values, rho, scenario, audit


def read_three_g_network_comparison():
    """Network games share the brain-map affine-TM conditions and seed set.

    Normalize within-network full Xi and exact between-network Shapley by the
    same condition's full Xi, then average seeds at each G. No G pooling.
    Read the precomputed corrected conditions; no fitting during composition.
    """
    record = json.loads((ROI / 'summary.json').read_text())
    with np.load(ROI / 'roi_shapley.npz') as a:
        g, seeds = a['G'].copy(), a['seeds'].copy()
        labels, membership = a['region_labels'].copy(), a['network_membership'].copy()
        names, root_xi = a['network_names'].tolist(), a['xi_nats'].copy()
    if g.tolist() != TREE_COUPLINGS or seeds.tolist() != [3, 4, 5]:
        raise ValueError('Network panels must match the brain-map G/seed set')
    groups = [tuple(np.flatnonzero(membership == k)) for k in range(7)]
    sizes = np.array([len(group) for group in groups])
    if len(labels) != 100 or sizes.sum() != 100 or np.any(sizes <= 0):
        raise ValueError('Invalid network/ROI metadata')
    tolerance = 1e-8  # native nats; no clipping or re-normalization
    raw = np.empty((3, 3, 2, 7))
    audits = []
    for gi, coupling in enumerate(g):
        for si, seed in enumerate(seeds):
            matches = [c for c in record['provenance']['conditions']
                       if c['G'] == coupling and c['seed'] == seed]
            if len(matches) != 1:
                raise ValueError('Expected one brain-map source condition')
            source = matches[0]
            path = ROOT / source['cache']
            if digest(path) != source['cache_sha256']:
                raise ValueError(f'Brain-map density cache changed: {path}')
            with np.load(path) as a:
                if (str(a['source_sha256']) != source['source_sha256']
                        or int(a['noise_seed']) != source['noise_seed']):
                    raise ValueError('Network queries must share the brain-map samples')
                game = CommonTargetGame(a['covariance'])
                total = game.totals()['xi_nats']
                if not np.isfinite(total) or total <= 0 or abs(total-root_xi[gi, si]) > tolerance:
                    raise ValueError('Network queries and brain-map root Xi differ')
                reused = 'within_network_xi' in a and 'between_network_shapley' in a
                if reused:
                    within = a['within_network_xi'].copy()
                    between = a['between_network_shapley'].copy()
                    query_audit = json.loads(str(a['network_audit_json']))
                else:
                    within, between, _, query_audit = network_values(game, groups)
            if within.shape != (7,) or between.shape != (7,):
                raise ValueError('Invalid per-network contribution arrays')
            component_audit = audit_nonnegative(np.r_[within, between], tolerance)
            if query_audit['nonnegative_audit']['violation_count']:
                raise ArithmeticError(f'Network query nonnegativity violation: {query_audit}')
            closure = float(abs(within.sum()+between.sum()-total))
            if closure > tolerance:
                raise ArithmeticError(f'Network budget error: {closure} nats')
            raw[gi, si] = np.stack([within, between])
            audits.append(dict(G=float(coupling), seed=int(seed), cache=source['cache'],
                cache_sha256=source['cache_sha256'], source_sha256=source['source_sha256'],
                root_xi_nats=total, reused_network_queries=reused, component_audit=component_audit,
                query_audit=query_audit, closure_error_nats=closure))
    shares = 100 * raw / root_xi[:, :, None, None]
    # Known G=0 independence is exact; raw Shapley roundoff is preserved in the
    # cache/audit. Show its declared numerical-zero band as zero on this plot.
    if np.abs(raw[0, :, 1]).max() > tolerance:
        raise ArithmeticError('G=0 network plot fails the independent-component control')
    print('G=0 display numerical-zero audit:', dict(tolerance_nats=tolerance,
        affected_count=int(np.count_nonzero(raw[0, :, 1])), minimum_nats=float(raw[0, :, 1].min()),
        raw_values_preserved=True))
    shares[0, :, 1] = 0.
    means = shares.mean(1)
    component_sd = shares.std(1, ddof=1)
    total_sd = shares.sum(2).std(1, ddof=1)
    if np.abs(shares.sum((2, 3)) - 100).max() > 1e-8:
        raise ArithmeticError('Per-condition percentage budget does not close')
    print('Three-G network comparison audit:', dict(
        G=g.tolist(), seeds=seeds.tolist(), tolerance_nats=tolerance, conditions=audits,
        totals_percent=means.sum(2).tolist(), mean_percent=means.tolist(),
        component_seed_sd_percent=component_sd.tolist(), total_seed_sd_percent=total_sd.tolist(),
        normalization='same brain-map condition full Xi, then seed mean at each G; no G pooling'))
    return names, sizes, means, component_sd, total_sd


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
    conditions = load_comparison(PAIRED_SUMMARY, TREE_SEED, TREE_COUPLINGS[1:])
    for tree, source in conditions:
        print("Cached tree audit:", source, audit_tree(tree, tree.xi_bits))
    independent = summary['G0_hierarchy']
    audit_nonnegative(independent['local_roi_xi_nats'], 1e-8)
    if (independent['leaf_count'] != 100 or abs(independent['cross_roi_nats']) > 1e-8
            or abs(sum(independent['local_roi_xi_nats'])-independent['full_xi_nats']) > 1e-8):
        raise ArithmeticError('Corrected G=0 local-ROI information budget fails')
    conditions.insert(0, (None, independent))
    return g, values, rho, scenario, labels, membership, network_names, summary, conditions


def draw(dpi):
    g, values, rho, scenario, labels, membership, network_names, summary, conditions = load_inputs()
    images = cortex_crops()
    network_names, network_sizes, network_means, component_sd, total_sd = read_three_g_network_comparison()
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
            ("b", "xi", 80, 14, r"$\Xi$ · continuous TM"),
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
        colorbars = []
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=palette),
                          cax=axes_mm(fig, 135, 18, 1.6, 54), orientation="vertical")
        cb.set_label("Native SC spectral radius", fontsize=6.2, labelpad=3,
                     rotation=270, va='bottom')
        cb.ax.tick_params(labelsize=6.0, length=1.5, pad=1.2)
        cb.outline.set_linewidth(.5)
        colorbars.append(cb)

        ax = axes_mm(fig, 154, 14, 31, 62)
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
        hit_xlabel = ax.xaxis.label
        letter(fig, 144, 10.7, "e")
        hit_note = text_mm(fig, 169.5, 86.5, "Window width: 0.5 $G$\nPost hoc; $\Delta G=0.1$",
                fontsize=6.4, color=".35", ha="center", linespacing=1.4)

        # Shared state columns couple hierarchy and anatomy without extra empty cells.
        starts = [3, 62, 121]
        descriptions = ["No long-range coupling", "Transition range", "Extreme coupling"]
        couplings = [0, 1.3, 10]
        for i, (x, coupling, description, condition) in enumerate(zip(starts, couplings, descriptions, conditions)):
            tree, source = condition
            text_mm(fig, x+28.5, 95, rf"$G={coupling:g}$  ·  {description}",
                    ha="center", fontsize=7.7)
            ax = axes_mm(fig, x, 101, 57, 31)
            if coupling == 0:
                print('G=0 trivial tree display audit:', render_g0_trivial_tree(ax, source, membership))
                letter(fig, x, 101, 'f')
                text_mm(fig, x+28.5, 134.5, 'Cross-ROI Syn = 0 · ROI leaves',
                    ha='center', fontsize=5.6, color='.30')
                continue
            if coupling == 10:
                print('G=10 one-level display audit:', render_g10_one_level_tree(ax, tree, membership))
                letter(fig, x, 101, 'h')
                text_mm(fig, x+28.5, 134.5, 'One-level display · 99 binary forks condensed',
                        ha='center', fontsize=5.6, color='.30')
                continue
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
            letter(fig, x, 101, "fgh"[i])
            print("Displayed tree audit:", source, {k:v for k,v in tree_audit.items()
                  if k not in {"nodes", "roi_order", "original_tree_metrics"}})
            original_depth = tree_audit['original_tree_metrics']['maximum_depth']
            depth = str(tree_audit['display_maximum_depth'])
            if condense:
                depth += f" (from {original_depth})"
            text_mm(fig, x+28.5, 134.5,
                    f"Multi-ROI forks {tree_audit['multi_roi_fork_count']} · Display depth {depth}",
                    ha="center", fontsize=5.6, color=".30")
        network_labels = ["VIS", "SM", "DA", "SA/VA", "LIM", "FPN", "DMN"]
        fig.legend(handles=[Patch(facecolor=c, edgecolor="none", label=n)
                            for c, n in zip(NETWORK_COLORS, network_labels)],
                   loc="center left", bbox_to_anchor=(3/WIDTH, 1-(140.5)/HEIGHT),
                   ncol=7, frameon=False, fontsize=6.4, handlelength=.8,
                   handletextpad=.35, columnspacing=.9, borderaxespad=0.)
        tree_cmap = LinearSegmentedColormap.from_list("local_syn",
                    [_blend_with_white(SPLIT_COLOR, .12), _blend_with_white(SPLIT_COLOR, .74)])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 3), cmap=tree_cmap),
                          cax=axes_mm(fig, 179, 104, 1.6, 24), orientation="vertical",
                          ticks=[0, 1, 2, 3])
        cb.ax.tick_params(labelsize=5.4, length=1.5, pad=1.)
        cb.outline.set_linewidth(.4)
        cb.set_label(r"g: Local Syn / $\Xi$ (%)", fontsize=5.9, labelpad=3,
                     rotation=270, va='bottom')
        colorbars.append(cb)

        text_mm(fig, 3, 148, "ROI attribution · mean across 3 seeds",
                fontsize=6.6, weight="bold")
        for i, x in enumerate(starts):
            row = summary["state_totals"][i]
            letter(fig, x, 153, "ijk"[i])
            text_mm(fig, x+30, 153,
                    rf"$\Xi={row['xi_mean_nats']:.2f}\pm{row['xi_seed_sd_nats']:.2f}$ nats",
                    fontsize=6.8, ha="center")
            for col in range(4):
                ax = axes_mm(fig, x + (col % 2)*29, 157 + (col//2)*18.2, 28, 17.2)
                ax.imshow(images[4*i+col], interpolation="none")
                ax.axis("off")
        cb = fig.colorbar(plt.cm.ScalarMappable(
            norm=Normalize(summary["plot"]["vmin"], summary["plot"]["vmax"]), cmap="viridis"),
            cax=axes_mm(fig, 179, 158, 1.6, 34), orientation="vertical")
        cb.set_ticks(np.linspace(summary["plot"]["vmin"], summary["plot"]["vmax"], 5))
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        cb.ax.tick_params(labelsize=6.0, length=1.5, pad=1.2)
        cb.outline.set_linewidth(.5)
        cb.set_label(r"ROI attribution to overall $\Xi$ (%)", fontsize=6.2, labelpad=3,
                     rotation=270, va='bottom')
        colorbars.append(cb)
        # One stacked network comparison per G, aligned to the tree/cortex
        # columns above. Hatch distinguishes between-network attribution.
        order = np.argsort(network_means[1, 0])[::-1]  # freeze the G=1.3 within order
        ypos = np.arange(7)[::-1]
        short = ['VIS', 'SM', 'DA', 'SA/VA', 'LIM', 'FPN', 'DMN']
        rows = [f'{short[i]} / {network_sizes[i]}' for i in order]
        bar_colors = [NETWORK_COLORS[i] for i in order]
        faded = [_blend_with_white(color, .24) for color in bar_colors]
        upper = 5 * np.ceil(np.max(network_means.sum(1) + total_sd) * 1.06 / 5)
        with plt.rc_context({'hatch.linewidth': .35}):
            fig.legend(handles=[
                Patch(facecolor='#777777', edgecolor='none', label=r'Within-network $\Xi$'),
                Patch(facecolor='#EEEEEE', edgecolor='#777777', hatch='///', linewidth=.4,
                      label='Between-network Shapley')],
                loc='center', bbox_to_anchor=(90/WIDTH, 1-201.2/HEIGHT), ncol=2,
                fontsize=6.1, frameon=False, handlelength=1.6, columnspacing=1.7)
            for i, (x, coupling) in enumerate(zip([v+15 for v in starts], TREE_COUPLINGS)):
                within, between = network_means[i, :, order].T
                combined = within + between
                ax = axes_mm(fig, x, 212, 40, 32)
                ax.barh(ypos, within, height=.72, color=bar_colors, linewidth=0, zorder=2)
                ax.barh(ypos, between, left=within, height=.72, color=faded,
                        edgecolor=bar_colors, hatch='///', linewidth=.35, zorder=2)
                # Internal endpoint SD uses the within share; outer endpoint SD
                # uses each seed's combined share, preserving their covariance.
                ax.errorbar(within, ypos, xerr=component_sd[i, 0, order], fmt='none',
                            ecolor='#202020', elinewidth=.55, capsize=1.1, capthick=.55, zorder=3)
                if coupling != 0:  # At G=0 this is the same endpoint/SD already drawn.
                    ax.errorbar(combined, ypos, xerr=total_sd[i, order], fmt='none',
                                ecolor='#202020', elinewidth=.55, capsize=1.1, capthick=.55, zorder=3)
                ax.set(xlim=(0, upper), ylim=(-.65, 6.65), xticks=np.arange(0, upper+.1, 10),
                       yticks=ypos, yticklabels=rows, xlabel=r'Overall $\Xi$ share (%)')
                ax.tick_params(axis='y', labelsize=6.1, pad=1.2, length=0)
                ax.tick_params(axis='x', labelsize=6.1)
                ax.set_xlabel(ax.get_xlabel(), fontsize=6.3, labelpad=2)
                ax.grid(axis='x', color='.91', lw=.5, zorder=0)
                letter(fig, starts[i], 208, 'lmn'[i])
                text_mm(fig, starts[i]+28.5, 208, rf'$G={coupling:g}$', fontsize=7.4, ha='center')
                text_mm(fig, starts[i]+28.5, 253,
                        f'Within {within.sum():.1f}% · Between {between.sum():.1f}%',
                        ha='center', fontsize=5.7, color='.35')

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        # Audit final renderer extents of vertical guides and outside legends.
        # This is an export/layout check; no scientific data are transformed.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        guide_axes = {cb.ax for cb in colorbars}
        data_axes = [a for a in fig.axes if a not in guide_axes]
        data_boxes = [a.get_window_extent(renderer) for a in data_axes]
        label_boxes = [(artist, artist.get_window_extent(renderer)) for a in data_axes if a.axison
            for artist in [a.xaxis.label, a.yaxis.label, a.title,
                           *a.get_xticklabels(), *a.get_yticklabels()]
            if artist.get_visible() and artist.get_text()]
        guide_boxes = [cb.ax.get_tightbbox(renderer) for cb in colorbars]
        guide_boxes += [legend.get_window_extent(renderer) for legend in fig.legends]
        for box in guide_boxes:
            if (box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height):
                raise ValueError('Vertical colorbar or legend exceeds the export canvas')
            if any(box.overlaps(data) for data in data_boxes):
                raise ValueError('Vertical colorbar or legend overlaps a data panel')
            for artist, label in label_boxes:
                if box.overlaps(label):
                    raise ValueError(f'Vertical colorbar or legend overlaps panel label: {artist.get_text()}')
        if hit_note.get_window_extent(renderer).overlaps(hit_xlabel.get_window_extent(renderer)):
            raise ValueError('Hit-window note overlaps the axis label')
        print('Vertical guide audit:', dict(colorbars=len(colorbars),
            all_guides_within_canvas=True, no_data_overlap=True,
            canvas_mm=[WIDTH, HEIGHT], previous_height_mm=269.))
        for suffix in (".png", ".pdf"):
            path = OUTPUT.with_suffix(suffix)
            fig.savefig(path, dpi=dpi, bbox_inches=None)
            print("Saved:", path)
        plt.close(fig)


def draw_g0_detail(dpi):
    """A larger view makes all 100 ROI leaves inspectable."""
    load_manifest()
    record = json.loads((ROI / 'summary.json').read_text())['G0_hierarchy']
    with np.load(ROI / 'roi_shapley.npz') as a:
        membership = a['network_membership'].copy()
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':9,
                         'pdf.fonttype':42, 'savefig.facecolor':'white'}):
        fig = plt.figure(figsize=(7.48, 2.55), facecolor='white')
        ax = fig.add_axes([.035, .23, .93, .66])
        render_g0_trivial_tree(ax, record, membership, fontsize=10)
        fig.text(.5, .99, rf'$G=0$, seed 4 · overall $\Xi={record["full_xi_nats"]:.3f}$ nats',
                 ha='center', va='top', fontsize=10)
        fig.text(.5, .17, 'Each ROI leaf contains E/I coordinates. Cross-ROI Syn = 0.',
                 ha='center', fontsize=8.5, color='.30')
        fig.legend(handles=[Patch(facecolor=c, edgecolor='none', label=n)
            for c,n in zip(NETWORK_COLORS, ['VIS','SM','DA','SA/VA','LIM','FPN','DMN'])],
            loc='lower center', bbox_to_anchor=(.5, .025), ncol=7,
            frameon=False, fontsize=8, columnspacing=1.2)
        output = OUTPUT.parent / 'dmf_G0_trivial_tree.png'
        fig.savefig(output, dpi=dpi, bbox_inches=None)
        plt.close(fig)
        print('Saved:', output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=450)
    dpi = parser.parse_args().dpi
    draw(dpi)
    draw_g0_detail(dpi)
