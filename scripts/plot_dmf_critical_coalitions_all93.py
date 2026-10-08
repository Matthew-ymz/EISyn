#!/usr/bin/env python3
"""All-subject exact-coalition overview using the actual G grid and raw Syn."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_dmf_critical_coalitions import BASE
from scripts.run_dmf_subject_consistency import atomic_json, digest
from scripts.analyze_dmf_critical_coalitions import wilson


def aligned_rates(repeated, c, subjects):
    by_relative = {}
    for pi, subject in enumerate(subjects):
        mid = c['windows'][subject]['transition']['midpoint']
        for gi, g in enumerate(c['G']):
            key = int(round((g - mid) * 20))
            if abs(g - mid - key / 20) > 1e-10:
                raise ValueError('Actual relative G does not lie on the shared half-step grid')
            by_relative.setdefault(key, []).append(bool(repeated[pi, gi]))
    keys = sorted(by_relative)
    return np.array(keys) / 20, np.array([sum(by_relative[k]) for k in keys]), np.array([len(by_relative[k]) for k in keys])


def plot_main_figures(base, output_dir, presence_only=False, per_seed_figures=False,
                      combined_seeds_figure=False):
    """C1–C3 individual scans and aligned recurrence from the same frozen cache."""
    c = json.loads((base / 'contract.json').read_text())
    summary = json.loads((base / 'all93_summary.json').read_text())
    sha = digest(base / 'contract.json')
    extension_sha = digest(base / 'all93_extension_contract.json')
    if (summary['status'] != 'complete' or summary['subject_count'] != 93
            or summary['condition_count'] != 11439):
        raise ValueError('Main figures require the complete 93-subject result')
    if (summary['scientific_contract_sha256'] != sha
            or summary['extension_contract_sha256'] != extension_sha):
        raise ValueError('Main figure summary provenance mismatch')
    with np.load(base / 'all93_display.npz') as a:
        subjects = a['subject_ids'].tolist()
        g = a['G'].copy()
        values = a['raw_syn_nats'].copy()
        raw = a['repeated_presence'].copy()
        strong = a['repeated_above_background'].copy()
        if (str(a['scientific_contract_sha256']) != sha
                or str(a['extension_contract_sha256']) != extension_sha):
            raise ValueError('Main figure display provenance mismatch')
        if json.loads(str(a['members_json'])) != [r['members'] for r in summary['selected']]:
            raise ValueError('Display membership/summary mismatch')
        if not np.array_equal(a['seeds'], summary['seeds']):
            raise ValueError('Display seed grid mismatch')
    if subjects != summary['subject_ids'] or not np.array_equal(g, c['G']):
        raise ValueError('Display subject/G grid mismatch')
    selected = summary['selected']
    n = len(subjects)
    expected_shape = (len(selected), n, len(g), len(summary['seeds']))
    if values.shape != expected_shape or raw.shape != expected_shape[:3] or strong.shape != raw.shape:
        raise ValueError('Display array shape mismatch')
    if np.isinf(values).any():
        raise ArithmeticError('Nonfinite selected-node Syn: infinity in display cache')
    finite = values[np.isfinite(values)]
    tolerance = c['syn_tolerance_nats']
    violations = int((finite < -tolerance).sum())
    if violations:
        raise ArithmeticError(f'Invalid Syn: minimum={finite.min()} nats, '
                              f'threshold={-tolerance} nats, affected={violations}')
    if not np.array_equal(raw, np.isfinite(values).sum(3) >= c['repeat_seed_count']):
        raise ValueError('Repeated node presence does not match raw display values')
    for ci, candidate in enumerate(selected):
        threshold = candidate['threshold_nats']
        expected = np.zeros_like(strong[ci]) if threshold is None else (
            (values[ci] > threshold).sum(2) >= c['repeat_seed_count'])
        if not np.array_equal(strong[ci], expected):
            raise ValueError('Strong-node occurrence does not match frozen threshold')

    midpoints = np.array([c['windows'][p]['transition']['midpoint'] for p in subjects])
    permutation = sorted(range(n), key=lambda pi: midpoints[pi])
    for pi, subject in enumerate(subjects):
        relative_window = g[c['windows'][subject]['grid_indices']] - midpoints[pi]
        if not np.allclose(relative_window, [-.25, -.15, -.05, .05, .15, .25], atol=1e-10):
            raise ValueError('Six-point window differs from the frozen aligned grid')
    # These are cell edges, not a wider scientific window.
    cell_halfwidth = float(np.diff(g)[0] / 2)
    xlim = (float((g[0] - midpoints).min() - cell_halfwidth),
            float((g[-1] - midpoints).max() + cell_halfwidth))
    window_edges = [-.3, .3]
    colours = ['#236A91', '#BA5636', '#717A38']
    recurrence = raw if presence_only else strong
    output_suffix = '_presence' if presence_only else ''
    output_dir.mkdir(parents=True, exist_ok=True)
    individual_outputs = [output_dir / (name + output_suffix + '.png') for name in (
        'critical_coalition_all93_individuals',
        'critical_coalition_all93_individuals_c2',
        'critical_coalition_all93_individuals_c3')]
    recurrence_output = output_dir / ('critical_coalition_all93_recurrence' + output_suffix + '.png')
    peaks = []
    individual_state_counts = {}
    seed_outputs = []
    combined_seed_output = None
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
            'axes.spines.top': False, 'axes.spines.right': False,
            'axes.labelsize': 11, 'savefig.facecolor': 'white'}):
        state_colours = (['#EBECEE', '#BCC0C5', colours[0]] if presence_only else
                         ['#EBECEE', '#BCC0C5', '#7A8B9B', colours[0]])
        state_labels = (['Absent in all 3 seeds', 'Present in 1 seed', 'Present in ≥2 of 3 seeds']
                        if presence_only else ['Absent in all 3 seeds', 'Present in 1 seed',
                            'Present in ≥2 seeds;\nreference not met',
                            'Above reference in\n≥2 of 3 seeds'])
        cmap = ListedColormap(state_colours)
        norm = BoundaryNorm(np.arange(len(state_colours) + 1) - .5, len(state_colours))
        for ci, individual_output in enumerate(individual_outputs):
            candidate = selected[ci]
            fig, ax = plt.subplots(figsize=(10.8, 6.8), layout='constrained')
            states = ((np.isfinite(values[ci]).sum(2) > 0).astype(int)
                      + raw[ci].astype(int))
            if not presence_only:
                states += strong[ci].astype(int)
            individual_state_counts[candidate['display_id']] = np.bincount(
                states.ravel(), minlength=len(state_colours)).tolist()
            for row, pi in enumerate(permutation):
                edges = np.r_[g - cell_halfwidth, g[-1] + cell_halfwidth] - midpoints[pi]
                ax.pcolormesh(edges, [row - .5, row + .5], states[pi][None, :],
                              cmap=cmap, norm=norm, shading='flat', rasterized=True)
            for x in window_edges:
                ax.axvline(x, color=colours[1], ls='--', lw=1)
            positions = list(range(0, n, 6))
            ax.set_yticks(positions, [subjects[permutation[i]].replace('sub-', '') for i in positions])
            ax.set(xlabel=r'$\Delta G = G - G_i^{*}$ (individual rate-transition midpoint)',
                   ylabel='Individual SC (93 rows; sorted by transition midpoint)',
                   xlim=xlim, ylim=(n - .5, -.5))
            handles = [Patch(facecolor=cmap(i), edgecolor='#BBBBBB', label=label)
                       for i, label in enumerate(state_labels)]
            handles += [Patch(facecolor='white', edgecolor='#BBBBBB', label='Outside scanned G'),
                        Line2D([], [], color=colours[1], ls='--', lw=1,
                               label='Six-point window\n(display cell edges)')]
            title = candidate['display_id'] + ': ROI ' + ', '.join(map(str, candidate['members_one_based']))
            ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, .5),
                      frameon=False, title=title, labelspacing=1.2)
            fig.savefig(individual_output, dpi=240, bbox_inches='tight')
            plt.close(fig)

        if per_seed_figures:
            for si, seed in enumerate(summary['seeds']):
                fig, axes = plt.subplots(1, 3, figsize=(14.4, 6.8), sharey=True,
                                         layout='constrained')
                seed_cmap = ListedColormap(['#EBECEE', colours[0]])
                seed_norm = BoundaryNorm([-.5, .5, 1.5], 2)
                counts = {}
                for ci, ax in enumerate(axes):
                    candidate = selected[ci]
                    present = np.isfinite(values[ci, :, :, si])
                    counts[candidate['display_id']] = int(present.sum())
                    for row, pi in enumerate(permutation):
                        edges = np.r_[g - cell_halfwidth, g[-1] + cell_halfwidth] - midpoints[pi]
                        ax.pcolormesh(edges, [row - .5, row + .5], present[pi][None, :],
                                      cmap=seed_cmap, norm=seed_norm, shading='flat', rasterized=True)
                    for edge in window_edges:
                        ax.axvline(edge, color=colours[1], ls='--', lw=1)
                    ax.set(xlim=xlim, ylim=(n - .5, -.5), xlabel=r'$\Delta G = G - G_i^{*}$')
                    ax.set_title(candidate['display_id'] + ': ROI ' + ', '.join(
                        map(str, candidate['members_one_based'])), fontsize=10, pad=10)
                positions = list(range(0, n, 6))
                axes[0].set_yticks(positions, [subjects[permutation[i]].replace('sub-', '') for i in positions])
                axes[0].set_ylabel('Individual SC (93 rows; same order in all panels)')
                handles = [Patch(facecolor='#EBECEE', edgecolor='#BBBBBB', label='Absent in this seed'),
                    Patch(facecolor=colours[0], label='Present in this seed'),
                    Patch(facecolor='white', edgecolor='#BBBBBB', label='Outside scanned G'),
                    Line2D([], [], color=colours[1], ls='--', lw=1,
                           label='Six-point window\n(display cell edges)')]
                axes[-1].legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, .5),
                                title=f'Seed {seed}', frameon=False, labelspacing=1.2)
                output = output_dir / f'critical_coalition_all93_seed{seed}_candidates.png'
                fig.savefig(output, dpi=240, bbox_inches='tight')
                plt.close(fig)
                seed_outputs.append(dict(seed=int(seed), output=str(output),
                    figure_sha256=digest(output), present_subject_G_counts=counts))

        if combined_seeds_figure:
            if summary['seeds'] != [3, 4, 5]:
                raise ValueError('The original three-seed combined plate requires seeds 3, 4, 5')
            # One cell per person/G: count exact-node presence across the three seeds.
            seed_counts = np.isfinite(values[:3]).sum(3)
            count_colours = ['#EBECEE', '#B9CEDA', '#6D9DB6', colours[0]]
            count_cmap = ListedColormap(count_colours)
            count_norm = BoundaryNorm(np.arange(5) - .5, 4)
            fig, axes = plt.subplots(1, 3, figsize=(14.4, 6.8), sharey=True,
                                     layout='constrained')
            counts = {}
            for ci, ax in enumerate(axes):
                candidate = selected[ci]
                counts[candidate['display_id']] = np.bincount(seed_counts[ci].ravel(), minlength=4).tolist()
                for row, pi in enumerate(permutation):
                    edges = np.r_[g - cell_halfwidth, g[-1] + cell_halfwidth] - midpoints[pi]
                    ax.pcolormesh(edges, [row - .5, row + .5], seed_counts[ci, pi][None, :],
                                  cmap=count_cmap, norm=count_norm, shading='flat', rasterized=True)
                for edge in window_edges:
                    ax.axvline(edge, color=colours[1], ls='--', lw=1)
                ax.set(xlim=xlim, ylim=(n - .5, -.5), xlabel=r'$\Delta G = G - G_i^{*}$')
                ax.set_title(candidate['display_id'] + ': ROI ' + ', '.join(
                    map(str, candidate['members_one_based'])), fontsize=10, pad=10)
            positions = list(range(0, n, 6))
            axes[0].set_yticks(positions, [subjects[permutation[i]].replace('sub-', '') for i in positions])
            axes[0].set_ylabel('Individual SC (93 rows; same order in all panels)')
            handles = [Patch(facecolor=count_colours[k], edgecolor='#BBBBBB',
                             label=f'{k}/3 seeds' + (' (absent)' if k == 0 else '')) for k in range(4)]
            handles += [Patch(facecolor='white', edgecolor='#BBBBBB', label='Outside scanned G'),
                Line2D([], [], color=colours[1], ls='--', lw=1,
                       label='Six-point window\n(display cell edges)')]
            axes[-1].legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, .5),
                            title='Presence across seeds 3, 4, 5', frameon=False, labelspacing=1.2)
            output = output_dir / 'critical_coalition_all93_seeds345_combined.png'
            fig.savefig(output, dpi=240, bbox_inches='tight')
            plt.close(fig)
            combined_seed_output = dict(output=str(output), figure_sha256=digest(output), seeds=[3, 4, 5],
                scanned_cell_state_counts=counts,
                aggregation='One cell per individual/G; number of seeds containing the fixed complete natural node',
                colour_semantics='Gray=0/3; increasing blue=1/3, 2/3, 3/3; white=not scanned; no Syn strength reference')

        fig, (ax, coverage_ax) = plt.subplots(2, 1, figsize=(10.8, 5.6), sharex=True,
            layout='constrained', gridspec_kw={'height_ratios': [3.5, 1], 'hspace': .05})
        for ci in range(3):
            x, hits, denominators = aligned_rates(recurrence[ci], c, subjects)
            rate = hits / denominators * 100
            candidate = selected[ci]
            label = candidate['display_id'] + ': ROI ' + ', '.join(map(str, candidate['members_one_based']))
            ax.plot(x, rate, color=colours[ci], lw=1.5, ls=['-', '--', '-.'][ci],
                    marker=['o', 's', '^'][ci], ms=3.7, label=label)
            if ci == 0:
                coverage_x, coverage_n = x.copy(), denominators.copy()
                bounds = np.array([wilson(int(a), int(b)) for a, b in zip(hits, denominators)]) * 100
                ax.fill_between(x, bounds[:, 0], bounds[:, 1], color=colours[ci], alpha=.13, lw=0)
            else:
                if not (np.array_equal(x, coverage_x) and np.array_equal(denominators, coverage_n)):
                    raise ValueError('Candidate aligned denominator mismatch')
            peak = int(rate.argmax())
            peaks.append(dict(display_id=candidate['display_id'], relative_G=float(x[peak]),
                              hit_subjects=int(hits[peak]), covered_subjects=int(denominators[peak]),
                              recurrence_percent=float(rate[peak])))
        for axes in (ax, coverage_ax):
            axes.axvspan(*window_edges, color=colours[1], alpha=.08, lw=0)
            axes.set_xlim(*xlim)
        handles, labels = ax.get_legend_handles_labels()
        handles += [Patch(facecolor=colours[0], alpha=.13), Patch(facecolor=colours[1], alpha=.08)]
        labels += ['C1: descriptive Wilson 95%\ninterval after selection', 'Six-point transition window\n(display cell footprint)']
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, .5),
                  frameon=False, labelspacing=1.2)
        ax.set(ylabel=('Subjects with repeated node (%)' if presence_only else 'Subjects with strong node (%)'),
               ylim=(-2, 102), yticks=[0, 25, 50, 75, 100])
        coverage_ax.plot(coverage_x, coverage_n, color='#5A6066', lw=1.2, marker='.', ms=3)
        coverage_ax.set(ylabel='Covered\nsubjects (n)', ylim=(0, 100), yticks=[0, 50, 93],
                        xlabel=r'$\Delta G = G - G_i^{*}$ (individual rate-transition midpoint)')
        fig.savefig(recurrence_output, dpi=240, bbox_inches='tight')
        plt.close(fig)
    metadata = dict(subject_count=n, condition_count=summary['condition_count'],
        summary_sha256=digest(base / 'all93_summary.json'),
        display_sha256=digest(base / 'all93_display.npz'), scientific_contract_sha256=sha,
        extension_contract_sha256=extension_sha, plotting_sha256=digest(Path(__file__)),
        source_panels='Existing all93 overview panels c/d; identical individual heatmap extended to C2/C3; no new simulation or inference',
        figures=[dict(output=str(p), figure_sha256=digest(p)) for p in (*individual_outputs, recurrence_output)],
        per_seed_figures=seed_outputs,
        combined_seed_figure=combined_seed_output,
        selected_members_one_based=[r['members_one_based'] for r in selected[:3]],
        readout_mode='natural_presence' if presence_only else 'above_background',
        candidate_selection='C1–C3 fixed from the completed original all93 exploratory search; not reselected',
        individual_row_order=[subjects[pi] for pi in permutation],
        individual_state_counts=individual_state_counts,
        individual_colour_semantics=('Same three-state categorical palette for C1/C2/C3; blue=≥2/3 seed natural presence'
            if presence_only else 'Same four-state categorical palette for C1/C2/C3; blue=strong occurrence, curve hues identify candidates'),
        background_filter_disagreement_count={r['display_id']: int((raw[ci] != strong[ci]).sum())
            for ci, r in enumerate(selected[:3])},
        actual_window_centres=[-.25, -.15, -.05, .05, .15, .25],
        critical_cell_footprint=window_edges, xlim=xlim, peaks=peaks,
        relative_coverage=dict(relative_G=coverage_x.tolist(), subject_count=coverage_n.tolist()),
        occurrence_rule=('Exact natural node present in at least 2/3 seeds; no Syn strength threshold'
            if presence_only else 'Exact natural node; >same-order frozen G=0 reference in at least 2/3 seeds'),
        normalization='At each actual aligned G: hits / covered individual SCs; no smoothing',
        uncertainty='C1 Wilson 95% descriptive pointwise subject proportion interval after selection',
        missing_rule='Absent node remains NaN in raw Syn; scanned absence is gray; not scanned is white',
        native_syn_tolerance_nats=tolerance, tolerance_negative_count=int((finite < 0).sum()),
        significant_negative_count=violations, legend_placement='Outside every data axes',
        manuscript=dict(checked_date='2026-10-08', parent_key='P6UJCVG8',
            title='Emergent hierarchical organization of causal interactions in complex systems',
            main_attachment='DXGC7JEA', supplement_attachment='MWIWKSVG',
            version='No explicit draft version/date; attachment metadata does not resolve version',
            relevant_locations='Main Brain/Fig.2 pp6–7, Methods Eqs7–12 pp16–17; supplement S5/Algorithm S1 pp8–9'),
        visual_review='pending')
    metadata_name = 'all93_presence_figures_metadata.json' if presence_only else 'all93_main_figures_metadata.json'
    atomic_json(base / metadata_name, metadata)
    for output in (*individual_outputs, recurrence_output):
        print(output, flush=True)
    for record in seed_outputs:
        print(record['output'], flush=True)
    if combined_seed_output:
        print(combined_seed_output['output'], flush=True)


def plot(base, output, development_only=False):
    prefix = 'all93_devcheck' if development_only else 'all93'
    c = json.loads((base / 'contract.json').read_text())
    summary = json.loads((base / (prefix + '_summary.json')).read_text())
    if summary['scientific_contract_sha256'] != digest(base / 'contract.json'):
        raise ValueError('Summary scientific provenance mismatch')
    with np.load(base / (prefix + '_display.npz')) as a:
        subjects = a['subject_ids'].tolist(); g = a['G'].copy()
        values = a['raw_syn_nats'].copy(); raw = a['repeated_presence'].copy()
        strong = a['repeated_above_background'].copy()
        if json.loads(str(a['members_json'])) != [r['members'] for r in summary['selected']]:
            raise ValueError('Display membership/summary mismatch')
    n = len(subjects); selected = summary['selected']; k = min(3, len(selected))
    colours = ['#236A91', '#BA5636', '#717A38']
    relative_coverage = None
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':9,
            'axes.spines.top':False, 'axes.spines.right':False, 'axes.labelsize':10,
            'savefig.facecolor':'white'}):
        fig = plt.figure(figsize=(12.5, 11.2), layout='constrained')
        gs = fig.add_gridspec(2,2, width_ratios=[1.08,1], height_ratios=[1,1.8], wspace=.13, hspace=.14)
        ax = fig.add_subplot(gs[0,0])
        by_order = summary['by_order'][1:]; orders = [r['order'] for r in by_order]
        for mode, metric, color, ls, label in [
            ('natural_presence','critical_ge4','#85898E','--','Natural node, ≥4/6 window points'),
            ('above_background','critical_ge4','#236A91','-','Strong node, ≥4/6 window points'),
            ('above_background','critical_ge3','#BA5636','-','Strong node, ≥3/6 window points'),
            ('above_background','critical_ge1','#717A38','-','Strong node, ≥1/6 window points')]:
            y = [r['maxima'][mode][metric] for r in by_order]
            ax.plot(orders, y, color=color, ls=ls, lw=1.1, marker='.', ms=3, label=label)
        ax.set(xlabel='Exact coalition order (ROIs)', ylabel=f'Maximum subject count (of {n})',
            xlim=(2,100), ylim=(-.025*n,1.025*n))
        ax.legend(loc='lower left', bbox_to_anchor=(0,1.02), frameon=False, fontsize=8)
        ax.text(-.13,1.04,'a',transform=ax.transAxes,fontweight='bold',fontsize=12)

        ax = fig.add_subplot(gs[0,1])
        metrics = ['low_any','critical_ge1','critical_ge3','critical_ge4','high_any']
        for ci in range(k):
            row = selected[ci]; stats = row['above_background']
            y = np.array([stats[name] for name in metrics]) / n * 100
            bounds = np.array([wilson(stats[name],n) for name in metrics]) * 100
            ax.errorbar(np.arange(5)+(ci-(k-1)/2)*.13, y,
                yerr=np.array([y-bounds[:,0],bounds[:,1]-y]),
                color=colours[ci], marker=['o','s','^'][ci], ms=4.8, lw=1.1, capsize=2.5,
                label=row['display_id']+': '+','.join(str(i+1) for i in row['members']))
        ax.set(xticks=range(5), xticklabels=['Low G\n(any)','Window\n≥1/6','Window\n≥3/6','Window\n≥4/6','High G\n(any)'],
            ylabel='Subjects with reproducible strong node (%)', ylim=(-3,103), xlim=(-.4,4.4))
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),frameon=False,fontsize=8)
        ax.text(-.14,1.04,'b',transform=ax.transAxes,fontweight='bold',fontsize=12)

        ax = fig.add_subplot(gs[1,0])
        cmap = ListedColormap(['#EBECEE','#BCC0C5','#7A8B9B','#246A91'])
        norm = BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
        permutation = sorted(range(n),key=lambda pi:c['windows'][subjects[pi]]['transition']['midpoint'])
        for row, pi in enumerate(permutation):
            midpoint = c['windows'][subjects[pi]]['transition']['midpoint']
            state = (np.isfinite(values[0,pi]).sum(1)>0).astype(int) + raw[0,pi].astype(int) + strong[0,pi].astype(int)
            edges = np.r_[g-.05, g[-1]+.05] - midpoint
            ax.pcolormesh(edges,[row-.5,row+.5],state[None,:],cmap=cmap,norm=norm,shading='flat',rasterized=True)
        for x in [-.3,.3]:
            ax.axvline(x,color='#BA5636',ls='--',lw=.8)
        positions = list(range(0,n,max(1,n//14)))
        ax.set_yticks(positions,[subjects[permutation[i]].replace('sub-','') for i in positions])
        ax.set(xlabel='G − individual rate-transition midpoint', ylabel=f'{selected[0]["display_id"]}: individual SC (sorted by transition)',
            xlim=(-3.05,3.65), ylim=(n-.5,-.5))
        handles=[Patch(color=cmap(i),label=label) for i,label in enumerate(
            ['Absent in all seeds','One seed only','Repeated, below reference','Repeated, above reference'])]
        handles.append(Patch(facecolor='white',edgecolor='#BBBBBB',label='Outside scanned G'))
        ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,ncol=2,fontsize=7.6)
        ax.text(-.13,1.04,'c',transform=ax.transAxes,fontweight='bold',fontsize=12)

        nested = gs[1,1].subgridspec(2,1,height_ratios=[1,1],hspace=.16)
        ax = fig.add_subplot(nested[0,0])
        for ci in range(k):
            x, hits, denominators = aligned_rates(strong[ci],c,subjects)
            rate = hits / denominators * 100
            ax.plot(x,rate,color=colours[ci],lw=1.1,marker=['o','s','^'][ci],ms=2.7,label=selected[ci]['display_id'])
            if ci==0:
                bounds=np.array([wilson(int(a),int(b)) for a,b in zip(hits,denominators)])*100
                ax.fill_between(x,bounds[:,0],bounds[:,1],color=colours[ci],alpha=.12,lw=0)
                relative_coverage=dict(relative_G=x.tolist(),subject_count=denominators.tolist())
        ax.axvspan(-.3,.3,color='#BA5636',alpha=.08,lw=0)
        ax.set(xlabel='G − individual rate-transition midpoint', ylabel='Strong-node recurrence (%)',
            xlim=(-3.05,3.65),ylim=(-2,102))
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,ncol=3,fontsize=8)
        ax.text(-.14,1.04,'d',transform=ax.transAxes,fontweight='bold',fontsize=12)

        ax = fig.add_subplot(nested[1,0])
        stage_colours=['#A4A9AF','#236A91','#BA5636','#766294']
        labels=['Other G','Low G','Transition window','High G']
        for pi, subject in enumerate(subjects):
            x = g - c['windows'][subject]['transition']['midpoint']
            stage=np.zeros(len(g),int); stage[c['low_grid_indices']]=1; stage[c['high_grid_indices']]=3
            stage[c['windows'][subject]['grid_indices']]=2
            for si in range(3):
                for label in range(4):
                    keep=(stage==label)&np.isfinite(values[0,pi,:,si])
                    ax.scatter(x[keep],values[0,pi,keep,si],s=9,color=stage_colours[label],
                        alpha=.65 if label else .22,edgecolors='none')
        threshold=selected[0]['threshold_nats']
        if threshold is not None:
            ax.axhline(threshold,color='#343434',ls='--',lw=1)
        ax.set(xlabel='G − individual rate-transition midpoint',ylabel='Selected-node raw Syn (nats)',
            xlim=(-3.05,3.65))
        # Preserve tolerance-scale negative values if any: do not force a zero limit.
        finite=values[0][np.isfinite(values[0])]
        if len(finite) and finite.min()>=0:
            ax.set_ylim(bottom=0)
        handles=[Line2D([],[],color=co,ls='',marker='o',ms=4,label=la) for co,la in zip(stage_colours,labels)]
        if threshold is not None:
            handles.append(Line2D([],[],color='#343434',ls='--',lw=1,label='G=0 order reference'))
        ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,ncol=3,fontsize=7.5)
        ax.text(-.14,1.04,'e',transform=ax.transAxes,fontweight='bold',fontsize=12)
        output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output,dpi=220,bbox_inches='tight'); plt.close(fig)
    metadata=dict(output=str(output), subject_count=n, summary_sha256=digest(base/(prefix+'_summary.json')),
        plotting_sha256=digest(Path(__file__)), panel_c_e_members=selected[0]['members_one_based'],
        panel_a='Selected maxima by order, all exact sets, counts not independent hypothesis tests',
        panel_b='Top strict preferred candidates; Wilson95% descriptive subject intervals after selection',
        panel_c='All individuals, actual shifted grid, four node states; white=not scanned',
        panel_d='Actual aligned grid, variable coverage, C1 Wilson95% descriptive band; no smoothing',
        panel_e='One raw selected-node Syn per subject/G/seed; missing node remains NaN',
        relative_coverage=relative_coverage, legend_placement='Outside every data axes',
        critical_cell_footprint=[-.3,.3], visual_review='pending')
    atomic_json(base/(prefix+'_figure_metadata.json'),metadata)
    print(output,flush=True)


def plot_seed_extension_combined(base, output):
    """Five-seed presence plate from completed caches; never run simulations."""
    extension = json.loads((base / 'seed_extension_contract.json').read_text())
    c = json.loads((base / 'contract.json').read_text())
    summary = json.loads((base / 'seed5_summary.json').read_text())
    sha, extension_sha = digest(base / 'contract.json'), digest(base / 'seed_extension_contract.json')
    identities = [{key: row[key] for key in ['display_id', 'members', 'members_one_based']}
                  for row in summary['candidates']]
    if (summary['status'] != 'complete' or summary['condition_count'] != 19065
            or summary['subject_count'] != 93 or summary['seeds'] != [3, 4, 5, 6, 7]
            or summary['scientific_kernel_contract_sha256'] != sha
            or summary['extension_contract_sha256'] != extension_sha
            or identities != extension['fixed_candidates']):
        raise ValueError('Incomplete or inconsistent five-seed candidate summary')
    with np.load(base / 'seed5_display.npz') as a:
        subjects, grid, seeds = a['subject_ids'].tolist(), a['G'].copy(), a['seeds'].tolist()
        values, presence = a['raw_syn_nats'].copy(), a['presence'].copy()
        if (str(a['scientific_kernel_contract_sha256']) != sha
                or str(a['extension_contract_sha256']) != extension_sha
                or json.loads(str(a['members_json'])) != [r['members'] for r in extension['fixed_candidates']]):
            raise ValueError('Five-seed display provenance mismatch')
    if (subjects != c['subject_ids'] or not np.array_equal(grid, c['G']) or seeds != [3, 4, 5, 6, 7]
            or values.shape != (3, 93, 41, 5) or presence.shape != values.shape
            or not np.array_equal(presence, np.isfinite(values)) or np.isinf(values).any()):
        raise ValueError('Five-seed display identity/shape/presence mismatch')
    finite = values[np.isfinite(values)]
    tolerance = c['syn_tolerance_nats']
    violations = int((finite < -tolerance).sum())
    if not finite.size or violations:
        raise ArithmeticError(f'Invalid Syn: minimum={finite.min() if finite.size else None}, '
                              f'threshold={-tolerance} nats, affected={violations}')
    original_base = Path(extension['original_result_directory'])
    with np.load(original_base / 'all93_display.npz') as a:
        if not np.array_equal(values[:, :, :, :3], a['raw_syn_nats'][:3], equal_nan=True):
            raise ValueError('Original three-seed values changed')
    midpoints = np.array([c['windows'][p]['transition']['midpoint'] for p in subjects])
    permutation = sorted(range(93), key=lambda pi: midpoints[pi])
    halfwidth = float(np.diff(grid)[0] / 2)
    xlim = (float((grid[0] - midpoints).min() - halfwidth),
            float((grid[-1] - midpoints).max() + halfwidth))
    colours = ['#EBECEE', '#C9D9E2', '#ABC5D3', '#7FA9BD', '#4E87A4', '#236A91']
    counts = presence.sum(3)
    state_counts = {}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
            'axes.spines.top': False, 'axes.spines.right': False,
            'axes.labelsize': 11, 'savefig.facecolor': 'white'}):
        fig, axes = plt.subplots(1, 3, figsize=(14.4, 6.8), sharey=True, layout='constrained')
        cmap, norm = ListedColormap(colours), BoundaryNorm(np.arange(7) - .5, 6)
        for ci, ax in enumerate(axes):
            row = extension['fixed_candidates'][ci]
            state_counts[row['display_id']] = np.bincount(counts[ci].ravel(), minlength=6).tolist()
            for ri, pi in enumerate(permutation):
                edges = np.r_[grid - halfwidth, grid[-1] + halfwidth] - midpoints[pi]
                ax.pcolormesh(edges, [ri - .5, ri + .5], counts[ci, pi][None, :],
                              cmap=cmap, norm=norm, shading='flat', rasterized=True)
            for edge in [-.3, .3]:
                ax.axvline(edge, color='#BA5636', ls='--', lw=1)
            ax.set(xlim=xlim, ylim=(92.5, -.5), xlabel=r'$\Delta G = G - G_i^{*}$')
            ax.set_title(row['display_id'] + ': ROI ' + ', '.join(map(str, row['members_one_based'])),
                         fontsize=10, pad=10)
        positions = list(range(0, 93, 6))
        axes[0].set_yticks(positions, [subjects[permutation[i]].replace('sub-', '') for i in positions])
        axes[0].set_ylabel('Individual SC (93 rows; same order in all panels)')
        handles = [Patch(facecolor=colours[k], edgecolor='#BBBBBB',
                         label=f'{k}/5 seeds' + (' (absent)' if k == 0 else '')) for k in range(6)]
        handles += [Patch(facecolor='white', edgecolor='#BBBBBB', label='Outside scanned G'),
            Line2D([], [], color='#BA5636', ls='--', lw=1, label='Six-point window\n(display cell edges)')]
        axes[-1].legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, .5),
                        title='Presence across seeds 3–7', frameon=False, labelspacing=1.2)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=240, bbox_inches='tight')
        plt.close(fig)
    atomic_json(base / 'seed5_combined_figure_metadata.json', dict(output=str(output), figure_sha256=digest(output),
        summary_sha256=digest(base/'seed5_summary.json'), display_sha256=digest(base/'seed5_display.npz'),
        plotting_sha256=digest(Path(__file__)), scientific_kernel_contract_sha256=sha,
        extension_contract_sha256=extension_sha, subject_count=93, seeds=seeds,
        members_one_based=[r['members_one_based'] for r in extension['fixed_candidates']],
        individual_row_order=[subjects[pi] for pi in permutation], scanned_cell_state_counts=state_counts,
        aggregation='Number of seeds containing the fixed complete natural node per individual/G; no strength reference',
        missing_rule='Gray=0/5 scanned presence; white=outside actual shifted scan',
        native_syn_tolerance_nats=tolerance, tolerance_negative_count=int((finite < 0).sum()),
        significant_negative_count=violations, legend_placement='Outside all data', visual_review='pending'))
    print(output, flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=BASE)
    p.add_argument('--output',type=Path,default=ROOT/'fig/dmf_schaefer100/critical_coalition_all93.png')
    p.add_argument('--development-only',action='store_true')
    p.add_argument('--main-figures',action='store_true',
                   help='Export C1–C3 individual heatmaps and recurrence to --output parent directory')
    p.add_argument('--presence-only',action='store_true',
                   help='Use only ≥2/3 seed natural presence for main figures; retain reference-based exports')
    p.add_argument('--per-seed-figures',action='store_true',
                   help='Also export one C1–C3 all-subject plate per original seed')
    p.add_argument('--combined-seeds-figure',action='store_true',
                   help='Also export one C1–C3 plate counting presence across original seeds 3, 4, 5')
    p.add_argument('--seed-extension-combined',action='store_true',
                   help='Export only the completed five-seed C1–C3 plate from the seed_extension5 cache')
    a=p.parse_args()
    if a.seed_extension_combined:
        if any([a.development_only, a.main_figures, a.presence_only, a.per_seed_figures, a.combined_seeds_figure]):
            p.error('--seed-extension-combined is a separate export mode')
        plot_seed_extension_combined(a.output_dir / 'seed_extension5',
                                     a.output.parent / 'critical_coalition_seed5_combined.png')
        return
    if a.presence_only and not a.main_figures:
        p.error('--presence-only requires --main-figures')
    if a.per_seed_figures and not (a.main_figures and a.presence_only):
        p.error('--per-seed-figures requires --main-figures --presence-only')
    if a.combined_seeds_figure and not (a.main_figures and a.presence_only):
        p.error('--combined-seeds-figure requires --main-figures --presence-only')
    if a.main_figures:
        if a.development_only:
            p.error('Main-report figures require all 93 subjects')
        plot_main_figures(a.output_dir, a.output.parent, a.presence_only, a.per_seed_figures,
                          a.combined_seeds_figure)
    else:
        plot(a.output_dir,a.output,a.development_only)


if __name__=='__main__':
    main()
