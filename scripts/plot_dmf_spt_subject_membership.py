#!/usr/bin/env python3
"""Cache-only Fig. 2 preview: seed repeatability versus subject membership.

No simulations or tree searches. Each individual contributes one actual C10
from the fixed main seed (4); seeds 3 and 5 are sensitivity summaries. The
matched group-mean reference uses the same affine-TM pilot and search rules.
Columns are categorical sampled G values, with no interpolation. G=0 is
excluded because shared random samples generate spurious common membership.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'results/dmf_schaefer100/subject_consistency_pilot'
LEGACY = ROOT / 'results/dmf_schaefer100/unconstrained_spt_wide/membership.npz'
OUTPUT = ROOT / 'fig/brain_dmf_spt_subject_membership_preview.png'
GS = (1., 1.3, 2.2)
MAIN_SEED = 4


def describe(memberships, labels):
    counts = memberships.sum(axis=0)
    cores = [tuple(np.flatnonzero(row).tolist()) for row in memberships]
    pairs = [len(set(a) & set(b)) / len(set(a) | set(b))
             for a, b in combinations(cores, 2)]
    maximum = int(counts.max())
    return dict(replicate_count=len(cores), core_sizes=[len(c) for c in cores],
                unique_complete_cores=len(set(cores)),
                most_frequent_complete_core_count=max(Counter(cores).values()),
                maximum_roi_count=maximum,
                most_frequent_rois=[str(labels[i]) for i in np.flatnonzero(counts == maximum)],
                all_replicate_common_rois=[str(labels[i]) for i in np.flatnonzero(counts == len(cores))],
                mean_pairwise_jaccard=float(np.mean(pairs)),
                pairwise_comparisons=len(pairs),
                inference='descriptive; overlapping pairs are not independent replicates')


def load_data():
    contract = json.loads((PILOT / 'contract.json').read_text())
    subjects = [s for s in contract['subject_ids'] if s != 'group_mean_93']
    seeds = contract['seeds']
    if len(subjects) != 8 or seeds != [3, 4, 5]:
        raise ValueError('Expected the frozen eight-subject development pilot')
    tolerance = float(contract['syn_tolerance_nats'])
    if tolerance != 1e-8:
        raise ValueError('Unexpected native Syn tolerance')
    with np.load(PILOT / 'inputs.npz') as a:
        labels = a['labels'].copy()
        if json.loads(str(a['contract_json'])) != contract:
            raise ValueError('Input cache contract mismatch')
    for subject in subjects:
        path = ROOT / 'data/neuromodulator_receptor_sc_100/CON_SC_1mio' / (subject + '.csv')
        if hashlib.sha256(path.read_bytes()).hexdigest() != contract['subject_hashes'][subject]:
            raise ValueError(f'SC input changed: {subject}')
    with np.load(LEGACY) as a:
        if not np.array_equal(labels, a['labels']):
            raise ValueError('ROI order differs between caches')
        if a['selected'].shape != (8, 55, 100) or a['selected'].dtype != bool:
            raise ValueError('Incomplete original Fig. 2 cache')
        indices = [int(np.flatnonzero(np.isclose(a['G'], g))[0]) for g in GS]
        original = a['selected'][:, indices].copy()
    individuals = np.zeros((len(subjects), len(seeds), len(GS), 100), dtype=bool)
    matched_mean = np.zeros((len(seeds), len(GS), 100), dtype=bool)
    minimum, negative_count, node_count = np.inf, 0, 0
    for si, subject in enumerate(subjects + ['group_mean_93']):
        for ri, seed in enumerate(seeds):
            for gi, g in enumerate(GS):
                path = PILOT / 'organization' / f'{subject}_G{g:.2f}_seed{seed}.npz'
                with np.load(path) as a:
                    if json.loads(str(a['contract_json'])) != contract:
                        raise ValueError(f'Organization cache contract mismatch: {path.name}')
                    tree = json.loads(str(a['trees_json']))['multisource']
                core = tree['core10']
                if not 2 <= len(core) <= 10 or len(set(core)) != len(core):
                    raise ValueError(f'Invalid/nonexistent C10: {path.name}')
                if any(i < 0 or i >= 100 for i in core):
                    raise ValueError(f'Invalid ROI: {path.name}')
                node = tree['tree']
                while len(node['members']) > 10 and node['children']:
                    node = sorted(node['children'],
                                  key=lambda child: (-child['value_nats'], tuple(child['members'])))[0]
                if core != node['members']:
                    raise ValueError(f'C10 is not the retained actual tree node: {path.name}')
                values = []

                def visit(node):
                    if node['children']:
                        values.append(float(node['syn_nats_raw']))
                    for child in node['children']:
                        visit(child)

                visit(tree['tree'])
                values = np.asarray(values)
                if len(values) != 99 or not np.isfinite(values).all():
                    raise ArithmeticError(f'Incomplete or nonfinite tree: {path.name}')
                affected = int(np.sum(values < -tolerance))
                if affected:
                    raise ArithmeticError(f'Syn violation: minimum={values.min()} nats; '
                                          f'threshold={-tolerance}; affected_count={affected}; {path.name}')
                minimum = min(minimum, float(values.min()))
                negative_count += int(np.sum((values < 0) & (values >= -tolerance)))
                node_count += len(values)
                if abs(tree['closure_error_nats']) > tolerance:
                    raise ArithmeticError(f'Tree closure failure: {path.name}')
                target = matched_mean[ri, gi] if subject == 'group_mean_93' else individuals[si, ri, gi]
                target[core] = True
    fixed_seed = individuals[:, seeds.index(MAIN_SEED)]
    summary = dict(
        status='cache-only eight-subject preview; all 93 subjects not computed',
        G=list(GS), subject_ids=subjects, main_seed=MAIN_SEED, sensitivity_seeds=seeds,
        subject_selection=contract['subject_selection'],
        aggregation='one actual C10 per subject at fixed seed 4; one vote per subject',
        exclusions={'G=0': 'shared initial states/noise produce common finite-sample artifacts'},
        display='equal-width sampled G columns, common membership percentage scale; no interpolation',
        roi_label_status=contract['roi_label_status'],
        method_comparison={'original': 'empirical Gaussian conditional covariance, 8 seeds (3-10)',
                           'matched_mean_and_individuals': contract['estimator'],
                           'SPT': contract['SPT'],
                           'JFIC': contract['JFIC'],
                           'shared_random_samples': contract['paired_inputs_and_noise_across_G_and_subjects'],
                           'interpretation': 'fixed-reference-model development sample, no population inference'},
        manuscript={'parent_key': 'P6UJCVG8', 'attachment_key': 'DXGC7JEA',
                    'checked_on': '2026-10-02', 'fulltext_pages': 19,
                    'locations': 'Brain pp.6-7; Methods Eqs.4-12 pp.15-17',
                    'version_date': 'not specified; metadata dates do not establish manuscript version',
                    'appendices': 'not available in the current Zotero item',
                    'scope_difference': 'manuscript first level uses Yeo-7; this preview uses unconstrained ROI SPT'},
        nonnegative_audit={'tolerance_nats': tolerance, 'minimum_selected_syn_nats': minimum,
                           'tolerance_negative_count': negative_count, 'violation_count': 0,
                           'selected_node_count': node_count,
                           'handling': 'raw cached Syn retained; no clipping/projection'},
        comparisons={}, sensitivity={}, figure=str(OUTPUT.relative_to(ROOT)))
    for gi, g in enumerate(GS):
        summary['comparisons'][str(g)] = dict(
            original_mean_sc_seeds=describe(original[:, gi], labels),
            matched_mean_sc_seeds=describe(matched_mean[:, gi], labels),
            individual_sc_subjects=describe(fixed_seed[:, gi], labels))
        summary['sensitivity'][str(g)] = {
            str(seed): describe(individuals[:, ri, gi], labels) for ri, seed in enumerate(seeds)}
    return labels, [original, matched_mean, fixed_seed], summary


def plot(labels, data):
    colors = plt.cm.Blues(np.linspace(0, 1, 256))
    colors[0] = (1, 1, 1, 1)
    cmap = ListedColormap(colors)
    norm = Normalize(0, 100)
    titles = [('a  Original Fig. 2', 'Mean SC · 8 seeds'),
              ('b  Matched-method reference', 'Mean SC · 3 seeds'),
              ('c  Across subjects', '8 individual SC · fixed seed 4')]
    style = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
             'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(10.4, 15.4), sharey='row',
                                 gridspec_kw={'wspace': .12, 'hspace': .13})
        fig.subplots_adjust(left=.30, right=.98, bottom=.065, top=.895)
        for ci, array in enumerate(data):
            counts = array.sum(axis=0)
            for hi in range(2):
                ax = axes[hi, ci]
                values = counts[:, hi*50:(hi+1)*50].T
                mesh = ax.pcolormesh(np.arange(4)-.5, np.arange(51)-.5,
                                     100*values/len(array), cmap=cmap, norm=norm,
                                     edgecolors='#e7ebef', linewidth=.23)
                names = [str(s).replace('7Networks_LH_', 'L-').replace('7Networks_RH_', 'R-')
                         for s in labels[hi*50:(hi+1)*50]]
                ax.set(xlim=(-.5, 2.5), ylim=(49.5, -.5),
                       yticks=np.arange(50), yticklabels=names,
                       xticks=np.arange(3), xticklabels=['1.0', '1.3', '2.2'],
                       xlabel='Sampled coupling G')
                ax.tick_params(axis='y', length=0, labelsize=8.5, pad=4)
                ax.tick_params(axis='x', length=0, pad=5)
                for row, col in zip(*np.nonzero(values)):
                    frequency = values[row, col] / len(array)
                    ax.text(col, row, f'{values[row, col]}/{len(array)}', ha='center', va='center',
                            fontsize=8, color='white' if frequency >= .7 else '#203846')
                if hi == 0:
                    ax.text(0, 1.09, titles[ci][0], transform=ax.transAxes,
                            fontsize=10, fontweight='bold')
                    ax.text(0, 1.055, titles[ci][1], transform=ax.transAxes, fontsize=9)
                if ci == 0:
                    ax.text(-.01, 1.015, 'Left hemisphere' if hi == 0 else 'Right hemisphere',
                            transform=ax.transAxes, fontsize=9, color='#53616c')
        cb = fig.colorbar(mesh, cax=fig.add_axes([.30, .97, .68, .009]), orientation='horizontal')
        cb.set_ticks([0, 25, 50, 75, 100])
        cb.ax.xaxis.set_ticks_position('top')
        cb.set_label('ROI membership frequency (%) · cell labels give actual included / total', fontsize=9)
        fig.text(.64, .027, 'Same ROI order and sampled G; no interpolation. C10 = retained SPT branch with 2–10 ROI.',
                 ha='center', fontsize=8)
        fig.text(.64, .013, 'Eight-subject development preview. G=0 excluded; atlas label order inferred.',
                 ha='center', fontsize=8, color='#53616c')
        OUTPUT.parent.mkdir(exist_ok=True)
        fig.savefig(OUTPUT, dpi=240, bbox_inches='tight', facecolor='white')
        plt.close(fig)


def main():
    labels, data, summary = load_data()
    plot(labels, data)
    path = PILOT / 'subject_membership_preview.json'
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    for g, rows in summary['comparisons'].items():
        print(g, {key: {'max_roi_count': row['maximum_roi_count'],
                        'unique_cores': row['unique_complete_cores'],
                        'mean_pairwise_jaccard': round(row['mean_pairwise_jaccard'], 4)}
                  for key, row in rows.items()})
    print(summary['nonnegative_audit'])
    print(OUTPUT)


if __name__ == '__main__':
    main()
