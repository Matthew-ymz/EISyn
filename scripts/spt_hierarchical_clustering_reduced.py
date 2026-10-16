"""Three literature-defined hierarchy baselines, without parameter searches.

Ward and UPGMA use recorded numeric source profiles directly. Song et al. (2012)
uses its published AUV2 tree-building stage and author-software defaults. This is
a Python translation, not a claim to have executed the original R environment or
reproduced its downstream dynamic tree cutting / GO enrichment application.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_hierarchical_clustering_screen import OUT, POOLS, hierarchy_from_linkage, save, score
from scripts.spt_kuramoto_finite_tm import digest
from scripts.spt_kuramoto_natural_mlp import chain_partitions

METHODS = ('ward', 'average', 'song2012_auv2')
LABELS = ('Ward', 'Average / UPGMA', 'MI + average (Song 2012)')
SUMMARY = OUT / 'paper_summary.json'
PROTOCOL = OUT / 'paper_protocol.json'
FIGURE = ROOT / 'docs/reports/assets/kuramoto_hierarchy/spt_hierarchical_clustering_reduced.png'


def equal_width(x, bins):
    """infotheo discEW: [left,right), with maximum in the last bin (zero based)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or bins < 2:
        raise ValueError('Finite observation-by-variable matrix and at least two bins required')
    result = np.empty(x.shape, dtype=np.int64)
    for i in range(x.shape[1]):
        low, high = x[:, i].min(), x[:, i].max()
        if low == high:
            result[:, i] = 0
            continue
        # Use the actual source boundaries, rather than floor((x-min)/width),
        # whose rounding at exact bin boundaries can disagree with discEW.
        boundaries = low + np.arange(1, bins) * ((high - low) / bins)
        result[:, i] = np.searchsorted(boundaries, x[:, i], side='right')
    return result


def entropy_mm(counts):
    """Miller-Madow in nats, using occupied cells as in author dependencies."""
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    n = counts.sum()
    return float(np.log(n) - np.dot(counts, np.log(counts)) / n + (len(counts) - 1) / (2 * n))


def song_auv2(x):
    """Published equal-width -> MM -> AUV2 -> 1-adjacency pipeline.

    Sources: Song et al. 2012 Eqs(10),(16),(17), Supplement S1 Eq(10);
    WGCNA::mutualInfoAdjacency; infotheo::discEW / entropy_miller_madow.
    Negative MI handling follows the author's explicit software rule. It is
    recorded separately from raw MI and has no connection to PEID Syn.
    """
    n, p = x.shape
    bins = int(np.sqrt(n))  # original author's as.integer(sqrt(nrow(datE)))
    discrete = equal_width(x, bins)
    h = np.array([entropy_mm(np.bincount(discrete[:, i], minlength=bins)) for i in range(p)])
    if np.any(h <= 0):
        raise ValueError('AUV2 is undefined for zero-entropy profiles')
    raw = np.diag(h)
    joint_occupied = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(i):
            counts = np.bincount(discrete[:, i] * bins + discrete[:, j], minlength=bins * bins)
            raw[i, j] = raw[j, i] = h[i] + h[j] - entropy_mm(counts)
            joint_occupied[i, j] = joint_occupied[j, i] = np.count_nonzero(counts)
    mi = raw.copy()
    mi[mi < 0] = 0  # explicit WGCNA/minet rule; raw estimates retained below
    adjacency = mi / np.maximum(h[:, None], h[None, :])
    distance = 1 - adjacency
    np.fill_diagonal(distance, 0)
    if not np.isfinite(distance).all() or distance.min() < 0:
        raise ArithmeticError('Author formula produced an invalid dissimilarity')
    upper = raw[np.triu_indices(p, 1)]
    return distance, dict(bins=bins, samples=n, entropy_nats=h.tolist(), raw_mi_nats=raw.tolist(),
                         adjacency=adjacency.tolist(), distance=distance.tolist(),
                         negative_raw_mi_pairs=int(np.count_nonzero(upper < 0)),
                         minimum_raw_mi_nats=float(upper.min()),
                         negative_mi_rule='set to zero as explicitly specified in WGCNA/minet source; raw retained',
                         occupied_joint_cells=joint_occupied.tolist())


def run():
    started = time.monotonic()
    protocol = dict(
        date='2026-10-16', methods=list(METHODS), labels=list(LABELS), selection_uses_f1=False,
        selection='Two long-used generic HAC representatives, plus one published nonlinear-dependence tree; not global usage ranks',
        usage_priority=[['ward', 'average'], ['song2012_auv2']],
        input='Recorded numeric source profiles x.T; no extra centering, scaling, sin/cos, rank copula, lag mixing or kernel',
        input_scope='All methods see the same 48 training chains; only source states are used. MLP additionally learns paired futures.',
        domain_limits='Kuramoto wrapped phases are treated as recorded numeric values; these generic methods are not circular algorithms',
        ward='SciPy ward on Euclidean profile vectors; original minimum incremental within-cluster SSE criterion',
        average='SciPy average on Euclidean profile distances; mean over all cross-cluster original pairs (UPGMA)',
        song2012='floor(sqrt(m)) equal-width bins; occupied-cell Miller-Madow; author negative-MI rule; I/max(Hx,Hy); distance 1-A; average linkage',
        reproduction_scope='Published tree-building definitions; Python translation of Song author software; no original R runtime parity claim',
        excluded_downstream='Song dynamicTreeCut produces a flat partition after the dendrogram; not part of the full-hierarchy metric here',
        parameter_search=False, preprocessing_search=False, new_simulations=0, new_mlp_fits=0, new_ei_or_syn_estimates=0,
        data_partition_seed=20261016, chain_subset_count=4,
        inference='One trajectory pool per case; four disjoint chain quarters are sensitivity checks, not independent system replicates',
        score='Frozen module F1; strict clade F1 also retained; primitive-internal refinements ignored equally',
        sources={
            'Ward 1963': 'https://doi.org/10.1080/01621459.1963.10500845',
            'Ward definition audit 2014': 'https://arxiv.org/html/1111.6285v2',
            'UPGMA 1958': 'https://www.biodiversitylibrary.org/part/385476',
            'SciPy used version': 'https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.cluster.hierarchy.linkage.html',
            'Song 2012': 'https://doi.org/10.1186/1471-2105-13-328',
            'Song Supplement 1': 'https://media.springernature.com/original/springer-static/esm/art:10.1186%2F1471-2105-13-328/MediaObjects/12859_2012_5682_MOESM1_ESM.pdf',
            'Author MI software': 'https://raw.githubusercontent.com/cran/WGCNA/master/R/mutualInfoAdjacency.R',
            'Author discretization dependency': 'https://raw.githubusercontent.com/cran/infotheo/master/src/discretize.cpp',
            'Author entropy dependency': 'https://raw.githubusercontent.com/cran/infotheo/master/src/entropy.cpp',
            'Author MI dependency': 'https://raw.githubusercontent.com/bioc/minet/devel/R/build.mim.R'},
        manuscript=dict(parent='P6UJCVG8', main='DXGC7JEA', si='MWIWKSVG', checked='2026-10-16',
                        version='sole main/SI, no explicit manuscript date', locations='Methods(10)-(12); SI S5.1(S49)-(S50)'))
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != protocol:
        raise ValueError('Do not modify a previously scored protocol')
    save(PROTOCOL, protocol)  # fixed methods, parameters, input BEFORE new scores
    original = json.loads((OUT / 'summary.json').read_text())
    result = dict(status='running', protocol=protocol, records=[],
                  exploratory_summary_sha256=hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest())
    for (display, native, circular, filename), old in zip(POOLS, original['records']):
        with np.load(ROOT / filename, allow_pickle=False) as pool:
            states = pool['states'].copy()
        _, manifest = chain_partitions(states)
        if display != old['case'] or manifest['splits'] != old['shared_data']['splits']:
            raise ValueError('Input differs from the original screen')
        train = manifest['splits']['train']['chains']
        record = dict(case=display, native_case=native, pool=filename, shared_data=manifest,
                      pool_sha256=hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(), batches=[])
        for batch_id, chains in enumerate([train] + [train[i*12:(i+1)*12] for i in range(4)]):
            x, y = (a.reshape(-1, 9) for a in (states[:-1, chains], states[1:, chains]))
            if digest(x, y) != old['batches'][batch_id]['paired_arrays_sha256']:
                raise ValueError('Not the identical registered training pairs')
            distances = pdist(x.T, metric='euclidean')
            batch = dict(chains=chains, pairs=len(x), paired_arrays_sha256=digest(x, y), methods={})
            for key in METHODS:
                tick = time.monotonic()
                if key == 'song2012_auv2':
                    d, audit = song_auv2(x)
                    z = linkage(squareform(d, checks=True), method='average')
                else:
                    z = linkage(distances, method=key)
                    audit = dict(distance='Euclidean', source='untransformed numeric training source profiles')
                batch['methods'][key] = dict(status='complete', metrics=score(hierarchy_from_linkage(z), native),
                                             audit=audit, elapsed_seconds=time.monotonic()-tick)
            record['batches'].append(batch)
        result['records'].append(record)
    result.update(status='complete', elapsed_seconds=time.monotonic()-started)
    save(SUMMARY, result)
    return result


def plot(data):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch
    from scripts.plot_spt_matched_cases_composite import draw_tree, verify_tree_text
    from scripts.plot_spt_hierarchical_clustering_screen import COLORS
    cmap = LinearSegmentedColormap.from_list('recovery', COLORS)
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'svg.fonttype': 'none'}):
        fig = plt.figure(figsize=(12.5, 8.5), dpi=240)
        grid = fig.add_gridspec(3, 4, left=.095, right=.985, bottom=.17, top=.86,
                               width_ratios=[1, 1, 1, 1.18], wspace=.28, hspace=.30)
        axes = []
        for row, case in enumerate(data['records']):
            for col, key in enumerate(METHODS):
                ax = fig.add_subplot(grid[row, col]); axes.append(ax)
                record = case['batches'][0]['methods'][key]
                draw_tree(ax, record, case['native_case'])
                full = record['metrics']['module_f1']
                subset = [b['methods'][key]['metrics']['module_f1'] for b in case['batches'][1:]]
                ax.text(.5, -.08, f'F1 {full:.2f}   |   subsets {min(subset):.2f}–{max(subset):.2f}',
                        ha='center', va='top', fontsize=7.5, color='#596771', transform=ax.transAxes)
                if row == 0:
                    ax.set_title(LABELS[col], fontsize=10, pad=22)
            box = axes[-3].get_position()
            fig.text(.025, (box.y0+box.y1)/2, case['case'].replace(' ', '\n', 1), va='center', ha='center', fontsize=10)
        heat = fig.add_subplot(grid[:, 3])
        vals = np.array([[r['batches'][0]['methods'][k]['metrics']['module_f1'] for r in data['records']] for k in METHODS])
        heat.imshow(vals, vmin=0, vmax=1, cmap=cmap, aspect=1.8)
        for i in range(3):
            for j in range(3):
                heat.text(j, i, f'{vals[i,j]:.2f}', ha='center', va='center', fontsize=10,
                          color='white' if vals[i,j] > .7 else '#26333D')
        heat.set_yticks(range(3), ['Ward', 'Average', 'MI + average'], fontsize=8.5)
        heat.yaxis.tick_right()
        heat.set_xticks(range(3), ['Kuramoto\nA', 'HO-LV\nA', 'Kuramoto\nB'], fontsize=8.5)
        heat.xaxis.tick_top(); heat.tick_params(length=0, pad=7)
        heat.set_xlabel('Module F1 (0–1)', labelpad=12)
        for spine in heat.spines.values():
            spine.set_visible(False)
        fig.text(.53, .958, 'Three fixed literature-defined hierarchy baselines', ha='center', fontsize=12, weight='bold')
        fig.text(.53, .917, 'Identical natural training rows; recorded numeric states; no outcome-based parameter selection', ha='center', fontsize=9, color='#596771')
        fig.legend(handles=[Patch(fc='#E5F1EB', ec='#217B68', label='Reference module'),
                            Patch(fc='#FCEFEA', ec='#B74738', label='Wrong clade'),
                            Patch(fc='white', ec='#B7C0C6', label='Leaf / unscored refinement')],
                   loc='lower center', bbox_to_anchor=(.53, .09), ncol=3, frameon=False, fontsize=9)
        fig.text(.53, .060, 'Subsets: four disjoint training-chain quarters, not confidence intervals. Trees show topology, not original merge heights.',
                 ha='center', fontsize=8, color='#596771')
        fig.text(.53, .031, 'Generic numeric-input algorithms: phase wrapping is a limitation. Song: published dendrogram stage; no dynamic tree cut.',
                 ha='center', fontsize=8, color='#596771')
        fig.canvas.draw()
        for ax in axes:
            verify_tree_text(ax, fig.canvas.get_renderer())
        fig.savefig(FIGURE, bbox_inches='tight', pad_inches=.12, metadata={
            'Description': 'Reduced paper-defined baselines, all fixed cases; ' + hashlib.sha256(SUMMARY.read_bytes()).hexdigest()})
        fig.savefig(FIGURE.with_suffix('.svg'), bbox_inches='tight', pad_inches=.12)
        plt.close(fig)
    return FIGURE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    data = json.loads(SUMMARY.read_text()) if args.plot_only else run()
    print(plot(data))
    for record in data['records']:
        print(record['case'], {k: record['batches'][0]['methods'][k]['metrics']['module_f1'] for k in METHODS})
    print('compute seconds', data['elapsed_seconds'])


if __name__ == '__main__':
    main()
