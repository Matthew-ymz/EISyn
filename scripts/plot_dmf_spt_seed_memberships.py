#!/usr/bin/env python3
"""Plot cached C10 memberships separately and raw between-seed differences.

Figure contract: quantitative grid (robustness) with one column per seed and
hemisphere rows; fixed named ROI order and native G coordinates. Companion metric
comparison shows all 28 pairwise symmetric-difference counts and their mean.
One PNG per figure, Python only. No new simulation, core definition, or estimator.
"""
from collections import Counter
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE = '#234f77'


def load_data():
    root = ROOT/'results/dmf_schaefer100/unconstrained_spt_wide'
    with np.load(root/'membership.npz') as a:
        gs, seeds, selected, labels = (a[k] for k in ('G', 'seeds', 'selected', 'labels'))
    if selected.shape != (8, 55, 100) or selected.dtype != bool:
        raise ValueError('Expected complete Boolean memberships: 8 seeds, 55 G, 100 ROI')
    if not np.array_equal(seeds, np.arange(3, 11)) or not np.all(np.diff(gs) > 0):
        raise ValueError('Unexpected seeds or G ordering')
    # Check all 440 plotted sets against the original tree records.
    trees = list((root/'shards').glob('seeds*/trees/seed*.json'))
    if len(trees) != 440:
        raise ValueError(f'Expected 440 tree records; found {len(trees)}')
    seen = set()
    for path in trees:
        r = json.loads(path.read_text())
        si = int(np.flatnonzero(seeds == r['seed'])[0])
        gi = int(np.flatnonzero(np.isclose(gs, r['G']))[0])
        if (si, gi) in seen:
            raise ValueError('Duplicate tree condition')
        seen.add((si, gi))
        if set(np.flatnonzero(selected[si, gi])) != set(r['cores']['10']['members']):
            raise ValueError(f'Membership cache differs from tree: {path.name}')
    return gs, seeds, selected, labels


def membership_figure(gs, seeds, selected, labels, output):
    fig = plt.figure(figsize=(19.2, 14.9))
    # Explicit GridSpec margins reserve space for the shared legend and 100 names.
    grid = fig.add_gridspec(2, 8, left=.165, right=.99, bottom=.075, top=.925,
                           wspace=.13, hspace=.17)
    edges = np.r_[gs[0]-(gs[1]-gs[0])/2, (gs[:-1]+gs[1:])/2,
                  gs[-1]+(gs[-1]-gs[-2])/2]
    cmap = ListedColormap(['white', BLUE])
    for hi in range(2):
        first = hi*50
        names = [str(v).replace('7Networks_LH_', 'L-').replace('7Networks_RH_', 'R-')
                 for v in labels[first:first+50]]
        for si, seed in enumerate(seeds):
            ax = fig.add_subplot(grid[hi, si])
            ax.pcolormesh(edges, np.arange(51)-.5, selected[si, :, first:first+50].T,
                          cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5], 2), rasterized=True)
            ax.set(ylim=(49.5, -.5), xlim=(edges[0], edges[-1]),
                   xticks=[0, 1, 2, 3], yticks=np.arange(50))
            ax.set_yticklabels(names if si == 0 else [])
            ax.tick_params(axis='y', length=0, labelsize=8.3, pad=3)
            ax.tick_params(axis='x', labelsize=8)
            ax.set_xlabel('G', fontsize=9)
            ax.text(.5, 1.022, f'Seed {seed}', transform=ax.transAxes, ha='center',
                    fontsize=10, fontweight='bold')
            if si == 0:
                ax.text(-.025, 1.068, 'a  Left hemisphere' if hi == 0 else 'b  Right hemisphere',
                        transform=ax.transAxes, fontsize=11, fontweight='bold', ha='left')
    fig.legend(handles=[Patch(facecolor=BLUE, label='Included in this seed\'s C10'),
                        Patch(facecolor='white', edgecolor='#a9b4bd', label='Not included')],
               loc='upper center', bbox_to_anchor=(.57, .995), ncol=2, frameon=False, fontsize=10)
    fig.text(.57, .029, 'Same G grid and ROI order in all columns; 2,048 samples per seed; '
             'paired inputs and noise across G within each seed.', ha='center', fontsize=9)
    fig.text(.57, .011, 'C10: follow the child with higher cross-ROI Xi until 2–10 ROI remain. '
             'Atlas label order is inferred.', ha='center', fontsize=8, color='#56636d')
    fig.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def difference_figure(gs, differences, output):
    fig = plt.figure(figsize=(10.8, 5.1))
    ax = fig.add_axes([.09, .19, .85, .64])
    for d in differences:
        ax.plot(gs, d, color='#adb8c0', lw=.6, alpha=.48, marker='.', ms=2.5)
    ax.plot(gs, differences.mean(0), color=BLUE, lw=2, marker='o', ms=3, zorder=3)
    ax.set(xlim=(0, 3), ylim=(-.5, 20.5), xticks=np.arange(0, 3.01, .2),
           yticks=[0, 5, 10, 15, 20], xlabel='Global coupling G',
           ylabel='Number of differing ROI (entering + leaving)')
    ax.tick_params(labelsize=9)
    ax.legend(handles=[Line2D([], [], color='#adb8c0', lw=.8, marker='.',
                              label='Each of 28 seed pairs'),
                       Line2D([], [], color=BLUE, lw=2, marker='o', ms=3,
                              label='Mean across seed pairs')],
              loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=2, frameon=False)
    fig.text(.52, .072, 'At each G, compare the actual C10 sets from every pair of seeds. '
             'One replacement counts as two ROI differences.', ha='center', fontsize=9)
    fig.text(.52, .027, '0 = identical sets. Lines connect sampled G values; '
             'pairwise observations share seeds and are not independent replicates.',
             ha='center', fontsize=8, color='#56636d')
    fig.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .8})
    gs, seeds, selected, labels = load_data()
    pairs = list(itertools.combinations(range(len(seeds)), 2))
    differences = np.array([np.sum(selected[a] != selected[b], axis=1) for a, b in pairs])
    out = ROOT/'fig'
    out.mkdir(exist_ok=True)
    membership_figure(gs, seeds, selected, labels, out/'brain_dmf_spt_roi_by_seed.png')
    difference_figure(gs, differences, out/'brain_dmf_spt_between_seed_differences.png')
    for gi, g in enumerate(gs):
        groups = Counter(tuple(np.flatnonzero(x)) for x in selected[:, gi])
        print(f'G={g:.2f}: mean={differences[:,gi].mean():.3f}, '
              f'range={differences[:,gi].min()}..{differences[:,gi].max()}, '
              f'common={selected[:,gi].all(0).sum()}, '
              f'group_counts={sorted(groups.values(), reverse=True)}')
    print('Verified all 440 memberships against original trees; saved two PNGs.')


if __name__ == '__main__':
    main()
