#!/usr/bin/env python3
"""Display the per-G intersection of all eight cached C10 memberships.

Figure contract: show where all seeds retain the same named ROI; two quantitative
membership panels, native G coordinates, one PNG. An empty column means no unanimous
member, not an absent individual-seed core. Preserve atlas order and skip rows that
never meet the unanimity criterion. No simulations, smoothing, or new core search.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    with np.load(ROOT/'results/dmf_schaefer100/unconstrained_spt_wide/membership.npz') as a:
        gs, seeds, selected, labels = (a[k] for k in ('G', 'seeds', 'selected', 'labels'))
    if selected.shape != (8, 55, 100) or not np.array_equal(seeds, np.arange(3, 11)):
        raise ValueError('Expected the complete 55-G, eight-seed, 100-ROI scan')
    unanimous = selected.all(axis=0)
    edges = np.r_[gs[0]-(gs[1]-gs[0])/2, (gs[:-1]+gs[1:])/2,
                  gs[-1]+(gs[-1]-gs[-2])/2]
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': .8})
    fig = plt.figure(figsize=(12.8, 6.5))
    cmap = ListedColormap(['white', '#234f77'])
    for panel, (start, stop, title) in enumerate(((0, 50, 'Left hemisphere'),
                                                (50, 100, 'Right hemisphere'))):
        rows = np.flatnonzero(unanimous[:, start:stop].any(axis=0)) + start
        ax = fig.add_axes([.18 if panel == 0 else .68, .16, .29, .72])
        ax.pcolormesh(edges, np.arange(len(rows)+1)-.5, unanimous[:, rows].T,
                      cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5], 2), rasterized=True)
        names = [str(labels[r]).replace('7Networks_LH_', 'L-').replace('7Networks_RH_', 'R-')
                 for r in rows]
        ax.set(ylim=(len(rows)-.5, -.5), xlim=(edges[0], edges[-1]),
               yticks=np.arange(len(rows)), yticklabels=names,
               xticks=np.arange(0, 3.01, .5), xlabel='Global coupling G')
        ax.tick_params(axis='y', length=0, labelsize=8.5, pad=3)
        ax.text(0, 1.035, f'{"ab"[panel]}  {title}', transform=ax.transAxes,
                fontweight='bold', fontsize=11)
    fig.legend(handles=[Patch(facecolor='#234f77', label='Included in all 8 seeds'),
                        Patch(facecolor='white', edgecolor='#a9b4bd', label='Not unanimous')],
               loc='upper center', bbox_to_anchor=(.57, .995), ncol=2, frameon=False, fontsize=10)
    fig.text(.57, .055, 'At each G: intersection of eight C10 sets (2,048 samples per seed).',
             ha='center', fontsize=9)
    fig.text(.57, .024, 'Only ROI unanimous at least once are shown; atlas label order is inferred.',
             ha='center', fontsize=8, color='#56636d')
    output = ROOT/'fig/brain_dmf_spt_roi_unanimous.png'
    fig.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'{output}: {unanimous.any(0).sum()} ROI; '
          f'{np.sum(unanimous.sum(1)==0)}/{len(gs)} G values have no unanimous member')


if __name__ == '__main__':
    main()
