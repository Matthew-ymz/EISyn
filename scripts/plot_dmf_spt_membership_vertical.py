#!/usr/bin/env python3
"""Stack the two hemispheres using cached native membership counts.

Contract: quantitative data overview; compare member persistence on aligned G
axes, preserving all 100 atlas-ordered names and integer 0..8 colors. Python,
one PNG. Anatomical interpretation inherits the inferred ROI-order limitation.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    with np.load(ROOT/'results/dmf_schaefer100/unconstrained_spt_wide/membership.npz') as a:
        gs, seeds, selected, labels = (a[k] for k in ('G', 'seeds', 'selected', 'labels'))
    if selected.shape != (8, 55, 100) or selected.dtype != bool:
        raise ValueError('Expected the complete cached memberships')
    edges = np.r_[gs[0]-(gs[1]-gs[0])/2, (gs[:-1]+gs[1:])/2,
                  gs[-1]+(gs[-1]-gs[-2])/2]
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = plt.cm.Blues(np.linspace(0, 1, 9)); colors[0] = [1, 1, 1, 1]
    cmap = ListedColormap(colors)
    fig = plt.figure(figsize=(9.2, 16.1))
    for hi in range(2):
        ax = fig.add_axes([.32, .505 if hi == 0 else .07, .65, .415])
        start = hi*50
        mesh = ax.pcolormesh(edges, np.arange(51)-.5, selected.sum(0)[:, start:start+50].T,
                            cmap=cmap, norm=BoundaryNorm(np.arange(-.5, 9.5), 9), rasterized=True)
        names = [str(v).replace('7Networks_LH_', 'L-').replace('7Networks_RH_', 'R-')
                 for v in labels[start:start+50]]
        ax.set(xlim=(edges[0], edges[-1]), ylim=(49.5, -.5), yticks=np.arange(50),
               yticklabels=names, xticks=np.arange(0, 3.01, .25), xlabel='Global coupling G')
        ax.tick_params(axis='y', length=0, labelsize=9, pad=3)
        ax.tick_params(axis='x', labelsize=9)
        if hi == 0:
            ax.set_xlabel('')
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.spines['bottom'].set_visible(False)
        ax.text(0, 1.02, 'a  Left hemisphere' if hi == 0 else 'b  Right hemisphere',
                transform=ax.transAxes, fontsize=11, fontweight='bold')
    cb = fig.colorbar(mesh, cax=fig.add_axes([.32, .964, .65, .011]), orientation='horizontal')
    cb.set_ticks(np.arange(9)); cb.ax.xaxis.set_ticks_position('top')
    cb.set_label('Number of seeds including the ROI (out of 8)', fontsize=9)
    fig.text(.645, .024, '55 G values; 8 seeds; 2,048 samples per condition. '
             'Colors indicate membership frequency, not synergy strength.', ha='center', fontsize=8)
    fig.text(.645, .010, 'C10 selection rule is unchanged. '
             'Atlas label order is inferred; G=0 is finite-sample background.',
             ha='center', fontsize=8, color='#56636d')
    fig.savefig(ROOT/'fig/brain_dmf_spt_roi_membership_vertical.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
