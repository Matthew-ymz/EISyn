#!/usr/bin/env python3
"""Individual SC rank distributions and bidirectional SC/SPT comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_dmf_sc_topology_comparison import CANDIDATES, METRIC_NAMES, OUTPUT

FIG = ROOT / 'fig/dmf_schaefer100/sc_topology_comparison'
TITLES = ['Mean internal weight', 'Weighted clique intensity', 'Modularity contribution']
COLOURS = {'C1': '#236A91', 'C2': '#BA5636', 'C3': '#717A38'}
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10,
         'axes.titlesize': 11, 'axes.spines.top': False, 'axes.spines.right': False,
         'savefig.facecolor': 'white', 'pdf.fonttype': 42}


def load(base):
    summary = json.loads((base/'summary.json').read_text())
    if summary['status'] != 'complete' or summary['subject_count'] != 93:
        raise ValueError('Figures require the full paired cohort')
    arrays = {}
    for k in (3, 4):
        with np.load(base/f'order{k}.npz') as a:
            arrays[k] = {key:a[key].copy() for key in a.files}
        if arrays[k]['subject_ids'].tolist() != summary['subject_ids']:
            raise ValueError('Unpaired retained SC ranks')
        if arrays[k]['metric_names'].tolist() != METRIC_NAMES:
            raise ValueError('Metric ordering mismatch')
    return summary, arrays


def candidate_percentiles(arrays):
    result = {}
    for name, members in CANDIDATES.items():
        a = arrays[len(members)]
        found = np.flatnonzero(np.all(a['registered_members'] == members, axis=1))
        if len(found) != 1:
            raise ValueError('Candidate missing or duplicated')
        result[name] = a['registered_percentiles'][:, found[0], :] * 100
    return result


def ecdf_figure(arrays, path):
    p = candidate_percentiles(arrays)
    xmin = 10 ** np.floor(np.log10(min(x.min() for x in p.values())))
    if xmin <= 0:
        raise ValueError('Log axis needs positive midrank percentiles')
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), layout='constrained', sharex=True, sharey=True)
    for mi, ax in enumerate(axes):
        for name, values in p.items():
            x = np.sort(values[:, mi])
            y = np.arange(1, len(x)+1) / len(x) * 100
            ax.step(np.r_[x[0], x], np.r_[0, y], where='post',
                    color=COLOURS[name], lw=1.8,
                    linestyle={'C1':'-', 'C2':'--', 'C3':':' }[name])
        ax.axvline(1, color='#999999', lw=.8, ls='--', zorder=0)
        ax.set_xscale('log')
        ax.set_xlim(xmin, 100)
        ax.set_ylim(0, 103)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel('Individual SC rank (top %; lower is better)')
        ax.set_title(TITLES[mi])
        ax.text(-.09, 1.05, 'abc'[mi], transform=ax.transAxes, fontweight='bold')
        ax.grid(axis='y', color='#E7E7E7', lw=.55)
    axes[0].set_ylabel('Cumulative participants (%)')
    handles = [Line2D([],[], color=COLOURS[name], lw=1.8,
               linestyle={'C1':'-', 'C2':'--', 'C3':':' }[name],
               label=f'{name}: ROI '+', '.join(str(v+1) for v in members))
               for name,members in CANDIDATES.items()]
    handles.append(Line2D([],[], color='#999999', lw=.8, ls='--',label='Top 1% threshold'))
    fig.legend(handles=handles, loc='outside upper center', ncol=2, frameon=False, fontsize=10)
    fig.savefig(path, dpi=260, bbox_inches='tight')
    plt.close(fig)


def relationship_figure(summary, arrays, path):
    all_mean = np.concatenate([a['union_mean_percentiles'].ravel()*100 for a in arrays.values()])
    xmin = 10 ** np.floor(np.log10(all_mean.min()))
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), sharex=True, sharey=True, layout='constrained')
    for row, k in enumerate((3, 4)):
        a = arrays[k]
        registered = a['registered_union_indices']
        for mi, ax in enumerate(axes[row]):
            x = a['union_mean_percentiles'][:, mi] * 100
            y = a['union_counts'][:, 1, 2]
            ax.scatter(x[registered], y[registered], s=12, color='#A4ABB1', alpha=.45,
                       edgecolors='none', zorder=1)
            spt = a['spt_top20_union_indices']
            sc = a['sc_top20_union_indices'][mi]
            ax.scatter(x[spt], y[spt], s=44, facecolors='none', edgecolors='#58508D', lw=1.0,zorder=2)
            ax.scatter(x[sc], y[sc], s=34, marker='D', facecolors='none', edgecolors='#CF7A24',lw=1.0,zorder=3)
            for name, members in CANDIDATES.items():
                if len(members) != k:
                    continue
                found = np.flatnonzero(np.all(a['union_members'] == members, axis=1))[0]
                ax.scatter(x[found], y[found], s=56, color=COLOURS[name], edgecolors='white',lw=.7,zorder=4)
                ax.annotate(name, (x[found], y[found]), xytext=(6,5), textcoords='offset points',
                            color=COLOURS[name], fontsize=10, fontweight='bold')
            ax.set_xscale('log')
            ax.set_xlim(xmin, 100)
            ax.set_ylim(-2.8, 62)
            ax.set_yticks([0, 15, 30, 45, 60])
            ax.grid(axis='y',color='#E7E7E7',lw=.55)
            rho = summary['orders'][str(k)]['correlations'][METRIC_NAMES[mi]]['spearman_rho']
            ax.set_title((TITLES[mi]+'\n' if row == 0 else '')+f'Quality–SPT ρ = {rho:.2f}',fontsize=10)
            ax.text(-.09,1.045,'abcdef'[row*3+mi],transform=ax.transAxes,fontweight='bold')
            if row == 1:
                ax.set_xlabel('Mean individual SC rank (top %; lower is better)')
        axes[row,0].set_ylabel(('Triplets' if k==3 else 'Quartets')+
             '\nParticipants with ≥4/6 window points')
    handles = [Line2D([],[],marker='o',ls='none',color='#A4ABB1',markersize=5,label='Other observed SPT nodes'),
        Line2D([],[],marker='o',ls='none',markerfacecolor='none',markeredgecolor='#58508D',markersize=6,label='SPT top 20'),
        Line2D([],[],marker='D',ls='none',markerfacecolor='none',markeredgecolor='#CF7A24',markersize=5,label='SC top 20')]
    fig.legend(handles=handles,loc='outside upper center',ncol=3,frameon=False,fontsize=10)
    fig.savefig(path,dpi=260,bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=OUTPUT)
    parser.add_argument('--figure-dir',type=Path,default=FIG)
    args = parser.parse_args()
    summary,arrays = load(args.base)
    args.figure_dir.mkdir(parents=True,exist_ok=True)
    with plt.rc_context(STYLE):
        ecdf_figure(arrays,args.figure_dir/'candidate_sc_rank_ecdf.png')
        relationship_figure(summary,arrays,args.figure_dir/'sc_vs_spt_ranking.png')
    print(json.dumps({'figures':[str(args.figure_dir/x) for x in ['candidate_sc_rank_ecdf.png','sc_vs_spt_ranking.png']]}))


if __name__ == '__main__':
    main()
