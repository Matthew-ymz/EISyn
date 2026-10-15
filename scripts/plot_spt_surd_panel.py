"""Revise the supplied phase-benchmark panel: remove HAC, add SURD weight tree.

Plot-only reuse of the screenshot's Gaussian A / phase B,C records and SURD.
The completed comparison uses the same 20/240/60 labelings/variants for every
method. An earlier three-case input remains supported with a scope footnote.
No simulations, estimator fits, new SURD integration, or changed tree rules.

Zotero freshly rechecked 2026-10-15: P6UJCVG8, sole main DXGC7JEA Methods
(10)-(12), sole SI MWIWKSVG S5/Algorithm S1. No explicit draft date/version;
the existing manuscript-version ambiguity remains. Only layout is revised.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.compose_spt_framework_comparison import (
    GREEN, INK, current_tree, load_current_comparison,
)
from scripts.plot_spt_heterogeneous_hierarchy import (
    FILLS, PAIR, HIGHER, network, verify_export_clearance,
)

METHODS = ('spt','surd_maximum_weight','block_phi_r','normalized_block_phi_r')
NAMES = ('SPT','SURD +\nmax-weight tree',r'Block $\Phi_R$','Normalized\n'+r'Block $\Phi_R$')
OUTPUT = ROOT/'docs/reports/assets/kuramoto_hierarchy/spt_hierarchy_surd_comparison.png'
FULL_INPUT = ROOT/'results/spt_surd_full_benchmark/summary.json'


def load(surd_path):
    records, representatives = load_current_comparison(finite=False)
    data = json.loads(surd_path.read_text())
    full = data.get('status')=='complete'
    surd_records = data['records']
    if full:
        expected = {(r['case'],r['variant'],r['permutation_id']) for r in records}
        found = {(r['case'],r['variant'],r['permutation_id']) for r in surd_records}
        if len(surd_records)!=320 or expected!=found:
            raise ValueError('The full SURD comparison must match every baseline condition')
    elif len(surd_records)!=3 or {r['case'] for r in surd_records}!=set('ABC'):
        raise ValueError('Expected the full comparison or the earlier three identity cases')
    if data['protocol']['source']!='all nine original node blocks' or data['protocol']['clipping']:
        raise ValueError('Unexpected SURD input protocol')
    outputs = []
    for r in representatives:
        s = next(s for s in surd_records if s['case']==r['case']
                 and (not full or (s['variant']==0 and s['permutation_id']==0)))
        if {tuple(e) for e in r['hyperedges']}!={tuple(e) for e in s['hyperedges']}:
            raise ValueError('SURD architecture does not match the screenshot')
        a = s['surd_audit']
        if a['specific_significant_violation_count'] or a['maximum_statewise_closure_error_nats']>1e-8:
            raise ArithmeticError('Invalid SURD decomposition cannot be plotted')
        outputs.append({**r,'methods':{**r['methods'],
                       'surd_maximum_weight':s['methods']['surd_maximum_weight']}})
    scores = np.array([[np.mean([x['methods'][method]['clade_f1']
                                for x in (surd_records if method=='surd_maximum_weight' else records)
                                if x['case']==r['case']])
                        for r in outputs] for method in METHODS])
    return records,outputs,scores,full


def plot(surd_path,output):
    records,representatives,scores,full = load(surd_path)
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':10,'text.color':INK,'savefig.facecolor':'white'}):
        fig = plt.figure(figsize=(18,10.6),facecolor='white',dpi=300)
        fig.text(.019,.979,'c',weight='bold',fontsize=13,va='top')
        fig.legend(handles=[Patch(fc=FILLS[2],ec=PAIR,label='Pairwise'),
                            Patch(fc=FILLS[3],ec=HIGHER,label='Higher-order'),
                            Line2D([],[],color='#8982B4',ls='--',label='Global')],
                   loc='upper center',bbox_to_anchor=(.51,.996),ncol=3,frameon=False,
                   fontsize=10,handlelength=1.2,columnspacing=1.3)
        networks = []
        grid = fig.add_gridspec(1,3,left=.048,right=.985,top=.936,bottom=.682,wspace=.10)
        for i,record in enumerate(representatives):
            ax = fig.add_subplot(grid[i])
            supports = record['hyperedges']
            primitive = [e for e in supports if not any(set(g)<set(e) for g in supports)]
            network(ax,{'case':record['case'],'hyperedges':supports,'planted_bottom':primitive},
                    operator_symbol='H',node_size=300)
            label = ('Mixed branches','Deep cascade','Triadic modules')[i]
            ax.text(.5,1.01,f"{record['case']}  {label}",ha='center',va='bottom',
                    transform=ax.transAxes,fontsize=12,weight='bold')
            networks.append(ax)
        grid = fig.add_gridspec(3,4,left=.052,right=.716,top=.622,bottom=.109,
                                hspace=.20,wspace=.20)
        trees = []
        for row,record in enumerate(representatives):
            for col,(method,name) in enumerate(zip(METHODS,NAMES)):
                ax = fig.add_subplot(grid[row,col])
                current_tree(ax,record,method)
                trees.append(ax)
                if row==0:
                    ax.set_title(name,fontsize=11,pad=9,color=GREEN if method=='spt' else INK,
                                 weight='bold' if method=='spt' else 'normal')
                if col==0:
                    ax.text(-.15,.5,f"Case {record['case']}",ha='right',va='center',rotation=90,
                            transform=ax.transAxes,fontsize=10.6,color='#57636D')
        score_ax = fig.add_axes((.862,.265,.119,.253))
        cmap = LinearSegmentedColormap.from_list('recovery',['#FCF5F0','#C5DECF','#1D7967'])
        score_ax.imshow(scores,cmap=cmap,vmin=0,vmax=1,aspect='auto',interpolation='nearest')
        score_ax.set_xticks(range(3),list('ABC'))
        score_ax.set_yticks(range(len(METHODS)),
                           ('SPT','SURD +\nmax-weight tree'+('' if full else r'$\dagger$'),
                            r'Block $\Phi_R$','Normalized\n'+r'Block $\Phi_R$'))
        score_ax.tick_params(axis='x',top=True,labeltop=True,bottom=False,labelbottom=False,
                             length=0,pad=7,labelsize=11)
        score_ax.tick_params(axis='y',length=0,pad=7,labelsize=10)
        for i,j in np.ndindex(scores.shape):
            score_ax.text(j,i,f'{scores[i,j]:.2f}',ha='center',va='center',fontsize=11,
                          color='white' if scores[i,j]>.82 else INK,
                          weight='bold' if i==0 else 'normal')
        for spine in score_ax.spines.values():
            spine.set_visible(False)
        score_ax.set_xlabel('Support F1',fontsize=11,labelpad=10)
        fig.legend(handles=[Patch(fc='#E5F1EB',ec=GREEN,label='True support'),
                            Patch(fc='#FCEFEA',ec='#B74738',label='Wrong clade'),
                            Patch(fc='white',ec='#B7C0C6',label='Leaf / minimal-support refinement')],
                   loc='lower center',bbox_to_anchor=(.50,.055),ncol=3,frameon=False,
                   fontsize=9.5,handlelength=1,columnspacing=1.2)
        scope_note = ('All scores: common labelings/variants (A:20, B:240, C:60).'
                      if full else r'$\dagger$ SURD: identity labels, variant 0 (1/case). Other scores: all labelings/variants (A:20, B:240, C:60).')
        fig.text(.50,.036,scope_note,
                 ha='center',fontsize=9.2,color='#57636D')
        fig.text(.50,.016,
                 r'Trees: identity labels, variant 0. SURD tree: experimental adaptation. Normalized Block: $\Phi_R/\min(|L|,|R|)$.',
                 ha='center',fontsize=9.2,color='#57636D')
        fig.canvas.draw()
        for ax in networks:
            verify_export_clearance(ax)
        renderer = fig.canvas.get_renderer()
        for ax in trees:
            boxes = [label.get_bbox_patch().get_window_extent(renderer) for label in ax._spt_node_labels]
            if any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]):
                raise ValueError('A displayed tree has overlapping labels')
        # Reject stale HAC labels in every rendered text object.
        if any('HAC' in text.get_text() for text in fig.findobj(plt.Text)):
            raise ValueError('A removed HAC label remains in the figure')
        output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output,dpi=300,bbox_inches=None,facecolor='white')
        plt.close(fig)
    return {'output':str(output),'methods':METHODS,'support_f1':scores.tolist(),
            'baseline_records':{c:sum(r['case']==c for r in records) for c in 'ABC'},
            'surd_records':{c:sum(r['case']==c for r in records) if full else 1 for c in 'ABC'},
            'new_dynamics_fits':0,
            'new_surd_integrations':0,'pixels':[5400,3180]}


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--surd-input',type=Path,
                        default=FULL_INPUT if FULL_INPUT.exists() else Path('/tmp/eisyn_spt_surd_three_cases.json'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(plot(args.surd_input,args.output),indent=2))
