"""Schematic-led heterogeneous comparison; pair-matrix and mean panels removed.

Three genuinely different nine-node architectures have the same gate/noise
profile and source/target dimensions. Top regions denote declared noisy parity
supports; node shapes encode actual driver gates, not decoration. Representative
trees are fixed to the deepest architecture B; all six conditions appear in the
score panel, including ties and baseline successes. Internal refinements of
minimal supports are unscored under the predeclared mechanistic support metric.
Native raw Syn tolerance=1e-10 bits; invalid input fails, no clipping.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.plot_spt_hierarchy_comparison import METHODS,NAMES,INK,GRAY,BLUE,GREEN,RED,tree,method_result

OUT=ROOT/'docs/reports/assets/kuramoto_hierarchy/spt_heterogeneous_hierarchy_comparison.png'
COLORS={2:BLUE,3:'#297FAB',4:GREEN,5:'#BE7727',6:'#81649F',9:GRAY}
LAYOUTS={
    'A':{1:(-1.55,.62),2:(-1.98,-.25),3:(-1.10,-.25),4:(-.50,.45),5:(-.10,-.28),
         6:(.78,.62),7:(1.55,.52),8:(.90,-.55),9:(1.65,-.40)},
    'B':{1:(-2.10,.55),2:(-2.03,-.02),3:(-1.47,-.28),4:(-.90,.34),5:(-.28,-.28),
         6:(.80,.53),7:(.85,-.13),8:(1.55,-.26),9:(2.05,.39)},
    'C':{1:(-2.10,.53),2:(-2.27,-.22),3:(-1.56,-.19),4:(-.95,.50),5:(-1.08,-.30),6:(-.30,-.14),
         7:(1.12,.57),8:(.77,-.28),9:(1.69,-.24)},
}


def load(path):
    from scripts.spt_heterogeneous_hierarchy_case import run
    records=json.loads(path.read_text())['records'] if path else run()['records']
    ordered=[]
    for mode in ('copied','secret_shared'):
        for case in 'ABC':
            selected=[r for r in records if r['readout_mode']==mode and r['case']==case]
            if len(selected)!=1:
                raise ValueError(f'Missing or duplicate {case}/{mode}')
            r=selected[0]
            a=r['audit']
            if a['minimum_syn_bits'] < -a['tolerance_bits'] or a['significant_violation_count'] or a['clipping'] or abs(a['closure_error_bits'])>a['tolerance_bits']:
                raise ArithmeticError(f'Invalid native Syn input: {a}')
            ordered.append(r)
    if len(records)!=6:
        raise ValueError('Expected all six conditions')
    return ordered


def network(ax,record):
    positions=LAYOUTS[record['case']]
    supports=record['hyperedges']
    for h in sorted(supports,key=len,reverse=True):
        order=len(h)
        color=COLORS[order]
        if order==2:
            a,b=(positions[i] for i in h)
            ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=2.2,zorder=12)
            continue
        pad=.14+.055*(order-3) if order<9 else .42
        theta=np.linspace(0,2*np.pi,64,endpoint=False)
        offsets=pad*np.c_[np.cos(theta),np.sin(theta)]
        points=np.concatenate([np.asarray(positions[i])+offsets for i in h])
        boundary=points[ConvexHull(points).vertices]
        inside={i for i,p in positions.items() if MplPath(boundary).contains_point(p)}
        if inside!=set(h):
            raise ValueError(f'{record["case"]}: region {h} includes {inside}')
        alpha=.032 if order==9 else .075 if order>=4 else .15
        zorder=10-order
        ax.add_patch(Polygon(boundary,fc=color,ec='none',alpha=alpha,zorder=zorder))
        ax.add_patch(Polygon(boundary,fill=False,ec=color,lw=.85 if order==9 else 1.2,
                             linestyle=(0,(4,3)) if order==9 else '-',zorder=zorder+.1))
    for i,(x,y) in positions.items():
        gate=record['node_gates'][i-1]
        marker='D' if gate=='XOR' else 's' if gate=='COPY' else '^' if gate in ('OR','NOR') else 'o'
        ax.scatter([x],[y],s=255,marker=marker,facecolor='white',edgecolor=INK,linewidth=.9,zorder=20)
        ax.text(x,y-.04 if marker=='^' else y,str(i),ha='center',va='center',
                fontsize=10.5,color=INK,zorder=21)
    ax.set(xlim=(-2.76,2.56),ylim=(-1.04,1.17),aspect='equal')
    ax.axis('off')


def plot(records,path):
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':11,'text.color':INK,'axes.labelcolor':INK,'savefig.facecolor':'white'}):
        fig=plt.figure(figsize=(18,8.5),facecolor='white')
        gs=fig.add_gridspec(2,1,height_ratios=[1.0,2.55],left=.048,right=.986,top=.907,bottom=.060,hspace=.14)
        top=gs[0].subgridspec(1,3,wspace=.12)
        for j,r in enumerate(records[:3]):
            ax=fig.add_subplot(top[j])
            network(ax,r)
            ax.text(.5,1.01,f'{r["case"]}  {r["case_label"]}',ha='center',va='bottom',
                    transform=ax.transAxes,fontsize=12.0,weight='bold')
        # Compact legends occupy dedicated strips outside all network data.
        handles=[Line2D([],[],color=BLUE,lw=2,label='2-node')]+[
            Patch(fc=COLORS[k],ec=COLORS[k],alpha=.35,label=f'{k}-node') for k in (3,4,5,6)]+[
            Line2D([],[],color=GRAY,ls='--',label='9-node')]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.35,.993),ncol=6,
                   frameon=False,fontsize=10.0,handlelength=1.2,handletextpad=.4,columnspacing=1.0)
        fig.legend(handles=[Line2D([],[],marker='o',ls='',mfc='white',mec=INK,label='AND / NAND'),
                            Line2D([],[],marker='^',ls='',mfc='white',mec=INK,label='OR / NOR'),
                            Line2D([],[],marker='D',ls='',mfc='white',mec=INK,label='XOR'),
                            Line2D([],[],marker='s',ls='',mfc='white',mec=INK,label='COPY')],
                   loc='upper right',bbox_to_anchor=(.986,.993),ncol=4,frameon=False,
                   fontsize=10.0,handletextpad=.3,columnspacing=.9)
        bottom=gs[1].subgridspec(1,2,width_ratios=[3.95,1.12],wspace=.18)
        grid=bottom[0].subgridspec(2,4,hspace=.14,wspace=.14)
        for row,(r,label) in enumerate(((records[1],'Copied'),(records[4],'Shared'))):
            for col,method in enumerate(METHODS):
                ax=fig.add_subplot(grid[row,col])
                tree(ax,r,method)
                if row==0:
                    ax.set_title(NAMES[method],fontsize=12.0,pad=7,weight='bold',color=GREEN if col==0 else INK)
                if col==0:
                    ax.text(-.075,.5,label,ha='right',va='center',rotation=90,transform=ax.transAxes,
                            fontsize=11.1,color='#57636D')
        fig.text(.047,.638,'Case B',fontsize=11.3,weight='bold',color=INK)
        fig.legend(handles=[Patch(fc='#E5F1EB',ec=GREEN,label='True support'),
                            Patch(fc='#FCEFEA',ec=RED,label='Wrong clade')],
                   loc='lower center',bbox_to_anchor=(.395,.006),ncol=2,frameon=False,
                   fontsize=10.1,handlelength=1.1,columnspacing=1.2)
        score_ax=fig.add_subplot(bottom[1])
        score=np.asarray([[method_result(r,m)['clade_f1'] for m in METHODS] for r in records])
        cmap=LinearSegmentedColormap.from_list('recovery',['#FCF5F0','#C5DECF','#1D7967'])
        score_ax.imshow(score,cmap=cmap,vmin=0,vmax=1,aspect='auto',interpolation='nearest')
        for i,j in itertools.product(range(6),range(4)):
            score_ax.text(j,i,f'{score[i,j]:.2f}',ha='center',va='center',fontsize=11.3,
                          color='white' if score[i,j]>.82 else INK,weight='bold' if j==0 else 'normal')
        score_ax.set_xticks(range(4),['SPT','Pair $\Phi_R$\n+ HAC','Future MI\n+ HAC','Block $\Phi_R$*'])
        score_ax.set_yticks(range(6),[f'{mode} {case}' for mode in ('Copied','Shared') for case in 'ABC'])
        score_ax.tick_params(axis='x',top=True,labeltop=True,bottom=False,labelbottom=False,length=0,labelsize=9.2,pad=7)
        score_ax.tick_params(axis='y',length=0,labelsize=10.0)
        score_ax.axhline(2.5,color='white',lw=3)
        score_ax.set_xlabel('Support F1*',fontsize=11,labelpad=9)
        for spine in score_ax.spines.values():
            spine.set_visible(False)
        path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=300,bbox_inches=None,facecolor='white')
        plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    plot(load(args.input),args.output)
    print(args.output)
