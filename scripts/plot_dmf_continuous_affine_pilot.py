#!/usr/bin/env python3
"""Plot the paired continuous-G pilot; raw scores and three-seed SD retained."""
from __future__ import annotations
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_dmf_continuous_affine_pilot import BASE

OUTPUT=ROOT/'fig/dmf_schaefer100/dmf_continuous_affine_pilot.png'
METHODS=['Unrestricted','Current components','Continuous constraint']
COLORS=['#B77A45','#737D88','#178B85']
STYLES=[':', '--', '-']


def values(rows, method, gs, field, *, score=False):
    data=[]
    for g in gs:
        selected=sorted((r for r in rows if r['method']==method and r['G']==g),key=lambda r:r['seed'])
        data.append([r['test_scores'][field] if score else r[field] for r in selected])
    return np.asarray(data)


def draw():
    summary=json.loads((BASE/'summary.json').read_text())
    rows=summary['records']; gs=np.asarray(summary['protocol']['G'])
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'lines.linewidth':1.5,'savefig.facecolor':'white'}):
        fig,axes=plt.subplots(2,2,figsize=(10.4,6.7),layout='constrained')
        for ax,letter in zip(axes.ravel(),'abcd'):
            ax.text(-.10,1.03,letter,transform=ax.transAxes,fontweight='bold',fontsize=11)
            ax.tick_params(width=.7,length=3)
            ax.grid(axis='y',color='.92',lw=.6,zorder=0)
        for method,color,style in zip(METHODS,COLORS,STYLES):
            data=values(rows,method,gs,'xi_nats'); mean,sd=data.mean(1),data.std(1,ddof=1)
            axes[0,0].plot(gs,mean,color=color,ls=style,marker='o',ms=3)
            axes[0,0].fill_between(gs,mean-sd,mean+sd,color=color,alpha=.12,lw=0)
            use=gs<=.1
            data=values(rows,method,gs[use],'cross_roi_nats'); mean,sd=data.mean(1),data.std(1,ddof=1)
            axes[0,1].plot(gs[use],mean,color=color,ls=style,marker='o',ms=3)
            axes[0,1].fill_between(gs[use],mean-sd,mean+sd,color=color,alpha=.12,lw=0)
        axes[0,0].set(xlabel=r'$G$',ylabel=r'Overall $\Xi$ (nats)',xlim=(0,3))
        axes[0,1].set(xlabel=r'$G$ (symmetric log near zero)',ylabel='Cross-ROI increment (nats)')
        axes[0,1].set_xscale('symlog',linthresh=1e-6)
        axes[0,1].set_xticks([0,1e-6,.001,.1],[r'$0$',r'$10^{-6}$',r'$10^{-3}$',r'$10^{-1}$'])
        current=values(rows,'Current components',gs,'nll_per_transition',score=True)
        continuous=values(rows,'Continuous constraint',gs,'nll_per_transition',score=True)
        delta=continuous-current
        axes[1,0].axhline(0,color='.4',lw=.7)
        axes[1,0].errorbar(gs,delta.mean(1),yerr=delta.std(1,ddof=1),color=COLORS[2],
            marker='o',ms=4,capsize=2,elinewidth=.8)
        axes[1,0].set(xlabel='$G$\nContinuous minus current; lower is better',
                     ylabel='Test NLL change (nats / transition)',xlim=(0,3))
        counts=[512,1024,2048]
        for g,color in [(1e-6,'#B77A45'),(1.3,'#178B85')]:
            for method,style in [('Current components','--'),('Continuous constraint','-')]:
                data=np.array([[r['cross_roi_nats'] for r in summary['sample_size']
                    if r['G']==g and r['method']==method and r['count']==count] for count in counts])
                g_label=r'$G=10^{-6}$' if g==1e-6 else r'$G=1.3$'
                axes[1,1].errorbar(counts,data.mean(1),yerr=data.std(1,ddof=1),color=color,ls=style,
                    marker='o',ms=3,capsize=2,elinewidth=.8,
                    label=g_label+' · '+('current' if method=='Current components' else 'continuous'))
        axes[1,1].set(xlabel='Training transitions',ylabel='Cross-ROI increment (nats)',xticks=counts)
        axes[1,1].legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False,fontsize=8)
        fig.legend(handles=[Line2D([0],[0],color=c,ls=s,label=m) for m,c,s in zip(METHODS,COLORS,STYLES)],
            loc='outside upper center',ncol=3,frameon=False,fontsize=9)
        fig.supxlabel('Mean SC · paired seeds 3, 4, 5 · bands / error bars: seed SD · exploratory pilot',fontsize=8)
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(OUTPUT,dpi=300,bbox_inches='tight')
        plt.close(fig)
    print(OUTPUT)


if __name__=='__main__':
    draw()
