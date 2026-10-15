"""Measured B comparison, with numerical replication distinct from statistics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.summarize_spt_kuramoto_b import summarize,ORDER

LABELS={'spt':'SPT','block_phi_r':'Block ΦR',
        'normalized_block_phi_r':'Normalized Block ΦR'}


def main():
    data=summarize(); methods=data['methods']
    supports=[(1,2),(1,2,3),(1,2,3,4),(1,2,3,4,5),(6,7),(6,7,8),(6,7,8,9)]
    count=len(ORDER); recovery=np.full((count,7),np.nan)
    style={'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
           'axes.spines.right':False,'axes.linewidth':.7}
    with plt.rc_context(style):
        fig,(ax,heat)=plt.subplots(1,2,figsize=(11.3,4.3),sharey=True,
            gridspec_kw={'width_ratios':[1.15,1.65]},layout='constrained')
        for row,name in enumerate(ORDER):
            record=methods[name]; metrics=record.get('metrics')
            if metrics is None:
                ax.text(.02,row,'Not evaluated',va='center',color='#687680',fontsize=9)
                continue
            color='#16836A' if name=='spt' else '#9AA7AF'
            f1=metrics['clade_f1']; ax.barh(row,f1,height=.55,color=color,alpha=.85)
            replicas=[r['clade_f1'] for r in record['numerical_replicate_metrics'] if 'clade_f1' in r]
            if replicas:
                ax.scatter(replicas,row+np.linspace(-.13,.13,len(replicas)),s=20,
                           edgecolors='#253946',facecolors='white',linewidths=.8,zorder=3)
            ax.text(f1+.025,row,f'{f1:.3f}',va='center',fontsize=9)
            found={tuple(sorted(c)) for c in metrics['clades']}
            recovery[row]=[float(s in found) for s in supports]
        ax.set_xlim(0,1.18); ax.set_xticks([0,.25,.5,.75,1.]); ax.set_xlabel('Support F1')
        ax.set_yticks(range(count),[LABELS[k] for k in ORDER]); ax.set_ylim(count-.4,-.6)
        ax.set_title('a   Hierarchy recovery',loc='left',weight='bold')
        cmap=ListedColormap(['#F3F0EC','#16836A']); cmap.set_bad('#DDDFE0')
        heat.imshow(np.ma.masked_invalid(recovery),cmap=cmap,vmin=0,vmax=1,aspect='auto')
        heat.tick_params(axis='y',left=False,labelleft=False)
        heat.set_xticks(range(7),['12','123','1234','12345','67','678','6789'],rotation=35,ha='right')
        heat.set_xlabel('Planted non-root support'); heat.set_title('b   Supports in the estimated tree',loc='left',weight='bold')
        for i in range(count):
            for j in range(7):
                if np.isnan(recovery[i,j]): label='NA'; color='#65717A'
                else: label='✓' if recovery[i,j] else '–'; color='white' if recovery[i,j] else '#8A8F92'
                heat.text(j,i,label,ha='center',va='center',color=color,fontsize=11)
        legend=[Patch(facecolor='#9AA7AF',label='Tree from averaged information'),
                Line2D([],[],marker='o',mfc='white',mec='#253946',ls='none',label='Numerical replicate tree')]
        fig.legend(handles=legend,loc='outside upper center',ncol=2,frameon=False,fontsize=9)
        fig.get_layout_engine().set(rect=(0,.14,1,.94))
        fig.text(.5,.071,'B · One frozen fitted joint · Four numerical repetitions · No population significance test',ha='center',fontsize=9,color='#45535E')
        fig.text(.5,.025,'SPT / candidate-based refinements may remain numerically unresolved; see report for per-method diagnostics.',ha='center',fontsize=8,color='#687680')
        out=ROOT/'docs/reports/assets/kuramoto_hierarchy/spt_kuramoto_b_comparison.png'
        out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=240,bbox_inches='tight',facecolor='white')
        plt.close(fig); print(out)


if __name__=='__main__': main()
