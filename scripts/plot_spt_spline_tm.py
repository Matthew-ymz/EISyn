"""Evidence plot for the nonlinear finite-data TM pilot; no refits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
COLORS={'A':'#0072B2','B':'#D55E00','C':'#009E73'}
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
       'font.size':10,'axes.labelsize':10,'axes.spines.top':False,
       'axes.spines.right':False,'xtick.labelsize':9,'ytick.labelsize':9}


def plot(data,path):
    records={r['case']:r for r in data['records']}
    all_changes=[level['maximum_nested_change_nats'] for record in records.values()
                 for level in record.get('integration',{}).get('levels',[])
                 if level['maximum_nested_change_nats'] is not None]
    change_limit=max([.01,*all_changes])*1.2
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(2,3,figsize=(10.8,6.2),layout='constrained')
        for col,case in enumerate('ABC'):
            record=records.get(case,{})
            ax=axes[0,col]
            values=[record.get('affine_baseline',{}).get('heldout',{}).get('excess_nll_nats'),
                    record.get('heldout',{}).get('excess_nll_nats')]
            errors=[record.get('affine_baseline',{}).get('heldout',{}).get('excess_nll_iid_sem_nats'),
                    record.get('heldout',{}).get('excess_nll_iid_sem_nats')]
            for j,value in enumerate(values):
                if value is None:
                    ax.text(j,.5,'N/A',transform=ax.get_xaxis_transform(),ha='center')
                    continue
                ax.bar(j,value,width=.56,color='#A9B1B9' if j==0 else COLORS[case],
                       yerr=errors[j],error_kw={'lw':1,'capsize':3})
                ax.annotate(f'{value:.2f}',(j,value),xytext=(0,7),textcoords='offset points',
                            ha='center',va='bottom',fontsize=10)
            ax.set_xticks([0,1],['Affine TM','Spline TM'])
            ax.set_ylabel('Excess test NLL (nats)')
            ax.set_title(f"{'abc'[col]}   {case}",loc='left',weight='bold',fontsize=11)
            finite=[v for v in values if v is not None]
            ax.set_ylim(bottom=0,top=max(finite)*1.22 if finite else 1)
            ax.grid(axis='y',alpha=.15,lw=.5);ax.set_axisbelow(True)
            ax=axes[1,col]
            levels=record.get('integration',{}).get('levels',[])
            points=[(level['particles'],level['maximum_nested_change_nats'])
                    for level in levels if level['maximum_nested_change_nats'] is not None]
            if points:
                p,v=zip(*points)
                ax.plot(p,v,color=COLORS[case],marker='o',lw=1.6,ms=5)
                ax.axhline(.01,color='#777777',lw=1,ls='--')
                ax.annotate('0.01',(1,.01),xycoords=('axes fraction','data'),
                            xytext=(4,0),textcoords='offset points',va='center',
                            fontsize=8,color='#666666')
            else:
                ax.text(.5,.5,'Not evaluated',transform=ax.transAxes,ha='center',color='#666666')
            ax.set_xlabel('Marginal integration particles')
            ax.set_ylabel('Maximum EI change (nats)')
            ax.set_xticks([512,1024])
            ax.set_xlim(475,1060)
            ax.set_ylim(-change_limit*.045,change_limit)
            ax.set_title(f"{'def'[col]}   {case}",loc='left',weight='bold',fontsize=11)
            ax.grid(axis='y',alpha=.15,lw=.5)
        fig.savefig(path,dpi=260,bbox_inches='tight',facecolor='white')
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/spt_finite_data_spline_tm/summary.json')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/reports/assets/kuramoto_hierarchy/spt_spline_tm_diagnostics.png')
    args=parser.parse_args()
    data=json.loads(args.input.read_text())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    plot(data,args.output)


if __name__=='__main__': main()
