"""Figures for the finite-data TM pilot, without rerunning any fit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
METHODS=('spt','pair_phi_r_hac','pair_xi_hac','future_mi_hac','block_phi_r','normalized_block_phi_r')
NAMES=('SPT',r'Pair $\Phi_R$ + HAC',r'Pair $\Xi$ + HAC','Future MI + HAC',
       r'Block $\Phi_R$',r'Normalized Block $\Phi_R$')
COLORS={'A':'#0072B2','B':'#D55E00','C':'#009E73'}
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
       'font.size':10,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
       'xtick.labelsize':9,'ytick.labelsize':9}


def group(data,case,n):
    rows=[r for r in data['records'] if r['case']==case and r['n_train']==n]
    if len(rows)!=len(data['protocol']['seeds']) or any(r['status']!='valid' for r in rows):
        return None
    return sorted(rows,key=lambda r:r['seed'])


def recovery(data,path,*,case_titles=None):
    sizes=data['protocol']['train_sizes']
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,3,figsize=(11.5,4.1),layout='constrained')
        cmap=plt.get_cmap('Blues').copy(); cmap.set_bad('#EEEEEE')
        for ax,case in zip(axes,'ABC'):
            values=np.array([[np.mean([r['methods'][m]['clade_f1'] for r in rows]) if rows else np.nan
                              for n in sizes for rows in [group(data,case,n)]] for m in METHODS])
            ax.imshow(np.ma.masked_invalid(values),cmap=cmap,vmin=0,vmax=1,aspect='auto')
            ax.set_xticks(range(len(sizes)),[f'{n:,}' for n in sizes])
            ax.set_yticks(range(len(METHODS)),NAMES if case=='A' else ['']*len(METHODS))
            ax.tick_params(length=0,pad=7)
            ax.set_xlabel('Training samples')
            title=(case_titles[case] if case_titles else ('Gaussian system' if case=='A' else 'Circular system'))
            ax.set_title(f"{case}   "+title,
                         loc='left',fontweight='bold',fontsize=11,pad=12)
            for i,j in np.ndindex(values.shape):
                label=f'{values[i,j]:.2f}' if np.isfinite(values[i,j]) else 'N/A'
                ax.text(j,i,label,ha='center',va='center',fontsize=10,
                        color='white' if values[i,j]>.65 else '#17212B')
        fig.savefig(path,dpi=260,bbox_inches='tight',facecolor='white'); plt.close(fig)


def diagnostics(data,path):
    sizes=np.asarray(data['protocol']['train_sizes'])
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(2,3,figsize=(10.9,6.5),layout='constrained')
        for col,case in enumerate('ABC'):
            rows=[group(data,case,int(n)) for n in sizes]
            for row,(source,key,label) in enumerate((
                ('error','ei_rmse_nats','EI RMSE (nats)'),
                ('heldout','excess_nll_nats','Excess test NLL (nats)'))):
                ax=axes[row,col]
                values=np.array([[r[source][key] for r in part] if part else [np.nan]*len(data['protocol']['seeds'])
                                 for part in rows])
                for j in range(values.shape[1]):
                    ax.scatter(sizes,values[:,j],s=20,color=COLORS[case],alpha=.45,zorder=3)
                ax.plot(sizes,values.mean(1),lw=1.7,color=COLORS[case],marker='o',ms=4)
                if row==1: ax.axhline(0,color='#777777',lw=.7,ls=':')
                ax.set_xscale('log',base=2); ax.set_xticks(sizes,[f'{n:,}' for n in sizes])
                ax.set_xlabel('Training samples')
                ax.set_ylabel(label)
                ax.set_title(f"{'abcdef'[row*3+col]}   {case}",loc='left',fontweight='bold',fontsize=11,pad=9)
                ax.grid(axis='y',alpha=.18,lw=.5)
                ax.set_ylim(bottom=0)
        fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#555555',alpha=.45,label='Independent data sets'),
                            Line2D([],[],marker='o',color='#555555',lw=1.7,label='Mean')],
                   loc='outside upper center',ncol=2,frameon=False,fontsize=9)
        fig.savefig(path,dpi=260,bbox_inches='tight',facecolor='white'); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/spt_finite_data_tm/summary.json')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/reports/assets/kuramoto_hierarchy')
    args=parser.parse_args(); data=json.loads(args.input.read_text())
    if data['status']!='complete': raise ValueError('Plot requires the completed predeclared experiment')
    args.output.mkdir(parents=True,exist_ok=True)
    recovery(data,args.output/'spt_finite_data_tm_recovery.png')
    diagnostics(data,args.output/'spt_finite_data_tm_diagnostics.png')


if __name__=='__main__': main()
