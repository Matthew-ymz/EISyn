"""Frozen-fit comparison from native JSON caches; one dataset, no error bars."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'results/spt_kuramoto_A_fit_improvement'
OUT = ROOT/'docs/reports/assets/kuramoto_hierarchy/spt_kuramoto_A_fit_improvement.png'


def main():
    a = json.loads((FOLDER/'summary.json').read_text())
    assert a['status'] == 'complete'
    methods = ['historical_stationary','baseline','periodic']
    labels = ['Original MLP\nold stationary data','Original MLP\nnew natural data','Periodic MLP\nnew natural data']
    panels = [a['independent_final_conditional_mean']['uniform_independent']['models'],
              a['sensitivity']['jacobian']['models'],a['sensitivity']['joint_response']['models']]
    keys = ['nrmse','relative_frobenius_error','relative_frobenius_error']
    axis_labels = ['Conditional mean error (%)\nTarget ≤5%',
                   'Increment Jacobian error (%)\nTarget ≤10%',
                   'Mixed response error (%)\nAll 36 input pairs']
    colors = ['#777777','#0072B2','#009E73']
    # No aggregation across training seeds; no uncertainty bars are inferred.
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.linewidth':.7,'xtick.labelsize':8}):
        fig,axes = plt.subplots(1,3,figsize=(10.8,3.05),sharey=True,layout='constrained')
        for k,(ax,panel,key,label) in enumerate(zip(axes,panels,keys,axis_labels)):
            values = [100*panel[m][key] for m in methods]
            ax.barh([2,1,0],values,height=.47,color=colors)
            for y,value in zip([2,1,0],values):
                ax.text(value+1.8,y,f'{value:.1f}',va='center',fontsize=9)
            if k < 2:
                ax.axvline([5,10][k],color='#444444',ls='--',lw=.9,zorder=0)
            ax.set(xlim=(0,115),ylim=(-.6,2.6),xticks=[0,25,50,75,100],xlabel=label,
                   yticks=[2,1,0],yticklabels=labels)
            ax.tick_params(axis='y',length=0,pad=8)
            ax.text(0,1.045,'abc'[k],transform=ax.transAxes,fontweight='bold',fontsize=11)
        OUT.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(OUT,dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    print(OUT)


if __name__ == '__main__':
    main()
