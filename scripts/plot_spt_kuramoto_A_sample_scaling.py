"""Paired frozen-model diagnostics for the registered sample-size extension."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'results/spt_kuramoto_A_sample_scaling'
OUT = ROOT/'docs/reports/assets/kuramoto_hierarchy/spt_kuramoto_A_sample_scaling.png'


def main():
    a = json.loads((FOLDER/'summary.json').read_text())
    if a['status'] != 'complete':
        raise RuntimeError('Use completed diagnostics only')
    methods = ['pairs65536','pairs262144']
    labels = ['65,536 pairs','262,144 pairs']
    panels = [a['fresh_final_conditional_mean']['uniform_independent']['models'],
              a['sensitivity']['jacobian']['models'],a['sensitivity']['joint_response']['models']]
    keys = ['nrmse','relative_frobenius_error','relative_frobenius_error']
    axis_labels = ['Conditional mean error (%)\nTarget ≤5%',
                   'Increment Jacobian error (%)\nTarget ≤10%',
                   'Mixed response error (%)\nAll 36 input pairs']
    xmax = max(v[m][k]*100 for v,k in zip(panels,keys) for m in methods)*1.23
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.linewidth':.7,'xtick.labelsize':8}):
        fig,axes = plt.subplots(1,3,figsize=(9,2.6),sharey=True,layout='constrained')
        for i,(ax,panel,key,label) in enumerate(zip(axes,panels,keys,axis_labels)):
            values = [100*panel[m][key] for m in methods]
            ax.barh([1,0],values,height=.43,color=['white','#009E73'],
                    edgecolor='#009E73',linewidth=1.1,hatch=['///',None])
            for y,value in zip([1,0],values):
                ax.text(value+.018*xmax,y,f'{value:.2f}',va='center',fontsize=9)
            if i < 2:
                ax.axvline([5,10][i],color='#444444',ls='--',lw=.9,zorder=0)
            ax.set(xlim=(0,xmax),ylim=(-.5,1.5),xlabel=label,yticks=[1,0],yticklabels=labels)
            ax.tick_params(axis='y',length=0,pad=8)
            ax.text(0,1.04,'abc'[i],transform=ax.transAxes,fontweight='bold',fontsize=11)
        fig.savefig(OUT,dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    print(OUT)


if __name__ == '__main__':
    main()
