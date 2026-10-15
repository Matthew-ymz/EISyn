"""Render only measured numerical/model diagnostics; never fill missing F1."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))


def main():
    folder=ROOT/'results/spt_local_kuramoto_tm'
    data={}
    flow_file='flow_mixture_dt0025.json' if (folder/'flow_mixture_dt0025.json').exists() else 'flow_dt0025.json'
    for name,filename in [('Finite-map TM','pilot_dt0025.json'),('Flow-mean TM',flow_file)]:
        path=folder/filename
        if path.exists():
            data[name]=json.loads(path.read_text())
    colors={'Finite-map TM':'#79828A','Flow-mean TM':'#16836A'}
    style={'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'xtick.labelsize':8,'ytick.labelsize':8}
    with plt.rc_context(style):
        fig,axes=plt.subplots(3,3,figsize=(11.7,9.1),layout='constrained')
        maximum_mean_ratio=max((float(np.max(np.asarray(r['heldout']['conditional_mean_mse_raw_rad2'])/
            np.asarray(r['heldout']['irreducible_noise_variance_rad2'])))
            for payload in data.values() for r in payload['records'] if 'heldout' in r),default=1)
        for column,case in enumerate('ABC'):
            records={family:next((r for r in payload['records'] if r['case']==case),{})
                     for family,payload in data.items()}
            reference=next((r for r in records.values() if r.get('step_precision')),None)
            ax=axes[0,column]
            if reference:
                check=reference['step_precision']
                ax.loglog(check['dt'][1:],check['rms_difference_rad'],'o-',color='#284B63',lw=1.5,ms=4)
                ax.axhline(1e-4,color='#BD5548',ls='--',lw=1)
                ax.set_xticks([.00125,.0025,.005],['.00125','.0025','.005'])
                ax.xaxis.set_minor_locator(NullLocator())
                ax.set_xlabel('Finer time step')
                ax.set_ylim(2e-5,5e-4)
            ax.set_title(f'{chr(97+column)}   {case} · '+{'A':'Mixed branches','B':'Deep cascade','C':'Triadic modules'}[case],loc='left',weight='bold')
            if column==0: ax.set_ylabel('Coupled-path RMS error (rad)')
            ax=axes[1,column]
            for family,record in records.items():
                heldout=record.get('heldout')
                if not heldout: continue
                ratio=np.asarray(heldout['conditional_mean_mse_raw_rad2'])/np.asarray(heldout['irreducible_noise_variance_rad2'])
                ax.plot(range(1,10),ratio,'o-',color=colors[family],lw=1.3,ms=3.4)
            ax.axhline(1/16,color='#3B434A',ls=':',lw=1)
            ax.set_xticks(range(1,10)); ax.set_xlabel('Node'); ax.set_ylim(0,maximum_mean_ratio*1.08)
            ax.set_title(f'{chr(100+column)}   Conditional-mean fidelity',loc='left',weight='bold')
            if column==0: ax.set_ylabel('Mean MSE / process-noise variance')
            ax=axes[2,column]
            ax.set_xscale('log',base=2)
            ax.set_yscale('log')
            found=False
            statuses=[]
            for family,record in records.items():
                statuses.append(family+': '+record.get('status','pending').replace('_',' '))
                summary=record.get('precision_summary',[])
                if not summary:
                    # A running cache still has completed per-resolution rows.
                    from scripts.spt_kuramoto_finite_tm import summarize_precision,pilot_masks
                    if record.get('precision'):
                        summary=summarize_precision(record['precision'],pilot_masks(case),case)
                summary=[r for r in summary if r['particles']==256 and r['replicates']==4
                         and r['maximum_ei_sem_nats'] is not None]
                if summary:
                    found=True
                    ax.loglog([r['outer'] for r in summary],[r['maximum_ei_sem_nats'] for r in summary],
                              'o-',color=colors[family],lw=1.3,ms=4)
            ax.axhline(.01,color='#BD5548',ls='--',lw=1)
            flow_status=records.get('Flow-mean TM',{}).get('status','pending')
            gate={'precision_passed':'passed','precision_failed':'not passed',
                  'budget_exhausted':'incomplete','precision_running':'running',
                  'fit_complete':'pending','fitted':'pending'}.get(flow_status,'not certified')
            ax.set_xlabel('Outer samples\nFlow pilot: '+gate)
            ax.set_xticks([256,1024,4096],['256','1024','4096'])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_title(f'{chr(103+column)}   EI numerical uncertainty',loc='left',weight='bold')
            ax.set_ylim(.0003,.1)
            if column==0: ax.set_ylabel('Largest EI scramble SEM (nats)')
            if not found:
                inverse_ok=any(r.get('inverse_check',{}).get('max_center_difference_rad',1)>=0
                               and r.get('inverse_check',{}).get('max_center_difference_rad',1)<1e-4 for r in records.values())
                label='EI integration\nnot completed' if inverse_ok else 'Posterior inversion\nnot certified'
                ax.text(.5,.53,label,ha='center',va='center',transform=ax.transAxes,color='#5D6771',fontsize=9)
        handles=[Line2D([],[],color=c,marker='o',ms=4,label=family) for family,c in colors.items()]
        handles.append(Line2D([],[],color='#BD5548',ls='--',label='Numerical tolerance (a–c, g–i)'))
        handles.append(Line2D([],[],color='#3B434A',ls=':',label='16-repeat noise floor (d–f)'))
        fig.legend(handles=handles,loc='outside upper center',ncol=2,frameon=False,fontsize=9)
        fig.get_layout_engine().set(rect=(0,.035,1,.94))
        fig.text(.5,.008,'Independent uniform phase interventions · Full nine-phase future · Fixed couplings and noise',ha='center',fontsize=9,color='#4D5862')
        out=ROOT/'docs/reports/assets/kuramoto_hierarchy/spt_local_kuramoto_precision.png'
        out.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(out,dpi=240,bbox_inches='tight',facecolor='white')
        plt.close(fig)
        print(out)


if __name__=='__main__': main()
