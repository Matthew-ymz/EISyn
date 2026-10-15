"""Compare cached model/numerical diagnostics without inventing missing F1."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'docs/reports/assets/kuramoto_hierarchy'
INPUTS={'Kuramoto':ROOT/'results/spt_local_kuramoto_tm/pilot_dt0025.json',
        'HO-gLV':ROOT/'results/spt_local_glv_tm/pilot.json'}
COLORS={'Kuramoto':'#0072B2','HO-gLV':'#D55E00'}
LABELS={'Kuramoto':'Kuramoto: Fourier TM','HO-gLV':'HO-gLV: monomial TM'}
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
       'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'}


def read_data():
    data={name:json.loads(path.read_text()) for name,path in INPUTS.items()}
    common=('train_samples','validation_samples','test_samples','seed','tau','dt','coupling',
            'decay','noise','syn_tolerance_nats','defensive_prior_fraction','precision_target_nats','alphas')
    a,b=data.values()
    for key in common:
        if a['protocol'][key]!=b['protocol'][key]:
            raise ValueError(f'Parallel conditions differ at {key}: {a["protocol"][key]} != {b["protocol"][key]}')
    full_path=ROOT/'results/spt_local_glv_tm/full_table.json'
    if full_path.exists():
        full=json.loads(full_path.read_text())
        if full['protocol']['pilot_sha256']!=hashlib.sha256(INPUTS['HO-gLV'].read_bytes()).hexdigest():
            raise ValueError('Full-table screen belongs to a different pilot snapshot')
        data['HO-gLV']['full_table']=full
        exploratory_path=ROOT/'results/spt_local_glv_tm/exploratory_spt.json'
        if exploratory_path.exists():
            exploratory=json.loads(exploratory_path.read_text())
            if exploratory.get('pilot_sha256')!=full['protocol']['pilot_sha256']:
                raise ValueError('Exploratory SPT belongs to a different pilot')
            data['HO-gLV']['exploratory_spt']=exploratory
    return data


def plot(data,path):
    with plt.rc_context(STYLE):
        fig=plt.figure(figsize=(13.5,4.6),layout='constrained')
        grid=fig.add_gridspec(3,3,height_ratios=(.12,1,.08),width_ratios=(1,1,1.22))
        axes=[fig.add_subplot(grid[1,i]) for i in range(3)]
        legend_axis=fig.add_subplot(grid[0,:]); legend_axis.set_axis_off()
        footer_axis=fig.add_subplot(grid[2,:]); footer_axis.set_axis_off()
        for name,experiment in data.items():
            for row in experiment['records']:
                case=row['case']; marker={'A':'o','B':'s','C':'^'}[case]
                precision=row.get('step_precision',{})
                values=precision.get('rms_difference_rad',precision.get('rms_difference_coordinate'))
                if values:
                    axes[0].plot(precision['dt'][1:],values,color=COLORS[name],marker=marker,
                                 lw=1.3,ms=4,ls='-' if name=='Kuramoto' else '--')
                if 'fit' in row and 'heldout' in row:
                    heldout=row['heldout']
                    noise=heldout.get('irreducible_noise_variance_rad2',heldout.get('irreducible_noise_variance_coordinate2'))
                    fit=np.array([v.get('validation_rmse_rad',v.get('validation_rmse_coordinate')) for v in row['fit']])
                    ratios=fit/np.sqrt(noise)
                    x='ABC'.index(case)+(-.13 if name=='Kuramoto' else .13)
                    axes[1].scatter(x+np.linspace(-.035,.035,len(ratios)),ratios,s=15,color=COLORS[name],alpha=.32)
                    axes[1].scatter([x],[ratios.mean()],s=45,color=COLORS[name],marker=marker,edgecolor='white',lw=.7,zorder=4)
        axes[0].axhline(1e-4,color='#777777',ls=':',lw=.9)
        axes[0].set(xscale='log',yscale='log',xlabel='Finer time step',ylabel='Paired-path RMS difference')
        axes[0].set_xticks([.005,.0025,.00125],['.005','.0025','.00125'])
        axes[0].xaxis.set_minor_locator(NullLocator())
        axes[0].grid(axis='y',alpha=.15,lw=.5)
        axes[1].axhline(1,color='#777777',ls=':',lw=.9)
        axes[1].set(xticks=range(3),xticklabels=list('ABC'),xlabel='Case',ylabel='Validation RMSE / transition-noise SD')
        axes[1].set_ylim(bottom=.85); axes[1].grid(axis='y',alpha=.15,lw=.5)
        statuses={'inverse_proposal_failed':('Inverse check\nfailed','#F3D4CD'),
                  'precision_failed':('EI precision\nfailed','#F3D4CD'),
                  'precision_passed':('EI pilot\npassed','#DAEBDD'),
                  'numerical_failure':('Numerical\nfailure','#F3D4CD'),
                  'budget_exhausted':('Budget\nreached','#ECE6D8'),
                  'step_precision_failed':('Time-step\ncheck failed','#F3D4CD')}
        for i,name in enumerate(INPUTS):
            rows={r['case']:r for r in data[name]['records']}
            for j,case in enumerate('ABC'):
                record=rows.get(case,{})
                label,color=statuses.get(record.get('status'),('Pending','#EEEEEE'))
                if name=='HO-gLV':
                    full=next((r for r in data[name].get('full_table',{}).get('records',[]) if r['case']==case),{})
                    if full.get('status')=='screen_complete':
                        label='Full EI screen\npassed' if full['screen_passed'] else 'Full EI / Syn\ncheck failed'
                        color='#DAEBDD' if full['screen_passed'] else '#F3D4CD'
                    elif full.get('status')=='budget_exhausted':
                        label,color='Full EI\nbudget reached','#ECE6D8'
                    elif full.get('status')=='numerical_failure':
                        label,color='Full EI\nnumerical failure','#F3D4CD'
                    elif data[name].get('full_table') and record.get('precision_passed'):
                        label,color='Pilot passed\nFull EI pending','#ECE6D8'
                    exploratory=next((r for r in data[name].get('exploratory_spt',{}).get('records',[]) if r['case']==case),{})
                    mean=exploratory.get('mean_table',{})
                    if mean.get('status')=='exploratory_only':
                        label=f"EI screen passed\nSPT F1* = {mean['metrics']['clade_f1']:.2f}\n(exploratory)"
                        color='#DAEBDD'
                if record.get('methods',{}).get('spt',{}).get('clade_f1') is not None:
                    label=f"SPT F1\n{record['methods']['spt']['clade_f1']:.2f}"; color='#DAEBDD'
                axes[2].add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=color,edgecolor='white',lw=2))
                axes[2].text(j,i,label,ha='center',va='center',fontsize=9.5,color='#24323C')
        axes[2].set(xlim=(-.5,2.5),ylim=(1.5,-.5),xticks=range(3),xticklabels=list('ABC'),
                    yticks=range(2),yticklabels=list(INPUTS),xlabel='Case')
        axes[2].tick_params(length=0,pad=7)
        for spine in axes[2].spines.values(): spine.set_visible(False)
        for ax,title in zip(axes,('a   Time-step check','b   Prediction fit','c   Evaluation status')):
            ax.set_title(title,loc='left',weight='bold',fontsize=11,pad=12)
        legend_axis.legend(handles=[Line2D([],[],color=color,lw=1.5,label=LABELS[name]) for name,color in COLORS.items()]
                   +[Line2D([],[],color='#555555',marker=marker,ls='',label=f'Case {case}')
                     for case,marker in zip('ABC','os^')],
                   loc='center',ncol=5,frameon=False,fontsize=9)
        footer_axis.text(.5,.5,'Dots: output coordinates · Exploratory F1*: one data seed, four integration scrambles · Six-method comparison pending',
                 ha='center',va='center',fontsize=9,color='#57636D')
        path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=270,bbox_inches='tight',facecolor='white'); plt.close(fig)


if __name__=='__main__':
    data=read_data(); plot(data,ASSETS/'spt_glv_kuramoto_parallel.png')
