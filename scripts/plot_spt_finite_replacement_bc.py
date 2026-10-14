"""Render accepted finite-data comparisons and original continuous dynamics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from numpy.polynomial.hermite import hermgauss

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.plot_spt_finite_data_tm import recovery, diagnostics, METHODS, NAMES, STYLE, COLORS
from scripts.plot_spt_continuous_original_dynamics import plot_topologies
from scripts.spt_finite_replacement_bc import ReplacementBroadcast
from scripts.spt_nested_broadcast_case import simulate as simulate_a

OUT=ROOT/'docs/reports/assets/kuramoto_hierarchy'


def load_data():
    a=json.loads((ROOT/'results/spt_finite_data_tm/summary.json').read_text())
    bc=json.loads((ROOT/'results/spt_finite_replacement_bc/summary.json').read_text())
    if bc['status']!='complete' or not bc['acceptance']['replace_main_figure']:
        raise ValueError('Cannot replace current figures without passing predeclared finite-data checks')
    records=[r for r in a['records'] if r['case']=='A']
    records.extend({**r,'methods':r['mean_methods']} for r in bc['records'])
    return {'protocol':{**bc['protocol'],'seeds':(0,1,2)},'records':records}


def current_comparison(data,path):
    metrics=(('clade_f1','Support F1'),('upper_support_recall','Upper-support recall'),
             ('correct_root','Root recovery rate'))
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,3,figsize=(11.8,4.0),layout='constrained')
        for ax,(metric,title),letter in zip(axes,metrics,'abc'):
            values=np.array([[np.mean([r['methods'][method][metric] for r in data['records']
                                      if r['case']==case and r['n_train']==16384]) for case in 'ABC']
                             for method in METHODS])
            ax.imshow(values,cmap='Blues',vmin=0,vmax=1,aspect='auto')
            ax.set_xticks(range(3),list('ABC'))
            ax.set_yticks(range(6),NAMES if ax is axes[0] else ['']*6)
            ax.tick_params(length=0,pad=7)
            ax.set_xlabel('Case')
            ax.set_title(f'{letter}   {title}',loc='left',weight='bold',fontsize=11,pad=12)
            for i,j in np.ndindex(values.shape):
                ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=10,
                        color='white' if values[i,j]>.65 else '#17212B')
        fig.savefig(path,dpi=260,bbox_inches='tight',facecolor='white'); plt.close(fig)


def simulate_originals(steps=80,samples=4096,seed=20261014):
    records={'A':simulate_a(steps,samples,seed)}
    for case in 'BC':
        model=ReplacementBroadcast(case)
        streams=np.random.SeedSequence([seed,ord(case)]).spawn(steps+3)
        state=np.random.default_rng(streams[0]).normal(size=(1,9,model.coordinates_per_node))
        states=[state[0].copy()]; means=[]
        for t in range(steps):
            state=model.step(state,np.random.default_rng(streams[t+1]))
            states.append(state[0].copy()); means.append(state[0,:,model.channels:].mean(0))
        source=np.random.default_rng(streams[-2]).normal(size=(samples,9,model.coordinates_per_node))
        future,drive=model.step(source,np.random.default_rng(streams[-1]),return_drive=True)
        # Fixed largest nonroot support, chosen by generator design, not outcomes.
        slot=max(range(model.channels-1),key=lambda g:len(model.supports[g]))
        records[case]={'case':case,'supports':model.supports,'states':np.asarray(states),
            'joint':np.asarray(means),'response_edge':model.supports[slot],
            'source_sum':drive[:,slot],'response':future[:,:,model.channels+slot].mean(1),
            'model':'replacement_gaussian_B' if case=='B' else 'replacement_monotone_C'}
    return records


def dynamics(records,path):
    gh,weights=hermgauss(64)
    with plt.rc_context(STYLE):
        fig=plt.figure(figsize=(13.1,9.2),layout='constrained')
        grid=fig.add_gridspec(3,3,width_ratios=(1.0,1.03,1.0))
        for row,case in enumerate('ABC'):
            record=records[case]
            left,middle,right=[fig.add_subplot(grid[row,j]) for j in range(3)]
            for ax,values,label,ticks in (
                (left,record['states'][:,:,0].T,'Driver value',list(range(1,10))),
                (middle,record['joint'].T,'Readout value',[''.join(map(str,e)) for e in record['supports']])):
                limit=float(np.max(abs(values)))
                image=ax.imshow(values,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,
                                origin='upper',interpolation='nearest')
                ax.set_yticks(range(len(values)),ticks)
                tick_indices=np.linspace(0,values.shape[1]-1,5).astype(int)
                ax.set_xticks(tick_indices,tick_indices+(0 if ax is left else 1))
                ax.set_ylabel(f'{case} · Node' if ax is left else 'Source support',weight='bold' if ax is left else 'normal')
                bar=fig.colorbar(image,ax=ax,location='bottom',fraction=.07,shrink=.60,pad=.04,aspect=30)
                bar.set_label(label,fontsize=8)
                bar.ax.tick_params(labelsize=8,length=2)
                if row==2:
                    ax.set_xlabel('Time step' if ax is left else 'Future step')
            right.scatter(record['source_sum'],record['response'],s=2.2,alpha=.14,
                          color=COLORS[case],edgecolors='none',rasterized=True)
            extent=1.03*float(np.max(abs(record['source_sum'])))
            drive=np.linspace(-extent,extent,401)
            mean=(np.arcsinh(drive[:,None]+2*gh[None,:])@weights/np.sqrt(np.pi)) if case=='C' else drive
            right.plot(drive,mean,color='#17212B',lw=1.6)
            right.set_xlim(-extent,extent)
            right.set_ylabel('Mean future readout')
            right.set_xlabel(r'Source drive $s_e(t)$')
            right.set_title('Support '+''.join(map(str,record['response_edge'])),loc='left',fontsize=10,pad=5)
            if row==0:
                for ax,title in zip((left,middle,right),('a   Node driver','b   Public readout','c   One-step response')):
                    ax.text(0,1.22,title,transform=ax.transAxes,weight='bold',fontsize=11)
        fig.get_layout_engine().set(rect=(0,.025,1,.965))
        fig.legend(handles=[Line2D([],[],lw=1.6,color='#17212B',label='Conditional mean')],
                   loc='outside upper right',frameon=False,fontsize=9)
        fig.text(.5,.007,'Continuous state maps · Independent one-step interventions',ha='center',fontsize=9,color='#57636D')
        fig.savefig(path,dpi=260,bbox_inches='tight',facecolor='white'); plt.close(fig)


def main():
    data=load_data(); OUT.mkdir(parents=True,exist_ok=True)
    recovery(data,OUT/'spt_finite_replacement_recovery.png',case_titles={
        'A':'Mixed branches','B':'Deep cascade','C':'Triadic modules'})
    diagnostics(data,OUT/'spt_finite_replacement_diagnostics.png')
    current_comparison(data,OUT/'spt_continuous_current_comparison.png')
    records=simulate_originals()
    plot_topologies(records,OUT/'spt_continuous_original_networks.png')
    dynamics(records,OUT/'spt_continuous_original_dynamics.png')
    print('Finite-data recovery, fit diagnostics, original supports and dynamics rendered.')


if __name__=='__main__':
    main()
