#!/usr/bin/env python3
"""Scientific overview of the frozen natural-SPT exact-member search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_dmf_critical_coalitions import BASE
from scripts.run_dmf_subject_consistency import atomic_json, digest
from scripts.analyze_dmf_critical_coalitions import wilson


def plot(base, phase, output):
    c=json.loads((base/'contract.json').read_text())
    s=json.loads((base/(phase+'_summary.json')).read_text())
    registry=json.loads((base/(phase+'_registry.json')).read_text())['candidates']
    with np.load(base/(phase+'_display.npz')) as a:
        members=json.loads(str(a['members_json'])); subjects=a['subject_ids'].tolist()
        g=a['G'];values=a['raw_syn_nats'].copy();raw=a['repeated_presence'].copy()
        strong=a['repeated_above_background'].copy()
    if not members:
        raise ValueError('No registered candidate available for diagnostic display')
    selected=[next(r for r in registry if r['members']==m) for m in members]
    colours=['#236A91','#BA5636','#717A38','#766294']
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelsize':10,'savefig.facecolor':'white'}):
        fig=plt.figure(figsize=(12,10.2),layout='constrained')
        gs=fig.add_gridspec(2,2,width_ratios=[1.05,1.0],height_ratios=[1,1.8],wspace=.12,hspace=.12)
        ax=fig.add_subplot(gs[0,0])
        orders=np.arange(3,100)
        maxraw=[];maxstrong=[]
        for order in orders:
            rows=[r for r in registry if r['order']==order]
            maxraw.append(max((r['raw_presence']['critical']['count'] for r in rows),default=0))
            maxstrong.append(max((r['above_background']['critical']['count'] for r in rows),default=0))
        ax.plot(orders,maxraw,color='#777777',lw=1,marker='.',ms=3,label='Natural node, ≥2/3 seeds')
        ax.plot(orders,maxstrong,color='#236A91',lw=1.3,marker='o',ms=3,label='Above background, ≥2/3 seeds')
        gate=c['discovery_gate' if phase=='discovery' else 'validation_gate']['minimum_critical_subjects']
        ax.axhline(gate,color='#BA5636',lw=1,ls='--',label=f'Frozen target: {gate}/{len(subjects)}')
        ax.set(xlabel='Exact coalition order (ROI count)',ylabel='Best transition recurrence (subjects)',
               xlim=(2,100),ylim=(-.25,len(subjects)+.6),
               yticks=np.arange(0,len(subjects)+1,max(2,len(subjects)//5)))
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,fontsize=8)
        ax.text(-.12,1.16,'a',transform=ax.transAxes,weight='bold',fontsize=12)

        ax=fig.add_subplot(gs[0,1])
        for ci,row in enumerate(selected[:3]):
            stats=row['above_background']; names=['low_any_leak','critical','high_any_leak']
            y=np.array([stats[k]['rate'] for k in names])*100
            interval=np.array([stats[k]['ci95_wilson'] for k in names])*100
            x=np.arange(3)+(ci-1)*.12
            ax.errorbar(x,y,yerr=np.array([y-interval[:,0],interval[:,1]-y]),
                color=colours[ci],lw=1.2,marker=['o','s','^'][ci],ms=5,capsize=3,
                label=f'C{ci+1}: '+','.join(str(i+1) for i in row['members']))
        ax.set(xticks=[0,1,2],xticklabels=['Low G\n(any point)','Transition\n(≥4/6 points)','High G\n(any point)'],
            ylabel='Subjects with reproducible strong node (%)',xlim=(-.3,2.3),ylim=(-3,103))
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,fontsize=8)
        ax.text(-.14,1.16,'b',transform=ax.transAxes,weight='bold',fontsize=12)

        ax=fig.add_subplot(gs[1,0])
        cmap=ListedColormap(['#EBECEE','#BCC0C5','#7A8B9B','#246A91'])
        nrm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
        piorder=sorted(range(len(subjects)),key=lambda i:c['windows'][subjects[i]]['transition']['midpoint'])
        for row,pi in enumerate(piorder):
            mid=c['windows'][subjects[pi]]['transition']['midpoint']
            present_count=np.isfinite(values[0,pi]).sum(1)
            state=(present_count>0).astype(int)+raw[0,pi].astype(int)+strong[0,pi].astype(int)
            edges=np.r_[g-.05,g[-1]+.05]-mid
            ax.pcolormesh(edges,[row-.5,row+.5],state[None,:],cmap=cmap,norm=nrm,
                          shading='flat',rasterized=True)
        for x in [-.3,.3]:
            ax.axvline(x,color='#BA5636',lw=.8,ls='--')
        ax.set(xlabel='G - individual rate-transition midpoint',ylabel='Discovery subject' if phase=='discovery' else 'Validation subject',
               ylim=(len(subjects)-.5,-.5),xlim=(-3.05,3.65))
        positions=list(range(0,len(subjects),max(1,len(subjects)//16)))
        ax.set_yticks(positions,[subjects[piorder[i]].replace('sub-','') for i in positions])
        ax.text(-.12,1.14,'c',transform=ax.transAxes,weight='bold',fontsize=12)
        handles=[Patch(color=cmap(i),label=label) for i,label in enumerate(
            ['Absent in all seeds','One seed only','Repeated, below reference','Repeated, above reference'])]
        ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,ncol=2,fontsize=7.5)

        nested=gs[1,1].subgridspec(2,1,height_ratios=[1,1.1],hspace=.12)
        ax=fig.add_subplot(nested[0,0])
        for ci in range(min(3,len(selected))):
            by_relative={}
            for pi,p in enumerate(subjects):
                mid=c['windows'][p]['transition']['midpoint']
                for gi,delta in enumerate(g-mid):
                    key=int(round(delta*20))
                    if abs(delta-key/20)>1e-10:
                        raise ValueError('Relative-G points do not share the actual half-step grid')
                    by_relative.setdefault(key,[]).append(bool(strong[ci,pi,gi]))
            keys=sorted(by_relative)
            x=np.array(keys)/20
            rate=np.array([np.mean(by_relative[k])*100 for k in keys])
            ax.plot(x,rate,color=colours[ci],lw=1.1,marker=['o','s','^'][ci],ms=2.7,label=f'C{ci+1}')
            if ci==0:
                intervals=np.array([wilson(sum(by_relative[k]),len(by_relative[k])) for k in keys])*100
                ax.fill_between(x,intervals[:,0],intervals[:,1],color=colours[0],alpha=.12,lw=0)
        ax.axvspan(-.3,.3,color='#BA5636',alpha=.08,lw=0)
        ax.set(xlabel='G - individual rate-transition midpoint',
               ylabel='Strong node recurrence (%)',xlim=(-3.05,3.65),ylim=(-2,102))
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,ncol=3,fontsize=8)
        ax.text(-.14,1.14,'d',transform=ax.transAxes,weight='bold',fontsize=12)

        ax=fig.add_subplot(nested[1,0])
        stage_colours=['#B6BABE','#236A91','#BA5636','#766294']
        stage_labels=['Other G','Low G','Transition window','High G']
        for pi,p in enumerate(subjects):
            x=g-c['windows'][p]['transition']['midpoint']
            stage=np.zeros(len(g),int);stage[c['low_grid_indices']]=1;stage[c['high_grid_indices']]=3
            stage[c['windows'][p]['grid_indices']]=2
            for si in range(3):
                for k in range(4):
                    keep=(stage==k)&np.isfinite(values[0,pi,:,si])
                    ax.scatter(x[keep],values[0,pi,keep,si],s=10,color=stage_colours[k],
                               alpha=.7 if k else .35,edgecolors='none')
        threshold=selected[0]['threshold_nats']
        if threshold is not None:
            ax.axhline(threshold,color='#343434',lw=1,ls='--')
        ax.set(xlabel='G - individual rate-transition midpoint',ylabel='Selected-node raw Syn (nats)',
               xlim=(-3.05,3.65),ylim=(0,None))
        ax.text(-.14,1.14,'e',transform=ax.transAxes,weight='bold',fontsize=12)
        handles=[Line2D([],[],color=co,ls='',marker='o',ms=4,label=la) for co,la in zip(stage_colours,stage_labels)]
        if threshold is not None:
            handles.append(Line2D([],[],color='#343434',ls='--',lw=1,label='G=0 order reference'))
        ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,ncol=3,fontsize=7.5)
        output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output,dpi=220,bbox_inches='tight')
        plt.close(fig)
    atomic_json(base/(phase+'_figure_metadata.json'),dict(output=str(output),
        panel_c_e_members_zero_based=members[0],panel_c_e_members_one_based=[i+1 for i in members[0]],
        summary_sha256=digest(base/(phase+'_summary.json')),missing_node_syn='NaN, never zero',
        panel_a='Order-specific maximum across exact sets, selected descriptive statistic',
        panel_b='Individual-SC proportions and Wilson95% intervals, discovery selected estimates',
        panel_c='Actual relative-G grid, no interpolation; four seed-repetition/strength states',
        panel_d='Strong-node proportion on actual aligned grid; available-subject denominator; C1 Wilson95% band',
        panel_e='Raw selected node Syn, one point per subject/G/seed; absent node omitted',
        legend_placement='Outside all data axes',critical_cell_footprint=[-.3,.3]))
    print(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=BASE)
    p.add_argument('--phase',choices=['discovery','validation'],default='discovery')
    p.add_argument('--output',type=Path,default=ROOT/'fig/dmf_schaefer100/critical_coalition_search.png')
    a=p.parse_args();plot(a.output_dir,a.phase,a.output)


if __name__=='__main__':
    main()
