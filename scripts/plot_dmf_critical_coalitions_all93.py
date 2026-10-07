#!/usr/bin/env python3
"""All-subject exact-coalition overview using the actual G grid and raw Syn."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_dmf_critical_coalitions import BASE
from scripts.run_dmf_subject_consistency import atomic_json, digest
from scripts.analyze_dmf_critical_coalitions import wilson


def aligned_rates(repeated, c, subjects):
    by_relative = {}
    for pi, subject in enumerate(subjects):
        mid = c['windows'][subject]['transition']['midpoint']
        for gi, g in enumerate(c['G']):
            key = int(round((g - mid) * 20))
            if abs(g - mid - key / 20) > 1e-10:
                raise ValueError('Actual relative G does not lie on the shared half-step grid')
            by_relative.setdefault(key, []).append(bool(repeated[pi, gi]))
    keys = sorted(by_relative)
    return np.array(keys) / 20, np.array([sum(by_relative[k]) for k in keys]), np.array([len(by_relative[k]) for k in keys])


def plot(base, output, development_only=False):
    prefix = 'all93_devcheck' if development_only else 'all93'
    c = json.loads((base / 'contract.json').read_text())
    summary = json.loads((base / (prefix + '_summary.json')).read_text())
    if summary['scientific_contract_sha256'] != digest(base / 'contract.json'):
        raise ValueError('Summary scientific provenance mismatch')
    with np.load(base / (prefix + '_display.npz')) as a:
        subjects = a['subject_ids'].tolist(); g = a['G'].copy()
        values = a['raw_syn_nats'].copy(); raw = a['repeated_presence'].copy()
        strong = a['repeated_above_background'].copy()
        if json.loads(str(a['members_json'])) != [r['members'] for r in summary['selected']]:
            raise ValueError('Display membership/summary mismatch')
    n = len(subjects); selected = summary['selected']; k = min(3, len(selected))
    colours = ['#236A91', '#BA5636', '#717A38']
    relative_coverage = None
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':9,
            'axes.spines.top':False, 'axes.spines.right':False, 'axes.labelsize':10,
            'savefig.facecolor':'white'}):
        fig = plt.figure(figsize=(12.5, 11.2), layout='constrained')
        gs = fig.add_gridspec(2,2, width_ratios=[1.08,1], height_ratios=[1,1.8], wspace=.13, hspace=.14)
        ax = fig.add_subplot(gs[0,0])
        by_order = summary['by_order'][1:]; orders = [r['order'] for r in by_order]
        for mode, metric, color, ls, label in [
            ('natural_presence','critical_ge4','#85898E','--','Natural node, ≥4/6 window points'),
            ('above_background','critical_ge4','#236A91','-','Strong node, ≥4/6 window points'),
            ('above_background','critical_ge3','#BA5636','-','Strong node, ≥3/6 window points'),
            ('above_background','critical_ge1','#717A38','-','Strong node, ≥1/6 window points')]:
            y = [r['maxima'][mode][metric] for r in by_order]
            ax.plot(orders, y, color=color, ls=ls, lw=1.1, marker='.', ms=3, label=label)
        ax.set(xlabel='Exact coalition order (ROIs)', ylabel=f'Maximum subject count (of {n})',
            xlim=(2,100), ylim=(-.025*n,1.025*n))
        ax.legend(loc='lower left', bbox_to_anchor=(0,1.02), frameon=False, fontsize=8)
        ax.text(-.13,1.04,'a',transform=ax.transAxes,fontweight='bold',fontsize=12)

        ax = fig.add_subplot(gs[0,1])
        metrics = ['low_any','critical_ge1','critical_ge3','critical_ge4','high_any']
        for ci in range(k):
            row = selected[ci]; stats = row['above_background']
            y = np.array([stats[name] for name in metrics]) / n * 100
            bounds = np.array([wilson(stats[name],n) for name in metrics]) * 100
            ax.errorbar(np.arange(5)+(ci-(k-1)/2)*.13, y,
                yerr=np.array([y-bounds[:,0],bounds[:,1]-y]),
                color=colours[ci], marker=['o','s','^'][ci], ms=4.8, lw=1.1, capsize=2.5,
                label=row['display_id']+': '+','.join(str(i+1) for i in row['members']))
        ax.set(xticks=range(5), xticklabels=['Low G\n(any)','Window\n≥1/6','Window\n≥3/6','Window\n≥4/6','High G\n(any)'],
            ylabel='Subjects with reproducible strong node (%)', ylim=(-3,103), xlim=(-.4,4.4))
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),frameon=False,fontsize=8)
        ax.text(-.14,1.04,'b',transform=ax.transAxes,fontweight='bold',fontsize=12)

        ax = fig.add_subplot(gs[1,0])
        cmap = ListedColormap(['#EBECEE','#BCC0C5','#7A8B9B','#246A91'])
        norm = BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
        permutation = sorted(range(n),key=lambda pi:c['windows'][subjects[pi]]['transition']['midpoint'])
        for row, pi in enumerate(permutation):
            midpoint = c['windows'][subjects[pi]]['transition']['midpoint']
            state = (np.isfinite(values[0,pi]).sum(1)>0).astype(int) + raw[0,pi].astype(int) + strong[0,pi].astype(int)
            edges = np.r_[g-.05, g[-1]+.05] - midpoint
            ax.pcolormesh(edges,[row-.5,row+.5],state[None,:],cmap=cmap,norm=norm,shading='flat',rasterized=True)
        for x in [-.3,.3]:
            ax.axvline(x,color='#BA5636',ls='--',lw=.8)
        positions = list(range(0,n,max(1,n//14)))
        ax.set_yticks(positions,[subjects[permutation[i]].replace('sub-','') for i in positions])
        ax.set(xlabel='G − individual rate-transition midpoint', ylabel=f'{selected[0]["display_id"]}: individual SC (sorted by transition)',
            xlim=(-3.05,3.65), ylim=(n-.5,-.5))
        handles=[Patch(color=cmap(i),label=label) for i,label in enumerate(
            ['Absent in all seeds','One seed only','Repeated, below reference','Repeated, above reference'])]
        handles.append(Patch(facecolor='white',edgecolor='#BBBBBB',label='Outside scanned G'))
        ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,ncol=2,fontsize=7.6)
        ax.text(-.13,1.04,'c',transform=ax.transAxes,fontweight='bold',fontsize=12)

        nested = gs[1,1].subgridspec(2,1,height_ratios=[1,1],hspace=.16)
        ax = fig.add_subplot(nested[0,0])
        for ci in range(k):
            x, hits, denominators = aligned_rates(strong[ci],c,subjects)
            rate = hits / denominators * 100
            ax.plot(x,rate,color=colours[ci],lw=1.1,marker=['o','s','^'][ci],ms=2.7,label=selected[ci]['display_id'])
            if ci==0:
                bounds=np.array([wilson(int(a),int(b)) for a,b in zip(hits,denominators)])*100
                ax.fill_between(x,bounds[:,0],bounds[:,1],color=colours[ci],alpha=.12,lw=0)
                relative_coverage=dict(relative_G=x.tolist(),subject_count=denominators.tolist())
        ax.axvspan(-.3,.3,color='#BA5636',alpha=.08,lw=0)
        ax.set(xlabel='G − individual rate-transition midpoint', ylabel='Strong-node recurrence (%)',
            xlim=(-3.05,3.65),ylim=(-2,102))
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,ncol=3,fontsize=8)
        ax.text(-.14,1.04,'d',transform=ax.transAxes,fontweight='bold',fontsize=12)

        ax = fig.add_subplot(nested[1,0])
        stage_colours=['#A4A9AF','#236A91','#BA5636','#766294']
        labels=['Other G','Low G','Transition window','High G']
        for pi, subject in enumerate(subjects):
            x = g - c['windows'][subject]['transition']['midpoint']
            stage=np.zeros(len(g),int); stage[c['low_grid_indices']]=1; stage[c['high_grid_indices']]=3
            stage[c['windows'][subject]['grid_indices']]=2
            for si in range(3):
                for label in range(4):
                    keep=(stage==label)&np.isfinite(values[0,pi,:,si])
                    ax.scatter(x[keep],values[0,pi,keep,si],s=9,color=stage_colours[label],
                        alpha=.65 if label else .22,edgecolors='none')
        threshold=selected[0]['threshold_nats']
        if threshold is not None:
            ax.axhline(threshold,color='#343434',ls='--',lw=1)
        ax.set(xlabel='G − individual rate-transition midpoint',ylabel='Selected-node raw Syn (nats)',
            xlim=(-3.05,3.65))
        # Preserve tolerance-scale negative values if any: do not force a zero limit.
        finite=values[0][np.isfinite(values[0])]
        if len(finite) and finite.min()>=0:
            ax.set_ylim(bottom=0)
        handles=[Line2D([],[],color=co,ls='',marker='o',ms=4,label=la) for co,la in zip(stage_colours,labels)]
        if threshold is not None:
            handles.append(Line2D([],[],color='#343434',ls='--',lw=1,label='G=0 order reference'))
        ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,ncol=3,fontsize=7.5)
        ax.text(-.14,1.04,'e',transform=ax.transAxes,fontweight='bold',fontsize=12)
        output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output,dpi=220,bbox_inches='tight'); plt.close(fig)
    metadata=dict(output=str(output), subject_count=n, summary_sha256=digest(base/(prefix+'_summary.json')),
        plotting_sha256=digest(Path(__file__)), panel_c_e_members=selected[0]['members_one_based'],
        panel_a='Selected maxima by order, all exact sets, counts not independent hypothesis tests',
        panel_b='Top strict preferred candidates; Wilson95% descriptive subject intervals after selection',
        panel_c='All individuals, actual shifted grid, four node states; white=not scanned',
        panel_d='Actual aligned grid, variable coverage, C1 Wilson95% descriptive band; no smoothing',
        panel_e='One raw selected-node Syn per subject/G/seed; missing node remains NaN',
        relative_coverage=relative_coverage, legend_placement='Outside every data axes',
        critical_cell_footprint=[-.3,.3], visual_review='pending')
    atomic_json(base/(prefix+'_figure_metadata.json'),metadata)
    print(output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=BASE)
    p.add_argument('--output',type=Path,default=ROOT/'fig/dmf_schaefer100/critical_coalition_all93.png')
    p.add_argument('--development-only',action='store_true')
    a=p.parse_args();plot(a.output_dir,a.output,a.development_only)


if __name__=='__main__':
    main()
