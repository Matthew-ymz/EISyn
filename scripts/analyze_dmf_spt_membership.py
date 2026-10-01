#!/usr/bin/env python3
"""Show named ROI core combinations with counts and native-bit strengths."""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.run_dmf_unconstrained_spt_sweep import scan_grid
from scripts.analyze_dmf_unconstrained_spt import ROIXiOracle,weakest_split

ANCHORS=(.8,1.3,2.)
COLORS=('#436882','#aa725a','#68947b')


def exact_paired_p(differences):
    d=np.asarray(differences,dtype=float)
    if d.ndim!=1 or not len(d) or not np.isfinite(d).all():
        raise ValueError('Expected finite paired differences')
    observed=abs(d.mean())
    null=np.array([abs(np.mean(d*np.array(s))) for s in itertools.product((-1,1),repeat=len(d))])
    return float(np.mean(null>=observed-1e-12))


def mcnemar_exact(before,after):
    a=np.asarray(before,dtype=bool);b=np.asarray(after,dtype=bool)
    entering=int(np.sum(~a&b));leaving=int(np.sum(a&~b));n=entering+leaving
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(entering,leaving)+1))/2**n) if n else 1.
    return entering,leaving,float(p)


def bh_q(pvalues):
    p=np.asarray(pvalues,dtype=float);order=np.argsort(p,kind='stable')
    ranked=p[order]*len(p)/np.arange(1,len(p)+1)
    adjusted=np.minimum.accumulate(ranked[::-1])[::-1]
    q=np.empty_like(p);q[order]=np.minimum(adjusted,1.)
    return q


def membership_data(root):
    gs=scan_grid();seeds=np.arange(3,11);records={}
    for directory in sorted((root/'shards').glob('seeds*')):
        for path in (directory/'trees').glob('seed*.json'):
            r=json.loads(path.read_text());key=(int(r['seed']),round(float(r['G']),2))
            if key in records:
                raise RuntimeError(f'Duplicate condition {key}')
            r['covariance_path']=str(directory/'covariance'/path.with_suffix('.npz').name)
            records[key]=r
    selected=np.zeros((8,len(gs),100),dtype=bool);present=np.zeros((8,len(gs)),dtype=bool)
    for si,s in enumerate(seeds):
        for gi,g in enumerate(gs):
            r=records.get((int(s),float(g)))
            if r is not None:
                selected[si,gi,r['cores']['10']['members']]=True;present[si,gi]=True
    return gs,seeds,records,selected,present


def membership_statistics(gs,seeds,selected):
    def pos(g):return int(np.flatnonzero(np.isclose(gs,g))[0])
    small=np.sum(selected[:,pos(1.28)]!=selected[:,pos(1.32)],axis=1)
    large=np.sum(selected[:,pos(.8)]!=selected[:,pos(1.8)],axis=1)
    delta=large-small
    roi_tests=[]
    for a,b in itertools.combinations(ANCHORS,2):
        for roi in range(100):
            entering,leaving,p=mcnemar_exact(selected[:,pos(a),roi],selected[:,pos(b),roi])
            roi_tests.append({'G_pair':[a,b],'roi_index':roi,'entering':entering,'leaving':leaving,'p':p})
    for record,q in zip(roi_tests,bh_q([r['p'] for r in roi_tests])):record['q']=float(q)
    entering=np.sum(~selected[:,:-1]&selected[:,1:],axis=2)
    leaving=np.sum(selected[:,:-1]&~selected[:,1:],axis=2)
    fine_changes=(entering+leaving)[:,np.isclose(np.diff(gs),.02)]
    same_G={str(g):[int(np.sum(selected[a,pos(g)]!=selected[b,pos(g)]))
                    for a,b in itertools.combinations(range(len(seeds)),2)] for g in ANCHORS}
    return {'same_center_contrast':{'small_pair':[1.28,1.32],'large_pair':[.8,1.8],
        'small_changed_ROI':small.tolist(),'large_changed_ROI':large.tolist(),'large_minus_small':delta.tolist(),
        'paired_two_sided_p':exact_paired_p(delta),'positive_seed_count':int(np.sum(delta>0))},
        'ROI_anchor_tests':roi_tests,'ROI_anchor_tests_BH_family_size':len(roi_tests),
        'adjacent_entering_by_seed':entering.tolist(),'adjacent_leaving_by_seed':leaving.tolist(),
        'actual_core_size_by_seed':selected.sum(axis=2).tolist(),
        'same_G_seed_pair_changed_ROI':same_G,
        'fine_step_summary':{'delta_G':.02,'comparison_count':int(fine_changes.size),
                             'unchanged_count':int(np.sum(fine_changes==0)),
                             'at_most_2_changed_count':int(np.sum(fine_changes<=2)),
                             'at_least_10_changed_count':int(np.sum(fine_changes>=10)),
                             'maximum_changed_ROI':int(fine_changes.max())}}


def fixed_coalitions(gs,seeds,records):
    refs=[records[4,g]['cores']['10']['members'] for g in ANCHORS]
    rows=[];zeros=0
    for seed in seeds:
        if seed==4:continue
        for g in gs:
            with np.load(records[int(seed),float(g)]['covariance_path']) as a:
                oracle=ROIXiOracle(a['conditional_covariance'])
            for ref,members in zip(ANCHORS,refs):
                if not members:continue
                weak=weakest_split(oracle,members);zeros+=weak['tolerance_zero_count']
                rows.append({'reference_G':ref,'reference_seed':4,'G':float(g),'seed':int(seed),
                             'members':members,'xi_bits':oracle.xi(members),'weakest_bits_raw':weak['bits_raw']})
    return rows,zeros


def replication_checks(root,seeds):
    rows=[json.loads(p.read_text()) for p in (root/'sample_validation').glob('seeds*/seed*.json')]
    index={(r['seed'],r['G']):r for r in rows}
    required=(0.,.8,1.28,1.32,1.8)
    if not all((int(s),g) in index for s in seeds for g in required):
        raise RuntimeError('4096-sample replication is incomplete')
    small=[];large=[]
    for s in seeds:
        members=lambda g:set(index[int(s),g]['cores']['10']['members'])
        small.append(len(members(1.28)^members(1.32)))
        large.append(len(members(.8)^members(1.8)))
    delta=np.array(large)-np.array(small)
    search=json.loads((root/'search_validation/summary.json').read_text())
    if not set(ANCHORS)<=set(r['G'] for r in search['records']):
        raise RuntimeError('Three-anchor search replication is incomplete')
    return {'sample_count':4096,'condition_count':len(index),'small_changed_ROI':small,
            'large_changed_ROI':large,'large_minus_small':delta.tolist(),
            'paired_two_sided_p':exact_paired_p(delta),'positive_seed_count':int(np.sum(delta>0)),
            'maximum_original_prefix_covariance_error':max(r['prefix_covariance_max_error'] for r in rows),
            'search_validation':search['records']}


def short_label(name):
    return str(name).replace('7Networks_LH_','L-').replace('7Networks_RH_','R-')


def plot_roi_combinations(gs,seeds,records,selected,present,labels,fixed,statistics,output):
    plt.rcParams.update({'font.family':'sans-serif','font.size':8,'axes.spines.top':False,
                         'axes.spines.right':False,'legend.frameon':False})
    complete=bool(present.all())
    # Explicit spacing keeps 100 ROI names, cards, and outside legends readable.
    fig=plt.figure(figsize=(13.4,13.1))
    counts=selected.sum(axis=0).T.astype(float)
    counts[:,~present.all(axis=0)]=np.nan
    colors=plt.cm.Blues(np.linspace(0,1,9));colors[0]=[1,1,1,1]
    cmap=ListedColormap(colors);cmap.set_bad('#ededed')
    edges=np.r_[gs[0]-(gs[1]-gs[0])/2,(gs[:-1]+gs[1:])/2,gs[-1]+(gs[-1]-gs[-2])/2]
    hemi_axes=[]
    for panel,hemisphere in enumerate(('Left hemisphere','Right hemisphere')):
        ax=fig.add_axes([.16 if panel==0 else .65,.44,.29,.445]);hemi_axes.append(ax)
        first=panel*50
        image=ax.pcolormesh(edges,np.arange(51)-.5,counts[first:first+50],cmap=cmap,
                            norm=BoundaryNorm(np.arange(-.5,9.5),9),rasterized=True)
        ax.set(ylim=(49.5,-.5),xlim=(edges[0],edges[-1]),yticks=np.arange(50),
               yticklabels=[short_label(v) for v in labels[first:first+50]],
               xticks=np.arange(0,3.01,.5),xlabel='Global coupling G')
        ax.tick_params(axis='y',labelsize=7.7,pad=2,length=0)
        ax.tick_params(axis='x',labelsize=8)
        ax.text(0,1.012,'ab'[panel]+'  '+hemisphere,transform=ax.transAxes,fontweight='bold',fontsize=10)
    cb=fig.colorbar(image,cax=fig.add_axes([.16,.945,.78,.016]),orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top');cb.ax.xaxis.set_label_position('bottom')
    cb.set_ticks(range(9));cb.set_label('Number of seeds including the ROI (out of 8)',fontsize=8)
    for i,(ref,color) in enumerate(zip(ANCHORS,COLORS)):
        card=fig.add_axes([.08+i*.31,.275,.27,.10]);card.axis('off')
        record=records.get((4,ref));members=record['cores']['10']['members'] if record else []
        card.text(0,1,f'c{i+1}  Seed 4, G={ref:g}   ({len(members)} ROI)',transform=card.transAxes,
                  va='top',fontweight='bold',color=color,fontsize=9)
        if not record:
            card.text(0,.76,'Calculation pending',transform=card.transAxes,va='top',color='#7e8c96')
        else:
            for j,roi in enumerate(members):
                column=j//5;row=j%5
                card.text(column*.53,.79-row*.145,short_label(labels[roi]),transform=card.transAxes,
                          va='top',fontsize=7.7,color='#263440')
    ax=fig.add_axes([.40,.06,.245,.175])
    if fixed:
        for ref,color in zip(ANCHORS,COLORS):
            rows=[r for r in fixed if r['reference_G']==ref]
            if not rows:continue
            values=np.array([[next(r['xi_bits'] for r in rows if r['seed']==int(s) and r['G']==float(g))
                              for g in gs] for s in seeds if s!=4])
            for value in values:ax.plot(gs,value,color=color,lw=.55,alpha=.22)
            ax.plot(gs,values.mean(0),color=color,lw=1.7,label=f'G={ref:g} core')
    ax.set(xlabel='Global coupling G',ylabel='Fixed-coalition Xi (bits)',xlim=(0,3))
    ax.text(0,1.13,'e  Fixed cores; other 7 seeds',transform=ax.transAxes,fontweight='bold',fontsize=9)
    if fixed:ax.legend(loc='lower center',bbox_to_anchor=(.5,1.005),ncol=3,fontsize=6.7,columnspacing=.7)
    ax=fig.add_axes([.09,.06,.245,.175])
    entering=np.sum(~selected[:,:-1]&selected[:,1:],axis=2).astype(float)
    leaving=np.sum(selected[:,:-1]&~selected[:,1:],axis=2).astype(float)
    valid=present[:,:-1]&present[:,1:];entering[~valid]=np.nan;leaving[~valid]=np.nan
    for values,color,label in [(entering,'#538778','ROI entering'),(leaving,'#b37b63','ROI leaving')]:
        for v in values:ax.plot(gs[1:],v,lw=.4,color=color,alpha=.17)
        means=np.array([v[np.isfinite(v)].mean() if np.isfinite(v).any() else np.nan for v in values.T])
        ax.plot(gs[1:],means,color=color,lw=1.5,label=label)
    ax.axvspan(1.1,1.7,color='#ecf0f3',zorder=-1)
    ax.set(xlabel='G (compared with the preceding G)',ylabel='Number of ROI',ylim=(-.2,10.4),xlim=(0,3))
    ax.text(0,1.13,'d  Adjacent G: member changes',transform=ax.transAxes,fontweight='bold',fontsize=9)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.005),ncol=2,fontsize=7)
    ax=fig.add_axes([.72,.06,.245,.175])
    if statistics:
        c=statistics['same_center_contrast'];small=np.array(c['small_changed_ROI']);large=np.array(c['large_changed_ROI'])
        for i,(a,b) in enumerate(zip(small,large)):
            offset=(i-3.5)*.025
            ax.plot([0+offset,1+offset],[a,b],color='#c2ccd4',lw=.7,marker='o',ms=3)
        ax.scatter([0,1],[small.mean(),large.mean()],color='#436882',marker='D',s=32,zorder=3)
        ax.text(.5,1.01,f'Paired exact p={c["paired_two_sided_p"]:.4g}',transform=ax.transAxes,ha='center',fontsize=8)
    ax.set(xticks=[0,1],xticklabels=['G 1.28 to 1.32\ndelta G=0.04','G 0.8 to 1.8\ndelta G=1.0'],
           ylabel='Number of changed ROI',ylim=(-.5,20.5),xlim=(-.25,1.25))
    ax.tick_params(axis='x',labelsize=7)
    ax.text(0,1.13,'f  Same center G=1.3',transform=ax.transAxes,fontweight='bold',fontsize=9)
    sizes=selected.sum(axis=2)[present]
    caption=f'440 paired conditions; selected cores: {sizes.min()}–{sizes.max()} ROI (limit: 10). Shaded window in d: delta G=0.02; elsewhere: 0.1.'
    if not complete:caption=f'IN PROGRESS: {int(present.sum())}/440 conditions; grey columns are pending. '+caption
    fig.suptitle(caption+'\nLines: individual seeds and mean. G=0 is finite-sample background; atlas label order is inferred.',fontsize=9,y=1.005)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=300,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=ROOT/'results/dmf_schaefer100/unconstrained_spt_wide')
    p.add_argument('--figure',type=Path,default=ROOT/'fig/brain_dmf_spt_roi_membership.png')
    p.add_argument('--partial',action='store_true')
    p.add_argument('--require-replication',action='store_true')
    args=p.parse_args();gs,seeds,records,selected,present=membership_data(args.input_dir)
    with np.load(ROOT/'results/dmf_schaefer100/full/critical_yeo7.npz',allow_pickle=True) as a:labels=a['region_labels'].tolist()
    if not present.all() and not args.partial:raise RuntimeError(f'Only {present.sum()}/440 conditions complete')
    statistics=membership_statistics(gs,seeds,selected) if present.all() else None
    fixed,zeros=fixed_coalitions(gs,seeds,records) if present.all() else ([],0)
    if present.all():
        payload={'status':'complete','condition_count':len(records),'G':gs.tolist(),'seeds':seeds.tolist(),
                 'labels':labels,'statistics':statistics,'fixed_coalitions':fixed,
                 'fixed_core_tolerance_zero_count':zeros,
                 'audit':{'maximum_absolute_closure_error_bits':max(abs(r['closure_error_bits']) for r in records.values()),
                          'minimum_candidate_syn_bits':min(r['audit']['minimum_candidate_syn'] for r in records.values()),
                          'candidate_syn_tolerance_zero_count':sum(r['audit']['tolerance_zero_count'] for r in records.values()),
                          'pair_syn_tolerance_zero_count':sum(r['pair_tolerance_zero_count'] for r in records.values()),
                          'xi_tolerance_zero_count':sum(r['xi_tolerance_zero_count'] for r in records.values())},
                 'reference_cores':[{'G':g,'members':records[4,g]['cores']['10']['members'],
                                     'ROI_names':[labels[i] for i in records[4,g]['cores']['10']['members']]} for g in ANCHORS]}
        if args.require_replication:payload['replication']=replication_checks(args.input_dir,seeds)
        payload['scale_sensitivity']={}
        for limit in ('5','20'):
            payload['scale_sensitivity'][limit]={
                'small_changed_ROI':[len(set(records[int(s),1.28]['cores'][limit]['members'])^
                                        set(records[int(s),1.32]['cores'][limit]['members'])) for s in seeds],
                'large_changed_ROI':[len(set(records[int(s),.8]['cores'][limit]['members'])^
                                        set(records[int(s),1.8]['cores'][limit]['members'])) for s in seeds]}
        (args.input_dir/'membership_summary.json').write_text(json.dumps(payload,separators=(',',':'),allow_nan=False))
        np.savez_compressed(args.input_dir/'membership.npz',G=gs,seeds=seeds,selected=selected,labels=np.array(labels))
    plot_roi_combinations(gs,seeds,records,selected,present,labels,fixed,statistics,args.figure)
    print(f'done: {present.sum()}/440 conditions; {args.figure}',flush=True)


if __name__=='__main__':main()
