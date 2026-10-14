"""Compose the current three-panel framework / continuous-hierarchy figure.

Default: preserve supplied a/b, use validated finite-data A/B/C in c, and omit
the former Kuramoto d. Representative trees use predeclared identity labels,
N=16384 and first dataset seed; scores retain all six methods and three datasets.
--phase-bc reproduces the former Gaussian A / oracle phase B/C plate.
--legacy reproduces the previous four-panel layout from its cached pilot.

Current authority rechecked 2026-10-14: Zotero P6UJCVG8, main DXGC7JEA
Methods (10)-(12), SI MWIWKSVG S5.1; no explicit manuscript version/date.

Legacy layout (--legacy only):
Header: original intervention schematic plus the user-provided confounding
comparison image (no digitization, refitting, rescaling of scores or exclusions).
Middle: existing heterogeneous hierarchy plate, embedded intact.
Footer: continuous mixed-order Kuramoto SPT versus pair-Xi average linkage,
both queried from identical cached global-target EI tables in the approved pilot.
All 16 original coarse K values x 3 seeds are retained; invalid EI is missing.
The learned-component track is used consistently, including its weak-coupling
factorization approximation. No new simulations, estimator or training job.
Native tolerance=.1 bits; no clipping. Pair-Xi is not local-target PhiR.
Module retention and SPT/HAC clade Jaccard describe organization, not accuracy
against a unique high-coupling ground truth or a proof of universal superiority.

Authority freshly located 2026-10-13: Zotero P6UJCVG8; DXGC7JEA Methods (5)-(12),
MWIWKSVG S5 and S7; current attachments lack an explicit version/date.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Patch
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.spt import SPTConfig,build_spt_from_ei_table,flatten_nodes,nontrivial_bipartitions
from scripts.spt_multilevel_parity_case import graph_tree

ASSETS=ROOT/'docs/reports/assets/kuramoto_hierarchy'
THEORY=ROOT/'fig/part1_synergy_comparison/figure1_sources/interventional_peid_decomposition.png'
CONFOUND=ASSETS/'spt_confounding_reference.png'
DISCRETE=ASSETS/'spt_heterogeneous_hierarchy_comparison.png'
PILOT=ROOT/'results/spt_kuramoto_transition_pilot/summary.json'
LEGACY_OUT=ASSETS/'spt_framework_discrete_continuous_comparison.png'
OUT=ASSETS/'spt_framework_continuous_comparison.png'
GREEN,BLUE,PURPLE,INK='#217B68','#286DA0','#735B96','#24323C'
SOURCES=tuple(range(1,7))
MODULES={frozenset((1,2,3)),frozenset((4,5,6))}


def numbers(s):
    return tuple(int(i.removeprefix('theta')) for i in s)


def compare_condition(r):
    if r['status']!='valid':
        return {'k_out':r['k_out'],'seed':r['seed'],'valid':False,'reason':'native Syn audit failed'}
    a=r['nonnegative_audit']
    tolerance=a['tolerance_bits']
    ei={numbers(e['sources']):e['ei_bits'] for e in r['ei_table']}
    if not np.isfinite(list(ei.values())).all():
        raise ArithmeticError('Nonfinite cached EI')
    if len(ei)!=63 or a['nonnegative_clipping'] or a['violating_partition_count'] or a['violating_xi_count']:
        raise ArithmeticError(f'Invalid cached input at {r["k_out"]}/{r["seed"]}: {a}')
    residuals=[ei[s]-ei[l]-ei[q] for s in ei if len(s)>1 for l,q in nontrivial_bipartitions(s)]
    count=sum(v < -tolerance for v in residuals)
    if count:
        raise ArithmeticError(f'Syn: minimum={min(residuals)} bits, threshold={-tolerance}, count={count}')
    # Rebuild from the same table, without planted split constraints, to verify
    # the cached SPT before introducing the graph-only representation.
    result=build_spt_from_ei_table(SOURCES,ei,config=SPTConfig(syn_tolerance=tolerance))
    spt_clades=[n.sources for n in flatten_nodes(result.root) if 1<len(n.sources)<6]
    native={frozenset(numbers(c)) for c in r['nontrivial_clades']}
    if {frozenset(c) for c in spt_clades}!=native or abs(result.closure_error)>1e-8:
        raise ArithmeticError('Canonical SPT differs from the cached hierarchy')
    w=np.zeros((6,6))
    for i,j in itertools.combinations(range(6),2):
        w[i,j]=w[j,i]=ei[(i+1,j+1)]-ei[(i+1,)]-ei[(j+1,)]
    if w.max()<=0:
        raise ValueError('Nonpositive maximum pair affinity; no distance normalization specified')
    # Retain tolerance-scale negative raw affinities; d=1-W/max(W) is valid
    # and nonnegative even for those entries. No max(0, W) projection.
    hac_root,hac_clades=graph_tree(w.tolist())
    found={frozenset(c) for c in hac_clades}
    return {'k_out':r['k_out'],'seed':r['seed'],'valid':True,
        'spt_clades':spt_clades,'hac_clades':hac_clades,
        'spt_root':tuple(c.sources for c in result.root.children),'hac_root':hac_root,
        'spt_module_retention':len(native&MODULES)/2,
        'hac_module_retention':len(found&MODULES)/2,
        'clade_jaccard':len(native&found)/len(native|found),
        'tolerance_bits':tolerance,'minimum_syn_bits':min(residuals),
        'tolerance_negative_count':sum(-tolerance<=v<0 for v in residuals),
        'minimum_pair_xi_bits':float(w.min()),'closure_error_bits':result.closure_error}


def load_comparison():
    payload=json.loads(PILOT.read_text())
    contract=payload['contract']
    rows=[r for r in payload['conditions'] if r['track']=='learned_components']
    expected={(k,s) for k in contract['grid'] for s in contract['seeds']}
    if {(r['k_out'],r['seed']) for r in rows}!=expected or len(rows)!=len(expected):
        raise ValueError('Incomplete original coarse comparison grid')
    return contract,[compare_condition(r) for r in sorted(rows,key=lambda r:(r['seed'],r['k_out']))]


def image_panel(fig,bounds,path):
    image=Image.open(path).convert('RGB')
    ax=fig.add_axes(bounds)
    ax.imshow(image,interpolation='lanczos')
    ax.set_axis_off()
    return ax


def continuous_tree(ax,clades,shared,color):
    sets={tuple(sorted(c)) for c in clades}|{SOURCES}|{(i,) for i in SOURCES}
    children={}
    for s in sets:
        proper=[c for c in sets if set(c)<set(s)]
        children[s]=tuple(sorted(c for c in proper if not any(set(c)<set(d) for d in proper)))
        if len(s)>1 and (len(children[s])!=2 or set().union(*map(set,children[s]))!=set(s)):
            raise ValueError(f'Incomplete displayed tree at {s}')
    positions,counter={},[0]
    def visit(s,depth=0):
        if children[s]:
            x=np.mean([visit(c,depth+1) for c in children[s]])
        else:
            x=counter[0];counter[0]+=1
        positions[s]=(x,-depth)
        return x
    visit(SOURCES)
    for s in sorted(sets,key=len,reverse=True):
        x,y=positions[s]
        for child in children[s]:
            cx,cy=positions[child]
            ax.plot([x,cx],[y,cy],color='#ADB7BF',lw=1.0,zorder=0)
        if len(s)==1:
            edge,face,label='#A9B5BF','white',str(s[0])
        elif s==SOURCES:
            edge,face,label='#84919A','#EFF2F4','1–6'
        else:
            edge,face=('#76848E','#EBF0F3') if frozenset(s) in shared else (color,'#E4F0E9' if color==GREEN else '#E7EFF8')
            label=''.join(map(str,s))
        ax.text(x,y,label,ha='center',va='center',fontsize=9.3,color=INK,
                bbox={'boxstyle':'round,pad=.23','fc':face,'ec':edge,'lw':.9})
    ax.set(xlim=(-.42,5.42),ylim=(min(y for x,y in positions.values())-.24,.28))
    ax.axis('off')


def kuramoto_network(ax):
    positions={1:(-1.48,.55),2:(-1.57,-.54),3:(-.53,.02),4:(.56,.02),5:(1.59,-.53),6:(1.49,.55)}
    for i,j in itertools.product((1,2,3),(4,5,6)):
        x,y=positions[i];u,v=positions[j]
        ax.plot([x,u],[y,v],color='#CBD0D6',lw=.6,zorder=0)
    for i,j,width in ((1,2,2.6),(1,3,.9),(2,3,.9)):
        x,y=positions[i];u,v=positions[j]
        ax.plot([x,u],[y,v],color='#891D16',lw=width,zorder=1)
    ax.add_patch(Polygon([positions[i] for i in (4,5,6)],fc='#E2E0EF',ec='#17106F',lw=1.4,zorder=1))
    for i,(x,y) in positions.items():
        ax.add_patch(Circle((x,y),.12,fc='#38772B',ec='#38772B',zorder=2))
        ax.text(x,y,str(i),ha='center',va='center',color='white',fontsize=10,zorder=3)
    ax.text(1.20,.015,r'$K_3$',color='#54208D',ha='center',va='center',fontsize=11,zorder=4)
    ax.text(0,-.72,r'9 cross-pairs: $K_{\rm out}/3$',ha='center',color='#57636D',fontsize=9.2)
    ax.set(xlim=(-1.90,1.90),ylim=(-.91,.86),aspect='equal')
    ax.axis('off')


def plot_curves(fig,contract,rows,bounds):
    x0,y0,width,height=bounds
    axes=[fig.add_axes((x0,y0+height*.54,width,height*.37)),fig.add_axes((x0,y0,width,height*.37))]
    grid=np.asarray(contract['grid'])
    by_key={(r['k_out'],r['seed']):r for r in rows}
    n_complete=[]
    summaries={name:[] for name in ('spt_module_retention','hac_module_retention','clade_jaccard')}
    for k in grid:
        group=[by_key[(float(k),seed)] for seed in contract['seeds']]
        complete=all(r['valid'] for r in group)
        n_complete.append(complete)
        for name in summaries:
            summaries[name].append(float(np.mean([r[name] for r in group])) if complete else np.nan)
    for field,color,label in (('spt_module_retention',GREEN,'SPT'),('hac_module_retention',BLUE,r'Pair $\Xi$ + HAC')):
        axes[0].plot(grid,summaries[field],color=color,lw=1.8,marker='o',ms=3.0,label=label)
    axes[1].plot(grid,summaries['clade_jaccard'],color=PURPLE,lw=1.7,marker='o',ms=3.0)
    for ax,label in zip(axes,('Original-module retention','SPT–HAC clade Jaccard')):
        ax.set(xlim=(0,5),ylim=(-.045,1.06),yticks=(0,.5,1))
        ax.set_title(label,loc='left',fontsize=9.2,pad=4)
        ax.tick_params(labelsize=8.1,length=2.2)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_linewidth(.7)
        ax.set_xscale('symlog',linthresh=.1,linscale=.45)
        ax.set_xticks((0,.04,.2,1,5),('0','0.04','0.2','1','5'))
        for k,complete in zip(grid,n_complete):
            if not complete:
                ax.plot(k,-.035,marker='x',ms=4,color='#B74738',clip_on=False)
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel(r'$K_{\rm out}$',fontsize=10,labelpad=3)
    axes[0].legend(loc='lower left',bbox_to_anchor=(0,1.33),ncol=2,frameon=False,
                   fontsize=9,handlelength=1.4,columnspacing=1.1,borderaxespad=0)
    return summaries


def compose_legacy(output):
    contract,rows=load_comparison()
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':10,'text.color':INK,'savefig.facecolor':'white'}):
        fig=plt.figure(figsize=(18,15.4),facecolor='white')
        # Three content bands. Existing numeric panels remain intact; large
        # whitespace is not added between rows and no legend covers plot data.
        image_panel(fig,(.025,.807,.335,.170),THEORY)
        image_panel(fig,(.380,.797,.600,.199),CONFOUND)
        fig.text(.019,.980,'a',weight='bold',fontsize=13,va='top')
        # The supplied confounding panel already has its own b label.
        image_panel(fig,(.010,.246,.980,.541),DISCRETE)
        fig.text(.019,.782,'c',weight='bold',fontsize=13,va='top')
        fig.text(.019,.242,'d  Continuous mixed-order Kuramoto',fontsize=12.2,weight='bold',va='top')
        ax=fig.add_axes((.023,.066,.164,.147))
        kuramoto_network(ax)
        fig.text(.105,.035,r'Uniform phases; $\tau=0.20$',ha='center',fontsize=9.1,color='#57636D')
        # Fixed representative seed and couplings, declared before inspection.
        selected=[next(r for r in rows if r['k_out']==k and r['seed']==0) for k in (.04,5.)]
        xstarts=(.207,.477)
        for r,xstart in zip(selected,xstarts):
            shared={frozenset(c) for c in r['spt_clades']} & {frozenset(c) for c in r['hac_clades']}
            fig.text(xstart+.118,.215,rf'$K_{{\rm out}}={r["k_out"]:g}$',ha='center',fontsize=10.8)
            for col,(method,color) in enumerate((('spt',GREEN),('hac',BLUE))):
                tree_ax=fig.add_axes((xstart+col*.130,.039,.119,.154))
                continuous_tree(tree_ax,r[f'{method}_clades'],shared,color)
                tree_ax.set_title('SPT' if method=='spt' else r'Pair $\Xi$ + HAC',fontsize=10.2,pad=3,color=color)
        fig.legend(handles=[Patch(fc='#EBF0F3',ec='#76848E',label='Shared clade'),
                            Patch(fc='#E4F0E9',ec=GREEN,label='SPT-only'),
                            Patch(fc='#E7EFF8',ec=BLUE,label='HAC-only')],
                   loc='lower center',bbox_to_anchor=(.470,.008),ncol=3,frameon=False,
                   fontsize=8.8,handlelength=1,columnspacing=1.0)
        summaries=plot_curves(fig,contract,rows,(.800,.048,.180,.150))
        fig.text(.800,.019,'Mean of 3 seeds; × incomplete condition',fontsize=8.0,color='#57636D')
        output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output,dpi=300,bbox_inches=None,facecolor='white')
        plt.close(fig)
    return {'coarse_conditions':len(rows),'valid_conditions':sum(r['valid'] for r in rows),
            'missing_conditions':[(r['k_out'],r['seed']) for r in rows if not r['valid']],
            'representatives':selected,'grid_summaries':{k:[v if np.isfinite(v) else None for v in values]
                                                       for k,values in summaries.items()},
            'minimum_valid_syn_bits':min(r['minimum_syn_bits'] for r in rows if r['valid']),
            'valid_tolerance_negative_count':sum(r['tolerance_negative_count'] for r in rows if r['valid']),
            'minimum_valid_pair_xi_bits':min(r['minimum_pair_xi_bits'] for r in rows if r['valid']),
            'maximum_closure_error_bits':max(abs(r['closure_error_bits']) for r in rows if r['valid']),
            'new_dynamics_fits':0,'track':'learned_components'}


CURRENT_METHODS=('spt','pair_phi_r_hac','pair_xi_hac','future_mi_hac','block_phi_r','normalized_block_phi_r')
CURRENT_NAMES=('SPT',r'Pair $\Phi_R$'+'\n+ HAC',r'Pair $\Xi$'+'\n+ HAC','Future MI\n+ HAC',
               r'Block $\Phi_R$','Normalized\n'+r'Block $\Phi_R$')


def load_current_comparison(*, finite=True):
    if finite:
        from scripts.spt_nested_broadcast_case import SUPPORTS
        data=json.loads((ROOT/'results/spt_finite_replacement_bc/summary.json').read_text())
        a=json.loads((ROOT/'results/spt_finite_data_tm/summary.json').read_text())
        if data['status']!='complete' or not data['acceptance']['replace_main_figure']:
            raise ValueError('Finite-data replacements require completed, accepted B/C')
        records=[]
        for r in a['records']:
            if r['case']=='A' and r['n_train']==16384:
                records.append({**r,'variant':0,'permutation_id':0,'permutation':tuple(range(1,10)),
                                'hyperedges':SUPPORTS,'evidence':'finite_data'})
        if sorted(r['seed'] for r in records)!=[0,1,2]:
            raise ValueError('A requires all three existing independent datasets')
        for r in data['records']:
            if r['n_train']==16384:
                if r['status']!='valid':
                    raise ValueError('Failed finite fit cannot be plotted as a measured tree')
                records.extend({**s,'evidence':'finite_data'} for s in r['label_records'])
        for r in records:
            audit=r['audit']
            if audit['significant_violation_count'] or audit['clipping'] or abs(audit['closure_error_nats'])>1e-8:
                raise ArithmeticError('Invalid finite-data input for main figure')
            if set(CURRENT_METHODS)-r['methods'].keys():
                raise ValueError('Missing finite-data baseline')
        representatives=[next(r for r in records if r['case']==c and r['seed']==(0 if c=='A' else 101)
                              and r['permutation_id']==0) for c in 'ABC']
        return records,representatives
    from scripts.plot_spt_continuous_refinement import normalized_phase_controls
    phase=json.loads((ROOT/'results/spt_continuous_phase_benchmark/summary.json').read_text())
    nested=json.loads((ROOT/'results/spt_nested_broadcast_case/summary.json').read_text())
    records=nested['records']+normalized_phase_controls(phase)
    if [tuple(r['permutation']) for r in nested['records']]!=[tuple(p) for p in phase['protocol']['permutations']]:
        raise ValueError('A and B/C must use the same 20-label schedule')
    for record in records:
        audit=record['audit']; tolerance=audit['tolerance_nats']
        if audit['significant_violation_count'] or audit['minimum_syn_nats'] < -tolerance:
            raise ArithmeticError(f"Invalid Syn: minimum={audit['minimum_syn_nats']}; threshold={-tolerance}; "
                                  f"affected_count={audit['significant_violation_count']}")
        if audit['clipping'] or audit.get('monotone_projection',False) or abs(audit['closure_error_nats'])>tolerance:
            raise ArithmeticError('Clipped or unclosed continuous hierarchy input')
        if set(CURRENT_METHODS)-record['methods'].keys():
            raise ValueError('Missing method in current continuous record')
    representatives=[]
    for case in 'ABC':
        found=[r for r in records if r['case']==case and r['variant']==0 and r['permutation_id']==0]
        if len(found)!=1:
            raise ValueError(f'Missing or duplicate identity-label representative for {case}')
        representatives.append(found[0])
    return records,representatives


def current_tree(ax,record,method):
    """Render only the measured hierarchy, including explicit no-information cases."""
    result=record['methods'][method]
    if result['status']=='no_structure_information':
        ax.text(.5,.5,'No pair\ninformation',ha='center',va='center',transform=ax.transAxes,
                fontsize=10,color='#7B858D',linespacing=1.5)
        ax.axis('off'); return
    sources=tuple(range(1,10))
    primitive=[set(e) for e in record['hyperedges'] if not any(set(g)<set(e) for g in record['hyperedges'])]
    truth={tuple(sorted(e)) for e in record['hyperedges']}
    sets={tuple(sorted(c)) for c in result['clades']}|{sources}|{(i,) for i in sources}
    children={}
    for parent in sets:
        proper=[c for c in sets if set(c)<set(parent)]
        children[parent]=tuple(sorted(c for c in proper if not any(set(c)<set(d) for d in proper)))
        if len(parent)>1 and (len(children[parent])!=2 or set().union(*map(set,children[parent]))!=set(parent)):
            raise ValueError(f'Incomplete cached {record["case"]}/{method} tree at {parent}')
    positions={}; counter=[0]
    def visit(parent,depth=0):
        if children[parent]:
            x=float(np.mean([visit(child,depth+1) for child in children[parent]]))
        else:
            x=counter[0]; counter[0]+=1
        positions[parent]=(x,-depth)
        return x
    visit(sources)
    def refinement(group):
        return any(set(group)<block for block in primitive)
    labels=[]
    for parent in sorted(sets,key=lambda s:(-len(s),s)):
        x,y=positions[parent]
        for child in children[parent]:
            cx,cy=positions[child]
            wrong=len(child)>1 and child not in truth and not refinement(child)
            ax.plot([x,cx],[y,cy],color='#B74738' if wrong else '#ADB7BF',
                    lw=.85,ls='--' if wrong else '-',zorder=0)
        if parent==sources:
            edge,face,label='#84919A','#EFF2F4','1–9'
        elif len(parent)==1 or refinement(parent):
            edge,face,label='#B7C0C6','white',''.join(map(str,parent))
        elif parent in truth:
            edge,face,label=GREEN,'#E5F1EB',''.join(map(str,parent))
        else:
            edge,face,label='#B74738','#FCEFEA',''.join(map(str,parent))
        labels.append(ax.text(x,y,label,ha='center',va='center',fontsize=8.9,color=INK,
            bbox={'boxstyle':'round,pad=.16','fc':face,'ec':edge,'lw':.8}))
    ax.set(xlim=(-.48,8.48),ylim=(min(y for x,y in positions.values())-.35,.38))
    ax.axis('off'); ax._spt_node_labels=labels


def compose(output, *, finite=True):
    from scripts.plot_spt_heterogeneous_hierarchy import (
        FILLS,PAIR,HIGHER,network,verify_export_clearance,
    )
    records,representatives=load_current_comparison(finite=finite)
    scores=np.asarray([[np.mean([r['methods'][method]['clade_f1'] for r in records if r['case']==case])
                        for case in 'ABC'] for method in CURRENT_METHODS])
    from matplotlib.colors import LinearSegmentedColormap
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':10,'text.color':INK,'savefig.facecolor':'white'}):
        fig=plt.figure(figsize=(18,13.2),facecolor='white',dpi=300)
        image_panel(fig,(.025,.783,.335,.205),THEORY)
        image_panel(fig,(.380,.756,.600,.240),CONFOUND)
        fig.text(.019,.980,'a',weight='bold',fontsize=13,va='top')
        # b is already embedded in the original supplied confounding panel.
        fig.text(.019,.748,'c',weight='bold',fontsize=13,va='top')
        fig.legend(handles=[Patch(fc=FILLS[2],ec=PAIR,label='Pairwise'),
                            Patch(fc=FILLS[3],ec=HIGHER,label='Higher-order'),
                            Line2D([],[],color='#8982B4',ls='--',label='Global')],
                   loc='upper center',bbox_to_anchor=(.51,.754),ncol=3,frameon=False,
                   fontsize=10,handlelength=1.2,columnspacing=1.3)
        networks=[]
        grid=fig.add_gridspec(1,3,left=.048,right=.985,top=.710,bottom=.527,wspace=.10)
        for i,record in enumerate(representatives):
            ax=fig.add_subplot(grid[i]); supports=record['hyperedges']
            primitive=[e for e in supports if not any(set(g)<set(e) for g in supports)]
            network(ax,{'case':record['case'],'hyperedges':supports,'planted_bottom':primitive},
                    operator_symbol='H',node_size=300)
            label=('Mixed branches','Deep cascade','Triadic modules')[i]
            ax.text(.5,1.01,f"{record['case']}  {label}",ha='center',va='bottom',
                    transform=ax.transAxes,fontsize=12,weight='bold')
            networks.append(ax)
        grid=fig.add_gridspec(3,6,left=.052,right=.730,top=.479,bottom=.061,hspace=.20,wspace=.15)
        trees=[]
        for row,record in enumerate(representatives):
            for col,(method,name) in enumerate(zip(CURRENT_METHODS,CURRENT_NAMES)):
                ax=fig.add_subplot(grid[row,col]); current_tree(ax,record,method); trees.append(ax)
                if row==0:
                    ax.set_title(name,fontsize=10,pad=8,color=GREEN if method=='spt' else INK,
                                 weight='bold' if method=='spt' else 'normal')
                if col==0:
                    ax.text(-.15,.5,f"Case {record['case']}",ha='right',va='center',rotation=90,
                            transform=ax.transAxes,fontsize=10.6,color='#57636D')
        score_ax=fig.add_axes((.864,.137,.119,.306))
        cmap=LinearSegmentedColormap.from_list('recovery',['#FCF5F0','#C5DECF','#1D7967'])
        score_ax.imshow(scores,cmap=cmap,vmin=0,vmax=1,aspect='auto',interpolation='nearest')
        # Evaluation counts and label/variant schedules belong in the report
        # and the eventual manuscript appendix, not in the main figure.
        score_ax.set_xticks(range(3),list('ABC'))
        score_ax.set_yticks(range(6),[s.replace('\n',' ') for s in CURRENT_NAMES])
        score_ax.tick_params(axis='x',top=True,labeltop=True,bottom=False,labelbottom=False,length=0,pad=7,labelsize=10)
        score_ax.tick_params(axis='y',length=0,pad=7,labelsize=9.3)
        for i,j in np.ndindex(scores.shape):
            note='*' if not finite and j==2 and CURRENT_METHODS[i] in ('pair_phi_r_hac','pair_xi_hac','future_mi_hac') else ''
            score_ax.text(j,i,f'{scores[i,j]:.2f}'+note,ha='center',va='center',fontsize=10.5,
                          color='white' if scores[i,j]>.82 else INK,weight='bold' if i==0 else 'normal')
        for spine in score_ax.spines.values(): spine.set_visible(False)
        score_ax.set_xlabel('Support F1',fontsize=11,labelpad=10)
        if not finite:
            fig.text(.866,.092,'* No pair information; no reported tree.',fontsize=8.8,color='#57636D',ha='center')
        fig.legend(handles=[Patch(fc='#E5F1EB',ec=GREEN,label='True support'),
                            Patch(fc='#FCEFEA',ec='#B74738',label='Wrong clade'),
                            Patch(fc='white',ec='#B7C0C6',label='Leaf / minimal-support refinement')],
                   loc='lower center',bbox_to_anchor=(.395,.018),ncol=3,frameon=False,
                   fontsize=9,handlelength=1,columnspacing=1.2)
        fig.text(.50,.006,r'Normalized Block: $\Phi_R/\min(|L|,|R|)$.',
                 ha='center',fontsize=8.8,color='#57636D')
        fig.canvas.draw()
        for ax in networks: verify_export_clearance(ax)
        renderer=fig.canvas.get_renderer()
        for ax in trees:
            boxes=[label.get_bbox_patch().get_window_extent(renderer) for label in getattr(ax,'_spt_node_labels',[])]
            if any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]):
                raise ValueError('Displayed tree has overlapping node-label boxes')
        output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output,dpi=300,bbox_inches=None,facecolor='white')
        plt.close(fig)
    return {'panels':['a','b','c'],'removed_panel':'d','pixels':[5400,3960],
            'records_by_case':{case:sum(r['case']==case for r in records) for case in 'ABC'},
            'representatives':('identity labels, N=16384; A seed0, B/C seed101; fixed before fitting' if finite
                               else 'identity labels, variant zero, all A/B/C; no selection by method score'),
            'evidence':'finite_data' if finite else 'oracle_phase_bc',
            'aggregation':('equal-weight independent datasets; B/C labels averaged within fit' if finite
                           else 'all common labels and variants'),
            'methods':CURRENT_METHODS,'support_f1':scores.tolist(),'new_dynamics_fits':0,
            'source_images_unchanged':[str(THEORY),str(CONFOUND)],
            'minimum_syn_nats':min(r['audit']['minimum_syn_nats'] for r in records),
            'tolerance_negative_count':sum(r['audit']['tolerance_negative_count'] for r in records),
            'significant_syn_violations':sum(r['audit']['significant_violation_count'] for r in records),
            'syn_tolerance_nats':1e-8,'clipping':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--legacy',action='store_true',help='Reproduce the former four-panel discrete + Kuramoto composite')
    parser.add_argument('--phase-bc',action='store_true',help='Reproduce the former oracle phase B/C three-panel figure')
    args=parser.parse_args()
    output=args.output or (LEGACY_OUT if args.legacy else OUT.with_stem(OUT.stem+'_phase_bc') if args.phase_bc else OUT)
    summary=compose_legacy(output) if args.legacy else compose(output,finite=not args.phase_bc)
    print(json.dumps(summary,indent=2,allow_nan=False))
