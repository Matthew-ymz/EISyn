"""Full Figure 2 preview: transition hierarchy focal, endpoint controls compact.

No statistical pooling across G, simulations, fitting or tree search. The two
endpoint trees remain separate: zero cross-ROI Syn at G=0 and a display-contracted
positive binary tree at G=10. All original cortex views and network bars remain.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
import numpy as np

from scripts.compose_dmf_brain_main_figure import (
    ROOT, WIDTH, HEIGHT, axes_mm, text_mm, letter, METHOD_LABELS, TREE_SEED,
    render_g0_trivial_tree, render_g10_one_level_tree, _blend_with_white, SPLIT_COLOR,
)
from scripts.plot_dmf_main_roi_highlights import GROUPS, HIGHLIGHT, miniature, COUNTS
from scripts.plot_dmf_spt_collapsed_tree import display_tree, render_collapsed_tree
from scripts.plot_dmf_schaefer100_tree_examples import NETWORK_COLORS
from scripts.spt import flatten_nodes


def draw_focus_layout(fig, threshold, conditions, labels, membership, names,
                      images, order_summary, network_sizes, network_means,
                      component_sd, total_sd):
    bars = []
    network_labels = ['VIS','SM','DA','SA/VA','LIM','FPN','DMN']
    order = np.argsort(network_means[1,0])[::-1]
    tick_labels = [f'{network_labels[i]} / {network_sizes[i]}' for i in order]
    bar_colors = [NETWORK_COLORS[i] for i in order]
    faded = [_blend_with_white(c,.24) for c in bar_colors]
    upper = 5*np.ceil(np.max(network_means.sum(1)+total_sd)*1.06/5)

    def cortex(i, x, top, width, height):
        for v in range(4):
            ax = axes_mm(fig, x+(v%2)*(width/2+.4),
                         top+(v//2)*(height/2+.3),width/2-.4,height/2-.3)
            ax.imshow(images[12+4*i+v],interpolation='none');ax.axis('off')

    def networks(i,x,top,width,height,small=False):
        ax = axes_mm(fig,x,top,width,height)
        within,between=network_means[i,:,order].T
        with plt.rc_context({'hatch.linewidth':.35}):
            ax.barh(np.arange(7)[::-1],within,height=.72,color=bar_colors,lw=0,zorder=2)
            ax.barh(np.arange(7)[::-1],between,left=within,height=.72,color=faded,
                    edgecolor=bar_colors,hatch='///',lw=.3,zorder=2)
        ax.errorbar(within,np.arange(7)[::-1],xerr=component_sd[i,0,order],fmt='none',
                    ecolor='#202020',elinewidth=.45,capsize=.9,capthick=.45,zorder=3)
        if i:
            ax.errorbar(within+between,np.arange(7)[::-1],xerr=total_sd[i,order],fmt='none',
                        ecolor='#202020',elinewidth=.45,capsize=.9,capthick=.45,zorder=3)
        ax.set(xlim=(0,upper),ylim=(-.65,6.65),xticks=[0,20,40] if small else [0,10,20,30,40],
               yticks=np.arange(7)[::-1],yticklabels=tick_labels)
        ax.tick_params(axis='y',labelsize=5.1 if small else 6.0,pad=1.3,length=0)
        ax.tick_params(axis='x',labelsize=5.3 if small else 5.8,pad=1.5,length=2)
        ax.grid(axis='x',color='.91',lw=.45,zorder=0)
        return within.sum(),between.sum(),ax

    # The lower evidence region has a large transition block followed by two
    # compact controls. Every G retains its own tree, four cortex views and bars.
    letter(fig,3,96,'f')
    text_mm(fig,8,96,r'$G=1.3$  ·  Transition range',fontsize=8.1,weight='bold')
    fig.legend(handles=[Patch(facecolor=c,edgecolor='none',label=n)
                        for c,n in zip(NETWORK_COLORS,network_labels)],
        loc='center',bbox_to_anchor=(59/WIDTH,1-102/HEIGHT),ncol=7,frameon=False,
        fontsize=6.0,handlelength=.75,handletextpad=.3,columnspacing=.8,borderaxespad=0.)
    tree,source=conditions[1]
    protected=[tuple(i-1 for i in t) for t in GROUPS.values()]
    raw={frozenset(n.indices):n for n in flatten_nodes(tree)}
    identities={frozenset(t):k for k,t in zip(GROUPS,protected)}
    if not all(t in raw for t in identities):raise ValueError('Highlight absent from native tree')
    ax=axes_mm(fig,3,106,110,38)
    audit=render_collapsed_tree(tree,axis=ax,labels=labels,network_membership=membership,
        network_names=names,seed=TREE_SEED,coupling_g=1.3,show_heading=False,
        show_colorbar=False,show_roi_labels=False,node_label_limit=0,protected_groups=protected)
    for line in ax.lines:line.set_linewidth(.40)
    for collection in ax.collections:
        collection.set_sizes(np.full_like(collection.get_sizes(),3.0 if collection.get_zorder()==3 else 1.7))
        collection.set_linewidths(.42)
    _,nodes,leaf_order,positions,_=display_tree(tree,protected_groups=protected)
    shown={frozenset(n.indices):n for n in nodes}
    for key in ('B','C','D'):
        n=shown[frozenset(i-1 for i in GROUPS[key])]
        xs=[leaf_order.index(i) for i in n.indices];_,y=positions[id(n)]
        ax.add_patch(Rectangle((min(xs)-.65,-.018),max(xs)-min(xs)+1.3,y+.045,
                              color=HIGHLIGHT,alpha=.06,lw=0,zorder=0))
        for d in nodes:
            if d.children and set(d.indices)<=set(n.indices):
                dx,dy=positions[id(d)];pts=[positions[id(c)] for c in d.children]
                ax.plot([p[0] for p in pts],[dy,dy],color=HIGHLIGHT,lw=.75,zorder=2)
                for cx,cy in pts:ax.plot([cx,cx],[cy,dy],color=HIGHLIGHT,lw=.75,zorder=2)
    for key,offset in {'A':(-3,-.215),'B':(-3,.11),'C':(2,.11),'D':(2,.11)}.items():
        n=shown[frozenset(i-1 for i in GROUPS[key])];x,y=positions[id(n)];dx,dy=offset
        ax.plot([x,x+dx],[y,y+dy],color=HIGHLIGHT,lw=.55,zorder=5)
        ax.text(x+dx,y+dy,key,ha='center',va='center',fontsize=6.5,weight='bold',
                color=HIGHLIGHT,zorder=8,bbox=dict(facecolor='white',edgecolor=HIGHLIGHT,
                lw=.65,boxstyle='circle,pad=.18'))
    ax.set_ylim(-.16,1.06)
    frequencies=json.loads(COUNTS.read_text())
    counts={tuple(r['members_one_based']):r['count'] for k in ('3','4')
            for r in frequencies['orders'][k]['methods']['spt']['top3']}
    for x,key,title in [(3,'B','Nested SM–SA/VA'),(41,'C','SM–DA'),(79,'D','SA/VA–DMN')]:
        text_mm(fig,x,151,title,fontsize=6.4,weight='bold')
        aa=axes_mm(fig,x,154,33,36)
        miniature(aa,raw[frozenset(i-1 for i in GROUPS[key])],membership,identities,fontsize_scale=.86)
        txt=(f'A: {counts[GROUPS["A"]]}/93 · B: {counts[GROUPS["B"]]}/93' if key=='B'
             else f'{key}: {counts[GROUPS[key]]}/93 participants')
        text_mm(fig,x,194,txt,fontsize=5.9,color='.30')

    letter(fig,122,96,'g')
    text_mm(fig,127,96,'High-order cortical allocation',fontsize=6.8,weight='bold')
    r=order_summary['by_G'][1]
    text_mm(fig,154,102, f'≥ {threshold} ROI · within-band % · 3-seed means',
            fontsize=5.4,ha='center',color='.3')
    text_mm(fig,154,107,f'Total: {r["high_mean_nats"]:.3f} ± {r["high_seed_sd_nats"]:.3f} nats',
            fontsize=5.8,ha='center')
    cortex(1,124,110,60,33)
    cb=fig.colorbar(plt.cm.ScalarMappable(
        norm=Normalize(order_summary['plot']['vmin'],order_summary['plot']['vmax']),cmap='viridis'),
        cax=axes_mm(fig,132,144.5,44,1.35),orientation='horizontal',ticks=[0,.95,1.9,2.85,3.8])
    cb.ax.tick_params(labelsize=5.1,length=1.3,pad=.8);cb.outline.set_linewidth(.4);bars.append(cb)

    letter(fig,122,151,'h')
    text_mm(fig,127,151,'Functional-network contributions',fontsize=6.5,weight='bold')
    within,between,na=networks(1,143,159,41,30)
    fig.legend(handles=[Patch(facecolor='#777777',edgecolor='none',label='Within-network'),
        Patch(facecolor='#EEE',edgecolor='#777',hatch='///',lw=.3,label='Between-network Shapley')],
        loc='center',bbox_to_anchor=(156/WIDTH,1-155.3/HEIGHT),ncol=2,frameon=False,fontsize=4.8,
        handlelength=1.2,handletextpad=.3,columnspacing=.6,borderaxespad=0.)
    na.set_xlabel(f'Overall {METHOD_LABELS["xi"]} share (%)',fontsize=5.5,labelpad=1.4)
    text_mm(fig,158,197,f'Within {within:.1f}% · Between {between:.1f}%',fontsize=5.2,ha='center',color='.3')
    tree_cmap=LinearSegmentedColormap.from_list('focus_syn',
        [_blend_with_white(SPLIT_COLOR,.12),_blend_with_white(SPLIT_COLOR,.74)])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,3),cmap=tree_cmap),
        cax=axes_mm(fig,115,110,1.35,26),orientation='vertical',ticks=[0,1,2,3])
    cb.ax.tick_params(labelsize=5.1,length=1.3,pad=.8);cb.outline.set_linewidth(.4);bars.append(cb)
    cb.set_label(f'Local Syn / {METHOD_LABELS["xi"]} (%)',fontsize=5.3,
                 labelpad=2,rotation=270,va='bottom')

    control_audits=[]
    for i,x,panel,title in [(0,3,'i',r'$G=0$  ·  No long-range coupling'),
                            (2,97,'j',r'$G=10$  ·  Extreme coupling')]:
        letter(fig,x,202,panel)
        text_mm(fig,x+5,202,title,fontsize=7.1,weight='bold')
        text_mm(fig,x+12,208,'SPT',fontsize=5.8,ha='center',color='.3')
        text_mm(fig,x+43,208,'High-order cortex',fontsize=5.6,ha='center',color='.3')
        text_mm(fig,x+80,208,'Networks',fontsize=5.6,ha='center',color='.3')
        ta=axes_mm(fig,x,212,25,25)
        if i==0:
            ca=render_g0_trivial_tree(ta,conditions[i][1],membership,fontsize=4.7)
            note='Cross-ROI Syn = 0'
        else:
            ca=render_g10_one_level_tree(ta,conditions[i][0],membership,fontsize=4.7)
            note='99 forks condensed'
        for line in ta.lines:line.set_linewidth(.27)
        for collection in ta.collections:collection.set_sizes(collection.get_sizes()*.6)
        control_audits.append(ca)
        cortex(i,x+26,211,34,29)
        wn,bn,na=networks(i,x+73,211,17,29,small=True)
        r=order_summary['by_G'][i]
        note_c=('High-order total = 0' if i==0 else
                f'{r["high_mean_nats"]:.3f} ± {r["high_seed_sd_nats"]:.3f} nats')
        text_mm(fig,x+12,241,note,fontsize=4.8,ha='center',color='.3')
        text_mm(fig,x+43,243,note_c,fontsize=4.8,ha='center',color='.3')
        text_mm(fig,x+78,248,f'{wn:.1f}% / {bn:.1f}%',fontsize=4.8,ha='center',color='.3')
        na.set_xlabel('Overall share (%)',fontsize=4.9,labelpad=1.4)
    text_mm(fig,95,252,'Control panels use the same cortical and network scales as the transition panel.',
            fontsize=5.2,ha='center',color='.35')
    out=ROOT/'tmp/dmf_roi_highlight_20261024'
    out.mkdir(exist_ok=True)
    (out/'full_figure_layout_audit.json').write_text(json.dumps(dict(
        canvas_mm=[WIDTH,HEIGHT],highlight_tree=audit,controls=control_audits,
        cortex_native_pixels_preserved=True,all_three_G_retained=True,
        data_reestimated=False,highlights={k:list(v) for k,v in GROUPS.items()}),indent=2))
    return bars
