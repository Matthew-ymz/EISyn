#!/usr/bin/env python3
"""Cache-only preview of recurrent coalitions in the existing main-text tree.

The group-mean SC, G=1.3, seed-4 SPT is unchanged. Only singleton-run
contraction stops at the four highlighted native coalitions. Counts are from
the separate frozen 93-participant, individual-SC transition-window analysis;
they are not frequencies of the displayed mean-SC tree or significance scores.
Anatomical labels use the inherited inferred SC row order, pending confirmation.
No simulation, information query, ranking or tree search is performed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plot_dmf_spt_peel_groups import load_comparison, PAIRED_SUMMARY, YEO_METADATA
from scripts.plot_dmf_spt_collapsed_tree import display_tree, render_collapsed_tree
from scripts.plot_dmf_schaefer100_tree_examples import NETWORK_COLORS
from scripts.spt import flatten_nodes

BASE = ROOT / 'results/dmf_schaefer100'
DEST = ROOT / 'fig/dmf_schaefer100/dmf_main_roi_highlights_preview.png'
COUNTS = BASE / 'sc_spt_participant_frequency/summary.json'
GROUPS = {
    'A': (59, 60, 61), 'B': (59, 60, 61, 76),
    'C': (62, 63, 64, 68), 'D': (25, 26, 42),
}
INK = '#26343F'
HIGHLIGHT = '#A44833'


def badge(ax, x, y, label, **kwargs):
    return ax.text(x, y, label, ha='center', va='center', fontsize=8,
                   fontweight='bold', color=HIGHLIGHT, zorder=8,
                   bbox=dict(facecolor='white', edgecolor=HIGHLIGHT,
                             linewidth=.8, boxstyle='circle,pad=.22'), **kwargs)


def miniature(ax, root, membership, identities, fontsize_scale=1.):
    leaves = []
    positions = {}
    def position(n):
        if not n.children:
            leaves.append(n.indices[0]); x, h = len(leaves)-1, 0
        else:
            children = [position(c) for c in n.children]
            x = np.mean([p[0] for p in children]); h = 1+max(p[1] for p in children)
        positions[id(n)] = (x,h)
        return x,h
    position(root)
    ax.set(xlim=(-.7, len(leaves)-.3), ylim=(-1.16, 3.9)); ax.axis('off')
    for n in flatten_nodes(root):
        x,y = positions[id(n)]
        if n.children:
            points = [positions[id(c)] for c in n.children]
            ax.plot([p[0] for p in points], [y,y], color='#8EA4AA', lw=.8, zorder=1)
            for cx,cy in points:
                ax.plot([cx,cx], [cy,y], color='#8EA4AA', lw=.8, zorder=1)
            ax.scatter(x,y,s=15,facecolor='#D4E5E0',edgecolor='#63928B',lw=.6,zorder=3)
            key = identities.get(frozenset(n.indices))
            if key:
                ax.annotate(key, xy=(x,y), xytext=(-17,9), textcoords='offset points',
                            color=HIGHLIGHT, fontsize=9*fontsize_scale, weight='bold',
                            arrowprops=dict(arrowstyle='-',color=HIGHLIGHT,lw=.7),
                            ha='center',va='center', zorder=7)
        else:
            ax.scatter(x,0,s=24,facecolor='white',edgecolor=NETWORK_COLORS[membership[n.indices[0]]],
                       linewidth=1.1,zorder=4)
    for x,i in enumerate(leaves):
        net = ['VIS','SM','DA','SA/VA','LIM','FPN','DMN'][membership[i]]
        ax.text(x,-.30,str(i+1),ha='center',va='top',fontsize=8.2*fontsize_scale, color=INK)
        ax.text(x,-.76,net,ha='center',va='top',fontsize=6.8*fontsize_scale,
                color=NETWORK_COLORS[membership[i]])


def main():
    tree, provenance = load_comparison(PAIRED_SUMMARY,4,[1.3])[0]
    raw = {frozenset(n.indices): n for n in flatten_nodes(tree)}
    protected = [tuple(i-1 for i in t) for t in GROUPS.values()]
    if not all(frozenset(t) in raw for t in protected):
        raise ValueError('A proposed highlight is not an exact native subtree')
    with np.load(YEO_METADATA,allow_pickle=True) as a:
        labels,membership,names = a['region_labels'],a['network_membership'],a['network_names'].tolist()
    summary = json.loads(COUNTS.read_text())
    counts = {tuple(r['members_one_based']): r['count'] for k in ('3','4')
              for r in summary['orders'][k]['methods']['spt']['top3']}
    _,nodes,order,pos,_ = display_tree(tree,protected_groups=protected)
    shown = {frozenset(n.indices): n for n in nodes}
    if not all(frozenset(t) in shown and not shown[frozenset(t)].collapsed for t in protected):
        raise ValueError('Highlighted native fork was hidden by display contraction')
    identities = {frozenset(t): key for key,t in zip(GROUPS,protected)}
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8,
                         'text.color':INK,'savefig.facecolor':'white'}):
        fig = plt.figure(figsize=(7.48,4.92),facecolor='white')
        fig.text(.045,.955,'Existing transition-range SPT',fontsize=10,weight='bold')
        fig.text(.045,.914,r'Mean SC · $G=1.3$ · original tree retained',fontsize=8,color='.35')
        ax = fig.add_axes([.045,.515,.91,.35])
        audit = render_collapsed_tree(tree,axis=ax,labels=labels,network_membership=membership,
            network_names=names,seed=4,coupling_g=1.3,show_heading=False,
            show_colorbar=False,show_roi_labels=False,node_label_limit=0,
            protected_groups=protected)
        # B contains A, so one outline identifies the nested branch. All other
        # background edges, leaf order and quantitative node colors are retained.
        for key in ('B','C','D'):
            ids = GROUPS[key]; n = shown[frozenset(i-1 for i in ids)]
            x,y = pos[id(n)]; xs=[order.index(i-1) for i in ids]
            ax.add_patch(Rectangle((min(xs)-.70,-.017),max(xs)-min(xs)+1.4,y+.045,
                facecolor=HIGHLIGHT,alpha=.055,edgecolor='none',zorder=0))
            for d in nodes:
                if d.children and set(d.indices)<=set(n.indices):
                    dx,dy=pos[id(d)]; pts=[pos[id(c)] for c in d.children]
                    ax.plot([p[0] for p in pts],[dy,dy],color=HIGHLIGHT,lw=.85,zorder=2)
                    for cx,cy in pts: ax.plot([cx,cx],[cy,dy],color=HIGHLIGHT,lw=.85,zorder=2)
        offsets={'A':(-1.9,-.21),'B':(-1.8,.095),'C':(1.3,.095),'D':(1.1,.095)}
        for key,ids in GROUPS.items():
            n=shown[frozenset(i-1 for i in ids)];x,y=pos[id(n)];dx,dy=offsets[key]
            ax.plot([x,x+dx],[y,y+dy],color=HIGHLIGHT,lw=.6,zorder=5)
            badge(ax,x+dx,y+dy,key)
        ax.set_ylim(-.15,1.06)
        fig.text(.045,.487,'Enlarged native subtrees',fontsize=9,weight='bold')
        panels=[(.045,'B','Nested right SM–SA/VA coalition'),
                (.365,'C','Right SM–DA coalition'),
                (.685,'D','Left SA/VA–DMN coalition')]
        for x,key,title in panels:
            fig.text(x,.446,title,fontsize=7.4,weight='bold')
            aa=fig.add_axes([x,.204,.27,.225])
            miniature(aa,raw[frozenset(i-1 for i in GROUPS[key])],membership,identities)
            keys=('A','B') if key=='B' else (key,)
            lines=[f"{k}: {{"+', '.join(map(str,GROUPS[k]))+f"}} · {counts[GROUPS[k]]}/93" for k in keys]
            fig.text(x,.173,'\n'.join(lines),fontsize=7.1,va='top',linespacing=1.4)
        fig.text(.045,.074,'Counts: individual-SC transition-window occurrence; not a score of the displayed mean-SC tree.',
                 fontsize=6.6,color='.37')
        fig.text(.045,.039,'Network interpretation follows the inherited, inferred Schaefer row order; anatomical confirmation is pending.',
                 fontsize=6.6,color='.37')
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        for t in fig.findobj(plt.Text):
            if not t.get_visible() or not t.get_text():continue
            b=t.get_window_extent(renderer)
            if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height:
                raise ValueError(f'Clipped label: {t.get_text()}')
        DEST.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(DEST,dpi=400,bbox_inches=None)
        plt.close(fig)
    record={'tree':provenance,'display_audit':audit,
            'frequency_source':str(COUNTS.relative_to(ROOT)),
            'frequency_sha256':hashlib.sha256(COUNTS.read_bytes()).hexdigest(),
            'groups':[{ 'id':k,'members_one_based':list(t),'count_of_93':counts[t],
                        'atlas_labels':[str(labels[i-1]) for i in t]} for k,t in GROUPS.items()],
            'omitted_from_this_tree':[[10,11,12],[10,11,12,25]],
            'new_simulations':0,'anatomical_row_order':'inherited, inferred; not independently confirmed'}
    out=ROOT/'tmp/dmf_roi_highlight_20261024'
    out.mkdir(exist_ok=True)
    (out/'preview_audit.json').write_text(json.dumps(record,indent=2))
    print(DEST)


if __name__=='__main__':
    main()
