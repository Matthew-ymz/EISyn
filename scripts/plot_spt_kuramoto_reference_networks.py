"""Simplified A/C Kuramoto supports in the user's green-node ribbon style.

Use the frozen sampler cache, checked against the current generator. Only real
pairwise and triadic forces are drawn; there are no >=4-body or global forces.
Envelope outline width is proportional to the registered coupling K; area and
overlap do not encode coupling or inferred Syn. Color denotes interaction arity.
No simulation, EI estimation, hierarchy fitting, or worker changes occur here.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.contour import ContourSet
from matplotlib.path import Path as MplPath
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.plot_spt_kuramoto_simplified_networks import CACHE, edges
from scripts.plot_spt_heterogeneous_hierarchy import (
    FILLS, PAIR, HIGHER, LAYOUTS, network, verify_export_clearance,
)

OUT=ROOT/'docs/reports/assets/kuramoto_hierarchy/spt_kuramoto_simplified_reference_style.png'
LINEWIDTH_PER_COUPLING=2.4  # points: K=1,.5,.25 -> 2.4,1.2,.6 pt (4:2:1)
POSITIONS={case:dict(LAYOUTS[case]) for case in ('A','C')}
# Lengthen the last dyad to separate its label from the added 7--8 bridge.
POSITIONS['A'].update({8:(.88,-.65),9:(2.08,-.65)})
ANCHORS={
    'A':{(1,2,3):(-1.54,.02),(4,5):(-.22,.15),(6,7):(1.17,.74),
         (8,9):(1.48,-.65),(3,4):(-.80,.05),(7,8):(1.24,.00),(5,6):(.31,.17)},
    'C':{(1,2,3):(-1.99,.04),(4,5,6):(-.77,.02),(7,8,9):(1.19,.03),
         (3,4):(-1.26,.15),(6,7):(.41,.22)},
}


def validate_rendered(ax, expected):
    """Exact support membership plus whole glyph/node/outline clearance."""
    case,boundaries,positions,labels=ax._parity_geometry
    if set(boundaries)!=set(expected):
        raise ValueError(f'{case}: omitted or invented force')
    for support,boundary in boundaries.items():
        actual={i for i,p in positions.items() if MplPath(boundary).contains_point(p)}
        if actual!=set(support):
            raise ValueError(f'{case}: envelope {support} contains {actual}')
    verify_export_clearance(ax)
    renderer=ax.figure.canvas.get_renderer()
    centers=ax.transData.transform(np.array(list(positions.values())))
    radius=renderer.points_to_pixels(.5*np.sqrt(360)+1.)
    for label in labels:
        # Include half the widest stroke plus a one-point gap to the text.
        pad=1.+.5*LINEWIDTH_PER_COUPLING*max(expected.values())
        box=label.get_window_extent(renderer).padded(renderer.points_to_pixels(pad))
        if any(MplPath(ax.transData.transform(boundary)).intersects_bbox(box,filled=False)
               for boundary in boundaries.values()):
            raise ValueError(f'{case}: label overlaps a weighted outline')
        dx=np.maximum(np.maximum(box.x0-centers[:,0],centers[:,0]-box.x1),0)
        dy=np.maximum(np.maximum(box.y0-centers[:,1],centers[:,1]-box.y1),0)
        if np.any(dx*dx+dy*dy<radius**2):
            raise ValueError(f'{case}: label overlaps a node')
    for support,outline in ax._weighted_outlines.items():
        if not np.allclose(outline.get_linewidths(),LINEWIDTH_PER_COUPLING*expected[support]):
            raise ValueError(f'{case}: incorrect displayed coupling for {support}')


def apply_coupling_widths(ax, weights):
    """Restyle local contour artists without changing the shared renderer."""
    contours=[artist for artist in ax.collections if isinstance(artist,ContourSet)]
    supports=sorted(weights,key=len,reverse=True)  # shared renderer's stable order
    if len(contours)!=2*len(supports):
        raise ValueError('Unexpected contour count: cannot bind widths to actual supports')
    outlines={}
    for support,fill,outline in zip(supports,contours[::2],contours[1::2]):
        if not fill.filled or outline.filled:
            raise ValueError('Unexpected fill / outline ordering')
        outline.set_linewidths(LINEWIDTH_PER_COUPLING*weights[support])
        outlines[support]=outline
    ax._weighted_outlines=outlines


def coupling_reference(weights):
    """Connectivity filtration, independent of any EI or recovered tree.

    Include hyperedges with K_e >= threshold. Connected components at all
    distinct thresholds form a laminar module family, possibly multifurcating.
    """
    nodes=tuple(range(1,10))
    family={(i,) for i in nodes}; height={(i,):0 for i in nodes}
    partitions={}
    for level,threshold in enumerate(sorted(set(weights.values()),reverse=True),start=1):
        parent={i:i for i in nodes}
        def find(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]]; i=parent[i]
            return i
        for edge,weight in weights.items():
            if weight>=threshold:
                for node in edge[1:]:
                    parent[find(node)]=find(edge[0])
        groups={}
        for node in nodes:
            groups.setdefault(find(node),[]).append(node)
        parts=tuple(sorted(tuple(g) for g in groups.values()))
        partitions[threshold]=parts
        for group in parts:
            family.add(group); height.setdefault(group,level)
    if nodes not in family:
        raise ValueError('Reference expects one connected final component')
    children={}
    for group in family:
        proper=[child for child in family if set(child)<set(group)]
        children[group]=tuple(sorted(child for child in proper
                                    if not any(set(child)<set(other) for other in proper)))
    for a in family:
        for b in family:
            if set(a)&set(b) and not (set(a)<=set(b) or set(b)<=set(a)):
                raise ValueError('Reference family is not laminar')
    return family,children,height,partitions


def reference_trees():
    from scripts.spt_local_hyperedge_kuramoto import SUPPORTS
    networks=edges()
    output=OUT.with_name('spt_kuramoto_simplified_module_reference.png')
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':10,'text.color':'#27343D','svg.fonttype':'none'}):
        fig=plt.figure(figsize=(11.8,3.65),dpi=300,facecolor='white')
        grid=fig.add_gridspec(1,2,left=.025,right=.975,bottom=.145,top=.87,wspace=.10)
        text_objects=[]
        for index,case in enumerate(('A','C')):
            ax=fig.add_subplot(grid[index])
            family,children,heights,partitions=coupling_reference(networks[case])
            nonleaves={g for g in family if len(g)>1}
            if nonleaves!=set(SUPPORTS[case]):
                raise ValueError(f'{case}: coupling-derived modules differ from existing scoring reference')
            for group in sorted(family,key=lambda g:(-len(g),g)):
                x=float(np.mean(group))-1; y=heights[group]
                for child in children[group]:
                    # A triad specifies a module, not its unique binary refinement.
                    unspecified=len(group)==3 and len(children[group])==3
                    ax.plot([x,float(np.mean(child))-1],[y,heights[child]],
                            color='#9AAAB4',lw=1.15,ls=(0,(2,2)) if unspecified else '-',zorder=0)
                if len(group)==1:
                    ax.scatter([x],[y],s=210,facecolor='#38772B',edgecolor='#38772B',zorder=3)
                    label=ax.text(x,y,str(group[0]),ha='center',va='center',fontsize=9.4,color='white',zorder=4)
                else:
                    root=len(group)==9
                    label=ax.text(x,y,'1–9' if root else ''.join(map(str,group)),ha='center',va='center',
                        fontsize=11,color='#27343D',zorder=4,
                        bbox={'boxstyle':'round,pad=.22','fc':'#EFF2F4' if root else '#E5F1EB',
                              'ec':'#84919A' if root else '#217B68','lw':1.0})
                text_objects.append(label)
            ax.set(xlim=(-.5,8.5),ylim=(-.34,3.34)); ax.axis('off')
            ax.set_title(f'{case}  Module reference',weight='bold',fontsize=12,pad=10)
            print(case,'coupling-threshold partitions:',partitions)
        fig.legend(handles=[Patch(fc='#E5F1EB',ec='#217B68',label='Reference module'),
                            Line2D([],[],color='#9AAAB4',lw=1.15,ls=(0,(2,2)),
                                   label='Triad: binary refinement unspecified')],
                   loc='lower center',bbox_to_anchor=(.5,.005),ncol=2,frameon=False,fontsize=9)
        fig.canvas.draw(); renderer=fig.canvas.get_renderer()
        boxes=[t.get_window_extent(renderer) for t in text_objects]
        if any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]):
            raise ValueError('Reference tree labels overlap')
        output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output,dpi=300,bbox_inches='tight',pad_inches=.06,facecolor='white',metadata={
            'Description':'Predeclared module reference from coupling-threshold connected components; not an oracle SPT.',
            'Source':str(CACHE.relative_to(ROOT)),
            'SourceSHA256':hashlib.sha256(CACHE.read_bytes()).hexdigest()})
        fig.savefig(output.with_suffix('.svg'),bbox_inches='tight',pad_inches=.06,facecolor='white')
        plt.close(fig)
    print(output)


def main():
    networks=edges()
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':10,'text.color':'#27343D','svg.fonttype':'none'}):
        fig=plt.figure(figsize=(12.6,3.15),dpi=300,facecolor='white')
        grid=fig.add_gridspec(1,2,left=.012,right=.988,bottom=.025,top=.79,wspace=.075)
        axes=[]
        for index,case in enumerate(('A','C')):
            ax=fig.add_subplot(grid[index]); supports=tuple(networks[case])
            primitive=tuple(e for e in supports if not any(set(g)<set(e) for g in supports))
            network(ax,{'case':case,'hyperedges':supports,'planted_bottom':primitive},
                    operator_symbol='H',node_size=360,positions=POSITIONS[case],
                    operator_positions=ANCHORS[case])
            apply_coupling_widths(ax,networks[case])
            title='Mixed modules' if case=='A' else 'Triadic modules'
            ax.text(.5,1.035,f'{case}  {title}',ha='center',va='bottom',transform=ax.transAxes,
                    fontsize=12,weight='bold')
            axes.append(ax)
        # Shared extents align titles, node sizes and row heights across cases.
        points=np.concatenate([boundary for ax in axes for boundary in ax._parity_geometry[1].values()])
        for ax in axes:
            ax.set(xlim=(points[:,0].min()-.12,points[:,0].max()+.12),
                   ylim=(points[:,1].min()-.14,points[:,1].max()+.17))
        fig.legend(handles=[Patch(fc=FILLS[2],ec=PAIR,label='Pairwise'),
                            Patch(fc=FILLS[3],ec=HIGHER,label='Three-body'),
                            *[Line2D([],[],color='#424B52',lw=LINEWIDTH_PER_COUPLING*w,
                                     label=rf'$K={w:g}$') for w in (1.,.5,.25)]],
                   loc='upper center',bbox_to_anchor=(.5,1.01),ncol=5,frameon=False,
                   fontsize=9,handlelength=2.1,columnspacing=1.7)
        fig.canvas.draw()
        for ax,case in zip(axes,('A','C')):
            validate_rendered(ax,networks[case])
        OUT.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(OUT,dpi=300,bbox_inches='tight',pad_inches=.055,facecolor='white',metadata={
            'Description':'Simplified A/C Kuramoto force supports; H_m denotes arity; outline width = 2.4 K points.',
            'Source':str(CACHE.relative_to(ROOT)),
            'SourceSHA256':hashlib.sha256(CACHE.read_bytes()).hexdigest()})
        fig.savefig(OUT.with_suffix('.svg'),bbox_inches='tight',pad_inches=.055,facecolor='white',metadata={
            'Description':'Color denotes arity; outline width = 2.4 K points. Envelope area does not encode coupling or Syn.'})
        plt.close(fig)
    print(OUT)
    print(OUT.with_suffix('.svg'))
    print('Exact memberships, full labels, node clearance and absence of global forces verified.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-trees',action='store_true',help='Draw module references, without asserting a unique oracle SPT')
    args=parser.parse_args()
    reference_trees() if args.reference_trees else main()
