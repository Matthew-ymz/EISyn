"""Schematic-led heterogeneous comparison; pair-matrix and mean panels removed.

Three genuinely different nine-node architectures have the same gate/noise
profile and source/target dimensions. Top regions denote declared noisy parity
supports; all source nodes are green circles. Heterogeneous driver gates remain
in the model/protocol and are no longer encoded by marker shape. The
2026-10-13 style revision uses burgundy curved pair ribbons, navy/lavender
rounded higher-order regions, lobed primitive triads and purple parity symbols,
following the user's reference image. Colors do not encode measured strength.
Representative
trees are fixed to the deepest architecture B; all six conditions appear in the
score panel, including ties and baseline successes. Internal refinements of
minimal supports are unscored under the predeclared mechanistic support metric.
Native raw Syn tolerance=1e-10 bits; invalid input fails, no clipping.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.spatial import ConvexHull
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.plot_spt_hierarchy_comparison import METHODS,NAMES,INK,GRAY,BLUE,GREEN,RED,tree,method_result

OUT=ROOT/'docs/reports/assets/kuramoto_hierarchy/spt_heterogeneous_hierarchy_comparison.png'
PAIR, HIGHER, NODE, OPERATOR = '#891D16', '#17106F', '#38772B', '#54208D'
FILLS={2:'#F1E6E3',3:'#E2E0EF',4:'#ECE9F6',5:'#F0EDF7',6:'#F2EFF8',9:'#FCFBFE'}
OPERATOR_POSITIONS={
    'A':{(1,2,3):(-1.54,.02),(4,5):(-.22,.15),(6,7):(1.17,.74),
         (8,9):(1.23,-.38),(1,2,3,4,5):(-.76,.47),(6,7,8,9):(1.26,.04),
         tuple(range(1,10)):(.15,-.65)},
    'B':{(1,2):(-1.95,.28),(1,2,3):(-1.76,-.02),(1,2,3,4):(-1.40,.47),
         (1,2,3,4,5):(-.51,.19),(6,7):(.94,.20),(6,7,8):(1.18,-.04),
         (6,7,8,9):(1.69,.42),tuple(range(1,10)):(.14,-.66)},
    'C':{(1,2,3):(-1.99,.04),(4,5,6):(-.77,.02),(7,8,9):(1.19,.03),
         (1,2,3,4,5,6):(-1.29,.48),tuple(range(1,10)):(.18,-.60)},
}
LAYOUTS={
    'A':{1:(-1.55,.62),2:(-1.98,-.25),3:(-1.10,-.25),4:(-.50,.45),5:(-.10,-.28),
         6:(.78,.62),7:(1.55,.52),8:(.90,-.55),9:(1.65,-.40)},
    'B':{1:(-2.10,.55),2:(-2.03,-.26),3:(-1.22,-.28),4:(-.90,.34),5:(-.28,-.28),
         6:(.80,.53),7:(.85,-.13),8:(1.55,-.26),9:(2.05,.39)},
    'C':{1:(-2.10,.53),2:(-2.27,-.22),3:(-1.56,-.19),4:(-.95,.50),5:(-1.08,-.30),6:(-.30,-.14),
         7:(1.12,.57),8:(.77,-.28),9:(1.69,-.24)},
}


def load(path):
    from scripts.spt_heterogeneous_hierarchy_case import run
    records=json.loads(path.read_text())['records'] if path else run()['records']
    ordered=[]
    for mode in ('copied','secret_shared'):
        for case in 'ABC':
            selected=[r for r in records if r['readout_mode']==mode and r['case']==case]
            if len(selected)!=1:
                raise ValueError(f'Missing or duplicate {case}/{mode}')
            r=selected[0]
            a=r['audit']
            if a['minimum_syn_bits'] < -a['tolerance_bits'] or a['significant_violation_count'] or a['clipping'] or abs(a['closure_error_bits'])>a['tolerance_bits']:
                raise ArithmeticError(f'Invalid native Syn input: {a}')
            ordered.append(r)
    if len(records)!=6:
        raise ValueError('Expected all six conditions')
    return ordered


def network(ax,record,*,operator_symbol=r'\oplus',node_size=225,positions=None,operator_positions=None):
    """Reference-inspired ribbons/regions for unchanged mechanistic supports.

    Smooth unions of capsules make primitive triads lobed; enclosing supports
    use round convex offsets. Grid contours are only a geometric rendering,
    not simulation data. Every drawn region must contain exactly its source
    nodes; hierarchy nesting is checked on its entire rendered boundary.
    """
    positions=LAYOUTS[record['case']] if positions is None else positions
    supports=record['hyperedges']
    primitive=tuple(map(tuple,record['planted_bottom']))
    x=np.linspace(-3.02,2.81,600)
    y=np.linspace(-1.28,1.33,280)
    xx,yy=np.meshgrid(x,y)
    grid=np.stack([xx,yy],axis=-1)
    boundaries={}

    def segment_distance(a,b):
        delta=b-a
        fraction=np.sum((grid-a)*delta,axis=-1)/np.dot(delta,delta)
        projected=a+np.clip(fraction,0,1)[...,None]*delta
        return np.linalg.norm(grid-projected,axis=-1)

    for h in sorted(supports,key=len,reverse=True):
        h=tuple(h)
        order=len(h)
        points=np.asarray([positions[i] for i in h])
        if order==2:
            # A curved two-endpoint ribbon, rather than an added third endpoint.
            a,b=points
            delta=b-a
            normal=np.array([-delta[1],delta[0]])/np.linalg.norm(delta)
            control=(a+b)/2 + .14*normal
            t=np.linspace(0,1,28)[:,None]
            curve=(1-t)**2*a+2*(1-t)*t*control+t**2*b
            field=np.min(np.stack([segment_distance(a,b) for a,b in zip(curve[:-1],curve[1:])]),axis=0)-.22
        elif order==3 and h in primitive:
            hub=points.mean(axis=0)
            fields=np.stack([segment_distance(point,hub)-.135 for point in points])
            field=-logsumexp(-22*fields,axis=0)/22
        else:
            hull=points[ConvexHull(points).vertices]
            distance=np.min(np.stack([segment_distance(a,b) for a,b in zip(hull,np.roll(hull,-1,axis=0))]),axis=0)
            within=MplPath(hull).contains_points(grid.reshape(-1,2)).reshape(xx.shape)
            field=np.where(within,-distance,distance)-(.27+.09*(order-3) if order<9 else .62)
        zorder=10-order if order>2 else 12
        color=PAIR if order==2 else HIGHER
        ax.contourf(xx,yy,field,levels=[-100,0],colors=[FILLS[order]],zorder=zorder)
        outline=ax.contour(xx,yy,field,levels=[0],colors=[color if order<9 else '#8982B4'],
                           linewidths=1.35 if order<9 else .9,
                           linestyles='solid' if order<9 else [(0,(4,3))],zorder=zorder+.1)
        boundary=max(outline.allsegs[0],key=len)
        boundaries[h]=boundary
        inside={i for i,p in positions.items() if MplPath(boundary).contains_point(p)}
        if inside!=set(h):
            raise ValueError(f'{record["case"]}: region {h} includes {inside}')
    for h,boundary in boundaries.items():
        for parent,parent_boundary in boundaries.items():
            if set(h)<set(parent) and not MplPath(parent_boundary).contains_points(boundary).all():
                raise ValueError(f'{record["case"]}: contour {h} protrudes from {parent}')
    for i,(x,y) in positions.items():
        ax.scatter([x],[y],s=node_size,marker='o',facecolor=NODE,edgecolor=NODE,linewidth=.9,zorder=20)
        ax.text(x,y,str(i),ha='center',va='center',
                fontsize=10.5,color='white',zorder=21)
    ax.set(xlim=(-3.0,2.79),ylim=(-1.26,1.31),aspect='equal')
    ax.axis('off')
    place_operators(ax,record['case'],boundaries,positions,operator_symbol=operator_symbol,
                    node_size=node_size,operator_positions=operator_positions)


def place_operators(ax,case,boundaries,positions,*,operator_symbol=r'\oplus',node_size=225,operator_positions=None):
    """Place the entire glyph/subscript box away from every contour and node.

    Use actual rendered text extents, not just a center-point containment test.
    No border masking/white patches: if the ribbon has insufficient space,
    drawing must fail rather than hide a mechanistic boundary behind a label.
    """
    fig=ax.figure
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    outlines={h:MplPath(ax.transData.transform(b)) for h,b in boundaries.items()}
    nodes=np.asarray([ax.transData.transform(p) for p in positions.values()])
    padding=renderer.points_to_pixels(1.3)
    node_radius=renderer.points_to_pixels(.5*np.sqrt(node_size)+1.)
    occupied=[]
    labels=[]
    for h in sorted(boundaries,key=len):
        anchors=OPERATOR_POSITIONS[case] if operator_positions is None else operator_positions
        anchor=np.asarray(anchors[h])
        # Smaller labels still need room for the lower/right source-arity index.
        label=ax.text(*anchor,rf'${operator_symbol}_{{{len(h)}}}$',ha='center',va='center',
                      fontsize=11.4,color=OPERATOR,zorder=18)
        delta=np.array(list(itertools.product(np.arange(-.64,.641,.025),repeat=2)))
        candidates=np.vstack([anchor,anchor+delta[np.argsort(np.sum(delta**2,axis=1))]])
        found=False
        for point in candidates:
            label.set_position(point)
            box=label.get_window_extent(renderer).padded(padding)
            corners=np.array([[box.x0,box.y0],[box.x1,box.y0],[box.x1,box.y1],[box.x0,box.y1]])
            if not outlines[h].contains_points(corners).all():
                continue
            if any(path.intersects_bbox(box,filled=False) for path in outlines.values()):
                continue
            dx=np.maximum(np.maximum(box.x0-nodes[:,0],nodes[:,0]-box.x1),0)
            dy=np.maximum(np.maximum(box.y0-nodes[:,1],nodes[:,1]-box.y1),0)
            if np.any(dx*dx+dy*dy < node_radius**2) or any(box.overlaps(b) for b in occupied):
                continue
            occupied.append(box)
            labels.append(label)
            found=True
            break
        if not found:
            raise ValueError(f'{case}: no unobstructed parity label position for {h}')
    # Final pass guards against an accidental change to transforms/extent rules.
    if any(path.intersects_bbox(box,filled=False) for box in occupied for path in outlines.values()):
        raise ValueError(f'{case}: parity glyph/subscript intersects a contour')
    ax._parity_geometry=(case,boundaries,positions,labels)


def verify_export_clearance(ax):
    """Check actual glyph bounds again at the final 300-dpi export resolution."""
    case,boundaries,positions,labels=ax._parity_geometry
    renderer=ax.figure.canvas.get_renderer()
    outlines=[MplPath(ax.transData.transform(b)) for b in boundaries.values()]
    boxes=[label.get_window_extent(renderer).padded(renderer.points_to_pixels(1.0)) for label in labels]
    if any(path.intersects_bbox(box,filled=False) for box in boxes for path in outlines):
        raise ValueError(f'{case}: glyph/subscript intersects a contour at export resolution')
    if any(a.overlaps(b) for a,b in itertools.combinations(boxes,2)):
        raise ValueError(f'{case}: parity labels overlap at export resolution')


def plot(records,path):
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':11,'text.color':INK,'axes.labelcolor':INK,'savefig.facecolor':'white'}):
        fig=plt.figure(figsize=(18,8.5),facecolor='white')
        gs=fig.add_gridspec(2,1,height_ratios=[1.0,2.55],left=.048,right=.986,top=.907,bottom=.060,hspace=.14)
        top=gs[0].subgridspec(1,3,wspace=.12)
        networks=[]
        for j,r in enumerate(records[:3]):
            ax=fig.add_subplot(top[j])
            network(ax,r)
            networks.append(ax)
            ax.text(.5,1.01,f'{r["case"]}  {r["case_label"]}',ha='center',va='bottom',
                    transform=ax.transAxes,fontsize=12.0,weight='bold')
        # Compact legends occupy dedicated strips outside all network data.
        handles=[Patch(fc=FILLS[2],ec=PAIR,label='Pairwise'),
                 Patch(fc=FILLS[3],ec=HIGHER,label='Higher-order'),
                 Line2D([],[],color='#8982B4',ls='--',label='Global'),
                 Line2D([],[],marker=r'$\oplus$',ls='',color=OPERATOR,markersize=11,label='Parity')]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.50,.993),ncol=4,
                   frameon=False,fontsize=10.0,handlelength=1.2,handletextpad=.4,columnspacing=1.0)
        bottom=gs[1].subgridspec(1,2,width_ratios=[3.95,1.12],wspace=.18)
        grid=bottom[0].subgridspec(2,4,hspace=.14,wspace=.14)
        for row,(r,label) in enumerate(((records[1],'Copied'),(records[4],'Shared'))):
            for col,method in enumerate(METHODS):
                ax=fig.add_subplot(grid[row,col])
                tree(ax,r,method)
                if row==0:
                    ax.set_title(NAMES[method],fontsize=12.0,pad=7,weight='bold',color=GREEN if col==0 else INK)
                if col==0:
                    ax.text(-.075,.5,label,ha='right',va='center',rotation=90,transform=ax.transAxes,
                            fontsize=11.1,color='#57636D')
        fig.text(.047,.638,'Case B',fontsize=11.3,weight='bold',color=INK)
        fig.legend(handles=[Patch(fc='#E5F1EB',ec=GREEN,label='True support'),
                            Patch(fc='#FCEFEA',ec=RED,label='Wrong clade')],
                   loc='lower center',bbox_to_anchor=(.395,.006),ncol=2,frameon=False,
                   fontsize=10.1,handlelength=1.1,columnspacing=1.2)
        score_ax=fig.add_subplot(bottom[1])
        score=np.asarray([[method_result(r,m)['clade_f1'] for m in METHODS] for r in records])
        cmap=LinearSegmentedColormap.from_list('recovery',['#FCF5F0','#C5DECF','#1D7967'])
        score_ax.imshow(score,cmap=cmap,vmin=0,vmax=1,aspect='auto',interpolation='nearest')
        for i,j in itertools.product(range(6),range(4)):
            score_ax.text(j,i,f'{score[i,j]:.2f}',ha='center',va='center',fontsize=11.3,
                          color='white' if score[i,j]>.82 else INK,weight='bold' if j==0 else 'normal')
        score_ax.set_xticks(range(4),['SPT','Pair $\Phi_R$\n+ HAC','Future MI\n+ HAC','Block $\Phi_R$*'])
        score_ax.set_yticks(range(6),[f'{mode} {case}' for mode in ('Copied','Shared') for case in 'ABC'])
        score_ax.tick_params(axis='x',top=True,labeltop=True,bottom=False,labelbottom=False,length=0,labelsize=9.2,pad=7)
        score_ax.tick_params(axis='y',length=0,labelsize=10.0)
        score_ax.axhline(2.5,color='white',lw=3)
        score_ax.set_xlabel('Support F1*',fontsize=11,labelpad=9)
        for spine in score_ax.spines.values():
            spine.set_visible(False)
        path.parent.mkdir(parents=True,exist_ok=True)
        fig.set_dpi(300)
        fig.canvas.draw()
        for ax in networks:
            verify_export_clearance(ax)
        fig.savefig(path,dpi=300,bbox_inches=None,facecolor='white')
        plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    plot(load(args.input),args.output)
    print(args.output)
