"""Two network schematics, all three case trees, and five-method F1 summary.

Read-only presentation of the final native summary. Display B maps to native C;
the former experimental deep-cascade B is never loaded. No estimator is run.
Canonical stored mean trees are shown, including the explicitly provisional
SURD mean and user-authorized precision-failed diagnostic trees marked double dagger.
Only complete stored outputs are used; the LV Block diagnostic is the coarser grid.
Manuscript rediscovered 2026-10-16: P6UJCVG8, sole main DXGC7JEA and SI
MWIWKSVG, Methods/S5.1 and Algorithm S1 read; explicit draft date unavailable.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.plot_spt_matched_networks import frozen_networks,DISPLAY_NETWORK
from scripts.plot_spt_kuramoto_reference_networks import (
    plt,np,Patch,Line2D,FILLS,PAIR,HIGHER,POSITIONS,ANCHORS,network,
    apply_coupling_widths,validate_rendered,LINEWIDTH_PER_COUPLING,
)
from scripts.spt_local_hyperedge_kuramoto import SUPPORTS
from scripts.summarize_spt_matched_comparison import displayed_metrics
from matplotlib.colors import LinearSegmentedColormap

SOURCE=ROOT/'results/spt_matched_three_case/full_comparison/summary.json'
OUT=ROOT/'docs/reports/assets/kuramoto_hierarchy/spt_matched_cases_composite.png'
METHODS=('spt','surd','block_phi_r','wms_natural','shap_profiles')
HEADERS=('SPT','SURD\n+ max-weight tree',r'Block $\Phi_R$','WMS\n(natural sampling)','SHAP\noutput-profile tree')
ROW_LABELS=('Kuramoto\nNetwork A','HO-LV\nNetwork A','Kuramoto\nNetwork B')
NATIVE_CASES=('A','A','C')
INK,GREEN,RED,GRAY='#24323C','#217B68','#B74738','#ABB6BF'


def draw_tree(ax,record,case):
    metric=displayed_metrics(record)
    if not metric:
        reason='Incomplete\nestimation' if record['status']=='budget_exhausted' else 'Numerically\nunresolved'
        ax.text(.5,.53,'N/A',ha='center',va='center',transform=ax.transAxes,fontsize=9,color='#66717B')
        ax.text(.5,.36,reason,ha='center',va='center',transform=ax.transAxes,fontsize=7.5,color='#87939C')
        ax.axis('off');ax._tree_labels=[];return
    sources=tuple(range(1,10));clades={tuple(sorted(map(int,c))) for c in metric['clades']}
    all_sets=clades|{sources}|{(i,) for i in sources};children={}
    for s in all_sets:
        if len(s)==1:children[s]=();continue
        proper=[c for c in all_sets if set(c)<set(s)]
        maximal=[c for c in proper if not any(set(c)<set(d) for d in proper)]
        if len(maximal)!=2 or set(maximal[0])&set(maximal[1]) or set().union(*map(set,maximal))!=set(s):
            raise ValueError(f'Incomplete or invalid native binary hierarchy at {s}')
        children[s]=tuple(sorted(maximal))
    if {frozenset(c) for c in children[sources]}!={frozenset(c) for c in metric['root_split']}:
        raise ValueError('Native clades disagree with stored root')
    truth={tuple(sorted(c)) for c in SUPPORTS[case]}
    primitive=[s for s in truth if not any(set(t)<set(s) for t in truth)]
    def refinement(s):return any(set(s)<set(p) for p in primitive)
    scored={s for s in clades if not refinement(s)};reference=truth-{sources}
    recovered=2*len(scored&reference)/(len(scored)+len(reference))
    if not np.isclose(recovered,metric['clade_f1'],rtol=0,atol=1e-12):
        raise ValueError('Tree module colors and stored module F1 are inconsistent')
    positions={};leaf_order=[]
    def visit(s,depth=0):
        if children[s]:x=float(np.mean([visit(c,depth+1) for c in children[s]]))
        else:x=len(leaf_order);leaf_order.append(s[0])
        positions[s]=(x,-depth);return x
    visit(sources)
    if sorted(leaf_order)!=list(sources):raise ValueError('Omitted or repeated identity label')
    labels=[]
    for s in sorted(all_sets,key=lambda x:(-len(x),x)):
        x,y=positions[s]
        for c in children[s]:
            cx,cy=positions[c];wrong=len(c)>1 and c not in truth and not refinement(c)
            ax.plot([x,cx],[y,cy],color=RED if wrong else GRAY,lw=.75,
                    ls=(0,(3,2)) if wrong else '-',zorder=0)
        if s==sources:edge,face,label='#78858F','#EFF2F4','1–9'
        elif len(s)==1 or refinement(s):edge,face,label='#B7C0C6','white',''.join(map(str,s))
        elif s in truth:edge,face,label=GREEN,'#E5F1EB',''.join(map(str,s))
        else:edge,face,label=RED,'#FCEFEA',''.join(map(str,s))
        labels.append(ax.text(x,y,label,ha='center',va='center',fontsize=7.0,color=INK,
            bbox={'boxstyle':'round,pad=.12','fc':face,'ec':edge,'lw':.7},zorder=2))
    depth=min(y for _,y in positions.values())
    ax.set(xlim=(-.48,8.48),ylim=(depth-.5,.55));ax.axis('off');ax._tree_labels=labels
    marker=record.get('marker','')
    if marker:ax.text(.99,1.0,marker,ha='right',va='top',transform=ax.transAxes,fontsize=9,color='#66717B')


def verify_tree_text(ax,renderer):
    boxes=[t.get_bbox_patch().get_window_extent(renderer).padded(.5) for t in ax._tree_labels]
    for i,a in enumerate(boxes):
        for b in boxes[i+1:]:
            if a.overlaps(b):raise ValueError('Tree node boxes overlap at export resolution')
    if any(not ax.bbox.contains(b.x0,b.y0) or not ax.bbox.contains(b.x1,b.y1) for b in boxes):
        raise ValueError('Tree labels extend outside their allocated panel')


def main():
    snapshot=json.loads(SOURCE.read_text());data=snapshot['records']
    if [c['case'] for c in data]!=['Kuramoto A','HO-LV A','Kuramoto C']:
        raise ValueError('Unexpected case identity or order')
    if not snapshot['finalization']['all_workers_stopped']:raise ValueError('Final snapshot required')
    networks,network_sources=frozen_networks()
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':9,'text.color':INK,'svg.fonttype':'none'}):
        fig=plt.figure(figsize=(18,10.2),dpi=300,facecolor='white')
        outer=fig.add_gridspec(2,1,left=.075,right=.988,bottom=.10,top=.94,
                              height_ratios=(1,2.20),hspace=.20)
        top=outer[0].subgridspec(1,2,wspace=.16);network_axes=[]
        for col,case in enumerate('AC'):
            ax=fig.add_subplot(top[col]);supports=tuple(networks[case])
            primitive=tuple(e for e in supports if not any(set(g)<set(e) for g in supports))
            network(ax,{'case':case,'hyperedges':supports,'planted_bottom':primitive},
                    operator_symbol='H',node_size=360,positions=POSITIONS[case],operator_positions=ANCHORS[case])
            apply_coupling_widths(ax,networks[case])
            ax.text(.5,1.04,DISPLAY_NETWORK[case],ha='center',va='bottom',transform=ax.transAxes,fontsize=12,weight='bold')
            network_axes.append(ax)
        points=np.concatenate([b for ax in network_axes for b in ax._parity_geometry[1].values()])
        for ax in network_axes:
            ax.set(xlim=(points[:,0].min()-.12,points[:,0].max()+.12),
                   ylim=(points[:,1].min()-.14,points[:,1].max()+.17))
        fig.legend(handles=[Patch(fc=FILLS[2],ec=PAIR,label='Pairwise'),
                            Patch(fc=FILLS[3],ec=HIGHER,label='Three-body'),
                            *[Line2D([],[],color='#424B52',lw=LINEWIDTH_PER_COUPLING*w,label=rf'$K={w:g}$') for w in (1.,.5,.25)]],
                   loc='upper center',bbox_to_anchor=(.535,.996),ncol=5,frameon=False,
                   fontsize=8.5,handlelength=1.9,columnspacing=1.6)
        bottom=outer[1].subgridspec(1,2,width_ratios=(5.25,1.10),wspace=.25)
        trees=bottom[0].subgridspec(3,5,hspace=.20,wspace=.15);tree_axes=[]
        for row,(case,record) in enumerate(zip(NATIVE_CASES,data)):
            for col,method in enumerate(METHODS):
                ax=fig.add_subplot(trees[row,col]);draw_tree(ax,record['methods'][method],case);tree_axes.append(ax)
                if row==0:ax.set_title(HEADERS[col],fontsize=9,pad=12,color=GREEN if method=='spt' else INK,weight='bold' if method=='spt' else 'normal')
                if col==0:
                    box=ax.get_position();fig.text(.012,(box.y0+box.y1)/2,ROW_LABELS[row],ha='left',va='center',fontsize=9,color='#57636D')
        score=fig.add_subplot(bottom[1]);box=score.get_position()
        score.set_position([box.x0,box.y0+.12*box.height,box.width,.76*box.height])
        values=np.array([[(displayed_metrics(c['methods'][m]) or {}).get('clade_f1',np.nan) for c in data] for m in METHODS])
        cmap=LinearSegmentedColormap.from_list('module_recovery',['#FCF5F0','#C5DECF','#1D7967']);cmap.set_bad('#E9ECEF')
        score.imshow(values,cmap=cmap,vmin=0,vmax=1,aspect='auto',interpolation='nearest')
        for i,method in enumerate(METHODS):
            for j,c in enumerate(data):
                value=values[i,j];label=f'{value:.2f}{c["methods"][method].get("marker","")}' if np.isfinite(value) else 'N/A'
                score.text(j,i,label,ha='center',va='center',fontsize=8,color='white' if value>.82 else INK,weight='bold' if value==1 else 'normal')
        score.set_xticks(range(3),['Kuramoto\nA','HO-LV\nA','Kuramoto\nB'])
        score.xaxis.tick_top();score.set_yticks(range(5),['SPT','SURD','Block $\Phi_R$','WMS','SHAP'])
        score.tick_params(length=0,labelsize=8,pad=7);score.set_xlabel('Module-reference F1',fontsize=9,labelpad=10)
        for spine in score.spines.values():spine.set_visible(False)
        fig.legend(handles=[Patch(fc='#E5F1EB',ec=GREEN,label='Reference module'),
                            Patch(fc='#FCEFEA',ec=RED,label='Wrong clade'),
                            Patch(fc='white',ec='#B7C0C6',label='Root / leaf / within-primitive refinement')],
                   loc='lower center',bbox_to_anchor=(.435,.059),ncol=3,frameon=False,fontsize=8,handlelength=1.1,columnspacing=1.5)
        fig.text(.075,.037,'Trees: stored canonical means. * Sensitivity unresolved. † SURD: 2/4 full tables. ‡ Precision not passed; diagnostic display.',fontsize=7.4,color='#66717B')
        fig.text(.075,.019,'‡ LV Block uses completed N256/P256; high grid incomplete. One fitted dataset per case; numerical repeats are not independent experiments.',fontsize=7.4,color='#66717B')
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        for ax,case in zip(network_axes,'AC'):validate_rendered(ax,networks[case])
        for ax in tree_axes:verify_tree_text(ax,renderer)
        provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE,*network_sources]}
        description='Matched three-case comparison: displayed Network B=native simplified C. All five methods. Double dagger marks user-authorized precision-failed complete diagnostic trees; LV Block uses completed coarse grid. No numerical gates are reclassified as passed. '+json.dumps(provenance,sort_keys=True)
        fig.savefig(OUT,dpi=300,bbox_inches='tight',pad_inches=.055,facecolor='white',metadata={'Description':description})
        fig.savefig(OUT.with_suffix('.svg'),bbox_inches='tight',pad_inches=.055,facecolor='white',metadata={'Description':description})
        plt.close(fig)
    print(OUT);print(OUT.with_suffix('.svg'));print('All 15 method/case panels verified against stored results; exact network memberships and text clearance passed.')


if __name__=='__main__':main()
