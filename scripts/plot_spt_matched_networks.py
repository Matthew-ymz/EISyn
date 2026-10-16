"""Draw the two frozen matched-case networks in the agreed green/ribbon style.

Topology comes from frozen experiment inputs, never from a recovered tree.
Red/blue encode arity, outline width is proportional to K, and region area is
only schematic. Original user artwork and earlier plotting sources are retained.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.plot_spt_kuramoto_reference_networks import (
    plt,np,Patch,Line2D,FILLS,PAIR,HIGHER,POSITIONS,ANCHORS,network,
    apply_coupling_widths,validate_rendered,LINEWIDTH_PER_COUPLING,
)
from scripts.plot_spt_kuramoto_simplified_networks import edges

OUT=ROOT/'docs/reports/assets/kuramoto_hierarchy/spt_matched_networks.png'
DISPLAY_NETWORK={'A':'A','C':'B'}  # Presentation only; frozen scientific keys stay unchanged.


def frozen_networks():
    networks=edges()
    sources=[]
    for case in 'AC':
        path=ROOT/f'results/spt_local_kuramoto_tm/ac_comparison/{case}/frozen_model.json'
        frozen=json.loads(path.read_text())
        if {tuple(e['edge']):e['weight'] for e in frozen['network']}!=networks[case]:
            raise ValueError(f'{case}: displayed supports differ from frozen experiment')
        sources.append(path)
    path=ROOT/'results/spt_matched_three_case/glv_simplified_A/pilot.json'
    glv=json.loads(path.read_text())['generator']['weights']
    if glv!={','.join(map(str,e)):w for e,w in networks['A'].items()}:
        raise ValueError('HO-LV and Kuramoto A do not have identical weighted supports')
    sources.append(path)
    return networks,sources


def main():
    networks,sources=frozen_networks()
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':10,'text.color':'#27343D','svg.fonttype':'none'}):
        fig=plt.figure(figsize=(12.6,3.65),dpi=300,facecolor='white')
        grid=fig.add_gridspec(1,2,left=.012,right=.988,bottom=.025,top=.75,wspace=.075)
        axes=[]
        for index,case in enumerate('AC'):
            ax=fig.add_subplot(grid[index]);supports=tuple(networks[case])
            primitive=tuple(e for e in supports if not any(set(g)<set(e) for g in supports))
            network(ax,{'case':case,'hyperedges':supports,'planted_bottom':primitive},
                    operator_symbol='H',node_size=360,positions=POSITIONS[case],
                    operator_positions=ANCHORS[case])
            apply_coupling_widths(ax,networks[case])
            ax.text(.5,1.105,DISPLAY_NETWORK[case],ha='center',va='bottom',transform=ax.transAxes,
                    fontsize=12,weight='bold')
            subtitle='Kuramoto + HO-LV' if case=='A' else 'Kuramoto'
            ax.text(.5,1.025,subtitle,ha='center',va='bottom',transform=ax.transAxes,fontsize=9)
            axes.append(ax)
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
        for ax,case in zip(axes,'AC'):validate_rendered(ax,networks[case])
        provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        description='Frozen matched networks displayed as A/B (native keys A/C). H_m denotes arity; outline width=2.4 K pt. Region area/overlap do not encode strength. Network A shared by Kuramoto and HO-LV.'
        OUT.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(OUT,dpi=300,bbox_inches='tight',pad_inches=.055,facecolor='white',metadata={
            'Description':description,'Sources':json.dumps(provenance,sort_keys=True)})
        fig.savefig(OUT.with_suffix('.svg'),bbox_inches='tight',pad_inches=.055,facecolor='white',metadata={
            'Description':description+' Sources: '+json.dumps(provenance,sort_keys=True)})
        plt.close(fig)
    print(OUT)
    print(OUT.with_suffix('.svg'))
    print('Frozen weighted supports, exact envelope membership, labels and outline widths verified.')


if __name__=='__main__':main()
