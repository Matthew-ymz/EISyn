"""Target-resolved UniCM pair synergy under the corrected independent-source TM."""
from pathlib import Path
import sys
import json
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compute_unicm_order_syn import independent_source_ei_table, TOLERANCE, ESTIMATOR_VERSION
from scripts.unicm_peid_syn_analysis import MODE_NAMES, sample_full_history_mode_inputs
from plot_unicm_mode_regions import MODES, region_center
from plot_runge_gateway_mediator_map import LAND_URL, COASTLINE_URL, load_geojson, extract_polygons, extract_lines, draw_world, format_lat_label

LEADS = (1, 8, 24)
PAIRS = tuple(itertools.combinations(range(11), 2))
OUT = ROOT / 'results/unicm_pair_hypergraph_independent'
FIG = ROOT / 'docs/reports/assets/unicm_pair_hypergraph_leads.png'


def calculate():
    OUT.mkdir(parents=True, exist_ok=True)
    history = sample_full_history_mode_inputs(n_samples=16384, intervention_bound=4., seed=20260901)
    names = tuple(MODE_NAMES)
    values = np.empty((3, 3, 55, 11))
    audits = []
    for cp in (1, 2, 3):
        p = ROOT / f'results/unicm_xi_hierarchy_uniform_n16384/cache/checkpoint{cp}_samples16384_sampling20260901_bound4_fullhist12_start0_cpu.npz'
        with np.load(p) as data:
            predictions = data['all_mode_targets']
        if predictions.shape != (16384,24,11):
            raise ValueError('Unexpected prediction cache shape')
        for li, lead in enumerate(LEADS):
            for target in range(11):
                table, audit = independent_source_ei_table(history, predictions[:,lead-1,target:target+1])
                values[cp-1,li,:,target] = [table[(names[a],names[b])] - table[(names[a],)] - table[(names[b],)] for a,b in PAIRS]
                audits.append(dict(checkpoint=cp, lead=lead, target=target, **audit))
            print(f'checkpoint={cp} lead={lead} complete', flush=True)
    audit = dict(estimator=ESTIMATOR_VERSION, approximation='degree-1 TM / Gaussian readout', n_samples=16384,
                 sampling_seed=20260901, intervention_bound=4., start_month=0, history_months=12,
                 checkpoints=[1,2,3], leads=list(LEADS), target='one scalar future mode',
                 tolerance_bits=TOLERANCE, minimum_bits=float(values.min()),
                 affected_count=int((values < -TOLERANCE).sum()),
                 tolerance_negative_count=int(((values<0)&(values>=-TOLERANCE)).sum()),
                 projection='none', count=int(values.size), fits=audits)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    if not np.isfinite(values).all() or audit['affected_count']:
        raise ValueError(f"Syn violation: minimum={values.min()}, threshold={-TOLERANCE}, affected_count={audit['affected_count']}")
    np.savez_compressed(OUT/'pair_syn.npz', values_bits=values, pairs=PAIRS, leads=LEADS, checkpoints=[1,2,3], metadata=json.dumps({k:v for k,v in audit.items() if k!='fits'}))
    return values


def select(values):
    mean = values.mean(axis=0)
    selected = []
    for li, lead in enumerate(LEADS):
        rows = [dict(lead=lead, a=a,b=b,t=t,mean_bits=float(mean[li,pi,t]),
                     std_bits=float(values[:,li,pi,t].std(ddof=1)))
                for pi,(a,b) in enumerate(PAIRS) for t in range(11) if t not in (a,b)]
        selected.append(sorted(rows, key=lambda r:-r['mean_bits'])[:10])
    (OUT/'displayed_edges.json').write_text(json.dumps(dict(selection='top 10 of 495 distinct-source/target triples per lead; equal checkpoint mean; no significance filtering',panels=selected),indent=2)+'\n')
    return selected


def render(selected):
    land, coast = extract_polygons(load_geojson(LAND_URL)), extract_lines(load_geojson(COASTLINE_URL))
    if not land or not coast:
        raise RuntimeError('Geography unavailable')
    fig = plt.figure(figsize=(18,4.7),facecolor='white')
    axes = [fig.add_axes([0.025+i*0.327,0.235,0.31,0.70],projection='mollweide') for i in range(3)]
    for li,ax in enumerate(axes):
        draw_world(ax,land,coast)
        ax.set_xticks([])
        ticks = np.arange(-60,61,30)
        ax.set_yticks(np.radians(ticks))
        ax.set_yticklabels([format_lat_label(t) for t in ticks],fontsize=6)
        ax.grid(axis='y',color='#b8b8b8',ls='--',lw=.4,alpha=.7)
        ax.spines['geo'].set_linewidth(.9)
        ax.set_title(
            f'Lead = {LEADS[li]} ' + ('month' if LEADS[li] == 1 else 'months'),
            fontsize=11,
            pad=12,
        )
        ax.text(
            -0.06,
            1.02,
            chr(97 + li),
            transform=ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=11,
            fontweight='bold',
            color='#111111',
            clip_on=False,
        )
    fig.canvas.draw()
    vmax = max(r['mean_bits'] for rows in selected for r in rows)
    for ax,rows in zip(axes,selected):
        positions = []
        for mode in MODES:
            lon,lat = np.array([region_center(r) for r in mode.regions]).mean(axis=0)
            xy = ax.transData.transform(np.radians([lon,lat]))
            # Same schematic displacement as the approved basemap; no locator lines.
            offset = {9:(-3,13),10:(-8,-17)}.get(mode.index,(0,0))
            xy += np.array(offset)*fig.dpi/72
            positions.append(ax.transAxes.inverted().transform(xy))
        for idx,r in enumerate(rows):
            a,b,t = [positions[r[k]] for k in ('a','b','t')]
            mid=(a+b)/2
            direction=t-mid
            perpendicular=np.array([-direction[1],direction[0]])
            perpendicular/=max(np.linalg.norm(perpendicular),1e-9)
            hub=.58*mid+.42*t + (-1 if idx%2 else 1)*(.04+.012*(idx//2))*perpendicular
            strength=r['mean_bits']/vmax
            lw=.5+2.3*strength
            alpha=.22+.56*strength
            for source in (a,b):
                ax.plot([source[0],hub[0]],[source[1],hub[1]],transform=ax.transAxes,color='#8056a5',lw=.75*lw,alpha=alpha*.65,zorder=3)
            ax.add_patch(FancyArrowPatch(hub,t,transform=ax.transAxes,arrowstyle='-|>',mutation_scale=8,shrinkB=8.5,shrinkA=2,connectionstyle=f'arc3,rad={.08 if idx%2 else -.08}',lw=lw,color='#8056a5',alpha=alpha,zorder=3.5))
            ax.scatter(*hub,transform=ax.transAxes,s=13+13*strength,c='#8056a5',edgecolors='white',linewidths=.35,zorder=5)
        for i,xy in enumerate(positions):
            ax.text(*xy,str(i),transform=ax.transAxes,ha='center',va='center',fontsize=8,color='white',zorder=6,bbox=dict(boxstyle='circle,pad=.38',fc='#3976a1',ec='#263e4c',lw=.85))
    for i,mode in enumerate(MODES):
        fig.text(.04+i*.085,.15,f'{i}  {mode.name}',fontsize=9,color='#263e4c',ha='left')
    fig.text(.025,.075,'Top 10 per lead · distinct source / target modes · mean of 3 checkpoints · 12-month source histories',fontsize=9,color='#444444')
    for i,value in enumerate([.25*vmax,.5*vmax,vmax]):
        x=.66+i*.10
        line=plt.Line2D([x,x+.025],[.075,.075],transform=fig.transFigure,color='#8056a5',lw=.5+2.3*value/vmax,alpha=.22+.56*value/vmax)
        fig.add_artist(line)
        fig.text(x+.03,.075,f'{value:.4f} bits',fontsize=8,va='center')
    fig.text(.025,.018,'Affine TM with independent-source prior; hub = two-source hyperedge; arrow = target. Ranked strengths, not significance tests.',fontsize=8,color='#666666')
    FIG.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG,dpi=260,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
    print(FIG)


if __name__ == '__main__':
    render(select(calculate()))
