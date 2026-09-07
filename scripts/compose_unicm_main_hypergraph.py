"""Combine synchronized UniCM intervention panels and observational validation."""
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/reports/assets/unicm_main_with_hypergraph.png'

def main():
    tree=json.loads((ROOT/'results/unicm_xi_hierarchy_uniform_n16384/summary.json').read_text())
    shap=json.loads((ROOT/'results/unicm_11mode_shapley_affine_n16384_independent/summary.json').read_text())
    hyper=json.loads((ROOT/'results/unicm_pair_hypergraph_independent/audit.json').read_text())
    with np.load(ROOT/'results/unicm_spt_order_mass/order_mass.npz') as d:
        order=json.loads(str(d['metadata']))
        order_tolerance=float(order['syn_tolerance_bits'])
    keys=['n_samples','sampling_seed','intervention_bound','start_month','estimator']
    assert all(order[k]==hyper[k] for k in keys)
    assert all(tree[k]==hyper[k] for k in keys)
    assert all(shap['method'][k]==hyper[k] for k in keys)
    tolerances = {
        'order_bits': order_tolerance,
        'spt_bits': float(tree['syn_tolerance_bits']),
        'hypergraph_bits': float(hyper['tolerance_bits']),
        'shapley_bits': float(shap['audit']['syn_nonnegative_tolerance_bits']),
    }
    assert len(set(tolerances.values())) == 1
    caches=[]
    for cp in [1,2,3]:
        p=ROOT/f'results/unicm_xi_hierarchy_uniform_n16384/cache/checkpoint{cp}_samples16384_sampling20260901_bound4_fullhist12_start0_cpu.npz'
        with np.load(p) as d:
            metadata=json.loads(str(d['metadata']))
            assert d['all_mode_targets'].shape==(16384,24,11)
            assert all(metadata[k]==hyper[k] for k in ['n_samples','sampling_seed','intervention_bound','start_month'])
        caches.append(dict(checkpoint=cp,path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    report=dict(all_intervention_panels_same_samples_and_parameters=True,
        synchronized_intervention_parameters={k:hyper[k] for k in keys},
        synchronized_nonnegative_tolerance=tolerances,
        shared_prediction_caches=caches,
        sample_identity_evidence='same deterministic sampler, full seed/configuration and exact prediction cache files; cached source arrays are not stored',
        tree=dict(n_samples=tree['n_samples'],sampling_seed=tree['sampling_seed'],checkpoint_displayed=2,algorithm=tree['estimator']),
        shapley=shap['method'],
        target_definitions=dict(hypergraph='one scalar future mode',spt_order_shapley='all 11 future modes jointly'),
        forecast_validation=dict(kind='observational held-out prediction evaluation',fit_samples=253,validation_samples=36,test_samples=96,note='Not an intervention-sample experiment; its real-data split is intentionally not forced to n=16,384.'),
        action='SPT, SPT-selected order mass, exact Shapley, and pair hypergraphs share the corrected estimator and identical intervention samples.')
    audit=ROOT/'results/unicm_pair_hypergraph_independent/composite_provenance_audit.json'
    audit.write_text(json.dumps(report,indent=2)+'\n')
    main_image=plt.imread(ROOT/'fig/earth_unicm_hierarchical_ei.png')
    maps=plt.imread(ROOT/'docs/reports/assets/unicm_pair_hypergraph_leads.png')
    main_image=main_image.copy()
    ih,iw=main_image.shape[:2]
    main_image=main_image[int(.055*ih):]
    maps=maps[int(.075*maps.shape[0]):int(.705*maps.shape[0])]
    w=13.
    hm=w*main_image.shape[0]/main_image.shape[1]
    hg=2.25
    height=hm+hg+.32
    fig=plt.figure(figsize=(w,height),facecolor='white')
    ax=fig.add_axes([0,0,1,hm/height]);ax.imshow(main_image);ax.axis('off')
    panel_lefts=(.09,.37,.65)
    for i,(lead,left,label) in enumerate(zip((1,8,24), panel_lefts, ('a','b','c'), strict=True)):
        panel=maps[:,int(i*maps.shape[1]/3):int((i+1)*maps.shape[1]/3)]
        ax=fig.add_axes([left,(hm+.03)/height,.28,hg/height]);ax.imshow(panel);ax.axis('off')
        fig.text(left+.14,(hm+hg+.18)/height,
                 f'$\\ell = {lead}$ '+('month' if lead==1 else 'months'),
                 fontsize=10,ha='center',va='center')
        fig.text(left-.025,(hm+hg-.1)/height,label,fontsize=10,weight='bold')
    # The three SPTs occupy the lower composite row in the same left-to-right order.
    for left,label in zip(panel_lefts, ('d','e','f'), strict=True):
        fig.text(left-.025,.79*hm/height,label,fontsize=10,weight='bold')
    fig.savefig(OUT,dpi=240,bbox_inches='tight',pad_inches=.05)
    plt.close(fig)
    print(OUT)

if __name__=='__main__': main()
