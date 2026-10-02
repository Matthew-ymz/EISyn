#!/usr/bin/env python3
"""Analyze the frozen individual-SC pilot; subjects, not pairs, are summary units."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.dmf_subject_consistency import jaccard, transition_interval
from scripts.run_dmf_subject_consistency import BASE, atomic_json

METHODS=['multisource','pairwise','whole_ei','sc_strength']
COLORS=['#B64D64','#287C9C','#BA8F34','#737B85']
METHOD_LABELS=['Multisource cross-ROI Shapley','Pairwise proxy Shapley','Whole EI Shapley','SC strength']
SHORT_NETWORKS=['Visual','SomMot','DAN','SVAN','Limbic','Control','DMN']


def rank(a,b):
    return float(spearmanr(a,b).statistic)


def top(a,k):
    return np.argsort(-np.asarray(a),kind='stable')[:k].tolist()


def loso(v,k=8):
    correlation,overlap=[],[]
    for i in range(len(v)):
        consensus=np.delete(v,i,axis=0).mean(0)
        correlation.append(rank(consensus,v[i]))
        overlap.append(jaccard(top(consensus,k),top(v[i],k)))
    return np.array(correlation),np.array(overlap)


def independent_loso(v,k=8):
    """All three rotations: training persons use the other two seeds only.

    Source/noise draws in the held subject never occur in the training consensus.
    Average rotations within person so subjects remain the summary units.
    """
    correlation=np.empty((len(v),3));overlap=np.empty_like(correlation)
    for i in range(len(v)):
        for si in range(3):
            consensus=np.delete(np.delete(v,i,axis=0),si,axis=1).mean(axis=(0,1))
            correlation[i,si]=rank(consensus,v[i,si])
            overlap[i,si]=jaccard(top(consensus,k),top(v[i,si],k))
    return correlation.mean(1),overlap.mean(1)


def matched_null(core, target, membership, strength, rng, draws=512):
    """Within-person matched random sets: size, hemisphere/Yeo, strength tertile.

    Joint cells preserve their exact counts. Small cells can force inclusion;
    this is retained and recorded, not relaxed to get a more favorable null.
    """
    if not core or not target:
        return None
    # Stable ranks avoid collapsing strength bins at tied values.
    ordered=np.argsort(strength,kind='stable');bins=np.empty(100,dtype=int)
    bins[ordered]=np.arange(100)*3//100
    cells=[(int(i>=50),int(membership[i]),int(bins[i])) for i in range(100)]
    counts={cell:sum(cells[i]==cell for i in core) for cell in set(cells[i] for i in core)}
    pools={cell:np.array([i for i,c in enumerate(cells) if c==cell]) for cell in counts}
    scores=[]
    for _ in range(draws):
        selected=[int(i) for cell,k in counts.items() for i in rng.choice(pools[cell],k,replace=False)]
        scores.append(jaccard(selected,target))
    forced=sum(k for cell,k in counts.items() if len(pools[cell])==k)
    return dict(mean_jaccard=float(np.mean(scores)),sd_jaccard=float(np.std(scores,ddof=1)),
                forced_member_count=forced,draws=draws)


def load(base, organization=True):
    contract=json.loads((base/'contract.json').read_text())
    ids=contract['subject_ids'];gs=np.array(contract['G']);seeds=contract['seeds']
    with np.load(base/'inputs.npz') as a:
        rho=a['spectral_radius'];sc=a['connectivity'];membership=a['network_membership'];labels=a['labels']
    keys=['xi_nats','whole_ei_nats','cross_roi_nats','roi_local_xi_nats','cross_network_nats','legacy_xi_nats']
    metrics={k:np.empty((9,3,7)) for k in keys}
    within=np.empty((9,3,7,7));between=np.empty_like(within)
    dyn=np.empty((9,3,7));susceptibility=np.empty_like(dyn)
    validity=[]
    for mi,subject in enumerate(ids):
        for si,seed in enumerate(seeds):
            for gi,g in enumerate(gs):
                with np.load(base/'conditions'/f'{subject}_G{g:.2f}_seed{seed}.npz') as a:
                    record=json.loads(str(a['metrics_json']))
                    for k in keys:metrics[k][mi,si,gi]=record[k]
                    within[mi,si,gi]=a['within_network_xi'];between[mi,si,gi]=a['between_network_shapley']
                    validity.append(dict(subject=subject,G=float(g),seed=seed,metrics=record,
                        state=json.loads(str(a['diagnostics_json'])),network=json.loads(str(a['network_audit_json']))))
        with np.load(base/'dynamics'/f'{subject}.npz') as a:
            dyn[mi]=a['mean_rate_hz'];susceptibility[mi]=a['susceptibility']
    if not organization:
        return dict(contract=contract,ids=ids,g=gs,rho=rho,sc=sc,membership=membership,labels=labels,
                    metrics=metrics,within=within,between=between,dyn=dyn,susceptibility=susceptibility)
    og=np.array(contract['organization_G'])
    ranking=np.empty((4,9,3,len(og),100));mcse=np.empty((9,3,len(og),100))
    trees={};organization_audits=[]
    for mi,subject in enumerate(ids):
        for si,seed in enumerate(seeds):
            for gi,g in enumerate(og):
                with np.load(base/'organization'/f'{subject}_G{g:.2f}_seed{seed}.npz') as a:
                    for ni,key in enumerate(['roi_cross_shapley','pair_shapley','whole_ei_shapley','sc_strength']):
                        ranking[ni,mi,si,gi]=a[key]
                    mcse[mi,si,gi]=a['roi_mc_se']
                    trees[mi,si,gi]=json.loads(str(a['trees_json']))
                    organization_audits.append(dict(subject=subject,G=float(g),seed=seed,
                        shapley=json.loads(str(a['shapley_audit_json'])),
                        values=json.loads(str(a['audit_json'])),pair=json.loads(str(a['pair_audit_json'])),
                        trees={name:dict(audit=t['audit'],closure_error_nats=t['closure_error_nats'])
                               for name,t in trees[mi,si,gi].items()}))
    return dict(contract=contract,ids=ids,g=gs,og=og,rho=rho,sc=sc,membership=membership,labels=labels,
                metrics=metrics,within=within,between=between,dyn=dyn,susceptibility=susceptibility,
                ranking=ranking,mcse=mcse,trees=trees,validity=validity,organization_audits=organization_audits)


def summaries(data):
    d=data;g=d['g'];og=d['og'];m=d['metrics'];n=8
    transitions=[transition_interval(g,r.mean(0)) for r in d['dyn']]
    rows=[]
    for i,subject in enumerate(d['ids']):
        xi=m['xi_nats'][i].mean(0);pi=int(np.argmax(xi))
        rows.append(dict(subject=subject,spectral_radius=float(d['rho'][i]),
            peak_G=float(g[pi]),peak_xi_nats=float(xi[pi]),peak_interior=0<pi<len(g)-1,
            seed_peak_G=[float(g[j]) for j in np.argmax(m['xi_nats'][i],axis=1)],
            transition=transitions[i],zero_cross_roi_nats=float(m['cross_roi_nats'][i,:,0].mean()),
            noise_to_G_amplitude=float(np.sqrt(np.mean(m['xi_nats'][i].var(0,ddof=1)))/np.ptp(xi))))
    ranks=np.empty((4,4,8));overlap=np.empty_like(ranks);repeat=np.empty_like(ranks)
    independent_ranks=np.empty_like(ranks);independent_overlap=np.empty_like(ranks)
    core_overlap=np.empty((2,4,8));core_null=np.empty_like(core_overlap);core_repeat=np.empty_like(core_overlap)
    independent_core=np.empty_like(core_overlap);independent_core_null=np.empty_like(core_overlap)
    rng=np.random.default_rng(202610021)
    core_rows=[];core_consensus=[]
    for ni in range(4):
        for gi in range(4):
            ranks[ni,gi],overlap[ni,gi]=loso(d['ranking'][ni,:8,:,gi].mean(1))
            independent_ranks[ni,gi],independent_overlap[ni,gi]=independent_loso(d['ranking'][ni,:8,:,gi])
            for i in range(8):
                repeat[ni,gi,i]=np.mean([rank(d['ranking'][ni,i,a,gi],d['ranking'][ni,i,b,gi])
                                       for a,b in combinations(range(3),2)])
    for ni,name in enumerate(METHODS[:2]):
        for gi,gval in enumerate(og):
            cores=[d['trees'][i,1,gi][name]['core10'] for i in range(8)] # frozen primary seed4
            counts=Counter(tuple(sorted(c)) for c in cores if c)
            modal,modes=counts.most_common(1)[0] if counts else ((),0)
            core_consensus.append(dict(method=name,G=float(gval),
                complete_core_modal_count=modes,complete_core_distinct_count=len(counts),
                modal_core_members=list(modal),sizes=[len(c) for c in cores],
                member_frequencies=[sum(i in c for c in cores)/8 for i in range(100)]))
            for i in range(8):
                raw=[];null=[];forced=[]
                for j in range(8):
                    if j==i:continue
                    raw.append(jaccard(cores[i],cores[j]))
                    result=matched_null(cores[i],cores[j],d['membership'],d['sc'][i].sum(1),rng)
                    null.append(result['mean_jaccard']);forced.append(result['forced_member_count'])
                core_overlap[ni,gi,i]=np.mean(raw);core_null[ni,gi,i]=np.mean(null)
                # Frozen primary core seed4 against other people's independent seed3/5.
                independent_raw=[];independent_null=[]
                for j in range(8):
                    if j==i:continue
                    for sj in (0,2):
                        target=d['trees'][j,sj,gi][name]['core10']
                        independent_raw.append(jaccard(cores[i],target))
                        independent_null.append(matched_null(cores[i],target,d['membership'],
                                                             d['sc'][i].sum(1),rng)['mean_jaccard'])
                independent_core[ni,gi,i]=np.mean(independent_raw)
                independent_core_null[ni,gi,i]=np.mean(independent_null)
                core_repeat[ni,gi,i]=np.mean([jaccard(d['trees'][i,a,gi][name]['core10'],d['trees'][i,b,gi][name]['core10'])
                                            for a,b in combinations(range(3),2)])
                members=cores[i]
                core_rows.append(dict(method=name,G=float(gval),subject=d['ids'][i],
                    members=members,labels=[str(d['labels'][r]) for r in members],
                    size=len(members),mean_other_subject_jaccard=float(core_overlap[ni,gi,i]),
                    matched_null_jaccard=float(core_null[ni,gi,i]),
                    independent_seed_other_subject_jaccard=float(independent_core[ni,gi,i]),
                    independent_seed_matched_null_jaccard=float(independent_core_null[ni,gi,i]),
                    matched_null_forced_member_count=int(np.mean(forced)),
                    same_person_new_seed_jaccard=float(core_repeat[ni,gi,i])))
    network_sizes=np.bincount(d['membership']);gi=int(np.flatnonzero(np.isclose(g,1.3))[0])
    w=d['within'][:8,:,gi].mean(1);b=d['between'][:8,:,gi].mean(1)
    perroi=w/network_sizes;opportunities=network_sizes*(100-network_sizes);peropp=b/opportunities
    network_rows=[];aligned_networks=[]
    assoc=[2,3,5,6]
    for i in range(8):
        between_sc=np.array([d['sc'][i][np.ix_(np.flatnonzero(d['membership']==h),
                                      np.flatnonzero(d['membership']!=h))].sum() for h in range(7)])
        network_rows.append(dict(subject=d['ids'][i],within_nats=w[i].tolist(),
            between_shapley_nats=b[i].tolist(),within_per_roi=perroi[i].tolist(),
            between_per_opportunity=peropp[i].tolist(),
            top_within=SHORT_NETWORKS[int(np.argmax(w[i]))],top_within_per_roi=SHORT_NETWORKS[int(np.argmax(perroi[i]))],
            top_between=SHORT_NETWORKS[int(np.argmax(b[i]))],top_between_per_opportunity=SHORT_NETWORKS[int(np.argmax(peropp[i]))],
            sensory_within_per_roi_contrast=float(perroi[i,:2].mean()-perroi[i,assoc].mean()),
            association_between_per_opportunity_contrast=float(peropp[i,assoc].mean()-peropp[i,:2].mean()),
            between_SC_rank_correlation=rank(b[i],between_sc),
            corrected_between_SC_rank_correlation=rank(peropp[i],between_sc/opportunities)))
        if transitions[i]['located']:
            matched_gi=int(np.argmin(np.round(abs(g-transitions[i]['midpoint']),12)))
            ww=d['within'][i,:,matched_gi].mean(0);bb=d['between'][i,:,matched_gi].mean(0)
            aligned_networks.append(dict(subject=d['ids'][i],G=float(g[matched_gi]),
                transition_interval=transitions[i]['interval'],within_nats=ww.tolist(),
                between_shapley_nats=bb.tolist(),within_per_roi=(ww/network_sizes).tolist(),
                between_per_opportunity=(bb/opportunities).tolist(),
                top_within=SHORT_NETWORKS[int(np.argmax(ww))],
                top_within_per_roi=SHORT_NETWORKS[int(np.argmax(ww/network_sizes))],
                top_between=SHORT_NETWORKS[int(np.argmax(bb))],
                top_between_per_opportunity=SHORT_NETWORKS[int(np.argmax(bb/opportunities))]))
    method_rows=[]
    for ni,name in enumerate(METHODS):
        for gi,gval in enumerate(og):
            vectors=d['ranking'][ni,:8,:,gi].mean(1)
            method_rows.append(dict(method=name,G=float(gval),
                loso_rank_mean=float(ranks[ni,gi].mean()),loso_rank_sd=float(ranks[ni,gi].std(ddof=1)),
                loso_rank_by_subject=ranks[ni,gi].tolist(),
                within_subject_rank_repeat_mean=float(repeat[ni,gi].mean()),
                loso_top8_jaccard_mean=float(overlap[ni,gi].mean()),
                independent_seed_loso_rank_mean=float(independent_ranks[ni,gi].mean()),
                independent_seed_loso_rank_sd=float(independent_ranks[ni,gi].std(ddof=1)),
                independent_seed_loso_rank_by_subject=independent_ranks[ni,gi].tolist(),
                independent_seed_loso_top8_jaccard_mean=float(independent_overlap[ni,gi].mean()),
                own_SC_rank_correlation_mean=float(np.mean([rank(vectors[i],d['sc'][i].sum(1)) for i in range(8)]))))
    agreement=[]
    for gi,gval in enumerate(og):
        a=d['ranking'][0,:8,:,gi].mean(1);b_=d['ranking'][1,:8,:,gi].mean(1)
        agreement.append(dict(G=float(gval),full_pair_rank_mean=float(np.mean([rank(x,y) for x,y in zip(a,b_)])),
            full_pair_top8_jaccard_mean=float(np.mean([jaccard(top(x,8),top(y,8)) for x,y in zip(a,b_)])),
            full_pair_core10_jaccard_mean=float(np.mean([jaccard(d['trees'][i,1,gi]['multisource']['core10'],
                                                               d['trees'][i,1,gi]['pairwise']['core10']) for i in range(8)])),
            paired_loso_rank_delta=(ranks[0,gi]-ranks[1,gi]).tolist(),
            paired_loso_top8_delta=(overlap[0,gi]-overlap[1,gi]).tolist(),
            paired_core10_delta=(core_overlap[0,gi]-core_overlap[1,gi]).tolist()))
        agreement[-1].update(
            independent_seed_loso_rank_delta=(independent_ranks[0,gi]-independent_ranks[1,gi]).tolist(),
            independent_seed_loso_top8_delta=(independent_overlap[0,gi]-independent_overlap[1,gi]).tolist(),
            independent_seed_core10_delta=(independent_core[0,gi]-independent_core[1,gi]).tolist())
    audit=dict(simulation_conditions=len(d['validity']),organization_conditions=len(d['organization_audits']),
        syn_tolerance_nats=1e-8,
        minimum_simulation_value_nats=min(r['metrics']['audit']['minimum_nats'] for r in d['validity']),
        simulation_tolerance_negative_count=sum(r['metrics']['audit']['tolerance_negative_count'] for r in d['validity']),
        state_outside_count=sum(r['state']['outside_state_count'] for r in d['validity']),
        max_global_budget_error_nats=max(r['metrics']['closure_error_nats'] for r in d['validity']),
        max_network_budget_error_nats=max(r['network']['budget_closure_error_nats'] for r in d['validity']),
        max_same_sample_legacy_bridge_error_nats=float(np.max(abs(m['xi_nats']-m['legacy_xi_nats']))),
        max_shapley_mc_se_nats=float(d['mcse'].max()),
        min_shapley_split_half_rank=min(r['shapley']['split_half_spearman'] for r in d['organization_audits']),
        min_shapley_split_half_top10=min(r['shapley']['split_half_top10_overlap'] for r in d['organization_audits']),
        min_tree_candidate_syn_nats=min(t['audit']['minimum_candidate_syn'] for r in d['organization_audits'] for t in r['trees'].values()),
        tree_tolerance_negative_count=sum(t['audit']['tolerance_zero_count'] for r in d['organization_audits'] for t in r['trees'].values()),
        max_tree_closure_error_nats=max(abs(t['closure_error_nats']) for r in d['organization_audits'] for t in r['trees'].values()))
    reference=[]
    for gi,gval in enumerate(og):
        ref=d['ranking'][0,8,:,gi].mean(0)
        persons=d['ranking'][0,:8,:,gi].mean(1)
        reference.append(dict(G=float(gval),
            group_SC_to_individual_rank_mean=float(np.mean([rank(ref,v) for v in persons])),
            individual_consensus_loso_rank_mean=float(ranks[0,gi].mean()),
            group_SC_core_members=d['trees'][8,1,gi]['multisource']['core10'],
            group_SC_core_to_individual_jaccard_mean=float(np.mean([
                jaccard(d['trees'][8,1,gi]['multisource']['core10'],d['trees'][i,1,gi]['multisource']['core10'])
                for i in range(8)])),
            caveat='93-person group SC and shared JFIC include the development individuals; descriptive comparison only'))
    summary=dict(curves=rows,networks_G1_3=network_rows,networks_state_matched=aligned_networks,
        state_matching_rule='closest sampled G to independent rate-slope interval midpoint; numerical ties choose lower G; boundary subjects unaligned',
        methods=method_rows,core10=core_rows,
        core_consensus=core_consensus,
        method_agreement=agreement,audit=audit,network_sizes=network_sizes.tolist(),
        group_reference=reference,
        inference='8 deliberately selected development subjects; descriptive means and SD, not population inference',
        uncertainty='seed SD nested in individual; antithetic Shapley MC SE separate; subject SD on 8 individual summaries',
        missing_comparisons=['native BOLD Phi-R','SURD','nonlinear TM','full structural-null simulations',
                             'independent functional-response/prediction validation','all 93 subjects'])
    arrays=dict(ranks=ranks,overlap=overlap,repeat=repeat,core_overlap=core_overlap,
                core_null=core_null,core_repeat=core_repeat,within=w,between=b,perroi=perroi,peropp=peropp)
    arrays.update(independent_ranks=independent_ranks,independent_overlap=independent_overlap,
                  independent_core=independent_core,independent_core_null=independent_core_null)
    return summary,arrays


def panel_letters(axes):
    for label,ax in zip('abcdefghijklmnopqrstuvwxyz',np.ravel(axes)):
        ax.text(-.10,1.045,label,transform=ax.transAxes,fontweight='bold',fontsize=11)


def curves_figure(d,s,output):
    g=d['g'];xi=d['metrics']['xi_nats'].mean(1);rates=d['dyn'].mean(1)
    colors=plt.cm.viridis(np.linspace(.08,.9,8))
    fig,axes=plt.subplots(2,3,figsize=(13.7,7.2),layout='constrained');ax=axes.ravel()
    for i in range(9):
        style=dict(color=colors[i] if i<8 else '#181A1D',lw=1.35 if i<8 else 2.1,
                   marker='o' if i<8 else 'D',markersize=3,label=d['ids'][i] if i<8 else '93-subject mean SC')
        ax[0].plot(g,xi[i],**style);ax[1].plot(g*d['rho'][i],xi[i],**style)
        ax[2].plot(g,rates[i],**style)
        t=s['curves'][i]['transition']
        if t['located']:
            ax[3].plot((g-t['midpoint'])/t['midpoint'],xi[i],**style)
        ax[4].scatter(d['rho'][i],s['curves'][i]['peak_G'],color=style['color'],
                      marker='o' if s['curves'][i]['peak_interior'] else '^',s=38)
        if t['located']:
            lo,hi=t['interval'];ax[4].vlines(d['rho'][i],lo,hi,color=style['color'],lw=2)
    ax[0].set(xlabel='$G$',ylabel='Whole-brain $\\Xi$ (nats)')
    ax[1].set(xlabel='$G\\rho(\\mathbf{C})$',ylabel='Whole-brain $\\Xi$ (nats)')
    ax[2].set(xlabel='$G$',ylabel='Independent mean E rate (Hz)')
    ax[3].set(xlabel='$(G-G_c)/G_c$ (coarse rate-slope alignment)',ylabel='$\\Xi$ (nats)')
    ax[4].set(xlabel='SC spectral radius',ylabel='$G$: $\\Xi$ peak / rate-rise interval')
    for ni,(key,label,color) in enumerate([('xi_nats','$\\Xi$','#B64D64'),('whole_ei_nats','Whole EI','#BA8F34')]):
        mean=d['metrics'][key][:8].mean(1)
        ratio=np.sqrt(d['metrics'][key][:8].var(1,ddof=1).mean(1))/np.ptp(mean,axis=1)
        ax[5].scatter(np.full(8,ni)+np.linspace(-.09,.09,8),ratio,c=colors,s=27)
        ax[5].scatter(ni,ratio.mean(),marker='D',color=color,s=48)
    ax[5].set(xticks=[0,1],xticklabels=['$\\Xi$','Whole EI'],ylabel='Seed RMS SD / across-$G$ amplitude')
    ax[5].set_ylim(bottom=0)
    fig.legend(*ax[0].get_legend_handles_labels(),loc='outside upper center',ncol=5,fontsize=8,frameon=False)
    panel_letters(axes);fig.savefig(output,dpi=240,bbox_inches='tight');plt.close(fig)


def network_figure(d,a,output,indices=None):
    fig,axes=plt.subplots(2,2,figsize=(12.1,7),layout='constrained')
    colors=plt.cm.viridis(np.linspace(.08,.9,8))
    indices=list(range(8)) if indices is None else indices
    keys=['within','perroi','between','peropp']
    labels=['Within-network $\\Xi$ (nats)','Within-network $\\Xi$ / ROI (nats)',
            'Between-network Shapley (nats)','Between-network Shapley / opportunity (nats)']
    for ax,key,ylabel in zip(axes.ravel(),keys,labels):
        v=a[key]
        for row,i in enumerate(indices):
            ax.scatter(np.arange(7)+(row-(len(indices)-1)/2)*.035,v[row],color=colors[i],s=19,alpha=.9,label=d['ids'][i])
        ax.errorbar(np.arange(7),v.mean(0),yerr=v.std(0,ddof=1),fmt='D',color='#20252A',
                    ms=4,capsize=3,lw=1,label=f'{len(indices)}-subject mean ± SD')
        ax.set(xticks=np.arange(7),xticklabels=SHORT_NETWORKS,ylabel=ylabel,ylim=(0,None))
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=5,frameon=False,fontsize=8)
    panel_letters(axes);fig.savefig(output,dpi=240,bbox_inches='tight');plt.close(fig)


def organization_figure(d,a,output):
    fig,axes=plt.subplots(4,2,figsize=(13.2,12.8),layout='constrained');ax=axes.ravel();g=d['og']
    for ni,name in enumerate(METHODS):
        for axis,key in [(ax[0],'independent_ranks'),(ax[1],'repeat')]:
            value=a[key][ni]
            axis.errorbar(g+(ni-1.5)*.015,value.mean(1),yerr=value.std(1,ddof=1),
                           color=COLORS[ni],marker='o',ms=3,capsize=2,label=METHOD_LABELS[ni])
    for ni in range(2):
        value=a['independent_core'][ni]
        ax[3].errorbar(g+(ni-.5)*.025,value.mean(1),yerr=value.std(1,ddof=1),color=COLORS[ni],
                       marker='o',capsize=2,label=METHOD_LABELS[ni])
        ax[3].plot(g,a['independent_core_null'][ni].mean(1),color=COLORS[ni],ls=':',lw=1.2)
        value=a['independent_overlap'][ni]
        ax[4].errorbar(g+(ni-.5)*.025,value.mean(1),yerr=value.std(1,ddof=1),color=COLORS[ni],
                       marker='o',capsize=2)
        value=a['core_repeat'][ni]
        ax[2].errorbar(g+(ni-.5)*.025,value.mean(1),yerr=value.std(1,ddof=1),color=COLORS[ni],
                       marker='o',capsize=2)
    for axis in ax[:5]:
        axis.set(xlabel='$G$')
        if axis in ax[:2]:axis.set_ylim(-.3,1.35)
        else:axis.set_ylim(-.15,1.15)
    ax[0].set_ylabel('Independent-seed LOSO rank ρ')
    ax[1].set_ylabel('Same-subject cross-seed rank ρ')
    ax[2].set_ylabel('$C_{10}$ cross-seed Jaccard\n(same subject)')
    ax[3].set_ylabel('$C_{10}$ cross-subject Jaccard\n(independent seeds)')
    ax[4].set_ylabel('LOSO top-8 Jaccard\n(independent seeds)')
    delta=a['independent_ranks'][0]-a['independent_ranks'][1]
    ax[5].axhline(0,color='#A0A0A0',lw=.7)
    for i in range(8):ax[5].plot(g,delta[:,i],lw=.65,color='#BDBFC3',marker='o',ms=2)
    ax[5].plot(g,delta.mean(1),color=COLORS[0],lw=1.8,marker='D',ms=4)
    ax[5].set(xlabel='$G$',ylabel='Independent-seed rank ρ difference\n(multisource − pairwise)')
    gi=int(np.flatnonzero(np.isclose(g,1.3))[0])
    for axis,name in zip(ax[6:],METHODS[:2]):
        image=np.zeros((8,100))
        for i in range(8):image[i,d['trees'][i,1,gi][name]['core10']]=1
        axis.imshow(image,aspect='auto',cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
        axis.set(yticks=range(8),yticklabels=d['ids'][:8],xlabel='ROI index (fixed atlas order)',
                 xticks=[0,24,49,74,99],xticklabels=[0,24,49,74,99])
        axis.text(.03,1.045,f'{name}: $C_{{10}}$, G=1.3, seed 4',transform=axis.transAxes,fontsize=9)
        axis.tick_params(axis='y',labelsize=8)
    handles,labels=ax[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([0],[0],color='#666666',linestyle=':'))
    labels.append('Hemisphere / Yeo / strength matched null (panel d)')
    fig.legend(handles,labels,loc='outside upper center',ncol=2,fontsize=8,frameon=False)
    panel_letters(axes);fig.savefig(output,dpi=240,bbox_inches='tight');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--curves-only',action='store_true',help='Plot completed acquisition while organization runs')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        data=load(args.base,organization=not args.curves_only)
        if args.curves_only:
            summary=dict(curves=[])
            for i,subject in enumerate(data['ids']):
                values=data['metrics']['xi_nats'][i].mean(0);pi=int(np.argmax(values))
                summary['curves'].append(dict(subject=subject,peak_G=float(data['g'][pi]),
                    peak_interior=0<pi<len(data['g'])-1,
                    transition=transition_interval(data['g'],data['dyn'][i].mean(0))))
            ns=np.bincount(data['membership']);gi=int(np.flatnonzero(np.isclose(data['g'],1.3))[0])
            w=data['within'][:8,:,gi].mean(1);b=data['between'][:8,:,gi].mean(1)
            arrays=dict(within=w,between=b,perroi=w/ns,peropp=b/(ns*(100-ns)))
        else:
            summary,arrays=summaries(data)
            atomic_json(args.base/'summary.json',summary)
        figures=ROOT/'docs/reports/assets/dmf_subject_consistency';figures.mkdir(parents=True,exist_ok=True)
        with plt.rc_context({'font.family':'sans-serif','font.size':9,'axes.spines.top':False,
                             'axes.spines.right':False,'legend.frameon':False,'axes.axisbelow':True}):
            curves_figure(data,summary,figures/'curves.png')
            network_figure(data,arrays,figures/'networks.png')
            # A separately labelled matched-state panel retains the raw-G panel.
            ns=np.bincount(data['membership']);indices=[];aw=[];ab=[]
            for i in range(8):
                t=transition_interval(data['g'],data['dyn'][i].mean(0))
                if not t['located']:continue
                gi=int(np.argmin(np.round(abs(data['g']-t['midpoint']),12)))
                indices.append(i);aw.append(data['within'][i,:,gi].mean(0));ab.append(data['between'][i,:,gi].mean(0))
            aw=np.array(aw);ab=np.array(ab)
            aligned=dict(within=aw,between=ab,perroi=aw/ns,peropp=ab/(ns*(100-ns)))
            network_figure(data,aligned,figures/'networks_state_matched.png',indices=indices)
            if not args.curves_only:organization_figure(data,arrays,figures/'organization.png')
        print(json.dumps({k:summary[k] for k in ['curves','networks_G1_3','method_agreement','audit'] if k in summary},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
