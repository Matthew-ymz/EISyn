#!/usr/bin/env python3
"""Descriptive cache-only recurrence of the displayed E/F ancestor sets.

Exact membership is primary and uses the frozen two-of-three seed / four-of-six
transition-window rule. Jaccard >= .8 is a separate exploratory similarity
description; different nearby sets may supply different conditions. It is not
exact recurrence, a calibrated null test, or evidence for one shared module.
No simulation, information estimation, optimization or significance testing.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_dmf_sc_topology_comparison import BASE, load_data, sha
from scripts.analyze_dmf_critical_coalitions_all93 import (
    METRICS, members_mask, word_mask, read_subject, empty_audit,
)
from scripts.run_dmf_paired_spt_pilot import paired_sources
from scripts.plot_dmf_spt_peel_groups import load_comparison, PAIRED_SUMMARY
from scripts.plot_dmf_main_roi_highlights import GROUPS
from scripts.spt import flatten_nodes

DEST=ROOT/'results/dmf_schaefer100/roi_ancestor_recurrence/summary.json'


def ancestor_sets(tree):
    nodes=flatten_nodes(tree)
    result={}
    for name,keys,size in [('E',('B','C'),19),('F',('B','C','D'),54)]:
        union={i-1 for k in keys for i in GROUPS[k]}
        node=min((n for n in nodes if union<=set(n.indices)),key=lambda n:n.size)
        if node.size!=size:raise ValueError('Reference ancestor identity changed')
        result[name]=node.indices
    return result


def main():
    started=time.perf_counter()
    _,subjects,words,orders,counts,registered,fingerprints,_=load_data()
    c=json.loads((BASE/'contract.json').read_text())
    bg=json.loads((BASE/'background_reference.json').read_text())
    source={seed:hashlib.sha256(np.concatenate(paired_sources(seed,c['sample_count'],100),axis=1)
                               .tobytes()).hexdigest() for seed in c['seeds']}
    if c['repeat_seed_count']!=2 or len(c['seeds'])!=3 or c['syn_tolerance_nats']!=1e-8:
        raise ValueError('Original repeat/validity protocol changed')
    tree,reference=load_comparison(PAIRED_SUMMARY,4,[1.3])[0]
    refs=ancestor_sets(tree)
    masks={k:members_mask(v) for k,v in refs.items()}
    global_counts={}
    for k,mask in masks.items():
        ix=np.flatnonzero((words[:,0]==np.uint64(mask&((1<<64)-1)))&
                          (words[:,1]==np.uint64(mask>>64)))
        if len(ix)>1:raise ValueError('Duplicate exact registry identity')
        vals=counts[ix[0]] if len(ix) else np.zeros((2,len(METRICS)),int)
        global_counts[k]={mode:{m:int(v) for m,v in zip(METRICS,vals[j])}
                          for j,mode in enumerate(['natural_presence','above_background'])}
    audit=empty_audit(); records=[]
    for pi,subject in enumerate(subjects):
        window=c['windows'][subject]
        if len(window['grid_indices'])!=6 or window['required_hits']!=4:
            raise ValueError('Original transition window changed')
        local=dict(c,G=[c['G'][i] for i in window['grid_indices']])
        events=read_subject(BASE,local,subject,bg['thresholds'],sha(BASE/'contract.json'),audit,source)
        record={'subject':subject,'G':local['G'],'ancestors':{}}
        for key,ref in masks.items():
            exact=events.get(ref,np.zeros((2,6),np.uint8))
            repeats=np.array([[int(x).bit_count()>=2 for x in row] for row in exact])
            best=np.zeros((6,3)); matched=np.zeros((6,3),dtype=object)
            for mask,values in events.items():
                j=(mask&ref).bit_count()/(mask|ref).bit_count()
                for gi,seeds in enumerate(values[0]):
                    for si in range(3):
                        if int(seeds)&(1<<si) and j>best[gi,si]:
                            best[gi,si]=j;matched[gi,si]=mask
            similar=(best>=.8).sum(1)>=2
            record['ancestors'][key]={
                'exact_natural_seed_masks':exact[0].tolist(),
                'exact_above_background_seed_masks':exact[1].tolist(),
                'exact_any_tree':bool(exact[0].any()),
                'exact_repeat_G_count':int(repeats[0].sum()),
                'exact_strong_repeat_G_count':int(repeats[1].sum()),
                'exact_window_recurrent':bool(repeats[0].sum()>=4),
                'exact_strong_window_recurrent':bool(repeats[1].sum()>=4),
                'nearest_Jaccard_by_G_seed':best.tolist(),
                'nearest_member_masks_decimal':[[str(m) for m in row] for row in matched],
                'median_nearest_Jaccard':float(np.median(best)),
                'similar_window_G_count':int(similar.sum()),
                'similar_window_recurrent':bool(similar.sum()>=4)}
        records.append(record)
        if (pi+1)%25==0:print(f'Checked {pi+1}/93 participants',flush=True)
    results={}
    for key,members in refs.items():
        rows=[r['ancestors'][key] for r in records]
        result={k:sum(int(r[k]) for r in rows) for k in [
            'exact_any_tree','exact_window_recurrent','exact_strong_window_recurrent',
            'similar_window_recurrent']}
        # Exact per-window counts must reproduce the existing all-G registry.
        for mode,field in [('natural_presence','exact_window_recurrent'),
                            ('above_background','exact_strong_window_recurrent')]:
            if result[field]!=global_counts[key][mode]['critical_ge4']:
                raise ArithmeticError('Targeted scan differs from original exact registry')
        vals=[r['median_nearest_Jaccard'] for r in rows]
        result.update(members_one_based=[i+1 for i in members],order=len(members),
            global_exact_registry=global_counts[key],
            median_nearest_Jaccard_subject_quartiles=np.quantile(vals,[.25,.5,.75]).tolist(),
            threshold_nats=bg['thresholds'][str(len(members))]['threshold_nats'])
        results[key]=result
    output=dict(status='complete',subject_count=93,reference=reference,
        definition='Original mean-SC, G=1.3, seed-4 E/F minimal common ancestors; fixed before participant lookup',
        protocol=dict(exact_identity='Full ROI set equality',repeat_seeds='At least 2 of 3 at same G',
            transition_window='At least 4 of 6 individually aligned G points',
            similarity='Exploratory nearest proper-node Jaccard >=0.8 with the same repeat/window rule; no strength threshold; nearby identities may vary',
            inference='Descriptive, no p values; large-set nearest overlaps are size/candidate dependent'),
        results=results,participants=records,syn_audit=audit,
        source_fingerprints=fingerprints,new_simulations=0,new_information_queries=0,
        elapsed_seconds=time.perf_counter()-started)
    DEST.parent.mkdir(parents=True,exist_ok=True)
    DEST.write_text(json.dumps(output,indent=2))
    print(json.dumps(results,indent=2));print('Saved',DEST)


if __name__=='__main__':main()
