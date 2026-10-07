#!/usr/bin/env python3
"""Exact-member registration, frozen discovery gate, and held-out validation."""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import beta, binomtest, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_dmf_critical_coalitions import BASE, TOL
from scripts.run_dmf_subject_consistency import atomic_json, digest
from scripts.run_dmf_subject_curve_baselines import atomic_savez


@lru_cache(maxsize=None)
def wilson(successes, n):
    z = norm.ppf(.975); p = successes / n
    centre = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [0. if successes == 0 else float(centre-half),
            1. if successes == n else float(centre+half)]


@lru_cache(maxsize=None)
def leakage_upper(successes, n):
    return 1. if successes == n else float(beta.ppf(.95, successes+1, n-successes))


def holm(pvalues):
    p = np.asarray(pvalues, float); order = np.argsort(p, kind='stable')
    result = np.empty_like(p); last = 0.
    for rank, index in enumerate(order):
        last = max(last, min(1., (len(p)-rank)*p[index])); result[index] = last
    return result


def paired_test(critical, endpoint, draws=10000, seed=20261007):
    a, b = np.asarray(critical, bool), np.asarray(endpoint, bool)
    up, down = int((a & ~b).sum()), int((b & ~a).sum())
    p = float(binomtest(up, up+down, .5, alternative='greater').pvalue) if up+down else 1.
    d = a.astype(float)-b.astype(float)
    index = np.random.default_rng(seed).integers(len(d), size=(draws, len(d)))
    return dict(critical_only=up, endpoint_only=down, p_value=p,
        difference=float(d.mean()), bootstrap_ci95=np.quantile(d[index].mean(1), [.025, .975]).tolist())


def calibrate(base, c):
    sha = digest(base/'contract.json')
    maxima = np.full((len(c['background']['seeds']), 99), np.nan)
    candidate_minimum = np.inf; candidate_negative = 0; selected_negative = 0
    for si, seed in enumerate(c['background']['seeds']):
        r = json.loads((base/'background_trees'/f'background_seed{seed}.json').read_text())
        if r['contract_sha256'] != sha:
            raise ValueError('Background provenance mismatch')
        t = r['tree']; ca = t['candidate_audit']
        candidate_minimum = min(candidate_minimum, ca['minimum_candidate_syn'])
        candidate_negative += ca['tolerance_zero_count']
        selected_negative += t['selected_syn_audit']['tolerance_negative_count']
        for n in t['nodes']:
            j = n['order']-2; v = n['syn_nats_raw']
            maxima[si,j] = v if np.isnan(maxima[si,j]) else max(maxima[si,j], v)
    thresholds = {}
    for j in range(99):
        values = maxima[:,j]; values = values[np.isfinite(values)]
        enough = len(values) >= c['background']['minimum_observed_trees_per_order']
        thresholds[str(j+2)] = dict(observed_tree_count=len(values),
            threshold_nats=float(np.quantile(values, .95, method='linear')) if enough else None)
    result = dict(contract_sha256=sha, tree_count=len(maxima), thresholds=thresholds,
        native_syn_tolerance_nats=TOL, minimum_candidate_syn_nats=float(candidate_minimum),
        candidate_tolerance_negative_count=candidate_negative,
        selected_tolerance_negative_count=selected_negative,
        warning='Background quantile is a screening reference, not a formal node-level p value')
    path=base/'background_reference.json'
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('Frozen background reference changed')
    atomic_json(path, result)
    atomic_savez(base/'background_maxima.npz', raw_maxima_nats=maxima, orders=np.arange(2,101),
                 seeds=c['background']['seeds'], contract_sha256=sha)
    return result


def read_events(base, c, subjects):
    sha = digest(base/'contract.json'); events = defaultdict(dict)
    audit = dict(tree_count=0, internal_node_count=0, minimum_candidate_syn_nats=float('inf'),
        candidate_tolerance_negative_count=0, selected_tolerance_negative_count=0,
        max_closure_error_nats=0., max_dense_xi_difference_nats=0.,
        candidate_count=0, imported_density_count=0)
    elapsed=[]; sim=[]; tree=[]
    for pi, p in enumerate(subjects):
        for gi, g in enumerate(c['G']):
            for si, seed in enumerate(c['seeds']):
                r = json.loads((base/'trees'/f'{p}_G{g:.2f}_seed{seed}.json').read_text())
                if r['contract_sha256'] != sha or r['subject'] != p or r['G'] != g or r['seed'] != seed:
                    raise ValueError('Condition identity/provenance mismatch')
                t=r['tree']; seen=set()
                for n in t['nodes']:
                    members=tuple(sorted(n['members']))
                    if members in seen or len(members)!=n['order'] or set(n['left']) & set(n['right']) or set(n['left'])|set(n['right']) != set(members):
                        raise ValueError('Invalid node identity/partition')
                    if n['syn_nats_raw'] < -TOL or not np.isfinite(n['syn_nats_raw']):
                        raise ArithmeticError('Invalid raw node Syn')
                    seen.add(members)
                    if 2 <= len(members) <= 99:
                        events[members][(pi,gi,si)] = n['syn_nats_raw']
                if len(seen)!=99:
                    raise ValueError('Incomplete natural tree')
                ca=t['candidate_audit']; audit['tree_count']+=1; audit['internal_node_count']+=len(seen)
                audit['minimum_candidate_syn_nats']=min(audit['minimum_candidate_syn_nats'],ca['minimum_candidate_syn'])
                audit['candidate_tolerance_negative_count']+=ca['tolerance_zero_count']
                audit['selected_tolerance_negative_count']+=t['selected_syn_audit']['tolerance_negative_count']
                audit['candidate_count']+=ca['candidate_count']
                audit['max_closure_error_nats']=max(audit['max_closure_error_nats'],abs(t['closure_error_nats']))
                audit['max_dense_xi_difference_nats']=max(audit['max_dense_xi_difference_nats'],r['dense_xi_difference_nats'])
                audit['imported_density_count']+=r['imported_density']
                elapsed.append(r['elapsed_seconds']); sim.append(r['simulation_seconds']); tree.append(r['tree_seconds'])
    audit['seconds']=dict(mean_condition=float(np.mean(elapsed)), mean_simulation=float(np.mean(sim)),
        mean_tree=float(np.mean(tree)), summed_condition=float(np.sum(elapsed)))
    return events, audit


def candidate_arrays(members, events, c, subjects, background):
    values=np.full((len(subjects),len(c['G']),len(c['seeds'])),np.nan)
    for (pi,gi,si), value in events.get(members, {}).items():
        values[pi,gi,si]=value
    threshold=background['thresholds'][str(len(members))]['threshold_nats']
    present=np.isfinite(values)
    strong=present & (values > threshold) if threshold is not None else np.zeros_like(present)
    return values, present.sum(2)>=c['repeat_seed_count'], strong.sum(2)>=c['repeat_seed_count']


def occurrence_stats(repeated, c, subjects):
    n=len(subjects); critical=[]; exclusive=[]; narrow=[]; critical_fraction=[]; outside=[]
    low=repeated[:,c['low_grid_indices']]; high=repeated[:,c['high_grid_indices']]
    for pi,p in enumerate(subjects):
        w=c['windows'][p]; idx=w['grid_indices']; count=int(repeated[pi,idx].sum())
        critical.append(count>=w['required_hits']); critical_fraction.append(count/len(idx))
        mask=np.ones(len(c['G']),bool);mask[idx]=False
        outside.append(bool(repeated[pi,mask].any()))
        exclusive.append(critical[-1] and not outside[-1])
        lo,hi=w['transition']['interval']
        endpoints=np.flatnonzero(np.isclose(c['G'],lo)|np.isclose(c['G'],hi))
        narrow.append(bool(repeated[pi,endpoints].all()))
    critical=np.array(critical); low_stage=low.sum(1)>=2; high_stage=high.sum(1)>=2
    low_any=low.any(1); high_any=high.any(1)
    def pack(a):
        count=int(np.sum(a))
        return dict(count=count, denominator=n, rate=count/n, ci95_wilson=wilson(count,n))
    output=dict(critical=pack(critical), low_stage=pack(low_stage), high_stage=pack(high_stage),
        low_any_leak=pack(low_any), high_any_leak=pack(high_any),
        narrow_both_endpoints=pack(narrow), window_exclusive=pack(exclusive),
        any_outside_window=pack(outside),
        low_leak_upper_one_sided95=leakage_upper(int(low_any.sum()),n),
        high_leak_upper_one_sided95=leakage_upper(int(high_any.sum()),n),
        critical_subjects=[p for p,v in zip(subjects,critical) if v],
        critical_grid_fractions=critical_fraction,
        low_grid_fractions=low.mean(1).tolist(), high_grid_fractions=high.mean(1).tolist())
    return output, critical, low_stage, high_stage


def analyze(base, phase):
    c=json.loads((base/'contract.json').read_text()); sha=digest(base/'contract.json')
    bg=calibrate(base,c)
    subjects=c['development_subject_ids'] if phase=='discovery' else c['holdout_subject_ids']
    events,audit=read_events(base,c,subjects)
    if phase=='discovery':
        members_list=sorted(events)
    else:
        frozen=json.loads((base/'frozen_candidates.json').read_text())
        if frozen['contract_sha256']!=sha or not frozen['candidates']:
            raise ValueError('No frozen validation candidate set')
        members_list=[tuple(r['members']) for r in frozen['candidates']]
    rows=[]; arrays={}; tests=[]; test_locations=[]
    for members in members_list:
        values,raw,strong=candidate_arrays(members,events,c,subjects,bg)
        raw_stats,*_=occurrence_stats(raw,c,subjects)
        strength_stats,critical,low_stage,high_stage=occurrence_stats(strong,c,subjects)
        gate=c['discovery_gate' if phase=='discovery' else 'validation_gate']
        qualifies=(len(members)>=3 and strength_stats['critical']['count']>=gate['minimum_critical_subjects']
            and strength_stats['low_any_leak']['count']<=gate['maximum_low_leak_subjects']
            and strength_stats['high_any_leak']['count']<=gate['maximum_high_leak_subjects'])
        row=dict(members=list(members),order=len(members),raw_presence=raw_stats,
            above_background=strength_stats, threshold_nats=bg['thresholds'][str(len(members))]['threshold_nats'],
            selected_node_count=int(np.isfinite(values).sum()), meets_empirical_gate=bool(qualifies),
            meets_window_only_target=bool(qualifies and strength_stats['window_exclusive']['count']>=gate['minimum_critical_subjects']))
        if phase=='validation':
            row['comparisons']={}
            for name,endpoint in [('low',low_stage),('high',high_stage)]:
                test=paired_test(critical,endpoint,c['statistical_protocol']['bootstrap_replicates'],c['statistical_protocol']['bootstrap_seed'])
                row['comparisons'][name]=test;tests.append(test['p_value']);test_locations.append(test)
        rows.append(row)
    if tests:
        for test,p in zip(test_locations,holm(tests)):
            test['holm_p_value']=float(p)
    # Ranking has no tunable weighted score: prevalence, leakage, order, members.
    rows.sort(key=lambda r:(-r['above_background']['critical']['count'],
        r['above_background']['low_any_leak']['count']+r['above_background']['high_any_leak']['count'],
        -r['raw_presence']['critical']['count'],r['order'],r['members']))
    preferred=[r for r in rows if 3<=r['order']<=10]
    higher=[r for r in rows if r['order']>=3]
    frozen_rows=[r for r in higher if r['meets_empirical_gate']]
    result=dict(status='complete',phase=phase,subject_ids=subjects,subject_count=len(subjects),
        contract_sha256=sha,background_reference_sha256=digest(base/'background_reference.json'),
        analysis_implementation_sha256=digest(Path(__file__)),
        registered_exact_sets=len(rows), registered_high_order_sets=len(higher),
        registered_preferred_sets=len(preferred), empirical_gate_candidate_count=len(frozen_rows),
        window_only_candidate_count=sum(r['meets_window_only_target'] for r in frozen_rows),
        maximum_raw_critical_count=max((r['raw_presence']['critical']['count'] for r in higher),default=0),
        maximum_strong_critical_count=max((r['above_background']['critical']['count'] for r in higher),default=0),
        top_preferred=preferred[:5],top_all_orders=higher[:5],audit=audit,
        pair_reference_top=[r for r in rows if r['order']==2][:3],
        uncalibrated_orders=[int(k) for k,v in bg['thresholds'].items() if v['threshold_nats'] is None],
        interpretation='Exploratory exact-member discovery, all selected candidates retained' if phase=='discovery' else 'Frozen-member held-out model-cohort validation')
    atomic_json(base/(phase+'_registry.json'),dict(contract_sha256=sha,candidates=rows))
    atomic_json(base/(phase+'_summary.json'),result)
    display=preferred[:3]
    if higher and higher[0] not in display:
        display.append(higher[0])
    vals=[];raws=[];strongs=[]
    for r in display:
        v,raw,strong=candidate_arrays(tuple(r['members']),events,c,subjects,bg)
        vals.append(v);raws.append(raw);strongs.append(strong)
    atomic_savez(base/(phase+'_display.npz'),members_json=json.dumps([r['members'] for r in display]),
        subject_ids=subjects,G=c['G'],seeds=c['seeds'],raw_syn_nats=vals,
        repeated_presence=raws,repeated_above_background=strongs,contract_sha256=sha)
    if phase=='discovery':
        frozen=dict(contract_sha256=sha,background_reference_sha256=digest(base/'background_reference.json'),
            candidates=frozen_rows,selection='Frozen empirical>=80% critical and <=10% endpoint-any leakage',
            discovery_summary_sha256=digest(base/'discovery_summary.json'))
        path=base/'frozen_candidates.json'
        if path.exists() and json.loads(path.read_text())!=frozen:
            raise ValueError('Frozen candidate list changed')
        atomic_json(path,frozen)
    print(json.dumps({k:result[k] for k in ['phase','subject_count','registered_high_order_sets',
        'empirical_gate_candidate_count','maximum_raw_critical_count','maximum_strong_critical_count']},ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=BASE)
    p.add_argument('--phase',choices=['discovery','validation'],default='discovery')
    a=p.parse_args();analyze(a.output_dir,a.phase)


if __name__=='__main__':
    main()
