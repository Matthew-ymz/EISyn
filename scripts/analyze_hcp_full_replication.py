#!/usr/bin/env python3
"""Prespecified fixed-effect statistics for the all-available HCP replication."""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse
import json
import time
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr,wilcoxon,norm

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_hcp_full_replication import OUT,STATES,TASKS,CONTRACT,core,dump,sha
from scripts import run_hcp_task_behavior_979 as oldstats
from scripts.screen_hcp_social_composite_scores_57 import infer_effective_trials
SEED=20261022;PERM=100000;BOOT=5000;LN2=float(np.log(2))
CON=json.loads(CONTRACT.read_text());FIXED=CON['fixed_behavior_hypotheses']
NAMES=list(oldstats.NAMES);NETWORKS=tuple(core.NETWORKS)


def load_rows(m):
    rows={}
    for sid,inp in m['inputs'].items():
        p=OUT/'subjects'/f'{sid}.json'
        if not p.exists():raise ValueError(f'Missing subject {sid}')
        saved=json.loads(p.read_text())['records']
        for s in inp['states']:
            r=saved[s]
            if r['fingerprint']!=inp['fingerprints'][s]:raise ValueError(f'Stale result {sid}/{s}')
            if r['numerical_audit']['all_candidate_syn']['significant_negative_count']:raise ValueError(f'Syn violation {sid}/{s}')
            rows[sid,s]=r
    if len(rows)!=sum(map(len,m['cohorts'].values())):raise ValueError('Model coverage mismatch')
    return rows


def feature(row):
    ei=row['coalition_ei_bits']
    return np.array([ei[name]-sum(ei[n] for n in name.split('+')) for name in NAMES])


def paired(left,right,seed):
    d=np.asarray(left)-np.asarray(right);nonzero=d[d!=0]
    if len(nonzero):
        p=float(wilcoxon(d,zero_method='wilcox',correction=False,alternative='two-sided').pvalue)
        ranks=rankdata(abs(nonzero));rb=float(np.sum(ranks*np.sign(nonzero))/ranks.sum())
    else:p=1.;rb=0.
    rng=np.random.default_rng(seed);means=[]
    for start in range(0,BOOT,250):
        idx=rng.integers(len(d),size=(min(250,BOOT-start),len(d)));means.extend(d[idx].mean(1))
    return dict(n=len(d),p=p,mean_difference=float(d.mean()),median_difference=float(np.median(d)),positive_fraction=float(np.mean(d>0)),rank_biserial=rb,mean_difference_ci95=np.quantile(means,[.025,.975]).tolist())


def panels(m,rows,subjects=None,spatial=False,windows=None):
    result=dict(A=[],B_root=[],C_control=[])
    flags=CON['anomaly_controls']['spatial_le5_voxel_flags']
    for i,s in enumerate(TASKS):
        ids=m['pairs'][s]
        if subjects is not None:ids=sorted(set(ids)&set(subjects))
        if spatial:
            bad={r['subject'] for r in flags[s].get('1000',{}).get('flagged_runs',[])};ids=[sid for sid in ids if sid not in bad]
        for field,kind,factor,col,sign in [('system_xi_bits','A',LN2,None,1),('order_mass_bits','B_root',LN2,-1,1),('network_percent','C_control',1.,5,-1)]:
            left=[];right=[]
            for sid in ids:
                rest=(windows[sid,s] if windows is not None else rows[sid,'REST'])[field]
                task=rows[sid,s][field]
                if col is not None:rest,task=rest[col],task[col]
                left.append(rest*factor if sign>0 else task*factor);right.append(task*factor if sign>0 else rest*factor)
            q=paired(left,right,SEED+100*i+len(kind));q.update(state=s,contrast='REST-task' if sign>0 else 'task-REST',unit='percentage points' if kind=='C_control' else 'nats');result[kind].append(q)
    for kind,rr in result.items():
        adjust=oldstats.bh if kind=='A' else oldstats.holm
        for q,p in zip(rr,adjust([q['p'] for q in rr])):
            q['p_adjusted']=float(p);q['correction']='BH7' if kind=='A' else 'Holm7';q['same_direction_significant']=bool(q['mean_difference']>0 and p<.05)
    return result


def descriptive(m,rows,subjects=None):
    desc={}
    for s,ids in m['cohorts'].items():
        if subjects is not None:ids=sorted(set(ids)&set(subjects))
        r=[rows[sid,s] for sid in ids];mass=np.array([v['order_mass_bits'] for v in r])*LN2;pct=np.array([v['network_percent'] for v in r])
        indiv=np.argmax(mass,axis=1)+2
        desc[s]=dict(n=len(ids),mean_system_xi_nats=float(np.mean([v['system_xi_bits'] for v in r])*LN2),median_system_xi_nats=float(np.median([v['system_xi_bits'] for v in r])*LN2),order_mass_mean_nats=mass.mean(0).tolist(),order_share_mean=np.mean([v['order_share'] for v in r],0).tolist(),mean_peak_order=int(np.argmax(mass.mean(0))+2),individual_peak_counts={str(k):int(np.sum(indiv==k)) for k in range(2,8)},network_percent_mean=pct.mean(0).tolist(),minimum_mean_share_network=NETWORKS[int(np.argmin(pct.mean(0)))],heldout_skill_median=float(np.median([v['heldout']['skill_ratio'] for v in r])))
    return desc


def main_summary(m,rows):
    desc=descriptive(m,rows)
    result=dict(descriptive=desc,primary=panels(m,rows),common=panels(m,rows,m['common']),spatial=panels(m,rows,spatial=True),numerical_audit=dict(models=len(rows),candidate_count=sum(r['numerical_audit']['all_candidate_syn']['checked_count'] for r in rows.values()),minimum_candidate_syn_bits=min(r['numerical_audit']['all_candidate_syn']['minimum_bits'] for r in rows.values()),near_zero_negative_count=sum(r['numerical_audit']['all_candidate_syn']['near_zero_negative_count'] for r in rows.values()),maximum_closure_error_bits=max(abs(v) for r in rows.values() for v in r['identity_errors_bits'].values()),covariance_floor_count=sum(r['covariance_floor']['affected_eigenvalue_count'] for r in rows.values())),manifest_sha256=sha(OUT/'manifest.json'))
    result['common_descriptive']=descriptive(m,rows,m['common'])
    result['analysis_code_sha256']=sha(Path(__file__))
    dump(OUT/'main_statistics.json',result);print(json.dumps(result['primary']),flush=True)


def behavior_frame(ids):
    raw=pd.read_csv(oldstats.BEHAVIOR,dtype={'Subject':str},low_memory=False).set_index('Subject');f=raw.loc[ids].copy()
    origin=pd.read_csv(ROOT/'results/hcp_language_story_math_coalitions_57/selected_candidate_source_data.tsv',sep='\t');lookup=dict(zip(origin.subject.str.removeprefix('sub-'),origin.cohort))
    f['origin']=[lookup.get(s,'expansion') for s in ids]
    return f


def endpoint(f,state,ep,fixed_social_trials=None):
    if state=='LANGUAGE':return pd.to_numeric(f[ep],errors='coerce').to_numpy(float)
    if state=='EMOTION':
        y=pd.to_numeric(f.Emotion_Task_Face_Median_RT,errors='coerce').to_numpy(float);shape=pd.to_numeric(f.Emotion_Task_Shape_Median_RT,errors='coerce').to_numpy(float)
        mask=np.isfinite(y)&(y>0)&np.isfinite(shape)&(shape>0)
        z=np.full(len(f),np.nan);z[mask]=-np.log(y[mask]);return z
    if state=='MOTOR':
        cfg=CON['motor_standardization'];v=f[cfg['components']].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        return ((v-np.array(cfg['means']))/np.array(cfg['sample_sds'])).mean(1)
    y=np.full(len(f),np.nan)
    for i,(_,row) in enumerate(f.iterrows()):
        h=float(row.Social_Task_TOM_Perc_TOM);cr=float(row.Social_Task_Random_Perc_Random)
        if not np.isfinite(h+cr):continue
        n=infer_effective_trials(row.to_dict()) if fixed_social_trials is None else fixed_social_trials
        hit=h*n/100;fa=(100-cr)*n/100
        if fixed_social_trials is None:hit=np.rint(hit);fa=np.rint(fa)
        y[i]=norm.ppf((hit+.5)/(n+1))-norm.ppf((fa+.5)/(n+1))
    return y


def design(f,state,global_xi=None,legacy=False):
    if legacy:
        age=np.array([38 if s=='36+' else np.mean(list(map(float,s.split('-')))) for s in f.Age])
        if state=='LANGUAGE':d=np.ones((len(f),1))
        else:
            d=np.column_stack([np.ones(len(f)),rankdata(age) if state in ['EMOTION','SOCIAL'] else age,(f.Gender=='M').to_numpy(float)])
            if state!='SOCIAL':d=np.column_stack([d,pd.get_dummies(f.origin,drop_first=True,dtype=float).to_numpy()])
        if state=='EMOTION':d=np.column_stack([d,rankdata(-np.log(f.Emotion_Task_Shape_Median_RT.to_numpy(float)))])
    else:d,_=oldstats.make_design(f,emotion=(state=='EMOTION'))
    if global_xi is not None:d=np.column_stack([d,rankdata(global_xi)])
    return d


def analysis_object(f,x,y,state,xi,global_adjust=False,legacy=False):
    d=design(f,state,xi if global_adjust else None,legacy);q=oldstats.basis(d)
    xr=oldstats.residual(rankdata(x,axis=0),q);yr=oldstats.residual(rankdata(y),q)
    return dict(frame=f,x=x,y=y,state=state,design=d,q=q,xr=xr,yr=yr,brain=oldstats.normalize(xr),observed=oldstats.normalize(xr).T@oldstats.normalize(yr),cohort=f.origin.to_numpy(),global_adjust=global_adjust,global_xi=xi)


def design_audit(f,state,xi=None):
    d,labels=oldstats.make_design(f,emotion=state=='EMOTION')
    if xi is not None:d=np.column_stack([d,rankdata(xi)]);labels.append('rank_system_xi')
    categories={c:sorted(f[c].astype(str).unique().tolist()) for c in ['Age','Gender','fMRI_3T_ReconVrs','origin']}
    if f[['Age','Gender','fMRI_3T_ReconVrs','origin']].isna().any().any():raise ValueError('Missing nuisance covariates')
    return dict(columns=labels,columns_count=d.shape[1],rank=int(np.linalg.matrix_rank(d)),
        category_levels=categories,reference_levels={c:v[0] for c,v in categories.items()},
        origin_counts={str(k):int(v) for k,v in f.origin.value_counts().items()})


def permutation(a,seed,blocks=True,max_t=False):
    rng=np.random.default_rng(seed);e=a['yr'];q=a['q'];x=a['brain'];groups=[np.flatnonzero(a['cohort']==g) for g in np.unique(a['cohort'])] if blocks else [np.arange(len(e))]
    counts=np.zeros(x.shape[1],int);maxcounts=np.zeros(x.shape[1],int)
    for start in range(0,PERM,500):
        n=min(500,PERM-start);ix=np.tile(np.arange(len(e)),(n,1))
        for group in groups:ix[:,group]=group[np.argsort(rng.random((n,len(group))),axis=1)]
        pe=e[ix];proj=pe@q;den=np.sqrt(np.sum(pe*pe,1)-np.sum(proj*proj,1));v=np.abs((pe@x)/den[:,None]);obs=np.abs(a['observed'])
        counts+=np.sum(v>=obs,axis=0)
        if max_t:maxcounts+=np.sum(v.max(1)[:,None]>=obs,axis=0)
    return (counts+1)/(PERM+1),((maxcounts+1)/(PERM+1) if max_t else None)


def bootstrap(a,seed):
    rng=np.random.default_rng(seed);groups=[np.flatnonzero(a['cohort']==g) for g in np.unique(a['cohort'])];values=[]
    for _ in range(BOOT):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);f=a['frame'].iloc[ix]
        d=design(f,a['state'],a['global_xi'][ix] if a['global_adjust'] else None);q=oldstats.basis(d)
        xr=oldstats.normalize(oldstats.residual(rankdata(a['x'][ix],axis=0),q));yr=oldstats.normalize(oldstats.residual(rankdata(a['y'][ix]),q));values.append(xr.T@yr)
    return np.quantile(values,[.025,.975],axis=0).T


def behavior(m,rows):
    results={};scatter={};audits={};t=time.monotonic()
    for mode in ['primary','global_xi_adjusted','spatial']:
        rr=[]
        for ei,ep in enumerate(dict.fromkeys(h['endpoint'] for h in FIXED)):
            hs=[h for h in FIXED if h['endpoint']==ep];state=hs[0]['state'];ids=m['cohorts'][state]
            if mode=='spatial':
                bad={r['subject'] for r in CON['anomaly_controls']['spatial_le5_voxel_flags'][state].get('1000',{}).get('flagged_runs',[])};ids=[sid for sid in ids if sid not in bad]
            all_ids=list(ids);f=behavior_frame(ids);y=endpoint(f,state,ep);mask=np.isfinite(y);ids=[sid for sid,ok in zip(ids,mask) if ok];f=f.iloc[np.flatnonzero(mask)];y=y[mask]
            allx=np.stack([feature(rows[sid,state]) for sid in ids]);columns=[NAMES.index(h['coalition']) for h in hs];x=allx[:,columns];xi=np.array([rows[sid,state]['system_xi_bits'] for sid in ids])
            a=analysis_object(f,x,y,state,xi,mode=='global_xi_adjusted');p,_=permutation(a,SEED+ei);ci=bootstrap(a,SEED+100+ei)
            audits[mode+':'+ep]=dict(**design_audit(f,state,xi if mode=='global_xi_adjusted' else None),
                missing_endpoint_ids=[sid for sid,ok in zip(all_ids,mask) if not ok])
            for j,h in enumerate(hs):
                rr.append(dict(**h,n=len(ids),rho=float(a['observed'][j]),p=float(p[j]),ci95=ci[j].tolist()))
                if mode=='primary':scatter[h['id']]=dict(subjects=ids,x_rank_residual_div_n=(a['xr'][:,j]/len(ids)).tolist(),y_rank_residual_div_n=(a['yr']/len(ids)).tolist())
            if mode=='primary':
                compat=analysis_object(f,allx if state=='SOCIAL' else x,y,state,xi,legacy=True);lp,lmax=permutation(compat,SEED+500+ei,blocks=(state!='SOCIAL'),max_t=(state=='SOCIAL'))
                for j,h in enumerate(hs):
                    k=NAMES.index(h['coalition']) if state=='SOCIAL' else j;q=next(q for q in rr if q['id']==h['id']);q['legacy_compatible_rho']=float(compat['observed'][k]);q['legacy_compatible_p']=float(lp[k])
                    if lmax is not None:q['legacy_compatible_max_t_120_p']=float(lmax[k])
            print(f'Behavior {mode}: {ep}, N={len(ids)}, {time.monotonic()-t:.1f}s',flush=True)
        for q,p in zip(rr,oldstats.holm([q['p'] for q in rr])):q['p_holm13']=float(p);q['same_direction_significant']=bool(q['rho']*q['direction']>0 and p<.05)
        results[mode]=rr
    # Fixed-n=5 uses fractional effective counts, preserving percentage precision.
    # The primary rule above rounds only after inferring a compatible denominator.
    ids=m['cohorts']['SOCIAL'];f=behavior_frame(ids);y=endpoint(f,'SOCIAL','finite_trial_corrected_dprime',fixed_social_trials=5)
    mask=np.isfinite(y);ids=[sid for sid,ok in zip(ids,mask) if ok];f=f.iloc[np.flatnonzero(mask)];y=y[mask]
    h=next(h for h in FIXED if h['id']=='f');x=np.stack([feature(rows[sid,'SOCIAL']) for sid in ids])[:,[NAMES.index(h['coalition'])]];xi=np.array([rows[sid,'SOCIAL']['system_xi_bits'] for sid in ids])
    a=analysis_object(f,x,y,'SOCIAL',xi);p,_=permutation(a,SEED+200);ci=bootstrap(a,SEED+300)
    sensitivity=dict(n=len(ids),rho=float(a['observed'][0]),p_pointwise=float(p[0]),ci95=ci[0].tolist(),trials=5,
        counts='fractional counts from reported percentages; no rounding',multiplicity='secondary pointwise sensitivity; outside Holm13 primary family')
    dump(OUT/'behavior_statistics.json',dict(results=results,scatter=scatter,design_audits=audits,social_fixed_n5=sensitivity,permutations=PERM,bootstraps=BOOT,seed=SEED,elapsed_seconds=time.monotonic()-t,manifest_sha256=sha(OUT/'manifest.json'),analysis_code_sha256=sha(Path(__file__))))


def legacy_behavior():
    t=time.monotonic();res=oldstats.legacy_gate()
    from scripts import analyze_hcp_all_task_behavior_coalitions_57 as oldsocial
    with np.load(oldsocial.common.SUBJECT_SOURCE) as f:ids=f['subjects'].astype(str)
    _,names,_,x=oldsocial.load_matrix('SOCIAL',ids);scores=oldsocial.social.load_scores(ids)
    result=oldsocial.screen(x,scores['dprime'],oldsocial.base_design(scores['age'],scores['sex']),100000,oldsocial.SEED+oldsocial.TASK_ORDER.index('SOCIAL'))
    ref=json.loads((oldsocial.OUTPUT/'summary.json').read_text())['tasks']['SOCIAL']['winner'];col=names.tolist().index('Vis+Limbic+Cont')
    if abs(result['rho'][col]-ref['rho'])>1e-8 or abs(result['p_max_t'][col]-ref['p_max_t_120'])>1e-12:raise ValueError('Old SOCIAL statistics mismatch')
    res['rows'].append(dict(endpoint='finite_trial_corrected_dprime',coalition='Vis+Limbic+Cont',rho=float(result['rho'][col]),p_raw=float(result['p_raw'][col]),p_max_t_120=float(result['p_max_t'][col])))
    # Original-input model EI tables must reproduce the frozen behavior features.
    maxerr=0.
    for state in ['LANGUAGE','SOCIAL','EMOTION','MOTOR']:
        ss,nn,_,matrix=oldsocial.load_matrix(state,ids)
        for i,sid0 in enumerate(ss):
            sid=str(sid0).removeprefix('sub-');record=next(r for r in json.loads((OUT/'legacy'/f'{sid}.json').read_text())['rows'] if r['variant']==state)
            reconstructed=feature(record);ordered=np.array([reconstructed[NAMES.index(str(n))] for n in nn]);maxerr=max(maxerr,float(np.max(np.abs(ordered-matrix[i]))))
    if maxerr>1e-8:raise ValueError(f'Legacy coalition feature mismatch {maxerr}')
    res.update(hypotheses=13,maximum_original_feature_error_bits=maxerr,elapsed_seconds=time.monotonic()-t,status='passed');dump(OUT/'legacy_behavior_gate.json',res);print(json.dumps(res),flush=True)


def window_summary(m,rows):
    windows={};qc=[]
    for s,ids in m['pairs'].items():
        for sid in ids:
            p=OUT/'windows'/f'{sid}_{s}.json';r=json.loads(p.read_text());windows[sid,s]=r;qc.append(r)
    from scripts.run_hcp_full_replication import window_fingerprint
    if any(r['fingerprint']!=window_fingerprint(m,r['subject'],r['task']) for r in qc):raise ValueError('Stale windows')
    result=dict(primary=panels(m,rows,windows=windows),common=panels(m,rows,m['common'],windows=windows),spatial=panels(m,rows,spatial=True,windows=windows),numerical_audit=dict(pairs=len(qc),window_models=sum(r['n_windows'] for r in qc),minimum_candidate_syn_bits=min(r['numerical_audit']['minimum_candidate_syn_bits'] for r in qc),candidate_count=sum(r['numerical_audit']['candidate_count'] for r in qc),near_zero_negative_count=sum(r['numerical_audit']['near_zero_negative_count'] for r in qc),significant_negative_count=sum(r['numerical_audit']['significant_negative_count'] for r in qc),maximum_closure_error_bits=max(r['maximum_closure_error_bits'] for r in qc)),manifest_sha256=sha(OUT/'manifest.json'))
    result['descriptive']={}
    for s,ids in m['pairs'].items():
        pair={}
        for label,rr in [('matched_rest',[windows[sid,s] for sid in ids]),('task',[rows[sid,s] for sid in ids])]:
            pair[label]=dict(n=len(ids),mean_system_xi_nats=float(np.mean([r['system_xi_bits'] for r in rr])*LN2),
                order_mass_mean_nats=(np.mean([r['order_mass_bits'] for r in rr],axis=0)*LN2).tolist(),
                order_share_mean=np.mean([r['order_share'] for r in rr],axis=0).tolist(),network_percent_mean=np.mean([r['network_percent'] for r in rr],axis=0).tolist())
        pair['task_minus_matched_rest_order_mass_nats']=(np.array(pair['task']['order_mass_mean_nats'])-np.array(pair['matched_rest']['order_mass_mean_nats'])).tolist()
        pair['task_minus_matched_rest_network_percentage_points']=(np.array(pair['task']['network_percent_mean'])-np.array(pair['matched_rest']['network_percent_mean'])).tolist()
        result['descriptive'][s]=pair
    result['numerical_audit']['covariance_floor_count']=sum(r['covariance_floor_count'] for r in qc)
    result['analysis_code_sha256']=sha(Path(__file__))
    dump(OUT/'window_statistics.json',result);print(json.dumps(result['primary']['A']),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['legacy-behavior','main','behavior','windows']);args=p.parse_args()
    if args.phase=='legacy-behavior':legacy_behavior();return
    m=json.loads((OUT/'manifest.json').read_text());rows=load_rows(m)
    if args.phase=='main':main_summary(m,rows)
    elif args.phase=='behavior':behavior(m,rows)
    else:window_summary(m,rows)


if __name__=='__main__':main()
