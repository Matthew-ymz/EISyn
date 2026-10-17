"""Approved, deadline-limited six-node topology search; no subagents."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.special import ndtr
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.kuramoto_six_topology_core import (
    N,TOL,JointIntegrator,LinearChannel,MLPChannel,configuration,dataset,digest,check,
    fit_mlp,channel_diagnostics,recovery,save,spt_from_ei,subsets,
)
from scripts.compare_kuramoto_short_trajectory_clustering import analyze_view

FOLDER=ROOT/'results/kuramoto_six_topology_search'
LOG=ROOT/'docs/log/kuramoto_six_topology_search'
HIERARCHICAL=('ward','correlation_hac','dcor_hac','mi_hac','mic')


def read(path):return json.loads(Path(path).read_text())
def budget():return read(FOLDER/'budget.json')
def boundary(phase,**fields):
    value=dict(phase=phase,updated_epoch=time.time(),pid=os.getpid(),**fields)
    save(FOLDER/'status.json',value)
    (LOG/'live_status.md').write_text('# 实验状态\n\n'+json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(value,ensure_ascii=False),flush=True)


def registry():
    unique=[];aliases={};seen={}
    for family in 'ABCD':
        for ratio in (.5,1.,2.):
            cfg=configuration(family,ratio)
            if cfg['mechanism_hash'] in seen:aliases[cfg['id']]=seen[cfg['mechanism_hash']]
            else:unique.append(cfg);seen[cfg['mechanism_hash']]=cfg['id']
    payload=dict(configurations=unique,aliases=aliases,units='nats',source='iid uniform(-pi,pi)^6',
                 target='complete unwrapped future phase, tau=.2',syn_tolerance_nats=TOL,
                 seed_discovery=0,seeds_validation=[1,2,3],baseline_views=['increment','transition'],
                 baseline_rank='max over all five hierarchical methods and both views',
                 fit_eligibility=dict(mean_nrmse_max=.15,test_joint_coverage_interval=[.88,.995]),
                 gain_target=dict(spt_min=.8,gap_min=.25),maximum_spt_candidates=3,
                 manuscript=dict(parent='P6UJCVG8',main='DXGC7JEA',si='MWIWKSVG',version='no explicit draft date/version',
                                  read_sections='Methods5-12;SI S5/S7'))
    save(FOLDER/'registry.json',payload);save(LOG/'search_space.json',payload)
    return payload


def preflight(deadline):
    path=FOLDER/'preflight.json'
    if path.exists():
        cached=read(path)
        if cached.get('control_nonnegativity_checked') and cached['passed']:return cached
        save(FOLDER/'preflight_initial.json',cached)
    boundary('numerical_preflight')
    started=time.time()
    uniform=JointIntegrator(LinearChannel(np.eye(N),np.eye(N)*.01),512,256,711,deadline)
    e,points,_=uniform.information()
    py=np.log(ndtr((uniform.y+np.pi)/.1)-ndtr((uniform.y-np.pi)/.1))-np.log(2*np.pi)
    conditional=-.5*((uniform.y-uniform.x)/.1)**2-np.log(.1*np.sqrt(2*np.pi))
    exact=[]
    for mask in range(1,64):
        exact.append(float(sum((conditional[:,i]-py[:,i]) for i in range(N) if mask>>i&1).mean()))
    uniform_error=float(np.max(abs(np.array(exact)-np.array(list(e.values())))))
    b=np.eye(N)+.06*(np.ones((N,N))-np.eye(N));cov=.12*np.eye(N)+.01*np.ones((N,N))
    gaussian=JointIntegrator(LinearChannel(b,cov),1024,256,712,deadline,prior='normal')
    ge,_,_=gaussian.information(); cy=b@b.T+cov
    expected=[]
    for mask in range(1,64):
        missing=[i for i in range(N) if not mask>>i&1]
        cc=cov+b[:,missing]@b[:,missing].T
        expected.append(.5*(np.linalg.slogdet(cy)[1]-np.linalg.slogdet(cc)[1]))
    gaussian_error=float(np.max(abs(np.array(expected)-np.array(list(ge.values())))))
    uniform_audit=spt_from_ei(e)['audit'];gaussian_audit=spt_from_ei(ge)['audit']
    payload=dict(uniform_max_ei_error_nats=uniform_error,gaussian_max_ei_error_nats=gaussian_error,
                 uniform_syn_audit=uniform_audit,gaussian_syn_audit=gaussian_audit,
                 gaussian_prior_is_test_only=True,control_nonnegativity_checked=True,
                 seconds=time.time()-started,passed=bool(uniform_error<.01 and gaussian_error<.02
                    and uniform_audit['violation_count']==uniform_audit['xi_violation_count']==0
                    and gaussian_audit['violation_count']==gaussian_audit['xi_violation_count']==0))
    save(path,payload)
    if not payload['passed']:raise ArithmeticError(f'Numerical preflight failed: {payload}')
    # Registered before viewing any learned-channel topology recovery score.
    integration=dict(ladder=[[128,128],[512,256],[2048,512]],scrambles=[880,1880],
                     max_grid_ei_change_nats=.01,max_scramble_ei_sem_nats=.01,
                     minimum_ess=10,maximum_weight=.25,
                     refinement='advance only if numerical precision is unresolved, never based on F1')
    save(FOLDER/'integration_protocol.json',integration)
    return payload


def baselines(x,y,cfg):
    delta=y-x
    views={}
    for view,blocks in (('increment',delta[:,:,None]),('transition',np.stack((np.sin(x),np.cos(x),delta),axis=2))):
        result=analyze_view(blocks,kout=.04)
        for name,row in result['methods'].items():
            for key in ('module_123_retained','module_456_retained','original_partition_ari','exact_original_two_groups','ari_interpretation'):
                row.pop(key,None)
            if row['kind']=='hierarchy':row['recovery']=recovery(row['tree'],cfg['supports'])
        views[view]=result
    best=max(views[v]['methods'][m]['recovery']['module_f1'] for v in views for m in HIERARCHICAL)
    return dict(views=views,best_hierarchical_f1=best)


def screen(cfg,seed,deadline):
    folder=FOLDER/f"{cfg['id']}_seed{seed}";folder.mkdir(exist_ok=True)
    summary_path=folder/'screen.json'
    if summary_path.exists():return read(summary_path)
    boundary('screening',candidate=cfg['id'],seed=seed)
    started=time.time();x,y,manifest=dataset(cfg,seed)
    save(folder/'data_manifest.json',manifest);save(folder/'configuration.json',cfg)
    baseline_path=folder/'baselines_raw.json'
    if baseline_path.exists():baseline=read(baseline_path)
    else:
        train_stop=manifest['split'][0]
        baseline=baselines(x[:train_stop],y[:train_stop],cfg);save(baseline_path,baseline)
    model_path=folder/'model.json'
    if model_path.exists():model=MLPChannel(read(model_path));fit=read(folder/'fit.json')
    else:
        model,fit=fit_mlp(x,y,seed,deadline);save(model_path,model.payload);save(folder/'fit.json',fit)
    diagnostic=channel_diagnostics(model,x,y,cfg,seed)
    save(folder/'diagnostics.json',diagnostic)
    summary=dict(id=cfg['id'],family=cfg['family'],ratio=cfg['ratio'],seed=seed,
                 best_baseline_f1=baseline['best_hierarchical_f1'],diagnostics=diagnostic,
                 model_sha256=digest(*model.weights,*model.biases,model.covariance),
                 data_manifest=manifest,seconds=time.time()-started,status='screened')
    save(summary_path,summary)
    with (LOG/'run_history.jsonl').open('a') as f:f.write(json.dumps(summary,ensure_ascii=False)+'\n')
    return summary


def evaluate(cfg,seed,deadline):
    folder=FOLDER/f"{cfg['id']}_seed{seed}";model=MLPChannel(read(folder/'model.json'))
    protocol=read(FOLDER/'integration_protocol.json');all_levels=[];result_path=folder/'spt.json'
    if result_path.exists():return read(result_path)
    for outer,particles in protocol['ladder']:
        rows=[]
        for scramble in protocol['scrambles']:
            path=folder/f'ei_N{outer}_P{particles}_R{scramble}.json'
            if path.exists():r=read(path)
            else:
                check(deadline);boundary('joint_ei',candidate=cfg['id'],seed=seed,outer=outer,particles=particles,scramble=scramble)
                started=time.time();integrator=JointIntegrator(model,outer,particles,scramble+10000*seed,deadline)
                values,points,diagnostics=integrator.information()
                r=dict(outer=outer,particles=particles,scramble=scramble,ei_nats=values,
                       diagnostics=diagnostics,seconds=time.time()-started)
                save(path,r);np.savez_compressed(path.with_suffix('.npz'),pointwise_ei=points)
            rows.append(r)
        data=np.array([[r['ei_nats'][str(i)] if str(i) in r['ei_nats'] else r['ei_nats'][i] for i in range(1,64)] for r in rows])
        mean=data.mean(0);sem=data.std(0,ddof=1)/np.sqrt(len(rows))
        spt=spt_from_ei(dict(zip(range(1,64),mean)))
        if spt['tree']:spt['recovery']=recovery(spt['tree'],cfg['supports'])
        diagnostics=[d for r in rows for d in r['diagnostics'].values()]
        grid_change=float(np.max(abs(mean-np.array(all_levels[-1]['mean_ei_nats'])))) if all_levels else None
        spt.update(outer=outer,particles=particles,mean_ei_nats=mean.tolist(),maximum_ei_sem_nats=float(sem.max()),
                   grid_change_nats=grid_change,minimum_ess=min(d['ess_min'] for d in diagnostics),
                   maximum_weight=max(d['maximum_weight'] for d in diagnostics))
        all_levels.append(spt);save(folder/'spt_levels.json',all_levels)
        if spt['status']=='valid' and grid_change is not None:
            numerical=(grid_change<protocol['max_grid_ei_change_nats'] and sem.max()<protocol['max_scramble_ei_sem_nats']
                       and spt['minimum_ess']>=protocol['minimum_ess'] and spt['maximum_weight']<=protocol['maximum_weight'])
            # Same mean EI is not an independent replication: compare module-scored trees from both scrambles.
            replicate_trees=[spt_from_ei(dict(zip(range(1,64),row))) for row in data]
            scores=[recovery(r['tree'],cfg['supports'])['scored_clades'] if r['tree'] else None for r in replicate_trees]
            stable=all(s is not None for s in scores) and {tuple(sorted(tuple(a) for a in s)) for s in scores}.__len__()==1
            spt['scramble_module_tree_stable']=stable
            if numerical and stable:break
    final=copy.deepcopy(all_levels[-1]);final['levels_completed']=len(all_levels)
    final['numerically_resolved']=bool(final['status']=='valid' and final.get('grid_change_nats') is not None
                                      and final['grid_change_nats']<.01 and final['maximum_ei_sem_nats']<.01
                                      and final['minimum_ess']>=10 and final['maximum_weight']<=.25
                                      and final.get('scramble_module_tree_stable',False))
    baseline=read(folder/'baselines_raw.json');final['best_baseline_f1']=baseline['best_hierarchical_f1']
    final['gap']=final['recovery']['module_f1']-final['best_baseline_f1'] if final.get('tree') else None
    final['fit_eligible']=read(folder/'diagnostics.json')['fit_eligible']
    final['meets_search_target']=bool(final['numerically_resolved'] and final['fit_eligible']
                                     and final['recovery']['module_f1']>=.8 and final['gap']>=.25)
    save(result_path,final)
    # Same frozen model readout, supplied equally to clustering; distinct from real observed data.
    generated=folder/'baselines_model.json'
    if not generated.exists():
        rng=np.random.default_rng(922+seed);x=rng.uniform(-np.pi,np.pi,(4096,N))
        save(generated,baselines(x,model.sample(x,rng.normal(size=x.shape)),cfg))
    return final


def worker():
    deadline=budget()['deadline_epoch'];reg=registry();preflight(deadline)
    summaries=[]
    for cfg in reg['configurations']:
        check(deadline)
        try:summaries.append(screen(cfg,0,deadline))
        except TimeoutError:raise
        except Exception as error:
            failed=dict(id=cfg['id'],status='failed',error=str(error),traceback=traceback.format_exc())
            summaries.append(failed);save(FOLDER/f"{cfg['id']}_seed0"/'failure.json',failed)
        save(FOLDER/'screen_summary.json',summaries)
    eligible=[s for s in summaries if s.get('diagnostics',{}).get('fit_eligible')]
    eligible.sort(key=lambda s:(s['best_baseline_f1'],s['id']))
    selected=eligible[:3];save(FOLDER/'candidate_selection.json',dict(rule='eligible then minimum strongest baseline; id tie break',selected=[s['id'] for s in selected]))
    results=[]
    for s in selected:
        check(deadline);cfg=next(c for c in reg['configurations'] if c['id']==s['id'])
        try:r=evaluate(cfg,0,deadline);results.append(dict(id=s['id'],result=r))
        except TimeoutError:raise
        except Exception as error:
            save(FOLDER/f"{cfg['id']}_seed0"/'evaluation_failure.json',dict(error=str(error),traceback=traceback.format_exc()))
    save(FOLDER/'discovery_results.json',results)
    qualified=[r for r in results if r['result']['meets_search_target']]
    if qualified:
        best=max(qualified,key=lambda r:(r['result']['gap'],r['result']['recovery']['module_f1']))
        save(FOLDER/'frozen_selection.json',best)
        cfg=next(c for c in reg['configurations'] if c['id']==best['id'])
        validations=[]
        for seed in (1,2,3):
            check(deadline);screen(cfg,seed,deadline)
            result=evaluate(cfg,seed,deadline);validations.append(dict(seed=seed,result=result))
            save(FOLDER/'validation_results.json',validations)
    boundary('complete',screened=len(summaries),evaluated=len(results),qualified=len(qualified))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();LOG.mkdir(parents=True,exist_ok=True)
    if args.worker:
        with threadpool_limits(limits=2):worker()
        return
    remaining=budget()['deadline_epoch']-time.time()
    if remaining<=0:raise TimeoutError('Existing absolute budget exhausted; cannot reset')
    log=(LOG/'run.log').open('a')
    process=subprocess.Popen([sys.executable,__file__,'--worker'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
    save(FOLDER/'process.json',dict(parent_pid=os.getpid(),worker_pid=process.pid,log=str(LOG/'run.log'),deadline_epoch=budget()['deadline_epoch']))
    try:code=process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        code=124;boundary('budget_exhausted')
    save(FOLDER/'exit.json',dict(code=code,finished_epoch=time.time(),elapsed_from_budget_start=time.time()-budget()['started_epoch']))
    log.close()
    if code:sys.exit(code)


if __name__=='__main__':main()
