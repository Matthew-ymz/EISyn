"""Fixed near-zero lambda: blind SPT construction, then planted-module F1.

Reuses the natural-data/shared-MLP pipeline. Ground-truth modules enter the
generator and final scoring, never model selection or SPT split selection.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import henon_high_order_comparison as base
import numpy as np

FOLDER=ROOT/'results/henon_fixed_lambda_module_recovery'
FIG=ROOT/'fig/henon_fixed_lambda_module_recovery'
LAMBDA=.05
MODULES={'A':((1,2,3),(4,5,6)), 'B':((1,2,3,4),(5,6))}
VERSION='henon-fixed005-module-recovery-v1'


def configure():
    base.FOLDER=FOLDER
    base.FIG=FIG
    base.VERSION=VERSION
    base.NETWORKS={k:tuple(tuple(i-1 for i in m) for m in groups) for k,groups in MODULES.items()}
    base.CONFIG.update(version=VERSION,networks=base.NETWORKS,lambdas=[LAMBDA])
    base.ENFORCE_BUDGET=__name__=='__main__'


def partition_f1(predicted,truth):
    """Label-invariant F1 of same-module node pairs at a predeclared two-cut.

    Only root children are predictions. Legitimate refinements inside modules
    are not additional predicted modules and hence are not false positives.
    """
    p=[frozenset(map(int,m)) for m in predicted]
    t=[frozenset(map(int,m)) for m in truth]
    for groups in (p,t):
        if not groups or any(not g for g in groups):raise ValueError('Empty module')
        union=set().union(*groups)
        if sum(map(len,groups))!=len(union):raise ValueError('Modules must be disjoint')
    if set().union(*p)!=set().union(*t):raise ValueError('Partitions must cover the same nodes')
    def pairs(groups):
        return {tuple(pair) for group in groups for pair in itertools.combinations(sorted(group),2)}
    pp,tt=pairs(p),pairs(t)
    nodes=set().union(*t);allpairs=set(itertools.combinations(sorted(nodes),2))
    tp,fp,fn=len(pp&tt),len(pp-tt),len(tt-pp)
    tn=len(allpairs-(pp|tt))
    precision=tp/len(pp) if pp else float(not tt)
    recall=tp/len(tt) if tt else float(not pp)
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
    common=len(set(p)&set(t))
    return dict(precision=precision,recall=recall,f1=f1,tp=tp,fp=fp,fn=fn,tn=tn,
        exact_module_set_precision=common/len(p),exact_module_set_recall=common/len(t),
        exact_module_set_f1=2*common/(len(p)+len(t)),exact_root_partition=set(p)==set(t),
        predicted_modules=[sorted(m) for m in p],true_modules=[sorted(m) for m in t],
        definition='same-module unordered node pairs; fixed root two-cut, no best-cut search')


def score_case(case):
    folder=FOLDER/f'{case}_lambda{LAMBDA:g}'
    ct=json.loads((folder/'conditional_tm.json').read_text())
    if ct.get('status')!='complete':
        result=dict(status='not_scored_invalid_syn',f1=None,reason=ct.get('error'))
    else:
        original=json.dumps(ct['spt']['tree'],sort_keys=True)
        root=ct['spt']['tree']
        result=partition_f1([c['members'] for c in root['children']],MODULES[case])
        assert json.dumps(ct['spt']['tree'],sort_keys=True)==original
        result.update(status='complete',lambda_value=LAMBDA,case=case,
            integrated_ei_nats=ct['integrated_ei_nats'],xi_mc_sem_nats=ct['xi_mc_sem_nats'],
            model_sha256=ct['model_sha256'],tree_sha256=hashlib.sha256(original.encode()).hexdigest(),
            nonnegative_audit=ct['spt']['audit'])
        repeated=[]
        for index,row in enumerate(ct['replicates']):
            try:
                tree=base.spt(np.array(row['table_nats']))['tree']
                metric=partition_f1([c['members'] for c in tree['children']],MODULES[case])
                repeated.append(dict(scramble=index,status='valid',f1=metric['f1'],
                                     exact_root_partition=metric['exact_root_partition']))
            except ArithmeticError as error:
                repeated.append(dict(scramble=index,status='invalid_syn',f1=None,error=str(error)))
        result['numerical_replicates']=repeated
        result['independent_physical_datasets']=1
    base.save(folder/'module_f1.json',result)
    return result


def worker(case):
    configure();base.check()
    folder=FOLDER/f'{case}_lambda{LAMBDA:g}'
    base.worker(case,LAMBDA)
    base.conditional_worker(case,LAMBDA)
    ct=json.loads((folder/'conditional_tm.json').read_text())
    if ct.get('status')!='complete' or ct['inner_prefix_mean_maximum_ei_change_nats']>=.01:
        base.conditional_worker(case,LAMBDA,refine=True)
    metric=score_case(case)
    print(json.dumps(dict(case=case,stage='module_f1',result=metric)),flush=True)
    base.pcmci_worker(case,LAMBDA)


def precision_worker(case):
    """One data-blind refinement: root cuts disagreed across numerical scrambles."""
    configure();base.check()
    folder=FOLDER/f'{case}_lambda{LAMBDA:g}'
    original=folder/'conditional_tm_n1024.json'
    if not original.exists():
        base.save(original,json.loads((folder/'conditional_tm.json').read_text()))
        base.save(folder/'module_f1_n1024.json',json.loads((folder/'module_f1.json').read_text()))
    path=folder/'conditional_tm_n4096_work.json'
    result=json.loads(path.read_text()) if path.exists() else dict(status='running',replicates=[],
        refinement_reason='root cuts disagreed across numerical scrambles; no ground-truth criterion',
        coarse_grid='conditional_tm_n1024.json',outer=4096,inner=256)
    model=base.MLP(json.loads((folder/'mlp.json').read_text()));result['model_sha256']=model.digest()
    for repeat in range(4):
        if len(result['replicates'])>repeat:continue
        table,diag=base.conditional_tm_table(model,4096,256,seed=9021+repeat*1000)
        result['replicates'].append(dict(table_nats=table.tolist(),diagnostic=diag))
        base.save(path,result);print(f'{case} high precision {repeat+1}/4',flush=True)
    tables=np.array([r['table_nats'] for r in result['replicates']]);mean=tables.mean(0)
    result.update(ei_table_nats=mean.tolist(),integrated_ei_nats=float(mean[63]-tables[:,[1,2,4,8,16,32]].mean(0).sum()),
        xi_mc_sem_nats=float(np.std(tables[:,63]-tables[:,[1,2,4,8,16,32]].sum(1),ddof=1)/2),
        maximum_ei_mc_sem_nats=float((tables.std(0,ddof=1)/2).max()),
        inner_prefix_mean_maximum_ei_change_nats=float(abs(mean-np.mean([r['diagnostic']['inner_prefix_table_nats'] for r in result['replicates']],axis=0)).max()))
    try:result['spt']=base.spt(mean);result['status']='complete'
    except ArithmeticError as error:result.update(status='numerically_unresolved',error=str(error))
    base.save(path,result);base.save(folder/'conditional_tm.json',result)
    score_case(case)


def launch_precision():
    configure();budget=json.loads((FOLDER/'budget.json').read_text());jobs=[]
    protocol=json.loads((FOLDER/'protocol.json').read_text())
    protocol['numerical_refinement']=dict(outer=4096,inner=256,scrambles=4,
        criterion='coarse numerical root cuts disagree, irrespective of planted labels',
        recorded_epoch=time.time(),model_and_generator_changed=False)
    base.save(FOLDER/'protocol.json',protocol)
    try:
        for case in MODULES:
            folder=FOLDER/f'{case}_lambda{LAMBDA:g}'
            work=folder/'conditional_tm_n4096_work.json'
            if work.exists() and json.loads(work.read_text()).get('status')=='complete':continue
            base.check();log=(folder/'precision.log').open('a')
            p=subprocess.Popen([sys.executable,__file__,'--precision-worker',case],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append((case,p,log));print(f'{case}: precision pid={p.pid}',flush=True)
        for case,p,_ in jobs:
            try:code=p.wait(timeout=max(.01,budget['deadline_epoch']-time.time()))
            except subprocess.TimeoutExpired:
                for _,child,_ in jobs:
                    if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
                raise TimeoutError('Shared fixed-point deadline reached during refinement')
            if code:base.save(FOLDER/f'{case}_precision_failure.json',dict(exit_code=code))
    finally:
        for _,_,log in jobs:log.close()


def launch():
    configure();FOLDER.mkdir(parents=True,exist_ok=True)
    budget_path=FOLDER/'budget.json'
    if not budget_path.exists():
        start=time.time()
        base.save(budget_path,dict(started_epoch=start,deadline_epoch=start+1200,budget_seconds=1200,
            scope='User follow-up: fixed near-zero lambda, planted modules and SPT F1; narrower two-network lightweight execution',
            relation_to_previous='new fixed-point follow-up; previous 45-minute budget is preserved and never reset'))
    budget=json.loads(budget_path.read_text())
    base.save(FOLDER/'protocol.json',dict(config=base.CONFIG,planted_modules=MODULES,
        cut='root children, K=2 fixed before fitting',primary_f1='same-module node-pair F1',
        secondary_f1='exact module-set F1 at the same root cut',
        topology_change='B overlapping four-source hyperedges replaced by disjoint four-plus-two modules; old B retained separately',
        lambda_selected_before_run=LAMBDA,select_using_f1=False,
        manuscript=dict(parent='P6UJCVG8',main='DXGC7JEA',si='MWIWKSVG',
            evidence='re-fetched full text this task; Methods5-12, SI S1.2/S5.1/S6.4',
            version='only one unversioned main/SI pair; metadata dates do not establish manuscript version')))
    processes=[]
    try:
        for case in MODULES:
            folder=FOLDER/f'{case}_lambda{LAMBDA:g}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'module_f1.json').exists() and (folder/'pcmci.json').exists():continue
            base.check();log=(folder/'run.log').open('a')
            p=subprocess.Popen([sys.executable,__file__,'--worker',case],stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
            processes.append((case,p,log));print(f'{case}: process={p.pid}',flush=True)
        for case,p,log in processes:
            remaining=budget['deadline_epoch']-time.time()
            try:code=p.wait(timeout=max(.01,remaining))
            except subprocess.TimeoutExpired:
                for _,child,_ in processes:
                    if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
                raise TimeoutError('Shared fixed-point follow-up 20-minute cap exhausted')
            if code:base.save(FOLDER/f'{case}_failure.json',dict(exit_code=code,log=str(FOLDER/f'{case}_lambda{LAMBDA:g}/run.log')))
        base.save(FOLDER/'execution.json',dict(finished_epoch=time.time(),elapsed_seconds=time.time()-budget['started_epoch'],
            within_budget=time.time()<budget['deadline_epoch']))
    finally:
        for _,_,log in processes:log.close()


def audit():
    """Read-only verification of cached samples, frozen models and scored trees."""
    configure();rows={}
    for case,groups in MODULES.items():
        folder=FOLDER/f'{case}_lambda{LAMBDA:g}'
        read=lambda name:json.loads((folder/name).read_text())
        core,ct,score,pcmci=[read(n) for n in ('metrics.json','conditional_tm.json','module_f1.json','pcmci.json')]
        model=base.MLP(read('mlp.json'));manifest=read('data_manifest.json')
        with np.load(folder/'natural.npz') as data:
            states,latent,order=[data[k] for k in ('states','latent','chain_order')]
        partitions={name:(states[:,chains].reshape(-1,6),latent[:,chains].reshape(-1,6))
                    for name,chains in [('train',order[:48]),('validation',order[48:56]),('test',order[56:])]}
        assert np.array_equal(np.tanh(latent[:-1]),states[1:])
        assert len(set(order))==64
        for name,pair in partitions.items():assert base.sha(*pair)==manifest['partitions'][name]['sha256']
        assert base.sha(states,latent)==manifest['array_sha256']
        assert model.payload['training_pairs_sha256']==base.sha(*partitions['train'])==core['neural_granger']['training_sha256']
        assert core['natural_information']['data_sha256']==base.sha(*partitions['test'])==pcmci['data_sha256']
        assert model.digest()==core['model_sha256']==core['shap']['model_sha256']==ct['model_sha256']==score['model_sha256']
        rng=np.random.default_rng(711)
        bg=partitions['train'][0][rng.choice(12288,32,replace=False)]
        fg=partitions['test'][0][rng.choice(2048,128,replace=False)]
        assert base.sha(bg)==core['shap']['background_sha256']
        assert base.sha(fg)==core['shap']['evaluation_sha256']
        tree=ct['spt']['tree'];tree_text=json.dumps(tree,sort_keys=True)
        assert hashlib.sha256(tree_text.encode()).hexdigest()==score['tree_sha256']
        assert ct['spt']['node_count']==5 and abs(ct['spt']['closure_error_nats'])<1e-9
        assert ct['spt']['audit']['significant_violation_count']==0
        truth_score=partition_f1([c['members'] for c in tree['children']],groups)
        assert truth_score['f1']==score['f1']
        # Diagnostics use the same fixed uniform rows as the original fit check.
        # Planted labels enter only here, after the frozen tree has been scored.
        u=base.qmc.Sobol(6,scramble=True,seed=511).random_base2(12)*2-1
        truth=base.mean_dynamics(u,case,LAMBDA);prediction=model.mean(u)
        baseline=.5*(1-1.4*u*u)+.4*(.3+1.7*LAMBDA)*u
        module_fit=[]
        for group in groups:
            cols=np.array(group)-1
            error=float(np.sqrt(np.mean((prediction[:,cols]-truth[:,cols])**2)))
            signal=float(np.sqrt(np.mean((truth[:,cols]-baseline[:,cols])**2)))
            module_fit.append(dict(members=group,mean_function_rmse=error,joint_signal_rms=signal,error_to_signal_rms=error/signal))
        xi=np.asarray(ct['spt']['all_xi_nats'])
        cuts=sorted([(float(xi[l]+xi[r]),base.members(l),base.members(r)) for l,r in base.splits(63)],reverse=True)
        replicates=score['numerical_replicates']
        rows[case]=dict(shared_data_and_mlp_verified=True,model_sha256=model.digest(),
            train_pairs=12288,validation_pairs=2048,test_pairs=2048,pcmci_effective_pairs=2040,
            tree_sha256=score['tree_sha256'],f1=score['f1'],nonnegative_audit=ct['spt']['audit'],
            closure_error_nats=ct['spt']['closure_error_nats'],module_fit=module_fit,
            root_objective_gap_to_second_cut_nats=cuts[0][0]-cuts[1][0],
            valid_numerical_replicates=sum(v['status']=='valid' for v in replicates),
            exact_numerical_root_recoveries=sum(v.get('exact_root_partition',False) for v in replicates))
    budget=json.loads((FOLDER/'budget.json').read_text())
    computation_finished=max((FOLDER/f'{case}_lambda{LAMBDA:g}'/'conditional_tm.json').stat().st_mtime for case in MODULES)
    result=dict(status='passed',cases=rows,computation_finished_epoch=computation_finished,
        computation_elapsed_seconds=computation_finished-budget['started_epoch'],
        within_fixed_point_budget=computation_finished<=budget['deadline_epoch'],
        scope='sample/model/tree verification; per-module error is post-fit diagnostic only')
    base.save(FOLDER/'completion_audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['A','B']);p.add_argument('--precision-worker',choices=['A','B']);p.add_argument('--precision',action='store_true');p.add_argument('--audit',action='store_true');args=p.parse_args()
    if args.worker:worker(args.worker)
    elif args.precision_worker:precision_worker(args.precision_worker)
    elif args.precision:launch_precision()
    elif args.audit:audit()
    else:launch()
