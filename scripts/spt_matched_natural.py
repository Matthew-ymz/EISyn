"""Common new-network natural trajectories, exact SHAP, and fixed-target WMS."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.spt_matched_common import FOLDER,deadline,fitted_model,read,state,guarded,model_sha
from scripts.spt_glv_simplified_a_pilot import MatchedSimplifiedGLV
from scripts.spt_glv_natural_baselines import source_tm,shap_profile,profile_tree,NaturalInformation,fixed_target_wms
from scripts.spt_kuramoto_natural_baselines import information_audit
from scripts.spt_kuramoto_finite_tm import check_time


def natural_states(end):
    path=FOLDER/'glv_natural_chains.npz'
    frozen=read(ROOT/'results/spt_local_kuramoto_tm/ac_comparison/A/frozen_model.json')
    generator=MatchedSimplifiedGLV({tuple(r['edge']):r['weight'] for r in frozen['network']})
    meta=dict(case='matched_simplified_A',generator=generator.metadata(),stream=[20261015,0,90001],chains=64,burn=256,retained_steps=512,tau=.2,dt=.0025)
    meta=json.loads(json.dumps(meta))
    if path.exists():
        with np.load(path,allow_pickle=False) as data:
            if json.loads(str(data['meta']))!=meta:raise ValueError('Natural trajectory provenance changed')
            return data['states'].copy(),meta
    rng=np.random.default_rng(np.random.SeedSequence(meta['stream']));current=generator.sample_source(64,rng)
    for _ in range(meta['burn']):
        check_time(end);current=generator.step(current,rng,dt=.0025)
    states=[current.copy()]
    for _ in range(meta['retained_steps']):
        check_time(end);current=generator.step(current,rng,dt=.0025);states.append(current.copy())
    states=np.asarray(states);temporary=path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,states=states,meta=json.dumps(meta));temporary.replace(path)
    return states,meta


def worker():
    end=deadline();model=fitted_model();path=FOLDER/'natural.json'
    result=read(path) if path.exists() else dict(status='running',methods={})
    result.update(status='running',stage='natural_sampling',model_sha=model_sha(model));state('natural',result)
    try:
        states,meta=natural_states(end);mean,cov,prior_audit=source_tm(states)
        result.update(natural_sampling=meta,natural_data_sha256=hashlib.sha256((FOLDER/'glv_natural_chains.npz').read_bytes()).hexdigest(),source_tm_audit=prior_audit)
        state('natural',result)
        if 'shap_profiles' not in result['methods']:
            result['stage']='shap_profiles';state('natural',result)
            train=states[:256].reshape(-1,9);hold=states[256:512].reshape(-1,9)
            profiles=result.setdefault('shap_replicates',[])
            for repeat in range(4):
                if any(r['repeat']==repeat for r in profiles):continue
                rng=np.random.default_rng(np.random.SeedSequence([20261015,0,91000+repeat]))
                p,audit=shap_profile(model,hold[rng.choice(len(hold),256,replace=False)],train[rng.choice(len(train),128,replace=False)],end)
                profiles.append(dict(repeat=repeat,profiles=p.tolist(),audit=audit));state('natural',result)
            mean_profile=np.mean([r['profiles'] for r in profiles],axis=0)
            result['methods']['shap_profiles']=dict(status='screen_passed',metrics=profile_tree(mean_profile),
                integration_replicates=[profile_tree(np.asarray(r['profiles'])) for r in profiles],profiles=mean_profile.tolist(),
                adaptation='exact512 coalitions; joint empirical missing-block replacement; cosine/single output-profile tree',
                repeat_unit='four evaluation/background subsamples of one physical dataset')
            state('natural',result)
        if not prior_audit['passed']:
            result['methods']['wms_natural']=dict(status='natural_source_tm_diagnostics_failed',metrics=None)
        elif 'wms_natural' not in result['methods']:
            result['stage']='wms_natural';state('natural',result)
            records=result.setdefault('wms_replicates',[])
            for repeat in range(4):
                row=next((r for r in records if r['scramble']==repeat),None)
                if row is None:
                    row=dict(scramble=repeat,information_nats={},diagnostics={});records.append(row)
                estimator=NaturalInformation(model,mean,cov,outer=1024,seed=20261015+1000*repeat,deadline=end,cache_dir=FOLDER/'glv_natural_marginals')
                for mask in range(1,512):
                    if str(mask) in row['information_nats']:continue
                    value,diags=estimator.information(mask,64);row['information_nats'][str(mask)]=value;row['diagnostics'][str(mask)]=diags
                    if mask%32==0:state('natural',result);print(f'WMS repeat={repeat} subsets={mask}/511',flush=True)
                state('natural',result)
            samples=np.array([[r['information_nats'][str(m)] for m in range(1,512)] for r in records]);sem=float(samples.std(0,ddof=1).max()/2)
            ds=[d for r in records for pair in r['diagnostics'].values() for d in pair if d.get('ess_minimum') is not None]
            changes=[]
            for repeat in range(4):
                estimator=NaturalInformation(model,mean,cov,outer=1024,seed=20261015+1000*repeat,deadline=end,cache_dir=FOLDER/'glv_natural_marginals')
                for mask in (1,3,7,24,31,96,384,480,511):
                    fine,_=estimator.information(mask,256);changes.append(abs(fine-records[repeat]['information_nats'][str(mask)]))
            audit=information_audit(samples);issues=[]
            if sem>=.01:issues.append('information_SEM>=.01')
            if max(changes)>=.01:issues.append('inner_change>=.01')
            if min(d['ess_minimum'] for d in ds)<10 or max(d['maximum_weight'] for d in ds)>=.25:issues.append('importance_weights_failed')
            if not audit['passed']:issues.append('MI_nonnegativity_or_inclusion_failed')
            method=dict(status='numerically_unresolved' if issues else 'screen_passed',metrics=None,failed_checks=issues,
                maximum_information_sem_nats=sem,maximum_inner_change_nats=max(changes),minimum_ess=min(d['ess_minimum'] for d in ds),
                maximum_weight=max(d['maximum_weight'] for d in ds),information_audit=audit,
                target='fixed complete nine-dimensional future',source='full-covariance natural Gaussian source TM approximation; predictor unchanged')
            if not issues:
                metrics,wms_audit=fixed_target_wms({m:float(v) for m,v in zip(range(1,512),samples.mean(0))})
                method.update(metrics=metrics,audit=wms_audit,integration_replicates=[fixed_target_wms({m:r['information_nats'][str(m)] for m in range(1,512)})[0] for r in records])
            result['methods']['wms_natural']=method;state('natural',result)
        result.update(status='complete',stage='ended');state('natural',result)
    except Exception as error:
        result.update(status='budget_exhausted' if isinstance(error,TimeoutError) else 'failed',error=f'{type(error).__name__}: {error}')
        state('natural',result)
        if not isinstance(error,TimeoutError):raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    worker() if args.worker else guarded('natural',__file__,worker,resume=args.resume)
