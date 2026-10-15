"""Natural fixed-target WMS and exact-coalition SHAP for the frozen gLV TM.

WMS keeps the nonlinear transition TM; only the source prior is replaced by
a full-covariance Gaussian source TM fitted to true, unshuffled gLV chains.
This source-density approximation is explicit and diagnosed, not a joint
Gaussian replacement of the nonlinear predictor. SHAP uses empirical natural
background rows, all512 coalitions, nine players/outputs, cosine/single tree.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import os
import sys
import time

import numpy as np
from scipy.special import ndtri,logsumexp
from scipy.stats import qmc,skew,kurtosis
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.spt_glv_finite_tm import RealConditionalTM,check_time,save_json
from scripts.spt_local_hyperedge_glv import LocalHyperedgeGLV
from scripts.spt_glv_a_comparison import quality,splits,labels

VERSION='glv-natural-wms-shap-v1'
DIRECTORY=ROOT/'results/spt_local_glv_tm/a_comparison'
MEMBERSHIP=((np.arange(512)[:,None]>>np.arange(9))&1).astype(bool)


def natural_data(directory,deadline):
    path=Path(directory)/'natural_chains.npz'
    meta=dict(version=VERSION,case='A',seed=0,stream=[20261015,0,90001],chains=64,burn=256,
        retained_steps=512,tau=.2,dt=.0025,trajectory='adjacent states; no independent node resampling')
    if path.exists():
        with np.load(path,allow_pickle=False) as data:
            if json.loads(str(data['meta']))!=meta: raise ValueError('Natural sample protocol mismatch')
            return data['states'].copy(),meta
    model=LocalHyperedgeGLV('A'); rng=np.random.default_rng(np.random.SeedSequence(meta['stream']))
    state=model.sample_source(64,rng)
    for step in range(meta['burn']):
        check_time(deadline); state=model.step(state,rng,dt=.0025)
    print(json.dumps(dict(stage='natural_burn_complete',steps=meta['burn'])),flush=True)
    states=[state.copy()]
    for step in range(meta['retained_steps']):
        check_time(deadline); state=model.step(state,rng,dt=.0025); states.append(state.copy())
    states=np.asarray(states)
    temporary=path.with_suffix('.tmp.npz'); np.savez_compressed(temporary,meta=json.dumps(meta),states=states); temporary.replace(path)
    return states,meta


def source_tm(states):
    train=states[:256].reshape(-1,9); hold=states[256:512].reshape(-1,9)
    mean=train.mean(0); covariance=np.cov(train,rowvar=False); chol=np.linalg.cholesky(covariance)
    white=np.linalg.solve(chol,(hold-mean).T).T
    hold_cov=np.cov(hold,rowvar=False)
    delta=float(abs((hold.mean(0)-mean)/np.sqrt(np.diag(covariance))).max())
    cov_delta=float(np.linalg.norm(hold_cov-covariance)/np.linalg.norm(covariance))
    max_skew=float(abs(skew(white,axis=0,bias=False)).max())
    max_kurt=float(abs(kurtosis(white,axis=0,fisher=True,bias=False)).max())
    corr=np.corrcoef(train,rowvar=False)-np.eye(9)
    audit=dict(train_source_states=len(train),heldout_source_states=len(hold),
        full_source_covariance=covariance.tolist(),mean=mean.tolist(),
        maximum_absolute_source_correlation=float(abs(corr).max()),
        mean_shift_in_source_sd=delta,relative_covariance_shift=cov_delta,
        maximum_whitened_skewness=max_skew,maximum_whitened_excess_kurtosis=max_kurt,
        source_approximation='full covariance Gaussian Cholesky source TM; transition TM remains nonlinear',
        source_tm_diagnostic_gates=dict(mean_shift_sd=.25,covariance_shift=.25,skewness=.5,excess_kurtosis=1.),
        outside_original_intervention_support_count=int(np.count_nonzero(abs(states)>np.pi)))
    audit['passed']=delta<.25 and cov_delta<.25 and max_skew<.5 and max_kurt<1.
    return mean,covariance,audit


class NaturalInformation:
    """Linear-Gaussian reference is an importance control, not the density model."""
    def __init__(self,model,mean,covariance,*,outer,seed,deadline,cache_dir):
        self.model=model; self.mean=mean; self.cov=covariance; self.outer=outer; self.deadline=deadline
        u=qmc.Sobol(18,scramble=True,seed=seed).random_base2(int(np.log2(outer)))
        self.source=mean+ndtri(u[:,:9])@np.linalg.cholesky(covariance).T
        self.target=model.sample(self.source,u[:,9:])
        mean0,jac=model.mean(mean[None,:],jacobian=True)
        self.jac=jac[0]; self.intercept=mean0[0]-self.jac@mean
        self.inner=qmc.Sobol(9,scramble=True,seed=seed+3000000).random_base2(int(np.log2(outer*256))).reshape(outer,256,9)
        self.cache={}
        self.meta=dict(version=VERSION,outer=outer,seed=seed,
            mean=mean.tolist(),covariance=covariance.tolist(),model_sha=hashlib.sha256(
                json.dumps(model.payload(),sort_keys=True).encode()).hexdigest())
        token=hashlib.sha256(json.dumps(self.meta,sort_keys=True).encode()).hexdigest()[:24]
        self.directory=Path(cache_dir)/token; self.directory.mkdir(parents=True,exist_ok=True)

    def marginal(self,mask,particles):
        key=(mask,particles)
        if key in self.cache: return self.cache[key]
        check_time(self.deadline); path=self.directory/f'{mask}_{particles}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as data:
                result=data['log_marginal'].copy(),json.loads(str(data['diagnostic']))
            self.cache[key]=result; return result
        known=np.flatnonzero(MEMBERSHIP[mask]); missing=np.flatnonzero(~MEMBERSHIP[mask])
        if not len(missing):
            result=self.model.log_prob(self.source,self.target),dict(ess_minimum=None,maximum_weight=None)
        else:
            if len(known):
                gain=np.linalg.solve(self.cov[np.ix_(known,known)],self.cov[np.ix_(known,missing)]).T
                cond_mean=self.mean[missing]+(self.source[:,known]-self.mean[known])@gain.T
                cond_cov=self.cov[np.ix_(missing,missing)]-gain@self.cov[np.ix_(known,missing)]
            else:
                cond_mean=np.broadcast_to(self.mean[missing],(self.outer,len(missing)))
                cond_cov=self.cov[np.ix_(missing,missing)]
            jm=self.jac[:,missing]
            target_cov=np.diag(self.model.variance)+jm@cond_cov@jm.T
            target_chol=np.linalg.cholesky(target_cov)
            m0=self.intercept+cond_mean@jm.T
            if len(known): m0=m0+self.source[:,known]@self.jac[:,known].T
            residual=self.target-m0
            white=np.linalg.solve(target_chol,residual.T).T
            log_reference=-.5*(np.sum(white**2,axis=1)+9*np.log(2*np.pi))-np.log(np.diag(target_chol)).sum()
            gain=cond_cov@jm.T@np.linalg.inv(target_cov)
            post_mean=cond_mean+residual@gain.T
            post_cov=cond_cov-gain@jm@cond_cov
            post_chol=np.linalg.cholesky((post_cov+post_cov.T)*.5)
            values=[]; effective=[]; maximum=[]
            for start in range(0,self.outer,max(1,32768//particles)):
                check_time(self.deadline); stop=min(self.outer,start+max(1,32768//particles)); n=stop-start
                z=ndtri(self.inner[start:stop,:particles][:,:,missing])
                completed=np.repeat(self.source[start:stop,None,:],particles,axis=1)
                completed[:,:,missing]=post_mean[start:stop,None,:]+z@post_chol.T
                x=completed.reshape(-1,9); y=np.repeat(self.target[start:stop],particles,axis=0)
                linear_mean=self.intercept+x@self.jac.T
                ll0=-.5*np.sum((y-linear_mean)**2/self.model.variance+np.log(2*np.pi*self.model.variance),axis=1)
                weights=(self.model.log_prob(x,y)-ll0).reshape(n,particles)
                normalizer=logsumexp(weights,axis=1)
                normalized=weights-normalizer[:,None]
                values.extend(log_reference[start:stop]+normalizer-np.log(particles))
                effective.extend(np.exp(-logsumexp(2*normalized,axis=1)))
                maximum.extend(np.exp(normalized.max(1)))
            result=np.asarray(values),dict(ess_minimum=float(np.min(effective)),maximum_weight=float(np.max(maximum)))
        if not np.isfinite(result[0]).all(): raise ArithmeticError('Nonfinite natural TM marginal')
        np.savez_compressed(path,log_marginal=result[0],diagnostic=json.dumps(result[1]),meta=json.dumps(self.meta))
        self.cache[key]=result; return result

    def information(self,mask,particles):
        a,da=self.marginal(mask,particles); b,db=self.marginal(0,particles)
        return float(np.mean(a-b)),(da,db)


def fixed_target_wms(table):
    clades=[]; selected=[]; all_residual=[]
    def visit(a):
        if a.bit_count()==1: return
        candidates=[]
        for l,r in splits(a):
            raw=table[a]-table[l]-table[r]; all_residual.append(raw)
            candidates.append((raw,labels(l),labels(r),l,r))
        raw,_,_,l,r=min(candidates)
        selected.append(dict(sources=labels(a),wms_nats=raw))
        for child in (l,r):
            if child.bit_count()>1: clades.append(labels(child))
            visit(child)
        return labels(l),labels(r)
    root=visit(511)
    closure=table[511]-sum(table[1<<i] for i in range(9))-sum(r['wms_nats'] for r in selected)
    if abs(closure)>1e-10: raise ArithmeticError('Signed WMS telescoping check failed')
    return quality(root,clades),dict(nodes=selected,minimum_selected_search_wms_nats=min(all_residual),
        negative_search_count=sum(v<0 for v in all_residual),
        below_minus_numerical_tolerance_count=sum(v<-.01 for v in all_residual),
        tolerance_band_negative_count=sum(-.01<=v<0 for v in all_residual),
        closure_error_nats=closure,negative_values_are_valid=True,clipping=False)


def shap_profile(model,evaluation,background,deadline):
    game=np.empty((len(evaluation),512,9))
    for mask,bits in enumerate(MEMBERSHIP):
        check_time(deadline)
        completed=np.where(bits[None,None,:],evaluation[:,None,:],background[None,:,:])
        game[:,mask]=model.mean(completed.reshape(-1,9)).reshape(len(evaluation),len(background),9).mean(1)
    values=np.zeros((len(evaluation),9,9))
    for i in range(9):
        absent=np.flatnonzero(~MEMBERSHIP[:,i])
        weights=np.array([1/(9*math.comb(8,int(m).bit_count())) for m in absent])
        values[:,i]=np.einsum('m,nmo->no',weights,game[:,absent|(1<<i)]-game[:,absent])
    error=float(abs(values.sum(1)-(game[:,511]-game[:,0])).max())
    if error>1e-9: raise ArithmeticError(f'Exact coalition Shapley efficiency failed: {error}')
    return abs(values).mean(0),dict(efficiency_error_output_units=error,coalitions=512,
        evaluation_samples=len(evaluation),background_samples=len(background),outputs=9,players=9)


def profile_tree(profiles):
    distances=pdist(profiles,metric='cosine')
    if not np.isfinite(distances).all(): raise ArithmeticError('Undefined SHAP profile distance')
    z=linkage(distances,method='single'); groups={i:(i+1,) for i in range(9)}; clades=[]
    for index,row in enumerate(z):
        left,right=(groups[int(i)] for i in row[:2]); group=tuple(sorted(left+right)); groups[9+index]=group
        if len(group)<9: clades.append(group)
    return quality((left,right),clades)


def run(directory=DIRECTORY):
    directory=Path(directory); budget=json.loads((directory/'budget.json').read_text())
    deadline=time.monotonic()+budget['deadline_epoch']-time.time(); path=directory/'natural_baselines.json'
    result=json.loads(path.read_text()) if path.exists() else dict(version=VERSION,status='running',methods={})
    if result['version']!=VERSION: raise ValueError('Natural baseline cache version differs')
    result.update(process_id=os.getpid(),status='running',stage='natural_sampling'); save_json(path,result)
    model=RealConditionalTM.load(json.loads((ROOT/'results/spt_local_glv_tm/pilot.json').read_text())['records'][0]['model'])
    try:
        states,meta=natural_data(directory,deadline); mean,cov,prior_audit=source_tm(states)
        result.update(natural_sampling=meta,source_tm_audit=prior_audit)
        # The new SHAP adaptation is fixed before examining its F1.
        result['stage']='shap_profiles'; save_json(path,result)
        if 'shap_profiles' not in result['methods']:
            train=states[:256].reshape(-1,9); hold=states[256:512].reshape(-1,9); profiles=[]; checks=[]
            for repeat in range(4):
                rng=np.random.default_rng(np.random.SeedSequence([20261015,0,91000+repeat]))
                e=hold[rng.choice(len(hold),256,replace=False)]
                b=train[rng.choice(len(train),128,replace=False)]
                p,audit=shap_profile(model,e,b,deadline); profiles.append(p); checks.append(audit)
                print(json.dumps(dict(stage='shap_profiles',repeat=repeat)),flush=True)
            metrics=profile_tree(np.mean(profiles,axis=0)); reps=[profile_tree(p) for p in profiles]
            result['methods']['shap_profiles']=dict(status='screen_passed',metrics=metrics,integration_replicates=reps,
                profiles=np.mean(profiles,axis=0).tolist(),checks=checks,
                repeat_unit='natural evaluation/background subsamples, not independent fitted datasets',
                attribution='interventional group Shapley with jointly sampled natural complement; full nine outputs',
                tree_adaptation='cosine distance and single linkage on mean absolute SHAP output profiles')
            save_json(path,result)
        result['stage']='wms_natural'; save_json(path,result)
        if not prior_audit['passed']:
            result['methods']['wms_natural']=dict(status='natural_source_tm_diagnostics_failed',metrics=None)
        elif 'wms_natural' not in result['methods']:
            records=result.setdefault('wms_replicates',[])
            for repeat in range(4):
                row=next((r for r in records if r['scramble']==repeat),None)
                if row is None:
                    row=dict(scramble=repeat,information_nats={},diagnostics={}); records.append(row)
                estimator=NaturalInformation(model,mean,cov,outer=1024,seed=20261015+1000*repeat,
                    deadline=deadline,cache_dir=directory/'natural_marginals')
                for mask in range(1,512):
                    if str(mask) in row['information_nats']: continue
                    value,diags=estimator.information(mask,64)
                    row['information_nats'][str(mask)]=value; row['diagnostics'][str(mask)]=diags
                    if mask%64==0:
                        save_json(path,result); print(json.dumps(dict(stage='wms_full',scramble=repeat,subsets=mask)),flush=True)
                save_json(path,result)
            samples=np.array([[r['information_nats'][str(m)] for m in range(1,512)] for r in records])
            sem=float(samples.std(0,ddof=1).max()/2); mean_table={m:float(samples[:,m-1].mean()) for m in range(1,512)}
            diagnostics=[d for row in records for ds in row['diagnostics'].values() for d in ds if d['ess_minimum'] is not None]
            min_ess=min(d['ess_minimum'] for d in diagnostics); max_weight=max(d['maximum_weight'] for d in diagnostics)
            # Independent inner-resolution check at the same main outer points.
            check_masks=(1,3,7,24,31,96,384,480,511); changes=[]
            for repeat in range(4):
                estimator=NaturalInformation(model,mean,cov,outer=1024,seed=20261015+1000*repeat,
                    deadline=deadline,cache_dir=directory/'natural_marginals')
                for mask in check_masks:
                    fine,_=estimator.information(mask,256); changes.append(abs(fine-records[repeat]['information_nats'][str(mask)]))
            issues=[]
            if sem>=.01: issues.append('information_SEM>=.01')
            if max(changes)>=.01: issues.append('inner64_to256_change>=.01')
            if min_ess<10 or max_weight>=.25: issues.append('importance_weight_gate_failed')
            if issues:
                result['methods']['wms_natural']=dict(status='numerically_unresolved',metrics=None,failed_checks=issues)
            else:
                metrics,audit=fixed_target_wms(mean_table)
                reps=[fixed_target_wms({m:row['information_nats'][str(m)] for m in range(1,512)})[0] for row in records]
                result['methods']['wms_natural']=dict(status='screen_passed',metrics=metrics,integration_replicates=reps,audit=audit,
                    maximum_information_sem_nats=sem,maximum_inner_change_nats=max(changes),minimum_ess=min_ess,maximum_weight=max_weight,
                    target='fixed complete nine-coordinate future',prior='full covariance natural-source Gaussian TM approximation',
                    transition='same frozen nonlinear TM as SPT; linear reference only importance control')
        result['status']='complete'
    except TimeoutError as error: result.update(status='budget_exhausted',error=str(error))
    except (ArithmeticError,np.linalg.LinAlgError,ValueError) as error:
        result.update(status='numerical_failure',error=f'{type(error).__name__}: {error}')
    except KeyboardInterrupt: result.update(status='cancelled')
    result['elapsed_wall_seconds']=time.time()-budget['started_epoch']; save_json(path,result)
    print(json.dumps(dict(status=result['status'],methods={k:v['status'] for k,v in result['methods'].items()})),flush=True)
    return result


if __name__=='__main__': run()
