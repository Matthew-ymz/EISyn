"""Continuous SURD from the SAME learned gLV joint, before target averaging.

i_A(y)=E_{X|y}[log q(y|X_A)-log q(y)], not a decomposition of mean EI.
Posterior importance sampling and conditional marginal integration use the
frozen nonlinear Gaussian TM. No Gaussian joint refit or known-model scores.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc

from scripts.spt_glv_finite_tm import (
    RealConditionalTM,RealFullTargetIntegrator,posterior_proposal,
    proposal_sample,proposal_log_prob,LOG_TWO_PI,check_time,save_json,
)
from scripts.unicm_observational_information import assign_surd_specific
from scripts.compare_spt_surd_three_cases import maximum_weight_tree

ROOT=Path(__file__).resolve().parents[1]
VERSION='glv-surd-specific-nested-v1'
TOLERANCE=.01
PILOT_MASKS=(1,2,4,7,8,16,24,31,32,64,96,128,256,384,480,511)


def diagnostic_issues(d):
    issues=[]
    if d.get('ess_minimum') is not None and d['ess_minimum']<10: issues.append('ESS<10')
    if d.get('maximum_normalized_weight',0)>=.25: issues.append('weight>=.25')
    if d.get('max_projected_remaining_step_coordinate',0)>=1e-3: issues.append('proposal_step>=1e-3')
    return issues


class SpecificInformation:
    def __init__(self,model,*,outer,posterior_particles,inner_particles,seed,deadline,cache_dir):
        check_time(deadline)
        self.model=model; self.outer=outer; self.post=posterior_particles; self.inner_particles=inner_particles
        self.deadline=deadline; self.seed=seed
        if posterior_particles>128 or inner_particles>256: raise ValueError('Unregistered SURD grid')
        self.meta=dict(version=VERSION,outer=outer,posterior_particles=posterior_particles,
            inner_particles=inner_particles,seed=seed,model_sha=hashlib.sha256(
                json.dumps(model.payload(),sort_keys=True).encode()).hexdigest(),
            posterior_proposal='90% bounded local Gaussian +10% std-times2; full cube support',
            target='complete nine-dimensional future',units='nats')
        token=hashlib.sha256(json.dumps(self.meta,sort_keys=True).encode()).hexdigest()[:24]
        self.directory=Path(cache_dir)/token; self.directory.mkdir(parents=True,exist_ok=True)
        self.base=RealFullTargetIntegrator(model,outer=outer,max_particles=256,seed=seed,deadline=deadline)
        self.denominator,den_diag=self.base.marginal(0,256)
        center,chol,proposal_diag=posterior_proposal(model,self.base.source,self.base.target,
            np.arange(9),np.arange(9),iterations=16)
        points=qmc.Sobol(10,scramble=True,seed=seed+300000).random_base2(
            int(np.log2(outer*128))).reshape(outer,128,10)[:,:posterior_particles]
        wide=points[:,:,0]<.1
        narrow_sample=proposal_sample(center,chol,points[:,:,1:])
        broad_sample=proposal_sample(center,chol*2.,points[:,:,1:])
        self.posterior_source=np.where(wide[:,:,None],broad_sample,narrow_sample)
        logproposal=np.logaddexp(np.log(.9)+proposal_log_prob(self.posterior_source,center,chol),
            np.log(.1)+proposal_log_prob(self.posterior_source,center,chol*2.))
        source=self.posterior_source.reshape(-1,9)
        target=np.repeat(self.base.target,posterior_particles,axis=0)
        likelihood=model.log_prob(source,target).reshape(outer,posterior_particles)
        logweight=likelihood-9*LOG_TWO_PI-logproposal
        normalized=logweight-logsumexp(logweight,axis=1)[:,None]
        self.weights=np.exp(normalized)
        self.posterior_diagnostic=dict(**proposal_diag,
            ess_minimum=float(np.exp(-logsumexp(2*normalized,axis=1)).min()),
            ess_median=float(np.median(np.exp(-logsumexp(2*normalized,axis=1)))),
            maximum_normalized_weight=float(self.weights.max()))
        self.common_diagnostics=dict(posterior=self.posterior_diagnostic,denominator=den_diag)
        self.conditional=RealFullTargetIntegrator(model,outer=outer*posterior_particles,max_particles=256,
            seed=seed+400000,deadline=deadline,source=source,target=target)
        self.common_issues=sorted(set(diagnostic_issues(den_diag)+diagnostic_issues(self.posterior_diagnostic)))

    def specific(self,mask):
        check_time(self.deadline)
        path=self.directory/f'specific_{mask:03d}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as data:
                if json.loads(str(data['meta']))!=self.meta: raise ValueError('SURD cache provenance differs')
                return data['specific'].copy(),json.loads(str(data['diagnostic']))
        value,diagnostic=self.conditional.marginal(mask,self.inner_particles)
        specific=np.sum(self.weights*(value.reshape(self.outer,self.post)-self.denominator[:,None]),axis=1)
        diagnostic={**diagnostic,'issues':sorted(set(self.common_issues+diagnostic_issues(diagnostic))),
                    'minimum_specific_nats':float(specific.min())}
        if specific.min() < -TOLERANCE:
            diagnostic['issues'].append('specific_below_negative_tolerance')
        temporary=path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,meta=json.dumps(self.meta),specific=specific,diagnostic=json.dumps(diagnostic))
        temporary.replace(path)
        # Avoid retaining 511 copies of an outer*posterior array in memory.
        self.conditional.cache.clear(); self.conditional.proposals.clear()
        return specific,diagnostic


def load_model():
    return RealConditionalTM.load(json.loads((ROOT/'results/spt_local_glv_tm/pilot.json').read_text())['records'][0]['model'])


def audit_and_allocate(specific):
    if specific.ndim!=2 or specific.shape[1]!=512 or not np.isfinite(specific).all():
        raise ValueError('SURD requires all512 target-specific columns')
    minimum=float(specific.min()); affected=int(np.count_nonzero(specific < -TOLERANCE))
    if affected:
        raise ArithmeticError(f'Specific KL minimum={minimum} nats; threshold={-TOLERANCE}; affected_count={affected}')
    # No projection of estimated KL or PEID Syn. Significant violations fail.
    # Valid complete-source KL dominates the order-ranking endpoint.
    red=[]; syn=[]; closure=[]
    for row in specific:
        r,s=assign_surd_specific(row,9); red.append(r); syn.append(s)
        closure.append(abs(float(r.sum()+s.sum()-row[-1])))
    syn=np.asarray(syn); red=np.asarray(red)
    violations=int(np.count_nonzero(syn < -TOLERANCE))
    if violations or max(closure)>1e-8:
        raise ArithmeticError(f'SURD allocation invalid: atom_min={syn.min()}, threshold={-TOLERANCE}, count={violations}, closure={max(closure)}')
    return red.mean(0),syn.mean(0),dict(minimum_specific_nats=minimum,
        specific_tolerance_negative_count=int(np.count_nonzero((specific<0)&(specific>=-TOLERANCE))),
        significant_specific_violation_count=affected,minimum_atom_nats=float(syn.min()),
        atom_tolerance_negative_count=int(np.count_nonzero((syn<0)&(syn>=-TOLERANCE))),
        significant_atom_violation_count=violations,maximum_statewise_closure_error_nats=max(closure),
        tolerance_nats=TOLERANCE,clipping=False)


def pilot(directory,deadline):
    directory=Path(directory); path=directory/'surd_pilot.json'
    payload=json.loads(path.read_text()) if path.exists() else dict(version=VERSION,status='running',records=[])
    if payload['version']!=VERSION: raise ValueError('SURD pilot version mismatch')
    if payload['status'] in ('precision_passed','precision_failed'): return payload
    model=load_model()
    grids=[(32,32,64),(32,64,64),(64,64,64),(64,64,256)]
    try:
        for n,p,k in grids:
            for scramble in range(4):
                if any((r['outer'],r['posterior'],r['inner'],r['scramble'])==(n,p,k,scramble) for r in payload['records']): continue
                estimate=SpecificInformation(model,outer=n,posterior_particles=p,inner_particles=k,
                    seed=20261015+1000*scramble,deadline=deadline,cache_dir=directory/'surd_cache')
                row=dict(outer=n,posterior=p,inner=k,scramble=scramble,mi_nats={},diagnostics={},
                    common_diagnostics=estimate.common_diagnostics)
                for mask in PILOT_MASKS:
                    value,diagnostic=estimate.specific(mask)
                    row['mi_nats'][str(mask)]=float(value.mean()); row['diagnostics'][str(mask)]=diagnostic
                payload['records'].append(row); save_json(path,payload)
                print(json.dumps(dict(stage='surd_pilot',outer=n,posterior=p,inner=k,scramble=scramble)),flush=True)
        means=[]; sems=[]
        for grid in grids:
            rows=[r for r in payload['records'] if (r['outer'],r['posterior'],r['inner'])==grid]
            values=np.asarray([[r['mi_nats'][str(m)] for m in PILOT_MASKS] for r in rows])
            means.append(values.mean(0)); sems.append(float(values.std(0,ddof=1).max()/2))
        changes=[float(abs(a-b).max()) for a,b in zip(means,means[1:])]
        finest=[r for r in payload['records'] if (r['outer'],r['posterior'],r['inner'])==grids[-1]]
        issues=sorted(set(s for r in finest for d in r['diagnostics'].values() for s in d['issues']))
        if any(x>=.01 for x in changes): issues.append('grid_mean_change>=.01')
        if sems[-1]>=.01: issues.append('maximum_information_SEM>=.01')
        payload.update(status='precision_failed' if issues else 'precision_passed',failed_checks=issues,
            grid_maximum_changes_nats=changes,grid_maximum_sem_nats=sems,
            full_grid=dict(outer=128,posterior=64,inner=64) if changes[-1]<.01 else dict(outer=128,posterior=64,inner=256))
    except TimeoutError as error:
        payload.update(status='budget_exhausted',error=str(error))
    except (ArithmeticError,np.linalg.LinAlgError,ValueError) as error:
        payload.update(status='numerical_failure',error=f'{type(error).__name__}: {error}')
    save_json(path,payload); return payload


def full(directory,deadline,pilot_result):
    directory=Path(directory); path=directory/'surd_full.json'
    if pilot_result['status']!='precision_passed':
        return dict(status='not_evaluated_surd_precision_failed',pilot_status=pilot_result['status'],
                    failed_checks=pilot_result.get('failed_checks',[]))
    payload=json.loads(path.read_text()) if path.exists() else dict(version=VERSION,status='running',replicates=[])
    model=load_model(); grid=pilot_result['full_grid']; payload['grid']=grid
    try:
        for scramble in range(4):
            row=next((r for r in payload['replicates'] if r['scramble']==scramble),None)
            if row and row.get('complete'): continue
            if row is None:
                row=dict(scramble=scramble,subsets=0,diagnostics={}); payload['replicates'].append(row)
            estimate=SpecificInformation(model,outer=grid['outer'],posterior_particles=grid['posterior'],
                inner_particles=grid['inner'],seed=20261015+1000*scramble,deadline=deadline,cache_dir=directory/'surd_cache')
            specific=np.zeros((grid['outer'],512))
            for mask in range(1,512):
                value,diagnostic=estimate.specific(mask); specific[:,mask]=value
                row['subsets']=mask; row['diagnostics'][str(mask)]=diagnostic
                if mask%32==0:
                    save_json(path,payload); print(json.dumps(dict(stage='surd_full',scramble=scramble,subsets=mask)),flush=True)
            red,syn,audit=audit_and_allocate(specific)
            row.update(complete=True,redundancy_unique_nats=red.tolist(),synergy_nats=syn.tolist(),
                mi_nats=specific.mean(0).tolist(),audit=audit)
            save_json(path,payload)
        issues=sorted(set(s for r in payload['replicates'] for d in r['diagnostics'].values() for s in d['issues']))
        mi=np.asarray([r['mi_nats'] for r in payload['replicates']]); sem=float(mi.std(0,ddof=1).max()/2)
        if sem>=.01: issues.append('maximum_information_SEM>=.01')
        payload.update(status='numerically_unresolved' if issues else 'screen_passed',failed_checks=issues,maximum_information_sem_nats=sem)
        if not issues:
            syn=np.asarray([r['synergy_nats'] for r in payload['replicates']])
            payload.update(tree=maximum_weight_tree(syn.mean(0)),
                replicate_trees=[maximum_weight_tree(v) for v in syn],mean_synergy_nats=syn.mean(0).tolist(),
                tree_adaptation='global maximum sum of SURD atom weights over binary laminar tree; root and singleton constants do not affect selection',
                reference_thread='01a13e55-8332-7fd3-b70e-bd811d10f02f')
    except TimeoutError as error:
        payload.update(status='budget_exhausted',error=str(error))
    except (ArithmeticError,np.linalg.LinAlgError,ValueError) as error:
        payload.update(status='numerical_failure',error=f'{type(error).__name__}: {error}')
    save_json(path,payload); return payload


def refine_pilot(directory,deadline,initial):
    """Only numerical resolution increases, before any SURD tree/F1 is built."""
    if initial['status']=='precision_passed': return initial
    if initial['status']!='precision_failed': return initial
    directory=Path(directory); path=directory/'surd_pilot_refined.json'
    payload=json.loads(path.read_text()) if path.exists() else dict(version=VERSION,status='running',
        initial_status=initial['status'],initial_failed_checks=initial.get('failed_checks',[]),records=[])
    if payload['status'] in ('precision_passed','precision_failed'): return payload
    model=load_model(); grids=[(64,128,64),(128,128,64)]
    try:
        for n,p,k in grids:
            for scramble in range(4):
                if any((r['outer'],r['posterior'],r['inner'],r['scramble'])==(n,p,k,scramble) for r in payload['records']): continue
                estimate=SpecificInformation(model,outer=n,posterior_particles=p,inner_particles=k,
                    seed=20261015+1000*scramble,deadline=deadline,cache_dir=directory/'surd_cache')
                row=dict(outer=n,posterior=p,inner=k,scramble=scramble,mi_nats={},diagnostics={},common_diagnostics=estimate.common_diagnostics)
                for mask in PILOT_MASKS:
                    value,d=estimate.specific(mask); row['mi_nats'][str(mask)]=float(value.mean()); row['diagnostics'][str(mask)]=d
                payload['records'].append(row); save_json(path,payload)
                print(json.dumps(dict(stage='surd_refined_pilot',outer=n,posterior=p,inner=k,scramble=scramble)),flush=True)
        def values(records,grid):
            return np.asarray([[r['mi_nats'][str(m)] for m in PILOT_MASKS] for r in records
                if (r['outer'],r['posterior'],r['inner'])==grid])
        coarse=values(initial['records'],(64,64,64)); middle=values(payload['records'],grids[0]); fine=values(payload['records'],grids[1])
        changes=[float(abs(coarse.mean(0)-middle.mean(0)).max()),float(abs(middle.mean(0)-fine.mean(0)).max())]
        sem=float(fine.std(0,ddof=1).max()/2)
        issues=sorted(set(s for r in payload['records'] if r['outer']==128 for d in r['diagnostics'].values() for s in d['issues']))
        if any(c>=.01 for c in changes): issues.append('refined_grid_mean_change>=.01')
        if sem>=.01: issues.append('refined_information_SEM>=.01')
        if initial['grid_maximum_changes_nats'][-1]>=.01: issues.append('inner64_to256_change>=.01')
        payload.update(status='precision_failed' if issues else 'precision_passed',failed_checks=issues,
            grid_maximum_changes_nats=changes,maximum_information_sem_nats=sem,
            full_grid=dict(outer=128,posterior=128,inner=64))
    except TimeoutError as error:
        payload.update(status='budget_exhausted',error=str(error))
    except (ArithmeticError,np.linalg.LinAlgError,ValueError) as error:
        payload.update(status='numerical_failure',error=f'{type(error).__name__}: {error}')
    save_json(path,payload); return payload


def refine_inner_pilot(directory,deadline,initial,refined):
    """Resolve conditional-weight failures with the already registered inner256.

    Keep earlier failed pilots intact. Recheck posterior, outer and inner mean
    changes at the same grids, as well as every original precision gate.
    """
    if refined['status']!='precision_failed': return refined
    if not set(refined.get('failed_checks',[]))<=set(('ESS<10','weight>=.25')): return refined
    directory=Path(directory); path=directory/'surd_pilot_inner256.json'
    payload=json.loads(path.read_text()) if path.exists() else dict(version=VERSION,status='running',
        reason='inner64 conditional-weight failures; no change to model, source or thresholds',records=[])
    if payload['status'] in ('precision_passed','precision_failed'): return payload
    model=load_model(); grids=[(64,128,256),(128,128,256)]
    try:
        for n,p,k in grids:
            for scramble in range(4):
                if any((r['outer'],r['posterior'],r['inner'],r['scramble'])==(n,p,k,scramble) for r in payload['records']): continue
                estimate=SpecificInformation(model,outer=n,posterior_particles=p,inner_particles=k,
                    seed=20261015+1000*scramble,deadline=deadline,cache_dir=directory/'surd_cache')
                row=dict(outer=n,posterior=p,inner=k,scramble=scramble,mi_nats={},diagnostics={},common_diagnostics=estimate.common_diagnostics)
                for mask in PILOT_MASKS:
                    value,d=estimate.specific(mask); row['mi_nats'][str(mask)]=float(value.mean()); row['diagnostics'][str(mask)]=d
                payload['records'].append(row); save_json(path,payload)
                print(json.dumps(dict(stage='surd_inner256_pilot',outer=n,posterior=p,inner=k,scramble=scramble)),flush=True)
        def values(records,grid):
            rows=sorted((r for r in records if (r['outer'],r['posterior'],r['inner'])==grid),key=lambda r:r['scramble'])
            if [r['scramble'] for r in rows]!=[0,1,2,3]: raise ValueError('Incomplete SURD reference grid')
            return np.asarray([[r['mi_nats'][str(m)] for m in PILOT_MASKS] for r in rows])
        coarse=values(initial['records'],(64,64,256)); middle=values(payload['records'],grids[0]); fine=values(payload['records'],grids[1])
        old_inner=values(refined['records'],(128,128,64))
        changes=dict(posterior64_to128=float(abs(coarse.mean(0)-middle.mean(0)).max()),
            outer64_to128=float(abs(middle.mean(0)-fine.mean(0)).max()),
            inner64_to256=float(abs(old_inner.mean(0)-fine.mean(0)).max()))
        sem=float(fine.std(0,ddof=1).max()/2)
        issues=sorted(set(s for r in payload['records'] for d in r['diagnostics'].values() for s in d['issues']))
        if any(c>=.01 for c in changes.values()): issues.append('inner256_grid_mean_change>=.01')
        if sem>=.01: issues.append('inner256_information_SEM>=.01')
        payload.update(status='precision_failed' if issues else 'precision_passed',failed_checks=issues,
            grid_maximum_changes_nats=changes,maximum_information_sem_nats=sem,
            full_grid=dict(outer=128,posterior=128,inner=256))
    except TimeoutError as error:
        payload.update(status='budget_exhausted',error=str(error))
    except (ArithmeticError,np.linalg.LinAlgError,ValueError) as error:
        payload.update(status='numerical_failure',error=f'{type(error).__name__}: {error}')
    save_json(path,payload); return payload
