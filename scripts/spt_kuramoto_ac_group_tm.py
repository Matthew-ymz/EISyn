"""Isolated A/C group-target marginals of the frozen circular conditional TM.

Snapshot of the shared importance evaluator, preserving 1024-particle prior
integration and the refined local-Jacobian conditional covariance. No SMC
replacement or mutable B wrapper is imported.

Missing self-target coordinates use conditional importance sampling; other
missing sources use their registered uniform prior. No marginal is re-fitted.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc

from scripts.spt_kuramoto_finite_tm import (
    FullTargetIntegrator,TWO_PI,LOG_TWO_PI,DEFENSIVE,WIDE_FRACTION,WIDE_SCALE,
    posterior_proposal,proposal_sample,proposal_log_prob,check_time,
)
from scripts.spt_local_hyperedge_kuramoto import wrap


def indices(mask): return np.array([i for i in range(9) if mask>>i&1],dtype=int)


class GroupTargetIntegrator:
    def __init__(self,model,*,outer,particles,seed,deadline,max_particles=256):
        if particles>max_particles or any(v&(v-1) for v in (outer,particles,max_particles)):
            raise ValueError('Nested power-of-two sample counts required')
        self.model=model; self.outer=outer; self.particles=particles; self.deadline=deadline
        self.seed=seed
        self.full=FullTargetIntegrator(model,outer=outer,max_particles=256,seed=seed,deadline=deadline)
        self.source=self.full.source; self.target=self.full.target
        # U coordinates 0:9, branch 9, D innovations 10:19 are disjoint.
        points=qmc.Sobol(19,scramble=True,seed=seed+200000).random_base2(int(np.log2(outer*max_particles)))
        self.inner=points.reshape(outer,max_particles,19)
        self.cache={}; self.info_cache={}

    def marginal(self,source_mask,target_mask):
        key=(source_mask,target_mask)
        if key in self.cache: return self.cache[key]
        check_time(self.deadline)
        if target_mask==511:
            result=self.full.marginal(source_mask,min(self.particles,256))
            self.cache[key]=result; return result
        target_indices=indices(target_mask)
        if not len(target_indices): raise ValueError('Nonempty target required')
        direct_diag={'ess_minimum':None,'ess_median':None,'maximum_normalized_weight':None,
                     'max_remaining_step_rad':0.,'max_proposal_std_rad':0.}
        if source_mask==511:
            result=self.model.log_prob(self.source,self.target,target_indices),direct_diag
            self.cache[key]=result; return result
        d=indices(target_mask & (511^source_mask))
        u=indices(511 ^ (source_mask|target_mask))
        if len(d) and not len(u):
            center,chol,center_diag=posterior_proposal(self.model,self.source,self.target,d,target_indices)
        else: center_diag={'max_remaining_step_rad':0.,'max_proposal_std_rad':0.}
        values=[]; prefix_values=[]; effective=[]; largest=[]
        # Pure prior integrals can have thin likelihood ridges. Use the
        # registered 1024-particle net there; this affects no source support.
        p=self.inner.shape[1] if not len(d) else self.particles
        for start in range(0,self.outer,max(1,8192//p)):
            check_time(self.deadline)
            stop=min(self.outer,start+max(1,8192//p)); n=stop-start
            random=self.inner[start:stop,:p]
            completed=np.repeat(self.source[start:stop,None,:],p,axis=1)
            if len(u): completed[:,:,u]=random[:,:,u]*TWO_PI-np.pi
            targets=np.repeat(self.target[start:stop],p,axis=0)
            if len(d):
                if len(u):
                    # These centers depend on the current nuisance draw. The
                    # proposal is conditional on U, whose prior cancels below.
                    flat=completed.reshape(-1,9)
                    flat[:,d]=targets[:,d]
                    for _ in range(4):
                        mean=self.model.mean(flat)
                        flat[:,d]=wrap(flat[:,d]+wrap(targets[:,d]-mean[:,d]))
                    c=flat[:,d].copy()
                    mean,jac=self.model.mean(flat,jacobian=True)
                    derivative=jac[:,target_indices][:,:,d]
                    precision=np.einsum('nti,t,ntj->nij',derivative,
                        self.model.kappa[target_indices],derivative)
                    l=np.linalg.cholesky(1.2*np.linalg.inv(precision))
                    remaining=wrap(targets[:,d]-mean[:,d])
                    bad=np.max(abs(remaining)*np.sqrt(self.model.kappa[d]),axis=1)>.25
                    fallback_step=0.
                    if bad.any():
                        gc,gl,gd=posterior_proposal(self.model,flat[bad],targets[bad],
                                                   d,target_indices,iterations=4)
                        c[bad]=gc; l[bad]=gl
                        fallback_step=gd['max_remaining_step_rad']
                    diag={'max_proposal_std_rad':float(np.sqrt(np.sum(l*l,axis=2)).max()),
                          'max_picard_center_residual_rad':float(abs(remaining).max()),
                          'maximum_fallback_fraction':float(bad.mean()),
                          'maximum_fallback_newton_step_rad':float(fallback_step)}
                    center_diag.pop('max_remaining_step_rad',None)
                    center_diag['center_method']='four Picard steps; local Jacobian covariance; GN fallback above 0.25 noise SD residual'
                    draws=random[:,:,10+d].reshape(-1,1,len(d))
                    branch=random[:,:,9].reshape(-1,1)
                else:
                    c=center[start:stop]; l=chol[start:stop]; diag=center_diag
                    draws=random[:,:,10+d]; branch=random[:,:,9]
                proposed=proposal_sample(c,l,draws)
                wide=proposal_sample(c,WIDE_SCALE*l,draws)
                proposed=np.where(((branch>=DEFENSIVE)&(branch<DEFENSIVE+WIDE_FRACTION))[:,:,None],wide,proposed)
                proposed=np.where((branch<DEFENSIVE)[:,:,None],draws*TWO_PI-np.pi,proposed)
                q=proposal_log_prob(proposed,c,l)
                qw=proposal_log_prob(proposed,c,WIDE_SCALE*l)
                qmix=np.logaddexp(np.logaddexp(np.log(1-DEFENSIVE-WIDE_FRACTION)+q,
                    np.log(WIDE_FRACTION)+qw),np.log(DEFENSIVE)-len(d)*LOG_TWO_PI)
                completed[:,:,d]=proposed.reshape(n,p,len(d))
                correction=(-len(d)*LOG_TWO_PI-qmix).reshape(n,p)
                for k,v in diag.items(): center_diag[k]=max(center_diag.get(k,0.),v)
            else: correction=0.
            likelihood=self.model.log_prob(completed.reshape(-1,9),targets,target_indices).reshape(n,p)
            weights=likelihood+correction
            normalizer=logsumexp(weights,axis=1); normalized=weights-normalizer[:,None]
            values.append(normalizer-np.log(p))
            prefix=min(64,p)
            prefix_values.append(logsumexp(weights[:,:prefix],axis=1)-np.log(prefix))
            effective.extend(np.exp(-logsumexp(2*normalized,axis=1)))
            largest.extend(np.exp(np.max(normalized,axis=1)))
        result=np.concatenate(values),{**center_diag,'ess_minimum':float(np.min(effective)),
            'ess_median':float(np.median(effective)),'maximum_normalized_weight':float(np.max(largest)),
            'informed_missing_count':len(d),'uniform_nuisance_count':len(u),
            'particles':p,
            'prefix_particles':min(64,p),'prefix_log_marginal':np.concatenate(prefix_values).tolist()}
        self.cache[key]=result; return result

    def information(self,source_mask,target_mask):
        key=(source_mask,target_mask)
        if key not in self.info_cache:
            numerator,numer_diag=self.marginal(source_mask,target_mask)
            denominator,denom_diag=self.marginal(0,target_mask)
            pointwise=numerator-denominator
            self.info_cache[key]=(float(pointwise.mean()),pointwise,
                {'numerator':numer_diag,'denominator':denom_diag})
        return self.info_cache[key]
