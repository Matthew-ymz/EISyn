"""Real-domain marginals of one frozen nonlinear conditional Gaussian TM.

No marginal refits, source/target wrapping, density truncation or truth access.
The QR/Gauss--Newton construction affects only the importance proposal.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import qr
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import qmc

from scripts.spt_glv_finite_tm import (
    TWO_PI, LOG_TWO_PI, proposal_sample, proposal_log_prob, check_time,
)

VERSION = 'real-group-v4-row-refinement'
MIXTURE = np.array([.80, .15, .05])


def columns(mask):
    return np.array([i for i in range(9) if int(mask) >> i & 1], dtype=int)


def informed_variables(model, fixed_mask, target_mask):
    """Choose proposal coordinates from the fitted, whitened reference Jacobian."""
    missing = columns(511 ^ fixed_mask)
    targets = columns(target_mask)
    if not len(missing):
        return missing, missing
    _, jac = model.mean_selected(np.zeros((1, 9)), targets, missing)
    whitened = jac[0] / np.sqrt(model.variance[targets])[:, None]
    _, r, pivot = qr(whitened, mode='economic', pivoting=True)
    # A change across one half-support must exceed one fitted residual SD.
    rank = int(np.count_nonzero(abs(np.diag(r)) >= 1 / np.pi))
    guided = missing[pivot[:rank]]
    nuisance = np.array([i for i in missing if i not in guided], dtype=int)
    return guided, nuisance


def conditional_proposal(model, completed, future, targets, guided, *, iterations=32):
    """Bounded, regularized Gauss--Newton; reference penalty is proposal-only."""
    x = completed.copy()
    x[:, guided] = 0.
    for k, variable in enumerate(guided):
        if variable in targets:
            x[:, variable] = np.clip(future[:, variable], -np.pi+1e-9, np.pi-1e-9)
    regularizer = np.eye(len(guided)) / np.pi**2
    active = np.ones(len(x), dtype=bool)
    def direction(value,hessian,gradient):
        blocked=((value<=-np.pi+2e-9)&(gradient<0))|((value>=np.pi-2e-9)&(gradient>0))
        h=hessian.copy(); g=gradient.copy()
        h[blocked,:]=0.
        h.transpose(0,2,1)[blocked,:]=0.
        for j in range(len(guided)):
            h[:,j,j]=np.where(blocked[:,j],1.,h[:,j,j])
        g[blocked]=0.
        return np.linalg.solve(h,g[...,None])[...,0]
    for _ in range(iterations):
        if not active.any():
            break
        ix = np.flatnonzero(active)
        mean, jac = model.mean_selected(x[ix], targets, guided)
        residual = future[ix][:, targets] - mean
        hessian = np.einsum('nti,t,ntj->nij', jac, model.precision[targets], jac) + regularizer
        gradient = np.einsum('nti,t,nt->ni', jac, model.precision[targets], residual) - x[ix][:, guided]/np.pi**2
        step = direction(x[ix][:,guided],hessian,gradient)
        scale = np.maximum(1., abs(step).max(axis=1))
        candidate = np.clip(x[ix][:, guided] + step/scale[:, None], -np.pi+1e-9, np.pi-1e-9)
        change=candidate-x[ix][:,guided]
        bad=(gradient*change).sum(1)<=0
        if bad.any():
            fallback=gradient[bad]/np.diagonal(hessian[bad],axis1=1,axis2=2)
            fallback/=np.maximum(1.,abs(fallback).max(1))[:,None]
            candidate[bad]=np.clip(x[ix[bad]][:,guided]+fallback,-np.pi+1e-9,np.pi-1e-9)
        objective=.5*np.sum(residual**2*model.precision[targets],axis=1)+.5*np.sum(x[ix][:,guided]**2,axis=1)/np.pi**2
        pending=np.ones(len(ix),dtype=bool)
        accepted=x[ix][:,guided].copy()
        for _ in range(16):
            if not pending.any(): break
            rows=np.flatnonzero(pending)
            trial=x[ix[rows]].copy(); trial[:,guided]=candidate[rows]
            trial_mean=model.mean_selected(trial,targets)
            trial_objective=.5*np.sum((future[ix[rows]][:,targets]-trial_mean)**2*model.precision[targets],axis=1)+.5*np.sum(candidate[rows]**2,axis=1)/np.pi**2
            good=trial_objective<=objective[rows]+1e-12
            accepted[rows[good]]=candidate[rows[good]]
            pending[rows[good]]=False
            candidate[rows[~good]]=(candidate[rows[~good]]+x[ix[rows[~good]]][:,guided])*.5
        difference = abs(accepted - x[ix][:, guided]).max(axis=1)
        x[np.ix_(ix, guided)] = accepted
        active[ix[difference < 1e-5]] = False
    mean, jac = model.mean_selected(x, targets, guided)
    hessian = np.einsum('nti,t,ntj->nij', jac, model.precision[targets], jac) + regularizer
    gradient = np.einsum('nti,t,nt->ni', jac, model.precision[targets], future[:, targets]-mean) - x[:, guided]/np.pi**2
    remaining = direction(x[:,guided],hessian,gradient)
    projected = np.clip(x[:, guided]+remaining, -np.pi+1e-9, np.pi-1e-9)-x[:, guided]
    # Polish only difficult bounded optima. This affects proposal efficiency;
    # the likelihood and explicit importance correction remain unchanged.
    for row in np.flatnonzero(abs(projected).max(1)>=1e-3):
        point=x[row:row+1].copy()
        def objective(v):
            point[0,guided]=v
            mu,j=model.mean_selected(point,targets,guided)
            residual=future[row,targets]-mu[0]
            loss=.5*np.sum(residual**2*model.precision[targets])+.5*np.sum(v*v)/np.pi**2
            gradient=-(j[0].T@(model.precision[targets]*residual))+v/np.pi**2
            return loss,gradient
        result=minimize(objective,x[row,guided],jac=True,method='L-BFGS-B',
            bounds=[(-np.pi+1e-9,np.pi-1e-9)]*len(guided),
            options=dict(maxiter=300,ftol=1e-14,gtol=1e-9,maxls=30))
        x[row,guided]=result.x
    if np.any(abs(projected).max(1)>=1e-3):
        mean,jac=model.mean_selected(x,targets,guided)
        hessian=np.einsum('nti,t,ntj->nij',jac,model.precision[targets],jac)+regularizer
        gradient=np.einsum('nti,t,nt->ni',jac,model.precision[targets],future[:,targets]-mean)-x[:,guided]/np.pi**2
        remaining=direction(x[:,guided],hessian,gradient)
        projected=np.clip(x[:,guided]+remaining,-np.pi+1e-9,np.pi-1e-9)-x[:,guided]
    largest = float(abs(projected).max())
    covariance = 1.2*np.linalg.inv(hessian)
    return x[:, guided], np.linalg.cholesky(covariance), largest


class RealGroupIntegrator:
    def __init__(self, model, *, outer, max_particles=1024, seed, deadline, cache_dir=None,proposal='qr'):
        if outer & (outer-1) or max_particles & (max_particles-1):
            raise ValueError('Sobol sizes must be powers of two')
        self.model = model
        self.outer, self.max_particles, self.seed = outer, max_particles, seed
        self.deadline = deadline
        if proposal not in ('qr','full'): raise ValueError('Unknown real proposal')
        self.proposal=proposal
        points = qmc.Sobol(18, scramble=True, seed=seed).random_base2(int(np.log2(outer)))
        self.source = (points[:, :9]-.5)*TWO_PI
        self.target = model.sample(self.source, points[:, 9:])
        self.inner = qmc.Sobol(19, scramble=True, seed=seed+200000).random_base2(
            int(np.log2(outer*max_particles))).reshape(outer, max_particles, 19)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache = {}
        self.model_sha = hashlib.sha256(json.dumps(model.payload(), sort_keys=True).encode()).hexdigest()
        self.query_count = 0

    def marginal(self, fixed_mask, target_mask, particles=256):
        check_time(self.deadline)
        if not 0 <= fixed_mask <= 511 or not 1 <= target_mask <= 511:
            raise ValueError('Invalid source/target masks')
        guided, nuisance = informed_variables(self.model, fixed_mask, target_mask)
        if self.proposal=='full':
            guided=columns(511^fixed_mask)
            nuisance=np.empty(0,dtype=int)
        actual_particles = 0 if fixed_mask == 511 else (particles if len(guided) else self.max_particles)
        key = (fixed_mask, target_mask, actual_particles)
        if key in self.cache:
            return self.cache[key]
        meta = dict(version=VERSION,proposal=self.proposal, model_sha=self.model_sha, outer=self.outer,
                    max_particles=self.max_particles, seed=self.seed, source=fixed_mask,
                    target=target_mask, particles=actual_particles)
        path = None
        if self.cache_dir:
            token = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:24]
            path = self.cache_dir/f'{token}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    if json.loads(str(data['meta'])) != meta:
                        raise ValueError('Marginal cache provenance mismatch')
                    value = data['log_marginal'].copy()
                    diagnostic = json.loads(str(data['diagnostic']))
                self.cache[key] = value, diagnostic
                return self.cache[key]
        targets = columns(target_mask)
        if fixed_mask == 511:
            values = self.model.log_prob(self.source, self.target, targets)
            diagnostic = dict(ess_minimum=None, ess_median=None, maximum_normalized_weight=None,
                              max_projected_step=0., inner_prefix_change_nats=0.)
        else:
            p = actual_particles
            estimates, prefixes, effective, maximum = [], [], [], []
            worst_step = 0.
            for start in range(0, self.outer, max(1, 16384//p)):
                check_time(self.deadline)
                stop = min(self.outer, start+max(1, 16384//p))
                n = stop-start
                uniforms = self.inner[start:stop, :p]
                completed = np.repeat(self.source[start:stop, None, :], p, axis=1)
                # EVERY unobserved coordinate is overwritten, never left at outer truth.
                completed[:, :, nuisance] = uniforms[:, :, nuisance]*TWO_PI-np.pi
                future = np.repeat(self.target[start:stop], p, axis=0)
                if len(guided):
                    if not len(nuisance):
                        center, chol, step = conditional_proposal(self.model, self.source[start:stop],
                            self.target[start:stop], targets, guided)
                        center = np.repeat(center, p, axis=0)
                        chol = np.repeat(chol, p, axis=0)
                    else:
                        center, chol, step = conditional_proposal(self.model, completed.reshape(-1, 9),
                            future, targets, guided)
                    worst_step = max(worst_step, step)
                    choice = uniforms[:, :, 9].ravel()
                    innovation = uniforms[:, :, 10+guided].reshape(-1, 1, len(guided))
                    wide = (choice >= .8) & (choice < .95)
                    scales = np.where(wide, 2., 1.)
                    sample = proposal_sample(center, chol*scales[:, None, None], innovation)
                    prior = choice >= .95
                    sample[prior, 0] = innovation[prior, 0]*TWO_PI-np.pi
                    local = proposal_log_prob(sample, center, chol)[:, 0]
                    broad = proposal_log_prob(sample, center, chol*2.)[:, 0]
                    mixture = logsumexp(np.stack([np.log(.8)+local, np.log(.15)+broad,
                        np.full(len(local), np.log(.05)-len(guided)*LOG_TWO_PI)]), axis=0)
                    completed[:, :, guided] = sample[:, 0].reshape(n, p, len(guided))
                    correction = -len(guided)*LOG_TWO_PI-mixture.reshape(n, p)
                else:
                    correction = 0.
                likelihood = self.model.log_prob(completed.reshape(-1, 9), future, targets).reshape(n, p)
                weights = likelihood+correction
                normalizer = logsumexp(weights, axis=1)
                normalized = weights-normalizer[:, None]
                estimates.append(normalizer-np.log(p))
                prefix = min(64, p)
                prefixes.append(logsumexp(weights[:, :prefix], axis=1)-np.log(prefix))
                effective.extend(np.exp(-logsumexp(2*normalized, axis=1)))
                maximum.extend(np.exp(np.max(normalized, axis=1)))
            values = np.concatenate(estimates)
            initial_ess=float(np.min(effective))
            initial_weight=float(np.max(maximum))
            refined=[]
            if self.proposal=='full':
                bad=np.flatnonzero((np.asarray(effective)<10)|(np.asarray(maximum)>=.25))
                for row in bad:
                    check_time(self.deadline)
                    value,ess,weight,used=self.refine_row(fixed_mask,target_mask,int(row),guided,p)
                    values[row]=value; effective[row]=ess; maximum[row]=weight
                    refined.append(dict(row=int(row),particles=used,ess=ess,maximum_weight=weight))
            diagnostic = dict(ess_minimum=float(np.min(effective)), ess_median=float(np.median(effective)),
                maximum_normalized_weight=float(np.max(maximum)), max_projected_step=worst_step,
                inner_prefix_change_nats=float(np.mean(values-np.concatenate(prefixes))),
                initial_ess_minimum=initial_ess,initial_maximum_weight=initial_weight,refined_rows=refined)
        if not np.isfinite(values).all():
            raise ArithmeticError('Nonfinite group marginal')
        diagnostic.update(guided=guided.tolist(), nuisance=nuisance.tolist(), **meta)
        self.cache[key] = values, diagnostic
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, meta=json.dumps(meta), diagnostic=json.dumps(diagnostic), log_marginal=values)
            temporary.replace(path)
        self.query_count += 1
        return self.cache[key]

    def refine_row(self,fixed_mask,target_mask,row,guided,initial_particles):
        """Spend precision only on concentrated rows; preserve full support/density.

        Fresh per-outer-row Sobol stream, nested through 65536 points. This is
        numerical precision adaptation, never a source of independent data.
        """
        targets=columns(target_mask)
        source=self.source[row:row+1]
        future=self.target[row:row+1]
        center,chol,_=conditional_proposal(self.model,source,future,targets,guided)
        points=qmc.Sobol(19,scramble=True,seed=self.seed+700000+row).random_base2(16)
        for p in (1024,4096,16384,65536):
            if p<initial_particles: continue
            check_time(self.deadline)
            choice=points[:p,9]
            innovation=np.asarray(points[:p][:,10+guided])[None,:,:]
            wide=(choice>=.8)&(choice<.95)
            local=proposal_sample(center,chol,innovation)
            broad=proposal_sample(center,chol*2.,innovation)
            sample=np.where(wide[None,:,None],broad,local)
            sample[0,choice>=.95]=innovation[0,choice>=.95]*TWO_PI-np.pi
            loglocal=proposal_log_prob(sample,center,chol)[0]
            logbroad=proposal_log_prob(sample,center,chol*2.)[0]
            mixture=logsumexp(np.stack([np.log(.8)+loglocal,np.log(.15)+logbroad,
                np.full(p,np.log(.05)-len(guided)*LOG_TWO_PI)]),axis=0)
            completed=np.repeat(source,p,axis=0); completed[:,guided]=sample[0]
            likelihood=self.model.log_prob(completed,np.repeat(future,p,axis=0),targets)
            weights=likelihood-len(guided)*LOG_TWO_PI-mixture
            normalizer=logsumexp(weights); normalized=weights-normalizer
            ess=float(np.exp(-logsumexp(2*normalized))); weight=float(np.exp(normalized.max()))
            value=float(normalizer-np.log(p))
            if ess>=10 and weight<.25: break
        return value,ess,weight,p

    def information(self, source_mask, target_mask, particles=256):
        numerator, a = self.marginal(source_mask, target_mask, particles)
        denominator, b = self.marginal(0, target_mask, particles)
        return float(np.mean(numerator-denominator)), (a, b)

    def future_information(self, i, j, particles=256):
        joint, a = self.marginal(0, (1 << i) | (1 << j), particles)
        left, b = self.marginal(0, 1 << i, particles)
        right, c = self.marginal(0, 1 << j, particles)
        return float(np.mean(joint-left-right)), (a, b, c)
