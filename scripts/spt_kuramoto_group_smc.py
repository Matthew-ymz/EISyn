"""Annealed SMC evidence for a frozen circular likelihood and uniform prior.

Pilot information fixes the symmetric proposal scale, never its normalizer.
The likelihood evidence estimator, not its logarithm, is unbiased.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc

from scripts.spt_kuramoto_finite_tm import TWO_PI,check_time
from scripts.spt_local_hyperedge_kuramoto import wrap


def smc_evidence(model,source,target,missing,targets,*,particles,rng,fisher,
                 deadline,layers=16,moves=2):
    """Batched bootstrap SMC; every ESS is measured before resampling."""
    n=len(source); p=particles; m=len(missing)
    x=np.repeat(source[:,None,:],p,axis=1)
    x[:,:,missing]=rng.uniform(-np.pi,np.pi,(n,p,m))
    y=np.repeat(target,p,axis=0)
    def likelihood(z):
        return model.log_prob(z.reshape(-1,9),y,targets).reshape(n,p)
    ll=likelihood(x); evidence=np.zeros(n); ancestor=np.tile(np.arange(p),(n,1))
    ess=[]; largest=[]; surviving=[]; accepted=[0,0]; proposed=[0,0]
    beta=0.
    for layer in range(1,layers+1):
        check_time(deadline)
        next_beta=(layer/layers)**2
        w=(next_beta-beta)*ll; normalizer=logsumexp(w,axis=1)
        evidence+=normalizer-np.log(p)
        weights=np.exp(w-normalizer[:,None])
        ess.append(1/np.sum(weights*weights,axis=1)); largest.append(weights.max(1))
        if layer==layers: break
        cdf=np.cumsum(weights,axis=1); cdf[:,-1]=1.
        positions=(np.arange(p)[None,:]+rng.random((n,1)))/p
        take=np.array([np.searchsorted(cdf[i],positions[i]) for i in range(n)])
        x=np.take_along_axis(x,take[:,:,None],axis=1)
        ll=np.take_along_axis(ll,take,axis=1)
        ancestor=np.take_along_axis(ancestor,take,axis=1)
        surviving.append(np.array([len(np.unique(a)) for a in ancestor]))
        # Conditional on the independent pilot, scale is fixed for this outer
        # observation and temperature. Both proposal components are symmetric.
        scale=.7/np.sqrt(.3+next_beta*fisher)
        for _ in range(moves):
            candidate=x.copy(); jump=rng.random((n,p))<.1
            local=wrap(x[:,:,missing]+rng.normal(size=(n,p,m))*scale[:,None,:])
            uniform=rng.uniform(-np.pi,np.pi,(n,p,m))
            candidate[:,:,missing]=np.where(jump[:,:,None],uniform,local)
            candidate_ll=likelihood(candidate)
            accept=np.log(rng.random((n,p)))<next_beta*(candidate_ll-ll)
            x=np.where(accept[:,:,None],candidate,x); ll=np.where(accept,candidate_ll,ll)
            for k,which in enumerate((~jump,jump)):
                proposed[k]+=int(which.sum()); accepted[k]+=int((accept&which).sum())
        beta=next_beta
    return evidence,{'pre_resampling_ess_minimum':float(np.min(ess)),
        'pre_resampling_ess_median':float(np.median(ess)),
        'maximum_incremental_normalized_weight':float(np.max(largest)),
        'minimum_surviving_initial_ancestors':int(np.min(surviving)) if surviving else p,
        'local_acceptance':accepted[0]/max(proposed[0],1),
        'uniform_acceptance':accepted[1]/max(proposed[1],1),
        'layers':layers,'moves_per_layer':moves,'particles':p}


def diagonal_evidence(model,source,target,missing,targets,*,seed,deadline,
                      particles=1024,pilot_particles=1024,layers=16,moves=2):
    """Independent pilot chooses plain MC or SMC; no production selection."""
    n=len(source); values=[]; lower=[]; summaries=[]; count=0
    # Small fixed chunks bound memory and are part of the cache protocol.
    for start in range(0,n,32):
        check_time(deadline); stop=min(n,start+32); size=stop-start
        sx=source[start:stop]; sy=target[start:stop]
        pilot_rng=np.random.default_rng(np.random.SeedSequence([*seed,start,1]))
        prod_rng=np.random.default_rng(np.random.SeedSequence([*seed,start,2]))
        pilot=np.repeat(sx[:,None,:],pilot_particles,axis=1)
        pilot[:,:,missing]=pilot_rng.uniform(-np.pi,np.pi,(size,pilot_particles,len(missing)))
        py=np.repeat(sy,pilot_particles,axis=0)
        ll=model.log_prob(pilot.reshape(-1,9),py,targets).reshape(size,pilot_particles)
        w=np.exp(ll-logsumexp(ll,axis=1)[:,None])
        hard=(1/np.sum(w*w,axis=1)<64)|(w.max(1)>.05)
        logz=np.empty(size); low=np.empty(size)
        if (~hard).any():
            easy=sx[~hard]; ey=sy[~hard]; q=len(easy)
            x=np.repeat(easy[:,None,:],particles,axis=1)
            # Fresh scrambled uniform net, independent of the pilot. Generate
            # the full fixed chunk before choosing the easy rows.
            net=qmc.Sobol(len(missing),scramble=True,seed=prod_rng)
            uniform=net.random_base2(int(np.log2(32*particles))).reshape(32,particles,len(missing))
            x[:,:,missing]=uniform[:size][~hard]*TWO_PI-np.pi
            density=model.log_prob(x.reshape(-1,9),np.repeat(ey,particles,axis=0),targets).reshape(q,particles)
            logz[~hard]=logsumexp(density,axis=1)-np.log(particles)
            low[~hard]=logsumexp(density[:,:256],axis=1)-np.log(min(256,particles))
            nw=np.exp(density-logsumexp(density,axis=1)[:,None])
            summaries.append({'ess':float((1/(nw*nw).sum(1)).min()),'weight':float(nw.max())})
        if hard.any():
            best=pilot[np.arange(size),ll.argmax(1)][hard]
            _,jac=model.mean(best,jacobian=True)
            fisher=np.einsum('ntm,t,ntm->nm',jac[:,targets][:,:,missing],model.kappa[targets],
                             jac[:,targets][:,:,missing])
            rng=np.random.default_rng(np.random.SeedSequence([*seed,start,3,particles]))
            rng_low=np.random.default_rng(np.random.SeedSequence([*seed,start,3,256]))
            high,diag=smc_evidence(model,sx[hard],sy[hard],missing,targets,particles=particles,
                rng=rng,fisher=fisher,deadline=deadline,layers=layers,moves=moves)
            low_values,low_diag=smc_evidence(model,sx[hard],sy[hard],missing,targets,particles=256,
                rng=rng_low,fisher=fisher,deadline=deadline,layers=layers,moves=moves)
            logz[hard]=high; low[hard]=low_values; count+=int(hard.sum())
            summaries.append({'ess':diag['pre_resampling_ess_minimum'],
                'weight':diag['maximum_incremental_normalized_weight'],'smc':diag,'lower_smc':low_diag})
        values.append(logz); lower.append(low)
    return np.concatenate(values),{'ess_minimum':min(d['ess'] for d in summaries),
        'ess_median':None,'maximum_normalized_weight':max(d['weight'] for d in summaries),
        'integration':'independent pilot-selected QMC / annealed SMC v3',
        'smc_outer_count':count,'outer_count':n,'particles':particles,
        'smc_layers':layers,'smc_moves_per_layer':moves,
        'prefix_particles':256,'prefix_semantics':'SMC particle-number check, independent resampling; MC prefix',
        'prefix_log_marginal':np.concatenate(lower).tolist(),
        'chunk_diagnostics':summaries,'seed_components':list(seed),
        'max_remaining_step_rad':0.,'max_proposal_std_rad':0.}
