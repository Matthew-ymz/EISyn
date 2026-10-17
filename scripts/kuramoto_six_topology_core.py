"""Six-node higher-order Kuramoto, learned full conditional TM and exact-node SPT.

All information values are nats. No source components, truth-restricted cuts,
truth-specific dictionary, or clipping of information estimates is used.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import expit, logsumexp, ndtr, ndtri
from scipy.stats import chi2, qmc
from scripts.spt import SPTConfig, build_spt_from_ei_table, flatten_nodes, nontrivial_bipartitions

N = 6
TOL = .01
FREQUENCIES = np.array([-.4, 0., .4, -.4, 0., .4])
SUPPORTS = {"A": [(1,2,3),(4,5,6)], "B": [(1,2,3),(1,2,3,4),(5,6)],
            "C": [(1,2),(1,2,3),(1,2,3,4),(5,6)],
            "D": [(1,2),(3,4),(1,2,3,4),(5,6)]}


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str((a.shape, str(a.dtype))).encode()); h.update(a.tobytes())
    return h.hexdigest()


def check(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Persisted 60-minute absolute deadline reached")


def configuration(family, ratio):
    edges = SUPPORTS[family]
    raw = np.array([ratio if any(set(g) < set(h) for g in edges) else 1. for h in edges])
    scale = .75 * math.sqrt(12 / sum(len(h)*w*w for h,w in zip(edges,raw)))
    weights = raw * scale
    payload = dict(family=family, ratio=ratio, supports=[list(e) for e in edges],
                   weights=weights.tolist(), frequency=FREQUENCIES.tolist(), sigma=.08, tau=.2, dt=.01,
                   average_coupling_rms=.75)
    payload['id'] = family + '_' + f'{ratio:g}'.replace('.','p')
    payload['mechanism_hash'] = hashlib.sha256(json.dumps({k:v for k,v in payload.items() if k not in ('id','family','ratio')},sort_keys=True).encode()).hexdigest()
    return payload


def drift(x, cfg):
    result = np.broadcast_to(FREQUENCIES, x.shape).copy()
    for edge, weight in zip(cfg['supports'],cfg['weights']):
        ids = np.array(edge)-1
        total = x[:,ids].sum(1)
        result[:,ids] += weight*np.sin(total[:,None]-len(ids)*x[:,ids])
    return result


def propagate(x, cfg, rng, *, noise=True):
    value=np.asarray(x).copy(); dt=cfg['dt']
    for _ in range(round(cfg['tau']/dt)):
        k1=drift(value,cfg); k2=drift(value+.5*dt*k1,cfg)
        k3=drift(value+.5*dt*k2,cfg); k4=drift(value+dt*k3,cfg)
        value+=dt*(k1+2*k2+2*k3+k4)/6
        if noise:value+=cfg['sigma']*math.sqrt(dt)*rng.normal(size=value.shape)
    return value


def dataset(cfg, seed, blocks=1):
    if blocks>1:
        pieces=[]
        for block in range(blocks):
            if block==0:bx,by,_=dataset(cfg,seed)
            else:
                rng=np.random.default_rng(np.random.SeedSequence([20261017,seed,601,block]))
                bx=rng.uniform(-np.pi,np.pi,(8192,N));by=propagate(bx,cfg,rng)
            pieces.append((bx,by))
        x=np.concatenate([np.concatenate([a[start:stop] for a,b in pieces]) for start,stop in ((0,6144),(6144,7168),(7168,8192))])
        y=np.concatenate([np.concatenate([b[start:stop] for a,b in pieces]) for start,stop in ((0,6144),(6144,7168),(7168,8192))])
        train,val,test=6144*blocks,1024*blocks,1024*blocks
        return x,y,dict(source_future_sha256=digest(x,y),source_sha256=digest(x),
                        train_sha256=digest(x[:train],y[:train]),validation_sha256=digest(x[train:train+val],y[train:train+val]),
                        test_sha256=digest(x[train+val:],y[train+val:]),split=[train,val,test],seed=seed,blocks=blocks,
                        old_partitions_preserved=True,independent_blocks=True)
    rng=np.random.default_rng(np.random.SeedSequence([20261017,seed,601]))
    x=rng.uniform(-np.pi,np.pi,(8192,N)); y=propagate(x,cfg,rng)
    return x,y,dict(source_future_sha256=digest(x,y),source_sha256=digest(x),
                   train_sha256=digest(x[:6144],y[:6144]),validation_sha256=digest(x[6144:7168],y[6144:7168]),
                   test_sha256=digest(x[7168:],y[7168:]),split=[6144,1024,1024],seed=seed)


@lru_cache(maxsize=1)
def universal_frequencies():
    """All possible 2/3/4-body terms, independent of the candidate's true supports."""
    vectors=set()
    for size in (2,3,4):
        for support in itertools.combinations(range(N),size):
            for receiver in support:
                vector=np.zeros(N,dtype=int);vector[list(support)]=1;vector[receiver]=-(size-1)
                if vector[np.flatnonzero(vector)[0]]<0:vector=-vector
                vectors.add(tuple(vector))
    return np.array(sorted(vectors),dtype=float)


def features(x, jacobian=False, kind='node'):
    columns=[]; derivatives=[]
    for harmonic in (1,2,3):
        columns.extend((np.sin(harmonic*x),np.cos(harmonic*x)))
        if jacobian:
            derivatives.extend((harmonic*np.cos(harmonic*x),-harmonic*np.sin(harmonic*x)))
    value=np.concatenate(columns,axis=1)
    if kind=='universal_fourier':
        frequencies=universal_frequencies();phase=x@frequencies.T
        value=np.c_[value,np.sin(phase),np.cos(phase)]
    if not jacobian:return value
    j=np.zeros((len(x),36,N))
    for block,d in enumerate(derivatives):
        for i in range(N):j[:,block*N+i,i]=d[:,i]
    if kind=='universal_fourier':
        j=np.concatenate((j,np.cos(phase)[:,:,None]*frequencies[None],-np.sin(phase)[:,:,None]*frequencies[None]),axis=1)
    return value,j


class MLPChannel:
    def __init__(self,payload):
        self.payload=payload
        for name in ('weights','biases'):
            setattr(self,name,[np.asarray(a,dtype=float) for a in payload[name]])
        for name in ('feature_mean','feature_scale','output_mean','output_scale','covariance'):
            setattr(self,name,np.asarray(payload[name],dtype=float))
        self.chol=np.linalg.cholesky(self.covariance)
        self.precision=np.linalg.inv(self.covariance)
        self.lognorm=-.5*(N*np.log(2*np.pi)+np.linalg.slogdet(self.covariance)[1])

    def mean(self,x,jacobian=False):
        if jacobian:
            h,j=features(x,True,self.payload.get('feature_kind','node')); j/=self.feature_scale[None,:,None]
        else:h=features(x,kind=self.payload.get('feature_kind','node'))
        h=(h-self.feature_mean)/self.feature_scale
        for i,(w,b) in enumerate(zip(self.weights,self.biases)):
            h=h@w+b
            if jacobian:j=np.einsum('nki,ko->noi',j,w,optimize=True)
            if i<2:
                sigmoid=expit(h)
                if jacobian:j*=(sigmoid+h*sigmoid*(1-sigmoid))[:,:,None]
                h=h*sigmoid
        value=x+h*self.output_scale+self.output_mean
        return (value,np.eye(N)[None]+j*self.output_scale[None,:,None]) if jacobian else value

    def log_prob(self,x,y):
        r=y-self.mean(x)
        return self.lognorm-.5*np.einsum('ni,ij,nj->n',r,self.precision,r,optimize=True)

    def sample(self,x,z):
        return self.mean(x)+z@self.chol.T


def fit_mlp(x,y,seed,deadline,feature_kind='node',split=(6144,1024,1024)):
    import torch
    from torch import nn
    torch.set_num_threads(2); torch.manual_seed(20261017+seed)
    f=features(x,kind=feature_kind); delta=y-x
    train=split[0];validation_end=train+split[1]
    fm=f[:train].mean(0); fs=f[:train].std(0)
    om=delta[:train].mean(0); os=delta[:train].std(0)
    tx=torch.tensor((f[:train]-fm)/fs,dtype=torch.float32)
    ty=torch.tensor((delta[:train]-om)/os,dtype=torch.float32)
    vx=torch.tensor((f[train:validation_end]-fm)/fs,dtype=torch.float32)
    vy=torch.tensor((delta[train:validation_end]-om)/os,dtype=torch.float32)
    net=nn.Sequential(nn.Linear(f.shape[1],128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,N))
    optimizer=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=1e-5)
    best=None; best_loss=float('inf'); best_epoch=0; history=[]; started=time.time()
    for epoch in range(1,801):
        check(deadline)
        for ids in torch.randperm(len(tx)).split(1024):
            optimizer.zero_grad(); loss=((net(tx[ids])-ty[ids])**2).mean();loss.backward();optimizer.step()
        with torch.no_grad():validation=float(((net(vx)-vy)**2).mean())
        history.append(validation)
        if validation<best_loss-1e-7:
            best_loss=validation;best_epoch=epoch
            best={k:v.detach().clone() for k,v in net.state_dict().items()}
        if epoch%100==0:print(json.dumps(dict(stage='mlp',epoch=epoch,validation=validation)),flush=True)
        if epoch>=200 and epoch-best_epoch>=100:break
    net.load_state_dict(best)
    linear=[a for a in net if isinstance(a,nn.Linear)]
    payload=dict(weights=[a.weight.detach().numpy().T.tolist() for a in linear],
                 biases=[a.bias.detach().numpy().tolist() for a in linear],
                 feature_mean=fm.tolist(),feature_scale=fs.tolist(),output_mean=om.tolist(),output_scale=os.tolist(),
                 covariance=np.eye(N).tolist(),family='six_periodic_mlp_full_gaussian_tm',source_count=N,
                 feature_kind=feature_kind)
    model=MLPChannel(payload)
    residual=y[train:validation_end]-model.mean(x[train:validation_end])
    payload['covariance']=np.cov(residual,rowvar=False).tolist()
    model=MLPChannel(payload)
    validation_prediction=x[train:validation_end]+net(vx).detach().numpy()*os+om
    error=float(abs(validation_prediction-model.mean(x[train:validation_end])).max())
    if error>1e-5:raise ArithmeticError(f'NumPy/Torch mismatch {error}')
    return model,dict(epochs=epoch,selected_epoch=best_epoch,validation_history=history,
                      training_seconds=time.time()-started,numpy_torch_error=error,
                      residual_mean=residual.mean(0).tolist(),noise_floor=None)


def residual_state_diagnostics(model,x,y):
    """Descriptive held-out checks; these do not certify a conditional density."""
    residual=(y-model.mean(x))@np.linalg.inv(model.chol.T)
    f=features(x);correlation=np.corrcoef(np.c_[f,residual,residual*residual],rowvar=False)
    radius=np.sum(residual*residual,axis=1);bins=[]
    boundaries=np.linspace(-np.pi,np.pi,5)
    for node in range(N):
        for left,right in zip(boundaries[:-1],boundaries[1:]):
            mask=(x[:,node]>=left)&(x[:,node]<right)
            bins.append(dict(node=node+1,left=float(left),right=float(right),count=int(mask.sum()),
                             mean_whitened_squared_radius=float(radius[mask].mean()),
                             joint95coverage=float(np.mean(radius[mask]<=chi2.ppf(.95,N)))))
    return dict(test_count=len(x),max_residual_feature_correlation=float(abs(correlation[:36,36:42]).max()),
                max_squared_residual_feature_correlation=float(abs(correlation[:36,42:]).max()),phase_bins=bins)


def channel_diagnostics(model,x,y,cfg,seed,split=(6144,1024,1024)):
    test_start=split[0]+split[1]
    test=x[test_start:]; observed=y[test_start:]; residual=observed-model.mean(test)
    mahal=np.einsum('ni,ij,nj->n',residual,model.precision,residual)
    standardized=residual@np.linalg.inv(model.chol.T)
    corr=np.corrcoef(standardized,rowvar=False)-np.eye(N)
    rng=np.random.default_rng(np.random.SeedSequence([20261017,seed,699]))
    probes=rng.uniform(-np.pi,np.pi,(128,N))
    # Independent stochastic repeats; never passed to model fitting or EI.
    repeated=propagate(np.tile(probes,(128,1)),cfg,rng).reshape(128,128,N)
    mean=repeated.mean(0); reference_sem=np.sqrt(np.mean(repeated.var(0,ddof=1)/128))
    signal=float(np.sqrt(np.mean((mean-probes)**2)))
    mean_rmse=float(np.sqrt(np.mean((model.mean(probes)-mean)**2)))
    coverage=float(np.mean(mahal<=chi2.ppf(.95,N)))
    result=dict(test_rmse_rad=float(np.sqrt(np.mean(residual**2))),
                test_conditional_nll_nats=float(-model.log_prob(test,observed).mean()),
                joint_95_coverage=coverage,whitened_residual_correlation_max=float(abs(corr).max()),
                independent_mean_rmse_rad=mean_rmse,independent_mean_nrmse=mean_rmse/signal,
                independent_reference_sem_rad=float(reference_sem),signal_rms_rad=signal)
    result['fit_eligible']=bool(mean_rmse/signal<=.15 and .88<=coverage<=.995)
    result['interpretation']='engineering eligibility, not proof of exact conditional Gaussian dynamics'
    return result


class LinearChannel:
    def __init__(self,matrix,covariance):
        self.matrix=np.asarray(matrix);self.covariance=np.asarray(covariance)
        self.chol=np.linalg.cholesky(self.covariance);self.precision=np.linalg.inv(self.covariance)
        self.lognorm=-.5*(N*np.log(2*np.pi)+np.linalg.slogdet(self.covariance)[1])
    def mean(self,x,jacobian=False):
        mean=x@self.matrix.T
        return (mean,np.broadcast_to(self.matrix,(len(x),N,N)).copy()) if jacobian else mean
    def log_prob(self,x,y):
        r=y-self.mean(x);return self.lognorm-.5*np.einsum('ni,ij,nj->n',r,self.precision,r)
    def sample(self,x,z):return self.mean(x)+z@self.chol.T


class JointIntegrator:
    """Full-joint conditional TM; importance proposals change efficiency only."""
    def __init__(self,model,outer,particles,seed,deadline,prior='uniform'):
        self.model=model;self.outer=outer;self.particles=particles;self.deadline=deadline;self.prior=prior
        u=qmc.Sobol(12,scramble=True,seed=seed).random_base2(round(np.log2(outer)))
        self.x=(u[:,:N]-.5)*2*np.pi if prior=='uniform' else ndtri(u[:,:N])
        self.y=model.sample(self.x,ndtri(u[:,N:]))
        self.u=qmc.Sobol(7,scramble=True,seed=seed+1).random_base2(round(np.log2(outer*particles))).reshape(outer,particles,7)
        self.marginals={};self.diagnostics={}

    def proposal(self,missing):
        completed=self.x.copy();completed[:,missing]=self.y[:,missing]
        if self.prior=='uniform':completed[:,missing]=np.clip(completed[:,missing],-np.pi,np.pi)
        remaining=0.
        for _ in range(12):
            mean,jac=self.model.mean(completed,True);j=jac[:,:,missing]
            h=np.einsum('nti,tu, nuj->nij',j,self.model.precision,j,optimize=True)
            gradient=np.einsum('nti,tu,nu->ni',j,self.model.precision,self.y-mean,optimize=True)
            if self.prior=='normal':
                h+=np.eye(len(missing))[None];gradient-=completed[:,missing]
            h+=1e-10*np.eye(len(missing))[None]
            step=np.linalg.solve(h,gradient)
            step*=np.minimum(1.,.5/np.maximum(abs(step).max(1),1e-300))[:,None]
            updated=completed[:,missing]+step
            if self.prior=='uniform':updated=np.clip(updated,-np.pi,np.pi)
            remaining=float(abs(updated-completed[:,missing]).max())
            completed[:,missing]=updated
            if remaining<1e-7:break
        _,jac=self.model.mean(completed,True);j=jac[:,:,missing]
        h=np.einsum('nti,tu,nuj->nij',j,self.model.precision,j,optimize=True)
        if self.prior=='normal':h+=np.eye(len(missing))[None]
        covariance=1.5*np.linalg.inv(h+1e-10*np.eye(len(missing))[None])
        return completed[:,missing],np.linalg.cholesky(covariance),remaining

    @staticmethod
    def normal_logpdf(value,center,chol):
        z=np.linalg.solve(chol[:,None,:,:],(value-center[:,None,:])[...,None])[...,0]
        return -.5*np.sum(z*z,axis=-1)-np.log(np.diagonal(chol,axis1=1,axis2=2)).sum(1)[:,None]-.5*value.shape[-1]*np.log(2*np.pi)

    def marginal(self,mask):
        check(self.deadline)
        if mask in self.marginals:return self.marginals[mask]
        if mask==63:
            value=self.model.log_prob(self.x,self.y)
            self.marginals[mask]=value;return value
        missing=np.array([i for i in range(N) if not mask>>i&1]);d=len(missing)
        center,chol,remaining=self.proposal(missing)
        values=[];ess=[];maximum=[]
        for start in range(0,self.outer,max(1,32768//self.particles)):
            check(self.deadline);stop=min(self.outer,start+max(1,32768//self.particles))
            u=self.u[start:stop];z=ndtri(u[:,:,1:d+1]);c=center[start:stop];l=chol[start:stop]
            normal=c[:,None,:]+np.einsum('npk,njk->npj',z,l)
            wide=c[:,None,:]+np.einsum('npk,njk->npj',z,l*3)
            prior=(u[:,:,1:d+1]-.5)*2*np.pi if self.prior=='uniform' else z
            proposed=np.where((u[:,:,0]<.05)[:,:,None],prior,
                              np.where((u[:,:,0]<.20)[:,:,None],wide,normal))
            local_log=self.normal_logpdf(proposed,c,l)
            wide_log=self.normal_logpdf(proposed,c,l*3)
            if self.prior=='uniform':
                logprior=np.where(np.all(abs(proposed)<=np.pi,axis=2),-d*np.log(2*np.pi),-np.inf)
            else:logprior=-.5*np.sum(proposed**2,axis=2)-.5*d*np.log(2*np.pi)
            logproposal=logsumexp(np.stack((np.log(.8)+local_log,np.log(.15)+wide_log,np.log(.05)+logprior)),axis=0)
            full=np.repeat(self.x[start:stop,None,:],self.particles,axis=1);full[:,:,missing]=proposed
            target=np.repeat(self.y[start:stop],self.particles,axis=0)
            weights=self.model.log_prob(full.reshape(-1,N),target).reshape(stop-start,self.particles)+logprior-logproposal
            normalizer=logsumexp(weights,axis=1)
            normalized=weights-normalizer[:,None]
            values.extend(normalizer-np.log(self.particles))
            ess.extend(np.exp(-logsumexp(2*normalized,axis=1)))
            maximum.extend(np.exp(normalized.max(1)))
        value=np.asarray(values)
        if not np.isfinite(value).all():raise ArithmeticError(f'Nonfinite marginal {mask}')
        self.marginals[mask]=value
        self.diagnostics[str(mask)]=dict(ess_min=float(np.min(ess)),ess_median=float(np.median(ess)),
                                        maximum_weight=float(np.max(maximum)),remaining_mode_step=remaining)
        return value

    def information(self):
        denominator=self.marginal(0)
        points=np.array([self.marginal(mask)-denominator for mask in range(1,64)])
        estimates={mask:float(points[mask-1].mean()) for mask in range(1,64)}
        return estimates,points,self.diagnostics


def subsets(mask):return tuple(i+1 for i in range(N) if mask>>i&1)
def mask_of(s):return sum(1<<(i-1) for i in s)


def spt_from_ei(estimates):
    table={subsets(int(mask)):float(value) for mask,value in estimates.items()}
    residual=[];xi=[];monotonic=[]
    for a,value in table.items():
        xi.append(value-sum(table[(i,)] for i in a))
        for l,r in nontrivial_bipartitions(a):residual.append(value-table[l]-table[r])
        for i in a:
            if len(a)>1:monotonic.append(value-table[tuple(j for j in a if j!=i)])
    residual=np.array(residual);xi=np.array(xi)
    audit=dict(tolerance_nats=TOL,minimum_syn_nats=float(residual.min()),
               tolerance_negative_count=int(np.sum((residual<0)&(residual>=-TOL))),
               violation_count=int(np.sum(residual < -TOL)),minimum_xi_nats=float(xi.min()),
               xi_violation_count=int(np.sum(xi < -TOL)),candidate_count=len(residual),
               minimum_monotonic_difference=float(min(monotonic)),clipping=False)
    if audit['violation_count'] or audit['xi_violation_count']:
        return dict(status='failed_nonnegativity',audit=audit,tree=None)
    result=build_spt_from_ei_table(tuple(range(1,7)),table,config=SPTConfig(syn_tolerance=TOL,eps=0,complete_to_singletons=True))
    def convert(n):return dict(sources=list(n.sources),children=[convert(c) for c in n.children],syn_nats=n.syn_value)
    return dict(status='valid',tree=convert(result.root),audit=audit,closure_error_nats=result.closure_error)


def tree_clades(tree):
    value=[set(tree['sources'])] if 1<len(tree['sources'])<N else []
    for child in tree['children']:value+=tree_clades(child)
    return value


def recovery(tree,supports):
    truth=[set(s) for s in supports];pred=tree_clades(tree)
    primitives=[h for h in truth if not any(g<h for g in truth)]
    filtered=[p for p in pred if not any(p<h for h in primitives)]
    def score(p):
        true=sum(s in truth for s in p)
        return 2*true/(len(p)+len(truth)) if p or truth else 1.
    return dict(module_f1=score(filtered),strict_f1=score(pred),
                recovered_supports=[sorted(s) for s in truth if s in pred],
                missing_supports=[sorted(s) for s in truth if s not in pred],
                scored_clades=[sorted(p) for p in filtered])
