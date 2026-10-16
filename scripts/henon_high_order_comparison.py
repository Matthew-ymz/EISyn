"""Bounded natural-trajectory Hénon-inspired, six-node higher-order benchmark.

The Gaussian transport in a degree-four Legendre feature space is an explicit
approximation. It is neither an exact nonlinear density nor a pure-order PID.
No planted edges are supplied to a fitted model or any information estimator.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit, ndtri, logsumexp
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'results/henon_high_order_comparison'
FIG = ROOT / 'fig/henon_high_order_comparison'
NETWORKS = {'A': ((0, 1, 2), (3, 4, 5)),
            'B': ((0, 1, 2, 3), (2, 3, 4, 5))}
TOLERANCE = .01
VERSION = 'henon-ho-natural-v2-bounded-product'
ENFORCE_BUDGET = __name__ == '__main__'
CONFIG = dict(version=VERSION, networks=NETWORKS, lambdas=[0., .5, 1.], seed=0,
              chains=64, burn_in=256, steps=256, split_chains=[48, 8, 8],
              sigma=.5, degree=4, quadrature_orders=[5, 6],
              mlp_hidden=[64, 64], mlp_epochs=240, patience=30,
              mlp_lr=.003, mlp_batch=1024, covariance_ridge=1e-6,
              shap_evaluation=128, shap_background=32,
              granger_epochs=120, granger_penalty=.01,
              pcmci_knn=.1, pcmci_shuffles=30, pcmci_alpha=.05,
              pcmci_max_conditions=2, syn_tolerance_nats=TOLERANCE)
CONFIG['conditional_tm'] = dict(outer=1024, inner=256, scrambles=4,
    refinement_outer=2048, refinement_inner=512,
    refinement_rule='refine inner when 128->256 EI change >=0.01; refine outer on significant negative mean Syn; shared deadline')
CONFIG['random_streams'] = dict(sampling=20261016, chain_split=9926, mlp=20261016,
                                shap=711, neural_granger_base=881, pcmci=20261016,
                                conditional_tm_base=9021, conditional_tm_scramble_step=1000)


def sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


def check():
    if not ENFORCE_BUDGET:return
    if (FOLDER / 'budget.json').exists():
        budget = json.loads((FOLDER / 'budget.json').read_text())
        if time.time() >= budget['deadline_epoch']:
            raise TimeoutError('Approved absolute 45-minute budget exhausted')


def mean_dynamics(x, case, lam):
    x = np.asarray(x, float)
    gamma, kappa = .3 + 1.7 * lam, .5 - .4 * lam
    joint, degree = np.zeros_like(x), np.zeros(6)
    for edge in NETWORKS[case]:
        # Same amplitude bound at both orders; avoids a 3**(order/2)
        # positive-feedback amplification in a closed stochastic recurrence.
        value = 2*np.prod(x[:, edge], axis=1)
        joint[:, edge] += value[:, None]
        degree[list(edge)] += 1
    return .5 * (1 - 1.4*x*x) + .4*gamma*x + kappa*joint/degree


def natural_data(case, lam, folder):
    path = folder / 'natural.npz'
    if path.exists():
        with np.load(path) as d:
            states, latent, parts = d['states'], d['latent'], d['chain_order']
    else:
        # Identical innovations and initial conditions pair the six conditions.
        rng = np.random.default_rng(20261016)
        x = rng.uniform(-.9, .9, (64, 6))
        states, latent = [], []
        for t in range(CONFIG['burn_in'] + CONFIG['steps']):
            check()
            y = mean_dynamics(x, case, lam) + .5*rng.normal(size=x.shape)
            nx = np.tanh(y)
            if t >= CONFIG['burn_in']:
                states.append(x.copy()); latent.append(y.copy())
            x = nx
        states = np.asarray(states); latent = np.asarray(latent)
        parts = np.random.default_rng(9926).permutation(64)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, states=states, latent=latent, chain_order=parts)
    indices = {'train': parts[:48], 'validation': parts[48:56], 'test': parts[56:]}
    data = {k: (states[:, c].reshape(-1, 6), latent[:, c].reshape(-1, 6)) for k, c in indices.items()}
    manifest = dict(generator=VERSION, array_sha256=sha(states, latent),
                    pairing='common initial states and innovations; states differ across conditions',
                    partitions={k: dict(chains=c.tolist(), pairs=len(data[k][0]),
                                        sha256=sha(*data[k])) for k, c in indices.items()},
                    target='all six atanh(x[t+1]) coordinates; lag=1',
                    shared_by=['SPT training', 'SHAP training', 'Neural Granger training',
                               'WMS natural pairs', 'MMI natural pairs', 'SURD natural pairs',
                               'PCMCI natural test trajectories'])
    save(folder/'data_manifest.json', manifest)
    return data, states, latent, indices, manifest


class MLP:
    def __init__(self, payload):
        self.payload = payload
        for k in ('weights', 'biases'):
            setattr(self, k, [np.asarray(a, float) for a in payload[k]])
        for k in ('xm', 'xs', 'ym', 'ys', 'noise_covariance'):
            setattr(self, k, np.asarray(payload[k], float))

    def mean(self, x):
        a = (np.asarray(x)-self.xm)/self.xs
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            a = a @ w + b
            if i < len(self.weights)-1: a = a*expit(a)
        return a*self.ys + self.ym

    def digest(self):
        return hashlib.sha256(json.dumps(self.payload, sort_keys=True).encode()).hexdigest()


def fit_mlp(data, folder):
    path = folder/'mlp.json'
    if path.exists():
        payload=json.loads(path.read_text())
        if payload.get('version')!=VERSION or payload.get('training_pairs_sha256')!=sha(*data['train']):
            raise ValueError('Cached MLP does not match generator version or natural training pairs')
        return MLP(payload)
    import torch
    torch.set_num_threads(1); torch.manual_seed(20261016)
    x, y = data['train']; vx, vy = data['validation']
    xm, xs = x.mean(0), x.std(0); ym, ys = y.mean(0), y.std(0)
    if min(xs.min(), ys.min()) <= 0: raise ArithmeticError('Degenerate training coordinate')
    tx = torch.tensor(((x-xm)/xs).tolist(), dtype=torch.float32)
    ty = torch.tensor(((y-ym)/ys).tolist(), dtype=torch.float32)
    tvx = torch.tensor(((vx-xm)/xs).tolist(), dtype=torch.float32)
    tvy = torch.tensor(((vy-ym)/ys).tolist(), dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(6, 64), torch.nn.SiLU(),
                              torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 6))
    opt = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=1e-5)
    best, selected, state = float('inf'), 0, None
    history = []
    for epoch in range(1, CONFIG['mlp_epochs']+1):
        check(); net.train()
        for ids in torch.randperm(len(tx)).split(1024):
            opt.zero_grad(set_to_none=True)
            loss = ((net(tx[ids])-ty[ids])**2).mean(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): val = float(((net(tvx)-tvy)**2).mean())
        history.append(val)
        if val < best:
            best, selected = val, epoch
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if epoch-selected >= CONFIG['patience']: break
    if state is None: raise RuntimeError('No completed MLP epoch')
    net.load_state_dict(state)
    layers = [v for v in net if isinstance(v, torch.nn.Linear)]
    payload = dict(version=VERSION, weights=[v.weight.detach().t().tolist() for v in layers],
                   biases=[v.bias.detach().tolist() for v in layers],
                   xm=xm.tolist(), xs=xs.tolist(), ym=ym.tolist(), ys=ys.tolist(),
                   noise_covariance=np.eye(6).tolist(), selected_epoch=selected,
                   completed_epochs=epoch, validation_scaled_mse=best,
                   training_pairs_sha256=sha(x,y), validation_loss=history)
    model = MLP(payload); residual = y-model.mean(x)
    # One triangular Gaussian residual TM, fitted on natural training pairs only.
    cov = residual.T@residual/len(residual)
    np.linalg.cholesky(cov)
    payload['noise_covariance'] = cov.tolist(); save(path, payload)
    return MLP(payload)


def diagnostics(model, data, case, lam):
    def score(x, y):
        p = model.mean(x); residual=y-p
        total=((y-y.mean(0))**2).sum(0)
        return dict(rmse=float(np.sqrt(np.mean(residual**2))),
                    per_output_r2=(1-(residual**2).sum(0)/total).tolist(),
                    mean_r2=float(np.mean(1-(residual**2).sum(0)/total)),
                    mean_function_rmse=float(np.sqrt(np.mean((p-mean_dynamics(x,case,lam))**2))),
                    noise_sd=.5)
    tx, ty = data['test']
    u=qmc.Sobol(6, scramble=True, seed=511).random_base2(12)*2-1
    # Truth is diagnostic only: never train, choose epochs or tune from these rows.
    uy=mean_dynamics(u,case,lam)+np.random.default_rng(9921).normal(0,.5,u.shape)
    baseline=.5*(1-1.4*u*u)+.4*(.3+1.7*lam)*u
    signal=float(np.sqrt(np.mean((mean_dynamics(u,case,lam)-baseline)**2)))
    uniform=score(u,uy)
    uniform.update(joint_signal_rms=signal,
                   model_error_to_joint_signal_rms=uniform['mean_function_rmse']/signal)
    return dict(natural_test=score(tx,ty), uniform_diagnostic=uniform,
                uniform_diagnostic_used_for_training_or_selection=False,
                natural_test_source_sha256=sha(tx), model_sha256=model.digest())


def exponents(degree=4):
    return np.asarray([e for e in itertools.product(range(degree+1), repeat=6)
                       if 0 < sum(e) <= degree], int)


EXPONENTS = exponents()
SUPPORT = np.asarray([sum(1 << i for i in np.flatnonzero(e)) for e in EXPONENTS])


def features(x):
    x = np.asarray(x)
    polys = [np.ones_like(x), x, (3*x*x-1)/2,
             (5*x**3-3*x)/2, (35*x**4-30*x*x+3)/8]
    out = np.ones((len(x), len(EXPONENTS)))
    for i in range(6):
        for degree in range(1,5):
            col = EXPONENTS[:, i] == degree
            out[:, col] *= (np.sqrt(2*degree+1)*polys[degree][:, i])[:,None]
    return out


def quadrature(order):
    v,w = np.polynomial.legendre.leggauss(order)
    ids = np.asarray(list(itertools.product(range(order), repeat=6)))
    return v[ids], np.prod((w/2)[ids], axis=1)


def ld(a):
    sign,value = np.linalg.slogdet(a)
    if sign <= 0 or not np.isfinite(value): raise ArithmeticError('Non-positive TM covariance')
    return float(value)


def intervention_table(model, order):
    check(); x,w=quadrature(order); f=features(x); mu=model.mean(x)
    centered=mu-w@mu
    cy=(centered*w[:,None]).T@centered+model.noise_covariance
    cross=(f*w[:,None]).T@centered
    table=np.zeros(64); conditional=[]
    for mask in range(64):
        cols=(SUPPORT & mask)==SUPPORT
        c=cross[cols]
        residual=cy-c.T@c
        table[mask]=.5*(ld(cy)-ld(residual)); conditional.append(residual)
    # At order >=5, all degree-four source products are integrated exactly.
    gram=(f*w[:,None]).T@f
    return table, dict(quadrature_order=order, states=len(x),
        feature_mean_maximum=float(abs(w@f).max()),
        feature_orthogonality_error=float(abs(gram-np.eye(len(gram))).max()),
        minimum_conditional_eigenvalue=float(min(np.linalg.eigvalsh(c).min() for c in conditional)),
        source='independent Uniform(-1,1)^6', degree=4,
        family='shared affine Gaussian transport in orthonormal Legendre feature space',
        exact_physical_mutual_information=False, wishart_correction=False,
        reason_no_wishart='weighted deterministic quadrature; covariance is not an iid Wishart estimate')


def conditional_tm_table(model, outer=512, inner=512, seed=9021):
    """Marginals of one normalized conditional Gaussian TM, same full joint.

    Independent source coordinates, triangular residual transport; uniform
    missing-coordinate marginalization. No Gaussian source-feature proxy.
    Inner prefix, outer prefix, and between-scramble errors remain visible.
    """
    check();started=time.monotonic()
    points=qmc.Sobol(12,scramble=True,seed=seed).random_base2(int(np.log2(outer)))
    source=points[:,:6]*2-1
    chol=np.linalg.cholesky(model.noise_covariance)
    target=model.mean(source)+ndtri(points[:,6:])@chol.T
    inverse=np.linalg.inv(chol.T)
    constant=-3*np.log(2*np.pi)-np.log(np.diag(chol)).sum()
    background=qmc.Sobol(6,scramble=True,seed=seed+100000).random_base2(int(np.log2(inner)))*2-1
    logs=np.zeros((64,outer));prefix=np.zeros((64,outer))
    for mask in range(64):
        check()
        if mask==63:
            e=(target-model.mean(source))@inverse
            logs[mask]=constant-.5*(e*e).sum(1);prefix[mask]=logs[mask]
            continue
        cols=[i for i in range(6) if mask>>i&1]
        for start in range(0,outer,32):
            check(); stop=min(outer,start+32);count=stop-start
            rows=np.broadcast_to(background,(count,inner,6)).copy()
            rows[:,:,cols]=source[start:stop,None,cols]
            means=model.mean(rows.reshape(-1,6)).reshape(count,inner,6)
            e=(target[start:stop,None,:]-means)@inverse
            ll=constant-.5*(e*e).sum(2)
            logs[mask,start:stop]=logsumexp(ll,axis=1)-np.log(inner)
            prefix[mask,start:stop]=logsumexp(ll[:,:inner//2],axis=1)-np.log(inner//2)
    rows=logs-logs[0]; half=prefix-prefix[0]
    table=rows.mean(1);small=half.mean(1)
    return table,dict(outer=outer,inner=inner,seed=seed,
        inner_prefix_maximum_ei_change_nats=float(abs(table-small).max()),
        outer_prefix_maximum_ei_change_nats=float(abs(table-rows[:,:outer//2].mean(1)).max()),
        elapsed_seconds=time.monotonic()-started,
        inner_prefix_table_nats=small.tolist(),
        source='independent Uniform(-1,1)^6',target='same complete six latent future coordinates',
        family='nonlinear conditional residual Gaussian TM; prior Sobol marginal integration',
        model_sha256=model.digest(),wishart_correction=False,
        reason_no_wishart='normalized conditional TM with numerical marginal integration, not an affine joint Wishart estimate')


def splits(mask):
    first=mask & -mask
    left=(mask-1)&mask
    while left:
        if left & first and left != mask: yield left,mask^left
        left=(left-1)&mask


def members(mask): return [i+1 for i in range(6) if mask >> i & 1]


def audit_nonnegative(values):
    a=np.asarray(values, float)
    affected=int((a < -TOLERANCE).sum())
    audit=dict(tolerance_nats=TOLERANCE, minimum_nats=float(a.min()),
        tolerance_band_negative_count=int(((a < 0)&(a >= -TOLERANCE)).sum()),
        significant_violation_count=affected, checked_count=int(a.size), clipping=False)
    if affected:
        raise ArithmeticError(f'Syn minimum={a.min()} nats; threshold={-TOLERANCE}; affected_count={affected}')
    return audit


def spt(table):
    xi=np.array([table[m]-sum(table[1<<i] for i in range(6) if m>>i&1) for m in range(64)])
    raw=[xi[m] for m in range(64)]
    raw.extend(table[m]-table[l]-table[r] for m in range(1,64) for l,r in splits(m))
    audit=audit_nonnegative(raw)
    def build(mask):
        node=dict(mask=mask, members=members(mask), xi_nats=float(xi[mask]))
        if mask.bit_count()>1:
            l,r=min(splits(mask), key=lambda p:(-xi[p[0]]-xi[p[1]],members(p[0]),members(p[1])))
            node.update(syn_nats=float(table[mask]-table[l]-table[r]),
                        children=[build(l),build(r)])
        return node
    tree=build(63)
    nodes=[]
    def walk(n):
        if 'children' in n:
            nodes.append(n)
            for c in n['children']:walk(c)
    walk(tree)
    closure=float(sum(n['syn_nats'] for n in nodes)-xi[63])
    if abs(closure)>1e-9:raise ArithmeticError('SPT telescoping identity failed')
    return dict(tree=tree, integrated_ei_nats=float(xi[63]), joint_ei_nats=float(table[63]),
        single_ei_nats=table[[1,2,4,8,16,32]].tolist(), node_count=len(nodes),
        closure_error_nats=closure, audit=audit, all_xi_nats=xi.tolist(),
        high_order_mass_nats=float(sum(n['syn_nats'] for n in nodes if len(n['members'])>=3)))


def natural_information(data):
    # Same held-out natural pairs for WMS, grouped MMI and statewise SURD.
    x,y=data['test']; f=features(x); f-=f.mean(0); scales=f.std(0)
    if scales.min()<=0: raise ArithmeticError('Degenerate natural feature')
    f/=scales
    yy=y-y.mean(0); ys=yy.std(0); yy/=ys
    n=len(x); ridge=CONFIG['covariance_ridge']
    cx=f.T@f/n+ridge*np.eye(f.shape[1]); cy=yy.T@yy/n+ridge*np.eye(6)
    cross=f.T@yy/n; inv=cho_solve(cho_factor(cy),np.eye(6))
    table=np.zeros(64); specific=np.zeros((n,64))
    for mask in range(1,64):
        check(); cols=np.flatnonzero((SUPPORT & mask)==SUPPORT)
        c=cross[cols]; k=c.T@cho_solve(cho_factor(cx[np.ix_(cols,cols)]),c)
        table[mask]=.5*(ld(cy)-ld(cy-k))
        operator=inv@k@inv
        specific[:,mask]=table[mask]-.5*np.trace(inv@k)+.5*np.einsum('ni,ij,nj->n',yy,operator,yy)
    # Native multivariate SURD allocation, target state first, then average.
    red,syn=np.zeros(64),np.zeros(64)
    masks=np.arange(1,64); sizes=np.array([int(m).bit_count() for m in masks])
    max_closure=0.
    for row in specific:
        adjusted=row[1:].copy()
        for order in range(2,7):
            adjusted[(sizes==order)&(adjusted<adjusted[sizes==order-1].max())]=0.
        ranking=np.lexsort((masks,sizes,adjusted)); increments=np.diff(adjusted[ranking],prepend=0.)
        r,s=np.zeros(64),np.zeros(64); remaining=63
        for pos,increment in zip(ranking,increments):
            mask=int(masks[pos])
            if sizes[pos]==1:r[remaining]+=increment;remaining^=mask
            else:s[mask]+=increment
        max_closure=max(max_closure,abs(float(r.sum()+s.sum()-row[63])))
        red+=r/n;syn+=s/n
    if max_closure>1e-8:raise ArithmeticError('Statewise SURD closure failed')
    mmi=[(table[63]-max(table[l],table[r]),l,r) for l,r in splits(63)]
    mv,l,r=min(mmi)
    return dict(data_sha256=sha(x,y), family='degree-four feature-lifted observational affine Gaussian TM',
        observational_mi_nats=table.tolist(), wms_nats=float(table[63]-table[[1,2,4,8,16,32]].sum()),
        mmi_grouped_nats=float(mv), mmi_split=[members(l),members(r)],
        mmi_rule='minimum bivariate grouped MMI over all unordered root bipartitions; not a multivariate PID atom',
        surd_total_synergy_nats=float(syn.sum()),
        surd_high_order_synergy_nats=float(sum(syn[m] for m in range(1,64) if m.bit_count()>=3)),
        surd_synergy_by_coalition_nats=syn.tolist(), surd_redundancy_unique_nats=red.tolist(),
        surd_maximum_statewise_closure_error_nats=max_closure,
        surd_minimum_specific_nats=float(specific.min()),
        covariance_ridge=ridge, test_pairs=n,
        wishart_correction=False, reason_no_wishart='non-Gaussian polynomial features and serially dependent natural pairs',
        maximum_specific_mean_vs_mi_difference=float(abs(specific.mean(0)-table).max()))


def shap_interactions(model,data):
    rng=np.random.default_rng(711); tx=data['train'][0]; ex=data['test'][0]
    background=tx[rng.choice(len(tx),32,replace=False)]
    evaluation=ex[rng.choice(len(ex),128,replace=False)]
    values=np.zeros((64,128,6))
    for mask in range(64):
        check(); rows=np.broadcast_to(background,(128,32,6)).copy()
        cols=[i for i in range(6) if mask>>i&1]
        rows[:,:,cols]=evaluation[:,None,cols]
        values[mask]=model.mean(rows.reshape(-1,6)).reshape(128,32,6).mean(1)
    shap=np.zeros((128,6,6)); interactions={}
    for group in range(1,64):
        k=group.bit_count()
        if k>4:continue
        result=np.zeros((128,6)); complement=63^group
        for context in range(64):
            if context & group:continue
            t=context.bit_count()
            weight=math.factorial(t)*math.factorial(6-k-t)/math.factorial(7-k)
            delta=np.zeros((128,6)); sub=group
            while True:
                delta+=(-1)**(k-sub.bit_count())*values[context|sub]
                if sub==0:break
                sub=(sub-1)&group
            result+=weight*delta
        if k==1:shap[:,members(group)[0]-1,:]=result
        if k>=2:interactions[str(group)]=float(np.mean(np.abs(result)))
    error=float(abs(shap.sum(1)-(values[63]-values[0])).max())
    if error>1e-8:raise ArithmeticError('Exact Shapley efficiency failed')
    return dict(model_sha256=model.digest(), evaluation_sha256=sha(evaluation),
        background_sha256=sha(background), target='same six latent future coordinates as MLP',
        exact_all_coalition_shapley=True, efficiency_maximum_error=error,
        generalized_shapley_interactions=interactions,
        high_order_interaction=float(sum(v for k,v in interactions.items() if int(k).bit_count()>=3)),
        mean_absolute_shapley_by_source=np.abs(shap).mean((0,2)).tolist(),
        convention='SII weight |T|!(n-|S|-|T|)!/(n-|S|+1)!; pair index is twice conventional SHAP off-diagonal',
        samples=128, background=32, units='latent-output amplitude')


def granger(data):
    import torch
    torch.set_num_threads(1)
    x,y=data['train']; vx,vy=data['test']
    xm,xs=x.mean(0),x.std(0);ym,ys=y.mean(0),y.std(0)
    xt=torch.tensor(((x-xm)/xs).tolist(),dtype=torch.float32)
    result=np.zeros((6,6)); errors=[]
    for j in range(6):
        check();torch.manual_seed(881+j)
        net=torch.nn.Sequential(torch.nn.Linear(6,24),torch.nn.Tanh(),torch.nn.Linear(24,1))
        opt=torch.optim.Adam(net.parameters(),lr=.01)
        yt=torch.tensor(((y[:,j:j+1]-ym[j])/ys[j]).tolist(),dtype=torch.float32)
        for epoch in range(CONFIG['granger_epochs']):
            check(); opt.zero_grad(set_to_none=True)
            loss=((net(xt)-yt)**2).mean()+.01*torch.linalg.vector_norm(net[0].weight,dim=0).sum()
            loss.backward();opt.step()
        result[:,j]=np.linalg.norm(np.array(net[0].weight.detach().tolist()),axis=0)
        with torch.no_grad(): p=np.array(net(torch.tensor(((vx-xm)/xs).tolist(),dtype=torch.float32)).tolist()).ravel()*ys[j]+ym[j]
        errors.append(float(np.mean((vy[:,j]-p)**2)))
    return dict(source_by_target=result.tolist(), cross_mean=float(result[~np.eye(6,dtype=bool)].mean()),
        test_mse_by_target=errors, training_sha256=sha(x,y),
        algorithm='target-specific cMLP, group-L2 first-layer penalty, Part1 implementation',
        epochs=CONFIG['granger_epochs'], group_penalty=.01, units='first-layer weight norm')


def worker(case,lam):
    folder=FOLDER/f'{case}_lambda{lam:g}';folder.mkdir(parents=True,exist_ok=True)
    resultpath=folder/'metrics.json'
    result=json.loads(resultpath.read_text()) if resultpath.exists() else dict(case=case,lambda_value=lam,status='running')
    started=time.time()
    data,states,latent,indices,manifest=natural_data(case,lam,folder)
    model=fit_mlp(data,folder)
    result.update(model_sha256=model.digest(),diagnostics=diagnostics(model,data,case,lam),data_manifest=manifest)
    save(resultpath,result);print(f'{case} lambda={lam:g}: MLP fitted',flush=True)
    for key,fn in [('spt',lambda:spt(intervention_table(model,5)[0])),
                   ('natural_information',lambda:natural_information(data)),
                   ('shap',lambda:shap_interactions(model,data)),
                   ('neural_granger',lambda:granger(data))]:
        if key not in result:
            check(); result[key]=fn();save(resultpath,result)
            print(f'{case} lambda={lam:g}: {key} complete',flush=True)
    if 'quadrature_diagnostics' not in result:
        coarse,cd=intervention_table(model,5);fine,fd=intervention_table(model,6)
        result['quadrature_diagnostics']=dict(coarse=cd,fine=fd,
            maximum_ei_change_nats=float(abs(fine-coarse).max()),
            note='primary order-five and independent order-six rule; both exactly integrate degree-four source Gram matrix')
        result['spt']['ei_table_nats']=coarse.tolist()
    result['shared_mlp_verified']=result['model_sha256']==result['shap']['model_sha256']
    if not result['shared_mlp_verified']:raise ArithmeticError('SPT/SHAP model identity failed')
    result.update(status='core_complete',elapsed_seconds=time.time()-started)
    save(resultpath,result)


def pcmci_compute(case,lam):
    sys.path.insert(0,'/tmp/eisyn_henon_deps')
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.cmiknn import CMIknn
    folder=FOLDER/f'{case}_lambda{lam:g}';data,states,latent,indices,manifest=natural_data(case,lam,folder)
    # Separate chains prevent artificial temporal links across chain boundaries.
    series={int(c):np.vstack([states[:,c],np.tanh(latent[-1,c])]) for c in indices['test']}
    frame=pp.DataFrame(series,analysis_mode='multiple',var_names=[f'x{i+1}' for i in range(6)])
    test=CMIknn(knn=.1,significance='shuffle_test',sig_samples=30,workers=1,seed=20261016,
                sig_blocklength=1,shuffle_neighbors=5)
    engine=PCMCI(dataframe=frame,cond_ind_test=test,verbosity=0)
    check()
    out=engine.run_pcmci(tau_min=1,tau_max=1,pc_alpha=.05,alpha_level=.05,
                         max_conds_dim=2,max_conds_py=2,max_conds_px=2)
    v=out['val_matrix'][:,:,1];p=out['p_matrix'][:,:,1]
    off=~np.eye(6,dtype=bool)
    q=engine.get_corrected_pvalues(p_matrix=out['p_matrix'],tau_min=1,tau_max=1,fdr_method='fdr_bh')[:,:,1]
    save(folder/'pcmci.json',dict(status='complete',source_by_target=v.tolist(),p_values=p.tolist(),q_values=q.tolist(),
        cross_mean=float(abs(v[off]).mean()),significant_cross_count=int((q[off]<.05).sum()),
        data_sha256=manifest['partitions']['test']['sha256'],chains=len(series),pairs=len(data['test'][0]),
        lag=1,knn=.1,conditional_shuffles=30,maximum_conditioning_variables=2,
        family='PCMCI-CMIknn; bounded native state coordinates; information-invariant target transformation in theory',
        limitation='lightweight conditioning cap 2; 30 shuffles have coarse p-value resolution; not hyperedge synergy',
        package='tigramite 5.2.10.1',units='native CMIknn statistic'))


def pcmci_worker(case,lam):
    import fcntl
    folder=FOLDER/f'{case}_lambda{lam:g}';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'pcmci.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (folder/'pcmci.json').exists() and json.loads((folder/'pcmci.json').read_text()).get('status')=='complete':return
        check();pcmci_compute(case,lam)


def parallel_pcmci():
    """Schedule independent late-queue cases once, sharing the absolute budget."""
    budget=json.loads((FOLDER/'budget.json').read_text());processes=[]
    for case,lam in [('B',.5),('B',1.)]:
        folder=FOLDER/f'{case}_lambda{lam:g}';check()
        log=(folder/'pcmci_parallel.log').open('a')
        p=subprocess.Popen([sys.executable,__file__,'--worker','pcmci','--case',case,'--lam',str(lam)],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append((p,log));print(f'PCMCI {case} lambda={lam} pid={p.pid}',flush=True)
    for p,log in processes:
        try:p.wait(timeout=max(.01,budget['deadline_epoch']-time.time()))
        except subprocess.TimeoutExpired:
            for child,_ in processes:
                if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            raise TimeoutError('Shared global deadline exhausted')
        finally:log.close()


def conditional_worker(case,lam,refine=False):
    folder=FOLDER/f'{case}_lambda{lam:g}'
    path=folder/'conditional_tm.json'
    result=json.loads(path.read_text()) if path.exists() else dict(status='running',replicates=[])
    model=MLP(json.loads((folder/'mlp.json').read_text()))
    result['model_sha256']=model.digest()
    outer,inner=1024,256
    if refine:
        initial=(json.loads((folder/'conditional_tm_initial_grid.json').read_text())
                 if 'initial_grid' in result else result)
        if initial.get('status')=='numerically_unresolved':outer=2048
        if 'initial_grid' not in result:
            save(folder/'conditional_tm_initial_grid.json',result)
            result=dict(status='running',replicates=[],initial_grid='conditional_tm_initial_grid.json',
                        model_sha256=model.digest(),
                        reason_for_refinement='Observed inner prefix EI difference >=0.01 nats or significant mean Syn violation')
        inner=512
    for repeat in range(4):
        if len(result['replicates'])>repeat:continue
        table,diag=conditional_tm_table(model,outer,inner,seed=9021+repeat*1000)
        result['replicates'].append(dict(table_nats=table.tolist(),diagnostic=diag))
        save(path,result)
        print(f'{case} lambda={lam:g}: conditional TM scramble {repeat+1}/4',flush=True)
    tables=np.asarray([r['table_nats'] for r in result['replicates']])
    mean=tables.mean(0)
    result.update(ei_table_nats=mean.tolist(),
        maximum_ei_mc_sem_nats=float((tables.std(0,ddof=1)/2).max()),
        inner_prefix_mean_maximum_ei_change_nats=float(abs(mean-np.mean([
            r['diagnostic']['inner_prefix_table_nats'] for r in result['replicates']],axis=0)).max()),
        integrated_ei_nats=float(mean[63]-sum(mean[1<<i] for i in range(6))),
        xi_mc_sem_nats=float(np.std(tables[:,63]-tables[:,[1,2,4,8,16,32]].sum(1),ddof=1)/2))
    result['precision_status']=('inner_prefix_above_0.01_nats' if result['inner_prefix_mean_maximum_ei_change_nats']>=.01 else 'inner_prefix_below_0.01_nats')
    try:
        result['spt']=spt(mean);result['status']='complete'
    except ArithmeticError as error:
        result.update(status='numerically_unresolved',error=str(error))
    save(path,result)


def natural_shift_controls():
    """Small data-reuse diagnostic: destroy lag alignment, preserve chain order."""
    for case in NETWORKS:
        for lam in CONFIG['lambdas']:
            folder=FOLDER/f'{case}_lambda{lam:g}'
            path=folder/'natural_shift_control.json'
            if path.exists():continue
            check();data,states,latent,indices,manifest=natural_data(case,lam,folder)
            tx=data['test'][0]; rng=np.random.default_rng(3716);records=[]
            for repeat in range(4):
                shifts=rng.integers(32,225,size=8)
                shifted=np.stack([np.roll(latent[:,c],int(s),axis=0) for c,s in zip(indices['test'],shifts)],axis=1).reshape(-1,6)
                estimate=natural_information({'test':(tx,shifted)})
                records.append(dict(shifts=shifts.tolist(),wms_nats=estimate['wms_nats'],
                    mmi_grouped_nats=estimate['mmi_grouped_nats'],surd_total_synergy_nats=estimate['surd_total_synergy_nats'],
                    source_sha256=sha(tx),shifted_target_sha256=sha(shifted)))
            save(path,dict(status='complete',replicates=records,
                rule='independent circular shifts 32..224 within each original natural test chain; complete target tuple shifted jointly',
                role='finite-sample/feature-estimator diagnostic; no null subtraction or parameter selection'))
            print(f'{case} lambda={lam:g}: natural shift controls complete',flush=True)


def launch_conditional(refine=False):
    budget=json.loads((FOLDER/'budget.json').read_text())
    save(FOLDER/'conditional_config.json',CONFIG['conditional_tm'])
    for case in NETWORKS:
        for lam in CONFIG['lambdas']:
            folder=FOLDER/f'{case}_lambda{lam:g}'
            path=folder/'conditional_tm.json'
            if path.exists():
                existing=json.loads(path.read_text())
                if refine:
                    if existing.get('status')=='running':continue
                    if existing.get('inner_prefix_mean_maximum_ei_change_nats',0)<.01 and existing.get('status')=='complete':continue
                    if 'initial_grid' in existing:continue
                elif existing.get('status')=='complete':continue
            check();remaining=budget['deadline_epoch']-time.time()
            with (folder/'conditional_tm.log').open('a') as log:
                command=[sys.executable,__file__,'--worker','refine' if refine else 'conditional','--case',case,'--lam',str(lam)]
                p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=p.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5);raise TimeoutError('Global absolute budget exhausted')
            if code: print(f'{case} lambda={lam:g}: conditional worker exit={code}',flush=True)


def launch():
    FOLDER.mkdir(parents=True,exist_ok=True)
    budgetpath=FOLDER/'budget.json'
    if not budgetpath.exists():
        # Task starts at approval, not at each worker restart.
        start=1792147756.0
        save(budgetpath,dict(started_epoch=start,deadline_epoch=start+2700,budget_seconds=2700,
                            authorization='User approved two networks x three lambdas x seed0, 45 minutes'))
    save(FOLDER/'config.json',CONFIG)
    budget=json.loads(budgetpath.read_text())
    jobs=[(kind,c,l) for kind in ('core','pcmci') for c in NETWORKS for l in CONFIG['lambdas']]
    ran_any=False
    for kind,case,lam in jobs:
        folder=FOLDER/f'{case}_lambda{lam:g}'
        if kind=='core' and (folder/'metrics.json').exists() and json.loads((folder/'metrics.json').read_text()).get('status')=='core_complete':continue
        if kind=='pcmci' and (folder/'pcmci.json').exists() and json.loads((folder/'pcmci.json').read_text()).get('status')=='complete':continue
        remaining=budget['deadline_epoch']-time.time()
        if remaining<=0:break
        folder.mkdir(parents=True,exist_ok=True)
        logpath=folder/f'{kind}.log'
        with logpath.open('a') as log:
            ran_any=True
            p=subprocess.Popen([sys.executable,__file__,'--worker',kind,'--case',case,'--lam',str(lam)],
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print(json.dumps(dict(stage=kind,case=case,lam=lam,pid=p.pid)),flush=True)
            try:code=p.wait(timeout=min(remaining,600 if kind=='pcmci' else remaining))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                save(folder/f'{kind}_failure.json',dict(status='budget_exhausted',stage=kind))
                continue
        if code:save(folder/f'{kind}_failure.json',dict(status='failed',exit_code=code,log=str(logpath)))
    if ran_any:
        save(FOLDER/'run_complete.json',dict(finished_epoch=time.time(),elapsed_seconds=time.time()-budget['started_epoch']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['core','pcmci','conditional','refine']);p.add_argument('--conditional',action='store_true');p.add_argument('--refine-conditional',action='store_true');p.add_argument('--shift-controls',action='store_true');p.add_argument('--parallel-pcmci',action='store_true');p.add_argument('--case',choices=['A','B']);p.add_argument('--lam',type=float)
    a=p.parse_args()
    if a.worker=='core':worker(a.case,a.lam)
    elif a.worker=='pcmci':pcmci_worker(a.case,a.lam)
    elif a.worker=='conditional':conditional_worker(a.case,a.lam)
    elif a.worker=='refine':conditional_worker(a.case,a.lam,True)
    elif a.conditional:launch_conditional()
    elif a.refine_conditional:launch_conditional(True)
    elif a.shift_controls:natural_shift_controls()
    elif a.parallel_pcmci:parallel_pcmci()
    else:launch()
