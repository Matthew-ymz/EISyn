"""Approved fit-only Kuramoto A comparison, bounded by a persisted 30-minute budget."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_kuramoto_finite_tm import save_json, digest, transition_with_increments
from scripts.spt_kuramoto_natural_mlp import NaturalMLPConditionalTM
from scripts.spt_kuramoto_pair_mlp import PairMLPConditionalTM, PAIRS
from scripts.spt_kuramoto_simplified_ac import SimplifiedKuramoto
from scripts.spt_local_hyperedge_kuramoto import wrap

FOLDER = ROOT/'results/spt_kuramoto_A_fit_improvement'


def read(path):
    return json.loads(Path(path).read_text())


def emit(stage, **items):
    print(json.dumps(dict(stage=stage, **items)), flush=True)


def partitions(states, seed=20261016):
    if states.shape != (5,16384,9) or not np.isfinite(states).all():
        raise ValueError('Registered shape is (5,16384,9)')
    order = np.random.default_rng(seed).permutation(16384)
    groups = dict(train=order[:12288], validation=order[12288:14336], test=order[14336:])
    data, manifest = {}, dict(seed=seed, split_unit='independent trajectory',
        pair_row_id='age * 16384 + trajectory_id', splits={},
        shared_by=['SPT MLP', 'SHAP same frozen MLP', 'WMS', 'SURD'],
        baseline_rerun_status='not run in this fit-only experiment')
    for name, idx in groups.items():
        x, y = states[:-1,idx].reshape(-1,9).copy(), states[1:,idx].reshape(-1,9).copy()
        data[name] = x,y
        manifest['splits'][name] = dict(trajectories=idx.tolist(), pairs=len(x), paired_arrays_sha256=digest(x,y))
    return data, manifest


def observed(model, data):
    x,y = data
    residual = wrap(y-model.mean(x))
    return dict(rmse_rad=float(np.sqrt(np.mean(residual**2))),
        mae_rad=float(np.mean(abs(residual))),
        by_age_rmse_rad=[float(np.sqrt(np.mean(r*r))) for r in residual.reshape(4,-1,9)],
        persistence_rmse_rad=float(np.sqrt(np.mean(wrap(y-x)**2))), paired_arrays_sha256=digest(x,y))


def reference(generator, x, rng, repeats=256):
    sample = generator.step(np.repeat(x,repeats,axis=0),rng,dt=.0025).reshape(len(x),repeats,9)
    mu = np.angle(np.mean(np.exp(1j*sample),axis=1))
    residual = wrap(sample-mu[:,None])
    variance = np.mean(residual**2,axis=1)*repeats/(repeats-1)
    return mu, variance/repeats


def evaluate_means(generator, models, test_x, stage, seed, *, folder=FOLDER):
    rng = np.random.default_rng(seed)
    domains = {'uniform_independent':rng.uniform(-np.pi,np.pi,(128,9)),
               'natural_test':test_x[rng.choice(len(test_x),128,replace=False)]}
    result = {}
    cache = {}
    for domain,x in domains.items():
        mu, variance = reference(generator,x,rng)
        signal_rms = float(np.sqrt(np.mean(wrap(mu-x)**2)))
        result[domain] = dict(probes=len(x), repeats=256, reference_mc_noise_rms_rad=float(np.sqrt(variance.mean())),
            signal_rms_rad=signal_rms, models={})
        for name,model in models.items():
            error = wrap(model.mean(x)-mu)
            rmse = float(np.sqrt(np.mean(error**2)))
            result[domain]['models'][name] = dict(rmse_rad=rmse, nrmse=rmse/signal_rms,
                engineering_mean_gate=rmse/signal_rms <= .05,
                reference_mc_noise_below_half_target=bool(np.sqrt(variance.mean()) < .025*signal_rms))
        cache[domain+'_source'],cache[domain+'_reference'],cache[domain+'_mc_variance'] = x,mu,variance
    np.savez_compressed(folder/(stage+'_reference.npz'),**cache)
    return result


def sensitivities(generator, models, seed=20261103, *, folder=FOLDER):
    rng = np.random.default_rng(seed)
    n,r,h = 16,64,.001
    x = rng.uniform(-np.pi,np.pi,(n,9))
    jacobian_source = x.copy()
    probes = np.broadcast_to(x[:,None,None,:],(n,9,2,9)).copy()
    for j in range(9):
        probes[:,j,0,j] += h
        probes[:,j,1,j] -= h
    source = np.repeat(probes[:,:,:,None,:],r,axis=3)
    noise = np.sqrt(.0025)*rng.normal(size=(80,n,1,1,r,9))
    increments = np.broadcast_to(noise,(80,n,9,2,r,9)).reshape(80,-1,9)
    future = transition_with_increments(generator,source.reshape(-1,9),increments,.0025).reshape(n,9,2,r,9)
    means = np.angle(np.mean(np.exp(1j*future),axis=3))
    reference_j = (wrap(means[:,:,0]-means[:,:,1])/(2*h)).transpose(0,2,1)-np.eye(9)[None]
    answer = dict(probes=n,repeats=r,step_rad=h,comparison='increment Jacobian (full future Jacobian minus identity)',
                  reference='common Brownian increments finite difference of MC circular means',models={})
    for name,model in models.items():
        prediction = model.mean(x,jacobian=True)[1]-np.eye(9)[None]
        error = prediction-reference_j
        rel = float(np.linalg.norm(error)/np.linalg.norm(reference_j))
        answer['models'][name] = dict(relative_frobenius_error=rel,rmse=float(np.sqrt(np.mean(error**2))),engineering_gate=rel<=.10)
    # Every input pair, without selecting planted interactions.
    n,r,h = 8,16,.05
    x = rng.uniform(-np.pi,np.pi,(n,9))
    corners = [(1,1),(1,-1),(-1,1),(-1,-1)]
    source = np.broadcast_to(x[:,None,None,:],(n,36,4,9)).copy()
    for p,(i,j) in enumerate(PAIRS):
        for c,(si,sj) in enumerate(corners):
            source[:,p,c,i] += si*h
            source[:,p,c,j] += sj*h
    repeated = np.repeat(source[:,:,:,None,:],r,axis=3)
    noise = np.sqrt(.0025)*rng.normal(size=(80,n,36,1,r,9))
    increments = np.broadcast_to(noise,(80,n,36,4,r,9)).reshape(80,-1,9)
    future = transition_with_increments(generator,repeated.reshape(-1,9),increments,.0025).reshape(n,36,4,r,9)
    mean = np.angle(np.mean(np.exp(1j*future),axis=3))
    contrast = lambda a:(wrap(a[:,:,0]-a[:,:,1])-wrap(a[:,:,2]-a[:,:,3]))/(4*h*h)
    ref = contrast(mean)
    joint = dict(probes=n,repeats=r,step_rad=h,pairs=36,reference_rms=float(np.sqrt(np.mean(ref**2))),models={})
    for name,model in models.items():
        pred = contrast(model.mean(source.reshape(-1,9)).reshape(n,36,4,9))
        joint['models'][name] = dict(relative_frobenius_error=float(np.linalg.norm(pred-ref)/np.linalg.norm(ref)),
                                     rmse=float(np.sqrt(np.mean((pred-ref)**2))))
    np.savez_compressed(folder/'sensitivity_references.npz',jacobian_source=jacobian_source,
                        increment_jacobian=reference_j,joint_source=source,joint_reference=ref)
    return dict(jacobian=answer,joint_response=joint)


def residual_diagnostics(model, validation, test):
    vx,vy = validation
    tx,ty = test
    vr,tr = wrap(vy-model.mean(vx)),wrap(ty-model.mean(tx))
    center,cov = vr.mean(0),np.cov(vr,rowvar=False)
    if np.linalg.eigvalsh(cov).min() <= 0:
        raise ArithmeticError('Residual covariance not positive definite; no silent floor')
    corr = np.corrcoef(vr,rowvar=False)
    d2 = np.einsum('ni,ij,nj->n',tr-center,np.linalg.inv(cov),tr-center)
    return dict(calibration='natural validation only; full covariance retained separately',
        covariance_rad2=cov.tolist(),center_rad=center.tolist(),minimum_eigenvalue=float(np.linalg.eigvalsh(cov).min()),
        maximum_abs_offdiagonal_correlation=float(np.max(abs(corr-np.eye(9)))),
        local_gaussian_joint_95_test_coverage=float(np.mean(d2<=chi2.ppf(.95,9))),
        coverage_approximation='small-noise tangent Gaussian; not exact joint circular conditional density',
        density_status='model adapter remains independent constant von Mises; full conditional density not validated')


def worker():
    budget = read(FOLDER/'budget.json')
    deadline = time.monotonic()+budget['deadline_epoch']-time.time()
    if time.monotonic() >= deadline-120:
        raise TimeoutError('Persisted budget exhausted')
    protocol = dict(scope=budget['scope'],seed=20261016,shape=[5,16384,9],horizon=.2,dt=.0025,noise=.08,
        data='iid Uniform(-pi,pi)^9 initial states, four unforced natural transitions, no burn-in, no within-trajectory interventions',
        model_selection='common natural validation circular RMSE only; no oracle/F1/edge labels',
        models=['unchanged 18-64-64-9 tanh fitter','all36 pair k1/2 144-128-128-9 standardized SiLU residual fitter'],
        ei_spt_f1_computed=False,syn_consumed=False,scope_replication='one natural dataset and initialization; no statistical inference',
        manuscript=dict(parent='P6UJCVG8',main='DXGC7JEA',si='MWIWKSVG',checked='2026-10-16',
            version='unique current attachment pair; no explicit manuscript date/version, ambiguity unresolved',
            locations='Methods(10)-(12), SI S1.2 and S7',difference='nine-node natural trajectories and full future target; not six-node SI intervention training'))
    save_json(FOLDER/'protocol.json',protocol)
    generator = SimplifiedKuramoto('A')
    pool = FOLDER/'natural_transients.npz'
    emit('generate_natural_data')
    if pool.exists():
        with np.load(pool,allow_pickle=False) as z:
            states = z['states'].copy()
    else:
        rng = np.random.default_rng(20261016)
        states = [rng.uniform(-np.pi,np.pi,(16384,9))]
        for age in range(4):
            states.append(generator.step(states[-1],rng,dt=.0025))
        states = np.stack(states)
        np.savez_compressed(pool,states=states,metadata=json.dumps(protocol))
    data,manifest = partitions(states)
    manifest.update(pool_sha256=hashlib.sha256(pool.read_bytes()).hexdigest(),pool_path=str(pool))
    save_json(FOLDER/'shared_data.json',manifest)
    summary = dict(status='running',protocol=protocol,models={})
    models = {}
    for name,cls in [('baseline',NaturalMLPConditionalTM),('periodic',PairMLPConditionalTM)]:
        directory = FOLDER/name
        directory.mkdir(exist_ok=True)
        emit('fit',model=name)
        if (directory/'model.json').exists():
            model,fit = cls.load(read(directory/'model.json')),read(directory/'fit.json')
        else:
            model,fit = cls.fit(*data['train'],*data['validation'],deadline=deadline,
                progress=lambda a:emit('training',model=name,**a))
            save_json(directory/'model.json',model.payload())
            fit['common_training_data_sha256'] = manifest['splits']['train']['paired_arrays_sha256']
            save_json(directory/'fit.json',fit)
        models[name] = model
        summary['models'][name] = dict(model_sha256=model.sha256(),fit=fit,
            natural_validation=observed(model,data['validation']),natural_test=observed(model,data['test']),
            residual_diagnostics=residual_diagnostics(model,data['validation'],data['test']))
        save_json(FOLDER/'summary.json',summary)
    selected = min(models,key=lambda k:summary['models'][k]['natural_validation']['rmse_rad'])
    summary['selected_by_natural_validation'] = selected
    save_json(FOLDER/'selection_frozen_before_oracle.json',dict(selected=selected,
        model_sha256={k:m.sha256() for k,m in models.items()},selection='natural validation circular RMSE only',timestamp=time.time()))
    historical = ROOT/'results/spt_natural_mlp_pilot/kuramoto_A/model.json'
    if historical.exists():
        models['historical_stationary'] = NaturalMLPConditionalTM.load(read(historical))
    emit('independent_fit_diagnostics',selected=selected)
    summary['engineering_conditional_mean'] = evaluate_means(generator,models,data['test'][0],'engineering',20261101)
    save_json(FOLDER/'summary.json',summary)
    summary['sensitivity'] = sensitivities(generator,models)
    save_json(FOLDER/'summary.json',summary)
    summary['independent_final_conditional_mean'] = evaluate_means(generator,models,data['test'][0],'final',20261102)
    summary.update(status='complete',finished_epoch=time.time(),elapsed_seconds=time.time()-budget['started_epoch'],
                   fitted_models_frozen_before_oracle=True,ei_spt_f1_computed=False)
    save_json(FOLDER/'summary.json',summary)
    emit('complete',selected=selected,elapsed_seconds=summary['elapsed_seconds'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker()
        return
    summary_path = FOLDER/'summary.json'
    if summary_path.exists() and read(summary_path).get('status') == 'complete':
        emit('reused_completed_experiment')
        return
    budget = read(FOLDER/'budget.json')
    if time.time() >= budget['deadline_epoch']:
        raise TimeoutError('Persisted absolute budget exhausted; cannot be reset')
    with (FOLDER/'run.log').open('a',buffering=1) as log:
        process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        save_json(FOLDER/'process.json',dict(parent_pid=os.getpid(),worker_pid=process.pid,deadline_epoch=budget['deadline_epoch']))
        try:
            code = process.wait(timeout=max(0,budget['deadline_epoch']-time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
            save_json(FOLDER/'watchdog.json',dict(status='deadline_stopped',deadline_epoch=budget['deadline_epoch']))
            raise SystemExit(124)
        raise SystemExit(code)


if __name__ == '__main__':
    main()
