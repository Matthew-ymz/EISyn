"""Approved 4x natural-data extension; fixed periodic MLP, no EI/SPT/F1."""
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.spt_kuramoto_finite_tm import save_json, digest
from scripts.spt_kuramoto_pair_mlp import PairMLPConditionalTM
from scripts.spt_kuramoto_simplified_ac import SimplifiedKuramoto
from scripts.spt_local_hyperedge_kuramoto import wrap
from scripts.run_spt_kuramoto_A_fit_improvement import (
    observed, evaluate_means, sensitivities, residual_diagnostics, read,
)

FOLDER = ROOT/'results/spt_kuramoto_A_sample_scaling'
PREVIOUS = ROOT/'results/spt_kuramoto_A_fit_improvement'


def emit(stage,**items):
    print(json.dumps(dict(stage=stage,**items)),flush=True)


def nested_partitions(states,previous_manifest,seed=20261017):
    """Extend every old partition separately: old held-out trajectories stay held out."""
    states = np.asarray(states,float)
    groups = {k:np.asarray(v['trajectories'],int) for k,v in previous_manifest['splits'].items()}
    old_ids = np.concatenate(list(groups.values()))
    old_n = len(old_ids)
    if states.shape != (5,4*old_n,9) or not np.isfinite(states).all():
        raise ValueError('Expected exactly 4x the previous independent trajectory count')
    if len(np.unique(old_ids)) != old_n or set(old_ids) != set(range(old_n)):
        raise ValueError('Previous partitions must cover disjoint original trajectory IDs')
    extra_n = 3*old_n
    order = np.random.default_rng(seed).permutation(extra_n)+old_n
    extra = dict(train=order[:extra_n*3//4],validation=order[extra_n*3//4:extra_n*7//8],
                 test=order[extra_n*7//8:])
    data,manifest = {},dict(seed=seed,split_unit='independent trajectory',splits={},
        pair_row_id=f'age * {4*old_n} + trajectory_id',
        old_trajectory_prefix=old_n,old_split_assignments_preserved=True,
        shared_by=['SPT MLP','SHAP same frozen MLP','WMS','SURD'],
        other_method_reruns='not performed in this fit-only study')
    for name in ['train','validation','test']:
        idx = np.concatenate([groups[name],extra[name]])
        x,y = states[:-1,idx].reshape(-1,9).copy(),states[1:,idx].reshape(-1,9).copy()
        data[name] = x,y
        manifest['splits'][name] = dict(trajectories=idx.tolist(),pairs=len(x),
            original_trajectories=groups[name].tolist(),paired_arrays_sha256=digest(x,y))
    return data,manifest


def fresh_natural_final(generator,models,seed=20261121):
    from scripts.run_spt_kuramoto_A_fit_improvement import reference
    rng = np.random.default_rng(seed)
    states = [rng.uniform(-np.pi,np.pi,(128,9))]
    for age in range(3):
        states.append(generator.step(states[-1],rng,dt=.0025))
    x = np.concatenate([s[age*32:(age+1)*32] for age,s in enumerate(states)])
    mu,mcvar = reference(generator,x,rng)
    signal = float(np.sqrt(np.mean(wrap(mu-x)**2)))
    result = dict(seed=seed,probes=128,repeats=256,independent_of_training_validation_test_and_prior_probes=True,
        signal_rms_rad=signal,reference_mc_noise_rms_rad=float(np.sqrt(mcvar.mean())),models={})
    for name,model in models.items():
        rmse = float(np.sqrt(np.mean(wrap(model.mean(x)-mu)**2)))
        result['models'][name] = dict(rmse_rad=rmse,nrmse=rmse/signal,engineering_mean_gate=rmse/signal<=.05)
    np.savez_compressed(FOLDER/'independent_natural_reference.npz',source=x,mean=mu,mc_variance=mcvar)
    return result


def worker():
    budget = read(FOLDER/'budget.json')
    end = time.monotonic()+budget['deadline_epoch']-time.time()
    if time.monotonic() >= end-120:
        raise TimeoutError('Insufficient remaining absolute budget')
    generator = SimplifiedKuramoto('A')
    expected = {tuple(v['edge']):v['weight'] for v in read(ROOT/'results/spt_local_kuramoto_tm/ac_comparison/A/frozen_model.json')['network']}
    assert generator.weights == expected
    previous_manifest = read(PREVIOUS/'shared_data.json')
    previous_path = PREVIOUS/'natural_transients.npz'
    assert hashlib.sha256(previous_path.read_bytes()).hexdigest() == previous_manifest['pool_sha256']
    protocol = dict(scope=budget['scope'],factor='shared natural trajectory count, 16384 vs65536 (4 steps each)',
        pair_counts=[65536,262144],source='same nine phases',target='complete nine future phases after adding predicted wrapped increment',
        horizon=.2,dt=.0025,noise=.08,network=read(ROOT/'results/spt_local_kuramoto_tm/ac_comparison/A/frozen_model.json')['network'],
        data='independent Uniform(-pi,pi)^9 initial states; four unforced natural transitions; retain transients',
        new_data_seed=20261017,fit_seed=20261016,warm_start=False,
        architecture=[144,128,128,9],features='all36 phase-difference pairs k1/2 sin/cos',activation='SiLU',
        fitting='unchanged PairMLPConditionalTM.fit; train-only standardization; AdamW .003 wd1e-5 batch1024; cosine800; minimum200/patience100',
        budget_comparison='same epoch/stopping recipe, not equal gradient updates or equal compute; 4x rows imply more updates per epoch',
        selection='natural validation only; all models frozen before oracle diagnostics; no labels/F1',
        replication='one nested dataset pair, one initialization; descriptive sample-scaling comparison',
        gates=dict(conditional_mean_relative_rmse=.05,increment_jacobian_relative_frobenius=.10),
        ei_spt_f1_computed=False,syn_consumed=False,
        manuscript=dict(parent='P6UJCVG8',main='DXGC7JEA',si='MWIWKSVG',checked='2026-10-16',
            version='unique attachments, no explicit manuscript date/version; item revision is not manuscript version',
            locations='Methods(10)-(12), SI S1.2(S7)-(S15), S7(S67)-(S71)',
            differences='nine-node natural trajectories/full future vs SI six-node interventions/increments; product constant von Mises residual remains approximation'))
    source_paths = [Path(__file__),ROOT/'scripts/spt_kuramoto_pair_mlp.py',ROOT/'scripts/run_spt_kuramoto_A_fit_improvement.py',
                    ROOT/'scripts/spt_local_hyperedge_kuramoto.py',ROOT/'scripts/spt_kuramoto_simplified_ac.py']
    protocol['source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    save_json(FOLDER/'protocol.json',protocol)
    emit('extend_shared_natural_pool')
    with np.load(previous_path,allow_pickle=False) as z:
        original = z['states'].copy()
    pool = FOLDER/'natural_transients.npz'
    if pool.exists():
        with np.load(pool,allow_pickle=False) as z:
            states = z['states'].copy()
    else:
        rng = np.random.default_rng(20261017)
        extra = [rng.uniform(-np.pi,np.pi,(49152,9))]
        for age in range(4):
            extra.append(generator.step(extra[-1],rng,dt=.0025))
        states = np.concatenate([original,np.stack(extra)],axis=1)
        np.savez_compressed(pool,states=states,metadata=json.dumps(protocol))
    np.testing.assert_array_equal(states[:,:16384],original)
    data,manifest = nested_partitions(states,previous_manifest)
    manifest.update(pool_path=str(pool),pool_sha256=hashlib.sha256(pool.read_bytes()).hexdigest(),
        previous_pool_sha256=previous_manifest['pool_sha256'])
    save_json(FOLDER/'shared_data.json',manifest)
    assert [len(data[k][0]) for k in ['train','validation','test']] == [196608,32768,32768]
    models = {'pairs65536':PairMLPConditionalTM.load(read(PREVIOUS/'periodic/model.json'))}
    emit('fit',model='pairs262144',train_pairs=len(data['train'][0]))
    directory = FOLDER/'periodic'
    directory.mkdir(exist_ok=True)
    if (directory/'model.json').exists():
        model,fit = PairMLPConditionalTM.load(read(directory/'model.json')),read(directory/'fit.json')
        assert fit['training_data_sha256'] == manifest['splits']['train']['paired_arrays_sha256']
    else:
        model,fit = PairMLPConditionalTM.fit(*data['train'],*data['validation'],deadline=end,
            progress=lambda a:emit('training',**a))
        fit.update(training_data_sha256=manifest['splits']['train']['paired_arrays_sha256'],
                   validation_data_sha256=manifest['splits']['validation']['paired_arrays_sha256'])
        save_json(directory/'model.json',model.payload())
        save_json(directory/'fit.json',fit)
    models['pairs262144'] = model
    summary = dict(status='running',protocol=protocol,fit=fit,models={})
    for name,model in models.items():
        summary['models'][name] = dict(model_sha256=model.sha256(),natural_validation=observed(model,data['validation']),
            natural_test=observed(model,data['test']),residual_diagnostics=residual_diagnostics(model,data['validation'],data['test']))
    selected = min(models,key=lambda k:summary['models'][k]['natural_validation']['rmse_rad'])
    summary['selected_by_natural_validation'] = selected
    save_json(FOLDER/'selection_frozen_before_oracle.json',dict(selected=selected,model_sha256={k:m.sha256() for k,m in models.items()},
        timestamp=time.time(),rule='common expanded natural validation circular RMSE only'))
    save_json(FOLDER/'summary.json',summary)
    emit('frozen_fit_diagnostics',selected=selected)
    # Prior final probes are now paired engineering diagnostics, never called untouched tests.
    with np.load(PREVIOUS/'final_reference.npz',allow_pickle=False) as z:
        x,mu = z['uniform_independent_source'],z['uniform_independent_reference']
    signal = float(np.sqrt(np.mean(wrap(mu-x)**2)))
    summary['previous_viewed_paired_uniform_diagnostic'] = dict(reference='previous viewed final_reference.npz; not new untouched test',models={})
    for name,model in models.items():
        error = float(np.sqrt(np.mean(wrap(model.mean(x)-mu)**2)))
        summary['previous_viewed_paired_uniform_diagnostic']['models'][name] = dict(rmse_rad=error,nrmse=error/signal)
    summary['fresh_final_conditional_mean'] = evaluate_means(generator,models,data['test'][0],'fresh_final',20261120,folder=FOLDER)
    save_json(FOLDER/'summary.json',summary)
    summary['sensitivity'] = sensitivities(generator,models,seed=20261122,folder=FOLDER)
    save_json(FOLDER/'summary.json',summary)
    summary['independent_natural_final'] = fresh_natural_final(generator,models)
    summary.update(status='complete',finished_epoch=time.time(),elapsed_seconds=time.time()-budget['started_epoch'],
                   ei_spt_f1_computed=False,training_only_natural=True)
    save_json(FOLDER/'summary.json',summary)
    emit('complete',selected=selected,elapsed_seconds=summary['elapsed_seconds'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker()
        return
    summary = FOLDER/'summary.json'
    if summary.exists() and read(summary).get('status') == 'complete':
        emit('reuse_complete')
        return
    budget = read(FOLDER/'budget.json')
    if time.time() >= budget['deadline_epoch']:
        raise TimeoutError('Absolute approved budget expired; cannot reset')
    with (FOLDER/'run.log').open('a',buffering=1) as log:
        process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
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
