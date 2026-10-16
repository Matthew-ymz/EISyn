"""Cheap frozen-model audits; never fits, changes, or selects a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.run_spt_kuramoto_A_fit_improvement import FOLDER, read, reference
from scripts.spt_kuramoto_finite_tm import PeriodicConditionalTM, save_json
from scripts.spt_kuramoto_simplified_ac import SimplifiedKuramoto
from scripts.spt_local_hyperedge_kuramoto import wrap


def main():
    if time.time() >= read(FOLDER/'budget.json')['deadline_epoch']:
        raise TimeoutError('Approved absolute budget expired')
    summary = read(FOLDER/'summary.json')
    assert summary['status'] == 'complete'
    models = {k:PeriodicConditionalTM.load(read(FOLDER/k/'model.json')) for k in ['baseline','periodic']}
    models['historical_stationary'] = PeriodicConditionalTM.load(read(ROOT/'results/spt_natural_mlp_pilot/kuramoto_A/model.json'))
    frozen = read(FOLDER/'selection_frozen_before_oracle.json')
    assert all(m.sha256() == frozen['model_sha256'][k] for k,m in models.items() if k in frozen['model_sha256'])
    generator = SimplifiedKuramoto('A')
    old_network = read(ROOT/'results/spt_local_kuramoto_tm/ac_comparison/A/frozen_model.json')['network']
    expected = {tuple(v['edge']):v['weight'] for v in old_network}
    assert expected == generator.weights
    output = dict(training_or_selection_performed=False,network_identical_to_previous_A=True,models={})
    rng = np.random.default_rng(20261112)
    # A fresh final natural cohort, independent of both the master pool and engineering probes.
    states = [rng.uniform(-np.pi,np.pi,(128,9))]
    for age in range(3):
        states.append(generator.step(states[-1],rng,dt=.0025))
    source = np.concatenate([s[age*32:(age+1)*32] for age,s in enumerate(states)])
    mu,mcvar = reference(generator,source,rng)
    signal = float(np.sqrt(np.mean(wrap(mu-source)**2)))
    final = dict(probes=128,repeats=256,seed=20261112,independent_trajectory_cohort=True,
        reference_mc_noise_rms_rad=float(np.sqrt(mcvar.mean())),signal_rms_rad=signal,models={})
    for name,model in models.items():
        rmse = float(np.sqrt(np.mean(wrap(model.mean(source)-mu)**2)))
        final['models'][name] = dict(rmse_rad=rmse,nrmse=rmse/signal)
    output['independent_final_natural_cohort'] = final
    np.savez_compressed(FOLDER/'independent_final_natural_reference.npz',source=source,mean=mu,mc_variance=mcvar)
    # Mean-map rollouts: deterministic repeated fitted mean vs Monte Carlo true trajectory means.
    # These differ from exact stochastic-model predictive means after the first step (Jensen effect).
    rng = np.random.default_rng(20261113)
    initial = rng.uniform(-np.pi,np.pi,(64,9))
    trajectories = np.repeat(initial[:,None,:],256,axis=1)
    predicted = {k:initial.copy() for k in models}
    rollout = []
    for step in range(1,5):
        trajectories = generator.step(trajectories.reshape(-1,9),rng,dt=.0025).reshape(64,256,9)
        mean = np.angle(np.mean(np.exp(1j*trajectories),axis=1))
        row = dict(step=step,time=step*.2,models={})
        for name,model in models.items():
            predicted[name] = wrap(model.mean(predicted[name]))
            row['models'][name] = dict(rmse_rad=float(np.sqrt(np.mean(wrap(predicted[name]-mean)**2))))
        rollout.append(row)
    output['deterministic_mean_map_rollout'] = dict(probes=64,repeats=256,seed=20261113,
        limitation='after step1, iterated mean map need not equal exact stochastic predictive mean',rows=rollout)
    with np.load(FOLDER/'final_reference.npz',allow_pickle=False) as z:
        local_var = z['uniform_independent_mc_variance']*256
    output['state_dependent_variance'] = dict(reference='MC conditional variance, independent uniform final states',
        approximation='small circular residual tangent variance',p10=np.quantile(local_var,.1,axis=0).tolist(),
        p90=np.quantile(local_var,.9,axis=0).tolist(),p90_p10_ratio=(np.quantile(local_var,.9,axis=0)/np.quantile(local_var,.1,axis=0)).tolist())
    with np.load(FOLDER/'natural_transients.npz',allow_pickle=False) as z:
        states = z['states']
    manifest = read(FOLDER/'shared_data.json')
    for name,model in models.items():
        output['models'][name] = {}
        for split in ['train','validation']:
            idx = manifest['splits'][split]['trajectories']
            x,y = states[:-1,idx].reshape(-1,9),states[1:,idx].reshape(-1,9)
            error = wrap(y-model.mean(x))
            output['models'][name][split+'_rmse_rad'] = float(np.sqrt(np.mean(error**2)))
    # Repair only the reference-source metadata of a worker already loaded before this one-line fix.
    path = FOLDER/'sensitivity_references.npz'
    with np.load(path,allow_pickle=False) as z:
        cached = {k:z[k].copy() for k in z.files}
    expected_source = np.random.default_rng(20261103).uniform(-np.pi,np.pi,(16,9))
    cached['jacobian_source'] = expected_source
    np.savez_compressed(path,**cached)
    output['source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),ROOT/'scripts/spt_kuramoto_pair_mlp.py',ROOT/'scripts/run_spt_kuramoto_A_fit_improvement.py',
                  ROOT/'scripts/spt_kuramoto_natural_mlp.py',ROOT/'scripts/spt_local_hyperedge_kuramoto.py',ROOT/'scripts/spt_kuramoto_simplified_ac.py']}
    output['finished_epoch'] = time.time()
    output['total_elapsed_seconds'] = time.time()-read(FOLDER/'budget.json')['started_epoch']
    save_json(FOLDER/'additional_diagnostics.json',output)
    print(json.dumps(dict(status='complete',total_elapsed_seconds=output['total_elapsed_seconds'])))


if __name__ == '__main__':
    main()
