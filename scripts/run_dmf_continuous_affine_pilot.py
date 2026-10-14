#!/usr/bin/env python3
"""Bounded paired mean-SC pilot; does not replace production DMF readouts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dmf_continuous_affine import VERSION, ContinuousAffine, fit_continuous_affine, sample_moments
from scripts.dmf_joint_readout import CommonTargetGame, audit_nonnegative
from scripts.dmf_response_benchmark import fit_affine_joint, fit_dmf_affine_joint, simulate_sources
from scripts.run_dmf_paired_spt_pilot import paired_sources, noise_seed
from scripts.run_dmf_subject_consistency import atomic_json, digest
from scripts.run_dmf_subject_curve_baselines import atomic_savez
from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module

BASE = ROOT/'results/dmf_schaefer100/continuous_affine_pilot_v1'
SOURCE = ROOT/'results/dmf_schaefer100/source/group_mean_native_mean_rate.npz'
G_VALUES = [0., 1e-6, .001, .01, .1, .5, 1.3, 3.]
SELECTION_G = [.01, .1, .5, 1.3]
PENALTIES = [.001, .01, .1, 1., 10.]
TOL_NATS = 1e-8


def array_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def response(dmf, p, sc, jf, g, seed, split, protocol_sha):
    offset, count = {'train':(0, 2048), 'validation':(100, 512), 'test':(200, 512)}[split]
    data_seed = seed+offset
    x = np.concatenate(paired_sources(data_seed, count, 100), axis=1)
    path = BASE/'responses'/f'seed{seed}_{split}_G{g:.8g}.npz'
    input_sha = array_digest(x)
    if path.exists():
        with np.load(path) as a:
            if str(a['protocol_sha256']) != protocol_sha or str(a['input_sha256']) != input_sha:
                raise ValueError(f'Response cache provenance changed: {path}')
            y = a['y'].copy(); diagnostics = json.loads(str(a['diagnostics_json']))
        return x, y, diagnostics
    start = time.perf_counter()
    y, diagnostics = simulate_sources(dmf, x, sc, jf, g, p, noise_seed(data_seed))
    if diagnostics['outside_state_count'] or diagnostics['abnormal_rate_count']:
        raise ArithmeticError(f'Native response violates bounds/rates: {g}, {seed}, {split}, {diagnostics}')
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(path, y=y, protocol_sha256=protocol_sha, input_sha256=input_sha,
        target_sha256=array_digest(y), noise_seed=noise_seed(data_seed), G=g,
        seed=seed, split=split, diagnostics_json=json.dumps(diagnostics))
    print(f'Response G={g:g}, seed={seed}, {split}: {time.perf_counter()-start:.1f}s', flush=True)
    return x, y, diagnostics


def baseline(moments, x, y, sc, g, *, unrestricted=False):
    cov = (fit_affine_joint(x, y, .2) if unrestricted else
           fit_dmf_affine_joint(x, y, .2, connectivity=sc, coupling=g))
    d = len(moments.xx); prior = cov[:d, :d]
    b = cov[:d, d:] / np.diag(prior)[:, None]
    r = cov[d:, d:] - b.T@prior@b; r = .5*(r+r.T)
    return ContinuousAffine(cov, b, r, np.linalg.cholesky(r), moments, g, 0., 0., {})


def describe(model, groups, test):
    game = CommonTargetGame(model.covariance)
    values = game.totals()
    between = values['whole_ei_nats']-sum(game.ei(group) for group in groups)
    values['between_network_nats'] = float(between)
    nonnegative = audit_nonnegative([*game.audited, *values.values()], TOL_NATS)
    return dict(**values, test_scores=model.scores(*test), syn_audit=nonnegative,
                parameter_diagnostics=model.diagnostics)


def run(seeds):
    started = time.perf_counter(); BASE.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE) as a:
        sc, jf = a['connectivity'].copy(), a['j_fic'][0].copy()
        np.testing.assert_array_equal(a['j_fic'], np.broadcast_to(jf, a['j_fic'].shape))
    with np.load(ROOT/'results/dmf_schaefer100/main_independent_components_v2/roi/roi_shapley.npz') as a:
        membership = a['network_membership'].copy()
    groups = [tuple(np.flatnonzero(membership == i)) for i in range(7)]
    protocol = dict(version=VERSION, source_sha256=digest(SOURCE),
        native_kernel_sha256=digest(ROOT/'scripts/dmf_response_benchmark.py'),
        implementation_sha256=digest(ROOT/'scripts/dmf_continuous_affine.py'),
        source_target='200 E/I source, complete 200 E/I future', support=[.3,.7],
        horizon_steps=300, dt=.001, sigma=.01, ridge=1e-6, fixed_JFIC_G=1.,
        G=G_VALUES, seeds=seeds, train_count=2048, validation_count=512, test_count=512,
        source_noise_seed_offsets={'train':0, 'validation':100, 'test':200},
        pairing='identical actual input arrays and reset noise generators within each split across G',
        validation_G=SELECTION_G, penalties=PENALTIES, tolerance_nats=TOL_NATS,
        selection='equal-weight mean validation conditional NLL over selection G and seeds; test untouched',
        design='exploratory estimator comparison; paired native responses, model/capacity differ',
        practical_parameterization='G-scaled mean and residual modified-Cholesky regressions; penalty on T_cross, not directly L_cross',
        limitations='Gaussian moment surrogate for uniform intervention; finite-sample bias remains; ordered noise regularization')
    protocol_sha = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (BASE/'protocol.json').exists():
        if json.loads((BASE/'protocol.json').read_text()) != protocol:
            raise ValueError('Pilot provenance changed; use a new versioned result directory')
    atomic_json(BASE/'protocol.json', protocol)
    dmf=load_dmf_module(); p=dmf.DMFParameters(dt=.001, sigma=.01)
    data, moments, pairing = {}, {}, []
    for seed in seeds:
        for g in G_VALUES:
            for split in ['train', 'validation', 'test']:
                x, y, diagnostics = response(dmf, p, sc, jf, g, seed, split, protocol_sha)
                data[seed, g, split] = (x, y)
                pairing.append(dict(seed=seed, G=g, split=split, input_sha256=array_digest(x),
                    target_sha256=array_digest(y), native_diagnostics=diagnostics))
            moments[seed, g] = sample_moments(*data[seed,g,'train'])
    for seed in seeds:
        for split in ['train','validation','test']:
            hashes = {r['input_sha256'] for r in pairing if r['seed']==seed and r['split']==split}
            if len(hashes) != 1:
                raise ValueError('Cross-G inputs are not paired')
        hashes = {r['input_sha256'] for r in pairing if r['seed']==seed}
        if len(hashes) != 3:
            raise ValueError('Train/validation/test inputs must be distinct')
    validation=[]
    for mean_penalty in PENALTIES:
        for noise_penalty in PENALTIES:
            scores=[]
            for seed in seeds:
                for g in SELECTION_G:
                    model = fit_continuous_affine(moments[seed,g], sc, g, mean_penalty, noise_penalty)
                    scores.append(dict(seed=seed, G=g, **model.scores(*data[seed,g,'validation'])))
            row=dict(mean_penalty=mean_penalty, noise_penalty=noise_penalty,
                     mean_validation_nll=float(np.mean([r['nll_per_transition'] for r in scores])), scores=scores)
            validation.append(row)
            print(f'Validation penalties {mean_penalty:g}/{noise_penalty:g}: {row["mean_validation_nll"]:.4f}', flush=True)
    selected=min(validation, key=lambda r:r['mean_validation_nll'])
    pa, pn=selected['mean_penalty'],selected['noise_penalty']
    print('Selected penalties:', pa, pn, flush=True)
    records=[]; sample_size=[]
    for seed in seeds:
        for g in G_VALUES:
            x,y=data[seed,g,'train']; test=data[seed,g,'test']; m=moments[seed,g]
            for method in ['Unrestricted', 'Current components', 'Continuous constraint']:
                model = (fit_continuous_affine(m, sc, g, pa, pn) if method=='Continuous constraint'
                         else baseline(m,x,y,sc,g,unrestricted=method=='Unrestricted'))
                record=dict(seed=seed,G=g,method=method,**describe(model,groups,test))
                records.append(record)
                if g==0 and method=='Current components':
                    path=ROOT/f'results/dmf_schaefer100/main_independent_components_v2/conditions/group_mean_93_G0.00_seed{seed}.npz'
                    with np.load(path) as a:
                        np.testing.assert_allclose(model.covariance,a['covariance'],atol=1e-12,rtol=0)
            if g in [0.,1e-6,.1,1.3]:
                for count in [512,1024,2048]:
                    ms=sample_moments(x[:count],y[:count])
                    for method in ['Current components','Continuous constraint']:
                        model=(fit_continuous_affine(ms,sc,g,pa,pn) if method=='Continuous constraint'
                               else baseline(ms,x[:count],y[:count],sc,g))
                        sample_size.append(dict(seed=seed,G=g,count=count,method=method,
                            **describe(model,groups,test)))
        print('Evaluation complete:',seed,flush=True)
    result=dict(protocol_sha256=protocol_sha,protocol=protocol,selected_penalties=selected,
        selection_on_grid_boundary=pa in [PENALTIES[0],PENALTIES[-1]] or pn in [PENALTIES[0],PENALTIES[-1]],
        validation=validation,records=records,sample_size=sample_size,pairing=pairing,
        elapsed_seconds=time.perf_counter()-started)
    atomic_json(BASE/'summary.json', result)
    print('Saved:',BASE/'summary.json',f'({result["elapsed_seconds"]:.1f}s)',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',nargs='+',type=int,default=[3,4,5])
    args=parser.parse_args()
    with threadpool_limits(1):
        run(args.seeds)
