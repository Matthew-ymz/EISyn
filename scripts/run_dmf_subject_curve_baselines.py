#!/usr/bin/env python3
"""Native observational PhiR and source-WMS on the frozen individual-SC pilot."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dmf_response_benchmark import simulate_sources
from scripts.run_dmf_subject_consistency import BASE, atomic_json, digest
from scripts.run_dmf_schaefer100_observational_phi_r import (
    pairwise_gaussian_mmi_phi_r, validate_vectorization,
)
from scripts import validate_dmf_83_region_oracle_phi_eid as wms_module

VERSION = 'individual-native-observational-curves-v1'


def atomic_savez(path, **payload):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def prepare(base):
    pilot = json.loads((base/'contract.json').read_text())
    with np.load(base/'inputs.npz') as a:
        if json.loads(str(a['contract_json'])) != pilot:
            raise ValueError('Frozen inputs and pilot contract disagree')
        ids = a['subject_ids'].tolist()
        matrices = a['connectivity'].copy()
        jf = a['j_fic'].copy()
    if ids != pilot['subject_ids'] or matrices.shape != (9,100,100):
        raise ValueError('Expected frozen eight subjects plus group-mean reference')
    dmf = wms_module.load_dmf_module()
    p = dmf.DMFParameters(t_total=1.5, burn_in=.3, dt=.001, sigma=.01)
    stabilization = dmf.StabilizationParameters(window=.05, tolerance_hz=.15, confirm_windows=2)
    sources = [Path(__file__), ROOT/'exp/brain/dmf_fig6.py',
               ROOT/'scripts/dmf_response_benchmark.py',
               ROOT/'scripts/run_dmf_schaefer100_observational_phi_r.py',
               ROOT/'scripts/validate_dmf_83_region_oracle_phi_eid.py']
    contract = dict(version=VERSION, pilot_contract=pilot,
        frozen_input_sha256=digest(base/'inputs.npz'),
        native_parameters=asdict(p), stabilization=asdict(stabilization),
        hemodynamic_parameters=asdict(dmf.BalloonWindkesselParameters()),
        natural_state_boundary='Inherited [0,1] hard clipping; every recorded step and final state audited for boundary hits',
        future_state_boundary='none; fail on any outside-[0,1] state or abnormal/nonfinite rate',
        seed_rules=dict(natural='62000 + nominal seed', sampling='63000 + nominal seed',
                        future='64000 + nominal seed',
                        pairing='Same stream across subjects and G; disjoint from intervention/diagnostic streams'),
        phi_r=dict(observable='Balloon-Windkessel BOLD-like from post-stabilization E rates',
                   definition='I(pair past; pair future)-I(left past; left future)-I(right past; right future)+min four scalar MIs',
                   pair_count=4950, lag_steps=1, lag_seconds=.001, reduction='mean over all unordered ROI pairs',
                   covariance_eigenvalue_floor=1e-10, one_minus_squared_correlation_floor=1e-10,
                   inherited_MI_nonnegative_projection=True, nonnegative_tolerance_bits=1e-10,
                   numerical_zero_rule='Pair values in [-1e-10,0) bits set to zero; raw values and count retained'),
        wms=dict(observable='Natural full 200-scalar E/I states -> full future 200-scalar E/I states',
                 definition='I(full past; full future)-sum_i I(past scalar i; full future)',
                 sample_count=2048, sample_rule='Uniform tail time-index sampling with replacement',
                 horizon_steps=300, horizon_seconds=.3, empirical_correlated_source_prior=True,
                 gaussian_ridge=1e-6, eigenvalue_floor=1e-12,
                 estimator='Native Gaussian least-squares transition; inherited ridge at covariance and subquery levels; no WMS sign projection',
                 interpretation='Signed observational source residual, not PEID Syn; joint_ei is native ordinary MI'),
        units='Native estimates bits; stored comparison arrays nats = bits * log(2)',
        implementation_sha256={str(path.relative_to(ROOT)):digest(path) for path in sources},
        manuscript_recheck=dict(parent_key='P6UJCVG8',attachment_key='DXGC7JEA',pages=19,
            version_date=None,sections='Brain/Fig.2; Methods Eqs.4-8',supplementary_appendices='unavailable',
            discrepancy='Main text calls both observational baselines BOLD-like; native repository WMS uses full E/I natural states, only PhiR uses BOLD-like signals'))
    path = base/'curve_native_contract.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Native contract/source changed; do not reuse incompatible condition caches')
    if not path.exists():
        atomic_json(path,contract)
    return contract,ids,matrices,jf,dmf,p,stabilization


def phi_floor_audit(bold):
    """Count inherited numerical floors without changing the native estimate."""
    joined = np.column_stack((bold[:-1],bold[1:]))
    joined = (joined-joined.mean(0))/joined.std(0,ddof=1)
    correlation = np.cov(joined,rowvar=False)
    n=bold.shape[1];left,right=np.triu_indices(n,1)
    indices=np.column_stack((left,right,left+n,right+n))
    batch=correlation[indices[:,:,None],indices[:,None,:]]
    result={}
    for name,block in [('source',batch[:,:2,:2]),('target',batch[:,2:,2:]),('joint',batch)]:
        eigen=np.linalg.eigvalsh(.5*(block+block.swapaxes(-1,-2)))
        result[name+'_eigenvalue_floor_count']=int((eigen<1e-10).sum())
        result[name+'_minimum_eigenvalue']=float(eigen.min())
    cross=batch[:,:2,2:]
    result['singleton_correlation_floor_count']=int(((1-cross**2)<1e-10).sum())
    return result


def native_wms(x,y):
    """Same native logdet calculation, additionally count its floor activations."""
    original=wms_module.safe_logdet_psd
    audit=dict(logdet_query_count=0,eigenvalue_floor_count=0,minimum_eigenvalue=None)
    def audited(matrix,*,floor=1e-12):
        eigen=np.linalg.eigvalsh(.5*(matrix+matrix.T))
        audit['logdet_query_count']+=1
        audit['eigenvalue_floor_count']+=int((eigen<floor).sum())
        minimum=float(eigen.min())
        audit['minimum_eigenvalue']=minimum if audit['minimum_eigenvalue'] is None else min(minimum,audit['minimum_eigenvalue'])
        return float(np.log(np.maximum(eigen,floor)).sum())
    try:
        wms_module.safe_logdet_psd=audited
        result=wms_module.gaussian_singleton_source_phi(x,y,ridge=1e-6,factorize_source_covariance=False)
    finally:
        wms_module.safe_logdet_psd=original
    if not np.isfinite(result['singleton_ei']).all():
        raise ArithmeticError('Nonfinite native singleton MI')
    scalar={k:v for k,v in result.items() if np.isscalar(v)}
    if not np.isfinite([v for v in scalar.values()]).all():
        raise ArithmeticError('Nonfinite WMS estimates/diagnostics')
    if abs(result['raw_phi']-(result['joint_ei']-result['singleton_ei_sum']))>1e-10:
        raise ArithmeticError('Native WMS component identity failed')
    return scalar,audit


def condition(sc,jf,g,seed,dmf,p,stabilization):
    started=time.perf_counter()
    sim=dmf.simulate_dmf(sc,g,jf,parameters=p,stabilization_parameters=stabilization,
                       seed=62000+seed,record_state_trace=True,record_rate_trace=True)
    start=int(sim['stabilization_start_step'])
    states=np.column_stack((sim['state_se_trace'],sim['state_si_trace']))
    final=np.concatenate((sim['final_se'],sim['final_si']))
    rates=np.asarray(sim['region_rate_trace_hz'])
    if not np.isfinite(states).all() or not np.isfinite(final).all() or not np.isfinite(rates).all():
        raise ArithmeticError('Nonfinite natural state or rate')
    if states.min()<0 or states.max()>1 or final.min()<0 or final.max()>1 or rates.min()<0 or rates.max()>500:
        raise ArithmeticError('Invalid native natural state/rate bounds')
    tail=states[start:]
    if len(tail)<4:
        raise ArithmeticError('Natural tail too short')
    bold=dmf.transform_rates_to_bold(rates[start:],dt=p.dt)
    phi=pairwise_gaussian_mmi_phi_r(bold,tau=1,tolerance_bits=1e-10)
    if not np.isfinite(phi['phi_r_raw']).all():
        raise ArithmeticError('Nonfinite PhiR')
    audit_phi=phi_floor_audit(bold)
    indices=np.random.default_rng(63000+seed).integers(0,len(tail),size=2048)
    x=tail[indices].copy()
    y,diagnostic=simulate_sources(dmf,x,sc,jf,g,p,64000+seed)
    if diagnostic['outside_state_count'] or diagnostic['abnormal_rate_count'] or not np.isfinite(y).all():
        raise ArithmeticError(f'Native future state/rate violation: {diagnostic}')
    if np.any(x.std(0,ddof=1)<=1e-12) or np.any(y.std(0,ddof=1)<=1e-12):
        raise ArithmeticError('Constant natural source/target channel')
    wms,audit_wms=native_wms(wms_module.standardize(x)[0],wms_module.standardize(y)[0])
    drift=float(sim['stabilization_last_drift_hz'])
    natural=dict(stabilization_detected=bool(sim['stabilization_detected']),start_step=start,
        start_seconds=start*p.dt,last_drift_hz=drift if np.isfinite(drift) else None,
        tail_timepoint_count=len(tail),sampled_unique_timepoint_count=int(np.unique(indices).size),
        boundary_hit_count=int(((states==0)|(states==1)).sum()+((final==0)|(final==1)).sum()),
        state_min=float(min(states.min(),final.min())),state_max=float(max(states.max(),final.max())),
        mean_rate_hz=float(rates[start:].mean()))
    metrics=dict(phi_r_bits=float(np.mean(phi['phi_r'])),phi_r_raw_mean_bits=float(np.mean(phi['phi_r_raw'])),
        phi_r_minimum_raw_bits=phi['minimum_raw_phi_r'],phi_r_numerical_zero_count=phi['numerical_zero_count'],
        wms_bits=wms['raw_phi'],whole_mi_bits=wms['joint_ei'],singleton_mi_sum_bits=wms['singleton_ei_sum'],
        source_condition_number=wms['source_condition_number'],noise_condition_number=wms['noise_condition_number'],
        elapsed_seconds=time.perf_counter()-started)
    return dict(metrics_json=json.dumps(metrics,allow_nan=False),natural_json=json.dumps(natural,allow_nan=False),
        future_diagnostics_json=json.dumps(diagnostic,allow_nan=False),
        phi_audit_json=json.dumps(audit_phi,allow_nan=False),wms_audit_json=json.dumps(audit_wms,allow_nan=False),
        pair_indices=phi['pair_indices'],phi_r_raw_bits=phi['phi_r_raw'],phi_r_bits=phi['phi_r'],
        source_sha256=hashlib.sha256(x.tobytes()).hexdigest())


def aggregate(base,contract,ids):
    shape=(len(ids),len(contract['pilot_contract']['seeds']),len(contract['pilot_contract']['G']))
    phi=np.empty(shape);wms=np.empty(shape);audits=[]
    for pi,subject in enumerate(ids):
        for si,seed in enumerate(contract['pilot_contract']['seeds']):
            for gi,g in enumerate(contract['pilot_contract']['G']):
                path=base/'curve_native_conditions'/f'{subject}_G{g:.2f}_seed{seed}.npz'
                with np.load(path) as a:
                    if json.loads(str(a['contract_json']))!=contract:
                        raise ValueError(f'Native cache contract mismatch: {path.name}')
                    row={key:json.loads(str(a[key+'_json'])) for key in
                         ['metrics','natural','future_diagnostics','phi_audit','wms_audit']}
                phi[pi,si,gi]=row['metrics']['phi_r_bits']*np.log(2)
                wms[pi,si,gi]=row['metrics']['wms_bits']*np.log(2)
                audits.append(dict(subject=subject,G=g,seed=seed,**row))
    if not np.isfinite(phi).all() or not np.isfinite(wms).all():
        raise ArithmeticError('Nonfinite baseline aggregate')
    audit=dict(condition_count=len(audits),
        phi_r_numerical_zero_count=sum(r['metrics']['phi_r_numerical_zero_count'] for r in audits),
        phi_r_minimum_raw_bits=min(r['metrics']['phi_r_minimum_raw_bits'] for r in audits),
        natural_boundary_hit_count=sum(r['natural']['boundary_hit_count'] for r in audits),
        stabilization_detected_condition_count=sum(r['natural']['stabilization_detected'] for r in audits),
        future_outside_state_count=sum(r['future_diagnostics']['outside_state_count'] for r in audits),
        phi_r_covariance_floor_count=sum(sum(r['phi_audit'][key] for key in
            ['source_eigenvalue_floor_count','target_eigenvalue_floor_count','joint_eigenvalue_floor_count']) for r in audits),
        phi_r_singleton_correlation_floor_count=sum(r['phi_audit']['singleton_correlation_floor_count'] for r in audits),
        wms_covariance_floor_count=sum(r['wms_audit']['eigenvalue_floor_count'] for r in audits),
        source_condition_number_max=max(r['metrics']['source_condition_number'] for r in audits),
        noise_condition_number_max=max(r['metrics']['noise_condition_number'] for r in audits),
        minimum_sampled_unique_timepoints=min(r['natural']['sampled_unique_timepoint_count'] for r in audits),
        conditions=audits)
    atomic_savez(base/'curve_baselines.npz',subject_ids=ids,seeds=contract['pilot_contract']['seeds'],
        G=contract['pilot_contract']['G'],completed=np.ones(shape,bool),phi_r_nats=phi,wms_nats=wms,
        contract_json=json.dumps(contract),audit_json=json.dumps(audit,allow_nan=False))
    atomic_json(base/'curve_native_audit.json',audit)
    atomic_json(base/'curve_native_completed.json',dict(status='simulations_complete',condition_count=len(audits),
        analysis_complete=False,baseline_sha256=digest(base/'curve_baselines.npz'),updated_at=time.time()))
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--limit',type=int,help='Only compute this many missing conditions; no partial aggregate')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        contract,ids,matrices,jf,dmf,p,stabilization=prepare(args.base)
        error=validate_vectorization()
        print(f'Native PhiR scalar/vector check: max error {error:.3g} bits',flush=True)
        cache=args.base/'curve_native_conditions';cache.mkdir(exist_ok=True)
        new=0;started=time.perf_counter()
        for subject,sc in zip(ids,matrices):
            for g in contract['pilot_contract']['G']:
                for seed in contract['pilot_contract']['seeds']:
                    path=cache/f'{subject}_G{g:.2f}_seed{seed}.npz'
                    if path.exists():
                        with np.load(path) as a:
                            if json.loads(str(a['contract_json']))!=contract:
                                raise ValueError(f'Native cache contract mismatch: {path.name}')
                        continue
                    payload=condition(sc,jf,g,seed,dmf,p,stabilization)
                    atomic_savez(path,contract_json=json.dumps(contract),**payload)
                    metric=json.loads(payload['metrics_json']);natural=json.loads(payload['natural_json'])
                    new+=1
                    print(f'done {path.stem}: PhiR={metric["phi_r_bits"]:.6f}, WMS={metric["wms_bits"]:.6f} bits; '
                          f'natural_boundary_hits={natural["boundary_hit_count"]}, future_outside=0; {metric["elapsed_seconds"]:.2f}s',flush=True)
                    if args.limit and new>=args.limit:
                        print('Requested condition limit reached; full aggregate not written',flush=True)
                        return
        audit=aggregate(args.base,contract,ids)
        print(f'Native baseline simulations complete: {audit["condition_count"]} conditions, new={new}, '
              f'elapsed={time.perf_counter()-started:.1f}s',flush=True)


if __name__=='__main__':main()
