#!/usr/bin/env python3
"""Versioned migration of all 93+mean-SC curves to the validated continuous TM.

Individual trajectories are recovered exactly from unrestricted OLS sufficient
statistics. Mean-SC conditions without a compatible cache alone are replayed.
Frozen supports, source/noise streams, G grid, penalties and windows stay fixed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.dmf_continuous_affine import VERSION as ESTIMATOR_VERSION
from scripts.dmf_continuous_curve_refresh import recover_moments, fit_continuous_batched
from scripts.dmf_joint_readout import CommonTargetGame, audit_nonnegative
from scripts.dmf_response_benchmark import fit_affine_joint, simulate_sources
from scripts.dmf_main_independent_components import BASE as PREVIOUS
from scripts.run_dmf_paired_spt_pilot import paired_sources, noise_seed
from scripts.run_dmf_subject_consistency import atomic_json, digest
from scripts.run_dmf_subject_curve_baselines import atomic_savez
from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module

BASE=ROOT/'results/dmf_schaefer100/main_continuous_curves_v1'
DENSE=ROOT/'results/dmf_schaefer100/subject_curves_93_dense'
LEGACY=ROOT/'results/dmf_schaefer100/critical_coalitions/density'
PILOT=ROOT/'results/dmf_schaefer100/continuous_affine_pilot_v1'
MEAN_OLD=ROOT/'results/dmf_schaefer100/subject_consistency_pilot/conditions'
SOURCE=ROOT/'results/dmf_schaefer100/source/group_mean_native_mean_rate.npz'
TOL=1e-8
FIELDS=['xi','whole_ei','partial_ei_sum','cross_roi','local_roi_xi']
CONTEXT={}


def initialize(identity, identity_sha):
    CONTEXT.update(identity=identity,identity_sha=identity_sha)
    for seed in [3,4,5]:
        x=np.concatenate(paired_sources(seed,2048,100),axis=1)
        CONTEXT[seed]=(x,hashlib.sha256(x.tobytes()).hexdigest())


def load_legacy(path, seed, expected_contract=None):
    payload=path.read_bytes()
    with np.load(io.BytesIO(payload)) as a:
        if (str(a['source_sha256'])!=CONTEXT[seed][1] or int(a['noise_seed'])!=noise_seed(seed)
                or expected_contract and str(a['contract_sha256'])!=expected_contract):
            raise ValueError(f'Incompatible OLS density provenance: {path}')
        covariance=a['covariance'].copy()
    return covariance,hashlib.sha256(payload).hexdigest()


def mean_covariance(g,seed,sc,jf):
    original=MEAN_OLD/f'group_mean_93_G{g:.2f}_seed{seed}.npz'
    if original.exists():
        covariance,sha=load_legacy(original,seed)
        return covariance,dict(path=str(original.relative_to(ROOT)),sha256=sha,replayed=False)
    path=BASE/'mean_OLS'/f'group_mean_93_G{g:.2f}_seed{seed}.npz'
    if path.exists():
        covariance,sha=load_legacy(path,seed,CONTEXT['identity_sha'])
        return covariance,dict(path=str(path.relative_to(ROOT)),sha256=sha,replayed=True,cached=True)
    x=CONTEXT[seed][0]
    old_response=PILOT/'responses'/f'seed{seed}_train_G{g:.8g}.npz'
    reused=False
    if old_response.exists():
        with np.load(old_response) as a:
            if (str(a['protocol_sha256'])!=CONTEXT['identity']['pilot_protocol_sha256']
                    or str(a['input_sha256'])!=CONTEXT[seed][1] or int(a['noise_seed'])!=noise_seed(seed)):
                raise ValueError('Pilot response provenance changed')
            y=a['y'].copy(); diagnostics=json.loads(str(a['diagnostics_json']))
        reused=True
    else:
        dmf=load_dmf_module(); p=dmf.DMFParameters(dt=.001,sigma=.01)
        y,diagnostics=simulate_sources(dmf,x,sc,jf,g,p,noise_seed(seed))
    if diagnostics['outside_state_count'] or diagnostics['abnormal_rate_count']:
        raise ArithmeticError(f'Mean-SC response invalid: {g}, {seed}, {diagnostics}')
    covariance=fit_affine_joint(x,y,.2)
    path.parent.mkdir(exist_ok=True)
    atomic_savez(path,covariance=covariance,contract_sha256=CONTEXT['identity_sha'],
        source_sha256=CONTEXT[seed][1],noise_seed=noise_seed(seed),
        diagnostics_json=json.dumps(diagnostics),pilot_response_reused=reused)
    return covariance,dict(path=str(path.relative_to(ROOT)),sha256=digest(path),replayed=not reused,pilot_response_reused=reused)


def refresh_one(index,subject,sc,jf,gs,original):
    started=time.perf_counter(); path=BASE/'subjects'/f'{subject}.npz'
    if path.exists():
        with np.load(path) as a:
            if str(a['identity_sha256'])!=CONTEXT['identity_sha']:
                raise ValueError(f'Subject curve provenance changed: {subject}')
        return dict(index=index,subject=subject,reused=True,path=str(path.relative_to(ROOT)))
    metrics=np.empty((3,len(gs),5)); audited=[]; sources=[]; diagnostics=[]
    with threadpool_limits(1):
        for si,seed in enumerate([3,4,5]):
            for gi,g in enumerate(gs):
                if subject=='group_mean_93':
                    covariance,source=mean_covariance(float(g),seed,sc,jf)
                else:
                    raw=LEGACY/f'{subject}_G{g:.2f}_seed{seed}.npz'
                    covariance,sha=load_legacy(raw,seed,CONTEXT['identity']['legacy_contract_sha256'])
                    source=dict(path=str(raw.relative_to(ROOT)),sha256=sha,replayed=False)
                old=CommonTargetGame(covariance).totals()
                np.testing.assert_allclose([old['xi_nats'],old['whole_ei_nats']],
                    original[si,gi,:2],atol=TOL,rtol=0,
                    err_msg=f'Legacy density differs from original frozen curve: {subject}, {seed}, {g}')
                m=recover_moments(covariance,CONTEXT[seed][0])
                model=fit_continuous_batched(m,sc,float(g),
                    CONTEXT['identity']['mean_penalty'],CONTEXT['identity']['noise_penalty'])
                game=CommonTargetGame(model.covariance); value=game.totals()
                metrics[si,gi]=[value['xi_nats'],value['whole_ei_nats'],float(game.scalar_ei.sum()),
                    value['cross_roi_nats'],value['roi_local_xi_nats']]
                audited.extend([*game.audited,*value.values()])
                sources.append(dict(seed=seed,G=float(g),**source))
                diagnostics.append(dict(seed=seed,G=float(g),**model.diagnostics))
                if g==0 and abs(value['cross_roi_nats'])>TOL:
                    raise ArithmeticError(f'G0 independence failed for {subject}, {seed}')
    audit=audit_nonnegative(audited,TOL)
    closure=float(np.abs(metrics[:,:,1]-metrics[:,:,2]-metrics[:,:,0]).max())
    if closure>TOL:
        raise ArithmeticError(f'Full EI budget failed for {subject}: {closure} nats')
    atomic_savez(path,metrics=metrics,G=gs,seeds=np.array([3,4,5]),subject=subject,
        identity_sha256=CONTEXT['identity_sha'],audit_json=json.dumps(audit),
        source_json=json.dumps(sources),diagnostics_json=json.dumps(diagnostics),
        elapsed_seconds=time.perf_counter()-started)
    return dict(index=index,subject=subject,reused=False,path=str(path.relative_to(ROOT)),
                elapsed_seconds=time.perf_counter()-started)


def run(workers):
    started=time.perf_counter(); BASE.mkdir(parents=True,exist_ok=True)
    (BASE/'subjects').mkdir(exist_ok=True)
    with np.load(DENSE/'analysis_curves.npz') as a:
        original={k:a[k].copy() for k in a.files}
    with np.load(PREVIOUS/'analysis_curves.npz') as a:
        output={k:a[k].copy() for k in a.files}
    with np.load(DENSE/'inputs.npz') as a:
        sc=a['connectivity'].copy(); jf=a['j_fic'].copy(); ids=a['subject_ids'].tolist()
    if ids!=original['subject_ids'].tolist() or ids!=output['subject_ids'].tolist() or len(ids)!=94:
        raise ValueError('Subject order or 93+mean-SC denominator changed')
    gs=original['G']
    if not np.array_equal(gs,np.round(np.arange(41)*.1,1)):
        raise ValueError('Frozen G grid changed')
    pilot=json.loads((PILOT/'summary.json').read_text()); selected=pilot['selected_penalties']
    if selected['mean_penalty']!=.1 or selected['noise_penalty']!=10.:
        raise ValueError('Pilot-selected penalties changed')
    contract=json.loads((DENSE/'contract.json').read_text())
    expected=dict(sample_count=2048,support=[.3,.7],horizon_steps=300,dt_seconds=.001,sigma=.01,ridge=1e-6)
    if any(contract[k]!=v for k,v in expected.items()):
        raise ValueError('Frozen native protocol changed')
    files=['scripts/dmf_continuous_affine.py','scripts/dmf_continuous_curve_refresh.py',
           'scripts/run_dmf_continuous_subject_curves.py','scripts/dmf_response_benchmark.py',
           'exp/brain/dmf_fig6.py','exp/TM/transport_map_density.py']
    identity=dict(version='main-continuous-curves-v1',estimator=ESTIMATOR_VERSION,
        original_curves_sha256=digest(DENSE/'analysis_curves.npz'),
        previous_curves_sha256=digest(PREVIOUS/'analysis_curves.npz'),
        frozen_inputs_sha256=digest(DENSE/'inputs.npz'),frozen_contract_sha256=digest(DENSE/'contract.json'),
        fixed_window_sha256=digest(DENSE/'fixed_window_view.json'),source_sha256=digest(SOURCE),
        legacy_contract_sha256=digest(LEGACY.parent/'contract.json'),
        pilot_summary_sha256=digest(PILOT/'summary.json'),pilot_protocol_sha256=pilot['protocol_sha256'],
        mean_penalty=.1,noise_penalty=10.,selection='frozen mean-SC validation selection; no per-subject tuning',
        subject_ids=ids,G=gs.tolist(),seeds=[3,4,5],protocol=expected,tolerance_nats=TOL,
        implementation_sha256={f:digest(ROOT/f) for f in files},
        manuscript=dict(parent='P6UJCVG8',attachments=['DXGC7JEA','MWIWKSVG'],checked_date='2026-10-14',
            explicit_version_or_date=None,locations='Main Methods Eqs5-10 pp15-16; SI S1.2 pp3-4, S12.2.1 p23',
            ambiguity='No explicit draft date/revision; current accessible attachments re-read.'),
        scope='all 93 curves plus mean-SC reference; natural comparators and independent rate unchanged; lower-panel attribution/hierarchy retained and labeled')
    identity_sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if (BASE/'identity.json').exists() and json.loads((BASE/'identity.json').read_text())!=identity:
        raise ValueError('Migration identity changed; use a new result version')
    atomic_json(BASE/'identity.json',identity)
    # Verify every individual condition exists before dispatching the migration.
    missing=[str(LEGACY/f'{subject}_G{g:.2f}_seed{seed}.npz') for subject in ids[:-1]
             for seed in [3,4,5] for g in gs if not (LEGACY/f'{subject}_G{g:.2f}_seed{seed}.npz').exists()]
    if missing:
        raise FileNotFoundError(f'Missing unrestricted densities: {len(missing)}, first={missing[0]}')
    completed=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(identity,identity_sha)) as pool:
        # Dispatch the missing mean-SC trajectories first; other workers refit caches.
        order=[93,*range(93)]
        futures={pool.submit(refresh_one,i,ids[i],sc[i],jf,gs,
            np.stack([original['xi'][i],original['whole_ei'][i]],axis=-1)):i for i in order}
        for future in as_completed(futures):
            row=future.result(); completed.append(row)
            print(f'Completed {len(completed)}/94: {row["subject"]}; '
                  f'{"cache" if row["reused"] else str(round(row["elapsed_seconds"],1))+"s"}',flush=True)
    audits=[]; subject_sources={}; diagnostics=[]
    for i,subject in enumerate(ids):
        path=BASE/'subjects'/f'{subject}.npz'
        with np.load(path) as a:
            if str(a['identity_sha256'])!=identity_sha:
                raise ValueError('Subject cache identity changed before assembly')
            metrics=a['metrics'].copy(); audits.append(json.loads(str(a['audit_json'])))
            subject_sources[subject]=json.loads(str(a['source_json']))
            diagnostics.extend(json.loads(str(a['diagnostics_json'])))
        for j,name in enumerate(FIELDS):
            if name not in output:
                output[name]=np.empty((94,3,41))
            output[name][i]=metrics[:,:,j]
    for name in ['rate','phi_r','wms','susceptibility']:
        np.testing.assert_array_equal(output[name],original[name])
    np.testing.assert_allclose(output['xi'][:,:,0],output['local_roi_xi'][:,:,0],atol=TOL,rtol=0)
    if np.abs(output['cross_roi'][:,:,0]).max()>TOL:
        raise ArithmeticError('G0 cross-ROI increment exceeds native tolerance')
    output['curve_identity_sha256']=np.asarray(identity_sha)
    output['curve_estimator_version']=np.asarray(ESTIMATOR_VERSION)
    output.pop('correction_identity_sha256',None)
    atomic_savez(BASE/'analysis_curves.npz',**output)
    old_peaks=original['G'][np.argmax(np.load(PREVIOUS/'analysis_curves.npz')['xi'].mean(1),axis=-1)]
    new_peaks=gs[np.argmax(output['xi'].mean(1),axis=-1)]
    summary=dict(identity_sha256=identity_sha,identity=identity,subject_count=93,reference_count=1,
        condition_count=94*3*41,curves_sha256=digest(BASE/'analysis_curves.npz'),
        subject_sources=subject_sources,subject_audits=audits,parameter_diagnostics=diagnostics,
        audit=dict(tolerance_nats=TOL,query_count=sum(r['count'] for r in audits),
            minimum_nats=min(r['minimum_nats'] for r in audits),
            tolerance_negative_count=sum(r['tolerance_negative_count'] for r in audits),
            violation_count=sum(r['violation_count'] for r in audits),
            EI_closure_max_nats=float(np.abs(output['whole_ei']-output['partial_ei_sum']-output['xi']).max()),
            unchanged_fields=['rate','phi_r','wms','susceptibility'],
            G0_cross_roi_max_abs_nats=float(np.abs(output['cross_roi'][:,:,0]).max())),
        peak_changes=[dict(subject=ids[i],old_G=float(old_peaks[i]),new_G=float(new_peaks[i]))
                      for i in range(94) if old_peaks[i]!=new_peaks[i]],
        G0_mean_nats=float(output['xi'][-1,:,0].mean()),G01_mean_nats=float(output['xi'][-1,:,1].mean()),
        old_G01_mean_nats=float(np.load(PREVIOUS/'analysis_curves.npz')['xi'][-1,:,1].mean()),
        elapsed_seconds=time.perf_counter()-started)
    atomic_json(BASE/'summary.json',summary)
    print('Completed continuous curve migration:',summary['condition_count'],'conditions;',
          f'{summary["elapsed_seconds"]:.1f}s',flush=True)


def load_curve_manifest():
    summary=json.loads((BASE/'summary.json').read_text())
    identity=json.loads((BASE/'identity.json').read_text())
    sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if summary['identity_sha256']!=sha or summary['identity']!=identity:
        raise ValueError('Continuous curve identity differs')
    for source,key in [(DENSE/'analysis_curves.npz','original_curves_sha256'),
        (DENSE/'inputs.npz','frozen_inputs_sha256'),(DENSE/'fixed_window_view.json','fixed_window_sha256'),
        (PILOT/'summary.json','pilot_summary_sha256')]:
        if digest(source)!=identity[key]:
            raise ValueError(f'Continuous curve source changed: {source}')
    if digest(BASE/'analysis_curves.npz')!=summary['curves_sha256'] or summary['audit']['violation_count']:
        raise ValueError('Continuous curve output changed or failed Syn audit')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    run(args.workers)
