#!/usr/bin/env python3
"""Frozen all-available HCP Schaefer1000 replication, with resumable subject jobs."""
from __future__ import annotations

import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_k]='1'
os.environ.setdefault('MPLBACKEND','Agg')
import argparse
import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_hcp_timeseries_abc_979 as core
from scripts.verify_hcp_57_abc_equivalence import errors as legacy_errors
from scripts.run_hcp_schaefer500_length_matched_variance import evenly_spaced_window_starts

OUT=ROOT/'results/hcp_schaefer1000_full_replication_20261022'
CONTRACT=ROOT/'docs/log/hcp_schaefer1000_full_replication_contract_20261022.json'
AUDIT=ROOT/'docs/log/hcp_s1200_download_audit_20261022.json'
STATES=('REST','EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM')
TASKS=STATES[1:]
LENGTHS=dict(REST=1200,EMOTION=176,GAMBLING=253,LANGUAGE=316,MOTOR=284,RELATIONAL=232,SOCIAL=274,WM=405)
OLD_REST=ROOT/'data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_57_brain'
OLD_TASK=core.OLD_TASK
EXCLUDED_TASK_DESIGN={'735148'}  # Explicit user authorization on 2026-10-22.
WINDOW_MANIFEST=None


def init_window_worker(manifest):
    global WINDOW_MANIFEST
    WINDOW_MANIFEST=manifest


def sha(path):
    return core.sha(Path(path))


def dump(path,value):
    core.atomic_json(Path(path),value)


def digest(value):
    return core.digest(value)


def key(state):
    return 'Schaefer1000_NoGSR' if state=='REST' else f'{state}_LR_Schaefer1000_NoGSR'


def prep():
    contract=json.loads(CONTRACT.read_text());audit=json.loads(AUDIT.read_text())
    root=Path(audit['data_root']);designroot=Path(contract['design_readiness']['design_root'])
    cohort={s:list(audit['cohorts']['REST1_LR' if s=='REST' else s]['standard_valid']) for s in STATES}
    for s in TASKS:cohort[s]=sorted(set(cohort[s])-EXCLUDED_TASK_DESIGN)
    implementation={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/run_hcp_timeseries_abc_979.py',ROOT/'scripts/spt.py',ROOT/'scripts/run_hcp_schaefer500_yeo7_pc1_phi_null.py']}
    model=dict(contract['model']);config=digest(dict(model=model,implementation=implementation,labels=sha(core.LABELS),tolerance_bits=core.TOL))
    inputs={};provenance={}
    with zipfile.ZipFile(root/'metadata/nas_publication_records.zip') as z:
        for name in z.namelist():
            if '/' not in name or not name.endswith('.json'):continue
            d=json.loads(z.read(name))
            if d.get('status')=='complete' and d.get('task') in TASKS:
                provenance[str(d['subject']),d['task']]=d
    for row in audit['files']:
        sid=row['subject'];states=[s for s in STATES if sid in cohort[s]]
        if not states:continue
        path=root/f'{sid}.mat';stat=path.stat()
        if stat.st_size!=row['bytes']:raise ValueError(f'Input size changed: {sid}')
        inp=dict(path=str(path),sha256=row['sha256'],bytes=row['bytes'],states=states,designs={},fingerprints={},array_hashes={})
        for s in states:
            inp['array_hashes'][s]=row['matrices'][key(s)]['array_sha256']
            if s!='REST':
                p=designroot/f'sub-{sid}/{s}_LR_design.tsv';q=designroot/f'sub-{sid}/{s}_LR.json'
                design=np.loadtxt(p,ndmin=2);qc=json.loads(q.read_text());pr=provenance[sid,s]
                if design.shape[0]!=LENGTHS[s] or not np.isfinite(design).all() or qc.get('subject')!=sid or qc.get('task')!=s or qc.get('direction')!='LR':
                    raise ValueError(f'Design identity invalid: {sid}/{s}')
                if qc.get('official_fsf_sha256')!=pr['fsf']['official_sha256']:
                    raise ValueError(f'FSF source mismatch: {sid}/{s}')
                inp['designs'][s]=dict(path=str(p),sha256=sha(p),qc_path=str(q),qc_sha256=sha(q),shape=list(design.shape),source_fsf_sha256=qc['official_fsf_sha256'])
            inp['fingerprints'][s]=digest(dict(config=config,subject=sid,state=s,array=inp['array_hashes'][s],design=inp['designs'].get(s)))
        inputs[sid]=inp
    pairs={s:sorted(set(cohort['REST'])&set(cohort[s])) for s in TASKS}
    common=sorted(set.intersection(*(set(v) for v in cohort.values())))
    windows={sid:[s for s in TASKS if sid in pairs[s]] for sid in inputs if sid in cohort['REST']}
    windows={sid:v for sid,v in windows.items() if v}
    m=dict(status='prepared',config_hash=config,contract_sha256=sha(CONTRACT),contract_path=str(CONTRACT),cohorts=cohort,pairs=pairs,common=common,inputs=inputs,window_jobs=windows,
        methods=model,implementation=implementation,labels=dict(path=str(core.LABELS),sha256=sha(core.LABELS)),
        amendment=dict(date='2026-10-22',user_authorized='Skip only the seven task runs of 735148 because design matrices are missing; keep eligible REST',excluded_task_design_subjects=sorted(EXCLUDED_TASK_DESIGN)),
        manuscript=dict(parent='P6UJCVG8',main='DXGC7JEA',supplement='MWIWKSVG',rechecked='2026-10-22 at execution',version='No explicit manuscript version/date; only one current attachment per document',sections=['Main Fig3 pp7-9','SI S1.2-S1.3 EqsS7-S19','SI S5.1 EqsS49-S50','SI S12.2.2 pp23,25']))
    dump(OUT/'manifest.json',m)
    print(json.dumps(dict(phase='prepared',counts={s:len(v) for s,v in cohort.items()},pairs={s:len(v) for s,v in pairs.items()},common=len(common),models=sum(map(len,cohort.values())),matched_models=12*sum(map(len,pairs.values())))),flush=True)
    return m


def read_input(inp,states):
    raw=Path(inp['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=inp['sha256']:raise ValueError(f'Input SHA256 changed: {inp["path"]}')
    d=loadmat(io.BytesIO(raw),variable_names=[key(s) for s in states],verify_compressed_data_integrity=True)
    return {s:np.asarray(d[key(s)],dtype=float) for s in states}


def record_path(sid):return OUT/'subjects'/f'{sid}.json'


def existing(sid,inp):
    p=record_path(sid)
    if not p.exists():return {}
    saved=json.loads(p.read_text())
    return {s:r for s,r in saved.get('records',{}).items() if r['fingerprint']==inp['fingerprints'].get(s)}


def subject_worker(sid,inp):
    records=existing(sid,inp);missing=[s for s in inp['states'] if s not in records]
    if not missing:return 0
    data=read_input(inp,missing);groups=core.load_yeo7_groups(core.LABELS,expected_parcels=1000)
    cache={};cache_path=OUT/'pc1_cache'/f'{sid}.npz'
    if cache_path.exists():
        with np.load(cache_path) as f:cache={k:f[k] for k in f.files}
    for s in missing:
        raw=data[s]
        if raw.shape!=(LENGTHS[s],1000) or not np.isfinite(raw).all():raise ValueError(f'Invalid input {sid}/{s}')
        if s=='REST':fitting=raw
        else:
            design=inp['designs'][s]
            if sha(design['path'])!=design['sha256']:raise ValueError(f'Changed design {sid}/{s}')
            fitting=core.task_component(raw,design)
        row,arrays=core.fit_series(raw,fitting,groups,sid,s,inp['fingerprints'][s])
        records[s]=row;cache.update(arrays)
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    tmp=cache_path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**cache);os.replace(tmp,cache_path)
    dump(record_path(sid),dict(subject=sid,records=records))
    return len(missing)


def legacy_worker(sid,refs,inp):
    groups=core.load_yeo7_groups(core.LABELS,expected_parcels=1000);rows=[];checks=[]
    current=read_input(inp,inp['states']);new=existing(sid,inp);cache={}
    for s in STATES:
        if s=='REST':
            p=OLD_REST/f'sub-{sid}/sub-{sid}_hcp_s1200_rfMRI_REST1_LR_schaefer500-1000_yeo7.mat'
            raw=loadmat(p,variable_names=['Schaefer1000'])['Schaefer1000'].astype(float);fitting=raw
        else:
            p=OLD_TASK/f'sub-{sid}/{s}_LR.mat';d=loadmat(p,variable_names=['Schaefer1000_taskRetained','Schaefer1000_taskRegressed'])
            raw=d['Schaefer1000_taskRetained'].astype(float);regressed=d['Schaefer1000_taskRegressed'].astype(float);fitting=raw-regressed
        old,_=core.fit_series(raw,fitting,groups,sid,s,'legacy-original')
        err=legacy_errors(old,refs[s]);fail=[k for k,v in err.items() if (k.endswith('_bits') and v>1e-8) or (k.endswith('_fraction') and v>1e-9)]
        if err['changed_spt_source_sets'] or err['heldout_skill_ratio']>1e-9:fail.append('topology_or_prediction')
        if fail:raise ValueError(f'Legacy replay failed: {sid}/{s}: {fail}: {err}')
        old['input_sha256']=sha(p);rows.append(old)
        signal=current[s]
        if s=='REST':component=signal;reg_error=None
        else:
            rebuilt=raw-core.task_component(raw,inp['designs'][s]);reg_error=float(np.linalg.norm(rebuilt-regressed)/np.linalg.norm(regressed))
            if reg_error>1e-6:raise ValueError(f'Regression reconstruction failed {sid}/{s}: {reg_error}')
            component=core.task_component(signal,inp['designs'][s])
        row,arrays=core.fit_series(signal,component,groups,sid,s,inp['fingerprints'][s]);new[s]=row;cache.update(arrays)
        x=signal-signal.mean(0);y=raw-raw.mean(0);slopes=np.sum(x*y,axis=0)/np.sum(x*x,axis=0)
        affine=float(np.linalg.norm(y-x*slopes)/np.linalg.norm(y))
        if affine>1e-4:raise ValueError(f'Run/parcel correspondence mismatch {sid}/{s}: {affine}')
        checks.append(dict(state=s,errors=err,regression_relative_error=reg_error,roi_affine_error=affine,new_minus_old_system_bits=row['system_xi_bits']-old['system_xi_bits'],old_system_xi_bits=old['system_xi_bits'],new_system_xi_bits=row['system_xi_bits']))
    p=OUT/'pc1_cache'/f'{sid}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**cache)
    dump(record_path(sid),dict(subject=sid,records=new))
    dump(OUT/'legacy'/f'{sid}.json',dict(subject=sid,rows=rows,checks=checks))
    return checks


def run_legacy(m,workers):
    ref={}
    for line in (ROOT/'results/hcp_schaefer1000_task_evoked_xi_57/full/records.jsonl').open():
        r=json.loads(line)
        if r['config_id']=='k1_p3_a1':ref.setdefault(r['subject'].removeprefix('sub-'),{})[r['state']]=r
    jobs=[]
    for sid,refs in ref.items():
        p=OUT/'legacy'/f'{sid}.json'
        if not p.exists() or not all(s in existing(sid,m['inputs'][sid]) for s in STATES):jobs.append(sid)
    print(f'Legacy original-input replay and new-input calibration: {len(jobs)} subjects',flush=True)
    t=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs={pool.submit(legacy_worker,sid,ref[sid],m['inputs'][sid]):sid for sid in jobs}
        for i,f in enumerate(as_completed(fs),1):f.result();print(f'Legacy calibrated {i}/{len(jobs)}: {fs[f]}',flush=True)
    checks=[q for sid in ref for q in json.loads((OUT/'legacy'/f'{sid}.json').read_text())['checks']]
    summary=dict(status='passed',original_models=len(checks),new_models=len(checks),elapsed_seconds=time.monotonic()-t,
        maximum_error_bits=max(v for q in checks for k,v in q['errors'].items() if k.endswith('_bits')),
        maximum_regression_relative_error=max(q['regression_relative_error'] or 0 for q in checks),
        maximum_roi_affine_error=max(q['roi_affine_error'] for q in checks),
        new_minus_old_mean_system_bits={s:float(np.mean([q['new_minus_old_system_bits'] for q in checks if q['state']==s])) for s in STATES})
    dump(OUT/'legacy_gate.json',summary);print(json.dumps(summary),flush=True)


def window_fingerprint(m,sid,s):
    return digest(dict(config=m['config_hash'],rest_array=m['inputs'][sid]['array_hashes']['REST'],task=s,length=LENGTHS[s],starts=evenly_spaced_window_starts(1200,LENGTHS[s],12).tolist()))


def window_worker(sid,states,m=None):
    if m is None:m=WINDOW_MANIFEST
    todo=[]
    for s in states:
        p=OUT/'windows'/f'{sid}_{s}.json';fp=window_fingerprint(m,sid,s)
        if p.exists() and p.with_suffix('.npz').exists() and json.loads(p.read_text()).get('fingerprint')==fp:continue
        todo.append(s)
    if not todo:return 0
    raw=read_input(m['inputs'][sid],['REST'])['REST'];groups=core.load_yeo7_groups(core.LABELS,expected_parcels=1000)
    for s in todo:
        starts=evenly_spaced_window_starts(1200,LENGTHS[s],12);rows=[];transitions=[];noises=[];ei=[]
        for start in starts:
            x=raw[start:start+LENGTHS[s]];r,a=core.fit_series(x,x,groups,sid,s,window_fingerprint(m,sid,s));rows.append(r)
            transitions.append(a[f'{s}__transition']);noises.append(a[f'{s}__noise']);ei.append([r['coalition_ei_bits'][name] for name in rows[0]['coalition_ei_bits']])
        qc=[r['numerical_audit']['all_candidate_syn'] for r in rows]
        result=dict(subject=sid,task=s,fingerprint=window_fingerprint(m,sid,s),window_starts=starts.tolist(),n_windows=12,
            system_xi_bits=float(np.mean([r['system_xi_bits'] for r in rows])),cross_network_xi_bits=float(np.mean([r['cross_network_xi_bits'] for r in rows])),
            order_mass_bits=np.mean([r['order_mass_bits'] for r in rows],axis=0).tolist(),order_share=np.mean([r['order_share'] for r in rows],axis=0).tolist(),
            network_percent=np.mean([r['network_percent'] for r in rows],axis=0).tolist(),
            window_system_xi_bits=[r['system_xi_bits'] for r in rows],
            numerical_audit=dict(minimum_candidate_syn_bits=min(q['minimum_bits'] for q in qc),candidate_count=sum(q['checked_count'] for q in qc),near_zero_negative_count=sum(q['near_zero_negative_count'] for q in qc),significant_negative_count=sum(q['significant_negative_count'] for q in qc),tolerance_bits=core.TOL),
            maximum_closure_error_bits=max(abs(v) for r in rows for v in r['identity_errors_bits'].values()),
            covariance_floor_count=sum(r['covariance_floor']['affected_eigenvalue_count'] for r in rows))
        p=OUT/'windows'/f'{sid}_{s}.npz';p.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(p,starts=starts,transition=transitions,noise=noises,coalition_ei_bits=ei,coalition_names=list(rows[0]['coalition_ei_bits']),system_xi_bits=[r['system_xi_bits'] for r in rows],order_mass_bits=[r['order_mass_bits'] for r in rows],network_percent=[r['network_percent'] for r in rows])
        dump(p.with_suffix('.json'),result)
    return len(todo)*12


def run_models(m,workers,phase):
    if not (OUT/'legacy_gate.json').exists():raise ValueError('Legacy gate must pass first')
    subjects=sorted(m['inputs'])
    if phase=='smoke':subjects=m['common'][:20]
    if phase=='windows':subjects=sorted(m['window_jobs'])
    started=time.monotonic();total=0
    pool_kwargs=dict(max_workers=workers)
    if phase=='windows':pool_kwargs.update(initializer=init_window_worker,initargs=(m,))
    with ProcessPoolExecutor(**pool_kwargs) as pool:
        fs={}
        for sid in subjects:
            if phase=='windows':f=pool.submit(window_worker,sid,m['window_jobs'][sid])
            else:f=pool.submit(subject_worker,sid,m['inputs'][sid])
            fs[f]=sid
        for i,f in enumerate(as_completed(fs),1):
            total+=f.result()
            if i==1 or i%25==0 or i==len(fs):print(f'{phase}: {i}/{len(fs)} subject batches; {total} new fits; {time.monotonic()-started:.1f}s',flush=True)
    result=dict(phase=phase,new_models=total,elapsed_seconds=time.monotonic()-started,subjects=len(subjects))
    if phase=='smoke':
        t=time.monotonic()
        for sid in subjects:window_worker(sid,m['window_jobs'][sid],m)
        result['matched_smoke_seconds']=time.monotonic()-t;result['matched_smoke_subjects']=len(subjects)
    dump(OUT/f'{phase}_completion.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','legacy','smoke','main','windows']);p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.phase=='prepare':prep();return
    m=json.loads((OUT/'manifest.json').read_text())
    for name,h in m['implementation'].items():
        if sha(ROOT/name)!=h:raise ValueError(f'Implementation changed after manifest: {name}; prepare again')
    if args.phase=='legacy':run_legacy(m,args.workers)
    else:run_models(m,args.workers,args.phase)


if __name__=='__main__':main()
