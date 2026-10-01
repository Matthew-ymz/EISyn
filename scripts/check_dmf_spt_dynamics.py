#!/usr/bin/env python3
"""Independent deterministic trajectory/tangent audit of DMF regime labels."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.run_dmf_paired_spt_pilot import paired_sources
from scripts.run_dmf_unconstrained_spt_sweep import scan_grid
from scripts.run_dmf_schaefer100_critical_horizon import transfer_derivative
from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module


def drift_and_tangent(dmf,se,si,ve,vi,sc,g,jf,p):
    ie=p.w_e*p.i0+p.w_plus*p.j_nmda*se+g*p.j_nmda*(sc@se)-jf*si
    ii=p.w_i*p.i0+p.j_nmda*se-si
    re=dmf.transfer_function(ie,gain=p.gain_e,threshold=p.threshold_e,shape=p.shape_e)
    ri=dmf.transfer_function(ii,gain=p.gain_i,threshold=p.threshold_i,shape=p.shape_i)
    de=transfer_derivative(dmf,ie,gain=p.gain_e,threshold=p.threshold_e,shape=p.shape_e)
    di=transfer_derivative(dmf,ii,gain=p.gain_i,threshold=p.threshold_i,shape=p.shape_i)
    dx=-se/p.tau_e+(1-se)*p.gamma_e*re
    dy=-si/p.tau_i+ri
    dvx=(-1/p.tau_e-p.gamma_e*re)*ve+(1-se)*p.gamma_e*de*(p.w_plus*p.j_nmda*ve+g*p.j_nmda*(sc@ve)-jf*vi)
    dvy=di*p.j_nmda*ve+(-1/p.tau_i-di)*vi
    return dx,dy,dvx,dvy,re


def trajectory_audit(dmf,sc,jf,p,g,seed,burn_steps=5000,measure_steps=10000):
    se,si=paired_sources(seed,1,100); se=se[0].copy();si=si[0].copy()
    vector=np.random.default_rng(int(seed)*100000+31033).normal(size=200)
    vector/=np.linalg.norm(vector); ve=vector[:100].copy();vi=vector[100:].copy()
    loggrowth=[]; rates=[]; drifts=[]
    for step in range(burn_steps+measure_steps):
        dx,dy,dvx,dvy,re=drift_and_tangent(dmf,se,si,ve,vi,sc,g,jf,p)
        se+=p.dt*dx; si+=p.dt*dy;ve+=p.dt*dvx;vi+=p.dt*dvy
        if (step+1)%10==0:
            norm=float(np.linalg.norm(np.concatenate((ve,vi))))
            if not np.isfinite(norm) or norm<=0:
                raise RuntimeError(f'Invalid tangent norm at G={g}, seed={seed}')
            ve/=norm;vi/=norm
            if step>=burn_steps:
                loggrowth.append(np.log(norm)); rates.append(re.mean())
                drifts.append(np.linalg.norm(np.concatenate((dx,dy))))
    blocks=np.array(loggrowth).reshape(10,-1).sum(axis=1)/(measure_steps*p.dt/10)
    return {'G':float(g),'seed':int(seed),'ftle_per_second':float(blocks.mean()),
            'last_half_ftle_per_second':float(blocks[5:].mean()),'block_ftle_per_second':blocks.tolist(),
            'deterministic_mean_rate_hz':float(np.mean(rates)),
            'deterministic_temporal_rate_sd_hz':float(np.std(rates)),
            'final_drift_norm':float(drifts[-1]),'final_state_min':float(min(se.min(),si.min())),
            'final_state_max':float(max(se.max(),si.max()))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/dmf_schaefer100/unconstrained_spt_wide/dynamics')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    source=ROOT/'results/dmf_schaefer100/source/group_mean_native_mean_rate.npz'
    with np.load(source) as a:
        sc=a['connectivity'];jf=a['j_fic'][0]
    dmf=load_dmf_module(); params=dmf.DMFParameters(dt=.001,sigma=0,t_total=15,burn_in=5)
    config={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'dt':.001,
            'sigma':0,'burn_seconds':5,'measurement_seconds':10,'renormalize_steps':10,
            'derivative':'existing central-difference transfer derivative','measurement':'finite-time Euler tangent exponent',
            'interpretation':'positive sustained growth is a chaos candidate; negative values do not prove all attractors nonchaotic'}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    jobs=[(4,g) for g in scan_grid()]
    jobs += [(s,g) for s in range(3,11) if s!=4 for g in (0.,.8,1.3,2.,3.)]
    if args.smoke: jobs=[(4,0.),(4,1.3),(4,3.)]
    records=[];start=time.perf_counter()
    for seed,g in jobs:
        path=args.output_dir/f'seed{seed:02d}_G{g:.2f}.json'
        record=json.loads(path.read_text()) if path.exists() else None
        if record is None or record.get('config')!=config:
            t=time.perf_counter();record=trajectory_audit(dmf,sc,jf,params,float(g),seed)
            record.update(config=config,elapsed_seconds=time.perf_counter()-t)
            path.write_text(json.dumps(record,separators=(',',':'),allow_nan=False))
        records.append(record)
        print(f'done {len(records)}/{len(jobs)} seed={seed} G={g:.2f}: '
              f'FTLE={record["ftle_per_second"]:.5f}, late={record["last_half_ftle_per_second"]:.5f}; '
              f'{record["elapsed_seconds"]:.2f}s',flush=True)
    (args.output_dir/'summary.json').write_text(json.dumps({'config':config,'records':records,
        'elapsed_seconds':time.perf_counter()-start,'status':'complete'},separators=(',',':'),allow_nan=False))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
