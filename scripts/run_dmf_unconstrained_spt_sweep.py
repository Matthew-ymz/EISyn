#!/usr/bin/env python3
"""Paired wide DMF coupling sweep, preserving actual ROI core memberships."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.analyze_dmf_unconstrained_spt import (
    ROIXiOracle,TOL,make_selector,retained_core,weakest_split,node_record,from_record,
)
from scripts.run_dmf_paired_spt_pilot import paired_sources,noise_seed
from scripts.analyze_dmf_critical_phi_hierarchy_topology import conditional_source_covariance
from scripts.run_dmf_diffusive_fullstate_control import rollout
from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module,standardize
from scripts.spt import SPTConfig,build_spt,pairwise_syn_affinity


def scan_grid():
    return np.array(sorted(set(np.round(np.arange(0,3.00001,.1),2)) |
                           set(np.round(np.arange(1.1,1.70001,.02),2))))


def build_record(conditional, *, seed, g, extra=256):
    begin=time.perf_counter(); oracle=ROIXiOracle(conditional)
    affinity,pair_zeros=pairwise_syn_affinity(oracle,oracle.count,tolerance=TOL)
    gaps=[]
    result=build_spt(tuple(range(oracle.count)),oracle,config=SPTConfig(syn_tolerance=TOL),
                     candidate_selector=make_selector(affinity,oracle,extra,gaps))
    if abs(result.closure_error)>TOL:
        raise RuntimeError(f'Closure failed: seed={seed}, G={g}, error={result.closure_error}')
    cores={str(limit):retained_core(result.root,limit) for limit in (5,10,20)}
    for limit in ('5','10'):
        cores[limit]['weakest_split']=weakest_split(oracle,cores[limit]['members'])
    return {'seed':int(seed),'G':float(g),'extra_candidates':int(extra),'cross_roi_xi_bits':result.root.xi_value,
            'closure_error_bits':result.closure_error,'audit':asdict(result.audit),
            'pair_tolerance_zero_count':pair_zeros,'xi_tolerance_zero_count':oracle.tolerance_zero_count,
            'cores':cores,'candidate_gaps':gaps,'tree':node_record(result.root),
            'elapsed_seconds':time.perf_counter()-begin}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('smoke','full'),default='smoke')
    p.add_argument('--source',type=Path,default=ROOT/'results/dmf_schaefer100/source/group_mean_native_mean_rate.npz')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/dmf_schaefer100/unconstrained_spt_wide')
    p.add_argument('--extra-candidates',type=int,default=256)
    p.add_argument('--g-values',type=str)
    p.add_argument('--seeds',type=str)
    p.add_argument('--no-aggregate-covariance',action='store_true',help='Keep per-condition caches only in a sweep shard')
    args=p.parse_args()
    gs=scan_grid() if args.mode=='full' else np.array([0.,1.3,3.])
    seeds=np.arange(3,11) if args.mode=='full' else np.array([4])
    if args.g_values:
        gs=np.array(sorted(set(float(v) for v in args.g_values.split(','))))
    if args.seeds:
        seeds=np.array([int(v) for v in args.seeds.split(',')])
    if args.extra_candidates<=0 or not np.isfinite(gs).all() or gs.min()<0 or gs.max()>3:
        p.error('Use positive candidate budget and finite G values in [0,3]')
    with np.load(args.source) as a:
        sc=a['connectivity'].copy(); allg=a['G']; jfic=a['j_fic'].copy()
    with np.load(ROOT/'results/dmf_schaefer100/full/critical_yeo7.npz',allow_pickle=True) as a:
        labels=a['region_labels'].copy()
    ref=np.flatnonzero(np.isclose(allg,1.))
    if sc.shape!=(100,100) or len(ref)!=1 or not np.allclose(jfic,jfic[int(ref[0])],atol=0,rtol=0):
        raise ValueError('Require original Schaefer100 SC and G=1 calibrated, frozen JFIC')
    jf=jfic[int(ref[0])]; args.output_dir.mkdir(parents=True,exist_ok=True)
    covdir=args.output_dir/'covariance'; treedir=args.output_dir/'trees'
    covdir.mkdir(exist_ok=True); treedir.mkdir(exist_ok=True)
    simconfig={'version':1,'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
               'sample_count':2048,'support':[.3,.7],'horizon':300,'dt':.001,'sigma':.01,'ridge':1e-6,
               'boundary':'none','JFIC_reference_G':1.,'source_seed_offset':31000,'noise_seed_offset':31017,
               'paired_inputs_across_G':True,'paired_noise_across_G':True}
    searchconfig={'version':1,'exact_max_size':10,'extra_candidates':args.extra_candidates,
                  'include_all_singletons':True,'objective':'raw residual','tolerance_bits':TOL}
    (args.output_dir/'contract.json').write_text(json.dumps({'simulation':simconfig,'search':searchconfig,
        'G':gs.tolist(),'seeds':seeds.tolist(),'estimator':'Gaussian conditional covariance, planned high-dimensional exception',
        'core_definition':'Follow child with higher cross-ROI Xi; first coalition at 2..10 ROI; C5/C20 sensitivity',
        'ROI_label_status':'inferred atlas ordering, inherited from connectivity preparation audit',
        'phase_labels':'Require independent dynamical evidence; Xi peak alone is not criticality'},indent=2))
    dmf=load_dmf_module(); params=dmf.DMFParameters(t_total=1,burn_in=0,dt=.001,sigma=.01)
    cov=np.empty((len(seeds),len(gs),200,200)); cross=np.empty((len(seeds),len(gs)))
    fine=np.empty_like(cross); within=np.empty_like(cross); records=[]; started=time.perf_counter()
    # G-first schedule obtains all seeds at each G before moving to its neighbor.
    sources={}
    for seed in seeds:
        se,si=paired_sources(seed,2048,100); x=np.concatenate((se,si),axis=1)
        sources[int(seed)]=(se,si,standardize(x)[0],hashlib.sha256(x.tobytes()).hexdigest())
    for gi,g in enumerate(gs):
        for si,seed in enumerate(seeds):
            se,inh,x,digest=sources[int(seed)]; begin=time.perf_counter()
            path=covdir/f'seed{seed:02d}_G{g:.2f}.npz'; reuse=False
            if path.exists():
                with np.load(path) as old:
                    reuse=json.loads(str(old['config_json'].item()))==simconfig and str(old['input_sha256'].item())==digest
                    if reuse:
                        conditional=old['conditional_covariance'].copy()
            if not reuse:
                te,ti=rollout(dmf,se,inh,connectivity=sc,coupling_g=float(g),j_fic=jf,parameters=params,
                              mode='direct',state_boundary='none',horizon=300,rng=np.random.default_rng(noise_seed(seed)))
                target=np.concatenate((te,ti),axis=1)
                if not np.isfinite(target).all():
                    raise RuntimeError(f'Nonfinite target: seed={seed}, G={g}')
                _,conditional,_=conditional_source_covariance(x,standardize(target)[0],ridge=1e-6)
                np.savez_compressed(path,conditional_covariance=conditional,config_json=json.dumps(simconfig),
                                    input_sha256=digest,seed=seed,G=g,elapsed_seconds=time.perf_counter()-begin,
                                    target_min=target.min(),target_max=target.max())
            treepath=treedir/f'seed{seed:02d}_G{g:.2f}.json'
            record=json.loads(treepath.read_text()) if treepath.exists() else None
            if record is None or record.get('simulation_config')!=simconfig or record.get('search_config')!=searchconfig:
                record=build_record(conditional,seed=seed,g=g,extra=args.extra_candidates)
                record.update(simulation_config=simconfig,search_config=searchconfig)
                treepath.write_text(json.dumps(record,separators=(',',':'),allow_nan=False))
            cov[si,gi]=conditional; cross[si,gi]=record['cross_roi_xi_bits']; records.append(record)
            record['covariance_path']=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            oracle=ROIXiOracle(conditional)
            fine[si,gi]=(np.log(np.diag(conditional)).sum()-oracle.logdet(tuple(range(200))))/(2*math.log(2))
            within[si,gi]=(np.log(np.diag(conditional)).sum()-oracle.block_logdet.sum())/(2*math.log(2))
            if abs(fine[si,gi]-cross[si,gi]-within[si,gi])>TOL:
                raise RuntimeError('Scalar/ROI budget failed to close')
            print(f'done {len(records)}/{len(gs)*len(seeds)} seed={seed} G={g:.2f}: '
                  f'Xi={cross[si,gi]:.6f}, C10={record["cores"]["10"]["members"]}; '
                  f'{time.perf_counter()-begin:.2f}s',flush=True)
    if not args.no_aggregate_covariance:
        np.savez_compressed(args.output_dir/'paired_covariance.npz',G=gs,seeds=seeds,region_labels=labels,
                            conditional_covariance=cov,cross_roi=cross,fine_phi=fine,within_roi=within,
                            paired_inputs_across_G=True,paired_noise_across_G=True)
    (args.output_dir/'summary.json').write_text(json.dumps({'G':gs.tolist(),'seeds':seeds.tolist(),
        'labels':labels.tolist(),'records':records,'elapsed_seconds':time.perf_counter()-started,
        'simulation_config':simconfig,'search_config':searchconfig,'status':'complete'},separators=(',',':'),allow_nan=False))
    print(f'COMPLETE: {len(records)} conditions, {time.perf_counter()-started:.1f}s',flush=True)


if __name__=='__main__':
    main()
