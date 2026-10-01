#!/usr/bin/env python3
"""Prepare common-input/common-noise DMF covariances for a bounded SPT pilot.

The default is 24 conditions, not the proposed 440-condition wide sweep.
Each costly simulation is resumable and cached separately.
"""
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
from scripts.analyze_dmf_unconstrained_spt import ROIXiOracle, TOL
from scripts.analyze_dmf_critical_phi_hierarchy_topology import conditional_source_covariance
from scripts.run_dmf_diffusive_fullstate_control import rollout
from scripts.run_dmf_fixed_uniform_multihorizon import fixed_uniform_initial_state
from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module, standardize


def paired_sources(seed, count, dimension):
    se,si=fixed_uniform_initial_state(
        np.random.default_rng(int(seed)*100_000+31_000),sample_count=count,dimension=dimension,
        source_state='se_si',se_low=.3,se_high=.7,si_low=.3,si_high=.7)
    return se,si


def noise_seed(seed):
    return int(seed)*100_000+31_017


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'results/dmf_schaefer100/source/group_mean_native_mean_rate.npz')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/dmf_schaefer100/unconstrained_spt_paired_pilot')
    p.add_argument('--sample-count',type=int,default=2048)
    args=p.parse_args()
    if args.sample_count<401:
        p.error('Use at least 401 samples for the 200-dimensional covariance pilot')
    with np.load(args.source) as a:
        sc=a['connectivity'].copy(); allg=a['G']; jfic=a['j_fic'].copy()
    with np.load(ROOT/'results/dmf_schaefer100/full/critical_yeo7.npz',allow_pickle=True) as a:
        labels=a['region_labels'].copy()
    if sc.shape!=(100,100) or len(labels)!=100:
        raise ValueError('Pilot requires the existing 100-ROI connectivity and label order')
    ref=np.flatnonzero(np.isclose(allg,1.0))
    if len(ref)!=1 or not np.allclose(jfic,jfic[int(ref[0])],atol=0,rtol=0):
        raise ValueError('Expected JFIC calibrated at G=1.0 and frozen for the original scan')
    jf=jfic[int(ref[0])]
    gs=np.array([1.2,1.3,1.4]); seeds=np.arange(3,11)
    config={'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
            'sample_count':args.sample_count,'support':[.3,.7],'horizon':300,'dt':.001,'sigma':.01,
            'ridge':1e-6,'boundary':'none','JFIC_reference_G':1.0,'paired_inputs_across_G':True,
            'paired_noise_across_G':True,'source_seed_offset':31000,'noise_seed_offset':31017,
            'estimator':'Gaussian conditional covariance; high-dimensional exception as in experiment plan'}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'contract.json').write_text(json.dumps(config,indent=2))
    dmf=load_dmf_module(); params=dmf.DMFParameters(t_total=1,burn_in=0,dt=.001,sigma=.01)
    cov=np.empty((8,3,200,200)); cross=np.empty((8,3)); fine=np.empty((8,3)); within=np.empty((8,3))
    digests=[]; started=time.perf_counter()
    for si,seed in enumerate(seeds):
        se,inh=paired_sources(seed,args.sample_count,100)
        source=np.concatenate((se,inh),axis=1)
        digest=hashlib.sha256(source.tobytes()).hexdigest(); digests.append(digest)
        source_z=standardize(source)[0]
        for gi,g in enumerate(gs):
            path=args.output_dir/f'seed{seed:02d}_G{g:.2f}.npz'
            reuse=False
            if path.exists():
                with np.load(path) as old:
                    reuse=json.loads(str(old['config_json'].item()))==config and str(old['input_sha256'].item())==digest
                    if reuse:
                        conditional=old['conditional_covariance'].copy()
            if not reuse:
                begin=time.perf_counter()
                # Reset the same generator at every G; every step draws arrays of identical shape.
                te,ti=rollout(dmf,se,inh,connectivity=sc,coupling_g=float(g),j_fic=jf,parameters=params,
                              mode='direct',state_boundary='none',horizon=300,rng=np.random.default_rng(noise_seed(seed)))
                target=np.concatenate((te,ti),axis=1)
                if not np.isfinite(target).all():
                    raise RuntimeError(f'Nonfinite DMF output: seed={seed}, G={g}')
                _,conditional,_=conditional_source_covariance(source_z,standardize(target)[0],ridge=1e-6)
                np.savez_compressed(path,conditional_covariance=conditional,config_json=json.dumps(config),
                                    input_sha256=digest,seed=seed,G=g,elapsed_seconds=time.perf_counter()-begin,
                                    target_min=target.min(),target_max=target.max())
            oracle=ROIXiOracle(conditional)
            cov[si,gi]=conditional; cross[si,gi]=oracle.xi(range(100))
            full_logdet=oracle.logdet(tuple(range(200)))
            fine[si,gi]=(np.log(np.diag(conditional)).sum()-full_logdet)/(2*np.log(2))
            within[si,gi]=(np.log(np.diag(conditional)).sum()-oracle.block_logdet.sum())/(2*np.log(2))
            if abs(fine[si,gi]-cross[si,gi]-within[si,gi])>TOL:
                raise RuntimeError('Scalar/ROI covariance budget failed to close')
            print(f"{'reuse' if reuse else 'done'} seed={seed} G={g}: cross-ROI Xi={cross[si,gi]:.6f} bits",flush=True)
    output=args.output_dir/'paired_covariance.npz'
    np.savez_compressed(output,G=gs,seeds=seeds,region_labels=labels,conditional_covariance=cov,
                        cross_roi=cross,fine_phi=fine,within_roi=within,config_json=json.dumps(config),
                        source_sha256_by_seed=np.array(digests),paired_inputs_across_G=True,paired_noise_across_G=True)
    print(f'complete: {time.perf_counter()-started:.2f}s; {output}',flush=True)


if __name__=='__main__':
    main()
