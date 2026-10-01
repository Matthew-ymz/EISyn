#!/usr/bin/env python3
"""4096-sample checks with exact 2048-sample intervention/noise prefixes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_dmf_paired_spt_pilot import paired_sources,noise_seed
from scripts.run_dmf_unconstrained_spt_sweep import build_record
from scripts.run_dmf_diffusive_fullstate_control import rollout
from scripts.analyze_dmf_critical_phi_hierarchy_topology import conditional_source_covariance
from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module,standardize


class NestedNoise:
    def __init__(self,seed,prefix_count=2048):
        self.prefix=np.random.default_rng(noise_seed(seed))
        self.suffix=np.random.default_rng(noise_seed(seed+1000))
        self.count=prefix_count
    def standard_normal(self,shape):
        if shape[0]!=2*self.count:raise ValueError('Unexpected nested sample shape')
        block=(self.count,*shape[1:])
        return np.concatenate((self.prefix.standard_normal(block),self.suffix.standard_normal(block)),axis=0)


def nested_sources(seed,count=2048,dimension=100):
    se,si=paired_sources(seed,count,dimension)
    extra_e,extra_i=paired_sources(seed+1000,count,dimension)
    return np.concatenate((se,extra_e)),np.concatenate((si,extra_i))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=ROOT/'results/dmf_schaefer100/unconstrained_spt_wide')
    p.add_argument('--seeds',default='3,4,5,6,7,8,9,10')
    p.add_argument('--g-values',default='0,0.8,1.28,1.32,1.8')
    args=p.parse_args();seeds=[int(v) for v in args.seeds.split(',')];gs=[float(v) for v in args.g_values.split(',')]
    out=args.input_dir/'sample_validation'/('seeds'+'_'.join(map(str,seeds)))
    out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/dmf_schaefer100/source/group_mean_native_mean_rate.npz'
    with np.load(source) as a:sc=a['connectivity'];jf=a['j_fic'][0]
    dmf=load_dmf_module();params=dmf.DMFParameters(dt=.001,sigma=.01,t_total=1,burn_in=0)
    records=[];start=time.perf_counter()
    for seed in seeds:
        se,si=nested_sources(seed);x=np.concatenate((se,si),axis=1)
        for g in gs:
            bases=list((args.input_dir/'shards').glob(f'seeds*/covariance/seed{seed:02d}_G{g:.2f}.npz'))
            if len(bases)!=1:raise RuntimeError(f'Expected completed 2048-sample cache: seed={seed}, G={g}')
            with np.load(bases[0]) as a:base=a['conditional_covariance'].copy()
            path=out/f'seed{seed:02d}_G{g:.2f}.npz';treepath=path.with_suffix('.json')
            if path.exists():
                with np.load(path) as a:
                    conditional=a['conditional_covariance'].copy();prefix_error=float(a['prefix_covariance_max_error'])
            else:
                te,ti=rollout(dmf,se,si,connectivity=sc,coupling_g=g,j_fic=jf,parameters=params,
                              mode='direct',state_boundary='none',horizon=300,rng=NestedNoise(seed))
                y=np.concatenate((te,ti),axis=1)
                if not np.isfinite(y).all():raise RuntimeError('Nonfinite 4096-sample target')
                _,prefix,_=conditional_source_covariance(standardize(x[:2048])[0],standardize(y[:2048])[0],ridge=1e-6)
                prefix_error=float(np.max(np.abs(prefix-base)))
                if prefix_error>1e-9:raise RuntimeError(f'2048-sample prefix changed: covariance error={prefix_error}')
                _,conditional,_=conditional_source_covariance(standardize(x)[0],standardize(y)[0],ridge=1e-6)
                np.savez_compressed(path,conditional_covariance=conditional,prefix_covariance_max_error=prefix_error,
                                    sample_count=4096,seed=seed,G=g)
            record=json.loads(treepath.read_text()) if treepath.exists() else None
            if record is None:
                record=build_record(conditional,seed=seed,g=g,extra=256)
                record.update(sample_count=4096,prefix_covariance_max_error=prefix_error)
                treepath.write_text(json.dumps(record,separators=(',',':'),allow_nan=False))
            records.append(record)
            print(f'done seed={seed} G={g}: C10={record["cores"]["10"]["members"]}; prefix error={prefix_error:.3g}',flush=True)
    (out/'summary.json').write_text(json.dumps({'status':'complete','records':records,
        'elapsed_seconds':time.perf_counter()-start,'nested_prefix_count':2048},separators=(',',':'),allow_nan=False))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
