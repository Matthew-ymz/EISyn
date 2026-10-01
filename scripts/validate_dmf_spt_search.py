#!/usr/bin/env python3
"""Increase nested candidate budget on frozen DMF covariances."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_dmf_unconstrained_spt_sweep import build_record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=ROOT/'results/dmf_schaefer100/unconstrained_spt_wide')
    p.add_argument('--g-values',default='0.8,1.3,2.0')
    p.add_argument('--seed',type=int,default=4)
    p.add_argument('--extra-candidates',type=int,default=1024)
    args=p.parse_args();out=args.input_dir/'search_validation';out.mkdir(exist_ok=True)
    rows=[]
    for g in map(float,args.g_values.split(',')):
        bases=list((args.input_dir/'shards').glob(f'seeds*/trees/seed{args.seed:02d}_G{g:.2f}.json'))
        if len(bases)!=1:raise RuntimeError(f'Need exactly one finished base tree at G={g}, seed={args.seed}')
        base=json.loads(bases[0].read_text())
        covpath=bases[0].parent.parent/'covariance'/bases[0].with_suffix('.npz').name
        with np.load(covpath) as a:conditional=a['conditional_covariance'].copy()
        digest=hashlib.sha256(conditional.tobytes()).hexdigest()
        path=out/f'seed{args.seed:02d}_G{g:.2f}_n{args.extra_candidates}.json'
        record=json.loads(path.read_text()) if path.exists() else None
        if record is None or record.get('covariance_sha256')!=digest:
            record=build_record(conditional,seed=args.seed,g=g,extra=args.extra_candidates)
            record['covariance_sha256']=digest
            path.write_text(json.dumps(record,separators=(',',':'),allow_nan=False))
        original=set(base['cores']['10']['members']);expanded=set(record['cores']['10']['members'])
        rows.append({'G':g,'seed':args.seed,'base_members':sorted(original),'expanded_members':sorted(expanded),
                     'entered':sorted(expanded-original),'left':sorted(original-expanded),
                     'base_extra_candidates':base['extra_candidates'],'expanded_extra_candidates':args.extra_candidates})
        print(f'G={g}, seed={args.seed}: entered={sorted(expanded-original)}, left={sorted(original-expanded)}',flush=True)
    (out/'summary.json').write_text(json.dumps({'records':rows,'status':'complete'},indent=2))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
