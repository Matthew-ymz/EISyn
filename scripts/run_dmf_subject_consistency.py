#!/usr/bin/env python3
"""Bounded, resumable 8-subject DMF consistency pilot; no parameter search."""
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
from scripts.dmf_subject_consistency import network_values, pair_values, roi_shapley, tree_record, TOL
from scripts.dmf_joint_readout import CommonTargetGame, audit_nonnegative
from scripts.dmf_response_benchmark import fit_affine_joint, simulate_sources
from scripts.run_dmf_paired_spt_pilot import paired_sources, noise_seed
from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module, standardize
from scripts.analyze_dmf_critical_phi_hierarchy_topology import conditional_source_covariance
from scripts.analyze_dmf_critical_phi_yeo7_hierarchy import yeo7_membership
from scripts.run_dmf_schaefer100_critical_horizon import stochastic_rate_trace

BASE = ROOT/'results/dmf_schaefer100/subject_consistency_pilot'
G = [0., .5, 1., 1.3, 1.6, 2.2, 3.]
SEEDS = [3, 4, 5]
ORGANIZATION_G = [0., 1., 1.3, 2.2]
VERSION = 'individual-sc-affine-tm-v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def prepare(output):
    source = ROOT/'results/dmf_schaefer100/source/group_mean_native_mean_rate.npz'
    paths = list((ROOT/'data/neuromodulator_receptor_sc_100/CON_SC_1mio').glob('sub-*.csv'))
    rows = []
    for path in sorted(paths):
        sc = np.loadtxt(path, delimiter=',')
        if sc.shape != (100, 100) or not np.isfinite(sc).all() or sc.min()<0 or not np.allclose(sc,sc.T,atol=1e-12,rtol=0):
            raise ValueError(f'Invalid SC: {path}')
        rows.append((float(np.linalg.eigvalsh(sc)[-1]), path.stem, path, sc))
    if len(rows) != 93:
        raise ValueError(f'Expected 93 subjects, got {len(rows)}')
    rows.sort(key=lambda row:(row[0],row[1]))
    chosen = [rows[i] for i in np.rint(np.linspace(0,92,8)).astype(int)]
    with np.load(source) as a:
        index = int(np.flatnonzero(np.isclose(a['G'],1.))[0])
        jf = a['j_fic'][index].copy()
        if not np.allclose(a['j_fic'],jf,atol=0,rtol=0):
            raise ValueError('Expected frozen reference JFIC')
        mean_sc = a['connectivity'].copy()
    labels = (ROOT/'results/dmf_schaefer100/schaefer100_labels.txt').read_text().splitlines()
    membership, groups, names = yeo7_membership(labels)
    if len(labels)!=100 or len(groups)!=7:
        raise ValueError('Expected 100 ROI and seven networks')
    ids = [row[1] for row in chosen] + ['group_mean_93']
    matrices = [row[3] for row in chosen] + [mean_sc]
    hashes = {row[1]:digest(row[2]) for row in chosen}
    hashes['group_mean_93'] = digest(source)
    contract = dict(version=VERSION, subject_ids=ids, subject_hashes=hashes,
        reference_source_sha256=digest(source), G=G, seeds=SEEDS, organization_G=ORGANIZATION_G,
        sample_count=2048, support=[.3,.7], horizon_steps=300, dt_seconds=.001,
        sigma=.01, ridge=1e-6, state_boundary='none', JFIC='93-person mean, calibrated G=1, frozen',
        estimator='one full-system affine TM; moment-matched diagonal Gaussian source prior; linear transition and Gaussian residual',
        units='nats', syn_tolerance_nats=TOL, source_seed_offset=31000, noise_seed_offset=31017,
        paired_inputs_and_noise_across_G_and_subjects=True,
        diagnostic_seeds=[103,104,105], diagnostic_steps=5000, diagnostic_burn_steps=3000,
        subject_selection='8 equally spaced ranks of native SC spectral radius; development sample',
        roi_label_status='inferred atlas order; inherited preparation audit',
        SPT=dict(exact_max_size=10, extra_random_candidates=256, include_all_singletons=True,
                 same_candidate_rule_for_both_methods=True, residual_objective='raw'),
        shapley=dict(antithetic_pairs=256, rng_seed=20261002, fixed_pilot_budget=True),
        manuscript=dict(parent_key='P6UJCVG8',attachment_key='DXGC7JEA',pages=19,
                        version_date='unavailable',locations='Methods Eqs.4-12; Brain/Fig.2',
                        supplementary_appendices='not available'),
        interpretation='exploratory fixed-reference-model organization; not independent model training or functional validation')
    output.mkdir(parents=True,exist_ok=True)
    path=output/'contract.json'
    if path.exists() and json.loads(path.read_text())!=contract:
        raise ValueError('Existing contract mismatch; use another output directory')
    if not path.exists():
        atomic_json(path,contract)
    input_path=output/'inputs.npz'
    if input_path.exists():
        with np.load(input_path) as a:
            if json.loads(str(a['contract_json']))!=contract:
                raise ValueError('Input cache mismatch')
            if not np.array_equal(a['connectivity'],np.asarray(matrices)) or not np.array_equal(a['j_fic'],jf):
                raise ValueError('Input data changed')
    else:
        temp_path=input_path.with_name(f'inputs.{os.getpid()}.tmp.npz')
        np.savez_compressed(temp_path,subject_ids=ids,connectivity=matrices,
            spectral_radius=[np.linalg.eigvalsh(sc)[-1] for sc in matrices],j_fic=jf,
            labels=labels,network_membership=membership,network_names=names,contract_json=json.dumps(contract))
        temp_path.replace(input_path)
    return contract,ids,matrices,jf,groups


def simulate(output, limit=None):
    contract,ids,matrices,jf,groups=prepare(output)
    dmf=load_dmf_module()
    p=dmf.DMFParameters(t_total=1,burn_in=0,dt=.001,sigma=.01)
    cache=output/'conditions';cache.mkdir(exist_ok=True)
    sources={s:np.concatenate(paired_sources(s,2048,100),1) for s in SEEDS}
    done=0
    start=time.perf_counter()
    for subject,sc in zip(ids,matrices):
        for g in G:
            for seed in SEEDS:
                key=f'{subject}_G{g:.2f}_seed{seed}'
                path=cache/(key+'.npz')
                if path.exists():
                    with np.load(path) as a:
                        if json.loads(str(a['contract_json']))!=contract:
                            raise ValueError(f'Cache contract mismatch: {key}')
                    continue
                t=time.perf_counter();x=sources[seed]
                y,diag=simulate_sources(dmf,x,sc,jf,g,p,noise_seed(seed))
                if diag['outside_state_count']:
                    raise ArithmeticError(f'DMF state violation: {key}: {diag}')
                cov=fit_affine_joint(x,y,.2,ridge=1e-6);game=CommonTargetGame(cov)
                metrics=game.totals()
                within,between,total,network_audit=network_values(game,groups)
                metrics['cross_network_nats']=total
                metrics['audit']=audit_nonnegative(game.audited)
                # Same-sample bridge to the old estimator, not a separate biological contrast.
                _,old,_=conditional_source_covariance(standardize(x)[0],standardize(y)[0],ridge=1e-6)
                metrics['legacy_xi_nats']=float(.5*(np.log(np.diag(old)).sum()-np.linalg.slogdet(old)[1]))
                inp=hashlib.sha256(x.tobytes()).hexdigest()
                np.savez_compressed(path,covariance=cov,conditional=game.conditional,
                    within_network_xi=within,between_network_shapley=between,
                    metrics_json=json.dumps(metrics),diagnostics_json=json.dumps(diag),
                    network_audit_json=json.dumps(network_audit),contract_json=json.dumps(contract),
                    source_sha256=inp,noise_seed=noise_seed(seed),elapsed_seconds=time.perf_counter()-t)
                done+=1
                print(f'done {key}: Xi={metrics["xi_nats"]:.4f} cross={metrics["cross_roi_nats"]:.4f} nats; {time.perf_counter()-t:.2f}s',flush=True)
                if limit and done>=limit:
                    return
    print(f'simulation complete: {len(ids)*len(G)*len(SEEDS)} conditions; new={done}; elapsed={time.perf_counter()-start:.1f}s',flush=True)


def diagnose(output):
    contract,ids,matrices,jf,groups=prepare(output)
    dmf=load_dmf_module();p=dmf.DMFParameters(dt=.001,sigma=.01)
    cache=output/'dynamics';cache.mkdir(exist_ok=True)
    for subject,sc in zip(ids,matrices):
        path=cache/(subject+'.npz')
        if path.exists():
            with np.load(path) as a:
                if json.loads(str(a['contract_json']))!=contract:
                    raise ValueError('Dynamics cache mismatch')
            continue
        means=np.empty((3,len(G)));susceptibility=np.empty_like(means)
        boundary=np.empty_like(means);start=time.perf_counter()
        for si,seed in enumerate(contract['diagnostic_seeds']):
            for gi,g in enumerate(G):
                rates,outside=stochastic_rate_trace(dmf,initial_se=np.full(100,p.init_se),
                    initial_si=np.full(100,p.init_si),connectivity=sc,coupling_g=g,
                    j_fic=jf,parameters=p,steps=5000,seed=seed)
                if not np.isfinite(rates).all() or rates.min()<0 or rates.max()>500 or outside:
                    raise ArithmeticError(f'Diagnostic invalid state: {subject}, G={g}, seed={seed}')
                late=rates[3000:];means[si,gi]=late.mean()
                susceptibility[si,gi]=100*np.var(late.mean(1),ddof=1)
                boundary[si,gi]=outside
        np.savez_compressed(path,mean_rate_hz=means,susceptibility=susceptibility,
                            boundary_fraction=boundary,contract_json=json.dumps(contract))
        print(f'dynamics {subject}: mean rates={means.mean(0).round(2).tolist()}; {time.perf_counter()-start:.1f}s',flush=True)


def organize(output,limit=None,subjects=None):
    contract,ids,matrices,jf,groups=prepare(output)
    if subjects:
        if set(subjects)-set(ids):
            raise ValueError('Unknown organization subject')
        selected=[(subject,sc) for subject,sc in zip(ids,matrices) if subject in subjects]
    else:
        selected=list(zip(ids,matrices))
    cache=output/'organization';cache.mkdir(exist_ok=True);done=0
    for subject,sc in selected:
        for g in ORGANIZATION_G:
            for seed in SEEDS:
                key=f'{subject}_G{g:.2f}_seed{seed}'
                path=cache/(key+'.npz')
                if path.exists():
                    with np.load(path) as a:
                        if json.loads(str(a['contract_json']))!=contract:
                            raise ValueError('Organization cache mismatch')
                    continue
                t=time.perf_counter()
                with np.load(output/'conditions'/(key+'.npz')) as a:
                    cov=a['covariance']
                game=CommonTargetGame(cov)
                affinity,pair_audit=pair_values(game)
                full,mcse,cross,shapley_audit=roi_shapley(game)
                trees={name:tree_record(game,affinity,pairwise=pair) for name,pair in [('multisource',False),('pairwise',True)]}
                audit=audit_nonnegative(game.audited)
                np.savez_compressed(path,roi_xi_shapley=full,roi_cross_shapley=cross,
                    roi_mc_se=mcse,roi_local_xi=game.local_xi,
                    whole_ei_shapley=full+game.scalar_ei[:100]+game.scalar_ei[100:],
                    pair_affinity=affinity,pair_shapley=affinity.sum(1)/2,
                    sc_strength=sc.sum(1),trees_json=json.dumps(trees),
                    pair_audit_json=json.dumps(pair_audit),shapley_audit_json=json.dumps(shapley_audit),
                    audit_json=json.dumps(audit),contract_json=json.dumps(contract))
                done+=1
                print(f'organization {key}: full core={trees["multisource"]["core_size"]}, pair core={trees["pairwise"]["core_size"]}; {time.perf_counter()-t:.2f}s',flush=True)
                if limit and done>=limit:
                    return
    print(f'organization complete: {len(selected)*len(ORGANIZATION_G)*len(SEEDS)} conditions',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['simulate','diagnose','organize','acquire','run'],required=True)
    parser.add_argument('--output-dir',type=Path,default=BASE)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--subjects',type=lambda s:s.split(','),help='Disjoint organization shard; simulation contract remains full')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase in ('run','acquire'):
            if args.limit:
                parser.error('--limit only applies to individual smoke phases')
            simulate(args.output_dir)
            diagnose(args.output_dir)
            if args.phase=='run':
                organize(args.output_dir)
                atomic_json(args.output_dir/'completed.json',dict(version=VERSION,
                    simulation_condition_count=189,organization_condition_count=108,
                    diagnostic_subject_count=9,status='complete'))
                print('pilot computation complete',flush=True)
            else:
                print('pilot acquisition complete',flush=True)
        else:
            options={'limit':args.limit} if args.phase!='diagnose' else {}
            if args.phase=='organize':options['subjects']=args.subjects
            elif args.subjects:parser.error('--subjects only applies to organize')
            {'simulate':simulate,'diagnose':diagnose,'organize':organize}[args.phase](args.output_dir,**options)


if __name__=='__main__':
    main()
