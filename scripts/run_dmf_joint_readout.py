#!/usr/bin/env python3
"""Execute the new plan's fixed-G k=2/4 readout pilot, never a formal sweep."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
from scipy.stats import spearmanr
from scripts.dmf_joint_readout import (CommonTargetGame, TOL_NATS, audit_nonnegative,
    exact_spt, exact_shapley, label_codes, fit_decoder, posterior, decoder_metrics, paired_effect)
from scripts.dmf_response_benchmark import (fit_affine_joint, relax, simulate_sources,
    MIOracle, score)
from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module

BASE = ROOT / 'results/dmf_schaefer100/joint_readout_pilot'
OLD = ROOT / 'results/dmf_schaefer100/response_benchmark_pilot'
SOURCE = ROOT / 'results/dmf_schaefer100/source/group_mean_native_mean_rate.npz'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


class DiscreteParityGame:
    """Exact enumeration of an independent binary intervention with noisy XOR."""
    def __init__(self, n, noise):
        self.n, self.noise = n, noise
        self.source = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1)
        self.target = self.source.sum(1) % 2

    def u(self, members):
        members = tuple(sorted(members))
        if not members:
            return 0.
        codes = self.source[:, members] @ (1 << np.arange(len(members)))
        joint = np.zeros((1 << len(members), 2))
        for i, z in enumerate(self.target):
            joint[codes[i], z] += (1-self.noise)/len(codes)
            joint[codes[i], 1-z] += self.noise/len(codes)
        marginal = joint.sum(1, keepdims=True) @ joint.sum(0, keepdims=True)
        good = joint > 0
        return float(np.sum(joint[good] * np.log(joint[good]/marginal[good])))


def analytic_controls():
    records = []
    for n in (2, 3, 4, 6, 8):
        for noise in (0., .1):
            g = DiscreteParityGame(n, noise); members = tuple(range(n))
            whole = g.u(members)
            tree = exact_spt(g, members); attribution = exact_shapley(g, members)
            pair = sum(g.u((i,j)) for i in members for j in members if i < j)
            entropy = 0. if noise == 0 else -noise*np.log(noise)-(1-noise)*np.log(1-noise)
            if abs(whole-(np.log(2)-entropy)) > 1e-12:
                raise ArithmeticError('Exact noisy-XOR MI failed')
            audit_nonnegative([whole, pair, *attribution['cross_shapley_nats']], 1e-12)
            records.append(dict(n=n, bit_flip_probability=noise, xi_nats=whole,
                pair_syn_sum_nats=pair, shapley=attribution, spt=tree,
                bayes_joint_mode_accuracy=(1-noise)*2**(1-n),
                bayes_factorized_mode_accuracy=2**(-n),
                note='Exact discrete distribution; no continuous estimator. Whole EI also detects XOR.'))
    return records


def freeze_pool(sc, seed, count):
    rng = np.random.default_rng(seed)
    pools = {}
    for k in (2, 4):
        selected = []
        while len(selected) < count:
            origin = 'uniform' if len(selected) < count//2 else 'SC'
            if origin == 'uniform':
                member = tuple(sorted(rng.choice(len(sc), k, replace=False).tolist()))
            else:
                anchor = int(rng.integers(len(sc)))
                weights = sc[anchor].copy(); weights[anchor] = 0
                weights /= weights.sum()
                member = tuple(sorted([anchor, *rng.choice(len(sc), k-1, replace=False, p=weights).tolist()]))
            if member not in [tuple(s['members']) for s in selected]:
                selected.append(dict(members=list(member), origin=origin))
        pools[str(k)] = selected
    return pools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=int, default=1024)
    parser.add_argument('--validation', type=int, default=512)
    parser.add_argument('--test', type=int, default=1024)
    parser.add_argument('--candidates', type=int, default=8)
    parser.add_argument('--output', type=Path, default=BASE)
    args = parser.parse_args()
    if min(args.train,args.validation,args.test) < 32 or args.candidates < 2 or args.candidates % 2:
        parser.error('Use >=32 samples per split and an even candidate count >=2')
    base = args.output.resolve(); base.mkdir(parents=True, exist_ok=True)
    old = json.loads((OLD/'contract.json').read_text())
    if digest(SOURCE) != old['source_sha256']:
        raise ValueError('Source/JFIC cache provenance changed')
    for path, sha in old['implementation_sha256'].items():
        if path in ('scripts/dmf_response_benchmark.py', 'exp/brain/dmf_fig6.py', 'exp/TM/transport_map_density.py'):
            if digest(ROOT/path) != sha:
                raise ValueError(f'Scoring cache implementation changed: {path}')
    with np.load(SOURCE) as archive:
        sc = archive['connectivity'].copy(); jf_all = archive['j_fic'].copy()
        index = np.flatnonzero(np.isclose(archive['G'], 1.))[0]; jf = jf_all[index]
    if not np.array_equal(jf_all,np.broadcast_to(jf,jf_all.shape)):
        raise ValueError('Expected fixed JFIC calibrated at G=1')
    dmf = load_dmf_module(); p = dmf.DMFParameters(**old['dmf_parameters'])
    if (old['G'],old['horizon_steps'],old['halfwidth'],p.dt,p.sigma) != (1.3,300,.02,.001,.01):
        raise ValueError('Existing scoring cache does not match declared pilot support')
    pools = freeze_pool(sc, 32001, args.candidates)
    sizes = dict(train=args.train, validation=args.validation, test=args.test)
    seeds = dict(train=[32101,32102],validation=[32201,32202],test=[32301,32302])
    implementation = {s:digest(ROOT/s) for s in ('scripts/dmf_joint_readout.py','scripts/run_dmf_joint_readout.py',
                       'scripts/dmf_response_benchmark.py','exp/brain/dmf_fig6.py','exp/TM/transport_map_density.py')}
    contract = dict(stage='fixed G, k=2/4 readout smoke; exploratory only',G=1.3,
        n_roi=100,n_scalar=200,target='all 200 future E/I coordinates', horizon_ms=300,
        dt_seconds=p.dt,sigma=p.sigma,halfwidth=.02,prior='independent uniform full-system box',
        pulse='sham only; labels are signs of actual initial E offsets, not injected current',
        source_order='E[0:100], I[100:200]',roi_ids='zero-based in machine outputs, one-based in report',
        candidates=pools,candidate_seed=32001,split_sizes=sizes,split_seeds=seeds,
        scoring_cache=str(OLD/'score_seed901.npz'),scoring_cache_sha256=digest(OLD/'score_seed901.npz'),
        source_sha256=digest(SOURCE),implementation_sha256=implementation,
        estimator='one full-system affine TM; Gaussian prior moments and residual approximation',
        estimator_limits='Box prior Gaussianized for scores; actual task uses uniform box. Not nonlinear evidence.',
        syn_tolerance_nats=TOL_NATS,decoders=['factorized','conditional_pairwise','joint'],
        decoder_features='200 future states standardized using decoder training split only, plus bias',
        decoder_label_basis='orthogonal Walsh terms through orders 1, 2, k; linear future-dependent energies',
        decoder_penalties=[.001,.01,.1],decoder_selection='minimum validation negative log likelihood',
        optimizer='L-BFGS-B, maxiter=300, ftol=1e-10, gtol=1e-5',
        selection_methods=['cross_roi_u','fine_xi_v','pair_syn_sum','whole_ei','o_increment',
             'native_pair_phi_r','native_weak_phi_r','native_phi_r_lu','native_wms','internal_sc','validation_Q','random'],
        selection_rule='stable first maximum, computed before accessing final test; random exact pool mean',
        allowed_target_difference='Native Phi-R/WMS use corresponding coalition futures; others B=V',
        stopping_rule='single scoring cache and independent train/validation/test; no full sweep or formal power claim',
        deferred=['global SPT search','global Shapley','G and k sweeps','directional matched interventions','real-brain transfer'],
        literature=dict(parent='P6UJCVG8',attachment='DXGC7JEA',title='Emergent hierarchical organization of causal interactions in complex systems',
            attachment_added='2026-10-02',version='only available 19-page main text, no explicit version/date, supplementary absent',
            fulltext_read='Methods (5)-(12), Brain, Discussion'),
        platform=platform.platform(),versions=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__))
    fingerprint = hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    contract['fingerprint'] = fingerprint
    summary_path = base/'summary.json'
    if summary_path.exists() and json.loads(summary_path.read_text()).get('fingerprint') == fingerprint:
        print(f'Reusing completed matching pilot: {base}',flush=True); return
    write(base/'contract.json',contract)  # Freeze before any new scores/labels.
    start = time.perf_counter()
    controls = analytic_controls(); write(base/'analytic_controls.json',controls)
    se,si,rates,baseline_diag = relax(dmf,sc,jf,1.3,p,901,seconds=10.,sigma=0.)
    reference = np.r_[se[0],si[0]]
    drift = float(np.max(np.abs(rates[-100:].mean(0)-rates[-200:-100].mean(0))))
    if drift > .05:
        raise ArithmeticError(f'Unstable baseline: {drift} Hz')
    with np.load(OLD/'score_seed901.npz') as archive:
        score_x = archive['source'].copy(); score_y = archive['target'].copy(); cov = archive['covariance'].copy()
    expected = reference + np.random.default_rng(901000+1).uniform(-.02,.02,size=score_x.shape)
    if not np.array_equal(expected,score_x):
        raise ValueError('Cached source differs from declared reference/support/seed')
    if not np.allclose(fit_affine_joint(score_x,score_y,.02),cov,atol=1e-12,rtol=1e-12):
        raise ValueError('Cached common density cannot be reconstructed')
    game = CommonTargetGame(cov); totals = game.totals(); native = MIOracle(cov)
    scoring_started = time.perf_counter(); candidate_scores = {}; coalition_diagnostics = []
    for k_text,pool in pools.items():
        k = int(k_text); records = []
        for candidate in pool:
            s = tuple(candidate['members'])
            records.append(dict(cross_roi_u=game.u(s),fine_xi_v=game.v(s),
                pair_syn_sum=sum(game.u((i,j)) for i,j in __import__('itertools').combinations(s,2)),
                whole_ei=game.ei(s),o_increment=(k-1)*game.ei(s)-sum(game.ei(tuple(j for j in s if j != i)) for i in s),
                native_pair_phi_r=score(native,s,'pair_phi_r')*np.log(2),
                native_weak_phi_r=score(native,s,'weak_phi_r')*np.log(2),
                native_phi_r_lu=score(native,s,'phi_r_lu')*np.log(2),
                native_wms=score(native,s,'wms')*np.log(2),internal_sc=float(sc[np.ix_(s,s)].sum())))
            coalition_diagnostics.append(dict(k=k,members=list(s),spt=exact_spt(game,s),shapley=exact_shapley(game,s)))
        candidate_scores[k_text] = records
    scoring_seconds = time.perf_counter()-scoring_started
    syn_audit = audit_nonnegative(game.audited)
    write(base/'scores.json',dict(totals=totals,candidates=candidate_scores,coalition_diagnostics=coalition_diagnostics,audit=syn_audit))
    samples = {}; timings = {}; diagnostics = {}; reuse = {}
    for split,count in sizes.items():
        path = base/f'{split}.npz'
        split_start = time.perf_counter()
        x = reference + np.random.default_rng(seeds[split][0]).uniform(-.02,.02,size=(count,200))
        if ((x<0)|(x>1)).any():
            raise ArithmeticError('Initial support outside state domain')
        if path.exists():
            with np.load(path) as archive:
                valid = str(archive['fingerprint'].item()) == fingerprint and np.array_equal(archive['source'],x)
                if valid:
                    y = archive['target'].copy(); diag = json.loads(str(archive['diagnostics'].item()))
            reuse[split] = bool(valid)
        else:
            reuse[split] = False
        if not reuse[split]:
            y,diag = simulate_sources(dmf,x,sc,jf,1.3,p,seeds[split][1])
            np.savez_compressed(path,source=x,target=y,reference=reference,fingerprint=fingerprint,
                               diagnostics=json.dumps(diag))
        if diag['outside_state_count'] or diag['abnormal_rate_count']:
            raise ArithmeticError(f'Invalid {split} trajectory: {diag}')
        samples[split] = (x,y); diagnostics[split] = diag
        timings[split] = time.perf_counter()-split_start
        print(f'{split}: {count} independent full-system trajectories in {timings[split]:.2f}s; reused={reuse[split]}',flush=True)
    hashes = {s:hashlib.sha256(x.tobytes()).hexdigest() for s,(x,y) in samples.items()}
    if len(set(hashes.values())) != 3:
        raise ArithmeticError('Splits are not independent')
    mu = samples['train'][1].mean(0); scale = samples['train'][1].std(0,ddof=1)
    features = {s:np.c_[(y-mu)/scale,np.ones(len(y))] for s,(x,y) in samples.items()}
    # A = training/validation phase. All methods' winners frozen before any test predictions.
    fitted = {}; validation = {}; model_records = []; selection = {}
    for k_text,pool in pools.items():
        k = int(k_text); validation[k_text] = []
        for ci,candidate in enumerate(pool):
            labels = {s:label_codes(x,reference,candidate['members'])[0] for s,(x,y) in samples.items() if s != 'test'}
            family_best = {}; val_metrics = {}
            for family,order in (('factorized',1),('conditional_pairwise',2),('joint',k)):
                if family == 'joint' and k == 2:
                    family_best[family] = family_best['conditional_pairwise']
                    val_metrics[family] = val_metrics['conditional_pairwise']
                    continue
                options = []
                for penalty in contract['decoder_penalties']:
                    model = fit_decoder(features['train'],labels['train'],k,order,penalty)
                    metrics = decoder_metrics(posterior(model,features['validation']),labels['validation'],k)
                    options.append((metrics['nll_nats'],penalty,model,metrics))
                    model_records.append(dict(k=k,candidate=ci,family=family,penalty=penalty,
                        validation_nll_nats=metrics['nll_nats'],**{a:model[a] for a in ('iterations','seconds','parameters','gradient_max')}))
                _,penalty,model,metrics = min(options,key=lambda t:t[:2])
                family_best[family] = model; val_metrics[family] = metrics
            fitted[(k,ci)] = family_best
            validation[k_text].append(paired_effect(val_metrics['joint']['correct'],val_metrics['factorized']['correct']))
            print(f'k={k}, candidate={ci+1}/{len(pool)}: validation Q={validation[k_text][-1]["difference"]:.4f}',flush=True)
        metrics = candidate_scores[k_text][0].keys()
        selection[k_text] = {m:int(np.argmax([r[m] for r in candidate_scores[k_text]])) for m in metrics}
        selection[k_text]['validation_Q'] = int(np.argmax([r['difference'] for r in validation[k_text]]))
    write(base/'selection.json',dict(selection=selection,validation=validation,model_fit_records=model_records))
    # B = final test phase. No selection, tuning or candidate change below this line.
    test = {}; selected = {}; probabilities = {}; gate = []
    for k_text,pool in pools.items():
        k = int(k_text); records = []; correct = []
        for ci,candidate in enumerate(pool):
            labels,bits = label_codes(samples['test'][0],reference,candidate['members'])
            model_metrics = {}; family_correct = {}
            for family,model in fitted[(k,ci)].items():
                prob = posterior(model,features['test'])
                met = decoder_metrics(prob,labels,k); family_correct[family] = met.pop('correct')
                model_metrics[family] = met
                probabilities[f'k{k}_c{ci}_{family}'] = prob
            q = paired_effect(family_correct['joint'],family_correct['factorized'])
            strong = paired_effect(family_correct['joint'],family_correct['conditional_pairwise'])
            records.append(dict(members=candidate['members'],origin=candidate['origin'],chance=2**(-k),
                Q=q,joint_minus_pairwise=strong,models=model_metrics,mode_counts=np.bincount(labels,minlength=1<<k).tolist()))
            correct.append(family_correct['joint'].astype(float)-family_correct['factorized'].astype(float))
        test[k_text] = records
        q_values = np.array([r['Q']['difference'] for r in records])
        q_sem = np.array([r['Q']['trajectory_sem'] for r in records])
        selected[k_text] = {m:dict(candidate=ci,Q=records[ci]['Q'],models=records[ci]['models'],
            paired_Q_minus_random=paired_effect(correct[ci],np.mean(correct,axis=0)))
            for m,ci in selection[k_text].items()}
        selected[k_text]['random'] = dict(Q=float(q_values.mean()))
        correlations = {m:(float(spearmanr([r[m] for r in candidate_scores[k_text]],q_values).statistic)
            if np.ptp([r[m] for r in candidate_scores[k_text]]) and np.ptp(q_values) else None)
            for m in candidate_scores[k_text][0]}
        gate.append(dict(k=k,Q_range=float(np.ptp(q_values)),max_trajectory_sem=float(q_sem.max()),
            descriptive_distinguishable=bool(np.ptp(q_values)>2*q_sem.max()),correlations=correlations,
            joint_mode_accuracy_range=[min(r['models']['joint']['mode_accuracy'] for r in records),max(r['models']['joint']['mode_accuracy'] for r in records)],
            note='Exploratory smoke gate; one fixed training/test split is not independent seed replication.'))
    np.savez_compressed(base/'test_posteriors.npz',**probabilities)
    total = time.perf_counter()-start
    summary = dict(fingerprint=fingerprint,status='pilot complete',contract_path=str(base/'contract.json'),
        totals=totals,syn_audit=syn_audit,test=test,selected=selected,gates=gate,
        baseline_drift_hz=drift,baseline_diagnostics=baseline_diag,trajectory_diagnostics=diagnostics,
        split_source_hashes=hashes,split_cache_reused=reuse,simulation_seconds=timings,
        scoring_seconds=scoring_seconds,decoder_fit_seconds=sum(r['seconds'] for r in model_records),
        elapsed_seconds=total,formal_go=False,
        limitations=['one scoring seed and one decoder split; SEM covers test trajectories only',
            'affine TM Gaussianizes the intervention prior; no general nonlinear conclusion',
            'conditional pairwise baseline includes all pairwise label terms with linear future-dependent energies',
            'global attribution/search, directional interventions and real-brain transfer not run'],
        next_step='Assess readable signal, candidate uncertainty and decoder limitations before a formal budget/sample-size decision.')
    write(summary_path,summary)
    print(f'Pilot complete in {total:.1f}s; formal_go=False',flush=True)


if __name__ == '__main__':
    main()
