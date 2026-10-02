#!/usr/bin/env python3
"""Execute approved stages A-C only; formal D requires a separate budget decision."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# These must be set before importing numpy in fresh timing workers.
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import scipy
from scipy.stats import spearmanr
from scripts.dmf_response_benchmark import (
    METHODS, LABELS, TOL, MIOracle, fit_affine_joint, score, full_phi_lu,
    check_syn, weaken_return, relax, responses, simulate_sources,
)
from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module

BASE = ROOT / 'results/dmf_schaefer100/response_benchmark_pilot'
SOURCE = ROOT / 'results/dmf_schaefer100/source/group_mean_native_mean_rate.npz'


def write_json(path, obj):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def algorithm_audit():
    maximum = dict(closure=0., scalar_full=0., xi_conditional=0., second_order=0.)
    for k in (2, 3, 4):
        for seed in (11, 12, 13, 14):
            rng = np.random.default_rng(seed)
            b = rng.normal(size=(2*k, 2*k)) * .15; d = np.eye(2*k)
            q = MIOracle(np.block([[d, b], [b.T, b.T @ b + np.eye(2*k)]])); a = tuple(range(k))
            value, closure, _ = full_phi_lu(q, a)
            maximum['closure'] = max(maximum['closure'], abs(closure))
            maximum['scalar_full'] = max(maximum['scalar_full'], abs(value-score(q, a, 'phi_r_lu')))
            maximum['xi_conditional'] = max(maximum['xi_conditional'], abs(score(q, a, 'xi')-score(q, a, 'xi_fast')))
            check_syn([score(q, a, 'xi')])
            if k == 2:
                maximum['second_order'] = max(maximum['second_order'], abs(value-score(q, a, 'pair_phi_r')))
    if max(maximum.values()) > TOL:
        raise ArithmeticError(maximum)
    return maximum


def freeze_candidates(sc):
    rng = np.random.default_rng(901)
    strengths = sc.sum(axis=1)
    sites = []
    # Pilot's two sites: left low-strength and right high-strength strata.
    for hemisphere, high in ((np.arange(50), False), (np.arange(50, 100), True)):
        median = np.median(strengths[hemisphere])
        eligible = hemisphere[strengths[hemisphere] >= median] if high else hemisphere[strengths[hemisphere] < median]
        sites.append(int(rng.choice(eligible)))
    pools = []; origins = []; ranges = []
    for s in sites:
        raw = []; kind = []; bvalues = []
        others = np.array([i for i in range(100) if i != s])
        for structural in (False, True):
            weights = sc[s, others] + sc[others, s]
            probabilities = (weights + weights.mean()*.1) / (weights + weights.mean()*.1).sum() if structural else None
            for _ in range(256):
                a = tuple(sorted([s, *rng.choice(others, 7, replace=False, p=probabilities).tolist()]))
                if a in raw:
                    continue
                raw.append(a); kind.append('SC' if structural else 'uniform')
                mask = np.zeros(100, bool); mask[list(a)] = True
                bvalues.append(float(sc[np.ix_(mask, ~mask)].sum()))
        low, high = np.quantile(bvalues, [.2, .8]); selected = []
        for origin in ('uniform', 'SC'):
            good = [i for i in range(len(raw)) if kind[i] == origin and low <= bvalues[i] <= high]
            selected.extend(rng.choice(good, 4, replace=False).tolist())
        pools.append([raw[i] for i in selected]); origins.append([kind[i] for i in selected])
        ranges.append([float(low), float(high)])
    return sites, np.asarray(pools), origins, ranges


def timing_worker(args):
    with np.load(BASE/'score_seed901.npz') as data:
        cov = data['covariance'].copy(); x = data['source'].copy(); y = data['target'].copy()
        halfwidth = float(data['halfwidth']); pools = data['candidates'].copy()
    method = args.method; k = args.k
    a = tuple(range(k)); candidates = [a] if args.layer in ('core','cold') else [tuple(v) for v in pools.reshape(-1, 8)]
    if args.layer not in ('core','cold') and k != 8:
        raise ValueError('Pool timing uses the actual frozen k=8 pools')
    score(MIOracle(cov), a, method)  # Identical warmup, discarded.
    if args.layer == 'cold':
        from scripts.dmf_response_benchmark import lattice, subsets
        lattice.cache_clear(); subsets.cache_clear()
    # Measure warm structure reuse but fresh MI work; never reuse result scores.
    begin = time.perf_counter(); cpu = time.process_time()
    if args.layer == 'raw':
        cov = fit_affine_joint(x, y, halfwidth)
    oracle = MIOracle(cov)
    values = [score(oracle, candidate, method) for candidate in candidates]
    np.argsort(-np.asarray(values), kind='stable')
    wall = time.perf_counter()-begin; cputime = time.process_time()-cpu
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        rss *= 1024
    print(json.dumps(dict(method=method, k=k, layer=args.layer, repeat=args.repeat,
                         wall_seconds=wall, cpu_seconds=cputime, peak_rss_bytes=rss,
                         unique_mi=len(oracle.cache), total_mi=oracle.requests,
                         repeats_mi=oracle.requests-len(oracle.cache),
                         atom_count={2:16,3:324,4:27556}.get(k, 0) if method=='phi_full' else 0,
                         partitions=2**(k-1)-1 if method=='weak_phi_r' else 0,
                         lu_subsets=2**k-1 if method=='phi_r_lu' else 0,
                         state='complete')))


def timing_pilot():
    tasks = []
    for k in (2, 3, 4, 6, 8):
        for method in (*METHODS, 'xi_fast', *(['phi_full'] if k<=4 else [])):
            for repeat in range(5):
                tasks.append((method, k, 'core', repeat))
        for method in ('phi_r_lu', *(['phi_full'] if k<=4 else [])):
            for repeat in range(5):
                tasks.append((method,k,'cold',repeat))
    for layer in ('pool', 'raw'):
        for method in (*METHODS, 'xi_fast'):
            for repeat in range(5):
                tasks.append((method, 8, layer, repeat))
    np.random.default_rng(903).shuffle(tasks)
    records = []
    for ti, (method, k, layer, repeat) in enumerate(tasks):
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--method', method,
                   '--k', str(k), '--layer', layer, '--repeat', str(repeat)]
        start = time.perf_counter()
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=60)
            if run.returncode:
                records.append(dict(method=method,k=k,layer=layer,repeat=repeat,state='failed',reason=run.stderr[-2000:]))
            else:
                record = json.loads(run.stdout.strip().splitlines()[-1])
                if record['peak_rss_bytes'] > 2*1024**3:
                    record['state'] = 'over_memory_limit'
                records.append(record)
        except subprocess.TimeoutExpired:
            records.append(dict(method=method,k=k,layer=layer,repeat=repeat,state='timeout',elapsed=time.perf_counter()-start))
        write_json(BASE/'timing.json', records)
        if (ti+1) % 30 == 0:
            print(f'timing {ti+1}/{len(tasks)}', flush=True)
    return records


def anytime_pilot(cov, candidates, loss_means, gates):
    records=[]
    methods=('xi_fast','pair_phi_r','weak_phi_r','phi_r_lu','wms','o_increment','whole_mi')
    for round_i in range(5):
        for site_i, pool in enumerate(candidates):
            order=np.random.default_rng(906+round_i*10+site_i).permutation(len(pool))
            for method in methods:
                q=MIOracle(cov); best=-np.inf; chosen=None; curve=[]
                start=time.perf_counter()
                for visited,ai in enumerate(order):
                    value=score(q,pool[ai],method)
                    if value>best or (value==best and (chosen is None or ai<chosen)):
                        best=value; chosen=int(ai)
                    elapsed=time.perf_counter()-start
                    nreg=((loss_means[site_i].max()-loss_means[site_i,chosen])/np.ptp(loss_means[site_i])) if gates[site_i]['distinguishable'] else None
                    curve.append(dict(seconds=elapsed,nreg=nreg,chosen=chosen,visited=visited+1))
                records.append(dict(round=round_i,site_index=site_i,method=method,order=order.tolist(),curve=curve))
    write_json(BASE/'anytime.json',records)
    return records


def nonlinear_check(x, y, candidates, sites, covariance):
    """Low-dimensional shared estimator sensitivity, never response-based tuning."""
    records = []
    pairs=[]; rng=np.random.default_rng(908)
    for site,pool in zip(sites,candidates):
        for a8 in pool:
            eligible=[int(i) for i in a8 if i!=site and tuple(sorted((site,int(i)))) not in pairs]
            if eligible:
                pairs.append(tuple(sorted((site,int(rng.choice(eligible))))))
    q=MIOracle(covariance)
    for a in pairs:
        dims = [*a, *(100+np.array(a))]
        xs = x[:, dims]; ys = y[:, dims]
        train = slice(0, 1536); test = slice(1536, None)
        from exp.TM.transport_map_density import fit_polynomial_triangular_transport_map_density
        results = {}
        for degree in (1, 2):
            cache = {}
            def mi(dx, dy):
                key = (tuple(dx), tuple(dy))
                if key not in cache:
                    xx, yy = xs[:, dx], ys[:, dy]
                    joint = np.c_[xx, yy]
                    models = [fit_polynomial_triangular_transport_map_density(z[train], degree=degree)
                              for z in (joint, xx, yy)]
                    value = (models[0].log_prob(joint[test])-models[1].log_prob(xx[test])-models[2].log_prob(yy[test])).mean()/np.log(2)
                    cache[key] = float(value)
                return cache[key]
            all_dims = (0, 1, 2, 3); u = (0, 2); v = (1, 3)
            whole = mi(all_dims, all_dims)
            xi = whole-mi(u, all_dims)-mi(v, all_dims)
            phi = whole-mi(u,u)-mi(v,v)+min(mi(u,u),mi(u,v),mi(v,u),mi(v,v))
            results[str(degree)] = dict(xi_raw_bits=xi, phi_r_bits=phi, whole_mi_bits=whole)
        results['main_affine']=dict(xi_raw_bits=score(q,a,'xi'),phi_r_bits=score(q,a,'phi_r_lu'),whole_mi_bits=score(q,a,'whole_mi'))
        records.append(dict(rois=list(map(int,a)), results=results))
    comparisons = {}
    for metric in ('xi_raw_bits','phi_r_bits','whole_mi_bits'):
        linear = [r['results']['main_affine'][metric] for r in records]
        nonlinear = [r['results']['2'][metric] for r in records]
        comparisons[metric] = dict(rank_spearman=float(spearmanr(linear, nonlinear).statistic),
                                   median_absolute_change_bits=float(np.median(np.abs(np.array(linear)-nonlinear))))
    raw=np.array([r['results']['2']['xi_raw_bits'] for r in records])
    try:
        check_syn(raw)
        nonnegative=dict(passed=True,minimum_bits=float(raw.min()),threshold_bits=-TOL,affected_count=0)
    except ArithmeticError as error:
        nonnegative=dict(passed=False,minimum_bits=float(raw.min()),threshold_bits=-TOL,
                         affected_count=int((raw < -TOL).sum()),failure=str(error))
    return dict(records=records, comparison=comparisons, degree=2, train_count=1536, test_count=512,
                nonnegative_audit=nonnegative,
                note='Held-out polynomial TM estimates are sensitivity diagnostics, not consumed as PEID Syn. '
                     'Separate density fits need not yield a coherent factorized joint. Any negative Xi is an estimator failure, not negative Syn.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--method')
    parser.add_argument('--k',type=int); parser.add_argument('--layer'); parser.add_argument('--repeat',type=int)
    parser.add_argument('--skip-timing',action='store_true')
    args = parser.parse_args()
    if args.worker:
        timing_worker(args); return
    BASE.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    audit = algorithm_audit(); write_json(BASE/'algorithm_audit.json', audit)
    with np.load(SOURCE) as data:
        sc = data['connectivity'].copy(); allg = data['G'].copy(); jfs = data['j_fic'].copy()
    index = np.flatnonzero(np.isclose(allg,1.))[0]; jf = jfs[index]
    if not np.array_equal(jfs, np.broadcast_to(jf, jfs.shape)):
        raise ValueError('JFIC must be fixed at G=1')
    dmf = load_dmf_module(); p = dmf.DMFParameters(dt=.001, sigma=.01)
    sites, candidates, origins, ranges = freeze_candidates(sc)
    budgets = [float(.5*min(weaken_return(sc,a,0.)[3] for a in pool)) for pool in candidates]
    contract = dict(stage='A-C exploratory pilot only', G=1.3, sites=sites, candidates=candidates.tolist(),
                    candidate_origins=origins, return_strength_filter=ranges, budgets=budgets,
                    k=8, sample_count=2048, response_repeats=2, scoring_seeds=[901,902],
                    evaluation_seed=1901, candidate_seed=901, dt=.001, sigma=.01, horizon_steps=300,
                    pulse_steps=10, cut_start_steps=10, pulse_amplitude=.01*p.i0,
                    early_ms=[10,50], late_ms=[50,300], state_boundary='none',
                    baseline_seconds=10., no_recalibration=True, tonic_compensation=True,
                    baseline_adjustment='Initial 2 s baseline failed 0.05 Hz drift gate (0.12540291075418786 Hz); '
                                        'extended to 10 s before scores or labels existed.',
                    ridge_standardized=1e-6, syn_tolerance_bits=TOL,
                    estimator='factorized-prior full-system affine triangular TM, then marginal queries',
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    sc_sha256=hashlib.sha256(sc.tobytes()).hexdigest(),
                    criterion='pool loss range > 2 x largest candidate loss SEM; exploratory gate, no confirmatory CI',
                    threads=1, precision='float64', dmf_parameters=vars(p),
                    hardware=dict(platform=platform.platform(),machine=platform.machine(),processor=platform.processor()),
                    versions=dict(numpy=np.__version__,scipy=scipy.__version__,python=sys.version),
                    implementation_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest()
                        for s in ('scripts/dmf_response_benchmark.py','scripts/run_dmf_response_benchmark.py',
                                  'exp/brain/dmf_fig6.py','exp/TM/transport_map_density.py')},
                    literature=dict(PEID='MYATYWAJ full text 26 pages',Momi='P7L7F9FT full text 12 pages',PhiID='26Q48H8Y main 12 pages and SI 10 pages'))
    # Write candidate membership and protocol BEFORE measuring any scores or labels.
    write_json(BASE/'contract.json',contract)
    se, si, baseline_rates, baseline_diag = relax(dmf, sc, jf,1.3,p,901,sigma=0.,seconds=10.)
    reference = np.r_[se[0],si[0]]
    drift = float(np.max(np.abs(baseline_rates[-100:].mean(axis=0)-baseline_rates[-200:-100].mean(axis=0))))
    if drift > .05:
        raise ArithmeticError(f'Deterministic baseline not steady: {drift} Hz')
    initial_e, initial_i, rest, initial_diag = relax(dmf,sc,jf,1.3,p,1901,count=2,initial=(se[0],si[0]))
    support_trials = []
    for width in (.005,.01,.02):
        rng = np.random.default_rng(904)
        x = reference + rng.uniform(-width,width,size=(512,200))
        y, diag = simulate_sources(dmf,x,sc,jf,1.3,p,905)
        initially_outside = int(((x<0)|(x>1)).sum())
        support_trials.append(dict(halfwidth=width,initial_outside_count=initially_outside,diagnostics=diag))
    valid = [r['halfwidth'] for r in support_trials if r['initial_outside_count']==0 and r['diagnostics']['outside_state_count']==0]
    if not valid:
        write_json(BASE/'support_trials.json',support_trials)
        raise ArithmeticError('No valid common intervention support')
    width = max(valid)  # Maximize perturbation signal subject to validity, never use response/method rankings.
    contract.update(halfwidth=width, support_selection='largest of .005/.01/.02 with no initial or integrated state violations',
                    support_trials=support_trials, baseline_drift_hz=drift)
    write_json(BASE/'contract.json',contract)
    response_records = []; gates = []
    for site_index, site in enumerate(sites):
        begin = time.perf_counter()
        result = responses(dmf,sc,jf,1.3,p,(initial_e,initial_i),site,candidates[site_index],budgets[site_index],.01*p.i0,1902)
        np.savez_compressed(BASE/f'response_site{site}.npz',**{k:v for k,v in result.items() if k!='diagnostics'})
        means = result['loss'].mean(axis=1); sem = result['loss'].std(axis=1,ddof=1)/np.sqrt(2)
        span = float(np.ptp(means)); detectable = bool(span>2*sem.max())
        drift_rate = float(np.sqrt(np.mean((result['rates'][1:,1]-result['rates'][0,1])**2)))
        peak_response = float(result['rms'][0].max())
        # Baseline diagnosis is reported, never used to select favorable candidates.
        gates.append(dict(site=site,loss_range_hz_s=span,max_loss_sem_hz_s=float(sem.max()),
                          distinguishable=detectable,intact_peak_rms_hz=peak_response,
                          sham_drift_rms_hz=drift_rate, pulse_detectable=peak_response>1e-6,
                          elapsed_seconds=time.perf_counter()-begin,diagnostics=result['diagnostics']))
        response_records.append(result)
        print(f'response site={site}: range={span:.5g} Hz*s; distinguishable={detectable}',flush=True)
    rest_fc = np.corrcoef(rest[-1000:].reshape(-1,100),rowvar=False)
    scores = np.zeros((2,2,8,len(LABELS)-1)); timing_data = []
    zero_count = 0; equality_errors = []
    for seed_index, seed in enumerate((901,902)):
        begin = time.perf_counter(); rng = np.random.default_rng(seed*1000+1)
        x = reference+rng.uniform(-width,width,size=(2048,200))
        y, diag = simulate_sources(dmf,x,sc,jf,1.3,p,seed*1000+2)
        sim_seconds = time.perf_counter()-begin; fit_start=time.perf_counter()
        cov = fit_affine_joint(x,y,width); fit_seconds=time.perf_counter()-fit_start
        np.savez_compressed(BASE/f'score_seed{seed}.npz',source=x,target=y,covariance=cov,halfwidth=width,candidates=candidates)
        q = MIOracle(cov)
        for site_index,pool in enumerate(candidates):
            for ai,a in enumerate(pool):
                for mi,method in enumerate(METHODS):
                    scores[seed_index,site_index,ai,mi] = score(q,a,method)
                equality_errors.append(abs(scores[seed_index,site_index,ai,0]-score(q,a,'xi_fast')))
                scores[seed_index,site_index,ai,7] = weaken_return(sc,a,0.)[3]
                scores[seed_index,site_index,ai,8] = sc[np.ix_(a,a)].sum()
                fc = rest_fc[np.ix_(a,a)]; scores[seed_index,site_index,ai,9] = fc[np.triu_indices(8,1)].mean()
                # At fixed site the pre-cut response is identical for every A: report ties explicitly.
                scores[seed_index,site_index,ai,10] = response_records[site_index]['rms'][0,:,0:10].mean()
        zero_count += check_syn(scores[seed_index,:,:,0])
        timing_data.append(dict(seed=seed,simulation_seconds=sim_seconds,fit_seconds=fit_seconds,diagnostics=diag))
        print(f'score seed={seed}: samples + affine fit {sim_seconds+fit_seconds:.2f}s',flush=True)
    nonlinear = nonlinear_check(x,y,candidates,sites,cov)
    write_json(BASE/'nonlinear_tm_check.json',nonlinear)
    loss = np.stack([r['loss'] for r in response_records])
    means = loss.mean(axis=-1); sem = loss.std(axis=-1,ddof=1)/np.sqrt(2)
    anytime_pilot(cov,candidates,means,gates)
    selection = np.argmax(scores,axis=2)
    nreg = np.full((2,2,len(LABELS)),np.nan); selected_loss=np.empty_like(nreg); rho=np.full_like(nreg,np.nan)
    top_loss=np.empty_like(nreg)
    for seed_i in range(2):
        for si in range(2):
            for mi in range(len(LABELS)):
                if mi == len(LABELS)-1:
                    chosen = means[si].mean(); top = chosen; correlation = 0.
                else:
                    chosen = means[si,selection[seed_i,si,mi]]
                    order=np.argsort(-scores[seed_i,si,:,mi],kind='stable')
                    top=means[si,order[:1]].mean()
                    correlation=float(spearmanr(scores[seed_i,si,:,mi],means[si]).statistic) if np.ptp(scores[seed_i,si,:,mi])>0 else np.nan
                selected_loss[seed_i,si,mi]=chosen; top_loss[seed_i,si,mi]=top; rho[seed_i,si,mi]=correlation
                if gates[si]['distinguishable']:
                    nreg[seed_i,si,mi]=(means[si].max()-chosen)/np.ptp(means[si])
    np.savez_compressed(BASE/'comparison.npz',scores=scores,loss=loss,loss_sem=sem,nreg=nreg,selected_loss=selected_loss,
                        rank_spearman=rho,top10_loss=top_loss,selection=selection,candidates=candidates,sites=sites,labels=np.array(LABELS))
    summary=dict(stage='A-C pilot complete; D not started',algorithm_audit=audit,response_gates=gates,
                 syn_raw_min_bits=float(scores[:,:,:,0].min()),syn_tolerance_bits=TOL,syn_tolerance_zero_count=zero_count,
                 xi_direct_fast_max_error_bits=float(max(equality_errors)), scoring_cost=timing_data,
                 baseline_diagnostics=baseline_diag,initial_diagnostics=initial_diag,
                 nonlinear_rank_comparison=nonlinear['comparison'],
                 nonlinear_nonnegative_audit=nonlinear['nonnegative_audit'],
                 formal_go=False,
                 formal_stop_reasons=['Nonlinear TM joint-density/finite-sample audit is not stable.',
                                      'Only one G, two scoring seeds and two response trials; formal power and repeated-label stability not assessed.',
                                      'No user approval for D in this session.'],
                 note='Only two scoring seeds, two sites and two response repetitions. No confirmatory CI or noninferiority claim. '
                      'Initial-response baseline is constant in each fixed-site pool; deterministic tie selects candidate 0. '
                      'Random baseline is exact expectation over eight candidates. Top 10% rounds up to one candidate.')
    write_json(BASE/'summary.json',summary)
    if not args.skip_timing:
        print('Starting isolated timing tasks (5 repeats, randomized order).',flush=True)
        timing_pilot()
    summary['elapsed_seconds']=time.perf_counter()-started
    write_json(BASE/'summary.json',summary)
    print(f'A-C complete in {summary["elapsed_seconds"]:.1f}s',flush=True)


if __name__=='__main__':
    main()
